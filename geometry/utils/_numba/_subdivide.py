"""
Optimized Numba implementations for Catmull-Clark subdivision.

This module provides parallel, JIT-compiled implementations of the
Catmull-Clark subdivision algorithm as described in:
https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface

Key optimizations:
- Fully parallel computation of face points, edge points, and vertex points
- Fused border handling directly in the vertex computation
- Pre-computed valence and connectivity counts
- Vectorized topology rebuilding

Algorithm Overview:
1. Face Points: Centroid of each face
2. Edge Points: For interior edges, average of edge midpoint and adjacent face centroids
                For border edges, just the edge midpoint
3. Vertex Points: P = (Q + 2*R + (n-3)*S) / n
   - Q = average of adjacent face centroids
   - R = average of adjacent edge midpoints
   - S = original vertex position
   - n = vertex valence (number of adjacent edges)
4. Topology: Each original face with k vertices becomes k quad faces
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                          Core computation kernels                           #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compute_face_points(points, f2v):
    """
    Compute face centroids (face points) in parallel.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions (V, 3), float64.
    f2v : np.ndarray
        Face-to-vertex connectivity (F, max_verts), int32.
        Padded with -1 for faces with fewer vertices.

    Returns
    -------
    np.ndarray
        Face centroids (F, 3), float64.
    """
    n_faces     = f2v.shape[0]
    n_dims      = points.shape[1]
    max_verts   = f2v.shape[1]

    face_points = np.zeros((n_faces, n_dims), dtype=points.dtype)

    for i in prange(n_faces):
        count = 0
        for d in range(n_dims):
            acc = 0.0
            for j in range(max_verts):
                vi = f2v[i, j]
                if vi >= 0:
                    acc += points[vi, d]
                    if d == 0:
                        count += 1
            face_points[i, d] = acc

        # Divide by count
        if count > 0:
            inv_count = 1.0 / count
            for d in range(n_dims):
                face_points[i, d] *= inv_count

    return face_points


@njit(parallel=True, fastmath=True, cache=True)
def _compute_edge_midpoints(points, e2v):
    """
    Compute edge midpoints in parallel.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions (V, 3), float64.
    e2v : np.ndarray
        Edge-to-vertex connectivity (E, 2), int32.

    Returns
    -------
    np.ndarray
        Edge midpoints (E, 3), float64.
    """
    n_edges        = e2v.shape[0]
    n_dims         = points.shape[1]
    edge_midpoints = np.empty((n_edges, n_dims), dtype=points.dtype)

    for i in prange(n_edges):
        v0 = e2v[i, 0]
        v1 = e2v[i, 1]
        for d in range(n_dims):
            edge_midpoints[i, d] = (points[v0, d] + points[v1, d]) * 0.5

    return edge_midpoints


@njit(parallel=True, fastmath=True, cache=True)
def _compute_edge_points(edge_midpoints, face_points, e2f):
    """
    Compute Catmull-Clark edge points in parallel.

    For interior edges: average of edge midpoint and adjacent face centroids
    For border edges: just the edge midpoint

    Parameters
    ----------
    edge_midpoints : np.ndarray
        Edge midpoint positions (E, 3), float64.
    face_points : np.ndarray
        Face centroid positions (F, 3), float64.
    e2f : np.ndarray
        Edge-to-face connectivity (E, 2), int32.
        Padded with -1 for border edges.

    Returns
    -------
    np.ndarray
        Catmull-Clark edge points (E, 3), float64.
    """
    n_edges     = e2f.shape[0]
    n_dims      = edge_midpoints.shape[1]
    edge_points = np.empty((n_edges, n_dims), dtype=edge_midpoints.dtype)

    for i in prange(n_edges):
        f0 = e2f[i, 0]
        f1 = e2f[i, 1]

        # Count valid faces
        face_count = 0
        if f0 >= 0:
            face_count += 1
        if f1 >= 0:
            face_count += 1

        if face_count == 2:
            # Interior edge: average of midpoint and face centroids
            # edge_point = (midpoint + avg_face_centroids) / 2
            # = (midpoint + (face0 + face1) / 2) / 2
            # = midpoint / 2 + face0 / 4 + face1 / 4
            for d in range(n_dims):
                edge_points[i, d] = (
                    edge_midpoints[i, d] * 0.5
                    + face_points[f0, d] * 0.25
                    + face_points[f1, d] * 0.25
                )
        elif face_count == 1:
            # Border edge: just the midpoint
            for d in range(n_dims):
                edge_points[i, d] = edge_midpoints[i, d]
        else:
            # Isolated edge (shouldn't happen in valid mesh)
            for d in range(n_dims):
                edge_points[i, d] = edge_midpoints[i, d]

    return edge_points


@njit(parallel=True, fastmath=True, cache=True)
def _compute_vertex_points(
    points,
    face_points,
    edge_midpoints,
    v2f,
    v2e,
    e2f,
):
    """
    Compute Catmull-Clark vertex points in parallel.

    Standard formula: P = (Q + 2*R + (n-3)*S) / n
    Where:
    - Q = average of face centroids touching vertex
    - R = average of edge midpoints touching vertex
    - S = original vertex position
    - n = vertex valence

    Border vertices get special treatment:
    - Corner vertices (valence < 3): keep original position
    - Border edge vertices: average of border edge midpoints, blended with S

    Parameters
    ----------
    points : np.ndarray
        Original vertex positions (V, 3), float64.
    face_points : np.ndarray
        Face centroid positions (F, 3), float64.
    edge_midpoints : np.ndarray
        Edge midpoint positions (E, 3), float64.
    v2f : np.ndarray
        Vertex-to-face connectivity (V, max_faces), int32.
    v2e : np.ndarray
        Vertex-to-edge connectivity (V, max_edges), int32.
    e2f : np.ndarray
        Edge-to-face connectivity (E, 2), int32.

    Returns
    -------
    np.ndarray
        New vertex positions (V, 3), float64.
    """
    n_verts    = points.shape[0]
    n_dims     = points.shape[1]
    max_faces  = v2f.shape[1]
    max_edges  = v2e.shape[1]

    new_points = np.empty_like(points)

    for i in prange(n_verts):
        # Count faces and edges
        face_valence = 0
        edge_valence = 0

        for j in range(max_faces):
            if v2f[i, j] >= 0:
                face_valence += 1

        for j in range(max_edges):
            if v2e[i, j] >= 0:
                edge_valence += 1

        # Detect border vertex
        is_border = edge_valence != face_valence
        is_corner = edge_valence < 3

        if is_corner:
            # Corner vertex: keep original position
            for d in range(n_dims):
                new_points[i, d] = points[i, d]

        elif is_border:
            # Border vertex: average of border edge midpoints, blended with S
            # Find border edges (edges with only one face)
            border_count = 0
            border_sum   = np.zeros(n_dims, dtype=points.dtype)

            for j in range(max_edges):
                ei = v2e[i, j]
                if ei >= 0:
                    # Check if this is a border edge
                    f0 = e2f[ei, 0]
                    f1 = e2f[ei, 1]
                    if (f0 < 0) or (f1 < 0):
                        # This is a border edge
                        border_count += 1
                        for d in range(n_dims):
                            border_sum[d] += edge_midpoints[ei, d]

            if border_count > 0:
                inv_border = 1.0 / border_count
                for d in range(n_dims):
                    avg_border       = border_sum[d] * inv_border
                    new_points[i, d] = (points[i, d] + avg_border) * 0.5
            else:
                for d in range(n_dims):
                    new_points[i, d] = points[i, d]

        else:
            # Interior vertex: standard Catmull-Clark formula
            # P = (Q + 2*R + (n-3)*S) / n
            n     = float(edge_valence)
            inv_n = 1.0 / n

            for d in range(n_dims):
                # Q: average of face centroids
                q_sum = 0.0
                for j in range(max_faces):
                    fi = v2f[i, j]
                    if fi >= 0:
                        q_sum += face_points[fi, d]
                Q = q_sum * inv_n

                # R: average of edge midpoints
                r_sum = 0.0
                for j in range(max_edges):
                    ei = v2e[i, j]
                    if ei >= 0:
                        r_sum += edge_midpoints[ei, d]
                R = r_sum * inv_n

                # S: original position
                S = points[i, d]

                # Catmull-Clark formula
                new_points[i, d] = (Q + 2.0 * R + (n - 3.0) * S) * inv_n

    return new_points


@njit(parallel=True, fastmath=True, cache=True)
def _compute_vertex_points_keep_borders(
    points,
    face_points,
    edge_midpoints,
    v2f,
    v2e,
    e2f,
    border_verts,
):
    """
    Compute vertex points with border vertices kept at original positions.

    Parameters
    ----------
    points : np.ndarray
        Original vertex positions (V, 3), float64.
    face_points : np.ndarray
        Face centroid positions (F, 3), float64.
    edge_midpoints : np.ndarray
        Edge midpoint positions (E, 3), float64.
    v2f : np.ndarray
        Vertex-to-face connectivity (V, max_faces), int32.
    v2e : np.ndarray
        Vertex-to-edge connectivity (V, max_edges), int32.
    e2f : np.ndarray
        Edge-to-face connectivity (E, 2), int32.
    border_verts : np.ndarray
        Boolean mask of border vertices (V,), bool.

    Returns
    -------
    np.ndarray
        New vertex positions (V, 3), float64.
    """
    n_verts    = points.shape[0]
    n_dims     = points.shape[1]
    max_faces  = v2f.shape[1]
    max_edges  = v2e.shape[1]

    new_points = np.empty_like(points)

    for i in prange(n_verts):
        if border_verts[i]:
            # Keep border vertex at original position
            for d in range(n_dims):
                new_points[i, d] = points[i, d]
        else:
            # Interior vertex: standard Catmull-Clark formula
            edge_valence = 0
            for j in range(max_edges):
                if v2e[i, j] >= 0:
                    edge_valence += 1

            n     = float(edge_valence)
            inv_n = 1.0 / n if edge_valence > 0 else 0.0

            for d in range(n_dims):
                # Q: average of face centroids
                q_sum      = 0.0
                face_count = 0
                for j in range(max_faces):
                    fi = v2f[i, j]
                    if fi >= 0:
                        q_sum      += face_points[fi, d]
                        face_count += 1
                Q = q_sum * inv_n if face_count > 0 else 0.0

                # R: average of edge midpoints
                r_sum = 0.0
                for j in range(max_edges):
                    ei = v2e[i, j]
                    if ei >= 0:
                        r_sum += edge_midpoints[ei, d]
                R = r_sum * inv_n

                # S: original position
                S = points[i, d]

                # Catmull-Clark formula
                if edge_valence > 0:
                    new_points[i, d] = (Q + 2.0 * R + (n - 3.0) * S) * inv_n
                else:
                    new_points[i, d] = S

    return new_points


# --------------------------------------------------------------------------- #
#                          Topology rebuilding                                #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _rebuild_subdivision_topology(counts, f2v, f2e, e2v, edge_offset, face_offset):
    """
    Rebuild topology for Catmull-Clark subdivision in parallel.

    Each original face with k vertices becomes k quad faces.
    Each new quad connects: face_point, edge_point0, vertex, edge_point1

    Parameters
    ----------
    counts : np.ndarray
        Number of vertices per face (F,), int32.
    f2v : np.ndarray
        Face-to-vertex connectivity (F, max_verts), int32.
    f2e : np.ndarray
        Face-to-edge connectivity (F, max_edges), int32.
    e2v : np.ndarray
        Edge-to-vertex connectivity (E, 2), int32.
    edge_offset : int
        Offset to add to edge indices (= original vertex count).
    face_offset : int
        Offset to add to face indices (= original vertex count + edge count).

    Returns
    -------
    tuple
        (new_indices, new_counts) for the subdivided mesh.
    """
    n_faces         = counts.shape[0]
    total_new_faces = np.sum(counts)
    max_verts       = f2v.shape[1]

    # Each new face is a quad (4 vertices)
    new_indices = np.empty(total_new_faces * 4, dtype=np.int32)
    new_counts  = np.full(total_new_faces, 4, dtype=np.int32)

    # Pre-compute face starts for parallel processing
    face_starts = np.zeros(n_faces + 1, dtype=np.int32)
    for i in range(n_faces):
        face_starts[i + 1] = face_starts[i] + counts[i]

    # Process faces in parallel
    for i in prange(n_faces):
        count_i       = counts[i]
        base_face_idx = face_starts[i]
        f             = i + face_offset

        for j in range(count_i):
            v0 = f2v[i, j]
            e0 = f2e[i, j]

            # Find the other vertex on this edge
            v1 = e2v[e0, 0]
            if v0 == v1:
                v1 = e2v[e0, 1]

            # Next edge
            k  = (j + 1) % count_i
            e1 = f2e[i, k]

            # Apply offsets
            e0_idx = e0 + edge_offset
            e1_idx = e1 + edge_offset

            # Write quad: face_point, edge0, vertex, edge1
            out_idx                  = (base_face_idx + j) * 4
            new_indices[out_idx]     = f
            new_indices[out_idx + 1] = e0_idx
            new_indices[out_idx + 2] = v1
            new_indices[out_idx + 3] = e1_idx

    return new_indices, new_counts