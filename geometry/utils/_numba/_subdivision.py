"""
Optimized Numba implementations for mesh subdivision operations.

These functions provide parallel, JIT-compiled implementations for
Catmull-Clark subdivision index rebuilding.

Key optimizations:
- Parallel face processing with pre-computed cumulative indices
- Eliminated sequential index dependency
- Direct output array creation (no unnecessary copies)
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                     Original implementation                                 #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, cache=True)
def _rebuild_indices(
    new_indices, counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset
):
    """
    Rebuild face indices for Catmull-Clark subdivision.

    Original sequential implementation kept for compatibility.
    For parallel version, use _rebuild_indices_parallel.

    For each original face, creates new quad faces connecting:
    - Face centroid (f)
    - Edge midpoints (e0, e1)
    - Original vertex (v1)

    Parameters
    ----------
    new_indices : np.ndarray
        Template array for new indices.
    counts : np.ndarray
        Number of edges per face (N,), int32.
    cumsum : np.ndarray
        Cumulative sum for indexing into result.
    f2v : np.ndarray
        Face-to-vertex mapping (N, max_verts), int32.
    f2e : np.ndarray
        Face-to-edge mapping (N, max_edges), int32.
    e2v : np.ndarray
        Edge-to-vertex mapping (E, 2), int32.
    face_offset : int
        Offset for face indices in new topology.
    edge_offset : int
        Offset for edge indices in new topology.

    Returns
    -------
    np.ndarray
        Rebuilt indices array.
    """
    result = np.copy(new_indices)
    index  = 0

    for i in range(counts.size):
        for j in range(counts[i]):
            v0 = f2v[i, j]
            e0 = f2e[i, j]
            v1 = e2v[e0, 0]

            if v0 == v1:
                v1 = e2v[e0, 1]

            k  = (j + 1) % (counts[i])
            e1 = f2e[i, k]

            f  = i + face_offset
            e0 += edge_offset
            e1 += edge_offset

            ii = cumsum[index]
            result[ii] = f
            result[ii + 1] = e0
            result[ii + 2] = v1
            result[ii + 3] = e1

            index += 1

    return result


# --------------------------------------------------------------------------- #
#                     Parallelized implementation                             #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _rebuild_indices_parallel(
    new_indices, counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset
):
    """
    Parallelized version of _rebuild_indices.

    Processes faces in parallel by pre-computing per-face base indices,
    eliminating the sequential index dependency.

    Parameters
    ----------
    new_indices : np.ndarray
        Template array for new indices.
    counts : np.ndarray
        Number of edges per face (N,), int32.
    cumsum : np.ndarray
        Cumulative sum for indexing into result.
    f2v : np.ndarray
        Face-to-vertex mapping (N, max_verts), int32.
    f2e : np.ndarray
        Face-to-edge mapping (N, max_edges), int32.
    e2v : np.ndarray
        Edge-to-vertex mapping (E, 2), int32.
    face_offset : int
        Offset for face indices in new topology.
    edge_offset : int
        Offset for edge indices in new topology.

    Returns
    -------
    np.ndarray
        Rebuilt indices array.
    """
    n_faces = counts.size
    result  = np.empty_like(new_indices)

    # Copy template data (parallel)
    for i in prange(new_indices.size):
        result[i] = new_indices[i]

    # Pre-compute per-face base index (cumulative sum of counts)
    # This allows each face to compute its global index independently
    face_base_idx = np.zeros(n_faces + 1, dtype=np.int32)
    for i in range(n_faces):
        face_base_idx[i + 1] = face_base_idx[i] + counts[i]

    # Process faces in parallel
    for i in prange(n_faces):
        count_i  = counts[i]
        base_idx = face_base_idx[i]
        f        = i + face_offset

        for j in range(count_i):
            v0 = f2v[i, j]
            e0 = f2e[i, j]
            v1 = e2v[e0, 0]

            if v0 == v1:
                v1 = e2v[e0, 1]

            k  = (j + 1) % count_i
            e1 = f2e[i, k]

            # Compute global index from face base + local edge index
            index = base_idx + j
            ii    = cumsum[index]

            result[ii] = f
            result[ii + 1] = e0 + edge_offset
            result[ii + 2] = v1
            result[ii + 3] = e1 + edge_offset

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _rebuild_indices_optimized(
    counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset, output_size
):
    """
    Fully optimized version that creates output directly.

    Removes unnecessary input copy by creating output array directly.
    Use this when you don't need to preserve any data from a template.

    Parameters
    ----------
    counts : np.ndarray
        Number of edges per face (N,), int32.
    cumsum : np.ndarray
        Cumulative sum for indexing into result.
    f2v : np.ndarray
        Face-to-vertex mapping (N, max_verts), int32.
    f2e : np.ndarray
        Face-to-edge mapping (N, max_edges), int32.
    e2v : np.ndarray
        Edge-to-vertex mapping (E, 2), int32.
    face_offset : int
        Offset for face indices in new topology.
    edge_offset : int
        Offset for edge indices in new topology.
    output_size : int
        Size of output array.

    Returns
    -------
    np.ndarray
        Rebuilt indices array.
    """
    n_faces = counts.size
    result  = np.empty(output_size, dtype=np.int32)

    # Pre-compute per-face cumulative count
    face_cumsum = np.zeros(n_faces + 1, dtype=np.int32)
    for i in range(n_faces):
        face_cumsum[i + 1] = face_cumsum[i] + counts[i]

    # Process faces in parallel
    for i in prange(n_faces):
        count_i  = counts[i]
        base_idx = face_cumsum[i]
        f        = i + face_offset

        for j in range(count_i):
            v0 = f2v[i, j]
            e0 = f2e[i, j]
            v1 = e2v[e0, 0]

            if v0 == v1:
                v1 = e2v[e0, 1]

            k     = (j + 1) % count_i
            e1    = f2e[i, k]

            index = base_idx + j
            ii    = cumsum[index]

            result[ii] = f
            result[ii + 1] = e0 + edge_offset
            result[ii + 2] = v1
            result[ii + 3] = e1 + edge_offset

    return result


# --------------------------------------------------------------------------- #
#                     Additional subdivision helpers                          #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compute_face_centroids(points, face_indices, counts, face_starts):
    """
    Compute face centroids in parallel.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions (V, 3), float64.
    face_indices : np.ndarray
        Flat array of vertex indices per face.
    counts : np.ndarray
        Number of vertices per face (F,), int32.
    face_starts : np.ndarray
        Start index in face_indices for each face (F,), int32.

    Returns
    -------
    np.ndarray
        Face centroids (F, 3), float64.
    """
    n_faces   = counts.size
    n_dims    = points.shape[1]
    centroids = np.zeros((n_faces, n_dims), dtype=points.dtype)

    for i in prange(n_faces):
        count     = counts[i]
        start     = face_starts[i]
        inv_count = 1.0 / count

        for d in range(n_dims):
            acc = 0.0
            for j in range(count):
                vi = face_indices[start + j]
                acc += points[vi, d]
            centroids[i, d] = acc * inv_count

    return centroids


@njit(parallel=True, fastmath=True, cache=True)
def _compute_edge_midpoints(points, edge_vertices):
    """
    Compute edge midpoints in parallel.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions (V, 3), float64.
    edge_vertices : np.ndarray
        Edge-to-vertex mapping (E, 2), int32.

    Returns
    -------
    np.ndarray
        Edge midpoints (E, 3), float64.
    """
    n_edges   = edge_vertices.shape[0]
    n_dims    = points.shape[1]
    midpoints = np.empty((n_edges, n_dims), dtype=points.dtype)

    for i in prange(n_edges):
        v0 = edge_vertices[i, 0]
        v1 = edge_vertices[i, 1]

        for d in range(n_dims):
            midpoints[i, d] = (points[v0, d] + points[v1, d]) * 0.5

    return midpoints


@njit(parallel=True, fastmath=True, cache=True)
def _compute_vertex_averages(
    points,
    vertex_face_counts,
    vertex_edge_counts,
    face_centroids,
    edge_midpoints,
    vertex_faces,
    vertex_edges,
    vertex_face_starts,
    vertex_edge_starts,
):
    """
    Compute new vertex positions using Catmull-Clark averaging.

    For each vertex:
    new_pos = (F + 2*R + (n-3)*P) / n

    Where:
    - F = average of face centroids touching vertex
    - R = average of edge midpoints touching vertex
    - P = original vertex position
    - n = number of faces/edges touching vertex

    Parameters
    ----------
    points : np.ndarray
        Original vertex positions (V, 3), float64.
    vertex_face_counts : np.ndarray
        Number of faces per vertex (V,), int32.
    vertex_edge_counts : np.ndarray
        Number of edges per vertex (V,), int32.
    face_centroids : np.ndarray
        Face centroid positions (F, 3), float64.
    edge_midpoints : np.ndarray
        Edge midpoint positions (E, 3), float64.
    vertex_faces : np.ndarray
        Flat array of face indices per vertex.
    vertex_edges : np.ndarray
        Flat array of edge indices per vertex.
    vertex_face_starts : np.ndarray
        Start index in vertex_faces for each vertex (V,), int32.
    vertex_edge_starts : np.ndarray
        Start index in vertex_edges for each vertex (V,), int32.

    Returns
    -------
    np.ndarray
        New vertex positions (V, 3), float64.
    """
    n_verts    = points.shape[0]
    n_dims     = points.shape[1]
    new_points = np.empty_like(points)

    for i in prange(n_verts):
        n_faces = vertex_face_counts[i]
        n_edges = vertex_edge_counts[i]

        if n_faces == 0 or n_edges == 0:
            # Isolated vertex - keep original position
            for d in range(n_dims):
                new_points[i, d] = points[i, d]
            continue

        face_start = vertex_face_starts[i]
        edge_start = vertex_edge_starts[i]

        n          = float(n_faces)  # valence
        inv_n      = 1.0 / n

        for d in range(n_dims):
            # Average of face centroids (F)
            f_avg = 0.0
            for j in range(n_faces):
                fi = vertex_faces[face_start + j]
                f_avg += face_centroids[fi, d]
            f_avg *= inv_n

            # Average of edge midpoints (R)
            r_avg = 0.0
            for j in range(n_edges):
                ei = vertex_edges[edge_start + j]
                r_avg += edge_midpoints[ei, d]
            r_avg /= float(n_edges)

            # Original position (P)
            p = points[i, d]

            # Catmull-Clark formula: (F + 2*R + (n-3)*P) / n
            new_points[i, d] = (f_avg + 2.0 * r_avg + (n - 3.0) * p) * inv_n

    return new_points


@njit(parallel=True, fastmath=True, cache=True)
def _compute_new_edge_points(
    edge_midpoints,
    edge_faces,
    face_centroids,
    edge_vertices,
    points,
    is_boundary,
):
    """
    Compute new edge point positions for Catmull-Clark subdivision.

    For interior edges:
    new_pos = (midpoint + avg_face_centroids) / 2

    For boundary edges:
    new_pos = midpoint

    Parameters
    ----------
    edge_midpoints : np.ndarray
        Edge midpoint positions (E, 3), float64.
    edge_faces : np.ndarray
        Faces adjacent to each edge (E, 2), int32. -1 for boundary.
    face_centroids : np.ndarray
        Face centroid positions (F, 3), float64.
    edge_vertices : np.ndarray
        Edge-to-vertex mapping (E, 2), int32.
    points : np.ndarray
        Original vertex positions (V, 3), float64.
    is_boundary : np.ndarray
        Boolean mask for boundary edges (E,), bool.

    Returns
    -------
    np.ndarray
        New edge point positions (E, 3), float64.
    """
    n_edges         = edge_midpoints.shape[0]
    n_dims          = edge_midpoints.shape[1]
    new_edge_points = np.empty_like(edge_midpoints)

    for i in prange(n_edges):
        if is_boundary[i]:
            # Boundary edge - just use midpoint
            for d in range(n_dims):
                new_edge_points[i, d] = edge_midpoints[i, d]
        else:
            # Interior edge - average with face centroids
            f0 = edge_faces[i, 0]
            f1 = edge_faces[i, 1]

            for d in range(n_dims):
                avg_centroid = (face_centroids[f0, d] + face_centroids[f1, d]) * 0.5
                new_edge_points[i, d] = (edge_midpoints[i, d] + avg_centroid) * 0.5

    return new_edge_points