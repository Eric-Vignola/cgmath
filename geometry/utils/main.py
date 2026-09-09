"""
Optimized version of main.py with Numba-accelerated implementations.

This module provides the same API as main.py but uses parallel Numba kernels
for improved performance on large datasets.
"""

from __future__ import annotations

import numbers

import numpy as np


# --------------------- PXR and Numba module management ---------------------- #


# --- USD --- #
try:
    import pxr as _pxr
except ImportError:
    _pxr = None


def pxr():
    """Returns the pxr module (USD) or raises an ImportError"""
    if _pxr is None:
        raise ImportError("USD not found.")
    return _pxr


# -------------------------------- Utilities --------------------------------- #


def matrix_index_lookup(matrix, indices):
    # get the first occurrence of each index in matrix
    from cgmath.geometry.utils._numba._topology import _matrix_index_lookup

    matrix  = np.asarray(matrix, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    return _matrix_index_lookup(matrix, indices)


def matrix_row_overlaps(matrix, exact=True, sort_indices=False):
    """Optimized: uses Numba hash-based row comparison."""
    if matrix is None or not matrix.shape[0]:
        return matrix

    from cgmath.geometry.utils._numba._main import matrix_row_overlaps_fast

    return matrix_row_overlaps_fast(matrix, exact=exact, sort_indices=sort_indices)


def point_row_overlaps(matrix, tolerance=1e-6):
    """
    Optimized: uses Numba for parallel pair processing.

    Returns relevant mapping data where matrix rows are identical.
    """
    if matrix is None or not matrix.shape[0]:
        return matrix

    from cgmath.geometry.utils._numba._main import point_row_overlaps_fast

    return point_row_overlaps_fast(matrix, tolerance=tolerance)


def matrix_to_stream(matrix):
    """converts matrix to index stream, skips over -1 pads"""
    matrix  = np.asarray(matrix)
    counts  = np.sum(matrix >= 0, axis=1)
    indices = matrix.ravel()
    indices = indices[indices >= 0]

    return indices, counts


def stream_to_matrix(values, counts):
    """Optimized: uses parallel Numba kernel for scatter operation."""
    from cgmath.geometry.utils._numba._main import stream_to_matrix_fast

    return stream_to_matrix_fast(values, counts)


def indices_replace(matrix, from_indices, to_indices, collapse=False):
    """Optimized: uses parallel Numba kernel with binary search."""
    from cgmath.geometry.utils._numba._main import indices_replace_fast

    return indices_replace_fast(matrix, from_indices, to_indices, collapse=collapse)


def average_points(points, geometry):
    """Optimized: uses parallel Numba kernel for averaging."""
    from cgmath.geometry.utils._numba._main import average_points_fast

    return average_points_fast(points, geometry)


def matrix_row_combine(matrix, from_rows, to_rows):
    """
    creates a new resized matrix where elements of `from_rows`
    are uniquely appended to destinaton elements `to_rows`
    """
    from cgmath.geometry.utils._numba._topology import _matrix_row_combine

    matrix    = np.asarray(matrix,    dtype=np.int32)
    from_rows = np.asarray(from_rows, dtype=np.int32)
    to_rows   = np.asarray(to_rows,   dtype=np.int32)
    return _matrix_row_combine(matrix, from_rows, to_rows)


# --------------------------------- MeshData --------------------------------- #


def compute_neighbors(i2x, x2i, n: int = 0):
    """builds vertex neighbors given v2x and x2v matrices"""
    from cgmath.geometry.utils._numba._topology import _compute_neighbors

    i2x       = np.asarray(i2x, dtype=np.int32)
    x2i       = np.asarray(x2i, dtype=np.int32)
    neighbors = _compute_neighbors(i2x, x2i)

    # optionally grow the neighbors n times
    if n > 0:
        neighbors = grow_neighbors(neighbors, n=n)

    return neighbors


def compute_neighbor_distances(
    neighbors: np.ndarray,
    points:    np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distances for each edge in the neighbor matrix.

    Args:
        neighbors: Connectivity matrix (N, W) with -1 as sentinel.
        points: Point positions (N, D).

    Returns:
        Distance matrix (N, W) with -1.0 as sentinel for missing neighbors.
    """
    from cgmath.geometry.utils._numba._connectivity import _compute_neighbor_distances

    neighbors = np.asarray(neighbors, dtype=np.int32)
    points    = np.asarray(points, dtype=np.float64)
    return _compute_neighbor_distances(neighbors, points)


def minimize_neighbor_distances(
    neighborhood:       np.ndarray,
    neighborhood_dists: np.ndarray,
    num_vertices:       int,
) -> np.ndarray:
    """Find the shortest distance to each unique vertex index across all rows.

    Uses a parallel min-reduction pattern where each thread writes the minimum
    distance for vertex indices it encounters.

    Parameters
    ----------
    neighborhood : np.ndarray
        Neighborhood indices of shape (num_indices, max_neighbors).
        Padded with -1 for invalid entries.
    neighborhood_dists : np.ndarray
        Corresponding distances of shape (num_indices, max_neighbors).
        Padded with -1.0 for invalid entries.
    num_vertices : int
        Total number of vertices (determines output array size).

    Returns
    -------
    np.ndarray
        Minimum distance per vertex index, shape (num_vertices,) with dtype float32.
        Vertices not referenced in neighborhood will have value inf.
    """
    from cgmath.geometry.utils._numba._connectivity import _minimize_neighbor_distances

    neighborhood       = np.asarray(neighborhood, dtype=np.int32)
    neighborhood_dists = np.asarray(neighborhood_dists, dtype=np.float32)
    return _minimize_neighbor_distances(neighborhood, neighborhood_dists, num_vertices)


def compute_topological_neighborhood(
    connectivity,
    num_hops        = None,
    indices         = None,
    distances       = None,
    max_distance    = None,
    distance_format = "sparse",
):
    """
    Compute the topological neighborhood of vertices within num_hops. (Dijkstra's algorithm)

    Uses parallel processing with Numba for high performance.

    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity matrix of shape (n, m) with dtype np.int32.
        Padded with -1 for vertices with fewer than m neighbors.
    num_hops : int, optional
        Number of hops to grow the neighborhood.
        If None, explores all reachable vertices (equivalent to num_vertices - 1).
    indices : np.ndarray, optional
        Vertex indices to compute neighborhoods for.
        If None, computes for all vertices.
    distances : np.ndarray, optional
        Distance matrix of shape (n, m) with dtype np.float32.
        Same shape as connectivity, with -1 for padding.
        If provided, returns shortest path distances to each neighbor.
    max_distance : float, optional
        Maximum cumulative distance threshold. Neighbors beyond this distance
        are excluded from results entirely.
        Only used when distances is provided. Default is None (no limit).
    distance_format : str, optional
        Output format for distances when distances is provided. Options:
        - "shortest": distances as a 1-D array of shape (num_vertices,)
          containing the shortest distance to each vertex across all rows,
          computed via minimize_neighbor_distances().
        - "sparse": (default) distances as fixed-width arrays padded with -1.0.
        - "dense": distances as a dense matrix of shape
          (len(indices), num_vertices) with 0's for self-distance and inf
          for unreachable vertices.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        If distances is None: neighborhood indices array
            Shape (len(indices), max_neighbors) or (n, max_neighbors)
        If distances provided: (neighborhood, distances) tuple
            Neighborhood is always the sparse indices array.
            Distances format depends on dist_format.

    Notes
    -----
    The max_distance parameter uses cumulative shortest path distance from
    the source vertices. This matches Dijkstra's algorithm semantics.
    """

    from cgmath.geometry.utils._numba._connectivity import (
        _compute_topological_neighborhood,
        _compute_topological_neighborhood_with_distances,
        _convert_to_dense_format,
    )

    connectivity = np.asarray(connectivity, dtype=np.int32)
    num_vertices = connectivity.shape[0]

    if num_hops is None or num_hops == -1:
        num_hops = num_vertices - 1
    if indices is None:
        indices = np.arange(num_vertices, dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)

    if distance_format not in ("sparse", "dense", "shortest"):
        raise ValueError(
            f"distance_format must be 'sparse', 'dense', or 'shortest', got '{distance_format}'"
        )

    if distance_format in ("dense", "shortest") and distances is None:
        raise ValueError(
            f"distance_format='{distance_format}' requires distances to be provided"
        )

    if distances is None:
        return _compute_topological_neighborhood(connectivity, num_hops, indices)
    else:
        distances = np.asarray(distances, dtype=np.float32)
        if max_distance is None:
            max_distance = np.float32(np.inf)
        else:
            max_distance = np.float32(max_distance)

        neighborhood, neighborhood_dists = (
            _compute_topological_neighborhood_with_distances(
                connectivity, distances, num_hops, indices, max_distance
            )
        )

        if distance_format == "dense":
            dense_dists = _convert_to_dense_format(
                indices, neighborhood, neighborhood_dists, len(indices), num_vertices
            )
            return neighborhood, dense_dists

        if distance_format == "shortest":
            minimized_dists = minimize_neighbor_distances(
                neighborhood, neighborhood_dists, num_vertices
            )
            idx = np.unique(neighborhood)
            idx = idx[idx > -1]

            return idx, minimized_dists[idx]

        return neighborhood, neighborhood_dists


def grow_neighbors(
    neighbors: np.ndarray,
    indices:   np.ndarray | None = None,
    n:         int               = 1,
    full_view: bool              = True,
) -> np.ndarray:
    """grows a neighborhood matrix by n iterations

    Args:
        neighbors: Connectivity matrix with -1 as sentinel
        indices: Optional array of vertex indices to process. If None, processes all vertices.
                 Use np.intp dtype for best compatibility.
        n: Number of grow iterations (default 1)
        full_view: If True, returns the full view of the grown neighbors. If False, returns only the indices specified in `indices`.

    Returns:
        Grown connectivity for the specified indices (or all vertices if indices=None)
    """

    from cgmath.geometry.utils._numba._connectivity import (
        _compute_topological_neighborhood,
    )
    from cgmath.geometry.utils._numba._topology import _grow_neighbors

    if n <= 0:
        if indices is None:
            return neighbors.copy()
        else:
            return neighbors[indices].copy()

    # If no indices given, do the whole array
    if indices is None:
        idx = np.arange(neighbors.shape[0], dtype=np.int32)
    else:
        idx = np.asarray(indices, dtype=np.int32)

    dtype = neighbors.dtype
    if neighbors.dtype != np.int32:
        neighbors = neighbors.astype(np.int32)

    # compute the grown neighbors
    if n <= 5:
        matrix = _grow_neighbors(neighbors, idx, n)

    # _compute_topological_neighborhood is a lot faster for n > 5
    else:
        matrix = _compute_topological_neighborhood(neighbors, n, idx)

    # if indices is None, full_view is returned by default
    if indices is None:
        return matrix.astype(dtype)

    # return full view
    if full_view:
        expanded = np.full((neighbors.shape[0], matrix.shape[1]), -1, dtype=np.int32)
        expanded[:, : neighbors.shape[1]] = neighbors
        expanded[idx] = matrix
        return expanded.astype(dtype)

    # return only the indices specified in `indices`
    return matrix.astype(dtype)


def compute_e2v_e2f(f2v, counts):
    """builds edges_to_vertices and edges_to_faces components"""
    from cgmath.geometry.utils._numba._topology import _compute_e2v_e2f

    f2v    = np.asarray(f2v, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32)
    return _compute_e2v_e2f(f2v, counts)


def compute_v2x(f2v, shape):
    """general purpose vertex_to_x builder"""
    from cgmath.geometry.utils._numba._topology import _compute_v2x

    f2v   = np.asarray(f2v, dtype=np.int32)
    shape = np.asarray(shape, dtype=np.int32)
    return _compute_v2x(f2v, shape)


def shared_edges_test(edges):
    """a numba powered test to see if edges are repeated"""
    from cgmath.geometry.utils._numba._topology import _shared_edges_test

    edges = np.asarray(edges, dtype=np.int32)
    edges = edges.reshape(edges.shape[0], -1)
    edges = np.sort(edges, axis=1)
    return np.where(_shared_edges_test(edges))


# ----------------------------- Normals & Tangents --------------------------- #


def split_at_hard_edges(indices, counts, e2v, e2f, hard_edges):
    """Build face-varying normal slot indices via union-find at smooth edges.

    Two face-vertex positions referencing the same mesh vertex are
    merged into the same normal slot when they share a smooth edge.

    Args:
        indices: ``(n_fv,)`` flat face-vertex stream.
        counts:  ``(n_faces,)`` vertex count per face.
        e2v:     ``(n_edges, 2)`` edge -> vertex pair.
        e2f:     ``(n_edges, k)`` edge -> faces (k >= 1; -1 padded).
        hard_edges: ``(n_hard,)`` indices of hard edges (treated as
            discontinuities).

    Returns:
        Tuple ``(normal_indices, num_normals)`` where ``normal_indices``
        is ``(n_fv,)`` with the same dtype as ``indices`` and
        ``num_normals`` is the count of unique normal slots.
    """
    from cgmath.geometry.utils._numba._normals import _split_at_hard_edges

    indices = np.asarray(indices)
    counts  = np.asarray(counts, dtype=np.int32)
    e2v     = np.asarray(e2v,    dtype=np.int32)
    e2f     = np.asarray(e2f,    dtype=np.int32)

    if indices.size == 0:
        return np.zeros(0, dtype=indices.dtype), 0

    # Promote to int32 for the kernel; preserve original dtype on output.
    out_dtype   = indices.dtype
    indices_i32 = indices.astype(np.int32, copy=False)

    hard_mask   = np.zeros(e2f.shape[0], dtype=np.bool_)
    if hard_edges is not None:
        hard_edges = np.asarray(hard_edges, dtype=np.int64)
        if hard_edges.size:
            hard_mask[hard_edges] = True

    roots = _split_at_hard_edges(indices_i32, counts, e2v, e2f, hard_mask)
    _, normal_indices = np.unique(roots, return_inverse=True)
    return normal_indices.astype(out_dtype), int(normal_indices.max() + 1)


def build_uv_to_normal_map(uv_indices, normal_indices, num_uv_verts):
    """Map each UV vertex to a normal slot via face-vertex stream alignment.

    For each UV vertex, picks the normal slot from the FIRST face-vertex
    position that references it (matches the prior loop's "first-write-wins"
    semantics).

    Args:
        uv_indices:     ``(n_fv,)`` UV face-vertex stream.
        normal_indices: ``(n_fv,)`` parallel normal-slot stream.
        num_uv_verts:   total number of UV vertices.

    Returns:
        ``(num_uv_verts,)`` int array -- normal slot per UV vertex.
        Vertices not referenced in ``uv_indices`` get value 0.
    """
    out            = np.zeros(int(num_uv_verts), dtype=int)
    uv_indices     = np.asarray(uv_indices)
    normal_indices = np.asarray(normal_indices)
    if uv_indices.size == 0:
        return out
    # ``return_index`` gives the FIRST occurrence of each unique uv id.
    _, first_pos = np.unique(uv_indices, return_index=True)
    out[uv_indices[first_pos]] = normal_indices[first_pos]
    return out


def build_tri_expand_map(counts, rules):
    """Map original face-vertex stream positions to triangulated positions.

    For tris (count == 3) emits ``[pos, pos+1, pos+2]``.  For quads
    (count == 4) emits two triangles per the per-face split rule.

    Args:
        counts: ``(n_faces,)`` vertex count per face. Must contain only
            3 or 4 (no n-gons).
        rules: ``(n_faces,)`` quad split rule (only consulted when
            count == 4). 0 = split 0-2 diagonal, otherwise split 1-3.

    Returns:
        ``(n_tri_fv,) int64`` mapping each triangulated face-vertex back
        to its original face-vertex position.
    """
    from cgmath.geometry.utils._numba._normals import _build_tri_expand_map

    counts = np.asarray(counts, dtype=np.int32)
    rules  = np.asarray(rules, dtype=np.int32)
    return _build_tri_expand_map(counts, rules)


# ------------------------------- Quadrangulate ------------------------------ #


def quad_match_greedy(score, fa, fb, n_faces):
    """Greedy maximum-weight matching on the triangle dual graph.

    Picks edges in descending score order, marking each incident face
    as used after a pick so it cannot be re-used.  Stops at the first
    non-finite score.

    Args:
        score:   ``(n_cand,)`` per-candidate quality (use ``-inf`` to skip).
        fa:      ``(n_cand,)`` first incident triangle id.
        fb:      ``(n_cand,)`` second incident triangle id.
        n_faces: total number of faces (size of the ``used`` mask).

    Returns:
        ``(n_cand,) bool`` -- True where the candidate edge was chosen.
    """
    from cgmath.geometry.utils._numba._topology import _quad_match_greedy

    score = np.ascontiguousarray(score, dtype=np.float64)
    fa    = np.ascontiguousarray(fa,    dtype=np.int64)
    fb    = np.ascontiguousarray(fb,    dtype=np.int64)
    if score.size == 0:
        return np.zeros(0, dtype=np.bool_)
    return _quad_match_greedy(score, fa, fb, int(n_faces))


# ------------------------- Bilinear Surface Sampler ------------------------- #


def compute_centroids(points, geometry, return_radiuses=False):
    """Optimized: uses parallel Numba kernel for centroid computation."""
    from cgmath.geometry.utils._numba._main import compute_centroids_fast

    return compute_centroids_fast(points, geometry, return_radiuses=return_radiuses)


def compute_samples(values, weights, geometry):
    from cgmath.geometry.utils._numba._bilinear import _compute_samples

    values   = np.asarray(values,   dtype=np.float64)
    weights  = np.asarray(weights,  dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)
    return _compute_samples(values, weights, geometry)


def remap_indices(vert_indices, uv_indices, distances):
    """returns closest unique corresponding streams of indices"""
    from cgmath.geometry.utils._numba._bilinear import _remap_indices

    vert_indices = np.asarray(vert_indices, dtype=np.int32)
    uv_indices   = np.asarray(uv_indices,   dtype=np.int32)
    distances    = np.asarray(distances,    dtype=np.float64)
    return _remap_indices(vert_indices, uv_indices, distances)


def bilinear_vectors(points, geometry, uv):
    """returns bilinear quad surface vectors at given uvs"""
    from cgmath.geometry.utils._numba._bilinear import _bilinear_vectors

    points   = np.asarray(points,   dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)
    uv       = np.asarray(uv,       dtype=np.float64)

    return _bilinear_vectors(points, geometry, uv)


def bezier_vectors(points, normals, geometry, uv):
    """Returns PN Quad Bezier surface tangent vectors at given UVs.

    For each query, builds a 4x4 bicubic control-point grid from the
    face corner positions and vertex normals (PN Quad scheme), then
    evaluates the partial derivatives dS/du and dS/dv.

    Args:
        points: (num_verts, 3) vertex positions.
        normals: (num_verts, 3) vertex normals.
        geometry: (num_queries, 4) per-query face vertex indices.
        uv: (num_queries, 2) parametric coordinates.

    Returns:
        U: (num_queries, 3) dS/du tangent vectors.
        V: (num_queries, 3) dS/dv tangent vectors.
    """
    from cgmath.geometry.utils._numba._bilinear import _bezier_vectors

    points   = np.asarray(points,   dtype=np.float64)
    normals  = np.asarray(normals,  dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)
    uv       = np.asarray(uv,       dtype=np.float64)

    return _bezier_vectors(points, normals, geometry, uv)


def bezier_evaluate(points, normals, geometry, uv):
    """Evaluates PN Quad Bezier surface positions at given UVs.

    For each query, builds a 4x4 bicubic control-point grid from the
    face corner positions and vertex normals (PN Quad scheme), then
    evaluates the surface position S(u, v).

    Args:
        points: (num_verts, 3) vertex positions.
        normals: (num_verts, 3) vertex normals.
        geometry: (num_queries, 4) per-query face vertex indices.
        uv: (num_queries, 2) parametric coordinates.

    Returns:
        positions: (num_queries, 3) surface positions.
    """
    from cgmath.geometry.utils._numba._bilinear import _bezier_evaluate

    points   = np.asarray(points,   dtype=np.float64)
    normals  = np.asarray(normals,  dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)
    uv       = np.asarray(uv,       dtype=np.float64)

    return _bezier_evaluate(points, normals, geometry, uv)


def bilinear_integrate(points, geometry, samples=10, method="gauss", tolerance=1e-4):
    """returns bilinear quad surface areas"""
    from cgmath.geometry.utils._numba._bilinear import (
        _bilinear_integrate_adaptive,
        _bilinear_integrate_gauss,
        _bilinear_integrate_quadrature,
    )

    points   = np.asarray(points, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)

    if method == "gauss":
        return _bilinear_integrate_gauss(points, geometry)
    elif method == "quadrature":
        return _bilinear_integrate_quadrature(points, geometry, np.int32(samples))
    elif method == "adaptive":
        return _bilinear_integrate_adaptive(
            points,
            geometry,
            np.int32(samples),
            np.int32(samples * 4),
            np.float64(tolerance),
        )
    else:
        raise ValueError(f"Unknown bilinear integration method: {method}")


def bilinear_sample(
    p,
    points,
    geometry,
    centroids,
    radiuses,
    centroid_distance_tolerance,
    iteration_count     = 100,
    iteration_tolerance = 1e-8,
    uv_border_tolerance = 1e-5,
    normals             = None,
):
    """returns a bilinear surface sample

    When normals are provided, uses PN Quad bicubic Bezier patches
    for higher accuracy on smooth meshes. Normals should be per-vertex
    unit normals of shape (num_points, 3).
    """
    from cgmath.geometry.utils._numba._bilinear import _bilinear_sample

    p                           = np.asarray(p,         dtype=np.float64)
    points                      = np.asarray(points,    dtype=np.float64)
    geometry                    = np.asarray(geometry,  dtype=np.int32)
    centroids                   = np.asarray(centroids, dtype=np.float64)
    radiuses                    = np.asarray(radiuses,  dtype=np.float64)
    centroid_distance_tolerance = np.float64(centroid_distance_tolerance)
    iteration_count             = np.int32(iteration_count)
    iteration_tolerance         = np.int32(iteration_tolerance)
    uv_border_tolerance         = np.float64(uv_border_tolerance)

    if normals is not None:
        from cgmath.geometry.utils._numba._bilinear import (
            _bezier_sample,
            _compute_all_pn_quad_cps,
            _compute_bezier_face_bounds,
        )

        normals        = np.asarray(normals, dtype=np.float64)
        control_points = _compute_all_pn_quad_cps(geometry, points, normals)
        bezier_centroids, bezier_radii = _compute_bezier_face_bounds(control_points)

        return _bezier_sample(
            p,
            points,
            geometry,
            control_points,
            bezier_centroids,
            bezier_radii,
            centroid_distance_tolerance,
            iteration_count,
            iteration_tolerance,
            uv_border_tolerance,
        )

    return _bilinear_sample(
        p,
        points,
        geometry,
        centroids,
        radiuses,
        centroid_distance_tolerance,
        iteration_count,
        iteration_tolerance,
        uv_border_tolerance,
    )


# ------------------------------- BSplineData -------------------------------- #


def compute_basis(u, kv, c, d):
    """returns bilinear quad surface areas"""
    from cgmath.geometry.utils._numba._bspline import _compute_basis

    u  = np.asarray(u, dtype=np.float64)
    kv = np.asarray(kv, dtype=np.int32)
    c  = np.int32(c)
    d  = np.int32(d)

    return _compute_basis(u, kv, c, d)


# --------------------------- Subdivision Surface ---------------------------- #
def rebuild_indices(
    new_indices, counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset
):
    """rebuilds the topology inplace"""
    from cgmath.geometry.utils._numba._subdivision import _rebuild_indices

    new_indices = np.asarray(new_indices, dtype=np.int32)
    counts      = np.asarray(counts,      dtype=np.int32)
    cumsum      = np.asarray(cumsum,      dtype=np.int32)
    f2v         = np.asarray(f2v,         dtype=np.int32)
    f2e         = np.asarray(f2e,         dtype=np.int32)
    e2v         = np.asarray(e2v,         dtype=np.int32)
    face_offset = np.int32(face_offset)
    edge_offset = np.int32(edge_offset)

    return _rebuild_indices(
        new_indices, counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset
    )


def subdivide_catmull_clark(
    points,
    indices,
    counts,
    f2v,
    f2e,
    e2v,
    e2f,
    v2f,
    v2e,
    keep_borders = False,
    keep_edges   = False,
    border_verts = None,
):
    """
    Perform one step of Catmull-Clark subdivision using optimized Numba kernels.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions (V, 3), float64.
    indices : np.ndarray
        Face vertex indices (flattened), int32.
    counts : np.ndarray
        Number of vertices per face (F,), int32.
    f2v : np.ndarray
        Face-to-vertex connectivity (F, max_verts), int32.
    f2e : np.ndarray
        Face-to-edge connectivity (F, max_edges), int32.
    e2v : np.ndarray
        Edge-to-vertex connectivity (E, 2), int32.
    e2f : np.ndarray
        Edge-to-face connectivity (E, 2), int32.
    v2f : np.ndarray
        Vertex-to-face connectivity (V, max_faces), int32.
    v2e : np.ndarray
        Vertex-to-edge connectivity (V, max_edges), int32.
    keep_borders : bool
        If True, keep border vertices at original positions.
    keep_edges : bool
        If True, keep all vertices at original positions and use edge midpoints.
    border_verts : np.ndarray or None
        Boolean mask of border vertices (V,). If None, computed internally.

    Returns
    -------
    tuple
        (new_points, new_indices, new_counts) for the subdivided mesh.
    """
    from cgmath.geometry.utils._numba._subdivide import (
        _compute_edge_midpoints,
        _compute_edge_points,
        _compute_face_points,
        _compute_vertex_points,
        _compute_vertex_points_keep_borders,
        _rebuild_subdivision_topology,
    )

    points         = np.asarray(points, dtype=np.float64)
    counts         = np.asarray(counts, dtype=np.int32)
    f2v            = np.asarray(f2v,    dtype=np.int32)
    f2e            = np.asarray(f2e,    dtype=np.int32)
    e2v            = np.asarray(e2v,    dtype=np.int32)
    e2f            = np.asarray(e2f,    dtype=np.int32)
    v2f            = np.asarray(v2f,    dtype=np.int32)
    v2e            = np.asarray(v2e,    dtype=np.int32)

    face_points    = _compute_face_points(points, f2v)
    edge_midpoints = _compute_edge_midpoints(points, e2v)

    if keep_edges:
        edge_points = edge_midpoints
    else:
        edge_points = _compute_edge_points(edge_midpoints, face_points, e2f)

    if keep_edges:
        vertex_points = points.copy()
    elif keep_borders:
        if border_verts is None:
            edge_valence = np.sum(v2e >= 0, axis=1)
            face_valence = np.sum(v2f >= 0, axis=1)
            border_verts = edge_valence != face_valence

        vertex_points = _compute_vertex_points_keep_borders(
            points, face_points, edge_midpoints, v2f, v2e, e2f, border_verts
        )
    else:
        vertex_points = _compute_vertex_points(
            points, face_points, edge_midpoints, v2f, v2e, e2f
        )

    new_points  = np.concatenate([vertex_points, edge_points, face_points])

    n_verts     = vertex_points.shape[0]
    n_edges     = edge_points.shape[0]
    edge_offset = n_verts
    face_offset = n_verts + n_edges

    new_indices, new_counts = _rebuild_subdivision_topology(
        counts, f2v, f2e, e2v, edge_offset, face_offset
    )

    return new_points, new_indices, new_counts


# weights, positive, negative, center, max_inf
def balance_center_weights(
    weights:  np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    center:   np.ndarray,
    max_inf:  int,
):
    """balances the weights of a center vertices up to max_inf"""
    from cgmath.geometry.utils._numba._blur import _balance_center_weights

    weights  = np.asarray(weights,  dtype=np.float64)
    positive = np.asarray(positive, dtype=np.int32)
    negative = np.asarray(negative, dtype=np.int32)
    center   = np.asarray(center,   dtype=np.int32)
    max_inf  = int(max_inf)

    return _balance_center_weights(weights, positive, negative, center, max_inf)


#  ----------------------------------- BLUR ----------------------------------- #
def inpaint(
    indices:       np.ndarray,
    weights:       np.ndarray,
    neighbors:     np.ndarray,
    iterations:    int        | np.ndarray = 9,
    receptions:    float      | np.ndarray = 0.5,
    contributions: float      | np.ndarray = 1.0,
):
    """a blur algorithm with default values tuned to match Maya's delta mush"""

    # indices        1d array of indices to inpaint
    # weights:       2d array of skin weights
    # neighbors:     neighborhood matrix
    # iteration:     per vertex iteration count, if count <= 0 ---> stop blurring this vertex
    # receptions:    multiplyer to the sum of neighberhood weights applied to a target vertex
    # contributions: how much this vertex contributes to the sum calculation (experimental)
    from cgmath.geometry.utils._numba._blur import _inpaint

    if isinstance(iterations, numbers.Real):
        iterations = np.ones(weights.shape[0], dtype=np.int32) * iterations
    else:
        iterations = np.array(iterations, dtype=np.int32)

    if isinstance(receptions, numbers.Real):
        receptions = np.ones(weights.shape[0]) * receptions

    if isinstance(contributions, numbers.Real):
        contributions = np.ones(weights.shape[0]) * contributions

    weights         = np.asarray(weights,   dtype=np.float64)
    indices         = np.asarray(indices,   dtype=np.int32)
    neighbors       = np.asarray(neighbors, dtype=np.int32)
    inpaint_indices = np.ones(weights.shape[0], dtype=np.int32)
    inpaint_indices[indices] = 0

    return _inpaint(
        inpaint_indices, weights, neighbors, iterations, receptions, contributions
    )


def blur(
    weights:       np.ndarray,
    neighbors:     np.ndarray,
    iterations:    int        | np.ndarray = 9,
    receptions:    float      | np.ndarray = 0.5,
    contributions: float      | np.ndarray = 1.0,
):
    """a blur algorithm with default values tuned to match Maya's delta mush"""

    # weights:       2d array of weights to be blurred
    # neighbors:     neighborhood matrix
    # iteration:     per vertex iteration count, if count <= 0 ---> stop blurring this vertex
    # receptions:    multiplyer to the sum of neighberhood weights applied to a target vertex
    # contributions: how much this vertex contributes to the sum calculation (experimental)
    from cgmath.geometry.utils._numba._blur import _blur

    neighbors = np.asarray(neighbors, dtype=np.int32)

    if isinstance(iterations, numbers.Real):
        iterations = np.ones(weights.shape[0], dtype=np.int32) * iterations
    else:
        iterations = np.array(iterations, dtype=np.int32)

    if isinstance(receptions, numbers.Real):
        receptions = np.ones(weights.shape[0]) * receptions

    if isinstance(contributions, numbers.Real):
        contributions = np.ones(weights.shape[0]) * contributions

    # if the weights is 1d array
    if weights.ndim == 1:
        return _blur(
            weights[:, None], neighbors, iterations, receptions, contributions
        ).ravel()

    return _blur(weights, neighbors, iterations, receptions, contributions)


# -------------------------------- DELTA MUSH -------------------------------- #


def encode_local_deltas(
    rest_points:   np.ndarray,
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
) -> np.ndarray:
    """Encode each rest vertex displacement in its smoothed local tangent frame.

    For every vertex builds an orthonormal frame ``(T, B, N)`` from the
    smoothed neighbour offsets, then projects ``rest - smooth`` into it.

    Args:
        rest_points: ``(N, 3)`` rest positions.
        smooth_points: ``(N, 3)`` smoothed rest positions.
        neighbors: ``(N, K)`` int32 neighbour matrix with ``-1`` padding.

    Returns:
        ``(N, 3)`` per-vertex local-frame deltas.
    """
    from cgmath.geometry.utils._numba._delta_mush import _encode_local_deltas

    rest_points   = np.ascontiguousarray(np.asarray(rest_points, dtype=np.float64))
    smooth_points = np.ascontiguousarray(np.asarray(smooth_points, dtype=np.float64))
    neighbors     = np.ascontiguousarray(np.asarray(neighbors, dtype=np.int32))

    out           = np.empty_like(rest_points)
    _encode_local_deltas(rest_points, smooth_points, neighbors, out)
    return out


def decode_local_deltas(
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
    deltas:        np.ndarray,
) -> np.ndarray:
    """Reconstruct world positions by re-applying stored local deltas.

    Inverse of :func:`encode_local_deltas`. Rebuilds the local tangent
    frame from ``smooth_points`` and applies the stored local delta.

    Args:
        smooth_points: ``(N, 3)`` smoothed deformed positions.
        neighbors: ``(N, K)`` int32 neighbour matrix with ``-1`` padding.
        deltas: ``(N, 3)`` local-frame deltas from :func:`encode_local_deltas`.

    Returns:
        ``(N, 3)`` reconstructed world-space positions.
    """
    from cgmath.geometry.utils._numba._delta_mush import _decode_local_deltas

    smooth_points = np.ascontiguousarray(np.asarray(smooth_points, dtype=np.float64))
    neighbors     = np.ascontiguousarray(np.asarray(neighbors, dtype=np.int32))
    deltas        = np.ascontiguousarray(np.asarray(deltas, dtype=np.float64))

    out           = np.empty_like(smooth_points)
    _decode_local_deltas(smooth_points, neighbors, deltas, out)
    return out


def blend_deltas(
    base_points:      np.ndarray,
    deltamush_points: np.ndarray,
    weight:           float,
) -> np.ndarray:
    """Linear blend ``base + weight * (deltamush - base)`` per vertex.

    Args:
        base_points: ``(N, 3)`` original (unfiltered) deformed positions.
        deltamush_points: ``(N, 3)`` reconstructed delta-mushed positions.
        weight: Mush weight in ``[0, 1]``.  ``0`` returns the smoothed
            mesh, ``1`` returns the fully reconstructed mesh.

    Returns:
        ``(N, 3)`` blended positions.
    """
    from cgmath.geometry.utils._numba._delta_mush import _blend_deltas

    base_points = np.ascontiguousarray(np.asarray(base_points, dtype=np.float64))
    deltamush_points = np.ascontiguousarray(
        np.asarray(deltamush_points, dtype=np.float64)
    )

    out = np.empty_like(base_points)
    _blend_deltas(base_points, deltamush_points, float(weight), out)
    return out


# ----------------------- alternative-frame wrappers ------------------------ #


def encode_local_deltas_tbn(
    rest_points:   np.ndarray,
    smooth_points: np.ndarray,
    first_nbrs:    np.ndarray,
) -> np.ndarray:
    """Encode rest displacement in each vertex's first-face TBN frame.

    Builds the per-vertex orthonormal frame ``(T, B, N)`` from the
    smoothed neighbour offsets to the *next* and *previous* vertex of the
    vertex's first face (``cross(next_off, prev_off)`` for ``N``).  This
    deterministic single-frame choice matches Maya's ``deltaMush`` output
    far more closely than the averaged-Laplacian normal of
    :func:`encode_local_deltas`.

    Args:
        rest_points: ``(N, 3)`` rest positions.
        smooth_points: ``(N, 3)`` smoothed rest positions.
        first_nbrs: ``(N, 2)`` int32 ``[next, prev]`` from each vertex's
            first face.  ``-1`` means "no usable face" and the world delta
            is stored verbatim.

    Returns:
        ``(N, 3)`` per-vertex local-frame deltas.
    """
    from cgmath.geometry.utils._numba._delta_mush import _encode_local_deltas_tbn

    rest_points   = np.ascontiguousarray(np.asarray(rest_points, dtype=np.float64))
    smooth_points = np.ascontiguousarray(np.asarray(smooth_points, dtype=np.float64))
    first_nbrs    = np.ascontiguousarray(np.asarray(first_nbrs, dtype=np.int32))

    out           = np.empty_like(rest_points)
    _encode_local_deltas_tbn(rest_points, smooth_points, first_nbrs, out)
    return out


def decode_local_deltas_tbn(
    smooth_points: np.ndarray,
    first_nbrs:    np.ndarray,
    deltas:        np.ndarray,
) -> np.ndarray:
    """Reconstruct world positions using first-face TBN frames.

    Inverse of :func:`encode_local_deltas_tbn`.  See its docstring for the
    frame definition.

    Args:
        smooth_points: ``(N, 3)`` smoothed deformed positions.
        first_nbrs: ``(N, 2)`` int32 ``[next, prev]`` per vertex.
        deltas: ``(N, 3)`` local-frame deltas from
            :func:`encode_local_deltas_tbn`.

    Returns:
        ``(N, 3)`` reconstructed world-space positions.
    """
    from cgmath.geometry.utils._numba._delta_mush import _decode_local_deltas_tbn

    smooth_points = np.ascontiguousarray(np.asarray(smooth_points, dtype=np.float64))
    first_nbrs    = np.ascontiguousarray(np.asarray(first_nbrs, dtype=np.int32))
    deltas        = np.ascontiguousarray(np.asarray(deltas, dtype=np.float64))

    out           = np.empty_like(smooth_points)
    _decode_local_deltas_tbn(smooth_points, first_nbrs, deltas, out)
    return out


def decode_world_deltas_procrustes(
    smooth_rest:  np.ndarray,
    smooth_def:   np.ndarray,
    neighbors:    np.ndarray,
    world_deltas: np.ndarray,
) -> np.ndarray:
    """Apply per-vertex Procrustes (SVD) rotation to world-space deltas.

    For each vertex, computes the optimal rigid rotation that maps its
    smoothed-rest neighbour offsets onto the smoothed-deformed offsets
    (cross-covariance polar decomposition), then applies that rotation to
    the cached world-space rest delta.

    Args:
        smooth_rest: ``(N, 3)`` smoothed rest positions (cached at bind).
        smooth_def: ``(N, 3)`` smoothed deformed positions.
        neighbors: ``(N, K)`` int32 neighbour matrix with ``-1`` padding.
        world_deltas: ``(N, 3)`` cached ``rest - smooth_rest`` offsets.

    Returns:
        ``(N, 3)`` reconstructed world-space positions.
    """
    from cgmath.geometry.utils._numba._delta_mush import (
        _decode_world_deltas_procrustes,
    )

    smooth_rest  = np.ascontiguousarray(np.asarray(smooth_rest, dtype=np.float64))
    smooth_def   = np.ascontiguousarray(np.asarray(smooth_def, dtype=np.float64))
    neighbors    = np.ascontiguousarray(np.asarray(neighbors, dtype=np.int32))
    world_deltas = np.ascontiguousarray(np.asarray(world_deltas, dtype=np.float64))

    out          = np.empty_like(smooth_rest)
    _decode_world_deltas_procrustes(
        smooth_rest, smooth_def, neighbors, world_deltas, out
    )
    return out


def ddm_precompute(
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute weighted neighbour offsets for the DDM apply step.

    Used during ``DeltaMushData.bind`` when ``method='DDM'``.

    Args:
        smooth_points: ``(N, 3)`` smoothed rest positions.
        neighbors: ``(N, K)`` int32 neighbour matrix with ``-1`` padding.

    Returns:
        Tuple ``(rest_offsets, rest_weights)`` where ``rest_offsets`` is
        ``(N, K, 3)`` and ``rest_weights`` is ``(N, K)``.  Both are zero
        in padded slots.
    """
    from cgmath.geometry.utils._numba._delta_mush import _ddm_precompute

    smooth_points = np.ascontiguousarray(np.asarray(smooth_points, dtype=np.float64))
    neighbors     = np.ascontiguousarray(np.asarray(neighbors, dtype=np.int32))

    n, k = neighbors.shape
    out_offsets = np.zeros((n, k, 3), dtype=np.float64)
    out_weights = np.zeros((n, k), dtype=np.float64)
    _ddm_precompute(smooth_points, neighbors, out_offsets, out_weights)
    return out_offsets, out_weights


def decode_world_deltas_ddm(
    smooth_def:   np.ndarray,
    neighbors:    np.ndarray,
    rest_offsets: np.ndarray,
    rest_weights: np.ndarray,
    world_deltas: np.ndarray,
) -> np.ndarray:
    """Apply Direct-Delta-Mush per-vertex rotation to world-space deltas.

    Like :func:`decode_world_deltas_procrustes` but uses precomputed
    smoothed-rest neighbour offsets and length-based weights to stabilise
    the local rotation against irregular topology (Le & Lewis 2019).

    Args:
        smooth_def: ``(N, 3)`` smoothed deformed positions.
        neighbors: ``(N, K)`` int32 neighbour matrix with ``-1`` padding.
        rest_offsets: ``(N, K, 3)`` cached offsets (from
            :func:`ddm_precompute`).
        rest_weights: ``(N, K)`` cached weights.
        world_deltas: ``(N, 3)`` cached ``rest - smooth_rest`` offsets.

    Returns:
        ``(N, 3)`` reconstructed world-space positions.
    """
    from cgmath.geometry.utils._numba._delta_mush import _decode_world_deltas_ddm

    smooth_def   = np.ascontiguousarray(np.asarray(smooth_def, dtype=np.float64))
    neighbors    = np.ascontiguousarray(np.asarray(neighbors, dtype=np.int32))
    rest_offsets = np.ascontiguousarray(np.asarray(rest_offsets, dtype=np.float64))
    rest_weights = np.ascontiguousarray(np.asarray(rest_weights, dtype=np.float64))
    world_deltas = np.ascontiguousarray(np.asarray(world_deltas, dtype=np.float64))

    out          = np.empty_like(smooth_def)
    _decode_world_deltas_ddm(
        smooth_def, neighbors, rest_offsets, rest_weights, world_deltas, out
    )
    return out


def batch_procrustes_rotations(
    H:        np.ndarray,
    max_iter: int        = 20,
) -> np.ndarray:
    """Per-cluster polar-decomposition rotations from covariance matrices.

    Drop-in replacement for the SVD step in batch Procrustes problems.
    Given the batch of 3x3 covariance matrices ``H = vectors1.T @ vectors0``
    returns the rotation factors of each ``H``.

    Args:
        H: ``(B, 3, 3)`` per-cluster covariance matrices.
        max_iter: Newton iteration cap (typically 12-20 is plenty).

    Returns:
        ``(B, 3, 3)`` rotation matrices.
    """
    from cgmath.geometry.utils._numba._delta_mush import _batch_procrustes_rotations

    H   = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    out = np.empty_like(H)
    _batch_procrustes_rotations(H, out, int(max_iter))
    return out


# ------------------------------ RASTERIZATION ------------------------------- #


def composite_sprite(
    sprite: np.ndarray,
    buffer: np.ndarray,
    mask:   np.ndarray | None                 = None,
    offset: np.ndarray | tuple[int, int]      = (0, 0),
    color:  np.ndarray | tuple[int, int, int] = (255, 255, 255),
):
    """composites a sprite to a buffer with offset and computes the overlap to a mask"""
    from cgmath.geometry.utils._numba._rasterize import _composite_sprite

    sprite = np.asarray(sprite, dtype=np.uint8)
    buffer = np.asarray(buffer, dtype=np.uint8)
    offset = np.asarray(offset, dtype=np.int32)
    color  = np.asarray(color,  dtype=np.uint8)

    if mask is None:
        mask = np.zeros(buffer.shape, dtype=np.uint8)
    else:
        mask = np.asarray(mask, dtype=np.uint8)

    return _composite_sprite(sprite, buffer, mask, offset, color)


def composite_sprite_aa(sprite: np.ndarray, buffer: np.ndarray):
    """composites an antialiased sprite to anothert sprite buffer"""
    from cgmath.geometry.utils._numba._rasterize import _composite_sprite_aa

    sprite = np.asarray(sprite, dtype=np.uint8)
    buffer = np.asarray(buffer, dtype=np.uint8)

    _composite_sprite_aa(sprite, buffer)


def compare_normals(normal_a, normal_b):
    """Optimized: uses parallel Numba kernel for angle computation."""
    from cgmath.geometry.utils._numba._main import compare_normals_fast

    return compare_normals_fast(normal_a, normal_b)


def vector_magnitude_difference(vector_a, vector_b):
    """returns the difference in magnitude between two vectors"""
    return np.abs(np.linalg.norm(vector_a) - np.linalg.norm(vector_b))


def vector_angle_difference(vector_a, vector_b):
    """Optimized: uses parallel Numba kernel for angle computation."""
    from cgmath.geometry.utils._numba._main import vector_angle_difference_fast

    return vector_angle_difference_fast(vector_a, vector_b)


def remap_array(
    arr: np.ndarray, t_min: int | np.ndarray = 0, t_max: int | np.ndarray = 1
) -> np.ndarray:
    """Optimized: uses parallel Numba kernel for remapping.

    Raises ValueError if any axis to remap has zero range (all identical values).
    """
    from cgmath.geometry.utils._numba._main import remap_array_fast

    arr = np.asarray(arr)
    if arr.size > 0:
        arr_min = arr.min(axis=-1)
        arr_max = arr.max(axis=-1)
        if np.any(arr_min == arr_max):
            raise ValueError(
                "Cannot remap array with identical values along an axis (zero range)"
            )

    return remap_array_fast(arr, t_min, t_max)


# ---------------------------------- Raycast --------------------------------- #


def bilinear_raycast(points, geometry, origins, directions, twosided=True, bvh=None):
    """Forward-only ray-mesh intersection on bilinear patches.

    Returns (t, uv, face_indices) for the closest forward hit per ray.
    When twosided=False, back-face hits are rejected.

    Args:
        bvh: optional pre-built ``BVHData`` (see
            ``cgmath.geometry.utils._numba._bvh``).  When supplied, the
            BVH-accelerated kernel is used instead of the per-face linear
            scan - typically 10-30x faster on meshes with thousands of
            faces.  When ``None``, the linear-scan kernel is used.
    """
    from cgmath.geometry.utils._numba._bilinear import (
        _compute_face_aabbs,
        _intersect_mesh_parallel,
        _intersect_mesh_parallel_bvh,
    )

    points     = np.ascontiguousarray(points, dtype=np.float64)
    geometry   = np.asarray(geometry, dtype=np.int32)
    origins    = np.ascontiguousarray(origins, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)

    # pad geometry to 4-wide if needed
    if geometry.shape[1] > 4:
        raise ValueError("ngons detected")
    elif geometry.shape[1] < 4:
        buffer = np.full((geometry.shape[0], 4), -1, dtype=np.int32)
        buffer[:, : geometry.shape[1]] = geometry
        geometry = buffer

    # normalize directions
    d_norms    = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / d_norms

    if bvh is not None:
        return _intersect_mesh_parallel_bvh(
            geometry,
            points,
            bvh.node_aabb_min,
            bvh.node_aabb_max,
            bvh.node_left,
            bvh.node_right,
            bvh.node_first,
            bvh.node_count,
            bvh.face_perm,
            origins,
            directions,
            twosided,
        )

    aabb_min, aabb_max = _compute_face_aabbs(geometry, points)
    return _intersect_mesh_parallel(
        geometry, points, aabb_min, aabb_max, origins, directions, twosided
    )


def bezier_raycast(
    points, geometry, normals, origins, directions, twosided=True, bvh=None
):
    """Forward-only ray-mesh intersection on bicubic Bezier patches (PN Quads).

    When normals are provided, constructs PN Quad bicubic Bezier patches
    for higher accuracy on smooth meshes. Normals should be per-vertex
    unit normals of shape (num_points, 3).

    Returns (t, uv, face_indices) for the closest forward hit per ray.
    When twosided=False, back-face hits are rejected.

    Args:
        bvh: optional pre-built ``BVHData`` over the bicubic Bezier
            control-point AABBs.  When supplied, the BVH-accelerated
            kernel is used instead of the per-face linear scan.  When
            ``None``, the linear-scan kernel is used.  Note: the BVH
            must be built over the **control point AABBs**, not the
            input vertex AABBs - see :func:`MeshData.bvh_bezier`.
    """
    from cgmath.geometry.utils._numba._bilinear import (
        _compute_all_pn_quad_cps,
        _compute_bezier_face_aabbs,
        _intersect_bezier_mesh_parallel,
        _intersect_bezier_mesh_parallel_bvh,
    )

    points     = np.ascontiguousarray(points, dtype=np.float64)
    geometry   = np.asarray(geometry, dtype=np.int32)
    normals    = np.ascontiguousarray(normals,    dtype=np.float64)
    origins    = np.ascontiguousarray(origins,    dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)

    # pad geometry to 4-wide if needed
    if geometry.shape[1] > 4:
        raise ValueError("ngons detected")
    elif geometry.shape[1] < 4:
        buffer = np.full((geometry.shape[0], 4), -1, dtype=np.int32)
        buffer[:, : geometry.shape[1]] = geometry
        geometry = buffer

    # normalize directions
    d_norms        = np.linalg.norm(directions, axis=1, keepdims=True)
    directions     = directions / d_norms

    control_points = _compute_all_pn_quad_cps(geometry, points, normals)

    if bvh is not None:
        return _intersect_bezier_mesh_parallel_bvh(
            control_points,
            bvh.node_aabb_min,
            bvh.node_aabb_max,
            bvh.node_left,
            bvh.node_right,
            bvh.node_first,
            bvh.node_count,
            bvh.face_perm,
            origins,
            directions,
            twosided,
        )

    aabb_min, aabb_max = _compute_bezier_face_aabbs(control_points)
    return _intersect_bezier_mesh_parallel(
        control_points, aabb_min, aabb_max, origins, directions, twosided
    )


# ---------------------------- Free-Form Deformation ------------------------- #


def get_cell_corners(lattice: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Gather 8 corners of each hex cell from a structured lattice.

    Args:
        lattice: (lx, ly, lz, 3) control point grid.
        cells:   (N, 3) integer cell indices (i, j, k).

    Returns:
        (N, 8, 3) corner positions.
        Corner order: (0,0,0) (1,0,0) (0,1,0) (1,1,0)
                      (0,0,1) (1,0,1) (0,1,1) (1,1,1)
    """
    i, j, k = cells[:, 0], cells[:, 1], cells[:, 2]
    return np.stack(
        [
            lattice[i, j, k],
            lattice[i + 1, j, k],
            lattice[i, j + 1, k],
            lattice[i + 1, j + 1, k],
            lattice[i, j, k + 1],
            lattice[i + 1, j, k + 1],
            lattice[i, j + 1, k + 1],
            lattice[i + 1, j + 1, k + 1],
        ],
        axis=1,
    )


def trilinear(uvw: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Forward trilinear interpolation.

    Args:
        uvw:     (N, 3) parametric coordinates.
        corners: (N, 8, 3) cell corner positions.

    Returns:
        (N, 3) interpolated positions.
    """
    u, v, w = uvw[:, 0:1], uvw[:, 1:2], uvw[:, 2:3]
    c00 = corners[:, 0] * (1 - u) + corners[:, 1] * u
    c01 = corners[:, 2] * (1 - u) + corners[:, 3] * u
    c10 = corners[:, 4] * (1 - u) + corners[:, 5] * u
    c11 = corners[:, 6] * (1 - u) + corners[:, 7] * u
    c0  = c00 * (1 - v) + c01 * v
    c1  = c10 * (1 - v) + c11 * v
    return c0 * (1 - w) + c1 * w


def trilinear_jacobian(uvw: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Jacobian of the trilinear mapping dP/d(uvw).

    Args:
        uvw:     (N, 3) parametric coordinates.
        corners: (N, 8, 3) cell corner positions.

    Returns:
        (N, 3, 3) Jacobian matrices where J[n, i, j] = dP_i / d_uvw_j.
    """
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

    dN_du = np.stack(
        [
            -(1 - v) * (1 - w),
            (1 - v) * (1 - w),
            -v * (1 - w),
            v * (1 - w),
            -(1 - v) * w,
            (1 - v) * w,
            -v * w,
            v * w,
        ],
        axis=-1,
    )

    dN_dv = np.stack(
        [
            -(1 - u) * (1 - w),
            -u * (1 - w),
            (1 - u) * (1 - w),
            u * (1 - w),
            -(1 - u) * w,
            -u * w,
            (1 - u) * w,
            u * w,
        ],
        axis=-1,
    )

    dN_dw = np.stack(
        [
            -(1 - u) * (1 - v),
            -u * (1 - v),
            -(1 - u) * v,
            -u * v,
            (1 - u) * (1 - v),
            u * (1 - v),
            (1 - u) * v,
            u * v,
        ],
        axis=-1,
    )

    J = np.empty((len(u), 3, 3))
    J[:, :, 0] = np.einsum("ni,nij->nj", dN_du, corners)
    J[:, :, 1] = np.einsum("ni,nij->nj", dN_dv, corners)
    J[:, :, 2] = np.einsum("ni,nij->nj", dN_dw, corners)
    return J


def inverse_trilinear(
    points:   np.ndarray,
    corners:  np.ndarray,
    uvw_init: np.ndarray | None = None,
    max_iter: int               = 20,
    tol:      float             = 1e-10,
) -> np.ndarray:
    """Newton-Raphson solve for parametric coords.

    Finds uvw such that ``trilinear(uvw, corners) ~= points``.
    Uses a Numba kernel with per-point early termination and
    Cramer's-rule 3x3 solve when available.

    Args:
        points:   (N, 3) target positions.
        corners:  (N, 8, 3) cell corner positions.
        uvw_init: (N, 3) initial guess (defaults to cell centre).
        max_iter: maximum Newton iterations.
        tol:      convergence tolerance.

    Returns:
        (N, 3) parametric coordinates.
    """
    from cgmath.geometry.utils._numba._ffd import _inverse_trilinear

    uvw = uvw_init.copy() if uvw_init is not None else np.full((len(points), 3), 0.5)

    _inverse_trilinear(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(corners, dtype=np.float64),
        uvw,
        max_iter,
        tol,
    )
    return uvw


def assign_cells(
    points:   np.ndarray,
    lattice:  np.ndarray,
    max_hops: int        = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """For each point find its containing cell and parametric (u, v, w).

    Strategy:
      1. Bounding-box normalisation -> initial cell guess
      2. Newton refinement
      3. Cell-hopping for any point whose uvw escaped [0, 1]

    Args:
        points:   (N, 3) mesh positions.
        lattice:  (lx, ly, lz, 3) rest lattice.
        max_hops: maximum cell-hop iterations.

    Returns:
        cells (N, 3) int, uvw (N, 3) float.
    """
    lx, ly, lz, _ = lattice.shape
    max_cells = np.array([lx - 2, ly - 2, lz - 2])

    flat      = lattice.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    size = hi - lo
    size[size < 1e-12] = 1.0

    global_param = (points - lo) / size * (max_cells + 1)
    cells        = np.clip(np.floor(global_param).astype(np.int64), 0, max_cells)
    uvw          = np.clip(global_param - cells, 0.0, 1.0)

    corners      = get_cell_corners(lattice, cells)
    uvw          = inverse_trilinear(points, corners, uvw)

    for _ in range(max_hops):
        escaped = np.any((uvw < -0.01) | (uvw > 1.01), axis=1)
        if not escaped.any():
            break

        idx = np.where(escaped)[0]
        ec  = cells[idx].copy()
        eu  = uvw[idx]

        for axis in range(3):
            ec[eu[:, axis] < -0.01, axis] -= 1
            ec[eu[:, axis] > 1.01, axis] += 1

        ec = np.clip(ec, 0, max_cells)
        cells[idx] = ec

        uvw[idx] = inverse_trilinear(
            points[idx],
            get_cell_corners(lattice, ec),
            np.full((len(idx), 3), 0.5),
        )

    return cells, np.ascontiguousarray(uvw, dtype=np.float64)


def build_lattice_topology(lx: int, ly: int, lz: int) -> tuple[np.ndarray, np.ndarray]:
    """Build quad face indices/counts for a structured (lx, ly, lz) grid.

    Generates all internal and surface quads -- every face of every hex cell.

    Args:
        lx, ly, lz: number of points per axis.

    Returns:
        (indices, counts) suitable for MeshData.  All faces are quads.
    """

    def idx(i, j, k):
        return i * ly * lz + j * lz + k

    faces = []

    # i-j planes (constant k)
    for k in range(lz):
        for i in range(lx - 1):
            for j in range(ly - 1):
                faces.append(
                    [
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                    ]
                )

    # i-k planes (constant j)
    for j in range(ly):
        for i in range(lx - 1):
            for k in range(lz - 1):
                faces.append(
                    [
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j, k + 1),
                        idx(i, j, k + 1),
                    ]
                )

    # j-k planes (constant i)
    for i in range(lx):
        for j in range(ly - 1):
            for k in range(lz - 1):
                faces.append(
                    [
                        idx(i, j, k),
                        idx(i, j + 1, k),
                        idx(i, j + 1, k + 1),
                        idx(i, j, k + 1),
                    ]
                )

    indices = np.array(faces, dtype=np.int32).ravel()
    counts  = np.full(len(faces), 4, dtype=np.int32)
    return indices, counts


def bernstein_basis_1d(t: np.ndarray, degree: int) -> np.ndarray:
    """Bernstein polynomial basis weights.

    Args:
        t:      (N,) parameter values (may be outside [0, 1] for extrapolation).
        degree: polynomial degree.

    Returns:
        (N, degree + 1) basis weights.
    """
    d     = degree
    binom = np.ones(d + 1, dtype=np.float64)
    for k in range(1, d + 1):
        binom[k] = binom[k - 1] * (d - k + 1) / k
    i = np.arange(d + 1, dtype=np.float64)
    return binom * t[:, None] ** i * (1.0 - t[:, None]) ** (d - i)


def _bernstein_eval_numpy(
    cells:           np.ndarray,
    uvw:             np.ndarray,
    delta:           np.ndarray,
    local_influence: tuple[int, int, int],
    divisions:       np.ndarray,
) -> np.ndarray:
    """Pure-NumPy Bernstein polynomial FFD evaluation (fallback).

    For each axis the evaluation window spans ``local_influence`` cells
    (clamped to lattice size).  When the window covers the entire lattice
    this reduces to the classic global-Bernstein FFD.

    Args:
        cells:           (N, 3) int cell indices.
        uvw:             (N, 3) float per-cell parametric coords.
        delta:           (lx, ly, lz, 3) displacement field.
        local_influence: (S, T, U) cells of influence per axis.
        divisions:       (3,) int cells per axis.

    Returns:
        (N, 3) displacement vectors.
    """
    lx, ly, lz, _ = delta.shape
    N            = len(uvw)

    axis_weights = []
    axis_windows = []

    for axis in range(3):
        n        = min(int(local_influence[axis]), int(divisions[axis]))
        num_cps  = int(divisions[axis]) + 1

        t_global = cells[:, axis].astype(np.float64) + uvw[:, axis]

        window_start = np.clip(
            np.floor(t_global - n * 0.5 + 0.5).astype(int),
            0,
            num_cps - n - 1,
        )

        t_local = (t_global - window_start) / n

        weights = bernstein_basis_1d(t_local, n)
        axis_weights.append(weights)
        axis_windows.append(window_start)

    ws, wt, wu = axis_weights
    win_i, win_j, win_k = axis_windows
    ni, nj, nk = ws.shape[1], wt.shape[1], wu.shape[1]

    result = np.zeros((N, 3))
    for a in range(ni):
        ii = np.clip(win_i + a, 0, lx - 1)
        wa = ws[:, a]
        for b in range(nj):
            jj  = np.clip(win_j + b, 0, ly - 1)
            wab = wa * wt[:, b]
            for c in range(nk):
                kk = np.clip(win_k + c, 0, lz - 1)
                w  = wab * wu[:, c]
                result += w[:, None] * delta[ii, jj, kk]

    return result


def bernstein_eval(
    cells:           np.ndarray,
    uvw:             np.ndarray,
    delta:           np.ndarray,
    local_influence: tuple[int, int, int],
    divisions:       np.ndarray,
) -> np.ndarray:
    """Bernstein polynomial FFD evaluation (Sederberg & Parry formulation).

    Uses a parallel Numba kernel when available, falling back to a pure
    NumPy implementation if the Numba kernel cannot be imported or fails
    to compile.

    Args:
        cells:           (N, 3) int cell indices.
        uvw:             (N, 3) float per-cell parametric coords.
        delta:           (lx, ly, lz, 3) displacement field.
        local_influence: (S, T, U) cells of influence per axis.
        divisions:       (3,) int cells per axis.

    Returns:
        (N, 3) displacement vectors.
    """
    try:
        from cgmath.geometry.utils._numba._ffd import _bernstein_eval

        out = np.empty((len(uvw), 3), dtype=np.float64)
        _bernstein_eval(
            np.ascontiguousarray(cells, dtype=np.int64),
            np.ascontiguousarray(uvw, dtype=np.float64),
            np.ascontiguousarray(delta, dtype=np.float64),
            int(local_influence[0]),
            int(local_influence[1]),
            int(local_influence[2]),
            int(divisions[0]),
            int(divisions[1]),
            int(divisions[2]),
            out,
        )
        return out
    except Exception:
        return _bernstein_eval_numpy(cells, uvw, delta, local_influence, divisions)


# ------------------------- Patch-based relaxation --------------------------- #


def build_vertex_rings(
    indices:     np.ndarray,
    counts:      np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build winding-ordered 1-ring adjacency for every vertex.

    Unlike ``MeshData.get_edge_vertex_neighbors`` the neighbours come back
    ordered around the vertex, which is what the decal-map
    parameterization needs.  Faces with fewer than three corners are
    ignored.

    Args:
        indices: Flat face-vertex index stream.
        counts: Per-face corner counts.
        point_count: Number of vertices in the mesh.

    Returns:
        Tuple ``(ring, valence, is_boundary)`` where ``ring`` is
        ``(N, K)`` int32 with ``-1`` padding, ``valence`` is ``(N,)``
        int32 and ``is_boundary`` is ``(N,)`` bool -- True for vertices
        whose ring does not close.
    """
    from cgmath.geometry.utils._numba._patch_relax import _chain_vertex_rings

    indices     = np.asarray(indices, dtype=np.int64).ravel()
    counts      = np.asarray(counts, dtype=np.int64).ravel()
    point_count = int(point_count)

    starts      = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    # per-corner (prev, next) pairs -- each face contributes one ring
    # segment to each of its corners
    corner_face = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    face_start  = starts[corner_face]
    face_size   = counts[corner_face]
    corner_pos  = np.arange(indices.size, dtype=np.int64) - face_start

    valid       = face_size >= 3
    if not np.any(valid):
        return (
            np.full((point_count, 1), -1, dtype=np.int32),
            np.zeros(point_count, dtype=np.int32),
            np.ones(point_count, dtype=np.bool_),
        )

    face_start = face_start[valid]
    face_size  = face_size[valid]
    corner_pos = corner_pos[valid]
    center     = indices[valid]

    prev_v     = indices[face_start + (corner_pos - 1) % face_size]
    next_v     = indices[face_start + (corner_pos + 1) % face_size]

    # group the pairs by their center vertex (CSR)
    order        = np.argsort(center, kind="stable")
    pair_prev    = np.ascontiguousarray(prev_v[order], dtype=np.int32)
    pair_next    = np.ascontiguousarray(next_v[order], dtype=np.int32)

    pair_counts  = np.bincount(center, minlength=point_count).astype(np.int64)
    pair_offsets = np.zeros(point_count + 1, dtype=np.int64)
    np.cumsum(pair_counts, out=pair_offsets[1:])

    # an open fan holds one more neighbour than it has segments
    capacity    = max(int(pair_counts.max()) + 1, 1)

    ring        = np.full((point_count, capacity), -1, dtype=np.int32)
    valence     = np.zeros(point_count,             dtype=np.int32)
    is_boundary = np.zeros(point_count,             dtype=np.bool_)
    used        = np.zeros((point_count, capacity), dtype=np.bool_)

    _chain_vertex_rings(
        pair_offsets, pair_prev, pair_next, ring, valence, is_boundary, used
    )
    return ring, valence, is_boundary


def compute_decal_maps(
    points:      np.ndarray,
    ring:        np.ndarray,
    valence:     np.ndarray,
    is_boundary: np.ndarray,
    out:         np.ndarray | None = None,
) -> np.ndarray:
    """Flatten each vertex's edge stencil into geodesic polar coordinates.

    Section 2 of "Patch-based Surface Relaxation" (de Goes et al. 2018).
    Edge lengths are preserved and the angles between consecutive edges
    are rescaled to sum to ``2*pi``, or ``pi`` on a boundary half-disk.

    Args:
        points: ``(N, 3)`` positions.
        ring: ``(N, K)`` int32 winding-ordered neighbours, ``-1`` padded.
        valence: ``(N,)`` int32 neighbour counts.
        is_boundary: ``(N,)`` bool open-fan flags.
        out: Optional ``(N, K, 2)`` buffer to write into, letting a
            relaxation loop reuse one allocation.

    Returns:
        ``(N, K, 2)`` decal coordinates, zero in padded slots.
    """
    from cgmath.geometry.utils._numba._patch_relax import _compute_decal_maps

    points      = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    ring        = np.ascontiguousarray(np.asarray(ring, dtype=np.int32))
    valence     = np.ascontiguousarray(np.asarray(valence, dtype=np.int32))
    is_boundary = np.ascontiguousarray(np.asarray(is_boundary, dtype=np.bool_))

    if out is None:
        out = np.zeros((ring.shape[0], ring.shape[1], 2), dtype=np.float64)

    _compute_decal_maps(points, ring, valence, is_boundary, out)
    return out


def compute_span_weights(
    decals:  np.ndarray,
    valence: np.ndarray,
) -> np.ndarray:
    """Span-aware edge weights from a decal map.

    Section 3 of "Patch-based Surface Relaxation" (de Goes et al. 2018).
    Favours neighbours forming the shorter part of an edge flow that sit
    orthogonal to long spans, so relaxation follows the patch layout
    instead of washing it out the way uniform Laplacian weights do.

    Args:
        decals: ``(N, K, 2)`` decal coordinates.
        valence: ``(N,)`` int32 neighbour counts.

    Returns:
        ``(N, K)`` weights, normalised per stencil and zero in padded slots.
    """
    from cgmath.geometry.utils._numba._patch_relax import _compute_span_weights

    decals  = np.ascontiguousarray(np.asarray(decals, dtype=np.float64))
    valence = np.ascontiguousarray(np.asarray(valence, dtype=np.int32))

    out     = np.zeros((decals.shape[0], decals.shape[1]), dtype=np.float64)
    _compute_span_weights(decals, valence, out)
    return out


def patch_relax(
    points:        np.ndarray,
    rest_points:   np.ndarray,
    ring:          np.ndarray,
    valence:       np.ndarray,
    is_boundary:   np.ndarray,
    weights:       np.ndarray,
    rest_vectors:  np.ndarray,
    rest_decals:   np.ndarray,
    iterations:    int               = 30,
    alpha:         float             = 1.0,
    step_size:     float             = 0.5,
    surface_blend: float             = 0.0,
    step_scale:    np.ndarray | None = None,
    polar_iters:   int               = 12,
) -> np.ndarray:
    """Run patch-based surface relaxation on a deformed pose.

    Sections 4 and 5 of "Patch-based Surface Relaxation" (de Goes et al.
    2018).  Jacobi iterations transfer the baseline patch arrangement
    onto the deformed mesh; with *surface_blend* above zero the
    displacement is additionally computed in decal space and lifted back
    onto the surface, which preserves volume.

    Args:
        points: ``(N, 3)`` deformed positions.
        rest_points: ``(N, 3)`` baseline patch layout.
        ring: ``(N, K)`` int32 winding-ordered neighbours.
        valence: ``(N,)`` int32 neighbour counts.
        is_boundary: ``(N,)`` bool open-fan flags.
        weights: ``(N, K)`` rest-based span-aware weights.
        rest_vectors: ``(N, 3)`` weighted baseline offsets.
        rest_decals: ``(N, K, 2)`` baseline decal maps.
        iterations: Number of Jacobi sweeps.
        alpha: Blends the baseline correction against plain relaxation --
            ``0`` is pure span-aware smoothing, ``1`` full restoration.
        step_size: Explicit update fraction; the paper's half step.
        surface_blend: Mix of the surface-constrained result in ``[0, 1]``.
        step_scale: Optional ``(N,)`` per-vertex step multiplier.  ``0``
            pins a vertex, which is how borders and paint masks are
            expressed.
        polar_iters: Newton iterations for the per-vertex rotation fit.

    Returns:
        ``(N, 3)`` relaxed positions.  The input array is not modified.
    """
    from cgmath.geometry.utils._numba._patch_relax import _relax_step

    points       = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    rest_points  = np.ascontiguousarray(np.asarray(rest_points, dtype=np.float64))
    ring         = np.ascontiguousarray(np.asarray(ring, dtype=np.int32))
    valence      = np.ascontiguousarray(np.asarray(valence, dtype=np.int32))
    is_boundary  = np.ascontiguousarray(np.asarray(is_boundary, dtype=np.bool_))
    weights      = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    rest_vectors = np.ascontiguousarray(np.asarray(rest_vectors, dtype=np.float64))
    rest_decals  = np.ascontiguousarray(np.asarray(rest_decals, dtype=np.float64))

    if step_scale is None:
        step_scale = np.ones(points.shape[0], dtype=np.float64)
    else:
        step_scale = np.ascontiguousarray(np.asarray(step_scale, dtype=np.float64))

    surface_blend = float(np.clip(surface_blend, 0.0, 1.0))
    use_surface   = surface_blend > 1e-6

    src           = points.copy()
    dst           = np.empty_like(src)

    # only allocated when the surface-constrained regime is active
    posed_decals = (
        np.zeros_like(rest_decals) if use_surface else rest_decals[:1, :1].copy()
    )

    for _ in range(max(int(iterations), 0)):
        if use_surface:
            compute_decal_maps(src, ring, valence, is_boundary, out=posed_decals)

        _relax_step(
            src,
            rest_points,
            ring,
            valence,
            is_boundary,
            weights,
            rest_vectors,
            rest_decals,
            posed_decals,
            step_scale,
            float(alpha),
            float(step_size),
            surface_blend,
            use_surface,
            int(polar_iters),
            dst,
        )
        src, dst = dst, src

    return src