"""
Optimized Numba implementations for functions in main.py.

These functions provide parallel, JIT-compiled alternatives to the
pure NumPy implementations for improved performance on large datasets.
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                           point_row_overlaps                                #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _point_row_overlaps_process_pairs(
    query_indices: np.ndarray,
    query_indptr:  np.ndarray,
    num_points:    int,
) -> tuple:
    """
    Process cKDTree query_ball_point results to find overlapping points.

    Since cKDTree returns variable-length neighbor lists, we use a CSR-like
    format (indices + indptr) to pass the data to Numba.

    Parameters
    ----------
    query_indices : np.ndarray
        Flattened array of neighbor indices from query_ball_point (int32).
    query_indptr : np.ndarray
        Index pointers where query_indptr[i] is the start of neighbors for point i,
        and query_indptr[i+1] is the end (int32). Length is num_points + 1.
    num_points : int
        Number of points in the dataset.

    Returns
    -------
    tuple
        (unique_mask, pairs_src, pairs_dst) where:
        - unique_mask: boolean array indicating unique points
        - pairs_src: source indices of duplicate pairs
        - pairs_dst: destination indices (the unique point each duplicate maps to)
    """
    pair_counts = np.zeros(num_points, dtype=np.int32)

    # First pass: count pairs. Each duplicate point (whose smallest
    # neighbor index is less than itself) contributes exactly one pair.
    for i in prange(num_points):
        start       = query_indptr[i]
        end         = query_indptr[i + 1]
        n_neighbors = end - start

        if n_neighbors > 1:
            first_neighbor = query_indices[start]
            if first_neighbor < i:
                pair_counts[i] = 1

    total_pairs = np.int32(0)
    for i in range(num_points):
        total_pairs += pair_counts[i]

    if total_pairs == 0:
        unique_mask = np.ones(num_points, dtype=np.bool_)
        return unique_mask, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    pair_offsets = np.zeros(num_points + 1, dtype=np.int32)
    for i in range(num_points):
        pair_offsets[i + 1] = pair_offsets[i] + pair_counts[i]

    pairs_src = np.empty(total_pairs, dtype=np.int32)
    pairs_dst = np.empty(total_pairs, dtype=np.int32)

    # Second pass: collect one pair per duplicate point
    for i in prange(num_points):
        start       = query_indptr[i]
        end         = query_indptr[i + 1]
        n_neighbors = end - start

        if n_neighbors > 1:
            first_neighbor = query_indices[start]
            if first_neighbor < i:
                write_idx            = pair_offsets[i]
                pairs_src[write_idx] = i
                pairs_dst[write_idx] = first_neighbor

    # Compute unique_mask from collected pairs (race-free)
    unique_mask = np.ones(num_points, dtype=np.bool_)
    for k in range(total_pairs):
        unique_mask[pairs_src[k]] = False

    return unique_mask, pairs_src, pairs_dst


def point_row_overlaps_fast(matrix: np.ndarray, tolerance: float = 1e-6) -> tuple:
    """
    Fast version of point_row_overlaps using Numba for pair processing.

    This function still uses scipy's cKDTree for the spatial query (which is
    already highly optimized C code), but parallelizes the pair extraction.

    Parameters
    ----------
    matrix : np.ndarray
        Point matrix of shape (N, D) where N is number of points and D is dimension.
    tolerance : float
        Distance tolerance for considering points as overlapping.

    Returns
    -------
    tuple
        (unique_indices, duplicate_indices, original_indices) where:
        - unique_indices: indices of unique points
        - duplicate_indices: indices of duplicate points
        - original_indices: the unique point each duplicate maps to
    """
    from scipy.spatial import cKDTree

    if matrix is None or not matrix.shape[0]:
        empty = np.array([], dtype=np.int32)
        return np.array([], dtype=np.intp), empty, empty

    matrix          = np.asarray(matrix, dtype=np.float64)
    tree            = cKDTree(matrix)
    query_results   = tree.query_ball_point(matrix, tolerance)

    total_neighbors = sum(len(x) for x in query_results)
    query_indices   = np.empty(total_neighbors, dtype=np.int32)
    query_indptr    = np.zeros(len(query_results) + 1, dtype=np.int32)

    idx = 0
    for i, neighbors in enumerate(query_results):
        for n in neighbors:
            query_indices[idx] = n
            idx += 1
        query_indptr[i + 1] = idx

    unique_mask, pairs_src, pairs_dst = _point_row_overlaps_process_pairs(
        query_indices, query_indptr, len(matrix)
    )

    unique_indices = np.where(unique_mask)[0]

    if len(pairs_src) > 0:
        sort_idx  = np.argsort(pairs_dst)
        pairs_src = pairs_src[sort_idx]
        pairs_dst = pairs_dst[sort_idx]

    return unique_indices, pairs_src, pairs_dst


# --------------------------------------------------------------------------- #
#                             compute_centroids                               #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compute_centroids_parallel(
    points:   np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    """
    Compute face centroids in parallel.

    Fuses counting, masking, and summing into a single pass per face.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions of shape (V, D) with dtype float64.
    geometry : np.ndarray
        Face-vertex connectivity of shape (F, W) with dtype int32.
        Padded with -1 for faces with fewer vertices.

    Returns
    -------
    np.ndarray
        Centroids of shape (F, D) with dtype float64.
    """
    n_faces   = geometry.shape[0]
    n_cols    = geometry.shape[1]
    dim       = points.shape[1]
    centroids = np.zeros((n_faces, dim), dtype=np.float64)

    for i in prange(n_faces):
        count = np.int32(0)
        for j in range(n_cols):
            idx = geometry[i, j]
            if idx >= 0:
                count += 1
                for d in range(dim):
                    centroids[i, d] += points[idx, d]

        if count > 0:
            inv_count = 1.0 / count
            for d in range(dim):
                centroids[i, d] *= inv_count

    return centroids


@njit(parallel=True, fastmath=True, cache=True)
def _compute_centroids_and_radiuses_parallel(
    points:   np.ndarray,
    geometry: np.ndarray,
) -> tuple:
    """
    Compute face centroids and squared radiuses in parallel.

    Fuses all operations into a single pass per face, avoiding
    intermediate array allocations.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions of shape (V, D) with dtype float64.
    geometry : np.ndarray
        Face-vertex connectivity of shape (F, W) with dtype int32.
        Padded with -1 for faces with fewer vertices.

    Returns
    -------
    tuple
        (centroids, radiuses_squared) where:
        - centroids: shape (F, D) with dtype float64
        - radiuses_squared: shape (F,) with dtype float64, max squared distance
          from centroid to any vertex of the face
    """
    n_faces   = geometry.shape[0]
    n_cols    = geometry.shape[1]
    dim       = points.shape[1]
    centroids = np.zeros((n_faces, dim), dtype=np.float64)
    radiuses  = np.zeros(n_faces, dtype=np.float64)

    for i in prange(n_faces):
        count = np.int32(0)
        for j in range(n_cols):
            idx = geometry[i, j]
            if idx >= 0:
                count += 1
                for d in range(dim):
                    centroids[i, d] += points[idx, d]

        if count > 0:
            inv_count = 1.0 / count
            for d in range(dim):
                centroids[i, d] *= inv_count

            max_dist_sq = 0.0
            for j in range(n_cols):
                idx = geometry[i, j]
                if idx >= 0:
                    dist_sq = 0.0
                    for d in range(dim):
                        diff = points[idx, d] - centroids[i, d]
                        dist_sq += diff * diff
                    if dist_sq > max_dist_sq:
                        max_dist_sq = dist_sq

            radiuses[i] = max_dist_sq

    return centroids, radiuses


# --------------------------------------------------------------------------- #
#                       compare_normals / vector_angle                        #
# --------------------------------------------------------------------------- #


@njit(parallel=True, cache=True)
def _compare_normals_parallel(
    normal_a: np.ndarray,
    normal_b: np.ndarray,
) -> np.ndarray:
    """
    Compute angles between pairs of normal vectors in parallel.

    Fuses dot product, clipping, arccos, and degree conversion into
    a single pass per vector pair.

    Parameters
    ----------
    normal_a : np.ndarray
        First set of normal vectors, shape (N, D) with dtype float64.
    normal_b : np.ndarray
        Second set of normal vectors, shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        Angles in degrees, shape (N,) with dtype float64.
    """
    n          = normal_a.shape[0]
    dim        = normal_a.shape[1]
    deg_angles = np.empty(n, dtype=np.float64)

    rad_to_deg = 180.0 / np.pi

    for i in prange(n):
        dot = 0.0
        for d in range(dim):
            dot += normal_a[i, d] * normal_b[i, d]

        if dot < -1.0:
            dot = -1.0
        elif dot > 1.0:
            dot = 1.0

        deg_angles[i] = np.arccos(dot) * rad_to_deg

    return deg_angles


@njit(parallel=True, cache=True)
def _vector_angle_difference_parallel(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> np.ndarray:
    """
    Compute angles between pairs of arbitrary vectors in parallel.

    Unlike _compare_normals_parallel, this normalizes each vector
    before computing the dot product, so it works on non-unit vectors.

    Parameters
    ----------
    vector_a : np.ndarray
        First set of vectors, shape (N, D) with dtype float64.
    vector_b : np.ndarray
        Second set of vectors, shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        Angles in degrees, shape (N,) with dtype float64.
    """
    n          = vector_a.shape[0]
    dim        = vector_a.shape[1]
    deg_angles = np.empty(n, dtype=np.float64)
    rad_to_deg = 180.0 / np.pi

    for i in prange(n):
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        dot       = 0.0
        for d in range(dim):
            norm_a_sq += vector_a[i, d] * vector_a[i, d]
            norm_b_sq += vector_b[i, d] * vector_b[i, d]
            dot       += vector_a[i, d] * vector_b[i, d]

        norm_a = np.sqrt(norm_a_sq)
        norm_b = np.sqrt(norm_b_sq)

        if norm_a > 0.0 and norm_b > 0.0:
            dot /= norm_a * norm_b

            if dot < -1.0:
                dot = -1.0
            elif dot > 1.0:
                dot = 1.0

            deg_angles[i] = np.arccos(dot) * rad_to_deg
        else:
            deg_angles[i] = 0.0

    return deg_angles


# --------------------------------------------------------------------------- #
#                              stream_to_matrix                               #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _stream_to_matrix_parallel(
    values:         np.ndarray,
    counts:         np.ndarray,
    chunk_starts:   np.ndarray,
    max_chunk_size: int,
) -> np.ndarray:
    """
    Convert index stream to matrix with -1 padding in parallel.

    Parameters
    ----------
    values : np.ndarray
        Flattened array of values (int32).
    counts : np.ndarray
        Number of elements per row (int32).
    chunk_starts : np.ndarray
        Cumulative sum giving start index for each row (int32).
    max_chunk_size : int
        Maximum row width (determines output matrix width).

    Returns
    -------
    np.ndarray
        Matrix of shape (len(counts), max_chunk_size) with dtype int32.
        Padded with -1 for rows shorter than max_chunk_size.
    """
    n_rows = counts.shape[0]
    result = np.full((n_rows, max_chunk_size), -1, dtype=np.int32)

    for i in prange(n_rows):
        start = chunk_starts[i]
        count = counts[i]
        for j in range(count):
            result[i, j] = values[start + j]

    return result


def stream_to_matrix_fast(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Fast version of stream_to_matrix using parallel Numba kernel.

    Parameters
    ----------
    values : np.ndarray
        Flattened array of values.
    counts : np.ndarray
        Number of elements per row.

    Returns
    -------
    np.ndarray
        Matrix with -1 padding for short rows.
    """
    values = np.asarray(values, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32)

    if counts.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    max_chunk_size   = int(np.max(counts))
    chunk_starts     = np.zeros(len(counts), dtype=np.int32)
    chunk_starts[1:] = np.cumsum(counts[:-1])

    return _stream_to_matrix_parallel(values, counts, chunk_starts, max_chunk_size)


# --------------------------------------------------------------------------- #
#                              indices_replace                                #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _indices_replace_parallel(
    matrix:       np.ndarray,
    from_indices: np.ndarray,
    to_indices:   np.ndarray,
) -> np.ndarray:
    """
    Replace indices in a matrix using binary search in parallel.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to modify (will be copied), int32.
    from_indices : np.ndarray
        Sorted array of source indices to replace, int32.
    to_indices : np.ndarray
        Corresponding replacement indices, int32.

    Returns
    -------
    np.ndarray
        Matrix with replaced indices, int32.
    """
    result     = matrix.copy()
    n_elements = result.size
    n_from     = from_indices.shape[0]

    for i in prange(n_elements):
        val = result.flat[i]

        lo  = 0
        hi  = n_from
        while lo < hi:
            mid = (lo + hi) // 2
            if from_indices[mid] < val:
                lo = mid + 1
            else:
                hi = mid

        if lo < n_from and from_indices[lo] == val:
            result.flat[i] = to_indices[lo]

    return result


def indices_replace_fast(
    matrix:       np.ndarray,
    from_indices: np.ndarray,
    to_indices:   np.ndarray,
    collapse:     bool       = False,
) -> np.ndarray:
    """
    Fast version of indices_replace using parallel Numba kernel.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    from_indices : np.ndarray
        Source indices to replace.
    to_indices : np.ndarray
        Replacement indices.
    collapse : bool
        If True, renumber remaining indices to be contiguous.

    Returns
    -------
    np.ndarray
        Matrix with replaced indices.
    """
    if matrix is None:
        return matrix

    matrix = np.asarray(matrix, dtype=np.int32)
    if not matrix.shape[0]:
        return matrix

    from_indices = np.asarray(from_indices, dtype=np.int32)
    to_indices   = np.asarray(to_indices, dtype=np.int32)

    shape = matrix.shape

    sort_idx    = np.argsort(from_indices)
    from_sorted = from_indices[sort_idx]
    to_sorted   = to_indices[sort_idx]

    result = _indices_replace_parallel(matrix.ravel(), from_sorted, to_sorted)
    result = result.reshape(shape)

    if collapse:
        unique = np.unique(result)
        unique = unique[unique >= 0]
        return indices_replace_fast(
            result,
            unique,
            np.arange(len(unique), dtype=np.int32),
            collapse=False,
        )

    return result


# --------------------------------------------------------------------------- #
#                              average_points                                 #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _average_points_parallel(
    points:   np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    """
    Average points per geometry row, ignoring -1 indices.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions of shape (V, D) with dtype float64.
    geometry : np.ndarray
        Index matrix of shape (N, W) with dtype int32.
        Padded with -1 for missing indices.

    Returns
    -------
    np.ndarray
        Averaged points of shape (N, D) with dtype float64.
    """
    n_rows = geometry.shape[0]
    n_cols = geometry.shape[1]
    dim    = points.shape[1]
    result = np.zeros((n_rows, dim), dtype=np.float64)

    for i in prange(n_rows):
        count = np.int32(0)
        for j in range(n_cols):
            idx = geometry[i, j]
            if idx >= 0:
                count += 1
                for d in range(dim):
                    result[i, d] += points[idx, d]

        if count > 0:
            inv_count = 1.0 / count
            for d in range(dim):
                result[i, d] *= inv_count

    return result


def average_points_fast(points: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """
    Fast version of average_points using parallel Numba kernel.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions.
    geometry : np.ndarray
        Index matrix with -1 padding.

    Returns
    -------
    np.ndarray
        Averaged points.
    """
    points   = np.asarray(points, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)
    return _average_points_parallel(points, geometry)


# --------------------------------------------------------------------------- #
#                            matrix_row_overlaps                              #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compute_row_hashes(matrix: np.ndarray) -> np.ndarray:
    """
    Compute simple hash for each row for quick comparison.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix, int32.

    Returns
    -------
    np.ndarray
        Hash values, int64.
    """
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    hashes = np.empty(n_rows, dtype=np.int64)

    for i in prange(n_rows):
        h = np.int64(0)
        for j in range(n_cols):
            val = np.int64(matrix[i, j])
            h   = h * 31 + val + 1
        hashes[i] = h

    return hashes


@njit(fastmath=True, cache=True)
def _rows_equal(matrix: np.ndarray, i: int, j: int) -> bool:
    """Check if two rows are equal."""
    n_cols = matrix.shape[1]
    for k in range(n_cols):
        if matrix[i, k] != matrix[j, k]:
            return False
    return True


@njit(fastmath=True, cache=True)
def _find_row_overlaps(
    matrix:       np.ndarray,
    hashes:       np.ndarray,
    sort_indices: np.ndarray,
) -> tuple:
    """
    Find overlapping rows using hash-based comparison.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (sorted rows if exact=False), int32.
    hashes : np.ndarray
        Row hashes, int64.
    sort_indices : np.ndarray
        Indices that would sort the hashes, int32.

    Returns
    -------
    tuple
        (unique_indices, dup_from, dup_to)
    """
    n_rows = matrix.shape[0]

    is_unique        = np.ones(n_rows, dtype=np.bool_)
    first_occurrence = np.arange(n_rows, dtype=np.int32)

    for si in range(n_rows):
        i = sort_indices[si]
        if not is_unique[i]:
            continue

        for sj in range(si + 1, n_rows):
            j = sort_indices[sj]
            if hashes[j] != hashes[i]:
                break

            if is_unique[j] and _rows_equal(matrix, i, j):
                is_unique[j]        = False
                first_occurrence[j] = i

    n_unique = 0
    n_dups   = 0
    for i in range(n_rows):
        if is_unique[i]:
            n_unique += 1
        else:
            n_dups += 1

    unique_indices = np.empty(n_unique, dtype=np.int32)
    dup_from       = np.empty(n_dups,   dtype=np.int32)
    dup_to         = np.empty(n_dups,   dtype=np.int32)

    ui = 0
    di = 0
    for i in range(n_rows):
        if is_unique[i]:
            unique_indices[ui] = i
            ui += 1
        else:
            dup_from[di] = i
            dup_to[di]   = first_occurrence[i]
            di += 1

    return unique_indices, dup_from, dup_to


def matrix_row_overlaps_fast(
    matrix:       np.ndarray,
    exact:        bool       = True,
    sort_indices: bool       = False,
) -> tuple:
    """
    Fast version of matrix_row_overlaps using Numba.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    exact : bool
        If False, rows are sorted before comparison.
    sort_indices : bool
        If True, sort output indices.

    Returns
    -------
    tuple
        (unique_indices, duplicate_indices, original_indices)
    """
    if matrix is None or not matrix.shape[0]:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    matrix = np.asarray(matrix, dtype=np.int32)

    if not exact:
        matrix = np.sort(matrix, axis=1)

    hashes            = _compute_row_hashes(matrix)
    hash_sort_indices = np.argsort(hashes, kind="stable").astype(np.int32)

    unique_idx, dup_from, dup_to = _find_row_overlaps(matrix, hashes, hash_sort_indices)

    if sort_indices and len(dup_to) > 0:
        sort_order = np.argsort(dup_to)
        dup_from   = dup_from[sort_order]
        dup_to     = dup_to[sort_order]

    return unique_idx, dup_from, dup_to


# --------------------------------------------------------------------------- #
#                         Public API (matches main.py)                        #
# --------------------------------------------------------------------------- #


def compute_centroids_fast(
    points:          np.ndarray,
    geometry:        np.ndarray,
    return_radiuses: bool       = False,
) -> np.ndarray | tuple:
    """
    Fast version of compute_centroids using parallel Numba kernels.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions.
    geometry : np.ndarray
        Face-vertex connectivity with -1 padding.
    return_radiuses : bool
        If True, also return squared radiuses.

    Returns
    -------
    np.ndarray or tuple
        Centroids, or (centroids, radiuses_squared) if return_radiuses=True.
    """
    points   = np.asarray(points, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.int32)

    if return_radiuses:
        return _compute_centroids_and_radiuses_parallel(points, geometry)
    else:
        return _compute_centroids_parallel(points, geometry)


def compare_normals_fast(
    normal_a: np.ndarray,
    normal_b: np.ndarray,
) -> np.ndarray:
    """
    Fast version of compare_normals using parallel Numba kernel.

    Parameters
    ----------
    normal_a : np.ndarray
        First set of normal vectors.
    normal_b : np.ndarray
        Second set of normal vectors.

    Returns
    -------
    np.ndarray
        Angles in degrees.
    """
    normal_a = np.asarray(normal_a, dtype=np.float64)
    normal_b = np.asarray(normal_b, dtype=np.float64)
    return _compare_normals_parallel(normal_a, normal_b)


def vector_angle_difference_fast(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> np.ndarray:
    """
    Fast version of vector_angle_difference using parallel Numba kernel.

    Parameters
    ----------
    vector_a : np.ndarray
        First set of vectors.
    vector_b : np.ndarray
        Second set of vectors.

    Returns
    -------
    np.ndarray
        Angles in degrees.
    """
    vector_a = np.asarray(vector_a, dtype=np.float64)
    vector_b = np.asarray(vector_b, dtype=np.float64)
    return _vector_angle_difference_parallel(vector_a, vector_b)