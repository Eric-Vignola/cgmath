"""
Optimized Numba implementations for skin_weights.py functions.

These functions provide parallel, JIT-compiled alternatives to the
pure NumPy implementations for improved performance on large skin weight datasets.
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                            _round_normalize                                 #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _round_normalize_parallel(
    weights: np.ndarray,
    n:       int,
) -> np.ndarray:
    """
    Round weights to n decimal places and normalize by adding delta to max value.

    Fuses argmax, round, sum, and scatter-add into a single pass per row.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M) with dtype float64.
    n : int
        Number of decimal places to round to.

    Returns
    -------
    np.ndarray
        Rounded and normalized weight matrix.
    """
    n_rows = weights.shape[0]
    n_cols = weights.shape[1]
    result = np.empty_like(weights)

    factor = 10.0**n

    for i in prange(n_rows):
        max_idx   = 0
        max_val   = -np.inf
        first_idx = -1
        row_sum   = 0.0

        # Single pass: round, find max index, compute sum
        for j in range(n_cols):
            # Round to n decimal places
            rounded = np.round(weights[i, j] * factor) / factor
            result[i, j] = rounded
            row_sum += rounded

            # Track max value and its index
            if rounded > max_val:
                max_val = rounded
                max_idx = j

            if first_idx < 0 and rounded != 0.0:
                first_idx = j

        # Add delta to the max value to ensure sum = 1.0
        delta = 1.0 - row_sum
        if max_val + delta < 0.0:
            # The row holds more influences than n decimals can express, so the
            # residual is bigger than the column meant to absorb it. Collapse
            # onto the first influence the quantization can still hold.
            for j in range(n_cols):
                result[i, j] = 0.0
            if first_idx < 0:
                first_idx = 0
            result[i, first_idx] = 1.0
        else:
            result[i, max_idx] += delta

    return result


# --------------------------------------------------------------------------- #
#                               normalize                                     #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _normalize_weights_parallel(
    weights:      np.ndarray,
    indices:      np.ndarray,
    include_mask: np.ndarray,
) -> None:
    """
    Normalize weights in-place so they sum to 1.0 for specified rows.

    Only columns where include_mask[j] == True are modified.
    The normalization distributes the delta proportionally.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M) with dtype float64. Modified in-place.
    indices : np.ndarray
        Row indices to normalize, int32 or int64.
    include_mask : np.ndarray
        Boolean mask of shape (M,) indicating which columns to include.
    """
    n_indices = indices.shape[0]
    n_cols    = weights.shape[1]

    for ii in prange(n_indices):
        i = indices[ii]

        # Compute full sum and partial sum (only included columns)
        full_sum    = 0.0
        partial_sum = 0.0

        for j in range(n_cols):
            val = np.abs(weights[i, j])
            full_sum += val
            if include_mask[j]:
                partial_sum += val

        delta = 1.0 - full_sum

        # Distribute delta proportionally to included columns
        if partial_sum > 0.0:
            inv_partial = 1.0 / partial_sum
            for j in range(n_cols):
                if include_mask[j]:
                    ratio = weights[i, j] * inv_partial
                    weights[i, j] += ratio * delta
        else:
            # If all included weights are zero, bind the first included column
            for j in range(n_cols):
                if include_mask[j]:
                    weights[i, j] += delta
                    break


# --------------------------------------------------------------------------- #
#                          Scatter operations                                 #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _scatter_weights_to_dense(
    influence_indices: np.ndarray,
    weights_flat:      np.ndarray,
    n_verts:           int,
    n_influences:      int,
    max_influences:    int,
) -> np.ndarray:
    """
    Scatter compact skin weights to dense matrix format.

    Replaces the slow np.add.at operation with parallel per-row scattering.

    Parameters
    ----------
    influence_indices : np.ndarray
        Flat array of influence indices, shape (n_verts * max_influences,), int32.
    weights_flat : np.ndarray
        Flat array of weights, shape (n_verts * max_influences,), float64.
    n_verts : int
        Number of vertices.
    n_influences : int
        Total number of influences (columns in output).
    max_influences : int
        Maximum influences per vertex.

    Returns
    -------
    np.ndarray
        Dense weight matrix of shape (n_verts, n_influences), float64.
    """
    result = np.zeros((n_verts, n_influences), dtype=np.float64)

    for i in prange(n_verts):
        base = i * max_influences
        for j in range(max_influences):
            idx = influence_indices[base + j]
            if idx >= 0:  # Handle potential -1 padding
                result[i, idx] += weights_flat[base + j]

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _gather_top_k_weights(
    weights: np.ndarray,
    k:       int,
) -> tuple:
    """
    Gather top k weights and their indices per row.

    Fuses argpartition + take_along_axis into a single parallel operation.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M), float64.
    k : int
        Number of top weights to keep per row.

    Returns
    -------
    tuple
        (top_indices, top_weights) both of shape (N, k)
    """
    n_rows      = weights.shape[0]
    n_cols      = weights.shape[1]

    top_indices = np.empty((n_rows, k), dtype=np.int32)
    top_weights = np.empty((n_rows, k), dtype=np.float64)

    for i in prange(n_rows):
        # Initialize with first k elements
        for j in range(k):
            if j < n_cols:
                top_indices[i, j] = j
                top_weights[i, j] = weights[i, j]
            else:
                top_indices[i, j] = -1
                top_weights[i, j] = 0.0

        # Find minimum in current top k
        min_idx = 0
        min_val = top_weights[i, 0]
        for j in range(1, k):
            if top_weights[i, j] < min_val:
                min_val = top_weights[i, j]
                min_idx = j

        # Check remaining elements
        for j in range(k, n_cols):
            if weights[i, j] > min_val:
                # Replace minimum
                top_indices[i, min_idx] = j
                top_weights[i, min_idx] = weights[i, j]

                # Find new minimum
                min_val = top_weights[i, 0]
                min_idx = 0
                for jj in range(1, k):
                    if top_weights[i, jj] < min_val:
                        min_val = top_weights[i, jj]
                        min_idx = jj

    return top_indices, top_weights


# --------------------------------------------------------------------------- #
#                                 prune                                       #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _prune_weights_parallel(
    weights:   np.ndarray,
    min_value: float,
) -> np.ndarray:
    """
    Prune weights below min_value and ensure at least one weight per row.

    Fuses argmax, threshold masking, and fallback assignment.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M), float64.
    min_value : float
        Minimum weight value to keep.

    Returns
    -------
    np.ndarray
        Pruned weight matrix (copy), float64.
    """
    n_rows = weights.shape[0]
    n_cols = weights.shape[1]
    result = np.empty_like(weights)

    for i in prange(n_rows):
        # Find max index and check if any weight survives pruning
        max_idx  = 0
        max_val  = weights[i, 0]
        any_kept = False

        for j in range(n_cols):
            val = weights[i, j]
            if val > max_val:
                max_val = val
                max_idx = j

            if val >= min_value:
                result[i, j] = val
                if val != 0.0:
                    any_kept = True
            else:
                result[i, j] = 0.0

        # If all weights were pruned, restore the max weight to 1.0
        if not any_kept:
            result[i, max_idx] = 1.0

    return result


# --------------------------------------------------------------------------- #
#                          set_max_influences                                 #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _set_max_influences_parallel(
    weights:   np.ndarray,
    max_count: int,
) -> np.ndarray:
    """
    Zero out all but the top max_count weights per row.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M), float64.
    max_count : int
        Maximum number of non-zero weights per row.

    Returns
    -------
    np.ndarray
        Modified weight matrix (copy), float64.
    """
    n_rows = weights.shape[0]
    n_cols = weights.shape[1]
    result = np.zeros_like(weights)

    if max_count >= n_cols:
        # No pruning needed
        for i in prange(n_rows):
            for j in range(n_cols):
                result[i, j] = weights[i, j]
        return result

    # Use a simple selection algorithm for top k
    for i in prange(n_rows):
        # Copy weights to temp array for this row
        temp_vals = np.empty(n_cols, dtype=np.float64)
        temp_idxs = np.empty(n_cols, dtype=np.int32)

        for j in range(n_cols):
            temp_vals[j] = weights[i, j]
            temp_idxs[j] = j

        # Partial sort to find top max_count (simple selection)
        for k in range(max_count):
            max_idx = k
            for j in range(k + 1, n_cols):
                if temp_vals[j] > temp_vals[max_idx]:
                    max_idx = j

            # Swap
            if max_idx != k:
                temp_vals[k], temp_vals[max_idx] = temp_vals[max_idx], temp_vals[k]
                temp_idxs[k], temp_idxs[max_idx] = temp_idxs[max_idx], temp_idxs[k]

        # Copy top k to result
        for k in range(max_count):
            j = temp_idxs[k]
            result[i, j] = weights[i, j]

    return result


# --------------------------------------------------------------------------- #
#                              counts                                         #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _count_nonzero_per_row(
    weights: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Count non-zero weights per row (weights > epsilon).

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M), float64.
    epsilon : float
        Tolerance for considering a weight as zero.

    Returns
    -------
    np.ndarray
        Count per row, shape (N,), int32.
    """
    n_rows = weights.shape[0]
    n_cols = weights.shape[1]
    counts = np.empty(n_rows, dtype=np.int32)

    for i in prange(n_rows):
        count = 0
        for j in range(n_cols):
            if np.abs(weights[i, j]) > epsilon:
                count += 1
        counts[i] = count

    return counts


# --------------------------------------------------------------------------- #
#                         symmetry comparisons                                #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compare_symmetric_weights(
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """
    Compare two weight arrays for symmetry, return indices where they differ.

    Parameters
    ----------
    weights_a : np.ndarray
        First weight matrix, shape (N, K).
    weights_b : np.ndarray
        Second weight matrix, shape (N, K).
    tolerance : float
        Tolerance for comparison.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates row differs, shape (N,).
    """
    n_rows  = weights_a.shape[0]
    n_cols  = weights_a.shape[1]
    differs = np.zeros(n_rows, dtype=np.bool_)

    for i in prange(n_rows):
        for j in range(n_cols):
            diff = np.abs(weights_a[i, j] - weights_b[i, j])
            if diff > tolerance:
                differs[i] = True
                break

    return differs


@njit(parallel=True, fastmath=True, cache=True)
def _mirror_weights_parallel(
    weights:        np.ndarray,
    mirror_indices: np.ndarray,
    src_indices:    np.ndarray,
    sym_indices:    np.ndarray,
) -> np.ndarray:
    """
    Mirror weights from source to destination using symmetry mapping.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M), float64.
    mirror_indices : np.ndarray
        Influence mirror map, shape (M,), int32.
    src_indices : np.ndarray
        Source vertex indices to copy from, shape (K,), int32.
    sym_indices : np.ndarray
        Symmetric vertex mapping, shape (N,), int32.

    Returns
    -------
    np.ndarray
        Mirrored weight matrix (copy), float64.
    """
    result = weights.copy()
    n_cols = weights.shape[1]
    n_src  = src_indices.shape[0]

    for ii in prange(n_src):
        src = src_indices[ii]
        dst = sym_indices[src]

        # Copy mirrored weights
        for j in range(n_cols):
            mirror_j = mirror_indices[j]
            result[dst, j] = weights[src, mirror_j]

    return result


# --------------------------------------------------------------------------- #
#                          Public wrapper functions                           #
# --------------------------------------------------------------------------- #


def round_normalize_fast(weights: np.ndarray, n: int) -> np.ndarray:
    """
    Fast version of _round_normalize using parallel Numba kernel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix to round and normalize.
    n : int
        Number of decimal places.

    Returns
    -------
    np.ndarray
        Rounded and normalized weights.
    """
    # normalize runs in place, so copy rather than alias the caller's array
    weights = np.array(weights, dtype=np.float64)
    normalize_fast(weights)
    return _round_normalize_parallel(weights, n)


def normalize_fast(
    weights:           np.ndarray,
    locked_influences: np.ndarray | list | None = None,
    indices:           np.ndarray | list | None = None,
) -> None:
    """
    Fast in-place normalization using parallel Numba kernel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix to normalize (modified in-place).
    locked_influences : array-like, optional
        Column indices to exclude from normalization.
    indices : array-like, optional
        Row indices to normalize. If None, normalizes all rows.
    """
    n_rows, n_cols = weights.shape

    # Handle indices (use int32 for consistency with other functions and efficiency)
    if indices is None or len(indices) == 0:
        indices = np.arange(n_rows, dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)

    # Build include mask
    include_mask = np.ones(n_cols, dtype=np.bool_)
    if locked_influences is not None and len(locked_influences) > 0:
        include_mask[locked_influences] = False

    _normalize_weights_parallel(weights, indices, include_mask)


def scatter_to_dense_fast(
    influence_indices: np.ndarray,
    weights_flat:      np.ndarray,
    n_verts:           int,
    n_influences:      int,
    max_influences:    int,
) -> np.ndarray:
    """
    Fast scatter operation for CompactSkinData.to_skin_data().

    Parameters
    ----------
    influence_indices : np.ndarray
        Flat influence indices.
    weights_flat : np.ndarray
        Flat weights array.
    n_verts : int
        Number of vertices.
    n_influences : int
        Number of influences (output columns).
    max_influences : int
        Max influences per vertex.

    Returns
    -------
    np.ndarray
        Dense weight matrix.
    """
    influence_indices = np.asarray(influence_indices, dtype=np.int32)
    weights_flat      = np.asarray(weights_flat, dtype=np.float64)
    return _scatter_weights_to_dense(
        influence_indices, weights_flat, n_verts, n_influences, max_influences
    )


def prune_fast(
    weights:   np.ndarray,
    min_value: float      = 0.0,
    normalize: bool       = True,
) -> np.ndarray:
    """
    Fast pruning using parallel Numba kernel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix to prune.
    min_value : float
        Minimum weight to keep.
    normalize : bool
        Whether to normalize after pruning.

    Returns
    -------
    np.ndarray
        Pruned (and optionally normalized) weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    result  = _prune_weights_parallel(weights, min_value)

    if normalize:
        normalize_fast(result)

    return result


def set_max_influences_fast(
    weights:   np.ndarray,
    max_count: int,
    normalize: bool       = True,
) -> np.ndarray:
    """
    Fast max influence limiting using parallel Numba kernel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix.
    max_count : int
        Maximum influences per vertex.
    normalize : bool
        Whether to normalize after limiting.

    Returns
    -------
    np.ndarray
        Modified weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    result  = _set_max_influences_parallel(weights, max_count)

    if normalize:
        normalize_fast(result)

    return result


def count_nonzero_fast(weights: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Fast non-zero count per row using parallel Numba kernel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix.
    epsilon : float
        Tolerance for zero.

    Returns
    -------
    np.ndarray
        Count per row.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return _count_nonzero_per_row(weights, epsilon)


def gather_top_k_fast(weights: np.ndarray, k: int) -> tuple:
    """
    Fast top-k gathering using parallel Numba kernel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix.
    k : int
        Number of top weights to gather.

    Returns
    -------
    tuple
        (indices, weights) of top k per row.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return _gather_top_k_weights(weights, k)


def compare_symmetric_weights_fast(
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    tolerance: float      = 1e-7,
) -> np.ndarray:
    """
    Fast symmetric weight comparison using parallel Numba kernel.

    Parameters
    ----------
    weights_a : np.ndarray
        First weight matrix, shape (N, K).
    weights_b : np.ndarray
        Second weight matrix, shape (N, K).
    tolerance : float
        Tolerance for comparison.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates row differs, shape (N,).
    """
    weights_a = np.asarray(weights_a, dtype=np.float64)
    weights_b = np.asarray(weights_b, dtype=np.float64)
    return _compare_symmetric_weights(weights_a, weights_b, tolerance)


def mirror_weights_fast(
    weights:        np.ndarray,
    mirror_indices: np.ndarray,
    src_indices:    np.ndarray,
    sym_indices:    np.ndarray,
) -> np.ndarray:
    """
    Fast weight mirroring using parallel Numba kernel.

    Copies mirrored weights from source vertices to their symmetric
    destinations. Requires sym_indices to be an involution
    (sym_indices[sym_indices[i]] == i).

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape (N, M), float64.
    mirror_indices : np.ndarray
        Influence mirror map, shape (M,).
    src_indices : np.ndarray
        Source vertex indices to copy from.
    sym_indices : np.ndarray
        Symmetric vertex mapping, shape (N,).

    Returns
    -------
    np.ndarray
        Mirrored weight matrix (copy).
    """
    weights        = np.asarray(weights,        dtype=np.float64)
    mirror_indices = np.asarray(mirror_indices, dtype=np.int32)
    src_indices    = np.asarray(src_indices,    dtype=np.int32)
    sym_indices    = np.asarray(sym_indices,    dtype=np.int32)
    return _mirror_weights_parallel(weights, mirror_indices, src_indices, sym_indices)