"""
Optimized Numba implementations for blur and inpaint algorithms.

These functions provide parallel, JIT-compiled implementations for
iterative Laplacian blur and inpaint operations on vertex data.

Key optimizations:
- Per-vertex delta buffers (eliminates race conditions)
- Fully parallelized main computation loops
- Pre-computed inverse values to avoid division in loops
- Binary search for membership testing in sorted arrays
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                        Binary search helper                                 #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, cache=True)
def _binary_search(sorted_arr, value):
    """
    Binary search for value in sorted array.

    Returns index if found, -1 otherwise.
    """
    left  = 0
    right = sorted_arr.shape[0] - 1

    while left <= right:
        mid     = (left + right) >> 1
        mid_val = sorted_arr[mid]

        if mid_val == value:
            return mid
        elif mid_val < value:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# --------------------------------------------------------------------------- #
#                     _balance_center_weights                                 #
# --------------------------------------------------------------------------- #


@njit(parallel=False, fastmath=True, cache=True)
def _balance_center_weights(weights, positive, negative, center, max_inf):
    """
    Balances center weights to gracefully meet max influences.

    Original sequential implementation kept for compatibility.
    For parallel version, use _balance_center_weights_parallel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix (N, M), float64. Modified in-place.
    positive : np.ndarray
        Positive side influence indices.
    negative : np.ndarray
        Negative side influence indices.
    center : np.ndarray
        Center influence indices.
    max_inf : int
        Maximum allowed influences per vertex.
    """
    for i in range(weights.shape[0]):
        count = 0
        for j in range(weights.shape[1]):
            if weights[i, j] > 0.0:
                count += 1

        while count > max_inf:
            count = 0
            index = -1
            w     = 1.0

            for j in range(weights.shape[1]):
                if weights[i, j] > 0.0:
                    count += 1
                    if weights[i, j] < w:
                        index = j
                        w     = weights[i, j]

            found = False
            for j in range(center.size):
                if index == center[j]:
                    weights[i, center[j]] = 0.0
                    found                 = True
                    count -= 1
                    break

            if not found and count > 2:
                for j in range(positive.size):
                    if index == positive[j] or index == negative[j]:
                        weights[i, positive[j]] = 0.0
                        weights[i, negative[j]] = 0.0
                        found                   = True
                        count -= 2
                        break

            if not found:
                break


@njit(parallel=True, fastmath=True, cache=True)
def _balance_center_weights_parallel(weights, positive, negative, center, max_inf):
    """
    Parallelized version of _balance_center_weights.

    Each vertex is processed independently in parallel.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix (N, M), float64. Modified in-place.
    positive : np.ndarray
        Positive side influence indices.
    negative : np.ndarray
        Negative side influence indices.
    center : np.ndarray
        Center influence indices.
    max_inf : int
        Maximum allowed influences per vertex.
    """
    n_verts = weights.shape[0]
    n_cols  = weights.shape[1]

    # Pre-sort for faster lookup
    center_sorted   = np.sort(center)
    positive_sorted = np.sort(positive)
    negative_sorted = np.sort(negative)

    for i in prange(n_verts):
        # Count non-zero weights
        count = 0
        for j in range(n_cols):
            if weights[i, j] > 0.0:
                count += 1

        # While count exceeds max_inf
        while count > max_inf:
            count = 0
            index = -1
            w     = 1.0

            # Find index with lowest weight
            for j in range(n_cols):
                if weights[i, j] > 0.0:
                    count += 1
                    if weights[i, j] < w:
                        index = j
                        w     = weights[i, j]

            # Check if this is a center index (binary search)
            found = False
            if _binary_search(center_sorted, index) >= 0:
                weights[i, index] = 0.0
                found             = True
                count -= 1

            # If not found, look for positive/negative pair
            if not found and count > 2:
                pos_idx = _binary_search(positive_sorted, index)
                if pos_idx >= 0:
                    # Found in positive - zero both
                    weights[i, positive[pos_idx]] = 0.0
                    weights[i, negative[pos_idx]] = 0.0
                    found                         = True
                    count -= 2
                else:
                    neg_idx = _binary_search(negative_sorted, index)
                    if neg_idx >= 0:
                        weights[i, positive[neg_idx]] = 0.0
                        weights[i, negative[neg_idx]] = 0.0
                        found                         = True
                        count -= 2

            if not found:
                break


# --------------------------------------------------------------------------- #
#                              _blur                                          #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _blur(values, neighbors, iterations, receptions, contributions):
    """
    Fully optimized iterative Laplacian blur algorithm.

    Additional optimizations over _blur:
    - Pre-count neighbors to avoid recomputation
    - Fused delta accumulation and change detection
    - Reduced memory traffic with local variables

    Parameters
    ----------
    values : np.ndarray
        Values to blur (N, M), float64.
    neighbors : np.ndarray
        Neighbor connectivity matrix (N, K), int32. -1 for padding.
    iterations : np.ndarray
        Per-vertex iteration count (N,), int32. Modified in-place.
    receptions : np.ndarray
        Per-vertex reception factor (N,), float64.
    contributions : np.ndarray
        Per-vertex contribution factor (N,), float64.

    Returns
    -------
    np.ndarray
        Blurred values (N, M), float64.
    """
    n_verts     = values.shape[0]
    n_vals      = values.shape[1]
    n_neighbors = neighbors.shape[1]

    # Pre-count valid neighbors per vertex
    neighbor_counts = np.zeros(n_verts, dtype=np.float64)
    for i in prange(n_verts):
        count = 0.0
        for j in range(n_neighbors):
            if neighbors[i, j] > -1:
                count += 1.0
        neighbor_counts[i] = count

    # Allocate buffers
    new_values  = values.copy()
    old_values  = np.empty_like(values)
    deltas      = np.zeros((n_verts, n_vals), dtype=values.dtype)
    has_changed = np.ones(n_verts, dtype=np.bool_)

    # Iterate until no changes
    while np.any(has_changed):
        # Swap buffers (parallel)
        for i in prange(n_verts):
            has_changed[i] = False
            for j in range(n_vals):
                old_values[i, j] = new_values[i, j]

        # Main blur computation (FULLY PARALLEL)
        for i in prange(n_verts):
            iter_count = iterations[i]
            if iter_count == 0:
                continue

            count = neighbor_counts[i]
            if count == 0.0:
                continue

            local_changed = False

            # Reset and accumulate delta
            for k in range(n_vals):
                deltas[i, k] = 0.0

            for j in range(n_neighbors):
                ni = neighbors[i, j]
                if ni > -1:
                    contrib = contributions[ni]
                    for k in range(n_vals):
                        diff = old_values[ni, k] - new_values[i, k]
                        deltas[i, k] += diff * contrib
                        if diff != 0.0:
                            local_changed = True

            has_changed[i] = local_changed

            # Apply delta
            if local_changed:
                inv_count = 1.0 / count
                reception = receptions[i]
                for k in range(n_vals):
                    new_values[i, k] += deltas[i, k] * inv_count * reception

            # Update iteration count
            if iter_count > 0:
                iterations[i] = iter_count - 1
            elif local_changed:
                iterations[i] = iter_count + 1

    return new_values


# --------------------------------------------------------------------------- #
#                              _inpaint                                       #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _inpaint(inpaint, values, neighbors, iterations, receptions, contributions):
    """
    Fully optimized iterative Laplacian inpaint algorithm.

    Additional optimizations over _inpaint:
    - Pre-count valid inpainted neighbors
    - Reduced redundant condition checks
    - Better memory access patterns

    Parameters
    ----------
    inpaint : np.ndarray
        Inpaint tags (N,), int32. 0=needs inpaint, 1=already done.
    values : np.ndarray
        Values to inpaint (N, M), float64.
    neighbors : np.ndarray
        Neighbor connectivity matrix (N, K), int32. -1 for padding.
    iterations : np.ndarray
        Per-vertex iteration count (N,), int32. Modified in-place.
    receptions : np.ndarray
        Per-vertex reception factor (N,), float64.
    contributions : np.ndarray
        Per-vertex contribution factor (N,), float64.

    Returns
    -------
    np.ndarray
        Inpainted values (N, M), float64.
    """
    n_verts     = values.shape[0]
    n_vals      = values.shape[1]
    n_neighbors = neighbors.shape[1]

    # Allocate buffers
    old_values   = np.empty_like(values)
    new_values   = values.copy()
    old_inpaints = np.empty_like(inpaint)
    new_inpaints = inpaint.copy()
    deltas       = np.zeros((n_verts, n_vals), dtype=values.dtype)
    progressed   = np.zeros(n_verts, dtype=np.bool_)

    # Iterate until all iterations complete
    max_iter = iterations.max()
    while max_iter > 0:
        # Swap buffers (parallel)
        for i in prange(n_verts):
            progressed[i]   = False
            old_inpaints[i] = new_inpaints[i]
            for j in range(n_vals):
                old_values[i, j] = new_values[i, j]

        # Reset max_iter for next check
        max_iter = 0

        # Main inpaint computation (FULLY PARALLEL)
        for i in prange(n_verts):
            iter_i = iterations[i]
            if iter_i <= 0:
                continue

            sample_count = 0.0
            is_initial   = new_inpaints[i] == 0

            # Reset delta
            for k in range(n_vals):
                deltas[i, k] = 0.0

            # Sample neighbors
            for j in range(n_neighbors):
                ni = neighbors[i, j]
                if ni > -1 and old_inpaints[ni] == 1:
                    sample_count += 1.0

                    if is_initial:
                        for k in range(n_vals):
                            deltas[i, k] += old_values[ni, k]
                    else:
                        contrib = contributions[ni]
                        for k in range(n_vals):
                            deltas[i, k] += (
                                old_values[ni, k] - new_values[i, k]
                            ) * contrib

            if sample_count > 0.0:
                progressed[i] = True
                inv_count     = 1.0 / sample_count

                if is_initial:
                    for k in range(n_vals):
                        new_values[i, k] = deltas[i, k] * inv_count
                    new_inpaints[i] = 1
                else:
                    iterations[i] = iter_i - 1
                    reception     = receptions[i]
                    for k in range(n_vals):
                        new_values[i, k] += deltas[i, k] * inv_count * reception

            # Track max remaining iterations. This must stay in the
            # max(a, b) form: numba does not recognise a conditional
            # assignment as a parallel reduction, so an 'if x > max_iter'
            # here silently leaves max_iter at 0 and the loop runs once.
            max_iter = max(max_iter, iterations[i])

        # A vertex only decrements once a neighbour is inpainted, so a
        # component made entirely of holes keeps max_iter pinned above zero
        # forever. No progress in a whole pass means no progress is possible.
        if not np.any(progressed):
            break

    return new_values