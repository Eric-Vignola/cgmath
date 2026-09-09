import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_index_lookup(matrix, indices):
    """
    for each element of indices will return the first found
    row and column indices in the matrix
    """
    faces = np.empty(indices.size, dtype=indices.dtype)
    idx   = np.empty(indices.size, dtype=indices.dtype)

    for i in prange(indices.size):
        found = False
        for j in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                if matrix[j, k] == indices[i]:
                    found = True
                    break

            if found:
                faces[i] = j
                idx[i]   = k
                break

    return faces, idx


@njit(fastmath=True, cache=True)
def _matrix_row_combine(matrix, from_rows, to_rows):
    """
    creates a new resized matrix where elements of `from_rows`
    are uniquely appended to destinaton elements `to_rows`
    """
    if matrix is None or not matrix.shape[0]:
        return matrix

    # make an educated guess as to how wide an array this will produce
    counts   = np.sum(matrix >= 0, axis=1)
    offsets  = np.copy(counts)
    maximum_ = np.copy(counts)
    for i in range(to_rows.size):
        if to_rows[i] != from_rows[i]:
            maximum_[to_rows[i]] += maximum_[from_rows[i]]

    # create the widest possible merged matrix
    maximum = maximum_.max()

    merged                       = np.full((matrix.shape[0], maximum), -1, dtype=matrix.dtype)
    merged[:, : matrix.shape[1]] = matrix

    # insert any new value at the next available index
    for i in range(from_rows.size):
        # skip identical rows
        if from_rows[i] != to_rows[i]:
            # for elements of the source row
            for j in range(counts[from_rows[i]]):
                unique = True
                for k in range(offsets[to_rows[i]]):
                    if matrix[from_rows[i], j] == merged[to_rows[i], k]:
                        unique = False
                        break

                # append unique value
                if unique:
                    merged[to_rows[i], offsets[to_rows[i]]] = matrix[from_rows[i], j]
                    offsets[to_rows[i]] += 1

    # clip off the extra fat and return
    return merged[:, : offsets.max()]


@njit(parallel=True, fastmath=True, cache=True)
def _shared_edges_test(vals):
    shared = np.empty(vals.shape[0], dtype=np.bool_)

    for i in prange(vals.shape[0]):
        shared[i] = True

        for j in range(1, vals.shape[1]):
            if vals[i, j] > -1:
                if vals[i, j - 1] > -1:
                    if vals[i, j] == vals[i, j - 1]:
                        shared[i] = False
                        break

    return shared


@njit(fastmath=True, parallel=True, cache=True)
def _compute_e2v_e2f(f2v, counts):
    """builds edges_to_vertices and edges_to_faces components"""

    # build unoptimized e2v and e2f matrices
    n_counts = np.sum(counts)
    cumsum   = np.cumsum(counts)
    cumsum   = np.append(cumsum[::-1], 0)[::-1][:-1]

    e2v = np.empty((n_counts, 2), dtype=f2v.dtype)
    e2f = np.empty((n_counts, 1), dtype=f2v.dtype)

    for i in prange(len(counts)):
        for j in range(counts[i]):
            k  = cumsum[i] + j
            k_ = cumsum[i] + ((j + 1) % counts[i])

            e2v[k, 0] = f2v[k]
            e2v[k, 1] = f2v[k_]
            e2f[k, 0] = i

    return e2v, e2f


@njit(fastmath=True, cache=True)
def _compute_v2x(f2v, shape):
    """general purpose vertex_to_x builder"""

    v2x = np.ones((shape[0], shape[1]), dtype=f2v.dtype)
    v2x = np.negative(v2x)
    for i in range(f2v.shape[0]):
        for j in range(f2v.shape[1]):
            if f2v[i, j] == -1:
                break
            for k in range(shape[1]):
                if v2x[f2v[i, j], k] == -1:
                    v2x[f2v[i, j], k] = i
                    break

    return v2x


@njit(fastmath=True, parallel=True, cache=True)
def _grow_neighbors(neighbors: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """
    Numba parallel with adaptive buffer sizing.
    Based on the working double-buffer approach.

    Args:
        neighbors: Connectivity matrix (num_verts, width) with -1 as sentinel
        indices: Array of vertex indices to process (np.intp)
        n: Number of grow iterations
    """
    num_verts      = neighbors.shape[0]
    original_width = neighbors.shape[1]
    num_indices    = indices.shape[0]

    # For iteration 0, estimate buffer size from original connectivity
    # Each vertex can gain at most (original_width * original_width) new neighbors
    # But cap at num_verts - 1
    initial_estimate = min(
        original_width * original_width + original_width, num_verts - 1
    )

    buffer_a = np.full((num_indices, initial_estimate), -1, dtype=np.int32)
    buffer_b = np.full((num_indices, initial_estimate), -1, dtype=np.int32)

    # Copy initial connectivity to buffer_a for specified indices only
    for i in prange(num_indices):
        v = indices[i]
        for j in range(original_width):
            if neighbors[v, j] == -1:
                break
            buffer_a[i, j] = neighbors[v, j]

    # Track actual max width used
    current_max_width = original_width
    prev_total_count  = 0  # For early termination detection

    converged      = False
    last_iteration = 0

    for iteration in range(n):
        # Buffer selection - always do this to keep variables defined
        if iteration % 2 == 0:
            current_buf = buffer_a
            next_buf    = buffer_b
        else:
            current_buf = buffer_b
            next_buf    = buffer_a

        # Buffer resize - only when not converged
        if not converged:
            # Check if we need larger buffers
            # Estimate: current neighbors + their original neighbors
            needed_width = current_max_width + current_max_width * original_width
            needed_width = min(needed_width, num_verts - 1)

            if needed_width > next_buf.shape[1]:
                # Reallocate larger buffers
                new_size = min(needed_width * 2, num_verts - 1)
                if iteration % 2 == 0:
                    buffer_b = np.full((num_indices, new_size), -1, dtype=np.int32)
                    next_buf = buffer_b
                else:
                    buffer_a = np.full((num_indices, new_size), -1, dtype=np.int32)
                    next_buf = buffer_a

        next_width    = next_buf.shape[1]
        current_width = current_buf.shape[1]

        # Track new max width for this iteration
        counts = np.zeros(num_indices, dtype=np.int32)

        # Combined reset and work loop - prange must be at top level (not inside if)
        # Move converged check inside the loop to avoid Numba parfor analysis issues
        for i in prange(num_indices):
            if converged:
                continue

            # Reset this row of next_buf
            for j in range(next_width):
                next_buf[i, j] = -1

            v       = indices[i]
            seen    = np.zeros(num_verts, dtype=np.bool_)
            seen[v] = True
            count   = 0

            # Copy existing neighbors
            for j in range(current_width):
                neighbor = current_buf[i, j]
                if neighbor == -1:
                    break
                seen[neighbor] = True
                if count < next_width:
                    next_buf[i, count] = neighbor
                    count += 1

            # Add original neighbors of current neighbors
            for j in range(current_width):
                neighbor = current_buf[i, j]
                if neighbor == -1:
                    break
                for k in range(original_width):
                    neighbor2 = neighbors[neighbor, k]
                    if neighbor2 == -1:
                        break
                    if not seen[neighbor2]:
                        seen[neighbor2] = True
                        if count < next_width:
                            next_buf[i, count] = neighbor2
                            count += 1

            counts[i] = count

        if not converged:
            last_iteration    = iteration
            current_max_width = np.max(counts)

            # Early termination: if total neighbor count didn't change, we've converged
            total_count = np.sum(counts)
            if total_count == prev_total_count:
                converged = True
            prev_total_count = total_count

    # Get final result (use last_iteration+1 to know which buffer has the latest data)
    if (last_iteration + 1) % 2 == 1:
        result = buffer_b
    else:
        result = buffer_a

    # Find actual max width used
    max_used = 0
    for i in range(num_indices):
        for j in range(result.shape[1]):
            if result[i, j] == -1:
                break
            if j + 1 > max_used:
                max_used = j + 1

    return result[:, : max(max_used, 1)]


@njit(parallel=True, fastmath=True, cache=True)
def _compute_neighbors(v2x, x2v):
    """builds vertex neighbors given v2x and x2v matrices"""

    max_neighbors = v2x.shape[1] * x2v.shape[1]
    neighbors_    = np.ones((v2x.shape[0], max_neighbors), dtype=v2x.dtype) * -1
    total         = np.zeros(v2x.shape[0], dtype=v2x.dtype)

    # for each vertex
    for i in prange(v2x.shape[0]):
        # for each face connected to vertex
        for j in range(v2x.shape[1]):
            if v2x[i, j] == -1:
                break

            # for each vertex connected to face
            for k in range(x2v.shape[1]):
                if x2v[v2x[i, j], k] != i:
                    # insert the new neighbor only if not already in
                    for jj in range(max_neighbors):
                        if neighbors_[i, jj] == x2v[v2x[i, j], k]:
                            break
                        elif neighbors_[i, jj] == -1:
                            neighbors_[i, jj] = x2v[v2x[i, j], k]
                            total[i] += 1
                            break

    # matrix will be too large, clip it to the longest axis
    max_total = total.max()
    neighbors = np.empty((v2x.shape[0], max_total), dtype=v2x.dtype) * -1

    for i in prange(v2x.shape[0]):
        for j in range(max_total):
            neighbors[i, j] = neighbors_[i, j]

    return neighbors


@njit(cache=True)
def _quad_match_greedy(
    score:   np.ndarray,
    fa:      np.ndarray,
    fb:      np.ndarray,
    n_faces: int,
) -> np.ndarray:
    """Greedy maximum-weight matching on the triangle dual graph.

    Picks edges in descending score order, marking each incident face as
    used after a pick so it cannot be re-used.  Stops at the first
    non-finite score.

    NOT marked ``parallel=True`` -- each iteration depends on the
    cumulative ``used`` mask, so the loop is intentionally sequential.

    Args:
        score: ``(n_cand,)`` per-candidate quality (use ``-inf`` to skip).
        fa: ``(n_cand,)`` first incident triangle id.
        fb: ``(n_cand,)`` second incident triangle id.
        n_faces: total number of faces (size of the ``used`` mask).

    Returns:
        ``(n_cand,) bool`` -- True where the candidate edge was chosen.
    """
    order = np.argsort(-score)
    used  = np.zeros(n_faces, dtype=np.bool_)
    pick  = np.zeros(score.size, dtype=np.bool_)
    for k_idx in range(order.size):
        k = order[k_idx]
        if not np.isfinite(score[k]):
            break
        a = fa[k]
        b = fb[k]
        if not used[a] and not used[b]:
            used[a] = True
            used[b] = True
            pick[k] = True
    return pick