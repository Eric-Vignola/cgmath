"""
Topological Neighborhood Computation (Dijkstra's algorithm)

Computes the topological neighborhood of mesh vertices within N hops
using Numba-accelerated processing.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _compute_single_vertex_neighborhood(
    v, connectivity, neighbor_counts, num_hops, max_output, num_vertices
):
    """Compute neighborhood for a single vertex using BFS."""
    visited    = np.zeros(num_vertices, dtype=np.bool_)
    visited[v] = True  # Exclude self

    result        = np.full(max_output, np.int32(-1), dtype=np.int32)
    frontier      = np.empty(max_output, dtype=np.int32)
    next_frontier = np.empty(max_output, dtype=np.int32)

    frontier_size = np.int32(0)
    output_idx    = np.int32(0)

    # Hop 0: direct neighbors
    nc = neighbor_counts[v]
    for j in range(nc):
        n = connectivity[v, j]
        if not visited[n]:
            visited[n] = True
            if output_idx < max_output:
                result[output_idx] = n
                output_idx += 1
            if frontier_size < max_output:
                frontier[frontier_size] = n
                frontier_size += 1

    # Remaining hops
    for _ in range(1, num_hops):
        if frontier_size == 0:
            break

        next_size = np.int32(0)

        for f in range(frontier_size):
            fv   = frontier[f]
            nc_f = neighbor_counts[fv]

            for j in range(nc_f):
                n = connectivity[fv, j]
                if not visited[n]:
                    visited[n] = True
                    if output_idx < max_output:
                        result[output_idx] = n
                        output_idx += 1
                    if next_size < max_output:
                        next_frontier[next_size] = n
                        next_size += 1

        # Swap by copying
        for i in range(next_size):
            frontier[i] = next_frontier[i]
        frontier_size = next_size

    return result, output_idx


@njit(cache=True)
def _count_neighborhood_size(v, connectivity, neighbor_counts, num_hops, num_vertices):
    """Count neighborhood size for a vertex without storing results (for estimation)."""
    visited    = np.zeros(num_vertices, dtype=np.bool_)
    visited[v] = True

    frontier      = np.empty(num_vertices, dtype=np.int32)
    next_frontier = np.empty(num_vertices, dtype=np.int32)

    frontier_size = np.int32(0)
    count         = np.int32(0)

    nc = neighbor_counts[v]
    for j in range(nc):
        n = connectivity[v, j]
        if not visited[n]:
            visited[n] = True
            count += 1
            frontier[frontier_size] = n
            frontier_size += 1

    for _ in range(1, num_hops):
        if frontier_size == 0:
            break

        next_size = np.int32(0)
        for f in range(frontier_size):
            fv   = frontier[f]
            nc_f = neighbor_counts[fv]
            for j in range(nc_f):
                n = connectivity[fv, j]
                if not visited[n]:
                    visited[n] = True
                    count += 1
                    next_frontier[next_size] = n
                    next_size += 1

        for i in range(next_size):
            frontier[i] = next_frontier[i]
        frontier_size = next_size

    return count


@njit(cache=True)
def _find_top_connected_vertices(neighbor_counts, num_vertices, top_k):
    """Find indices of vertices with highest neighbor counts."""
    top_indices = np.empty(top_k, dtype=np.int32)
    top_counts  = np.zeros(top_k, dtype=np.int32)
    for i in range(num_vertices):
        nc = neighbor_counts[i]
        for j in range(top_k):
            if nc > top_counts[j]:
                for k in range(top_k - 1, j, -1):
                    top_counts[k]  = top_counts[k - 1]
                    top_indices[k] = top_indices[k - 1]
                top_counts[j]  = nc
                top_indices[j] = i
                break
    return top_indices


@njit(parallel=True, cache=True)
def _estimate_max_neighborhood(connectivity, neighbor_counts, num_hops, num_vertices):
    """Estimate max neighborhood size by sampling vertices in parallel."""
    # Sample evenly distributed vertices plus vertices with highest connectivity
    num_samples = min(64, num_vertices)
    step        = max(1, num_vertices // num_samples)

    # Find top connected vertices (serial - fast O(n) scan)
    top_k       = np.int32(16)
    top_indices = _find_top_connected_vertices(neighbor_counts, num_vertices, top_k)

    # Total samples: evenly distributed + top connected
    total_samples = num_samples + min(top_k, num_vertices)
    sample_sizes  = np.zeros(total_samples, dtype=np.int32)

    # Sample all vertices in parallel
    for i in prange(total_samples):
        if i < num_samples:
            # Evenly distributed sample
            v = i * step
            if v < num_vertices:
                sample_sizes[i] = _count_neighborhood_size(
                    v, connectivity, neighbor_counts, num_hops, num_vertices
                )
        else:
            # Top connected vertex sample
            idx = i - num_samples
            if idx < min(top_k, num_vertices):
                v = top_indices[idx]
                sample_sizes[i] = _count_neighborhood_size(
                    v, connectivity, neighbor_counts, num_hops, num_vertices
                )

    # Find max from all samples
    max_size = np.int32(0)
    for i in range(total_samples):
        if sample_sizes[i] > max_size:
            max_size = sample_sizes[i]

    # Add 50% buffer to handle variance, cap at num_vertices - 1
    estimated = np.int32(min(max_size * 1.5 + 50, num_vertices - 1))
    return estimated


@njit(cache=True)
def _compute_single_vertex_neighborhood_with_distances(
    v,
    connectivity,
    distances,
    neighbor_counts,
    num_hops,
    max_output,
    num_vertices,
    max_distance,
):
    """Compute neighborhood with distance tracking for a single vertex using BFS.

    Uses best-distance tracking to update distances when shorter paths are found,
    without re-exploring from updated vertices (to maintain BFS performance).

    The visited array stores: -1 = unvisited, >= 0 = result index where vertex is stored.
    This allows O(1) lookup for distance updates.
    """
    # visited stores result index (-1 = unvisited, >= 0 = index in result array)
    visited       = np.full(num_vertices, np.int32(-1), dtype=np.int32)
    visited[v]    = -3  # Mark source as visited (special marker, never add to results)

    result        = np.full(max_output, np.int32(-1), dtype=np.int32)
    result_dists  = np.full(max_output, np.float32(-1.0), dtype=np.float32)

    frontier      = np.empty(max_output, dtype=np.int32)
    next_frontier = np.empty(max_output, dtype=np.int32)

    frontier_size = np.int32(0)
    output_idx    = np.int32(0)

    # Hop 0: direct neighbors
    nc = neighbor_counts[v]
    for j in range(nc):
        n = connectivity[v, j]
        d = distances[v, j]

        if visited[n] == -1:
            # First visit - only add if within max_distance
            if d <= max_distance:
                if output_idx < max_output:
                    result[output_idx]       = n
                    result_dists[output_idx] = d
                    visited[n]               = output_idx  # Store result index
                    output_idx += 1
                    if frontier_size < max_output:
                        frontier[frontier_size] = n
                        frontier_size += 1
                else:
                    visited[n] = -2  # Mark visited but not in results (full)
            else:
                visited[n] = -2  # Mark visited but excluded by max_distance
        elif visited[n] >= 0 and d < result_dists[visited[n]]:
            # Already in results - update if shorter
            result_dists[visited[n]] = d

    # Remaining hops
    for _ in range(1, num_hops):
        if frontier_size == 0:
            break

        next_size = np.int32(0)

        for f in range(frontier_size):
            fv      = frontier[f]
            fv_dist = result_dists[visited[fv]]
            nc_f    = neighbor_counts[fv]

            for j in range(nc_f):
                n        = connectivity[fv, j]
                new_dist = fv_dist + distances[fv, j]

                if visited[n] == -1 or visited[n] == -2:
                    # First visit OR previously excluded - add if within max_distance
                    if new_dist <= max_distance:
                        if output_idx < max_output:
                            result[output_idx]       = n
                            result_dists[output_idx] = new_dist
                            visited[n]               = output_idx  # Store result index
                            output_idx += 1
                            if next_size < max_output:
                                next_frontier[next_size] = n
                                next_size += 1
                        else:
                            visited[n] = -2  # Mark visited but not in results
                    else:
                        visited[n] = -2  # Mark visited but excluded
                elif visited[n] >= 0 and new_dist < result_dists[visited[n]]:
                    # Already in results - update if shorter
                    result_dists[visited[n]] = new_dist

        # Swap by copying
        for i in range(next_size):
            frontier[i] = next_frontier[i]
        frontier_size = next_size

    return result, result_dists, output_idx


@njit(parallel=True, cache=True)
def _compute_topological_neighborhood(connectivity, num_hops, indices):
    """
    JIT-compiled implementation for computing topological neighborhoods.

    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity matrix of shape (n, m) with dtype np.int32.
    num_hops : int
        Number of hops to grow the neighborhood.
    indices : np.ndarray
        Vertex indices to compute neighborhoods for (dtype np.int32).

    Returns
    -------
    np.ndarray
        Neighborhood matrix of shape (len(indices), max_neighbors).
    """
    num_vertices = np.int32(connectivity.shape[0])
    num_indices  = np.int32(len(indices))
    max_conn     = np.int32(connectivity.shape[1])

    # Pre-compute neighbor counts for all vertices
    neighbor_counts = np.empty(num_vertices, dtype=np.int32)
    for i in range(num_vertices):
        c = np.int32(0)
        for j in range(max_conn):
            if connectivity[i, j] >= 0:
                c += 1
        neighbor_counts[i] = c

    # Estimate max output size by sampling from the provided indices
    if num_indices <= 64:
        # For small index sets, sample all of them
        max_output = np.int32(0)
        for i in range(num_indices):
            v = indices[i]
            size = _count_neighborhood_size(
                v, connectivity, neighbor_counts, num_hops, num_vertices
            )
            if size > max_output:
                max_output = size
        max_output = np.int32(min(max_output + 50, num_vertices - 1))
    else:
        # For larger sets, use the standard estimation
        max_output = _estimate_max_neighborhood(
            connectivity, neighbor_counts, num_hops, num_vertices
        )

    neighborhood       = np.full((num_indices, max_output), np.int32(-1), dtype=np.int32)
    neighborhood_sizes = np.zeros(num_indices, dtype=np.int32)

    # Process vertices in parallel
    for i in prange(num_indices):
        v = indices[i]
        result, size = _compute_single_vertex_neighborhood(
            v, connectivity, neighbor_counts, num_hops, max_output, num_vertices
        )
        neighborhood[i, :]    = result
        neighborhood_sizes[i] = size

    max_used = np.int32(0)
    for i in range(num_indices):
        if neighborhood_sizes[i] > max_used:
            max_used = neighborhood_sizes[i]

    if max_used == 0:
        return np.empty((num_indices, 0), dtype=np.int32)

    return neighborhood[:, :max_used].copy()


@njit(parallel=True, cache=True)
def _compute_topological_neighborhood_with_distances(
    connectivity, distances, num_hops, indices, max_distance
):
    """
    JIT-compiled implementation with distance tracking.

    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity matrix of shape (n, m) with dtype np.int32.
    distances : np.ndarray
        Distance matrix of shape (n, m) with dtype np.float32.
    num_hops : int
        Number of hops to grow the neighborhood.
    indices : np.ndarray
        Vertex indices to compute neighborhoods for (dtype np.int32).
    max_distance : np.float32
        Maximum cumulative distance threshold. Neighbors beyond this
        distance are excluded from results entirely.

    Returns
    -------
    tuple of np.ndarray
        (neighborhood indices, neighborhood distances)
    """
    num_vertices = np.int32(connectivity.shape[0])
    num_indices  = np.int32(len(indices))
    max_conn     = np.int32(connectivity.shape[1])

    # Pre-compute neighbor counts for all vertices
    neighbor_counts = np.empty(num_vertices, dtype=np.int32)
    for i in range(num_vertices):
        c = np.int32(0)
        for j in range(max_conn):
            if connectivity[i, j] >= 0:
                c += 1
        neighbor_counts[i] = c

    # Estimate max output size by sampling from the provided indices
    if num_indices <= 64:
        max_output = np.int32(0)
        for i in range(num_indices):
            v = indices[i]
            size = _count_neighborhood_size(
                v, connectivity, neighbor_counts, num_hops, num_vertices
            )
            if size > max_output:
                max_output = size
        max_output = np.int32(min(max_output + 50, num_vertices - 1))
    else:
        max_output = _estimate_max_neighborhood(
            connectivity, neighbor_counts, num_hops, num_vertices
        )

    neighborhood = np.full((num_indices, max_output), np.int32(-1), dtype=np.int32)
    neighborhood_dists = np.full(
        (num_indices, max_output), np.float32(-1.0), dtype=np.float32
    )
    neighborhood_sizes = np.zeros(num_indices, dtype=np.int32)

    # Process vertices in parallel
    for i in prange(num_indices):
        v = indices[i]
        result, result_dists, size = _compute_single_vertex_neighborhood_with_distances(
            v,
            connectivity,
            distances,
            neighbor_counts,
            num_hops,
            max_output,
            num_vertices,
            max_distance,
        )
        neighborhood[i, :]       = result
        neighborhood_dists[i, :] = result_dists
        neighborhood_sizes[i]    = size

    max_used = np.int32(0)
    for i in range(num_indices):
        if neighborhood_sizes[i] > max_used:
            max_used = neighborhood_sizes[i]

    if max_used == 0:
        return (
            np.empty((num_indices, 0), dtype=np.int32),
            np.empty((num_indices, 0), dtype=np.float32),
        )

    return neighborhood[:, :max_used].copy(), neighborhood_dists[:, :max_used].copy()


@njit(parallel=True, cache=True)
def _convert_to_dense_format(
    indices, neighborhood, neighborhood_dists, num_indices, num_vertices
):
    """Convert sparse neighborhood format to dense distance matrix.

    Parameters
    ----------
    indices : np.ndarray
        Query vertex indices
    neighborhood : np.ndarray
        Sparse neighborhood indices (n, max_neighbors), padded with -1
    neighborhood_dists : np.ndarray
        Sparse distances (n, max_neighbors), padded with -1.0
    num_indices : int
        Number of query vertices
    num_vertices : int
        Total number of vertices in the mesh

    Returns
    -------
    np.ndarray
        Dense distance matrix of shape (num_indices, num_vertices)
        Self-distance is 0.0, unreachable vertices are inf
    """
    dist_matrix = np.full(
        (num_indices, num_vertices), np.float32(np.inf), dtype=np.float32
    )

    for i in range(num_indices):
        dist_matrix[i, indices[i]] = np.float32(0.0)

    for i in prange(num_indices):
        for j in range(neighborhood.shape[1]):
            neighbor = neighborhood[i, j]
            if neighbor < 0:
                break
            dist_matrix[i, neighbor] = neighborhood_dists[i, j]

    return dist_matrix


@njit(parallel=True, fastmath=True, cache=True)
def _compute_neighbor_distances(
    neighbors: np.ndarray,  # int32 (N, W), -1 sentinels
    points:    np.ndarray,  # float64 (N, D)
) -> np.ndarray:  # float64 (N, W), -1.0 sentinels
    """Compute Euclidean distances for each edge in the neighbor matrix.

    Args:
        neighbors: Connectivity matrix (N, W) with -1 as sentinel.
        points: Point positions (N, D).

    Returns:
        Distance matrix (N, W) with -1.0 as sentinel for missing neighbors.
    """
    N         = neighbors.shape[0]
    W         = neighbors.shape[1]
    D         = points.shape[1]
    distances = np.full((N, W), -1.0, dtype=np.float64)

    for i in prange(N):
        for j in range(W):
            nb = neighbors[i, j]
            if nb == -1:
                break
            s = 0.0
            for d in range(D):
                diff = points[i, d] - points[nb, d]
                s += diff * diff
            distances[i, j] = np.sqrt(s)

    return distances


@njit(parallel=False, cache=True)
def _minimize_neighbor_distances(
    neighborhood:       np.ndarray,
    neighborhood_dists: np.ndarray,
    num_vertices:       int,
) -> np.ndarray:
    """Find the shortest distance to each unique vertex index across all rows.

    Uses a parallel min-reduction pattern where each thread writes the minimum
    distance for vertex indices it encounters. This has a benign race condition
    since min of concurrent float writes still converges to the correct minimum.

    Parameters
    ----------
    neighborhood : np.ndarray
        Neighborhood indices of shape (num_indices, max_neighbors) with dtype int32.
        Padded with -1 for invalid entries.
    neighborhood_dists : np.ndarray
        Corresponding distances of shape (num_indices, max_neighbors) with dtype float32.
        Padded with -1.0 for invalid entries.
    num_vertices : int
        Total number of vertices (determines output array size).

    Returns
    -------
    np.ndarray
        Minimum distance per vertex index, shape (num_vertices,) with dtype float32.
        Vertices not referenced in neighborhood will have value inf.
    """
    min_distances = np.full(num_vertices, np.inf, dtype=np.float32)

    num_indices   = neighborhood.shape[0]
    max_neighbors = neighborhood.shape[1]

    for i in prange(num_indices):
        for j in range(max_neighbors):
            idx = neighborhood[i, j]
            if idx < 0:
                break
            dist = neighborhood_dists[i, j]
            if dist < min_distances[idx]:
                min_distances[idx] = dist

    return min_distances