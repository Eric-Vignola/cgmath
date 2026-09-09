import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _cdist_euclidean(X, Y):
    """Pairwise Euclidean distance between the rows of X and Y."""
    m         = X.shape[0]
    n         = Y.shape[0]
    k         = X.shape[1]

    distances = np.empty((m, n), dtype=np.float64)

    for i in prange(m):  # Parallel loop over rows of X
        for j in range(n):
            d = 0.0
            for l in range(k):
                diff = X[i, l] - Y[j, l]
                d += diff * diff
            distances[i, j] = np.sqrt(d)

    return distances