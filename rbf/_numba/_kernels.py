import numpy as np
from numba import njit, prange


# -----------------------------------------------------------------------------
# Infinitely Smooth Kernels
# -----------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _gaussian(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] * epsilon
            result[i, j] = np.exp(-r * r)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _multiquadric(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] * epsilon
            result[i, j] = np.sqrt(1.0 + r * r)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _inverse_multiquadric(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] * epsilon
            result[i, j] = 1.0 / np.sqrt(1.0 + r * r)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _inverse_quadratic(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] * epsilon
            result[i, j] = 1.0 / (1.0 + r * r)

    return result


# -----------------------------------------------------------------------------
# Piecewise Smooth / Polyharmonic Kernels
# -----------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _linear(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            result[i, j] = X[i, j] / epsilon

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _cubic(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] / epsilon
            result[i, j] = r * r * r

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _quintic(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r  = X[i, j] / epsilon
            r2 = r * r
            result[i, j] = r2 * r2 * r

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _thin_plate_spline(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] / epsilon
            if r > 0.0:
                result[i, j] = r * r * np.log(r)
            else:
                result[i, j] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _polyharmonic(X, k=2, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)
    k_odd  = k & 1

    for i in prange(n):
        for j in range(m):
            r = X[i, j] / epsilon

            if k_odd:
                result[i, j] = r**k
            else:
                if r == 0.0:
                    result[i, j] = 0.0
                elif r < 1.0:
                    result[i, j] = (r ** (k - 1)) * r * np.log(r)
                else:
                    result[i, j] = (r**k) * np.log(r)

    return result


# -----------------------------------------------------------------------------
# Compactly Supported Kernels (Wendland Functions)
# -----------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _wendland_c0(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg  = X[i, j] / r
            diff = 1.0 - arg
            if diff > 0.0:
                result[i, j] = diff * diff
            else:
                result[i, j] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _wendland_c2(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg  = X[i, j] / r
            diff = 1.0 - arg
            if diff > 0.0:
                diff2 = diff * diff
                diff4 = diff2 * diff2
                result[i, j] = diff4 * (4.0 * arg + 1.0)
            else:
                result[i, j] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _wendland_c4(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg  = X[i, j] / r
            diff = 1.0 - arg
            if diff > 0.0:
                diff2 = diff * diff
                diff6 = diff2 * diff2 * diff2
                arg2  = arg * arg
                result[i, j] = diff6 * (35.0 * arg2 + 18.0 * arg + 3.0)
            else:
                result[i, j] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _wendland_c6(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg  = X[i, j] / r
            diff = 1.0 - arg
            if diff > 0.0:
                diff2 = diff * diff
                diff4 = diff2 * diff2
                diff8 = diff4 * diff4
                arg2  = arg * arg
                arg3  = arg2 * arg
                result[i, j] = diff8 * (32.0 * arg3 + 25.0 * arg2 + 8.0 * arg + 1.0)
            else:
                result[i, j] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _wu_c2(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg  = X[i, j] / r
            diff = 1.0 - arg
            if diff > 0.0:
                diff2 = diff * diff
                diff5 = diff2 * diff2 * diff
                arg2  = arg * arg
                result[i, j] = diff5 * (8.0 * arg2 + 5.0 * arg + 1.0)
            else:
                result[i, j] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _wu_c4(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg  = X[i, j] / r
            diff = 1.0 - arg
            if diff > 0.0:
                diff2 = diff * diff
                diff4 = diff2 * diff2
                diff7 = diff4 * diff2 * diff
                arg2  = arg * arg
                arg3  = arg2 * arg
                result[i, j] = diff7 * (16.0 * arg3 + 12.0 * arg2 + 5.0 * arg + 1.0)
            else:
                result[i, j] = 0.0

    return result


# -----------------------------------------------------------------------------
# Specialized Kernels (Matern and Others)
# -----------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _matern_12(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            result[i, j] = np.exp(-X[i, j] / epsilon)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _matern_32(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)
    sqrt3  = 1.7320508075688772  # np.sqrt(3)

    for i in prange(n):
        for j in range(m):
            z = sqrt3 * X[i, j] / epsilon
            result[i, j] = (1.0 + z) * np.exp(-z)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _matern_52(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)
    sqrt5  = 2.23606797749979  # np.sqrt(5)

    for i in prange(n):
        for j in range(m):
            r          = X[i, j]
            z          = sqrt5 * r / epsilon
            r_over_eps = r / epsilon
            result[i, j] = (1.0 + z + (5.0 / 3.0) * r_over_eps * r_over_eps) * np.exp(
                -z
            )

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _cauchy(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            r = X[i, j] / epsilon
            result[i, j] = 1.0 / (1.0 + r * r)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _log_kernel(X, epsilon=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            result[i, j] = np.log(1.0 + X[i, j] / epsilon)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _bump(X, r=1.0):
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)

    for i in prange(n):
        for j in range(m):
            arg = X[i, j] / r
            if arg < 1.0:
                arg2  = arg * arg
                denom = 1.0 - arg2
                if denom > 1e-10:
                    result[i, j] = np.exp(-1.0 / denom)
                else:
                    result[i, j] = 0.0
            else:
                result[i, j] = 0.0

    return result