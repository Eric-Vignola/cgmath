import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _lu_decompose(A):
    """
    LU decomposition with partial pivoting (Doolittle algorithm).

    Decomposes matrix A into lower triangular L and upper triangular U
    such that PA = LU, where P is a permutation matrix.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Square matrix to decompose.

    Returns
    -------
    L : ndarray of shape (n, n)
        Lower triangular matrix with ones on diagonal.
    U : ndarray of shape (n, n)
        Upper triangular matrix.
    perm : ndarray of shape (n,)
        Permutation vector representing row swaps.
    """
    n    = A.shape[0]
    L    = np.eye(n, dtype=np.float64)
    U    = A.copy()
    perm = np.arange(n, dtype=np.int64)

    for k in range(n - 1):
        max_idx = k
        max_val = abs(U[k, k])
        for i in range(k + 1, n):
            if abs(U[i, k]) > max_val:
                max_val = abs(U[i, k])
                max_idx = i

        if max_idx != k:
            for j in range(n):
                U[k, j], U[max_idx, j] = U[max_idx, j], U[k, j]
            for j in range(k):
                L[k, j], L[max_idx, j] = L[max_idx, j], L[k, j]
            perm[k], perm[max_idx] = perm[max_idx], perm[k]

        for i in range(k + 1, n):
            if abs(U[k, k]) > 1e-15:
                L[i, k] = U[i, k] / U[k, k]
                for j in range(k, n):
                    U[i, j] -= L[i, k] * U[k, j]

    return L, U, perm


@njit(fastmath=True, cache=True)
def _forward_substitute(L, b, perm):
    """
    Solve Ly = Pb using forward substitution.

    Parameters
    ----------
    L : ndarray of shape (n, n)
        Lower triangular matrix.
    b : ndarray of shape (n,) or (n, m)
        Right-hand side vector or matrix.
    perm : ndarray of shape (n,)
        Permutation vector.

    Returns
    -------
    y : ndarray of shape (n,) or (n, m)
        Solution vector or matrix.
    """
    n = L.shape[0]

    if b.ndim == 1:
        Pb = np.empty(n, dtype=np.float64)
        for i in range(n):
            Pb[i] = b[perm[i]]

        y = np.empty(n, dtype=np.float64)
        for i in range(n):
            y[i] = Pb[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
    else:
        m  = b.shape[1]
        Pb = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            for k in range(m):
                Pb[i, k] = b[perm[i], k]

        y = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            for k in range(m):
                y[i, k] = Pb[i, k]
                for j in range(i):
                    y[i, k] -= L[i, j] * y[j, k]

    return y


@njit(fastmath=True, cache=True)
def _backward_substitute(U, y):
    """
    Solve Ux = y using backward substitution.

    Parameters
    ----------
    U : ndarray of shape (n, n)
        Upper triangular matrix.
    y : ndarray of shape (n,) or (n, m)
        Right-hand side vector or matrix.

    Returns
    -------
    x : ndarray of shape (n,) or (n, m)
        Solution vector or matrix.
    """
    n = U.shape[0]

    if y.ndim == 1:
        x = np.empty(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= U[i, j] * x[j]
            if abs(U[i, i]) > 1e-15:
                x[i] /= U[i, i]
            else:
                x[i] = 0.0
    else:
        m = y.shape[1]
        x = np.empty((n, m), dtype=np.float64)
        for i in range(n - 1, -1, -1):
            for k in range(m):
                x[i, k] = y[i, k]
                for j in range(i + 1, n):
                    x[i, k] -= U[i, j] * x[j, k]
                if abs(U[i, i]) > 1e-15:
                    x[i, k] /= U[i, i]
                else:
                    x[i, k] = 0.0

    return x


@njit(fastmath=True, cache=True)
def lu_solve(A, B):
    """
    Solve the linear system AX = B using LU decomposition.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Coefficient matrix.
    B : ndarray of shape (n,) or (n, m)
        Right-hand side vector or matrix.

    Returns
    -------
    X : ndarray of shape (n,) or (n, m)
        Solution vector or matrix.
    """
    L, U, perm = _lu_decompose(A)
    y = _forward_substitute(L, B, perm)
    X = _backward_substitute(U, y)
    return X


@njit(fastmath=True, cache=True)
def lu_solve_factored(L, U, perm, B):
    """
    Solve the linear system AX = B using pre-computed LU factorization.

    This is O(n^2) compared to O(n^3) for full solve, making it efficient
    for repeated solves with the same coefficient matrix.

    Parameters
    ----------
    L : ndarray of shape (n, n)
        Lower triangular matrix from LU decomposition.
    U : ndarray of shape (n, n)
        Upper triangular matrix from LU decomposition.
    perm : ndarray of shape (n,)
        Permutation vector from LU decomposition.
    B : ndarray of shape (n,) or (n, m)
        Right-hand side vector or matrix.

    Returns
    -------
    X : ndarray of shape (n,) or (n, m)
        Solution vector or matrix.
    """
    y = _forward_substitute(L, B, perm)
    X = _backward_substitute(U, y)
    return X