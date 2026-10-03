"""
Optimized Numba implementations for B-spline basis computation.

Uses the Cox-de Boor recursion formula with parallel evaluation
across sample points.

Key optimizations:
- Parallel processing of independent sample points
- Loop reordering to enable parallelization
- Local per-thread buffers to avoid race conditions
- Fused basis computation and curve/surface evaluation
- Division-by-zero protection
"""

import math

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                     Knot span lookup                                        #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, cache=True)
def _find_span(u, kv, d, limit):
    """
    Finds the knot span holding a parameter value.

    Spans are counted from the start of the curve's domain: span ``s``
    covers ``kv[d + s] <= u < kv[d + s + 1]``. Values before the domain
    get span 0, values at or past the start of the last span get
    ``limit``.

    Uniform integer knots take a shortcut (the span is ``floor(u)``);
    any other non-decreasing knot vector falls back to a binary search.

    Parameters
    ----------
    u : float
        Parameter value.
    kv : np.ndarray
        Knot vector (c + d + 1,).
    d : int
        Degree of the B-spline.
    limit : int
        Index of the last span, c - d - 1.

    Returns
    -------
    int
        Span index in [0, limit].
    """
    if u >= kv[d + limit]:
        return limit

    s = int(np.floor(u))
    if 0 <= s < limit and kv[d + s] <= u < kv[d + s + 1]:
        return s

    lo = 0
    hi = limit
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if kv[d + mid] <= u:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
#                     Original implementation                                 #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, cache=True)
def _compute_basis(u, kv, c, d):
    """
    Computes the B-spline basis functions.

    Original sequential implementation kept for compatibility.

    Uses the Cox-de Boor recursion formula to compute basis functions
    for all sample points.

    Parameters
    ----------
    u : np.ndarray
        Parameter values (n,), float64.
    kv : np.ndarray
        Knot vector (c + d + 1,), float64.
    c : int
        Number of control points.
    d : int
        Degree of the B-spline.

    Returns
    -------
    np.ndarray
        Basis function values (n, c), float64.
    """
    n     = u.shape[0]
    limit = c - d - 1
    left  = np.empty(n, dtype=np.int64)
    right = np.empty(n, dtype=np.int64)

    for i in range(n):
        left[i]  = _find_span(u[i], kv, d, limit)
        right[i] = left[i] + d + 1

    b  = np.empty((n, c), dtype=u.dtype)
    bb = np.empty((n, c), dtype=u.dtype)

    for i in range(n):
        for j in range(c):
            b[i, j]  = 0.0
            bb[i, j] = 0.0

    for i in range(n):
        b[i, int(left[i])] = 1.0

    for j in range(1, d + 1):
        for i in range(j):
            for k in range(n):
                li            = int(left[k])
                bb[k, li + i] = b[k, li + i]
                b[k, li]      = 0.0

        for i in range(j):
            for k in range(n):
                li    = int(left[k])
                ri    = int(right[k])
                denom = kv[ri + i] - kv[ri + i - j]
                if denom != 0.0:
                    f = bb[k, li + i] / denom
                else:
                    f = 0.0
                b[k, li + i]     = b[k, li + i] + f * (kv[ri + i] - u[k])
                b[k, li + i + 1] = f * (u[k] - kv[ri + i - j])

    return b


# --------------------------------------------------------------------------- #
#                     Parallelized implementation                             #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compute_basis_parallel(u, kv, c, d):
    """
    Parallelized B-spline basis computation.

    Each sample point is processed independently in parallel.
    Uses local per-thread buffers to avoid race conditions.

    Parameters
    ----------
    u : np.ndarray
        Parameter values (n,), float64.
    kv : np.ndarray
        Knot vector (c + d + 1,), float64.
    c : int
        Number of control points.
    d : int
        Degree of the B-spline.

    Returns
    -------
    np.ndarray
        Basis function values (n, c), float64.
    """
    n     = u.shape[0]
    limit = c - d - 1
    d1    = d + 1

    # Output basis matrix
    b = np.zeros((n, c), dtype=u.dtype)

    # Process each sample point in parallel
    for k in prange(n):
        # Compute left/right indices for this sample
        left_k  = _find_span(u[k], kv, d, limit)
        right_k = left_k + d1

        u_k = u[k]

        # Local basis buffer for this sample (size d+2 is enough)
        # We only need to track basis values in the local support
        local_b  = np.zeros(d1 + 1, dtype=u.dtype)
        local_bb = np.zeros(d1 + 1, dtype=u.dtype)

        # Initialize: B_{left,0}(u) = 1
        local_b[0] = 1.0

        # Cox-de Boor recursion
        for j in range(1, d1):
            # Copy to buffer and reset
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i]  = 0.0

            # Compute new basis values
            for i in range(j):
                ri    = right_k + i
                denom = kv[ri] - kv[ri - j]
                if denom != 0.0:
                    f = local_bb[i] / denom
                else:
                    f = 0.0
                local_b[i] += f * (kv[ri] - u_k)
                local_b[i + 1] = f * (u_k - kv[ri - j])

        # Write local results to output matrix
        for i in range(d1):
            if left_k + i < c:
                b[k, left_k + i] = local_b[i]

    return b


# --------------------------------------------------------------------------- #
#                     Fused evaluation functions                              #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_bspline_curve(u, kv, control_points, d):
    """
    Evaluate B-spline curve at parameter values in parallel.

    Combines basis computation and control point evaluation in a single
    pass for better cache efficiency.

    Parameters
    ----------
    u : np.ndarray
        Parameter values (n,), float64.
    kv : np.ndarray
        Knot vector (c + d + 1,), float64.
    control_points : np.ndarray
        Control point positions (c, dims), float64.
    d : int
        Degree of the B-spline.

    Returns
    -------
    np.ndarray
        Evaluated points (n, dims), float64.
    """
    n     = u.shape[0]
    c     = control_points.shape[0]
    dims  = control_points.shape[1]
    limit = c - d - 1
    d1    = d + 1

    # Output points
    result = np.zeros((n, dims), dtype=control_points.dtype)

    # Process each sample point in parallel
    for k in prange(n):
        # Compute left/right indices
        left_k  = _find_span(u[k], kv, d, limit)
        right_k = left_k + d1

        u_k = u[k]

        # Local basis buffer
        local_b  = np.zeros(d1 + 1, dtype=u.dtype)
        local_bb = np.zeros(d1 + 1, dtype=u.dtype)

        # Initialize
        local_b[0] = 1.0

        # Cox-de Boor recursion
        for j in range(1, d1):
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i]  = 0.0

            for i in range(j):
                ri    = right_k + i
                denom = kv[ri] - kv[ri - j]
                if denom != 0.0:
                    f = local_bb[i] / denom
                else:
                    f = 0.0
                local_b[i] += f * (kv[ri] - u_k)
                local_b[i + 1] = f * (u_k - kv[ri - j])

        # Evaluate curve point using basis
        for i in range(d1):
            cp_idx = left_k + i
            if cp_idx < c:
                basis_val = local_b[i]
                for dim in range(dims):
                    result[k, dim] += basis_val * control_points[cp_idx, dim]

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_bspline_surface(u, v, kv_u, kv_v, control_points, du, dv):
    """
    Evaluate B-spline surface at (u, v) parameter values in parallel.

    Computes both U and V basis functions and combines them with the
    control point grid.

    Parameters
    ----------
    u : np.ndarray
        U parameter values (n,), float64.
    v : np.ndarray
        V parameter values (n,), float64.
    kv_u : np.ndarray
        U knot vector (cu + du + 1,), float64.
    kv_v : np.ndarray
        V knot vector (cv + dv + 1,), float64.
    control_points : np.ndarray
        Control point grid (cu, cv, dims), float64.
    du : int
        Degree in U direction.
    dv : int
        Degree in V direction.

    Returns
    -------
    np.ndarray
        Evaluated points (n, dims), float64.
    """
    n    = u.shape[0]
    cu   = control_points.shape[0]
    cv   = control_points.shape[1]
    dims = control_points.shape[2]

    limit_u = cu - du - 1
    limit_v = cv - dv - 1
    du1     = du + 1
    dv1     = dv + 1

    result  = np.zeros((n, dims), dtype=control_points.dtype)

    for k in prange(n):
        u_k = u[k]
        v_k = v[k]

        # Compute U basis
        left_u = int(np.floor(u_k))
        if left_u < 0:
            left_u = 0
        elif left_u > limit_u:
            left_u = limit_u
        right_u = left_u + du1

        basis_u    = np.zeros(du1 + 1, dtype=u.dtype)
        buf_u      = np.zeros(du1 + 1, dtype=u.dtype)
        basis_u[0] = 1.0

        for j in range(1, du1):
            for i in range(j):
                buf_u[i]   = basis_u[i]
                basis_u[i] = 0.0
            for i in range(j):
                ri    = right_u + i
                denom = kv_u[ri] - kv_u[ri - j]
                if denom != 0.0:
                    f = buf_u[i] / denom
                else:
                    f = 0.0
                basis_u[i] += f * (kv_u[ri] - u_k)
                basis_u[i + 1] = f * (u_k - kv_u[ri - j])

        # Compute V basis
        left_v = int(np.floor(v_k))
        if left_v < 0:
            left_v = 0
        elif left_v > limit_v:
            left_v = limit_v
        right_v = left_v + dv1

        basis_v    = np.zeros(dv1 + 1, dtype=v.dtype)
        buf_v      = np.zeros(dv1 + 1, dtype=v.dtype)
        basis_v[0] = 1.0

        for j in range(1, dv1):
            for i in range(j):
                buf_v[i]   = basis_v[i]
                basis_v[i] = 0.0
            for i in range(j):
                ri    = right_v + i
                denom = kv_v[ri] - kv_v[ri - j]
                if denom != 0.0:
                    f = buf_v[i] / denom
                else:
                    f = 0.0
                basis_v[i] += f * (kv_v[ri] - v_k)
                basis_v[i + 1] = f * (v_k - kv_v[ri - j])

        # Evaluate surface point using tensor product of bases
        for iu in range(du1):
            cp_u = left_u + iu
            if cp_u >= cu:
                continue
            basis_u_val = basis_u[iu]

            for iv in range(dv1):
                cp_v = left_v + iv
                if cp_v >= cv:
                    continue

                weight = basis_u_val * basis_v[iv]
                for dim in range(dims):
                    result[k, dim] += weight * control_points[cp_u, cp_v, dim]

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_bspline_surface_derivatives(u, v, kv_u, kv_v, control_points, du, dv):
    """
    Evaluate B-spline surface and first partial derivatives in parallel.

    Computes S(u,v), dS/du, dS/dv in a single fused pass using
    Cox-de Boor recursion at degrees (du, dv) and (du-1, dv-1).

    Parameters
    ----------
    u : np.ndarray
        U parameter values (n,), float64.
    v : np.ndarray
        V parameter values (n,), float64.
    kv_u : np.ndarray
        U knot vector (cu + du + 1,), float64.
    kv_v : np.ndarray
        V knot vector (cv + dv + 1,), float64.
    control_points : np.ndarray
        Control point grid (cu, cv, dims), float64.
    du : int
        Degree in U direction.
    dv : int
        Degree in V direction.

    Returns
    -------
    points : np.ndarray
        Surface points (n, dims), float64.
    tangents_u : np.ndarray
        Partial derivatives dS/du (n, dims), float64.
    tangents_v : np.ndarray
        Partial derivatives dS/dv (n, dims), float64.
    """
    n    = u.shape[0]
    cu   = control_points.shape[0]
    cv   = control_points.shape[1]
    dims = control_points.shape[2]

    limit_u = cu - du - 1
    limit_v = cv - dv - 1
    du1     = du + 1
    dv1     = dv + 1

    points     = np.zeros((n, dims), dtype=control_points.dtype)
    tangents_u = np.zeros((n, dims), dtype=control_points.dtype)
    tangents_v = np.zeros((n, dims), dtype=control_points.dtype)

    for k in prange(n):
        u_k = u[k]
        v_k = v[k]

        # === U basis at degree du ===
        left_u = int(np.floor(u_k))
        if left_u < 0:
            left_u = 0
        elif left_u > limit_u:
            left_u = limit_u
        right_u = left_u + du1

        basis_u    = np.zeros(du1 + 1, dtype=u.dtype)
        buf_u      = np.zeros(du1 + 1, dtype=u.dtype)
        basis_u[0] = 1.0

        for j in range(1, du1):
            for i in range(j):
                buf_u[i]   = basis_u[i]
                basis_u[i] = 0.0
            for i in range(j):
                ri    = right_u + i
                denom = kv_u[ri] - kv_u[ri - j]
                if denom != 0.0:
                    f = buf_u[i] / denom
                else:
                    f = 0.0
                basis_u[i] += f * (kv_u[ri] - u_k)
                basis_u[i + 1] = f * (u_k - kv_u[ri - j])

        # === U basis at degree du-1 (for dS/du) ===
        basis_u_d1 = np.zeros(du1, dtype=u.dtype)
        buf_u_d1   = np.zeros(du1, dtype=u.dtype)
        if du >= 1:
            basis_u_d1[0] = 1.0
            for j in range(1, du):
                for i in range(j):
                    buf_u_d1[i]   = basis_u_d1[i]
                    basis_u_d1[i] = 0.0
                for i in range(j):
                    ri    = right_u + i
                    denom = kv_u[ri] - kv_u[ri - j]
                    if denom != 0.0:
                        f = buf_u_d1[i] / denom
                    else:
                        f = 0.0
                    basis_u_d1[i] += f * (kv_u[ri] - u_k)
                    basis_u_d1[i + 1] = f * (u_k - kv_u[ri - j])

        # === V basis at degree dv ===
        left_v = int(np.floor(v_k))
        if left_v < 0:
            left_v = 0
        elif left_v > limit_v:
            left_v = limit_v
        right_v = left_v + dv1

        basis_v    = np.zeros(dv1 + 1, dtype=v.dtype)
        buf_v      = np.zeros(dv1 + 1, dtype=v.dtype)
        basis_v[0] = 1.0

        for j in range(1, dv1):
            for i in range(j):
                buf_v[i]   = basis_v[i]
                basis_v[i] = 0.0
            for i in range(j):
                ri    = right_v + i
                denom = kv_v[ri] - kv_v[ri - j]
                if denom != 0.0:
                    f = buf_v[i] / denom
                else:
                    f = 0.0
                basis_v[i] += f * (kv_v[ri] - v_k)
                basis_v[i + 1] = f * (v_k - kv_v[ri - j])

        # === V basis at degree dv-1 (for dS/dv) ===
        basis_v_d1 = np.zeros(dv1, dtype=v.dtype)
        buf_v_d1   = np.zeros(dv1, dtype=v.dtype)
        if dv >= 1:
            basis_v_d1[0] = 1.0
            for j in range(1, dv):
                for i in range(j):
                    buf_v_d1[i]   = basis_v_d1[i]
                    basis_v_d1[i] = 0.0
                for i in range(j):
                    ri    = right_v + i
                    denom = kv_v[ri] - kv_v[ri - j]
                    if denom != 0.0:
                        f = buf_v_d1[i] / denom
                    else:
                        f = 0.0
                    basis_v_d1[i] += f * (kv_v[ri] - v_k)
                    basis_v_d1[i + 1] = f * (v_k - kv_v[ri - j])

        # === Evaluate surface point S(u,v) ===
        for iu in range(du1):
            cp_u = left_u + iu
            if cp_u >= cu:
                continue
            bu = basis_u[iu]
            for iv in range(dv1):
                cp_v = left_v + iv
                if cp_v >= cv:
                    continue
                weight = bu * basis_v[iv]
                for dim in range(dims):
                    points[k, dim] += weight * control_points[cp_u, cp_v, dim]

        # === Evaluate dS/du ===
        if du >= 1:
            for iu in range(du):
                cp_u = left_u + iu
                if cp_u + 1 >= cu:
                    continue
                kv_idx   = cp_u + 1
                denom_kv = kv_u[kv_idx + du] - kv_u[kv_idx]
                if denom_kv != 0.0:
                    for iv in range(dv1):
                        cp_v = left_v + iv
                        if cp_v >= cv:
                            continue
                        scale = du * basis_u_d1[iu] * basis_v[iv] / denom_kv
                        for dim in range(dims):
                            tangents_u[k, dim] += scale * (
                                control_points[cp_u + 1, cp_v, dim]
                                - control_points[cp_u, cp_v, dim]
                            )

        # === Evaluate dS/dv ===
        if dv >= 1:
            for iu in range(du1):
                cp_u = left_u + iu
                if cp_u >= cu:
                    continue
                for iv in range(dv):
                    cp_v = left_v + iv
                    if cp_v + 1 >= cv:
                        continue
                    kv_idx   = cp_v + 1
                    denom_kv = kv_v[kv_idx + dv] - kv_v[kv_idx]
                    if denom_kv != 0.0:
                        scale = dv * basis_u[iu] * basis_v_d1[iv] / denom_kv
                        for dim in range(dims):
                            tangents_v[k, dim] += scale * (
                                control_points[cp_u, cp_v + 1, dim]
                                - control_points[cp_u, cp_v, dim]
                            )

    return points, tangents_u, tangents_v


# --------------------------------------------------------------------------- #
#                     Derivative computation                                  #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _compute_basis_derivatives(u, kv, c, d, order=1):
    """
    Compute B-spline basis function derivatives in parallel.

    The Cox-de Boor recursion runs up to degree d - order; each of the
    last ``order`` degrees then applies the derivative recursion

        N^(k)_{m,p} = p * (N^(k-1)_{m,p-1} / (t_{m+p} - t_m)
            - N^(k-1)_{m+1,p-1} / (t_{m+p+1} - t_{m+1}))

    with 0/0 taken as 0. This holds for any non-decreasing knot vector:
    clamped, uniform, non-uniform or with repeated knots.

    Parameters
    ----------
    u : np.ndarray
        Parameter values (n,), float64.
    kv : np.ndarray
        Knot vector (c + d + 1,), float64.
    c : int
        Number of control points.
    d : int
        Degree of the B-spline.
    order : int
        Derivative order (default 1 for first derivative). Order 0 gives
        the basis itself, orders above d give zeros.

    Returns
    -------
    np.ndarray
        Basis derivative values (n, c), float64.

    Raises
    ------
    ValueError
        If order is negative.
    """
    if order < 0:
        raise ValueError("order must be non-negative")

    n     = u.shape[0]
    limit = c - d - 1
    d1    = d + 1

    # Output derivative matrix
    db = np.zeros((n, c), dtype=u.dtype)

    if order > d:
        # Derivative order exceeds degree - result is zero
        return db

    # Process each sample point in parallel
    for k in prange(n):
        s    = _find_span(u[k], kv, d, limit)
        span = s + d
        u_k  = u[k]

        # After degree j, local_b[i] belongs to basis function s + d - j + i
        local_b    = np.zeros(d1, dtype=u.dtype)
        local_bb   = np.zeros(d1, dtype=u.dtype)
        local_b[0] = 1.0

        for j in range(1, d1):
            for i in range(j):
                local_bb[i] = local_b[i]
            local_b[0] = 0.0

            for i in range(1, j + 1):
                left  = kv[span + i - j]
                right = kv[span + i]
                denom = right - left
                if denom == 0.0:
                    # Zero-length support, the 0/0 term is taken as 0
                    local_b[i] = 0.0
                elif j <= d - order:
                    # Cox-de Boor step
                    f = local_bb[i - 1] / denom
                    local_b[i - 1] += f * (right - u_k)
                    local_b[i] = f * (u_k - left)
                else:
                    # Derivative step
                    f = j * local_bb[i - 1] / denom
                    local_b[i - 1] -= f
                    local_b[i] = f

        # Write local results to output matrix
        for i in range(d1):
            db[k, s + i] = local_b[i]

    return db


@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_bspline_curve_derivative(u, kv, control_points, d):
    """
    Evaluate B-spline curve first derivative (tangent) in parallel.

    Parameters
    ----------
    u : np.ndarray
        Parameter values (n,), float64.
    kv : np.ndarray
        Knot vector (c + d + 1,), float64.
    control_points : np.ndarray
        Control point positions (c, dims), float64.
    d : int
        Degree of the B-spline.

    Returns
    -------
    np.ndarray
        Tangent vectors (n, dims), float64.
    """
    n      = u.shape[0]
    c      = control_points.shape[0]
    dims   = control_points.shape[1]
    limit  = c - d - 1
    d1     = d + 1

    result = np.zeros((n, dims), dtype=control_points.dtype)

    if d < 1:
        return result

    for k in prange(n):
        left_k  = _find_span(u[k], kv, d, limit)
        right_k = left_k + d1

        u_k = u[k]

        # Compute basis at degree d-1 for derivative
        d_minus_1  = d
        local_b    = np.zeros(d_minus_1 + 1, dtype=u.dtype)
        local_bb   = np.zeros(d_minus_1 + 1, dtype=u.dtype)
        local_b[0] = 1.0

        for j in range(1, d_minus_1):
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i]  = 0.0

            for i in range(j):
                ri    = right_k + i
                denom = kv[ri] - kv[ri - j]
                if denom != 0.0:
                    f = local_bb[i] / denom
                else:
                    f = 0.0
                local_b[i] += f * (kv[ri] - u_k)
                local_b[i + 1] = f * (u_k - kv[ri - j])

        # Compute derivative using difference of control points
        # C'(u) = d * sum_i (Q_i * B_{i,d-1}(u))
        # where Q_i = (P_{i+1} - P_i) / (kv[i+d+1] - kv[i+1])
        for i in range(d_minus_1):
            cp_idx = left_k + i
            if cp_idx + 1 < c:
                kv_idx = cp_idx + 1
                denom  = kv[kv_idx + d] - kv[kv_idx]
                if denom != 0.0:
                    scale = d * local_b[i] / denom
                    for dim in range(dims):
                        diff = (
                            control_points[cp_idx + 1, dim]
                            - control_points[cp_idx, dim]
                        )
                        result[k, dim] += scale * diff

    return result


# --------------------------------------------------------------------------- #
#                     Knot vector utilities                                   #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, cache=True)
def _create_uniform_knot_vector(n_control_points, degree):
    """
    Create a uniform clamped knot vector.

    Parameters
    ----------
    n_control_points : int
        Number of control points.
    degree : int
        Degree of the B-spline.

    Returns
    -------
    np.ndarray
        Knot vector (n_control_points + degree + 1,), float64.
    """
    n_knots = n_control_points + degree + 1
    kv      = np.zeros(n_knots, dtype=np.float64)

    # Clamped: first (degree+1) knots are 0, last (degree+1) knots are 1
    n_internal = n_knots - 2 * (degree + 1)

    for i in range(degree + 1):
        kv[i]               = 0.0
        kv[n_knots - 1 - i] = float(n_internal + 1)

    # Internal knots are uniformly spaced
    for i in range(n_internal):
        kv[degree + 1 + i] = float(i + 1)

    return kv


@njit(parallel=True, fastmath=True, cache=True)
def _normalize_parameters(u, kv, d):
    """
    Normalize parameter values to valid range for the knot vector.

    Parameters
    ----------
    u : np.ndarray
        Parameter values (n,), float64.
    kv : np.ndarray
        Knot vector.
    d : int
        Degree.

    Returns
    -------
    np.ndarray
        Normalized parameters clamped to valid range.
    """
    n      = u.shape[0]
    result = np.empty(n, dtype=u.dtype)

    u_min  = kv[d]
    u_max  = kv[kv.shape[0] - d - 1]

    for i in prange(n):
        val = u[i]
        if val < u_min:
            val = u_min
        elif val > u_max:
            val = u_max
        result[i] = val

    return result


# --------------------------------------------------------------------------- #
#                     Ordered closest points                                  #
# --------------------------------------------------------------------------- #
#
# fit()'s best fit gives each fit point a curve parameter u. The kernels
# below repair those u: a point only ever moves to a strictly closer spot on
# the curve between its neighbours' current u, so the points keep their
# order and none gets farther from the curve.
#
# Parameters are native: the curve runs on [0, spans]. On periodic curves
# they may be unwrapped (any lap); the curve is evaluated modulo spans.
#
# The curve and its derivatives are evaluated with scipy's own de Boor
# recursion, so every position, slope and Newton step is bit-identical to
# scipy's BSpline.
#
# fastmath stays off: the "strictly closer" tests compare distances at
# float-noise level.


@njit(cache=True)
def _curve_derivatives(x, kv, cv, degree, order, work, out):
    """
    The curve's derivatives up to ``order`` at one native parameter, as scipy's.

    Parameters
    ----------
    x : float
        Native parameter, inside the domain [kv[degree], kv[c]].
    kv, cv : np.ndarray
        Knot vector (c + degree + 1,) in scipy's layout and control points
        (c, dims), wrapped on periodic curves; float64.
    degree, order : int
        Degree of the B-spline; highest derivative wanted (past degree: zero).
    work, out : np.ndarray
        Scratch (degree + 2, degree + 1); output (order + 1 or more, dims),
        row m the m-th derivative.

    Returns
    -------
    None
    """
    # Right-sided at the knots, like scipy. A repeated knot at either end of
    # the domain leaves an empty span there (scipy returns zeros): the nearest
    # non-empty span takes it
    limit = cv.shape[0] - degree - 1
    span  = _find_span(x, kv, degree, limit)
    while span > 0 and not kv[degree + span] < kv[degree + span + 1]:
        span -= 1
    while span < limit and not kv[degree + span] < kv[degree + span + 1]:
        span += 1

    # scipy's de Boor recursion, so every value is bit-identical to
    # BSpline(x, m): degree - m Cox-de Boor steps (shared, highest m first)
    # raise the basis functions to degree - m, in place, then m steps
    # differentiate them back up. A zero knot difference adds nothing
    knot     = span + degree
    basis    = work[degree + 1]
    basis[0] = 1.0
    level    = 0
    top      = min(order, degree)
    for m in range(top, -1, -1):
        while level < degree - m:
            level += 1
            carry = 0.0
            for n in range(1, level + 1):
                upper = kv[knot + n]
                lower = kv[knot + n - level]
                if upper == lower:
                    basis[n - 1] = carry
                    carry        = 0.0
                    continue
                w            = basis[n - 1] / (upper - lower)
                basis[n - 1] = carry + w * (upper - x)
                carry        = w * (x - lower)
            basis[level] = carry

        row              = work[m]
        row[: level + 1] = basis[: level + 1]
        for j in range(level + 1, degree + 1):
            carry = 0.0
            for n in range(1, j + 1):
                upper = kv[knot + n]
                lower = kv[knot + n - j]
                if upper == lower:
                    row[n - 1] = carry
                    carry      = 0.0
                    continue
                w          = j * row[n - 1] / (upper - lower)
                row[n - 1] = carry - w
                carry      = w
            row[j] = carry

        for dim in range(cv.shape[1]):
            total = 0.0
            for j in range(degree + 1):
                total += cv[span + j, dim] * row[j]
            out[m, dim] = total
    out[top + 1 : order + 1] = 0.0


@njit(cache=True, inline="always")
def _knot_on_lap(m, knots, spans):
    """
    Knot m counted over every lap: knot m % spans, moved m // spans laps.

    Parameters
    ----------
    m : int
        Knot index, any integer (negative = earlier laps).
    knots, spans
        The domain's knots (spans + 1,), from 0 to spans; the span count.

    Returns
    -------
    float
        The knot's unwrapped native parameter.
    """
    return knots[m % spans] + (m // spans) * float(spans)


@njit(cache=True)
def _distance(point, x, kv, cv, degree, periodic, work, derivs, slope):
    """
    |C(x) - point|, or with ``slope`` (C(x) - point) . C'(x).

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    x : float
        Native parameter, unwrapped on periodic curves.
    kv, cv, degree, periodic
        The curve, as in ``_curve_derivatives``.
    work, derivs : np.ndarray
        Scratch for ``_curve_derivatives``; derivs (3, dims) or more.
    slope : bool
        Return half the squared distance's derivative instead (negative while
        the curve still comes closer).

    Returns
    -------
    float
        The distance or the slope.
    """
    if periodic:
        x = x % float(cv.shape[0] - degree)
    _curve_derivatives(x, kv, cv, degree, int(slope), work, derivs)
    total = 0.0
    for dim in range(point.shape[0]):
        offset = derivs[0, dim] - point[dim]
        total += offset * (derivs[1, dim] if slope else offset)
    return total if slope else np.sqrt(total)


@njit(cache=True)
def _keep_closer(point, x, rank, best, kv, cv, degree, periodic, work, derivs):
    """
    The closer of candidate x and the best so far; the smaller rank on a tie.

    Parameters
    ----------
    point, x, kv, cv, degree, periodic, work, derivs
        As in ``_distance``.
    rank : int
        The candidate's rank (see ``_window_closest``).
    best : tuple
        (x, distance, rank) of the best so far.

    Returns
    -------
    tuple
        (x, distance, rank) of the closer one.
    """
    dist = _distance(point, x, kv, cv, degree, periodic, work, derivs, False)
    if dist < best[1] or (dist == best[1] and rank < best[2]):
        return x, dist, rank
    return best


@njit(cache=True)
def _score_newton(point, y, lower, upper, stay, rank, group, best, kv, cv, degree,
                  periodic, work, derivs):
    """
    Newton's method on the distance slope inside a bracket, then scored.

    Parameters
    ----------
    point, kv, cv, degree, periodic, work, derivs
        As in ``_distance``.
    y : float
        Start; clipped into the bracket first.
    lower, upper : float
        The bracket, native parameters.
    stay : bool
        Where Newton fails, stop (a root of the piece's polynomial, already
        close) instead of bisecting (a lone minimum known to be inside).
    rank, group : int
        The result's rank; the floats before and after it rank group and
        2 x group later.
    best : tuple
        (x, distance, rank) of the best so far.

    Returns
    -------
    tuple
        The new best (x, distance, rank).
    """
    # Each step shrinks the bracket to the side the slope points at. A step
    # that leaves it, or a non-convex spot, stops or bisects. Done once a step
    # moves less than 1e-12 x max(spans, 1), or after 60 steps
    period   = float(cv.shape[0] - degree)
    step_tol = 1e-12 * max(period, 1.0)
    y        = min(max(y, lower), upper)
    low      = lower
    high     = upper
    for _ in range(60):
        x = y % period if periodic else y
        _curve_derivatives(x, kv, cv, degree, 2, work, derivs)

        # slope = (C - p).C', bend = its derivative |C'|^2 + (C - p).C''
        slope = 0.0
        speed = 0.0
        pull  = 0.0
        for dim in range(point.shape[0]):
            offset = derivs[0, dim] - point[dim]
            slope += offset * derivs[1, dim]
            speed += derivs[1, dim] * derivs[1, dim]
            pull  += offset * derivs[2, dim]
        bend = speed + pull

        if slope < 0.0:
            low = y
        if slope > 0.0:
            high = y
        inside = False
        if bend > 0.0:
            y_new  = y - slope / bend
            inside = low <= y_new <= high
        if not inside:
            y_new = y if stay else (low + high) / 2.0
        moved = abs(y_new - y)
        y     = y_new
        if not moved > step_tol:
            break

    # The distance is flat at a minimum: the floats either side may be a hair
    # closer than the one Newton stopped on
    best = _keep_closer(point, y, rank, best, kv, cv, degree, periodic, work, derivs)
    best = _keep_closer(point, np.nextafter(y, lower), rank + group, best, kv, cv,
                        degree, periodic, work, derivs)
    best = _keep_closer(point, np.nextafter(y, upper), rank + 2 * group, best, kv, cv,
                        degree, periodic, work, derivs)
    return best


@njit(cache=True)
def _polynomial_roots(poly, matrix, real, imag):
    """
    The real roots in [0, 1] of a polynomial, ascending: companion eigenvalues.

    Parameters
    ----------
    poly : np.ndarray
        Power coefficients (size + 1,), lowest first, float64.
    matrix : np.ndarray
        Scratch (size, size), complex128.
    real, imag : np.ndarray
        Scratch (size,), float64; real[:found] receives the roots.

    Returns
    -------
    int
        found: eigenvalues within 1e-2 of real and 1e-3 of [0, 1]; none when
        the matrix is not finite or the eigenvalues do not converge.
    """
    # The coefficients go in the first row (numpy.roots' layout: its roots
    # come out slightly more accurate than the last column's). A vanishing leading
    # coefficient is held at 1e-13 (a huge, harmless root). Complex: below
    # numpy 2.5, numba's eigvals will not turn a real matrix's eigenvalues
    # complex
    size      = matrix.shape[0]
    matrix[:] = 0.0
    for i in range(1, size):
        matrix[i, i - 1] = 1.0
    lead = poly[size]
    if abs(lead) < 1e-13:
        lead = 1e-13
    for i in range(size):
        matrix[0, i] = -poly[size - 1 - i] / lead

    # eigvals raises on a matrix that is not finite (checked first, which
    # avoids the raise) and when it does not converge
    if not np.isfinite(matrix[0]).all():
        return 0
    try:
        roots = np.linalg.eigvals(matrix)
    except Exception:
        return 0
    real[:] = roots.real
    imag[:] = roots.imag

    # The real ones in [0, 1], in ascending order (insertion sort): on an
    # exact tie the earlier root wins
    found = 0
    for i in range(size):
        if abs(imag[i]) < 1e-2 and -1e-3 < real[i] < 1.0 + 1e-3:
            root = real[i]
            at   = found
            while at > 0 and real[at - 1] > root:
                real[at] = real[at - 1]
                at -= 1
            real[at] = root
            found += 1
    return found


@njit(cache=True)
def _window_closest(point, lo, hi, kv, cv, degree, periodic, wild, scratch):
    """
    One point's exact closest spot on the curve inside a parameter window.

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    lo, hi : float
        The window, native, finite and within 2 ** 31 laps of 0; unwrapped on
        periodic curves, where it may cross the seam or reach into other laps
        (cut to three laps from lo: it holds every spot of the curve anyway).
    kv, cv, degree, periodic
        The curve, as in ``_curve_derivatives``.
    wild : bool
        Whether the control points are beyond 1e5 x the points.
    scratch : tuple
        (work, derivs, coef, poly, bern, companion, real, imag, to_bern, fact)
        from ``_ordered_closest_params``.

    Returns
    -------
    tuple
        (u, distance) of the closest spot in [lo, hi]; on an exact tie the
        smallest rank.
    """
    work, derivs, coef, poly, bern, companion, real, imag, to_bern, fact = scratch
    dims   = point.shape[0]
    spans  = cv.shape[0] - degree
    period = float(spans)
    knots  = kv[degree : degree + spans + 1]
    size   = 2 * degree - 1  # the slope's degree

    if periodic and hi - lo > 3.0 * period:
        hi = lo + 3.0 * period

    # The window is cut into pieces at the knots: the knot spans it crosses,
    # counted over every lap
    lap_lo = int(np.floor(lo / period))
    lap_hi = int(np.floor(hi / period))
    first  = lap_lo * spans + np.searchsorted(knots, lo % period, side="right") - 1
    last   = lap_hi * spans + np.searchsorted(knots, hi % period) - 1
    if not periodic:
        first = min(max(first, 0), spans - 1)
        last  = min(max(last, 0), spans - 1)
    pieces = max(last - first + 1, 1)

    # Of two candidates exactly as close the smaller rank wins: group x (0 piece
    # start, 1 piece end, 2 Newton result, 3 the float before it, 4 the float
    # after it) + kinds x (0 polynomial root, 1 lone minimum, 2 sampled
    # bracket) + 64 x piece + root or bracket number
    kinds = pieces * 64
    group = 3 * kinds
    best  = (lo, np.inf, 5 * group)
    for piece in range(pieces):
        span  = first + piece
        knot0 = _knot_on_lap(span, knots, spans)
        start = max(lo, knot0)
        end   = max(min(hi, _knot_on_lap(span + 1, knots, spans)), lo)
        width = end - start
        at    = piece * 64

        # The piece ends always compete; an empty piece has nothing else
        best = _keep_closer(point, start, at, best, kv, cv, degree, periodic, work,
                            derivs)
        best = _keep_closer(point, end, group + at, best, kv, cv, degree, periodic,
                            work, derivs)
        if not width > 0.0:
            continue

        # (C - p).C' on the piece, as a polynomial in t in [0, 1]: on one knot
        # span the Taylor expansion C(base + t w) = sum_m C^(m)(base) (t w)^m / m!
        # is exact, so (C - p).dC/dt has degree 2 x degree - 1. base is on the
        # domain's lap, measured from the span's own knot
        base = knots[span % spans] + (start - knot0)
        _curve_derivatives(base, kv, cv, degree, degree, work, derivs)
        for m in range(degree + 1):
            scale = width**m / fact[m]
            for dim in range(dims):
                coef[m, dim] = derivs[m, dim] * scale
        for dim in range(dims):
            coef[0, dim] -= point[dim]

        # (sum_a c_a t^a) . (sum_b b c_b t^(b - 1)): c_a . c_b lands on t^(a + b - 1)
        poly[:] = 0.0
        for a in range(degree + 1):
            for b in range(1, degree + 1):
                dot = 0.0
                for dim in range(dims):
                    dot += coef[a, dim] * coef[b, dim]
                poly[a + b - 1] += b * dot

        # Scaled so the largest is 1, then in Bernstein form, whose sign changes
        # bound the roots in [0, 1]: 0 means none inside, 1 exactly one; 2 more
        # when a coefficient is within 1e-12 of the largest from zero (a root
        # may hide there)
        largest = 0.0
        for c in range(size + 1):
            largest = max(largest, abs(poly[c]))
        largest = max(largest, 1e-300)
        poly /= largest
        for c in range(size + 1):
            total = 0.0
            for r in range(size + 1):
                total += poly[r] * to_bern[r, c]
            bern[c] = total
        signs   = 0
        largest = 0.0
        for c in range(size + 1):
            largest = max(largest, abs(bern[c]))
            if c > 0 and (bern[c] >= 0.0) != (bern[c - 1] >= 0.0):
                signs += 1
        for c in range(size + 1):
            if abs(bern[c]) < 1e-12 * largest:
                signs += 2
                break

        # Several possible minima: every real root of the polynomial in the
        # piece, polished by Newton
        if signs > 1:
            found = _polynomial_roots(poly, companion, real, imag)
            for index in range(found):
                best = _score_newton(point, start + width * real[index], start, end,
                                     True, 2 * group + at + index, group, best, kv, cv,
                                     degree, periodic, work, derivs)

        # A lone minimum: the slope starts negative and either crosses zero
        # once, or the curve says it ends positive. Newton from where the
        # Bernstein control polygon crosses zero, or mid-piece
        slope_end = _distance(point, end, kv, cv, degree, periodic, work, derivs, True)
        if bern[0] < 0.0 and (signs == 1 or slope_end > 0.0):
            cross = 0
            for c in range(1, size + 1):
                if bern[c] >= 0.0:
                    cross = c - 1
                    break
            den = 1.0
            if bern[cross] != bern[cross + 1]:
                den = bern[cross] - bern[cross + 1]
            t = (cross + bern[cross] / den) / size if signs == 1 else 0.5
            best = _score_newton(point, start + width * t, start, end, False,
                                 2 * group + kinds + at, group, best, kv, cv, degree,
                                 periodic, work, derivs)

        # The expansion lost the far end (its last sign disagrees with the
        # curve's, or is near zero: a long piece of a fast curve): the curve's
        # own slope signs at 17 evenly spaced points bracket the minima
        disagree = np.sign(bern[size]) != np.sign(slope_end)
        if disagree or abs(bern[size]) < 1e-8 * largest:
            prev_t = start
            prev_slope = _distance(point, start, kv, cv, degree, periodic, work, derivs,
                                   True)
            for k in range(1, 17):
                t = start + width * (k / 16.0)
                slope = _distance(point, t, kv, cv, degree, periodic, work, derivs,
                                  True)
                if not prev_slope >= 0.0 and slope >= 0.0:
                    best = _score_newton(point, (prev_t + t) / 2.0, prev_t, t, False,
                                         2 * group + 2 * kinds + at + k - 1, group,
                                         best, kv, cv, degree, periodic, work, derivs)
                prev_t     = t
                prev_slope = slope

    # Wild control points: at the float level the curve is noise, so try the
    # 16 floats either side of the best (np.spacing: the gap to the next float
    # away from zero), again around a closer one found there, at most 8 times:
    # which float Newton stopped on no longer decides which noise dip is in reach
    best_x    = best[0]
    best_dist = best[1]
    if wild:
        for _ in range(8):
            away        = np.inf if best_x >= 0.0 else -np.inf
            gap         = np.nextafter(best_x, away) - best_x
            center      = best_x
            center_dist = best_dist
            best_dist   = np.inf
            for step in range(-16, 17):
                x = min(max(center + gap * step, lo), hi)
                dist = _distance(point, x, kv, cv, degree, periodic, work, derivs,
                                 False)
                if dist < best_dist:
                    best_x    = x
                    best_dist = dist
            if not best_dist < center_dist:
                break

    return best_x, best_dist


@njit(cache=True)
def _unjam(u, dist, dirty, stopped, free, wide, points, grid, kv, cv, degree, periodic,
           tol, work, derivs, min_run, chunk):
    """
    Re-assigns runs of stuck points at once, by an ordered DP.

    Parameters
    ----------
    u, dist : np.ndarray
        Current native parameters and distances (n,), float64.
    dirty, stopped, free : np.ndarray
        (n,) bool: points to revisit; points that moved this round and were
        stopped; points allowed to move (not the ends of open curves).
    wide : int
        How far to widen each run, in its own lengths.
    points, grid : np.ndarray
        Fit points (n, dims); the spots to try, the lap grid; float64.
    kv, cv, degree, periodic, work, derivs
        As in ``_distance``.
    tol : float
        Float noise of a distance.
    min_run, chunk : int
        Shortest run worth re-assigning; most points re-assigned at once.

    Returns
    -------
    None
        u and dist change where points moved; around each chunk (and one
        neighbour either side) the free points are marked dirty.
    """
    # A leg that has to slide along the curve moves one point per round, each
    # stopped by its neighbour. Here every run of stopped points (and the
    # points tied with them) is widened by its own length x wide either side,
    # overlapping runs merge, and each run of at least min_run points is
    # re-assigned in one go, in chunks of at most chunk points
    n      = u.shape[0]
    dims   = points.shape[1]
    period = float(cv.shape[0] - degree)
    shift  = 1 if periodic else 0

    # Mark the stopped points and those tied with them (same u). Index p + 1
    # is point p: index 0 and n + 1 are the anchors either side
    marked = np.zeros(n + 2, dtype=np.bool_)
    first  = 0
    while first < n:
        last = first + 1
        while last < n and u[last] == u[last - 1]:
            last += 1
        if stopped[first:last].any():
            marked[first + 1 : last + 1] = free[first:last]
        first = last

    # Widen each run by its length x wide, never past an open curve's ends
    # (first free point at index 2, last at n - 1), and merge
    cover = np.zeros(n + 2, dtype=np.int64)
    p     = 1
    while p <= n:
        if marked[p] and not marked[p - 1]:
            q = p
            while marked[q]:
                q += 1
            reach = wide * (q - p)
            cover[max(p - reach, 2 - shift)] += 1
            cover[min(q + reach, n + shift)] -= 1
            p = q
        p += 1
    for p in range(1, n + 2):
        cover[p] += cover[p - 1]

    # Each chunk: the cheapest non-decreasing assignment of its points to the
    # spots between its two neighbours (grid samples, the neighbours and the
    # points' own u), by sum of squared distances. A point may only take a
    # spot strictly closer than its current one (by 1e-10 relative, tol, and
    # 1e-14 of the squared sizes for the expansion's rounding), or keep its
    # own. A chunk touching a u or a distance that is not finite is left alone
    p = 1
    while p <= n:
        if cover[p] > 0 and cover[p - 1] <= 0:
            run_end = p
            while cover[run_end] > 0:
                run_end += 1
            if run_end - p >= min_run:
                # The chunk: indices a .. b - 1, neighbours a - 1 and b
                for a in range(p, run_end, chunk):
                    b = min(a + chunk, run_end)
                    m = b - a

                    # The neighbours' u; past the ends of the point list, the
                    # other end's u a lap away (periodic) or the end itself
                    before = u[n - 1] - period if periodic else u[0]
                    after  = u[0] + period if periodic else u[n - 1]
                    left   = before if a - 1 == 0 else u[a - 2]
                    right  = after if b == n + 1 else u[b - 1]
                    usable = np.isfinite(left) and np.isfinite(right)
                    for j in range(a - 1, b - 1):
                        usable = usable and np.isfinite(u[j]) and np.isfinite(dist[j])
                    if not usable:
                        continue

                    # Spots: grid samples strictly between the neighbours, the
                    # neighbours and the chunk's own u
                    values = np.empty(grid.shape[0] + m + 2)
                    k      = 0
                    for g in range(grid.shape[0]):
                        if left < grid[g] < right:
                            values[k] = grid[g]
                            k += 1
                    values[k]                 = left
                    values[k + 1 : k + 1 + m] = u[a - 1 : b - 1]
                    values[k + 1 + m]         = right
                    spots                     = np.unique(values[: k + m + 2])
                    count                     = spots.shape[0]

                    # Squared distances by expansion around the chunk's mean
                    # point: |P - Q|^2 = |P|^2 + |Q|^2 - 2 P.Q
                    center = np.zeros(dims)
                    for j in range(m):
                        for dim in range(dims):
                            center[dim] += points[a - 1 + j, dim]
                    center /= m

                    curve = np.empty((count, dims))
                    size2 = np.empty(count)
                    for x in range(count):
                        at = spots[x] % period if periodic else spots[x]
                        _curve_derivatives(at, kv, cv, degree, 0, work, derivs)
                        total = 0.0
                        for dim in range(dims):
                            curve[x, dim] = derivs[0, dim] - center[dim]
                            total += curve[x, dim] * curve[x, dim]
                        size2[x] = total

                    # cost[x] = cheapest assignment of the points so far with the
                    # last one at spot x; back[j, x] = where point j - 1 sits on
                    # the cheapest way for point j to reach spot x (the first
                    # cheapest spot at or before x)
                    cost     = np.empty(count)
                    cost_new = np.empty(count)
                    back     = np.empty((m, count), dtype=np.int32)
                    offset   = np.empty(dims)
                    for x in range(count):
                        cost[x] = np.inf if spots[x] > left else 0.0

                    for j in range(m):
                        point = a - 1 + j
                        own   = np.searchsorted(spots, u[point])
                        total = 0.0
                        for dim in range(dims):
                            offset[dim] = points[point, dim] - center[dim]
                            total += offset[dim] * offset[dim]
                        limit  = max(dist[point] * (1.0 - 1e-10) - tol, 0.0) ** 2

                        low    = cost[0]
                        low_at = 0
                        for x in range(count):
                            if cost[x] < low:
                                low    = cost[x]
                                low_at = x
                            back[j, x] = low_at

                            both = total + size2[x]
                            dot  = 0.0
                            for dim in range(dims):
                                dot += offset[dim] * curve[x, dim]
                            here = both - 2.0 * dot
                            if here >= limit - 1e-14 * both:  # only strictly closer
                                here = np.inf
                            if x == own:  # or where it is now
                                here = dist[point] ** 2
                            cost_new[x] = here + low
                        cost, cost_new = cost_new, cost

                    # Way back from the first cheapest end
                    end = 0
                    for x in range(1, count):
                        if cost[x] < cost[end]:
                            end = x
                    seat        = np.empty(m, dtype=np.int64)
                    seat[m - 1] = end
                    for j in range(m - 1, 0, -1):
                        seat[j - 1] = back[j, seat[j]]

                    for j in range(m):
                        point = a - 1 + j
                        moved = spots[seat[j]]
                        if moved != u[point]:
                            u[point] = moved
                            dist[point] = _distance(points[point], moved, kv, cv,
                                                    degree, periodic, work, derivs,
                                                    False)

                    for q in range(a - 2, b):
                        if free[q % n]:
                            dirty[q % n] = True
            p = run_end
        p += 1


@njit(cache=True)
def _ordered_closest_params(points, u, kv, cv, degree, periodic, rounds=200, samples=32,
                            min_run=4, chunk=1024, jam_every=4):
    """
    Repairs best-fit u: each point moves only to a strictly closer spot, in order.

    Parameters
    ----------
    points : np.ndarray
        Fit points (n, dims), float64.
    u : np.ndarray
        Starting native parameters (n,), float64, ordered; unwrapped on
        periodic curves (within one lap of each other).
    kv, cv, degree, periodic
        The curve, as in ``_curve_derivatives``; degree 1 or more.
    rounds, samples, min_run, chunk, jam_every : int
        Most rounds; grid samples per knot span, shortest run and most points
        at once for ``_unjam``; rounds between ``_unjam`` passes.

    Returns
    -------
    np.ndarray
        Repaired native parameters (n,), float64. Periodic ones stay unwrapped
        (points that never moved keep their float); a point tied with point 0
        across the seam takes point 0's wrapped value.
    """
    # Starts from u (fit()'s partitioned Newton). Red-black rounds over the
    # dirty points (all free points first, then the neighbours of every point
    # that moved): each searches the whole window between its neighbours with
    # _window_closest and moves when strictly closer (by 1e-10 relative and
    # the float noise of a distance). Every jam_every rounds, runs of points
    # stopped by a neighbour (later: of points that moved) are re-assigned at
    # once by _unjam. Stops when nothing is dirty, or after rounds rounds.
    #
    # A start u that is not finite (or beyond 2 ** 31 laps) never moves, and
    # neither do its neighbours, whose window it bounds; points or control
    # points that are not all finite leave every u as it is
    n      = points.shape[0]
    dims   = points.shape[1]
    spans  = cv.shape[0] - degree
    period = float(spans)
    u      = u.copy()
    if n == 0:
        return u

    size = np.abs(points).max()
    if not (np.isfinite(size) and np.isfinite(np.abs(cv).max())):
        return u
    tol   = 1e-15 * size                # float noise of a distance
    wild  = np.abs(cv).max() > 1e5 * size
    reach = 2.0**31 * max(period, 1.0)  # farthest u the lap arithmetic takes

    # The slope polynomial (C - p).C' has terms = 2 x degree coefficients. Its
    # power coefficients to Bernstein ones on [0, 1]: b = a @ to_bern, row r,
    # column c holding C(c, r) / C(terms - 1, r), from Pascal's triangle (exact)
    terms = 2 * degree
    binom = np.zeros((terms, terms))
    for r in range(terms):
        binom[r, 0] = 1.0
        for c in range(1, r + 1):
            binom[r, c] = binom[r - 1, c - 1] + binom[r - 1, c]
    to_bern = np.zeros((terms, terms))
    for r in range(terms):
        for c in range(r, terms):
            to_bern[r, c] = binom[c, r] / binom[terms - 1, r]
    fact = np.ones(degree + 1)  # 0! .. degree!
    for m in range(2, degree + 1):
        fact[m] = fact[m - 1] * m

    # Scratch, allocated once per fit (eigvals still allocates its own): work
    # for _curve_derivatives, derivs up to C'' for Newton, then
    # _window_closest's coef (a piece's power coefficients), poly and bern (its
    # slope polynomial in power and Bernstein form), companion (complex), real
    # and imag (its eigenvalues)
    work      = np.zeros((degree + 2, degree + 1))
    derivs    = np.zeros((max(degree, 2) + 1, dims))
    companion = np.zeros((terms - 1, terms - 1), dtype=np.complex128)
    scratch = (work, derivs, np.zeros((degree + 1, dims)), np.zeros(terms),
               np.zeros(terms), companion, np.zeros(terms - 1), np.zeros(terms - 1),
               to_bern, fact)

    # _unjam's spots: samples evenly spaced in every knot span (each span's
    # knot and samples - 1 inside it, then the last knot) of the laps -2 .. 3
    # on periodic curves (native [-2 spans, 4 spans]), of the domain on open
    knots = kv[degree : degree + spans + 1]
    first = -2 * spans if periodic else 0
    last  = 4 * spans if periodic else spans
    grid  = np.empty((last - first) * samples + 1)
    for q in range(last - first):
        knot0 = _knot_on_lap(first + q, knots, spans)
        step  = _knot_on_lap(first + q + 1, knots, spans) - knot0
        for t in range(samples):
            grid[q * samples + t] = knot0 + step * t / samples
    grid[-1] = _knot_on_lap(last, knots, spans)

    dist = np.full(n, np.nan)
    for i in range(n):
        if abs(u[i]) <= reach:
            dist[i] = _distance(points[i], u[i], kv, cv, degree, periodic, work, derivs,
                                False)

    # Open curves keep their ends. Red-black: a point's neighbours have the
    # other colour (an odd ring gives its last point a third one)
    free   = np.ones(n, dtype=np.bool_)
    colour = np.arange(n) % 2
    if periodic and n % 2 == 1:
        colour[n - 1] = 2
    if not periodic:
        free[0]     = False
        free[n - 1] = False
    dirty   = free.copy()
    stopped = np.zeros(n, dtype=np.bool_)

    for it in range(rounds):
        stopped[:] = False
        for c in range(3):
            for i in range(n):
                if not dirty[i] or colour[i] != c:
                    continue
                dirty[i] = False

                # The window: between the neighbours' current u (a lap away
                # across a periodic seam), never excluding its own
                before = u[i - 1] if i > 0 else u[n - 1] - period
                after  = u[i + 1] if i < n - 1 else u[0] + period
                if not (
                    abs(before) <= reach and abs(after) <= reach and abs(u[i]) <= reach
                ):
                    continue
                lo = min(before, u[i])
                hi = max(after, u[i])

                x, d = _window_closest(points[i], lo, hi, kv, cv, degree, periodic,
                                       wild, scratch)
                if d < dist[i] * (1.0 - 1e-10) - tol:
                    stopped[i] = it > jam_every or x == lo or x == hi
                    u[i]       = x
                    dist[i]    = d
                    dirty[(i - 1) % n] |= free[(i - 1) % n]
                    dirty[(i + 1) % n] |= free[(i + 1) % n]

        if not dirty.any():
            break
        if it % jam_every == 1:
            # 2 ** (it // jam_every), capped at n: any wider reaches both
            # ends anyway, and the product with a run's length stays small
            wide = min(2 ** (it // jam_every), n)
            _unjam(u, dist, dirty, stopped, free, wide, points, grid, kv, cv, degree,
                   periodic, tol, work, derivs, min_run, chunk)

    # A tie across the seam: point 0's value, or the float just before it,
    # so the cyclic steps add up to one lap (u that never moved for being
    # out of reach stay as they are)
    if periodic and abs(u[0]) <= reach:
        wrapped = u[0] % period
        seam    = min(wrapped, (wrapped + period) - period)
        lap_end = u[0] + period * (1 - 1e-12)
        for i in range(1, n):
            if lap_end <= u[i] <= reach:
                u[i] = seam

    return u


# --------------------------------------------------------------------------- #
#                     Closest points on the whole curve                       #
# --------------------------------------------------------------------------- #
#
# sample() finds each point's exact closest spot on the whole curve. The
# curve is cut into pieces, every knot span split evenly into 16 / spans
# pieces rounded up, at most 8 (so 16 or more in all, but 8 on a one-span
# curve and fewer where knots repeat), each held in Bezier form, built once
# per call. A piece's ends are exactly the spots _curve_derivatives (scipy's
# evaluator) gives there, so an open curve's ends tie exactly when compute()
# says they do.
# On a piece the distance slope (C - p).C' is a polynomial of degree
# 2 x degree - 1 in the piece's own t in [0, 1], in Bernstein form too: its
# coefficients are const - p . linear, both built with the piece, so a point
# costs a dot product per coefficient. Their sign changes bound the slope's
# roots: none means no root, one exactly one. A part with more is halved (de
# Casteljau) until each part holds at most one, which Newton's method finds
# without leaving the part.
#
# A piece lies inside the box around its Bezier control points, so a piece
# whose box is farther than the best spot so far cannot hold a closer one.
# The nearest box is solved first, then every other box no farther than the
# best, nearest first; long curves group their pieces by 16, whose boxes are
# tried the same way first.
#
# Distances are compared squared. So that a huge or a tiny curve neither
# overflows nor underflows them, the search runs on the curve and the points
# scaled by a power of two to a largest control point coordinate in
# [0.5, 1): exact, it changes no result of a curve that does neither.
#
# Parameters are native: the curve runs on [0, spans]. The winner is polished
# on _curve_derivatives (scipy's own evaluator), within its piece. fastmath
# stays off: ties are decided on exactly equal distances.


@njit(cache=True)
def _bezier_pieces(kv, cv, degree, per_span):
    """
    The curve cut into pieces, each in Bezier form, for ``_closest_points``.

    Parameters
    ----------
    kv, cv, degree
        The curve, as in ``_curve_derivatives``; degree 1 or more, no knot
        inside the domain repeated past degree.
    per_span : int
        Pieces per knot span, cut evenly; empty spans have none.

    Returns
    -------
    tuple
        (windows, bezier, const, linear, lower, upper): windows (pieces, 2)
        each piece's native start and end, in order; bezier (pieces,
        degree + 1, dims) its Bezier control points, the first and last the
        curve at its start and end as ``_curve_derivatives`` gives; const
        (pieces, 2 x degree) and linear (pieces, 2 x degree, dims) the
        Bernstein coefficients of its distance slope at a point p,
        const - p . linear; lower and upper (pieces, dims) the box around its
        control points, padded by 1e-12 x the largest control point
        coordinate.
    """
    spans = cv.shape[0] - degree
    dims  = cv.shape[1]
    knots = kv[degree : degree + spans + 1]
    count = 0
    for s in range(spans):
        if knots[s + 1] > knots[s]:
            count += per_span

    # Pascal's triangle up to row 2 x degree (exact), and 0! .. degree!
    binom = np.zeros((2 * degree + 1, 2 * degree + 1))
    for r in range(2 * degree + 1):
        binom[r, 0] = 1.0
        for c in range(1, r + 1):
            binom[r, c] = binom[r - 1, c - 1] + binom[r - 1, c]
    fact = np.ones(degree + 1)
    for m in range(2, degree + 1):
        fact[m] = fact[m - 1] * m

    windows = np.empty((count, 2))
    bezier  = np.empty((count, degree + 1, dims))
    const   = np.zeros((count, 2 * degree))
    linear  = np.zeros((count, 2 * degree, dims))
    lower   = np.empty((count, dims))
    upper   = np.empty((count, dims))
    work    = np.zeros((degree + 2, degree + 1))
    derivs  = np.zeros((degree + 1, dims))
    power   = np.zeros((degree + 1, dims))
    pad     = 1e-12 * np.abs(cv).max() + 1e-300

    piece   = 0
    for s in range(spans):
        width = knots[s + 1] - knots[s]
        if not width > 0.0:
            continue
        for p in range(per_span):
            start = knots[s] + width * p / per_span
            end   = knots[s] + width * (p + 1) / per_span
            if p == per_span - 1:
                end = knots[s + 1]
            windows[piece, 0] = start
            windows[piece, 1] = end

            # On a knot span the Taylor expansion C(start + t w) =
            # sum_m C^(m)(start) (t w)^m / m!, w the piece's width, is exact.
            # Its power coefficients a_m give the Bezier control points
            # b_i = sum_(m <= i) C(i, m) / C(degree, m) a_m
            _curve_derivatives(start, kv, cv, degree, degree, work, derivs)
            for m in range(degree + 1):
                scale = (end - start) ** m / fact[m]
                for dim in range(dims):
                    power[m, dim] = derivs[m, dim] * scale
            for i in range(degree + 1):
                for dim in range(dims):
                    total = 0.0
                    for m in range(i + 1):
                        total += binom[i, m] / binom[degree, m] * power[m, dim]
                    bezier[piece, i, dim] = total

            # The first control point is C(start) exactly, the last one a sum
            # that can round a step off C(end): it is replaced by C(end), so
            # that piece ends are measured where scipy puts them and an open
            # curve's ends tie exactly when compute() says they do. At a knot
            # C(end) is the next span's start, the same spot: the curve's
            # setup rejects a knot repeated past degree, where it would jump
            _curve_derivatives(end, kv, cv, degree, 0, work, derivs)
            for dim in range(dims):
                bezier[piece, degree, dim] = derivs[0, dim]

            # dC/dt = degree x sum_j B_j(t) (b_(j + 1) - b_j), B_j of degree
            # degree - 1. A product of Bernstein polynomials is one of degree
            # 2 x degree - 1: B_i B_j = C(degree, i) C(degree - 1, j) /
            # C(2 x degree - 1, i + j) B_(i + j), so (b_i - p).(b_(j + 1) - b_j)
            # lands on coefficient i + j
            for i in range(degree + 1):
                for j in range(degree):
                    w = (degree * binom[degree, i] * binom[degree - 1, j]
                         / binom[2 * degree - 1, i + j])
                    for dim in range(dims):
                        step = w * (bezier[piece, j + 1, dim] - bezier[piece, j, dim])
                        const[piece, i + j]       += bezier[piece, i, dim] * step
                        linear[piece, i + j, dim] += step

            # The piece lies inside the box around its control points
            for dim in range(dims):
                low  = bezier[piece, 0, dim]
                high = low
                for i in range(1, degree + 1):
                    low  = min(low, bezier[piece, i, dim])
                    high = max(high, bezier[piece, i, dim])
                lower[piece, dim] = low - pad
                upper[piece, dim] = high + pad
            piece += 1

    return windows, bezier, const, linear, lower, upper


@njit(cache=True)
def _group_boxes(lower, upper, size):
    """
    The box around each run of ``size`` pieces, the last run maybe shorter.

    Parameters
    ----------
    lower, upper : np.ndarray
        The pieces' boxes (pieces, dims), from ``_bezier_pieces``.
    size : int
        Pieces per group, 1 or more.

    Returns
    -------
    tuple
        (lower, upper) of the groups' boxes, (groups, dims) each.
    """
    count  = lower.shape[0]
    groups = (count + size - 1) // size
    low    = np.empty((groups, lower.shape[1]))
    high   = np.empty((groups, lower.shape[1]))
    for g in range(groups):
        for dim in range(lower.shape[1]):
            low[g, dim]  = lower[g * size, dim]
            high[g, dim] = upper[g * size, dim]
            for b in range(g * size + 1, min(count, (g + 1) * size)):
                low[g, dim]  = min(low[g, dim], lower[b, dim])
                high[g, dim] = max(high[g, dim], upper[b, dim])
    return low, high


@njit(cache=True, inline="always")
def _box_distance(point, lower, upper, index):
    """
    Squared distance from a point to box ``index``; 0 inside it.

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    lower, upper : np.ndarray
        Boxes (boxes, dims).
    index : int
        The box.

    Returns
    -------
    float
        The squared distance.
    """
    total = 0.0
    for dim in range(point.shape[0]):
        gap = 0.0
        if point[dim] < lower[index, dim]:
            gap = lower[index, dim] - point[dim]
        elif point[dim] > upper[index, dim]:
            gap = point[dim] - upper[index, dim]
        total += gap * gap
    return total


@njit(cache=True, inline="always")
def _rank(t, wrap):
    """
    A spot's rank on a piece, for ties: t, or t + 2 before ``wrap``.

    Parameters
    ----------
    t : float
        Where on the piece, in [0, 1].
    wrap : float
        Where the user's parameter range starts inside the piece, or 0.0.

    Returns
    -------
    float
        The rank, the smaller the earlier in the user's range.
    """
    return t if t >= wrap else t + 2.0


@njit(cache=True, inline="always")
def _closer(dist, rank, best, best_rank):
    """
    Whether a spot beats the best so far: closer, or as close and ranked first.

    Parameters
    ----------
    dist, rank : float
        The spot's squared distance and rank (the smaller the earlier).
    best, best_rank : float
        The best's.

    Returns
    -------
    bool
        True when the spot wins.
    """
    return dist < best or (dist == best and rank < best_rank)


@njit(cache=True)
def _bezier_distance(point, bezier, piece, t, work):
    """
    Squared distance from a point to a piece at t, by de Casteljau.

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    bezier : np.ndarray
        The pieces' Bezier control points, from ``_bezier_pieces``.
    piece : int
        The piece.
    t : float
        Where on the piece, in [0, 1].
    work : np.ndarray
        Scratch (degree + 1,) or more.

    Returns
    -------
    float
        The squared distance.
    """
    degree = bezier.shape[1] - 1
    total  = 0.0
    for dim in range(point.shape[0]):
        for i in range(degree + 1):
            work[i] = bezier[piece, i, dim]
        for r in range(1, degree + 1):
            for i in range(degree + 1 - r):
                work[i] = (1.0 - t) * work[i] + t * work[i + 1]
        offset = work[0] - point[dim]
        total += offset * offset
    return total


@njit(cache=True)
def _bernstein(coef, t, work):
    """
    A Bernstein polynomial's value and derivative at t, by de Casteljau.

    Parameters
    ----------
    coef : np.ndarray
        Its coefficients (degree + 1,), degree 1 or more.
    t : float
        Where, in [0, 1].
    work : np.ndarray
        Scratch (degree + 1,) or more.

    Returns
    -------
    tuple
        (value, derivative).
    """
    size = coef.shape[0] - 1
    for i in range(size + 1):
        work[i] = coef[i]
    for r in range(1, size):
        for i in range(size + 1 - r):
            work[i] = (1.0 - t) * work[i] + t * work[i + 1]
    return (1.0 - t) * work[0] + t * work[1], size * (work[1] - work[0])


@njit(cache=True)
def _piece_closest(point, piece, bezier, const, linear, wrap, coef, parts, ends,
                   work):
    """
    A point's exact closest spot on one piece.

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    piece : int
        The piece.
    bezier, const, linear : np.ndarray
        The pieces, from ``_bezier_pieces``.
    wrap : float
        Where the user's parameter range starts inside the piece, as a t (a
        registered periodic curve's start); 0.0 when it does not: on a tie,
        spots before it rank after the others.
    coef, parts, ends, work : np.ndarray
        Scratch: coef (2 x degree,); parts (depth, 2 x degree) and ends
        (depth, 2), a stack of parts still to search; work (2 x degree,).

    Returns
    -------
    tuple
        (t, squared distance) of the closest spot, t in [0, 1] across the
        piece; on an exact tie the one ranked first.
    """
    size = coef.shape[0] - 1  # the slope's degree

    # The slope's Bernstein coefficients at this point
    for c in range(size + 1):
        value = const[piece, c]
        for dim in range(point.shape[0]):
            value -= point[dim] * linear[piece, c, dim]
        coef[c] = value

    # The piece's ends always compete: its first and last control points
    degree = bezier.shape[1] - 1
    best_t = 0.0
    best   = 0.0
    dist   = 0.0
    for dim in range(point.shape[0]):
        best += (bezier[piece, 0, dim] - point[dim]) ** 2
        dist += (bezier[piece, degree, dim] - point[dim]) ** 2
    if _closer(dist, _rank(1.0, wrap), best, _rank(0.0, wrap)):
        best_t = 1.0
        best   = dist

    # A stack of parts of [0, 1] still to search: parts[row] the slope's
    # coefficients over the part, ends[row] its start and end
    parts[0, :] = coef
    ends[0, 0]  = 0.0
    ends[0, 1]  = 1.0
    top         = 1
    while top > 0:
        top -= 1
        lo = ends[top, 0]
        hi = ends[top, 1]

        # The coefficients' sign changes (zeros skipped): the roots in the
        # part, or more by an even number
        changes = 0
        last    = 0.0
        for c in range(size + 1):
            value = parts[top, c]
            if value != 0.0:
                if last != 0.0 and (value > 0.0) != (last > 0.0):
                    changes += 1
                last = value
        if changes == 0:
            continue

        # Several: halve the part by de Casteljau, the left half on top. The
        # middle competes when the slope is exactly zero there (neither half
        # sees that root)
        if changes > 1 and hi - lo > 1e-12 and top + 2 <= parts.shape[0]:
            mid = 0.5 * (lo + hi)
            for c in range(size + 1):
                work[c] = parts[top, c]
            parts[top + 1, 0] = work[0]
            parts[top, size]  = work[size]
            for r in range(1, size + 1):
                for i in range(size + 1 - r):
                    work[i] = 0.5 * (work[i] + work[i + 1])
                parts[top + 1, r]    = work[0]
                parts[top, size - r] = work[size - r]
            ends[top + 1, 0] = lo
            ends[top + 1, 1] = mid
            ends[top, 0]     = mid
            ends[top, 1]     = hi
            if parts[top, 0] == 0.0:
                dist = _bezier_distance(point, bezier, piece, mid, work)
                if _closer(dist, _rank(mid, wrap), best, _rank(best_t, wrap)):
                    best_t = mid
                    best   = dist
            top += 2
            continue

        # One root (or a part too small to halve): a minimum when the slope
        # goes from negative to positive. Newton's method from where the
        # coefficients' polygon crosses zero, each step kept inside a bracket
        # that shrinks to the side the slope points at. A step too small to
        # move t means t is the root: halving the bracket then would only
        # walk back to it
        if changes == 1 and not (parts[top, 0] <= 0.0 <= parts[top, size]):
            continue
        t = 0.5 * (lo + hi)
        for c in range(size):
            if parts[top, c] < 0.0 <= parts[top, c + 1]:
                cross = (c + parts[top, c] / (parts[top, c] - parts[top, c + 1])) / size
                t     = lo + (hi - lo) * cross
                break
        low  = lo
        high = hi
        for _ in range(64):
            slope, bend = _bernstein(coef, t, work)
            if slope < 0.0:
                low = t
            elif slope > 0.0:
                high = t
            else:
                break
            step = t - slope / bend if bend > 0.0 else -1.0
            if not abs(step - t) > 4e-16:
                break
            if not low < step < high:
                step = 0.5 * (low + high)
            moved = abs(step - t)
            t     = step
            if not moved > 4e-16:
                break
        dist = _bezier_distance(point, bezier, piece, t, work)
        if _closer(dist, _rank(t, wrap), best, _rank(best_t, wrap)):
            best_t = t
            best   = dist

    return best_t, best


@njit(cache=True)
def _polish(point, x, lo, hi, kv, cv, degree, periodic, work, derivs, spot, tangent):
    """
    Newton's method on the curve's own evaluator, from a spot found in Bezier form.

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    x : float
        The spot, native.
    lo, hi : float
        Its piece's window; every step stays inside it.
    kv, cv, degree, periodic
        The curve, as in ``_curve_derivatives``.
    work, derivs : np.ndarray
        Scratch for ``_curve_derivatives``; derivs (3, dims) or more.
    spot, tangent : np.ndarray
        Receive the curve and its derivative at the result (dims,), as
        scipy's.

    Returns
    -------
    tuple
        (x, squared distance): x, or the last of up to 3 steps from it while
        each came strictly closer, and the point's squared distance there.
    """
    period = float(cv.shape[0] - degree)
    best   = 0.0
    best_x = x
    y      = x
    for k in range(4):
        _curve_derivatives(y % period if periodic else y, kv, cv, degree, 2, work,
                           derivs)
        dist = 0.0
        for dim in range(point.shape[0]):
            offset = derivs[0, dim] - point[dim]
            dist += offset * offset

        # x itself is always kept, so spot and tangent are always written
        if k > 0 and not dist < best:
            break
        best_x     = y
        best       = dist
        spot[:]    = derivs[0]
        tangent[:] = derivs[1]

        # slope = (C - p).C', bend = its derivative |C'|^2 + (C - p).C''
        slope = 0.0
        bend  = 0.0
        for dim in range(point.shape[0]):
            offset = derivs[0, dim] - point[dim]
            slope += offset * derivs[1, dim]
            bend  += derivs[1, dim] * derivs[1, dim] + offset * derivs[2, dim]
        if not bend > 0.0:
            break
        y = min(max(y - slope / bend, lo), hi)
        if y == best_x:
            break

    return best_x, best


@njit(cache=True, inline="always")
def _piece_param(windows, piece, t):
    """
    The native parameter at t on a piece; t = 1 is exactly the piece's end.

    Parameters
    ----------
    windows : np.ndarray
        The pieces' native start and end (pieces, 2).
    piece : int
        The piece.
    t : float
        Where on the piece, in [0, 1].

    Returns
    -------
    float
        The native parameter.
    """
    if t == 1.0:
        return windows[piece, 1]
    return windows[piece, 0] + t * (windows[piece, 1] - windows[piece, 0])


@njit(cache=True)
def _search_pieces(point, pieces, size, start, period, periodic, scratch):
    """
    One point's exact closest spot on the curve, among its pieces.

    Parameters
    ----------
    point : np.ndarray
        The point (dims,), float64.
    pieces : tuple
        (windows, bezier, const, linear, lower, upper, group_lower,
        group_upper, wrap): ``_bezier_pieces``' arrays, the boxes of the
        groups of ``size`` pieces, and where the user's range starts inside
        each piece (``_piece_closest``'s wrap).
    size : int
        Pieces per group.
    start, period : float
        Where the user's range starts, native, and the curve's period
        (periodic curves).
    periodic : bool
        Whether the curve is periodic.
    scratch : tuple
        (group_bound, bound, order, coef, parts, ends, work): a float per
        group, a float and an int per piece, then ``_piece_closest``'s
        scratch.

    Returns
    -------
    tuple
        (piece, x): the closest spot's piece and native parameter; on an
        exact tie the spot first in the user's range.
    """
    windows, bezier, const, linear, lower, upper = pieces[:6]
    group_lower, group_upper, wrap = pieces[6:]
    group_bound, bound, order, coef, parts, ends, work = scratch
    count = windows.shape[0]

    # A first best: the nearest box of the nearest group
    near = 0
    for g in range(group_lower.shape[0]):
        group_bound[g] = _box_distance(point, group_lower, group_upper, g)
        if group_bound[g] < group_bound[near]:
            near = g
    first = near * size
    for b in range(near * size, min(count, (near + 1) * size)):
        bound[b] = _box_distance(point, lower, upper, b)
        if bound[b] < bound[first]:
            first = b
    t, best = _piece_closest(point, first, bezier, const, linear, wrap[first], coef,
                             parts, ends, work)
    best_b = first
    best_x = _piece_param(windows, first, t)

    # Every other box no farther than the best, nearest first (insertion
    # sort: few are left), while the next one is still no farther. A box
    # exactly as far as the best may hold a spot exactly as close, a tie:
    # rounding never puts a box farther than a spot inside it, the box's
    # distance being summed the same way from coordinates no farther out
    found = 0
    for g in range(group_lower.shape[0]):
        if group_bound[g] > best:
            continue
        for b in range(g * size, min(count, (g + 1) * size)):
            bound[b] = _box_distance(point, lower, upper, b)
            if b == first or bound[b] > best:
                continue
            at = found
            while at > 0 and bound[order[at - 1]] > bound[b]:
                order[at] = order[at - 1]
                at -= 1
            order[at] = b
            found += 1
    for k in range(found):
        b = order[k]
        if bound[b] > best:
            break
        t, dist = _piece_closest(point, b, bezier, const, linear, wrap[b], coef, parts,
                                 ends, work)
        x = _piece_param(windows, b, t)

        # On a tie the spot first in the user's range wins: on a periodic
        # curve, the one the least far past its start
        rank      = (x - start) % period if periodic else x
        best_rank = (best_x - start) % period if periodic else best_x
        if _closer(dist, rank, best, best_rank):
            best_b = b
            best_x = x
            best   = dist

    return best_b, best_x


@njit(parallel=True, cache=True)
def _closest_points(points, kv, cv, degree, periodic, origin, chunk=16):
    """
    Each point's exact closest spot on the whole curve, in parallel.

    Parameters
    ----------
    points : np.ndarray
        The points (n, dims), float64, finite. Scaled with the curve, every
        coordinate under 2^500 (sample() checks), so that no squared distance
        overflows.
    kv, cv, degree, periodic
        The curve, as in ``_curve_derivatives``; degree 1 or more, its domain
        longer than zero.
    origin : float
        Where the user's parameter range starts on a periodic curve, native
        (a registered curve's offset, else 0.0); unused on open curves.
    chunk : int
        Points per parallel task, which allocates its scratch once.

    Returns
    -------
    tuple
        (u, spots, tangents, distances): native parameters (n,), in
        [0, spans], periodic ones in [0, spans); the curve there (n, dims)
        and its derivative (n, dims), as scipy's; each point's distance to
        its spot (n,). On an exact tie, the spot first in the user's range.
    """
    n         = points.shape[0]
    dims      = points.shape[1]
    spans     = cv.shape[0] - degree
    period    = float(spans)
    u         = np.empty(n)  # np.zeros would start a parallel loop of its own
    spots     = np.empty((n, dims))
    tangents  = np.empty((n, dims))
    distances = np.empty(n)

    # The curve scaled by 2^-exponent, its largest coordinate in [0.5, 1); the
    # points are scaled the same way, one at a time
    exponent = math.frexp(np.abs(cv).max())[1]
    scaled   = np.empty((cv.shape[0], dims))
    for i in range(cv.shape[0]):
        for dim in range(dims):
            scaled[i, dim] = math.ldexp(cv[i, dim], -exponent)

    # Every span cut into 16 / spans pieces rounded up, at most 8; groups of
    # 16 pieces on curves of more than 32, else one group
    per_span = min(8, max(1, (16 + spans - 1) // spans))
    windows, bezier, const, linear, lower, upper = _bezier_pieces(kv, scaled, degree,
                                                                  per_span)
    count = windows.shape[0]
    size  = 16 if count > 32 else count
    group_lower, group_upper = _group_boxes(lower, upper, size)

    # On a periodic curve the user's range starts at origin: a spot's rank on
    # a tie is how far past it the spot is, and inside the piece holding it,
    # spots before it rank after the others
    start = origin % period if periodic else 0.0
    wrap  = np.empty(count)
    for b in range(count):
        wrap[b] = 0.0
        if windows[b, 0] < start < windows[b, 1]:
            wrap[b] = (start - windows[b, 0]) / (windows[b, 1] - windows[b, 0])
    pieces = (windows, bezier, const, linear, lower, upper, group_lower, group_upper,
              wrap)

    # Each task's scratch. The stack of parts is 96 deep: halving stops at
    # parts 1e-12 wide, 40 halvings, and each adds one part to the stack
    blocks = (n + chunk - 1) // chunk
    for block in prange(blocks):
        scratch = (
            np.empty(group_lower.shape[0]),
            np.empty(count),
            np.empty(count, dtype=np.int64),
            np.empty(2 * degree),
            np.empty((96, 2 * degree)),
            np.empty((96, 2)),
            np.empty(2 * degree),
        )
        work   = np.zeros((degree + 2, degree + 1))
        derivs = np.zeros((max(degree, 2) + 1, dims))
        point  = np.empty(dims)
        for i in range(block * chunk, min(n, (block + 1) * chunk)):
            for dim in range(dims):
                point[dim] = math.ldexp(points[i, dim], -exponent)
            b, x = _search_pieces(point, pieces, size, start, period, periodic, scratch)

            # The winner, polished within its piece, then scaled back
            x, dist = _polish(point, x, windows[b, 0], windows[b, 1], kv, scaled, degree,
                              periodic, work, derivs, spots[i], tangents[i])
            for dim in range(dims):
                spots[i, dim]    = math.ldexp(spots[i, dim], exponent)
                tangents[i, dim] = math.ldexp(tangents[i, dim], exponent)
            distances[i] = math.ldexp(math.sqrt(dist), exponent)
            if periodic:
                x = x % period
                if x >= period:
                    x = 0.0
            u[i] = x

    return u, spots, tangents, distances


# --------------------------------------------------------------------------- #
#            2D Newton-Raphson closest point (surface)                         #
# --------------------------------------------------------------------------- #
#
# BSplinePatchData.sample()'s search. From each seed, a Newton search on the
# squared distance that only ever takes a step that comes closer. An open
# edge stops a parameter: one whose slope points out of the patch stays on
# the edge while the other slides along it, the closest point of that edge's
# iso-curve; a corner where both point out is the answer.
#
# A degree-1 direction folds on every knot (the patch is only C0 there): the
# search keeps to one span of it at a time, its ends stopping the parameter
# like edges, and crosses into the next span when the slope there points on.
# A span of no width (two rows the same, within roundoff) is one more such
# line: the search steps across it into the span on either side that slopes
# on down, and the other side, when it comes closer too, starts one more.
# A pole (an open edge collapsed to one point, or nearly) that a search
# reaches is left by more searches, from where along it the patch heads
# toward the query most steeply.
#
# Parameters are native: each direction runs on [0, max_param], periodic ones
# modulo it. The basis functions come from scipy's own de Boor recursion, as
# in _curve_derivatives.
#
# fastmath stays off: every step is kept only when it is strictly closer.


@njit(cache=True)
def _patch_basis(x, kv, degree, order, below, work, out):
    """
    One direction's nonzero basis functions and their derivatives at x.

    Parameters
    ----------
    x : float
        Native parameter, inside the domain [kv[degree], kv[c]].
    kv : np.ndarray
        Knot vector (c + degree + 1,) in scipy's layout, float64.
    degree, order : int
        Degree of the direction; highest derivative wanted (past degree:
        zero).
    below : bool
        On a knot, take the span below it (left-sided derivatives) instead
        of the one above.
    work, out : np.ndarray
        Scratch (degree + 1,); output (order + 1 or more, degree + 1), row m
        the m-th derivatives of the degree + 1 functions.

    Returns
    -------
    int
        The first of those functions: the control point they start at.
    """
    # Right-sided at the knots, like scipy, unless below; an empty span at
    # either end of the domain hands x to the nearest non-empty one
    limit = kv.shape[0] - 2 * degree - 2
    span  = _find_span(x, kv, degree, limit)
    if below and span > 0 and x <= kv[degree + span]:
        span -= 1
    while span > 0 and not kv[degree + span] < kv[degree + span + 1]:
        span -= 1
    while span < limit and not kv[degree + span] < kv[degree + span + 1]:
        span += 1

    # _curve_derivatives' recursion: raise the basis to degree - m, then m
    # steps differentiate it back up (highest m first, the raising shared)
    knot    = span + degree
    work[0] = 1.0
    level   = 0
    top     = min(order, degree)
    for m in range(top, -1, -1):
        while level < degree - m:
            level += 1
            carry = 0.0
            for n in range(1, level + 1):
                upper = kv[knot + n]
                lower = kv[knot + n - level]
                if upper == lower:
                    work[n - 1] = carry
                    carry       = 0.0
                    continue
                w           = work[n - 1] / (upper - lower)
                work[n - 1] = carry + w * (upper - x)
                carry       = w * (x - lower)
            work[level] = carry

        row              = out[m]
        row[: level + 1] = work[: level + 1]
        for j in range(level + 1, degree + 1):
            carry = 0.0
            for n in range(1, j + 1):
                upper = kv[knot + n]
                lower = kv[knot + n - j]
                if upper == lower:
                    row[n - 1] = carry
                    carry      = 0.0
                    continue
                w          = j * row[n - 1] / (upper - lower)
                row[n - 1] = carry - w
                carry      = w
            row[j] = carry
    out[top + 1 : order + 1] = 0.0
    return span


@njit(cache=True)
def _patch_derivatives(u, v, kv_u, kv_v, cv, du, dv, order, below_u, below_v, work,
                       basis_u, basis_v, out):
    """
    The patch and its partial derivatives at one native (u, v).

    Parameters
    ----------
    u, v : float
        Native parameters, inside the domain (periodic ones wrapped).
    kv_u, kv_v, cv, du, dv
        The patch: knot vectors in scipy's layout, control points
        (cu, cv_n, dims) wrapped on periodic directions, degrees.
    order : int
        1: position and first partials; 2: second partials too.
    below_u, below_v : bool
        Per direction: on a knot, the derivatives of the span below it
        (``_patch_basis``).
    work, basis_u, basis_v : np.ndarray
        Scratch for ``_patch_basis``: (max(du, dv) + 1,), (3, du + 1) and
        (3, dv + 1).
    out : np.ndarray
        Output (6, dims): rows S, S_u, S_v, then with order 2 S_uu, S_uv
        and S_vv.

    Returns
    -------
    None
    """
    first_u = _patch_basis(u, kv_u, du, order, below_u, work, basis_u)
    first_v = _patch_basis(v, kv_v, dv, order, below_v, work, basis_v)
    second  = order >= 2
    out[: 6 if second else 3] = 0.0
    for i in range(du + 1):
        for j in range(dv + 1):
            w00 = basis_u[0, i] * basis_v[0, j]
            w10 = basis_u[1, i] * basis_v[0, j]
            w01 = basis_u[0, i] * basis_v[1, j]
            w20 = basis_u[2, i] * basis_v[0, j] if second else 0.0
            w11 = basis_u[1, i] * basis_v[1, j] if second else 0.0
            w02 = basis_u[0, i] * basis_v[2, j] if second else 0.0
            for k in range(cv.shape[2]):
                p = cv[first_u + i, first_v + j, k]
                out[0, k] += w00 * p
                out[1, k] += w10 * p
                out[2, k] += w01 * p
                if second:
                    out[3, k] += w20 * p
                    out[4, k] += w11 * p
                    out[5, k] += w02 * p


@njit(cache=True)
def _patch_slopes(query, derivs):
    """
    The squared distance from a query to the patch and its derivatives.

    Parameters
    ----------
    query : np.ndarray
        The point (dims,), float64.
    derivs : np.ndarray
        ``_patch_derivatives``' order 2 output at (u, v).

    Returns
    -------
    tuple
        (f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv): f = |S - Q|^2;
        g = (S - Q) . S_u and . S_v, half its slopes; h its Hessian, halved;
        j the part of h without (S - Q) . S_uu and the like (S_u . S_u ...).
    """
    f = g_u = g_v = h_uu = h_uv = h_vv = j_uu = j_uv = j_vv = 0.0
    for k in range(query.shape[0]):
        r = derivs[0, k] - query[k]
        f    += r * r
        g_u  += r * derivs[1, k]
        g_v  += r * derivs[2, k]
        h_uu += r * derivs[3, k]
        h_uv += r * derivs[4, k]
        h_vv += r * derivs[5, k]
        j_uu += derivs[1, k] * derivs[1, k]
        j_uv += derivs[1, k] * derivs[2, k]
        j_vv += derivs[2, k] * derivs[2, k]
    return f, g_u, g_v, h_uu + j_uu, h_uv + j_uv, h_vv + j_vv, j_uu, j_uv, j_vv


@njit(cache=True)
def _span_exit(x, lo, g, end, periodic):
    """
    The next span along a degree-1 direction, for a search on an end of its
    span (lo, lo + 1) whose slope points past that end.

    Parameters
    ----------
    x, lo : float
        The parameter and the start of its span.
    g : float
        The squared distance's slope along the direction, halved.
    end : float
        End of the direction's domain.
    periodic : bool
        Whether the direction wraps: past either end of the domain is the
        span across the seam.

    Returns
    -------
    tuple
        (x, lo): the same spot on the next span (0 and end swap across the
        seam); lo is -1 when there is none.
    """
    if x <= lo and g > 0.0:
        if lo > 0.0:
            return x, lo - 1.0
        if periodic:
            return end, end - 1.0
    elif x >= lo + 1.0 and g < 0.0:
        if lo + 1.0 < end:
            return x, lo + 1.0
        if periodic:
            return 0.0, 0.0
    return x, -1.0


@njit(cache=True)
def _pinned(x, g, low, high, ends):
    """
    Whether a parameter sits on an end of its range (low, high) with the
    squared distance's slope pointing out of it; ends: whether the range
    has ends (a periodic direction of degree 2 or more has none).
    """
    return ends and ((x <= low and g > 0.0) or (x >= high and g < 0.0))


@njit(cache=True)
def _patch_poles(cv, periodic_u, periodic_v):
    """
    Which open edges are poles, or nearly: collapsed to one point, or to a
    spot so small next to the patch's first span off it (all of the edge's
    control points within a tenth of how far the next row's lie from them)
    that a search can stall on it.

    Parameters
    ----------
    cv : np.ndarray
        Control point grid (cu, cv_n, dims), float64.
    periodic_u, periodic_v : bool
        Whether each direction wraps (and has no edges).

    Returns
    -------
    np.ndarray
        (4,) int8 for the edges u = 0, u = end, v = 0 and v = end: 0 not a
        pole, 1 nearly one, 2 one point (its control points within 1e-12
        of that distance).
    """
    count_u, count_v, dims = cv.shape
    low   = np.empty(dims)
    high  = np.empty(dims)
    poles = np.zeros(4, dtype=np.int8)
    for e in range(4):
        if periodic_u if e < 2 else periodic_v:
            continue

        # The edge's extent, and the farthest one of its control points is
        # from the next row's
        outer   = 0 if e % 2 == 0 else (count_u if e < 2 else count_v) - 1
        inner   = 1 if e % 2 == 0 else outer - 1
        low[:]  = np.inf
        high[:] = -np.inf
        reach   = 0.0
        for j in range(count_v if e < 2 else count_u):
            if e < 2:
                point = cv[outer, j]
                after = cv[inner, j]
            else:
                point = cv[j, outer]
                after = cv[j, inner]
            gap = 0.0
            for k in range(dims):
                low[k]  = min(low[k], point[k])
                high[k] = max(high[k], point[k])
                gap += (after[k] - point[k]) ** 2
            reach = max(reach, np.sqrt(gap))
        extent = (high - low).max()
        if extent <= 1e-12 * reach:
            poles[e] = 2
        elif extent <= 1e-1 * reach:
            poles[e] = 1
    return poles


@njit(cache=True)
def _mark_pole(u, v, poles, max_u, max_v, on_poles):
    """
    The pole (0 to 3, as in ``_patch_poles``) that (u, v) lies on, flagged
    in on_poles (4,); -1 when none.
    """
    pole = -1
    if u <= 0.0 and poles[0] > 0:
        pole = 0
    elif u >= max_u and poles[1] > 0:
        pole = 1
    elif v <= 0.0 and poles[2] > 0:
        pole = 2
    elif v >= max_v and poles[3] > 0:
        pole = 3
    if pole >= 0:
        on_poles[pole] = True
    return pole


@njit(cache=True)
def _pole_slope(query, edge, x, on_u, kv_u, kv_v, cv, du, dv, work, basis_u, basis_v,
                trial):
    """
    The squared distance's slope (halved) from a pole into the patch, at x
    along the pole: negative when leaving the pole there comes closer.

    Parameters
    ----------
    query : np.ndarray
        The point (dims,), float64.
    edge : float
        The pole's parameter: 0 or its direction's end.
    x : float
        The parameter along the pole.
    on_u : bool
        Whether the pole is a u edge (x is v) or a v edge (x is u).
    kv_u, kv_v, cv, du, dv
        The patch, as in ``_patch_derivatives``.
    work, basis_u, basis_v, trial : np.ndarray
        Scratch for ``_patch_derivatives``.

    Returns
    -------
    float
    """
    if on_u:
        _patch_derivatives(edge, x, kv_u, kv_v, cv, du, dv, 1, False, False, work,
                           basis_u, basis_v, trial)
    else:
        _patch_derivatives(x, edge, kv_u, kv_v, cv, du, dv, 1, False, False, work,
                           basis_u, basis_v, trial)
    row   = 1 if on_u else 2
    slope = 0.0
    for k in range(query.shape[0]):
        slope += (trial[0, k] - query[k]) * trial[row, k]
    return slope if edge <= 0.0 else -slope


@njit(cache=True)
def _patch_pole_turns(query, edge, on_u, end, periodic, kv_u, kv_v, cv, du, dv, work,
                      basis_u, basis_v, trial, rim, starts):
    """
    Where along a pole searches leave it: where the patch heads toward the
    query most steeply, one spot per way out that comes closer, and on a
    nearly collapsed pole each spot of its rim closest to the query.

    Every spot along a pole is the same point, or nearly, so a search
    stopped there can leave it anywhere along it: 4 samples per span, then
    a golden-section search between the neighbours of each sample whose
    slope is negative and lower than theirs. A nearly collapsed pole is
    also a small rim: each sample closer to the query than its neighbours
    starts a search too, unless the rim looks like one point from the query.

    Parameters
    ----------
    query : np.ndarray
        The point (dims,), float64.
    edge : float
        The pole's parameter: 0 or its direction's end.
    on_u : bool
        Whether the pole is a u edge (it runs along v) or a v edge.
    end : float
        End of the domain along the pole.
    periodic : bool
        Whether the direction along the pole wraps.
    kv_u, kv_v, cv, du, dv
        The patch, as in ``_patch_derivatives``.
    work, basis_u, basis_v, trial : np.ndarray
        Scratch for ``_patch_derivatives``.
    rim : bool
        Whether the pole is nearly collapsed (``_patch_poles`` 1).
    starts : np.ndarray
        Output, (8 * spans + 2,) or more: the parameters along the pole,
        inside its domain, the ways out first.

    Returns
    -------
    int
        How many starts were found.
    """
    spans  = max(int(end + 0.5), 1)
    step   = end / (4 * spans)
    count  = 4 * spans + (0 if periodic else 1)
    slopes = np.empty(count)
    dists  = np.empty(count)
    for i in range(count):
        slopes[i] = _pole_slope(query, edge, i * step, on_u, kv_u, kv_v, cv, du, dv,
                                work, basis_u, basis_v, trial)
        dists[i] = 0.0
        for k in range(query.shape[0]):
            dists[i] += (trial[0, k] - query[k]) ** 2

    # Each sample lower than its neighbours (the first of a tie; around the
    # seam on a periodic direction), refined by a golden-section search
    # between them
    ratio = 0.6180339887498949
    found = 0
    for i in range(count):
        before = (i - 1) % count if periodic else i - 1
        after  = (i + 1) % count if periodic else i + 1
        if not slopes[i] < 0.0:
            continue
        if before >= 0 and not slopes[i] < slopes[before]:
            continue
        if after < count and slopes[after] < slopes[i]:
            continue

        best_x = i * step
        best   = slopes[i]
        lo     = best_x - step
        hi     = best_x + step
        if not periodic:
            lo = max(lo, 0.0)
            hi = min(hi, end)
        a = hi - ratio * (hi - lo)
        b = lo + ratio * (hi - lo)
        f_a = _pole_slope(query, edge, a % end if periodic else a, on_u, kv_u, kv_v, cv,
                          du, dv, work, basis_u, basis_v, trial)
        f_b = _pole_slope(query, edge, b % end if periodic else b, on_u, kv_u, kv_v, cv,
                          du, dv, work, basis_u, basis_v, trial)
        for _ in range(30):
            if f_a < f_b:
                hi, b, f_b = b, a, f_a
                a = hi - ratio * (hi - lo)
                f_a = _pole_slope(query, edge, a % end if periodic else a, on_u, kv_u,
                                  kv_v, cv, du, dv, work, basis_u, basis_v, trial)
            else:
                lo, a, f_a = a, b, f_b
                b = lo + ratio * (hi - lo)
                f_b = _pole_slope(query, edge, b % end if periodic else b, on_u, kv_u,
                                  kv_v, cv, du, dv, work, basis_u, basis_v, trial)
        if f_a < best:
            best_x = a % end if periodic else a
            best   = f_a
        if f_b < best:
            best_x = b % end if periodic else b
            best   = f_b

        if periodic and best_x >= end:
            best_x -= end
        starts[found] = best_x
        found += 1

    # A nearly collapsed pole is a rim too: each sample closer to the query
    # than its neighbours, unless the rim's spots all lie within a thousandth
    # of one distance from the query (it looks like one point from there)
    near = np.sqrt(dists.min())
    if rim and np.sqrt(dists.max()) - near > 1e-3 * near:
        for i in range(count):
            before = (i - 1) % count if periodic else i - 1
            after  = (i + 1) % count if periodic else i + 1
            if before >= 0 and not dists[i] < dists[before]:
                continue
            if after < count and dists[after] < dists[i]:
                continue
            starts[found] = i * step
            found += 1
    return found


@njit(cache=True)
def _patch_flat_exit(query, u, v, lo, on_u, f, below, kv_u, kv_v, cv, du, dv, end,
                     periodic, work, basis_u, basis_v, trial):
    """
    A way out of a flat span: for a search on a span (lo, lo + 1) of a
    degree-1 direction along which the patch hardly moves (its two rows
    within roundoff of each other, say), a spot strictly closer than f on
    the first span on either side along which the patch moves, when that
    span slopes on down away from it.

    Every spot of a flat span is on the same curve, so its two ends are one
    knot line with a span on each side. On each side (past any more flat
    spans), the closest spot along the direction, the other parameter
    held: the patch is straight along it there, so that is one Newton step,
    kept inside the span. The closer of the two wins; the other, when it
    comes closer too, may hold another basin (``_patch_descend``'s fork).

    Parameters
    ----------
    query : np.ndarray
        The point (dims,), float64.
    u, v : float
        The search's spot, native.
    lo : float
        Start of the flat span along the direction.
    on_u : bool
        Whether the direction is u (else v).
    f : float
        The squared distance to beat.
    below : bool
        The other direction's ``below_u`` / ``below_v`` for
        ``_patch_derivatives``.
    kv_u, kv_v, cv, du, dv
        The patch, as in ``_patch_derivatives``.
    end : float
        End of the direction's domain.
    periodic : bool
        Whether the direction wraps.
    work, basis_u, basis_v, trial : np.ndarray
        Scratch for ``_patch_derivatives``.

    Returns
    -------
    tuple
        (x, lo, other_x, other_lo, moves): the closer side's parameter along
        the direction and the start of its span (lo -1 when neither side
        comes closer), the other side's (other_lo -1 unless it comes closer
        too), and whether the patch moves along the direction on any span
        at all (else it is collapsed there, every span flat).
    """
    spans    = int(end + 0.5)
    best_x   = 0.0
    best_lo  = -1.0
    best_f   = f
    other_x  = 0.0
    other_lo = -1.0
    moves    = False
    for side in range(2):
        # Through the end on this side, span by span, to the first along
        # which the patch moves: the same spot, within the flat spans'
        # width. Its derivatives are its own, one-sided on that end
        g     = 1.0 if side == 0 else -1.0
        x     = lo if side == 0 else lo + 1.0
        at    = lo
        found = False
        for _ in range(spans - 1):
            x, at = _span_exit(x, at, g, end, periodic)
            if at < 0.0:
                break
            if on_u:
                _patch_derivatives(x, v, kv_u, kv_v, cv, du, dv, 2, side == 0, below,
                                   work, basis_u, basis_v, trial)
            else:
                _patch_derivatives(u, x, kv_u, kv_v, cv, du, dv, 2, below, side == 0,
                                   work, basis_u, basis_v, trial)
            slopes = _patch_slopes(query, trial)
            along  = slopes[6] if on_u else slopes[8]
            across = slopes[8] if on_u else slopes[6]
            if not along < 1e-20 * across:
                found = True
                break
            x = at if side == 0 else at + 1.0
        if not found:
            continue
        moves = True

        # Its slope must point on, away from the flat span
        slope = slopes[1] if on_u else slopes[2]
        if not slope * g > 0.0:
            continue
        trial_x = min(max(x - slope / along, at), at + 1.0)
        if on_u:
            _patch_derivatives(trial_x, v, kv_u, kv_v, cv, du, dv, 1, False, below,
                               work, basis_u, basis_v, trial)
        else:
            _patch_derivatives(u, trial_x, kv_u, kv_v, cv, du, dv, 1, below, False,
                               work, basis_u, basis_v, trial)
        f_new = 0.0
        for k in range(query.shape[0]):
            r = trial[0, k] - query[k]
            f_new += r * r
        if not f_new < f:
            continue
        if f_new < best_f:
            other_x  = best_x
            other_lo = best_lo
            best_x   = trial_x
            best_lo  = at
            best_f   = f_new
        else:
            other_x  = trial_x
            other_lo = at
    return best_x, best_lo, other_x, other_lo, moves


@njit(cache=True)
def _patch_descend(query, u, v, kv_u, kv_v, cv, du, dv, max_u, max_v, periodic_u,
                   periodic_v, poles, on_poles, fork, off, below, max_iters, tolerance,
                   work, basis_u, basis_v, here, trial):
    """
    Newton's method on the squared distance, keeping only steps that come closer.

    Parameters
    ----------
    query : np.ndarray
        The point (dims,), float64.
    u, v : float
        Start, native, inside the domain.
    kv_u, kv_v, cv, du, dv
        The patch, as in ``_patch_derivatives``.
    max_u, max_v : float
        End of each direction's domain.
    periodic_u, periodic_v : bool
        Whether each direction wraps; an open one has edges.
    poles, on_poles : np.ndarray
        ``_patch_poles``' (4,) codes; (4,) flags set here for each pole the
        search reaches (``_mark_pole``).
    fork : np.ndarray
        (3,): (u, v, set). Unless already set (1), set here to a start on
        the other side of a flat span the search left, when that side came
        closer too.
    off : bool
        Whether the start is on a pole, to leave it: the first step goes
        straight across it, from the start's span.
    below : bool
        Whether a start on a knot of a degree-1 direction takes the span
        below it (across the seam from 0 on a periodic one) instead of the
        one above.
    max_iters : int
        Most steps.
    tolerance : float
        Done once a step moves (u, v) less than this.
    work, basis_u, basis_v : np.ndarray
        Scratch for ``_patch_derivatives``.
    here, trial : np.ndarray
        Scratch (6, dims); ``here`` ends holding the derivatives at the
        result (right-sided on a knot, as everywhere else).

    Returns
    -------
    tuple
        (u, v, f): the result, periodic parameters in [0, end), and its
        squared distance, never more than the start's.
    """
    # Each step: Newton's (the Hessian when it curves up, else its S_u . S_v
    # part, which always does unless S_u and S_v are parallel), cut to half a
    # span, then halved until it comes strictly closer. When it never does,
    # the slopes scaled per direction (a step that always comes closer once
    # short enough). Open directions are clipped to their domain. A
    # parameter on its edge whose slope points out stays there: the other
    # one solves its own 1-D Newton along the edge. So does a direction
    # along which the patch hardly moves, as on a patch collapsed onto a
    # curve: its slopes are roundoff, and its share of a 2-D step would be
    # so large that the other's, cut with it, never comes closer
    #
    # A degree-1 direction keeps to one span (lo, lo + 1) at a time: its
    # derivatives are that span's (one-sided on its ends), and its ends hold
    # the parameter like edges unless the next span slopes on down. A span
    # along which the patch hardly moves is not held but crossed, as a knot
    # line, into the closer side that slopes on down (``_patch_flat_exit``);
    # the other side, when it comes closer too, is the fork
    span_u = du == 1
    span_v = dv == 1
    lo_u   = min(np.floor(u), max_u - 1.0) if span_u else 0.0
    lo_v   = min(np.floor(v), max_v - 1.0) if span_v else 0.0
    if below and span_u:
        next_u, next_lo = _span_exit(u, lo_u, 1.0, max_u, periodic_u)
        if next_lo >= 0.0:
            u    = next_u
            lo_u = next_lo
    if below and span_v:
        next_v, next_lo = _span_exit(v, lo_v, 1.0, max_v, periodic_v)
        if next_lo >= 0.0:
            v    = next_v
            lo_v = next_lo
    _patch_derivatives(u, v, kv_u, kv_v, cv, du, dv, 2, span_u and u >= lo_u + 1.0,
                       span_v and v >= lo_v + 1.0, work, basis_u, basis_v, here)
    f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv = _patch_slopes(query, here)
    shortest  = 1e-15 * max(max_u, max_v, 1.0)
    collapsed = False

    for _ in range(max_iters):
        # On a span's end with the slope pointing past it: into the next
        # span when the slope there points on too (the same spot, so the
        # same distance); not before the first step off a pole
        if span_u and not off:
            next_u, next_lo = _span_exit(u, lo_u, g_u, max_u, periodic_u)
            if next_lo >= 0.0:
                _patch_derivatives(next_u, v, kv_u, kv_v, cv, du, dv, 2,
                                   next_u >= next_lo + 1.0, span_v and v >= lo_v + 1.0,
                                   work, basis_u, basis_v, trial)
                slopes = _patch_slopes(query, trial)
                if slopes[1] * g_u > 0.0:
                    u       = next_u
                    lo_u    = next_lo
                    here[:] = trial
                    f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv = slopes
        if span_v and not off:
            next_v, next_lo = _span_exit(v, lo_v, g_v, max_v, periodic_v)
            if next_lo >= 0.0:
                _patch_derivatives(u, next_v, kv_u, kv_v, cv, du, dv, 2,
                                   span_u and u >= lo_u + 1.0, next_v >= next_lo + 1.0,
                                   work, basis_u, basis_v, trial)
                slopes = _patch_slopes(query, trial)
                if slopes[2] * g_v > 0.0:
                    v       = next_v
                    lo_v    = next_lo
                    here[:] = trial
                    f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv = slopes

        low_u  = lo_u if span_u else 0.0
        low_v  = lo_v if span_v else 0.0
        high_u = lo_u + 1.0 if span_u else max_u
        high_v = lo_v + 1.0 if span_v else max_v
        ends_u = span_u or not periodic_u
        ends_v = span_v or not periodic_v
        pin_u  = _pinned(u, g_u, low_u, high_u, ends_u)
        pin_v  = _pinned(v, g_v, low_v, high_v, ends_v)

        # A pole a search reaches is left by more searches, from where the
        # patch heads toward the query most steeply
        # (``_newton_closest_point_surface_parallel``); their first step
        # goes straight across it (along it the patch hardly moves, so a
        # 2-D step would mostly turn). On a pole that is one point, so does
        # every step, and a slope pointing out ends the search: every spot
        # along it is the same point. A nearly collapsed one is an edge to
        # slide along
        pole = _mark_pole(u, v, poles, max_u, max_v, on_poles)
        held = pole >= 0 and (off or poles[pole] == 2)
        if held:
            if (pin_u if pole < 2 else pin_v) and not off:
                break
            pin_u = pole >= 2
            pin_v = pole < 2
        off = False

        # A direction along which the patch hardly moves (first and second
        # derivatives under 1e-10 of the other's first) is held like a
        # pinned one. Its second derivative counts: an iso-line where only
        # the first vanishes, as along a doubled row, still bends away. The
        # other's first must not vanish: the step left needs a slope
        bend_u = 0.0
        bend_v = 0.0
        for k in range(query.shape[0]):
            bend_u += here[3, k] * here[3, k]
            bend_v += here[5, k] * here[5, k]
        flat_u = j_uu + bend_u < 1e-20 * j_vv
        flat_v = j_vv + bend_v < 1e-20 * j_uu

        # Such a span of a degree-1 direction (two of its rows within
        # roundoff of each other, say) is one knot line, not a place to
        # stop: first, a step into the span on either side of it when that
        # comes closer (``_patch_flat_exit``). Once no span along it moves
        # (a patch collapsed onto a curve), the search stops looking
        if not held and not collapsed and (span_u and flat_u or span_v and flat_v):
            if flat_u:
                exit_x, exit_lo, other_x, other_lo, moves = _patch_flat_exit(
                    query, u, v, lo_u, True, f, span_v and v >= lo_v + 1.0, kv_u, kv_v,
                    cv, du, dv, max_u, periodic_u, work, basis_u, basis_v, trial)
                if other_lo >= 0.0 and fork[2] == 0.0:
                    fork[0] = other_x
                    fork[1] = v
                    fork[2] = 1.0
                if exit_lo >= 0.0:
                    u    = exit_x
                    lo_u = exit_lo
            else:
                exit_x, exit_lo, other_x, other_lo, moves = _patch_flat_exit(
                    query, u, v, lo_v, False, f, span_u and u >= lo_u + 1.0, kv_u, kv_v,
                    cv, du, dv, max_v, periodic_v, work, basis_u, basis_v, trial)
                if other_lo >= 0.0 and fork[2] == 0.0:
                    fork[0] = u
                    fork[1] = other_x
                    fork[2] = 1.0
                if exit_lo >= 0.0:
                    v    = exit_x
                    lo_v = exit_lo
            collapsed = not moves
            if exit_lo >= 0.0:
                _patch_derivatives(u, v, kv_u, kv_v, cv, du, dv, 2,
                                   span_u and u >= lo_u + 1.0,
                                   span_v and v >= lo_v + 1.0, work, basis_u, basis_v,
                                   here)
                slopes = _patch_slopes(query, here)
                f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv = slopes
                continue
        if flat_u:
            pin_u = True
        if flat_v:
            pin_v = True
        if pin_u and pin_v:
            break

        moved = False
        new_u = u
        new_v = v
        for kind in range(2):
            step_u = 0.0
            step_v = 0.0
            if kind == 1:
                if not pin_u and j_uu > 0.0:
                    step_u = -g_u / j_uu
                if not pin_v and j_vv > 0.0:
                    step_v = -g_v / j_vv
            elif pin_u:
                curve  = h_vv if h_vv > 0.0 else j_vv
                step_v = -g_v / curve if curve > 0.0 else 0.0
            elif pin_v:
                curve  = h_uu if h_uu > 0.0 else j_uu
                step_u = -g_u / curve if curve > 0.0 else 0.0
            else:
                det = h_uu * h_vv - h_uv * h_uv
                if h_uu > 0.0 and det > 0.0:
                    step_u = (h_uv * g_v - h_vv * g_u) / det
                    step_v = (h_uv * g_u - h_uu * g_v) / det
                else:
                    det = j_uu * j_vv - j_uv * j_uv
                    if det > 1e-12 * j_uu * j_vv:
                        step_u = (j_uv * g_v - j_vv * g_u) / det
                        step_v = (j_uv * g_u - j_uu * g_v) / det

            size = max(abs(step_u), abs(step_v))
            if not size > 0.0:
                continue
            if size > 0.5:
                step_u *= 0.5 / size
                step_v *= 0.5 / size
                size = 0.5

            t = 1.0
            while t * size > shortest:
                new_u = u + t * step_u
                new_v = v + t * step_v
                if ends_u:
                    new_u = min(max(new_u, low_u), high_u)
                else:
                    new_u = new_u % max_u
                    if new_u >= max_u:
                        new_u -= max_u
                if ends_v:
                    new_v = min(max(new_v, low_v), high_v)
                else:
                    new_v = new_v % max_v
                    if new_v >= max_v:
                        new_v -= max_v
                _patch_derivatives(new_u, new_v, kv_u, kv_v, cv, du, dv, 2,
                                   span_u and new_u >= high_u,
                                   span_v and new_v >= high_v, work, basis_u, basis_v,
                                   trial)
                f_new = 0.0
                for k in range(query.shape[0]):
                    r = trial[0, k] - query[k]
                    f_new += r * r
                if f_new < f:
                    moved = True
                    break
                t *= 0.5
            if moved:
                break
        if not moved:
            break

        # How far it moved, the short way round a periodic direction
        gap_u = abs(new_u - u)
        gap_v = abs(new_v - v)
        if periodic_u:
            gap_u = min(gap_u, max_u - gap_u)
        if periodic_v:
            gap_v = min(gap_v, max_v - gap_v)

        u       = new_u
        v       = new_v
        here[:] = trial
        f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv = _patch_slopes(query, here)
        if max(gap_u, gap_v) < tolerance:
            break
    _mark_pole(u, v, poles, max_u, max_v, on_poles)

    # A degree-1 direction: back to the right-sided derivatives compute()
    # gives on a knot, and from the seam's end to 0
    if span_u or span_v:
        if periodic_u and u >= max_u:
            u = 0.0
        if periodic_v and v >= max_v:
            v = 0.0
        _patch_derivatives(u, v, kv_u, kv_v, cv, du, dv, 2, False, False, work, basis_u,
                           basis_v, here)

    return u, v, f


@njit(parallel=True, cache=True)
def _newton_closest_point_surface_parallel(
    queries,
    u_init,
    v_init,
    kv_u,
    kv_v,
    cv,
    du,
    dv,
    max_param_u,
    max_param_v,
    periodic_u,
    periodic_v,
    max_iters,
    tolerance,
    reach_u,
    reach_v,
):
    """
    Parallel closest point on a B-spline surface.

    Finds (u, v) such that S(u, v) is closest to each query point: from
    each seed, ``_patch_descend``'s Newton search, which only takes steps
    that come closer and slides along an open edge or a degree-1
    direction's fold when the closest spot is on it. On a patch open both
    ways, a corner closer than the best result starts one more search, and
    so does the first fork of a query's searches (the other side of a flat
    span, ``_patch_descend``). A pole any of them reached is left by more
    searches, from each spot ``_patch_pole_turns`` finds. The closest result
    wins.

    Parameters
    ----------
    queries : np.ndarray
        Query points (n_queries, dims), float64.
    u_init, v_init : np.ndarray
        Seeds (n_queries, n_seeds), float64, native and inside the domain,
        best first. A seed within reach of where an earlier search of its
        query ended, in both directions, is skipped: it would end there too.
    kv_u, kv_v : np.ndarray
        Knot vectors, float64.
    cv : np.ndarray
        Control point grid (cu, cv_n, dims), float64.
    du, dv : int
        Surface degrees.
    max_param_u, max_param_v : float
        Maximum parameter values per direction.
    periodic_u, periodic_v : bool
        Whether the surface is periodic per direction.
    max_iters : int
        Most Newton steps per search.
    tolerance : float
        Convergence tolerance for parameter change.
    reach_u, reach_v : float
        How close, in native parameters, a seed must be to an earlier
        search's end to be skipped; 0 skips only a seed right on one.

    Returns
    -------
    u_out, v_out : np.ndarray
        Refined parameter values (n_queries,), float64.
    points_out : np.ndarray
        Closest points on surface (n_queries, dims), float64.
    tangents_u_out, tangents_v_out : np.ndarray
        Tangent vectors at closest points (n_queries, dims), float64.
    """
    n_queries = queries.shape[0]
    n_seeds   = u_init.shape[1]
    dims      = cv.shape[2]

    u_out          = np.empty(n_queries,         dtype=np.float64)
    v_out          = np.empty(n_queries,         dtype=np.float64)
    points_out     = np.empty((n_queries, dims), dtype=np.float64)
    tangents_u_out = np.empty((n_queries, dims), dtype=np.float64)
    tangents_v_out = np.empty((n_queries, dims), dtype=np.float64)

    # Which open edges are poles, and the corners of a patch open both ways:
    # (u, v) and position
    poles   = _patch_poles(cv, periodic_u, periodic_v)
    corners = 0 if periodic_u or periodic_v else 4
    corner  = np.zeros((4, 2 + dims))
    if corners:
        work    = np.empty(max(du, dv) + 1)
        basis_u = np.empty((3, du + 1))
        basis_v = np.empty((3, dv + 1))
        derivs  = np.empty((6, dims))
        for c in range(4):
            corner[c, 0] = max_param_u if c % 2 else 0.0
            corner[c, 1] = max_param_v if c // 2 else 0.0
            _patch_derivatives(corner[c, 0], corner[c, 1], kv_u, kv_v, cv, du, dv, 1,
                               False, False, work, basis_u, basis_v, derivs)
            corner[c, 2:] = derivs[0]

    for qi in prange(n_queries):
        query = queries[qi]

        # Thread-local scratch; best holds the derivatives at the result,
        # on_poles the poles this query's searches reached, fork the other
        # side of the first flat span one left (``_patch_descend``)
        work     = np.empty(max(du, dv) + 1)
        basis_u  = np.empty((3, du + 1))
        basis_v  = np.empty((3, dv + 1))
        here     = np.empty((6, dims))
        trial    = np.empty((6, dims))
        best     = np.empty((6, dims))
        ends     = np.empty((n_seeds, 2))
        on_poles = np.zeros(4, dtype=np.bool_)
        fork     = np.zeros(3)
        spans    = max(int(max_param_u + 0.5), int(max_param_v + 0.5), 1)
        starts   = np.empty(8 * spans + 2)

        ui = u_init[qi, 0]
        vi = v_init[qi, 0]
        fi = np.inf
        for s in range(n_seeds):
            su = u_init[qi, s]
            sv = v_init[qi, s]

            # Skip a seed within reach of where an earlier search ended: it
            # would end there too. Along a degree-1 direction, only one on
            # the span that search ended on: a fold can part two basins
            near = False
            for e in range(s):
                gap_u = abs(su - ends[e, 0])
                gap_v = abs(sv - ends[e, 1])
                if periodic_u:
                    gap_u = min(gap_u, max_param_u - gap_u)
                if periodic_v:
                    gap_v = min(gap_v, max_param_v - gap_v)
                if du == 1 and not np.floor(su) <= ends[e, 0] <= np.floor(su) + 1.0:
                    gap_u = np.inf
                if dv == 1 and not np.floor(sv) <= ends[e, 1] <= np.floor(sv) + 1.0:
                    gap_v = np.inf
                if gap_u <= reach_u and gap_v <= reach_v:
                    near    = True
                    ends[s] = ends[e]
                    break
            if near:
                continue

            u, v, f = _patch_descend(query, su, sv, kv_u, kv_v, cv, du, dv, max_param_u,
                                     max_param_v, periodic_u, periodic_v, poles,
                                     on_poles, fork, False, False, max_iters, tolerance,
                                     work, basis_u, basis_v, here, trial)
            ends[s, 0] = u
            ends[s, 1] = v
            if s == 0 or f < fi:
                ui, vi, fi = u, v, f
                best[:] = here

        # A corner closer than the result: search from there too
        for c in range(corners):
            f = 0.0
            for k in range(dims):
                r = corner[c, 2 + k] - query[k]
                f += r * r
            if not f < fi:
                continue
            u, v, f = _patch_descend(query, corner[c, 0], corner[c, 1], kv_u, kv_v, cv,
                                     du, dv, max_param_u, max_param_v, periodic_u,
                                     periodic_v, poles, on_poles, fork, False, False,
                                     max_iters, tolerance, work, basis_u, basis_v, here,
                                     trial)
            if f < fi:
                ui, vi, fi = u, v, f
                best[:] = here

        # A flat span's two sides can hold two basins: when a search left
        # one with both sides closer, one more search from the other
        if fork[2] > 0.0:
            u, v, f = _patch_descend(query, fork[0], fork[1], kv_u, kv_v, cv, du, dv,
                                     max_param_u, max_param_v, periodic_u, periodic_v,
                                     poles, on_poles, fork, False, False, max_iters,
                                     tolerance, work, basis_u, basis_v, here, trial)
            if f < fi:
                ui, vi, fi = u, v, f
                best[:] = here

        # A search that reached a pole stopped there, slid along it (a
        # nearly collapsed one) or left it where it reached it, so others
        # leave it: one per way off it that comes closer, from where the
        # patch heads toward the query most steeply, and on a nearly
        # collapsed pole one from each spot of its rim closest to the query
        for pole in range(4):
            if not on_poles[pole]:
                continue
            on_u = pole < 2
            if on_u:
                edge = 0.0 if pole == 0 else max_param_u
                found = _patch_pole_turns(query, edge, True, max_param_v, periodic_v,
                                          kv_u, kv_v, cv, du, dv, work, basis_u,
                                          basis_v, trial, poles[pole] == 1, starts)
            else:
                edge = 0.0 if pole == 2 else max_param_v
                found = _patch_pole_turns(query, edge, False, max_param_u, periodic_u,
                                          kv_u, kv_v, cv, du, dv, work, basis_u,
                                          basis_v, trial, poles[pole] == 1, starts)
            # Along a degree-1 direction, a start on a knot (two facets meet
            # there) is tried on the span on each side of it
            facets = (dv if on_u else du) == 1
            for t in range(2 * found):
                x     = starts[t // 2]
                below = t % 2 == 1
                if below and not (facets and x == np.floor(x)):
                    continue
                su = edge if on_u else x
                sv = x if on_u else edge
                u, v, f = _patch_descend(query, su, sv, kv_u, kv_v, cv, du, dv,
                                         max_param_u, max_param_v, periodic_u,
                                         periodic_v, poles, on_poles, fork, True,
                                         below, max_iters, tolerance, work, basis_u,
                                         basis_v, here, trial)
                if f < fi:
                    ui, vi, fi = u, v, f
                    best[:] = here

        u_out[qi] = ui
        v_out[qi] = vi
        for k in range(dims):
            points_out[qi, k]     = best[0, k]
            tangents_u_out[qi, k] = best[1, k]
            tangents_v_out[qi, k] = best[2, k]

    return u_out, v_out, points_out, tangents_u_out, tangents_v_out


# --------------------------------------------------------------------------- #
#                Ray-surface intersection (raycast)                            #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _raycast_bspline_surface_parallel(
    origins,
    directions,
    grid_u,
    grid_v,
    grid_pts,
    kv_u,
    kv_v,
    cv,
    du,
    dv,
    max_param_u,
    max_param_v,
    periodic_u,
    periodic_v,
    forward_only,
    twosided,
    max_iters,
    tolerance,
    hit_eps,
):
    """
    Parallel ray-B-spline surface intersection.

    Per-ray algorithm (parallel across rays):

    1. Coarse search: scan the precomputed grid samples and pick the
       sample with the smallest perpendicular distance to the ray
       (with t along ray >= 0 when ``forward_only=True``) as the
       initial (u, v) guess.
    2. Gauss-Newton refinement on the 3-component residual
       ``F(u, v) = (S(u, v) - origin) x direction``.  The Jacobian is
       ``J = [S_u x d, S_v x d]`` and the step is solved via
       Cramer's rule on the 2x2 normal equations.
    3. Compute ``t = (S(u, v) - origin) . direction`` (directions are
       assumed unit length by the caller).
    4. Validate as a hit when the residual is below ``hit_eps``,
       ``t >= 0`` if ``forward_only``, and (for ``twosided=False``)
       the surface normal does not face the same way as the ray.

    Parameters
    ----------
    origins : np.ndarray
        Ray origins (n, 3), float64.
    directions : np.ndarray
        Ray directions (n, 3), float64. Should be unit length.
    grid_u, grid_v : np.ndarray
        Precomputed coarse sample parameters (n_grid,), float64.
    grid_pts : np.ndarray
        Precomputed surface positions at grid samples (n_grid, 3), float64.
    kv_u, kv_v : np.ndarray
        Knot vectors for U and V, float64.
    cv : np.ndarray
        Control point grid (cu, cv_n, dims), float64. Requires dims == 3.
    du, dv : int
        Surface degrees in U and V.
    max_param_u, max_param_v : float
        Maximum native parameter values per direction.
    periodic_u, periodic_v : bool
        Whether the surface is periodic per direction.
    forward_only : bool
        If True, only consider candidates with t >= 0 along the ray.
    twosided : bool
        If False, reject hits where the surface normal faces the same
        direction as the ray (back-face hits).
    max_iters : int
        Maximum Newton iterations per ray.
    tolerance : float
        Convergence tolerance: stop when ||F||^2 < tolerance^2.
    hit_eps : float
        Maximum allowed residual ||F|| at convergence to count as a hit.

    Returns
    -------
    u_out, v_out : np.ndarray
        Refined parameter values (n,), float64. 0.0 for misses.
    t_out : np.ndarray
        Distance along ray at hit (n,), float64. NaN for misses.
    hit_out : np.ndarray
        Per-ray hit mask (n,), bool.
    """
    n_rays  = origins.shape[0]
    n_grid  = grid_pts.shape[0]
    n_cv_u  = cv.shape[0]
    n_cv_v  = cv.shape[1]

    order_u = du + 1
    order_v = dv + 1

    if periodic_u:
        limit_u = int(max_param_u) - 1
    else:
        limit_u = n_cv_u - du - 1

    if periodic_v:
        limit_v = int(max_param_v) - 1
    else:
        limit_v = n_cv_v - dv - 1

    u_out   = np.zeros(n_rays, dtype=np.float64)
    v_out   = np.zeros(n_rays, dtype=np.float64)
    t_out   = np.empty(n_rays, dtype=np.float64)
    hit_out = np.zeros(n_rays, dtype=np.bool_)

    for k in prange(n_rays):
        ox = origins[k, 0]
        oy = origins[k, 1]
        oz = origins[k, 2]
        dx = directions[k, 0]
        dy = directions[k, 1]
        dz = directions[k, 2]

        # ===== Phase 1: Coarse sample search for initial (u, v) =====
        best_perp_sq = np.inf
        best_u_init  = 0.0
        best_v_init  = 0.0

        for gi in range(n_grid):
            ex = grid_pts[gi, 0] - ox
            ey = grid_pts[gi, 1] - oy
            ez = grid_pts[gi, 2] - oz

            t_along = ex * dx + ey * dy + ez * dz
            if forward_only and t_along < 0.0:
                continue

            diff_sq = ex * ex + ey * ey + ez * ez
            perp_sq = diff_sq - t_along * t_along
            if perp_sq < 0.0:  # numerical safety
                perp_sq = 0.0

            if perp_sq < best_perp_sq:
                best_perp_sq = perp_sq
                best_u_init  = grid_u[gi]
                best_v_init  = grid_v[gi]

        if best_perp_sq == np.inf:
            # No grid sample with t >= 0 found - definite miss
            t_out[k] = np.nan
            continue

        # ===== Phase 2: Gauss-Newton refinement =====
        ui = best_u_init
        vi = best_v_init

        # Thread-local basis buffers
        b_u     = np.zeros(order_u + 1, dtype=np.float64)
        bb_u    = np.zeros(order_u + 1, dtype=np.float64)
        b_u_d1  = np.zeros(order_u,     dtype=np.float64)
        bb_u_d1 = np.zeros(order_u,     dtype=np.float64)

        b_v     = np.zeros(order_v + 1, dtype=np.float64)
        bb_v    = np.zeros(order_v + 1, dtype=np.float64)
        b_v_d1  = np.zeros(order_v,     dtype=np.float64)
        bb_v_d1 = np.zeros(order_v,     dtype=np.float64)

        Sx  = 0.0
        Sy  = 0.0
        Sz  = 0.0
        Sux = 0.0
        Suy = 0.0
        Suz = 0.0
        Svx = 0.0
        Svy = 0.0
        Svz = 0.0

        for _ in range(max_iters):
            # --- Find spans ---
            left_u = int(np.floor(ui))
            if left_u < 0:
                left_u = 0
            elif left_u > limit_u:
                left_u = limit_u
            right_u = left_u + order_u

            left_v  = int(np.floor(vi))
            if left_v < 0:
                left_v = 0
            elif left_v > limit_v:
                left_v = limit_v
            right_v = left_v + order_v

            # --- U basis at degree du ---
            for i in range(order_u + 1):
                b_u[i] = 0.0
            b_u[0] = 1.0
            for j in range(1, order_u):
                for i in range(j):
                    bb_u[i] = b_u[i]
                    b_u[i]  = 0.0
                for i in range(j):
                    ri    = right_u + i
                    denom = kv_u[ri] - kv_u[ri - j]
                    if denom != 0.0:
                        f = bb_u[i] / denom
                    else:
                        f = 0.0
                    b_u[i] += f * (kv_u[ri] - ui)
                    b_u[i + 1] = f * (ui - kv_u[ri - j])

            # --- U basis at degree du-1 (for dS/du) ---
            for i in range(order_u):
                b_u_d1[i] = 0.0
            b_u_d1[0] = 1.0
            for j in range(1, du):
                for i in range(j):
                    bb_u_d1[i] = b_u_d1[i]
                    b_u_d1[i]  = 0.0
                for i in range(j):
                    ri    = right_u + i
                    denom = kv_u[ri] - kv_u[ri - j]
                    if denom != 0.0:
                        f = bb_u_d1[i] / denom
                    else:
                        f = 0.0
                    b_u_d1[i] += f * (kv_u[ri] - ui)
                    b_u_d1[i + 1] = f * (ui - kv_u[ri - j])

            # --- V basis at degree dv ---
            for i in range(order_v + 1):
                b_v[i] = 0.0
            b_v[0] = 1.0
            for j in range(1, order_v):
                for i in range(j):
                    bb_v[i] = b_v[i]
                    b_v[i]  = 0.0
                for i in range(j):
                    ri    = right_v + i
                    denom = kv_v[ri] - kv_v[ri - j]
                    if denom != 0.0:
                        f = bb_v[i] / denom
                    else:
                        f = 0.0
                    b_v[i] += f * (kv_v[ri] - vi)
                    b_v[i + 1] = f * (vi - kv_v[ri - j])

            # --- V basis at degree dv-1 (for dS/dv) ---
            for i in range(order_v):
                b_v_d1[i] = 0.0
            b_v_d1[0] = 1.0
            for j in range(1, dv):
                for i in range(j):
                    bb_v_d1[i] = b_v_d1[i]
                    b_v_d1[i]  = 0.0
                for i in range(j):
                    ri    = right_v + i
                    denom = kv_v[ri] - kv_v[ri - j]
                    if denom != 0.0:
                        f = bb_v_d1[i] / denom
                    else:
                        f = 0.0
                    b_v_d1[i] += f * (kv_v[ri] - vi)
                    b_v_d1[i + 1] = f * (vi - kv_v[ri - j])

            # --- Evaluate S ---
            Sx = 0.0
            Sy = 0.0
            Sz = 0.0
            for iu in range(order_u):
                cp_u = left_u + iu
                if cp_u < n_cv_u:
                    bu_val = b_u[iu]
                    for iv in range(order_v):
                        cp_v = left_v + iv
                        if cp_v < n_cv_v:
                            w = bu_val * b_v[iv]
                            Sx += w * cv[cp_u, cp_v, 0]
                            Sy += w * cv[cp_u, cp_v, 1]
                            Sz += w * cv[cp_u, cp_v, 2]

            # --- Evaluate dS/du ---
            Sux = 0.0
            Suy = 0.0
            Suz = 0.0
            for iu in range(du):
                cp_u = left_u + iu
                if cp_u + 1 < n_cv_u:
                    kv_idx_u = cp_u + 1
                    denom_u  = kv_u[kv_idx_u + du] - kv_u[kv_idx_u]
                    if denom_u != 0.0:
                        for iv in range(order_v):
                            cp_v = left_v + iv
                            if cp_v < n_cv_v:
                                sc = du * b_u_d1[iu] * b_v[iv] / denom_u
                                Sux += sc * (cv[cp_u + 1, cp_v, 0] - cv[cp_u, cp_v, 0])
                                Suy += sc * (cv[cp_u + 1, cp_v, 1] - cv[cp_u, cp_v, 1])
                                Suz += sc * (cv[cp_u + 1, cp_v, 2] - cv[cp_u, cp_v, 2])

            # --- Evaluate dS/dv ---
            Svx = 0.0
            Svy = 0.0
            Svz = 0.0
            for iu in range(order_u):
                cp_u = left_u + iu
                if cp_u < n_cv_u:
                    for iv in range(dv):
                        cp_v = left_v + iv
                        if cp_v + 1 < n_cv_v:
                            kv_idx_v = cp_v + 1
                            denom_v  = kv_v[kv_idx_v + dv] - kv_v[kv_idx_v]
                            if denom_v != 0.0:
                                sc = dv * b_u[iu] * b_v_d1[iv] / denom_v
                                Svx += sc * (cv[cp_u, cp_v + 1, 0] - cv[cp_u, cp_v, 0])
                                Svy += sc * (cv[cp_u, cp_v + 1, 1] - cv[cp_u, cp_v, 1])
                                Svz += sc * (cv[cp_u, cp_v + 1, 2] - cv[cp_u, cp_v, 2])

            # --- Residual F = (S - O) x d ---
            ex = Sx - ox
            ey = Sy - oy
            ez = Sz - oz

            fx = ey * dz - ez * dy
            fy = ez * dx - ex * dz
            fz = ex * dy - ey * dx

            f_norm_sq = fx * fx + fy * fy + fz * fz
            if f_norm_sq < tolerance * tolerance:
                break

            # --- Gauss-Newton step: J = [S_u x d, S_v x d] ---
            J00   = Suy * dz - Suz * dy
            J10   = Suz * dx - Sux * dz
            J20   = Sux * dy - Suy * dx
            J01   = Svy * dz - Svz * dy
            J11   = Svz * dx - Svx * dz
            J21   = Svx * dy - Svy * dx

            JTJ00 = J00 * J00 + J10 * J10 + J20 * J20
            JTJ01 = J00 * J01 + J10 * J11 + J20 * J21
            JTJ11 = J01 * J01 + J11 * J11 + J21 * J21

            JTf0  = J00 * fx + J10 * fy + J20 * fz
            JTf1  = J01 * fx + J11 * fy + J21 * fz

            det   = JTJ00 * JTJ11 - JTJ01 * JTJ01
            if abs(det) < 1e-14:
                break  # singular - bail out

            inv_det = 1.0 / det
            delta_u = -(JTJ11 * JTf0 - JTJ01 * JTf1) * inv_det
            delta_v = -(JTJ00 * JTf1 - JTJ01 * JTf0) * inv_det

            # Trust region
            if delta_u > 0.5:
                delta_u = 0.5
            elif delta_u < -0.5:
                delta_u = -0.5
            if delta_v > 0.5:
                delta_v = 0.5
            elif delta_v < -0.5:
                delta_v = -0.5

            ui_new = ui + delta_u
            vi_new = vi + delta_v

            # Handle bounds
            if periodic_u:
                while ui_new < 0:
                    ui_new += max_param_u
                while ui_new >= max_param_u:
                    ui_new -= max_param_u
            else:
                if ui_new < 0:
                    ui_new = 0.0
                elif ui_new > max_param_u:
                    ui_new = max_param_u

            if periodic_v:
                while vi_new < 0:
                    vi_new += max_param_v
                while vi_new >= max_param_v:
                    vi_new -= max_param_v
            else:
                if vi_new < 0:
                    vi_new = 0.0
                elif vi_new > max_param_v:
                    vi_new = max_param_v

            ui = ui_new
            vi = vi_new

        # ===== Phase 3: Validate hit at final (ui, vi) =====
        # S/S_u/S_v above are at the (ui, vi) BEFORE the last update
        # if Newton converged, or at the previous iteration if iter limit
        # was hit. For correctness, do one final evaluation pass.
        left_u = int(np.floor(ui))
        if left_u < 0:
            left_u = 0
        elif left_u > limit_u:
            left_u = limit_u
        right_u = left_u + order_u

        left_v  = int(np.floor(vi))
        if left_v < 0:
            left_v = 0
        elif left_v > limit_v:
            left_v = limit_v
        right_v = left_v + order_v

        # U basis at degree du
        for i in range(order_u + 1):
            b_u[i] = 0.0
        b_u[0] = 1.0
        for j in range(1, order_u):
            for i in range(j):
                bb_u[i] = b_u[i]
                b_u[i]  = 0.0
            for i in range(j):
                ri    = right_u + i
                denom = kv_u[ri] - kv_u[ri - j]
                if denom != 0.0:
                    f = bb_u[i] / denom
                else:
                    f = 0.0
                b_u[i] += f * (kv_u[ri] - ui)
                b_u[i + 1] = f * (ui - kv_u[ri - j])

        # U basis at degree du-1
        for i in range(order_u):
            b_u_d1[i] = 0.0
        b_u_d1[0] = 1.0
        for j in range(1, du):
            for i in range(j):
                bb_u_d1[i] = b_u_d1[i]
                b_u_d1[i]  = 0.0
            for i in range(j):
                ri    = right_u + i
                denom = kv_u[ri] - kv_u[ri - j]
                if denom != 0.0:
                    f = bb_u_d1[i] / denom
                else:
                    f = 0.0
                b_u_d1[i] += f * (kv_u[ri] - ui)
                b_u_d1[i + 1] = f * (ui - kv_u[ri - j])

        # V basis at degree dv
        for i in range(order_v + 1):
            b_v[i] = 0.0
        b_v[0] = 1.0
        for j in range(1, order_v):
            for i in range(j):
                bb_v[i] = b_v[i]
                b_v[i]  = 0.0
            for i in range(j):
                ri    = right_v + i
                denom = kv_v[ri] - kv_v[ri - j]
                if denom != 0.0:
                    f = bb_v[i] / denom
                else:
                    f = 0.0
                b_v[i] += f * (kv_v[ri] - vi)
                b_v[i + 1] = f * (vi - kv_v[ri - j])

        # V basis at degree dv-1
        for i in range(order_v):
            b_v_d1[i] = 0.0
        b_v_d1[0] = 1.0
        for j in range(1, dv):
            for i in range(j):
                bb_v_d1[i] = b_v_d1[i]
                b_v_d1[i]  = 0.0
            for i in range(j):
                ri    = right_v + i
                denom = kv_v[ri] - kv_v[ri - j]
                if denom != 0.0:
                    f = bb_v_d1[i] / denom
                else:
                    f = 0.0
                b_v_d1[i] += f * (kv_v[ri] - vi)
                b_v_d1[i + 1] = f * (vi - kv_v[ri - j])

        # Final S
        Sx = 0.0
        Sy = 0.0
        Sz = 0.0
        for iu in range(order_u):
            cp_u = left_u + iu
            if cp_u < n_cv_u:
                bu_val = b_u[iu]
                for iv in range(order_v):
                    cp_v = left_v + iv
                    if cp_v < n_cv_v:
                        w = bu_val * b_v[iv]
                        Sx += w * cv[cp_u, cp_v, 0]
                        Sy += w * cv[cp_u, cp_v, 1]
                        Sz += w * cv[cp_u, cp_v, 2]

        # Final dS/du, dS/dv (for back-face check)
        Sux = 0.0
        Suy = 0.0
        Suz = 0.0
        for iu in range(du):
            cp_u = left_u + iu
            if cp_u + 1 < n_cv_u:
                kv_idx_u = cp_u + 1
                denom_u  = kv_u[kv_idx_u + du] - kv_u[kv_idx_u]
                if denom_u != 0.0:
                    for iv in range(order_v):
                        cp_v = left_v + iv
                        if cp_v < n_cv_v:
                            sc = du * b_u_d1[iu] * b_v[iv] / denom_u
                            Sux += sc * (cv[cp_u + 1, cp_v, 0] - cv[cp_u, cp_v, 0])
                            Suy += sc * (cv[cp_u + 1, cp_v, 1] - cv[cp_u, cp_v, 1])
                            Suz += sc * (cv[cp_u + 1, cp_v, 2] - cv[cp_u, cp_v, 2])

        Svx = 0.0
        Svy = 0.0
        Svz = 0.0
        for iu in range(order_u):
            cp_u = left_u + iu
            if cp_u < n_cv_u:
                for iv in range(dv):
                    cp_v = left_v + iv
                    if cp_v + 1 < n_cv_v:
                        kv_idx_v = cp_v + 1
                        denom_v  = kv_v[kv_idx_v + dv] - kv_v[kv_idx_v]
                        if denom_v != 0.0:
                            sc = dv * b_u[iu] * b_v_d1[iv] / denom_v
                            Svx += sc * (cv[cp_u, cp_v + 1, 0] - cv[cp_u, cp_v, 0])
                            Svy += sc * (cv[cp_u, cp_v + 1, 1] - cv[cp_u, cp_v, 1])
                            Svz += sc * (cv[cp_u, cp_v + 1, 2] - cv[cp_u, cp_v, 2])

        # Compute residual and t at final (u, v)
        ex = Sx - ox
        ey = Sy - oy
        ez = Sz - oz

        fx          = ey * dz - ez * dy
        fy          = ez * dx - ex * dz
        fz          = ex * dy - ey * dx
        residual_sq = fx * fx + fy * fy + fz * fz

        # t along ray (assumes direction is unit length)
        t_along = ex * dx + ey * dy + ez * dz

        is_hit  = residual_sq < hit_eps * hit_eps
        if forward_only and t_along < 0.0:
            is_hit = False

        if is_hit and not twosided:
            # Surface normal n = S_u x S_v
            nx = Suy * Svz - Suz * Svy
            ny = Suz * Svx - Sux * Svz
            nz = Sux * Svy - Suy * Svx
            if nx * dx + ny * dy + nz * dz > 0.0:
                is_hit = False

        if is_hit:
            u_out[k]   = ui
            v_out[k]   = vi
            t_out[k]   = t_along
            hit_out[k] = True
        else:
            t_out[k] = np.nan

    return u_out, v_out, t_out, hit_out