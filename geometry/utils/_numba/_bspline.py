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
#                     Newton-Raphson closest point                            #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _newton_closest_point_parallel(
    queries,
    u_init,
    kv,
    cv,
    degree,
    max_param,
    periodic,
    max_iters,
    tolerance,
):
    """
    Parallel Newton-Raphson closest point computation on B-spline curve.

    Finds the parameter u such that the curve point C(u) is closest to
    each query point. Uses the perpendicularity condition:
        f(u) = (C(u) - Q) * C'(u) = 0

    Each query point is processed independently in parallel.

    Parameters
    ----------
    queries : np.ndarray
        Query points (n_queries, dims), float64.
    u_init : np.ndarray
        Initial parameter guesses (n_queries,), float64.
    kv : np.ndarray
        Knot vector, float64.
    cv : np.ndarray
        Control vertices (n_cv, dims), float64.
    degree : int
        Curve degree.
    max_param : float
        Maximum parameter value.
    periodic : bool
        Whether the curve is periodic.
    max_iters : int
        Maximum Newton-Raphson iterations.
    tolerance : float
        Convergence tolerance for parameter change.

    Returns
    -------
    u_out : np.ndarray
        Refined parameter values (n_queries,), float64.
    points_out : np.ndarray
        Closest points on curve (n_queries, dims), float64.
    tangents_out : np.ndarray
        Tangent vectors at closest points (n_queries, dims), float64.
    """
    n_queries = queries.shape[0]
    n_cv      = cv.shape[0]
    dims      = cv.shape[1]
    order     = degree + 1

    # For periodic curves, we need the "base" control point count
    # The cv array may have wrapped points appended
    if periodic:
        # limit for span finding in periodic case
        limit = int(max_param) - 1
    else:
        limit = n_cv - degree - 1

    # Output arrays
    u_out        = np.empty(n_queries,         dtype=np.float64)
    points_out   = np.empty((n_queries, dims), dtype=np.float64)
    tangents_out = np.empty((n_queries, dims), dtype=np.float64)

    # Process each query point in parallel
    for qi in prange(n_queries):
        query = queries[qi]
        ui    = u_init[qi]

        # Local arrays for basis computation
        local_b  = np.zeros(order + 1, dtype=np.float64)
        local_bb = np.zeros(order + 1, dtype=np.float64)

        # Basis at degree-1 for first derivative
        local_b_d1  = np.zeros(order, dtype=np.float64)
        local_bb_d1 = np.zeros(order, dtype=np.float64)

        # Basis at degree-2 for second derivative
        local_b_d2  = np.zeros(order, dtype=np.float64)
        local_bb_d2 = np.zeros(order, dtype=np.float64)

        # Newton-Raphson iterations
        for _ in range(max_iters):
            # Find knot span for current u
            left_k  = _find_span(ui, kv, degree, limit)
            right_k = left_k + order

            # === Compute basis functions at degree d ===
            for i in range(order + 1):
                local_b[i] = 0.0
            local_b[0] = 1.0

            for j in range(1, order):
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
                    local_b[i] += f * (kv[ri] - ui)
                    local_b[i + 1] = f * (ui - kv[ri - j])

            # === Compute basis functions at degree d-1 (for 1st derivative) ===
            for i in range(order):
                local_b_d1[i] = 0.0
            local_b_d1[0] = 1.0

            for j in range(1, degree):
                for i in range(j):
                    local_bb_d1[i] = local_b_d1[i]
                    local_b_d1[i]  = 0.0

                for i in range(j):
                    ri    = right_k + i
                    denom = kv[ri] - kv[ri - j]
                    if denom != 0.0:
                        f = local_bb_d1[i] / denom
                    else:
                        f = 0.0
                    local_b_d1[i] += f * (kv[ri] - ui)
                    local_b_d1[i + 1] = f * (ui - kv[ri - j])

            # === Compute basis functions at degree d-2 (for 2nd derivative) ===
            if degree >= 2:
                for i in range(order):
                    local_b_d2[i] = 0.0
                local_b_d2[0] = 1.0

                for j in range(1, degree - 1):
                    for i in range(j):
                        local_bb_d2[i] = local_b_d2[i]
                        local_b_d2[i]  = 0.0

                    for i in range(j):
                        ri    = right_k + i
                        denom = kv[ri] - kv[ri - j]
                        if denom != 0.0:
                            f = local_bb_d2[i] / denom
                        else:
                            f = 0.0
                        local_b_d2[i] += f * (kv[ri] - ui)
                        local_b_d2[i + 1] = f * (ui - kv[ri - j])

            # === Evaluate C(u) ===
            C = np.zeros(dims, dtype=np.float64)
            for i in range(order):
                cp_idx = left_k + i
                if cp_idx < n_cv:
                    basis_val = local_b[i]
                    for d in range(dims):
                        C[d] += basis_val * cv[cp_idx, d]

            # === Evaluate C'(u) using derivative formula ===
            # C'(u) = degree * sum_i (Q_i * B_{i,degree-1}(u))
            # where Q_i = (P_{i+1} - P_i) / (kv[i+degree+1] - kv[i+1])
            C_prime = np.zeros(dims, dtype=np.float64)
            for i in range(degree):
                cp_idx = left_k + i
                if cp_idx + 1 < n_cv:
                    kv_idx = cp_idx + 1
                    denom  = kv[kv_idx + degree] - kv[kv_idx]
                    if denom != 0.0:
                        scale = degree * local_b_d1[i] / denom
                        for d in range(dims):
                            diff = cv[cp_idx + 1, d] - cv[cp_idx, d]
                            C_prime[d] += scale * diff

            # === Evaluate C''(u) using second derivative formula ===
            C_double_prime = np.zeros(dims, dtype=np.float64)
            if degree >= 2:
                for i in range(degree - 1):
                    cp_idx = left_k + i
                    if cp_idx + 2 < n_cv:
                        # Q_i for first derivative
                        kv_idx1 = cp_idx + 1
                        denom1  = kv[kv_idx1 + degree] - kv[kv_idx1]

                        kv_idx2 = cp_idx + 2
                        denom2  = kv[kv_idx2 + degree] - kv[kv_idx2]

                        if denom1 != 0.0 and denom2 != 0.0:
                            # Second derivative uses difference of Q vectors,
                            # divided by kv[i+degree+1] - kv[i+2]
                            kv_idx_dd = cp_idx + 2
                            denom_dd  = kv[kv_idx_dd + degree - 1] - kv[kv_idx_dd]
                            if denom_dd != 0.0:
                                scale = (
                                    degree
                                    * (degree - 1)
                                    * local_b_d2[i]
                                    / (denom_dd * denom1)
                                )
                                for d in range(dims):
                                    # Q_{i+1} - Q_i approximation
                                    q1 = (
                                        cv[cp_idx + 2, d] - cv[cp_idx + 1, d]
                                    ) / denom2
                                    q0 = (cv[cp_idx + 1, d] - cv[cp_idx, d]) / denom1
                                    C_double_prime[d] += scale * denom1 * (q1 - q0)

            # === Compute f(u) = (C(u) - Q) * C'(u) ===
            f_val = 0.0
            for d in range(dims):
                f_val += (C[d] - query[d]) * C_prime[d]

            # === Compute f'(u) = |C'(u)|^2 + (C(u) - Q) * C''(u) ===
            f_prime = 0.0
            for d in range(dims):
                f_prime += C_prime[d] * C_prime[d]
                f_prime += (C[d] - query[d]) * C_double_prime[d]

            # Avoid division by zero
            if abs(f_prime) < 1e-14:
                f_prime = 1e-14 if f_prime >= 0 else -1e-14

            # Newton step with trust region
            du = -f_val / f_prime
            if du > 0.5:
                du = 0.5
            elif du < -0.5:
                du = -0.5

            # Update parameter
            ui_new = ui + du

            # Handle bounds
            if periodic:
                while ui_new < 0:
                    ui_new += max_param
                while ui_new >= max_param:
                    ui_new -= max_param
            else:
                if ui_new < 0:
                    ui_new = 0.0
                elif ui_new > max_param:
                    ui_new = max_param

            # Check convergence
            if abs(du) < tolerance:
                ui = ui_new
                break

            ui = ui_new

        # === Final evaluation at converged parameter ===
        left_k  = _find_span(ui, kv, degree, limit)
        right_k = left_k + order

        # Recompute basis at final u
        for i in range(order + 1):
            local_b[i] = 0.0
        local_b[0] = 1.0

        for j in range(1, order):
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
                local_b[i] += f * (kv[ri] - ui)
                local_b[i + 1] = f * (ui - kv[ri - j])

        # Recompute basis at degree-1 for tangent
        for i in range(order):
            local_b_d1[i] = 0.0
        local_b_d1[0] = 1.0

        for j in range(1, degree):
            for i in range(j):
                local_bb_d1[i] = local_b_d1[i]
                local_b_d1[i]  = 0.0

            for i in range(j):
                ri    = right_k + i
                denom = kv[ri] - kv[ri - j]
                if denom != 0.0:
                    f = local_bb_d1[i] / denom
                else:
                    f = 0.0
                local_b_d1[i] += f * (kv[ri] - ui)
                local_b_d1[i + 1] = f * (ui - kv[ri - j])

        # Evaluate final C(u)
        for d in range(dims):
            C[d] = 0.0
        for i in range(order):
            cp_idx = left_k + i
            if cp_idx < n_cv:
                basis_val = local_b[i]
                for d in range(dims):
                    C[d] += basis_val * cv[cp_idx, d]

        # Evaluate final C'(u)
        for d in range(dims):
            C_prime[d] = 0.0
        for i in range(degree):
            cp_idx = left_k + i
            if cp_idx + 1 < n_cv:
                kv_idx = cp_idx + 1
                denom  = kv[kv_idx + degree] - kv[kv_idx]
                if denom != 0.0:
                    scale = degree * local_b_d1[i] / denom
                    for d in range(dims):
                        diff = cv[cp_idx + 1, d] - cv[cp_idx, d]
                        C_prime[d] += scale * diff

        # Store results
        u_out[qi] = ui
        for d in range(dims):
            points_out[qi, d]   = C[d]
            tangents_out[qi, d] = C_prime[d]

    return u_out, points_out, tangents_out


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
#            2D Newton-Raphson closest point (surface)                         #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
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
):
    """
    Parallel 2D Newton-Raphson closest point on B-spline surface.

    Finds (u, v) such that S(u, v) is closest to each query point.
    Solves the perpendicularity system:
        f1 = (S - Q) . S_u = 0
        f2 = (S - Q) . S_v = 0

    Parameters
    ----------
    queries : np.ndarray
        Query points (n_queries, dims), float64.
    u_init, v_init : np.ndarray
        Initial parameter guesses (n_queries,), float64.
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
        Maximum Newton iterations.
    tolerance : float
        Convergence tolerance for parameter change.

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
    n_cv_u    = cv.shape[0]
    n_cv_v    = cv.shape[1]
    dims      = cv.shape[2]

    order_u   = du + 1
    order_v   = dv + 1

    if periodic_u:
        limit_u = int(max_param_u) - 1
    else:
        limit_u = n_cv_u - du - 1

    if periodic_v:
        limit_v = int(max_param_v) - 1
    else:
        limit_v = n_cv_v - dv - 1

    u_out          = np.empty(n_queries,         dtype=np.float64)
    v_out          = np.empty(n_queries,         dtype=np.float64)
    points_out     = np.empty((n_queries, dims), dtype=np.float64)
    tangents_u_out = np.empty((n_queries, dims), dtype=np.float64)
    tangents_v_out = np.empty((n_queries, dims), dtype=np.float64)

    for qi in prange(n_queries):
        query = queries[qi]
        ui    = u_init[qi]
        vi    = v_init[qi]

        # Thread-local basis buffers
        b_u     = np.zeros(order_u + 1, dtype=np.float64)
        bb_u    = np.zeros(order_u + 1, dtype=np.float64)
        b_u_d1  = np.zeros(order_u,     dtype=np.float64)
        bb_u_d1 = np.zeros(order_u,     dtype=np.float64)
        b_u_d2  = np.zeros(order_u,     dtype=np.float64)
        bb_u_d2 = np.zeros(order_u,     dtype=np.float64)

        b_v     = np.zeros(order_v + 1, dtype=np.float64)
        bb_v    = np.zeros(order_v + 1, dtype=np.float64)
        b_v_d1  = np.zeros(order_v,     dtype=np.float64)
        bb_v_d1 = np.zeros(order_v,     dtype=np.float64)
        b_v_d2  = np.zeros(order_v,     dtype=np.float64)
        bb_v_d2 = np.zeros(order_v,     dtype=np.float64)

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

            # === U basis at degree du ===
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

            # === U basis at degree du-1 ===
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

            # === U basis at degree du-2 ===
            if du >= 2:
                for i in range(order_u):
                    b_u_d2[i] = 0.0
                b_u_d2[0] = 1.0
                for j in range(1, du - 1):
                    for i in range(j):
                        bb_u_d2[i] = b_u_d2[i]
                        b_u_d2[i]  = 0.0
                    for i in range(j):
                        ri    = right_u + i
                        denom = kv_u[ri] - kv_u[ri - j]
                        if denom != 0.0:
                            f = bb_u_d2[i] / denom
                        else:
                            f = 0.0
                        b_u_d2[i] += f * (kv_u[ri] - ui)
                        b_u_d2[i + 1] = f * (ui - kv_u[ri - j])

            # === V basis at degree dv ===
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

            # === V basis at degree dv-1 ===
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

            # === V basis at degree dv-2 ===
            if dv >= 2:
                for i in range(order_v):
                    b_v_d2[i] = 0.0
                b_v_d2[0] = 1.0
                for j in range(1, dv - 1):
                    for i in range(j):
                        bb_v_d2[i] = b_v_d2[i]
                        b_v_d2[i]  = 0.0
                    for i in range(j):
                        ri    = right_v + i
                        denom = kv_v[ri] - kv_v[ri - j]
                        if denom != 0.0:
                            f = bb_v_d2[i] / denom
                        else:
                            f = 0.0
                        b_v_d2[i] += f * (kv_v[ri] - vi)
                        b_v_d2[i + 1] = f * (vi - kv_v[ri - j])

            # === Evaluate S ===
            S = np.zeros(dims, dtype=np.float64)
            for iu in range(order_u):
                cp_u = left_u + iu
                if cp_u < n_cv_u:
                    bu_val = b_u[iu]
                    for iv in range(order_v):
                        cp_v = left_v + iv
                        if cp_v < n_cv_v:
                            w = bu_val * b_v[iv]
                            for d in range(dims):
                                S[d] += w * cv[cp_u, cp_v, d]

            # === Evaluate S_u ===
            S_u = np.zeros(dims, dtype=np.float64)
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
                                for d in range(dims):
                                    S_u[d] += sc * (
                                        cv[cp_u + 1, cp_v, d] - cv[cp_u, cp_v, d]
                                    )

            # === Evaluate S_v ===
            S_v = np.zeros(dims, dtype=np.float64)
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
                                for d in range(dims):
                                    S_v[d] += sc * (
                                        cv[cp_u, cp_v + 1, d] - cv[cp_u, cp_v, d]
                                    )

            # === Evaluate S_uu ===
            S_uu = np.zeros(dims, dtype=np.float64)
            if du >= 2:
                for iu in range(du - 1):
                    cp_u = left_u + iu
                    if cp_u + 2 < n_cv_u:
                        ki1 = cp_u + 1
                        d1  = kv_u[ki1 + du] - kv_u[ki1]
                        ki2 = cp_u + 2
                        d2  = kv_u[ki2 + du] - kv_u[ki2]
                        if d1 != 0.0 and d2 != 0.0:
                            # divided by kv_u[i+du+1] - kv_u[i+2]
                            kid = cp_u + 2
                            dd  = kv_u[kid + du - 1] - kv_u[kid]
                            if dd != 0.0:
                                for iv in range(order_v):
                                    cp_v = left_v + iv
                                    if cp_v < n_cv_v:
                                        sc = (
                                            du
                                            * (du - 1)
                                            * b_u_d2[iu]
                                            * b_v[iv]
                                            / (dd * d1)
                                        )
                                        for d in range(dims):
                                            q1 = (
                                                cv[cp_u + 2, cp_v, d]
                                                - cv[cp_u + 1, cp_v, d]
                                            ) / d2
                                            q0 = (
                                                cv[cp_u + 1, cp_v, d]
                                                - cv[cp_u, cp_v, d]
                                            ) / d1
                                            S_uu[d] += sc * d1 * (q1 - q0)

            # === Evaluate S_vv ===
            S_vv = np.zeros(dims, dtype=np.float64)
            if dv >= 2:
                for iu in range(order_u):
                    cp_u = left_u + iu
                    if cp_u < n_cv_u:
                        for iv in range(dv - 1):
                            cp_v = left_v + iv
                            if cp_v + 2 < n_cv_v:
                                ki1 = cp_v + 1
                                d1  = kv_v[ki1 + dv] - kv_v[ki1]
                                ki2 = cp_v + 2
                                d2  = kv_v[ki2 + dv] - kv_v[ki2]
                                if d1 != 0.0 and d2 != 0.0:
                                    # divided by kv_v[i+dv+1] - kv_v[i+2]
                                    kid = cp_v + 2
                                    dd  = kv_v[kid + dv - 1] - kv_v[kid]
                                    if dd != 0.0:
                                        sc = (
                                            dv
                                            * (dv - 1)
                                            * b_u[iu]
                                            * b_v_d2[iv]
                                            / (dd * d1)
                                        )
                                        for d in range(dims):
                                            q1 = (
                                                cv[cp_u, cp_v + 2, d]
                                                - cv[cp_u, cp_v + 1, d]
                                            ) / d2
                                            q0 = (
                                                cv[cp_u, cp_v + 1, d]
                                                - cv[cp_u, cp_v, d]
                                            ) / d1
                                            S_vv[d] += sc * d1 * (q1 - q0)

            # === Evaluate S_uv ===
            S_uv = np.zeros(dims, dtype=np.float64)
            if du >= 1 and dv >= 1:
                for iu in range(du):
                    cp_u = left_u + iu
                    if cp_u + 1 < n_cv_u:
                        kv_idx_u = cp_u + 1
                        denom_u  = kv_u[kv_idx_u + du] - kv_u[kv_idx_u]
                        if denom_u != 0.0:
                            for iv in range(dv):
                                cp_v = left_v + iv
                                if cp_v + 1 < n_cv_v:
                                    kv_idx_v = cp_v + 1
                                    denom_v  = kv_v[kv_idx_v + dv] - kv_v[kv_idx_v]
                                    if denom_v != 0.0:
                                        sc = (
                                            du
                                            * dv
                                            * b_u_d1[iu]
                                            * b_v_d1[iv]
                                            / (denom_u * denom_v)
                                        )
                                        for d in range(dims):
                                            dj1 = (
                                                cv[cp_u + 1, cp_v + 1, d]
                                                - cv[cp_u, cp_v + 1, d]
                                            )
                                            dj0 = (
                                                cv[cp_u + 1, cp_v, d]
                                                - cv[cp_u, cp_v, d]
                                            )
                                            S_uv[d] += sc * (dj1 - dj0)

            # === Perpendicularity conditions ===
            f1 = 0.0
            f2 = 0.0
            for d in range(dims):
                diff_d = S[d] - query[d]
                f1 += diff_d * S_u[d]
                f2 += diff_d * S_v[d]

            # === 2x2 Jacobian ===
            J11 = 0.0
            J12 = 0.0
            J22 = 0.0
            for d in range(dims):
                diff_d = S[d] - query[d]
                J11 += S_u[d] * S_u[d] + diff_d * S_uu[d]
                J12 += S_u[d] * S_v[d] + diff_d * S_uv[d]
                J22 += S_v[d] * S_v[d] + diff_d * S_vv[d]

            # === Cramer's rule ===
            det = J11 * J22 - J12 * J12
            if abs(det) < 1e-14:
                det = 1e-14 if det >= 0 else -1e-14

            delta_u = (-f1 * J22 + f2 * J12) / det
            delta_v = (f1 * J12 - f2 * J11) / det

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

            # Convergence
            conv = abs(delta_u)
            if abs(delta_v) > conv:
                conv = abs(delta_v)

            if conv < tolerance:
                ui = ui_new
                vi = vi_new
                break

            ui = ui_new
            vi = vi_new

        # === Final evaluation at converged (u, v) ===
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

        # Recompute U basis at degree du
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

        # Recompute U basis at degree du-1
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

        # Recompute V basis at degree dv
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

        # Recompute V basis at degree dv-1
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

        # Final S(u,v)
        for d in range(dims):
            S[d] = 0.0
        for iu in range(order_u):
            cp_u = left_u + iu
            if cp_u < n_cv_u:
                bu_val = b_u[iu]
                for iv in range(order_v):
                    cp_v = left_v + iv
                    if cp_v < n_cv_v:
                        w = bu_val * b_v[iv]
                        for d in range(dims):
                            S[d] += w * cv[cp_u, cp_v, d]

        # Final S_u
        for d in range(dims):
            S_u[d] = 0.0
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
                            for d in range(dims):
                                S_u[d] += sc * (
                                    cv[cp_u + 1, cp_v, d] - cv[cp_u, cp_v, d]
                                )

        # Final S_v
        for d in range(dims):
            S_v[d] = 0.0
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
                            for d in range(dims):
                                S_v[d] += sc * (
                                    cv[cp_u, cp_v + 1, d] - cv[cp_u, cp_v, d]
                                )

        # Store results
        u_out[qi] = ui
        v_out[qi] = vi
        for d in range(dims):
            points_out[qi, d]     = S[d]
            tangents_u_out[qi, d] = S_u[d]
            tangents_v_out[qi, d] = S_v[d]

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