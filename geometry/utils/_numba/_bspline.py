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
    left  = np.empty(n, dtype=kv.dtype)
    right = np.empty(n, dtype=kv.dtype)

    for i in range(n):
        left[i] = np.floor(u[i])
        if left[i] < 0:
            left[i] = 0
        elif left[i] > limit:
            left[i] = limit
        right[i] = left[i] + d + 1

    b  = np.empty((n, c), dtype=u.dtype)
    bb = np.empty((n, c), dtype=u.dtype)

    for i in range(n):
        for j in range(c):
            b[i, j] = 0.0
            bb[i, j] = 0.0

    for i in range(n):
        b[i, int(left[i])] = 1.0

    for j in range(1, d + 1):
        for i in range(j):
            for k in range(n):
                li = int(left[k])
                bb[k, li + i] = b[k, li + i]
                b[k, li] = 0.0

        for i in range(j):
            for k in range(n):
                li    = int(left[k])
                ri    = int(right[k])
                denom = kv[ri + i] - kv[ri + i - j]
                if denom != 0.0:
                    f = bb[k, li + i] / denom
                else:
                    f = 0.0
                b[k, li + i] = b[k, li + i] + f * (kv[ri + i] - u[k])
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
        left_k = int(np.floor(u[k]))
        if left_k < 0:
            left_k = 0
        elif left_k > limit:
            left_k = limit
        right_k = left_k + d1

        u_k     = u[k]

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
                local_b[i] = 0.0

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
        left_k = int(np.floor(u[k]))
        if left_k < 0:
            left_k = 0
        elif left_k > limit:
            left_k = limit
        right_k = left_k + d1

        u_k     = u[k]

        # Local basis buffer
        local_b  = np.zeros(d1 + 1, dtype=u.dtype)
        local_bb = np.zeros(d1 + 1, dtype=u.dtype)

        # Initialize
        local_b[0] = 1.0

        # Cox-de Boor recursion
        for j in range(1, d1):
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i] = 0.0

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
    n       = u.shape[0]
    cu      = control_points.shape[0]
    cv      = control_points.shape[1]
    dims    = control_points.shape[2]

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

        basis_u = np.zeros(du1 + 1, dtype=u.dtype)
        buf_u   = np.zeros(du1 + 1, dtype=u.dtype)
        basis_u[0] = 1.0

        for j in range(1, du1):
            for i in range(j):
                buf_u[i] = basis_u[i]
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

        basis_v = np.zeros(dv1 + 1, dtype=v.dtype)
        buf_v   = np.zeros(dv1 + 1, dtype=v.dtype)
        basis_v[0] = 1.0

        for j in range(1, dv1):
            for i in range(j):
                buf_v[i] = basis_v[i]
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
    n          = u.shape[0]
    cu         = control_points.shape[0]
    cv         = control_points.shape[1]
    dims       = control_points.shape[2]

    limit_u    = cu - du - 1
    limit_v    = cv - dv - 1
    du1        = du + 1
    dv1        = dv + 1

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

        basis_u = np.zeros(du1 + 1, dtype=u.dtype)
        buf_u   = np.zeros(du1 + 1, dtype=u.dtype)
        basis_u[0] = 1.0

        for j in range(1, du1):
            for i in range(j):
                buf_u[i] = basis_u[i]
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
                    buf_u_d1[i] = basis_u_d1[i]
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

        basis_v = np.zeros(dv1 + 1, dtype=v.dtype)
        buf_v   = np.zeros(dv1 + 1, dtype=v.dtype)
        basis_v[0] = 1.0

        for j in range(1, dv1):
            for i in range(j):
                buf_v[i] = basis_v[i]
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
                    buf_v_d1[i] = basis_v_d1[i]
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
        Derivative order (default 1 for first derivative).

    Returns
    -------
    np.ndarray
        Basis derivative values (n, c), float64.
    """
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
        left_k = int(np.floor(u[k]))
        if left_k < 0:
            left_k = 0
        elif left_k > limit:
            left_k = limit
        right_k = left_k + d1

        u_k     = u[k]

        # Compute basis at degree d and d-1
        # We need both for the derivative formula
        local_b  = np.zeros(d1 + 1, dtype=u.dtype)
        local_bb = np.zeros(d1 + 1, dtype=u.dtype)
        local_b[0] = 1.0

        # Store basis values at each degree level for derivative computation
        basis_levels = np.zeros((d1, d1 + 1), dtype=u.dtype)
        basis_levels[0, 0] = 1.0

        # Cox-de Boor recursion, storing each level
        for j in range(1, d1):
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i] = 0.0

            for i in range(j):
                ri    = right_k + i
                denom = kv[ri] - kv[ri - j]
                if denom != 0.0:
                    f = local_bb[i] / denom
                else:
                    f = 0.0
                local_b[i] += f * (kv[ri] - u_k)
                local_b[i + 1] = f * (u_k - kv[ri - j])

            # Store this level
            for i in range(j + 1):
                basis_levels[j, i] = local_b[i]

        # Compute first derivative using:
        # B'_{i,d}(u) = d * (B_{i,d-1}(u) / (kv[i+d] - kv[i]) - B_{i+1,d-1}(u) / (kv[i+d+1] - kv[i+1]))
        if order >= 1 and d >= 1:
            for i in range(d1):
                cp_idx = left_k + i
                if cp_idx < c:
                    deriv = 0.0

                    # Left term
                    if i < d:
                        kv_idx = left_k + i
                        denom  = kv[kv_idx + d] - kv[kv_idx]
                        if denom != 0.0:
                            deriv += d * basis_levels[d - 1, i] / denom

                    # Right term
                    if i > 0:
                        kv_idx = left_k + i
                        denom  = kv[kv_idx + d] - kv[kv_idx]
                        if denom != 0.0:
                            deriv -= d * basis_levels[d - 1, i - 1] / denom

                    db[k, cp_idx] = deriv

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
        left_k = int(np.floor(u[k]))
        if left_k < 0:
            left_k = 0
        elif left_k > limit:
            left_k = limit
        right_k = left_k + d1

        u_k     = u[k]

        # Compute basis at degree d-1 for derivative
        d_minus_1 = d
        local_b   = np.zeros(d_minus_1 + 1, dtype=u.dtype)
        local_bb  = np.zeros(d_minus_1 + 1, dtype=u.dtype)
        local_b[0] = 1.0

        for j in range(1, d_minus_1):
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i] = 0.0

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
        kv[i] = 0.0
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
            left_k = int(np.floor(ui))
            if left_k < 0:
                left_k = 0
            elif left_k > limit:
                left_k = limit
            right_k = left_k + order

            # === Compute basis functions at degree d ===
            for i in range(order + 1):
                local_b[i] = 0.0
            local_b[0] = 1.0

            for j in range(1, order):
                for i in range(j):
                    local_bb[i] = local_b[i]
                    local_b[i] = 0.0

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
                    local_b_d1[i] = 0.0

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
                        local_b_d2[i] = 0.0

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
                            # Second derivative uses difference of Q vectors
                            kv_idx_dd = cp_idx + 1
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
        left_k = int(np.floor(ui))
        if left_k < 0:
            left_k = 0
        elif left_k > limit:
            left_k = limit
        right_k = left_k + order

        # Recompute basis at final u
        for i in range(order + 1):
            local_b[i] = 0.0
        local_b[0] = 1.0

        for j in range(1, order):
            for i in range(j):
                local_bb[i] = local_b[i]
                local_b[i] = 0.0

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
                local_b_d1[i] = 0.0

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
            points_out[qi, d] = C[d]
            tangents_out[qi, d] = C_prime[d]

    return u_out, points_out, tangents_out


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
                    b_u[i] = 0.0
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
                    b_u_d1[i] = 0.0
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
                        b_u_d2[i] = 0.0
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
                    b_v[i] = 0.0
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
                    b_v_d1[i] = 0.0
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
                        b_v_d2[i] = 0.0
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
                            kid = cp_u + 1
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
                                    kid = cp_v + 1
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
                b_u[i] = 0.0
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
                b_u_d1[i] = 0.0
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
                b_v[i] = 0.0
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
                b_v_d1[i] = 0.0
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
            points_out[qi, d] = S[d]
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
            ex      = grid_pts[gi, 0] - ox
            ey      = grid_pts[gi, 1] - oy
            ez      = grid_pts[gi, 2] - oz

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

        Sx      = 0.0
        Sy      = 0.0
        Sz      = 0.0
        Sux     = 0.0
        Suy     = 0.0
        Suz     = 0.0
        Svx     = 0.0
        Svy     = 0.0
        Svz     = 0.0

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
                    b_u[i] = 0.0
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
                    b_u_d1[i] = 0.0
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
                    b_v[i] = 0.0
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
                    b_v_d1[i] = 0.0
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
            ex        = Sx - ox
            ey        = Sy - oy
            ez        = Sz - oz

            fx        = ey * dz - ez * dy
            fy        = ez * dx - ex * dz
            fz        = ex * dy - ey * dx

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
                b_u[i] = 0.0
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
                b_u_d1[i] = 0.0
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
                b_v[i] = 0.0
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
                b_v_d1[i] = 0.0
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
        ex          = Sx - ox
        ey          = Sy - oy
        ez          = Sz - oz

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
            u_out[k] = ui
            v_out[k] = vi
            t_out[k] = t_along
            hit_out[k] = True
        else:
            t_out[k] = np.nan

    return u_out, v_out, t_out, hit_out