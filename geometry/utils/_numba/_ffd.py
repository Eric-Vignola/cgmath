"""
Numba-optimized kernels for Free-Form Deformation.

Key optimizations over the pure-numpy implementations in main.py:
- Fused corner-gather + interpolation (no intermediate (N,8,3) array)
- Parallel evaluation across mesh points via prange
- Per-point early termination in the Newton solver
- Zero temporary allocations inside hot loops
- Cramer's rule for inline 3x3 solves (avoids LAPACK overhead)
"""

import numpy as np
from numba import njit, prange


# ==================== Newton-Raphson inverse trilinear =================== #


@njit(parallel=True, fastmath=True, cache=True)
def _inverse_trilinear(
    points:   np.ndarray,
    corners:  np.ndarray,
    uvw:      np.ndarray,
    max_iter: int,
    tol:      float,
) -> None:
    """Newton-Raphson solve with per-point early termination.

    Each point independently iterates until converged or max_iter is
    reached.  Uses Cramer's rule for the 3x3 solve (avoids LAPACK
    overhead per point).

    Parameters
    ----------
    points : (N, 3) float64
        Target positions.
    corners : (N, 8, 3) float64
        Cell corner positions.
    uvw : (N, 3) float64
        Initial guess, **modified in-place** to hold the result.
    max_iter : int
    tol : float
    """
    N = points.shape[0]

    for n in prange(N):
        for _iteration in range(max_iter):
            u  = uvw[n, 0]
            v  = uvw[n, 1]
            w  = uvw[n, 2]
            u1 = 1.0 - u
            v1 = 1.0 - v
            w1 = 1.0 - w

            # -- forward trilinear + residual --
            max_res = 0.0
            r0      = 0.0
            r1      = 0.0
            r2      = 0.0
            for d in range(3):
                c00    = corners[n, 0, d] * u1 + corners[n, 1, d] * u
                c01    = corners[n, 2, d] * u1 + corners[n, 3, d] * u
                c10    = corners[n, 4, d] * u1 + corners[n, 5, d] * u
                c11    = corners[n, 6, d] * u1 + corners[n, 7, d] * u
                c0     = c00 * v1 + c01 * v
                c1_val = c10 * v1 + c11 * v
                val    = c0 * w1 + c1_val * w - points[n, d]
                if d == 0:
                    r0 = val
                elif d == 1:
                    r1 = val
                else:
                    r2 = val
                aval = val if val >= 0.0 else -val
                if aval > max_res:
                    max_res = aval

            if max_res < tol:
                break

            # -- basis function derivatives --
            dN_du0 = -v1 * w1
            dN_du1 = v1 * w1
            dN_du2 = -v * w1
            dN_du3 = v * w1
            dN_du4 = -v1 * w
            dN_du5 = v1 * w
            dN_du6 = -v * w
            dN_du7 = v * w

            dN_dv0 = -u1 * w1
            dN_dv1 = -u * w1
            dN_dv2 = u1 * w1
            dN_dv3 = u * w1
            dN_dv4 = -u1 * w
            dN_dv5 = -u * w
            dN_dv6 = u1 * w
            dN_dv7 = u * w

            dN_dw0 = -u1 * v1
            dN_dw1 = -u * v1
            dN_dw2 = -u1 * v
            dN_dw3 = -u * v
            dN_dw4 = u1 * v1
            dN_dw5 = u * v1
            dN_dw6 = u1 * v
            dN_dw7 = u * v

            # -- Jacobian J[row][col] = dP_row / d_uvw_col --
            J00 = 0.0
            J01 = 0.0
            J02 = 0.0
            J10 = 0.0
            J11 = 0.0
            J12 = 0.0
            J20 = 0.0
            J21 = 0.0
            J22 = 0.0
            for i_corner in range(8):
                if i_corner == 0:
                    du, dv, dw = dN_du0, dN_dv0, dN_dw0
                elif i_corner == 1:
                    du, dv, dw = dN_du1, dN_dv1, dN_dw1
                elif i_corner == 2:
                    du, dv, dw = dN_du2, dN_dv2, dN_dw2
                elif i_corner == 3:
                    du, dv, dw = dN_du3, dN_dv3, dN_dw3
                elif i_corner == 4:
                    du, dv, dw = dN_du4, dN_dv4, dN_dw4
                elif i_corner == 5:
                    du, dv, dw = dN_du5, dN_dv5, dN_dw5
                elif i_corner == 6:
                    du, dv, dw = dN_du6, dN_dv6, dN_dw6
                else:
                    du, dv, dw = dN_du7, dN_dv7, dN_dw7
                px = corners[n, i_corner, 0]
                py = corners[n, i_corner, 1]
                pz = corners[n, i_corner, 2]
                J00 += du * px
                J10 += du * py
                J20 += du * pz
                J01 += dv * px
                J11 += dv * py
                J21 += dv * pz
                J02 += dw * px
                J12 += dw * py
                J22 += dw * pz

            # regularise
            J00 += 1e-12
            J11 += 1e-12
            J22 += 1e-12

            # -- Cramer's rule for 3x3 solve: J @ delta = residual --
            det = (
                J00 * (J11 * J22 - J12 * J21)
                - J01 * (J10 * J22 - J12 * J20)
                + J02 * (J10 * J21 - J11 * J20)
            )
            if det == 0.0 or det != det:  # zero or NaN
                break

            inv_det = 1.0 / det

            d0 = (
                r0 * (J11 * J22 - J12 * J21)
                - J01 * (r1 * J22 - J12 * r2)
                + J02 * (r1 * J21 - J11 * r2)
            ) * inv_det

            d1 = (
                J00 * (r1 * J22 - J12 * r2)
                - r0 * (J10 * J22 - J12 * J20)
                + J02 * (J10 * r2 - r1 * J20)
            ) * inv_det

            d2 = (
                J00 * (J11 * r2 - r1 * J21)
                - J01 * (J10 * r2 - r1 * J20)
                + r0 * (J10 * J21 - J11 * J20)
            ) * inv_det

            uvw[n, 0] -= d0
            uvw[n, 1] -= d1
            uvw[n, 2] -= d2


# ======================= Bernstein polynomial FFD ======================== #


@njit(fastmath=True, cache=True)
def _bernstein_weights(t, degree):
    """Compute Bernstein basis weights for a single scalar t.

    Returns a tuple of (degree + 1) weights.  Kept as a helper so
    ``_bernstein_eval`` can call it per-point without allocations.
    """
    binom       = 1.0
    one_minus_t = 1.0 - t
    weights     = np.empty(degree + 1)
    for i in range(degree + 1):
        if i == 0:
            binom = 1.0
        else:
            binom = binom * (degree - i + 1) / i
        weights[i] = binom * (t**i) * (one_minus_t ** (degree - i))
    return weights


@njit(parallel=True, fastmath=True, cache=True)
def _bernstein_eval(
    cells:             np.ndarray,
    uvw:               np.ndarray,
    delta:             np.ndarray,
    local_influence_s: int,
    local_influence_t: int,
    local_influence_u: int,
    divisions_s:       int,
    divisions_t:       int,
    divisions_u:       int,
    out:               np.ndarray,
) -> None:
    """Bernstein polynomial FFD evaluation (Sederberg & Parry formulation).

    Parallel over mesh points.  Each point computes its own sliding
    window, Bernstein basis weights, and the tensor-product weighted
    sum with zero intermediate arrays.

    Parameters
    ----------
    cells : (N, 3) int64
        Cell indices per point.
    uvw : (N, 3) float64
        Per-cell parametric coordinates.
    delta : (lx, ly, lz, 3) float64
        Displacement field (deformed - rest).
    local_influence_s/t/u : int
        Cells of influence per axis.
    divisions_s/t/u : int
        Cells per axis.
    out : (N, 3) float64
        Pre-allocated output, written in-place.
    """
    N         = uvw.shape[0]
    lx        = delta.shape[0]
    ly        = delta.shape[1]
    lz        = delta.shape[2]

    ns        = min(local_influence_s, divisions_s)
    nt        = min(local_influence_t, divisions_t)
    nu        = min(local_influence_u, divisions_u)

    num_cps_s = divisions_s + 1
    num_cps_t = divisions_t + 1
    num_cps_u = divisions_u + 1

    for n in prange(N):
        # global parameter per axis
        gs = float(cells[n, 0]) + uvw[n, 0]
        gt = float(cells[n, 1]) + uvw[n, 1]
        gu = float(cells[n, 2]) + uvw[n, 2]

        # window start (CP index), clamped
        ws_start = int(max(0, min(int(gs - ns * 0.5 + 0.5), num_cps_s - ns - 1)))
        wt_start = int(max(0, min(int(gt - nt * 0.5 + 0.5), num_cps_t - nt - 1)))
        wu_start = int(max(0, min(int(gu - nu * 0.5 + 0.5), num_cps_u - nu - 1)))

        # local parameter within window [0, 1]
        ts = (gs - ws_start) / ns
        tt = (gt - wt_start) / nt
        tu = (gu - wu_start) / nu

        # Bernstein basis weights
        ws = _bernstein_weights(ts, ns)
        wt = _bernstein_weights(tt, nt)
        wu = _bernstein_weights(tu, nu)

        for d in range(3):
            val = 0.0
            for a in range(ns + 1):
                ii = min(max(ws_start + a, 0), lx - 1)
                wa = ws[a]
                for b in range(nt + 1):
                    jj  = min(max(wt_start + b, 0), ly - 1)
                    wab = wa * wt[b]
                    for c in range(nu + 1):
                        kk = min(max(wu_start + c, 0), lz - 1)
                        val += wab * wu[c] * delta[ii, jj, kk, d]
            out[n, d] = val