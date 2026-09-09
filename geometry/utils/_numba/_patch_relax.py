"""
Numba-accelerated kernels for patch-based surface relaxation.

Implements the per-vertex machinery of "Patch-based Surface Relaxation"
(de Goes et al., Pixar, SIGGRAPH '18 Talks -- https://doi.org/10.1145/3214745.3214768):

    - :func:`_chain_vertex_rings` -- orders each vertex's 1-ring by winding
    - :func:`_compute_decal_maps` -- Sec.2, geodesic-polar stencil flattening
    - :func:`_compute_span_weights` -- Sec.3, span-aware edge weights
    - :func:`_relax_step` -- Sec.4/Sec.5, one Jacobi iteration with optional
      surface-constrained (volume preserving) displacement

Every kernel is ``prange``-parallel over vertices and writes into
pre-allocated outputs, so a relaxation loop allocates nothing per
iteration.  The 3x3 Procrustes rotation of Sec.4 reuses
:func:`~cgmath.geometry.utils._numba._delta_mush._polar_decompose_3x3`
rather than dispatching ``np.linalg.svd`` per vertex; that helper already
returns a proper rotation for the improper polar factor a folded-over
stencil produces.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from cgmath.geometry.utils._numba._delta_mush import _polar_decompose_3x3


_TWO_PI = 2.0 * math.pi


# =========================== ring adjacency ============================== #


@njit(cache=True)
def _find_fan_start(pair_prev, pair_next, lo, n_pairs):
    """Return the neighbour no segment points to -- an open fan's first spoke.

    Falls back to the first pair when every neighbour is a target, which
    is the closed-ring case where any starting point will do.
    """
    for a in range(n_pairs):
        p         = pair_prev[lo + a]
        is_target = False
        for b in range(n_pairs):
            if pair_next[lo + b] == p:
                is_target = True
                break
        if not is_target:
            return p
    return pair_prev[lo]


@njit(cache=True)
def _walk_vertex_ring(pair_prev, pair_next, lo, n_pairs, start, out_row, used_row):
    """Follow ``prev -> next`` segments from *start*, filling *out_row*.

    Returns the number of neighbours written and whether the ring closed
    back onto *start*.
    """
    capacity   = out_row.shape[0]
    out_row[0] = start
    length     = 1
    cur        = start

    for _ in range(n_pairs):
        sel = -1
        for b in range(n_pairs):
            if not used_row[b] and pair_prev[lo + b] == cur:
                sel = b
                break
        if sel < 0:
            break

        used_row[sel] = True
        nxt           = pair_next[lo + sel]
        if nxt == start:
            return length, True
        if length >= capacity:
            break

        out_row[length] = nxt
        length += 1
        cur = nxt

    return length, False


@njit(parallel=True, cache=True)
def _chain_vertex_rings(
    pair_offsets: np.ndarray,
    pair_prev:    np.ndarray,
    pair_next:    np.ndarray,
    out_ring:     np.ndarray,
    out_valence:  np.ndarray,
    out_boundary: np.ndarray,
    used:         np.ndarray,
) -> None:
    """Chain per-face ``(prev, next)`` corner pairs into an ordered 1-ring.

    Each face incident to vertex ``i`` contributes the ring segment
    ``prev -> next``.  Following those segments walks the 1-ring in
    winding order.  Interior vertices close the loop; boundary vertices
    form an open fan, which is detected by starting the walk at the
    neighbour that is never a segment's ``next``.

    Parameters
    ----------
    pair_offsets : (N + 1,) int64
        CSR offsets into ``pair_prev`` / ``pair_next``.
    pair_prev, pair_next : (P,) int32
        Corner pairs grouped by vertex.
    out_ring : (N, K) int32
        Pre-filled with ``-1``; receives the ordered neighbour indices.
    out_valence : (N,) int32
        Receives the number of valid entries per ring.
    out_boundary : (N,) bool
        Receives True when the ring did not close.
    used : (N, K) bool
        Scratch buffer, one row per vertex.
    """
    n_verts = out_valence.shape[0]

    for i in prange(n_verts):
        lo      = pair_offsets[i]
        n_pairs = pair_offsets[i + 1] - lo

        if n_pairs <= 0:
            out_valence[i]  = 0
            out_boundary[i] = True
            continue

        for a in range(n_pairs):
            used[i, a] = False

        start = _find_fan_start(pair_prev, pair_next, lo, n_pairs)
        length, closed = _walk_vertex_ring(
            pair_prev, pair_next, lo, n_pairs, start, out_ring[i], used[i]
        )

        out_valence[i]  = length
        out_boundary[i] = not closed


# ============================= decal maps ================================ #


@njit(fastmath=True, cache=True)
def _edge_direction(points, i, j):
    """Return the unit vector from vertex ``i`` to ``j`` and its length."""
    dx     = points[j, 0] - points[i, 0]
    dy     = points[j, 1] - points[i, 1]
    dz     = points[j, 2] - points[i, 2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
    inv    = 1.0 / length
    return dx * inv, dy * inv, dz * inv, length


@njit(fastmath=True, cache=True)
def _stencil_angle(points, i, ja, jb):
    """Angle at vertex ``i`` between the edges to ``ja`` and ``jb``."""
    ax, ay, az, _ = _edge_direction(points, i, ja)
    bx, by, bz, _ = _edge_direction(points, i, jb)
    dot = ax * bx + ay * by + az * bz
    return math.acos(min(max(dot, -1.0), 1.0))


@njit(parallel=True, fastmath=True, cache=True)
def _compute_decal_maps(
    points:      np.ndarray,
    ring:        np.ndarray,
    valence:     np.ndarray,
    is_boundary: np.ndarray,
    out_decals:  np.ndarray,
) -> None:
    """Flatten every vertex's edge stencil into geodesic polar coordinates (Sec.2).

    Edge lengths are preserved exactly and the angles between consecutive
    edges are uniformly rescaled to sum to ``2*pi`` (``pi`` for boundary
    vertices, which map to a half-disk).

    Parameters
    ----------
    points : (N, 3) float64
    ring : (N, K) int32
        Winding-ordered neighbours, ``-1`` padded.
    valence : (N,) int32
    is_boundary : (N,) bool
    out_decals : (N, K, 2) float64
        Pre-allocated output, written in-place.
    """
    n_verts = points.shape[0]

    for i in prange(n_verts):
        val = valence[i]
        for k in range(out_decals.shape[1]):
            out_decals[i, k, 0] = 0.0
            out_decals[i, k, 1] = 0.0
        if val == 0:
            continue

        # a boundary fan has one fewer angular gap than it has edges
        n_gaps = val - 1 if is_boundary[i] else val

        total  = 0.0
        for k in range(n_gaps):
            kn = k + 1 if k + 1 < val else 0
            total += _stencil_angle(points, i, ring[i, k], ring[i, kn])

        target = math.pi if is_boundary[i] else _TWO_PI
        scale  = target / total if total > 1e-8 else 1.0

        cum = 0.0
        for k in range(val):
            _, _, _, length = _edge_direction(points, i, ring[i, k])
            out_decals[i, k, 0] = length * math.cos(cum)
            out_decals[i, k, 1] = length * math.sin(cum)

            if k < n_gaps:
                kn = k + 1 if k + 1 < val else 0
                cum += _stencil_angle(points, i, ring[i, k], ring[i, kn]) * scale


# ========================== span-aware weights =========================== #


@njit(fastmath=True, cache=True)
def _span_weight(decal_row, val, k):
    """Span-aware weight of stencil edge *k* (Sec.3).

    Builds an orthonormal 2D frame aligned to the edge, projects the rest
    of the stencil onto it, and combines the most negative tangential
    extent (circulation) with the perpendicular spread (flux).
    """
    if val < 2:
        return 0.0

    ex   = decal_row[k, 0]
    ey   = decal_row[k, 1]
    elen = math.sqrt(ex * ex + ey * ey) + 1e-12
    tx   = ex / elen
    ty   = ey / elen
    nx   = -ty
    ny   = tx

    # seed the extents from a real neighbour so a stencil that sits
    # wholly to one side keeps its true spread rather than picking up
    # the origin as a spurious extremum
    first = 1 if k == 0 else 0
    fx    = decal_row[first, 0]
    fy    = decal_row[first, 1]
    c_min = tx * fx + ty * fy
    f_max = nx * fx + ny * fy
    f_min = f_max

    for m in range(val):
        if m == k:
            continue
        mx    = decal_row[m, 0]
        my    = decal_row[m, 1]
        ct    = tx * mx + ty * my
        cn    = nx * mx + ny * my
        c_min = min(c_min, ct)
        f_max = max(f_max, cn)
        f_min = min(f_min, cn)

    return abs(c_min) * max(f_max - f_min, 1e-8)


@njit(parallel=True, fastmath=True, cache=True)
def _compute_span_weights(
    decals:      np.ndarray,
    valence:     np.ndarray,
    out_weights: np.ndarray,
) -> None:
    """Span-aware edge weights from a decal map (Sec.3).

    For every stencil edge an orthonormal 2D frame is aligned to it and
    the remaining stencil edges are projected onto that frame.  The
    weight combines the most negative tangential extent (flow
    circulation) with the perpendicular spread (in/out flux):

        ``w_ij = |c_min| * (f_max - f_min)``

    Weights are normalised so each stencil sums to one, falling back to
    uniform weights on degenerate stencils.

    Parameters
    ----------
    decals : (N, K, 2) float64
    valence : (N,) int32
    out_weights : (N, K) float64
        Pre-allocated output, written in-place.
    """
    n_verts = valence.shape[0]

    for i in prange(n_verts):
        val = valence[i]
        for k in range(out_weights.shape[1]):
            out_weights[i, k] = 0.0
        if val == 0:
            continue

        total = 0.0
        for k in range(val):
            w                 = _span_weight(decals[i], val, k)
            out_weights[i, k] = w
            total += w

        if total > 1e-12:
            inv = 1.0 / total
            for k in range(val):
                out_weights[i, k] *= inv
        else:
            uniform = 1.0 / val
            for k in range(val):
                out_weights[i, k] = uniform


# =========================== surface lifting ============================= #


@njit(fastmath=True, cache=True)
def _lift_to_3d(tx, ty, decals, ring, points, i, val, boundary):
    """Lift a 2D decal-space target back onto the surface (Sec.5).

    Walks the stencil's triangle fan, clamps the barycentric coordinates
    of the target into each triangle, and keeps the candidate whose 2D
    reconstruction lands closest to the target.  Because it interpolates
    the fan rather than projecting along a normal, the lifted point
    always stays on the surface.
    """
    best_x    = points[i, 0]
    best_y    = points[i, 1]
    best_z    = points[i, 2]
    best_dist = 1e30

    n_tri = val - 1 if boundary else val

    for k in range(n_tri):
        kn    = k + 1 if k + 1 < val else 0
        b0    = decals[i, k, 0]
        b1    = decals[i, k, 1]
        c0    = decals[i, kn, 0]
        c1    = decals[i, kn, 1]

        d00   = c0 * c0 + c1 * c1
        d01   = c0 * b0 + c1 * b1
        d02   = c0 * tx + c1 * ty
        d11   = b0 * b0 + b1 * b1
        d12   = b0 * tx + b1 * ty

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-15:
            continue
        inv = 1.0 / denom

        ub  = (d11 * d02 - d01 * d12) * inv
        vb  = (d00 * d12 - d01 * d02) * inv
        ub  = min(max(ub, 0.0),            1.0)
        vb  = min(max(vb, 0.0),            1.0)
        wb  = min(max(1.0 - ub - vb, 0.0), 1.0)

        s   = ub + vb + wb
        if s > 1e-12:
            inv_s = 1.0 / s
            ub *= inv_s
            vb *= inv_s
            wb *= inv_s

        p2x  = vb * b0 + ub * c0
        p2y  = vb * b1 + ub * c1
        dx   = p2x - tx
        dy   = p2y - ty
        dist = dx * dx + dy * dy

        if dist < best_dist:
            j0        = ring[i, k]
            j1        = ring[i, kn]
            best_dist = dist
            best_x    = wb * points[i, 0] + vb * points[j0, 0] + ub * points[j1, 0]
            best_y    = wb * points[i, 1] + vb * points[j0, 1] + ub * points[j1, 1]
            best_z    = wb * points[i, 2] + vb * points[j0, 2] + ub * points[j1, 2]

    return best_x, best_y, best_z


# =========================== relaxation step ============================= #


@njit(fastmath=True, cache=True)
def _surface_candidate(
    points,
    ring,
    weights,
    rest_decals,
    posed_decals,
    i,
    val,
    boundary,
    alpha,
    step_size,
):
    """The Sec.4 displacement solved inside the decal chart instead of 3D (Sec.5).

    Working in the chart and lifting back through the stencil's triangle
    fan keeps the vertex on the surface, so relaxation cannot deflate the
    shape.
    """
    up0     = 0.0
    up1     = 0.0
    uh0     = 0.0
    uh1     = 0.0
    sin_acc = 0.0
    cos_acc = 0.0

    for k in range(val):
        w  = weights[i, k]
        u0 = posed_decals[i, k, 0]
        u1 = posed_decals[i, k, 1]
        h0 = rest_decals[i, k, 0]
        h1 = rest_decals[i, k, 1]

        up0 += w * u0
        up1 += w * u1
        uh0 += w * h0
        uh1 += w * h1

        sin_acc += w * (h0 * u1 - h1 * u0)
        cos_acc += w * (h0 * u0 + h1 * u1)

    # 2D Procrustes rotation carrying the rest stencil onto the posed one
    mag = math.sqrt(sin_acc * sin_acc + cos_acc * cos_acc)
    if mag > 1e-12:
        cs = cos_acc / mag
        sn = sin_acc / mag
    else:
        cs = 1.0
        sn = 0.0

    rh0 = cs * uh0 - sn * uh1
    rh1 = sn * uh0 + cs * uh1

    t0  = step_size * (up0 - alpha * rh0)
    t1  = step_size * (up1 - alpha * rh1)

    return _lift_to_3d(t0, t1, posed_decals, ring, points, i, val, boundary)


@njit(parallel=True, fastmath=True, cache=True)
def _relax_step(
    points:        np.ndarray,
    rest_points:   np.ndarray,
    ring:          np.ndarray,
    valence:       np.ndarray,
    is_boundary:   np.ndarray,
    weights:       np.ndarray,
    rest_vectors:  np.ndarray,
    rest_decals:   np.ndarray,
    posed_decals:  np.ndarray,
    step_scale:    np.ndarray,
    alpha:         float,
    step_size:     float,
    surface_blend: float,
    use_surface:   bool,
    polar_iters:   int,
    out_points:    np.ndarray,
) -> None:
    """One Jacobi relaxation iteration (Sec.4, plus Sec.5 when *use_surface*).

    Every vertex is updated from the previous iterate only, so the sweep
    is order-independent and safe to run in parallel.

    Parameters
    ----------
    points : (N, 3) float64
        Current iterate.
    rest_points : (N, 3) float64
        Baseline patch layout.
    ring, valence, is_boundary
        Ordered ring adjacency.
    weights : (N, K) float64
        Rest-based span-aware weights.
    rest_vectors : (N, 3) float64
        ``sum_k w_k (rest_j - rest_i)``, precomputed once.
    rest_decals, posed_decals : (N, K, 2) float64
        Decal maps of the baseline and current iterate.  *posed_decals*
        is only read when *use_surface* is set.
    step_scale : (N,) float64
        Per-vertex step multiplier; ``0`` pins the vertex.
    alpha : float
        Blends the baseline-layout correction against plain relaxation.
    step_size : float
        Explicit update fraction -- the paper's half step.
    surface_blend : float
        Mixes the surface-constrained result over the 3D one.
    out_points : (N, 3) float64
        Pre-allocated output, written in-place.
    """
    n_verts = points.shape[0]

    for i in prange(n_verts):
        xi0   = points[i, 0]
        xi1   = points[i, 1]
        xi2   = points[i, 2]

        val   = valence[i]
        scale = step_scale[i]
        if val == 0 or scale <= 0.0:
            out_points[i, 0] = xi0
            out_points[i, 1] = xi1
            out_points[i, 2] = xi2
            continue

        ri0 = rest_points[i, 0]
        ri1 = rest_points[i, 1]
        ri2 = rest_points[i, 2]

        # weighted posed offset and rest -> posed cross-covariance
        vp0 = 0.0
        vp1 = 0.0
        vp2 = 0.0
        c00 = 0.0
        c01 = 0.0
        c02 = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = 0.0
        c20 = 0.0
        c21 = 0.0
        c22 = 0.0

        for k in range(val):
            j  = ring[i, k]
            w  = weights[i, k]

            cx = points[j, 0] - xi0
            cy = points[j, 1] - xi1
            cz = points[j, 2] - xi2
            hx = rest_points[j, 0] - ri0
            hy = rest_points[j, 1] - ri1
            hz = rest_points[j, 2] - ri2

            vp0 += w * cx
            vp1 += w * cy
            vp2 += w * cz

            c00 += w * cx * hx
            c01 += w * cx * hy
            c02 += w * cx * hz
            c10 += w * cy * hx
            c11 += w * cy * hy
            c12 += w * cy * hz
            c20 += w * cz * hx
            c21 += w * cz * hy
            c22 += w * cz * hz

        # the polar factor of C is the least-squares rotation carrying the
        # rest stencil onto the posed one; the helper already flips an
        # improper factor, which a folded-over stencil would otherwise
        # turn into a mirror
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = _polar_decompose_3x3(
            c00, c01, c02, c10, c11, c12, c20, c21, c22, polar_iters
        )

        hv0   = rest_vectors[i, 0]
        hv1   = rest_vectors[i, 1]
        hv2   = rest_vectors[i, 2]
        rv0   = r00 * hv0 + r01 * hv1 + r02 * hv2
        rv1   = r10 * hv0 + r11 * hv1 + r12 * hv2
        rv2   = r20 * hv0 + r21 * hv1 + r22 * hv2

        cand0 = xi0 + step_size * (vp0 - alpha * rv0)
        cand1 = xi1 + step_size * (vp1 - alpha * rv1)
        cand2 = xi2 + step_size * (vp2 - alpha * rv2)

        if use_surface:
            sx, sy, sz = _surface_candidate(
                points,
                ring,
                weights,
                rest_decals,
                posed_decals,
                i,
                val,
                is_boundary[i],
                alpha,
                step_size,
            )

            keep  = 1.0 - surface_blend
            cand0 = keep * cand0 + surface_blend * sx
            cand1 = keep * cand1 + surface_blend * sy
            cand2 = keep * cand2 + surface_blend * sz

        out_points[i, 0] = xi0 + scale * (cand0 - xi0)
        out_points[i, 1] = xi1 + scale * (cand1 - xi1)
        out_points[i, 2] = xi2 + scale * (cand2 - xi2)