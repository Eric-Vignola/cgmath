"""
Numba-accelerated kernels for the Delta Mush deformation operator.

Numba kernels (parallel via ``prange``):
    - :func:`_encode_local_deltas` -- per-vertex tangent-frame projection
    - :func:`_decode_local_deltas` -- inverse: rebuild world-space points
    - :func:`_blend_deltas` -- linear blend between raw and reconstructed
    - :func:`_batch_procrustes_rotations` -- drop-in for the SVD step in
      :class:`cgmath.constraints.ProcrustesData`; uses Higham's
      iterative polar decomposition to avoid per-cluster
      ``np.linalg.svd`` dispatch overhead

Numpy fallbacks for the kernels above are also provided so callers can
gracefully degrade when the numba kernels fail to compile/dispatch.

Key optimizations over pure-numpy implementations:
- Per-vertex tangent-frame construction with zero allocations
- Parallel delta encode/decode via prange
- Analytic 3x3 polar decomposition (Higham's iterative algorithm) -- avoids
  per-cluster np.linalg.svd dispatch overhead which dominates batch Procrustes
- Fused covariance accumulation + frame extraction
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ========================== tangent frames =============================== #


@njit(fastmath=True, cache=True)
def _orthonormal_frame(t0, t1, t2, n0, n1, n2):
    """Gram-Schmidt orthonormalisation given a tangent and a normal hint.

    Returns (t, b, n) as a 3x3 row-stacked matrix-of-floats tuple.
    Falls back to a canonical X-axis tangent on degenerate input.
    """
    # normalise normal
    nm = (n0 * n0 + n1 * n1 + n2 * n2) ** 0.5
    if nm < 1e-12:
        n0, n1, n2 = 0.0, 1.0, 0.0
        nm = 1.0
    n0 /= nm
    n1 /= nm
    n2 /= nm

    # project tangent onto plane perpendicular to normal
    dot = t0 * n0 + t1 * n1 + t2 * n2
    t0 -= dot * n0
    t1 -= dot * n1
    t2 -= dot * n2

    tm = (t0 * t0 + t1 * t1 + t2 * t2) ** 0.5
    if tm < 1e-12:
        # pick a robust orthogonal axis to the normal
        if abs(n0) < 0.9:
            t0, t1, t2 = 1.0 - n0 * n0, -n0 * n1, -n0 * n2
        else:
            t0, t1, t2 = -n1 * n0, 1.0 - n1 * n1, -n1 * n2
        tm = (t0 * t0 + t1 * t1 + t2 * t2) ** 0.5
        if tm < 1e-12:
            tm = 1.0
    t0 /= tm
    t1 /= tm
    t2 /= tm

    # bitangent = normal x tangent (right-handed)
    b0 = n1 * t2 - n2 * t1
    b1 = n2 * t0 - n0 * t2
    b2 = n0 * t1 - n1 * t0

    return t0, t1, t2, b0, b1, b2, n0, n1, n2


@njit(parallel=True, fastmath=True, cache=True)
def _encode_local_deltas(
    rest_points:   np.ndarray,
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
    out_deltas:    np.ndarray,
) -> None:
    """Encode each rest vertex displacement in its smoothed local tangent frame.

    For every vertex ``i`` builds an orthonormal frame ``(T, B, N)`` using:

        N = mean of normalised vectors from smooth[i] to its smooth neighbours
        T = vector from smooth[i] to its first valid smooth neighbour
        B = N x T

    Then stores ``rest[i] - smooth[i]`` projected into that frame.

    Parameters
    ----------
    rest_points : (N, 3) float64
    smooth_points : (N, 3) float64
        Smoothed rest positions (output of ``MeshData.smooth()``).
    neighbors : (N, K) int32
        Per-vertex neighbour matrix; ``-1`` marks padding slots.
    out_deltas : (N, 3) float64
        Pre-allocated output, written in-place.
    """
    n_verts     = rest_points.shape[0]
    n_neighbors = neighbors.shape[1]

    for i in prange(n_verts):
        # accumulate average normal direction from neighbour offsets
        nx = 0.0
        ny = 0.0
        nz = 0.0
        # tangent hint = first valid neighbour offset
        tx            = 0.0
        ty            = 0.0
        tz            = 0.0
        found_tangent = False

        for j in range(n_neighbors):
            ni = neighbors[i, j]
            if ni < 0:
                continue
            dx  = smooth_points[ni, 0] - smooth_points[i, 0]
            dy  = smooth_points[ni, 1] - smooth_points[i, 1]
            dz  = smooth_points[ni, 2] - smooth_points[i, 2]
            mag = (dx * dx + dy * dy + dz * dz) ** 0.5
            if mag < 1e-12:
                continue

            inv = 1.0 / mag
            nx += dx * inv
            ny += dy * inv
            nz += dz * inv

            if not found_tangent:
                tx            = dx
                ty            = dy
                tz            = dz
                found_tangent = True

        # rest displacement (world-space)
        wx = rest_points[i, 0] - smooth_points[i, 0]
        wy = rest_points[i, 1] - smooth_points[i, 1]
        wz = rest_points[i, 2] - smooth_points[i, 2]

        if not found_tangent:
            # isolated vertex -- store identity delta (world == local)
            out_deltas[i, 0] = wx
            out_deltas[i, 1] = wy
            out_deltas[i, 2] = wz
            continue

        t0, t1, t2, b0, b1, b2, n0, n1, n2 = _orthonormal_frame(tx, ty, tz, nx, ny, nz)

        # project world delta into local frame: row . delta
        out_deltas[i, 0] = wx * t0 + wy * t1 + wz * t2
        out_deltas[i, 1] = wx * b0 + wy * b1 + wz * b2
        out_deltas[i, 2] = wx * n0 + wy * n1 + wz * n2


@njit(parallel=True, fastmath=True, cache=True)
def _decode_local_deltas(
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
    deltas:        np.ndarray,
    out_points:    np.ndarray,
) -> None:
    """Reconstruct world positions by re-applying stored local deltas.

    Inverse of :func:`_encode_local_deltas`.  For each vertex ``i`` it
    rebuilds the local frame from ``smooth_points`` (which is the smoothed
    deformed mesh in the apply phase) and applies the stored local delta.

    Parameters
    ----------
    smooth_points : (N, 3) float64
        Smoothed deformed positions.
    neighbors : (N, K) int32
        Same neighbourhood matrix used in :func:`_encode_local_deltas`.
    deltas : (N, 3) float64
        Local-frame deltas produced by :func:`_encode_local_deltas`.
    out_points : (N, 3) float64
        Pre-allocated output, written in-place.
    """
    n_verts     = smooth_points.shape[0]
    n_neighbors = neighbors.shape[1]

    for i in prange(n_verts):
        nx            = 0.0
        ny            = 0.0
        nz            = 0.0
        tx            = 0.0
        ty            = 0.0
        tz            = 0.0
        found_tangent = False

        for j in range(n_neighbors):
            ni = neighbors[i, j]
            if ni < 0:
                continue
            dx  = smooth_points[ni, 0] - smooth_points[i, 0]
            dy  = smooth_points[ni, 1] - smooth_points[i, 1]
            dz  = smooth_points[ni, 2] - smooth_points[i, 2]
            mag = (dx * dx + dy * dy + dz * dz) ** 0.5
            if mag < 1e-12:
                continue

            inv = 1.0 / mag
            nx += dx * inv
            ny += dy * inv
            nz += dz * inv

            if not found_tangent:
                tx            = dx
                ty            = dy
                tz            = dz
                found_tangent = True

        dl0 = deltas[i, 0]
        dl1 = deltas[i, 1]
        dl2 = deltas[i, 2]

        if not found_tangent:
            out_points[i, 0] = smooth_points[i, 0] + dl0
            out_points[i, 1] = smooth_points[i, 1] + dl1
            out_points[i, 2] = smooth_points[i, 2] + dl2
            continue

        t0, t1, t2, b0, b1, b2, n0, n1, n2 = _orthonormal_frame(tx, ty, tz, nx, ny, nz)

        # column . local_delta = world delta
        wx = t0 * dl0 + b0 * dl1 + n0 * dl2
        wy = t1 * dl0 + b1 * dl1 + n1 * dl2
        wz = t2 * dl0 + b2 * dl1 + n2 * dl2

        out_points[i, 0] = smooth_points[i, 0] + wx
        out_points[i, 1] = smooth_points[i, 1] + wy
        out_points[i, 2] = smooth_points[i, 2] + wz


@njit(parallel=True, fastmath=True, cache=True)
def _blend_deltas(
    base_points:      np.ndarray,
    deltamush_points: np.ndarray,
    weight:           float,
    out_points:       np.ndarray,
) -> None:
    """Linear blend ``base + weight * (deltamush - base)`` per vertex.

    Parameters
    ----------
    base_points : (N, 3) float64
        Original (unfiltered) deformed positions.
    deltamush_points : (N, 3) float64
        Reconstructed delta-mushed positions.
    weight : float
        Mush weight in ``[0, 1]``.  ``0`` returns the smoothed mesh,
        ``1`` returns the fully reconstructed mesh.
    out_points : (N, 3) float64
        Pre-allocated output, written in-place.
    """
    n_verts = base_points.shape[0]
    for i in prange(n_verts):
        for d in range(3):
            out_points[i, d] = (
                base_points[i, d]
                + (deltamush_points[i, d] - base_points[i, d]) * weight
            )


# ======================== batch Procrustes (3x3) ========================= #


@njit(fastmath=True, cache=True)
def _polar_decompose_3x3(
    h00,
    h01,
    h02,
    h10,
    h11,
    h12,
    h20,
    h21,
    h22,
    max_iter,
):
    """Iterative polar decomposition of a 3x3 matrix.

    Higham's scaled-Newton iteration converges quadratically to the
    rotation factor of M = R * S where R is the closest rotation and
    S = sqrt(M^T * M).  Avoids per-call SVD dispatch which is the
    bottleneck of batch Procrustes for many small clusters.

    Higham's update A_{k+1} = 1/2(gamma A_k + gamma^-1 A_k^-T) requires the input
    to be invertible -- A^-T doesn't exist for rank-deficient matrices.
    When the rest pose is planar (every neighbour offset lives in a 2D
    subspace) the cross-covariance has all-zero z-row/column and is
    rank-deficient, which causes Higham's iteration to break out
    immediately and return the input matrix as the "rotation".  We
    detect this up-front and add a tiny eps*I perturbation that's
    negligible in magnitude but makes A invertible; the iteration then
    converges to the identity rotation, which is the correct closest
    rotation for any SPSD input.

    Returns the 9 entries of R (row-major).
    """
    # current iterate
    a00, a01, a02 = h00, h01, h02
    a10, a11, a12 = h10, h11, h12
    a20, a21, a22 = h20, h21, h22

    # rank-deficient guard: Higham's iteration needs an invertible
    # iterate.  For a planar SPSD covariance det(M) == 0 and the
    # iteration would otherwise break out after one step and return
    # M itself.  Trace-relative eps keeps the perturbation scale-
    # invariant and small enough not to bias the recovered rotation.
    det0 = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if det0 < 1e-18 and det0 > -1e-18:
        eps = 1e-9 * (a00 * a00 + a11 * a11 + a22 * a22 + 1e-30) ** 0.5
        a00 += eps
        a11 += eps
        a22 += eps

    for _ in range(max_iter):
        # cofactor matrix entries (transpose of inverse times det)
        c00 = a11 * a22 - a12 * a21
        c01 = a12 * a20 - a10 * a22
        c02 = a10 * a21 - a11 * a20
        c10 = a02 * a21 - a01 * a22
        c11 = a00 * a22 - a02 * a20
        c12 = a01 * a20 - a00 * a21
        c20 = a01 * a12 - a02 * a11
        c21 = a02 * a10 - a00 * a12
        c22 = a00 * a11 - a01 * a10

        det = a00 * c00 + a01 * c01 + a02 * c02
        if det == 0.0 or det != det:
            break

        inv_det = 1.0 / det

        # inverse-transpose entries
        it00 = c00 * inv_det
        it01 = c01 * inv_det
        it02 = c02 * inv_det
        it10 = c10 * inv_det
        it11 = c11 * inv_det
        it12 = c12 * inv_det
        it20 = c20 * inv_det
        it21 = c21 * inv_det
        it22 = c22 * inv_det

        # gamma scaling factor (Higham '86) for fast convergence
        norm_a = (
            a00 * a00
            + a01 * a01
            + a02 * a02
            + a10 * a10
            + a11 * a11
            + a12 * a12
            + a20 * a20
            + a21 * a21
            + a22 * a22
        )
        norm_it = (
            it00 * it00
            + it01 * it01
            + it02 * it02
            + it10 * it10
            + it11 * it11
            + it12 * it12
            + it20 * it20
            + it21 * it21
            + it22 * it22
        )
        if norm_a < 1e-30 or norm_it < 1e-30:
            break
        gamma     = (norm_it / norm_a) ** 0.25
        inv_gamma = 1.0 / gamma

        # next iterate = 0.5 * (gamma * A + (1/gamma) * inv_T(A))
        n00 = 0.5 * (gamma * a00 + inv_gamma * it00)
        n01 = 0.5 * (gamma * a01 + inv_gamma * it01)
        n02 = 0.5 * (gamma * a02 + inv_gamma * it02)
        n10 = 0.5 * (gamma * a10 + inv_gamma * it10)
        n11 = 0.5 * (gamma * a11 + inv_gamma * it11)
        n12 = 0.5 * (gamma * a12 + inv_gamma * it12)
        n20 = 0.5 * (gamma * a20 + inv_gamma * it20)
        n21 = 0.5 * (gamma * a21 + inv_gamma * it21)
        n22 = 0.5 * (gamma * a22 + inv_gamma * it22)

        # convergence test: max abs change
        diff = 0.0
        d    = abs(n00 - a00)
        if d > diff:
            diff = d
        d = abs(n01 - a01)
        if d > diff:
            diff = d
        d = abs(n02 - a02)
        if d > diff:
            diff = d
        d = abs(n10 - a10)
        if d > diff:
            diff = d
        d = abs(n11 - a11)
        if d > diff:
            diff = d
        d = abs(n12 - a12)
        if d > diff:
            diff = d
        d = abs(n20 - a20)
        if d > diff:
            diff = d
        d = abs(n21 - a21)
        if d > diff:
            diff = d
        d = abs(n22 - a22)
        if d > diff:
            diff = d

        a00, a01, a02 = n00, n01, n02
        a10, a11, a12 = n10, n11, n12
        a20, a21, a22 = n20, n21, n22

        if diff < 1e-9:
            break

    # ensure right-handed (det > 0); if not, flip the column with smallest
    # diagonal to negative to keep a proper rotation
    det_R = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if det_R < 0.0:
        # flip column 2 (matches the SVD reflection-fix convention used
        # by ProcrustesData.compute)
        a02 = -a02
        a12 = -a12
        a22 = -a22

    return a00, a01, a02, a10, a11, a12, a20, a21, a22


@njit(parallel=True, fastmath=True, cache=True)
def _batch_procrustes_rotations(
    H:        np.ndarray,
    out_R:    np.ndarray,
    max_iter: int,
) -> None:
    """Per-cluster polar-decomposition rotations from covariance matrices.

    Drop-in replacement for the SVD step in
    ``ProcrustesData.compute()``: given the batch of 3x3 covariance
    matrices ``H = vectors1.T @ vectors0`` the function fills ``out_R``
    with the rotation factors of each ``H``.

    Parameters
    ----------
    H : (B, 3, 3) float64
        Per-cluster covariance matrices.
    out_R : (B, 3, 3) float64
        Pre-allocated output rotations, written in-place.
    max_iter : int
        Newton iteration cap (typically 12-20 is plenty).
    """
    B = H.shape[0]
    for b in prange(B):
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = _polar_decompose_3x3(
            H[b, 0, 0],
            H[b, 0, 1],
            H[b, 0, 2],
            H[b, 1, 0],
            H[b, 1, 1],
            H[b, 1, 2],
            H[b, 2, 0],
            H[b, 2, 1],
            H[b, 2, 2],
            max_iter,
        )
        out_R[b, 0, 0] = r00
        out_R[b, 0, 1] = r01
        out_R[b, 0, 2] = r02
        out_R[b, 1, 0] = r10
        out_R[b, 1, 1] = r11
        out_R[b, 1, 2] = r12
        out_R[b, 2, 0] = r20
        out_R[b, 2, 1] = r21
        out_R[b, 2, 2] = r22


# ------------------- numpy fallbacks for the numba kernels ----------------- #


def _delta_mush_frames_numpy(
    smooth_points: np.ndarray, neighbors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-vertex (T, B, N) frames from smoothed points + neighbours.

    Used as a numpy fallback when the numba kernel is unavailable.
    Returns ``(frames, valid)`` where ``frames`` is ``(N, 3, 3)`` with
    rows ``[T; B; N]`` and ``valid`` is the bool mask of vertices that
    had at least one usable neighbour.
    """
    diffs = smooth_points[neighbors] - smooth_points[:, None, :]
    mask  = neighbors >= 0
    diffs[~mask] = 0.0

    mags    = np.linalg.norm(diffs, axis=2)
    safe    = mags > 1e-12
    inv     = np.where(safe, 1.0 / np.where(safe, mags, 1.0), 0.0)
    unit    = diffs * inv[..., None]

    normals = unit.sum(axis=1)

    # tangent hint: first valid (non-zero magnitude) neighbour offset per row
    valid_per_row = safe.any(axis=1)
    first_valid   = np.argmax(safe, axis=1)
    rows          = np.arange(neighbors.shape[0])
    tangents      = diffs[rows, first_valid]

    # default frame for isolated vertices
    out_t = np.zeros_like(smooth_points)
    out_n = np.zeros_like(smooth_points)
    out_n[:, 1] = 1.0  # canonical Y-up normal
    out_t[:, 0] = 1.0  # canonical X tangent

    nm     = np.linalg.norm(normals, axis=1)
    nz     = nm > 1e-12
    n_unit = np.zeros_like(normals)
    n_unit[nz] = normals[nz] / nm[nz, None]

    # only fill rows that had at least one valid neighbour
    rows_ok = valid_per_row & nz
    n_use   = np.where(rows_ok[:, None], n_unit, out_n)

    # project tangent onto plane perpendicular to the normal
    dot    = np.einsum("ij,ij->i", tangents, n_use)
    t_proj = tangents - dot[:, None] * n_use
    tm     = np.linalg.norm(t_proj, axis=1)
    tz     = tm > 1e-12
    t_unit = np.zeros_like(t_proj)
    t_unit[tz] = t_proj[tz] / tm[tz, None]

    # robust fallback tangent for degenerate cases
    fallback = np.where(
        np.abs(n_use[:, 0:1]) < 0.9,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    fall         = fallback - np.einsum("ij,ij->i", fallback, n_use)[:, None] * n_use
    fm           = np.linalg.norm(fall, axis=1)
    fall         = fall / np.where(fm > 1e-12, fm, 1.0)[:, None]

    use_fallback = ~rows_ok | ~tz
    t_use        = np.where(use_fallback[:, None], fall, t_unit)
    b_use        = np.cross(n_use, t_use)

    frames       = np.stack([t_use, b_use, n_use], axis=1)
    return frames, rows_ok


def _delta_mush_encode_numpy(
    rest_points:   np.ndarray,
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for ``_encode_local_deltas``."""
    frames, valid = _delta_mush_frames_numpy(smooth_points, neighbors)
    world = rest_points - smooth_points
    local = np.einsum("nij,nj->ni", frames, world)
    # isolated vertices get the world delta as-is
    local[~valid] = world[~valid]
    return local


def _delta_mush_decode_numpy(
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
    deltas:        np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for ``_decode_local_deltas``."""
    frames, valid = _delta_mush_frames_numpy(smooth_points, neighbors)
    world = np.einsum("nji,nj->ni", frames, deltas)
    out   = smooth_points + world
    out[~valid] = smooth_points[~valid] + deltas[~valid]
    return out


# ======================= TBN (first-face) kernels ========================= #
#
# This family builds the per-vertex tangent frame from the *first face* that
# contains the vertex, using a deterministic ``cross(next_offset, prev_offset)``
# normal.  This deterministic choice matches Maya's ``deltaMush`` output much
# more closely than the averaged-Laplacian normal used by the original
# :func:`_encode_local_deltas` family.
#
# ``first_nbrs`` is a precomputed ``(N, 2)`` int32 array storing the indices of
# the *next* (column 0) and *previous* (column 1) vertex in the vertex's first
# face.  ``-1`` means the vertex has no usable face (e.g. isolated point) and
# the world-space delta is stored / re-applied as identity.


@njit(fastmath=True, cache=True)
def _tbn_frame(t0, t1, t2, p0, p1, p2):
    """Build (T, B, N) from two edge offsets ``next`` and ``prev``.

    Returns 9 floats packed as (T, B, N) columns.  Falls back to
    canonical-axis frames on degenerate inputs (collinear or zero-length
    edges).
    """
    # N = next x prev (right-handed face normal of a triangle fan)
    n0 = t1 * p2 - t2 * p1
    n1 = t2 * p0 - t0 * p2
    n2 = t0 * p1 - t1 * p0

    nm = (n0 * n0 + n1 * n1 + n2 * n2) ** 0.5
    if nm < 1e-12:
        # degenerate (collinear) -- fall back to canonical Y-up
        n0, n1, n2 = 0.0, 1.0, 0.0
        nm = 1.0
    n0 /= nm
    n1 /= nm
    n2 /= nm

    # T from next-edge offset, projected onto plane perpendicular to N
    dot = t0 * n0 + t1 * n1 + t2 * n2
    t0 -= dot * n0
    t1 -= dot * n1
    t2 -= dot * n2

    tm = (t0 * t0 + t1 * t1 + t2 * t2) ** 0.5
    if tm < 1e-12:
        if abs(n0) < 0.9:
            t0, t1, t2 = 1.0 - n0 * n0, -n0 * n1, -n0 * n2
        else:
            t0, t1, t2 = -n1 * n0, 1.0 - n1 * n1, -n1 * n2
        tm = (t0 * t0 + t1 * t1 + t2 * t2) ** 0.5
        if tm < 1e-12:
            tm = 1.0
    t0 /= tm
    t1 /= tm
    t2 /= tm

    # B = N x T
    b0 = n1 * t2 - n2 * t1
    b1 = n2 * t0 - n0 * t2
    b2 = n0 * t1 - n1 * t0
    return t0, t1, t2, b0, b1, b2, n0, n1, n2


@njit(parallel=True, fastmath=True, cache=True)
def _encode_local_deltas_tbn(
    rest_points:   np.ndarray,
    smooth_points: np.ndarray,
    first_nbrs:    np.ndarray,
    out_deltas:    np.ndarray,
) -> None:
    """Encode ``rest - smooth`` in each vertex's first-face TBN frame.

    Parameters
    ----------
    rest_points : (N, 3) float64
    smooth_points : (N, 3) float64
        Smoothed rest positions.
    first_nbrs : (N, 2) int32
        ``[next_vertex, prev_vertex]`` from each vertex's first face.
        ``-1`` marks "no usable face" (isolated vertex).
    out_deltas : (N, 3) float64
        Per-vertex local-frame deltas, written in-place.
    """
    n_verts = rest_points.shape[0]
    for i in prange(n_verts):
        n_idx = first_nbrs[i, 0]
        p_idx = first_nbrs[i, 1]

        wx    = rest_points[i, 0] - smooth_points[i, 0]
        wy    = rest_points[i, 1] - smooth_points[i, 1]
        wz    = rest_points[i, 2] - smooth_points[i, 2]

        if n_idx < 0 or p_idx < 0:
            # no face -- store identity delta
            out_deltas[i, 0] = wx
            out_deltas[i, 1] = wy
            out_deltas[i, 2] = wz
            continue

        # next, prev offsets in the smoothed mesh
        tx = smooth_points[n_idx, 0] - smooth_points[i, 0]
        ty = smooth_points[n_idx, 1] - smooth_points[i, 1]
        tz = smooth_points[n_idx, 2] - smooth_points[i, 2]
        px = smooth_points[p_idx, 0] - smooth_points[i, 0]
        py = smooth_points[p_idx, 1] - smooth_points[i, 1]
        pz = smooth_points[p_idx, 2] - smooth_points[i, 2]

        t0, t1, t2, b0, b1, b2, n0, n1, n2 = _tbn_frame(tx, ty, tz, px, py, pz)

        out_deltas[i, 0] = wx * t0 + wy * t1 + wz * t2
        out_deltas[i, 1] = wx * b0 + wy * b1 + wz * b2
        out_deltas[i, 2] = wx * n0 + wy * n1 + wz * n2


@njit(parallel=True, fastmath=True, cache=True)
def _decode_local_deltas_tbn(
    smooth_points: np.ndarray,
    first_nbrs:    np.ndarray,
    deltas:        np.ndarray,
    out_points:    np.ndarray,
) -> None:
    """Reconstruct world positions from ``deltas`` using TBN frames.

    Inverse of :func:`_encode_local_deltas_tbn`.
    """
    n_verts = smooth_points.shape[0]
    for i in prange(n_verts):
        n_idx = first_nbrs[i, 0]
        p_idx = first_nbrs[i, 1]

        dl0   = deltas[i, 0]
        dl1   = deltas[i, 1]
        dl2   = deltas[i, 2]

        if n_idx < 0 or p_idx < 0:
            out_points[i, 0] = smooth_points[i, 0] + dl0
            out_points[i, 1] = smooth_points[i, 1] + dl1
            out_points[i, 2] = smooth_points[i, 2] + dl2
            continue

        tx = smooth_points[n_idx, 0] - smooth_points[i, 0]
        ty = smooth_points[n_idx, 1] - smooth_points[i, 1]
        tz = smooth_points[n_idx, 2] - smooth_points[i, 2]
        px = smooth_points[p_idx, 0] - smooth_points[i, 0]
        py = smooth_points[p_idx, 1] - smooth_points[i, 1]
        pz = smooth_points[p_idx, 2] - smooth_points[i, 2]

        t0, t1, t2, b0, b1, b2, n0, n1, n2 = _tbn_frame(tx, ty, tz, px, py, pz)

        wx = t0 * dl0 + b0 * dl1 + n0 * dl2
        wy = t1 * dl0 + b1 * dl1 + n1 * dl2
        wz = t2 * dl0 + b2 * dl1 + n2 * dl2

        out_points[i, 0] = smooth_points[i, 0] + wx
        out_points[i, 1] = smooth_points[i, 1] + wy
        out_points[i, 2] = smooth_points[i, 2] + wz


# ======================= Procrustes (SVD) kernels ========================= #
#
# This family computes the *optimal* per-vertex rigid rotation that maps the
# smoothed-rest neighbour offsets onto the smoothed-deformed neighbour offsets
# via the polar decomposition of the cross-covariance ``M = A^T B``.  The
# rotation is then applied to the cached world-space rest delta to produce the
# output position.
#
# Numba does not provide ``np.linalg.svd``, so we reuse the
# :func:`_polar_decompose_3x3` Higham iteration kernel that already exists in
# this module for batch Procrustes.
#
# Encoding for this family is trivial -- the local delta *is* the world-space
# delta -- so there is no dedicated encode kernel.  We still expose a
# helper that computes it for API symmetry.


@njit(parallel=True, fastmath=True, cache=True)
def _decode_world_deltas_procrustes(
    smooth_rest:    np.ndarray,
    smooth_def:     np.ndarray,
    neighbors:      np.ndarray,
    world_deltas:   np.ndarray,
    out_points:     np.ndarray,
    polar_max_iter: int        = 16,
) -> None:
    """Apply per-vertex Procrustes rotation to ``world_deltas``.

    For each vertex ``i`` builds the cross-covariance matrix
    ``M = A^T B`` from the smoothed-rest neighbour offsets ``A`` and
    smoothed-deformed neighbour offsets ``B``, polar-decomposes it to
    extract the optimal rotation ``R``, and writes
    ``out[i] = smooth_def[i] + R^T @ world_deltas[i]``.

    Parameters
    ----------
    smooth_rest : (N, 3) float64
    smooth_def : (N, 3) float64
    neighbors : (N, K) int32
        Per-vertex neighbour matrix; ``-1`` marks padding slots.
    world_deltas : (N, 3) float64
        Cached ``rest - smooth_rest`` offsets.
    out_points : (N, 3) float64
        Pre-allocated output, written in-place.
    polar_max_iter : int
        Newton iteration cap for the polar decomposition (typically 12-20).
    """
    n_verts     = smooth_rest.shape[0]
    n_neighbors = neighbors.shape[1]

    for i in prange(n_verts):
        # accumulate covariance M = A^T B  (3x3, row-major)
        m00         = 0.0
        m01         = 0.0
        m02         = 0.0
        m10         = 0.0
        m11         = 0.0
        m12         = 0.0
        m20         = 0.0
        m21         = 0.0
        m22         = 0.0
        valid_count = 0

        srx         = smooth_rest[i, 0]
        sry         = smooth_rest[i, 1]
        srz         = smooth_rest[i, 2]
        sdx         = smooth_def[i, 0]
        sdy         = smooth_def[i, 1]
        sdz         = smooth_def[i, 2]

        for j in range(n_neighbors):
            ni = neighbors[i, j]
            if ni < 0:
                continue
            ax = smooth_rest[ni, 0] - srx
            ay = smooth_rest[ni, 1] - sry
            az = smooth_rest[ni, 2] - srz
            bx = smooth_def[ni, 0] - sdx
            by = smooth_def[ni, 1] - sdy
            bz = smooth_def[ni, 2] - sdz
            m00 += ax * bx
            m01 += ax * by
            m02 += ax * bz
            m10 += ay * bx
            m11 += ay * by
            m12 += ay * bz
            m20 += az * bx
            m21 += az * by
            m22 += az * bz
            valid_count += 1

        wx = world_deltas[i, 0]
        wy = world_deltas[i, 1]
        wz = world_deltas[i, 2]

        if valid_count < 2:
            # not enough neighbours -- pass-through identity rotation
            out_points[i, 0] = sdx + wx
            out_points[i, 1] = sdy + wy
            out_points[i, 2] = sdz + wz
            continue

        # polar decomposition: M = R * S  =>  R is the rotation factor
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = _polar_decompose_3x3(
            m00, m01, m02, m10, m11, m12, m20, m21, m22, polar_max_iter
        )

        # Apply R^T to the world delta.  R^T has rows = R's columns:
        # R^T @ w = (R^T)_row0 * w, ...
        ox = r00 * wx + r10 * wy + r20 * wz
        oy = r01 * wx + r11 * wy + r21 * wz
        oz = r02 * wx + r12 * wy + r22 * wz

        out_points[i, 0] = sdx + ox
        out_points[i, 1] = sdy + oy
        out_points[i, 2] = sdz + oz


# =================== Direct Delta Mush (DDM) kernels ====================== #
#
# DDM in this implementation is a precomputed-cross-covariance variant of
# Procrustes: the smoothed-rest neighbour offsets ``A`` are cached during
# ``bind`` (alongside per-neighbour weights derived from rest-edge length --
# see :func:`_ddm_precompute`).  At runtime the kernel fetches ``A`` from the
# cache and only iterates over ``B`` (the smoothed-deformed neighbour
# offsets), which both saves memory traffic and lets us use *length-weighted*
# neighbours so that closer ring vertices dominate the local rotation
# estimate -- the same trick used by the Le & Lewis 2019 Direct Delta Mush
# paper to stabilise the rotation against irregular topology.


@njit(parallel=True, fastmath=True, cache=True)
def _ddm_precompute(
    smooth_points: np.ndarray,
    neighbors:     np.ndarray,
    out_offsets:   np.ndarray,
    out_weights:   np.ndarray,
) -> None:
    """Precompute weighted smoothed-rest neighbour offsets for DDM apply.

    Parameters
    ----------
    smooth_points : (N, 3) float64
    neighbors : (N, K) int32, ``-1`` padded
    out_offsets : (N, K, 3) float64
        Pre-allocated output, written in-place.  Padded slots are zero.
    out_weights : (N, K) float64
        Per-neighbour weight ``1 / |smooth_points[ni] - smooth_points[i]|``,
        zero for padded slots and degenerate edges.
    """
    n_verts     = smooth_points.shape[0]
    n_neighbors = neighbors.shape[1]

    for i in prange(n_verts):
        sx = smooth_points[i, 0]
        sy = smooth_points[i, 1]
        sz = smooth_points[i, 2]
        for j in range(n_neighbors):
            ni = neighbors[i, j]
            if ni < 0:
                out_offsets[i, j, 0] = 0.0
                out_offsets[i, j, 1] = 0.0
                out_offsets[i, j, 2] = 0.0
                out_weights[i, j] = 0.0
                continue
            dx  = smooth_points[ni, 0] - sx
            dy  = smooth_points[ni, 1] - sy
            dz  = smooth_points[ni, 2] - sz
            mag = (dx * dx + dy * dy + dz * dz) ** 0.5
            out_offsets[i, j, 0] = dx
            out_offsets[i, j, 1] = dy
            out_offsets[i, j, 2] = dz
            if mag < 1e-12:
                out_weights[i, j] = 0.0
            else:
                out_weights[i, j] = 1.0 / mag


@njit(parallel=True, fastmath=True, cache=True)
def _decode_world_deltas_ddm(
    smooth_def:     np.ndarray,
    neighbors:      np.ndarray,
    rest_offsets:   np.ndarray,
    rest_weights:   np.ndarray,
    world_deltas:   np.ndarray,
    out_points:     np.ndarray,
    polar_max_iter: int        = 16,
) -> None:
    """Apply Direct-Delta-Mush per-vertex polar rotation to ``world_deltas``.

    Like :func:`_decode_world_deltas_procrustes`, but uses *precomputed*
    smoothed-rest neighbour offsets and *length-based* neighbour weights to
    stabilise the local-rotation estimate.

    Parameters
    ----------
    smooth_def : (N, 3) float64
    neighbors : (N, K) int32
    rest_offsets : (N, K, 3) float64
        Cached smoothed-rest neighbour offsets (output of
        :func:`_ddm_precompute`).
    rest_weights : (N, K) float64
        Per-neighbour weights (1 / rest edge length).
    world_deltas : (N, 3) float64
        Cached ``rest - smooth_rest`` offsets.
    out_points : (N, 3) float64
        Pre-allocated output, written in-place.
    polar_max_iter : int
    """
    n_verts     = smooth_def.shape[0]
    n_neighbors = neighbors.shape[1]

    for i in prange(n_verts):
        m00         = 0.0
        m01         = 0.0
        m02         = 0.0
        m10         = 0.0
        m11         = 0.0
        m12         = 0.0
        m20         = 0.0
        m21         = 0.0
        m22         = 0.0
        valid_count = 0

        sdx         = smooth_def[i, 0]
        sdy         = smooth_def[i, 1]
        sdz         = smooth_def[i, 2]

        for j in range(n_neighbors):
            ni = neighbors[i, j]
            if ni < 0:
                continue
            w = rest_weights[i, j]
            if w == 0.0:
                continue
            ax = rest_offsets[i, j, 0]
            ay = rest_offsets[i, j, 1]
            az = rest_offsets[i, j, 2]
            bx = smooth_def[ni, 0] - sdx
            by = smooth_def[ni, 1] - sdy
            bz = smooth_def[ni, 2] - sdz
            # weighted outer product (sqrt-weight on each side preserves
            # the geometric meaning of the cross-covariance)
            sw = w
            m00 += sw * ax * bx
            m01 += sw * ax * by
            m02 += sw * ax * bz
            m10 += sw * ay * bx
            m11 += sw * ay * by
            m12 += sw * ay * bz
            m20 += sw * az * bx
            m21 += sw * az * by
            m22 += sw * az * bz
            valid_count += 1

        wx = world_deltas[i, 0]
        wy = world_deltas[i, 1]
        wz = world_deltas[i, 2]

        if valid_count < 2:
            out_points[i, 0] = sdx + wx
            out_points[i, 1] = sdy + wy
            out_points[i, 2] = sdz + wz
            continue

        r00, r01, r02, r10, r11, r12, r20, r21, r22 = _polar_decompose_3x3(
            m00, m01, m02, m10, m11, m12, m20, m21, m22, polar_max_iter
        )

        ox = r00 * wx + r10 * wy + r20 * wz
        oy = r01 * wx + r11 * wy + r21 * wz
        oz = r02 * wx + r12 * wy + r22 * wz

        out_points[i, 0] = sdx + ox
        out_points[i, 1] = sdy + oy
        out_points[i, 2] = sdz + oz


# ------------------- numpy fallbacks for the new kernels ------------------- #


def _delta_mush_encode_tbn_numpy(
    rest_points:   np.ndarray,
    smooth_points: np.ndarray,
    first_nbrs:    np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for :func:`_encode_local_deltas_tbn`."""
    out   = (rest_points - smooth_points).copy()
    n_idx = first_nbrs[:, 0]
    p_idx = first_nbrs[:, 1]
    valid = (n_idx >= 0) & (p_idx >= 0)
    if not valid.any():
        return out

    rows     = np.where(valid)[0]
    next_off = smooth_points[n_idx[valid]] - smooth_points[rows]
    prev_off = smooth_points[p_idx[valid]] - smooth_points[rows]
    normals  = np.cross(next_off, prev_off)
    nm       = np.linalg.norm(normals, axis=1)
    safe_n   = nm > 1e-12
    n_unit   = np.zeros_like(normals)
    n_unit[safe_n] = normals[safe_n] / nm[safe_n, None]

    # tangent: project next_off onto plane perp to n_unit
    dot    = np.einsum("ij,ij->i", next_off, n_unit)
    t      = next_off - dot[:, None] * n_unit
    tm     = np.linalg.norm(t, axis=1)
    safe_t = tm > 1e-12
    t_unit = np.zeros_like(t)
    t_unit[safe_t] = t[safe_t] / tm[safe_t, None]

    # robust fallback for degenerate t -- perp to n_unit
    if (~safe_t).any():
        bad = ~safe_t
        # pick e_x or e_y as far from n_unit as possible
        canon = np.where(
            np.abs(n_unit[bad, 0:1]) < 0.9,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        proj   = canon - np.einsum("ij,ij->i", canon, n_unit[bad])[:, None] * n_unit[bad]
        proj_n = np.linalg.norm(proj, axis=1)
        proj   = proj / np.where(proj_n > 1e-12, proj_n, 1.0)[:, None]
        t_unit[bad] = proj

    b_unit = np.cross(n_unit, t_unit)

    world  = out[rows]
    out[rows] = np.column_stack(
        [
            np.einsum("ij,ij->i", world, t_unit),
            np.einsum("ij,ij->i", world, b_unit),
            np.einsum("ij,ij->i", world, n_unit),
        ]
    )
    return out


def _delta_mush_decode_tbn_numpy(
    smooth_points: np.ndarray,
    first_nbrs:    np.ndarray,
    deltas:        np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for :func:`_decode_local_deltas_tbn`."""
    out   = smooth_points + deltas  # default = identity for invalid rows
    n_idx = first_nbrs[:, 0]
    p_idx = first_nbrs[:, 1]
    valid = (n_idx >= 0) & (p_idx >= 0)
    if not valid.any():
        return out

    rows     = np.where(valid)[0]
    next_off = smooth_points[n_idx[valid]] - smooth_points[rows]
    prev_off = smooth_points[p_idx[valid]] - smooth_points[rows]
    normals  = np.cross(next_off, prev_off)
    nm       = np.linalg.norm(normals, axis=1)
    safe_n   = nm > 1e-12
    n_unit   = np.zeros_like(normals)
    n_unit[safe_n] = normals[safe_n] / nm[safe_n, None]

    dot    = np.einsum("ij,ij->i", next_off, n_unit)
    t      = next_off - dot[:, None] * n_unit
    tm     = np.linalg.norm(t, axis=1)
    safe_t = tm > 1e-12
    t_unit = np.zeros_like(t)
    t_unit[safe_t] = t[safe_t] / tm[safe_t, None]

    if (~safe_t).any():
        bad = ~safe_t
        canon = np.where(
            np.abs(n_unit[bad, 0:1]) < 0.9,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        proj   = canon - np.einsum("ij,ij->i", canon, n_unit[bad])[:, None] * n_unit[bad]
        proj_n = np.linalg.norm(proj, axis=1)
        proj   = proj / np.where(proj_n > 1e-12, proj_n, 1.0)[:, None]
        t_unit[bad] = proj

    b_unit = np.cross(n_unit, t_unit)

    d      = deltas[rows]
    world  = d[:, 0:1] * t_unit + d[:, 1:2] * b_unit + d[:, 2:3] * n_unit
    out[rows] = smooth_points[rows] + world
    return out


def _delta_mush_decode_procrustes_numpy(
    smooth_rest:  np.ndarray,
    smooth_def:   np.ndarray,
    neighbors:    np.ndarray,
    world_deltas: np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for :func:`_decode_world_deltas_procrustes`."""
    n_verts = smooth_rest.shape[0]
    out     = smooth_def + world_deltas
    for i in range(n_verts):
        ni = neighbors[i]
        ni = ni[ni >= 0]
        if ni.size < 2:
            continue
        a = smooth_rest[ni] - smooth_rest[i]
        b = smooth_def[ni] - smooth_def[i]
        m = a.T @ b
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt = vt.copy()
            vt[-1] *= -1
            r = u @ vt
        out[i] = smooth_def[i] + r.T @ world_deltas[i]
    return out


def _delta_mush_decode_ddm_numpy(
    smooth_def:   np.ndarray,
    neighbors:    np.ndarray,
    rest_offsets: np.ndarray,
    rest_weights: np.ndarray,
    world_deltas: np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for :func:`_decode_world_deltas_ddm`."""
    n_verts = smooth_def.shape[0]
    out     = smooth_def + world_deltas
    for i in range(n_verts):
        ni   = neighbors[i]
        mask = ni >= 0
        ni   = ni[mask]
        if ni.size < 2:
            continue
        w = rest_weights[i, mask]
        a = rest_offsets[i, mask]
        b = smooth_def[ni] - smooth_def[i]
        m = (a * w[:, None]).T @ b
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt = vt.copy()
            vt[-1] *= -1
            r = u @ vt
        out[i] = smooth_def[i] + r.T @ world_deltas[i]
    return out