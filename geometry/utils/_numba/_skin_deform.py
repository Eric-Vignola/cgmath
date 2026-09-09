"""
Numba-optimized kernels for skin deformation.

Row-vector convention throughout, matching the rest of the package: a point
transforms as ``p' = p @ M`` with the translation in ``M[3, :3]`` -- see
:func:`transforms._numba._matrix._matrix_point_multiply`.  A skin
matrix is therefore ``inverse_bind @ world_posed``, in that order.

Weights arrive compact -- ``(N, K)`` columns into the joint list plus the
matching ``(N, K)`` weights -- so an animation loop never densifies to
``(N, J)``.
"""

import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaError


# --------------------------------------------------------------------------- #
#                          linear blend skinning                              #
# --------------------------------------------------------------------------- #


@njit(parallel=True, fastmath=True, cache=True)
def _lbs_compact(
    points:            np.ndarray,
    influence_indices: np.ndarray,
    weights:           np.ndarray,
    matrices:          np.ndarray,
    out:               np.ndarray,
) -> None:
    """Linear blend skinning over compact per-vertex influences.

    Args:
        points: ``(N, 3)`` bind-pose positions.
        influence_indices: ``(N, K)`` int32 columns into *matrices*.
        weights: ``(N, K)`` blend weights.
        matrices: ``(J, 4, 4)`` skin matrices, row-vector.
        out: ``(N, 3)`` destination, written in place.
    """
    n = points.shape[0]
    k = influence_indices.shape[1]

    for i in prange(n):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]

        ax = 0.0
        ay = 0.0
        az = 0.0

        for c in range(k):
            w = weights[i, c]
            if w == 0.0:
                continue

            m = matrices[influence_indices[i, c]]

            ax += w * (m[0, 0] * px + m[1, 0] * py + m[2, 0] * pz + m[3, 0])
            ay += w * (m[0, 1] * px + m[1, 1] * py + m[2, 1] * pz + m[3, 1])
            az += w * (m[0, 2] * px + m[1, 2] * py + m[2, 2] * pz + m[3, 2])

        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az


def _lbs_compact_numpy(
    points:            np.ndarray,
    influence_indices: np.ndarray,
    weights:           np.ndarray,
    matrices:          np.ndarray,
) -> np.ndarray:
    """Pure-numpy fallback for :func:`_lbs_compact`.

    Safety net for a numba typing failure, not a supported standalone path --
    this module imports numba unconditionally.  Materializes an
    ``(N, K, 4, 4)`` gather, so it is far more memory-hungry than the kernel
    (1M verts at K=8 measured ~1.3 GiB transient against ~21 MiB).
    """
    gathered = matrices[influence_indices]
    posed    = np.einsum("nd,nkdc->nkc", points, gathered[:, :, :3, :3])
    posed    = posed + gathered[:, :, 3, :3]

    # the kernel skips w == 0 entirely, so a NaN or Inf sitting behind a zero
    # weight contributes nothing there. Multiplying through would give NaN.
    posed = np.where(weights[..., None] == 0.0, 0.0, posed)

    return np.einsum("nk,nkc->nc", weights, posed)


def lbs_compact_fast(
    points:            np.ndarray,
    influence_indices: np.ndarray,
    weights:           np.ndarray,
    matrices:          np.ndarray,
) -> np.ndarray:
    """Linear blend skinning, numba-accelerated with a numpy fallback.

    Args:
        points: ``(N, 3)`` bind-pose positions.
        influence_indices: ``(N, K)`` columns into *matrices*.
        weights: ``(N, K)`` blend weights.
        matrices: ``(J, 4, 4)`` skin matrices, ``inverse_bind @ world_posed``.

    Returns:
        ``(N, 3)`` deformed positions.  The input is not modified.
    """
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    influence_indices = np.ascontiguousarray(
        np.asarray(influence_indices, dtype=np.int32)
    )
    weights  = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    matrices = np.ascontiguousarray(np.asarray(matrices, dtype=np.float64))

    out      = np.empty_like(points)
    try:
        _lbs_compact(points, influence_indices, weights, matrices, out)
    except NumbaError:
        # only a compilation/typing failure falls back. A bare `except` here
        # swallowed caller mistakes and re-surfaced them as a confusing
        # IndexError from deep inside the numpy path.
        return _lbs_compact_numpy(points, influence_indices, weights, matrices)
    return out


# --------------------------------------------------------------------------- #
#                       dual quaternion skinning                              #
# --------------------------------------------------------------------------- #
#
# A dual quaternion cannot represent scale, so a skin matrix is split first:
# the rigid part becomes the dual quaternion, and whatever is left over --
# scale, shear, reflection -- stays a 3x3 and is blended linearly.  Feeding a
# scaled matrix straight in silently drops the scale instead of failing.
#
# Quaternions are ``(x, y, z, w)`` with the scalar last, matching
# :mod:`transforms._numba._quaternion`, and a point rotates as
# ``q p conj(q)`` -- verified against ``_matrix_point_multiply``, which is the
# oracle for the row-vector convention.

_POLAR_ITERATIONS = 32
_POLAR_TOLERANCE  = 1e-12
_DEGENERATE       = 1e-12


@njit(fastmath=True, cache=True)
def _determinant(m) -> float:
    """determinant of the upper-left 3x3 of *m*"""
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


@njit(fastmath=True, cache=True)
def _polar_step(rot, nxt) -> float:
    """One Higham step, ``R <- (R + R^-T) / 2``, in place.

    Returns how far the matrix moved, or ``-1`` when it is singular and there
    is no rotation to recover.
    """
    det = _determinant(rot)
    if abs(det) < _DEGENERATE:
        return -1.0

    # inverse transpose, written out: nxt = R^-T
    nxt[0, 0] = (rot[1, 1] * rot[2, 2] - rot[1, 2] * rot[2, 1]) / det
    nxt[1, 0] = (rot[0, 2] * rot[2, 1] - rot[0, 1] * rot[2, 2]) / det
    nxt[2, 0] = (rot[0, 1] * rot[1, 2] - rot[0, 2] * rot[1, 1]) / det
    nxt[0, 1] = (rot[1, 2] * rot[2, 0] - rot[1, 0] * rot[2, 2]) / det
    nxt[1, 1] = (rot[0, 0] * rot[2, 2] - rot[0, 2] * rot[2, 0]) / det
    nxt[2, 1] = (rot[0, 2] * rot[1, 0] - rot[0, 0] * rot[1, 2]) / det
    nxt[0, 2] = (rot[1, 0] * rot[2, 1] - rot[1, 1] * rot[2, 0]) / det
    nxt[1, 2] = (rot[0, 1] * rot[2, 0] - rot[0, 0] * rot[2, 1]) / det
    nxt[2, 2] = (rot[0, 0] * rot[1, 1] - rot[0, 1] * rot[1, 0]) / det

    shift = 0.0
    for r in range(3):
        for c in range(3):
            half = 0.5 * (rot[r, c] + nxt[r, c])
            shift += abs(half - rot[r, c])
            rot[r, c] = half

    return shift


@njit(fastmath=True, cache=True)
def _polar_rotation(source, index, rot, nxt) -> None:
    """Write the orthonormal factor closest to ``source[index, :3, :3]``.

    A mirrored joint converges to a reflection, whose quaternion is
    meaningless, so the sign is flipped back out and left for the stretch to
    carry.
    """
    for r in range(3):
        for c in range(3):
            rot[r, c] = source[index, r, c]

    for _ in range(_POLAR_ITERATIONS):
        shift = _polar_step(rot, nxt)

        if shift < 0.0:
            # singular: no rotation to recover, leave the whole thing in S
            for r in range(3):
                for c in range(3):
                    rot[r, c] = 1.0 if r == c else 0.0
            return

        if shift < _POLAR_TOLERANCE:
            break

    if _determinant(rot) < 0.0:
        for r in range(3):
            for c in range(3):
                rot[r, c] = -rot[r, c]


@njit(fastmath=True, cache=True)
def _rotation_quaternion(rot):
    """Unit ``(x, y, z, w)`` for an orthonormal row-vector 3x3.

    Same trace branches as
    :func:`transforms._numba._matrix._matrix_to_quaternion`.
    """
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0.0:
        s  = 0.5 / (trace + 1.0) ** 0.5
        qx = (rot[1, 2] - rot[2, 1]) * s
        qy = (rot[2, 0] - rot[0, 2]) * s
        qz = (rot[0, 1] - rot[1, 0]) * s
        qw = 0.25 / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s  = 2.0 * (1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) ** 0.5
        qx = 0.25 * s
        qy = (rot[1, 0] + rot[0, 1]) / s
        qz = (rot[2, 0] + rot[0, 2]) / s
        qw = (rot[1, 2] - rot[2, 1]) / s
    elif rot[1, 1] > rot[2, 2]:
        s  = 2.0 * (1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) ** 0.5
        qx = (rot[1, 0] + rot[0, 1]) / s
        qy = 0.25 * s
        qz = (rot[2, 1] + rot[1, 2]) / s
        qw = (rot[2, 0] - rot[0, 2]) / s
    else:
        s  = 2.0 * (1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) ** 0.5
        qx = (rot[2, 0] + rot[0, 2]) / s
        qy = (rot[2, 1] + rot[1, 2]) / s
        qz = 0.25 * s
        qw = (rot[0, 1] - rot[1, 0]) / s

    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm > 0.0:
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

    return qx, qy, qz, qw


@njit(fastmath=True, cache=True)
def _dqs_decompose(matrices, quats, duals, stretch) -> None:
    """Split skin matrices into rigid quaternions and a leftover 3x3.

    Runs over joints rather than vertices -- there are a few hundred of the
    former and millions of the latter -- so it is deliberately not parallel.

    Args:
        matrices: ``(J, 4, 4)`` skin matrices, row-vector.
        quats: ``(J, 4)`` destination, unit rotation quaternions ``(x,y,z,w)``.
        duals: ``(J, 4)`` destination, dual parts carrying the translation.
        stretch: ``(J, 3, 3)`` destination, the non-rigid remainder ``S`` in
            ``A = S @ R``.
    """
    count = matrices.shape[0]

    rot   = np.empty((3, 3), dtype=np.float64)
    nxt   = np.empty((3, 3), dtype=np.float64)

    for j in range(count):
        _polar_rotation(matrices, j, rot, nxt)

        # S = A @ R^T, exact because R is orthonormal
        for r in range(3):
            for c in range(3):
                total = 0.0
                for k in range(3):
                    total += matrices[j, r, k] * rot[c, k]
                stretch[j, r, c] = total

        qx, qy, qz, qw = _rotation_quaternion(rot)

        quats[j, 0] = qx
        quats[j, 1] = qy
        quats[j, 2] = qz
        quats[j, 3] = qw

        # dual = 0.5 * translation_quaternion * rotation, hamilton, xyzw
        tx = matrices[j, 3, 0]
        ty = matrices[j, 3, 1]
        tz = matrices[j, 3, 2]

        duals[j, 0] = 0.5 * (qw * tx + ty * qz - tz * qy)
        duals[j, 1] = 0.5 * (qw * ty + tz * qx - tx * qz)
        duals[j, 2] = 0.5 * (qw * tz + tx * qy - ty * qx)
        duals[j, 3] = -0.5 * (tx * qx + ty * qy + tz * qz)


@njit(parallel=True, fastmath=True, cache=True)
def _dqs_compact(
    points:            np.ndarray,
    influence_indices: np.ndarray,
    weights:           np.ndarray,
    quats:             np.ndarray,
    duals:             np.ndarray,
    stretch:           np.ndarray,
    out:               np.ndarray,
) -> None:
    """Dual quaternion skinning over compact per-vertex influences.

    Args:
        points: ``(N, 3)`` bind-pose positions.
        influence_indices: ``(N, K)`` int32 columns into the joint arrays.
        weights: ``(N, K)`` blend weights.
        quats: ``(J, 4)`` unit rotation quaternions from :func:`_dqs_decompose`.
        duals: ``(J, 4)`` matching dual parts.
        stretch: ``(J, 3, 3)`` matching non-rigid remainders.
        out: ``(N, 3)`` destination, written in place.
    """
    n = points.shape[0]
    k = influence_indices.shape[1]

    for i in prange(n):
        # q and -q are the same rotation but average to nonsense, so every
        # influence is signed against the heaviest one before accumulating.
        pivot = 0
        best  = -1.0
        for c in range(k):
            if weights[i, c] > best:
                best  = weights[i, c]
                pivot = c

        pj  = influence_indices[i, pivot]
        px  = quats[pj, 0]
        py  = quats[pj, 1]
        pz  = quats[pj, 2]
        pw  = quats[pj, 3]

        qx  = 0.0
        qy  = 0.0
        qz  = 0.0
        qw  = 0.0
        dx  = 0.0
        dy  = 0.0
        dz  = 0.0
        dw  = 0.0

        s00 = 0.0
        s01 = 0.0
        s02 = 0.0
        s10 = 0.0
        s11 = 0.0
        s12 = 0.0
        s20 = 0.0
        s21 = 0.0
        s22 = 0.0

        for c in range(k):
            w = weights[i, c]
            if w == 0.0:
                continue

            j = influence_indices[i, c]

            dot = (
                quats[j, 0] * px
                + quats[j, 1] * py
                + quats[j, 2] * pz
                + quats[j, 3] * pw
            )
            sw = -w if dot < 0.0 else w

            qx += sw * quats[j, 0]
            qy += sw * quats[j, 1]
            qz += sw * quats[j, 2]
            qw += sw * quats[j, 3]

            dx += sw * duals[j, 0]
            dy += sw * duals[j, 1]
            dz += sw * duals[j, 2]
            dw += sw * duals[j, 3]

            s00 += w * stretch[j, 0, 0]
            s01 += w * stretch[j, 0, 1]
            s02 += w * stretch[j, 0, 2]
            s10 += w * stretch[j, 1, 0]
            s11 += w * stretch[j, 1, 1]
            s12 += w * stretch[j, 1, 2]
            s20 += w * stretch[j, 2, 0]
            s21 += w * stretch[j, 2, 1]
            s22 += w * stretch[j, 2, 2]

        norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        if norm < _DEGENERATE:
            # no usable rotation. LBS writes the origin for an all-zero row,
            # so match it rather than inventing a different answer.
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0
            continue

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
        dx /= norm
        dy /= norm
        dz /= norm
        dw /= norm

        # No re-orthogonalization of the dual part against the real part.
        # The translation is the VECTOR part of 2 * dual * conj(real), and
        # adding any multiple of `real` to `dual` contributes
        # lambda * real * conj(real) = lambda * |real|^2, a pure scalar. It
        # cannot reach the vector part, so the projection is exactly inert.

        # non-rigid part first, in row-vector order
        vx = points[i, 0] * s00 + points[i, 1] * s10 + points[i, 2] * s20
        vy = points[i, 0] * s01 + points[i, 1] * s11 + points[i, 2] * s21
        vz = points[i, 0] * s02 + points[i, 1] * s12 + points[i, 2] * s22

        # rotate: p + w*t + q x t, with t = 2 * (q x p)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)

        # translate: 2 * (dual * conj(real)), vector part
        out[i, 0] = rx + 2.0 * (qw * dx - dw * qx + qy * dz - qz * dy)
        out[i, 1] = ry + 2.0 * (qw * dy - dw * qy + qz * dx - qx * dz)
        out[i, 2] = rz + 2.0 * (qw * dz - dw * qz + qx * dy - qy * dx)


def _dqs_compact_numpy(
    points:            np.ndarray,
    influence_indices: np.ndarray,
    weights:           np.ndarray,
    quats:             np.ndarray,
    duals:             np.ndarray,
    stretch:           np.ndarray,
) -> np.ndarray:
    """Pure-numpy counterpart of :func:`_dqs_compact`.

    Doubles as the parity oracle for the kernel: an independent expression of
    the same math catches a transposed index the kernel alone would not.
    Materializes an ``(N, K, 3, 3)`` gather, so it is far hungrier than the
    kernel and is not a supported standalone path.
    """
    joints     = quats[influence_indices]
    pivot      = np.argmax(weights, axis=1)
    anchor     = quats[influence_indices[np.arange(len(points)), pivot]]

    signed     = np.where(np.einsum("nkc,nc->nk", joints, anchor) < 0.0, -1.0, 1.0)
    signed     = signed * weights

    real       = np.einsum("nk,nkc->nc",   signed,  joints)
    dual       = np.einsum("nk,nkc->nc",   signed,  duals[influence_indices])
    scale      = np.einsum("nk,nkij->nij", weights, stretch[influence_indices])

    norm       = np.linalg.norm(real, axis=1, keepdims=True)
    degenerate = (norm < _DEGENERATE).ravel()
    safe       = np.where(norm < _DEGENERATE, 1.0, norm)
    real       = real / safe
    dual       = dual / safe

    scaled     = np.einsum("ni,nij->nj", points, scale)

    vector     = real[:, :3]
    scalar     = real[:, 3:4]
    twice      = 2.0 * np.cross(vector, scaled)
    rotated    = scaled + scalar * twice + np.cross(vector, twice)

    shift = 2.0 * (
        scalar * dual[:, :3] - dual[:, 3:4] * vector + np.cross(vector, dual[:, :3])
    )

    result = rotated + shift
    result[degenerate] = 0.0
    return result


def dqs_compact_fast(
    points:            np.ndarray,
    influence_indices: np.ndarray,
    weights:           np.ndarray,
    matrices:          np.ndarray,
) -> np.ndarray:
    """Dual quaternion skinning, numba-accelerated with a numpy fallback.

    Args:
        points: ``(N, 3)`` bind-pose positions.
        influence_indices: ``(N, K)`` columns into *matrices*.
        weights: ``(N, K)`` blend weights.
        matrices: ``(J, 4, 4)`` skin matrices, ``inverse_bind @ world_posed``.

    Returns:
        ``(N, 3)`` deformed positions.  The input is not modified.
    """
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    influence_indices = np.ascontiguousarray(
        np.asarray(influence_indices, dtype=np.int32)
    )
    weights  = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    matrices = np.ascontiguousarray(np.asarray(matrices, dtype=np.float64))

    count    = matrices.shape[0]
    quats    = np.empty((count, 4),    dtype=np.float64)
    duals    = np.empty((count, 4),    dtype=np.float64)
    stretch  = np.empty((count, 3, 3), dtype=np.float64)

    out      = np.empty_like(points)
    try:
        _dqs_decompose(matrices, quats, duals, stretch)
        _dqs_compact(points, influence_indices, weights, quats, duals, stretch, out)
    except NumbaError:
        _dqs_decompose.py_func(matrices, quats, duals, stretch)
        return _dqs_compact_numpy(
            points, influence_indices, weights, quats, duals, stretch
        )
    return out