from math import atan2, cos, sin, sqrt

import numpy as np
from numba import njit, prange

EPSILON  = np.finfo(np.float32).eps

ROUNDING = 2.0**-50
"""A perpendicular this short is rounding: the two unit quaternions it was
taken from are opposite to within a few ulps, and give it no direction."""

SQUARES_LOW = 2.0**-968
"""A sum of squares from this up kept every bit: below it, the squares of
the smaller components can have lost bits as subnormals."""

SQUARES_HIGH = np.finfo(np.float64).max
"""A sum of squares above this overflowed, or had an inf or a nan in it."""


@njit(cache=True, error_model="numpy")
def _unit4(x, y, z, w):
    """The length of ``(x, y, z, w)`` and the four scaled to unit length.

    A quaternion under about 1e-146 or over about 1e154 long is first
    divided by its largest component, so that the squares neither lose
    bits nor overflow. A zero quaternion has a length of 0 and comes back
    as it is; a nan or an inf in it makes all five nan.
    """
    squares = x * x + y * y + z * z + w * w
    if not (SQUARES_LOW <= squares <= SQUARES_HIGH):
        m = max(max(abs(x), abs(y)), max(abs(z), abs(w)))
        if m == 0.0 and squares == 0.0:
            return 0.0, x, y, z, w

        x       = x / m
        y       = y / m
        z       = z / m
        w       = w / m
        squares = x * x + y * y + z * z + w * w
        n       = sqrt(squares)
        r       = 1.0 / n
        return m * n, x * r, y * r, z * r, w * r

    n = sqrt(squares)
    r = 1.0 / n
    return n, x * r, y * r, z * r, w * r


@njit(parallel=True, cache=True)
def _quaternion_slerp(quat0, quat1, weight):
    """Spherical linear interpolation between two lists of quaternions.

    Turns ``quat0`` toward ``quat1`` along the great arc between them, as
    given: picking the shorter arc, by negating ``quat1``, is the caller's.

    The arc's angle is ``2 * atan2(|q1 - q0|, |q1 + q0|)``, and ``q0`` turns
    toward ``q1``'s part perpendicular to it, taken from the difference of
    the two within a right angle and from their sum past one: the
    difference keeps every digit of nearly parallel quaternions, the sum of
    nearly opposite ones. The ``acos`` of their dot product, divided by
    ``sqrt(1 - dot ** 2)``, lost half the digits of a small angle.

    Both quaternions are normalised first, so the angle is the one between
    their rotations whatever their lengths, 1e-300 to 1e300. A weight of 0
    or 1 returns ``quat0`` or ``quat1`` exactly, any other weight a unit
    quaternion. Quaternions opposite to within rounding -- one rotation,
    asked to go the long way round -- give no axis to turn about: the
    result stays at ``quat0``. A zero quaternion has no rotation to
    normalise: the result is ``quat0 + weight * (quat1 - quat0)``. Between
    the ends, a nan or an inf component gives nans.
    """
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        t = weight[i]

        n0, x0, y0, z0, w0 = _unit4(quat0[i, 0], quat0[i, 1], quat0[i, 2], quat0[i, 3])
        n1, x1, y1, z1, w1 = _unit4(quat1[i, 0], quat1[i, 1], quat1[i, 2], quat1[i, 3])

        if t == 0.0:
            quat[i, 0] = quat0[i, 0]
            quat[i, 1] = quat0[i, 1]
            quat[i, 2] = quat0[i, 2]
            quat[i, 3] = quat0[i, 3]

        elif t == 1.0:
            quat[i, 0] = quat1[i, 0]
            quat[i, 1] = quat1[i, 1]
            quat[i, 2] = quat1[i, 2]
            quat[i, 3] = quat1[i, 3]

        elif n0 == 0.0 or n1 == 0.0:
            # a zero quaternion: no rotation to turn from or to
            quat[i, 0] = quat0[i, 0] + t * (quat1[i, 0] - quat0[i, 0])
            quat[i, 1] = quat0[i, 1] + t * (quat1[i, 1] - quat0[i, 1])
            quat[i, 2] = quat0[i, 2] + t * (quat1[i, 2] - quat0[i, 2])
            quat[i, 3] = quat0[i, 3] + t * (quat1[i, 3] - quat0[i, 3])

        else:
            # the angle between the two from their difference and their sum
            dx    = x1 - x0
            dy    = y1 - y0
            dz    = z1 - z0
            dw    = w1 - w0
            sx    = x1 + x0
            sy    = y1 + y0
            sz    = z1 + z0
            sw    = w1 + w0
            dn    = sqrt(dx * dx + dy * dy + dz * dz + dw * dw)
            sn    = sqrt(sx * sx + sy * sy + sz * sz + sw * sw)
            angle = 2.0 * atan2(dn, sn)

            # q1's part perpendicular to q0, from the shorter of the two
            near = dn <= sn
            if near:
                dot = x0 * dx + y0 * dy + z0 * dz + w0 * dw
                px  = dx - dot * x0
                py  = dy - dot * y0
                pz  = dz - dot * z0
                pw  = dw - dot * w0
            else:
                dot = x0 * sx + y0 * sy + z0 * sz + w0 * sw
                px  = sx - dot * x0
                py  = sy - dot * y0
                pz  = sz - dot * z0
                pw  = sw - dot * w0
            pn = sqrt(px * px + py * py + pz * pz + pw * pw)

            if pn > 0.0 and (near or pn > ROUNDING):
                a = cos(t * angle)
                b = sin(t * angle) / pn
            else:
                a = 1.0
                b = 0.0

            # unit to an ulp already: the normalisation is a guard
            x = a * x0 + b * px
            y = a * y0 + b * py
            z = a * z0 + b * pz
            w = a * w0 + b * pw
            r = 1.0 / sqrt(x * x + y * y + z * z + w * w)

            quat[i, 0] = x * r
            quat[i, 1] = y * r
            quat[i, 2] = z * r
            quat[i, 3] = w * r

    return quat


@njit(parallel=True, cache=True)
def _quaternion_multiply(quat0, quat1):
    """Multiplies 2 lists of quaternions"""
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        quat[i, 0] = (
            quat0[i, 0] * quat1[i, 3]
            + quat0[i, 1] * quat1[i, 2]
            - quat0[i, 2] * quat1[i, 1]
            + quat0[i, 3] * quat1[i, 0]
        )
        quat[i, 1] = (
            -quat0[i, 0] * quat1[i, 2]
            + quat0[i, 1] * quat1[i, 3]
            + quat0[i, 2] * quat1[i, 0]
            + quat0[i, 3] * quat1[i, 1]
        )
        quat[i, 2] = (
            quat0[i, 0] * quat1[i, 1]
            - quat0[i, 1] * quat1[i, 0]
            + quat0[i, 2] * quat1[i, 3]
            + quat0[i, 3] * quat1[i, 2]
        )
        quat[i, 3] = (
            -quat0[i, 0] * quat1[i, 0]
            - quat0[i, 1] * quat1[i, 1]
            - quat0[i, 2] * quat1[i, 2]
            + quat0[i, 3] * quat1[i, 3]
        )

    return quat


@njit(parallel=True, cache=True)
def _quaternion_add(quat0, quat1):
    """Adds 2 lists of quaternions"""
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        quat[i, 0] = quat0[i, 0] + quat1[i, 0]
        quat[i, 1] = quat0[i, 1] + quat1[i, 1]
        quat[i, 2] = quat0[i, 2] + quat1[i, 2]
        quat[i, 3] = quat0[i, 3] + quat1[i, 3]

    return quat


@njit(parallel=True, cache=True)
def _quaternion_sub(quat0, quat1):
    """Subtracts 2 lists of quaternions"""
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        quat[i, 0] = quat0[i, 0] - quat1[i, 0]
        quat[i, 1] = quat0[i, 1] - quat1[i, 1]
        quat[i, 2] = quat0[i, 2] - quat1[i, 2]
        quat[i, 3] = quat0[i, 3] - quat1[i, 3]

    return quat


@njit(parallel=True, cache=True)
def _quaternion_to_matrix(quat):
    # Init Matrix
    matrix = np.empty((quat.shape[0], 4, 4), dtype=quat.dtype)

    # For every quaternion
    for i in prange(quat.shape[0]):
        xx = quat[i, 0] * quat[i, 0]
        xy = quat[i, 0] * quat[i, 1]
        xz = quat[i, 0] * quat[i, 2]
        xw = quat[i, 0] * quat[i, 3]

        yy = quat[i, 1] * quat[i, 1]
        yz = quat[i, 1] * quat[i, 2]
        yw = quat[i, 1] * quat[i, 3]

        zz = quat[i, 2] * quat[i, 2]
        zw = quat[i, 2] * quat[i, 3]

        matrix[i, 0, 0] = 1 - 2 * (yy + zz)
        matrix[i, 1, 0] = 2 * (xy - zw)
        matrix[i, 2, 0] = 2 * (xz + yw)

        matrix[i, 0, 1] = 2 * (xy + zw)
        matrix[i, 1, 1] = 1 - 2 * (xx + zz)
        matrix[i, 2, 1] = 2 * (yz - xw)

        matrix[i, 0, 2] = 2 * (xz - yw)
        matrix[i, 1, 2] = 2 * (yz + xw)
        matrix[i, 2, 2] = 1 - 2 * (xx + yy)

        matrix[i, 0, 3] = 0.0
        matrix[i, 1, 3] = 0.0
        matrix[i, 2, 3] = 0.0
        matrix[i, 3, 0] = 0.0
        matrix[i, 3, 1] = 0.0
        matrix[i, 3, 2] = 0.0
        matrix[i, 3, 3] = 1.0

    return matrix


@njit(parallel=True, cache=True)
def _quaternion_conjugate(quat):
    """Conjugates a list of quaternions"""
    quat_ = np.empty((quat.shape[0], 4), dtype=quat.dtype)

    for i in prange(quat.shape[0]):
        quat_[i, 0] = -quat[i, 0]
        quat_[i, 1] = -quat[i, 1]
        quat_[i, 2] = -quat[i, 2]
        quat_[i, 3] = quat[i, 3]

    return quat_


@njit(parallel=True, cache=True)
def _quaternion_inverse(quat):
    """Inverses a list of quaternions"""
    quat_ = np.empty((quat.shape[0], 4), dtype=quat.dtype)

    for i in prange(quat.shape[0]):
        lenSquared = (
            quat[i, 0] ** 2 + quat[i, 1] ** 2 + quat[i, 2] ** 2 + quat[i, 3] ** 2
        )

        quat_[i, 0] = -quat[i, 0] / lenSquared
        quat_[i, 1] = -quat[i, 1] / lenSquared
        quat_[i, 2] = -quat[i, 2] / lenSquared
        quat_[i, 3] = quat[i, 3] / lenSquared

    return quat_


@njit(parallel=True, cache=True)
def _quaternion_negate(quat):
    """Negates a list of quaternions"""
    quat_ = np.empty((quat.shape[0], 4), dtype=quat.dtype)

    for i in prange(quat.shape[0]):
        quat_[i, 0] = -quat[i, 0]
        quat_[i, 1] = -quat[i, 1]
        quat_[i, 2] = -quat[i, 2]
        quat_[i, 3] = -quat[i, 3]

    return quat_


@njit(parallel=True, cache=True)
def _quaternion_log(quat):
    """Logarithm of unit quaternions -> (N, 3) rotation vectors (axis * angle).

    For a unit quaternion (x, y, z, w): theta = atan2(|v|, w) and the result is
    theta * v / |v|. The identity quaternion (and any with a vanishing vector
    part) maps to the zero vector.
    """
    out = np.empty((quat.shape[0], 3), dtype=quat.dtype)

    for i in prange(quat.shape[0]):
        vnorm = (quat[i, 0] ** 2 + quat[i, 1] ** 2 + quat[i, 2] ** 2) ** 0.5

        if vnorm < EPSILON:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0
        else:
            theta     = atan2(vnorm, quat[i, 3])
            scale     = theta / vnorm
            out[i, 0] = quat[i, 0] * scale
            out[i, 1] = quat[i, 1] * scale
            out[i, 2] = quat[i, 2] * scale

    return out


@njit(parallel=True, cache=True)
def _quaternion_exp(rvec):
    """Exponential of rotation vectors (N, 3) -> unit quaternions (N, 4).

    Inverse of ``_quaternion_log``: theta = |rvec|, returns
    (sin(theta) * rvec / theta, cos(theta)). The zero vector maps to identity.
    """
    out = np.empty((rvec.shape[0], 4), dtype=rvec.dtype)

    for i in prange(rvec.shape[0]):
        theta = (rvec[i, 0] ** 2 + rvec[i, 1] ** 2 + rvec[i, 2] ** 2) ** 0.5

        if theta < EPSILON:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0
            out[i, 3] = 1.0
        else:
            s         = sin(theta) / theta
            out[i, 0] = rvec[i, 0] * s
            out[i, 1] = rvec[i, 1] * s
            out[i, 2] = rvec[i, 2] * s
            out[i, 3] = cos(theta)

    return out


@njit(parallel=True, cache=True)
def _quaternion_nlerp(quat0, quat1, weight):
    """Normalised linear interpolation between two lists of quaternions.

    Hemisphere alignment (flipping ``quat1`` when the dot product is negative)
    is the caller's responsibility, matching ``_quaternion_slerp``.
    """
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        a   = 1.0 - weight[i]
        b   = weight[i]

        x   = quat0[i, 0] * a + quat1[i, 0] * b
        y   = quat0[i, 1] * a + quat1[i, 1] * b
        z   = quat0[i, 2] * a + quat1[i, 2] * b
        w   = quat0[i, 3] * a + quat1[i, 3] * b

        mag = (x * x + y * y + z * z + w * w) ** 0.5
        if mag < EPSILON:
            quat[i, 0] = 0.0
            quat[i, 1] = 0.0
            quat[i, 2] = 0.0
            quat[i, 3] = 1.0
        else:
            quat[i, 0] = x / mag
            quat[i, 1] = y / mag
            quat[i, 2] = z / mag
            quat[i, 3] = w / mag

    return quat