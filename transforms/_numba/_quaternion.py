from math import acos, atan2, cos, sin

import numpy as np
from numba import njit, prange

EPSILON = np.finfo(np.float32).eps


@njit(fastmath=True, parallel=True, cache=True)
def _quaternion_slerp(quat0, quat1, weight):
    """Calculates spherical interpolation between two lists quaternions"""
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        cosHalfTheta = (
            quat0[i, 3] * quat1[i, 3]
            + quat0[i, 0] * quat1[i, 0]
            + quat0[i, 1] * quat1[i, 1]
            + quat0[i, 2] * quat1[i, 2]
        )

        if abs(cosHalfTheta) >= 1.0:
            quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3] = (
                quat0[i, 0],
                quat0[i, 1],
                quat0[i, 2],
                quat0[i, 3],
            )

        else:
            halfTheta    = acos(cosHalfTheta)
            sinHalfTheta = (1.0 - cosHalfTheta * cosHalfTheta) ** 0.5

            if abs(sinHalfTheta) < EPSILON:
                quat[i, 0] = quat0[i, 0] * 0.5 + quat1[i, 0] * 0.5
                quat[i, 1] = quat0[i, 1] * 0.5 + quat1[i, 1] * 0.5
                quat[i, 2] = quat0[i, 2] * 0.5 + quat1[i, 2] * 0.5
                quat[i, 3] = quat0[i, 3] * 0.5 + quat1[i, 3] * 0.5

            else:
                ratioA = sin((1 - weight[i]) * halfTheta) / sinHalfTheta
                ratioB = sin(weight[i] * halfTheta) / sinHalfTheta

                quat[i, 0] = quat0[i, 0] * ratioA + quat1[i, 0] * ratioB
                quat[i, 1] = quat0[i, 1] * ratioA + quat1[i, 1] * ratioB
                quat[i, 2] = quat0[i, 2] * ratioA + quat1[i, 2] * ratioB
                quat[i, 3] = quat0[i, 3] * ratioA + quat1[i, 3] * ratioB

    return quat


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
def _quaternion_add(quat0, quat1):
    """Adds 2 lists of quaternions"""
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        quat[i, 0] = quat0[i, 0] + quat1[i, 0]
        quat[i, 1] = quat0[i, 1] + quat1[i, 1]
        quat[i, 2] = quat0[i, 2] + quat1[i, 2]
        quat[i, 3] = quat0[i, 3] + quat1[i, 3]

    return quat


@njit(fastmath=True, parallel=True, cache=True)
def _quaternion_sub(quat0, quat1):
    """Subtracts 2 lists of quaternions"""
    quat = np.empty((quat0.shape[0], 4), dtype=quat0.dtype)

    for i in prange(quat0.shape[0]):
        quat[i, 0] = quat0[i, 0] - quat1[i, 0]
        quat[i, 1] = quat0[i, 1] - quat1[i, 1]
        quat[i, 2] = quat0[i, 2] - quat1[i, 2]
        quat[i, 3] = quat0[i, 3] - quat1[i, 3]

    return quat


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
def _quaternion_conjugate(quat):
    """Conjugates a list of quaternions"""
    quat_ = np.empty((quat.shape[0], 4), dtype=quat.dtype)

    for i in prange(quat.shape[0]):
        quat_[i, 0] = -quat[i, 0]
        quat_[i, 1] = -quat[i, 1]
        quat_[i, 2] = -quat[i, 2]
        quat_[i, 3] = quat[i, 3]

    return quat_


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
def _quaternion_negate(quat):
    """Negates a list of quaternions"""
    quat_ = np.empty((quat.shape[0], 4), dtype=quat.dtype)

    for i in prange(quat.shape[0]):
        quat_[i, 0] = -quat[i, 0]
        quat_[i, 1] = -quat[i, 1]
        quat_[i, 2] = -quat[i, 2]
        quat_[i, 3] = -quat[i, 3]

    return quat_


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
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