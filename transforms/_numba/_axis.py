from math import cos, sin

import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def _axis_angle_to_quaternion(axis, angle):
    """Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles"""
    quat = np.empty((axis.shape[0], 4), dtype=axis.dtype)

    for i in prange(axis.shape[0]):
        axis_ = np.empty(3, dtype=axis.dtype)

        mag        = (axis[i, 0] ** 2 + axis[i, 1] ** 2 + axis[i, 2] ** 2) ** 0.5
        axis_[0]   = axis[i, 0] / mag
        axis_[1]   = axis[i, 1] / mag
        axis_[2]   = axis[i, 2] / mag

        s          = sin(angle[i] / 2)
        quat[i, 0] = axis_[0] * s
        quat[i, 1] = axis_[1] * s
        quat[i, 2] = axis_[2] * s
        quat[i, 3] = cos(angle[i] / 2)

    return quat


@njit(fastmath=True, parallel=True, cache=True)
def _axis_angle_to_matrix(axis, angle):
    """Computes a list of orthogonal 4x4 matrices from lists of axes and angles"""

    matrix = np.empty((axis.shape[0], 4, 4), dtype=axis.dtype)

    for i in prange(axis.shape[0]):
        sin_    = sin(angle[i])
        cos_    = cos(angle[i])
        inv_cos = 1 - cos_

        mag  = (axis[i, 0] ** 2 + axis[i, 1] ** 2 + axis[i, 2] ** 2) ** 0.5
        u    = axis[i, 0] / mag
        v    = axis[i, 1] / mag
        w    = axis[i, 2] / mag
        uv   = u * v
        uw   = u * w
        vw   = v * w
        usin = u * sin_
        vsin = v * sin_
        wsin = w * sin_
        u2   = u**2
        v2   = v**2
        w2   = w**2

        matrix[i, 0, 0] = u2 + ((v2 + w2) * cos_)
        matrix[i, 0, 1] = uv * inv_cos + (wsin)
        matrix[i, 0, 2] = uw * inv_cos - (vsin)
        matrix[i, 1, 0] = uv * inv_cos - (wsin)
        matrix[i, 1, 1] = v2 + ((u2 + w2) * cos_)
        matrix[i, 1, 2] = vw * inv_cos + (usin)
        matrix[i, 2, 0] = uw * inv_cos + (vsin)
        matrix[i, 2, 1] = vw * inv_cos - (usin)
        matrix[i, 2, 2] = w2 + ((u2 + v2) * cos_)
        matrix[i, 0, 3] = 0.0
        matrix[i, 1, 3] = 0.0
        matrix[i, 2, 3] = 0.0
        matrix[i, 3, 0] = 0.0
        matrix[i, 3, 1] = 0.0
        matrix[i, 3, 2] = 0.0
        matrix[i, 3, 3] = 1.0

    return matrix