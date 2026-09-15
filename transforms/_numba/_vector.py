from math import acos, sin

import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def _vector_arc(vector0, vector1):
    angle = np.empty(vector0.shape[0], dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        vector0_ = np.empty(3, dtype=vector0.dtype)
        vector1_ = np.empty(3, dtype=vector0.dtype)

        m = (vector0[i, 0] ** 2 + vector0[i, 1] ** 2 + vector0[i, 2] ** 2) ** 0.5
        if m > 0.0:
            vector0_[0] = vector0[i, 0] / m
            vector0_[1] = vector0[i, 1] / m
            vector0_[2] = vector0[i, 2] / m

        m = (vector1[i, 0] ** 2 + vector1[i, 1] ** 2 + vector1[i, 2] ** 2) ** 0.5
        if m > 0.0:
            vector1_[0] = vector1[i, 0] / m
            vector1_[1] = vector1[i, 1] / m
            vector1_[2] = vector1[i, 2] / m

        dot = (
            (vector0_[0] * vector1_[0])
            + (vector0_[1] * vector1_[1])
            + (vector0_[2] * vector1_[2])
        )
        if dot > 1.0:
            dot = 1.0
        elif dot < -1.0:
            dot = -1.0

        angle[i] = acos(dot)

    return angle


@njit(fastmath=True, parallel=True, cache=True)
def _vector_slerp(vector0, vector1, weight):
    v = np.empty((vector0.shape[0], 3), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        X0, Y0, Z0 = 0.0, 0.0, 0.0
        X1, Y1, Z1 = 0.0, 0.0, 0.0

        m = (vector0[i, 0] ** 2.0 + vector0[i, 1] ** 2.0 + vector0[i, 2] ** 2) ** 0.5
        if m > 0.0:
            X0 = vector0[i, 0] / m
            Y0 = vector0[i, 1] / m
            Z0 = vector0[i, 2] / m

        m = (vector1[i, 0] ** 2.0 + vector1[i, 1] ** 2.0 + vector1[i, 2] ** 2.0) ** 0.5
        if m > 0.0:
            X1 = vector1[i, 0] / m
            Y1 = vector1[i, 1] / m
            Z1 = vector1[i, 2] / m

        dot    = (X0 * X1) + (Y0 * Y1) + (Z0 * Z1)
        angle  = acos(dot)
        sangle = sin(angle)

        if sangle > 0.0:
            w0 = sin((1.0 - weight[i]) * angle)
            w1 = sin(weight[i] * angle)

            v[i, 0] = (vector0[i, 0] * w0 + vector1[i, 0] * w1) / sangle
            v[i, 1] = (vector0[i, 1] * w0 + vector1[i, 1] * w1) / sangle
            v[i, 2] = (vector0[i, 2] * w0 + vector1[i, 2] * w1) / sangle
        else:
            v[i, 0] = vector0[i, 0]
            v[i, 1] = vector0[i, 1]
            v[i, 2] = vector0[i, 2]

    return v


@njit(fastmath=True, parallel=True, cache=True)
def _vector_lerp(vector0, vector1, weight):
    v = np.empty((vector0.shape[0], vector0.shape[1]), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        for j in range(vector0.shape[1]):
            v[i, j] = vector0[i, j] + weight[i] * (vector1[i, j] - vector0[i, j])

    return v


# NOTE: no explicit signature. Levelled inputs may arrive as read-only
# broadcast views, which an eager ``float64[:,:]`` signature rejects; lazy
# compilation specialises per layout instead, as every other kernel here does.
@njit(fastmath=True, parallel=True, cache=True)
def _vector_cross(vector0, vector1):
    vector = np.zeros((vector0.shape[0], 3), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        vector[i, 0] = vector0[i, 1] * vector1[i, 2] - vector0[i, 2] * vector1[i, 1]
        vector[i, 1] = vector0[i, 2] * vector1[i, 0] - vector0[i, 0] * vector1[i, 2]
        vector[i, 2] = vector0[i, 0] * vector1[i, 1] - vector0[i, 1] * vector1[i, 0]

    return vector


@njit(fastmath=True, parallel=True, cache=True)
def _vector_dot(vector0, vector1):
    dot = np.empty(vector0.shape[0], dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        dot[i] = 0
        for j in range(vector0.shape[1]):
            dot[i] += vector0[i, j] * vector1[i, j]

    return dot


@njit(fastmath=True, parallel=True, cache=True)
def _vector_magnitude(vector):
    mag = np.empty(vector.shape[0], dtype=vector.dtype)
    for i in prange(vector.shape[0]):
        mag[i] = 0.0
        for j in range(vector.shape[1]):
            mag[i] = mag[i] + vector[i, j] ** 2

        mag[i] = mag[i] ** 0.5

    return mag


@njit(fastmath=True, parallel=True, cache=True)
def _vector_normalize(vector):
    vector_ = np.empty(vector.shape, dtype=vector.dtype)

    for i in prange(vector.shape[0]):
        mag = 0
        for j in range(vector.shape[1]):
            mag += vector[i, j] ** 2

        mag = mag**0.5

        for j in range(vector.shape[1]):
            vector_[i, j] = vector[i, j] / mag

    return vector_


@njit(fastmath=True, parallel=True, cache=True)
def _vector_to_matrix(vector0, vector1, aim_axis, up_axis):
    matrix = np.empty((vector0.shape[0], 4, 4), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        vector0_ = np.empty(3, dtype=vector0.dtype)
        vector1_ = np.empty(3, dtype=vector0.dtype)

        ii   = aim_axis[i]
        jj   = up_axis[i]
        kk   = (min(ii, jj) - max(ii, jj) + min(ii, jj)) % 3

        flip = 0
        if ii == 0 and jj == 2:
            flip = 1
        elif ii == 1 and jj == 0:
            flip = 1
        elif ii == 2 and jj == 1:
            flip = 1

        for j in range(3):
            vector0_[j] = vector0[i, j]
            vector1_[j] = vector1[i, j]

        # init matrix output
        matrix[i, 0, 0] = 1.0
        matrix[i, 0, 1] = 0.0
        matrix[i, 0, 2] = 0.0
        matrix[i, 0, 3] = 0.0

        matrix[i, 1, 0] = 0.0
        matrix[i, 1, 1] = 1.0
        matrix[i, 1, 2] = 0.0
        matrix[i, 1, 3] = 0.0

        matrix[i, 2, 0] = 0.0
        matrix[i, 2, 1] = 0.0
        matrix[i, 2, 2] = 1.0
        matrix[i, 2, 3] = 0.0

        matrix[i, 3, 0] = 0.0
        matrix[i, 3, 1] = 0.0
        matrix[i, 3, 2] = 0.0
        matrix[i, 3, 3] = 1.0

        x  = vector0_[1] * vector1_[2] - vector0_[2] * vector1_[1]
        y  = vector0_[2] * vector1_[0] - vector0_[0] * vector1_[2]
        z  = vector0_[0] * vector1_[1] - vector0_[1] * vector1_[0]

        na = (vector0_[0] ** 2 + vector0_[1] ** 2 + vector0_[2] ** 2) ** 0.5
        nc = (x**2 + y**2 + z**2) ** 0.5

        if na > 0.0 and nc > 0.0:
            matrix[i, kk, 0] = x / nc
            matrix[i, ii, 0] = vector0_[0] / na
            matrix[i, kk, 1] = y / nc
            matrix[i, ii, 1] = vector0_[1] / na
            matrix[i, kk, 2] = z / nc
            matrix[i, ii, 2] = vector0_[2] / na

            matrix[i, jj, 0] = (
                matrix[i, kk, 1] * matrix[i, ii, 2]
                - matrix[i, kk, 2] * matrix[i, ii, 1]
            )
            matrix[i, jj, 1] = (
                matrix[i, kk, 2] * matrix[i, ii, 0]
                - matrix[i, kk, 0] * matrix[i, ii, 2]
            )
            matrix[i, jj, 2] = (
                matrix[i, kk, 0] * matrix[i, ii, 1]
                - matrix[i, kk, 1] * matrix[i, ii, 0]
            )

            if flip:
                matrix[i, kk, 0] = 0 - matrix[i, kk, 0]
                matrix[i, kk, 1] = 0 - matrix[i, kk, 1]
                matrix[i, kk, 2] = 0 - matrix[i, kk, 2]

    return matrix


@njit(fastmath=True, parallel=True, cache=True)
def _vector_arc_to_quaternion(vector0, vector1):
    """Computes a list of quaternions qi,qj,qk,qw representing the arc between lists of vectors"""

    quat = np.empty((vector0.shape[0], 4), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        v0 = np.empty(3, dtype=vector0.dtype)
        v1 = np.empty(3, dtype=vector0.dtype)
        h  = np.empty(3, dtype=vector0.dtype)

        mag   = (vector0[i, 0] ** 2 + vector0[i, 1] ** 2 + vector0[i, 2] ** 2) ** 0.5
        v0[0] = vector0[i, 0] / mag
        v0[1] = vector0[i, 1] / mag
        v0[2] = vector0[i, 2] / mag

        mag   = (vector1[i, 0] ** 2 + vector1[i, 1] ** 2 + vector1[i, 2] ** 2) ** 0.5
        v1[0] = vector1[i, 0] / mag
        v1[1] = vector1[i, 1] / mag
        v1[2] = vector1[i, 2] / mag

        h[0]  = v0[0] + v1[0]
        h[1]  = v0[1] + v1[1]
        h[2]  = v0[2] + v1[2]

        mag   = (h[0] ** 2 + h[1] ** 2 + h[2] ** 2) ** 0.5

        h[0]  = h[0] / mag
        h[1]  = h[1] / mag
        h[2]  = h[2] / mag

        quat[i, 0] = v0[1] * h[2] - v0[2] * h[1]
        quat[i, 1] = v0[2] * h[0] - v0[0] * h[2]
        quat[i, 2] = v0[0] * h[1] - v0[1] * h[0]
        quat[i, 3] = v0[0] * h[0] + v0[1] * h[1] + v0[2] * h[2]

    return quat