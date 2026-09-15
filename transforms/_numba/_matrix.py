from math import atan2

import numpy as np
from numba import njit, prange
from cgmath.transforms._numba._euler import _get_euler_order, EA_MAYA, EPSILON


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_identity(count):
    matrix = np.zeros((count, 4, 4), dtype=np.float64)
    for i in prange(count):
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

    return matrix


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_to_quaternion(matrix):
    """Converts 4x4 matrix to quaternions qi,qj,qk,qw"""
    quat = np.zeros((matrix.shape[0], 4), dtype=matrix.dtype)

    for i in prange(matrix.shape[0]):
        trace = matrix[i, 0, 0] + matrix[i, 1, 1] + matrix[i, 2, 2]

        if trace > 0.0:
            s          = 0.5 / (trace + 1.0) ** 0.5
            quat[i, 0] = (matrix[i, 1, 2] - matrix[i, 2, 1]) * s
            quat[i, 1] = (matrix[i, 2, 0] - matrix[i, 0, 2]) * s
            quat[i, 2] = (matrix[i, 0, 1] - matrix[i, 1, 0]) * s
            quat[i, 3] = 0.25 / s

        elif matrix[i, 0, 0] > matrix[i, 1, 1] and matrix[i, 0, 0] > matrix[i, 2, 2]:
            s          = 2.0 * (1.0 + matrix[i, 0, 0] - matrix[i, 1, 1] - matrix[i, 2, 2]) ** 0.5
            quat[i, 0] = 0.25 * s
            quat[i, 1] = (matrix[i, 1, 0] + matrix[i, 0, 1]) / s
            quat[i, 2] = (matrix[i, 2, 0] + matrix[i, 0, 2]) / s
            quat[i, 3] = (matrix[i, 1, 2] - matrix[i, 2, 1]) / s

        elif matrix[i, 1, 1] > matrix[i, 2, 2]:
            s          = 2.0 * (1.0 + matrix[i, 1, 1] - matrix[i, 0, 0] - matrix[i, 2, 2]) ** 0.5
            quat[i, 0] = (matrix[i, 1, 0] + matrix[i, 0, 1]) / s
            quat[i, 1] = 0.25 * s
            quat[i, 2] = (matrix[i, 2, 1] + matrix[i, 1, 2]) / s
            quat[i, 3] = (matrix[i, 2, 0] - matrix[i, 0, 2]) / s

        else:
            s          = 2.0 * (1.0 + matrix[i, 2, 2] - matrix[i, 0, 0] - matrix[i, 1, 1]) ** 0.5
            quat[i, 0] = (matrix[i, 2, 0] + matrix[i, 0, 2]) / s
            quat[i, 1] = (matrix[i, 2, 1] + matrix[i, 1, 2]) / s
            quat[i, 2] = 0.25 * s
            quat[i, 3] = (matrix[i, 0, 1] - matrix[i, 1, 0]) / s

    return quat


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_inverse(matrix):
    """Numba prange equivalent of `matrix_inverse`.

    Same assumptions and result, written as an explicit parallel loop over the
    batch axis instead of vectorized numpy.
    """

    # Init inverse Matrix
    m_ = np.zeros(matrix.shape, dtype=matrix.dtype)

    # For every matrix
    for i in prange(matrix.shape[0]):
        # Squared scale per source row of the upper-left 3x3
        sx = matrix[i, 0, 0] ** 2 + matrix[i, 0, 1] ** 2 + matrix[i, 0, 2] ** 2
        sy = matrix[i, 1, 0] ** 2 + matrix[i, 1, 1] ** 2 + matrix[i, 1, 2] ** 2
        sz = matrix[i, 2, 0] ** 2 + matrix[i, 2, 1] ** 2 + matrix[i, 2, 2] ** 2

        # Inverse rotation+scale: transpose, dividing each column by its
        # source row's squared norm (M^T @ diag(1/s))
        m_[i, 0, 0] = matrix[i, 0, 0] / sx
        m_[i, 0, 1] = matrix[i, 1, 0] / sy
        m_[i, 0, 2] = matrix[i, 2, 0] / sz
        m_[i, 1, 0] = matrix[i, 0, 1] / sx
        m_[i, 1, 1] = matrix[i, 1, 1] / sy
        m_[i, 1, 2] = matrix[i, 2, 1] / sz
        m_[i, 2, 0] = matrix[i, 0, 2] / sx
        m_[i, 2, 1] = matrix[i, 1, 2] / sy
        m_[i, 2, 2] = matrix[i, 2, 2] / sz

        # Inverse translation: -t @ inv_rot
        m_[i, 3, 0] = -1 * (
            matrix[i, 3, 0] * m_[i, 0, 0]
            + matrix[i, 3, 1] * m_[i, 1, 0]
            + matrix[i, 3, 2] * m_[i, 2, 0]
        )
        m_[i, 3, 1] = -1 * (
            matrix[i, 3, 0] * m_[i, 0, 1]
            + matrix[i, 3, 1] * m_[i, 1, 1]
            + matrix[i, 3, 2] * m_[i, 2, 1]
        )
        m_[i, 3, 2] = -1 * (
            matrix[i, 3, 0] * m_[i, 0, 2]
            + matrix[i, 3, 1] * m_[i, 1, 2]
            + matrix[i, 3, 2] * m_[i, 2, 2]
        )
        m_[i, 3, 3] = 1.0

    return m_


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_transpose(matrix):
    """Assumes matrix is 4x4 orthogonal"""

    # Init inverse Matrix
    m_ = np.zeros(matrix.shape, dtype=matrix.dtype)

    # For every matrix
    for i in prange(matrix.shape[0]):
        # Transpose
        m_[i, 0, 0] = matrix[i, 0, 0]
        m_[i, 0, 1] = matrix[i, 1, 0]
        m_[i, 0, 2] = matrix[i, 2, 0]
        m_[i, 0, 3] = matrix[i, 3, 0]
        m_[i, 1, 0] = matrix[i, 0, 1]
        m_[i, 1, 1] = matrix[i, 1, 1]
        m_[i, 1, 2] = matrix[i, 2, 1]
        m_[i, 1, 3] = matrix[i, 3, 1]
        m_[i, 2, 0] = matrix[i, 0, 2]
        m_[i, 2, 1] = matrix[i, 1, 2]
        m_[i, 2, 2] = matrix[i, 2, 2]
        m_[i, 2, 3] = matrix[i, 3, 2]
        m_[i, 3, 0] = matrix[i, 0, 3]
        m_[i, 3, 1] = matrix[i, 1, 3]
        m_[i, 3, 2] = matrix[i, 2, 3]
        m_[i, 3, 3] = matrix[i, 3, 3]

    return m_


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_normalize(matrix):
    """Normalizes the rotation component of a transform matrix"""
    m_ = np.zeros(matrix.shape, dtype=matrix.dtype)

    # For every matrix
    for i in prange(matrix.shape[0]):
        x = (matrix[i, 0, 0] ** 2 + matrix[i, 0, 1] ** 2 + matrix[i, 0, 2] ** 2) ** 0.5
        y = (matrix[i, 1, 0] ** 2 + matrix[i, 1, 1] ** 2 + matrix[i, 1, 2] ** 2) ** 0.5
        z = (matrix[i, 2, 0] ** 2 + matrix[i, 2, 1] ** 2 + matrix[i, 2, 2] ** 2) ** 0.5

        m_[i, 0, 0] = matrix[i, 0, 0] / x
        m_[i, 0, 1] = matrix[i, 0, 1] / x
        m_[i, 0, 2] = matrix[i, 0, 2] / x
        m_[i, 0, 3] = matrix[i, 0, 3]

        m_[i, 1, 0] = matrix[i, 1, 0] / y
        m_[i, 1, 1] = matrix[i, 1, 1] / y
        m_[i, 1, 2] = matrix[i, 1, 2] / y
        m_[i, 1, 3] = matrix[i, 1, 3]

        m_[i, 2, 0] = matrix[i, 2, 0] / z
        m_[i, 2, 1] = matrix[i, 2, 1] / z
        m_[i, 2, 2] = matrix[i, 2, 2] / z
        m_[i, 2, 3] = matrix[i, 2, 3]

        m_[i, 3, 0] = matrix[i, 3, 0]
        m_[i, 3, 1] = matrix[i, 3, 1]
        m_[i, 3, 2] = matrix[i, 3, 2]
        m_[i, 3, 3] = matrix[i, 3, 3]

    return m_


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_multiply(matrix0, matrix1):
    m = np.zeros((matrix0.shape[0], 4, 4), dtype=matrix0.dtype)

    for i in prange(matrix0.shape[0]):
        m[i, 0, 0] = (
            matrix0[i, 0, 0] * matrix1[i, 0, 0]
            + matrix0[i, 0, 1] * matrix1[i, 1, 0]
            + matrix0[i, 0, 2] * matrix1[i, 2, 0]
            + matrix0[i, 0, 3] * matrix1[i, 3, 0]
        )
        m[i, 0, 1] = (
            matrix0[i, 0, 0] * matrix1[i, 0, 1]
            + matrix0[i, 0, 1] * matrix1[i, 1, 1]
            + matrix0[i, 0, 2] * matrix1[i, 2, 1]
            + matrix0[i, 0, 3] * matrix1[i, 3, 1]
        )
        m[i, 0, 2] = (
            matrix0[i, 0, 0] * matrix1[i, 0, 2]
            + matrix0[i, 0, 1] * matrix1[i, 1, 2]
            + matrix0[i, 0, 2] * matrix1[i, 2, 2]
            + matrix0[i, 0, 3] * matrix1[i, 3, 2]
        )
        m[i, 0, 3] = (
            matrix0[i, 0, 0] * matrix1[i, 0, 3]
            + matrix0[i, 0, 1] * matrix1[i, 1, 3]
            + matrix0[i, 0, 2] * matrix1[i, 2, 3]
            + matrix0[i, 0, 3] * matrix1[i, 3, 3]
        )

        m[i, 1, 0] = (
            matrix0[i, 1, 0] * matrix1[i, 0, 0]
            + matrix0[i, 1, 1] * matrix1[i, 1, 0]
            + matrix0[i, 1, 2] * matrix1[i, 2, 0]
            + matrix0[i, 1, 3] * matrix1[i, 3, 0]
        )
        m[i, 1, 1] = (
            matrix0[i, 1, 0] * matrix1[i, 0, 1]
            + matrix0[i, 1, 1] * matrix1[i, 1, 1]
            + matrix0[i, 1, 2] * matrix1[i, 2, 1]
            + matrix0[i, 1, 3] * matrix1[i, 3, 1]
        )
        m[i, 1, 2] = (
            matrix0[i, 1, 0] * matrix1[i, 0, 2]
            + matrix0[i, 1, 1] * matrix1[i, 1, 2]
            + matrix0[i, 1, 2] * matrix1[i, 2, 2]
            + matrix0[i, 1, 3] * matrix1[i, 3, 2]
        )
        m[i, 1, 3] = (
            matrix0[i, 1, 0] * matrix1[i, 0, 3]
            + matrix0[i, 1, 1] * matrix1[i, 1, 3]
            + matrix0[i, 1, 2] * matrix1[i, 2, 3]
            + matrix0[i, 1, 3] * matrix1[i, 3, 3]
        )

        m[i, 2, 0] = (
            matrix0[i, 2, 0] * matrix1[i, 0, 0]
            + matrix0[i, 2, 1] * matrix1[i, 1, 0]
            + matrix0[i, 2, 2] * matrix1[i, 2, 0]
            + matrix0[i, 2, 3] * matrix1[i, 3, 0]
        )
        m[i, 2, 1] = (
            matrix0[i, 2, 0] * matrix1[i, 0, 1]
            + matrix0[i, 2, 1] * matrix1[i, 1, 1]
            + matrix0[i, 2, 2] * matrix1[i, 2, 1]
            + matrix0[i, 2, 3] * matrix1[i, 3, 1]
        )
        m[i, 2, 2] = (
            matrix0[i, 2, 0] * matrix1[i, 0, 2]
            + matrix0[i, 2, 1] * matrix1[i, 1, 2]
            + matrix0[i, 2, 2] * matrix1[i, 2, 2]
            + matrix0[i, 2, 3] * matrix1[i, 3, 2]
        )
        m[i, 2, 3] = (
            matrix0[i, 2, 0] * matrix1[i, 0, 3]
            + matrix0[i, 2, 1] * matrix1[i, 1, 3]
            + matrix0[i, 2, 2] * matrix1[i, 2, 3]
            + matrix0[i, 2, 3] * matrix1[i, 3, 3]
        )

        m[i, 3, 0] = (
            matrix0[i, 3, 0] * matrix1[i, 0, 0]
            + matrix0[i, 3, 1] * matrix1[i, 1, 0]
            + matrix0[i, 3, 2] * matrix1[i, 2, 0]
            + matrix0[i, 3, 3] * matrix1[i, 3, 0]
        )
        m[i, 3, 1] = (
            matrix0[i, 3, 0] * matrix1[i, 0, 1]
            + matrix0[i, 3, 1] * matrix1[i, 1, 1]
            + matrix0[i, 3, 2] * matrix1[i, 2, 1]
            + matrix0[i, 3, 3] * matrix1[i, 3, 1]
        )
        m[i, 3, 2] = (
            matrix0[i, 3, 0] * matrix1[i, 0, 2]
            + matrix0[i, 3, 1] * matrix1[i, 1, 2]
            + matrix0[i, 3, 2] * matrix1[i, 2, 2]
            + matrix0[i, 3, 3] * matrix1[i, 3, 2]
        )
        m[i, 3, 3] = (
            matrix0[i, 3, 0] * matrix1[i, 0, 3]
            + matrix0[i, 3, 1] * matrix1[i, 1, 3]
            + matrix0[i, 3, 2] * matrix1[i, 2, 3]
            + matrix0[i, 3, 3] * matrix1[i, 3, 3]
        )

    return m


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_point_multiply(point, matrix):
    p = np.zeros((point.shape[0], 3), dtype=point.dtype)

    for i in prange(point.shape[0]):
        p[i, 0] = (
            (matrix[i, 0, 0] * point[i, 0])
            + (matrix[i, 1, 0] * point[i, 1])
            + (matrix[i, 2, 0] * point[i, 2])
            + matrix[i, 3, 0]
        )
        p[i, 1] = (
            (matrix[i, 0, 1] * point[i, 0])
            + (matrix[i, 1, 1] * point[i, 1])
            + (matrix[i, 2, 1] * point[i, 2])
            + matrix[i, 3, 1]
        )
        p[i, 2] = (
            (matrix[i, 0, 2] * point[i, 0])
            + (matrix[i, 1, 2] * point[i, 1])
            + (matrix[i, 2, 2] * point[i, 2])
            + matrix[i, 3, 2]
        )

    return p


@njit(fastmath=True, parallel=True, cache=True)
def _matrix_to_euler(matrix, axes):
    """Converts list of 4x4 matrices to Maya euler angles."""
    euler = np.empty((matrix.shape[0], 3), dtype=matrix.dtype)

    for ii in prange(matrix.shape[0]):
        m_ = np.empty((3, 3), dtype=matrix.dtype)
        i, j, k, h, n, s, f = _get_euler_order(axes[ii])

        # Normalize xyz axes in case of scale
        x, y, z = 0.0, 0.0, 0.0
        for jj in range(3):
            x = x + matrix[ii, 0, jj] ** 2
            y = y + matrix[ii, 1, jj] ** 2
            z = z + matrix[ii, 2, jj] ** 2

        x = x**0.5
        y = y**0.5
        z = z**0.5

        for jj in range(3):
            m_[jj, 0] = matrix[ii, 0, jj] / x
            m_[jj, 1] = matrix[ii, 1, jj] / y
            m_[jj, 2] = matrix[ii, 2, jj] / z

        if s:
            yy           = (m_[i, j] ** 2 + m_[i, k] ** 2) ** 0.5
            euler[ii, 1] = atan2(yy, m_[i, i])

            if yy > EPSILON:
                euler[ii, 0] = atan2(m_[i, j], m_[i, k])
                euler[ii, 2] = atan2(m_[j, i], -m_[k, i])
            else:
                euler[ii, 0] = atan2(-m_[j, k], m_[j, j])
                euler[ii, 2] = 0.0

        else:
            yy = (m_[i, i] ** 2 + m_[j, i] ** 2) ** 0.5

            euler[ii, 1] = atan2(-m_[k, i], yy)

            if yy > EPSILON:
                euler[ii, 0] = atan2(m_[k, j], m_[k, k])
                euler[ii, 2] = atan2(m_[j, i], m_[i, i])
            else:
                euler[ii, 0] = atan2(-m_[j, k], m_[j, j])
                euler[ii, 2] = 0.0

        if n:
            euler[ii, 0], euler[ii, 1], euler[ii, 2] = (
                -euler[ii, 0],
                -euler[ii, 1],
                -euler[ii, 2],
            )

        if f:
            euler[ii, 0], euler[ii, 2] = euler[ii, 2], euler[ii, 0]

        # From euler angle to maya
        euler[ii, 0], euler[ii, 1], euler[ii, 2] = (
            euler[ii, EA_MAYA[axes[ii], 0]],
            euler[ii, EA_MAYA[axes[ii], 1]],
            euler[ii, EA_MAYA[axes[ii], 2]],
        )

    return euler