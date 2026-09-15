"""Unified transforms math module.

Exposes every public vector, axis, euler, matrix, and quaternion
function under a single namespace using **Numba-style fully-qualified
names** that mirror the underlying compiled kernels.

Old API (the split ``vector`` / ``axis`` / ``euler`` / ``matrix`` /
``quaternion`` modules, dropped when this package was extracted)::

    M = quaternion.to_matrix(q)
    q = matrix.to_quaternion(M)
    P = matrix.point(p, M)
    v = vector.cross(a, b)

New API (this module)::

    from cgmath.transforms import (
        quaternion_to_matrix,
        matrix_to_quaternion,
        matrix_point_multiply,
        vector_cross,
    )
    M = quaternion_to_matrix(q)
    q = matrix_to_quaternion(M)
    P = matrix_point_multiply(p, M)
    v = vector_cross(a, b)

The split modules are not part of this package; the unified names
below are the only supported API.

All Numba kernel imports are deferred to first call -- importing this
module is cheap and does not trigger any JIT compilation.

Naming convention
-----------------

For every public function ``foo`` in module ``X``, the unified name is
``X_foo`` (e.g. ``quaternion.slerp`` -> :func:`quaternion_slerp`).
Where the original function already had a domain-style name (e.g.
``matrix_delta``, ``axis.angle_to_quaternion``), the prefix is kept.

Constants
---------

Maya rotate-order indices::

    XYZ = 0, YZX = 1, ZXY = 2, XZY = 3, YXZ = 4, ZYX = 5

XYZ axis indices::

    X = 0, Y = 1, Z = 2
"""

from __future__ import annotations

import typing

import numpy as np
from cgmath.transforms.utils import _match_depth, _set_dimension


# axes as mapped by Maya's rotate order indices
XYZ = 0
YZX = 1
ZXY = 2
XZY = 3
YXZ = 4
ZYX = 5

# XYZ axes indices
X = 0
Y = 1
Z = 2


# ===========================================================================
# QUATERNION
# ===========================================================================


def quaternion_slerp(quat0, quat1, weight=0.5, shortest=True):
    """Spherical linear interpolation between two lists of quaternions.

    Mirrors the legacy ``quaternion.slerp``.
    Backed by ``_quaternion_slerp``.

    Args:
        quat0: ``(qi, qj, qk, qw)`` or stack ``(N, 4)``. weight=0 endpoint.
        quat1: ``(qi, qj, qk, qw)`` or stack ``(N, 4)``. weight=1 endpoint.
        weight: scalar or ``(N,)`` blend weights in ``[0, 1]``.
        shortest: take the shortest arc when ``True`` (default).

    Returns:
        ``(N, 4)`` interpolated quaternions.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_slerp

    quat0  = _set_dimension(quat0,  2)
    quat1  = _set_dimension(quat1,  2)
    weight = _set_dimension(weight, 1)

    quat0, quat1, weight = _match_depth(quat0, quat1, weight)

    dot = np.einsum("...i,...i", quat0, quat1)
    if shortest:
        sign = dot < 0.0
    else:
        sign = dot >= 0.0
    # np.where rather than fancy-index assignment: _set_dimension hands back
    # the caller's own array when it is already (N, 4) float, so writing in
    # place would negate quaternions the caller still holds
    quat1 = np.where(sign[..., None], -quat1, quat1)

    return _quaternion_slerp(quat0, quat1, weight)


def quaternion_dot(quat0, quat1):
    """Dot product between two lists of quaternions.

    Mirrors the legacy ``quaternion.dot``.
    Backed by ``_vector_dot`` (quaternion dot is just 4-vector dot).

    Returns:
        ``(N,)`` floats.
    """
    from cgmath.transforms._numba._vector import _vector_dot

    quat0 = _set_dimension(quat0, 2)
    quat1 = _set_dimension(quat1, 2)
    quat0, quat1 = _match_depth(quat0, quat1)

    return _vector_dot(quat0, quat1)


def quaternion_conjugate(quat):
    """Per-quaternion conjugate.  Backed by ``_quaternion_conjugate``."""
    from cgmath.transforms._numba._quaternion import _quaternion_conjugate

    quat = _set_dimension(quat, 2)
    return _quaternion_conjugate(quat)


def quaternion_inverse(quat):
    """Per-quaternion inverse.  Backed by ``_quaternion_inverse``."""
    from cgmath.transforms._numba._quaternion import _quaternion_inverse

    quat = _set_dimension(quat, 2)
    return _quaternion_inverse(quat)


def quaternion_negate(quat):
    """Per-quaternion negation.  Backed by ``_quaternion_negate``."""
    from cgmath.transforms._numba._quaternion import _quaternion_negate

    quat = _set_dimension(quat, 2)
    return _quaternion_negate(quat)


def quaternion_multiply(quat0, quat1):
    """Hamilton product of two lists of quaternions.

    Backed by ``_quaternion_multiply``.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_multiply

    quat0 = _set_dimension(quat0, 2)
    quat1 = _set_dimension(quat1, 2)
    quat0, quat1 = _match_depth(quat0, quat1)

    return _quaternion_multiply(quat0, quat1)


def quaternion_add(quat0, quat1):
    """Component-wise quaternion addition.  Backed by ``_quaternion_add``."""
    from cgmath.transforms._numba._quaternion import _quaternion_add

    quat0 = _set_dimension(quat0, 2)
    quat1 = _set_dimension(quat1, 2)
    quat0, quat1 = _match_depth(quat0, quat1)

    return _quaternion_add(quat0, quat1)


def quaternion_sub(quat0, quat1):
    """Component-wise quaternion subtraction.  Backed by ``_quaternion_sub``."""
    from cgmath.transforms._numba._quaternion import _quaternion_sub

    quat0 = _set_dimension(quat0, 2)
    quat1 = _set_dimension(quat1, 2)
    quat0, quat1 = _match_depth(quat0, quat1)

    return _quaternion_sub(quat0, quat1)


def quaternion_to_matrix(quat):
    """Convert quaternions to 4x4 rotation matrices.

    Backed by ``_quaternion_to_matrix``.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_to_matrix

    quat = _set_dimension(quat, 2)
    return _quaternion_to_matrix(quat)


def quaternion_normalize(quat):
    """Normalise quaternions to unit length.

    Backed by ``_vector_normalize`` (treats the 4-tuple as a vector).
    """
    from cgmath.transforms._numba._vector import _vector_normalize

    quat = _set_dimension(quat, 2)
    return _vector_normalize(quat)


def quaternion_to_euler(quat, axes=XYZ):
    """Convert quaternions to euler angles for a given rotate order.

    Backed by ``_quaternion_to_matrix`` -> ``_matrix_to_euler``.

    Args:
        quat: ``(N, 4)`` quaternions.
        axes: rotate-order index (``XYZ`` .. ``ZYX``) or per-quaternion list.

    Returns:
        ``(N, 3)`` euler angles (radians).
    """
    from cgmath.transforms._numba._matrix import _matrix_to_euler
    from cgmath.transforms._numba._quaternion import _quaternion_to_matrix

    quat = _set_dimension(quat, 2)
    axes = _set_dimension(axes, 1, dtype=np.int32)
    quat, axes = _match_depth(quat, axes)

    return _matrix_to_euler(_quaternion_to_matrix(quat), axes)


def quaternion_random(n, seed=None):
    """Generate ``n`` random unit quaternions.

    Backed by ``_euler_to_quaternion`` with random euler input.
    """
    from cgmath.transforms._numba._euler import _euler_to_quaternion

    np.random.seed(seed)
    eu = np.radians(360 - np.random.random((n, 3)) * 720)
    return _euler_to_quaternion(eu, np.zeros(n, dtype="int32"))


def quaternion_log(quat):
    """Logarithm of unit quaternions -> ``(N, 3)`` rotation vectors.

    The rotation vector is ``axis * angle`` (the Lie-algebra tangent at
    identity). Inverse of :func:`quaternion_exp`. Backed by ``_quaternion_log``.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_log

    quat = _set_dimension(quat, 2)
    return _quaternion_log(quat)


def quaternion_exp(rotation_vector):
    """Exponential of ``(N, 3)`` rotation vectors -> ``(N, 4)`` unit quaternions.

    Inverse of :func:`quaternion_log`. Backed by ``_quaternion_exp``.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_exp

    rotation_vector = _set_dimension(rotation_vector, 2)
    return _quaternion_exp(rotation_vector)


def quaternion_nlerp(quat0, quat1, weight=0.5, shortest=True):
    """Normalised linear interpolation between two lists of quaternions.

    A cheap, small-angle-accurate alternative to :func:`quaternion_slerp`
    (does not preserve constant angular velocity). Backed by ``_quaternion_nlerp``.

    Args:
        quat0: ``(qi, qj, qk, qw)`` or stack ``(N, 4)``. weight=0 endpoint.
        quat1: ``(qi, qj, qk, qw)`` or stack ``(N, 4)``. weight=1 endpoint.
        weight: scalar or ``(N,)`` blend weights in ``[0, 1]``.
        shortest: take the shortest arc when ``True`` (default).

    Returns:
        ``(N, 4)`` unit quaternions.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_nlerp

    quat0  = _set_dimension(quat0,  2)
    quat1  = _set_dimension(quat1,  2)
    weight = _set_dimension(weight, 1)
    quat0, quat1, weight = _match_depth(quat0, quat1, weight)

    dot = np.einsum("...i,...i", quat0, quat1)
    if shortest:
        sign = dot < 0.0
    else:
        sign = dot >= 0.0
    # see quaternion_slerp: never write into the caller's array
    quat1 = np.where(sign[..., None], -quat1, quat1)

    return _quaternion_nlerp(quat0, quat1, weight)


def quaternion_intermediate(quat_prev, quat_cur, quat_next):
    """Squad inner-quadrangle control quaternions ``s_i`` from three keys.

    ``s = q_cur * exp(-(log(q_cur^-1 * q_next) + log(q_cur^-1 * q_prev)) / 4)``.
    The relative rotations are hemisphere-canonicalised (shortest arc) before
    the log so the control point follows the intended path. All inputs are
    ``(N, 4)`` unit quaternions. Used to feed :func:`quaternion_squad`.
    """
    from cgmath.transforms._numba._quaternion import (
        _quaternion_conjugate,
        _quaternion_exp,
        _quaternion_log,
        _quaternion_multiply,
    )

    quat_prev = _set_dimension(quat_prev, 2)
    quat_cur  = _set_dimension(quat_cur,  2)
    quat_next = _set_dimension(quat_next, 2)
    quat_prev, quat_cur, quat_next = _match_depth(quat_prev, quat_cur, quat_next)

    cur_inv  = _quaternion_conjugate(quat_cur)  # unit quats: conjugate == inverse
    rel_next = _quaternion_multiply(cur_inv, quat_next)
    rel_prev = _quaternion_multiply(cur_inv, quat_prev)

    # shortest-arc canonicalisation so the log takes the intended branch
    rel_next[rel_next[:, 3] < 0.0] = -rel_next[rel_next[:, 3] < 0.0]
    rel_prev[rel_prev[:, 3] < 0.0] = -rel_prev[rel_prev[:, 3] < 0.0]

    inner = _quaternion_exp(
        -(_quaternion_log(rel_next) + _quaternion_log(rel_prev)) / 4.0
    )
    return _quaternion_multiply(quat_cur, inner)


def quaternion_squad(quat0, control0, control1, quat1, weight=0.5):
    """Cubic (C1-continuous) quaternion spline segment (spherical-and-quadrangle).

    ``squad = slerp(slerp(q0, q1, t), slerp(s0, s1, t), 2t(1-t))`` where
    ``control0``/``control1`` are the :func:`quaternion_intermediate` control
    quaternions of the segment endpoints. Reduces to :func:`quaternion_slerp`
    when the controls lie on the geodesic. Backed by :func:`quaternion_slerp`.

    Args:
        quat0: ``(N, 4)`` weight=0 endpoint.
        control0: ``(N, 4)`` control quaternion at ``quat0``.
        control1: ``(N, 4)`` control quaternion at ``quat1``.
        quat1: ``(N, 4)`` weight=1 endpoint.
        weight: scalar or ``(N,)`` blend weights in ``[0, 1]``.

    Returns:
        ``(N, 4)`` unit quaternions.
    """
    quat0    = _set_dimension(quat0,    2)
    control0 = _set_dimension(control0, 2)
    control1 = _set_dimension(control1, 2)
    quat1    = _set_dimension(quat1,    2)
    weight   = _set_dimension(weight,   1)
    quat0, control0, control1, quat1, weight = _match_depth(
        quat0, control0, control1, quat1, weight
    )

    base  = quaternion_slerp(quat0, quat1, weight)
    ctrl  = quaternion_slerp(control0, control1, weight)
    blend = 2.0 * weight * (1.0 - weight)
    return quaternion_slerp(base, ctrl, blend)


# ===========================================================================
# MATRIX
# ===========================================================================


def matrix_identity(count):
    """Build a stack of identity 4x4 matrices.  Backed by ``_matrix_identity``."""
    from cgmath.transforms._numba._matrix import _matrix_identity

    return _matrix_identity(count)


def matrix_to_euler(matrix, axes=XYZ):
    """Convert 4x4 transform matrices to euler angles.

    Backed by ``_matrix_to_euler``.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_euler

    matrix = _set_dimension(matrix, 3, reshape_matrix=True)
    axes   = _set_dimension(axes, 1, dtype=np.int32)
    matrix, axes = _match_depth(matrix, axes)

    return _matrix_to_euler(matrix, axes)


def matrix_to_quaternion(matrix):
    """Convert 4x4 transform matrices to quaternions.

    Backed by ``_matrix_to_quaternion``.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_quaternion

    matrix = _set_dimension(matrix, 3, reshape_matrix=True)
    return _matrix_to_quaternion(matrix)


def matrix_normalize(matrix):
    """Re-orthonormalise the rotation block of transform matrices.

    Backed by ``_matrix_normalize``.
    """
    from cgmath.transforms._numba._matrix import _matrix_normalize

    matrix = _set_dimension(matrix, 3, reshape_matrix=True)
    return _matrix_normalize(matrix)


def matrix_inverse(matrix):
    """Invert a list of 4x4 transform matrices.  Backed by ``_matrix_inverse``."""
    from cgmath.transforms._numba._matrix import _matrix_inverse

    matrix = _set_dimension(matrix, 3, reshape_matrix=True)
    return _matrix_inverse(matrix)


def matrix_transpose(matrix):
    """Transpose a list of 4x4 transform matrices.  Backed by ``_matrix_transpose``."""
    from cgmath.transforms._numba._matrix import _matrix_transpose

    matrix = _set_dimension(matrix, 3, reshape_matrix=True)
    return _matrix_transpose(matrix)


def matrix_point_multiply(point, matrix):
    """Transform points by 4x4 matrices.  ``P * M`` row-vector convention.

    Backed by ``_matrix_point_multiply``.

    Args:
        point: ``(N, 3)`` or single ``(3,)`` point(s).
        matrix: ``(N, 4, 4)`` or single ``(4, 4)`` matrix.

    Returns:
        ``(N, 3)`` transformed points.
    """
    from cgmath.transforms._numba._matrix import _matrix_point_multiply

    point  = _set_dimension(point, 2)
    matrix = _set_dimension(matrix, 3, reshape_matrix=True)
    point, matrix = _match_depth(point, matrix)

    return _matrix_point_multiply(point[:, :3], matrix)


def matrix_multiply(matrix0, matrix1):
    """Multiply pairs of 4x4 transform matrices.

    Backed by ``_matrix_multiply``.
    """
    from cgmath.transforms._numba._matrix import _matrix_multiply

    matrix0 = _set_dimension(matrix0, 3, reshape_matrix=True)
    matrix1 = _set_dimension(matrix1, 3, reshape_matrix=True)
    matrix0, matrix1 = _match_depth(matrix0, matrix1)

    return _matrix_multiply(matrix0, matrix1)


def matrix_slerp(matrix0, matrix1, weight=0.5, shortest=True):
    """Spherical interpolation of rotation between two lists of matrices.

    Translation is **ignored**.  Backed by ``_matrix_to_quaternion`` ->
    ``_quaternion_slerp`` -> ``_quaternion_to_matrix``.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_quaternion
    from cgmath.transforms._numba._quaternion import (
        _quaternion_slerp,
        _quaternion_to_matrix,
    )

    matrix0 = _set_dimension(matrix0, 3, reshape_matrix=True)
    matrix1 = _set_dimension(matrix1, 3, reshape_matrix=True)
    weight  = _set_dimension(weight, 1)
    matrix0, matrix1, weight = _match_depth(matrix0, matrix1, weight)

    q0  = _matrix_to_quaternion(matrix0)
    q1  = _matrix_to_quaternion(matrix1)

    dot = np.einsum("...i,...i", q0, q1)
    if shortest:
        sign = dot < 0.0
    else:
        sign = dot >= 0.0
    q1[sign] = -q1[sign]

    q = _quaternion_slerp(q0, q1, weight)
    return _quaternion_to_matrix(q)


def matrix_interpolate(matrix0, matrix1, weight=0.5, shortest=True):
    """Decomposed SRT interpolation between two lists of transform matrices.

    Scale: lerp.  Rotation: slerp (shortest-arc by default).  Translation: lerp.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_quaternion
    from cgmath.transforms._numba._quaternion import (
        _quaternion_slerp,
        _quaternion_to_matrix,
    )
    from cgmath.transforms._numba._vector import _vector_lerp

    matrix0 = _set_dimension(matrix0, 3, reshape_matrix=True)
    matrix1 = _set_dimension(matrix1, 3, reshape_matrix=True)
    weight  = _set_dimension(weight, 1)
    matrix0, matrix1, weight = _match_depth(matrix0, matrix1, weight)

    # the scale is divided out in place below
    matrix0 = matrix0.copy()
    matrix1 = matrix1.copy()

    scale0  = np.einsum("...i,...i", matrix0[:, :3, :3], matrix0[:, :3, :3]) ** 0.5
    scale1  = np.einsum("...i,...i", matrix1[:, :3, :3], matrix1[:, :3, :3]) ** 0.5

    matrix0[:, 0, :3] /= scale0[:, 0][:, None]
    matrix0[:, 1, :3] /= scale0[:, 1][:, None]
    matrix0[:, 2, :3] /= scale0[:, 2][:, None]

    matrix1[:, 0, :3] /= scale1[:, 0][:, None]
    matrix1[:, 1, :3] /= scale1[:, 1][:, None]
    matrix1[:, 2, :3] /= scale1[:, 2][:, None]

    q0  = _matrix_to_quaternion(matrix0)
    q1  = _matrix_to_quaternion(matrix1)

    dot = np.einsum("...i,...i", q0, q1)
    if shortest:
        sign = dot < 0.0
    else:
        sign = dot >= 0.0
    q1[sign] = -q1[sign]

    matrix   = _quaternion_to_matrix(_quaternion_slerp(q0, q1, weight))

    scale = _vector_lerp(scale0, scale1, weight)
    matrix[:, 0, :3] *= scale[:, 0][:, None]
    matrix[:, 1, :3] *= scale[:, 1][:, None]
    matrix[:, 2, :3] *= scale[:, 2][:, None]

    matrix[:, 3, :3] = _vector_lerp(matrix0[:, 3, :3], matrix1[:, 3, :3], weight)

    return matrix


def matrix_local(matrix, parent_matrix):
    """Return matrix expressed in the local space of ``parent_matrix``.

    Backed by ``_matrix_multiply`` of ``matrix`` and ``_matrix_inverse(parent)``.
    """
    from cgmath.transforms._numba._matrix import _matrix_inverse, _matrix_multiply

    matrix        = _set_dimension(matrix, 3, reshape_matrix=True)
    parent_matrix = _set_dimension(parent_matrix, 3, reshape_matrix=True)
    matrix, parent_matrix = _match_depth(matrix, parent_matrix)

    return _matrix_multiply(matrix, _matrix_inverse(parent_matrix))


def matrix_random(n, seed=None, random_position=False):
    """Generate ``n`` random 4x4 rotation matrices.

    Backed by ``_euler_to_matrix`` with random euler input.
    """
    from cgmath.transforms._numba._euler import _euler_to_matrix

    np.random.seed(seed)
    euler = np.radians(360 - np.random.random((n, 3)) * 720)
    M     = _euler_to_matrix(euler, np.zeros(euler.shape[0], dtype=np.int32))
    if random_position:
        M[:, 3, :3] = 1 - np.random.random((n, 3)) * 2
    return M


def matrix_flatten(matrix: typing.List[typing.List[float]]) -> typing.List[float]:
    """Flatten a 4x4 matrix to a Maya-style flat 16-element list.

    Pure Python helper (no Numba backing).
    """
    if len(matrix) == 16:
        return matrix
    return [item for sublist in matrix for item in sublist]


def matrix_decompose(
    matrix: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4x4 matrix into ``(translation, rotation, scale)`` 4x4s.

    Pure-NumPy SVD; no Numba backing.  Mirrors
    the legacy ``matrix.decompose_matrix_as_matrices``.
    """
    translation    = matrix[3, :3]
    rotation_scale = matrix[:3, :3]
    U, S, Vt = np.linalg.svd(rotation_scale)
    rotation_matrix              = np.dot(U, Vt)
    scale_matrix                 = np.diag(S)
    translation_matrix           = np.eye(4)
    translation_matrix[3, :3]    = translation
    full_rotation_matrix         = np.eye(4)
    full_rotation_matrix[:3, :3] = rotation_matrix
    full_scale_matrix            = np.eye(4)
    full_scale_matrix[:3, :3]    = scale_matrix
    return translation_matrix, full_rotation_matrix, full_scale_matrix


def matrix_weighted_rotational(
    matrices: typing.List[np.ndarray], weights: typing.List[float]
) -> np.ndarray:
    """Weighted average of 3x3 rotation matrices via scipy ``Rotation``.

    Mirrors the legacy ``matrix.compute_weighed_rotational_matrices``.
    """
    from scipy.spatial.transform import Rotation

    if len(matrices) != len(weights):
        raise ValueError(
            f"length mismatch: got {[len(matrices), len(weights)]}; "
            f"matrices and weights must be the same length"
        )

    weighted_sum = np.sum(
        [matrix * weight for matrix, weight in zip(matrices, weights)], axis=0
    )
    resultant_rotation = Rotation.from_matrix(weighted_sum)
    return resultant_rotation.as_matrix()


def matrix_weighted_transformation(
    transformation_matrices: typing.List[np.ndarray],
    weights:                 typing.List[float],
    flatten:                 bool                    = False,
) -> np.ndarray:
    """Compute a weighted average of 4x4 transformation matrices.

    Mirrors the legacy ``matrix.compute_weighted_transformation``.
    """
    transformation_matrices = np.array(transformation_matrices)
    weights                 = np.array(weights)

    if len(transformation_matrices) != len(weights):
        raise ValueError(
            f"length mismatch: got "
            f"{[len(transformation_matrices), len(weights)]}; "
            f"matrices and weights must be the same length"
        )

    weights_normalized   = weights / np.sum(weights)

    weighted_translation = np.zeros((1, 3))
    rotation_matrices    = []
    weighted_scale       = np.zeros((3, 3))

    for i, matrix in enumerate(transformation_matrices):
        translation_matrix, rotation_matrix, scale_matrix = matrix_decompose(matrix)
        rotation_matrices.append(rotation_matrix[:3, :3])
        weight = weights_normalized[i]
        weighted_translation += weight * np.array(translation_matrix[3, :3])
        weighted_scale       += weight * np.array(scale_matrix[:3, :3])

    weighted_rotation = matrix_weighted_rotational(
        rotation_matrices, weights_normalized
    )
    scale_matrix          = np.eye(4)
    result_matrix         = np.eye(4)
    result_matrix[:3, :3] = weighted_rotation
    scale_matrix[:3, :3]  = weighted_scale
    result_matrix[:3, :3] = np.dot(weighted_rotation, weighted_scale)
    result_matrix[3, :3]  = weighted_translation

    if flatten:
        result_matrix = matrix_flatten(result_matrix)

    return result_matrix


def matrix_delta(
    parent_matrix: typing.List[float], child_matrix: typing.List[float]
) -> typing.List[float]:
    """Offset transform between a parent and child matrix.

    Mirrors the legacy ``matrix.matrix_delta``.
    """
    parent_inverse = matrix_inverse(parent_matrix)
    return matrix_multiply(child_matrix, parent_inverse)


# ===========================================================================
# EULER
# ===========================================================================


def euler_to_matrix(euler, axes=XYZ):
    """Convert euler angles to 4x4 transform matrices.

    Backed by ``_euler_to_matrix``.
    """
    from cgmath.transforms._numba._euler import _euler_to_matrix

    euler = _set_dimension(euler, 2)
    axes  = _set_dimension(axes, 1, dtype=np.int32)
    euler, axes = _match_depth(euler, axes)

    return _euler_to_matrix(euler, axes)


def euler_to_quaternion(euler, axes=XYZ):
    """Convert euler angles to quaternions.

    Backed by ``_euler_to_quaternion``.
    """
    from cgmath.transforms._numba._euler import _euler_to_quaternion

    euler = _set_dimension(euler, 2)
    axes  = _set_dimension(axes, 1, dtype=np.int32)
    euler, axes = _match_depth(euler, axes)

    return _euler_to_quaternion(euler, axes)


def euler_slerp(euler0, euler1, weight=0.5, axes0=XYZ, axes1=XYZ, axes=XYZ):
    """Spherical interpolation between two lists of euler angles.

    Backed by ``_euler_to_quaternion`` -> ``_quaternion_slerp`` ->
    ``_quaternion_to_matrix`` -> ``_matrix_to_euler``.
    """
    from cgmath.transforms._numba._euler import _euler_to_quaternion
    from cgmath.transforms._numba._matrix import _matrix_to_euler
    from cgmath.transforms._numba._quaternion import (
        _quaternion_slerp,
        _quaternion_to_matrix,
    )

    euler0 = _set_dimension(euler0, 2)
    euler1 = _set_dimension(euler1, 2)
    weight = _set_dimension(weight, 1)
    axes0  = _set_dimension(axes0, 1, dtype=np.int32)
    axes1  = _set_dimension(axes1, 1, dtype=np.int32)
    axes   = _set_dimension(axes,  1, dtype=np.int32)

    euler0, euler1, weight, axes0, axes1, axes = _match_depth(
        euler0, euler1, weight, axes0, axes1, axes
    )

    q0 = _euler_to_quaternion(euler0, axes0)
    q1 = _euler_to_quaternion(euler1, axes1)
    q  = _quaternion_slerp(q0, q1, weight)

    return _matrix_to_euler(_quaternion_to_matrix(q), axes)


def euler_reorder(euler, from_axes, to_axes):
    """Convert euler angles from one rotate order to another.

    Mirrors the legacy ``euler.to_euler`` (renamed to make
    the intent unambiguous in the unified namespace).
    Backed by ``_euler_to_matrix`` -> ``_matrix_to_euler``.
    """
    from cgmath.transforms._numba._euler import _euler_to_matrix
    from cgmath.transforms._numba._matrix import _matrix_to_euler

    euler     = _set_dimension(euler, 2)
    from_axes = _set_dimension(from_axes, 1, dtype=np.int32)
    to_axes   = _set_dimension(to_axes, 1, dtype=np.int32)
    euler, from_axes, to_axes = _match_depth(euler, from_axes, to_axes)

    M = _euler_to_matrix(euler, from_axes)
    return _matrix_to_euler(M, to_axes)


def euler_filter(euler, axes):
    """Remove branch jumps from euler angles sampled over time.

    Frames run along the first axis. Every frame is replaced by whichever
    representation of the same rotation sits closest to the frame before
    it, so the poses are untouched while the curves stop stepping by 180
    or 360 degrees.

    ``axes`` is required rather than defaulting: the flip identity depends
    on the rotate order, so the wrong one would quietly stop preserving
    the pose instead of producing a differently valid answer.

    Motion faster than half a turn per frame cannot be recovered, being
    indistinguishable from the same motion running backwards.
    """
    from cgmath.transforms._numba._euler import MAYA_EA

    euler = _set_dimension(euler, 2)
    if euler.shape[-1] != 3:
        raise ValueError(f"expected a trailing axis of 3, got {euler.shape[-1]}")

    axes = _set_dimension(axes, 1, dtype=np.intp)
    if axes.size and (axes.min() < 0 or axes.max() >= len(MAYA_EA)):
        raise ValueError(f"axes must be in 0..{len(MAYA_EA) - 1}")

    shape  = euler.shape
    frames = shape[0]
    if frames < 2 or euler.size == 0:
        return euler.copy()

    curves = euler.reshape(frames, -1, 3)
    count  = curves.shape[1]

    # the axis the flip reflects rather than turns is the middle one of the
    # rotate order, which is what MAYA_EA already stores
    try:
        per_curve = np.broadcast_to(axes, (count,))
    except ValueError:
        raise ValueError(
            f"length mismatch: got {[count, len(axes)]}; every input must "
            f"be length {count} or 1"
        ) from None

    middle = MAYA_EA[per_curve][:, 1]
    rows   = np.arange(count)

    out = curves.copy()
    for f in range(1, frames):
        previous = out[f - 1]
        current  = curves[f]

        flipped               = current + np.pi
        flipped[rows, middle] = np.pi - current[rows, middle]

        # both candidates slide onto the turn nearest the previous frame,
        # then the shorter move wins. summing the axes rather than squaring
        # them is what maya's filterCurve does, and the two disagree once
        # the pose moves more than about 30 degrees in a frame
        candidates = np.stack((current, flipped))
        candidates += 2.0 * np.pi * np.round((previous - candidates) / (2.0 * np.pi))
        cost   = np.abs(candidates - previous).sum(axis=-1)
        out[f] = candidates[np.argmin(cost, axis=0), rows]

    return out.reshape(shape)


def euler_random(n, seed=None):
    """Generate ``n`` random euler angles in radians."""
    np.random.seed(seed)
    return np.radians(360 - np.random.random((n, 3)) * 720)


# ===========================================================================
# AXIS / ANGLE
# ===========================================================================


def axis_angle_to_quaternion(axis, angle=0.0):
    """Convert axis-angle pairs to quaternions.  Backed by ``_axis_angle_to_quaternion``."""
    from cgmath.transforms._numba._axis import _axis_angle_to_quaternion

    axis  = _set_dimension(axis, 2)
    angle = _set_dimension(angle, 1)
    axis, angle = _match_depth(axis, angle)

    return _axis_angle_to_quaternion(axis, angle)


def axis_angle_to_matrix(axis, angle=0.0):
    """Convert axis-angle pairs to 4x4 rotation matrices.

    Backed by ``_axis_angle_to_matrix``.
    """
    from cgmath.transforms._numba._axis import _axis_angle_to_matrix

    axis  = _set_dimension(axis, 2)
    angle = _set_dimension(angle, 1)
    axis, angle = _match_depth(axis, angle)

    return _axis_angle_to_matrix(axis, angle)


def axis_angle_to_euler(axis, angle=0.0, axes=XYZ):
    """Convert axis-angle pairs to euler angles.

    Backed by ``_axis_angle_to_matrix`` -> ``_matrix_to_euler``.
    """
    from cgmath.transforms._numba._axis import _axis_angle_to_matrix
    from cgmath.transforms._numba._matrix import _matrix_to_euler

    axis  = _set_dimension(axis, 2)
    angle = _set_dimension(angle, 1)
    axes  = _set_dimension(axes, 1, dtype=np.int32)
    axis, angle, axes = _match_depth(axis, angle, axes)

    M = _axis_angle_to_matrix(axis, angle)
    return _matrix_to_euler(M, axes)


# ===========================================================================
# VECTOR
# ===========================================================================


def vector_to_matrix(vector0, vector1, aim_axis=X, up_axis=Y):
    """Build 4x4 rotation matrices from aim + up vector pairs.

    Backed by ``_vector_to_matrix``.
    """
    from cgmath.transforms._numba._vector import _vector_to_matrix

    vector0  = _set_dimension(vector0, 2)
    vector1  = _set_dimension(vector1, 2)
    aim_axis = _set_dimension(aim_axis, 1, dtype=np.int32) % 3
    up_axis  = _set_dimension(up_axis, 1, dtype=np.int32) % 3
    vector0, vector1, aim_axis, up_axis = _match_depth(
        vector0, vector1, aim_axis, up_axis
    )

    return _vector_to_matrix(vector0, vector1, aim_axis, up_axis)


def vector_to_quaternion(vector0, vector1, aim_axis=X, up_axis=Y):
    """Build quaternions from aim + up vector pairs.

    Backed by ``_vector_to_matrix`` -> ``_matrix_to_quaternion``.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_quaternion
    from cgmath.transforms._numba._vector import _vector_to_matrix

    vector0  = _set_dimension(vector0, 2)
    vector1  = _set_dimension(vector1, 2)
    aim_axis = _set_dimension(aim_axis, 1, dtype=np.int32) % 3
    up_axis  = _set_dimension(up_axis, 1, dtype=np.int32) % 3
    vector0, vector1, aim_axis, up_axis = _match_depth(
        vector0, vector1, aim_axis, up_axis
    )

    M = _vector_to_matrix(vector0, vector1, aim_axis, up_axis)
    return _matrix_to_quaternion(M)


def vector_to_euler(vector0, vector1, aim_axis=X, up_axis=Y, axes=XYZ):
    """Build euler angles from aim + up vector pairs.

    Backed by ``_vector_to_matrix`` -> ``_matrix_to_euler``.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_euler
    from cgmath.transforms._numba._vector import _vector_to_matrix

    vector0  = _set_dimension(vector0, 2)
    vector1  = _set_dimension(vector1, 2)
    aim_axis = _set_dimension(aim_axis, 1, dtype=np.int32) % 3
    up_axis  = _set_dimension(up_axis, 1, dtype=np.int32) % 3
    axes     = _set_dimension(axes, 1, dtype=np.int32)
    vector0, vector1, aim_axis, up_axis, axes = _match_depth(
        vector0, vector1, aim_axis, up_axis, axes
    )

    return _matrix_to_euler(
        _vector_to_matrix(vector0, vector1, aim_axis, up_axis), axes
    )


def vector_cross(vector0, vector1):
    """Cross product of two lists of vectors.  Backed by ``_vector_cross``."""
    from cgmath.transforms._numba._vector import _vector_cross

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    vector0, vector1 = _match_depth(vector0, vector1)

    return _vector_cross(vector0, vector1)


def vector_dot(vector0, vector1):
    """Dot product of two lists of vectors.  Backed by ``_vector_dot``."""
    from cgmath.transforms._numba._vector import _vector_dot

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    vector0, vector1 = _match_depth(vector0, vector1)

    return _vector_dot(vector0, vector1)


def vector_magnitude(vector):
    """Magnitude (L2 norm) of vectors.  Backed by ``_vector_magnitude``."""
    from cgmath.transforms._numba._vector import _vector_magnitude

    vector = _set_dimension(vector, 2)
    return _vector_magnitude(vector)


def vector_normalize(vector):
    """Normalise vectors to unit length.  Backed by ``_vector_normalize``."""
    from cgmath.transforms._numba._vector import _vector_normalize

    vector = _set_dimension(vector, 2)
    return _vector_normalize(vector)


def vector_slerp(vector0, vector1, weight=0.5):
    """Spherical interpolation between two lists of vectors.

    Backed by ``_vector_slerp``.
    """
    from cgmath.transforms._numba._vector import _vector_slerp

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    weight  = _set_dimension(weight,  1)
    vector0, vector1, weight = _match_depth(vector0, vector1, weight)

    return _vector_slerp(vector0, vector1, weight)


def vector_lerp(vector0, vector1, weight=0.5):
    """Linear interpolation between two lists of vectors.

    Backed by ``_vector_lerp``.  NaNs are mapped to 0.
    """
    from cgmath.transforms._numba._vector import _vector_lerp

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    weight  = _set_dimension(weight,  1)
    vector0, vector1, weight = _match_depth(vector0, vector1, weight)

    return np.nan_to_num(_vector_lerp(vector0, vector1, weight))


def vector_arc_to_quaternion(vector0, vector1):
    """Shortest-arc rotation between two lists of vectors as quaternions.

    Backed by ``_vector_arc_to_quaternion``.
    """
    from cgmath.transforms._numba._vector import _vector_arc_to_quaternion

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    vector0, vector1 = _match_depth(vector0, vector1)

    return _vector_arc_to_quaternion(vector0, vector1)


def vector_arc_to_matrix(vector0, vector1):
    """Shortest-arc rotation between two lists of vectors as 4x4 matrices.

    Backed by ``_vector_arc_to_quaternion`` -> ``_quaternion_to_matrix``.
    """
    from cgmath.transforms._numba._quaternion import _quaternion_to_matrix
    from cgmath.transforms._numba._vector import _vector_arc_to_quaternion

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    vector0, vector1 = _match_depth(vector0, vector1)

    return _quaternion_to_matrix(_vector_arc_to_quaternion(vector0, vector1))


def vector_arc_to_euler(vector0, vector1, axes=XYZ):
    """Shortest-arc rotation between two lists of vectors as euler angles.

    Backed by ``_vector_arc_to_quaternion`` -> ``_quaternion_to_matrix``
    -> ``_matrix_to_euler``.
    """
    from cgmath.transforms._numba._matrix import _matrix_to_euler
    from cgmath.transforms._numba._quaternion import _quaternion_to_matrix
    from cgmath.transforms._numba._vector import _vector_arc_to_quaternion

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    axes    = _set_dimension(axes, 1, dtype=np.int32)
    vector0, vector1, axes = _match_depth(vector0, vector1, axes)

    return _matrix_to_euler(
        _quaternion_to_matrix(_vector_arc_to_quaternion(vector0, vector1)), axes
    )


def vector_angle(vector0, vector1):
    """Arc angle (radians) between two lists of vectors.

    Backed by ``_vector_arc``.
    """
    from cgmath.transforms._numba._vector import _vector_arc

    vector0 = _set_dimension(vector0, 2)
    vector1 = _set_dimension(vector1, 2)
    vector0, vector1 = _match_depth(vector0, vector1)

    return _vector_arc(vector0, vector1)


def vector_random(n, seed=None, normalize=False):
    """Generate ``n`` random vectors in ``[-1, 1]^3``.

    When ``normalize=True`` returns unit-length vectors via
    ``_vector_normalize``.
    """
    from cgmath.transforms._numba._vector import _vector_normalize

    if seed is not None:
        np.random.seed(seed)
    if normalize:
        return _vector_normalize(1 - np.random.random((n, 3)) * 2)
    return 1 - np.random.random((n, 3)) * 2


# ===========================================================================
# Public re-export list
# ===========================================================================

__all__ = [
    # constants
    "XYZ",
    "YZX",
    "ZXY",
    "XZY",
    "YXZ",
    "ZYX",
    "X",
    "Y",
    "Z",
    # quaternion
    "quaternion_slerp",
    "quaternion_dot",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_negate",
    "quaternion_multiply",
    "quaternion_add",
    "quaternion_sub",
    "quaternion_to_matrix",
    "quaternion_normalize",
    "quaternion_to_euler",
    "quaternion_random",
    "quaternion_log",
    "quaternion_exp",
    "quaternion_nlerp",
    "quaternion_intermediate",
    "quaternion_squad",
    # matrix
    "matrix_identity",
    "matrix_to_euler",
    "matrix_to_quaternion",
    "matrix_normalize",
    "matrix_inverse",
    "matrix_transpose",
    "matrix_point_multiply",
    "matrix_multiply",
    "matrix_slerp",
    "matrix_interpolate",
    "matrix_local",
    "matrix_random",
    "matrix_flatten",
    "matrix_decompose",
    "matrix_weighted_rotational",
    "matrix_weighted_transformation",
    "matrix_delta",
    # euler
    "euler_to_matrix",
    "euler_to_quaternion",
    "euler_slerp",
    "euler_reorder",
    "euler_random",
    "euler_filter",
    # axis / angle
    "axis_angle_to_quaternion",
    "axis_angle_to_matrix",
    "axis_angle_to_euler",
    # vector
    "vector_to_matrix",
    "vector_to_quaternion",
    "vector_to_euler",
    "vector_cross",
    "vector_dot",
    "vector_magnitude",
    "vector_normalize",
    "vector_slerp",
    "vector_lerp",
    "vector_arc_to_quaternion",
    "vector_arc_to_matrix",
    "vector_arc_to_euler",
    "vector_angle",
    "vector_random",
]