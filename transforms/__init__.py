"""
Move all data types into the same namespace
"""

__version__ = "1.0.0"

from cgmath.transforms.main import (
    # constants
    XYZ,
    YZX,
    ZXY,
    XZY,
    YXZ,
    ZYX,
    X,
    Y,
    Z,

    # axis / angle
    axis_angle_to_euler,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,

    # euler
    euler_filter,
    euler_random,
    euler_reorder,
    euler_slerp,
    euler_to_matrix,
    euler_to_quaternion,

    # matrix
    matrix_decompose,
    matrix_delta,
    matrix_flatten,
    matrix_identity,
    matrix_interpolate,
    matrix_inverse,
    matrix_local,
    matrix_multiply,
    matrix_normalize,
    matrix_point_multiply,
    matrix_random,
    matrix_slerp,
    matrix_to_euler,
    matrix_to_quaternion,
    matrix_transpose,
    matrix_weighted_rotational,
    matrix_weighted_transformation,

    # quaternion
    quaternion_add,
    quaternion_conjugate,
    quaternion_dot,
    quaternion_exp,
    quaternion_intermediate,
    quaternion_inverse,
    quaternion_log,
    quaternion_multiply,
    quaternion_negate,
    quaternion_nlerp,
    quaternion_normalize,
    quaternion_random,
    quaternion_slerp,
    quaternion_squad,
    quaternion_sub,
    quaternion_to_euler,
    quaternion_to_matrix,

    # vector
    vector_angle,
    vector_arc_to_euler,
    vector_arc_to_matrix,
    vector_arc_to_quaternion,
    vector_cross,
    vector_dot,
    vector_lerp,
    vector_magnitude,
    vector_normalize,
    vector_random,
    vector_slerp,
    vector_to_euler,
    vector_to_matrix,
    vector_to_quaternion,
)
