"""What Maya 2025's ``xform -rotateOrder <order> -preserve`` did to a joint,
for the set_rotate_order tests.

Every ``reorder_*`` row of ``test_assets/maya_euler.npz``, which
tools/make_maya_euler_fixture.py recorded and _tests/test_euler_maya.py
checks Maya's rule against: 85 rotations through all 30 pairs of rotate
orders, 2550 joints. The first 1200 are 40 rotations, random or picked by
hand, on a joint orient of (20, -30, 45); the other 1350 are edge rotations
(zero, axes at +-90, +-180 and 720, the middle axis 1e-7 from 90, whole
turns) on a random joint orient each.
"""
import numpy as np
from cgmath._tests.test_euler_maya import (
    angle_error,
    load_fixture,
    maya_rotate_order_change,
)
from cgmath.transforms._numba._euler import MAYA_EA


def maya_rotate_order_cases() -> dict:
    """dict of arrays, one row per joint

    rotate (N, 3), before (N,), after (N,): the rotate set, the rotate order
    it was set in and the one xform changed it to. maya (N, 3): rotate as
    Maya read it afterwards, with one change: where Maya wrote +180, or
    3e-14 short of it, and its rule writes the same angle as -180 (83
    joints, 2 of them the short ones), or Maya stopped 3e-14 short of the
    rule's -180 (2 joints), it is written as the rule writes it: 85 joints,
    so a curve built from these rows compares number for number. Maya never
    wrote -180 where the rule writes +180. joint_orient (N, 3),
    world_before (N, 4, 4) and world_after (N, 4, 4) as Maya read them.

    near_gimbal (N,): the 120 joints whose middle axis starts 1e-7 from
    +-90, where Maya's world matrix moves by up to 4.3e-9 through the
    change. twice (N,): the 8 of them whose new middle axis lands 1e-7 from
    +-90 again, where the new rotate holds the pose to 3.1e-9, in Maya and
    in euler_reorder alike.
    """
    data   = load_fixture()
    rotate = data["reorder_rotate"]
    before = data["reorder_from"].astype(int)
    after  = data["reorder_to"].astype(int)
    maya   = data["reorder_maya"]

    rule   = maya_rotate_order_change(rotate, before, after)
    half   = np.isclose(np.abs(maya), 180.0, rtol=0.0, atol=1e-9)
    same   = angle_error(maya, rule) < 1e-9
    rows   = np.arange(len(rotate))
    start  = np.abs(np.abs(rotate[rows, MAYA_EA[before][:, 1]]) - 90.0)
    end    = np.abs(np.abs(maya[rows, MAYA_EA[after][:, 1]]) - 90.0)

    near_gimbal = (start > 0.0) & (start < 1e-6)
    return {
        "rotate":       rotate,
        "before":       before,
        "after":        after,
        "maya":         np.where(half & same, rule, maya),
        "joint_orient": data["reorder_joint_orient"],
        "world_before": data["reorder_world_before"],
        "world_after":  data["reorder_world_before"] + data["reorder_world_change"],
        "near_gimbal":  near_gimbal,
        "twice":        near_gimbal & (end < 1e-6),
    }
