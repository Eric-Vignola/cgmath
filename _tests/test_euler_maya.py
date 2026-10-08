"""Euler filtering and rotate order changes, checked against Maya.

test_assets/maya_euler.npz holds what Maya 2025 did, recorded by
tools/make_maya_euler_fixture.py (its docstring lists every array). Nothing
here needs Maya.
"""
import os
import unittest

import numpy as np
from cgmath.transforms import XYZ, euler_filter, euler_reorder, euler_to_matrix
from cgmath.transforms._numba._euler import MAYA_EA

FIXTURE = os.path.join(os.path.dirname(__file__), "test_assets", "maya_euler.npz")

# degrees. cgmath and Maya agree to 2.8e-14 on the filtered keys and to
# 3.2e-13 on the reordered rotates, in python 3.11 and in mayapy alike; a
# wrong pick between two solutions is off by whole degrees
TOLERANCE = 1e-11


def load_fixture():
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def maya_rotate_order_change(rotate, from_order, to_order):
    """The rotate Maya's ``xform -rotateOrder <order> -preserve`` leaves, in
    degrees, one row per rotate: the rule the coming set_rotate_order is to
    follow.

    Take euler_reorder's solution with every angle in [-180, 180) (+180
    becomes -180), and the other solution of the same rotation: the first
    and last axes the new order applies turned by 180, the middle one
    reflected (180 - angle), not wrapped, so its angles can pass 180. Keep
    whichever has the smaller sum of absolute angles; on a tie, the first.

    Maya follows it on all 2550 joints of the fixture, the other solution
    winning on 157, except that where it ends on an angle of exactly 180 it
    writes +180 or -180 as its own arithmetic falls (145 and 82 times), so
    on 83 joints this rule's -180 is Maya's +180. Compare with
    ``angle_error``.
    """
    rotate   = np.asarray(rotate, dtype=float)
    to_order = np.asarray(to_order)
    rows     = np.arange(len(rotate))
    first, middle, last = MAYA_EA[to_order].T

    standard = np.degrees(euler_reorder(np.radians(rotate), from_order, to_order))
    standard = (standard + 180.0) % 360.0 - 180.0

    other = standard.copy()
    other[rows, first] += 180.0
    other[rows, last]  += 180.0
    other[rows, middle] = 180.0 - standard[rows, middle]

    shorter = np.abs(other).sum(axis=1) < np.abs(standard).sum(axis=1)
    return np.where(shorter[:, None], other, standard)


def angle_error(a, b):
    """|a - b| per angle, where +180 and -180 are the same angle."""
    a, b = np.asarray(a), np.asarray(b)
    error = np.abs(a - b)
    half  = np.isclose(np.abs(a), 180.0, rtol=0.0, atol=1e-9)
    half &= np.isclose(np.abs(b), 180.0, rtol=0.0, atol=1e-9)
    return np.where(half, np.abs(np.abs(a) - np.abs(b)), error)


class TestEulerFilterMaya(unittest.TestCase):
    """
    euler_filter against Maya 2025's ``filterCurve -filter euler`` on the rx,
    ry and rz curves of a transform, in every rotate order: a smooth rotation
    decomposed at 8, 25, 60 and 110 degrees a frame, a curve with whole turns
    and flips, one with the middle axis near 90 and one starting far outside
    -180..180 (42 curves); 150 two-key pairs where summing the axes and
    squaring them pick a different solution; 30 two-key pairs with the
    middle axis at 90, 6 of them exact ties (Maya keeps the key) and 24 near
    ties (the other solution 10 to 40 degrees shorter, and Maya takes it).
    Maya moves 1458 of the 2232 keys, on 139 of the 222 curves.
    """

    @classmethod
    def setUpClass(cls):
        data       = load_fixture()
        cls.names  = data["filter_name"]
        cls.orders = data["filter_order"]
        cls.start  = data["filter_start"]
        cls.keys   = data["filter_keys"]
        cls.maya   = data["filter_maya"]

    def curves(self, length):
        """(frames, curves, 3) keys and Maya's keys of every curve that long,
        and the curves' indices"""
        picked = np.flatnonzero(np.diff(self.start) == length)
        rows   = self.start[picked][:, None] + np.arange(length)
        keys   = self.keys[rows].transpose(1, 0, 2)
        maya   = self.maya[rows].transpose(1, 0, 2)
        return keys, maya, picked

    def test_every_key_is_mayas(self):
        lengths = np.unique(np.diff(self.start))
        self.assertEqual(lengths.tolist(), [2, 24, 48])
        for length in lengths:
            keys, maya, picked = self.curves(length)
            ours  = np.degrees(euler_filter(np.radians(keys), self.orders[picked]))
            error = np.abs(ours - maya).max(axis=(0, 2))
            bad   = self.names[picked][error > TOLERANCE]
            self.assertEqual(bad.tolist(), [], f"worst {error.max():.3g} degrees")

    def test_maya_keeps_the_first_key(self):
        first = self.start[:-1]
        self.assertTrue(np.allclose(self.maya[first], self.keys[first], rtol=0.0, atol=TOLERANCE))

    def test_maya_sums_the_axes(self):
        # on every pair, the squared distance would take the other solution:
        # Maya sums the axes, as euler_filter does
        keys, maya, picked = self.curves(2)
        pairs    = np.char.find(self.names[picked], "_pair") >= 0
        previous = keys[0, pairs]
        current  = keys[1, pairs]
        rows     = np.arange(len(current))
        first, middle, last = MAYA_EA[self.orders[picked][pairs]].T

        other = current.copy()
        other[rows, first] += 180.0
        other[rows, last]  += 180.0
        other[rows, middle] = 180.0 - current[rows, middle]

        candidates = np.stack((current, other))
        candidates += 360.0 * np.round((previous - candidates) / 360.0)
        squared    = ((candidates - previous) ** 2).sum(axis=-1)
        by_squares = candidates[squared.argmin(axis=0), rows]

        self.assertEqual(int(pairs.sum()), 150)
        self.assertTrue((np.abs(by_squares - maya[1, pairs]).max(axis=1) > 1.0).all())


class TestRotateOrderChangeMaya(unittest.TestCase):
    """
    Rotate order changes against Maya 2025's ``xform -rotateOrder <order>
    -preserve`` on a joint with a joint orient: 85 rotations through all 30
    pairs of rotate orders, 2550 joints. 40 rotations are random or picked
    by hand; the other 45 are zero, single axes at +-90, +-180 and 720, one
    axis at +-90, 1e-7 either side of it or +-180 with the others not zero,
    every axis at 180, every axis at -180, (180, -180, 90), and two +-720
    spins.
    """

    @classmethod
    def setUpClass(cls):
        data             = load_fixture()
        cls.rotate       = data["reorder_rotate"]
        cls.from_order   = data["reorder_from"]
        cls.to_order     = data["reorder_to"]
        cls.joint_orient = data["reorder_joint_orient"]
        cls.maya         = data["reorder_maya"]
        cls.orient_after = data["reorder_joint_orient_after"]
        cls.world_before = data["reorder_world_before"]
        cls.world_change = data["reorder_world_change"]

        # the 120 joints near gimbal: the middle axis of the order they
        # start in is 1e-7 from +-90. an angle 1e-7 from 90 on another axis
        # (240 more joints) does not trouble Maya: 7.8e-16 at most
        rows            = np.arange(len(cls.rotate))
        middle          = cls.rotate[rows, MAYA_EA[cls.from_order][:, 1]]
        off             = np.abs(np.abs(middle) - 90.0)
        cls.near_gimbal = (off > 0.0) & (off < 1e-6)

        # the 8 of those whose new rotate lands 1e-7 from gimbal again, its
        # new middle axis 4.2e-8 or 9.1e-8 from 90
        middle    = cls.maya[rows, MAYA_EA[cls.to_order][:, 1]]
        cls.twice = cls.near_gimbal & (np.abs(np.abs(middle) - 90.0) < 1e-6)

    def test_mayas_rule(self):
        ours = maya_rotate_order_change(self.rotate, self.from_order, self.to_order)
        self.assertLess(angle_error(ours, self.maya).max(), TOLERANCE)

        # the other solution stays unwrapped, as in Maya
        self.assertGreater(self.maya.max(), 180.0)

    def test_maya_keeps_the_joint_orient_and_the_pose(self):
        self.assertTrue(np.array_equal(self.orient_after, self.joint_orient))

        # Maya's world matrix moves by up to 4.3e-9 through the change on the
        # 120 joints near gimbal, and by 3.7e-15 at most on the other 2430
        moved = np.abs(self.world_change).max(axis=(1, 2))
        self.assertEqual(int(self.near_gimbal.sum()), 120)
        self.assertLess(moved[self.near_gimbal].max(), 1e-8)
        self.assertLess(moved[~self.near_gimbal].max(), 1e-14)

    def test_cgmath_poses_are_mayas(self):
        # a joint's world matrix is its rotate, then its joint orient (xyz)
        orient = euler_to_matrix(np.radians(self.joint_orient), XYZ)
        before = euler_to_matrix(np.radians(self.rotate), self.from_order) @ orient
        after  = euler_to_matrix(np.radians(self.maya), self.to_order) @ orient
        self.assertLess(np.abs(before - self.world_before).max(), 1e-14)

        # Maya's new rotate holds the old pose to 3.8e-15, except on the 8
        # joints that start and end 1e-7 from gimbal: the decomposition
        # loses digits there, up to 3.1e-9 (on 6 of the 8), in Maya and in
        # euler_reorder alike
        moved = np.abs(after - self.world_before).max(axis=(1, 2))
        self.assertEqual(int(self.twice.sum()), 8)
        self.assertLess(moved[self.twice].max(), 1e-8)
        self.assertLess(moved[~self.twice].max(), 1e-14)


if __name__ == "__main__":
    unittest.main()
