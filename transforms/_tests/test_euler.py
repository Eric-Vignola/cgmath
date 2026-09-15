import unittest

import numpy as np
from cgmath.transforms import (
    euler_filter,
    euler_random,
    euler_reorder,
    euler_slerp,
    euler_to_matrix,
    euler_to_quaternion,
    matrix_to_euler,
    quaternion_to_euler,
)

EPSILON = np.finfo(np.float32).eps

# spelled out rather than read from MAYA_EA, so a wrong lookup in
# euler_filter cannot agree with a wrong lookup here
MIDDLE_AXIS = (1, 2, 0, 2, 0, 1)

# Adding a random seed so that we can reproduce the same random numbers and the test is deterministic
RANDOM_SEED = 12345


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestEuler(unittest.TestCase):
    def testToEuler(self):
        ea  = euler_random(10**6, RANDOM_SEED)
        ea_ = euler_reorder(ea, 0, 3)   # change to rotate order to 3
        ea_ = euler_reorder(ea_, 3, 0)  # bring back to rotate order to 0

        self.assertEqual(
            allclose(euler_to_matrix(ea, 0), euler_to_matrix(ea_, 0)), True
        )

    def testToMatrix(self):
        ea  = euler_random(10**6, RANDOM_SEED)
        M   = euler_to_matrix(ea, 0)
        ea_ = matrix_to_euler(M, 0)

        self.assertEqual(allclose(euler_to_matrix(ea), euler_to_matrix(ea_)), True)

    def testToQuaternion(self):
        ea  = euler_random(10**6, RANDOM_SEED)
        Q   = euler_to_quaternion(ea, 0)
        ea_ = quaternion_to_euler(Q, 0)

        self.assertEqual(allclose(euler_to_matrix(ea), euler_to_matrix(ea_)), True)

    def testSlerp(self):
        ea0 = euler_random(10**6, RANDOM_SEED)
        ea1 = euler_random(10**6, RANDOM_SEED)
        w   = np.random.random(10**6)

        # slerp ea0 to ea1
        forward = euler_slerp(ea0, ea1, w)

        # do the opposite
        backward = euler_slerp(ea1, ea0, 1 - w)

        self.assertEqual(
            allclose(euler_to_matrix(forward), euler_to_matrix(backward)), True
        )


class TestEulerFilter(unittest.TestCase):
    # the module level allclose() drops the atol it is handed, so every
    # tolerance here goes through np.allclose directly

    def flip(self, euler, axes):
        """the other representation of the same rotation"""
        out                         = np.array(euler) + np.pi
        out[..., MIDDLE_AXIS[axes]] = np.pi - np.array(euler)[..., MIDDLE_AXIS[axes]]
        return out

    def sweep(self, axis, start, end, count=48, rest=(0.0, 0.0, 0.0), axes=0):
        """a smooth quaternion sweep decomposed one frame at a time"""
        quats = []
        for angle in np.linspace(start, end, count):
            euler = np.array(rest, dtype=float)
            euler[axis] += angle
            quats.append(euler_to_quaternion(np.radians(euler).reshape(1, 3), 0)[0])
        return np.array([quaternion_to_euler(q.reshape(1, 4), axes)[0] for q in quats])

    def step(self, euler):
        return np.degrees(np.abs(np.diff(euler, axis=0)).max())

    def testFlipIdentityHoldsForEveryOrder(self):
        ea = euler_random(4000, RANDOM_SEED)
        for axes in range(6):
            self.assertTrue(
                np.allclose(
                    euler_to_matrix(ea, axes),
                    euler_to_matrix(self.flip(ea, axes), axes),
                    atol=1e-12,
                ),
                f"flip identity broken for rotate order {axes}",
            )

    def testReflectingYAlwaysBreaksFourOfTheOrders(self):
        """guards the table: y is only the middle axis of xyz and zyx"""
        ea            = euler_random(1000, RANDOM_SEED)
        naive         = np.array(ea) + np.pi
        naive[..., 1] = np.pi - np.array(ea)[..., 1]
        for axes in range(6):
            matches = np.allclose(
                euler_to_matrix(ea, axes), euler_to_matrix(naive, axes), atol=1e-12
            )
            self.assertEqual(matches, axes in (0, 5))

    def testPreservesThePose(self):
        ea = euler_random(500, RANDOM_SEED)
        for axes in range(6):
            self.assertTrue(
                np.allclose(
                    euler_to_matrix(ea, axes),
                    euler_to_matrix(euler_filter(ea, axes), axes),
                    atol=1e-12,
                ),
                f"pose not preserved for rotate order {axes}",
            )

    def testPreservesThePoseOfADecomposedSweep(self):
        for axes in range(6):
            ea = self.sweep(1, -170.0, 170.0, rest=(20.0, 0.0, -35.0), axes=axes)
            self.assertTrue(
                np.allclose(
                    euler_to_matrix(ea, axes),
                    euler_to_matrix(euler_filter(ea, axes), axes),
                    atol=1e-12,
                ),
                f"pose not preserved for rotate order {axes}",
            )

    def testRepairsWhatAnUnrollCannot(self):
        """a plain 360 unroll leaves both of these stepping by half a turn"""
        for start, end in ((-95.0, 95.0), (-170.0, 170.0)):
            ea = self.sweep(1, start, end)
            self.assertGreater(self.step(ea), 179.0)
            self.assertLess(self.step(euler_filter(ea, 0)), 10.0)

    def testRepairsAPlainWrap(self):
        ea = self.sweep(2, 0.0, 350.0)
        self.assertGreater(self.step(ea), 350.0)
        self.assertLess(self.step(euler_filter(ea, 0)), 10.0)

    def testLeavesAContinuousCurveAlone(self):
        ea = self.sweep(0, -170.0, 170.0)
        self.assertTrue(np.allclose(euler_filter(ea, 0), ea, atol=1e-12))

    def testCannotRecoverMoreThanHalfATurnPerFrame(self):
        """pins the aliasing limit, which maya's filterCurve shares"""
        authored       = np.zeros((11, 3))
        authored[:, 0] = np.radians(np.arange(11) * 240.0)

        filtered = euler_filter(authored, 0)

        # the spin comes back reversed and half as long
        self.assertAlmostEqual(np.degrees(filtered[-1, 0]), -1200.0, places=6)
        self.assertTrue(
            np.allclose(
                euler_to_matrix(authored, 0), euler_to_matrix(filtered, 0), atol=1e-12
            )
        )

    def testLeavesSlowerAuthoredSpinsAlone(self):
        authored       = np.zeros((11, 3))
        authored[:, 0] = np.radians(np.arange(11) * 179.0)
        self.assertTrue(np.allclose(euler_filter(authored, 0), authored, atol=1e-12))

    def testFiltersEachCurveOnItsOwnOrder(self):
        block = np.stack(
            [
                self.sweep(1, -95.0, 95.0, 30, axes=0),
                self.sweep(1, -95.0, 95.0, 30, axes=3),
            ],
            axis=1,
        )
        together = euler_filter(block, [0, 3])
        apart = np.stack(
            [euler_filter(block[:, 0], 0), euler_filter(block[:, 1], 3)], axis=1
        )
        self.assertTrue(np.array_equal(together, apart))

    def testShapes(self):
        for shape in ((3,), (1, 3), (2, 3), (0, 3), (5, 0, 3), (4, 2, 3)):
            expected = (1, 3) if shape == (3,) else shape
            self.assertEqual(euler_filter(np.zeros(shape), 0).shape, expected)

    def testAcceptsListsAndIntegers(self):
        self.assertEqual(
            euler_filter([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], 0).shape, (2, 3)
        )
        self.assertEqual(euler_filter(np.zeros((3, 3), dtype=int), 0).dtype, np.float64)

    def testReturnsAWritableResultForAReadOnlyInput(self):
        """a clip hands back a read only view of its blocks"""
        ea = euler_random(20, RANDOM_SEED)
        ea.setflags(write=False)

        filtered       = euler_filter(ea, 0)
        filtered[0, 0] = 1.0

        self.assertEqual(filtered[0, 0], 1.0)

    def testRejectsBadInput(self):
        with self.assertRaises(ValueError):
            euler_filter(np.zeros((4, 4)), 0)
        for axes in (6, -1):
            with self.assertRaises(ValueError):
                euler_filter(np.zeros((4, 3)), axes)

    def testRejectsABadOrderWithNothingToFilter(self):
        """the order check has to run before the single frame short circuit"""
        for euler in (np.zeros((1, 3)), np.zeros(3)):
            with self.assertRaises(ValueError):
                euler_filter(euler, 6)