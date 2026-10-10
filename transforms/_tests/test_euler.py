import sys
import unittest
from unittest import mock

import numba
import numpy as np
from cgmath.transforms import _numba
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


def frame_by_frame(euler, axes):
    """The reference: euler_filter as it was before its kernel, one frame at
    a time, each frame batched over the curves. That loop is the one checked
    key for key against maya's filterCurve, every rotate order."""
    euler  = np.asarray(euler, dtype=float)
    curves = euler.reshape(len(euler), -1, 3)
    rows   = np.arange(curves.shape[1])
    middle = np.asarray(MIDDLE_AXIS)[np.broadcast_to(axes, rows.shape)]

    out = curves.copy()
    for f in range(1, len(curves)):
        previous = out[f - 1]
        current  = curves[f]

        flipped               = current + np.pi
        flipped[rows, middle] = np.pi - current[rows, middle]

        candidates = np.stack((current, flipped))
        candidates += 2.0 * np.pi * np.round((previous - candidates) / (2.0 * np.pi))
        cost   = np.abs(candidates - previous).sum(axis=-1)
        out[f] = candidates[np.argmin(cost, axis=0), rows]

    return out.reshape(euler.shape)


def decomposed(frames, count, step, seed=RANDOM_SEED):
    """(frames, count, 3) rotates rebuilt from matrices, and their rotate
    orders: a random walk of up to ``step`` degrees a frame on every axis,
    decomposed frame by frame in each curve's own order, so the curves jump
    from branch to branch the way a clip op's rebuilt rotate does"""
    rng     = np.random.default_rng(seed)
    orders  = np.arange(count) % 6
    steps   = rng.uniform(-step, step, (frames, count, 3))
    walk    = np.radians(np.cumsum(steps, axis=0))
    matrix  = euler_to_matrix(walk.reshape(-1, 3), 0)
    rebuilt = matrix_to_euler(matrix, np.tile(orders, frames))
    return rebuilt.reshape(frames, count, 3), orders


class TestEulerFilterKernel(unittest.TestCase):
    """euler_filter filters every curve in one numba kernel, frames in order
    within each curve, and must give back exactly what the frame loop did."""

    def assertSameAsFrameByFrame(self, euler, axes):
        filtered  = euler_filter(euler, axes)
        reference = frame_by_frame(euler, axes)
        first     = np.asarray(euler)[0]
        self.assertTrue(np.array_equal(filtered, reference, equal_nan=True))
        self.assertTrue(np.array_equal(filtered[0], first, equal_nan=True))
        return filtered

    def testMatchesTheFrameByFrameFilterBitForBit(self):
        # slow and fast motion: from 4 degrees a frame up to past half a turn
        for step in (4.0, 25.0, 60.0, 110.0, 170.0, 240.0):
            euler, orders = decomposed(120, 60, step)
            self.assertSameAsFrameByFrame(euler, orders)

    def testMatchesTheFrameByFrameFilterOnAClip(self):
        """a clip's rotate block, (frames, nodes, 3), any rotate order per node"""
        rng = np.random.default_rng(RANDOM_SEED)
        for frames in (47, 300):
            euler, _ = decomposed(frames, 271, 4.0, seed=frames)
            orders = rng.integers(0, 6, 271)
            self.assertSameAsFrameByFrame(euler, orders)

    def testMatchesTheFrameByFrameFilterFarFromZero(self):
        """whole turns added at random to every key, starting far outside
        -180..180"""
        rng = np.random.default_rng(RANDOM_SEED)
        euler, ax = decomposed(90, 30, 40.0)
        turns = rng.integers(-3, 4, euler.shape) * 2.0 * np.pi
        self.assertSameAsFrameByFrame(euler + turns + np.radians(1000.0), ax)

    def testMatchesTheFrameByFrameFilterOnALongCurve(self):
        euler, _ = decomposed(5000, 1, 60.0)
        for axes in range(6):
            self.assertSameAsFrameByFrame(euler, axes)

    def testKeepsTheFrameOnATie(self):
        """with the middle axis at 90 degrees both representations can move
        exactly as far: the frame keeps its own, as filterCurve does"""
        for axes in range(6):
            euler    = np.zeros((2, 3))
            euler[0] = np.pi / 2.0
            euler[:, MIDDLE_AXIS[axes]] = np.pi / 2.0
            filtered = self.assertSameAsFrameByFrame(euler, axes)
            self.assertTrue(np.array_equal(filtered[1], euler[1]))

    def testLeavesANonFiniteKeyOnItsOwnFrame(self):
        """a frame holding a nan or an inf is a gap: it stays as it is, and
        the frames after it filter as if it were not there. The frame loop
        carried a nan to every later key of its curve, which turned a whole
        clip curve into nan after one degenerate frame (a joint scaled to 0)"""
        euler, orders = decomposed(30, 12, 30.0)
        euler[5, 2, 0]  = np.nan
        euler[9, 7, 1]  = np.inf
        euler[0, 4, 2]  = -np.inf  # a gap on the first frame
        euler[:3, 5, 1] = np.nan   # leading gaps: the first finite frame is kept
        euler[4:7, 8]   = np.inf   # a run of gaps
        euler[:, 10, 0] = np.nan   # nothing finite at all
        euler[1:, 11]   = np.nan   # one finite frame, the first

        filtered  = euler_filter(euler, orders)

        reference = euler.copy()
        for c in range(euler.shape[1]):
            finite = np.flatnonzero(np.isfinite(euler[:, c]).all(axis=-1))
            if len(finite):
                reference[finite, c] = frame_by_frame(euler[finite, c], orders[c])
        self.assertTrue(np.array_equal(filtered, reference, equal_nan=True))

        # the gaps are left exactly as given, nan, inf and -inf alike
        gaps = ~np.isfinite(euler)
        self.assertTrue(np.array_equal(filtered[gaps], euler[gaps], equal_nan=True))
        self.assertTrue(np.isfinite(filtered[:, 2, 0][6:]).all())

    def testFiltersAcrossAGap(self):
        """a flipped key right after a nan is filtered against the last
        finite key: (190, 170, -170) in xyz is (10, 10, 10)"""
        euler = np.radians(
            [[0.0, 0.0, 0.0], [np.nan, 5.0, 5.0], [190.0, 170.0, -170.0], [15.0, 15.0, 15.0]]
        )
        filtered = np.degrees(euler_filter(euler, 0))
        self.assertTrue(np.isnan(filtered[1, 0]))
        np.testing.assert_allclose(filtered[1, 1:], [5.0, 5.0],         atol=1e-12)
        np.testing.assert_allclose(filtered[2],     [10.0, 10.0, 10.0], atol=1e-12)
        np.testing.assert_allclose(filtered[3],     [15.0, 15.0, 15.0], atol=1e-12)

    def testSolvesEveryFrameAtOnce(self):
        """no python work per frame: the python calls one euler_filter makes
        do not grow with the frame count, on one thread or on every thread
        (fetching the single-threaded kernel is a call of its own)"""
        def calls(euler, axes):
            count = [0]

            def profile(frame, event, arg):
                count[0] += 1

            euler_filter(euler, axes)  # numba loads the kernel on its first call
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                euler_filter(euler, axes)
            finally:
                sys.setprofile(previous)
            return count[0]

        short, orders = decomposed(8, 20, 30.0)
        long, _ = decomposed(800, 20, 30.0)
        for threshold in (0, sys.maxsize):
            with mock.patch.object(_numba, "SINGLE_THREAD_BELOW", threshold):
                self.assertEqual(calls(short, orders), calls(long, orders), threshold)

    def testLeavesItsInputAlone(self):
        """the kernel filters in place, on a copy of its own"""
        euler, orders = decomposed(40, 10, 110.0)
        before   = euler.copy()

        filtered = euler_filter(euler, orders)

        self.assertTrue(np.array_equal(euler, before))
        self.assertFalse(np.shares_memory(filtered, euler))
        self.assertFalse(np.array_equal(filtered, euler))

    def testAnyMemoryLayout(self):
        euler, orders = decomposed(40, 10, 110.0)
        expected   = euler_filter(euler, orders)

        transposed = np.ascontiguousarray(euler.transpose(1, 0, 2)).transpose(1, 0, 2)
        fortran    = np.asfortranarray(euler)
        strided    = np.repeat(euler, 2, axis=-1)[..., ::2]
        for view in (transposed, fortran, strided):
            self.assertTrue(np.array_equal(view, euler))
            self.assertTrue(np.array_equal(euler_filter(view, orders), expected))

    def testOneThreadOrMany(self):
        euler, orders = decomposed(60, 48, 110.0)
        threads = numba.get_num_threads()
        many    = euler_filter(euler, orders)
        numba.set_num_threads(1)
        try:
            one = euler_filter(euler, orders)
        finally:
            numba.set_num_threads(threads)
        self.assertTrue(np.array_equal(one, many))
