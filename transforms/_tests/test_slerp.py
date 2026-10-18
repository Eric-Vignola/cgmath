"""The slerps keep their precision without fast math.

2.0.0 took fast math off every kernel so that the single-threaded twins
answer bit for bit (see test_threads), and the slerps' arc angle, the
``acos`` of a dot product divided by ``sqrt(1 - dot ** 2)``, then lost half
its digits for nearly parallel pairs: slerped quaternions came out 1e-9 off
unit length, matrix_slerp 1e-8 off a rotation, and euler_slerp near
gimbal answered up to 7e-3 radians off at weight 0, where it should give
back the first rotation. 1.0.4, with fast math, kept slerped quaternions
unit to 2e-15 and matrix_slerp's matrices rotations to 1.3e-14, though it
too turned nearly parallel pairs up to about 1e-7 radians wrong.

The reference is scipy's float64 ``Slerp``, one pair at a time, and for
``vector_slerp`` its own formula, ``(vector0 * sin((1 - w) * angle) +
vector1 * sin(w * angle)) / sin(angle)`` on the vectors as given, taken
another way.
"""

import sys
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cgmath import transforms
from cgmath.transforms import (
    euler_slerp,
    euler_to_matrix,
    euler_to_quaternion,
    matrix_slerp,
    quaternion_slerp,
    quaternion_squad,
    quaternion_to_euler,
    quaternion_to_matrix,
    vector_slerp,
)
from cgmath.transforms import _numba
from cgmath.transforms._numba._quaternion import _quaternion_slerp

THRESHOLD = _numba.SINGLE_THREAD_BELOW
WEIGHTS   = (0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9, 1.0)

# radians between two rotations, against a reference that is itself float64
ROTATION_TOL = 4e-15


def _unit(a):
    return a / np.linalg.norm(a, axis=-1)[..., None]


def _random_quaternions(rng, n):
    return _unit(rng.normal(size=(n, 4)))


def _turned(rng, quat, angles):
    """``quat`` turned by ``angles`` radians about random axes"""
    rotvec = _unit(rng.normal(size=(len(quat), 3))) * np.asarray(angles)[:, None]
    return (Rotation.from_quat(quat) * Rotation.from_rotvec(rotvec)).as_quat()


def _near_gimbal(rng, n, low=1e-5, high=5.0):
    """euler angles (xyz) whose middle angle is ``low`` to ``high`` degrees
    from 90, log-uniform, and a second set up to a degree from each"""
    off    = np.radians(10.0 ** rng.uniform(np.log10(low), np.log10(high), n))
    middle = np.where(rng.random(n) < 0.5, np.pi / 2 - off, -np.pi / 2 + off)
    euler0 = np.stack(
        [rng.uniform(-np.pi, np.pi, n), middle, rng.uniform(-np.pi, np.pi, n)], -1
    )
    euler1 = euler0 + np.radians(rng.uniform(-1.0, 1.0, (n, 3)))
    return euler0, euler1


def _scipy_slerp(quat0, quat1, weight):
    """scipy's float64 Slerp of each pair, along the shorter arc"""
    weight = np.broadcast_to(weight, (len(quat0),))
    return np.array(
        [
            Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))([t]).as_quat()[0]
            for q0, q1, t in zip(quat0, quat1, weight)
        ]
    )


def _quaternion_angle(quat0, quat1):
    """radians between the rotations of two lists of quaternions"""
    quat0 = _unit(np.asarray(quat0))
    quat1 = _unit(np.asarray(quat1))
    d     = np.linalg.norm(quat0 - quat1, axis=-1)
    s     = np.linalg.norm(quat0 + quat1, axis=-1)
    return 4.0 * np.arctan2(np.minimum(d, s), np.maximum(d, s))


def _matrix_angle(matrix0, matrix1):
    """radians between the rotations of two lists of matrices"""
    d = np.linalg.norm(
        (matrix0[:, :3, :3] - matrix1[:, :3, :3]).reshape(len(matrix0), 9), axis=1
    )
    return 2.0 * np.arcsin(np.minimum(d / 8.0**0.5, 1.0))


def _arc4(angle):
    """unit quaternions ``angle`` radians from the identity, toward +X"""
    return np.stack([np.sin(angle), 0 * angle, 0 * angle, np.cos(angle)], -1)


def _flipped(quat0, quat1):
    """quat1, negated where the shorter arc needs it"""
    dot = np.einsum("...i,...i", quat0, quat1)
    return np.where((dot < 0.0)[:, None], -quat1, quat1)


class TestQuaternionSlerp(unittest.TestCase):
    def check(self, quat0, quat1, label, tol=ROTATION_TOL):
        rng = np.random.default_rng(len(quat0))
        for weight in WEIGHTS + (rng.random(len(quat0)),):
            got   = quaternion_slerp(quat0, quat1, weight)
            want  = _scipy_slerp(quat0, quat1, weight)
            where = "%s, weight %s" % (label, np.ravel(weight)[:1])
            self.assertLess(_quaternion_angle(got, want).max(), tol, where)
            inner = (np.broadcast_to(weight, (len(quat0),)) > 0.0) & (
                np.broadcast_to(weight, (len(quat0),)) < 1.0
            )
            # the arc keeps the length to 1e-15 by itself: the kernel's last
            # normalisation guards an ulp, which no 1e-15 check can see
            length = np.linalg.norm(got[inner], axis=1)
            if inner.any():
                self.assertLess(np.abs(length - 1.0).max(), 1e-15, where)

    def test_random_pairs(self):
        rng   = np.random.default_rng(1)
        quat0 = _random_quaternions(rng, 400)
        quat1 = _random_quaternions(rng, 400)
        self.check(quat0, quat1, "random")

    def test_nearly_parallel_pairs(self):
        rng    = np.random.default_rng(2)
        quat0  = _random_quaternions(rng, 400)
        angles = 10.0 ** rng.uniform(-15, -2, 400)
        self.check(quat0, _turned(rng, quat0, angles), "nearly parallel")

    def test_pairs_near_gimbal(self):
        rng = np.random.default_rng(3)
        euler0, euler1 = _near_gimbal(rng, 400, low=1e-6)
        quat0 = euler_to_quaternion(euler0, 0)
        quat1 = euler_to_quaternion(euler1, 0)
        self.check(quat0, quat1, "near gimbal")

    def test_pairs_in_opposite_hemispheres(self):
        # the shorter arc negates quat1: nearly parallel rotations whose
        # quaternions are nearly opposite
        rng    = np.random.default_rng(4)
        quat0  = _random_quaternions(rng, 400)
        angles = 10.0 ** rng.uniform(-15, -1, 400)
        self.check(quat0, -_turned(rng, quat0, angles), "opposite hemispheres")

    def test_quaternions_of_any_length(self):
        rng   = np.random.default_rng(5)
        quat0 = _random_quaternions(rng, 400)
        quat1 = _turned(rng, quat0, 10.0 ** rng.uniform(-12, 0, 400))
        scale = 10.0 ** rng.uniform(-3, 3, (400, 1))
        self.check(quat0 * scale, quat1 / scale, "scaled")

    def test_quaternions_too_short_or_too_long_to_square(self):
        # under about 1e-146 long the squares of the components lose bits
        # as subnormals, over about 1e154 they overflow
        rng   = np.random.default_rng(19)
        quat0 = _random_quaternions(rng, 400)
        quat1 = _turned(rng, quat0, 10.0 ** rng.uniform(-12, 0, 400))
        for scale in (1e-300, 1e-160, 1e160, 1e300):
            for weight in WEIGHTS[1:-1]:
                got  = quaternion_slerp(quat0 * scale, quat1 / scale, weight)
                want = quaternion_slerp(quat0, quat1, weight)
                self.assertLess(_quaternion_angle(got, want).max(), ROTATION_TOL)
                self.assertLess(np.abs(np.linalg.norm(got, axis=1) - 1.0).max(), 1e-15)

    def test_both_too_short_or_too_long_for_their_dot_product(self):
        # the shorter arc is picked from the dot product of the two as
        # given: two under about 1.5e-154 long leave it subnormal, and its
        # sign uncertain, under about 1e-162 underflowed to 0, and two over
        # about 1.3e154 overflowed, its sign lost with it. One 1e-300 long
        # and the other 1e-10 leave it subnormal too
        rng   = np.random.default_rng(24)
        quat0 = _random_quaternions(rng, 400)
        quat1 = _random_quaternions(rng, 400)
        self.assertGreater((np.einsum("ij,ij->i", quat0, quat1) < 0.0).sum(), 150)
        scales = (
            (3e-162, 3e-162),
            (1e-170, 1e-170),
            (1e-300, 1e-300),
            (1e-300, 1e-10),
            (1e160, 1e160),
            (1e300, 1e300),
        )
        for scale0, scale1 in scales:
            scaled0 = quat0 * scale0
            scaled1 = quat1 * scale1
            for shortest in (True, False):
                for weight in WEIGHTS[1:-1]:
                    got  = quaternion_slerp(scaled0, scaled1, weight, shortest)
                    want = quaternion_slerp(quat0, quat1, weight, shortest)
                    self.assertLess(_quaternion_angle(got, want).max(), ROTATION_TOL)
            np.testing.assert_array_equal(
                quaternion_slerp(scaled0, scaled1, 1.0), _flipped(quat0, quat1) * scale1
            )
            got  = quaternion_squad(scaled0, scaled0, scaled1, scaled1)
            want = quaternion_squad(quat0, quat0, quat1, quat1)
            self.assertLess(_quaternion_angle(got, want).max(), 2 * ROTATION_TOL)

        # one product overflowed, the other three finite and larger
        # together: the dot product is +inf, the sign the wrong one
        a     = (2.75 / 5.75) ** 0.5
        b     = (1.0 / 5.75) ** 0.5
        quat0 = np.array([[a, b, b, b]])
        quat1 = np.array([[a, -b, -b, -b]])
        got   = quaternion_slerp(quat0 * 2.03e154, quat1 * 2.03e154, 0.5)
        want  = quaternion_slerp(quat0, quat1, 0.5)
        self.assertLess(_quaternion_angle(got, want).max(), ROTATION_TOL)

    def test_a_tie_keeps_quat1_as_given(self):
        # a half turn about X apart: a dot product of exactly 0, both arcs
        # as long. Taken again from the two divided by their largest
        # components, (10, 45, 75) and (190, 45, 75) degrees give -1.1e-16,
        # the rounding of another sum. The half turn goes the way quat1 as
        # given says, as in 1.0.4 and 2.0.0, along either arc, and for the
        # two scaled by 2^-400 or 2^400
        grid   = np.arange(0.0, 90.0, 5.0)
        euler0 = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
        euler1 = euler0 + [180.0, 0.0, 0.0]
        quat0  = euler_to_quaternion(np.radians(euler0))
        quat1  = euler_to_quaternion(np.radians(euler1))
        tie    = np.einsum("ij,ij->i", quat0, quat1) == 0.0
        self.assertTrue(tie[np.all(euler0 == [10.0, 45.0, 75.0], axis=1)].any())
        self.assertGreater(tie.sum(), 1000)
        quat0 = quat0[tie]
        quat1 = quat1[tie]
        for scale in (1.0, 2.0**-400, 2.0**400):
            scaled0 = quat0 * scale
            scaled1 = quat1 * scale
            for weight in WEIGHTS[1:-1]:
                weights = np.full(len(quat0), weight)
                for shortest, given in ((True, scaled1), (False, -scaled1)):
                    np.testing.assert_array_equal(
                        quaternion_slerp(scaled0, scaled1, weight, shortest),
                        _quaternion_slerp(scaled0, given, weights),
                    )
            got  = quaternion_squad(scaled0, scaled0, scaled1, scaled1)
            want = _quaternion_slerp(scaled0, scaled1, np.full(len(quat0), 0.5))
            self.assertLess(_quaternion_angle(got, want).max(), ROTATION_TOL)

    def test_a_zero_quaternion_lerps(self):
        # no rotation to normalise, to turn from or to, and no arc to pick:
        # the longer one too keeps the other quaternion as given
        rng  = np.random.default_rng(20)
        quat = _random_quaternions(rng, 400)
        zero = np.zeros((400, 4))
        for weight in WEIGHTS:
            for quat0, quat1 in ((zero, quat), (quat, zero)):
                for shortest in (True, False):
                    np.testing.assert_array_equal(
                        quaternion_slerp(quat0, quat1, weight, shortest),
                        quat0 + weight * (quat1 - quat0),
                    )
        np.testing.assert_array_equal(quaternion_slerp(zero, zero, 0.5), zero)

    def test_a_nan_or_an_inf_gives_nans(self):
        # between the ends. the last one's nan is the one the search for the
        # largest component passes over. none of them gives an arc to pick,
        # -inf and its dot product of -inf included: at the ends, each of the
        # two comes back as given
        quat = np.array([0.1, 0.2, 0.3, 0.9])
        nan  = np.nan
        inf  = np.inf
        bads = (
            [nan, 0.0, 0.0, 1.0],
            [inf, 0.0, 0.0, 1.0],
            [-inf, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, nan],
        )
        for bad in bads:
            for quat0, quat1 in ((bad, quat), (quat, bad)):
                for shortest in (True, False):
                    got = quaternion_slerp(quat0, quat1, 0.5, shortest)
                    self.assertTrue(np.isnan(got).all(), (quat0, quat1, got))
                    got = quaternion_slerp(quat0, quat1, [0.0, 1.0], shortest)
                    np.testing.assert_array_equal(got, [quat0, quat1])

    def test_weight_0_and_1_return_the_quaternions(self):
        rng   = np.random.default_rng(6)
        quat0 = _random_quaternions(rng, 400) * 1.5
        quat1 = _turned(rng, quat0, 10.0 ** rng.uniform(-15, 0, 400))
        np.testing.assert_array_equal(quaternion_slerp(quat0, quat1, 0.0), quat0)
        np.testing.assert_array_equal(
            quaternion_slerp(quat0, quat1, 1.0), _flipped(quat0, quat1)
        )
        np.testing.assert_array_equal(
            quaternion_slerp(quat0, quat1, 1.0, shortest=False),
            -_flipped(quat0, -quat1),
        )

    def test_the_longer_arc(self):
        # the other way round about the same axis. that axis is the
        # direction of the shorter arc's rotation vector, as precise as
        # eps / angle, and the longer arc turns by nearly 2 pi about it
        rng    = np.random.default_rng(7)
        quat0  = _random_quaternions(rng, 400)
        quat1  = _turned(rng, quat0, 10.0 ** rng.uniform(-6, np.log10(3.0), 400))
        turn   = Rotation.from_quat(quat0).inv() * Rotation.from_quat(quat1)
        rotvec = turn.as_rotvec()
        angle  = np.linalg.norm(rotvec, axis=1)
        longer = rotvec * ((angle - 2.0 * np.pi) / angle)[:, None]
        tol    = ROTATION_TOL * (1.0 + 2.0 * np.pi / angle)
        for weight in WEIGHTS[1:-1]:
            got = quaternion_slerp(quat0, quat1, weight, shortest=False)
            want = (
                Rotation.from_quat(quat0) * Rotation.from_rotvec(longer * weight)
            ).as_quat()
            self.assertTrue((_quaternion_angle(got, want) < tol).all(), weight)

    def test_one_rotation_the_long_way_round_stays(self):
        # q and -q are one rotation: the longer arc between them is a whole
        # turn about no axis in particular
        rng   = np.random.default_rng(8)
        quat0 = _random_quaternions(rng, 400)
        # a scale other than a power of two leaves the two, normalised,
        # opposite to within rounding rather than exactly
        for quat1 in (quat0, -quat0, quat0 * 2.0, quat0 * 3.0, -quat0 * 0.7):
            for weight in WEIGHTS[1:-1]:
                got = quaternion_slerp(quat0, quat1, weight, shortest=False)
                self.assertLess(_quaternion_angle(got, quat0).max(), ROTATION_TOL)
                np.testing.assert_allclose(got, quat0, rtol=0, atol=1e-15)

    def test_continuous_down_to_parallel_and_through_a_right_angle(self):
        # every angle down to none
        angles = np.array([0.0, 1e-300, 1e-150, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4])
        quat0  = _arc4(0 * angles)
        quat1  = _arc4(angles)
        for weight in WEIGHTS[1:-1]:
            got  = quaternion_slerp(quat0, quat1, weight)
            want = _arc4(weight * angles)
            np.testing.assert_allclose(got, want, rtol=1e-15, atol=1e-299)

        # past a right angle, the arc is taken from the sum of the two: only
        # quaternion_slerp(shortest=False) gets there, and only from past it,
        # so the kernel itself, through the switch
        angles = np.pi / 2.0 + np.array([-1e-9, -1e-15, 0.0, 1e-15, 1e-9, 1.0])
        quat0  = _arc4(0 * angles)
        quat1  = _arc4(angles)
        for weight in WEIGHTS[1:-1]:
            got = _quaternion_slerp(quat0, quat1, np.full(len(angles), weight))
            np.testing.assert_allclose(got, _arc4(weight * angles), rtol=0, atol=4e-16)

        # nearly opposite, from the sum: down to just past rounding
        angles = np.pi - np.array([1e-15, 1e-13, 1e-10, 1e-7, 1e-4])
        quat0  = _arc4(0 * angles)
        quat1  = _arc4(angles)
        for weight in WEIGHTS[1:-1]:
            got = _quaternion_slerp(quat0, quat1, np.full(len(angles), weight))
            np.testing.assert_allclose(got, _arc4(weight * angles), rtol=0, atol=4e-16)


class TestBuiltOnTheQuaternionSlerp(unittest.TestCase):
    def test_euler_slerp_near_gimbal(self):
        # middle angles from 1e-5 degrees of 90: closer, matrix_to_euler
        # itself drops up to 2.3e-7 radians, a float32 epsilon branch
        rng = np.random.default_rng(9)
        euler0, euler1 = _near_gimbal(rng, 600)
        quat0 = euler_to_quaternion(euler0, 0)
        quat1 = euler_to_quaternion(euler1, 0)
        for weight in WEIGHTS + (rng.random(600),):
            got  = euler_to_matrix(euler_slerp(euler0, euler1, weight), 0)
            want = quaternion_to_matrix(_scipy_slerp(quat0, quat1, weight))
            # an euler angle near gimbal answers to eps / cos(middle angle)
            self.assertLess(_matrix_angle(got, want).max(), 1e-8, weight)

    def test_euler_slerp_away_from_gimbal(self):
        rng    = np.random.default_rng(10)
        euler0 = rng.uniform(-1.2, 1.2, (600, 3))
        step   = rng.normal(size=(600, 3)) * 10.0 ** rng.uniform(-12, 0, (600, 1))
        euler1 = euler0 + step
        quat0  = euler_to_quaternion(euler0, 0)
        quat1  = euler_to_quaternion(euler1, 0)
        for weight in WEIGHTS + (rng.random(600),):
            got  = euler_to_matrix(euler_slerp(euler0, euler1, weight), 0)
            want = quaternion_to_matrix(_scipy_slerp(quat0, quat1, weight))
            self.assertLess(_matrix_angle(got, want).max(), 2e-14, weight)

    def test_matrix_slerp_is_a_rotation(self):
        rng   = np.random.default_rng(11)
        quat0 = _random_quaternions(rng, 600)
        angles = np.concatenate(
            [10.0 ** rng.uniform(-15, -1, 300), rng.uniform(0, 3, 300)]
        )
        quat1   = _turned(rng, quat0, angles)
        matrix0 = quaternion_to_matrix(quat0)
        matrix1 = quaternion_to_matrix(quat1)
        for weight in WEIGHTS + (rng.random(600),):
            got  = matrix_slerp(matrix0, matrix1, weight)
            want = quaternion_to_matrix(_scipy_slerp(quat0, quat1, weight))
            rows = got[:, :3, :3]
            gram = np.einsum("nij,nkj->nik", rows, rows) - np.eye(3)
            self.assertLess(np.abs(gram).max(), 1e-14, weight)
            self.assertLess(_matrix_angle(got, want).max(), 1e-14, weight)

    def test_squad_with_its_controls_on_the_arc_is_the_slerp(self):
        rng   = np.random.default_rng(12)
        quat0 = _random_quaternions(rng, 400)
        quat1 = _turned(rng, quat0, 10.0 ** rng.uniform(-15, 0, 400))
        for weight in WEIGHTS + (rng.random(400),):
            got  = quaternion_squad(quat0, quat0, quat1, quat1, weight)
            want = _scipy_slerp(quat0, quat1, weight)
            self.assertLess(_quaternion_angle(got, want).max(), 2 * ROTATION_TOL)
            self.assertLess(np.abs(np.linalg.norm(got, axis=1) - 1.0).max(), 1e-15)
        np.testing.assert_array_equal(
            quaternion_squad(quat0, quat1, quat0, quat1, 0.0), quat0
        )


def _euler_grid():
    """euler pairs of a grid, in every rotate order for the two and the
    result: each grid angle paired with the grid rolled by one and by two
    places, and with itself a half turn and a whole turn about X away
    (both arcs as long, and one rotation, where axes0 == axes1), every
    (axes0, axes1, axes) in turn"""
    grid   = np.radians([-330.0, -190.0, -90.0, 0.0, 45.0, 180.0, 350.0])
    angles = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    euler0 = np.concatenate([angles] * 4)
    euler1 = np.concatenate(
        [
            np.roll(angles, 1, axis=0),
            np.roll(angles, 2, axis=0),
            angles + [np.pi, 0.0, 0.0],
            angles + [2.0 * np.pi, 0.0, 0.0],
        ]
    )
    orders = np.array(np.meshgrid(*[range(6)] * 3, indexing="ij")).reshape(3, -1).T
    count  = len(euler0)
    euler0 = np.tile(euler0, (len(orders), 1))
    euler1 = np.tile(euler1, (len(orders), 1))
    orders = np.repeat(orders, count, axis=0).astype(np.int32)
    return euler0, euler1, orders[:, 0], orders[:, 1], orders[:, 2]


def _as_converted(euler0, euler1, weight, axes0, axes1, axes):
    """euler_slerp before it picked an arc, on today's kernel: the two
    quaternions as converted from the angles, along whichever arc their
    signs gave. 2.0.0 took the same arcs, with its bits from before the
    precision fix"""
    quat0  = euler_to_quaternion(euler0, axes0)
    quat1  = euler_to_quaternion(euler1, axes1)
    weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), (len(quat0),))
    return quaternion_to_euler(_quaternion_slerp(quat0, quat1, weight.copy()), axes)


class TestEulerSlerpArc(unittest.TestCase):
    """euler_slerp takes the shorter arc, as quaternion_slerp does: before
    3.0.0 it slerped the quaternions as converted from the angles, and
    (0, 0, 0) to (350, 0, 0) degrees went the 350 degree way round"""

    def assertSameBits(self, got, want, where):
        nan = np.isnan(want)
        np.testing.assert_array_equal(np.isnan(got), nan, err_msg=where)
        np.testing.assert_array_equal(
            got[~nan].view(np.uint64), want[~nan].view(np.uint64), err_msg=where
        )

    def test_the_shorter_arc(self):
        zero = np.zeros((1, 3))
        cases = (
            # (euler0, euler1, the shorter arc's, the longer arc's), degrees
            (zero, [[350.0, 0.0, 0.0]], [[-5.0, 0.0, 0.0]], [[175.0, 0.0, 0.0]]),
            (zero, [[10.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]], [[-175.0, 0.0, 0.0]]),
            (zero, [[0.0, 0.0, -350.0]], [[0.0, 0.0, 5.0]], [[0.0, 0.0, -175.0]]),
        )
        for euler0, euler1, shorter, longer in cases:
            euler0 = np.radians(euler0)
            euler1 = np.radians(euler1)
            got    = np.degrees(euler_slerp(euler0, euler1, 0.5))
            np.testing.assert_allclose(got, shorter, rtol=0, atol=1e-12)
            got = np.degrees(euler_slerp(euler0, euler1, 0.5, shortest=False))
            np.testing.assert_allclose(got, longer, rtol=0, atol=1e-12)

        # Maya 2025's pairBlend, rotInterpolation quaternion, at weight 0.5
        euler0 = np.radians([[10.0, 20.0, 30.0]])
        euler1 = np.radians([[200.0, -150.0, 170.0]])
        maya   = [[10.397823958316941, -5.816739662881113, 11.53819611942565]]
        got    = np.degrees(euler_slerp(euler0, euler1, 0.5))
        np.testing.assert_allclose(got, maya, rtol=0, atol=1e-12)

    def test_one_rotation_stays_along_either_arc(self):
        # a whole turn apart: the shorter arc has no length, the longer one
        # no axis to turn about. Small angles only, with 2 pi added in
        # radians (every angle under 7.5 rad here): elsewhere, sometimes
        # even within a turn, the angles' own rounding can leave the two
        # quaternions further than about 2^-50 from opposite, and the
        # longer arc then turns a whole turn
        rng    = np.random.default_rng(25)
        euler0 = rng.uniform(-1.2, 1.2, (600, 3))
        euler1 = euler0 + 2.0 * np.pi * np.tile(np.eye(3), (200, 1))
        want   = euler_to_matrix(euler0, 0)
        for shortest in (True, False):
            for weight in WEIGHTS:
                got = euler_slerp(euler0, euler1, weight, shortest=shortest)
                got = euler_to_matrix(got, 0)
                self.assertLess(_matrix_angle(got, want).max(), 2e-14, (shortest, weight))

    def test_every_rotate_order_matches_quaternion_slerp(self):
        euler0, euler1, axes0, axes1, axes = _euler_grid()
        quat0 = euler_to_quaternion(euler0, axes0)
        quat1 = euler_to_quaternion(euler1, axes1)
        dot   = np.einsum("ij,ij->i", quat0, quat1)
        self.assertGreater((dot < 0.0).mean(), 0.3)
        self.assertGreater((dot > 0.0).mean(), 0.3)
        self.assertGreater((dot == 0.0).sum(), 5000)
        rng = np.random.default_rng(26)
        for weight in WEIGHTS + (rng.random(len(euler0)),):
            for shortest in (True, False):
                got = euler_slerp(euler0, euler1, weight, axes0, axes1, axes, shortest)
                want = quaternion_to_euler(
                    quaternion_slerp(quat0, quat1, weight, shortest), axes
                )
                self.assertSameBits(got, want, (np.ravel(weight)[:1], shortest))

    def test_each_arc_keeps_the_answers_it_gave_before(self):
        # where the quaternions as converted took the shorter arc, it is
        # still the default's, and where they took the longer one, that is
        # shortest=False's, bit for bit; at weights 0 and 1 the arc gives
        # the ends either way, as before
        euler0, euler1, axes0, axes1, axes = _euler_grid()
        rng   = np.random.default_rng(27)
        quat0 = euler_to_quaternion(euler0, axes0)
        quat1 = euler_to_quaternion(euler1, axes1)
        took  = np.einsum("ij,ij->i", quat0, quat1) < 0.0
        for weight in WEIGHTS + (rng.random(len(euler0)),):
            before  = _as_converted(euler0, euler1, weight, axes0, axes1, axes)
            shorter = euler_slerp(euler0, euler1, weight, axes0, axes1, axes)
            longer  = euler_slerp(euler0, euler1, weight, axes0, axes1, axes, False)
            where   = np.ravel(weight)[:1]
            self.assertSameBits(shorter[~took], before[~took], where)
            self.assertSameBits(longer[took], before[took], where)
            if np.ndim(weight) == 0 and weight in (0.0, 1.0):
                self.assertSameBits(shorter, before, where)
                self.assertSameBits(longer, before, where)


def _assert_rows_close(test, got, want, tol):
    """each row of ``got`` within ``tol`` of the length of ``want``'s"""
    off = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
    test.assertLess(off.max(), tol)


def _arc(length, angle):
    """``length`` along the direction ``angle`` radians from +X toward +Y"""
    return np.stack([length * np.cos(angle), length * np.sin(angle), 0 * angle], -1)


def _formula(vector0, vector1, weight):
    """vector_slerp's formula, the angle from the cross product"""
    cross  = np.linalg.norm(np.cross(vector0, vector1), axis=1)
    angle  = np.arctan2(cross, np.einsum("ij,ij->i", vector0, vector1))
    weight = np.broadcast_to(weight, angle.shape)
    a      = np.sin((1.0 - weight) * angle) / np.sin(angle)
    b      = np.sin(weight * angle) / np.sin(angle)
    return a[:, None] * vector0 + b[:, None] * vector1


def _planar(vector0, vector1, weight):
    """vector_slerp's formula from ``(x0, 0, 0)``, ``x0 > 0``, to ``(x1, y1,
    0)``, with the angle and the sine of the second's floats: parallel ones
    lerp, the formula's limit"""
    weight = np.broadcast_to(weight, (len(vector0),))
    angle  = np.arctan2(vector1[:, 1], vector1[:, 0])
    flat   = angle == 0.0
    length = np.hypot(vector1[:, 0], vector1[:, 1])
    sine   = np.where(flat, 1.0,          vector1[:, 1] / length)
    a      = np.where(flat, 1.0 - weight, np.sin((1.0 - weight) * angle) / sine)
    b      = np.where(flat, weight,       np.sin(weight * angle) / sine)
    return a[:, None] * vector0 + b[:, None] * vector1


def _two_long(rng, n):
    """n random vectors 2 long to the last bit: of two lengths an ulp apart,
    the formula adds ulp / (pi - angle) of it along the second"""
    vector  = _unit(rng.normal(size=(3 * n, 3))) * 2.0
    squares = vector[:, 0] * vector[:, 0] + vector[:, 1] * vector[:, 1]
    squares = squares + vector[:, 2] * vector[:, 2]
    return vector[squares == 4.0][:n]


class TestVectorSlerp(unittest.TestCase):
    def test_the_formula_on_random_vectors(self):
        # of any lengths, 1e-3 to 1e3
        rng     = np.random.default_rng(13)
        vector0 = rng.normal(size=(600, 3)) * 10.0 ** rng.uniform(-3, 3, (600, 1))
        vector1 = rng.normal(size=(600, 3)) * 10.0 ** rng.uniform(-3, 3, (600, 1))
        for weight in WEIGHTS[1:-1] + (rng.random(600),):
            got = vector_slerp(vector0, vector1, weight)
            _assert_rows_close(self, got, _formula(vector0, vector1, weight), 1e-14)

    def test_vectors_of_different_lengths(self):
        # the formula on the vectors as given: (0, 0, -5) to (10, 0, 0) half
        # way is (7.07, 0, -3.54), not 7.5 long at 45 degrees
        np.testing.assert_allclose(
            vector_slerp([0.0, 0.0, -5.0], [10.0, 0.0, 0.0], 0.5),
            [[50.0**0.5, 0.0, -(12.5**0.5)]],
            rtol = 1e-15,
            atol = 0,
        )
        np.testing.assert_allclose(
            vector_slerp([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], 0.5),
            [[0.5**0.5, 2.0**0.5, 0.0]],
            rtol = 1e-15,
            atol = 0,
        )

    def test_one_length_turns_at_a_steady_rate(self):
        # against scipy: turned about the normal of the two, by the weight's
        # share of the angle between them, the length kept
        rng     = np.random.default_rng(23)
        vector0 = _two_long(rng, 600)
        vector1 = _two_long(rng, 600)
        normal  = np.cross(vector0, vector1)
        angle = np.arctan2(
            np.linalg.norm(normal, axis=1), np.einsum("ij,ij->i", vector0, vector1)
        )
        for weight in WEIGHTS[1:-1] + (rng.random(600),):
            got  = vector_slerp(vector0, vector1, weight)
            turn = Rotation.from_rotvec(_unit(normal) * (angle * weight)[:, None])
            np.testing.assert_allclose(got, turn.apply(vector0), rtol=0, atol=4e-15)

    def test_nearly_parallel_vectors(self):
        # down to none, and continuous on the way: parallel ones lerp, the
        # formula's limit
        angles = np.array(
            [0.0, 1e-300, 1e-100, 1e-20, 1e-16, 1e-12, 1e-8, 1e-6, 1e-5, 1e-4, 0.1]
        )
        vector0 = _arc(np.full(len(angles), 2.0), 0 * angles)
        for length in (2.0, 5.0, 0.5):
            vector1 = _arc(np.full(len(angles), length), angles)
            for weight in WEIGHTS[1:-1]:
                got  = vector_slerp(vector0, vector1, weight)
                want = _planar(vector0, vector1, weight)
                np.testing.assert_allclose(got, want, rtol=1e-15, atol=1e-299)

        # down among the subnormals: the two directions' difference is 0,
        # and the formula is the lerp to the last bit
        angles  = np.array([1e-155, 1e-200, 1e-305, 1e-310, 1e-315])
        vector0 = _arc(np.full(5, 2.0), 0 * angles)
        vector1 = _arc(np.full(5, 5.0), angles)
        for weight in WEIGHTS[1:-1]:
            got  = vector_slerp(vector0, vector1, weight)
            want = vector0 + weight * (vector1 - vector0)
            _assert_rows_close(self, got, want, 1e-15)

    def test_nearly_opposite_vectors(self):
        # pi less an ever smaller angle, through +Y or -Y. of one length, to
        # the last bit, they turn along the arc and do not blow up on the
        # way: a length off by an ulp would add ulp / (pi - angle) to it
        less    = np.array([1e-2, 1e-4, 1e-8, 1e-12, 1e-14])
        vector0 = _arc(np.full(5, 2.0), 0 * less)
        for sign in (1.0, -1.0):
            vector1 = _arc(np.full(5, 2.0), sign * (np.pi - less))
            np.testing.assert_array_equal((vector1 * vector1).sum(axis=1), 4.0)
            angle = np.arctan2(vector1[:, 1], vector1[:, 0])
            for weight in WEIGHTS[1:-1]:
                got  = vector_slerp(vector0, vector1, weight)
                want = _arc(np.full(5, 2.0), weight * angle)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)

        # of different lengths the formula grows as 1 / (pi - angle). the
        # second is a length times (-1, +-less): its angle, pi less
        # atan(less), is no float's, so sin(angle) has to come from the
        # vectors, not from the angle. within 1e-9 of an end, the sine of
        # a weight's share of the angle, near pi, keeps eps / 1e-9 of
        # itself, in the kernel as in any float reference: the weights in
        # between
        less = np.array([1e-1, 1e-4, 1e-8, 1e-12, 1e-14])
        for length0, length1 in ((2.0, 5.0), (5.0, 2.0)):
            vector0 = _arc(np.full(5, length0), 0 * less)
            for sign in (1.0, -1.0):
                vector1 = np.stack([-1.0 + 0 * less, sign * less, 0 * less], -1)
                vector1 = vector1 * length1
                for weight in (0.25, 0.5, 0.75):
                    got  = vector_slerp(vector0, vector1, weight)
                    want = _planar(vector0, vector1, weight)
                    _assert_rows_close(self, got, want, 1e-15)

    def test_opposite_vectors_turn_toward_the_smallest_axis(self):
        # no plane to turn in, and the formula has no answer: the direction
        # turns toward a world axis and the length lerps, the formula's
        # limit between two of one length
        axes   = np.eye(3)
        toward = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        for weight in WEIGHTS[1:-1]:
            for sign in (1.0, -1.0):
                got   = vector_slerp(sign * axes, -3.0 * sign * axes, weight)
                angle = np.pi * weight
                want = (
                    sign * axes * np.cos(angle) + toward * np.sin(angle)
                ) * (1.0 + 2.0 * weight)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)

        # any direction, opposite to within rounding (a scale other than a
        # power of two): half way is at a right angle to both, in the plane
        # of the first and the axis along its smallest component
        rng      = np.random.default_rng(14)
        vector0  = rng.normal(size=(600, 3))
        half     = vector_slerp(vector0, -3.0 * vector0, 0.5)
        unit0    = _unit(vector0)
        smallest = np.eye(3)[np.argmin(np.abs(unit0), axis=1)]
        normal   = np.cross(unit0, smallest)
        length   = 2.0 * np.linalg.norm(vector0, axis=1)
        along    = np.einsum("ij,ij->i", _unit(half), unit0)
        np.testing.assert_allclose(along, 0, atol=1e-15)
        np.testing.assert_allclose(np.einsum("ij,ij->i", half, normal), 0, atol=1e-14)
        self.assertTrue((np.einsum("ij,ij->i", half, smallest) > 0.0).all())
        np.testing.assert_allclose(np.linalg.norm(half, axis=1), length, rtol=1e-15)

    def test_parallel_vectors_lerp(self):
        rng     = np.random.default_rng(15)
        vector0 = rng.normal(size=(600, 3))
        unit0   = _unit(vector0)
        for weight in WEIGHTS[1:-1]:
            got = vector_slerp(vector0, 3.0 * vector0, weight)
            np.testing.assert_allclose(_unit(got), unit0, rtol=0, atol=4e-16)
            _assert_rows_close(self, got, vector0 * (1.0 + 2.0 * weight), 1e-15)

    def test_vectors_too_short_or_too_long_to_square(self):
        # the formula scales with the vectors, under about 1e-146 long, where
        # the squares of the components lose bits as subnormals, and over
        # about 1e154, where they overflow
        rng     = np.random.default_rng(22)
        vector0 = rng.normal(size=(600, 3))
        vector1 = rng.normal(size=(600, 3)) * 2.0
        for scale in (1e-300, 1e-160, 1e160, 1e300):
            for weight in WEIGHTS[1:-1]:
                got  = vector_slerp(vector0 * scale, vector1 * scale, weight)
                want = vector_slerp(vector0, vector1, weight)
                _assert_rows_close(self, got / scale, want, 4e-15)

    def test_a_nan_or_an_inf_gives_nans(self):
        # between the ends: at them, each comes back as given
        vector = np.array([0.1, 0.2, 0.9])
        for bad in ([np.nan, 0.0, 1.0], [np.inf, 0.0, 1.0], [0.0, 0.0, np.nan]):
            for vector0, vector1 in ((bad, vector), (vector, bad)):
                got = vector_slerp(vector0, vector1, 0.5)
                self.assertTrue(np.isnan(got).all(), (vector0, vector1, got))
                got = vector_slerp(vector0, vector1, [0.0, 1.0])
                np.testing.assert_array_equal(got, [vector0, vector1])

    def test_a_zero_vector_lerps(self):
        rng    = np.random.default_rng(16)
        vector = rng.normal(size=(600, 3))
        zero   = np.zeros((600, 3))
        for weight in WEIGHTS:
            for vector0, vector1 in ((zero, vector), (vector, zero)):
                np.testing.assert_array_equal(
                    vector_slerp(vector0, vector1, weight),
                    vector0 + weight * (vector1 - vector0),
                )
        np.testing.assert_array_equal(vector_slerp(zero, zero, 0.5), zero)

    def test_weight_0_and_1_return_the_vectors(self):
        rng     = np.random.default_rng(17)
        vector0 = rng.normal(size=(600, 3))
        vector1 = np.concatenate([rng.normal(size=(300, 3)), -2.0 * vector0[:300]])
        np.testing.assert_array_equal(vector_slerp(vector0, vector1, 0.0), vector0)
        np.testing.assert_array_equal(vector_slerp(vector0, vector1, 1.0), vector1)


def _fixture(rng, n):
    """n of every kind of pair the slerps branch on, shuffled together:
    random, nearly and exactly parallel, nearly and exactly opposite, zero
    quaternions and vectors, ones too short or too long to square, nans and
    infs, euler angles near gimbal, and weights of 0, 1, between, outside
    and nan"""
    kinds = 6
    quat0 = _random_quaternions(rng, n)
    angle = np.concatenate(
        [rng.uniform(0, 3, n // kinds), 10.0 ** rng.uniform(-16, -2, n - n // kinds)]
    )
    quat1             = _turned(rng, quat0, rng.permutation(angle))
    quat1[0::kinds]   = -quat1[0::kinds]
    quat1[1::kinds]   = quat0[1::kinds]
    quat1[2::kinds]   = -quat0[2::kinds]

    vector0           = rng.normal(size=(n, 3))
    vector1           = quaternion_to_matrix(quat1)[:, 0, :3] * 3.0
    vector0[3::kinds] = quaternion_to_matrix(quat0)[3::kinds, 0, :3]
    vector1[1::kinds] = -2.0 * vector0[1::kinds]
    vector1[1::2 * kinds] += 1e-9 * rng.normal(size=vector1[1::2 * kinds].shape)
    vector1[2::kinds] = 2.0 * vector0[2::kinds]
    vector0[4::kinds] = 0.0
    vector1[5::kinds] = 0.0

    # the edges, on rows of their own: zero, too short or too long to square,
    # one of the two or both, a nan, an inf
    edges = 19
    for array in (quat0, vector0):
        array[0::edges]    = 0.0
        array[1::edges, 0] = np.nan
        array[2::edges] *= 1e-300
        array[3::edges] *= 1e-310
    for array in (quat1, vector1):
        array[4::edges]    = 0.0
        array[5::edges, 1] = np.inf
        array[6::edges] *= 1e300
    for array in (quat0, quat1, vector0, vector1):
        array[7::edges] *= 1e-170
        array[8::edges] *= 1e160

    euler0, euler1 = _near_gimbal(rng, n, low=1e-7)
    weight            = rng.random(n)
    weight[0::7]      = 0.0
    weight[1::7]      = 1.0
    weight[9::edges]  = -0.5
    weight[10::edges] = 1.5
    weight[11::edges] = np.nan
    return quat0, quat1, vector0, vector1, euler0, euler1, weight


class TestTwins(unittest.TestCase):
    """one thread answers every thread's bits on every kind of pair"""

    def assertSameBits(self, got, want, where):
        self.assertEqual(got.shape, want.shape, where)
        nan = np.isnan(want)
        np.testing.assert_array_equal(np.isnan(got), nan, err_msg=where)
        np.testing.assert_array_equal(
            got[~nan].view(np.uint64), want[~nan].view(np.uint64), err_msg=where
        )

    def test_both_sides_of_the_threshold(self):
        rng = np.random.default_rng(18)
        q0, q1, v0, v1, e0, e1, w = _fixture(rng, THRESHOLD + 1)
        m0, m1 = quaternion_to_matrix(q0), quaternion_to_matrix(q1)

        # euler pairs anywhere, along both arcs, in every rotate order, a nan
        # and an inf among them: the ones near gimbal above are a degree apart
        f0, f1, a0, a1, a = (array[::293][: THRESHOLD + 1] for array in _euler_grid())
        f0[::7]      = rng.uniform(-2.0 * np.pi, 2.0 * np.pi, f0[::7].shape)
        f0[1::19, 0] = np.nan
        f1[2::19, 1] = np.inf

        def wide(k, shortest):
            return transforms.euler_slerp(
                f0[:k], f1[:k], w[:k], a0[:k], a1[:k], a[:k], shortest
            )

        calls = {
            "quaternion_slerp": lambda k: transforms.quaternion_slerp(
                q0[:k], q1[:k], w[:k]
            ),
            "quaternion_slerp longer": lambda k: transforms.quaternion_slerp(
                q0[:k], q1[:k], w[:k], shortest=False
            ),
            "quaternion_squad": lambda k: transforms.quaternion_squad(
                q0[:k], q1[:k], q0[:k], q1[:k], w[:k]
            ),
            "matrix_slerp": lambda k: transforms.matrix_slerp(m0[:k], m1[:k], w[:k]),
            "matrix_interpolate": lambda k: transforms.matrix_interpolate(
                m0[:k] * 2.0, m1[:k], w[:k]
            ),
            "euler_slerp": lambda k: transforms.euler_slerp(e0[:k], e1[:k], w[:k]),
            "euler_slerp anywhere": lambda k: wide(k, True),
            "euler_slerp anywhere longer": lambda k: wide(k, False),
            "vector_slerp": lambda k: transforms.vector_slerp(v0[:k], v1[:k], w[:k]),
        }
        for name, call in calls.items():
            for size in (THRESHOLD - 1, THRESHOLD, THRESHOLD + 1):
                got = []
                for threshold in (0, sys.maxsize, THRESHOLD):
                    with mock.patch.object(_numba, "SINGLE_THREAD_BELOW", threshold):
                        with np.errstate(all="ignore"):
                            got.append(np.asarray(call(size)))
                threaded, one, picked = got
                where = "%s on %d" % (name, size)
                self.assertSameBits(one, threaded, where)
                self.assertSameBits(picked, threaded, where)


if __name__ == "__main__":
    unittest.main()
