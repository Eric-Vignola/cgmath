"""Cross-conversion coverage for euler / quaternion / matrix across every
rotate order, pinned against Autodesk Maya.

The golden values in :data:`MAYA_EULER_TO_QUATERNION` were produced by
``maya.api.OpenMaya.MEulerRotation.asQuaternion()`` under Maya 2026, so the
parity is asserted without needing Maya installed to run the suite.
"""

import unittest

import numpy as np
from cgmath.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_random,
    euler_reorder,
    euler_to_matrix,
    euler_to_quaternion,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_random,
    quaternion_slerp,
    quaternion_to_euler,
    quaternion_to_matrix,
)
from cgmath.transforms import quaternion_nlerp

EPSILON = np.finfo(np.float32).eps

# these kernels are njit(fastmath=True); ~1e-5 degrees is the achievable floor
ANGLE_TOL_DEGREES = 1e-4

RANDOM_SEED = 12345
ORDERS      = (0, 1, 2, 3, 4, 5)  # xyz yzx zxy xzy yxz zyx
ORDER_NAMES = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")

# Generated from Maya 2026 via maya.api.OpenMaya MEulerRotation.asQuaternion()
# euler degrees -> {rotate_order: (qx, qy, qz, qw)}
MAYA_EULER_TO_QUATERNION = {
    (10.0, 20.0, 30.0): {
        0: (
            0.03813457647485015,
            0.18930785741199999,
            0.2392983377447303,
            0.9515485246437885,
        ),
        1: (
            0.03813457647485015,
            0.14487812541736916,
            0.2685358227515692,
            0.9515485246437885,
        ),
        2: (
            0.12767944069578063,
            0.14487812541736916,
            0.2392983377447303,
            0.9515485246437886,
        ),
        3: (
            0.12767944069578063,
            0.189307857412,
            0.2392983377447303,
            0.9437143641474891,
        ),
        4: (
            0.03813457647485015,
            0.18930785741199999,
            0.2685358227515692,
            0.943714364147489,
        ),
        5: (
            0.12767944069578063,
            0.14487812541736916,
            0.2685358227515692,
            0.943714364147489,
        ),
    },
    (90.0, 0.0, 0.0): {
        0: (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        1: (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        2: (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        3: (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        4: (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        5: (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
    },
    (0.0, 90.0, 0.0): {
        0: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        1: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        2: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        3: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        4: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        5: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
    },
    (0.0, 0.0, 90.0): {
        0: (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        1: (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        2: (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        3: (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        4: (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        5: (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
    },
    (-135.0, 47.0, 88.0): {
        0: (
            -0.7154639869440903,
            -0.47878399718998066,
            0.5087879561032558,
            -0.003461667968316595,
        ),
        1: (
            -0.7154639869440902,
            0.6983188250832918,
            -0.021216002810019258,
            -0.0034616679683166507,
        ),
        2: (
            -0.5034616679683167,
            0.6983188250832918,
            0.5087879561032558,
            -0.0034616679683166507,
        ),
        3: (
            -0.5034616679683167,
            -0.47878399718998066,
            0.5087879561032558,
            0.5083572057575427,
        ),
        4: (
            -0.7154639869440903,
            -0.47878399718998066,
            -0.021216002810019258,
            0.5083572057575427,
        ),
        5: (
            -0.5034616679683166,
            0.6983188250832918,
            -0.021216002810019258,
            0.5083572057575427,
        ),
    },
    (170.0, -80.0, 25.0): {
        0: (
            0.7571657153164072,
            0.1104767639859647,
            0.6396135839763396,
            -0.07341271934429819,
        ),
        1: (
            0.7571657153164072,
            -0.21986610696958336,
            -0.6107123276911219,
            -0.07341271934429816,
        ),
        2: (
            0.7329146818269211,
            -0.21986610696958336,
            0.6396135839763395,
            -0.07341271934429817,
        ),
        3: (
            0.7329146818269211,
            0.11047676398596468,
            0.6396135839763395,
            0.203777861836546,
        ),
        4: (
            0.7571657153164072,
            0.1104767639859647,
            -0.610712327691122,
            0.20377786183654598,
        ),
        5: (
            0.7329146818269211,
            -0.21986610696958336,
            -0.6107123276911219,
            0.20377786183654595,
        ),
    },
}


def geodesic_degrees(quat0, quat1):
    """Largest angle between two stacks of quaternions, ignoring q/-q sign."""
    a   = np.asarray(quat0, dtype=float).reshape(-1, 4)
    b   = np.asarray(quat1, dtype=float).reshape(-1, 4)
    a   = a / np.linalg.norm(a, axis=1, keepdims=True)
    b   = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=1)).clip(0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)).max())


class TestMayaParity(unittest.TestCase):
    """euler_to_quaternion once wrote its imaginary components to slots 0,1,2
    instead of the order-dependent slots i,j,k, which is correct only for xyz.
    """

    def test_euler_to_quaternion_matches_maya(self):
        for euler_degrees, per_order in MAYA_EULER_TO_QUATERNION.items():
            radians = np.radians([euler_degrees])
            for order, expected in per_order.items():
                got   = euler_to_quaternion(radians, axes=order)
                error = geodesic_degrees(got, [expected])
                self.assertLess(
                    error,
                    ANGLE_TOL_DEGREES,
                    f"{euler_degrees} order {ORDER_NAMES[order]}: "
                    f"{np.asarray(got)[0]} != {expected} ({error} deg)",
                )

    def test_single_axis_rotation_is_order_independent(self):
        """With two angles at zero the rotate order cannot change the result."""
        for axis, expected in enumerate(
            (
                (0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
                (0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
                (0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
            )
        ):
            euler          = np.zeros((1, 3))
            euler[0, axis] = np.radians(90.0)
            for order in ORDERS:
                got = euler_to_quaternion(euler, axes=order)
                self.assertLess(
                    geodesic_degrees(got, [expected]),
                    ANGLE_TOL_DEGREES,
                    f"90 deg about axis {axis}, order {ORDER_NAMES[order]}",
                )


class TestConversionConsistency(unittest.TestCase):
    def setUp(self):
        self.euler      = euler_random(2000, RANDOM_SEED)
        self.quaternion = quaternion_random(2000, RANDOM_SEED)

    def test_euler_to_quaternion_agrees_with_matrix_path(self):
        for order in ORDERS:
            direct = euler_to_quaternion(self.euler, axes=order)
            viamat = matrix_to_quaternion(euler_to_matrix(self.euler, axes=order))
            self.assertLess(
                geodesic_degrees(direct, viamat),
                ANGLE_TOL_DEGREES,
                f"order {ORDER_NAMES[order]}",
            )

    def test_quaternion_to_euler_agrees_with_matrix_path(self):
        for order in ORDERS:
            direct = quaternion_to_euler(self.quaternion, axes=order)
            viamat = matrix_to_euler(quaternion_to_matrix(self.quaternion), axes=order)
            self.assertLess(
                geodesic_degrees(
                    euler_to_quaternion(direct, axes=order),
                    euler_to_quaternion(viamat, axes=order),
                ),
                ANGLE_TOL_DEGREES,
                f"order {ORDER_NAMES[order]}",
            )

    def test_roundtrip_euler_quaternion_euler(self):
        for order in ORDERS:
            quat = euler_to_quaternion(self.euler, axes=order)
            back = quaternion_to_euler(quat, axes=order)
            self.assertLess(
                geodesic_degrees(quat, euler_to_quaternion(back, axes=order)),
                ANGLE_TOL_DEGREES,
                f"order {ORDER_NAMES[order]}",
            )

    def test_roundtrip_euler_matrix_euler(self):
        for order in ORDERS:
            matrix = euler_to_matrix(self.euler, axes=order)
            back   = matrix_to_euler(matrix, axes=order)
            self.assertTrue(
                np.allclose(matrix, euler_to_matrix(back, axes=order), atol=EPSILON),
                f"order {ORDER_NAMES[order]}",
            )

    def test_roundtrip_quaternion_matrix_quaternion(self):
        back = matrix_to_quaternion(quaternion_to_matrix(self.quaternion))
        self.assertLess(geodesic_degrees(self.quaternion, back), ANGLE_TOL_DEGREES)

    def test_axis_angle_agrees_with_matrix_path(self):
        rng  = np.random.default_rng(RANDOM_SEED)
        axis = rng.normal(size=(2000, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        angle  = rng.uniform(-np.pi, np.pi, size=2000)

        quat   = axis_angle_to_quaternion(axis, angle)
        matrix = axis_angle_to_matrix(axis, angle)
        self.assertLess(
            geodesic_degrees(quat, matrix_to_quaternion(matrix)), ANGLE_TOL_DEGREES
        )


class TestEulerReorder(unittest.TestCase):
    def setUp(self):
        self.euler = euler_random(1000, RANDOM_SEED)

    def test_all_order_pairs_preserve_the_rotation(self):
        for source in ORDERS:
            for target in ORDERS:
                if source == target:
                    continue
                moved = euler_reorder(self.euler, from_axes=source, to_axes=target)
                self.assertLess(
                    geodesic_degrees(
                        euler_to_quaternion(self.euler, axes=source),
                        euler_to_quaternion(moved, axes=target),
                    ),
                    ANGLE_TOL_DEGREES,
                    f"{ORDER_NAMES[source]} -> {ORDER_NAMES[target]}",
                )

    def test_reorder_roundtrip(self):
        for target in ORDERS:
            if target == 0:
                continue
            moved = euler_reorder(self.euler, from_axes=0, to_axes=target)
            back  = euler_reorder(moved, from_axes=target, to_axes=0)
            self.assertTrue(
                np.allclose(
                    euler_to_matrix(self.euler, axes=0),
                    euler_to_matrix(back, axes=0),
                    atol=EPSILON,
                ),
                f"xyz -> {ORDER_NAMES[target]} -> xyz",
            )


class TestInterpolationDoesNotMutateInputs(unittest.TestCase):
    """These wrappers used to flip ``quat1`` in place, and ``_set_dimension``
    hands back the caller's own array when it is already ``(N, 4)`` float.
    """

    def setUp(self):
        self.quat0 = quaternion_random(500, RANDOM_SEED)
        self.quat1 = quaternion_random(500, RANDOM_SEED + 1)
        # a mix of hemispheres, so the flip branch is exercised
        self.assertTrue(
            (np.einsum("...i,...i", self.quat0, self.quat1) < 0.0).any(),
            "fixture should contain opposite-hemisphere pairs",
        )

    def _assert_untouched(self, function, **kwargs):
        before0 = self.quat0.copy()
        before1 = self.quat1.copy()
        function(self.quat0, self.quat1, weight=0.5, **kwargs)
        self.assertTrue(np.array_equal(before0, self.quat0), "quat0 was modified")
        self.assertTrue(np.array_equal(before1, self.quat1), "quat1 was modified")

    def test_quaternion_slerp_shortest(self):
        self._assert_untouched(quaternion_slerp, shortest=True)

    def test_quaternion_slerp_longest(self):
        self._assert_untouched(quaternion_slerp, shortest=False)

    def test_quaternion_nlerp(self):
        self._assert_untouched(quaternion_nlerp, shortest=True)

    def test_repeated_calls_are_stable(self):
        """The mutation only showed up as drift across successive calls."""
        first = np.asarray(
            quaternion_slerp(self.quat0, self.quat1, weight=0.5, shortest=True)
        ).copy()
        for _ in range(5):
            again = np.asarray(
                quaternion_slerp(self.quat0, self.quat1, weight=0.5, shortest=True)
            )
            self.assertTrue(np.allclose(first, again, atol=EPSILON))


class TestSlerpBehaviour(unittest.TestCase):
    def setUp(self):
        self.quat0 = quaternion_random(500, RANDOM_SEED)
        self.quat1 = quaternion_random(500, RANDOM_SEED + 1)

    def test_endpoints(self):
        self.assertLess(
            geodesic_degrees(
                quaternion_slerp(self.quat0, self.quat1, weight=0.0), self.quat0
            ),
            ANGLE_TOL_DEGREES,
        )
        self.assertLess(
            geodesic_degrees(
                quaternion_slerp(self.quat0, self.quat1, weight=1.0), self.quat1
            ),
            ANGLE_TOL_DEGREES,
        )

    def test_shortest_arc_is_never_longer_than_the_other(self):
        short    = quaternion_slerp(self.quat0, self.quat1, weight=0.5, shortest=True)
        other    = quaternion_slerp(self.quat0, self.quat1, weight=0.5, shortest=False)
        to_short = geodesic_degrees(self.quat0, short)
        to_other = geodesic_degrees(self.quat0, other)
        self.assertLessEqual(to_short, to_other + ANGLE_TOL_DEGREES)

    def test_symmetry(self):
        forward  = quaternion_slerp(self.quat0, self.quat1, weight=0.25, shortest=True)
        backward = quaternion_slerp(self.quat1, self.quat0, weight=0.75, shortest=True)
        self.assertLess(geodesic_degrees(forward, backward), ANGLE_TOL_DEGREES)


if __name__ == "__main__":
    unittest.main()