"""Regression tests for the input levelling rules.

Every function taking more than one batched argument routes its inputs through
:func:`cgmath.transforms.utils._match_depth`, which follows NumPy broadcasting rules:
an input is accepted when its length matches the longest one, or is exactly 1.

These tests exist because the four cases in :class:`TestSilentPaths` all used
to return plausible-looking numbers instead of raising -- three of them by
reading past the end of an array -- and nothing else in the suite exercises a
mismatched length, so none of them were caught.
"""

import unittest

import numpy as np
from cgmath.transforms import (
    XYZ,
    ZYX,
    axis_angle_to_quaternion,
    euler_filter,
    euler_random,
    euler_slerp,
    euler_to_matrix,
    matrix_identity,
    matrix_interpolate,
    matrix_multiply,
    matrix_point_multiply,
    matrix_random,
    matrix_to_euler,
    matrix_weighted_rotational,
    matrix_weighted_transformation,
    quaternion_multiply,
    quaternion_random,
    quaternion_slerp,
    vector_angle,
    vector_cross,
    vector_dot,
    vector_random,
    vector_slerp,
)
from cgmath.transforms.utils import _match_depth, _set_dimension

EPSILON         = np.finfo(np.float32).eps
RANDOM_SEED     = 54345
RANDOM_SEED_TWO = 12345


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=atol)


class TestSilentPaths(unittest.TestCase):
    """The four calls that used to succeed with garbage."""

    def testVectorAngleOutOfBounds(self):
        # never called _match_depth, so the kernel read past the end of the
        # second argument and returned len(vector0) plausible angles
        V0 = vector_random(5, RANDOM_SEED)
        V1 = vector_random(2, RANDOM_SEED_TWO)
        with self.assertRaises(ValueError):
            vector_angle(V0, V1)

    def testEmptyInputOutOfBounds(self):
        # a length-0 input was skipped by the levelling guard, so the kernel
        # read uninitialised memory for the whole batch
        V0 = vector_random(5, RANDOM_SEED)
        with self.assertRaises(ValueError):
            vector_dot(V0, np.zeros((0, 3)))

    def testEulerSlerpSecondArgumentOutOfBounds(self):
        # euler1 was handed to _match_depth as euler0, so it was never levelled
        EA0 = euler_random(5, RANDOM_SEED)
        EA1 = euler_random(2, RANDOM_SEED_TWO)
        with self.assertRaises(ValueError):
            euler_slerp(EA0, EA1, 0.5)

    def testWeightedRotationalTruncation(self):
        # zip() silently dropped the surplus matrices
        matrices = [m[:3, :3] for m in matrix_random(5, RANDOM_SEED)]
        with self.assertRaises(ValueError):
            matrix_weighted_rotational(matrices, np.array([0.5, 0.5]))


class TestMismatchRaises(unittest.TestCase):
    """N against M, where M is neither N nor 1, is a mistake everywhere."""

    def testVector(self):
        V0 = vector_random(5, RANDOM_SEED)
        V1 = vector_random(2, RANDOM_SEED_TWO)
        for call in (
            lambda: vector_dot(V0, V1),
            lambda: vector_cross(V0, V1),
            lambda: vector_slerp(V0, V1, 0.5),
            lambda: vector_slerp(V0, V0, np.array([0.0, 1.0])),
        ):
            with self.assertRaises(ValueError):
                call()

    def testMatrix(self):
        M0 = matrix_random(5, RANDOM_SEED)
        M1 = matrix_random(2, RANDOM_SEED_TWO)
        for call in (
            lambda: matrix_multiply(M0, M1),
            lambda: matrix_point_multiply(vector_random(5, RANDOM_SEED), M1),
            lambda: matrix_interpolate(M0, M1, 0.5),
        ):
            with self.assertRaises(ValueError):
                call()

    def testQuaternion(self):
        Q0 = quaternion_random(5, RANDOM_SEED)
        Q1 = quaternion_random(2, RANDOM_SEED_TWO)
        for call in (
            lambda: quaternion_multiply(Q0, Q1),
            lambda: quaternion_slerp(Q0, Q1, 0.5),
        ):
            with self.assertRaises(ValueError):
                call()

    def testRotateOrderList(self):
        # the sharpest case: a wrong rotate order looks almost right
        with self.assertRaises(ValueError):
            matrix_to_euler(matrix_random(10, RANDOM_SEED), [XYZ, ZYX, XYZ])

    def testAxisAngle(self):
        with self.assertRaises(ValueError):
            axis_angle_to_quaternion(vector_random(5, RANDOM_SEED), [0.1, 0.2])

    def testEulerFilterAxes(self):
        # euler_filter broadcasts axes against the curve count rather than the
        # frame count, so it checks its own shapes -- but must raise alike
        with self.assertRaises(ValueError):
            euler_filter(euler_random(5, RANDOM_SEED), [XYZ, ZYX])

    def testWeightedTransformation(self):
        with self.assertRaises(ValueError):
            matrix_weighted_transformation(matrix_random(5, RANDOM_SEED), [0.2, 0.8])

    def testMessageNamesBothLengths(self):
        V0 = vector_random(5, RANDOM_SEED)
        V1 = vector_random(2, RANDOM_SEED_TWO)
        with self.assertRaises(ValueError) as caught:
            vector_dot(V0, V1)

        message = str(caught.exception)
        self.assertIn("length mismatch", message)
        self.assertIn("[5, 2]", message)


class TestLegalShapes(unittest.TestCase):
    """Equal lengths, a length of 1, and bare unbatched input all still work."""

    def testEqualLengths(self):
        V0 = vector_random(5, RANDOM_SEED)
        V1 = vector_random(5, RANDOM_SEED_TWO)
        self.assertEqual(vector_dot(V0, V1).shape, (5,))

    def testOneAgainstMany(self):
        V0 = vector_random(5, RANDOM_SEED)
        V1 = vector_random(1, RANDOM_SEED_TWO)

        self.assertEqual(vector_dot(V0, V1).shape,   (5,))
        self.assertEqual(vector_dot(V1, V0).shape,   (5,))
        self.assertEqual(vector_angle(V0, V1).shape, (5,))

    def testBareInputIsPromoted(self):
        V0 = vector_random(5, RANDOM_SEED)

        self.assertEqual(vector_dot([1.0, 0.0, 0.0], V0).shape, (5,))
        self.assertEqual(matrix_point_multiply(V0, np.eye(4)).shape, (5, 3))
        self.assertEqual(
            matrix_to_euler(matrix_random(5, RANDOM_SEED), XYZ).shape, (5, 3)
        )

    def testScalarRotateOrderAndWeight(self):
        Q0 = quaternion_random(5, RANDOM_SEED)
        Q1 = quaternion_random(5, RANDOM_SEED_TWO)

        self.assertEqual(quaternion_slerp(Q0, Q1, 0.5).shape, (5, 4))
        self.assertEqual(quaternion_slerp(Q0, Q1, np.linspace(0, 1, 5)).shape, (5, 4))

    def testShortSideMayBeEitherArgument(self):
        # one matrix decoded under six rotate orders -- here the length-1
        # input is the matrix, not the axes
        M = euler_to_matrix([0.1, 0.2, 0.3], XYZ)
        self.assertEqual(matrix_to_euler(M, [0, 1, 2, 3, 4, 5]).shape, (6, 3))

    def testAllEmptyIsLegal(self):
        empty = np.zeros((0, 3))
        self.assertEqual(vector_dot(empty, empty).shape, (0,))


class TestBroadcastSemantics(unittest.TestCase):
    """A length-1 input must be a zero-copy view *and* give the right answer."""

    def testExpansionIsZeroCopy(self):
        V0 = vector_random(10**5, RANDOM_SEED)
        V1 = vector_random(1, RANDOM_SEED_TWO)
        levelled0, levelled1 = _match_depth(V0, V1)

        self.assertEqual(levelled1.shape, (10**5, 3))
        self.assertTrue(np.shares_memory(levelled1, V1))
        self.assertIs(levelled0, V0)

    def testExpansionMatchesAMaterialisedCopy(self):
        V0       = vector_random(5, RANDOM_SEED)
        V1       = vector_random(1, RANDOM_SEED_TWO)
        repeated = np.repeat(V1, 5, axis=0)

        self.assertTrue(allclose(vector_dot(V0, V1), vector_dot(V0, repeated)))
        self.assertTrue(allclose(vector_cross(V0, V1), vector_cross(V0, repeated)))
        self.assertTrue(
            allclose(vector_slerp(V0, V1, 0.5), vector_slerp(V0, repeated, 0.5))
        )

    def testScalarWeightMatchesAFullArray(self):
        Q0 = quaternion_random(5, RANDOM_SEED)
        Q1 = quaternion_random(5, RANDOM_SEED_TWO)

        self.assertTrue(
            allclose(
                quaternion_slerp(Q0, Q1, 0.25),
                quaternion_slerp(Q0, Q1, np.full(5, 0.25)),
            )
        )

    def testEveryArgumentPositionAcceptsLengthOne(self):
        # a levelled length-1 input is a read-only, non-contiguous view. Any
        # kernel carrying an eager ``@njit("float64[:,:](...)")`` signature
        # rejects that outright, so sweep every batched argument of every
        # function rather than trusting a sample. ``_vector_cross`` shipped
        # with exactly such a signature.
        from cgmath import transforms as tr
        from cgmath.transforms import (
            quaternion_intermediate,
            quaternion_nlerp,
            quaternion_squad,
        )

        size = 5
        make = {
            "q": lambda n, s: tr.quaternion_random(n, s),
            "m": lambda n, s: tr.matrix_random(n, s, random_position=True),
            "e": lambda n, s: tr.euler_random(n, s),
            "v": lambda n, s: tr.vector_random(n, s) + 1.0,
            "w": lambda n, s: np.linspace(0.15, 0.85, n),
            "a": lambda n, s: np.full(n, s % 6, dtype=np.int64),
            "x": lambda n, s: np.full(n, s % 3, dtype=np.int64),
        }
        specs = [
            (tr.quaternion_slerp, "qqw"),
            (tr.quaternion_dot, "qq"),
            (tr.quaternion_multiply, "qq"),
            (tr.quaternion_add, "qq"),
            (tr.quaternion_sub, "qq"),
            (tr.quaternion_to_euler, "qa"),
            (quaternion_nlerp, "qqw"),
            (quaternion_intermediate, "qqq"),
            (quaternion_squad, "qqqqw"),
            (tr.matrix_to_euler, "ma"),
            (tr.matrix_point_multiply, "vm"),
            (tr.matrix_multiply, "mm"),
            (tr.matrix_slerp, "mmw"),
            (tr.matrix_interpolate, "mmw"),
            (tr.matrix_local, "mm"),
            (tr.matrix_delta, "mm"),
            (tr.euler_to_matrix, "ea"),
            (tr.euler_to_quaternion, "ea"),
            (tr.euler_reorder, "eaa"),
            (tr.euler_slerp, "eewaaa"),
            (tr.axis_angle_to_quaternion, "vw"),
            (tr.axis_angle_to_matrix, "vw"),
            (tr.axis_angle_to_euler, "vwa"),
            (tr.vector_to_matrix, "vvxx"),
            (tr.vector_to_quaternion, "vvxx"),
            (tr.vector_to_euler, "vvxxa"),
            (tr.vector_cross, "vv"),
            (tr.vector_dot, "vv"),
            (tr.vector_slerp, "vvw"),
            (tr.vector_lerp, "vvw"),
            (tr.vector_arc_to_quaternion, "vv"),
            (tr.vector_arc_to_matrix, "vv"),
            (tr.vector_arc_to_euler, "vva"),
            (tr.vector_angle, "vv"),
        ]

        def astuple(result):
            return result if isinstance(result, tuple) else (result,)

        for function, kinds in specs:
            for position in range(len(kinds)):
                short, full = [], []
                for index, kind in enumerate(kinds):
                    length = 1 if index == position else size
                    value  = make[kind](length, index + 1)
                    short.append(value)
                    full.append(
                        np.repeat(value, size, axis=0) if length == 1 else value
                    )

                where = "%s argument %d (%s)" % (
                    function.__name__,
                    position,
                    kinds[position],
                )
                got  = astuple(function(*short))
                want = astuple(function(*full))

                self.assertEqual(len(got), len(want), where)
                for expanded, materialised in zip(got, want):
                    self.assertEqual(
                        np.shape(expanded), np.shape(materialised), where
                    )
                    self.assertTrue(
                        np.allclose(
                            expanded, materialised, atol=1e-9, equal_nan=True
                        ),
                        where,
                    )

    def testMatchDepthRules(self):
        five = _set_dimension(vector_random(5, RANDOM_SEED),     2)
        one  = _set_dimension(vector_random(1, RANDOM_SEED_TWO), 2)
        two  = _set_dimension(vector_random(2, RANDOM_SEED_TWO), 2)

        self.assertEqual([len(d) for d in _match_depth(five, one)],  [5, 5])
        self.assertEqual([len(d) for d in _match_depth(one, five)],  [5, 5])
        self.assertEqual([len(d) for d in _match_depth(five, five)], [5, 5])
        self.assertEqual([len(d) for d in _match_depth(one, one)],   [1, 1])

        with self.assertRaises(ValueError):
            _match_depth(five, two)
        with self.assertRaises(ValueError):
            _match_depth(five, np.zeros((0, 3)))


class TestNoInputMutation(unittest.TestCase):
    """Levelling hands back read-only views, so nothing may write to inputs."""

    def testMatrixInterpolateLeavesArgumentsAlone(self):
        scaled = euler_to_matrix([0.0, 0.0, np.radians(45)], XYZ)
        scaled[:, :3, :3] *= 3.0
        before = scaled.copy()

        matrix_interpolate(scaled, matrix_identity(1), 0.5)
        self.assertTrue(allclose(scaled, before))

    def testMatrixInterpolateAcceptsALengthOneArgument(self):
        # the length-1 side arrives as a read-only broadcast view
        M = matrix_random(5, RANDOM_SEED)
        self.assertEqual(
            matrix_interpolate(M, matrix_identity(1), 0.5).shape, (5, 4, 4)
        )


if __name__ == "__main__":
    unittest.main()
