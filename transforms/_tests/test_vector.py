import unittest

import numpy as np
from cgmath.transforms import (
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

EPSILON = np.finfo(np.float32).eps
# Adding a random seed so that we can reproduce the same random numbers and the test is deterministic
RANDOM_SEED     = 54345
RANDOM_SEED_TWO = 12345


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestVector(unittest.TestCase):
    def testRandom(self):
        V    = vector_random(10**6, normalize=True)
        mag  = np.einsum("...i,...i", V, V) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(allclose(mag, ones), True)
        self.assertEqual(not allclose(V[0], V[1]), True)

    def testNormalize(self):
        V    = vector_random(10**6, RANDOM_SEED, normalize=False)
        mag  = np.einsum("...i,...i", V, V) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(allclose(mag, ones), False)

        V   = vector_normalize(V)
        mag = np.einsum("...i,...i", V, V) ** 0.5
        self.assertEqual(allclose(mag, ones), True)

    def testMagnitude(self):
        V    = vector_random(10**6, RANDOM_SEED, normalize=False)
        mag0 = np.einsum("...i,...i", V, V) ** 0.5
        mag1 = vector_magnitude(V)
        self.assertEqual(allclose(mag0, mag1), True)

    def testLerp(self):
        V0 = vector_random(10**6, RANDOM_SEED, normalize=False)
        V1 = vector_random(10**6, RANDOM_SEED_TWO, normalize=False)
        w  = np.random.random(10**6)

        V  = vector_lerp(V0, V1, w)

        mag0  = vector_magnitude(V1 - V0)
        mag1  = vector_magnitude(V - V0)
        ratio = mag1 / mag0

        self.assertEqual(allclose(w, ratio), True)

    def testSlerp(self):
        V0   = vector_random(10**6, RANDOM_SEED, normalize=True)
        V1   = vector_random(10**6, RANDOM_SEED_TWO, normalize=True)
        w    = np.random.random(10**6) * 0.1

        V0   = np.array([1, 0, 0])
        V1   = np.array([0, 1, 0])
        w    = np.random.random(10**6)

        V    = vector_slerp(V0, V1, w)

        ang0 = np.arccos(np.clip(np.einsum("...i,...i", V1, V0), -1.0, 1.0))
        ang1 = np.arccos(np.clip(np.einsum("...i,...i", V, V0), -1.0, 1.0))

        self.assertEqual(allclose((ang1 / w), ang0), True)

    def testDot(self):
        V0   = vector_random(10**6, RANDOM_SEED, normalize=False)
        V1   = vector_random(10**6, RANDOM_SEED_TWO, normalize=True)

        dot0 = vector_dot(V0, V1)
        dot1 = np.einsum("...i,...i", V1, V0)

        self.assertEqual(allclose(dot0, dot1), True)

    def testCross(self):
        V0 = vector_random(10**6, RANDOM_SEED, normalize=True)
        V1 = vector_random(10**6, RANDOM_SEED_TWO, normalize=True)
        V0 = np.random.random((10**6, 3))
        V1 = np.random.random((10**6, 3))

        cross0 = vector_cross(V0, V1)
        cross1 = np.cross(V0, V1)

        self.assertEqual(allclose(cross0, cross1), True)

    def testAngle(self):
        V0   = vector_random(500, normalize=True)
        V1   = vector_random(500, normalize=True)

        V0   = [0.84041616, -0.38676315, 0.37962473]
        V1   = [-0.07168087, 0.24989775, -0.96561533]

        ang0 = vector_angle(V0, V1)
        ang1 = np.arccos(np.clip(np.einsum("...i,...i", V0, V1), -1.0, 1.0))

        self.assertEqual(allclose(ang0, ang1), True)

    def testToEuler(self):
        x  = np.array([1, 0, 0])
        y  = np.array([0, 1, 0])
        eu = vector_to_euler(x, y)
        self.assertEqual(allclose(eu, [0, 0, 0]), True)

    def testToMatrix(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        M = vector_to_matrix(x, y)

        self.assertEqual(
            allclose(
                M,
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
            ),
            True,
        )

    def testToQuaternion(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        Q = vector_to_quaternion(x, y)

        self.assertEqual(allclose(Q, [[0.0, 0.0, 0.0, 1.0]]), True)

    def testArcToEuler(self):
        x     = np.array([1, 0, 0])
        y     = np.array([0, 1, 0])
        angle = np.degrees(vector_arc_to_euler(x, y))

        self.assertEqual(allclose(angle, [[0, 0, 90]]), True)

    def testArcToMatrix(self):
        x = np.array([1, 0, 0])
        y = np.array([1, 0, 0])
        M = vector_arc_to_matrix(x, y)

        self.assertEqual(
            allclose(
                M,
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
            ),
            True,
        )

    def testArcToQuaternion(self):
        x = np.array([1, 0, 0])
        y = np.array([1, 0, 0])
        Q = vector_arc_to_quaternion(x, y)

        self.assertEqual(allclose(Q, [[0.0, 0.0, 0.0, 1.0]]), True)