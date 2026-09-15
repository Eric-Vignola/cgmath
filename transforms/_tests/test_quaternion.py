import unittest

import numpy as np
from cgmath.transforms import (
    euler_to_matrix,
    quaternion_conjugate,
    quaternion_dot,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_negate,
    quaternion_normalize,
    quaternion_random,
    quaternion_slerp,
    quaternion_sub,
    quaternion_to_euler,
    quaternion_to_matrix,
)

EPSILON = np.finfo(np.float32).eps
# Adding a random seed so that we can reproduce the same random numbers and the test is deterministic
RANDOM_SEED     = 12321
RANDOM_SEED_TWO = 54345


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestQuaternion(unittest.TestCase):
    def testRandom(self):
        Q = quaternion_random(10**6, RANDOM_SEED)
        M = quaternion_to_matrix(Q)

        x    = np.einsum("...i,...i", M[:, 0], M[:, 0]) ** 0.5
        y    = np.einsum("...i,...i", M[:, 1], M[:, 1]) ** 0.5
        z    = np.einsum("...i,...i", M[:, 2], M[:, 2]) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(allclose(x, ones), True)
        self.assertEqual(allclose(y, ones), True)
        self.assertEqual(allclose(z, ones), True)

    def testToEuler(self):
        Q  = quaternion_random(10**6, RANDOM_SEED)
        M  = quaternion_to_matrix(Q)

        ea = quaternion_to_euler(Q)
        M_ = euler_to_matrix(ea)
        self.assertEqual(allclose(M, M_), True)

    def testToMatrix(self):
        Q = quaternion_random(10**6, RANDOM_SEED)
        M = quaternion_to_matrix(Q)

        x    = np.einsum("...i,...i", M[:, 0], M[:, 0]) ** 0.5
        y    = np.einsum("...i,...i", M[:, 1], M[:, 1]) ** 0.5
        z    = np.einsum("...i,...i", M[:, 2], M[:, 2]) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(allclose(x, ones), True)
        self.assertEqual(allclose(y, ones), True)
        self.assertEqual(allclose(z, ones), True)

    def testSlerp(self):
        Q0 = quaternion_random(10**6, RANDOM_SEED)
        Q1 = quaternion_random(10**6, RANDOM_SEED_TWO)
        w  = np.random.random(10**6)

        # slerp ea0 to ea1
        forward = quaternion_slerp(Q0, Q1, w)

        # do the opposite
        backward = quaternion_slerp(Q1, Q0, 1 - w)

        self.assertEqual(
            allclose(quaternion_to_matrix(forward), quaternion_to_matrix(backward)),
            True,
        )

    def testNormalize(self):
        Q    = quaternion_random(10**6, RANDOM_SEED) * 0.1
        Q_   = quaternion_normalize(Q)

        ones = np.ones(10**6)

        mag  = np.einsum("...i,...i", Q_, Q_) ** 0.5
        self.assertEqual(allclose(mag, ones), True)

    def testNegate(self):
        Q  = quaternion_random(10**6, RANDOM_SEED)
        Q_ = quaternion_negate(Q)

        self.assertEqual(allclose(Q_, -Q), True)

    def testConjugate(self):
        Q    = quaternion_random(10**6, RANDOM_SEED)
        Q_   = quaternion_conjugate(Q)

        test = np.array(Q)
        test[:, :3] *= -1

        self.assertEqual(allclose(Q_, test), True)

    def testInverse(self):
        Q    = quaternion_random(10**6, RANDOM_SEED)
        Q_   = quaternion_inverse(Q)

        test = np.array(Q)
        test[:, :3] *= -1
        test        /= np.einsum("...i,...i", Q, Q)[:, None]

        self.assertEqual(allclose(Q_, test), True)

    def testSub(self):
        Q0 = quaternion_random(10**6, RANDOM_SEED)
        Q1 = quaternion_random(10**6, RANDOM_SEED)
        Q_ = quaternion_sub(Q0, Q1)

        self.assertEqual(allclose(Q_, Q0 - Q1), True)

    def testMultiply(self):
        Q0       = np.zeros((10**6, 4))
        Q0[:, 3] = 1.0

        Q1 = quaternion_random(10**6, RANDOM_SEED)
        Q_ = quaternion_multiply(Q0, Q1)

        self.assertEqual(allclose(Q_, Q1), True)

    def testDot(self):
        Q0   = quaternion_random(10**6, RANDOM_SEED)
        Q1   = quaternion_random(10**6, RANDOM_SEED_TWO)

        dot  = quaternion_dot(Q0, Q1)
        dot_ = np.einsum("...i,...i", Q0, Q1)

        self.assertEqual(allclose(dot, dot_), True)