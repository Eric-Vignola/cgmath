import unittest

import numpy as np
from cgmath.transforms import (
    euler_to_matrix,
    matrix_identity,
    matrix_interpolate,
    matrix_local,
    matrix_multiply,
    matrix_normalize,
    matrix_point_multiply,
    matrix_random,
    matrix_slerp,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

EPSILON = np.finfo(np.float32).eps

# Adding a random seed so that we can reproduce the same random numbers and the test is deterministic
RANDOM_SEED     = 54321
RANDOM_SEED_TWO = 12345


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestMatrix(unittest.TestCase):
    def testRandom(self):
        M    = matrix_random(10**6, RANDOM_SEED)
        x    = np.einsum("...i,...i", M[:, 0], M[:, 0]) ** 0.5
        y    = np.einsum("...i,...i", M[:, 1], M[:, 1]) ** 0.5
        z    = np.einsum("...i,...i", M[:, 2], M[:, 2]) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(allclose(x, ones), True)
        self.assertEqual(allclose(y, ones), True)
        self.assertEqual(allclose(z, ones), True)

    def testInterpolate(self):
        M0 = matrix_random(10**6, RANDOM_SEED)
        M1 = matrix_random(10**6, RANDOM_SEED_TWO)
        w  = np.random.random(10**6)

        # inerp M0 to M1
        forward = matrix_interpolate(M0, M1, w)

        # do the opposite
        backward = matrix_interpolate(M1, M0, 1 - w)

        self.assertEqual(allclose(forward, backward), True)

    def testToEuler(self):
        M  = matrix_random(10**6, RANDOM_SEED)
        ea = matrix_to_euler(M)
        M_ = euler_to_matrix(ea)

        self.assertEqual(allclose(M, M_), True)

    def testToQuaternion(self):
        M  = matrix_random(10**6, RANDOM_SEED)
        Q  = matrix_to_quaternion(M)
        M_ = quaternion_to_matrix(Q)

        self.assertEqual(allclose(M, M_), True)

    def testSlerp(self):
        M0 = matrix_random(10**6, RANDOM_SEED)
        M1 = matrix_random(10**6, RANDOM_SEED_TWO)
        w  = np.random.random(10**6)

        # inerp M0 to M1
        forward = matrix_slerp(M0, M1, w)

        # do the opposite
        backward = matrix_slerp(M1, M0, 1 - w)

        self.assertEqual(allclose(forward, backward), True)

    def testNormalize(self):
        M    = matrix_random(10**6, RANDOM_SEED) * 0.1
        ones = np.ones(10**6)

        M_   = matrix_normalize(M)
        x    = np.einsum("...i,...i", M_[:, 0], M_[:, 0]) ** 0.5
        y    = np.einsum("...i,...i", M_[:, 1], M_[:, 1]) ** 0.5
        z    = np.einsum("...i,...i", M_[:, 2], M_[:, 2]) ** 0.5

        self.assertEqual(allclose(x, ones), True)
        self.assertEqual(allclose(y, ones), True)
        self.assertEqual(allclose(z, ones), True)

    def testLocal(self):
        M           = matrix_random(10**6, RANDOM_SEED)
        M[:, 3, :3] = np.random.random((10**6, 3))

        P           = matrix_identity(10**6)
        P[:, 3, :3] = np.random.random((10**6, 3))

        L = matrix_local(M, P)

        delta = M[:, 3, :3] - P[:, 3, :3]

        self.assertEqual(allclose(L[:, 3, :3], delta), True)

    def testMultiply(self):
        M0           = matrix_identity(10**6)
        M0[:, 3, :3] = np.random.random((10**6, 3))

        M1           = matrix_identity(10**6)
        M1[:, 3, :3] = np.random.random((10**6, 3))

        test = matrix_multiply(M1, M0)

        self.assertEqual(allclose(test[:, 3, :3], M0[:, 3, :3] + M1[:, 3, :3]), True)

    def testPoint(self):
        M0           = matrix_identity(10**6)
        M0[:, 3, :3] = np.random.random((10**6, 3))

        p    = np.random.random((10**6, 3))
        test = matrix_point_multiply(p, M0)
        self.assertEqual(allclose(test, M0[:, 3, :3] + p), True)