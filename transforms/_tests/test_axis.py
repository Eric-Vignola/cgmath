import unittest

import numpy as np
from cgmath.transforms import (
    axis_angle_to_euler,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    X,
    XYZ,
    XZY,
    Y,
    YXZ,
    YZX,
    Z,
    ZXY,
    ZYX,
)

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestAxis(unittest.TestCase):
    def testAngleToEuler(self):
        eu = np.degrees(axis_angle_to_euler([1, 0, 0], np.radians(90)))
        self.assertEqual(allclose(eu, [[90, 0, 0]]), True)

    def testAngleToMatrix(self):
        M = axis_angle_to_matrix([1, 0, 0], 0)

        self.assertEqual(
            np.allclose(
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

    def testAngleToQuaternion(self):
        Q = axis_angle_to_quaternion([1, 0, 0], 0)
        self.assertEqual(allclose(Q, [[0.0, 0.0, 0.0, 1.0]]), True)

    def test_angle_to_quaternion_90_degrees(self):
        """Test axis_angle_to_quaternion with 90 degree rotation around X axis"""
        Q = axis_angle_to_quaternion([1, 0, 0], np.radians(90))
        # 90 degree rotation around X axis should give specific quaternion
        expected = np.array([[0.70710678, 0.0, 0.0, 0.70710678]])
        self.assertTrue(allclose(Q, expected, atol=1e-6))

    def test_angle_to_quaternion_y_axis(self):
        """Test axis_angle_to_quaternion with rotation around Y axis"""
        Q        = axis_angle_to_quaternion([0, 1, 0], np.radians(90))
        expected = np.array([[0.0, 0.70710678, 0.0, 0.70710678]])
        self.assertTrue(allclose(Q, expected, atol=1e-6))

    def test_angle_to_quaternion_z_axis(self):
        """Test axis_angle_to_quaternion with rotation around Z axis"""
        Q        = axis_angle_to_quaternion([0, 0, 1], np.radians(90))
        expected = np.array([[0.0, 0.0, 0.70710678, 0.70710678]])
        self.assertTrue(allclose(Q, expected, atol=1e-6))

    def test_angle_to_quaternion_multiple(self):
        """Test axis_angle_to_quaternion with multiple axes and angles"""
        axes   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        angles = np.array([0, np.radians(90), np.radians(180)])
        Q      = axis_angle_to_quaternion(axes, angles)

        # Should return 3 quaternions
        self.assertEqual(Q.shape, (3, 4))

        # First should be identity
        self.assertTrue(allclose(Q[0], [0.0, 0.0, 0.0, 1.0]))

    def test_angle_to_matrix_90_degrees(self):
        """Test axis_angle_to_matrix with -90 degree rotation around X axis"""
        M = axis_angle_to_matrix([1, 0, 0], np.radians(-90))

        # Check shape
        self.assertEqual(M.shape, (1, 4, 4))

        # 90 degree rotation around X should swap Y and Z with sign change
        # Applying to [0, 1, 0] should give [0, 0, 1]
        point    = np.array([0, 1, 0, 1])
        result   = M[0] @ point
        expected = np.array([0, 0, 1, 1])
        self.assertTrue(allclose(result, expected, atol=1e-6))

    def test_angle_to_matrix_identity(self):
        """Test axis_angle_to_matrix with zero rotation"""
        M        = axis_angle_to_matrix([1, 0, 0], 0)
        identity = np.eye(4)
        self.assertTrue(allclose(M[0], identity))

    def test_angle_to_matrix_multiple(self):
        """Test axis_angle_to_matrix with multiple axes and angles"""
        axes   = np.array([[1, 0, 0], [0, 1, 0]])
        angles = np.array([np.radians(45), np.radians(90)])
        M      = axis_angle_to_matrix(axes, angles)

        # Should return 2 matrices
        self.assertEqual(M.shape, (2, 4, 4))

    def test_angle_to_euler_y_axis(self):
        """Test axis_angle_to_euler with rotation around Y axis"""
        eu = np.degrees(axis_angle_to_euler([0, 1, 0], np.radians(90)))
        self.assertTrue(allclose(eu, [[0, 90, 0]], atol=1e-5))

    def test_angle_to_euler_z_axis(self):
        """Test axis_angle_to_euler with rotation around Z axis"""
        eu = np.degrees(axis_angle_to_euler([0, 0, 1], np.radians(90)))
        self.assertTrue(allclose(eu, [[0, 0, 90]], atol=1e-5))

    def test_angle_to_euler_with_rotate_order(self):
        """Test axis_angle_to_euler with different rotate orders"""
        # Test with YZX rotate order
        eu = np.degrees(axis_angle_to_euler([1, 0, 0], np.radians(90), axes=YZX))
        self.assertEqual(eu.shape, (1, 3))

    def test_angle_to_euler_multiple(self):
        """Test axis_angle_to_euler with multiple axes and angles"""
        axes   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        angles = np.array([np.radians(45), np.radians(90), np.radians(180)])
        eu     = axis_angle_to_euler(axes, angles)

        # Should return 3 euler angle sets
        self.assertEqual(eu.shape, (3, 3))

    def test_constants(self):
        """Test that axis constants are properly defined"""
        self.assertEqual(XYZ, 0)
        self.assertEqual(YZX, 1)
        self.assertEqual(ZXY, 2)
        self.assertEqual(XZY, 3)
        self.assertEqual(YXZ, 4)
        self.assertEqual(ZYX, 5)

        self.assertEqual(X,   0)
        self.assertEqual(Y,   1)
        self.assertEqual(Z,   2)

    def test_angle_to_quaternion_180_degrees(self):
        """Test axis_angle_to_quaternion with 180 degree rotation"""
        Q = axis_angle_to_quaternion([1, 0, 0], np.radians(180))
        # 180 degree rotation should have w=0
        self.assertTrue(allclose(Q[0, 3], 0.0, atol=1e-6))
        # And the axis should be in i component
        self.assertTrue(allclose(Q[0, 0], 1.0, atol=1e-6))

    def test_angle_to_matrix_180_degrees(self):
        """Test axis_angle_to_matrix with 180 degree rotation"""
        M = axis_angle_to_matrix([0, 1, 0], np.radians(180))

        # 180 degree rotation around Y should flip X and Z
        point    = np.array([1, 0, 0, 1])
        result   = M[0] @ point
        expected = np.array([-1, 0, 0, 1])
        self.assertTrue(allclose(result, expected, atol=1e-6))