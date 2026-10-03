import os
import pickle
import re
import tempfile
import unittest

import numpy as np
from cgmath.constraints import ProcrustesData
from cgmath.geometry.mesh import MeshData

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestProcrustesData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define a base shapes
        self.box = MeshData(
            points=np.array(
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                ]
            ),
            indices=np.array(
                [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )


    def test_field_shapes_name_the_documented_dimensions(self):
        # N points, M constraints, K the widest cluster, as in the README
        constraint = ProcrustesData(self.box)
        with self.assertRaisesRegex(ValueError, r"points must have shape \(N, 3\)"):
            constraint.points = np.zeros((8, 2))
        with self.assertRaisesRegex(ValueError, r"transforms must have shape \(M, 4, 4\)"):
            constraint.transforms = np.eye(3)
        with self.assertRaisesRegex(ValueError, r"clusters must have shape \(M, K\)"):
            constraint.clusters = np.zeros((2, 3, 4), dtype=np.int32)

    def test_every_attach_checks_the_matrix_shape(self):
        # the first attach and every later one give the same shape error,
        # and none of them leaves a cluster without its transform
        for bad in (np.eye(3), np.eye(4)[None], np.zeros(0)):
            message = rf"must have shape \(4, 4\), got {re.escape(str(bad.shape))}"

            with self.subTest("first attach", shape=bad.shape):
                first = ProcrustesData(self.box)
                with self.assertRaisesRegex(ValueError, message):
                    first.attach(bad, [0, 1, 2])
                self.assertIsNone(first.transforms)
                self.assertIsNone(first.clusters)

            with self.subTest("later attach", shape=bad.shape):
                later = ProcrustesData(self.box)
                later.attach(np.eye(4), [0, 1, 2, 3])
                with self.assertRaisesRegex(ValueError, message):
                    later.attach(bad, [4, 5, 6])
                self.assertEqual(later.transforms.shape, (1, 4, 4))
                self.assertEqual(later.clusters.shape, (1, 4))
                self.assertTrue(later.valid)

    def test_procrustes(self):

        # attach a transform
        transform = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 1., 1.]])
        cluster    = [0,1,2,3]
        mesh       = self.box.copy()
        constraint = ProcrustesData(mesh)
        constraint.attach(transform, cluster)

        # move some points
        mesh.points[0,2] += 1
        mesh.points[1,2] += 1
        constraint.update(mesh)

        # check that the transform is updated
        self.assertTrue(allclose(constraint.matrix, np.array([[[ 0.81649658,  0.        ,  0.        ,  0.        ],
                                                               [ 0.        ,  0.57735027, -0.57735027,  0.        ],
                                                               [ 0.        ,  0.57735027,  0.57735027,  0.        ],
                                                               [ 0.        ,  0.4330127 ,  1.4330127 ,  1.        ]]])))