import os
import pickle
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