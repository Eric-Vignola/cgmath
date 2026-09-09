import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData, UVList


class TestSymmetry(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define a base shapes
        quad_mesh = {
            "indices": np.array([0, 1, 4, 3, 1, 2, 5, 4]),
            "counts":  np.array([4, 4]),
            "points": np.array([[-1.0, -0.5,  0.5],
                                [ 0.0, -0.5,  0.5],
                                [ 1.0, -0.5,  0.5],
                                [-1.0,  0.5,  0.5],
                                [ 0.0,  0.5,  0.5],
                                [ 1.0,  0.5,  0.5]]),
            "name": "pCubeShape1"
        }
        self.quad_mesh = MeshData.from_dict(quad_mesh)

        quad_uvs = {
            "map1": {
                "indices": np.array([0, 1, 4, 3, 1, 2, 5, 4]),
                "counts":  np.array([4, 4]),
                "points": np.array([[0.375, 0.0  ],
                                    [0.5  , 0.0  ],
                                    [0.625, 0.0  ],
                                    [0.375, 0.25 ],
                                    [0.5  , 0.25 ],
                                    [0.625, 0.25 ]]),
                "name": "map1"
            }
        }
        self.quad_uvs = UVList.from_dict(quad_uvs)



        tri_mesh = {
            "indices": np.array([0, 1, 3, 3, 1, 4, 2, 5, 1, 1, 5, 4]),
            "counts":  np.array([3, 3, 3, 3]),
            "points": np.array([[-1.0, -0.5,  0.5],
                                [ 0.0, -0.5,  0.5],
                                [ 1.0, -0.5,  0.5],
                                [-1.0,  0.5,  0.5],
                                [ 0.0,  0.5,  0.5],
                                [ 1.0,  0.5,  0.5]]),
            "name": "pCubeShape2"
        }
        self.tri_mesh = MeshData.from_dict(tri_mesh)

        tri_uvs = {
            "map1": {
                "indices": np.array([0, 1, 3, 3, 1, 4, 2, 5, 1, 1, 5, 4]),
                "counts":  np.array([3, 3, 3, 3]),
                "points": np.array([[0.375, 0.0  ],
                                    [0.5  , 0.0  ],
                                    [0.625, 0.0  ],
                                    [0.375, 0.25 ],
                                    [0.5  , 0.25 ],
                                    [0.625, 0.25 ]]),
                "name": "map1"
            }
        }
        self.tri_uvs = UVList.from_dict(tri_uvs)


        fast_triangulate = {
            "indices": [0, 1, 3, 3, 1, 4, 1, 2, 4, 4, 2, 5],
            "counts":  [3, 3, 3, 3],
            "points": [[-1.0, -0.5,  0.5],
                       [ 0.0, -0.5,  0.5],
                       [ 1.0, -0.5,  0.5],
                       [-1.0,  0.5,  0.5],
                       [ 0.0,  0.5,  0.5],
                       [ 1.0,  0.5,  0.5]],
            "name": "pCubeShape1"
        }
        self.fast = MeshData.from_dict(fast_triangulate)


    def test_triangulate(self):

        # test trinity triangulation
        mesh  = self.quad_mesh.copy()
        uvs   = self.quad_uvs.copy()

        rules = mesh.get_triangulate_rules(method='trinity')
        mesh.triangulate(rules)
        uvs.triangulate(rules)

        self.assertTrue(mesh == self.tri_mesh)
        self.assertTrue(uvs == self.tri_uvs)


        # test fast triangulation
        mesh  = self.quad_mesh.copy()
        rules = mesh.get_triangulate_rules(method='fast')
        mesh.triangulate(rules)
        self.assertTrue(mesh == self.fast)

        # getting rules from uv's is perfectly fine
        # and should not error out
        uvs = self.quad_uvs.copy()
        uvs[0].get_triangulate_rules(method='fast')