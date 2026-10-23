import unittest

import numpy as np
from cgmath.geometry import MeshData, SkinData


class TestSymmetry(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define a base shapes
        self.symmetrical = MeshData(points=np.array([[-2.260915756225586, -0.5              ,  0.5              ],
                                                     [ 0.0              , -0.5              ,  0.5              ],
                                                     [ 2.260915756225586, -0.5              ,  0.5              ],
                                                     [-1.0              ,  0.5              ,  0.5              ],
                                                     [ 0.0              ,  0.5              ,  0.5              ],
                                                     [ 1.0              ,  0.5              ,  0.5              ],
                                                     [-1.0              ,  0.5              , -0.5              ],
                                                     [ 0.0              ,  0.5              , -0.5              ],
                                                     [ 1.0              ,  0.5              , -0.5              ],
                                                     [-1.884657025337219, -0.5              , -1.995251417160034],
                                                     [ 0.0              , -0.5              , -0.5              ],
                                                     [ 1.884657025337219, -0.5              , -1.995251417160034]]),
                                    indices=np.array([ 0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 6, 7, 10, 9, 7, 8, 11, 10, 9, 10, 1, 0, 10, 11, 2, 1, 2, 11, 8, 5, 9, 0, 3, 6]),
                                    counts=np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]))
        
        self.missing_face = MeshData(points=np.array([[-2.260915756225586, -0.5              ,  0.5              ],
                                                      [ 0.0              , -0.5              ,  0.5              ],
                                                      [ 2.260915756225586, -0.5              ,  0.5              ],
                                                      [-1.0              ,  0.5              ,  0.5              ],
                                                      [ 0.0              ,  0.5              ,  0.5              ],
                                                      [ 1.0              ,  0.5              ,  0.5              ],
                                                      [-1.0              ,  0.5              , -0.5              ],
                                                      [ 0.0              ,  0.5              , -0.5              ],
                                                      [ 1.0              ,  0.5              , -0.5              ],
                                                      [-1.884657025337219, -0.5              , -1.995251417160034],
                                                      [ 0.0              , -0.5              , -0.5              ],
                                                      [ 1.884657025337219, -0.5              , -1.995251417160034]]),
                                     indices=np.array([ 0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 7, 8, 11, 10, 9, 10, 1, 0, 10, 11, 2, 1, 2, 11, 8, 5, 9, 0, 3, 6]),
                                     counts=np.array([4, 4, 4, 4, 4, 4, 4, 4, 4]))
         
        self.flipped_face = MeshData(points=np.array([[-2.260915756225586, -0.5              ,  0.5              ],
                                                      [ 0.0              , -0.5              ,  0.5              ],
                                                      [ 2.260915756225586, -0.5              ,  0.5              ],
                                                      [-1.0              ,  0.5              ,  0.5              ],
                                                      [ 0.0              ,  0.5              ,  0.5              ],
                                                      [ 1.0              ,  0.5              ,  0.5              ],
                                                      [-1.0              ,  0.5              , -0.5              ],
                                                      [ 0.0              ,  0.5              , -0.5              ],
                                                      [ 1.0              ,  0.5              , -0.5              ],
                                                      [-1.884657025337219, -0.5              , -1.995251417160034],
                                                      [ 0.0              , -0.5              , -0.5              ],
                                                      [ 1.884657025337219, -0.5              , -1.995251417160034]]),
                                     indices=np.array([ 0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 6, 9, 10, 7, 7, 8, 11, 10, 9, 10, 1, 0, 10, 11, 2, 1, 2, 11, 8, 5, 9, 0, 3, 6]),
                                     counts=np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]))
        
        self.fix_me = MeshData(points=np.array([[-2.260915756225586, -0.5              ,  0.5              ],
                                                [ 0.0              , -0.5              ,  0.5              ],
                                                [ 2.260915756225586, -0.5              ,  0.5              ],
                                                [-1.171887516975403,  0.54886257648468 ,  1.037285804748535],
                                                [ 0.0              ,  0.5              ,  0.5              ],
                                                [ 1.0              ,  0.5              ,  0.5              ],
                                                [-1.0              ,  0.5              , -0.5              ],
                                                [ 0.0              ,  0.5              , -0.5              ],
                                                [ 1.0              ,  0.5              , -0.5              ],
                                                [-1.884657859802246, -0.499999791383743, -1.99525260925293 ],
                                                [ 0.0              , -0.5              , -0.5              ],
                                                [ 1.884657025337219, -0.5              , -1.995251417160034]]),
                               indices=np.array([ 0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 6, 7, 10, 9, 7, 8, 11, 10, 9, 10, 1, 0, 10, 11, 2, 1, 2, 11, 8, 5, 9, 0, 3, 6]),
                               counts=np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]))        
   
   
   
   
        self.skin_mesh = MeshData(indices=np.array([0, 1, 4, 3, 1, 2, 5, 4]),
                                  counts=np.array([4, 4]),
                                  points=np.array([[-1. ,  0.5,  0.5],
                                                   [ 0. ,  0.5,  0.5],
                                                   [ 1. ,  0.5,  0.5],
                                                   [-1. ,  0.5, -0.5],
                                                   [ 0. ,  0.5, -0.5],
                                                   [ 1. ,  0.5, -0.5]]),
                                  name ='body_geometryShape')
   
   
        self.bad_skin = SkinData(weights= np.array([[0. , 0.5, 0.5],
                                                    [0.5, 0. , 0.5],
                                                    [0. , 1. , 0. ],
                                                    [0. , 0.5, 0.5],
                                                    [0.5, 0. , 0.5],
                                                    [0. , 1. , 0. ]]),
                                 influences= ['root', 'l_joint', 'r_joint'])
   
        self.good_skin = SkinData(weights= np.array([[0.  , 0.  , 1.  ],
                                                     [0.5 , 0.25, 0.25],
                                                     [0.  , 1.  , 0.  ],
                                                     [0.  , 0.  , 1.  ],
                                                     [0.5 , 0.25, 0.25],
                                                     [0.  , 1.  , 0.  ]]),
                                 influences= ['root', 'l_joint', 'r_joint'])   
   


    def test_skin_symmetry(self):
        obj1 = self.bad_skin.copy()
        self.assertFalse(obj1.symmetrical(self.skin_mesh))
        self.assertTrue(obj1.fix_symmetry(self.skin_mesh))
        self.assertTrue(obj1.symmetrical(self.skin_mesh))
        self.assertTrue(obj1 == self.good_skin)
        
        
        
    def test_mesh_symmetry(self):
        obj1 = self.symmetrical.copy()
        obj2 = self.missing_face.copy()
        obj3 = self.flipped_face.copy()
        obj4 = self.fix_me.copy()
        
        self.assertTrue(obj1.symmetrical())
        
        self.assertTrue(obj2.symmetrical(check_topology=False))
        self.assertFalse(obj2.symmetrical(check_topology=True))
        
        self.assertTrue(obj3.symmetrical(check_topology=False))
        self.assertTrue(obj3.symmetrical(check_normals=False))
        self.assertFalse(obj3.symmetrical(check_normals=True))
        
        self.assertFalse(obj4.symmetrical())
        self.assertTrue(obj4.fix_symmetry())
        self.assertTrue(obj4==obj1)

    def test_mesh_symmetry_about_any_pivot_and_axis(self):
        # moved to x = 1 the mesh is fixed about pivot=1.0 as it is about 0,
        # and with x and z swapped about axis=2 as about axis 0
        def moved(mesh, shift, axes):
            new        = mesh.copy()
            new.points = mesh.points[:, axes] + shift
            return new

        cases = (
            (1.0, 0, [1.0, 0.0, 0.0], [0, 1, 2]),
            (0.0, 2, [0.0, 0.0, 0.0], [2, 1, 0]),
            (-0.5, 2, [0.0, 0.0, -0.5], [2, 1, 0]),
        )
        for side in (1.0, -1.0, 0.0):
            for tolerance in (None, 0.7):
                fixed = self.fix_me.copy()
                self.assertTrue(fixed.fix_symmetry(side=side, tolerance=tolerance))
                for pivot, axis, shift, axes in cases:
                    with self.subTest(side=side, tolerance=tolerance, pivot=pivot, axis=axis):
                        obj = moved(self.fix_me, shift, axes)
                        self.assertFalse(obj.symmetrical(pivot=pivot, axis=axis))
                        self.assertTrue(
                            obj.fix_symmetry(pivot=pivot, axis=axis, side=side, tolerance=tolerance)
                        )
                        self.assertTrue(obj.symmetrical(pivot=pivot, axis=axis))
                        self.assertTrue(obj == moved(fixed, shift, axes))
