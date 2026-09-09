import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.morph_target import MorphData, MorphList



class TestMorphList(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.jawOpen = MorphData(name='jawOpen',
                                       indices=np.array([5, 3, 4]),
                                       offsets=np.array([[ 0.0,  0.0,  0.0],
                                                         [ 9.5,  1.5,  2.5],
                                                         [-0.5,  1.5, -2.5]]))

        self.jawClosed = MorphData(name='jawClosed',
                                         indices=np.array([1, 6, 4]),
                                         offsets=np.array([[ 0.0,  0.0,  0.0],
                                                           [ 9.5,  1.5,  2.5],
                                                           [-0.5,  1.5, -2.5]]))

        self.eyesClosed = MorphData(name='eyesClosed',
                                          indices=np.array([3, 12, 1]),
                                          offsets=np.array([[ 10.0,  10.0,  10.0],
                                                            [ 19.5,  11.5,  12.5],
                                                            [-10.5,  11.5, -12.5]]))


        self.list = MorphList([self.eyesClosed, self.jawClosed, self.jawOpen])


    def test_bytes(self):
        obj1 = self.list.copy()
        b    = obj1.to_bytes()
        obj2 = MorphList.from_bytes(b)
        self.assertTrue(obj1 == obj2)


    def test_serialize(self):

        obj1 = self.list.copy()

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # save obj1 to file
            obj1.save(f)

            # load obj2 from file
            obj2 = MorphList.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # do the same via pickle, put obj1 inside a list
            with open(f, "wb") as io:
                pickle.dump([obj1], io)

            with open(f, "rb") as io:
                obj3 = pickle.load(io)[0]

                # confirm they're identical
                self.assertTrue(obj1 == obj3)

            # test json and npz
            obj1.save(f, mode='json')
            obj4 = MorphList.load(f)
            self.assertTrue(obj1 == obj4)

            obj1.save(f, mode='npz')
            obj5 = MorphList.load(f)
            self.assertTrue(obj1 == obj5)


    def test_serialize_array_types(self):
        """indices is an optional field, make sure it still loads back as an array"""

        obj1 = self.list.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ('pkl', 'json', 'npz'):
                f = os.path.join(temp_dir, "test." + mode)
                obj1.save(f, mode=mode)
                obj2 = MorphList.load(f)

                for shape in obj2:
                    self.assertIsInstance(shape.indices, np.ndarray, msg=mode)
                    self.assertIsInstance(shape.offsets, np.ndarray, msg=mode)

                # operations masking indices fail on anything but an array
                obj2.prune_offsets()
                self.assertEqual(obj2.size.tolist(), [3, 2, 2], msg=mode)


    def test_list_like(self):

        # randomize list
        import random

        obj1 = self.list.copy()
        obj2 = self.list.copy()

        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj2.sort()
        self.assertTrue(obj1 == obj2)

        # use sorted
        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj3 = MorphList(sorted(obj2))
        self.assertTrue(obj1 == obj3)


        # sum a list like a list of numbers
        obj3 = sum(obj1)
        self.assertTrue(obj3 == (self.eyesClosed + self.jawClosed + self.jawOpen))

        # array indexing, slicing
        self.assertTrue(obj1[1] == self.jawClosed)
        self.assertTrue(sum(obj1[:2]) == (self.eyesClosed + self.jawClosed))

        # reverse
        obj1.sort()
        obj2.sort()
        obj1.reverse()
        self.assertTrue(obj1 == obj2[::-1])

        # numpy like indexing
        obj1.sort()
        obj2.sort()
        self.assertTrue(sum(obj1[[1, 2]]) == (obj2[1]+obj2[2]))

        # mutability
        eyesClosed   = self.eyesClosed.copy()
        jawClosed    = self.jawClosed.copy()
        jawOpen      = self.jawOpen.copy()

        obj1         = MorphList([eyesClosed, jawClosed, jawOpen])
        obj2         = obj1[[2, 1, 0]]
        jawOpen.name = 'POOF!'
        self.assertTrue(obj1[2].name == 'POOF!')
        self.assertTrue(obj2[0].name == 'POOF!')

        # dict like indexting
        obj1 = self.list.copy()
        self.assertTrue(obj2['jawClosed'] == self.jawClosed)

        index = obj1.index('eyesClosed')
        self.assertTrue(obj1[index] == self.eyesClosed)

        # is a shape in the list?
        self.assertTrue(obj1[0] in obj1)                # shape object in list by name
        self.assertTrue('eyesClosed' in obj1)           # string in list names
        self.assertFalse('SDFAFDAEWFASDFAFSD' in obj1)  # string not in list names

    def test_functions(self):
        obj1 = self.list.copy()

        # pattern matching
        obj2 = obj1.match("jaw*")
        self.assertTrue(sum(obj2) == (obj1['jawClosed'] + obj1['jawOpen']))

        # indexing workflow
        indices = np.where(obj1.mse < 15)[0]
        obj2    = sum(obj1[indices])
        self.assertTrue(obj2 == (obj1['jawClosed'] + obj1['jawOpen']))


    def test_vectorize(self):

        # add
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  += 2
        shape1 += 2
        self.assertTrue(list1[0] == shape1)

        # sub
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  -= 2
        shape1 -= 2
        self.assertTrue(list1[0] == shape1)

        # mult
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  *= 2
        shape1 *= 2
        self.assertTrue(list1[0] == shape1)

        # div
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  /= 2
        shape1 /= 2
        self.assertTrue(list1[0] == shape1)


        # size
        list1 = self.list.copy()
        self.assertTrue(np.allclose(list1.size, [3, 3, 3]))

        # min
        self.assertTrue(np.allclose(list1.min, [17.32050808, 0, 0]))

        # max
        self.assertTrue(np.allclose(list1.max, [25.86020108,  9.93730346,  9.93730346]))

        # mse
        self.assertTrue(np.allclose(list1.mse, [151.94444444,  11.94444444,  11.94444444]))

    def test_pow_list(self):
        """Test power operations for MorphList"""
        # Test in-place power
        list1  = self.list.copy()
        shape1 = list1[0].copy()
        
        # Set all offsets to 2 for easier verification
        for shape in list1:
            shape.offsets[:] = 2.0
        shape1.offsets[:] = 2.0
        
        list1  **= 2
        shape1 **= 2
        
        self.assertTrue(list1[0] == shape1)
        self.assertTrue(np.all(list1[0].offsets == 4.0))
        self.assertTrue(np.all(list1[1].offsets == 4.0))
        self.assertTrue(np.all(list1[2].offsets == 4.0))
        
        # Test power operator
        list2 = self.list.copy()
        for shape in list2:
            shape.offsets[:] = 3.0
        
        list3 = list2 ** 2
        self.assertTrue(np.all(list3[0].offsets == 9.0))
        self.assertTrue(np.all(list3[1].offsets == 9.0))
        self.assertTrue(np.all(list3[2].offsets == 9.0))
        
        # Verify original is unchanged
        self.assertTrue(np.all(list2[0].offsets == 3.0))
        
        # Test with higher powers
        list4 = self.list.copy()
        for shape in list4:
            shape.offsets[:] = 2.0
        
        list5 = list4 ** 3
        self.assertTrue(np.all(list5[0].offsets == 8.0))

    def test_prune_offsets(self):
        obj = self.list.copy()
        obj.prune_offsets()
        # zero-magnitude offset rows are dropped within each shape
        self.assertTrue(np.allclose(obj.size, [3, 2, 2]))

    def test_remove_unused(self):
        # an entirely-zero shape is dropped from the list
        dead = MorphData(name='dead',
                               indices=np.array([0, 1]),
                               offsets=np.array([[0., 0., 0.],
                                                 [0., 0., 0.]]))
        obj = MorphList([self.eyesClosed.copy(), dead, self.jawOpen.copy()])
        obj.remove_unused()
        self.assertTrue([shape.name for shape in obj] == ['eyesClosed', 'jawOpen'])

        # tolerance drops shapes whose offsets are all below threshold
        tiny = MorphData(name='tiny',
                               indices=np.array([0]),
                               offsets=np.array([[0.0005, 0.0, 0.0]]))
        obj = MorphList([self.eyesClosed.copy(), tiny])
        obj.remove_unused(tolerance=0.001)
        self.assertTrue([shape.name for shape in obj] == ['eyesClosed'])

        # prune_offsets then remove_unused drops shapes emptied by pruning
        obj = MorphList([self.jawOpen.copy(), dead.copy()])
        obj.prune_offsets()
        self.assertTrue(np.allclose(obj.size, [2, 0]))
        obj.remove_unused()
        self.assertTrue([shape.name for shape in obj] == ['jawOpen'])





class TestMorphData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.box = MeshData(points=np.array([[-0.5, -0.5,  0.5],
                                             [ 0.5, -0.5,  0.5],
                                             [-0.5,  0.5,  0.5],
                                             [ 0.5,  0.5,  0.5],
                                             [-0.5,  0.5, -0.5],
                                             [ 0.5,  0.5, -0.5],
                                             [-0.5, -0.5, -0.5],
                                             [ 0.5, -0.5, -0.5]]),
                            indices=np.array([0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]),
                            counts=np.array([4, 4, 4, 4, 4, 4]))

        self.shape = MorphData(name='target',
                                     indices=np.array([5, 3, 4]),
                                     offsets=np.array([[ 0.0,  0.0,  0.0],
                                                       [ 9.5,  1.5,  2.5],
                                                       [-0.5,  1.5, -2.5]]))

        self.shape2 = MorphData(name='target2',
                                      indices=np.array([1, 6, 4]),
                                      offsets=np.array([[ 0.0,  0.0,  0.0],
                                                        [ 9.5,  1.5,  2.5],
                                                        [-0.5,  1.5, -2.5]]))
        
        
    def test_default_indices(self):
        shape = MorphData(name='target',
                                offsets=np.array([[ 0.0,  0.0,  0.0],
                                                  [ 9.5,  1.5,  2.5],
                                                  [-0.5,  1.5, -2.5]]))
        self.assertTrue(np.allclose(shape.indices, [0, 1, 2]))
        
        
        
    def test_bytes(self):
        obj1 = self.shape.copy()
        b    = obj1.to_bytes()
        obj2 = MorphData.from_bytes(b)
        self.assertTrue(obj1 == obj2)

    def test_serialize(self):

        # add custom private data
        obj1              = self.shape.copy()
        obj1._custom_data = "test"

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # save obj1 to file
            obj1.save(f)

            # load obj2 from file
            obj2 = MorphData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # confirm obj2 has no private data
            self.assertFalse(hasattr(obj2, "_custom_data"))

            # do the same via pickle, put obj1 inside a list
            with open(f, "wb") as io:
                pickle.dump([obj1], io)

            with open(f, "rb") as io:
                obj3 = pickle.load(io)[0]

                # confirm they're identical
                self.assertTrue(obj1 == obj3)

                # confirm obj3 has no cached data
                self.assertFalse(hasattr(obj3, "_custom_data"))

            # test json and npz
            obj1.save(f, mode='json')
            obj4 = MorphData.load(f)
            self.assertTrue(obj1 == obj4)

            obj1.save(f, mode='npz')
            obj5 = MorphData.load(f)
            self.assertTrue(obj1 == obj5)



    def test_properties(self):
        # prune unused for this test
        shape = self.shape.copy()
        shape.prune_offsets()
        self.assertTrue(shape.size == len(shape.offsets))
        self.assertTrue(shape.min == shape.magnitudes[1])
        self.assertTrue(shape.max == shape.magnitudes[0])
        self.assertTrue(shape.mse == 17.916666666666668)

    def test_unit(self):
        unit = np.linalg.norm(self.shape.unit, axis=1)
        self.assertTrue(np.allclose(unit, [0.0, 1.0, 1.0]))

    def test_sort(self):
        shape = self.shape.copy()
        shape.sort()
        unit = np.linalg.norm(shape.unit, axis=1)
        self.assertTrue(np.allclose(unit, [1.0, 1.0, 0.0]))

    def test_add(self):
        shape = self.shape.copy()
        shape += 2
        expected = np.array([7, 5, 6])
        self.assertTrue(np.allclose(shape.indices, expected))

        shape  = self.shape.copy()
        shape2 = self.shape2.copy()
        shape3 = shape + shape2
        expected_offsets = np.array([[ 0. ,  0. ,  0. ],
                                     [ 9.5,  1.5,  2.5],
                                     [-1. ,  3. , -5. ],
                                     [ 0. ,  0. ,  0. ],
                                     [ 9.5,  1.5,  2.5]])
        expected_indices = np.array([1, 3, 4, 5, 6])

        self.assertTrue(shape3.name == "target_target2")
        self.assertTrue(np.allclose(shape3.offsets, expected_offsets))
        self.assertTrue(np.allclose(shape3.indices, expected_indices))

        # try sum!
        shape3 = sum([shape, shape2])
        self.assertTrue(shape3.name == "target_target2")
        self.assertTrue(np.allclose(shape3.offsets, expected_offsets))
        self.assertTrue(np.allclose(shape3.indices, expected_indices))


    def test_mult(self):
        shape = self.shape.copy()
        mag   = shape.magnitudes
        shape *= 0.5
        self.assertTrue(np.allclose(shape.magnitudes, mag * 0.5))

    def test_div(self):
        shape = self.shape.copy()
        mag   = shape.magnitudes
        shape /= 2
        self.assertTrue(np.allclose(shape.magnitudes, mag / 2))

    def test_pow(self):
        shape            = self.shape.copy()
        shape.offsets[:] = 3
        shape **= 2
        self.assertTrue(np.all(shape.offsets == 9))

    def test_sub(self):
        """Test subtraction operations for MorphData"""
        shape  = self.shape.copy()
        shape2 = self.shape2.copy()
        
        # Test in-place subtraction with another MorphData
        shape3 = shape.copy()
        shape3 -= shape2
        
        # After subtraction, indices should be the union of both sets
        # shape: indices [5, 3, 4] with offsets [[0, 0, 0], [9.5, 1.5, 2.5], [-0.5, 1.5, -2.5]]
        # shape2: indices [1, 6, 4] with offsets [[0, 0, 0], [9.5, 1.5, 2.5], [-0.5, 1.5, -2.5]]
        # Result should have indices [1, 3, 4, 5, 6]
        expected_indices = np.array([1, 3, 4, 5, 6])
        self.assertTrue(np.allclose(shape3.indices, expected_indices))
        
        # Verify subtraction logic
        # Index 1: 0 - 0 = 0
        # Index 3: 9.5,1.5,2.5 - 0 = 9.5,1.5,2.5  
        # Index 4: -0.5,1.5,-2.5 - (-0.5,1.5,-2.5) = 0,0,0
        # Index 5: 0 - 0 = 0
        # Index 6: 0 - 9.5,1.5,2.5 = -9.5,-1.5,-2.5
        self.assertTrue(shape3.indices.shape[0] == 5)
        
        # Test subtraction operator
        shape4 = shape - shape2
        self.assertTrue(np.allclose(shape4.indices, expected_indices))
        
        # Test subtraction with scalar (index offset)
        shape5 = self.shape.copy()
        shape5 -= 2
        expected_indices_scalar = np.array([3, 1, 2])
        self.assertTrue(np.allclose(shape5.indices, expected_indices_scalar))
        
        # Test right-hand subtraction with 0
        shape6               = 0 - self.shape
        expected_offsets_neg = self.shape.offsets * -1
        self.assertTrue(np.allclose(shape6.offsets, expected_offsets_neg))

    def test_is_zero(self):
        """Test is_zero method"""
        # Test with non-zero offsets
        self.assertFalse(self.shape.is_zero())
        
        # Test with all zero offsets
        zero_shape = MorphData(name='zero',
                                     indices=np.array([0, 1, 2]),
                                     offsets=np.array([[0., 0., 0.],
                                                      [0., 0., 0.],
                                                      [0., 0., 0.]]))
        self.assertTrue(zero_shape.is_zero())
        
        # Test with empty offsets
        empty_shape = MorphData(name='empty',
                                      indices=np.array([]),
                                      offsets=np.array([]).reshape(0, 3))
        self.assertTrue(empty_shape.is_zero())
        
        # Test with tolerance
        tiny_shape = MorphData(name='tiny',
                                    indices=np.array([0, 1]),
                                    offsets=np.array([[0.0001, 0.0001, 0.0001],
                                                     [0.0002, 0.0002, 0.0002]]))
        self.assertFalse(tiny_shape.is_zero(tolerance=0.0))
        self.assertTrue(tiny_shape.is_zero(tolerance=0.001))

    def test_from_mesh_data(self):
        """Test from_mesh_data class method"""
        # Create base and target meshes
        base_mesh   = self.box.copy()
        target_mesh = self.box.copy()
        
        # Modify some vertices in target mesh
        target_mesh.points[0] += [0.1, 0.2, 0.3]
        target_mesh.points[3] += [0.5, -0.1, 0.2]
        target_mesh.points[5] += [-0.2, 0.3, -0.1]
        
        # Create morph target
        morph = MorphData.from_mesh_data(
            base_mesh, target_mesh, target_name='test_morph'
        )
        
        # Verify the morph target has correct data
        self.assertEqual(morph.name, 'test_morph')
        self.assertEqual(morph.size, 3)
        self.assertTrue(0 in morph.indices)
        self.assertTrue(3 in morph.indices)
        self.assertTrue(5 in morph.indices)
        
        # Verify offsets are correct
        self.assertTrue(np.allclose(morph.offsets[np.where(morph.indices == 0)[0][0]], 
                                   [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(morph.offsets[np.where(morph.indices == 3)[0][0]], 
                                   [0.5, -0.1, 0.2]))
        self.assertTrue(np.allclose(morph.offsets[np.where(morph.indices == 5)[0][0]], 
                                   [-0.2, 0.3, -0.1]))
        
        # Test with tolerance
        base_mesh2   = self.box.copy()
        target_mesh2 = self.box.copy()
        target_mesh2.points[0] += [0.0001, 0.0001, 0.0001]  # Very small change
        
        morph_notol = MorphData.from_mesh_data(
            base_mesh2, target_mesh2, target_name='notol', tolerance=0.0
        )
        self.assertGreater(morph_notol.size, 0)
        
        morph_tol = MorphData.from_mesh_data(
            base_mesh2, target_mesh2, target_name='tol', tolerance=0.001
        )
        self.assertEqual(morph_tol.size, 0)
        
        # Test error case - mismatched vertex counts
        base_mesh3 = self.box.copy()
        target_mesh3 = MeshData(
            points  = np.array([[0., 0., 0.], [1., 0., 0.]]),
            indices = np.array([0, 1]),
            counts  = np.array([2])
        )
        
        with self.assertRaises(RuntimeError):
            MorphData.from_mesh_data(base_mesh3, target_mesh3)