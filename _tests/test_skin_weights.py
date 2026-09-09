import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.skin_weights import CompactSkinData, SkinData, SkinList


EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestSkinData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.plane1 = SkinData(name='plane1',
                               weights=np.array([[1., 0.],
                                                 [1., 0.],
                                                 [0., 1.],
                                                 [0., 1.]]),
                               influences=['joint1', 'joint2'])

        self.plane2 = SkinData(name='plane2',
                               weights=np.array([[0., 1.],
                                                 [0., 1.],
                                                 [1., 0.],
                                                 [1., 0.]]),
                               influences=['joint2', 'joint1'])

        self.plane3 = SkinData(name='plane3',
                               weights=np.array([[0., 1.],
                                                 [0., 1.],
                                                 [0.8, 0.2],
                                                 [0.8, 0.2]]),
                               influences=['joint3', 'joint4'])

        self.skin_plane = SkinData(name='skin_plane',
                                   weights=np.array([[0.01405836, 0.71521821, 0.0405709 , 0.05326534, 0.00346092, 0.00484247, 0.00758003, 0.00435469, 0.0055533 , 0.1441818 , 0.00325081, 0.00366317],
                                                     [0.04135172, 0.01619038, 0.20850496, 0.00357411, 0.06391709, 0.39765825, 0.00284631, 0.00187167, 0.00182385, 0.01430106, 0.13699825, 0.11096235],
                                                     [0.01537315, 0.19351703, 0.01109295, 0.32888121, 0.00876703, 0.00728549, 0.0356738 , 0.05176977, 0.11187182, 0.23076046, 0.00239536, 0.00261191],
                                                     [0.05697418, 0.03931085, 0.03252276, 0.01729457, 0.5147892 , 0.20056238, 0.01692728, 0.01802512, 0.0180919 , 0.05054792, 0.01753333, 0.01742051]]),
                                   influences=['joint7', 'joint5', 'joint10', 'joint4', 'joint9', 'joint11', 'joint8', 'joint1', 'joint6', 'joint12', 'joint3', 'joint2'])


        self.skin_cube = SkinData(name='skin_cube',
                                  weights=np.array([[0.08888162, 0.01283826, 0.01813473, 0.35191556, 0.01032683, 0.1260927 , 0.23278095, 0.04371558, 0.0104856 , 0.01850666, 0.01713871, 0.06918282],
                                                    [0.00333714, 0.03681173, 0.00374266, 0.76355839, 0.00911119, 0.00299795, 0.00447123, 0.00161529, 0.00057977, 0.12426658, 0.04820704, 0.00130103],
                                                    [0.37062319, 0.00320995, 0.05517871, 0.01278562, 0.00455828, 0.00301951, 0.45781972, 0.00227121, 0.01332836, 0.00288782, 0.00489577, 0.06942186],
                                                    [0.02293708, 0.04977289, 0.64228035, 0.02618337, 0.09128398, 0.00173832, 0.01914152, 0.00133656, 0.00334257, 0.02358382, 0.11363001, 0.00476954],
                                                    [0.10314325, 0.00051369, 0.00795042, 0.00150383, 0.0022514 , 0.00189147, 0.0369799 , 0.0023725 , 0.75760431, 0.00048855, 0.00149244, 0.08380823],
                                                    [0.0582171 , 0.01246171, 0.16366736, 0.01491289, 0.54970806, 0.0064753 , 0.03452932, 0.00701842, 0.02984891, 0.00993075, 0.09649871, 0.01673147],
                                                    [0.02348922, 0.00136399, 0.00362707, 0.00839881, 0.00347742, 0.25458103, 0.03103826, 0.62510186, 0.00979881, 0.00171471, 0.00366502, 0.03374379],
                                                    [0.05211638, 0.03486958, 0.03808993, 0.09727245, 0.21356049, 0.09276199, 0.05249645, 0.0925172 , 0.01899913, 0.04547338, 0.23598803, 0.02585499]]),
                                  influences=['joint12', 'joint3', 'joint7', 'joint10', 'joint9', 'joint8', 'joint5', 'joint1', 'joint6', 'joint2', 'joint11', 'joint4'])


        self.list = SkinList([self.plane1, self.plane2, self.skin_cube])

    def test_validity(self):
        # mess with the weights
        skin = self.skin_cube.copy()
        self.assertTrue(skin.valid)
        skin.weights[0] = 0
        self.assertFalse(skin.valid)

        # mess with influences
        skin = self.skin_cube.copy()
        self.assertTrue(skin.valid)
        skin.influences = skin.influences[:-1]
        self.assertFalse(skin.valid)

    def test_transfer(self):
        skin_data = SkinData(
            influences=["a", "b", "c", "d"], weights=np.array([[0.1, 0.1, 0.2, 0.6]])
        )

        copy = skin_data.copy()
        copy.transfer_influences(["a", "b"], ["c", "d"])
        self.assertTrue(np.allclose(copy.weights, [[0.0, 0.0, 0.25, 0.75]]))
        self.assertTrue(copy.valid)

        copy = skin_data.copy()
        copy.transfer_influences(["a", "b"], ["c", "d"], weighted=False)
        self.assertTrue(np.allclose(copy.weights, [[0.0, 0.0, 0.3, 0.7]]))
        self.assertTrue(copy.valid)

        copy = skin_data.copy()
        copy.transfer_influences(["a", "b", "c"], ["c", "d"])
        self.assertTrue(np.allclose(copy.weights, [[0.0, 0.0, 0.1, 0.9]]))
        self.assertTrue(copy.valid)

        copy = skin_data.copy()
        copy.transfer_influences(["a", "b", "c"], ["c", "d"], weighted=False)
        self.assertTrue(np.allclose(copy.weights, [[0.0, 0.0, 0.2, 0.8]]))
        self.assertTrue(copy.valid)

    def test_max_influences(self):
        # create a copy of a skin
        skin = self.skin_cube.copy()

        # set max influences to 4
        skin.set_max_influences(4)
        self.assertFalse(np.allclose(skin.counts, self.skin_cube.counts) and skin.valid)

    def test_prune(self):
        # prune all weights below 1.0
        skin = self.skin_cube.copy()
        skin.prune(1.0)
        self.assertTrue(skin.counts.max() == 1 and skin.valid)
        self.assertTrue(skin.influences == self.skin_cube.influences)

        # prune all weights below 1.0 and optimize as an option
        skin = self.skin_cube.copy()
        skin.prune(1.0, optimize=True)
        self.assertTrue(skin.counts.max() == 1 and skin.valid)
        self.assertTrue(skin.influences != self.skin_cube.influences)

        # prune all weights below 1.0 and optimize in place
        skin = self.skin_cube.copy()
        skin.prune(1.0, optimize=False)
        skin.remove_unused()
        self.assertTrue(skin.counts.max() == 1 and skin.valid)
        self.assertTrue(skin.influences != self.skin_cube.influences)

    def test_sort(self):
        skin               = self.skin_cube.copy()
        skin.weights[0]    = 0.0
        skin.weights[0, 0] = 1.0
        self.assertTrue(skin.weights[0, 0] == 1)
        self.assertTrue(skin.weights[0, -1] == 0)

        last_name          = "|zzzzzzzzzzzzzzzzzzzzz"
        skin.influences[0] = last_name

        skin.sort()

        self.assertTrue(skin.influences[-1] == last_name)
        self.assertTrue(skin.weights[0, 0] == 0)
        self.assertTrue(skin.weights[0, -1] == 1)

    def test_normalize(self):
        skin = self.skin_cube.copy()
        skin.weights *= 0.5
        self.assertFalse(skin.valid)
        skin.normalize()
        self.assertTrue(skin.valid)

    def test_round(self):
        skin   = self.skin_cube.copy()

        before = len(f"{skin.weights[0].max():.10f}".split(".")[-1].rstrip("0"))
        skin.round(3)  # round to 3 decimal places
        after = len(f"{skin.weights[0].max():.10f}".split(".")[-1].rstrip("0"))

        self.assertTrue(before != after)
        self.assertTrue(after <= 3)

    def test_add(self):
        skin_cube  = self.skin_cube.copy()
        skin_plane = self.skin_plane.copy()

        skin_cube.prune(1)
        skin_plane.prune(1)
        combined = skin_cube + skin_plane

        self.assertTrue(combined.valid)

        combined = self.plane1 + self.plane3
        self.assertTrue(combined.valid)
        self.assertTrue(combined.influences == ["joint1", "joint2", "joint3", "joint4"])

        expected_weights = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.8, 0.2],
                [0.0, 0.0, 0.8, 0.2],
            ]
        )

        self.assertTrue(allclose(combined.weights, expected_weights))

    def test_fnmatch(self):
        # test fnmatch
        skin_cube = self.skin_cube.copy()
        infl      = skin_cube.get_influences("joint1*")
        self.assertTrue(sorted(infl) == ["joint1", "joint10", "joint11", "joint12"])

        infl = skin_cube.get_influences("Joint1*", exact=False)
        self.assertTrue(sorted(infl) == ["joint1", "joint10", "joint11", "joint12"])

        infl = skin_cube.get_influences("Joint1*", exact=True)
        self.assertTrue(len(infl) == 0)

    def test_insert(self):
        skin_cube  = self.skin_cube.copy()
        skin_plane = self.skin_plane.copy()
        combined   = skin_cube + skin_plane

        # create a new joint
        joint = "|awesome_joint"

        # insert a joint at index 1 and check for validity
        combined.insert(1, joint)
        self.assertTrue(combined.valid)

        # do the same and set weight to 0.1
        combined = skin_cube + skin_plane
        combined.insert(1, joint, weight=0.1)
        self.assertTrue(combined.valid)
        self.assertTrue(np.all(np.isclose(combined.weights[:, 1], 0.1)))
        self.assertTrue(combined.counts.max() > 1)

        # do the same and set weight to 1
        combined = skin_cube + skin_plane
        combined.insert(1, joint, weight=1)

        self.assertTrue(combined.valid)
        self.assertTrue(np.all(np.isclose(combined.weights[:, 1], 1)))
        self.assertTrue(combined.counts.max() == 1)

    def test_slice(self):
        skin_cube  = self.skin_cube.copy()
        skin_plane = self.skin_plane.copy()

        skin_cube.sort()
        skin_cube.prune(1, optimize=True)

        skin_plane.sort()
        skin_plane.prune(1, optimize=True)

        combo = skin_plane + skin_cube

        # slice and extract original plane from combined
        split = combo[: skin_plane.size]
        split.remove_unused()
        self.assertTrue(split == skin_plane)

        # same operation with list of indices
        split = combo[np.arange(skin_plane.size)]
        split.remove_unused()
        self.assertTrue(split == skin_plane)

        # extract cube out of combo, influence order will be different
        # but we can still test for equivalency
        split = combo[skin_plane.size :]
        self.assertTrue(split.is_equivalent(skin_cube))

    def test_max_counts(self):
        skin_cube = self.skin_cube.copy()
        before    = skin_cube.get_max_influences()
        skin_cube.set_max_influences(4)
        after = skin_cube.get_max_influences()
        self.assertTrue(before != after)
        self.assertTrue(after == 4)

        skin_cube = self.skin_cube.copy()
        skin_cube.set_max_influences(2)
        everything = skin_cube.get_indices_over_max(0)
        nothing    = skin_cube.get_indices_over_max(2)
        self.assertTrue(everything.size == skin_cube.size)
        self.assertTrue(nothing.size == 0)

    def test_compact_data(self):
        compact_data = self.skin_cube.to_compact_skin_data()
        new_data     = compact_data.to_skin_data()
        self.assertEqual(self.skin_cube, new_data)

    def test_append_remove(self):
        obj = self.plane1.copy()
        obj.append("test", 0.2)
        self.assertFalse(obj == self.plane1)
        obj.remove("test", optimize=True)

    def test_conform(self):
        # conform test, makes sure all skin weight are aligned

        plane1 = self.plane1.copy()
        plane2 = self.plane2.copy()
        plane3 = self.plane3.copy()

        self.assertFalse(plane1.influences == plane2.influences)
        SkinData.conform(plane1, plane2)
        self.assertTrue(plane1.influences == plane2.influences)

        SkinData.conform(plane1, plane2, plane3)
        self.assertTrue(plane1.influences == plane2.influences)
        self.assertTrue(plane1.influences == plane3.influences)

    def test_bytes(self):
        obj1 = self.skin_cube.copy()
        b    = obj1.to_bytes()
        obj2 = SkinData.from_bytes(b)
        self.assertTrue(obj1 == obj2)

    def test_serialize(self):
        obj1 = self.skin_cube.copy()

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # default mode (pkl)
            obj1.save(f)
            obj2 = SkinData.load(f)
            self.assertTrue(obj1 == obj2)

            # json mode
            obj1.save(f, mode="json")
            obj3 = SkinData.load(f)
            self.assertTrue(obj1 == obj3)

            # npz mode
            obj1.save(f, mode="npz")
            obj4 = SkinData.load(f)
            self.assertTrue(obj1 == obj4)

            # via pickle
            with open(f, "wb") as io:
                pickle.dump(obj1, io)

            with open(f, "rb") as io:
                obj5 = pickle.load(io)
                self.assertTrue(obj1 == obj5)

    def test_list_like(self):
        # randomize list
        import random

        obj1 = self.list.copy()
        obj2 = self.list.copy()

        while any(str(x) == str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj2.sort()
        self.assertTrue(obj1 == obj2)

        # use sorted
        while any(str(x) == str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj3 = SkinList(sorted(obj2))
        self.assertTrue(obj1 == obj3)

        # array indexing, slicing
        self.assertTrue(obj1[1] == self.plane2)

        # reverse
        obj1.sort()
        obj2.sort()
        obj1.reverse()
        self.assertTrue(obj1 == obj2[::-1])

        # numpy like indexing
        obj1.sort()
        obj2.sort()
        self.assertTrue(obj1[[1, 2]] == obj2[1:])

        # mutability
        plane1    = self.plane1.copy()
        plane2    = self.plane2.copy()
        skin_cube = self.skin_cube.copy()

        obj1           = SkinList([plane1, plane2, skin_cube])
        obj2           = obj1[[2, 1, 0]]
        skin_cube.name = "POOF!"
        self.assertTrue(obj1[2].name == "POOF!")
        self.assertTrue(obj2[0].name == "POOF!")

        # dict like indexting
        obj1 = self.list.copy()
        obj2 = self.list.copy()
        self.assertTrue(obj2["skin_cube"] == self.skin_cube)

        index = obj1.index("skin_cube")
        self.assertTrue(obj1[index] == self.skin_cube)

        # is a uv name in the list?
        self.assertTrue(obj1[0] in obj1)                # uv object in list by name
        self.assertTrue("skin_cube" in obj1)            # string in list names
        self.assertFalse("SDFAFDAEWFASDFAFSD" in obj1)  # string not in list names

    def test_functions(self):
        obj1 = self.list.copy()

        # pattern matching
        obj2 = obj1.match("*_cube")
        self.assertTrue(obj2[0] == self.skin_cube)


class TestCompactSkinData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Create a test SkinData and convert to compact
        self.skin_data = SkinData(
            name="test_skin",
            weights=np.array(
                [
                    [0.8, 0.2, 0.0, 0.0],
                    [0.0, 0.7, 0.3, 0.0],
                    [0.0, 0.0, 0.6, 0.4],
                    [0.5, 0.0, 0.0, 0.5],
                ]
            ),
            influences=["joint1", "joint2", "joint3", "joint4"],
        )
        self.compact_skin = self.skin_data.to_compact_skin_data()

    def test_compact_valid(self):
        """Test CompactSkinData.valid property"""
        # Valid compact skin data
        self.assertTrue(self.compact_skin.valid)

        # Make invalid by zeroing weights
        invalid_compact            = self.compact_skin.copy()
        invalid_compact.weights[:] = 0
        self.assertFalse(invalid_compact.valid)

        # Create compact data with non-normalized weights
        invalid_compact2 = CompactSkinData(
            max_influences    = 2,
            influence_indices = np.array([0, 1, 0, 1]),
            weights           = np.array([0.3, 0.3, 0.4, 0.4]),  # Sums to 0.6 and 0.8, not 1.0
            influences        = ["joint1", "joint2"],
        )
        self.assertFalse(invalid_compact2.valid)

    def test_compact_round(self):
        """Test CompactSkinData.round() method"""
        # Create compact skin with precise weights
        compact = CompactSkinData(
            max_influences    = 2,
            influence_indices = np.array([0, 1, 0, 1]),
            weights           = np.array([0.12345678, 0.87654322, 0.33333333, 0.66666667]),
            influences        = ["joint1", "joint2"],
        )

        # Round to 2 decimal places
        compact.round(2)

        # Check that weights are rounded
        weights_2d = compact.weights.reshape(-1, 2)
        self.assertTrue(np.allclose(weights_2d[0, 0], 0.12))
        self.assertTrue(
            np.allclose(weights_2d[0, 1], 0.88)
        )  # Should be adjusted to sum to 1

        # Should still be valid after rounding
        self.assertTrue(compact.valid)

    def test_compact_to_skin_data(self):
        """Test CompactSkinData.to_skin_data() conversion"""
        # Convert compact back to full SkinData
        converted = self.compact_skin.to_skin_data()

        # Should have same influences
        self.assertEqual(converted.influences, self.compact_skin.influences)

        # Should be valid
        self.assertTrue(converted.valid)

        # Weights should match original (approximately)
        self.assertTrue(converted.is_equivalent(self.skin_data))

        # Test round-trip conversion
        compact2   = converted.to_compact_skin_data()
        converted2 = compact2.to_skin_data()
        self.assertTrue(converted.is_equivalent(converted2))


class TestSkinDataExtended(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.skin_data = SkinData(
            name       = "test_skin",
            weights    = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]),
            influences = ["joint1", "joint2"],
        )

    def test_extend(self):
        """Test SkinData.extend() method"""
        skin = self.skin_data.copy()

        # Extend with new influences (zero weights by default)
        skin.extend(["joint3", "joint4"])

        # Should have 4 influences now
        self.assertEqual(len(skin.influences), 4)
        self.assertIn("joint3", skin.influences)
        self.assertIn("joint4", skin.influences)

        # New influences should have zero weights
        joint3_idx = skin.index("joint3")
        joint4_idx = skin.index("joint4")
        self.assertTrue(np.all(skin.weights[:, joint3_idx] == 0))
        self.assertTrue(np.all(skin.weights[:, joint4_idx] == 0))

        # Should still be valid
        self.assertTrue(skin.valid)

    def test_extend_with_weights(self):
        """Test SkinData.extend() with custom weights"""
        skin = self.skin_data.copy()

        # Extend with new influences and weights
        skin.extend(["joint3", "joint4"], weights=[0.1, 0.2])

        # Should have 4 influences
        self.assertEqual(len(skin.influences), 4)

        # New influences should have specified weights (before normalization)
        joint3_idx = skin.index("joint3")
        joint4_idx = skin.index("joint4")

        # After normalization, weights won't be exactly 0.1 and 0.2
        # but they should be non-zero
        self.assertTrue(np.all(skin.weights[:, joint3_idx] > 0))
        self.assertTrue(np.all(skin.weights[:, joint4_idx] > 0))

        # Should be normalized
        self.assertTrue(skin.valid)

    def test_extend_error_cases(self):
        """Test SkinData.extend() error handling"""
        skin = self.skin_data.copy()

        # Should raise error if influence already exists
        with self.assertRaises(ValueError):
            skin.extend(["joint1"])

        # Should raise error if number of influences doesn't match number of weights
        with self.assertRaises(ValueError):
            skin.extend(["joint3", "joint4"], weights=[0.1])

    def test_pattern_properties(self):
        """Test pattern property getters and setters"""
        skin = self.skin_data.copy()

        # Test default patterns
        self.assertIsNotNone(skin.patterns)
        self.assertIn("positive", skin.patterns)
        self.assertIn("negative", skin.patterns)

        # Test positive_patterns property
        pos = skin.positive_patterns
        self.assertIsInstance(pos, list)

        # Test negative_patterns property
        neg = skin.negative_patterns
        self.assertIsInstance(neg, list)

        # Test setting patterns
        from cgmath.geometry.skin_weights import Patterns

        skin.patterns = Patterns.SHORT
        self.assertEqual(skin.patterns,          Patterns.SHORT)
        self.assertEqual(skin.positive_patterns, Patterns.SHORT["positive"])
        self.assertEqual(skin.negative_patterns, Patterns.SHORT["negative"])

    def test_size_property(self):
        """Test size property"""
        skin = self.skin_data.copy()

        # Size should equal number of vertices (rows in weights matrix)
        self.assertEqual(skin.size, skin.weights.shape[0])
        self.assertEqual(skin.size, 3)

        # Test with different sized skin
        skin2 = SkinData(
            name="test",
            weights=np.array(
                [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1]]
            ),
            influences=["joint1", "joint2"],
        )
        self.assertEqual(skin2.size, 5)

        # Test empty skin
        empty_skin = SkinData(
            name       = "empty",
            weights    = np.array([]).reshape(0, 2),
            influences = ["joint1", "joint2"],
        )
        self.assertEqual(empty_skin.size, 0)

    def test_counts_property(self):
        """Test counts property"""
        skin = SkinData(
            name="test",
            weights=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],  # 1 influence
                    [0.5, 0.5, 0.0, 0.0],  # 2 influences
                    [0.4, 0.3, 0.3, 0.0],  # 3 influences
                    [0.25, 0.25, 0.25, 0.25],  # 4 influences
                ]
            ),
            influences=["joint1", "joint2", "joint3", "joint4"],
        )

        # Test counts for each vertex
        counts = skin.counts
        self.assertEqual(counts.shape[0], 4)
        self.assertEqual(counts[0],       1)  # First vertex has 1 non-zero weight
        self.assertEqual(counts[1],       2)  # Second vertex has 2 non-zero weights
        self.assertEqual(counts[2],       3)  # Third vertex has 3 non-zero weights
        self.assertEqual(counts[3],       4)  # Fourth vertex has 4 non-zero weights

        # Test with small non-zero values (should be treated as zero)
        skin_epsilon = SkinData(
            name="epsilon_test",
            weights=np.array(
                [
                    [0.5, 0.5, 1e-10, 1e-10],  # Should count as 2 influences
                    [0.3, 0.3, 0.4, 1e-15],  # Should count as 3 influences
                ]
            ),
            influences=["joint1", "joint2", "joint3", "joint4"],
        )
        counts_epsilon = skin_epsilon.counts
        self.assertEqual(counts_epsilon[0], 2)
        self.assertEqual(counts_epsilon[1], 3)


class TestSkinDataSymmetry(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Create a simple symmetric mesh (cube) - use float type
        self.mesh_points = np.array(
            [
                [-1.0, -1.0, -1.0],  # 0
                [1.0, -1.0, -1.0],  # 1
                [-1.0, 1.0, -1.0],  # 2
                [1.0, 1.0, -1.0],  # 3
                [-1.0, -1.0, 1.0],  # 4
                [1.0, -1.0, 1.0],  # 5
                [-1.0, 1.0, 1.0],  # 6
                [1.0, 1.0, 1.0],  # 7
            ]
        )

        self.mesh = MeshData(
            points=self.mesh_points,
            indices=np.array(
                [0, 1, 3, 2, 4, 5, 7, 6, 0, 4, 6, 2, 1, 5, 7, 3, 0, 1, 5, 4, 2, 3, 7, 6]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

        # Create symmetric skin weights
        self.skin_symmetric = SkinData(
            name="symmetric",
            weights=np.array(
                [
                    [0.8, 0.2, 0.0, 0.0],  # left vertex
                    [0.0, 0.0, 0.8, 0.2],  # right vertex (mirrored)
                    [0.7, 0.3, 0.0, 0.0],  # left vertex
                    [0.0, 0.0, 0.7, 0.3],  # right vertex (mirrored)
                    [0.9, 0.1, 0.0, 0.0],  # left vertex
                    [0.0, 0.0, 0.9, 0.1],  # right vertex (mirrored)
                    [0.6, 0.4, 0.0, 0.0],  # left vertex
                    [0.0, 0.0, 0.6, 0.4],  # right vertex (mirrored)
                ]
            ),
            influences=[
                "joint_l_shoulder",
                "joint_l_elbow",
                "joint_r_shoulder",
                "joint_r_elbow",
            ],
        )

        # Create asymmetric skin weights
        self.skin_asymmetric            = self.skin_symmetric.copy()
        self.skin_asymmetric.weights[1] = [0.1, 0.1, 0.6, 0.2]  # Break symmetry

    def test_get_symmetry_map(self):
        """Test get_symmetry_map() method"""
        result = self.skin_symmetric.get_symmetry_map(
            positive_patterns=["*_l_*"], negative_patterns=["*_r_*"]
        )

        self.assertIsNotNone(result)
        mirror_map, positive_indices, negative_indices, center_indices = result

        # Should have mirror mappings
        self.assertEqual(len(mirror_map), 4)

        # Positive and negative should have same length
        self.assertEqual(len(positive_indices), len(negative_indices))

        # joint_l_shoulder should map to joint_r_shoulder
        l_idx = self.skin_symmetric.index("joint_l_shoulder")
        r_idx = self.skin_symmetric.index("joint_r_shoulder")
        self.assertEqual(mirror_map[l_idx], r_idx)
        self.assertEqual(mirror_map[r_idx], l_idx)

    def test_symmetrical(self):
        """Test symmetrical() method"""
        # Test symmetric skin
        is_sym = self.skin_symmetric.symmetrical(
            self.mesh,
            positive_patterns=["*_l_*"],
            negative_patterns=["*_r_*"],
            pivot=0.0,
            axis=0,
        )
        self.assertTrue(is_sym)

        # Test asymmetric skin
        is_sym = self.skin_asymmetric.symmetrical(
            self.mesh,
            positive_patterns=["*_l_*"],
            negative_patterns=["*_r_*"],
            pivot=0.0,
            axis=0,
        )
        self.assertFalse(is_sym)

    def test_get_asymmetric_weights(self):
        """Test get_asymmetric_weights() method"""
        # Symmetric skin should have no asymmetric vertices
        asym_indices = self.skin_symmetric.get_asymmetric_weights(
            self.mesh,
            positive_patterns=["*_l_*"],
            negative_patterns=["*_r_*"],
            pivot=0.0,
            axis=0,
        )

        if asym_indices is not None:
            self.assertEqual(len(asym_indices), 0)

        # Asymmetric skin should have asymmetric vertices
        asym_indices = self.skin_asymmetric.get_asymmetric_weights(
            self.mesh,
            positive_patterns=["*_l_*"],
            negative_patterns=["*_r_*"],
            pivot=0.0,
            axis=0,
        )

        self.assertIsNotNone(asym_indices)
        self.assertGreater(len(asym_indices), 0)

    def test_fix_symmetry(self):
        """Test fix_symmetry() method"""
        skin = self.skin_asymmetric.copy()

        # Fix symmetry by copying from positive to negative side
        result = skin.fix_symmetry(
            self.mesh,
            positive_patterns=["*_l_*"],
            negative_patterns=["*_r_*"],
            pivot=0.0,
            axis=0,
            side=1.0,  # Copy from positive (left) to negative (right)
        )

        # Should report success
        self.assertTrue(result)

        # Should be symmetric now
        is_sym = skin.symmetrical(
            self.mesh,
            positive_patterns=["*_l_*"],
            negative_patterns=["*_r_*"],
            pivot=0.0,
            axis=0,
        )
        self.assertTrue(is_sym)


class TestSkinDataWeightProcessing(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Create a simple grid mesh for testing
        points = []
        for y in range(4):
            for x in range(4):
                points.append([x, y, 0])

        self.mesh = MeshData(
            points=np.array(points),
            indices=np.array(
                [
                    0,
                    1,
                    5,
                    4,
                    1,
                    2,
                    6,
                    5,
                    2,
                    3,
                    7,
                    6,
                    4,
                    5,
                    9,
                    8,
                    5,
                    6,
                    10,
                    9,
                    6,
                    7,
                    11,
                    10,
                    8,
                    9,
                    13,
                    12,
                    9,
                    10,
                    14,
                    13,
                    10,
                    11,
                    15,
                    14,
                ]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4, 4, 4, 4]),
        )

        # Create skin data with some variation
        weights = np.random.rand(16, 3)
        weights = weights / weights.sum(axis=1, keepdims=True)

        self.skin_data = SkinData(
            name="test_skin", weights=weights, influences=["joint1", "joint2", "joint3"]
        )

    def test_smooth(self):
        """Test smooth() method"""
        skin             = self.skin_data.copy()
        original_weights = skin.weights.copy()

        # Get mesh neighbors
        neighbors = self.mesh.get_edge_vertex_neighbors()

        # Smooth specific vertices
        skin.smooth(neighbors, indices=[4, 5, 6], iterations=1, receptions=0.5)

        # Weights should have changed for smoothed vertices
        self.assertFalse(np.allclose(skin.weights[4], original_weights[4]))
        self.assertFalse(np.allclose(skin.weights[5], original_weights[5]))
        self.assertFalse(np.allclose(skin.weights[6], original_weights[6]))

        # Weights should be unchanged for non-smoothed vertices
        self.assertTrue(np.allclose(skin.weights[0], original_weights[0]))

        # Should still be valid
        self.assertTrue(skin.valid)

    def test_inpaint(self):
        """Test inpaint() method"""
        skin = self.skin_data.copy()

        # Zero out some weights to simulate missing data
        skin.weights[5] = 0
        skin.weights[6] = 0

        # Get mesh neighbors
        neighbors = self.mesh.get_edge_vertex_neighbors()

        # Inpaint the zeroed weights
        skin.inpaint(neighbors, indices=[5, 6], iterations=5, receptions=0.5)

        # Inpainted weights should no longer be zero
        self.assertFalse(np.allclose(skin.weights[5], 0))
        self.assertFalse(np.allclose(skin.weights[6], 0))

        # Should still be valid
        self.assertTrue(skin.valid)

    def test_subdivide(self):
        """Test subdivide() method"""
        skin          = self.skin_data.copy()
        original_size = skin.size

        # Subdivide once
        skin.subdivide(self.mesh, steps=1, keep_size=False)

        # Should have more vertices after subdivision
        self.assertGreater(skin.size, original_size)

        # Should still be valid
        self.assertTrue(skin.valid)

    def test_subdivide_keep_size(self):
        """Test subdivide() with keep_size=True"""
        skin          = self.skin_data.copy()
        original_size = skin.size

        # Subdivide but keep original size
        skin.subdivide(self.mesh, steps=1, keep_size=True)

        # Should have same number of vertices
        self.assertEqual(skin.size, original_size)

        # Should still be valid
        self.assertTrue(skin.valid)


class TestSkinListExtended(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.skin1 = SkinData(
            name       = "skin1",
            weights    = np.array([[0.8, 0.2], [0.6, 0.4]]),
            influences = ["joint1", "joint2"],
        )

        self.skin2 = SkinData(
            name       = "skin2",
            weights    = np.array([[0.7, 0.3], [0.5, 0.5]]),
            influences = ["joint1", "joint2"],
        )

        self.skin_list = SkinList([self.skin1, self.skin2])

    def test_transfer_influences_list(self):
        """Test SkinList.transfer_influences() method"""
        skin_list = self.skin_list.copy()

        # Transfer weights from joint1 to joint2
        skin_list.transfer_influences(["joint1"], ["joint2"], weighted=True)

        # All weights should now be on joint2
        for skin in skin_list:
            self.assertTrue(np.allclose(skin.weights[:, 0], 0))
            self.assertTrue(np.allclose(skin.weights[:, 1], 1.0))
            self.assertTrue(skin.valid)