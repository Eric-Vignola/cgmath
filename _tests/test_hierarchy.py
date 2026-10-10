import copy
import gc
import itertools
import os
import pickle
import random
import sys
import tempfile
import unittest
import uuid as uuid_module
import warnings
import weakref
from collections import deque
from unittest import mock

import numpy as np
from cgmath.formats.glb import pygltflib
from cgmath.hierarchy import HierarchyData, TransformData, TransformList, generate_uuid
from cgmath.hierarchy import validate_uuid
from cgmath.hierarchy import _fbx_enum, _rotation, FBX
from cgmath._tests._maya_rotate_order import maya_rotate_order_cases
from cgmath.transforms import (
    euler_to_matrix,
    euler_to_quaternion,
    matrix_multiply,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_matrix,
)

if pygltflib is not None:
    from pygltflib import (
        Accessor,
        Animation,
        AnimationChannel,
        AnimationChannelTarget,
        AnimationSampler,
        Buffer,
        BufferView,
        FLOAT,
        GLTF2,
        Mesh,
        Node,
        Scene,
    )

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestHierarchy(unittest.TestCase):
    def setUp(self):
        super().setUp()

        root, fname = os.path.split(__file__)
        fname, ext = os.path.splitext(fname)

        self.skeleton_file = os.path.join(root, "test_assets", fname + ".skeleton.json")
        self.results       = os.path.join(root, "test_assets", fname + ".results.npz")

        # .npz rather than .pkl: pickled ndarrays are not portable across NumPy
        # major versions (a NumPy 2 pickle references numpy._core, which Maya's
        # NumPy 1.24 does not have); the .npy container is version-agnostic.
        with np.load(self.results) as archive:
            self.results = {key: archive[key] for key in archive.files}


    def test_serialize(self):

        skeleton = HierarchyData.load(self.skeleton_file)

        # save the data to a scratch folder, never the user's home
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pkl_file  = skeleton.save(os.path.join(tmp.name, "skeleton.pkl"),  mode="pkl")
        npz_file  = skeleton.save(os.path.join(tmp.name, "skeleton.npz"),  mode="npz")
        json_file = skeleton.save(os.path.join(tmp.name, "skeleton.json"), mode="json")

        # reload from their respective formats
        pkl  = HierarchyData.load(pkl_file,  mode="pkl")
        npz  = HierarchyData.load(npz_file,  mode="npz")
        json = HierarchyData.load(json_file, mode="json")

        # data all matches source
        same = [skeleton == x for x in [pkl, npz, json]]
        self.assertTrue(all(same))

    def test_parenting(self):

        # test new parent world
        skeleton = HierarchyData.load(self.skeleton_file)
        obj      = TransformData(name='new_root', rotate=np.array([30,-45,257]))
        skeleton.insert(0, obj)
        for node in skeleton.get_roots()[1:]:
            node.set_parent(obj)
            self.assertEqual(node.parent_name, 'new_root')

        self.assertTrue(np.allclose(self.results['new_parent_world'], skeleton.matrix))


        # test new parent local
        skeleton = HierarchyData.load(self.skeleton_file)
        obj      = TransformData(name='new_root', rotate=np.array([30,-45,257]))
        skeleton.insert(0, obj)
        for node in skeleton.get_roots()[1:]:
            node.set_parent(obj, world_space=False)
            self.assertEqual(node.parent_name, 'new_root')

        self.assertTrue(np.allclose(self.results['new_parent_local'], skeleton.matrix))


        # Weird reparenting
        skeleton = HierarchyData.load(self.skeleton_file)
        skeleton['hip_left_joint'].set_parent(None)
        skeleton['hip_right_joint'].set_parent(None)
        skeleton['clavicle_left_joint'].set_parent('head_joint', world_space=False)
        skeleton['clavicle_right_joint'].set_parent('head_joint', world_space=False)
        self.assertTrue(np.allclose(self.results['weird_reparanting'], skeleton.matrix))


        # parent everything to the world
        skeleton = HierarchyData.load(self.skeleton_file)
        skeleton.set_parent(None)
        self.assertTrue(np.allclose(self.results['parenting_all_to_world'], skeleton.matrix))


        # parent everything willy nilly to something silly
        skeleton = HierarchyData.load(self.skeleton_file)
        skeleton.set_parent("handThumb_02_left_joint")
        self.assertTrue(np.allclose(self.results['willy_nilly_parenting'], skeleton.matrix))

        # use name matching to parent to something silly
        skeleton = HierarchyData.load(self.skeleton_file)
        skeleton.match("*_twist_*").set_parent("pelvis_joint")
        self.assertTrue(np.allclose(self.results['name_match_reparenting'], skeleton.matrix))



        # extract a node to poke around
        node = skeleton['elbow_left_joint']

        # --- simple queries --- #
        # is this node's name unique?
        self.assertTrue(node.unique_name is True)

        # who's this node's parent
        self.assertTrue(node.get_parent() == 'shoulder_left_joint') # shoulder_left_joint

        # find a path from the root to this node
        self.assertTrue(node.get_root().name == ['root_joint', 'pelvis_joint', 'spineLower_joint', 'spineMiddle_joint', 'spineUpper_joint', 'chest_joint', 'clavicle_left_joint', 'shoulder_left_joint', 'elbow_left_joint']) 

        # find this node's children
        self.assertTrue(node.get_children().name == ['elbow_partial_left_joint', 'handWrist_left_joint'])

        # find this node's branch (self + all decendents)
        self.assertTrue(node.get_branch().name == ['elbow_left_joint', 'elbow_partial_left_joint', 'handWrist_left_joint', 'handIndexMeta_left_joint', 'handIndex_00_left_joint', 'handIndex_01_left_joint', 'handIndex_02_left_joint', 'handMiddleMeta_left_joint', 'handMiddle_00_left_joint', 'handMiddle_01_left_joint', 'handMiddle_02_left_joint', 'handWrist_partial_left_joint', 'handPinkyMeta_left_joint', 'handPinky_00_left_joint', 'handPinky_01_left_joint', 'handPinky_02_left_joint', 'handRingMeta_left_joint', 'handRing_00_left_joint', 'handRing_01_left_joint', 'handRing_02_left_joint', 'handThumbMeta_left_joint', 'handThumb_00_left_joint', 'handThumb_01_left_joint', 'handThumb_02_left_joint']) 



        # add a prefix to the entire skeleton and verify it affected the node
        skeleton.add_prefix('skeleton_')
        self.assertTrue(node == 'skeleton_elbow_left_joint')                  # skeleton_elbow_left_joint
        self.assertTrue(node.get_parent() == 'skeleton_shoulder_left_joint')  # skeleton_shoulder_left_joint

        # add a suffix only to this node's children
        node.get_children().add_suffix('_JOINT')
        self.assertTrue(skeleton.name == ['skeleton_root_joint', 'skeleton_pelvis_joint', 'skeleton_spineLower_joint', 'skeleton_spineMiddle_joint', 'skeleton_spineUpper_joint', 'skeleton_chest_joint', 'skeleton_clavicle_left_joint', 'skeleton_shoulder_left_joint', 'skeleton_elbow_left_joint', 'skeleton_elbow_partial_left_joint_JOINT', 'skeleton_handWrist_left_joint_JOINT', 'skeleton_handIndexMeta_left_joint', 'skeleton_handIndex_00_left_joint', 'skeleton_handIndex_01_left_joint', 'skeleton_handIndex_02_left_joint', 'skeleton_handMiddleMeta_left_joint', 'skeleton_handMiddle_00_left_joint', 'skeleton_handMiddle_01_left_joint', 'skeleton_handMiddle_02_left_joint', 'skeleton_handWrist_partial_left_joint', 'skeleton_handPinkyMeta_left_joint', 'skeleton_handPinky_00_left_joint', 'skeleton_handPinky_01_left_joint', 'skeleton_handPinky_02_left_joint', 'skeleton_handRingMeta_left_joint', 'skeleton_handRing_00_left_joint', 'skeleton_handRing_01_left_joint', 'skeleton_handRing_02_left_joint', 'skeleton_handThumbMeta_left_joint', 'skeleton_handThumb_00_left_joint', 'skeleton_handThumb_01_left_joint', 'skeleton_handThumb_02_left_joint', 'skeleton_neck_joint', 'skeleton_head_joint', 'skeleton_eye_offset_right_joint', 'skeleton_eye_right_joint', 'skeleton_eye_offset_left_joint', 'skeleton_eye_left_joint', 'skeleton_muzzle_joint', 'skeleton_dentureTp_joint', 'skeleton_dentureBt_joint', 'skeleton_tongueBase_joint', 'skeleton_tongueTip_joint', 'skeleton_clavicle_right_joint', 'skeleton_shoulder_right_joint', 'skeleton_elbow_right_joint', 'skeleton_elbow_partial_right_joint', 'skeleton_handWrist_right_joint', 'skeleton_handIndexMeta_right_joint', 'skeleton_handIndex_00_right_joint', 'skeleton_handIndex_01_right_joint', 'skeleton_handIndex_02_right_joint', 'skeleton_handMiddleMeta_right_joint', 'skeleton_handMiddle_00_right_joint', 'skeleton_handMiddle_01_right_joint', 'skeleton_handMiddle_02_right_joint', 'skeleton_handWrist_partial_right_joint', 'skeleton_handPinkyMeta_right_joint', 'skeleton_handPinky_00_right_joint', 'skeleton_handPinky_01_right_joint', 'skeleton_handPinky_02_right_joint', 'skeleton_handRingMeta_right_joint', 'skeleton_handRing_00_right_joint', 'skeleton_handRing_01_right_joint', 'skeleton_handRing_02_right_joint', 'skeleton_handThumbMeta_right_joint', 'skeleton_handThumb_00_right_joint', 'skeleton_handThumb_01_right_joint', 'skeleton_handThumb_02_right_joint', 'skeleton_hip_left_joint', 'skeleton_knee_left_joint', 'skeleton_footAnkle_left_joint', 'skeleton_footBall_left_joint', 'skeleton_footToe_left_joint', 'skeleton_footAnkle_partial_left_joint', 'skeleton_knee_partial_left_joint', 'skeleton_hip_right_joint', 'skeleton_knee_right_joint', 'skeleton_footAnkle_right_joint', 'skeleton_footBall_right_joint', 'skeleton_footToe_right_joint', 'skeleton_footAnkle_partial_right_joint', 'skeleton_knee_partial_right_joint', 'skeleton_armLower_twist_01_left_joint', 'skeleton_armLower_twist_02_left_joint', 'skeleton_armLower_twist_03_left_joint', 'skeleton_armUpper_twist_01_left_joint', 'skeleton_armUpper_twist_02_left_joint', 'skeleton_armUpper_twist_03_left_joint', 'skeleton_neck_twist_01_joint', 'skeleton_neck_twist_02_joint', 'skeleton_armLower_twist_01_right_joint', 'skeleton_armLower_twist_02_right_joint', 'skeleton_armLower_twist_03_right_joint', 'skeleton_armUpper_twist_01_right_joint', 'skeleton_armUpper_twist_02_right_joint', 'skeleton_armUpper_twist_03_right_joint', 'skeleton_legLower_twist_01_left_joint', 'skeleton_legLower_twist_02_left_joint', 'skeleton_legLower_twist_03_left_joint', 'skeleton_legUpper_twist_01_left_joint', 'skeleton_legUpper_twist_02_left_joint', 'skeleton_legUpper_twist_03_left_joint', 'skeleton_legLower_twist_01_right_joint', 'skeleton_legLower_twist_02_right_joint', 'skeleton_legLower_twist_03_right_joint', 'skeleton_legUpper_twist_01_right_joint', 'skeleton_legUpper_twist_02_right_joint', 'skeleton_legUpper_twist_03_right_joint'])

    def test_append_assigns_uuid_and_adds_node(self):
        # a freshly built node has no uuid; append must assign one AND
        # actually add the node to the list (regression: the old append
        # left list.append unreachable after `break`)
        skeleton = HierarchyData()
        node     = TransformData(name='solo')
        self.assertIsNone(node.uuid)

        skeleton.append(node)
        self.assertEqual(len(skeleton), 1)
        self.assertIsNotNone(node.uuid)
        self.assertTrue(node in skeleton)
        self.assertTrue(node._hierarchy is skeleton)

        # re-appending the same node raises (it already has a uuid)
        with self.assertRaises(RuntimeError):
            skeleton.append(node)

        # wrong-typed input raises TypeError (not AttributeError)
        with self.assertRaises(TypeError):
            skeleton.append("not a node")

    def test_set_parent_local_invalidates_world_cache(self):
        # set_parent(world_space=False) only changes parent_uuid; any
        # cached world matrix on the node must be invalidated so reads
        # reflect the new parent (regression: cache was left stale)
        skeleton = HierarchyData.load(self.skeleton_file)
        node     = skeleton['elbow_left_joint']

        # prime the cached world matrix against the original parent
        stale = node.world_matrix.copy()

        node.set_parent('head_joint', world_space=False)
        new_parent = skeleton['head_joint']
        expected   = np.dot(node.matrix, new_parent.world_matrix)

        self.assertTrue(np.allclose(node.world_matrix, expected))
        # head_joint != original parent, so the value must have changed
        self.assertFalse(np.allclose(node.world_matrix, stale))

    def test_equality_without_uuid_does_not_crash(self):
        # comparing nodes that have no uuid yet must not raise
        # (regression: validate_uuid did not catch TypeError on None)
        a = TransformData(name='a')
        b = TransformData(name='b')
        self.assertFalse(a == b)
        self.assertFalse(a == 'b')

    def test_channel_write_is_in_place(self):
        # a channel bound to a view must keep writing through it, so external
        # storage sees the write and the binding survives
        storage         = np.zeros((2, 3))
        node            = TransformData(name='a')
        node._translate = storage[1]

        node.translate  = [1.0, 2.0, 3.0]

        self.assertTrue(np.allclose(storage[1], [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(storage[0], 0.0))
        self.assertTrue(np.shares_memory(node._translate, storage))

    def test_channel_write_does_not_mutate_the_class_default(self):
        # copy() goes through from_dict rather than __init__, and the channel
        # it leaves at its default must not be the class default, which the
        # class shares
        node       = TransformData(name='a').copy()
        node.scale = [3.0, 3.0, 3.0]

        self.assertTrue(np.allclose(node.scale, 3.0))
        self.assertTrue(np.allclose(TransformData._scale, 1.0))
        self.assertTrue(np.allclose(TransformData(name='b').scale, 1.0))

    def test_a_small_channel_write_far_from_the_origin_is_a_change(self):
        # the change check was relative to the value: 0.009 at x = 1000 is
        # below it, so the move kept the cached world matrix at 1000
        node = TransformData(name='a', translate=[1000.0, 0.0, 0.0])
        node.world_matrix

        node.translate = [1000.009, 0.0, 0.0]

        self.assertAlmostEqual(node.world_matrix[3, 0], 1000.009, places=9)

    def test_a_small_matrix_write_far_from_the_origin_is_a_change(self):
        # the matrix setters had the same check: a matrix moved 0.009 at
        # x = 1000 was taken for the one the node already held
        for setter in ('matrix', 'world_matrix'):
            with self.subTest(setter):
                node    = TransformData(name='a', translate=[1000.0, 0.0, 0.0])
                M       = getattr(node, setter)
                M[3, 0] = 1000.009
                setattr(node, setter, M)

                self.assertAlmostEqual(node.translate[0], 1000.009, places=9)
                self.assertAlmostEqual(node.world_matrix[3, 0], 1000.009, places=9)

    def test_a_world_matrix_written_back_keeps_the_channels(self):
        # through the parent's inverse the local comes back off in its last
        # bits. that is no change: rebuilding the channels from it would turn
        # rotate [0, 180, 0] into [180, 0, -180], the same rotation
        h      = HierarchyData()
        parent = TransformData(name='p', translate=[3.0, 1.0, -2.0], rotate=[10.0, 20.0, 30.0])
        child = TransformData(
            name        = 'c',
            translate   = [1.0, 2.0, 3.0],
            rotate      = [0.0, 180.0, 0.0],
            rotate_axis = [15.0, 0.0, 0.0],
        )
        h.append(parent)
        h.append(child)
        child.set_parent('p', world_space=False)

        child.set_parent('p', world_space=True)

        self.assertTrue(np.allclose(child.rotate, [0.0, 180.0, 0.0]))
        self.assertTrue(np.allclose(child.rotate_axis, [15.0, 0.0, 0.0]))

    def test_channel_defaults_are_float(self):
        # an integer default would silently truncate an in place write
        node        = TransformData(name='a').copy()
        node.rotate = [1.5, 2.5, 3.5]

        self.assertTrue(np.allclose(node.rotate, [1.5, 2.5, 3.5]))
        for channel in ('_scale', '_rotate', '_translate', '_rotate_axis',
                        '_joint_orient'):
            default = np.asarray(getattr(TransformData, channel))
            self.assertEqual(default.dtype, np.float64)

    def test_channel_write_with_the_wrong_length_names_the_channel(self):
        # the constructor names the channel and the shape it needs; a write
        # through the property has to say the same, not numpy's broadcast error
        node = TransformData(name='a', translate=[1.0, 2.0, 3.0])

        for channel in ('scale', 'rotate', 'translate', 'rotate_axis', 'joint_orient'):
            message = rf'^TransformData\.{channel} must have shape \(3,\), got \(2,\)$'
            with self.subTest(channel=channel):
                with self.assertRaisesRegex(ValueError, message):
                    setattr(node, channel, (1, 2))
                with self.assertRaisesRegex(ValueError, message):
                    TransformData(name='b', **{channel: (1, 2)})

        # the refused write changed nothing, and one number still fills all three
        self.assertTrue(np.allclose(node.translate, [1.0, 2.0, 3.0]))
        node.scale = 2.0
        self.assertTrue(np.allclose(node.scale, [2.0, 2.0, 2.0]))

    def test_channel_write_takes_a_one_row_value(self):
        # a transforms function returns a (1, 3) row for one matrix, and a
        # channel write has always taken it, as numpy writes it into the (3,)
        # channel; checking the length must not refuse it, nor one number
        # held in a list or a (1,) array
        matrix = euler_to_matrix(np.radians([10.0, 20.0, 30.0]), 0)[0]
        node   = TransformData(name='a')

        node.rotate = np.degrees(matrix_to_euler(matrix, 0))
        self.assertTrue(np.allclose(node.rotate, [10.0, 20.0, 30.0]))

        for value in ([5.0], np.array([5.0]), np.array([[5.0]]), np.array([[1.0, 2.0, 3.0]])):
            with self.subTest(value=value):
                node.translate = value
                self.assertTrue(np.allclose(node.translate, np.ravel(value)))
                self.assertEqual(node._translate.shape, (3,))

        # a column is not a row: numpy can not write it, so the channel says so
        message = r'^TransformData\.translate must have shape \(3,\), got \(3, 1\)$'
        with self.assertRaisesRegex(ValueError, message):
            node.translate = np.array([[1.0], [2.0], [3.0]])

    def test_building_from_another_hierarchy_keeps_every_node(self):
        # append takes ownership by removing the node from its previous
        # hierarchy, so extend has to iterate a snapshot; iterating the live
        # list skipped every other node and left both objects half a rig
        ids = [generate_uuid() for _ in range(6)]
        source = HierarchyData([
            TransformData(name=f'j{i}', uuid=ids[i],
                          parent_uuid=None if i == 0 else ids[i - 1])
            for i in range(6)
        ])

        moved = HierarchyData(source)

        self.assertEqual(moved.name, ['j0', 'j1', 'j2', 'j3', 'j4', 'j5'])
        self.assertEqual(len(source), 0)
        self.assertTrue(all(node._hierarchy is moved for node in moved))

    def test_match_rotate_invalid_length_raises(self):
        # a 1-D sequence that is neither a length-3 euler nor a length-4
        # quaternion must raise a clear error (regression: UnboundLocalError)
        skeleton = HierarchyData.load(self.skeleton_file)
        node     = skeleton['head_joint']
        with self.assertRaises(ValueError):
            node.match_rotate([1.0, 2.0])

    def test_indexing_does_not_steal_hierarchy(self):
        # slicing / fancy-indexing a HierarchyData must return a
        # NON-owning view; source nodes keep pointing at the real owner
        skeleton = HierarchyData.load(self.skeleton_file)
        node     = skeleton['elbow_left_joint']

        sub = skeleton[1:]
        self.assertFalse(isinstance(sub, HierarchyData))
        self.assertTrue(node._hierarchy is skeleton)
        for n in sub:
            self.assertTrue(n._hierarchy is skeleton)

        fancy = skeleton[np.array([2, 5, 7])]
        for n in fancy:
            self.assertTrue(n._hierarchy is skeleton)

        # hierarchy queries still resolve correctly after indexing
        self.assertTrue(node.get_parent() == 'shoulder_left_joint')

    def test_index_by_node_or_uuid_counts_a_negative_bound_from_the_end(self):
        # a node or its uuid is looked up apart from a name; every start and
        # stop must find what list.index finds for all three, negative ones too
        names = ['a', 'b', 'c', 'd']
        rig   = HierarchyData([TransformData(name=n) for n in names])
        other = HierarchyData([TransformData(name='e')])['e']

        for name, node in [(n, rig[n]) for n in names] + [('e', other)]:
            for start in range(-5, 6):
                for stop in list(range(-5, 6)) + [None]:
                    bounds = (start,) if stop is None else (start, stop)
                    try:
                        expected = names.index(name, *bounds)
                    except ValueError:
                        expected = ValueError
                    for key in (node, node.uuid, name):
                        with self.subTest(key=key, start=start, stop=stop):
                            if expected is ValueError:
                                with self.assertRaises(ValueError):
                                    rig.index(key, start, stop)
                            else:
                                self.assertEqual(rig.index(key, start, stop), expected)

    def test_boolean_mask_indexing(self):
        # a boolean mask selects nodes by position; numpy booleans must not
        # be read as the indices 0 and 1
        skeleton = HierarchyData.load(self.skeleton_file)
        names    = [n.name for n in skeleton]

        mask         = np.zeros(len(skeleton), dtype=bool)
        mask[[1, 3]] = True

        selected = skeleton[mask]
        self.assertEqual([n.name for n in selected], [names[1], names[3]])

        # still a non-owning view
        self.assertFalse(isinstance(selected, HierarchyData))
        for n in selected:
            self.assertTrue(n._hierarchy is skeleton)

        with self.assertRaises(IndexError):
            skeleton[np.ones(len(skeleton) + 1, dtype=bool)]

    def test_reparent_across_hierarchies_detaches(self):
        # appending a node owned by one hierarchy into another must
        # detach it from the first (a node lives in exactly one list)
        h1 = HierarchyData()
        n  = TransformData(name='shared')
        h1.append(n)
        self.assertTrue(n._hierarchy is h1)

        h2 = HierarchyData()
        h2.append(n)
        self.assertTrue(n._hierarchy is h2)
        self.assertTrue(n in h2)
        self.assertFalse(n in h1)
        self.assertEqual(len(h1), 0)

    def test_pose_algebra_round_trips(self):
        # (a - b) + b == a for translate, rotation, and segment-scale-
        # compensated scale across the full skeleton
        a = HierarchyData.load(self.skeleton_file)
        b = HierarchyData.load(self.skeleton_file)
        for node in b:
            node.translate = node.translate + np.array([1.0, 2.0, 3.0])
            node.scale     = node.scale * 1.5
            node.rotate    = node.rotate + np.array([5.0, -3.0, 2.0])

        a_scale   = a.scale.copy()
        a_quat    = a.quaternion.copy()
        a_trans   = a.translate.copy()

        recovered = (a - b) + b

        self.assertTrue(np.allclose(recovered.scale, a_scale, atol=1e-6))
        self.assertTrue(np.allclose(recovered.translate, a_trans, atol=1e-6))
        self.assertTrue(np.allclose(recovered.quaternion, a_quat, atol=1e-5))

    def test_pose_algebra_requires_matching_names(self):
        a = HierarchyData.load(self.skeleton_file)
        b = HierarchyData()
        b.append(TransformData(name='totally_unrelated'))
        with self.assertRaises(ValueError):
            _ = a.get_delta(b, by='name')

    def test_pose_algebra_rejects_duplicate_names(self):
        a = HierarchyData()
        a.append(TransformData(name='dup'))
        a.append(TransformData(name='dup'))
        b = HierarchyData()
        b.append(TransformData(name='dup'))
        with self.assertRaises(ValueError):
            _ = a.get_delta(b, by='name')

    def _axis_convention_skeletons(self):
        # two skeletons over the same bones and the same world positions, one
        # aiming +X down the bone with its orientation in joint_orient, the
        # other aiming +Z with the orientation baked into rotate
        names     = ['shoulder', 'elbow', 'wrist']
        positions = np.array([[0., 0., 0.], [10., 0., 0.], [18., 0., 0.]])

        # rows hold the world direction of each joint's local X, Y and Z
        aim_x = np.eye(3)
        aim_z = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])

        def build(rotation):
            h = HierarchyData()
            for i, name in enumerate(names):
                h.append(TransformData(name, node_type='joint'))
                if i:
                    h[name].set_parent(names[i - 1], world_space=False)

            # parent first, so each assignment sees a solved parent
            for i, name in enumerate(names):
                M                    = np.eye(4)
                M[:3, :3]            = rotation
                M[3, :3]             = positions[i]
                h[name].world_matrix = M

            h.segment_scale_compensate = False
            return h

        a = build(aim_x)
        a.set_rotate_to_joint_orient()
        return a, build(aim_z)

    def test_pose_algebra_converts_joint_axes(self):
        # a delta between skeletons that differ only in axis convention must
        # re-pose correctly at ANY pose of the source -- a local rotation
        # difference is only valid at the pose it was measured in, so this
        # fails if the delta is applied one-sided instead of as a frame offset
        a, b = self._axis_convention_skeletons()
        delta = a - b

        # a general 3 axis pose, applied in each joint's own local frame
        posed = b.copy()
        for name, euler in (('shoulder', [20., 30., -15.]), ('elbow', [5., 40., 10.])):
            P                      = euler_to_matrix(np.radians(euler), 0)[0]
            current                = quaternion_to_matrix(posed[name].quaternion)[0]
            posed[name].quaternion = matrix_to_quaternion(np.dot(P, current))[0]

        result = delta + posed

        # same physical skeleton, so the joints must land on the source's
        self.assertTrue(
            allclose(result.world_matrix[:, 3, :3], posed.world_matrix[:, 3, :3])
        )

        # ...while keeping a's joint orients and its +X down-the-bone aim
        self.assertTrue(allclose(result.joint_orient, a.joint_orient))
        for i in range(len(result) - 1):
            aim  = result[i].world_matrix[0, :3]
            bone = result[i + 1].world_matrix[3, :3] - result[i].world_matrix[3, :3]
            aim  = aim / np.linalg.norm(aim)
            bone = bone / np.linalg.norm(bone)
            self.assertTrue(allclose(np.dot(aim, bone), 1.0))

    def test_pose_algebra_frame_delta_round_trips(self):
        # the axis-conversion delta is still an exact inverse at rest
        a, b = self._axis_convention_skeletons()
        recovered = (a - b) + b

        self.assertTrue(allclose(recovered.world_matrix, a.world_matrix))
        self.assertTrue(allclose(recovered.joint_orient, a.joint_orient))

    def test_pose_algebra_methods_match_operators(self):
        # get_delta / add_delta are the full form of - and +, and the receiver
        # flips: `delta + rig` is spelled `rig.add_delta(delta)`
        a, b = self._axis_convention_skeletons()

        self.assertTrue(allclose((a - b).world_matrix, a.get_delta(b).world_matrix))
        self.assertTrue(
            allclose(((a - b) + b).world_matrix, b.add_delta(a.get_delta(b)).world_matrix)
        )

    def test_pose_algebra_pairs_by_index(self):
        # rigs that are structurally identical but named differently cannot
        # pair on names, and must convert correctly through by="index"
        a, b = self._axis_convention_skeletons()
        b.add_prefix('flipped_')

        # names no longer line up, so name pairing is the one that refuses
        with self.assertRaises(ValueError):
            _ = a.get_delta(b, by='name')

        posed = b.copy()
        for name, euler in (
            ('flipped_shoulder', [20., 30., -15.]),
            ('flipped_elbow', [5., 40., 10.]),
        ):
            P                      = euler_to_matrix(np.radians(euler), 0)[0]
            current                = quaternion_to_matrix(posed[name].quaternion)[0]
            posed[name].quaternion = matrix_to_quaternion(np.dot(P, current))[0]

        result = posed.add_delta(a.get_delta(b, by='index'), by='index')

        # lands on the source's joints, keeps a's orients and its +X aim, and
        # hands back a's names rather than the flipped rig's
        self.assertTrue(
            allclose(result.world_matrix[:, 3, :3], posed.world_matrix[:, 3, :3])
        )
        self.assertTrue(allclose(result.joint_orient, a.joint_orient))
        self.assertEqual(result.name, a.name)
        for i in range(len(result) - 1):
            aim  = result[i].world_matrix[0, :3]
            bone = result[i + 1].world_matrix[3, :3] - result[i].world_matrix[3, :3]
            aim  = aim / np.linalg.norm(aim)
            bone = bone / np.linalg.norm(bone)
            self.assertTrue(allclose(np.dot(aim, bone), 1.0))

    def test_pose_algebra_index_pairing_rejects_structural_mismatch(self):
        # same length, different parenting: pairing on position would quietly
        # pair unrelated joints, so it has to raise
        a, b = self._axis_convention_skeletons()
        b['wrist'].set_parent('shoulder', world_space=False)

        with self.assertRaises(ValueError):
            _ = a.get_delta(b, by='index')

    def test_pose_algebra_index_pairing_rejects_length_mismatch(self):
        a, b = self._axis_convention_skeletons()
        b.pop()

        with self.assertRaises(ValueError):
            _ = a.get_delta(b, by='index')

    def test_pose_algebra_rejects_unknown_pairing(self):
        a, b = self._axis_convention_skeletons()

        with self.assertRaises(ValueError):
            _ = a.get_delta(b, by='uuid')

    def test_pose_algebra_partial_delta(self):
        # a delta may cover only part of a skeleton -- nodes outside the name
        # intersection pass through untouched
        a = HierarchyData.load(self.skeleton_file)
        b = HierarchyData.load(self.skeleton_file)

        # a sub-rig of b; its top joint still names a parent left behind,
        # which reads as a root
        branch = HierarchyData([node.copy() for node in b[0].get_branch()[1:]])
        self.assertGreater(len(branch), 1)
        self.assertLess(len(branch), len(b))

        for node in branch:
            node.rotate = node.rotate + np.array([5., -3., 2.])

        recovered = branch.add_delta(a.get_delta(branch, by='name'), by='name')

        # the delta spans all of `a`; the paired nodes round trip and the
        # unpaired ones still hold a's values
        self.assertEqual(len(recovered), len(a))
        covered = branch.name
        for node in recovered:
            self.assertTrue(allclose(node.quaternion, a[node.name].quaternion))
            if node.name not in covered:
                self.assertTrue(allclose(node.translate, a[node.name].translate))

    def test_matrix_setter_accounts_for_parent_scale_inverse(self):
        # the local matrix is [S][RO][R][JO][IS][T]; when a parent has
        # non-unit scale and the child is segment-scale-compensated, IS != I
        # and the matrix setter must strip it or the S/R decomposition (and
        # therefore the matrix round-trip) is wrong
        h      = HierarchyData()
        parent = TransformData(name='parent', scale=np.array([2.0, 3.0, 4.0]))
        child = TransformData(
            name='child',
            node_type='joint',
            scale=np.array([1.5, 0.5, 2.5]),
            rotate=np.array([10.0, 20.0, 30.0]),
            joint_orient=np.array([5.0, -5.0, 15.0]),
            translate=np.array([1.0, 2.0, 3.0]),
            segment_scale_compensate=True,
        )
        h.append(parent)
        h.append(child)
        child.set_parent('parent', world_space=False)

        # IS must be non-identity for this test to exercise the fix
        self.assertFalse(np.allclose(child.parent_scale_inverse, np.ones(3)))

        target       = child.matrix.copy()
        target_scale = child.scale.copy()

        # perturb, then restore via the matrix setter
        child.scale  = np.array([9.0, 9.0, 9.0])
        child.rotate = np.array([0.0, 0.0, 0.0])
        child.matrix = target

        self.assertTrue(np.allclose(child.matrix, target, atol=1e-6))
        self.assertTrue(np.allclose(child.scale, target_scale, atol=1e-6))

    def test_world_matrix_setter_round_trips_under_scaled_parent(self):
        # setting then reading world_matrix must round-trip even when the
        # parent carries compound scale (the parent world matrix then has
        # non-orthogonal rows, so the setter needs a true inverse, and the
        # local decomposition needs the IS factor)
        h = HierarchyData()
        gp = TransformData(
            name   = 'gp',
            scale  = np.array([2.0, 1.5, 0.5]),
            rotate = np.array([10.0, 0.0, 25.0]),
        )
        p = TransformData(
            name='p',
            node_type='joint',
            scale=np.array([1.2, 0.8, 1.4]),
            rotate=np.array([5.0, -8.0, 12.0]),
            translate=np.array([3.0, 1.0, -2.0]),
            segment_scale_compensate=True,
        )
        c = TransformData(
            name='c',
            node_type='joint',
            scale=np.array([0.9, 1.3, 1.1]),
            rotate=np.array([20.0, 15.0, -5.0]),
            joint_orient=np.array([7.0, -3.0, 4.0]),
            translate=np.array([1.0, -1.0, 2.0]),
            segment_scale_compensate=True,
        )
        h.append(gp)
        h.append(p)
        h.append(c)
        p.set_parent('gp', world_space=False)
        c.set_parent('p', world_space=False)

        # the parent world matrix must be non-orthonormal for this to bite
        W = c.get_parent_matrix()
        self.assertFalse(np.allclose(W[:3, :3] @ W[:3, :3].T, np.eye(3), atol=1e-3))

        target_world = c.world_matrix.copy()

        # perturb the child, then restore via the world_matrix setter
        c.translate    = np.array([0.0, 0.0, 0.0])
        c.rotate       = np.array([0.0, 0.0, 0.0])
        c.scale        = np.array([1.0, 1.0, 1.0])
        c.world_matrix = target_world

        self.assertTrue(np.allclose(c.world_matrix, target_world, atol=1e-6))

    def test_quaternion_setter_preserves_rotate_axis(self):
        # the orientation chain is RO * R * JO; setting a quaternion must
        # solve for R given the existing rotate_axis (RO), or RO is
        # double-applied and the round-trip orientation is wrong
        node = TransformData(
            name         = 'j',
            node_type    = 'joint',
            rotate_axis  = np.array([12.0, -7.0, 25.0]),
            joint_orient = np.array([5.0, 10.0, -15.0]),
            rotate       = np.array([3.0, 4.0, 5.0]),
        )
        h = HierarchyData()
        h.append(node)

        # target orientation as a quaternion (from a different rotate)
        node.rotate = np.array([40.0, -20.0, 33.0])
        target_q    = node.quaternion.copy()

        # change rotate, then drive it back via the quaternion setter
        node.rotate     = np.array([0.0, 0.0, 0.0])
        node.quaternion = target_q

        q = node.quaternion
        self.assertTrue(
            np.allclose(q, target_q, atol=1e-6)
            or np.allclose(q, -target_q, atol=1e-6)
        )
        # rotate_axis and joint_orient are untouched by the setter
        self.assertTrue(np.allclose(node.rotate_axis, [12.0, -7.0, 25.0]))
        self.assertTrue(np.allclose(node.joint_orient, [5.0, 10.0, -15.0]))

    def test_match_rotate_lands_on_the_target_orientation(self):
        # the world orientation is RO * R * JO * IS * P with the scale divided
        # out, so match_rotate solves R around the rotate_axis, joint_orient
        # and compensated parent scale the node already has. leaving RO out
        # missed by up to 1.0 per matrix element, leaving IS out by 0.5. a
        # non-uniform parent scale without compensation shears the child,
        # which no rotate can follow, so that case is left out
        target = TransformData('t', rotate=[33.0, -48.0, 121.0], translate=[4.0, 5.0, 6.0])
        want   = _rotation(target.world_matrix)
        cases = itertools.product(
            range(6),
            ([0.0, 0.0, 0.0], [25.0, -40.0, 60.0]),
            ((False, [1.0, 1.0, 1.0]), (True, [1.0, 1.0, 1.0]), (True, [2.0, 0.5, 1.5])),
        )
        for order, rotate_axis, (compensate, parent_scale) in cases:
            operands = {
                'node':       target,
                'euler':      np.degrees(matrix_to_euler(want, order))[0],
                'quaternion': matrix_to_quaternion(want)[0],
            }
            for kind, other in operands.items():
                with self.subTest(order=order, rotate_axis=rotate_axis,
                                  compensate=compensate, scale=parent_scale, operand=kind):
                    a = TransformData('a', node_type='joint', rotate=[10.0, 20.0, 30.0],
                                      uuid=generate_uuid(), scale=parent_scale,
                                      joint_orient=[5.0, -10.0, 15.0])
                    rig = HierarchyData([
                        a,
                        TransformData('b', node_type='joint', parent_uuid=a.uuid,
                                      translate=[1.0, 2.0, 3.0], rotate=[40.0, -15.0, 70.0],
                                      rotate_axis=rotate_axis, joint_orient=[-20.0, 35.0, 10.0],
                                      scale=[1.3, 0.7, 1.1], rotate_order=order,
                                      segment_scale_compensate=compensate),
                    ])
                    node = rig['b']
                    node.match_rotate(other)

                    got = _rotation(node.world_matrix)
                    self.assertTrue(np.allclose(got[:3, :3], want[:3, :3], atol=1e-9))
                    # rotate is all it writes
                    self.assertTrue(np.array_equal(node.translate, [1.0, 2.0, 3.0]))
                    self.assertTrue(np.array_equal(node.scale, [1.3, 0.7, 1.1]))
                    self.assertTrue(np.array_equal(node.rotate_axis, rotate_axis))

    def test_matrix_setters_keep_rotate_axis(self):
        # the local matrix is [S][RO][R][JO][IS][T]; writing matrix or
        # world_matrix solves S, R and T around the RO, JO and IS the node
        # already has, so what is written reads back and rotate_axis stays.
        # a mirrored child (a negative scale) reads back as well
        rng = np.random.default_rng(11)
        grid = itertools.product(
            range(6),
            (None, [12.0, -30.0, 45.0]),
            (None, [25.0, -10.0, 60.0]),
            (False, True),
            ([1.3, 0.8, 1.1], [-1.3, 0.8, 1.1]),
        )
        for order, joint_orient, rotate_axis, compensate, scale in grid:
            parent = TransformData(
                name      = 'p',
                node_type = 'joint',
                uuid      = generate_uuid(),
                scale     = [1.5, 0.7, 2.0],
                rotate    = [10.0, 20.0, -35.0],
                translate = [1.0, 2.0, 3.0],
            )
            child = TransformData(
                name='c',
                node_type='joint',
                parent_uuid=parent.uuid,
                rotate_order=order,
                joint_orient=joint_orient,
                rotate_axis=rotate_axis,
                segment_scale_compensate=compensate,
                scale=scale,
                rotate=rng.uniform(-170.0, 170.0, 3),
                translate=[0.5, -2.0, 1.0],
            )
            HierarchyData([parent, child])
            axis   = child.rotate_axis
            orient = child.joint_orient
            world  = child.world_matrix
            local  = child.matrix

            for setter, target in (('world_matrix', world), ('matrix', local)):
                with self.subTest(
                    setter       = setter,
                    order        = order,
                    joint_orient = joint_orient,
                    rotate_axis  = rotate_axis,
                    compensate   = compensate,
                    scale        = scale,
                ):
                    child.rotate    = rng.uniform(-170.0, 170.0, 3)
                    child.scale     = [2.0, 2.0, 2.0]
                    child.translate = [9.0, 9.0, 9.0]

                    setattr(child, setter, target)

                    self.assertTrue(np.allclose(getattr(child, setter), target, atol=1e-9))
                    self.assertTrue(np.allclose(child.world_matrix, world, atol=1e-9))
                    self.assertTrue(np.allclose(child.rotate_axis, axis))
                    self.assertTrue(np.allclose(child.joint_orient, orient))
                    self.assertEqual(child.segment_scale_compensate, compensate)

    def test_matrix_setters_read_back_mirrors_and_zero_scales(self):
        # a mirror is told by the sign of the determinant and goes on scale
        # z, the way Maya splits a matrix, even at 90 degrees where an euler
        # guess loses it. a zero scale leaves part of the rotation free: the
        # node's current rotation fills it, where NaN or a miss came back
        angles  = (-90.0, -45.0)
        mirrors = ([1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, -1.0])
        zeros = (
            [0.0, 0.8, 1.1],
            [1.3, 0.0, -1.1],
            [1.3, 0.8, 0.0],
            [0.0, 0.0, 1.1],
            [1.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        grid = itertools.product(
            range(6),
            (None, [10.0, 20.0, 30.0]),
            mirrors + zeros,
            itertools.product(angles, repeat=3),
        )
        misses = []
        for order, rotate_axis, scale, rotate in grid:
            target = TransformData(
                name         = 's',
                rotate_order = order,
                rotate_axis  = rotate_axis,
                scale        = scale,
                rotate       = rotate,
                translate    = [1.0, 2.0, 3.0],
            ).matrix
            node = TransformData(
                name         = 'd',
                rotate_order = order,
                rotate_axis  = rotate_axis,
                rotate       = [5.0, 6.0, 7.0],
            )
            before            = node.rotate_matrix
            node.world_matrix = target

            mirrored = np.prod(scale) < 0
            if not (
                np.allclose(node.world_matrix, target, atol=1e-9)
                and np.allclose(node.rotate_axis, rotate_axis or 0.0)
                and np.allclose(np.abs(node.scale), np.abs(scale))
                and (not mirrored or np.array_equal(np.sign(node.scale), [1.0, 1.0, -1.0]))
                and (np.any(scale) or np.allclose(node.rotate_matrix, before))
            ):
                misses.append((order, rotate_axis, scale, rotate))

        self.assertEqual([], misses[:5], f"{len(misses)} writes miss")

    def test_set_parent_keeps_rotate_axis(self):
        # set_parent goes through the world_matrix setter: the node keeps its
        # world matrix, and its rotate_axis with it
        h = HierarchyData([
            TransformData(
                name      = 'a',
                rotate    = [30.0, -45.0, 60.0],
                scale     = [2.0, 2.0, 2.0],
                translate = [1.0, 2.0, 3.0],
            ),
            TransformData(
                name         = 'b',
                node_type    = 'joint',
                rotate_axis  = [15.0, 5.0, -20.0],
                joint_orient = [0.0, 0.0, 30.0],
                rotate       = [10.0, 20.0, 30.0],
                translate    = [4.0, 0.0, 0.0],
            ),
        ])
        world = h['b'].world_matrix

        for parent in ('a', None):
            with self.subTest(parent=parent):
                h['b'].set_parent(parent)

                self.assertTrue(np.allclose(h['b'].world_matrix, world, atol=1e-9))
                self.assertTrue(np.allclose(h['b'].rotate_axis, [15.0, 5.0, -20.0]))
                self.assertTrue(np.allclose(h['b'].joint_orient, [0.0, 0.0, 30.0]))

    def test_set_rotate_to_joint_orient_preserves_orientation(self):
        # consolidating rotation into joint_orient must preserve the overall
        # orientation and zero BOTH rotate and rotate_axis (or RO is left
        # applied on top, double-counting it)
        node = TransformData(
            name         = 'j',
            node_type    = 'joint',
            rotate_axis  = np.array([8.0, -4.0, 11.0]),
            rotate       = np.array([20.0, 15.0, -10.0]),
            joint_orient = np.array([5.0, 5.0, 5.0]),
        )
        h = HierarchyData()
        h.append(node)
        before = node.quaternion.copy()

        node.set_rotate_to_joint_orient()

        self.assertTrue(np.allclose(node.rotate, 0.0))
        self.assertTrue(np.allclose(node.rotate_axis, 0.0))
        after = node.quaternion
        self.assertTrue(
            np.allclose(after, before, atol=1e-6)
            or np.allclose(after, -before, atol=1e-6)
        )

    def test_set_joint_orient_to_rotate_preserves_orientation(self):
        # consolidating rotation into rotate must preserve the overall
        # orientation and zero BOTH joint_orient and rotate_axis
        node = TransformData(
            name         = 'j',
            node_type    = 'joint',
            rotate_axis  = np.array([8.0, -4.0, 11.0]),
            rotate       = np.array([20.0, 15.0, -10.0]),
            joint_orient = np.array([5.0, 5.0, 5.0]),
        )
        h = HierarchyData()
        h.append(node)
        before = node.quaternion.copy()

        node.set_joint_orient_to_rotate()

        self.assertTrue(np.allclose(node.joint_orient, 0.0))
        self.assertTrue(np.allclose(node.rotate_axis, 0.0))
        after = node.quaternion
        self.assertTrue(
            np.allclose(after, before, atol=1e-6)
            or np.allclose(after, -before, atol=1e-6)
        )

    # --- vectorized TransformList views --- #
    def _chain(self, name='j', count=3, offset=10.0):
        # a straight chain, each joint `offset` along +X of its parent
        hierarchy = HierarchyData()
        for i in range(count):
            node = TransformData(name=f'{name}{i}', node_type='joint')
            if i:
                node.parent_uuid = hierarchy[i - 1].uuid
            hierarchy.append(node)
            node.translate = np.array([offset if i else 0.0, 0.0, 0.0])
        return hierarchy

    def test_view_covers_every_settable_transform_property(self):
        # a TransformList is the vectorized view of a hierarchy, so every
        # channel settable on a node must be settable on the view. parent_node
        # is no channel: parent_uuid's deprecated name, a field a view sets no
        # more than it sets parent_uuid
        settable = {
            name
            for name, value in vars(TransformData).items()
            if isinstance(value, property) and value.fset is not None
        } - {"parent_node"}
        for name in settable:
            prop = getattr(TransformList, name, None)
            self.assertTrue(
                isinstance(prop, property) and prop.fset is not None,
                f'TransformList.{name} is not settable',
            )

    def test_view_component_setters_broadcast_and_index(self):
        skeleton = self._chain()
        view     = skeleton[1:]

        view.translate_y = 4.0
        self.assertTrue(np.allclose(view.translate_y, [4.0, 4.0]))
        self.assertTrue(np.allclose(skeleton[0].translate_y, 0.0))

        view.translate_y = [1.0, 2.0]
        self.assertTrue(np.allclose(view.translate_y, [1.0, 2.0]))
        self.assertTrue(np.allclose(skeleton.translate_y, [0.0, 1.0, 2.0]))

        view.rotate_z = [30.0, -30.0]
        self.assertTrue(np.allclose(view.rotate[:, 2], [30.0, -30.0]))

    def test_view_write_invalidates_nodes_outside_the_view(self):
        # nodes hold a back-pointer to their real hierarchy, so a write
        # through a view must invalidate the whole downstream branch
        skeleton = self._chain()
        _        = skeleton.world_matrix

        skeleton[1:2].translate = np.array([10.0, 5.0, 0.0])

        self.assertIsNone(skeleton[2]._world_matrix)
        self.assertTrue(np.allclose(skeleton[2].world_matrix[3, :3], [20.0, 5.0, 0.0]))

    def test_view_world_matrix_setter_is_order_independent(self):
        # solving a local matrix reads the parent's world matrix, so an
        # unsorted view must still land every node on its target
        target = self._chain().world_matrix.copy()
        target[:, 3, 1] += 4.0

        for order in ([0, 1, 2], [2, 1, 0]):
            skeleton          = self._chain()
            view              = skeleton[order]
            view.world_matrix = target[order]
            self.assertTrue(
                np.allclose(skeleton.world_matrix, target), f'failed for {order}'
            )

    def test_view_matrix_setter_is_order_independent(self):
        source        = self._chain()
        source.rotate = np.array([[10.0, 20.0, 30.0]] * 3)
        target        = source.matrix.copy()

        for order in ([0, 1, 2], [2, 1, 0]):
            skeleton = self._chain()
            for node in skeleton:
                node.segment_scale_compensate = True
            skeleton[order].matrix = target[order]
            self.assertTrue(
                np.allclose(skeleton.matrix, target), f'failed for {order}'
            )

    def test_view_quaternion_and_rotate_order_setters(self):
        skeleton = self._chain()
        view     = skeleton[1:]

        view.rotate_order = 2
        self.assertTrue(np.allclose(view.rotate_order, [2, 2]))
        view.rotate_order = [1, 3]
        self.assertTrue(np.allclose(view.rotate_order, [1, 3]))
        self.assertEqual(len(view.rotate_axes), 2)

        Q = matrix_to_quaternion(
            euler_to_matrix(np.radians([[10.0, 0.0, 0.0], [0.0, 25.0, 0.0]]), 0)
        )
        view.quaternion = Q
        self.assertTrue(
            np.allclose(np.abs(np.einsum('ij,ij->i', view.quaternion, Q)), 1.0)
        )
        self.assertTrue(np.allclose(view.rotate_x, [10.0, 0.0]))
        self.assertTrue(np.allclose(view.rotate_y, [0.0, 25.0]))

    def test_view_search_methods_chain(self):
        skeleton = HierarchyData()
        for name in ('root', 'L_arm', 'L_pinky_01', 'L_pinky_02', 'R_pinky_01'):
            node = TransformData(name=name, node_type='joint')
            if name != 'root':
                node.parent_uuid = skeleton[0].uuid
            skeleton.append(node)

        left = skeleton.match('L_*')
        self.assertTrue(isinstance(left, TransformList))
        self.assertEqual(left.name, ['L_arm', 'L_pinky_01', 'L_pinky_02'])

        pinky = left.match('*pinky*')
        self.assertEqual(pinky.name, ['L_pinky_01', 'L_pinky_02'])

    def test_view_hierarchy_queries(self):
        skeleton = self._chain(count=4)
        view     = skeleton[0:2]

        self.assertEqual(view.get_children().name, ['j1', 'j2'])
        self.assertEqual(view.get_branch().name, ['j0', 'j1', 'j2', 'j3'])
        self.assertTrue(np.allclose(view.indices, [0, 1]))
        self.assertEqual(view.get_parent_matrix().shape, (2, 4, 4))
        self.assertEqual(len(view.to_attributes()), 2)

    def test_view_match_methods(self):
        skeleton        = self._chain()
        other           = self._chain(name='k', offset=7.0)
        other.translate = other.translate + np.array([0.0, 3.0, 0.0])

        skeleton[1:].match_translate(other[1:])
        self.assertTrue(
            np.allclose(skeleton[1:].world_matrix[:, 3, :], other[1:].world_matrix[:, 3, :])
        )

        skeleton = self._chain()
        skeleton.match_translate([0.0, 1.0, 0.0], x=False, z=False)
        self.assertTrue(np.allclose(skeleton.world_matrix[:, 3, 1], 1.0))

        with self.assertRaises(ValueError):
            skeleton[1:].match_translate(other)

    # --- swapaxes --- #
    def _permutation(self, axis0, axis1, negate):
        P               = np.eye(4)
        other           = 3 - axis0 - axis1
        P[axis0, axis0] = P[axis1, axis1] = 0.0
        P[axis0, axis1] = -1.0 if negate else 1.0
        P[axis1, axis0] = 1.0
        P[other, other] = 1.0 if negate else -1.0
        return P

    def test_swapaxes_permutes_the_world_frame(self):
        # the resulting world matrix must be exactly the permutation applied
        # to the old one, for every axis pair and both sign conventions
        for axis0 in range(3):
            for axis1 in range(3):
                if axis0 == axis1:
                    continue
                for negate in (False, True):
                    skeleton        = self._chain()
                    skeleton.rotate = np.array([[15.0, -25.0, 40.0]] * 3)
                    node            = skeleton[1]
                    expected = np.dot(
                        self._permutation(axis0, axis1, negate), node.world_matrix
                    )

                    node.swapaxes(axis0, axis1, negate=negate)

                    self.assertTrue(
                        np.allclose(node.world_matrix, expected, atol=1e-9),
                        f'failed for {axis0}, {axis1}, negate={negate}',
                    )
                    self.assertAlmostEqual(
                        np.linalg.det(node.world_matrix[:3, :3]), 1.0, places=9
                    )

    def test_swapaxes_at_identity(self):
        # X takes Y's place, Y takes X's, and the sign lands where documented
        for negate, rows in (
            (False, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
            (True, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ):
            skeleton = self._chain(count=1)
            skeleton[0].swapaxes(0, 1, negate=negate)
            self.assertTrue(
                np.allclose(skeleton[0].world_matrix[:3, :3], rows),
                f'failed for negate={negate}',
            )

    def test_swapaxes_leaves_children_and_position_alone(self):
        skeleton = self._chain(count=4)
        before   = [n.world_matrix.copy() for n in skeleton]

        skeleton[1].swapaxes(0, 2)

        self.assertTrue(np.allclose(skeleton[0].world_matrix, before[0]))
        self.assertTrue(np.allclose(skeleton[1].world_matrix[3, :3], before[1][3, :3]))
        for i in (2, 3):
            self.assertTrue(np.allclose(skeleton[i].world_matrix, before[i]))

    def test_swapaxes_keeps_the_swap_in_joint_orient(self):
        # an unposed joint must come out of a swap still unposed
        skeleton = self._chain()
        skeleton.set_rotate_to_joint_orient()

        skeleton[1].swapaxes(0, 1)

        self.assertTrue(np.allclose(skeleton[1].rotate, 0.0, atol=1e-9))
        self.assertTrue(np.allclose(skeleton[1].rotate_axis, 0.0, atol=1e-9))
        self.assertFalse(np.allclose(skeleton[1].joint_orient, 0.0))

    def test_swapaxes_inverses(self):
        # the plain swap undoes itself; the negated swap is undone by the
        # same call with the axes given the other way round
        skeleton        = self._chain()
        skeleton.rotate = np.array([[10.0, 20.0, 30.0]] * 3)

        start = skeleton[1].world_matrix.copy()
        skeleton[1].swapaxes(0, 1)
        skeleton[1].swapaxes(0, 1)
        self.assertTrue(np.allclose(skeleton[1].world_matrix, start, atol=1e-9))

        skeleton[1].swapaxes(0, 1, negate=True)
        skeleton[1].swapaxes(1, 0, negate=True)
        self.assertTrue(np.allclose(skeleton[1].world_matrix, start, atol=1e-9))

    def test_swapaxes_permutes_scale(self):
        skeleton          = self._chain()
        skeleton[1].scale = np.array([2.0, 3.0, 5.0])
        for node in skeleton:
            node.segment_scale_compensate = True
        before   = [n.world_matrix.copy() for n in skeleton]
        expected = np.dot(self._permutation(0, 1, False), skeleton[1].world_matrix)

        skeleton[1].swapaxes(0, 1)

        self.assertTrue(np.allclose(skeleton[1].scale, [3.0, 2.0, 5.0]))
        self.assertTrue(np.allclose(skeleton[1].world_matrix, expected, atol=1e-9))
        self.assertTrue(np.allclose(skeleton[2].world_matrix, before[2], atol=1e-9))

    def test_swapaxes_rejects_bad_axes(self):
        node = self._chain(count=1)[0]
        with self.assertRaises(ValueError):
            node.swapaxes(0, 0)
        with self.assertRaises(ValueError):
            node.swapaxes(0, 3)

    def test_swapaxes_vectorized_is_order_independent(self):
        results = []
        for order in ([0, 1, 2], [2, 1, 0]):
            skeleton        = self._chain()
            skeleton.rotate = np.array([[10.0, 20.0, 30.0]] * 3)
            skeleton[order].swapaxes(0, 1, negate=True)
            results.append(skeleton.world_matrix.copy())

        self.assertTrue(np.allclose(results[0], results[1], atol=1e-9))

        # and it matches doing it one node at a time
        skeleton        = self._chain()
        skeleton.rotate = np.array([[10.0, 20.0, 30.0]] * 3)
        for node in skeleton[:]:
            node.swapaxes(0, 1, negate=True)
        self.assertTrue(np.allclose(skeleton.world_matrix, results[0], atol=1e-9))

    # --- pose algebra over selections --- #
    def test_pose_algebra_works_on_a_view(self):
        # a delta may be taken over any selection, not just a whole rig
        a = HierarchyData.load(self.skeleton_file)
        b = HierarchyData.load(self.skeleton_file)
        for node in b:
            node.rotate = node.rotate + np.array([7.0, -4.0, 3.0])

        view_a, view_b = a[1:], b[1:]
        self.assertTrue(isinstance(view_a, TransformList))
        self.assertFalse(isinstance(view_a, HierarchyData))

        delta = view_a.get_delta(view_b)

        # a delta owns its nodes whatever the receiver was
        self.assertTrue(isinstance(delta, HierarchyData))
        self.assertEqual(len(delta), len(a) - 1)
        for node in delta:
            self.assertTrue(node._hierarchy is delta)

        recovered = view_b.add_delta(delta)
        for node in recovered:
            self.assertTrue(allclose(node.quaternion, a[node.name].quaternion))
            self.assertTrue(
                allclose(node.world_matrix[3, :3], a[node.name].world_matrix[3, :3])
            )

    def test_unresolved_parent_reads_as_a_root(self):
        # copying a branch out of a rig leaves parent_uuid naming a joint that
        # is not there: the copy joins its hierarchy as a root. One written
        # back since is not an error either: that node reads as a root
        b      = HierarchyData.load(self.skeleton_file)
        nodes  = [node.copy() for node in b[0].get_branch()[1:]]
        lost   = nodes[0].parent_uuid
        branch = HierarchyData(nodes)
        self.assertIsNone(branch[0].parent_uuid)
        branch[0].parent_uuid = lost

        # the unresolved reference is still visible through get_parent
        self.assertTrue(isinstance(branch[0].get_parent(), str))
        self.assertTrue(np.allclose(branch[0].get_parent_matrix(), np.eye(4)))
        self.assertTrue(np.allclose(branch[0].parent_scale_inverse, 1.0))
        self.assertTrue(np.allclose(branch[0].world_matrix, branch[0].matrix))

    def test_partial_hierarchy_does_not_need_its_parent_cleared(self):
        # a partial rig used to have to have its dangling parent cleared by
        # hand before it could take part in a delta
        b      = HierarchyData.load(self.skeleton_file)
        branch = HierarchyData([node.copy() for node in b[0].get_branch()[1:]])
        a      = HierarchyData.load(self.skeleton_file)

        # as the operand: reading it must not raise
        delta = a.get_delta(branch, by='name')
        self.assertEqual(len(delta), len(a))

        # as the receiver: the delta owns its nodes, so the reference that
        # pointed outside the selection is dropped
        delta = branch.get_delta(a, by='name')
        self.assertTrue(isinstance(delta, HierarchyData))
        self.assertIsNone(delta[0].get_parent())
        for node in delta:
            self.assertFalse(isinstance(node.get_parent(), str))

    def test_index_pairing_uses_subset_positions(self):
        # `node.index` is the position in the OWNING rig; on a selection that
        # points at the wrong node, and for a tail slice at the node itself
        a = self._chain(count=4)
        b = self._chain(count=4, name='k', offset=7.0)

        _, _, parents = TransformList._pair_by_index(a[1:], b[1:])

        # within the 3 node pairing: first is a root, then a chain
        self.assertTrue(np.array_equal(parents, [-1, 0, 1]))

        # a full rig is unaffected -- subset positions are hierarchy positions
        _, _, parents = TransformList._pair_by_index(a, b)
        self.assertTrue(np.array_equal(parents, [-1, 0, 1, 2]))

    def test_index_pairing_over_views_round_trips(self):
        a         = self._chain(count=4)
        b         = self._chain(count=4, name='k', offset=7.0)
        b.rotate  = np.array([[12.0, -8.0, 20.0]] * 4)

        delta     = a[1:].get_delta(b[1:], by='index')
        recovered = b[1:].add_delta(delta, by='index')

        self.assertEqual(recovered.name, ['j1', 'j2', 'j3'])
        for i, node in enumerate(recovered):
            self.assertTrue(allclose(node.world_matrix, a[i + 1].world_matrix))

    def test_pose_algebra_defaults_to_index_pairing(self):
        # the common case is the same skeleton twice, so names are ignored
        # unless asked for; by="name" is the opt-in that allows partial deltas
        a = self._chain(count=3)
        b = self._chain(count=3, name='k', offset=11.0)

        self.assertTrue(allclose((a - b).world_matrix, a.get_delta(b).world_matrix))
        self.assertTrue(
            allclose(a.get_delta(b).world_matrix, a.get_delta(b, by='index').world_matrix)
        )

        with self.assertRaises(ValueError):
            a.get_delta(b, by='name')

    def test_retarget_without_translation_keeping_root_motion(self):
        # the documented recipe: orientation only everywhere, then let the
        # root carry the animation's world position
        source = self._chain(count=4, offset=10.0)
        other  = self._chain(count=4, offset=11.0, name='k')
        posed  = self._chain(count=4, offset=11.0, name='k')
        posed.rotate = np.array(
            [[20.0, 35.0, -15.0], [10.0, -50.0, 25.0], [-30.0, 12.0, 40.0], [0.0] * 3]
        )
        posed[0].translate = posed[0].translate + np.array([0.0, 5.0, 0.0])

        delta  = source.get_delta(other, translate=False)
        result = posed.add_delta(delta, translate=False)
        result[0].match_translate(posed[0])

        self.assertTrue(np.allclose(result[0].world_matrix[3, :3], [0.0, 5.0, 0.0]))
        self.assertTrue(np.allclose(self._bone_lengths(result), 10.0))

    # --- orientation-only deltas --- #
    def _bone_lengths(self, hierarchy):
        positions = np.array([n.world_matrix[3, :3] for n in hierarchy])
        return np.linalg.norm(np.diff(positions, axis=0), axis=1)

    def test_orientation_only_delta_round_trips(self):
        a = HierarchyData.load(self.skeleton_file)
        b = HierarchyData.load(self.skeleton_file)
        for node in b:
            node.rotate = node.rotate + np.array([9.0, -6.0, 4.0])

        recovered = b.add_delta(a.get_delta(b, translate=False), translate=False)

        for node in recovered:
            self.assertTrue(allclose(node.quaternion, a[node.name].quaternion))
            self.assertTrue(
                allclose(node.world_matrix[3, :3], a[node.name].world_matrix[3, :3])
            )

    def test_orientation_only_delta_keeps_proportions_when_posed(self):
        # rigs of different bone lengths: the result must come out with the
        # source's proportions no matter how the other rig is posed
        source = self._chain(count=4, offset=10.0)
        other  = self._chain(count=4, offset=11.0, name='k')

        posed  = self._chain(count=4, offset=11.0, name='k')
        posed.rotate = np.array(
            [[20.0, 35.0, -15.0], [10.0, -50.0, 25.0], [-30.0, 12.0, 40.0], [0.0] * 3]
        )

        delta  = source.get_delta(other, by='index', translate=False)
        result = posed.add_delta(delta, by='index', translate=False)

        self.assertTrue(np.allclose(self._bone_lengths(source), 10.0))
        self.assertTrue(np.allclose(self._bone_lengths(posed), 11.0))
        self.assertTrue(
            np.allclose(self._bone_lengths(result), 10.0),
            f'expected source proportions, got {self._bone_lengths(result)}',
        )

        # orientation still comes from the posed rig, carried by the frame
        # offset -- that part is what the delta is for
        F = quaternion_to_matrix(np.array([n.quaternion for n in delta]))
        expected = matrix_multiply(
            F, _rotation(np.array([n.world_matrix for n in posed]))
        )
        for i, node in enumerate(result):
            self.assertTrue(allclose(node.world_matrix[:3, :3], expected[i][:3, :3]))

    def test_orientation_only_delta_does_not_follow_translation(self):
        # documented limit: a joint that moves rather than rotates stays put
        source             = self._chain(count=3)
        other              = self._chain(count=3, name='k')
        moved              = self._chain(count=3, name='k')
        moved[0].translate = moved[0].translate + np.array([0.0, 5.0, 0.0])

        delta  = source.get_delta(other, by='index', translate=False)
        result = moved.add_delta(delta, by='index', translate=False)
        self.assertTrue(np.allclose(result[0].world_matrix[3, :3], [0.0, 0.0, 0.0]))

        # with translate on, the move comes across
        delta  = source.get_delta(other, by='index')
        result = moved.add_delta(delta, by='index')
        self.assertTrue(np.allclose(result[0].world_matrix[3, :3], [0.0, 5.0, 0.0]))

    def test_delta_translate_flag_defaults_to_previous_behaviour(self):
        a = HierarchyData.load(self.skeleton_file)
        b = HierarchyData.load(self.skeleton_file)
        for node in b:
            node.rotate = node.rotate + np.array([5.0, 5.0, 5.0])

        self.assertTrue(
            allclose(
                a.get_delta(b, translate=True).translate,
                (a - b).translate,
            )
        )
        self.assertTrue(
            allclose(
                b.add_delta(a.get_delta(b), translate=True).world_matrix,
                ((a - b) + b).world_matrix,
            )
        )


class TestFbxEnum(unittest.TestCase):
    """the sdk moved its enums into scope classes at some point"""

    class Scoped:
        class EType:
            eSkeleton = 3

    class Flat:
        eSkeleton = 3

    def test_reads_a_scoped_enum(self):
        self.assertEqual(_fbx_enum(self.Scoped, "EType", "eSkeleton"), 3)

    def test_falls_back_to_the_owner(self):
        self.assertEqual(_fbx_enum(self.Flat, "EType", "eSkeleton"), 3)

    def test_raises_when_neither_spelling_has_it(self):
        with self.assertRaises(AttributeError):
            _fbx_enum(self.Flat, "EType", "eNotAThing")


@unittest.skipIf(FBX is None, "fbx sdk is not installed")
class TestLoadFbx(unittest.TestCase):
    """load_fbx could not run at all against a scoped enum binding"""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def chain(self, count=4):
        rig = HierarchyData(
            [
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    translate    = [0.0, 10.0, 0.0],
                    rotate       = [i * 7.0, i * 5.0, i * 3.0],
                    joint_orient = [i * 2.0, 0.0, float(i)],
                )
                for i in range(count)
            ]
        )
        for i in range(1, count):
            rig[f"j{i}"].set_parent(f"j{i - 1}", world_space=False)
        return rig

    def saved(self, rig, name="rig.fbx"):
        path = os.path.join(self.tmp.name, name)
        rig.save_fbx(path)
        return path

    def round_trip(self, rig, name="rig.fbx"):
        return HierarchyData.load_fbx(self.saved(rig, name))

    def test_a_joint_chain_round_trips(self):
        rig = self.chain()

        loaded = self.round_trip(rig)

        self.assertEqual(list(loaded.name), list(rig.name))
        self.assertEqual([node.node_type for node in loaded], ["joint"] * 4)
        self.assertTrue(
            np.allclose(
                [rig[name].world_matrix for name in rig.name],
                [loaded[name].world_matrix for name in rig.name],
                atol=1e-9,
            )
        )

    def test_parenting_survives(self):
        loaded = self.round_trip(self.chain())

        parents = [
            node.get_parent().name if node.get_parent() is not None else None
            for node in loaded
        ]

        self.assertEqual(parents, [None, "j0", "j1", "j2"])

    def test_every_rotate_order_survives(self):
        """the sdk hands the order back as an enum, which is not a dict key"""
        for order in range(6):
            rig = HierarchyData(
                [
                    TransformData(
                        name         = "j",
                        node_type    = "joint",
                        rotate       = [11.0, 22.0, 33.0],
                        rotate_order = order,
                    )
                ]
            )

            loaded = self.round_trip(rig, f"order{order}.fbx")

            self.assertEqual(loaded["j"].rotate_order, order)
            self.assertTrue(
                np.allclose(
                    rig["j"].world_matrix, loaded["j"].world_matrix, atol=1e-9
                )
            )

    def test_scale_factor_reaches_translation(self):
        path   = self.saved(self.chain(), "scaled.fbx")

        loaded = HierarchyData.load_fbx(path, scale_factor=2.0)

        self.assertTrue(np.allclose(loaded["j1"].translate, [0.0, 20.0, 0.0]))


@unittest.skipIf(FBX is None, "fbx sdk is not installed")
class TestLoadFbxDroppedNodes(unittest.TestCase):
    """the reader keeps a subset of attribute types, and used to sever the
    chain at anything it skipped
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def locator_rig(self):
        rig = HierarchyData(
            [
                TransformData(name="root", node_type="joint"),
                TransformData(name="spine", node_type="joint",
                              translate=[0.0, 10.0, 0.0]),
                TransformData(name="loc", node_type="locator",
                              translate=[0.0, 10.0, 0.0], rotate=[45.0, 0.0, 0.0]),
                TransformData(name="tip", node_type="joint",
                              translate=[0.0, 10.0, 0.0]),
                TransformData(name="tip_end", node_type="joint",
                              translate=[0.0, 5.0, 0.0]),
            ]
        )
        for child, parent in (
            ("spine", "root"),
            ("loc", "spine"),
            ("tip", "loc"),
            ("tip_end", "tip"),
        ):
            rig[child].set_parent(parent, world_space=False)
        return rig

    def three_deep(self, middle):
        """A -> B -> C, each 100 up in y, with B carrying the attribute given"""
        manager = FBX.FbxManager.Create()
        scene   = FBX.FbxScene.Create(manager, "scene")

        nodes   = []
        parent  = scene.GetRootNode()
        for name in ("A", "B", "C"):
            node = FBX.FbxNode.Create(scene, name)
            node.LclTranslation.Set(FBX.FbxDouble3(0.0, 100.0, 0.0))
            parent.AddChild(node)
            parent = node
            nodes.append(node)

        nodes[0].SetNodeAttribute(FBX.FbxSkeleton.Create(scene, "a"))
        nodes[2].SetNodeAttribute(FBX.FbxSkeleton.Create(scene, "c"))
        if middle is not None:
            nodes[1].SetNodeAttribute(middle(scene, "b"))

        # the attribute class names the file; str(middle) would put "<" and
        # ">" in it, which Windows rejects
        label    = getattr(getattr(middle, "__self__", None), "__name__", "none")
        path     = os.path.join(self.tmp.name, f"mid_{label}.fbx")
        exporter = FBX.FbxExporter.Create(manager, "")
        self.assertTrue(exporter.Initialize(path, -1, manager.GetIOSettings()))
        self.assertTrue(exporter.Export(scene))
        exporter.Destroy()
        manager.Destroy()
        return path

    def load(self, rig, name):
        path = os.path.join(self.tmp.name, name)
        rig.save_fbx(path)
        return HierarchyData.load_fbx(path)

    def test_a_locator_round_trips(self):
        """save_fbx writes a locator as a null drawn as a cross, and the reader
        brings it back as a locator, not a transform"""
        rig = self.locator_rig()

        loaded = self.load(rig, "loc.fbx")

        self.assertEqual(list(loaded.name), list(rig.name))
        self.assertEqual(loaded["loc"].node_type, "locator")
        self.assertTrue(
            np.allclose(
                [rig[name].world_matrix for name in rig.name],
                [loaded[name].world_matrix for name in rig.name],
                atol=1e-9,
            )
        )

    def test_a_marker_or_a_crossed_null_reads_as_a_locator(self):
        """a bare null is a group, one drawn as a cross is a locator, a marker too"""

        def null(look):
            def create(scene, name):
                attr = FBX.FbxNull.Create(scene, name)
                attr.Look.Set(look.value)
                return attr
            return create

        cases = (
            (FBX.FbxMarker.Create, "locator"),
            (null(FBX.FbxNull.ELook.eCross), "locator"),
            (null(FBX.FbxNull.ELook.eNone), "transform"),
        )
        for middle, node_type in cases:
            loaded = HierarchyData.load_fbx(self.three_deep(middle))
            self.assertEqual(loaded["B"].node_type, node_type)

    def test_a_locator_keeps_its_children(self):
        loaded = self.load(self.locator_rig(), "loc_children.fbx")

        self.assertEqual(loaded["tip"].get_parent().name, "loc")
        self.assertEqual(loaded["tip_end"].get_parent().name, "tip")

    def test_a_dropped_node_hands_its_children_to_the_nearest_kept_ancestor(self):
        """a camera in the middle used to drop its child to the world"""
        for middle in (FBX.FbxCamera.Create, FBX.FbxLight.Create, None):
            loaded = HierarchyData.load_fbx(self.three_deep(middle))

            self.assertEqual(list(loaded.name), ["A", "C"])
            self.assertEqual(loaded["C"].get_parent().name, "A")

            # A's own 100 still applies; only B's is lost with B
            self.assertAlmostEqual(loaded["C"].world_matrix[3, 1], 200.0, places=6)

    def test_the_scene_root_still_produces_world_parented_roots(self):
        """the fbx root carries no attribute, so it is a dropped node too"""
        loaded = HierarchyData.load_fbx(self.three_deep(FBX.FbxMarker.Create))

        self.assertIsNone(loaded["A"].get_parent())
        self.assertEqual(loaded["B"].get_parent().name, "A")
        self.assertAlmostEqual(loaded["C"].world_matrix[3, 1], 300.0, places=6)


@unittest.skipIf(FBX is None, "fbx sdk is not installed")
class TestLoadFbxTransformReads(unittest.TestCase):
    """three static properties the reader used to drop on the floor

    Ground truth throughout is fbx's own EvaluateLocalTransform, so these
    say "the library composes the same matrix fbx does", not "the library
    agrees with itself".
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pivot = FBX.FbxNode.EPivotSet.eSourcePivot

    def authored(self, name, **kwargs):
        """write a parent/child joint pair straight through the sdk

        Returns the path and fbx's own local matrix for the child.
        """
        manager = FBX.FbxManager.Create()
        scene   = FBX.FbxScene.Create(manager, "scene")

        parent  = FBX.FbxNode.Create(scene, "parent")
        parent.SetNodeAttribute(FBX.FbxSkeleton.Create(scene, "p"))
        if kwargs.get("parent_scale"):
            parent.LclScaling.Set(FBX.FbxDouble3(*kwargs["parent_scale"]))
        scene.GetRootNode().AddChild(parent)

        child = FBX.FbxNode.Create(scene, "child")
        child.SetNodeAttribute(FBX.FbxSkeleton.Create(scene, "c"))
        child.LclTranslation.Set(FBX.FbxDouble3(0.0, 10.0, 0.0))
        child.LclRotation.Set(FBX.FbxDouble3(*kwargs.get("rotate", (11.0, 22.0, 33.0))))
        child.SetRotationActive(kwargs.get("rotation_active", True))
        if kwargs.get("pre"):
            child.SetPreRotation(self.pivot, FBX.FbxVector4(*kwargs["pre"], 1.0))
        if kwargs.get("post"):
            child.SetPostRotation(self.pivot, FBX.FbxVector4(*kwargs["post"], 1.0))
        if kwargs.get("inherit") is not None:
            child.SetTransformationInheritType(kwargs["inherit"])
        parent.AddChild(child)

        path     = os.path.join(self.tmp.name, name)
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()

        local = child.EvaluateLocalTransform()
        truth = np.array(
            [[local.Get(row, col) for col in range(4)] for row in range(4)]
        )
        manager.Destroy()
        return path, truth

    def test_rotate_axis_comes_from_the_inverse_of_post_rotation(self):
        """fbx composes post rotation inverted, so reading it straight is wrong"""
        path, truth = self.authored("post.fbx", post=(5.0, -12.0, 7.0))

        rig = HierarchyData.load_fbx(path)

        self.assertTrue(np.allclose(rig["child"].matrix, truth, atol=1e-9))

        # the two plausible wrong readings both miss, by a lot
        for wrong in ([0.0, 0.0, 0.0], [-5.0, 12.0, -7.0]):
            probe                      = HierarchyData.load_fbx(path)
            probe["child"].rotate_axis = wrong
            self.assertGreater(np.abs(probe["child"].matrix - truth).max(), 1e-3)

    def test_pre_and_post_are_ignored_when_rotation_is_not_active(self):
        path, truth = self.authored(
            "inactive.fbx",
            pre             = (3.0, 4.0, 5.0),
            post            = (5.0, -12.0, 7.0),
            rotation_active = False,
        )

        rig = HierarchyData.load_fbx(path)

        self.assertTrue(np.allclose(rig["child"].joint_orient, 0.0))
        self.assertTrue(np.allclose(rig["child"].rotate_axis, 0.0))
        self.assertTrue(np.allclose(rig["child"].matrix, truth, atol=1e-9))

    def test_segment_scale_compensate_follows_the_inherit_type(self):
        inherit = FBX.FbxTransform.EInheritType
        for kind, expected in (
            (inherit.eInheritRrs, True),
            (inherit.eInheritRSrs, False),
            (inherit.eInheritRrSs, False),
        ):
            path, _ = self.authored(
                f"ssc_{kind}.fbx", inherit=kind, parent_scale=(2.0, 3.0, 4.0)
            )

            rig = HierarchyData.load_fbx(path)

            self.assertEqual(rig["child"].segment_scale_compensate, expected)

    def test_eInheritRrSs_is_exact_when_the_parent_scale_is_uniform(self):
        """the library has two states for three fbx ones, so pin the gap

        eInheritRrSs is fbx's default and maps to False. That is only an
        approximation, and this is where it stops being one.
        """
        inherit = FBX.FbxTransform.EInheritType.eInheritRrSs
        path, _ = self.authored(
            "rrss_uniform.fbx", inherit=inherit, parent_scale=(3.0, 3.0, 3.0)
        )

        rig      = HierarchyData.load_fbx(path)
        manager  = FBX.FbxManager.Create()
        scene    = FBX.FbxScene.Create(manager, "check")
        importer = FBX.FbxImporter.Create(manager, "")
        importer.Initialize(path, -1, manager.GetIOSettings())
        importer.Import(scene)
        importer.Destroy()
        node  = scene.GetRootNode().GetChild(0).GetChild(0)
        world = node.EvaluateGlobalTransform()
        truth = np.array(
            [[world.Get(row, col) for col in range(4)] for row in range(4)]
        )
        manager.Destroy()

        self.assertTrue(np.allclose(rig["child"].world_matrix, truth, atol=1e-9))

    def test_rotate_axis_round_trips_through_save_fbx(self):
        """the writer stored it un-inverted, so a fixed reader would flip it"""
        rig = HierarchyData(
            [
                TransformData(
                    name         = "j",
                    node_type    = "joint",
                    rotate       = [11.0, 22.0, 33.0],
                    rotate_axis  = [5.0, -12.0, 7.0],
                    joint_orient = [3.0, 4.0, 5.0],
                )
            ]
        )
        path = os.path.join(self.tmp.name, "round.fbx")
        rig.save_fbx(path)

        loaded = HierarchyData.load_fbx(path)

        self.assertTrue(np.allclose(loaded["j"].rotate_axis, [5.0, -12.0, 7.0]))
        self.assertTrue(
            np.allclose(loaded["j"].world_matrix, rig["j"].world_matrix, atol=1e-9)
        )


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestLoadGlbPoseFrame(unittest.TestCase):
    """the posed branch of load_glb has to agree with the rest branch"""

    REST  = [0.5, 1.25, -0.75]
    POSED = [1.0, 2.0, 3.0]

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.quat = euler_to_quaternion(np.radians([[30.0, 45.0, 60.0]]), 0)[0]
        self.path = os.path.join(self.tmp.name, "posed.glb")
        self.build(self.path, self.quat)

    def build(self, path, quat):
        """one joint carrying a single translation and rotation key"""
        times        = np.array([0.0],        dtype=np.float32)
        translation  = np.array([self.POSED], dtype=np.float32)
        rotation     = np.array([list(quat)], dtype=np.float32)
        blob         = times.tobytes() + translation.tobytes() + rotation.tobytes()

        gltf         = GLTF2()
        gltf.scene   = 0
        gltf.scenes  = [Scene(nodes=[0])]
        gltf.nodes   = [Node(name="joint1", translation=self.REST)]
        gltf.buffers = [Buffer(byteLength=len(blob))]
        gltf.bufferViews = [
            BufferView(buffer=0, byteOffset=0, byteLength=4),
            BufferView(buffer=0, byteOffset=4, byteLength=12),
            BufferView(buffer=0, byteOffset=16, byteLength=16),
        ]
        gltf.accessors = [
            Accessor(bufferView=0, componentType=FLOAT, count=1, type="SCALAR",
                     min=[0.0], max=[0.0]),
            Accessor(bufferView=1, componentType=FLOAT, count=1, type="VEC3"),
            Accessor(bufferView=2, componentType=FLOAT, count=1, type="VEC4"),
        ]
        gltf.animations = [
            Animation(
                samplers=[
                    AnimationSampler(input=0, output=1, interpolation="LINEAR"),
                    AnimationSampler(input=0, output=2, interpolation="LINEAR"),
                ],
                channels=[
                    AnimationChannel(
                        sampler = 0,
                        target  = AnimationChannelTarget(node=0, path="translation"),
                    ),
                    AnimationChannel(
                        sampler = 1,
                        target  = AnimationChannelTarget(node=0, path="rotation"),
                    ),
                ],
            )
        ]
        gltf.set_binary_blob(blob)
        gltf.save(path)

    def test_posed_translation_uses_the_same_scale_factor_as_the_rest_pose(self):
        for factor in (1.0, 2.54, 100.0):
            rig = TransformList.load_glb(self.path, scale_factor=factor, pose_frame=0)
            self.assertTrue(
                np.allclose(rig["joint1"].translate,
                            np.array(self.POSED) * factor, atol=1e-5),
                f"posed translation ignored scale_factor={factor}",
            )

    def test_the_rest_and_posed_branches_agree_on_scale_factor(self):
        rest  = TransformList.load_glb(self.path, scale_factor=2.54)
        posed = TransformList.load_glb(self.path, scale_factor=2.54, pose_frame=0)

        ratio_rest  = np.array(rest["joint1"].translate) / np.array(self.REST)
        ratio_posed = np.array(posed["joint1"].translate) / np.array(self.POSED)

        self.assertTrue(np.allclose(ratio_rest, ratio_posed, atol=1e-5))

    def test_the_posed_joint_lands_on_the_orientation_in_the_file(self):
        rig = TransformList.load_glb(self.path, scale_factor=1.0, pose_frame=0)

        dot = np.abs(np.dot(rig["joint1"].quaternion.ravel(), self.quat.ravel()))

        self.assertAlmostEqual(dot, 1.0, places=6)

    def test_rebuilding_the_delta_in_xyz_only_works_because_glb_has_no_order(self):
        """gltf carries no rotate order, so every node load_glb builds is xyz

        That is what made hard coding the order harmless, and it is the only
        reason. The moment a caller or an extension supplies one, rebuilding
        the delta in xyz puts the joint somewhere else, so the conversion asks
        the node.
        """
        rig = TransformList.load_glb(self.path, scale_factor=1.0, pose_frame=0)
        self.assertEqual([node.rotate_order for node in rig], [0])

        q0 = TransformList.load_glb(self.path, scale_factor=1.0)["joint1"].quaternion
        delta = quaternion_multiply(
            quaternion_conjugate(q0), self.quat.reshape(1, 4)
        )

        for order in range(6):
            node        = TransformData(name="probe", rotate_order=order)
            node.rotate = np.degrees(quaternion_to_euler(delta, order)[0])
            asked       = np.abs(np.dot(node.quaternion.ravel(), delta.ravel()))

            node.rotate = np.degrees(quaternion_to_euler(delta, 0)[0])
            xyz         = np.abs(np.dot(node.quaternion.ravel(), delta.ravel()))

            self.assertAlmostEqual(asked, 1.0, places=6)
            if order != 0:
                self.assertLess(xyz, 1.0 - 1e-3)


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestLoadGlbNodeOrder(unittest.TestCase):
    """a glb's node array is free to list a child before its parent

    Exporters write it in whatever order suits them, and a dcc has to create a
    parent before it can parent anything to it, so the loader emits the tree
    parent first rather than the order the array happened to be read in.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def chain(self, name="order.glb"):
        """a three joint chain whose node array runs leaf first"""
        gltf       = GLTF2()
        gltf.scene = 0
        gltf.nodes = [
            Node(name="leaf", translation=[0.0, 1.0, 0.0]),
            Node(name="mid", translation=[0.0, 2.0, 0.0], children=[0]),
            Node(name="root", translation=[0.0, 3.0, 0.0], children=[1]),
        ]
        gltf.scenes  = [Scene(nodes=[2])]
        gltf.buffers = [Buffer(byteLength=4)]
        gltf.set_binary_blob(b"\x00\x00\x00\x00")

        path = os.path.join(self.tmp.name, name)
        gltf.save(path)
        return path

    def test_a_leaf_first_file_loads_parent_first(self):
        rig = HierarchyData.load_glb(self.chain(), scale_factor=1.0)

        self.assertEqual([str(node.name) for node in rig], ["root", "mid", "leaf"])

    def test_no_node_is_listed_before_its_parent(self):
        rig     = HierarchyData.load_glb(self.chain(), scale_factor=1.0)
        parents = rig.get_parents()

        early   = np.flatnonzero((parents >= 0) & (parents > np.arange(len(parents))))
        self.assertEqual(len(early), 0)

    def test_reordering_does_not_disturb_the_parenting(self):
        rig = HierarchyData.load_glb(self.chain(), scale_factor=1.0)

        self.assertIsNone(rig["root"].get_parent())
        self.assertEqual(str(rig["mid"].get_parent().name), "root")
        self.assertEqual(str(rig["leaf"].get_parent().name), "mid")

    def test_the_world_positions_still_stack_down_the_chain(self):
        """the reorder must move rows, not values"""
        rig = HierarchyData.load_glb(self.chain(), scale_factor=1.0)

        self.assertAlmostEqual(float(rig["root"].world_matrix[3, 1]), 3.0, places=6)
        self.assertAlmostEqual(float(rig["mid"].world_matrix[3, 1]),  5.0, places=6)
        self.assertAlmostEqual(float(rig["leaf"].world_matrix[3, 1]), 6.0, places=6)


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestLoadGlbDuplicateLeaves(unittest.TestCase):
    """three.js writes every joint of a skinned rig twice

    The second is childless, carries the same name, and hangs straight off the
    first. Both land in the same dcc namespace, and a parent reference by name
    can then mean either one.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, nodes, name="dupes.glb"):
        gltf         = GLTF2()
        gltf.scene   = 0
        gltf.nodes   = nodes
        gltf.scenes  = [Scene(nodes=[0])]
        gltf.buffers = [Buffer(byteLength=4)]
        gltf.set_binary_blob(b"\x00\x00\x00\x00")

        path = os.path.join(self.tmp.name, name)
        gltf.save(path)
        return path

    def twinned(self):
        """every joint shadowed by a childless namesake, plus a real end joint"""
        return self.write(
            [
                Node(name="Hips", translation=[0.0, 1.0, 0.0], children=[1, 2]),
                Node(name="Hips", translation=[0.0, 0.0, 0.0]),
                Node(name="Spine", translation=[0.0, 1.0, 0.0], children=[3, 4]),
                Node(name="Spine", translation=[0.0, 0.0, 0.0]),
                Node(name="Head", translation=[0.0, 1.0, 0.0]),
            ]
        )

    def test_the_childless_namesakes_are_dropped(self):
        rig = HierarchyData.load_glb(self.twinned(), scale_factor=1.0)

        self.assertEqual([str(node.name) for node in rig], ["Hips", "Spine", "Head"])

    def test_an_end_joint_under_a_different_name_survives(self):
        """Head is childless too, and it is not a copy of anything"""
        rig = HierarchyData.load_glb(self.twinned(), scale_factor=1.0)

        self.assertIn("Head", [str(node.name) for node in rig])

    def test_the_survivor_keeps_its_children(self):
        rig = HierarchyData.load_glb(self.twinned(), scale_factor=1.0)

        self.assertEqual(str(rig["Spine"].get_parent().name), "Hips")
        self.assertEqual(str(rig["Head"].get_parent().name), "Spine")

    def test_a_namesake_that_has_children_is_kept(self):
        """only the childless copy is a shadow, and dropping more would cut
        the rig in half at that joint
        """
        path = self.write(
            [
                Node(name="A", translation=[0.0, 1.0, 0.0], children=[1]),
                Node(name="A", translation=[0.0, 1.0, 0.0], children=[2]),
                Node(name="B", translation=[0.0, 1.0, 0.0]),
            ],
            name="kept.glb",
        )

        rig = HierarchyData.load_glb(path, scale_factor=1.0)

        self.assertEqual([str(node.name) for node in rig], ["A", "A", "B"])


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestLoadGlbMeshNodes(unittest.TestCase):
    """a node holding a mesh is not a joint, whatever its name says

    The loader used to match mesh nodes by the part of their name before the
    first colon, which read a namespaced mesh as a joint and dropped a real
    joint named like that namespace.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, nodes, roots, name="meshes.glb"):
        gltf         = GLTF2()
        gltf.scene   = 0
        gltf.nodes   = nodes
        gltf.meshes  = [Mesh(name="geo", primitives=[])]
        gltf.scenes  = [Scene(nodes=roots)]
        gltf.buffers = [Buffer(byteLength=4)]
        gltf.set_binary_blob(b"\x00\x00\x00\x00")

        path = os.path.join(self.tmp.name, name)
        gltf.save(path)
        return path

    def test_a_namespaced_mesh_neither_loads_nor_hides_a_joint(self):
        path = self.write(
            [
                Node(name="ns", translation=[0.0, 1.0, 0.0], children=[1]),
                Node(name="ns:spine", translation=[0.0, 1.0, 0.0]),
                Node(name="ns:body", mesh=0),
            ],
            roots=[0, 2],
        )

        rig = HierarchyData.load_glb(path, scale_factor=1.0)

        self.assertEqual([str(node.name) for node in rig], ["ns", "ns:spine"])
        self.assertEqual(str(rig["ns:spine"].get_parent().name), "ns")

    def test_a_plain_mesh_is_still_skipped(self):
        path = self.write(
            [
                Node(name="root", translation=[0.0, 1.0, 0.0]),
                Node(name="body", mesh=0),
            ],
            roots=[0, 1],
        )

        rig = HierarchyData.load_glb(path, scale_factor=1.0)

        self.assertEqual([str(node.name) for node in rig], ["root"])

    def namesake(self):
        """root -> head, plus a mesh node also named head listed before it

        Both are keyed in translation, the mesh's channel last: the joint
        from [0, 1, 0] to [7, 1, 0], the mesh from [0, 0, 0] to [0, 5, 0].
        """
        times = np.array([0.0, 1.0], dtype=np.float32)
        joint = np.array([[0.0, 1.0, 0.0], [7.0, 1.0, 0.0]], dtype=np.float32)
        mesh  = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)

        views, blob = [], b""
        for array in (times, joint, mesh):
            views.append(
                BufferView(buffer=0, byteOffset=len(blob), byteLength=array.nbytes)
            )
            blob += array.tobytes()

        gltf       = GLTF2()
        gltf.scene = 0
        gltf.nodes = [
            Node(name="root", translation=[0.0, 1.0, 0.0], children=[2]),
            Node(name="head", mesh=0),
            Node(name="head", translation=[0.0, 1.0, 0.0]),
        ]
        gltf.meshes      = [Mesh(name="geo", primitives=[])]
        gltf.scenes      = [Scene(nodes=[0, 1])]
        gltf.buffers     = [Buffer(byteLength=len(blob))]
        gltf.bufferViews = views
        gltf.accessors = [
            Accessor(bufferView=0, componentType=FLOAT, count=2, type="SCALAR"),
            Accessor(bufferView=1, componentType=FLOAT, count=2, type="VEC3"),
            Accessor(bufferView=2, componentType=FLOAT, count=2, type="VEC3"),
        ]
        gltf.animations = [
            Animation(
                samplers=[
                    AnimationSampler(input=0, output=1),
                    AnimationSampler(input=0, output=2),
                ],
                channels=[
                    AnimationChannel(
                        sampler = 0,
                        target  = AnimationChannelTarget(node=2, path="translation"),
                    ),
                    AnimationChannel(
                        sampler = 1,
                        target  = AnimationChannelTarget(node=1, path="translation"),
                    ),
                ],
            )
        ]
        gltf.set_binary_blob(blob)

        path = os.path.join(self.tmp.name, "namesake.glb")
        gltf.save(path)
        return path

    def test_pose_frame_reads_the_joint_not_a_mesh_of_its_name(self):
        """the pose looked channels up by name, so the mesh's key, read
        last, overwrote the joint's
        """
        path = self.namesake()

        for frame, want in ((0, [0.0, 1.0, 0.0]), (1, [7.0, 1.0, 0.0])):
            with self.subTest(pose_frame=frame):
                rig = HierarchyData.load_glb(path, scale_factor=1.0, pose_frame=frame)

                self.assertEqual([str(node.name) for node in rig], ["root", "head"])
                self.assertTrue(
                    np.allclose(rig["head"].translate, want), rig["head"].translate
                )


class TestSerializeStaysOutOfHome(unittest.TestCase):
    """test_serialize used to save skeleton.pkl, .npz and .json into ~"""

    def test_nothing_lands_in_the_home_folder(self):
        with tempfile.TemporaryDirectory() as home:
            # expanduser reads HOME on posix and USERPROFILE on windows
            with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                result = unittest.TestResult()
                TestHierarchy("test_serialize").run(result)

            self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
            self.assertEqual(os.listdir(home), [])


class TestTransformListCopy(unittest.TestCase):
    """a copy holds new nodes that no hierarchy owns, so it owns them itself

    A view can point every node back at the rig it was selected from. A copy
    of one has no rig to point at, and leaving its nodes unowned does not
    preserve that link -- it drops every query that reads it.
    """

    def chain(self, count=4):
        rig = HierarchyData(
            [
                TransformData(
                    name      = f"j{i}",
                    node_type = "joint",
                    translate = [0.0, 10.0, 0.0],
                    rotate    = [0.0, 0.0, i * 3.0],
                )
                for i in range(count)
            ]
        )
        for i in range(1, count):
            rig[f"j{i}"].set_parent(f"j{i - 1}", world_space=False)
        return rig

    def test_a_copied_view_owns_its_nodes(self):
        rig    = self.chain()
        copied = rig[::-1].copy()

        for node in copied:
            self.assertIs(node._hierarchy, copied)

    def test_a_copied_view_resolves_its_own_parents(self):
        """the parent has to be the copy's node, not the source's"""
        rig    = self.chain()
        copied = rig[::-1].copy()

        parent = copied["j2"].get_parent()

        self.assertIsInstance(parent, TransformData)
        self.assertIs(parent, copied["j1"])
        self.assertIsNone(copied["j0"].get_parent())

    def test_a_copied_view_keeps_its_world_matrices(self):
        """an unowned copy reads every parent as identity, which collapses
        the chain onto its own local transforms without saying so
        """
        rig    = self.chain()
        copied = rig[::-1].copy()

        for name in ("j0", "j1", "j2", "j3"):
            self.assertTrue(
                np.allclose(copied[name].world_matrix, rig[name].world_matrix)
            )

    def test_copying_a_view_leaves_the_source_alone(self):
        rig    = self.chain()
        copied = rig[::-1].copy()

        self.assertEqual(len(rig), 4)
        for node in rig:
            self.assertIs(node._hierarchy, rig)

        self.assertIsNot(copied["j0"], rig["j0"])

    def test_a_partial_selection_roots_the_nodes_it_cut(self):
        """j0 is left behind, so j1 is a root of the copy rather than a node
        naming a parent the copy cannot produce
        """
        rig    = self.chain()
        copied = rig[1:].copy()

        self.assertIsNone(copied["j1"].get_parent())
        self.assertEqual([str(node.name) for node in copied.get_roots()], ["j1"])

        for node in copied:
            self.assertNotIsInstance(node.get_parent(), str)

    def test_an_owning_hierarchy_copies_to_its_own_class(self):
        rig = self.chain()

        self.assertIs(type(rig.copy()), HierarchyData)

    def test_a_subclass_keeps_its_own_class(self):
        """Scene and ClipData both subclass HierarchyData, and neither may
        come back as the base class
        """

        class Subclass(HierarchyData):
            pass

        rig = Subclass([TransformData(name="j0", node_type="joint")])

        self.assertIs(type(rig.copy()), Subclass)

    def test_the_copy_is_independent_of_the_source(self):
        rig    = self.chain()
        copied = rig[::-1].copy()

        copied["j2"].translate = [5.0, 5.0, 5.0]

        self.assertFalse(np.allclose(rig["j2"].translate, [5.0, 5.0, 5.0]))


class TestNamespaces(unittest.TestCase):
    """strip_namespace and namespace on nodes, views and hierarchies"""

    def rig(self):
        """VLR_RIG:SK:root -> VLR_RIG:SK:spine -> VLR_RIG:ctrl"""
        rig = HierarchyData(
            [
                TransformData(name="VLR_RIG:SK:root", node_type="joint"),
                TransformData(name="VLR_RIG:SK:spine", node_type="joint"),
                TransformData(name="VLR_RIG:ctrl"),
            ]
        )
        rig["VLR_RIG:SK:spine"].set_parent("VLR_RIG:SK:root", world_space=False)
        rig["VLR_RIG:ctrl"].set_parent("VLR_RIG:SK:spine", world_space=False)
        rig["VLR_RIG:SK:spine"].translate = [0.0, 10.0, 0.0]
        rig["VLR_RIG:ctrl"].translate     = [0.0, 5.0, 0.0]
        return rig

    @staticmethod
    def parents(rig):
        return [n.get_parent().name if n.get_parent() else None for n in rig]

    def test_namespace(self):
        rig = self.rig()
        self.assertEqual(rig["VLR_RIG:SK:root"].namespace, "VLR_RIG:SK")
        self.assertEqual(rig.namespace, ["VLR_RIG:SK", "VLR_RIG:SK", "VLR_RIG"])
        self.assertEqual(TransformData(name="root").namespace, "")
        self.assertEqual(TransformData(name=":A:root").namespace, "A")

    def test_strip_every_namespace(self):
        rig   = self.rig()
        world = [n.world_matrix for n in rig]
        rig.strip_namespace()
        self.assertEqual(rig.name, ["root", "spine", "ctrl"])
        self.assertEqual(self.parents(rig), [None, "root", "spine"])
        self.assertTrue(np.allclose([n.world_matrix for n in rig], world))

    def test_strip_one_namespace_moves_its_content_up(self):
        rig = self.rig()
        rig.strip_namespace("VLR_RIG:SK")
        self.assertEqual(rig.name, ["VLR_RIG:root", "VLR_RIG:spine", "VLR_RIG:ctrl"])

        rig = self.rig()
        rig.strip_namespace("VLR_RIG")
        self.assertEqual(rig.name, ["SK:root", "SK:spine", "ctrl"])
        self.assertEqual(self.parents(rig), [None, "SK:root", "SK:spine"])

    def test_a_path_matches_from_the_top(self):
        # SK sits inside VLR_RIG, and a node name is not a namespace
        rig = self.rig()
        for path in ("SK", "OTHER", "VLR_RIG:SK:root"):
            rig.strip_namespace(path)
        self.assertEqual(rig.name, self.rig().name)

    def test_leading_colon_and_empty_path(self):
        rig = self.rig()
        rig.strip_namespace(":VLR_RIG:SK")
        self.assertEqual(rig.name, ["VLR_RIG:root", "VLR_RIG:spine", "VLR_RIG:ctrl"])
        for path in ("", ":"):
            with self.assertRaises(ValueError):
                rig.strip_namespace(path)

    def test_a_clash_renames_nothing(self):
        rig = HierarchyData(
            [TransformData(name="hero:root"), TransformData(name="villain:root")]
        )
        with self.assertRaisesRegex(ValueError, "root <- hero:root, villain:root"):
            rig.strip_namespace()
        self.assertEqual(rig.name, ["hero:root", "villain:root"])

    def test_names_already_shared_do_not_block(self):
        rig = HierarchyData(
            [
                TransformData(name="dup"),
                TransformData(name="dup"),
                TransformData(name="A:x"),
            ]
        )
        rig.strip_namespace()
        self.assertEqual(rig.name, ["dup", "dup", "x"])

    def test_a_view_is_checked_against_its_whole_hierarchy(self):
        rig = HierarchyData([TransformData(name="root"), TransformData(name="A:root")])
        with self.assertRaises(ValueError):
            rig.match("A:*").strip_namespace()
        self.assertEqual(rig.name, ["root", "A:root"])

        rig = HierarchyData([TransformData(name="A:x"), TransformData(name="B:y")])
        rig.match("A:*").strip_namespace()
        self.assertEqual(rig.name, ["x", "B:y"])

    def test_a_single_node(self):
        rig = HierarchyData(
            [TransformData(name="hero:root"), TransformData(name="villain:root")]
        )
        rig["hero:root"].strip_namespace()
        self.assertEqual(rig.name, ["root", "villain:root"])
        with self.assertRaises(ValueError):
            rig["villain:root"].strip_namespace()
        self.assertEqual(rig.name, ["root", "villain:root"])

        node = TransformData(name="A:B:node")
        node.strip_namespace("A")
        self.assertEqual(node.name, "B:node")

    def test_prefix_and_suffix_leave_the_namespace_in_front(self):
        rig = self.rig()
        rig.add_prefix("L_")
        rig.add_suffix("_JNT")
        self.assertEqual(
            rig.name,
            ["VLR_RIG:SK:L_root_JNT", "VLR_RIG:SK:L_spine_JNT", "VLR_RIG:L_ctrl_JNT"],
        )

        node = TransformData(name="root")
        node.add_prefix("L_")
        self.assertEqual(node.name, "L_root")


class TestUserAttributes(unittest.TestCase):
    """user defined attributes read and set as properties"""

    @staticmethod
    def node():
        return TransformData(
            name="root",
            user_defined_attributes={
                "heroHeight": {"attributeType": "double", "value": 1.8, "keyable": True},
                "count":      {"attributeType": "long", "value": 3, "keyable": True},
                "flag":       {"attributeType": "bool", "value": False, "keyable": True},
                "mode": {
                    "attributeType": "enum",
                    "enumName":      "a=1:b=5:c",
                    "value":         5,
                    "keyable":       True,
                },
                "label": {"dataType": "string", "value": "hero", "keyable": False},
                "uuids": {"dataType": "stringArray", "value": ["a"], "keyable": False},
                "vec":   {"attributeType": "double3", "numberOfChildren": 3},
                "vecX":  {"attributeType": "double", "value": 1.0, "parent": "vec"},
                "vecY":  {"attributeType": "double", "value": 2.0, "parent": "vec"},
                "vecZ":  {"attributeType": "double", "value": 3.0, "parent": "vec"},
                "grp":   {"attributeType": "compound", "numberOfChildren": 2},
                "grpA":  {"attributeType": "double", "value": 4.5, "parent": "grp"},
                "grpB":  {"attributeType": "bool", "value": True, "parent": "grp"},
                "weights": {
                    "attributeType": "double",
                    "multi":         True,
                    "value":         [[0, 1.5], [2, 3.5]],
                },
                "radius": {"attributeType": "double", "value": 9.0},
            },
        )

    def test_read(self):
        node = self.node()
        self.assertEqual(node.heroHeight, 1.8)
        self.assertEqual(node.label,      "hero")
        self.assertEqual(node.uuids,      ["a"])
        self.assertEqual(node.mode,       5)
        self.assertEqual(node.weights,    [[0, 1.5], [2, 3.5]])

    def test_write_converts_to_the_type(self):
        node            = self.node()
        node.heroHeight = 3
        node.count      = 4.0
        node.flag       = 1
        node.uuids      = ("b", "c")
        self.assertEqual((node.heroHeight, node.count, node.flag), (3.0, 4, True))
        self.assertIsInstance(node.heroHeight, float)
        self.assertIsInstance(node.count, int)
        self.assertEqual(node.uuids, ["b", "c"])
        self.assertEqual(node.user_defined_attributes["heroHeight"]["value"], 3.0)

    def test_write_raises_and_keeps_the_value(self):
        node = self.node()
        for name, value, error in (
            ("heroHeight", "tall", TypeError),
            ("count", 3.5, ValueError),
            ("flag", 2, ValueError),
            ("label", 5, TypeError),
            ("uuids", ["a", 1], TypeError),
            ("mode", "z", ValueError),
            ("mode", 3, ValueError),
        ):
            before = node.user_defined_attributes[name]["value"]
            with self.assertRaises(error):
                setattr(node, name, value)
            self.assertEqual(node.user_defined_attributes[name]["value"], before)

    def test_enum_by_name_or_number(self):
        node      = self.node()
        node.mode = "c"
        self.assertEqual(node.mode, 6)
        node.mode = 1
        self.assertEqual(node.mode, 1)

    def test_compound(self):
        node = self.node()
        self.assertIsInstance(node.vec, np.ndarray)
        self.assertTrue(np.array_equal(node.vec, [1.0, 2.0, 3.0]))
        self.assertEqual(node.grp, (4.5, True))

        node.vec = np.array([4, 5, 6])
        self.assertEqual([node.vecX, node.vecY, node.vecZ], [4.0, 5.0, 6.0])
        node.grp = [0.5, 0]
        self.assertEqual(node.grp, (0.5, False))
        with self.assertRaises(ValueError):
            node.vec = (1, 2)
        with self.assertRaises(TypeError):
            node.vec = 1.0

    def test_multi(self):
        node         = self.node()
        node.weights = [[1, 2]]
        self.assertEqual(node.weights, [[1, 2.0]])
        with self.assertRaises(TypeError):
            node.weights = [1, 2]

    def test_missing_names_still_raise(self):
        node = self.node()
        with self.assertRaises(AttributeError):
            node.heroHieght
        with self.assertRaisesRegex(AttributeError, "add_user_attribute"):
            node.heroHieght = 3
        self.assertNotIn("heroHieght", node.user_defined_attributes)

    def test_built_in_names_win(self):
        node = self.node()
        self.assertEqual(node.radius, 1.0)
        node.radius = 2.0
        self.assertEqual(node.radius, 2.0)
        self.assertEqual(node.user_defined_attributes["radius"]["value"], 9.0)

    def test_add_user_attribute(self):
        node = TransformData(name="root")
        node.add_user_attribute("heroHeight", 3.1416)
        node.add_user_attribute("count",      3)
        node.add_user_attribute("flag",       True)
        node.add_user_attribute("label",      "hero")
        node.add_user_attribute("uuids",      ["a", "b"])
        node.add_user_attribute("weights",    [1, 2.5])
        node.add_user_attribute(
            "mode", "b", attribute_type="enum", enum_names="a=1:b=5:c"
        )
        node.add_user_attribute("size", 2, attribute_type="short", keyable=False)

        attrs = node.user_defined_attributes
        kinds = {k: v.get("attributeType", v.get("dataType")) for k, v in attrs.items()}
        self.assertEqual(
            kinds,
            {
                "heroHeight": "double",
                "count":      "long",
                "flag":       "bool",
                "label":      "string",
                "uuids":      "stringArray",
                "weights":    "doubleArray",
                "mode":       "enum",
                "size":       "short",
            },
        )
        self.assertEqual(node.heroHeight, 3.1416)
        self.assertEqual(node.mode,       5)
        self.assertEqual(node.weights,    [1.0, 2.5])

        # Maya can not key strings or arrays
        self.assertTrue(attrs["heroHeight"]["keyable"])
        self.assertFalse(attrs["label"]["keyable"])
        self.assertFalse(attrs["uuids"]["keyable"])
        self.assertFalse(attrs["size"]["keyable"])

    def test_an_enum_name_reads_as_maya_reads_it(self):
        # the fields Maya 2025's addAttr makes: a "=" gives an index only when
        # it splits the field into a name and a whole number, spaces or tabs
        # around it, and a run of "=", or one at either end, splits nothing;
        # any other field keeps its "=" in its name. a field holding one, as
        # another tool's fbx can, raised a bare int() ValueError on any read
        # or write of the attr, and "k==2" was one field, Maya's field k at 2
        from cgmath.hierarchy.hierarchy import _enum_fields

        for enum_name, fields in (
            ("a=1:b=5:c", [("a", 1), ("b", 5), ("c", 6)]),
            ("size=small:size=large", [("size=small", 0), ("size=large", 1)]),
            ("a= 1 :b", [("a", 1), ("b", 2)]),
            ("a=-1:b", [("a", -1), ("b", 0)]),
            ("a=1=2:b", [("a=1=2", 0), ("b", 1)]),
            ("a=1_0:b", [("a=1_0", 0), ("b", 1)]),
            ("a=1.0:b", [("a=1.0", 0), ("b", 1)]),
            ("a=:b", [("a=", 0), ("b", 1)]),
            ("=3:b", [("=3", 0), ("b", 1)]),
            ("a b=2:c", [("a b", 2), ("c", 3)]),
            ("k==2:b", [("k", 2), ("b", 3)]),
            ("k=2=:b", [("k", 2), ("b", 3)]),
            ("=k=2:b", [("k", 2), ("b", 3)]),
            ("k===2:b", [("k", 2), ("b", 3)]),
            ("k=2==:b", [("k", 2), ("b", 3)]),
            ("=a==1==:c", [("a", 1), ("c", 2)]),
            ("a=1:b==5:c", [("a", 1), ("b", 5), ("c", 6)]),
            ("==2:b", [("==2", 0), ("b", 1)]),
            ("a==:b", [("a==", 0), ("b", 1)]),
            ("a==b:c", [("a==b", 0), ("c", 1)]),
            ("k= =2:b", [("k= =2", 0), ("b", 1)]),
            ("= =1:b", [(" ", 1), ("b", 2)]),
            ("a=\t1\t:b", [("a", 1), ("b", 2)]),
            ("a=\n1:b", [("a=\n1", 0), ("b", 1)]),
            ("a=\x1c1:b", [("a=\x1c1", 0), ("b", 1)]),
        ):
            with self.subTest(enum_name=enum_name):
                self.assertEqual(_enum_fields(enum_name), fields)

        node = TransformData(name="root")
        node.add_user_attribute(
            "mode", "size=large", attribute_type="enum", enum_names="size=small:size=large"
        )
        self.assertEqual(node.mode, 1)
        node.mode = 0
        self.assertEqual(node.mode, 0)
        with self.assertRaisesRegex(ValueError, "root.mode has no field 'size'"):
            node.mode = "size"

        node.add_user_attribute("runs", "b", attribute_type="enum", enum_names="k==2:b")
        self.assertEqual(node.runs, 3)
        node.runs = "k"
        self.assertEqual(node.runs, 2)

    def test_add_user_attribute_refuses(self):
        node = TransformData(name="root")
        node.add_user_attribute("heroHeight", 1.8)
        for args, kwargs, error in (
            (("heroHeight", 2.0), {}, ValueError),
            (("translate", 2.0), {}, ValueError),
            (("1abc", 2.0), {}, ValueError),
            (("_x", 2.0), {}, ValueError),
            (("empty", []), {}, TypeError),
            (("vec", 1.0), {"attribute_type": "double3"}, ValueError),
            (("mode", 1), {"attribute_type": "enum"}, ValueError),
            (("tall", "x"), {"attribute_type": "double"}, TypeError),
        ):
            with self.assertRaises(error):
                node.add_user_attribute(*args, **kwargs)
        self.assertEqual(list(node.user_defined_attributes), ["heroHeight"])

    def test_tab_completion(self):
        node = self.node()
        self.assertIn("heroHeight", dir(node))
        self.assertIn("heroHeight", dir(HierarchyData([node])))

    def test_lists(self):
        rig = HierarchyData([self.node(), TransformData(name="hip")])
        self.assertEqual(rig.heroHeight, [1.8, None])
        rig.heroHeight = 2
        self.assertEqual(rig.heroHeight, [2.0, None])
        self.assertTrue(np.array_equal(rig.vec[0], [1.0, 2.0, 3.0]))

        with self.assertRaises(AttributeError):
            rig.nope
        with self.assertRaisesRegex(AttributeError, "add_user_attribute"):
            rig.nope = 1
        self.assertNotIn("nope", rig.__dict__)

    def test_a_subclass_sets_what_it_declares(self):
        class Settings(HierarchyData):
            quality: float

        rig         = Settings([TransformData(name="root")])
        rig.quality = 1.5
        self.assertEqual(rig.quality, 1.5)
        with self.assertRaises(AttributeError):
            rig.qualty = 1.5

    def test_list_writes_are_all_or_nothing(self):
        # a double takes 2.5, a bool does not
        a = TransformData(
            name="a", user_defined_attributes={"v": {"attributeType": "double", "value": 1.0}}
        )
        b = TransformData(
            name="b", user_defined_attributes={"v": {"attributeType": "bool", "value": False}}
        )
        rig = HierarchyData([a, b])
        with self.assertRaises(ValueError):
            rig.v = 2.5
        self.assertEqual(rig.v, [1.0, False])

    def test_a_view_writes_only_its_nodes(self):
        rig = HierarchyData([self.node(), TransformData(name="hip")])
        rig["hip"].add_user_attribute("heroHeight", 1.0)
        rig.match("hip").heroHeight = 5
        self.assertEqual(rig.heroHeight, [1.8, 5.0])

    def test_list_add_user_attribute(self):
        rig = HierarchyData([TransformData(name="a"), TransformData(name="b")])
        rig.add_user_attribute("uuids", ["x"])
        self.assertEqual(rig.uuids, [["x"], ["x"]])

        # each node owns its value
        rig["a"].uuids = ["y"]
        self.assertEqual(rig.uuids, [["y"], ["x"]])

        rig["b"].add_user_attribute("tag", "t")
        with self.assertRaises(ValueError):
            rig.add_user_attribute("tag", "t")
        self.assertNotIn("tag", rig["a"].user_defined_attributes)

    def test_values_survive_copy_pickle_and_json(self):
        node            = self.node()
        node.heroHeight = 2.5
        self.assertEqual(node.copy().heroHeight, 2.5)
        self.assertEqual(pickle.loads(pickle.dumps(node)).heroHeight, 2.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = HierarchyData([node]).save(os.path.join(tmp, "rig.json"))
            self.assertEqual(HierarchyData.load(path)["root"].heroHeight, 2.5)

    def test_none_stores_an_empty_dict(self):
        """the field is a dict; None means no attributes, never None itself"""
        node                         = self.node()
        node.user_defined_attributes = None
        self.assertEqual(node.user_defined_attributes, {})

        # and every reader takes it as the dict it is
        node.add_user_attribute("hero", 1.0)
        self.assertEqual(node.hero, 1.0)

        rig = HierarchyData([TransformData("a"), TransformData("b")])
        for each in rig:
            each.user_defined_attributes = None
        rig.add_user_attribute("tag", 2)
        self.assertEqual(rig.tag, [2, 2])
        self.assertIn("tag", dir(rig))

        # a tree that holds None, as an older file can, loads as {}
        tree                            = TransformData("c").to_dict()
        tree["user_defined_attributes"] = None
        self.assertEqual(TransformData.from_dict(tree).user_defined_attributes, {})


class TestNodeType(unittest.TestCase):
    """setting node_type on every node at once, and on one node"""

    @staticmethod
    def rig():
        """a joint chain with joint orients, rotate axes and mixed rotate orders"""
        hierarchy = HierarchyData()
        for i in range(4):
            node = TransformData(
                name=f"j{i}",
                node_type="joint",
                translate=[2.0 if i else 0.0, 0.5 * i, 0.0],
                rotate=[10.0 * i, -7.0 * i, 3.0 + i],
                rotate_order=i % 6,
                rotate_axis=[5.0, 0.0, -3.0 * i],
                joint_orient=[0.0, 15.0 * i, 4.0 * i - 20.0],
                radius=2.5,
                draw_style=2,
                segment_scale_compensate=True,
            )
            if i:
                node.parent_uuid = hierarchy[i - 1].uuid
            hierarchy.append(node)
        return hierarchy

    def test_joint_to_transform_keeps_the_pose(self):
        hierarchy = self.rig()
        local     = np.array([node.matrix for node in hierarchy])
        world     = np.array([node.world_matrix for node in hierarchy])
        axes      = np.array([node.rotate_axis for node in hierarchy])

        hierarchy.node_type = "transform"

        self.assertEqual(hierarchy.node_type, ["transform"] * 4)
        self.assertTrue(np.allclose([node.matrix for node in hierarchy], local, atol=1e-9))
        self.assertTrue(np.allclose([node.world_matrix for node in hierarchy], world, atol=1e-9))
        self.assertTrue(np.allclose([node.joint_orient for node in hierarchy], 0.0))
        self.assertTrue(np.allclose([node.rotate_axis for node in hierarchy], axes))
        self.assertFalse(np.any(hierarchy.segment_scale_compensate))
        self.assertTrue(np.all(hierarchy.radius == 1.0))
        self.assertTrue(np.all(hierarchy.draw_style == 0))

    def test_one_type_per_node(self):
        hierarchy           = self.rig()
        hierarchy.node_type = ["joint", "locator", "transform", "joint"]
        self.assertEqual(hierarchy.node_type, ["joint", "locator", "transform", "joint"])

        # joints keep their joint-only attributes
        self.assertEqual(hierarchy[0].radius, 2.5)
        self.assertEqual(hierarchy[3].radius, 2.5)
        self.assertEqual(hierarchy[1].radius, 1.0)

        with self.assertRaises(ValueError):
            hierarchy.node_type = ["joint", "locator"]

    def test_a_slice_sets_a_subset(self):
        hierarchy                = self.rig()
        hierarchy[1:3].node_type = "locator"
        self.assertEqual(hierarchy.node_type, ["joint", "locator", "locator", "joint"])

    def test_other_changes_only_set_the_type(self):
        node = TransformData(
            name="t",
            node_type="transform",
            joint_orient=[0.0, 10.0, 0.0],
            radius=3.0,
            draw_style=1,
            segment_scale_compensate=True,
        )
        for node_type in ("locator", "joint"):
            node.set_node_type(node_type)
            self.assertEqual(node.node_type, node_type)
            self.assertTrue(np.allclose(node.joint_orient, [0.0, 10.0, 0.0]))
            self.assertEqual(node.radius, 3.0)
            self.assertEqual(node.draw_style, 1)
            self.assertTrue(node.segment_scale_compensate)

    def test_set_node_type_on_one_node(self):
        whole, each = self.rig(), self.rig()
        whole.node_type = "locator"
        for node in each:
            node.set_node_type("locator")

        for a, b in zip(whole, each):
            self.assertEqual(a.node_type, b.node_type)
            self.assertTrue(np.allclose(a.rotate, b.rotate))
            self.assertTrue(np.allclose(a.joint_orient, b.joint_orient))
            self.assertEqual(a.segment_scale_compensate, b.segment_scale_compensate)
            self.assertEqual((a.radius, a.draw_style), (b.radius, b.draw_style))

        # a plain assignment sets the type and nothing else
        raw                 = self.rig()
        raw["j2"].node_type = "transform"
        self.assertEqual(raw["j2"].radius, 2.5)
        self.assertTrue(np.allclose(raw["j2"].joint_orient, [0.0, 30.0, -12.0]))

    def test_segment_scale_compensate_turns_off_under_a_scaled_parent(self):
        hierarchy           = self.rig()
        hierarchy[0].scale  = [2.0, 2.0, 2.0]
        before              = hierarchy[1].matrix

        hierarchy.node_type = "transform"
        after               = hierarchy[1].matrix

        # the child no longer cancels its parent's 2x scale; translate stays
        self.assertTrue(np.allclose(after[:3, :3], 2.0 * before[:3, :3], atol=1e-9))
        self.assertTrue(np.allclose(after[3], before[3], atol=1e-9))


def _chain() -> HierarchyData:
    """root (1, 0, 0) > spine (0, 2, 0) > head (0, 1, 0), parented by uuid"""
    root  = TransformData("root",  translate=(1.0, 0.0, 0.0))
    spine = TransformData("spine", translate=(0.0, 2.0, 0.0))
    head  = TransformData("head",  translate=(0.0, 1.0, 0.0))
    rig   = HierarchyData([root, spine, head])
    spine.set_parent(root, world_space=False)
    head.set_parent(spine, world_space=False)
    return rig


def _ids(*names) -> dict:
    """a new uuid for each name"""
    return {name: generate_uuid() for name in names}


class TestParentLookup(unittest.TestCase):
    """_parent_indices and _owning_parent_first look parents up through dicts;
    they must give get_parent's answer for every kind of parent_uuid"""

    def _rig(self) -> HierarchyData:
        named  = TransformData("named")
        first  = TransformData("twin", translate=(1.0, 0.0, 0.0))
        second = TransformData("twin", translate=(2.0, 0.0, 0.0))
        shaped = TransformData(generate_uuid())   # a name shaped like a uuid
        odd    = TransformData("odd", uuid="j1")  # a uuid not shaped like one
        rig    = HierarchyData([named, first, second, shaped, odd])

        # a name equal to another node's uuid, and a cycle, written in place
        rig.append(TransformData(rig["twin"].uuid))
        cycle = [TransformData("cycle_a"), TransformData("cycle_b")]
        rig.extend(cycle)
        cycle[0].parent_uuid = cycle[1].uuid
        cycle[1].parent_uuid = cycle[0].uuid
        refs = [
            None, named.uuid, first.uuid, second.uuid, shaped.name, generate_uuid(),
            named.uuid.lower(), cycle[0].uuid,
        ]
        children = [TransformData(f"child{i}") for i in range(len(refs))]
        rig.extend(children)
        for child, ref in zip(children, refs):
            child.parent_uuid = ref  # those no node carries left in place
        for i, child in enumerate(children):
            rig.append(TransformData(f"grandchild{i}", parent_uuid=child.uuid))
        return rig

    @staticmethod
    def _position(nodes, parent) -> int:
        for i, node in enumerate(nodes):
            if node is parent:
                return i
        return -1

    def test_parent_indices_match_get_parent(self):
        rig = self._rig()
        for nodes in (rig.list, rig.list[::-1], rig.list[::3]):
            expected = [self._position(nodes, node.get_parent()) for node in nodes]
            self.assertEqual(list(TransformList._parent_indices(nodes)), expected)

    def test_owning_parent_first_matches_a_get_parent_walk(self):
        def depth(node):
            seen, parent, count = set(), node.get_parent(), 0
            while isinstance(parent, TransformData) and id(parent) not in seen:
                seen.add(id(parent))
                count += 1
                parent = parent.get_parent()
            return count

        rig = self._rig()
        for nodes in (rig.list, rig.list[::-1], rig.list[::3]):
            expected = np.argsort([depth(node) for node in nodes], kind="stable")
            self.assertEqual(
                list(TransformList._owning_parent_first(nodes)), list(expected)
            )


class TestCopiesOwnTheirNodes(unittest.TestCase):
    """a node belongs to one hierarchy: an edit to a copy leaves the original whole"""

    def test_a_shallow_copy_of_a_hierarchy_owns_its_nodes(self):
        rig  = _chain()
        twin = copy.copy(rig)
        twin.pop(1)
        self.assertEqual(rig.name, ["root", "spine", "head"])
        self.assertIs(rig["spine"].get_parent(), rig["root"])
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [1.0, 3.0, 0.0])

    def test_a_shallow_copy_of_a_view_is_a_view(self):
        rig  = _chain()
        view = TransformList(rig[1:])
        twin = copy.copy(view)
        self.assertIs(type(twin), TransformList)
        self.assertIs(twin[0], rig["spine"])
        twin.pop(0)
        self.assertEqual(len(view), 2)

    def test_pop_moves_the_orphans_to_the_root(self):
        rig = _chain()
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [1.0, 3.0, 0.0])
        rig.pop(1)
        # head lost its parent: its cached world matrix goes with it
        self.assertIsNone(rig["head"].parent_uuid)
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [0.0, 1.0, 0.0])

    def test_nodes_built_from_one_attribute_dict_do_not_share_it(self):
        attrs  = {"hero": {"attributeType": "double", "value": 1.0, "keyable": True}}
        p      = TransformData("p", user_defined_attributes=attrs)
        q      = TransformData("q", user_defined_attributes=attrs)
        p.hero = 3.0
        self.assertEqual(q.hero, 1.0)
        self.assertEqual(attrs["hero"]["value"], 1.0)


def _mixed_rig() -> HierarchyData:
    """a branching rig with a name two nodes share: root > spine > neck >
    head > tip, spine > twin > hand, root > twin > toe, each node holding its
    parent's uuid; "twin" names the first twin"""
    rng = np.random.default_rng(23)
    specs = [
        ("root", None),
        ("spine", 0),
        ("neck", 1),
        ("head", 2),
        ("twin", 1),
        ("hand", 4),
        ("twin", 0),
        ("toe", 6),
        ("tip", 3),
    ]
    nodes = []
    for name, parent in specs:
        nodes.append(
            TransformData(
                name=name,
                uuid=generate_uuid(),
                parent_uuid=None if parent is None else nodes[parent].uuid,
                translate=rng.normal(size=3),
                rotate=rng.uniform(-90.0, 90.0, size=3),
                scale=rng.uniform(0.7, 1.4, size=3),
                rotate_order=int(rng.integers(0, 6)),
                segment_scale_compensate=bool(rng.integers(0, 2)),
            )
        )
    return HierarchyData(nodes)


def _children_by_scan(node) -> list:
    """get_children's rule, by a scan of every node of the hierarchy: the
    nodes whose parent_uuid is the node's uuid"""
    rig = node._hierarchy
    if rig is None or node.uuid is None:
        return []
    return [other for other in rig.list if other.parent_uuid == node.uuid]


def _branch_by_scan(node) -> list:
    branch = [node]
    for child in _children_by_scan(node):
        branch.extend(_branch_by_scan(child))
    return branch


class _Walked(list):
    """a node list that counts the passes over it, from either end"""

    walks = 0

    def __iter__(self):
        self.walks += 1
        return super().__iter__()

    def __reversed__(self):
        self.walks += 1
        return super().__reversed__()


def _swap_twins(rig) -> None:
    first  = rig.index("twin")
    second = [i for i, n in enumerate(rig) if n.name == "twin"][1]
    rig[first], rig[second] = rig[second], rig[first]


def _assert_index_is_a_build(case, rig, label) -> None:
    """the index a rig keeps, patched in place or built, holds what a build
    of the rig as it now stands holds, its name table too once built"""
    from cgmath.hierarchy.hierarchy import _RigIndex

    kept = rig._kept_index()
    if kept is None:
        return
    fresh = _RigIndex.build(rig)
    case.assertIsNotNone(fresh, f"{label}: an index kept where a build gives none")
    for table in ("uuids", "refs"):
        # a long row of refs a patch made a deque holds what a build's list does
        case.assertEqual(
            {key: value if isinstance(value, int) else list(value) for key, value in getattr(kept, table).items()},
            {key: value if isinstance(value, int) else list(value) for key, value in getattr(fresh, table).items()},
            f"{label}: index {table}",
        )
    if kept.names is not None:
        fresh.named(None)
        case.assertEqual(kept.names, fresh.names, f"{label}: index names")
        # every name two nodes carry is known to be repeated
        case.assertLessEqual(fresh.repeated, kept.repeated, f"{label}: repeated names")


def _assert_lookups_are_a_scan(case, rig, label) -> None:
    """the index the rig keeps is a build's, and every node's parent and
    children, and the rig's indices, are the ones a scan finds"""
    _assert_index_is_a_build(case, rig, label)
    first = {}
    for i, member in enumerate(rig.list):
        first.setdefault(member.uuid, i)
    case.assertEqual(rig.indices.tolist(), [first[n.uuid] for n in rig.list], label)
    for member in rig.list:
        truth = _parent_by_scan(member)
        if isinstance(truth, TransformData):
            case.assertIs(member.get_parent(), truth, f"{label}: parent of {member.name}")
        else:
            case.assertEqual(member.get_parent(), truth, f"{label}: parent of {member.name}")
        case.assertEqual(
            [id(n) for n in member.get_children()],
            [id(n) for n in _children_by_scan(member)],
            f"{label}: children of {member.name}",
        )


def _parent_by_scan(node):
    """get_parent's rule, by a scan of every node of the hierarchy: the first
    node carrying the parent_uuid, the uuid itself when none does"""
    rig, ref = node._hierarchy, node.parent_uuid
    if rig is None or ref is None:
        return None
    return next((n for n in rig.list if n.uuid == ref), ref)


def _first_named_by_scan(rig, name):
    return next((n for n in rig.list if n.name == name), None)


def _worlds_by_scan(rig, made: dict) -> dict:
    """id(node) -> its world matrix, worked out from each node's channels and
    the parents a scan finds; None for a node whose parents lead back to one.
    ``made`` keeps each local matrix worked out, by the values it is made of"""
    parents = {id(n): _parent_by_scan(n) for n in rig.list}
    worlds  = {}
    for node in rig.list:
        chain, seen, up = [], set(), node
        while isinstance(up, TransformData) and id(up) not in worlds:
            if id(up) in seen:
                for member in chain:
                    worlds[id(member)] = None
                break
            seen.add(id(up))
            chain.append(up)
            up = parents[id(up)]

        for member in reversed(chain):
            if id(member) in worlds:
                continue
            parent = parents[id(member)]
            above  = worlds[id(parent)] if isinstance(parent, TransformData) else np.eye(4)
            if above is None:
                worlds[id(member)] = None
                continue
            compensated = member.segment_scale_compensate and isinstance(parent, TransformData)
            key = tuple(
                np.concatenate(
                    [
                        member.scale,
                        member.rotate_axis,
                        member.rotate,
                        member.joint_orient,
                        member.translate,
                        parent.scale if compensated else (),
                    ]
                ).tolist()
            ) + (member.rotate_order,)
            local = made.get(key)
            if local is None:
                IS = np.eye(4)
                if compensated:
                    IS[:3, :3] /= parent.scale
                local = made[key] = (
                    member.scale_matrix
                    @ member.rotate_axis_matrix
                    @ member.rotate_matrix
                    @ member.joint_orient_matrix
                    @ IS
                    @ member.translate_matrix
                )
            worlds[id(member)] = local @ above
    return worlds


# names the fuzz draws from: shared, namespaced, and one shaped like a uuid
_FUZZ_NAMES = ("a", "b", "c", "ns:a", "ns:b", "0F0F0F0F-0000-4000-8000-0000000000AA")


def _fuzz_rig_edits(case, seed: int, edits: int, size: int = 16) -> dict:
    """``edits`` random edits of every kind on one rig: parents by uuid, of
    nodes the rig holds, of none, and loops, written or joined; names shared,
    renamed, given up and taken, looked up through set_parent. After each,
    every lookup the rig answers, through the index it keeps patched or built
    again, equals a scan of the rig, a node looked up by name is the first of
    that name, every node in the rig holds a parent_uuid that is a uuid or
    None, and every world matrix, most of them cached before the edit, is
    the one its channels make. Returns how often each edit ran"""
    from cgmath.hierarchy.hierarchy import _held_node

    rng       = random.Random(seed)
    made      = itertools.count()
    rig       = HierarchyData()
    elsewhere = HierarchyData()

    def name():
        return rng.choice(_FUZZ_NAMES) if rng.random() < 0.5 else f"n{next(made)}"

    def ref():
        roll = rng.random()
        if not len(rig) or roll < 0.2:
            return None
        if roll < 0.85:
            return rng.choice(rig.list).uuid
        return generate_uuid()  # no node carries it

    def node():
        return TransformData(
            name(),
            uuid=generate_uuid(),
            parent_uuid=ref(),
            translate=[rng.uniform(-1.0, 1.0) for _ in range(3)],
            rotate=[rng.uniform(-90.0, 90.0) for _ in range(3)],
            scale=[rng.uniform(0.7, 1.4) for _ in range(3)],
            rotate_order=rng.randrange(6),
            segment_scale_compensate=rng.random() < 0.3,
        )

    def batch():
        # nodes parented among themselves, in any order, and to the rig
        nodes = [node() for _ in range(rng.randint(2, 5))]
        for added in nodes:
            if rng.random() < 0.5:
                added.parent_uuid = rng.choice(nodes).uuid
        rng.shuffle(nodes)
        return nodes

    def pick():
        return rng.choice(rig.list)

    def swap():
        i, j = rng.randrange(len(rig)), rng.randrange(len(rig))
        rig[i], rig[j] = rig[j], rig[i]

    def view():
        return TransformList(rng.sample(rig.list, rng.randint(1, min(4, len(rig)))))

    def rename_view():
        rng.choice(
            (lambda v: v.add_suffix("_L"), lambda v: v.add_prefix("ns:"), lambda v: v.strip_namespace())
        )(view())

    def renames():
        # renames in a row, nothing read between, after a lookup by name
        # built the name table or a uuid written retired the index: a name
        # given up and taken by another node now and then, unique names read
        # now and then, ahead of the first rename too
        roll = rng.random()
        if roll < 0.3:
            pick().uuid = generate_uuid()
        elif roll < 0.8 and rig._kept_index() is not None:
            rig._kept_index().named(pick().name)
        if rng.random() < 0.5:
            rig.unique_name
        for _ in range(rng.randint(2, 6)):
            if rng.random() < 0.5:
                given, taker = pick(), pick()
                taken      = given.name
                given.name = f"n{next(made)}"
                taker.name = taken
            else:
                pick().name = name()
            if rng.random() < 0.3:
                rig.unique_name

    def inserts():
        # inserts ahead of the end in a row, nothing read between: the first
        # shifts the index the last read fetched, the next ones retire it
        for _ in range(rng.randint(2, 5)):
            rig.insert(rng.randint(-len(rig), len(rig) - 1), node())

    def loaded():
        # a loader's loop, nothing read between: nodes appended, then named
        # and parented, after a uuid written retires the index or a rename
        if rng.random() < 0.5:
            pick().uuid = generate_uuid()
        if rng.random() < 0.5:
            pick().name = name()
        for _ in range(rng.randint(1, 4)):
            added = node()
            rig.append(added)
            added.name        = name()
            added.parent_uuid = ref()

    def moved_to_the_end():
        # remove() and append back, or append(pop()), in a row: each takes
        # a node out, which retires the index an append built, if any
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.5:
                moved = pick()
                rig.remove(moved)
                rig.append(moved)
            else:
                rig.append(rig.pop(rng.randrange(len(rig))))

    def target():
        roll = rng.random()
        if roll < 0.2:
            return None
        if roll < 0.4:
            return pick()
        return pick().uuid if roll < 0.7 else pick().name

    def write_world():
        angles   = np.radians([rng.uniform(-180.0, 180.0) for _ in range(3)])
        W        = euler_to_matrix(angles[None], rng.randrange(6))[0]
        W[3, :3] = [rng.uniform(-3.0, 3.0) for _ in range(3)]
        pick().world_matrix = W

    def read():
        for _ in range(rng.randint(1, 6)):
            pick().world_matrix

    def loop():
        # a parent below the node, written in place: no set_parent checks it
        a, b = pick(), pick()
        a.parent_uuid = b.uuid
        b.parent_uuid = a.uuid

    def assigned():
        # item assignments in a row, nothing else editing the list between:
        # a node's copy put back in its place, a node no rig owns naming a
        # parent the rig holds, or not, or a uuid written away, one carrying
        # no uuid; between them now and then a uuid written on a node the rig
        # lists without owning it, over another or on one carrying none.
        # Each keeps its parent only when a node of the rig carries it once
        # it is in. The nodes listed, not owned, are then appended back
        gone = []
        for _ in range(rng.randint(2, 6)):
            at, roll = rng.randrange(len(rig)), rng.random()
            if roll < 0.3:
                put = rig.list[at].copy()
            else:
                parent = rng.choice(gone) if gone and roll < 0.45 else ref()
                put    = TransformData(name(), uuid=None if roll > 0.85 else generate_uuid(), parent_uuid=parent)
            wanted  = put.parent_uuid
            rig[at] = put
            held    = wanted is None or any(n.uuid == wanted for n in rig.list)
            case.assertEqual(put.parent_uuid, wanted if held else None, "an item assigned")

            listed = [n for n in rig.list if n._hierarchy is not rig]
            if listed and rng.random() < 0.4:
                changed = rng.choice(listed)
                if changed.uuid is None and rng.random() < 0.5:
                    HierarchyData().append(changed)  # gives it a uuid, listed here still
                else:
                    if changed.uuid is not None:
                        gone.append(changed.uuid)
                    changed.uuid = generate_uuid()

        for at in reversed(range(len(rig))):
            listed = rig.list[at]
            if listed._hierarchy is not rig:
                del rig[at]
                rig.append(listed)

    def listed_uuid():
        # a node listed by an item assignment, no rig's or another rig's,
        # its uuid written while an index a read built is current, then
        # nodes joining under the uuid given up and under the new one: the
        # index goes, and each joins as a scan of the rig says. The node is
        # then appended back
        put = node()
        if rng.random() < 0.5:
            elsewhere.append(put)
        rig[rng.randrange(len(rig))] = put
        rig._rig_index()
        given    = put.uuid
        put.uuid = generate_uuid()
        _assert_index_is_a_build(case, rig, "a uuid written on a node listed")
        for wanted in (given, put.uuid):
            joined = TransformData(name(), parent_uuid=wanted)
            held   = any(n.uuid == wanted for n in rig.list)
            if rng.random() < 0.5:
                rig.append(joined)
            else:
                rig.insert(rng.randint(-len(rig), len(rig)), joined)
            case.assertEqual(joined.parent_uuid, wanted if held else None, "a node joined")
        del rig[next(i for i, n in enumerate(rig.list) if n is put)]
        rig.append(put)

    kinds = {
        "append": (8, lambda: rig.append(node())),
        "insert": (3, lambda: rig.insert(rng.randint(-len(rig) - 2, len(rig) + 2), node())),
        "inserts in a row": (2, inserts),
        "extend": (2, lambda: rig.extend(batch())),
        "pop": (2, lambda: rig.pop(rng.randrange(len(rig)))),
        "del": (1, lambda: rig.__delitem__(rng.randrange(len(rig)))),
        "remove": (1, lambda: rig.remove(pick())),
        "setitem swap": (1, swap),
        "assigned in a row": (2, assigned),
        "a uuid written on a node listed": (1, listed_uuid),
        "reverse": (1, lambda: rig.reverse()),
        "sort": (1, lambda: rig.sort()),
        "rename": (8, lambda: setattr(pick(), "name", name())),
        "rename a view": (3, rename_view),
        "renames in a row": (5, renames),
        "appended, named, parented": (2, loaded),
        "moved to the end": (1, moved_to_the_end),
        "uuid": (1, lambda: setattr(pick(), "uuid", generate_uuid())),
        "parent_uuid": (8, lambda: setattr(pick(), "parent_uuid", ref())),
        "a loop written": (1, loop),
        "set_parent local": (3, lambda: pick().set_parent(target(), world_space=False)),
        "set_parent world": (3, lambda: pick().set_parent(target(), world_space=True)),
        "view set_parent": (1, lambda: view().set_parent(target(), world_space=False)),
        "channel": (5, lambda: setattr(pick(), "translate", [rng.uniform(-1, 1) for _ in range(3)])),
        "world_matrix =": (5, write_world),
        "read": (6, read),
        "claimed elsewhere": (1, lambda: HierarchyData().append(pick())),
    }
    labels   = list(kinds)
    weights  = [kinds[k][0] for k in labels]
    ran      = dict.fromkeys(labels, 0)
    matrices = {}

    for _ in range(size):
        rig.append(node())

    for step in range(edits):
        label = rng.choices(labels, weights)[0]
        if not len(rig) or (len(rig) < size // 2 and label in ("pop", "del", "remove")):
            label = "append"
        elif len(rig) > 2 * size and label in ("append", "insert", "extend", "inserts in a row", "appended, named, parented"):
            label = "pop"
        ran[label] += 1

        # a parent the node's own branch holds raises before anything moves;
        # a world matrix read through a loop of parents raises on the way
        try:
            kinds[label][1]()
        except (ValueError, RecursionError):
            pass

        what = f"edit {step} ({label})"
        _assert_index_is_a_build(case, rig, what)
        for member in rig.list:
            case.assertTrue(member.parent_uuid is None or validate_uuid(member.parent_uuid), what)
            parent = member.get_parent()
            truth  = _parent_by_scan(member)
            if isinstance(truth, TransformData):
                case.assertIs(parent, truth, f"{what}: parent of {member.name}")
            else:
                case.assertEqual(parent, truth, f"{what}: parent of {member.name}")
            case.assertEqual(
                [id(n) for n in member.get_children()],
                [id(n) for n in _children_by_scan(member)],
                f"{what}: children of {member.name}",
            )

        # a node looked up by name, through the name table once a lookup has
        # built it, is the first node carrying it
        if rig._kept_index() is not None and rig._kept_index().names is not None:
            for wanted in {n.name for n in rig.list} | {"a", "nobody"}:
                truth = _first_named_by_scan(rig, wanted)
                if truth is None:
                    with case.assertRaises(ValueError, msg=what):
                        _held_node(rig, wanted)
                elif not validate_uuid(wanted):
                    case.assertIs(_held_node(rig, wanted), truth, f"{what}: named {wanted}")

        first, last = {}, {}
        for i, member in enumerate(rig.list):
            first.setdefault(member.uuid, i)
            last[member.uuid] = i
        parents = [_parent_by_scan(n) for n in rig.list]
        case.assertEqual(rig.indices.tolist(), [first[n.uuid] for n in rig.list], what)
        case.assertEqual(
            rig.get_parents().tolist(),
            [first[p.uuid] if isinstance(p, TransformData) else -1 for p in parents],
            what,
        )

        # the parent rows the rig keeps with its index, which a clip edit
        # reads: a patch has to drop them
        case.assertEqual(
            rig._parent_rows().tolist(),
            [last[p.uuid] if isinstance(p, TransformData) else -1 for p in parents],
            what,
        )
        case.assertEqual(
            rig.unique_name.tolist(),
            [all(m.name != n.name or m.uuid == n.uuid for m in rig.list) for n in rig.list],
            what,
        )

        # most worlds were cached before the edit: each read is the one the
        # channels make, or none for a node whose parents lead back to it
        worlds = _worlds_by_scan(rig, matrices)
        for member in rig.list:
            truth = worlds[id(member)]
            if truth is None:
                with case.assertRaises(RecursionError, msg=what):
                    member.world_matrix
                continue
            world = member.world_matrix
            if (np.abs(world - truth) > 1e-9 * (1.0 + np.abs(truth))).any():
                np.testing.assert_allclose(
                    world,
                    truth,
                    rtol    = 1e-9,
                    atol    = 1e-9,
                    err_msg = f"{what}: world of {member.name}",
                )
    return ran


def _insert_early(rig) -> None:
    rig.insert(1, TransformData("early", parent_uuid=rig["head"].uuid, translate=(1, 0, 0)))


def _extend_a_batch(rig) -> None:
    x0 = TransformData("x0", uuid=generate_uuid(), parent_uuid=rig["twin"].uuid)
    x1 = TransformData("x1", parent_uuid=x0.uuid)
    rig.extend([x1, x0])  # a child ahead of its parent


class TestRigIndex(unittest.TestCase):
    """get_children reads the index a hierarchy keeps; every edit that can
    give a node other children has to patch it or retire it"""

    EDITS = {
        "append": lambda rig: rig.append(
            TransformData("extra", parent_uuid=rig["neck"].uuid, translate=(0, 1, 0))
        ),
        "insert": _insert_early,
        "extend": _extend_a_batch,
        "pop": lambda rig: rig.pop(rig.index("spine")),
        "del": lambda rig: rig.__delitem__(rig.index("neck")),
        "remove": lambda rig: rig.remove(rig["head"]),
        "setitem swap": _swap_twins,
        "reverse": lambda rig: rig.reverse(),
        "sort": lambda rig: rig.sort(reverse=True),
        "name taken": lambda rig: setattr(rig["root"], "name", "twin"),
        "name dropped": lambda rig: setattr(rig["spine"], "name", "back"),
        "add_prefix": lambda rig: rig["twin"].add_prefix("L_"),
        "add_suffix": lambda rig: rig["head"].add_suffix("_x"),
        "uuid": lambda rig: setattr(rig["spine"], "uuid", generate_uuid()),
        "parent_uuid": lambda rig: setattr(rig["toe"], "parent_uuid", rig["head"].uuid),
        "parent_uuid none": lambda rig: setattr(rig["spine"], "parent_uuid", None),
        "set_parent local": lambda rig: rig["twin"].set_parent("head", world_space=False),
        "set_parent world": lambda rig: rig["neck"].set_parent("root", world_space=True),
        "claimed elsewhere": lambda rig: HierarchyData().append(rig["neck"]),
        "list replaced": lambda rig: setattr(rig, "list", rig.list[::-1]),
        "list popped": lambda rig: rig.list.pop(rig.index("tip")),
    }

    def _assert_current(self, rig, label):
        """children and branches match a scan of the hierarchy, so do the
        parents and unique names read through the index, and a channel write
        on any node reaches its whole branch: every cached world matrix then
        equals one computed from scratch"""
        for node in rig:
            self.assertEqual(
                [id(n) for n in node.get_children()],
                [id(n) for n in _children_by_scan(node)],
                f"{label}: children of {node.name}",
            )
            self.assertEqual(
                [id(n) for n in node.get_branch()],
                [id(n) for n in _branch_by_scan(node)],
                f"{label}: branch of {node.name}",
            )

        parents = [node.get_parent() for node in rig]
        self.assertEqual(
            rig.get_parents().tolist(),
            [rig.index(p) if isinstance(p, TransformData) else -1 for p in parents],
            f"{label}: parents",
        )
        self.assertEqual(
            rig.unique_name.tolist(),
            [node.unique_name for node in rig],
            f"{label}: unique names",
        )

        for i in range(len(rig)):
            rig.world_matrix  # every cache filled
            rig[i].translate = rig[i].translate + (0.25, -0.5, 0.125)
            np.testing.assert_allclose(
                rig.world_matrix,
                rig.copy().world_matrix,
                atol    = 1e-12,
                err_msg = f"{label}: after a write on {rig[i].name}",
            )

    def test_every_edit_patches_or_retires_the_index(self):
        for label, edit in self.EDITS.items():
            for named in (False, True):
                with self.subTest(label, names_built=named):
                    rig = _mixed_rig()
                    self._assert_current(rig, label + " (before)")

                    # the index built just ahead of the edit, its name table
                    # too or not, which the edit then has to patch or retire
                    rig._rig_index()
                    if named:
                        rig._kept_index().named("twin")
                    edit(rig)
                    _assert_index_is_a_build(self, rig, label)

                    # an edit that moves a node to another parent leaves the
                    # cached world matrices it moved stale: that is not the
                    # map's to fix, so they are dropped here, and what is
                    # checked is that every later write reaches the branch
                    # the edit made
                    for node in rig:
                        node._transform_changed()
                    self._assert_current(rig, label)

    def test_no_edit_leaves_a_stale_world_matrix(self):
        # every cached world matrix filled, then the edit: a world matrix read
        # after it is the one the rig now makes, though no branch was cleared.
        # a node the edit moved used to keep the world of its old parent
        for label, edit in self.EDITS.items():
            with self.subTest(label):
                rig = _mixed_rig()
                rig.world_matrix
                edit(rig)
                np.testing.assert_allclose(
                    rig.world_matrix,
                    rig.copy().world_matrix,
                    atol    = 1e-12,
                    err_msg = label,
                )

    def test_a_sort_that_raises_retires_the_index(self):
        # list.sort can leave the list part sorted when a comparison raises
        b   = TransformData("b", uuid=generate_uuid(), translate=(0.0, 1.0, 0.0))
        a   = TransformData("a", parent_uuid=b.uuid, translate=(0.0, 1.0, 0.0))
        rig = HierarchyData([b, a, TransformData("c"), TransformData("x")])
        rig["x"].__dict__["name"] = None  # a name sort() cannot compare
        rig._rig_index()
        a.world_matrix

        with self.assertRaises(TypeError):
            rig.sort()

        self.assertFalse(rig._rig_index_is_current())
        self.assertEqual([id(n) for n in b.get_children()], [id(a)])
        self.assertEqual(rig.indices.tolist(), list(range(len(rig))))
        b.translate = (0.0, 100.0, 0.0)
        np.testing.assert_allclose(a.world_matrix[3, :3], [0.0, 101.0, 0.0])

    def test_popping_leaves_builds_no_index(self):
        # every pop retires the index, so a pop that read it would rebuild it
        # on each call: clear() on a rig of N nodes cost N builds
        from cgmath.hierarchy.hierarchy import _RigIndex

        build  = _RigIndex.build.__func__
        builds = []

        def counted(cls, hierarchy):
            builds.append(1)
            return build(cls, hierarchy)

        rig = _mixed_rig()
        for i in range(30):
            rig.append(TransformData(f"leaf{i}", parent_uuid=rig["head"].uuid))
        with mock.patch.object(_RigIndex, "build", classmethod(counted)):
            for _ in range(30):
                rig.pop()
            self.assertEqual(len(builds), 0)
            rig.clear()
        self.assertEqual(len(rig), 0)
        self.assertLessEqual(len(builds), 1)

    def test_a_pop_orphans_the_children_it_finds_by_scan(self):
        # the index retired by an edit, the children found by a scan
        rig             = _mixed_rig()
        rig["toe"].uuid = generate_uuid()  # retires the index
        self.assertFalse(rig._rig_index_is_current())
        twin = rig.index("twin")
        kids = _children_by_scan(rig[twin])
        rig.world_matrix
        rig.pop(twin)
        self.assertTrue(kids)
        self.assertTrue(all(n.parent_uuid is None for n in kids))
        np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)

    def test_a_rename_moves_no_node(self):
        # a node holds its parent's uuid: a rename, onto a name another node
        # carries or one a node of the rig used to carry, moves no child, and
        # every cached world matrix stands
        rig  = _mixed_rig()
        hand = rig["hand"]
        rig.world_matrix
        built  = rig._rig_index()
        before = [n.get_parent() for n in rig]

        rig["root"].name = "twin"
        rig["twin"].name = "hand"
        hand.name        = "root"
        self.assertEqual([n.get_parent() for n in rig], before)
        self.assertIs(hand.get_parent(), rig.list[4])
        self.assertIs(rig._kept_index(), built)
        self.assertTrue(all(n._cached_world() is not None for n in rig))
        np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)

    def test_the_index_is_kept_until_an_edit(self):
        rig                       = _mixed_rig()
        built                     = rig._rig_index()
        rig["spine"].rotate       = (10.0, 20.0, 30.0)
        rig["spine"].rotate_order = 3
        rig["spine"].get_branch()
        self.assertIs(rig._rig_index(), built)

        # a rename, a node added and a parent changed patch it in place
        rig["neck"].name = "collar"
        rig.append(TransformData("extra", parent_uuid=rig["collar"].uuid))
        rig["toe"].parent_uuid = rig["tip"].uuid
        self.assertIs(rig._rig_index(), built)
        _assert_index_is_a_build(self, rig, "patched")

        # an edit that reorders the nodes retires it
        rig.sort()
        self.assertIsNot(rig._rig_index(), built)

        # a strip that renames nothing keeps it
        built = rig._rig_index()
        rig.strip_namespace()
        self.assertIs(rig._rig_index(), built)

    def test_a_rename_through_a_view_reaches_the_name_table(self):
        # a view renames its nodes one by one: the name table a lookup by
        # name built follows them, so set_parent finds the new names, and
        # every child keeps its parent
        rig                  = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
        rig["c"].parent_uuid = rig["a"].uuid
        rig._rig_index()
        rig["c"].set_parent("b", world_space=False)  # builds the name table
        self.assertIsNotNone(rig._kept_index().names)

        TransformList([rig["a"], rig["b"]]).add_suffix("_L")
        self.assertEqual(rig.name, ["a_L", "b_L", "c"])
        self.assertEqual(rig.get_parents().tolist(), [-1, -1, 1])
        _assert_index_is_a_build(self, rig, "a view renamed")
        rig["c"].set_parent("a_L", world_space=False)
        self.assertEqual(rig["c"].parent_name, "a_L")
        with self.assertRaises(ValueError):
            rig["c"].set_parent("b", world_space=False)
        self.assertEqual(rig.unique_name.tolist(), [True, True, True])

    def test_a_channel_write_does_not_scan_the_hierarchy_per_node(self):
        # get_children used to scan every node, once for each node of the
        # branch, so a write on a root cost the size of the rig squared
        uuids = [generate_uuid() for _ in range(60)]
        rig = HierarchyData(
            [
                TransformData(f"j{i}", uuid=u, parent_uuid=uuids[i // 2] if i else None)
                for i, u in enumerate(uuids)
            ]
        )
        rig.list = _Walked(rig.list)
        rig[0].world_matrix
        rig.list.walks = 0

        # the first write finds the index the rig's construction retired:
        # one scan tells the root has children, one pass builds the index,
        # and the other nine writes read it
        for i in range(10):
            rig[0].translate = (float(i), 0.0, 0.0)
        self.assertLessEqual(rig.list.walks, 2)

    def test_a_leaf_written_after_each_node_added_leaves_the_index_unbuilt(self):
        # with no index kept, an append to a small rig has none to patch: a
        # write on the node just added learns from one scan that it has no
        # children, where building the index would cost several scans on
        # every write
        from cgmath.hierarchy.hierarchy import _RigIndex

        rig    = _mixed_rig()
        head   = rig["head"]
        build  = _RigIndex.build.__func__
        builds = []

        def counted(cls, hierarchy):
            builds.append(1)
            return build(cls, hierarchy)

        leaves = []
        with mock.patch.object(_RigIndex, "build", classmethod(counted)):
            for i in range(20):
                leaf = TransformData(f"leaf{i}", parent_uuid=head.uuid)
                rig.append(leaf)
                leaf.translate = (float(i), 1.0, 0.0)
                leaves.append(leaf)
            self.assertEqual(len(builds), 0)

            # a write on their parent builds it once and reaches them all
            for leaf in leaves:
                leaf.world_matrix
            head.translate = head.translate + 1.0
            self.assertEqual(len(builds), 1)

        self.assertTrue(all(leaf._world_matrix is None for leaf in leaves))

    def test_an_append_to_a_large_rig_builds_the_index_once(self):
        # a loop appending, naming and parenting nodes, nothing read: with no
        # index kept, each append scanned the list for the uuid, 1.2 times
        # 1.0.4's loop at 2000 nodes. From _BUILT_FOR_APPEND nodes on, the
        # first append builds the index and the others patch it
        from cgmath.hierarchy.hierarchy import _BUILT_FOR_APPEND

        def loaded(count):
            rig      = HierarchyData([TransformData(f"r{i}") for i in range(_BUILT_FOR_APPEND)])
            rig.list = _Walked(rig.list)

            def edit():
                above = rig.list[-1]
                for i in range(count):
                    node = TransformData("tmp", translate=(0.0, 1.0, 0.0))
                    rig.append(node)
                    node.name        = f"j{i}"
                    node.parent_uuid = above.uuid
                    above            = node

            spent          = self._counted(edit)
            spent["walks"] = rig.list.walks
            self.assertEqual(rig.list[-1].world_matrix[3, 1], float(count))
            _assert_lookups_are_a_scan(self, rig, f"{count} appended")
            return spent

        small, large = loaded(40), loaded(160)
        self.assertEqual(small, large)
        self.assertEqual(large["builds"], 1)

    def test_appends_with_an_edit_retiring_the_index_between_build_a_few_times(self):
        # an edit between appends -- a remove, a pop, a del, a uuid written --
        # retires the index the append to a large rig built, unused: a build
        # at every append ran these loops 2.7 to 8.7 times 938ae44's over
        # 2000 nodes. An append finding the index it built last retired
        # scans for 1, 2, 4, ... appends before building again: the builds
        # grow as the log of the loop, and the lookups after are a scan's
        from cgmath.hierarchy.hierarchy import _BUILT_FOR_APPEND

        def moved(rig, i):
            node = rig.list[0]
            rig.remove(node)
            rig.append(node)

        def renewed(rig, i):
            node = TransformData(f"k{i}")
            rig.append(node)
            node.uuid = generate_uuid()

        loops = {
            "remove, append": moved,
            "append(pop(0))": lambda rig, i: rig.append(rig.pop(0)),
            "append, uuid =": renewed,
            "append, del [0]": lambda rig, i: (rig.append(TransformData(f"k{i}")), rig.__delitem__(0)),
            # an insert ahead of the end retires an index no lookup has read:
            # an append's own build is no lookup. Taken for one, the insert
            # shifted it, the next append found it current and started the
            # back-off over, and every second append built: 32 builds over 64
            # appends, 1.6 to 2.4 times 938ae44's and 1.0.4's loops
            "insert(0), append": lambda rig, i: (
                rig.insert(0, TransformData(f"a{i}")),
                rig.append(TransformData(f"b{i}")),
            ),
            "insert(middle), append": lambda rig, i: (
                rig.insert(len(rig) // 2, TransformData(f"a{i}")),
                rig.append(TransformData(f"b{i}")),
            ),
        }

        def builds(count, loop):
            # flat, as a pop orphaning children walks their branches, which
            # builds the index for the walk
            rig = HierarchyData([TransformData(f"j{i}") for i in range(_BUILT_FOR_APPEND + 8)])
            rig.world_matrix
            spent = self._counted(lambda: [loop(rig, i) for i in range(count)])
            _assert_lookups_are_a_scan(self, rig, f"{count} appends")
            return spent["builds"]

        for label, loop in loops.items():
            with self.subTest(label):
                small, large = builds(64, loop), builds(256, loop)
                self.assertLessEqual(small, 8)
                self.assertLessEqual(large, small + 2)

    def test_item_assignments_make_no_pass_over_the_list(self):
        # reverse(), each node put back in its place, a node no rig owns
        # and with no parent put in: a node the rig lists already, or one
        # naming no parent, has nothing to check. A set of every uuid in the
        # rig made at each assignment ran reverse() over 2000 nodes in
        # 179-184 ms, where 938ae44's ran in 1.8 ms. A node no rig owns
        # naming a parent -- a node's copy put back, a child of the root
        # made with a uuid or with none -- reads the count of each uuid the
        # first such assignment makes, which each later one patches: made
        # at each, the copies put back ran in 187-189 ms over 2000 nodes,
        # against 938ae44's 2.6 ms. So do a slice of one put back, a copy
        # given a new uuid, and children put in after a node with no uuid:
        # counts kept only while no uuid was written on any node ran them in
        # 182-283 ms over 2000 nodes, against 938ae44's 2.4-21 ms
        def walks(count, edit):
            rig = HierarchyData([TransformData(f"j{i}") for i in range(count)])
            for above, node in zip(rig.list, rig.list[1:]):
                node.parent_uuid = above.uuid
            rig.list       = _Walked(rig.list)
            rig.list.walks = 0
            edit(rig)
            return rig.list.walks

        def children(made):
            def edit(rig):
                root = rig.list[0]
                for i in range(1, len(rig)):
                    rig[i] = TransformData(f"x{i}", uuid=made(), parent_uuid=root.uuid)
                    self.assertEqual(rig.list[i].parent_uuid, root.uuid)

            return edit

        def copies_put_back(put, renew=False):
            def edit(rig):
                copies = [node.copy() for node in rig.list]
                for i, copied in enumerate(copies):
                    if renew:
                        # a uuid given to a node no rig lists, its child's
                        # copy named after it, retires no count
                        copied.uuid = generate_uuid()
                        if i + 1 < len(copies):
                            copies[i + 1].parent_uuid = copied.uuid
                    put(rig, i, copied)
                self.assertEqual([n.parent_uuid for n in rig.list], [None] + [n.uuid for n in copies[:-1]])

            return edit

        def bare_then_children(rig):
            # a node with no uuid listed: a uuid given to another node, as one
            # is made, retires no count
            root   = rig.list[0]
            rig[1] = TransformData("bare")
            for i in range(2, len(rig)):
                rig[i] = TransformData(f"x{i}", uuid=generate_uuid(), parent_uuid=root.uuid)
                self.assertEqual(rig.list[i].parent_uuid, root.uuid)
            self.assertIsNone(rig.list[1].uuid)

        edits = {
            "reverse()":          lambda rig: rig.reverse(),
            "each node put back": lambda rig: [rig.__setitem__(i, rig.list[i]) for i in range(len(rig))],
            "a loose node at each place": lambda rig: [
                rig.__setitem__(i, TransformData(f"x{i}")) for i in range(len(rig))
            ],
            "each node's copy put back": copies_put_back(lambda rig, i, copied: rig.__setitem__(i, copied)),
            "each copy put back, a slice": copies_put_back(
                lambda rig, i, copied: rig.__setitem__(slice(i, i + 1), [copied])
            ),
            "each copy given a uuid, put back": copies_put_back(
                lambda rig, i, copied: rig.__setitem__(i, copied), renew=True
            ),
            "a child of the root, no uuid": children(lambda: None),
            "a child of the root, a uuid": children(generate_uuid),
            "a node with no uuid, then children of the root": bare_then_children,
        }
        for label, edit in edits.items():
            with self.subTest(label):
                self.assertEqual(walks(40, edit), walks(160, edit))

    def test_an_item_assignment_reads_the_uuids_the_list_holds_then(self):
        # the uuid counts one assignment keeps for the next follow what
        # comes between: the node each takes out, another node carrying its
        # uuid, a uuid written on a node the rig lists without owning it --
        # over another, or on one carrying none, as another rig's append
        # gives it one. A node no rig owns keeps its parent only when a node
        # of the rig carries it once the node is in
        rig   = HierarchyData([TransformData(f"j{i}") for i in range(8)])
        nodes = list(rig.list)

        def put(i, parent, made=generate_uuid):
            node   = TransformData(f"p{i}", uuid=made(), parent_uuid=parent)
            rig[i] = node
            return node

        # the node taken out: its uuid is held no more
        self.assertEqual(put(1, nodes[0].uuid).parent_uuid, nodes[0].uuid)
        self.assertIsNone(put(2, nodes[1].uuid).parent_uuid)

        # a uuid another node carries too stays held when one is taken out
        twin   = nodes[0].copy()
        rig[3] = twin
        put(3, None)
        self.assertEqual(put(4, nodes[0].uuid).parent_uuid, nodes[0].uuid)

        # a uuid written over another on a node listed, not owned
        loose      = rig.list[4]
        before     = loose.uuid
        loose.uuid = generate_uuid()
        self.assertEqual(put(5, loose.uuid).parent_uuid, loose.uuid)
        self.assertIsNone(put(6, before).parent_uuid)

        # a uuid given to a node listed with none: written, or by an append
        # to another rig, which leaves it listed here
        bare = put(1, nodes[0].uuid, made=lambda: None)
        self.assertIsNone(bare.uuid)
        bare.uuid = generate_uuid()
        self.assertEqual(put(2, bare.uuid).parent_uuid, bare.uuid)
        bare = put(4, nodes[0].uuid, made=lambda: None)
        HierarchyData().append(bare)
        self.assertIs(rig.list[4], bare)
        self.assertEqual(put(7, bare.uuid).parent_uuid, bare.uuid)

    def test_an_item_assignment_reads_the_list_after_other_edits(self):
        # the counts an assignment patched, rather than made, follow the node
        # it took out and the node it put in; and no counts stand past
        # another edit: a del and an append between (the structure count
        # moves, not the length), an append patching an index a read built
        # (the length moves, not the count). A node of another rig assigned
        # keeps its parent there
        rig   = HierarchyData([TransformData(f"j{i}") for i in range(8)])
        nodes = list(rig.list)

        def put(i, parent):
            node   = TransformData(f"p{i}", uuid=generate_uuid(), parent_uuid=parent)
            rig[i] = node
            return node

        put(1, nodes[0].uuid)
        held = put(2, nodes[0].uuid)  # patches the counts, j2 taken out
        self.assertIsNone(put(3, nodes[2].uuid).parent_uuid)
        self.assertEqual(put(4, held.uuid).parent_uuid, held.uuid)

        added = TransformData("added")
        rig.append(added)
        del rig[7]
        self.assertEqual(len(rig), 8)
        self.assertEqual(put(4, added.uuid).parent_uuid, added.uuid)
        self.assertIsNone(put(5, nodes[7].uuid).parent_uuid)

        other             = HierarchyData([TransformData("o0"), TransformData("o1")])
        moved             = other.list[1]
        moved.parent_uuid = other.list[0].uuid
        rig[6]            = moved
        self.assertEqual(moved.parent_uuid, other.list[0].uuid)
        self.assertIs(moved.get_parent(), other.list[0])

        rig   = HierarchyData([TransformData(f"j{i}") for i in range(8)])
        nodes = list(rig.list)
        put(1, nodes[0].uuid)
        rig[1]  = nodes[1]  # put back: every node listed is the rig's own
        version = rig._index_version
        nodes[0].get_children()  # builds the index, which the append patches
        added = TransformData("added")
        rig.append(added)
        self.assertEqual(rig._index_version, version)
        self.assertEqual(put(2, added.uuid).parent_uuid, added.uuid)

        # a slice assigned patches the counts too: the nodes it takes out are
        # held no more, the nodes it puts in are
        rig   = HierarchyData([TransformData(f"j{i}") for i in range(8)])
        nodes = list(rig.list)

        def sliced(i, parent):
            node           = TransformData(f"s{i}", uuid=generate_uuid(), parent_uuid=parent)
            rig[i : i + 1] = [node]
            return node

        sliced(1, nodes[0].uuid)
        held = sliced(2, nodes[0].uuid)  # patches the counts, j2 taken out
        self.assertIsNone(sliced(3, nodes[2].uuid).parent_uuid)
        self.assertEqual(sliced(4, held.uuid).parent_uuid, held.uuid)
        self.assertEqual(put(5, held.uuid).parent_uuid, held.uuid)
        self.assertIsNone(sliced(6, nodes[5].uuid).parent_uuid)
        rig[7:8] = [moved]  # a node of another rig keeps its parent there
        self.assertEqual(moved.parent_uuid, other.list[0].uuid)

    def test_a_uuid_written_on_a_node_listed_not_owned_retires_the_index(self):
        # a node an item assignment lists, which the rig does not own, tells
        # the rig of a uuid written on it: the index a read built goes, and
        # the world matrices cached under the node. Kept, as 938ae44 kept
        # it, the index holds the uuid given up and not the new one: a node
        # joining under the uuid given up would keep it, one joining under
        # the new one become a root though the node is listed, and a node
        # carrying the new one be taken; a world kept would still sit under
        # the node. A slice lists it as an item does
        for size, owner, sliced in itertools.product((8, 80), ("no rig", "another rig"), (False, True)):
            with self.subTest(size=size, owner=owner, sliced=sliced):
                rig   = HierarchyData([TransformData(f"j{i}") for i in range(size)])
                x     = TransformData("x", uuid=generate_uuid(), translate=(0, 7, 0))
                other = HierarchyData([x] if owner == "another rig" else [])
                if sliced:
                    rig[5:6] = [x]
                else:
                    rig[5] = x
                below             = rig.list[-1]
                below.parent_uuid = x.uuid  # the rig's own node, under x
                np.testing.assert_allclose(below.world_matrix[3, :3], [0, 7, 0])
                rig.list[0].get_children()  # a read builds the index
                self.assertIsNotNone(rig._kept_index())

                before = x.uuid
                x.uuid = generate_uuid()
                self.assertIsNone(rig._kept_index())
                np.testing.assert_allclose(below.world_matrix, below.matrix)
                gone = TransformData("gone", parent_uuid=before)
                rig.append(gone)
                self.assertIsNone(gone.parent_uuid)
                for joined in (TransformData("a"), TransformData("b")):
                    joined.parent_uuid = x.uuid
                    if joined.name == "a":
                        rig.append(joined)
                    else:
                        rig.insert(3, joined)
                    self.assertEqual(joined.parent_uuid, x.uuid)
                    self.assertIs(joined.get_parent(), x)
                with self.assertRaisesRegex(RuntimeError, "already"):
                    rig.append(TransformData("twin", uuid=x.uuid))
                self.assertIs(x._hierarchy, other if owner == "another rig" else None)
                _assert_index_is_a_build(self, rig, f"{size}, {owner}, sliced={sliced}")

        # each rig listing the node hears of it; a rig is held weakly, so the
        # node keeps none alive, and a uuid written once a rig is gone
        # retires the others alone
        x    = TransformData("x", uuid=generate_uuid())
        rigs = [HierarchyData([TransformData("j0"), TransformData("j1")]) for _ in range(3)]
        for rig in rigs:
            rig[1] = x
            rig._rig_index()
        x.uuid = generate_uuid()
        self.assertEqual([rig._kept_index() for rig in rigs], [None] * 3)

        held = weakref.ref(rigs[0])
        del rig, rigs[0]
        gc.collect()
        self.assertIsNone(held())
        for rig in rigs:
            rig._rig_index()
        x.uuid = generate_uuid()
        self.assertEqual([rig._kept_index() for rig in rigs], [None] * 2)

        # a node the rig owns listed twice, then popped or taken by another
        # rig, stays listed in the other place, which no longer owns it: a
        # uuid written on it reaches the rig all the same
        for how in ("pop", "claimed"):
            with self.subTest(how):
                rig    = HierarchyData([TransformData(f"j{i}") for i in range(8)])
                node   = rig.list[5]
                rig[2] = node
                if how == "pop":
                    rig.pop(5)
                else:
                    other = HierarchyData()
                    other.append(node)
                self.assertEqual(sum(n is node for n in rig.list), 1)
                self.assertIsNot(node._hierarchy, rig)
                rig._rig_index()
                node.uuid = generate_uuid()
                self.assertIsNone(rig._kept_index())
                child = TransformData("c", parent_uuid=node.uuid)
                rig.append(child)
                self.assertEqual(child.parent_uuid, node.uuid)
                _assert_index_is_a_build(self, rig, how)

    def test_a_pop_of_a_node_listed_not_owned_leaves_its_owner_alone(self):
        # a rig popping a node an item assignment listed, which another rig
        # owns, takes it out of its own list and no more: the owner keeps
        # the node and the node's children there their parent, a uuid
        # written on it retires the owner's index, and a node joining the
        # owner under the new uuid keeps it. 938ae44 cleared the node's
        # owner and its children's parents in the owner, which kept its
        # index: a node joining under the new uuid became a root there. The
        # rig's own node under it is a root once it is out. A node no rig
        # owns likewise; the rig's index current or not
        for owner, current in itertools.product(("another rig", "no rig"), (False, True)):
            with self.subTest(owner=owner, current=current):
                chain = HierarchyData([TransformData(f"a{i}", translate=(1, 0, 0)) for i in range(4)])
                for above, node in zip(chain.list, chain.list[1:]):
                    node.parent_uuid = above.uuid
                if owner == "another rig":
                    x = chain.list[1]
                else:
                    x = TransformData("x", uuid=generate_uuid(), translate=(2, 0, 0))
                child = chain.list[2]

                rig               = HierarchyData([TransformData(f"r{i}") for i in range(4)])
                rig[2]            = x
                below             = rig.list[3]
                below.parent_uuid = x.uuid
                np.testing.assert_allclose(below.world_matrix[3, :3], [2, 0, 0])
                if current:
                    rig._rig_index()
                self.assertIs(rig.pop(2), x)
                self.assertEqual([n.name for n in rig.list], ["r0", "r1", "r3"])
                np.testing.assert_allclose(below.world_matrix, below.matrix)
                _assert_index_is_a_build(self, rig, f"{owner}, current={current}")

                self.assertIs(x._hierarchy, chain if owner == "another rig" else None)
                self.assertEqual([n.name for n in chain.list], ["a0", "a1", "a2", "a3"])
                self.assertEqual(child.parent_uuid, chain.list[1].uuid)
                self.assertIs(child.get_parent(), chain.list[1])
                np.testing.assert_allclose(child.world_matrix[3, :3], [3, 0, 0])
                if owner == "no rig":
                    continue

                chain._rig_index()
                before = x.uuid
                x.uuid = generate_uuid()
                self.assertIsNone(chain._kept_index())
                self.assertFalse(chain._holds(before))
                self.assertTrue(chain._holds(x.uuid))
                late = TransformData("late", parent_uuid=x.uuid)
                chain.append(late)
                self.assertEqual(late.parent_uuid, x.uuid)
                self.assertIs(late.get_parent(), x)
                _assert_index_is_a_build(self, chain, f"{owner}, current={current}")

        # listed twice, owned elsewhere: popped from one place, it stays
        # listed in the other, which hears of a uuid written on it
        chain  = HierarchyData([TransformData(f"a{i}") for i in range(4)])
        x      = chain.list[1]
        rig    = HierarchyData([TransformData(f"r{i}") for i in range(4)])
        rig[1] = x
        rig[2] = x
        rig.pop(2)
        self.assertEqual(sum(n is x for n in rig.list), 1)
        self.assertIs(x._hierarchy, chain)
        rig._rig_index()
        chain._rig_index()
        x.uuid = generate_uuid()
        self.assertIsNone(rig._kept_index())
        self.assertIsNone(chain._kept_index())

    def test_a_world_below_a_node_listed_not_owned_follows_it(self):
        # the rig's own nodes hung below a node an item assignment lists,
        # which another rig owns or none does, read the world it has now.
        # What moves it -- a channel written on it or above it, an edit of
        # its owner -- walks its owner's branches and counts on its owner
        # alone, so a world cached here under this rig's count stood: 938ae44
        # and 1.0.4 left c and d at their first world after every edit below
        # but the sort. A world cached below such a node holds the node's
        # cached world as well, and stands while the node has it
        def built(owned=True):
            owner = HierarchyData([
                TransformData("p", translate=(10, 0, 0)),
                TransformData("x", translate=(1, 0, 0)),
                TransformData("q", translate=(0, 5, 0)),
            ])
            p, x, q = owner.list
            x.parent_uuid = p.uuid
            if not owned:
                owner.pop(1)
                x.translate = (11, 0, 0)  # where it was, with no parent
            rig    = HierarchyData([TransformData(f"r{i}") for i in range(4)])
            rig[1] = x
            c, d = rig.list[2], rig.list[3]
            d.translate   = (0, 0, 1)
            c.parent_uuid = x.uuid  # the rig's own node, under x
            d.parent_uuid = c.uuid  # and one below it
            np.testing.assert_allclose(d.world_matrix[3, :3], [11, 0, 1])
            np.testing.assert_allclose(c.world_matrix[3, :3], [11, 0, 0])
            return owner, rig, p, x, q, c, d

        moved        = np.eye(4)
        moved[3, :3] = (3, 4, 0)
        edits = {
            "x.translate":           ((12, 0, 0), lambda owner, p, x, q: setattr(x, "translate", (2, 0, 0))),
            "p.translate":           ((21, 0, 0), lambda owner, p, x, q: setattr(p, "translate", (20, 0, 0))),
            "x.world_matrix =":      ((3, 4, 0), lambda owner, p, x, q: setattr(x, "world_matrix", moved)),
            "owner.pop(p)":          ((1, 0, 0), lambda owner, p, x, q: owner.pop(0)),
            "del owner[0]":          ((1, 0, 0), lambda owner, p, x, q: owner.__delitem__(0)),
            "owner[0] = a new node": ((1, 0, 0), lambda owner, p, x, q: owner.__setitem__(0, TransformData("n"))),
            "x.set_parent(q)":       ((1, 5, 0), lambda owner, p, x, q: x.set_parent(q, world_space=False)),
            "owner.sort()":          ((11, 0, 0), lambda owner, p, x, q: owner.sort()),
        }
        for (label, (want, edit)), c_first in itertools.product(edits.items(), (False, True)):
            with self.subTest(label, c_first=c_first):
                owner, rig, p, x, q, c, d = built()
                edit(owner, p, x, q)
                if c_first:
                    np.testing.assert_allclose(c.world_matrix[3, :3], want, atol=1e-12)
                np.testing.assert_allclose(d.world_matrix[3, :3], np.add(want, (0, 0, 1)), atol=1e-12)
                np.testing.assert_allclose(c.world_matrix[3, :3], want, atol=1e-12)
                np.testing.assert_allclose(c.world_matrix, c.matrix @ x.world_matrix, atol=1e-12)

        # a node no rig owns, written
        with self.subTest("no rig, x.translate"):
            owner, rig, p, x, q, c, d = built(owned=False)
            self.assertIsNone(x._hierarchy)
            x.translate = (2, 0, 0)
            np.testing.assert_allclose(d.world_matrix[3, :3], [2, 0, 1])

        # an edit of the rig itself: c hung from the rig's root instead, by a
        # raw write, its index current or not
        for current in (False, True):
            with self.subTest("c.parent_uuid = r0", current=current):
                owner, rig, p, x, q, c, d = built()
                if current:
                    rig._rig_index()
                c.parent_uuid = rig.list[0].uuid
                np.testing.assert_allclose(d.world_matrix[3, :3], [0, 0, 1])
                np.testing.assert_allclose(c.world_matrix[3, :3], [0, 0, 0])

        # d read down from c's cache, which rests on x's: x moved again
        with self.subTest("d read through c, x moved twice"):
            owner, rig, p, x, q, c, d = built()
            x.translate = (2, 0, 0)
            np.testing.assert_allclose(c.world_matrix[3, :3], [12, 0, 0])
            np.testing.assert_allclose(d.world_matrix[3, :3], [12, 0, 1])
            x.translate = (3, 0, 0)
            np.testing.assert_allclose(d.world_matrix[3, :3], [13, 0, 1])

        # a link further up: x's owner lists x's parent, which a third rig
        # owns; that rig moves it
        with self.subTest("a link above x"):
            owner, rig, p, x, q, c, d = built()
            third         = HierarchyData([TransformData("y", translate=(0, 0, 3))])
            y             = third.list[0]
            owner[2]      = y  # q's place
            x.parent_uuid = y.uuid
            np.testing.assert_allclose(d.world_matrix[3, :3], [1, 0, 4])
            y.translate = (0, 0, 7)
            np.testing.assert_allclose(d.world_matrix[3, :3], [1, 0, 8])

        # what nothing moved stays cached, below x too: reading every world
        # of a chain hung below x again computes no matrix, and after x
        # moves each node's once, where reading each from x computed the
        # square of the chain
        owner, rig, p, x, q, c, d = built()
        chain    = HierarchyData([TransformData(f"e{i}", translate=(0, 1, 0)) for i in range(40)])
        chain[0] = x
        for above, node in zip(chain.list, chain.list[1:]):
            node.parent_uuid = above.uuid

        def read():
            for node in chain.list:
                node.world_matrix

        read()
        self.assertEqual(self._counted(read)["matrices"], 0)
        x.translate = (2, 0, 0)
        self.assertEqual(self._counted(read)["matrices"], 40)
        np.testing.assert_allclose(chain.list[-1].world_matrix[3, :3], [12, 39, 0])
        self.assertEqual(self._counted(read)["matrices"], 0)

        # a uuid written on x that a root of the rig names already: the root
        # hangs below x now, and its world cached as a root goes
        made             = generate_uuid()
        late             = rig.list[0]
        late.parent_uuid = made  # names no node yet
        np.testing.assert_allclose(late.world_matrix, late.matrix)
        x.uuid = made
        self.assertIs(late.get_parent(), x)
        np.testing.assert_allclose(late.world_matrix[3, :3], [12, 0, 0])

    def test_a_move_takes_no_node_of_another_rig(self):
        # set_parent moves a node's branch to the end of its rig's list. A
        # node of the branch the rig lists, which another rig owns, has its
        # children looked up in that rig: they stay out of this list, which
        # moves only the nodes it holds. 938ae44 put them in, not owned and
        # not known to list them, so a uuid written on one left the rig's
        # index standing. By a node and by a view, local and world
        for view, world_space in itertools.product((False, True), (False, True)):
            with self.subTest(view=view, world_space=world_space):
                rig           = HierarchyData([TransformData(name) for name in "abc"])
                a, b          = rig.list[0], rig.list[1]
                other         = HierarchyData([TransformData("x"), TransformData("y")])
                x, y          = other.list
                y.parent_uuid = x.uuid
                x.parent_uuid = a.uuid  # a node the other rig does not hold
                rig[2]        = x       # listed here: a's child
                if view:
                    TransformList([a]).set_parent(b, world_space=world_space)
                else:
                    a.set_parent(b, world_space=world_space)

                self.assertEqual([n.name for n in rig.list], ["b", "a", "x"])
                self.assertEqual([n.name for n in other.list], ["x", "y"])
                self.assertIs(y._hierarchy, other)
                self.assertIs(y.get_parent(), x)
                rig._rig_index()
                y.uuid = generate_uuid()
                _assert_index_is_a_build(self, rig, f"view={view}, world_space={world_space}")

    def test_an_append_by_the_scan_moves_the_nodes_naming_it(self):
        # an append with no index to read, of a node naming a parent, whose
        # uuid a listed node names already: that node's cached world goes
        rig   = HierarchyData([TransformData(f"j{i}") for i in range(8)])
        made  = generate_uuid()
        below = TransformData("below", translate=(0, 1, 0))
        rig.append(below)
        below.parent_uuid = made  # names no node yet
        np.testing.assert_allclose(below.world_matrix[3, :3], [0, 1, 0])
        rig.insert(0, TransformData("x"))  # shifts the index the read built
        rig.insert(0, TransformData("y"))  # retires it alone: worlds stand
        self.assertIsNone(rig._kept_index())
        above = TransformData("above", uuid=made, parent_uuid=rig.list[2].uuid, translate=(5, 0, 0))
        rig.append(above)
        self.assertIs(below.get_parent(), above)
        np.testing.assert_allclose(below.world_matrix[3, :3], [5, 1, 0])

    def test_a_node_naming_its_parent_joins_through_the_index(self):
        # appended, or extended one at a time, each node made naming the
        # node before it: from _BUILT_FOR_APPEND nodes on, the first append
        # builds the index and every join reads it, as many passes over the
        # list over 40 nodes as over 160. A join scanning the list at each
        # append, or a batch's join counting every uuid at each extend, ran
        # these loops quadratic: 2000 nodes extended in 126 ms against 27
        from cgmath.hierarchy.hierarchy import _BUILT_FOR_APPEND

        def spent(count, add):
            rig      = HierarchyData([TransformData(f"r{i}") for i in range(_BUILT_FOR_APPEND)])
            rig.list = _Walked(rig.list)

            def edit():
                above = rig.list[-1]
                for i in range(count):
                    node = TransformData(f"j{i}", parent_uuid=above.uuid, translate=(0.0, 1.0, 0.0))
                    add(rig, node)
                    above = node

            counts          = self._counted(edit)
            counts["walks"] = rig.list.walks
            self.assertEqual(rig.list[-1].world_matrix[3, 1], float(count))
            return counts

        adds = {
            "append": lambda rig, node: rig.append(node),
            "extend([node])": lambda rig, node: rig.extend([node]),
        }
        for label, add in adds.items():
            with self.subTest(label):
                small, large = spent(40, add), spent(160, add)
                self.assertEqual(small, large)
                self.assertEqual(large["builds"], 1)

    def test_an_append_with_no_index_naming_a_parent_makes_one_pass(self):
        # with no index to read -- a rig under _BUILT_FOR_APPEND nodes, or
        # one whose appends scan while builds go to waste -- the check that
        # the uuid is new and the join read the list in one pass, from the
        # end. A pass for each ran a loop appending a child of the last node
        # and deleting the first at 1.9 to 2.6 times 938ae44's and 1.0.4's
        parents = {
            "the last node":      lambda rig: rig.list[-1].uuid,
            "the first node":     lambda rig: rig.list[0].uuid,
            "no node of the rig": lambda rig: generate_uuid(),
        }
        for label, parent_of in parents.items():
            for fresh in (True, False):
                with self.subTest(label, fresh=fresh):
                    rig            = HierarchyData([TransformData(f"j{i}") for i in range(40)])
                    rig.list       = _Walked(rig.list)
                    rig.list.walks = 0
                    for i in range(10):
                        parent = parent_of(rig)
                        node   = TransformData(f"k{i}", uuid=None if fresh else generate_uuid(), parent_uuid=parent)
                        rig.append(node)
                        self.assertEqual(node.parent_uuid, None if label == "no node of the rig" else parent)
                        self.assertTrue(validate_uuid(node.uuid))
                    self.assertEqual(rig.list.walks, 10)
                    self.assertEqual(len({n.uuid for n in rig.list}), 50)

        # a uuid the rig holds raises, met ahead of the parent or after it,
        # and nothing is added
        rig = HierarchyData([TransformData(f"j{i}") for i in range(8)])
        for parent in (rig.list[0], rig.list[-1]):
            with self.assertRaisesRegex(RuntimeError, "already"):
                rig.append(TransformData("twin", uuid=rig.list[3].uuid, parent_uuid=parent.uuid))
            self.assertEqual(len(rig), 8)

    def test_roots_chained_in_list_order_search_no_rows(self):
        # a chain in use made flat and chained again: each parent written
        # takes the node out of the rows of the roots, the second of them,
        # read in place where a search of the rows, a deque past
        # _DEQUE_ROWS, ran the loop over aba1ce5's on one thread
        import bisect

        def searches(count):
            rig = HierarchyData([TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)])
            for above, node in zip(rig.list, rig.list[1:]):
                node.parent_uuid = above.uuid
            rig.world_matrix
            for node in rig.list:
                node.parent_uuid = None

            found, calls = bisect.bisect_left, []

            def counted(*args, **kwargs):
                calls.append(args)
                return found(*args, **kwargs)

            with mock.patch.object(bisect, "bisect_left", counted):
                for above, node in zip(rig.list, rig.list[1:]):
                    node.parent_uuid = above.uuid
            self.assertIsNotNone(rig._kept_index())  # patched all the way
            _assert_lookups_are_a_scan(self, rig, f"{count} chained again")
            self.assertEqual(rig.list[-1].world_matrix[3, 1], float(count))
            return len(calls)

        self.assertEqual((searches(40), searches(160)), (0, 0))

    def test_parents_read_on_a_rig_just_made_build_the_index_once(self):
        # get_parent, and parent_name, on every node of a rig just made or
        # loaded, which keeps no index: a few scans, then one build read from
        # there on. A scan at each, the rig's size squared over the loop, ran
        # 39 ms over 2000 nodes against 0.64 ms in use (938ae44's 104 ms).
        # An edit retiring the index between lookups leaves each a scan
        from cgmath.hierarchy.hierarchy import _BUILT_FOR_APPEND, _SCANS_BEFORE_BUILD

        def spent(count, read):
            nodes = [TransformData(f"j{i}") for i in range(_BUILT_FOR_APPEND + count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_uuid = above.uuid
            rig.list        = _Walked(rig.list)
            counts          = self._counted(lambda: read(rig))
            counts["walks"] = rig.list.walks
            return counts

        def names(rig):
            self.assertEqual([n.parent_name for n in rig.list[1:]], [n.name for n in rig.list[:-1]])

        small, large = spent(40, names), spent(160, names)
        self.assertEqual(small, large)
        self.assertEqual(large["builds"], 1)
        self.assertLessEqual(large["walks"], _SCANS_BEFORE_BUILD + 2)

        def retired(rig):
            for node in rig.list[1:]:
                node.get_parent()
                rig.list[0].uuid = rig.list[0].uuid  # an edit retiring the index

        self.assertEqual(spent(40, retired)["builds"], 0)

    def test_a_chain_deeper_than_the_recursion_limit(self):
        depth = sys.getrecursionlimit() + 50
        uuids = [generate_uuid() for _ in range(depth)]
        rig = HierarchyData(
            [
                TransformData(
                    f"j{i}",
                    uuid        = u,
                    parent_uuid = uuids[i - 1] if i else None,
                    translate   = (0.0, 1.0, 0.0),
                )
                for i, u in enumerate(uuids)
            ]
        )
        self.assertEqual(len(rig[0].get_branch()), depth)

        # parent first, so each read stops at its parent's cached matrix
        for node in rig:
            node.world_matrix
        rig[0].translate = (2.0, 0.0, 0.0)
        self.assertTrue(all(node._world_matrix is None for node in rig))

        for node in rig:
            node.world_matrix
        np.testing.assert_allclose(rig[-1].world_matrix[3, :3], [2.0, depth - 1.0, 0.0])

    def test_a_cyclic_parent_is_walked_once(self):
        rig                  = HierarchyData([TransformData("a"), TransformData("b")])
        rig["b"].parent_uuid = rig["a"].uuid
        rig["a"].parent_uuid = rig["b"].uuid
        self.assertEqual(rig["a"].get_branch().name, ["a", "b"])
        rig["a"].translate = (1.0, 0.0, 0.0)
        self.assertIsNone(rig["b"]._world_matrix)

        # nor has a world matrix, read by a loop that names the cycle
        with self.assertRaises(RecursionError):
            rig["b"].world_matrix

    def test_an_edit_elsewhere_keeps_a_deep_chain_cached(self):
        # the structure count is the rig's own: a node built, or an edit in
        # another rig, leaves every cached world matrix of this one standing.
        # it was one count for every rig, so each of these left the whole
        # chain to be read again by recursion, past the recursion limit
        depth = max(1500, sys.getrecursionlimit() + 50)
        uuids = [generate_uuid() for _ in range(depth)]
        chain = HierarchyData(
            [
                TransformData(
                    f"j{i}",
                    uuid        = uuids[i],
                    parent_uuid = uuids[i - 1] if i else None,
                    translate   = (0.0, 1.0, 0.0),
                )
                for i in range(depth)
            ]
        )
        for node in chain:
            node.world_matrix
        leaf                 = chain[depth - 1]
        other                = HierarchyData([TransformData("x"), TransformData("w")])
        other[1].parent_uuid = other[0].uuid

        edits = (
            ("a node built", lambda: TransformData("z")),
            ("a rename in another rig", lambda: setattr(other[0], "name", "y")),
            ("a parent_uuid in another rig", lambda: setattr(other[1], "parent_uuid", None)),
            ("a node added to another rig", lambda: other.append(TransformData("v"))),
            ("a node of another rig popped", lambda: other.pop()),
        )
        for label, edit in edits:
            with self.subTest(label):
                edit()
                self.assertTrue(all(node._cached_world() is not None for node in chain))
                self.assertEqual(leaf.world_matrix[3, 1], float(depth))

        # a node added to its own rig, which no node names, moves none of it,
        # and neither does a rename there
        chain.append(TransformData("extra"))
        chain[3].name = "renamed"
        self.assertTrue(all(node._cached_world() is not None for node in chain[:depth]))

        # an edit of its own rig that can move them: read again from the
        # root down, by a loop, every node of the chain cached on the way
        chain.pop()
        self.assertIsNone(leaf._cached_world())
        self.assertEqual(leaf.world_matrix[3, 1], float(depth))
        self.assertTrue(all(node._cached_world() is not None for node in chain[:depth]))

    def test_an_edit_elsewhere_leaves_the_world_setter_keep_test_alone(self):
        # a world matrix written within 1e-8 of the cached one keeps the
        # channels. under a parent scaled 1e-4 the local matrix it solves to
        # is 5e-5 off, so whether the cache is current decides the answer: a
        # node built, or an edit in another rig, used to retire it
        def written(edit):
            parent = TransformData("parent", uuid=generate_uuid(), scale=(1e-4, 1e-4, 1e-4))
            child  = TransformData("child", parent_uuid=parent.uuid, translate=(1.0, 2.0, 3.0))
            HierarchyData([parent, child])
            W = child.world_matrix
            W[3, :3] += 5e-9
            edit()
            child.world_matrix = W
            return np.array(child.translate)

        kept = written(lambda: None)
        np.testing.assert_array_equal(kept, [1.0, 2.0, 3.0])
        edits = (
            ("a node built", lambda: TransformData("z")),
            ("a rename in another rig", lambda: setattr(HierarchyData([TransformData("x")])[0], "name", "y")),
        )
        for label, edit in edits:
            with self.subTest(label):
                np.testing.assert_array_equal(written(edit), kept)

    def test_reads_between_edits_cost_a_chain_not_a_rig(self):
        # ten chains 15 deep under a root. a rename keeps the cached world
        # matrices, so a loop of renames and reads costs what the renames
        # alone cost; an edit that can move a node retires the rig's cached
        # worlds, and a read after it computes its chain once, its parents
        # read from one index of the rig rather than two scans of the rig a
        # level
        from cgmath._tests.test_parent_lookups import _Counted

        depth, chains = 15, 10
        matrix   = TransformData.matrix
        computed = []

        def counted(node):
            computed.append(node)
            return matrix.fget(node)

        def counted_node(name, parent=None):
            # counted strings set past the constructor, which copies to str
            made      = TransformData(name, translate=(0.0, 1.0, 0.0))
            made.name = _Counted(name)
            made.uuid = _Counted(generate_uuid())
            if parent is not None:
                made.parent_uuid = _Counted(str(parent))
            return made

        def rig():
            nodes = [counted_node("root")]
            for k in range(chains):
                for j in range(depth):
                    parent = nodes[0] if j == 0 else nodes[-1]
                    nodes.append(counted_node(f"c{k}_{j}", parent.uuid))
            return HierarchyData(nodes)

        def cost(rig, edit):
            """comparisons and world matrices computed, per leaf"""
            leaves = [rig[k * depth + depth] for k in range(chains)]
            for node in rig:
                node.world_matrix
            _Counted.compared = 0
            del computed[:]
            for leaf in leaves:
                edit(leaf)
            return _Counted.compared / chains, len(computed) / chains

        def rename(node):
            node.name = _Counted(node.name + "_x")

        def rename_and_read(node):
            rename(node)
            np.testing.assert_allclose(node.world_matrix[3, 1], depth + 1)

        def reparent_and_read(node):
            node.parent_uuid = _Counted(str(node.parent_uuid))
            np.testing.assert_allclose(node.world_matrix[3, 1], depth + 1)

        with mock.patch.object(TransformData, "matrix", property(counted, matrix.fset)):
            h     = rig()
            alone = cost(h, rename)
            read  = cost(h, rename_and_read)
            self.assertEqual(read, alone)
            self.assertEqual(read[1], 0)

            # one index build, a few comparisons a node, and the chain
            # computed once; two scans a level made 2160 comparisons
            compared, matrices = cost(h, reparent_and_read)
            self.assertLessEqual(matrices, depth + 1)
            self.assertLess(compared, len(h) / 2)

    @staticmethod
    def _counted(edit) -> dict:
        """index builds and local matrices computed while ``edit`` runs"""
        from cgmath.hierarchy.hierarchy import _RigIndex

        build, matrix = _RigIndex.build.__func__, TransformData.matrix
        counts = {"builds": 0, "matrices": 0}

        def counted_build(cls, hierarchy):
            counts["builds"] += 1
            return build(cls, hierarchy)

        def counted_matrix(node):
            counts["matrices"] += 1
            return matrix.fget(node)

        with mock.patch.object(_RigIndex, "build", classmethod(counted_build)), mock.patch.object(
            TransformData, "matrix", property(counted_matrix, matrix.fset)
        ):
            edit()
        return counts

    def test_a_rig_built_node_by_node_costs_each_node_the_same(self):
        # a node appended, parented to the one before and given a world
        # matrix, one at a time on a rig in use: each append and each parent
        # change retired the index, so each world written built it again and
        # worked out the whole chain above the node. 300 joints took 1 s on
        # one thread where 1.0.4 took 0.1 s. The index is patched instead and
        # the chain above stays cached: no build, no pass over the list, the
        # same few matrices a node however long the chain. An append moves no
        # position of the index, and a parent is asked of a few nodes a node:
        # set_parent's check for a loop climbs no chain above a node that has
        # no children
        from cgmath.hierarchy.hierarchy import _Lookups, _RigIndex

        parents = {
            "parent_uuid": lambda node, above: setattr(node, "parent_uuid", above.uuid),
            "set_parent local, by name": lambda node, above: node.set_parent(above.name, world_space=False),
            "set_parent local, by uuid": lambda node, above: node.set_parent(above.uuid, world_space=False),
            "set_parent world, by node": lambda node, above: node.set_parent(above),
        }

        def built(count, parent):
            rig      = HierarchyData([TransformData("root", translate=(0.0, 1.0, 0.0))])
            rig.list = _Walked(rig.list)
            rig.world_matrix  # a rig in use: its index built, its worlds cached
            rig.list.walks = 0

            def edit():
                above = rig.list[0]
                for i in range(count):
                    node = TransformData(f"j{i}", node_type="joint")
                    rig.append(node)
                    parent(node, above)
                    W                 = np.eye(4)
                    W[3, :3]          = (0.5, i + 2.0, 0.0)
                    node.world_matrix = W
                    above             = node

            shift, index_parent, lookups_parent = _RigIndex._shift, _RigIndex.parent, _Lookups.parent
            asked = {"shifts": 0, "parents": 0}

            def counted_shift(index, at):
                asked["shifts"] += 1
                return shift(index, at)

            def counted_index_parent(index, node):
                asked["parents"] += 1
                return index_parent(index, node)

            def counted_lookups_parent(lookups, node):
                asked["parents"] += 1
                return lookups_parent(lookups, node)

            with mock.patch.object(_RigIndex, "_shift", counted_shift), mock.patch.object(
                _RigIndex, "parent", counted_index_parent
            ), mock.patch.object(_Lookups, "parent", counted_lookups_parent):
                counts = self._counted(edit)
            counts.update(asked)
            counts["walks"] = rig.list.walks
            np.testing.assert_allclose(rig.list[-1].world_matrix[3, :3], [0.5, count + 1.0, 0.0])
            self.assertEqual(len(rig.list[0].get_branch()), count + 1)
            return counts

        for label, parent in parents.items():
            with self.subTest(label):
                small, large = built(40, parent), built(160, parent)
                self.assertEqual(small["builds"], large["builds"])
                self.assertEqual(small["walks"], large["walks"])
                self.assertLessEqual(large["matrices"], 4 * small["matrices"])
                self.assertEqual((small["shifts"], large["shifts"]), (0, 0))
                self.assertLessEqual(large["parents"], 4 * small["parents"] + 8)

    def test_a_rig_renamed_node_by_node_costs_each_node_the_same(self):
        # node.name = in a loop. On 938ae44 a rename looked for the children
        # naming the node: a rig with no current index -- just made, loaded,
        # copied, flat, or one a pop or a sort retired it in -- made a pass
        # over the list at the first rename and built the index at the
        # second. A name links no node now: a rename is the field, and the
        # name table of the index once a lookup by name built one. No pass,
        # no build, no matrix, in any state of the rig, every cached world
        # matrix standing, and the index, kept or not, left as it was
        def renamed(count, state):
            name     = "joint" if state == "one name" else None
            nodes    = [TransformData(name or f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig      = HierarchyData(nodes)
            rig.list = _Walked(rig.list)
            if state != "flat":
                for above, node in zip(nodes, nodes[1:]):
                    node.parent_uuid = above.uuid
            if state in ("in use", "a name looked up", "one name", "flat", "popped", "sorted", "copied"):
                rig.world_matrix  # its worlds cached, the index built if any parent
            if state == "a name looked up":
                nodes[-1].set_parent(nodes[-2].name, world_space=False)  # the name table built
            elif state == "popped":
                rig.append(TransformData("extra"))
                rig.pop()
            elif state == "sorted":
                rig.sort(key=lambda node: int(node.name[1:]))  # in the order they are
            elif state == "copied":
                rig      = rig.copy()
                rig.list = _Walked(rig.list)
                nodes    = list(rig.list)
            elif state == "loaded":
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                rig      = HierarchyData.load(rig.save(os.path.join(tmp.name, "rig.json")))
                rig.list = _Walked(rig.list)
                nodes    = list(rig.list)
            rig.list.walks = 0
            kept           = rig._kept_index()
            worlds         = [node._cached_world() is not None for node in nodes]

            def edit():
                for i, node in enumerate(nodes):
                    node.name = f"r{i}"

            counts          = self._counted(edit)
            counts["walks"] = rig.list.walks
            self.assertEqual(rig.name, [f"r{i}" for i in range(count)])
            self.assertIs(rig._kept_index(), kept)
            self.assertEqual([node._cached_world() is not None for node in nodes], worlds)
            if state != "flat":
                self.assertEqual([n.name for n in nodes[0].get_branch()], rig.name)
                self.assertEqual(nodes[-1].world_matrix[3, 1], float(count))
            _assert_lookups_are_a_scan(self, rig, state)
            return counts

        for state in (
            "in use", "a name looked up", "made", "loaded", "copied", "popped", "sorted", "one name", "flat",
        ):
            with self.subTest(state):
                small, large = renamed(40, state), renamed(160, state)
                self.assertEqual(small, large)
                self.assertEqual(large, {"builds": 0, "matrices": 0, "walks": 0})

    def test_a_rename_patches_the_name_table_built(self):
        # the name table a lookup by name built follows each rename, one name
        # given up, taken, shared, and given up again: no build, no pass, the
        # table a build makes; a rename of the first of a repeated name, which
        # can not tell the next, drops it, and the next lookup builds it
        rig = HierarchyData([TransformData(f"j{i}") for i in range(6)])
        for above, node in zip(rig.list, rig.list[1:]):
            node.parent_uuid = above.uuid
        rig.world_matrix
        built = rig._rig_index()
        rig.list[5].set_parent("j3", world_space=False)  # builds the name table
        table = built.names
        self.assertIsNotNone(table)

        # unique_name's table built ahead of the first rename too: a name given
        # up and then taken must not read the node that gave it up
        rig.unique_name

        for label, at, name, kept in (
            ("given up", 0, "fresh", True),
            ("taken", 2, "j0", True),
            ("shared behind", 4, "j0", True),
            ("shared ahead", 1, "j0", True),
            ("the first of a shared one given up", 1, "j1", False),
        ):
            rig.list[at].name = name
            self.assertIs(rig._kept_index(), built, label)
            self.assertEqual(built.names is table, kept, label)
            _assert_index_is_a_build(self, rig, label)
            self.assertEqual(
                rig.unique_name.tolist(),
                [all(m.name != n.name or m.uuid == n.uuid for m in rig.list) for n in rig.list],
                label,
            )
            leaf = rig.list[5]
            for wanted in set(rig.name) - {leaf.name}:
                leaf.set_parent(wanted, world_space=False)
                self.assertIs(leaf.get_parent(), _first_named_by_scan(rig, wanted), label)

    def test_a_name_table_no_lookup_reads_lapses(self):
        # renames past a quarter of the nodes and _RENAMES_SPARED, no lookup
        # by name between: the table lapses rather than being patched at each
        # rename, the index standing, and the next lookup builds it again. A
        # lookup between renames spares it again. Patched at every rename, a
        # loop renaming a rig whose table a lookup built ran at 1.0.4's speed
        from cgmath.hierarchy.hierarchy import _RENAMES_SPARED

        count = 160
        rig   = HierarchyData([TransformData(f"j{i}") for i in range(count)])
        for above, node in zip(rig.list, rig.list[1:]):
            node.parent_uuid = above.uuid
        index = rig._rig_index()
        leaf  = rig.list[-1]
        leaf.set_parent("j0", world_space=False)  # builds the name table
        spare = count // 4 + _RENAMES_SPARED
        for i, node in enumerate(rig.list[:spare]):
            node.name = f"r{i}"
        self.assertIsNotNone(index.names)
        _assert_index_is_a_build(self, rig, "spared")

        rig.list[spare].name = "lapsed"
        self.assertIsNone(index.names)
        self.assertIs(rig._kept_index(), index)
        leaf.set_parent("r3", world_space=False)
        self.assertIs(leaf.get_parent(), rig.list[3])
        _assert_index_is_a_build(self, rig, "built again")

        # a lookup every few renames keeps it, patched, the whole loop
        table = index.names
        for i, node in enumerate(rig.list[:-1]):
            node.name = f"s{i}"
            if i % 8 == 7:
                leaf.set_parent(f"s{i}", world_space=False)
                self.assertIs(leaf.get_parent(), node)
        self.assertIs(index.names, table)
        _assert_index_is_a_build(self, rig, "looked up between")

    def test_parent_uuid_loops_on_a_rig_in_use_patch_the_index(self):
        # a parent_uuid written on every node of a rig in use, nothing read
        # between: each patches the index, no build, no pass over the list,
        # and no node's children listed, only asked whether it has any; the
        # worlds read after are the ones the channels make
        from cgmath.hierarchy.hierarchy import _RigIndex

        def chain(count):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_uuid = above.uuid
            return rig

        def star(count):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for node in nodes[1:]:
                node.parent_uuid = nodes[0].uuid
            return rig

        def flattened(nodes):
            for node in nodes:
                node.parent_uuid = None

        def leaves_moved_and_back(nodes):
            for node in nodes[2:]:
                node.parent_uuid = nodes[1].uuid
            for node in nodes[2:]:
                node.parent_uuid = nodes[0].uuid

        def leaves_given_their_parent(nodes):
            for node in nodes[1:]:
                node.parent_uuid = nodes[0].uuid

        def under_the_root(nodes):
            for node in nodes[2:]:
                node.parent_uuid = nodes[0].uuid

        listed = _RigIndex.children
        for label, make, loop in (
            ("a chain flattened", chain, flattened),
            ("a star's leaves moved and back", star, leaves_moved_and_back),
            ("a star's leaves given the parent they have", star, leaves_given_their_parent),
            ("a chain put under its root", chain, under_the_root),
        ):
            counts = []
            for count in (40, 160):
                with self.subTest(label, count=count):
                    rig      = make(count)
                    rig.list = _Walked(rig.list)
                    rig.world_matrix
                    built          = rig._rig_index()
                    nodes          = list(rig.list)
                    rig.list.walks = 0
                    asked          = []

                    def counted_children(index, node):
                        asked.append(node)
                        return listed(index, node)

                    with mock.patch.object(_RigIndex, "children", counted_children):
                        spent = self._counted(lambda: loop(nodes))
                    spent["walks"]    = rig.list.walks
                    spent["children"] = len(asked)
                    counts.append(spent)

                    self.assertIs(rig._kept_index(), built)
                    if loop is leaves_given_their_parent:
                        # a parent_uuid naming the parent it had drops nothing
                        self.assertTrue(all(node._cached_world() is not None for node in nodes))
                    _assert_lookups_are_a_scan(self, rig, f"{label}, {count}")
                    np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)
            self.assertEqual(counts, [{"builds": 0, "matrices": 0, "walks": 0, "children": 0}] * 2, label)

    def test_a_long_row_taken_from_ahead_of_its_end_is_a_deque(self):
        # the children of one parent reparented one at a time leave its row
        # from the front: a list moved every later row each time, the square
        # of the children over the loop (5 of the 9 ms such a loop took over
        # a 2000 node star). A long row becomes a deque the first time a row
        # is put in or taken out ahead of its end; a short one, or one taken
        # from or put at its end, stays a list. Every patch keeps the rows in
        # order, as a sorted list of them is
        from cgmath.hierarchy.hierarchy import _DEQUE_ROWS, _insort_row, _remove_row

        long = _DEQUE_ROWS
        for label, rows, edit, kind in (
            ("short, the first taken", list(range(long - 1)), lambda t: _remove_row(t, "k", 0), list),
            ("long, the last taken", list(range(long)), lambda t: _remove_row(t, "k", long - 1), list),
            ("long, one put at the end", list(range(long)), lambda t: _insort_row(t, "k", long), list),
            ("long, the first taken", list(range(long)), lambda t: _remove_row(t, "k", 0), deque),
            ("long, one from the middle", list(range(long)), lambda t: _remove_row(t, "k", 5), deque),
            ("long, one put first", list(range(1, long + 1)), lambda t: _insort_row(t, "k", 0), deque),
        ):
            with self.subTest(label):
                table = {"k": rows}
                edit(table)
                self.assertIs(type(table["k"]), kind)

        rng   = random.Random(5)
        table = {"k": list(range(0, 4 * long, 2))}
        truth = list(table["k"])
        for _ in range(400):
            row = rng.randrange(4 * long)
            if row in truth:
                self.assertTrue(_remove_row(table, "k", row))
                truth.remove(row)
            else:
                self.assertFalse(_remove_row(table, "k", row))
                _insort_row(table, "k", row)
                truth = sorted(truth + [row])
            self.assertEqual(list(table.get("k", ())), truth)

    def test_a_rig_named_and_parented_node_by_node_builds_no_index(self):
        # renames with appends or parent_uuid writes between them, nothing
        # read: no rename builds the index or makes a pass over the rig, so
        # the loop costs each node the same
        def loaded(count):
            # a loader's loop: a node appended, then named and parented
            rig      = HierarchyData()
            rig.list = _Walked(rig.list)

            def edit():
                above = None
                for i in range(count):
                    node = TransformData("tmp", translate=(0.0, 1.0, 0.0))
                    rig.append(node)
                    node.name = f"j{i}"
                    if above is not None:
                        node.parent_uuid = above.uuid
                    above = node

            return rig, edit

        def made(count):
            # every node of a rig just made renamed, then parented, the first
            # made a root
            nodes    = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig      = HierarchyData(nodes)
            rig.list = _Walked(rig.list)

            def edit():
                for above, node in zip([None] + nodes, nodes):
                    node.name        = node.name + "_x"
                    node.parent_uuid = None if above is None else above.uuid

            return rig, edit

        for label, make, names in (
            ("appended, named, parented", loaded, "j{}"),
            ("renamed, parented", made, "j{}_x"),
        ):
            with self.subTest(label):
                for count in (40, 160):
                    rig, edit = make(count)
                    rig.list.walks = 0
                    spent          = self._counted(edit)
                    self.assertEqual(spent, {"builds": int(label == "appended, named, parented" and count > 64), "matrices": 0})
                    self.assertEqual(rig.name, [names.format(i) for i in range(count)])
                    self.assertEqual(rig.list[0].get_branch().name, rig.name)
                    self.assertEqual(rig.list[-1].world_matrix[3, 1], float(count))
                    _assert_lookups_are_a_scan(self, rig, f"{label}, {count}")

    def test_inserts_ahead_shift_the_index_only_for_a_read_between(self):
        # a node inserted ahead of the end moves every later position of the
        # index up one, a pass over all of it. Inserts with nothing read
        # between made that pass each time, 5 times 1.0.4's loop at 2000
        # nodes: past the first, they retire the index, as before it was
        # patched. With a world read between, the pass is made, cheaper than
        # the build a retired index would cost the read
        from cgmath.hierarchy.hierarchy import _RigIndex

        def in_use(count):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_uuid = above.uuid
            rig.world_matrix
            return rig

        def front(rig, count):
            for i in range(count):
                rig.insert(0, TransformData(f"k{i}"))

        def under_the_middle(rig, count):
            # each node put right after the middle one, under it, its world
            # read: the chain above is cached, so one local matrix a node
            for i in range(count):
                at = len(rig) // 2
                rig.insert(
                    at + 1,
                    TransformData(f"k{i}", parent_uuid=rig.list[at].uuid, translate=(1.0, 0.0, 0.0)),
                )
                rig.list[at + 1].world_matrix

        def front_renamed(rig, count):
            # each node put first, then named
            for i in range(count):
                node = TransformData("tmp")
                rig.insert(0, node)
                node.name = f"k{i}"

        def counted(loop, count):
            rig    = in_use(count)
            shift  = _RigIndex._shift
            shifts = []

            def counted_shift(index, at):
                shifts.append(at)
                return shift(index, at)

            with mock.patch.object(_RigIndex, "_shift", counted_shift):
                spent = self._counted(lambda: loop(rig, count))
            spent["shifts"] = len(shifts)

            _assert_lookups_are_a_scan(self, rig, f"{loop.__name__}, {count}")
            np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)
            return spent

        for loop, want in (
            (front, lambda count: {"builds": 0, "matrices": 0, "shifts": 1}),
            (under_the_middle, lambda count: {"builds": 0, "matrices": count, "shifts": count}),
            (front_renamed, lambda count: {"builds": 0, "matrices": 0, "shifts": 1}),
        ):
            for count in (40, 160):
                with self.subTest(loop.__name__, count=count):
                    self.assertEqual(counted(loop, count), want(count))

    def test_an_insert_that_retires_the_index_drops_the_worlds_it_moves(self):
        # the second of two inserts with nothing read between retires the
        # index rather than shift it: the cached world matrices stand, unless
        # a node names the one put in by its uuid, a parent_uuid written
        # before no node carried it
        ghost = generate_uuid()
        for label, added, kept in (
            ("named by none", lambda: TransformData("r", translate=(5.0, 0.0, 0.0)), True),
            ("named by uuid", lambda: TransformData("r", uuid=ghost, translate=(5.0, 0.0, 0.0)), False),
        ):
            with self.subTest(label):
                rig = HierarchyData(
                    [
                        TransformData("p", translate=(9.0, 0.0, 0.0)),
                        TransformData("z", translate=(0.0, 1.0, 0.0)),
                        TransformData("w", translate=(0.0, 1.0, 0.0)),
                        TransformData("x", translate=(0.0, 1.0, 0.0)),
                    ]
                )
                rig["z"].parent_uuid = rig["p"].uuid
                rig["w"].parent_uuid = rig["z"].uuid
                rig["x"].parent_uuid = ghost  # a uuid no node carries, written in place
                rig.world_matrix
                rig.insert(0, TransformData("q"))  # shifts the index the read fetched
                self.assertIsNotNone(rig._kept_index())
                node = added()
                rig.insert(1, node)

                self.assertIsNone(rig._kept_index())
                self.assertEqual([n._cached_world() is not None for n in rig.list[2:]], [kept] * 4)
                by_uuid = label == "named by uuid"
                self.assertIs(rig["z"].get_parent(), rig.list[2])
                np.testing.assert_allclose(rig["w"].world_matrix[3, :3], [9.0, 2.0, 0.0])
                if by_uuid:
                    self.assertIs(rig["x"].get_parent(), node)
                else:
                    self.assertEqual(rig["x"].get_parent(), ghost)
                np.testing.assert_allclose(rig["x"].world_matrix[3, :3], [5.0 if by_uuid else 0.0, 1.0, 0.0])
                _assert_lookups_are_a_scan(self, rig, label)

    def test_an_insert_at_or_past_either_end_patches_the_index(self):
        # insert(-1) and insert(len - 1) put the node ahead of the last one,
        # whose position the shift of the index moves too; a position past
        # either end puts it where list.insert does, first or last
        for label, where, order in (
            ("insert(-1)", lambda rig: -1, "abcxk"),
            ("insert(len - 1)", lambda rig: len(rig) - 1, "abcxk"),
            ("insert(len)", len, "abckx"),
            ("insert(len + 2)", lambda rig: len(rig) + 2, "abckx"),
            ("insert(-len - 2)", lambda rig: -len(rig) - 2, "xabck"),
        ):
            with self.subTest(label):
                rig                  = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
                rig["c"].parent_uuid = rig["a"].uuid
                kid                  = TransformData("k", translate=(0.0, 1.0, 0.0))
                rig.append(kid)
                kid.parent_uuid = rig["c"].uuid
                rig.world_matrix
                built = rig._rig_index()
                rig.insert(where(rig), TransformData("x", parent_uuid=rig["b"].uuid))

                self.assertIs(rig._kept_index(), built)
                self.assertEqual(rig.name, list(order))
                _assert_lookups_are_a_scan(self, rig, label)
                self.assertIs(kid.get_parent(), rig["c"])
                self.assertEqual(rig["b"].get_children().name, ["x"])

    def test_set_parent_given_a_node_takes_that_node_not_its_name(self):
        # a node handed to set_parent is found by its uuid, in the index or
        # by a scan, never as the first node carrying its name
        for kept in (False, True):
            with self.subTest(kept=kept):
                rig = HierarchyData(
                    [
                        TransformData("arm", translate=(5.0, 0.0, 0.0)),
                        TransformData("arm", translate=(-5.0, 0.0, 0.0)),
                        TransformData("hand"),
                    ]
                )
                if kept:
                    rig._rig_index()  # a flat rig's world reads build none
                self.assertEqual(rig._kept_index() is not None, kept)
                rig["hand"].set_parent(rig.list[1], world_space=False)
                self.assertIs(rig["hand"].get_parent(), rig.list[1])
                self.assertEqual(rig["hand"].world_matrix[3, 0], -5.0)
                _assert_lookups_are_a_scan(self, rig, f"kept={kept}")

    def test_a_name_or_a_parent_no_index_can_hold_retires_it(self):
        # name = 7 on a rig in use: no build can stand for the rig then, so
        # no patched index may either. parent_uuid = 7 raises and leaves the
        # index alone, and a node appended or inserted with a parent_uuid of
        # 7 written past the check joins as a root, which the index holds
        def appended(rig, at=None):
            node                         = TransformData("d")
            node.__dict__["parent_uuid"] = 7  # past the check
            if at is None:
                rig.append(node)
            else:
                rig.insert(at, node)
            self.assertIsNone(node.parent_uuid)

        for label, edit, kept in (
            ("name", lambda rig: setattr(rig.list[1], "name", 7), False),
            ("parent_uuid", lambda rig: setattr(rig.list[1], "parent_uuid", 7), True),
            ("appended", appended, True),
            ("inserted", lambda rig: appended(rig, 1), True),
        ):
            with self.subTest(label):
                rig                  = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
                rig["b"].parent_uuid = rig["a"].uuid
                rig["c"].parent_uuid = rig["b"].uuid
                rig.world_matrix
                built = rig._kept_index()
                self.assertIsNotNone(built)
                try:
                    edit(rig)
                except ValueError:
                    pass
                self.assertIs(rig._kept_index(), built if kept else None)
                self.assertEqual(rig._rig_index() is not None, label != "name")
                _assert_lookups_are_a_scan(self, rig, label)

    def test_a_rename_in_a_rig_no_index_can_stand_for(self):
        # a node whose uuid is not a string: the rig keeps None for its
        # index, a rename is the field alone, and every lookup a scan
        rig                          = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
        rig["b"].parent_uuid         = rig["a"].uuid
        rig["c"].parent_uuid         = rig["b"].uuid
        rig.list[2].__dict__["uuid"] = 12345
        self.assertIsNone(rig._rig_index())
        rig.list[1].name = "z"
        self.assertEqual(rig.name, ["a", "z", "c"])
        self.assertIs(rig.list[2].get_parent(), rig.list[1])
        self.assertIs(rig.list[1].get_parent(), rig.list[0])
        _assert_lookups_are_a_scan(self, rig, "a uuid that is not a string")

    def test_renames_in_a_rig_no_index_can_stand_for_build_nothing(self):
        # a uuid that is not a string, written past __setattr__: no index can
        # stand for the rig. A rename never builds one, so a loop of renames,
        # appending or parenting on the way, tries no build per rename. On
        # 938ae44 a parent_node of 7 kept the index out, and its renames each
        # made a pass over the rig, 248 ms over 2000 nodes where 1.0.4 took
        # 0.58 ms
        def spoilt(count):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_uuid = above.uuid
            nodes[-1].__dict__["uuid"] = 7  # a leaf: no child loses it
            self.assertIsNone(rig._rig_index())
            return rig, nodes

        def renamed(rig, nodes):
            for node in nodes:
                node.name = node.name + "_x"

        def renamed_and_reparented(rig, nodes):
            for above, node in zip(nodes, nodes[1:-1]):
                node.name        = node.name + "_x"
                node.parent_uuid = above.uuid

        for loop in (renamed, renamed_and_reparented):
            with self.subTest(loop.__name__):
                for count in (40, 160):
                    rig, nodes = spoilt(count)
                    rig.list = _Walked(rig.list)
                    spent    = self._counted(lambda: loop(rig, nodes))
                    self.assertEqual((spent["builds"], rig.list.walks), (0, 0))
                    self.assertIsNone(rig._rig_index())
                    _assert_lookups_are_a_scan(self, rig, f"{loop.__name__}, {count}")

        # the uuid written again as a string: the next lookup builds one
        rig, nodes = spoilt(12)
        nodes[-1].uuid = generate_uuid()
        self.assertIsNotNone(rig._rig_index())
        _assert_lookups_are_a_scan(self, rig, "fixed")
        self.assertEqual(nodes[-1].world_matrix[3, 1], 12.0)

    def test_a_node_added_takes_the_children_naming_its_uuid(self):
        # a parent_uuid written in place that no node carried hangs its node
        # under the node added with that uuid, its cached world matrix
        # dropped; the index patched
        ghost_uuid = generate_uuid()
        rig = HierarchyData(
            [
                TransformData("p", translate=(9.0, 0.0, 0.0)),
                TransformData("y", translate=(0.0, 1.0, 0.0)),
                TransformData("z", translate=(0.0, 1.0, 0.0)),
                TransformData("w", translate=(0.0, 1.0, 0.0)),
            ]
        )
        rig["y"].parent_uuid = ghost_uuid
        rig["z"].parent_uuid = rig["p"].uuid
        rig["w"].parent_uuid = rig["z"].uuid
        rig.world_matrix
        built = rig._rig_index()
        node  = TransformData("g", uuid=ghost_uuid, translate=(5.0, 0.0, 0.0))
        rig.append(node)

        self.assertIs(rig._rig_index(), built)
        _assert_index_is_a_build(self, rig, "by uuid")
        np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)
        self.assertIs(rig["y"].get_parent(), node)
        np.testing.assert_allclose(rig["y"].world_matrix[3, :3], [5.0, 1.0, 0.0])

    def test_a_node_the_rig_holds_is_not_added_again(self):
        # the check that the uuid is new reads the index the rig keeps
        rig = _mixed_rig()
        rig.world_matrix
        built = rig._rig_index()
        neck  = rig["neck"]
        for label, add in (
            ("append", lambda node: rig.append(node)),
            ("insert", lambda node: rig.insert(2, node)),
        ):
            for node in (neck, TransformData("other", uuid=neck.uuid)):
                with self.subTest(label, node=node.name):
                    listed = [id(n) for n in rig.list]
                    with self.assertRaises(RuntimeError):
                        add(node)
                    self.assertEqual([id(n) for n in rig.list], listed)
                    self.assertIs(rig._kept_index(), built)

    def test_a_node_added_reads_its_world_in_this_rig(self):
        # its world matrix cached in another rig is not trusted in this one,
        # whose index is patched and whose own cached worlds stand. Its parent
        # stays behind: it joins this rig as a root
        rig = _mixed_rig()
        rig.world_matrix
        other = HierarchyData([TransformData("lift", translate=(0.0, 50.0, 0.0))])
        moved = TransformData("moved", parent_uuid=other["lift"].uuid, translate=(1.0, 0.0, 0.0))
        other.append(moved)
        np.testing.assert_allclose(moved.world_matrix[3, :3], [1.0, 50.0, 0.0])

        built = rig._rig_index()
        rig.append(moved)
        self.assertIsNone(moved.parent_uuid)
        self.assertIs(rig._kept_index(), built)
        self.assertTrue(all(n._cached_world() is not None for n in rig.list[:-1]))
        np.testing.assert_allclose(moved.world_matrix[3, :3], [1.0, 0.0, 0.0])

        moved.parent_uuid = rig["head"].uuid
        self.assertIs(rig._kept_index(), built)
        np.testing.assert_allclose(moved.world_matrix, moved.matrix @ rig["head"].world_matrix)

    def test_a_structural_edit_leaves_a_primitive_field_clean(self):
        # a node added or given another parent drops cached world matrices
        # only, patched or not: _transform_changed, which a primitive
        # extends to mark its field dirty, is a channel write's, as before
        from cgmath.geometry.sdf import DMCField, SDFSphere

        rig                    = HierarchyData([TransformData("root"), TransformData("arm")])
        rig["arm"].parent_uuid = rig["root"].uuid
        rig.world_matrix
        field = DMCField()
        ball  = SDFSphere(radius=0.5, name="ball")
        field.add(ball)
        for label, edit in (
            ("append", lambda: rig.append(ball)),
            ("parent_uuid", lambda: setattr(ball, "parent_uuid", rig["arm"].uuid)),
            ("parent_uuid of the root", lambda: setattr(ball, "parent_uuid", rig["root"].uuid)),
        ):
            with self.subTest(label):
                self.assertIsNotNone(rig._kept_index())
                field._dirty = False
                edit()
                self.assertFalse(field._dirty)
        ball.translate = (1.0, 0.0, 0.0)
        self.assertTrue(field._dirty)

    def test_unique_names_follow_a_name_given_up_and_taken(self):
        # one node renamed at a time: the names the index keeps for
        # unique_name go with each rename, the index kept
        rig = HierarchyData([TransformData(f"j{i}") for i in range(4)])
        for above, node in zip(rig.list, rig.list[1:]):
            node.parent_uuid = above.uuid
        rig.world_matrix
        built = rig._rig_index()
        self.assertEqual(rig.unique_name.tolist(), [True] * 4)

        def by_scan():
            return [all(m.name != n.name or m.uuid == n.uuid for m in rig.list) for n in rig.list]

        for label, node, name in (
            ("j0 given up", 0, "fresh"),
            ("j0 taken", 2, "j0"),
            ("j0 shared", 1, "j0"),
            ("j0 given up again", 2, "j2"),
        ):
            rig.list[node].name = name
            self.assertEqual(rig.unique_name.tolist(), by_scan(), label)
            self.assertIs(rig._kept_index(), built, label)

    def test_a_name_shaped_like_a_uuid_names_no_node(self):
        # set_parent given text shaped like a uuid, with or without its
        # hyphens, looks for that uuid, read from the index or found by a
        # scan: a node carrying it as a name is not found, and a node whose
        # parent_uuid it is hangs from no node, whatever the node carrying
        # it as a name is renamed
        for odd in (generate_uuid(), generate_uuid().replace("-", "")):
            for kept in (False, True):
                with self.subTest(len(odd), kept=kept):
                    rig = HierarchyData(
                        [TransformData("a"), TransformData(odd), TransformData("c"), TransformData("d")]
                    )
                    rig["d"].parent_uuid = odd
                    if kept:
                        rig._rig_index()
                    self.assertEqual(rig._kept_index() is not None, kept)
                    with self.assertRaises(ValueError):
                        rig["c"].set_parent(odd)
                    self.assertIsNone(rig["c"].parent_uuid)

                    self.assertEqual(rig._kept_index() is not None, kept)
                    rig.list[1].name = "plain"
                    self.assertEqual(rig["d"].parent_uuid, odd)
                    self.assertEqual(rig["d"].get_parent(), odd)
                    _assert_index_is_a_build(self, rig, f"{len(odd)}, kept={kept}")

        # a digit short of a uuid's 32 is a name like any other: set_parent
        # finds it, and keeps the node's uuid
        short = generate_uuid().replace("-", "")[:31]
        rig   = HierarchyData([TransformData(short), TransformData("d")])
        rig._rig_index()
        rig["d"].set_parent(short, world_space=False)
        self.assertIs(rig["d"].get_parent(), rig.list[0])
        rig.list[0].name = "plain"
        self.assertEqual(rig["d"].parent_uuid, rig.list[0].uuid)
        self.assertIs(rig["d"].get_parent(), rig.list[0])

    def test_a_rename_reads_no_index_it_does_not_need(self):
        # after an edit that retired the index, the list shrunk raw, or the
        # list replaced past __setattr__, a rename keeps every child, each
        # found by a scan before it, under its parent
        def retired(rig):
            rig["toe"].uuid        = generate_uuid()
            rig["toe"].parent_uuid = rig["head"].uuid

        def shrunk(rig):
            rig.list.pop()  # tip, under head

        def replaced(rig):
            nodes                = list(rig.list)
            late                 = TransformData("late", parent_uuid=rig["head"].uuid)
            late._hierarchy      = rig
            nodes[7]             = late  # in place of toe
            rig.__dict__["list"] = nodes

        for label, edit in (("retired", retired), ("shrunk raw", shrunk), ("replaced raw", replaced)):
            with self.subTest(label):
                rig = _mixed_rig()
                rig._rig_index()
                edit(rig)
                head      = rig["head"]
                kids      = _children_by_scan(head)
                head.name = "skull"
                self.assertEqual([id(n) for n in head.get_children()], [id(n) for n in kids])
                self.assertTrue(all(kid.get_parent() is head for kid in kids))
                _assert_index_is_a_build(self, rig, label)

    def test_an_index_out_of_step_is_retired_not_patched(self):
        # a name or a parent_uuid written past __setattr__, two nodes carrying
        # one uuid, or a name the index can not key on: the edit answers as a
        # scan does, and the next lookup builds an index where one can stand
        def checked(rig, label, indexed=True):
            _assert_index_is_a_build(self, rig, label)
            self.assertEqual(rig._rig_index() is not None, indexed, label)
            for node in rig.list:
                self.assertEqual(
                    [id(n) for n in node.get_children()],
                    [id(n) for n in _children_by_scan(node)],
                    f"{label}: children of {node.name}",
                )

        # b no longer names a, though the index has it naming a
        rig                  = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
        rig["b"].parent_uuid = rig["a"].uuid
        rig["c"].parent_uuid = rig["a"].uuid
        rig._rig_index()
        rig["b"].__dict__["parent_uuid"] = None
        rig["b"].parent_uuid             = rig["c"].uuid
        self.assertEqual(rig["b"].parent_uuid, rig["c"].uuid)
        checked(rig, "a parent_uuid written past __setattr__")

        # b carries a's name, though the table has a alone carrying it: the
        # rename of b drops the table rather than patch it
        rig = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
        rig._rig_index().named("a")
        first, second = rig.list[0], rig.list[1]
        second.__dict__["name"] = "a"
        second.name             = "z"
        self.assertIsNone(rig._kept_index().names)
        from cgmath.hierarchy.hierarchy import _held_node

        self.assertIs(_held_node(rig, "a"), first)
        with self.assertRaises(ValueError):
            _held_node(rig, "b")
        checked(rig, "a name written past __setattr__")

        # the index holds the first node of a uuid; the second is not it
        rig = HierarchyData(
            [TransformData("p"), TransformData("q"), TransformData("a"), TransformData("b"), TransformData("c")]
        )
        rig["b"].uuid        = rig["a"].uuid
        rig["c"].parent_uuid = rig["b"].uuid
        rig._rig_index()
        rig["b"].parent_uuid = rig["q"].uuid
        checked(rig, "a uuid two nodes carry, a parent_uuid")
        rig._rig_index().named("b")
        rig["b"].name = "z"
        checked(rig, "a uuid two nodes carry, a rename")

        # a name two nodes carry written past __setattr__ on a third, between
        # them: the table does not have it carrying that name
        rig = HierarchyData([TransformData("a"), TransformData("a"), TransformData("q"), TransformData("a")])
        rig._rig_index().named("a")
        rig.list[2].__dict__["name"] = "a"
        rig.list[2].name             = "z"
        self.assertEqual(rig.name, ["a", "a", "z", "a"])
        self.assertIsNone(rig._kept_index().names)
        checked(rig, "a shared name written past __setattr__")

        # a name that is not a string: no index can stand for the rig
        rig        = _mixed_rig()
        seven      = TransformData("seven", parent_uuid=rig["spine"].uuid)
        seven.name = 7  # past the constructor, which makes it a string
        rig._rig_index()
        rig.append(seven)
        self.assertIsNone(rig._rig_index())
        self.assertIn(7, [n.name for n in rig["spine"].get_children()])
        checked(rig, "a name that is not a string", indexed=False)

    def test_random_edits_match_a_scan(self):
        # every kind of edit, at random: through the index the rig keeps,
        # patched or built again, every lookup is the one a scan gives and
        # every world matrix, most cached ahead of the edit, the one its
        # channels make
        ran = {}
        for seed in (1, 2, 3):
            for label, count in _fuzz_rig_edits(self, seed, 300).items():
                ran[label] = ran.get(label, 0) + count
        self.assertTrue(all(ran.values()), ran)

    def test_random_edits_on_a_large_rig_match_a_scan(self):
        # the same on a rig of _BUILT_FOR_APPEND nodes or more, where an
        # append or a parent looked up with no current index builds one, or
        # scans while such builds go to waste
        from cgmath.hierarchy.hierarchy import _BUILT_FOR_APPEND

        ran = _fuzz_rig_edits(self, 4, 120, size=_BUILT_FOR_APPEND + 8)
        self.assertGreater(ran["moved to the end"] + ran["appended, named, parented"], 0, ran)


class TestParentUuid(unittest.TestCase):
    """a node's parent is a uuid: parent_uuid takes a uuid or None, a name
    goes through set_parent, parent_name reads the parent's name, and
    parent_node is parent_uuid's deprecated name until 3.0.0"""

    def test_parent_uuid_takes_a_uuid_or_none(self):
        rig  = _chain()
        head = rig["head"]
        for value in (rig["root"].uuid, rig["root"].uuid.lower(), None):
            head.parent_uuid = value
            self.assertEqual(head.parent_uuid, value)

        # any form validate_uuid takes: no hyphens, braces, a urn
        loose = TransformData("loose")
        bare  = generate_uuid()
        for value in (bare.replace("-", ""), "{" + bare + "}", "urn:uuid:" + bare):
            loose.parent_uuid = value
            self.assertEqual(loose.parent_uuid, value)

        # a uuid.UUID is kept as its text, as the uuid keyword keeps one: in
        # the constructor, through the deprecated keyword, written, as
        # 938ae44 kept its text, rather than refused as "not UUID('...')"
        child, parent = uuid_module.uuid4(), uuid_module.uuid4()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for given in ({"parent_uuid": parent}, {"parent_node": parent}):
                with self.subTest(given=list(given)):
                    kid = TransformData("kid", uuid=child, **given)
                    self.assertEqual(kid.parent_uuid, str(parent))
                    HierarchyData([kid, TransformData("root", uuid=parent)])
                    self.assertEqual(kid.parent_name, "root")
        loose.parent_uuid = parent
        self.assertEqual(loose.parent_uuid, str(parent))

    def test_anything_else_raises_and_names_set_parent(self):
        rig  = _chain()
        head = rig["head"]
        held = head.parent_uuid
        for value, message in (
            ("spine", r"^TransformData\.parent_uuid takes a uuid or None, not 'spine'; "
                      r"set_parent\('spine'\) parents by name$"),
            ("", r"^TransformData\.parent_uuid takes a uuid or None, not ''; set_parent\(''\)"),
            (7, r"^TransformData\.parent_uuid takes a uuid or None, not 7$"),
            (["x"], r"^TransformData\.parent_uuid takes a uuid or None, not \['x'\]$"),
            (generate_uuid()[:-1], r"takes a uuid or None"),
            # 36 characters, the usual length, yet no uuid: a joint's long name
            (generate_uuid()[:-1] + "g", r"takes a uuid or None"),
            ("x" * 36, r"takes a uuid or None"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    head.parent_uuid = value
                self.assertEqual(head.parent_uuid, held)
                with self.assertRaisesRegex(ValueError, message):
                    TransformData("x", parent_uuid=value)

                # a load, of a node and of a rig: the tree it reads names it
                # under parent_uuid
                tree                                = _chain().to_dict()
                tree["__items__"][2]["parent_uuid"] = value
                with self.assertRaisesRegex(ValueError, message):
                    HierarchyData.from_dict(tree)
                with self.assertRaisesRegex(ValueError, message):
                    TransformData.from_dict(tree["__items__"][2])

        # a subclass names itself
        from cgmath.geometry.sdf import SDFSphere

        with self.assertRaisesRegex(ValueError, r"^SDFSphere\.parent_uuid takes a uuid"):
            SDFSphere().parent_uuid = "group"

    def test_a_node_with_no_uuid_is_no_parent(self):
        rig                       = HierarchyData([TransformData("a"), TransformData("b")])
        rig["a"].__dict__["uuid"] = "j1"  # a uuid not shaped like one
        with self.assertRaisesRegex(ValueError, r"^'a' can not be a parent: its uuid 'j1' is not a uuid$"):
            rig["b"].set_parent("a")
        with self.assertRaisesRegex(ValueError, "can not be a parent"):
            rig[1:].set_parent("a")
        self.assertIsNone(rig["b"].parent_uuid)
        self.assertEqual(rig.name, ["a", "b"])

    def test_parent_name_reads_the_parent(self):
        rig = _chain()
        self.assertEqual([n.parent_name for n in rig], [None, "root", "spine"])
        rig["root"].name = "hips"
        self.assertEqual(rig["spine"].parent_name, "hips")

        # a node of no hierarchy, and a parent the hierarchy does not hold
        self.assertIsNone(TransformData("loose", parent_uuid=generate_uuid()).parent_name)
        rig["head"].parent_uuid = generate_uuid()
        self.assertIsNone(rig["head"].parent_name)

        # read only, and never saved
        with self.assertRaises(AttributeError):
            rig["spine"].parent_name = "head"
        self.assertNotIn("parent_name", rig["spine"].to_dict())
        self.assertIn("parent_uuid", rig["spine"].to_dict())
        self.assertNotIn("parent_node", rig["spine"].to_dict())

    def test_set_parent_takes_a_node_a_uuid_or_a_name(self):
        # whichever way it is given, the node keeps the parent's uuid; a name
        # is the first node of the hierarchy carrying it
        for kept in (False, True):
            for how in ("node", "uuid", "name"):
                with self.subTest(how, kept=kept):
                    rig = HierarchyData(
                        [
                            TransformData("arm", translate=(5.0, 0.0, 0.0)),
                            TransformData("arm", translate=(-5.0, 0.0, 0.0)),
                            TransformData("hand", translate=(1.0, 0.0, 0.0)),
                        ]
                    )
                    if kept:
                        rig._rig_index()
                    first = rig.list[0]
                    given = {"node": first, "uuid": first.uuid, "name": "arm"}[how]
                    rig["hand"].set_parent(given, world_space=False)
                    self.assertEqual(rig["hand"].parent_uuid, first.uuid)
                    self.assertIs(rig["hand"].get_parent(), first)
                    self.assertEqual(rig["hand"].parent_name, "arm")
                    np.testing.assert_allclose(rig["hand"].world_matrix[3, :3], [6.0, 0.0, 0.0])

                    # a view the same, and a rename after moves nothing
                    rig["hand"].set_parent(None, world_space=False)
                    TransformList([rig["hand"]]).set_parent(given, world_space=False)
                    self.assertIs(rig["hand"].get_parent(), first)
                    first.name = "shoulder"
                    self.assertIs(rig["hand"].get_parent(), first)

                    with self.assertRaises(ValueError):
                        rig["hand"].set_parent("nobody")
                    with self.assertRaises(ValueError):
                        rig["hand"].set_parent(generate_uuid())

    def test_a_view_takes_the_first_node_of_a_repeated_name(self):
        # a view takes its nodes to the world first, each branch moved to
        # the end of the list: a name is the first node carrying it once
        # they have moved. The second arm hangs below hand and moves with it,
        # so the first arm is the parent, and the check for a loop, which
        # the second arm would fail, is made on it
        for kept in (False, True):
            for world_space in (False, True):
                with self.subTest(kept=kept, world_space=world_space):
                    first, hand, second = (
                        TransformData("arm", translate=(5.0, 0.0, 0.0)),
                        TransformData("hand", translate=(1.0, 0.0, 0.0)),
                        TransformData("arm", translate=(0.0, 3.0, 0.0)),
                    )
                    rig = HierarchyData([first, hand, second])
                    second.set_parent(hand, world_space=False)
                    if kept:
                        rig._rig_index()
                    TransformList([hand]).set_parent("arm", world_space=world_space)
                    self.assertIs(hand.get_parent(), first)
                    self.assertIs(second.get_parent(), hand)
                    np.testing.assert_allclose(hand.world_matrix[3, :3], [1.0 if world_space else 6.0, 0.0, 0.0])

    def test_a_parent_uuid_written_is_matched_once(self):
        # a parent_uuid written is matched as a uuid once: a text found to be
        # one is kept, and so is each uuid generate_uuid makes. A loop
        # chaining the nodes a rig just gave uuids matches none, and one
        # writing a parent over and over matches it once: matching each ran
        # the chain 1.2 times aba1ce5's
        module   = sys.modules[HierarchyData.__module__]
        pattern  = module._UUID_TEXT
        matched  = []
        counting = mock.Mock(fullmatch=lambda value: matched.append(value) or pattern.fullmatch(value))

        def matches(count):
            del matched[:]
            with mock.patch.object(module, "_UUIDS_SEEN", set()), mock.patch.object(module, "_UUID_TEXT", counting):
                rig = HierarchyData([TransformData(f"j{i}") for i in range(count)])
                for above, node in zip(rig.list, rig.list[1:]):
                    node.parent_uuid = above.uuid
                chained = len(matched)

                text = str(uuid_module.uuid4())  # no uuid generate_uuid made
                for node in rig.list:
                    node.parent_uuid = text
                self.assertTrue(all(node.parent_uuid == text for node in rig.list))
                return chained, len(matched) - chained

        self.assertEqual(matches(40), (0, 1))
        self.assertEqual(matches(160), (0, 1))

    def test_the_alias_finds_its_caller_once_a_file(self):
        # each write through parent_node files its warning under the first
        # caller outside cgmath, known by the file each frame runs in: the
        # answer for each file is kept, where working it out at each write
        # ran a loop of such writes 3.6 to 3.8 times 938ae44's. Each count
        # starts with no file known, whatever ran before it
        absolute = os.path.abspath
        answers  = sys.modules[HierarchyData.__module__]._OUTSIDE

        def made(count):
            paths = []

            def counted(path):
                paths.append(path)
                return absolute(path)

            rig = _chain()
            with warnings.catch_warnings(), mock.patch.object(os.path, "abspath", counted), mock.patch.dict(
                answers, clear=True
            ):
                warnings.simplefilter("ignore")
                for _ in range(count):
                    rig["head"].parent_node = rig["root"].uuid
            self.assertIs(rig["head"].get_parent(), rig["root"])
            return len(paths)

        small, large = made(4), made(16)
        self.assertGreater(small, 0)
        self.assertEqual(small, large)

    def test_parent_node_is_a_deprecated_alias(self):
        rig  = _chain()
        head = rig["head"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(head.parent_node, rig["spine"].uuid)
            head.parent_node = rig["root"].uuid
            self.assertIs(head.get_parent(), rig["root"])
            self.assertEqual(head.parent_uuid, rig["root"].uuid)
            node = TransformData("x", parent_node=rig["root"].uuid)
            self.assertEqual(node.parent_uuid, rig["root"].uuid)
        self.assertEqual(len(caught), 3)
        for warning in caught:
            self.assertIs(warning.category, DeprecationWarning)
            self.assertEqual(os.path.normcase(warning.filename), os.path.normcase(__file__))
            self.assertRegex(
                str(warning.message),
                r"^TransformData\.parent_node is deprecated and goes in cgmath 3\.0\.0: "
                r"it is parent_uuid now, and set_parent\(\) parents by name$",
            )

        # a name through it raises as parent_uuid does, and changes nothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaisesRegex(ValueError, r"not 'spine'; set_parent\('spine'\) parents by name$"):
                head.parent_node = "spine"
            self.assertEqual(head.parent_uuid, rig["root"].uuid)
            with self.assertRaisesRegex(ValueError, "takes a uuid or None"):
                TransformData("x", parent_node="spine")
            with self.assertRaisesRegex(TypeError, "parent_uuid or parent_node"):
                TransformData("x", parent_uuid=rig["root"].uuid, parent_node=rig["root"].uuid)

    def test_the_alias_warns_the_caller_of_a_render_node(self):
        # an Object, a Camera and a Light hand the keyword on to
        # TransformData's constructor: the warning is the caller's still,
        # which Python's default filters show from a script, not filed
        # under render/scene.py, which they hide
        from cgmath.render import Camera, Light, Object

        root = generate_uuid()
        for cls in (Object, Camera, Light):
            with self.subTest(cls.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    node             = cls(name="n", parent_node=root)
                    node.parent_node = None
                self.assertIsNone(node.parent_uuid)
                self.assertEqual(
                    [(w.category, os.path.normcase(w.filename)) for w in caught],
                    [(DeprecationWarning, os.path.normcase(__file__))] * 2,
                )
                self.assertTrue(str(caught[0].message).startswith(f"{cls.__name__}.parent_node is deprecated"))

    def test_the_alias_maps_parent_uuid_in_to_attributes(self):
        # a to_attributes mapping naming parent_node maps parent_uuid, with
        # the warning, once a call: a node, a rig, a clip's frames. Read as
        # a field, the alias mapped nothing and the parent was left out
        from cgmath.hierarchy import ClipData

        rig  = _chain()
        head = rig["head"]
        clip = ClipData(_chain(), frames=2)
        for label, call, expected in (
            ("a node", lambda m: head.to_attributes(m), {"parent": rig["spine"].uuid}),
            ("a rig", lambda m: rig.to_attributes(m)[2], {"parent": rig["spine"].uuid}),
            ("a clip's frames", lambda m: clip.frames.to_attributes(m)[2], {"parent": clip["spine"].uuid}),
        ):
            with self.subTest(label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertEqual(call({"parent_node": "parent"}), expected)
                self.assertEqual(
                    [(w.category, os.path.normcase(w.filename)) for w in caught],
                    [(DeprecationWarning, os.path.normcase(__file__))],
                )
                self.assertEqual(call({"parent_uuid": "parent"}), expected)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    with self.assertRaisesRegex(
                        ValueError,
                        r"^TransformData\.parent_uuid is given twice in the mapping, as 'parent_node' "
                        r"and 'parent_uuid'$",
                    ):
                        call({"parent_node": "a", "parent_uuid": "b"})

    def test_a_batch_joins_in_any_order(self):
        # the constructor, extend and a load check the parents once every
        # node is in: a child listed ahead of its parent keeps it
        from cgmath.hierarchy import ClipData

        ids = _ids("root", "spine", "head")
        nodes = lambda: [
            TransformData("head", uuid=ids["head"], parent_uuid=ids["spine"], translate=(0.0, 1.0, 0.0)),
            TransformData("spine", uuid=ids["spine"], parent_uuid=ids["root"], translate=(0.0, 2.0, 0.0)),
            TransformData("root", uuid=ids["root"], translate=(1.0, 0.0, 0.0)),
        ]
        extended = HierarchyData([TransformData("other")])
        extended.extend(nodes())
        by_name = _chain()
        by_name.sort()  # head, root, spine: a child ahead of its parent
        for label, rig in (
            ("constructor", HierarchyData(nodes())),
            ("extend", extended),
            ("from_dict", HierarchyData.from_dict(HierarchyData(nodes()).to_dict())),
            ("clip", ClipData(nodes(), frames=2)),
            # copy.deepcopy adds every copy at once: appended each in turn,
            # a child copied ahead of its parent would become a root
            ("deepcopy", copy.deepcopy(HierarchyData(nodes()))),
            ("deepcopy, a rig sorted", copy.deepcopy(by_name)),
            ("deepcopy, a clip", copy.deepcopy(ClipData(nodes(), frames=2))),
        ):
            with self.subTest(label):
                self.assertEqual(rig["head"].parent_name, "spine")
                self.assertEqual(rig["spine"].parent_name, "root")
                np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [1.0, 3.0, 0.0])

    def test_a_batch_keeps_a_parent_the_rig_holds(self):
        # nodes extending a rig and hanging from a node it held already keep
        # that parent, whether the rig keeps a current index or not
        for kept in (False, True):
            with self.subTest(kept=kept):
                rig                      = HierarchyData([TransformData("root"), TransformData("spine")])
                rig["spine"].parent_uuid = rig["root"].uuid
                if kept:
                    rig._rig_index()
                self.assertEqual(rig._kept_index() is not None, kept)
                spine = rig["spine"].uuid
                for count in (1, 5):  # one node, and a batch extend() sets up a set for
                    rig.extend([TransformData(f"leaf{i}", parent_uuid=spine) for i in range(count)])
                self.assertEqual([n.parent_name for n in rig], [None, "root"] + ["spine"] * 6)

    def test_a_node_joining_without_its_parent_is_a_root(self):
        # appended, inserted, in a batch, put in place of another while no
        # rig owns it, copied out of its rig, or taken from another rig: a
        # parent the hierarchy does not hold is cleared
        def joined(join):
            rig  = _chain()
            node = TransformData("loose", parent_uuid=generate_uuid(), translate=(0.0, 1.0, 0.0))
            join(rig, node)
            return rig, node

        for label, join in (
            ("append", lambda rig, node: rig.append(node)),
            ("insert", lambda rig, node: rig.insert(1, node)),
            ("extend", lambda rig, node: rig.extend([node])),
            ("a batch", lambda rig, node: rig.extend([node, TransformData("x"), TransformData("y")])),
            ("setitem", lambda rig, node: rig.__setitem__(2, node)),
            ("a slice", lambda rig, node: rig.__setitem__(slice(1, 2), [node])),
        ):
            with self.subTest(label):
                rig, node = joined(join)
                self.assertIn(node, rig.list)
                self.assertIsNone(node.parent_uuid)
                np.testing.assert_allclose(node.world_matrix[3, :3], [0.0, 1.0, 0.0])

        # a lone copy of a node, or the node itself, into another rig
        rig  = _chain()
        twin = rig["head"].copy()
        self.assertEqual(twin.parent_uuid, rig["spine"].uuid)
        HierarchyData([TransformData("lift")]).append(twin)
        self.assertIsNone(twin.parent_uuid)
        head = rig["head"]
        HierarchyData([TransformData("lift")]).append(head)
        self.assertIsNone(head.parent_uuid)
        self.assertEqual(rig.name, ["root", "spine"])

        # a view's copy, a delta's operand: the nodes whose parent it leaves out
        rig  = _chain()
        part = rig[1:].copy()
        self.assertEqual([n.parent_name for n in part], [None, "spine"])

        # a node of another rig put in place keeps its rig and its parent
        rig, other = _chain(), _chain()
        held   = other["head"]
        rig[0] = held
        self.assertIs(held._hierarchy, other)
        self.assertEqual(held.parent_uuid, other["spine"].uuid)

    def test_a_parent_added_later_does_not_take_a_root_back(self):
        # joining cleared the parent_uuid: the parent added after finds no
        # child; one written in place since names it again, and is found
        ids = _ids("p", "c")
        rig = HierarchyData([TransformData("c", uuid=ids["c"], parent_uuid=ids["p"])])
        self.assertIsNone(rig["c"].parent_uuid)
        rig.append(TransformData("p", uuid=ids["p"], translate=(3.0, 0.0, 0.0)))
        self.assertIsNone(rig["c"].get_parent())
        self.assertEqual(len(rig["p"].get_children()), 0)

        rig["c"].parent_uuid = ids["p"]
        self.assertIs(rig["c"].get_parent(), rig["p"])

        # the same for a batch whose parent comes in a later batch
        rig = HierarchyData([TransformData("c", uuid=generate_uuid(), parent_uuid=ids["p"])])
        rig.extend([TransformData("p", uuid=ids["p"])])
        self.assertIsNone(rig["c"].parent_uuid)

    def test_a_node_taken_out_with_del_or_remove_can_still_be_edited(self):
        # del and remove() leave the node pointing at its old hierarchy, where
        # a child still holds its uuid: an edit's branch walk must not trip
        # on that. The child hands the uuid back and reads as a root
        for how in ("del", "remove"):
            for kept in (False, True):
                with self.subTest(how, kept=kept):
                    rig  = _chain()
                    root = rig["root"]
                    if kept:
                        rig.world_matrix  # the index built, the worlds cached
                    if how == "del":
                        del rig[0]
                    else:
                        rig.remove(root)

                    root.translate = (4.0, 0.0, 0.0)
                    np.testing.assert_allclose(root.translate, [4.0, 0.0, 0.0])
                    self.assertEqual(rig["spine"].get_parent(), root.uuid)
                    self.assertIsNone(rig["spine"].parent_name)
                    np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [0.0, 3.0, 0.0])

    def test_the_skeleton_asset_still_loads(self):
        # saved under "parent_node", every reference a uuid
        rig = HierarchyData.load(SKELETON)
        self.assertEqual(len(rig), 109)
        self.assertEqual(rig.get_roots().name, ["root_joint"])
        self.assertTrue(all(validate_uuid(n.parent_uuid) for n in rig[1:]))
        self.assertEqual(rig["pelvis_joint"].parent_name, "root_joint")
        self.assertIn("parent_uuid", rig.to_dict()["__items__"][1])


# a rig 938ae44 saved, its parent_node holding uuids and names: each node's
# parent as 938ae44 read it, and the world translations it read
_LEGACY_PARENTS = [
    ("kid", "late"),
    ("root", None),
    ("spine", "root"),
    ("arm", "spine"),
    ("hand", "arm"),
    ("twin", "root"),
    ("twin", "spine"),
    ("finger", "twin"),
    ("thumb", "twin"),
    ("late", "arm"),
    ("lost", None),
    ("gone", None),
]
_LEGACY_WORLDS = [
    [0.765518, -2.11103, -0.471442],
    [0.403077, -0.997951, -1.516363],
    [-0.458234, -1.425214, -2.101667],
    [-0.008342, -2.292097, -1.739381],
    [2.826262, -2.266727, -0.946889],
    [0.997004, -1.253873, -0.222164],
    [-0.888374, -2.74749, -2.679295],
    [-0.387786, -1.156694, -2.471681],
    [-0.717067, -2.923732, -1.746601],
    [-0.217131, -2.365982, -0.431773],
    [0.612456, 0.25832, 0.345498],
    [-0.248027, -0.7314, 0.371559],
]


class TestLegacyParentNode(unittest.TestCase):
    """a file older than parent_uuid -- written here by 938ae44 -- names each
    parent under "parent_node", by uuid or by name: a uuid reads as it is,
    a name as the first node of that name in the loaded rig, a child ahead
    of its parent included, and a parent no node is a root"""

    ASSETS = os.path.join(os.path.dirname(__file__), "test_assets")

    def _assert_read_as_938ae44_read(self, rig, label):
        self.assertEqual([(n.name, n.parent_name) for n in rig], _LEGACY_PARENTS, label)
        twins = [n for n in rig if n.name == "twin"]
        self.assertIs(rig["finger"].get_parent(), twins[0], label)
        self.assertIs(rig["thumb"].get_parent(), twins[1], label)
        self.assertIsNone(rig["lost"].parent_uuid, label)
        self.assertIsNone(rig["gone"].parent_uuid, label)
        self.assertTrue(all(n.parent_uuid is None or validate_uuid(n.parent_uuid) for n in rig), label)
        np.testing.assert_allclose(rig.world_matrix[:, 3, :3], _LEGACY_WORLDS, atol=2e-6, err_msg=label)

        # a deep copy reads the same parents, kid ahead of late included
        self.assertEqual([(n.name, n.parent_name) for n in copy.deepcopy(rig)], _LEGACY_PARENTS, label)

        # saved again, it names parent_uuid alone and reads back the same
        tree = rig.to_dict()
        self.assertTrue(all("parent_node" not in node for node in tree["__items__"]), label)
        again = type(rig).from_dict(tree)
        self.assertEqual([n.parent_uuid for n in again], [n.parent_uuid for n in rig], label)

    def test_a_rig_in_every_format(self):
        for ext in ("json", "npz", "pkl"):
            with self.subTest(ext):
                path = os.path.join(self.ASSETS, f"test_hierarchy.legacy_parent_node.{ext}")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    rig = HierarchyData.load(path)
                # a load reads the old name, and warns of nothing
                self.assertEqual([w for w in caught if "parent_node" in str(w.message)], [])
                self.assertIs(type(rig), HierarchyData)
                self._assert_read_as_938ae44_read(rig, ext)

    def test_a_clip(self):
        from cgmath.hierarchy import ClipData

        clip = ClipData.load(os.path.join(self.ASSETS, "test_hierarchy.legacy_parent_node_clip.json"))
        self.assertIs(type(clip), ClipData)
        self.assertEqual((clip.frame_count, clip.frame, clip.start_frame), (3, 1, 10))
        self.assertEqual([(n.name, n.parent_name) for n in clip], _LEGACY_PARENTS)
        frames = clip.frames.world_matrix
        for frame in range(3):
            clip.frame = frame
            np.testing.assert_allclose(frames[frame], clip.world_matrix, atol=1e-12)

    def test_a_tree_by_hand(self):
        # a dict of node dicts, as a tool of one's own builds it, by name and
        # by uuid; "parent_node" and "parent_uuid" both given is refused
        root_id = generate_uuid()
        nodes = {
            "kid":  {"name": "kid", "parent_node": "root"},
            "root": {"name": "root", "uuid": root_id, "translate": [1.0, 0.0, 0.0]},
            "arm":  {"name": "arm", "parent_node": root_id},
            "lost": {"name": "lost", "parent_node": "nobody"},
        }
        rig = HierarchyData.from_dict(nodes)
        self.assertEqual([n.parent_name for n in rig], ["root", None, "root", None])
        self.assertEqual(rig["kid"].parent_uuid, root_id)

        node = TransformData.from_dict({"name": "alone", "parent_node": "root"})
        self.assertIsNone(node.parent_uuid)  # no hierarchy to look in
        node = TransformData.from_dict({"name": "alone", "parent_node": root_id})
        self.assertEqual(node.parent_uuid, root_id)

        with self.assertRaisesRegex(
            ValueError,
            r"^TransformData\.parent_uuid is given twice in the data, as 'parent_node' and 'parent_uuid'$",
        ):
            TransformData.from_dict({"name": "x", "parent_node": root_id, "parent_uuid": root_id})


SKELETON = os.path.join(
    os.path.dirname(__file__), "test_assets", "test_hierarchy.skeleton.json"
)


def _hard_rig(count=30, seed=7, order=None) -> HierarchyData:
    """every rotate order, joint orients, rotate axes, non unit scale and
    segment scale compensation mixed, a branching tree parented by uuid"""
    rng   = np.random.default_rng(seed)
    nodes = []
    for i in range(count):
        nodes.append(
            TransformData(
                name=f"j{i}",
                node_type="joint",
                uuid=str(uuid_module.UUID(int=int(rng.integers(1, 2**62)))).upper(),
                parent_uuid=None if i == 0 else nodes[int(rng.integers(0, i))].uuid,
                translate=rng.normal(size=3),
                rotate=rng.uniform(-180.0, 180.0, size=3),
                scale=rng.uniform(0.6, 1.6, size=3),
                rotate_axis=rng.uniform(-40.0, 40.0, size=3),
                joint_orient=rng.uniform(-90.0, 90.0, size=3),
                rotate_order=i % 6 if order is None else order,
                segment_scale_compensate=bool(rng.integers(0, 2)),
            )
        )
    return HierarchyData(nodes)


def _channels(rig) -> dict:
    channels     = ("scale", "rotate", "translate", "rotate_axis", "joint_orient")
    out          = {c: np.array([getattr(node, c) for node in rig]) for c in channels}
    out["world"] = np.array([node.world_matrix for node in rig])
    return out


class TestBatchedSwapaxes(unittest.TestCase):
    """a list's swapaxes solves every node at once; it must give what
    swapping the nodes one at a time gives"""

    SWAPS = ((0, 1, False), (0, 1, True), (2, 0, False), (1, 2, True), (2, 1, True))

    def _one_at_a_time(self, nodes, axis0, axis1, negate):
        for node in nodes:
            node.swapaxes(axis0, axis1, negate=negate)

    def test_a_whole_rig(self):
        for seed in (7, 8):
            for axis0, axis1, negate in self.SWAPS:
                with self.subTest(seed=seed, axes=(axis0, axis1, negate)):
                    single, batch = _hard_rig(seed=seed), _hard_rig(seed=seed)
                    self._one_at_a_time(single[:], axis0, axis1, negate)
                    batch.swapaxes(axis0, axis1, negate=negate)

                    want, got = _channels(single), _channels(batch)
                    for key in want:
                        self.assertLess(np.abs(got[key] - want[key]).max(), 1e-9, key)

    def test_a_shuffled_view(self):
        # the view's order is not the rig's, and a node's parent may or may
        # not be in it
        order = np.random.default_rng(3).permutation(30)[:17]
        single, batch = _hard_rig(), _hard_rig()
        self._one_at_a_time(single[order], 0, 2, True)
        batch[order].swapaxes(0, 2, negate=True)

        want, got = _channels(single), _channels(batch)
        for key in want:
            self.assertLess(np.abs(got[key] - want[key]).max(), 1e-9, key)

    def test_the_test_skeleton(self):
        # joints held at exactly 180 degrees and within 1e-5 of 90: the euler
        # angles may come out on the other branch of the same rotation, so
        # rotate is compared as matrices. matrix_to_euler takes a matrix
        # within float32 epsilon (1.2e-7) of gimbal lock for one exactly in
        # it, which both routes meet a little differently: the bound is that
        # epsilon at the skeleton's reach, about 165 units
        reach = 165.0 * 1.2e-7
        for axis0, axis1, negate in self.SWAPS:
            with self.subTest(axes=(axis0, axis1, negate)):
                single = HierarchyData.load(SKELETON)
                batch  = HierarchyData.load(SKELETON)
                self._one_at_a_time(single[:], axis0, axis1, negate)
                batch.swapaxes(axis0, axis1, negate=negate)

                want, got = _channels(single), _channels(batch)
                self.assertLess(np.abs(got["world"] - want["world"]).max(), reach)
                self.assertLess(
                    np.abs(single.rotate_matrix - batch.rotate_matrix).max(), 1.2e-7
                )
                self.assertLess(np.abs(got["translate"] - want["translate"]).max(), reach)
                for key in ("scale", "rotate_axis", "joint_orient"):
                    self.assertLess(np.abs(got[key] - want[key]).max(), 1e-9, key)

    def test_children_keep_their_world_and_swapped_nodes_permute(self):
        rig    = _hard_rig()
        before = np.array([node.world_matrix for node in rig])
        picked = [3, 5, 11]
        rig[picked].swapaxes(1, 2, negate=True)

        P         = np.eye(4)
        P[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        after     = np.array([node.world_matrix for node in rig])
        for i in range(len(rig)):
            want = P @ before[i] if i in picked else before[i]
            self.assertLess(np.abs(after[i] - want).max(), 1e-9, rig[i].name)

    def test_bad_axes_change_nothing(self):
        rig    = _hard_rig()
        before = _channels(rig)
        with self.assertRaises(ValueError):
            rig.swapaxes(1, 1)
        with self.assertRaises(ValueError):
            rig[2:].swapaxes(0, 3)
        for key, values in _channels(rig).items():
            self.assertTrue(np.array_equal(values, before[key]), key)


class TestSwapaxesUnderAMirror(unittest.TestCase):
    """a child is put back through the matrix setter, which puts a mirror on
    scale z: under a parent chain mirroring another axis its scale signs
    move, and a segment scale compensated child of it moves too. The list
    method gives what the per node method run parent first gives, that
    exception included"""

    def _rig(self, mirror):
        ids = [generate_uuid() for _ in range(4)]
        return HierarchyData(
            [
                TransformData("g", node_type="joint", uuid=ids[0], rotate=(10, 20, 30)),
                TransformData("a", node_type="joint", uuid=ids[1], parent_uuid=ids[0], scale=mirror,
                              translate=(0, 1, 0), rotate=(-20, 5, 40)),
                TransformData("c", node_type="joint", uuid=ids[2], parent_uuid=ids[1], translate=(0, 1, 0),
                              segment_scale_compensate=True, rotate=(5, 5, 5)),
                TransformData("d", node_type="joint", uuid=ids[3], parent_uuid=ids[2], translate=(1, 0, 0)),
            ]
        )

    def test_the_list_swap_is_the_parent_first_node_swap(self):
        for mirror in ((1.2, -0.8, 1.5), (-1.2, 0.8, 1.5), (1.2, 0.8, -1.5)):
            with self.subTest(mirror=mirror):
                batch, single = self._rig(mirror), self._rig(mirror)
                TransformList([batch["g"], batch["a"]]).swapaxes(0, 1, negate=True)
                for name in ("g", "a"):
                    single[name].swapaxes(0, 1, negate=True)
                np.testing.assert_allclose(
                    batch.world_matrix, single.world_matrix, rtol=0.0, atol=1e-12
                )

    def test_a_mirror_off_z_moves_a_compensated_grandchild(self):
        # documented, not a promise kept: c keeps its world only when a's
        # mirror is already on z
        for mirror, kept in (((1.2, -0.8, 1.5), False), ((1.2, 0.8, -1.5), True)):
            with self.subTest(mirror=mirror):
                rig    = self._rig(mirror)
                before = rig["c"].world_matrix
                rig["g"].swapaxes(0, 1, negate=True)
                self.assertEqual(np.allclose(rig["c"].world_matrix, before, atol=1e-9), kept)


class TestSetParentCost(unittest.TestCase):
    """set_parent over a whole rig is one pass over it: it built the index
    once a node, 2000 builds and 1.7 s at 2000 joints"""

    def _counts(self, h, edit):
        from cgmath.hierarchy.hierarchy import _RigFrames, _RigIndex

        build, frames = _RigIndex.build.__func__, _RigFrames.__init__
        counts = {"builds": 0, "frames": 0}

        def counted_build(cls, hierarchy):
            counts["builds"] += 1
            return build(cls, hierarchy)

        def counted_frames(self, *args, **kwargs):
            counts["frames"] += 1
            return frames(self, *args, **kwargs)

        with mock.patch.object(_RigIndex, "build", classmethod(counted_build)), mock.patch.object(
            _RigFrames, "__init__", counted_frames
        ):
            edit(h)
        return counts

    def _rig(self, count, scaled):
        from cgmath.hierarchy import ClipData

        rng   = np.random.default_rng(count)
        nodes = []
        for i in range(count):
            nodes.append(
                TransformData(
                    f"j{i}",
                    node_type="joint",
                    uuid=generate_uuid(),
                    parent_uuid=nodes[int(rng.integers(max(0, i - 5), i))].uuid if i else None,
                    translate=rng.normal(size=3),
                    scale=rng.uniform(0.7, 1.4, 3) if scaled else None,
                    segment_scale_compensate=bool(scaled and i % 2),
                )
            )
        # the clip copies the pose's nodes: a second HierarchyData of the
        # same nodes would claim them and leave the pose empty
        pose = HierarchyData(nodes)
        return pose, ClipData(pose, frames=3)

    def test_the_cost_does_not_grow_with_the_rig(self):
        edits = {
            "whole rig to the world": lambda h: h.set_parent(None),
            "under a root": lambda h: h[1:].set_parent("j0"),
            "local": lambda h: h[1:].set_parent("j0", world_space=False),
        }
        for scaled in (False, True):
            for label, edit in edits.items():
                with self.subTest(label, scaled=scaled):
                    counts = []
                    for count in (150, 300):
                        pose, clip = self._rig(count, scaled)
                        self.assertEqual((len(pose), len(clip)), (count, count))
                        counts.append((self._counts(pose, edit), self._counts(clip, edit)))
                    self.assertEqual(counts[0], counts[1])
                    for pair in counts[1]:
                        self.assertLessEqual(pair["builds"], 5)
                        self.assertLessEqual(pair["frames"], 1)

    def test_one_at_a_time_down_a_chain_grows_with_the_chain(self):
        # a chain of transforms with non-uniform scale and no compensation
        # shears, so set_parent goes one node at a time. Leaf first, each
        # node's world was worked out again from the root, and so was the
        # chain's end for each leaf hanging off it when the selection went
        # back up the chain between leaves: 46,346 and 48,140 rows through
        # the local factors at 300 nodes, 182,696 and 186,290 at 600. Now
        # 1,497 and 2,992, then 2,997 and 5,992, and 1,497 and 2,997 in list
        # order, which took 1,795 and 3,595
        from cgmath.hierarchy import ClipData
        from cgmath.hierarchy import hierarchy as module

        real = module._factors

        def composed(count, order):
            rng   = np.random.default_rng(1)
            uuids = [generate_uuid() for _ in range(2 * count)]
            nodes = [
                TransformData(
                    f"j{i}",
                    uuid        = uuids[i],
                    parent_uuid = uuids[i - 1] if i else None,
                    translate   = rng.normal(size=3),
                    rotate      = rng.uniform(-40.0, 40.0, 3),
                    scale       = rng.uniform(0.9, 1.1, 3),
                )
                for i in range(count)
            ]
            leaves = [
                TransformData(f"l{i}", uuid=uuids[count + i], parent_uuid=uuids[count - 1])
                for i in range(count)
            ]
            clip  = ClipData(HierarchyData(nodes + leaves), frames=3)
            chain = list(clip.list[1:count])
            pick = {
                "leaf first":            chain[::-1],
                "list order":            chain,
                "up between the leaves": [n for pair in zip(chain, clip.list[count + 1 :]) for n in pair],
            }[order]
            rows = [0]

            def counted(channels, *args):
                rows[0] += channels["_scale"].shape[1]
                return real(channels, *args)

            with mock.patch.object(module, "_factors", counted):
                TransformList(pick).set_parent(None)
            return rows[0], len(pick)

        for order in ("leaf first", "list order", "up between the leaves"):
            with self.subTest(order=order):
                (small, picked), (large, _) = composed(300, order), composed(600, order)
                self.assertLessEqual(small, 7 * picked)
                self.assertLessEqual(large, 2 * small + 8)

    def test_one_at_a_time_walks_each_chain_once_for_its_products(self):
        # j1 and the leaves hanging off the end of a sheared chain: each
        # leaf's world at its turn is the chain from it up to j2 under j1's
        # new local, a product the union-find hands back. Pointing every node
        # a walk passes at the top, the leaves after the first take two steps
        # each: 446 and 896 steps at 300 and 600 nodes. Pointing only the
        # leaf asked, each leaf walks the whole chain again: 22,200 and
        # 89,400 steps, the products as many
        from cgmath.hierarchy import ClipData
        from cgmath.hierarchy import hierarchy as module

        real = module._compress

        def steps(count):
            rng   = np.random.default_rng(4)
            uuids = [generate_uuid() for _ in range(count)]
            half  = count // 2
            nodes = [
                TransformData(
                    f"j{i}",
                    uuid        = uuids[i],
                    parent_uuid = None if i == 0 else uuids[min(i - 1, half - 1)],
                    translate   = rng.normal(size=3),
                    rotate      = rng.uniform(-40.0, 40.0, 3),
                    scale       = rng.uniform(0.8, 1.25, 3),
                )
                for i in range(count)
            ]
            clip   = ClipData(HierarchyData(nodes), frames=3)
            walked = [0]

            def counted(link, reach, node):
                up = node
                while link.get(up, up) != up:
                    walked[0] += 1
                    up = link[up]
                return real(link, reach, node)

            with mock.patch.object(module, "_compress", counted):
                TransformList([clip.list[1]] + clip.list[half:]).set_parent(None)
            return walked[0]

        small, large = steps(300), steps(600)
        self.assertLessEqual(small, 4 * 150)
        self.assertLessEqual(large, 2 * small + 8)

    def test_the_cascade_check_climbs_each_chain_once(self):
        # the leaves alone off the end of a sheared chain: each leaf's climb
        # stops where an earlier one went, so _cascades reads the parents
        # once a leaf, 300 and 600 times at 300 and 600 nodes, pose or clip.
        # Climbing each leaf's chain again read them 22,650 and 90,300 times.
        # No picked node hangs below another there, so _cascades answers
        # before _carried: with j1 picked too, j0 unscaled and the joints
        # 0.001 apart, j1 is a top by its round-off alone and every leaf
        # drops a shear, so each of four calls to _carried (the change alone
        # first, then weighed by the gain of each leaf's split three ways)
        # climbs every leaf once, 3,593 and 7,193 reads. Climbing each leaf's
        # chain again in _carried read them 137,258 and 544,508 times, and
        # the leaves alone did not see it
        from cgmath.hierarchy import ClipData
        from cgmath.hierarchy import hierarchy as module

        real, carried = module._cascades, module._carried

        def reads(count, clip, with_j1):
            rng   = np.random.default_rng(2)
            uuids = [generate_uuid() for _ in range(count)]
            half  = count // 2
            size  = 0.001 if with_j1 else 1.0
            nodes = [
                TransformData(
                    f"j{i}",
                    uuid        = uuids[i],
                    parent_uuid = None if i == 0 else uuids[min(i - 1, half - 1)],
                    translate   = rng.normal(size=3) * size,
                    rotate      = rng.uniform(-40.0, 40.0, 3),
                    scale       = (1.0, 1.0, 1.0) if with_j1 and i == 0 else rng.uniform(0.8, 1.25, 3),
                )
                for i in range(count)
            ]
            h     = HierarchyData(nodes)
            h     = ClipData(h, frames=3) if clip else h
            pick  = ([h.list[1]] if with_j1 else []) + h.list[half:]
            count = [0, 0]

            class Counted(np.ndarray):
                def __getitem__(self, key):
                    count[0] += 1
                    return super().__getitem__(key)

            def counted(frames, rows, parents, *args):
                return real(frames, rows, parents.view(Counted), *args)

            def climbed(*args):
                count[1] += 1
                return carried(*args)

            with mock.patch.object(module, "_cascades", counted), mock.patch.object(module, "_carried", climbed):
                TransformList(pick).set_parent(None)
            return count

        for with_j1 in (False, True):
            for clip in (False, True):
                with self.subTest(clip=clip, with_j1=with_j1):
                    (small, calls), (large, _) = reads(300, clip, with_j1), reads(600, clip, with_j1)
                    self.assertLessEqual(small, (28 if with_j1 else 5) * 150)
                    self.assertLessEqual(large, 2 * small + 8)
                    self.assertEqual(calls > 0, with_j1)


class TestSetRotateOrder(unittest.TestCase):
    """set_rotate_order keeps the pose, picking the euler solution Maya's
    xform -rotateOrder <order> -preserve picks, on all 2550 joints of the
    Maya fixture (_maya_rotate_order); the rotate_order setter stays raw, as
    Maya's setAttr is"""

    ORDERS = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")

    @classmethod
    def setUpClass(cls):
        cls.cases = maya_rotate_order_cases()

    def _rig(self, cases):
        return HierarchyData(
            [
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    rotate       = cases["rotate"][i],
                    rotate_order = int(cases["before"][i]),
                    joint_orient = cases["joint_orient"][i],
                )
                for i in range(len(cases["rotate"]))
            ]
        )

    def _assert_pose_kept(self, before, after):
        # the 8 joints that start and end 1e-7 from gimbal lose up to 3.1e-9
        # of the pose in the new decomposition, in Maya and in euler_reorder
        # alike; every other joint keeps it to round-off
        twice = self.cases["twice"]
        moved = np.abs(after - before).max(axis=(1, 2))
        self.assertLess(moved[~twice].max(), 1e-12)
        self.assertLess(moved[twice].max(), 1e-8)

    def test_every_case_on_a_pose(self):
        cases = self.cases
        rotate, before, after = [], [], []
        for i in range(len(cases["rotate"])):
            node = TransformData(
                "joint",
                node_type    = "joint",
                rotate       = cases["rotate"][i],
                rotate_order = int(cases["before"][i]),
                joint_orient = cases["joint_orient"][i],
            )
            HierarchyData([node])
            before.append(node.world_matrix)
            node.set_rotate_order(self.ORDERS[cases["after"][i]])

            self.assertEqual(node.rotate_order, cases["after"][i])
            self.assertTrue(np.array_equal(node.joint_orient, cases["joint_orient"][i]))
            rotate.append(node.rotate)
            after.append(node.world_matrix)

        rotate, before, after = np.array(rotate), np.array(before), np.array(after)
        self.assertLess(np.abs(rotate - cases["maya"]).max(), 1e-9)
        self._assert_pose_kept(before, after)

        # Maya's own world matrix moves by up to 4.3e-9 on the 120 joints
        # whose middle axis starts 1e-7 from gimbal
        near = cases["near_gimbal"]
        off  = np.abs(after - cases["world_after"]).max(axis=(1, 2))
        self.assertLess(off[~near].max(), 1e-12)
        self.assertLess(off[near].max(), 1e-8)

    def test_a_whole_rig_one_order_per_node(self):
        cases = self.cases
        rig   = self._rig(cases)
        world = np.array([node.world_matrix for node in rig])
        self.assertLess(np.abs(world - cases["world_before"]).max(), 1e-12)

        rig.set_rotate_order(cases["after"])

        self.assertTrue(np.array_equal(rig.rotate_order, cases["after"]))
        self.assertLess(np.abs(rig.rotate - cases["maya"]).max(), 1e-9)
        self._assert_pose_kept(world, np.array([node.world_matrix for node in rig]))
        self.assertTrue(np.array_equal(rig.joint_orient, cases["joint_orient"]))

    def test_a_view(self):
        cases  = self.cases
        rig    = self._rig(cases)
        before = rig.rotate.copy()
        picked = np.arange(0, len(rig), 7)

        rig[picked].set_rotate_order([self.ORDERS[o] for o in cases["after"][picked]])

        rest = np.setdiff1d(np.arange(len(rig)), picked)
        self.assertTrue(np.array_equal(rig.rotate[rest], before[rest]))
        self.assertTrue(np.array_equal(rig.rotate_order[rest], cases["before"][rest]))
        self.assertLess(np.abs(rig.rotate[picked] - cases["maya"][picked]).max(), 1e-9)

    def test_children_keep_their_world(self):
        rig    = _hard_rig()
        before = np.array([node.world_matrix for node in rig])
        rig.set_rotate_order("zyx")
        self.assertTrue(np.array_equal(rig.rotate_order, [5] * len(rig)))
        after = np.array([node.world_matrix for node in rig])
        self.assertLess(np.abs(after - before).max(), 1e-12)

    def test_the_order_it_has_changes_nothing(self):
        node = TransformData("j", rotate=[400.0, 30.0, -200.0], rotate_order=2)
        HierarchyData([node])
        node.set_rotate_order("zxy")
        self.assertTrue(np.array_equal(node.rotate, [400.0, 30.0, -200.0]))

    def test_an_order_is_a_number_or_its_axes(self):
        node = TransformData("j", rotate=[10.0, 20.0, 30.0])
        HierarchyData([node])
        node.set_rotate_order(np.int64(4))
        self.assertEqual(node.rotate_order, 4)
        node.set_rotate_order("ZXY")
        self.assertEqual(node.rotate_order, 2)
        for bad in (6, -1, 2.5, "xxy", "", None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                node.set_rotate_order(bad)

        rig = _hard_rig(count=4)
        with self.assertRaises(ValueError):
            rig.set_rotate_order([1, 2])
        with self.assertRaises(ValueError):
            rig.set_rotate_order([1, 2, 3, 9])
        self.assertTrue(np.array_equal(rig.rotate_order, [0, 1, 2, 3]))

    def test_the_rotate_order_setter_stays_raw(self):
        # as Maya's setAttr: rotate keeps its angles, so the joint turns
        node = TransformData("j", rotate=[10.0, 20.0, 30.0])
        HierarchyData([node])
        world             = node.world_matrix
        node.rotate_order = 5
        self.assertTrue(np.array_equal(node.rotate, [10.0, 20.0, 30.0]))
        self.assertGreater(np.abs(node.world_matrix - world).max(), 1e-3)
