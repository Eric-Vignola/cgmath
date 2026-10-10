import copy
import itertools
import os
import pickle
import random
import sys
import tempfile
import unittest
import uuid as uuid_module
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
            node.parent_node = 'new_root'

        self.assertTrue(np.allclose(self.results['new_parent_world'], skeleton.matrix))


        # test new parent local
        skeleton = HierarchyData.load(self.skeleton_file)
        obj      = TransformData(name='new_root', rotate=np.array([30,-45,257]))
        skeleton.insert(0, obj)
        for node in skeleton.get_roots()[1:]:
            node.set_parent(obj, world_space=False)
            node.parent_node = 'new_root'

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
        # set_parent(world_space=False) only changes parent_node; any
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
        source = HierarchyData([
            TransformData(name=f'j{i}',
                          parent_node=None if i == 0 else f'j{i - 1}')
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
                    rig = HierarchyData([
                        TransformData('a', node_type='joint', rotate=[10.0, 20.0, 30.0],
                                      scale=parent_scale, joint_orient=[5.0, -10.0, 15.0]),
                        TransformData('b', node_type='joint', parent_node='a',
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
                scale     = [1.5, 0.7, 2.0],
                rotate    = [10.0, 20.0, -35.0],
                translate = [1.0, 2.0, 3.0],
            )
            child = TransformData(
                name='c',
                node_type='joint',
                parent_node='p',
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
                node.parent_node = hierarchy[i - 1].uuid
            hierarchy.append(node)
            node.translate = np.array([offset if i else 0.0, 0.0, 0.0])
        return hierarchy

    def test_view_covers_every_settable_transform_property(self):
        # a TransformList is the vectorized view of a hierarchy, so every
        # channel settable on a node must be settable on the view
        settable = {
            name
            for name, value in vars(TransformData).items()
            if isinstance(value, property) and value.fset is not None
        }
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
                node.parent_node = skeleton[0].uuid
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
        # copying a branch out of a rig leaves parent_node naming a joint that
        # is not there; that node is a root of the copy, not an error
        b      = HierarchyData.load(self.skeleton_file)
        branch = HierarchyData([node.copy() for node in b[0].get_branch()[1:]])

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
                node.parent_node = hierarchy[i - 1].uuid
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


def _named_chain() -> HierarchyData:
    """the same chain as _chain, parented by name"""
    return HierarchyData(
        [
            TransformData("root", translate=(1.0, 0.0, 0.0)),
            TransformData("spine", translate=(0.0, 2.0, 0.0), parent_node="root"),
            TransformData("head", translate=(0.0, 1.0, 0.0), parent_node="spine"),
        ]
    )


class TestParentedByName(unittest.TestCase):
    """parent_node may hold a name; children resolve the way get_parent does"""

    def test_children_and_branch_follow_a_name(self):
        rig = _named_chain()
        self.assertIs(rig["spine"].get_parent(), rig["root"])
        self.assertEqual(rig["root"].get_children().name, ["spine"])
        self.assertEqual(rig["root"].get_branch().name, ["root", "spine", "head"])

    def test_a_parent_edit_clears_the_childrens_cached_world_matrix(self):
        rig = _named_chain()
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [1.0, 3.0, 0.0])

        rig["root"].translate = (5.0, 0.0, 0.0)
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [5.0, 3.0, 0.0])

        rig["spine"].rotate = (0.0, 0.0, 90.0)
        np.testing.assert_allclose(
            rig["head"].world_matrix[3, :3], [4.0, 2.0, 0.0], atol=1e-9
        )

    def test_a_shared_name_resolves_to_the_first_node_only(self):
        # get_parent takes the first node of a name; get_children must agree
        first  = TransformData("twin",  translate=(1.0, 0.0, 0.0))
        second = TransformData("twin",  translate=(9.0, 0.0, 0.0))
        child  = TransformData("child", parent_node="twin")
        rig    = HierarchyData([first, second, child])

        self.assertIs(rig["twin"], first)
        self.assertIs(child.get_parent(), first)
        self.assertEqual(first.get_children().name, ["child"])
        self.assertEqual(len(second.get_children()), 0)

    def test_a_uuid_shaped_name_is_matched_as_a_uuid_only(self):
        # a parent_node shaped like a uuid looks for that uuid: a node that
        # merely carries it as a name is not the parent, either way round
        shaped = TransformData(generate_uuid())
        child  = TransformData("child", parent_node=shaped.name)
        rig    = HierarchyData([shaped, child])

        self.assertEqual(rig["child"].get_parent(),  shaped.name)
        self.assertEqual(len(shaped.get_children()), 0)
        self.assertEqual(shaped.get_branch().name,   [shaped.name])

    def test_uuid_and_name_parents_mix(self):
        rig  = _named_chain()
        leaf = TransformData("leaf", parent_node=rig["head"].uuid)
        rig.append(leaf)
        self.assertEqual(rig["root"].get_branch().name, ["root", "spine", "head", "leaf"])

    def test_a_node_taken_out_with_del_or_remove_can_still_be_edited(self):
        # del and remove() leave the node pointing at its old hierarchy, where
        # a child still names it: an edit's branch walk must not trip on that
        for how in ("del", "remove"):
            with self.subTest(how):
                rig  = _named_chain()
                root = rig["root"]
                if how == "del":
                    del rig[0]
                else:
                    rig.remove(root)

                root.translate = (4.0, 0.0, 0.0)
                np.testing.assert_allclose(root.translate, [4.0, 0.0, 0.0])

    def test_set_parent_keeps_a_link_to_a_shared_name(self):
        # set_parent moves the branch to the end of the list. g names `c` and
        # finds the first one, inside the branch; the move must not hand it
        # to the other `c`, which the move leaves first
        rig = HierarchyData(
            [
                TransformData("r"),
                TransformData("p", parent_node="r"),
                TransformData("c", translate=(1.0, 0.0, 0.0), parent_node="p"),
                TransformData("c", translate=(9.0, 0.0, 0.0), parent_node="r"),
                TransformData("g", parent_node="c"),
            ]
        )
        first, g = rig.list[2], rig["g"]
        self.assertIs(g.get_parent(), first)

        rig["p"].set_parent("r", world_space=False)
        self.assertEqual(rig.name, ["r", "c", "p", "c", "g"])
        self.assertIs(g.get_parent(), first)
        np.testing.assert_allclose(g.world_matrix[3, :3], [1.0, 0.0, 0.0])

        # only a shared name is pinned to a uuid: `p` names one node
        self.assertEqual(first.parent_node, "p")
        self.assertEqual(g.parent_node, first.uuid)


class TestParentLookup(unittest.TestCase):
    """_parent_indices and _owning_parent_first look parents up through dicts;
    they must give get_parent's answer for every kind of parent_node"""

    def _rig(self) -> HierarchyData:
        named  = TransformData("named")
        first  = TransformData("twin", translate=(1.0, 0.0, 0.0))
        second = TransformData("twin", translate=(2.0, 0.0, 0.0))
        shaped = TransformData(generate_uuid())   # a name shaped like a uuid
        odd    = TransformData("odd", uuid="j1")  # a uuid not shaped like one
        rig    = HierarchyData([named, first, second, shaped, odd])

        # a name equal to another node's uuid, and a cycle
        rig.append(TransformData(rig["twin"].uuid))
        rig.append(TransformData("cycle_a", parent_node="cycle_b"))
        rig.append(TransformData("cycle_b", parent_node="cycle_a"))
        refs = [
            None, "named", "twin", second.uuid, shaped.name, "j1", "odd",
            first.uuid, generate_uuid(), "ghost", named.uuid.lower(), "cycle_a",
        ]
        for i, ref in enumerate(refs):
            rig.append(TransformData(f"child{i}", parent_node=ref))
        for i, ref in enumerate(refs):
            rig.append(TransformData(f"grandchild{i}", parent_node=f"child{i}"))
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


class TestRenameKeepsChildren(unittest.TestCase):
    """a rename inside a hierarchy keeps every child that names the node"""

    class Walked(list):
        """a node list that counts the passes over it"""

        walks = 0

        def __iter__(self):
            self.walks += 1
            return super().__iter__()

    def _chain(self) -> HierarchyData:
        return HierarchyData(
            [
                TransformData("root", translate=(1.0, 0.0, 0.0)),
                TransformData("a", translate=(0.0, 1.0, 0.0), parent_node="root"),
                TransformData("b", translate=(0.0, 1.0, 0.0), parent_node="a"),
                TransformData("c", translate=(0.0, 1.0, 0.0), parent_node="b"),
            ]
        )

    def test_a_suffix_over_a_branch_keeps_the_branch(self):
        rig = self._chain()
        rig["a"].get_branch().add_suffix("_L")
        self.assertEqual(rig["root"].get_branch().name, ["root", "a_L", "b_L", "c_L"])
        self.assertEqual(rig["b_L"].parent_node, rig["a_L"].uuid)

        # every cached world matrix follows the root, as a fresh copy does
        rig["root"].translate = (5.0, 0.0, 0.0)
        fresh                 = rig.copy()
        for node in rig:
            np.testing.assert_array_equal(node.world_matrix, fresh[node.name].world_matrix)

    def test_a_rename_onto_an_earlier_nodes_name_keeps_the_children(self):
        # the new name resolves to the earlier node: a child left holding a
        # name would follow it, so it holds the renamed node's uuid instead
        rig = HierarchyData(
            [
                TransformData("a_L", translate=(9.0, 0.0, 0.0)),
                TransformData("a", translate=(1.0, 0.0, 0.0)),
                TransformData("b", translate=(0.0, 1.0, 0.0), parent_node="a"),
            ]
        )
        renamed = rig.list[1]
        renamed.add_suffix("_L")

        self.assertIs(rig["b"].get_parent(), renamed)
        np.testing.assert_allclose(rig["b"].world_matrix[3, :3], [1.0, 1.0, 0.0])

    def test_a_child_the_new_name_captures_drops_its_cached_world_matrix(self):
        rig = HierarchyData(
            [
                TransformData("x", translate=(5.0, 0.0, 0.0)),
                TransformData("y", translate=(0.0, 1.0, 0.0), parent_node="ghost"),
            ]
        )
        np.testing.assert_allclose(rig["y"].world_matrix[3, :3], [0.0, 1.0, 0.0])

        rig["x"].name = "ghost"
        self.assertIs(rig["y"].get_parent(), rig["ghost"])
        np.testing.assert_allclose(rig["y"].world_matrix[3, :3], [5.0, 1.0, 0.0])

    def test_strip_namespace_keeps_a_rig_parented_by_name(self):
        rig = HierarchyData(
            [
                TransformData("ns:root", translate=(1.0, 0.0, 0.0)),
                TransformData("ns:a", translate=(0.0, 1.0, 0.0), parent_node="ns:root"),
                TransformData("ns:b", translate=(0.0, 1.0, 0.0), parent_node="ns:a"),
            ]
        )
        rig.strip_namespace()

        self.assertEqual(rig.name, ["root", "a", "b"])
        self.assertEqual(rig["root"].get_branch().name, ["root", "a", "b"])
        np.testing.assert_allclose(rig["b"].world_matrix[3, :3], [1.0, 2.0, 0.0])

    def test_a_suffix_onto_a_name_another_renamed_node_gives_up(self):
        # a takes a_L's name as a_L moves on to a_L_L: renamed a node at a
        # time, a took a_L's children over before a_L was renamed
        rig = HierarchyData(
            [
                TransformData("root"),
                TransformData("a", translate=(1.0, 0.0, 0.0), parent_node="root"),
                TransformData("a_L", translate=(9.0, 0.0, 0.0), parent_node="root"),
                TransformData("x", translate=(0.0, 1.0, 0.0), parent_node="a"),
                TransformData("y", translate=(0.0, 1.0, 0.0), parent_node="a_L"),
            ]
        )
        a, a_l = rig.list[1], rig.list[2]
        _ = [node.world_matrix for node in rig]
        TransformList([a, a_l]).add_suffix("_L")

        self.assertEqual(rig.name, ["root", "a_L", "a_L_L", "x", "y"])
        self.assertIs(rig["x"].get_parent(), a)
        self.assertIs(rig["y"].get_parent(), a_l)
        np.testing.assert_allclose(rig["y"].world_matrix[3, :3], [9.0, 1.0, 0.0])

    def test_strip_namespace_moving_names_up_one_level_keeps_the_children(self):
        # "A:A:x" becomes "A:x" as "A:x" becomes "x": each keeps its own child
        rig = HierarchyData(
            [
                TransformData("A:A:x", translate=(1.0, 0.0, 0.0)),
                TransformData("A:x", translate=(9.0, 0.0, 0.0)),
                TransformData("inner", translate=(0.0, 1.0, 0.0), parent_node="A:A:x"),
                TransformData("outer", translate=(0.0, 1.0, 0.0), parent_node="A:x"),
            ]
        )
        _ = [node.world_matrix for node in rig]
        rig.strip_namespace("A")

        self.assertEqual(rig.name, ["A:x", "x", "inner", "outer"])
        self.assertIs(rig["inner"].get_parent(), rig["A:x"])
        self.assertIs(rig["outer"].get_parent(), rig["x"])
        np.testing.assert_allclose(rig["outer"].world_matrix[3, :3], [9.0, 1.0, 0.0])

    def test_a_captured_child_drops_its_renamed_childrens_cached_world(self):
        # s_L captures c, renamed in the same call: the walk that clears c's
        # branch has to find d, which named c by its old name
        rig = HierarchyData(
            [
                TransformData("s", translate=(5.0, 0.0, 0.0)),
                TransformData("c", translate=(0.0, 1.0, 0.0), parent_node="s_L"),
                TransformData("d", translate=(0.0, 1.0, 0.0), parent_node="c"),
            ]
        )
        _ = [node.world_matrix for node in rig]
        TransformList([rig["s"], rig["c"]]).add_suffix("_L")

        self.assertIs(rig["c_L"].get_parent(), rig["s_L"])
        self.assertIs(rig["d"].get_parent(), rig["c_L"])
        np.testing.assert_allclose(rig["d"].world_matrix[3, :3], [5.0, 2.0, 0.0])

    def test_a_rename_of_the_second_node_of_a_name_leaves_the_first_its_children(self):
        # c follows the first "a": renaming the second neither hands c to it
        # nor pins c, whose link the rename did not touch
        rig = HierarchyData(
            [
                TransformData("a", translate=(1.0, 0.0, 0.0)),
                TransformData("a", translate=(9.0, 0.0, 0.0)),
                TransformData("c", translate=(0.0, 1.0, 0.0), parent_node="a"),
            ]
        )
        first, second = rig.list[0], rig.list[1]
        _           = [node.world_matrix for node in rig]
        second.name = "b"

        self.assertIs(rig["c"].get_parent(), first)
        self.assertEqual(rig["c"].parent_node, "a")
        np.testing.assert_allclose(rig["c"].world_matrix[3, :3], [1.0, 1.0, 0.0])

    def test_a_rename_ahead_of_a_parent_takes_its_children_and_clears_them(self):
        # x takes "a" ahead of the node carrying it, so y now follows x: y
        # and z below it drop their cached world matrices
        rig = HierarchyData(
            [
                TransformData("x", translate=(5.0, 0.0, 0.0)),
                TransformData("a", translate=(9.0, 0.0, 0.0)),
                TransformData("y", translate=(0.0, 1.0, 0.0), parent_node="a"),
                TransformData("z", translate=(0.0, 1.0, 0.0), parent_node="y"),
            ]
        )
        renamed      = rig.list[0]
        _            = [node.world_matrix for node in rig]
        renamed.name = "a"

        self.assertIs(rig["y"].get_parent(), renamed)
        np.testing.assert_allclose(rig["y"].world_matrix[3, :3], [5.0, 1.0, 0.0])
        np.testing.assert_allclose(rig["z"].world_matrix[3, :3], [5.0, 2.0, 0.0])

    def test_a_view_over_two_rigs_and_a_loose_node_renames_each(self):
        rigs  = [self._chain(), self._chain()]
        loose = TransformData("loose")
        TransformList([rigs[0]["a"], rigs[1]["b"], loose]).add_suffix("_L")

        self.assertEqual(loose.name, "loose_L")
        self.assertEqual(rigs[0]["root"].get_branch().name, ["root", "a_L", "b", "c"])
        self.assertEqual(rigs[1]["root"].get_branch().name, ["root", "a", "b_L", "c"])

    def test_a_rename_over_a_rig_walks_it_as_often_whatever_its_size(self):
        # renamed a node at a time, every rename walked the whole rig for the
        # children naming the node: a bulk rename cost N^2 in nodes
        def walks(count: int, by_name: bool) -> list:
            uuids   = [generate_uuid() for _ in range(count)]
            names   = [f"ns:j{i}" for i in range(count)]
            parents = [None] + (names if by_name else uuids)[:-1]
            rig = HierarchyData(
                [
                    TransformData(name, uuid=uuid, parent_node=parent)
                    for name, uuid, parent in zip(names, uuids, parents)
                ]
            )
            rig.list = self.Walked(rig.list)
            out      = []
            for rename in (
                lambda: rig.add_suffix("_L"),
                lambda: rig.add_prefix("L_"),
                lambda: rig.strip_namespace(),
            ):
                before = rig.list.walks
                rename()
                out.append(rig.list.walks - before)

            self.assertEqual(rig.name, [f"L_j{i}_L" for i in range(count)])
            self.assertEqual(len(rig[0].get_branch()), count)
            return out

        for by_name in (False, True):
            self.assertEqual(walks(10, by_name), walks(200, by_name))

    def test_a_rename_capturing_a_rig_walks_it_as_often_whatever_its_size(self):
        # every child names a node the rig lacks until the rename: clearing
        # each captured child's branch on a walk of its own scanned the rig
        # once per node below it, which cost N^2 in nodes
        renames = (
            ("j{}", "j{}_L", lambda rig: rig.add_suffix("_L")),
            ("j{}", "L_j{}", lambda rig: rig.add_prefix("L_")),
            ("ns:j{}", "j{}", lambda rig: rig.strip_namespace()),
        )

        def walks(count: int) -> list:
            out = []
            for name, parent, rename in renames:
                rig = HierarchyData(
                    [
                        TransformData(
                            name.format(i),
                            translate   = (0.0, 1.0, 0.0),
                            parent_node = parent.format(i - 1) if i else None,
                        )
                        for i in range(count)
                    ]
                )
                _        = [node.world_matrix for node in rig]
                rig.list = self.Walked(rig.list)
                rename(rig)
                out.append(rig.list.walks)

                # the rename captures the whole chain, read as a fresh copy is
                self.assertEqual(len(rig[0].get_branch()), count)
                fresh = rig.copy()
                for node, copied in zip(rig, fresh):
                    np.testing.assert_array_equal(node.world_matrix, copied.world_matrix)
            return out

        self.assertEqual(walks(10), walks(200))


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
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [0.0, 1.0, 0.0])

    def test_pop_orphans_children_parented_by_name(self):
        rig = _named_chain()
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [1.0, 3.0, 0.0])
        rig.pop(1)
        self.assertIsNone(rig["head"].parent_node)
        np.testing.assert_allclose(rig["head"].world_matrix[3, :3], [0.0, 1.0, 0.0])

    def test_nodes_built_from_one_attribute_dict_do_not_share_it(self):
        attrs  = {"hero": {"attributeType": "double", "value": 1.0, "keyable": True}}
        p      = TransformData("p", user_defined_attributes=attrs)
        q      = TransformData("q", user_defined_attributes=attrs)
        p.hero = 3.0
        self.assertEqual(q.hero, 1.0)
        self.assertEqual(attrs["hero"]["value"], 1.0)


def _mixed_rig() -> HierarchyData:
    """a branching rig parented by uuid and by name, with a name two nodes
    share: root > spine > neck > head > tip, spine > twin > hand, root > twin
    > toe. neck, hand and tip name their parent; "twin" resolves to the first
    twin"""
    rng = np.random.default_rng(23)
    specs = [
        ("root", None),
        ("spine", "root"),
        ("neck", "spine"),
        ("head", "neck"),
        ("twin", "spine"),
        ("hand", "twin"),
        ("twin", "root"),
        ("toe", -1),
        ("tip", "head"),
    ]
    by_name = ("neck", "hand", "tip")
    nodes   = []
    for name, parent in specs:
        if parent == -1:
            ref = nodes[-1].uuid  # the second twin, which a name can not reach
        elif parent is None or name in by_name:
            ref = parent
        else:
            ref = next(n for n in nodes if n.name == parent).uuid
        nodes.append(
            TransformData(
                name=name,
                uuid=generate_uuid(),
                parent_node=ref,
                translate=rng.normal(size=3),
                rotate=rng.uniform(-90.0, 90.0, size=3),
                scale=rng.uniform(0.7, 1.4, size=3),
                rotate_order=int(rng.integers(0, 6)),
                segment_scale_compensate=bool(rng.integers(0, 2)),
            )
        )
    return HierarchyData(nodes)


def _children_by_scan(node) -> list:
    """get_children's rule, by a scan of every node of the hierarchy"""
    rig = node._hierarchy
    if rig is None:
        return []
    first    = next((n for n in rig.list if n.name == node.name), None)
    children = []
    for other in rig.list:
        ref = other.parent_node
        if ref == node.uuid:
            children.append(other)
        elif ref is not None and ref == node.name and not validate_uuid(ref):
            if first is node:
                children.append(other)
    return children


def _branch_by_scan(node) -> list:
    branch = [node]
    for child in _children_by_scan(node):
        branch.extend(_branch_by_scan(child))
    return branch


class _Walked(list):
    """a node list that counts the passes over it"""

    walks = 0

    def __iter__(self):
        self.walks += 1
        return super().__iter__()


def _swap_twins(rig) -> None:
    first  = rig.index("twin")
    second = [i for i, n in enumerate(rig) if n.name == "twin"][1]
    rig[first], rig[second] = rig[second], rig[first]


def _assert_index_is_a_build(case, rig, label) -> None:
    """the index a rig keeps, patched in place or built, holds what a build
    of the rig as it now stands holds"""
    from cgmath.hierarchy.hierarchy import _RigIndex

    kept = rig._kept_index()
    if kept is None:
        return
    fresh = _RigIndex.build(rig)
    case.assertIsNotNone(fresh, f"{label}: an index kept where a build gives none")
    for table in ("uuids", "names", "shared", "refs"):
        # a long row of refs a patch made a deque holds what a build's list does
        case.assertEqual(
            {key: value if isinstance(value, int) else list(value) for key, value in getattr(kept, table).items()},
            {key: value if isinstance(value, int) else list(value) for key, value in getattr(fresh, table).items()},
            f"{label}: index {table}",
        )


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
    """get_parent's rule, by a scan of every node of the hierarchy"""
    rig, ref = node._hierarchy, node.parent_node
    if rig is None or ref is None:
        return None
    field = "uuid" if validate_uuid(ref) else "name"
    return next((n for n in rig.list if getattr(n, field) == ref), ref)


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
    """``edits`` random edits of every kind on one rig, parented by uuid and
    by name, names shared, captured and given up, parents the rig lacks and
    loops: after each, every lookup the rig answers, through the index it
    keeps patched or built again, equals a scan of the rig, and every world
    matrix, most of them cached before the edit, the one its channels make.
    Returns how often each edit ran"""
    rng  = random.Random(seed)
    made = itertools.count()
    rig  = HierarchyData()

    def name():
        return rng.choice(_FUZZ_NAMES) if rng.random() < 0.5 else f"n{next(made)}"

    def ref():
        roll = rng.random()
        if not len(rig) or roll < 0.15:
            return None
        if roll < 0.5:
            return rng.choice(rig.list).uuid
        if roll < 0.8:
            return rng.choice(rig.list).name
        return rng.choice(_FUZZ_NAMES) if roll < 0.9 else generate_uuid()

    def node():
        return TransformData(
            name(),
            parent_node=ref(),
            translate=[rng.uniform(-1.0, 1.0) for _ in range(3)],
            rotate=[rng.uniform(-90.0, 90.0) for _ in range(3)],
            scale=[rng.uniform(0.7, 1.4) for _ in range(3)],
            rotate_order=rng.randrange(6),
            segment_scale_compensate=rng.random() < 0.3,
        )

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
        # renames in a row, nothing read between: through the index the last
        # edit left, or, after a uuid written retires it, by a pass and then
        # an index built; unique names read now and then
        if rng.random() < 0.5:
            pick().uuid = generate_uuid()
        for _ in range(rng.randint(2, 6)):
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
        # that makes the pass
        if rng.random() < 0.5:
            pick().uuid = generate_uuid()
        if rng.random() < 0.5:
            pick().name = name()
        for _ in range(rng.randint(1, 4)):
            added = node()
            rig.append(added)
            added.name        = name()
            added.parent_node = ref()

    def target():
        roll = rng.random()
        if roll < 0.2:
            return None
        return pick().uuid if roll < 0.6 else pick().name

    def write_world():
        angles   = np.radians([rng.uniform(-180.0, 180.0) for _ in range(3)])
        W        = euler_to_matrix(angles[None], rng.randrange(6))[0]
        W[3, :3] = [rng.uniform(-3.0, 3.0) for _ in range(3)]
        pick().world_matrix = W

    def read():
        for _ in range(rng.randint(1, 6)):
            pick().world_matrix

    kinds = {
        "append": (8, lambda: rig.append(node())),
        "insert": (3, lambda: rig.insert(rng.randint(-len(rig) - 2, len(rig) + 2), node())),
        "inserts in a row": (2, inserts),
        "extend": (2, lambda: rig.extend([node() for _ in range(rng.randint(1, 5))])),
        "pop": (2, lambda: rig.pop(rng.randrange(len(rig)))),
        "del": (1, lambda: rig.__delitem__(rng.randrange(len(rig)))),
        "remove": (1, lambda: rig.remove(pick())),
        "setitem swap": (1, swap),
        "reverse": (1, lambda: rig.reverse()),
        "sort": (1, lambda: rig.sort()),
        "rename": (8, lambda: setattr(pick(), "name", name())),
        "rename a view": (3, rename_view),
        "renames in a row": (3, renames),
        "appended, named, parented": (2, loaded),
        "uuid": (1, lambda: setattr(pick(), "uuid", generate_uuid())),
        "parent_node": (8, lambda: setattr(pick(), "parent_node", ref())),
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


class TestRigIndex(unittest.TestCase):
    """get_children reads the index a hierarchy keeps; every edit that can
    give a node other children has to patch it or retire it"""

    EDITS = {
        "append": lambda rig: rig.append(
            TransformData("extra", parent_node=rig["neck"].uuid, translate=(0, 1, 0))
        ),
        "insert": lambda rig: rig.insert(
            1, TransformData("early", parent_node="head", translate=(1, 0, 0))
        ),
        "extend": lambda rig: rig.extend(
            [TransformData("x0", parent_node="twin"), TransformData("x1", parent_node="x0")]
        ),
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
        "parent_node uuid": lambda rig: setattr(rig["toe"], "parent_node", rig["head"].uuid),
        "parent_node name": lambda rig: setattr(rig["toe"], "parent_node", "neck"),
        "parent_node none": lambda rig: setattr(rig["spine"], "parent_node", None),
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
            with self.subTest(label):
                rig = _mixed_rig()
                self._assert_current(rig, label + " (before)")

                # the index built just ahead of the edit, which then has to
                # patch it or retire it on its own
                rig._rig_index()
                edit(rig)
                _assert_index_is_a_build(self, rig, label)

                # an edit that moves a node to another parent leaves the
                # cached world matrices it moved stale: that is not the map's
                # to fix, so they are dropped here, and what is checked is
                # that every later write reaches the branch the edit made
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
        a   = TransformData("a", parent_node=b.uuid, translate=(0.0, 1.0, 0.0))
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
            rig.append(TransformData(f"leaf{i}", parent_node=rig["head"].uuid))
        with mock.patch.object(_RigIndex, "build", classmethod(counted)):
            for _ in range(30):
                rig.pop()
            self.assertEqual(len(builds), 0)
            rig.clear()
        self.assertEqual(len(rig), 0)
        self.assertLessEqual(len(builds), 1)

    def test_a_pop_orphans_the_children_it_finds_by_scan(self):
        # the index retired by an edit, the children found by a scan: by
        # uuid and by name alike
        rig             = _mixed_rig()
        rig["toe"].name = "foot"  # retires the index
        self.assertFalse(rig._rig_index_is_current())
        twin = rig.index("twin")
        kids = _children_by_scan(rig[twin])
        rig.world_matrix
        rig.pop(twin)
        self.assertTrue(kids)
        self.assertTrue(all(n.parent_node is None for n in kids))
        np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)

    def test_a_name_moves_children_to_the_node_now_first(self):
        rig  = _mixed_rig()
        hand = rig["hand"]
        self.assertIs(hand.get_parent(), rig["twin"])
        self.assertEqual(rig["twin"].get_children().name, ["hand"])

        rig["root"].name = "twin"
        self.assertIs(hand.get_parent(), rig[0])
        self.assertIn("hand", rig[0].get_children().name)
        self.assertNotIn("hand", rig[4].get_children().name)

    def test_the_index_is_kept_until_an_edit(self):
        rig                       = _mixed_rig()
        built                     = rig._rig_index()
        rig["spine"].rotate       = (10.0, 20.0, 30.0)
        rig["spine"].rotate_order = 3
        rig["spine"].get_branch()
        self.assertIs(rig._rig_index(), built)

        # a rename, a node added and a parent changed patch it in place
        rig["neck"].name = "collar"
        rig.append(TransformData("extra", parent_node="collar"))
        rig["toe"].parent_node = rig["tip"].uuid
        self.assertIs(rig._rig_index(), built)
        _assert_index_is_a_build(self, rig, "patched")

        # an edit that reorders the nodes retires it
        rig.sort()
        self.assertIsNot(rig._rig_index(), built)

        # a strip that renames nothing keeps it
        built = rig._rig_index()
        rig.strip_namespace()
        self.assertIs(rig._rig_index(), built)

    def test_a_rename_through_a_view_reaches_the_index(self):
        # a view renames its nodes together, past __setattr__: the child that
        # names a node by the name the rename gives it finds it at once
        rig = HierarchyData(
            [
                TransformData("a"),
                TransformData("b"),
                TransformData("c", parent_node="a_L"),
            ]
        )
        self.assertEqual(rig.get_parents().tolist(), [-1, -1, -1])

        TransformList([rig["a"], rig["b"]]).add_suffix("_L")
        self.assertEqual(rig.get_parents().tolist(),     [-1, -1, 0])
        self.assertEqual(rig["a_L"].get_children().name, ["c"])
        self.assertEqual(rig.unique_name.tolist(),       [True, True, True])

    def test_a_channel_write_does_not_scan_the_hierarchy_per_node(self):
        # get_children used to scan every node, once for each node of the
        # branch, so a write on a root cost the size of the rig squared
        uuids = [generate_uuid() for _ in range(60)]
        rig = HierarchyData(
            [
                TransformData(f"j{i}", uuid=u, parent_node=uuids[i // 2] if i else None)
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
        # with no index kept, an append has none to patch: a write on the
        # node just added learns from one scan that it has no children,
        # where building the index would cost several scans on every write
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
                leaf = TransformData(f"leaf{i}", parent_node=head.uuid)
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

    def test_a_chain_deeper_than_the_recursion_limit(self):
        depth = sys.getrecursionlimit() + 50
        uuids = [generate_uuid() for _ in range(depth)]
        rig = HierarchyData(
            [
                TransformData(
                    f"j{i}",
                    uuid        = u,
                    parent_node = uuids[i - 1] if i else None,
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
        rig                  = HierarchyData([TransformData("a"), TransformData("b", parent_node="a")])
        rig["a"].parent_node = "b"
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
        chain = HierarchyData(
            [
                TransformData(
                    f"j{i}",
                    parent_node = f"j{i - 1}" if i else None,
                    translate   = (0.0, 1.0, 0.0),
                )
                for i in range(depth)
            ]
        )
        for node in chain:
            node.world_matrix
        leaf  = chain[depth - 1]
        other = HierarchyData([TransformData("x"), TransformData("w", parent_node="x")])

        edits = (
            ("a node built", lambda: TransformData("z")),
            ("a rename in another rig", lambda: setattr(other[0], "name", "y")),
            ("a parent_node in another rig", lambda: setattr(other[1], "parent_node", None)),
            ("a node added to another rig", lambda: other.append(TransformData("v"))),
            ("a node of another rig popped", lambda: other.pop()),
        )
        for label, edit in edits:
            with self.subTest(label):
                edit()
                self.assertTrue(all(node._cached_world() is not None for node in chain))
                self.assertEqual(leaf.world_matrix[3, 1], float(depth))

        # a node added to its own rig, which no node names, moves none of it
        chain.append(TransformData("extra"))
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
            parent = TransformData("parent", scale=(1e-4, 1e-4, 1e-4))
            child  = TransformData("child", parent_node="parent", translate=(1.0, 2.0, 3.0))
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
        # ten chains 15 deep under a root, parented by uuid and by name. a
        # rename keeps the cached world matrices it does not move, so a loop
        # of renames and reads costs what the renames alone cost; an edit
        # that can move a node retires the rig's cached worlds, and a read
        # after it computes its chain once, its parents read from one index
        # of the rig rather than two scans of the rig a level
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
                made.parent_node = _Counted(parent)
            return made

        def rig(by_name):
            nodes = [counted_node("root")]
            for k in range(chains):
                for j in range(depth):
                    parent = nodes[0] if j == 0 else nodes[-1]
                    nodes.append(counted_node(f"c{k}_{j}", parent.name if by_name else parent.uuid))
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
            node.parent_node = _Counted(str(node.parent_node))
            np.testing.assert_allclose(node.world_matrix[3, 1], depth + 1)

        for by_name in (False, True):
            with self.subTest(by_name=by_name), mock.patch.object(
                TransformData, "matrix", property(counted, matrix.fset)
            ):
                h     = rig(by_name)
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
            "parent_node uuid": lambda node, above: setattr(node, "parent_node", above.uuid),
            "parent_node name": lambda node, above: setattr(node, "parent_node", above.name),
            "set_parent local": lambda node, above: node.set_parent(above.name, world_space=False),
            "set_parent world": lambda node, above: node.set_parent(above),
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
        # node.name = in a loop: each rename scanned the whole rig for the
        # children naming the node, 0.25 s over 2,000 nodes where 1.0.4 took
        # 0.001 s. The children come from the index, which is patched: on a
        # rig in use no pass over the list, no build, and every cached world
        # matrix stands. A rig with no index -- just made, flat, or one a pop
        # retired it in -- makes one pass at the first rename and builds the
        # index at the second, whatever its size
        def renamed(count, by_name, state):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            if state != "flat":
                for above, node in zip(nodes, nodes[1:]):
                    node.parent_node = above.name if by_name else above.uuid
            rig.list = _Walked(rig.list)
            if state in ("in use", "flat", "popped"):
                rig.world_matrix  # its worlds cached, the index built if any parent
            if state == "popped":
                rig.append(TransformData("extra"))
                rig.pop()
            rig.list.walks = 0

            def edit():
                for node in nodes:
                    node.name = node.name + "_x"

            counts          = self._counted(edit)
            counts["walks"] = rig.list.walks
            if state in ("in use", "flat"):
                self.assertTrue(all(node._cached_world() is not None for node in nodes))
            self.assertEqual(rig.name, [f"j{i}_x" for i in range(count)])
            if state != "flat":
                self.assertEqual([n.name for n in nodes[0].get_branch()], rig.name)
                self.assertEqual(nodes[-1].world_matrix[3, 1], float(count))
            return counts, rig

        unread = {"builds": 1, "matrices": 0, "walks": 2}  # a pass, then a build
        for state, want in (
            ("in use", {"builds": 0, "matrices": 0, "walks": 0}),
            ("made", unread),
            ("flat", unread),
            ("popped", unread),
        ):
            for by_name in (False, True):
                with self.subTest(state, by_name=by_name):
                    small, _ = renamed(40, by_name, state)
                    large, rig = renamed(160, by_name, state)
                    self.assertEqual(small, large)
                    self.assertEqual(large, want)
                    self.assertIsNotNone(rig._kept_index())
                    _assert_index_is_a_build(self, rig, f"{state}, by_name={by_name}")

    def test_a_rig_of_one_name_renamed_node_by_node_patches_the_index(self):
        # every node named alike and parented by uuid, then renamed one at a
        # time on a rig in use: no child names the shared name, so a rename
        # changes the name table alone, the next node of the name its first.
        # Each went through the batch rule, 7.9 times 1.0.4's loop at 2000
        # nodes. First to last, last to first or from the middle out: no
        # build, no pass over the list, no batch rule, every world cached
        from cgmath.hierarchy import hierarchy as module

        def renamed(count, order):
            nodes = [TransformData("joint", translate=(0.0, 1.0, 0.0)) for _ in range(count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_node = above.uuid
            rig.list = _Walked(rig.list)
            rig.world_matrix
            built          = rig._rig_index()
            rig.list.walks = 0

            def edit():
                for i in order(count):
                    nodes[i].name = f"joint{i}"

            with mock.patch.object(module, "_rename_in_index", wraps=module._rename_in_index) as batch:
                counts = self._counted(edit)
            counts["walks"] = rig.list.walks
            counts["batch"] = batch.call_count

            self.assertIs(rig._kept_index(), built)
            _assert_index_is_a_build(self, rig, f"{count} nodes")
            self.assertTrue(all(node._cached_world() is not None for node in nodes))
            self.assertEqual(rig.name, [f"joint{i}" for i in range(count)])
            self.assertEqual(rig.unique_name.tolist(), [True] * count)
            self.assertEqual(nodes[-1].world_matrix[3, 1], float(count))
            return counts

        for label, order in (
            ("first to last", lambda count: range(count)),
            ("last to first", lambda count: range(count - 1, -1, -1)),
            ("middle out", lambda count: sorted(range(count), key=lambda i: abs(i - count // 2))),
        ):
            with self.subTest(label):
                small, large = renamed(40, order), renamed(160, order)
                self.assertEqual(small, large)
                self.assertEqual(large, {"builds": 0, "matrices": 0, "walks": 0, "batch": 0})

    def test_parent_node_loops_on_a_rig_in_use_patch_the_index(self):
        # a parent_node written on every node of a rig in use, nothing read
        # between: each patches the index, no build, no pass over the list,
        # and no node's children listed, only asked whether it has any; the
        # worlds read after are the ones the channels make
        from cgmath.hierarchy.hierarchy import _RigIndex

        def chain(count, by_name=False):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_node = above.name if by_name else above.uuid
            return rig

        def star(count):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for node in nodes[1:]:
                node.parent_node = nodes[0].uuid
            return rig

        def flattened(nodes):
            for node in nodes:
                node.parent_node = None

        def leaves_by_name_then_uuid(nodes):
            for node in nodes[1:]:
                node.parent_node = nodes[0].name
            for node in nodes[1:]:
                node.parent_node = nodes[0].uuid

        def under_the_root(nodes):
            for node in nodes[2:]:
                node.parent_node = nodes[0].name

        listed = _RigIndex.children
        for label, make, loop in (
            ("a chain flattened", chain, flattened),
            ("a star's leaves by name, then by uuid", star, leaves_by_name_then_uuid),
            ("a chain by name put under its root", lambda count: chain(count, True), under_the_root),
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
                    if loop is leaves_by_name_then_uuid:
                        # a parent_node naming the parent it had drops nothing
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

    def test_a_rename_after_an_edit_that_retires_the_index_makes_the_pass(self):
        # a rename that follows another edit, rather than a rename, makes the
        # pass over the list it made before: a build costs about three passes
        from cgmath.hierarchy.hierarchy import _RigIndex

        nodes = [TransformData(f"j{i}") for i in range(30)]
        rig   = HierarchyData(nodes)
        for above, node in zip(nodes, nodes[1:]):
            node.parent_node = above.name
        rig.world_matrix

        build  = _RigIndex.build.__func__
        builds = []

        def counted(cls, hierarchy):
            builds.append(1)
            return build(cls, hierarchy)

        with mock.patch.object(_RigIndex, "build", classmethod(counted)):
            for node in nodes:
                node.uuid = generate_uuid()  # retires the index
                node.name = node.name + "_x"
        self.assertEqual(builds, [])
        self.assertEqual([n.name for n in nodes[0].get_branch()], [f"j{i}_x" for i in range(30)])
        _assert_index_is_a_build(self, rig, "uuid, then rename")

    def test_a_rig_named_and_parented_node_by_node_builds_the_index_once(self):
        # renames with appends or parent_node writes between them, nothing
        # read: each append or parent_node retired the index the rename
        # before had left no current one of, so each rename made its pass
        # over the rig, 3.7 and 150 times 1.0.4's loop at 2000 nodes. One
        # pass, one build at the second rename, then each node patches it
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
                        node.parent_node = above.uuid
                    above = node

            return rig, edit

        def reparented(by_name):
            # every node of a rig just made renamed, then parented, the
            # first made a root
            def made(count):
                nodes    = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
                rig      = HierarchyData(nodes)
                rig.list = _Walked(rig.list)

                def edit():
                    for above, node in zip([None] + nodes, nodes):
                        node.name = node.name + "_x"
                        if above is None:
                            node.parent_node = None
                        else:
                            node.parent_node = above.name if by_name else above.uuid

                return rig, edit

            return made

        for label, make, names in (
            ("appended, named, parented", loaded, "j{}"),
            ("renamed, parented by uuid", reparented(False), "j{}_x"),
            ("renamed, parented by name", reparented(True), "j{}_x"),
        ):
            with self.subTest(label):
                counts = []
                for count in (40, 160):
                    rig, edit = make(count)
                    rig.list.walks = 0
                    spent          = self._counted(edit)
                    spent["walks"] = rig.list.walks
                    counts.append(spent)
                    self.assertEqual(rig.name, [names.format(i) for i in range(count)])
                    self.assertEqual(rig.list[0].get_branch().name, rig.name)
                    self.assertEqual(rig.list[-1].world_matrix[3, 1], float(count))
                    _assert_lookups_are_a_scan(self, rig, f"{label}, {count}")
                self.assertEqual(counts[0],             counts[1])
                self.assertEqual(counts[1]["builds"],   1)
                self.assertEqual(counts[1]["matrices"], 0)

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
                node.parent_node = above.uuid
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
                    TransformData(f"k{i}", parent_node=rig.list[at].uuid, translate=(1.0, 0.0, 0.0)),
                )
                rig.list[at + 1].world_matrix

        def front_renamed(rig, count):
            # each node put first, then named: an insert ahead is no edit a
            # rename loop builds the index across, as its shift would cost
            # more than the rename's pass
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
        # a node names the one put in: by name, here a node of a name it now
        # resolves to ahead of the first one, which takes that node's
        # children, or by uuid, a child that named no node until now
        ghost = generate_uuid()
        for label, added, kept in (
            ("named by none", lambda: TransformData("r", translate=(5.0, 0.0, 0.0)), True),
            ("named", lambda: TransformData("p", translate=(5.0, 0.0, 0.0)), False),
            ("named by uuid", lambda: TransformData("r", uuid=ghost, translate=(5.0, 0.0, 0.0)), False),
        ):
            with self.subTest(label):
                rig = HierarchyData(
                    [
                        TransformData("p", translate=(9.0, 0.0, 0.0)),
                        TransformData("z", translate=(0.0, 1.0, 0.0), parent_node="p"),
                        TransformData("w", translate=(0.0, 1.0, 0.0), parent_node="z"),
                        TransformData("x", translate=(0.0, 1.0, 0.0), parent_node=ghost),
                    ]
                )
                rig.world_matrix
                rig.insert(0, TransformData("q"))  # shifts the index the read fetched
                self.assertIsNotNone(rig._kept_index())
                node = added()
                rig.insert(1, node)

                self.assertIsNone(rig._kept_index())
                self.assertEqual([n._cached_world() is not None for n in rig.list[2:]], [kept] * 4)
                by_name, by_uuid = label == "named", label == "named by uuid"
                self.assertIs(rig["z"].get_parent(), node if by_name else rig.list[2])
                np.testing.assert_allclose(rig["w"].world_matrix[3, :3], [5.0 if by_name else 9.0, 2.0, 0.0])
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
                rig = HierarchyData(
                    [TransformData("a"), TransformData("b"), TransformData("c", parent_node="a")]
                )
                kid = TransformData("k", translate=(0.0, 1.0, 0.0))
                rig.append(kid)
                kid.parent_node = rig["c"].uuid
                rig.world_matrix
                built = rig._rig_index()
                rig.insert(where(rig), TransformData("x", parent_node="b"))

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
        # name = 7 or parent_node = 7 on a rig in use, or a node appended or
        # inserted with a parent_node of 7: no build can stand for the rig
        # then, so no patched index may either
        def appended(rig, at=None):
            node             = TransformData("d")
            node.parent_node = 7  # past the constructor, which makes it a string
            if at is None:
                rig.append(node)
            else:
                rig.insert(at, node)

        for label, edit in (
            ("name", lambda rig: setattr(rig.list[1], "name", 7)),
            ("parent_node", lambda rig: setattr(rig.list[1], "parent_node", 7)),
            ("appended", appended),
            ("inserted", lambda rig: appended(rig, 1)),
        ):
            with self.subTest(label):
                rig = HierarchyData(
                    [
                        TransformData("a"),
                        TransformData("b", parent_node="a"),
                        TransformData("c", parent_node="b"),
                    ]
                )
                rig.world_matrix
                self.assertIsNotNone(rig._kept_index())
                edit(rig)
                self.assertIsNone(rig._kept_index())
                self.assertIsNone(rig._rig_index())
                _assert_lookups_are_a_scan(self, rig, label)

    def test_a_rename_in_a_rig_no_index_can_stand_for(self):
        # a node whose uuid is not a string: the rig keeps None for its
        # index, and a rename finds the children by a scan
        rig = HierarchyData(
            [TransformData("a"), TransformData("b", parent_node="a"), TransformData("c", parent_node="b")]
        )
        rig.list[0].__dict__["uuid"] = 12345
        self.assertIsNone(rig._rig_index())
        rig.list[1].name = "z"
        self.assertEqual(rig.name, ["a", "z", "c"])
        self.assertIs(rig.list[2].get_parent(), rig.list[1])
        self.assertIs(rig.list[1].get_parent(), rig.list[0])
        _assert_lookups_are_a_scan(self, rig, "a uuid that is not a string")

    def test_renames_in_a_rig_no_index_can_stand_for_try_one_build(self):
        # a parent_node that is not a string, written past the constructor:
        # no index can stand for the rig, and a rename makes its pass. Each
        # rename tried a build first, which went as far as that node, 5.5
        # times aba1ce5's loop at 2000 nodes. The build that failed is kept
        # across the renames, appends and parent_node writes that can not
        # change that, and tried again after one that can. Parented by name,
        # each rename pins the child naming the node to its uuid on the way
        def spoilt(count, asked, by_name=False):
            nodes = [TransformData(f"j{i}", translate=(0.0, 1.0, 0.0)) for i in range(count)]
            rig   = HierarchyData(nodes)
            for above, node in zip(nodes, nodes[1:]):
                node.parent_node = above.name if by_name else above.uuid
            nodes[-1].parent_node = 7
            if asked:
                self.assertIsNone(rig._rig_index())  # a lookup asked first
            return rig, nodes

        def renamed(rig, nodes):
            for node in nodes:
                node.name = node.name + "_x"

        def renamed_and_appended(rig, nodes):
            for node in nodes:
                node.name = node.name + "_x"
                rig.append(TransformData(node.name + "_y", parent_node=node.uuid))

        def renamed_and_reparented(rig, nodes):
            for above, node in zip(nodes, nodes[1:-1]):
                node.name        = node.name + "_x"
                node.parent_node = above.name

        for loop in (renamed, renamed_and_appended, renamed_and_reparented):
            for asked, by_name in itertools.product((False, True), (False, True)):
                with self.subTest(loop.__name__, asked=asked, by_name=by_name):
                    spent = []
                    for count in (40, 160):
                        rig, nodes = spoilt(count, asked, by_name)
                        spent.append(self._counted(lambda: loop(rig, nodes)))
                        self.assertIsNone(rig._rig_index())
                        _assert_lookups_are_a_scan(self, rig, f"{loop.__name__}, {count}")
                        if by_name and loop is not renamed_and_reparented:
                            self.assertEqual(nodes[1].parent_node, nodes[0].uuid)  # pinned
                    self.assertEqual(spent[0], spent[1])
                    self.assertLessEqual(spent[1]["builds"], 1 - asked)

        # the value that kept the index out written again, as a string, or
        # renamed so: the next lookup builds one, and it holds what a build does
        def parent_node_fixed(rig, nodes):
            nodes[-1].parent_node = nodes[-2].uuid

        def name_fixed(rig, nodes):
            nodes[-1].parent_node = nodes[-2].uuid
            nodes[3].name         = 7  # past the constructor, which makes it a string
            self.assertIsNone(rig._rig_index())
            nodes[0].name = "first"  # takes no such value away: still none
            self.assertIsNone(rig._rig_index())
            nodes[3].name = "j3"

        for fix in (parent_node_fixed, name_fixed):
            with self.subTest(fix.__name__):
                rig, nodes = spoilt(12, asked=True)
                renamed(rig, nodes)
                fix(rig, nodes)
                self.assertIsNotNone(rig._rig_index())
                _assert_lookups_are_a_scan(self, rig, fix.__name__)
                self.assertEqual(nodes[-1].world_matrix[3, 1], 12.0)

    def test_a_node_added_takes_the_children_naming_it(self):
        # a child naming a node the rig lacked, by name or by uuid, hangs
        # under the node once it is added, its cached world matrix dropped;
        # one inserted ahead of a node of its name takes that node's children
        ghost_uuid = generate_uuid()
        for label, added, at in (
            ("by name", lambda: TransformData("ghost", translate=(5.0, 0.0, 0.0)), None),
            ("by uuid", lambda: TransformData("g", uuid=ghost_uuid, translate=(5.0, 0.0, 0.0)), None),
            ("ahead", lambda: TransformData("p", translate=(5.0, 0.0, 0.0)), 0),
        ):
            with self.subTest(label):
                rig = HierarchyData(
                    [
                        TransformData("p", translate=(9.0, 0.0, 0.0)),
                        TransformData("x", translate=(0.0, 1.0, 0.0), parent_node="ghost"),
                        TransformData("y", translate=(0.0, 1.0, 0.0), parent_node=ghost_uuid),
                        TransformData("z", translate=(0.0, 1.0, 0.0), parent_node="p"),
                        TransformData("w", translate=(0.0, 1.0, 0.0), parent_node="z"),
                    ]
                )
                rig.world_matrix
                built = rig._rig_index()
                node  = added()
                if at is None:
                    rig.append(node)
                else:
                    rig.insert(at, node)

                self.assertIs(rig._rig_index(), built)
                _assert_index_is_a_build(self, rig, label)
                np.testing.assert_allclose(rig.world_matrix, rig.copy().world_matrix, atol=1e-12)
                if label == "ahead":
                    self.assertIs(rig["z"].get_parent(), node)
                    np.testing.assert_allclose(rig["w"].world_matrix[3, :3], [5.0, 2.0, 0.0])
                else:
                    child = rig["x" if label == "by name" else "y"]
                    self.assertIs(child.get_parent(), node)
                    np.testing.assert_allclose(child.world_matrix[3, :3], [5.0, 1.0, 0.0])

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
        # whose index is patched and whose own cached worlds stand
        rig = _mixed_rig()
        rig.world_matrix
        other = HierarchyData([TransformData("lift", translate=(0.0, 50.0, 0.0))])
        moved = TransformData("moved", parent_node="lift", translate=(1.0, 0.0, 0.0))
        other.append(moved)
        np.testing.assert_allclose(moved.world_matrix[3, :3], [1.0, 50.0, 0.0])

        built = rig._rig_index()
        rig.append(moved)
        self.assertIs(rig._kept_index(), built)
        self.assertTrue(all(n._cached_world() is not None for n in rig.list[:-1]))
        np.testing.assert_allclose(moved.world_matrix[3, :3], [1.0, 0.0, 0.0])

        moved.parent_node = rig["head"].uuid
        self.assertIs(rig._kept_index(), built)
        np.testing.assert_allclose(moved.world_matrix, moved.matrix @ rig["head"].world_matrix)

    def test_a_structural_edit_leaves_a_primitive_field_clean(self):
        # a node added or given another parent drops cached world matrices
        # only, patched or not: _transform_changed, which a primitive
        # extends to mark its field dirty, is a channel write's, as before
        from cgmath.geometry.sdf import DMCField, SDFSphere

        rig = HierarchyData([TransformData("root"), TransformData("arm", parent_node="root")])
        rig.world_matrix
        field = DMCField()
        ball  = SDFSphere(radius=0.5, name="ball")
        field.add(ball)
        for label, edit in (
            ("append", lambda: rig.append(ball)),
            ("parent_node", lambda: setattr(ball, "parent_node", "arm")),
            ("parent_node by uuid", lambda: setattr(ball, "parent_node", rig["root"].uuid)),
        ):
            with self.subTest(label):
                self.assertIsNotNone(rig._kept_index())
                field._dirty = False
                edit()
                self.assertFalse(field._dirty)
        ball.translate = (1.0, 0.0, 0.0)
        self.assertTrue(field._dirty)

    def test_unique_names_follow_a_name_given_up_and_taken(self):
        # one node renamed at a time, through the index: the names it keeps
        # for unique_name go with each rename
        rig = HierarchyData([TransformData(f"j{i}") for i in range(4)])
        for above, node in zip(rig.list, rig.list[1:]):
            node.parent_node = above.uuid
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
        # a parent_node or a query shaped like a uuid, with or without its
        # hyphens, names a node by uuid alone, read from the index or found
        # by a scan: a node carrying it as a name is not its parent, and
        # keeps no child naming it when renamed
        for odd in (generate_uuid(), generate_uuid().replace("-", "")):
            for kept in (False, True):
                with self.subTest(len(odd), kept=kept):
                    rig = HierarchyData(
                        [
                            TransformData("a"),
                            TransformData(odd),
                            TransformData("c"),
                            TransformData("d", parent_node=odd),
                        ]
                    )
                    if kept:
                        rig._rig_index()
                    self.assertEqual(rig._kept_index() is not None, kept)
                    with self.assertRaises(ValueError):
                        rig["c"].set_parent(odd)
                    self.assertIsNone(rig["c"].parent_node)

                    self.assertEqual(rig._kept_index() is not None, kept)
                    rig.list[1].name = "plain"
                    self.assertEqual(rig["d"].parent_node, odd)
                    self.assertEqual(rig["d"].get_parent(), odd)
                    _assert_index_is_a_build(self, rig, f"{len(odd)}, kept={kept}")

        # a digit short of a uuid's 32 is a name like any other
        short = generate_uuid().replace("-", "")[:31]
        rig   = HierarchyData([TransformData(short), TransformData("d", parent_node=short)])
        rig._rig_index()
        self.assertIs(rig["d"].get_parent(), rig.list[0])
        rig.list[0].name = "plain"
        self.assertEqual(rig["d"].parent_node, rig.list[0].uuid)

    def test_a_rename_reads_only_a_current_index(self):
        # the index a rig keeps answers a rename only while it is current:
        # after an edit that retired it, the list shrunk raw, or the list
        # replaced past __setattr__, the children naming the node are the
        # ones a scan finds, each kept under it
        def retired(rig):
            rig["toe"].uuid        = generate_uuid()
            rig["toe"].parent_node = "head"

        def shrunk(rig):
            rig.list.pop()  # tip, which names head

        def replaced(rig):
            nodes                = list(rig.list)
            late                 = TransformData("late", parent_node="head")
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
        # a name or a parent_node written past __setattr__, two nodes carrying
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
        rig = HierarchyData(
            [TransformData("a"), TransformData("b", parent_node="a"), TransformData("c", parent_node="a")]
        )
        rig._rig_index()
        rig["b"].__dict__["parent_node"] = None
        rig["a"].name                    = "z"
        self.assertIsNone(rig["b"].parent_node)
        self.assertEqual(rig["c"].parent_node, rig["z"].uuid)
        checked(rig, "a parent_node written past __setattr__")

        # b carries a's name, though the index has a alone carrying it: c
        # follows a, the first node of the name, whatever b is renamed
        rig = HierarchyData(
            [TransformData("a"), TransformData("b"), TransformData("c", parent_node="a")]
        )
        first, second = rig.list[0], rig.list[1]
        rig._rig_index()
        second.__dict__["name"] = "a"
        second.name             = "z"
        self.assertEqual(rig["c"].parent_node, "a")
        self.assertIs(rig["c"].get_parent(), first)
        checked(rig, "a name written past __setattr__")

        # the index holds the first node of a uuid; the second is not it. Its
        # children by name are pinned to the uuid, as the pass pins them
        rig = HierarchyData(
            [
                TransformData("p"),
                TransformData("q"),
                TransformData("a", parent_node="p"),
                TransformData("b", parent_node="p"),
                TransformData("c", parent_node="b"),
            ]
        )
        rig["b"].uuid = rig["a"].uuid
        rig._rig_index()
        rig["b"].parent_node = "q"
        checked(rig, "a uuid two nodes carry, a parent_node")
        rig._rig_index()
        rig["b"].name = "z"
        self.assertEqual(rig["c"].parent_node, rig["a"].uuid)
        checked(rig, "a uuid two nodes carry, a rename")

        # the same, the name renamed one an earlier node carries too, which
        # no child names: the index holds the earlier node, not this one
        rig              = HierarchyData([TransformData("a"), TransformData("a"), TransformData("b")])
        rig.list[1].uuid = rig.list[0].uuid
        rig._rig_index()
        rig.list[1].name = "z"
        self.assertEqual(rig.name, ["a", "z", "b"])
        checked(rig, "a uuid two nodes carry, a shared name renamed")

        # a name two nodes carry written past __setattr__ on a third, between
        # them: the index does not have it carrying that name
        rig = HierarchyData([TransformData("a"), TransformData("a"), TransformData("q"), TransformData("a")])
        rig._rig_index()
        rig.list[2].__dict__["name"] = "a"
        rig.list[2].name             = "z"
        self.assertEqual(rig.name, ["a", "a", "z", "a"])
        checked(rig, "a shared name written past __setattr__")

        # a name that is not a string: no index can stand for the rig
        rig        = _mixed_rig()
        seven      = TransformData("seven", parent_node="spine")
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
                parent_node=None if i == 0 else nodes[int(rng.integers(0, i))].uuid,
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
        return HierarchyData(
            [
                TransformData("g", node_type="joint", rotate=(10, 20, 30)),
                TransformData("a", node_type="joint", parent_node="g", scale=mirror,
                              translate=(0, 1, 0), rotate=(-20, 5, 40)),
                TransformData("c", node_type="joint", parent_node="a", translate=(0, 1, 0),
                              segment_scale_compensate=True, rotate=(5, 5, 5)),
                TransformData("d", node_type="joint", parent_node="c", translate=(1, 0, 0)),
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
                    parent_node=nodes[int(rng.integers(max(0, i - 5), i))].uuid if i else None,
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
                    parent_node = uuids[i - 1] if i else None,
                    translate   = rng.normal(size=3),
                    rotate      = rng.uniform(-40.0, 40.0, 3),
                    scale       = rng.uniform(0.9, 1.1, 3),
                )
                for i in range(count)
            ]
            leaves = [
                TransformData(f"l{i}", uuid=uuids[count + i], parent_node=uuids[count - 1])
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
                    parent_node = None if i == 0 else uuids[min(i - 1, half - 1)],
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
                    parent_node = None if i == 0 else uuids[min(i - 1, half - 1)],
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
