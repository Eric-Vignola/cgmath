import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath.formats.glb import pygltflib
from cgmath.hierarchy import HierarchyData, TransformData, TransformList
from cgmath.hierarchy import _fbx_enum, _rotation, FBX
from cgmath.transforms import (
    euler_to_matrix,
    euler_to_quaternion,
    matrix_multiply,
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
 
        # save the data to disk
        pkl_file  = skeleton.save("~/skeleton.pkl",  mode="pkl")
        npz_file  = skeleton.save("~/skeleton.npz",  mode="npz")
        json_file = skeleton.save("~/skeleton.json", mode="json")

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
        self.assertTrue(node.get_root() == ['root_joint', 'pelvis_joint', 'spineLower_joint', 'spineMiddle_joint', 'spineUpper_joint', 'chest_joint', 'clavicle_left_joint', 'shoulder_left_joint', 'elbow_left_joint']) 

        # find this node's children
        self.assertTrue(node.get_children() == ['elbow_partial_left_joint', 'handWrist_left_joint'])

        # find this node's branch (self + all decendents)
        self.assertTrue(node.get_branch() == ['elbow_left_joint', 'elbow_partial_left_joint', 'handWrist_left_joint', 'handIndexMeta_left_joint', 'handIndex_00_left_joint', 'handIndex_01_left_joint', 'handIndex_02_left_joint', 'handMiddleMeta_left_joint', 'handMiddle_00_left_joint', 'handMiddle_01_left_joint', 'handMiddle_02_left_joint', 'handWrist_partial_left_joint', 'handPinkyMeta_left_joint', 'handPinky_00_left_joint', 'handPinky_01_left_joint', 'handPinky_02_left_joint', 'handRingMeta_left_joint', 'handRing_00_left_joint', 'handRing_01_left_joint', 'handRing_02_left_joint', 'handThumbMeta_left_joint', 'handThumb_00_left_joint', 'handThumb_01_left_joint', 'handThumb_02_left_joint']) 



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
        # copy() bypasses __init__, so an untouched channel still points at
        # the class default, which is shared by every instance
        node       = TransformData(name='a').copy()
        node.scale = [3.0, 3.0, 3.0]

        self.assertTrue(np.allclose(node.scale, 3.0))
        self.assertTrue(np.allclose(TransformData._scale, 1.0))
        self.assertTrue(np.allclose(TransformData(name='b').scale, 1.0))

    def test_channel_defaults_are_float(self):
        # an integer default would silently truncate an in place write
        node        = TransformData(name='a').copy()
        node.rotate = [1.5, 2.5, 3.5]

        self.assertTrue(np.allclose(node.rotate, [1.5, 2.5, 3.5]))
        for channel in ('_scale', '_rotate', '_translate', '_rotate_axis',
                        '_joint_orient'):
            default = np.asarray(getattr(TransformData, channel))
            self.assertEqual(default.dtype, np.float64)

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
