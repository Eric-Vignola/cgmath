import contextlib
import copy
import itertools
import json as std_json
import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from cgmath.transforms._numba._euler import MAYA_EA
from cgmath import hierarchy
from cgmath.formats.glb import pygltflib
from cgmath.transforms import (
    euler_filter,
    euler_to_matrix,
    euler_to_quaternion,
    quaternion_slerp,
    quaternion_to_matrix,
)
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList
from cgmath.hierarchy import _fbx_frame_rate, FBX
from cgmath.hierarchy.hierarchy import _filtered, _rotation
from cgmath._tests._maya_rotate_order import maya_rotate_order_cases


def _parent_ids(names, parents) -> tuple:
    """a new uuid for each of ``names``, and for each of ``parents`` -- a
    name, or None for a root -- the uuid of the first node carrying that
    name, None for a name no node carries: what a rig whose parents are
    given by name is built with, each node holding its parent's uuid"""
    ids   = [hierarchy.generate_uuid() for _ in names]
    first = {}
    for name, uuid in zip(names, ids):
        first.setdefault(name, uuid)
    return ids, [first.get(parent) for parent in parents]


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


def chain(count=4):
    """a simple parent to child chain of joints"""
    ids = [hierarchy.generate_uuid() for _ in range(count)]
    return HierarchyData(
        [
            TransformData(
                name        = f"j{i}",
                uuid        = ids[i],
                parent_uuid = None if i == 0 else ids[i - 1],
                translate   = [0.0, float(i), 0.0],
                rotate      = [0.0, 0.0, float(i) * 3.0],
                scale       = [1.0, 1.0, 1.0],
            )
            for i in range(count)
        ]
    )


class TestClipData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.rig = chain(4)

    def test_building_a_clip_leaves_the_source_rig_alone(self):
        # the clip copies its nodes; adopting them would empty the caller's rig
        clip = ClipData(self.rig, frames=5)

        self.assertEqual(len(self.rig),    4)
        self.assertEqual(self.rig.name,    ["j0", "j1", "j2", "j3"])
        self.assertEqual(len(clip),        4)
        self.assertEqual(clip.frame_count, 5)

    def test_every_frame_starts_on_the_rest_pose(self):
        clip = ClipData(self.rig, frames=3)

        for frame in range(3):
            clip.frame = frame
            self.assertTrue(np.allclose(clip.translate, self.rig.translate))
            self.assertTrue(np.allclose(clip.rotate, self.rig.rotate))

    def test_scrubbing_reads_and_writes_the_current_frame(self):
        clip = ClipData(self.rig, frames=4)

        clip.frame     = 2
        clip.translate = np.full((4, 3), 7.0)

        self.assertTrue(np.allclose(clip.frames.translate[2], 7.0))
        for frame in (0, 1, 3):
            self.assertFalse(np.allclose(clip.frames.translate[frame], 7.0))

    def test_frames_index_edits_in_place(self):
        # clip.frames[12].match(...) is the same object as clip scrubbed to 12
        clip   = ClipData(self.rig, frames=4)
        before = clip.frames.rotate[1].copy()

        clip.frames[1].match("j*").rotate_y += 5.0

        self.assertTrue(np.allclose(clip.frames.rotate[1][:, 1], before[:, 1] + 5.0))
        self.assertTrue(np.allclose(clip.frames.rotate[0], before))
        self.assertIs(clip.frames[1], clip)

    def test_node_write_reaches_the_block(self):
        clip                 = ClipData(self.rig, frames=3)
        clip.frame           = 1
        clip["j2"].translate = [4.0, 5.0, 6.0]

        self.assertTrue(np.allclose(clip.frames.translate[1, 2], [4.0, 5.0, 6.0]))
        self.assertFalse(np.allclose(clip.frames.translate[0, 2], [4.0, 5.0, 6.0]))

    def test_block_getters_are_read_only(self):
        clip = ClipData(self.rig, frames=3)

        with self.assertRaises(ValueError):
            clip.frames.translate[0, 0, 0] = 1.0

    def test_block_setters_broadcast(self):
        clip = ClipData(self.rig, frames=3)

        clip.frames.scale = 2.0
        self.assertTrue(np.allclose(clip.frames.scale, 2.0))
        self.assertTrue(np.allclose(clip.scale, 2.0))

        values                = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)
        clip.frames.translate = values
        self.assertTrue(np.allclose(clip.frames.translate, values))

        clip.frame = 2
        self.assertTrue(np.allclose(clip.translate, values[2]))

    def test_frames_setitem_stores_a_pose(self):
        clip = ClipData(self.rig, frames=3)

        pose           = self.rig.copy()
        pose.translate = np.full((4, 3), 9.0)
        clip.frames[1] = pose

        self.assertTrue(np.allclose(clip.frames.translate[1], 9.0))
        self.assertFalse(np.allclose(clip.frames.translate[0], 9.0))

    def test_frames_setitem_rejects_a_mismatched_pose(self):
        clip = ClipData(self.rig, frames=2)
        with self.assertRaises(ValueError):
            clip.frames[0] = chain(3)

    def test_from_poses(self):
        poses = []
        for i in range(4):
            pose           = self.rig.copy()
            pose.translate = np.full((4, 3), float(i))
            poses.append(pose)

        clip = ClipData.from_poses(poses, start_frame=1001, fps=30.0)

        self.assertEqual(clip.frame_count, 4)
        self.assertEqual(clip.start_frame, 1001)
        self.assertEqual(clip.end_frame,   1004)
        self.assertEqual(clip.fps,         30.0)
        for i in range(4):
            self.assertTrue(np.allclose(clip.frames.translate[i], float(i)))

    def test_timebase(self):
        clip = ClipData(self.rig, frames=5, start_frame=1001, fps=25.0)

        self.assertEqual(clip.start_frame, 1001)
        self.assertEqual(clip.end_frame, 1005)
        self.assertTrue(np.allclose(clip.frames.times, np.arange(1001, 1006) / 25.0))

    def test_end_frame_follows_the_block_length(self):
        clip = ClipData(self.rig, frames=5, start_frame=10)
        self.assertEqual(clip.end_frame, 14)

        clip.pop()
        self.assertEqual(len(clip), 3)
        self.assertEqual(clip.end_frame, 14)

    def test_frame_out_of_range_raises(self):
        clip = ClipData(self.rig, frames=3)
        with self.assertRaises(IndexError):
            clip.frame = 3
        with self.assertRaises(IndexError):
            clip.frame = -4

    def test_negative_frame_indexes_from_the_end(self):
        clip                  = ClipData(self.rig, frames=3)
        clip.frames.translate = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)

        clip.frame = -1
        self.assertEqual(clip.frame, 2)
        self.assertTrue(np.allclose(clip.translate, clip.frames.translate[2]))

    def test_append_and_pop_resize_the_blocks(self):
        clip = ClipData(self.rig, frames=3)

        clip.append(TransformData(name="extra", translate=[1.0, 2.0, 3.0]))
        self.assertEqual(len(clip), 5)
        self.assertEqual(clip.frames.translate.shape, (3, 5, 3))
        for frame in range(3):
            self.assertTrue(
                np.allclose(clip.frames.translate[frame, 4], [1.0, 2.0, 3.0])
            )

        clip.pop()
        self.assertEqual(len(clip), 4)
        self.assertEqual(clip.frames.translate.shape, (3, 4, 3))

    def test_copy_is_independent(self):
        clip                  = ClipData(self.rig, frames=3)
        clip.frames.translate = np.full((3, 4, 3), 2.0)

        other                  = clip.copy()
        other.frames.translate = np.full((3, 4, 3), 8.0)

        self.assertTrue(np.allclose(clip.frames.translate, 2.0))
        self.assertTrue(np.allclose(other.frames.translate, 8.0))
        self.assertEqual(other.frame_count, 3)

    def test_copy_keeps_every_frame_distinct(self):
        # a copy that lost its column list would reseed each frame from the
        # loaded pose on the next realign, which a uniform block cannot show
        clip                  = ClipData(self.rig, frames=4)
        block                 = np.arange(4 * 4 * 3, dtype=float).reshape(4, 4, 3)
        clip.frames.translate = block
        clip.frame            = 2

        other = clip.copy()

        self.assertTrue(np.array_equal(other.frames.translate, block))
        self.assertEqual(other.frame, 2)

    def test_a_shallow_copy_leaves_the_clip_whole(self):
        # the copy owns its nodes and blocks: growing or reordering it leaves
        # the original's blocks matching its own nodes
        clip                  = ClipData(self.rig, frames=3)
        clip.frames.translate = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)

        twin = copy.copy(clip)
        twin.append(TransformData("extra"))
        twin.sort(reverse=True)
        twin.frame = 1

        back = ClipData.from_dict(clip.to_dict())
        self.assertEqual(len(back), 4)
        self.assertTrue(np.array_equal(back.frames.translate, clip.frames.translate))

    def test_equality_discriminates_on_animation(self):
        a = ClipData(self.rig, frames=3)
        b = a.copy()
        self.assertEqual(a, b)

        b.frames.rotate = np.full((3, 4, 3), 45.0)
        self.assertNotEqual(a, b)

    def test_round_trip(self):
        clip                  = ClipData(self.rig, frames=4, start_frame=1001, fps=30.0)
        clip.frames.translate = np.arange(4 * 4 * 3, dtype=float).reshape(4, 4, 3)
        clip.frame            = 2

        for mode in ("json", "npz", "pkl"):
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, f"clip.{mode}")
                clip.save(path)
                loaded = ClipData.load(path)

                self.assertEqual(loaded.frame_count, 4)
                self.assertEqual(len(loaded),        4)
                self.assertEqual(loaded.start_frame, 1001)
                self.assertEqual(loaded.fps,         30.0)
                self.assertEqual(loaded.name,        clip.name)
                self.assertTrue(
                    np.allclose(loaded.frames.translate, clip.frames.translate)
                )
                self.assertEqual(loaded, clip)

    def test_a_single_frame_clip_matches_the_rig_it_came_from(self):
        clip = ClipData(self.rig, frames=1)

        self.assertEqual(clip.frame_count, 1)
        self.assertTrue(np.allclose(clip.translate, self.rig.translate))
        self.assertTrue(np.allclose(clip.world_matrix, self.rig.world_matrix))

    def test_world_matrix_follows_the_scrub(self):
        clip                     = ClipData(self.rig, frames=2)
        clip.frames[1].translate = np.full((4, 3), 5.0)

        clip.frame = 0
        first      = clip.world_matrix.copy()
        clip.frame = 1
        second     = clip.world_matrix.copy()

        self.assertFalse(np.allclose(first, second))
        clip.frame = 0
        self.assertTrue(np.allclose(clip.world_matrix, first))

    def test_structural_channels_are_shared_by_every_frame(self):
        clip                    = ClipData(self.rig, frames=3)
        clip.frame              = 0
        clip["j1"].rotate_order = 3

        clip.frame = 2
        self.assertEqual(clip["j1"].rotate_order, 3)


class TestClipFiles(unittest.TestCase):
    """a loaded clip is the clip that was saved, nodes still bound to the blocks"""

    def setUp(self):
        super().setUp()
        self.clip                  = ClipData(chain(4), frames=3, start_frame=1001, fps=30.0)
        self.clip.frames.translate = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)
        self.clip.frame            = 1

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def loaded(self):
        """the clip back from every file format, pickle and deepcopy"""
        out = {
            "pickle":   pickle.loads(pickle.dumps(self.clip)),
            "deepcopy": copy.deepcopy(self.clip),
        }
        for mode in ("json", "npz", "pkl"):
            path      = self.clip.save(os.path.join(self.tmp.name, f"clip.{mode}"))
            out[mode] = ClipData.load(path)
        return out

    def test_a_ragged_block_is_named(self):
        # a hand-edited json with one row short says which block is wrong
        path = self.clip.save(os.path.join(self.tmp.name, "ragged.json"))
        with open(path, encoding="utf-8") as f:
            tree = std_json.load(f)
        tree["blocks"]["translate"][0][0] = [0.0, 0.0]
        with open(path, "w", encoding="utf-8") as f:
            std_json.dump(tree, f)
        with self.assertRaisesRegex(ValueError, "ClipData block translate must have shape"):
            ClipData.load(path)

    def test_a_loaded_clip_writes_through_to_its_blocks(self):
        # a pickled clip used to come back with every node channel a copy of
        # its row, so a write at the loaded frame never reached the block
        for how, clip in self.loaded().items():
            with self.subTest(how):
                clip["j2"].translate = [7.0, 8.0, 9.0]
                self.assertTrue(np.allclose(clip.frames.translate[1, 2], [7.0, 8.0, 9.0]))

                clip.frame = 0
                clip.frame = 1
                self.assertTrue(np.allclose(clip["j2"].translate, [7.0, 8.0, 9.0]))

                for channel in ClipData.FRAMED_CHANNELS:
                    for node in clip:
                        self.assertTrue(
                            np.shares_memory(getattr(node, channel), clip._blocks[channel])
                        )

    def test_a_loaded_clip_is_the_saved_clip(self):
        for how, clip in self.loaded().items():
            with self.subTest(how):
                self.assertIsInstance(clip, ClipData)
                self.assertEqual(clip,             self.clip)
                self.assertEqual(clip.frame,       1)
                self.assertEqual(clip.start_frame, 1001)
                self.assertEqual(clip.fps,         30.0)
                self.assertEqual(clip.uuid,        self.clip.uuid)
                self.assertTrue(all(node._hierarchy is clip for node in clip))
                for channel, block in clip._blocks.items():
                    self.assertEqual(block.dtype, np.float64)
                    self.assertEqual(block.shape, (3, 4, 3))

    def test_the_module_loader_hands_back_a_clip(self):
        # cgmath.hierarchy.load() used to rebuild every file as a HierarchyData
        for mode in ("json", "npz", "pkl"):
            with self.subTest(mode):
                path   = self.clip.save(os.path.join(self.tmp.name, f"clip.{mode}"))
                loaded = hierarchy.load(path)

                self.assertIsInstance(loaded, ClipData)
                self.assertEqual(loaded, self.clip)

    def test_a_block_of_the_wrong_shape_is_refused(self):
        # each error names the block as the file does
        bad = {
            "a channel short": ("scale", np.ones((3, 4, 2)), "block scale must have shape"),
            "a node short":    ("rotate", np.ones((3, 3, 3)), "block rotate must have shape"),
            "no frames":       ("rotate", np.ones((0, 4, 3)), "block rotate must have shape"),
            "a frame short":   ("scale", np.ones((2, 4, 3)), "different frame counts"),
            "an unknown name": ("shear", np.ones((3, 4, 3)), "no framed channel 'shear'"),
        }
        for label, (channel, block, message) in bad.items():
            with self.subTest(label):
                tree                    = self.clip.to_dict()
                tree["blocks"]          = dict(tree["blocks"])
                tree["blocks"][channel] = block
                with self.assertRaisesRegex(ValueError, message):
                    ClipData.from_dict(tree)

        tree           = self.clip.to_dict()
        tree["blocks"] = {"scale": tree["blocks"]["scale"]}
        with self.assertRaisesRegex(ValueError, r"missing the blocks \['joint_orient', 'rotate'"):
            ClipData.from_dict(tree)

    def test_equality_includes_the_loaded_frame(self):
        # the nodes hold the loaded frame, so two clips on different frames
        # do not hold the same values
        other = self.clip.copy()
        self.assertEqual(other, self.clip)

        other.frame = 2
        self.assertNotEqual(other, self.clip)


def joint_chain(count=4):
    """a chain of joints, which is what the orient helpers act on"""
    ids = [hierarchy.generate_uuid() for _ in range(count)]
    return HierarchyData(
        [
            TransformData(
                name         = f"j{i}",
                node_type    = "joint",
                uuid         = ids[i],
                parent_uuid  = None if i == 0 else ids[i - 1],
                translate    = [0.0, float(i), 0.0],
                rotate       = [float(i) * 7.0, float(i) * 5.0, float(i) * 3.0],
                joint_orient = [0.0, float(i) * 2.0, 0.0],
            )
            for i in range(count)
        ]
    )


def uuid_chain(count=4, root=False):
    """a chain parented through set_parent by name, which keeps each
    parent's uuid; joint_chain() gives the uuids as it builds the nodes
    """
    h = HierarchyData(
        [
            TransformData(
                name         = f"j{i}",
                node_type    = "joint",
                translate    = [0.0, float(i), 0.0],
                rotate       = [float(i) * 7.0, float(i) * 5.0, float(i) * 3.0],
                joint_orient = [0.0, float(i) * 2.0, 0.0],
            )
            for i in range(count)
        ]
    )
    if root:
        h.append(TransformData(name="root", node_type="joint"))
    for i in range(1, count):
        h[f"j{i}"].set_parent(f"j{i - 1}", world_space=False)
    return h


class TestClipStructuralMutation(unittest.TestCase):
    """rig level edits run on every frame of an animated clip; the rest of
    the structural edits still work"""

    def _varied_clip(self, rig, frames=3):
        clip   = ClipData(rig, frames=frames)
        rotate = clip.frames.rotate.copy()
        for frame in range(frames):
            rotate[frame] += frame * 11.0
        clip.frames.rotate = rotate
        return clip

    # --- every frame at once --- #
    def test_every_rig_edit_runs_on_an_animated_clip(self):
        # each one refused a clip of 2+ frames before
        edits = {
            "swapaxes": lambda c: c.swapaxes(0, 1, negate=True),
            "set_rotate_to_joint_orient": lambda c: c.set_rotate_to_joint_orient(),
            "set_joint_orient_to_rotate": lambda c: c.set_joint_orient_to_rotate(),
            "set_parent": lambda c: c.set_parent("root"),
            "node_type": lambda c: setattr(c, "node_type", "transform"),
            "set_rotate_order": lambda c: c.set_rotate_order("zyx"),
        }
        for name, edit in edits.items():
            with self.subTest(edit=name):
                clip = self._varied_clip(uuid_chain(4, root=True))
                edit(clip)
                self.assertEqual(clip.frames.rotate.shape, (3, 5, 3))

    def test_a_selection_and_a_single_node_run_too(self):
        # every frame of the swapped node turns by the permutation, and its
        # child stays where it was on every frame
        P         = np.eye(4)
        P[:3, :3] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        for edit in (
            lambda c: c[1:3].swapaxes(0, 1),
            lambda c: c.match("j1").swapaxes(0, 1),
            lambda c: c["j1"].swapaxes(0, 1),
        ):
            clip   = self._varied_clip(uuid_chain(4))
            before = clip.frames.world_matrix.copy()
            edit(clip)
            after = clip.frames.world_matrix
            self.assertTrue(np.allclose(after[:, 1], P @ before[:, 1], atol=1e-9))
            self.assertTrue(np.allclose(after[:, 0], before[:, 0], atol=1e-9))
            self.assertTrue(np.allclose(after[:, 3], before[:, 3], atol=1e-9))

    def test_an_unknown_parent_leaves_the_clip_untouched(self):
        clip    = self._varied_clip(uuid_chain(4, root=True))
        before  = np.array(clip.frames.rotate)
        parents = [n.parent_uuid for n in clip]

        with self.assertRaises(ValueError):
            clip.set_parent("nowhere")
        with self.assertRaises(ValueError):
            clip["j2"].set_parent("nowhere")
        with self.assertRaises(ValueError):
            clip.swapaxes(0, 0)

        self.assertTrue(np.array_equal(clip.frames.rotate, before))
        self.assertEqual([n.parent_uuid for n in clip], parents)

    # --- still as they were --- #
    def test_a_single_frame_clip_runs_them_too(self):
        clip   = ClipData(uuid_chain(4), frames=1)
        before = clip.world_matrix.copy()

        clip.swapaxes(0, 1, negate=True)
        clip.set_rotate_to_joint_orient()

        self.assertTrue(np.allclose(clip.rotate, 0.0, atol=1e-9))
        self.assertFalse(np.allclose(clip.world_matrix, before))

    def test_a_frameless_clip_runs_them_as_a_pose(self):
        clip = ClipData(uuid_chain(4))
        self.assertEqual(clip.frame_count, 0)
        clip.swapaxes(0, 1, negate=True)
        clip.set_joint_orient_to_rotate()
        self.assertTrue(np.allclose(clip.joint_orient, 0.0, atol=1e-9))

    def test_a_plain_hierarchy_still_runs_them(self):
        rig    = uuid_chain(4, root=True)
        before = rig.world_matrix.copy()

        rig[:2].swapaxes(0, 1)
        rig["j2"].set_parent("root")
        rig.match("j3").set_rotate_to_joint_orient()

        self.assertFalse(np.allclose(rig.world_matrix, before))

    def test_the_orient_helpers_leave_a_transform_alone(self):
        # they act on joints only
        rig = HierarchyData(
            [
                TransformData(
                    name="grp", node_type="transform", joint_orient=[0.0, 30.0, 0.0]
                )
            ]
        )
        clip   = self._varied_clip(rig, frames=4)
        before = {c: np.array(getattr(clip.frames, c)) for c in ("rotate", "joint_orient")}
        clip.set_rotate_to_joint_orient()
        clip.set_joint_orient_to_rotate()
        for channel, values in before.items():
            self.assertTrue(np.array_equal(getattr(clip.frames, channel), values))

    def test_set_parent_without_world_space_keeps_local_values(self):
        rig = joint_chain(3)
        rig.append(TransformData(name="root", node_type="joint"))
        clip   = self._varied_clip(rig)

        before = {}
        for frame in range(clip.frame_count):
            clip.frame    = frame
            before[frame] = {n.name: n.translate.copy() for n in clip}

        clip.set_parent(None, world_space=False)

        for frame, want in before.items():
            clip.frame = frame
            for node in clip:
                self.assertTrue(np.allclose(node.translate, want[node.name]))

    def test_reordering_nodes_keeps_each_column_with_its_node(self):
        # set_parent sorts the new parent ahead of its children, so the block
        # columns have to follow the nodes rather than stay where they were
        rig = joint_chain(3)
        rig.append(TransformData(name="root", node_type="joint"))
        clip   = self._varied_clip(rig)

        before = {}
        for frame in range(clip.frame_count):
            clip.frame    = frame
            before[frame] = {n.name: n.rotate.copy() for n in clip}

        clip.set_parent("root", world_space=False)

        self.assertEqual(clip.frames.rotate.shape, (3, 4, 3))
        for frame, want in before.items():
            clip.frame = frame
            for node in clip:
                self.assertTrue(np.allclose(node.rotate, want[node.name]))


def _per_frame(clip, edit):
    """the reference: every frame of the clip as a pose of its own, edited by
    the single pose methods"""
    poses = []
    for frame in range(clip.frame_count):
        clip.frame = frame
        pose       = TransformList(clip.list).copy()
        edit(pose)
        poses.append(pose)
    clip.frame = 0
    return poses


def _one_at_a_time(view, parent, world_space=True):
    """TransformList.set_parent as its docstring states it: every node to the
    world, then each under ``parent``, one node at a time"""
    for node in view:
        node.set_parent(None, world_space=world_space)
    if parent is not None:
        for node in view:
            if node != parent:
                node.set_parent(parent, world_space=world_space)


def _stacked(poses, channel):
    return np.array([[getattr(node, channel) for node in pose] for pose in poses])


class TestClipRigEditsEveryFrame(unittest.TestCase):
    """a rig edit on a clip solves every frame at once. It must equal the
    single pose methods run frame by frame: world matrices within 1e-9, and
    every euler curve equal to the frame by frame one once that is put
    through euler_filter too, for every rotate order"""

    def _rig(self, order, count=8, seed=7, scaled=True):
        rng   = np.random.default_rng(seed)
        ids   = [hierarchy.generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    uuid=ids[i],
                    parent_uuid=None if i == 0 else ids[rng.integers(0, i)],
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-180.0, 180.0, size=3),
                    scale=rng.uniform(0.6, 1.6, size=3) if scaled else None,
                    rotate_axis=rng.uniform(-40.0, 40.0, size=3),
                    joint_orient=rng.uniform(-90.0, 90.0, size=3),
                    rotate_order=order,
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
            )
        return HierarchyData(nodes)

    def _animated(self, rig, frames=24, seed=3, scaled=True):
        # every framed channel moves, so each frame solves through its own
        # rotate_axis and joint_orient
        rng  = np.random.default_rng(seed)
        clip = ClipData(rig, frames=frames, start_frame=1001, fps=30.0)
        t    = np.arange(frames)[:, None, None]
        size = (1, len(clip), 3)

        def wave(amplitude):
            speed = rng.uniform(0.05, 0.25, size=size)
            return amplitude * np.sin(t * speed + rng.uniform(0.0, 6.0, size=size))

        clip.frames.rotate       = clip.frames.rotate + wave(60.0)
        clip.frames.translate    = clip.frames.translate + wave(0.3)
        clip.frames.joint_orient = clip.frames.joint_orient + wave(15.0)
        clip.frames.rotate_axis  = clip.frames.rotate_axis + wave(10.0)
        if scaled:
            clip.frames.scale = clip.frames.scale * (1.0 + wave(0.2))
        return clip

    def _assert_matches(self, clip, poses):
        first = poses[0]
        self.assertEqual(clip.name, first.name)
        self.assertEqual([n.parent_uuid for n in clip], [n.parent_uuid for n in first])
        self.assertEqual(list(clip.rotate_order), [n.rotate_order for n in first])
        self.assertEqual(clip.node_type, first.node_type)

        world = np.array([[node.world_matrix for node in pose] for pose in poses])
        self.assertLess(np.abs(clip.frames.world_matrix - world).max(), 1e-9)

        orders = np.array([node.rotate_order for node in first])
        xyz    = np.zeros(len(first), dtype=int)
        for channel, axes in (
            ("rotate", orders),
            ("rotate_axis", xyz),
            ("joint_orient", xyz),
            ("scale", None),
            ("translate", None),
        ):
            want = _stacked(poses, channel)
            if axes is not None:
                want = _filtered(want, axes)
            error = np.abs(getattr(clip.frames, channel) - want).max()
            self.assertLess(error, 1e-9, channel)

    def _check(self, clip_edit, pose_edit, scaled=True):
        for order in range(6):
            with self.subTest(order=order):
                clip  = self._animated(self._rig(order, scaled=scaled), scaled=scaled)
                poses = _per_frame(clip, pose_edit)
                clip_edit(clip)
                self._assert_matches(clip, poses)

    def test_swapaxes_on_the_whole_clip(self):
        for axis0, axis1, negate in ((0, 1, True), (2, 0, False), (1, 2, True)):
            with self.subTest(axes=(axis0, axis1, negate)):
                self._check(
                    lambda c: c.swapaxes(axis0, axis1, negate=negate),
                    lambda p: [n.swapaxes(axis0, axis1, negate=negate) for n in p[:]],
                )

    def test_swapaxes_on_a_node_and_a_view(self):
        self._check(lambda c: c["j3"].swapaxes(1, 2), lambda p: p["j3"].swapaxes(1, 2))
        self._check(
            lambda c: c[[5, 1, 3]].swapaxes(0, 2, negate=True),
            lambda p: [n.swapaxes(0, 2, negate=True) for n in p[[5, 1, 3]]],
        )

    def test_set_rotate_to_joint_orient(self):
        self._check(
            lambda c: c.set_rotate_to_joint_orient(),
            lambda p: [n.set_rotate_to_joint_orient() for n in p],
        )
        self._check(
            lambda c: c["j2"].set_rotate_to_joint_orient(),
            lambda p: p["j2"].set_rotate_to_joint_orient(),
        )

    def test_set_joint_orient_to_rotate(self):
        self._check(
            lambda c: c.set_joint_orient_to_rotate(),
            lambda p: [n.set_joint_orient_to_rotate() for n in p],
        )
        self._check(
            lambda c: c[2:5].set_joint_orient_to_rotate(),
            lambda p: [n.set_joint_orient_to_rotate() for n in p[2:5]],
        )

    def test_set_parent_of_one_node_in_world_space(self):
        self._check(lambda c: c["j4"].set_parent("j0"), lambda p: p["j4"].set_parent("j0"))
        self._check(lambda c: c["j4"].set_parent(None), lambda p: p["j4"].set_parent(None))

    def test_set_parent_of_a_selection_in_world_space(self):
        # unscaled, every node keeps its world exactly, so the selection is
        # solved in one batch per step; scaled, a sheared or resized node
        # moves the ones below it, so they go one at a time. Either way the
        # result is the single pose method's, which goes one at a time. A
        # pose's own selection is solved the same way: only the clip's
        # calls are counted
        module = sys.modules[ClipData.__module__]
        real   = module._cascades
        seen   = []

        def spy(frames, *args):
            found = real(frames, *args)
            if getattr(frames.owner, "_blocks", None):
                seen.append(found)
            return found

        edits = {
            "[1:] to the world": (
                lambda h: h[1:].set_parent(None), lambda h: _one_at_a_time(h[1:], None)
            ),
            "[1:] under j0": (
                lambda h: h[1:].set_parent("j0"), lambda h: _one_at_a_time(h[1:], "j0")
            ),
            "the whole rig under j0": (
                lambda h: h.set_parent("j0"), lambda h: _one_at_a_time(h[:], "j0")
            ),
            "shuffled under j2": (
                lambda h: h[[3, 1, 2, 0]].set_parent("j2"),
                lambda h: _one_at_a_time(h[[3, 1, 2, 0]], "j2"),
            ),
        }
        for scaled in (False, True):
            seen.clear()
            for name, (edit, reference) in edits.items():
                with self.subTest(scaled=scaled, edit=name):
                    with mock.patch.object(module, "_cascades", spy):
                        self._check(edit, reference, scaled=scaled)

            self.assertEqual(len(seen), 6 * len(edits))
            if scaled:
                self.assertIn(True, seen)
            else:
                self.assertNotIn(True, seen)

    def test_set_parent_resolves_a_shared_name_as_one_node_at_a_time(self):
        # "x" names two roots, the first of them in the selection: one node
        # at a time, the first pass moves it behind the second, which the
        # second pass then finds. Resolved once up front, the batch hung b
        # under the first x, and only on a clip whose scale kept it batched
        for scaled in (False, True):
            for frames in (1, 3):
                with self.subTest(scaled=scaled, frames=frames):
                    first = TransformData("x", node_type="joint", uuid=hierarchy.generate_uuid(), translate=(1, 0, 0))
                    nodes = [
                        first,
                        TransformData("b", node_type="joint", parent_uuid=first.uuid),
                        TransformData("x", node_type="joint", translate=(0, 0, 3)),
                        TransformData("c", node_type="joint", translate=(0, 2, 0)),
                    ]
                    if scaled:
                        nodes[0].scale = (1.5, 0.7, 1.2)
                    clip   = self._animated(HierarchyData(nodes), frames=frames, scaled=scaled)
                    poses  = _per_frame(clip, lambda p: _one_at_a_time(p[[0, 1]], "x"))
                    second = clip[2]
                    clip[[0, 1]].set_parent("x")
                    self._assert_matches(clip, poses)
                    self.assertIs(clip["b"].get_parent(), second)
                    self.assertIs(clip.list[-2].get_parent(), second)

                    # the list method on a pose is the same
                    pose = HierarchyData([n.copy() for n in nodes])
                    pose[[0, 1]].set_parent("x")
                    self.assertEqual(pose.name, poses[0].name)
                    self.assertEqual(
                        [n.parent_uuid for n in pose], [n.parent_uuid for n in poses[0]]
                    )

    def test_set_parent_of_a_selection_holding_only_the_parent(self):
        # the second pass has nothing to move: it raised "length mismatch"
        for frames in (1, 3):
            with self.subTest(frames=frames):
                clip  = self._animated(self._rig(2), frames=frames)
                poses = _per_frame(clip, lambda p: _one_at_a_time(p[[0]], "j0"))
                clip[[0]].set_parent("j0")
                self._assert_matches(clip, poses)

    def test_set_parent_down_a_chain_in_any_order(self):
        # a scaled chain shears, so the selection goes one node at a time:
        # each node's world is its chain up to the lowest node solved before
        # it, under that one's new local. Leaf first, root first, a shuffle,
        # an order that goes back up the chain between its far nodes, and one
        # that goes up a chain between leaves hanging off its end, with and
        # without compensation, to the world and under the root
        module = sys.modules[ClipData.__module__]
        real   = module._cascades
        seen   = []

        def spy(frames, *args):
            found = real(frames, *args)
            if getattr(frames.owner, "_blocks", None):
                seen.append(found)
            return found

        def chain(order, compensate, leaves):
            rng = np.random.default_rng(11)
            ids = [hierarchy.generate_uuid() for _ in range(10)]
            return HierarchyData(
                [
                    TransformData(
                        name=f"j{i}",
                        node_type="joint",
                        uuid=ids[i],
                        parent_uuid=None if i == 0 else ids[min(i - 1, 5) if leaves else i - 1],
                        translate=rng.normal(size=3),
                        rotate=rng.uniform(-180.0, 180.0, size=3),
                        scale=rng.uniform(0.6, 1.6, size=3),
                        rotate_axis=rng.uniform(-40.0, 40.0, size=3),
                        joint_orient=rng.uniform(-90.0, 90.0, size=3),
                        rotate_order=order,
                        segment_scale_compensate=compensate(i, rng),
                    )
                    for i in range(10)
                ]
            )

        picks = {
            "leaf first":        (False, [9, 8, 7, 6, 5, 4, 3, 2, 1]),
            "root first":        (False, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "shuffled":          (False, [4, 9, 1, 7, 2, 8, 3, 6, 5]),
            "up between ends":   (False, [2, 9, 3, 8, 4, 7, 5, 6]),
            "up between leaves": (True, [1, 6, 2, 7, 3, 8, 4, 9, 5]),
        }
        compensations = {
            "none": lambda i, rng: False,
            "some": lambda i, rng: bool(rng.integers(0, 2)),
        }
        for index, (name, (leaves, pick)) in enumerate(picks.items()):
            for (kind, compensate), parent in itertools.product(compensations.items(), (None, "j0")):
                with self.subTest(order=name, compensation=kind, parent=parent):
                    seen.clear()
                    clip  = self._animated(chain((2 * index + 1) % 6, compensate, leaves))
                    poses = _per_frame(clip, lambda p: _one_at_a_time(p[pick], parent))
                    with mock.patch.object(module, "_cascades", spy):
                        clip[pick].set_parent(parent)
                    self._assert_matches(clip, poses)
                    self.assertEqual(seen, [True])

    def test_set_parent_refuses_a_parent_below_the_node(self):
        # one node at a time it recursed without end on a pose; the clip left
        # a parent cycle behind and said nothing. A node's own set_parent onto
        # itself is the smallest such loop: on a clip it rewrote the node's
        # channels on every frame and left it its own parent
        def held(h, frames):
            channels = ("scale", "rotate", "translate")
            if frames:
                return {c: np.array(getattr(h.frames, c)) for c in channels}
            return {c: np.array(getattr(h, c)) for c in channels}

        for frames in (0, 1, 3):
            for world_space in (True, False):
                with self.subTest(frames=frames, world_space=world_space):
                    rig    = self._rig(3)
                    deep   = next(n for n in rig if len(n.get_root()) > 2)
                    root   = deep.get_root()[0]
                    h      = rig if not frames else self._animated(rig, frames=frames)
                    root   = h[root.name]
                    deep   = h[deep.name]
                    before = (list(h.uuid), [n.parent_uuid for n in h], held(h, frames))
                    with self.assertRaisesRegex(ValueError, "hangs below it"):
                        root.set_parent(deep.name, world_space=world_space)
                    with self.assertRaisesRegex(ValueError, "hangs below it"):
                        TransformList([root]).set_parent(deep.name, world_space=world_space)
                    for own in (deep.name, deep.uuid):
                        with self.assertRaisesRegex(ValueError, "can not be its own parent"):
                            deep.set_parent(own, world_space=world_space)
                    self.assertEqual(list(h.uuid), before[0])
                    self.assertEqual([n.parent_uuid for n in h], before[1])
                    for channel, values in held(h, frames).items():
                        np.testing.assert_array_equal(values, before[2][channel], channel)

    def test_set_parent_refuses_a_name_whose_carriers_all_move_before_anything_moves(self):
        # "x" names a root and a child of s. Once s and the root x go to the
        # world, the root x lands behind s's branch, so the name finds the x
        # below s: one node at a time loops, and the call raises before it
        # moves anything. Naming the first x in the list as it stands raised
        # only once the first pass had moved every node
        for frames in (0, 1, 3):
            for world_space in (True, False):
                with self.subTest(frames=frames, world_space=world_space):
                    rig = HierarchyData(
                        [
                            TransformData("x", node_type="joint", translate=(0, 0, 2)),
                            TransformData(
                                "s",
                                node_type = "joint",
                                translate = (1, 0, 0),
                                rotate    = (10, 20, 30),
                                scale     = (1, 2, 1),
                            ),
                            TransformData("x", node_type="joint", translate=(0, 1, 0)),
                        ]
                    )
                    rig.list[2].parent_uuid = rig["s"].uuid
                    h                       = ClipData(rig, frames=frames) if frames else rig
                    owner                   = h.frames if frames else h
                    before = (
                        list(h.uuid),
                        [n.parent_uuid for n in h],
                        {c: np.array(getattr(owner, c)) for c in ("scale", "rotate", "translate")},
                    )
                    with self.assertRaisesRegex(ValueError, "hangs below it"):
                        TransformList([h.list[1], h.list[0]]).set_parent("x", world_space=world_space)
                    self.assertEqual(list(h.uuid), before[0])
                    self.assertEqual([n.parent_uuid for n in h], before[1])
                    for channel, values in before[2].items():
                        np.testing.assert_array_equal(getattr(owner, channel), values, channel)

    def _rig_against_one_at_a_time(self, nodes, pick, parent, cascades=None, animated=True):
        """``TransformList(pick).set_parent(parent)`` on a pose of ``nodes``
        and on a three frame clip of them, each against one node at a time;
        ``cascades``, when given, is what _cascades must answer on the clip.
        ``animated`` False holds every frame of the clip at the pose"""
        module = sys.modules[ClipData.__module__]
        real   = module._cascades
        seen   = []

        def spy(frames, *args):
            found = real(frames, *args)
            if getattr(frames.owner, "_blocks", None):
                seen.append(found)
            return found

        def links(h):
            """each node's name and where its parent sits in the list: two
            copies of a rig carry uuids of their own"""
            at = {id(node): i for i, node in enumerate(h.list)}
            return [(n.name, at.get(id(n.get_parent()), n.get_parent())) for n in h.list]

        pose = HierarchyData([n.copy() for n in nodes])
        want = HierarchyData([n.copy() for n in nodes])
        _one_at_a_time([want.list[i] for i in pick], parent)
        TransformList([pose.list[i] for i in pick]).set_parent(parent)
        self.assertEqual(links(pose), links(want))
        world = np.array([n.world_matrix for n in want])
        self.assertLess(np.abs(np.array([n.world_matrix for n in pose]) - world).max(), 1e-9)

        # the scales are left as the rig holds them: a wave on them would
        # shear every node and take the clip one at a time whatever the case
        clip  = HierarchyData([n.copy() for n in nodes])
        clip  = self._animated(clip, frames=3, scaled=False) if animated else ClipData(clip, frames=3)
        poses = _per_frame(clip, lambda p: _one_at_a_time([p.list[i] for i in pick], parent))
        with mock.patch.object(module, "_cascades", spy):
            TransformList([clip.list[i] for i in pick]).set_parent(parent)
        self._assert_matches(clip, poses)
        if cascades is not None:
            self.assertEqual(seen, [cascades])

    def test_set_parent_weighs_a_shear_by_how_far_the_nodes_below_sit(self):
        # scales uniform to within e shear by e, which the channels drop. A
        # node below moves by that times its distance: measured against the
        # translation it was ignored up to 1e-10 of the distance from the
        # origin, so the batch left the nodes below where one at a time
        # moves them, 1.8e-4 off at e = 1e-8, a thousand units out, on poses
        # too (they were exact one at a time). The smaller shears batch
        # within 1e-9 only weighed by the spread and held to 1e-11
        for e, far in (
            (1e-10, 100.0),
            (1e-9, 100.0),
            (1e-8, 1000.0),
            (1e-11, 1000.0),
            (1e-12, 1000.0),
            (3e-13, 10000.0),
        ):
            with self.subTest(e=e, far=far):
                rng   = np.random.default_rng(int(far) + int(-np.log10(e)))
                uuids = [hierarchy.generate_uuid() for _ in range(12)]
                nodes = [
                    TransformData(
                        f"j{i}",
                        uuid         = uuids[i],
                        parent_uuid  = uuids[i - 1] if i else None,
                        node_type    = "joint",
                        translate    = rng.normal(size=3) * far,
                        rotate       = rng.uniform(-180.0, 180.0, 3),
                        scale        = (1.0, 1.0 + e, 1.0 - e),
                        rotate_axis  = rng.uniform(-40.0, 40.0, 3),
                        joint_orient = rng.uniform(-90.0, 90.0, 3),
                        rotate_order = int(rng.integers(0, 6)),
                    )
                    for i in range(12)
                ]
                self._rig_against_one_at_a_time(nodes, range(1, 12), None, cascades=True)

    def test_set_parent_of_compensated_children_of_resized_nodes(self):
        # uniform scales shear nothing, but each node taken to the world
        # takes on its parents' scale, which resizes its compensated child
        # before the child's turn: _cascades has to see the new scale over
        # the compensated child (2.5 off when it did not)
        rng   = np.random.default_rng(3)
        uuids = [hierarchy.generate_uuid() for _ in range(8)]
        nodes = [
            TransformData(
                f"j{i}",
                uuid=uuids[i],
                parent_uuid=uuids[i - 1] if i else None,
                node_type="joint",
                translate=rng.normal(size=3),
                rotate=rng.uniform(-90.0, 90.0, 3),
                scale=(float(rng.uniform(0.5, 2.0)),) * 3,
                segment_scale_compensate=i % 2 == 0,
            )
            for i in range(8)
        ]
        self._rig_against_one_at_a_time(nodes, range(1, 8), None, cascades=True)

    def test_set_parent_of_every_other_node_down_a_sheared_chain(self):
        # each picked node hangs below a picked one through a node that is
        # not picked: a hit has to reach down the chain past the direct
        # child (4.2 off when it reached the child alone)
        rng   = np.random.default_rng(5)
        uuids = [hierarchy.generate_uuid() for _ in range(12)]
        nodes = [
            TransformData(
                f"j{i}",
                uuid        = uuids[i],
                parent_uuid = uuids[i - 1] if i else None,
                node_type   = "joint",
                translate   = rng.normal(size=3),
                rotate      = rng.uniform(-90.0, 90.0, 3),
                scale       = rng.uniform(0.6, 1.6, 3),
            )
            for i in range(12)
        ]
        self._rig_against_one_at_a_time(nodes, [1, 3, 5, 7, 9, 11], None, cascades=True)

    def _chain(self, count, far, scale, seed, top=None, compensate=None):
        """``count`` joints down a chain parented by uuid, about ``far``
        units apart, scaled ``scale(i)``, compensated where ``compensate(i)``
        says, every other channel random; below a transform ``top`` scales
        uniformly, when given"""
        rng   = np.random.default_rng(seed)
        nodes = []
        if top is not None:
            nodes.append(
                TransformData(
                    "top", node_type="transform", uuid=hierarchy.generate_uuid(), scale=(top,) * 3
                )
            )
        for i in range(count):
            nodes.append(
                TransformData(
                    f"j{i}",
                    uuid=hierarchy.generate_uuid(),
                    parent_uuid=nodes[-1].uuid if nodes else None,
                    node_type="joint",
                    translate=rng.normal(size=3) * far,
                    rotate=rng.uniform(-180.0, 180.0, 3),
                    scale=scale(i),
                    rotate_axis=rng.uniform(-40.0, 40.0, 3),
                    joint_orient=rng.uniform(-90.0, 90.0, 3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(compensate and compensate(i)),
                )
            )
        return nodes

    def test_set_parent_holds_a_dropped_shear_well_inside_the_bound(self):
        # a node a dropped shear moves sits at offset v from the node that
        # drops it, and moves by v @ inv(W3) @ (L3 - W3): up to three times
        # the largest entry times the reach. Turned so one column of that
        # matrix sums to 2.9 times it, x placed along that column's signs
        # 1000 units out, at e = 7e-13 the product is 6.0e-10 and x moves
        # 1.8e-9 one node at a time: held to 1e-9 the batch would miss it
        e, far = 7e-13, 1000.0
        uuids = [hierarchy.generate_uuid() for _ in range(3)]
        nodes = [
            TransformData(
                "p",
                uuid      = uuids[0],
                node_type = "transform",
                rotate    = (0.0, 0.0, 30.0),
                scale     = (1.0, 1.0 + e, 1.0 - e),
            ),
            TransformData(
                "a",
                uuid        = uuids[1],
                parent_uuid = uuids[0],
                node_type   = "joint",
                rotate      = (-95.684, 53.929, -75.522),
                translate   = (0.5, 0.2, 0.1),
            ),
            TransformData("x", uuid=uuids[2], parent_uuid=uuids[1], node_type="joint"),
        ]
        rig   = HierarchyData([n.copy() for n in nodes])
        W     = np.array(rig["a"].world_matrix)
        alone = rig["a"].copy()
        HierarchyData([alone])
        alone.world_matrix = W

        M      = np.linalg.solve(W[:3, :3], np.array(alone.world_matrix)[:3, :3] - W[:3, :3])
        column = np.abs(M).sum(axis=0).argmax()
        self.assertGreater(np.abs(M[:, column]).sum(), 2.9 * np.abs(M).max())
        nodes[2].translate = np.linalg.solve(W[:3, :3].T, far * np.sign(M[:, column]))
        self.assertGreater(np.abs(M).sum(axis=0).max() * far, 1.5e-9)
        self.assertLess(np.abs(M).max() * far, 0.7e-9)
        self._rig_against_one_at_a_time(nodes, [1, 2], None, cascades=True, animated=False)

    def test_set_parent_weighs_a_shear_in_the_nodes_own_axes(self):
        # under a transform scaled s a shear of e in a node's own axes is e
        # times s in the world's, while it moves the nodes below as far as
        # ever: weighed in the world's axes, a chain of near uniform scales
        # under a top node scaled 1e-3 or 1e-5 batched, 1.5e-9 and 4.5e-8
        # off one node at a time, and a near uniform node alone under one
        # scaled 1e-3 2.4e-9, on poses too (they were exact one at a time)
        for top, e, far in ((1e-3, 1e-10, 1000.0), (1e-5, 3e-8, 10000.0)):
            with self.subTest(top=top, e=e, far=far):
                nodes = self._chain(11, far, lambda i: (1.0, 1.0 + e, 1.0 - e), 4, top=top)
                self._rig_against_one_at_a_time(nodes, range(1, 12), None, cascades=True)

        # a near uniform node alone under the scaled top, the rest picked
        nodes = self._chain(11, 100.0, lambda i: (1.0, 1.0 + 3e-9, 1.0 - 3e-9) if i == 0 else None, 2, top=1e-3)
        self._rig_against_one_at_a_time(nodes, range(2, 12), None, cascades=True)

    def test_set_parent_weighs_a_shear_by_the_nodes_riding_on_a_picked_one(self):
        # j2 is picked after j1, so one node at a time it moves with the
        # shear j1's channels drop, and so does everything below it, picked
        # or not: j5, which nobody picked, 1000 or 10,000 units below j4,
        # or 100,000 units out under a j4 scaled 1e5, and j6 as far out
        # under a j5 that compensates a j4 scaled 1e-5. Weighed by the
        # spread of the picked nodes alone (2.2e-9 to 1.2e-8 off), or by a
        # bound of the branches that leaves out a scale or a compensation,
        # the batch left them where they were
        def out(distance):
            def place(nodes):
                nodes[5].translate = (distance, 0.3 * distance, -0.2 * distance)

            return place

        def scaled(nodes):
            nodes[4].scale     = (1e5,) * 3
            nodes[5].translate = (1.0, 0.3, -0.2)

        def compensated(nodes):
            nodes[4].scale                    = (1e-5,) * 3
            nodes[5].segment_scale_compensate = True
            nodes[6].translate                = (1e5, 3e4, -2e4)

        cases = {
            "1000 units out": (3e-12, out(1000.0)),
            "10,000 units out": (1e-12, out(10000.0)),
            "under a scale of 1e5": (1e-13, scaled),
            "compensating a scale of 1e-5": (1e-13, compensated),
        }
        for name, (e, place) in cases.items():
            with self.subTest(name, e=e):
                nodes = self._chain(7, 1.0, lambda i: (1.0, 1.0 + e, 1.0 - e) if i == 0 else None, 1)

                place(nodes)
                self._rig_against_one_at_a_time(nodes, [1, 2], None, cascades=True)

        # j5 10,000 units out again, listed before its parent: the branches
        # are walked parents first, so it counts whatever the list order.
        # Walked in list order, it was missed and the batch left it 7.4e-9
        # off one node at a time
        with self.subTest("10,000 units out, listed before its parent", e=1e-12):
            nodes = self._chain(7, 1.0, lambda i: (1.0, 1.0 + 1e-12, 1.0 - 1e-12) if i == 0 else None, 1)
            out(10000.0)(nodes)
            self._rig_against_one_at_a_time([nodes[5]] + nodes[:5] + nodes[6:], [2, 3], None, cascades=True)

        # 1000 joints nobody picked below j6, 20 units apart in a line: the
        # last 20,000 units out, while no local holds more than 20. The
        # branches' worlds put together from the locals alone batched it,
        # 3.7e-9 off one node at a time, once the stretch of the worlds
        # weighed in too: a node scaled 1e5 above shows in its own local
        with self.subTest("20,000 units out down a chain nobody picked", e=3e-13):
            nodes = self._chain(7, 1.0, lambda i: (1.0, 1.0 + 3e-13, 1.0 - 3e-13) if i == 0 else None, 1)
            for i in range(1000):
                nodes.append(
                    TransformData(
                        f"far{i}",
                        uuid        = hierarchy.generate_uuid(),
                        parent_uuid = nodes[-1].uuid,
                        node_type   = "joint",
                        translate   = (20.0, 0.0, 0.0),
                    )
                )
            self._rig_against_one_at_a_time(nodes, [1, 2], None, cascades=True, animated=False)

    def test_set_parent_counts_a_node_the_keep_test_moves(self):
        # the matrix setter keeps a node's channels when its new local is
        # within 1e-8 of the one it has: under a root 3e-9 off the origin
        # the hips keep theirs as they go to the world, which moves them,
        # and everything below them, by 3e-9. Held to 1e-10 of its distance
        # from the origin, that move batched for hips 100 and 1000 units up
        # and left the chain below them up to 9e-9 off one node at a time
        for d, up in ((3e-9, 100.0), (9e-9, 1000.0)):
            with self.subTest(d=d, up=up):
                rng   = np.random.default_rng(3)
                nodes = [
                    TransformData(
                        "root", node_type="transform", uuid=hierarchy.generate_uuid(), translate=(d, 0.0, 0.0)
                    )
                ]
                for i, name in enumerate(("hips", "spine", "chest", "neck", "head")):
                    nodes.append(
                        TransformData(
                            name,
                            node_type    = "joint",
                            uuid         = hierarchy.generate_uuid(),
                            parent_uuid  = nodes[-1].uuid,
                            translate    = (0.0, up if i == 0 else 10.0, 0.0),
                            rotate       = rng.uniform(-60.0, 60.0, 3),
                            joint_orient = rng.uniform(-30.0, 30.0, 3),
                            rotate_order = int(rng.integers(0, 6)),
                        )
                    )
                self._rig_against_one_at_a_time(nodes, range(1, 6), None, cascades=True, animated=False)

    def test_set_parent_counts_the_keep_test_of_a_node_a_picked_one_carries(self):
        # x, which is picked too, rides on a, which drops the shear p's scale
        # leaves (e): one node at a time x is solved from its world moved by
        # about e, the batch from the world it has. Where what the matrix
        # setter's keep test compares for x sits just below 1e-8, the batch
        # keeps x's channels and one node at a time solves them, a step about
        # 1e-8 tall. Each sits within 1e-12 of the step here; weighed by the
        # change alone, the batch left x and c off one node at a time, on
        # poses too (they were exact one at a time):
        # - going to the world, a's world turned so x's channels make x's
        #   world to within that: 1.0e-8 with c 1 unit out, 4.9e-8 5 out;
        # - then under q, a root scaled 100 that x compensates, x at the
        #   world's origin: x's channels with IS 1/100 against x's world
        #   over q's differ by x's translation times 0.99: 1.0e-6;
        # - under q, which a carries and the first pass moves, scaled 1.01,
        #   x a root that compensates it, where the two differ by x's
        #   translation less q's over 1.01: 1.0e-8;
        # - going to the world, a turn alone reaching x: 1.0e-8, as it is
        #   with the keep test's gain left off a turn
        from cgmath.transforms import matrix_to_euler

        def angles(R, order=0):
            M         = np.eye(4)
            M[:3, :3] = R
            return np.degrees(np.array(matrix_to_euler(M[None], order))[0])

        def rotation(degrees, order=0):
            return np.array(euler_to_matrix(np.radians([degrees]), order))[0][:3, :3]

        def turned(u, axis=(0.3, -0.5, 0.81)):
            # a turn of u radians about ``axis``
            axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
            K    = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            return np.eye(3) + np.sin(u) * K + (1.0 - np.cos(u)) * K @ K

        def under(rig, name, world):
            # the local that puts node ``name`` at ``world`` (3,)
            W = np.array(rig[name].get_parent().world_matrix)
            return np.linalg.solve(W[:3, :3].T, np.asarray(world, dtype=float) - W[3, :3])

        def just_below(nodes, gap, step=1e-8):
            # u in [0, 3e-8] where gap(nodes(u)) sits just below ``step``
            lo, hi = 0.0, 3e-8
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if gap(nodes(mid)) < step:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-30:
                    break
            built = nodes(lo)
            self.assertLess(gap(built), step)
            self.assertGreater(gap(built), step - 1e-12)
            return built

        def gap_of(name):
            # how far node ``name``'s channels make its world
            def gap(nodes):
                node = HierarchyData([n.copy() for n in nodes])[name]
                return float(np.abs(np.array(node.matrix) - np.array(node.world_matrix)).max())

            return gap

        def sheared(e, uuid):
            return TransformData(
                "p", uuid=uuid, node_type="transform", rotate=(10.0, 20.0, 30.0), scale=(1.0, 1.0 + e, 1.0 - e)
            )

        def to_the_world(far):
            # a at the origin, its world a turn of u about an axis: x's world
            # is its local turned by u
            P = rotation((10.0, 20.0, 30.0))

            def nodes(u):
                ids  = [hierarchy.generate_uuid() for _ in range(4)]
                turn = turned(u)
                return [
                    sheared(1e-12, ids[0]),
                    TransformData("a", uuid=ids[1], parent_uuid=ids[0], node_type="joint", rotate=angles(turn @ P.T)),
                    TransformData(
                        "x",
                        uuid        = ids[2],
                        parent_uuid = ids[1],
                        node_type   = "joint",
                        rotate      = (37.0, -52.0, 118.0),
                        translate   = (0.3, -0.2, 0.1),
                    ),
                    TransformData("c", uuid=ids[3], parent_uuid=ids[2], node_type="joint", translate=(far, 0.0, 0.0)),
                ]

            return nodes, gap_of("x")

        def under_q(scale, e, carried):
            # x at u along x from where its channels under q, with IS 1/scale,
            # make its world over q's; q a root, or under a and turned square
            def nodes(u):
                ids = [hierarchy.generate_uuid() for _ in range(5)]
                built = [
                    sheared(e, ids[0]),
                    TransformData(
                        "a",
                        uuid        = ids[1],
                        parent_uuid = ids[0],
                        node_type   = "joint",
                        rotate      = (-95.7, 53.9, -75.5),
                        translate   = (0.5, 0.2, 0.1),
                    ),
                    TransformData(
                        "q",
                        uuid        = ids[4],
                        parent_uuid = ids[1] if carried else None,
                        node_type   = "transform",
                        scale       = (scale,) * 3,
                    ),
                    TransformData(
                        "x",
                        uuid=ids[2],
                        parent_uuid=None if carried else ids[1],
                        node_type="joint",
                        rotate=(37.0, -52.0, 118.0),
                        segment_scale_compensate=True,
                    ),
                    TransformData("c", uuid=ids[3], parent_uuid=ids[2], node_type="joint", translate=(1.0, 0.0, 0.0)),
                ]
                rig = HierarchyData(built)
                if carried:
                    A = np.array(rig["a"].world_matrix)
                    U, _, Vt = np.linalg.svd(A[:3, :3])
                    rig["q"].rotate    = angles((U @ Vt).T)
                    rig["q"].translate = np.linalg.solve(A[:3, :3].T, np.array([0.1, 0.0, 0.0]))
                q                  = np.array(rig["q"].world_matrix)[3, :3]
                spot               = (np.array([u, 0.0, 0.0]) - q / scale) / (1.0 - 1.0 / scale)
                rig["x"].translate = under(rig, "x", spot) if rig["x"].get_parent() else spot
                return [n.copy() for n in rig]

            def gap(nodes):
                rig = HierarchyData([n.copy() for n in nodes])
                rig["q" if carried else "x"].set_parent(None)
                x, q = rig["x"], rig["q"]
                world         = np.array(x.world_matrix)
                x.parent_uuid = q.uuid
                return float(np.abs(np.array(x.matrix) - world @ np.linalg.inv(np.array(q.world_matrix))).max())

            return nodes, gap

        def a_turn_alone():
            # a kept at the origin under r, a root turned by about 7e-12
            # radians, so one node at a time a's world goes back by a turn and
            # nothing else; x below n, which nobody picks, turned by u
            def nodes(u):
                ids  = [hierarchy.generate_uuid() for _ in range(5)]
                turn = turned(u)
                return [
                    TransformData("r", uuid=ids[0], node_type="transform", rotate=(3e-10, -2.25e-10, 1.5e-10)),
                    TransformData("a", uuid=ids[1], parent_uuid=ids[0], node_type="joint"),
                    TransformData("n", uuid=ids[2], parent_uuid=ids[1], node_type="transform", rotate=angles(turn)),
                    TransformData(
                        "x",
                        uuid        = ids[3],
                        parent_uuid = ids[2],
                        node_type   = "joint",
                        rotate      = (37.0, -52.0, 118.0),
                        translate   = (0.3, -0.2, 0.1),
                    ),
                    TransformData("c", uuid=ids[4], parent_uuid=ids[3], node_type="joint", translate=(1.0, 0.0, 0.0)),
                ]

            return nodes, gap_of("x")

        cases = {
            "to the world, c 1 unit out":  (to_the_world(1.0), [1, 2], None),
            "to the world, c 5 units out": (to_the_world(5.0), [1, 2], None),
            "under q, x compensating it":  (under_q(100.0, 1e-12, False), [1, 3], "q"),
            "under q, which a carries":    (under_q(1.01, 1e-13, True), [1, 2, 3], "q"),
            "a turn alone, to the world":  (a_turn_alone(), [1, 3], None),
        }
        for name, ((nodes, gap), pick, parent) in cases.items():
            with self.subTest(name):
                built = just_below(nodes, gap)
                self._rig_against_one_at_a_time(built, pick, parent, cascades=True, animated=False)

        # y, picked, rides on x, picked too and 0.13 degrees off gimbal, and
        # x on a, which goes back by s's stretch of 1e-13 across the axes. One
        # node at a time x's split gains about 440 on that, which turns x's
        # world, and so y's, by about 6e-11; y's keep test sits 2.2e-11 above
        # its step. x's world is square (n2's stretch, put along the axes by
        # n1 and n3, is one x's scale undoes), so what reaches y is a's change
        # and x's own, tiny. Weighed by those alone and not by what x makes of
        # a's, the batch left y and c 1.0e-8 off one node at a time, on poses
        # too (they were exact one at a time)
        def through_a_picked_node(u, e=1e-13, off=0.13, order=4):
            ids = [hierarchy.generate_uuid() for _ in range(11)]
            R0  = rotation((-47.37, 54.23, 14.79))
            S1  = R0.T @ np.diag([1.0, 1.0 + e, 1.0 - e]) @ R0 - np.eye(3)
            D   = np.diag([-0.8, -0.13, -0.04]) * e
            w, V = np.linalg.eigh((D - S1) / e)
            if np.linalg.det(V) < 0:
                V[:, 2] *= -1.0
            middle = np.radians([-36.98, 5.69, -23.59])
            middle[MAYA_EA[order, 1]] = np.radians(90.0 - off)
            G  = np.array(euler_to_matrix(middle[None], order))[0][:3, :3]
            T  = turned(u, (-0.67, -1.06, -0.39))
            Rm = rotation((-102.08, 70.37, -115.9))

            def node(k, name, parent, node_type="transform", **channels):
                up = None if parent is None else ids[parent]
                return TransformData(name, uuid=ids[k], parent_uuid=up, node_type=node_type, **channels)

            return [
                node(0, "r", None, rotate=(-47.37, 54.23, 14.79)),
                node(1, "s", 0, scale=(1.0, 1.0 + e, 1.0 - e)),
                node(2, "t", 1, rotate=angles(R0.T)),
                node(3, "a", 2, "joint"),
                node(4, "n1", 3, rotate=angles(V.T)),
                node(5, "n2", 4, scale=tuple(1.0 + w * e)),
                node(6, "n3", 5, rotate=angles(V)),
                node(7, "m", 6, rotate=(-102.08, 70.37, -115.9)),
                node(
                    8,
                    "x",
                    7,
                    "joint",
                    rotate       = angles(T @ Rm.T @ T.T @ G, order),
                    joint_orient = angles(G.T @ T),
                    rotate_order = order,
                    scale        = tuple(1.0 / (1.0 + np.diag(D))),
                ),
                node(9, "y", 8, "joint", rotate=(-77.69, 53.48, 70.64), translate=(-0.06, 0.01, 0.46)),
                node(10, "c", 9, "joint", translate=(1.0, 0.0, 0.0)),
            ]

        with self.subTest("through a picked node near gimbal"):
            built = just_below(through_a_picked_node, gap_of("y"), 1e-8 + 2.2e-11)
            self._rig_against_one_at_a_time(built, [3, 8, 9], None, cascades=True, animated=False)

    def test_set_parent_weighs_a_new_scale_by_how_far_the_nodes_below_sit(self):
        # a node taken to the world takes on its parents' scale, and its
        # compensated child, dividing by it, resizes everything below the
        # child. Uniform scales within 1e-5 of 1 and 1 + 1e-9 a thousand
        # units out have to go one at a time; a parent scaled 1 + 5e-13
        # over a node 10,000 units below batched, held to 1e-12 of the
        # scale whatever the reach, 4.1e-9 off one node at a time
        cases = {
            "within 1e-5":   (8, 1.0, 1e-5),
            "1e-9, far out": (8, 1000.0, 1e-9),
        }
        for name, (count, far, d) in cases.items():
            with self.subTest(name):
                rng   = np.random.default_rng(3)
                sizes = 1.0 + d * rng.uniform(-1.0, 1.0, count)
                nodes = self._chain(count, far, lambda i: (sizes[i],) * 3, 3, compensate=lambda i: i % 2 == 0)
                self._rig_against_one_at_a_time(nodes, range(1, count), None, cascades=True)

        with self.subTest("5e-13 above, 10,000 units below"):
            nodes = self._chain(5, 1.0, lambda i: (1.3,) * 3 if i == 1 else None, 4, compensate=lambda i: i == 2)

            nodes[0].scale     = (1.0 + 5e-13,) * 3
            nodes[4].translate = (10000.0, 4000.0, -3000.0)
            self._rig_against_one_at_a_time(nodes, [1, 4], None, cascades=True)

    def test_set_parent_weighs_a_shear_by_the_world_scale_below(self):
        # a dropped shear or a new scale turns the axes of every node below
        # by the change times their own world scale, however close they sit:
        # weighed by the spread of the positions alone, rigs scaled 1000 or
        # 5000 a few units across batched, 2.6e-9 to 5.2e-9 off one node at a
        # time, on poses too (they were exact one at a time)
        def joints(nodes, names, apart, seed, **changes):
            rng = np.random.default_rng(seed)
            for name in names:
                channels = dict(
                    translate    = rng.normal(size=3) * apart,
                    rotate       = rng.uniform(-180.0, 180.0, 3),
                    rotate_axis  = rng.uniform(-40.0, 40.0, 3),
                    joint_orient = rng.uniform(-90.0, 90.0, 3),
                    rotate_order = int(rng.integers(0, 6)),
                )
                channels.update(changes.get(name, {}))
                nodes.append(
                    TransformData(
                        name,
                        uuid        = hierarchy.generate_uuid(),
                        parent_uuid = nodes[-1].uuid,
                        node_type   = "joint",
                        **channels,
                    )
                )
            return nodes

        def sheared(s, e):
            return TransformData(
                "p",
                uuid      = hierarchy.generate_uuid(),
                node_type = "transform",
                rotate    = (10.0, 20.0, 30.0),
                scale     = (s, s * (1.0 + e), s * (1.0 - e)),
            )

        # a top scaled s, sheared by e: a > b > c a few world units apart
        for s, e, apart in ((1000.0, 3e-12, 0.001), (5000.0, 1e-12, 0.0002)):
            with self.subTest("sheared top", s=s, e=e):
                nodes = joints([sheared(s, e)], "abc", apart, 5)
                self._rig_against_one_at_a_time(nodes, [1, 2], None, cascades=True)

        # a top scaled 1 + 3e-12 over a scaled to 1000: a to the world takes
        # it on, and the compensated c resizes b, scaled 1000 too, before b's
        # turn
        with self.subTest("a new scale over a compensated child"):
            nodes = joints(
                [sheared(1.0 + 3e-12, 0.0)],
                "acbd",
                0.001,
                6,
                a = {"scale": (1000.0,) * 3},
                c = {"segment_scale_compensate": True},
                b = {"scale": (1000.0,) * 3, "translate": (0.6, -0.3, 0.2)},
            )
            self._rig_against_one_at_a_time(nodes, [1, 3], None, cascades=True)

        # a top sheared by 3e-12 over a > b > c, c scaled 1000 and picked by
        # nobody: only the bound of the branches and their worlds see it
        with self.subTest("a node nobody picked scaled 1000"):
            nodes = joints([sheared(1.0, 3e-12)], "abc", 0.001, 9, c={"scale": (1000.0,) * 3})
            self._rig_against_one_at_a_time(nodes, [1, 2], None, cascades=True)

        # a top scaled 1000 sheared by 3e-12 and d where a sits (an end or a
        # twist joint), or a top sheared by 3e-12 and d at n, scaled 1000
        # next to a
        with self.subTest("a node where the shear's node sits"):
            nodes = joints([sheared(1000.0, 3e-12)], "ad", 0.0, 7)
            self._rig_against_one_at_a_time(nodes, [1, 2], None, cascades=True)
        with self.subTest("a node at a scale of 1000 next to it"):
            nodes = joints(
                [sheared(1.0, 3e-12)],
                "and",
                0.5,
                8,
                n = {"scale": (1000.0,) * 3},
                d = {"translate": (0.0, 0.0, 0.0)},
            )
            self._rig_against_one_at_a_time(nodes, [1, 3], None, cascades=True)

    def test_set_parent_weighs_a_shear_by_how_far_the_split_moves_what_it_reaches(self):
        # x, which is picked too, rides on a: one node at a time it is solved
        # from its world sheared by what a's channels drop, and the euler
        # split moves it by that over the cosine of its middle angle, or by
        # more where the matrix is far from square. Put under a parent, what
        # the first pass leaves reaches x's new local through the parent's
        # world. Weighed by the change alone, the batch left x and c off one
        # node at a time, on poses too (they were exact one at a time):
        # - 0.001 degrees off gimbal, below a top sheared by 1e-12: 5.1e-8
        #   in xyz, 6.4e-8 in zyx;
        # - under a parent q scaled (100, 1, 0.01): 3.0e-8 and 2.4e-8, and
        #   9.4e-5 in xyz with x square to q, whose new local the split then
        #   holds whole, where a turn reaches it as a shear 1e4 times its size;
        # - under y, which a carries, 0.003 degrees off gimbal there: 9.7e-9
        #   and 7.8e-9;
        # - 1e-5 degrees off gimbal under the top, sheared by 1e-14, c 100
        #   units out: the round-off of the split alone, 7.6e-8 in zyx;
        # - at the cosine where matrix_to_euler snaps, the batch on one side
        #   and one node at a time on the other: 1.5e-6 and 4.6e-7;
        # - 0.001 degrees off gimbal below c, which compensates a, under a
        #   top scaled (1, 1 + 1e-12, 1 + 2e-12) that turns nothing: a's new
        #   scale alone reaches x, 9.8e-9 and 2.1e-8
        from cgmath.transforms import matrix_to_euler

        def facing(node, world, order, middle, seed):
            # turns the joint so the rotation its matrix setter splits, for
            # its world or for its local under the 3x3 world, has the cosine
            # ``middle`` in its middle angle, the other two random; none at
            # all when ``middle`` is "square"
            angles = np.radians(np.random.default_rng(seed).uniform(-170.0, 170.0, 3))
            if middle == "square":
                angles = np.zeros(3)
            elif middle is not None:
                angles[MAYA_EA[order, 1]] = np.arccos(middle)
            orient = np.radians(np.asarray(node.joint_orient, dtype=float))
            orient = np.array(euler_to_matrix(orient[None], 0))[0][:3, :3]
            want   = np.array(euler_to_matrix(angles[None], order))[0][:3, :3] @ orient @ world
            above  = np.array(node.get_parent().world_matrix)[:3, :3] if node.get_parent() else np.eye(3)
            U, _, Vt = np.linalg.svd(want @ np.linalg.inv(above) @ orient.T)
            turn              = np.eye(4)
            turn[:3, :3]      = U @ Vt
            node.rotate_order = order
            node.rotate       = np.degrees(np.array(matrix_to_euler(turn[None], order))[0])

        def rig(parent, e, order, middle, seed):
            # p, turned and scaled (1, 1 + e, 1 - e), over a > x > c; x a
            # root under y below a, or q apart, by the parent
            uuids = [hierarchy.generate_uuid() for _ in range(5)]
            nodes = [
                TransformData(
                    "p",
                    uuid      = uuids[0],
                    node_type = "transform",
                    rotate    = (10.0, 20.0, 30.0),
                    scale     = (1.0, 1.0 + e, 1.0 - e),
                ),
                TransformData(
                    "a",
                    uuid        = uuids[1],
                    parent_uuid = uuids[0],
                    node_type   = "joint",
                    rotate      = (-95.7, 53.9, -75.5),
                    translate   = (0.5, 0.2, 0.1),
                ),
            ]
            x = TransformData("x", uuid=uuids[2], parent_uuid=uuids[1], node_type="joint", translate=(0.3, -0.4, 0.2))
            if parent == "y":
                nodes.append(
                    TransformData(
                        "y",
                        uuid        = uuids[4],
                        parent_uuid = uuids[1],
                        node_type   = "joint",
                        rotate      = (20.0, -35.0, 50.0),
                        translate   = (0.2, 0.6, -0.3),
                    )
                )
                x = TransformData(
                    "x",
                    uuid         = uuids[2],
                    node_type    = "joint",
                    translate    = (1.5, -0.4, 0.2),
                    joint_orient = (30.0, -20.0, 45.0),
                )
            far = 100.0 if parent == "p" else 1.0
            nodes += [x, TransformData("c", uuid=uuids[3], parent_uuid=uuids[2], node_type="joint", translate=(far, 0.0, 0.0))]
            if parent == "q":
                nodes.append(
                    TransformData(
                        "q",
                        uuid      = uuids[4],
                        node_type = "transform",
                        rotate    = (40.0, -25.0, 70.0),
                        scale     = (100.0, 1.0, 0.01),
                        translate = (2.0, 1.0, -1.0),
                    )
                )
            rig   = HierarchyData(nodes)
            world = np.eye(3)
            if parent in ("p", "y") or middle == "square":
                world = np.array(rig[parent].world_matrix)[:3, :3]
            if middle == "square":
                world = world / np.linalg.norm(world, axis=1)[:, None]
            facing(rig["x"], world, order, middle, seed)
            return [n.copy() for n in rig]

        snap = float(np.finfo(np.float32).eps)
        cases = {
            "0.001 degrees off gimbal": (None, 1e-12, np.sin(np.radians(1e-3)), ((0, 0), (5, 0))),
            "under a parent scaled unevenly": ("q", 1e-12, None, ((0, 0), (5, 0))),
            "square to a parent scaled unevenly": ("q", 1e-12, "square", ((0, 0),)),
            "under a picked parent the first pass moves": ("y", 5e-13, np.sin(np.radians(3e-3)), ((0, 2), (5, 2))),
            "the split's round-off": ("p", 1e-14, np.sin(np.radians(1e-5)), ((5, 0),)),
            "where the split snaps": (None, 5e-13, snap * (1.0 - 2.5e-9), ((0, 0), (5, 2))),
        }
        for name, (parent, e, middle, runs) in cases.items():
            for order, seed in runs:
                with self.subTest(name, order=order):
                    nodes = rig(parent, e, order, middle, seed)
                    pick  = [1, 2, 3] if parent == "y" else [1, 2]
                    self._rig_against_one_at_a_time(nodes, pick, parent, cascades=True, animated=False)

        # the gain read off every frame: x clear of gimbal on the first of
        # three, 0.001 degrees off it on the others
        with self.subTest("near gimbal on a later frame"):
            module = sys.modules[ClipData.__module__]
            real   = module._cascades
            seen   = []

            def spy(frames, *args):
                seen.append(real(frames, *args))
                return seen[-1]

            nodes = rig(None, 1e-12, 0, np.sin(np.radians(1e-3)), 0)
            clear = rig(None, 1e-12, 0, None, 0)
            clip  = ClipData(HierarchyData([n.copy() for n in nodes]), frames=3)
            x     = clip.frames["x"]
            turns = np.array(x.rotate)

            turns[0, 0] = clear[2].rotate
            x.rotate    = turns
            poses       = _per_frame(clip, lambda p: _one_at_a_time([p.list[1], p.list[2]], None))
            with mock.patch.object(module, "_cascades", spy):
                TransformList([clip.list[1], clip.list[2]]).set_parent(None)
            self._assert_matches(clip, poses)
            self.assertEqual(seen, [True])

        # x and x1 hang off n, which a carries and nobody picks: x1's climb
        # marks n first, and x's stops there and reads what n has. Read as
        # nothing, x 0.001 degrees off gimbal was not weighed, 5.1e-8 off
        # in xyz and 6.4e-8 in zyx
        for order in (0, 5):
            with self.subTest("a node off gimbal beside another below one nobody picked", order=order):
                nodes = rig(None, 1e-12, order, None, 0)
                nodes.insert(
                    2,
                    TransformData(
                        "n",
                        uuid        = hierarchy.generate_uuid(),
                        parent_uuid = nodes[1].uuid,
                        node_type   = "joint",
                        rotate      = (20.0, -35.0, 50.0),
                        translate   = (0.2, 0.6, -0.3),
                    ),
                )
                nodes.insert(
                    3,
                    TransformData(
                        "x1",
                        uuid        = hierarchy.generate_uuid(),
                        parent_uuid = nodes[2].uuid,
                        node_type   = "joint",
                        rotate      = (5.0, 15.0, -25.0),
                        translate   = (0.4, 0.1, 0.0),
                    ),
                )
                nodes[4].parent_uuid = nodes[2].uuid
                built                = HierarchyData([n.copy() for n in nodes])
                facing(built["x"], np.eye(3), order, np.sin(np.radians(1e-3)), 0)
                self._rig_against_one_at_a_time([n.copy() for n in built], [1, 3, 4], None, cascades=True, animated=False)

        # x turned by a rotate_axis and a joint_orient, its rotate set so
        # what its matrix setter splits going to the world is 1e-4 degrees
        # off gimbal: weighed on rotate_axis @ that instead, it batched, up
        # to 3.0e-7 off
        def polar(M):
            U, _, Vt = np.linalg.svd(M)
            return U @ Vt

        for order, seed in ((0, 1), (5, 1)):
            with self.subTest("off gimbal through a rotate_axis and a joint_orient", order=order):
                built = HierarchyData(rig(None, 1e-12, order, None, seed))
                x     = built["x"]

                x.rotate_axis  = (25.0, -40.0, 60.0)
                x.joint_orient = (30.0, -20.0, 45.0)
                angles = np.radians(np.random.default_rng(seed).uniform(-170.0, 170.0, 3))
                angles[MAYA_EA[order, 1]] = np.radians(90.0 - 1e-4)
                G  = np.array(euler_to_matrix(angles[None], order))[0][:3, :3]
                JO = np.array(euler_to_matrix(np.radians([x.joint_orient]), 0))[0][:3, :3]
                A  = polar(np.array(built["a"].world_matrix)[:3, :3])
                R  = np.eye(4)

                R[:3, :3] = G @ JO @ A.T @ JO.T
                x.rotate  = np.degrees(np.array(matrix_to_euler(R[None], order))[0])
                self._rig_against_one_at_a_time([n.copy() for n in built], [1, 2], None, cascades=True, animated=False)

        # a turn passes the split as it is, however near gimbal: a, kept at
        # the origin under a root turned by about 5e-14 radians, turns x,
        # 0.57 degrees off gimbal, by that much, and c 100 units out by 100
        # times it. Weighed by the split's gain like the rest of a change,
        # it went one node at a time where the batch is within 1e-11
        for order in (0, 5):
            with self.subTest("a turn reaching a node near gimbal", order=order):
                nodes              = rig(None, 0.0, order, None, 0)
                nodes[0].rotate    = (2e-12, -1.5e-12, 1e-12)
                nodes[0].scale     = (1.0, 1.0, 1.0)
                nodes[1].translate = (0.0, 0.0, 0.0)
                nodes[3].translate = (100.0, 0.0, 0.0)
                built              = HierarchyData([n.copy() for n in nodes])
                facing(built["x"], np.eye(3), order, np.sin(np.radians(0.57)), 0)
                self._rig_against_one_at_a_time([n.copy() for n in built], [1, 2], None, cascades=False, animated=False)

        # a turn of about 7e-12 radians reaching x 1e-12 short of where the
        # split snaps carries it across: the snap weighs a turn as it weighs
        # the rest. Weighed without the snap, by 1 and what the turn takes
        # along of a split far from square alone, the batch was 2.3e-7 off
        # one node at a time
        with self.subTest("a turn reaching a node at the snap"):
            nodes              = rig(None, 0.0, 0, None, 0)
            nodes[0].rotate    = (3e-10, -2.25e-10, 1.5e-10)
            nodes[0].scale     = (1.0, 1.0, 1.0)
            nodes[1].translate = (0.0, 0.0, 0.0)
            built              = HierarchyData([n.copy() for n in nodes])
            facing(built["x"], np.eye(3), 0, snap - 1e-12, 0)
            self._rig_against_one_at_a_time([n.copy() for n in built], [1, 2], None, cascades=True, animated=False)

        # a turn of about 7e-14 radians reaching x under n, scaled (1, 100,
        # 0.01): what x's split reads off is far from square, and the turn
        # takes that along, a change that is not one, which the split gains
        # on. Passed as it is, the batch was 2.3e-9 and 1.8e-9 off one node
        # at a time
        for rotate in ((-83.2745, 46.2769, -141.877), (-100.6545, 52.8268, -144.2076)):
            with self.subTest("a turn reaching a split far from square", rotate=rotate):
                t     = 3e-12
                uuids = [hierarchy.generate_uuid() for _ in range(5)]
                nodes = [
                    TransformData("r", uuid=uuids[0], node_type="transform", rotate=(t, -0.75 * t, 0.5 * t)),
                    TransformData(
                        "a", uuid=uuids[1], parent_uuid=uuids[0], node_type="joint", rotate=(-95.7, 53.9, -75.5)
                    ),
                    TransformData(
                        "n",
                        uuid        = uuids[2],
                        parent_uuid = uuids[1],
                        node_type   = "transform",
                        rotate      = (20.0, -35.0, 50.0),
                        scale       = (1.0, 100.0, 0.01),
                        translate   = (0.2, 0.6, -0.3),
                    ),
                    TransformData(
                        "x",
                        uuid         = uuids[3],
                        parent_uuid  = uuids[2],
                        node_type    = "joint",
                        rotate       = rotate,
                        translate    = (0.3, -0.4, 0.2),
                        rotate_order = 5,
                    ),
                    TransformData("c", uuid=uuids[4], parent_uuid=uuids[3], node_type="joint", translate=(1.0, 0.0, 0.0)),
                ]
                self._rig_against_one_at_a_time(nodes, [1, 3], None, cascades=True, animated=False)

        # a stretch with nothing on the diagonal: a, kept at the origin under
        # a scale (1, 1 + 1e-12, 1 - 1e-12) turned 45 degrees about x, goes
        # back by a change that is all off the diagonal and no turn, which
        # x's split, 0.001 degrees off gimbal, gains on. Its rest read off
        # the diagonal alone, the batch was 9.7e-8 off in xyz
        for order in (0, 5):
            with self.subTest("a stretch across the axes reaching a node near gimbal", order=order):
                uuids = [hierarchy.generate_uuid() for _ in range(6)]
                built = HierarchyData(
                    [
                        TransformData("r", uuid=uuids[0], node_type="transform", rotate=(45.0, 0.0, 0.0)),
                        TransformData(
                            "s",
                            uuid        = uuids[1],
                            parent_uuid = uuids[0],
                            node_type   = "transform",
                            scale       = (1.0, 1.0 + 1e-12, 1.0 - 1e-12),
                        ),
                        TransformData(
                            "u", uuid=uuids[2], parent_uuid=uuids[1], node_type="transform", rotate=(-45.0, 0.0, 0.0)
                        ),
                        TransformData(
                            "a", uuid=uuids[3], parent_uuid=uuids[2], node_type="joint", rotate=(-95.7, 53.9, -75.5)
                        ),
                        TransformData(
                            "x", uuid=uuids[4], parent_uuid=uuids[3], node_type="joint", translate=(0.3, -0.4, 0.2)
                        ),
                        TransformData("c", uuid=uuids[5], parent_uuid=uuids[4], node_type="joint", translate=(1.0, 0.0, 0.0)),
                    ]
                )
                facing(built["x"], np.eye(3), order, np.sin(np.radians(1e-3)), 0)
                self._rig_against_one_at_a_time([n.copy() for n in built], [3, 4], None, cascades=True, animated=False)

        # a, kept at the origin, goes back by exactly what is above it, a
        # turn or a stretch, which reaches x, picked too
        def kept_under(above, x, rest=()):
            uuids = [hierarchy.generate_uuid() for _ in range(len(above) + 3)]
            nodes = [
                TransformData(f"t{k}", uuid=uuids[k], parent_uuid=uuids[k - 1] if k else None, node_type="transform", **channels)
                for k, channels in enumerate(above)
            ]
            nodes += [
                TransformData(
                    "a", uuid=uuids[-3], parent_uuid=uuids[-4], node_type="joint", rotate=(-95.7, 53.9, -75.5)
                ),
                TransformData(
                    "x", uuid=uuids[-2], parent_uuid=uuids[-3], node_type="joint", translate=(0.3, -0.4, 0.2), **x
                ),
                TransformData("c", uuid=uuids[-1], parent_uuid=uuids[-2], node_type="joint", translate=(1.0, 0.0, 0.0)),
            ]
            return HierarchyData(nodes + [node.copy() for node in rest])

        # a stretch of 1e-12 along the axes is one across x's own, which
        # carry a joint_orient, 0.001 degrees off gimbal: its rest read
        # without the diagonal of the change, the batch was 8.1e-8 off in xyz
        # (4.8e-8 in zyx)
        for order in (0, 5):
            with self.subTest("a stretch along the axes reaching a node near gimbal", order=order):
                built = kept_under([{"scale": (1.0, 1.0 + 1e-12, 1.0 - 1e-12)}], {"joint_orient": (30.0, -20.0, 45.0)})
                facing(built["x"], np.eye(3), order, np.sin(np.radians(1e-3)), 4)
                self._rig_against_one_at_a_time([n.copy() for n in built], [1, 2], None, cascades=True, animated=False)

        # then put under q, scaled (70, 1, 1 / 70): a turn of about 4.7e-12
        # radians, or a stretch of 4e-12 across the axes, weighed by the
        # first pass alone, batched 2.4e-9 and 1.4e-9 off
        q = TransformData(
            "q", node_type="transform", rotate=(40.0, -25.0, 70.0), scale=(70.0, 1.0, 1.0 / 70.0), translate=(2.0, 1.0, -1.0)
        )
        cases = {
            "a turn": [{"rotate": (2e-10, -1.5e-10, 1e-10)}],
            "a stretch": [
                {"rotate": (45.0, 0.0, 0.0)},
                {"scale": (1.0, 1.0 + 4e-12, 1.0 - 4e-12)},
                {"rotate": (-45.0, 0.0, 0.0)},
            ],
        }
        for name, above in cases.items():
            with self.subTest(f"{name} reaching a node put under a parent scaled unevenly"):
                built = kept_under(above, {"rotate": (37.0, -52.0, 118.0)}, [q])
                pick  = [len(above), len(above) + 1]
                self._rig_against_one_at_a_time([n.copy() for n in built], pick, "q", cascades=True, animated=False)

        # a new scale over a compensated child, nothing sheared
        for order in (0, 5):
            with self.subTest("a new scale reaching it through a compensated child", order=order):
                uuids = [hierarchy.generate_uuid() for _ in range(5)]
                rig = HierarchyData(
                    [
                        TransformData(
                            "top",
                            uuid      = uuids[0],
                            node_type = "transform",
                            translate = (5.0, 1.0, 2.0),
                            rotate    = (10.0, 20.0, 30.0),
                            scale     = (1.0, 1.0 + 1e-12, 1.0 + 2e-12),
                        ),
                        TransformData("a", uuid=uuids[1], parent_uuid=uuids[0], node_type="joint", translate=(0.5, 0.2, 0.1)),
                        TransformData(
                            "c",
                            uuid=uuids[2],
                            parent_uuid=uuids[1],
                            node_type="joint",
                            translate=(0.3, 0.1, 0.0),
                            rotate=(30.0, -20.0, 50.0),
                            segment_scale_compensate=True,
                        ),
                        TransformData("x", uuid=uuids[3], parent_uuid=uuids[2], node_type="joint", translate=(0.3, -0.4, 0.2)),
                        TransformData("d", uuid=uuids[4], parent_uuid=uuids[3], node_type="joint", translate=(1.0, 0.0, 0.0)),
                    ]
                )
                facing(rig["x"], np.eye(3), order, np.sin(np.radians(1e-3)), 0)
                self._rig_against_one_at_a_time([n.copy() for n in rig], [1, 3], None, cascades=True, animated=False)

    def test_set_parent_onto_a_name_a_picked_branch_carries(self):
        # "x" names a child of the picked s and a root: once s's branch moves
        # behind the root x, the second pass finds the root. Resolved before
        # the move, the cycle check named the x below s and refused it
        s = TransformData("s", node_type="joint", uuid=hierarchy.generate_uuid(), translate=(1, 0, 0))
        nodes = [
            s,
            TransformData("x", node_type="joint", parent_uuid=s.uuid, translate=(0, 1, 0)),
            TransformData("x", node_type="joint", translate=(0, 0, 2)),
            TransformData("b", node_type="joint", translate=(3, 0, 0)),
        ]
        self._rig_against_one_at_a_time(nodes, [0, 3], "x")

    def test_set_parent_keeps_a_curve_it_does_not_rebuild(self):
        # a root that goes to the world, and a node already under an identity
        # parent, keep their world on every frame: their authored curves,
        # stepping by whole turns, come back bit for bit. Filtering every
        # picked node's curve rewrote them to [0, -10, 0, -10, -20]
        steps = np.array([0.0, 350.0, 0.0, 350.0, 700.0])
        edits = {
            "the whole clip":   lambda h: h.set_parent(None),
            "the root alone":   lambda h: h["a"].set_parent(None),
            "a selection":      lambda h: h[[0, 1]].set_parent(None),
            "already under it": lambda h: h["b"].set_parent("a"),
        }
        for name, edit in edits.items():
            with self.subTest(edit=name):
                rig = HierarchyData(
                    [
                        TransformData("a", node_type="joint"),
                        TransformData("b", node_type="joint", translate=(0, 1, 0)),
                    ]
                )
                rig["b"].parent_uuid = rig["a"].uuid
                clip                 = ClipData(rig, frames=5)
                rotate               = np.zeros((5, 2, 3))
                rotate[:, 0, 0]      = steps
                rotate[:, 1, 2]      = steps
                clip.frames.rotate   = rotate
                poses                = _per_frame(clip, edit)
                edit(clip)
                # one pose at a time, the root keeps its keys as they are
                np.testing.assert_array_equal(_stacked(poses, "rotate")[:, 0], rotate[:, 0])
                if name == "the whole clip" or name == "a selection":
                    # b went to the world under a turning a: rebuilt, filtered
                    np.testing.assert_array_equal(clip.frames.rotate[:, 0], rotate[:, 0])
                else:
                    np.testing.assert_array_equal(clip.frames.rotate, rotate)
                world = np.array([[node.world_matrix for node in pose] for pose in poses])
                self.assertLess(np.abs(clip.frames.world_matrix - world).max(), 1e-9)

    def test_node_type(self):
        self._check(
            lambda c: setattr(c, "node_type", "transform"),
            lambda p: [n.set_node_type("transform") for n in p],
        )
        self._check(
            lambda c: c["j2"].set_node_type("locator"),
            lambda p: p["j2"].set_node_type("locator"),
        )

    def test_set_rotate_order(self):
        for target in range(6):
            with self.subTest(target=target):
                self._check(
                    lambda c: c.set_rotate_order(target),
                    lambda p: [n.set_rotate_order(target) for n in p],
                )

    def test_a_joint_folds_only_the_frames_holding_a_joint_orient(self):
        # j1 holds no joint_orient on any frame and spins past 180, beside a
        # joint that holds one: only j2's curve was rebuilt, so only it is
        # filtered. Folding every frame filtered j1's authored curve too,
        # 360 degrees off. An unfolded frame 0 out at 200 the same
        for first in (0.0, 200.0):
            with self.subTest(frame_0=first):
                rig = HierarchyData(
                    [
                        TransformData("j0", node_type="joint"),
                        TransformData("j1", node_type="joint"),
                        TransformData("j2", node_type="joint"),
                    ]
                )
                rig["j1"].parent_uuid = rig["j0"].uuid
                rig["j2"].parent_uuid = rig["j1"].uuid
                clip   = ClipData(rig, frames=10)
                rotate = np.zeros((10, 3, 3))
                rotate[:, 1, 0] = first + 40.0 * np.arange(10)
                rotate[:, 2, 2] = 15.0 * np.arange(10)
                orient = np.zeros((10, 3, 3))
                orient[3:, 2, 1]         = 30.0
                clip.frames.rotate       = rotate
                clip.frames.joint_orient = orient

                poses          = _per_frame(clip, lambda p: [n.set_node_type("transform") for n in p])
                clip.node_type = "transform"
                self._assert_matches(clip, poses)
                np.testing.assert_array_equal(clip.frames["j1"].rotate[:, 0], rotate[:, 1])

    def test_swapaxes_filters_rotate_axis(self):
        # a rotate_axis curve crossing +-180: rebuilt from matrices it steps
        # by 360 unless it is filtered as joint_orient is
        rig  = HierarchyData([TransformData("j0", node_type="joint")])
        clip = ClipData(rig, frames=12)
        axis = np.zeros((12, 1, 3))
        axis[:, 0, 0]           = 30.0 * np.arange(12)
        clip.frames.rotate_axis = axis
        poses = _per_frame(clip, lambda p: p["j0"].swapaxes(0, 1, negate=True))
        clip["j0"].swapaxes(0, 1, negate=True)
        self._assert_matches(clip, poses)
        self.assertLess(np.abs(np.diff(clip.frames.rotate_axis, axis=0)).max(), 31.0)

    def test_a_node_taken_out_of_its_clip_is_edited_as_a_pose(self):
        # a node taken out with del or remove() still points at its clip,
        # and every rig edit on it raised a KeyError naming its id
        edits = {
            "set_rotate_order": lambda n: n.set_rotate_order(5),
            "swapaxes": lambda n: n.swapaxes(0, 1),
            "set_joint_orient_to_rotate": lambda n: n.set_joint_orient_to_rotate(),
            "set_rotate_to_joint_orient": lambda n: n.set_rotate_to_joint_orient(),
            "set_node_type": lambda n: n.set_node_type("transform"),
            "set_parent": lambda n: n.set_parent("j1"),
            "list swapaxes": lambda n: TransformList([n]).swapaxes(0, 1),
            "list set_rotate_order": lambda n: TransformList([n]).set_rotate_order(4),
        }
        for frames in (1, 3):
            for take_out in ("del", "remove"):
                for label, edit in edits.items():
                    with self.subTest(frames=frames, take_out=take_out, edit=label):
                        clip = self._animated(self._rig(2), frames=frames)
                        node = clip["j5"]
                        if take_out == "del":
                            del clip[clip.index("j5")]
                        else:
                            clip.remove(node)
                        self.assertIs(node._hierarchy, clip)
                        blocks = {c: b.copy() for c, b in clip._blocks.items()}
                        pose   = node.copy()

                        edit(node)
                        if label != "set_parent":
                            edit(pose)
                            np.testing.assert_allclose(node.rotate, pose.rotate, atol=1e-9)

                        # the clip's own nodes are left as they were
                        for channel, block in blocks.items():
                            keep = [clip._columns.index(u) for u in clip.uuid]
                            np.testing.assert_array_equal(
                                clip._blocks[channel][:, keep], block[:, keep]
                            )

    def test_a_list_of_nodes_of_two_clips_and_a_taken_out_one(self):
        # grouped by owner: the node a clip no longer lists goes alone, and
        # nothing is half applied when it is met
        first, second = self._animated(self._rig(1)), self._animated(self._rig(2))
        gone = second["j6"]
        second.remove(gone)
        TransformList([first["j1"], gone, second["j2"]]).set_rotate_order(5)
        self.assertEqual(
            (first["j1"].rotate_order, gone.rotate_order, second["j2"].rotate_order), (5, 5, 5)
        )

    def test_a_one_node_edit_reads_the_node_not_the_clip(self):
        # a rig edit of one node read every node of the clip -- its parent
        # map, rotate orders, compensation, columns -- and cleared every
        # node's cache, so a loop over the nodes cost the clip's size
        # squared; now it reads the rows it edits
        class Walked(list):
            walks = 0

            def __iter__(self):
                self.walks += 1
                return super().__iter__()

        clip      = self._animated(self._rig(3, count=60))
        clip.list = Walked(clip.list)
        clip["j5"].set_rotate_order(4)  # the index, once
        edits = {
            "set_rotate_order": lambda n: n.set_rotate_order((n.rotate_order + 1) % 6),
            "swapaxes": lambda n: n.swapaxes(0, 2),
            "set_joint_orient_to_rotate": lambda n: n.set_joint_orient_to_rotate(),
            "set_rotate_to_joint_orient": lambda n: n.set_rotate_to_joint_orient(),
        }
        for label, edit in edits.items():
            with self.subTest(label):
                clip.list.walks = 0
                for name in ("j9", "j31", "j50"):
                    edit(clip[name])
                # clip[name] scans for the name; the edits walk nothing
                self.assertLessEqual(clip.list.walks, 3, label)

    def test_the_frame_axis_points_a_rig_edit_at_the_clip(self):
        # rig edits change what every frame shares, so they are the clip's;
        # on clip.frames they raised a bare AttributeError
        clip = self._animated(self._rig(0))
        for name in ("swapaxes", "set_rotate_to_joint_orient", "set_rotate_order", "set_parent"):
            for axis in (clip.frames, clip.frames["j3"]):
                with self.subTest(name), self.assertRaisesRegex(
                    AttributeError, f"use clip.{name}"
                ):
                    getattr(axis, name)
        with self.assertRaisesRegex(AttributeError, "has no attribute 'nothing'"):
            clip.frames.nothing


class TestClipSmoothSpin(unittest.TestCase):
    """a joint spinning two whole turns: every rebuilt curve stays smooth,
    where the frame by frame route steps by up to 360 degrees each time a
    curve crosses +-180

    The spin is about the first or last axis of the joint's rotate order and
    never y, which rotate_to_joint_orient's xyz would hold in the middle, so
    no result passes near gimbal lock, where any euler curve moves fast. A
    swap above the spinning joint turns its rotate by a constant quarter
    turn, which can; TestClipRigEditsEveryFrame covers those swaps.
    """

    AXIS = {0: 0, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2}

    def _spin(self, order, frames=120):
        rng   = np.random.default_rng(order)
        names = ["root", "a", "b", "c"]
        ids   = [hierarchy.generate_uuid() for _ in names]
        nodes = [
            TransformData(
                name         = name,
                node_type    = "joint",
                uuid         = ids[i],
                parent_uuid  = None if i == 0 else ids[i - 1],
                translate    = [0.0, float(i > 0), 0.0],
                rotate       = rng.uniform(-8.0, 8.0, size=3),
                rotate_axis  = rng.uniform(-8.0, 8.0, size=3),
                joint_orient = rng.uniform(-8.0, 8.0, size=3),
                rotate_order = order,
            )
            for i, name in enumerate(names)
        ]
        clip                           = ClipData(HierarchyData(nodes), frames=frames)
        rotate                         = clip.frames.rotate.copy()
        rotate[:, 3]                   = 0.0
        rotate[:, 3, self.AXIS[order]] = 3.0 + 6.0 * np.arange(frames)
        clip.frames.rotate             = rotate
        return clip

    def test_no_step_over_ten_degrees(self):
        edits = {
            "swapaxes": (
                lambda c: c["c"].swapaxes(0, 1, negate=True),
                lambda p: p["c"].swapaxes(0, 1, negate=True),
            ),
            "swapaxes, a view": (
                lambda c: c.match("c").swapaxes(2, 1),
                lambda p: p["c"].swapaxes(2, 1),
            ),
            "set_rotate_to_joint_orient": (
                lambda c: c.set_rotate_to_joint_orient(),
                lambda p: [n.set_rotate_to_joint_orient() for n in p],
            ),
            "set_joint_orient_to_rotate": (
                lambda c: c.set_joint_orient_to_rotate(),
                lambda p: [n.set_joint_orient_to_rotate() for n in p],
            ),
            "set_parent": (
                lambda c: c["c"].set_parent("root"),
                lambda p: p["c"].set_parent("root"),
            ),
            "set_parent, a view": (
                lambda c: c[1:].set_parent(None),
                lambda p: p[1:].set_parent(None),
            ),
            "node_type": (
                lambda c: setattr(c, "node_type", "transform"),
                lambda p: [n.set_node_type("transform") for n in p],
            ),
            "set_rotate_order": (
                lambda c: c.set_rotate_order((c[0].rotate_order + 3) % 6),
                lambda p: [n.set_rotate_order((n.rotate_order + 3) % 6) for n in p],
            ),
        }
        for order in range(6):
            for name, (clip_edit, pose_edit) in edits.items():
                with self.subTest(order=order, edit=name):
                    clip  = self._spin(order)
                    poses = _per_frame(clip, pose_edit)
                    clip_edit(clip)

                    channels = ("rotate", "joint_orient", "rotate_axis")
                    smooth = max(
                        np.abs(np.diff(getattr(clip.frames, c), axis=0)).max()
                        for c in channels
                    )
                    stepped = max(
                        np.abs(np.diff(_stacked(poses, c), axis=0)).max() for c in channels
                    )
                    self.assertLessEqual(smooth, 10.0)
                    self.assertGreater(stepped, 170.0)

                    world = np.array([[n.world_matrix for n in pose] for pose in poses])
                    self.assertLess(np.abs(clip.frames.world_matrix - world).max(), 1e-9)


class TestClipSetRotateOrder(unittest.TestCase):
    """set_rotate_order over a clip: every frame re-expressed at once, frame 0
    the way Maya's xform -rotateOrder -preserve picks it, the rest filtered,
    no world matrix moved and joint_orient kept"""

    @classmethod
    def setUpClass(cls):
        cls.cases = maya_rotate_order_cases()

    def _clip(self, count=271, frames=300, seed=5):
        # node i takes one of the 30 pairs of orders and plays that pair's 40
        # random and hand picked rotations over and over, from its own
        # offset. The edge rotations stay out of a curve: played in a row,
        # exact gimbal poses and jumps of exactly 180 degrees leave
        # euler_filter ties that round-off settles, and they have tests of
        # their own
        cases = self.cases
        pairs = sorted(set(zip(cases["before"].tolist(), cases["after"].tolist())))
        rng   = np.random.default_rng(seed)
        rows  = np.empty((frames, count), dtype=int)
        uuids = [hierarchy.generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            before, after = pairs[i % len(pairs)]
            pair       = (cases["before"] == before) & (cases["after"] == after)
            ids        = np.flatnonzero(pair[:1200])
            rows[:, i] = ids[(np.arange(frames) + i) % len(ids)]
            nodes.append(
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    uuid         = uuids[i],
                    parent_uuid  = None if i == 0 else uuids[rng.integers(0, i)],
                    translate    = rng.normal(size=3),
                    rotate_order = before,
                )
            )
        clip                     = ClipData(HierarchyData(nodes), frames=frames)
        clip.frames.rotate       = cases["rotate"][rows]
        clip.frames.joint_orient = cases["joint_orient"][rows]
        after                    = np.array([pairs[i % len(pairs)][1] for i in range(count)])
        return clip, rows, after

    def test_a_271_joint_300_frame_clip(self):
        clip, rows, after = self._clip()
        world  = clip.frames.world_matrix.copy()
        orient = clip.frames.joint_orient.copy()
        maya   = self.cases["maya"][rows]

        clip.set_rotate_order(after)

        self.assertTrue(np.array_equal(clip.rotate_order, after))
        self.assertLess(np.abs(clip.frames.rotate[0] - maya[0]).max(),             1e-9)
        self.assertLess(np.abs(clip.frames.rotate - _filtered(maya, after)).max(), 1e-9)
        self.assertLess(np.abs(clip.frames.world_matrix - world).max(),            1e-12)
        self.assertTrue(np.array_equal(clip.frames.joint_orient, orient))

    def test_a_view_of_a_clip(self):
        clip, rows, after = self._clip(count=60, frames=50)
        world  = clip.frames.world_matrix.copy()
        rotate = clip.frames.rotate.copy()
        maya   = self.cases["maya"][rows]
        picked = np.arange(0, 60, 3)

        names  = ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]
        clip[picked].set_rotate_order([names[o] for o in after[picked]])

        rest = np.setdiff1d(np.arange(60), picked)
        self.assertTrue(np.array_equal(clip.frames.rotate[:, rest], rotate[:, rest]))
        want = _filtered(maya[:, picked], after[picked])
        self.assertLess(np.abs(clip.frames.rotate[:, picked] - want).max(), 1e-9)
        self.assertLess(np.abs(clip.frames.world_matrix - world).max(), 1e-12)

    def test_frame_0_takes_mayas_pick_on_every_recorded_joint(self):
        # a clip of all 2550 recorded joints side by side, edge rotations
        # included. The second frame holds the next joint's angles, read in
        # this joint's order: some other pose, which only has to stay put
        cases  = self.cases
        joints = np.arange(len(cases["rotate"]))
        rows   = np.stack([joints, np.roll(joints, -1)])
        nodes = [
            TransformData(f"j{i}", node_type="joint", rotate_order=int(order))
            for i, order in enumerate(cases["before"])
        ]
        clip = ClipData(HierarchyData(nodes), frames=2)
        clip.frames.rotate       = cases["rotate"][rows]
        clip.frames.joint_orient = cases["joint_orient"][rows]
        world  = clip.frames.world_matrix.copy()
        orient = clip.frames.joint_orient.copy()

        clip.set_rotate_order(cases["after"])

        self.assertTrue(np.array_equal(clip.rotate_order, cases["after"]))
        self.assertLess(np.abs(clip.frames.rotate[0] - cases["maya"]).max(), 1e-9)
        self.assertTrue(np.array_equal(clip.frames.joint_orient, orient))

        # the 8 joints that start and end 1e-7 from gimbal lose up to 3.1e-9
        # of the pose in Maya too, as do 6 of the second frame's poses that
        # land there; every other one keeps it to round-off
        moved = np.abs(clip.frames.world_matrix - world).max(axis=(2, 3))
        twice = cases["twice"]
        self.assertLess(moved[0, ~twice].max(), 1e-12)
        self.assertLess(moved[0, twice].max(),  1e-8)
        self.assertLess(moved[1].max(),         1e-8)
        self.assertLessEqual(int((moved[1] > 1e-12).sum()), 6)


class TestClipEvaluator(unittest.TestCase):
    """the batched evaluator must agree with scrubbing, exactly"""

    def _hard_rig(self, count=12):
        # every feature that makes the composition non trivial: varied
        # rotate_order, mixed segment_scale_compensate, non zero joint_orient
        # and rotate_axis, non unit scale, deep and shallow chains
        rng   = np.random.default_rng(7)
        ids   = [hierarchy.generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            parent = None
            if i:
                parent = ids[rng.integers(0, i)]
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    uuid=ids[i],
                    parent_uuid=parent,
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-180.0, 180.0, size=3),
                    scale=rng.uniform(0.5, 2.0, size=3),
                    rotate_axis=rng.uniform(-40.0, 40.0, size=3),
                    joint_orient=rng.uniform(-90.0, 90.0, size=3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
            )
        return HierarchyData(nodes)

    def _animated(self, frames=6, count=12):
        rng  = np.random.default_rng(11)
        clip = ClipData(self._hard_rig(count), frames=frames)
        for channel in ("rotate", "translate"):
            block = getattr(clip.frames, channel).copy()
            block += rng.normal(size=block.shape) * 10.0
            setattr(clip.frames, channel, block)
        scale = clip.frames.scale.copy()
        scale *= rng.uniform(0.8, 1.25, size=scale.shape)
        clip.frames.scale = scale
        return clip

    def test_batched_local_matches_scrubbing(self):
        clip    = self._animated()
        batched = clip.frames.matrix

        for frame in range(clip.frame_count):
            clip.frame = frame
            self.assertTrue(np.array_equal(batched[frame], clip.matrix))

    def test_batched_world_matches_scrubbing(self):
        clip    = self._animated()
        batched = clip.frames.world_matrix

        for frame in range(clip.frame_count):
            clip.frame = frame
            self.assertTrue(np.array_equal(batched[frame], clip.world_matrix))

    def test_batched_shapes(self):
        clip = self._animated(frames=5, count=9)
        self.assertEqual(clip.frames.matrix.shape, (5, 9, 4, 4))
        self.assertEqual(clip.frames.world_matrix.shape, (5, 9, 4, 4))

    def test_batched_world_respects_segment_scale_compensate(self):
        # a scaled parent reaches its child only when the flag is off, so the
        # two must not agree
        rig = HierarchyData(
            [
                TransformData(name="a", node_type="joint", scale=[2.0, 3.0, 4.0]),
                TransformData(
                    name="b",
                    node_type="joint",
                    translate=[1.0, 1.0, 1.0],
                    segment_scale_compensate=True,
                ),
            ]
        )
        rig["b"].parent_uuid = rig["a"].uuid
        on                   = ClipData(rig, frames=2).frames.world_matrix

        rig["b"].segment_scale_compensate = False
        off                               = ClipData(rig, frames=2).frames.world_matrix

        self.assertFalse(np.allclose(on[:, 1], off[:, 1]))

    def test_batched_follows_a_block_edit(self):
        clip   = self._animated(frames=4, count=6)
        before = clip.frames.world_matrix.copy()

        translate = clip.frames.translate.copy()
        translate[2] += 5.0
        clip.frames.translate = translate

        after = clip.frames.world_matrix
        self.assertFalse(np.allclose(before[2], after[2]))
        self.assertTrue(np.allclose(before[0], after[0]))


class TestClipDeltas(unittest.TestCase):
    """deltas have to cover the clip, not just the loaded frame"""

    def _rig(self, count=5, seed=5):
        rng = np.random.default_rng(seed)
        ids = [hierarchy.generate_uuid() for _ in range(count)]
        return HierarchyData(
            [
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    uuid         = ids[i],
                    parent_uuid  = None if i == 0 else ids[i - 1],
                    translate    = rng.normal(size=3),
                    rotate       = rng.uniform(-90.0, 90.0, size=3),
                    joint_orient = rng.uniform(-30.0, 30.0, size=3),
                )
                for i in range(count)
            ]
        )

    def _animated(self, rig, frames=4, seed=9):
        rng    = np.random.default_rng(seed)
        clip   = ClipData(rig, frames=frames)
        rotate = clip.frames.rotate.copy()
        rotate += rng.normal(size=rotate.shape) * 15.0
        clip.frames.rotate = rotate
        return clip

    def test_delta_against_a_pose_round_trips(self):
        rig   = self._rig()
        clip  = self._animated(rig)

        delta = clip.get_delta(rig)
        self.assertIsInstance(delta, ClipData)
        self.assertEqual(delta.frame_count, clip.frame_count)

        recovered = ClipData(rig, frames=clip.frame_count).add_delta(delta)
        self.assertIsInstance(recovered, ClipData)

        for frame in range(clip.frame_count):
            clip.frame      = frame
            recovered.frame = frame
            self.assertTrue(
                np.allclose(recovered.world_matrix, clip.world_matrix, atol=1e-6)
            )

    def test_a_single_delta_poses_every_frame(self):
        # the retarget case: one fixed offset applied across a whole clip
        source = self._rig(seed=5)
        target = self._rig(seed=17)
        clip   = self._animated(source)

        delta  = target.get_delta(source)
        posed  = clip.add_delta(delta)

        self.assertIsInstance(posed, ClipData)
        self.assertEqual(posed.frame_count, clip.frame_count)
        # the animation still varies frame to frame after retargeting
        self.assertFalse(np.allclose(posed.frames.rotate[0], posed.frames.rotate[-1]))

    def test_delta_between_two_clips_pairs_frame_for_frame(self):
        rig   = self._rig()
        a     = self._animated(rig, seed=9)
        b     = self._animated(rig, seed=21)

        delta = a.get_delta(b)
        self.assertEqual(delta.frame_count, a.frame_count)

        recovered = b.add_delta(delta)
        for frame in range(a.frame_count):
            a.frame         = frame
            recovered.frame = frame
            self.assertTrue(
                np.allclose(recovered.world_matrix, a.world_matrix, atol=1e-6)
            )

    def test_delta_rejects_a_clip_of_a_different_length(self):
        rig = self._rig()
        a   = self._animated(rig, frames=4)
        b   = self._animated(rig, frames=3)
        with self.assertRaises(ValueError):
            a.get_delta(b)

    def test_operator_matches_the_method(self):
        rig  = self._rig()
        clip = self._animated(rig)

        by_operator = clip - rig
        by_method   = clip.get_delta(rig)

        self.assertIsInstance(by_operator, ClipData)
        self.assertTrue(np.allclose(by_operator.frames.rotate, by_method.frames.rotate))

    def test_timebase_survives_the_delta(self):
        rig   = self._rig()
        clip  = ClipData(rig, frames=3, start_frame=1001, fps=30.0)
        delta = clip.get_delta(rig)

        self.assertEqual(delta.start_frame, 1001)
        self.assertEqual(delta.fps, 30.0)

    def test_translate_flag_is_forwarded(self):
        rig  = self._rig()
        clip = self._animated(rig)

        with_translate = clip.get_delta(rig, translate=True)
        without        = clip.get_delta(rig, translate=False)

        self.assertEqual(without.frame_count, clip.frame_count)
        self.assertFalse(
            np.allclose(with_translate.frames.translate, without.frames.translate)
        )


CHANNELS = ("rotate", "translate", "scale", "joint_orient", "rotate_axis")


def pose_at(clip, frame):
    """one frame of a clip as a plain single pose HierarchyData of its own"""
    clip.frame = frame
    return TransformList._owned_copy(clip)


class _DeltaRigs:
    """random rigs and clips for the clip pose algebra tests, and the frame
    by frame reference they are held to"""

    def _rig(self, names, parents, seed, scale=False):
        """``names``, each node under the first node of the name ``parents``
        gives, by its uuid"""
        rng = np.random.default_rng(seed)
        ids, ups = _parent_ids(names, parents)
        return HierarchyData(
            [
                TransformData(
                    name         = name,
                    node_type    = "joint",
                    uuid         = uuid,
                    parent_uuid  = up,
                    translate    = rng.normal(size=3),
                    rotate       = rng.uniform(-90.0, 90.0, size=3),
                    joint_orient = rng.uniform(-30.0, 30.0, size=3),
                    rotate_axis  = rng.uniform(-20.0, 20.0, size=3),
                    rotate_order = int(rng.integers(0, 6)),
                    # mixed: compensation is the one solve that reads the
                    # parent's solved scale
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                    scale=rng.uniform(0.8, 1.25, size=3) if scale else None,
                )
                for name, uuid, up in zip(names, ids, ups)
            ]
        )

    def _chain(self, count, seed, **kwargs):
        names = [f"j{i}" for i in range(count)]
        return self._rig(names, [None] + names[:-1], seed, **kwargs)

    def _animated(self, rig, frames=6, seed=9, scale=True, orient=False):
        rng                   = np.random.default_rng(seed)
        clip                  = ClipData(rig, frames=frames, start_frame=1001, fps=30.0)
        shape                 = clip.frames.rotate.shape
        clip.frames.rotate    = clip.frames.rotate + rng.normal(size=shape) * 20.0
        clip.frames.translate = clip.frames.translate + rng.normal(size=shape) * 0.3
        if scale:
            clip.frames.scale = 1.0 + rng.uniform(-0.3, 0.3, size=shape)
        if orient:
            # framed too: each frame must solve through its own RO * R * JO
            clip.frames.joint_orient = clip.frames.joint_orient + rng.normal(size=shape) * 10.0
            clip.frames.rotate_axis  = clip.frames.rotate_axis + rng.normal(size=shape) * 10.0
        return clip

    def _per_frame(self, method, receiver, other, by="index", **kwargs):
        """the reference: ``receiver.<method>(other)`` on every frame, both
        sides plain single poses -- a clip read one frame at a time, frame for
        frame against a clip -- gathered by from_poses on the clip's timebase.
        ``method`` is get_delta / add_delta or an in place operator, __isub__
        / __iadd__.

        Each frame picks its euler branch on its own, so the rotate curves of
        the paired nodes then go through euler_filter with each node's rotate
        order, frame 0 kept as solved bit for bit, as the clip algebra's must.
        """
        clip  = receiver if isinstance(receiver, ClipData) else other
        poses = []
        for frame in range(clip.frame_count):
            mine   = pose_at(receiver, frame) if isinstance(receiver, ClipData) else receiver
            theirs = pose_at(other, frame) if isinstance(other, ClipData) else other
            poses.append(getattr(mine, method)(theirs, by=by, **kwargs))
        result = ClipData.from_poses(poses, start_frame=clip.start_frame, fps=clip.fps)

        # by name, only the nodes both sides hold are solved
        paired               = set(receiver.name) & set(other.name) if by == "name" else None
        cols                 = [i for i, n in enumerate(result.name) if paired is None or n in paired]
        orders               = np.array([result[i].rotate_order for i in cols], dtype=int)
        rotate               = np.array(result.frames.rotate)
        rotate[:, cols]      = _filtered(rotate[:, cols], orders)
        result.frames.rotate = rotate
        return result

    def _frozen(self, clip, frame):
        """the whole clip at one frame, as a pose of its own"""
        clip.frame = frame
        return HierarchyData([node.copy() for node in clip])

    def _assert_same(self, fast, slow):
        # exact: the same algebra, step for step, and the same filter give
        # the same bits
        self.assertIsInstance(fast, ClipData)
        self.assertEqual(fast.name, slow.name)
        self.assertEqual(fast.uuid, slow.uuid)
        self.assertEqual([n.parent_uuid for n in fast], [n.parent_uuid for n in slow])
        self.assertEqual(fast.frame_count, slow.frame_count)
        self.assertEqual(fast.start_frame, slow.start_frame)
        self.assertEqual(fast.fps, slow.fps)
        for channel in ("rotate", "translate", "scale", "joint_orient", "rotate_axis"):
            np.testing.assert_array_equal(
                getattr(fast.frames, channel), getattr(slow.frames, channel), channel
            )
        np.testing.assert_array_equal(fast.frames.world_matrix, slow.frames.world_matrix)


class TestClipDeltaVectorised(_DeltaRigs, unittest.TestCase):
    """clip pose algebra solves every frame at once; it must agree with the
    single pose HierarchyData.get_delta / add_delta run on each frame, its
    rotate curves euler filtered"""

    # --- add_delta with one delta --- #
    def test_index_pairing_matches_per_frame(self):
        source = self._chain(6, seed=5)
        target = self._chain(6, seed=17)
        clip   = self._animated(source)
        delta  = target.get_delta(source)

        self._assert_same(clip.add_delta(delta), self._per_frame("add_delta", clip, delta))

    def test_orientation_only_matches_per_frame(self):
        source = self._chain(6, seed=5)
        target = self._chain(6, seed=17)
        clip   = self._animated(source)
        delta  = target.get_delta(source, translate=False)

        self._assert_same(
            clip.add_delta(delta, translate=False),
            self._per_frame("add_delta", clip, delta, translate=False),
        )

    def test_name_pairing_over_part_of_the_rig_matches_per_frame(self):
        # the delta covers joints the clip never drives (d_extra, j2) and the
        # clip drives one the delta lacks (c_extra)
        names   = ["j0", "j1", "j2", "j3", "d_extra"]
        parents = [None, "j0", "j1", "j2", "j1"]
        target  = self._rig(names, parents, seed=17)
        bind    = self._rig(names, parents, seed=5)
        delta   = target.get_delta(bind, by="name")

        animated = self._rig(
            ["j0", "j1", "j3", "c_extra"], [None, "j0", "j1", "j3"], seed=5
        )
        clip = self._animated(animated)

        self._assert_same(
            clip.add_delta(delta, by="name"),
            self._per_frame("add_delta", clip, delta, by="name"),
        )

    def test_a_scaled_delta_matches_per_frame(self):
        # a delta scale other than 1: under compensation each joint's scale
        # delta is rebuilt through its parent's
        source       = self._chain(6, seed=5)
        target       = self._chain(6, seed=17)
        target.scale = np.linspace(0.8, 1.4, 18).reshape(6, 3)
        clip         = self._animated(source)
        delta        = target.get_delta(source)
        self.assertFalse(np.allclose(delta.scale, 1.0))

        self._assert_same(clip.add_delta(delta), self._per_frame("add_delta", clip, delta))

    def test_a_clip_of_deltas_still_pairs_frame_for_frame(self):
        # no scale animation: a non-uniform parent scale without segment
        # scale compensation shears the child, which a delta (a rotation per
        # node) cannot carry, so that round trip was never exact
        rig = self._chain(5, seed=5)
        a   = self._animated(rig, seed=9, scale=False)
        b   = self._animated(rig, seed=21, scale=False)

        recovered = b.add_delta(a.get_delta(b))
        self.assertTrue(
            np.allclose(recovered.frames.world_matrix, a.frames.world_matrix, atol=1e-6)
        )

    def test_source_clip_is_left_alone(self):
        source     = self._chain(5, seed=5)
        clip       = self._animated(source)
        clip.frame = 3
        before     = {c: np.array(getattr(clip.frames, c)) for c in ("rotate", "translate", "scale")}

        clip.add_delta(self._chain(5, seed=17).get_delta(source))

        self.assertEqual(clip.frame, 3)
        for channel, values in before.items():
            self.assertTrue(np.array_equal(getattr(clip.frames, channel), values))

    def test_a_view_of_the_clip_is_one_delta_at_its_bound_frame(self):
        # a selection is one pose, even one of the clip that receives it: it
        # is read at the frame the clip is on, not paired frame for frame
        clip   = self._animated(self._chain(5, seed=5))
        frozen = self._frozen(clip, 1)

        self._assert_same(
            clip.add_delta(clip.match("j1", "j2", "j3"), by="name"),
            clip.add_delta(frozen.match("j1", "j2", "j3"), by="name"),
        )

    def test_a_clip_with_no_frames_is_one_delta(self):
        source = self._chain(5, seed=5)
        clip   = self._animated(source)
        delta  = self._chain(5, seed=17).get_delta(source)

        self._assert_same(clip.add_delta(ClipData(delta)), clip.add_delta(delta))

    def test_no_frame_is_solved_on_its_own(self):
        # every frame at once: none of these may run the single pose solve,
        # which is what a loop over the frames calls
        source = self._chain(5, seed=5)
        target = self._chain(5, seed=17)
        clip   = self._animated(source, frames=12)
        other  = self._animated(target, frames=12, seed=21)
        delta  = target.get_delta(source)
        deltas = other.get_delta(clip)

        isub = mock.patch.object(
            TransformList, "__isub__", autospec=True, side_effect=TransformList.__isub__
        )
        iadd = mock.patch.object(
            TransformList, "__iadd__", autospec=True, side_effect=TransformList.__iadd__
        )
        with isub as sub, iadd as add:
            clip.get_delta(target)
            clip.get_delta(other, by="name")
            clip.add_delta(delta)
            clip.add_delta(deltas, by="name")

        self.assertEqual((sub.call_count, add.call_count), (0, 0))

    # --- get_delta --- #
    def test_get_delta_against_a_pose_matches_per_frame(self):
        target = self._chain(6, seed=17)
        clip   = self._animated(self._chain(6, seed=5), orient=True)

        self._assert_same(
            clip.get_delta(target), self._per_frame("get_delta", clip, target)
        )

    def test_get_delta_against_a_clip_matches_per_frame(self):
        clip  = self._animated(self._chain(6, seed=5), seed=9, orient=True)
        other = self._animated(self._chain(6, seed=17), seed=21, orient=True)

        self._assert_same(
            clip.get_delta(other), self._per_frame("get_delta", clip, other)
        )

    def test_get_delta_orientation_only_matches_per_frame(self):
        clip  = self._animated(self._chain(6, seed=5), seed=9)
        other = self._animated(self._chain(6, seed=17), seed=21)

        self._assert_same(
            clip.get_delta(other, translate=False),
            self._per_frame("get_delta", clip, other, translate=False),
        )

    def test_get_delta_name_pairing_over_part_of_the_rig_matches_per_frame(self):
        names   = ["j0", "j1", "j2", "j3", "c_extra"]
        parents = [None, "j0", "j1", "j2", "j1"]
        clip    = self._animated(self._rig(names, parents, seed=5))
        pose    = self._rig(["j0", "j1", "j3", "p_extra"], [None, "j0", "j1", "j3"], seed=17)

        self._assert_same(
            clip.get_delta(pose, by="name"),
            self._per_frame("get_delta", clip, pose, by="name"),
        )

    def test_get_delta_on_a_clip_missing_a_parent_makes_its_child_a_root(self):
        # a clip of part of a rig: j1's parent is not in it. Its nodes join
        # the clip, which makes such a node a root, and the delta keeps it one
        clip = self._animated(ClipData(self._chain(5, seed=5)[1:]))
        pose = self._chain(5, seed=17)[1:]
        self.assertIsNone(clip["j1"].parent_uuid)

        delta = clip.get_delta(pose)
        self.assertIsNone(delta["j1"].parent_uuid)
        self._assert_same(delta, self._per_frame("get_delta", clip, pose))

    def test_get_delta_against_a_view_of_the_clip_reads_its_bound_frame(self):
        clip   = self._animated(self._chain(5, seed=5), orient=True)
        frozen = self._frozen(clip, 1)

        self._assert_same(
            clip.get_delta(clip.match("j1", "j2", "j3"), by="name"),
            clip.get_delta(frozen.match("j1", "j2", "j3"), by="name"),
        )

    def test_get_delta_against_a_clip_with_no_frames_is_one_pose(self):
        target = self._chain(5, seed=17)
        clip   = self._animated(self._chain(5, seed=5))

        self._assert_same(clip.get_delta(ClipData(target)), clip.get_delta(target))

    def test_get_delta_rejects_a_clip_of_a_different_length(self):
        clip  = self._animated(self._chain(4, seed=5), frames=6)
        other = self._animated(self._chain(4, seed=17), frames=5)
        with self.assertRaisesRegex(ValueError, "clips hold 6 and 5 frames"):
            clip.get_delta(other)

    def test_the_other_clip_of_get_delta_is_left_alone(self):
        other       = self._animated(self._chain(5, seed=17), seed=21)
        other.frame = 2
        before      = {c: np.array(getattr(other.frames, c)) for c in ("rotate", "translate", "scale")}

        self._animated(self._chain(5, seed=5)).get_delta(other)

        self.assertEqual(other.frame, 2)
        for channel, values in before.items():
            self.assertTrue(np.array_equal(getattr(other.frames, channel), values))

    # --- add_delta with a clip of deltas --- #
    def test_a_clip_of_deltas_matches_per_frame(self):
        deltas = self._animated(self._chain(6, seed=17), seed=21, orient=True)
        clip   = self._animated(self._chain(6, seed=5), seed=9)

        self._assert_same(
            clip.add_delta(deltas), self._per_frame("add_delta", clip, deltas)
        )

    def test_a_clip_of_deltas_by_name_matches_per_frame(self):
        names   = ["j0", "j1", "j2", "j3", "d_extra"]
        parents = [None, "j0", "j1", "j2", "j1"]
        deltas  = self._animated(self._rig(names, parents, seed=17), seed=21, orient=True)
        clip = self._animated(
            self._rig(["j0", "j1", "j3", "c_extra"], [None, "j0", "j1", "j3"], seed=5)
        )

        self._assert_same(
            clip.add_delta(deltas, by="name"),
            self._per_frame("add_delta", clip, deltas, by="name"),
        )

    def test_a_clip_of_deltas_orientation_only_matches_per_frame(self):
        deltas = self._animated(self._chain(5, seed=17), seed=21)
        clip   = self._animated(self._chain(5, seed=5), seed=9)

        self._assert_same(
            clip.add_delta(deltas, translate=False),
            self._per_frame("add_delta", clip, deltas, translate=False),
        )

    def test_a_clip_of_deltas_of_a_different_length_is_rejected(self):
        deltas = self._animated(self._chain(4, seed=17), frames=5)
        clip   = self._animated(self._chain(4, seed=5), frames=6)
        with self.assertRaisesRegex(ValueError, "clips hold 6 and 5 frames"):
            clip.add_delta(deltas)

    def test_the_delta_clip_is_left_alone(self):
        deltas       = self._animated(self._chain(5, seed=17), seed=21)
        deltas.frame = 2
        before       = {c: np.array(getattr(deltas.frames, c)) for c in ("rotate", "translate", "scale")}

        self._animated(self._chain(5, seed=5)).add_delta(deltas)

        self.assertEqual(deltas.frame, 2)
        for channel, values in before.items():
            self.assertTrue(np.array_equal(getattr(deltas.frames, channel), values))


class TestDeltaSolveOrder(unittest.TestCase):
    """a paired node under an unpaired one must still solve after its paired
    ancestors, or the ancestor moves afterwards and drags it off target"""

    children_first = False  # a subclass lists every rig children first

    def _rig(self, names, parents, seed):
        rng = np.random.default_rng(seed)
        ids, ups = _parent_ids(names, parents)
        nodes = [
            TransformData(
                name         = name,
                node_type    = "joint",
                uuid         = uuid,
                parent_uuid  = up,
                translate    = rng.normal(size=3),
                rotate       = rng.uniform(-90.0, 90.0, size=3),
                joint_orient = rng.uniform(-30.0, 30.0, size=3),
            )
            for name, uuid, up in zip(names, ids, ups)
        ]
        return HierarchyData(nodes[::-1] if self.children_first else nodes)

    def setUp(self):
        super().setUp()
        # under a name pairing `c` is a subset root (its parent `b` is not in
        # the pose), so the pairing alone would rank it beside `root`, ahead
        # of `a` -- which it hangs under
        names       = ["root", "a", "b", "c"]
        parents     = [None, "root", "a", "b"]
        self.target = self._rig(names, parents, seed=17)
        self.bind   = self._rig(names, parents, seed=5)
        self.delta  = self.target.get_delta(self.bind, by="name")
        self.pose   = self._rig(["root", "a", "c"], [None, "root", "a"], seed=23)

        # by index the pairing is the whole rig, so its parents are the real
        # ones: only the list order can still put a child ahead of its parent
        self.full_delta = self.target.get_delta(self.bind)
        self.full_pose  = self._rig(names, parents, seed=29)

    def _expected(self, pose, delta):
        """the world matrix each paired node of the result must land on"""
        expected = {}
        for node in pose:
            offset              = delta[node.name]
            M                   = quaternion_to_matrix(offset.quaternion)[0]
            W                   = node.world_matrix
            target              = np.dot(M, _rotation(W))
            target[3, :3]       = W[3, :3] + offset.translate
            expected[node.name] = target
        return expected

    def _assert_lands(self, pose, delta, by):
        # read through a copy, whose world matrices are rebuilt from the
        # solved channels: a cached matrix can hide a wrong solve
        posed = pose.add_delta(delta, by=by).copy()
        for name, target in self._expected(pose, delta).items():
            np.testing.assert_allclose(
                posed[name].world_matrix, target, rtol=0, atol=1e-9, err_msg=name
            )

    def _assert_clip_lands(self, pose, delta, by):
        clip               = ClipData(pose, frames=3)
        rng                = np.random.default_rng(3)
        clip.frames.rotate = clip.frames.rotate + rng.normal(size=(3, len(pose), 3)) * 20.0

        posed = clip.add_delta(delta, by=by)
        world = posed.frames.world_matrix
        for frame in range(clip.frame_count):
            clip.frame = frame
            for name, target in self._expected(clip, delta).items():
                np.testing.assert_allclose(
                    world[frame, posed.index(name)], target, rtol=0, atol=1e-9,
                    err_msg=f"{name} frame {frame}",
                )

    def test_single_pose_lands_every_paired_node_on_target(self):
        self._assert_lands(self.pose, self.delta, by="name")

    def test_clip_lands_every_paired_node_on_target(self):
        self._assert_clip_lands(self.pose, self.delta, by="name")

    def test_single_pose_by_index_lands_every_node_on_target(self):
        self._assert_lands(self.full_pose, self.full_delta, by="index")

    def test_clip_by_index_lands_every_node_on_target(self):
        self._assert_clip_lands(self.full_pose, self.full_delta, by="index")


class TestDeltaSolveOrderChildrenFirst(TestDeltaSolveOrder):
    """the same with every rig listing children ahead of their parents: the
    solve order comes from the hierarchy, never from the list"""

    children_first = True


class TestClipAlgebraEveryFrame(_DeltaRigs, unittest.TestCase):
    """clip.frames stands for every frame of a clip, where the clip itself is
    the frame it is bound to: ``clip.frames -= pose``, ``pose.get_delta(
    clip.frames)`` and ``rig.add_delta(deltas.frames)`` solve every frame at
    once and equal the single pose algebra on each frame; ``clip -= pose``
    and ``pose.get_delta(clip)`` are the bound frame alone"""

    # a partial overlap by name: each side holds a node the other lacks
    PARTIAL = (["j0", "j1", "j2", "j3", "c_extra"], [None, "j0", "j1", "j2", "j1"])
    POSE    = (["j0", "j1", "j3", "p_extra"], [None, "j0", "j1", "j3"])

    def _scaled(self, count, seed):
        return self._chain(count, seed, scale=True)

    # --- clip.frames -= ... --- #
    def test_subtracting_a_pose_from_every_frame_matches_per_frame(self):
        clip       = self._animated(self._scaled(6, seed=5), orient=True)
        pose       = self._scaled(6, seed=17)
        expected   = self._per_frame("__isub__", clip, pose)

        clip.frame = 2
        frames     = clip.frames
        frames -= pose

        self.assertIs(frames._clip, clip)
        self.assertEqual(clip.frame, 2)
        self._assert_same(clip, expected)
        # the bound nodes read the solved frame
        expected.frame = 2
        np.testing.assert_array_equal(clip["j3"].rotate, expected["j3"].rotate)

    def test_subtracting_a_clip_from_every_frame_matches_per_frame(self):
        clip     = self._animated(self._scaled(6, seed=5), seed=9, orient=True)
        other    = self._animated(self._scaled(6, seed=17), seed=21, orient=True)
        expected = self._per_frame("__isub__", clip, other)
        twin     = clip.copy()

        clip.frames -= other
        self._assert_same(clip, expected)

        # its frame axis pairs the same way
        twin.frames -= other.frames
        self._assert_same(twin, expected)

    def test_subtracting_by_name_orientation_only_matches_per_frame(self):
        clip     = self._animated(self._rig(*self.PARTIAL, seed=5))
        pose     = self._rig(*self.POSE, seed=17)
        expected = self._per_frame("__isub__", clip, pose, by="name", translate=False)

        clip.frames.__isub__(pose, by="name", translate=False)
        self._assert_same(clip, expected)

    # --- clip.frames += ... --- #
    def test_adding_a_pose_to_every_frame_matches_per_frame(self):
        deltas   = self._animated(self._scaled(6, seed=17), seed=21, orient=True)
        pose     = self._scaled(6, seed=5)
        expected = self._per_frame("__iadd__", deltas, pose)

        deltas.frame = 4
        frames       = deltas.frames
        frames += pose

        self.assertIs(frames._clip, deltas)
        self.assertEqual(deltas.frame, 4)
        self._assert_same(deltas, expected)

    def test_adding_a_clip_to_every_frame_matches_per_frame(self):
        deltas   = self._animated(self._scaled(6, seed=17), seed=21, orient=True)
        source   = self._animated(self._scaled(6, seed=5), seed=9)
        expected = self._per_frame("__iadd__", deltas, source)

        deltas.frames += source
        self._assert_same(deltas, expected)

    def test_adding_by_name_orientation_only_matches_per_frame(self):
        deltas   = self._animated(self._rig(*self.PARTIAL, seed=17), seed=21)
        pose     = self._rig(*self.POSE, seed=5)
        expected = self._per_frame("__iadd__", deltas, pose, by="name", translate=False)

        deltas.frames.__iadd__(pose, by="name", translate=False)
        self._assert_same(deltas, expected)

    def test_every_frame_refuses_a_clip_of_another_length(self):
        clip   = self._animated(self._chain(4, seed=5), frames=6)
        other  = self._animated(self._chain(4, seed=17), frames=5)
        before = {c: np.array(getattr(clip.frames, c)) for c in CHANNELS}

        with self.assertRaisesRegex(ValueError, "clips hold 6 and 5 frames"):
            clip.frames -= other
        with self.assertRaisesRegex(ValueError, "clips hold 6 and 5 frames"):
            clip.frames += other

        for channel, values in before.items():
            np.testing.assert_array_equal(getattr(clip.frames, channel), values)

    def test_every_frame_needs_a_whole_axis_of_a_clip_with_frames(self):
        clip = self._animated(self._chain(4, seed=5))
        pose = self._chain(4, seed=17)
        with self.assertRaises(TypeError):
            clip.frames["j1"].__isub__(pose)
        with self.assertRaises(TypeError):
            pose.get_delta(clip.frames[["j1", "j2"]])
        with self.assertRaisesRegex(ValueError, "holds no frames"):
            ClipData(pose).frames.__isub__(pose)
        with self.assertRaises(AttributeError):
            clip.frames = self._animated(self._chain(4, seed=5)).frames

    def test_a_clip_with_no_frames_is_one_pose_in_place(self):
        rig  = self._chain(4, seed=5)
        pose = self._chain(4, seed=17)

        clip = ClipData(rig)
        clip -= pose
        self.assertEqual(clip.frame_count, 0)
        np.testing.assert_array_equal(clip.rotate, rig.get_delta(pose).rotate)

    # --- the clip itself: the frame it is bound to --- #
    def test_in_place_on_the_clip_is_the_bound_frame_alone(self):
        # clip.<op> acts on the frame the clip is bound to, as every other
        # write on the clip does: a view of it, clip.frames[i] (the clip
        # scrubbed to i) and a scrub loop all agree
        pose = self._scaled(6, seed=17)
        forms = {
            "clip -= pose":           lambda c: c.__isub__(pose),
            "clip += pose":           lambda c: c.__iadd__(pose),
            "clip[:] -= pose":        lambda c: c[:].__isub__(pose),
            "clip.frames[2] -= pose": lambda c: c.frames[2].__isub__(pose),
        }
        for label, write in forms.items():
            with self.subTest(label):
                clip       = self._animated(self._scaled(6, seed=5), orient=True)
                clip.frame = 2
                single     = pose_at(clip, 2)
                (single.__iadd__ if "+=" in label else single.__isub__)(pose)
                clip.frame = 2
                before     = {c: np.array(getattr(clip.frames, c)) for c in CHANNELS}

                write(clip)

                self.assertEqual(clip.frame, 2)
                others = [f for f in range(clip.frame_count) if f != 2]
                for channel, values in before.items():
                    got = np.asarray(getattr(clip.frames, channel))
                    np.testing.assert_array_equal(got[others], values[others], channel)
                    np.testing.assert_allclose(
                        got[2], getattr(single, channel), rtol=0.0, atol=1e-12,
                        err_msg=channel,
                    )

    def test_a_frame_range_written_in_place_is_its_first_frame_alone(self):
        # clip.frames[a:b] is an independent cut, bound to its own first
        # frame, and the cut is the clip it stands for: the statement
        # clip.frames[a:b] -= pose solves that one frame, which the paste
        # then writes back, frame a alone (hierarchy/CHEATSHEET.md, Gotchas).
        # The whole range is the cut's frame axis, pasted back
        pose = self._scaled(6, seed=17)

        def isub_range(clip):
            clip.frames[1:4] -= pose

        def iadd_range(clip):
            clip.frames[1:4] += pose

        def isub_all(clip):
            clip.frames[:] -= pose

        for label, statement, first in (
            ("clip.frames[1:4] -= pose", isub_range, 1),
            ("clip.frames[1:4] += pose", iadd_range, 1),
            ("clip.frames[:] -= pose", isub_all, 0),
        ):
            with self.subTest(label):
                clip   = self._animated(self._scaled(6, seed=5), orient=True)
                single = pose_at(clip, first)
                (single.__iadd__ if "+=" in label else single.__isub__)(pose)
                clip.frame = 2
                before     = {c: np.array(getattr(clip.frames, c)) for c in CHANNELS}

                statement(clip)

                self.assertEqual(clip.frame, 2)
                others = [f for f in range(clip.frame_count) if f != first]
                for channel, values in before.items():
                    got = np.asarray(getattr(clip.frames, channel))
                    np.testing.assert_array_equal(got[others], values[others], channel)
                    np.testing.assert_allclose(
                        got[first], getattr(single, channel), rtol=0.0, atol=1e-12,
                        err_msg=channel,
                    )

        # the range, every frame of it: solved on the cut's own frame axis
        clip       = self._animated(self._scaled(6, seed=5), orient=True)
        clip.frame = 2
        whole      = clip.copy()
        whole.frames -= pose
        before = {c: np.array(getattr(clip.frames, c)) for c in CHANNELS}

        cut = clip.frames[1:4]
        cut.frames -= pose
        clip.frames[1:4] = cut

        self.assertEqual(clip.frame, 2)
        for channel, values in before.items():
            got = np.asarray(getattr(clip.frames, channel))
            np.testing.assert_array_equal(got[[0, 4, 5]], values[[0, 4, 5]], channel)
        np.testing.assert_allclose(
            np.asarray(clip.frames.world_matrix)[1:4],
            np.asarray(whole.frames.world_matrix)[1:4],
            rtol = 0.0,
            atol = 1e-12,
        )

    def test_a_scrub_loop_of_in_place_writes_is_each_frame_once(self):
        # before clip.frames took the every frame form, scrubbing and writing
        # once a frame was how a clip was covered, and it still is: written
        # every frame, each pass solved the whole clip again, 72 off in world
        clip = self._animated(self._scaled(6, seed=5), orient=True)
        pose = self._scaled(6, seed=17)
        want = self._per_frame("__isub__", clip, pose)

        for frame in range(clip.frame_count):
            clip.frame = frame
            clip -= pose

        np.testing.assert_allclose(
            clip.frames.world_matrix, want.frames.world_matrix, rtol=0.0, atol=1e-12
        )

    def test_a_clip_operand_is_its_bound_frame(self):
        pose         = self._scaled(6, seed=17)
        clip         = self._animated(self._scaled(6, seed=5), orient=True)
        deltas       = self._animated(self._scaled(6, seed=23), seed=21)
        clip.frame   = 3
        deltas.frame = 1

        delta = pose.get_delta(clip)
        self.assertNotIsInstance(delta, ClipData)
        np.testing.assert_array_equal(delta.rotate, pose.get_delta(pose_at(clip, 3)).rotate)
        np.testing.assert_array_equal((pose - clip).rotate, delta.rotate)

        posed = pose.add_delta(deltas)
        self.assertNotIsInstance(posed, ClipData)
        np.testing.assert_array_equal(
            posed.rotate, pose.add_delta(pose_at(deltas, 1)).rotate
        )
        np.testing.assert_array_equal((deltas + pose).rotate, posed.rotate)
        self.assertEqual((clip.frame, deltas.frame), (3, 1))

    # --- one pose against every frame of a clip --- #
    def test_a_pose_less_every_frame_is_a_clip_of_every_frame(self):
        pose  = self._scaled(6, seed=17)
        clip  = self._animated(self._scaled(6, seed=5), orient=True)
        delta = pose.get_delta(clip.frames)

        self._assert_same(delta, self._per_frame("get_delta", pose, clip))
        self.assertEqual(delta.frame, 0)
        self._assert_same(pose - clip.frames, delta)
        self._assert_same(clip.frames - pose, clip.get_delta(pose))

    def test_a_pose_less_every_frame_by_name_orientation_only_matches_per_frame(self):
        pose = self._rig(*self.POSE, seed=17)
        clip = self._animated(self._rig(*self.PARTIAL, seed=5))

        self._assert_same(
            pose.get_delta(clip.frames, by="name", translate=False),
            self._per_frame("get_delta", pose, clip, by="name", translate=False),
        )

    def test_a_selection_less_every_frame_matches_per_frame(self):
        view = self._scaled(6, seed=17)[2:5]
        clip = self._animated(self._scaled(6, seed=5))

        self._assert_same(
            view.get_delta(clip.frames, by="name"),
            self._per_frame("get_delta", view, clip, by="name"),
        )

    def test_a_rig_posed_by_every_delta_is_a_clip_of_every_frame(self):
        rig          = self._scaled(6, seed=5)
        deltas       = self._animated(self._scaled(6, seed=17), seed=21, orient=True)
        deltas.frame = 3
        posed        = rig.add_delta(deltas.frames)

        self._assert_same(posed, self._per_frame("add_delta", rig, deltas))
        self.assertEqual(posed.frame, 0)
        self._assert_same(deltas.frames + rig, posed)

    def test_a_rig_posed_by_every_delta_by_name_orientation_only_matches_per_frame(self):
        rig    = self._rig(*self.POSE, seed=5)
        deltas = self._animated(self._rig(*self.PARTIAL, seed=17), seed=21)

        self._assert_same(
            rig.add_delta(deltas.frames, by="name", translate=False),
            self._per_frame("add_delta", rig, deltas, by="name", translate=False),
        )

    def test_a_selection_posed_by_every_delta_matches_per_frame(self):
        # the source is read where it sits: in its whole rig
        view   = self._scaled(6, seed=5)[[1, 3, 4]]
        deltas = self._animated(self._scaled(6, seed=17), seed=21)

        self._assert_same(
            view.add_delta(deltas.frames, by="name"),
            self._per_frame("add_delta", view, deltas, by="name"),
        )

    def test_a_pose_receiver_leaves_both_sides_alone(self):
        rig        = self._chain(5, seed=5)
        clip       = self._animated(self._chain(5, seed=17), seed=21)
        clip.frame = 3
        before     = {c: np.array(getattr(clip.frames, c)) for c in CHANNELS}
        rest       = {c: np.array(getattr(rig, c)) for c in CHANNELS}

        rig.get_delta(clip.frames)
        rig.add_delta(clip.frames)
        rig.get_delta(clip)
        rig.add_delta(clip)

        self.assertEqual(clip.frame, 3)
        for channel, values in before.items():
            np.testing.assert_array_equal(getattr(clip.frames, channel), values)
        for channel, values in rest.items():
            np.testing.assert_array_equal(getattr(rig, channel), values)

    def test_a_one_frame_clip_is_still_one_pose(self):
        rig  = self._chain(4, seed=5)
        clip = ClipData(self._chain(4, seed=17), frames=1)

        for operand in (clip, clip.frames):
            self.assertNotIsInstance(rig.get_delta(operand), ClipData)
            self.assertNotIsInstance(rig.add_delta(operand), ClipData)

    def test_a_one_frame_clip_beside_a_longer_one_is_one_pose(self):
        # a clip of one frame is one pose wherever it stands; against a clip
        # of six it raised "clips hold 6 and 1 frames", where a pose did not
        long   = self._animated(self._scaled(6, seed=5), orient=True)
        short  = ClipData(self._scaled(6, seed=17), frames=1)
        deltas = self._animated(self._scaled(6, seed=23), seed=21)
        pose   = pose_at(short, 0)

        fast   = long.copy()
        fast.frames -= short
        slow = long.copy()
        slow.frames -= pose
        self._assert_same(fast, slow)

        fast = deltas.copy()
        fast.frames += short
        slow = deltas.copy()
        slow.frames += pose
        self._assert_same(fast, slow)

        self._assert_same(long.get_delta(short), long.get_delta(pose))
        self._assert_same(long.add_delta(short), long.add_delta(pose))

        matched = long.copy()
        matched.frames.match_matrix(short)
        held = long.copy()
        held.frames.match_matrix(pose)
        self._assert_same(matched, held)

        # the one frame clip as the receiver: one pose against every frame
        self._assert_same(short.get_delta(long), pose.get_delta(long.frames))
        self._assert_same(short.add_delta(deltas), pose.add_delta(deltas.frames))

        # in place it stays one frame, reading the other's bound frame
        long.frame = 4
        one        = short.copy()
        one.frames -= long
        self.assertEqual(one.frame_count, 1)
        np.testing.assert_array_equal(
            one.rotate, pose.get_delta(pose_at(long, 4)).rotate
        )

    def test_no_frame_is_solved_on_its_own(self):
        # every frame at once: none of these may run the single pose solve,
        # which is what solving one frame at a time calls
        rig    = self._chain(5, seed=5)
        clip   = self._animated(self._chain(5, seed=17), frames=12)
        deltas = self._animated(self._chain(5, seed=23), frames=12, seed=21)

        isub = mock.patch.object(
            TransformList, "__isub__", autospec=True, side_effect=TransformList.__isub__
        )
        iadd = mock.patch.object(
            TransformList, "__iadd__", autospec=True, side_effect=TransformList.__iadd__
        )
        with isub as sub, iadd as add:
            rig.get_delta(clip.frames)
            rig - clip.frames
            rig.add_delta(deltas.frames)
            deltas.frames + rig
            work = clip.copy()
            work.frames -= rig
            work.frames += rig

        self.assertEqual((sub.call_count, add.call_count), (0, 0))


class TestClipAlgebraEulerFilter(unittest.TestCase):
    """every rotate the clip algebra rebuilds from matrices goes through
    euler_filter per node, frame 0 kept as solved: one frame at a time,
    matrix_to_euler wraps a turn through 180 degrees into a 360 degree step"""

    def _rest(self, extra=False):
        # nothing in the orientation chain, so a delta's rotate is the clip's
        # own rotation read back through matrix_to_euler
        ids = [hierarchy.generate_uuid() for _ in range(3)]
        nodes = [
            TransformData(
                name        = f"j{i}",
                node_type   = "joint",
                uuid        = ids[i],
                parent_uuid = None if i == 0 else ids[i - 1],
                translate   = [0.0, 1.0 if i else 0.0, 0.0],
            )
            for i in range(3)
        ]
        if extra:
            nodes.append(TransformData(name="extra", parent_uuid=ids[0], translate=[1.0, 0.0, 0.0]))
        return HierarchyData(nodes)

    def _sweep(self, first, extra=False):
        """j1 turns about x from ``first`` in 10 degree steps over 9 frames"""
        rest            = self._rest(extra)
        clip            = ClipData(rest, frames=9, start_frame=1, fps=24.0)
        sweep           = first + 10.0 * np.arange(9)
        rotate          = np.zeros((9, len(rest), 3))
        rotate[:, 1, 0] = sweep
        if extra:
            # authored with steps: no matrix rebuilds it, so it is kept as is
            rotate[:, 3, 0] = np.where(np.arange(9) % 2, -170.0, 170.0)
        clip.frames.rotate = rotate
        return rest, clip, sweep

    def _assert_curve(self, clip, node, expected):
        """``node``'s rotate x follows ``expected`` and y, z stay at rest"""
        rotate = np.asarray(clip.frames[node].rotate)[:, 0]
        self.assertTrue(np.allclose(rotate[:, 0], expected, atol=1e-9), rotate[:, 0])
        self.assertTrue(np.allclose(rotate[:, 1:], 0.0, atol=1e-9), rotate[:, 1:])

    def test_a_delta_against_a_pose_does_not_step(self):
        rest, clip, sweep = self._sweep(140.0)
        delta = clip.get_delta(rest)

        # j2 hangs under the turn, so its offset is the same turn
        self._assert_curve(delta, "j1", sweep)
        self._assert_curve(delta, "j2", sweep)
        self._assert_curve(delta, "j0", np.zeros(9))

    def test_a_delta_between_clips_does_not_step(self):
        rest, clip, sweep = self._sweep(140.0)
        delta = clip.get_delta(ClipData(rest, frames=9))
        self._assert_curve(delta, "j1", sweep)

    def test_an_identity_delta_hands_the_curves_back(self):
        rest, clip, sweep = self._sweep(140.0)

        posed = clip.add_delta(rest.get_delta(rest))
        self._assert_curve(posed, "j1", sweep)

        # and a clip of them
        posed = clip.add_delta(ClipData(rest.get_delta(rest), frames=9))
        self._assert_curve(posed, "j1", sweep)

    def test_the_in_place_operators_filter_too(self):
        rest, clip, sweep = self._sweep(140.0)

        delta = clip.copy()
        delta.frames -= rest
        self._assert_curve(delta, "j2", sweep)

        # rest + (clip - rest) is the clip again
        delta.frames += rest
        self._assert_curve(delta, "j1", sweep)
        self._assert_curve(delta, "j2", np.zeros(9))

    def test_a_pose_receiver_filters_too(self):
        rest, clip, sweep = self._sweep(140.0)

        self._assert_curve(rest.get_delta(clip.frames), "j1", -sweep)
        self._assert_curve(rest.add_delta(clip.get_delta(rest).frames), "j1", sweep)

    def test_a_frame_scaled_to_zero_stays_on_its_own_frame(self):
        # a joint scaled to 0 on one frame, the way a game hides it, has no
        # rotation there: nan. euler_filter carried that nan to every later
        # frame of the curve, so every later frame of the joint and the
        # joints below lost their rotate and world matrices
        rest, clip, sweep = self._sweep(140.0)
        scale             = np.array(clip.frames.scale)
        scale[3, 1]       = 0.0
        clip.frames.scale = scale
        identity          = rest.get_delta(rest)
        deltas            = clip.get_delta(rest)

        def reference(solve, source=clip):
            poses             = [solve(pose_at(source, frame)) for frame in range(source.frame_count)]
            out               = ClipData.from_poses(poses, start_frame=clip.start_frame, fps=clip.fps)
            out.frames.rotate = _filtered(np.array(out.frames.rotate), out.rotate_order)
            return out

        cases = {
            "get_delta": (
                lambda: clip.get_delta(rest), lambda p: p.get_delta(rest)
            ),
            "add_delta": (
                lambda: clip.add_delta(identity), lambda p: p.add_delta(identity)
            ),
            "frames -=": (
                lambda: clip.copy().frames.__isub__(rest)._clip, lambda p: p.__isub__(rest)
            ),
            "frames +=": (
                lambda: deltas.copy().frames.__iadd__(rest)._clip,
                lambda p: p.__iadd__(rest),
                deltas,
            ),
            "pose - frames": (
                lambda: rest - clip.frames, lambda p: rest.get_delta(p)
            ),
            "pose.add_delta(frames)": (
                lambda: rest.add_delta(deltas.frames), lambda p: rest.add_delta(p), deltas
            ),
        }
        for label, (fast, slow, *source) in cases.items():
            with self.subTest(label):
                got, want = fast(), reference(slow, *source)
                rotate = np.array(got.frames.rotate)
                np.testing.assert_allclose(
                    rotate, want.frames.rotate, rtol=0.0, atol=1e-9, equal_nan=True
                )
                np.testing.assert_allclose(
                    got.frames.world_matrix, want.frames.world_matrix,
                    rtol=0.0, atol=1e-9, equal_nan=True,
                )
                # whatever is lost is lost on the degenerate frame alone
                lost = np.flatnonzero(np.isnan(rotate).any(axis=(1, 2)))
                self.assertTrue(set(lost.tolist()) <= {3}, lost)

    def test_frame_0_is_kept_as_solved(self):
        # 200 degrees solves to -160 on its own; the filter starts from there
        rest, clip, sweep = self._sweep(200.0)
        delta = clip.get_delta(rest)

        alone = pose_at(clip, 0).get_delta(rest)
        self.assertAlmostEqual(alone["j1"].rotate_x, -160.0, places=9)
        self._assert_curve(delta, "j1", sweep - 360.0)

    def test_a_curve_no_matrix_rebuilds_is_left_as_authored(self):
        rest, clip, _ = self._sweep(140.0, extra=True)
        authored = np.array(clip.frames["extra"].rotate)
        part     = HierarchyData([node.copy() for node in rest if node.name != "extra"])

        # get_delta keeps the clip's own channels on an unpaired node
        delta = clip.get_delta(part, by="name")
        np.testing.assert_array_equal(delta.frames["extra"].rotate, authored)

        # add_delta keeps the delta's own channels on one
        posed = part.add_delta(clip.frames, by="name")
        np.testing.assert_array_equal(posed.frames["extra"].rotate, authored)


class TestPoseOperandWorld(_DeltaRigs, unittest.TestCase):
    """a pose operand's world matrices come from one batched evaluation of
    the hierarchy owning it, and are the ones world_matrix reads node by
    node, bit for bit"""

    def _assert_world(self, nodes):
        expected = np.array([node.world_matrix for node in nodes])
        np.testing.assert_array_equal(ClipData._pose_world(nodes), expected)

    def _scaled(self, count, seed):
        return self._chain(count, seed, scale=True)

    def test_a_whole_rig(self):
        self._assert_world(list(self._scaled(6, seed=3)))

    def test_a_branching_rig_parented_by_uuid(self):
        uuids   = [hierarchy.generate_uuid() for _ in range(6)]
        parents = [None, 0, 1, 0, 3, 1]
        rng     = np.random.default_rng(4)
        rig = HierarchyData(
            [
                TransformData(
                    name=f"n{i}",
                    uuid=uuids[i],
                    parent_uuid=None if p is None else uuids[p],
                    rotate=rng.uniform(-90.0, 90.0, size=3),
                    translate=rng.normal(size=3),
                    scale=rng.uniform(0.5, 2.0, size=3),
                    rotate_axis=rng.uniform(-30.0, 30.0, size=3),
                    joint_orient=rng.uniform(-60.0, 60.0, size=3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(i % 2),
                )
                for i, p in enumerate(parents)
            ]
        )
        self._assert_world(list(rig))

    def test_a_selection_is_placed_in_its_whole_rig(self):
        rig = self._scaled(6, seed=3)
        self._assert_world(rig[2:5])
        self._assert_world(rig[[4, 1]])

    def test_nodes_of_two_rigs_and_nodes_no_rig_owns(self):
        a = self._scaled(4, seed=3)
        b = self._scaled(4, seed=4)
        loose = TransformData(
            name        = "loose",
            parent_uuid = a["j0"].uuid,
            rotate      = [10.0, 20.0, 30.0],
            translate   = [1.0, 2.0, 3.0],
            scale       = [1.0, 2.0, 3.0],
        )
        self._assert_world([a["j2"], b["j3"], loose, a["j0"], b["j1"]])

    def test_a_node_its_owner_no_longer_lists_reads_its_own(self):
        # del leaves the node pointing at the rig it left
        rig  = self._scaled(5, seed=3)
        gone = rig[4]
        del rig[4]
        self._assert_world([rig["j1"], gone])

        # an owner left listing nothing at all
        alone = self._scaled(1, seed=4)
        last  = alone[0]
        del alone[0]
        self._assert_world([last])

    def test_a_clip_view_reads_the_bound_frame(self):
        clip       = self._animated(self._scaled(5, seed=3), orient=True)
        clip.frame = 4
        self._assert_world(clip[1:4])

    def test_get_delta_against_a_selection_matches_per_frame(self):
        clip = self._animated(self._scaled(6, seed=5), orient=True)
        view = self._scaled(6, seed=17)[[1, 3, 4]]

        self._assert_same(
            clip.get_delta(view, by="name"),
            self._per_frame("get_delta", clip, view, by="name"),
        )


class TestClipBranchResets(unittest.TestCase):
    """a write on the bound frame clears the cached world matrices of the
    written node's whole branch, so every read after it, and every node
    solved after it, sees the new pose. The reference is the same write made
    on a single pose copy of that frame, its world matrices then computed
    from scratch"""

    def _rig(self, count=14, seed=3):
        rng   = np.random.default_rng(seed)
        uuids = [hierarchy.generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            parent = None
            if i:
                p      = int(rng.integers(max(0, i - 4), i))
                parent = uuids[p]
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    uuid=uuids[i],
                    parent_uuid=parent,
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-90.0, 90.0, size=3),
                    scale=rng.uniform(0.7, 1.4, size=3),
                    joint_orient=rng.uniform(-45.0, 45.0, size=3),
                    rotate_axis=rng.uniform(-20.0, 20.0, size=3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
            )
        return HierarchyData(nodes)

    def _clip(self, rig, frames=4, seed=4):
        rng                   = np.random.default_rng(seed)
        clip                  = ClipData(rig, frames=frames)
        shape                 = clip.frames.rotate.shape
        clip.frames.rotate    = clip.frames.rotate + rng.normal(size=shape) * 25.0
        clip.frames.translate = clip.frames.translate + rng.normal(size=shape) * 0.3
        clip.frames.scale     = clip.frames.scale * rng.uniform(0.8, 1.25, size=shape)
        clip.frame            = 0
        return clip

    @staticmethod
    def _pose(clip) -> HierarchyData:
        """the bound frame as a single pose hierarchy of its own"""
        return TransformList._owned_copy(clip)

    def _assert_like(self, clip, pose, label):
        """the clip's cached reads equal the pose's channels and its world
        matrices computed from scratch"""
        self.assertEqual(clip.name, pose.name, label)
        for channel in ("rotate", "translate", "scale", "rotate_axis", "joint_orient"):
            np.testing.assert_allclose(
                getattr(clip, channel), getattr(pose, channel), atol=1e-9,
                err_msg=f"{label}: {channel}",
            )
        np.testing.assert_allclose(
            clip.world_matrix, pose.copy().world_matrix, atol=1e-9, err_msg=label
        )

    # --- one write per frame, as a user scrubs and writes --- #
    WRITES = {
        "rotate": lambda h, f, o: setattr(h, "rotate", o.frames.rotate[f]),
        "rotate_x": lambda h, f, o: setattr(h, "rotate_x", o.frames.rotate_x[f]),
        "translate": lambda h, f, o: setattr(h, "translate", o.frames.translate[f]),
        "scale": lambda h, f, o: setattr(h, "scale", o.frames.scale[f]),
        "joint_orient": lambda h, f, o: setattr(h, "joint_orient", o.frames.joint_orient[f]),
        "rotate_axis": lambda h, f, o: setattr(h, "rotate_axis", o.frames.rotate_axis[f]),
        "one node": lambda h, f, o: setattr(h["j1"], "translate_y", 3.0 + f),
        "isub": lambda h, f, o: h.__isub__(TransformList._owned_copy(o)),
    }

    def test_a_write_on_every_frame_matches_the_single_pose_write(self):
        for label, write in self.WRITES.items():
            with self.subTest(label):
                clip  = self._clip(self._rig())
                other = self._clip(self._rig(), seed=9)
                for frame in range(clip.frame_count):
                    clip.frame  = frame
                    other.frame = frame
                    pose        = self._pose(clip)

                    clip.world_matrix  # every node's world matrix cached
                    write(clip, frame, other)
                    write(pose, frame, other)
                    self._assert_like(clip, pose, f"{label} frame {frame}")

    def test_a_solve_on_every_frame_lands_on_its_target(self):
        # each node solves against its parent's world matrix: a parent whose
        # write missed a child's cache would leave that child off target. The
        # two rigs share every structural value, so each frame is reachable
        for label in ("world_matrix", "match_matrix"):
            with self.subTest(label):
                clip   = self._clip(self._rig())
                other  = self._clip(self._rig(), seed=9)
                target = other.frames.world_matrix
                for frame in range(clip.frame_count):
                    clip.frame  = frame
                    other.frame = frame
                    pose        = self._pose(clip)

                    clip.world_matrix
                    if label == "world_matrix":
                        clip.world_matrix = target[frame]
                        pose.world_matrix = target[frame]
                    else:
                        clip.match_matrix(other)
                        pose.match_matrix(TransformList._owned_copy(other))

                    np.testing.assert_allclose(
                        clip.world_matrix, target[frame], atol=1e-9,
                        err_msg=f"{label} frame {frame}",
                    )
                    self._assert_like(clip, pose, f"{label} frame {frame}")

    def test_a_write_scans_the_clip_a_few_times_not_once_per_node(self):
        # finding a node's children used to scan the whole clip, once for
        # every node of every branch a write reset: 406 passes over the
        # clip's 40 nodes for one rotate write, 1739 for world_matrix. A
        # node's get_parent still scans the clip, and solving a world matrix
        # asks it a few times a node, so that write is held to 3 passes a
        # node rather than 3 in all
        class Walked(list):
            walks = 0

            def __iter__(self):
                self.walks += 1
                return super().__iter__()

        rig       = self._rig(count=40)
        clip      = ClipData(rig, frames=3)
        clip.list = Walked(clip.list)
        other     = self._clip(rig, seed=9)
        writes = {
            "rotate":       lambda: setattr(clip, "rotate", other.frames.rotate[1]),
            "rotate_order": lambda: setattr(clip, "rotate_order", (clip.rotate_order + 1) % 6),
            "segment_scale_compensate": lambda: setattr(
                clip, "segment_scale_compensate", ~clip.segment_scale_compensate
            ),
            "world_matrix": lambda: setattr(clip, "world_matrix", other.frames.world_matrix[2]),
            "isub":         lambda: clip.__isub__(rig),
        }
        limits = {"world_matrix": 3 * len(clip)}
        for label, write in writes.items():
            clip.world_matrix
            clip.list.walks = 0
            write()
            self.assertLessEqual(clip.list.walks, limits.get(label, 3), label)

    # --- one structural edit, shared by every frame --- #
    EDITS = {
        "rotate_order": lambda h: setattr(
            h, "rotate_order", [(i * 5) % 6 for i in range(len(h))]
        ),
        "segment_scale_compensate": lambda h: setattr(
            h, "segment_scale_compensate", [i % 3 == 0 for i in range(len(h))]
        ),
        "set_parent local": lambda h: h[5:9].set_parent("j2", world_space=False),
        "set_parent to the world": lambda h: h["j6"].set_parent(None, world_space=False),
        "pop": lambda h: h.pop(3),
        # raw structural writes: nothing clears a branch, the world matrices
        # cached before them must not be trusted after
        "parent_uuid to the world": lambda h: setattr(h["j6"], "parent_uuid", None),
        "parent_uuid of another node": lambda h: setattr(h["j7"], "parent_uuid", h["j1"].uuid),
        "del": lambda h: h.__delitem__(3),
    }

    def test_a_structural_edit_matches_the_single_pose_edit(self):
        for label, edit in self.EDITS.items():
            with self.subTest(label):
                clip  = self._clip(self._rig())
                poses = []
                for frame in range(clip.frame_count):
                    clip.frame = frame
                    poses.append(self._pose(clip))
                    edit(poses[-1])

                # made on the last frame with every cache filled, the
                # clip's index too, then read before any scrub could
                # clear them
                clip.world_matrix
                clip.get_branch()
                edit(clip)
                self._assert_like(clip, poses[-1], f"{label} bound frame")

                framed = clip.frames.world_matrix
                for frame, pose in enumerate(poses):
                    clip.frame = frame
                    self._assert_like(clip, pose, f"{label} frame {frame}")
                    np.testing.assert_allclose(
                        framed[frame], pose.copy().world_matrix, atol=1e-9,
                        err_msg=f"{label} frames.world_matrix {frame}",
                    )


class TestClipFramedSolve(unittest.TestCase):
    """world_matrix / matrix / quaternion writes and match_* solve every frame
    of a clip at once. Each must equal the single pose route: every frame
    copied out as a HierarchyData and solved by HierarchyData / TransformData
    methods alone, world matrices within 1e-9, rotate equal once that route's
    curves go through euler_filter as well"""

    def _rig(self, seed, count=9):
        # a branching tree with everything that makes a solve non trivial:
        # rotate_axis, joint_orient, every rotate order, non-uniform scale and
        # mixed segment scale compensation
        rng   = np.random.default_rng(seed)
        ids   = [hierarchy.generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            parent = None
            if i:
                p      = int(rng.integers(0, i))
                parent = ids[p]
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    uuid=ids[i],
                    parent_uuid=parent,
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-60.0, 60.0, size=3),
                    scale=rng.uniform(0.8, 1.25, size=3),
                    joint_orient=rng.uniform(-90.0, 90.0, size=3),
                    rotate_axis=rng.uniform(-30.0, 30.0, size=3),
                    rotate_order=i % 6,
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
            )
        return HierarchyData(nodes)

    def _animated(self, rig, frames=6, seed=9, orient=False):
        rng                   = np.random.default_rng(seed)
        clip                  = ClipData(rig, frames=frames, start_frame=1001, fps=30.0)
        shape                 = clip.frames.rotate.shape
        clip.frames.rotate    = clip.frames.rotate + np.cumsum(rng.normal(size=shape) * 8.0, axis=0)
        clip.frames.translate = clip.frames.translate + rng.normal(size=shape) * 0.2
        clip.frames.scale     = clip.frames.scale * rng.uniform(0.85, 1.15, size=shape)
        if orient:
            clip.frames.joint_orient = clip.frames.joint_orient + rng.normal(size=shape) * 10.0
            clip.frames.rotate_axis  = clip.frames.rotate_axis + rng.normal(size=shape) * 10.0
        clip.frame = 0
        return clip

    def _pair(self, orient=False):
        clip  = self._animated(self._rig(5), seed=9, orient=orient)
        other = self._animated(self._rig(17), seed=21, orient=orient)
        return clip, other

    @staticmethod
    def _view(rig, names):
        """the nodes of ``rig`` called ``names``, as a TransformList view"""
        return TransformList([rig[name] for name in names])

    @staticmethod
    def _pose(clip, frame):
        """frame ``frame`` of a clip as a plain single pose HierarchyData"""
        clip.frame = frame
        return clip._owned_copy()

    def _reference(self, clip, solve, names=None):
        """the single pose route: ``solve(pose, frame)`` on a copy of every
        frame, gathered back into a clip, the rotate of ``names`` (every
        node by default) through euler_filter, frame 0 kept"""
        poses = []
        for frame in range(clip.frame_count):
            pose = self._pose(clip, frame)
            solve(pose, frame)
            poses.append(pose)
        out               = ClipData.from_poses(poses, start_frame=clip.start_frame, fps=clip.fps)
        cols              = [out.index(n) for n in (names or out.name)]
        orders            = np.asarray(out.rotate_order, dtype=np.intp)[cols]
        rotate            = np.array(out.frames.rotate)
        rotate[:, cols]   = np.degrees(euler_filter(np.radians(rotate[:, cols]), orders))
        out.frames.rotate = rotate
        return out

    def _assert_same(self, fast, slow):
        self.assertEqual(fast.name, slow.name)
        self.assertTrue(
            np.allclose(fast.frames.world_matrix, slow.frames.world_matrix, rtol=0.0, atol=1e-9),
            "world",
        )
        for channel in ("rotate", "translate", "scale", "rotate_axis", "joint_orient"):
            self.assertTrue(
                np.allclose(
                    getattr(fast.frames, channel), getattr(slow.frames, channel),
                    rtol=0.0, atol=1e-9,
                ),
                channel,
            )

    # --- framed writes --- #
    def test_world_matrix_matches_per_frame(self):
        for orient in (False, True):
            with self.subTest(orient=orient):
                clip, other = self._pair(orient)
                W    = np.array(other.frames.world_matrix)
                slow = self._reference(clip, lambda p, f: setattr(p, "world_matrix", W[f]))

                clip.frames.world_matrix = W
                self._assert_same(clip, slow)

    def test_a_world_the_rig_can_hold_reads_back(self):
        # another take of the same rig: the same compensation flags, so every
        # world it holds can be held here too
        clip  = self._animated(self._rig(5), seed=9, orient=True)
        other = self._animated(self._rig(5), seed=21, orient=True)
        W     = np.array(other.frames.world_matrix)
        clip.frames.world_matrix = W
        self.assertTrue(np.allclose(clip.frames.world_matrix, W, rtol=0.0, atol=1e-9))

        clip = self._animated(self._rig(5), seed=9)
        clip.frames.match_matrix(other)
        self.assertTrue(np.allclose(clip.frames.world_matrix, W, rtol=0.0, atol=1e-9))

    def test_matrix_matches_per_frame(self):
        clip, other = self._pair(orient=True)
        L    = np.array(other.frames.matrix)
        slow = self._reference(clip, lambda p, f: setattr(p, "matrix", L[f]))

        clip.frames.matrix = L
        self._assert_same(clip, slow)

    def test_quaternion_reads_and_writes_every_frame(self):
        clip, other = self._pair(orient=True)

        # the read: TransformData.quaternion of every node on every frame
        Q = np.array(other.frames.quaternion)
        self.assertEqual(Q.shape, (other.frame_count, len(other), 4))
        for frame in range(other.frame_count):
            other.frame = frame
            each        = np.array([node.quaternion for node in other])
            self.assertTrue(np.allclose(Q[frame], each, rtol=0.0, atol=1e-12))
            self.assertTrue(np.allclose(other.quaternion, each, rtol=0.0, atol=1e-12))

        # the write: was silently dropped on a throwaway axis before
        slow                   = self._reference(clip, lambda p, f: setattr(p, "quaternion", Q[f]))
        clip.frames.quaternion = Q
        self._assert_same(clip, slow)

    def test_values_broadcast_over_nodes_and_frames(self):
        clip, other = self._pair()
        frames = clip.frame_count
        names  = ["j3", "j7", "j8"]
        cols   = [clip.index(n) for n in names]
        W      = np.array(other.frames.world_matrix)

        # (n, 4, 4): one matrix per node, held on every frame
        held = W[2, cols]
        slow = self._reference(
            clip, lambda p, f: setattr(self._view(p, names), "world_matrix", held), names
        )
        fast                            = clip.copy()
        fast.frames[names].world_matrix = held
        self._assert_same(fast, slow)

        # (4, 4): one matrix for every node and frame
        one  = W[4, 0]
        slow = self._reference(clip, lambda p, f: setattr(p["j8"], "world_matrix", one), ["j8"])
        fast = clip.copy()
        fast.frames["j8"].world_matrix = one
        self._assert_same(fast, slow)

        # (F, n, 4, 4) on a scoped axis
        slow = self._reference(
            clip, lambda p, f: setattr(self._view(p, names), "world_matrix", W[f, cols]), names
        )
        fast                            = clip.copy()
        fast.frames[names].world_matrix = W[:, cols]
        self._assert_same(fast, slow)
        self.assertEqual(frames, fast.frame_count)

    # --- framed match_* --- #
    def test_match_against_a_clip_pairs_frame_for_frame(self):
        clip, other = self._pair(orient=True)
        for method in ("match_matrix", "match_translate", "match_rotate"):
            with self.subTest(method=method):
                slow = self._reference(
                    clip, lambda p, f: getattr(p, method)(self._pose(other, f))
                )
                fast = clip.copy()
                getattr(fast.frames, method)(other)
                self._assert_same(fast, slow)

    def test_match_against_a_pose_holds_it_on_every_frame(self):
        clip, other = self._pair()
        pose = self._pose(other, 3)
        for method in ("match_matrix", "match_translate", "match_rotate"):
            with self.subTest(method=method):
                slow = self._reference(clip, lambda p, f: getattr(p, method)(pose))
                fast = clip.copy()
                getattr(fast.frames, method)(pose)
                self._assert_same(fast, slow)

    def test_match_a_scoped_axis_against_another(self):
        clip, other = self._pair()
        names = ["j1", "j4", "j6"]
        slow = self._reference(
            clip,
            lambda p, f: self._view(p, names).match_matrix(
                self._view(self._pose(other, f), names)
            ),
            names,
        )
        fast = clip.copy()
        fast.frames[names].match_matrix(other.frames[names])
        self._assert_same(fast, slow)

        # one node against one node, held on every frame
        node = self._pose(other, 2)["j5"]
        slow = self._reference(clip, lambda p, f: p["j6"].match_translate(node, y=False), ["j6"])
        fast = clip.copy()
        fast.frames["j6"].match_translate(node, y=False)
        self._assert_same(fast, slow)

    def test_match_translate_takes_positions(self):
        clip, other = self._pair()
        P = np.array(other.frames.world_matrix)[..., 3, :3]
        for label, value, at in (
            ("(F, n, 3)", P, lambda f: P[f]),
            ("(n, 3)", P[1], lambda f: P[1]),
            ("(3,)", P[2, 0], lambda f: P[2, 0]),
        ):
            with self.subTest(value=label):
                slow = self._reference(
                    clip, lambda p, f: p.match_translate(at(f), x=True, y=False, z=True)
                )
                fast = clip.copy()
                fast.frames.match_translate(value, y=False)
                self._assert_same(fast, slow)

    def test_match_rotate_takes_eulers_and_quaternions(self):
        clip, other = self._pair(orient=True)
        E = np.array(other.frames.rotate)
        Q = np.array(other.frames.quaternion)
        for label, value, at in (
            ("eulers (F, n, 3)", E, lambda f: E[f]),
            ("quaternions (F, n, 4)", Q, lambda f: Q[f]),
            ("one euler", E[3, 2], lambda f: E[3, 2]),
            ("quaternions (n, 4)", Q[1], lambda f: Q[1]),
        ):
            with self.subTest(value=label):
                slow = self._reference(clip, lambda p, f: p.match_rotate(at(f)))
                fast = clip.copy()
                fast.frames.match_rotate(value)
                self._assert_same(fast, slow)

    # --- what the solve keeps --- #
    def test_rebuilt_rotate_is_euler_filtered(self):
        # one joint spun a full turn and a half about x over the clip: read
        # back one frame at a time its rotate steps by 360 at +-180, which
        # euler_filter takes out, frame 0 kept as computed
        rig    = HierarchyData([TransformData("a", node_type="joint", rotate_order=0)])
        clip   = ClipData(rig, frames=19)
        angles = np.linspace(0.0, 540.0, 19)
        spun   = ClipData(rig, frames=19)
        spun.frames.rotate = np.stack(
            [angles, np.zeros(19), np.zeros(19)], axis=-1
        )[:, None]
        for write in ("world_matrix", "matrix", "quaternion"):
            with self.subTest(write=write):
                fast = clip.copy()
                setattr(fast.frames, write, np.asarray(getattr(spun.frames, write)))
                rotate = np.asarray(fast.frames.rotate)[:, 0]
                self.assertTrue(np.allclose(rotate[:, 0], angles, atol=1e-9))
                self.assertTrue(np.abs(np.diff(rotate, axis=0)).max() < 31.0)

    def test_an_element_already_on_target_keeps_its_channels(self):
        # a clip written its own world is left exactly as it was, wrapped
        # curves and all: nothing was rebuilt, so nothing is filtered
        clip, _ = self._pair()
        rotate = np.array(clip.frames.rotate)
        rotate[3:, 2] += 360.0
        clip.frames.rotate = rotate
        before             = {c: np.array(getattr(clip.frames, c)) for c in ("scale", "rotate", "translate")}

        clip.frames.world_matrix = np.array(clip.frames.world_matrix)
        clip.frames.matrix       = np.array(clip.frames.matrix)
        clip.frames.match_matrix(clip.copy())

        for channel, values in before.items():
            self.assertTrue(np.array_equal(getattr(clip.frames, channel), values), channel)

    def test_untargeted_nodes_keep_their_channels_and_follow(self):
        clip, other = self._pair()
        below  = [node.name for node in clip["j0"].get_branch()][1:]
        before = {c: np.array(getattr(clip.frames, c)) for c in ("scale", "rotate", "translate")}
        world  = np.array(clip.frames[below].world_matrix)

        clip.frames["j0"].match_matrix(other.frames["j0"])

        rest = [clip.index(name) for name in below]
        for channel, values in before.items():
            got = np.asarray(getattr(clip.frames, channel))
            self.assertTrue(np.array_equal(got[:, rest], values[:, rest]), channel)
        self.assertTrue(np.allclose(
            clip.frames["j0"].world_matrix, other.frames["j0"].world_matrix, rtol=0.0, atol=1e-9
        ))
        # the branch below rides along with the root it hangs from
        self.assertTrue(len(below) > 1)
        self.assertFalse(np.allclose(clip.frames[below].world_matrix, world))

    def test_a_zero_or_mirrored_scale_splits_as_the_setter_does(self):
        # a flattened row takes what it leaves out from the node's current
        # orientation; a mirror goes on scale z. a zero scale goes on a leaf:
        # a child can not solve under a parent with no inverse
        clip, other = self._pair()
        parents             = clip._parent_indices(clip.list)
        leaves              = [i for i in range(len(clip)) if i not in parents]
        W                   = np.array(other.frames.world_matrix)
        W[1, leaves[0], 0]  = 0.0          # one flattened row
        W[2, leaves[1], :2] = 0.0          # two
        W[4, leaves[0], :3] = 0.0          # all three
        W[3, 1, 2]          = -W[3, 1, 2]  # a mirror, above other nodes
        slow                = self._reference(clip, lambda p, f: setattr(p, "world_matrix", W[f]))

        clip.frames.world_matrix = W
        self._assert_same(clip, slow)

    def test_a_node_listed_twice_keeps_its_last_value(self):
        clip, other = self._pair()
        W = np.array(other.frames.world_matrix)
        first, last = W[:, 2], W[:, 3]

        fast                                   = clip.copy()
        fast.frames[["j4", "j4"]].world_matrix = np.stack([first, last], axis=1)
        once                                   = clip.copy()
        once.frames["j4"].world_matrix         = last[:, None]
        self._assert_same(fast, once)

    def test_bad_operands_raise_before_writing(self):
        clip, other = self._pair()
        before = np.array(clip.frames.rotate)
        short  = self._animated(self._rig(17), frames=4, seed=21)

        with self.assertRaises(ValueError):
            clip.frames.world_matrix = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            clip.frames.world_matrix = np.zeros((2, 3, 4, 4))
        with self.assertRaises(ValueError):
            clip.frames.match_matrix(short)
        with self.assertRaises(ValueError):
            clip.frames.match_rotate(np.zeros(5))
        with self.assertRaises(ValueError):
            clip.frames[["j1", "j2"]].match_translate(other)
        with self.assertRaises(TypeError):
            clip.frames.match_matrix(np.eye(4))
        self.assertTrue(np.array_equal(clip.frames.rotate, before))

    def test_a_held_axis_solves_the_nodes_it_was_given(self):
        # an axis kept across an edit that reordered the nodes solved the
        # node now at its old position: clip['j2'].set_parent('j0') moved j6
        # where j5 was written, and children with it
        reorders = {
            "set_parent local": lambda c: c["j2"].set_parent(None, world_space=False),
            "sort":             lambda c: c.sort(key=lambda node: -int(node.name[1:])),
        }
        _, other = self._pair()
        W = np.array(other.frames.world_matrix)
        L = np.array(other.frames.matrix)
        Q = np.array(other.frames.quaternion)
        writes = {
            "world_matrix": lambda a, n: setattr(a, "world_matrix", W[:, n]),
            "matrix": lambda a, n: setattr(a, "matrix", L[:, n]),
            "quaternion": lambda a, n: setattr(a, "quaternion", Q[:, n]),
            "match_matrix": lambda a, n: a.match_matrix(TransformList([other[i] for i in n])),
            "match_rotate": lambda a, n: a.match_rotate(TransformList([other[i] for i in n])),
            "match_translate": lambda a, n: a.match_translate(
                TransformList([other[i] for i in n])
            ),
        }
        for label, reorder in reorders.items():
            for scope in (["j5"], ["j7", "j3"], None):
                for write_label, write in writes.items():
                    with self.subTest(label, scope=scope, write=write_label):
                        held, fresh = self._pair()[0], self._pair()[0]
                        axis = held.frames if scope is None else held.frames[scope]
                        reorder(held)
                        reorder(fresh)
                        nodes = list(range(len(other))) if scope is None else [
                            int(name[1:]) for name in scope
                        ]

                        write(axis, nodes)
                        write(fresh.frames if scope is None else fresh.frames[scope], nodes)
                        self.assertEqual(held.name, fresh.name)
                        for channel in ("scale", "rotate", "translate"):
                            self.assertTrue(np.array_equal(
                                getattr(held.frames, channel), getattr(fresh.frames, channel)
                            ), channel)

    def test_a_node_taken_out_of_its_clip_takes_the_single_pose_route(self):
        # a removed node still points at its clip: the list setters and the
        # quaternion read took it for a playback head and looked for its
        # column, a KeyError
        for take_out in ("remove", "del"):
            with self.subTest(take_out):
                clip, other = self._pair()
                gone = clip["j3"]
                if take_out == "remove":
                    clip.remove(gone)
                else:
                    del clip[clip.index("j3")]
                self.assertIs(gone._hierarchy, clip)
                kept = clip["j1"]

                np.testing.assert_allclose(
                    TransformList([gone]).quaternion[0], gone.quaternion, atol=1e-12
                )
                TransformList([gone]).quaternion = [0.0, 0.0, 0.0, 1.0]
                TransformList([gone]).matrix = np.eye(4)
                world = gone.world_matrix
                world[3, :3] += 1.0
                TransformList([gone]).world_matrix = world
                np.testing.assert_allclose(gone.world_matrix, world, atol=1e-9)

                pair = TransformList([gone, kept])
                pair.match_matrix(other["j0"])
                pair.match_rotate([10.0, 20.0, 30.0])
                pair.match_translate([1.0, 2.0, 3.0])
                for node in pair:
                    np.testing.assert_allclose(
                        node.world_matrix[3, :3], [1.0, 2.0, 3.0], atol=1e-9
                    )

    def _small_scale_chain(self, frames=4, count=8, seed=3):
        rng   = np.random.default_rng(seed)
        nodes = []
        for i in range(count):
            nodes.append(
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    parent_uuid  = nodes[-1].uuid if nodes else None,
                    uuid         = hierarchy.generate_uuid(),
                    translate    = rng.normal(size=3),
                    rotate       = rng.uniform(-60.0, 60.0, size=3),
                    scale        = [1e-4] * 3 if i == 0 else rng.uniform(0.8, 1.25, size=3),
                    rotate_order = i % 6,
                )
            )
        return self._animated(HierarchyData(nodes), frames=frames, seed=seed)

    def test_the_keep_test_is_on_the_local_matrix(self):
        # under a parent scaled 1e-4 a child within 1e-8 of its target in
        # world is not within 1e-8 in local: a fresh copy of the frame solves
        # it, so the clip has to as well. testing the world too kept the
        # channels of every child 5e-9 from its target
        clip = self._small_scale_chain()
        W    = np.array(clip.frames.world_matrix)
        W[..., 3, :3] += 5e-9
        slow = self._reference(clip, lambda p, f: setattr(p, "world_matrix", W[f]))

        clip.frames.world_matrix = W
        self._assert_same(clip, slow)
        below = np.array(clip.frames.world_matrix)[:, 1:]
        self.assertTrue(np.allclose(below, W[:, 1:], rtol=0.0, atol=1e-12))

    def _gimbal_chain(self, eps, seed, count=12, frames=5):
        # every joint's middle euler angle eps degrees from +-90, where
        # matrix_to_euler snaps the angles a rotation reads back as
        rng   = np.random.default_rng(seed)
        nodes = []
        for i in range(count):
            order          = i % 6
            rotate         = rng.uniform(-170.0, 170.0, size=3)
            middle         = MAYA_EA[order][1]
            rotate[middle] = (90.0 - eps) * rng.choice([-1.0, 1.0])
            nodes.append(
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    parent_uuid  = nodes[-1].uuid if nodes else None,
                    uuid         = hierarchy.generate_uuid(),
                    translate    = rng.normal(size=3),
                    rotate       = rotate,
                    scale        = rng.uniform(0.8, 1.25, size=3),
                    rotate_order = order,
                )
            )
        clip                  = ClipData(HierarchyData(nodes), frames=frames)
        shape                 = clip.frames.translate.shape
        clip.frames.translate = clip.frames.translate + rng.normal(size=shape) * 0.1
        return clip

    def test_next_to_gimbal_lock_the_solve_matches_the_single_pose_route(self):
        # the eulers a rotation reads back as there hold a slightly other
        # rotation: carried down as a matrix it left the children up to 2e-7
        # from where the frame by frame route puts them
        for eps in (1e-7, 1e-3):
            for seed in (1, 2, 3):
                with self.subTest(eps=eps, seed=seed):
                    clip  = self._gimbal_chain(eps, seed)
                    other = self._gimbal_chain(eps, seed + 10)
                    W     = np.array(other.frames.world_matrix)
                    slow  = self._reference(clip, lambda p, f: setattr(p, "world_matrix", W[f]))

                    clip.frames.world_matrix = W
                    self.assertTrue(np.allclose(
                        clip.frames.world_matrix, slow.frames.world_matrix, rtol=0.0, atol=1e-9
                    ))

    def test_a_sheared_target_makes_the_round_trip_on_its_level(self):
        # a target its channels can not hold, here a shear, is whatever the
        # eulers make of it, and the nodes below solve under that, as they do
        # frame by frame. carried down as the sheared matrix instead, the
        # child landed 0.4 to 1.0 off in world, 19 to 62 degrees in rotate
        rng   = np.random.default_rng(5)
        nodes = []
        for i in range(3):
            nodes.append(
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    parent_uuid  = nodes[-1].uuid if nodes else None,
                    uuid         = hierarchy.generate_uuid(),
                    translate    = rng.normal(size=3),
                    rotate       = rng.uniform(-60.0, 60.0, size=3),
                    rotate_order = i,
                )
            )
        rig  = HierarchyData(nodes)
        clip = self._animated(rig, frames=5, seed=7)
        W    = np.array(self._animated(rig, frames=5, seed=8).frames.world_matrix)
        W[:, 1, 0, :3] += 0.45 * W[:, 1, 1, :3]  # j1 sheared, j2 targeted below it
        slow                     = self._reference(clip, lambda p, f: setattr(p, "world_matrix", W[f]))
        fast                     = clip.copy()
        fast.frames.world_matrix = W
        self._assert_same(fast, slow)

        # the same through match_matrix: a source whose j1 hangs under a
        # non-uniformly scaled j0 that does not compensate, while j1 is a
        # root of the clip
        source             = HierarchyData([node.copy() for node in rig])
        source["j0"].scale = (1.6, 0.7, 1.0)
        source             = self._animated(source, frames=5, seed=9)
        target             = clip.copy()
        target["j1"].set_parent(None, world_space=False)
        slow = self._reference(
            target,
            lambda p, f: TransformList([p["j1"], p["j2"]]).match_matrix(
                TransformList(self._pose(source, f)[1:])
            ),
            names=["j1", "j2"],
        )
        target.frames[["j1", "j2"]].match_matrix(source.frames[["j1", "j2"]])
        self._assert_same(target, slow)

    def test_the_filter_keeps_frame_0_and_untouched_keys_bit_for_bit(self):
        # euler_filter works in radians: sending every key there and back
        # moved some by an ulp, frame 0 included, which the filter keeps
        from cgmath.hierarchy.hierarchy import _filtered

        rng     = np.random.default_rng(3)
        degrees = rng.uniform(-170.0, 170.0, (6, 40, 3))
        degrees[2:, :20] += 360.0  # a turn the filter takes out
        orders = np.arange(40) % 6
        self.assertFalse(np.array_equal(np.degrees(np.radians(degrees)), degrees))

        out = _filtered(degrees, orders)
        self.assertTrue(np.array_equal(out[0], degrees[0]))
        reference = np.degrees(euler_filter(np.radians(degrees), orders))
        moved     = np.abs(reference - degrees) > 1e-9
        self.assertTrue(moved.any() and not moved.all())
        self.assertTrue(np.array_equal(out[~moved], degrees[~moved]))
        self.assertTrue(np.allclose(out, reference, rtol=0.0, atol=1e-10))

    def test_every_framed_solve_filters_through_the_one_helper(self):
        clip, other = self._pair(orient=True)
        writes = {
            "world_matrix": lambda c: setattr(c.frames, "world_matrix", other.frames.world_matrix),
            "matrix":       lambda c: setattr(c.frames, "matrix", other.frames.matrix),
            "quaternion":   lambda c: setattr(c.frames, "quaternion", other.frames.quaternion),
            "match_rotate": lambda c: c.frames.match_rotate(other),
        }
        for label, write in writes.items():
            with self.subTest(write=label), mock.patch.object(
                hierarchy.hierarchy, "_filtered", wraps=hierarchy.hierarchy._filtered
            ) as helper, mock.patch.object(
                hierarchy.hierarchy, "euler_filter", wraps=hierarchy.hierarchy.euler_filter
            ) as kernel:
                write(clip.copy())
                self.assertEqual(helper.call_count, 1)
                self.assertEqual(kernel.call_count, 1)

    # --- the bound frame --- #
    def test_bound_frame_writes_solve_that_frame_alone(self):
        # a setter on the clip's nodes writes the frame they are bound to,
        # as it always has: the other frames are left exactly as they were
        clip, other = self._pair(orient=True)
        frame       = 3
        W           = np.array(other.frames.world_matrix)
        L           = np.array(other.frames.matrix)
        other.frame = frame
        Q           = np.array([node.quaternion for node in other])
        cases = {
            "clip.world_matrix": lambda c, o: setattr(c, "world_matrix", W[frame]),
            "clip.matrix": lambda c, o: setattr(c, "matrix", L[frame]),
            "clip.quaternion": lambda c, o: setattr(c, "quaternion", Q),
            "clip.match_matrix": lambda c, o: c.match_matrix(o),
            "clip.match_translate": lambda c, o: c.match_translate(o, z=False),
            "clip.match_rotate": lambda c, o: c.match_rotate(o),
            "view.match_rotate": lambda c, o: self._view(c, ["j2", "j5"]).match_rotate(
                self._view(o, ["j1", "j3"])
            ),
            "node.world_matrix": lambda c, o: setattr(c["j6"], "world_matrix", W[frame, 2]),
            "node.matrix": lambda c, o: setattr(c["j6"], "matrix", W[frame, 2]),
            "node.quaternion": lambda c, o: setattr(c["j6"], "quaternion", [0.5, 0.5, 0.5, 0.5]),
            "node.match_matrix": lambda c, o: c["j8"].match_matrix(o["j8"]),
            "node.match_translate": lambda c, o: c["j8"].match_translate(o["j1"], x=False),
            "node.match_rotate": lambda c, o: c["j8"].match_rotate([10.0, 20.0, 30.0]),
        }
        for label, write in cases.items():
            with self.subTest(write=label):
                fast = clip.copy()
                twin = other.copy()
                fast.frame, twin.frame = frame, frame
                before = {c: np.array(getattr(fast.frames, c)) for c in ("scale", "rotate", "translate")}
                write(fast, twin)

                pose = self._pose(clip, frame)
                write(pose, self._pose(other, frame))

                self.assertEqual(fast.frame, frame)
                fast_pose = self._pose(fast, frame)
                self.assertTrue(np.allclose(
                    fast_pose.world_matrix, pose.world_matrix, rtol=0.0, atol=1e-9
                ))
                for channel, values in before.items():
                    got = np.asarray(getattr(fast.frames, channel))
                    self.assertTrue(np.allclose(
                        got[frame], getattr(pose, channel), rtol=0.0, atol=1e-9
                    ), channel)
                    others = [f for f in range(clip.frame_count) if f != frame]
                    self.assertTrue(np.array_equal(got[others], values[others]), channel)

    def test_bound_frame_list_writes_solve_the_frame_at_once(self):
        # the single pose list setters solve node by node, every channel they
        # write clearing that node's branch: 3 kernel calls and a branch walk
        # per channel. on the head of a multi-frame clip the frame is solved
        # for every node at once, the blocks written and the clip rebound
        # once, so no node's branch is ever cleared on its own
        clip, other = self._pair(orient=True)
        clip.frame = other.frame = 2
        W          = np.array(other.frames.world_matrix)[2]
        L          = np.array(other.frames.matrix)[2]
        Q          = np.array([node.quaternion for node in other])
        writes = {
            "clip.world_matrix":    lambda c: setattr(c, "world_matrix", W),
            "clip.matrix":          lambda c: setattr(c, "matrix", L),
            "clip.quaternion":      lambda c: setattr(c, "quaternion", Q),
            "clip.match_matrix":    lambda c: c.match_matrix(other),
            "clip.match_translate": lambda c: c.match_translate(other),
            "clip.match_rotate":    lambda c: c.match_rotate(other),
            "view.match_rotate": lambda c: self._view(c, ["j2", "j5"]).match_rotate(
                np.zeros((2, 3))
            ),
        }
        for label, write in writes.items():
            fast       = clip.copy()
            fast.frame = 2
            with self.subTest(write=label), mock.patch.object(
                TransformData,
                "_reset_branch_world_matrices",
                autospec    = True,
                side_effect = TransformData._reset_branch_world_matrices,
            ) as resets:
                write(fast)
                self.assertEqual(resets.call_count, 0)

        # a node's own setter keeps the single pose route and lands as before
        clip["j4"].match_translate(other["j4"])
        self.assertTrue(np.allclose(
            clip["j4"].world_matrix[3, :3], other["j4"].world_matrix[3, :3], rtol=0.0, atol=1e-9
        ))

class TestClipScopedFrames(unittest.TestCase):
    """the frame axis narrowed to some nodes, or to a range of frames"""

    def setUp(self):
        super().setUp()
        self.clip                  = ClipData(chain(4), frames=6)
        block                      = np.arange(6 * 4 * 3, dtype=float).reshape(6, 4, 3)
        self.clip.frames.translate = block
        self.block                 = block

    # --- node scoping --- #
    def test_a_name_narrows_the_read_to_one_column(self):
        scoped = self.clip.frames["j2"]
        self.assertEqual(scoped.translate.shape, (6, 1, 3))
        self.assertTrue(np.array_equal(scoped.translate[:, 0], self.block[:, 2]))

    def test_a_name_narrows_the_write_to_one_column(self):
        self.clip.frames["j2"].translate = np.zeros((6, 1, 3))

        self.assertTrue(
            np.array_equal(self.clip.frames.translate[:, 2], np.zeros((6, 3)))
        )
        self.assertTrue(
            np.array_equal(self.clip.frames.translate[:, 1], self.block[:, 1])
        )

    def test_a_list_of_names_scopes_to_those_columns(self):
        scoped = self.clip.frames[["j0", "j3"]]
        self.assertEqual(scoped.translate.shape, (6, 2, 3))
        self.assertTrue(np.array_equal(scoped.translate[:, 1], self.block[:, 3]))

    def test_indices_and_names_are_interchangeable(self):
        by_name  = self.clip.frames[["j1", "j2"]].translate
        by_index = self.clip.frames[[1, 2]].translate
        self.assertTrue(np.array_equal(by_name, by_index))

    def test_match_scopes_the_whole_clip(self):
        rig = HierarchyData(
            [
                TransformData(name="L_arm", translate=[1.0, 0.0, 0.0]),
                TransformData(name="R_arm", translate=[2.0, 0.0, 0.0]),
                TransformData(name="spine", translate=[3.0, 0.0, 0.0]),
            ]
        )
        clip = ClipData(rig, frames=4)

        clip.frames.match("L_*", exact=False).rotate_y += 5.0

        self.assertTrue(np.array_equal(clip.frames.rotate[:, 0, 1], np.full(4, 5.0)))
        self.assertTrue(np.array_equal(clip.frames.rotate[:, 1, 1], np.zeros(4)))
        self.assertTrue(np.array_equal(clip.frames.rotate[:, 2, 1], np.zeros(4)))

    def test_a_scoped_write_reaches_the_loaded_pose(self):
        # the nodes are bound into the block, so the current frame has to see it
        self.clip.frame                  = 3
        self.clip.frames["j1"].translate = np.zeros((6, 1, 3))
        self.assertTrue(np.array_equal(self.clip[1].translate, np.zeros(3)))

    # --- components --- #
    def test_a_component_reads_one_axis_of_every_frame(self):
        self.assertTrue(
            np.array_equal(self.clip.frames.translate_y, self.block[..., 1])
        )

    def test_a_component_writes_one_axis_and_leaves_the_others(self):
        self.clip.frames.translate_y = np.zeros((6, 4))

        got = self.clip.frames.translate
        self.assertTrue(np.array_equal(got[..., 1], np.zeros((6, 4))))
        self.assertTrue(np.array_equal(got[..., 0], self.block[..., 0]))
        self.assertTrue(np.array_equal(got[..., 2], self.block[..., 2]))

    def test_a_scoped_component_augments_in_place(self):
        self.clip.frames["j0"].translate_x += 100.0

        got = self.clip.frames.translate
        self.assertTrue(np.array_equal(got[:, 0, 0], self.block[:, 0, 0] + 100.0))
        self.assertTrue(np.array_equal(got[:, 1, 0], self.block[:, 1, 0]))

    def test_one_node_takes_a_flat_list_of_frames(self):
        """the read is (F, 1); a flat list of F values means the same thing"""
        scoped = self.clip.frames["j2"]
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

        self.assertEqual(scoped.translate_y.shape, (6, 1))

        for value in (values, np.array(values)):
            with self.subTest(kind=type(value).__name__):
                scoped.translate_y = value

                got = self.clip.frames.translate
                self.assertTrue(np.array_equal(got[:, 2, 1], values))
                self.assertTrue(np.array_equal(got[:, 2, 0], self.block[:, 2, 0]))
                self.assertTrue(np.array_equal(got[:, 1], self.block[:, 1]))

        # the shape it reads back as still writes
        scoped.translate_y = np.zeros((6, 1))
        self.assertTrue(np.array_equal(self.clip.frames.translate[:, 2, 1], np.zeros(6)))

        # a flat list of the wrong length is still refused
        with self.assertRaises(ValueError):
            scoped.translate_y = [1.0, 2.0]

    def test_every_channel_has_three_components(self):
        for channel in ("scale", "rotate", "translate", "rotate_axis", "joint_orient"):
            for axis in "xyz":
                self.assertTrue(hasattr(self.clip.frames, f"{channel}_{axis}"))

    # --- frame ranges --- #
    def test_a_slice_cuts_a_new_clip(self):
        cut = self.clip.frames[2:5]

        self.assertIsInstance(cut, ClipData)
        self.assertEqual(cut.frame_count, 3)
        self.assertEqual(len(cut), 4)
        self.assertTrue(np.array_equal(cut.frames.translate, self.block[2:5]))

    def test_a_cut_carries_the_timebase_forward(self):
        clip = ClipData(chain(2), frames=10, start_frame=1001, fps=30.0)
        cut  = clip.frames[3:7]

        self.assertEqual(cut.start_frame, 1004)
        self.assertEqual(cut.end_frame,   1007)
        self.assertEqual(cut.fps,         30.0)

    def test_a_cut_does_not_write_back_to_its_parent(self):
        cut                  = self.clip.frames[0:3]
        cut.frames.translate = np.zeros((3, 4, 3))

        self.assertTrue(np.array_equal(self.clip.frames.translate, self.block))

    def test_a_cut_survives_a_realign(self):
        # the columns have to come across, or the first realign flattens it
        cut       = self.clip.frames[1:4]
        cut.frame = 2

        self.assertTrue(np.array_equal(cut.frames.translate, self.block[1:4]))

    def test_a_range_can_be_pasted_from_another_clip(self):
        source                  = ClipData(chain(4), frames=3)
        source.frames.translate = np.full((3, 4, 3), -1.0)

        self.clip.frames[1:4] = source

        got = self.clip.frames.translate
        self.assertTrue(np.array_equal(got[1:4], np.full((3, 4, 3), -1.0)))
        self.assertTrue(np.array_equal(got[0], self.block[0]))
        self.assertTrue(np.array_equal(got[4:], self.block[4:]))

    def test_a_range_can_be_filled_with_one_pose(self):
        pose           = chain(4)
        pose.translate = np.full((4, 3), 7.0)

        self.clip.frames[0:2] = pose

        got = self.clip.frames.translate
        self.assertTrue(np.array_equal(got[0], np.full((4, 3), 7.0)))
        self.assertTrue(np.array_equal(got[1], np.full((4, 3), 7.0)))
        self.assertTrue(np.array_equal(got[2], self.block[2]))

    def test_pasting_the_wrong_length_raises(self):
        source = ClipData(chain(4), frames=2)
        with self.assertRaises(ValueError):
            self.clip.frames[1:5] = source

    # --- guard rails --- #
    def test_a_scoped_axis_refuses_to_be_sliced(self):
        with self.assertRaises(TypeError):
            self.clip.frames["j1"][0:2]

    def test_a_scoped_axis_refuses_a_pose_write(self):
        with self.assertRaises(TypeError):
            self.clip.frames["j1"][0] = chain(4)

    def test_an_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            self.clip.frames["nope"]

    def test_a_scoped_read_is_still_read_only(self):
        view = self.clip.frames["j1"].translate
        with self.assertRaises(ValueError):
            view[0, 0, 0] = 1.0


class TestClipFramedReads(unittest.TestCase):
    """what a node builds from its channels, read for every frame at once

    Each read must equal the single pose TransformData read of the same
    name, taken frame by frame on a plain HierarchyData copy of that frame.
    """

    MATRICES = (
        "scale_matrix",
        "rotate_axis_matrix",
        "rotate_matrix",
        "joint_orient_matrix",
        "parent_scale_inverse",
        "parent_scale_inverse_matrix",
        "translate_matrix",
    )
    FRAMED = ("scale", "rotate", "translate", "rotateAxis", "jointOrient")

    def _rig(self, count=14, seed=7):
        # every feature a factor reads: rotate orders, segment scale
        # compensation both ways, a branching tree
        rng   = np.random.default_rng(seed)
        uuids = [hierarchy.generate_uuid() for _ in range(count)]
        names = [f"j{i}" for i in range(count)]
        nodes = []
        for i in range(count):
            parent = None
            if i:
                p      = int(rng.integers(0, i))
                parent = uuids[p]
            nodes.append(
                TransformData(
                    name=names[i],
                    uuid=uuids[i],
                    node_type="joint",
                    parent_uuid=parent,
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-180.0, 180.0, size=3),
                    scale=rng.uniform(0.5, 2.0, size=3),
                    rotate_axis=rng.uniform(-40.0, 40.0, size=3),
                    joint_orient=rng.uniform(-90.0, 90.0, size=3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(i % 2),
                )
            )
        return HierarchyData(nodes)

    def _animated(self, rig, frames=7, seed=11):
        # all five framed channels move, scale included, so a parent's scale
        # differs from frame to frame
        rng   = np.random.default_rng(seed)
        clip  = ClipData(rig, frames=frames, start_frame=1001, fps=30.0)
        shape = clip.frames.rotate.shape
        for channel in ("rotate", "translate", "rotate_axis", "joint_orient"):
            block = getattr(clip.frames, channel) + rng.normal(size=shape) * 10.0
            setattr(clip.frames, channel, block)
        clip.frames.scale = clip.frames.scale * rng.uniform(0.7, 1.3, size=shape)
        clip.frame        = 0
        return clip

    def _clips(self):
        yield "uuid", self._animated(self._rig())

    def _per_frame(self, clip, read):
        """the reference: ``read(node)`` on every node of a single pose copy
        of each frame, as a list of frames of lists of nodes"""
        out = []
        for frame in range(clip.frame_count):
            clip.frame = frame
            pose       = hierarchy.TransformList._owned_copy(clip)
            self.assertIs(type(pose), HierarchyData)
            out.append([read(node) for node in pose])
        return out

    def test_every_factor_matches_the_node_read_on_every_frame(self):
        for parented, clip in self._clips():
            for name in self.MATRICES:
                with self.subTest(parented_by=parented, read=name):
                    want = np.array(self._per_frame(clip, lambda node: getattr(node, name)))
                    got  = getattr(clip.frames, name)

                    self.assertEqual(got.shape, want.shape)
                    self.assertTrue(np.array_equal(got, want))

    def test_parent_matrix_matches_get_parent_matrix_on_every_frame(self):
        for parented, clip in self._clips():
            with self.subTest(parented_by=parented):
                want = np.array(self._per_frame(clip, lambda node: node.get_parent_matrix()))
                got  = clip.frames.parent_matrix

                self.assertEqual(got.shape, want.shape)
                self.assertTrue(np.allclose(got, want, rtol=0.0, atol=1e-9))
                self.assertTrue(np.array_equal(got, want))
                self.assertTrue(np.array_equal(clip.frames.get_parent_matrix(), got))

    def test_the_parent_scale_reads_follow_the_parents_animated_scale(self):
        # the per frame scale of a parent is what IS divides out, so a
        # compensating child must see it change, and a root must not
        clip   = self._animated(self._rig())
        parent = clip.frames.parent_scale_inverse
        matrix = clip.frames.parent_scale_inverse_matrix

        self.assertTrue(np.array_equal(parent[:, 0], np.ones((clip.frame_count, 3))))
        self.assertFalse(np.allclose(parent[0, 1:], parent[-1, 1:]))

        identity = np.broadcast_to(np.eye(4), (clip.frame_count, 4, 4))
        for i, node in enumerate(clip):
            diagonal = matrix[:, i, [0, 1, 2], [0, 1, 2]]
            if node.segment_scale_compensate:
                self.assertTrue(np.array_equal(diagonal, 1.0 / parent[:, i]))
            else:
                self.assertTrue(np.array_equal(matrix[:, i], identity))

    def test_a_parent_outside_the_clip_reads_as_a_root(self):
        # a clip of part of a rig: the first nodes' parents are left behind.
        # They join it as roots; a parent_uuid written back since names a
        # node the clip lacks, which get_parent_matrix answers with identity
        # and a scale of one
        rig   = self._rig()
        nodes = [node.copy() for node in rig][3:]
        lost  = {node.uuid: node.parent_uuid for node in nodes}
        clip  = self._animated(HierarchyData(nodes))
        orphans = [
            i for i, node in enumerate(clip) if node.parent_uuid is None and lost[node.uuid] is not None
        ]
        self.assertTrue(orphans)
        for i in orphans:
            clip[i].parent_uuid = lost[clip[i].uuid]
            self.assertIsInstance(clip[i].get_parent(), str)

        world    = clip.frames.parent_matrix
        scale    = clip.frames.parent_scale_inverse
        identity = np.broadcast_to(np.eye(4), (clip.frame_count, 4, 4))
        for i in orphans:
            self.assertTrue(np.array_equal(world[:, i], identity))
            self.assertTrue(np.array_equal(scale[:, i], np.ones((clip.frame_count, 3))))

        # and the node itself, scrubbed, says the same
        for frame in range(clip.frame_count):
            clip.frame = frame
            for i in orphans:
                self.assertTrue(np.array_equal(world[frame, i], clip[i].get_parent_matrix()))
                self.assertTrue(np.array_equal(scale[frame, i], clip[i].parent_scale_inverse))

    def test_a_scoped_axis_reads_its_own_columns(self):
        clip    = self._animated(self._rig())
        names   = ["j9", "j0", "j4"]
        columns = [9, 0, 4]
        for name in self.MATRICES + ("parent_matrix",):
            with self.subTest(read=name):
                full   = getattr(clip.frames, name)
                scoped = getattr(clip.frames[names], name)
                self.assertTrue(np.array_equal(scoped, full[:, columns]))

        one = clip.frames["j5"].rotate_matrix
        self.assertEqual(one.shape, (clip.frame_count, 1, 4, 4))

        full = clip.frames.to_attributes()
        self.assertEqual(clip.frames[names].to_attributes(), [full[i] for i in columns])

    def test_to_attributes_gathers_every_frame(self):
        for parented, clip in self._clips():
            with self.subTest(parented_by=parented):
                got  = clip.frames.to_attributes()
                want = self._per_frame(clip, lambda node: node.to_attributes())

                self.assertEqual(len(got), len(clip))
                for i, attributes in enumerate(got):
                    # same keys, in the same order, as one frame's dict
                    self.assertEqual(list(attributes), list(want[0][i]))
                    for key, value in attributes.items():
                        frames = [want[f][i][key] for f in range(clip.frame_count)]
                        if key in self.FRAMED:
                            self.assertEqual(value, frames)
                        else:
                            self.assertTrue(all(value == v for v in frames))

    def test_to_attributes_takes_a_mapping(self):
        clip    = self._animated(self._rig())
        mapping = {"_rotate": "r", "_rotate_order": "ro"}

        got  = clip.frames.to_attributes(mapping)
        want = self._per_frame(clip, lambda node: node.to_attributes(mapping))
        for i, attributes in enumerate(got):
            self.assertEqual(set(attributes),  {"r", "ro"})
            self.assertEqual(attributes["r"],  [want[f][i]["r"] for f in range(clip.frame_count)])
            self.assertEqual(attributes["ro"], want[0][i]["ro"])

    def test_the_reads_follow_an_edit_and_change_nothing(self):
        clip       = self._animated(self._rig())
        clip.frame = 3
        blocks     = {c: b.copy() for c, b in clip._blocks.items()}

        before = clip.frames.parent_matrix.copy()
        for name in self.MATRICES:
            getattr(clip.frames, name)
        clip.frames.to_attributes()

        # nothing scrubbed, nothing written
        self.assertEqual(clip.frame, 3)
        for channel, block in clip._blocks.items():
            self.assertTrue(np.array_equal(block, blocks[channel]))

        # a write through the bound frame is seen on that frame only
        clip[0].scale = [3.0, 3.0, 3.0]
        scale         = clip.frames.scale_matrix
        self.assertTrue(np.array_equal(scale[3, 0], np.diag([3.0, 3.0, 3.0, 1.0])))
        self.assertTrue(
            np.array_equal(scale[2, 0], np.diag(np.append(blocks["_scale"][2, 0], 1.0)))
        )

        after = clip.frames.parent_matrix
        child = [i for i, node in enumerate(clip) if node.get_parent() is clip[0]]
        self.assertTrue(child)
        self.assertFalse(np.allclose(after[3, child], before[3, child]))
        self.assertTrue(np.array_equal(after[2], before[2]))

    def test_the_reads_cannot_be_written(self):
        clip = self._animated(self._rig())
        for name in self.MATRICES + ("parent_matrix",):
            with self.subTest(read=name):
                value = getattr(clip.frames, name)
                with self.assertRaises(AttributeError):
                    setattr(clip.frames, name, value)

    def test_a_write_the_axis_does_not_hold_raises(self):
        # each clip.frames is a new axis, so a write it does not hold used to
        # land on that throwaway object and vanish without a word
        clip   = self._animated(self._rig())
        blocks = {c: b.copy() for c, b in clip._blocks.items()}
        orders = list(clip.rotate_order)

        axes = {
            "whole clip": clip.frames,
            "one node":   clip.frames["j2"],
            "matched":    clip.frames.match("j1*", exact=False),
        }
        for label, axis in axes.items():
            for name, value in (
                ("rotate_order", 0),
                ("segment_scale_compensate", True),
                ("radius", 2.0),
                ("rotate_xyz", np.zeros(3)),
            ):
                with self.subTest(axis=label, write=name):
                    with self.assertRaises(AttributeError):
                        setattr(axis, name, value)

        self.assertEqual(list(clip.rotate_order), orders)
        for channel, block in clip._blocks.items():
            self.assertTrue(np.array_equal(block, blocks[channel]))

        # the channels it does hold still write
        clip.frames["j2"].rotate_y += 5.0
        self.assertTrue(
            np.array_equal(clip.frames.rotate[:, 2, 1], blocks["_rotate"][:, 2, 1] + 5.0)
        )

    # every read of an axis, by name; to_attributes as its framed channels
    READS = MATRICES + (
        "scale",
        "rotate",
        "translate",
        "rotate_axis",
        "joint_orient",
        "rotate_y",
        "matrix",
        "world_matrix",
        "parent_matrix",
    )

    REORDERS = {
        "sort":             lambda clip: clip.sort(key=lambda node: -int(node.name[1:])),
        "reverse":          lambda clip: clip.reverse(),
        "set_parent local": lambda clip: clip["j2"].set_parent(None, world_space=False),
        "a node added":     lambda clip: clip.insert(0, TransformData("extra", translate=(1, 2, 3))),
    }

    def _reads(self, axis) -> dict:
        out = {name: np.array(getattr(axis, name)) for name in self.READS}
        out["to_attributes"] = [
            [node[key] for key in self.FRAMED] for node in axis.to_attributes()
        ]
        return out

    def test_an_axis_held_across_a_reorder_reads_its_own_nodes(self):
        # an axis held across an edit that reordered the nodes read block
        # columns in the old order against rotate orders, compensation and
        # parents in the new one: up to 7.9 off in world_matrix after a sort
        for label, reorder in self.REORDERS.items():
            for scope in (None, ["j5", "j9", "j2"]):
                with self.subTest(label, scope=scope):
                    clip = self._animated(self._rig())
                    held = clip.frames if scope is None else clip.frames[scope]
                    reorder(clip)

                    # read before anything else lines the clip up again
                    got   = self._reads(held)
                    fresh = clip.frames if scope is None else clip.frames[scope]
                    want  = self._reads(fresh)
                    for name in want:
                        self.assertTrue(
                            np.array_equal(got[name], want[name]), f"{label}: {name}"
                        )
                    if scope is not None:
                        self.assertEqual([n.name for n in held._nodes()], scope)

    def test_a_held_scoped_axis_writes_its_own_nodes(self):
        clip = self._animated(self._rig())
        held = clip.frames["j5"]
        clip.sort(key=lambda node: -int(node.name[1:]))
        before = np.array(clip.frames.rotate)

        held.rotate_y = 7.0

        after = np.array(clip.frames.rotate)
        j5    = clip.index("j5")
        self.assertTrue(np.array_equal(after[:, j5, 1], np.full(clip.frame_count, 7.0)))
        others = [i for i in range(len(clip)) if i != j5]
        self.assertTrue(np.array_equal(after[:, others], before[:, others]))

    def test_a_node_taken_out_of_the_clip_is_refused(self):
        clip = self._animated(self._rig())
        held = clip.frames[["j3", "j4"]]
        clip.remove(clip["j4"])
        for name in ("rotate", "world_matrix", "to_attributes"):
            with self.subTest(read=name):
                with self.assertRaisesRegex(ValueError, "no longer holds"):
                    value = getattr(held, name)
                    if callable(value):
                        value()
        with self.assertRaisesRegex(ValueError, "no longer holds"):
            held.rotate_x = 1.0

    def test_a_clip_with_no_frames_has_nothing_to_read(self):
        # the reads raised a bare KeyError naming a channel, to_attributes
        # gave back the pose the nodes hold
        clip = ClipData(self._rig())
        self.assertEqual(clip.frame_count, 0)
        for axis in (clip.frames, clip.frames[["j1", "j2"]]):
            for name in self.READS + ("to_attributes",):
                with self.subTest(read=name, scoped=axis._uuids is not None):
                    with self.assertRaisesRegex(ValueError, "holds no frames"):
                        value = getattr(axis, name)
                        if callable(value):
                            value()
            with self.assertRaisesRegex(ValueError, "holds no frames"):
                axis.rotate = np.zeros((0, len(clip), 3))
        self.assertEqual(len(clip.frames), 0)
        self.assertEqual(clip.frames.times.shape, (0,))


class TestClipPoseWrites(unittest.TestCase):
    """from_poses, a pose filled down a range and a clip pasted into one
    write every frame at once

    Each must agree with a reference read one frame at a time through the
    single pose HierarchyData / TransformData API: channels exactly, since a
    write copies them, and world matrices to 1e-9. Rotate wanders well past
    180 degrees, so a write that refiltered its keys would show.
    """

    CHANNELS = ("scale", "rotate", "translate", "rotate_axis", "joint_orient")

    def _rig(self, count=9, seed=3, prefix="j"):
        # branching, parented by uuid as files are, with every factor the
        # world matrix composes: rotate_order, joint_orient, rotate_axis,
        # non unit scale and mixed segment scale compensation
        rng   = np.random.default_rng(seed)
        nodes = []
        for i in range(count):
            nodes.append(
                TransformData(
                    name=f"{prefix}{i}",
                    node_type="joint",
                    parent_uuid=nodes[int(rng.integers(0, i))].uuid if i else None,
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-90.0, 90.0, size=3),
                    scale=rng.uniform(0.5, 2.0, size=3),
                    rotate_axis=rng.uniform(-30.0, 30.0, size=3),
                    joint_orient=rng.uniform(-60.0, 60.0, size=3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
            )
        return HierarchyData(nodes)

    def _animated(self, rig, frames=7, seed=5):
        rng   = np.random.default_rng(seed)
        clip  = ClipData(rig, frames=frames, start_frame=1001, fps=30.0)
        shape = clip.frames.rotate.shape
        walk  = np.cumsum(rng.normal(size=shape) * 90.0, axis=0)
        clip.frames.rotate       = clip.frames.rotate + walk
        clip.frames.translate    = clip.frames.translate + rng.normal(size=shape)
        clip.frames.scale        = clip.frames.scale * rng.uniform(0.8, 1.25, size=shape)
        clip.frames.rotate_axis  = clip.frames.rotate_axis + rng.normal(size=shape) * 9
        clip.frames.joint_orient = clip.frames.joint_orient + rng.normal(size=shape) * 9
        clip.frame               = 0
        return clip

    # --- the reference: one pose at a time --- #
    def _read(self, pose):
        """one pose through the single pose API"""
        out                 = {c: np.array([getattr(node, c) for node in pose]) for c in self.CHANNELS}
        out["world_matrix"] = np.array(pose.world_matrix)
        return out

    def _pose_at(self, clip, frame):
        """frame ``frame`` of ``clip`` as a plain hierarchy of its own"""
        clip.frame = frame
        return TransformList._owned_copy(clip)

    def _frames(self, clip):
        """every frame of ``clip`` read one at a time, scrubbing

        Back on the frame it was bound to, that frame is read once more, so
        its nodes hold cached world matrices that a write has to clear.
        """
        bound = clip.frame
        reads = []
        for frame in range(clip.frame_count):
            clip.frame = frame
            reads.append(self._read(clip))
        clip.frame = bound
        clip.world_matrix
        return reads

    def _assert_frames(self, clip, expected):
        """``clip`` holds ``expected``, one single pose read per frame, and
        the frame it is bound to reads the same through its nodes"""
        self.assertEqual(clip.frame_count, len(expected))
        for channel in self.CHANNELS:
            got  = getattr(clip.frames, channel)
            want = np.array([pose[channel] for pose in expected])
            self.assertTrue(np.array_equal(got, want), channel)

        want = np.array([pose["world_matrix"] for pose in expected])
        self.assertLessEqual(np.abs(clip.frames.world_matrix - want).max(), 1e-9)

        # the bound nodes must not keep a world matrix cached before the write
        for key, values in self._read(clip).items():
            error = np.abs(values - expected[clip.frame][key]).max()
            self.assertLessEqual(error, 1e-9, key)

    def _target(self, frame=4):
        """an animated clip bound to ``frame``"""
        clip       = self._animated(self._rig(), seed=5)
        clip.frame = frame
        return clip

    @contextlib.contextmanager
    def _binds(self):
        """counts ClipData._bind calls: each is a pass over every node"""
        calls    = []
        original = ClipData._bind

        def counting(clip, frame):
            calls.append(frame)
            return original(clip, frame)

        with mock.patch.object(ClipData, "_bind", counting):
            yield calls

    @contextlib.contextmanager
    def _channel_reads(self, nodes):
        """counts reads of ``nodes``' framed channel fields"""
        watched = {id(node) for node in nodes}
        fields  = set(ClipData.FRAMED_CHANNELS)
        reads   = []

        def getattribute(node, name):
            if name in fields and id(node) in watched:
                reads.append(name)
            return object.__getattribute__(node, name)

        with mock.patch.object(TransformData, "__getattribute__", getattribute):
            yield reads

    # --- from_poses --- #
    def test_from_poses_matches_the_poses(self):
        source = self._animated(self._rig(), seed=8)
        poses  = [self._pose_at(source, f) for f in range(source.frame_count)]

        clip   = ClipData.from_poses(poses, start_frame=1001, fps=30.0)

        self._assert_frames(clip, [self._read(pose) for pose in poses])
        self.assertEqual(clip.frame, 0)
        self.assertEqual(clip.name, poses[0].name)
        self.assertTrue(np.array_equal(clip.rotate_order, poses[0].rotate_order))
        self.assertEqual((clip.start_frame, clip.fps), (1001, 30.0))

    def test_from_poses_binds_the_nodes_once(self):
        # not once per pose: every pose goes straight into the blocks
        source = self._animated(self._rig(), seed=8)
        poses  = [self._pose_at(source, f) for f in range(source.frame_count)]

        with self._binds() as calls:
            ClipData.from_poses(poses)

        self.assertEqual(calls, [0])

    def test_from_poses_leaves_the_poses_alone(self):
        source = self._animated(self._rig(), seed=8)
        poses  = [self._pose_at(source, f) for f in range(source.frame_count)]
        before = [self._read(pose) for pose in poses]

        clip                  = ClipData.from_poses(poses)
        clip.frames.translate = np.zeros(clip.frames.translate.shape)

        for pose, want in zip(poses, before):
            for key, values in self._read(pose).items():
                self.assertTrue(np.array_equal(values, want[key]), key)

    def test_from_poses_refuses_a_pose_of_another_size(self):
        poses = [self._rig(), self._rig(count=8)]
        with self.assertRaises(ValueError):
            ClipData.from_poses(poses)

    # --- one pose: a frame, or filled down a range --- #
    def test_one_pose_write_matches_the_pose(self):
        pose = self._pose_at(self._animated(self._rig(), seed=8), 3)

        for frame in (0, 4, 6, -2):
            with self.subTest(frame=frame):
                clip     = self._target()
                expected = self._frames(clip)

                clip.frames[frame] = pose

                expected[frame] = self._read(pose)
                self._assert_frames(clip, expected)

    def test_a_pose_fills_any_range(self):
        pose = self._pose_at(self._animated(self._rig(), seed=8), 3)
        read = self._read(pose)

        for span in (
            slice(None),
            slice(2, 5),
            slice(1, None, 2),
            slice(None, None, -3),
            slice(5, 1, -1),
            slice(3, 3),
        ):
            with self.subTest(span=span):
                clip     = self._target()
                expected = self._frames(clip)

                clip.frames[span] = pose

                for frame in range(*span.indices(clip.frame_count)):
                    expected[frame] = read
                self._assert_frames(clip, expected)

    def test_any_sequence_of_nodes_is_a_pose(self):
        # a hierarchy, a selection view of one and a plain list of its nodes
        pose = self._pose_at(self._animated(self._rig(), seed=8), 3)
        read = self._read(pose)

        for given in (pose, pose[:], list(pose)):
            with self.subTest(kind=type(given).__name__):
                clip     = self._target()
                expected = self._frames(clip)

                clip.frames[1:4] = given

                expected[1:4] = [read] * 3
                self._assert_frames(clip, expected)

    def test_a_fill_reads_the_pose_once(self):
        # not once per frame: 5 framed channels of each node, read one time
        clip = self._target()
        pose = self._pose_at(self._animated(self._rig(), seed=8), 3)

        with self._channel_reads(pose.list) as reads:
            clip.frames[:] = pose

        self.assertEqual(len(reads), 5 * len(pose))

    def test_an_axis_held_across_a_reorder_writes_by_the_current_order(self):
        # set_parent sorts the new parent ahead of its child after the axis
        # was taken; the pose still pairs with the nodes as they are now
        rig  = HierarchyData([TransformData(name=n) for n in ("a", "b", "root")])
        clip = ClipData(rig, frames=3)
        axis = clip.frames
        clip["a"].set_parent("root", world_space=False)

        names = clip.name
        pose = HierarchyData(
            [
                TransformData(name=name, translate=[i + 1.0, 0.0, 0.0])
                for i, name in enumerate(names)
            ]
        )
        axis[0:3] = pose

        self.assertEqual(names, ["b", "root", "a"])
        want = [1.0, 2.0, 3.0]
        self.assertTrue(np.array_equal(clip.frames.translate[..., 0], [want] * 3))
        self.assertTrue(np.array_equal(clip.translate[:, 0], want))

    def test_a_pose_of_another_size_is_refused(self):
        clip = self._target()
        for span in (2, slice(1, 4)):
            with self.subTest(span=span):
                with self.assertRaises(ValueError):
                    clip.frames[span] = self._rig(count=8)

    # --- a clip pasted into a range --- #
    def test_a_paste_matches_the_source_read_frame_by_frame(self):
        for span in (slice(None), slice(2, 5), slice(None, None, -2), slice(6, 1, -2)):
            with self.subTest(span=span):
                clip     = self._target()
                frames   = range(*span.indices(clip.frame_count))
                source   = self._animated(self._rig(), frames=len(frames), seed=13)
                expected = self._frames(clip)
                pasted   = self._frames(source)

                clip.frames[span] = source

                for i, frame in enumerate(frames):
                    expected[frame] = pasted[i]
                self._assert_frames(clip, expected)

    def test_a_paste_leaves_the_source_alone(self):
        # the source is read whole, never scrubbed: its frame stays put
        clip         = self._target()
        source       = self._animated(self._rig(), frames=3, seed=13)
        source.frame = 1
        before       = self._frames(source)

        clip.frames[2:5] = source

        self.assertEqual(source.frame, 1)
        self._assert_frames(source, before)

    def test_a_paste_binds_the_nodes_once(self):
        clip   = self._target()
        source = self._animated(self._rig(), seed=13)

        with self._binds() as calls:
            clip.frames[:] = source

        self.assertEqual(calls, [4])

    def test_a_clip_pasted_into_itself_reversed_plays_backwards(self):
        # every frame is read before any is written over
        clip     = self._target()
        expected = self._frames(clip)

        clip.frames[::-1] = clip

        self._assert_frames(clip, expected[::-1])

    def test_a_paste_pairs_nodes_by_position(self):
        # as a single pose does: the source's names play no part
        clip   = self._target()
        source = self._animated(self._rig(seed=21, prefix="other"), frames=3, seed=13)
        pasted = self._frames(source)

        clip.frames[0:3] = source

        want = [pose["rotate"] for pose in pasted]
        self.assertTrue(np.array_equal(clip.frames.rotate[0:3], want))
        self.assertEqual(clip.name[0], "j0")

    def test_a_paste_of_another_node_count_is_refused(self):
        clip   = self._target()
        source = self._animated(self._rig(count=8), frames=3, seed=13)
        with self.assertRaises(ValueError):
            clip.frames[0:3] = source

    def test_a_source_whose_nodes_were_reordered_pastes_by_its_current_order(self):
        # a sort or a set_parent reorders a clip's nodes and leaves its block
        # columns behind until something lines them up: the paste has to do
        # that first, or node i takes the curves of whichever node had
        # column i, 96 to 106 degrees off here
        reorders = {
            "sort":             lambda c: c.sort(reverse=True),
            "set_parent local": lambda c: c[c.index("j1")].set_parent(None, world_space=False),
        }
        for label, reorder in reorders.items():
            with self.subTest(label):
                clip   = self._target()
                source = self._animated(self._rig(), frames=clip.frame_count, seed=13)
                reorder(source)
                self.assertNotEqual(source._columns, [node.uuid for node in source.list])

                # each node's curves, read off the raw blocks by its uuid
                column = {uuid: j for j, uuid in enumerate(source._columns)}
                cols   = [column[node.uuid] for node in source.list]
                want   = {c: source._blocks["_" + c][:, cols] for c in self.CHANNELS}

                clip.frames[:] = source

                for channel in self.CHANNELS:
                    got = getattr(clip.frames, channel)
                    self.assertTrue(np.array_equal(got, want[channel]), channel)


class TestClipSaveFbx(unittest.TestCase):
    """the fbx exporter writes one pose and no animation stack

    Only the tests that actually reach the exporter are skipped without the
    sdk, rather than the whole class: the refusal cases raise before the sdk
    is ever touched, so they still say something useful on a machine that
    does not have it.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.clip               = ClipData(uuid_chain(3), frames=10, start_frame=1001, fps=24.0)
        rotate                  = np.array(self.clip.frames.rotate)
        rotate[:, 1, 2]         = np.arange(10) * 10.0
        self.clip.frames.rotate = rotate
        self.clip.frame         = 3

    def path(self, name="clip.fbx"):
        return os.path.join(self.tmp.name, name)

    def test_a_multi_frame_clip_refuses(self):
        with self.assertRaises(ValueError) as caught:
            self.clip.save_fbx(self.path())

        self.assertIn("10 frames", str(caught.exception))

    def test_the_message_names_the_loaded_frame(self):
        with self.assertRaises(ValueError) as caught:
            self.clip.save_fbx(self.path())

        self.assertIn("frames[3:4]", str(caught.exception))

    @unittest.skipIf(FBX is None, "fbx sdk is not installed")
    def test_the_escape_hatch_in_the_message_works(self):
        """frames[f] scrubs and hands back the clip, so it has to be a slice"""
        self.assertIs(self.clip.frames[3], self.clip)
        self.assertEqual(self.clip.frames[3].frame_count, 10)

        cut = self.clip.frames[3:4]

        self.assertEqual(cut.frame_count, 1)
        cut.save_fbx(self.path("cut.fbx"))
        self.assertTrue(os.path.exists(self.path("cut.fbx")))

    @unittest.skipIf(FBX is None, "fbx sdk is not installed")
    def test_a_single_frame_clip_still_exports(self):
        clip = ClipData(uuid_chain(3), frames=1)

        clip.save_fbx(self.path("one.fbx"))

        self.assertTrue(os.path.exists(self.path("one.fbx")))

    @unittest.skipIf(FBX is None, "fbx sdk is not installed")
    def test_a_frameless_clip_still_exports(self):
        """frame_count is 0 there, so the guard has to be > 1 and not != 1"""
        clip = ClipData(uuid_chain(3))
        self.assertEqual(clip.frame_count, 0)

        clip.save_fbx(self.path("none.fbx"))

        self.assertTrue(os.path.exists(self.path("none.fbx")))

    def test_save_still_keeps_every_frame(self):
        """the whole clip round trips through save/load, only fbx is lossy"""
        path = os.path.join(self.tmp.name, "clip.json")
        self.clip.save(path)

        loaded = ClipData.load(path)

        self.assertEqual(loaded.frame_count, 10)
        self.assertTrue(
            np.allclose(loaded.frames.rotate, np.array(self.clip.frames.rotate))
        )


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestClipLoadGlb(unittest.TestCase):
    """gltf keys every property on its own timeline, so the samplers have to
    be resampled onto a frame grid rather than read off frame by frame
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(
        self,
        name          = "anim.glb",
        *,
        times         = None,
        translation   = None,
        rotation      = None,
        scale         = None,
        interpolation = "LINEAR",
        rest_rotation = None,
        nodes         = ("joint1",),
        target        = 0,
    ):
        """writes a glb of one animated node, returning the path"""
        blob = b""
        views, accessors, samplers, channels = [], [], [], []

        def accessor(array, kind):
            nonlocal blob
            data = np.asarray(array, dtype=np.float32)
            views.append(
                BufferView(buffer=0, byteOffset=len(blob), byteLength=data.nbytes)
            )
            blob += data.tobytes()
            accessors.append(
                Accessor(
                    bufferView    = len(views) - 1,
                    componentType = FLOAT,
                    count         = len(array),
                    type          = kind,
                    min           = [float(np.min(data))] if kind == "SCALAR" else None,
                    max           = [float(np.max(data))] if kind == "SCALAR" else None,
                )
            )
            return len(accessors) - 1

        if times is not None:
            keys = accessor(np.asarray(times, dtype=np.float32).ravel(), "SCALAR")
            for values, kind, path in (
                (translation, "VEC3", "translation"),
                (rotation, "VEC4", "rotation"),
                (scale, "VEC3", "scale"),
            ):
                if values is None:
                    continue
                samplers.append(
                    AnimationSampler(
                        input         = keys,
                        output        = accessor(values, kind),
                        interpolation = interpolation,
                    )
                )
                channels.append(
                    AnimationChannel(
                        sampler = len(samplers) - 1,
                        target  = AnimationChannelTarget(node=target, path=path),
                    )
                )

        gltf       = GLTF2()
        gltf.scene = 0
        gltf.nodes = [
            Node(name=node, translation=[0.0, float(index + 1), 0.0])
            for index, node in enumerate(nodes)
        ]
        if rest_rotation is not None:
            gltf.nodes[target].rotation = list(rest_rotation)
        gltf.scenes      = [Scene(nodes=list(range(len(nodes))))]
        gltf.buffers     = [Buffer(byteLength=len(blob))]
        gltf.bufferViews = views
        gltf.accessors   = accessors
        if channels:
            gltf.animations = [Animation(samplers=samplers, channels=channels)]
        gltf.set_binary_blob(blob)

        path = os.path.join(self.tmp.name, name)
        gltf.save(path)
        return path

    def test_a_linear_channel_lands_on_the_frame_grid(self):
        path = self.build(
            times=[0.0, 1.0], translation=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        clip = ClipData.load_glb(path, scale_factor=1.0, fps=24.0)

        # a second at 24fps is 25 frames, both ends included
        self.assertEqual(clip.frame_count, 25)
        self.assertTrue(
            np.allclose(
                np.array(clip.frames.translate)[:, 0, 0], np.linspace(0.0, 1.0, 25)
            )
        )

    def test_a_step_channel_holds_and_keeps_its_last_key(self):
        """the final key owns the instant it lands on, not the segment ahead"""
        path = self.build(
            times         = [0.0, 0.5, 1.0],
            translation   = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            interpolation = "STEP",
        )

        clip = ClipData.load_glb(path, scale_factor=1.0, fps=10.0)

        tx   = np.array(clip.frames.translate)[:, 0, 0]
        self.assertTrue(np.allclose(tx, [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 9]))

    def test_a_cubic_channel_scales_its_tangents_by_the_span(self):
        """unequal key spacing, so a missing dt would show up in the middle"""
        cubic = []
        for value, tangent in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0)):
            cubic += [[tangent, 0, 0], [value, 0, 0], [tangent, 0, 0]]

        path = self.build(
            times=[0.0, 1.0, 3.0], translation=cubic, interpolation="CUBICSPLINE"
        )

        clip = ClipData.load_glb(path, scale_factor=1.0, fps=4.0)

        def hermite(p0, m0, p1, m1, u):
            uu, uuu = u * u, u * u * u
            return (
                (2 * uuu - 3 * uu + 1) * p0
                + (uuu - 2 * uu + u) * m0
                + (-2 * uuu + 3 * uu) * p1
                + (uuu - uu) * m1
            )

        tx   = np.array(clip.frames.translate)[:, 0, 0]
        want = []
        for time in np.arange(len(tx)) / 4.0:
            if time <= 1.0:
                want.append(hermite(0.0, 1.0, 1.0, 0.0, time))
            else:
                want.append(hermite(1.0, 0.0, 0.0, -2.0, (time - 1.0) / 2.0))

        self.assertTrue(np.allclose(tx, want, atol=1e-5))

    def test_rotation_slerps_rather_than_lerping_its_components(self):
        """every frame, not just the ends and the middle: a normalised lerp
        agrees with a slerp at exactly those three, so sampling only them
        would pass either way
        """
        first = euler_to_quaternion(np.radians([[0.0, 0.0, 0.0]]), 0)[0]
        last  = euler_to_quaternion(np.radians([[0.0, 0.0, 150.0]]), 0)[0]
        path  = self.build(times=[0.0, 1.0], rotation=[list(first), list(last)])

        clip  = ClipData.load_glb(path, scale_factor=1.0, fps=8.0)

        for frame in range(9):
            clip.frame = frame
            want = quaternion_slerp(
                first.reshape(1, 4), last.reshape(1, 4), frame / 8.0
            )
            # sign is free in a quaternion, so compare the axis and not the sign
            dot = abs(float(np.dot(clip["joint1"].quaternion.ravel(), want.ravel())))
            self.assertAlmostEqual(dot, 1.0, places=6)

    def test_a_gimbal_sweep_comes_out_continuous(self):
        """each frame picks its own euler branch, so the curve needs filtering"""
        rotation = [
            list(euler_to_quaternion(np.radians([[0.0, angle, 0.0]]), 0)[0])
            for angle in np.linspace(-170.0, 170.0, 40)
        ]
        path   = self.build(times=list(np.linspace(0.0, 1.0, 40)), rotation=rotation)

        clip   = ClipData.load_glb(path, scale_factor=1.0, fps=39.0)

        rotate = np.array(clip.frames.rotate)[:, 0, :]
        self.assertLess(np.abs(np.diff(rotate, axis=0)).max(), 30.0)

    def test_a_file_with_no_animation_gives_one_frame(self):
        """a rig that does not move is an accurate reading, not an error"""
        path = self.build(name="still.glb")

        clip = ClipData.load_glb(path, scale_factor=1.0)

        self.assertEqual(clip.frame_count, 1)
        self.assertEqual(np.array(clip.frames.rotate).shape, (1, 1, 3))

    def test_the_dispatcher_hands_back_a_clip(self):
        path = self.build(
            name        = "disp.glb",
            times       = [0.0, 1.0],
            translation = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )

        self.assertIsInstance(ClipData.load(path), ClipData)
        # the base class still reads the same file as a single pose
        self.assertNotIsInstance(HierarchyData.load(path), ClipData)

    def test_scale_factor_reaches_translation(self):
        path = self.build(
            times=[0.0, 1.0], translation=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        clip = ClipData.load_glb(path, scale_factor=100.0, fps=2.0)

        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[-1, 0, 0]), 100.0, places=4
        )

    def test_an_unanimated_node_keeps_its_rest_pose(self):
        """the blocks are seeded from the rig, so a node with no channel holds"""
        path = self.build(
            nodes       = ("joint1", "joint2"),
            times       = [0.0, 1.0],
            translation = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            target      = 0,
        )

        clip  = ClipData.load_glb(path, scale_factor=1.0, fps=4.0)

        rest  = HierarchyData.load_glb(path, scale_factor=1.0)["joint2"].translate
        other = np.array(clip.frames["joint2"].translate)
        self.assertTrue(np.allclose(other, rest))
        self.assertTrue(np.allclose(other[0], other[-1]))

    def test_a_duplicated_node_name_is_matched_by_position(self):
        """a rig exported once per mesh carries the whole skeleton twice, so
        the channel's node index is the only thing saying which copy moves

        Aimed at the second of the two, which matching on name could never
        get right.
        """
        path = self.build(
            nodes       = ("joint1", "joint1"),
            times       = [0.0, 1.0],
            translation = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            target      = 1,
        )

        clip = ClipData.load_glb(path, scale_factor=1.0, fps=4.0)

        self.assertEqual(len(clip), 2)
        translate = np.array(clip.frames.translate)
        self.assertAlmostEqual(float(translate[-1, 1, 0]), 5.0, places=4)
        self.assertTrue(np.allclose(translate[:, 0, 0], translate[0, 0, 0]))

    def test_start_frame_and_fps_are_carried(self):
        path = self.build(
            times=[0.0, 1.0], translation=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        clip = ClipData.load_glb(path, scale_factor=1.0, fps=30.0, start_frame=1001)

        self.assertEqual(clip.start_frame, 1001)
        self.assertEqual(clip.fps,         30.0)
        self.assertEqual(clip.frame_count, 31)

    def test_a_negative_animation_index_counts_back(self):
        path = self.build(
            times=[0.0, 1.0], translation=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        clip = ClipData.load_glb(path, scale_factor=1.0, fps=2.0, animation=-1)

        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[-1, 0, 0]), 1.0, places=5
        )

    def test_bad_arguments_refuse(self):
        path = self.build(
            times=[0.0, 1.0], translation=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        for kwargs in ({"fps": 0.0}, {"fps": -24.0}):
            with self.assertRaises(ValueError):
                ClipData.load_glb(path, scale_factor=1.0, **kwargs)

        with self.assertRaises(IndexError):
            ClipData.load_glb(path, scale_factor=1.0, animation=7)


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestClipLoadGlbNodeOrder(unittest.TestCase):
    """load_glb hands back a rig ordered parent first, not in file order

    Channels address nodes by index and the rig only ever drops nodes, so the
    two are paired by walking them in step. That walk has to follow the order
    the rig was built in, or every channel lands on the wrong joint.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, target):
        """a leaf first node array, animating whichever node is asked for"""
        times  = np.array([0.0, 1.0], dtype=np.float32)
        values = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32)
        blob   = times.tobytes() + values.tobytes()

        gltf       = GLTF2()
        gltf.scene = 0
        gltf.nodes = [
            Node(name="leaf", translation=[0.0, 1.0, 0.0]),
            Node(name="mid", translation=[0.0, 2.0, 0.0], children=[0]),
            Node(name="root", translation=[0.0, 3.0, 0.0], children=[1]),
        ]
        gltf.scenes  = [Scene(nodes=[2])]
        gltf.buffers = [Buffer(byteLength=len(blob))]
        gltf.bufferViews = [
            BufferView(buffer=0, byteOffset=0, byteLength=times.nbytes),
            BufferView(buffer=0, byteOffset=times.nbytes, byteLength=values.nbytes),
        ]
        gltf.accessors = [
            Accessor(
                bufferView    = 0,
                componentType = FLOAT,
                count         = 2,
                type          = "SCALAR",
                min           = [0.0],
                max           = [1.0],
            ),
            Accessor(bufferView=1, componentType=FLOAT, count=2, type="VEC3"),
        ]
        gltf.animations = [
            Animation(
                samplers=[AnimationSampler(input=0, output=1, interpolation="LINEAR")],
                channels=[
                    AnimationChannel(
                        sampler = 0,
                        target  = AnimationChannelTarget(node=target, path="translation"),
                    )
                ],
            )
        ]
        gltf.set_binary_blob(blob)

        path = os.path.join(self.tmp.name, "ordered.glb")
        gltf.save(path)
        return path

    def test_the_clip_holds_the_rig_parent_first(self):
        clip = ClipData.load_glb(self.build(target=2), scale_factor=1.0, fps=4.0)

        self.assertEqual([str(node.name) for node in clip], ["root", "mid", "leaf"])

    def test_the_animation_lands_on_the_node_the_channel_named(self):
        """the channel names node 0, the leaf, which the rig lists last

        Pairing against the file's array order would walk the two out of step
        and refuse the file outright.
        """
        clip = ClipData.load_glb(self.build(target=0), scale_factor=1.0, fps=4.0)

        names     = [str(node.name) for node in clip]
        moved     = names.index("leaf")
        translate = np.array(clip.frames.translate)

        self.assertAlmostEqual(float(translate[-1, moved, 0]), 7.0, places=4)

        for column in range(len(clip)):
            if column != moved:
                self.assertTrue(
                    np.allclose(translate[:, column, 0], translate[0, column, 0])
                )

    def test_a_mid_chain_channel_pairs_too(self):
        clip = ClipData.load_glb(self.build(target=1), scale_factor=1.0, fps=4.0)

        names     = [str(node.name) for node in clip]
        translate = np.array(clip.frames.translate)

        self.assertAlmostEqual(
            float(translate[-1, names.index("mid"), 0]), 7.0, places=4
        )


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestClipLoadGlbMeshNamesake(unittest.TestCase):
    """a mesh node is no part of the rig, even when a joint shares its name

    The rig skips the mesh and keeps the joint. Channels used to be paired
    with rows by name over every node, so a mesh walked before the joint took
    the joint's row: the joint played the mesh's animation and lost its own.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, mesh_first):
        """root -> neck -> head, plus a mesh node also named head

        The joint moves in x from 0 to 7, the mesh in y from 0 to 5. The
        node array order is the order load_model walks the two in.
        """
        times = np.array([0.0, 1.0], dtype=np.float32)
        joint = np.array([[0.0, 1.0, 0.0], [7.0, 1.0, 0.0]], dtype=np.float32)
        mesh  = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)

        head, prop = (3, 2) if mesh_first else (2, 3)

        nodes       = [None] * 4
        nodes[0]    = Node(name="root", translation=[0.0, 1.0, 0.0], children=[1])
        nodes[1]    = Node(name="neck", translation=[0.0, 1.0, 0.0], children=[head])
        nodes[head] = Node(name="head", translation=[0.0, 1.0, 0.0])
        nodes[prop] = Node(name="head", mesh=0)

        views, blob = [], b""
        for array in (times, joint, mesh):
            views.append(
                BufferView(buffer=0, byteOffset=len(blob), byteLength=array.nbytes)
            )
            blob += array.tobytes()

        gltf             = GLTF2()
        gltf.scene       = 0
        gltf.nodes       = nodes
        gltf.meshes      = [Mesh(name="geo", primitives=[])]
        gltf.scenes      = [Scene(nodes=[0, prop])]
        gltf.buffers     = [Buffer(byteLength=len(blob))]
        gltf.bufferViews = views
        gltf.accessors = [
            Accessor(
                bufferView    = 0,
                componentType = FLOAT,
                count         = 2,
                type          = "SCALAR",
                min           = [0.0],
                max           = [1.0],
            ),
            Accessor(bufferView=1, componentType=FLOAT, count=2, type="VEC3"),
            Accessor(bufferView=2, componentType=FLOAT, count=2, type="VEC3"),
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
                        target  = AnimationChannelTarget(node=head, path="translation"),
                    ),
                    AnimationChannel(
                        sampler = 1,
                        target  = AnimationChannelTarget(node=prop, path="translation"),
                    ),
                ],
            )
        ]
        gltf.set_binary_blob(blob)

        path = os.path.join(self.tmp.name, f"namesake_{int(mesh_first)}.glb")
        gltf.save(path)
        return path

    def test_the_joint_plays_its_own_channel(self):
        """either way round, and the mesh's curve lands on no node"""
        want = np.stack([np.linspace(0.0, 7.0, 5), np.ones(5), np.zeros(5)], axis=1)

        for mesh_first in (True, False):
            with self.subTest(mesh_first=mesh_first):
                clip = ClipData.load_glb(
                    self.build(mesh_first), scale_factor=1.0, fps=4.0
                )

                names     = [str(node.name) for node in clip]
                translate = np.array(clip.frames.translate)
                head      = translate[:, names.index("head")]

                self.assertEqual(names, ["root", "neck", "head"])
                self.assertTrue(np.allclose(head, want), head.tolist())
                self.assertFalse(np.isclose(translate[..., 1], 5.0).any())


@unittest.skipIf(FBX is None, "fbx sdk is not installed")
class TestClipLoadFbx(unittest.TestCase):
    """the fbx exporter writes no animation, so these fixtures are authored
    straight through the sdk
    """

    CHANNELS = {
        "tx": ("LclTranslation", "X"),
        "ty": ("LclTranslation", "Y"),
        "tz": ("LclTranslation", "Z"),
        "rx": ("LclRotation", "X"),
        "ry": ("LclRotation", "Y"),
        "rz": ("LclRotation", "Z"),
        "sx": ("LclScaling", "X"),
        "sy": ("LclScaling", "Y"),
        "sz": ("LclScaling", "Z"),
    }

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def authored(
        self,
        name        = "anim.fbx",
        *,
        joints      = ("joint1",),
        keys        = None,
        span        = (0.0, 1.0),
        time_mode   = "eFrames24",
        stacks      = 1,
        keyed_stack = 0,
        layer_keys  = None,
        unit        = None,
    ):
        """a chain of joints, keys given as {joint: {channel: [(sec, value)]}}

        Only ``keyed_stack`` is keyed, so selecting any other take has to show
        up as a node that no longer moves. Every stack still gets the same
        time span, which is what makes an unkeyed one look plausible.
        ``layer_keys``, keys of the same kind, go on a second layer of the
        keyed stack. ``unit`` names the file's ``FbxSystemUnit``, cm when
        None; the values are written as given, in that unit.
        """
        manager = FBX.FbxManager.Create()
        manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
        scene = FBX.FbxScene.Create(manager, "scene")
        scene.GetGlobalSettings().SetTimeMode(getattr(FBX.FbxTime.EMode, time_mode))
        if unit is not None:
            scene.GetGlobalSettings().SetSystemUnit(getattr(FBX.FbxSystemUnit, unit))

        nodes  = {}
        parent = scene.GetRootNode()
        for joint in joints:
            attr = FBX.FbxSkeleton.Create(scene, f"{joint}Attr")
            attr.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
            node = FBX.FbxNode.Create(scene, joint)
            node.SetNodeAttribute(attr)
            node.LclTranslation.Set(FBX.FbxDouble3(0.0, 1.0, 0.0))
            parent.AddChild(node)
            nodes[joint] = node
            parent       = node

        for index in range(stacks):
            stack = FBX.FbxAnimStack.Create(
                scene, "Take 001" if not index else f"Take {index + 1:03d}"
            )
            layer = FBX.FbxAnimLayer.Create(scene, "Base Layer")
            stack.AddMember(layer)

            if index == keyed_stack:
                keyed = [(layer, keys)]
                if layer_keys:
                    second = FBX.FbxAnimLayer.Create(scene, "Layer 2")
                    stack.AddMember(second)
                    keyed.append((second, layer_keys))

                for target, pairs_of in keyed:
                    for joint, curves in (pairs_of or {}).items():
                        for channel, pairs in curves.items():
                            prop, axis = self.CHANNELS[channel]
                            curve = getattr(nodes[joint], prop).GetCurve(target, axis, True)
                            curve.KeyModifyBegin()
                            for seconds, value in pairs:
                                time = FBX.FbxTime()
                                time.SetSecondDouble(float(seconds))
                                key = curve.KeyAdd(time)[0]
                                curve.KeySetValue(key, float(value))
                                curve.KeySetInterpolation(
                                    key,
                                    FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear,
                                )
                            curve.KeyModifyEnd()

            first, last = FBX.FbxTime(), FBX.FbxTime()
            first.SetSecondDouble(float(span[0]))
            last.SetSecondDouble(float(span[1]))
            stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))

        path     = os.path.join(self.tmp.name, name)
        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
        manager.Destroy()
        return path

    def test_a_take_lands_on_the_frame_grid(self):
        path = self.authored(keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 24.0)]}})

        clip = ClipData.load_fbx(path)

        self.assertEqual(clip.frame_count, 25)
        self.assertEqual(clip.fps, 24.0)
        self.assertTrue(
            np.allclose(
                np.array(clip.frames.translate)[:, 0, 0], np.linspace(0.0, 24.0, 25)
            )
        )

    def test_rotation_arrives_as_the_euler_curve_the_file_holds(self):
        """no euler filtering on this path: fbx already stores euler"""
        path = self.authored(keys={"joint1": {"rz": [(0.0, 0.0), (1.0, 90.0)]}})

        clip = ClipData.load_fbx(path, fps=4.0)

        self.assertTrue(
            np.allclose(
                np.array(clip.frames.rotate)[:, 0, 2], [0.0, 22.5, 45.0, 67.5, 90.0]
            )
        )

    def test_a_wrapping_euler_curve_is_left_alone(self):
        """a filter would rewrite these branches and change the animation

        Sampled so the curve turns a full 360 per frame. Every frame is then
        the same pose as the one before it, so a filter would flatten the
        whole thing to zero while the file plainly says it spins twice.
        """
        path = self.authored(
            keys={"joint1": {"rx": [(0.0, 0.0), (1.0, 720.0)]}}, span=(0.0, 1.0)
        )

        clip = ClipData.load_fbx(path, fps=2.0)

        self.assertTrue(
            np.allclose(np.array(clip.frames.rotate)[:, 0, 0], [0.0, 360.0, 720.0])
        )

    def test_a_ragged_span_rounds_up_so_the_tail_is_covered(self):
        """the take ends between two frames, so truncating would cut it short"""
        path = self.authored(
            keys={"joint1": {"tx": [(0.0, 0.0), (0.51, 10.0)]}}, span=(0.0, 0.51)
        )

        clip = ClipData.load_fbx(path, fps=24.0)

        # 0.51s at 24fps is 12.24 frames, so the grid has to run to frame 13
        self.assertEqual(clip.frame_count, 14)
        self.assertGreaterEqual((clip.frame_count - 1) / 24.0, 0.51)
        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[-1, 0, 0]), 10.0, places=4
        )

    def test_the_default_range_reaches_a_key_past_the_stored_span(self):
        """Unreal stores the take's span a frame short of its last key"""
        path = self.authored(
            keys = {"joint1": {"tx": [(0.0, 0.0), (1.0, 24.0)]}},
            span = (0.0, 23.0 / 24.0),
        )

        clip = ClipData.load_fbx(path)

        self.assertEqual((clip.start_frame, clip.end_frame), (0, 24))
        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[-1, 0, 0]), 24.0, places=4
        )

    def test_the_default_range_keeps_a_stored_span_wider_than_the_keys(self):
        """a Maya export can hold frames either side of its keys"""
        path = self.authored(
            keys = {"joint1": {"tx": [(0.25, 6.0), (0.75, 18.0)]}},
            span = (0.0, 1.0),
        )

        clip = ClipData.load_fbx(path)

        self.assertEqual((clip.start_frame, clip.end_frame), (0, 24))

    def test_a_key_before_the_stored_span_starts_the_range(self):
        path = self.authored(
            keys = {"joint1": {"tx": [(-0.25, -6.0), (1.0, 24.0)]}},
            span = (0.0, 1.0),
        )

        clip = ClipData.load_fbx(path)

        self.assertEqual((clip.start_frame, clip.end_frame), (-6, 24))
        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[0, 0, 0]), -6.0, places=4
        )

    def test_a_first_key_between_two_frames_starts_on_the_frame_before(self):
        """the start rounds down as the end rounds up, so the first key is
        inside the grid: a key at 5.6 frames before a span starting on 6, and
        one at -5.6 before a span starting on 0, which rounds down to -6, not
        toward zero"""
        for key, span, start in ((5.6, 6.0, 5), (-5.6, 0.0, -6)):
            with self.subTest(key=key):
                path = self.authored(
                    name = f"first_key_{key}.fbx",
                    keys = {"joint1": {"tx": [(key / 24.0, 0.0), (1.0, 24.0)]}},
                    span = (span / 24.0, 1.0),
                )

                clip = ClipData.load_fbx(path)

                self.assertEqual((clip.start_frame, clip.end_frame), (start, 24))

    def test_a_key_on_a_later_layer_counts_too(self):
        path = self.authored(
            keys       = {"joint1": {"tx": [(0.0, 0.0), (0.5, 12.0)]}},
            layer_keys = {"joint1": {"ty": [(0.0, 0.0), (1.5, 1.0)]}},
            span       = (0.0, 0.5),
        )

        clip = ClipData.load_fbx(path)

        self.assertEqual((clip.start_frame, clip.end_frame), (0, 36))

    def test_a_span_of_whole_frames_counts_its_own_frames(self):
        """float noise in the span's seconds read frames 60-121 at 24 fps as
        63 frames and frames 1-14 as 15, the last a copy of the one before"""
        for first, last in ((60, 121), (1, 14)):
            with self.subTest(first=first, last=last):
                path = self.authored(
                    name = f"span_{first}_{last}.fbx",
                    keys = {"joint1": {"tx": [(first / 24.0, 0.0), (last / 24.0, 1.0)]}},
                    span = (first / 24.0, last / 24.0),
                )

                clip = ClipData.load_fbx(path)

                self.assertEqual((clip.start_frame, clip.end_frame), (first, last))
                self.assertEqual(clip.frame_count, last - first + 1)

    def test_a_time_a_hair_off_a_whole_frame_is_that_frame(self):
        """two ticks either side of frames 0 and 24 add no frame"""
        tick = 2.0 / hierarchy.hierarchy._fbx_ticks_per_second()
        path = self.authored(
            keys = {"joint1": {"tx": [(-tick, 0.0), (1.0 + tick, 24.0)]}},
            span = (0.0, 1.0),
        )

        clip = ClipData.load_fbx(path)

        self.assertEqual((clip.start_frame, clip.end_frame), (0, 24))

    def test_a_drop_frame_scene_still_gets_a_frame_rate(self):
        """the sdk reports 0.0 for this mode, which would divide by zero"""
        path = self.authored(
            keys      = {"joint1": {"tx": [(0.0, 0.0), (1.0, 1.0)]}},
            time_mode = "eFrames30Drop",
        )

        clip = ClipData.load_fbx(path)

        self.assertAlmostEqual(clip.fps, 30000.0 / 1001.0, places=6)
        self.assertEqual(clip.frame_count, 31)

    def test_scale_is_sampled_too(self):
        path = self.authored(keys={"joint1": {"sy": [(0.0, 1.0), (1.0, 3.0)]}})

        clip = ClipData.load_fbx(path, fps=2.0)

        self.assertTrue(
            np.allclose(np.array(clip.frames.scale)[:, 0, 1], [1.0, 2.0, 3.0])
        )

    def test_an_unanimated_node_holds_its_rest_pose(self):
        path = self.authored(
            joints = ("joint1", "joint2"),
            keys   = {"joint1": {"tx": [(0.0, 0.0), (1.0, 5.0)]}},
        )

        clip  = ClipData.load_fbx(path, fps=4.0)

        other = np.array(clip.frames["joint2"].translate)[:, 0, :]
        self.assertTrue(np.allclose(other, [0.0, 1.0, 0.0]))

    def test_scale_factor_reaches_translation(self):
        path = self.authored(keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 5.0)]}})

        clip = ClipData.load_fbx(path, scale_factor=100.0, fps=2.0)

        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[-1, 0, 0]), 500.0, places=3
        )

    def test_the_file_unit_loads_in_centimeters(self):
        """by default: translations, static and keyed, and nothing else"""
        keys = {
            "joint1": {
                "tx": [(0.0, 0.0), (1.0, 2.0)],
                "rz": [(0.0, 0.0), (1.0, 90.0)],
                "sy": [(0.0, 1.0), (1.0, 3.0)],
            }
        }
        for unit, cm in (("m", 100.0), ("Inch", 2.54), ("mm", 0.1), ("cm", 1.0)):
            with self.subTest(unit=unit):
                path = self.authored(
                    name   = f"{unit}.fbx",
                    joints = ("joint1", "joint2"),
                    keys   = keys,
                    unit   = unit,
                )

                clip = ClipData.load_fbx(path, fps=2.0)
                rig  = HierarchyData.load_fbx(path)

                translate = np.array(clip.frames.translate)
                self.assertTrue(np.allclose(translate[:, 0, 0], [0.0, cm, 2.0 * cm]))
                self.assertTrue(np.allclose(translate[:, 0, 1], cm))  # held by the curve node
                self.assertTrue(np.allclose(translate[:, 1], [0.0, cm, 0.0]))
                self.assertTrue(np.allclose(np.array(clip.frames.rotate)[:, 0, 2], [0.0, 45.0, 90.0]))
                self.assertTrue(np.allclose(np.array(clip.frames.scale)[:, 0, 1], [1.0, 2.0, 3.0]))
                self.assertTrue(np.allclose(rig["joint2"].translate, [0.0, cm, 0.0]))
                self.assertTrue(np.allclose(rig["joint1"].scale, 1.0))
                self.assertTrue(np.allclose(hierarchy.load_fbx(path)["joint2"].translate, [0.0, cm, 0.0]))

    def test_a_number_is_the_callers_own_factor(self):
        """the file's unit is not read: a meters file at 1.0 stays in meters"""
        path = self.authored(
            joints = ("joint1", "joint2"),
            keys   = {"joint1": {"tx": [(0.0, 0.0), (1.0, 2.0)]}},
            unit   = "m",
        )

        clip   = ClipData.load_fbx(path, fps=2.0, scale_factor=1.0)
        rig    = HierarchyData.load_fbx(path, scale_factor=3.0)
        module = hierarchy.load_fbx(path, scale_factor=3.0)

        self.assertTrue(np.allclose(np.array(clip.frames.translate)[:, 0, 0], [0.0, 1.0, 2.0]))
        self.assertTrue(np.allclose(rig["joint2"].translate, [0.0, 3.0, 0.0]))
        self.assertTrue(np.allclose(module["joint2"].translate, [0.0, 3.0, 0.0]))

    def test_the_scene_time_mode_sets_the_frame_rate(self):
        for mode, rate, frames in (
            ("eFrames30", 30.0, 31),
            ("ePAL", 25.0, 26),
            ("eFrames60", 60.0, 61),
        ):
            with self.subTest(mode=mode):
                path = self.authored(
                    name      = f"{mode}.fbx",
                    keys      = {"joint1": {"tx": [(0.0, 0.0), (1.0, 1.0)]}},
                    time_mode = mode,
                )

                clip = ClipData.load_fbx(path)

                self.assertEqual(clip.fps, rate)
                self.assertEqual(clip.frame_count, frames)

    def test_an_explicit_fps_overrides_the_scene(self):
        path = self.authored(
            keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 1.0)]}}, time_mode="eFrames60"
        )

        clip = ClipData.load_fbx(path, fps=8.0)

        self.assertEqual(clip.fps, 8.0)
        self.assertEqual(clip.frame_count, 9)

    def test_start_and_end_frame_trim_the_range(self):
        path = self.authored(keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 24.0)]}})

        clip = ClipData.load_fbx(path, fps=24.0, start_frame=6, end_frame=12)
        full = ClipData.load_fbx(path, fps=24.0)

        self.assertEqual(clip.frame_count, 7)
        self.assertEqual(clip.start_frame, 6)
        self.assertEqual(clip.end_frame,   12)
        self.assertTrue(
            np.allclose(
                np.array(clip.frames.translate),
                np.array(full.frames.translate)[6:13],
            )
        )

    def test_each_end_left_out_defaults_on_its_own(self):
        """a start_frame passed alone keeps the take's last frame as the end,
        not the take's length after it (frame 34 here)"""
        path  = self.authored(keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 24.0)]}})

        late  = ClipData.load_fbx(path, start_frame=10)
        early = ClipData.load_fbx(path, end_frame=12)
        full  = ClipData.load_fbx(path)

        self.assertEqual((late.start_frame, late.end_frame), (10, 24))
        self.assertEqual((early.start_frame, early.end_frame), (0, 12))
        translate = np.array(full.frames.translate)
        self.assertTrue(np.allclose(np.array(late.frames.translate), translate[10:]))
        self.assertTrue(np.allclose(np.array(early.frames.translate), translate[:13]))

    def test_an_end_passed_alone_past_the_take_names_the_take_frame(self):
        """the other end is the take's, not one the caller passed"""
        path = self.authored(keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 24.0)]}})

        with self.assertRaises(ValueError) as caught:
            ClipData.load_fbx(path, start_frame=30)
        self.assertIn("start_frame 30 is past the take's last frame 24", str(caught.exception))
        self.assertIn("pass end_frame too", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            ClipData.load_fbx(path, end_frame=-3)
        self.assertIn("end_frame -3 is before the take's first frame 0", str(caught.exception))
        self.assertIn("pass start_frame too", str(caught.exception))

    def test_take_selects_the_stack(self):
        """only the first stack is keyed, so a later take must not move"""
        path = self.authored(
            keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 9.0)]}}, stacks=3
        )

        first  = np.array(ClipData.load_fbx(path, fps=4.0, take=0).frames.translate)
        second = np.array(ClipData.load_fbx(path, fps=4.0, take=1).frames.translate)

        self.assertAlmostEqual(float(first[-1, 0, 0]), 9.0, places=4)
        self.assertTrue(np.allclose(second[:, 0, 0], 0.0))

    def test_the_default_take_skips_takes_that_hold_no_curves(self):
        """an unkeyed take reads back as the rest pose on every frame, which
        looks like a broken loader rather than an empty take
        """
        path = self.authored(
            keys        = {"joint1": {"tx": [(0.0, 0.0), (1.0, 9.0)]}},
            stacks      = 3,
            keyed_stack = 2,
        )

        clip = ClipData.load_fbx(path, fps=4.0)

        translate = np.array(clip.frames.translate)[:, 0, 0]
        self.assertGreater(float(np.abs(translate - translate[0]).max()), 1.0)
        self.assertAlmostEqual(float(translate[-1]), 9.0, places=4)

    def test_an_explicit_take_still_reaches_an_empty_one(self):
        """the default is a convenience, not a filter -- indexes stay literal"""
        path = self.authored(
            keys        = {"joint1": {"tx": [(0.0, 0.0), (1.0, 9.0)]}},
            stacks      = 3,
            keyed_stack = 2,
        )

        clip = ClipData.load_fbx(path, fps=4.0, take=0)

        self.assertTrue(np.allclose(np.array(clip.frames.translate)[:, 0, 0], 0.0))

    def test_a_file_whose_takes_are_all_empty_still_loads(self):
        path = self.authored(stacks=2, keyed_stack=-1)

        clip = ClipData.load_fbx(path, fps=4.0)

        self.assertEqual(clip.frame_count, 5)
        self.assertTrue(np.allclose(np.array(clip.frames.translate)[:, 0, 1], 1.0))

    def test_a_negative_take_counts_back(self):
        path = self.authored(
            keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 9.0)]}}, stacks=3
        )

        clip = ClipData.load_fbx(path, fps=4.0, take=-3)

        self.assertAlmostEqual(
            float(np.array(clip.frames.translate)[-1, 0, 0]), 9.0, places=4
        )

    def test_a_file_with_no_animation_gives_one_frame(self):
        path = self.authored(stacks=0)

        clip = ClipData.load_fbx(path)

        self.assertEqual(clip.frame_count, 1)

    def test_the_dispatcher_hands_back_a_clip(self):
        path = self.authored(
            name="disp.fbx", keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 1.0)]}}
        )

        self.assertIsInstance(ClipData.load(path), ClipData)
        self.assertNotIsInstance(HierarchyData.load(path), ClipData)

    def test_the_rig_matches_a_plain_hierarchy_read(self):
        """the clip's rest pose is the same read HierarchyData does"""
        path = self.authored(
            joints = ("joint1", "joint2"),
            keys   = {"joint1": {"tx": [(0.0, 0.0), (1.0, 5.0)]}},
        )

        clip = ClipData.load_fbx(path, fps=4.0)
        rig  = HierarchyData.load_fbx(path)

        self.assertEqual(clip.name, rig.name)
        self.assertTrue(np.array_equal(clip.rotate_order, rig.rotate_order))
        self.assertTrue(np.array_equal(clip.get_parents(), rig.get_parents()))
        self.assertTrue(np.allclose(clip.frames.translate[0], rig.translate))

    def test_bad_arguments_refuse(self):
        path = self.authored(keys={"joint1": {"tx": [(0.0, 0.0), (1.0, 1.0)]}})

        for kwargs in ({"fps": 0.0}, {"fps": -24.0}):
            with self.assertRaises(ValueError):
                ClipData.load_fbx(path, **kwargs)

        with self.assertRaises(ValueError):
            ClipData.load_fbx(path, fps=24.0, start_frame=10, end_frame=2)

        with self.assertRaises(IndexError):
            ClipData.load_fbx(path, take=9)


@unittest.skipIf(FBX is None, "fbx sdk is not installed")
class TestClipLoadFbxSampling(unittest.TestCase):
    """load_fbx reads each curve's keys once and evaluates every frame at
    once. These hold it to the sdk's own evaluation, asked one frame at a
    time the way the loader used to ask it.

    The sdk evaluates between keys in single precision, so there the two
    agree to its rounding; on keys, outside them and across flat spans they
    are the same numbers. A weighted span whose two handles crowd one end
    is the exception: the sdk's own time solve stops short there, and the
    loader follows Maya's own import instead, the exact bezier, which the
    crowded span tests hold it to.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rng = np.random.default_rng(2026)

    def take(self, fill, *, name="take.fbx", joints=4, time_mode="eFrames24", layers=1):
        """a chain of joints at a random rest pose over a one second take;
        ``fill(nodes, layers)`` keys it"""
        manager = FBX.FbxManager.Create()
        manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
        scene = FBX.FbxScene.Create(manager, "scene")
        scene.GetGlobalSettings().SetTimeMode(getattr(FBX.FbxTime.EMode, time_mode))

        nodes  = []
        parent = scene.GetRootNode()
        for index in range(joints):
            attr = FBX.FbxSkeleton.Create(scene, f"j{index}Attr")
            attr.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
            node = FBX.FbxNode.Create(scene, f"j{index}")
            node.SetNodeAttribute(attr)
            node.LclTranslation.Set(FBX.FbxDouble3(*self.rng.normal(size=3)))
            node.LclRotation.Set(FBX.FbxDouble3(*(self.rng.normal(size=3) * 30.0)))
            node.LclScaling.Set(FBX.FbxDouble3(*(1.0 + 0.1 * self.rng.normal(size=3))))
            parent.AddChild(node)
            nodes.append(node)
            parent = node

        stack = FBX.FbxAnimStack.Create(scene, "Take 001")
        made  = []
        for index in range(layers):
            layer = FBX.FbxAnimLayer.Create(scene, f"Layer {index}" if index else "Base Layer")
            stack.AddMember(layer)
            made.append(layer)

        fill(nodes, made)

        first, last = FBX.FbxTime(), FBX.FbxTime()
        first.SetSecondDouble(0.0)
        last.SetSecondDouble(1.0)
        stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))

        path     = os.path.join(self.tmp.name, name)
        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
        manager.Destroy()
        return path

    @staticmethod
    def key(
        curve,
        time,
        value,
        kind    = "cubic",
        tangent = "eTangentAuto",
        data    = (0.0, 0.0),
        weights = None,
        to_next = False,
    ):
        """one key at ``time``, in seconds (a float) or in ticks (an int);
        ``weights`` is (weighted mode, right weight, next left weight)"""
        definition = FBX.FbxAnimCurveDef
        when       = FBX.FbxTime()
        if isinstance(time, float):
            when.SetSecondDouble(time)
        else:
            when.Set(int(time))

        index = curve.KeyAdd(when)[0]
        curve.KeySet(
            index,
            when,
            float(value),
            getattr(definition.EInterpolationType, f"eInterpolation{kind.title()}"),
            getattr(definition.ETangentMode, tangent),
            float(data[0]),
            float(data[1]),
            definition.EWeightedMode.eWeightedNone if weights is None else weights[0],
            1.0 / 3.0 if weights is None else float(weights[1]),
            1.0 / 3.0 if weights is None else float(weights[2]),
        )
        mode = "eConstantNext" if to_next else "eConstantStandard"
        curve.KeySetConstantMode(index, getattr(definition.EConstantMode, mode))

    def random_curves(self, nodes, layers):
        """every channel of every joint keyed, a few keys each, on and off
        the 24 fps frame grid, with every interpolation, tangent mode and
        weighting the sdk resolves"""
        definition = FBX.FbxAnimCurveDef
        second     = FBX.FbxTime()
        second.SetSecondDouble(1.0)
        per_second = second.Get()
        tangents = (
            "eTangentAuto",
            "eTangentUser",
            "eTangentBreak",
            "eTangentTCB",
            "eTangentAutoBreak",
            "eTangentGenericClamp",
            "eTangentGenericTimeIndependent",
        )
        weightings = (
            definition.EWeightedMode.eWeightedAll,
            definition.EWeightedMode.eWeightedRight,
            definition.EWeightedMode.eWeightedNextLeft,
        )
        for node in nodes:
            for prop, size in ((node.LclTranslation, 5.0), (node.LclRotation, 90.0), (node.LclScaling, 0.5)):
                for axis in "XYZ":
                    curve = prop.GetCurve(layers[0], axis, True)
                    count = int(self.rng.integers(1, 8))
                    ticks = set(self.rng.integers(per_second // 10, 9 * per_second // 10, count).tolist())
                    ticks |= {round(int(frame) * per_second / 24) for frame in self.rng.integers(3, 21, 2)}

                    curve.KeyModifyBegin()
                    for tick in sorted(ticks):
                        kind    = ("constant", "linear", "cubic", "cubic", "cubic")[int(self.rng.integers(5))]
                        tangent = tangents[int(self.rng.integers(len(tangents)))]
                        weights = None
                        if kind == "cubic" and self.rng.random() < 0.3:
                            weights = (
                                weightings[int(self.rng.integers(3))],
                                self.rng.uniform(0.05, 0.95),
                                self.rng.uniform(0.05, 0.95),
                            )
                        self.key(
                            curve,
                            int(tick),
                            self.rng.normal() * size,
                            kind    = kind,
                            tangent = tangent,
                            data    = self.rng.normal(size=2) * (0.5 if tangent == "eTangentTCB" else size),
                            weights = weights,
                            to_next = bool(self.rng.integers(2)),
                        )
                    curve.KeyModifyEnd()

    def reference(self, path, clip, scale_factor=1.0):
        """the sdk's own evaluation, one frame and one node at a time, of the
        nodes the clip holds: translate, rotate and scale stacked"""
        with hierarchy.hierarchy.open_fbx(path, FBX) as scene:
            criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimStack.ClassId)
            scene.SetCurrentAnimationStack(scene.GetSrcObject(criteria, 0))

            found = {}

            def walk(node):
                found[node.GetName()] = node
                for index in range(node.GetChildCount()):
                    walk(node.GetChild(index))

            walk(scene.GetRootNode())
            nodes  = [found[name] for name in clip.name]

            frames = np.zeros((3, clip.frame_count, len(nodes), 3))
            for frame in range(clip.frame_count):
                time = FBX.FbxTime()
                time.SetSecondDouble((clip.start_frame + frame) / clip.fps)
                for column, node in enumerate(nodes):
                    evaluations = (
                        node.EvaluateLocalTranslation,
                        node.EvaluateLocalRotation,
                        node.EvaluateLocalScaling,
                    )
                    for channel, evaluate in enumerate(evaluations):
                        values                         = evaluate(time)
                        frames[channel, frame, column] = (values[0], values[1], values[2])

        frames[0] *= scale_factor
        return frames

    @staticmethod
    def sampled(clip):
        """the clip's translate, rotate and scale stacked"""
        return np.stack(
            [
                np.array(clip.frames.translate),
                np.array(clip.frames.rotate),
                np.array(clip.frames.scale),
            ]
        )

    @staticmethod
    def error(got, want):
        """the largest difference as a fraction of its curve's largest value:
        between keys the sdk rounds terms of the curve's size to single
        precision, so a value near zero carries an error of that size too"""
        scale = np.maximum(1.0, np.abs(want).max(axis=1, keepdims=True))
        return float((np.abs(got - want) / scale).max())

    def counted(self, path, **kwargs):
        """load_fbx, with every Evaluate* call on an fbx node counted"""
        from unittest import mock

        calls = []
        walk  = hierarchy.hierarchy._walk_fbx

        class Counted:
            """an FbxNode whose Evaluate* calls are counted"""

            def __init__(self, node):
                self._node = node

            def __getattr__(self, name):
                found = getattr(self._node, name)
                if not name.startswith("Evaluate"):
                    return found

                def counting(*args):
                    calls.append(name)
                    return found(*args)

                return counting

        def counted_walk(scene, scale_factor):
            data, nodes = walk(scene, scale_factor)
            return data, {uuid: Counted(node) for uuid, node in nodes.items()}

        with mock.patch.object(hierarchy.hierarchy, "_walk_fbx", counted_walk):
            clip = ClipData.load_fbx(path, **kwargs)
        return clip, calls

    def test_no_frame_is_handed_to_the_sdk(self):
        """the keys are read once and the grid evaluated from them: not one
        EvaluateLocal* call, where a per frame loader makes three per node
        and frame
        """
        path = self.take(self.random_curves)

        clip, calls = self.counted(path, start_frame=-3, end_frame=27)

        self.assertEqual(clip.frame_count, 31)
        self.assertEqual(calls, [])

    def test_every_frame_matches_the_sdk(self):
        """random keys of every kind, a grid that starts before them and ends
        after them, and translation scaled"""
        path = self.take(self.random_curves)

        clip = ClipData.load_fbx(path, scale_factor=2.5, start_frame=-3, end_frame=27)
        got  = self.sampled(clip)
        want = self.reference(path, clip, scale_factor=2.5)

        # between keys, to the sdk's single precision
        self.assertLess(self.error(got, want), 1e-5)

        # before the first key and after the last: the same numbers
        self.assertTrue(np.array_equal(got[:, :3], want[:, :3]))
        self.assertTrue(np.array_equal(got[:, -3:], want[:, -3:]))

    def test_a_long_take_goes_a_batch_of_curves_at_a_time(self):
        """the same numbers when the curves are split into small batches"""
        from unittest import mock

        path  = self.take(self.random_curves)
        whole = self.sampled(ClipData.load_fbx(path, start_frame=-3, end_frame=27))

        # five curves of 31 frames a batch: the 36 curves take eight
        with mock.patch.object(hierarchy.hierarchy, "_FBX_SAMPLE_BUDGET", 5 * 31):
            batched = self.sampled(ClipData.load_fbx(path, start_frame=-3, end_frame=27))

        self.assertTrue(np.array_equal(batched, whole))

    def test_a_take_baked_on_its_frames_is_the_sdk_bit_for_bit(self):
        """a key on every frame, as exported takes have: every sample lands on
        a key, at each rate tried and from a negative start too"""
        for mode, start in (
            ("eFrames24", 0),
            ("eFrames30", -7),
            ("eNTSCFullFrame", 3),
            ("ePAL", 0),
            ("eFrames60", -2),
        ):
            with self.subTest(mode=mode, start=start):
                fps = _fbx_frame_rate(getattr(FBX.FbxTime.EMode, mode))

                def baked(nodes, layers):
                    for node in nodes:
                        for prop in (node.LclTranslation, node.LclRotation, node.LclScaling):
                            for axis in "XYZ":
                                curve = prop.GetCurve(layers[0], axis, True)
                                curve.KeyModifyBegin()
                                for frame in range(start, start + 20):
                                    self.key(curve, frame / fps, self.rng.normal() * 20.0)
                                curve.KeyModifyEnd()

                path = self.take(baked, name=f"{mode}.fbx", time_mode=mode)
                clip = ClipData.load_fbx(path, start_frame=start, end_frame=start + 19)

                self.assertEqual(clip.fps, fps)
                self.assertTrue(np.array_equal(self.sampled(clip), self.reference(path, clip)))

    def test_frame_times_round_to_ticks_the_way_the_sdk_does(self):
        """a frame time a hair short of a whole tick has to land on it, or a
        key baked on that frame is missed"""
        per_second = hierarchy.hierarchy._fbx_ticks_per_second()
        rates      = np.array([24.0, 30.0, 25.0, 30000.0 / 1001.0, 60.0, 23.976, 7.0, 6000.0])
        seconds = np.concatenate(
            [
                (np.arange(-300, 3000)[:, None] / rates).ravel(),
                self.rng.normal(size=4000) * 100.0,
            ]
        )

        want = []
        for value in seconds.tolist():
            time = FBX.FbxTime()
            time.SetSecondDouble(value)
            want.append(time.Get())

        got = hierarchy.hierarchy._fbx_ticks(seconds, per_second)

        self.assertTrue(np.array_equal(got, np.array(want, dtype=np.int64)))

    def test_channels_without_keys_hold_what_the_sdk_holds(self):
        """a curve node's own value where its channel has no keys, and the
        property where the node has no curve node at all"""

        def partly(nodes, layers):
            first, second = nodes[0], nodes[1]

            curve = first.LclTranslation.GetCurve(layers[0], "X", True)
            curve.KeyModifyBegin()
            self.key(curve, 0.25, 4.0)
            self.key(curve, 0.75, -2.0)
            curve.KeyModifyEnd()

            # Y gets an empty curve, which the exporter drops, and Z none;
            # the curve node holds values the property does not
            first.LclTranslation.GetCurve(layers[0], "Y", True)
            curve_node = first.LclTranslation.GetCurveNode(layers[0])
            curve_node.SetChannelValue(1, 0.1)
            curve_node.SetChannelValue(2, 7.3)

            curve = second.LclRotation.GetCurve(layers[0], "Z", True)
            curve.KeyModifyBegin()
            self.key(curve, 0.5, 45.0, kind="linear")
            curve.KeyModifyEnd()

        path = self.take(partly)
        clip = ClipData.load_fbx(path, fps=8.0)

        got  = self.sampled(clip)
        want = self.reference(path, clip)

        self.assertTrue(np.array_equal(got, want))
        self.assertEqual(float(got[0, 0, 0, 1]), 0.1)
        self.assertEqual(float(got[0, 0, 0, 2]), 7.3)

    def test_an_empty_curve_holds_its_value_in_single_precision(self):
        """the exporter drops a curve with no keys, so this one is built in
        memory and sampled straight from the scene"""
        manager = FBX.FbxManager.Create()
        self.addCleanup(manager.Destroy)
        scene = FBX.FbxScene.Create(manager, "scene")
        node  = FBX.FbxNode.Create(scene, "n")
        scene.GetRootNode().AddChild(node)
        stack = FBX.FbxAnimStack.Create(scene, "Take 001")
        layer = FBX.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)
        scene.SetCurrentAnimationStack(stack)

        node.LclTranslation.Set(FBX.FbxDouble3(0.1, 0.2, 0.3))
        curve = node.LclTranslation.GetCurve(layer, "Y", True)
        curve.KeyModifyBegin()
        self.key(curve, 0.5, 4.0)
        curve.KeyModifyEnd()
        node.LclTranslation.GetCurve(layer, "X", True)

        per_second   = hierarchy.hierarchy._fbx_ticks_per_second()
        ticks        = hierarchy.hierarchy._fbx_ticks(np.arange(5) / 4.0, per_second)
        blocks       = [np.zeros((5, 1, 3)), np.zeros((5, 1, 3)), np.ones((5, 1, 3))]
        blocks[0][:] = (0.1, 0.2, 0.3)

        hierarchy.hierarchy._fbx_sample_take(stack, [(node, 0)], ticks, 1.0, blocks)

        want = []
        for tick in ticks.tolist():
            time = FBX.FbxTime()
            time.Set(tick)
            values = node.EvaluateLocalTranslation(time)
            want.append((values[0], values[1], values[2]))

        self.assertTrue(np.array_equal(blocks[0][:, 0], np.array(want)))
        self.assertEqual(float(blocks[0][0, 0, 0]), float(np.float32(0.1)))
        self.assertEqual(float(blocks[0][0, 0, 2]), 0.3)

    def test_no_take_goes_through_the_sdk_frame_by_frame(self):
        """several layers, a muted one, a curve repeating past its keys and a
        take with no layer at all come back as the sdk evaluates them, and
        not one frame is asked of the sdk: blending the layers is done here
        too. Before the first key and after the last, the very same numbers;
        between keys, to the sdk's single precision. The muted take's only
        layer is muted, so it holds the rest pose"""

        def keyed(nodes, layers):
            for node in nodes[:2]:
                curve = node.LclTranslation.GetCurve(layers[0], "X", True)
                curve.KeyModifyBegin()
                self.key(curve, 0.25, 1.0, tangent="eTangentUser", data=(8.0, -3.0))
                self.key(curve, 0.5, -1.0, tangent="eTangentUser", data=(2.0, 5.0))
                self.key(curve, 0.75, 3.0)
                curve.KeyModifyEnd()

        def layered(nodes, layers):
            keyed(nodes, layers)
            curve = nodes[0].LclTranslation.GetCurve(layers[1], "X", True)
            curve.KeyModifyBegin()
            self.key(curve, 0.0, 5.0, kind="linear")
            self.key(curve, 1.0, -5.0, kind="linear")
            curve.KeyModifyEnd()

        def muted(nodes, layers):
            keyed(nodes, layers)
            layers[0].Mute.Set(True)

        def cycling(nodes, layers):
            keyed(nodes, layers)
            curve = nodes[1].LclTranslation.GetCurve(layers[0], "X")
            curve.SetPreExtrapolation(FBX.FbxAnimCurve.EExtrapolationType.eRepetition)
            curve.SetPreExtrapolationCount(3)
            curve.SetPostExtrapolation(FBX.FbxAnimCurve.EExtrapolationType.eMirrorRepetition)
            curve.SetPostExtrapolationCount(3)

        def bare(nodes, layers):
            pass

        frames = 37
        for name, fill, layers in (
            ("layered", layered, 2),
            ("muted", muted, 1),
            ("cycling", cycling, 1),
            ("bare", bare, 0),
        ):
            with self.subTest(name=name):
                path = self.take(fill, name=f"{name}.fbx", layers=layers)

                clip, calls = self.counted(path, fps=24.0, start_frame=-6, end_frame=30)
                got  = self.sampled(clip)
                want = self.reference(path, clip)

                self.assertEqual(clip.frame_count, frames)
                self.assertEqual(calls, [])
                self.assertLess(self.error(got, want), 1e-5)
                if name != "cycling":
                    # before frame 0 and after frame 24, past every key
                    self.assertTrue(np.array_equal(got[:, :6], want[:, :6]))
                    self.assertTrue(np.array_equal(got[:, -6:], want[:, -6:]))
                if name == "muted":
                    self.assertTrue(np.array_equal(got, want))

    def layered(self, rng, nodes, layers, baked, weight_curves=True):
        """every layer set at random -- blend mode, weight in and out of
        0..100, mute, solo, rotation and scaling accumulation, the double3
        blend mode bypassed -- and keyed at random: each node's channels on
        some layers and not others, a curve node's own values set where it
        has no curve, every rotate order, scaling signed. ``baked`` keys
        every 24 fps frame from -3 to 27, where every sample lands on a key;
        otherwise a few keys of every interpolation off the grid. Some layers
        carry a weight curve, keyed the same way, most in the layer itself
        and the others in the layer below, which the sdk does not read"""
        per_second = hierarchy.hierarchy._fbx_ticks_per_second()
        orders     = list(FBX.EFbxRotationOrder)[:6]
        source     = FBX.FbxNode.EPivotSet.eSourcePivot
        kinds      = ("constant", "linear", "cubic")
        for index, layer in enumerate(layers):
            layer.BlendMode.Set(float(rng.integers(3)))
            layer.RotationAccumulationMode.Set(float(rng.integers(2)))
            layer.ScaleAccumulationMode.Set(float(rng.integers(2)))
            layer.Weight.Set(float(rng.choice([100.0, rng.uniform(-30.0, 130.0)])))
            layer.Mute.Set(bool(rng.random() < 0.15))
            layer.Solo.Set(bool(rng.random() < 0.15))
            if rng.random() < 0.15:
                layer.SetBlendModeBypass(FBX.EFbxType.eFbxDouble3, True)

        for node in nodes:
            node.SetRotationOrder(source, orders[int(rng.integers(6))])
            node.SetRotationActive(bool(rng.integers(2)))
            for layer in layers:
                channels = ((node.LclTranslation, 5.0), (node.LclRotation, 180.0), (node.LclScaling, 1.0))
                for prop, size in channels:
                    if rng.random() < 0.3:
                        continue
                    for axis in "XYZ":
                        if rng.random() < 0.3:
                            continue
                        curve = prop.GetCurve(layer, axis, True)
                        if baked:
                            ticks = [round(frame * per_second / 24) for frame in range(-3, 28)]
                        else:
                            ticks = sorted(set(rng.integers(0, per_second, int(rng.integers(1, 5))).tolist()))
                        curve.KeyModifyBegin()
                        for tick in ticks:
                            value = rng.normal() * size
                            if prop is node.LclScaling:
                                value = rng.uniform(0.3, 2.0) * rng.choice([1.0, -1.0], p=[0.85, 0.15])
                            self.key(curve, int(tick), value, kind=kinds[int(rng.integers(3))])
                        curve.KeyModifyEnd()
                    curve_node = prop.GetCurveNode(layer)
                    if curve_node is not None and rng.random() < 0.5:
                        for axis in range(3):
                            curve_node.SetChannelValue(axis, float(rng.uniform(0.3, 2.0) * size))

        if weight_curves:
            for index, layer in enumerate(layers[1:], 1):
                if rng.random() < 0.6:
                    host  = layer if rng.random() < 0.8 else layers[index - 1]
                    curve = layer.Weight.GetCurve(host, True)
                    if baked:
                        ticks = [round(frame * per_second / 24) for frame in range(-3, 28)]
                    else:
                        ticks = sorted(set(rng.integers(0, per_second, 3).tolist()))
                    curve.KeyModifyBegin()
                    for tick in ticks:
                        self.key(curve, int(tick), rng.uniform(-20.0, 120.0), kind="linear")
                    curve.KeyModifyEnd()

    @staticmethod
    def turned(got, want):
        """the largest difference of two arrays of angles, a whole turn apart
        counting as none"""
        return float(np.abs((got - want + 180.0) % 360.0 - 180.0).max())

    def rotation_against(self, got, want, orders, tolerance):
        """each joint's rotate curve against the sdk's: the same angles, or,
        where the loader rebuilt the rotation from quaternions and filtered
        it, the same rotation on every frame, frame 0 the same angles and
        the curve one the filter leaves as it is. Within ``tolerance``
        degrees, a whole turn counting as none. How many joints took the
        filter"""
        filtered = 0
        for joint in range(got.shape[1]):
            mine, sdk = got[:, joint], want[:, joint]
            if self.turned(mine, sdk) < tolerance:
                continue
            order = int(orders[joint])
            turns = euler_to_matrix(np.radians(mine), order) - euler_to_matrix(np.radians(sdk), order)
            self.assertLess(float(np.abs(turns).max()), np.radians(tolerance), f"joint {joint}")
            self.assertLess(self.turned(mine[:1], sdk[:1]), tolerance, f"joint {joint}")
            again = _filtered(mine[:, None], orders[[joint]])[:, 0]
            self.assertLess(self.turned(again, mine), tolerance, f"joint {joint}")
            filtered += 1
        return filtered

    def test_layers_blend_as_the_sdk_blends_them(self):
        """30 takes of 2 to 4 layers set and keyed at random over 6 joints:
        no frame goes through the sdk. Baked on every frame, translation and
        scaling are the sdk's bit for bit and rotation, blended by layer
        through quaternions, within 1e-9 degrees of it, or of it put through
        the euler filter where the sdk's curve steps (the loader filters a
        rotation it rebuilds); keyed off the grid, within the single
        precision the sdk works in between keys"""
        filtered = 0
        for seed in range(30):
            rng    = np.random.default_rng(seed)
            baked  = seed % 2 == 0
            layers = int(rng.integers(2, 5))
            with self.subTest(seed=seed, baked=baked, layers=layers):
                path = self.take(
                    lambda nodes, made: self.layered(rng, nodes, made, baked),
                    name   = f"layered{seed}.fbx",
                    joints = 6,
                    layers = layers,
                )
                clip, calls = self.counted(path, fps=24.0, start_frame=-3, end_frame=27)
                got, want = self.sampled(clip), self.reference(path, clip)

                self.assertEqual(calls, [])
                orders = np.array(clip.rotate_order)
                if baked:
                    self.assertTrue(np.array_equal(got[0], want[0]))
                    self.assertTrue(np.array_equal(got[2], want[2]))
                    filtered += self.rotation_against(got[1], want[1], orders, 1e-9)
                else:
                    self.assertLess(self.error(got[[0, 2]], want[[0, 2]]), 1e-5)
                    scale = np.maximum(1.0, np.abs(want[1]).max())
                    filtered += self.rotation_against(got[1], want[1], orders, 1e-5 * scale)
        self.assertGreater(filtered, 0)

    def test_a_rotation_blended_by_layer_is_euler_filtered(self):
        """an additive layer turns a joint spinning a whole turn a little
        more: blended by layer, the sdk rebuilds each frame's rotation from
        quaternions and its curve steps by a whole turn; the loader's goes
        through euler_filter in the joint's rotate order, frame 0 the sdk's,
        and holds the same rotation every frame. Blended by channel, angles
        add and nothing is rebuilt, and a joint no layer above touches keeps
        the file's own curve, a whole turn step the file holds included:
        both the sdk's bit for bit"""

        def fill(nodes, layers):
            source = FBX.FbxNode.EPivotSet.eSourcePivot
            for index, node in enumerate(nodes[:3]):
                node.SetRotationOrder(source, list(FBX.EFbxRotationOrder)[(0, 4, 1)[index]])
                curve = node.LclRotation.GetCurve(layers[0], "X", True)
                curve.KeyModifyBegin()
                for frame in range(25):
                    # joints 1 and 2 hold a whole turn step of their own
                    turn = 360.0 if index and frame > 12 else 0.0
                    self.key(curve, frame / 24.0, 15.0 * frame - turn, kind="linear")
                curve.KeyModifyEnd()
            for node, layer in ((nodes[0], layers[1]), (nodes[1], layers[2])):
                for axis, value in (("Y", 10.0), ("Z", 5.0)):
                    curve = node.LclRotation.GetCurve(layer, axis, True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, value, kind="linear")
                    curve.KeyModifyEnd()
            layers[2].RotationAccumulationMode.Set(
                float(FBX.FbxAnimLayer.ERotationAccumulationMode.eRotationByChannel.value)
            )

        path = self.take(fill, name="spin.fbx", layers=3)
        clip, calls = self.counted(path, fps=24.0, start_frame=0, end_frame=24)
        got, want = self.sampled(clip)[1], self.reference(path, clip)[1]
        orders = np.array(clip.rotate_order)

        self.assertEqual(calls, [])
        steps = np.abs(np.diff(want[:, 0], axis=0)).max()
        self.assertGreater(steps, 300.0)
        self.assertLess(np.abs(np.diff(got[:, 0], axis=0)).max(), 20.0)
        self.assertLess(np.abs(got[0, 0] - want[0, 0]).max(), 1e-9)
        self.assertLess(np.abs(got[:, 0] - _filtered(want[:, [0]], orders[[0]])[:, 0]).max(), 1e-9)
        mine   = euler_to_matrix(np.radians(got[:, 0]), int(orders[0]))
        theirs = euler_to_matrix(np.radians(want[:, 0]), int(orders[0]))
        self.assertLess(np.abs(mine - theirs).max(), 1e-12)

        self.assertTrue(np.array_equal(got[:, 1:], want[:, 1:]))
        for joint in (1, 2):
            self.assertGreater(np.abs(np.diff(got[:, joint, 0])).max(), 300.0)

    def test_the_blend_starts_from_the_first_live_layer_holding_anything(self):
        """the sdk blends from the first layer that is not muted and holds
        anything: that layer's value is taken whatever its weight and blend
        mode. A channel it holds no curve node for starts from 0 under the
        later layers, scaling too, and a channel no live layer holds keeps
        the rest pose. Layer 1 adds 100 to translate x at weight 50 over
        each base"""

        def fill(base):
            def keyed(nodes, layers):
                if base == "muted":
                    curve = nodes[0].LclTranslation.GetCurve(layers[0], "X", True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, 2.0, kind="linear")
                    curve.KeyModifyEnd()
                    layers[0].Mute.Set(True)
                if base == "another node's":
                    curve = nodes[1].LclRotation.GetCurve(layers[0], "Z", True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, 2.0, kind="linear")
                    curve.KeyModifyEnd()
                if base == "its own":
                    curve = nodes[0].LclTranslation.GetCurve(layers[0], "X", True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, 2.0, kind="linear")
                    curve.KeyModifyEnd()
                for prop, value in ((nodes[0].LclTranslation, 100.0), (nodes[0].LclScaling, 3.0)):
                    curve = prop.GetCurve(layers[1], "X", True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, value, kind="linear")
                    curve.KeyModifyEnd()
                layers[1].Weight.Set(50.0)

            return keyed

        for base, x, scale in (
            ("muted", 100.0, 3.0),
            ("empty", 100.0, 3.0),
            ("another node's", 50.0, 0.0),
            ("its own", 52.0, 0.0),
        ):
            with self.subTest(base=base):
                path = self.take(fill(base), name=f"base_{base[:5]}.fbx", layers=2)
                clip, calls = self.counted(path, fps=24.0, start_frame=0, end_frame=4)
                got, want = self.sampled(clip), self.reference(path, clip)

                self.assertEqual(calls, [])
                self.assertTrue(np.array_equal(got, want))
                self.assertEqual(float(got[0, 0, 0, 0]), x)
                self.assertEqual(float(got[2, 0, 0, 0]), scale)

    def test_a_layer_weight_curve_is_read_from_its_own_layer(self):
        """a weight curve the layer holds itself animates its weight; one
        held in another layer is not read, as the sdk does not read it"""
        for host in (1, 0):

            def fill(nodes, layers):
                for index, value in ((0, 2.0), (1, 100.0)):
                    curve = nodes[0].LclTranslation.GetCurve(layers[index], "X", True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, value, kind="linear")
                    curve.KeyModifyEnd()
                curve = layers[1].Weight.GetCurve(layers[host], True)
                curve.KeyModifyBegin()
                self.key(curve, 0.0, 0.0, kind="linear")
                self.key(curve, 1.0, 100.0, kind="linear")
                curve.KeyModifyEnd()

            with self.subTest(host=host):
                path = self.take(fill, name=f"weight{host}.fbx", layers=2)
                clip, calls = self.counted(path, fps=4.0, start_frame=0, end_frame=4)
                got, want = self.sampled(clip), self.reference(path, clip)

                self.assertEqual(calls, [])
                self.assertTrue(np.array_equal(got, want))
                expected = [2.0, 27.0, 52.0, 77.0, 102.0] if host == 1 else [102.0] * 5
                self.assertEqual(got[0, :, 0, 0].tolist(), expected)

    def test_a_curve_repeating_past_its_keys_is_read_where_the_sdk_reads_it(self):
        """repetition, mirror repetition and relative repetition (Maya's
        cycle with offset), before the keys, after them or both, for 0 to 4
        periods: each frame past the keys reads the keys where the sdk reads
        them, a relative one shifted a period at a time, and holds the end
        key once the count runs out. Keep slope holds the end key in this
        sdk. No frame goes through the sdk; a curve repeating only after its
        keys used to, flagged by its post extrapolation alone, and a relative
        one held its end key"""
        kinds = {
            "constant": "eConstant",
            "repeat":   "eRepetition",
            "mirror":   "eMirrorRepetition",
            "slope":    "eKeepSlope",
            "offset":   5,  # relative repetition, which the bindings leave unnamed
        }
        cases = [
            ("repeat", 3, "mirror", 3),
            ("constant", 0, "repeat", 2),  # after the keys only
            ("mirror", 4, "constant", 0),  # before them only
            ("repeat", 1, "repeat", 0),  # a count of 0 holds
            ("mirror", 2, "mirror", 1),
            ("slope", 2, "slope", 2),
            ("offset", 3, "offset", 2),
            ("constant", 0, "offset", 4),
            ("offset", 1, "mirror", 2),
            ("offset", 0, "offset", 1),
            ("offset", 2**31, "offset", 2**31),
        ]
        extrapolation = FBX.FbxAnimCurve.EExtrapolationType

        def past(name):
            spelled = kinds[name]
            return extrapolation(spelled) if isinstance(spelled, int) else getattr(extrapolation, spelled)

        def cycling(nodes, layers):
            for index, (pre, pre_count, post, post_count) in enumerate(cases):
                node = nodes[index % len(nodes)]
                prop = (node.LclTranslation, node.LclRotation, node.LclScaling)[index // len(nodes)]
                for axis in "XYZ":
                    curve = prop.GetCurve(layers[0], axis, True)
                    curve.KeyModifyBegin()
                    times = np.sort(self.rng.uniform(0.25, 0.75, 3))
                    times[0], times[-1] = 0.25, 0.75
                    for when, kind in zip(times, ("cubic", "linear", "cubic")):
                        self.key(curve, float(when), self.rng.normal() * 10.0, kind=kind)
                    curve.KeyModifyEnd()
                    curve.SetPreExtrapolation(past(pre))
                    curve.SetPreExtrapolationCount(pre_count)
                    curve.SetPostExtrapolation(past(post))
                    curve.SetPostExtrapolationCount(post_count)

        path = self.take(cycling, name="cycling.fbx", joints=6)
        clip, calls = self.counted(path, fps=24.0, start_frame=-36, end_frame=72)
        got, want = self.sampled(clip), self.reference(path, clip)

        self.assertEqual(calls, [])
        self.assertLess(self.error(got, want), 1e-5)

        # every frame a whole number of key spans from a key reads that key
        whole = (np.arange(clip.frame_count) + clip.start_frame - 6) % 12 == 0
        self.assertTrue(whole.sum() > 4)
        self.assertTrue(np.array_equal(got[:, whole], want[:, whole]))

    def test_what_the_sdk_does_not_know_reads_as_it_reads_it(self):
        """an extrapolation number the sdk does not know (a file carries one
        as 0) reads 0 for as many periods as the count says, then holds the
        end key; an interpolation it does not know (carried as 0 too) gives 0
        between its key and the next. Both as the sdk evaluates them, and no
        frame asked of it, where the loader sent such a joint through the
        sdk frame by frame. Keep slope is checked once a curve and side, with
        the curve's own Evaluate at the frame farthest past its keys: an sdk
        that carries the slope on is refused"""
        extrapolation = FBX.FbxAnimCurve.EExtrapolationType
        interpolation = FBX.FbxAnimCurveDef.EInterpolationType

        def fill(nodes, layers):
            for node, how in ((nodes[1], "slope"), (nodes[2], "past"), (nodes[3], "between")):
                curve = node.LclTranslation.GetCurve(layers[0], "X", True)
                curve.KeyModifyBegin()
                self.key(curve, 0.25, 1.0, kind="linear")
                self.key(curve, 0.5, 4.0)
                self.key(curve, 0.75, 2.0, kind="linear")
                curve.KeyModifyEnd()
                if how == "slope":
                    curve.SetPostExtrapolation(extrapolation.eKeepSlope)
                elif how == "past":
                    curve.SetPreExtrapolation(extrapolation(7))
                    curve.SetPreExtrapolationCount(2)
                    curve.SetPostExtrapolation(extrapolation(7))
                    curve.SetPostExtrapolationCount(1)
                else:
                    curve.KeySetInterpolation(1, interpolation(16))

        path  = self.take(fill, name="unknown.fbx")
        holds = hierarchy.hierarchy._fbx_agrees_past
        with mock.patch.object(hierarchy.hierarchy, "_fbx_agrees_past", wraps=holds) as checked:
            clip, calls = self.counted(path, fps=24.0, start_frame=-12, end_frame=36)
        got, want = self.sampled(clip), self.reference(path, clip)

        self.assertEqual(checked.call_count, 1)
        self.assertEqual(calls, [])
        self.assertLess(self.error(got, want), 1e-5)

        # the keys sit on frames 6, 12 and 18: joint 2 reads 0 for two spans
        # before them and one after, joint 3 between its last two keys
        frame = np.arange(clip.frame_count) + clip.start_frame
        past  = (frame < 6) | (frame > 18)
        self.assertTrue(np.array_equal(got[0, past, 2, 0], want[0, past, 2, 0]))
        self.assertEqual(got[0, (frame >= -18) & (frame < 6), 2, 0].tolist(), [0.0] * 18)
        self.assertEqual(got[0, (frame > 18) & (frame < 30), 2, 0].tolist(), [0.0] * 11)
        self.assertEqual(got[0, frame >= 30, 2, 0].tolist(), [2.0] * 7)
        inside = (frame > 12) & (frame < 18)
        self.assertEqual(got[0, inside, 3, 0].tolist(), [0.0] * 5)
        self.assertTrue(np.array_equal(got[0, inside, 3], want[0, inside, 3]))

        with mock.patch.object(hierarchy.hierarchy, "_fbx_agrees_past", lambda *args: False):
            with self.assertRaisesRegex(ValueError, "keep slope"):
                ClipData.load_fbx(path, fps=24.0, start_frame=-12, end_frame=36)

    def test_a_repetition_lands_on_the_ticks_the_sdk_reads(self):
        """the periods counted in whole ticks: a tick a whole number of
        periods past the last key starts the next period, as far before the
        first key ends the previous one. An offset repetition reads the
        ticks a repetition reads, each shifted by its period, the count once
        it runs out; nothing else is shifted. An extrapolation number the sdk
        does not know reads 0 for the count's periods, counted the same way,
        then the end key"""
        from cgmath.hierarchy.hierarchy import _fbx_extrapolated

        first, last = 100, 160
        ticks  = np.array([30, 40, 41, 99, 100, 160, 161, 219, 220, 221, 280, 281])
        repeat = _fbx_extrapolated(ticks, first, last, ("repeat", 2), ("repeat", 2))
        mirror = _fbx_extrapolated(ticks, first, last, ("mirror", 2), ("mirror", 2))
        held   = _fbx_extrapolated(ticks, first, last, (None, 0),     ("repeat", 0))
        offset = _fbx_extrapolated(ticks, first, last, ("offset", 2), ("offset", 2))
        zero   = _fbx_extrapolated(ticks, first, last, ("zero", 2),   ("zero", 1))

        self.assertEqual(repeat[0].tolist(), [150, 160, 101, 159, 100, 160, 101, 159, 100, 101, 160, 160])
        self.assertEqual(mirror[0].tolist(), [150, 160, 159, 101, 100, 160, 159, 101, 100, 101, 160, 160])
        self.assertEqual(held[0].tolist(),   np.clip(ticks, first, last).tolist())
        self.assertEqual(offset[0].tolist(), repeat[0].tolist())
        self.assertEqual(offset[1].tolist(), [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2])
        for other in (repeat, mirror, held):
            self.assertFalse(other[1].any())
        self.assertEqual(zero[0].tolist(), [100] * 5 + [160] * 7)
        self.assertEqual(zero[2].tolist(), [True] * 4 + [False] * 2 + [True] * 2 + [False] * 4)
        self.assertFalse(zero[1].any())
        for other in (repeat, mirror, held, offset):
            self.assertFalse(other[2].any())

    def by_layer(self, nodes, layers, cases):
        """``cases`` is (layer index, blend mode, weight, base angles, layer
        angles), one per node: the base keyed on layer 0, the layer's angles
        on its own layer, rotation blended by layer; each case's layer touches
        its own nodes only, every node a rotate order of its own"""
        source = FBX.FbxNode.EPivotSet.eSourcePivot
        orders = list(FBX.EFbxRotationOrder)[:6]
        # by position: the binding's == holds for any two fbx objects
        for at, (node, (index, mode, weight, base, turn)) in enumerate(zip(nodes, cases)):
            node.SetRotationOrder(source, orders[at % 6])
            layers[index].BlendMode.Set(float(mode))
            layers[index].Weight.Set(float(weight))
            for layer, angles in ((layers[0], base), (layers[index], turn)):
                for axis, angle in zip("XYZ", angles):
                    curve = node.LclRotation.GetCurve(layer, axis, True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, float(angle), kind="constant")
                    curve.KeyModifyEnd()

    def test_a_rotation_blended_onto_gimbal_splits_as_the_sdk_splits_it(self):
        """where a blend by layer lands exactly on a middle angle of +-90, the
        sdk gives the last angle 0 and the first the whole turn, and
        everything a cosine of 16 double epsilons from it counts: the base
        held at (0, +-90, 0) or (30, +-90, 10) under an additive layer of 0,
        an override layer holding it and a passthrough layer holding it too,
        and 45 under an additive 45, in every rotate order. The loader gave
        (180, 90, 180) for (0, 90, 0), a turn of half a revolution in four of
        the orders"""
        additive, override, passthrough = 0, 1, 2
        held  = [(0, 90, 0), (0, -90, 0), (30, 90, 10), (30, -90, 10)]
        cases = []
        for angles in held:
            cases += [(1, additive, 100, angles, (0, 0, 0))] * 6
            cases += [(2, override, 100, (0, 0, 0), angles)] * 6
            cases += [(3, passthrough, 50, angles, angles)] * 6
        cases += [(4, additive, 100, (0, 45, 0), (0, 45, 0))] * 6

        path = self.take(
            lambda nodes, layers: self.by_layer(nodes, layers, cases),
            name   = "gimbal.fbx",
            joints = len(cases),
            layers = 5,
        )
        clip, calls = self.counted(path, fps=24.0, start_frame=0, end_frame=2)
        got, want = self.sampled(clip)[1], self.reference(path, clip)[1]
        orders = np.array(clip.rotate_order)

        self.assertEqual(calls, [])
        self.assertTrue(np.all(np.abs(np.abs(want[0, :, 1]) - 90.0) < 1e-9))
        for joint in range(len(cases)):
            with self.subTest(joint=joint, case=cases[joint]):
                self.assertLess(self.turned(got[:, joint], want[:, joint]), 1e-9)
                mine   = euler_to_matrix(np.radians(got[:, joint]), int(orders[joint]))
                theirs = euler_to_matrix(np.radians(want[:, joint]), int(orders[joint]))
                self.assertLess(float(np.abs(mine - theirs).max()), 1e-12)
        # the sdk's split, the last angle 0
        self.assertTrue(np.array_equal(want[0, -6:, 2], np.zeros(6)))

    def test_a_rotation_blended_next_to_gimbal_keeps_its_three_angles(self):
        """a hair off gimbal the sdk does not split, and neither does the
        loader: the base held at (30, +-89.99999, 10) under an additive layer
        of 0 (single precision keys, 89.9999924, a cosine of 1.3e-7), and a
        base of (0, 0, 10) under an additive y = 90 layer at weight 100 (1 -
        d), d from 1e-7 to 7e-15 (a cosine of 1.1e-14), an additive x = 30
        layer on top, in every rotate order. Reading the angles back loses a
        few times 1e-16 / cos(middle angle) radians in the sdk as here, so
        the two agree to that, as matrices within 1e-15 / cos. Split at 16
        single precision epsilons, as Shoemake's published code splits, the
        loader gave (40, 89.9999924, 0) for the sdk's (30, 89.9999924, 10);
        split at 1e-9, 1e-12 or 64 double epsilons, rotations 0.17 to 0.18
        off as matrices from d = 1e-10, 1e-14 and 7e-15 on. The two sides'
        rounding of the cosine picks the branch within about 1e-15 of the
        split, so no d here comes nearer"""
        additive = 0
        offsets  = (1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 1e-10, 1e-12, 1e-14, 7e-15)

        def fill(nodes, layers):
            source = FBX.FbxNode.EPivotSet.eSourcePivot
            orders = list(FBX.EFbxRotationOrder)[:6]
            for layer in layers[1:]:
                layer.BlendMode.Set(float(additive))
                layer.Weight.Set(100.0)
            for at, offset in enumerate(offsets):
                layers[2 + at].Weight.Set(100.0 * (1.0 - offset))

            for index, node in enumerate(nodes):
                node.SetRotationOrder(source, orders[index % 6])
                case = index // 6
                if case < 2:
                    keyed = ((0, (30.0, (89.99999, -89.99999)[case], 10.0)), (1, (0.0, 0.0, 0.0)))
                else:
                    keyed = ((0, (0.0, 0.0, 10.0)), (case, (0.0, 90.0, 0.0)), (len(layers) - 1, (30.0, 0.0, 0.0)))
                for at, angles in keyed:
                    for axis, angle in zip("XYZ", angles):
                        curve = node.LclRotation.GetCurve(layers[at], axis, True)
                        curve.KeyModifyBegin()
                        self.key(curve, 0.0, angle, kind="constant")
                        curve.KeyModifyEnd()

        path = self.take(
            fill,
            name   = "near_gimbal.fbx",
            joints = 6 * (2 + len(offsets)),
            layers = 3 + len(offsets),
        )
        clip, calls = self.counted(path, fps=24.0, start_frame=0, end_frame=1)
        got, want = self.sampled(clip)[1], self.reference(path, clip)[1]
        cosine = np.abs(np.cos(np.radians(want[..., 1])))
        orders = np.array(clip.rotate_order)

        self.assertEqual(calls, [])
        self.assertTrue(np.all((cosine > 5e-15) & (cosine < 2e-7)))
        self.assertTrue(np.all(want[..., 2] != 0.0))
        np.testing.assert_allclose(want[:, :12, 0::2], np.tile([30.0, 10.0], (2, 12, 1)), atol=1e-4)
        for joint in range(got.shape[1]):
            with self.subTest(joint=joint):
                bound = np.degrees(3e-15 / cosine[:, joint]).max()
                self.assertLess(self.turned(got[:, joint], want[:, joint]), bound)
                mine   = euler_to_matrix(np.radians(got[:, joint]), int(orders[joint]))
                theirs = euler_to_matrix(np.radians(want[:, joint]), int(orders[joint]))
                apart  = np.abs(mine - theirs).max(axis=(-2, -1)) * cosine[:, joint]
                self.assertLess(float(apart.max()), 1e-15)

    def test_a_tie_breaks_the_way_the_sdk_breaks_it(self):
        """every layer angle a multiple of 45 degrees, where the sdk's own
        rounding decides the sign of a quaternion (a turn of a third of a
        revolution has a trace of exactly 0, diagonal entries tie, half
        turns) and so which way an additive layer turns at a partial weight,
        and where a passthrough layer sits exactly half a revolution from
        the one below; over four poses below, in every rotate order. The
        sdk builds each quaternion from the matrix of the angles in the
        order's own axes and sums a dot product w first; the loader built
        them straight from the angles and broke the ties its own way, off by
        up to a half turn"""
        steps = (-180, -90, -45, 0, 45, 90, 180, 270)
        turns = list(itertools.product(steps, repeat=3))
        below = ((0, 0, 0), (90, 0, 0), (0, 90, 0), (45, 135, -90))
        blend = ((0, 25.0), (0, 150.0), (2, 25.0), (2, -50.0))
        frame = 1.0 / 24.0

        def fill(nodes, layers):
            source = FBX.FbxNode.EPivotSet.eSourcePivot
            orders = list(FBX.EFbxRotationOrder)[:6]
            for index, node in enumerate(nodes):
                kind, pose = divmod(index // 6, len(below))
                layer = layers[1 + kind]
                layer.BlendMode.Set(float(blend[kind][0]))
                layer.Weight.Set(blend[kind][1])
                node.SetRotationOrder(source, orders[index % 6])
                for axis in range(3):
                    curve = node.LclRotation.GetCurve(layers[0], "XYZ"[axis], True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, float(below[pose][axis]), kind="constant")
                    curve.KeyModifyEnd()
                    curve = node.LclRotation.GetCurve(layer, "XYZ"[axis], True)
                    curve.KeyModifyBegin()
                    for at, turn in enumerate(turns):
                        self.key(curve, at * frame, float(turn[axis]), kind="constant")
                    curve.KeyModifyEnd()

        path = self.take(fill, name="ties.fbx", joints=6 * len(below) * len(blend), layers=1 + len(blend))
        clip = ClipData.load_fbx(path, fps=24.0, start_frame=0, end_frame=len(turns) - 1)
        got, want = self.sampled(clip)[1], self.reference(path, clip)[1]
        orders = np.array(clip.rotate_order)

        for joint in range(got.shape[1]):
            with self.subTest(joint=joint):
                mine   = euler_to_matrix(np.radians(got[:, joint]), int(orders[joint]))
                theirs = euler_to_matrix(np.radians(want[:, joint]), int(orders[joint]))
                self.assertLess(float(np.abs(mine - theirs).max()), 1e-11)
                self.assertLess(self.turned(got[:1, joint], want[:1, joint]), 1e-9)

    def test_a_quarter_turn_layer_fading_in_blends_as_the_sdk_blends_it(self):
        """an additive layer turning (-90, 0, 90) by layer, its weight keyed
        from 0 to 100 over a second, a key a frame, over a base of 0: a turn
        of a third of a revolution, whose quaternion's sign the sdk's
        rounding picks, so on every frame in between the loader turned the
        other way round, off by up to 1.7 as a matrix. Every rotate order"""

        def fill(nodes, layers):
            self.by_layer(nodes, layers, [(1, 0, 0, (0, 0, 0), (-90, 0, 90))] * 6)
            curve = layers[1].Weight.GetCurve(layers[1], True)
            curve.KeyModifyBegin()
            for frame in range(25):
                self.key(curve, frame / 24.0, frame * 100.0 / 24.0, kind="linear")
            curve.KeyModifyEnd()

        path = self.take(fill, name="fade.fbx", joints=6, layers=2)
        clip, calls = self.counted(path, fps=24.0, start_frame=0, end_frame=24)
        got, want = self.sampled(clip)[1], self.reference(path, clip)[1]

        self.assertEqual(calls, [])
        self.rotation_against(got, want, np.array(clip.rotate_order), 1e-9)
        self.assertLess(self.turned(got, want), 1e-9)

    def test_keep_slope_carries_on_at_the_count_maya_writes(self):
        """keep slope with a count of 2**32 - 1, which Maya writes for every
        infinity it exports (linear infinity among them): the sdk carries the
        end key's slope on, the first key's left derivative before the keys
        and the last key's right one after them, in single precision; any
        other count holds the end key. Linear, user and auto tangents, before
        and after the keys, one side repeating or mirroring while the other
        carries on, plain and under an additive layer: the sdk's numbers bit
        for bit past the keys, where the loader raised ValueError"""
        extrapolation = FBX.FbxAnimCurve.EExtrapolationType
        forever       = 2**32 - 1
        slope         = (extrapolation.eKeepSlope, forever)
        sloped = (
            ("linear", "eTangentAuto", (0.0, 0.0), slope, slope),
            ("cubic", "eTangentUser", (5.0, -7.0), slope, slope),
            ("cubic", "eTangentAuto", (0.0, 0.0), slope, slope),
            ("cubic", "eTangentUser", (5.0, -7.0), (extrapolation.eKeepSlope, 2**32 - 2), (extrapolation.eKeepSlope, 3)),
            ("cubic", "eTangentUser", (5.0, -7.0), (extrapolation.eRepetition, 2), slope),
            ("linear", "eTangentAuto", (0.0, 0.0), slope, (extrapolation.eMirrorRepetition, 3)),
        )

        def fill(layered):
            def keyed(nodes, layers):
                for node, (kind, tangent, data, before, after) in zip(nodes, sloped):
                    curve = node.LclTranslation.GetCurve(layers[0], "X", True)
                    curve.KeyModifyBegin()
                    for when, value in ((0.25, 1.0), (0.5, 3.0), (0.75, 2.0)):
                        self.key(curve, when, value, kind=kind, tangent=tangent, data=data)
                    curve.KeyModifyEnd()
                    curve.SetPreExtrapolation(before[0])
                    curve.SetPreExtrapolationCount(before[1])
                    curve.SetPostExtrapolation(after[0])
                    curve.SetPostExtrapolationCount(after[1])
                    if layered:
                        curve = node.LclTranslation.GetCurve(layers[1], "X", True)
                        curve.KeyModifyBegin()
                        self.key(curve, 0.0, 2.0, kind="linear")
                        self.key(curve, 1.0, -3.0, kind="linear")
                        curve.KeyModifyEnd()
                        layers[1].Weight.Set(40.0)

            return keyed

        for layered in (False, True):
            with self.subTest(layered=layered):
                path = self.take(fill(layered), name=f"slope{int(layered)}.fbx", joints=6, layers=1 + layered)
                clip, calls = self.counted(path, fps=24.0, start_frame=-24, end_frame=48)
                got, want = self.sampled(clip)[0], self.reference(path, clip)[0]

                self.assertEqual(calls, [])
                self.assertLess(self.error(got[None], want[None]), 1e-5)
                # frames 6 and 18 hold the end keys, and frames 0 and 24 the
                # layer's: past them, on the sides that do not repeat, the
                # very same numbers
                frame  = np.arange(clip.frame_count) + clip.start_frame
                before = frame < (0 if layered else 6)
                after  = frame > (24 if layered else 18)
                for node, (_, _, _, pre, post) in enumerate(sloped):
                    for side, (kind, _) in ((before, pre), (after, post)):
                        if kind == extrapolation.eKeepSlope:
                            self.assertTrue(np.array_equal(got[side, node], want[side, node]), f"node {node}")
                if not layered:
                    # the slopes carry on, the other counts hold
                    self.assertEqual(got[frame == -6, 0, 0].tolist(),  [-3.0])
                    self.assertEqual(got[frame == 30, 0, 0].tolist(),  [0.0])
                    self.assertEqual(float(got[frame == 30, 1, 0][0]), float(np.float32(2.0 - 3.5)))
                    self.assertEqual(got[frame == 30, 3, 0].tolist(),  [2.0])
                    self.assertEqual(got[frame == -6, 3, 0].tolist(),  [1.0])
                    self.assertEqual(float(got[frame == 30, 4, 0][0]), float(np.float32(2.0 - 3.5)))
                    self.assertEqual(got[frame == -6, 5, 0].tolist(),  [-3.0])

    def test_a_take_maya_exported_with_linear_infinity_loads(self):
        """Maya's own export of spline curves with linear infinity, before
        and after the keys, plain and under an additive animation layer:
        Maya writes the infinity as keep slope with a count of 2**32 - 1,
        which the loader raised ValueError on. Now the sdk's numbers, and
        Maya's own values, on every frame. Needs mayapy with the fbxmaya
        plug-in"""
        try:
            import maya.cmds as cmds
            import maya.mel as mel

            cmds.loadPlugin("fbxmaya", quiet=True)
        except Exception:  # not running in Maya, or no fbx plug-in
            self.skipTest("needs Maya and its fbxmaya plug-in")

        frames = list(range(-12, 37))
        self.addCleanup(cmds.file, new=True, force=True)
        for layered in (False, True):
            with self.subTest(layered=layered):
                cmds.file(new=True, force=True)
                cmds.currentUnit(time="film")
                joints = []
                for index in range(3):
                    cmds.select(clear=True)
                    joints.append(cmds.joint(name=f"j{index}", position=(index, 0, 0)))
                    cmds.setAttr(f"{joints[-1]}.rotateOrder", index)
                for index, joint in enumerate(joints):
                    for frame, value in ((0, 1.0), (12, 4.0 + index), (24, 2.0)):
                        for attr in ("translateX", "rotateY", "scaleZ"):
                            cmds.setKeyframe(joint, attribute=attr, time=frame, value=value * (10.0 if attr == "rotateY" else 1.0))
                    for attr in ("translateX", "rotateY", "scaleZ"):
                        cmds.keyTangent(f"{joint}.{attr}", inTangentType="spline", outTangentType="spline")
                        cmds.setInfinity(f"{joint}.{attr}", preInfinite="linear", postInfinite="linear")
                if layered:
                    layer = cmds.animLayer("L1")
                    for joint in joints:
                        cmds.animLayer(layer, edit=True, attribute=f"{joint}.translateX")
                        for frame, value in ((0, 0.5), (24, -1.5)):
                            cmds.setKeyframe(joint, attribute="translateX", time=frame, value=value, animLayer=layer)
                    cmds.setAttr(f"{layer}.weight", 0.5)

                maya = np.zeros((len(frames), len(joints)))
                for row, frame in enumerate(frames):
                    for column, joint in enumerate(joints):
                        maya[row, column] = cmds.getAttr(f"{joint}.translateX", time=frame)

                path = os.path.join(self.tmp.name, f"maya_linear{int(layered)}.fbx").replace("\\", "/")
                mel.eval("FBXResetExport")
                mel.eval("FBXExportBakeComplexAnimation -v false")
                cmds.select(joints)
                mel.eval(f'FBXExport -f "{path}" -s')

                clip, calls = self.counted(path, fps=24.0, start_frame=frames[0], end_frame=frames[-1])
                got, want = self.sampled(clip), self.reference(path, clip)

                self.assertEqual(calls, [])
                self.assertLess(self.error(got, want), 1e-5)
                past = (np.array(frames) < 0) | (np.array(frames) > 24)
                self.assertTrue(np.array_equal(got[:, past], want[:, past]))
                # the infinity carried on, as Maya has it
                self.assertLess(float(np.abs(got[0, :, :, 0] - maya).max()), 1e-5)
                self.assertGreater(float(np.abs(got[0, past, :, 0] - got[0, frames.index(0), :, 0]).max()), 1.0)

    def crowded(self):
        """a take of weighted spans whose two handles crowd one end, each on
        its own joint's translate x: a weight of 0.0001, 0.001, 0.005 or 0.01
        on the first key's right handle or on the next key's left one,
        against 0.95 or 0.99 on the other, over five spans a second, half a
        second and two seconds long, rising, falling and overshooting their
        keys. Hands back the file and, joint by joint, (small weight, large
        weight, whether the small one is the next key's, span), each span
        its two keys as (seconds, value) and the first key's right
        derivative and the next key's left one"""
        spans = (
            ((0.0, 0.0), (1.0, 1.0), (0.0, 5.0)),
            ((0.0, 0.0), (1.0, 1.0), (3.0, -4.0)),
            ((0.25, -2.0), (0.75, 6.0), (40.0, -10.0)),
            ((0.0, 3.0), (2.0, -1.5), (-6.0, 1.0)),
            ((0.5, 10.0), (1.5, 10.5), (-30.0, 30.0)),
        )
        cases    = list(itertools.product((0.0001, 0.001, 0.005, 0.01), (0.95, 0.99), (False, True), spans))
        weighted = (FBX.FbxAnimCurveDef.EWeightedMode.eWeightedAll, 1.0 / 3.0, 1.0 / 3.0)

        def fill(nodes, layers):
            for node, (small, large, on_next, (first, last, slopes)) in zip(nodes, cases):
                curve = node.LclTranslation.GetCurve(layers[0], "X", True)
                curve.KeyModifyBegin()
                for when, value in (first, last):
                    self.key(curve, when, value, tangent="eTangentUser", weights=weighted)
                curve.KeySetRightDerivative(0, slopes[0])
                curve.KeySetLeftDerivative(1, slopes[1])
                curve.KeySetRightTangentWeight(0, large if on_next else small)
                curve.KeySetLeftTangentWeight(1, small if on_next else large)
                curve.KeyModifyEnd()

        return self.take(fill, name="crowded.fbx", joints=len(cases)), cases

    def bezier(self, path, clip):
        """(frames, joints) translate x of the clip's joints, a span of two
        keys each, as the closed-form weighted bezier of the keys the file
        holds gives it, written out here rather than taken from the loader.
        A span from (k0, v0) to (k1, v1), s seconds long, runs through (k0 +
        wr (k1 - k0), v0 + wr s d0) and (k1 - wl (k1 - k0), v1 - wl s d1), d0
        and wr the first key's right derivative and weight, d1 and wl the
        next key's left ones. A frame a fraction u of the way along the span
        is where the bezier's time, 3 wr t (1 - t)**2 + 3 (1 - wl) t**2 (1 -
        t) + t**3, is u: its one root t in 0..1, which numpy's roots solves,
        gives the value. The keys, and the frames before and after them,
        hold the keys' values"""
        from cgmath.formats._fbx_io import typed

        with hierarchy.hierarchy.open_fbx(path, FBX) as scene:
            stacks = FBX.FbxCriteria.ObjectType(FBX.FbxAnimStack.ClassId)
            layers = FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId)
            stack  = typed(scene.GetSrcObject(stacks, 0), FBX.FbxAnimStack, FBX)
            layer  = typed(stack.GetSrcObject(layers, 0), FBX.FbxAnimLayer, FBX)

            found  = {}

            def walk(node):
                found[node.GetName()] = node
                for index in range(node.GetChildCount()):
                    walk(node.GetChild(index))

            walk(scene.GetRootNode())

            second = FBX.FbxTime()
            second.SetSecondDouble(1.0)
            ticks = []
            for frame in range(clip.frame_count):
                time = FBX.FbxTime()
                time.SetSecondDouble((clip.start_frame + frame) / clip.fps)
                ticks.append(time.Get())
            ticks  = np.array(ticks)

            values = np.zeros((clip.frame_count, len(clip.name)))
            for column, name in enumerate(clip.name):
                curve = typed(found[name].LclTranslation.GetCurve(layer, "X"), FBX.FbxAnimCurve, FBX)
                k0, k1 = curve.KeyGetTime(0).Get(), curve.KeyGetTime(1).Get()
                v0, v1 = curve.KeyGetValue(0), curve.KeyGetValue(1)
                d0, d1 = curve.KeyGetRightDerivative(0), curve.KeyGetLeftDerivative(1)
                wr, wl = curve.KeyGetRightTangentWeight(0), curve.KeyGetLeftTangentWeight(1)
                s       = (k1 - k0) / second.Get()
                handles = np.array([v0, v0 + wr * s * d0, v1 - wl * s * d1, v1])

                values[:, column] = np.where(ticks <= k0, v0, v1)
                for row in np.flatnonzero((ticks > k0) & (ticks < k1)):
                    u      = (ticks[row] - k0) / (k1 - k0)
                    roots  = np.roots([3.0 * (wr + wl) - 2.0, 3.0 * (1.0 - wl) - 6.0 * wr, 3.0 * wr, -u])
                    inside = (np.abs(roots.imag) < 1e-9) & (roots.real > -1e-9) & (roots.real < 1.0 + 1e-9)
                    self.assertEqual(int(inside.sum()), 1, f"{name} at tick {ticks[row]}: {roots}")
                    t         = min(max(float(roots[inside].real[0]), 0.0), 1.0)
                    bernstein = np.array([(1.0 - t) ** 3, 3.0 * (1.0 - t) ** 2 * t, 3.0 * (1.0 - t) * t**2, t**3])

                    values[row, column] = bernstein @ handles
        return values

    def test_a_crowded_weighted_span_is_the_exact_bezier(self):
        """a weighted span whose two handles crowd one end, 80 of them
        (crowded): the clip is the closed-form weighted bezier of the file's
        keys (bezier) on every frame, within 1e-12 of the span's range, the
        values Maya's own import gives. The sdk's own solve for the bezier's
        time stops short there, more than a tenth of the range off at
        weights 0.0001 and 0.99 either way round, and the loader does not
        follow it"""
        path, cases = self.crowded()

        clip  = ClipData.load_fbx(path, fps=24.0, start_frame=0, end_frame=48)
        got   = self.sampled(clip)[0, :, :, 0]
        want  = self.bezier(path, clip)
        sdk   = self.reference(path, clip)[0, :, :, 0]
        names = list(clip.name)

        self.assertEqual(clip.frame_count, 49)
        for joint, (small, large, on_next, span) in enumerate(cases):
            column = names.index(f"j{joint}")
            size   = float(np.ptp(want[:, column]))
            with self.subTest(small=small, large=large, on_next=on_next, span=span):
                self.assertLess(float(np.abs(got[:, column] - want[:, column]).max()), 1e-12 * size)
                if (small, large) == (0.0001, 0.99):
                    self.assertGreater(float(np.abs(sdk[:, column] - want[:, column]).max()), 0.1 * size)

    def test_a_crowded_weighted_span_loads_as_maya_imports_it(self):
        """Maya's own FBXImport of the 80 crowded spans: its values on every
        frame are the clip's, within 1e-12 of the span's range, where the
        sdk's own evaluation is up to half the range off. Needs mayapy with
        the fbxmaya plug-in"""
        try:
            import maya.cmds as cmds
            import maya.mel as mel

            cmds.loadPlugin("fbxmaya", quiet=True)
        except Exception:  # not running in Maya, or no fbx plug-in
            self.skipTest("needs Maya and its fbxmaya plug-in")

        path, cases = self.crowded()
        clip  = ClipData.load_fbx(path, fps=24.0, start_frame=0, end_frame=48)
        got   = self.sampled(clip)[0, :, :, 0]
        names = list(clip.name)

        self.addCleanup(cmds.file, new=True, force=True)
        cmds.file(new=True, force=True)
        cmds.currentUnit(time="film")
        mel.eval("FBXResetImport")
        mel.eval(f'FBXImport -f "{path.replace(os.sep, "/")}"')

        frames = range(clip.start_frame, clip.start_frame + clip.frame_count)
        for joint, (small, large, on_next, span) in enumerate(cases):
            column = names.index(f"j{joint}")
            maya   = np.array([cmds.getAttr(f"j{joint}.translateX", time=frame) for frame in frames])
            with self.subTest(small=small, large=large, on_next=on_next, span=span):
                self.assertLess(float(np.abs(maya - got[:, column]).max()), 1e-12 * float(np.ptp(maya)))

    def test_an_axis_a_curve_node_holds_no_channel_for_keeps_the_layer_below(self):
        """an upper layer's curve node holding an X channel alone (an
        exporter can write one): the sdk keeps what the layers below hold on
        Y and Z, and the loader blended 0 there, scaling to 0 and translation
        pulled to it. Every blend mode, rotation by layer and by channel,
        scaling multiplied and added: as the sdk evaluates them, rotation by
        layer reading the missing angles as 0 as the sdk does"""
        modes = list(itertools.product((0, 1, 2), (0, 1), (0, 1)))

        def fill(nodes, layers):
            scene = nodes[0].GetScene()
            for node, layer, (mode, rotation, scaling) in zip(nodes, layers[1:], modes):
                layer.BlendMode.Set(float(mode))
                layer.Weight.Set(60.0)
                layer.RotationAccumulationMode.Set(float(rotation))
                layer.ScaleAccumulationMode.Set(float(scaling))
                for prop, values in (
                    (node.LclTranslation, (10.0, 20.0)),
                    (node.LclRotation, (10.0, 20.0)),
                    (node.LclScaling, (1.5, 2.5)),
                ):
                    curve = prop.GetCurve(layers[0], "Y", True)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, values[0], kind="linear")
                    self.key(curve, 1.0, values[1], kind="linear")
                    curve.KeyModifyEnd()

                    curve_node = FBX.FbxAnimCurveNode.Create(scene, "x only")
                    curve_node.AddChannel("X", 0.0)
                    layer.AddMember(curve_node)
                    prop.ConnectSrcObject(curve_node)
                    curve = FBX.FbxAnimCurve.Create(scene, "x")
                    curve_node.ConnectToChannel(curve, 0)
                    curve.KeyModifyBegin()
                    self.key(curve, 0.0, 1.5, kind="linear")
                    self.key(curve, 1.0, 2.5, kind="linear")
                    curve.KeyModifyEnd()

        path = self.take(fill, name="x_only.fbx", joints=len(modes), layers=1 + len(modes))
        clip, calls = self.counted(path, fps=24.0, start_frame=0, end_frame=24)
        got, want = self.sampled(clip), self.reference(path, clip)

        self.assertEqual(calls, [])
        for joint, (mode, rotation, scaling) in enumerate(modes):
            with self.subTest(mode=mode, rotation=rotation, scaling=scaling):
                self.assertLess(self.error(got[[0, 2], :, [joint]], want[[0, 2], :, [joint]]), 1e-5)
                orders = np.array(clip.rotate_order)[[joint]]
                self.rotation_against(got[1][:, [joint]], want[1][:, [joint]], orders, 1e-5)
        # the base's own Y, 15 at frame 12, kept under an override
        self.assertAlmostEqual(float(got[0, 12, 4, 1]), 15.0, places=5)

    def test_wrappers_of_a_closed_scene_leave_the_load_alone(self):
        """the fbx binding can hand back a live wrapper of an object of a
        scene already closed, of whatever type it was, for an object that
        sits where it did: load_fbx re-types the stacks, layers, curve nodes
        and curves it looks up, so holding every object of other scenes
        across a load leaves it as a clean load, where the stack lookup
        raised TypeError"""
        first  = self.take(self.random_curves, name="first.fbx", joints=6)
        second = self.take(lambda nodes, made: self.layered(np.random.default_rng(1), nodes, made, True), name="second.fbx", joints=3, layers=3)
        clean  = {path: self.sampled(ClipData.load_fbx(path, fps=24.0)) for path in (first, second)}

        held   = []
        for trial in range(6):
            source, target = (first, second) if trial % 2 else (second, first)
            with hierarchy.hierarchy.open_fbx(source, FBX) as scene:
                held.append([scene.GetSrcObject(index) for index in range(scene.GetSrcObjectCount())])
            with self.subTest(trial=trial):
                got = self.sampled(ClipData.load_fbx(target, fps=24.0))
                self.assertTrue(np.array_equal(got, clean[target], equal_nan=True))


class TestFbxWholeFrames(unittest.TestCase):
    """fbx times to frames, a time a hair off a whole frame put on it"""

    PER_SECOND = 141120000  # the sdk's ticks per second; any count works

    def frames(self, ticks, fps=24.0):
        return hierarchy.hierarchy._fbx_whole_frames(ticks, fps, self.PER_SECOND).tolist()

    def test_a_whole_frame_comes_back_whole(self):
        frame = self.PER_SECOND // 24
        self.assertEqual(self.frames([60 * frame, 121 * frame]), [60.0, 121.0])

    def test_float_noise_is_put_on_the_frame(self):
        # a tick is 1.7e-7 of a frame at 24 fps
        frame = self.PER_SECOND // 24
        self.assertEqual(self.frames([-2, 24 * frame + 2]), [0.0, 24.0])

    def test_a_time_further_off_stays_where_it_is(self):
        frame  = self.PER_SECOND // 24
        got    = self.frames([12 * frame + 12, 24 * frame - frame // 4])
        wanted = [12.0 + 12 / frame, 23.75]
        self.assertTrue(np.allclose(got, wanted, rtol=0.0, atol=1e-12))
        self.assertNotEqual(got[0], 12.0)


@unittest.skipIf(FBX is None, "fbx sdk is not installed")
class TestFbxFrameRate(unittest.TestCase):
    """the sdk reports one of its own time modes as having no frame rate"""

    def rate(self, mode):
        return _fbx_frame_rate(getattr(FBX.FbxTime.EMode, mode))

    def test_the_ordinary_modes_come_straight_from_the_sdk(self):
        self.assertEqual(self.rate("eFrames24"), 24.0)
        self.assertEqual(self.rate("eFrames30"), 30.0)
        self.assertEqual(self.rate("ePAL"),      25.0)

    def test_thirty_drop_is_repaired(self):
        """GetFrameRate hands back 0.0 for it, which would divide by zero"""
        self.assertEqual(FBX.FbxTime.GetFrameRate(FBX.FbxTime.EMode.eFrames30Drop), 0.0)

        self.assertAlmostEqual(self.rate("eFrames30Drop"), 30000.0 / 1001.0, places=6)

    def test_thirty_drop_names_the_same_rate_as_ntsc_drop(self):
        self.assertEqual(self.rate("eFrames30Drop"), self.rate("eNTSCDropFrame"))