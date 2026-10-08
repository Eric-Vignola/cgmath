import contextlib
import copy
import json as std_json
import os
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
from cgmath.transforms._numba._euler import MAYA_EA
from cgmath import hierarchy
from cgmath.formats.glb import pygltflib
from cgmath.transforms import (
    euler_filter,
    euler_to_quaternion,
    quaternion_slerp,
    quaternion_to_matrix,
)
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList
from cgmath.hierarchy import _fbx_frame_rate, FBX
from cgmath.hierarchy.hierarchy import _rotation

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
    return HierarchyData(
        [
            TransformData(
                name        = f"j{i}",
                parent_node = None if i == 0 else f"j{i - 1}",
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
        tree["_blocks"]["_translate"][0][0] = [0.0, 0.0]
        with open(path, "w", encoding="utf-8") as f:
            std_json.dump(tree, f)
        with self.assertRaisesRegex(ValueError, "ClipData block _translate must have shape"):
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
        bad = {
            "a channel short": ("_scale", np.ones((3, 4, 2))),
            "a node short":    ("_rotate", np.ones((3, 3, 3))),
            "no frames":       ("_rotate", np.ones((0, 4, 3))),
            "a frame short":   ("_scale", np.ones((2, 4, 3))),
            "an unknown name": ("_shear", np.ones((3, 4, 3))),
        }
        for label, (channel, block) in bad.items():
            with self.subTest(label):
                tree                     = self.clip.to_dict()
                tree["_blocks"]          = dict(tree["_blocks"])
                tree["_blocks"][channel] = block
                with self.assertRaises(ValueError):
                    ClipData.from_dict(tree)

        tree            = self.clip.to_dict()
        tree["_blocks"] = {"_scale": tree["_blocks"]["_scale"]}
        with self.assertRaises(ValueError):
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
    return HierarchyData(
        [
            TransformData(
                name         = f"j{i}",
                node_type    = "joint",
                parent_node  = None if i == 0 else f"j{i - 1}",
                translate    = [0.0, float(i), 0.0],
                rotate       = [float(i) * 7.0, float(i) * 5.0, float(i) * 3.0],
                joint_orient = [0.0, float(i) * 2.0, 0.0],
            )
            for i in range(count)
        ]
    )


def uuid_chain(count=4, root=False):
    """a chain parented through set_parent, so get_children() sees children

    joint_chain() sets parent_node to a name, which get_children compares
    against uuid, so nothing there ever has a child. The paths that restore
    children need a rig that really has them.
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
    """rig level edits refuse an animated clip; the rest still work"""

    def _varied_clip(self, rig, frames=3):
        clip   = ClipData(rig, frames=frames)
        rotate = clip.frames.rotate.copy()
        for frame in range(frames):
            rotate[frame] += frame * 11.0
        clip.frames.rotate = rotate
        return clip

    # --- refused on an animated clip --- #
    def test_swapaxes_refuses_an_animated_clip(self):
        clip = self._varied_clip(uuid_chain(4))
        with self.assertRaises(RuntimeError):
            clip.swapaxes(0, 1, negate=True)

    def test_set_rotate_to_joint_orient_refuses_an_animated_clip(self):
        clip = self._varied_clip(uuid_chain(4))
        with self.assertRaises(RuntimeError):
            clip.set_rotate_to_joint_orient()

    def test_set_joint_orient_to_rotate_refuses_an_animated_clip(self):
        clip = self._varied_clip(uuid_chain(4))
        with self.assertRaises(RuntimeError):
            clip.set_joint_orient_to_rotate()

    def test_node_type_refuses_an_animated_clip(self):
        # joints with a joint orient would need rotate rebuilt per frame;
        # nothing changes, not even the joint without one
        clip = self._varied_clip(uuid_chain(4))
        with self.assertRaises(RuntimeError):
            clip.node_type = "transform"
        self.assertEqual(clip.node_type, ["joint"] * 4)

    def test_set_parent_in_world_space_refuses_an_animated_clip(self):
        clip = self._varied_clip(uuid_chain(4, root=True))
        with self.assertRaises(RuntimeError):
            clip.set_parent("root")

    def test_a_selection_and_a_single_node_refuse_too(self):
        # every view form funnels into the TransformData methods, so one
        # guard there covers the slice, the match and the bare node
        clip = self._varied_clip(uuid_chain(4))

        with self.assertRaises(RuntimeError):
            clip[:2].swapaxes(0, 1)
        with self.assertRaises(RuntimeError):
            clip.match("j1").swapaxes(0, 1)
        with self.assertRaises(RuntimeError):
            clip["j1"].swapaxes(0, 1)
        with self.assertRaises(RuntimeError):
            clip["j2"].set_parent(None)

    def test_the_refusal_leaves_the_clip_untouched(self):
        clip    = self._varied_clip(uuid_chain(4, root=True))
        before  = np.array(clip.frames.rotate)
        parents = [n.parent_node for n in clip]

        with self.assertRaises(RuntimeError):
            clip.set_parent("root")

        self.assertTrue(np.array_equal(clip.frames.rotate, before))
        self.assertEqual([n.parent_node for n in clip], parents)

    # --- still allowed --- #
    def test_a_single_frame_clip_still_allows_them(self):
        # one frame has no curve to break
        clip   = ClipData(uuid_chain(4), frames=1)
        before = clip.world_matrix.copy()

        clip.swapaxes(0, 1, negate=True)
        clip.set_rotate_to_joint_orient()

        self.assertTrue(np.allclose(clip.rotate, 0.0, atol=1e-9))
        self.assertFalse(np.allclose(clip.world_matrix, before))

    def test_a_frameless_clip_still_allows_them(self):
        clip = ClipData(uuid_chain(4))
        self.assertEqual(clip.frame_count, 0)
        clip.swapaxes(0, 1, negate=True)
        clip.set_joint_orient_to_rotate()
        self.assertTrue(np.allclose(clip.joint_orient, 0.0, atol=1e-9))

    def test_a_plain_hierarchy_is_untouched(self):
        # the guard reads frame_count, which only a ClipData has
        rig    = uuid_chain(4, root=True)
        before = rig.world_matrix.copy()

        rig[:2].swapaxes(0, 1)
        rig["j2"].set_parent("root")
        rig.match("j3").set_rotate_to_joint_orient()

        self.assertFalse(np.allclose(rig.world_matrix, before))

    def test_a_transform_is_not_refused_by_the_orient_helpers(self):
        # they are a no-op off a joint, so there is nothing to refuse
        rig  = HierarchyData([TransformData(name="grp", node_type="transform")])
        clip = ClipData(rig, frames=4)
        clip.set_rotate_to_joint_orient()
        clip.set_joint_orient_to_rotate()

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


class TestClipEvaluator(unittest.TestCase):
    """the batched evaluator must agree with scrubbing, exactly"""

    def _hard_rig(self, count=12):
        # every feature that makes the composition non trivial: varied
        # rotate_order, mixed segment_scale_compensate, non zero joint_orient
        # and rotate_axis, non unit scale, deep and shallow chains
        rng   = np.random.default_rng(7)
        nodes = []
        for i in range(count):
            parent = None
            if i:
                parent = f"j{rng.integers(0, i)}"
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    parent_node=parent,
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
                    parent_node="a",
                    translate=[1.0, 1.0, 1.0],
                    segment_scale_compensate=True,
                ),
            ]
        )
        on = ClipData(rig, frames=2).frames.world_matrix

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
        return HierarchyData(
            [
                TransformData(
                    name         = f"j{i}",
                    node_type    = "joint",
                    parent_node  = None if i == 0 else f"j{i - 1}",
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


def _parent_by_uuid(rig):
    """points every parent_node at the parent's uuid, the form every file
    reader writes"""
    for node in rig:
        parent = node.get_parent()
        if isinstance(parent, TransformData):
            node.parent_node = parent.uuid
    return rig


class TestClipDeltaVectorised(unittest.TestCase):
    """clip pose algebra solves every frame at once; it must agree with the
    single pose HierarchyData.get_delta / add_delta run on each frame"""

    by_uuid = False  # parent_node holds names; a subclass points it at uuids

    def _rig(self, names, parents, seed):
        rng = np.random.default_rng(seed)
        rig = HierarchyData(
            [
                TransformData(
                    name         = name,
                    node_type    = "joint",
                    parent_node  = parent,
                    translate    = rng.normal(size=3),
                    rotate       = rng.uniform(-90.0, 90.0, size=3),
                    joint_orient = rng.uniform(-30.0, 30.0, size=3),
                    rotate_axis  = rng.uniform(-20.0, 20.0, size=3),
                    rotate_order = int(rng.integers(0, 6)),
                    # mixed: compensation is the one solve that reads the
                    # parent's solved scale
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
                for name, parent in zip(names, parents)
            ]
        )
        return _parent_by_uuid(rig) if self.by_uuid else rig

    def _chain(self, count, seed):
        names = [f"j{i}" for i in range(count)]
        return self._rig(names, [None] + names[:-1], seed)

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

    def _per_frame(self, method, clip, other, **kwargs):
        """the reference: the single pose get_delta / add_delta on every frame,
        against one pose or frame for frame against a clip"""
        poses = []
        for frame in range(clip.frame_count):
            clip.frame = frame
            if isinstance(other, ClipData):
                other.frame = frame
            poses.append(getattr(HierarchyData, method)(clip, other, **kwargs))
        return ClipData.from_poses(poses, start_frame=clip.start_frame, fps=clip.fps)

    def _frozen(self, clip, frame):
        """the whole clip at one frame, as a pose of its own"""
        clip.frame = frame
        return HierarchyData([node.copy() for node in clip])

    def _assert_same(self, fast, slow):
        # exact: the same algebra, step for step, gives the same bits
        self.assertEqual(fast.name, slow.name)
        self.assertEqual(fast.uuid, slow.uuid)
        self.assertEqual([n.parent_node for n in fast], [n.parent_node for n in slow])
        self.assertEqual(fast.frame_count, slow.frame_count)
        self.assertEqual(fast.start_frame, slow.start_frame)
        self.assertEqual(fast.fps, slow.fps)
        for channel in ("rotate", "translate", "scale", "joint_orient", "rotate_axis"):
            np.testing.assert_array_equal(
                getattr(fast.frames, channel), getattr(slow.frames, channel), channel
            )
        np.testing.assert_array_equal(fast.frames.world_matrix, slow.frames.world_matrix)

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
        # a clip of part of a rig: j1 names a parent the clip lacks. The
        # delta, like a copy of a selection, makes such a node a root
        clip = self._animated(ClipData(self._chain(5, seed=5)[1:]))
        pose = self._chain(5, seed=17)[1:]
        self.assertIsInstance(clip["j1"].get_parent(), str)

        delta = clip.get_delta(pose)
        self.assertIsNone(delta["j1"].parent_node)
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


class TestClipDeltaVectorisedByUuid(TestClipDeltaVectorised):
    """the same, parented by uuid as a file reader writes a rig"""

    by_uuid = True


class TestDeltaSolveOrder(unittest.TestCase):
    """a paired node under an unpaired one must still solve after its paired
    ancestors, or the ancestor moves afterwards and drags it off target"""

    by_uuid        = False  # parent_node holds names; a subclass points it at uuids
    children_first = False  # a subclass lists every rig children first

    def _rig(self, names, parents, seed):
        rng = np.random.default_rng(seed)
        nodes = [
            TransformData(
                name         = name,
                node_type    = "joint",
                parent_node  = parent,
                translate    = rng.normal(size=3),
                rotate       = rng.uniform(-90.0, 90.0, size=3),
                joint_orient = rng.uniform(-30.0, 30.0, size=3),
            )
            for name, parent in zip(names, parents)
        ]
        rig = HierarchyData(nodes[::-1] if self.children_first else nodes)
        return _parent_by_uuid(rig) if self.by_uuid else rig

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


class TestDeltaSolveOrderByUuid(TestDeltaSolveOrder):
    """the same, parented by uuid as a file reader writes a rig"""

    by_uuid = True


class TestDeltaSolveOrderChildrenFirst(TestDeltaSolveOrder):
    """the same with every rig listing children ahead of their parents: the
    solve order comes from the hierarchy, never from the list"""

    children_first = True


class TestClipBranchResets(unittest.TestCase):
    """a write on the bound frame clears the cached world matrices of the
    written node's whole branch, so every read after it, and every node
    solved after it, sees the new pose. The reference is the same write made
    on a single pose copy of that frame, its world matrices then computed
    from scratch"""

    def _rig(self, by_name, count=14, seed=3):
        rng   = np.random.default_rng(seed)
        uuids = [hierarchy.generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            parent = None
            if i:
                p      = int(rng.integers(max(0, i - 4), i))
                parent = f"j{p}" if by_name else uuids[p]
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    uuid=uuids[i],
                    parent_node=parent,
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
        for by_name in (False, True):
            for label, write in self.WRITES.items():
                with self.subTest(label, by_name=by_name):
                    clip  = self._clip(self._rig(by_name))
                    other = self._clip(self._rig(by_name), seed=9)
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
        for by_name in (False, True):
            for label in ("world_matrix", "match_matrix"):
                with self.subTest(label, by_name=by_name):
                    clip   = self._clip(self._rig(by_name))
                    other  = self._clip(self._rig(by_name), seed=9)
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

        rig       = self._rig(by_name=False, count=40)
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
        "parent_node to the world": lambda h: setattr(h["j6"], "parent_node", None),
        "parent_node by uuid": lambda h: setattr(h["j7"], "parent_node", h["j1"].uuid),
        "del": lambda h: h.__delitem__(3),
    }

    def test_a_structural_edit_matches_the_single_pose_edit(self):
        for by_name in (False, True):
            for label, edit in self.EDITS.items():
                with self.subTest(label, by_name=by_name):
                    clip  = self._clip(self._rig(by_name))
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
        # mixed segment scale compensation; parented by name and by uuid
        rng   = np.random.default_rng(seed)
        nodes = []
        for i in range(count):
            parent = None
            if i:
                p      = int(rng.integers(0, i))
                parent = f"j{p}" if i % 2 else nodes[p].uuid
            nodes.append(
                TransformData(
                    name=f"j{i}",
                    node_type="joint",
                    parent_node=parent,
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
                    parent_node  = nodes[-1].uuid if nodes else None,
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
                    parent_node  = nodes[-1].uuid if nodes else None,
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
                    parent_node  = nodes[-1].uuid if nodes else None,
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

    def _rig(self, count=14, seed=7, by_name=True):
        # every feature a factor reads: rotate orders, segment scale
        # compensation both ways, a branching tree, parented by name or uuid
        rng   = np.random.default_rng(seed)
        uuids = [hierarchy.generate_uuid() for _ in range(count)]
        names = [f"j{i}" for i in range(count)]
        nodes = []
        for i in range(count):
            parent = None
            if i:
                p      = int(rng.integers(0, i))
                parent = names[p] if by_name else uuids[p]
            nodes.append(
                TransformData(
                    name=names[i],
                    uuid=uuids[i],
                    node_type="joint",
                    parent_node=parent,
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
        for by_name in (True, False):
            yield ("name" if by_name else "uuid"), self._animated(self._rig(by_name=by_name))

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
        # a clip of part of a rig: the first node's parent is left behind,
        # which get_parent_matrix answers with identity and a scale of one
        rig     = self._rig(by_name=False)
        clip    = self._animated(HierarchyData([node.copy() for node in rig][3:]))
        orphans = [i for i, node in enumerate(clip) if isinstance(node.get_parent(), str)]
        self.assertTrue(orphans)

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
                    clip = self._animated(self._rig(by_name=False))
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
        clip = self._animated(self._rig(by_name=False))
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
                    parent_node=nodes[int(rng.integers(0, i))].uuid if i else None,
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
    ):
        """a chain of joints, keys given as {joint: {channel: [(sec, value)]}}

        Only ``keyed_stack`` is keyed, so selecting any other take has to show
        up as a node that no longer moves. Every stack still gets the same
        time span, which is what makes an unkeyed one look plausible.
        """
        manager = FBX.FbxManager.Create()
        manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
        scene = FBX.FbxScene.Create(manager, "scene")
        scene.GetGlobalSettings().SetTimeMode(getattr(FBX.FbxTime.EMode, time_mode))

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
                for joint, curves in (keys or {}).items():
                    for channel, pairs in curves.items():
                        prop, axis = self.CHANNELS[channel]
                        curve = getattr(nodes[joint], prop).GetCurve(layer, axis, True)
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