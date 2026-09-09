import os
import tempfile
import unittest

import numpy as np
from cgmath.formats.glb import pygltflib
from transforms import euler_to_quaternion, quaternion_slerp
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath.hierarchy import _fbx_frame_rate, FBX

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
        clip = ClipData(self.rig, frames=3)
        clip.frame = 1
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

        values = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)
        clip.frames.translate = values
        self.assertTrue(np.allclose(clip.frames.translate, values))

        clip.frame = 2
        self.assertTrue(np.allclose(clip.translate, values[2]))

    def test_frames_setitem_stores_a_pose(self):
        clip = ClipData(self.rig, frames=3)

        pose = self.rig.copy()
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
            pose = self.rig.copy()
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
        clip = ClipData(self.rig, frames=3)
        clip.frames.translate = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)

        clip.frame            = -1
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
        clip = ClipData(self.rig, frames=3)
        clip.frames.translate = np.full((3, 4, 3), 2.0)

        other = clip.copy()
        other.frames.translate = np.full((3, 4, 3), 8.0)

        self.assertTrue(np.allclose(clip.frames.translate, 2.0))
        self.assertTrue(np.allclose(other.frames.translate, 8.0))
        self.assertEqual(other.frame_count, 3)

    def test_copy_keeps_every_frame_distinct(self):
        # a copy that lost its column list would reseed each frame from the
        # loaded pose on the next realign, which a uniform block cannot show
        clip  = ClipData(self.rig, frames=4)
        block = np.arange(4 * 4 * 3, dtype=float).reshape(4, 4, 3)
        clip.frames.translate = block
        clip.frame            = 2

        other = clip.copy()

        self.assertTrue(np.array_equal(other.frames.translate, block))
        self.assertEqual(other.frame, 2)

    def test_equality_discriminates_on_animation(self):
        a = ClipData(self.rig, frames=3)
        b = a.copy()
        self.assertEqual(a, b)

        b.frames.rotate = np.full((3, 4, 3), 45.0)
        self.assertNotEqual(a, b)

    def test_round_trip(self):
        clip = ClipData(self.rig, frames=4, start_frame=1001, fps=30.0)
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
        clip = ClipData(self.rig, frames=2)
        clip.frames[1].translate = np.full((4, 3), 5.0)

        clip.frame = 0
        first = clip.world_matrix.copy()
        clip.frame = 1
        second = clip.world_matrix.copy()

        self.assertFalse(np.allclose(first, second))
        clip.frame = 0
        self.assertTrue(np.allclose(clip.world_matrix, first))

    def test_structural_channels_are_shared_by_every_frame(self):
        clip = ClipData(self.rig, frames=3)
        clip.frame = 0
        clip["j1"].rotate_order = 3

        clip.frame = 2
        self.assertEqual(clip["j1"].rotate_order, 3)


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
            clip.frame = frame
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
            clip.frame = frame
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
        off = ClipData(rig, frames=2).frames.world_matrix

        self.assertFalse(np.allclose(on[:, 1], off[:, 1]))

    def test_batched_follows_a_block_edit(self):
        clip      = self._animated(frames=4, count=6)
        before    = clip.frames.world_matrix.copy()

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
        rig         = self._rig()
        clip        = self._animated(rig)

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
        rig            = self._rig()
        clip           = self._animated(rig)

        with_translate = clip.get_delta(rig, translate=True)
        without        = clip.get_delta(rig, translate=False)

        self.assertEqual(without.frame_count, clip.frame_count)
        self.assertFalse(
            np.allclose(with_translate.frames.translate, without.frames.translate)
        )


class TestClipScopedFrames(unittest.TestCase):
    """the frame axis narrowed to some nodes, or to a range of frames"""

    def setUp(self):
        super().setUp()
        self.clip = ClipData(chain(4), frames=6)
        block = np.arange(6 * 4 * 3, dtype=float).reshape(6, 4, 3)
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
        self.clip.frame = 3
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
        cut = self.clip.frames[0:3]
        cut.frames.translate = np.zeros((3, 4, 3))

        self.assertTrue(np.array_equal(self.clip.frames.translate, self.block))

    def test_a_cut_survives_a_realign(self):
        # the columns have to come across, or the first realign flattens it
        cut = self.clip.frames[1:4]
        cut.frame = 2

        self.assertTrue(np.array_equal(cut.frames.translate, self.block[1:4]))

    def test_a_range_can_be_pasted_from_another_clip(self):
        source = ClipData(chain(4), frames=3)
        source.frames.translate = np.full((3, 4, 3), -1.0)

        self.clip.frames[1:4] = source

        got = self.clip.frames.translate
        self.assertTrue(np.array_equal(got[1:4], np.full((3, 4, 3), -1.0)))
        self.assertTrue(np.array_equal(got[0], self.block[0]))
        self.assertTrue(np.array_equal(got[4:], self.block[4:]))

    def test_a_range_can_be_filled_with_one_pose(self):
        pose = chain(4)
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

        self.clip = ClipData(uuid_chain(3), frames=10, start_frame=1001, fps=24.0)
        rotate = np.array(self.clip.frames.rotate)
        rotate[:, 1, 2] = np.arange(10) * 10.0
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

        gltf = GLTF2()
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

        gltf   = GLTF2()
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
        clip      = ClipData.load_glb(self.build(target=0), scale_factor=1.0, fps=4.0)

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
        clip      = ClipData.load_glb(self.build(target=1), scale_factor=1.0, fps=4.0)

        names     = [str(node.name) for node in clip]
        translate = np.array(clip.frames.translate)

        self.assertAlmostEqual(
            float(translate[-1, names.index("mid"), 0]), 7.0, places=4
        )


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
            parent = node

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

        clip      = ClipData.load_fbx(path, fps=4.0)

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