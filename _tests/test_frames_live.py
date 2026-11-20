"""The pose of one frame of a clip is live.

``clip.at(n)`` is a HierarchyData that is a view of that frame: its nodes
read the clip's values for the frame, a write to one lands in the clip, on
that frame alone, and ``copy()`` gives a pose of its own. What every frame
shares, a name, a parent, a rotate order, is the clip's.
"""

import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath._tests.test_views import chain, pick, walk
from cgmath.hierarchy import ClipData, HierarchyData, TransformData


def column(clip, name, channel, axis):
    """one node's component of ``channel`` on every frame, (F,)"""
    place = [node.name for node in clip].index(name)
    return np.array(getattr(clip.frames, channel))[:, place, axis]


class TestLivePose(unittest.TestCase):
    def test_it_is_a_view_of_the_frame(self):
        clip = walk()
        pose = clip.at(3)
        self.assertIsInstance(pose, HierarchyData)
        self.assertNotIsInstance(pose, ClipData)
        self.assertTrue(pose.is_view)
        self.assertIs(pose.owner, clip)
        self.assertEqual([node.name for node in pose], [node.name for node in clip])
        self.assertEqual([node.parent_name for node in pose], [node.parent_name for node in clip])
        self.assertTrue(all(isinstance(node, TransformData) for node in pose))
        self.assertEqual(repr(pose), "HierarchyData(['root', 'props', 'spine', 'neck', 'head', 'arm'], view)")
        # one object a frame while it is held; the clip stays where it was
        self.assertIs(clip.at(3), pose)
        self.assertIsNot(clip.at(4), pose)
        self.assertEqual(clip.frame, 0)

    def test_a_frame_number_is_read_as_at_always_read_it(self):
        clip                          = ClipData(chain(), frames=4, start_frame=1001, fps=30.0)
        clip.frames["spine"].rotate_x = [0.0, 10.0, 20.0, 30.0]
        for number, turn in ((1001, 0.0), (1003, 20.0), (990, 0.0), (2000, 30.0)):
            self.assertEqual(float(clip.at(number)["spine"].rotate[0]), turn)
        with self.assertRaises(ValueError):
            ClipData(chain()).at(0)

    def test_it_reads_the_frame(self):
        clip = walk()
        for frame in (0, 3, 5):
            pose = clip.at(frame)
            np.testing.assert_allclose(np.asarray(pose.translate), np.asarray(clip.frames.translate)[frame])
            np.testing.assert_allclose(np.asarray(pose.rotate), np.asarray(clip.frames.rotate)[frame])
            np.testing.assert_allclose(np.asarray(pose.world_matrix), np.asarray(clip.frames.world_matrix)[frame], atol=1e-12)

    def test_a_write_lands_on_its_frame_alone(self):
        clip                     = walk()
        pose                     = clip.at(3)
        pose["neck"].translate_y = 9.0
        np.testing.assert_array_equal(column(clip, "neck", "translate", 1), [3.0, 3.0, 3.0, 9.0, 3.0, 3.0])
        # a constant channel written on one frame is animated now
        np.testing.assert_array_equal(clip.keyed, [False, False, True, True, False, True])
        # a write the node solves: a world matrix
        world                     = np.array(pose["head"].world_matrix)
        world[3, 0]               = 7.0
        pose["head"].world_matrix = world
        np.testing.assert_allclose(np.asarray(clip.frames["head"].world_matrix)[3, 3, 0], 7.0, atol=1e-9)
        self.assertFalse(np.isclose(np.asarray(clip.frames["head"].world_matrix)[2, 3, 0], 7.0))
        # a write on the whole pose
        before           = np.array(clip.frames.translate_x)
        pose.translate_x = np.asarray(pose.translate_x) + 1.0
        after            = np.array(clip.frames.translate_x)
        np.testing.assert_allclose(after[3], before[3] + 1.0)
        np.testing.assert_allclose(np.delete(after, 3, axis=0), np.delete(before, 3, axis=0))

    def test_two_frames_are_held_at_once(self):
        clip = walk()
        early, late   = clip.at(1), clip.at(4)
        early["arm"].translate_x = -5.0
        late["arm"].translate_x  = 5.0
        np.testing.assert_array_equal(column(clip, "arm", "translate", 0), [3.0, -5.0, 3.0, 3.0, 5.0, 3.0])
        self.assertEqual((float(early["arm"].translate[0]), float(late["arm"].translate[0])), (-5.0, 5.0))
        # each reads its own frame of an animated channel
        self.assertEqual((float(early["spine"].rotate[0]), float(late["spine"].rotate[0])), (5.0, 20.0))

    def test_the_clip_reads_what_a_pose_wrote(self):
        clip = walk()
        clip.frame = 3
        before = np.array(clip["head"].world_matrix)
        pose   = clip.at(3)
        pose["neck"].translate_y = float(pose["neck"].translate[1]) + 2.0
        # the clip's own node, on that frame, and every frame read at once
        moved = np.array(clip["head"].world_matrix)
        self.assertFalse(np.allclose(moved, before))
        np.testing.assert_allclose(moved, np.asarray(pose["head"].world_matrix), atol=1e-12)
        np.testing.assert_allclose(moved, np.asarray(clip.frames["head"].world_matrix)[3], atol=1e-12)

    def test_a_pose_reads_what_the_clip_wrote(self):
        clip   = walk()
        pose   = clip.at(2)
        before = np.array(pose["head"].world_matrix)
        # every frame at once
        moved = np.array(clip.frames.translate)
        moved[:, :, 0] += 4.0
        clip.frames.translate = moved
        np.testing.assert_allclose(np.asarray(pose.translate), moved[2])
        self.assertFalse(np.allclose(np.asarray(pose["head"].world_matrix), before))
        np.testing.assert_allclose(np.asarray(pose.world_matrix), np.asarray(clip.frames.world_matrix)[2], atol=1e-12)
        # what every frame shares, changed on the clip: the pose held follows
        clip.set_rotate_order("zxy")
        self.assertEqual([int(order) for order in pose.rotate_order], [int(order) for order in clip.rotate_order])
        np.testing.assert_allclose(np.asarray(pose.rotate), np.asarray(clip.frames.rotate)[2], atol=1e-9)
        np.testing.assert_allclose(np.asarray(pose.world_matrix), np.asarray(clip.frames.world_matrix)[2], atol=1e-9)
        clip["arm"].set_parent(clip["neck"], world_space=True)
        self.assertEqual(pose["arm"].parent_name, "neck")
        self.assertEqual([node.name for node in pose], [node.name for node in clip])
        np.testing.assert_allclose(np.asarray(pose.world_matrix), np.asarray(clip.frames.world_matrix)[2], atol=1e-9)

    def test_what_every_frame_shares_is_the_clip_s(self):
        clip = walk()
        pose = clip.at(1)
        node = pose["neck"]
        for name, write in (
            ("rotate_order", lambda: setattr(node, "rotate_order", 2)),
            ("name", lambda: setattr(node, "name", "x")),
            ("node_type", lambda: setattr(node, "node_type", "transform")),
            ("set_parent()", lambda: node.set_parent(pose["root"])),
            ("swapaxes()", lambda: node.swapaxes(0, 1)),
            ("set_rotate_order()", lambda: node.set_rotate_order("zxy")),
            ("append()", lambda: pose.append(TransformData("more"))),
            ("pop()", lambda: pose.pop(0)),
            ("renew_uuids()", lambda: pose.renew_uuids()),
            ("set_parent()", lambda: pose.match("ne*").set_parent("root")),
            ("a rig edit", lambda: pose.match("ne*").set_rotate_order("zxy")),
        ):
            with self.subTest(write=name), self.assertRaises((AttributeError, TypeError)) as caught:
                write()
            self.assertIn(f"{name} changes what every frame of a clip shares", str(caught.exception))
        self.assertEqual(([n.name for n in clip], [int(o) for o in clip.rotate_order]), ([n.name for n in walk()], [0] * 6))
        # a name no node has is refused as a TransformData refuses it
        with self.assertRaises(AttributeError) as caught:
            node.nope = 1
        self.assertIn("'TransformData' has no attribute 'nope'", str(caught.exception))

    def test_copy_is_a_pose_of_its_own(self):
        clip = walk()
        pose = clip.at(5)
        path = os.path.join(tempfile.mkdtemp(), "pose.json")
        pose.save(path)
        for made in (pose.copy(), pickle.loads(pickle.dumps(pose)), HierarchyData.load(path)):
            self.assertIs(type(made), HierarchyData)
            self.assertFalse(made.is_view)
            self.assertTrue(all(type(node) is TransformData for node in made))
            self.assertEqual(float(made["spine"].rotate[0]), 25.0)
            self.assertEqual([node.parent_name for node in made], [node.parent_name for node in clip])
            made["spine"].rotate_x = 1.0
            made["spine"].name     = "back"
        np.testing.assert_allclose(column(clip, "spine", "rotate", 0), np.linspace(0.0, 25.0, 6))
        single = pose["arm"].copy()
        self.assertIs(type(single), TransformData)
        single.translate_y = 0.0
        self.assertEqual(float(clip.frames["arm"].translate_y[5]), 3.0)

    def test_a_selection_of_a_pose_writes_the_frame(self):
        clip = walk()
        pose = clip.at(2)
        part = pose.match("ne*", "he*")
        self.assertTrue(part.is_view and part.owner is pose)
        part.translate_z = 6.0
        np.testing.assert_array_equal(column(clip, "head", "translate", 2), [0.25, 0.25, 6.0, 0.25, 0.25, 0.25])
        # a view of the clip's nodes, at a frame: the frame's pose of those nodes
        mine = pick(clip, "arm", "spine").at(4)
        self.assertTrue(mine.is_view)
        self.assertEqual([node.name for node in mine], ["arm", "spine"])
        self.assertEqual(float(mine["spine"].rotate[0]), 20.0)
        mine["arm"].translate_x = 8.0
        np.testing.assert_array_equal(column(clip, "arm", "translate", 0), [3.0, 3.0, 3.0, 3.0, 8.0, 3.0])

    def test_the_pose_algebra_answers_of_the_frame(self):
        clip  = walk()
        rest  = chain()
        delta = clip.at(4) - rest
        self.assertIs(type(delta), HierarchyData)
        clip.frame = 4
        np.testing.assert_allclose(np.asarray(delta.rotate), np.asarray((clip.copy() - rest).at(4).rotate), atol=1e-9)
        clip.frame = 0
        # in place: frame 4 becomes its delta, the other frames stay
        before = np.array(clip.frames.rotate)
        pose   = clip.at(4)
        pose -= rest
        after = np.array(clip.frames.rotate)
        np.testing.assert_allclose(after[4], np.asarray(delta.rotate), atol=1e-9)
        np.testing.assert_allclose(np.delete(after, 4, axis=0), np.delete(before, 4, axis=0))

    def test_visibility_and_user_channels_of_a_frame(self):
        rig = chain()
        rig["spine"].add_user_attribute("heroHeight", 1.0)
        rig["spine"].add_user_attribute("label", "a")
        clip                    = ClipData(rig, frames=4)
        pose                    = clip.at(2)
        pose["neck"].visibility = False
        np.testing.assert_array_equal(np.asarray(clip.frames["neck"].visibility), [True, True, False, True])
        self.assertEqual((pose["neck"].visibility, clip.at(1)["neck"].visibility), (False, True))
        # a user channel written on one frame is animated, the others keep its value
        pose["spine"].heroHeight = 5.0
        np.testing.assert_allclose(np.asarray(clip.frames["spine"].heroHeight).reshape(-1), [1.0, 1.0, 5.0, 1.0])
        self.assertEqual((float(pose["spine"].heroHeight), float(clip.at(0)["spine"].heroHeight)), (5.0, 1.0))
        # a user attribute no frame keys is shared by every frame
        with self.assertRaises(AttributeError) as caught:
            pose["spine"].label = "b"
        self.assertIn("label changes what every frame of a clip shares", str(caught.exception))
        self.assertEqual(clip["spine"].label, "a")


if __name__ == "__main__":
    unittest.main()
