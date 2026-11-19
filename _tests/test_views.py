"""A selection of a hierarchy is a HierarchyData, of a clip a ClipData: a view.

Slicing, indexing, match() and the branch queries give an object of the
class they are asked of, holding the very nodes its owner holds: a write to
it reaches the owner, and copy() gives one that owns its nodes. A view of a
clip keeps the clip's frames. A list of nodes built by hand stays a
TransformList, a view of nothing.
"""

import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath.hierarchy import ClipData, ClipList, HierarchyData, TransformData, TransformList


def chain() -> HierarchyData:
    """root > spine > neck > head, an arm under the spine, a group beside the root"""
    nodes = HierarchyData(
        [
            TransformData("root", node_type="joint", translate=[0.0, 10.0, 0.0]),
            TransformData("spine", node_type="joint", translate=[0.0, 4.0, 0.5], rotate=[5.0, 0.0, 10.0]),
            TransformData("neck", node_type="joint", translate=[0.0, 3.0, 0.0]),
            TransformData("head", node_type="joint", translate=[0.0, 2.0, 0.25]),
            TransformData("arm", node_type="joint", translate=[3.0, 2.0, 0.0]),
            TransformData("props", node_type="transform", translate=[5.0, 0.0, 0.0]),
        ]
    )
    for child, parent in (("spine", "root"), ("neck", "spine"), ("head", "neck"), ("arm", "spine")):
        nodes[child].set_parent(nodes[parent], world_space=False)
    return nodes


def walk(frames: int = 6) -> ClipData:
    """:func:`chain` over ``frames`` frames, the spine turning and the arm rising"""
    clip                           = ClipData(chain(), frames=frames, fps=30.0, take="walk")
    clip.frames["spine"].rotate_x  = np.linspace(0.0, 25.0, frames)
    clip.frames["arm"].translate_y = np.linspace(2.0, 3.0, frames)
    return clip


def pick(held, *names):
    """the nodes named ``names`` of ``held``, in that order, as a selection"""
    listed = [node.name for node in held]
    return held[[listed.index(name) for name in names]]


def queries(held) -> dict:
    """every way of selecting several nodes of ``held``"""
    mask     = np.zeros(len(held), dtype=bool)
    mask[:2] = True
    return {
        "x[1:3]":              held[1:3],
        "x[[0, 1]]":           held[[0, 1]],
        "x[np.array([0, 1])]": held[np.array([0, 1])],
        "x[mask]":             held[mask],
        "x.match('*')":        held.match("*"),
        "x.get_roots()":       held.get_roots(),
        "x.get_children()":    held.get_children(),
        "x.get_branch()":      held.get_branch(),
        "x[0].get_children()": held[0].get_children(),
        "x[0].get_branch()":   held[0].get_branch(),
        "x[-1].get_root()":    held[-1].get_root(),
        "x[1:3].match('*')":   held[1:3].match("*"),
        "x[1:3][0:1]":         held[1:3][0:1],
    }


class TestHierarchyViews(unittest.TestCase):
    def test_every_selection_is_a_view_of_the_hierarchy(self):
        rig = chain()
        for label, view in queries(rig).items():
            with self.subTest(query=label):
                self.assertIsInstance(view, HierarchyData)
                self.assertNotIsInstance(view, ClipData)
                self.assertTrue(view.is_view)
                self.assertIs(view.owner, rig)
                self.assertTrue(all(node is rig[node.name] for node in view))
        self.assertEqual((rig.is_view, rig.owner), (False, None))

    def test_a_write_reaches_the_owner(self):
        rig  = chain()
        view = rig.match("sp*")
        view.translate_y += 10.0
        view[0].name = "back"
        self.assertEqual(float(rig["back"].translate[1]), 14.0)
        self.assertEqual(repr(view), "HierarchyData(['back'], view)")

    def test_copy_owns_its_nodes(self):
        rig  = chain()
        view = pick(rig, "spine", "neck")
        for made in (view.copy(), pickle.loads(pickle.dumps(view))):
            self.assertIs(type(made), HierarchyData)
            self.assertFalse(made.is_view)
            self.assertEqual([node.name for node in made], [node.name for node in view])
            self.assertFalse(any(node is rig[node.name] for node in made))
            # a node whose parent the selection leaves out is a root of the copy
            self.assertEqual([node.parent_name for node in made], [None, "spine"])
        path = os.path.join(tempfile.mkdtemp(), "view.json")
        view.save(path)
        self.assertIs(type(HierarchyData.load(path)), HierarchyData)

    def test_a_view_holds_its_nodes_as_a_list_does(self):
        """a node added to a view, or taken off it, changes the selection alone"""
        rig  = chain()
        view = pick(rig, "spine", "neck")
        view.append(rig["head"])
        view.pop(0)
        self.assertEqual([node.name for node in view], ["neck", "head"])
        self.assertEqual(len(rig), 6)
        self.assertTrue(all(node._hierarchy is rig for node in rig))
        with self.assertRaises(TypeError):
            view.renew_uuids()

    def test_a_list_built_by_hand_is_a_view_of_nothing(self):
        rig, other = chain(), chain()
        free = [TransformData("a"), TransformData("b")]
        for held in (TransformList([rig[1], other[2]]), TransformList(free), TransformList([])):
            self.assertIs(type(held), TransformList)
            self.assertEqual((held.is_view, held.owner), (False, None))
            self.assertIs(type(held[0:1]), TransformList)
            self.assertIs(type(held.match("*")), TransformList)
        self.assertIs(type(free[0].get_children()), TransformList)


class TestClipViews(unittest.TestCase):
    def test_every_selection_is_a_view_of_the_clip(self):
        clip = walk()
        for label, view in queries(clip).items():
            with self.subTest(query=label):
                self.assertIsInstance(view, ClipData)
                self.assertTrue(view.is_view)
                self.assertIs(view.owner, clip)
                self.assertTrue(all(node is clip[node.name] for node in view))

    def test_it_keeps_the_clip_s_frames(self):
        clip = walk()
        view = pick(clip, "arm", "spine")
        self.assertEqual(
            (view.frame_count, view.start_frame, view.end_frame, view.fps, view.take, view.frame),
            (clip.frame_count, clip.start_frame, clip.end_frame, clip.fps, clip.take, clip.frame),
        )
        np.testing.assert_array_equal(view.keyed, [True, True])
        np.testing.assert_array_equal(pick(clip, "head").keyed, [False])
        np.testing.assert_allclose(np.asarray(view.frames.translate_y)[:, 0], np.linspace(2.0, 3.0, 6))
        pose = view.at(5)
        self.assertIs(type(pose), HierarchyData)
        self.assertEqual([node.name for node in pose], ["arm", "spine"])
        np.testing.assert_allclose(float(pose["spine"].rotate[0]), 25.0)
        self.assertEqual(repr(view), "ClipData(['arm', 'spine'], view)")

    def test_a_write_reaches_every_frame_of_the_clip(self):
        clip                    = walk()
        view                    = clip.match("arm")
        view.frames.translate_x = np.asarray(view.frames.translate_x) + 1.0
        np.testing.assert_allclose(np.asarray(clip.frames["arm"].translate_x), 4.0)
        view.frame = 3
        self.assertEqual(clip.frame, 3)

    def test_its_frames_count_and_name_the_nodes_viewed(self):
        clip = walk()
        view = pick(clip, "arm", "spine")
        rise = np.linspace(2.0, 3.0, 6)
        turn = np.linspace(0.0, 25.0, 6)
        # a place and a name are the view's own, never the clip's
        np.testing.assert_allclose(np.asarray(view.frames[[0]].translate_y)[:, 0],        rise)
        np.testing.assert_allclose(np.asarray(view.frames[[-1]].rotate_x)[:, 0],          turn)
        np.testing.assert_allclose(np.asarray(view.frames["spine"].rotate_x),             turn)
        np.testing.assert_allclose(np.asarray(view.frames.match("a*").translate_y)[:, 0], rise)
        self.assertEqual(np.asarray(view.frames.match("ro*").translate_y).shape, (6, 0))
        for reach in (lambda: view.frames["head"], lambda: view.frames[["root"]]):
            with self.assertRaises(ValueError) as caught:
                reach()
            self.assertIn("is no node this axis covers", str(caught.exception))
        with self.assertRaises(IndexError):
            view.frames[[2]]
        # a frame moves the clip and hands the view back; a slice is a clip of its own
        self.assertIs(view.frames[3], view)
        self.assertEqual(clip.frame, 3)
        clip.frame = 0
        cut        = view.frames[1:3]
        self.assertEqual((type(cut), cut.is_view, cut.frame_count), (ClipData, False, 2))
        self.assertEqual([node.name for node in cut], ["arm", "spine"])
        np.testing.assert_allclose(np.asarray(cut.frames["arm"].translate_y), rise[1:3])
        # as an operand it stands for every frame of the nodes viewed
        rest  = pick(chain(), "arm", "spine")
        delta = view.frames - rest
        self.assertEqual((type(delta), delta.frame_count), (ClipData, 6))
        self.assertEqual(TransformList(rest.list).get_delta(view.frames).frame_count, 6)
        # the algebra in place and a pose written to a frame are the clip's own
        with self.assertRaises(TypeError):
            view.frames -= rest
        with self.assertRaises(TypeError):
            view.frames[2] = rest
        np.testing.assert_allclose(np.asarray(clip.frames["spine"].rotate_x), turn)

    def test_a_scoped_axis_narrows_among_its_own_nodes(self):
        clip = walk()
        axis = clip.frames[["arm", "spine"]]
        np.testing.assert_allclose(np.asarray(axis[[0]].translate_y)[:, 0], np.linspace(2.0, 3.0, 6))
        np.testing.assert_allclose(np.asarray(axis["spine"].rotate_x), np.linspace(0.0, 25.0, 6))
        self.assertEqual(np.asarray(axis.match("*").translate_y).shape, (6, 2))
        with self.assertRaises(ValueError):
            axis["root"]

    def test_copy_is_a_clip_of_every_frame(self):
        clip = walk()
        view = pick(clip, "arm", "spine", "neck")
        path = os.path.join(tempfile.mkdtemp(), "view.json")
        view.save(path)
        for made in (view.copy(), pickle.loads(pickle.dumps(view)), ClipData.load(path)):
            self.assertIs(type(made), ClipData)
            self.assertFalse(made.is_view)
            self.assertEqual(([node.name for node in made], made.frame_count), (["arm", "spine", "neck"], 6))
            np.testing.assert_allclose(np.asarray(made.frames["spine"].rotate_x), np.linspace(0.0, 25.0, 6))
            np.testing.assert_allclose(np.asarray(made.frames["arm"].translate_y), np.linspace(2.0, 3.0, 6))
            # the spine's parent is left out: a root; the arm and the neck keep the spine
            self.assertEqual([node.parent_name for node in made], ["spine", None, "spine"])
        made                           = view.copy()
        made.frames["arm"].translate_y = np.zeros(6)
        np.testing.assert_allclose(np.asarray(clip.frames["arm"].translate_y), np.linspace(2.0, 3.0, 6))

    def test_its_delta_covers_every_frame(self):
        clip  = walk()
        rest  = chain()
        view  = pick(clip, "spine", "arm")
        delta = view - pick(rest, "spine", "arm")
        self.assertIs(type(delta), ClipData)
        self.assertEqual((delta.frame_count, delta.fps, delta.take, clip.frame), (6, 30.0, "walk", 0))
        # frame by frame, what a selection of the pose on that frame gives
        for frame in (0, 3, 5):
            clip.frame = frame
            pose       = TransformList(view.list).get_delta(pick(rest, "spine", "arm"))
            for name in ("spine", "arm"):
                np.testing.assert_allclose(np.asarray(delta.at(frame)[name].rotate), np.asarray(pose[name].rotate), atol=1e-9)
                np.testing.assert_allclose(np.asarray(delta.at(frame)[name].translate), np.asarray(pose[name].translate), atol=1e-9)
        clip.frame = 0
        moved      = view.add_delta(pick(chain(), "spine", "arm"))
        self.assertEqual((type(moved), moved.frame_count), (ClipData, 6))
        with self.assertRaises(ValueError):
            view.get_delta(walk(4).frames)

    def test_a_node_the_clip_no_longer_holds(self):
        clip = walk()
        view = pick(clip, "arm", "head")
        del clip[[node.name for node in clip].index("arm")]
        with self.assertRaises(ValueError) as caught:
            view.frames
        self.assertIn("'arm' is no longer a node of the clip", str(caught.exception))

    def test_a_view_is_no_take(self):
        clip = walk()
        view = clip[1:3]
        self.assertFalse(clip == view)
        with self.assertRaises(TypeError) as caught:
            ClipList([view])
        self.assertIn("pass its copy()", str(caught.exception))
        takes = ClipList([clip, walk(4)])
        # a selection of every take's nodes, moved to a frame: each take moves
        takes.match("sp*")[2]
        self.assertEqual([take.frame for take in takes], [2, 2])


if __name__ == "__main__":
    unittest.main()
