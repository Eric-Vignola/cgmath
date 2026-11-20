"""A selection of a hierarchy is a HierarchyData, of a clip a ClipData: a view.

Slicing, indexing, match() and the branch queries give an object of the
class they are asked of, holding the very nodes its owner holds: a write to
it reaches the owner, and copy() gives one that owns its nodes. A view of a
clip keeps the clip's frames: each of its nodes is the clip's node on every
frame, and a run of the clip's frames is a view too. A list of nodes built
by hand stays a TransformList, a view of nothing.
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
    clip                    = ClipData(chain(), frames=frames, fps=30.0, take="walk")
    clip["spine"].rotate_x  = np.linspace(0.0, 25.0, frames)
    clip["arm"].translate_y = np.linspace(2.0, 3.0, frames)
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
                # each node is the clip's node of that name, on every frame
                self.assertTrue(all(node == clip[node.name] for node in view))
        self.assertEqual((clip.is_view, clip.owner), (False, None))
        # one node picked is that node on every frame: a view too, and no TransformData
        for label, node in (("x['spine']", clip["spine"]), ("x[0]", clip[0]), ("x[1:3][0]", clip[1:3][0])):
            with self.subTest(query=label):
                self.assertNotIsInstance(node, (ClipData, TransformData))
                self.assertTrue(node.is_view)
                self.assertIs(node.owner, clip)
                self.assertEqual(node, clip[node.name])
                self.assertEqual(np.asarray(node.translate).shape, (6, 3))
        # frames picked are a view of the clip as well, and combine with a pick of nodes
        for label, view in (("x.frames[1:3]", clip.frames[1:3]), ("x[1:3].frames[1:3]", clip[1:3].frames[1:3])):
            with self.subTest(query=label):
                self.assertIs(type(view), ClipData)
                self.assertTrue(view.is_view)
                self.assertIs(view.owner, clip)
                self.assertEqual(view.frame_count, 2)
                self.assertTrue(all(node == clip.frames[1:3][node.name] for node in view))

    def test_it_keeps_the_clip_s_frames(self):
        clip = walk()
        view = pick(clip, "arm", "spine")
        self.assertEqual(
            (view.frame_count, view.start_frame, view.end_frame, view.fps, view.take, len(view.frames)),
            (clip.frame_count, clip.start_frame, clip.end_frame, clip.fps, clip.take, len(clip.frames)),
        )
        np.testing.assert_array_equal(view.keyed, [True, True])
        np.testing.assert_array_equal(pick(clip, "head").keyed, [False])
        np.testing.assert_allclose(np.asarray(view.frames.translate_y)[:, 0], np.linspace(2.0, 3.0, 6))
        pose = view.at(5)
        self.assertIsInstance(pose, HierarchyData)
        self.assertTrue(pose.is_view)
        self.assertEqual([node.name for node in pose], ["arm", "spine"])
        np.testing.assert_allclose(float(pose["spine"].rotate[0]), 25.0)
        self.assertEqual(repr(view), "ClipData(['arm', 'spine'], view)")

    def test_a_write_reaches_every_frame_of_the_clip(self):
        clip                    = walk()
        view                    = clip.match("arm")
        view.frames.translate_x = np.asarray(view.frames.translate_x) + 1.0
        np.testing.assert_allclose(np.asarray(clip["arm"].translate_x), 4.0)
        # written on the view itself: the same, every frame
        view.translate_x += 1.0
        np.testing.assert_allclose(np.asarray(clip["arm"].translate_x), 5.0)
        # its pose of one frame is that frame of the clip: a write lands there alone
        view.frames[3]["arm"].translate_x = 9.0
        np.testing.assert_allclose(np.asarray(clip["arm"].translate_x), [5.0, 5.0, 5.0, 9.0, 5.0, 5.0])
        np.testing.assert_allclose(float(clip.frames[3]["arm"].translate[0]), 9.0)

    def test_its_frames_count_and_name_the_nodes_viewed(self):
        clip = walk()
        view = pick(clip, "arm", "spine")
        rise = np.linspace(2.0, 3.0, 6)
        turn = np.linspace(0.0, 25.0, 6)
        # a place and a name are the view's own, never the clip's
        np.testing.assert_allclose(np.asarray(view[[0]].translate_y)[:, 0],        rise)
        np.testing.assert_allclose(np.asarray(view[[-1]].rotate_x)[:, 0],          turn)
        np.testing.assert_allclose(np.asarray(view["spine"].rotate_x),             turn)
        np.testing.assert_allclose(np.asarray(view.match("a*").translate_y)[:, 0], rise)
        self.assertEqual(np.asarray(view.match("ro*").translate_y).shape, (6, 0))
        for name, reach in (("head", lambda: view["head"]), ("root", lambda: view[["root"]])):
            with self.assertRaises(ValueError) as caught:
                reach()
            self.assertIn(f"{name} is not in list", str(caught.exception))
        with self.assertRaises(IndexError):
            view[[2]]
        # a frame is the live pose of the nodes viewed, the clip's own frame
        pose = view.frames[3]
        self.assertIsInstance(pose, HierarchyData)
        self.assertNotIsInstance(pose, ClipData)
        self.assertTrue(pose.is_view)
        self.assertEqual([node.name for node in pose], ["arm", "spine"])
        self.assertEqual((float(pose["arm"].translate[1]), float(pose["spine"].rotate[0])), (rise[3], turn[3]))
        # a slice is a view of those frames; its copy is a clip of its own
        cut = view.frames[1:3]
        self.assertEqual((type(cut), cut.is_view, cut.frame_count), (ClipData, True, 2))
        self.assertIs(cut.owner, clip)
        self.assertEqual([node.name for node in cut], ["arm", "spine"])
        np.testing.assert_allclose(np.asarray(cut["arm"].translate_y), rise[1:3])
        own = cut.copy()
        self.assertEqual((type(own), own.is_view, own.frame_count), (ClipData, False, 2))
        self.assertEqual([node.name for node in own], ["arm", "spine"])
        np.testing.assert_allclose(np.asarray(own["arm"].translate_y), rise[1:3])
        # as an operand it stands for every frame of the nodes viewed
        rest = pick(chain(), "arm", "spine")
        for operand in (view, view.frames):
            delta = operand - rest
            self.assertEqual((type(delta), delta.is_view, delta.frame_count), (ClipData, False, 6))
            self.assertEqual(TransformList(rest.list).get_delta(operand).frame_count, 6)
        # the algebra in place and a pose written to a frame are the clip's own
        with self.assertRaises(TypeError):
            view -= rest
        with self.assertRaises(TypeError):
            view.frames -= rest
        with self.assertRaises(TypeError):
            view.frames[2] = rest
        np.testing.assert_allclose(np.asarray(clip["spine"].rotate_x), turn)

    def test_a_scoped_axis_narrows_among_its_own_nodes(self):
        clip = walk()
        axis = clip[["arm", "spine"]]
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
            np.testing.assert_allclose(np.asarray(made["spine"].rotate_x), np.linspace(0.0, 25.0, 6))
            np.testing.assert_allclose(np.asarray(made["arm"].translate_y), np.linspace(2.0, 3.0, 6))
            # the spine's parent is left out: a root; the arm and the neck keep the spine
            self.assertEqual([node.parent_name for node in made], ["spine", None, "spine"])
        made                    = view.copy()
        made["arm"].translate_y = np.zeros(6)
        np.testing.assert_allclose(np.asarray(clip["arm"].translate_y), np.linspace(2.0, 3.0, 6))

    def test_its_delta_covers_every_frame(self):
        clip  = walk()
        rest  = chain()
        view  = pick(clip, "spine", "arm")
        delta = view - pick(rest, "spine", "arm")
        self.assertIs(type(delta), ClipData)
        self.assertEqual((delta.frame_count, delta.fps, delta.take, delta.is_view), (6, 30.0, "walk", False))
        # frame by frame, what a selection of the pose on that frame gives
        for frame in (0, 3, 5):
            pose = TransformList(view.frames[frame].list).get_delta(pick(rest, "spine", "arm"))
            for name in ("spine", "arm"):
                np.testing.assert_allclose(np.asarray(delta.at(frame)[name].rotate), np.asarray(pose[name].rotate), atol=1e-9)
                np.testing.assert_allclose(np.asarray(delta.at(frame)[name].translate), np.asarray(pose[name].translate), atol=1e-9)
        moved = view.add_delta(pick(chain(), "spine", "arm"))
        self.assertEqual((type(moved), moved.frame_count), (ClipData, 6))
        for other in (walk(4), walk(4).frames):
            with self.assertRaises(ValueError):
                view.get_delta(other)

    def test_a_node_the_clip_no_longer_holds(self):
        clip = walk()
        view = pick(clip, "arm", "head")
        del clip[[node.name for node in clip].index("arm")]
        # the view says so wherever it is read: its values, a frame of it, its copy
        for read in (lambda: view.translate, lambda: view.frames[0], lambda: view.at(0), lambda: view.copy()):
            with self.assertRaises(ValueError) as caught:
                read()
            self.assertIn("'arm' is no longer a node of the clip", str(caught.exception))

    def test_a_view_is_no_take(self):
        clip = walk()
        view = clip[1:3]
        self.assertFalse(clip == view)
        for part in (view, clip.frames[1:3]):
            with self.assertRaises(TypeError) as caught:
                ClipList([part])
            self.assertIn("pass its copy()", str(caught.exception))
        takes = ClipList([clip, walk(4)])
        # a selection of every take's nodes, written: each take is written, every frame
        takes.match("sp*").translate_y = 9.0
        for take, frames in zip(takes, (6, 4)):
            np.testing.assert_array_equal(np.asarray(take["spine"].translate_y), np.full(frames, 9.0))
            np.testing.assert_array_equal(np.asarray(take["neck"].translate_y), np.full(frames, 3.0))


if __name__ == "__main__":
    unittest.main()
