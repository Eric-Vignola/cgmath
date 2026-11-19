"""FbxFile.save of a changed layout: records added to, taken off or moved in
the open file.

A node, a shape or a record added to an open handle's lists, taken off them
or moved is written into the file's own scene and saved with it: the saved
file reads as the handle's records hold, in their order, and everything of
the file no record holds is kept. What a change leaves stale of the file's
unread data is refused, or with force=True dropped and named. The tests need
the Autodesk SDK; the Maya files were written by Maya 2025, recorded once
(maya_fixtures), and need no Maya.
"""

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from cgmath._tests.test_fbx_edit import _Edits, loaded
from cgmath._tests.test_fbx_file import FBX, USER, census, maya_file
from cgmath.formats import FbxFile, save_fbx
from cgmath.formats._fbx_io import open_fbx, typed
from cgmath.hierarchy import ClipData, ClipList, HierarchyData, TransformData, generate_uuid

SHARED = {}


def chain() -> HierarchyData:
    """root > spine > neck > head, an arm under the spine, a group beside
    the root: joints of other orients, orders and user attributes"""
    nodes = HierarchyData(
        [
            TransformData("root", node_type="joint", translate=[0.0, 10.0, 0.0]),
            TransformData("spine", node_type="joint", translate=[0.0, 4.0, 0.5], rotate=[5.0, 0.0, 10.0], joint_orient=[0.0, 0.0, 20.0]),
            TransformData("neck", node_type="joint", translate=[0.0, 3.0, 0.0], rotate_order=2),
            TransformData("head", node_type="joint", translate=[0.0, 2.0, 0.25]),
            TransformData("arm", node_type="joint", translate=[3.0, 2.0, 0.0], rotate=[0.0, 30.0, 0.0]),
            TransformData("props", node_type="transform", translate=[5.0, 0.0, 0.0]),
        ]
    )
    for child, parent in (("spine", "root"), ("neck", "spine"), ("head", "neck"), ("arm", "spine")):
        nodes[child].set_parent(nodes[parent], world_space=False)
    nodes["head"].user_defined_attributes["size"] = {"attributeType": "double", "value": 1.5}
    return nodes


def takes() -> ClipList:
    """two takes over :func:`chain`, the spine and the arm keyed"""
    found = []
    for name, count in (("walk", 6), ("run", 4)):
        take                           = ClipData(chain(), frames=count, fps=30.0, take=name)
        take.frames["spine"].rotate_x  = np.linspace(0.0, 25.0, count)
        take.frames["arm"].translate_y = np.linspace(2.0, 3.0, count)
        found.append(take)
    return ClipList(found)


def setUpModule():
    if FBX is not None:
        folder          = SHARED["folder"] = tempfile.mkdtemp(prefix="cgmath_fbx_layout_")
        SHARED["still"] = os.path.join(folder, "still.fbx")
        SHARED["takes"] = os.path.join(folder, "takes.fbx")
        save_fbx(SHARED["still"], hierarchy=chain())
        save_fbx(SHARED["takes"], clip=takes())


def tearDownModule():
    shutil.rmtree(SHARED.pop("folder", ""), True)


def lists(f: FbxFile) -> list:
    """the handle's records holding nodes: its hierarchy, or its takes"""
    return [f.hierarchy] if f.hierarchy is not None else list(f.clip)


def between(nodes, name: str, low: str, high: str, **values) -> TransformData:
    """a joint added to ``nodes`` between ``low`` and its child ``high``,
    which keeps its place in the world; the same node in every record it is
    added to"""
    nodes.append(TransformData(name, node_type="joint", uuid=UUIDS.setdefault(name, generate_uuid()), **values))
    made = nodes[name]
    made.set_parent(nodes[low], world_space=False)
    nodes[high].set_parent(made)
    return made


# the uuid of each node the tests add, the same in every record
UUIDS = {}


def added(nodes, name: str, **values) -> TransformData:
    """a node added to ``nodes``, the same node in every record"""
    nodes.append(TransformData(name, uuid=UUIDS.setdefault(name, generate_uuid()), **values))
    return nodes[name]


def user_props(path: str) -> int:
    return census(path)[2]


def loose_curves(path: str) -> int:
    """the curve nodes of a file that animate nothing"""
    with open_fbx(path, FBX) as scene:
        criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimCurveNode.ClassId)
        return sum(
            typed(scene.GetSrcObject(criteria, at), FBX.FbxAnimCurveNode, FBX).GetDstPropertyCount() == 0
            for at in range(scene.GetSrcObjectCount(criteria))
        )


class TestClipNodes(unittest.TestCase):
    """a node added to a clip, before any file: a column of its own"""

    def test_added_parented_and_keyed(self):
        """a joint appended to a clip takes a parent, a child that keeps
        its place in the world on every frame, and keys"""
        take = takes()[0]
        take.frames["neck"].translate_x = np.linspace(0.0, 5.0, take.frame_count)
        worlds = [np.array(take.at(frame)["head"].world_matrix) for frame in range(take.frame_count)]
        made   = between(take, "mid", "spine", "neck", translate=[0.0, 1.5, 0.0])
        self.assertEqual((made.parent_name, take["neck"].parent_name), ("spine", "mid"))
        for frame, world in enumerate(worlds):
            np.testing.assert_allclose(np.asarray(take.at(frame)["head"].world_matrix), world, atol=1e-9)
        take.frames["mid"].translate_z = np.linspace(0.0, 2.0, take.frame_count)
        np.testing.assert_allclose(np.asarray(take.at(take.frame_count - 1)["mid"].translate), [0.0, 1.5, 2.0])
        self.assertEqual(np.asarray(take.frames.translate).shape, (take.frame_count, 7, 3))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestNodes(_Edits):
    """a node added, taken off or moved under another parent"""

    def test_added(self):
        """a joint between two others and a locator at the top, each with
        values of its own: one more node each, the rest of the file as it
        was"""

        def edit(f):
            for nodes in lists(f):
                made = between(
                    nodes, "mid", "spine", "neck", translate=[0.0, 1.5, 0.0], rotate=[0.0, 15.0, 0.0],
                    joint_orient=[10.0, 0.0, 0.0], rotate_order=3, radius=2.0,
                )
                made.user_defined_attributes["note"] = {"dataType": "string", "value": "added"}
                added(nodes, "mark", node_type="locator", translate=[1.0, 2.0, 3.0], visibility=False)

        for name in ("still", "takes"):
            with self.subTest(file=name):
                self.assertEqual(self.assertSaves(SHARED[name], edit, name=f"{name}.fbx"), [])
                before, after = census(SHARED[name])[0], census(self.path(f"{name}.fbx"))[0]
                self.assertEqual(after["FbxNode"], before["FbxNode"] + 2)
                self.assertEqual(after["FbxSkeleton"], before["FbxSkeleton"] + 1)
                read = loaded(self.path(f"{name}.fbx"))
                held = read.hierarchy if read.hierarchy is not None else read.clip.active
                self.assertEqual(held["neck"].parent_name, "mid")
                self.assertEqual(held["mid"].parent_name, "spine")

    def test_keyed(self):
        """a joint added to every take and keyed in one"""

        def edit(f):
            for take in f.clip:
                between(take, "mid", "spine", "neck", translate=[0.0, 1.5, 0.0])
            walk                           = f.clip[0]
            walk.frames["mid"].translate_z = np.linspace(0.0, 2.0, walk.frame_count)

        self.assertEqual(self.assertSaves(SHARED["takes"], edit), [])
        read = loaded(self.path("saved.fbx")).clip
        np.testing.assert_allclose(np.asarray(read[0].frames["mid"].translate_z), np.linspace(0.0, 2.0, 6), atol=1e-5)
        np.testing.assert_allclose(np.asarray(read[1].frames["mid"].translate_z), 0.0, atol=1e-9)

    def test_taken_off(self):
        """a leaf taken off, and a joint taken off from between two others,
        its child moved under its parent first; their keys go with them"""

        def edit(f):
            for nodes in lists(f):
                nodes["head"].set_parent(nodes["spine"])
                for name in ("arm", "neck"):
                    del nodes[[node.name for node in nodes].index(name)]

        for name in ("still", "takes"):
            with self.subTest(file=name):
                self.assertEqual(self.assertSaves(SHARED[name], edit, name=f"{name}.fbx"), [])
                before, after = census(SHARED[name]), census(self.path(f"{name}.fbx"))
                self.assertEqual(after[0]["FbxNode"], before[0]["FbxNode"] - 2)
                self.assertEqual(after[0]["FbxSkeleton"], before[0]["FbxSkeleton"] - 2)
                # the keys of the nodes taken off went with them
                self.assertEqual(loose_curves(self.path(f"{name}.fbx")), 0)

    def test_moved(self):
        """a node under another parent, at its place in the world or at its
        own values; a root moved under a node, a node made a root"""

        def edit(f):
            for nodes in lists(f):
                nodes["arm"].set_parent(nodes["head"])
                nodes["props"].set_parent(nodes["root"], world_space=False)
                nodes["neck"].set_parent(None)

        for name in ("still", "takes"):
            with self.subTest(file=name):
                self.assertEqual(self.assertSaves(SHARED[name], edit, name=f"{name}.fbx"), [])
                self.assertEqual(census(self.path(f"{name}.fbx"))[0]["FbxNode"], census(SHARED[name])[0]["FbxNode"])

    def test_maya_files(self):
        """a joint added under the first node of Maya's files, Z-up in cm
        and in metres among them, and their last node moved under it"""

        def edit(f):
            for nodes in lists(f):
                first, last = nodes.list[0].name, nodes.list[-1]
                added(nodes, "added", node_type="joint", translate=[1.0, 2.0, 3.0], rotate=[10.0, 20.0, 30.0])
                nodes["added"].set_parent(nodes[first], world_space=False)
                if last.name != first and not last.get_children():
                    last.set_parent(nodes["added"])

        for case, name in (("zup", "z_m.fbx"), ("pivots_export", "maya.fbx"), ("takes_each", "ours.fbx"), ("rrss", "source.fbx")):
            with self.subTest(file=f"{case}/{name}"):
                self.assertSaves(maya_file(case, name), edit, name=f"{case}.fbx")

    def test_refused(self):
        """a node in one take alone, a node of another kind: refused, each
        naming what to do"""

        def one_take(f):
            between(f.clip[0], "mid", "spine", "neck")

        def another_kind(f):
            f.hierarchy["props"].node_type = "joint"

        self.assertRefuses(SHARED["takes"], one_take, "a file holds one set of nodes")
        self.assertRefuses(SHARED["still"], another_kind, "a node of another kind is another node")

    def test_ties(self):
        """a node a display layer of the file lists: taken off, the save is
        refused, and with force saved, the layer kept and named in the
        warning; another node taken off saves as it is"""
        source = self.path("tied.fbx")
        with FbxFile(SHARED["still"], verbose=False) as f:
            scene = f._scene
            layer = FBX.FbxDisplayLayer.Create(scene, "shown")
            layer.AddMember(scene.FindNodeByName("arm"))
            f.save(source)

        def arm(f):
            nodes = f.hierarchy
            del nodes[[node.name for node in nodes].index("arm")]

        def props(f):
            nodes = f.hierarchy
            del nodes[[node.name for node in nodes].index("props")]

        self.assertRefuses(source, arm, "the DisplayLayer 'shown', which refers to node 'arm'")
        told = self.assertSaves(source, arm, force=True, name="forced.fbx")
        self.assertEqual(len(told), 1, told)
        self.assertIn("the DisplayLayer 'shown', which refers to node 'arm'", told[0])
        self.assertEqual(census(self.path("forced.fbx"))[0].get("FbxDisplayLayer"), 1)
        self.assertEqual(self.assertSaves(source, props, name="free.fbx"), [])

    def test_order(self):
        """the saved file lists its nodes as the records do: the file's own
        order where that is the records', a place kept on each node where it
        is not"""

        def at_the_end(f):
            f.hierarchy.append(TransformData("last", node_type="transform"))

        def out_of_file_order(f):
            nodes = f.hierarchy
            nodes.append(TransformData("mid", node_type="joint"))
            nodes["mid"].set_parent(nodes["root"], world_space=False)

        for edit, kept in ((at_the_end, False), (out_of_file_order, None)):
            with self.subTest(edit=edit.__name__):
                self.assertSaves(SHARED["still"], edit, name=f"{edit.__name__}.fbx")
                with FbxFile(self.path(f"{edit.__name__}.fbx"), verbose=False) as f:
                    scene = f._scene
                    marked = [
                        scene.GetNode(at).FindProperty("cgmath:index").IsValid() for at in range(1, scene.GetNodeCount())
                    ]
                # save_fbx keeps a place on each node of its files: the save keeps them up
                self.assertTrue(all(marked))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestUserAttributes(_Edits):
    """a node's user defined attributes added, changed and taken off"""

    def test_written(self):
        """a bool, a text, a number, an enum and a vector added; one given
        another value, one another kind, one taken off: each reads back as
        its record holds it"""

        def added(f):
            for nodes in lists(f):
                attrs           = nodes["arm"].user_defined_attributes
                attrs["locked"] = {"attributeType": "bool", "value": True}
                attrs["author"] = {"dataType": "string", "value": "Eric Vignola"}
                attrs["weight"] = {"attributeType": "double", "value": 0.25, "keyable": True}
                attrs["side"]   = {"attributeType": "enum", "enumName": "left:right", "value": 1}
                attrs["count"]  = {"attributeType": "long", "value": 7}

        def changed(f):
            for nodes in lists(f):
                attrs         = nodes["head"].user_defined_attributes
                attrs["size"] = {"attributeType": "double", "value": 2.5}

        def retyped(f):
            for nodes in lists(f):
                nodes["head"].user_defined_attributes["size"] = {"dataType": "string", "value": "large"}

        def taken_off(f):
            for nodes in lists(f):
                del nodes["head"].user_defined_attributes["size"]

        for name in ("still", "takes"):
            for edit, more in ((added, 5), (changed, 0), (retyped, 0), (taken_off, -1)):
                with self.subTest(file=name, edit=edit.__name__):
                    out = f"{name}_{edit.__name__}.fbx"
                    self.assertEqual(self.assertSaves(SHARED[name], edit, name=out), [])
                    self.assertEqual(user_props(self.path(out)), user_props(SHARED[name]) + more)

    def test_types(self):
        """the properties made are the ones Maya's importer reads: a bool a
        Bool, a text a KString, each flagged user defined"""

        def edit(f):
            attrs           = f.hierarchy["arm"].user_defined_attributes
            attrs["locked"] = {"attributeType": "bool", "value": True}
            attrs["author"] = {"dataType": "string", "value": "Eric Vignola"}

        self.assertSaves(SHARED["still"], edit)
        with FbxFile(self.path("saved.fbx"), verbose=False) as f:
            node  = f._scene.FindNodeByName("arm")
            kinds = {}
            for name in ("locked", "author"):
                prop = node.FindProperty(name)
                self.assertTrue(prop.GetFlag(USER), name)
                kinds[name] = str(prop.GetPropertyDataType().GetName())
            self.assertEqual(kinds, {"locked": "Bool", "author": "KString"})
            self.assertTrue(FBX.FbxPropertyBool1(node.FindProperty("locked")).Get())
            self.assertEqual(str(FBX.FbxPropertyString(node.FindProperty("author")).Get()), "Eric Vignola")

    def test_refused(self):
        """an attribute that names no kind, and a text the SDK does not take"""

        def no_kind(f):
            f.hierarchy["arm"].user_defined_attributes["locked"] = {"type": "bool", "value": True}

        def no_ascii(f):
            f.hierarchy["arm"].user_defined_attributes["author"] = {"dataType": "string", "value": "Éric"}

        self.assertRefuses(SHARED["still"], no_kind, "an attribute is a dict naming its kind")
        with FbxFile(SHARED["still"], verbose=False) as f:
            no_ascii(f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as caught:
                    f.save(self.path("refused.fbx"))
        self.assertIn("plain ASCII", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
