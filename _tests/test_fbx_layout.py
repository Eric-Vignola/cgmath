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
from cgmath._tests.test_fbx_file import FBX, USER, census, every_kind, extended, maya_file
from cgmath.formats import FbxFile, save_fbx
from cgmath.formats._fbx_io import open_fbx, typed
from cgmath.geometry import MaterialData, MaterialList, MeshData, SkinData, UVData, UVList
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

    def test_another_kind(self):
        """a joint made a transform, its orient folded into its rotate, a
        transform made a joint, a joint made a locator: each the node it
        was, its children where they were in the world"""

        def edit(f):
            for nodes in lists(f):
                nodes["spine"].set_node_type("transform")
                nodes["props"].set_node_type("joint")
                nodes["props"].radius = 2.5
                nodes["arm"].set_node_type("locator")

        for name in ("still", "takes"):
            with self.subTest(file=name):
                source = loaded(SHARED[name])
                held   = source.hierarchy if source.hierarchy is not None else source.clip.active
                world  = np.array(held["head"].world_matrix)
                self.assertEqual(self.assertSaves(SHARED[name], edit, name=f"{name}.fbx"), [])
                read = loaded(self.path(f"{name}.fbx"))
                held = read.hierarchy if read.hierarchy is not None else read.clip.active
                self.assertEqual([held[node].node_type for node in ("spine", "props", "arm")], ["transform", "joint", "locator"])
                np.testing.assert_allclose(np.asarray(held["spine"].joint_orient), 0.0, atol=1e-12)
                np.testing.assert_allclose(np.asarray(held["head"].world_matrix), world, atol=1e-5)
                before, after = census(SHARED[name])[0], census(self.path(f"{name}.fbx"))[0]
                # spine and arm leave the joints, props joins them: one skeleton fewer, one null more
                self.assertEqual(after["FbxSkeleton"], before["FbxSkeleton"] - 1)
                self.assertEqual(after["FbxNull"], before["FbxNull"] + 1)
                self.assertEqual(after["FbxNode"], before["FbxNode"])
                self.assertEqual(loose_curves(self.path(f"{name}.fbx")), 0)

    def test_locator_and_back(self):
        """a transform made a locator and a locator a transform: the null
        the node carries is kept, its look changed"""

        def there(f):
            f.hierarchy["props"].set_node_type("locator")

        def back(f):
            f.hierarchy["props"].set_node_type("transform")

        self.assertEqual(self.assertSaves(SHARED["still"], there, name="locator.fbx"), [])
        self.assertEqual(loaded(self.path("locator.fbx")).hierarchy["props"].node_type, "locator")
        self.assertEqual(self.assertSaves(self.path("locator.fbx"), back, name="back.fbx"), [])
        self.assertEqual(census(self.path("back.fbx"))[0], census(SHARED["still"])[0])

    def test_another_kind_refused(self):
        """a kind given without set_node_type(), which leaves a joint's
        orient on a transform; a kind changed in one take alone"""

        def by_hand(f):
            f.hierarchy["spine"].node_type = "transform"

        def one_take(f):
            f.clip[0]["spine"].set_node_type("transform")

        self.assertRefuses(SHARED["still"], by_hand, "which Maya holds on joints only")
        self.assertRefuses(SHARED["takes"], one_take, "a file holds one set of nodes")

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

        self.assertRefuses(SHARED["takes"], one_take, "a file holds one set of nodes")

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



def every_list(f: FbxFile) -> list:
    """every record of the handle holding nodes: its hierarchy and its takes"""
    return ([f.hierarchy] if f.hierarchy is not None else []) + (list(f.clip) if f.clip is not None else [])


def carrier(f: FbxFile, name: str) -> str:
    """a transform added at the top of every record holding nodes, to carry
    a shape: its uuid"""
    for nodes in every_list(f):
        added(nodes, name, node_type="transform")
    return UUIDS[name]


def on(f: FbxFile, node: str):
    """the mesh on the node named ``node``"""
    uuid = every_list(f)[0][node].uuid
    return next(mesh for mesh in f.meshes if mesh.parent_uuid == uuid)


def cube(parent_uuid: str, at=(0.0, 0.0, 0.0), name: str = "cubeShape") -> MeshData:
    corners = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]) + np.asarray(at)
    faces   = np.array([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    return MeshData(points=corners, indices=faces, counts=np.full(6, 4), name=name, parent_uuid=parent_uuid)


def kinds(path: str) -> dict:
    return census(path)[0]


def taken(records, record) -> None:
    """``record`` itself taken off ``records``"""
    del records[[id(held) for held in records].index(id(record))]


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMeshes(_Edits):
    """a mesh added, taken off or given other faces, with the records that
    name it"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.every = every_kind(os.path.join(SHARED["folder"], "every.fbx"))

    def test_taken_off(self):
        """every mesh taken off: their UV sets, skins, materials and blend
        shapes go with them, and the materials' textures; the nodes stay"""

        def edit(f):
            f.meshes = None

        self.assertEqual(self.assertSaves(self.every, edit), [])
        before, after = kinds(self.every), kinds(self.path("saved.fbx"))
        for kind in ("FbxMesh", "FbxSkin", "FbxCluster", "FbxSurfaceLambert", "FbxSurfacePhong", "FbxFileTexture", "FbxBlendShape", "FbxShape"):
            self.assertEqual(after.get(kind, 0), 0, kind)
        self.assertEqual(after["FbxNode"], before["FbxNode"])
        self.assertEqual(loose_curves(self.path("saved.fbx")), 0)
        with FbxFile(self.every, verbose=False) as f:
            f.meshes = None
            f.save(self.path("lists.fbx"))
            self.assertEqual((f.meshes, f.skins, f.uvs, f.materials, f.morphs), (None, None, None, None, None))

    def test_taken_off_with_their_nodes(self):
        """every mesh and the node it sat on taken off in one save: the
        skins, materials and bind pose entries go with them"""

        def edit(f):
            carriers = {mesh.parent_uuid for mesh in f.meshes}
            f.meshes = None
            for nodes in every_list(f):
                for at in reversed(range(len(nodes))):
                    if nodes[at].uuid in carriers:
                        del nodes[at]

        self.assertEqual(self.assertSaves(self.every, edit), [])
        before, after = kinds(self.every), kinds(self.path("saved.fbx"))
        for kind in ("FbxMesh", "FbxSkin", "FbxCluster", "FbxSurfaceLambert", "FbxSurfacePhong", "FbxFileTexture"):
            self.assertEqual(after.get(kind, 0), 0, kind)
        self.assertEqual(after["FbxNode"], before["FbxNode"] - 3)

    def test_one_taken_off(self):
        """one mesh of three taken off: the others keep their records, and
        a material no mesh is left with goes"""

        def edit(f):
            taken(f.meshes, on(f, "planeShape"))

        self.assertEqual(self.assertSaves(self.every, edit), [])
        after = kinds(self.path("saved.fbx"))
        self.assertEqual((after["FbxMesh"], after.get("FbxSurfaceLambert", 0), after["FbxSurfacePhong"]), (2, 0, 1))
        read = loaded(self.path("saved.fbx"))
        self.assertEqual((len(read.meshes), read.uvs, len(read.materials), len(read.skins)), (2, None, 1, 1))

    def test_added(self):
        """a cube on a node of its own, with a UV set, a material of its
        own and a skin to one joint, in Maya's files, Z-up among them"""

        def edit(f):
            mesh = cube(carrier(f, "crate"), at=(0.0, 5.0, 0.0))
            f.meshes.append(mesh)
            uv          = UVData(points=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), indices=np.tile([0, 1, 2, 3], 6), counts=np.full(6, 4), name="map1", mesh_uuid=mesh.uuid)
            f.uvs       = UVList(list(f.uvs or ()) + [uv])
            f.materials = MaterialList(list(f.materials or ()) + [MaterialData(name="crateLambert", mesh_uuid=mesh.uuid, color=[0.2, 0.4, 0.6])])
            f.skins.append(SkinData(weights=np.ones((8, 1)), influences=["head"], mesh_uuid=mesh.uuid, name="crateSkin"))

        for case, name in (("skin_geo", "maya.fbx"), ("zup_skinned", "z_cm.fbx")):
            with self.subTest(file=case):
                source = maya_file(case, name)
                self.assertEqual(self.assertSaves(source, edit, name=f"{case}.fbx"), [])
                before, after = kinds(source), kinds(self.path(f"{case}.fbx"))
                for kind, more in (("FbxMesh", 1), ("FbxSkin", 1), ("FbxCluster", 1), ("FbxSurfaceLambert", 1), ("FbxNode", 1)):
                    self.assertEqual(after.get(kind, 0), before.get(kind, 0) + more, kind)

    def test_other_faces(self):
        """a skinned mesh subdivided, its skin with it, and a mesh of two
        materials and a UV set triangulated, each record following by the
        same call"""

        def skinned(f):
            mesh = on(f, "geo")
            skin = next(skin for skin in f.skins if skin.mesh_uuid == mesh.uuid)
            skin.subdivide(mesh, 1)
            mesh.subdivide(1)

        def covered(f):
            mesh = on(f, "plane")
            uv   = next(uv for uv in f.uvs if uv.mesh_uuid == mesh.uuid)
            f.materials.subdivide(mesh, 1)
            mesh.subdivide(1)
            uv.subdivide(1)

        self.assertEqual(self.assertSaves(maya_file("zup_skinned", "z_cm.fbx"), skinned, name="skinned.fbx"), [])
        read = loaded(self.path("skinned.fbx"))
        self.assertEqual((len(read.meshes[0].points), len(read.skins[0].weights)), (len(read.meshes[0].points),) * 2)
        self.assertGreater(len(read.meshes[0].counts), 368 * 3)
        source = maya_file("materials_export", "maya_embedded.fbx")
        self.assertSaves(source, covered, name="covered.fbx")
        read = loaded(self.path("covered.fbx"))
        mesh = read.meshes[0]
        held = [material for material in read.materials if material.mesh_uuid == mesh.uuid]
        self.assertEqual((len(mesh.counts), sorted(len(material.faces) for material in held)), (16, [8, 8]))
        self.assertEqual(kinds(self.path("covered.fbx")).get("FbxFileTexture"), kinds(source).get("FbxFileTexture"))

    def test_split(self):
        """a mesh split in two by its faces: each part keeps the materials
        of its faces, the new part's records naming the file's own
        materials, and its UVs"""

        def edit(f):
            mesh              = on(f, "plane")
            uv                = next(uv for uv in f.uvs if uv.mesh_uuid == mesh.uuid)
            faces             = [0, 1]
            part              = mesh.from_faces(faces)
            part.uuid         = generate_uuid()
            part.parent_uuid  = carrier(f, "half")
            part_uv           = uv.from_faces(faces)
            part_uv.mesh_uuid = part.uuid
            f.materials.extend(f.materials.from_faces(mesh, faces, part))
            f.materials.delete_faces(mesh, faces)
            mesh.delete_faces(faces)
            uv.delete_faces(faces)
            f.meshes.append(part)
            f.uvs.append(part_uv)

        source = maya_file("materials_export", "maya_embedded.fbx")
        self.assertSaves(source, edit)
        before, after = kinds(source), kinds(self.path("saved.fbx"))
        self.assertEqual(after["FbxMesh"], before["FbxMesh"] + 1)
        # no material made: the parts share the file's
        for kind in ("FbxSurfaceLambert", "FbxSurfacePhong", "FbxSurfaceMaterial", "FbxFileTexture"):
            self.assertEqual(after.get(kind, 0), before.get(kind, 0), kind)
        read  = loaded(self.path("saved.fbx"))
        faces = {mesh.uuid: len(mesh.counts) for mesh in read.meshes}
        cover = {}
        for material in read.materials:
            cover[material.mesh_uuid] = cover.get(material.mesh_uuid, 0) + (faces[material.mesh_uuid] if material.faces is None else len(material.faces))
        self.assertEqual(cover, faces)

    def test_another_kind(self):
        """a joint a skin weighs made a transform in Maya's Z-up skinned
        file: the skin still weighs it; the node of a mesh made a joint,
        and a node carrying what no record holds made one, are refused"""

        def influence(f):
            for nodes in every_list(f):
                nodes["hand"].set_node_type("transform")

        def mesh_node(f):
            for nodes in every_list(f):
                nodes["geo"].set_node_type("joint")

        def unread(f):
            for nodes in every_list(f):
                nodes["lod"].set_node_type("joint")

        source = maya_file("zup_skinned", "z_cm.fbx")
        self.assertEqual(self.assertSaves(source, influence), [])
        read = loaded(self.path("saved.fbx"))
        self.assertEqual(read.hierarchy["hand"].node_type, "transform")
        self.assertIn("hand", read.skins[0].influences)
        self.assertEqual(kinds(self.path("saved.fbx"))["FbxCluster"], kinds(source)["FbxCluster"])
        self.assertRefuses(source, mesh_node, "holds a mesh on a transform only")
        self.assertRefuses(extended(self.every, self.path("extended.fbx")), unread, "carries a shape, or something no record holds")

    def test_split_skinned(self):
        """a skinned mesh split in two by its faces: each part keeps the
        weights of its points, by the same calls the mesh takes"""
        faces = np.arange(100)

        def edit(f):
            mesh             = on(f, "geo")
            part             = mesh.from_faces(faces)
            part.uuid        = generate_uuid()
            part.name        = "half"
            part.parent_uuid = carrier(f, "half")
            f.skins.extend(f.skins.from_faces(mesh, faces, part))
            f.skins.delete_faces(mesh, faces)
            mesh.delete_faces(faces)
            f.meshes.append(part)

        source = maya_file("zup_skinned", "z_cm.fbx")
        self.assertSaves(source, edit)
        before, after = kinds(source), kinds(self.path("saved.fbx"))
        self.assertEqual((after["FbxMesh"], after["FbxSkin"]), (before["FbxMesh"] + 1, before["FbxSkin"] + 1))
        read = loaded(self.path("saved.fbx"))
        self.assertEqual(sum(len(mesh.counts) for mesh in read.meshes), 368)
        for mesh in read.meshes:
            skin = next(skin for skin in read.skins if skin.mesh_uuid == mesh.uuid)
            self.assertEqual(len(skin.weights), len(mesh.points))
            np.testing.assert_allclose(np.asarray(skin.weights).sum(axis=1), 1.0, atol=1e-6)

    def test_records(self):
        """a UV set added and one taken off, a skin given an influence the
        file's lacks, a skin taken off, a material added on some faces and
        one taken off"""

        def uv_added(f):
            first     = f.uvs[0]
            more      = first.copy()
            more.name = "extra"
            f.uvs.append(more)

        def uv_taken_off(f):
            f.uvs.pop()

        def influence(f):
            # a joint the save adds, under the chest in every record holding nodes
            for nodes in every_list(f):
                added(nodes, "extra", node_type="joint", translate=[0.0, 1.0, 0.5])
                nodes["extra"].set_parent(nodes["chest"], world_space=False)
            skin            = f.skins[0]
            free            = "extra"
            weights         = np.array(skin.weights)
            column          = np.zeros((len(weights), 1))
            column[:10]     = 1.0
            weights[:10]    = 0.0
            skin.influences = list(skin.influences) + [free]
            skin.weights    = np.hstack([weights, column])

        def unskinned(f):
            f.skins = None

        def material_added(f):
            mesh       = on(f, "cube")
            held       = next(material for material in f.materials if material.mesh_uuid == mesh.uuid)
            held.faces = np.array([0, 1, 2], dtype=np.int32)
            f.materials.append(MaterialData(name="patch", mesh_uuid=mesh.uuid, faces=np.array([3, 4, 5], dtype=np.int32), kind="phong"))

        def material_taken_off(f):
            mesh = on(f, "plane")
            held = [material for material in f.materials if material.mesh_uuid == mesh.uuid]
            taken(f.materials, held[1])
            held[0].faces = None

        uvs       = maya_file("uvs_export",       "maya.fbx")
        materials = maya_file("materials_export", "maya_embedded.fbx")
        skinned   = maya_file("zup_skinned",      "z_cm.fbx")
        self.assertEach(
            saves=[
                (uvs, uv_added), (uvs, uv_taken_off), (skinned, influence), (uvs, influence), (skinned, unskinned),
                (materials, material_added), (materials, material_taken_off),
            ],
        )

    def test_refused(self):
        """a mesh on no node, a mesh on a joint, a mesh of blend shapes
        given other faces, a skin and a material left on the faces their
        mesh had, a material that brings a texture"""

        def no_node(f):
            f.meshes.append(cube(None))

        def on_a_joint(f):
            f.meshes.append(cube(f.hierarchy["head"].uuid))

        def morphed(f):
            mesh = on(f, "boxShape")
            f.materials.subdivide(mesh, 1)
            mesh.subdivide(1)

        def skin_left(f):
            on(f, "geo").subdivide(1)

        def material_left(f):
            mesh = on(f, "plane")
            uv   = next(uv for uv in f.uvs if uv.mesh_uuid == mesh.uuid)
            mesh.subdivide(1)
            uv.subdivide(1)

        def textured(f):
            mesh = on(f, "cube")
            taken(f.materials, next(material for material in f.materials if material.mesh_uuid == mesh.uuid))
            f.materials.append(MaterialData(name="new", mesh_uuid=mesh.uuid, texture=np.zeros((2, 2, 3), dtype=np.uint8)))

        def one_uuid(f):
            mesh             = on(f, "cube")
            part             = mesh.from_faces([0, 1])
            part.parent_uuid = carrier(f, "half")
            f.meshes.append(part)

        materials = maya_file("materials_export", "maya_embedded.fbx")
        self.assertEach(
            refusals=[
                (self.every, no_node, "names the node that carries it"),
                (self.every, on_a_joint, "holds a mesh on a transform only"),
                (self.every, morphed, "blend shape"),
                (maya_file("zup_skinned", "z_cm.fbx"), skin_left, "holds 288 rows"),
                (materials, material_left, "material"),
                (materials, textured, "brings a texture or a normal map"),
                (materials, one_uuid, "share one uuid"),
            ],
        )

    def test_kept(self):
        """what no record holds stays through a change of layout: a bind
        pose, an LOD group and user properties, with a mesh added and
        another given other faces"""
        source = extended(self.every, self.path("extended.fbx"))

        def edit(f):
            f.meshes.append(cube(carrier(f, "crate")))
            body = on(f, "geo")
            skin = next(skin for skin in f.skins if skin.mesh_uuid == body.uuid)
            skin.subdivide(body, 1)
            body.subdivide(1)

        self.assertSaves(source, edit)
        before, after = census(source), census(self.path("saved.fbx"))
        self.assertEqual(after[0].get("FbxLODGroup"), 1)
        self.assertEqual((after[2], after[4]), (before[2], before[4]))


if __name__ == "__main__":
    unittest.main()
