"""strip_namespace / namespace on every data type and list, and file-type detection.

The hierarchy-specific rules (a node checked against its whole hierarchy, a
view against every hierarchy its nodes belong to) live in test_hierarchy's
TestNamespaces. These cover what Data and DataList give every other type.
"""

import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry import BSplineData, MeshData, MeshList
from cgmath.geometry._base import _strip_namespace, get_file_type
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.skin_weights import CompactSkinData, SkinData, SkinList


def triangle(name):
    return MeshData(
        points  = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        indices = np.arange(3),
        counts  = np.array([3]),
        name    = name,
    )


def skin(name, influences):
    return SkinData(weights=np.eye(3, len(influences)), influences=list(influences), name=name)


class TestStripName(unittest.TestCase):
    """the name rules shared by every type"""

    def test_paths_strip_each_part(self):
        self.assertEqual(_strip_namespace("|A:grp|A:jnt"),      "|grp|jnt")
        self.assertEqual(_strip_namespace("|A:grp|B:jnt", "A"), "|grp|B:jnt")
        self.assertEqual(_strip_namespace("A:B:node", "A"),     "B:node")

    def test_none_passes_through(self):
        self.assertIsNone(_strip_namespace(None))
        mesh = triangle(None)
        mesh.strip_namespace()
        self.assertIsNone(mesh.name)
        self.assertEqual(mesh.namespace, "")

    def test_an_empty_path_raises(self):
        for path in ("", ":"):
            with self.assertRaises(ValueError):
                _strip_namespace("A:x", path)

    def test_namespace_reads_the_last_path_part(self):
        self.assertEqual(triangle("A:B:tri").namespace,      "A:B")
        self.assertEqual(triangle("|A:grp|B:tri").namespace, "B")
        self.assertEqual(triangle("tri").namespace,          "")

    def test_a_type_without_a_name_raises(self):
        with self.assertRaises(AttributeError):
            BSplineData().strip_namespace()

    def test_an_empty_path_raises_whatever_the_items(self):
        for items in (MeshList([]), MeshList([triangle(None)])):
            with self.assertRaises(ValueError):
                items.strip_namespace("")


class TestSkins(unittest.TestCase):
    """influences and joints are node names: they go with the name"""

    def test_a_skin_strips_its_influences(self):
        data = skin("A:body", ["A:root", "A:tip"])
        data.strip_namespace()
        self.assertEqual((data.name, data.influences), ("body", ["root", "tip"]))

    def test_two_influences_colliding_renames_nothing(self):
        data = skin("A:body", ["A:root", "B:root"])
        with self.assertRaisesRegex(ValueError, "influences"):
            data.strip_namespace()
        self.assertEqual((data.name, data.influences), ("A:body", ["A:root", "B:root"]))

    def test_a_compact_skin_strips_its_influences(self):
        data = CompactSkinData(
            max_influences    = 1,
            influence_indices = np.array([0]),
            weights           = np.array([1.0]),
            influences        = ["A:root"],
        )
        data.strip_namespace()
        self.assertEqual(data.influences, ["root"])

    def test_a_deformer_strips_its_joints_and_its_skin(self):
        weights = skin("A:body", ["A:root", "A:tip"])
        deformer = SkinDeformData(
            mesh=np.zeros((3, 3)),
            skin=weights,
            inverse_bind_matrices = np.tile(np.eye(4), (2, 1, 1)),
            name="A:body",
        )
        deformer.strip_namespace()
        self.assertEqual((deformer.name, deformer.joints), ("body", ["root", "tip"]))
        self.assertEqual((weights.name, weights.influences), ("body", ["root", "tip"]))

    def test_an_object_strips_its_mesh_and_its_skin(self):
        from cgmath.render import Object

        weights = skin("A:body", ["A:root"])
        deformer = SkinDeformData(
            mesh=triangle("A:body"),
            skin=weights,
            inverse_bind_matrices = np.eye(4)[None],
            name="A:body",
        )
        obj = Object(name="A:body", mesh=triangle("A:body"), skin=deformer)
        obj.strip_namespace()
        self.assertEqual(
            (obj.name, obj.mesh.name, obj.skin.name, obj.skin.joints, weights.influences),
            ("body", "body", "body", ["root"], ["root"]),
        )

    def test_posing_after_a_strip_keeps_the_stripped_name(self):
        # posing deforms a cached copy of the bind-pose mesh: it is stripped too
        from cgmath.hierarchy import HierarchyData, TransformData
        from cgmath.render import Object

        deformer = SkinDeformData(
            mesh=triangle("A:body"),
            skin=skin("A:body", ["A:root"]),
            inverse_bind_matrices = np.eye(4)[None],
            name="A:body",
        )
        obj = Object(name="A:body", mesh=triangle("A:body"), skin=deformer)
        rig = HierarchyData([TransformData(name="A:root", node_type="joint")])
        obj.pose(rig)

        obj.strip_namespace()
        rig.strip_namespace()
        obj.pose(rig)
        self.assertEqual(obj.mesh.name, "body")
        obj.restore_bind_pose()
        self.assertEqual(obj.mesh.name, "body")


class TestLists(unittest.TestCase):
    """every list: all or nothing, and the name lookups"""

    def test_a_clash_renames_nothing(self):
        skins = SkinList([skin("A:body", ["A:root"]), skin("B:body", ["B:root"])])
        with self.assertRaisesRegex(ValueError, "body <- A:body, B:body"):
            skins.strip_namespace()
        self.assertEqual(skins.name, ["A:body", "B:body"])
        self.assertEqual([x.influences for x in skins], [["A:root"], ["B:root"]])

    def test_one_namespace_path(self):
        skins = SkinList([skin("A:body", ["A:root"]), skin("B:body", ["A:root"])])
        skins.strip_namespace("A")
        self.assertEqual(skins.name, ["body", "B:body"])
        self.assertEqual([x.influences for x in skins], [["root"], ["root"]])

    def test_name_namespace_and_get(self):
        meshes = MeshList([triangle("A:one"), triangle("two")])
        self.assertEqual(meshes.name, ["A:one", "two"])
        self.assertEqual(meshes.namespace, ["A", ""])
        self.assertIs(meshes.get("two"), meshes[1])
        self.assertIsNone(meshes.get("three"))
        self.assertEqual(meshes.get("three", "default"), "default")

        meshes.strip_namespace()
        self.assertEqual(meshes.name, ["one", "two"])


class TestFileTypes(unittest.TestCase):
    """load() picks the reader from the file: header first, then extension"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_types(self):
        cases = [
            ("a.glb", b"glTF\x02\x00\x00\x00", "glb"),
            ("a.fbx", b"Kaydara FBX Binary  \x00", "fbx"),
            ("a.fbx", b"; FBX 7.4.0 project file", "fbx"),
            ("a.gltf", b'{\n  "asset": {}}', "glb"),
            ("a.npz", b"PK\x03\x04", "npz"),
            ("a.json", b'{\n    "a": 1\n}', "json"),
            ("a.obj", b"v 0 0 0\n", "obj"),
            ("a.usda", b"#usda 1.0\n", "usd"),
            ("a.usdz", b"PK\x03\x04", "usd"),
            ("a.pkl", b"\x80\x04", "pkl"),
        ]
        for name, data, expected in cases:
            with self.subTest(name=name, header=data[:8]):
                self.assertEqual(get_file_type(self.file(name, data)), expected)

    def test_obj_goes_to_each_types_reader(self):
        from cgmath.geometry import UVData, UVList

        path = self.file(
            "two.obj",
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 2 0 0\nv 3 0 0\nv 2 1 0\n"
            b"vt 0.1 0.1\nvt 0.9 0.1\nvt 0.1 0.9\n"
            b"g body\nf 1/1 2/2 3/3\ng prop\nf 4/1 5/2 6/3\n",
        )
        meshes = MeshList.load(path)
        self.assertEqual(len(meshes), 2)

        self.assertIsInstance(UVList.load(path), UVList)  # the first mesh's channels
        np.testing.assert_allclose(
            UVList.load_obj(path, name=meshes.name[1])[0].points,
            [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]],
        )

        # a UV channel, not the mesh's 3D positions
        channel = UVData.load(path)
        self.assertIsInstance(channel, UVData)
        self.assertEqual(channel.points.shape[1], 2)
        with self.assertRaises(IndexError):
            UVData.load_obj(path, channel=1)

    def test_a_class_without_that_reader_raises(self):
        path = self.file("a.obj", b"v 0 0 0\n")
        with self.assertRaisesRegex(ValueError, "no reader for obj"):
            SkinList.load(path)


if __name__ == "__main__":
    unittest.main()
