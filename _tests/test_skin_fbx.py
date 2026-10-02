"""Tests for reading skin weights out of an fbx.

The Autodesk FBX Python SDK is an optional binary dependency that is not
installed in CI, so the reading tests skip without it and only the guard tests
run everywhere. Fixtures are built through the SDK and exported to a temporary
file rather than shipped as binaries.

The load bearing behaviour here is that influences are resolved through the
``FbxNode`` behind each cluster rather than by name. Fbx lets two nodes share a
name, so a name lookup cannot say which one a cluster meant.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cgmath.geometry import SkinData
from cgmath.geometry.skin_weights import SkinList, load_fbx

try:
    import fbx
except ImportError:
    fbx = None


def build_scene(manager):
    scene = fbx.FbxScene.Create(manager, "scene")
    return scene


def add_mesh(scene, name, count):
    node = fbx.FbxNode.Create(scene, name)
    mesh = fbx.FbxMesh.Create(scene, f"{name}Shape")
    mesh.InitControlPoints(count)
    for index in range(count):
        mesh.SetControlPointAt(fbx.FbxVector4(float(index), 0.0, 0.0), index)
    node.SetNodeAttribute(mesh)
    scene.GetRootNode().AddChild(node)
    return node, mesh


def add_joint(scene, name, parent=None):
    node = fbx.FbxNode.Create(scene, name)
    node.SetNodeAttribute(fbx.FbxSkeleton.Create(scene, f"{name}Attr"))
    (parent or scene.GetRootNode()).AddChild(node)
    return node


def bind(scene, mesh, clusters, transform_link=None):
    """clusters is a list of (joint node, {vertex: weight}) pairs

    transform_link maps a joint name to that joint's global ``FbxAMatrix`` at
    bind time.  Left out, the cluster keeps the SDK default of identity, which
    is only the truth for a rig that binds at the origin.
    """
    skin = fbx.FbxSkin.Create(scene, "skin")
    for joint, weights in clusters:
        cluster = fbx.FbxCluster.Create(scene, f"cl_{joint.GetName()}")
        cluster.SetLink(joint)
        for vertex, weight in weights.items():
            cluster.AddControlPointIndex(vertex, weight)
        if transform_link is not None and joint.GetName() in transform_link:
            cluster.SetTransformLinkMatrix(transform_link[joint.GetName()])
        skin.AddCluster(cluster)
    mesh.AddDeformer(skin)
    return skin


def export(manager, scene, path):
    manager.SetIOSettings(fbx.FbxIOSettings.Create(manager, fbx.IOSROOT))
    exporter = fbx.FbxExporter.Create(manager, "")
    if not exporter.Initialize(path, -1, manager.GetIOSettings()):
        raise RuntimeError(exporter.GetStatus().GetErrorString())
    exporter.Export(scene)
    exporter.Destroy()
    return path


class FbxSkinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def write(self, name, populate):
        """populate(scene) builds the fixture; the manager dies with the file"""
        manager = fbx.FbxManager.Create()
        try:
            scene = build_scene(manager)
            populate(scene)
            return export(manager, scene, self.path(name))
        finally:
            manager.Destroy()

    def simple(self, name="simple.fbx"):
        """4 verts, root owns 0 and 1, tip owns 2 and 3"""

        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            bind(scene, mesh, [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})])

        return self.write(name, populate)


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestReading(FbxSkinTest):
    def test_influences_and_weights(self):
        data = load_fbx(self.simple())

        self.assertIsInstance(data, SkinList)
        self.assertEqual(len(data), 1)
        skin = data[0]

        self.assertEqual(skin.name, "body_geo")
        self.assertEqual(skin.influences, ["root", "tip"])
        np.testing.assert_allclose(skin.weights, [[1, 0], [1, 0], [0, 1], [0, 1]])

    def test_fractional_weights_survive(self):
        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 2)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            bind(
                scene,
                mesh,
                [(root, {0: 0.75, 1: 0.25}), (tip, {0: 0.25, 1: 0.75})],
            )

        skin = load_fbx(self.write("fractional.fbx", populate))[0]

        np.testing.assert_allclose(skin.weights, [[0.75, 0.25], [0.25, 0.75]])

    def test_two_clusters_on_one_joint_accumulate(self):
        """a joint reached twice collapses onto one column instead of
        producing a repeated influence name
        """

        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 2)
            root = add_joint(scene, "root")
            skin = fbx.FbxSkin.Create(scene, "skin")
            for index, weights in enumerate(({0: 0.25}, {0: 0.75, 1: 1.0})):
                cluster = fbx.FbxCluster.Create(scene, f"cl{index}")
                cluster.SetLink(root)
                for vertex, weight in weights.items():
                    cluster.AddControlPointIndex(vertex, weight)
                skin.AddCluster(cluster)
            mesh.AddDeformer(skin)

        skin = load_fbx(self.write("twice.fbx", populate))[0]

        self.assertEqual(skin.influences, ["root"])
        np.testing.assert_allclose(skin.weights, [[1.0], [1.0]])

    def test_a_mesh_with_no_skin_is_left_out(self):
        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 2)
            add_mesh(scene, "prop_geo", 3)
            root = add_joint(scene, "root")
            bind(scene, mesh, [(root, {0: 1.0, 1: 1.0})])

        data = load_fbx(self.write("partial.fbx", populate))

        self.assertEqual(data.name, ["body_geo"])

    def test_two_skinned_meshes_both_come_back(self):
        def populate(scene):
            _, first = add_mesh(scene, "a_geo", 2)
            _, second = add_mesh(scene, "b_geo", 2)
            root = add_joint(scene, "root")
            bind(scene, first, [(root, {0: 1.0, 1: 1.0})])
            bind(scene, second, [(root, {0: 1.0, 1: 1.0})])

        data = load_fbx(self.write("two.fbx", populate))

        self.assertEqual(sorted(data.name), ["a_geo", "b_geo"])

    def test_rows_match_the_meshs_control_point_count(self):
        skin = load_fbx(self.simple())[0]

        self.assertEqual(skin.weights.shape[0], 4)

    def test_influences_resolve_against_a_rig_from_the_same_file(self):
        from cgmath.hierarchy import HierarchyData

        path = self.simple()
        skin = load_fbx(path)[0]
        rig  = [str(x) for x in HierarchyData.load_fbx(path).name]

        for influence in skin.influences:
            self.assertIn(influence, rig)


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestClassMethod(FbxSkinTest):
    def test_first_skinned_mesh_by_default(self):
        self.assertEqual(SkinData.load_fbx(self.simple()).influences, ["root", "tip"])

    def test_selection_by_name(self):
        def populate(scene):
            _, first = add_mesh(scene, "a_geo", 2)
            _, second = add_mesh(scene, "b_geo", 2)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            bind(scene, first, [(root, {0: 1.0, 1: 1.0})])
            bind(scene, second, [(tip, {0: 1.0, 1: 1.0})])

        path = self.write("named.fbx", populate)

        self.assertEqual(SkinData.load_fbx(path, name="b_geo").influences, ["tip"])

    def test_an_unknown_name_lists_what_is_there(self):
        with self.assertRaises(ValueError) as caught:
            SkinData.load_fbx(self.simple(), name="nope")

        self.assertIn("available", str(caught.exception))
        self.assertIn("body_geo", str(caught.exception))

    def test_a_file_with_no_skin_raises(self):
        def populate(scene):
            add_mesh(scene, "prop_geo", 3)
            add_joint(scene, "root")

        with self.assertRaises(RuntimeError):
            SkinData.load_fbx(self.write("bare.fbx", populate))


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestDeformRoundTrip(FbxSkinTest):
    def test_the_loaded_pieces_bind_and_deform(self):
        from cgmath.geometry.deform import SkinDeformData
        from cgmath.hierarchy import HierarchyData

        path = self.simple()
        skin = load_fbx(path)[0]
        rig  = HierarchyData.load_fbx(path)
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64
        )

        deformer = SkinDeformData(mesh=points, skin=skin, bind_rig=rig)
        deformer.bind()

        self.assertTrue(deformer.valid)
        np.testing.assert_allclose(deformer.apply(rig), points, atol=1e-9)


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestOneConvention(FbxSkinTest):
    """every module reads an fbx the same way, names exactly as in the file"""

    def namespaced(self):
        """simple(), with every node in the namespace ns"""

        def populate(scene):
            _, mesh = add_mesh(scene, "ns:body_geo", 4)
            root = add_joint(scene, "ns:root")
            tip  = add_joint(scene, "ns:tip", parent=root)
            bind(scene, mesh, [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})])

        return self.write("namespaced.fbx", populate)

    def test_every_module_reads_it(self):
        from cgmath import hierarchy
        from cgmath.geometry import MeshList, mesh, skin_weights

        path = self.namespaced()
        rig  = hierarchy.load_fbx(path)

        self.assertIsInstance(rig, hierarchy.HierarchyData)
        self.assertEqual(hierarchy.load(path).name, rig.name)
        self.assertEqual(rig.namespace[-1], "ns")
        self.assertEqual(skin_weights.load(path).name, ["ns:body_geo"])
        self.assertEqual(SkinList.load(path).name, ["ns:body_geo"])
        self.assertEqual([m.name for m, _ in mesh.load(path)], ["ns:body_geo"])
        self.assertEqual(MeshList.load(path).name, ["ns:body_geo"])

    def test_strip_the_rig_then_its_skins(self):
        from cgmath.geometry.deform import SkinDeformData
        from cgmath.hierarchy import load_fbx as load_rig

        path  = self.namespaced()
        rig   = load_rig(path)
        skins = load_fbx(path)
        rig.strip_namespace()
        skins.strip_namespace()

        self.assertEqual(skins.name, ["body_geo"])
        self.assertEqual(skins[0].influences, ["root", "tip"])

        points = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64
        )
        deformer = SkinDeformData(mesh=points, skin=skins[0], bind_rig=rig)
        deformer.bind()
        np.testing.assert_allclose(deformer.apply(rig), points, atol=1e-9)


class TestGuards(FbxSkinTest):
    """these run with or without the sdk"""

    def test_a_missing_sdk_raises_import_error(self):
        with patch("cgmath.geometry.skin_weights.fbx", None):
            with self.assertRaises(ImportError):
                load_fbx("whatever.fbx")

    @unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
    def test_a_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_fbx(self.path("absent.fbx"))


if __name__ == "__main__":
    unittest.main()