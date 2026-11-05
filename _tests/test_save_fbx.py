"""cgmath.formats.save_fbx: one fbx file from cgmath's objects

Each class's save_fbx is a one-line call to it. The files are read back
through cgmath's own readers (and Maya's, under mayapy).
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from cgmath import formats
from cgmath._tests.test_clip_list import hand_built
from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList, _l2_normalize, _unit_normals
from cgmath.geometry.skin_weights import SkinData, SkinList
from cgmath.hierarchy import ClipData, ClipList, FBX, HierarchyData, PerTake, TransformData


def ngon_mesh(name="patch", normals=None):
    """a triangle, a quad and a pentagon, at points a float32 holds exactly
    (Maya's mesh points are float32); ``normals`` None, "vertex" (one a
    vertex), "corner" (one a corner, no indices) or "hard" (indexed)"""
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [3, 1, 0.5], [4, 3, 1], [1, 4, 0.25], [-1, 3, 0]])
    mesh = MeshData(
        points  = points * [1.5, 1.0, 0.75] + 0.125,
        indices = np.array([1, 4, 2, 0, 1, 2, 3, 3, 2, 5, 6, 7]),
        counts  = np.array([3, 4, 5]),
        name    = name,
    )
    if normals == "hard":
        mesh.set_normals(hard_edge_angle=10.0)
    elif normals is not None:
        mesh.set_normals()
        if normals == "corner":
            mesh.normals = np.asarray(mesh.normals)[mesh.indices]
    return mesh


def ngon_uvs(name="map1", shift=0.0):
    """a UV set for ngon_mesh, at values a float32 holds exactly (Maya's UVs
    are float32): the quad and the pentagon share an edge's UVs, the
    triangle's are its own"""
    points = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0.75, 0.25], [0.875, 0.625],
                       [0.25, 0.875], [0.125, 0.75], [0.625, 0.125], [1, 0.5]]) + shift
    return UVData(points=points, indices=np.array([8, 9, 4, 0, 1, 2, 3, 3, 2, 5, 6, 7]),
                  counts=np.array([3, 4, 5]), name=name)


def skinned(name="body", group=None):
    """``(rig, mesh, skin)``: a chain of three joints, root, mid and tip,
    turned at mid and oriented at tip, beside a group ``group`` (moved) when
    one is named, and a column of five quads skinned to the chain, at points
    and weights a float32 holds exactly"""
    rig = HierarchyData()
    rig.append(TransformData("root", node_type="joint", translate=(0.0, 1.0, 0.0)))
    rig.append(TransformData("mid", node_type="joint", translate=(0.0, 2.0, 0.0), rotate=(0.0, 0.0, 10.0)))
    rig.append(TransformData("tip", node_type="joint", translate=(0.0, 2.0, 0.0), joint_orient=(0.0, 0.0, -5.0)))
    rig[1].parent_uuid = rig[0].uuid
    rig[2].parent_uuid = rig[1].uuid
    if group is not None:
        rig.append(TransformData(group, node_type="transform", translate=(0.5, 0.0, 0.25)))
    heights = np.repeat(np.arange(6.0), 2)
    points  = np.stack([np.tile([-0.5, 0.5], 6), heights, np.zeros(12)], axis=1)
    quads   = [[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(5)]
    mesh    = MeshData(points=points, indices=np.ravel(quads), counts=np.full(5, 4), name=name)
    weights = np.zeros((12, 3))
    for row, height in enumerate(heights):
        weights[row] = {0: (1, 0, 0), 1: (1, 0, 0), 2: (0.5, 0.5, 0), 3: (0, 0.75, 0.25), 4: (0, 0, 1), 5: (0, 0, 1)}[height]
    return rig, mesh, SkinData(weights=weights, influences=["root", "mid", "tip"], name=name)


def column_uvs(mesh, name="map1"):
    """a UV set for skinned()'s column, one UV a vertex, at values a
    float32 holds exactly"""
    points = np.asarray(mesh.points)[:, :2] / 8.0 + [0.5, 0.0]
    return UVData(points=points, indices=np.asarray(mesh.indices), counts=np.asarray(mesh.counts), name=name)


def joint_takes(rig, counts=(4, 3)):
    """a ClipList of the rig's three joints, a take a count, mid and tip
    turning, root moving"""
    clips = []
    for index, count in enumerate(counts):
        joints                = HierarchyData([node.copy() for node in rig[:3]])
        joints[1].parent_uuid = joints[0].uuid
        joints[2].parent_uuid = joints[1].uuid
        clip                  = ClipData(joints, frames=count, take=f"take{index}")
        turns                 = np.array(clip.frames.rotate)
        places                = np.array(clip.frames.translate)
        turns[:, 1:, 0] += np.arange(count)[:, None] * 7.5
        places[:, 0, 2] += np.arange(count) * 0.5
        clip.frames.rotate, clip.frames.translate = turns, places
        clips.append(clip)
    return ClipList(clips)


def corner_ids(mesh):
    """the normal each face corner names"""
    return mesh.normal_indices if mesh.normal_indices is not None else mesh.indices


class TestSaveFbx(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_what_each_class_writes(self):
        """a rig, a clip of one frame (its pose), a clip of more (one take)
        and a ClipList (every take) write as the class's save_fbx writes them"""
        clip = hand_built("idle", 6)
        rig  = HierarchyData(hand_built()[:])  # a rig takes the nodes it is built from
        formats.save_fbx(self.path("rig.fbx"), rig)
        back = HierarchyData.load_fbx(self.path("rig.fbx"), axis_system=None)
        np.testing.assert_allclose(np.array(back.world_matrix), np.array(rig.world_matrix), rtol=0, atol=1e-12)

        formats.save_fbx(self.path("pose.fbx"), clip.frames[2:3])
        self.assertEqual(len(ClipList.load_fbx(self.path("pose.fbx"))), 0)  # a pose, no take
        pose = HierarchyData.load_fbx(self.path("pose.fbx"), axis_system=None)
        np.testing.assert_allclose(np.array(pose.rotate), np.array(clip.frames.rotate)[2], rtol=0, atol=1e-9)

        formats.save_fbx(self.path("take.fbx"), clip)
        takes = ClipList.load_fbx(self.path("take.fbx"))
        self.assertEqual((takes.take, takes.frame_count), (["idle"], [6]))

        both = ClipList([clip, hand_built("walk", 4)])
        formats.save_fbx(self.path("takes.fbx"), both, zero_root=True)
        takes = ClipList.load_fbx(self.path("takes.fbx"))
        self.assertEqual((takes.take, takes.frame_count), (["idle", "walk"], [6, 4]))
        self.assertTrue(np.allclose(np.array(takes[0].frames.translate)[:, 0], 0.0))

    def test_the_methods_call_it(self):
        clip, written = hand_built("idle", 6), []
        with mock.patch("cgmath.hierarchy.hierarchy._save_fbx", lambda *a, **k: written.append((a, k))):
            HierarchyData(hand_built()[:]).save_fbx("a.fbx")
            clip.save_fbx("b.fbx", zero_root=True)
            ClipList([clip]).save_fbx("c.fbx", as_ascii=True)
        self.assertEqual([a[0] for a, _ in written], ["a.fbx", "b.fbx", "c.fbx"])
        self.assertIs(written[1][0][1], clip)
        self.assertEqual([k for _, k in written][1:], [{"zero_root": True, "as_ascii": False},
                                                       {"zero_root": False, "as_ascii": True}])

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_refusals(self):
        """nothing to write, or an object of no kind it writes, raises
        TypeError, and a ClipList's takes are checked before the sdk is
        reached"""
        for objects in ((), ("rig",), (object(),)):
            with self.subTest(objects=[type(x).__name__ for x in objects]):
                with self.assertRaises(TypeError):
                    formats.save_fbx(self.path("x.fbx"), *objects)
        with self.assertRaisesRegex(ValueError, "no take to save"):
            formats.save_fbx(self.path("x.fbx"), ClipList())
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_home_is_expanded(self):
        home = self.path("home")
        os.makedirs(home)
        # expanduser reads HOME on posix and USERPROFILE on windows
        with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
            formats.save_fbx(os.path.join("~", "takes.fbx"), ClipList([hand_built("idle", 3)]))
            formats.save_fbx(os.path.join("~", "rig.fbx"), HierarchyData(hand_built()[:]))
        self.assertEqual(sorted(os.listdir(home)), ["rig.fbx", "takes.fbx"])



class TestSaveFbxMeshes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_round_trip(self):
        """load_fbx reads back every point, face and normal exactly: normals
        one a vertex come back face-varying, the mesh's indices naming them"""
        for normals in (None, "vertex", "corner", "hard"):
            with self.subTest(normals=normals):
                mesh = ngon_mesh(normals=normals)
                mesh.save_fbx(self.path("mesh.fbx"))
                back = MeshData.load_fbx(self.path("mesh.fbx"))
                self.assertEqual(back.name, "patch")
                np.testing.assert_array_equal(back.points,  mesh.points)
                np.testing.assert_array_equal(back.indices, mesh.indices)
                np.testing.assert_array_equal(back.counts,  mesh.counts)
                if normals is None:
                    self.assertIsNone(back.normals)
                    continue
                # the reader makes every normal unit length, as Maya's import does
                np.testing.assert_array_equal(back.normals, _unit_normals(np.asarray(mesh.normals)))
                want = {"vertex": None, "corner": np.arange(12), "hard": mesh.normal_indices}[normals]
                if want is None:
                    self.assertIsNone(back.normal_indices)
                    continue
                np.testing.assert_array_equal(back.normal_indices, want)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_the_matrix_is_the_node(self):
        mesh        = ngon_mesh()
        placed      = TransformData("p", translate=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), scale=(1.0, 2.0, 0.5))
        mesh.matrix = placed.matrix
        formats.save_fbx(self.path("placed.fbx"), mesh)
        rig = HierarchyData.load_fbx(self.path("placed.fbx"), axis_system=None)
        np.testing.assert_allclose(np.asarray(rig["patch"].world_matrix), mesh.matrix, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(MeshData.load_fbx(self.path("placed.fbx")).points, mesh.points)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_a_mesh_list(self):
        meshes           = MeshList([ngon_mesh("a", "hard"), ngon_mesh("b")])
        meshes[1].points = meshes[1].points + 10.0
        meshes.save_fbx(self.path("two.fbx"))
        back = MeshList.load_fbx(self.path("two.fbx"))
        self.assertEqual(back.name, ["a", "b"])
        for a, b in zip(meshes, back):
            np.testing.assert_array_equal(b.points, a.points)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_refusals(self):
        """two meshes of one name, or a UV set, raise before the sdk is
        reached"""
        with self.assertRaisesRegex(ValueError, r"node names repeat.*\['a'\]"):
            formats.save_fbx(self.path("x.fbx"), ngon_mesh("a"), MeshList([ngon_mesh("b"), ngon_mesh("a")]))
        uv = UVData(points=np.zeros((3, 2)), indices=np.arange(3), counts=np.array([3]))
        calls = (
            lambda: uv.save_fbx(self.path("x.fbx")),
            lambda: UVList([uv]).save_fbx(self.path("x.fbx")),
            lambda: formats.save_fbx(self.path("x.fbx"), uv),
        )
        for call in calls:
            with self.assertRaises(TypeError):
                call()
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_what_the_sdk_writer_refuses(self):
        holed              = ngon_mesh("holed")
        holed.hole_faces   = np.array([2])
        holed.hole_counts  = np.array([3])
        holed.hole_indices = np.array([0, 1, 2])
        odd                = ngon_mesh("odd", "vertex")
        odd.normals        = np.asarray(odd.normals)[:5]
        for mesh, message in ((holed, "polygon holes"), (odd, "5 normals")):
            with self.subTest(mesh.name), self.assertRaisesRegex(ValueError, message):
                mesh.save_fbx(self.path("x.fbx"))
            self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_routes(self):
        path = ngon_mesh().save(self.path("routed.fbx"))
        self.assertEqual(MeshData.load(path).name, "patch")
        uv = UVData(points=np.zeros((3, 2)), indices=np.arange(3), counts=np.array([3]))
        with self.assertRaisesRegex(ValueError, "UVData has no writer for fbx files .*a UVData is a UV set"):
            uv.save(self.path("uv.fbx"))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_maya_reads_the_mesh(self):
        """Maya 2025's FBXImport: the point count, each face's vertices in
        order, the points to 1e-9 cm and each face corner's normal"""
        from cgmath._tests.test_fbx_labels import _maya

        maya = _maya()
        if maya is None:
            self.skipTest("needs Maya and its fbxmaya plug-in")
        cmds, mel = maya
        self.addCleanup(cmds.file, new=True, force=True)
        for normals in ("vertex", "hard"):
            with self.subTest(normals=normals):
                mesh = ngon_mesh(normals=normals)
                mesh.save_fbx(self.path("maya.fbx"))
                cmds.file(new=True, force=True)
                mel.eval("FBXResetImport")
                mel.eval(f'FBXImport -f "{self.path("maya.fbx").replace(os.sep, "/")}"')
                (shape,) = cmds.listRelatives("patch", shapes=True, fullPath=True)
                self.assertEqual(cmds.polyEvaluate(shape, vertex=True), 8)
                points = np.array(cmds.xform(f"{shape}.vtx[0:7]", q=True, os=True, t=True)).reshape(-1, 3)
                np.testing.assert_allclose(points, mesh.points, rtol=0, atol=1e-9)
                start, corners = 0, np.asarray(mesh.normals)[corner_ids(mesh)]
                for face, size in enumerate(mesh.counts):
                    info  = cmds.polyInfo(f"{shape}.f[{face}]", faceToVertex=True)[0]
                    found = [int(x) for x in info.split(":")[1].split()]
                    self.assertEqual(found, list(mesh.indices[start : start + size]))
                    for vertex, corner in zip(found, corners[start : start + size]):
                        got = cmds.polyNormalPerVertex(f"{shape}.vtxFace[{vertex}][{face}]", q=True, xyz=True)
                        np.testing.assert_allclose(got, corner, rtol=0, atol=1e-6)
                    start += size



class TestSaveFbxUVs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_round_trip(self):
        """every set of a mesh, read back by UVList.load_fbx exactly, named
        as it was, in its order"""
        mesh = ngon_mesh("patch", "hard")
        uvs  = UVList([ngon_uvs("map1"), ngon_uvs("shifted", 1.0)])
        formats.save_fbx(self.path("uvs.fbx"), uvs, mesh)
        back = UVList.load_fbx(self.path("uvs.fbx"), name="patch")
        self.assertEqual(back.name, ["map1", "shifted"])
        for a, b in zip(uvs, back):
            np.testing.assert_array_equal(b.points,  a.points)
            np.testing.assert_array_equal(b.indices, a.indices)
            np.testing.assert_array_equal(b.counts,  a.counts)
        np.testing.assert_array_equal(MeshData.load_fbx(self.path("uvs.fbx")).points, mesh.points)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_pairing(self):
        """by name where a name matches, else by the mesh's faces; ValueError
        naming the candidates when two fit, or none, before the sdk is
        reached"""
        from cgmath.formats._write import _pair_uvs

        a, b = ngon_mesh("a"), ngon_mesh("b")
        grid = MeshData(points=np.zeros((4, 3)), indices=np.arange(4), counts=np.array([4]), name="grid")
        flat = UVData(points=np.zeros((4, 2)), indices=np.arange(4), counts=np.array([4]), name="map1")
        sets = _pair_uvs([a, grid], [[ngon_uvs()], [flat]])
        self.assertEqual([[uv.name for uv in held] for held in sets], [["map1"], ["map1"]])
        self.assertIs(sets[1][0], flat)
        sets = _pair_uvs([a, b], [[ngon_uvs("b")], [ngon_uvs("map2"), ngon_uvs("a")]])
        self.assertEqual([[uv.name for uv in held] for held in sets], [["map2", "a"], ["b"]])

        cases = (
            ("two meshes fit", [a, b], [[ngon_uvs()]], r"fit meshes \['a', 'b'\]"),
            ("none fits", [grid], [[ngon_uvs()]], "fit no mesh"),
            ("a name, other faces", [a, grid], [[ngon_uvs("grid")]], "other faces than mesh 'grid'"),
            ("a list of two layouts", [a], [[ngon_uvs(), flat]], "other faces than each other"),
            ("two map1 on a mesh", [a], [[ngon_uvs()], [ngon_uvs()]], r"two UV sets named \['map1'\]"),
        )
        for label, meshes, groups, message in cases:
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                _pair_uvs(meshes, groups)
        with self.assertRaisesRegex(TypeError, "with its mesh"):
            formats.save_fbx(self.path("x.fbx"), ngon_uvs())
        with self.assertRaisesRegex(ValueError, "fit meshes"):
            formats.save_fbx(self.path("x.fbx"), a, b, ngon_uvs())
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_sets_that_name_their_mesh(self):
        """the readers give each set its mesh's name, and save_fbx pairs a
        set with the mesh of that name first: meshes of one layout, each with
        a set named map1, save in one call and read back"""
        from cgmath.geometry import mesh as mesh_module

        meshes = MeshList([ngon_mesh("a"), ngon_mesh("b"), ngon_mesh("c")])
        sets   = UVList([ngon_uvs(shift=float(i)) for i in range(3)])
        with self.assertRaisesRegex(ValueError, "fit meshes"):
            formats.save_fbx(self.path("x.fbx"), meshes, sets)
        for uv, mesh in zip(sets, ("c", "a", "b")):
            uv.mesh_name = mesh
        formats.save_fbx(self.path("named.fbx"), sets, meshes)
        for uv, mesh in zip(sets, ("c", "a", "b")):
            (back,) = UVList.load_fbx(self.path("named.fbx"), name=mesh)
            self.assertEqual((back.name, back.mesh_name), ("map1", mesh))
            np.testing.assert_array_equal(back.points, uv.points)
            np.testing.assert_array_equal(back.indices, uv.indices)
        again = [uv for _, held in mesh_module.load_fbx(self.path("named.fbx")) for uv in held]
        self.assertEqual([uv.mesh_name for uv in again], ["a", "b", "c"])
        formats.save_fbx(self.path("again.fbx"), UVList(again), meshes)
        for name in "abc":
            np.testing.assert_array_equal(UVList.load_fbx(self.path("again.fbx"), name=name)[0].points,
                                          UVList.load_fbx(self.path("named.fbx"), name=name)[0].points)

        sets[0].mesh_name = "a"
        sets[0].counts    = np.array([4, 3, 5])
        with self.assertRaisesRegex(ValueError, "'map1' holds other faces than mesh 'a'"):
            formats.save_fbx(self.path("x.fbx"), meshes, sets)
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_maya_reads_the_uv_sets(self):
        """Maya 2025's FBXImport: the set names, the UV count of each and the
        UV of every face corner, to 1e-12"""
        from cgmath._tests.test_fbx_labels import _maya

        maya = _maya()
        if maya is None:
            self.skipTest("needs Maya and its fbxmaya plug-in")
        cmds, mel = maya
        self.addCleanup(cmds.file, new=True, force=True)
        mesh = ngon_mesh("patch", "vertex")
        uvs  = UVList([ngon_uvs("map1"), ngon_uvs("shifted", 1.0)])
        formats.save_fbx(self.path("maya.fbx"), mesh, uvs)
        cmds.file(new=True, force=True)
        mel.eval("FBXResetImport")
        mel.eval(f'FBXImport -f "{self.path("maya.fbx").replace(os.sep, "/")}"')
        (shape,) = cmds.listRelatives("patch", shapes=True, fullPath=True)
        self.assertEqual(cmds.polyUVSet(shape, q=True, allUVSets=True), ["map1", "shifted"])
        for uv in uvs:
            with self.subTest(uv.name):
                cmds.polyUVSet(shape, currentUVSet=True, uvSet=uv.name)
                self.assertEqual(cmds.polyEvaluate(shape, uvcoord=True, uvSetName=uv.name), 10)
                start = 0
                for face, size in enumerate(mesh.counts):
                    for vertex, index in zip(mesh.indices[start : start + size], uv.indices[start : start + size]):
                        found = cmds.polyListComponentConversion(
                            f"{shape}.vtxFace[{vertex}][{face}]", fromVertexFace=True, toUV=True
                        )
                        np.testing.assert_allclose(cmds.polyEditUV(found, q=True), uv.points[index], rtol=0, atol=1e-12)
                    start += size



class TestSaveFbxSkins(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_round_trip(self):
        """the skin reader reads back every weight and influence exactly, the
        rig read at its bind holds the rig's worlds, and the inverse binds the
        file holds pose the mesh where it is; a mesh named after a group of
        the rig is written on it"""
        from cgmath.geometry.skin_weights import _read_skins_fbx

        for group in (None, "body"):
            with self.subTest(group=group):
                rig, mesh, skin = skinned(group=group)
                formats.save_fbx(self.path("skin.fbx"), skin, mesh, rig)
                back = SkinList.load_fbx(self.path("skin.fbx"))
                self.assertEqual((back.name, back[0].influences), (["body"], ["root", "mid", "tip"]))
                np.testing.assert_array_equal(back[0].weights, skin.weights)
                bound = HierarchyData.load_fbx(self.path("skin.fbx"), pose="bind", axis_system=None)
                for node in rig:
                    np.testing.assert_allclose(np.asarray(bound[node.name].world_matrix), np.asarray(node.world_matrix),
                                               rtol=0, atol=1e-12)
                placed = np.asarray(rig["body"].world_matrix) if group else np.eye(4)
                np.testing.assert_allclose(np.asarray(bound["body"].world_matrix), placed, rtol=0, atol=1e-12)
                ((_, inverses),) = _read_skins_fbx(self.path("skin.fbx"), bind_matrices=True)
                for node, inverse in zip(rig[:3], inverses):
                    want = placed @ np.linalg.inv(np.asarray(node.world_matrix))
                    np.testing.assert_allclose(inverse, want, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(MeshData.load_fbx(self.path("skin.fbx")).points, mesh.points)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_refusals(self):
        """each raises before the sdk is reached, naming what fails"""
        rig, mesh, skin = skinned()
        twin            = mesh.copy()
        twin.name       = "twin"
        stray           = SkinData(weights=skin.weights, influences=["root", "mid", "toe"], name="body")
        short           = SkinData(weights=skin.weights[:6], influences=skin.influences, name="body")
        jointed         = mesh.copy()
        jointed.name    = "mid"
        moved           = skinned(group="body")
        moved[1].matrix = TransformData("m", translate=(1.0, 0.0, 0.0)).matrix
        repeated, _, _ = skinned()
        repeated[2].name = "mid"
        cases = (
            ("a skin with no rig", (mesh, skin), TypeError, "with the rig"),
            ("a skin with no mesh", (rig, skin), TypeError, "with its mesh"),
            ("two rigs", (mesh, rig, skinned()[0]), ValueError, "2 rigs"),
            ("an influence the rig lacks", (mesh, rig, stray), ValueError, r"influences .*\['toe'\]"),
            ("two meshes fit", (mesh, twin, rig, SkinData(weights=skin.weights, influences=skin.influences)),
             ValueError, r"fits meshes \['body', 'twin'\]"),
            ("other rows", (mesh, rig, short), ValueError, "weighs 6 vertices, mesh 'body' holds 12"),
            ("two skins on a mesh", (mesh, rig, SkinList([skin, skin.copy()])), ValueError, "two skins on mesh 'body'"),
            ("a mesh named after a joint", (jointed, rig), ValueError, "named after the rig's joint"),
            ("a matrix and a group", moved[:2], ValueError, "has a matrix and a group"),
            ("repeated rig names", (mesh, repeated), ValueError, r"names repeat.*\['mid'\]"),
        )
        for label, objects, error, message in cases:
            with self.subTest(label), self.assertRaisesRegex(error, message):
                formats.save_fbx(self.path("x.fbx"), *objects)
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_maya_binds_the_mesh(self):
        """Maya 2025's FBXImport: one skinCluster, its influences and every
        vertex's weights to 1e-12, a bindPose holding the rig's worlds, and
        Go to Bind Pose, after a joint is turned, putting the mesh back where
        cgmath has it"""
        from cgmath._tests.test_fbx_labels import _maya

        maya = _maya()
        if maya is None:
            self.skipTest("needs Maya and its fbxmaya plug-in")
        cmds, mel = maya
        self.addCleanup(cmds.file, new=True, force=True)
        rig, mesh, skin = skinned(group="body")
        formats.save_fbx(self.path("maya.fbx"), mesh, rig, skin)
        cmds.file(new=True, force=True)
        mel.eval("FBXResetImport")
        mel.eval(f'FBXImport -f "{self.path("maya.fbx").replace(os.sep, "/")}"')
        (cluster,) = cmds.ls(type="skinCluster")
        self.assertEqual(cmds.skinCluster(cluster, q=True, influence=True), ["root", "mid", "tip"])
        (shape,) = cmds.listRelatives("body", shapes=True, noIntermediate=True, fullPath=True)
        for row in range(12):
            got = cmds.skinPercent(cluster, f"{shape}.vtx[{row}]", q=True, value=True)
            np.testing.assert_allclose(got, skin.weights[row], rtol=0, atol=1e-12)
        (pose,) = cmds.ls(type="dagPose")
        held = {}
        for index in cmds.getAttr(f"{pose}.members", multiIndices=True):
            (member,) = cmds.listConnections(f"{pose}.members[{index}]", source=True, destination=False)
            held[member] = np.array(cmds.getAttr(f"{pose}.worldMatrix[{index}]")).reshape(4, 4)
        # Maya keeps the joints, as its own bindPose holds them, 2.47e-07
        # off the worlds the file holds in this rig's turned axes
        self.assertEqual(sorted(held), ["mid", "root", "tip"])
        for node in rig[:3]:
            np.testing.assert_allclose(held[node.name], np.asarray(node.world_matrix), rtol=0, atol=1e-6)
        placed = mesh.points @ np.asarray(rig["body"].world_matrix)[:3, :3] + np.asarray(rig["body"].world_matrix)[3, :3]
        for turned in (False, True):
            if turned:
                cmds.setAttr("mid.rotateX", 30.0)
                moved = np.array(cmds.xform(f"{shape}.vtx[0:11]", q=True, ws=True, t=True)).reshape(-1, 3)
                self.assertGreater(float(np.abs(moved - placed).max()), 0.1)
                cmds.dagPose("root", restore=True, g=True, bindPose=True)
            got = np.array(cmds.xform(f"{shape}.vtx[0:11]", q=True, ws=True, t=True)).reshape(-1, 3)
            np.testing.assert_allclose(got, placed, rtol=0, atol=1e-5)



class TestSaveFbxMany(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def read_back(self, path):
        """the objects a file holds, as cgmath's readers read them"""
        return (
            HierarchyData.load_fbx(path, pose="bind", axis_system=None),
            MeshData.load_fbx(path),
            UVList.load_fbx(path, name="body"),
            SkinList.load_fbx(path),
        )

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_the_users_example_in_any_order(self):
        """save_fbx(path, mesh, uvlist, hier, skin), its objects in any
        order, writes one file: the readers read back the same objects"""
        import itertools

        rig, mesh, skin = skinned(group="body")
        uvs           = UVList([column_uvs(mesh), column_uvs(mesh, "flipped")])
        uvs[1].points = uvs[1].points[:, ::-1].copy()
        first         = None
        for order in itertools.permutations((mesh, uvs, rig, skin)):
            formats.save_fbx(self.path("order.fbx"), *order)
            bound, back, sets, skins = self.read_back(self.path("order.fbx"))
            self.assertEqual(sets.name, ["map1", "flipped"])
            for a, b in zip(uvs, sets):
                np.testing.assert_array_equal(b.points, a.points)
            np.testing.assert_array_equal(back.points, mesh.points)
            np.testing.assert_array_equal(skins[0].weights, skin.weights)
            worlds = np.array(bound.world_matrix)
            if first is None:
                first = worlds
            np.testing.assert_array_equal(worlds, first)
        np.testing.assert_allclose(first, np.array(rig.world_matrix), rtol=0, atol=1e-12)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_takes_on_a_rig(self):
        """takes beside a rig key the rig's nodes of their names, the rig's
        others unkeyed, the rig's pose the bind; two clips of more than one
        frame are two takes"""
        rig, mesh, skin = skinned(group="body")
        clips = joint_takes(rig)
        formats.save_fbx(self.path("takes.fbx"), clips, skin, rig, mesh)
        takes = ClipList.load_fbx(self.path("takes.fbx"), axis_system=None)
        self.assertEqual((takes.take, takes.frame_count), (["take0", "take1"], [4, 3]))
        for clip, back in zip(clips, takes):
            for node in clip:
                column = back.index(back[node.name])
                for channel in ("translate", "rotate"):
                    want = np.asarray(getattr(clip.frames, channel))[:, clip.index(node)]
                    got  = np.asarray(getattr(back.frames, channel))[:, column]
                    np.testing.assert_allclose(got, want.astype(np.float32), rtol=0, atol=1e-9)
            group = back.index(back["body"])
            self.assertTrue(np.allclose(np.asarray(back.frames.translate)[:, group], (0.5, 0.0, 0.25)))
        bound = HierarchyData.load_fbx(self.path("takes.fbx"), pose="bind", axis_system=None)
        np.testing.assert_allclose(np.array(bound.world_matrix), np.array(rig.world_matrix), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(SkinList.load_fbx(self.path("takes.fbx"))[0].weights, skin.weights)

        formats.save_fbx(self.path("two.fbx"), clips[0], clips[1])
        self.assertEqual(ClipList.load_fbx(self.path("two.fbx")).take, ["take0", "take1"])

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_takes_oriented_other_than_the_rig(self):
        """a take whose joint orient or rotate axis is not the rig's keys the
        difference in its rotate: every node's world on every frame is the
        take's, read back, and the rig's own orient is the file's"""
        rig, mesh, skin = skinned(group="body")
        clips = joint_takes(rig)
        clips.frames.joint_orient = PerTake(
            [np.full(np.asarray(clip.frames.joint_orient).shape, 3.0) for clip in clips]
        )
        clips.frames.rotate_axis = PerTake(
            [np.full(np.asarray(clip.frames.rotate_axis).shape, -4.0) for clip in clips]
        )
        formats.save_fbx(self.path("oriented.fbx"), rig, clips, mesh, skin)
        takes = ClipList.load_fbx(self.path("oriented.fbx"), axis_system=None)
        for clip, back in zip(clips, takes):
            for node in clip:
                want = np.asarray(clip.frames.world_matrix)[:, clip.index(node)]
                got  = np.asarray(back.frames.world_matrix)[:, back.index(back[node.name])]
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)
        bound = HierarchyData.load_fbx(self.path("oriented.fbx"), pose="bind", axis_system=None)
        np.testing.assert_allclose(np.asarray(bound.joint_orient), np.asarray(rig.joint_orient), rtol=0, atol=1e-12)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_shear_is_refused(self):
        """a rig's node or a take's frame holding a shear over 1e-9 raises
        ValueError naming them, nothing written; 1e-9 and an uneven scale
        without shear save"""
        rig, mesh, skin = skinned(group="body")
        clips = joint_takes(rig)
        formats.save_fbx(self.path("plain.fbx"), rig, clips, mesh, skin)
        rig["body"].scale = (1.0, 2.0, 0.5)
        rig["mid"].shear  = (0.0, 1e-9, -1e-9)
        formats.save_fbx(self.path("plain.fbx"), rig, clips, mesh, skin)
        np.testing.assert_array_equal(HierarchyData.load_fbx(self.path("plain.fbx"), axis_system=None).shear, 0.0)

        sheared              = rig.copy()
        sheared["mid"].shear = (0.0, 2e-9, 0.0)
        with self.assertRaisesRegex(ValueError, r"no shear.*the rig's \['mid'\]"):
            formats.save_fbx(self.path("x.fbx"), sheared, clips, mesh, skin)
        block                 = np.array(clips[1].frames.shear)
        block[2, 2]           = (0.0, 0.0, 0.25)
        clips[1].frames.shear = block
        for objects in ((rig, clips, mesh, skin), (clips,), (clips[1],)):
            with self.subTest(len(objects)), self.assertRaisesRegex(ValueError, r"take 'take1''s \['tip'\] \(frames 2 to 2\)"):
                formats.save_fbx(self.path("x.fbx"), *objects)
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_a_joint_orient_off_a_joint_is_refused(self):
        """a group's or a locator's joint orient over 1e-9, in the rig or on
        a take's frame, raises ValueError naming them, nothing written; a
        zero one saves, and reads back zero"""
        rig, mesh, skin = skinned(group="body")
        rig["body"].joint_orient = (0.0, -0.0, 1e-9)
        clip                     = ClipData(rig, frames=4, take="move")
        formats.save_fbx(self.path("zero.fbx"), rig, ClipList([clip]), mesh, skin)
        back = HierarchyData.load_fbx(self.path("zero.fbx"), axis_system=None)
        np.testing.assert_array_equal(back["body"].joint_orient, 0.0)

        oriented                      = rig.copy()
        oriented["body"].joint_orient = (0.0, 2e-9, 0.0)
        for objects in ((oriented, mesh, skin), (oriented,), (oriented, ClipList([clip]))):
            with self.subTest(len(objects)), self.assertRaisesRegex(ValueError, r"no joint orient.*the rig's \['body'\]"):
                formats.save_fbx(self.path("x.fbx"), *objects)
        with self.assertRaisesRegex(ValueError, r"the rig's \['body'\]"):
            oriented.save_fbx(self.path("x.fbx"))
        block                    = np.array(clip.frames.joint_orient)
        block[1:3, 3]            = (0.0, 0.0, 30.0)
        clip.frames.joint_orient = block
        for objects in ((rig, ClipList([clip]), mesh, skin), (ClipList([clip]),), (clip,)):
            with self.subTest(len(objects)), self.assertRaisesRegex(ValueError, r"take 'move''s \['body'\] \(frames 1 to 2\)"):
                formats.save_fbx(self.path("x.fbx"), *objects)
        with self.assertRaisesRegex(ValueError, r"take 'move''s \['body'\]"):
            clip.save_fbx(self.path("x.fbx"))
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_takes_that_do_not_fit_the_rig(self):
        """each raises before the sdk is reached, naming what fails"""
        rig, mesh, _ = skinned(group="body")
        stray = joint_takes(rig)
        for clip in stray:
            clip[2].name = "toe"
        relabelled      = joint_takes(rig)
        relabelled.unit = 100.0
        repeated, _, _ = skinned(group="body")
        repeated[3].name = "tip"
        cases = (
            ("a node the rig lacks", (rig, stray), r"nodes the rig does not: \['toe'\]"),
            ("another unit", (rig, relabelled), "one up axis and one unit"),
            ("repeated rig names", (repeated, joint_takes(rig)), r"names repeat.*\['tip'\]"),
        )
        for label, objects, message in cases:
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                formats.save_fbx(self.path("x.fbx"), *objects)
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_what_it_leaves_out(self):
        """blend shapes, cameras and lights are left out with one warning
        naming them, the rest written; a render Scene, or any other object,
        raises TypeError"""
        from cgmath.geometry.morph_target import MorphData, MorphList
        from cgmath.render.scene import Camera, Light, Scene

        rig, mesh, skin = skinned()
        morph = MorphData("smile", np.zeros((12, 3)))
        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as caught:
            formats.save_fbx(self.path("left.fbx"), mesh, MorphList([morph]), rig, Camera("cam"), skin,
                             Light("key"), Light("fill"))
        left = [record.getMessage() for record in caught.records if "left out" in record.getMessage()]
        self.assertEqual(len(left), 1)
        self.assertIn("1 Camera, 2 Light, 1 MorphList", left[0])
        self.assertEqual(SkinList.load_fbx(self.path("left.fbx")).name, ["body"])
        for item in (Scene(), object()):
            with self.subTest(type(item).__name__), self.assertRaisesRegex(TypeError, "not a"):
                formats.save_fbx(self.path("x.fbx"), mesh, item)
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_maya_imports_the_users_example_at_the_bind(self):
        """Maya 2025's FBXImport of save_fbx(path, mesh, uvlist, hier,
        skin): the joints at the rig's worlds, the UV set, one skinCluster,
        and the mesh where cgmath has it, posed at the bind"""
        from cgmath._tests.test_fbx_labels import _maya

        maya = _maya()
        if maya is None:
            self.skipTest("needs Maya and its fbxmaya plug-in")
        cmds, mel = maya
        self.addCleanup(cmds.file, new=True, force=True)
        rig, mesh, skin = skinned(group="body")
        formats.save_fbx(self.path("maya.fbx"), mesh, UVList([column_uvs(mesh)]), rig, skin)
        cmds.file(new=True, force=True)
        mel.eval("FBXResetImport")
        mel.eval(f'FBXImport -f "{self.path("maya.fbx").replace(os.sep, "/")}"')
        for node in rig:
            got = np.array(cmds.xform(node.name, q=True, ws=True, m=True)).reshape(4, 4)
            np.testing.assert_allclose(got, np.asarray(node.world_matrix), rtol=0, atol=1e-6)
        (shape,) = cmds.listRelatives("body", shapes=True, noIntermediate=True, fullPath=True)
        self.assertEqual(cmds.polyUVSet(shape, q=True, allUVSets=True), ["map1"])
        self.assertEqual(len(cmds.ls(type="skinCluster")), 1)
        world  = np.asarray(rig["body"].world_matrix)
        placed = mesh.points @ world[:3, :3] + world[3, :3]
        got    = np.array(cmds.xform(f"{shape}.vtx[0:11]", q=True, ws=True, t=True)).reshape(-1, 3)
        np.testing.assert_allclose(got, placed, rtol=0, atol=1e-5)



def rewrite(path, out, edit):
    """``path`` read with the sdk, ``edit(scene)`` done, written to ``out``"""
    from cgmath.formats._fbx_io import import_scene, new_manager

    manager = new_manager(FBX)
    try:
        scene = import_scene(manager, path, FBX)
        edit(scene)
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(out, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()


class TestReadBackAsWritten(unittest.TestCase):
    """what save_fbx writes reads back as it was, and a load saved and read
    again reads the same"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_rotate_axis(self):
        """a joint's rotate axis reads back as written, though the post
        rotation holds its inverse, whose angles read as other angles for
        the same turn or as the same a few ulps off; a joint with none
        reads 0.0, not -0.0. A file with no record of it reads the post
        rotation's angles, the same turn"""
        axes = [(0.0, 0.0, 0.0), (10.5, -20.25, 30.125), (170.0, 120.0, -100.0), (33.3, 89.9, -0.1),
                (1e-3, -45.0, 179.0), (-91.7, 13.1, 66.6)]
        rig = HierarchyData()
        for index, axis in enumerate(axes):
            node = TransformData(f"j{index}", node_type="joint", translate=(0.0, 1.0, 0.0), rotate=(5.0, 6.0, 7.0),
                                 rotate_axis=axis, rotate_order=index % 6, joint_orient=(0.0, 3.0 * index, 0.0))
            if index:
                node.parent_uuid = rig[index - 1].uuid
            rig.append(node)
        rig.save_fbx(self.path("axes.fbx"))
        back = HierarchyData.load_fbx(self.path("axes.fbx"), axis_system=None)
        np.testing.assert_array_equal(back.rotate_axis, rig.rotate_axis)
        held = np.asarray(back.rotate_axis)
        self.assertFalse(np.signbit(held[held == 0.0]).any())
        np.testing.assert_allclose(np.asarray(back.world_matrix), np.asarray(rig.world_matrix), rtol=0, atol=1e-12)

        clip = ClipData(rig, frames=3, take="a")
        clip.save_fbx(self.path("take.fbx"))
        (take,) = ClipList.load_fbx(self.path("take.fbx"), axis_system=None)
        np.testing.assert_array_equal(take.frames.rotate_axis, clip.frames.rotate_axis)

        def forget(scene):
            for index in range(scene.GetNodeCount()):
                prop = scene.GetNode(index).FindProperty("cgmath:rotateAxis")
                if prop.IsValid():
                    prop.Destroy()

        rewrite(self.path("axes.fbx"), self.path("plain.fbx"), forget)
        plain = HierarchyData.load_fbx(self.path("plain.fbx"), axis_system=None)
        np.testing.assert_allclose(np.asarray(plain.world_matrix), np.asarray(rig.world_matrix), rtol=0, atol=1e-12)
        self.assertGreater(np.abs(np.asarray(plain.rotate_axis) - np.asarray(rig.rotate_axis)).max(), 90.0)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_normals(self):
        """normals read at unit length, as Maya's import makes them, but one
        unit to within 8 ulps reads as the file holds it, so a mesh read,
        saved and read again reads the same; normals one a vertex read with
        no normal_indices, as the mesh held them"""
        rng    = np.random.default_rng(7)
        points = np.stack(np.meshgrid(np.arange(21.0), np.arange(21.0)), axis=-1).reshape(-1, 2)
        quads  = [[r * 21 + c, r * 21 + c + 1, r * 21 + c + 22, r * 21 + c + 21] for r in range(20) for c in range(20)]
        mesh = MeshData(points=np.column_stack([points, np.zeros(len(points))]), indices=np.ravel(quads),
                        counts=np.full(len(quads), 4), name="grid")
        for label in ("vertex", "corner"):
            with self.subTest(label):
                held                = rng.normal(size=(len(points) if label == "vertex" else len(mesh.indices), 3)) * 3.0
                mesh.normals        = held
                mesh.normal_indices = None if label == "vertex" else np.arange(len(mesh.indices))
                mesh.save_fbx(self.path("normals.fbx"))
                first = MeshData.load_fbx(self.path("normals.fbx"))
                np.testing.assert_allclose(first.normals, _l2_normalize(held), rtol=0, atol=1e-15)
                if label == "vertex":
                    self.assertIsNone(first.normal_indices)
                first.save_fbx(self.path("again.fbx"))
                again = MeshData.load_fbx(self.path("again.fbx"))
                np.testing.assert_array_equal(again.normals, first.normals)


if __name__ == "__main__":
    unittest.main()
