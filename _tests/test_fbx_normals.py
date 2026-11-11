"""Shading normals as Maya derives them: its rule, measured.

Where a mesh holds no authored normal, Maya 2025 derives one for each face
corner. The faces around the corner's vertex that soft edges join are a fan;
each face of the fan gives its Newell vector (its normal, as long as twice
the area of the face's outline seen along it) times the angle of its corner
at the vertex, and the sum is made unit. A hard edge parts two fans; the
edges of a border are marked hard and part nothing. A locked normal is the
one authored, whatever the faces say.

MeshData computes the same normals: ``get_vertex_normals()`` with every edge
soft, ``set_normals(hard_edges=...)`` inside the groups the hard edges
leave. A mesh need not store them: ``get_face_vertex_normals()`` gives every
corner's normal whatever the mesh stores, derived by the same rule between
the edges ``hard_edges`` names when no normal is stored.
``load_fbx`` stores the least normals that shade a mesh as its file does:
none where the file's are the derived ones, with the edges that part them in
``hard_edges``, the file's own where they are authored (locked, in Maya).
The tests read what Maya 2025 held on 15 meshes, recorded once
(maya_fixtures, case derived_normals), and need no Maya.
"""

import unittest

import numpy as np
from cgmath._tests.test_skeleton_fbx import _Recorded
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk
from cgmath.formats._load import _least_normals
from cgmath.geometry import MeshData
from cgmath.geometry.mesh import MeshList, UVData, _corner_normals
from cgmath.geometry.resample import MeshDataResampler

try:
    from pxr import Usd
except ImportError:
    Usd = None

FBX = load_sdk()

# a component; Maya hands its normals out as floats
TOLERANCE = 1e-5


def held_mesh(case, name: str) -> MeshData:
    """the recorded mesh: Maya's points, face counts and corner vertices, no normals"""
    return MeshData(points=case.array(f"{name}/points"), indices=case.array(f"{name}/indices"), counts=case.array(f"{name}/counts"))


def hard_edges(case, name: str) -> np.ndarray:
    """the edges Maya marked hard, a pair of vertices each, the lower first"""
    return np.sort(case.array(f"{name}/edges")[~case.array(f"{name}/smooth")], axis=1)


def hard_mesh(case, name: str) -> MeshData:
    """the recorded mesh with no normal stored and the edges Maya marked
    hard in ``hard_edges``"""
    mesh            = held_mesh(case, name)
    mesh.hard_edges = hard_edges(case, name)
    return mesh


def groups(mesh: MeshData) -> np.ndarray:
    """the group each face corner shades with, named by the first corner of
    it: the same for two meshes whose hard edges part the same corners"""
    ids = np.asarray(mesh.indices) if mesh.hard_edges is None else mesh._get_hard_groups()[0]
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    return first[inverse]


def pair_set(pairs) -> set:
    """vertex pairs as a set, the lower vertex first"""
    return {tuple(pair) for pair in np.sort(np.asarray(pairs).reshape(-1, 2), axis=1).tolist()}


def stored_mesh(case, name: str) -> MeshData:
    """the recorded mesh with its normals stored, parted at Maya's hard edges"""
    mesh = held_mesh(case, name)
    mesh.set_normals(hard_edges=hard_edges(case, name))
    return mesh


def corner_uvs(mesh: MeshData) -> UVData:
    """a uv a face corner, each face laid out as a regular polygon"""
    counts = np.asarray(mesh.counts)
    total  = int(counts.sum())
    step   = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    angle  = 2.0 * np.pi * step / np.repeat(counts, counts)
    return UVData(points=np.stack([np.cos(angle), np.sin(angle)], axis=1), indices=np.arange(total), counts=counts)


class TestMayaDerivedNormals(_Recorded):
    """the normals Maya 2025 derives where none is authored, as recorded,
    against the ones MeshData computes"""

    def meshes(self, kind: str):
        case = self.case("derived_normals")
        info = case.value("info")
        return case, info, [name for name in case.value("meshes") if info[name]["kind"] == kind]

    def test_every_edge_soft(self):
        """one normal a vertex: get_vertex_normals() gives Maya's"""
        case, info, names = self.meshes("soft")
        self.assertEqual(len(names), 5)
        for name in names:
            with self.subTest(name):
                mesh    = held_mesh(case, name)
                indices = np.asarray(mesh.indices)
                maya    = case.array(f"{name}/normals")
                self.assertEqual(info[name]["locked_corners"], 0)
                self.assertEqual(info[name]["normals"], len(mesh.points))
                got = mesh.get_vertex_normals()[indices]
                np.testing.assert_allclose(got, maya, rtol=0, atol=TOLERANCE)
                # the faces bend: weighed by get_face_areas(), the area of
                # the surface and not of the outline, the normals are others
                by_surface = mesh.get_vertex_normals(_areas=mesh.get_face_areas(), force=True)[indices]
                self.assertGreater(np.abs(by_surface - maya).max(), 1e-3)

    def test_border_edges_are_marked_hard_and_part_nothing(self):
        """an open sheet softened all over: Maya marks its 18 border edges
        hard and still holds one normal a vertex"""
        case, info, names = self.meshes("soft")
        self.assertEqual([name for name in names if info[name]["hard_edges"]], ["softSheet"])
        mesh  = held_mesh(case, "softSheet")
        edges = hard_edges(case, "softSheet")
        self.assertEqual(len(edges), 18)
        faces = np.asarray(mesh.indices).reshape(-1, 4)
        pairs = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
        used  = {}
        for pair in pairs.tolist():
            used[tuple(pair)] = used.get(tuple(pair), 0) + 1
        self.assertEqual(sorted(tuple(pair) for pair in edges.tolist()), sorted(pair for pair, count in used.items() if count == 1))
        mesh.set_normals(hard_edges=edges)
        slots = np.asarray(mesh.indices) if mesh.normal_indices is None else np.asarray(mesh.normal_indices)
        self.assertEqual(len(mesh.normals), len(mesh.points))
        np.testing.assert_allclose(np.asarray(mesh.normals)[slots], case.array("softSheet/normals"), rtol=0, atol=TOLERANCE)

    def test_hard_edges(self):
        """set_normals(hard_edges=...) on the edges Maya marked hard gives
        Maya's normals, in Maya's groups of corners"""
        case, info, names = self.meshes("hard")
        self.assertEqual(len(names), 9)
        for name in names:
            with self.subTest(name):
                mesh  = held_mesh(case, name)
                maya  = case.array(f"{name}/normals")
                ids   = case.array(f"{name}/normal_ids")
                edges = hard_edges(case, name)
                self.assertEqual(info[name]["locked_corners"], 0)
                self.assertEqual(len(edges), info[name]["hard_edges"])
                mesh.set_normals(hard_edges=edges)
                slots = np.asarray(mesh.normal_indices)
                np.testing.assert_allclose(np.asarray(mesh.normals)[slots], maya, rtol=0, atol=TOLERANCE)
                # two corners share a normal here where they share one in Maya
                self.assertEqual(len(mesh.normals), info[name]["normals"])
                self.assertEqual(len(set(zip(ids.tolist(), slots.tolist()))), info[name]["normals"])
                # with every edge soft the normals are others
                soft = held_mesh(case, name).get_vertex_normals()[np.asarray(mesh.indices)]
                self.assertGreater(np.abs(soft - maya).max(), 1e-3)

    def test_groups_of_several_faces_follow_the_rule(self):
        """the jittered meshes softened by an angle hold groups of several
        bent faces: their normals are not their faces' own"""
        case, info, names = self.meshes("hard")
        several = [name for name in names if info[name]["jittered"] and info[name]["normals"] < case.array(f"{name}/indices").size]
        self.assertEqual(several, ["canJittered", "coneJittered", "ballSoft30", "ringSoft45", "sheetSoft20"])
        for name in several:
            with self.subTest(name):
                mesh = held_mesh(case, name)
                flat = np.repeat(mesh.get_face_normals(), np.asarray(mesh.counts), axis=0)
                self.assertGreater(np.abs(flat - case.array(f"{name}/normals")).max(), 0.1)

    def test_a_sphere_as_maya_makes_it_is_all_hard(self):
        """Maya's polySphere: every edge hard, a normal a corner, each its
        face's own"""
        case, info, names = self.meshes("hard")
        self.assertIn("ballDefault", names)
        mesh = held_mesh(case, "ballDefault")
        self.assertFalse(case.array("ballDefault/smooth").any())
        self.assertEqual(info["ballDefault"]["normals"], np.asarray(mesh.indices).size)
        flat = np.repeat(mesh.get_face_normals(), np.asarray(mesh.counts), axis=0)
        np.testing.assert_allclose(flat, case.array("ballDefault/normals"), rtol=0, atol=TOLERANCE)

    def test_a_locked_normal_is_the_one_authored(self):
        """a soft cube with vertex 0's normal set to +Y: its three corners
        are locked and hold +Y, every other corner follows the rule"""
        case, info, names = self.meshes("locked")
        self.assertEqual(names, ["lockedCube"])
        mesh    = held_mesh(case, "lockedCube")
        indices = np.asarray(mesh.indices)
        maya    = case.array("lockedCube/normals")
        locked  = case.array("lockedCube/locked")
        self.assertEqual(info["lockedCube"]["hard_edges"], 0)
        self.assertEqual((int(locked.sum()), int((indices == 0).sum())), (3, 3))
        self.assertTrue(np.all(indices[locked] == 0))
        np.testing.assert_allclose(maya[locked], [[0.0, 1.0, 0.0]] * 3, rtol=0, atol=1e-7)
        got = mesh.get_vertex_normals()[indices]
        np.testing.assert_allclose(got[~locked], maya[~locked], rtol=0, atol=TOLERANCE)
        self.assertGreater(np.abs(got[locked] - maya[locked]).max(), 1e-3)


class _Derived(_Recorded):
    """a test over the case derived_normals: its readings, the meshes of a
    kind, and Maya's normals of one, a row a face corner"""

    HARD = "canJittered"
    SOFT = "softCan"

    def setUp(self):
        super().setUp()
        self.readings = self.case("derived_normals")
        self.info     = self.readings.value("info")

    def names(self, kind: str):
        return [name for name in self.readings.value("meshes") if self.info[name]["kind"] == kind]

    def maya(self, name: str) -> np.ndarray:
        return self.readings.array(f"{name}/normals")


class TestNormalsNotStored(_Derived):
    """a mesh that stores no normal: ``get_face_vertex_normals()`` derives
    Maya's from its points, broken at the edges ``hard_edges`` names, and
    the writers, the tangent space and the resampler's rays read those"""

    def assertDerived(self, back: MeshData, name: str) -> None:
        """a mesh read back from a file gives Maya's normals of ``name``,
        parted at its hard edges"""
        got = back.get_face_vertex_normals()
        np.testing.assert_allclose(got, self.maya(name), rtol=0, atol=TOLERANCE)
        soft = held_mesh(self.readings, name).get_face_vertex_normals()
        self.assertGreater(np.abs(soft - got).max(), 1e-3)

    def test_every_edge_soft(self):
        """no normal and no hard edge: the vertex normals, one a corner"""
        names = self.names("soft")
        self.assertEqual(len(names), 5)
        for name in names:
            with self.subTest(name):
                mesh = held_mesh(self.readings, name)
                got  = mesh.get_face_vertex_normals()
                self.assertIsNone(mesh.normals)
                self.assertIsNone(mesh.normal_indices)
                np.testing.assert_array_equal(got, mesh.get_vertex_normals()[np.asarray(mesh.indices)])
                np.testing.assert_allclose(got, self.maya(name), rtol=0, atol=TOLERANCE)

    def test_hard_edges_derive_mayas_normals(self):
        """no normal, the edges Maya marked hard in ``hard_edges``: Maya's
        normals"""
        names = self.names("hard")
        self.assertEqual(len(names), 9)
        for name in names:
            with self.subTest(name):
                mesh = hard_mesh(self.readings, name)
                got  = mesh.get_face_vertex_normals()
                self.assertIsNone(mesh.normals)
                self.assertIsNone(mesh.normal_indices)
                self.assertEqual(_corner_normals(mesh)[0].shape[0], self.info[name]["normals"])
                np.testing.assert_allclose(got, self.maya(name), rtol=0, atol=TOLERANCE)
                # with every edge soft the normals are others
                soft = held_mesh(self.readings, name).get_face_vertex_normals()
                self.assertGreater(np.abs(soft - got).max(), 1e-3)

    def test_stored_normals_come_back_as_held(self):
        """indexed, one a vertex, one a corner: the normals stored, none
        derived"""
        mesh                = held_mesh(self.readings, self.HARD)
        indices             = np.asarray(mesh.indices)
        rng                 = np.random.default_rng(5)
        mesh.normals        = rng.normal(size=(7, 3))
        mesh.normal_indices = rng.integers(0, 7, indices.size)
        np.testing.assert_array_equal(mesh.get_face_vertex_normals(), np.asarray(mesh.normals)[np.asarray(mesh.normal_indices)])
        mesh.normal_indices = None
        mesh.normals        = rng.normal(size=(len(mesh.points), 3))
        np.testing.assert_array_equal(mesh.get_face_vertex_normals(), np.asarray(mesh.normals)[indices])
        mesh.normals = rng.normal(size=(indices.size, 3))
        np.testing.assert_array_equal(mesh.get_face_vertex_normals(), np.asarray(mesh.normals))

    def test_force_sees_points_written_in_place(self):
        """the corner angles kept are those of the points assigned: after a
        write inside the array, force=True gives the normals of a mesh
        assigned the same points"""
        mesh        = hard_mesh(self.readings, self.HARD)
        mesh.points = np.array(mesh.points)  # its own array, not the readings'
        before      = mesh.get_face_vertex_normals()
        mesh.points[:, 1] *= 3.0
        forced = mesh.get_face_vertex_normals(force=True)
        self.assertGreater(np.abs(forced - before).max(), 1e-3)
        assigned        = hard_mesh(self.readings, self.HARD)
        assigned.points = np.asarray(assigned.points) * [1.0, 3.0, 1.0]
        np.testing.assert_allclose(forced, assigned.get_face_vertex_normals(), rtol=0, atol=1e-12)

    def test_a_dict_keeps_the_hard_edges_and_no_normal(self):
        mesh = hard_mesh(self.readings, self.HARD)
        back = MeshData.from_dict(mesh.to_dict())
        self.assertIsNone(back.normals)
        self.assertIsNone(back.normal_indices)
        np.testing.assert_array_equal(back.hard_edges, mesh.hard_edges)
        self.assertEqual(back, mesh)

    def test_obj(self):
        """save_obj writes the normals derived between the hard edges, one a
        group of corners, and none for a mesh with neither normals nor hard
        edges"""
        mesh = hard_mesh(self.readings, self.HARD)
        back = MeshData.load_obj(mesh.save_obj(self.path("hard.obj")))
        self.assertEqual(len(back.normals), self.info[self.HARD]["normals"])
        self.assertDerived(back, self.HARD)
        soft = held_mesh(self.readings, self.SOFT)
        self.assertIsNone(MeshData.load_obj(soft.save_obj(self.path("soft.obj"))).normals)

    @unittest.skipIf(Usd is None, "USD (pxr) is not installed")
    def test_usd(self):
        """to_prim writes the normals derived between the hard edges, one a
        group of corners, and none for a mesh with neither normals nor hard
        edges"""
        stage = Usd.Stage.CreateInMemory()
        prim  = stage.DefinePrim("/hard", "Mesh")
        hard_mesh(self.readings, self.HARD).to_prim(prim)
        back = MeshData.from_prim(prim)
        self.assertEqual(len(back.normals), self.info[self.HARD]["normals"])
        self.assertDerived(back, self.HARD)
        prim = stage.DefinePrim("/soft", "Mesh")
        held_mesh(self.readings, self.SOFT).to_prim(prim)
        self.assertIsNone(MeshData.from_prim(prim).normals)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_fbx(self):
        """save_fbx writes the normals derived between the hard edges, and
        none for a mesh with neither normals nor hard edges"""
        mesh      = hard_mesh(self.readings, self.HARD)
        mesh.name = "hardShape"
        save_fbx(self.path("hard.fbx"), meshes=mesh)
        self.assertDerived(load_fbx(self.path("hard.fbx")).meshes[0], self.HARD)
        soft      = held_mesh(self.readings, self.SOFT)
        soft.name = "softShape"
        save_fbx(self.path("soft.fbx"), meshes=soft)
        self.assertIsNone(load_fbx(self.path("soft.fbx")).meshes[0].normals)

    def test_the_tangent_space_reads_the_hard_edges(self):
        """hard edges and no normal give the tangent space of the same
        normals stored, not the one of a mesh all soft"""
        for name in ("canJittered", "cubeJittered", "ringSoft45"):
            with self.subTest(name):
                mesh = hard_mesh(self.readings, name)
                uv   = corner_uvs(mesh)
                tangents, bitangents = mesh.get_tangent_space(uv)
                stored_tangents, stored_bitangents = stored_mesh(self.readings, name).get_tangent_space(uv)
                np.testing.assert_allclose(tangents, stored_tangents, rtol=0, atol=1e-12)
                np.testing.assert_allclose(bitangents, stored_bitangents, rtol=0, atol=1e-12)
                soft_tangents, _ = held_mesh(self.readings, name).get_tangent_space(uv)
                self.assertGreater(np.abs(soft_tangents - tangents).max(), 1e-3)

    def test_one_normal_a_corner_needs_no_indices(self):
        """as many normals as face corners and no normal_indices: each
        corner reads its own"""
        mesh         = held_mesh(self.readings, "cubeJittered")
        flat         = np.repeat(mesh.get_face_normals(), np.asarray(mesh.counts), axis=0)
        mesh.normals = flat
        uv           = corner_uvs(mesh)
        np.testing.assert_array_equal(mesh.get_face_vertex_normals(), flat)
        tangents, bitangents = mesh.get_tangent_space(uv)
        mesh.normal_indices = np.arange(flat.shape[0])
        indexed_tangents, indexed_bitangents = mesh.get_tangent_space(uv)
        np.testing.assert_array_equal(tangents, indexed_tangents)
        np.testing.assert_array_equal(bitangents, indexed_bitangents)

    def test_normals_that_fit_nothing_are_refused(self):
        mesh         = held_mesh(self.readings, "cubeJittered")
        uv           = corner_uvs(mesh)
        mesh.normals = np.tile([0.0, 1.0, 0.0], (7, 1))
        for call in (mesh.get_face_vertex_normals, lambda: mesh.get_tangent_space(uv)):
            with self.assertRaisesRegex(ValueError, "7 normals, which fit neither its 8 vertices nor its 24 face corners"):
                call()

    def test_normal_indices_that_fit_no_corner_are_refused(self):
        mesh                = held_mesh(self.readings, "cubeJittered")
        mesh.normals        = np.tile([0.0, 1.0, 0.0], (8, 1))
        mesh.normal_indices = np.arange(5)
        with self.assertRaisesRegex(ValueError, "5 normal_indices for its 24 face corners"):
            mesh.get_face_vertex_normals()
        mesh.normal_indices = np.full(24, -1)
        with self.assertRaisesRegex(ValueError, r"name normals it does not hold \(8\)"):
            mesh.get_face_vertex_normals()

    def test_normal_indices_without_normals_are_refused(self):
        """normal_indices name stored normals: held with none, every reader
        of the normals refuses them"""
        mesh                = held_mesh(self.readings, "cubeJittered")
        mesh.name           = "cubeShape"
        mesh.normal_indices = np.arange(24)
        source              = held_mesh(self.readings, "cubeJittered")
        calls = [
            mesh.get_face_vertex_normals,
            lambda: mesh.get_tangent_space(corner_uvs(mesh)),
            lambda: mesh.save_obj(self.path("refused.obj")),
            lambda: MeshDataResampler(source, mesh)._get_ray_directions(True, True, True),
        ]
        if FBX is not None:
            calls.append(lambda: save_fbx(self.path("refused.fbx"), meshes=mesh))
        if Usd is not None:
            stage = Usd.Stage.CreateInMemory()
            calls.append(lambda: mesh.to_prim(stage.DefinePrim("/refused", "Mesh")))
        for call in calls:
            with self.assertRaisesRegex(ValueError, "holds normal_indices and no normals"):
                call()

    def test_resample_rays_follow_the_hard_edges(self):
        """a destination with hard edges and no normal casts along the
        normals derived between them, summed a vertex, as it does when they
        are stored"""
        for name in ("canJittered", "ringSoft45"):
            with self.subTest(name):
                source  = held_mesh(self.readings, name)
                derived = MeshDataResampler(source, hard_mesh(self.readings, name))
                stored  = MeshDataResampler(source, stored_mesh(self.readings, name))
                rays    = derived._get_ray_directions(True, True, True)
                np.testing.assert_allclose(rays, stored._get_ray_directions(True, True, True), rtol=0, atol=1e-12)
                # asked not to read the mesh's normals, the rays are others
                self.assertGreater(np.abs(derived._get_ray_directions(False, True, True) - rays).max(), 1e-3)


class TestLeastNormals(_Derived):
    """what load_fbx stores of the normals a file holds
    (``formats._load._least_normals``): none where they are the ones Maya
    derives from the points, with the hard edges where those part them, the
    file's own where they are authored. Read here off the recorded arrays,
    without the FBX SDK"""

    def read(self, name: str, normals=None, ids=None) -> dict:
        """the mesh ``name`` as the reader holds it: Maya's normals, one a
        face corner, or ``normals`` through ``ids``"""
        if normals is None:
            normals = self.maya(name).copy()
        if ids is None:
            ids = np.arange(len(normals), dtype=np.int32)
        return {
            "points":         self.readings.array(f"{name}/points"),
            "counts":         self.readings.array(f"{name}/counts"),
            "indices":        self.readings.array(f"{name}/indices"),
            "normals":        normals,
            "normal_indices": ids,
        }

    def assertAsRead(self, held: dict) -> None:
        """the normals of ``held`` and their indices come back themselves,
        with no hard edge"""
        normals, ids, hard = _least_normals(held)
        self.assertIs(normals, held["normals"])
        self.assertIs(ids, held["normal_indices"])
        self.assertIsNone(hard)

    def test_every_edge_soft(self):
        """Maya's normals of a mesh all soft: nothing is stored"""
        names = self.names("soft")
        self.assertEqual(len(names), 5)
        for name in names:
            with self.subTest(name):
                self.assertEqual(_least_normals(self.read(name)), (None, None, None))
        # one normal a vertex, the corners naming their vertices
        mesh = held_mesh(self.readings, self.SOFT)
        held = self.read(self.SOFT, mesh.get_vertex_normals(), np.asarray(mesh.indices))
        self.assertEqual(_least_normals(held), (None, None, None))

    def test_hard_edges_come_back_as_pairs(self):
        """Maya's normals of a mesh with hard edges: no normal is stored,
        the edges that part them come back, a pair of vertices each, all of
        them edges Maya marked hard; they leave the groups of corners Maya's
        own do, as many as Maya held normals"""
        names = self.names("hard")
        self.assertEqual(len(names), 9)
        for name in names:
            with self.subTest(name):
                normals, ids, hard = _least_normals(self.read(name))
                self.assertIsNone(normals)
                self.assertIsNone(ids)
                self.assertEqual(hard.dtype, np.int32)
                self.assertLessEqual(pair_set(hard), pair_set(hard_edges(self.readings, name)))
                mesh            = held_mesh(self.readings, name)
                mesh.hard_edges = hard
                np.testing.assert_array_equal(groups(mesh), groups(hard_mesh(self.readings, name)))
                self.assertEqual(mesh._get_hard_groups()[1], self.info[name]["normals"])

    def test_authored_normals_come_back_as_read(self):
        """a locked normal is not the derived one: the file's normals stay"""
        self.assertEqual(self.info["lockedCube"]["locked_corners"], 3)
        self.assertAsRead(self.read("lockedCube"))

    def test_a_normal_is_derived_within_1e_5_a_component(self):
        """one component of one corner's normal 9e-6 off the derived one:
        still not stored, the hard edges leaving the same groups; 1.1e-5
        off: authored"""
        for name, mesh in ((self.SOFT, held_mesh(self.readings, self.SOFT)), (self.HARD, hard_mesh(self.readings, self.HARD))):
            with self.subTest(name):
                ours  = mesh.get_face_vertex_normals()
                close = ours.copy()
                close[7, 1] += 9e-6
                normals, ids, hard = _least_normals(self.read(name, close))
                self.assertIsNone(normals)
                self.assertIsNone(ids)
                if mesh.hard_edges is None:
                    self.assertIsNone(hard)
                else:
                    back            = held_mesh(self.readings, name)
                    back.hard_edges = hard
                    np.testing.assert_array_equal(groups(back), groups(mesh))
                apart = ours.copy()
                apart[7, 1] += 1.1e-5
                self.assertAsRead(self.read(name, apart))

    def test_an_edge_of_three_faces(self):
        """three quads on one edge. Each face its own normals: that edge
        is hard, no normal stored. Two of the faces sharing theirs along it:
        no hard edge says so (an edge is hard for every face on it), the
        normals stay stored"""
        points = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        for angle in np.deg2rad([0.0, 120.0, 240.0]):
            points += [[np.cos(angle), np.sin(angle), 0.0], [np.cos(angle), np.sin(angle), 1.0]]
        points  = np.array(points)
        counts  = np.full(3, 4, dtype=np.int32)
        indices = np.array([0, 2, 3, 1, 0, 4, 5, 1, 0, 6, 7, 1], dtype=np.int32)
        own     = np.repeat(MeshData(points=points, counts=counts, indices=indices).get_face_normals(), 4, axis=0)

        def read(normals):
            return {"points": points, "counts": counts, "indices": indices, "normals": normals, "normal_indices": np.arange(12, dtype=np.int32)}

        def sharing(a, b):
            normals = own.copy()
            both    = own[a * 4] + own[b * 4]
            for corner in (0, 3):
                normals[a * 4 + corner] = normals[b * 4 + corner] = both / np.linalg.norm(both)
            return normals

        normals, ids, hard = _least_normals(read(own))
        self.assertIsNone(normals)
        self.assertIsNone(ids)
        self.assertEqual(hard.tolist(), [[0, 1]])
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertAsRead(read(sharing(a, b)))

    def test_what_fits_no_corner_is_left_as_read(self):
        """a mesh with no normals, or indices outside them: as read"""
        held            = self.read(self.SOFT)
        held["normals"] = held["normal_indices"] = None
        self.assertEqual(_least_normals(held), (None, None, None))
        for wrong in (-1, len(self.maya(self.SOFT))):
            held                      = self.read(self.SOFT)
            held["normal_indices"][3] = wrong
            self.assertAsRead(held)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestLoadFbxNormals(_Derived):
    """load_fbx stores the least normals that shade a mesh as its file does"""

    def test_mayas_export(self):
        """Maya's FBXExport of the 15 meshes. All soft: no normal and no
        hard edge. Hard edges: no normal, edges Maya marked hard that leave
        Maya's groups of corners. The locked cube: the file's normals. Each
        shades as Maya held it"""
        read   = load_fbx(self.fixture("maya.fbx"))
        nodes  = read.clip.active
        meshes = {nodes[mesh.parent_uuid].name: mesh for mesh in read.meshes}
        self.assertEqual(sorted(meshes), sorted(self.readings.value("meshes")))
        for name, mesh in meshes.items():
            kind = self.info[name]["kind"]
            with self.subTest(name):
                np.testing.assert_array_equal(mesh.indices, self.readings.array(f"{name}/indices"))
                np.testing.assert_allclose(mesh.get_face_vertex_normals(), self.maya(name), rtol=0, atol=TOLERANCE)
                if kind == "locked":
                    self.assertEqual(len(mesh.normals), self.info[name]["normals"])
                    self.assertEqual(len(mesh.normal_indices), len(mesh.indices))
                    self.assertIsNone(mesh.hard_edges)
                    continue
                self.assertIsNone(mesh.normals)
                self.assertIsNone(mesh.normal_indices)
                if kind == "soft":
                    self.assertIsNone(mesh.hard_edges)
                    continue
                self.assertLessEqual(pair_set(mesh.hard_edges), pair_set(hard_edges(self.readings, name)))
                np.testing.assert_array_equal(groups(mesh), groups(hard_mesh(self.readings, name)))
                self.assertEqual(mesh._get_hard_groups()[1], self.info[name]["normals"])

    def test_stored_derived_normals_are_not_read_back(self):
        """a mesh saved with the normals Maya would derive stored reads back
        with none, the hard edges that part them in ``hard_edges``, shading
        the same; saved and read again, the same mesh"""
        for name in (self.SOFT, self.HARD, "ballDefault", "sheetSoft20"):
            with self.subTest(name):
                mesh      = stored_mesh(self.readings, name)
                mesh.name = f"{name}Shape"
                self.assertIsNotNone(mesh.normals)
                save_fbx(self.path(f"{name}.fbx"), meshes=mesh)
                back = load_fbx(self.path(f"{name}.fbx")).meshes[0]
                self.assertIsNone(back.normals)
                self.assertIsNone(back.normal_indices)
                if self.info[name]["kind"] == "hard":
                    np.testing.assert_array_equal(groups(back), groups(hard_mesh(self.readings, name)))
                else:
                    self.assertIsNone(back.hard_edges)
                np.testing.assert_allclose(back.get_face_vertex_normals(), mesh.get_face_vertex_normals(), rtol=0, atol=TOLERANCE)
                save_fbx(self.path(f"{name}_again.fbx"), meshes=back)
                again = load_fbx(self.path(f"{name}_again.fbx")).meshes[0]
                self.assertIsNone(again.normals)
                self.assertIsNone(again.normal_indices)
                if back.hard_edges is None:
                    self.assertIsNone(again.hard_edges)
                else:
                    np.testing.assert_array_equal(again.hard_edges, back.hard_edges)
                np.testing.assert_array_equal(again.get_face_vertex_normals(), back.get_face_vertex_normals())

    def test_authored_normals_come_back_as_held(self):
        """normals that are not the derived ones, here each group given its
        neighbour's: stored, every corner naming the one it was saved with;
        saved and read again, bit for bit"""
        mesh         = stored_mesh(self.readings, self.HARD)
        mesh.name    = "authoredShape"
        mesh.normals = np.roll(np.asarray(mesh.normals), 1, axis=0)
        save_fbx(self.path("authored.fbx"), meshes=mesh)
        back = load_fbx(self.path("authored.fbx")).meshes[0]
        self.assertEqual(len(back.normals), self.info[self.HARD]["normals"])
        self.assertIsNone(back.hard_edges)
        np.testing.assert_array_equal(back.get_face_vertex_normals(), mesh.get_face_vertex_normals())
        save_fbx(self.path("authored_again.fbx"), meshes=back)
        again = load_fbx(self.path("authored_again.fbx")).meshes[0]
        for field in ("normals", "normal_indices"):
            np.testing.assert_array_equal(getattr(again, field), getattr(back, field), err_msg=field)
        self.assertIsNone(again.hard_edges)

    def test_the_class_routes_read_the_same(self):
        """MeshData.load_fbx and MeshList.load_fbx read as load_fbx does"""
        mesh      = stored_mesh(self.readings, self.HARD)
        mesh.name = "routeShape"
        mesh.save_fbx(self.path("route.fbx"))
        want = load_fbx(self.path("route.fbx")).meshes[0].hard_edges
        self.assertGreater(len(want), 0)
        for back in (MeshData.load_fbx(self.path("route.fbx")), MeshList.load_fbx(self.path("route.fbx"))[0]):
            self.assertIsNone(back.normals)
            self.assertIsNone(back.normal_indices)
            np.testing.assert_array_equal(back.hard_edges, want)


if __name__ == "__main__":
    unittest.main()
