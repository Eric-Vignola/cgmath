"""A skin's rows follow its mesh's edits.

A mesh edit that renumbers points (delete_faces, detach_faces, from_faces,
from_vertices, subdivide, union) is called on the mesh's skin with the mesh
first: each row then weighs the point it weighed before, wherever the edit
put that point. The skins here weigh each point by its own place, so a row
that follows its point still says where the point is.
"""

import inspect
import unittest

import numpy as np
from cgmath._tests.test_material_faces import TestAligned as FaceEdits
from cgmath._tests.test_material_faces import TestMaterialFaces
from cgmath.geometry import MeshData, MorphData, MorphList, SkinData, SkinList
from cgmath.hierarchy import generate_uuid


def skinned(influences=("x", "z", "rest")) -> tuple:
    """the 4 x 4 grid of quads and a skin whose weights are its points'
    places: x / 8 under "x", z / 8 under "z", what is left under "rest\""""
    mesh, _ = TestMaterialFaces.grid()
    points        = np.asarray(mesh.points, dtype=float)
    held          = {"x": points[:, 0] / 8.0, "z": points[:, 2] / 8.0}
    held["rest"]  = 1.0 - held["x"] - held["z"]
    held["extra"] = np.zeros(len(points))
    weights       = np.stack([held[name] for name in influences], axis=1)
    return mesh, SkinData(weights=weights, influences=list(influences), mesh_uuid=mesh.uuid, name="gridSkin")


class TestSkinPoints(unittest.TestCase):
    """each call leaves a row on the point it weighed"""

    def assertFollows(self, mesh, skin) -> None:
        points  = np.asarray(mesh.points, dtype=float)
        weights = np.asarray(skin.weights, dtype=float)
        names   = list(skin.influences)
        self.assertEqual(len(weights), len(points))
        np.testing.assert_allclose(weights[:, names.index("x")], points[:, 0] / 8.0, atol=1e-9)
        np.testing.assert_allclose(weights[:, names.index("z")], points[:, 2] / 8.0, atol=1e-9)

    def test_delete_faces(self):
        for exclude, points in ((False, 20), (True, 10)):
            with self.subTest(exclude=exclude):
                mesh, skin = skinned()
                skin.delete_faces(mesh, [0, 1, 2, 3], exclude=exclude)
                mesh.delete_faces([0, 1, 2, 3], exclude=exclude)
                self.assertEqual(len(mesh.points), points)
                self.assertFollows(mesh, skin)

    def test_from_faces(self):
        mesh, skin = skinned()
        for exclude in (False, True):
            with self.subTest(exclude=exclude):
                part      = mesh.from_faces([5, 6, 15], exclude=exclude)
                part.uuid = generate_uuid()
                made      = skin.from_faces(mesh, [5, 6, 15], exclude=exclude, mesh_uuid=part.uuid)
                self.assertFollows(part, made)
                self.assertEqual((made.mesh_uuid, made.influences, made.name), (part.uuid, skin.influences, skin.name))
        # the skin it was made from is as it was
        self.assertFollows(mesh, skin)

    def test_from_vertices(self):
        mesh, skin = skinned()
        row = np.arange(10)
        for options in ({}, {"contained": False}, {"exclude": True}):
            with self.subTest(options=options):
                self.assertFollows(mesh.from_vertices(row, **options), skin.from_vertices(mesh, row, **options))
        rules = mesh.get_subset_rules(row)
        self.assertFollows(mesh.from_vertices(rules), skin.from_vertices(mesh, rules))

    def test_detach_faces(self):
        for indices, points in ((None, 64), ([0, 5], 32)):
            with self.subTest(indices=indices):
                mesh, skin = skinned()
                skin.detach_faces(mesh, indices)
                mesh.detach_faces(indices)
                self.assertEqual(len(mesh.points), points)
                self.assertFollows(mesh, skin)

    def test_subdivide(self):
        for options in ({}, {"keep_borders": True}):
            with self.subTest(options=options):
                mesh, skin = skinned()
                skin.subdivide(mesh, 2, **options)
                mesh.subdivide(2, **options)
                self.assertFollows(mesh, skin)

    def test_union(self):
        """the rows of the skins merged in come after, each skin's weights
        under its own influences' names, a new influence after the others"""
        mesh, skin   = skinned()
        other, more  = skinned(("z", "extra", "x", "rest"))
        other.points = np.asarray(other.points) + [0.0, 3.0, 0.0]
        third, last  = skinned(("rest", "x", "z"))
        skin.union(more, last)
        mesh.union(other, third)
        self.assertEqual(skin.influences, ["x", "z", "rest", "extra"])
        self.assertEqual(len(skin.weights), 75)
        self.assertFollows(mesh, skin)
        np.testing.assert_allclose(np.asarray(skin.weights).sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(np.asarray(skin.weights)[:, 3], 0.0)

    def test_another_mesh(self):
        """a skin given a mesh it is not of: ValueError, the skin left as it was"""
        mesh, skin = skinned()
        mesh.subdivide(1)
        with self.assertRaises(ValueError) as caught:
            skin.delete_faces(mesh, [0])
        self.assertIn("weighs 25 points, the mesh holds 81", str(caught.exception))
        self.assertEqual(len(skin.weights), 25)


class TestSkinList(unittest.TestCase):
    """the list's calls take the mesh first and reach its skin alone"""

    def held(self) -> tuple:
        mesh, skin   = skinned()
        other, more  = skinned(("z", "x", "rest"))
        other.uuid     = generate_uuid()
        more.mesh_uuid = other.uuid
        return mesh, other, SkinList([skin, more])

    def test_calls(self):
        mesh, other, skins = self.held()
        self.assertEqual(skins.of(other), [skins[1]])
        # each before the mesh's own: the skin reads the mesh as it is
        skins.subdivide(mesh, 1)
        mesh.subdivide(1)
        skins.delete_faces(mesh, [0])
        mesh.delete_faces([0])
        skins.detach_faces(mesh, [3])
        mesh.detach_faces([3])
        self.assertEqual(len(skins[0].weights), len(mesh.points))
        self.assertEqual(len(skins[1].weights), 25)

    def test_parts(self):
        mesh, other, skins = self.held()
        part      = mesh.from_faces([0, 1])
        part.uuid = generate_uuid()
        made      = skins.from_faces(mesh, [0, 1], part)
        self.assertIsInstance(made, SkinList)
        self.assertEqual([(skin.mesh_uuid, len(skin.weights)) for skin in made], [(part.uuid, len(part.points))])
        made = skins.from_vertices(other, np.arange(10), part, contained=False)
        self.assertEqual([(skin.mesh_uuid, len(skin.weights)) for skin in made], [(part.uuid, 15)])
        bare      = mesh.copy()
        bare.uuid = generate_uuid()
        self.assertEqual(len(skins.from_faces(bare, [0])), 0)

    def test_union(self):
        mesh, other, skins = self.held()
        first = skins[0]
        skins.union(mesh, other)
        mesh.union(other)
        self.assertEqual([id(skin) for skin in skins], [id(first)])
        self.assertEqual((len(first.weights), first.mesh_uuid), (len(mesh.points), mesh.uuid))

    def test_union_refused(self):
        """a skinned mesh merged with one no skin names; unskinned meshes are left alone"""
        mesh, other, skins = self.held()
        bare      = mesh.copy()
        bare.uuid = generate_uuid()
        with self.assertRaises(ValueError) as caught:
            skins.union(mesh, bare)
        self.assertIn("has no skin in this list", str(caught.exception))
        self.assertEqual(len(skins), 2)
        SkinList().union(mesh, bare)


def morphed() -> tuple:
    """the 4 x 4 grid of quads and a target moving its right half (x >= 2),
    each point by its own place: (x, z, x + z)"""
    mesh, _ = TestMaterialFaces.grid()
    points  = np.asarray(mesh.points, dtype=float)
    moved   = np.flatnonzero(points[:, 0] >= 2.0)
    offsets = np.stack([points[moved, 0], points[moved, 2], points[moved, 0] + points[moved, 2]], axis=1)
    return mesh, MorphData("smile", offsets, moved.astype(np.int32), mesh_uuid=mesh.uuid, weight=0.5, deformer="face")


class TestMorphPoints(unittest.TestCase):
    """each call leaves an offset on the point it moved"""

    def assertFollows(self, mesh, morph) -> None:
        points  = np.asarray(mesh.points, dtype=float)
        indices = np.asarray(morph.indices)
        # the right half is moved, each point by its own place, and nothing else
        np.testing.assert_array_equal(np.sort(indices), np.flatnonzero(points[:, 0] >= 2.0))
        held = points[indices]
        np.testing.assert_allclose(np.asarray(morph.offsets), np.stack([held[:, 0], held[:, 2], held[:, 0] + held[:, 2]], axis=1), atol=1e-9)

    def test_delete_and_detach_faces(self):
        for call, arguments in (("delete_faces", ([0, 1, 2, 3],)), ("delete_faces", ([0, 1, 2, 3], True)), ("detach_faces", ()), ("detach_faces", ([3, 6],))):
            with self.subTest(call=call, arguments=arguments):
                mesh, morph = morphed()
                getattr(morph, call)(mesh, *arguments)
                getattr(mesh, call)(*arguments)
                self.assertFollows(mesh, morph)

    def test_parts(self):
        mesh, morph = morphed()
        for exclude in (False, True):
            part = mesh.from_faces([2, 3, 6], exclude=exclude)
            made = morph.from_faces(mesh, [2, 3, 6], exclude=exclude, mesh_uuid="12345678-1234-5678-1234-567812345678")
            self.assertFollows(part, made)
            self.assertEqual((made.name, made.weight, made.at, made.deformer), ("smile", 0.5, 1.0, "face"))
        row = np.arange(10)
        self.assertFollows(mesh.from_vertices(row, contained=False), morph.from_vertices(mesh, row, contained=False))
        left = morph.from_faces(mesh, [0, 4])  # a part the target moves no point of
        self.assertEqual(len(left.offsets), 0)
        self.assertFollows(mesh, morph)

    def test_subdivide(self):
        """the subdivided mesh moved by the target is the moved mesh subdivided"""
        for options in ({}, {"keep_borders": True}):
            with self.subTest(options=options):
                mesh, morph = morphed()
                moved  = mesh.copy()
                points = np.array(moved.points)
                points[np.asarray(morph.indices)] += np.asarray(morph.offsets)
                moved.points = points
                morph.subdivide(mesh, 2, **options)
                mesh.subdivide(2, **options)
                moved.subdivide(2, **options)
                points = np.array(mesh.points)
                points[np.asarray(morph.indices)] += np.asarray(morph.offsets)
                np.testing.assert_allclose(points, np.asarray(moved.points), atol=1e-9)

    def test_union(self):
        """the targets of the meshes merged in move onto the first, after
        its points; one name on two meshes becomes one target"""
        mesh, morph  = morphed()
        other, more  = morphed()
        other.uuid = generate_uuid()
        more.mesh_uuid, more.deformer = other.uuid, "other"
        frown = MorphData("frown", np.ones((2, 3)), np.array([0, 1], dtype=np.int32), mesh_uuid=other.uuid)
        apart = MorphData("apart", np.ones((1, 3)), np.array([0], dtype=np.int32))
        held  = MorphList([morph, more, frown, apart])
        held.union(mesh, other)
        mesh.union(other)
        self.assertEqual([item.name for item in held],          ["smile", "frown", "apart"])
        self.assertEqual([item.deformer for item in held],      ["face", "face", None])
        self.assertEqual({item.mesh_uuid for item in held[:2]}, {mesh.uuid})
        self.assertFollows(mesh, held[0])
        np.testing.assert_array_equal(np.asarray(held[1].indices), [25, 26])
        np.testing.assert_array_equal(np.asarray(held[2].indices), [0])

    def test_list(self):
        mesh, morph = morphed()
        other      = mesh.copy()
        other.uuid = generate_uuid()
        held       = MorphList([morph, MorphData("far", np.ones((1, 3)), np.array([0], dtype=np.int32), mesh_uuid=other.uuid)])
        self.assertEqual(held.of(mesh), [morph])
        held.subdivide(mesh, 1)
        mesh.subdivide(1)
        held.delete_faces(mesh, [0])
        mesh.delete_faces([0])
        held.detach_faces(mesh, [1])
        mesh.detach_faces([1])
        self.assertEqual(len(held[1].offsets), 1)
        part  = mesh.from_faces([5, 6])
        parts = held.from_faces(mesh, [5, 6], part)
        self.assertEqual([(item.name, item.mesh_uuid) for item in parts], [("smile", part.uuid)])
        self.assertEqual(len(held.from_vertices(other, [0], part)), 1)

    def test_another_mesh(self):
        mesh, morph = morphed()
        small = mesh.from_faces([0])
        with self.assertRaises(ValueError) as caught:
            morph.delete_faces(small, [0])
        self.assertIn("moves points the mesh of 4 points does not hold", str(caught.exception))


class TestAligned(unittest.TestCase):
    """every MeshData edit that renumbers points has a call of its name on
    the records that hold a row a point, SkinData and MorphData, and on
    their lists: a mesh edit added without one fails here"""

    # the edits that renumber a mesh's points: each one a skin follows
    POINT_EDITS = ("delete_faces", "detach_faces", "from_faces", "from_vertices", "subdivide", "union")

    def renumbers(self, name: str):
        """whether calling ``name`` leaves the mesh, or the one it returns,
        with another count of points, or its points in another order; None
        for a method that takes arguments and has no entry in the face
        edits' CALLS"""
        mesh, _ = TestMaterialFaces.grid()
        other = mesh.copy()
        start, call = FaceEdits.CALLS.get(name, ("quads", None))
        if start == "triangles":
            mesh.triangulate(mesh.get_triangulate_rules())
        before = np.round(np.asarray(mesh.points, dtype=float), 9)
        if call is None:
            needed = [
                p for p in inspect.signature(getattr(mesh, name)).parameters.values()
                if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if needed:
                return None
            call = lambda mesh, other: getattr(mesh, name)()
        try:
            out = call(mesh, other)
        except Exception:
            return False  # an edit that does not run on a plain grid renumbers nothing of it
        held  = out if isinstance(out, MeshData) else mesh
        after = np.round(np.asarray(held.points, dtype=float), 9)
        if len(after) != len(before):
            return True
        # the same points in another order: an edit that moves points keeps each in its place
        return not np.array_equal(after, before) and sorted(map(tuple, after.tolist())) == sorted(map(tuple, before.tolist()))

    def test_every_point_edit_has_its_skin_call(self):
        found = []
        for name in sorted(dir(MeshData)):
            value = inspect.getattr_static(MeshData, name)
            if name.startswith("_") or isinstance(value, (property, classmethod, staticmethod)) or not callable(value):
                continue
            # a method this can not call is placed by the face edits' test
            if self.renumbers(name):
                found.append(name)
        self.assertEqual(sorted(found), sorted(self.POINT_EDITS), "the MeshData methods that renumber points")
        for name in self.POINT_EDITS:
            for cls in (SkinData, SkinList, MorphData, MorphList):
                self.assertTrue(callable(getattr(cls, name, None)), f"{cls.__name__}.{name} is missing")


if __name__ == "__main__":
    unittest.main()
