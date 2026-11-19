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
from cgmath.geometry import MeshData, SkinData, SkinList
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


class TestAligned(unittest.TestCase):
    """every MeshData edit that renumbers points has a SkinData and a
    SkinList call of its name: a mesh edit added without one fails here"""

    # the edits that renumber a mesh's points: each one a skin follows
    POINT_EDITS = ("delete_faces", "detach_faces", "from_faces", "from_vertices", "subdivide", "union")

    def renumbers(self, name: str) -> bool | None:
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
            for cls in (SkinData, SkinList):
                self.assertTrue(callable(getattr(cls, name, None)), f"{cls.__name__}.{name} is missing")


if __name__ == "__main__":
    unittest.main()
