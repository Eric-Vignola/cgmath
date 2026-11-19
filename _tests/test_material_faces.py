"""A material's faces follow its mesh's edits.

A mesh edit that renumbers faces (quadrangulate, triangulate, subdivide,
delete_faces, from_faces) is called on the mesh's materials as it is on its
UV sets and its skin: each material then covers, of the edited mesh, what it
covered of the mesh before.
"""

import inspect
import unittest

import numpy as np
from cgmath.geometry import MaterialData, MaterialList, MeshData


class TestMaterialFaces(unittest.TestCase):
    """a material's faces follow its mesh's edits, by the same calls"""

    @staticmethod
    def grid():
        """a flat 4 x 4 grid of quads, its left half one material, its
        right half another, its top row a third on a mesh of its own list"""
        side   = np.arange(5, dtype=float)
        points = np.array([[x, 0.0, z] for z in side for x in side])
        quads  = [[r * 5 + c, r * 5 + c + 1, (r + 1) * 5 + c + 1, (r + 1) * 5 + c] for r in range(4) for c in range(4)]
        mesh   = MeshData(points=points, indices=np.array(quads).ravel(), counts=np.full(16, 4))
        left   = np.array([f for f in range(16) if f % 4 < 2], dtype=np.int32)
        right  = np.array([f for f in range(16) if f % 4 >= 2], dtype=np.int32)
        held   = MaterialList(
            [
                MaterialData(name="left", mesh_uuid=mesh.uuid, faces=left),
                MaterialData(name="right", mesh_uuid=mesh.uuid, faces=right),
                MaterialData(name="elsewhere", faces=np.array([0, 1], dtype=np.int32)),
            ]
        )
        return mesh, held

    @staticmethod
    def sides(mesh, material) -> tuple:
        """the x of the middle of each face the material covers: its span,
        and how many faces"""
        corners = np.asarray(mesh.points)[np.asarray(mesh.indices)]
        starts  = np.concatenate([[0], np.cumsum(np.asarray(mesh.counts))])
        middles = np.array([corners[starts[f] : starts[f + 1], 0].mean() for f in np.asarray(material.faces).tolist()])
        return float(middles.min()), float(middles.max()), len(middles)

    def assertHalves(self, mesh, held, count: int) -> None:
        low, high, faces = self.sides(mesh, held[0])
        self.assertTrue(0.0 < low and high < 2.0 and faces == count, (low, high, faces))
        low, high, faces = self.sides(mesh, held[1])
        self.assertTrue(2.0 < low and high < 4.0 and faces == count, (low, high, faces))
        np.testing.assert_array_equal(np.asarray(held[2].faces), [0, 1])
        self.assertEqual(sorted(np.concatenate([held[0].faces, held[1].faces]).tolist()), list(range(len(mesh.counts))))

    def test_subdivide(self):
        mesh, held = self.grid()
        held.subdivide(mesh, 2)
        mesh.subdivide(2, keep_borders=True)
        self.assertHalves(mesh, held, 128)

    def test_triangulate_and_back(self):
        mesh, held = self.grid()
        held.triangulate(mesh)
        mesh.triangulate(mesh.get_triangulate_rules())
        self.assertHalves(mesh, held, 16)
        rules = mesh.get_quadrangulate_rules()
        kept  = held.keep_borders(mesh, rules)
        self.assertLessEqual(len(kept), len(rules))
        held.quadrangulate(mesh, kept)
        mesh.quadrangulate(kept)
        low, high, _ = self.sides(mesh, held[0])
        self.assertTrue(0.0 < low and high < 2.0)
        self.assertEqual(sorted(np.concatenate([held[0].faces, held[1].faces]).tolist()), list(range(len(mesh.counts))))

    def test_quadrangulate_across_a_border(self):
        mesh, held = self.grid()
        held.triangulate(mesh)
        mesh.triangulate(mesh.get_triangulate_rules())
        left  = np.asarray(held[0].faces)
        right = np.asarray(held[1].faces)
        rules = np.array([[int(left[0]), int(right[0]), 0, 0]])
        self.assertEqual(len(held.keep_borders(mesh, rules)), 0)
        with self.assertRaises(ValueError) as caught:
            held.quadrangulate(mesh, rules)
        self.assertIn("keep_borders()", str(caught.exception))

    def test_delete_and_from_faces(self):
        mesh, held = self.grid()
        faces = [0, 1, 2, 3, 4, 5, 6, 7]
        part  = mesh.from_faces(faces)
        parts = held.from_faces(mesh, faces, part)
        self.assertEqual([material.name for material in parts],      ["left", "right"])
        self.assertEqual([material.uuid for material in parts],      [held[0].uuid, held[1].uuid])
        self.assertEqual({material.mesh_uuid for material in parts}, {part.uuid})
        self.assertHalves(part, MaterialList(list(parts) + [held[2]]), 4)
        held.delete_faces(mesh, faces)
        mesh.delete_faces(faces)
        self.assertHalves(mesh, held, 4)
        held.delete_faces(mesh, np.asarray(held[1].faces))
        self.assertEqual([material.name for material in held], ["left", "elsewhere"])

    def test_whole(self):
        """a material covering its whole mesh still does, whatever the edit"""
        mesh, _ = self.grid()
        whole = MaterialData(name="all", mesh_uuid=mesh.uuid)
        whole.subdivide(mesh)
        whole.triangulate(mesh)
        whole.quadrangulate(np.array([[0, 1, 0, 0]]))
        whole.delete_faces([0])
        self.assertIsNone(whole.faces)
        self.assertIsNone(whole.from_faces([0, 1]).faces)

    def test_exclude(self):
        """exclude turns each call round, as it does the mesh's"""
        mesh, held = self.grid()
        faces = [0, 1, 2, 3, 4, 5, 6, 7]
        part  = mesh.from_faces(faces, exclude=True)
        parts = held.from_faces(mesh, faces, part, exclude=True)
        self.assertHalves(part, MaterialList(list(parts) + [held[2]]), 4)
        held.delete_faces(mesh, faces, exclude=True)
        mesh.delete_faces(faces, exclude=True)
        self.assertHalves(mesh, held, 4)
        self.assertIsNone(held[0].from_faces(np.asarray(held[1].faces)))

    def test_from_vertices(self):
        """the faces a set of vertices holds, the ones it touches, the
        others, and subset rules: the materials of each part"""
        mesh, held = self.grid()
        row = np.arange(10)  # the first two rows of points: the first row of faces
        for options, count in (({}, 4), ({"contained": False}, 8), ({"exclude": True}, 12)):
            with self.subTest(options=options):
                part  = mesh.from_vertices(row, **options)
                parts = held.from_vertices(mesh, row, part, **options)
                self.assertEqual(len(part.counts), count)
                self.assertEqual({material.mesh_uuid for material in parts}, {part.uuid})
                self.assertHalves(part, MaterialList(list(parts) + [held[2]]), count // 2)
        rules = mesh.get_subset_rules(row)
        part  = mesh.from_vertices(rules)
        parts = held.from_vertices(mesh, rules, part)
        self.assertEqual(sorted(np.concatenate([material.faces for material in parts]).tolist()), list(range(len(part.counts))))
        with self.assertRaises(ValueError):
            held[0].from_vertices(mesh, rules, exclude=True)

    def test_union(self):
        """two meshes merged: the second's materials move onto the first,
        after its faces; one material on both becomes one record; a
        material covering a whole mesh lists the faces it had"""
        mesh, held      = self.grid()
        other, more     = self.grid()
        other.points  = np.asarray(other.points) + [0.0, 0.0, 10.0]
        whole         = MaterialData(name="whole", mesh_uuid=other.uuid)
        more[0].uuid  = held[0].uuid  # the left material is one material on both
        records       = MaterialList([held[0], held[1], held[2], more[0], whole])
        whole.faces   = None
        more[0].faces = np.asarray(more[0].faces)
        records.union(mesh, other)
        mesh.union(other)
        self.assertEqual([material.name for material in records], ["left", "right", "elsewhere", "whole"])
        self.assertEqual({material.mesh_uuid for material in records if material.name != "elsewhere"}, {mesh.uuid})
        np.testing.assert_array_equal(np.asarray(records[3].faces), np.arange(16, 32))
        low, high, faces = self.sides(mesh, records[0])
        self.assertTrue(0.0 < low and high < 2.0 and faces == 16, (low, high, faces))
        np.testing.assert_array_equal(np.asarray(records[2].faces), [0, 1])
        alone = MaterialData(name="alone", mesh_uuid=mesh.uuid)
        alone.union(mesh)
        self.assertIsNone(alone.faces)


class TestAligned(unittest.TestCase):
    """every MeshData edit that renumbers faces has a MaterialData and a
    MaterialList call of its name: a mesh edit added without one fails here"""

    # the edits that renumber a mesh's faces: each one a material follows
    FACE_EDITS = ("delete_faces", "from_faces", "from_vertices", "quadrangulate", "subdivide", "triangulate", "union")

    # how the edits that take arguments are called: (the mesh to start from, the call)
    CALLS = {
        "delete_faces":  ("quads", lambda mesh, other: mesh.delete_faces([0, 1])),
        "from_faces":    ("quads", lambda mesh, other: mesh.from_faces([0, 1])),
        "from_vertices": ("quads", lambda mesh, other: mesh.from_vertices(np.arange(10))),
        "quadrangulate": ("triangles", lambda mesh, other: mesh.quadrangulate()),
        "triangulate":   ("quads", lambda mesh, other: mesh.triangulate(mesh.get_triangulate_rules())),
        "union":         ("quads", lambda mesh, other: mesh.union(other)),
    }

    # the methods that take arguments and renumber no face: by what their
    # name says they do, and by name
    QUERIES = ("get_", "grow_", "shrink_", "save", "load", "to_", "contains_")
    KEEPS   = ("apply_morph_target", "raycast", "recompute_normals", "sample", "sum_vertex_normals", "transform", "world_matrix")

    @staticmethod
    def faces_of(mesh) -> list:
        """each face as the places of its corners, a face a tuple"""
        points = np.round(np.asarray(mesh.points, dtype=float), 9)
        starts = np.concatenate([[0], np.cumsum(np.asarray(mesh.counts))])
        ids    = np.asarray(mesh.indices)
        return [tuple(sorted(map(tuple, points[ids[starts[f] : starts[f + 1]]].tolist()))) for f in range(len(mesh.counts))]

    @staticmethod
    def corners_of(mesh) -> list:
        starts = np.concatenate([[0], np.cumsum(np.asarray(mesh.counts))])
        ids    = np.asarray(mesh.indices)
        return [tuple(sorted(ids[starts[f] : starts[f + 1]].tolist())) for f in range(len(mesh.counts))]

    def renumbers(self, name: str):
        """whether calling ``name`` leaves the mesh, or the one it returns,
        with other faces than it had, or its faces in another order; None
        for a method that takes arguments and has no entry in CALLS"""
        mesh, _ = TestMaterialFaces.grid()
        other = mesh.copy()
        start, call = self.CALLS.get(name, ("quads", None))
        if start == "triangles":
            mesh.triangulate(mesh.get_triangulate_rules())
        before = (self.faces_of(mesh), self.corners_of(mesh))
        if call is None:
            method = getattr(mesh, name)
            needed = [
                p for p in inspect.signature(method).parameters.values()
                if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if needed:
                return None
            call = lambda mesh, other: getattr(mesh, name)()
        try:
            out = call(mesh, other)
        except Exception:
            return False  # an edit that does not run on a plain grid renumbers nothing of it
        held = out if isinstance(out, MeshData) else mesh
        if len(held.counts) != len(before[0]):
            return True
        # a face holding other points and other vertices than it did: the faces moved
        return self.faces_of(held) != before[0] and self.corners_of(held) != before[1]

    def test_every_face_edit_has_its_material_call(self):
        unknown = []
        found   = []
        for name in sorted(dir(MeshData)):
            value = inspect.getattr_static(MeshData, name)
            if name.startswith("_") or isinstance(value, (property, classmethod, staticmethod)) or not callable(value):
                continue
            moved = self.renumbers(name)
            if moved is None:
                if not (name.startswith(self.QUERIES) or "_to_" in name or name in self.KEEPS):
                    unknown.append(name)
            elif moved:
                found.append(name)
        self.assertEqual(
            unknown, [],
            "MeshData methods taking arguments that this test can not place: one that renumbers faces goes in "
            "FACE_EDITS and CALLS, with a MaterialData and MaterialList method of its name; any other in KEEPS",
        )
        self.assertEqual(sorted(found), sorted(self.FACE_EDITS), "the MeshData methods that renumber faces")
        for name in self.FACE_EDITS:
            for cls in (MaterialData, MaterialList):
                self.assertTrue(callable(getattr(cls, name, None)), f"{cls.__name__}.{name} is missing")


if __name__ == "__main__":
    unittest.main()
