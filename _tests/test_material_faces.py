"""A material's faces follow its mesh's edits.

A mesh edit that renumbers faces (quadrangulate, triangulate, subdivide,
delete_faces, from_faces) is called on the mesh's materials as it is on its
UV sets and its skin: each material then covers, of the edited mesh, what it
covered of the mesh before.
"""

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


if __name__ == "__main__":
    unittest.main()
