"""UVList, MaterialList and MorphList of one mesh: mesh_uuid, held on every way in.

A list's ``mesh_uuid`` is None for records of any mesh, as a file holds
them, or the one mesh every record names, as a Renderable holds them: set,
a record of another mesh is refused on append, insert, extend and item
assignment, and setting it on a list holding another mesh's records is
refused. It is saved beside the records and left out of ``==``.
"""

import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry import MaterialData, MaterialList, MeshData, MorphData, MorphList, UVData, UVList


def tri(name="tri") -> MeshData:
    return MeshData(points=np.zeros((3, 3)), indices=np.arange(3), counts=np.array([3]), name=name)


def uv(mesh, name="map1") -> UVData:
    return UVData(name=name, points=np.zeros((3, 2)), indices=np.arange(3), counts=np.array([3]), mesh_uuid=mesh.uuid)


def material(mesh, name="paint") -> MaterialData:
    return MaterialData(name=name, mesh_uuid=mesh.uuid)


def morph(mesh, name="smile") -> MorphData:
    return MorphData(name, np.zeros((1, 3)), [0], mesh_uuid=mesh.uuid)


# (the list class, a record of a mesh, the record class's name, the record's name)
CASES = ((UVList, uv, "UVData", "map1"), (MaterialList, material, "MaterialData", "paint"), (MorphList, morph, "MorphData", "smile"))


class TestOneMeshLists(unittest.TestCase):
    def setUp(self):
        self.a = tri("a")
        self.b = tri("b")

    def test_none_holds_any_mesh_s_records(self):
        for cls, make, _, _ in CASES:
            with self.subTest(cls.__name__):
                held = cls([make(self.a), make(self.b)])
                self.assertIsNone(held.mesh_uuid)
                held.append(make(self.a))
                held.insert(0, make(self.b))
                held[1] = make(self.b)
                self.assertEqual(len(held), 4)
                self.assertIsNone(cls().mesh_uuid)

    def test_of_one_mesh_refuses_another_s(self):
        a, b = self.a, self.b
        for cls, make, record, name in CASES:
            with self.subTest(cls.__name__):
                held = cls([make(a)], mesh_uuid=a.uuid)
                self.assertEqual(held.mesh_uuid, a.uuid)
                refused = f"{cls.__name__} of mesh '{a.uuid}' takes records naming it; {record} '{name}' names '{b.uuid}'"
                with self.assertRaisesRegex(ValueError, refused):
                    held.append(make(b))
                with self.assertRaisesRegex(ValueError, refused):
                    held.insert(0, make(b))
                with self.assertRaisesRegex(ValueError, refused):
                    held[0] = make(b)
                with self.assertRaisesRegex(ValueError, refused):
                    held[0:1] = [make(b)]
                with self.assertRaisesRegex(ValueError, refused):
                    held.extend([make(a), make(b)])
                with self.assertRaisesRegex(ValueError, refused):
                    cls([make(b)], mesh_uuid=a.uuid)
                self.assertTrue(all(item.mesh_uuid == a.uuid for item in held))
                held.append(make(a))
                held[0:1] = [make(a), make(a)]
                self.assertEqual(len(held), 4)

    def test_setting_the_mesh_checks_the_records(self):
        a, b = self.a, self.b
        for cls, make, record, name in CASES:
            with self.subTest(cls.__name__):
                mixed = cls([make(a), make(b)])
                with self.assertRaisesRegex(
                    ValueError, f"{cls.__name__}.mesh_uuid '{a.uuid}': the list holds 1 record\\(s\\) naming another mesh \\({record} '{name}'\\)"
                ):
                    mixed.mesh_uuid = a.uuid
                self.assertIsNone(mixed.mesh_uuid)
                held           = cls([make(a), make(a)])
                held.mesh_uuid = a.uuid
                self.assertEqual(held.mesh_uuid, a.uuid)
                held.mesh_uuid = None
                held.append(make(b))
                with self.assertRaisesRegex(ValueError, f"{cls.__name__}.mesh_uuid takes a uuid or None, not 'nope'"):
                    held.mesh_uuid = "nope"

    def test_saved_loaded_copied_and_compared(self):
        a = self.a
        for cls, make, _, _ in CASES:
            with self.subTest(cls.__name__):
                held = cls([make(a)], mesh_uuid=a.uuid)
                with tempfile.TemporaryDirectory() as folder:
                    for ext in ("json", "npz", "pkl"):
                        back = cls.load(held.save(os.path.join(folder, f"l.{ext}")))
                        self.assertEqual(back.mesh_uuid, a.uuid, ext)
                        self.assertEqual(back, held, ext)
                self.assertEqual(held.copy().mesh_uuid, a.uuid)
                self.assertEqual(cls([make(a)]), held)  # the mesh is left out of ==
                tree = held.to_dict()
                self.assertEqual(tree["mesh_uuid"], a.uuid)
                tree.pop("mesh_uuid")  # a file saved before lists had one
                self.assertIsNone(cls.from_dict(tree).mesh_uuid)


if __name__ == "__main__":
    unittest.main()
