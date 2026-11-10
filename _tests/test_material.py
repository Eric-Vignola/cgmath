"""MaterialData: Maya's lambert and phong attributes, on the faces of a mesh."""

import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry import MaterialData, MaterialList, MeshData
from cgmath.geometry.material import MATERIAL_KINDS, WRAP_MODES
from cgmath.hierarchy.hierarchy import _is_uuid


def quad_grid() -> MeshData:
    """a 2 by 2 plane of four quads"""
    points = [(x, 0.0, z) for z in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
    return MeshData(
        indices=[0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], counts=[4, 4, 4, 4], points=points, name="plane",
    )


def gradient(dtype=np.uint8) -> np.ndarray:
    pixels         = np.zeros((4, 8, 3), dtype=dtype)
    pixels[..., 0] = np.linspace(0, 255, 8)[None, :]
    pixels[..., 1] = np.linspace(0, 255, 4)[:, None]
    pixels[..., 2] = 128
    return pixels


class TestMaterialData(unittest.TestCase):
    def test_defaults_are_a_fresh_maya_shader(self):
        """Maya 2025's lambert and phong (measured): colour 0.5 grey, no
        transparency, no ambient colour, no incandescence, diffuse 0.8; a
        phong's specular colour 0.5 grey, cosine power 20, reflectivity 0.5;
        no texture, the UV set the mesh's first, both wraps repeat"""
        material = MaterialData()
        self.assertEqual((material.name, material.mesh_uuid, material.faces, material.kind), (None, None, None, "lambert"))
        np.testing.assert_array_equal(material.color,         [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(material.transparency,  [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(material.ambient_color, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(material.incandescence, [0.0, 0.0, 0.0])
        self.assertEqual(material.diffuse, 0.8)
        np.testing.assert_array_equal(material.specular_color, [0.5, 0.5, 0.5])
        self.assertEqual((material.cosine_power, material.reflectivity),             (20.0, 0.5))
        self.assertEqual((material.texture, material.texture_file, material.uv_set), (None, None, None))
        self.assertEqual((material.wrap_u, material.wrap_v, material.textured),      ("repeat", "repeat", False))
        # the colour defaults are each object's own
        other          = MaterialData()
        other.color[:] = 1.0
        np.testing.assert_array_equal(material.color, [0.5, 0.5, 0.5])

    def test_kind_and_wraps_are_mayas(self):
        material = MaterialData()
        for kind in MATERIAL_KINDS:
            material.kind = kind
        with self.assertRaisesRegex(ValueError, r"kind is one of \['lambert', 'phong'\], not 'blinn'"):
            material.kind = "blinn"
        with self.assertRaises(ValueError):
            MaterialData(kind="standardSurface")
        for mode in WRAP_MODES:
            material.wrap_u = mode
            material.wrap_v = mode
        with self.assertRaisesRegex(ValueError, r"wrap_v is one of \['repeat', 'clamp'\], not 'mirror'"):
            material.wrap_v = "mirror"

    def test_texture_keeps_its_dtype_and_is_h_w_c(self):
        for dtype in (np.uint8, np.uint16, np.float32):
            with self.subTest(dtype=dtype.__name__):
                material = MaterialData(texture=gradient(dtype))
                self.assertEqual((material.texture.dtype, material.texture.shape, material.textured), (dtype, (4, 8, 3), True))
        # a float64 map is kept as float32, the nearest declared dtype
        material = MaterialData(texture=np.ones((2, 2, 4), dtype=np.float64))
        self.assertEqual(material.texture.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, r"texture must have shape \(H, W, C\), got \(4, 8\)"):
            MaterialData(texture=np.zeros((4, 8), dtype=np.uint8))
        material.texture = None
        self.assertFalse(material.textured)

    def test_faces_and_covers(self):
        mesh     = quad_grid()
        material = MaterialData(faces=[0, 1], mesh_uuid=mesh.uuid)
        self.assertEqual(material.faces.dtype, np.int32)
        np.testing.assert_array_equal(material.covers(mesh), [True, True, False, False])
        np.testing.assert_array_equal(MaterialData().covers(mesh), [True] * 4)
        np.testing.assert_array_equal(MaterialData(faces=[]).covers(mesh), [False] * 4)
        with self.assertRaisesRegex(ValueError, "faces name faces the mesh of 4 faces does not hold"):
            MaterialData(faces=[3, 4]).covers(mesh)

    def test_equality_ignores_name_and_mesh(self):
        a = MaterialData(name="a", mesh_uuid=quad_grid().uuid, kind="phong", color=[1.0, 0.0, 0.0])
        b = MaterialData(name="b", kind="phong", color=[1.0, 0.0, 0.0])
        self.assertEqual(a, b)
        self.assertNotEqual(a, MaterialData(kind="phong", color=[1.0, 0.0, 0.5]))
        self.assertNotEqual(a, MaterialData(kind="lambert", color=[1.0, 0.0, 0.0]))
        self.assertNotEqual(MaterialData(texture=gradient()), MaterialData())
        self.assertNotEqual(MaterialData(faces=[0]), MaterialData())
        self.assertEqual(MaterialData(texture=gradient()), MaterialData(texture=gradient()))

    def test_saved_and_loaded(self):
        mesh = quad_grid()
        material = MaterialData(
            name="skin", mesh_uuid=mesh.uuid, faces=[0, 3], kind="phong", color=[0.9, 0.2, 0.1], transparency=[0.1, 0.2, 0.3],
            ambient_color=[0.05, 0.1, 0.15], incandescence=[0.2, 0.0, 0.1], diffuse=0.6, specular_color=[0.3, 0.4, 0.5],
            cosine_power=35.0, reflectivity=0.25, texture=gradient(np.uint16), texture_file="grad.png", uv_set="map1",
            wrap_u="clamp", wrap_v="repeat",
        )
        with tempfile.TemporaryDirectory() as folder:
            for ext in ("json", "npz", "pkl"):
                with self.subTest(ext):
                    back = MaterialData.load(material.save(os.path.join(folder, f"m.{ext}")))
                    self.assertEqual(back, material)
                    self.assertEqual(back.uuid, material.uuid)
                    self.assertEqual((back.name, back.mesh_uuid, back.kind, back.texture_file, back.uv_set), ("skin", mesh.uuid, "phong", "grad.png", "map1"))
                    self.assertEqual((back.wrap_u, back.wrap_v), ("clamp", "repeat"))
                    np.testing.assert_array_equal(back.faces, [0, 3])
                    self.assertEqual((back.texture.dtype, back.texture.shape), (np.uint16, (4, 8, 3)))
                    np.testing.assert_array_equal(back.texture, material.texture)
            materials = MaterialList([material, MaterialData()])
            back      = MaterialList.load(materials.save(os.path.join(folder, "materials.npz")))
            self.assertEqual(back, materials)
            self.assertEqual([m.name for m in back], ["skin", None])

    def test_uuid_is_made_kept_by_a_copy_and_left_out_of_equality(self):
        a = MaterialData(name="a")
        b = MaterialData(name="b")
        self.assertTrue(_is_uuid(a.uuid))
        self.assertNotEqual(a.uuid, b.uuid)
        self.assertEqual(a.copy().uuid, a.uuid)
        self.assertEqual(MaterialData(uuid=a.uuid), a)
        self.assertEqual(MaterialData(uuid="0F8FAD5B-D9CB-469F-A165-70867728950E").uuid, "0F8FAD5B-D9CB-469F-A165-70867728950E")
        with self.assertRaisesRegex(ValueError, "MaterialData.uuid takes a uuid or None, not 'nope'"):
            MaterialData(uuid="nope")
        with self.assertRaisesRegex(ValueError, "takes a uuid or None"):
            a.uuid = "x"
        self.assertEqual(a.uuid, a.copy().uuid)
        # a file saved before materials had one gives it one
        tree = MaterialData(name="old").to_dict()
        self.assertTrue(_is_uuid(tree.pop("uuid")))
        self.assertTrue(_is_uuid(MaterialData.from_dict(tree).uuid))


if __name__ == "__main__":
    unittest.main()
