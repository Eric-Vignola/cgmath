"""pkl / npz / json round trips of the render Scene, its settings and its lists."""

import os
import tempfile
import unittest

import numpy as np
from cgmath._tests._examples_render import build_scene, checker_texture, cube_mesh, cube_uv
from cgmath._tests.test_roundtrip import assert_same
from cgmath.geometry import CameraData, LightData, MaterialData
from cgmath.hierarchy import TransformData
from cgmath.render import Scene

try:
    import cv2  # noqa: F401

    HAVE_RASTERIZER = True
except ImportError:
    try:
        import skimage  # noqa: F401

        HAVE_RASTERIZER = True
    except ImportError:
        HAVE_RASTERIZER = False

FORMATS = (".pkl", ".npz", ".json")


def _round_trip(data, suffix, cls=None):
    """``data`` saved as ``suffix`` and loaded back by ``cls``, the class the
    file holds, ``type(data)`` unless given"""
    handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    try:
        data.save(path)
        return (cls or type(data)).load(path)
    finally:
        os.remove(path)


class RoundTripCase(unittest.TestCase):
    def assert_same(self, a, b, where="data"):
        """``b`` is exactly ``a``: its type, and every field's value, dtype
        and shape, as test_roundtrip compares every class"""
        assert_same(self, a, b, where)


class TestSceneRoundTrip(RoundTripCase):
    def test_every_format_gives_back_the_scene(self):
        scene = build_scene()
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(scene, suffix)
                self.assert_same(scene, loaded, "scene")
                self.assertEqual(loaded, scene)

    def test_the_loaded_nodes_belong_to_the_loaded_hierarchy(self):
        scene = build_scene()
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(scene, suffix)
                for node, original in zip(loaded.hierarchy, scene.hierarchy):
                    self.assertIs(node._hierarchy, loaded.hierarchy)
                    parent = original.get_parent()
                    if parent is None:
                        self.assertIsNone(node.get_parent())
                    else:
                        self.assertEqual(node.get_parent().uuid, parent.uuid)
                    np.testing.assert_allclose(node.world_matrix, original.world_matrix)
                # the shapes still name their nodes and meshes, by uuid
                self.assertEqual([loaded.hierarchy[m.parent_uuid].name for m in loaded.meshes], ["hero", "crate"])
                self.assertEqual([m.mesh_uuid for m in loaded.materials], [loaded.meshes[0].uuid] * 2 + [loaded.meshes[1].uuid])
                self.assertEqual(loaded.clip.active.take, "bend")

    def test_the_loaded_scene_picks_the_same_camera(self):
        scene = build_scene()
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(scene, suffix)
                camera, node = loaded.camera()
                self.assertEqual((camera.name, loaded.hierarchy[node].name), ("top", "side"))

    def test_a_loaded_scene_has_no_cached_render(self):
        scene       = build_scene()
        scene.frame = object()
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assertIsNone(_round_trip(scene, suffix).frame)

    def test_a_copy_keeps_the_settings(self):
        scene = build_scene()
        self.assert_same(scene, scene.copy(), "scene")

    def test_none_settings_round_trip(self):
        scene = Scene("bare", resolution=None)
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(scene, suffix)
                self.assertIsNone(loaded.resolution)
                self.assertIsNone(loaded.background)
                self.assertIsNone(loaded.meshes)
                self.assertEqual(loaded.name, "bare")


class TestArraySettings(unittest.TestCase):
    """small fixed settings are arrays of a fixed size"""

    def test_scene_settings_are_arrays(self):
        scene = Scene("s", background=(0.1, 0.2, 0.3))
        for name, dtype, shape in (
            ("base_color", np.float64, (3,)),
            ("background", np.float64, (4,)),
            ("resolution", np.int32, (2,)),
            ("wireframe_color", np.int32, (3,)),
        ):
            with self.subTest(name):
                value = getattr(scene, name)
                self.assertEqual(value.dtype, dtype)
                self.assertEqual(value.shape, shape)
        np.testing.assert_array_equal(scene.background, (0.1, 0.2, 0.3, 1.0))

    def test_a_wrong_size_raises(self):
        scene = Scene("s")
        for name, value in (
            ("base_color", (1.0, 0.0)),
            ("resolution", (1, 2, 3)),
            ("wireframe_color", (0, 0, 0, 0)),
            ("background", (0.5, 0.5)),
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    setattr(scene, name, value)
        with self.assertRaises(ValueError):
            Scene("s", background=(1.5, 0.0, 0.0))

    def test_the_render_signature_compares_array_settings(self):
        """a new array of the same value keeps the cached frame"""
        scene = Scene("s")
        mesh  = cube_mesh()
        scene.add(mesh, LightData(name="key"))

        marker                = object()
        scene.frame           = marker
        scene.lights[0].color = np.array([1.0, 1.0, 1.0])
        scene.resolution      = [500, 500]
        scene.base_color      = np.array([0.7, 0.7, 0.7])
        self.assertIs(scene.frame, marker)
        hash(scene._signature())

        scene.lights[0].color = (1.0, 0.5, 0.5)
        self.assertIsNone(scene.frame)


class TestSceneNamespaces(unittest.TestCase):
    def test_strip_namespace_strips_the_default_camera(self):
        scene = Scene("s", default_camera="A:side")
        scene.add(CameraData(name="A:front"), CameraData(name="A:side"))
        scene.strip_namespace()
        self.assertEqual(scene.default_camera, "side")
        self.assertEqual(scene.camera()[0].name, "side")

    def test_strip_one_namespace_path(self):
        scene = Scene("s", default_camera="A:B:side")
        scene.add(CameraData(name="A:B:side"))
        scene.strip_namespace("A")
        self.assertEqual(scene.default_camera, "B:side")

    def test_the_lists_are_stripped(self):
        scene = Scene("s")
        node  = TransformData(name="A:hero")
        scene.add(node)
        mesh             = cube_mesh("A:hero_geo")
        mesh.parent_uuid = node.uuid
        scene.add(mesh, MaterialData(name="A:skin", mesh_uuid=mesh.uuid))
        scene.strip_namespace()
        self.assertEqual((scene.hierarchy[0].name, scene.meshes[0].name, scene.materials[0].name), ("hero", "hero_geo", "skin"))

    def test_no_default_camera(self):
        scene = Scene("s")
        scene.add(CameraData(name="A:cam"))
        scene.strip_namespace()
        self.assertIsNone(scene.default_camera)


@unittest.skipIf(not HAVE_RASTERIZER, "the wireframe bake needs cv2 or scikit-image")
class TestWireframeBake(unittest.TestCase):
    def test_a_float_texture_bakes_as_bright_as_uint8(self):
        """a float32 [0, 1] texture is scaled before the bake, not cast"""
        baked = []
        for texture in (checker_texture(), checker_texture().astype(np.float32) / 255.0):
            scene = Scene("s", wireframe=True)
            mesh  = cube_mesh()
            uv    = cube_uv(mesh.uuid)
            scene.add(mesh, uv)
            material = MaterialData(mesh_uuid=mesh.uuid, texture=texture, uv_set="atlas")
            scene.add(material)
            baked.append(scene._texture(material, uv))
        np.testing.assert_allclose(baked[1], baked[0], atol=1.0 / 255.0)


if __name__ == "__main__":
    unittest.main()
