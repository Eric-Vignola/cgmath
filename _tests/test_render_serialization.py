"""pkl / npz / json round trips of the render classes: Object, Camera, Light and Scene."""

import os
import pickle
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath.geometry._base import Data, DataList
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.hierarchy import TransformData
from cgmath.render import Camera, Light, Object, Scene

try:
    from PIL import Image
except ImportError:
    Image = None

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


def _cube():
    """a quad cube and its UVs, one island per face"""
    points = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ]
    )
    quads  = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    counts = np.full(6, 4, dtype=np.int32)
    mesh = MeshData(
        points  = points,
        indices = np.array(quads, dtype=np.int32).ravel(),
        counts  = counts,
        name    = "cube",
    )
    uv = UVData(
        points  = np.tile([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], (6, 1)),
        indices = np.arange(24, dtype=np.int32),
        counts  = counts,
    )
    return mesh, uv


def _checker(n=8):
    tex           = np.full((n, n, 3), 40, dtype=np.uint8)
    tex[::2, ::2] = 220
    return tex


def _round_trip(data, suffix):
    """``data`` saved as ``suffix`` and loaded back"""
    handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    try:
        data.save(path)
        return type(data).load(path)
    finally:
        os.remove(path)


def _scene():
    """a group, an Object under it, a Camera under the Object, a Light under
    the Camera, and every Scene setting away from its default"""
    mesh, uv = _cube()
    scene = Scene(
        "stage",
        aspect_ratio        = 1.5,
        default_camera_name = "side",
        resolution          = (320, 200),
        samples_per_pixel   = 9,
        background          = (0.1, 0.2, 0.3),
        autofit             = False,
        default_light       = False,
        return_depth        = True,
        angle_of_view       = 40.0,
        fit_padding         = 1.25,
    )
    group = TransformData(name="grp", translate=[0.0, 1.0, 0.0])
    hero = Object(
        name="hero",
        mesh=mesh,
        uv=uv,
        texture=_checker(),
        base_color=(0.2, 0.4, 0.6),
        sample_method="bezier",
        wrap="clamp",
        ambient=0.5,
        twosided=False,
        cast_shadows=False,
        resolution=(64, 32),
        samples_per_pixel=16,
        wireframe=True,
        wireframe_color=(255, 128, 0),
        wireframe_thickness=2,
        background=(0.5, 0.5, 0.5, 0.25),
        translate=[1.0, 2.0, 3.0],
        rotate=[10.0, 20.0, 30.0],
    )
    front = Camera(name="front", translate=[0.0, 0.0, 10.0])
    side = Camera(
        name            = "side",
        angle_of_view   = 50.0,
        is_orthographic = True,
        ortho_height    = 12.5,
        near_plane      = 0.1,
        far_plane       = 500.0,
        aspect_ratio    = 2.0,
        translate       = [10.0, 0.0, 0.0],
    )
    key = Light(name="key", kind="infinite", color=(0.9, 0.8, 0.7), intensity=3.0, falloff=False)

    for node in (group, hero, front, side, key):
        scene.append(node)
    hero.set_parent(group)
    side.set_parent(hero)
    key.set_parent(side)
    return scene


class RoundTripCase(unittest.TestCase):
    def assert_same(self, a, b, where="data"):
        """``b`` is exactly ``a``: its type, and every field's value, dtype and shape"""
        self.assertIs(type(b), type(a), where)
        if isinstance(a, np.ndarray):
            self.assertEqual(b.dtype, a.dtype, where)
            self.assertEqual(b.shape, a.shape, where)
            np.testing.assert_array_equal(b, a, err_msg=where)
        elif isinstance(a, Data):
            for field in fields(a):
                if field.name not in type(a).TRANSIENT_FIELDS:
                    self.assert_same(
                        getattr(a, field.name), getattr(b, field.name), f"{where}.{field.name}"
                    )
        elif isinstance(a, DataList):
            self.assertEqual(len(b), len(a), where)
            for index, (x, y) in enumerate(zip(a, b)):
                self.assert_same(x, y, f"{where}[{index}]")
            for name in type(a).LIST_FIELDS:
                self.assert_same(getattr(a, name), getattr(b, name), f"{where}.{name}")
        elif isinstance(a, dict):
            self.assertEqual(set(b), set(a), where)
            for key in a:
                self.assert_same(a[key], b[key], f"{where}[{key!r}]")
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(b), len(a), where)
            for index, (x, y) in enumerate(zip(a, b)):
                self.assert_same(x, y, f"{where}[{index}]")
        else:
            self.assertEqual(b, a, where)


class TestSceneRoundTrip(RoundTripCase):
    def test_every_format_gives_back_the_scene(self):
        """bug 15: npz and json lost the node types (Objects came back as
        TransformData, Camera and Light raised) and every Scene setting"""
        scene = _scene()
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assert_same(scene, _round_trip(scene, suffix), "scene")

    def test_the_loaded_nodes_belong_to_the_loaded_scene(self):
        scene = _scene()
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(scene, suffix)
                for node, original in zip(loaded, scene):
                    self.assertIs(node._hierarchy, loaded)
                    parent = original.get_parent()
                    if parent is None:
                        self.assertIsNone(node.get_parent())
                    else:
                        self.assertEqual(node.get_parent().uuid, parent.uuid)
                    np.testing.assert_allclose(node.world_matrix, original.world_matrix)

    def test_the_loaded_scene_picks_the_same_camera(self):
        scene = _scene()
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assertEqual(_round_trip(scene, suffix).get_camera().name, "side")

    def test_a_loaded_scene_has_no_cached_render(self):
        scene       = _scene()
        scene.frame = object()
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assertIsNone(_round_trip(scene, suffix).frame)

    def test_a_copy_keeps_the_settings(self):
        scene = _scene()
        self.assert_same(scene, scene.copy(), "scene")

    def test_none_settings_round_trip(self):
        scene = Scene("bare", resolution=None)
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(scene, suffix)
                self.assertIsNone(loaded.resolution)
                self.assertIsNone(loaded.background)
                self.assertEqual(loaded.scene_name, "bare")


class TestObjectRoundTrip(RoundTripCase):
    def test_every_format_gives_back_the_object(self):
        obj = _scene()["hero"]
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assert_same(obj, _round_trip(obj, suffix), "hero")

    def test_a_texture_keeps_its_dtype(self):
        """D3: the saved file used to quantise every texture to uint8, and
        json read that back as int64, 255 times too bright"""
        textures = {
            "uint8":   _checker(),
            "uint16":  _checker().astype(np.uint16) * 257,
            "float32": _checker().astype(np.float32) / 255.0,
        }
        for label, texture in textures.items():
            obj = Object(name="o", texture=texture)
            for suffix in FORMATS:
                with self.subTest(label=label, format=suffix):
                    loaded = _round_trip(obj, suffix)
                    self.assert_same(obj.texture, loaded.texture, "texture")
                    np.testing.assert_allclose(
                        loaded.get_loaded_texture(), obj.get_loaded_texture(), atol=1e-6
                    )

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_a_texture_path_is_saved_as_its_pixels(self):
        tex = _checker()
        handle, path = tempfile.mkstemp(suffix=".png")
        os.close(handle)
        try:
            Image.fromarray(tex).save(path)
            obj = Object(name="o", texture=path)
            self.assertEqual(obj.texture, path)
            for suffix in FORMATS:
                with self.subTest(suffix):
                    self.assert_same(tex, _round_trip(obj, suffix).texture, "texture")
        finally:
            os.remove(path)

    def test_settings_take_new_values_after_a_reload(self):
        """D7: a reloaded setting is an array, and the setters compared tuples
        with it, which raised ValueError"""
        obj = Object(
            name            = "o",
            base_color      = (0.1, 0.2, 0.3),
            background      = (0.0, 0.0, 0.0),
            resolution      = (64, 32),
            wireframe_color = (255, 0, 0),
        )
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded                 = _round_trip(obj, suffix)
                loaded.base_color      = (1.0, 0.0, 0.0)
                loaded.background      = (1.0, 1.0, 1.0)
                loaded.resolution      = (16, 8)
                loaded.wireframe_color = (0, 255, 0)
                np.testing.assert_array_equal(loaded.base_color,      [1.0, 0.0, 0.0])
                np.testing.assert_array_equal(loaded.background,      [1.0, 1.0, 1.0, 1.0])
                np.testing.assert_array_equal(loaded.resolution,      [16, 8])
                np.testing.assert_array_equal(loaded.wireframe_color, [0, 255, 0])

    def test_caches_are_not_fields(self):
        names = {field.name for field in fields(Object)}
        for cache in (
            "_loaded_texture",
            "_wired_texture_cache",
            "_bind_mesh",
            "_render_transform_state",
            "_frame",
        ):
            self.assertNotIn(cache, names)


class TestCameraLightRoundTrip(RoundTripCase):
    def test_a_camera_round_trips(self):
        camera = _scene()["side"]
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assert_same(camera, _round_trip(camera, suffix), "side")

    def test_an_infinite_far_plane_round_trips(self):
        camera = Camera(name="cam", far_plane=float("inf"), near_plane=float("inf"))
        for suffix in FORMATS:
            with self.subTest(suffix):
                loaded = _round_trip(camera, suffix)
                self.assertEqual(loaded.near_plane, float("inf"))
                self.assertEqual(loaded.far_plane, float("inf"))

    def test_a_light_round_trips(self):
        light = _scene()["key"]
        for suffix in FORMATS:
            with self.subTest(suffix):
                self.assert_same(light, _round_trip(light, suffix), "key")


class TestArraySettings(unittest.TestCase):
    """D7: small fixed settings are arrays of a fixed size"""

    def test_object_settings_are_arrays(self):
        obj = Object(name="o", background=(0.1, 0.2, 0.3))
        for name, dtype, shape in (
            ("base_color", np.float64, (3,)),
            ("background", np.float64, (4,)),
            ("resolution", np.int32, (2,)),
            ("wireframe_color", np.int32, (3,)),
        ):
            with self.subTest(name):
                value = getattr(obj, name)
                self.assertEqual(value.dtype, dtype)
                self.assertEqual(value.shape, shape)

    def test_light_and_scene_settings_are_arrays(self):
        light = Light(name="l", color=[1, 0, 0])
        self.assertEqual(light.color.dtype, np.float64)
        scene = Scene("s", background=(0.1, 0.2, 0.3))
        self.assertEqual(scene.resolution.dtype, np.int32)
        self.assertEqual(scene.background.dtype, np.float64)
        self.assertEqual(scene.background.shape, (4,))

    def test_a_wrong_size_raises(self):
        """D4"""
        obj = Object(name="o")
        for name, value in (
            ("base_color", (1.0, 0.0)),
            ("resolution", (1, 2, 3)),
            ("wireframe_color", (0, 0, 0, 0)),
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    setattr(obj, name, value)
        with self.assertRaises(ValueError):
            Light(name="l").color = (1.0, 1.0)
        with self.assertRaises(ValueError):
            Scene("s").resolution = (1, 2, 3)

    def test_a_setting_is_the_objects_own_copy(self):
        """a tuple could not be edited in place; an array the caller keeps can"""
        color    = np.array([0.1, 0.2, 0.3])
        obj      = Object(name="o", base_color=color)
        color[0] = 1.0
        self.assertEqual(obj.base_color[0], 0.1)

    def test_the_render_signature_compares_array_settings(self):
        """a new array of the same value keeps the cached frame; it used to
        raise ValueError from comparing a tuple with an array"""
        mesh, uv = _cube()
        scene = Scene("s")
        obj   = Object(name="o", mesh=mesh, uv=uv)
        light = Light(name="key")
        scene.append(obj)
        scene.append(light)

        marker           = object()
        scene.frame      = marker
        light.color      = np.array([1.0, 1.0, 1.0])
        scene.resolution = [500, 500]
        obj.base_color   = np.array([0.7, 0.7, 0.7])
        self.assertIs(scene.frame, marker)
        hash(scene._compute_render_signature())

        light.color = (1.0, 0.5, 0.5)
        self.assertIsNone(scene.frame)


class TestSceneNamespaces(unittest.TestCase):
    def test_strip_namespace_strips_the_default_camera_name(self):
        """bug 15: the Scene's default camera name kept its namespace, so the
        Scene fell back to its first Camera"""
        scene = Scene("s", default_camera_name="A:side")
        scene.append(Camera(name="A:front"))
        scene.append(Camera(name="A:side"))

        scene.strip_namespace()

        self.assertEqual(scene.default_camera_name, "side")
        self.assertEqual(scene.get_camera().name, "side")

    def test_strip_one_namespace_path(self):
        scene = Scene("s", default_camera_name="A:B:side")
        scene.append(Camera(name="A:B:side"))
        scene.strip_namespace("A")
        self.assertEqual(scene.default_camera_name, "B:side")

    def test_a_clash_renames_nothing(self):
        scene = Scene("s", default_camera_name="A:cam")
        scene.append(Camera(name="A:cam"))
        scene.append(Camera(name="B:cam"))
        with self.assertRaises(ValueError):
            scene.strip_namespace()
        self.assertEqual(scene.default_camera_name, "A:cam")

    def test_no_default_camera_name(self):
        scene = Scene("s")
        scene.append(Camera(name="A:cam"))
        scene.strip_namespace()
        self.assertIsNone(scene.default_camera_name)


@unittest.skipIf(not HAVE_RASTERIZER, "the wireframe bake needs cv2 or scikit-image")
class TestWireframeBake(unittest.TestCase):
    def test_a_float_texture_bakes_as_bright_as_uint8(self):
        """a float32 [0, 1] texture, kept as float32 since D3, baked black:
        the bake cast it straight to uint8"""
        mesh, uv = _cube()
        baked = []
        for texture in (_checker(), _checker().astype(np.float32) / 255.0):
            obj = Object(name="o", mesh=mesh, uv=uv.copy(), texture=texture, wireframe=True)
            baked.append(obj.get_loaded_texture())
        np.testing.assert_allclose(baked[1], baked[0], atol=1.0 / 255.0)


if __name__ == "__main__":
    unittest.main()
