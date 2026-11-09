"""Round trips of the render Scene beyond the one example test_roundtrip saves:
the stage with every texture dtype a material keeps (uint16 and float32, grey
(H, W, 1) and RGBA (H, W, 4)), and a Scene given its sample method as a
SampleMethod."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath._tests._examples_render import build_scene, checker_texture, cube_mesh
from cgmath._tests.test_roundtrip import FORMATS, assert_same
from cgmath.geometry.mesh import SampleMethod
from cgmath.render import Scene
from cgmath.render.scene import _RENDER_CONFIG_KEYS, _SHADING_KEYS


def _textures() -> dict:
    """the checker in each layout and dtype a material keeps as given"""
    rgb  = checker_texture()
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 200, np.uint8)], axis=-1)
    return {
        "uint16 rgb":   rgb.astype(np.uint16) * 257,
        "uint16 rgba":  rgba.astype(np.uint16) * 257,
        "float32 rgb":  rgb.astype(np.float32) / 255.0,
        "float32 grey": rgb[..., 1:2].astype(np.float32) / 255.0,
        "uint8 rgba":   rgba,
    }


class TestTextureDtypes(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def round_trips(self, scene, label: str) -> None:
        """``scene`` loads back the same from every format, named and sniffed,
        and its copy too"""
        for ext in FORMATS:
            path = os.path.join(self.tmp, f"scene.{ext}")
            scene.save(path)
            for mode in (ext, None):
                assert_same(self, scene, Scene.load(path, mode=mode), f"{label}.{ext}")
        assert_same(self, scene, scene.copy(), f"{label}.copy()")

    def test_the_scene_keeps_each_texture_dtype(self):
        for label, texture in _textures().items():
            with self.subTest(texture=label):
                scene = build_scene(texture)
                self.assertEqual(scene.materials[0].texture.dtype, texture.dtype)
                self.round_trips(scene, label)

    def test_a_float64_texture_is_kept_as_float32(self):
        """float64 is not a texture dtype: it becomes float32 when it is set,
        so the file holds what the material holds"""
        texture = checker_texture().astype(np.float64) / 255.0
        scene   = build_scene(texture)
        self.assertEqual(scene.materials[0].texture.dtype, np.float32)
        self.round_trips(scene, "float64")

    def test_a_loaded_texture_samples_as_the_original(self):
        """what the renderer samples is the same after a load"""
        for label, texture in _textures().items():
            scene = build_scene(texture)
            uv    = scene.uvs[0]
            for ext in FORMATS:
                with self.subTest(texture=label, format=ext):
                    path   = scene.save(os.path.join(self.tmp, f"s.{ext}"))
                    loaded = Scene.load(path)
                    np.testing.assert_array_equal(
                        loaded._texture(loaded.materials[0], loaded.uvs[0]), scene._texture(scene.materials[0], uv)
                    )


class TestSampleMethodEnum(unittest.TestCase):
    """a Scene given SampleMethod.BEZIER keeps the enum's value, which every
    format saves and render() takes"""

    def test_the_enum_is_stored_as_its_value(self):
        by_init                    = Scene("a", sample_method=SampleMethod.BEZIER)
        by_attribute               = Scene("b")
        by_attribute.sample_method = SampleMethod.BEZIER
        for scene in (by_init, by_attribute):
            with self.subTest(scene.name):
                self.assertEqual(scene.sample_method, "bezier")

    def test_a_scene_given_the_enum_saves_and_renders(self):
        scene = Scene("o", sample_method=SampleMethod.BEZIER, resolution=(8, 8), samples_per_pixel=1)
        scene.add(cube_mesh())
        with tempfile.TemporaryDirectory() as tmp:
            for ext in FORMATS:
                with self.subTest(ext):
                    path = scene.save(os.path.join(tmp, f"o.{ext}"))
                    assert_same(self, scene, Scene.load(path), f"o.{ext}")
        frame = scene.render()
        self.assertEqual(frame.array.shape[:2], (8, 8))


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


class TestExamplesFillEveryField(unittest.TestCase):
    """the example changes every field it saves, so a field a load loses
    cannot pass for one left at its default"""

    def test_every_scene_setting_is_away_from_its_default(self):
        built, default = build_scene(), Scene()
        for name in _RENDER_CONFIG_KEYS + _SHADING_KEYS + ("name", "default_camera", "aspect_ratio"):
            with self.subTest(setting=name):
                self.assertFalse(_same(getattr(built, name), getattr(default, name)))

    def test_every_list_is_held_and_saved(self):
        built = build_scene()
        tree  = built.to_dict()
        for field in fields(built):
            if field.name not in Scene.TRANSIENT_FIELDS:
                with self.subTest(field=field.name):
                    self.assertIsNotNone(getattr(built, field.name))
                    self.assertIn(field.name, tree)


if __name__ == "__main__":
    unittest.main()
