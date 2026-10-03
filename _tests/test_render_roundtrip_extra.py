"""Round trips of the render classes beyond the one example per class that
test_roundtrip saves: the hero Object of _examples_render with every texture
dtype it keeps (uint16 and float32, grey (H, W) and RGBA (H, W, 4)), alone and
inside the Scene, and an Object given its sample method as a SampleMethod."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath._tests._examples_render import EXAMPLES, build_scene, checker_texture, cube_mesh
from cgmath._tests.test_roundtrip import FORMATS, assert_same
from cgmath.geometry.mesh import SampleMethod
from cgmath.render import Object, Scene


def _textures() -> dict:
    """the checker in each layout and dtype an Object keeps as given"""
    rgb  = checker_texture()
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 200, np.uint8)], axis=-1)
    return {
        "uint16 rgb":   rgb.astype(np.uint16) * 257,
        "uint16 rgba":  rgba.astype(np.uint16) * 257,
        "float32 rgb":  rgb.astype(np.float32) / 255.0,
        "float32 grey": rgb[..., 1].astype(np.float32) / 255.0,
        "uint8 rgba":   rgba,
    }


class TestTextureDtypes(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def round_trips(self, obj, label: str) -> None:
        """``obj`` loads back the same from every format, named and sniffed,
        and its copy too"""
        cls = type(obj)
        for ext in FORMATS:
            path = os.path.join(self.tmp, f"{cls.__name__}.{ext}")
            obj.save(path)
            for mode in (ext, None):
                assert_same(self, obj, cls.load(path, mode=mode), f"{label}.{ext}")
        assert_same(self, obj, obj.copy(), f"{label}.copy()")

    def test_the_object_keeps_each_texture_dtype(self):
        for label, texture in _textures().items():
            with self.subTest(texture=label):
                obj = build_scene(texture)["hero"]
                self.assertEqual(obj.texture.dtype, texture.dtype)
                self.round_trips(obj, label)

    def test_the_scene_keeps_each_texture_dtype(self):
        for label, texture in _textures().items():
            with self.subTest(texture=label):
                self.round_trips(build_scene(texture), label)

    def test_a_float64_texture_is_kept_as_float32(self):
        """float64 is not a texture dtype: it becomes float32 when it is set,
        so the file holds what the Object holds"""
        texture = checker_texture().astype(np.float64) / 255.0
        obj     = build_scene(texture)["hero"]
        self.assertEqual(obj.texture.dtype, np.float32)
        self.round_trips(obj, "float64")

    def test_a_loaded_texture_reads_as_the_original(self):
        """what the renderer samples is the same after a load"""
        for label, texture in _textures().items():
            obj = Object(name="o", texture=texture)
            for ext in FORMATS:
                with self.subTest(texture=label, format=ext):
                    path = obj.save(os.path.join(self.tmp, f"o.{ext}"))
                    np.testing.assert_array_equal(
                        Object.load(path).get_loaded_texture(), obj.get_loaded_texture()
                    )


class TestSampleMethodEnum(unittest.TestCase):
    """an Object given SampleMethod.BEZIER: the setter kept the enum, which no
    format could save (pkl, npz and json all raised TypeError), and the
    constructor kept str() of it, "SampleMethod.BEZIER", which render()
    rejected"""

    def test_the_enum_is_stored_as_its_value(self):
        by_init                    = Object(name="a", sample_method=SampleMethod.BEZIER)
        by_attribute               = Object(name="b")
        by_attribute.sample_method = SampleMethod.BEZIER
        for obj in (by_init, by_attribute):
            with self.subTest(obj.name):
                self.assertEqual(obj.sample_method, "bezier")

    def test_an_object_given_the_enum_saves(self):
        obj               = Object(name="o", mesh=cube_mesh())
        obj.sample_method = SampleMethod.BEZIER
        with tempfile.TemporaryDirectory() as tmp:
            for ext in FORMATS:
                with self.subTest(ext):
                    path = obj.save(os.path.join(tmp, f"o.{ext}"))
                    assert_same(self, obj, Object.load(path), f"o.{ext}")

    def test_an_object_given_the_enum_renders(self):
        obj   = Object(name="o", mesh=cube_mesh(), sample_method=SampleMethod.BEZIER)
        frame = obj.render(resolution=(8, 8), samples_per_pixel=1)
        self.assertEqual(frame.array.shape[:2], (8, 8))


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


class TestExamplesFillEveryField(unittest.TestCase):
    """the examples change every field they save, so a field a load loses
    cannot pass for one left at its default"""

    def test_every_scene_setting_is_away_from_its_default(self):
        built, default = build_scene(), Scene()
        for name in Scene.LIST_FIELDS:
            with self.subTest(setting=name):
                self.assertFalse(_same(getattr(built, name), getattr(default, name)))

    def test_every_field_is_saved(self):
        """a field equal to its default is left out of the tree"""
        for cls, builder in EXAMPLES.items():
            if cls is Scene:
                continue
            obj  = builder()
            tree = obj.to_dict()
            for field in fields(obj):
                if field.name not in cls.TRANSIENT_FIELDS:
                    with self.subTest(cls=cls.__name__, field=field.name):
                        self.assertIn(field.name, tree)


if __name__ == "__main__":
    unittest.main()
