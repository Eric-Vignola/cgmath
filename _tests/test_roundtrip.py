"""Every cgmath data class saves and loads back exactly, in pkl, npz and json.

Each class is built filled in, by a builder in one of the ``_examples_*``
modules beside this file, saved in each format, loaded back, and compared
strictly: the same class, every field, every array's dtype and shape and
values, list against tuple, None against empty. A class with no builder fails
the test, so a new class cannot slip past it.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import tempfile
import time
import unittest
from dataclasses import fields

import numpy as np
import cgmath
from cgmath.geometry._base import Data, DataList

# the builder modules: each holds EXAMPLES = {class: builder}, and may hold
# SAVES_AS = {class: function}, for a class that by design loads back as
# something else (a TransformList view loads as the hierarchy its copy() gives)
_EXAMPLE_MODULES = (
    "cgmath._tests._examples_hierarchy",
    "cgmath._tests._examples_render",
    "cgmath._tests._examples_geometry",
    "cgmath._tests._examples_curves",
    "cgmath._tests._examples_deform",
)

FORMATS = ("pkl", "npz", "json")


def data_classes():
    """every Data / DataList subclass cgmath defines, and the modules that would not import"""
    found, skipped = {}, []
    for module in pkgutil.walk_packages(cgmath.__path__, "cgmath."):
        if "._tests" in module.name or module.name.endswith("__main__"):
            continue
        try:
            imported = importlib.import_module(module.name)
        except Exception as exc:  # an optional dependency (pxr) is missing
            skipped.append(f"{module.name}: {type(exc).__name__}")
            continue
        for _, cls in inspect.getmembers(imported, inspect.isclass):
            if (
                issubclass(cls, (Data, DataList))
                and cls not in (Data, DataList)
                and cls.__module__.startswith("cgmath.")
                and "._tests" not in cls.__module__
            ):
                found[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return found, skipped


def examples():
    """every builder, and every SAVES_AS, by class"""
    builders, saves_as = {}, {}
    for name in _EXAMPLE_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            continue  # its classes then fail test_every_class_has_a_builder
        builders.update(module.EXAMPLES)
        saves_as.update(getattr(module, "SAVES_AS", {}))
    return builders, saves_as


def assert_same(test, a, b, path="obj"):
    """a and b hold exactly the same data"""
    if isinstance(a, (Data, DataList)) or isinstance(b, (Data, DataList)):
        test.assertIs(type(b), type(a), f"{path}: class")

    if isinstance(a, DataList):
        test.assertEqual(len(b), len(a), f"{path}: length")
        for index, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{path}[{index}]")
        for name in a.LIST_FIELDS:
            assert_same(test, getattr(a, name), getattr(b, name), f"{path}.{name}")

        # an owning list's items point back at it
        for index, item in enumerate(b):
            owner = getattr(item, "_hierarchy", None)
            if getattr(a[index], "_hierarchy", None) is a:
                test.assertIs(owner, b, f"{path}[{index}]: back-pointer")
        return

    if isinstance(a, Data):
        skip = set(type(a).TRANSIENT_FIELDS)
        for field in fields(a):
            if field.name in skip:
                continue
            missing = object()
            x, y = getattr(a, field.name, missing), getattr(b, field.name, missing)
            test.assertEqual(x is missing, y is missing, f"{path}.{field.name}: set")
            if x is not missing:
                assert_same(test, x, y, f"{path}.{field.name}")
        return

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        test.assertIsInstance(a, np.ndarray, f"{path}: array")
        test.assertIsInstance(b, np.ndarray, f"{path}: array")
        test.assertEqual(b.dtype, a.dtype, f"{path}: dtype")
        test.assertEqual(b.shape, a.shape, f"{path}: shape")
        test.assertTrue(
            np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), f"{path}: values"
        )
        return

    if isinstance(a, dict):
        test.assertIsInstance(b, dict, f"{path}: dict")
        test.assertEqual(list(b), list(a), f"{path}: keys")
        for key in a:
            assert_same(test, a[key], b[key], f"{path}[{key!r}]")
        return

    if isinstance(a, (list, tuple)):
        test.assertIs(type(b), type(a), f"{path}: list or tuple")
        test.assertEqual(len(b), len(a), f"{path}: length")
        for index, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{path}[{index}]")
        return

    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return  # both NaN

    # a numpy scalar comes back as the python number it equals
    if isinstance(a, np.generic):
        a = a.item()
    test.assertIs(type(b), type(a), f"{path}: type")
    test.assertEqual(b, a, path)


class TestEveryClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes, cls.skipped = data_classes()
        cls.builders, cls.saves_as = examples()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_every_class_has_a_builder(self):
        missing = sorted(name for name, cls in self.classes.items() if cls not in self.builders)
        self.assertEqual(missing, [], "write a builder in a _examples_* module")

    def test_every_format_loads_back_the_same_object(self):
        for name, cls in sorted(self.classes.items()):
            builder = self.builders.get(cls)
            if builder is None:
                continue
            with self.subTest(cls=name):
                obj       = builder()
                reference = self.saves_as.get(cls, lambda x: x)(obj)
                loads     = {}
                for ext in FORMATS:
                    path = os.path.join(self.tmp, f"{cls.__name__}.{ext}")
                    obj.save(path)
                    for mode in (ext, None):  # the format named, and sniffed
                        back = cls.load(path, mode=mode)
                        assert_same(self, reference, back, f"{cls.__name__}.{ext}")
                    loads[ext] = back

                # and all three hold the same tree
                trees = {ext: back.to_dict() for ext, back in loads.items()}
                assert_same(self, trees["pkl"], trees["npz"], f"{cls.__name__}: pkl vs npz")
                assert_same(self, trees["pkl"], trees["json"], f"{cls.__name__}: pkl vs json")

    def test_copy_is_the_same_object(self):
        for name, cls in sorted(self.classes.items()):
            builder = self.builders.get(cls)
            if builder is None:
                continue
            with self.subTest(cls=name):
                obj = builder()
                assert_same(self, self.saves_as.get(cls, lambda x: x)(obj), obj.copy(), f"{cls.__name__}.copy()")


class TestNpzSpeed(unittest.TestCase):
    def test_a_million_points_load_quickly(self):
        from cgmath.geometry import MeshData

        count  = 1_000_000
        points = np.random.default_rng(0).random((count, 3))
        mesh = MeshData(
            indices = np.arange(count - count % 4, dtype=np.int32),
            counts  = np.full(count // 4, 4, np.int32),
            points  = points,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path  = mesh.save(os.path.join(tmp, "big.npz"))
            start = time.perf_counter()
            MeshData.load(path)
            # the old reader took 2.7 s, almost all of it a per-element scan
            self.assertLess(time.perf_counter() - start, 0.5)


if __name__ == "__main__":
    unittest.main()
