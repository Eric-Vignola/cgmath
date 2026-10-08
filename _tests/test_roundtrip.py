"""Every cgmath data class saves and loads back exactly, in pkl, npz and json.

Each class is built filled in, by a builder in one of the ``_examples_*``
modules beside this file, saved in each format, loaded back, and compared
strictly: the same class, every field, every array's dtype and shape and
values, list against tuple, None against empty. A class with no builder fails
the test, so a new class cannot slip past it.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import os
import pkgutil
import tempfile
import time
import unittest
from dataclasses import fields

import numpy as np
import cgmath
from cgmath.geometry._base import CLASS_KEY, Data, DataList, ID_KEY

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

    def test_a_tree_that_shares_nothing_numbers_nothing(self):
        # only an object reached twice is numbered: an object that shares
        # nothing saves the same file as before numbers existed
        def numbered(value):
            if isinstance(value, dict):
                if ID_KEY in value or value.get(CLASS_KEY) == "~ref":
                    return True
                return any(numbered(x) for x in value.values())
            if isinstance(value, (list, tuple)):
                return any(numbered(x) for x in value)
            return False

        for name, cls in sorted(self.classes.items()):
            builder = self.builders.get(cls)
            if builder is None:
                continue
            with self.subTest(cls=name):
                self.assertFalse(numbered(builder().to_dict()))


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


class TestJsonSpeed(unittest.TestCase):
    def test_a_million_points_write_quickly(self):
        """to_json takes under three times as long as json.dumps of the same arrays"""
        from cgmath.geometry import MeshData

        count  = 1_000_000
        points = np.random.default_rng(0).random((count, 3))
        mesh = MeshData(
            indices = np.arange(count - count % 4, dtype=np.int32),
            counts  = np.full(count // 4, 4, np.int32),
            points  = points,
        )

        def seconds(write):
            start = time.perf_counter()
            write()
            return time.perf_counter() - start

        def dumps():
            json.dumps([mesh.points.tolist(), mesh.indices.tolist(), mesh.counts.tolist()])

        # a time in seconds failed on a slower machine (4.5 s against a 4 s
        # limit), so the yardstick is the standard library's json.dumps of the
        # same arrays as lists, timed in the same run: a slower machine slows
        # both. It is timed on either side of to_json and the quicker of the
        # two taken, as a burst of load can slow either: under load, with the
        # collector on, the per-value writer once took 1.95 times the slower,
        # which would pass, and 3.95 times the quicker. The garbage collector
        # is off while they run, as in timeit: json.dumps makes a million
        # small lists, and the collections they set off walk every object the
        # process holds, so with it on the same json.dumps took 1.1 to 3.3 s,
        # depending on what ran before.
        bound   = 3.0
        rounds  = []
        enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            # a round that misses the bound runs once more, so a burst of load
            # that lands on to_json alone does not fail the test
            for _ in range(2):
                before = seconds(dumps)
                took   = seconds(mesh.to_json)
                after  = seconds(dumps)
                ratio  = took / min(before, after)
                rounds.append(f"to_json took {took:.2f} s, json.dumps {before:.2f} s and {after:.2f} s")
                if ratio < bound:
                    break
        finally:
            if enabled:
                gc.enable()
        # the per-value writer took at least 4.6 times as long as the quicker
        # json.dumps, the bulk one 1.3 to 1.8 idle and at most 2.8 in a round
        # under load (python 3.11 and mayapy 2025, alone and in the full suite)
        self.assertLess(ratio, bound, "; ".join(rounds))


if __name__ == "__main__":
    unittest.main()
