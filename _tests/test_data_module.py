"""Data and DataList live in cgmath.data, and what cgmath 2.x saved still loads.

Up to cgmath 2.x the module was cgmath.geometry._base, which is gone. A file
saved then names its class, a plain Data or DataList under that module, a key
a load skips, and every pickle names _unpickle and _unpickle_array there.
cgmath's own loaders read them all; a bare pickle.load of such a pickle fails.
A pickle saved now holds plain data, which a bare pickle.load reads as a dict.

test_assets holds what 2.0.0 saved, one object of every class the module
defines and a few that derive from them, in every format, under numpy 1 and
numpy 2, in test_data_module.saved_by_2_0_0.zip, and three pickles of an icon
set: a cube mesh, a curve and a patch.
"""

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from importlib import util

import numpy as np
import cgmath.hierarchy
from cgmath import geometry
from cgmath.data import (
    _Unpickler,
    Array,
    Data,
    DataList,
    ImmutableArray,
)
from cgmath.geometry.bspline import NurbsCurveData
from cgmath.geometry.bspline_patch import NurbsSurfaceData
from cgmath.geometry.mesh import MeshData, MeshList
from cgmath.hierarchy import HierarchyData, TransformData

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assets")
OLD    = "cgmath.geometry._base"


def _fresh(code: str) -> str:
    """
    What ``code`` prints in a new interpreter that sees this one's sys.path,
    and imports this cgmath: python -c puts its working folder first, so it
    runs in the folder that holds this cgmath, whatever folder the suite runs in.
    """
    init   = os.path.abspath(cgmath.__file__)
    holder = os.path.dirname(os.path.dirname(init))
    env    = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in sys.path if x))
    check  = f"\nimport cgmath, os\nassert os.path.samefile(cgmath.__file__, {init!r}), cgmath.__file__\n"
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code) + check],
        cwd            = holder,
        capture_output = True,
        text           = True,
        env            = env,
        timeout        = 300,
    )
    if out.returncode != 0:
        raise AssertionError(out.stderr[-2000:])
    return out.stdout


class _Names(pickle.Unpickler):
    """loads a pickle and keeps every module it names"""

    def __init__(self, f):
        super().__init__(f)
        self.modules = set()

    def find_class(self, module, name):
        self.modules.add(module)
        return super().find_class(module, name)


def _modules_named(path: str) -> set:
    with open(path, "rb") as f:
        names = _Names(f)
        names.load()
    return names.modules


def _saved_by_2_0_0() -> dict:
    """{file stem: the object 2.0.0 saved under it}, built here"""
    mesh = MeshData(
        points  = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64),
        counts  = np.array([3], np.int32),
        indices = np.array([0, 1, 2], np.int32),
        name    = "tri",
    )
    root  = "11111111-1111-1111-1111-111111111111"
    child = "22222222-2222-2222-2222-222222222222"
    nodes = [
        TransformData(name="root", uuid=root),
        TransformData(name="child", uuid=child, parent_uuid=root),
    ]
    return {
        "bare_data":       Data(),
        "bare_datalist":   DataList([mesh, Data()]),
        "empty_datalist":  DataList(),
        "plain_datalist":  DataList([Data(), Data()]),
        "mesh":            mesh,
        "meshlist":        MeshList([mesh]),
        "hierarchy":       HierarchyData(nodes),
        "immutable_array": ImmutableArray(np.arange(4.0)),
        "array_spec":      Array(np.float64, "N", 3),
        "legacy_error":    None,  # a class the module no longer has: its pickle is refused
    }


# the classes of the module that are no data: a pickle of one is no file a
# loader gives back
NOT_DATA = ("immutable_array", "array_spec", "legacy_error")

# files of the base classes themselves, which build nothing: a file names no
# class, so the class wanted loads it
BARE = ("bare_data", "bare_datalist", "empty_datalist", "plain_datalist")


class TestTheModule(unittest.TestCase):
    def test_import_loads_nothing_of_geometry(self):
        out = _fresh(
            """
            import sys

            before = set(sys.modules)
            import cgmath.data
            added = sorted(set(sys.modules) - before)
            print([x for x in added if x.startswith("cgmath")])
            print([x for x in added if x.split(".")[0] in ("numba", "scipy")])
            """
        )
        loaded, heavy = out.splitlines()[-2:]
        self.assertNotIn("cgmath.geometry", loaded)
        self.assertIn("'cgmath.data'", loaded)
        self.assertEqual(heavy, "[]")

    def test_constraints_loads_nothing_of_geometry(self):
        # procrustes.py takes Data from cgmath.data, not through cgmath.geometry
        out = _fresh(
            """
            import sys

            before = set(sys.modules)
            import cgmath.constraints
            added = sorted(set(sys.modules) - before)
            print([x for x in added if x.startswith("cgmath.geometry")])
            print([x for x in added if x.split(".")[0] in ("numba", "scipy")])
            """
        )
        self.assertEqual(out.splitlines()[-2:], ["[]", "[]"])

    def test_the_names_live_in_cgmath_data(self):
        self.assertEqual(Data.__module__, "cgmath.data")
        self.assertEqual(DataList.__module__, "cgmath.data")
        # cgmath.geometry exports them still, the same objects
        self.assertIs(geometry.Data,           Data)
        self.assertIs(geometry.DataList,       DataList)
        self.assertIs(geometry.ImmutableArray, ImmutableArray)
        # no module is left behind under the old name
        self.assertIsNone(util.find_spec(OLD))

    def test_a_save_names_no_class_and_no_module(self):
        saved = _saved_by_2_0_0()
        self.assertNotIn("__class__", saved["meshlist"].to_dict())
        self.assertNotIn("__class__", saved["meshlist"].to_dict()["__items__"][0])

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for stem in ("mesh", "meshlist", "hierarchy"):
            with self.subTest(stem):
                path = saved[stem].save(os.path.join(tmp, stem + ".pkl"))
                # plain data: no global at all, no cgmath name in the bytes
                self.assertEqual(_modules_named(path), set())
                with open(path, "rb") as f:
                    self.assertNotIn(b"cgmath", f.read())
                # a bare pickle.load reads the tree; the class builds it
                with open(path, "rb") as f:
                    self.assertIsInstance(pickle.load(f), dict)
                self.assertEqual(type(saved[stem]).load(path), saved[stem])


class TestFilesSavedBy2(unittest.TestCase):
    """
    The 31 files 2.0.0 saved, 21 by Data.save and 10 by a bare pickle.dump,
    under numpy 1 (mayapy, numpy 1.24) and numpy 2 (python 3.11, numpy 2.4):
    their npz files differ, and so does the ndarray they pickled.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(os.path.join(ASSETS, "test_data_module.saved_by_2_0_0.zip")) as z:
            z.extractall(cls.tmp)
        cls.sets     = sorted(os.listdir(cls.tmp))
        cls.files    = [f"{x}/{name}" for x in cls.sets for name in sorted(os.listdir(os.path.join(cls.tmp, x)))]
        cls.expected = _saved_by_2_0_0()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)

    def stem(self, name: str) -> str:
        """the key of _saved_by_2_0_0() the file holds"""
        stem = os.path.splitext(os.path.basename(name))[0]
        return stem[4:] if stem.startswith("raw_") else stem

    def numpy_reads(self, name: str) -> bool:
        """whether this numpy reads the file: numpy 1 has no numpy._core, which numpy 2's ndarray pickle names"""
        return not (name == "numpy2/raw_immutable_array.pickle" and np.lib.NumpyVersion(np.__version__) < "2.0.0")

    def assertSame(self, got, want, label):
        if isinstance(want, np.ndarray):
            self.assertIs(type(got), type(want), label)
            np.testing.assert_array_equal(got, want, label)
        elif isinstance(want, BaseException):
            self.assertIs(type(got), type(want), label)
            self.assertEqual(got.args, want.args, label)
        elif hasattr(want, "__metadata__"):  # Array(): an Annotated type, equal by its spec
            self.assertEqual(repr(got), repr(want), label)
        else:
            self.assertIs(type(got), type(want), label)
            self.assertEqual(got, want, label)

    def test_the_sets(self):
        self.assertEqual(self.sets, ["numpy1", "numpy2"])
        for folder in self.sets:
            with self.subTest(folder):
                names = [x for x in self.files if x.startswith(folder + "/")]
                self.assertEqual(len(names), 31)
                self.assertEqual(sorted({self.stem(x) for x in names}), sorted(self.expected))
                pickles = [x for x in names if x.endswith((".pkl", ".pickle"))]
                self.assertEqual(len(pickles), 17)
                # every pickle names the old module, and so do the plain classes' tags
                for name in pickles:
                    with open(self.path(name), "rb") as f:
                        self.assertIn(OLD.encode(), f.read(), name)
                for name in (f"{folder}/bare_data.json", f"{folder}/plain_datalist.json"):
                    with open(self.path(name), encoding="utf-8") as f:
                        self.assertIn(f'"{OLD}.', f.read(), name)

    def test_every_file_loads(self):
        for name in self.files:
            stem = self.stem(name)
            if stem in NOT_DATA or stem in BARE:
                continue
            want = self.expected[stem]
            with self.subTest(name):
                # the class's own loader: the file's class, named under the
                # old module, is skipped
                self.assertSame(type(want).load(self.path(name)), want, name)
                if name.endswith((".pkl", ".pickle")):
                    self.assertSame(type(want).load_pickle(self.path(name)), want, name)

    def test_the_base_classes_build_nothing(self):
        for folder in self.sets:
            for stem in BARE:
                for ext in (".pkl", ".json", ".npz"):
                    name   = f"{folder}/{stem}{ext}"
                    loader = DataList if "list" in stem else Data
                    with self.subTest(name):
                        with self.assertRaisesRegex(TypeError, r"builds nothing: a file names no class"):
                            loader.load(self.path(name))

    def test_a_hierarchy_loads_through_its_module(self):
        for folder in self.sets:
            for ext in (".pkl", ".json", ".npz"):
                name = f"{folder}/hierarchy{ext}"
                with self.subTest(name):
                    got = cgmath.hierarchy.load(self.path(name))
                    self.assertSame(got, self.expected["hierarchy"], name)

    def test_the_other_classes_pickles_are_refused(self):
        # an ImmutableArray, an ArraySpec, an error: no data, and no name a
        # cgmath pickle may hold
        for folder in self.sets:
            for stem in NOT_DATA:
                name = f"{folder}/raw_{stem}.pickle"
                with self.subTest(name):
                    with open(self.path(name), "rb") as f:
                        with self.assertRaisesRegex(pickle.UnpicklingError, "plain data"):
                            _Unpickler(f).load()

    def test_a_bare_pickle_load_fails(self):
        for name in self.files:
            if not name.endswith((".pkl", ".pickle")) or not self.numpy_reads(name):
                continue
            with self.subTest(name):
                with open(self.path(name), "rb") as f:
                    with self.assertRaises(ModuleNotFoundError) as caught:
                        pickle.load(f)
                self.assertEqual(caught.exception.name, OLD)
                self.assertEqual(str(caught.exception), f"No module named '{OLD}'")

    def test_a_pickle_saved_now_needs_no_cgmath_to_read_its_tree(self):
        path = self.expected["mesh"].save(os.path.join(self.tmp, "mesh_now.pkl"))
        out = _fresh(
            f"""
            import pickle, sys

            with open({path!r}, "rb") as f:
                tree = pickle.load(f)
            print(sorted(tree))
            print(sorted(tree["points"]))
            print([x for x in sys.modules if x.startswith("cgmath")])
            """
        )
        keys, array, modules = out.splitlines()[-3:]
        self.assertEqual(keys,    "['counts', 'indices', 'name', 'points', 'uuid']")
        self.assertEqual(array,   "['__array__']")  # an array: its dtype, shape and bytes
        self.assertEqual(modules, "[]")


class TestOldPickles(unittest.TestCase):
    """pickles an older cgmath saved for an icon set, and a rig the suite keeps"""

    def path(self, name: str) -> str:
        return os.path.join(ASSETS, name)

    def test_each_loads(self):
        mesh = MeshData.load(self.path("test_data_module.mesh.pkl"))
        self.assertIs(type(mesh), MeshData)
        self.assertEqual(mesh.name, "pCubeShape1")
        self.assertEqual((mesh.point_count, mesh.counts.tolist()), (8, [4] * 6))
        np.testing.assert_allclose(np.abs(mesh.points), 0.5)

        curve = NurbsCurveData.load(self.path("test_data_module.curve.pkl"))
        self.assertIs(type(curve), NurbsCurveData)
        self.assertEqual((curve.points.shape, curve.degree, curve.periodic), ((8, 3), 3, True))

        patch = NurbsSurfaceData.load(self.path("test_data_module.patch.pkl"))
        self.assertIs(type(patch), NurbsSurfaceData)
        self.assertEqual(patch.points.shape, (4, 8, 3))
        self.assertEqual((patch.degree_u, patch.degree_v), (3, 3))
        self.assertEqual((patch.periodic_u, patch.periodic_v), (False, True))

        # the knots the curve and the surface gained since load at their defaults
        self.assertIsNone(curve.knots)
        self.assertIsNone(patch.knots_u)
        self.assertIsNone(patch.knots_v)

        rig = HierarchyData.load(self.path("test_hierarchy.legacy_parent_node.pkl"))
        self.assertIs(type(rig), HierarchyData)
        self.assertEqual(len(rig), 12)
        self.assertEqual(cgmath.hierarchy.load(self.path("test_hierarchy.legacy_parent_node.pkl")), rig)

        # through each class's own loaders too
        for obj, name in (
            (mesh, "test_data_module.mesh.pkl"),
            (curve, "test_data_module.curve.pkl"),
            (patch, "test_data_module.patch.pkl"),
            (rig, "test_hierarchy.legacy_parent_node.pkl"),
        ):
            with self.subTest(name):
                self.assertEqual(type(obj).load(self.path(name)), obj)
                self.assertEqual(type(obj).load_pickle(self.path(name)), obj)
                with open(self.path(name), "rb") as f:
                    self.assertIn(OLD.encode(), f.read())

    def test_saved_again_a_bare_pickle_load_reads_its_tree(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for cls, name in (
            (MeshData, "test_data_module.mesh.pkl"),
            (NurbsCurveData, "test_data_module.curve.pkl"),
            (NurbsSurfaceData, "test_data_module.patch.pkl"),
            (HierarchyData, "test_hierarchy.legacy_parent_node.pkl"),
        ):
            with self.subTest(name):
                with open(self.path(name), "rb") as f:
                    with self.assertRaises(ModuleNotFoundError) as caught:
                        pickle.load(f)
                self.assertEqual(caught.exception.name, OLD)

                obj  = cls.load(self.path(name))
                path = obj.save(os.path.join(tmp, name))
                self.assertEqual(_modules_named(path), set())  # plain data names nothing
                with open(path, "rb") as f:
                    self.assertIsInstance(pickle.load(f), dict)
                self.assertEqual(cls.load(path), obj)


if __name__ == "__main__":
    unittest.main()
