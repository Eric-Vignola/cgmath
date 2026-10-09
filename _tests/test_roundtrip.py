"""Every cgmath data class saves and loads back exactly, in pkl, npz and json.

Each class is built filled in, by a builder in one of the ``_examples_*``
modules beside this file, saved in each format, loaded back, and compared
strictly: the same class, every field, every array's dtype and shape and
values, list against tuple, None against empty. A class with no builder fails
the test, so a new class cannot slip past it. A file names each field as a
caller sets it, and still loads when it names one by its private field.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import os
import pickle
import pkgutil
import tempfile
import time
import unittest
import zipfile
from dataclasses import dataclass, fields

import numpy as np
import cgmath
from cgmath.geometry._base import (
    _CLASSES,
    _json_tree,
    _pickled,
    _unpickle,
    CLASS_KEY,
    Data,
    DataList,
    dict_to_zip,
    DTYPES_KEY,
    ID_KEY,
    ITEMS_KEY,
    SHAPES_KEY,
)
from cgmath.geometry.sdf import SDFCylinder, SDFSphere
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, generate_uuid
from cgmath.render.scene import Object
from cgmath.utils import pretty_json

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


def _private_spelling(value):
    """
    A tree as cgmath wrote it before a file named fields as their
    properties: every field, setting and clip channel under its own name,
    ``_translate``, ``_blocks``.
    """
    if isinstance(value, (list, tuple)):
        return type(value)(_private_spelling(x) for x in value)
    if not isinstance(value, dict):
        return value

    tag    = value.get(CLASS_KEY)
    cls    = _CLASSES.get(tag) if isinstance(tag, str) else None
    fields = {} if cls is None else cls._saved_names()[1]
    out    = {}
    for key, item in value.items():
        if key in (DTYPES_KEY, SHAPES_KEY):
            item = {fields.get(name, name): x for name, x in item.items()}
        out[fields.get(key, key)] = _private_spelling(item)

    if cls is not None and "_blocks" in out and hasattr(cls, "FRAMED_CHANNELS"):
        channels       = cls.DATA_LIST_CLASS._saved_names()[1]
        out["_blocks"] = {channels.get(name, name): x for name, x in out["_blocks"].items()}
    return out


class _SavedTree:
    """pickles as a cgmath pkl file holding ``tree``, the way Data.__reduce__ writes one"""

    def __init__(self, cls, tree: dict):
        self.cls  = cls
        self.tree = tree

    def __reduce__(self):
        return (_unpickle, (self.cls, _pickled(self.tree)))


class _Labelled(DataList):
    """a list of a caller's own, its one setting private behind a property"""

    LIST_FIELDS = ("_label",)

    _label = ""

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value


@dataclass(repr=False, eq=False)
class _SavesAMissingField(Data):
    """SAVED_NAMES naming a field the class does not have"""

    _width: float = 1.0

    SAVED_NAMES = {"_height": "height"}


@dataclass(repr=False, eq=False)
class _SavesOverAField(Data):
    """SAVED_NAMES saving a field under another field's name"""

    _width: float = 1.0
    height: float = 1.0

    SAVED_NAMES = {"_width": "height"}


@dataclass(repr=False, eq=False)
class _SavesTwoAsOne(Data):
    """SAVED_NAMES saving two fields under one name"""

    _width:  float = 1.0
    _height: float = 1.0

    SAVED_NAMES = {"_width": "size", "_height": "size"}


class _SavesAMissingSetting(DataList):
    """SAVED_NAMES naming a setting the list does not save"""

    LIST_FIELDS = ("_label",)
    SAVED_NAMES = {"_name": "name"}

    _label = ""


@dataclass(repr=False, eq=False)
class _KeepsItsName(Data):
    """SAVED_NAMES keeping a field's own name: the property of its public name does not read it"""

    _count: int = 0

    SAVED_NAMES = {"_count": "_count"}

    @property
    def count(self) -> int:
        return 42


class TestPublicSavedNames(unittest.TestCase):
    """
    A file names a field as a caller sets it: a private field behind a
    property of its own name is saved as the property, TransformData's
    _translate as "translate", and so are a ClipData's settings, its blocks
    ("blocks") and their channels. A load takes either spelling, so a file
    that names the private field still loads, and refuses a tree that names
    one field both ways.
    """

    # the classes holding such fields, and names each example saves publicly
    PUBLIC = {
        "TransformData":  ("translate", "rotate_order", "joint_orient", "radius"),
        "Camera":         ("translate", "rotate"),
        "Light":          ("translate", "visibility"),
        "Object":         ("mesh", "uv", "texture", "skin", "base_color", "wireframe"),
        "DeltaMushData":  ("smooth_iterations", "method", "weight"),
        "FFDData":        ("outside", "local_influence", "cells", "uvw"),
        "PatchRelaxData": ("iterations", "alpha", "surface_blend"),
        "UVData":         ("resolution", "antialias"),
        "SkinData":       ("patterns",),
        "ProcrustesData": ("scale_offset",),
        "ClipData":       ("start_frame", "fps", "frame", "blocks"),
    }

    @classmethod
    def setUpClass(cls):
        classes, _ = data_classes()
        builders, _ = examples()
        cls.builders = {
            name: builders[x] for name, x in classes.items() if x in builders
        }
        cls.listed = {
            x.__name__: (x, builders[x])
            for x in classes.values()
            if x.__name__ in cls.PUBLIC and x in builders
        }

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def private_keys(self, tree, path="tree"):
        """every key of ``tree``, nested trees too, that names a field by its private name"""
        found = []
        if isinstance(tree, (list, tuple)):
            for index, item in enumerate(tree):
                found += self.private_keys(item, f"{path}[{index}]")
            return found
        if not isinstance(tree, dict):
            return found

        tag = tree.get(CLASS_KEY)
        cls = _CLASSES.get(tag) if isinstance(tag, str) else None
        for key, item in tree.items():
            keys = [key]
            if key in (DTYPES_KEY, SHAPES_KEY):
                keys = list(item)
            for name in keys:
                public = name[1:]
                if (
                    cls is not None
                    and name.startswith("_")
                    and not name.startswith("__")
                    and isinstance(getattr(cls, public, None), property)
                    # a field the property does not read: an SDF primitive's
                    # joint radius, where "radius" is the primitive's own
                    and cls._saved_names()[1].get(public, name) == name
                ):
                    found.append(f"{path}.{name}")
            if key not in (DTYPES_KEY, SHAPES_KEY):
                found += self.private_keys(item, f"{path}.{key}")

        if cls is not None and isinstance(tree.get("blocks"), dict):
            found += [f"{path}.blocks.{x}" for x in tree["blocks"] if x.startswith("_")]
        return found

    def test_no_saved_name_is_a_private_field_behind_its_property(self):
        for name, builder in sorted(self.builders.items()):
            with self.subTest(cls=name):
                self.assertEqual(self.private_keys(builder().to_dict()), [])

    def test_the_fields_behind_properties_are_saved_as_the_properties(self):
        for name, (cls, builder) in sorted(self.listed.items()):
            with self.subTest(cls=name):
                tree = builder().to_dict()
                for public in self.PUBLIC[name]:
                    self.assertIn(public, tree)
                    self.assertNotIn("_" + public, tree)
        self.assertEqual(sorted(self.listed), sorted(self.PUBLIC))

        clip = self.listed["ClipData"][1]()
        self.assertEqual(
            list(clip.to_dict()["blocks"]),
            ["scale", "rotate", "translate", "rotate_axis", "joint_orient"],
        )

    def test_every_format_saves_the_public_names_and_loads_the_same_object(self):
        for name, (cls, builder) in sorted(self.listed.items()):
            obj    = builder()
            public = self.PUBLIC[name][0]
            for ext in FORMATS:
                with self.subTest(cls=name, ext=ext):
                    path = obj.save(os.path.join(self.tmp, f"{name}.{ext}"))
                    if ext == "npz":
                        with zipfile.ZipFile(path) as z:
                            members = {x.split("/")[0].removesuffix(".npy") for x in z.namelist()}
                        self.assertIn(public, members)
                        self.assertNotIn("_" + public, members)
                    elif ext == "json":
                        with open(path, encoding="utf-8") as f:
                            saved = json.load(f)
                        self.assertIn(public, saved)
                        self.assertNotIn("_" + public, saved)
                    assert_same(self, obj, cls.load(path), f"{name}.{ext}")

    def write_old_file(self, cls, tree: dict, path: str) -> None:
        """``tree`` as a file, the way cgmath writes one in the format ``path`` names"""
        ext = os.path.splitext(path)[1]
        if ext == ".npz":
            dict_to_zip(tree, path)
        elif ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(pretty_json(_json_tree(tree)))
        else:
            with open(path, "wb") as f:
                pickle.dump(_SavedTree(cls, tree), f)

    def test_a_file_naming_the_private_fields_still_loads(self):
        # as cgmath wrote every file before: "_translate", "_blocks"
        for name, (cls, builder) in sorted(self.listed.items()):
            obj = builder()
            old = _private_spelling(obj.to_dict())
            self.assertNotEqual(self.private_keys(old), [])
            self.assertIn("_" + self.PUBLIC[name][0], old)

            with self.subTest(cls=name, file="tree"):
                assert_same(self, obj, cls.from_dict(old), f"{name} from its tree")
            for ext in FORMATS:
                with self.subTest(cls=name, file=ext):
                    path = os.path.join(self.tmp, f"{name}_old.{ext}")
                    self.write_old_file(cls, old, path)
                    assert_same(self, obj, cls.load(path), f"{name} from {ext}")

    @staticmethod
    def flat_object() -> Object:
        """an Object whose texture is an empty (0, 4, 3) array, which json writes as []"""
        obj         = Object(name="flat")
        obj.texture = np.zeros((0, 4, 3), np.float32)
        return obj

    def test_an_empty_array_notes_its_shape_under_the_saved_name(self):
        # json keeps no shape for an empty array: the __shapes__ note does, keyed
        # as the field is saved, and a note keyed by the private field still loads
        obj  = self.flat_object()
        tree = obj.to_dict()
        self.assertEqual(tree[SHAPES_KEY], {"texture": [0, 4, 3]})

        old = _private_spelling(tree)
        self.assertEqual(old[SHAPES_KEY], {"_texture": [0, 4, 3]})

        for ext in FORMATS:
            with self.subTest(ext=ext):
                path = obj.save(os.path.join(self.tmp, f"flat.{ext}"))
                assert_same(self, obj, Object.load(path), f"flat.{ext}")
            with self.subTest(ext=ext, file="old"):
                path = os.path.join(self.tmp, f"flat_old.{ext}")
                self.write_old_file(Object, old, path)
                assert_same(self, obj, Object.load(path), f"flat_old.{ext}")

    def test_a_tree_naming_a_field_both_ways_is_refused(self):
        for name, (cls, builder) in sorted(self.listed.items()):
            tree   = builder().to_dict()
            public = self.PUBLIC[name][0]
            with self.subTest(cls=name):
                both               = dict(tree)
                both["_" + public] = tree[public]
                with self.assertRaisesRegex(ValueError, rf"^{name}\.{public} is given twice"):
                    cls.from_dict(both)

            if DTYPES_KEY in tree:
                with self.subTest(cls=name, part=DTYPES_KEY):
                    note             = dict(tree[DTYPES_KEY])
                    key              = next(iter(note))
                    note["_" + key]  = note[key]
                    both             = dict(tree)
                    both[DTYPES_KEY] = note
                    with self.assertRaisesRegex(ValueError, rf"^{name}\.{key} is given twice"):
                        cls.from_dict(both)

        # and an empty array's __shapes__ note
        tree             = self.flat_object().to_dict()
        tree[SHAPES_KEY] = {**tree[SHAPES_KEY], "_texture": [0, 4, 3]}
        with self.subTest(cls="Object", part=SHAPES_KEY):
            with self.assertRaisesRegex(ValueError, r"^Object\.texture is given twice"):
                Object.from_dict(tree)

        # and a clip's channel, named both ways in its blocks
        cls, builder = self.listed["ClipData"]
        tree                      = builder().to_dict()
        tree["blocks"]            = dict(tree["blocks"])
        tree["blocks"]["_rotate"] = tree["blocks"]["rotate"]
        with self.assertRaisesRegex(
            ValueError, r"^TransformData\.rotate is given twice in the blocks"
        ):
            cls.from_dict(tree)

    def test_a_clip_reads_its_blocks_and_channels_either_way(self):
        cls, builder = self.listed["ClipData"]
        clip     = builder()
        tree     = clip.to_dict()
        channels = {"_" + key: block for key, block in tree["blocks"].items()}

        # the settings, the blocks and their channels each named by field
        mixed = {
            "channels by field": {**tree, "blocks": channels},
            "blocks by field": {
                **{key: x for key, x in tree.items() if key != "blocks"},
                "_blocks": tree["blocks"],
            },
            "settings by field": {
                ("_" + key if key in ("start_frame", "fps", "frame") else key): x
                for key, x in tree.items()
            },
        }
        for how, data in mixed.items():
            with self.subTest(how):
                assert_same(self, clip, cls.from_dict(data), how)

        # a channel the clip does not frame is named as the file names it
        for channel in ("radius", "bogus"):
            with self.subTest(channel=channel):
                data = {**tree, "blocks": {**tree["blocks"], channel: tree["blocks"]["scale"]}}
                with self.assertRaisesRegex(
                    ValueError, rf"^ClipData has no framed channel '{channel}'$"
                ):
                    cls.from_dict(data)

    def test_each_list_class_keeps_its_own_saved_names(self):
        # a ClipData is a HierarchyData, which saves no setting by another
        # name: each class builds its own map, whichever of them saves first
        for cls in (HierarchyData, ClipData):
            cached = cls.__dict__.get("_SAVED_NAMES_CACHE")
            if cached is not None:
                delattr(cls, "_SAVED_NAMES_CACHE")
                self.addCleanup(setattr, cls, "_SAVED_NAMES_CACHE", cached)

        self.assertNotIn("_SAVED_NAMES_CACHE", ClipData.__dict__)
        HierarchyData().to_dict()
        tree = self.listed["ClipData"][1]().to_dict()
        self.assertLessEqual({"start_frame", "fps", "frame", "blocks"}, set(tree))
        self.assertEqual({"_start_frame", "_fps", "_frame", "_blocks"} & set(tree), set())

    def test_a_tree_built_by_hand_sets_a_property_through_its_setter(self):
        # a tree that names no class is built by hand, as from_dict takes one
        # and the asset readers build their nodes: a property's name in it
        # goes through the property, which checks and converts the value
        classes = {name: cls for name, (cls, _) in self.listed.items()}
        mush    = classes["DeltaMushData"]
        self.assertEqual(mush.from_dict({"weight": 2.0}).weight, 1.0)
        with self.assertRaisesRegex(ValueError, "smooth_iterations must be >= 0"):
            mush.from_dict({"smooth_iterations": -1})
        with self.assertRaisesRegex(ValueError, "Unknown outside mode"):
            classes["FFDData"].from_dict({"outside": "bogus"})
        self.assertEqual(classes["UVData"].from_dict({"resolution": 64}).resolution.tolist(), [64, 64])
        self.assertEqual(
            Object.from_dict({"background": (0.1, 0.2, 0.3)}).background.tolist(),
            [0.1, 0.2, 0.3, 1.0],
        )

        # a node of the dict of node dicts HierarchyData.from_dict takes
        root = generate_uuid()
        node = HierarchyData.from_dict(
            {root: {"name": "root", "uuid": root, "visibility": 0, "rotate_order": np.int64(2)}}
        )["root"]
        self.assertIs(node.visibility, False)
        self.assertIs(type(node.rotate_order), int)
        self.assertEqual(json.loads(json.dumps(node.to_attributes()))["rotateOrder"], 2)

        # a property with no setter, FFDData's cells and uvw, and a list's
        # settings, which have none, are read as the fields: a saved tree
        # with no "__class__" loads the same object
        for name in ("FFDData", "ClipData"):
            cls, builder = self.listed[name]
            obj  = builder()
            tree = {key: x for key, x in obj.to_dict().items() if key != CLASS_KEY}
            with self.subTest(cls=name, how="by hand"):
                assert_same(self, obj, cls.from_dict(tree), f"{name} with no class")

        # and neither a tree built by hand nor a saved one names a field both ways
        for how, tree in (("by hand", {}), ("saved", TransformData("a").to_dict())):
            with self.subTest(how):
                with self.assertRaisesRegex(
                    ValueError, r"^TransformData\.translate is given twice in the data"
                ):
                    TransformData.from_dict({**tree, "translate": [1, 2, 3], "_translate": [1, 2, 3]})

        # nor one built by hand a read-only property's field, or a list's
        # setting, both ways
        ffd  = self.listed["FFDData"][1]().to_dict()
        clip = self.listed["ClipData"][1]().to_dict()
        cases = (
            ("FFDData", "cells", {key: x for key, x in ffd.items() if key != CLASS_KEY}),
            ("ClipData", "fps", {ITEMS_KEY: clip[ITEMS_KEY], "fps": clip["fps"]}),
        )
        for name, public, tree in cases:
            with self.subTest(cls=name, how="by hand"):
                with self.assertRaisesRegex(ValueError, rf"^{name}\.{public} is given twice in the data"):
                    self.listed[name][0].from_dict({**tree, "_" + public: tree[public]})

    def test_to_attributes_names_a_field_either_way(self):
        rig  = self.builders["cgmath.hierarchy.hierarchy.HierarchyData"]()
        clip = self.listed["ClipData"][1]()
        node = rig[1]
        owners = {
            "node":   node,
            "rig":    rig,
            "view":   rig[1:],
            "clip":   clip,
            "frames": clip.frames,
        }
        for how, owner in owners.items():
            with self.subTest(how):
                public  = owner.to_attributes({"translate": "t", "rotate_order": "ro"})
                private = owner.to_attributes({"_translate": "t", "_rotate_order": "ro"})
                self.assertEqual(public, private)
                with self.assertRaisesRegex(
                    ValueError, r"^TransformData\.translate is given twice in the mapping"
                ):
                    owner.to_attributes({"translate": "t", "_translate": "T"})

        self.assertEqual(
            node.to_attributes({"translate": "t", "rotate_order": "ro"}),
            {"t": node.translate.tolist(), "ro": node.rotate_order},
        )

    def test_to_attributes_reads_an_sdf_radius_as_each_node_does(self):
        # on an SDF primitive "radius" is its own radius and "_radius" the
        # joint radius: a clip's frames read a mapping as its nodes do. Among
        # other nodes, on which both name the joint radius, "_sdf_radius"
        # names the primitive's
        def rig(*nodes):
            hierarchy = HierarchyData()
            for node in nodes:
                hierarchy.append(node)
            return hierarchy

        def sdf(cls, name, radius, joint):
            node         = cls(name=name)
            node.radius  = radius
            node._radius = joint
            return node

        cases = (
            (
                "primitives",
                rig(sdf(SDFSphere, "a", 2.0, 5.0), sdf(SDFCylinder, "b", 3.0, 1.0)),
                {"_radius": "joint", "radius": "sdf"},
                [{"joint": 5.0, "sdf": 2.0}, {"joint": 1.0, "sdf": 3.0}],
            ),
            (
                "mixed",
                rig(TransformData(name="root", radius=4.0), sdf(SDFSphere, "a", 2.0, 5.0)),
                {"_radius": "joint", "_sdf_radius": "sdf"},
                [{"joint": 4.0}, {"joint": 5.0, "sdf": 2.0}],
            ),
        )
        for how, nodes, mapping, expected in cases:
            clip   = ClipData(nodes, frames=2)
            owners = {"rig": nodes, "clip": clip, "frames": clip.frames}
            for owner, value in owners.items():
                with self.subTest(how, owner=owner):
                    self.assertEqual(value.to_attributes(mapping), expected)

    def test_an_sdf_primitive_saves_its_radius_as_radius(self):
        # its radius is _sdf_radius, behind the radius property, and is saved
        # as "radius"; the joint radius, _radius, which no property of the
        # primitive reads, keeps its own name, and to_attributes reads both so
        for cls in (SDFSphere, SDFCylinder):
            obj         = cls(radius=0.75)
            obj._radius = 2.0
            tree        = obj.to_dict()
            old         = {"_sdf_radius" if key == "radius" else key: x for key, x in tree.items()}
            with self.subTest(cls=cls.__name__):
                self.assertEqual((tree["radius"], tree["_radius"]), (0.75, 2.0))
                self.assertNotIn("_sdf_radius", tree)
                for how, data in (("saved", tree), ("private", old)):
                    back = cls.from_dict(data)
                    self.assertEqual((back.radius, back._radius), (0.75, 2.0), how)
                for ext in FORMATS:
                    path = obj.save(os.path.join(self.tmp, f"{cls.__name__}.{ext}"))
                    assert_same(self, obj, cls.load(path), f"{cls.__name__}.{ext}")

                self.assertEqual(obj.to_attributes({"radius": "r"}),      {"r": 0.75})
                self.assertEqual(obj.to_attributes({"_sdf_radius": "r"}), {"r": 0.75})
                self.assertEqual(obj.to_attributes({"_radius": "r"}),     {"r": 2.0})

    def test_a_list_of_ones_own_saves_its_settings_as_their_properties(self):
        # the rule every DataList's LIST_FIELDS follow, not ClipData's alone
        labelled       = _Labelled()
        labelled.label = "mouth"
        tree           = labelled.to_dict()
        old            = {"_label" if key == "label" else key: x for key, x in tree.items()}
        self.assertEqual(sorted(tree), [CLASS_KEY, ITEMS_KEY, "label"])
        self.assertEqual(sorted(old), [CLASS_KEY, ITEMS_KEY, "_label"])

        for how, data in (("saved", tree), ("private", old)):
            with self.subTest(how):
                assert_same(self, labelled, _Labelled.from_dict(data), how)
        for ext in FORMATS:
            with self.subTest(ext=ext):
                path = labelled.save(os.path.join(self.tmp, f"labelled.{ext}"))
                assert_same(self, labelled, _Labelled.load(path), f"labelled.{ext}")

        with self.assertRaisesRegex(
            ValueError, r"^_Labelled\.label is given twice in the data, as 'label' and '_label'$"
        ):
            _Labelled.from_dict({**tree, "_label": "lips"})

    def test_saved_names_must_name_a_field_and_a_name_no_other_has(self):
        # checked on a class's first save: a bad map would save two fields
        # under one name, one over the other, or name a field there is not
        cases = (
            (_SavesAMissingField, "_height", "height"),
            (_SavesOverAField, "_width", "height"),
            (_SavesTwoAsOne, "_height", "size"),
            (_SavesAMissingSetting, "_name", "name"),
        )
        for cls, name, saved in cases:
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(
                    TypeError, rf"^{cls.__name__}\.SAVED_NAMES cannot save '{name}' as '{saved}'$"
                ):
                    cls().to_dict()

        # a field's own name is no other's: mapped to itself, the field keeps
        # it, beside a property of its public name that does not read it
        obj  = _KeepsItsName(_count=3)
        tree = obj.to_dict()
        self.assertEqual(sorted(tree), [CLASS_KEY, "_count"])
        self.assertEqual(_KeepsItsName.from_dict(tree)._count, 3)
        self.assertEqual(_KeepsItsName._field_label("_count"), "_KeepsItsName._count")
        for ext in FORMATS:
            with self.subTest(cls="_KeepsItsName", ext=ext):
                path = obj.save(os.path.join(self.tmp, f"kept.{ext}"))
                assert_same(self, obj, _KeepsItsName.load(path), f"kept.{ext}")


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
