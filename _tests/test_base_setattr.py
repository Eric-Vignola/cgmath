"""Tests for the strict ``Data.__setattr__`` / ``DataList.__setattr__`` guard.

An instance takes a write only to what its class declares: a dataclass
field (a list's ``list``, ``LIST_FIELDS`` and annotations), a property
with a setter, or private state a class up the MRO names in its
``_STATE``. Anything else raises, a typo like ``obj.ambiant = 0.5`` (of
``ambient``) and a private name alike, and so does a write over a method,
a class setting or a read-only property, which names it for what it is.
"""

import contextlib
import gc
import inspect
import itertools
import json
import os
import re
import sys
import tempfile
import tracemalloc
import unittest
from abc import ABCMeta
from collections import Counter
from dataclasses import dataclass, fields
from typing import Optional
from unittest import mock

import numpy as np
from cgmath.geometry import _base
from cgmath.geometry._base import Data, DataList
from cgmath.geometry.mesh import MeshData, MeshList
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath.hierarchy import hierarchy as hierarchy_mod
from cgmath._tests.test_roundtrip import data_classes, examples


# -- test fixtures: minimal Data subclasses ----------------------------


@dataclass(repr=False, eq=False)
class _Leaf(Data):
    """Bare ``Data`` subclass exercising:
    - a public dataclass field (``name``),
    - a private dataclass field (``_value``),
    - a ``@property`` / setter pair (``value``),
    - a read-only ``@property`` (``doubled``),
    - a private cache it declares (``_cache``).
    """

    name:   Optional[str] = None
    _value: int = 0

    _STATE = ("_cache",)

    @property
    def doubled(self) -> int:
        return 2 * self._value

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        self._value = int(v)


@dataclass(repr=False, eq=False)
class _Child(_Leaf):
    """Inherits from ``_Leaf`` to verify MRO-based attribute lookup, and
    declares a cache of its own beside the one ``_Leaf`` declares."""

    _extra: int = 0

    _STATE = ("_more",)

    @property
    def extra(self) -> int:
        return self._extra

    @extra.setter
    def extra(self, v: int) -> None:
        self._extra = int(v)


@dataclass(repr=False, eq=False)
class _Required(Data):
    """Subclass with REQUIRED fields (no defaults).  Mirrors the layout
    of real ``Data`` subclasses like ``MeshData``, where mandatory raw-
    data fields (``indices`` / ``counts`` / ``points``) are declared
    without defaults and therefore do NOT appear in ``vars(cls)``.

    The strict ``__setattr__`` guard must accept these names anyway --
    this fixture is the regression for the mayapy bug where
    ``MeshData(indices=...)`` raised AttributeError because ``indices``
    was missing from the class dict.
    """

    indices: np.ndarray
    counts:  np.ndarray
    points:  np.ndarray
    name:    Optional[str] = None  # one optional field for contrast


class _Clamped:
    """A data descriptor that is no property: it keeps ``_weight`` in [0, 1]."""

    def __get__(self, obj, owner=None):
        return self if obj is None else obj._weight

    def __set__(self, obj, value):
        obj._weight = min(max(float(value), 0.0), 1.0)


@dataclass(repr=False, eq=False)
class _Weighted(Data):
    """A field behind a descriptor of its own, not a property."""

    _weight: float = 0.0

    weight = _Clamped()


# -- tests -------------------------------------------------------------


class TestDataStrictSetattr(unittest.TestCase):
    """``Data.__setattr__`` rejects assignments to anything the class does
    not declare, private names included, while leaving normal usage
    unaffected."""

    # -- happy path: declared attrs work ------------------------------

    def test_dataclass_field_assignment_succeeds(self):
        obj      = _Leaf()
        obj.name = "leaf"
        self.assertEqual(obj.name, "leaf")

    def test_property_setter_is_invoked(self):
        obj       = _Leaf()
        obj.value = 5
        self.assertEqual(obj._value, 5)
        self.assertEqual(obj.value, 5)

    def test_property_setter_coercion_runs(self):
        # Confirms we route through the descriptor (which casts to int),
        # not just blindly stuffing the instance dict.
        obj       = _Leaf()
        obj.value = "7"
        self.assertEqual(obj._value, 7)
        self.assertIsInstance(obj._value, int)

    def test_declared_private_attribute_succeeds(self):
        obj        = _Leaf()
        obj._cache = "anything"
        self.assertEqual(obj._cache, "anything")

    def test_undeclared_private_attribute_raises(self):
        obj = _Leaf()
        with self.assertRaisesRegex(AttributeError, "'_Leaf' has no attribute '_scratch'.*_STATE"):
            obj._scratch = 1
        self.assertNotIn("_scratch", vars(obj))

    def test_state_merges_up_the_mro(self):
        # _Child takes its own _more and _Leaf's _cache; _Leaf takes no _more
        obj        = _Child()
        obj._cache = 1
        obj._more  = 2
        self.assertEqual((obj._cache, obj._more), (1, 2))
        with self.assertRaises(AttributeError):
            _Leaf()._more = 2

    def test_private_dataclass_field_succeeds(self):
        # ``_value`` is a private dataclass field on _Leaf, the storage
        # behind the ``value`` property
        obj        = _Leaf()
        obj._value = 99
        self.assertEqual(obj._value, 99)

    def test_inherited_attribute_succeeds(self):
        # ``name`` and ``value`` are declared on _Leaf; assigning them
        # on the _Child subclass must still succeed via MRO lookup.
        obj       = _Child()
        obj.name  = "child"
        obj.value = 10
        obj.extra = 3
        self.assertEqual(obj.name,  "child")
        self.assertEqual(obj.value, 10)
        self.assertEqual(obj.extra, 3)

    def test_class_setting_assignment_raises(self):
        # ``EQUALITY_TEST_IGNORE`` is a setting of the class: an instance
        # holds none of its own
        obj = _Leaf()
        with self.assertRaisesRegex(AttributeError, "'EQUALITY_TEST_IGNORE' is a class setting of _Leaf"):
            obj.EQUALITY_TEST_IGNORE = ["name"]
        self.assertNotIn("EQUALITY_TEST_IGNORE", vars(obj))
        with self.assertRaisesRegex(AttributeError, "'_FIELD_NAMES_CACHE' is a class attribute of _Leaf"):
            obj._FIELD_NAMES_CACHE = frozenset()

    def test_method_assignment_raises(self):
        obj = _Leaf()
        for name, what in (
            ("copy", "a method"),
            ("_rebuilt", "a method"),
            ("load", "a class method"),
            ("from_dict", "a class method"),
            ("_reconstruct", "a static method"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                AttributeError, f"^'{name}' is {what} of _Leaf, not a setting$"
            ):
                setattr(obj, name, 5)
            self.assertNotIn(name, vars(obj))

    def test_read_only_property_assignment_raises(self):
        obj = _Leaf()
        for name in ("doubled", "namespace"):
            with self.subTest(name=name), self.assertRaisesRegex(
                AttributeError, f"^'{name}' is a read-only property of _Leaf$"
            ):
                setattr(obj, name, 5)

    def test_dunder_assignment_raises(self):
        # __dict__, __class__ and __weakref__ are descriptors with a __set__
        # every class has from object: __dict__ = {...} would swap every
        # field for whatever it holds, __class__ = ... the instance's class
        obj    = _Leaf(name="kept")
        before = dict(vars(obj))
        for name, value in (("__dict__", {"_junk": 1}), ("__class__", _Child), ("__weakref__", None)):
            with self.subTest(name=name), self.assertRaisesRegex(
                AttributeError, f"^'{name}' is Python's, not an attribute an instance of _Leaf takes$"
            ):
                setattr(obj, name, value)
        self.assertIs(type(obj), _Leaf)
        self.assertEqual(vars(obj), before)
        self.assertFalse({"__dict__", "__class__", "__weakref__"} & _Leaf._writable())

    def test_a_descriptor_with_a_set_takes_a_write(self):
        # any data descriptor takes one, not only a property
        obj        = _Weighted()
        obj.weight = 2
        self.assertEqual((obj.weight, obj._weight), (1.0, 1.0))
        self.assertIn("weight", _Weighted._writable())

    def test_a_state_given_as_a_str_raises(self):
        # ("_cache") is a str, not a tuple: it would declare "_", "c", "a",
        # "h" and "e", and not "_cache"
        @dataclass(repr=False, eq=False)
        class Typo(Data):
            name: Optional[str] = None

            _STATE = ("_cache")

        with self.assertRaisesRegex(
            TypeError, r"^Typo\._STATE is a str, not a tuple of names: \('_cache',\)$"
        ):
            Typo()

        class TypoList(DataList):
            _STATE = "_cache"

        with self.assertRaisesRegex(TypeError, r"^TypoList\._STATE is a str"):
            TypoList()

    def test_a_tree_naming_private_state_raises(self):
        # a file holds fields and public properties: a cache, a link, a name
        # _STATE declares, one it does not, or a dunder is never loaded
        node = TransformData(name="a")
        mesh = MeshData(indices=np.arange(3), counts=np.array([3]), points=np.eye(3))
        for obj, key, value in (
            (node, "_listers", 5),
            (node, "_hierarchy", "rig"),
            (mesh, "_f2v", "not a cache"),
            (node, "_junk", 1),
            (mesh, "__dict__", {"junk": 1}),
        ):
            what      = type(obj).__name__
            tree      = obj.to_dict()
            tree[key] = value
            with self.subTest(cls=what, key=key), self.assertRaisesRegex(
                AttributeError,
                f"^the data names '{key}', which is no field of {what}: a file holds no private state$",
            ):
                type(obj).from_dict(tree)

        # and a file on disk, a dict written over the instance's
        with tempfile.TemporaryDirectory() as tmp:
            path = mesh.save(os.path.join(tmp, "mesh.json"))
            with open(path) as f:
                tree = json.load(f)
            tree["__dict__"] = {"junk": 1}
            with open(path, "w") as f:
                json.dump(tree, f)
            with self.assertRaisesRegex(AttributeError, "^the data names '__dict__'"):
                MeshData.load(path)

    # -- the actual typo guard ----------------------------------------

    def test_typo_raises(self):
        obj = _Leaf()
        with self.assertRaises(AttributeError) as cm:
            obj.naem = "leaf"  # typo of ``name``
        self.assertIn("naem", str(cm.exception))
        self.assertIn("_Leaf", str(cm.exception))

    def test_typo_on_property_raises(self):
        obj = _Leaf()
        with self.assertRaises(AttributeError):
            obj.valeu = 5  # typo of ``value``

    def test_typo_on_inherited_attr_raises(self):
        obj = _Child()
        with self.assertRaises(AttributeError):
            obj.naem = "child"

    def test_completely_unknown_attr_raises(self):
        obj = _Leaf()
        with self.assertRaises(AttributeError):
            obj.completely_unknown_field = 1

    def test_error_message_lists_allowed_attrs(self):
        obj = _Leaf()
        with self.assertRaises(AttributeError) as cm:
            obj.completely_unknown = 1
        msg = str(cm.exception)
        # Should mention the offender and at least one valid name.
        self.assertIn("completely_unknown", msg)
        self.assertIn("name", msg)
        self.assertIn("value", msg)

    def test_error_message_excludes_dunders_and_methods(self):
        # ``_allowed_public_attrs`` should not list dunder names, names
        # starting with underscore, or method-like callables (regular
        # methods, classmethods, staticmethods).  Properties and plain
        # data attributes (dataclass field defaults, class constants)
        # ARE expected to be present.
        import inspect as _inspect

        names = _Leaf._allowed_public_attrs()
        for attr_name in names:
            self.assertFalse(attr_name.startswith("_"), f"{attr_name!r} is private")
            # Walk the MRO to find the raw class-dict value so we can
            # check the descriptor type.  Use membership (not the value)
            # for the "found" check because dataclass defaults like
            # ``name: Optional[str] = None`` legitimately store ``None``
            # in the class dict.
            raw   = None
            found = False
            for base in _Leaf.__mro__:
                if attr_name in vars(base):
                    raw   = vars(base)[attr_name]
                    found = True
                    break
            self.assertTrue(found, f"{attr_name!r} not found anywhere in MRO")
            # Reject regular methods and classmethod/staticmethod.
            self.assertFalse(
                _inspect.isfunction(raw),
                f"{attr_name!r} is a function -- should not be in list",
            )
            self.assertNotIsInstance(
                raw,
                (classmethod, staticmethod),
                f"{attr_name!r} is a class/static method -- should not be in list",
            )

    def test_allowed_list_includes_dataclass_fields(self):
        names = _Leaf._allowed_public_attrs()
        self.assertIn("name", names)

    def test_allowed_list_includes_properties(self):
        names = _Leaf._allowed_public_attrs()
        self.assertIn("value", names)

    def test_allowed_list_excludes_what_takes_no_write(self):
        # a class setting, a read-only property and private names are not
        # names to suggest for a typo
        names = _Leaf._allowed_public_attrs()
        for name in ("EQUALITY_TEST_IGNORE", "doubled", "namespace", "_value", "_cache"):
            self.assertNotIn(name, names)

    def test_allowed_list_excludes_known_classmethods(self):
        # Sanity: ``from_dict`` / ``load`` etc. are classmethods on
        # ``Data`` and must be filtered out.
        names = _Leaf._allowed_public_attrs()
        for cm_name in ("from_dict", "from_bytes", "load", "load_json"):
            self.assertNotIn(cm_name, names, f"{cm_name!r} leaked into allowed list")

    # -- preserves existing Data machinery ----------------------------

    def test_init_does_not_raise(self):
        # __init__ assigns dataclass fields -- those are class-level so
        # the check passes.  Smoke test for accidental ``self.foo``
        # assignments inside Data subclasses.
        _Leaf()
        _Leaf(name="x", _value=3)
        _Child(name="y", _value=1, _extra=4)

    def test_post_init_does_not_raise(self):
        # __post_init__ writes back via setattr(self, field.name, ...).
        # Field names are class-level, so the check passes.
        obj = _Leaf(name="x")
        obj.__post_init__()  # idempotent; should not raise

    def test_from_dict_round_trip(self):
        obj      = _Leaf(name="src", _value=12)
        data     = obj.to_dict()
        restored = _Leaf.from_dict(data)
        self.assertEqual(restored.name, "src")
        self.assertEqual(restored._value, 12)

    def test_to_dict_includes_only_declared_fields(self):
        obj = _Leaf(name="src", _value=12)
        # Add a private cache attr -- must NOT appear in to_dict (only
        # dataclass fields are serialized).
        obj._cache = "ignored"
        data       = obj.to_dict()
        self.assertIn("name", data)
        self.assertIn("value", data)  # _value, saved as its property
        self.assertNotIn("_cache", data)

    def test_reset_cached_data_does_not_raise(self):
        # A cache set on the instance is one the class declares;
        # reset_cached_data writes it back to None.
        obj        = _Leaf()
        obj._cache = "anything"
        obj.reset_cached_data()
        self.assertIsNone(obj._cache)

    def test_copy_preserves_fields(self):
        obj   = _Leaf(name="orig", _value=9)
        clone = obj.copy()
        self.assertEqual(clone.name, "orig")
        self.assertEqual(clone._value, 9)

    # -- regression: dataclass fields without defaults ----------------
    #
    # ``MeshData`` (and other domain types) declare mandatory raw-data
    # fields without defaults, e.g. ``indices: np.ndarray`` rather than
    # ``indices: Optional[np.ndarray] = None``.  These fields do NOT
    # appear in ``vars(cls)``, so a naive ``hasattr(type(self), name)``
    # check rejects them.  The guard must consult ``fields()`` to find
    # them.  Originally surfaced by:
    #
    #     from nodetypes import Mesh
    #     import maya.cmds as mc
    #     cube = mc.polyCube()[0]
    #     Mesh(cube).serialize()[0]   # raised on ``self.indices = ...``

    def test_required_field_construction_succeeds(self):
        """``MeshData``-style required fields can be passed to the
        constructor without tripping the guard."""
        obj = _Required(
            indices = np.array([0, 1, 2]),
            counts  = np.array([3]),
            points  = np.array([[0.0, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(obj.indices, [0, 1, 2])
        np.testing.assert_array_equal(obj.counts,  [3])
        np.testing.assert_array_equal(obj.points,  [[0.0, 0.0, 0.0]])

    def test_required_field_reassignment_succeeds(self):
        obj = _Required(
            indices = np.array([0, 1, 2]),
            counts  = np.array([3]),
            points  = np.zeros((1, 3)),
        )
        # Reassigning a required field after construction must also pass
        # the guard.
        obj.indices = np.array([3, 4, 5])
        np.testing.assert_array_equal(obj.indices, [3, 4, 5])

    def test_allowed_list_includes_required_fields(self):
        # The error-message helper must also see required fields so the
        # suggestion list is useful.
        names = _Required._allowed_public_attrs()
        self.assertIn("indices", names)
        self.assertIn("counts",  names)
        self.assertIn("points",  names)
        self.assertIn("name",    names)  # optional field still listed

    def test_required_field_typo_still_raises(self):
        obj = _Required(
            indices = np.array([0, 1, 2]),
            counts  = np.array([3]),
            points  = np.zeros((1, 3)),
        )
        with self.assertRaises(AttributeError) as cm:
            obj.indeces = np.array([1])  # typo of ``indices``
        msg = str(cm.exception)
        self.assertIn("indeces", msg)
        # The suggestion list must include the real field name.
        self.assertIn("indices", msg)

    def test_dataclass_field_names_cache_per_class(self):
        # Each subclass should cache its own field-name set; the cache
        # must NOT leak between siblings.  Regression guard against a
        # naive cache stored on ``Data`` itself.
        leaf_names     = _Leaf._dataclass_field_names()
        required_names = _Required._dataclass_field_names()
        self.assertNotEqual(leaf_names, required_names)
        self.assertIn("indices", required_names)
        self.assertNotIn("indices", leaf_names)
        self.assertIn("name", leaf_names)

    def test_dataclass_field_names_returns_frozenset(self):
        # Returning an immutable frozenset prevents callers from
        # accidentally mutating the per-class cache.
        names = _Required._dataclass_field_names()
        self.assertIsInstance(names, frozenset)

    def test_writable_names_cached_per_class(self):
        # worked out once a class, kept in the class's own dict, and not
        # shared with a sibling or a parent
        self.assertIs(_Child._writable(), _Child._writable())
        self.assertIs(vars(_Child)["_WRITABLE_CACHE"], _Child._writable())
        self.assertIn("_more", _Child._writable())
        self.assertNotIn("_more", _Leaf._writable())
        self.assertNotIn("_cache", _Required._writable())
        self.assertIsInstance(_Leaf._writable(), frozenset)

    def test_what_a_class_gains_after_its_first_write_is_taken(self):
        # a property or _STATE given to a class once its names are cached
        @dataclass(repr=False, eq=False)
        class Late(Data):
            name: Optional[str] = None

        obj      = Late()
        obj.name = "cached"
        self.assertIn("_WRITABLE_CACHE", vars(Late))
        with self.assertRaises(AttributeError):
            obj._later = 1

        Late._STATE = ("_later",)
        Late.label  = property(lambda self: self.name, lambda self, value: setattr(self, "name", value))
        obj._later  = 1
        obj.label   = "late"
        self.assertEqual((obj._later, obj.name), (1, "late"))
        with self.assertRaises(AttributeError):
            obj._never = 1


# -- the list flavour ----------------------------------------------------


class _Items(DataList):
    """A list saving one setting, annotating another, holding one cache and
    one property with a setter."""

    DATA_LIST_CLASS = _Leaf
    LIST_FIELDS     = ("label",)

    quality: float

    _STATE = ("_cache",)

    def __init__(self, iterable=None):
        super().__init__(iterable)
        self.label = "none"

    @property
    def tag(self) -> str:
        return self.label

    @tag.setter
    def tag(self, value: str) -> None:
        self.label = str(value)


class TestDataListStrictSetattr(unittest.TestCase):
    """``DataList.__setattr__`` holds a list to what its class declares."""

    def test_declared_names_succeed(self):
        items         = _Items([_Leaf(name="a")])
        items.list    = [_Leaf(name="b")]
        items.label   = "x"
        items.quality = 1.5
        items._cache  = "derived"
        items.tag     = "y"
        self.assertEqual((items.label, items.quality, items._cache), ("y", 1.5, "derived"))
        self.assertEqual(items.name, ["b"])

    def test_undeclared_names_raise(self):
        items = _Items()
        for name, message in (
            ("lable", "'_Items' has no attribute 'lable'; public attributes must be declared"),
            ("_scratch", "'_Items' has no attribute '_scratch'; private attributes must be declared"),
            ("append", "'append' is a method of _Items, not a setting"),
            ("load", "'load' is a class method of _Items, not a setting"),
            ("DATA_LIST_CLASS", "'DATA_LIST_CLASS' is a class setting of _Items"),
            ("LIST_FIELDS", "'LIST_FIELDS' is a class setting of _Items"),
            ("name", "'name' is a read-only property of _Items"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(AttributeError, message):
                setattr(items, name, 5)
            self.assertNotIn(name, vars(items))

    def test_typo_lists_the_public_names(self):
        with self.assertRaises(AttributeError) as cm:
            _Items().lable = "x"
        self.assertIn("['label', 'list', 'quality', 'tag']", str(cm.exception))

    def test_what_a_list_class_gains_after_its_first_write_is_taken(self):
        class Late(DataList):
            pass

        items      = Late()
        items.list = []
        with self.assertRaises(AttributeError):
            items._later = 1
        Late._STATE  = ("_later",)
        items._later = 1
        self.assertEqual(items._later, 1)

        # a rig, whose own setter looks for user attributes too
        class LateRig(HierarchyData):
            pass

        rig = LateRig([TransformData(name="a")])
        with self.assertRaises(AttributeError):
            rig._later = 1
        LateRig._STATE = ("_later",)
        rig._later     = 1
        self.assertEqual(rig._later, 1)


# -- the cost ------------------------------------------------------------


def _accepted(node=TransformData, mesh=MeshData, rig=HierarchyData, clip=ClipData, meshes=MeshList):
    """``(objects, writes)``: instances of the classes given, and the writes
    each takes that look up no user attribute -- a field, a settable
    property, _STATE and the user attributes' dict on a node, a mesh's field
    and cache, a rig's list and _STATE, a clip's frame and cache, a mesh
    list's list -- each made once, so every class's names are worked out"""
    a = node(name="a")
    m = mesh(indices=np.arange(3), counts=np.array([3]), points=np.zeros((3, 3)))
    r = rig([a, node(name="b")])
    c = clip(HierarchyData([node(name="c")]), frames=4)

    objects = {"node": a, "mesh": m, "rig": r, "clip": c, "meshes": meshes([m])}
    writes = [
        (a, "name", "a2"),
        (a, "translate", np.ones(3)),
        (a, "_world_matrix", None),
        (a, "user_defined_attributes", a.user_defined_attributes),
        (m, "name", "m"),
        (m, "_f2v", None),
        (r, "list", r.list),
        (r, "_index_version", None),
        (c, "frame", 2),
        (c, "_column_place_cache", None),
        (objects["meshes"], "list", [m]),
    ]
    for obj, name, value in writes:
        setattr(obj, name, value)
    return objects, writes


def _grown(base, depth: int = 30, width: int = 30):
    """``base`` under ``depth`` classes, each holding ``width`` methods and
    ``width`` class settings of its own, and each of one name, so that what
    names the class -- an error's message -- is as long on any of them"""
    cls = base
    for i in range(depth):
        body = {f"method_{i}_{j}": lambda self: None for j in range(width)}
        body.update((f"SETTING_{i}_{j}", j) for j in range(width))
        cls = type(cls)(f"{base.__name__}Sub", (cls,), body)
    return cls


@contextlib.contextmanager
def _no_collection():
    """no garbage collection while a write is measured: a finalizer it ran
    would be counted as the write's"""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _profiled(write) -> Counter:
    """{(event, name): count} of the Python and C functions ``write`` calls"""
    seen = Counter()

    def profile(frame, event, arg):
        if event == "call":
            seen["call", frame.f_code.co_name] += 1
        elif event == "c_call":
            seen["c_call", getattr(arg, "__name__", repr(arg))] += 1

    prior = sys.getprofile()
    with _no_collection():
        sys.setprofile(profile)
        try:
            write()
        finally:
            sys.setprofile(prior)
    return seen


def _cost(write) -> Counter:
    """{event: count} of the Python lines and calls ``write`` runs, and the
    most memory it holds at once beyond what it started with ("held"): the
    least of three runs, so a free list one run refills is not counted. A
    copy made in C -- tuple(cls.__dict__) -- is no line and no call."""
    seen = Counter()

    def trace(frame, event, arg):
        seen[event] += 1
        return trace

    prior   = sys.gettrace()
    started = not tracemalloc.is_tracing()
    held    = []
    with _no_collection():
        sys.settrace(trace)
        try:
            write()
        finally:
            sys.settrace(prior)

        if started:
            tracemalloc.start()
        try:
            for _ in range(3):
                tracemalloc.reset_peak()
                before = tracemalloc.get_traced_memory()[0]
                write()
                held.append(tracemalloc.get_traced_memory()[1] - before)
        finally:
            if started:
                tracemalloc.stop()
    seen["held"] = min(held)
    return seen


class TestAnAcceptedWriteWorksNoNamesOut(unittest.TestCase):
    """A write the guard takes is a dict lookup and a frozenset membership:
    it never works a class's names out again (_writable_names), and a
    refused one works them out once, for a class given a name since its
    first write. What an accepted write may do beyond what
    object.__setattr__ runs for it -- a setter's own work -- is listed, not
    what it may not: it calls nothing beyond __setattr__, dict.get and
    str.startswith (and a rig's _structure_changed, with its next()), looks
    up nothing on the class but its __dict__, and on a class 30 bases
    deeper, holding 1800 names more, runs the same lines and calls and
    holds as much memory as on a class one base deeper holding none."""

    def setUp(self):
        self.objects, self.walk_free = _accepted()
        node = self.objects["node"]
        node.add_user_attribute("heroHeight", 1.0)

        # a user attribute's write looks its name up on the class once, by
        # design -- a node's in _is_builtin_attr, as 2.0.0's did, a rig's to
        # be sure it is no class attribute -- so it is only counted here
        self.writes = self.walk_free + [(node, "heroHeight", 2.0), (self.objects["rig"], "heroHeight", 3.0)]
        for obj, name, value in self.writes:  # every class's names worked out
            setattr(obj, name, value)

    def counted(self, writes):
        """the classes whose names ``writes`` work out, a refusal let pass"""
        real  = _base._writable_names
        calls = []

        def counting(cls, own):
            calls.append(cls.__name__)
            return real(cls, own)

        with mock.patch.object(_base, "_writable_names", counting):
            for obj, name, value in writes:
                try:
                    setattr(obj, name, value)
                except AttributeError:
                    pass
        return calls

    def test_an_accepted_write_works_no_names_out(self):
        self.assertEqual(self.counted(self.writes * 3), [])

    def test_a_refused_write_works_them_out_once(self):
        for obj, name in (
            (self.objects["node"], "translat"),
            (self.objects["node"], "_scratch"),
            (self.objects["mesh"], "pointz"),
            (self.objects["mesh"], "copy"),
            (self.objects["rig"], "_scratch"),
            (self.objects["clip"], "match"),
            (self.objects["meshes"], "_scratch"),
        ):
            with self.subTest(cls=type(obj).__name__, name=name):
                self.assertEqual(self.counted([(obj, name, 5)]), [type(obj).__name__])

    # what an accepted write may call beyond what object.__setattr__ runs
    # for it: the guards' own frames, the dict.get and str.startswith they
    # make and the setattr they are reached by, and a rig's
    # _structure_changed, with its next(), when its list is set
    CALLS = {
        ("call", "__setattr__"),
        ("call", "_structure_changed"),
        ("c_call", "get"),
        ("c_call", "startswith"),
        ("c_call", "setattr"),
        ("c_call", "next"),
    }

    def test_an_accepted_write_calls_only_the_guards_own(self):
        """Each write profiled twice, through the guard and past it
        (object.__setattr__, which runs a setter all the same): what the
        first calls more of than the second is among CALLS."""
        for obj, name, value in self.walk_free:
            guarded = _profiled(lambda: setattr(obj, name, value))
            bare    = _profiled(lambda: object.__setattr__(obj, name, value))
            with self.subTest(cls=type(obj).__name__, name=name):
                self.assertEqual(sorted(set(guarded - bare) - self.CALLS), [])

    def test_an_accepted_write_looks_up_nothing_on_the_class_but_its_dict(self):
        """Classes that record each name looked up on them: beyond what
        object.__setattr__ looks up for the same write, an accepted one
        reads the class's __dict__, where its names are kept, and nothing
        else -- no __mro__, no __base__, not the name it writes."""
        seen = []

        class Recording(type):
            def __getattribute__(cls, name):
                seen.append(name)
                return super().__getattribute__(name)

        class RecordingABC(ABCMeta):  # a list is a MutableSequence
            def __getattribute__(cls, name):
                seen.append(name)
                return super().__getattribute__(name)

        class Node(TransformData, metaclass=Recording):
            pass

        @dataclass(repr=False, eq=False)
        class Mesh(MeshData, metaclass=Recording):
            pass

        class Rig(HierarchyData, metaclass=RecordingABC):
            pass

        class Clip(ClipData, metaclass=RecordingABC):
            pass

        class Meshes(MeshList, metaclass=RecordingABC):
            pass

        _, writes = _accepted(Node, Mesh, Rig, Clip, Meshes)
        for obj, name, value in writes:
            del seen[:]
            setattr(obj, name, value)
            guarded = Counter(seen)
            del seen[:]
            object.__setattr__(obj, name, value)
            bare = Counter(seen)
            with self.subTest(cls=type(obj).__name__, name=name):
                self.assertEqual(sorted(set(guarded - bare) - {"__dict__"}), [])

    def test_an_accepted_write_costs_no_more_on_a_bigger_class(self):
        """A walk of a class -- its MRO, a base's dict, its own, a copy of
        one -- costs more the more the class holds; the guard's dict lookup
        does not: each write runs as many lines and calls, and holds as much
        memory, on a class 30 bases deeper, each base with 30 methods and 30
        class settings of its own, as on a class one base deeper holding
        none. A loop over the class makes no call for the profile above to
        see, and a copy of its dict made in C, tuple(cls.__dict__), no line
        either.

        A rig's list write keeps the structure count it draws, from the one
        sequence every rig edit in the process draws from
        (hierarchy._structure_ticks). Up to 256 that is an int Python keeps
        ready, past it a new one, 28 B more. The sequence is restarted past
        256 for the test, so every count either class draws is a new int,
        however many edits ran before it in the process."""
        bases = (TransformData, MeshData, HierarchyData, ClipData, MeshList)
        with mock.patch.object(hierarchy_mod, "_structure_ticks", itertools.count(1 << 20)):
            _, small = _accepted(*(_grown(b, depth=1, width=0) for b in bases))
            _, big = _accepted(*(_grown(b) for b in bases))
            for (obj, name, value), (grown, _, grown_value) in zip(small, big):
                with self.subTest(cls=type(obj).__name__, name=name):
                    self.assertEqual(
                        _cost(lambda: setattr(grown, name, grown_value)),
                        _cost(lambda: setattr(obj, name, value)),
                    )

    def test_the_cost_holds_wherever_the_structure_count_stands(self):
        """The test above, run with the structure count the edits before it
        left at each of 192 to 256, so that the counts it draws pass 256 at
        any point of it: the starts it fails at are listed. Unpinned, a
        count that passed 256 between the bigger class's list writes and
        the smaller one's had the rig's list write hold 120 B on one and
        148 B on the other (Python 3.11)."""
        name    = "test_an_accepted_write_costs_no_more_on_a_bigger_class"
        failing = []
        first   = None
        for start in range(192, 257):
            result = unittest.TestResult()
            with mock.patch.object(hierarchy_mod, "_structure_ticks", itertools.count(start)):
                type(self)(name).run(result)
            for test, message in result.failures + result.errors:
                first = first or f"{test}: {message.strip().splitlines()[-1]}"
            if result.failures or result.errors:
                failing.append(start)
        self.assertEqual(failing, [], first)


# -- every class cgmath defines --------------------------------------------

# what the guard says when it refuses a write
_REFUSED = re.compile(
    r"has no attribute .*(; (public|private) attributes must be declared|and no node has)"
    r"|is a (method|class method|static method|read-only property|class setting|class attribute) of"
    r"|is Python's, not an attribute an instance of"
)


def _found(cls) -> dict:
    """every name ``cls`` has, dunders aside, as attribute lookup finds it, unbound"""
    found = {}
    for base in cls.__mro__:
        if base is object:
            continue
        for name, raw in vars(base).items():
            if name not in found and not (name.startswith("__") and name.endswith("__")):
                found[name] = raw
    return found


def _declared(cls) -> set:
    """the names the class declares, read off the class as the rule states them"""
    if issubclass(cls, Data):
        names = {f.name for f in fields(cls)}
    else:
        names = {"list", *cls.LIST_FIELDS}
        for base in cls.__mro__:
            names.update(vars(base).get("__annotations__", {}))
    for base in cls.__mro__:
        names.update(vars(base).get("_STATE", ()))
    for name, raw in _found(cls).items():
        if isinstance(raw, property) and raw.fset is not None:
            names.add(name)
    return names


def _undeclared(cls) -> dict:
    """{name: what the refusal calls it} for every name the class has and
    does not declare, and one public and one private name it has not"""
    declared = _declared(cls)
    out      = {"zz_undeclared": None, "_zz_undeclared": None}
    for name, raw in _found(cls).items():
        if name in declared:
            continue
        if inspect.isfunction(raw):
            out[name] = "a method"
        elif isinstance(raw, classmethod):
            out[name] = "a class method"
        elif isinstance(raw, staticmethod):
            out[name] = "a static method"
        elif isinstance(raw, property):
            out[name] = "a read-only property"
        elif name.startswith("_"):
            out[name] = "a class attribute"
        else:
            out[name] = "a class setting"
    return out


class TestEveryClassTakesOnlyWhatItDeclares(unittest.TestCase):
    """Every Data / DataList class cgmath defines, built filled in by its
    example builder (the ``_examples_*`` modules test_roundtrip reads): every
    field, list part and settable property takes a write, and so does every
    name its ``_STATE`` declares; nothing else does -- no undeclared public
    or private name, dunder, method, class method, static method, class
    setting or read-only property -- and a refused write leaves the instance
    as it was.
    """

    @classmethod
    def setUpClass(cls):
        cls.classes, _ = data_classes()
        cls.builders, _ = examples()

    def test_the_classes_are_all_here(self):
        # the 35 classes, each with its builder
        self.assertGreaterEqual(len(self.classes), 35)
        missing = sorted(name for name, cls in self.classes.items() if cls not in self.builders)
        self.assertEqual(missing, [])

    def test_every_declared_name_takes_a_write(self):
        for name, cls in sorted(self.classes.items()):
            with self.subTest(cls=name):
                obj = self.builders[cls]()
                for attr in sorted(_declared(cls)):
                    # each written back as it reads, or None when it reads
                    # nothing yet: the setter may refuse the value, the guard
                    # may not refuse the name
                    try:
                        value = getattr(obj, attr)
                    except Exception:
                        value = None
                    try:
                        setattr(obj, attr, value)
                    except AttributeError as error:
                        self.assertIsNone(_REFUSED.search(str(error)), f"{name}.{attr}: {error}")
                    except Exception:
                        pass

    def test_every_undeclared_name_is_refused(self):
        for name, cls in sorted(self.classes.items()):
            obj   = self.builders[cls]()
            names = _undeclared(cls)
            kinds = {kind for kind in names.values() if kind}
            with self.subTest(cls=name):
                # every kind there is to refuse, on every class
                self.assertTrue({"a method", "a class method", "a class setting"} <= kinds, kinds)
            for attr, kind in sorted(names.items()):
                with self.subTest(cls=name, attr=attr):
                    before = dict(vars(obj))
                    with self.assertRaises(AttributeError) as caught:
                        setattr(obj, attr, 5)
                    message = str(caught.exception)
                    if kind is None:
                        self.assertIn(f"has no attribute {attr!r}", message)
                    else:
                        self.assertIn(f"{attr!r} is {kind} of {cls.__name__}", message)
                    self.assertEqual(vars(obj).keys(), before.keys())
                    self.assertTrue(all(vars(obj)[k] is v for k, v in before.items()))

    def test_no_dunder_takes_a_write(self):
        # __dict__ = {...} would swap every field for what it holds, and
        # __class__ = another class with the same layout the instance's
        # class: refused on every class, and on a rig's views and a clip's
        root    = TransformData(name="root")
        rig     = HierarchyData([root, TransformData(name="a", parent_uuid=root.uuid)])
        clip    = ClipData(rig, frames=3)
        objects = [(name, self.builders[cls]()) for name, cls in sorted(self.classes.items())]
        objects += [
            ("rig[0:1]", rig[0:1]),
            ("rig.get_children()", rig.get_children()),
            ("clip.frames[0:1]", clip.frames[0:1]),
        ]
        for name, obj in objects:
            cls     = type(obj)
            swapped = _Leaf if isinstance(obj, Data) else _Items
            for attr, value in (("__dict__", {"_junk": 1}), ("__class__", swapped)):
                with self.subTest(obj=name, attr=attr):
                    before = dict(vars(obj))
                    with self.assertRaisesRegex(
                        AttributeError,
                        f"^'{attr}' is Python's, not an attribute an instance of {cls.__name__} takes$",
                    ):
                        setattr(obj, attr, value)
                    self.assertIs(type(obj), cls)
                    self.assertEqual(vars(obj).keys(), before.keys())
                    self.assertTrue(all(vars(obj)[k] is v for k, v in before.items()))

    def test_the_holes_are_closed(self):
        # what 2.0.0 took and stored on the instance
        node = TransformData(name="a")
        mesh = MeshData(indices=np.arange(3), counts=np.array([3]), points=np.zeros((3, 3)))
        rig  = HierarchyData([TransformData(name="b")])
        for obj, attr, message in (
            (node, "_test", "'TransformData' has no attribute '_test'; private attributes must be declared on the class, in its _STATE"),
            (mesh, "copy", "'copy' is a method of MeshData, not a setting"),
            (mesh, "load", "'load' is a class method of MeshData, not a setting"),
            (mesh, "EQUALITY_TEST_IGNORE", "'EQUALITY_TEST_IGNORE' is a class setting of MeshData, set on the class, not on an instance"),
            (rig, "copy", "'copy' is a method of HierarchyData, not a setting"),
            (rig, "match", "'match' is a method of HierarchyData, not a setting"),
            (rig, "_x", "'HierarchyData' has no attribute '_x'; private attributes must be declared on the class, in its _STATE"),
            (node, "parent_name", "'parent_name' is a read-only property of TransformData"),
        ):
            with self.subTest(cls=type(obj).__name__, attr=attr):
                with self.assertRaises(AttributeError) as caught:
                    setattr(obj, attr, 5)
                self.assertEqual(str(caught.exception), message)
                self.assertNotIn(attr, vars(obj))

        # and a user attribute still goes to its node, on a node and a list
        node.add_user_attribute("heroHeight", 1.0)
        node.heroHeight = 3.1416
        rig.append(node)
        rig.heroHeight = 2
        self.assertEqual(rig.heroHeight, [None, 2.0])


if __name__ == "__main__":
    unittest.main()