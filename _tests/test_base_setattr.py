"""Tests for the strict ``Data.__setattr__`` typo-guard.

The guard rejects writes to public attributes that are not declared on
the class (or any base in the MRO), catching mistakes like
``obj.ambiant = 0.5`` (typo of ``ambient``).  Private attributes
(prefixed with ``_``) and any attribute already discoverable on the
class -- dataclass fields, ``@property`` descriptors, methods, class
constants -- are still freely assignable.
"""

import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.geometry._base import Data


# -- test fixtures: minimal Data subclasses ----------------------------


@dataclass(repr=False, eq=False)
class _Leaf(Data):
    """Bare ``Data`` subclass exercising:
    - a public dataclass field (``name``),
    - a private dataclass field (``_value``),
    - a ``@property`` / setter pair (``value``).
    """

    name:   Optional[str] = None
    _value: int = 0

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        self._value = int(v)


@dataclass(repr=False, eq=False)
class _Child(_Leaf):
    """Inherits from ``_Leaf`` to verify MRO-based attribute lookup."""

    _extra: int = 0

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


# -- tests -------------------------------------------------------------


class TestDataStrictSetattr(unittest.TestCase):
    """``Data.__setattr__`` rejects assignments to undeclared public
    attributes (typo guard) while leaving normal usage unaffected."""

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

    def test_private_attribute_succeeds(self):
        obj        = _Leaf()
        obj._cache = "anything"
        self.assertEqual(obj._cache, "anything")

    def test_private_dataclass_field_succeeds(self):
        # ``_value`` is a private dataclass field on _Leaf -- writable
        # through both the ``_`` prefix bypass AND the class-level
        # default.
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

    def test_class_constant_assignment_succeeds(self):
        # ``EQUALITY_TEST_IGNORE`` is declared on Data -- shadowing it
        # on an instance must remain allowed.
        obj                      = _Leaf()
        obj.EQUALITY_TEST_IGNORE = ["name"]
        self.assertEqual(obj.EQUALITY_TEST_IGNORE, ["name"])

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

    def test_allowed_list_includes_class_constants(self):
        names = _Leaf._allowed_public_attrs()
        self.assertIn("EQUALITY_TEST_IGNORE", names)

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
        self.assertIn("_value", data)
        self.assertNotIn("_cache", data)

    def test_reset_cached_data_does_not_raise(self):
        # Cache slots set on the instance must already be private (start
        # with ``_``); confirms reset_cached_data still works.
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


if __name__ == "__main__":
    unittest.main()