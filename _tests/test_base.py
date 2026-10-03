import copy
import copyreg
import json as std_json
import os
import pickle
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from cgmath.geometry._base import (
    _as_declared,
    _CLASSES,
    _is_numeric_or_boolean,
    _is_numeric_sequence,
    _is_sequence,
    _register,
    Array,
    CLASS_KEY,
    Data,
    DataList,
    dict_to_zip,
    flatten_nested_lists,
    get_annotations,
    get_file_type,
    ImmutableArray,
    is_ndarray_annotation,
    LegacyFileError,
    zip_to_dict,
)

# from cgmath.geometry.mesh import  MeshData, UVData


@dataclass(repr=False, eq=False)
class ParentClass(Data):
    """Example parent class"""

    foo: int = 1
    bar: float = 1.0


@dataclass(repr=False, eq=False)
class ChildClass(ParentClass):
    """Example child class"""

    baz: bool = False


@dataclass(repr=False, eq=False)
class TestDataClass(Data):
    """Test data class with various types"""

    name:   str = "test"
    values: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
    count:  int = 5


class TestDataList(DataList):
    """Test data list class"""

    DATA_LIST_CLASS = TestDataClass


@dataclass(repr=False, eq=False)
class OptionalArrayDataClass(Data):
    """Test data class with an optional array field"""

    name:   str = "test"
    values: Optional[np.ndarray] = None


class OptionalArrayDataList(DataList):
    """Test data list class holding optional array fields"""

    DATA_LIST_CLASS = OptionalArrayDataClass


@dataclass(repr=False, eq=False)
class ArrayContainerDataClass(Data):
    """Test data class holding a list of arrays rather than an array"""

    name:   str = "test"
    chunks: List[np.ndarray] = field(default_factory=list)


@dataclass(repr=False, eq=False)
class TypedDataClass(Data):
    """Test data class declaring its array fields"""

    indices: Array(np.int32, "N")
    points:  Array(np.float64, "N", 3)
    name:    Optional[str] = None
    matrix:  Optional[Array(np.float64, 4, 4)] = ImmutableArray(np.eye(4))
    image:   Optional[Array((np.uint8, np.uint16, np.float32))] = None
    pair:    Tuple[int, int] = (1, 2)
    payload: Optional[dict] = None
    child:   Optional[Data] = None


class TypedDataList(DataList):
    """Test data list saving a setting of its own"""

    DATA_LIST_CLASS = TypedDataClass
    LIST_FIELDS     = ("label",)

    def __init__(self, iterable=None):
        super().__init__(iterable)
        self.label = "none"


def typed(**kwargs):
    """a populated TypedDataClass"""
    values = dict(indices=[0, 1, 2], points=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    values.update(kwargs)
    return TypedDataClass(**values)


class TestDataHelperFunctions(unittest.TestCase):
    """Test helper functions used by Data class"""

    def test_is_sequence(self):
        """Test _is_sequence() function"""
        # Should be sequences
        self.assertTrue(_is_sequence([1, 2, 3]))
        self.assertTrue(_is_sequence((1, 2, 3)))
        self.assertTrue(_is_sequence(np.array([1, 2, 3])))

        # Should not be sequences
        self.assertFalse(_is_sequence(5))
        self.assertFalse(_is_sequence("string"))
        self.assertFalse(_is_sequence({"key": "value"}))
        self.assertFalse(_is_sequence({1, 2, 3}))

        # 0-dimensional array is not a sequence
        self.assertFalse(_is_sequence(np.array(5)))

    def test_is_numeric_or_boolean(self):
        """Test _is_numeric_or_boolean() function"""
        # Numeric arrays
        self.assertTrue(_is_numeric_or_boolean(np.array([1, 2, 3])))
        self.assertTrue(_is_numeric_or_boolean(np.array([1.0, 2.0, 3.0])))

        # Boolean arrays
        self.assertTrue(_is_numeric_or_boolean(np.array([True, False, True])))

        # Non-numeric arrays
        self.assertFalse(_is_numeric_or_boolean(np.array(["a", "b", "c"])))
        self.assertFalse(_is_numeric_or_boolean(np.array([object(), object()])))

    def test_is_numeric_sequence(self):
        """Test _is_numeric_sequence() function"""
        # Numeric sequences
        self.assertTrue(_is_numeric_sequence([1, 2, 3]))
        self.assertTrue(_is_numeric_sequence([1.0, 2.0, 3.0]))
        self.assertTrue(_is_numeric_sequence([[1, 2], [3, 4]]))

        # Non-numeric sequences
        self.assertFalse(_is_numeric_sequence([1, 2, "three"]))
        self.assertFalse(_is_numeric_sequence(["a", "b", "c"]))

        # Not a sequence
        self.assertFalse(_is_numeric_sequence(5))
        self.assertFalse(_is_numeric_sequence("string"))

    def test_flatten_nested_lists(self):
        """Test flatten_nested_lists() function"""
        # Simple flat list
        result = list(flatten_nested_lists([1, 2, 3]))
        self.assertEqual(result, [1, 2, 3])

        # Nested lists
        result = list(flatten_nested_lists([1, [2, 3], [4, [5, 6]]]))
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])

        # Single value (not a sequence)
        result = list(flatten_nested_lists(5))
        self.assertEqual(result, [5])

    def test_get_annotations(self):
        """Test get_annotations() function"""
        # Parent class annotations
        parent_annot = get_annotations(ParentClass)
        self.assertIn("foo", parent_annot)
        self.assertIn("bar", parent_annot)

        # Child class annotations (includes inherited)
        child_annot = get_annotations(ChildClass)
        self.assertIn("foo", child_annot)
        self.assertIn("bar", child_annot)
        self.assertIn("baz", child_annot)

    def test_is_ndarray_annotation(self):
        """Test is_ndarray_annotation() function"""
        self.assertTrue(is_ndarray_annotation(np.ndarray))
        self.assertTrue(is_ndarray_annotation(Optional[np.ndarray]))
        self.assertTrue(is_ndarray_annotation(np.ndarray | None))
        self.assertTrue(is_ndarray_annotation(Union[np.ndarray, None]))

        # unresolved annotations, as left by `from __future__ import annotations`
        self.assertTrue(is_ndarray_annotation("np.ndarray"))
        self.assertTrue(is_ndarray_annotation("numpy.ndarray"))
        self.assertTrue(is_ndarray_annotation("Optional[np.ndarray]"))
        self.assertTrue(is_ndarray_annotation("typing.Optional[np.ndarray]"))
        self.assertTrue(is_ndarray_annotation("np.ndarray | None"))

        self.assertFalse(is_ndarray_annotation(int))
        self.assertFalse(is_ndarray_annotation(Optional[int]))
        self.assertFalse(is_ndarray_annotation("str"))
        self.assertFalse(is_ndarray_annotation(None))

        # a container of arrays is not an array: np.asarray() would either
        # collapse it into a single array or choke on ragged members
        self.assertFalse(is_ndarray_annotation(List[np.ndarray]))
        self.assertFalse(is_ndarray_annotation(Dict[str, np.ndarray]))
        self.assertFalse(is_ndarray_annotation(Optional[List[np.ndarray]]))
        self.assertFalse(is_ndarray_annotation("list[np.ndarray]"))
        self.assertFalse(is_ndarray_annotation("Optional[list[np.ndarray]]"))

        # a declared array, resolved or not
        self.assertTrue(is_ndarray_annotation(Array(np.int32, "N")))
        self.assertTrue(is_ndarray_annotation(Optional[Array(np.float64, "N", 3)]))
        self.assertTrue(is_ndarray_annotation('Array(np.int32, "N")'))
        self.assertTrue(is_ndarray_annotation("Optional[Array(np.float64, 4, 4)]"))

    def test_immutable_array(self):
        """Test ImmutableArray class"""
        arr = ImmutableArray([1, 2, 3])

        # Should be a numpy array subclass
        self.assertIsInstance(arr, np.ndarray)

        # Should be hashable
        hash_val = hash(arr)
        self.assertIsInstance(hash_val, int)

        # Two arrays with same values should have same hash
        arr2 = ImmutableArray([1, 2, 3])
        self.assertEqual(hash(arr), hash(arr2))


class TestData(unittest.TestCase):
    def test_to_dict(self):
        """Test that to_dict picks up parent class annotations in addition to child class annotations"""
        parent = ParentClass()
        child  = ChildClass()

        # fields left at their defaults are left out
        self.assertEqual(sorted(parent.to_dict()), [CLASS_KEY])

        parent = ParentClass(foo=2, bar=3.0)
        child  = ChildClass(foo=2, bar=3.0, baz=True)
        self.assertEqual(sorted(parent.to_dict()), sorted([CLASS_KEY, "bar", "foo"]))
        self.assertEqual(sorted(child.to_dict()), sorted([CLASS_KEY, "bar", "baz", "foo"]))

    def test_from_dict_preserves_array_containers(self):
        """A list of arrays must survive from_dict() as a list, not be merged
        into one array -- ragged members would make np.asarray() raise"""
        obj    = ArrayContainerDataClass(chunks=[np.array([1.0, 2.0]), np.array([3.0])])
        loaded = ArrayContainerDataClass.from_dict(obj.to_dict())

        self.assertIsInstance(loaded.chunks, list)
        self.assertEqual(len(loaded.chunks),        2)
        self.assertEqual(loaded.chunks[0].tolist(), [1.0, 2.0])
        self.assertEqual(loaded.chunks[1].tolist(), [3.0])

    def test_str_and_repr(self):
        """Test __str__ and __repr__ methods"""
        # Without name attribute
        obj      = ParentClass()
        repr_str = repr(obj)
        self.assertIn("ParentClass", repr_str)

        # With name attribute
        named_obj = TestDataClass(name="my_object")
        self.assertEqual(str(named_obj), "my_object")
        self.assertIn("TestDataClass", repr(named_obj))
        self.assertIn("my_object", repr(named_obj))

    def test_equality(self):
        """Test __eq__ and __ne__ operators"""
        obj1 = TestDataClass(name="test1", values=np.array([1.0, 2.0, 3.0]), count=5)
        obj2 = TestDataClass(name="test1", values=np.array([1.0, 2.0, 3.0]), count=5)
        obj3 = TestDataClass(name="test2", values=np.array([4.0, 5.0, 6.0]), count=10)

        # Equal objects
        self.assertTrue(obj1 == obj2)
        self.assertFalse(obj1 != obj2)

        # Unequal objects
        self.assertFalse(obj1 == obj3)
        self.assertTrue(obj1 != obj3)

        # Different types
        self.assertFalse(obj1 == "not a data object")

    def test_comparison_operators(self):
        """Test comparison operators (lt, le, gt, ge)"""
        obj1 = TestDataClass(name="alpha")
        obj2 = TestDataClass(name="beta")
        obj3 = TestDataClass(name="gamma")

        # Less than
        self.assertTrue(obj1 < obj2)
        self.assertFalse(obj2 < obj1)

        # Less than or equal
        self.assertTrue(obj1 <= obj2)
        self.assertTrue(obj1 <= TestDataClass(name="alpha"))

        # Greater than
        self.assertTrue(obj3 > obj2)
        self.assertFalse(obj2 > obj3)

        # Greater than or equal
        self.assertTrue(obj3 >= obj2)
        self.assertTrue(obj3 >= TestDataClass(name="gamma"))

    def test_hash(self):
        """Test __hash__ method"""
        obj1 = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)
        obj2 = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        # Equal objects should have equal hashes
        self.assertEqual(hash(obj1), hash(obj2))

        # Can be used in sets
        obj_set = {obj1, obj2}
        self.assertEqual(len(obj_set), 1)

    def test_copy(self):
        """Test copy() method creates deep copy"""
        obj      = TestDataClass(name="original", values=np.array([1.0, 2.0, 3.0]), count=5)
        obj_copy = obj.copy()

        # Should be equal but not same object
        self.assertEqual(obj, obj_copy)
        self.assertIsNot(obj, obj_copy)

        # Modifying copy should not affect original
        obj_copy.name      = "modified"
        obj_copy.values[0] = 999.0
        self.assertNotEqual(obj.name, obj_copy.name)
        self.assertNotEqual(obj.values[0], obj_copy.values[0])

    def test_match(self):
        """Test match() method"""
        obj = TestDataClass(name="test_object_123")

        # Exact match with wildcards
        self.assertTrue(obj.match("test_*"))
        self.assertTrue(obj.match("*_123"))
        self.assertTrue(obj.match("test_object_123"))

        # Case-sensitive by default (exact=True)
        self.assertFalse(obj.match("TEST_*", exact=True))

        # Case-insensitive (exact=False)
        self.assertTrue(obj.match("TEST_*", exact=False))

        # Multiple patterns
        self.assertTrue(obj.match("test_*", "other_*"))
        self.assertFalse(obj.match("other_*", "another_*"))

    def test_serialization_pickle(self):
        """Test pickle serialization"""
        obj = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.pkl")

            # Save
            obj.save_pickle(filename)
            self.assertTrue(os.path.exists(filename))

            # Load
            loaded = TestDataClass.load_pickle(filename)
            self.assertEqual(obj, loaded)

    def test_serialization_json(self):
        """Test JSON serialization"""
        obj = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.json")

            # Save
            obj.save_json(filename)
            self.assertTrue(os.path.exists(filename))

            # Load
            loaded = TestDataClass.load_json(filename)
            self.assertEqual(obj, loaded)

    def test_serialization_json_optional_array(self):
        """Test that optional array fields load back as arrays, not lists"""
        obj = OptionalArrayDataClass(values=np.array([1.0, 2.0, 3.0]))

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.json")

            obj.save_json(filename)
            loaded = OptionalArrayDataClass.load_json(filename)
            self.assertIsInstance(loaded.values, np.ndarray)
            self.assertEqual(obj, loaded)

            # a field holding no data stays None
            empty = OptionalArrayDataClass()
            empty.save_json(filename)
            self.assertIsNone(OptionalArrayDataClass.load_json(filename).values)

    def test_serialization_npz(self):
        """Test NPZ serialization"""
        obj = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.npz")

            # Save
            obj.save_npz(filename)
            self.assertTrue(os.path.exists(filename))

            # Load
            loaded = TestDataClass.load_npz(filename)
            self.assertEqual(obj, loaded)

    def test_serialization_auto_mode(self):
        """Test save/load with automatic mode detection"""
        obj = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test .pkl extension
            pkl_file = os.path.join(temp_dir, "test.pkl")
            obj.save(pkl_file)
            loaded = TestDataClass.load(pkl_file)
            self.assertEqual(obj, loaded)

            # Test .json extension
            json_file = os.path.join(temp_dir, "test.json")
            obj.save(json_file)
            loaded = TestDataClass.load(json_file)
            self.assertEqual(obj, loaded)

            # Test .npz extension
            npz_file = os.path.join(temp_dir, "test.npz")
            obj.save(npz_file)
            loaded = TestDataClass.load(npz_file)
            self.assertEqual(obj, loaded)

    def test_to_from_bytes(self):
        """Test to_bytes() and from_bytes() methods"""
        obj = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        # Convert to bytes
        data_bytes = obj.to_bytes()
        self.assertIsInstance(data_bytes, bytes)

        # Convert back from bytes
        loaded = TestDataClass.from_bytes(data_bytes)
        self.assertEqual(obj, loaded)

    def test_to_json(self):
        """Test to_json() method"""
        obj = TestDataClass(name="named", values=np.array([1.0, 2.0, 4.0]), count=6)

        json_str = obj.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("named", json_str)
        self.assertIn("count", json_str)


class TestArrayFields(unittest.TestCase):
    """declared array fields take their dtype and shape when set"""

    def test_dtype_is_converted_on_assignment(self):
        obj = typed(indices=np.arange(3, dtype=np.int64), points=np.zeros((3, 3), np.float32))
        self.assertEqual(obj.indices.dtype, np.int32)
        self.assertEqual(obj.points.dtype, np.float64)

        obj.indices = [2, 1, 0]
        self.assertEqual(obj.indices.dtype, np.int32)

    def test_values_no_declared_dtype_holds_are_refused(self):
        # a cast would wrap them: 70000 to 112 and -1 to 255 in a uint8
        for value in (70000, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fit none of uint8/uint16"):
                    typed(image=np.full((2, 2), value, np.int32))
        self.assertEqual(typed(image=np.full((2, 2), 300, np.int32)).image.dtype, np.uint16)

    def test_matching_dtype_is_not_copied(self):
        points = np.zeros((3, 3))
        self.assertIs(typed(points=points).points, points)

    def test_a_wrong_trailing_size_raises(self):
        with self.assertRaisesRegex(ValueError, "points must have shape"):
            typed(points=np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            typed().matrix = np.eye(3)

    def test_an_empty_value_takes_the_declared_size(self):
        obj = typed(indices=[], points=[])
        self.assertEqual(obj.points.shape, (0, 3))
        self.assertEqual(obj.indices.shape, (0,))

    def test_a_tuple_of_dtypes_keeps_the_values_own(self):
        for dtype in (np.uint8, np.uint16, np.float32):
            with self.subTest(dtype=dtype):
                self.assertEqual(typed(image=np.zeros((2, 2), dtype)).image.dtype, dtype)

        # anything else becomes the first declared dtype of its kind that holds it
        self.assertEqual(typed(image=np.full(4, 300)).image.dtype, np.uint16)
        self.assertEqual(typed(image=np.zeros(4)).image.dtype, np.float32)

    def test_defaults_are_not_shared(self):
        a, b = typed(), typed()
        a.matrix[3, 0] = 7.0
        self.assertEqual(b.matrix[3, 0], 0.0)
        self.assertEqual(TypedDataClass.matrix[3, 0], 0.0)


class TestFiles(unittest.TestCase):
    """pkl, npz and json all hold the same tree and load the same object"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def round_trips(self, obj):
        """obj saved and loaded back in each format"""
        for ext in ("pkl", "npz", "json"):
            path = os.path.join(self.tmp, f"data.{ext}")
            obj.save(path)
            yield ext, type(obj).load(path)

    def test_every_format_gives_back_the_same_object(self):
        obj = typed(
            name    = "thing",
            matrix  = np.diag([1.0, 2.0, 3.0, 1.0]),
            image   = np.full((2, 2), 65535, np.uint16),
            pair    = (3, 4),
            payload = {"a": None, "b": [1, 2], "c": ("x", 2.5), "d": {}, "e": np.nan},
            child   = ParentClass(foo=7),
        )
        for ext, back in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertIs(type(back), TypedDataClass)
                self.assertEqual(back.indices.dtype, np.int32)
                self.assertEqual(back.image.dtype, np.uint16)
                np.testing.assert_array_equal(back.image, obj.image)
                np.testing.assert_array_equal(back.matrix, obj.matrix)
                self.assertEqual(back.pair, (3, 4))
                self.assertIs(type(back.child), ParentClass)
                self.assertEqual(back.child.foo, 7)
                self.assertIsNone(back.payload["a"])
                self.assertEqual(back.payload["b"], [1, 2])
                self.assertEqual(back.payload["d"], {})
                self.assertTrue(np.isnan(back.payload["e"]))
                self.assertEqual(back, obj)

    def test_a_value_near_its_default_is_saved(self):
        matrix       = np.eye(4)
        matrix[3, 0] = 1e-9
        for ext, back in self.round_trips(typed(matrix=matrix)):
            with self.subTest(ext=ext):
                self.assertEqual(back.matrix[3, 0], 1e-9)

    def test_an_empty_array_keeps_its_size(self):
        for ext, back in self.round_trips(typed(indices=[], points=[])):
            with self.subTest(ext=ext):
                self.assertEqual(back.points.shape, (0, 3))
                self.assertEqual(back.indices.dtype, np.int32)

    def test_a_list_keeps_its_items_classes_and_its_fields(self):
        items       = TypedDataList([typed(name="a"), typed(name="b")])
        items.label = "kept"
        for ext, back in self.round_trips(items):
            with self.subTest(ext=ext):
                self.assertIs(type(back), TypedDataList)
                self.assertEqual(back.name,  ["a", "b"])
                self.assertEqual(back.label, "kept")
                self.assertEqual(back,       items)

    def test_another_class_is_refused(self):
        path = os.path.join(self.tmp, "parent.json")
        ParentClass(foo=2).save(path)
        with self.assertRaisesRegex(TypeError, "not a TestDataClass"):
            TestDataClass.load(path)

        # a subclass is fine
        ChildClass(baz=True).save(path)
        self.assertIs(type(ParentClass.load(path)), ChildClass)

    def test_older_files_are_refused(self):
        json_path = os.path.join(self.tmp, "old.json")
        with open(json_path, "w") as f:
            std_json.dump({"foo": 2}, f)

        npz_path = os.path.join(self.tmp, "old.npz")
        with zipfile.ZipFile(npz_path, "w") as zipf:
            with zipf.open("foo.npy", "w") as f:
                np.save(f, np.asarray(2))

        # how an older cgmath pickled data
        pkl_path = os.path.join(self.tmp, "old.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(_OldPickle(), f)

        for path in (json_path, npz_path, pkl_path):
            with self.subTest(path=os.path.basename(path)):
                with self.assertRaisesRegex(LegacyFileError, "older cgmath"):
                    ParentClass.load(path)

    def test_npz_holds_no_pickles(self):
        path = os.path.join(self.tmp, "data.npz")
        typed(payload={"s": ["a", "b"], "n": [1, None, 3]}).save(path)
        with zipfile.ZipFile(path) as zipf:
            for member in zipf.namelist():
                with zipf.open(member) as f:
                    np.load(f, allow_pickle=False)

    def test_npz_tree(self):
        tree = {
            "none":   None,
            "empty":  {},
            "list":   [1, 2, 3],
            "tuple":  (1.5, "a"),
            "strs":   ["x", "y"],
            "mixed":  [None, {"a": 1}, [2]],
            "array":  np.arange(4, dtype=np.uint16),
            "scalar": 2.5,
            "flag":   True,
            "text":   "hi",
        }
        path = os.path.join(self.tmp, "tree.npz")
        dict_to_zip(tree, path)
        back = zip_to_dict(path)
        self.assertEqual(set(back), set(tree))
        self.assertIsNone(back["none"])
        self.assertEqual(back["empty"], {})
        self.assertEqual(back["list"], [1, 2, 3])
        self.assertEqual(back["tuple"], (1.5, "a"))
        self.assertEqual(back["strs"], ["x", "y"])
        self.assertEqual(back["mixed"], [None, {"a": 1}, [2]])
        self.assertEqual(back["array"].dtype, np.uint16)
        self.assertEqual((back["scalar"], back["flag"], back["text"]), (2.5, True, "hi"))

    def test_a_dict_keeps_its_key_order(self):
        tree = {"a": 1, "b": None, "c": {"x": None, "y": 2}}
        path = os.path.join(self.tmp, "order.npz")
        dict_to_zip(tree, path)
        back = zip_to_dict(path)
        self.assertEqual(list(back), ["a", "b", "c"])
        self.assertEqual(list(back["c"]), ["x", "y"])

    def test_any_text_saves_as_json(self):
        obj = typed(name="\u814f \u5de6\u80a1\u95a2\u7bc0 caf\u00e9")
        for ext, back in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertEqual(back.name, obj.name)

    def test_compact_json_is_json(self):
        path = os.path.join(self.tmp, "compact.json")
        with open(path, "w") as f:
            f.write(std_json.dumps(ParentClass(foo=2).to_dict()))
        self.assertEqual(get_file_type(path), "json")
        self.assertEqual(ParentClass.load(path).foo, 2)

    def test_any_key_comes_back(self):
        # int and tuple keys, and text a zip member name cannot carry
        payload = {
            0: "root",
            (1, 2): 5,
            "C:\\tex\\a.png": 1.0,
            "a\x00b": 2,
            "~x": 3,
            "a/b": 4,
            "": 6,
        }
        obj = typed(payload=payload)
        for ext, back in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertEqual(list(back.payload.items()), list(payload.items()))
        self.assertEqual(copy.deepcopy(obj).payload, payload)

    def test_a_failed_save_leaves_the_file_as_it_was(self):
        for ext in ("pkl", "npz", "json"):
            with self.subTest(ext=ext):
                path = os.path.join(self.tmp, f"kept.{ext}")
                typed(name="good").save(path)
                with self.assertRaises(Exception):
                    # no format saves a function, found part way through
                    typed(name="bad", payload={"a": 1, "f": lambda: 0}).save(path)
                self.assertEqual(TypedDataClass.load(path).name, "good")
                self.assertEqual([x for x in os.listdir(self.tmp) if x.endswith(".tmp")], [])

    def test_a_read_only_file_is_not_replaced(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root writes read-only files")
        path = typed(name="kept").save(os.path.join(self.tmp, "locked.npz"))
        os.chmod(path, stat.S_IREAD)
        self.addCleanup(os.chmod, path, stat.S_IREAD | stat.S_IWRITE)
        with self.assertRaises(PermissionError):
            typed(name="new").save(path)
        self.assertEqual(TypedDataClass.load(path).name, "kept")

    def test_rows_pack_into_one_member(self):
        rows = [[float(i), 2.0, 3.0] for i in range(1000)]
        path = os.path.join(self.tmp, "rows.npz")
        dict_to_zip({"rows": rows, "pairs": [(1, 2), (3, 4)], "mixed": [[1, 2.0]]}, path)
        with zipfile.ZipFile(path) as zipf:
            self.assertLess(len(zipf.namelist()), 15)

        back = zip_to_dict(path)
        self.assertEqual(back["rows"],  rows)
        self.assertEqual(back["pairs"], [(1, 2), (3, 4)])
        self.assertEqual(back["mixed"], [[1, 2.0]])
        self.assertIs(type(back["mixed"][0][0]), int)

    def test_an_older_json_in_another_encoding_is_refused(self):
        path = os.path.join(self.tmp, "old_cp1252.json")
        with open(path, "wb") as f:
            f.write(b'{"name": "caf\xe9"}')
        with self.assertRaisesRegex(LegacyFileError, "older cgmath"):
            ParentClass.load(path)

    def test_an_older_pickle_of_nested_data_is_refused(self):
        # an older cgmath pickled through from_dict, nested data as bare fields
        path = os.path.join(self.tmp, "old_nested.pkl")
        with open(path, "wb") as f:
            pickle.dump(_OldNestedPickle(), f)
        with self.assertRaisesRegex(LegacyFileError, "older cgmath"):
            TypedDataClass.load(path)

    def test_a_pickle_holds_no_numpy_object(self):
        # so a pickle written under one numpy reads under another
        obj  = typed(image=np.zeros((2, 2), np.uint16), payload={"s": np.float64(2.5)})
        data = pickle.dumps(obj)
        self.assertNotIn(b"numpy", data)
        back = pickle.loads(data)
        self.assertEqual(back, obj)
        self.assertEqual(back.image.dtype, np.uint16)
        self.assertTrue(back.points.flags.writeable)

    def test_copy_is_shallow_and_deepcopy_is_deep(self):
        obj        = typed(child=ParentClass(foo=7))
        obj._cache = "derived"  # a cache, not a field
        shallow    = copy.copy(obj)
        self.assertIs(type(shallow),  TypedDataClass)
        self.assertIs(shallow.points, obj.points)
        self.assertIs(shallow.child,  obj.child)
        self.assertNotIn("_cache", vars(shallow))  # rebuilt, not carried over

        deep = copy.deepcopy(obj)
        self.assertEqual(deep, obj)
        self.assertFalse(np.shares_memory(deep.points, obj.points))
        self.assertIsNot(deep.child, obj.child)
        self.assertNotIn("_cache", vars(deep))

        # a list's shallow copy is a new list over the same items
        items = TypedDataList([obj])
        twin  = copy.copy(items)
        twin.append(typed())
        self.assertEqual(len(items), 1)
        self.assertIs(twin[0], obj)
        self.assertIsNot(copy.deepcopy(items)[0], obj)

    def test_deepcopy_keeps_what_is_shared_shared(self):
        child  = ParentClass(foo=7)
        items  = TypedDataList([typed(child=child), typed(child=child)])
        copied = copy.deepcopy({"items": items, "first": items[0], "child": child})
        self.assertIs(copied["first"],          copied["items"][0])
        self.assertIs(copied["items"][0].child, copied["items"][1].child)
        self.assertIs(copied["child"],          copied["items"][0].child)

    def test_copy_gives_what_a_file_gives(self):
        # numpy scalars come back as python ones from every file, and from copy()
        obj  = typed(payload={"n": np.int64(3), "x": np.float32(0.5), np.int64(1): "key"})
        back = obj.copy()
        self.assertIs(type(back.payload["n"]),     int)
        self.assertIs(type(back.payload["x"]),     float)
        self.assertIs(type(list(back.payload)[2]), int)
        for ext, loaded in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertEqual(list(map(type, loaded.payload)), list(map(type, back.payload)))
                self.assertEqual(hash(loaded), hash(obj))

    def test_a_nested_default_that_is_only_close_is_saved(self):
        # == on data is within tolerance and skips the name; omission is exact
        obj              = _WithDefaultChild()
        obj.child.name   = "renamed"
        obj.child.points = obj.child.points + 1e-9
        for ext, back in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertEqual(back.child.name, "renamed")
                self.assertEqual(back.child.points[1, 0], 1.0 + 1e-9)

    def test_declared_tuples_and_enums_come_back_from_every_format(self):
        obj = _Declared(
            pair   = (1, 2),
            pairs  = ((1, 2), (3, 4)),
            rows   = [(1.0, 2.0)],
            mode   = _Mode.FAST,
            modes  = (_Mode.SLOW, _Mode.FAST),
            names  = np.array(["a", "b"]),  # a list given as an array
            levels = [_Level.LOW, _Level.HIGH],
            color  = _Color.GREEN,
            arrays = [np.arange(3.0)],
            either = 1,                     # an int, which Union[int, _Color] keeps an int
            seq    = [1, 2],
            scalar = np.array(2.5),
        )
        for ext, back in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertEqual(back.pair, (1, 2))
                self.assertEqual(back.pairs, ((1, 2), (3, 4)))
                self.assertIs(type(back.pairs[0]), tuple)
                self.assertIs(type(back.rows[0]),  tuple)
                self.assertIs(back.mode,           _Mode.FAST)
                self.assertEqual(back.modes, (_Mode.SLOW, _Mode.FAST))
                self.assertEqual(back.names, ["a", "b"])
                self.assertIs(type(back.names), list)
                self.assertEqual(back.levels, [_Level.LOW, _Level.HIGH])
                self.assertIs(back.levels[0], _Level.LOW)
                self.assertIs(back.color, _Color.GREEN)
                self.assertIsInstance(back.arrays[0], np.ndarray)
                self.assertIs(type(back.either), int)
                self.assertEqual(back.seq, [1, 2])
                self.assertIsInstance(back.scalar, np.ndarray)
                self.assertEqual(back.scalar.shape, ())

    def test_awkward_values_come_back_from_every_format(self):
        payload = {
            "nul":       "a\x00",
            "nuls":      ["x\x00", "y"],
            "huge":      1 << 70,
            "bytes":     b"\x00\xff",
            "surrogate": "\ud800",
            "pairs":     [[1, 0.5], [2, 0.25]],
            "points":    [(0.0, 1.0), (2.0, 3.0)],
            "plain":     {"__class__": "not a class", "x": 1},
            "enum":      _Mode.FAST,
            "levels":    [_Level.LOW],
            "color":     _Color.RED,
            "mixed64":   [(1 << 63) + 1, 1],  # numpy would make them float64
            "\udc80":    "a key no zip member name can hold",
        }
        for ext, back in self.round_trips(typed(payload=payload)):
            with self.subTest(ext=ext):
                expected = dict(payload)
                # every file keeps an Enum member's value; a field's type restores it
                expected.update(enum="fast", levels=[1], color=1)
                if ext == "json":
                    # json has no tuple: an undeclared one reads back a list
                    expected["points"] = [list(x) for x in payload["points"]]
                self.assertEqual(back.payload, expected)
                self.assertIs(type(back.payload["pairs"][0][0]), int)
                self.assertIs(type(back.payload["mixed64"][0]), int)

    def test_what_a_file_loads_saves_again(self):
        # a loaded or copied value saves as before: a complex number that was
        # a numpy scalar, a 0-d array
        obj = typed(payload={"c": np.complex128(1 + 2j), "z": np.array(2.5)})
        for ext in ("pkl", "npz"):
            with self.subTest(ext=ext):
                path = obj.save(os.path.join(self.tmp, f"again.{ext}"))
                back = TypedDataClass.load(path)
                back.save(path)
                obj.copy().save(path)
                again = TypedDataClass.load(path)
                self.assertEqual(again.payload["c"], 1 + 2j)
                self.assertEqual(again.payload["z"].shape, ())

    def test_equal_objects_hash_the_same(self):
        # however their values are shared: one str held twice, or two equal ones
        shared = "".join(["jo", "int1"])
        obj    = typed(name=shared, payload={"a": shared})
        twin   = typed(name="joint1", payload={"a": "".join(["jo", "int1"])})
        self.assertEqual(hash(obj), hash(twin))
        for ext, back in self.round_trips(obj):
            with self.subTest(ext=ext):
                self.assertEqual(hash(back), hash(obj))

    @unittest.skipIf(sys.version_info < (3, 10), "slots=True needs python 3.10")
    def test_a_slots_class_copies(self):
        Slotted = dataclass(repr=False, eq=False, slots=True)(
            type("Slotted", (Data,), {"__annotations__": {"a": int}, "a": 0})
        )
        self.addCleanup(_CLASSES.pop, f"{__name__}.Slotted", None)
        obj = Slotted(a=5)
        self.assertEqual(copy.copy(obj).a, 5)
        self.assertEqual(copy.deepcopy(obj).a, 5)

    @unittest.skipIf(sys.version_info < (3, 10), "X | None needs python 3.10")
    def test_a_tuple_declared_with_a_bar_comes_back_a_tuple(self):
        self.assertEqual(_as_declared([1, 2], eval("tuple[int, int] | None")), (1, 2))

    def test_rows_of_pairs_pack_by_column(self):
        rows = [[i, i * 0.5] for i in range(2000)]
        path = os.path.join(self.tmp, "pairs.npz")
        dict_to_zip({"rows": rows}, path)
        with zipfile.ZipFile(path) as zipf:
            self.assertLess(len(zipf.namelist()), 6)
        self.assertEqual(zip_to_dict(path)["rows"], rows)

    def test_a_zero_d_array_keeps_its_shape(self):
        obj = typed(payload={"scalar": np.array(2.5), "when": np.array(["2024-01-01"], "datetime64[D]")})
        for protocol in (4, 5):
            with self.subTest(protocol=protocol):
                back = pickle.loads(pickle.dumps(obj, protocol=protocol))
                self.assertEqual(back.payload["scalar"].shape, ())
                self.assertEqual(back.payload["when"].dtype, np.dtype("datetime64[D]"))

    def test_out_of_band_buffers_load(self):
        obj     = typed(image=np.zeros((4, 4), np.uint8))
        buffers = []
        data    = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
        self.assertEqual(pickle.loads(data, buffers=buffers), obj)

    def test_a_save_over_an_open_file(self):
        # written in place, as a plain write: a reader holding it open does not
        # stop the save (Windows refuses to replace an open file)
        for ext in ("pkl", "npz", "json"):
            with self.subTest(ext=ext):
                path = typed(name="old").save(os.path.join(self.tmp, f"open.{ext}"))
                with open(path, "rb"):
                    typed(name="new").save(path)
                self.assertEqual(TypedDataClass.load(path).name, "new")

    def test_an_older_empty_list_pickle_is_refused(self):
        # an older cgmath pickled a list as its attributes: an empty one has
        # no item to refuse it
        path = os.path.join(self.tmp, "old_list.pkl")
        with open(path, "wb") as f:
            pickle.dump(_OldListPickle(), f)
        with self.assertRaisesRegex(LegacyFileError, "older cgmath"):
            TypedDataList.load(path)

    def test_a_reloaded_class_loads_and_copies(self):
        # importlib.reload makes a second class under the same name; files and
        # copies keep to the class asked for
        namespace = {"__module__": ParentClass.__module__, "__qualname__": ParentClass.__qualname__}
        namespace.update(__annotations__={"foo": int, "bar": float}, foo=1, bar=1.0)
        self.addCleanup(_register, ParentClass)
        twin = dataclass(repr=False, eq=False)(type("ParentClass", (Data,), namespace))

        for ext in ("pkl", "npz", "json"):
            with self.subTest(ext=ext):
                path = ParentClass(foo=3).save(os.path.join(self.tmp, f"p.{ext}"))
                self.assertEqual(ParentClass.load(path).foo, 3)
                self.assertEqual(twin.load(path).foo, 3)
        self.assertIs(type(ParentClass.load(path)), ParentClass)
        self.assertIs(type(twin.load(path)), twin)
        self.assertIs(type(ParentClass(foo=3).copy()), ParentClass)
        self.assertIs(type(ChildClass.load(ChildClass(baz=True).save(path))), ChildClass)

    def test_a_script_class_reaches_a_spawned_worker(self):
        # a worker runs the script as __mp_main__, which pickles call __main__
        script = os.path.join(self.tmp, "worker.py")
        with open(script, "w") as f:
            f.write(_WORKER_SCRIPT)
        # the child sees what this process sees: cgmath, and a runner's venv
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in sys.path if x))
        out = subprocess.run(
            [sys.executable, script], capture_output=True, text=True, env=env, timeout=300
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("Branch Leaf 7 [0.0, 1.0] {1: 'x'}", out.stdout)


_WORKER_SCRIPT = '''
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from cgmath.geometry._base import Data


@dataclass(repr=False, eq=False)
class Leaf(Data):
    a: int = 0
    v: np.ndarray = None


@dataclass(repr=False, eq=False)
class Branch(Data):
    leaf:  Leaf = None
    table: dict = field(default_factory=dict)


def echo(x):
    return x


if __name__ == "__main__":
    with ProcessPoolExecutor(1) as pool:
        back = list(pool.map(echo, [Branch(leaf=Leaf(a=7, v=np.arange(2.0)), table={1: "x"})]))[0]
    print(type(back).__name__, type(back.leaf).__name__, back.leaf.a, back.leaf.v.tolist(), back.table)
'''


class _OldPickle:
    """unpickles the way an older cgmath's data did"""

    def __reduce__(self):
        return (Data._reconstruct, (ParentClass, {"foo": 2}))


class _OldListPickle:
    """unpickles the way an older cgmath's lists did: the class, then its attributes"""

    def __reduce__(self):
        return (copyreg._reconstructor, (TypedDataList, object, None), {"list": [], "label": "none"})


class _Mode(str, Enum):
    SLOW = "slow"
    FAST = "fast"


class _Level(int, Enum):
    LOW  = 1
    HIGH = 2


class _Color(Enum):
    RED   = 1
    GREEN = 2


@dataclass(repr=False, eq=False)
class _Declared(Data):
    """fields whose type a file cannot carry: tuples, nested, Enum members and arrays"""

    pair:   Optional[Tuple[int, int]] = None
    pairs:  Tuple[Tuple[int, int], ...] = ()
    rows:   List[Tuple[float, float]] = field(default_factory=list)
    mode:   _Mode = _Mode.SLOW
    modes:  Tuple[_Mode, ...] = ()
    names:  List[str] = field(default_factory=list)
    levels: List[_Level] = field(default_factory=list)
    color:  Optional[_Color] = None
    arrays: List[np.ndarray] = field(default_factory=list)
    either: Union[int, _Color] = 0
    seq:    Union[List[int], Tuple[int, ...], None] = None
    scalar: Optional[np.ndarray] = None


@dataclass(repr=False, eq=False)
class _WithDefaultChild(Data):
    """a Data field whose default is itself data"""

    child: TypedDataClass = field(default_factory=lambda: typed(name="default"))


class _OldNestedPickle:
    """unpickles the way an older cgmath's Object did: from_dict, nested data bare"""

    def __reduce__(self):
        tree = {"indices": [0, 1, 2], "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "child": {"foo": 7}}
        return (TypedDataClass.from_dict, (tree,))


class TestDataListClass(unittest.TestCase):
    """Test DataList class functionality"""

    def setUp(self):
        super().setUp()

        self.obj1 = TestDataClass(name="obj1", values=np.array([1.0, 2.0]), count=1)
        self.obj2 = TestDataClass(name="obj2", values=np.array([3.0, 4.0]), count=2)
        self.obj3 = TestDataClass(name="obj3", values=np.array([5.0, 6.0]), count=3)

        self.data_list = TestDataList([self.obj1, self.obj2, self.obj3])

    def test_list_operations(self):
        """Test basic list operations"""
        # Length
        self.assertEqual(len(self.data_list), 3)

        # Indexing
        self.assertEqual(self.data_list[0],  self.obj1)
        self.assertEqual(self.data_list[1],  self.obj2)
        self.assertEqual(self.data_list[-1], self.obj3)

        # Slicing
        sliced = self.data_list[0:2]
        self.assertIsInstance(sliced, TestDataList)
        self.assertEqual(len(sliced), 2)

    def test_append_and_extend(self):
        """Test append() and extend() methods"""
        data_list = TestDataList()

        # Append single item
        data_list.append(self.obj1)
        self.assertEqual(len(data_list), 1)

        # Extend with multiple items
        data_list.extend([self.obj2, self.obj3])
        self.assertEqual(len(data_list), 3)

    def test_pop_and_delete(self):
        """Test pop() and __delitem__ methods"""
        data_list = self.data_list.copy()

        # Pop last item
        popped = data_list.pop()
        self.assertEqual(popped, self.obj3)
        self.assertEqual(len(data_list), 2)

        # Pop at index
        popped = data_list.pop(0)
        self.assertEqual(popped, self.obj1)
        self.assertEqual(len(data_list), 1)

        # Delete by index
        del data_list[0]
        self.assertEqual(len(data_list), 0)

    def test_contains(self):
        """Test __contains__ method"""
        # By object
        self.assertIn(self.obj1, self.data_list)
        self.assertIn(self.obj2, self.data_list)

        # By name
        self.assertIn("obj1", self.data_list)
        self.assertIn("obj2", self.data_list)
        self.assertNotIn("obj99", self.data_list)

    def test_index(self):
        """Test index() method"""
        # Find by name
        idx = self.data_list.index("obj2")
        self.assertEqual(idx, 1)

        # Not found raises ValueError
        with self.assertRaises(ValueError):
            self.data_list.index("nonexistent")

    def test_sort(self):
        """Test sort() method"""
        # Create unsorted list
        data_list = TestDataList([self.obj3, self.obj1, self.obj2])

        # Sort by name
        data_list.sort()

        # Should be sorted alphabetically by name
        self.assertEqual(data_list[0].name, "obj1")
        self.assertEqual(data_list[1].name, "obj2")
        self.assertEqual(data_list[2].name, "obj3")

        # Sort in reverse
        data_list.sort(reverse=True)
        self.assertEqual(data_list[0].name, "obj3")

    def test_getitem_by_name(self):
        """Test __getitem__ with string name"""
        obj = self.data_list["obj2"]
        self.assertEqual(obj, self.obj2)

    def test_getitem_by_list(self):
        """Test __getitem__ with list of indices"""
        subset = self.data_list[[0, 2]]
        self.assertIsInstance(subset, TestDataList)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0],   self.obj1)
        self.assertEqual(subset[1],   self.obj3)

    def test_getitem_by_index_array(self):
        """Test __getitem__ with a numpy array of indices"""
        subset = self.data_list[np.array([2, 0])]
        self.assertIsInstance(subset, TestDataList)
        self.assertEqual([obj.name for obj in subset], ["obj3", "obj1"])

    def test_getitem_by_boolean_mask(self):
        """Test __getitem__ with a boolean mask"""
        subset = self.data_list[np.array([True, False, True])]
        self.assertIsInstance(subset, TestDataList)
        self.assertEqual([obj.name for obj in subset], ["obj1", "obj3"])

        # the subset references the source objects, it does not copy them
        self.assertIs(subset[0], self.obj1)
        self.assertIs(subset[1], self.obj3)

        # python booleans are a mask too, not the indices 0 and 1
        subset = self.data_list[[True, False, True]]
        self.assertEqual([obj.name for obj in subset], ["obj1", "obj3"])

    def test_getitem_by_empty_and_full_boolean_mask(self):
        """Test __getitem__ with all-False and all-True boolean masks"""
        subset = self.data_list[np.zeros(3, dtype=bool)]
        self.assertIsInstance(subset, TestDataList)
        self.assertEqual(len(subset), 0)

        subset = self.data_list[np.ones(3, dtype=bool)]
        self.assertEqual([obj.name for obj in subset], ["obj1", "obj2", "obj3"])

    def test_getitem_by_mismatched_boolean_mask(self):
        """Test __getitem__ with a boolean mask of the wrong size"""
        with self.assertRaises(IndexError):
            self.data_list[np.array([True, False])]

        with self.assertRaises(IndexError):
            self.data_list[np.ones(4, dtype=bool)]

    def test_equality(self):
        """Test __eq__ and __ne__ methods"""
        list1 = TestDataList([self.obj1, self.obj2])
        list2 = TestDataList([self.obj1, self.obj2])
        list3 = TestDataList([self.obj1, self.obj3])

        # Equal lists
        self.assertTrue(list1 == list2)
        self.assertFalse(list1 != list2)

        # Unequal lists
        self.assertFalse(list1 == list3)
        self.assertTrue(list1 != list3)

        # anything else is not equal
        self.assertFalse(list1 == 5)
        self.assertFalse(list1 == None)  # noqa: E711

    def test_copy(self):
        """Test copy() method"""
        data_list_copy = self.data_list.copy()

        # Should be equal but not same object
        self.assertEqual(self.data_list, data_list_copy)
        self.assertIsNot(self.data_list, data_list_copy)

        # Modifying copy should not affect original
        data_list_copy[0].name = "modified"
        self.assertNotEqual(self.data_list[0].name, data_list_copy[0].name)

    def test_match(self):
        """Test match() method"""
        # Match with wildcard
        matched = self.data_list.match("obj*")
        self.assertEqual(len(matched), 3)

        # Match specific pattern
        matched = self.data_list.match("obj1", "obj3")
        self.assertEqual(len(matched), 2)

        # Exclude pattern
        matched = self.data_list.match("obj1", exclude=True)
        self.assertEqual(len(matched), 2)
        self.assertNotIn(self.obj1, matched.list)

    def test_serialization_methods(self):
        """Test DataList serialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test pickle
            pkl_file = os.path.join(temp_dir, "list.pkl")
            self.data_list.save_pickle(pkl_file)
            loaded = TestDataList.load_pickle(pkl_file)
            self.assertEqual(self.data_list, loaded)

            # Test JSON
            json_file = os.path.join(temp_dir, "list.json")
            self.data_list.save_json(json_file)
            loaded = TestDataList.load_json(json_file)
            self.assertEqual(self.data_list, loaded)

            # Test NPZ
            npz_file = os.path.join(temp_dir, "list.npz")
            self.data_list.save_npz(npz_file)
            loaded = TestDataList.load_npz(npz_file)
            self.assertEqual(self.data_list, loaded)

    def test_serialization_json_optional_array(self):
        """Test that optional array fields load back as arrays, not lists"""
        data_list = OptionalArrayDataList(
            [
                OptionalArrayDataClass(name="obj1", values=np.array([1.0, 2.0])),
                OptionalArrayDataClass(name="obj2", values=np.array([3.0, 4.0])),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "list.json")
            data_list.save_json(json_file)

            loaded = OptionalArrayDataList.load_json(json_file)
            self.assertEqual(data_list, loaded)
            for item in loaded:
                self.assertIsInstance(item.values, np.ndarray)

    def test_to_from_bytes(self):
        """Test to_bytes() and from_bytes() for DataList"""
        data_bytes = self.data_list.to_bytes()
        self.assertIsInstance(data_bytes, bytes)

        loaded = TestDataList.from_bytes(data_bytes)
        self.assertEqual(self.data_list, loaded)

    def test_str_and_repr(self):
        """Test __str__ and __repr__ methods"""
        str_repr = str(self.data_list)
        self.assertIn("obj1", str_repr)
        self.assertIn("obj2", str_repr)

        repr_str = repr(self.data_list)
        self.assertIn("TestDataList", repr_str)