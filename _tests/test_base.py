import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from cgmath.geometry._base import (
    _is_numeric_or_boolean,
    _is_numeric_sequence,
    _is_sequence,
    Data,
    DataList,
    flatten_nested_lists,
    get_annotations,
    ImmutableArray,
    is_ndarray_annotation,
    nan_to_none,
    none_to_nan,
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

    def test_none_to_nan(self):
        """Test none_to_nan() function"""
        # Simple None conversion
        self.assertTrue(np.isnan(none_to_nan(None)))

        # Non-None values unchanged
        self.assertEqual(none_to_nan(5), 5)
        self.assertEqual(none_to_nan("test"), "test")

        # Dict with None values
        result = none_to_nan({"a": 1, "b": None, "c": 3})
        self.assertEqual(result["a"], 1)
        self.assertTrue(np.isnan(result["b"]))
        self.assertEqual(result["c"], 3)

        # Nested dict
        result = none_to_nan({"outer": {"inner": None, "val": 5}})
        self.assertTrue(np.isnan(result["outer"]["inner"]))
        self.assertEqual(result["outer"]["val"], 5)

    def test_nan_to_none(self):
        """Test nan_to_none() function"""
        # NaN to None conversion
        self.assertIsNone(nan_to_none(np.nan))

        # Dict with NaN values
        result = nan_to_none({"a": 1, "b": np.nan, "c": 3})
        self.assertEqual(result["a"], 1)
        self.assertIsNone(result["b"])
        self.assertEqual(result["c"], 3)

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

        self.assertEqual(sorted(parent.to_dict().keys()), sorted(["bar", "foo"]))
        self.assertEqual(sorted(child.to_dict().keys()), sorted(["bar", "baz", "foo"]))

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
        obj = TestDataClass(name="test", values=np.array([1.0, 2.0, 3.0]), count=5)

        json_str = obj.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("test", json_str)
        self.assertIn("count", json_str)


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