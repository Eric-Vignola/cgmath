import enum
import json as stdlib_json
import math
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from cgmath import utils
from cgmath.utils import (
    _escape_json_string,
    _format_array,
    _format_nd_array,
    _get_shape,
    _is_flat_list,
    _is_uniform_nd,
    _list_array,
    json,
    pretty_json,
)


class TestEscapeJsonString(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(_escape_json_string("hello"), "hello")

    def test_double_quotes(self):
        self.assertEqual(_escape_json_string('say "hi"'), 'say \\"hi\\"')

    def test_backslash(self):
        self.assertEqual(_escape_json_string("a\\b"), "a\\\\b")

    def test_newline_and_tab(self):
        self.assertEqual(_escape_json_string("a\nb\tc"), "a\\nb\\tc")

    def test_control_characters(self):
        self.assertEqual(_escape_json_string("\r\b\f"), "\\r\\b\\f")


class TestIsFlatList(unittest.TestCase):
    def test_ints(self):
        self.assertTrue(_is_flat_list([1, 2, 3]))

    def test_floats(self):
        self.assertTrue(_is_flat_list([1.0, 2.5]))

    def test_strings(self):
        self.assertTrue(_is_flat_list(["a", "b", "c"]))

    def test_mixed_scalars(self):
        self.assertTrue(_is_flat_list([1, "two", 3.0, True, None]))

    def test_nested_list(self):
        self.assertFalse(_is_flat_list([[1, 2], [3, 4]]))

    def test_empty(self):
        self.assertTrue(_is_flat_list([]))

    def test_tuple(self):
        self.assertTrue(_is_flat_list((1, 2, 3)))

    def test_not_a_list(self):
        self.assertFalse(_is_flat_list("string"))
        self.assertFalse(_is_flat_list(42))


class TestIsUniformNd(unittest.TestCase):
    def test_flat(self):
        self.assertTrue(_is_uniform_nd([1, 2, 3]))

    def test_2d_uniform(self):
        self.assertTrue(_is_uniform_nd([[1, 2], [3, 4]]))

    def test_2d_jagged(self):
        self.assertFalse(_is_uniform_nd([[1, 2], [3]]))

    def test_3d_uniform(self):
        self.assertTrue(_is_uniform_nd([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_empty(self):
        self.assertTrue(_is_uniform_nd([]))

    def test_not_list(self):
        self.assertFalse(_is_uniform_nd(42))


class TestGetShape(unittest.TestCase):
    def test_1d(self):
        self.assertEqual(_get_shape([1, 2, 3]), (3,))

    def test_2d(self):
        self.assertEqual(_get_shape([[1, 2], [3, 4]]), (2, 2))

    def test_3d(self):
        self.assertEqual(_get_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (2, 2, 2))

    def test_empty(self):
        self.assertEqual(_get_shape([]), ())


class TestPrettyJsonScalars(unittest.TestCase):
    def test_dict_with_scalars(self):
        data   = {"name": "test", "count": 5, "ratio": 1.5, "active": True, "ref": None}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)

    def test_empty_dict(self):
        self.assertEqual(pretty_json({}), "{}")

    def test_empty_list(self):
        self.assertEqual(pretty_json([]), "[]")


class TestPrettyJsonNumericArrays(unittest.TestCase):
    def test_short_1d_single_line(self):
        data   = {"values": [1, 2, 3]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        self.assertIn("[ 1, 2, 3 ]", result)

    def test_long_1d_wraps(self):
        data   = {"values": list(range(30))}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        lines = result.splitlines()
        self.assertTrue(any("\n" in result for _ in lines))

    def test_2d_column_aligned(self):
        data   = {"matrix": [[1, 200, 3], [4000, 5, 60]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        # columns should be right-aligned
        self.assertIn("   1", result)
        self.assertIn("4000", result)

    def test_2d_inline_brackets(self):
        data   = {"pts": [[1, 2], [3, 4]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        # first row should be on the same line as the key
        self.assertIn('"pts": [[ ', result)
        # closing ]] should be on the last data row
        self.assertIn(" ]]", result)
        # closing ]] should NOT be on its own line
        for line in result.splitlines():
            stripped = line.strip()
            self.assertNotEqual(stripped, "]")

    def test_2d_float_decimal_aligned(self):
        data   = {"pts": [[0.0, -1.0], [123.456, 0.5]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        # decimal points should align within each column
        lines    = result.splitlines()
        dot_cols = []
        for line in lines:
            idx = line.find(".")
            if idx >= 0:
                dot_cols.append(idx)
        # first decimal of each row should be at the same column
        row_dots = [line.find(".") for line in lines if "[" in line and "." in line]
        if len(row_dots) >= 2:
            self.assertEqual(row_dots[0], row_dots[1])

    def test_float_precision(self):
        data   = {"vals": [1.0, 2.5, 3.14159265358979]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        for a, b in zip(parsed["vals"], data["vals"]):
            self.assertAlmostEqual(a, b, places=14)


class TestPrettyJsonStringArrays(unittest.TestCase):
    def test_short_string_list_single_line(self):
        data   = {"tags": ["a", "b", "c"]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        self.assertIn('[ "a", "b", "c" ]', result)

    def test_long_string_list_wraps(self):
        data   = {"names": [f"item_{i:03d}" for i in range(20)]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        # should wrap across multiple lines
        array_lines = [line for line in result.splitlines() if '"item_' in line]
        self.assertGreater(len(array_lines), 1)

    def test_2d_string_array_left_aligned(self):
        data   = {"grid": [["short", "x"], ["very_long_string", "y"]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        # strings should be left-justified (shorter strings padded with trailing spaces)
        # so the comma after column 0 aligns across rows
        lines = [
            line
            for line in result.splitlines()
            if '"short"' in line or '"very_long_string"' in line
        ]
        if len(lines) == 2:
            comma_pos = [line.index(",") for line in lines]
            self.assertEqual(comma_pos[0], comma_pos[1])

    def test_strings_with_special_chars(self):
        data   = {"notes": ["line1\nline2", 'say "hello"', "back\\slash"]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)


class TestPrettyJsonNumpyArrays(unittest.TestCase):
    def test_1d_ndarray(self):
        arr    = np.array([1.0, 2.0, 3.0])
        data   = {"values": arr}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        np.testing.assert_array_almost_equal(parsed["values"], arr)

    def test_2d_ndarray(self):
        arr    = np.array([[1, 2], [3, 4]])
        data   = {"matrix": arr}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        np.testing.assert_array_equal(parsed["matrix"], arr)

    def test_ndarray_integers(self):
        arr    = np.array([10, 20, 30], dtype=np.int32)
        data   = {"ids": arr}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed["ids"], [10, 20, 30])

    def test_top_level_ndarray(self):
        arr    = np.array([1, 2, 3])
        result = pretty_json(arr)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, [1, 2, 3])


class TestPrettyJsonJaggedLists(unittest.TestCase):
    def test_jagged_list(self):
        data   = {"seqs": [[1, 2], [3, 4, 5]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)

    def test_mixed_types_in_list(self):
        data   = {"mixed": [1, "two", [3, 4]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)

    def test_any_nesting_writes_valid_json(self):
        cases = [
            ({"v": list(np.linspace(0, 1, 3))}, {"v": [0.0, 0.5, 1.0]}),
            ({"v": [()]}, {"v": [[]]}),
            ({"v": [[1, 2], [[3, 4], [5, 6]]]}, {"v": [[1, 2], [[3, 4], [5, 6]]]}),
            ({"v": [[], [1]]}, {"v": [[], [1]]}),
            ({"v": [np.int64(3), np.float32(0.5), np.bool_(True)]}, {"v": [3, 0.5, True]}),
            ({"v": [[np.float64(1.5), 2.0], [3.0, 4.0]]}, {"v": [[1.5, 2.0], [3.0, 4.0]]}),
        ]
        for data, expected in cases:
            with self.subTest(data=repr(data)):
                self.assertEqual(stdlib_json.loads(pretty_json(data)), expected)

    def test_what_json_cannot_hold_raises(self):
        for value in ([1j], [[1j, 2j]], [object()]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    pretty_json({"v": value})


class TestPrettyJsonNested(unittest.TestCase):
    def test_nested_dict_with_arrays(self):
        data = {
            "mesh": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                "faces":    [[0, 1, 2]],
                "name":     "triangle",
            }
        }
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)

    def test_list_of_dicts(self):
        data   = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)


class TestPrettyJson3D(unittest.TestCase):
    def test_3d_array(self):
        data   = {"cube": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)


class TestPrettyJsonMaxLineWidth(unittest.TestCase):
    def test_narrow_width_forces_wrap(self):
        data   = {"vals": list(range(50))}
        result = pretty_json(data, max_line_width=40)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)
        array_lines = [
            line for line in result.splitlines() if any(c.isdigit() for c in line)
        ]
        self.assertGreater(len(array_lines), 1)

    def test_wide_width_keeps_single_line(self):
        data   = {"vals": list(range(20))}
        result = pretty_json(data, max_line_width=500)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)


class TestJsonClass(unittest.TestCase):
    def test_dumps_round_trip(self):
        data   = {"a": [1, 2, 3], "b": "hello"}
        result = json.dumps(data)
        parsed = json.loads(result)
        self.assertEqual(parsed, data)

    def test_dump_and_load(self):
        data = {"values": [1.0, 2.0, 3.0], "label": "test"}
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as f:
                path = f.name
                json.dump(data, f)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    def test_dumps_with_max_line_width(self):
        data   = {"vals": list(range(20))}
        narrow = json.dumps(data, max_line_width=40)
        wide   = json.dumps(data, max_line_width=500)
        self.assertGreater(narrow.count("\n"), wide.count("\n"))

    def test_loads(self):
        raw    = '{"key": [1, 2, 3]}'
        parsed = json.loads(raw)
        self.assertEqual(parsed, {"key": [1, 2, 3]})


class TestPrettyJsonEdgeCases(unittest.TestCase):
    def test_nan_and_infinities_read_back_as_floats(self):
        # bare tokens, which python's json reads back as the floats they were
        data   = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "row": [1.5, float("nan")]}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertTrue(math.isnan(parsed["nan"]))
        self.assertEqual(parsed["inf"],    float("inf"))
        self.assertEqual(parsed["ninf"],   float("-inf"))
        self.assertEqual(parsed["row"][0], 1.5)
        self.assertTrue(math.isnan(parsed["row"][1]))

    def test_numpy_scalars(self):
        data   = {"flag": np.bool_(True), "int": np.int32(3), "float": np.float32(0.5), "zero_d": np.array(2.0)}
        parsed = stdlib_json.loads(pretty_json(data))
        self.assertEqual(parsed, {"flag": True, "int": 3, "float": 0.5, "zero_d": 2.0})

    def test_unknown_types_raise(self):
        # str() of them would write a token json cannot read back
        for value in (b"bytes", object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                pretty_json({"val": value})

    def test_control_characters_escaped(self):
        data   = {"msg": "null\x00byte\x1f"}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)


# floats whose text differs in shape: with and without a point, an exponent,
# a sign on zero, the bare tokens, the longest and shortest
_EDGE_FLOATS = (
    0.0, -0.0, 0.5, -12.25, 7.0, 100.0, 0.1, 3.141592653589793, 123456789.0,
    1e-05, 1.5e-07, 1e16, 1.5e16, 1e-300, 5e-324, 1.7976931348623157e308,
    float("nan"), float("inf"), float("-inf"),
)
_DTYPES = (
    np.float64, np.float32, np.float16, np.int8, np.int32, np.int64, np.uint8,
    np.uint64, np.bool_,
)


def _random_array(rng):
    """an array of any dtype pretty_json lays out in bulk, of 1 to 4 dimensions"""
    ndim  = int(rng.integers(1, 5))
    shape = tuple(int(x) for x in rng.integers(1, 5, ndim))
    if rng.random() < 0.5:  # often past the bulk writer's thresholds
        shape = shape[:-1] + (int(rng.integers(8, 160)),)
    dtype = _DTYPES[rng.integers(len(_DTYPES))]
    if dtype == np.bool_:
        return rng.random(shape) < 0.5
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        high = min(info.max, int(rng.choice([9, 1000, info.max])))
        return rng.integers(max(info.min, -high), high, shape, dtype=dtype, endpoint=True)

    # a few of the edge floats, so some columns have a point throughout and some do not
    pool   = rng.choice(np.array(_EDGE_FLOATS), int(rng.integers(1, 6)))
    spread = rng.normal(0, 10.0 ** rng.integers(-3, 8), shape)
    values = np.where(rng.random(shape) < 0.5, rng.choice(pool, shape), spread)
    with np.errstate(over="ignore"):
        return values.astype(dtype)


def _random_tree(rng, depth=0):
    """arrays, their lists, scalars, strings, dicts and tuples, nested"""
    pick = int(rng.integers(7)) if depth < 3 else 0
    if pick == 0:
        scalars = (None, True, 3, -0.0, 1e-05, float("nan"), 'a\n"\u00e9"', np.float32(0.1))
        return scalars[rng.integers(len(scalars))]
    if pick == 1:
        return _random_array(rng)
    if pick == 2:
        return _random_array(rng).tolist()  # exact floats, ints or bools
    if pick == 3:
        return _random_array(rng).reshape(-1, 1).tolist() + [[]]  # ragged
    if pick == 4:
        return {f"key {i}": _random_tree(rng, depth + 1) for i in range(rng.integers(4))}
    items = [_random_tree(rng, depth + 1) for _ in range(rng.integers(4))]
    return items if pick == 5 else tuple(items)


def _json_from(least, data, **kwargs):
    """pretty_json with the bulk writer taking arrays of ``least`` values or more"""
    with mock.patch.multiple(utils, _BULK_MIN_SIZE=least, _BULK_MIN_SIZE_1D_INT=least):
        return pretty_json(data, **kwargs)


def _per_value_json(data, **kwargs):
    """pretty_json with the bulk writer off: every value written one at a time"""
    return _json_from(sys.maxsize, data, **kwargs)


def _bulk_json(data, **kwargs):
    """pretty_json with the bulk writer taking every numeric array, however short"""
    return _json_from(1, data, **kwargs)


class TestPrettyJsonBulk(unittest.TestCase):
    """Numeric arrays, and lists of one number type, are laid out a whole
    array at a time, to exactly the text the per-value path writes."""

    def test_a_tree_writes_the_text_it_always_did(self):
        # written by cgmath before the bulk writer existed
        rows = [
            [0.5, -12.25, 1e-05],
            [3.0, float("nan"), 0.125],
            [-0.0, 7.5, 100.0],
            [42.0, float("inf"), -2.5],
        ]
        scale = np.array([1, -10, 100, 7, 1000, 3])
        data = {
            "rows":  np.array(rows * 3),
            "ids":   np.arange(-3, 37, dtype=np.int32),
            "flags": np.arange(40) % 3 == 0,
            "block": np.arange(36, dtype=np.int64).reshape(2, 3, 6) * scale,
        }
        expected = "\n".join([
            '{',
            '    "rows": [[  0.5,   -12.25, 1e-05 ],',
            '             [  3.0,      NaN, 0.125 ],',
            '             [ -0.0,      7.5, 100.0 ],',
            '             [ 42.0, Infinity,  -2.5 ],',
            '             [  0.5,   -12.25, 1e-05 ],',
            '             [  3.0,      NaN, 0.125 ],',
            '             [ -0.0,      7.5, 100.0 ],',
            '             [ 42.0, Infinity,  -2.5 ],',
            '             [  0.5,   -12.25, 1e-05 ],',
            '             [  3.0,      NaN, 0.125 ],',
            '             [ -0.0,      7.5, 100.0 ],',
            '             [ 42.0, Infinity,  -2.5 ]],',
            '    "ids": [ -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,',
            '             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,',
            '             31, 32, 33, 34, 35, 36 ],',
            '    "flags": [  true, false, false,  true, false, false,  true, false, false,',
            '                true, false, false,  true, false, false,  true, false, false,',
            '                true, false, false,  true, false, false,  true, false, false,',
            '                true, false, false,  true, false, false,  true, false, false,',
            '                true, false, false,  true ],',
            '    "block": [[[  0,  -10,  200,  21,  4000, 15 ],',
            '               [  6,  -70,  800,  63, 10000, 33 ],',
            '               [ 12, -130, 1400, 105, 16000, 51 ]],',
            '',
            '              [[ 18, -190, 2000, 147, 22000,  69 ],',
            '               [ 24, -250, 2600, 189, 28000,  87 ],',
            '               [ 30, -310, 3200, 231, 34000, 105 ]]]',
            '}',
        ])
        # these arrays are under the bulk writer's thresholds, so it is made
        # to take them too
        lists = {key: value.tolist() for key, value in data.items()}
        for write in (pretty_json, _bulk_json):
            with self.subTest(write.__name__):
                self.assertEqual(write(data), expected)
                self.assertEqual(write(lists), expected)

    def test_any_tree_writes_as_the_per_value_path_does(self):
        rng = np.random.default_rng(2024)
        for case in range(300):
            tree   = {"root": _random_tree(rng)}
            indent = int(rng.choice([0, 2, 4]))
            width  = int(rng.choice([1, 20, 40, 80, 500]))
            text   = _per_value_json(tree, indent=indent, max_line_width=width)
            for write in (pretty_json, _bulk_json):
                with self.subTest(write.__name__, case=case, indent=indent, width=width):
                    self.assertEqual(write(tree, indent=indent, max_line_width=width), text)

    def test_small_arrays_lay_out_as_their_lists(self):
        # under the thresholds pretty_json keeps the per-value path; the bulk
        # one must agree there too
        rng = np.random.default_rng(5)
        for case in range(300):
            array = _random_array(rng)[..., :3]
            start = int(rng.integers(0, 40))
            width = int(rng.choice([1, 20, 80]))
            with self.subTest(case=case, shape=array.shape, dtype=array.dtype.name):
                self.assertEqual(
                    _format_array(array, start, width),
                    _format_nd_array(array.tolist(), start, 4, width),
                )

    def test_lists_of_mixed_or_derived_types_keep_the_per_value_path(self):
        class Mode(enum.IntEnum):
            A = 1

        # 200 values each, past the bulk writer's thresholds
        cases = {
            "int and float": [1, 2.5] * 100,  # the 1 stays 1, not 1.0
            "bool and int":  [True, 2] * 100,
            "np.float64":    [np.float64(0.5)] * 200,
            "np.float32":    [np.float32(0.1)] * 200,
            "IntEnum":       [Mode.A] * 200,
            "past int64":    [2**64] * 200,
            "ragged":        [[1.0, 2.0]] * 100 + [[1.0]],
            "empty rows":    [[]] * 200,
            "strings":       ["a"] * 200,
        }
        for name, value in cases.items():
            with self.subTest(name):
                self.assertIsNone(_list_array(value))
                self.assertEqual(pretty_json({"v": value}), _per_value_json({"v": value}))

    def test_a_masked_array_writes_its_masked_values_as_null(self):
        array = np.ma.masked_array(np.arange(200.0), mask=np.arange(200) % 2 == 0)
        text  = pretty_json({"v": array})
        self.assertEqual(stdlib_json.loads(text)["v"][:3], [None, 1.0, None])
        self.assertEqual(text, _per_value_json({"v": array}))

    def test_a_longdouble_array_is_refused_as_before(self):
        # its tolist() keeps numpy scalars, which would be written as
        # np.longdouble('0.0'); on Windows it has a float64's size
        for shape in ((200,), (100, 3)):
            array = np.zeros(shape, np.longdouble)
            with self.subTest(shape=shape):
                self.assertFalse(utils._is_bulk_array(array))
                with self.assertRaisesRegex(TypeError, "longdouble"):
                    pretty_json({"v": array})

    def test_a_short_list_is_turned_down_without_a_pass_over_it(self):
        # pretty_json asks this of every list it writes: a short one, or one
        # of strings or dicts, is answered from its first value down
        class Watched(list):
            passes = 0

            def __iter__(self):
                Watched.passes += 1
                return super().__iter__()

        cases = {
            "3 floats":          Watched([1.0, 2.0, 3.0]),
            "60 floats":         Watched([1.0] * 60),
            "100 ints":          Watched([1] * 100),
            "20 rows of 3 ints": Watched([[1, 2, 3]] * 20),
            "200 strings":       Watched(["a"] * 200),
            "200 dicts":         Watched([{"a": 1}] * 200),
        }
        for name, value in cases.items():
            with self.subTest(name):
                Watched.passes = 0
                self.assertIsNone(_list_array(value))
                self.assertEqual(Watched.passes, 0)

    def test_short_arrays_and_lists_keep_the_per_value_path(self):
        # where it can be the quicker one: under 64 values, or 128 for a 1D
        # array of ints or bools
        for shape, dtype, bulk in (
            ((63,), float, False), ((64,), float, True),
            ((127,), int, False), ((128,), int, True),
            ((127,), bool, False), ((128,), bool, True),
            ((21, 3), int, False), ((16, 4), int, True),
            ((3, 3, 7), float, False), ((2, 4, 8), float, True),
        ):
            array = np.ones(shape, dtype)
            with self.subTest(shape=shape, dtype=dtype.__name__):
                self.assertEqual(utils._is_bulk_array(array), bulk)
                self.assertEqual(_list_array(array.tolist()) is not None, bulk)

    def test_a_list_that_holds_itself_raises_as_before(self):
        # the per-value path's recursion ends in a RecursionError; the bulk
        # writer's walk down a list's first values has to end too
        a = []
        a.append(a)
        b = []
        b.extend([b, b])
        for name, value in {"a = [a]": a, "b = [b, b]": b, "[a]": [a]}.items():
            with self.subTest(name):
                # in a thread, so a walk that never ends fails the test rather
                # than hangs it
                answer = []
                thread = threading.Thread(
                    target=lambda: answer.append(_list_array(value)), daemon=True
                )
                thread.start()
                thread.join(10)
                self.assertEqual(answer, [None])
                with self.assertRaises(RecursionError):
                    pretty_json({"v": value})

    def test_a_list_nested_deeper_than_numpy_holds_writes_as_before(self):
        # numpy holds 32 dimensions (64 in numpy 2), the per-value path any number
        for ndim in (33, 70):
            value = [0.5] * 200
            for _ in range(ndim - 1):
                value = [value]
            with self.subTest(ndim=ndim):
                self.assertIsNone(_list_array(value))
                self.assertEqual(pretty_json({"v": value}), _per_value_json({"v": value}))


class TestRunTestsFindsEverySuite(unittest.TestCase):
    """``run_tests`` must reach every ``_tests`` package, not only
    ``cgmath._tests``.  Each case runs one small transforms module and never the
    whole suite: a no-argument call from inside the suite would run itself."""

    @staticmethod
    def _run(target):
        import contextlib
        import io

        from cgmath.utils import run_tests

        with contextlib.redirect_stderr(io.StringIO()):
            return run_tests(target, verbosity=0)

    def test_a_short_name_reaches_the_transforms_suite(self):
        result = self._run("test_exports")
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 4)

    def test_the_package_relative_name_works(self):
        result = self._run("transforms._tests.test_exports.TestRootExports")
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 4)

    def test_a_glob_matches_in_every_suite(self):
        result = self._run("test_exports.py")
        self.assertEqual(result.testsRun, 4)

    def test_an_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            self._run("test_no_such_module_anywhere")
