import json as stdlib_json
import os
import tempfile
import unittest

import numpy as np
from cgmath.utils import (
    _escape_json_string,
    _get_shape,
    _is_flat_list,
    _is_uniform_nd,
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
    def test_nan_serializes_as_null(self):
        data   = {"val": float("nan")}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertIsNone(parsed["val"])

    def test_inf_serializes_as_string(self):
        data   = {"val": float("inf")}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed["val"], "Infinity")

    def test_negative_inf_serializes_as_string(self):
        data   = {"val": float("-inf")}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed["val"], "-Infinity")

    def test_control_characters_escaped(self):
        data   = {"msg": "null\x00byte\x1f"}
        result = pretty_json(data)
        parsed = stdlib_json.loads(result)
        self.assertEqual(parsed, data)


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
