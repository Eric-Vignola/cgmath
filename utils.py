from __future__ import annotations

import cProfile
import functools
import inspect
import json as original_json
import locale
import math
import os
import pstats
import re
import statistics
import sys
import tempfile
import unittest
from io import StringIO
from itertools import chain, repeat
from typing import Any, IO, Sequence

import numpy as np


# ----------------------------------------------- UTILS -----------------------------------------------#


def info(obj: Any) -> str:
    """Return the docstring of any Python object as a plain string.

    Works for classes, instances, functions, methods, and properties.
    Returns an empty string if no docstring is set.

    Examples:
        >>> from cgmath.geometry import MeshData
        >>> from cgmath.utils import info
        >>> print(info(MeshData))             # class docstring
        >>> mesh = MeshData(...)
        >>> print(info(mesh))                 # class docstring (via instance)
        >>> print(info(mesh.get_normals))     # bound-method docstring
        >>> print(info(MeshData.num_points))  # property docstring

    Args:
        obj: any Python object (class, instance, function, method, property, ...).

    Returns:
        The cleaned docstring (leading indentation stripped), or "" if none.
    """
    doc = inspect.getdoc(obj)
    return doc if doc is not None else ""


# friendly alias
docstring = info


# --------------------------------- PROFILER --------------------------------- #


def profile(cmd: str, n: int = 1) -> float:
    """
    simple speed profiler
    """

    def _profile(cmd):
        tmp_dir   = tempfile._get_default_tempdir()
        statsfile = os.path.join(tmp_dir, "statsfile")

        cProfile.run(cmd, statsfile)

        stream = StringIO()
        stats  = pstats.Stats(statsfile, stream=stream)
        stats.print_stats()
        stream = stream.getvalue()
        values = re.findall(r"[\d\.\d]+", stream.splitlines()[2])
        return float(values[-1])

    vals = []
    for _ in range(n):
        vals.append(_profile(cmd))

    return statistics.median(vals)


# -------------------------------- TEST RUNNER ------------------------------- #


class _ProgressResult(unittest.TextTestResult):
    """TextTestResult that prefixes each verbose test line with ``[N/total]``.

    ``startTest`` writes the counter and then defers to the stdlib for the
    description and `` ... ``, so the line format stays whatever this Python's
    unittest produces. The hook has the same shape on 3.9 (Maya 2023) and 3.11
    (Maya 2025). ``**kwargs`` absorbs the ``durations=`` that 3.12+ passes.
    Only active when ``showAll`` is set, i.e. verbosity 2 -- dot mode is untouched.
    """

    def __init__(self, stream, descriptions, verbosity, total=0, **kwargs):
        super().__init__(stream, descriptions, verbosity, **kwargs)
        self._total = total
        self._width = len(str(total))

    def startTest(self, test):
        if self.showAll:
            self.stream.write(
                "[%*d/%d] " % (self._width, self.testsRun + 1, self._total)
            )
        super().startTest(test)


def _run(suite, stream, verbosity, failfast):
    """Run ``suite`` with the progress-counting result class."""
    return unittest.TextTestRunner(
        stream    = stream,
        verbosity = verbosity,
        failfast  = failfast,
        resultclass=functools.partial(
            _ProgressResult, total=suite.countTestCases()
        ),
    ).run(suite)


def _encoding_safe(stream):
    """Wrap ``stream`` so nothing written through it can raise UnicodeEncodeError.

    Text is escaped against the platform default codec -- the one a downstream
    ``open()`` or ``logging.FileHandler`` uses when given no encoding -- before
    it reaches the real stream. Everything the codec can carry passes through
    untouched; anything it cannot is written as its escaped code point. On Windows
    that codec is cp1252, so em-dashes and degree signs still render and only true
    exotics are escaped.

    Delegation keeps whatever ``sys.stderr`` really is -- Maya's Script Editor,
    a studio log tee -- in the loop, so those still receive every line, now
    guaranteed encodable. Without this, one ``->`` written as U+2192 in a test
    docstring can turn into a recursive logging flood when the tee's handler
    cannot encode it and reports the failure back through the same stream.
    """
    codec = locale.getpreferredencoding(False) or "ascii"

    class _Safe:
        def write(self, text):
            return stream.write(text.encode(codec, "backslashreplace").decode(codec))

        def flush(self):
            return stream.flush()

        def __getattr__(self, name):
            return getattr(stream, name)

    return _Safe()


def run_tests(
    target:    str | Sequence[str] | None = None,
    verbosity: int                        = 2,
    failfast:  bool                       = False,
) -> unittest.TestResult:
    """
    runs the package's unit tests: every ``_tests`` suite in the library

    The suites are found by walking the package for ``_tests`` packages --
    ``cgmath._tests`` and ``cgmath.transforms._tests`` today -- so a subpackage
    that adds its own ``_tests`` is picked up with no change here.

    Args:
        target: what to run, coarsest to finest::

            None                    every suite
            "test_skin*.py"                                        a filename glob
            "test_geometry"                                        one module
            "test_geometry.TestMesh"                               one class
            "test_geometry.TestMesh.test_bitangent_unit_length"    one test method

            A glob matches in every suite.  A list or tuple runs several in one
            pass, which is the quick way to re-run a handful of failures.

            Short dotted names are looked up in every suite, so you never write
            the fully qualified ``cgmath._tests....`` or
            ``cgmath.transforms._tests....`` path.  The package-relative form
            (``transforms._tests.test_matrix``) and the long form copied out of
            a failure line work too.
        verbosity: 0 for silent, 1 for a dot per test, 2 for a line each.
        failfast: stop on the first failure or error.

    Returns:
        The :class:`unittest.TestResult`.  ``result.wasSuccessful()`` is the
        pass/fail answer; ``result.errors`` and ``result.failures`` carry the
        detail.

    Raises:
        ValueError: a dotted target naming no such module, class or method.
            Unittest would otherwise fold that into the run as an ordinary
            test error, which reads like a real failure rather than a typo.
            Also raised for a short name that more than one suite carries,
            rather than silently running only one of them.

    Note:
        Discovery imports every matching module, so a missing optional
        dependency shows up as an error against that module rather than
        aborting the run.

        >>> from cgmath.utils import run_tests
        >>> run_tests()                        # everything, both suites
        >>> run_tests("test_skin*.py")                                       # a glob
        >>> run_tests("test_geometry.TestMesh")                              # one class
        >>> run_tests("test_geometry.TestMesh.test_bitangent_unit_length")   # one test
        >>> run_tests(["test_rbf", "test_pack"])                             # two modules
        >>> run_tests("test_matrix")                                         # a transforms module
        >>> run_tests(verbosity=1, failfast=True)
    """
    package_root = os.path.dirname(os.path.abspath(__file__))
    package      = os.path.basename(package_root)

    # Every test package in the library: ``_tests`` itself plus the suite a
    # subpackage carries (``transforms/_tests``).  Found by walking the package,
    # so a subpackage that grows its own ``_tests`` joins the run with no list to
    # update.  The walk does not descend into a test package: its own
    # subpackages are searched through it, below.
    test_dirs: list[str] = []
    for dirpath, dirnames, filenames in os.walk(package_root):
        dirnames[:] = sorted(
            d for d in dirnames if d != "__pycache__" and not d.startswith(".")
        )
        if os.path.basename(dirpath) == "_tests" and "__init__.py" in filenames:
            test_dirs.append(dirpath)
            dirnames[:] = []

    if not test_dirs:
        raise FileNotFoundError(f"no _tests package under {package_root!r}")

    # top_level_dir is the package's PARENT, so modules import as
    # ``cgmath._tests.<name>`` / ``cgmath.transforms._tests.<name>`` rather than
    # as a bare ``<name>``.  The suites import their own siblings by that dotted
    # path, so pointing this at a ``_tests`` folder would break them.
    top_level_dir = os.path.dirname(package_root)
    roots         = [
        ".".join([package] + os.path.relpath(d, package_root).split(os.sep)) + "."
        for d in test_dirs
    ]
    loader = unittest.TestLoader()

    stream = _encoding_safe(sys.stderr)   # resolved now, so an installed stderr tee is seen

    if target is None:
        targets: list[str] = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    if not targets:
        suite = unittest.TestSuite()
        for start_dir in test_dirs:
            suite.addTests(
                loader.discover(start_dir, pattern="test_*.py", top_level_dir=top_level_dir)
            )
        return _run(suite, stream, verbosity, failfast)

    # Namespaces a short name may live in: every test package, plus any test
    # subpackage inside one.  Callers should not have to know which suite a
    # module belongs to, or whether it sits one directory down.
    namespaces: list[str] = []
    for start_dir, root in zip(test_dirs, roots):
        namespaces.append(root)
        namespaces += [
            root + entry + "."
            for entry in sorted(os.listdir(start_dir))
            if os.path.isfile(os.path.join(start_dir, entry, "__init__.py"))
        ]
    # The same roots relative to the package: "_tests.", "transforms._tests."
    relative_roots = [root[len(package) + 1 :] for root in roots]

    suite = unittest.TestSuite()
    for item in targets:
        if "*" in item or "?" in item or item.endswith(".py"):
            for start_dir in test_dirs:
                suite.addTests(
                    loader.discover(start_dir, pattern=item, top_level_dir=top_level_dir)
                )
            continue

        # accept the short "test_x...", the package-relative "_tests.test_x..."
        # or "transforms._tests.test_x...", and the fully qualified
        # "cgmath._tests.test_x..." that a failure line prints
        if item.startswith(package + "."):
            candidates = [item]
        elif any(item.startswith(rel) for rel in relative_roots):
            candidates = [package + "." + item]
        else:
            candidates = [ns + item for ns in namespaces]

        # Try EVERY namespace rather than stopping at the first hit: a short name
        # that two suites both carry must not silently run only one of them.
        resolved = []
        for candidate in candidates:
            mark  = len(loader.errors)
            found = loader.loadTestsFromName(candidate)
            if len(loader.errors) == mark:
                resolved.append((candidate, found))
            else:
                del loader.errors[mark:]  # discard the miss, try the next namespace

        if not resolved:
            raise ValueError(
                "could not resolve test target "
                + repr(item)
                + "; tried "
                + ", ".join(candidates)
            )
        if len(resolved) > 1:
            raise ValueError(
                "test target "
                + repr(item)
                + " is ambiguous; it names "
                + " and ".join(candidate for candidate, _ in resolved)
                + ". Pass the full name of the one you mean."
            )
        suite.addTests(resolved[0][1])

    return _run(suite, stream, verbosity, failfast)


# ------------------------------- PRETTY JSON -------------------------------- #
#   (recursive serializer with columnar array formatting; numeric arrays are
#   laid out a whole array at a time)

_JSON_ESCAPE_MAP: dict[str, str] = {
    "\\": "\\\\",
    '"':  '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}
# Control characters \x00-\x1f that are not covered by the map above
_JSON_CONTROL_RANGE: range = range(0x20)


def _escape_json_string(s: str) -> str:
    """Escape a string for JSON output (including control chars \\x00-\\x1f)."""
    parts: list[str] = []
    for c in s:
        mapped = _JSON_ESCAPE_MAP.get(c)
        if mapped is not None:
            parts.append(mapped)
        elif ord(c) in _JSON_CONTROL_RANGE or 0xD800 <= ord(c) <= 0xDFFF:
            # a lone surrogate has no utf-8 form: json's escape carries it
            parts.append(f"\\u{ord(c):04x}")
        else:
            parts.append(c)
    return "".join(parts)


_JSON_SCALARS = (int, float, str, bool, type(None), np.bool_, np.integer, np.floating, np.str_)


def _is_flat_list(obj: Any) -> bool:
    """True if obj is a list/tuple of all JSON-compatible scalars."""
    if not isinstance(obj, (list, tuple)):
        return False
    return all(isinstance(x, _JSON_SCALARS) for x in obj)


def _uniform_shape(obj: Any) -> tuple[int, ...] | None:
    """The shape of a rectangular nest of lists of scalars, or None."""
    if not isinstance(obj, (list, tuple)):
        return None
    if not obj:
        return (0,)
    if _is_flat_list(obj):
        return (len(obj),)
    if all(isinstance(x, (list, tuple)) for x in obj):
        # every row the same shape, depth included
        shapes = {_uniform_shape(x) for x in obj}
        if len(shapes) != 1 or None in shapes:
            return None
        return (len(obj),) + shapes.pop()
    return None


def _is_uniform_nd(obj: Any) -> bool:
    """Recursively check if a nested list has uniform shape (array-like)."""
    return _uniform_shape(obj) is not None


def _get_shape(obj: list | tuple) -> tuple[int, ...]:
    """Return the shape of a uniform nested list."""
    if not isinstance(obj, (list, tuple)) or not obj:
        return ()
    return _uniform_shape(obj) or ()


def _max_depth(obj: Any) -> int:
    """Return the maximum nesting depth of a list/tuple (non-uniform safe)."""
    if not isinstance(obj, (list, tuple)) or not obj:
        return 0
    if _is_flat_list(obj):
        return 1
    if all(isinstance(x, (list, tuple)) for x in obj):
        return 1 + max(_max_depth(x) for x in obj)
    return 1


def _float_to_str(val: float) -> str:
    """Convert a float to its JSON string representation.

    NaN and the infinities are written as the bare tokens Python's json reads
    back as floats; a string or null would come back as another type.
    """
    if val != val:  # nan
        return "NaN"
    if val == float("inf"):
        return "Infinity"
    if val == float("-inf"):
        return "-Infinity"
    return repr(float(val))  # a np.float64's own repr is np.float64(0.5)


def _decimal_align(str_items: list[str]) -> list[str]:
    """Align float strings at the decimal point with trailing space padding."""
    if not all("." in s for s in str_items):
        max_width = max(len(s) for s in str_items)
        return [s.rjust(max_width) for s in str_items]

    int_parts:  list[str] = []
    frac_parts: list[str] = []
    for s in str_items:
        dot = s.index(".")
        int_parts.append(s[:dot])
        frac_parts.append(s[dot + 1 :])

    max_int_len  = max(len(p) for p in int_parts)
    max_frac_len = max(len(p) for p in frac_parts)

    result: list[str] = []
    for ip, fp in zip(int_parts, frac_parts):
        result.append(ip.rjust(max_int_len) + "." + fp.ljust(max_frac_len))
    return result


def _pad_column(items: list, str_items: list[str]) -> list[str]:
    """Pad items for column alignment based on value types.

    Floats are decimal-aligned, strings are left-justified, and
    integers (or mixed types) are right-justified.
    """
    if not str_items:
        return str_items
    if all(isinstance(x, float) for x in items):
        return _decimal_align(str_items)
    if all(isinstance(x, str) for x in items):
        max_width = max(len(s) for s in str_items)
        return [s.ljust(max_width) for s in str_items]
    max_width = max(len(s) for s in str_items)
    return [s.rjust(max_width) for s in str_items]


def _scalar_to_str(x: Any) -> str:
    """Convert a scalar value to its JSON string representation."""
    if isinstance(x, np.generic):
        x = x.item()
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(int(x))  # str() of an int Enum is its name, Mode.A
    if isinstance(x, float):
        return _float_to_str(x)
    if isinstance(x, str):
        return f'"{_escape_json_string(str.__str__(x))}"'
    # str(x) would write a bare token json cannot read back
    raise TypeError(f"cannot write a {type(x).__name__} to json")


def _format_flat_list(
    items:          list,
    start_col:      int,
    max_line_width: int,
) -> str:
    """Format a 1D list of scalars with inline brackets and columnar wrapping.

    Brackets are kept inline with inner spacing: ``[ val, val, ... ]``.
    When wrapping is needed, every value is padded for columnar alignment
    so that commas align vertically across rows.  If wrapping would produce
    only two content rows the array collapses to a single line.
    """
    str_items = [_scalar_to_str(x) for x in items]
    single    = "[ " + ", ".join(str_items) + " ]"

    if start_col + len(single) <= max_line_width:
        return single

    padded    = _pad_column(items, str_items)
    max_width = max(len(s) for s in padded)

    align_col      = start_col + 2  # after "[ "
    available      = max_line_width - align_col
    items_per_line = max(1, (available + 2) // (max_width + 2))

    lines: list[str] = []
    for chunk_start in range(0, len(padded), items_per_line):
        chunk = padded[chunk_start : chunk_start + items_per_line]
        lines.append(", ".join(chunk))

    if len(lines) <= 2:
        return single

    pad   = " " * align_col
    parts = ["[ " + lines[0] + ","]
    for i in range(1, len(lines)):
        suffix = "," if i < len(lines) - 1 else " ]"
        parts.append(pad + lines[i] + suffix)
    return "\n".join(parts)


def _format_nd_array(
    obj:            list | tuple,
    start_col:      int,
    indent_size:    int,
    max_line_width: int,
) -> str:
    """Format a uniform array with inline brackets and column-aligned rows.

    For 2D arrays the first row sits inline after ``[`` and continuation
    rows align under the first row's inner ``[``.  For 3D+ arrays, blank
    lines are inserted between sub-arrays: ``ndim - 2`` blank lines at
    each nesting level, matching numpy's formatting convention.
    """
    shape = _get_shape(obj)

    if len(shape) == 1:
        return _format_flat_list(obj, start_col, max_line_width)

    if len(shape) == 2:
        str_rows = [[_scalar_to_str(x) for x in row] for row in obj]
        num_cols = len(str_rows[0])
        num_rows = len(str_rows)

        for c in range(num_cols):
            col_items = [obj[r][c] for r in range(num_rows)]
            col_strs  = [str_rows[r][c] for r in range(num_rows)]
            aligned   = _pad_column(col_items, col_strs)
            for r in range(num_rows):
                str_rows[r][c] = aligned[r]

        row_strs = []
        for str_row in str_rows:
            row_strs.append("[ " + ", ".join(str_row) + " ]")

        align_col = start_col + 1
        pad       = " " * align_col

        parts = []
        for i, row_str in enumerate(row_strs):
            prefix = "[" if i == 0 else pad
            suffix = "," if i < len(row_strs) - 1 else "]"
            parts.append(prefix + row_str + suffix)

        return "\n".join(parts)

    # 3D+: recurse with inline brackets and ndim-based blank line spacing
    blank_lines = len(shape) - 2
    align_col   = start_col + 1
    pad         = " " * align_col
    parts       = []
    for i, sub in enumerate(obj):
        formatted = _format_nd_array(sub, align_col, indent_size, max_line_width)
        prefix    = "[" if i == 0 else pad
        suffix    = "," if i < len(obj) - 1 else "]"
        parts.append(prefix + formatted + suffix)

    separator = "\n" * (blank_lines + 1)
    return separator.join(parts)


# The same text for a numeric array, laid out a whole array at a time: saving
# a 1000 x 1000 quad grid mesh to json takes 7.5 s through the per-value path
# above, 1.8 s through this one.

# the exact types a list may hold to be laid out in bulk, and their dtype
_BULK_LIST_DTYPES: dict[type, type] = {float: np.float64, int: np.int64, bool: np.bool_}

# the fewest values the bulk path takes: from here on it is the quicker of the
# two for every kind and shape of array measured, while under it the
# per-value path can be. That path writes a 1D column of ints or bools
# cheaply, so one of those needs twice as many.
_BULK_MIN_SIZE:        int = 64
_BULK_MIN_SIZE_1D_INT: int = 128  # a 1D array of ints or bools

# the most dimensions numpy 1 holds (numpy 2 holds 64): a list nested deeper
# is left to the per-value path
_BULK_MAX_NDIM: int = 32


def _is_bulk_array(array: np.ndarray) -> bool:
    """True if _format_array writes ``array`` as the per-value path writes its tolist()."""
    # not a longdouble, whose tolist() keeps numpy scalars (on Windows it has
    # a float64's size, so only its type tells the two apart), nor a masked
    # array, whose tolist() writes its masked values as None
    kind  = array.dtype.kind
    least = _BULK_MIN_SIZE_1D_INT if array.ndim == 1 and kind in "biu" else _BULK_MIN_SIZE
    return (
        array.size >= least
        and (kind in "biu" or (kind == "f" and array.dtype.type is not np.longdouble))
        and type(array).tolist is np.ndarray.tolist
    )


def _list_array(obj: list | tuple) -> np.ndarray | None:
    """A rectangular nest of lists of one exact number type as an array, or None.

    float, int and bool qualify. A mix of them, a subclass (an IntEnum,
    np.float64), a string or an empty row is left to the per-value path,
    which writes each of those its own way. ``obj`` is not empty.
    """
    # a quick no for most lists, from the first value down: a nest that
    # qualifies holds as many values as its first rows' lengths multiply to,
    # all of the first value's type. The walk stops at _BULK_MAX_NDIM, so on
    # a list that holds itself it ends too, and the per-value path raises its
    # RecursionError as it always did.
    ndim, size, first = 1, len(obj), obj[0]
    while type(first) in (list, tuple) and first and ndim < _BULK_MAX_NDIM:
        ndim += 1
        size *= len(first)
        first = first[0]
    least = _BULK_MIN_SIZE_1D_INT if ndim == 1 and type(first) in (int, bool) else _BULK_MIN_SIZE
    if size < least or type(first) not in _BULK_LIST_DTYPES:
        return None

    shape = [len(obj)]
    items = obj
    types = set(map(type, items))
    while types <= {list, tuple}:
        sizes = set(map(len, items))
        if len(sizes) != 1 or 0 in sizes:
            return None  # ragged, or an empty row
        shape.append(sizes.pop())
        items = list(chain.from_iterable(items))
        types = set(map(type, items))

    dtype = _BULK_LIST_DTYPES.get(types.pop()) if len(types) == 1 else None
    if dtype is None:
        return None
    try:
        return np.array(items, dtype=dtype).reshape(shape)
    except OverflowError:  # an int past int64
        return None


def _number_strings(array: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Every value of a bool, int or float array as json writes it, in C order,
    and the length of each, shaped like the array.
    """
    flat  = array.ravel()
    kind  = flat.dtype.kind
    table = None
    if kind == "b":
        table, index = ["false", "true"], flat.astype(np.intp)
    elif kind == "i" or (kind == "u" and flat.dtype.itemsize < 8):  # what int64 holds
        low, high = int(flat.min()), int(flat.max())
        if high - low < flat.size:
            # values that repeat (indices, counts): each one written once
            table = list(map(str, range(low, high + 1)))
            index = flat.astype(np.int64) - low

    if table is not None:
        lens = np.fromiter(map(len, table), np.intp, len(table))[index]
        return np.array(table, dtype=object)[index].tolist(), lens.reshape(array.shape)

    values = flat.tolist()
    if kind == "f":
        strs = list(map(repr, values))
        # NaN and the infinities: the bare tokens _float_to_str writes
        for i in np.flatnonzero(~np.isfinite(flat)).tolist():
            strs[i] = _float_to_str(values[i])
    else:
        strs = list(map(str, values))
    return strs, np.fromiter(map(len, strs), np.intp, len(strs)).reshape(array.shape)


def _format_array(
    array:          np.ndarray,
    start_col:      int,
    max_line_width: int,
) -> str:
    """The text _format_nd_array writes for ``array.tolist()``, laid out in bulk.

    Each value is written once, and numpy works out every column's padding:
    a 1D array is one column, a 2D one has its columns, and a 3D+ one the
    columns of each of its 2D blocks. What goes between two values (padding,
    a comma, line breaks, brackets) comes from a handful of distinct texts,
    each built once, and one join strings the lot together.
    """
    strs, lens = _number_strings(array)
    count = len(strs)

    # a column runs down axis -2, or is the whole of a 1D array (_pad_column)
    axis  = 0 if array.ndim == 1 else -2
    width = lens.max(axis, keepdims=True)
    left  = width - lens  # right-justified
    if array.dtype.kind == "f":
        # decimal-aligned in a column whose every value has a point (_decimal_align)
        dots  = np.fromiter(map(str.find, strs, repeat(".")), np.intp, count).reshape(array.shape)
        point = (dots >= 0).all(axis, keepdims=True)
        head  = dots.max(axis, keepdims=True)           # the widest part before the point
        tail  = (lens - dots).max(axis, keepdims=True)  # and from the point on
        left  = np.where(point, head - dots, left)
        width = np.where(point, head + tail, width)
    right = width - left - lens

    # what follows each value is texts[kinds[i]]: a comma, or a line's end
    kinds = np.zeros(count, np.intp)
    if array.ndim == 1:
        # one line if it fits, or if wrapped it would take two (_format_flat_list)
        single   = int(lens.sum()) + 2 * count + 2
        per_line = max(1, (max_line_width - start_col) // (int(width[0]) + 2))
        if start_col + single <= max_line_width or count <= 2 * per_line:
            return "[ " + ", ".join(strs) + " ]"

        texts = [", ", ",\n" + " " * (start_col + 2)]
        # a line ends after every per_line values
        kinds[per_line - 1 :: per_line] = 1
        opening, closing = "[ ", " ]"
    else:
        # a row ends with its bracket, those of the blocks it closes, a comma,
        # a line break and a blank line per level it climbs, then the opening
        # brackets of the next row (_format_nd_array); innermost level first,
        # so a block's end overwrites its last row's
        ndim  = array.ndim
        texts = [", "]
        ends  = kinds[array.shape[-1] - 1 :: array.shape[-1]]  # a view: one per row
        for level in range(ndim - 2, -1, -1):
            block                    = math.prod(array.shape[level + 1 : -1])
            closed                   = ndim - 2 - level
            ends[block - 1 :: block] = len(texts)
            texts.append(
                " ]" + "]" * closed + "," + "\n" * (closed + 1)
                + " " * (start_col + 1 + level) + "[" * closed + "[ "
            )
        opening, closing = "[" * (ndim - 1) + "[ ", " ]" + "]" * (ndim - 1)

    kinds[-1] = len(texts)
    texts.append(closing)

    # after each value: its right padding, what follows it and the next
    # value's left padding, numbered so each distinct gap is built once
    left, right = left.ravel(), right.ravel()
    size  = int(width.max()) + 1
    codes = (kinds * size + right) * size + np.append(left[1:], 0)
    gaps  = np.empty(len(texts) * size * size, dtype=object)
    for code in np.flatnonzero(np.bincount(codes)).tolist():
        kind, pads = divmod(code, size * size)
        gaps[code] = " " * (pads // size) + texts[kind] + " " * (pads % size)

    pieces       = [""] * (2 * count + 1)
    pieces[0]    = opening + " " * int(left[0])
    pieces[1::2] = strs
    pieces[2::2] = gaps[codes].tolist()
    return "".join(pieces)


def _serialize(
    obj:            Any,
    depth:          int,
    indent:         int,
    max_line_width: int,
    start_col:      int | None = None,
) -> str:
    """Recursively serialize a Python object to formatted JSON."""
    current_indent = depth * indent
    if start_col is None:
        start_col = current_indent
    pad       = " " * current_indent
    inner_pad = " " * ((depth + 1) * indent)

    if obj is None:
        return "null"
    if isinstance(obj, (bool, np.bool_)):
        return "true" if obj else "false"
    if isinstance(obj, (int, np.integer)):
        return str(int(obj))
    if isinstance(obj, (float, np.floating)):
        return _float_to_str(float(obj))
    if isinstance(obj, (str, np.str_)):
        # the text itself: str() of a str Enum is its name, Mode.FAST
        return f'"{_escape_json_string(str.__str__(obj))}"'
    if isinstance(obj, np.ndarray):
        if _is_bulk_array(obj):
            return _format_array(np.asarray(obj), start_col, max_line_width)
        obj = obj.tolist()
        if not isinstance(obj, list):  # a 0-d array
            return _serialize(obj, depth, indent, max_line_width, start_col)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            text    = str.__str__(k) if isinstance(k, str) else str(k)
            key_str = f'"{_escape_json_string(text)}": '
            val_col = (depth + 1) * indent + len(key_str)
            val_str = _serialize(v, depth + 1, indent, max_line_width, val_col)
            items.append(f"{inner_pad}{key_str}{val_str}")
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"

        array = _list_array(obj)
        if array is not None:
            return _format_array(array, start_col, max_line_width)

        # one look at the shape here, where _is_uniform_nd then _get_shape
        # took two (_format_nd_array takes its own)
        if _uniform_shape(obj) is not None:
            return _format_nd_array(obj, start_col, indent, max_line_width)

        # Non-uniform but all elements are lists -> inline brackets with spacing
        if all(isinstance(x, (list, tuple)) for x in obj):
            ndim        = _max_depth(obj)
            blank_lines = max(0, ndim - 2)
            align_col   = start_col + 1
            pad_str     = " " * align_col
            parts       = []
            for i, x in enumerate(obj):
                sub_str = _serialize(
                    x, depth + 1, indent, max_line_width, start_col=align_col
                )
                prefix = "[" if i == 0 else pad_str
                suffix = "," if i < len(obj) - 1 else "]"
                parts.append(prefix + sub_str + suffix)
            separator = "\n" * (blank_lines + 1)
            return separator.join(parts)

        items = []
        for x in obj:
            val_col = (depth + 1) * indent
            val_str = _serialize(x, depth + 1, indent, max_line_width, val_col)
            items.append(f"{inner_pad}{val_str}")
        return "[\n" + ",\n".join(items) + f"\n{pad}]"

    # str(obj) would write a bare token json cannot read back
    raise TypeError(f"cannot write a {type(obj).__name__} to json")


def pretty_json(
    data: dict | list | tuple, indent: int = 4, max_line_width: int = 80
) -> str:
    """Serialize data to human-readable JSON with columnar array formatting.

    Arrays of uniform shape are formatted with aligned columns, wrapping
    at ``max_line_width``. Works identically for numeric and string lists.
    A bool, int or float array, and a list nest holding one of those types,
    is laid out a whole array at a time, to the same text, once it holds 64
    values (128 for a 1D one of ints or bools).

    Args:
        data: The data to serialize (dict, list, or tuple).
        indent: Number of spaces per indentation level.
        max_line_width: Maximum line width before wrapping 1D arrays.

    Returns:
        A valid JSON string with human-friendly formatting.
    """
    return _serialize(data, 0, indent, max_line_width)


class json:
    """
    Provides a json module-like interface to read/write pretty_json
    formatted data.
    """

    @staticmethod
    def dumps(
        obj:            list | dict | tuple | np.ndarray,
        indent:         int                              = 4,
        max_line_width: int                              = 80,
    ) -> str:
        return pretty_json(obj, indent=indent, max_line_width=max_line_width)

    @staticmethod
    def dump(
        obj:            list | dict | tuple | np.ndarray,
        fp:             IO[str],
        indent:         int                              = 4,
        max_line_width: int                              = 80,
    ) -> None:
        fp.write(pretty_json(obj, indent=indent, max_line_width=max_line_width))

    @staticmethod
    def load(*args, **kwargs):
        return original_json.load(*args, **kwargs)

    @staticmethod
    def loads(*args, **kwargs):
        return original_json.loads(*args, **kwargs)


# ``python -m cgmath.utils`` runs the suite and exits non-zero on failure,
# which is what a CI step or a pre-commit hook wants.  An optional argument
# narrows the run: ``python -m cgmath.utils test_geometry.TestMesh``
if __name__ == "__main__":
    sys.exit(0 if run_tests(*sys.argv[1:2]).wasSuccessful() else 1)