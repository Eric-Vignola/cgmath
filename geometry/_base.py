import base64
import copy
import fnmatch
import importlib
import inspect
import io
import numbers
import os
import pickle
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterable, MutableSequence
from dataclasses import dataclass, fields, is_dataclass, MISSING
from enum import Enum
from typing import Annotated, Any, get_type_hints, List, Optional, Union

import numpy as np
from cgmath.utils import json, pretty_json


# -------------------------- DATACLASS TYPING HACK --------------------------- #
class ImmutableArray(np.ndarray):
    """A hack used to bypass a python 3.10+ limitation with dataclasses fields and numpy arrays."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __hash__(self):
        # this is the hack that lets us use numpy arrays in a dataclass field
        # a __post_init__ function will convert the ImmutableArray back to a numpy array
        return hash(bytes(self))


# ------------------------------- NPY FILE I/O ------------------------------- #


def _is_sequence(obj) -> bool:
    """tests if given input is a sequence"""
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict, set)):
        if isinstance(obj, np.ndarray):
            return obj.ndim > 0
        return True
    return False


def _is_numeric_or_boolean(arr: np.ndarray) -> bool:
    """a specific test to see if the contents of a numpy array is numeric or booleans"""
    return np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)


def _is_numeric_sequence(seq) -> bool:
    """tests if all elements in a sequence are numeric"""
    if not _is_sequence(seq):
        return False

    for obj in seq:
        if _is_sequence(obj):
            if not _is_numeric_sequence(obj):
                return False
        elif not isinstance(obj, numbers.Real):
            return False

    return True


def _mask_to_indices(query, length: int):
    """converts a boolean mask to positional indices, anything else passes through

    numpy booleans coerce to the indices 0 and 1 when used to index a python
    list, so a mask reaching a positional lookup silently returns the first two
    elements instead of a selection.
    """
    if (
        isinstance(query, (list, tuple))
        and len(query) > 0
        and isinstance(query[0], (bool, np.bool_))
    ):
        query = np.asarray(query)

    if isinstance(query, np.ndarray) and query.dtype == np.bool_:
        if query.size != length:
            raise IndexError(
                f"boolean mask of size {query.size} does not match "
                f"a list of length {length}"
            )
        return np.flatnonzero(query)

    return query


# --------------------------------- FILES ------------------------------------ #
#
# A cgmath file, pkl, npz or json, holds one tree: the dict to_dict() returns.
# Its "__class__" names the class to rebuild. A file without one was written by
# an older cgmath and is refused.

CLASS_KEY   = "__class__"
ITEMS_KEY   = "__items__"
DTYPES_KEY  = "__dtypes__"
SHAPES_KEY  = "__shapes__"

_CLASSES    = {}  # every Data / DataList class, by its tag

_OLDER_FILE = "{} was saved by an older cgmath, whose files this version no longer reads"


class LegacyFileError(ValueError):
    """a file saved by an older cgmath, whose files are no longer read"""


def _class_tag(cls) -> str:
    """the name a class is saved under: its module and qualified name"""
    # a spawned worker runs the script as __mp_main__; the script is __main__
    module = "__main__" if cls.__module__ == "__mp_main__" else cls.__module__
    return f"{module}.{cls.__qualname__}"


def _register(cls) -> None:
    _CLASSES[_class_tag(cls)] = cls


def _derives(cls, base) -> bool:
    """
    Whether ``cls`` is ``base`` or derives from it, by name: a class and its
    copy from a reloaded module are one class to a file.
    """
    tag = _class_tag(base)
    return any(_class_tag(x) == tag for x in cls.__mro__)


def _imported_class(tag: str):
    """the class a tag names in a module already imported, or None"""
    parts = tag.split(".")
    for split in range(len(parts) - 1, 0, -1):
        # a spawned worker runs the script as __mp_main__, also known as __main__
        found = sys.modules.get(".".join(parts[:split]))
        if found is None:
            continue
        for name in parts[split:]:
            found = getattr(found, name, None)
        if isinstance(found, type) and issubclass(found, (Data, DataList)):
            return found
        return None
    return None


def _resolve_class(tag: Optional[str], expected=None):
    """
    The class a saved tree names, which must be ``expected`` or derive from
    it; ``expected`` itself when the tree names it. Only cgmath modules are
    imported to find one: a class from anywhere else must already be imported.
    """
    if tag is None:
        if expected is None:
            raise ValueError("the data names no class")
        return expected
    if expected is not None and tag == _class_tag(expected):
        return expected  # even when a reload has registered a newer copy

    found = _CLASSES.get(tag)
    if found is None:
        module = tag.rpartition(".")[0]
        if module == "cgmath" or module.startswith("cgmath."):
            importlib.import_module(module)
            found = _CLASSES.get(tag)
    if found is None:
        found = _imported_class(tag)

    if found is None:
        raise TypeError(f"the data names {tag!r}, which is not a known cgmath data class")
    if expected is not None and not _derives(found, expected):
        raise TypeError(f"the data holds a {found.__name__}, not a {expected.__name__}")
    return found


def _from_tagged(data: dict, expected=None):
    """the object a tagged tree describes"""
    return _resolve_class(data.get(CLASS_KEY), expected).from_dict(data)


# pickle: the tree to_dict() returns, its arrays as their raw bytes. A pickle
# then holds no numpy object, so one written under a numpy reads under another,
# and the class comes by reference, so pickle imports its module.


class _PickledArray:
    """an array going into a pickle, as its dtype, shape and bytes"""

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray):
        self.array = array

    def __reduce_ex__(self, protocol):
        array = self.array
        if not array.flags.c_contiguous:
            array = array.copy(order="C")  # np.ascontiguousarray makes a 0-d array 1-d

        if protocol >= 5:
            # as bytes, which every dtype can give (a datetime64 has no buffer)
            data = pickle.PickleBuffer(array.reshape(-1).view(np.uint8))
        else:
            data = array.tobytes()
        return (_unpickle_array, (np.lib.format.dtype_to_descr(array.dtype), array.shape, data))


def _unpickle_array(descr, shape: tuple, data) -> np.ndarray:
    dtype = np.lib.format.descr_to_dtype(descr)
    # data is bytes, or a buffer handed to pickle.loads(buffers=...)
    if dtype.itemsize == 0 or memoryview(data).nbytes == 0:
        return np.zeros(shape, dtype)

    array = np.frombuffer(data, dtype=dtype)
    if not array.flags.writeable:
        array = array.copy()
    return array.reshape(shape)


def _pickled(value):
    """
    A tree ready to pickle: arrays as _PickledArray, numpy scalars as python
    ones, as npz and json give them back, and each tagged tree in its class's
    own pickle form (a skin keeps only its nonzero weights).
    """
    if isinstance(value, np.ndarray):
        return value if value.dtype.hasobject else _PickledArray(value)
    if isinstance(value, dict):
        tag = value.get(CLASS_KEY)
        cls = _CLASSES.get(tag) if isinstance(tag, str) else None
        if cls is not None:
            value = cls._pickle_tree(dict(value))
        return {_pickled(key): _pickled(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pickled(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pickled(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpickle(cls, tree: dict):
    """unpickles a Data or DataList from its tree"""
    return cls.from_dict(tree)


# a plain dict that holds a "__class__" key of its own is saved as its pairs,
# under this tag, so a load does not take it for a saved object
_PLAIN_DICT = "~dict"


def _encode(value):
    """``value`` as plain data: a nested Data or DataList becomes its tree"""
    if isinstance(value, (Data, DataList)):
        return value.to_dict()
    if isinstance(value, Enum):
        # its value, in every format alike; a field declaring the Enum restores it
        return _encode(value.value)
    if isinstance(value, dict):
        if CLASS_KEY in value:
            return {CLASS_KEY: _PLAIN_DICT, "pairs": [[key, _encode(item)] for key, item in value.items()]}
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_encode(item) for item in value)
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _tree_copy(value):
    """a deep copy of a tree, numpy scalars as python ones, as every file gives them back"""
    if isinstance(value, dict):
        return {_tree_copy(key): _tree_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    return copy.deepcopy(value)


def _hashable(key):
    """a key read back: json gives a tuple key back as a list"""
    return tuple(_hashable(x) for x in key) if isinstance(key, list) else key


def _decode(value):
    """a saved value back as objects: a tagged dict becomes its class"""
    if isinstance(value, dict):
        if CLASS_KEY in value:
            if value[CLASS_KEY] == _PLAIN_DICT:
                return {_hashable(key): _decode(item) for key, item in value["pairs"]}
            return _from_tagged(value)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_decode(item) for item in value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                return [_decode(item) for item in value]
        return list(value)  # plain values, a json row of numbers: nothing to rebuild
    return value


# npz: one .npy member per array or scalar, at the path of its keys. Nothing is
# pickled, so files are read with allow_pickle=False. Markers carry what .npy
# members cannot: the keys holding None, an empty dict, a list or a tuple.
_NONE     = "~none"   # the keys, or list positions, holding None
_EMPTY    = "~empty"  # an empty dict
_LIST     = "~list"   # [length, is a tuple, packed]
_PACK     = "~pack"   # a packed list: plain scalars of one type, as one array
_ORDER    = "~order"  # a dict's keys, in order, when some hold None
_KEYS     = "~keys"   # a dict whose keys a member name cannot carry, as two lists
_VALUES   = "~values"

_COLS     = "~cols"   # rows packed as one array per column
_TEXT     = "~text"   # text holding a NUL, which a numpy string drops at its end, as utf-8
_INT      = "~int"    # an int beyond 64 bits, as its digits
_BYTES    = "~bytes"  # bytes, as uint8
_ZERO_D   = "~0d"     # a 0-d array, which a 0-d member would read back as a scalar

_PACKABLE = (bool, int, float, str)


def _npy(zipf, path: str, value) -> None:
    # past 2 GiB a member needs zip64, which has to be asked for up front
    big = getattr(value, "nbytes", 0) > 1 << 30
    with zipf.open(path + ".npy", "w", force_zip64=big) as f:
        np.save(f, value, allow_pickle=False)


def _plain_array(items) -> Optional[np.ndarray]:
    """items, all of one plain scalar type, as one array; None when they do not fit one"""
    kind = type(items[0])
    if kind not in _PACKABLE or not all(type(x) is kind for x in items):
        return None
    if kind is str and any("\x00" in x for x in items):
        return None
    return _array_of(items, kind)


# the dtype kinds each plain scalar type packs into
_KINDS = {bool: "b", int: "iu", float: "f", str: "U"}


def _array_of(value, kind) -> Optional[np.ndarray]:
    """value, of scalars of one type ``kind``, as an array of that kind; None when numpy would change them"""
    try:
        array = np.array(value)
    except (OverflowError, ValueError):
        return None  # an int too large for int64
    # an int64 and a uint64 together make float64, which rounds them
    return array if array.dtype.kind in _KINDS[kind] else None


def _member_key(key) -> bool:
    """whether a dict key can name a member: a str a zip keeps as it is"""
    if (
        type(key) is not str
        or key == ""
        or key.startswith("~")
        or "/" in key
        or "\\" in key  # a zip on Windows reads it as /
        or "\x00" in key  # a zip ends the name there
    ):
        return False
    try:
        key.encode("utf-8")  # a zip names its members in utf-8: no lone surrogate
    except UnicodeEncodeError:
        return False
    return True


def _pack_shape(value) -> Optional[tuple]:
    """the shape of a rectangular nest of lists of one plain scalar type, or None"""
    if not value:
        return None

    first = value[0]
    kind  = type(first)
    if kind in _PACKABLE:
        if all(type(x) is kind for x in value):
            return (len(value),), kind
        return None

    if kind is not list:
        return None  # an inner tuple would come back a list
    inner = _pack_shape(first)
    if inner is None:
        return None
    for item in value[1:]:
        if type(item) is not list or _pack_shape(item) != inner:
            return None
    return (len(value),) + inner[0], inner[1]


def _packed_array(value) -> Optional[np.ndarray]:
    """a list's packed array, or None when it does not pack"""
    shape = _pack_shape(value)
    if shape is None:
        return None
    if shape[1] is str:
        flat = value
        while type(flat[0]) is list:
            flat = [x for row in flat for x in row]
        if any("\x00" in x for x in flat):
            return None  # a numpy string drops a NUL at its end
    return _array_of(value, shape[1])


def _columns(value) -> Optional[list]:
    """
    The columns of a list of rows of one length, all lists or all tuples,
    each column of one plain scalar type: [index, weight] pairs, (x, y)
    tuples. None when they do not pack that way.
    """
    first = value[0] if value else None
    row   = type(first)
    if row not in (list, tuple) or not first:
        return None
    width = len(first)
    if not all(type(x) is row and len(x) == width for x in value):
        return None

    columns = []
    for column in zip(*value):
        array = _plain_array(column)
        if array is None:
            return None
        columns.append(array)
    return columns


def _npz_put(zipf, path: str, value) -> None:
    """writes ``value`` at ``path``"""

    def at(key) -> str:
        return f"{path}/{key}" if path else str(key)

    if isinstance(value, dict):
        if not value:
            _npy(zipf, at(_EMPTY), np.bool_(True))
            return

        if not all(_member_key(key) for key in value):
            _npz_put(zipf, at(_KEYS), list(value))
            _npz_put(zipf, at(_VALUES), list(value.values()))
            return

        nones = [key for key, item in value.items() if item is None]
        if nones:
            _npy(zipf, at(_NONE), np.array(nones))
            # None keys have no member to hold their place
            _npy(zipf, at(_ORDER), np.array(list(value)))

        for key, item in value.items():
            if item is not None:
                _npz_put(zipf, at(key), item)

    elif isinstance(value, (list, tuple)):
        # [length, is a tuple, how it packs: 0 item by item, 1 one array, 2 by
        # column], and for columns, whether the rows are tuples
        array   = _packed_array(value)
        columns = None if array is not None else _columns(value)
        packing = 1 if array is not None else 2 if columns is not None else 0
        marker  = [len(value), isinstance(value, tuple), packing]
        if columns is not None:
            marker.append(type(value[0]) is tuple)
        _npy(zipf, at(_LIST), np.array(marker))

        if array is not None:
            _npy(zipf, at(_PACK), array)
            return
        if columns is not None:
            for index, column in enumerate(columns):
                _npy(zipf, f"{at(_COLS)}/{index}", column)
            return

        nones = [str(index) for index, item in enumerate(value) if item is None]
        if nones:
            _npy(zipf, at(_NONE), np.array(nones))
        for index, item in enumerate(value):
            if item is not None:
                _npz_put(zipf, at(index), item)

    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"cannot save an object array to npz ({path})")
        if value.ndim == 0:
            _npy(zipf, at(_ZERO_D), value.reshape(1))
        else:
            _npy(zipf, path, value)

    elif isinstance(value, str):
        value = str.__str__(value)  # the text of a str subclass (a str Enum)
        if "\x00" in value:
            _npy(zipf, at(_TEXT), np.frombuffer(value.encode("utf-8", "surrogatepass"), np.uint8))
        else:
            _npy(zipf, path, np.asarray(value))

    elif isinstance(value, int) and not isinstance(value, bool) and not -(1 << 63) <= value < 1 << 64:
        _npy(zipf, at(_INT), np.asarray(str(int(value))))

    elif isinstance(value, (bool, int, float, complex, np.generic)):
        _npy(zipf, path, np.asarray(value))

    elif isinstance(value, bytes):
        _npy(zipf, at(_BYTES), np.frombuffer(value, np.uint8))

    else:
        raise TypeError(f"cannot save a {type(value).__name__} to npz ({path})")


def _npz_value(node):
    """a read member, or a tree of them, as what was saved"""
    if not isinstance(node, dict):
        # a 0-d member is the scalar it was saved from
        return node.item() if node.ndim == 0 else node

    if _EMPTY in node:
        return {}
    if _KEYS in node:
        return dict(zip(_npz_value(node[_KEYS]), _npz_value(node[_VALUES])))
    if _TEXT in node:
        return node[_TEXT].tobytes().decode("utf-8", "surrogatepass")
    if _INT in node:
        return int(node[_INT].item())
    if _BYTES in node:
        return node[_BYTES].tobytes()
    if _ZERO_D in node:
        return node[_ZERO_D].reshape(())

    nones = node.pop(_NONE, None)
    nones = [] if nones is None else nones.tolist()
    order = node.pop(_ORDER, None)

    if _LIST in node:
        marker = [int(x) for x in node.pop(_LIST)]
        length, is_tuple, packing = marker[:3]
        rows = len(marker) > 3 and marker[3]
        if packing == 1:
            items = node.pop(_PACK).tolist()
        elif packing == 2:
            columns = node.pop(_COLS)
            columns = [columns[str(index)].tolist() for index in range(len(columns))]
            items   = [tuple(row) if rows else list(row) for row in zip(*columns)]
        else:
            items = [None] * length
            for key, child in node.items():
                items[int(key)] = _npz_value(child)
        return tuple(items) if is_tuple else items

    out = {key: _npz_value(child) for key, child in node.items()}
    for key in nones:
        out[key] = None
    if order is not None:
        out = {key: out[key] for key in order.tolist()}
    return out


def _write_whole(filename: str, write) -> None:
    """
    Calls ``write`` with a scratch binary file, and only once it returns
    writes what it wrote to ``filename``: a save that fails part way leaves
    the file as it was. The file is written in place, as a plain write does,
    so one open elsewhere, a link or a device takes it as before.
    """
    # in memory up to 256 MB, on disk past that
    with tempfile.SpooledTemporaryFile(max_size=1 << 28) as scratch:
        write(scratch)
        scratch.seek(0)
        with open(filename, "wb") as f:
            shutil.copyfileobj(scratch, f, 1 << 24)


def _zip_tree(data: dict, file, compression) -> None:
    with zipfile.ZipFile(file, "w", compression=compression) as zipf:
        _npz_put(zipf, "", data)


def dict_to_zip(data: dict, filename, compression=zipfile.ZIP_DEFLATED) -> None:
    """writes a tree of dicts, lists, arrays and scalars as an npz file"""
    if isinstance(filename, (str, os.PathLike)):
        _write_whole(os.fspath(filename), lambda f: _zip_tree(data, f, compression))
    else:
        _zip_tree(data, filename, compression)


def zip_to_dict(filename) -> dict:
    """reads back the tree :func:`dict_to_zip` wrote"""
    tree = {}
    with zipfile.ZipFile(filename, "r") as zipf:
        for member in zipf.namelist():
            if not member.endswith(".npy"):
                continue
            keys = member[:-4].split("/")
            node = tree
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            with zipf.open(member) as f:
                node[keys[-1]] = np.load(f, allow_pickle=False)
    return _npz_value(tree)


def dict_to_bytes(data: dict, compression=zipfile.ZIP_DEFLATED) -> bytes:
    """the npz file of a tree, as bytes"""
    buffer = io.BytesIO()
    dict_to_zip(data, buffer, compression=compression)
    return buffer.getvalue()


def bytes_to_dict(data: bytes) -> dict:
    """the tree in npz bytes"""
    return zip_to_dict(io.BytesIO(data))


def _tagged(data, filename: str) -> dict:
    """``data`` read from ``filename``, refused unless it names its class"""
    if not isinstance(data, dict) or CLASS_KEY not in data:
        raise LegacyFileError(_OLDER_FILE.format(filename))
    return data


def _json_tree(value):
    """
    A tree as json writes it: a dict whose keys json would turn into text (an
    int, a tuple) or that holds one of the markers, as its keys and values.
    """
    if isinstance(value, dict):
        if all(type(key) is str and not key.startswith("~") for key in value):
            return {key: _json_tree(item) for key, item in value.items()}
        return {
            _KEYS:   [_json_tree(key) for key in value],
            _VALUES: [_json_tree(item) for item in value.values()],
        }
    if isinstance(value, (list, tuple)):
        if not any(isinstance(item, (dict, list, tuple, bytes)) for item in value):
            return value
        return [_json_tree(item) for item in value]  # json writes a tuple as a list
    if isinstance(value, bytes):
        return {_BYTES: base64.b64encode(value).decode("ascii")}
    return value


def _json_dict(data: dict) -> dict:
    """json.load's object_hook: undoes _json_tree"""
    if len(data) == 2 and _KEYS in data and _VALUES in data:
        return dict(zip((_hashable(key) for key in data[_KEYS]), data[_VALUES]))
    if len(data) == 1 and isinstance(data.get(_BYTES), str):
        return base64.b64decode(data[_BYTES])
    return data


def _read_json(filename: str) -> dict:
    filename = os.path.expanduser(filename)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return _tagged(json.load(f, object_hook=_json_dict), filename)
    except UnicodeDecodeError:
        # cgmath writes utf-8; an older one wrote the locale's encoding
        raise LegacyFileError(_OLDER_FILE.format(filename)) from None


def _read_npz(filename: str) -> dict:
    filename = os.path.expanduser(filename)
    with zipfile.ZipFile(filename, "r") as zipf:
        if CLASS_KEY + ".npy" not in zipf.namelist():
            raise LegacyFileError(_OLDER_FILE.format(filename))
    return zip_to_dict(filename)


def _read_pickle(cls, filename: str):
    filename = os.path.expanduser(filename)
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
    except LegacyFileError:
        raise LegacyFileError(_OLDER_FILE.format(filename)) from None

    if not _derives(type(obj), cls):
        raise TypeError(f"{filename} holds a {type(obj).__name__}, not a {cls.__name__}")
    return obj


def _save_file(obj, filename: str, mode: Optional[str] = None, compression=zipfile.ZIP_DEFLATED) -> str:
    """writes ``obj`` as ``mode``, or as the file's extension says"""
    filename = os.path.expanduser(filename)
    if mode is None:
        mode = os.path.splitext(filename)[1].lstrip(".")
    mode = {"pickle": "pkl"}.get(mode.lower(), mode.lower())

    if mode == "pkl":
        _write_whole(filename, lambda f: pickle.dump(obj, f))
    elif mode == "json":
        # utf-8 and LF whatever the platform: the same file everywhere
        text = obj.to_json().encode("utf-8")
        _write_whole(filename, lambda f: f.write(text))
    elif mode == "npz":
        dict_to_zip(obj.to_dict(), filename, compression=compression)
    else:
        raise ValueError(f"cannot save {mode!r} files: use .pkl, .npz or .json")
    return filename


# ------------------------------ ARRAY FIELDS -------------------------------- #


class ArraySpec:
    """
    What an array field holds: one dtype, or a few that are kept as given,
    and a shape in which an int fixes a dimension and a name ("N") leaves it
    free. No shape leaves every dimension free.
    """

    __slots__ = ("dtypes", "shape", "_fixed")

    def __init__(self, dtypes, shape):
        dtypes      = dtypes if isinstance(dtypes, (tuple, list)) else (dtypes,)
        self.dtypes = tuple(np.dtype(x) for x in dtypes)
        self.shape  = tuple(shape) if shape else None
        # the fixed dimensions, as (axis, size)
        self._fixed = tuple((i, x) for i, x in enumerate(self.shape or ()) if isinstance(x, int))

    def describe(self) -> str:
        """the shape as written in an error: ``(N, 3)``"""
        dims = ", ".join(str(x) for x in self.shape)
        return f"({dims},)" if len(self.shape) == 1 else f"({dims})"

    def __repr__(self) -> str:
        dtypes = "/".join(x.name for x in self.dtypes)
        return f"Array({dtypes}{'' if self.shape is None else ' ' + self.describe()})"

    def _dtype_for(self, value) -> np.dtype:
        if len(self.dtypes) == 1:
            return self.dtypes[0]

        array = np.asarray(value)
        if array.dtype in self.dtypes:
            return array.dtype

        # the first of the declared dtypes of the same kind that holds the values
        if array.dtype.kind == "f":
            fits = [x for x in self.dtypes if x.kind == "f"]
        else:
            ints = [x for x in self.dtypes if x.kind in "iu"]
            fits = ints
            if array.size and ints:
                low, high = array.min(), array.max()
                fits = [x for x in ints if np.iinfo(x).min <= low and high <= np.iinfo(x).max]
                if not fits:
                    # a cast would wrap them: 70000 to 112 in a uint8
                    names = "/".join(x.name for x in ints)
                    raise ValueError(f"values from {low} to {high} fit none of {names}")
        return (fits or self.dtypes)[0]

    def coerce(self, value, where: str) -> np.ndarray:
        """``value`` as this field's array; ``where`` names it in errors"""
        if type(value) is np.ndarray and value.dtype in self.dtypes:
            array = value  # already one of this field's: nothing to convert
        else:
            if isinstance(value, ImmutableArray):
                value = np.array(value)  # a class default: this object's own copy
            try:
                array = np.asarray(value, dtype=self._dtype_for(value))
            except ValueError as error:
                raise ValueError(f"{where}: {error}") from None
        if self.shape is None:
            return array

        if array.ndim != len(self.shape):
            if array.size == 0:
                return array.reshape(tuple(x if isinstance(x, int) else 0 for x in self.shape))
            raise ValueError(f"{where} must have shape {self.describe()}, got {array.shape}")

        shape = array.shape
        for axis, size in self._fixed:
            if shape[axis] != size:
                raise ValueError(f"{where} must have shape {self.describe()}, got {shape}")
        return array


def Array(dtype, *shape):
    """
    The annotation of an array field. ``Array(np.int32, "N")`` is a list of
    indices, ``Array(np.float64, "N", 3)`` points, ``Array(np.float64, 4, 4)``
    a matrix. A value is converted to the dtype when it is set and when it is
    loaded, so a file reads back the same from pkl, npz and json. A fixed
    dimension that does not match raises ValueError; an empty value takes the
    declared trailing dimensions. A tuple of dtypes keeps whichever of them
    the value has (a texture stays uint8, uint16 or float32).
    """
    return Annotated[np.ndarray, ArraySpec(dtype, shape)]


def _array_spec(annotation) -> Optional[ArraySpec]:
    """the ArraySpec of ``Array(...)`` or ``Optional[Array(...)]``"""
    metadata = getattr(annotation, "__metadata__", None)
    if metadata:
        return next((x for x in metadata if isinstance(x, ArraySpec)), None)

    if getattr(annotation, "__origin__", None) is Union or type(annotation).__name__ == "UnionType":
        for arg in annotation.__args__:
            spec = _array_spec(arg)
            if spec is not None:
                return spec
    return None


def _is_union(annotation) -> bool:
    """``Union[...]``, ``Optional[...]`` or ``X | Y``"""
    return getattr(annotation, "__origin__", None) is Union or type(annotation).__name__ == "UnionType"


def _as_declared(value, annotation):
    """
    A loaded value as its field declares it, where a file cannot say: a tuple
    (json reads one back as a list), an Enum member (every file keeps its
    value), an array (json reads one back as lists), each nested in tuples
    and lists as declared. Anything else is left as it is.
    """
    if value is None or annotation is None:
        return value

    if _is_union(annotation):
        arms = [x for x in annotation.__args__ if x is not type(None)]
        # the first arm the value already is, so an int stays an int in
        # Union[int, Color]; then the first one that can make it what it declares
        for arm in arms:
            if _fits(value, arm):
                return _as_declared(value, arm)
        for arm in arms:
            declared = _as_declared(value, arm)
            if declared is not value:
                return declared
        return value

    if annotation is np.ndarray or _array_spec(annotation) is not None:
        if isinstance(value, np.ndarray) or not isinstance(value, (list, tuple, bool, int, float, complex)):
            return value
        spec = _array_spec(annotation)
        return np.asarray(value) if spec is None else spec.coerce(value, "a nested array")

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError):
            return value

    origin = getattr(annotation, "__origin__", None)
    args   = getattr(annotation, "__args__", None) or ()
    if annotation is tuple and isinstance(value, list):
        return tuple(value)
    if origin is tuple and isinstance(value, (list, tuple)):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_as_declared(x, args[0]) for x in value)
        if len(args) == len(value):
            return tuple(_as_declared(x, arg) for x, arg in zip(value, args))
        return tuple(value)
    if (annotation is list or origin is list) and isinstance(value, (tuple, np.ndarray)):
        # a list given as an array or tuple: npz and pkl would keep it, json not
        value = value.tolist() if isinstance(value, np.ndarray) else list(value)
    if origin is list and isinstance(value, list) and args:
        declared = [_as_declared(x, args[0]) for x in value]
        return value if all(x is y for x, y in zip(declared, value)) else declared
    return value


def _fits(value, annotation) -> bool:
    """whether value already is the kind an annotation declares, whatever it holds"""
    if annotation is Any:
        return True
    if _is_union(annotation):
        return any(_fits(value, x) for x in annotation.__args__)
    kind = getattr(annotation, "__origin__", None) or annotation
    try:
        return isinstance(value, kind)
    except TypeError:
        return False


def _holds_data(annotation) -> bool:
    """whether a field is declared a Data or DataList (or None) and nothing else"""
    if getattr(annotation, "__origin__", None) is Union or type(annotation).__name__ == "UnionType":
        kinds = [x for x in annotation.__args__ if x is not type(None)]
    else:
        kinds = [annotation]
    return bool(kinds) and all(isinstance(x, type) and issubclass(x, (Data, DataList)) for x in kinds)


_NO_DEFAULT = object()


def _field_default(field):
    """a field's default, made fresh, or _NO_DEFAULT"""
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return _NO_DEFAULT


def _same_value(a, b) -> bool:
    """whether ``a`` is exactly ``b``: the test for a field left at its default"""
    if a is b:
        return True
    if isinstance(a, (Data, DataList)) or isinstance(b, (Data, DataList)):
        # == on data is within tolerance and skips EQUALITY_TEST_IGNORE (a name)
        return type(a) is type(b) and _same_value(a.to_dict(), b.to_dict())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)
    try:
        return type(a) is type(b) and bool(a == b)
    except (TypeError, ValueError):
        return False


def _close(a, b) -> bool:
    """whether two field values are equal, floats to within np.allclose"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            a, b = np.asarray(a), np.asarray(b)
        except (TypeError, ValueError):
            return False
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            try:
                return bool(np.allclose(a, b, equal_nan=True))
            except TypeError:
                return False
        return bool(np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real) and a != a and b != b:
        return True  # both NaN
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def get_file_type(filename: str) -> str:
    """
    The file's type: ``"npz"``, ``"json"`` or ``"pkl"`` for cgmath's own
    files, ``"glb"``, ``"fbx"``, ``"obj"`` or ``"usd"`` for assets.

    ``.gltf``, ``.usd``, ``.usda``, ``.usdc`` and ``.usdz`` go by their
    extension; then the header decides (a binary glTF, an FBX, a numpy zip,
    json), then ``.obj``; anything else is taken as a pickle.
    """
    # by extension first: a .gltf is json, a .usdz is a zip like an npz
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".gltf":
        return "glb"
    if ext in (".usd", ".usda", ".usdc", ".usdz"):
        return "usd"

    with open(filename, "rb") as file:
        header = file.read(18)

    if header[:4] == b"PK\x03\x04":
        return "npz"
    if header[:4] == b"glTF":
        return "glb"
    if b" FBX " in header:
        return "fbx"
    if header.lstrip()[:1] in (b"{", b"["):
        return "json"
    if ext == ".obj":
        return "obj"
    return "pkl"


def _load_as(cls, filename: str, mode: Optional[str] = None) -> Any:
    """
    ``cls.load()``: the reader for ``mode``, or for the file's type
    (:func:`get_file_type`). Assets go to the class's own ``load_glb`` /
    ``load_fbx`` / ``load_obj`` / ``load_usd``.

    Raises
    ------
    ValueError
        The class has no reader for that type.
    """
    filename = os.path.expanduser(filename)
    mode     = get_file_type(filename) if mode is None else mode.lower()
    mode     = {"gltf": "glb", "pkl": "pickle"}.get(mode, mode)

    reader   = getattr(cls, f"load_{mode}", None)
    if reader is None:
        raise ValueError(f"{cls.__name__} has no reader for {mode} files")
    return reader(filename)


# -------------------------------- NAMESPACES -------------------------------- #


def _strip_namespace(name: Optional[str], namespace: Optional[str] = None) -> Optional[str]:
    """
    ``name`` without its namespaces, or without the one namespace path given.

    With no path ``A:B:node`` becomes ``node``. Without ``"A:B"`` it becomes
    ``A:node``, without ``"A"`` it becomes ``B:node``: the namespace's content
    moves up one level, as Maya merges a deleted namespace into its parent. A
    name outside the namespace comes back unchanged, and so does ``None``. A
    leading ``:``, Maya's absolute form, is accepted on either. Each part of a
    ``|`` path is stripped on its own: ``|A:grp|A:node`` becomes ``|grp|node``.

    Raises
    ------
    ValueError
        An empty path (``""`` or ``":"``).
    """
    target = None
    if namespace is not None:
        target = [x for x in namespace.split(":") if x]
        if not target:
            raise ValueError("strip_namespace needs a namespace path, not an empty one")

    if name is None:
        return None

    def strip(part):
        head, _, short = part.rpartition(":")
        if target is None:
            return short

        spaces = [x for x in head.split(":") if x]
        depth  = len(target)
        if spaces[:depth] != target:
            return part
        return ":".join(target[:-1] + spaces[depth:] + [short])

    return "|".join(strip(part) if part else part for part in name.split("|"))


def _namespace_of(name: Optional[str]) -> str:
    """the namespace path of ``name``'s last ``|`` part: ``"A:B"`` for ``A:B:node``"""
    if name is None:
        return ""
    return name.rpartition("|")[2].rpartition(":")[0].lstrip(":")


def _strip_namespaces(items: list, namespace: Optional[str] = None) -> None:
    """
    ``strip_namespace()`` on every item, all or nothing.

    Each item's ``NAMESPACED_FIELDS`` are stripped (``name``, and lists of
    node names such as a skin's influences), and so are those of the parts it
    owns (``_namespace_parts``: an Object's mesh and skin). Every new value is
    worked out first. Nothing is renamed when a new name would be shared with
    another item, among the items and their ``_namespace_peers`` (a node's
    hierarchy), or when a list would hold one name twice. Names that were
    already shared are left alone.

    Raises
    ------
    AttributeError
        An item has none of its ``NAMESPACED_FIELDS``: it has no name.
    ValueError
        A clash, or an empty namespace path.
    """
    _strip_namespace(None, namespace)  # an empty path raises, whatever the items

    changes = []
    planned = set()

    def plan(obj, owned):
        if id(obj) in planned:
            return
        planned.add(id(obj))

        present = [f for f in type(obj).NAMESPACED_FIELDS if hasattr(obj, f)]
        if not present and not owned:
            raise AttributeError(
                f"{type(obj).__name__} has no name to strip a namespace from"
            )

        for field in present:
            old = getattr(obj, field)
            if old is None:
                continue

            if isinstance(old, str):
                new = _strip_namespace(old, namespace)
                if new != old:
                    changes.append((obj, field, new))
                continue

            new = [_strip_namespace(x, namespace) for x in old]
            if len(set(new)) < len(set(old)):
                twice = sorted({x for x in new if new.count(x) > 1}, key=str)
                raise ValueError(
                    f"strip_namespace would give one name to two of "
                    f"{type(obj).__name__}({getattr(obj, 'name', None)}).{field}, "
                    f"nothing renamed: {twice[:10]}"
                )
            if new != list(old):
                new = np.asarray(new) if isinstance(old, np.ndarray) else type(old)(new)
                changes.append((obj, field, new))

        for part in obj._namespace_parts():
            if part is not None:
                plan(part, True)

    for item in items:
        plan(item, False)

    # the names a new name must not take: the items and their peers, each
    # container of peers (a hierarchy) walked once
    pool       = {id(item): item for item in items}
    containers = set()
    for item in items:
        peers = item._namespace_peers()
        if id(peers) in containers:
            continue
        containers.add(id(peers))
        for peer in peers:
            pool.setdefault(id(peer), peer)

    renamed = {
        id(obj): new for obj, field, new in changes if field == "name" and id(obj) in pool
    }
    owners = {}
    for key, obj in pool.items():
        name = renamed.get(key, getattr(obj, "name", None))
        if name is not None:
            owners.setdefault(name, []).append(obj)

    clashes = [
        f"{name} <- {', '.join(sorted(o.name for o in group))}"
        for name, group in sorted(owners.items())
        if len(group) > 1 and any(id(o) in renamed for o in group)
    ]
    if clashes:
        more = " ..." if len(clashes) > 10 else ""
        raise ValueError(
            f"strip_namespace would give {len(clashes)} name(s) to more than "
            f"one item, nothing renamed: {'; '.join(clashes[:10])}{more}"
        )

    for obj, field, new in changes:
        setattr(obj, field, new)


# ------------------------------ BASE DATACLASS ------------------------------ #
def flatten_nested_lists(xs):
    if not _is_sequence(xs):
        yield xs
    else:
        for x in xs:
            if _is_sequence(x):
                yield from flatten_nested_lists(x)
            else:
                yield x


def get_annotations(cls):
    """returns a dict of all annotations including inherited ones

    Annotations are resolved to runtime types. Modules using
    `from __future__ import annotations` store them as strings, which callers
    inspecting them (eg: to detect np.ndarray fields) cannot introspect.
    """
    try:
        return get_type_hints(cls)
    except (NameError, TypeError):
        # a hint could not be resolved, fall back to the raw (string) annotations
        annotations = {}
        for base in inspect.getmro(cls):
            annotations.update(getattr(base, "__annotations__", {}))
        return annotations


def _is_ndarray_annotation_str(dtype: str) -> bool:
    """resolves an unresolved (string) annotation to an array/optional-array"""
    text = dtype.replace(" ", "")

    for prefix in ("Optional[", "typing.Optional["):
        if text.startswith(prefix) and text.endswith("]"):
            text = text[len(prefix) : -1]
            break

    text = text.replace("|None", "").replace("None|", "")

    # a container of arrays (eg: `list[np.ndarray]`) is not an array
    return text in ("ndarray", "np.ndarray", "numpy.ndarray") or text.startswith("Array(")


def is_ndarray_annotation(dtype) -> bool:
    """returns True if an annotation is np.ndarray, or a union holding it
    (eg: `Optional[np.ndarray]`, `np.ndarray | None`)"""
    if dtype is np.ndarray or _array_spec(dtype) is not None:
        return True

    # modules using `from __future__ import annotations` leave hints as strings
    if isinstance(dtype, str):
        return _is_ndarray_annotation_str(dtype)

    # read `__args__`. Containers expose it too (`List[np.ndarray]` gives
    # `(np.ndarray,)`), so require a `None` member to keep this to the
    # optional-array unions -- an array inside a container is not an array.
    args = getattr(dtype, "__args__", ())
    return np.ndarray in args and type(None) in args


@dataclass(repr=False, eq=False)
class Data:
    """Base class with managed serialization."""

    EQUALITY_TEST_IGNORE = []

    # fields holding node names, a str or a list of str: strip_namespace()
    # rewrites them
    NAMESPACED_FIELDS = ("name",)

    # fields never saved: caches and links rebuilt after a load
    TRANSIENT_FIELDS = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls)

    def __setattr__(self, name: str, value) -> None:
        """Strict attribute setter -- rejects writes to undeclared
        public attributes so typos raise instead of silently creating
        new instance state.

        Catches mistakes like ``obj.ambiant = 0.5`` (when the real
        attribute is ``obj.ambient``) by raising :class:`AttributeError`
        instead of letting Python create a stray instance attribute.

        Allowed targets:

          - Names starting with ``_`` (private / cache attrs).
          - Names declared as dataclass fields anywhere in the MRO --
            including required fields without defaults (those don't
            appear in ``vars(cls)`` so a plain ``hasattr`` check would
            miss them).
          - Names already discoverable on the class via the MRO -- this
            covers ``@property`` descriptors, methods, and class-level
            constants like :attr:`EQUALITY_TEST_IGNORE`.

        Anything else raises :class:`AttributeError`.

        Note:
            Dataclass fields with defaults appear as class attributes
            (default value lives on the class), so ``hasattr`` finds
            them.  Required fields (no default) do NOT appear in
            ``vars(cls)`` and require the explicit ``fields()`` check.
            See ``test_base_setattr.test_required_field_assignment``.
        """
        cls   = type(self)
        specs = cls.__dict__.get("_ARRAY_SPECS_CACHE")
        if specs is None:
            specs = cls._array_specs()

        # a declared array field takes its dtype and shape on the way in
        spec = specs.get(name)
        if spec is not None:
            if value is not None:
                value = spec.coerce(value, f"{cls.__name__}.{name}")
            super().__setattr__(name, value)
            return

        if name.startswith("_") or hasattr(cls, name) or name in cls._dataclass_field_names():
            super().__setattr__(name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}; "
            f"public attributes must be declared on the class "
            f"(typo? expected one of: {self._allowed_public_attrs()!r})"
        )

    @classmethod
    def _dataclass_field_names(cls) -> frozenset:
        """All declared dataclass field names on this class (including
        required fields without defaults, which don't appear in
        ``vars(cls)``).  Cached per-class on first access.

        Required for the ``__setattr__`` guard to accept assignments to
        fields like ``MeshData.indices`` / ``points`` / ``counts``,
        which are declared without defaults.
        """
        cached = cls.__dict__.get("_FIELD_NAMES_CACHE")
        if cached is not None:
            return cached
        names = (
            frozenset(f.name for f in fields(cls)) if is_dataclass(cls) else frozenset()
        )
        # Direct class-attribute write -- bypasses our __setattr__
        # because we're operating on the class object (instance of
        # ``type``), not on a Data instance.
        cls._FIELD_NAMES_CACHE = names
        return names

    @classmethod
    def _field_hints(cls) -> dict:
        """every field's annotation, ``Array(...)`` metadata kept; cached per class"""
        cached = cls.__dict__.get("_FIELD_HINTS_CACHE")
        if cached is not None:
            return cached

        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError):
            # one hint will not resolve: resolve the others one by one
            hints = {}
            for base in reversed(inspect.getmro(cls)):
                scope = vars(sys.modules.get(base.__module__, object))
                for name, hint in getattr(base, "__annotations__", {}).items():
                    if isinstance(hint, str):
                        try:
                            hint = eval(hint, dict(scope))
                        except Exception:
                            pass
                    hints[name] = hint

        cls._FIELD_HINTS_CACHE = hints
        return hints

    @classmethod
    def _array_specs(cls) -> dict:
        """the ArraySpec of every declared array field; cached per class"""
        cached = cls.__dict__.get("_ARRAY_SPECS_CACHE")
        if cached is not None:
            return cached

        specs = {}
        if is_dataclass(cls):
            hints = cls._field_hints()
            for field in fields(cls):
                spec = _array_spec(hints.get(field.name))
                if spec is not None:
                    specs[field.name] = spec

        cls._ARRAY_SPECS_CACHE = specs
        return specs

    @classmethod
    def _allowed_public_attrs(cls) -> list:
        """Public attribute names declared anywhere on this class's
        MRO.  Used for friendlier error messages from
        :meth:`__setattr__`.  Skips dunder, private, function, and
        ``classmethod`` / ``staticmethod`` names; properties, plain
        class-level data attributes, and dataclass field names (even
        those without defaults) are kept.
        """
        seen = set()
        out  = []
        for base in cls.__mro__:
            for attr_name, raw_value in vars(base).items():
                if attr_name.startswith("_") or attr_name in seen:
                    continue
                # ``inspect.isfunction`` catches plain methods; the
                # isinstance check catches ``classmethod`` /
                # ``staticmethod`` descriptors (which ``callable()``
                # would NOT flag when read from ``vars()``).
                if inspect.isfunction(raw_value) or isinstance(
                    raw_value, (classmethod, staticmethod)
                ):
                    continue
                seen.add(attr_name)
                out.append(attr_name)
        # Add dataclass field names that have no class-level default
        # (those are absent from ``vars(cls)`` entirely).
        for field_name in cls._dataclass_field_names():
            if field_name.startswith("_") or field_name in seen:
                continue
            seen.add(field_name)
            out.append(field_name)
        return sorted(out)

    def __post_init__(self):
        """hack to fix limitations in the dataclass type system: an array
        default is shared by the class, so each object gets its own copy"""
        for field in fields(self):
            value = getattr(self, field.name)

            if isinstance(value, ImmutableArray):
                setattr(self, field.name, np.array(value))

    def __str__(self) -> str:
        if hasattr(self, "name") and self.name is not None:
            return str(self.name)
        return self.__repr__()

    def __repr__(self) -> str:
        if hasattr(self, "name") and self.name is not None:
            return type(self).__name__ + f"({str(self.name)})"

        class_name = type(self).__name__
        fields     = [f"{field}={getattr(self, field)!r}" for field in self.__dict__]
        fields_str = ", ".join(fields)
        return f"{class_name}({fields_str})"

    def __hash__(self):
        # the tree as a pickle holds it, numpy scalars as python ones and
        # arrays as their bytes: a loaded copy hashes the same. With no memo,
        # a value held twice is written twice, shared or not
        buffer       = io.BytesIO()
        pickler      = pickle.Pickler(buffer, protocol=4)
        pickler.fast = True
        pickler.dump(_pickled(self.to_dict()))
        return hash(buffer.getvalue())

    def __eq__(self, other) -> bool:
        """every field equal, floats to within np.allclose, but EQUALITY_TEST_IGNORE"""
        if not isinstance(other, self.__class__):
            return False

        skip = set(self.EQUALITY_TEST_IGNORE) | set(self.TRANSIENT_FIELDS)
        for field in fields(self):
            if field.name in skip:
                continue
            if not _close(getattr(self, field.name, None), getattr(other, field.name, None)):
                return False
        return True

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:
        if not hasattr(self, "name") or self.name is None:
            return False
        return self.name.__lt__(other.name)

    def __le__(self, other) -> bool:
        if not hasattr(self, "name") or self.name is None:
            return False
        return self.name.__le__(other.name)

    def __gt__(self, other) -> bool:
        if not hasattr(self, "name") or self.name is None:
            return False
        return self.name.__gt__(other.name)

    def __ge__(self, other) -> bool:
        if not hasattr(self, "name") or self.name is None:
            return False
        return self.name.__ge__(other.name)

    def copy(self):
        """returns a deep copy of self: the object a save and load gives back"""
        return type(self).from_dict(_tree_copy(self.to_dict()))

    def __copy__(self):
        """
        A shallow copy: a new object over the same field values. Caches and
        links are not carried over; they are rebuilt, as after a load.
        """
        return self._rebuilt(lambda value: value)

    def __deepcopy__(self, memo):
        """
        A deep copy, as copy.deepcopy makes one: every field copied, and what
        the copied structure reaches twice copied once. Caches and links are
        rebuilt, as after a load.
        """
        return self._rebuilt(lambda value: copy.deepcopy(value, memo), memo)

    def _rebuilt(self, copy_value, memo: Optional[dict] = None):
        """a new object of this class, its fields copy_value(this one's)"""
        cls = type(self)
        new = cls.__new__(cls)
        if memo is not None:
            memo[id(self)] = new

        for field in fields(cls):
            value = _NO_DEFAULT
            if field.name not in cls.TRANSIENT_FIELDS:
                value = getattr(self, field.name, _NO_DEFAULT)
            if isinstance(value, ImmutableArray):
                value = np.array(value)  # a class default read through the class
            elif value is not _NO_DEFAULT:
                value = copy_value(value)
            else:
                value = _field_default(field)
                if value is _NO_DEFAULT:
                    continue
                if isinstance(value, ImmutableArray):
                    value = np.array(value)
            # as set already: no dtype or shape check to repeat (a slots class too)
            object.__setattr__(new, field.name, value)

        new._post_load()
        return new

    @classmethod
    def info(cls) -> str:
        """Returns this class's docstring as a string.

        Works on both class and instance:

            >>> MeshData.info()
            >>> mesh = MeshData(...)
            >>> mesh.info()
        """
        doc = inspect.getdoc(cls)
        return doc if doc is not None else ""

    def reset_cached_data(self):
        """resets cached data: every attribute that is not an annotated field"""
        annotated = type(self)._field_hints()
        for key in list(self.__dict__):
            if key not in annotated:
                setattr(self, key, None)

    @property
    def namespace(self) -> str:
        """the namespace path of the name: ``"A:B"`` for ``A:B:node``, ``""`` for none"""
        return _namespace_of(self.name)

    def strip_namespace(self, namespace: Optional[str] = None) -> None:
        """
        Removes every namespace from the name, or the one path given.

        ``strip_namespace()`` turns ``A:B:node`` into ``node``. Given a path,
        only that namespace goes and what was inside it moves up one level:
        ``"A:B"`` gives ``A:node``, ``"A"`` gives ``B:node``. A name outside
        the namespace is left as it is. Every field in ``NAMESPACED_FIELDS``
        is stripped (a skin's influences with its name), and so are the parts
        the item owns (an Object's mesh and skin).

        Raises
        ------
        AttributeError
            This type has no name.
        ValueError
            Renaming nothing: another item would share the new name (a node
            in the same hierarchy), or two influences would.
        """
        _strip_namespaces([self], namespace)

    def _namespace_peers(self):
        """
        The container of the items whose names a new name must not take, a
        node's hierarchy; one shared container is walked once.
        """
        return ()

    def _namespace_parts(self) -> list:
        """data the item owns whose names are stripped with its own"""
        return []

    def match(self, *args, exact: bool = True) -> bool:
        """returns True if name matches any arguments"""
        if not hasattr(self, "name"):
            return False

        matched = []
        for x in flatten_nested_lists(args):
            if exact:
                matched.append(fnmatch.fnmatchcase(self.name, str(x)))
            else:
                matched.append(fnmatch.fnmatch(self.name.lower(), str(x).lower()))

        return any(matched)

    @staticmethod
    def _reconstruct(cls, state: dict):
        """how an older cgmath pickled data, refused now"""
        raise LegacyFileError(_OLDER_FILE.format("the file"))

    def __reduce__(self):
        """pickles the tree to_dict() returns, as npz and json save it"""
        if is_dataclass(self):
            return (_unpickle, (type(self), _pickled(self.to_dict())))
        return super().__reduce__()

    @classmethod
    def _pickle_tree(cls, tree: dict) -> dict:
        """this class's tree as a pickle holds it; _from_state reads it back"""
        return tree

    def save_json(self, filename: str) -> str:
        return _save_file(self, filename, "json")

    def save_pickle(self, filename: str) -> str:
        return _save_file(self, filename, "pkl")

    def save_npz(self, filename: str, compression=zipfile.ZIP_DEFLATED) -> str:
        return _save_file(self, filename, "npz", compression)

    def save(self, filename: str, mode: Optional[str] = None) -> str:
        """saves the data as ``mode`` (pkl, npz or json), or as the extension says"""
        return _save_file(self, filename, mode)

    @classmethod
    def from_dict(cls, data: dict) -> Any:
        """
        The object a tree describes: the class it names, which must be this
        one or derive from it, or this class when it names none. Fields the
        tree leaves out take fresh copies of their defaults.
        """
        data = dict(data)
        return _resolve_class(data.pop(CLASS_KEY, None), cls)._from_state(data)

    @classmethod
    def _from_state(cls, data: dict) -> Any:
        obj    = cls.__new__(cls)
        dtypes = data.pop(DTYPES_KEY, None) or {}
        shapes = data.pop(SHAPES_KEY, None) or {}
        hints  = cls._field_hints()

        specs  = cls._array_specs()

        for field in fields(cls):
            if field.name in data:
                hint  = hints.get(field.name)
                value = data.pop(field.name)
                if field.name not in specs:  # an array's lists hold no object
                    value = _decode(value)
                if isinstance(value, (dict, list)) and _holds_data(hint):
                    # an older cgmath saved a nested object as its bare fields
                    raise LegacyFileError(_OLDER_FILE.format("the data"))

                if field.name in dtypes and value is not None:
                    value = np.asarray(value, dtype=dtypes[field.name])
                elif (
                    isinstance(value, (list, tuple, bool, int, float, complex))
                    and is_ndarray_annotation(hint)
                ):
                    value = np.asarray(value)  # json keeps an array as lists, a 0-d one as a number
                elif field.name not in specs:
                    value = _as_declared(value, hint)

                if field.name in shapes and value is not None:
                    value = np.asarray(value).reshape(shapes[field.name])
            else:
                value = _field_default(field)
                if value is _NO_DEFAULT:
                    continue
                if isinstance(value, ImmutableArray):
                    value = np.array(value)  # the class default: this object's own copy
            setattr(obj, field.name, value)

        # what a class saves beside its fields
        for key, value in data.items():
            setattr(obj, key, _decode(value))

        obj._post_load()
        return obj

    def _post_load(self) -> None:
        """rebuilds what is not saved, once a load has set every field"""

    @classmethod
    def from_bytes(cls, data: bytes) -> Any:
        return cls.from_dict(bytes_to_dict(data))

    @classmethod
    def load_pickle(cls, filename: str) -> Any:
        return _read_pickle(cls, filename)

    @classmethod
    def load_json(cls, filename: str) -> Any:
        return cls.from_dict(_read_json(filename))

    @classmethod
    def load_npz(cls, filename: str) -> Any:
        return cls.from_dict(_read_npz(filename))

    @classmethod
    def load(cls, filename: str, mode: Optional[str] = None) -> Any:
        """
        Loads the data from a file: ``mode``, or the type the file reads as
        (:func:`get_file_type`). ``"pkl"`` / ``"json"`` / ``"npz"`` are
        cgmath's own; ``"glb"`` / ``"fbx"`` / ``"obj"`` / ``"usd"`` go to the
        class's ``load_<mode>``.
        """
        return _load_as(cls, filename, mode)

    def to_dict(self) -> dict:
        """
        The data as a tree: its class, then every field that differs from its
        default (one exactly equal to it is left out, and a load gives it a
        fresh copy back). Pickle, npz and json all save this.
        """
        cls    = type(self)
        data   = {CLASS_KEY: _class_tag(cls)}
        specs  = cls._array_specs()
        dtypes = {}
        shapes = {}

        for field in fields(self):
            if field.name in cls.TRANSIENT_FIELDS:
                continue

            value = getattr(self, field.name, _NO_DEFAULT)
            if value is _NO_DEFAULT:
                continue  # never set: nothing to save

            default = _field_default(field)
            if default is not _NO_DEFAULT:
                if isinstance(default, ImmutableArray):
                    default = np.asarray(default)
                if _same_value(value, default):
                    continue

            data[field.name] = _encode(value)

            # json keeps no dtype: a field that may hold more than one notes it
            spec = specs.get(field.name)
            if spec is not None and len(spec.dtypes) > 1 and isinstance(value, np.ndarray):
                dtypes[field.name] = value.dtype.name

            # nor the shape of an empty array: [] could be (0,), (0, 3) or (0, 0)
            if isinstance(value, np.ndarray) and value.size == 0 and value.ndim > 1:
                shapes[field.name] = list(value.shape)

        if dtypes:
            data[DTYPES_KEY] = dtypes
        if shapes:
            data[SHAPES_KEY] = shapes
        return data

    def to_json(self) -> str:
        """formats the data as human readable json"""
        return pretty_json(_json_tree(self.to_dict()))

    def to_bytes(self) -> bytes:
        """returns a bytes object from the zipped data"""
        return dict_to_bytes(self.to_dict())


class DataList(MutableSequence):
    DATA_LIST_CLASS = Data
    UNIQUE_LIST     = False

    # attributes saved with the list, beside its items
    LIST_FIELDS = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls)

    def __init__(self, iterable=None):
        self.list = []
        if iterable is not None:
            self.extend(iterable)

    def __setitem__(self, index, value):
        return self.list.__setitem__(index, value)

    def __delitem__(self, index):
        return self.list.__delitem__(index)

    def __len__(self):
        return self.list.__len__()

    def insert(self, index, value):
        return self.list.insert(index, value)

    def pop(self, index=-1):
        if index < 0:
            index += len(self.list)
        if index < 0 or index >= len(self.list):
            raise IndexError("pop index out of range")

        return self.list.pop(index)

    def __str__(self):
        return str([str(x) for x in self.list])

    def __repr__(self):
        return type(self).__name__ + f"({self.__str__()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__) or len(self) != len(other):
            return False
        if any(a != b for a, b in zip(self.list, other.list)):
            return False
        return all(_close(getattr(self, x), getattr(other, x)) for x in self.LIST_FIELDS)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.list.__lt__(other)

    def __le__(self, other):
        return self.list.__le__(other)

    def __gt__(self, other):
        return self.list.__gt__(other)

    def __ge__(self, other):
        return self.list.__ge__(other)

    def __getitem__(self, query: Union[str, int, slice, List[int], np.ndarray]):
        # if query is a string, return the morph target with that name
        if isinstance(query, str):
            return self.list[self.index(query)]

        # if query is a sequence of indices or a boolean mask, return a subset
        elif _is_sequence(query):
            query = _mask_to_indices(query, len(self.list))
            return self.__class__([self.list[i] for i in query])

        # if query is a slice, return a new DataList class
        else:
            result = self.list.__getitem__(query)
            if isinstance(query, slice):
                return self.__class__(result)
            else:
                return result

    def __contains__(self, item):
        if isinstance(item, str):
            for shape in self.list:
                if hasattr(shape, "name") and shape.name == item:
                    return True
        else:
            for shape in self:
                if shape == item:
                    return True

        return False

    def index(self, value, start=0, stop=None):
        if stop is None:
            stop = len(self)
            for i, shape in enumerate(self.list[start:stop]):
                if shape.name == value:
                    return i + start
        raise ValueError(f"{value} is not in list")

    def get(self, name: str, default: Any = None) -> Any:
        """the first item named ``name``, or ``default``"""
        for item in self.list:
            if getattr(item, "name", None) == name:
                return item
        return default

    @property
    def name(self) -> List[Optional[str]]:
        """every item's name, in order"""
        return [item.name for item in self.list]

    @property
    def namespace(self) -> List[str]:
        """every item's namespace path, ``""`` where it has none"""
        return [item.namespace for item in self.list]

    def strip_namespace(self, namespace: Optional[str] = None) -> None:
        """
        ``strip_namespace()`` on every item, all or nothing.

        Every new name is worked out first. If one would be shared with
        another item -- in the list, or among the items' peers (the rest of
        the hierarchies a node view belongs to) -- ValueError lists them and
        nothing is renamed. Names that were already shared are left alone.
        """
        _strip_namespaces(list(self.list), namespace)

    def sort(self, key=None, reverse=False):
        if key is None:
            self.list.sort(key=lambda shape: shape.name, reverse=reverse)
        else:
            self.list.sort(key=key, reverse=reverse)

    def __hash__(self):
        return hash(tuple(self.list))

    def copy(self):
        """deep copy"""
        new = self.__class__()
        for elem in self.list:
            new.append(elem.copy())
        for name in self.LIST_FIELDS:
            setattr(new, name, copy.deepcopy(getattr(self, name)))
        return new

    def __copy__(self):
        """a shallow copy: a new list over the same items"""
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.list = list(self.list)
        return new

    def __deepcopy__(self, memo):
        """
        A deep copy, as copy.deepcopy makes one: each item copied, and what
        the copied structure reaches twice (an item also held elsewhere)
        copied once.
        """
        new            = type(self)()
        memo[id(self)] = new
        for item in self.list:
            new.append(copy.deepcopy(item, memo))
        for name in self.LIST_FIELDS:
            setattr(new, name, copy.deepcopy(getattr(self, name), memo))
        return new

    def __setstate__(self, state):
        """how an older cgmath's lists unpickled, refused now"""
        raise LegacyFileError(_OLDER_FILE.format("the file"))

    def reset_cached_data(self):
        """resets cached data"""
        for elem in self.list:
            elem.reset_cached_data()

    def append(self, value):
        if self.DATA_LIST_CLASS is not None:
            if not isinstance(value, self.DATA_LIST_CLASS):
                raise TypeError(f"{value} is not a valid object.")

        if isinstance(value, str):
            for shape in self.list:
                if hasattr(shape, "name") and shape.name == value:
                    raise TypeError(f"{value} is not a valid object.")

        elif self.UNIQUE_LIST:
            for shape in self.list:
                if shape == value:
                    raise ValueError(f"{value} already exists.")

        self.list.append(value)

    def extend(self, iterable):
        # snapshot first: an append that takes ownership removes the value
        # from its previous owner, and iterating a list that shrinks under
        # you skips every other entry
        for value in list(iterable):
            self.append(value)

    def match(self, *args, exclude: bool = False, exact: bool = True) -> Any:
        """returns a DataList object with matching names"""
        new = self.__class__()

        for shape in self.list:
            if exclude:
                if not shape.match(*args, exact=exact):
                    new.append(shape)
            elif shape.match(*args, exact=exact):
                new.append(shape)

        return new

    # --------------------------------- FILE I/O --------------------------------- #

    def to_dict(self) -> dict:
        """the list as a tree: its class, its items' trees, and its LIST_FIELDS"""
        data = {
            CLASS_KEY: _class_tag(type(self)),
            ITEMS_KEY: [item.to_dict() for item in self.list],
        }
        for name in self.LIST_FIELDS:
            data[name] = _encode(getattr(self, name))
        return data

    def to_json(self) -> str:
        """formats the data as human readable json"""
        return pretty_json(_json_tree(self.to_dict()))

    def to_bytes(self) -> bytes:
        """returns a bytes object from the zipped data"""
        return dict_to_bytes(self.to_dict())

    def __reduce__(self):
        """pickles the tree to_dict() returns, as npz and json save it"""
        return (_unpickle, (type(self), _pickled(self.to_dict())))

    @classmethod
    def _pickle_tree(cls, tree: dict) -> dict:
        """this class's tree as a pickle holds it; _from_state reads it back"""
        return tree

    def save_json(self, filename: str) -> str:
        return _save_file(self, filename, "json")

    def save_npz(self, filename: str, compression=zipfile.ZIP_DEFLATED) -> str:
        return _save_file(self, filename, "npz", compression)

    def save_pickle(self, filename: str) -> str:
        return _save_file(self, filename, "pkl")

    def save(self, filename: str, mode: Optional[str] = None) -> str:
        """saves the list as ``mode`` (pkl, npz or json), or as the extension says"""
        return _save_file(self, filename, mode)

    @classmethod
    def from_dict(cls, data: dict) -> Any:
        """
        The list a tree describes: the class it names, which must be this one
        or derive from it, each item rebuilt as the class it names.
        """
        data = dict(data)
        return _resolve_class(data.pop(CLASS_KEY, None), cls)._from_state(data)

    @classmethod
    def _from_state(cls, data: dict) -> Any:
        obj = cls()
        if ITEMS_KEY in data:
            items = data.pop(ITEMS_KEY)
        else:
            # a dict of item dicts, as the asset readers build them
            items, data = list(data.values()), {}

        for item in items:
            obj.append(cls.DATA_LIST_CLASS.from_dict(item))
        for name in cls.LIST_FIELDS:
            if name in data:
                setattr(obj, name, _decode(data.pop(name)))

        obj._post_load()
        return obj

    def _post_load(self) -> None:
        """rebuilds what is not saved, once a load has set every item and field"""

    @classmethod
    def from_bytes(cls, data: bytes) -> Any:
        return cls.from_dict(bytes_to_dict(data))

    @classmethod
    def load_json(cls, filename: str) -> Any:
        return cls.from_dict(_read_json(filename))

    @classmethod
    def load_npz(cls, filename: str) -> Any:
        return cls.from_dict(_read_npz(filename))

    @classmethod
    def load_pickle(cls, filename: str) -> Any:
        return _read_pickle(cls, filename)

    @classmethod
    def load(cls, filename: str, mode: Optional[str] = None) -> Any:
        """
        Loads the data from a file: ``mode``, or the type the file reads as
        (:func:`get_file_type`). ``"pkl"`` / ``"json"`` / ``"npz"`` are
        cgmath's own; ``"glb"`` / ``"fbx"`` / ``"obj"`` / ``"usd"`` go to the
        class's ``load_<mode>``.
        """
        return _load_as(cls, filename, mode)


_register(Data)
_register(DataList)
