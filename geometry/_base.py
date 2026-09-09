import copy
import fnmatch
import inspect
import io
import numbers
import os
import pickle
import zipfile
from collections.abc import Iterable, MutableSequence
from dataclasses import dataclass, fields, is_dataclass, MISSING
from typing import _GenericAlias, Any, get_type_hints, List, Optional, Union

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


def none_to_nan(obj):
    """convert None object to np.nan (not a number)"""
    if isinstance(obj, dict):
        return {k: none_to_nan(v) for k, v in obj.items()}

    elif obj is None:
        return np.nan

    else:
        return obj


def nan_to_none(obj):
    """convert np.nan (not a number) to None object"""

    if isinstance(obj, numbers.Real):
        if np.isnan(obj):
            return None
        return obj

    elif isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}

    elif not _is_sequence(obj) and obj.dtype.kind not in "SU" and np.isnan(obj):
        return None

    elif not (_is_numeric_sequence(obj) and _is_numeric_or_boolean(obj)):
        return obj.tolist()

    else:
        return obj


def dict_to_zip(data, filename, compression=zipfile.ZIP_DEFLATED):
    """takes a deep dict and saves a zip file with .npy data"""

    def add_to_zip(data, zipf, prefix):
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                add_to_zip(value, zipf, new_prefix)
        else:
            with zipf.open(prefix + ".npy", "w") as f:
                try:
                    arr = np.asanyarray(data)
                except ValueError:
                    arr = np.asanyarray(data, dtype=object)

                np.save(f, arr)

    with zipfile.ZipFile(filename, "w", compression=compression) as zipf:
        data = none_to_nan(data)
        add_to_zip(data, zipf, "")


def dict_to_bytes(data):
    """takes a deep dict and returns a zip file as bytes"""
    buffer = io.BytesIO()
    dict_to_zip(data, buffer)
    return buffer.getvalue()


def zip_to_dict(filename):
    """takes a zip file and returns a deep dict of numpy arrays"""

    def add_to_dict(path, value, dictionary):
        keys = path.split("/")
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value

    data_dict = {}
    with zipfile.ZipFile(filename, "r") as zipf:
        for file in zipf.namelist():
            with zipf.open(file) as f:
                path  = file[:-4]  # get rid of '.npy'
                array = np.load(f, allow_pickle=True)
                add_to_dict(path, array, data_dict)

    return nan_to_none(data_dict)


def bytes_to_dict(data):
    """takes a bytes object and returns a deep dict of numpy arrays"""
    buffer = io.BytesIO(data)
    return zip_to_dict(buffer)


def get_file_type(filename: str) -> str:
    """identifies the supported file type by reading the header"""

    with open(filename, "rb") as file:
        header = file.read(4)

        # assume its a numpy zip file
        if header == b"PK\x03\x04":
            return "npz"

        # assume its a json file
        elif header.strip() in [b"{", b"["]:
            return "json"

        # assume its a pickle file
        return "pkl"


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
    return text in ("ndarray", "np.ndarray", "numpy.ndarray")


def is_ndarray_annotation(dtype) -> bool:
    """returns True if an annotation is np.ndarray, or a union holding it
    (eg: `Optional[np.ndarray]`, `np.ndarray | None`)"""
    if dtype is np.ndarray:
        return True

    # modules using `from __future__ import annotations` leave hints as strings
    if isinstance(dtype, str):
        return _is_ndarray_annotation_str(dtype)

    # `typing.get_args()` is python 3.8+, and mayapy 2022 ships python 3.7, so
    # read `__args__` directly. Containers expose it too (`List[np.ndarray]`
    # gives `(np.ndarray,)`), so require a `None` member to keep this to the
    # optional-array unions -- an array inside a container is not an array.
    args = getattr(dtype, "__args__", ())
    return np.ndarray in args and type(None) in args


@dataclass(repr=False, eq=False)
class Data:
    """Base class with managed serialization."""

    EQUALITY_TEST_IGNORE = []

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
        if (
            name.startswith("_")
            or hasattr(type(self), name)
            or name in self._dataclass_field_names()
        ):
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
        """hack to fix limitations in the dataclass type system"""
        for field in fields(self):
            value = getattr(self, field.name)

            if isinstance(value, ImmutableArray):
                setattr(self, field.name, np.asarray(value))

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
        return hash(pickle.dumps(self.to_dict))

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False

        dict0 = self.to_dict()
        dict1 = other.to_dict()

        for key in dict0:
            if key not in self.EQUALITY_TEST_IGNORE:
                if key not in dict1:
                    return False

                if isinstance(dict0[key], np.ndarray) or isinstance(
                    dict1[key], np.ndarray
                ):
                    arr0 = np.asarray(dict0[key])
                    arr1 = np.asarray(dict1[key])
                    if arr0.shape != arr1.shape or not np.allclose(arr0, arr1):
                        return False

                else:
                    if dict0[key] != dict1[key]:
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
        """returns a deep copy of self"""
        data = copy.deepcopy(self.to_dict())
        return self.__class__.from_dict(data)

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
        """resets cached data"""
        for key in self.__dict__.keys():
            if key not in get_type_hints(type(self)):
                setattr(self, key, None)

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
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def __reduce__(self):
        """only pickle out annotated data"""
        if is_dataclass(self):
            return (self._reconstruct, (self.__class__, self.to_dict()))
        else:
            return super().__reduce__()

    def save_json(self, filename: str) -> str:
        filename = os.path.expanduser(filename)
        with open(filename, "w") as f:
            f.write(self.to_json())
        return filename

    def save_pickle(self, filename: str) -> str:
        filename = os.path.expanduser(filename)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    def save_npz(self, filename: str, compression=zipfile.ZIP_DEFLATED) -> str:
        filename = os.path.expanduser(filename)
        dict_to_zip(self.to_dict(), filename, compression=zipfile.ZIP_DEFLATED)
        return filename

    def save(self, filename: str, mode: Optional[str] = None) -> str:
        """saves the data to a file"""
        filename = os.path.expanduser(filename)

        if mode is None:
            # determine mode from file extension
            stem, ext = os.path.splitext(filename)
            mode = ext.lower().split(".")[-1]

        writer = {
            "pkl":  self.save_pickle,
            "json": self.save_json,
            "npz":  self.save_npz,
        }

        return writer[mode](filename)

    @classmethod
    def from_dict(cls, data: dict) -> Any:
        obj = cls.__new__(cls)

        # for each field
        field_data_types = get_annotations(cls)

        for field in data:
            value = data[field]

            # if this is a field annotated as an array, enforce np.ndarray.
            # optional fields (eg: `Optional[np.ndarray]`) are converted too,
            # unless they hold no data.
            if value is not None and is_ndarray_annotation(field_data_types.get(field)):
                value = np.asarray(value)

            setattr(obj, field, value)

        return obj

    @classmethod
    def from_bytes(cls, data: bytes) -> Any:
        return cls.from_dict(bytes_to_dict(data))

    @classmethod
    def load_pickle(cls, filename: str) -> Any:
        filename = os.path.expanduser(filename)
        with open(filename, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_json(cls, filename: str) -> Any:
        filename = os.path.expanduser(filename)
        with open(filename, "r") as f:
            data = json.load(f)
            return cls.from_dict(data)

    @classmethod
    def load_npz(cls, filename: str) -> Any:
        filename = os.path.expanduser(filename)
        return cls.from_dict(zip_to_dict(filename))

    @classmethod
    def load(cls, filename: str, mode: Optional[str] = None) -> Any:
        """loads the data from a file"""

        filename = os.path.expanduser(filename)

        # identify the filetype of None given
        if mode is None:
            mode = get_file_type(filename)

        reader = {
            "pkl":  cls.load_pickle,
            "json": cls.load_json,
            "npz":  cls.load_npz,
        }

        return reader[mode](filename)

    def to_dict(self) -> dict:
        """returns the annotated data as a dict"""

        # get type hints for all attributes of the class
        type_hints = get_type_hints(type(self))
        data       = {}

        # iterate over dataclass fields to access default values
        for field in fields(self):
            attr_name  = field.name
            attr_type  = type_hints[attr_name]
            attr_value = getattr(self, attr_name)

            default_value = (
                field.default
                if field.default is not MISSING
                else (
                    field.default_factory()
                    if field.default_factory is not MISSING
                    else None
                )
            )

            if not (
                isinstance(attr_type, _GenericAlias)
                and attr_type.__origin__ is Union
                and type(None) in attr_type.__args__
            ):
                data[attr_name] = getattr(self, attr_name, None)

            else:
                if not isinstance(attr_value, np.ndarray) and not isinstance(
                    default_value, np.ndarray
                ):
                    if attr_value != default_value:
                        data[attr_name] = getattr(self, attr_name, None)
                else:
                    attr_value    = np.asarray(attr_value)
                    default_value = np.asarray(default_value)
                    if attr_value.shape != default_value.shape or not np.allclose(
                        attr_value, default_value
                    ):
                        data[attr_name] = getattr(self, attr_name, None)

        return data

    def to_json(self) -> str:
        """formats the data as human readable json"""
        return pretty_json(self.to_dict())

    def to_bytes(self) -> bytes:
        """returns a bytes object from the zipped data"""
        return dict_to_bytes(self.to_dict())


class DataList(MutableSequence):
    DATA_LIST_CLASS = Data
    UNIQUE_LIST     = False

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
        if isinstance(other, self.__class__):
            if len(self) != len(other):
                return False

            for i in range(len(self.list)):
                if self.list[i] != other.list[i]:
                    return False

        return True

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
        return new

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
        data_dict = {}
        for i, item in enumerate(self.list):
            data_dict[f"index_{i}"] = item.to_dict()

        return data_dict

    def to_json(self) -> str:
        """formats the data as human readable json"""
        return pretty_json(self.to_dict())

    def to_bytes(self) -> bytes:
        """returns a bytes object from the zipped data"""
        return dict_to_bytes(self.to_dict())

    def save_json(self, filename: str) -> str:
        """saves the morph target list to a json file"""
        filename = os.path.expanduser(filename)
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)
            return filename

    def save_npz(self, filename: str) -> str:
        """saves the morph target list to a npz file"""
        filename = os.path.expanduser(filename)
        dict_to_zip(self.to_dict(), filename)
        return filename

    def save_pickle(self, filename: str) -> str:
        filename = os.path.expanduser(filename)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    def save(self, filename: str, mode: Optional[str] = None) -> str:
        """saves the data to a file"""
        filename = os.path.expanduser(filename)

        if mode is None:
            # determine mode from file extension
            stem, ext = os.path.splitext(filename)
            mode = ext.lower().split(".")[-1]

        writer = {
            "pkl":  self.save_pickle,
            "json": self.save_json,
            "npz":  self.save_npz,
        }

        return writer[mode](filename)

    @classmethod
    def from_dict(cls, data: dict) -> Any:
        obj = cls()
        for key in data:
            obj.append(cls.DATA_LIST_CLASS.from_dict(data[key]))

        return obj

    @classmethod
    def from_bytes(cls, data: bytes) -> Any:
        return cls.from_dict(bytes_to_dict(data))

    @classmethod
    def load_json(cls, filename: str) -> Any:
        # fields annotated as np.ndarray are converted by Data.from_dict()
        filename = os.path.expanduser(filename)
        with open(filename, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_npz(cls, filename: str) -> Any:
        filename = os.path.expanduser(filename)
        return cls.from_dict(zip_to_dict(filename))

    @classmethod
    def load_pickle(cls, filename: str) -> Any:
        filename = os.path.expanduser(filename)
        with open(filename, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load(cls, filename: str, mode: Optional[str] = None) -> Any:
        """loads the data from a file"""

        filename = os.path.expanduser(filename)

        reader = {
            "pkl":  cls.load_pickle,
            "json": cls.load_json,
            "npz":  cls.load_npz,
        }

        # identify the filetype of None given
        if mode is None:
            mode = get_file_type(filename)

        return reader[mode](filename)