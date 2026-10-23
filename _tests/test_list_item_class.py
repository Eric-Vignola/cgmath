"""A list class names the class it holds in brackets.

``class MeshList(DataList[MeshData])`` sets ``DATA_LIST_CLASS``, which append
and a load check each item against, and gives an editor the item class of
``meshes[0]`` and ``for mesh in meshes``. A class may still set
``DATA_LIST_CLASS`` itself, never to another class than its brackets name nor
beside brackets that leave the item's place open; a class naming neither holds
its base's. Nothing a file saves changes: a list class's tag is its module and
name, as before.
"""

import copy
import os
import pickle
import re
import sys
import tempfile
import types
import typing
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, TypeVar, Union

import numpy as np
from cgmath.data import _CLASSES, _class_tag, _Item, _List, Data, DataList
from cgmath.geometry import (
    MeshData,
    MeshList,
    MorphData,
    MorphList,
    SkinData,
    SkinList,
    UVData,
    UVList,
)
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList
from cgmath.render import Scene

try:
    import jedi
except ImportError:
    jedi = None


def _release(version: str) -> tuple:
    """``(major, minor)`` of a version string, ``(0, 0)`` where it starts with none"""
    found = re.match(r"(\d+)\.(\d+)", version)
    return (int(found[1]), int(found[2])) if found else (0, 0)


# jedi's inference below was measured on 0.20: an older jedi is not asked;
# and only when CGMATH_JEDI_TEST=1 is set, so a wheel user's suite never
# depends on what a third-party completer guesses
JEDI_RELEASE = _release(jedi.__version__) if jedi is not None else (0, 0)

# every list cgmath ships, and the class each holds
SHIPPED = {
    MeshList:      MeshData,
    UVList:        UVData,
    MorphList:     MorphData,
    SkinList:      SkinData,
    TransformList: TransformData,
    HierarchyData: TransformData,
    ClipData:      TransformData,
    Scene:         TransformData,
}


@dataclass(repr=False, eq=False)
class _Pin(Data):
    name:   Optional[str] = None
    weight: float = 1.0


@dataclass(repr=False, eq=False)
class _Bolt(Data):
    name: Optional[str] = None


class _Pins(DataList[_Pin]):
    """a list of pins, declared with brackets"""


def _declare(*bases, **body):
    """a list class made as a class statement makes it, with ``bases`` and ``body``"""
    body = {"__module__": __name__, **body}
    return types.new_class("_Probe", bases, {}, lambda namespace: namespace.update(body))


try:

    class _AliasAsClass(types.GenericAlias):
        """``list[_Pin]`` as python 3.9 and 3.10 see it: a class, to isinstance"""

        @property
        def __class__(self):
            return type

except TypeError:
    # python 3.9.0 and 3.9.1: GenericAlias takes no subclass, and list[int]
    # is a class to isinstance there itself
    _AliasAsClass = None


class TestTheBracketsNameTheItemClass(unittest.TestCase):
    def test_the_brackets_set_the_item_class(self):
        self.assertIs(_Pins.DATA_LIST_CLASS, _Pin)
        pins = _Pins([_Pin(name="a")])
        with self.assertRaises(TypeError):
            pins.append(_Bolt(name="b"))
        self.assertIs(_CLASSES[_class_tag(_Pins)], _Pins)

    def test_a_class_setting_DATA_LIST_CLASS_too_must_name_the_same_class(self):
        self.assertIs(_declare(DataList[_Pin], DATA_LIST_CLASS=_Pin).DATA_LIST_CLASS, _Pin)
        with self.assertRaisesRegex(TypeError, "holds _Pin, as its brackets say, and _Bolt"):
            _declare(DataList[_Pin], DATA_LIST_CLASS=_Bolt)
        with self.assertRaisesRegex(TypeError, "and None, as its DATA_LIST_CLASS"):
            _declare(DataList[_Pin], DATA_LIST_CLASS=None)

    def test_a_class_naming_neither_holds_its_bases(self):
        self.assertIs(_declare(_Pins).DATA_LIST_CLASS,    _Pin)
        self.assertIs(_declare(MeshList).DATA_LIST_CLASS, MeshData)
        self.assertIs(_declare(DataList).DATA_LIST_CLASS, Data)
        # the old spelling still works, under a base's brackets too: they are
        # the base's, not its subclass's
        self.assertIs(_declare(DataList, DATA_LIST_CLASS=_Bolt).DATA_LIST_CLASS, _Bolt)
        self.assertIs(_declare(MeshList, DATA_LIST_CLASS=UVData).DATA_LIST_CLASS, UVData)

    def test_the_brackets_hold_one_class(self):
        # Any is a class since python 3.11, list[int] one to isinstance
        # before it, None stands for NoneType: none names a class an item is.
        # The stand-in for 3.9 is list[_Pin], not list[int]: typing caches
        # DataList[...], and would hand back list[int]'s for an equal alias
        aliased = (list[int],) + ((_AliasAsClass(list, (_Pin,)),) if _AliasAsClass else ())
        for named in ("_Pin", Union[_Pin, _Bolt], Any, None, *aliased):
            with self.subTest(named=named), self.assertRaisesRegex(TypeError, "names no class"):
                _declare(DataList[named])
        with self.assertRaises(TypeError):
            DataList[_Pin, _Bolt]
        T       = TypeVar("T")
        K       = TypeVar("K")
        generic = _declare(DataList[T])
        with self.assertRaisesRegex(TypeError, "names two item classes, _Pin and _Bolt"):
            _declare(generic[_Pin], DataList[_Bolt])
        # one class named twice is one class, one place left open twice one
        # place; a class beside a place left open, or two places, are two
        other = _declare(DataList[K])
        self.assertIs(_declare(generic[_Pin], DataList[_Pin]).DATA_LIST_CLASS, _Pin)
        self.assertIs(_declare(_declare(generic[T], other[T])[_Pin]).DATA_LIST_CLASS, _Pin)
        with self.assertRaisesRegex(TypeError, "names its item class, _Bolt, and leaves it open, ~K"):
            _declare(generic[K], DataList[_Bolt])
        with self.assertRaisesRegex(TypeError, "names its item class, _Pin, and leaves it open, ~K"):
            _declare(other[_Pin], generic[K])
        with self.assertRaisesRegex(TypeError, "leaves two item parameters, ~T and ~K"):
            _declare(generic[T], other[K])
        # the old spelling names the class as the brackets do: beside a place
        # left open, it is refused too; under a list still generic, with no
        # brackets of its own, it is the class
        with self.assertRaisesRegex(TypeError, "names its item class, _Pin, and leaves it open, ~T"):
            _declare(DataList[T], DATA_LIST_CLASS=_Pin)
        with self.assertRaisesRegex(TypeError, "names its item class, _Pin, and leaves it open, ~K"):
            _declare(generic[K], DATA_LIST_CLASS=_Pin)
        self.assertIs(_declare(generic, DATA_LIST_CLASS=_Pin).DATA_LIST_CLASS, _Pin)

    def test_the_item_fills_its_own_parameter_among_a_lists_others(self):
        T = TypeVar("T")
        K = TypeVar("K")
        # the brackets of a list that takes a parameter of its own fill that
        # parameter, the item's place wherever it stands
        for params, args in (((T, K), (_Pin, int)), ((K, T), (int, _Pin))):
            pair = _declare(DataList[T], Generic[params])
            with self.subTest(params=params):
                self.assertIs(_declare(pair[args]).DATA_LIST_CLASS, _Pin)
        # a list that has named its class keeps it under brackets of its own
        tagged = _declare(DataList[_Pin], Generic[K])
        self.assertIs(_declare(tagged[str]).DATA_LIST_CLASS, _Pin)
        _declare(tagged[str])().append(_Pin(name="a"))
        tagged = _declare(_declare(DataList[T])[_Pin], Generic[K])
        self.assertIs(_declare(tagged[str]).DATA_LIST_CLASS, _Pin)

        # a generic base that is no list takes brackets of its own
        class Mixin(Generic[T]):
            pass

        self.assertIs(_declare(DataList[_Pin], Mixin[int]).DATA_LIST_CLASS, _Pin)
        self.assertIs(_declare(Mixin[int], DataList[_Pin]).DATA_LIST_CLASS, _Pin)

    def test_a_list_holding_its_class_takes_no_brackets(self):
        # a list class that names its item is no longer generic: a narrower
        # list names the narrower class beside it, as UVList does
        for cls in (_Pins, MeshList, HierarchyData):
            with self.subTest(cls=cls.__name__), self.assertRaisesRegex(TypeError, "not a generic class"):
                cls[_Pin]
        narrower = _declare(MeshList, DataList[UVData])
        self.assertIs(narrower.DATA_LIST_CLASS, UVData)
        self.assertTrue(issubclass(narrower, MeshList))

    def test_a_list_still_generic_leaves_the_class_to_its_subclasses(self):
        T       = TypeVar("T")
        U       = TypeVar("U")
        generic = _declare(DataList[T])
        self.assertIs(generic.DATA_LIST_CLASS, Data)
        self.assertIs(_declare(generic[_Pin]).DATA_LIST_CLASS, _Pin)
        self.assertIs(_declare(_declare(generic[U])[_Pin]).DATA_LIST_CLASS, _Pin)

    def test_a_subscript_alone_builds_a_plain_list(self):
        # as list[int]() is a plain list: the brackets name the class only
        # where a class is declared
        made = DataList[_Pin]()
        self.assertIs(type(made), DataList)
        self.assertIs(made.DATA_LIST_CLASS, Data)
        self.assertNotIn("__orig_class__", vars(made))
        with self.assertRaises(TypeError):
            isinstance(made, DataList[_Pin])

    def test_every_shipped_list_holds_its_class(self):
        for cls, item in SHIPPED.items():
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls.DATA_LIST_CLASS, item)
        for cls in (MeshList, MorphList, SkinList, TransformList):
            self.assertEqual(vars(cls)["__orig_bases__"], (DataList[SHIPPED[cls]],))
        self.assertEqual(vars(UVList)["__orig_bases__"], (MeshList, DataList[UVData]))
        for cls in (HierarchyData, ClipData, Scene):
            self.assertNotIn("__orig_bases__", vars(cls), "TransformList's brackets hold for it")


class TestAGenericListChangesNothingElse(unittest.TestCase):
    def test_an_instance_takes_the_same_writes(self):
        old = _declare(DataList, DATA_LIST_CLASS=_Pin, LIST_FIELDS=("label",))
        new = _declare(DataList[_Pin], LIST_FIELDS=("label",))
        self.assertEqual(new._writable(fresh=True), old._writable(fresh=True))
        pins = _Pins()
        for name in ("__orig_class__", "__parameters__", "__orig_bases__", "_is_protocol", "foo"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(pins, name, 1)

    def test_a_bracketed_list_saves_loads_pickles_and_copies(self):
        pins = _Pins([_Pin(name="a", weight=2.0), _Pin(name="b")])
        with tempfile.TemporaryDirectory() as tmp:
            for ext in ("pkl", "npz", "json"):
                path = pins.save(os.path.join(tmp, f"pins.{ext}"))
                for loader in (_Pins, Data, DataList):
                    with self.subTest(ext=ext, loader=loader.__name__):
                        back = loader.load(path)
                        self.assertIs(type(back), _Pins)
                        self.assertEqual(back, pins)
        self.assertEqual(pins.to_dict()["__class__"], f"{_Pins.__module__}._Pins")
        for label, made in (
            ("pickle", pickle.loads(pickle.dumps(pins))),
            ("deepcopy", copy.deepcopy(pins)),
            ("copy()", pins.copy()),
        ):
            with self.subTest(label):
                self.assertIs(type(made), _Pins)
                self.assertEqual(made, pins)
                self.assertIsNot(made[0], pins[0])
        shallow = copy.copy(pins)
        self.assertIs(type(shallow), _Pins)
        self.assertIs(shallow[0], pins[0])

    def test_iterating_gives_the_items_themselves(self):
        pins = _Pins([_Pin(name="a"), _Pin(name="b")])
        self.assertEqual([id(x) for x in pins], [id(x) for x in pins.list])
        rig = HierarchyData([TransformData(name="root"), TransformData(name="tip")])
        self.assertEqual([id(x) for x in rig], [id(x) for x in rig.list])
        self.assertEqual([x.name for x in reversed(rig)], ["tip", "root"])

    def test_a_loop_goes_on_over_the_list_it_started_with(self):
        # an edit in place it sees, as a loop over a list does; a new .list,
        # as MorphList.remove_unused sets, it does not
        pins = _Pins([_Pin(name="a"), _Pin(name="b")])
        seen = []
        for pin in pins:
            seen.append(pin.name)
            if pin.name == "a":
                pins.append(_Pin(name="c"))
        self.assertEqual(seen, ["a", "b", "c"])
        seen = []
        for pin in pins:
            seen.append(pin.name)
            pins.list = []
        self.assertEqual(seen, ["a", "b", "c"])


class TestAnEditorReadsTheItemClass(unittest.TestCase):
    def test_the_hooks_an_editor_reads_are_annotated(self):
        self.assertEqual(typing.get_type_hints(DataList.__iter__), {"return": Iterator[_Item]})
        self.assertIs(typing.get_type_hints(DataList.pop)["return"], _Item)
        # a shipped list's own pop, HierarchyData's, gives its class too
        for cls, item in SHIPPED.items():
            with self.subTest(cls=cls.__name__):
                hints = typing.get_type_hints(cls.pop, vars(sys.modules[cls.pop.__module__]))
                self.assertIn(hints.get("return"), (_Item, item))
        hints = typing.get_type_hints(HierarchyData.pop, vars(sys.modules[HierarchyData.__module__]))
        self.assertIs(hints["return"], TransformData)
        if sys.version_info < (3, 11):
            return  # typing.get_overloads came with python 3.11
        hints = [typing.get_type_hints(x) for x in typing.get_overloads(DataList.__getitem__)]
        self.assertEqual(
            hints,
            [
                {"query": Union[int, str], "return": _Item},
                {"self": _List, "query": Union[slice, List[int], np.ndarray], "return": _List},
            ],
        )
        hints = [
            typing.get_type_hints(x, vars(sys.modules[TransformList.__module__]))
            for x in typing.get_overloads(TransformList.__getitem__)
        ]
        self.assertEqual(
            hints,
            [
                {"query": Union[int, str], "return": TransformData},
                {"query": Union[slice, List[int], np.ndarray], "return": TransformList},
            ],
        )

    @unittest.skipIf(
        os.environ.get("CGMATH_JEDI_TEST") != "1" or JEDI_RELEASE < (0, 20),
        "runs only with CGMATH_JEDI_TEST=1 and jedi 0.20 or later installed",
    )
    def test_jedi_completes_an_item_with_its_class(self):
        # jedi's parse cache goes to a folder of the test's own, not the
        # user's profile
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.addCleanup(setattr, jedi.settings, "cache_directory", jedi.settings.cache_directory)
        jedi.settings.cache_directory = cache.name
        root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root    = os.path.dirname(root)  # the folder holding cgmath
        project = jedi.Project(path=root, added_sys_path=[root], smart_sys_path=False)
        modules = {
            MeshList:      "cgmath.geometry",
            UVList:        "cgmath.geometry",
            MorphList:     "cgmath.geometry",
            SkinList:      "cgmath.geometry",
            TransformList: "cgmath.hierarchy",
            HierarchyData: "cgmath.hierarchy",
            ClipData:      "cgmath.hierarchy",
            Scene:         "cgmath.render",
        }
        for cls, module in modules.items():
            name = cls.__name__
            for label, code in (
                ("x[0]", f"from {module} import {name}\nx = {name}()\nx[0]"),
                ("x['a']", f"from {module} import {name}\nx = {name}()\nx['a']"),
                ("x.pop()", f"from {module} import {name}\nx = {name}()\nx.pop()"),
                ("for m in x: m", f"from {module} import {name}\nx = {name}()\nfor m in x:\n    m"),
                ("x[1:]", f"from {module} import {name}\nx = {name}()\nx[1:]"),
            ):
                lines = code.split("\n")
                found = jedi.Script(code, project=project).infer(len(lines), len(lines[-1]))
                with self.subTest(cls=name, code=label):
                    got = sorted({x.name for x in found})
                    if label == "x[1:]":
                        # a slice is the list's own class; a view's, a TransformList
                        want = "TransformList" if issubclass(cls, TransformList) else name
                        self.assertEqual(got, [want])
                    elif cls is UVList:
                        # jedi reads MeshList's brackets before UVList's own
                        self.assertIn(got, (["MeshData"], ["UVData"]))
                    else:
                        self.assertEqual(got, [SHIPPED[cls].__name__])


if __name__ == "__main__":
    unittest.main()
