"""forward(): one line in a list class passes a member of its items on.

``merge = forward(MeshData.merge)`` in a list class's body makes a plain
function that calls each item's ``merge``, by name, in order; a property or a
field's name makes a real property, writable when the line gives
``per_item_ndim``. Names are fixed, and checked against the item class, when
the list class is made.
"""

import inspect
import pickle
import types
import unittest
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np
from cgmath.data import Data, DataList, FORWARD_MARK, forward


@dataclass(repr=False, eq=False)
class _Pin(Data):
    name:   Optional[str] = None
    weight: float = 1.0

    def bump(self, amount: float = 1.0, *, scale: float = 1.0) -> float:
        """adds ``amount`` times ``scale`` to the weight; returns the new weight"""
        self.weight += amount * scale
        return self.weight

    def tags(self) -> list:
        return [self.name, self.name]

    @property
    def doubled(self) -> float:
        """twice the weight"""
        return self.weight * 2.0

    @doubled.setter
    def doubled(self, value) -> None:
        self.weight = float(value) / 2.0

    @property
    def offset(self) -> np.ndarray:
        """the weight on x, y, z"""
        return np.full(3, self.weight)

    @offset.setter
    def offset(self, value) -> None:
        self.weight = float(np.asarray(value)[0])

    @property
    def frozen(self) -> float:
        return self.weight


@dataclass(repr=False, eq=False)
class _HeavyPin(_Pin):
    def bump(self, amount: float = 1.0, *, scale: float = 1.0) -> float:
        """a heavy pin bumps ten times as far"""
        self.weight += 10.0 * amount * scale
        return self.weight


class _Pins(DataList[_Pin]):
    bump_none  = forward(_Pin.bump)
    bump_list  = forward(_Pin.bump, returns="list")
    bump_array = forward(_Pin.bump, returns="array", doc="What only the list says.")
    tag_lists  = forward(_Pin.tags, returns="list")
    doubled    = forward(_Pin.doubled, returns="array", per_item_ndim=1)
    offset     = forward(_Pin.offset, returns="array", per_item_ndim=2)
    frozen     = forward(_Pin.frozen, returns="array")
    weights    = forward("weight", returns="list", per_item_ndim=1)
    names      = forward("name")


class _Echo(DataList[_Pin]):
    """a list whose method hands back another list class, named before it is defined"""
    again = forward(_Pin.tags, returns="_EchoBack")
    same  = forward(_Pin.tags, returns="same")


class _EchoBack(DataList[_Pin]):
    pass


_K = TypeVar("_K")
_T = TypeVar("_T")


class _Pair(DataList[_T], Generic[_K, _T]):
    """a generic list: its subclasses name the item class"""
    weights = forward("weight", returns="list")


class _PinPair(_Pair[str, _Pin]):
    pass


def _declare(*bases, **body):
    """a list class made as a class statement makes it, with ``bases`` and ``body``"""
    body = {"__module__": __name__, **body}
    return types.new_class("_Probe", bases, {}, lambda namespace: namespace.update(body))


def _pins(*weights):
    return _Pins([_Pin(name=f"p{i}", weight=w) for i, w in enumerate(weights)])


class TestForwardModes(unittest.TestCase):
    def test_each_return_mode(self):
        pins = _pins(1.0, 2.0)
        self.assertIsNone(pins.bump_none(1.0))
        np.testing.assert_array_equal([p.weight for p in pins], [2.0, 3.0])
        self.assertEqual(pins.bump_list(1.0, scale=2.0), [4.0, 5.0])
        found = pins.bump_array(0.5)
        self.assertIs(type(found), np.ndarray)
        np.testing.assert_array_equal(found, [4.5, 5.5])
        self.assertEqual(pins.tag_lists(), [["p0", "p0"], ["p1", "p1"]])

    def test_a_list_class_returned(self):
        echo = _Echo([_Pin(name="a"), _Pin(name="b")])
        # tags gives a list per pin: a list class built from them
        with self.assertRaises(TypeError):
            echo.again()
        self.assertIs(_Echo.again.__cgmath_forward__.returned_class(), _EchoBack)
        self.assertEqual(inspect.signature(_Echo.same).return_annotation, "_Echo")

    def test_an_empty_list(self):
        pins = _Pins()
        self.assertIsNone(pins.bump_none())
        self.assertEqual(pins.bump_list(),        [])
        self.assertEqual(pins.bump_array().shape, (0,))
        self.assertEqual(pins.weights,            [])

    def test_an_item_subclass_runs_its_own(self):
        pins = _Pins([_Pin(weight=1.0), _HeavyPin(weight=1.0)])
        self.assertEqual(pins.bump_list(1.0), [2.0, 11.0])

    def test_a_list_changed_while_called_is_read_once(self):
        pins = _pins(1.0, 2.0)

        class Growing(_Pin):
            def bump(self, amount: float = 1.0, *, scale: float = 1.0) -> float:
                pins.append(_Pin(weight=0.0))
                return super().bump(amount, scale=scale)

        pins[0] = Growing(weight=1.0)
        self.assertEqual(len(pins.bump_list(1.0)), 2)


class TestForwardProperties(unittest.TestCase):
    def test_a_real_property(self):
        for name in ("doubled", "offset", "frozen", "weights", "names"):
            with self.subTest(name):
                self.assertIs(type(vars(_Pins)[name]), property)
        self.assertIsNone(vars(_Pins)["frozen"].fset)
        self.assertIsNone(vars(_Pins)["names"].fset)

    def test_reads(self):
        pins = _pins(1.0, 2.0)
        np.testing.assert_array_equal(pins.doubled, [2.0, 4.0])
        np.testing.assert_array_equal(pins.offset, [[1.0] * 3, [2.0] * 3])
        self.assertEqual(pins.weights, [1.0, 2.0])
        self.assertEqual(pins.names, ["p0", "p1"])

    def test_a_write_per_item_or_whole(self):
        pins         = _pins(1.0, 2.0)
        pins.doubled = [10.0, 20.0]  # k = 1: one per item
        self.assertEqual(pins.weights, [5.0, 10.0])
        pins.doubled = 8.0  # a scalar: every item
        self.assertEqual(pins.weights, [4.0, 4.0])
        pins.offset = [[3.0, 0, 0], [6.0, 0, 0]]  # k = 2: one row per item
        self.assertEqual(pins.weights, [3.0, 6.0])
        pins.offset = [9.0, 0, 0]  # one row: every item
        self.assertEqual(pins.weights, [9.0, 9.0])
        pins.weights = 2.5
        self.assertEqual(pins.weights, [2.5, 2.5])

    def test_a_value_deeper_than_per_item_ndim_goes_whole(self):
        pins         = _pins(1.0, 2.0)
        value        = [[1.0, 2.0], [3.0, 4.0]]
        pins.weights = value  # k = 1, a value 2 deep: every item the whole value
        self.assertEqual(pins.weights, [value, value])

    def test_a_write_of_the_wrong_length(self):
        pins = _pins(1.0, 2.0, 3.0)
        with self.assertRaises(IndexError):
            pins.doubled = [10.0, 20.0]
        self.assertEqual(pins.weights, [5.0, 10.0, 3.0])  # two written first
        pins.doubled = [2.0, 4.0, 6.0, 8.0]  # the fourth is left over
        self.assertEqual(pins.weights, [1.0, 2.0, 3.0])

    def test_a_read_only_one_refuses_a_write(self):
        pins = _pins(1.0)
        with self.assertRaisesRegex(AttributeError, "'frozen' is a read-only property of _Pins"):
            pins.frozen = 3.0


class TestForwardFunctions(unittest.TestCase):
    def test_a_plain_function(self):
        method = vars(_Pins)["bump_array"]
        self.assertTrue(inspect.isfunction(method))
        self.assertEqual(method.__name__,     "bump_array")
        self.assertEqual(method.__qualname__, "_Pins.bump_array")
        self.assertEqual(method.__module__,   __name__)
        self.assertIs(method.__wrapped__, _Pin.bump)

    def test_the_items_parameters(self):
        for name in ("bump_none", "bump_list", "bump_array"):
            with self.subTest(name):
                self.assertEqual(
                    list(inspect.signature(getattr(_Pins, name)).parameters.values()),
                    list(inspect.signature(_Pin.bump).parameters.values()),
                )
        self.assertIsNone(inspect.signature(_Pins.bump_none).return_annotation)
        self.assertIs(inspect.signature(_Pins.bump_list).return_annotation, list)
        self.assertIs(inspect.signature(_Pins.bump_array).return_annotation, np.ndarray)

    def test_the_docstring(self):
        lines = _Pins.bump_array.__doc__.splitlines()
        self.assertEqual(lines[0], "_Pin.bump() on every item, in order")
        self.assertIn("What only the list says.", _Pins.bump_array.__doc__)
        self.assertIn("adds ``amount`` times ``scale``", _Pins.bump_array.__doc__)
        self.assertEqual(vars(_Pins)["doubled"].__doc__.splitlines()[0], "_Pin.doubled of every item")
        self.assertEqual(vars(_Pins)["names"].__doc__.splitlines()[0], "_Pin.name of every item")

    def test_a_method_refuses_a_write(self):
        with self.assertRaisesRegex(AttributeError, "'bump_list' is a method of _Pins, not a setting"):
            _pins(1.0).bump_list = 3

    def test_the_marker(self):
        spec = getattr(_Pins.bump_array, FORWARD_MARK)
        self.assertEqual((spec.label, spec.returns, spec.attribute), ("_Pin.bump", "array", "bump_array"))
        prop = vars(_Pins)["doubled"]
        for function in (prop.fget, prop.fset):
            spec = getattr(function, FORWARD_MARK)
            self.assertEqual((spec.label, spec.per_item_ndim), ("_Pin.doubled", 1))

    def test_pickled(self):
        self.assertIs(pickle.loads(pickle.dumps(_Pins.bump_list)), _Pins.bump_list)
        pins  = _pins(1.0, 2.0)
        bound = pickle.loads(pickle.dumps(pins.bump_list))
        self.assertEqual(bound(1.0), [2.0, 3.0])


class TestForwardChecks(unittest.TestCase):
    def test_a_member_the_item_class_lacks(self):
        with self.assertRaisesRegex(TypeError, "forwards the field 'wieght', which _Pin does not declare"):
            _declare(DataList[_Pin], x=forward("wieght"))

        @dataclass(repr=False, eq=False)
        class Other(Data):
            def spin(self):
                pass

        with self.assertRaisesRegex(TypeError, "forwards .*spin, which _Pin does not have"):
            _declare(DataList[_Pin], spin=forward(Other.spin))

    def test_an_inherited_line_is_checked_against_the_new_item_class(self):
        @dataclass(repr=False, eq=False)
        class Bare(Data):
            name: Optional[str] = None

        # _Pins' lines, inherited by a list of an item class that lacks them
        with self.assertRaisesRegex(TypeError, "_Probe\\.\\w+ forwards .*, which Bare does not (have|declare)"):
            _declare(_Pins, DataList[Bare])

    def test_a_name_a_list_needs(self):
        for name in ("copy", "save", "load", "index", "count", "sort", "append", "extend", "insert", "remove", "pop"):
            with self.subTest(name), self.assertRaisesRegex(TypeError, f"{name!r} is \\w+'s own member"):
                _declare(DataList[_Pin], **{name: forward(_Pin.bump)})

    def test_not_a_member(self):
        with self.assertRaises(TypeError):
            forward(len)
        with self.assertRaises(TypeError):
            forward(_Pin.bump, per_item_ndim=1)
        with self.assertRaises(TypeError):
            forward(_Pin.bump, returns=3)

    def test_a_generic_list(self):
        pair = _PinPair([_Pin(weight=4.0)])
        self.assertEqual(pair.weights, [4.0])
        with self.assertRaisesRegex(TypeError, "which _Pin does not declare"):
            _declare(_Pair[str, _Pin], x=forward("nope"))
        # the generic list itself checks nothing: it holds no class yet
        _declare(DataList[_T], Generic[_T], x=forward("nope"))


if __name__ == "__main__":
    unittest.main()
