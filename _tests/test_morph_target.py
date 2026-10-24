import operator
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view
from cgmath.geometry import morph_target
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.morph_target import MorphData, MorphList
from cgmath.geometry.skin_weights import SkinData, SkinList

try:
    from pxr import Usd
except ImportError:
    Usd = None

# the folder holding the cgmath package, for a fresh interpreter to import it
PROJECTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# a Data with no * of its own: skin * morph reaches the morph's __rmul__
SKIN = SkinData(weights=np.array([[1.0]]), influences=["root"])

# a DataList that is no MorphList, named as itself, not by the items it holds
SKINS = SkinList([SKIN.copy(), SKIN.copy(), SKIN.copy()])



class TestMorphList(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.jawOpen = MorphData(name='jawOpen',
                                       indices=np.array([5, 3, 4]),
                                       offsets=np.array([[ 0.0,  0.0,  0.0],
                                                         [ 9.5,  1.5,  2.5],
                                                         [-0.5,  1.5, -2.5]]))

        self.jawClosed = MorphData(name='jawClosed',
                                         indices=np.array([1, 6, 4]),
                                         offsets=np.array([[ 0.0,  0.0,  0.0],
                                                           [ 9.5,  1.5,  2.5],
                                                           [-0.5,  1.5, -2.5]]))

        self.eyesClosed = MorphData(name='eyesClosed',
                                          indices=np.array([3, 12, 1]),
                                          offsets=np.array([[ 10.0,  10.0,  10.0],
                                                            [ 19.5,  11.5,  12.5],
                                                            [-10.5,  11.5, -12.5]]))


        self.list = MorphList([self.eyesClosed, self.jawClosed, self.jawOpen])


    def test_bytes(self):
        obj1 = self.list.copy()
        b    = obj1.to_bytes()
        obj2 = MorphList.from_bytes(b)
        self.assertTrue(obj1 == obj2)


    def test_serialize(self):

        obj1 = self.list.copy()

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # save obj1 to file
            obj1.save(f)

            # load obj2 from file
            obj2 = MorphList.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # do the same via pickle, put obj1 inside a list
            with open(f, "wb") as io:
                pickle.dump([obj1], io)

            with open(f, "rb") as io:
                obj3 = pickle.load(io)[0]

                # confirm they're identical
                self.assertTrue(obj1 == obj3)

            # test json and npz
            obj1.save(f, mode='json')
            obj4 = MorphList.load(f)
            self.assertTrue(obj1 == obj4)

            obj1.save(f, mode='npz')
            obj5 = MorphList.load(f)
            self.assertTrue(obj1 == obj5)


    def test_serialize_array_types(self):
        """indices is an optional field, make sure it still loads back as an array"""

        obj1 = self.list.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ('pkl', 'json', 'npz'):
                f = os.path.join(temp_dir, "test." + mode)
                obj1.save(f, mode=mode)
                obj2 = MorphList.load(f)

                for shape in obj2:
                    self.assertIsInstance(shape.indices, np.ndarray, msg=mode)
                    self.assertIsInstance(shape.offsets, np.ndarray, msg=mode)

                # operations masking indices fail on anything but an array
                obj2.prune_offsets()
                self.assertEqual(obj2.size.tolist(), [3, 2, 2], msg=mode)


    def test_list_like(self):

        # randomize list
        import random

        obj1 = self.list.copy()
        obj2 = self.list.copy()

        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj2.sort()
        self.assertTrue(obj1 == obj2)

        # use sorted
        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj3 = MorphList(sorted(obj2))
        self.assertTrue(obj1 == obj3)


        # sum a list like a list of numbers
        obj3 = sum(obj1)
        self.assertTrue(obj3 == (self.eyesClosed + self.jawClosed + self.jawOpen))

        # array indexing, slicing
        self.assertTrue(obj1[1] == self.jawClosed)
        self.assertTrue(sum(obj1[:2]) == (self.eyesClosed + self.jawClosed))

        # reverse
        obj1.sort()
        obj2.sort()
        obj1.reverse()
        self.assertTrue(obj1 == obj2[::-1])

        # numpy like indexing
        obj1.sort()
        obj2.sort()
        self.assertTrue(sum(obj1[[1, 2]]) == (obj2[1]+obj2[2]))

        # mutability
        eyesClosed   = self.eyesClosed.copy()
        jawClosed    = self.jawClosed.copy()
        jawOpen      = self.jawOpen.copy()

        obj1         = MorphList([eyesClosed, jawClosed, jawOpen])
        obj2         = obj1[[2, 1, 0]]
        jawOpen.name = 'POOF!'
        self.assertTrue(obj1[2].name == 'POOF!')
        self.assertTrue(obj2[0].name == 'POOF!')

        # dict like indexting
        obj1 = self.list.copy()
        self.assertTrue(obj2['jawClosed'] == self.jawClosed)

        index = obj1.index('eyesClosed')
        self.assertTrue(obj1[index] == self.eyesClosed)

        # is a shape in the list?
        self.assertTrue(obj1[0] in obj1)                # shape object in list by name
        self.assertTrue('eyesClosed' in obj1)           # string in list names
        self.assertFalse('SDFAFDAEWFASDFAFSD' in obj1)  # string not in list names

    def test_functions(self):
        obj1 = self.list.copy()

        # pattern matching
        obj2 = obj1.match("jaw*")
        self.assertTrue(sum(obj2) == (obj1['jawClosed'] + obj1['jawOpen']))

        # indexing workflow
        indices = np.where(obj1.mse < 15)[0]
        obj2    = sum(obj1[indices])
        self.assertTrue(obj2 == (obj1['jawClosed'] + obj1['jawOpen']))


    def test_vectorize(self):

        # add
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  += self.jawOpen
        shape1 += self.jawOpen
        self.assertTrue(list1[0] == shape1)

        # sub
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  -= self.jawOpen
        shape1 -= self.jawOpen
        self.assertTrue(list1[0] == shape1)

        # mult
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  *= 2
        shape1 *= 2
        self.assertTrue(list1[0] == shape1)

        # div
        list1  = self.list.copy()
        shape1 = list1[0].copy()

        list1  /= 2
        shape1 /= 2
        self.assertTrue(list1[0] == shape1)


        # size
        list1 = self.list.copy()
        self.assertTrue(np.allclose(list1.size, [3, 3, 3]))

        # min
        self.assertTrue(np.allclose(list1.min, [17.32050808, 0, 0]))

        # max
        self.assertTrue(np.allclose(list1.max, [25.86020108,  9.93730346,  9.93730346]))

        # mse
        self.assertTrue(np.allclose(list1.mse, [151.94444444,  11.94444444,  11.94444444]))

    def test_operators_take_any_sequence(self):
        # shape by shape against a MorphList, a python list, a tuple or an
        # array of the same length, the same answer each way, the operands
        # left as they were; a list or a tuple on the left too, item first.
        # + and - take shapes, * and / numbers, one a shape
        shapes = [self.jawOpen.copy(), self.eyesClosed.copy(), self.jawClosed.copy()]
        before = self.list.copy(), MorphList(shapes).copy()
        ops = {
            "+": (lambda a, b: a + b, shapes),
            "-": (lambda a, b: a - b, shapes),
            "*": (lambda a, b: a * b, [2, 3, 4]),
            "/": (lambda a, b: a / b, [2, 4, 8]),
        }
        inplace = {"+": operator.iadd, "-": operator.isub, "*": operator.imul, "/": operator.itruediv}
        for name, (op, others) in ops.items():
            expected = MorphList([op(x.copy(), y) for x, y in zip(self.list, others)])
            first    = MorphList(others) if others is shapes else np.array(others)
            for other in (first, list(others), tuple(others)):
                with self.subTest(op=name, other=type(other).__name__, of=type(others[0]).__name__):
                    obj = self.list.copy()
                    if name == "+":
                        obj += other
                    elif name == "-":
                        obj -= other
                    elif name == "*":
                        obj *= other
                    else:
                        obj /= other
                    self.assertIsInstance(obj, MorphList)
                    self.assertTrue(obj == expected)
                    self.assertTrue(op(self.list.copy(), other) == expected)
                    self.assertTrue((self.list, MorphList(shapes)) == before)

            # on the left: each item op the shape it pairs with, names too;
            # a number over a shape has no meaning
            if name != "/":
                expected = MorphList([op(y, x.copy()) for x, y in zip(self.list, others)])
                for other in (list(others), tuple(others)):
                    with self.subTest(op=name, left=type(other).__name__, of=type(others[0]).__name__):
                        obj = op(other, self.list)
                        self.assertIsInstance(obj, MorphList)
                        self.assertTrue(obj == expected)
                        self.assertEqual([x.name for x in obj], [x.name for x in expected])
                        self.assertTrue((self.list, MorphList(shapes)) == before)

            # a sequence of another length, shorter or longer, is refused, on
            # either side and in place, the list left as it was
            for wrong in (others[:2], others + others[:1]):
                pairs = [(self.list.copy(), wrong)]
                if name != "/":
                    pairs.append((wrong, self.list.copy()))
                for args in pairs:
                    with self.subTest(op=name, length=len(wrong), left=args[0] is wrong):
                        with self.assertRaisesRegex(RuntimeError, "same length"):
                            op(*args)
                with self.subTest(op=name + "=", length=len(wrong)):
                    obj = self.list.copy()
                    with self.assertRaisesRegex(RuntimeError, "same length"):
                        inplace[name](obj, wrong)
                    self.assertTrue(obj == self.list)

        # 0 + a shape is a copy of the shape: the sum holds none of the operand's shapes
        obj = [0, 0, 0] + self.list
        self.assertTrue(obj == self.list)
        self.assertFalse(any(x is y for x in obj for y in self.list))

    def test_numbers_divide_shape_by_shape(self):
        # one number a shape, as * takes them: (shapes * s) / s is shapes
        for s in ([2.0, 4.0, 0.5], (2.0, 4.0, 0.5), np.array([2.0, 4.0, 0.5])):
            with self.subTest(s=type(s).__name__):
                scaled = self.list * s
                self.assertTrue(scaled / s == self.list)
                for shape, by, back in zip(self.list, s, scaled / s):
                    np.testing.assert_array_equal(back.offsets, (shape * by).offsets / by)
                obj = scaled.copy()
                obj /= s
                self.assertTrue(obj == self.list)
        # one number divides every shape
        halved = self.list / 2
        for shape, half in zip(self.list, halved):
            np.testing.assert_array_equal(half.offsets, shape.offsets / 2)
        # a sequence of another length, shorter or longer, is refused before
        # any shape is divided
        for wrong in ([2.0, 4.0], [2.0, 4.0, 0.5, 8.0]):
            with self.subTest(length=len(wrong)):
                obj = self.list.copy()
                with self.assertRaisesRegex(RuntimeError, "same length"):
                    obj /= wrong
                self.assertTrue(obj == self.list)

    def test_what_has_no_meaning_raises(self):
        # a number added or subtracted (it shifted the indices), a number
        # less or over the list, a morph times or over a morph, a power:
        # TypeError naming what has a meaning, nothing changed in place; a
        # list, a tuple or an array named by the item refused
        shapes = self.list
        before = self.list.copy()
        sums   = "has no meaning: a MorphData adds or subtracts another MorphData"
        scales = "has no meaning: a MorphData multiplies or divides by a number"
        cases = (
            ("shapes + 2", lambda x: x + 2, r"^MorphList \+ int " + sums),
            ("shapes - 2.5", lambda x: x - 2.5, r"^MorphList - float " + sums),
            ("2 + shapes", lambda x: 2 + x, r"^int \+ MorphList " + sums),
            ("5 - shapes", lambda x: 5 - x, r"^int - MorphList " + sums),
            ("shapes + [shape, 2, shape]", lambda x: x + [self.jawOpen, 2, self.jawOpen], r"^MorphList \+ a list holding an int " + sums),
            ("[1, 2, 3] + shapes", lambda x: [1, 2, 3] + x, r"^a list holding an int \+ MorphList " + sums),
            ("shapes + array", lambda x: x + np.array([2.0, 2.0, 2.0]), r"^MorphList \+ an array holding a float64 " + sums),
            ("shapes * shapes", lambda x: x * x.copy(), r"^MorphList \* MorphList " + scales),
            ("shapes * shape", lambda x: x * self.jawOpen, r"^MorphList \* MorphData " + scales),
            ("shape * shapes", lambda x: self.jawOpen * x, r"^MorphData \* MorphList " + scales),
            ("skin * shapes", lambda x: SKIN * x, r"^SkinData \* MorphList " + scales),
            ("shapes + skins", lambda x: x + SKINS, r"^MorphList \+ SkinList " + sums),
            ("skins + shapes", lambda x: SKINS + x, r"^SkinList \+ MorphList " + sums),
            ("shapes * [2, shape, 2]", lambda x: x * [2, self.jawOpen, 2], r"^MorphList \* a list holding a MorphData " + scales),
            ("shapes / shapes", lambda x: x / x.copy(), r"^MorphList / MorphList " + scales),
            ("2 / shapes", lambda x: 2 / x, r"^int / MorphList " + scales),
            ("[1, 2, 3] / shapes", lambda x: [1, 2, 3] / x, r"^list / MorphList " + scales),
            ("shapes ** 2", lambda x: x ** 2, r"unsupported operand type\(s\) for \*\* or pow\(\)"),
            ("2 ** shapes", lambda x: 2 ** x, r"unsupported operand type\(s\) for \*\* or pow\(\)"),
        )
        for label, op, refused in cases:
            with self.subTest(op=label):
                obj = shapes.copy()
                with self.assertRaisesRegex(TypeError, refused):
                    op(obj)
                self.assertTrue(obj == before)

        # in place, refused before any shape is changed
        inplace = (
            ("+= 2", lambda x: x.__iadd__(2)),
            ("-= [shape, shape, 2]", lambda x: x.__isub__([self.jawOpen, self.jawOpen, 2])),
            ("*= [2, 2, shape]", lambda x: x.__imul__([2, 2, self.jawOpen])),
            ("/= [2, 2, shape]", lambda x: x.__itruediv__([2, 2, self.jawOpen])),
        )
        for label, op in inplace:
            with self.subTest(op=label):
                obj   = shapes.copy()
                items = list(obj)
                with self.assertRaises(TypeError):
                    op(obj)
                self.assertTrue(obj == before)
                self.assertTrue(all(x is y for x, y in zip(obj, items)))
        with self.assertRaises(TypeError):
            obj = shapes.copy()
            obj **= 2

        # whatever the two lengths: an empty list refuses what a full one
        # does, and an operand of another length that has no meaning raises
        # TypeError, not the RuntimeError of a length that does not fit; a
        # list of MorphLists, on either side, of the list's length and of
        # another. Each list acted on, in place or not, left as it was
        empty = MorphList([])
        two   = MorphList([self.jawOpen.copy(), self.jawClosed.copy()])
        lists = [two, two, two]
        lengths = (
            ("empty + 2", empty, lambda x: x + 2, r"^MorphList \+ int " + sums),
            ("empty - 2", empty, lambda x: x - 2, r"^MorphList - int " + sums),
            ("empty += 2", empty.copy(), lambda x: x.__iadd__(2), r"^MorphList \+ int " + sums),
            ("empty * shape", empty, lambda x: x * self.jawOpen, r"^MorphList \* MorphData " + scales),
            ("empty / empty", empty, lambda x: x / empty.copy(), r"^MorphList / MorphList " + scales),
            ("[2] + empty", empty, lambda x: [2] + x, r"^a list holding an int \+ MorphList " + sums),
            ("[shape] * empty", empty, lambda x: [self.jawOpen] * x, r"^a list holding a MorphData \* MorphList " + scales),
            ("shapes * two", shapes, lambda x: x * two, r"^MorphList \* MorphList " + scales),
            ("shapes /= two", shapes.copy(), lambda x: x.__itruediv__(two), r"^MorphList / MorphList " + scales),
            ("shapes + [2, 2]", shapes, lambda x: x + [2, 2], r"^MorphList \+ a list holding an int " + sums),
            ("shapes -= [2, 2]", shapes.copy(), lambda x: x.__isub__([2, 2]), r"^MorphList - a list holding an int " + sums),
            ("shapes * [2, shape]", shapes, lambda x: x * [2, self.jawOpen], r"^MorphList \* a list holding a MorphData " + scales),
            ("[2, 2] - shapes", shapes, lambda x: [2, 2] - x, r"^a list holding an int - MorphList " + sums),
            ("[2, shape] * shapes", shapes, lambda x: [2, self.jawOpen] * x, r"^a list holding a MorphData \* MorphList " + scales),
            ("shapes + [two, two, two]", shapes, lambda x: x + lists, r"^MorphList \+ a list holding a MorphList " + sums),
            ("shapes -= [two, two, two]", shapes.copy(), lambda x: x.__isub__(lists), r"^MorphList - a list holding a MorphList " + sums),
            ("shapes += (two, two)", shapes.copy(), lambda x: x.__iadd__((two, two)), r"^MorphList \+ a tuple holding a MorphList " + sums),
            ("[two, two, two] + shapes", shapes, lambda x: lists + x, r"^a list holding a MorphList \+ MorphList " + sums),
            ("[two, two] - shapes", shapes, lambda x: lists[:2] - x, r"^a list holding a MorphList - MorphList " + sums),
            ("[two, two, two] * shapes", shapes, lambda x: lists * x, r"^a list holding a MorphList \* MorphList " + scales),
        )
        for label, obj, op, refused in lengths:
            with self.subTest(op=label):
                was   = obj.copy()
                items = list(obj)
                with self.assertRaisesRegex(TypeError, refused):
                    op(obj)
                self.assertTrue(obj == was)
                self.assertTrue(all(x is y for x, y in zip(obj, items)))
        self.assertTrue(shapes == before)
        self.assertTrue(two == MorphList([self.jawOpen, self.jawClosed]))
        # what has a meaning still works: an empty list plus one, a 0 on the left
        self.assertTrue(empty + MorphList([]) == empty)
        self.assertTrue([0] + MorphList([self.jawOpen]) == MorphList([self.jawOpen]))

    def test_zero_gives_a_new_list(self):
        # 0 + shapes and sum() of one list give a copy, 0 - shapes every shape negated
        for obj in (0 + self.list, sum([self.list])):
            self.assertTrue(obj == self.list)
            self.assertIsNot(obj, self.list)
            self.assertFalse(any(x is y for x in obj for y in self.list))
        obj = 0 - self.list
        self.assertTrue(obj == self.list * -1)
        self.assertFalse(any(x is y for x in obj for y in self.list))

    def test_a_shape_held_twice_is_acted_on_once_in_each_place(self):
        # a new list holds a new shape in each place: a shape the list holds
        # twice is acted on once in each, the operands left as they were
        a     = MorphData(name="a", offsets=np.array([[1.0, 0.0, 0.0]]), indices=np.array([0]))
        b     = MorphData(name="b", offsets=np.array([[0.0, 1.0, 0.0]]), indices=np.array([0]))
        twice = MorphList([a, a])
        cases = (
            ("twice * 2", lambda x: x * 2, [[2, 0, 0], [2, 0, 0]]),
            ("2 * twice", lambda x: 2 * x, [[2, 0, 0], [2, 0, 0]]),
            ("twice / 2", lambda x: x / 2, [[0.5, 0, 0], [0.5, 0, 0]]),
            ("twice * [2, 4]", lambda x: x * [2, 4], [[2, 0, 0], [4, 0, 0]]),
            ("twice / [2, 4]", lambda x: x / [2, 4], [[0.5, 0, 0], [0.25, 0, 0]]),
            ("0 - twice", lambda x: 0 - x, [[-1, 0, 0], [-1, 0, 0]]),
            ("0 + twice", lambda x: 0 + x, [[1, 0, 0], [1, 0, 0]]),
            ("twice + b", lambda x: x + b, [[1, 1, 0], [1, 1, 0]]),
            ("twice - [b, b]", lambda x: x - [b, b], [[1, -1, 0], [1, -1, 0]]),
            ("[b, b] - twice", lambda x: [b, b] - x, [[-1, 1, 0], [-1, 1, 0]]),
            ("sum([twice, twice])", lambda x: sum([x, x]), [[2, 0, 0], [2, 0, 0]]),
        )
        for label, op, expected in cases:
            with self.subTest(op=label):
                obj = op(twice)
                self.assertEqual([shape.offsets.tolist() for shape in obj], [[row] for row in expected])
                self.assertIsNot(obj[0], obj[1])
                self.assertTrue(twice[0] is twice[1] is a)
                self.assertEqual(a.offsets.tolist(), [[1.0, 0.0, 0.0]])
                self.assertEqual(b.offsets.tolist(), [[0.0, 1.0, 0.0]])
        # so sum() of one list scales in place once each
        total = sum([twice])
        total *= 2
        self.assertEqual([shape.offsets.tolist() for shape in total], [[[2.0, 0.0, 0.0]]] * 2)
        # in place, the list's own shapes change: the one it holds twice,
        # twice, whichever the operator
        in_place = (
            ("+= b", operator.iadd, b, [[1.0, 2.0, 0.0]]),
            ("-= b", operator.isub, b, [[1.0, -2.0, 0.0]]),
            ("*= 2", operator.imul, 2, [[4.0, 0.0, 0.0]]),
            ("/= 2", operator.itruediv, 2, [[0.25, 0.0, 0.0]]),
        )
        for label, op, other, expected in in_place:
            with self.subTest(in_place=label):
                shape = a.copy()
                obj   = MorphList([shape] * 2)
                self.assertIs(op(obj, other), obj)
                self.assertTrue(obj[0] is obj[1] is shape)
                self.assertEqual(shape.offsets.tolist(), expected)

    def test_in_place_takes_the_lists_own_shapes_as_they_were(self):
        # in place, an operand that is, or holds, one of the list's own
        # shapes is taken as it was before any shape changed, as a new list
        # takes it: on 58bbdde a shape changed in place was the operand of
        # the shapes after it as it had become, so shapes -= shapes[0] left
        # the second shape as it was. The list's shapes still change in
        # place: a caller's reference to one sees it, and a shape held twice
        # is changed in each place, each time by the operand as it was
        def make(*rows):
            return MorphList([MorphData(name=f"s{i}", offsets=np.array([row], dtype=float), indices=np.array([0]))
                              for i, row in enumerate(rows)])

        two   = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        three = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        cases = (
            ("L -= L[0]", two, operator.isub, lambda L: L[0], [[0, 0, 0], [-1, 1, 0]]),
            ("L += L[0]", two, operator.iadd, lambda L: L[0], [[2, 0, 0], [1, 1, 0]]),
            ("L += [L[1], L[0]]", two, operator.iadd, lambda L: [L[1], L[0]], [[1, 1, 0], [1, 1, 0]]),
            ("L -= [L[1], L[0]]", two, operator.isub, lambda L: [L[1], L[0]], [[1, -1, 0], [-1, 1, 0]]),
            ("L += (L[1], L[0])", two, operator.iadd, lambda L: (L[1], L[0]), [[1, 1, 0], [1, 1, 0]]),
            ("L += MorphList([L[1], L[0]])", two, operator.iadd, lambda L: MorphList([L[1], L[0]]),
             [[1, 1, 0], [1, 1, 0]]),
            ("L -= MorphList([L[1], L[0]])", two, operator.isub, lambda L: MorphList([L[1], L[0]]),
             [[1, -1, 0], [-1, 1, 0]]),
            ("L += L[::-1]", two, operator.iadd, lambda L: L[::-1], [[1, 1, 0], [1, 1, 0]]),
            ("L -= L[::-1]", two, operator.isub, lambda L: L[::-1], [[1, -1, 0], [-1, 1, 0]]),
            ("L += L", two, operator.iadd, lambda L: L, [[2, 0, 0], [0, 2, 0]]),
            ("L -= L", two, operator.isub, lambda L: L, [[0, 0, 0], [0, 0, 0]]),
            # each place's operand a shape an earlier place changed
            ("L += [L[2], L[0], L[1]]", three, operator.iadd, lambda L: [L[2], L[0], L[1]],
             [[1, 0, 1], [1, 1, 0], [0, 1, 1]]),
            ("L -= [L[1], L[2], L[0]]", three, operator.isub, lambda L: [L[1], L[2], L[0]],
             [[1, -1, 0], [0, 1, -1], [-1, 0, 1]]),
        )
        for label, rows, op, operand, expected in cases:
            with self.subTest(label):
                obj    = make(*rows)
                shapes = list(obj)
                self.assertIs(op(obj, operand(obj)), obj)
                self.assertTrue(all(x is y for x, y in zip(obj, shapes)))
                self.assertEqual([shape.offsets.tolist() for shape in obj], [[row] for row in expected])

        # * and / by arrays that hold the shapes' offsets: a's rows, and its
        # whole offsets as the operand of each shape, each as it was. The
        # first shape's offsets a view of another array's memory, as a
        # MorphData keeps an array it is given; and views of a's offsets
        # whose .base is no array, which only their bounds tie to its memory:
        # as_strided's, sliding_window_view's, a memoryview's
        def scaled():
            memory = np.arange(1.0, 7.0).reshape(2, 3)
            a      = MorphData(name="a", offsets=memory, indices=np.array([0, 1]))
            b      = MorphData(name="b", offsets=np.full((2, 3), 2.0), indices=np.array([0, 1]))
            return a, b, MorphList([a, b])

        a0, b0, _ = scaled()
        a0, b0 = a0.offsets.copy(), b0.offsets.copy()
        times = (a0 * a0[0], b0 * a0[1])
        over  = (a0 / a0[0], b0 / a0[1])
        for label, op, operand, expected in (
            ("L *= a.offsets", operator.imul, lambda a, b: a.offsets, times),
            ("L /= a.offsets", operator.itruediv, lambda a, b: a.offsets, over),
            ("L *= [b.offsets, a.offsets]", operator.imul, lambda a, b: [b.offsets, a.offsets], (a0 * b0, b0 * a0)),
            ("L /= [a.offsets[1], a.offsets[0]]", operator.itruediv, lambda a, b: [a.offsets[1], a.offsets[0]],
             (a0 / a0[1], b0 / a0[0])),
            ("L *= as_strided(a.offsets)", operator.imul, lambda a, b: as_strided(a.offsets), times),
            ("L /= sliding_window_view(a.offsets, (1, 3))[:, 0, 0]", operator.itruediv,
             lambda a, b: sliding_window_view(a.offsets, (1, 3))[:, 0, 0], over),
            ("L *= np.asarray(memoryview(a.offsets))", operator.imul,
             lambda a, b: np.asarray(memoryview(a.offsets)), times),
            ("L /= np.frombuffer(a.offsets.data)", operator.itruediv,
             lambda a, b: np.frombuffer(a.offsets.data).reshape(2, 3), over),
        ):
            with self.subTest(label):
                a, b, obj = scaled()
                self.assertFalse(a.offsets.flags.owndata)
                self.assertIs(op(obj, operand(a, b)), obj)
                self.assertTrue(obj[0] is a and obj[1] is b)
                np.testing.assert_allclose(a.offsets, expected[0], rtol=0, atol=1e-15)
                np.testing.assert_allclose(b.offsets, expected[1], rtol=0, atol=1e-15)

        # a's offsets themselves such a view, as_strided's of another array:
        # that array, and its rows, taken as they were
        def strided():
            memory = np.arange(1.0, 7.0).reshape(2, 3)
            a      = MorphData(name="a", offsets=as_strided(memory), indices=np.array([0, 1]))
            b      = MorphData(name="b", offsets=np.full((2, 3), 2.0), indices=np.array([0, 1]))
            return memory, a, b, MorphList([a, b])

        for label, op, operand, expected in (
            ("L *= the array a's offsets view", operator.imul, lambda memory: memory, times),
            ("L /= [its row 1, its row 0]", operator.itruediv, lambda memory: [memory[1], memory[0]],
             (a0 / a0[1], b0 / a0[0])),
        ):
            with self.subTest(label):
                memory, a, b, obj = strided()
                self.assertFalse(isinstance(a.offsets.base, np.ndarray))
                self.assertIs(op(obj, operand(memory)), obj)
                np.testing.assert_allclose(a.offsets, expected[0], rtol=0, atol=1e-15)
                np.testing.assert_allclose(b.offsets, expected[1], rtol=0, atol=1e-15)

        # a shape held twice, its offsets the operand of the place between:
        # taken as the first place left them
        with self.subTest("[a, b, a] *= [2, a.offsets, 3]"):
            a, b, _ = scaled()
            obj = MorphList([a, b, a])
            obj *= [2.0, a.offsets, 3.0]
            np.testing.assert_allclose(a.offsets, a0 * 6.0, rtol=0, atol=1e-15)
            np.testing.assert_allclose(b.offsets, b0 * a0, rtol=0, atol=1e-15)

        # a shape held twice, its operand that shape: changed in each place,
        # each time by the shape as it was, so three times it in all, and b
        # between its two places takes it as it was too
        for label, op, held, expected in (
            ("[a, a] += a", operator.iadd, 2, [[3, 0, 0], [3, 0, 0]]),
            ("[a, a] -= a", operator.isub, 2, [[-1, 0, 0], [-1, 0, 0]]),
            ("[a, b, a] -= a", operator.isub, 3, [[-1, 0, 0], [-1, 1, 0], [-1, 0, 0]]),
        ):
            with self.subTest(label):
                a, b = make(*two)
                obj = MorphList([a, a] if held == 2 else [a, b, a])
                self.assertIs(op(obj, a), obj)
                self.assertTrue(obj[0] is obj[-1] is a)
                self.assertEqual([shape.offsets.tolist() for shape in obj], [[row] for row in expected])

        # an operand is copied once, and only when an earlier place changes
        # it: shapes += shapes copies nothing, nor does an operand from
        # outside the list, and shapes -= shapes[0] copies the first shape
        # once for the two places after it
        real = MorphData.copy
        c    = MorphData(name="c", offsets=np.array([[0.0, 0.0, 2.0]]), indices=np.array([0]))
        d    = MorphData(name="d", offsets=np.array([[3.0, 0.0, 0.0]]), indices=np.array([0]))
        for label, rows, op, operand, copied in (
            ("L += L", two, operator.iadd, lambda L: L, []),
            ("L += c", two, operator.iadd, lambda L: c, []),
            ("L += [c, d]", two, operator.iadd, lambda L: [c, d], []),
            ("L -= MorphList([c, d])", two, operator.isub, lambda L: MorphList([c, d]), []),
            ("L -= L[0]", three, operator.isub, lambda L: L[0], ["s0"]),
            ("L += L[::-1]", three, operator.iadd, lambda L: L[::-1], ["s0"]),
            ("L += [L[2], L[0], L[1]]", three, operator.iadd, lambda L: [L[2], L[0], L[1]], ["s0", "s1"]),
        ):
            with self.subTest(copies=label):
                made = []

                def counted(shape):
                    made.append(shape.name)
                    return real(shape)

                obj = make(*rows)
                with mock.patch.object(MorphData, "copy", counted):
                    op(obj, operand(obj))
                self.assertEqual(made, copied)

        # and so is an array: each shape's own offsets at its own place, as
        # they are and as_strided's views of them, and arrays from outside
        # the list come back as they are, and so does the array a shape's
        # offsets view at that shape's place; a view of the first shape's
        # offsets comes back copied once for the two places after it
        obj = make(*three)
        own = [shape.offsets for shape in obj]
        for label, operand in (
            ("each shape's own offsets", own),
            ("as_strided views of them", [as_strided(x) for x in own]),
            ("arrays from outside", [np.ones((1, 3)), np.full((1, 3), 2.0), np.zeros((1, 3))]),
        ):
            with self.subTest(copies=label):
                got = obj._in_place_operands(operand, "*")
                self.assertTrue(all(x is y for x, y in zip(got, operand)))
        memory, a, b, pair = strided()
        got = pair._in_place_operands([memory, b.offsets], "*")
        self.assertTrue(got[0] is memory and got[1] is b.offsets)
        view = obj[0].offsets[0]
        got  = obj._in_place_operands([view] * 3, "/")
        self.assertIs(got[0], view)
        self.assertTrue(got[1] is got[2] and got[1] is not view)
        np.testing.assert_array_equal(got[1], view)

        # the rows of an array from outside the list find their owner through
        # .base, so no memory bounds are compared: a list of n shapes takes
        # no n * n checks
        table = np.arange(9.0).reshape(3, 3)
        with mock.patch.object(np, "may_share_memory", wraps=np.may_share_memory) as bounds:
            got = obj._in_place_operands(table, "*")
        self.assertEqual(bounds.call_count, 0)
        self.assertTrue(all(x.base is table.base for x in got))

        # and numbers alone, no array among them, look at no memory
        with mock.patch.object(morph_target, "_owner", wraps=morph_target._owner) as owner:
            obj *= [1.0, 2.0, 3.0]
            obj /= 2.0
        self.assertEqual(owner.call_count, 0)

        # a caller's reference to one of the list's shapes sees the change
        obj   = make(*two)
        first = obj[0]
        obj -= obj[0]
        self.assertIs(obj[0], first)
        self.assertEqual(first.offsets.tolist(), [[0.0, 0.0, 0.0]])

    def test_prune_offsets(self):
        obj = self.list.copy()
        obj.prune_offsets()
        # zero-magnitude offset rows are dropped within each shape
        self.assertTrue(np.allclose(obj.size, [3, 2, 2]))

    def test_remove_unused(self):
        # an entirely-zero shape is dropped from the list
        dead = MorphData(name='dead',
                               indices=np.array([0, 1]),
                               offsets=np.array([[0., 0., 0.],
                                                 [0., 0., 0.]]))
        obj = MorphList([self.eyesClosed.copy(), dead, self.jawOpen.copy()])
        obj.remove_unused()
        self.assertTrue([shape.name for shape in obj] == ['eyesClosed', 'jawOpen'])

        # tolerance drops shapes whose offsets are all below threshold
        tiny = MorphData(name='tiny',
                               indices=np.array([0]),
                               offsets=np.array([[0.0005, 0.0, 0.0]]))
        obj = MorphList([self.eyesClosed.copy(), tiny])
        obj.remove_unused(tolerance=0.001)
        self.assertTrue([shape.name for shape in obj] == ['eyesClosed'])

        # prune_offsets then remove_unused drops shapes emptied by pruning
        obj = MorphList([self.jawOpen.copy(), dead.copy()])
        obj.prune_offsets()
        self.assertTrue(np.allclose(obj.size, [2, 0]))
        obj.remove_unused()
        self.assertTrue([shape.name for shape in obj] == ['jawOpen'])





class TestMorphData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.box = MeshData(points=np.array([[-0.5, -0.5,  0.5],
                                             [ 0.5, -0.5,  0.5],
                                             [-0.5,  0.5,  0.5],
                                             [ 0.5,  0.5,  0.5],
                                             [-0.5,  0.5, -0.5],
                                             [ 0.5,  0.5, -0.5],
                                             [-0.5, -0.5, -0.5],
                                             [ 0.5, -0.5, -0.5]]),
                            indices=np.array([0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]),
                            counts=np.array([4, 4, 4, 4, 4, 4]))

        self.shape = MorphData(name='target',
                                     indices=np.array([5, 3, 4]),
                                     offsets=np.array([[ 0.0,  0.0,  0.0],
                                                       [ 9.5,  1.5,  2.5],
                                                       [-0.5,  1.5, -2.5]]))

        self.shape2 = MorphData(name='target2',
                                      indices=np.array([1, 6, 4]),
                                      offsets=np.array([[ 0.0,  0.0,  0.0],
                                                        [ 9.5,  1.5,  2.5],
                                                        [-0.5,  1.5, -2.5]]))
        
        
    def test_default_indices(self):
        shape = MorphData(name='target',
                                offsets=np.array([[ 0.0,  0.0,  0.0],
                                                  [ 9.5,  1.5,  2.5],
                                                  [-0.5,  1.5, -2.5]]))
        self.assertTrue(np.allclose(shape.indices, [0, 1, 2]))

    def test_fields(self):
        """offsets are float64 (N, 3), indices int32, whatever was given"""
        shape = MorphData(name="target", offsets=[[1, 2, 3], [4, 5, 6]], indices=[7, 9])
        self.assertEqual(shape.offsets.dtype, np.float64)
        self.assertEqual(shape.indices.dtype, np.int32)
        self.assertEqual(MorphData(name="target", offsets=[[1, 2, 3]]).indices.dtype, np.int32)

        with self.assertRaises(ValueError):
            MorphData(name="target", offsets=np.zeros((2, 2)))

    def test_no_indices_saves(self):
        """indices set to None save, and load back as every point"""
        shape         = self.shape.copy()
        shape.indices = None

        loaded = MorphData.from_dict(shape.to_dict())
        self.assertTrue(np.array_equal(loaded.indices, [0, 1, 2]))

        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "target.npz")
            shape.save(f)
            self.assertTrue(np.array_equal(MorphData.load(f).indices, [0, 1, 2]))

    def test_from_dict_without_indices(self):
        """a tree without indices reads them as every point, as __init__ does"""
        shape = MorphData.from_dict({"name": "target", "offsets": [[0, 0, 1], [1, 0, 0]]})
        self.assertEqual(shape.indices.dtype, np.int32)
        self.assertTrue(np.array_equal(shape.indices, [0, 1]))

    @unittest.skipIf(Usd is None, "USD is not installed")
    def test_prim_round_trip_without_usd_skel_imported(self):
        """to_prim / from_prim import pxr.UsdSkel themselves"""
        script = textwrap.dedent(
            """
            import sys
            from pxr import Usd
            from cgmath.geometry.morph_target import MorphData

            assert "pxr.UsdSkel" not in sys.modules
            stage = Usd.Stage.CreateInMemory()
            prim = stage.DefinePrim("/target")
            MorphData(name="target", offsets=[[1, 2, 3]], indices=[4]).to_prim(prim)

            loaded = MorphData.from_prim(prim)
            assert loaded.indices.tolist() == [4], loaded.indices
            assert loaded.offsets.tolist() == [[1, 2, 3]], loaded.offsets
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECTS, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bytes(self):
        obj1 = self.shape.copy()
        b    = obj1.to_bytes()
        obj2 = MorphData.from_bytes(b)
        self.assertTrue(obj1 == obj2)

    def test_serialize(self):

        # private data the class does not declare is refused, so none can
        # reach a file
        obj1 = self.shape.copy()
        with self.assertRaisesRegex(AttributeError, "_STATE"):
            obj1._custom_data = "test"

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # save obj1 to file
            obj1.save(f)

            # load obj2 from file
            obj2 = MorphData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # do the same via pickle, put obj1 inside a list
            with open(f, "wb") as io:
                pickle.dump([obj1], io)

            with open(f, "rb") as io:
                obj3 = pickle.load(io)[0]

                # confirm they're identical
                self.assertTrue(obj1 == obj3)

            # test json and npz
            obj1.save(f, mode='json')
            obj4 = MorphData.load(f)
            self.assertTrue(obj1 == obj4)

            obj1.save(f, mode='npz')
            obj5 = MorphData.load(f)
            self.assertTrue(obj1 == obj5)



    def test_properties(self):
        # prune unused for this test
        shape = self.shape.copy()
        shape.prune_offsets()
        self.assertTrue(shape.size == len(shape.offsets))
        self.assertTrue(shape.min == shape.magnitudes[1])
        self.assertTrue(shape.max == shape.magnitudes[0])
        self.assertTrue(shape.mse == 17.916666666666668)

    def test_unit(self):
        unit = np.linalg.norm(self.shape.unit, axis=1)
        self.assertTrue(np.allclose(unit, [0.0, 1.0, 1.0]))

    def test_sort(self):
        shape = self.shape.copy()
        shape.sort()
        unit = np.linalg.norm(shape.unit, axis=1)
        self.assertTrue(np.allclose(unit, [1.0, 1.0, 0.0]))

    def test_add(self):
        shape  = self.shape.copy()
        shape2 = self.shape2.copy()
        shape3 = shape + shape2
        expected_offsets = np.array([[ 0. ,  0. ,  0. ],
                                     [ 9.5,  1.5,  2.5],
                                     [-1. ,  3. , -5. ],
                                     [ 0. ,  0. ,  0. ],
                                     [ 9.5,  1.5,  2.5]])
        expected_indices = np.array([1, 3, 4, 5, 6])

        self.assertTrue(shape3.name == "target_target2")
        self.assertTrue(np.allclose(shape3.offsets, expected_offsets))
        self.assertTrue(np.allclose(shape3.indices, expected_indices))

        # try sum!
        shape3 = sum([shape, shape2])
        self.assertTrue(shape3.name == "target_target2")
        self.assertTrue(np.allclose(shape3.offsets, expected_offsets))
        self.assertTrue(np.allclose(shape3.indices, expected_indices))


    def test_mult(self):
        shape = self.shape.copy()
        mag   = shape.magnitudes
        shape *= 0.5
        self.assertTrue(np.allclose(shape.magnitudes, mag * 0.5))

    def test_div(self):
        shape = self.shape.copy()
        mag   = shape.magnitudes
        shape /= 2
        self.assertTrue(np.allclose(shape.magnitudes, mag / 2))

    def test_pow_has_no_meaning(self):
        # ** is gone, on either side and in place
        shape = self.shape.copy()
        for label, op in (("shape ** 2", lambda x: x ** 2), ("2 ** shape", lambda x: 2 ** x)):
            with self.subTest(op=label):
                with self.assertRaisesRegex(TypeError, r"unsupported operand type\(s\) for \*\* or pow\(\)"):
                    op(shape)
        with self.assertRaises(TypeError):
            shape **= 2
        self.assertTrue(shape == self.shape)

    def test_what_has_no_meaning_raises(self):
        # a number added or subtracted no longer shifts the indices; a number
        # less or over a morph, a morph times or over a morph: TypeError
        # naming what has a meaning, the morph left as it was
        sums   = "has no meaning: a MorphData adds or subtracts another MorphData"
        scales = "has no meaning: a MorphData multiplies or divides by a number"
        cases = (
            ("shape + 2", lambda x: x + 2, r"^MorphData \+ int " + sums),
            ("shape - 2", lambda x: x - 2, r"^MorphData - int " + sums),
            ("shape + 2.0", lambda x: x + 2.0, r"^MorphData \+ float " + sums),
            ("2 + shape", lambda x: 2 + x, r"^int \+ MorphData " + sums),
            ("5 - shape", lambda x: 5 - x, r"^int - MorphData " + sums),
            ("shape + shapes", lambda x: x + MorphList([self.shape2]), r"^MorphData \+ MorphList " + sums),
            ("shape * shape", lambda x: x * self.shape2, r"^MorphData \* MorphData " + scales),
            ("shape / shape", lambda x: x / self.shape2, r"^MorphData / MorphData " + scales),
            ("2 / shape", lambda x: 2 / x, r"^int / MorphData " + scales),
            ("2.0 / shape", lambda x: 2.0 / x, r"^float / MorphData " + scales),
            ("skin * shape", lambda x: SKIN * x, r"^SkinData \* MorphData " + scales),
        )
        for label, op, refused in cases:
            with self.subTest(op=label):
                shape = self.shape.copy()
                with self.assertRaisesRegex(TypeError, refused):
                    op(shape)
                self.assertTrue(shape == self.shape)

        inplace = (
            ("+= 2", lambda x: x.__iadd__(2)),
            ("-= 2", lambda x: x.__isub__(2)),
            ("*= shape", lambda x: x.__imul__(self.shape2)),
            ("/= shape", lambda x: x.__itruediv__(self.shape2)),
        )
        for label, op in inplace:
            with self.subTest(op=label):
                shape = self.shape.copy()
                with self.assertRaises(TypeError):
                    op(shape)
                self.assertTrue(shape == self.shape)

        # the refusal's hint is an expression that works: 0 - a is a * -1
        with self.assertRaisesRegex(TypeError, r"0 - a is a \* -1, so sum\(\) works"):
            self.shape + 2
        self.assertTrue(0 - self.shape == self.shape * -1)

        # numbers scale it, on either side
        np.testing.assert_array_equal((2 * self.shape).offsets, self.shape.offsets * 2)
        np.testing.assert_array_equal((self.shape * 2).offsets, self.shape.offsets * 2)
        np.testing.assert_array_equal((self.shape / 2).offsets, self.shape.offsets / 2)

    def test_zero_gives_a_new_morph(self):
        # sum() starts at 0: 0 + a morph is a copy, never the morph itself, as
        # every other sum is a new morph; 0 - a morph is it negated
        for obj in (0 + self.shape, sum([self.shape])):
            self.assertTrue(obj == self.shape)
            self.assertIsNot(obj, self.shape)
            self.assertIsNot(obj.offsets, self.shape.offsets)
        obj = 0 - self.shape
        np.testing.assert_array_equal(obj.offsets, -self.shape.offsets)
        np.testing.assert_array_equal(obj.indices, self.shape.indices)

    def test_sub(self):
        """Test subtraction operations for MorphData"""
        shape  = self.shape.copy()
        shape2 = self.shape2.copy()
        
        # Test in-place subtraction with another MorphData
        shape3 = shape.copy()
        shape3 -= shape2
        
        # After subtraction, indices should be the union of both sets
        # shape: indices [5, 3, 4] with offsets [[0, 0, 0], [9.5, 1.5, 2.5], [-0.5, 1.5, -2.5]]
        # shape2: indices [1, 6, 4] with offsets [[0, 0, 0], [9.5, 1.5, 2.5], [-0.5, 1.5, -2.5]]
        # Result should have indices [1, 3, 4, 5, 6]
        expected_indices = np.array([1, 3, 4, 5, 6])
        self.assertTrue(np.allclose(shape3.indices, expected_indices))
        
        # Verify subtraction logic
        # Index 1: 0 - 0 = 0
        # Index 3: 9.5,1.5,2.5 - 0 = 9.5,1.5,2.5  
        # Index 4: -0.5,1.5,-2.5 - (-0.5,1.5,-2.5) = 0,0,0
        # Index 5: 0 - 0 = 0
        # Index 6: 0 - 9.5,1.5,2.5 = -9.5,-1.5,-2.5
        self.assertTrue(shape3.indices.shape[0] == 5)
        
        # Test subtraction operator
        shape4 = shape - shape2
        self.assertTrue(np.allclose(shape4.indices, expected_indices))
        
        # Test right-hand subtraction with 0
        shape6               = 0 - self.shape
        expected_offsets_neg = self.shape.offsets * -1
        self.assertTrue(np.allclose(shape6.offsets, expected_offsets_neg))

    def test_is_zero(self):
        """Test is_zero method"""
        # Test with non-zero offsets
        self.assertFalse(self.shape.is_zero())
        
        # Test with all zero offsets
        zero_shape = MorphData(name='zero',
                                     indices=np.array([0, 1, 2]),
                                     offsets=np.array([[0., 0., 0.],
                                                      [0., 0., 0.],
                                                      [0., 0., 0.]]))
        self.assertTrue(zero_shape.is_zero())
        
        # Test with empty offsets
        empty_shape = MorphData(name='empty',
                                      indices=np.array([]),
                                      offsets=np.array([]).reshape(0, 3))
        self.assertTrue(empty_shape.is_zero())
        
        # Test with tolerance
        tiny_shape = MorphData(name='tiny',
                                    indices=np.array([0, 1]),
                                    offsets=np.array([[0.0001, 0.0001, 0.0001],
                                                     [0.0002, 0.0002, 0.0002]]))
        self.assertFalse(tiny_shape.is_zero(tolerance=0.0))
        self.assertTrue(tiny_shape.is_zero(tolerance=0.001))

    def test_from_mesh_data(self):
        """Test from_mesh_data class method"""
        # Create base and target meshes
        base_mesh   = self.box.copy()
        target_mesh = self.box.copy()
        
        # Modify some vertices in target mesh
        target_mesh.points[0] += [0.1, 0.2, 0.3]
        target_mesh.points[3] += [0.5, -0.1, 0.2]
        target_mesh.points[5] += [-0.2, 0.3, -0.1]
        
        # Create morph target
        morph = MorphData.from_mesh_data(
            base_mesh, target_mesh, target_name='test_morph'
        )
        
        # Verify the morph target has correct data
        self.assertEqual(morph.name, 'test_morph')
        self.assertEqual(morph.size, 3)
        self.assertTrue(0 in morph.indices)
        self.assertTrue(3 in morph.indices)
        self.assertTrue(5 in morph.indices)
        
        # Verify offsets are correct
        self.assertTrue(np.allclose(morph.offsets[np.where(morph.indices == 0)[0][0]], 
                                   [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(morph.offsets[np.where(morph.indices == 3)[0][0]], 
                                   [0.5, -0.1, 0.2]))
        self.assertTrue(np.allclose(morph.offsets[np.where(morph.indices == 5)[0][0]], 
                                   [-0.2, 0.3, -0.1]))
        
        # Test with tolerance
        base_mesh2   = self.box.copy()
        target_mesh2 = self.box.copy()
        target_mesh2.points[0] += [0.0001, 0.0001, 0.0001]  # Very small change
        
        morph_notol = MorphData.from_mesh_data(
            base_mesh2, target_mesh2, target_name='notol', tolerance=0.0
        )
        self.assertGreater(morph_notol.size, 0)
        
        morph_tol = MorphData.from_mesh_data(
            base_mesh2, target_mesh2, target_name='tol', tolerance=0.001
        )
        self.assertEqual(morph_tol.size, 0)
        
        # Test error case - mismatched vertex counts
        base_mesh3 = self.box.copy()
        target_mesh3 = MeshData(
            points  = np.array([[0., 0., 0.], [1., 0., 0.]]),
            indices = np.array([0, 1]),
            counts  = np.array([2])
        )
        
        with self.assertRaises(RuntimeError):
            MorphData.from_mesh_data(base_mesh3, target_mesh3)