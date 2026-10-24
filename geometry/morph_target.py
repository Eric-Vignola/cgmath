from __future__ import annotations

import numbers
import operator
from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.data import _is_sequence, Array, Data, DataList
from cgmath.geometry.utils import pxr

BLS_PRIM_TYPE = "MorphTarget"

# the item operation each of a MorphList's operators takes shape by shape
_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def _is_zero(value) -> bool:
    """whether ``value`` is the number 0, which builtin sum() starts from"""
    return isinstance(value, numbers.Number) and value == 0


class _Name(str):
    """an operand's name in a refusal, where its class would mislead"""


def _owner(array: np.ndarray) -> Optional[np.ndarray]:
    """the array that owns ``array``'s memory, found through the ``.base``
    of its views: itself, or the one they view. None where that walk ends
    on an array that owns no memory, its ``.base`` no array: an as_strided
    or sliding_window_view view, an array of a memoryview or of another
    object's buffer, whose memory only its bounds tell"""
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array if array.flags.owndata else None


def _holding(sequence, item) -> _Name:
    """a sequence named by the item a refusal is about: ``a list holding an int``"""
    names = ("array" if isinstance(x, np.ndarray) else type(x).__name__ for x in (sequence, item))
    return _Name(" holding ".join(("an " if x[:1].lower() in "aeiou" else "a ") + x for x in names))


def _refused(left, op: str, right) -> TypeError:
    """
    the error for ``left op right``, which has no meaning for morph targets,
    saying what has one; ``left`` and ``right`` each an operand, its class
    or a :class:`_Name`
    """
    if op in ("+", "-"):
        supported = (
            "a MorphData adds or subtracts another MorphData (a + b, a - b; 0 + a is a "
            "copy of a and 0 - a is a * -1, so sum() works), and a MorphList adds or "
            "subtracts a MorphData from every shape, or a MorphList or a sequence of "
            "MorphData shape by shape"
        )
    else:
        supported = (
            "a MorphData multiplies or divides by a number (a * 2, 2 * a, a / 2), and "
            "a MorphList by a number, or by a sequence of numbers, one a shape"
        )
    left, right = (
        x if isinstance(x, _Name) else x.__name__ if isinstance(x, type) else type(x).__name__
        for x in (left, right)
    )
    return TypeError(f"{left} {op} {right} has no meaning: {supported}")


@dataclass(repr=False, eq=False)
class MorphData(Data):
    """
    Morph target data class.

    Arithmetic: ``a + b`` and ``a - b`` add or subtract another MorphData,
    merging the two index sets; ``0 + a`` is a copy of ``a`` and ``0 - a``
    is ``a * -1``, so builtin sum() works; ``a * 2``, ``2 * a`` and
    ``a / 2`` scale the offsets, and ``a * [1, 1, 2]`` and
    ``a / [1, 1, 2]`` their x, y and z each by its own number. Nothing else
    has a meaning and raises TypeError: a number added or subtracted (which
    shifted the indices), a morph times or over a morph, a number over or
    less a morph, a power.
    """

    EQUALITY_TEST_IGNORE = ["name"]

    name:    str
    offsets: Array(np.float64, "N", 3)
    indices: Optional[Array(np.int32, "N")] = None

    def __init__(
        self, name: str, offsets: np.ndarray, indices: Optional[np.ndarray] = None
    ):
        self.name    = name
        self.offsets = offsets
        self.indices = indices
        self._post_load()

    def _post_load(self) -> None:
        """no indices means every point, in order"""
        if self.indices is None:
            self.indices = np.arange(self.offsets.shape[0], dtype=np.int32)

    @property
    def size(self) -> int:
        """returns the count of datapoints"""
        return self.indices.size

    @property
    def unit(self) -> np.ndarray:
        """returns offsets as unit vectors"""
        unit = np.array(self.offsets)
        mag  = self.magnitudes
        mask = mag > 0
        unit[mask] /= mag[mask][:, None]
        return unit

    @property
    def magnitudes(self) -> np.ndarray:
        """returns the offset's magnitudes"""
        return np.einsum("...i,...i", self.offsets, self.offsets) ** 0.5

    @property
    def min(self) -> float:
        """returns lowest magnitude of the offsets"""
        return self.magnitudes.min()

    @property
    def max(self) -> float:
        """returns highest magnitude of the offsets"""
        return self.magnitudes.max()

    @property
    def mse(self) -> float:
        """computes the mean square error of the offsets"""
        if self.offsets.shape[0] == 0:
            return 0.0
        return (self.offsets**2).mean()

    def prune_offsets(
        self, tolerance: float | None = None, neighbors: np.ndarray | None = None
    ) -> None:
        """
        prunes offsets with magnidutes <= tolerance

        if neighbors is given, offsets are pruned only if the sum of their neighbors
        present in the passing indices is <= to the neighbor_tolerance.
        """
        if tolerance is None:
            tolerance = 0.0

        # find which offsets exceeds tolerance
        passing = self.magnitudes > tolerance

        # if neighbor data is given, for every vertex that failed the test
        # check if any neighbor exceeded the tolerance. If any are found
        # then do not remove this vertex.
        if neighbors is not None:
            failing           = ~passing
            failing_neighbors = neighbors[self.indices[failing]]
            mag               = np.zeros((neighbors.shape[0],))
            mag[self.indices] = self.magnitudes

            magnitudes                                = mag[failing_neighbors] > tolerance
            magnitudes[failing_neighbors == -1]       = False
            magnitudes                                = np.any(magnitudes, axis=1)
            passing[np.where(failing)[0][magnitudes]] = True

        # set the offsets
        self.offsets = self.offsets[passing]
        self.indices = self.indices[passing]

    def is_zero(self, tolerance: float = 0.0) -> bool:
        """returns True if this morph target has no data or all offsets are zero"""
        if self.offsets.size == 0 or np.allclose(self.magnitudes, 0, atol=tolerance):
            return True
        return False

    def sort(self):
        """sorts indices."""
        sorted_indices = np.argsort(self.indices)
        self.indices   = self.indices[sorted_indices]
        self.offsets   = self.offsets[sorted_indices]

    # --------------------------------- arithmetic --------------------------------- #

    def __iadd__(self, other):
        """
        in place, another MorphData's offsets added: the two index sets
        merged, the offsets of an index both hold summed, the names joined
        on their ``_`` separated parts. A number added has no meaning
        (TypeError): it no longer shifts the indices
        """

        if isinstance(other, MorphData):
            if self.size == 0:
                self.indices = np.copy(other.indices)
                self.offsets = np.copy(other.offsets)
                self.name    = other.name

            else:
                merged_indices = np.concatenate([self.indices, other.indices])
                merged_indices = np.unique(merged_indices)

                offsets               = np.zeros((merged_indices.max() + 1, 3))
                offsets[self.indices] = self.offsets
                offsets[other.indices] += other.offsets

                self.indices = merged_indices
                self.offsets = offsets[merged_indices]

                # concatenate the names
                self_name = self.name.split("_")
                for name in other.name.split("_"):
                    if name not in self_name:
                        self_name.append(name)

                self.name = "_".join(self_name)

        # a number or anything else added has no meaning
        else:
            raise _refused(self, "+", other)

        return self

    def __add__(self, other):
        """a new MorphData, another one's offsets added, as ``+=`` adds them"""
        new = self.copy()
        return new.__iadd__(other)

    def __radd__(self, other):
        """
        ``0 + morph``, a copy of the morph, so builtin sum() works; any other
        number added has no meaning (TypeError)
        """
        if _is_zero(other):
            return self.copy()
        raise _refused(other, "+", self)

    def __isub__(self, other):
        """
        in place, another MorphData's offsets subtracted, the index sets
        merged and the names joined as ``+=`` does. A number subtracted has
        no meaning (TypeError): it no longer shifts the indices
        """

        if isinstance(other, MorphData):
            if self.size == 0:
                self.indices = np.copy(other.indices)
                self.offsets = np.copy(other.offsets) * -1
                self.name    = other.name

            else:
                merged_indices = np.concatenate([self.indices, other.indices])
                merged_indices = np.unique(merged_indices)

                offsets               = np.zeros((merged_indices.max() + 1, 3))
                offsets[self.indices] = self.offsets
                offsets[other.indices] -= other.offsets

                self.indices = merged_indices
                self.offsets = offsets[merged_indices]

                # concatenate the names
                self_name = self.name.split("_")
                for name in other.name.split("_"):
                    if name not in self_name:
                        self_name.append(name)

                self.name = "_".join(self_name)

        # a number or anything else subtracted has no meaning
        else:
            raise _refused(self, "-", other)

        return self

    def __sub__(self, other):
        """a new MorphData, another one's offsets subtracted, as ``-=`` subtracts them"""
        new = self.copy()
        return new.__isub__(other)

    def __rsub__(self, other):
        """
        ``0 - morph``, the morph negated, a new one; any other number less a
        morph has no meaning (TypeError)
        """
        if _is_zero(other):
            return self * -1
        raise _refused(other, "-", self)

    def __imul__(self, other):
        """
        in place, the offsets times a number (or what numpy multiplies an
        (N, 3) array by). A morph times a morph, a MorphData or a MorphList,
        has no meaning (TypeError)
        """
        if isinstance(other, (Data, DataList)):
            raise _refused(self, "*", other)

        self.offsets *= other
        return self

    def __mul__(self, other):
        """a new MorphData, its offsets times a number, as ``*=`` multiplies them"""
        new = self.copy()
        return new.__imul__(other)

    def __rmul__(self, other):
        """``2 * morph``, the same as ``morph * 2``"""
        if isinstance(other, (Data, DataList)):
            raise _refused(other, "*", self)
        return self.__mul__(other)

    def __itruediv__(self, other):
        """
        in place, the offsets divided by a number (or what numpy divides an
        (N, 3) array by). A morph over a morph has no meaning (TypeError)
        """
        if isinstance(other, (Data, DataList)):
            raise _refused(self, "/", other)

        self.offsets /= other
        return self

    def __truediv__(self, other):
        """a new MorphData, its offsets divided by a number, as ``/=`` divides them"""
        new = self.copy()
        return new.__itruediv__(other)

    def __rtruediv__(self, other):
        """``2 / morph`` has no meaning (TypeError): a morph divides by a number"""
        raise _refused(other, "/", self)

    @classmethod
    def from_mesh_data(
        cls,
        base_mesh_data,
        target_mesh_data,
        target_name: str          = "",
        tolerance:   float | None = None,
    ) -> MorphData:
        """Creates a MorphData object from two mesh data objects.

        Args:
            base_mesh_data: The base mesh data object.
            target_mesh_data: The target mesh data object.
            target_name: Name of this morph target.
            tolerance: Tolerance for comparing points.

        Returns:
            A MorphData object.
        """
        if len(base_mesh_data.points) != len(target_mesh_data.points):
            raise RuntimeError("Base and target meshes must have same vertex count.")
        tolerance = 0.0 if tolerance is None else tolerance
        indices = np.where(
            ~np.isclose(base_mesh_data.points, target_mesh_data.points, atol=tolerance)
        )
        indices = np.unique(indices[0])
        offsets = target_mesh_data.points[indices] - base_mesh_data.points[indices]
        return cls(indices=indices, offsets=offsets, name=target_name)

    # --- USD interface

    @classmethod
    def from_prim(cls, prim) -> MorphData:
        """Constructs a data object from a prim."""
        bls_api = pxr().UsdSkel.BlendShape(prim)
        offsets = bls_api.GetOffsetsAttr().Get()

        # ensure the correct data type
        indices = np.array(bls_api.GetPointIndicesAttr().Get(), dtype=np.int32)
        # ensure the correct shape even if no offsets are present
        offsets = np.array(offsets) if offsets else np.empty(dtype=float, shape=(0, 3))

        return cls(indices=indices, offsets=offsets, name=prim.GetName())

    def to_prim(self, prim) -> None:
        """Streams data into a prim."""
        if not prim.GetTypeName():
            prim.SetTypeName(BLS_PRIM_TYPE)
        bls_api = pxr().UsdSkel.BlendShape(prim)
        bls_api.CreateOffsetsAttr().Set(self.offsets)
        bls_api.CreatePointIndicesAttr().Set(self.indices)


class MorphList(DataList[MorphData]):
    """
    A list of MorphData, its arithmetic shape by shape.

    ``shapes + a`` and ``shapes - a`` add or subtract a MorphData from every
    shape; a MorphList, or a list or a tuple of MorphData, adds or
    subtracts shape by shape, a list or a tuple on the left too.
    ``shapes * 2`` and ``shapes / 2`` scale every shape, and a sequence of
    numbers, one a shape, each its own, so ``(shapes * s) / s == shapes``.
    ``0 + shapes`` and ``0 - shapes`` are a copy and every shape negated, so
    builtin sum() works. Each of these gives a new list, a new shape in
    each place, so a shape the list holds twice is acted on once in each;
    in place, ``+=``, ``-=``, ``*=`` and ``/=`` change the shapes the list
    holds, so one it holds twice is changed twice, each shape by its
    operand as it was before any shape changed: an operand that is, or
    holds, one of the list's own shapes, or an array of a shape's offsets,
    is taken as it was, so ``shapes -= shapes[0]`` subtracts the first
    shape from every shape, the first included. What has no meaning
    raises TypeError, whatever the two lengths, an empty list's too, before
    a sequence of another length raises RuntimeError and before any shape
    is changed. An operand numpy refuses, None or an array of another
    shape, raises numpy's error at the shape it reaches, in place the
    shapes before that one already changed.
    """

    # --------- vectorized properties and methods --------- #
    @property
    def size(self) -> np.ndarray:
        """returns the datapoint count for each shape"""
        return np.array([shape.size for shape in self.list])

    @property
    def min(self) -> np.ndarray:
        """returns lowest magnitude for each shape"""
        return np.array([shape.min for shape in self.list])

    @property
    def max(self) -> np.ndarray:
        """returns highest magnitude for each shape"""
        return np.array([shape.max for shape in self.list])

    @property
    def mse(self) -> np.ndarray:
        """computes the mean square error for each shape"""
        return np.array([shape.mse for shape in self.list])

    def prune_offsets(self, *args, **kwargs) -> None:
        """prunes offsets with magidutes <= tolerance for each shape"""
        for shape in self.list:
            shape.prune_offsets(*args, **kwargs)

    def remove_unused(self, tolerance: float = 0.0) -> None:
        """removes shapes that are unused (empty or all offsets zero within tolerance)"""
        self.list = [shape for shape in self.list if not shape.is_zero(tolerance)]

    # -------------------------- shape math operations --------------------------- #

    def _per_shape(self, other, op: str) -> list:
        """
        the operand of each shape for ``self op other``: a MorphList's or a
        sequence's items, one a shape, else ``other`` for every shape. What
        has no meaning raises TypeError first, whatever the two lengths: for
        ``+`` and ``-`` anything but a MorphData, for ``*`` and ``/`` a
        morph. Then a sequence of another length raises RuntimeError. The
        refusal names the operand, a python list, tuple or array by the item
        refused: ``MorphList + a list holding a MorphList``
        """
        many  = isinstance(other, MorphList) or _is_sequence(other)
        items = list(other) if many else [other]
        plain = many and not isinstance(other, DataList)
        if op in ("+", "-"):
            refused = [x for x in items if not isinstance(x, MorphData)]
        else:
            refused = [x for x in [other] + items if isinstance(x, (Data, DataList))]
        if refused:
            raise _refused(self, op, _holding(other, refused[0]) if plain else other)

        if not many:
            return items * len(self.list)
        if len(self.list) != len(items):
            raise RuntimeError("Lists must have the same length.")
        return items

    def _in_place_operands(self, other, op: str) -> list:
        """
        :meth:`_per_shape`'s operands for ``self op= other``, each as it is
        before any shape changes. An operand that an earlier place changes
        is copied first, once: for ``+`` and ``-`` one of the list's own
        shapes, for ``*`` and ``/`` an array that shares memory with a
        shape's offsets: the offsets, a view of them, an as_strided or a
        memoryview's array of them too. Two arrays share it where one array
        owns the memory of both (:func:`_owner`), or, where either has no
        owner, where their bounds meet (``np.may_share_memory``). So
        ``shapes -= shapes[0]`` subtracts the first shape as it was from
        every shape, the first included, and ``shapes += shapes[::-1]`` adds
        to each the shape it pairs with as it was; ``shapes += shapes``
        copies only a shape it holds twice, once, for its later place,
        each other shape its own operand, and an operand from outside
        the list is never copied
        """
        others = self._per_shape(other, op)
        morphs = op in ("+", "-")

        # numbers alone, no array among them: nothing a place changes
        if not morphs and not any(isinstance(x, np.ndarray) for x in others):
            return others

        # the first place of each shape, for * and / of its offsets' owner;
        # offsets with none are kept with their place, compared by bounds
        first = {}
        loose = []
        for place, shape in enumerate(self.list):
            if morphs:
                first.setdefault(id(shape), place)
                continue
            owner = _owner(shape.offsets)
            if owner is None:
                loose.append((place, shape.offsets))
            else:
                first.setdefault(id(owner), place)

        copies = {}
        for place, x in enumerate(others):
            if morphs:
                changed = first.get(id(x), place) < place
            elif isinstance(x, np.ndarray):
                owner = _owner(x)
                if owner is None:
                    changed = any(np.may_share_memory(x, shape.offsets) for shape in self.list[:place])
                else:
                    changed = first.get(id(owner), place) < place or any(
                        at < place and np.may_share_memory(x, offsets) for at, offsets in loose
                    )
            else:
                continue
            if changed:
                if id(x) not in copies:
                    copies[id(x)] = x.copy() if morphs else np.array(x)
                others[place] = copies[id(x)]
        return others

    def _new_shapes(self, other, op: str) -> MorphList:
        """
        ``self op other``, a new list: each shape op the operand
        :meth:`_per_shape` pairs with it, as the item operation gives it, a
        new shape in each place, so a shape the list holds twice is acted on
        once in each
        """
        return type(self)([_OPERATORS[op](x, y) for x, y in zip(self.list, self._per_shape(other, op))])

    def _sequence_first(self, other, op: str) -> MorphList:
        """
        ``other op self`` with ``other`` on the left a python list or tuple,
        or a DataList that is no MorphList: each of its items op the shape
        it pairs with, the item first, as the item operation does it, each
        giving a new shape. What that refuses raises TypeError first,
        whatever the two lengths: for ``+`` and ``-`` anything but a
        MorphData or 0, for ``*`` a morph. Then a sequence of another length
        raises RuntimeError. The refusal names a list or a tuple by the item
        refused, ``a list holding an int - MorphList``, a DataList by its
        class, ``SkinList + MorphList``
        """
        items = list(other)
        if op in ("+", "-"):
            refused = [x for x in items if not (isinstance(x, MorphData) or _is_zero(x))]
        else:
            refused = [x for x in items if isinstance(x, (Data, DataList))]
        if refused:
            raise _refused(other if isinstance(other, DataList) else _holding(other, refused[0]), op, self)

        if len(self.list) != len(items):
            raise RuntimeError("Lists must have the same length.")
        return type(self)([_OPERATORS[op](y, x) for x, y in zip(self.list, items)])

    def __iadd__(self, other):
        """
        in place, shape by shape: a MorphData added to every shape, or a
        MorphList's or a sequence's MorphData to the shape each pairs with,
        as ``MorphData +=`` adds one; a shape the list holds twice is
        changed in each place, so twice. Each operand is added as it was
        before any shape changed, one of the list's own shapes too
        (:meth:`_in_place_operands`). A number has no meaning (TypeError),
        and nothing is changed
        """
        others = self._in_place_operands(other, "+")
        for i in range(len(self.list)):
            self.list[i] += others[i]

        return self

    def __add__(self, other):
        """a new MorphList, as ``+=`` adds, a new shape in each place"""
        return self._new_shapes(other, "+")

    def __radd__(self, other):
        """
        ``0 + shapes``, a copy, a new shape in each place, so builtin sum()
        works; a list or a tuple of MorphData on the left, each item plus
        the shape it pairs with. A number has no meaning (TypeError)
        """
        if _is_zero(other):
            return type(self)([shape.copy() for shape in self.list])
        if _is_sequence(other):
            return self._sequence_first(other, "+")
        raise _refused(other, "+", self)

    def __isub__(self, other):
        """
        in place, shape by shape: a MorphData subtracted from every shape, or
        a MorphList's or a sequence's MorphData from the shape each pairs
        with, as ``MorphData -=`` subtracts one; a shape the list holds
        twice is changed in each place, so twice. Each operand is
        subtracted as it was before any shape changed, one of the list's
        own shapes too (:meth:`_in_place_operands`). A number has no
        meaning (TypeError), and nothing is changed
        """
        others = self._in_place_operands(other, "-")
        for i in range(len(self.list)):
            self.list[i] -= others[i]

        return self

    def __sub__(self, other):
        """a new MorphList, as ``-=`` subtracts, a new shape in each place"""
        return self._new_shapes(other, "-")

    def __rsub__(self, other):
        """
        ``0 - shapes``, every shape negated, a new list; a list or a tuple of
        MorphData on the left, each item less the shape it pairs with. Any
        other number has no meaning (TypeError)
        """
        if _is_zero(other):
            return self * -1
        if _is_sequence(other):
            return self._sequence_first(other, "-")
        raise _refused(other, "-", self)

    def __imul__(self, other):
        """
        in place, shape by shape: every shape times a number, or each times
        the number a sequence pairs with it; a shape the list holds twice is
        changed in each place, so twice. An array of a shape's offsets is
        taken as it was before any shape changed (:meth:`_in_place_operands`).
        A morph times a morph, a MorphData, or a MorphList shape by shape,
        has no meaning (TypeError), and nothing is changed
        """
        others = self._in_place_operands(other, "*")
        for i in range(len(self.list)):
            self.list[i] *= others[i]

        return self

    def __mul__(self, other):
        """a new MorphList, as ``*=`` multiplies, a new shape in each place"""
        return self._new_shapes(other, "*")

    def __rmul__(self, other):
        """``2 * shapes`` or ``[2, 3] * shapes``, each item times the shape it pairs with"""
        if isinstance(other, (Data, DataList)):
            raise _refused(other, "*", self)
        if _is_sequence(other):
            return self._sequence_first(other, "*")
        return self.__mul__(other)

    def __itruediv__(self, other):
        """
        in place, shape by shape: every shape divided by a number, or each
        by the number a sequence pairs with it, as ``*=`` multiplies, so
        ``(shapes * s) / s == shapes``; a shape the list holds twice is
        changed in each place, so twice. An array of a shape's offsets is
        taken as it was before any shape changed (:meth:`_in_place_operands`).
        A morph over a morph has no meaning (TypeError), and nothing is
        changed
        """
        others = self._in_place_operands(other, "/")
        for i in range(len(self.list)):
            self.list[i] /= others[i]

        return self

    def __truediv__(self, other):
        """a new MorphList, as ``/=`` divides, a new shape in each place"""
        return self._new_shapes(other, "/")

    def __rtruediv__(self, other):
        """``2 / shapes`` has no meaning (TypeError): a shape divides by a number"""
        raise _refused(other, "/", self)
