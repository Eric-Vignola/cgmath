from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.geometry._base import _is_sequence, Data, DataList
from cgmath.geometry.utils import pxr

BLS_PRIM_TYPE = "MorphTarget"


@dataclass(repr=False, eq=False)
class MorphData(Data):
    """
    Morph target data class.
    """

    EQUALITY_TEST_IGNORE = ["name"]

    name:    str
    offsets: np.ndarray
    indices: Optional[np.ndarray] = None

    def __init__(
        self, name: str, offsets: np.ndarray, indices: Optional[np.ndarray] = None
    ):
        self.name    = name
        self.offsets = np.asarray(offsets)

        # assume whole mesh indices if none given
        if indices is None:
            self.indices = np.arange(offsets.shape[0])
        else:
            self.indices = np.asarray(indices)

    def to_dict(self) -> dict:
        """returns the annotated data as a dict"""
        data = super().to_dict()
        if "indices" not in data:
            data.indices = np.arange(data.offsets.shape[0])
        return data

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

    def __iadd__(self, other):
        """inplace add of two morph targets"""

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

        # assume arithmetic offset of the indices
        else:
            self.indices += int(other)

        return self

    def __add__(self, other):
        """addition of two morph targets"""
        new = self.copy()
        return new.__iadd__(other)

    def __radd__(self, other):
        """
        right-hand side addition of two morph targets
        allows to use builtin sum()
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other):
        """inplace subtraction of two morph targets"""

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

        # assume arithmetic offset of the indices
        else:
            self.indices -= int(other)

        return self

    def __sub__(self, other):
        """subtraction of two morph targets"""
        new = self.copy()
        return new.__isub__(other)

    def __rsub__(self, other):
        """
        right-hand side subtraction of two morph targets
        """
        if other == 0:
            return self * -1
        return self.__sub__(other)

    def __imul__(self, other):
        """inplace multiplication of two morph targets"""

        if isinstance(other, MorphData):
            if self.size == 0:
                return self  # nothing to multiply
            else:
                merged_indices = np.concatenate([self.indices, other.indices])
                merged_indices = np.unique(merged_indices)

                offsets               = np.zeros((merged_indices.max() + 1, 3))
                offsets[self.indices] = self.offsets
                offsets[other.indices] *= other.offsets

                self.indices = merged_indices
                self.offsets = offsets[merged_indices]

        # assume arithmetic multiplication
        else:
            self.offsets *= other

        return self

    def __mul__(self, other):
        """multiplication of two morph targets"""
        new = self.copy()
        return new.__imul__(other)

    def __rmul__(self, other):
        """
        right-hand side multiplication of two morph targets
        """
        return self.__mul__(other)

    def __itruediv__(self, other):
        """inplace division of two morph targets"""

        # assume arithmetic division
        self.offsets /= other

        return self

    def __truediv__(self, other):
        """multiplication of two morph targets"""
        new = self.copy()
        return new.__itruediv__(other)

    def __rtruediv__(self, other):
        """
        right-hand side multiplication of two morph targets
        """
        return self.__truediv__(other)

    def __ipow__(self, other):
        """inplace exponent of two morph targets"""

        # assume arithmetic exponent
        abs_power    = np.abs(self.offsets) ** other
        self.offsets = np.sign(self.offsets) * abs_power
        return self

    def __pow__(self, other):
        """exponent of two morph targets"""
        new = self.copy()
        return new.__ipow__(other)

    def __rpow__(self, other):
        """
        right-hand side exponent of two morph targets
        """
        return self.__pow__(other)

    @classmethod
    def from_mesh_data(
        cls,
        base_mesh_data,
        target_mesh_data,
        target_name:     str          = "",
        tolerance:       float | None = None,
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
        indices = np.array(bls_api.GetPointIndicesAttr().Get(), dtype=int)
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


class MorphList(DataList):
    DATA_LIST_CLASS = MorphData

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

    def __iadd__(self, other):
        """inplace add of morph targets"""

        if isinstance(other, MorphList) or _is_sequence(other):
            if len(self.list) != len(other.list):
                raise RuntimeError("Lists must have the same length.")

            for i in range(len(self.list)):
                self.list[i] += other.list[i]

        # assume arithmetic addition or addition of MorphData
        else:
            for shape in self.list:
                shape += other

        return self

    def __add__(self, other):
        new = self.copy()
        return new.__iadd__(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other):
        """subtraction add of morph targets"""

        if isinstance(other, MorphList) or _is_sequence(other):
            if len(self.list) != len(other.list):
                raise RuntimeError("Lists must have the same length.")

            for i in range(len(self.list)):
                self.list[i] -= other.list[i]

        # assume arithmetic subtraction or subtraction of MorphData
        else:
            for shape in self.list:
                shape -= other

        return self

    def __sub__(self, other):
        new = self.copy()
        return new.__isub__(other)

    def __rsub__(self, other):
        if other == 0:
            return self * -1
        return self.__sub__(other)

    def __imul__(self, other):
        """inplace multiplication of morph targets"""

        if isinstance(other, MorphList) or _is_sequence(other):
            if len(self.list) != len(other.list):
                raise RuntimeError("Lists must have the same length.")

            for i in range(len(self.list)):
                self.list[i] *= other.list[i]

        # assume arithmetic multiplication or multiplication of MorphData
        else:
            for shape in self.list:
                shape *= other

        return self

    def __mul__(self, other):
        new = self.copy()
        return new.__imul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        """inplace division morph targets"""

        # assume arithmetic division
        for shape in self.list:
            shape /= other

        return self

    def __truediv__(self, other):
        """multiplication of two morph targets"""
        new = self.copy()
        return new.__itruediv__(other)

    def __rtruediv__(self, other):
        """
        right-hand side multiplication of morph targets
        """
        return self.__truediv__(other)

    def __ipow__(self, other):
        """inplace exponent of morph targets"""

        # assume arithmetic exponent
        for shape in self.list:
            shape **= other

        return self

    def __pow__(self, other):
        """exponent of two morph targets"""
        new = self.copy()
        return new.__ipow__(other)

    def __rpow__(self, other):
        """
        right-hand side exponent of two morph targets
        """
        return self.__pow__(other)