"""
Optimized version of skin_weights.py with Numba-accelerated implementations.

This module provides the same API as skin_weights.py but uses parallel Numba kernels
for improved performance on large skin weight datasets. It also includes benchmarking
utilities to compare performance between original and optimized implementations.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from cgmath.geometry._base import Data, DataList
from cgmath.geometry.utils import balance_center_weights, blur, inpaint, pxr
from scipy.sparse import coo_matrix

# autodesk fbx sdk
try:
    import fbx
except ImportError:
    fbx = None

JOINT_NAMES_ATTR = "skel:jointNames"
SKIN_PRIM_TYPE   = "SkinWeights"

EPSILON = np.finfo(np.float32).eps


# ---------------------- Patterns (unchanged) -------------------------------- #


class Patterns:
    """predefined positive/negative name search patterns"""

    SHORT        = {"positive": ["*_l_*"], "negative": ["*_r_*"]}
    SHORT_PREFIX = {"positive": ["l_*"], "negative": ["r_*"]}
    SHORT_SUFFIX = {"positive": ["*_l"], "negative": ["*_r"]}

    TRINITY3 = {"positive": ["b_l_*", "p_l_*"], "negative": ["b_r_*", "p_r_*"]}
    TRINITY4 = SHORT_PREFIX

    STYLE2   = {"positive": ["*_left_*"], "negative": ["*_right_*"]}
    TRINITY  = TRINITY4
    DEFAULT  = TRINITY


# ---------------------- Optimized _round_normalize -------------------------- #


def _round_normalize(weights: np.ndarray, n: int) -> np.ndarray:
    """Optimized: uses parallel Numba kernel."""
    from cgmath.geometry.utils._numba._skin_weights import round_normalize_fast

    return round_normalize_fast(weights, n)


# ---------------------- CompactSkinData ------------------------------------- #


@dataclass(repr=False, eq=False)
class CompactSkinData(Data):
    """Compact skin data class with optimized to_skin_data."""

    max_influences:    int
    influence_indices: np.ndarray
    weights:           np.ndarray
    influences:        List[str]

    @property
    def valid(self) -> bool:
        """checks whether the skin data is valid"""
        weights = self.weights.reshape(-1, self.max_influences)
        return np.all(np.isclose(weights.sum(axis=1), 1.0))

    def round(self, n: int):
        """round weights to n digits"""
        weights      = self.weights.reshape(-1, self.max_influences)
        self.weights = _round_normalize(weights, n).ravel()

    def to_skin_data(self) -> "SkinData":
        """Optimized: uses parallel Numba scatter operation."""
        from cgmath.geometry.utils._numba._skin_weights import scatter_to_dense_fast

        vert_count = self.weights.reshape(-1, self.max_influences).shape[0]
        weights = scatter_to_dense_fast(
            self.influence_indices,
            self.weights,
            vert_count,
            len(self.influences),
            self.max_influences,
        )

        return SkinData(
            weights    = weights,
            influences = list(self.influences),
        )

    def __reduce__(self):
        """use a sparse matrix for smaller file size (pickle)"""
        reconstructor = CompactSkinData._reconstruct
        state = (
            self.max_influences,
            self.influence_indices,
            self.weights,
            [str(x) for x in self.influences],
        )
        return reconstructor, (state,)

    @staticmethod
    def _reconstruct(state):
        """rebuilds a dense matrix from sparse data"""
        max_inf, inf_ids, weights, influences = state
        return CompactSkinData(
            max_influences    = max_inf,
            influence_indices = inf_ids,
            weights           = weights,
            influences        = influences,
        )

    @classmethod
    def from_prim(cls, prim: Any) -> "CompactSkinData":
        """Constructs a data object from a prim."""
        binding_api = pxr().UsdSkel.BindingAPI(prim)
        infs        = prim.GetAttribute(JOINT_NAMES_ATTR).Get()
        inf_ids     = binding_api.GetJointIndicesAttr().Get()
        weights     = binding_api.GetJointWeightsAttr().Get()
        max_infs    = binding_api.GetJointIndicesPrimvar().GetElementSize()

        weights          = np.array(weights, dtype=np.float64).reshape(-1, max_infs)
        total            = weights.sum(axis=1)
        half_precision   = np.all(np.isclose(total, 1.0, atol=1e-7, rtol=1e-7))
        double_precision = np.all(np.isclose(total, 1.0, atol=1e-15, rtol=1e-15))
        if half_precision and not double_precision:
            weights = _round_normalize(weights, 7)

        return cls(
            influences        = infs,
            max_influences    = max_infs,
            influence_indices = np.array(inf_ids),
            weights           = weights.ravel(),
        )

    def to_prim(self, prim: Any) -> None:
        """Streams data into a prim."""
        if not prim.GetTypeName():
            prim.SetTypeName(SKIN_PRIM_TYPE)
        binding_api = pxr().UsdSkel.BindingAPI(prim)
        binding_api.CreateJointIndicesAttr(self.influence_indices)
        binding_api.GetJointIndicesPrimvar().SetElementSize(self.max_influences)
        binding_api.GetJointIndicesPrimvar().SetInterpolation("vertex")
        binding_api.CreateJointWeightsAttr(self.weights)
        binding_api.GetJointWeightsPrimvar().SetElementSize(self.max_influences)
        binding_api.GetJointWeightsPrimvar().SetInterpolation("vertex")

        Sdf = pxr().Sdf
        prim.CreateAttribute(
            JOINT_NAMES_ATTR,
            Sdf.ValueTypeNames.TokenArray,
            False,
            Sdf.VariabilityVarying,
        ).Set([x.rsplit("|", 1)[-1].rsplit(":", 1)[-1] for x in self.influences])


# ---------------------- SkinData -------------------------------------------- #


@dataclass(repr=False, eq=False)
class SkinData(Data):
    """Skin data class with optimized operations."""

    EQUALITY_TEST_IGNORE = ["name"]

    weights:    np.ndarray
    influences: List[str]
    name:       Optional[str] = None

    _patterns = Patterns.DEFAULT

    @property
    def patterns(self) -> Patterns:
        return self._patterns

    @patterns.setter
    def patterns(self, pattern: Patterns) -> None:
        self._patterns = pattern

    @property
    def positive_patterns(self) -> List[str]:
        return self._patterns["positive"]

    @property
    def negative_patterns(self) -> List[str]:
        return self._patterns["negative"]

    def __contains__(self, item):
        return item in self.influences

    def is_equivalent(self, other: "SkinData") -> bool:
        """checks whether the skin data is equivalent to another"""
        try:
            bigger  = self
            smaller = other
            if len(other.influences) > len(self.influences):
                bigger  = other
                smaller = self

            idx = np.where(np.isin(bigger.influences, smaller.influences))[0]
            return all(
                [
                    all(x in bigger.influences for x in smaller.influences),
                    np.allclose(bigger.weights[:, idx], smaller.weights),
                ]
            )
        except ValueError:
            return False

    @property
    def valid(self) -> bool:
        """checks whether the skin data is valid"""
        return all(
            [
                np.all(np.isclose(self.weights.sum(axis=1), 1.0)),
                self.weights.shape[1] == len(self.influences),
            ]
        )

    @property
    def size(self) -> int:
        """returns the number of vertices"""
        return self.weights.shape[0]

    @property
    def counts(self) -> np.ndarray:
        """Optimized: uses parallel Numba kernel."""
        from cgmath.geometry.utils._numba._skin_weights import count_nonzero_fast

        return count_nonzero_fast(self.weights, EPSILON)

    def prune(
        self, min_value: float = 0.0, normalize: bool = True, optimize: bool = False
    ):
        """Optimized: uses parallel Numba kernel."""
        from cgmath.geometry.utils._numba._skin_weights import prune_fast

        self.weights = prune_fast(self.weights, min_value, normalize=normalize)

        if optimize:
            self.remove_unused()

    def normalize(self, locked_influences=None, indices=None):
        """Optimized: uses parallel Numba kernel when no locked influences."""
        from cgmath.geometry.utils._numba._skin_weights import normalize_fast

        # Ensure weights are float64
        if self.weights.dtype != np.float64:
            self.weights = self.weights.astype(np.float64)

        normalize_fast(self.weights, locked_influences, indices)

    def remove_unused(self, tolerance: float = 0.0, normalize: bool = True):
        """removes unused influences (max > tolerance)"""
        kept_indices    = np.where(np.max(self.weights, axis=0) > tolerance)[0]
        self.weights    = self.weights[:, kept_indices]
        self.influences = [self.influences[x] for x in kept_indices]

        if normalize:
            self.normalize()

    def set_max_influences(
        self, count: int, optimize: bool = False, sorting_method: str = "introselect"
    ):
        """Optimized: uses parallel Numba kernel."""
        from cgmath.geometry.utils._numba._skin_weights import set_max_influences_fast

        if count < 0:
            return

        if count > 0 and self.counts.max() > count:
            self.weights = set_max_influences_fast(self.weights, count, normalize=True)
        else:
            self.normalize()

        if optimize:
            self.remove_unused()

    def get_max_influences(self) -> int:
        """returns the maximum number of influences on vertices"""
        return int(self.counts.max())

    def get_indices_over_max(self, count: int) -> np.ndarray:
        """returns the indices of vertices with more than count influences"""
        return np.where(self.counts > count)[0]

    def round(self, n: int):
        """round weights to n digits"""
        self.weights = _round_normalize(self.weights, n)

    def sort(self):
        """sorts influences alphabetically"""
        sorted_influences = np.argsort(self.influences)
        self.weights      = self.weights[:, sorted_influences]
        self.influences   = np.array(self.influences)[sorted_influences].tolist()

    def rank(self, influences: List[str]) -> np.ndarray:
        """takes a list of influences and returns their ranked ids"""
        source_influences = np.array(self.influences)
        sorted_influences = np.argsort(source_influences)
        influences        = np.array(influences)
        ranked            = np.searchsorted(source_influences[sorted_influences], influences)
        ids               = sorted_influences[ranked]
        return ids

    def __iadd__(self, other):
        """concatenates two skin data objects inplace"""
        if not all([self.valid, other.valid]):
            raise ValueError("Invalid SkinData objects cannot be added.")
        self.conform(self, other)
        self.weights = np.concatenate([self.weights, other.weights])

    def __add__(self, other):
        """concatenates two skin data objects"""
        new = self.copy()
        new.__iadd__(other)
        return new

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def index(self, influence: str):
        """returns the index of an influence name"""
        return [str(x) for x in self.influences].index(str(influence))

    def remove(self, influence: str, optimize: bool = False):
        """removes an influence from the skin data"""
        idx          = self.index(influence)
        self.weights = np.delete(self.weights, idx, axis=1)
        self.normalize()
        self.influences.pop(idx)
        if optimize:
            self.remove_unused()

    def append(self, influence: str, weight: float = 0.0):
        """appends an influence to the end of the list"""
        if influence in self.influences:
            raise ValueError(f"Influence [{influence}] already in the list")

        self.influences.append(influence)
        new                             = np.ones((self.weights.shape[0], self.weights.shape[1] + 1)) * weight
        new[:, : self.weights.shape[1]] = self.weights
        self.weights                    = new

        if weight != 0:
            self.normalize(locked_influences=[-1])

    def extend(self, influences: List[str], weights: Optional[List[float]] = None):
        """appends multiple influences to the end of the list"""
        for influence in influences:
            if influence in self.influences:
                raise ValueError(f"Influence [{influence}] already in the list")

        if weights is None:
            weights = np.zeros(len(influences))
        else:
            weights = np.asarray(weights)

        if len(influences) != weights.size:
            raise ValueError("Number of influences must match number of weights")

        self.influences.extend(influences)
        new                             = np.ones((self.weights.shape[0], self.weights.shape[1] + len(influences)))
        new[:, -weights.size :]         = weights[None, :]
        new[:, : self.weights.shape[1]] = self.weights
        self.weights                    = new

        if np.any(weights != 0):
            idx = -1 - np.arange(weights.size)
            self.normalize(locked_influences=idx)

    def insert(self, index: int, influence: str, weight: float = 0.0):
        """inserts an influence at the specified index"""
        if influence not in self.influences:
            self.influences.insert(index, influence)
            new_column   = np.full((1, self.weights.shape[0]), weight)
            self.weights = np.insert(self.weights, index, new_column, axis=1)
            if weight != 0:
                self.normalize(locked_influences=[index])
        else:
            raise ValueError(f"Influence [{influence}] already in the list")

    def get_influences(
        self, queries: str | list[str] | tuple[str] | None = None, exact: bool = True
    ) -> list[str]:
        """uses fnmatch to find any influences that match the queries"""
        if queries is None:
            return list(self.influences)
        elif isinstance(queries, str):
            queries = [queries]

        matched = []
        for item in queries:
            for infl in self.influences:
                if exact:
                    success = fnmatch.fnmatchcase(infl, item)
                else:
                    success = fnmatch.fnmatch(infl.lower(), item.lower())
                if success and infl not in matched:
                    matched.append(infl)
        return matched

    def transfer_influences(
        self, src_influences: list, dst_influences: list, weighted: bool = True
    ) -> None:
        """transfers weights from source influence(s) to destination influence(s)"""
        if not isinstance(src_influences, (list, tuple)):
            src_influences = [src_influences]
        if not isinstance(dst_influences, (list, tuple)):
            dst_influences = [dst_influences]

        src_influences = sorted(set(src_influences))
        dst_influences = sorted(set(dst_influences))

        src_indices = np.array([self.index(x) for x in src_influences if x in self])
        if src_indices.size == 0:
            return

        [self.append(x) for x in dst_influences if x not in self]
        dst_indices = np.array([self.index(x) for x in dst_influences])

        weights = np.sum(self.weights[:, src_indices], axis=1)

        if not weighted or dst_indices.size == 1:
            weights /= dst_indices.size
            self.weights[:, src_indices] = 0.0
            self.weights[:, dst_indices] += weights[:, None]
        else:
            ratios   = np.ones((self.weights.shape[0], dst_indices.size)) * 0.5
            total    = np.sum(self.weights[:, dst_indices[:, None]], axis=1)
            non_zero = np.where(total > 0)[0]
            ratios[non_zero] = (
                self.weights[non_zero, dst_indices[:, None]].T / total[non_zero]
            )
            self.weights[:, src_indices] = 0.0
            self.weights[:, dst_indices] += ratios * weights[:, None]

    def __getitem__(self, obj):
        """supports slicing"""
        return SkinData(
            weights    = np.copy(self.weights[obj]),
            influences = list(self.influences),
        )

    def to_compact_skin_data(self) -> CompactSkinData:
        """Optimized: uses parallel Numba top-k gathering."""
        from cgmath.geometry.utils._numba._skin_weights import gather_top_k_fast

        max_inf = self.get_max_influences()
        k       = min(max_inf, self.weights.shape[1])

        indices, weights = gather_top_k_fast(self.weights, k)

        return CompactSkinData(
            max_influences    = max_inf,
            influence_indices = indices.ravel(),
            weights           = weights.ravel(),
            influences        = self.influences,
        )

    # ----------------------------------- GLB ------------------------------------ #

    @classmethod
    def load_glb(cls, filename: str, name: str | None = None) -> "SkinData":
        """returns a single SkinData from a glb (first skinned primitive if name is None)"""
        data = [x for x in load_glb(filename) if x is not None]
        if not data:
            raise RuntimeError("No skinned primitives found in glb file")
        if name is None:
            return data[0]
        for skin_data in data:
            if skin_data.name == name:
                return skin_data
        available = [x.name for x in data]
        raise ValueError(f"Skin '{name}' not found; available: {available}")

    # ----------------------------------- FBX ------------------------------------ #

    @classmethod
    def load_fbx(cls, filename: str, name: str | None = None) -> "SkinData":
        """returns a single SkinData from an fbx file (first skinned mesh if name is None)"""
        data = load_fbx(filename)
        if not data:
            raise RuntimeError("No skinned meshes found in FBX file")
        if name is None:
            return data[0][1]
        for mesh_name, skin_data in data:
            if mesh_name == name:
                return skin_data
        available = [x for x, _ in data]
        raise ValueError(f"Skin '{name}' not found; available: {available}")

    @staticmethod
    def conform(*objects: "SkinData") -> "SkinData":
        """conforms SkinData objects so their influence lists are identical"""
        all_influences = list(objects[0].influences)
        for obj in objects[1:]:
            for infl in obj.influences:
                if infl not in all_influences:
                    all_influences.append(infl)

        all_influences    = np.array(all_influences)
        sorted_influences = np.argsort(all_influences)

        for obj in objects:
            influences = np.array(obj.influences)
            ranked     = np.searchsorted(all_influences[sorted_influences], influences)
            ids        = sorted_influences[ranked]

            weights         = np.zeros((obj.weights.shape[0], all_influences.shape[0]))
            weights[:, ids] = obj.weights

            obj.influences  = all_influences.tolist()
            obj.weights     = weights

    # --------------------------------- SYMMETRY --------------------------------- #

    def get_symmetry_map(
        self,
        positive_patterns: Tuple[str] | None = None,
        negative_patterns: Tuple[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """computes the symmetric influence indices according to given patterns"""

        def rename(input_string, from_pattern, to_pattern):
            regex_pattern = from_pattern.replace("*", "(.*)")
            match = re.match(regex_pattern, input_string)
            if match:
                groups = match.groups()
                result = to_pattern
                for _, group in enumerate(groups, start=1):
                    result = result.replace("*", group, 1)
                return result
            else:
                return input_string

        if positive_patterns is None:
            positive_patterns = self.positive_patterns
        if negative_patterns is None:
            negative_patterns = self.negative_patterns

        matched        = np.ones(len(self.influences), dtype=int) * -1
        positive_match = []
        negative_match = []
        none_match     = []

        if not isinstance(positive_patterns, (list, tuple)):
            positive_patterns = [positive_patterns]
        if not isinstance(negative_patterns, (list, tuple)):
            negative_patterns = [negative_patterns]

        for i, infl in enumerate(self.influences):
            for j in range(len(positive_patterns)):
                if matched[i] == -1:
                    infl_      = rename(infl, positive_patterns[j], negative_patterns[j])
                    match_from = positive_match
                    match_to   = negative_match
                    if infl == infl_:
                        infl_      = rename(infl, negative_patterns[j], positive_patterns[j])
                        match_from = negative_match
                        match_to   = positive_match

                    if infl_ != infl:
                        for infl in self.influences[i:]:
                            if infl == infl_:
                                matched[i]          = self.influences.index(infl)
                                matched[matched[i]] = i
                                match_from.append(i)
                                match_to.append(matched[i])

            if matched[i] == -1:
                matched[i] = i
                none_match.append(i)

        if (
            len(match_from) > 0
            and len(match_to) > 0
            and len(match_from) == len(match_to)
        ):
            return (
                matched,
                np.array(match_from),
                np.array(match_to),
                np.array(none_match),
            )

        return None

    def get_asymmetric_weights(
        self,
        mesh_data,
        positive_patterns: Tuple[str]         | None = None,
        negative_patterns: Tuple[str]         | None = None,
        pivot:             float                     = 0.0,
        axis:              int                       = 0,
        tolerance:         Union[None, float]        = None,
    ) -> bool:
        """Optimized: uses parallel Numba kernel for symmetry comparisons."""
        from cgmath.geometry.utils._numba._skin_weights import (
            compare_symmetric_weights_fast,
        )

        if positive_patterns is None:
            positive_patterns = self.positive_patterns
        if negative_patterns is None:
            negative_patterns = self.negative_patterns

        data = self.get_symmetry_map(positive_patterns, negative_patterns)
        if data is None:
            return None

        positive_influences = data[1]
        negative_influences = data[2]

        if tolerance is None:
            tolerance = mesh_data.get_symmetry_deviation(pivot=pivot, axis=axis)[-1]

        center_indices = mesh_data.get_center_points(
            pivot=pivot, axis=axis, tolerance=tolerance
        )
        _, sym_indices = mesh_data.get_symmetry_map(pivot=pivot, axis=axis)

        a             = self.weights[:, negative_influences]
        b             = self.weights[sym_indices[:, None], positive_influences]
        positive_test = np.where(compare_symmetric_weights_fast(a, b, 1e-7))[0]
        positive_test = np.setdiff1d(positive_test, center_indices)

        a             = self.weights[:, positive_influences]
        b             = self.weights[sym_indices[:, None], negative_influences]
        negative_test = np.where(compare_symmetric_weights_fast(a, b, 1e-7))[0]
        negative_test = np.setdiff1d(negative_test, center_indices)

        a = self.weights[center_indices[:, None], negative_influences]
        b = self.weights[center_indices[:, None], positive_influences]
        center_test = center_indices[
            np.where(compare_symmetric_weights_fast(a, b, 1e-7))
        ]

        return np.unique(np.concatenate([positive_test, negative_test, center_test]))

    def symmetrical(
        self,
        mesh_data,
        positive_patterns: Tuple[str]         | None = None,
        negative_patterns: Tuple[str]         | None = None,
        pivot:             float                     = 0.0,
        axis:              int                       = 0,
        tolerance:         Union[None, float]        = None,
    ) -> bool:
        """Optimized: uses parallel Numba kernel for symmetry comparisons."""
        from cgmath.geometry.utils._numba._skin_weights import (
            compare_symmetric_weights_fast,
        )

        if positive_patterns is None:
            positive_patterns = self.positive_patterns
        if negative_patterns is None:
            negative_patterns = self.negative_patterns

        data = self.get_symmetry_map(positive_patterns, negative_patterns)
        if data is None:
            return True

        mirror_influences   = data[0]
        positive_influences = data[1]
        negative_influences = data[2]
        mirror_weights      = self.weights.T[mirror_influences].T

        if tolerance is None:
            tolerance = mesh_data.get_symmetry_deviation(pivot=pivot, axis=axis)[-1]

        center_indices = mesh_data.get_center_points(
            pivot=pivot, axis=axis, tolerance=tolerance
        )
        _, sym_indices = mesh_data.get_symmetry_map(pivot=pivot, axis=axis)
        match_indices = (mesh_data.points[:, axis].ravel() - pivot) < -tolerance
        match_indices = np.where(match_indices)[0]

        a     = self.weights[center_indices, negative_influences[:, None]]
        b     = self.weights[center_indices, positive_influences[:, None]]
        test1 = not np.any(compare_symmetric_weights_fast(a, b, 1e-7))

        a     = self.weights[match_indices]
        b     = mirror_weights[sym_indices[match_indices]]
        test2 = not np.any(compare_symmetric_weights_fast(a, b, 1e-7))

        return test1 and test2

    def fix_symmetry(
        self,
        mesh_data,
        positive_patterns: Tuple[str]         | None = None,
        negative_patterns: Tuple[str]         | None = None,
        pivot:             float                     = 0.0,
        axis:              int                       = 0,
        side:              float                     = 1.0,
        tolerance:         Union[None, float]        = None,
        force:             bool                      = False,
        max_influences:    int                       = -1,
    ) -> bool:
        """Optimized: uses parallel Numba kernel for weight mirroring."""
        from cgmath.geometry.utils._numba._skin_weights import mirror_weights_fast

        if positive_patterns is None:
            positive_patterns = self.positive_patterns
        if negative_patterns is None:
            negative_patterns = self.negative_patterns

        data = self.get_symmetry_map(positive_patterns, negative_patterns)
        if data is None:
            return True

        mirror_influences   = data[0]
        positive_influences = data[1]
        negative_influences = data[2]
        center_influences   = data[3]

        if tolerance is None:
            tolerance = mesh_data.get_symmetry_deviation(pivot=pivot, axis=axis)[-1]

        center_indices = mesh_data.get_center_points(
            pivot=pivot, axis=axis, tolerance=tolerance
        )
        _, sym_indices = mesh_data.get_symmetry_map(pivot=pivot, axis=axis)

        if side >= 0.0:
            match_indices = np.where(
                (mesh_data.points[:, axis].ravel() - pivot) < -tolerance
            )[0]
        else:
            match_indices = np.where(
                (mesh_data.points[:, axis].ravel() - pivot) > tolerance
            )[0]

        duplicate = self.copy()
        duplicate.weights = mirror_weights_fast(
            self.weights,
            mirror_influences,
            sym_indices[match_indices],
            sym_indices,
        )

        if center_indices.size > 0 and center_influences.size > 0:
            non_center = np.unique(
                np.concatenate([positive_influences, negative_influences])
            )

            if side >= 0.0:
                duplicate.weights[center_indices[:, None], positive_influences] = (
                    duplicate.weights[center_indices[:, None], negative_influences]
                )
            else:
                duplicate.weights[center_indices[:, None], negative_influences] = (
                    duplicate.weights[center_indices[:, None], positive_influences]
                )

            duplicate.normalize(
                locked_influences=center_influences, indices=center_indices
            )
            duplicate.normalize(locked_influences=non_center, indices=center_indices)

            if max_influences > 1:
                indices = duplicate.get_indices_over_max(max_influences)
                indices = indices[np.isin(indices, center_indices)]

                if indices.size > 0:
                    weights = duplicate.weights[indices]
                    balance_center_weights(
                        weights,
                        positive_influences,
                        negative_influences,
                        center_influences,
                        max_influences,
                    )
                    duplicate.weights[indices] = weights
                    duplicate.normalize()

        if max_influences > 0:
            duplicate.set_max_influences(max_influences)

        if force or duplicate.symmetrical(
            mesh_data=mesh_data,
            positive_patterns=positive_patterns,
            negative_patterns=negative_patterns,
            pivot=pivot,
            axis=axis,
            tolerance=EPSILON,
        ):
            self.weights = duplicate.weights
            return True

        return False

    # -------------------------------- INPAINTING -------------------------------- #

    def inpaint(
        self,
        neighbors:     np.ndarray,
        indices:       np.ndarray,
        iterations:    int        | np.ndarray = 1,
        receptions:    float      | np.ndarray = 0.5,
        contributions: float      | np.ndarray = 1.0,
        tolerance:     float                   = 1e-7,
    ):
        if indices is not None:
            indices = np.unique(indices).astype(int)
            if (
                indices.size == self.weights.shape[0]
                and indices[-1] == self.weights.shape[0] - 1
            ):
                return

        if isinstance(iterations, int):
            iterations = np.ones(self.weights.shape[0], dtype=int) * iterations
        else:
            iterations = np.array(iterations)

        mask             = np.ones(self.weights.shape[0], dtype=bool)
        mask[indices]    = False
        iterations[mask] = 0

        self.weights = inpaint(
            indices,
            self.weights,
            neighbors,
            iterations    = iterations,
            receptions    = receptions,
            contributions = contributions,
        )

    # ---------------------------------- SMOOTH ---------------------------------- #

    def smooth(
        self,
        neighbors:     np.ndarray,
        indices:       np.ndarray | None       = None,
        iterations:    int        | np.ndarray = 1,
        receptions:    float      | np.ndarray = 0.5,
        contributions: float      | np.ndarray = 1.0,
    ):
        if indices is not None:
            if isinstance(iterations, int):
                iterations = np.ones(neighbors.shape[0], dtype=int) * iterations
            else:
                iterations = np.array(iterations)

            mask             = np.ones(neighbors.shape[0], dtype=bool)
            mask[indices]    = False
            iterations[mask] = 0

        self.weights = blur(
            self.weights,
            neighbors,
            iterations    = iterations,
            receptions    = receptions,
            contributions = contributions,
        )

    # ------------------------ Catmull-Clark Subdivision ------------------------- #

    def subdivide(
        self,
        mesh_data,
        steps:        int  = 1,
        keep_borders: bool = False,
        keep_edges:   bool = False,
        keep_size:    bool = False,
    ) -> None:
        """uses the Catmull-Clark algorithm to subdivide the skin weights"""
        proxy        = mesh_data.copy()
        proxy.points = self.weights
        proxy.subdivide(steps=steps, keep_borders=keep_borders, keep_edges=keep_edges)

        if keep_size:
            self.weights = proxy.points[: self.weights.shape[0]]
        else:
            self.weights = proxy.points

    # ------------------ serialization ------------------ #

    def __reduce__(self):
        """use a sparse matrix for smaller file size (pickle)"""
        reconstructor = SkinData._reconstruct
        sparse_matrix = coo_matrix(self.weights)
        state = (
            [str(x) for x in self.influences],
            sparse_matrix.data,
            sparse_matrix.row,
            sparse_matrix.col,
            sparse_matrix.shape,
        )
        return reconstructor, (state,)

    @staticmethod
    def _reconstruct(state):
        """rebuilds a dense matrix from sparse data"""
        influences, data, row, col, shape = state
        sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
        return SkinData(influences=influences, weights=sparse_matrix.toarray())


# ---------------------- SkinList -------------------------------------------- #


class SkinList(DataList):
    DATA_LIST_CLASS = SkinData

    def transfer_influences(
        self, src_influences: list, dst_influences: list, weighted: bool = True
    ) -> None:
        """transfers weights from one influence to another"""
        for obj in self:
            obj.transfer_influences(src_influences, dst_influences, weighted=weighted)


# ---------------------- file readers ---------------------------------------- #


def _dense_weights(vertex_count: int, columns: list) -> Tuple[np.ndarray, List[str]]:
    """scatters ``(name, vertex indices, weights)`` columns into a dense matrix

    A joint reached through more than one cluster collapses onto a single
    column rather than appearing twice: a repeated influence name makes the
    joint mapping ambiguous for anything that binds against it.
    """
    influences = []
    lookup     = {}
    for name, _, _ in columns:
        if name not in lookup:
            lookup[name] = len(influences)
            influences.append(name)

    weights = np.zeros((vertex_count, len(influences)), dtype=np.float64)
    for name, indices, values in columns:
        if len(indices):
            np.add.at(weights, (indices, lookup[name]), values)

    return weights, influences


def _fbx_matrix(matrix) -> np.ndarray:
    """an ``FbxAMatrix`` as a row-vector 4x4

    ``Get(row, column)`` already indexes the row-vector layout cgmath uses --
    the translation lands in row 3 -- so nothing is transposed on the way in.
    """
    return np.array(
        [[matrix.Get(row, col) for col in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def load_fbx(filename: str, bind_matrices: bool = False) -> list:
    """loads an fbx file and returns a list of ``(mesh name, SkinData)`` tuples

    Influences carry the names ``HierarchyData.load_fbx`` gives the same file.
    They are resolved through the ``FbxNode`` behind each cluster rather than
    by name, because fbx lets two nodes share a name and a name lookup could
    not then say which one a cluster meant.

    Meshes carrying no skin deformer are left out, so the result is not
    positionally parallel to ``cgmath.geometry.mesh.load_fbx``.

    With *bind_matrices*, every tuple gains a third item: the ``(J, 4, 4)``
    inverse bind matrices the file authored for that mesh, ordered to the
    ``SkinData``'s influences.  A rig read back out of an animated file
    carries the take's pose rather than the pose the mesh was bound in, so
    these are the only trustworthy source for it.
    """
    if fbx is None:
        raise ImportError("Autodesk FBX Python SDK is not installed")

    # imported here rather than at module scope: transforms reaches
    # back into cgmath.geometry as it loads, so a top level import closes the
    # cycle. cgmath.geometry.mesh is lazy for symmetry, not necessity.
    from cgmath.geometry.mesh import _walk_fbx_mesh_nodes
    from cgmath.hierarchy import _fbx_enum, _read_fbx

    filename = os.path.expanduser(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FBX file not found: {filename}")

    # manager goes unread, but it owns the scene and every node reached below,
    # so the name has to stay bound for the rest of the call
    manager, scene, hierarchy_data, nodes = _read_fbx(
        filename, 1.0
    )

    named = {
        node.GetUniqueID(): hierarchy_data[unique_id]["name"]
        for unique_id, node in nodes.items()
    }

    skin_type  = _fbx_enum(fbx.FbxDeformer, "EDeformerType", "eSkin")

    mesh_nodes = []
    _walk_fbx_mesh_nodes(scene.GetRootNode(), mesh_nodes)

    data = []
    for node in mesh_nodes:
        mesh    = node.GetNodeAttribute()
        columns = []
        binds   = {}

        for index in range(mesh.GetDeformerCount(skin_type)):
            skin = mesh.GetDeformer(index, skin_type)

            for cluster_index in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(cluster_index)
                link    = cluster.GetLink()
                if link is None:
                    continue

                joint = named.get(link.GetUniqueID())
                if joint is None:
                    raise ValueError(
                        f"cluster {cluster.GetName()!r} links {link.GetName()!r}, "
                        f"which the rig read from {filename!r} does not hold"
                    )

                columns.append(
                    (
                        joint,
                        np.asarray(cluster.GetControlPointIndices(), dtype=np.int64),
                        np.asarray(cluster.GetControlPointWeights(), dtype=np.float64),
                    )
                )

                if bind_matrices:
                    link_bind, mesh_bind = fbx.FbxAMatrix(), fbx.FbxAMatrix()
                    cluster.GetTransformLinkMatrix(link_bind)
                    cluster.GetTransformMatrix(mesh_bind)
                    # first cluster wins, the way _dense_weights collapses a
                    # joint reached twice onto its first column
                    binds.setdefault(
                        joint,
                        _fbx_matrix(mesh_bind) @ np.linalg.inv(_fbx_matrix(link_bind)),
                    )

        if not columns:
            continue

        weights, influences = _dense_weights(mesh.GetControlPointsCount(), columns)
        entry = (
            node.GetName(),
            SkinData(weights=weights, influences=influences, name=node.GetName()),
        )
        if bind_matrices:
            entry += (np.asarray([binds[x] for x in influences]),)
        data.append(entry)

    return data


def load_glb(filename: str, bind_matrices: bool = False) -> list:
    """loads a glb and returns one entry per primitive, ``None`` where unskinned

    The list runs parallel to ``cgmath.formats.glb.load_model``'s primitives,
    not to ``MeshData.load_glb``: trimesh drops any primitive that is not built
    from triangles, so those two only line up on an all triangle file.

    Influences carry the names ``HierarchyData.load_glb`` gives the same file.
    A vertex's ``JOINTS_0`` entry indexes the skin's own joint list rather than
    the file's nodes, so it is read through ``skin.joints`` -- taking it for a
    node index binds vertices to whatever happens to sit there, silently.

    With *bind_matrices*, a skinned entry becomes a ``(SkinData, (J, 4, 4))``
    pair carrying the file's own inverse bind matrices, ordered to the
    influences.  They are in the file's units, which is not necessarily the
    unit the points get read in -- see ``scale_factor``.
    """
    from cgmath.formats.glb import load_model
    from cgmath.hierarchy import HierarchyData
    from cgmath.hierarchy import _glb_columns

    filename  = os.path.expanduser(filename)

    model     = load_model(filename)
    rig       = HierarchyData.load_glb(filename)
    rig_names = [str(x) for x in rig.name]
    column    = _glb_columns(model, rig, filename)

    # glb.py hands meshes and skins back as unrelated lists, so the tie between
    # the two has to be read off the nodes that reference both
    skin_of_mesh = {}
    for node in model.nodes:
        if node.mesh is not None and node.skin is not None:
            skin_of_mesh.setdefault(node.mesh, node.skin)

    data = []
    for index, mesh in enumerate(model.meshes):
        skin_index = skin_of_mesh.get(index)

        for primitive in mesh.primitives:
            if (
                skin_index is None
                or primitive.joints is None
                or primitive.weights is None
            ):
                data.append(None)
                continue

            skin = model.skins[skin_index]

            influences = []
            for node_index in skin.joints:
                row = column.get(node_index)
                if row is None:
                    raise ValueError(
                        f"{filename!r} skins to node {node_index}, which the rig "
                        f"read from it does not hold"
                    )
                influences.append(rig_names[row])

            joints = np.asarray(primitive.joints, dtype=np.int32)
            dense = CompactSkinData(
                max_influences    = joints.shape[1],
                influence_indices = joints.ravel(),
                weights           = np.asarray(primitive.weights, dtype=np.float64).ravel(),
                influences        = influences,
            ).to_skin_data()

            entry = SkinData(
                weights    = dense.weights,
                influences = dense.influences,
                name       = primitive.name,
            )
            if bind_matrices:
                # glTF stores each matrix column-major, so reading the 16
                # floats back in C order already gives the row-vector form
                # cgmath wants -- there is no transpose here
                entry = (
                    entry,
                    np.asarray(skin.inverse_bind_matrices, dtype=np.float64).reshape(
                        -1, 4, 4
                    ),
                )
            data.append(entry)

    return data