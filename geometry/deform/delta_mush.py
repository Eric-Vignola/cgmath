"""
Delta Mush deformation operator.

Public class:
    :class:`DeltaMushData` -- bind/apply deformation operator that removes
        low-frequency wobble while preserving high-frequency detail.

The class supports three frame-construction methods (selected via the
``method`` property):

* ``"TBN"`` (default) -- first-face cross-product TBN frame.  Empirically
  the closest match to Maya's ``deltaMush`` across the meshes we've
  benchmarked, while remaining a single-pass-per-vertex operation.
* ``"PROCRUSTES"`` -- per-vertex SVD/polar-decomposition optimal rigid
  rotation between smoothed-rest and smoothed-deformed neighbour
  offsets.  Mathematically the best least-squares rotation but it does
  not always match Maya's deterministic frame choice.
* ``"DDM"`` -- Direct Delta Mush flavour: precomputes weighted
  smoothed-rest neighbour offsets at bind, then runs the same
  Procrustes-style polar decomposition with edge-length weights at
  apply.  More robust on irregular topology and cheaper per apply call
  thanks to the precomputation (Le & Lewis 2019).

The numba-accelerated kernels backing this class live in
:mod:`cgmath.geometry.utils._numba._delta_mush`, with public Python
wrappers exposed via :mod:`cgmath.geometry.utils.main`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.geometry._base import Array, Data


# Allowed frame-construction methods.  Stored uppercase for easy
# case-insensitive comparison.
_METHODS = ("TBN", "PROCRUSTES", "DDM")


# ============================ DeltaMushData ============================== #


@dataclass(repr=False, eq=False)
class DeltaMushData(Data):
    """Delta Mush deformation operator.

    Removes low-frequency deformation noise (twists, jitters, candy-wrap)
    while preserving high-frequency surface detail.  Operates on a bound
    rest mesh and is then applied to any deformed pose of the same point
    cloud.

    The constructor reads what it needs from the rest mesh: its points,
    its edge neighbours, its border vertices (when pinning) and each
    vertex's first-face winding (for ``"TBN"``).  All of it is saved, so a
    loaded operator deforms without the mesh.

    The bind step caches:

    * a Laplacian-smoothed copy of the rest points (``smooth_iterations``
      passes of :func:`cgmath.geometry.utils.blur`, which is what
      :meth:`MeshData.smooth` runs), with no passes on the border
      vertices when pinning so they stay where they are
    * for each vertex, a representation of the rest displacement
      ``rest - smooth_rest``.  When :attr:`method` is ``"TBN"`` the
      displacement is projected into the vertex's first-face tangent
      frame; for ``"PROCRUSTES"`` and ``"DDM"`` the world-space delta is
      stored unchanged and the rotation is recovered at apply time.

    During :meth:`apply` the deformed positions are smoothed with the
    same schedule, the chosen per-vertex rotation is recovered from the
    smoothed-deformed neighbours, and the cached deltas are projected
    back to world space.  The final output is an optional linear blend
    between the raw deformed positions and the reconstructed
    delta-mushed positions, controlled by :attr:`weight`.

    The class also accepts a bare point cloud -- see :meth:`bind` --
    in which case the caller must supply a neighbours matrix
    (typically built from any compatible mesh / kNN graph).  Note that
    the ``"TBN"`` method requires *mesh* input because it uses the
    first-face winding to choose the (next, prev) reference vertices;
    point-cloud inputs automatically fall back to ``"DDM"``.

    Example::

        from cgmath.geometry import DeltaMushData, MeshData

        rest = MeshData.load_obj("body.obj")

        mush = DeltaMushData(rest, smooth_iterations=20, pin_borders=True)

        deformed = rest.copy()
        # ... apply skinning / blendshapes / animation to deformed ...

        result = mush.apply(deformed)  # auto-binds on first call
        # result is a MeshData copy with cleaned-up points

    Switching the frame-construction method::

        mush.method = "PROCRUSTES"  # invalidates bind, will rebind on apply
        result = mush.apply(deformed)
    """

    rest_points:     Optional[Array(np.float64, "N", 3)] = None
    neighbors:       Optional[Array(np.int32, "N", "K")] = None
    border_vertices: Optional[Array(np.int32, "B")] = None

    # --- settings, behind the properties that validate them --- #
    _smooth_iterations: int = 10
    _smooth_step_size:  float = 0.5
    _pin_borders:       bool = True
    _weight:            float = 1.0
    _method:            str = "TBN"

    # (N, 2) [next, prev] from each vertex's first face: the one part of the
    # TBN bind that needs the mesh, so it is read at construction and saved.
    # None for a point cloud, which binds as DDM.
    _first_nbrs: Optional[Array(np.int32, "N", 2)] = None

    # --- cached --- #
    _smooth_points = None
    _local_deltas  = None  # TBN-encoded local deltas (TBN method)
    _world_deltas  = None  # rest - smooth_rest (PROCRUSTES/DDM methods)
    _ddm_offsets   = None  # (N, K, 3) precomputed smoothed-rest offsets
    _ddm_weights   = None  # (N, K) precomputed neighbour weights
    _points        = None

    # --------------------------- construction -------------------------------- #

    def __init__(
        self,
        rest,
        neighbors:         np.ndarray | None = None,
        smooth_iterations: int               = 10,
        smooth_step_size:  float             = 0.5,
        pin_borders:       bool              = True,
        weight:            float             = 1.0,
        method:            str               = "TBN",
    ):
        """Create a Delta Mush operator.

        Args:
            rest: Rest pose.  Accepts a :class:`MeshData` (preferred -- its
                topology supplies edge-vertex neighbours, border
                vertices, *and* the per-vertex first-face winding needed
                by the ``"TBN"`` method) or a raw ``(N, 3)`` point cloud
                (in which case *neighbors* must be supplied and the
                method falls back to ``"DDM"`` on bind).
            neighbors: Optional ``(N, K)`` int32 neighbour matrix with
                ``-1`` padding.  Required when *rest* is a point cloud.
                For mesh inputs, defaults to
                :meth:`MeshData.get_edge_vertex_neighbors`.
            smooth_iterations: Laplacian smoothing iteration count.
                Higher values flatten lower-frequency wobble.
            smooth_step_size: Per-iteration reception factor passed to
                :meth:`MeshData.smooth`.  Smaller is gentler.
            pin_borders: When True (default) and *rest* is a
                :class:`MeshData`, the smoothing iteration count is
                forced to ``0`` on border vertices so they stay
                exactly where they started.
            weight: Final blend weight in ``[0, 1]``.  ``0`` returns the
                smoothed mesh, ``1`` (default) returns the fully
                reconstructed mesh.  Can be changed any time without
                rebinding.
            method: Frame-construction method -- one of ``"TBN"``
                (default), ``"PROCRUSTES"`` or ``"DDM"``.  Case-
                insensitive.  Changing the method invalidates the bind
                cache.
        """
        # local import to avoid circular dependency with cgmath.geometry.mesh
        from cgmath.geometry.mesh import MeshData

        if isinstance(rest, MeshData):
            self.rest_points = np.asarray(rest.points, dtype=np.float64).copy()
            self._first_nbrs = self._compute_first_face_neighbors(rest)
            if neighbors is None:
                neighbors = rest.get_edge_vertex_neighbors()
            if pin_borders:
                self.border_vertices = np.asarray(
                    rest.get_border_vertices(flatten=True), dtype=np.int32
                ).ravel()
            else:
                self.border_vertices = None
        else:
            pts              = rest.points if hasattr(rest, "points") else rest
            self.rest_points = np.asarray(pts, dtype=np.float64).copy()
            self._first_nbrs = None
            if neighbors is None:
                raise ValueError(
                    "neighbors must be supplied when rest is not a MeshData"
                )
            self.border_vertices = None

        self.neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        if self.neighbors.shape[0] != self.rest_points.shape[0]:
            raise ValueError(
                f"neighbors row count ({self.neighbors.shape[0]}) must match "
                f"rest_points count ({self.rest_points.shape[0]})"
            )

        self._smooth_iterations = int(smooth_iterations)
        self._smooth_step_size  = float(smooth_step_size)
        self._pin_borders       = bool(pin_borders)
        self._weight            = float(weight)
        self._method            = self._normalize_method(method)

    # ------------------------------ properties ------------------------------- #

    @staticmethod
    def _normalize_method(value: str) -> str:
        v = str(value).upper()
        if v not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}; got {value!r}")
        return v

    @property
    def smooth_iterations(self) -> int:
        """Laplacian smoothing iteration count."""
        return self._smooth_iterations

    @smooth_iterations.setter
    def smooth_iterations(self, value: int) -> None:
        v = int(value)
        if v < 0:
            raise ValueError("smooth_iterations must be >= 0")
        if v != self._smooth_iterations:
            self._smooth_iterations = v
            self._invalidate_bind()

    @property
    def smooth_step_size(self) -> float:
        """Per-iteration reception factor passed to ``MeshData.smooth``."""
        return self._smooth_step_size

    @smooth_step_size.setter
    def smooth_step_size(self, value: float) -> None:
        v = float(value)
        if v != self._smooth_step_size:
            self._smooth_step_size = v
            self._invalidate_bind()

    @property
    def pin_borders(self) -> bool:
        """Whether border vertices are pinned (iterations forced to 0)."""
        return self._pin_borders

    @pin_borders.setter
    def pin_borders(self, value: bool) -> None:
        v = bool(value)
        if v != self._pin_borders:
            self._pin_borders = v
            self._invalidate_bind()

    @property
    def weight(self) -> float:
        """Mush blend weight in ``[0, 1]``.  Cheap to change."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = float(np.clip(value, 0.0, 1.0))

    @property
    def method(self) -> str:
        """Frame-construction method -- ``"TBN"``, ``"PROCRUSTES"`` or ``"DDM"``."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        v = self._normalize_method(value)
        if v != self._method:
            self._method = v
            self._invalidate_bind()

    @property
    def smoothed_points(self) -> np.ndarray | None:
        """The smoothed rest positions cached during :meth:`bind`."""
        return self._smooth_points

    @property
    def local_deltas(self) -> np.ndarray | None:
        """Per-vertex local-frame displacements ``(N, 3)``.

        For the ``"TBN"`` method these are tangent-space coordinates;
        for ``"PROCRUSTES"`` and ``"DDM"`` they are world-space
        ``rest - smooth_rest`` offsets.
        """
        if self._method == "TBN":
            return self._local_deltas
        return self._world_deltas

    @property
    def points(self) -> np.ndarray | None:
        """Most recent reconstructed positions ``(N, 3)``."""
        return self._points

    @property
    def valid(self) -> bool:
        """True once :meth:`bind` has been called for the current method."""
        if self._smooth_points is None:
            return False
        if self._method == "TBN":
            return self._local_deltas is not None and self._first_nbrs is not None
        if self._method == "PROCRUSTES":
            return self._world_deltas is not None
        if self._method == "DDM":
            return (
                self._world_deltas is not None
                and self._ddm_offsets is not None
                and self._ddm_weights is not None
            )
        return False

    # ------------------------ private helpers -------------------------------- #

    def _invalidate_bind(self) -> None:
        """Drop all per-method bind caches.

        ``_first_nbrs`` is kept: it is topology read from the rest mesh at
        construction, and no setting changes it.
        """
        self._smooth_points = None
        self._local_deltas  = None
        self._world_deltas  = None
        self._ddm_offsets   = None
        self._ddm_weights   = None

    def _build_iteration_schedule(self) -> np.ndarray:
        """Build the per-vertex iteration count array.

        Maya's ``deltaMush`` performs ``smoothingIterations - 1`` actual
        blur passes -- the public :func:`cgmath.geometry.utils.blur` API
        documents the same off-by-one convention with its default of
        ``iterations=9`` mapping to Maya's "10".  We replicate that here
        so a caller asking for ``smooth_iterations=10`` matches Maya
        exactly.

        Border vertices are forced to ``0`` when ``pin_borders`` is set,
        so the blur leaves them exactly where they are.

        Built on every call rather than cached: it is O(N) against an
        O(N * K * iterations) blur, and a cache went stale when
        ``smooth_iterations`` or ``border_vertices`` changed.
        """
        n     = self.rest_points.shape[0]
        iters = max(0, self._smooth_iterations - 1)
        sched = np.full(n, iters, dtype=np.int32)
        if (
            self._pin_borders
            and self.border_vertices is not None
            and self.border_vertices.size > 0
        ):
            sched[self.border_vertices] = 0
        return sched

    def _smooth(self, points: np.ndarray) -> np.ndarray:
        """Run the bound smoothing schedule on an arbitrary point set.

        Calls ``blur`` -- what :meth:`MeshData.smooth` runs -- directly,
        with the per-vertex iteration count as an array (zeros at border
        vertices when pinning is enabled).  The input is not modified.
        """
        from cgmath.geometry.utils import blur

        # Maya's off-by-one means an input of 0 *or* 1 produces zero
        # actual blur passes -- short-circuit both for performance.
        if self._smooth_iterations <= 1:
            return np.asarray(points, dtype=np.float64).copy()

        # blur smooths a copy of the points it is given
        return blur(
            np.ascontiguousarray(points, dtype=np.float64),
            self.neighbors,
            iterations    = self._build_iteration_schedule(),
            receptions    = self._smooth_step_size,
            contributions = 1.0,
        )

    @staticmethod
    def _compute_first_face_neighbors(mesh) -> np.ndarray:
        """Per-vertex ``[next, prev]`` indices from the first face containing it.

        For every vertex, records the ``next`` (CCW successor) and ``prev``
        (CW predecessor) vertex at its first corner in the *first face*
        (lowest face_id) that contains it.  Vertices not present in any
        face -- or only used as isolated points -- get ``[-1, -1]`` and
        the TBN encoder falls back to a world-space identity store.
        """
        indices = np.asarray(mesh.indices, dtype=np.int64).ravel()
        counts  = np.asarray(mesh.counts, dtype=np.int64).ravel()
        out     = np.full((len(mesh.points), 2), -1, dtype=np.int32)
        if indices.size == 0:
            return out

        # each corner's face start, face size and position in its face
        starts = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        face  = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        start = starts[face]
        size  = counts[face]
        pos   = np.arange(indices.size, dtype=np.int64) - start

        # corners run face by face in winding order, so a vertex's first
        # occurrence is its first corner in its lowest face
        vertices, first = np.unique(indices, return_index=True)
        start, size, pos = start[first], size[first], pos[first]

        out[vertices, 0] = indices[start + (pos + 1) % size]  # next in winding
        out[vertices, 1] = indices[start + (pos - 1) % size]  # previous in winding
        return out

    # ---------------------------- bind / apply ------------------------------- #

    def bind(self) -> None:
        """Cache the smoothed rest mesh and per-vertex deltas / matrices.

        Calling this is **optional** -- :meth:`apply` will auto-bind on the
        first call.  Use it to pre-warm the cache (e.g. before a tight
        animation loop) or after changing several parameters that
        invalidate the bind (``smooth_iterations``, ``smooth_step_size``,
        ``pin_borders``, ``method``).
        """
        from cgmath.geometry.utils import ddm_precompute, encode_local_deltas_tbn
        from cgmath.geometry.utils._numba._delta_mush import (
            _delta_mush_encode_tbn_numpy,
        )

        self._smooth_points = self._smooth(self.rest_points)
        self._world_deltas  = (self.rest_points - self._smooth_points).copy()

        if self._method == "TBN":
            # TBN needs first-face winding -- only available for mesh inputs.
            # Fall back to DDM on point clouds.
            if self._first_nbrs is None:
                self._method = "DDM"
            else:
                try:
                    self._local_deltas = encode_local_deltas_tbn(
                        self.rest_points, self._smooth_points, self._first_nbrs
                    )
                except Exception:
                    self._local_deltas = _delta_mush_encode_tbn_numpy(
                        self.rest_points, self._smooth_points, self._first_nbrs
                    )

        if self._method == "DDM":
            self._ddm_offsets, self._ddm_weights = ddm_precompute(
                self._smooth_points, self.neighbors
            )

        # PROCRUSTES needs nothing beyond _world_deltas + _smooth_points.

        self._points = self.rest_points.copy()

    def _decode(self, smooth_deformed: np.ndarray) -> np.ndarray:
        """Dispatch to the configured frame-construction method's decode kernel.

        Falls back to the pure-numpy implementation if the numba kernel
        fails (e.g. unsupported environment, dtype issue).
        """
        from cgmath.geometry.utils import (
            decode_local_deltas_tbn,
            decode_world_deltas_ddm,
            decode_world_deltas_procrustes,
        )
        from cgmath.geometry.utils._numba._delta_mush import (
            _delta_mush_decode_ddm_numpy,
            _delta_mush_decode_procrustes_numpy,
            _delta_mush_decode_tbn_numpy,
        )

        if self._method == "TBN":
            try:
                return decode_local_deltas_tbn(
                    smooth_deformed, self._first_nbrs, self._local_deltas
                )
            except Exception:
                return _delta_mush_decode_tbn_numpy(
                    smooth_deformed, self._first_nbrs, self._local_deltas
                )

        if self._method == "PROCRUSTES":
            try:
                return decode_world_deltas_procrustes(
                    self._smooth_points,
                    smooth_deformed,
                    self.neighbors,
                    self._world_deltas,
                )
            except Exception:
                return _delta_mush_decode_procrustes_numpy(
                    self._smooth_points,
                    smooth_deformed,
                    self.neighbors,
                    self._world_deltas,
                )

        # DDM
        try:
            return decode_world_deltas_ddm(
                smooth_deformed,
                self.neighbors,
                self._ddm_offsets,
                self._ddm_weights,
                self._world_deltas,
            )
        except Exception:
            return _delta_mush_decode_ddm_numpy(
                smooth_deformed,
                self.neighbors,
                self._ddm_offsets,
                self._ddm_weights,
                self._world_deltas,
            )

    def _blend(self, base: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Linearly blend ``base`` and ``reconstructed`` by :attr:`weight`."""
        from cgmath.geometry.utils import blend_deltas

        if self._weight >= 1.0:
            return reconstructed
        if self._weight <= 0.0:
            return base.copy()
        try:
            return blend_deltas(base, reconstructed, float(self._weight))
        except Exception:
            return base + (reconstructed - base) * self._weight

    def apply(self, target):
        """Apply Delta Mush to a deformed pose of the bound mesh.

        Auto-binds on first call (or after any property change that
        invalidates the cache), so calling :meth:`bind` explicitly is
        only needed when you want to pay the smoothing cost up front.

        Args:
            target: Deformed positions.  Accepts a :class:`MeshData` (a
                copy is returned with updated points) or a raw
                ``(N, 3)`` array.

        Returns:
            If *target* is a MeshData-like object, a copy of it with
            delta-mushed points.  Otherwise the deformed point array.
        """
        if not self.valid:
            self.bind()

        is_mesh = hasattr(target, "points") and not isinstance(target, np.ndarray)
        pts     = target.points if is_mesh else target
        pts     = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))

        if pts.shape != self.rest_points.shape:
            raise ValueError(
                f"target shape {pts.shape} does not match bound shape "
                f"{self.rest_points.shape}"
            )

        # 1. smooth the deformed pose with the same schedule
        smooth_deformed = self._smooth(pts)

        # 2. decode using the configured frame-construction method
        reconstructed = self._decode(smooth_deformed)

        # 3. optional blend against the raw deformed input.  When weight is 0
        #    we want the smoothed mesh, not the unmodified input.
        if self._weight <= 0.0:
            self._points = smooth_deformed.copy()
        else:
            self._points = self._blend(pts, reconstructed)

        if is_mesh:
            result        = target.copy()
            result.points = self._points.copy()
            return result
        return self._points