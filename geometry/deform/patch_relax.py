"""
Patch-based surface relaxation operator.

Public class:
    :class:`PatchRelaxData` -- bind/apply relaxation operator that smooths
        a deformed mesh while reproducing the patch layout (edge flow) of
        a baseline mesh.

Implementation of "Patch-based Surface Relaxation" by de Goes, Sheffler,
Comet, Martinez and Kutt (Pixar Animation Studios, SIGGRAPH '18 Talks --
https://doi.org/10.1145/3214745.3214768).

Unlike Laplacian smoothing -- or Delta Mush, which collapses edge spans --
the weights here are derived from a per-vertex *decal map*: a geodesic
polar flattening of the edge stencil.  Those weights favour neighbours
along the shorter part of an edge flow and orthogonal to long spans, so
relaxation follows the patch layout instead of washing it out.  The
displacement then transfers the baseline patch arrangement onto the
deformed pose using a per-vertex Procrustes rotation, and can optionally
be constrained to the surface for volume control.

The choice of baseline selects the relaxation profile:

* **modeling** -- baseline is the mesh being relaxed, so relative span
  spacing is preserved (``PatchRelaxData(mesh).apply(mesh)``)
* **rigging** -- baseline is the undeformed mesh, restoring rest features
  such as tension folds under articulation
* **cleanup** -- baseline is any shape in the rig stack, used to resolve
  clumping and foldovers from simulation

The numba-accelerated kernels backing this class live in
:mod:`cgmath.geometry.utils._numba._patch_relax`, with public Python
wrappers exposed via :mod:`cgmath.geometry.utils.main`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cgmath.geometry._base import Data


# =========================== PatchRelaxData ============================== #


@dataclass(repr=False, eq=False)
class PatchRelaxData(Data):
    """Patch-based surface relaxation operator.

    Relaxes a deformed mesh while preserving the edge flow of a baseline
    patch layout.  Binds to a base :class:`MeshData` and is then applied
    to any deformed pose sharing its topology.

    The bind step caches, from the baseline mesh:

    * winding-ordered 1-ring adjacency (the decal map needs neighbours in
      angular order, which ``MeshData.get_edge_vertex_neighbors`` does
      not provide)
    * the baseline decal maps -- each vertex's edge stencil flattened to
      geodesic polar coordinates
    * the span-aware edge weights derived from those decal maps
    * the weighted baseline offset per vertex

    All of that depends only on the baseline, so an animation loop pays
    for it once and each :meth:`apply` is pure iteration.

    Example::

        from cgmath.geometry import MeshData, PatchRelaxData

        rest = MeshData.load_obj("body.obj")
        relaxer = PatchRelaxData(rest, iterations=30)

        deformed = rest.copy()
        # ... apply skinning / simulation to deformed ...

        result = relaxer.apply(deformed)  # auto-binds on first call

    Volume-preserving cleanup of simulation clumping, mixing 80% of the
    surface-constrained solve with 20% of the 3D one as in the paper::

        relaxer.surface_blend = 0.8
        result = relaxer.apply(deformed)

    Plain span-aware smoothing, with no baseline restoration::

        relaxer.alpha = 0.0
        result = relaxer.apply(deformed)
    """

    rest_points:     np.ndarray = None
    ring:            np.ndarray = None
    valence:         np.ndarray = None
    is_boundary:     np.ndarray = None
    border_vertices: np.ndarray | None = None
    mask:            np.ndarray | None = None

    # --- cached --- #
    _rest_mesh     = None
    _rest_decals   = None  # (N, K, 2) baseline decal maps
    _weights       = None  # (N, K) span-aware weights
    _rest_vectors  = None  # (N, 3) weighted baseline offsets
    _iterations    = None
    _alpha         = None
    _step_size     = None
    _surface_blend = None
    _pin_borders   = None
    _points        = None

    # --------------------------- construction -------------------------------- #

    def __init__(
        self,
        base,
        iterations:    int               = 30,
        alpha:         float             = 1.0,
        surface_blend: float             = 0.0,
        step_size:     float             = 0.5,
        pin_borders:   bool              = True,
        mask:          np.ndarray | None = None,
    ):
        """Create a patch relaxation operator.

        Args:
            base: Baseline patch layout as a :class:`MeshData`.  Its
                topology drives the ring adjacency and its points define
                the patch arrangement to reproduce.
            iterations: Number of Jacobi sweeps per :meth:`apply`.
            alpha: Blend between the baseline correction and plain
                relaxation.  ``1`` (default) fully restores the baseline
                layout, ``0`` reduces to span-aware smoothing.
            surface_blend: Fraction of the surface-constrained (volume
                preserving) solve mixed over the 3D one, in ``[0, 1]``.
                ``0`` (default) is unconstrained 3D relaxation.  Costs an
                extra decal-map rebuild per iteration when non-zero.
            step_size: Explicit update fraction -- the paper's half step.
                Larger converges faster but can oscillate.
            pin_borders: When True (default) border vertices are held in
                place, so an open mesh does not shrink at its edges.
            mask: Optional ``(N,)`` per-vertex weight in ``[0, 1]``
                scaling each vertex's step, for painted falloff.  ``0``
                pins a vertex; pinned vertices still influence their
                neighbours.

        Raises:
            ValueError: If *base* is not mesh-like, or *mask* does not
                match the point count.
        """
        # local import to avoid circular dependency with cgmath.geometry.mesh
        from cgmath.geometry.mesh import MeshData

        if not isinstance(base, MeshData):
            if not all(hasattr(base, x) for x in ("points", "indices", "counts")):
                raise ValueError(
                    "base must be a MeshData -- patch relaxation needs face "
                    "topology to order each vertex's 1-ring"
                )

        self._rest_mesh = base
        self.rest_points = np.ascontiguousarray(
            np.asarray(base.points, dtype=np.float64)
        )

        borders = base.get_border_vertices(flatten=True)
        borders = np.asarray(borders, dtype=np.int64).ravel()
        self.border_vertices = borders if borders.size else None

        self.mask            = None
        if mask is not None:
            mask = np.asarray(mask, dtype=np.float64).ravel()
            if mask.size != self.rest_points.shape[0]:
                raise ValueError(
                    f"mask size ({mask.size}) must match point count "
                    f"({self.rest_points.shape[0]})"
                )
            self.mask = np.clip(mask, 0.0, 1.0)

        self._iterations    = int(iterations)
        self._alpha         = float(alpha)
        self._surface_blend = float(np.clip(surface_blend, 0.0, 1.0))
        self._step_size     = float(step_size)
        self._pin_borders   = bool(pin_borders)

    # ------------------------------ properties ------------------------------- #

    @property
    def iterations(self) -> int:
        """Number of Jacobi sweeps per apply.  Cheap to change."""
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        v = int(value)
        if v < 0:
            raise ValueError("iterations must be >= 0")
        self._iterations = v

    @property
    def alpha(self) -> float:
        """Baseline-restoration blend in ``[0, 1]``.  Cheap to change."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(value)

    @property
    def surface_blend(self) -> float:
        """Surface-constrained mix in ``[0, 1]``.  Cheap to change."""
        return self._surface_blend

    @surface_blend.setter
    def surface_blend(self, value: float) -> None:
        self._surface_blend = float(np.clip(value, 0.0, 1.0))

    @property
    def step_size(self) -> float:
        """Explicit update fraction.  Cheap to change."""
        return self._step_size

    @step_size.setter
    def step_size(self, value: float) -> None:
        self._step_size = float(value)

    @property
    def pin_borders(self) -> bool:
        """Whether border vertices are held in place."""
        return self._pin_borders

    @pin_borders.setter
    def pin_borders(self, value: bool) -> None:
        self._pin_borders = bool(value)

    @property
    def weights(self) -> np.ndarray | None:
        """Span-aware edge weights ``(N, K)`` cached during :meth:`bind`."""
        return self._weights

    @property
    def decal_maps(self) -> np.ndarray | None:
        """Baseline decal maps ``(N, K, 2)`` cached during :meth:`bind`."""
        return self._rest_decals

    @property
    def points(self) -> np.ndarray | None:
        """Most recent relaxed positions ``(N, 3)``."""
        return self._points

    @property
    def valid(self) -> bool:
        """True once :meth:`bind` has run."""
        return (
            self.ring is not None
            and self._rest_decals is not None
            and self._weights is not None
            and self._rest_vectors is not None
        )

    # ------------------------ private helpers -------------------------------- #

    def _build_step_scale(self) -> np.ndarray:
        """Per-vertex step multiplier combining the mask and border pinning.

        Rebuilt per apply rather than cached: it is O(N) against an O(N *
        valence * iterations) solve, and caching it would go stale the
        moment ``mask`` is reassigned.
        """
        n     = self.rest_points.shape[0]
        scale = np.ones(n, dtype=np.float64) if self.mask is None else self.mask.copy()

        if self._pin_borders and self.border_vertices is not None:
            scale[self.border_vertices] = 0.0

        return scale

    def _compute_rest_vectors(self) -> np.ndarray:
        """``sum_k w_k (rest_j - rest_i)`` -- the baseline relative position."""
        ring    = np.where(self.ring < 0, 0, self.ring)
        offsets = self.rest_points[ring] - self.rest_points[:, None, :]
        return np.einsum("nk,nkd->nd", self._weights, offsets)

    # ---------------------------- bind / apply ------------------------------- #

    def bind(self) -> None:
        """Cache the ring adjacency, baseline decal maps and span weights.

        Calling this is **optional** -- :meth:`apply` auto-binds on the
        first call.  Use it to pre-warm the cache before a tight
        animation loop.
        """
        from cgmath.geometry.utils import (
            build_vertex_rings,
            compute_decal_maps,
            compute_span_weights,
        )

        self.ring, self.valence, self.is_boundary = build_vertex_rings(
            self._rest_mesh.indices,
            self._rest_mesh.counts,
            self.rest_points.shape[0],
        )

        # Sec.2 -- flatten the baseline stencils, Sec.3 -- weight them
        self._rest_decals = compute_decal_maps(
            self.rest_points, self.ring, self.valence, self.is_boundary
        )
        self._weights      = compute_span_weights(self._rest_decals, self.valence)
        self._rest_vectors = self._compute_rest_vectors()

        self._points       = self.rest_points.copy()

    def relax(self, points: np.ndarray) -> np.ndarray:
        """Relax a raw point array against the bound baseline.

        Args:
            points: ``(N, 3)`` deformed positions.

        Returns:
            ``(N, 3)`` relaxed positions.  The input is not modified.

        Raises:
            ValueError: If the point count does not match the baseline.
        """
        from cgmath.geometry.utils import patch_relax

        if not self.valid:
            self.bind()

        points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        if points.shape != self.rest_points.shape:
            raise ValueError(
                f"target shape {points.shape} does not match bound shape "
                f"{self.rest_points.shape}"
            )

        return patch_relax(
            points,
            self.rest_points,
            self.ring,
            self.valence,
            self.is_boundary,
            self._weights,
            self._rest_vectors,
            self._rest_decals,
            iterations    = self._iterations,
            alpha         = self._alpha,
            step_size     = self._step_size,
            surface_blend = self._surface_blend,
            step_scale    = self._build_step_scale(),
        )

    def apply(self, target):
        """Relax a deformed pose of the bound mesh.

        Auto-binds on first call, so calling :meth:`bind` explicitly is
        only needed to pay the setup cost up front.

        Args:
            target: Deformed positions.  Accepts a :class:`MeshData` (a
                copy is returned with updated points) or a raw ``(N, 3)``
                array.

        Returns:
            If *target* is mesh-like, a copy of it with relaxed points.
            Otherwise the relaxed point array.
        """
        is_mesh = hasattr(target, "points") and not isinstance(target, np.ndarray)
        pts     = target.points if is_mesh else target

        self._points = self.relax(pts)

        if is_mesh:
            result = target.copy()
            result.points = self._points.copy()
            return result
        return self._points