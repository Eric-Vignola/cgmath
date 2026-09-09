from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cgmath.geometry._base import Data
from cgmath.geometry.utils.main import (
    assign_cells,
    bernstein_eval,
    build_lattice_topology,
)


# ========================= FFDData class ================================= #


@dataclass(repr=False, eq=False)
class FFDData(Data):
    """Free-Form Deformation over a non-uniform structured lattice.

    The lattice is a 3-D grid of ``(lx x ly x lz)`` control points.
    Control points need NOT be uniformly spaced -- any initial shape is valid
    as long as the hexahedral cells are reasonably non-degenerate.

    Uses a displacement-based approach: displacements are interpolated from
    ``deformed_lattice - rest_lattice`` and added to the original mesh points.
    This guarantees that at rest (zero displacement) the mesh is unchanged.

    **Interpolation** is controlled by ``.local_influence`` (Maya convention)::

        local_influence = (2, 2, 2)   # trilinear -- only the containing cell (default)
        local_influence = (3, 3, 3)   # quadratic Bernstein -- 3 CPs per axis
        local_influence = (4, 4, 4)   # cubic Bernstein -- 4 CPs per axis

    **Local influence** (set via ``.local_influence``)::

        local_influence = (3, 3, 3)   # Bernstein smoothing with 3 CPs per axis

    All sizes use the Maya convention -- **control-point counts**, not cell
    counts.  A ``2x2x2`` lattice is the minimum (one cell per axis).

    **Outside modes** (set via ``.outside``)::

        outside = "extrapolate"   # continue deformation past the boundary
        outside = "freeze"        # no deformation outside the lattice
        outside = "falloff"       # smooth decay over ``falloff_radius`` cells

    Example::

        ffd = FFDData.from_mesh(mesh, divisions=(5, 5, 5))
        ffd.bind(mesh)

        ffd.local_influence = (4, 4, 4)  # cubic Bernstein smoothing
        ffd.outside = "falloff"           # smooth boundary decay
        ffd.falloff_radius = 2.0          # over 2 cell-widths

        deformed_lattice = ffd.lattice.copy()
        deformed_lattice[:, :, -1, 2] += 5.0
        ffd.update(deformed_lattice)

        mesh.points = ffd.points
    """

    lattice:   np.ndarray = None
    divisions: np.ndarray = None

    # --- cached --- #
    _source          = None
    _source_mesh     = None
    _cells           = None
    _uvw             = None
    _weights         = None
    _points          = None
    _outside         = None
    _falloff_radius  = None
    _local_influence = None

    # --------------------------- construction -------------------------------- #

    def __init__(self, lattice, divisions: tuple[int, int, int] | None = None):
        """Initialise the FFD with a rest-pose lattice.

        Uses the Maya convention: *divisions* is the number of **control
        points** per axis (minimum 2).  A ``(3, 3, 3)`` lattice has 3 CPs
        per axis and 2 cells per axis.

        Args:
            lattice: Control points.  Accepts:
                - ``np.ndarray`` shaped ``(lx, ly, lz, 3)`` -- *divisions* inferred
                - ``np.ndarray`` shaped ``(N, 3)`` -- *divisions* required
                - object with ``.points`` attribute (e.g. MeshData) -- *divisions*
                  required
            divisions: ``(nx, ny, nz)`` control points per axis (Maya convention).
                Minimum is ``(2, 2, 2)``.
        """
        pts = lattice
        if not isinstance(pts, np.ndarray) and hasattr(pts, "points"):
            pts = pts.points.copy()
        pts = np.asarray(pts, dtype=np.float64)

        if pts.ndim == 4:
            self.lattice   = pts
            self.divisions = np.array(pts.shape[:3], dtype=int)
        elif divisions is not None:
            d        = np.asarray(divisions, dtype=int)
            expected = int(np.prod(d))
            if pts.shape[0] != expected:
                raise ValueError(
                    f"Expected {expected} points for divisions {tuple(d)}, "
                    f"got {pts.shape[0]}"
                )
            self.lattice   = pts.reshape(*d, 3)
            self.divisions = d
        else:
            raise ValueError("divisions required when lattice is a flat (N, 3) array")

        self._outside         = "extrapolate"
        self._falloff_radius  = 2.0
        self._local_influence = (2, 2, 2)

    # ----------------------- convenience constructors ------------------------ #

    @staticmethod
    def create_lattice(
        divisions: tuple[int, int, int]        = (5, 5, 5),
        bbox_min:  np.ndarray           | None = None,
        bbox_max:  np.ndarray           | None = None,
        name:      str                         = "ffd1",
    ):
        """Create a structured lattice as a MeshData with quad faces.

        Returns a unit-cube ``[-0.5, 0.5]^3`` lattice by default.  Pass
        *bbox_min* / *bbox_max* to place it in world space.

        Uses the Maya convention: *divisions* is the number of **control
        points** per axis (minimum 2).

        Args:
            divisions: ``(nx, ny, nz)`` control points per axis.
            bbox_min: ``(3,)`` minimum corner.  Defaults to ``[-0.5, -0.5, -0.5]``.
            bbox_max: ``(3,)`` maximum corner.  Defaults to ``[0.5, 0.5, 0.5]``.

        Returns:
            MeshData with ``nx x ny x nz`` points and all grid quads.
        """
        from cgmath.geometry.mesh import MeshData

        lo = np.full(3, -0.5) if bbox_min is None else np.asarray(bbox_min)
        hi = np.full(3, 0.5) if bbox_max is None else np.asarray(bbox_max)

        d      = np.asarray(divisions, dtype=int)
        axes   = [np.linspace(a, b, n) for a, b, n in zip(lo, hi, d)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        lx, ly, lz = d
        indices, counts = build_lattice_topology(int(lx), int(ly), int(lz))

        return MeshData(indices=indices, counts=counts, points=points, name=name)

    @classmethod
    def from_mesh(
        cls,
        mesh,
        divisions: tuple[int, int, int],
    ) -> FFDData:
        """Create an FFD from a pre-shaped lattice mesh.

        The mesh must have exactly ``nx x ny x nz`` points in
        structured-grid order (as produced by :meth:`create_lattice`).

        Args:
            mesh: MeshData or ``(N, 3)`` array with lattice control points.
            divisions: ``(nx, ny, nz)`` control points per axis (Maya convention).

        Returns:
            A new :class:`FFDData` with the mesh points as the rest lattice.
        """
        pts = mesh.points if hasattr(mesh, "points") else np.asarray(mesh)
        return cls(pts, divisions=divisions)

    # ------------------- interpolation / outside settings -------------------- #

    @property
    def outside(self) -> str:
        """How points outside the lattice are handled.

        - ``"extrapolate"`` -- continue deformation past the boundary
        - ``"freeze"``      -- no deformation outside the lattice
        - ``"falloff"``     -- smooth decay over :attr:`falloff_radius` cells
        """
        return self._outside

    @outside.setter
    def outside(self, mode: str):
        if mode not in ("extrapolate", "freeze", "falloff"):
            raise ValueError(f"Unknown outside mode: {mode!r}")
        self._outside = mode
        self._recompute_weights()

    @property
    def local_influence(self) -> tuple[int, int, int]:
        """Per-axis Bernstein influence width in control points.

        Uses the Maya convention: ``(2, 2, 2)`` is the minimum (no
        smoothing, trilinear only -- default).  Higher values evaluate
        a wider Bernstein polynomial window, spreading each control
        point's influence over more of the lattice.  Matches Maya's
        ``localInfluenceS/T/U`` attributes.
        """
        return self._local_influence

    @local_influence.setter
    def local_influence(self, value: tuple[int, int, int] | int):
        if isinstance(value, (int, float)):
            value = (int(value), int(value), int(value))
        else:
            value = tuple(int(v) for v in value)
        if len(value) != 3 or any(v < 2 for v in value):
            raise ValueError("local_influence must be 3 ints >= 2")
        self._local_influence = value

    @property
    def falloff_radius(self) -> float:
        """Distance in cell-widths over which falloff decays to zero."""
        return self._falloff_radius

    @falloff_radius.setter
    def falloff_radius(self, value: float):
        self._falloff_radius = max(float(value), 1e-10)
        if self._outside == "falloff":
            self._recompute_weights()

    def _recompute_weights(self) -> None:
        """Recompute per-point deformation weights from source positions.

        Uses the lattice bounding box directly rather than solver-dependent
        parametric coordinates, so the result is stable regardless of
        Newton-solver convergence behaviour across platforms.
        """
        if self._source is None or self._uvw is None:
            return

        if self._outside == "extrapolate":
            self._weights = np.ones(len(self._uvw))
            return

        # Compute distance from each source point to the lattice bbox.
        flat = self.lattice.reshape(-1, 3)
        lo, hi = flat.min(axis=0), flat.max(axis=0)

        below     = np.maximum(lo - self._source, 0.0)
        above     = np.maximum(self._source - hi, 0.0)
        bbox_dist = np.max(below + above, axis=1)  # per-point scalar

        if self._outside == "freeze":
            self._weights = (bbox_dist < 1e-6).astype(np.float64)

        else:  # falloff
            # Convert falloff_radius from cell-widths to world units
            size          = hi - lo
            div           = np.maximum(self.divisions - 1, 1).astype(np.float64)
            cell_size     = np.mean(size / div)
            radius_world  = self._falloff_radius * max(cell_size, 1e-12)

            t             = np.clip(bbox_dist / radius_world, 0.0, 1.0)
            self._weights = 1.0 - t * t * (3.0 - 2.0 * t)  # smoothstep

    # ---------------------------- bind / update ------------------------------ #

    def bind(self, target) -> None:
        """Compute and store parametric coordinates for each mesh point.

        This is the expensive step (Newton solve) and only needs to be done
        once per mesh.  After binding, :attr:`local_influence`, :attr:`outside`,
        and :attr:`falloff_radius` can be changed freely without rebinding.

        If *target* is a MeshData-like object, :meth:`update` will return a
        copy of that object with updated points.

        Args:
            target: ``(N, 3)`` array or object with ``.points``.
        """
        if not isinstance(target, np.ndarray) and hasattr(target, "points"):
            self._source_mesh = target
            pts               = target.points
        else:
            self._source_mesh = None
            pts               = target
        self._source = np.asarray(pts, dtype=np.float64).copy()

        self._cells, self._uvw = assign_cells(self._source, self.lattice)
        self._recompute_weights()
        self._points = self._source.copy()

    def update(self, lattice):
        """Deform the bound mesh with a new lattice pose.

        Args:
            lattice: Deformed control points -- same shape / order as the rest
                lattice.  Accepts the same formats as ``__init__``.

        Returns:
            If :meth:`bind` was called with a MeshData-like object, returns
            a copy of that object with the deformed points.  Otherwise
            returns the deformed points array ``(N, 3)``.
        """
        pts = lattice
        if not isinstance(pts, np.ndarray) and hasattr(pts, "points"):
            pts = pts.points
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 2:
            pts = pts.reshape(self.lattice.shape)
        self.compute(pts)

        if self._source_mesh is not None:
            result        = self._source_mesh.copy()
            result.points = self._points.copy()
            return result

        return self._points

    def compute(self, deformed_lattice: np.ndarray) -> None:
        """Evaluate the deformation (displacement-based).

        Computes ``source + interp(deformed - rest) x weight`` for each
        bound mesh point.

        Args:
            deformed_lattice: ``(lx, ly, lz, 3)`` deformed control points.
        """
        if self._cells is None or self._uvw is None:
            raise RuntimeError("call bind() before update()")

        delta = np.ascontiguousarray(deformed_lattice - self.lattice)

        # Convert Maya-convention local_influence (CPs) to internal cells
        li_cells  = tuple(v - 1 for v in self._local_influence)
        div_cells = self.divisions - 1

        displacement = bernstein_eval(
            self._cells,
            self._uvw,
            delta,
            li_cells,
            div_cells,
        )

        self._points = self._source + displacement * self._weights[:, None]

    # --------------------------- output properties --------------------------- #

    @property
    def points(self) -> np.ndarray | None:
        """Deformed mesh points ``(N, 3)``, or ``None`` before ``bind()``."""
        return self._points

    @property
    def uvw(self) -> np.ndarray | None:
        """Stored parametric coordinates ``(N, 3)``."""
        return self._uvw

    @property
    def cells(self) -> np.ndarray | None:
        """Stored cell indices ``(N, 3)``."""
        return self._cells

    @property
    def weights(self) -> np.ndarray | None:
        """Per-point deformation weights ``(N,)``."""
        return self._weights

    @property
    def valid(self) -> bool:
        """True if a mesh has been bound."""
        return self._cells is not None and self._uvw is not None

    def to_mesh(self, lattice: np.ndarray | None = None):
        """Convert the lattice to a MeshData with quad faces.

        Useful for visualising, exporting, or round-tripping the lattice
        through external tools.

        Args:
            lattice: Optional deformed lattice ``(lx, ly, lz, 3)`` or
                ``(N, 3)``.  Defaults to the rest lattice.

        Returns:
            MeshData with the lattice's structured-grid quad topology.
        """
        from cgmath.geometry.mesh import MeshData

        lat = lattice if lattice is not None else self.lattice
        if lat.ndim == 4:
            points = lat.reshape(-1, 3).copy()
        else:
            points = np.asarray(lat, dtype=np.float64).copy()

        lx, ly, lz = self.divisions
        indices, counts = build_lattice_topology(int(lx), int(ly), int(lz))

        return MeshData(indices=indices, counts=counts, points=points)