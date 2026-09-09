from dataclasses import dataclass, field

import numpy as np
from cgmath.geometry._base import Data
from cgmath.geometry.bspline import SampleData
from cgmath.geometry.utils import compute_basis
from cgmath.geometry.utils._numba._bspline import (
    _compute_basis_parallel,
    _evaluate_bspline_surface,
    _evaluate_bspline_surface_derivatives,
    _newton_closest_point_surface_parallel,
    _raycast_bspline_surface_parallel,
)
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

try:
    from scipy.interpolate import NdBSpline as _NdBSpline
except ImportError:
    _NdBSpline = None


@dataclass(repr=False, eq=False)
class PatchSampleData(SampleData):
    """
    Sample data for B-spline surface projections.

    Fields
    ------
    points : np.ndarray, (m, dims)
        Projected points on the surface.
    tangents : np.ndarray, (m, 2, dims)
        tangents[:, 0] = dS/du, tangents[:, 1] = dS/dv.
    distances : np.ndarray, (m,)
        Distance from query to projected point.
    params : np.ndarray, (m, 2)
        params[:, 0] = u, params[:, 1] = v.
    basis : np.ndarray, (m, nu*nv)
        Flattened tensor-product basis values.
    """

    @property
    def normals(self):
        """Surface normals via cross(dS/du, dS/dv), normalized."""
        t_u   = self.tangents[:, 0]
        t_v   = self.tangents[:, 1]
        n     = np.cross(t_u, t_v)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-14)
        return n / norms

    def compute(self, values):
        """Compute weighted values from basis. values can be (nu, nv, d) or (nu*nv, d)."""
        d = values.shape[-1]
        return np.einsum("jk,ij->ik", values.reshape(-1, d), self.basis)

    def copy(self):
        return PatchSampleData(
            points    = np.copy(self.points),
            tangents  = np.copy(self.tangents),
            distances = np.copy(self.distances),
            params    = np.copy(self.params),
            basis     = np.copy(self.basis),
        )


@dataclass(repr=False, eq=False)
class PatchRaycastData(Data):
    """
    Ray-surface intersection data for a B-spline patch.

    Mirrors the shape of ``PatchSampleData`` with extra ``occluded`` and
    ``hit`` masks. For misses, ``points`` falls back to the ray origin,
    ``distances`` is NaN, ``params`` is NaN, and ``basis`` is computed at
    a clamped (0, 0) so the array still has a valid shape.

    Fields
    ------
    points : np.ndarray, (n, dims)
        Intersection points (origins for misses).
    tangents : np.ndarray, (n, 2, dims)
        ``tangents[:, 0]`` = dS/du, ``tangents[:, 1]`` = dS/dv at hits.
    distances : np.ndarray, (n,)
        Ray ``t`` values at hits, NaN for misses.
    params : np.ndarray, (n, 2)
        ``params[:, 0]`` = u, ``params[:, 1]`` = v at hits (user space).
    basis : np.ndarray, (n, count_u * count_v)
        Flattened tensor-product basis values at hits.
    occluded : np.ndarray, (n,)
        Boolean mask: ray direction faces into the surface (front-face hit).
    hit : np.ndarray, (n,)
        Boolean mask -- True where the ray intersected the surface.
    """

    points:    np.ndarray
    tangents:  np.ndarray
    distances: np.ndarray
    params:    np.ndarray
    basis:     np.ndarray
    occluded:  np.ndarray
    hit:       np.ndarray

    @property
    def normals(self):
        """Surface normals via cross(dS/du, dS/dv), normalized."""
        t_u   = self.tangents[:, 0]
        t_v   = self.tangents[:, 1]
        n     = np.cross(t_u, t_v)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-14)
        return n / norms

    def __call__(self, values):
        return self.compute(values)

    def compute(self, values):
        """Compute weighted values from basis. values can be (nu, nv, d) or (nu*nv, d)."""
        d = values.shape[-1]
        return np.einsum("jk,ij->ik", values.reshape(-1, d), self.basis)

    def copy(self):
        return PatchRaycastData(
            points    = np.copy(self.points),
            tangents  = np.copy(self.tangents),
            distances = np.copy(self.distances),
            params    = np.copy(self.params),
            basis     = np.copy(self.basis),
            occluded  = np.copy(self.occluded),
            hit       = np.copy(self.hit),
        )


@dataclass
class BSplinePatchData(Data):
    """
    B-spline tensor-product surface.

    Parameters
    ----------
    points : np.ndarray
        Control point grid (nu, nv, dims).
    degree_u, degree_v : int
        Degree per direction.
    periodic_u, periodic_v : bool
        Whether the surface is closed/periodic per direction.
    uniform_u, uniform_v : bool
        Arc-length parameterization per direction.
    use_numba : bool
        Use optimized Numba kernels (default True).
    registered_u, registered_v : bool
        Registration offset per direction.
    arc_length_samples : int
        Samples for arc-length lookup tables.
    """

    points:             np.ndarray
    degree_u:           int
    degree_v:           int
    periodic_u:         bool
    periodic_v:         bool
    uniform_u:          bool = field(default=False)
    uniform_v:          bool = field(default=False)
    use_numba:          bool = field(default=True)
    registered_u:       bool = field(default=False)
    registered_v:       bool = field(default=False)
    arc_length_samples: int = field(default=1000)

    _kv_u               = None
    _kv_v               = None
    _geometry_u         = None
    _geometry_v         = None
    _count_u            = None
    _count_v            = None
    _max_param_u        = None
    _max_param_v        = None
    _spl                = None  # scipy.interpolate.NdBSpline (real scipy fallback)
    _cv_f64             = None  # cached float64 control vertices with periodic wrap
    _arc_length_table_u = None
    _u_table_u          = None
    _total_length_u     = None
    _arc_length_table_v = None
    _u_table_v          = None
    _total_length_v     = None
    _cp_params_u        = None
    _cp_params_v        = None

    __repr__            = Data.__repr__

    def __eq__(self, other):
        return np.all(
            [
                np.allclose(self.cv, other.cv),
                self.degree_u == other.degree_u,
                self.degree_v == other.degree_v,
                self.periodic_u == other.periodic_u,
                self.periodic_v == other.periodic_v,
                self.uniform_u == other.uniform_u,
                self.uniform_v == other.uniform_v,
                self.registered_u == other.registered_u,
                self.registered_v == other.registered_v,
            ]
        )

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #

    def _init_bspline(self) -> None:
        """Build knot vectors, geometry indices, and optional arc-length tables."""
        self._count_u = self.points.shape[0]
        self._count_v = self.points.shape[1]

        # --- U direction ---
        if self.periodic_u:
            self._max_param_u = self._count_u
            self._kv_u = np.arange(
                -self.degree_u, self._count_u + self.degree_u + 1, dtype=np.float64
            )
            self._geometry_u = np.concatenate(
                [np.arange(self._count_u), np.arange(self.degree_u)]
            )
        else:
            self._max_param_u = self._count_u - self.degree_u
            self._kv_u = np.clip(
                np.arange(self._count_u + self.degree_u + 1, dtype=np.float64)
                - self.degree_u,
                0,
                self._count_u - self.degree_u,
            )
            self._geometry_u = np.arange(self._count_u)

        # --- V direction ---
        if self.periodic_v:
            self._max_param_v = self._count_v
            self._kv_v = np.arange(
                -self.degree_v, self._count_v + self.degree_v + 1, dtype=np.float64
            )
            self._geometry_v = np.concatenate(
                [np.arange(self._count_v), np.arange(self.degree_v)]
            )
        else:
            self._max_param_v = self._count_v - self.degree_v
            self._kv_v = np.clip(
                np.arange(self._count_v + self.degree_v + 1, dtype=np.float64)
                - self.degree_v,
                0,
                self._count_v - self.degree_v,
            )
            self._geometry_v = np.arange(self._count_v)

        # Cache float64 control vertices (with periodic wrap) for hot paths
        self._cv_f64 = self.cv.astype(np.float64)

        # Real scipy fallback via NdBSpline. Only initialized when scipy is
        # new enough to provide it (>= 1.13). Used when use_numba=False.
        if _NdBSpline is not None:
            self._spl = _NdBSpline(
                (self._kv_u, self._kv_v),
                self._cv_f64,
                (self.degree_u, self.degree_v),
            )
        else:
            self._spl = None

        if self.uniform_u:
            self._build_arc_length_table_u()
        if self.uniform_v:
            self._build_arc_length_table_v()

    def _build_arc_length_table_u(self) -> None:
        """Build arc-length table for U using a mid-V iso-curve."""
        n     = self.arc_length_samples
        v_mid = self._max_param_v / 2.0

        if self.registered_u and self.periodic_u:
            offset = self._max_param_u - (self.degree_u - 1) / 2.0
            self._u_table_u = np.linspace(
                offset, offset + self._max_param_u, n, dtype=np.float64
            )
            eval_u = self._u_table_u % self._max_param_u
        else:
            self._u_table_u = np.linspace(0, self._max_param_u, n, dtype=np.float64)
            eval_u = self._u_table_u

        eval_v = np.full(n, v_mid, dtype=np.float64)
        pts = _evaluate_bspline_surface(
            eval_u,
            eval_v,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )

        diffs   = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self._arc_length_table_u = np.zeros(n, dtype=np.float64)
        self._arc_length_table_u[1:] = np.cumsum(seg_len)
        self._total_length_u = self._arc_length_table_u[-1]
        if self._total_length_u > 0:
            self._arc_length_table_u /= self._total_length_u

    def _build_arc_length_table_v(self) -> None:
        """Build arc-length table for V using a mid-U iso-curve."""
        n     = self.arc_length_samples
        u_mid = self._max_param_u / 2.0

        if self.registered_v and self.periodic_v:
            offset = self._max_param_v - (self.degree_v - 1) / 2.0
            self._u_table_v = np.linspace(
                offset, offset + self._max_param_v, n, dtype=np.float64
            )
            eval_v = self._u_table_v % self._max_param_v
        else:
            self._u_table_v = np.linspace(0, self._max_param_v, n, dtype=np.float64)
            eval_v = self._u_table_v

        eval_u = np.full(n, u_mid, dtype=np.float64)
        pts = _evaluate_bspline_surface(
            eval_u,
            eval_v,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )

        diffs   = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self._arc_length_table_v = np.zeros(n, dtype=np.float64)
        self._arc_length_table_v[1:] = np.cumsum(seg_len)
        self._total_length_v = self._arc_length_table_v[-1]
        if self._total_length_v > 0:
            self._arc_length_table_v /= self._total_length_v

    # ------------------------------------------------------------------ #
    #  Parameter normalization                                            #
    # ------------------------------------------------------------------ #

    def _normalize_u(self, u):
        """User -> native parameter space for U."""
        u = np.asarray(u, dtype=np.float64)
        if self.registered_u and self.periodic_u and not self.uniform_u:
            offset = self._max_param_u - (self.degree_u - 1) / 2.0
            u      = u + offset
        if self.uniform_u:
            u = np.interp(
                u % 1.0 if self.periodic_u else u,
                self._arc_length_table_u,
                self._u_table_u,
            )
        return u

    def _normalize_v(self, v):
        """User -> native parameter space for V."""
        v = np.asarray(v, dtype=np.float64)
        if self.registered_v and self.periodic_v and not self.uniform_v:
            offset = self._max_param_v - (self.degree_v - 1) / 2.0
            v      = v + offset
        if self.uniform_v:
            v = np.interp(
                v % 1.0 if self.periodic_v else v,
                self._arc_length_table_v,
                self._u_table_v,
            )
        return v

    def _denormalize_u(self, u):
        """Native -> user parameter space for U."""
        u = np.asarray(u, dtype=np.float64)
        if self.uniform_u:
            if self.registered_u and self.periodic_u:
                offset = self._max_param_u - (self.degree_u - 1) / 2.0
                u      = np.where(u < offset, u + self._max_param_u, u)
            u = np.interp(u, self._u_table_u, self._arc_length_table_u)
        elif self.registered_u and self.periodic_u:
            offset = self._max_param_u - (self.degree_u - 1) / 2.0
            u      = u - offset
        return u

    def _denormalize_v(self, v):
        """Native -> user parameter space for V."""
        v = np.asarray(v, dtype=np.float64)
        if self.uniform_v:
            if self.registered_v and self.periodic_v:
                offset = self._max_param_v - (self.degree_v - 1) / 2.0
                v      = np.where(v < offset, v + self._max_param_v, v)
            v = np.interp(v, self._u_table_v, self._arc_length_table_v)
        elif self.registered_v and self.periodic_v:
            offset = self._max_param_v - (self.degree_v - 1) / 2.0
            v      = v - offset
        return v

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def cv(self) -> np.ndarray:
        """Control vertices with periodic wrapping applied (cu, cv, dims)."""
        if self._max_param_u is None:
            self._init_bspline()
        return self.points[np.ix_(self._geometry_u, self._geometry_v)]

    @property
    def count_u(self) -> int:
        if self._max_param_u is None:
            self._init_bspline()
        return self._count_u

    @property
    def count_v(self) -> int:
        if self._max_param_u is None:
            self._init_bspline()
        return self._count_v

    @property
    def kv_u(self) -> np.ndarray:
        """Unpadded U knot vector."""
        if self._max_param_u is None:
            self._init_bspline()
        return self._kv_u[1:-1]

    @property
    def kv_v(self) -> np.ndarray:
        """Unpadded V knot vector."""
        if self._max_param_u is None:
            self._init_bspline()
        return self._kv_v[1:-1]

    @property
    def geometry_u(self) -> np.ndarray:
        """Parametric U index map (with periodic wrap)."""
        if self._max_param_u is None:
            self._init_bspline()
        return self._geometry_u

    @property
    def geometry_v(self) -> np.ndarray:
        """Parametric V index map (with periodic wrap)."""
        if self._max_param_u is None:
            self._init_bspline()
        return self._geometry_v

    @property
    def max_param_u(self) -> int:
        if self._max_param_u is None:
            self._init_bspline()
        return self._max_param_u

    @property
    def max_param_v(self) -> int:
        if self._max_param_u is None:
            self._init_bspline()
        return self._max_param_v

    @property
    def length_u(self) -> float:
        """Fast approximation of total arc length along U (mid-V iso-curve)."""
        return self.get_length_u(fast=True)

    @property
    def length_v(self) -> float:
        """Fast approximation of total arc length along V (mid-U iso-curve)."""
        return self.get_length_v(fast=True)

    @property
    def total_length_u(self) -> float:
        """
        Total arc length along U.

        If uniform_u=True, returns the cached value from the arc-length table.
        Otherwise computes via fast sampling without mutating any state.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self._total_length_u is not None:
            return self._total_length_u
        return self.get_length_u(fast=True)

    @property
    def total_length_v(self) -> float:
        """
        Total arc length along V.

        If uniform_v=True, returns the cached value from the arc-length table.
        Otherwise computes via fast sampling without mutating any state.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self._total_length_v is not None:
            return self._total_length_v
        return self.get_length_v(fast=True)

    @property
    def area(self) -> float:
        return self.get_area()

    @property
    def control_point_params(self):
        """
        Per-direction parameter values for each control point row/column,
        representing the closest point on a representative iso-curve
        (mid-V for U, mid-U for V) to each CP.

        Uses Greville abscissae as initial guesses and Newton-Raphson
        refinement constrained to non-overlapping partitions derived
        from midpoints between adjacent Greville points.

        For periodic directions, returns count + 1 values where the
        last value is the first + max_param (closing the loop).

        For open directions, the first and last values are clamped at
        0 and max_param respectively.

        Returns are in user parameter space (denormalized for uniform
        and registered).

        Returns
        -------
        tuple of np.ndarray
            (params_u, params_v).
        """
        if self._cp_params_u is not None and self._cp_params_v is not None:
            return self._cp_params_u, self._cp_params_v

        if self._max_param_u is None:
            self._init_bspline()

        # Compute U params along iso-curve at v=mid
        v_mid = self._max_param_v / 2.0
        self._cp_params_u = self._cp_params_axis(
            n           = self._count_u,
            d           = self.degree_u,
            kv          = self._kv_u,
            max_param   = self._max_param_u,
            periodic    = self.periodic_u,
            uniform     = self.uniform_u,
            registered  = self.registered_u,
            denormalize = self._denormalize_u,
            iso_value   = v_mid,
            iso_axis    = "u",
        )

        # Compute V params along iso-curve at u=mid
        u_mid = self._max_param_u / 2.0
        self._cp_params_v = self._cp_params_axis(
            n           = self._count_v,
            d           = self.degree_v,
            kv          = self._kv_v,
            max_param   = self._max_param_v,
            periodic    = self.periodic_v,
            uniform     = self.uniform_v,
            registered  = self.registered_v,
            denormalize = self._denormalize_v,
            iso_value   = u_mid,
            iso_axis    = "v",
        )

        return self._cp_params_u, self._cp_params_v

    def _cp_params_axis(
        self,
        n,
        d,
        kv,
        max_param,
        periodic,
        uniform,
        registered,
        denormalize,
        iso_value,
        iso_axis,
    ):
        """
        Compute control-point parameters along one direction by Newton
        refinement of Greville abscissae against the representative
        iso-curve at the orthogonal direction's midpoint.
        """
        # Degree 1: parameters map directly to integer indices
        if d <= 1:
            return self._cp_params_degree_one(
                n, max_param, periodic, uniform, denormalize
            )

        # Greville abscissae and search bounds
        greville, u_min, u_max = self._cp_greville_bounds(n, d, kv, max_param, periodic)

        # Build query points (representative CPs along the iso direction)
        if iso_axis == "u":
            queries = self.points[np.arange(n), self._count_v // 2]
        else:
            queries = self.points[self._count_u // 2, np.arange(n)]

        # Newton refinement (skip clamped endpoints on open curves)
        mask = np.ones(n, dtype=bool)
        if not periodic:
            mask[0] = False
            mask[-1] = False
        u = self._newton_refine_iso(
            greville.copy(),
            queries,
            mask,
            u_min,
            u_max,
            max_param,
            periodic,
            iso_value,
            iso_axis,
        )

        # Convert to user space and close periodic loop
        return self._finalize_cp_params(
            u, n, max_param, periodic, uniform, registered, denormalize
        )

    def _cp_params_degree_one(self, n, max_param, periodic, uniform, denormalize):
        """Closed-form CP params for degree <= 1."""
        if periodic:
            u_native = np.arange(n, dtype=np.float64)
            u_user   = denormalize(u_native % max_param)
            period   = 1.0 if uniform else float(max_param)
            for i in range(1, n):
                if u_user[i] < u_user[i - 1]:
                    u_user[i:] += period
                    break
            return np.append(u_user, u_user[0] + period)
        return denormalize(np.linspace(0, float(max_param), n, dtype=np.float64))

    def _cp_greville_bounds(self, n, d, kv, max_param, periodic):
        """Greville abscissae and non-overlapping Newton search bounds."""
        cumsum   = np.concatenate([[0], np.cumsum(kv)])
        greville = (cumsum[d + 1 : d + 1 + n] - cumsum[1 : 1 + n]) / d
        if not periodic:
            greville[0] = 0.0
            greville[-1] = float(max_param)

        mids  = (greville[:-1] + greville[1:]) / 2.0
        u_min = np.empty(n, dtype=np.float64)
        u_max = np.empty(n, dtype=np.float64)
        if periodic:
            half_step_start = (greville[1] - greville[0]) / 2.0
            half_step_end   = (greville[-1] - greville[-2]) / 2.0
            u_min[0] = greville[0] - half_step_start
            u_min[1:] = mids
            u_max[:-1] = mids
            u_max[-1] = greville[-1] + half_step_end
        else:
            u_min[0] = 0.0
            u_min[1:] = mids
            u_max[:-1] = mids
            u_max[-1] = float(max_param)
        return greville, u_min, u_max

    def _newton_refine_iso(
        self,
        u,
        queries,
        mask,
        u_min,
        u_max,
        max_param,
        periodic,
        iso_value,
        iso_axis,
    ):
        """Gauss-Newton refinement of u against an iso-curve."""
        if not mask.any():
            return u
        for _ in range(10):
            u_eval = u % max_param if periodic else u
            if iso_axis == "u":
                iso = np.full_like(u_eval, iso_value)
                C, Cp, _ = _evaluate_bspline_surface_derivatives(
                    u_eval,
                    iso,
                    self._kv_u,
                    self._kv_v,
                    self._cv_f64,
                    self.degree_u,
                    self.degree_v,
                )
            else:
                iso = np.full_like(u_eval, iso_value)
                C, _, Cp = _evaluate_bspline_surface_derivatives(
                    iso,
                    u_eval,
                    self._kv_u,
                    self._kv_v,
                    self._cv_f64,
                    self.degree_u,
                    self.degree_v,
                )
            diff  = C - queries
            f     = np.einsum("ij,ij->i", diff, Cp)
            fp    = np.einsum("ij,ij->i", Cp, Cp)
            fp    = np.where(np.abs(fp) < 1e-14, 1e-14, fp)
            du    = np.where(mask, -f / fp, 0.0)
            du    = np.clip(du, -0.5, 0.5)
            u_new = np.clip(u + du, u_min, u_max)
            if np.max(np.abs(u_new[mask] - u[mask])) < 1e-10:
                return u_new
            u = u_new
        return u

    def _finalize_cp_params(
        self,
        u,
        n,
        max_param,
        periodic,
        uniform,
        registered,
        denormalize,
    ):
        """Denormalize, restore monotonicity, and close periodic loop."""
        if not periodic:
            return denormalize(u)

        u      = u % max_param
        u      = denormalize(u)
        period = 1.0 if uniform else float(max_param)

        # For registered curves, CP 0 sits at the registration boundary.
        if registered and u[0] > period / 2:
            u[0] -= period

        # Restore monotonicity in user space
        for i in range(1, n):
            if u[i] < u[i - 1]:
                u[i:] += period
                break

        return np.append(u, u[0] + period)

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(self, u, v, _native=False):
        """
        Position-only fast path using the existing surface evaluation kernel.

        Parameters
        ----------
        u, v : float or np.ndarray
            Parameter values.
        _native : bool
            If True, skip normalization (params already in native space).

        Returns
        -------
        np.ndarray
            Surface points (n, dims), or (dims,) for scalar input.

        Notes
        -----
        For open (non-periodic) directions, queries outside the natural
        domain are linearly extrapolated along the boundary tangent
        (see ``compute`` for the formula). When extrapolation is needed,
        the call is routed through ``compute`` (which evaluates partial
        derivatives) and tangents are discarded.
        """
        if self._max_param_u is None:
            self._init_bspline()

        scalar = np.isscalar(u) and np.isscalar(v)

        # If any direction is open and there are out-of-range queries,
        # route through compute() which handles linear extrapolation.
        if not _native and (not self.periodic_u or not self.periodic_v):
            u_arr   = np.atleast_1d(np.asarray(u, dtype=np.float64))
            v_arr   = np.atleast_1d(np.asarray(v, dtype=np.float64))
            any_out = False
            if not self.periodic_u:
                u_high  = 1.0 if self.uniform_u else float(self._max_param_u)
                any_out = any_out or bool((u_arr < 0.0).any() or (u_arr > u_high).any())
            if not self.periodic_v:
                v_high  = 1.0 if self.uniform_v else float(self._max_param_v)
                any_out = any_out or bool((v_arr < 0.0).any() or (v_arr > v_high).any())
            if any_out:
                pts, _, _ = self.compute(u, v, _native=_native)
                return pts

        if not _native:
            u = self._normalize_u(u)
            v = self._normalize_v(v)
            if self.periodic_u:
                u = u % self._max_param_u
            if self.periodic_v:
                v = v % self._max_param_v

        u = np.atleast_1d(u).astype(np.float64)
        v = np.atleast_1d(v).astype(np.float64)

        pts = _evaluate_bspline_surface(
            u, v, self._kv_u, self._kv_v, self._cv_f64, self.degree_u, self.degree_v
        )

        if scalar and pts.shape[0] == 1:
            return pts[0]
        return pts

    def compute(self, u, v, _native=False):
        """
        Compute surface points and first partial derivatives.

        When ``use_numba=True`` (default) uses the numba kernel.
        When ``use_numba=False`` and scipy >= 1.13 is available, uses
        ``scipy.interpolate.NdBSpline`` with analytical derivatives via
        ``nu=(1, 0)`` / ``nu=(0, 1)``.

        Parameters
        ----------
        u, v : float or np.ndarray
            Parameter values.
        _native : bool
            If True, skip normalization (params already in native space)
            AND skip linear extrapolation for out-of-domain queries.

        Returns
        -------
        tuple
            (points, tangent_u, tangent_v) arrays, each (n, dims),
            or (dims,) shaped for scalar input.

        Notes
        -----
        For open (non-periodic) directions, queries outside the natural
        domain are linearly extrapolated along the boundary tangent:

          - Per direction, the natural domain is [0, 1] for ``uniform=True``
            and [0, max_param] for ``uniform=False``.
          - For each query (u_q, v_q), let (u_c, v_c) be that query clamped
            to the natural domain. Position is::

              S_extrap = S(u_c, v_c)
                       + delta_u * vel_u(S_u(u_c, v_c))
                       + delta_v * vel_v(S_v(u_c, v_c))

            where ``delta_u = u_q - u_bound`` (zero when in-range) and
            ``vel_u`` is ``total_length_u * unit_S_u`` for ``uniform_u=True``
            or simply ``S_u`` for ``uniform_u=False`` (analogous for v).
          - Tangents in the extrapolated region are held constant at the
            values evaluated at the clamped boundary point.
          - When both u and v are out of range, the corner extrapolation is
            the linear sum of both deltas (bilinear-like).
        """
        if self._max_param_u is None:
            self._init_bspline()

        scalar = np.isscalar(u) and np.isscalar(v)

        # For open directions, capture original user-space values so we can
        # replace clamped boundary outputs with linear extrapolation below.
        extrapolate_u = (not self.periodic_u) and not _native
        extrapolate_v = (not self.periodic_v) and not _native
        extrapolate   = extrapolate_u or extrapolate_v
        if extrapolate:
            u_user_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
            v_user_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))

        if not _native:
            u = self._normalize_u(u)
            v = self._normalize_v(v)
            # For non-uniform open directions, clip native params to the valid
            # knot range so the evaluator is well-defined. (Uniform mode's
            # _normalize_u already clamps via np.interp.) Out-of-range entries
            # are overwritten by linear extrapolation below.
            if extrapolate_u and not self.uniform_u:
                u = np.clip(u, 0.0, self._max_param_u)
            if extrapolate_v and not self.uniform_v:
                v = np.clip(v, 0.0, self._max_param_v)
            if self.periodic_u:
                u = u % self._max_param_u
            if self.periodic_v:
                v = v % self._max_param_v

        u = np.atleast_1d(u).astype(np.float64)
        v = np.atleast_1d(v).astype(np.float64)

        if self.use_numba:
            pts, tu, tv = _evaluate_bspline_surface_derivatives(
                u,
                v,
                self._kv_u,
                self._kv_v,
                self._cv_f64,
                self.degree_u,
                self.degree_v,
            )
        elif self._spl is not None:
            # Real scipy fallback via NdBSpline analytical derivatives
            xi  = np.column_stack([u, v])
            pts = np.asarray(self._spl(xi))
            tu  = np.asarray(self._spl(xi, nu=(1, 0)))
            tv  = np.asarray(self._spl(xi, nu=(0, 1)))
        else:
            # No scipy NdBSpline available; fall back to numba kernel
            pts, tu, tv = _evaluate_bspline_surface_derivatives(
                u,
                v,
                self._kv_u,
                self._kv_v,
                self._cv_f64,
                self.degree_u,
                self.degree_v,
            )

        if extrapolate:
            pts, tu, tv = self._apply_open_extrapolation(
                u_user_arr, v_user_arr, pts, tu, tv
            )

        if scalar and pts.shape[0] == 1:
            return pts[0], tu[0], tv[0]
        return pts, tu, tv

    def compute_fast(self, u, v, _native=False):
        """
        Always-numba positions + tangents. Mirrors ``BSplineData.compute_fast``.

        Parameters
        ----------
        u, v : float or np.ndarray
            Parameter values.
        _native : bool
            If True, skip normalization AND skip linear extrapolation.

        Returns
        -------
        tuple
            (points, tangent_u, tangent_v).

        Notes
        -----
        Out-of-domain queries on open directions are linearly extrapolated;
        see ``compute`` for the formula.
        """
        if self._max_param_u is None:
            self._init_bspline()

        scalar        = np.isscalar(u) and np.isscalar(v)

        extrapolate_u = (not self.periodic_u) and not _native
        extrapolate_v = (not self.periodic_v) and not _native
        extrapolate   = extrapolate_u or extrapolate_v
        if extrapolate:
            u_user_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
            v_user_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))

        if not _native:
            u = self._normalize_u(u)
            v = self._normalize_v(v)
            if extrapolate_u and not self.uniform_u:
                u = np.clip(u, 0.0, self._max_param_u)
            if extrapolate_v and not self.uniform_v:
                v = np.clip(v, 0.0, self._max_param_v)
            if self.periodic_u:
                u = u % self._max_param_u
            if self.periodic_v:
                v = v % self._max_param_v

        u = np.atleast_1d(u).astype(np.float64)
        v = np.atleast_1d(v).astype(np.float64)

        pts, tu, tv = _evaluate_bspline_surface_derivatives(
            u, v, self._kv_u, self._kv_v, self._cv_f64, self.degree_u, self.degree_v
        )

        if extrapolate:
            pts, tu, tv = self._apply_open_extrapolation(
                u_user_arr, v_user_arr, pts, tu, tv
            )

        if scalar and pts.shape[0] == 1:
            return pts[0], tu[0], tv[0]
        return pts, tu, tv

    def _apply_open_extrapolation(self, u_user, v_user, points, tangents_u, tangents_v):
        """
        Replace clamped boundary values with linear extrapolation for open
        surface directions when (u_user, v_user) is outside the natural
        domain.

        Caller contract: ``points``, ``tangents_u``, ``tangents_v`` were
        evaluated at the clamped (u, v) pair, so they already hold the
        boundary surface values for any out-of-range query. This method
        adds the linear extrapolation offset to ``points``; tangents stay
        as the boundary values (held constant during extrapolation).

        See ``compute`` for the position formula.

        Parameters
        ----------
        u_user, v_user : np.ndarray
            Original 1D user-space parameter values (before normalization).
        points : np.ndarray
            Surface positions, (n, dims).
        tangents_u, tangents_v : np.ndarray
            Native partial derivatives dS/du, dS/dv, each (n, dims).

        Returns
        -------
        tuple
            (points, tangents_u, tangents_v) with extrapolated positions
            for out-of-range entries.
        """
        open_u = not self.periodic_u
        open_v = not self.periodic_v
        if not (open_u or open_v):
            return points, tangents_u, tangents_v

        u_high = (
            (1.0 if self.uniform_u else float(self._max_param_u)) if open_u else None
        )
        v_high = (
            (1.0 if self.uniform_v else float(self._max_param_v)) if open_v else None
        )

        # Per-query domain breach indicators
        if open_u:
            u_below = u_user < 0.0
            u_above = u_user > u_high
            u_out   = u_below | u_above
        else:
            u_out = np.zeros(u_user.shape, dtype=bool)

        if open_v:
            v_below = v_user < 0.0
            v_above = v_user > v_high
            v_out   = v_below | v_above
        else:
            v_out = np.zeros(v_user.shape, dtype=bool)

        out_any = u_out | v_out
        if not out_any.any():
            return points, tangents_u, tangents_v

        # User-space deltas past the boundary (zero when in-range).
        delta_u = np.zeros_like(u_user)
        if open_u:
            delta_u = np.where(u_below, u_user, delta_u)
            delta_u = np.where(u_above, u_user - u_high, delta_u)

        delta_v = np.zeros_like(v_user)
        if open_v:
            delta_v = np.where(v_below, v_user, delta_v)
            delta_v = np.where(v_above, v_user - v_high, delta_v)

        idx = np.where(out_any)[0]

        # Per-query velocity vectors. For uniform mode along an axis, scale
        # by total_length / |tangent| so du=1 covers one full curve length.
        # For non-uniform, the native tangent itself is the velocity.
        if open_u and self.uniform_u:
            tu_idx  = tangents_u[idx]
            tu_norm = np.linalg.norm(tu_idx, axis=1, keepdims=True)
            tu_norm = np.where(tu_norm > 1e-14, tu_norm, 1.0)
            vel_u   = self.total_length_u * (tu_idx / tu_norm)
        elif open_u:
            vel_u = tangents_u[idx]
        else:
            vel_u = None

        if open_v and self.uniform_v:
            tv_idx  = tangents_v[idx]
            tv_norm = np.linalg.norm(tv_idx, axis=1, keepdims=True)
            tv_norm = np.where(tv_norm > 1e-14, tv_norm, 1.0)
            vel_v   = self.total_length_v * (tv_idx / tv_norm)
        elif open_v:
            vel_v = tangents_v[idx]
        else:
            vel_v = None

        if open_u:
            points[idx] = points[idx] + delta_u[idx, None] * vel_u
        if open_v:
            points[idx] = points[idx] + delta_v[idx, None] * vel_v

        # Tangents stay as boundary values -- already correct from the
        # clipped evaluation, no overwrite needed.

        return points, tangents_u, tangents_v

    # ------------------------------------------------------------------ #
    #  Basis                                                              #
    # ------------------------------------------------------------------ #

    def basis(self, u, v, collapse=False, use_numba=None, _native=False):
        """
        Tensor-product basis functions at (u, v).

        Parameters
        ----------
        u, v : float or np.ndarray
            Parameter values.
        collapse : bool
            Fold periodic wrap columns into original control points.
        use_numba : bool or None
            If None, uses self.use_numba. Otherwise overrides.
        _native : bool
            If True, skip normalization (params already native).

        Returns
        -------
        np.ndarray
            Basis values (n, cu * cv) or (n, count_u * count_v) when collapsed.
        """
        if self._max_param_u is None:
            self._init_bspline()

        if not _native:
            u = self._normalize_u(u)
            v = self._normalize_v(v)
            if self.periodic_u:
                u = u % self._max_param_u
            if self.periodic_v:
                v = v % self._max_param_v

        u       = np.atleast_1d(u).astype(np.float64)
        v       = np.atleast_1d(v).astype(np.float64)

        cv_grid = self.cv
        cu      = cv_grid.shape[0]
        cv_n    = cv_grid.shape[1]

        if use_numba is None:
            use_numba = self.use_numba

        if use_numba:
            bu = _compute_basis_parallel(u, self._kv_u, cu, self.degree_u)
            bv = _compute_basis_parallel(v, self._kv_v, cv_n, self.degree_v)
        else:
            bu = compute_basis(u, self._kv_u, cu, self.degree_u)
            bv = compute_basis(v, self._kv_v, cv_n, self.degree_v)

        b = np.einsum("ki,kj->kij", bu, bv).reshape(u.shape[0], cu * cv_n)

        if collapse and (self.periodic_u or self.periodic_v):
            n_u = self._count_u
            n_v = self._count_v

            # Vectorized collapse: build flat destination index map once,
            # then accumulate columns with np.add.at to handle duplicates.
            i_idx     = np.arange(cu)
            j_idx     = np.arange(cv_n)
            i_orig    = (i_idx % n_u) if self.periodic_u else i_idx
            j_orig    = (j_idx % n_v) if self.periodic_v else j_idx
            dst_flat  = (i_orig[:, None] * n_v + j_orig[None, :]).ravel()

            collapsed = np.zeros((u.shape[0], n_u * n_v), dtype=b.dtype)
            np.add.at(collapsed, (slice(None), dst_flat), b)
            return collapsed

        return b

    # ------------------------------------------------------------------ #
    #  Sampling (closest-point projection)                                #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        obj,
        initial_samples  = 20,
        max_newton_iters = 10,
        tolerance        = 1e-10,
    ):
        """
        Project points onto the surface via Newton-Raphson refinement.

        Parameters
        ----------
        obj : array-like or object with .points attribute
            Query points to project.
        initial_samples : int
            Samples per control point per direction for coarse guess.
        max_newton_iters : int
            Maximum Newton iterations per query.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        PatchSampleData
        """
        if self._max_param_u is None:
            self._init_bspline()

        if hasattr(obj, "points"):
            queries = np.asarray(obj.points, dtype=np.float64)
        else:
            queries = np.asarray(obj, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        # --- Coarse sampling ---
        nu_coarse = max(initial_samples * self._count_u, 10)
        nv_coarse = max(initial_samples * self._count_v, 10)
        u_lin = np.linspace(
            0, self._max_param_u, nu_coarse, endpoint=not self.periodic_u
        )
        v_lin = np.linspace(
            0, self._max_param_v, nv_coarse, endpoint=not self.periodic_v
        )
        uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")
        u_flat = uu.ravel().astype(np.float64)
        v_flat = vv.ravel().astype(np.float64)

        pts_coarse = _evaluate_bspline_surface(
            u_flat,
            v_flat,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )

        tree = cKDTree(pts_coarse)
        _, indices = tree.query(queries)

        u_init = u_flat[indices]
        v_init = v_flat[indices]

        # --- Newton refinement ---
        if self.use_numba:
            u_out, v_out, points, tang_u, tang_v = (
                _newton_closest_point_surface_parallel(
                    queries,
                    u_init,
                    v_init,
                    self._kv_u.astype(np.float64),
                    self._kv_v.astype(np.float64),
                    self._cv_f64,
                    self.degree_u,
                    self.degree_v,
                    float(self._max_param_u),
                    float(self._max_param_v),
                    self.periodic_u,
                    self.periodic_v,
                    max_newton_iters,
                    tolerance,
                )
            )
        else:
            u_out, v_out, points, tang_u, tang_v = self._newton_fallback(
                queries, u_init, v_init, max_newton_iters, tolerance
            )

        distances = np.linalg.norm(queries - points, axis=1)
        tangents  = np.stack([tang_u, tang_v], axis=1)

        # Convert to user params
        params_u = self._denormalize_u(u_out)
        params_v = self._denormalize_v(v_out)
        params   = np.stack([params_u, params_v], axis=1)

        # Basis at native params
        b = self.basis(u_out, v_out, collapse=True, _native=True)

        return PatchSampleData(
            points    = points,
            tangents  = tangents,
            distances = distances,
            params    = params,
            basis     = b,
        )

    def _newton_fallback(self, queries, u_init, v_init, max_iters, tolerance):
        """
        Non-numba Newton-Raphson refinement using scipy.interpolate.NdBSpline
        for analytical position and partial derivative evaluation.

        If scipy NdBSpline is unavailable (older scipy), falls back to the
        numba surface kernel (which would be the same as use_numba=True for
        this step). Either way, the algorithm itself runs in pure numpy.
        """
        u = u_init.copy()
        v = v_init.copy()

        def _eval(uu, vv):
            if self._spl is not None:
                xi = np.column_stack([uu, vv])
                return (
                    np.asarray(self._spl(xi)),
                    np.asarray(self._spl(xi, nu=(1, 0))),
                    np.asarray(self._spl(xi, nu=(0, 1))),
                )
            return _evaluate_bspline_surface_derivatives(
                uu,
                vv,
                self._kv_u,
                self._kv_v,
                self._cv_f64,
                self.degree_u,
                self.degree_v,
            )

        for _ in range(max_iters):
            pts, su, sv = _eval(u, v)

            diff  = pts - queries
            f1    = np.einsum("ij,ij->i", diff, su)
            f2    = np.einsum("ij,ij->i", diff, sv)

            J11   = np.einsum("ij,ij->i", su,   su)
            J12   = np.einsum("ij,ij->i", su,   sv)
            J22   = np.einsum("ij,ij->i", sv,   sv)

            det   = J11 * J22 - J12 * J12
            det   = np.where(np.abs(det) < 1e-14, 1e-14, det)

            du    = (-f1 * J22 + f2 * J12) / det
            dv    = (f1 * J12 - f2 * J11) / det
            du    = np.clip(du, -0.5, 0.5)
            dv    = np.clip(dv, -0.5, 0.5)

            u_new = u + du
            v_new = v + dv

            if self.periodic_u:
                u_new = u_new % self._max_param_u
            else:
                u_new = np.clip(u_new, 0, self._max_param_u)
            if self.periodic_v:
                v_new = v_new % self._max_param_v
            else:
                v_new = np.clip(v_new, 0, self._max_param_v)

            if max(np.max(np.abs(du)), np.max(np.abs(dv))) < tolerance:
                u, v = u_new, v_new
                break
            u, v = u_new, v_new

        pts, tang_u, tang_v = _eval(u, v)
        return u, v, pts, tang_u, tang_v

    # Backwards-compat alias
    _newton_scipy = _newton_fallback

    # ------------------------------------------------------------------ #
    #  Raycast                                                            #
    # ------------------------------------------------------------------ #

    def raycast(
        self,
        origins,
        directions              = None,
        forward_only:     bool  = True,
        twosided:         bool  = True,
        initial_samples:  int   = 20,
        max_newton_iters: int   = 20,
        tolerance:        float = 1e-10,
        hit_eps:          float = 1e-4,
    ):
        """
        Ray-surface intersection on this B-spline patch.

        Parameters
        ----------
        origins : array-like or object with ``.points``
            Ray origins (n, 3). If the object has a ``.points`` attribute
            (e.g. ``MeshData``) those are used as origins.
        directions : array-like or None
            Ray directions (n, 3) or a single (3,) tiled to match origins.
            If None and ``origins`` exposes ``.get_vertex_normals()``
            (e.g. ``MeshData``), the origin's vertex normals are used.
            Directions are normalized internally so ``distances`` is the
            true Euclidean distance to the hit.
        forward_only : bool
            If True (default), only forward hits (t >= 0) are kept.
            If False, also casts backward and keeps the closer hit per ray.
        twosided : bool
            If False, back-face hits (where the surface normal faces the
            same direction as the ray) are rejected.
        initial_samples : int
            Coarse grid samples per control point per direction used to
            seed Newton refinement. Total grid is roughly
            ``initial_samples * count_u`` by ``initial_samples * count_v``.
        max_newton_iters : int
            Maximum Newton iterations per ray.
        tolerance : float
            Convergence tolerance: stop Newton when ``||F||^2 < tolerance^2``
            where ``F = (S(u, v) - origin) x direction``.
        hit_eps : float
            Maximum allowed residual ``||F||`` at convergence to count as
            a real intersection. Larger values are more permissive about
            grazing rays; smaller values reject near-misses.

        Returns
        -------
        PatchRaycastData

        Notes
        -----
        Requires a 3D control-point grid (``points.shape[2] == 3``).
        Algorithm matches the bilinear / Bezier mesh raycast pattern in
        ``MeshData.raycast``: precompute coarse samples, find best initial
        (u, v) per ray, then Gauss-Newton refine.
        """
        if self._max_param_u is None:
            self._init_bspline()

        if self.points.shape[-1] != 3:
            raise ValueError(
                "raycast requires a 3D surface (control points must have 3 dims)"
            )

        # --- resolve origins ---
        if hasattr(origins, "points"):
            o = np.asarray(origins.points, dtype=np.float64)
        else:
            o = np.asarray(origins, dtype=np.float64)
        if o.ndim == 1:
            o = o.reshape(1, -1)
        o = np.ascontiguousarray(o, dtype=np.float64)

        # --- resolve directions ---
        if directions is None and hasattr(origins, "get_vertex_normals"):
            d = np.asarray(origins.get_vertex_normals(), dtype=np.float64)
        elif hasattr(directions, "get_vertex_normals"):
            d = np.asarray(directions.get_vertex_normals(), dtype=np.float64)
        elif directions is None:
            raise ValueError(
                "directions=None requires `origins` to expose .get_vertex_normals()"
            )
        else:
            d = np.asarray(directions, dtype=np.float64)
            if d.ndim == 1:
                d = np.tile(d, (o.shape[0], 1))
        d = np.ascontiguousarray(d, dtype=np.float64)

        if o.shape != d.shape:
            raise ValueError(
                f"origins shape {o.shape} does not match directions shape {d.shape}"
            )
        if o.shape[1] != 3:
            raise ValueError(
                f"raycast requires 3D origins/directions, got shape {o.shape}"
            )

        # --- normalize directions so t is Euclidean distance ---
        d_norms = np.linalg.norm(d, axis=1, keepdims=True)
        d_norms = np.maximum(d_norms, 1e-14)
        d       = d / d_norms

        # --- build coarse sample grid for initial guess ---
        nu_coarse = max(initial_samples * self._count_u, 10)
        nv_coarse = max(initial_samples * self._count_v, 10)
        u_lin = np.linspace(
            0, self._max_param_u, nu_coarse, endpoint=not self.periodic_u
        )
        v_lin = np.linspace(
            0, self._max_param_v, nv_coarse, endpoint=not self.periodic_v
        )
        uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")
        grid_u = np.ascontiguousarray(uu.ravel(), dtype=np.float64)
        grid_v = np.ascontiguousarray(vv.ravel(), dtype=np.float64)
        grid_pts = _evaluate_bspline_surface(
            grid_u,
            grid_v,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )
        grid_pts = np.ascontiguousarray(grid_pts, dtype=np.float64)

        # --- forward cast ---
        u_out, v_out, t_out, hit_out = _raycast_bspline_surface_parallel(
            o,
            d,
            grid_u,
            grid_v,
            grid_pts,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
            float(self._max_param_u),
            float(self._max_param_v),
            self.periodic_u,
            self.periodic_v,
            forward_only,
            twosided,
            max_newton_iters,
            tolerance,
            hit_eps,
        )

        # --- optional backward cast ---
        if not forward_only:
            u_bwd, v_bwd, t_bwd, hit_bwd = _raycast_bspline_surface_parallel(
                o,
                -d,
                grid_u,
                grid_v,
                grid_pts,
                self._kv_u,
                self._kv_v,
                self._cv_f64,
                self.degree_u,
                self.degree_v,
                float(self._max_param_u),
                float(self._max_param_v),
                self.periodic_u,
                self.periodic_v,
                True,  # backward cast itself is "forward" along -d
                twosided,
                max_newton_iters,
                tolerance,
                hit_eps,
            )
            # Convert backward t back to forward-direction frame
            t_bwd_signed = -t_bwd
            # Prefer backward when fwd missed, or backward is closer in |t|
            t_fwd_abs  = np.where(hit_out, np.abs(t_out), np.inf)
            t_bwd_abs  = np.where(hit_bwd, np.abs(t_bwd_signed), np.inf)
            prefer_bwd = hit_bwd & (t_bwd_abs < t_fwd_abs)
            u_out      = np.where(prefer_bwd, u_bwd,        u_out)
            v_out      = np.where(prefer_bwd, v_bwd,        v_out)
            t_out      = np.where(prefer_bwd, t_bwd_signed, t_out)
            hit_out    = hit_out | hit_bwd

        # --- evaluate geometry at hit (u, v) ---
        # For misses, params are clamped to (0, 0) so the eval is well-defined;
        # the corresponding outputs are then overwritten below.
        safe_u = np.where(hit_out, u_out, 0.0).astype(np.float64)
        safe_v = np.where(hit_out, v_out, 0.0).astype(np.float64)
        points, tang_u, tang_v = _evaluate_bspline_surface_derivatives(
            safe_u,
            safe_v,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )

        # --- normals & occlusion (computed from native tangents) ---
        n      = np.cross(tang_u, tang_v)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n_norm = np.maximum(n_norm, 1e-14)
        n_unit = n / n_norm
        # Following MeshData convention: occluded = (direction * normal) < 0
        # which means the ray hits the front face of the surface.
        dot      = np.einsum("ij,ij->i", d, n_unit)
        occluded = hit_out & (dot < 0.0)

        # --- override miss outputs ---
        # Position falls back to origin so misses are predictable.
        points = np.where(hit_out[:, None], points, o)
        # Distances and params are NaN for misses.
        distances = np.where(hit_out, t_out, np.nan)

        params_u  = self._denormalize_u(safe_u)
        params_v  = self._denormalize_v(safe_v)
        params    = np.stack([params_u, params_v], axis=1)
        params[~hit_out] = np.nan

        tangents = np.stack([tang_u, tang_v], axis=1)

        # --- basis at native (safe) params ---
        b = self.basis(safe_u, safe_v, collapse=True, _native=True)

        return PatchRaycastData(
            points    = points,
            tangents  = tangents,
            distances = distances,
            params    = params,
            basis     = b,
            occluded  = occluded,
            hit       = hit_out,
        )

    # ------------------------------------------------------------------ #
    #  Cache management                                                   #
    # ------------------------------------------------------------------ #

    def invalidate(self) -> None:
        """Invalidate all cached data."""
        self._kv_u               = None
        self._kv_v               = None
        self._geometry_u         = None
        self._geometry_v         = None
        self._count_u            = None
        self._count_v            = None
        self._max_param_u        = None
        self._max_param_v        = None
        self._spl                = None
        self._cv_f64             = None
        self._arc_length_table_u = None
        self._u_table_u          = None
        self._total_length_u     = None
        self._arc_length_table_v = None
        self._u_table_v          = None
        self._total_length_v     = None
        self._cp_params_u        = None
        self._cp_params_v        = None

    def rebuild(self) -> None:
        """Full cache rebuild."""
        self.invalidate()
        self._init_bspline()

    def rebuild_arc_length_table(self) -> None:
        """
        Rebuild cached spline, cv cache, and arc-length tables.

        Always refreshes the cached float64 control vertices and the scipy
        NdBSpline (when available), then rebuilds arc-length tables for any
        direction with uniform parameterization enabled.
        """
        if self._max_param_u is None:
            self._init_bspline()
            return

        # ALWAYS refresh cached control vertices (fixes stale-points issue
        # after direct mutation of self.points).
        self._cv_f64 = self.cv.astype(np.float64)

        # Rebuild scipy NdBSpline if available
        if _NdBSpline is not None:
            self._spl = _NdBSpline(
                (self._kv_u, self._kv_v),
                self._cv_f64,
                (self.degree_u, self.degree_v),
            )

        if self.uniform_u:
            self._build_arc_length_table_u()
        if self.uniform_v:
            self._build_arc_length_table_v()

    # ------------------------------------------------------------------ #
    #  Open / Close                                                       #
    # ------------------------------------------------------------------ #

    def open_u(self):
        """Make U direction non-periodic."""
        if self._max_param_u is None:
            self._init_bspline()
        if self.periodic_u:
            self.periodic_u = False
            self.rebuild()

    def close_u(self):
        """Make U direction periodic."""
        if self._max_param_u is None:
            self._init_bspline()
        if not self.periodic_u:
            self.periodic_u = True
            self.rebuild()

    def open_v(self):
        """Make V direction non-periodic."""
        if self._max_param_u is None:
            self._init_bspline()
        if self.periodic_v:
            self.periodic_v = False
            self.rebuild()

    def close_v(self):
        """Make V direction periodic."""
        if self._max_param_u is None:
            self._init_bspline()
        if not self.periodic_v:
            self.periodic_v = True
            self.rebuild()

    # ------------------------------------------------------------------ #
    #  Smoothing                                                          #
    # ------------------------------------------------------------------ #

    def smooth(self, n_u: int, n_v: int, polyorder: int = 3):
        """
        Separable Savgol smoothing along U then V with periodic wrapping.

        Parameters
        ----------
        n_u, n_v : int
            Half-window widths for U and V directions.
        polyorder : int
            Polynomial order for the filter.
        """
        if self._max_param_u is None:
            self._init_bspline()

        pts = self.points.copy()
        nu, nv, dims = pts.shape

        # --- Smooth along U (axis 0) ---
        if n_u > 0:
            n_u   = min(n_u, nu - (not self.periodic_u))
            win_u = n_u * 2 + 1
            po    = min(polyorder, win_u - 1)

            if self.periodic_u:
                end      = pts[-(np.arange(n_u) % nu + 1)][::-1]
                start    = pts[np.arange(n_u + 1) % nu]
                padded   = np.concatenate((end, pts, start), axis=0)
                smoothed = savgol_filter(padded, win_u, po, axis=0)
                pts      = smoothed[n_u : -(n_u + 1)]
            else:
                pts = savgol_filter(pts, win_u, po, axis=0)

        # --- Smooth along V (axis 1) ---
        if n_v > 0:
            n_v   = min(n_v, nv - (not self.periodic_v))
            win_v = n_v * 2 + 1
            po    = min(polyorder, win_v - 1)

            if self.periodic_v:
                end      = pts[:, -(np.arange(n_v) % nv + 1)][:, ::-1]
                start    = pts[:, np.arange(n_v + 1) % nv]
                padded   = np.concatenate((end, pts, start), axis=1)
                smoothed = savgol_filter(padded, win_v, po, axis=1)
                pts      = smoothed[:, n_v : -(n_v + 1)]
            else:
                pts = savgol_filter(pts, win_v, po, axis=1)

        self.points = pts

    # ------------------------------------------------------------------ #
    #  Length & Area                                                      #
    # ------------------------------------------------------------------ #

    def get_length_u(self, fast: bool = True, samples: int = 100) -> float:
        """
        Approximate arc length along U using a mid-V iso-curve.

        Parameters
        ----------
        fast : bool
            If True (default), uses fast sampled segment-sum approximation.
            If False, uses scipy.integrate.quad on the iso-curve speed
            (slower but more accurate).
        samples : int
            Samples per control point in fast mode.

        Returns
        -------
        float
            Approximate length in U.
        """
        if self._max_param_u is None:
            self._init_bspline()

        if self._total_length_u is not None:
            return self._total_length_u

        v_mid = self._max_param_v / 2.0

        if not fast:
            import scipy.integrate as _si

            def _speed(uu):
                u_arr = np.atleast_1d(uu).astype(np.float64)
                v_arr = np.full_like(u_arr, v_mid)
                _, su, _ = _evaluate_bspline_surface_derivatives(
                    u_arr,
                    v_arr,
                    self._kv_u,
                    self._kv_v,
                    self._cv_f64,
                    self.degree_u,
                    self.degree_v,
                )
                return float(np.linalg.norm(su[0]))

            return float(_si.quad(_speed, 0, self._max_param_u)[0])

        n     = (self._count_u + self.periodic_u) * samples
        u_lin = np.linspace(0, self._max_param_u, n, endpoint=not self.periodic_u)
        v_arr = np.full(n, v_mid, dtype=np.float64)
        if self.periodic_u:
            u_lin = u_lin % self._max_param_u
        pts = _evaluate_bspline_surface(
            u_lin.astype(np.float64),
            v_arr,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )
        if self.periodic_u:
            pts = np.concatenate([pts, pts[:1]], axis=0)
        d = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))

    def get_length_v(self, fast: bool = True, samples: int = 100) -> float:
        """
        Approximate arc length along V using a mid-U iso-curve.

        Parameters
        ----------
        fast : bool
            If True (default), uses fast sampled segment-sum approximation.
            If False, uses scipy.integrate.quad on the iso-curve speed
            (slower but more accurate).
        samples : int
            Samples per control point in fast mode.

        Returns
        -------
        float
            Approximate length in V.
        """
        if self._max_param_u is None:
            self._init_bspline()

        if self._total_length_v is not None:
            return self._total_length_v

        u_mid = self._max_param_u / 2.0

        if not fast:
            import scipy.integrate as _si

            def _speed(vv):
                v_arr = np.atleast_1d(vv).astype(np.float64)
                u_arr = np.full_like(v_arr, u_mid)
                _, _, sv = _evaluate_bspline_surface_derivatives(
                    u_arr,
                    v_arr,
                    self._kv_u,
                    self._kv_v,
                    self._cv_f64,
                    self.degree_u,
                    self.degree_v,
                )
                return float(np.linalg.norm(sv[0]))

            return float(_si.quad(_speed, 0, self._max_param_v)[0])

        n     = (self._count_v + self.periodic_v) * samples
        v_lin = np.linspace(0, self._max_param_v, n, endpoint=not self.periodic_v)
        u_arr = np.full(n, u_mid, dtype=np.float64)
        if self.periodic_v:
            v_lin = v_lin % self._max_param_v
        pts = _evaluate_bspline_surface(
            u_arr,
            v_lin.astype(np.float64),
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )
        if self.periodic_v:
            pts = np.concatenate([pts, pts[:1]], axis=0)
        d = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))

    def get_area(self, samples: int = 50) -> float:
        """
        Numerical surface area via composite Simpson's rule on
        ``||S_u x S_v||`` over the parameter rectangle.

        Falls back to a Riemann sum when ``samples`` is even (Simpson
        requires an odd number of points per direction).

        Parameters
        ----------
        samples : int
            Samples per direction. Use an odd number for Simpson's rule.

        Returns
        -------
        float
            Approximate surface area.
        """
        if self._max_param_u is None:
            self._init_bspline()

        u_lin = np.linspace(0, self._max_param_u, samples, endpoint=not self.periodic_u)
        v_lin = np.linspace(0, self._max_param_v, samples, endpoint=not self.periodic_v)
        uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")
        u_flat = uu.ravel().astype(np.float64)
        v_flat = vv.ravel().astype(np.float64)

        _, su, sv = _evaluate_bspline_surface_derivatives(
            u_flat,
            v_flat,
            self._kv_u,
            self._kv_v,
            self._cv_f64,
            self.degree_u,
            self.degree_v,
        )

        cross = np.cross(su, sv)
        norms = np.linalg.norm(cross, axis=1).reshape(samples, samples)

        du    = self._max_param_u / (samples - (not self.periodic_u))
        dv    = self._max_param_v / (samples - (not self.periodic_v))

        # Composite Simpson's rule (1D weights then outer product) when
        # samples is odd; else fall back to Riemann sum.
        if samples >= 3 and samples % 2 == 1:
            w_u = np.ones(samples, dtype=np.float64)
            w_u[1:-1:2] = 4.0
            w_u[2:-1:2] = 2.0
            w_v = np.ones(samples, dtype=np.float64)
            w_v[1:-1:2] = 4.0
            w_v[2:-1:2] = 2.0
            weights = np.outer(w_u, w_v)
            return float(np.sum(weights * norms) * du * dv / 9.0)

        return float(np.sum(norms) * du * dv)