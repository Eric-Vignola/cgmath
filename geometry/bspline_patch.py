from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from cgmath.data import Array, Data, DataList
from cgmath.geometry.bspline import (
    CurveSampleData,
    _checked_weights,
    _knot_vector,
    _rational,
    _same_weights,
)
from cgmath.geometry.utils import compute_basis
from cgmath.hierarchy import ShapeData
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
class PatchSampleData(CurveSampleData):
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

    # a surface has two of each: the curve's fields, redeclared in place
    tangents: Array(np.float64, "N", 2, "D")
    params:   Array(np.float64, "N", 2)

    @property
    def normals(self):
        """
        Surface normals via cross(dS/du, dS/dv), normalized; None unless the
        patch is 3-D (a 2-D patch has no normal).
        """
        if self.tangents.shape[-1] != 3:
            return None
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
    points : np.ndarray, (n, 3)
        Intersection points (origins for misses).
    tangents : np.ndarray, (n, 2, 3)
        ``tangents[:, 0]`` = dS/du, ``tangents[:, 1]`` = dS/dv at hits.
    distances : np.ndarray, (n,)
        Ray ``t`` values at hits, NaN for misses.
    params : np.ndarray, (n, 2)
        ``params[:, 0]`` = u, ``params[:, 1]`` = v at hits (user space:
        each direction's [min_param, max_param], or [0, 1] when uniform).
    basis : np.ndarray, (n, count_u * count_v)
        Flattened tensor-product basis values at hits.
    occluded : np.ndarray, (n,)
        Boolean mask: ray direction faces into the surface (front-face hit).
    hit : np.ndarray, (n,)
        Boolean mask -- True where the ray intersected the surface.
    """

    points:    Array(np.float64, "N", 3)
    tangents:  Array(np.float64, "N", 2, 3)
    distances: Array(np.float64, "N")
    params:    Array(np.float64, "N", 2)
    basis:     Array(np.float64, "N", "C")
    occluded:  Array(np.bool_, "N")
    hit:       Array(np.bool_, "N")

    @property
    def normals(self):
        """
        Surface normals via cross(dS/du, dS/dv), normalized; None unless the
        patch is 3-D (a 2-D patch has no normal).
        """
        if self.tangents.shape[-1] != 3:
            return None
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


@dataclass
class NurbsSurfaceData(ShapeData):
    """
    B-spline tensor-product surface.

    Parameters
    ----------
    points : np.ndarray
        Control point grid (nu, nv, dims), float64; dims is free (3 in
        practice, and raycast() needs 3).
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
    knots_u, knots_v : np.ndarray, optional
        Knot vector per direction (K,), float64, in Maya's layout, the one
        ``kv_u`` / ``kv_v`` return (``MFnNurbsSurface.knotsInU()`` /
        ``knotsInV()``): count + degree - 1 values on an open direction,
        count + 2 * degree - 1 on a periodic one, count without the
        periodic wrap. None (default) gives uniform knots on [0, spans].
        As with ``NurbsCurveData.knots``, their range is the direction's
        parameter range, as in Maya: ``min_param_u`` to ``max_param_u``, so
        a (u, v) names the same point here and on a Maya surface built from
        ``cv``, ``kv_u`` and ``kv_v``, and tangents are derivatives with
        respect to it. A periodic direction rebuilds the knots outside that
        range from the ones inside it, so the seam stays smooth. No value
        may repeat more than the direction's degree times, and a degree-1
        direction's knots are its rows' parameters, so they must increase.
        A value repeated ``degree`` times in a direction of degree 2 or
        more creases the patch there (it is only C0). Cleared by
        ``open_u()`` / ``close_u()`` (``open_v()`` / ``close_v()``); call
        ``invalidate()`` after changing them by hand.
    weights : np.ndarray, optional
        One weight per control point (nu, nv), float64, finite and above
        zero: the surface is then rational, each control point pulling it
        by its weight. None (default) is every weight 1. ``points`` stay
        the control points themselves, not multiplied by their weights.
        ``evaluate()``, ``compute()``, ``basis()``, the lengths and the
        area follow the weights; ``sample()`` and ``raycast()`` refuse a
        surface that holds them. Call ``invalidate()`` after changing them
        by hand.
    """

    points:             Array(np.float64, "Nu", "Nv", "D")
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
    knots_u:            Optional[Array(np.float64, "K")] = None
    knots_v:            Optional[Array(np.float64, "K")] = None
    weights:            Optional[Array(np.float64, "Nu", "Nv")] = None

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
    _hw                 = None  # weighted surface: the control vertices times their weight, then the weight
    _arc_length_table_u = None
    _u_table_u          = None
    _total_length_u     = None
    _arc_length_table_v = None
    _u_table_v          = None
    _total_length_v     = None
    _cp_params_u        = None
    _cp_params_v        = None
    _lo_u               = None  # start of the U parameter range (the knots')
    _hi_u               = None  # end of the U parameter range
    _scale_u            = None  # U parameter units per native unit
    _edges_u            = None  # degree-1 U knots: the parameter span by span
    _offset_u           = None  # U registration offset of periodic patches
    _lo_v               = None  # the same along V
    _hi_v               = None
    _scale_v            = None
    _edges_v            = None
    _offset_v           = None

    # the private state an instance holds beside its fields: the caches above
    _STATE = (
        "_kv_u",
        "_kv_v",
        "_geometry_u",
        "_geometry_v",
        "_count_u",
        "_count_v",
        "_max_param_u",
        "_max_param_v",
        "_spl",
        "_cv_f64",
        "_hw",
        "_arc_length_table_u",
        "_u_table_u",
        "_total_length_u",
        "_arc_length_table_v",
        "_u_table_v",
        "_total_length_v",
        "_cp_params_u",
        "_cp_params_v",
        "_lo_u",
        "_hi_u",
        "_scale_u",
        "_edges_u",
        "_offset_u",
        "_lo_v",
        "_hi_v",
        "_scale_v",
        "_edges_v",
        "_offset_v",
    )

    __repr__ = Data.__repr__

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # The fields first: another grid shape, degree or wrap is another
        # patch, whose control points would not line up with these; so are
        # other knots
        return bool(
            self.points.shape == other.points.shape
            and self.degree_u == other.degree_u
            and self.degree_v == other.degree_v
            and self.periodic_u == other.periodic_u
            and self.periodic_v == other.periodic_v
            and self.uniform_u == other.uniform_u
            and self.uniform_v == other.uniform_v
            and self.registered_u == other.registered_u
            and self.registered_v == other.registered_v
            and np.allclose(self.cv, other.cv)
            and self.kv_u.shape == other.kv_u.shape
            and np.allclose(self.kv_u, other.kv_u)
            and self.kv_v.shape == other.kv_v.shape
            and np.allclose(self.kv_v, other.kv_v)
            and _same_weights(self.weights, other.weights, self.points.shape[:2])
        )

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #

    def _init_bspline(self) -> None:
        """
        Build knot vectors, geometry indices, and optional arc-length tables.

        With ``knots_u`` / ``knots_v`` set, a direction's knot vector comes
        from them instead, rescaled to the same native range [0, spans]; on
        a degree-1 direction it stays unit-spaced and the knots map the
        parameter span by span (``_direction_knots``).
        """
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

        # Custom knots replace the uniform ones. Inside, the patch still
        # runs on [0, spans] each way; the knots' own range is the parameter
        # range users see, mapped in _normalize_* / _denormalize_*
        self._kv_u, self._lo_u, self._hi_u, self._scale_u, self._edges_u = _direction_knots(
            self.knots_u, self._kv_u, self._count_u, self.degree_u, self.periodic_u,
            "NurbsSurfaceData.knots_u",
        )
        self._kv_v, self._lo_v, self._hi_v, self._scale_v, self._edges_v = _direction_knots(
            self.knots_v, self._kv_v, self._count_v, self.degree_v, self.periodic_v,
            "NurbsSurfaceData.knots_v",
        )

        # Registration offsets (periodic): CV 0's Greville abscissa, the
        # average of the knots under its basis function
        if self.knots_u is None:
            self._offset_u = self._max_param_u - (self.degree_u - 1) / 2.0
        else:
            self._offset_u = self._max_param_u + np.mean(self._kv_u[1 : self.degree_u + 1])
        if self.knots_v is None:
            self._offset_v = self._max_param_v - (self.degree_v - 1) / 2.0
        else:
            self._offset_v = self._max_param_v + np.mean(self._kv_v[1 : self.degree_v + 1])

        self._set_spline()

        if self.uniform_u:
            self._build_arc_length_table_u()
        if self.uniform_v:
            self._build_arc_length_table_v()

    def _set_spline(self) -> None:
        """
        Builds the evaluators from the current control points and weights.

        A surface that holds ``weights`` is evaluated on its homogeneous
        control points, each one times its weight with the weight as one
        more coordinate, and divided by that coordinate; it has no scipy
        evaluator.
        """
        # Cache float64 control vertices (with periodic wrap) for hot paths
        self._cv_f64 = self.cv.astype(np.float64)
        self._hw     = None
        self._spl    = None

        if self.weights is not None:
            weights = _checked_weights(self.weights, self.points.shape[:2])
            weights = weights[np.ix_(self._geometry_u, self._geometry_v)][..., None]
            self._hw = np.ascontiguousarray(
                np.concatenate([self._cv_f64 * weights, weights], axis=2)
            )

        # Real scipy fallback via NdBSpline. Only initialized when scipy is
        # new enough to provide it (>= 1.13). Used when use_numba=False.
        elif _NdBSpline is not None:
            self._spl = _NdBSpline(
                (self._kv_u, self._kv_v),
                self._cv_f64,
                (self.degree_u, self.degree_v),
            )

    def _points_at(self, u, v):
        """
        Points at native (u, v) (n,) from the Numba kernel. A weighted
        surface runs it on its homogeneous control points and divides.
        """
        if self._hw is None:
            return _evaluate_bspline_surface(
                u, v, self._kv_u, self._kv_v, self._cv_f64, self.degree_u, self.degree_v
            )
        whole = _evaluate_bspline_surface(
            u, v, self._kv_u, self._kv_v, self._hw, self.degree_u, self.degree_v
        )
        return whole[:, :-1] / whole[:, -1:]

    def _derivs_at(self, u, v):
        """
        Points and both partial derivatives at native (u, v) (n,) from the
        Numba kernel. A weighted surface runs it on its homogeneous control
        points and divides, the derivatives by the quotient rule.
        """
        if self._hw is None:
            return _evaluate_bspline_surface_derivatives(
                u, v, self._kv_u, self._kv_v, self._cv_f64, self.degree_u, self.degree_v
            )
        whole, slope_u, slope_v = _evaluate_bspline_surface_derivatives(
            u, v, self._kv_u, self._kv_v, self._hw, self.degree_u, self.degree_v
        )
        points, tangents_u = _rational(whole, slope_u)
        return points, tangents_u, _rational(whole, slope_v)[1]

    def _refuse_weights(self, what) -> None:
        """Raises on a weighted surface: ``what`` searches one whose weights are all 1."""
        if self.weights is not None:
            raise ValueError(
                f"NurbsSurfaceData.{what}: the surface holds weights, which {what}() "
                "does not search; set weights to None for the surface its control "
                "points give at weight 1"
            )

    def _build_arc_length_table_u(self) -> None:
        """Build arc-length table for U using a mid-V iso-curve."""
        n     = self.arc_length_samples
        v_mid = self._max_param_v / 2.0

        if self.registered_u and self.periodic_u:
            offset = self._offset_u
            self._u_table_u = np.linspace(
                offset, offset + self._max_param_u, n, dtype=np.float64
            )
            eval_u = self._u_table_u % self._max_param_u
        else:
            self._u_table_u = np.linspace(0, self._max_param_u, n, dtype=np.float64)
            eval_u          = self._u_table_u

        eval_v = np.full(n, v_mid, dtype=np.float64)
        pts    = self._points_at(eval_u, eval_v)

        diffs                        = np.diff(pts, axis=0)
        seg_len                      = np.linalg.norm(diffs, axis=1)
        self._arc_length_table_u     = np.zeros(n, dtype=np.float64)
        self._arc_length_table_u[1:] = np.cumsum(seg_len)
        self._total_length_u         = self._arc_length_table_u[-1]
        if self._total_length_u > 0:
            self._arc_length_table_u /= self._total_length_u

    def _build_arc_length_table_v(self) -> None:
        """Build arc-length table for V using a mid-U iso-curve."""
        n     = self.arc_length_samples
        u_mid = self._max_param_u / 2.0

        if self.registered_v and self.periodic_v:
            offset = self._offset_v
            self._u_table_v = np.linspace(
                offset, offset + self._max_param_v, n, dtype=np.float64
            )
            eval_v = self._u_table_v % self._max_param_v
        else:
            self._u_table_v = np.linspace(0, self._max_param_v, n, dtype=np.float64)
            eval_v          = self._u_table_v

        eval_u = np.full(n, u_mid, dtype=np.float64)
        pts    = self._points_at(eval_u, eval_v)

        diffs                        = np.diff(pts, axis=0)
        seg_len                      = np.linalg.norm(diffs, axis=1)
        self._arc_length_table_v     = np.zeros(n, dtype=np.float64)
        self._arc_length_table_v[1:] = np.cumsum(seg_len)
        self._total_length_v         = self._arc_length_table_v[-1]
        if self._total_length_v > 0:
            self._arc_length_table_v /= self._total_length_v

    # ------------------------------------------------------------------ #
    #  Parameter normalization                                            #
    # ------------------------------------------------------------------ #

    def _normalize_u(self, u):
        """
        User -> native parameter space for U. With ``knots_u`` (and not
        ``uniform_u``), maps the knots' range onto the native [0, spans]
        first.
        """
        u = np.asarray(u, dtype=np.float64)
        if self.knots_u is not None and not self.uniform_u:
            u = _to_native(u, self._lo_u, self._scale_u, self._edges_u, self.periodic_u)
        if self.registered_u and self.periodic_u and not self.uniform_u:
            u = u + self._offset_u
        if self.uniform_u:
            u = np.interp(
                u % 1.0 if self.periodic_u else u,
                self._arc_length_table_u,
                self._u_table_u,
            )
        return u

    def _normalize_v(self, v):
        """User -> native parameter space for V (see ``_normalize_u``)."""
        v = np.asarray(v, dtype=np.float64)
        if self.knots_v is not None and not self.uniform_v:
            v = _to_native(v, self._lo_v, self._scale_v, self._edges_v, self.periodic_v)
        if self.registered_v and self.periodic_v and not self.uniform_v:
            v = v + self._offset_v
        if self.uniform_v:
            v = np.interp(
                v % 1.0 if self.periodic_v else v,
                self._arc_length_table_v,
                self._u_table_v,
            )
        return v

    def _denormalize_u(self, u):
        """
        Native -> user parameter space for U. Registered and periodic (not
        ``uniform_u``), the offset comes off and u wraps into the range, as
        on a curve. With ``knots_u`` (and not ``uniform_u``), maps the
        native [0, spans] onto the knots' range last.
        """
        u = np.asarray(u, dtype=np.float64)
        if self.uniform_u:
            if self.registered_u and self.periodic_u:
                u = np.where(u < self._offset_u, u + self._max_param_u, u)
            u = np.interp(u, self._u_table_u, self._arc_length_table_u)
        else:
            if self.registered_u and self.periodic_u:
                # A value a hair below 0 wraps to max_param itself; that is 0
                u = (u - self._offset_u) % self._max_param_u
                u = np.where(u >= self._max_param_u, 0.0, u)
            if self.knots_u is not None:
                u = _to_user(u, self._lo_u, self._scale_u, self._edges_u, self.periodic_u)
        return u

    def _denormalize_v(self, v):
        """Native -> user parameter space for V (see ``_denormalize_u``)."""
        v = np.asarray(v, dtype=np.float64)
        if self.uniform_v:
            if self.registered_v and self.periodic_v:
                v = np.where(v < self._offset_v, v + self._max_param_v, v)
            v = np.interp(v, self._u_table_v, self._arc_length_table_v)
        else:
            if self.registered_v and self.periodic_v:
                v = (v - self._offset_v) % self._max_param_v
                v = np.where(v >= self._max_param_v, 0.0, v)
            if self.knots_v is not None:
                v = _to_user(v, self._lo_v, self._scale_v, self._edges_v, self.periodic_v)
        return v

    def _knot_tangents(self, u, v, tangents_u, tangents_v):
        """
        Partial derivatives at native (u, v), each (n, dims), as derivatives
        with respect to the knots' parameters, as in Maya: the native ones
        over each direction's parameter units per native unit (``_rate``).
        A direction without knots is left as it is.
        """
        if self.knots_u is not None:
            rate       = _rate(u, self._scale_u, self._edges_u)
            tangents_u = tangents_u / np.reshape(rate, (-1, 1))
        if self.knots_v is not None:
            rate       = _rate(v, self._scale_v, self._edges_v)
            tangents_v = tangents_v / np.reshape(rate, (-1, 1))
        return tangents_u, tangents_v

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
        """Unpadded U knot vector.

        Maya's layout (``MFnNurbsSurface.knotsInU()``), the one ``knots_u``
        takes, and with ``knots_u`` set it gives them back in their own range.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self.knots_u is not None:
            return _to_user(self._kv_u[1:-1], self._lo_u, self._scale_u, self._edges_u, self.periodic_u)
        return self._kv_u[1:-1]

    @property
    def kv_v(self) -> np.ndarray:
        """Unpadded V knot vector.

        Maya's layout (``MFnNurbsSurface.knotsInV()``), the one ``knots_v``
        takes, and with ``knots_v`` set it gives them back in their own range.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self.knots_v is not None:
            return _to_user(self._kv_v[1:-1], self._lo_v, self._scale_v, self._edges_v, self.periodic_v)
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
    def max_param_u(self) -> float:
        """
        End of the U parameter range: the end of the knots' range, or the
        span count without ``knots_u``. Ignores ``uniform_u`` and
        ``registered_u``.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self.knots_u is not None:
            return self._hi_u
        return self._max_param_u

    @property
    def max_param_v(self) -> float:
        """
        End of the V parameter range: the end of the knots' range, or the
        span count without ``knots_v``. Ignores ``uniform_v`` and
        ``registered_v``.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self.knots_v is not None:
            return self._hi_v
        return self._max_param_v

    @property
    def min_param_u(self) -> float:
        """
        Start of the U parameter range: the start of the knots' range, or 0
        without ``knots_u``. Ignores ``uniform_u`` and ``registered_u``.
        """
        if self._max_param_u is None:
            self._init_bspline()
        return self._lo_u

    @property
    def min_param_v(self) -> float:
        """
        Start of the V parameter range: the start of the knots' range, or 0
        without ``knots_v``. Ignores ``uniform_v`` and ``registered_v``.
        """
        if self._max_param_u is None:
            self._init_bspline()
        return self._lo_v

    @property
    def domain_u(self) -> tuple:
        """The U parameter range, (min_param_u, max_param_u), like Maya's knotDomainInU."""
        return (self.min_param_u, self.max_param_u)

    @property
    def domain_v(self) -> tuple:
        """The V parameter range, (min_param_v, max_param_v), like Maya's knotDomainInV."""
        return (self.min_param_v, self.max_param_v)

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
        last value is the first plus the period (closing the loop).

        For open directions, the first and last values are clamped at
        min_param and max_param respectively.

        Returns are in user parameter space (denormalized for uniform
        and registered; the knots' range with knots).

        Returns
        -------
        tuple of np.ndarray
            (params_u, params_v).
        """
        if self._cp_params_u is not None and self._cp_params_v is not None:
            return self._cp_params_u, self._cp_params_v

        if self._max_param_u is None:
            self._init_bspline()

        # Each direction's user range, where periodic values start and wrap
        start_u, period_u = (0.0, 1.0) if self.uniform_u else (self._lo_u, self._hi_u - self._lo_u)
        start_v, period_v = (0.0, 1.0) if self.uniform_v else (self._lo_v, self._hi_v - self._lo_v)

        # Compute U params along iso-curve at v=mid
        v_mid = self._max_param_v / 2.0
        self._cp_params_u = self._cp_params_axis(
            n           = self._count_u,
            d           = self.degree_u,
            kv          = self._kv_u,
            max_param   = self._max_param_u,
            periodic    = self.periodic_u,
            start       = start_u,
            period      = period_u,
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
            start       = start_v,
            period      = period_v,
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
        start,
        period,
        registered,
        denormalize,
        iso_value,
        iso_axis,
    ):
        """
        Compute control-point parameters along one direction by Newton
        refinement of Greville abscissae against the representative
        iso-curve at the orthogonal direction's midpoint. start and period:
        where the direction's user parameters start and how far a periodic
        one runs before it wraps.
        """
        # Degree 1: parameters map directly to integer indices (a degree-1
        # direction's native knots are unit-spaced, with knots or without)
        if d <= 1:
            return self._cp_params_degree_one(n, max_param, periodic, period, denormalize)

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
            mask[0]  = False
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
            u, n, max_param, periodic, start, period, registered, denormalize
        )

    def _cp_params_degree_one(self, n, max_param, periodic, period, denormalize):
        """Closed-form CP params for degree <= 1."""
        if periodic:
            u_native = np.arange(n, dtype=np.float64)
            u_user   = denormalize(u_native % max_param)
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

        # Open: the ends on the ends; unclamped knots put the inner ones
        # near an end outside the range, where the patch is not
        if not periodic:
            greville     = np.clip(greville, 0.0, float(max_param))
            greville[0]  = 0.0
            greville[-1] = float(max_param)

        mids  = (greville[:-1] + greville[1:]) / 2.0
        u_min = np.empty(n, dtype=np.float64)
        u_max = np.empty(n, dtype=np.float64)
        if periodic:
            half_step_start = (greville[1] - greville[0]) / 2.0
            half_step_end   = (greville[-1] - greville[-2]) / 2.0
            u_min[0]        = greville[0] - half_step_start
            u_min[1:]       = mids
            u_max[:-1]      = mids
            u_max[-1]       = greville[-1] + half_step_end
        else:
            u_min[0]   = 0.0
            u_min[1:]  = mids
            u_max[:-1] = mids
            u_max[-1]  = float(max_param)
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
                C, Cp, _ = self._derivs_at(u_eval, iso)
            else:
                iso = np.full_like(u_eval, iso_value)
                C, _, Cp = self._derivs_at(iso, u_eval)
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
        start,
        period,
        registered,
        denormalize,
    ):
        """Denormalize, restore monotonicity, and close periodic loop."""
        if not periodic:
            return denormalize(u)

        u = u % max_param
        u = denormalize(u)

        # For registered curves, CP 0 sits at the registration boundary.
        if registered and u[0] - start > period / 2:
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
            Parameter values: each direction's [min_param, max_param]
            (the knots' range, or [0, spans]), or [0, 1] when uniform.
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
                u_low, u_high = (0.0, 1.0) if self.uniform_u else (self._lo_u, self._hi_u)
                any_out = any_out or bool((u_arr < u_low).any() or (u_arr > u_high).any())
            if not self.periodic_v:
                v_low, v_high = (0.0, 1.0) if self.uniform_v else (self._lo_v, self._hi_v)
                any_out = any_out or bool((v_arr < v_low).any() or (v_arr > v_high).any())
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

        u   = np.atleast_1d(u).astype(np.float64)
        v   = np.atleast_1d(v).astype(np.float64)

        pts = self._points_at(u, v)

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
            Parameter values: each direction's [min_param, max_param]
            (the knots' range, or [0, spans]), or [0, 1] when uniform.
        _native : bool
            If True, skip normalization (params already in native space)
            AND skip linear extrapolation for out-of-domain queries.

        Returns
        -------
        tuple
            (points, tangent_u, tangent_v) arrays, each (n, dims),
            or (dims,) shaped for scalar input. The tangents are the
            partial derivatives with respect to the patch's parameters:
            the knots' ones in a direction with knots, which match Maya's.

        Notes
        -----
        For open (non-periodic) directions, queries outside the natural
        domain are linearly extrapolated along the boundary tangent:

          - Per direction, the natural domain is [0, 1] for ``uniform=True``
            and [min_param, max_param] for ``uniform=False``.
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
            pts, tu, tv = self._derivs_at(u, v)
        elif self._spl is not None:
            # Real scipy fallback via NdBSpline analytical derivatives
            xi  = np.column_stack([u, v])
            pts = np.asarray(self._spl(xi))
            tu  = np.asarray(self._spl(xi, nu=(1, 0)))
            tv  = np.asarray(self._spl(xi, nu=(0, 1)))
        else:
            # No scipy NdBSpline available; fall back to numba kernel
            pts, tu, tv = self._derivs_at(u, v)

        # derivatives with respect to the knots' parameters, as in Maya
        tu, tv = self._knot_tangents(u, v, tu, tv)

        if extrapolate:
            pts, tu, tv = self._apply_open_extrapolation(
                u_user_arr, v_user_arr, pts, tu, tv
            )

        if scalar and pts.shape[0] == 1:
            return pts[0], tu[0], tv[0]
        return pts, tu, tv

    def compute_fast(self, u, v, _native=False):
        """
        Always-numba positions + tangents. Mirrors ``NurbsCurveData.compute_fast``.

        Parameters
        ----------
        u, v : float or np.ndarray
            Parameter values: each direction's [min_param, max_param]
            (the knots' range, or [0, spans]), or [0, 1] when uniform.
        _native : bool
            If True, skip normalization AND skip linear extrapolation.

        Returns
        -------
        tuple
            (points, tangent_u, tangent_v), the tangents as ``compute()``
            gives them.

        Notes
        -----
        Out-of-domain queries on open directions are linearly extrapolated;
        see ``compute`` for the formula.
        """
        if self._max_param_u is None:
            self._init_bspline()

        scalar = np.isscalar(u) and np.isscalar(v)

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

        pts, tu, tv = self._derivs_at(u, v)

        # derivatives with respect to the knots' parameters, as in Maya
        tu, tv = self._knot_tangents(u, v, tu, tv)

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
            Partial derivatives dS/du, dS/dv with respect to the patch's
            parameters (``_knot_tangents``), each (n, dims).

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

        # Each open direction's natural domain
        u_low, u_high = (0.0, 1.0) if self.uniform_u else (self._lo_u, self._hi_u)
        v_low, v_high = (0.0, 1.0) if self.uniform_v else (self._lo_v, self._hi_v)

        # Per-query domain breach indicators
        if open_u:
            u_below = u_user < u_low
            u_above = u_user > u_high
            u_out   = u_below | u_above
        else:
            u_out = np.zeros(u_user.shape, dtype=bool)

        if open_v:
            v_below = v_user < v_low
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
            delta_u = np.where(u_below, u_user - u_low, delta_u)
            delta_u = np.where(u_above, u_user - u_high, delta_u)

        delta_v = np.zeros_like(v_user)
        if open_v:
            delta_v = np.where(v_below, v_user - v_low, delta_v)
            delta_v = np.where(v_above, v_user - v_high, delta_v)

        idx = np.where(out_any)[0]

        # Per-query velocity vectors. For uniform mode along an axis, scale
        # by total_length / |tangent| so du=1 covers one full curve length.
        # For non-uniform, the tangent itself is the velocity: it is the
        # derivative with respect to the user parameter already.
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
            Parameter values: each direction's [min_param, max_param]
            (the knots' range, or [0, spans]), or [0, 1] when uniform.
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
            On a surface that holds ``weights`` they are the rational ones:
            each control point's share of the weighted sum.
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

        u = np.atleast_1d(u).astype(np.float64)
        v = np.atleast_1d(v).astype(np.float64)

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

        # a weighted surface: each control point's share of the weighted sum
        if self._hw is not None:
            b = b * self._hw[..., -1].reshape(-1)
            b = b / b.sum(axis=1, keepdims=True)

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
        Project points onto the surface: each query's closest point.

        Each query starts from the 8 samples of a coarse (u, v) grid nearest
        to it, nearest first; a seed within 2.5 grid steps of where an
        earlier search ended is skipped, as it would end there too. From
        each seed, a Newton search that only takes steps that come closer.
        On an open edge, a parameter whose slope points out of the patch
        stays on the edge while the other one slides along it, to the
        closest point of that edge; on a patch open both ways, a corner
        closer than the result starts one more search. A direction along
        which the patch hardly moves (a patch collapsed onto a curve, every
        row the same) is held the same way. A degree-1 direction folds on
        every knot: the search keeps to one span of it at a time, and a knot
        line holds it like an edge unless the next span slopes on down. A
        span of no width (two rows the same) is one more knot line, crossed
        into the closer span beside it that slopes on down; when the other
        comes closer too, one more search starts there. A pole is an open
        edge collapsed to one point, as on a sphere, or nearly, as a small
        hole or a tapered tip: its control points within a tenth of how far
        the next row's lie from them. One that a search reaches is left by
        more searches, from each spot along it where the patch heads toward
        the query most steeply and, on a nearly collapsed one, each spot of
        it closest to the query. The closest result wins, so none is farther
        than its nearest grid sample.

        A knot repeated ``degree`` times in a direction of degree 2 or more
        (``knots_u`` / ``knots_v`` make one) creases the patch there too: the
        search runs on each smooth piece between such creases, a patch of
        its own whose edges hold it as the patch's do, and the closest of
        their results wins.

        Parameters
        ----------
        obj : array-like or object with .points attribute
            Query points to project.
        initial_samples : int
            Samples per control point per direction for the seed grid.
            Where another part of the patch comes nearly as close (a patch
            folded back onto itself, say), all 8 seeds can sit on that part;
            more samples find the closest point.
        max_newton_iters : int
            Maximum Newton steps per search.
        tolerance : float
            Convergence tolerance: a search stops once a step moves its
            native (u, v) less than this.

        Returns
        -------
        PatchSampleData
            params in each direction's [min_param, max_param] (the knots'
            range with knots), or [0, 1] when uniform; tangents the partial
            derivatives with respect to them, as ``compute()`` gives.
        """
        self._refuse_weights("sample")
        if self._max_param_u is None:
            self._init_bspline()

        if hasattr(obj, "points"):
            queries = np.asarray(obj.points, dtype=np.float64)
        else:
            queries = np.asarray(obj, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        # A crease (a knot repeated degree times in a direction of degree 2
        # or more): the same search on each smooth piece instead
        pieces_u = _direction_pieces(self._kv_u, self._count_u, self.degree_u, self.periodic_u)
        pieces_v = _direction_pieces(self._kv_v, self._count_v, self.degree_v, self.periodic_v)
        if pieces_u is not None or pieces_v is not None:
            return self._sample_pieces(
                queries, pieces_u, pieces_v, initial_samples, max_newton_iters, tolerance
            )

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

        pts_coarse = self._points_at(u_flat, v_flat)

        # The 8 nearest samples seed each query, nearest first (found on
        # every core): one sheet of a patch that folds back may hold the
        # nearest sample while the other holds the closest point
        seeds = min(8, pts_coarse.shape[0])
        tree  = cKDTree(pts_coarse)
        _, indices = tree.query(queries, k=seeds, workers=-1)
        indices = indices.reshape(queries.shape[0], seeds)

        u_init  = u_flat[indices]
        v_init  = v_flat[indices]
        reach_u = 2.5 * (u_lin[1] - u_lin[0])
        reach_v = 2.5 * (v_lin[1] - v_lin[0])

        # --- Newton refinement ---
        if self.use_numba or self._spl is None:
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
                    reach_u,
                    reach_v,
                )
            )
        else:
            u_out, v_out, points, tang_u, tang_v = self._newton_fallback(
                queries, u_init, v_init, max_newton_iters, tolerance, reach_u, reach_v
            )

        distances = np.linalg.norm(queries - points, axis=1)

        # derivatives with respect to the knots' parameters, as in Maya
        tang_u, tang_v = self._knot_tangents(u_out, v_out, tang_u, tang_v)
        tangents = np.stack([tang_u, tang_v], axis=1)

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

    def _sample_pieces(
        self, queries, pieces_u, pieces_v, initial_samples, max_newton_iters, tolerance
    ):
        """
        sample() on a creased patch: its search on each smooth piece between
        the creases (``_direction_pieces``), a patch of its own on the
        native parameters, whose open edges hold the search as the patch's
        do. The closest result wins, the first piece's on a tie. A direction
        without a crease is one piece, the whole direction, as it is.

        Parameters
        ----------
        queries : np.ndarray
            Query points (n, dims).
        pieces_u, pieces_v : list or None
            ``_direction_pieces``' per direction, None for no crease.
        initial_samples, max_newton_iters, tolerance
            As sample()'s.

        Returns
        -------
        PatchSampleData
            As sample()'s, with points, tangents and basis from compute()
            and basis() at the params found, the side a crease's tangents
            take included.
        """
        # Per direction, each piece's control point rows, native knots and wrap
        splits = []
        for pieces, count, degree, knots, kv, periodic in (
            (pieces_u, self._count_u, self.degree_u, self.knots_u, self._kv_u, self.periodic_u),
            (pieces_v, self._count_v, self.degree_v, self.knots_v, self._kv_v, self.periodic_v),
        ):
            if pieces is None:
                # its native knots: a degree-1 direction's are unit-spaced
                native = None if knots is None or degree == 1 else kv[1:-1]
                splits.append([(np.arange(count), native, periodic)])
            else:
                splits.append([(rows, piece, False) for rows, piece in pieces])

        n    = queries.shape[0]
        u    = np.zeros(n)
        v    = np.zeros(n)
        best = np.full(n, np.inf)
        for rows_u, knots_u, periodic_u in splits[0]:
            for rows_v, knots_v, periodic_v in splits[1]:
                piece = NurbsSurfaceData(
                    points     = self.points[np.ix_(rows_u, rows_v)],
                    degree_u   = self.degree_u,
                    degree_v   = self.degree_v,
                    periodic_u = periodic_u,
                    periodic_v = periodic_v,
                    use_numba  = self.use_numba,
                    knots_u    = knots_u,
                    knots_v    = knots_v,
                )
                found        = piece.sample(queries, initial_samples, max_newton_iters, tolerance)
                closer       = found.distances < best
                u[closer]    = found.params[closer, 0]
                v[closer]    = found.params[closer, 1]
                best[closer] = found.distances[closer]

        # A periodic direction's last piece runs on through the seam
        if self.periodic_u:
            u = u % self._max_param_u
            u = np.where(u >= self._max_param_u, 0.0, u)
        if self.periodic_v:
            v = v % self._max_param_v
            v = np.where(v >= self._max_param_v, 0.0, v)

        params_u = self._denormalize_u(u)
        params_v = self._denormalize_v(v)
        points, tangents_u, tangents_v = self.compute(params_u, params_v)
        return PatchSampleData(
            points    = points,
            tangents  = np.stack([tangents_u, tangents_v], axis=1),
            distances = np.linalg.norm(queries - points, axis=1),
            params    = np.stack([params_u, params_v], axis=1),
            basis     = self.basis(params_u, params_v, collapse=True),
        )

    def _newton_fallback(
        self, queries, u_init, v_init, max_iters, tolerance, reach_u=0.0, reach_v=0.0
    ):
        """
        sample()'s search without numba: the numba kernel's seeds, steps,
        folds, forks, poles and corners
        (``_newton_closest_point_surface_parallel``, ``_patch_descend``), in
        numpy, with positions and partial derivatives from
        scipy.interpolate.NdBSpline (``_spl``, scipy >= 1.13; sample() uses
        the kernel without it).

        Parameters
        ----------
        queries : np.ndarray
            Query points (n, dims).
        u_init, v_init : np.ndarray
            Native seeds (n, seeds), or (n,) for one each, nearest first.
        max_iters, tolerance : int, float
            As sample()'s max_newton_iters and tolerance.
        reach_u, reach_v : float
            A seed this close to where an earlier search of its query ended
            is skipped.

        Returns
        -------
        tuple
            (u, v, points, tangents_u, tangents_v): native parameters (n,)
            and positions and first partials there (n, dims).
        """
        n      = queries.shape[0]
        u_init = np.asarray(u_init, dtype=np.float64)
        v_init = np.asarray(v_init, dtype=np.float64)
        if u_init.ndim == 1:
            u_init = u_init[:, None]
            v_init = v_init[:, None]
        max_u = float(self._max_param_u)
        max_v = float(self._max_param_v)

        # Seed by seed: one within reach of where an earlier search of its
        # query ended is skipped (it would end there too; along a degree-1
        # direction, only one on the span that search ended on); the
        # closest result wins, the earlier one on a tie. on_poles: the poles
        # each query's searches reached; fork: the first fork among them
        seeds    = u_init.shape[1]
        ends_u   = np.zeros((n, seeds))
        ends_v   = np.zeros((n, seeds))
        on_poles = np.zeros((n, 4), dtype=bool)
        fork     = np.zeros((n, 3))
        for s in range(seeds):
            skip = np.zeros(n, dtype=bool)
            for e in range(s):
                gap_u = np.abs(u_init[:, s] - ends_u[:, e])
                gap_v = np.abs(v_init[:, s] - ends_v[:, e])
                if self.periodic_u:
                    gap_u = np.minimum(gap_u, max_u - gap_u)
                if self.periodic_v:
                    gap_v = np.minimum(gap_v, max_v - gap_v)
                if self.degree_u == 1:
                    span  = np.floor(u_init[:, s])
                    on    = (span <= ends_u[:, e]) & (ends_u[:, e] <= span + 1.0)
                    gap_u = np.where(on, gap_u, np.inf)
                if self.degree_v == 1:
                    span  = np.floor(v_init[:, s])
                    on    = (span <= ends_v[:, e]) & (ends_v[:, e] <= span + 1.0)
                    gap_v = np.where(on, gap_v, np.inf)
                near            = ~skip & (gap_u <= reach_u) & (gap_v <= reach_v)
                ends_u[near, s] = ends_u[near, e]
                ends_v[near, s] = ends_v[near, e]
                skip |= near

            rows = np.flatnonzero(~skip)
            u, v, f, derivs, reached, forks = self._descend_fallback(
                queries[rows], u_init[rows, s], v_init[rows, s], max_iters, tolerance
            )
            on_poles[rows] |= reached
            first             = (fork[rows, 2] == 0.0) & (forks[:, 2] > 0.0)
            fork[rows[first]] = forks[first]
            ends_u[rows, s]   = u
            ends_v[rows, s]   = v
            if s == 0:
                best_u, best_v, best_f, best_derivs = u, v, f, derivs
                continue
            better              = f < best_f[rows]
            hit                 = rows[better]
            best_u[hit]         = u[better]
            best_v[hit]         = v[better]
            best_f[hit]         = f[better]
            best_derivs[:, hit] = derivs[:, better]

        # A corner of a patch open both ways closer than the result: search
        # from there too, corner by corner
        if not (self.periodic_u or self.periodic_v):
            corners = ((0.0, 0.0), (max_u, 0.0), (0.0, max_v), (max_u, max_v))
            for corner_u, corner_v in corners:
                point  = np.asarray(self._spl(np.array([[corner_u, corner_v]])))[0]
                closer = np.flatnonzero(np.sum((queries - point) ** 2, axis=1) < best_f)
                if closer.size == 0:
                    continue
                u, v, f, derivs, reached, forks = self._descend_fallback(
                    queries[closer],
                    np.full(closer.size, corner_u),
                    np.full(closer.size, corner_v),
                    max_iters,
                    tolerance,
                )
                on_poles[closer] |= reached
                first               = (fork[closer, 2] == 0.0) & (forks[:, 2] > 0.0)
                fork[closer[first]] = forks[first]
                better              = f < best_f[closer]
                hit                 = closer[better]
                best_u[hit]         = u[better]
                best_v[hit]         = v[better]
                best_f[hit]         = f[better]
                best_derivs[:, hit] = derivs[:, better]

        # A flat span's two sides can hold two basins: where a search left
        # one with both sides closer, one more search from the other
        rows = np.flatnonzero(fork[:, 2] > 0.0)
        if rows.size:
            u, v, f, derivs, reached, _ = self._descend_fallback(
                queries[rows], fork[rows, 0], fork[rows, 1], max_iters, tolerance
            )
            on_poles[rows] |= reached
            better              = f < best_f[rows]
            hit                 = rows[better]
            best_u[hit]         = u[better]
            best_v[hit]         = v[better]
            best_f[hit]         = f[better]
            best_derivs[:, hit] = derivs[:, better]

        # A search that reached a pole stopped there, slid along it (a
        # nearly collapsed one) or left it where it reached it, so others
        # leave it from each spot ``_pole_turns`` finds, pole by pole; a
        # query's starts are tried in order
        poles = _poles(self._cv_f64, self.periodic_u, self.periodic_v)
        for pole in range(4):
            rows = np.flatnonzero(on_poles[:, pole])
            if rows.size == 0:
                continue
            on_u = pole < 2
            edge = 0.0 if pole % 2 == 0 else (max_u if on_u else max_v)
            at, x = self._pole_turns(queries[rows], edge, on_u, poles[pole] == 1)
            rows = rows[at]
            if rows.size == 0:
                continue

            # Along a degree-1 direction, a start on a knot (two facets
            # meet there) is tried on the span on each side of it, in turn
            facets = (self.degree_v if on_u else self.degree_u) == 1
            twice  = facets & (x == np.floor(x))
            order  = np.repeat(np.arange(rows.size), np.where(twice, 2, 1))
            below  = np.r_[False, order[1:] == order[:-1]]
            rows   = rows[order]
            x      = x[order]
            across = np.full(rows.size, edge)
            u, v, f, derivs, reached, _ = self._descend_fallback(
                queries[rows],
                across if on_u else x,
                x if on_u else across,
                max_iters,
                tolerance,
                off   = True,
                below = below,
            )
            np.logical_or.at(on_poles, rows, reached)
            first = np.r_[True, rows[1:] != rows[:-1]]
            rank = np.arange(rows.size) - np.maximum.accumulate(
                np.where(first, np.arange(rows.size), 0)
            )
            for k in range(int(rank.max(initial=-1)) + 1):
                pick                = np.flatnonzero(rank == k)
                better              = f[pick] < best_f[rows[pick]]
                pick                = pick[better]
                hit                 = rows[pick]
                best_u[hit]         = u[pick]
                best_v[hit]         = v[pick]
                best_f[hit]         = f[pick]
                best_derivs[:, hit] = derivs[:, pick]

        return best_u, best_v, best_derivs[0], best_derivs[1], best_derivs[2]

    # Backwards-compat alias
    _newton_scipy = _newton_fallback

    def _descend_fallback(
        self, queries, u, v, max_iters, tolerance, off=False, below=None
    ):
        """
        ``_patch_descend`` (the numba kernel's search, see there) from every
        start at once, in numpy, evaluated with ``_spl``.

        Parameters
        ----------
        queries : np.ndarray
            Query points (m, dims), one per start.
        u, v : np.ndarray
            Native starts (m,), inside the domain.
        max_iters, tolerance : int, float
            As sample()'s max_newton_iters and tolerance.
        off : bool
            Whether every start is on a pole, to leave it: the first step
            goes straight across it, from the start's span.
        below : np.ndarray, optional
            (m,) bool: whether a start on a knot of a degree-1 direction
            takes the span below it (across the seam from 0 on a periodic
            one) instead of the one above.

        Returns
        -------
        tuple
            (u, v, f, derivs, reached, fork): results (m,), periodic ones in
            [0, end), their squared distances (m,), ``_spl_derivatives``
            there (6, m, dims), the poles (``_poles``) each search reached
            (m, 4) and its fork, (u, v, set) as ``_patch_descend``'s (m, 3).
        """
        max_u    = float(self._max_param_u)
        max_v    = float(self._max_param_v)
        shortest = 1e-15 * max(max_u, max_v, 1.0)
        poles    = np.array(_poles(self._cv_f64, self.periodic_u, self.periodic_v))
        reached  = np.zeros((len(queries), 4), dtype=bool)
        fork     = np.zeros((len(queries), 3))

        # A degree-1 direction keeps to one span (lo, lo + 1) at a time; a
        # direction has ends unless it wraps at degree 2 or more
        span_u = self.degree_u == 1
        span_v = self.degree_v == 1
        ends_u = span_u or not self.periodic_u
        ends_v = span_v or not self.periodic_v

        u      = np.array(u, dtype=np.float64)
        v      = np.array(v, dtype=np.float64)
        lo_u   = np.minimum(np.floor(u), max_u - 1.0) if span_u else None
        lo_v   = np.minimum(np.floor(v), max_v - 1.0) if span_v else None
        if below is not None:
            for x, lo, end, wraps in (
                (u, lo_u, max_u, self.periodic_u),
                (v, lo_v, max_v, self.periodic_v),
            ):
                if lo is None:
                    continue
                next_x, next_lo = _span_exits(x, lo, np.ones_like(x), end, wraps)
                down     = below & (next_lo >= 0.0)
                x[down]  = next_x[down]
                lo[down] = next_lo[down]
        derivs    = self._span_derivatives(u, v, lo_u, lo_v)
        slopes    = _slopes(queries, derivs)
        live      = np.ones(u.shape[0], dtype=bool)
        collapsed = np.zeros(u.shape[0], dtype=bool)

        def _sides(rows):
            return (
                None if lo_u is None else lo_u[rows],
                None if lo_v is None else lo_v[rows],
            )

        for _ in range(max_iters):
            # On a span's end with the slope pointing past it: into the next
            # span when the slope there points on too (u first, then v); not
            # before the first step off a pole
            for axis in (0, 1):
                lo = (lo_u, lo_v)[axis]
                if lo is None or off:
                    continue
                x     = (u, v)[axis]
                end   = (max_u, max_v)[axis]
                wraps = (self.periodic_u, self.periodic_v)[axis]
                next_x, next_lo = _span_exits(x, lo, slopes[1 + axis], end, wraps)
                rows = np.flatnonzero(live & (next_lo >= 0.0))
                if rows.size == 0:
                    continue
                at             = [u[rows], v[rows]]
                sides          = list(_sides(rows))
                at[axis]       = next_x[rows]
                sides[axis]    = next_lo[rows]
                trial          = self._span_derivatives(*at, *sides)
                trial_slopes   = _slopes(queries[rows], trial)
                cross          = trial_slopes[1 + axis] * slopes[1 + axis, rows] > 0.0
                hit            = rows[cross]
                x[hit]         = next_x[hit]
                lo[hit]        = next_lo[hit]
                derivs[:, hit] = trial[:, cross]
                slopes[:, hit] = trial_slopes[:, cross]

            f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv = slopes
            low_u  = lo_u if span_u else np.zeros_like(u)
            low_v  = lo_v if span_v else np.zeros_like(v)
            high_u = lo_u + 1.0 if span_u else np.full_like(u, max_u)
            high_v = lo_v + 1.0 if span_v else np.full_like(v, max_v)
            pin_u  = ends_u & ((u <= low_u) & (g_u > 0.0) | (u >= high_u) & (g_u < 0.0))
            pin_v  = ends_v & ((v <= low_v) & (g_v > 0.0) | (v >= high_v) & (g_v < 0.0))

            # A pole a search reaches is left by more searches
            # (``_newton_fallback``), whose first step goes straight across
            # it. On a pole that is one point, so does every step, and a
            # slope pointing out ends the search. A nearly collapsed one is
            # an edge to slide along
            pole = _pole_at(u, v, poles, max_u, max_v)
            at   = live & (pole >= 0)
            reached[np.flatnonzero(at), pole[at]] = True
            special = at & (off | (poles[pole] == 2))
            out     = np.where(pole < 2, pin_u, pin_v)
            live &= ~(special & out & (not off))
            pin_u = np.where(special, pole >= 2, pin_u)
            pin_v = np.where(special, pole < 2, pin_v)
            off   = False

            # A direction along which the patch hardly moves (as on a patch
            # collapsed onto a curve) is held like a pinned one
            bend_u = np.sum(derivs[3] ** 2, axis=1)
            bend_v = np.sum(derivs[5] ** 2, axis=1)
            flat_u = j_uu + bend_u < 1e-20 * j_vv
            flat_v = j_vv + bend_v < 1e-20 * j_uu

            # Such a span of a degree-1 direction is one knot line: first, a
            # step into the span on either side of it when that comes closer
            # (``_flat_exits``); the other side, when it comes closer too, is
            # the search's fork unless it has one. Once no span along it
            # moves (collapsed), the search stops looking
            left = np.zeros_like(live)
            for axis, flat in ((0, flat_u), (1, flat_v)):
                lo = (lo_u, lo_v)[axis]
                if lo is None:
                    continue
                rows = np.flatnonzero(live & flat & ~special & ~collapsed)
                if rows.size == 0:
                    continue
                exit_x, exit_lo, other_x, other_lo, moves = self._flat_exits(
                    queries[rows], u[rows], v[rows], *_sides(rows), axis == 0, f[rows]
                )
                collapsed[rows]   = ~moves

                first             = (other_lo >= 0.0) & (fork[rows, 2] == 0.0)
                start             = np.ones((rows.size, 3))
                start[:, 0]       = u[rows]
                start[:, 1]       = v[rows]
                start[:, axis]    = other_x
                fork[rows[first]] = start[first]

                x        = (u, v)[axis]
                hit      = exit_lo >= 0.0
                rows     = rows[hit]
                x[rows]  = exit_x[hit]
                lo[rows] = exit_lo[hit]
                derivs[:, rows] = self._span_derivatives(
                    u[rows], v[rows], *_sides(rows)
                )
                slopes[:, rows] = _slopes(queries[rows], derivs[:, rows])
                left[rows]      = True
            pin_u |= flat_u
            pin_v |= flat_v
            live  &= ~(pin_u & pin_v) | left

            moved = np.zeros_like(live)
            new_u = u.copy()
            new_v = v.copy()
            with np.errstate(divide="ignore", invalid="ignore"):
                for kind in range(2):
                    if kind == 1:
                        step_u = np.where(~pin_u & (j_uu > 0.0), -g_u / j_uu, 0.0)
                        step_v = np.where(~pin_v & (j_vv > 0.0), -g_v / j_vv, 0.0)
                    else:
                        # Newton's step, its S_u . S_v part where the
                        # Hessian does not curve up; 1-D on a pinned edge
                        det     = h_uu * h_vv - h_uv * h_uv
                        newton  = (h_uu > 0.0) & (det > 0.0)
                        det_j   = j_uu * j_vv - j_uv * j_uv
                        gauss   = ~newton & (det_j > 1e-12 * j_uu * j_vv)
                        step_u  = np.where(newton, (h_uv * g_v - h_vv * g_u) / det, 0.0)
                        step_v  = np.where(newton, (h_uv * g_u - h_uu * g_v) / det, 0.0)
                        gauss_u = (j_uv * g_v - j_vv * g_u) / det_j
                        gauss_v = (j_uv * g_u - j_uu * g_v) / det_j
                        step_u  = np.where(gauss,         gauss_u,        step_u)
                        step_v  = np.where(gauss,         gauss_v,        step_v)
                        curve_u = np.where(h_uu > 0.0,    h_uu,           j_uu)
                        curve_v = np.where(h_vv > 0.0,    h_vv,           j_vv)
                        step_u  = np.where(pin_u,         0.0,            step_u)
                        along_v = np.where(curve_v > 0.0, -g_v / curve_v, 0.0)
                        along_u = np.where(curve_u > 0.0, -g_u / curve_u, 0.0)
                        step_v  = np.where(pin_u,         along_v,        step_v)
                        step_u  = np.where(pin_v,         along_u,        step_u)
                        step_v  = np.where(pin_v,         0.0,            step_v)

                    # Cut to half a span, then halved until strictly closer;
                    # kept inside each direction's span or domain
                    size   = np.maximum(np.abs(step_u), np.abs(step_v))
                    scale  = np.where(size > 0.5, 0.5 / size, 1.0)
                    step_u = step_u * scale
                    step_v = step_v * scale
                    size   = np.minimum(size, 0.5)
                    todo   = live & ~moved & ~left & (size > shortest)

                    t = np.ones_like(u)
                    while todo.any():
                        idx     = np.flatnonzero(todo)
                        trial_u = u[idx] + t[idx] * step_u[idx]
                        trial_v = v[idx] + t[idx] * step_v[idx]
                        trial_u = _keep_inside(
                            trial_u, low_u[idx], high_u[idx], max_u, ends_u
                        )
                        trial_v = _keep_inside(
                            trial_v, low_v[idx], high_v[idx], max_v, ends_v
                        )
                        trial  = np.column_stack([trial_u, trial_v])
                        points = np.asarray(self._spl(trial))
                        closer = np.sum((points - queries[idx]) ** 2, axis=1) < f[idx]

                        new_u[idx[closer]] = trial_u[closer]
                        new_v[idx[closer]] = trial_v[closer]
                        moved[idx[closer]] = True
                        todo[idx[closer]]  = False
                        t[idx[~closer]] *= 0.5
                        todo            &= t * size > shortest

            live &= moved | left
            idx = np.flatnonzero(live & moved)
            if not live.any():
                break
            if idx.size == 0:
                continue

            # How far each moved, the short way round a periodic direction
            gap_u = np.abs(new_u[idx] - u[idx])
            gap_v = np.abs(new_v[idx] - v[idx])
            if self.periodic_u:
                gap_u = np.minimum(gap_u, max_u - gap_u)
            if self.periodic_v:
                gap_v = np.minimum(gap_v, max_v - gap_v)

            u[idx] = new_u[idx]
            v[idx] = new_v[idx]
            side_u, side_v = _sides(idx)
            derivs[:, idx] = self._span_derivatives(u[idx], v[idx], side_u, side_v)
            slopes[:, idx] = _slopes(queries[idx], derivs[:, idx])
            live[idx] &= np.maximum(gap_u, gap_v) >= tolerance

        # A degree-1 direction: back to the right-sided derivatives compute()
        # gives on a knot, and from the seam's end to 0
        if span_u or span_v:
            if self.periodic_u:
                u = np.where(u >= max_u, 0.0, u)
            if self.periodic_v:
                v = np.where(v >= max_v, 0.0, v)
            derivs = self._spl_derivatives(u, v)
        pole                                  = _pole_at(u, v, poles, max_u, max_v)
        at                                    = pole >= 0
        reached[np.flatnonzero(at), pole[at]] = True

        return u, v, slopes[0], derivs, reached, fork

    def _span_derivatives(self, u, v, lo_u=None, lo_v=None):
        """
        ``_spl_derivatives``, a degree-1 direction's taken inside its span
        (lo, lo + 1) when its lo is given: the patch is linear along it
        there, so its first derivative is the difference of the span's two
        ends and its second is zero. On a knot, those are the one-sided
        derivatives of the span the search is on (scipy's are right-sided).
        """
        if lo_u is None and lo_v is None:
            return self._spl_derivatives(u, v)

        def part(uu, vv, a, b):
            if lo_u is not None and a:
                if a == 2:
                    return np.zeros((len(u), self._cv_f64.shape[2]))
                return part(lo_u + 1.0, vv, 0, b) - part(lo_u, vv, 0, b)
            if lo_v is not None and b:
                if b == 2:
                    return np.zeros((len(u), self._cv_f64.shape[2]))
                return part(uu, lo_v + 1.0, a, 0) - part(uu, lo_v, a, 0)
            return np.asarray(self._spl(np.column_stack([uu, vv]), nu=(a, b)))

        orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        return np.stack([part(u, v, a, b) for a, b in orders])

    def _flat_exits(self, queries, u, v, lo_u, lo_v, on_u, f):
        """
        ``_patch_flat_exit`` for many searches at once: for each, on a flat
        span of the degree-1 direction u (on_u) or v, the closest spot along
        that direction on the first span on either side along which the
        patch moves and that slopes on down away from it, when it comes
        closer than f.

        Parameters
        ----------
        queries : np.ndarray
            Query points (m, dims).
        u, v : np.ndarray
            The searches' spots (m,), native.
        lo_u, lo_v : np.ndarray or None
            The start of each degree-1 direction's span (m,), None for one
            of degree 2 or more.
        on_u : bool
            Whether the flat direction is u (else v).
        f : np.ndarray
            The squared distances to beat (m,).

        Returns
        -------
        tuple
            (x, lo, other_x, other_lo, moves), each (m,): the closer side's
            parameter along the direction and the start of its span (lo -1
            where neither side comes closer), the other side's (other_lo -1
            unless it comes closer too), and whether the patch moves along
            the direction on any span at all.
        """
        axis     = 0 if on_u else 1
        end      = float(self._max_param_u if on_u else self._max_param_v)
        periodic = self.periodic_u if on_u else self.periodic_v
        lo       = lo_u if on_u else lo_v
        best_x   = np.zeros_like(f)
        best_lo  = np.full_like(f, -1.0)
        best_f   = f.copy()
        other_x  = np.zeros_like(f)
        other_lo = np.full_like(f, -1.0)
        moves    = np.zeros(f.shape, dtype=bool)
        for side in range(2):
            # Through the end on this side, span by span, to the first along
            # which the patch moves: the same spot, within the flat spans'
            # width. Its derivatives are its own, one-sided on that end
            g      = np.full_like(f, 1.0 if side == 0 else -1.0)
            x      = lo.copy() if side == 0 else lo + 1.0
            at     = lo.copy()
            found  = np.zeros(f.shape, dtype=bool)
            walk   = np.ones(f.shape, dtype=bool)
            slopes = np.zeros((9,) + f.shape)
            for _ in range(int(end + 0.5) - 1):
                rows = np.flatnonzero(walk)
                next_x, at[rows] = _span_exits(
                    x[rows], at[rows], g[rows], end, periodic
                )
                x[rows]    = next_x
                walk[rows] = at[rows] >= 0.0
                rows       = rows[walk[rows]]
                if rows.size == 0:
                    break
                spot = [u[rows], v[rows]]
                sides = [
                    None if lo_u is None else lo_u[rows],
                    None if lo_v is None else lo_v[rows],
                ]
                spot[axis]      = x[rows]
                sides[axis]     = at[rows]
                derivs          = self._span_derivatives(*spot, *sides)
                slopes[:, rows] = _slopes(queries[rows], derivs)
                along           = slopes[6 + 2 * axis, rows]
                across          = slopes[8 - 2 * axis, rows]
                flat            = along < 1e-20 * across
                found[rows]     = ~flat
                walk[rows]      = flat
                rows            = rows[flat]
                x[rows]         = at[rows] if side == 0 else at[rows] + 1.0
            moves |= found

            # Its slope must point on, away from the flat span; the patch is
            # straight along the direction there, so its closest spot along
            # it is one Newton step, kept inside the span
            slope = slopes[1 + axis]
            along = slopes[6 + 2 * axis]
            rows  = np.flatnonzero(found & (slope * g > 0.0))
            if rows.size == 0:
                continue
            trial_x    = x[rows] - slope[rows] / along[rows]
            trial_x    = np.clip(trial_x, at[rows], at[rows] + 1.0)
            spot       = [u[rows], v[rows]]
            spot[axis] = trial_x
            points     = np.asarray(self._spl(np.column_stack(spot)))
            f_new      = np.sum((points - queries[rows]) ** 2, axis=1)
            closer     = f_new < f[rows]
            rows       = rows[closer]
            trial_x    = trial_x[closer]
            f_new      = f_new[closer]

            # The closer side wins; the other is kept when it comes closer
            # too (the search's fork)
            better         = f_new < best_f[rows]
            won            = rows[better]
            lost           = rows[~better]
            other_x[won]   = best_x[won]
            other_lo[won]  = best_lo[won]
            best_x[won]    = trial_x[better]
            best_lo[won]   = at[won]
            best_f[won]    = f_new[better]
            other_x[lost]  = trial_x[~better]
            other_lo[lost] = at[lost]
        return best_x, best_lo, other_x, other_lo, moves

    def _pole_turns(self, queries, edge, on_u, rim=False):
        """
        ``_patch_pole_turns`` for many queries at once: where along a pole
        searches leave it, the ways out that come closer (4 samples per
        span, then a golden-section search between the neighbours of each
        sample whose slope is negative and lower than theirs) and on a
        nearly collapsed pole the spots of its rim closest to each query.

        Parameters
        ----------
        queries : np.ndarray
            Query points (m, dims).
        edge : float
            The pole's parameter: 0 or its direction's end.
        on_u : bool
            Whether the pole is a u edge (it runs along v) or a v edge.
        rim : bool
            Whether the pole is nearly collapsed, a rim: each sample closer
            to the query than its neighbours starts a search too, after
            those, unless the rim looks like one point from the query.

        Returns
        -------
        tuple
            (at, x): for each start, its query's row (int, in order) and
            its parameter along the pole.
        """
        end      = float(self._max_param_v if on_u else self._max_param_u)
        periodic = self.periodic_v if on_u else self.periodic_u
        inward   = 1.0 if edge <= 0.0 else -1.0
        across   = (1, 0) if on_u else (0, 1)

        def slopes_at(rows, x, dists=False):
            # (k,) starts' rows and parameters along the pole -> (k,) slopes
            # into the patch there (and squared distances)
            pairs = [np.full(x.shape, edge), x] if on_u else [x, np.full(x.shape, edge)]
            xi    = np.column_stack(pairs)
            point = np.asarray(self._spl(xi)) - queries[rows]
            slope = np.asarray(self._spl(xi, nu=across))
            slope = inward * np.einsum("kd,kd->k", point, slope)
            if dists:
                return slope, np.einsum("kd,kd->k", point, point)
            return slope

        def wrap(x):
            return x % end if periodic else x

        spans = max(int(end + 0.5), 1)
        step  = end / (4 * spans)
        count = 4 * spans + (0 if periodic else 1)
        grid  = np.arange(count) * step
        rows  = np.repeat(np.arange(len(queries)), count)
        slope, dists = slopes_at(rows, np.tile(grid, len(queries)), True)
        slope = slope.reshape(-1, count)
        dists = dists.reshape(-1, count)

        def lowest(values):
            # each sample lower than its neighbours (the first of a tie;
            # around the seam on a periodic direction)
            if periodic:
                before = np.roll(values, 1, axis=1)
                after  = np.roll(values, -1, axis=1)
            else:
                past   = np.full((len(values), 1), np.inf)
                before = np.hstack([past, values[:, :-1]])
                after  = np.hstack([values[:, 1:], past])
            return (values < before) & ~(after < values)

        at, i = np.nonzero((slope < 0.0) & lowest(slope))
        best_x = grid[i]
        best   = slope[at, i]

        # Golden section between the neighbours of each
        ratio = 0.6180339887498949
        lo    = best_x - step
        hi    = best_x + step
        if not periodic:
            lo = np.maximum(lo, 0.0)
            hi = np.minimum(hi, end)
        a   = hi - ratio * (hi - lo)
        b   = lo + ratio * (hi - lo)
        f_a = slopes_at(at, wrap(a))
        f_b = slopes_at(at, wrap(b))
        for _ in range(30):
            left   = f_a < f_b
            hi     = np.where(left, b, hi)
            lo     = np.where(left, lo, a)
            kept   = np.where(left, a, b)
            kept_f = np.where(left, f_a, f_b)
            fresh  = np.where(left, hi - ratio * (hi - lo), lo + ratio * (hi - lo))
            f_new  = slopes_at(at, wrap(fresh))
            a      = np.where(left, fresh,  kept)
            b      = np.where(left, kept,   fresh)
            f_a    = np.where(left, f_new,  kept_f)
            f_b    = np.where(left, kept_f, f_new)
        for x, f in ((a, f_a), (b, f_b)):
            better         = f < best
            best_x[better] = wrap(x)[better]
            best[better]   = f[better]

        if periodic:
            best_x = np.where(best_x >= end, best_x - end, best_x)
        if not rim:
            return at, best_x

        # A nearly collapsed pole is a rim too: each sample closer to the
        # query than its neighbours, after the ways out of its query, unless
        # the rim's spots all lie within a thousandth of one distance from
        # it (it looks like one point from there)
        nearest = np.sqrt(dists.min(axis=1))
        seen    = np.sqrt(dists.max(axis=1)) - nearest > 1e-3 * nearest
        near, j = np.nonzero(lowest(dists) & seen[:, None])
        order = np.argsort(np.r_[at, near], kind="stable")
        return np.r_[at, near][order], np.r_[best_x, grid[j]][order]

    def _spl_derivatives(self, u, v):
        """
        S, S_u, S_v, S_uu, S_uv and S_vv at native (u, v) from ``_spl``,
        stacked (6, n, dims).
        """
        xi = np.column_stack([u, v])
        return np.stack(
            [
                np.asarray(self._spl(xi, nu=nu))
                for nu in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
            ]
        )

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
            params in each direction's [min_param, max_param] (the knots'
            range with knots), or [0, 1] when uniform; tangents the partial
            derivatives with respect to them, as ``compute()`` gives.

        Notes
        -----
        Requires a 3D control-point grid (``points.shape[2] == 3``).
        Algorithm matches the bilinear / Bezier mesh raycast pattern in
        ``MeshData.raycast``: precompute coarse samples, find best initial
        (u, v) per ray, then Gauss-Newton refine. The best start is the
        sample nearest the ray's line, wherever along it, so a ray that
        crosses the patch more than once gets the hit its search lands on,
        not always the first along it.
        """
        self._refuse_weights("raycast")
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
        grid_u   = np.ascontiguousarray(uu.ravel(), dtype=np.float64)
        grid_v   = np.ascontiguousarray(vv.ravel(), dtype=np.float64)
        grid_pts = self._points_at(grid_u, grid_v)
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
        points, tang_u, tang_v = self._derivs_at(safe_u, safe_v)

        # derivatives with respect to the knots' parameters, as in Maya
        tang_u, tang_v = self._knot_tangents(safe_u, safe_v, tang_u, tang_v)

        # --- normals & occlusion (computed from the tangents) ---
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

        params_u         = self._denormalize_u(safe_u)
        params_v         = self._denormalize_v(safe_v)
        params           = np.stack([params_u, params_v], axis=1)
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
        self._hw                 = None
        self._arc_length_table_u = None
        self._u_table_u          = None
        self._total_length_u     = None
        self._arc_length_table_v = None
        self._u_table_v          = None
        self._total_length_v     = None
        self._cp_params_u        = None
        self._cp_params_v        = None
        self._lo_u               = None
        self._hi_u               = None
        self._scale_u            = None
        self._edges_u            = None
        self._offset_u           = None
        self._lo_v               = None
        self._hi_v               = None
        self._scale_v            = None
        self._edges_v            = None
        self._offset_v           = None

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
        # after direct mutation of self.points) and the evaluators on them
        self._set_spline()

        if self.uniform_u:
            self._build_arc_length_table_u()
        if self.uniform_v:
            self._build_arc_length_table_v()

    # ------------------------------------------------------------------ #
    #  Open / Close                                                       #
    # ------------------------------------------------------------------ #

    def open_u(self):
        """
        Make U direction non-periodic.

        Custom ``knots_u`` are dropped: U goes back to uniform knots.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self.periodic_u:
            self.periodic_u = False
            self.knots_u    = None
            self.rebuild()

    def close_u(self):
        """
        Make U direction periodic.

        Custom ``knots_u`` are dropped: U goes back to uniform knots.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if not self.periodic_u:
            self.periodic_u = True
            self.knots_u    = None
            self.rebuild()

    def open_v(self):
        """
        Make V direction non-periodic.

        Custom ``knots_v`` are dropped: V goes back to uniform knots.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if self.periodic_v:
            self.periodic_v = False
            self.knots_v    = None
            self.rebuild()

    def close_v(self):
        """
        Make V direction periodic.

        Custom ``knots_v`` are dropped: V goes back to uniform knots.
        """
        if self._max_param_u is None:
            self._init_bspline()
        if not self.periodic_v:
            self.periodic_v = True
            self.knots_v    = None
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

        After smoothing, all internal caches are invalidated since
        self.points has been replaced.
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

        # self.points was replaced; every cached value (knots, cv float64,
        # scipy spline, arc-length tables, cp_params) is now stale.
        # Invalidate so the next access rebuilds from the new points.
        self.points = pts
        self.invalidate()

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
                _, su, _ = self._derivs_at(u_arr, v_arr)
                return float(np.linalg.norm(su[0]))

            return float(_si.quad(_speed, 0, self._max_param_u)[0])

        n     = (self._count_u + self.periodic_u) * samples
        u_lin = np.linspace(0, self._max_param_u, n, endpoint=not self.periodic_u)
        v_arr = np.full(n, v_mid, dtype=np.float64)
        if self.periodic_u:
            u_lin = u_lin % self._max_param_u
        pts = self._points_at(u_lin.astype(np.float64), v_arr)
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
                _, _, sv = self._derivs_at(u_arr, v_arr)
                return float(np.linalg.norm(sv[0]))

            return float(_si.quad(_speed, 0, self._max_param_v)[0])

        n     = (self._count_v + self.periodic_v) * samples
        v_lin = np.linspace(0, self._max_param_v, n, endpoint=not self.periodic_v)
        u_arr = np.full(n, u_mid, dtype=np.float64)
        if self.periodic_v:
            v_lin = v_lin % self._max_param_v
        pts = self._points_at(u_arr, v_lin.astype(np.float64))
        if self.periodic_v:
            pts = np.concatenate([pts, pts[:1]], axis=0)
        d = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))

    def get_area(self, samples: int = 51) -> float:
        """
        Numerical surface area: ``||S_u x S_v||`` summed over ``samples`` by
        ``samples`` parameters, each direction by its own rule.

        An open direction takes composite Simpson's rule: the 1/3 rule when
        ``samples`` is odd; when it is even, the 3/8 rule over three steps
        at an end and the 1/3 rule over the rest, the two ends averaged. A
        periodic direction's samples stop short of its end and weigh the
        same, the trapezoid rule around a closed direction. The samples are
        evenly spaced in the native parameters, not lined up with the knots:
        uneven knots (``knots_u`` / ``knots_v``), short end spans most of
        all, need more of them for the same accuracy.

        Parameters
        ----------
        samples : int
            Samples per direction, 51 by default; 2 or more, fewer raise
            ValueError.

        Returns
        -------
        float
            Approximate surface area.
        """
        if samples < 2:
            raise ValueError(f"NurbsSurfaceData.get_area: samples must be 2 or more, got {samples}")
        if self._max_param_u is None:
            self._init_bspline()

        u_lin = np.linspace(0, self._max_param_u, samples, endpoint=not self.periodic_u)
        v_lin = np.linspace(0, self._max_param_v, samples, endpoint=not self.periodic_v)
        uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")
        u_flat = uu.ravel().astype(np.float64)
        v_flat = vv.ravel().astype(np.float64)

        _, su, sv = self._derivs_at(u_flat, v_flat)

        cross = np.cross(su, sv)
        norms = np.linalg.norm(cross, axis=1).reshape(samples, samples)

        du = self._max_param_u / (samples - (not self.periodic_u))
        dv = self._max_param_v / (samples - (not self.periodic_v))

        w_u, over_u = _area_rule(samples, self.periodic_u)
        w_v, over_v = _area_rule(samples, self.periodic_v)
        return float(np.sum(np.outer(w_u, w_v) * norms) * du * dv / (over_u * over_v))


class NurbsSurfaceList(DataList[NurbsSurfaceData]):
    """A list of :class:`NurbsSurfaceData`, as a file holds its surfaces."""


def _area_rule(samples: int, periodic: bool) -> tuple:
    """
    The weights of one direction's samples in ``get_area()`` and the number
    they are over, the step between samples left out.

    A periodic direction's samples stop short of its end and weigh the
    same: the trapezoid rule around a closed direction. Simpson's weights
    are made for a first and a last sample at the two ends, which such a
    direction does not have.

    An open direction takes composite Simpson's rule. An odd ``samples``
    leaves an even number of steps: the 1/3 rule over each pair. An even
    one leaves an odd number: three steps at an end take the 3/8 rule and
    the rest the 1/3 rule, summed once from each end and averaged, so that
    turning the direction round leaves the sum as it was. Two samples are
    the trapezoid rule.
    """
    weights = np.ones(samples, dtype=np.float64)
    if periodic:
        return weights, 1.0
    if samples == 2:
        return weights, 2.0
    if samples % 2:
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return weights, 3.0
    # over 24: the 1/3 rule's 1 4 2 .. 4 1 by 8, the 3/8 rule's 1 3 3 1 by 9
    ending = np.zeros(samples, dtype=np.float64)
    if samples > 4:
        ending[:-3]    = 8.0
        ending[1:-4:2] = 32.0
        ending[2:-4:2] = 16.0
    ending[-4:] += (9.0, 27.0, 27.0, 9.0)
    return ending + ending[::-1], 48.0


def _direction_knots(knots, kv, count, degree, periodic, label):
    """
    One direction's knot vector and parameter map, from its ``knots_u`` or
    ``knots_v``, checked as ``NurbsCurveData.knots`` is.

    The patch runs on the native [0, spans] each way, as with uniform knots.
    A direction of degree 2 or more takes the knots rescaled to it, as a
    curve does, and one scale maps its parameter. A degree-1 direction keeps
    its unit-spaced knots, and its knots map the parameter span by span
    instead: there it is the same patch, and the closest point search's
    degree-1 spans stay unit ones.

    Parameters
    ----------
    knots : array-like or None
        The field: Maya's layout, count + degree - 1 values (open) or
        count + 2 * degree - 1 (periodic); None for uniform knots.
    kv : np.ndarray
        The direction's uniform knot vector, scipy's layout.
    count : int
        Control points along the direction, without the periodic wrap.
    degree : int
        Degree of the direction.
    periodic : bool
        Whether the direction wraps.
    label : str
        The field, as errors name it: ``"NurbsSurfaceData.knots_u"``.

    Returns
    -------
    tuple
        (kv, lo, hi, scale, edges): the knot vector the patch runs on; its
        parameter range; parameter units per native unit; and on a degree-1
        direction with knots, the knots (spans + 1,), one per span join,
        else None.

    Raises
    ------
    ValueError
        As ``NurbsCurveData.knots``, the field named: the wrong count, knots
        that are not finite or that decrease, a range of length zero, too
        few control points for the degree, a value repeated more than
        ``degree`` times (on a degree-1 direction, any value repeated: its
        knots are its rows' parameters).
    """
    spans = count if periodic else count - degree
    if knots is None:
        return kv, 0.0, float(spans), 1.0, None

    try:
        full, lo, hi = _knot_vector(knots, count, degree, periodic, "direction")
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from None
    if degree > 1:
        return full, lo, hi, (hi - lo) / spans, None

    # degree 1: count + 2 * degree - 1 (periodic) or count + degree - 1
    # (open) values are spans + 1 either way, one per span join, and they
    # increase (no value repeats more than the degree)
    edges = np.asarray(knots, dtype=np.float64).ravel()
    return kv, lo, hi, (hi - lo) / spans, edges


def _direction_pieces(kv, count, degree, periodic):
    """
    One direction's smooth pieces: the patch split at each knot repeated
    ``degree`` times inside the direction's range, where it is only C0 (a
    crease; ``knots_u`` / ``knots_v`` make one). On its own control points
    and knots a piece is the same patch: no basis function crosses a
    crease, and the knot before a crease's run counts nowhere after it.

    Parameters
    ----------
    kv : np.ndarray
        The direction's knot vector, scipy's layout, native.
    count : int
        Control points along the direction, without the periodic wrap.
    degree : int
        Degree of the direction.
    periodic : bool
        Whether the direction wraps.

    Returns
    -------
    list or None
        None on a degree-1 direction (the search keeps its folds itself) or
        one without a crease. Else per piece, in order, (rows, knots): its
        control points' indices along the direction, into the unwrapped
        ones, and its knots in Maya's layout, native. A periodic
        direction's last piece runs on through the seam, past the end of
        its range.
    """
    if degree < 2:
        return None

    # The knots in a row: a periodic direction's over two periods on (a
    # period on, each knot is the period further)
    if periodic:
        index = np.arange(2 * count + 2 * degree + 1)
        knots = kv[index % count] + (index // count) * count
    else:
        knots = kv

    # Runs of one value, and the creases among them: degree long, inside the
    # range (on a periodic direction, starting within its first period)
    starts = np.flatnonzero(np.r_[True, knots[1:] != knots[:-1]])
    ends   = np.r_[starts[1:], knots.shape[0]] - 1
    if periodic:
        inside = (starts >= degree) & (starts < degree + count)
    else:
        inside = (knots[starts] > knots[degree]) & (knots[starts] < knots[count])
    crease = inside & (ends - starts + 1 >= degree)
    if not crease.any():
        return None

    # From each crease to the next: an open direction's ends are its first
    # and last, a periodic one's last goes round to its first. A piece's
    # control points are those whose basis functions reach inside it
    runs = list(zip(starts[crease], ends[crease]))
    if periodic:
        runs.append((runs[0][0] + count, runs[0][1] + count))
    else:
        runs = [(degree, degree)] + runs + [(count, count)]
    return [
        (np.arange(last - degree, first) % count, knots[last - degree + 1 : first + degree])
        for (_, last), (first, _) in zip(runs[:-1], runs[1:])
    ]


def _to_native(x, lo, scale, edges, periodic):
    """
    A direction's parameters (the knots' range) as native ones ([0, spans]):
    one scale, or with edges (a degree-1 direction's knots) span by span.
    Past an open direction's range the end span's map goes on; a periodic
    one wraps, whole periods added back.
    """
    x = np.asarray(x, dtype=np.float64)
    if edges is None:
        return (x - lo) / scale

    spans  = edges.shape[0] - 1
    native = np.arange(spans + 1, dtype=np.float64)
    if periodic:
        period = edges[-1] - edges[0]
        laps   = np.where((x >= edges[0]) & (x <= edges[-1]), 0.0, np.floor((x - edges[0]) / period))
        return np.interp(x - laps * period, edges, native) + laps * spans

    s = np.interp(x, edges, native)
    s = np.where(x < edges[0], (x - edges[0]) / (edges[1] - edges[0]), s)
    return np.where(x > edges[-1], spans + (x - edges[-1]) / (edges[-1] - edges[-2]), s)


def _to_user(s, lo, scale, edges, periodic):
    """``_to_native``'s inverse: native parameters as the direction's own."""
    s = np.asarray(s, dtype=np.float64)
    if edges is None:
        return lo + s * scale

    spans  = edges.shape[0] - 1
    native = np.arange(spans + 1, dtype=np.float64)
    if periodic:
        laps = np.where((s >= 0.0) & (s <= spans), 0.0, np.floor(s / spans))
        return np.interp(s - laps * spans, native, edges) + laps * (edges[-1] - edges[0])

    x = np.interp(s, native, edges)
    x = np.where(s < 0.0, edges[0] + s * (edges[1] - edges[0]), x)
    return np.where(s > spans, edges[-1] + (s - spans) * (edges[-1] - edges[-2]), x)


def _rate(s, scale, edges):
    """
    Parameter units per native unit at native s: the direction's scale, or
    with edges the width of the span s lies on, the one the kernels pick
    (the one above a knot, the last at the end).
    """
    if edges is None:
        return scale
    widths = np.diff(edges)
    span   = np.clip(np.floor(np.asarray(s, dtype=np.float64)), 0, widths.shape[0] - 1)
    return widths[np.nan_to_num(span).astype(np.intp)]  # a nan's span: the first, as the kernels'


def _slopes(queries, derivs):
    """
    The numba kernel's ``_patch_slopes`` for many points at once: the
    squared distances from queries to the patch and their derivatives.

    Parameters
    ----------
    queries : np.ndarray
        Query points (m, dims).
    derivs : np.ndarray
        S, S_u, S_v, S_uu, S_uv and S_vv at each point's (u, v), stacked
        (6, m, dims).

    Returns
    -------
    np.ndarray
        (f, g_u, g_v, h_uu, h_uv, h_vv, j_uu, j_uv, j_vv) stacked (9, m):
        see ``_patch_slopes``.
    """
    r    = derivs[0] - queries
    j_uu = np.einsum("ij,ij->i", derivs[1], derivs[1])
    j_uv = np.einsum("ij,ij->i", derivs[1], derivs[2])
    j_vv = np.einsum("ij,ij->i", derivs[2], derivs[2])
    return np.stack(
        [
            np.einsum("ij,ij->i", r, r),
            np.einsum("ij,ij->i", r, derivs[1]),
            np.einsum("ij,ij->i", r, derivs[2]),
            np.einsum("ij,ij->i", r, derivs[3]) + j_uu,
            np.einsum("ij,ij->i", r, derivs[4]) + j_uv,
            np.einsum("ij,ij->i", r, derivs[5]) + j_vv,
            j_uu,
            j_uv,
            j_vv,
        ]
    )


def _span_exits(x, lo, g, end, periodic):
    """
    The numba kernel's ``_span_exit`` for many points at once: where each
    search on an end of its degree-1 span (lo, lo + 1), its slope pointing
    past that end, would go on.

    Parameters
    ----------
    x, lo, g : np.ndarray
        Parameters, the starts of their spans and the slopes along them
        (m,).
    end : float
        End of the direction's domain.
    periodic : bool
        Whether the direction wraps.

    Returns
    -------
    tuple
        (x, lo), each (m,): the same spot on the next span, lo -1 where
        there is none.
    """
    down                  = (x <= lo) & (g > 0.0)
    up                    = (x >= lo + 1.0) & (g < 0.0)
    inner                 = lo > 0.0
    below                 = lo + 1.0 < end
    next_x                = x.copy()
    next_lo               = np.full_like(lo, -1.0)
    next_lo[down & inner] = lo[down & inner] - 1.0
    next_lo[up & below]   = lo[up & below] + 1.0
    if periodic:
        next_x[down & ~inner]  = end
        next_lo[down & ~inner] = end - 1.0
        next_x[up & ~below]    = 0.0
        next_lo[up & ~below]   = 0.0
    return next_x, next_lo


def _keep_inside(x, low, high, end, ends):
    """
    Trial parameters kept inside their range: clipped to (low, high) when
    the direction has ends, else wrapped into [0, end).
    """
    if ends:
        return np.minimum(np.maximum(x, low), high)
    x = x % end
    return np.where(x >= end, x - end, x)


def _poles(cv, periodic_u, periodic_v):
    """
    The numba kernel's ``_patch_poles``: for the edges u = 0, u = end,
    v = 0 and v = end, 0 not a pole, 1 nearly one (all of the edge's
    control points within a tenth of how far the next row's lie from
    them), 2 one point (within 1e-12 of that).
    """
    rows  = (cv[0], cv[-1], cv[:, 0], cv[:, -1])
    after = (cv[1], cv[-2], cv[:, 1], cv[:, -2])
    edged = (not periodic_u, not periodic_u, not periodic_v, not periodic_v)
    codes = []
    for open_edge, row, inner in zip(edged, rows, after):
        extent = np.ptp(row, axis=0).max()
        reach  = np.linalg.norm(inner - row, axis=-1).max()
        if not open_edge:
            codes.append(0)
        elif extent <= 1e-12 * reach:
            codes.append(2)
        elif extent <= 1e-1 * reach:
            codes.append(1)
        else:
            codes.append(0)
    return codes


def _pole_at(u, v, poles, max_u, max_v):
    """
    The numba kernel's ``_mark_pole`` for many points at once: the pole
    (0 to 3, as in ``_poles``) each (u, v) lies on, -1 where none.
    """
    pole = np.full(u.shape, -1)
    for index, on in enumerate((u <= 0.0, u >= max_u, v <= 0.0, v >= max_v)):
        pole = np.where((pole < 0) & on & (poles[index] > 0), index, pole)
    return pole
