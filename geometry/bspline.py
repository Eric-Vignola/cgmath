from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.integrate as si
from cgmath.geometry._base import Data, ImmutableArray as numpy_array
from cgmath.geometry.utils import compute_basis
from cgmath.geometry.utils._numba._bspline import (
    _compute_basis_parallel,
    _evaluate_bspline_curve,
    _evaluate_bspline_curve_derivative,
    _newton_closest_point_parallel,
)
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


@dataclass(repr=False, eq=False)
class SampleData(Data):
    """
    A dataclass to hold bspline sample data
    """

    points:    np.ndarray
    tangents:  np.ndarray
    distances: np.ndarray
    params:    np.ndarray
    basis:     np.ndarray

    def __call__(self, values):
        return self.compute(values)

    def compute(self, values):
        """computes weighted values from given data"""
        return np.einsum("jk,ij->ik", values, self.basis)

    def copy(self):
        return type(self)(
            points    = np.copy(self.points),
            tangents  = np.copy(self.tangents),
            distances = np.copy(self.distances),
            params    = np.copy(self.params),
            basis     = np.copy(self.basis),
        )


@dataclass
class BSplineData(Data):
    """
    BSpline data class.

    Parameters
    ----------
    points : np.ndarray
        Control point positions (N, dims). Empty by default, so
        ``BSplineData()`` is an empty cubic for ``fit()`` to fill.
    degree : int
        Degree of the B-spline curve. Default is 3.
    periodic : bool
        Whether the curve is closed/periodic. Default is False.
    uniform : bool
        If True, uses arc-length parameterization where equal spacing in
        parameter space [0, 1] gives equal spacing along the curve.
        If False (default), uses the curve's parameter range
        [min_param, max_param]: the knots' own range, or [0, spans].
    use_numba : bool
        If True (default), uses optimized Numba kernels for evaluation.
        If False, uses scipy.interpolate.BSpline.
    registered : bool
        If True, the start of the parameter range on periodic curves aligns
        with the curve point nearest to points[0]. The offset is CV 0's
        Greville abscissa (the average of the knots under its basis
        function), which on uniform knots is
        offset = max_param - (degree - 1) / 2.
        Has no effect on non-periodic curves.
    arc_length_samples : int
        Number of samples used to build the arc-length lookup table.
        Higher values give more accurate arc-length parameterization.
        Default is 1000.
    knots : np.ndarray, optional
        Knot vector in Maya's layout, the one ``kv`` returns:
        count + degree - 1 values on open curves, count + 2 * degree - 1
        on periodic ones. None (default) gives uniform knots on
        [0, spans]. Their range is the curve's parameter range, as in Maya:
        ``min_param`` to ``max_param``, so a parameter names the same point
        here and on a Maya curve built from ``points`` and ``kv``, and
        tangents are derivatives with respect to it. Periodic curves
        rebuild the knots outside that range from the ones inside it, so
        the seam stays smooth. Set by ``fit()``, cleared by ``open()`` and
        ``close()``; call ``invalidate()`` after changing it by hand.
    """

    points:             np.ndarray = numpy_array(np.zeros((0, 3)))
    degree:             int = field(default=3)
    periodic:           bool = field(default=False)
    uniform:            bool = field(default=False)
    use_numba:          bool = field(default=True)
    registered:         bool = field(default=False)
    arc_length_samples: int = field(default=1000)
    knots:              Optional[np.ndarray] = None

    # --- cached attributes --- #
    _kv               = None  # curve knot vector
    _geometry         = None  # parametric point indices
    _count            = None  # curve's physical control point count
    _max_param        = None  # curve's max parameter
    _spl              = None  # scipy.interpolate.BSpline instance
    _der              = None  # first derivative of _spl (for tangents)
    _cv_f64           = None  # cached float64 control vertices (with periodic wrap)
    _arc_length_table = None  # normalized arc lengths [0, 1]
    _u_table          = None  # corresponding native parameter values
    _total_length     = None  # total arc length of curve
    _cp_params        = None  # cached control point parameters
    _offset           = None  # registration offset of periodic curves
    _lo               = None  # start of the parameter range (the knots')
    _hi               = None  # end of the parameter range
    _scale            = None  # parameter units per native unit

    __repr__ = Data.__repr__

    def __eq__(self, other):
        return np.all(
            [
                np.allclose(self.cv, other.cv),
                self.degree == other.degree,
                self.periodic == other.periodic,
                self.uniform == other.uniform,
                self.registered == other.registered,
                self.kv.shape == other.kv.shape and np.allclose(self.kv, other.kv),
            ]
        )

    def _init_bspline(self) -> None:
        """
        Sets up internals and generates knot vector.

        For periodic curves:
        - All n control points are used
        - Control points wrap: first 'degree' points are appended to the end
        - Parameter range is [0, n] for a full loop
        - Knot vector is uniform from -degree to n+degree+1

        For open curves:
        - All n control points are used
        - Parameter range is [0, n-degree]
        - Knot vector is clamped (repeated values at ends)

        With ``knots`` set, the knot vector comes from them instead,
        rescaled to the same parameter range.
        """
        self._count = self.points.shape[0]

        if self.periodic:
            # For periodic curves: max_param equals the number of control points
            # This gives a full loop around the curve
            self._max_param = self._count

            # Knot vector for periodic: uniform spacing
            # Needs to cover the wrapped control points
            self._kv = np.arange(
                -self.degree, self._count + self.degree + 1, dtype=np.float64
            )

            # Geometry indices: wrap first 'degree' control points to the end
            # This creates C(degree-1) continuity at the seam
            self._geometry = np.concatenate(
                [
                    np.arange(self._count),
                    np.arange(self.degree),  # Wrap first 'degree' points
                ]
            )
        else:
            # For open curves: standard clamped B-spline
            self._max_param = self._count - self.degree

            # Clamped knot vector: repeated values at ends
            self._kv = np.clip(
                np.arange(self._count + self.degree + 1, dtype=np.float64)
                - self.degree,
                0,
                self._count - self.degree,
            )

            # Geometry indices: direct mapping
            self._geometry = np.arange(self._count)

        # Custom knots replace the uniform ones. Internally the curve still
        # runs on [0, spans]; the knots' own range is the parameter range
        # users see, mapped in _normalize_u / _denormalize_u
        self._lo, self._hi, self._scale = 0.0, float(self._max_param), 1.0
        if self.knots is not None:
            self._kv, self._lo, self._hi = _knot_vector(
                self.knots, self._count, self.degree, self.periodic
            )
            self._scale = (self._hi - self._lo) / self._max_param

        # Registration offset (periodic): CV 0's Greville abscissa, the
        # average of the knots under its basis function
        if self.knots is None:
            self._offset = self._max_param - (self.degree - 1) / 2.0
        else:
            self._offset = self._max_param + np.mean(self._kv[1 : self.degree + 1])

        # Create scipy BSpline with control vertices (may be wrapped for periodic)
        self._cv_f64 = self.cv.astype(np.float64)
        self._spl    = BSpline(self._kv, self._cv_f64, self.degree)
        self._der    = self._spl.derivative()

        # Build arc-length table if uniform parameterization is enabled
        if self.uniform:
            self._build_arc_length_table()

    def _build_arc_length_table(self) -> None:
        """
        Build arc-length lookup table for uniform parameterization.

        Creates a mapping from normalized arc-length [0, 1] to native
        parameter values, enabling equal spacing on the curve.

        When registered and periodic, the table starts from the
        registration offset so that arc-length 0 corresponds to
        the curve point nearest points[0].
        """
        n_samples = self.arc_length_samples

        if self.registered and self.periodic:
            offset = self._offset
            self._u_table = np.linspace(
                offset, offset + self._max_param, n_samples, dtype=np.float64
            )
            eval_u = self._u_table % self._max_param
        else:
            self._u_table = np.linspace(0, self._max_param, n_samples, dtype=np.float64)
            eval_u        = self._u_table

        curve_points = self._spl(eval_u)

        diffs           = np.diff(curve_points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)

        self._arc_length_table     = np.zeros(n_samples, dtype=np.float64)
        self._arc_length_table[1:] = np.cumsum(segment_lengths)

        self._total_length = self._arc_length_table[-1]
        if self._total_length > 0:
            self._arc_length_table /= self._total_length

    def _arc_length_to_param(self, s):
        """
        Convert arc-length parameter(s) in [0, 1] to native parameter space.

        Uses linear interpolation on the pre-computed lookup table.

        Parameters
        ----------
        s : float or np.ndarray
            Arc-length parameter(s) in [0, 1].

        Returns
        -------
        np.ndarray
            Native parameter value(s) in [0, spans].
        """
        s = np.asarray(s, dtype=np.float64)
        return np.interp(s, self._arc_length_table, self._u_table)

    def _param_to_arc_length(self, u):
        """
        Convert native parameter(s) to arc-length parameter space [0, 1].

        Uses linear interpolation on the pre-computed lookup table.

        Parameters
        ----------
        u : float or np.ndarray
            Native parameter value(s) in [0, spans].

        Returns
        -------
        np.ndarray
            Arc-length parameter(s) in [0, 1].
        """
        u = np.asarray(u, dtype=np.float64)
        return np.interp(u, self._u_table, self._arc_length_table)

    def _normalize_u(self, u):
        """
        Convert user parameter space to native parameter space.

        With ``knots`` (non-uniform), maps the knots' range onto the
        native [0, spans].

        If registered and periodic (non-uniform), offsets u so that
        the start of the range aligns with the curve point nearest
        points[0].

        If uniform, converts arc-length [0, 1] to native parameter
        space. When also registered, the arc-length table already
        starts from the registered point, so no explicit offset
        is needed.

        Parameters
        ----------
        u : float or np.ndarray
            User parameter value(s).

        Returns
        -------
        np.ndarray
            Native parameter value(s).
        """
        u = np.asarray(u, dtype=np.float64)

        # the knots' range -> the native [0, spans]
        if self.knots is not None and not self.uniform:
            u = (u - self._lo) / self._scale

        if self.registered and self.periodic and not self.uniform:
            u = u + self._offset

        if self.uniform:
            u = self._arc_length_to_param(u % 1.0 if self.periodic else u)

        return u

    def _denormalize_u(self, u):
        """
        Convert native parameter space to user parameter space.

        If uniform, converts native [0, spans] to arc-length [0, 1].
        When also registered, native params are unwrapped to the table's
        domain [offset, offset + spans] before lookup.

        If registered and periodic (non-uniform), removes the offset
        so that the returned values are relative to points[0], wrapped
        into the range.

        With ``knots`` (non-uniform), maps the native [0, spans] onto the
        knots' range.

        Parameters
        ----------
        u : float or np.ndarray
            Native parameter value(s).

        Returns
        -------
        np.ndarray
            User parameter value(s).
        """
        u = np.asarray(u, dtype=np.float64)

        if self.uniform:
            if self.registered and self.periodic:
                u = np.where(u < self._offset, u + self._max_param, u)
            u = self._param_to_arc_length(u)
        else:
            if self.registered and self.periodic:
                # A value a hair below 0 wraps to max_param itself; that is 0
                u = (u - self._offset) % self._max_param
                u = np.where(u >= self._max_param, 0.0, u)

            # the native [0, spans] -> the knots' range
            if self.knots is not None:
                u = self._lo + u * self._scale

        return u

    def compute(self, u):
        """
        Computes points and tangents on curve at given u coordinate.

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s). If uniform=True, expected in [0, 1] range.
            If uniform=False, expected in [min_param, max_param] range.
            If registered=True on periodic curves, the start of the range
            aligns with points[0].

            For open (non-periodic) curves, values outside the natural domain
            are linearly extrapolated along the endpoint tangent direction:
              - uniform=True: outside [0, 1]; one unit of u corresponds to one
                full curve length (`total_length`) of physical motion.
              - uniform=False: outside [min_param, max_param]; one unit of u
                moves one tangent length.
            The returned tangent in the extrapolated region equals the endpoint
            tangent (held constant during extrapolation). Tangents are
            derivatives with respect to the curve parameter: the knots' one
            on curves with ``knots``, which matches Maya's tangent.

        Returns
        -------
        tuple
            (points, tangents) arrays.
        """
        if self._max_param is None:
            self._init_bspline()

        # For open curves (uniform OR non-uniform), capture original user-space
        # values so we can replace clamped/out-of-domain outputs with linear
        # extrapolation below.
        extrapolate = not self.periodic
        if extrapolate:
            u_user_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))

        # Convert from user space to native parameter space
        # This handles uniform scaling and registration offset
        u = self._normalize_u(u)

        # For non-uniform open curves, clip native u to [0, spans] so the
        # evaluator never sees out-of-knot values (scipy would extrapolate
        # polynomially, numba behaviour is undefined). Out-of-range entries
        # are overwritten with linear extrapolation below.
        if extrapolate and not self.uniform:
            u = np.clip(u, 0.0, self._max_param)

        if self.periodic:
            u = u % self._max_param

        if self.use_numba:
            points, tangents = self._compute_numba(u)
        else:
            points, tangents = self._compute_scipy(u)

        if extrapolate:
            points, tangents = self._apply_open_extrapolation(
                u_user_arr, points, tangents
            )

        # derivatives with respect to the knots' parameter, as in Maya
        if self.knots is not None:
            tangents = tangents / self._scale

        return points, tangents

    def _compute_scipy(self, u):
        """Compute using scipy.interpolate.BSpline (original implementation)."""
        u_arr    = np.atleast_1d(u).astype(np.float64)
        points   = self._spl(u_arr)
        tangents = self._der(u_arr)
        if u_arr.shape[0] == 1:
            return points[0], tangents[0]
        return points, tangents

    def _compute_numba(self, u):
        """
        Compute using optimized Numba kernels.

        This is faster than scipy for large numbers of sample points
        due to parallel evaluation and fused basis + point computation.
        """
        u = np.atleast_1d(u).astype(np.float64)

        # Evaluate positions using fused kernel
        points = _evaluate_bspline_curve(u, self._kv, self._cv_f64, self.degree)

        # Evaluate tangents using derivative kernel
        tangents = _evaluate_bspline_curve_derivative(
            u, self._kv, self._cv_f64, self.degree
        )

        # Handle scalar input
        if u.shape[0] == 1:
            return points[0], tangents[0]

        return points, tangents

    def compute_fast(self, u):
        """
        Compute points and tangents using the fastest available method (Numba).

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s).

            For open (non-periodic) curves -- both uniform and non-uniform --
            values outside the natural domain are linearly extrapolated along
            the endpoint tangent direction (see `compute` for details).

        Returns
        -------
        tuple
            (points, tangents) arrays.
        """
        if self._max_param is None:
            self._init_bspline()

        extrapolate = not self.periodic
        if extrapolate:
            u_user_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))

        # Convert from user space to native parameter space
        u = self._normalize_u(u)

        # Clip native u for non-uniform open curves so the evaluator stays in
        # the valid knot range; out-of-range entries get overwritten below.
        if extrapolate and not self.uniform:
            u = np.clip(u, 0.0, self._max_param)

        if self.periodic:
            u = u % self._max_param

        u = np.atleast_1d(u).astype(np.float64)

        points = _evaluate_bspline_curve(u, self._kv, self._cv_f64, self.degree)
        tangents = _evaluate_bspline_curve_derivative(
            u, self._kv, self._cv_f64, self.degree
        )

        if u.shape[0] == 1:
            points   = points[0]
            tangents = tangents[0]

        if extrapolate:
            points, tangents = self._apply_open_extrapolation(
                u_user_arr, points, tangents
            )

        # derivatives with respect to the knots' parameter, as in Maya
        if self.knots is not None:
            tangents = tangents / self._scale

        return points, tangents

    def _apply_open_extrapolation(self, u_user, points, tangents):
        """
        Replace clamped endpoint values with linear extrapolation for open
        (non-periodic) curves when u_user is outside the natural domain.

        Natural domain depends on parameterization:
          - uniform=True:  [0, 1]
          - uniform=False: [min_param, max_param]

        Position formulas (let `lo` and `hi` denote the domain bounds):
          - For u_user > hi:
              uniform=True:  P_end + (u_user - 1) * total_length * unit_tangent_end
              uniform=False: P_end + (u_user - max_param) * tangent_end_native
          - For u_user < lo (lo == 0 in both modes):
              uniform=True:  P_start + u_user * total_length * unit_tangent_start
              uniform=False: P_start + u_user * tangent_start_native

        Tangents in the extrapolated region are held constant at the endpoint
        native-space tangent (matching the value at the corresponding domain
        bound in-range).

        Parameters
        ----------
        u_user : np.ndarray
            Original user-space parameter values (1D, before normalization).
        points : np.ndarray
            Evaluated positions; (n, dims) for array input or (dims,) for
            scalar input. Modified in place where indices fall outside the
            natural domain.
        tangents : np.ndarray
            Evaluated native-space tangents; same shape as `points`.

        Returns
        -------
        tuple
            (points, tangents) with extrapolated values for out-of-range u.
            Output shape matches input shape (1D vs 2D).
        """
        if self.uniform:
            u_low, u_high = 0.0, 1.0
        else:
            u_low, u_high = self._lo, self._hi

        out_low  = u_user < u_low
        out_high = u_user > u_high
        if not (out_low.any() or out_high.any()):
            return points, tangents

        # Work with 2D arrays internally; restore 1D shape on return if needed.
        was_1d = points.ndim == 1
        if was_1d:
            points   = points[None, :]
            tangents = tangents[None, :]

        n_dims = points.shape[1]

        # Helper: per-unit-u position velocity in the extrapolation region.
        # In uniform mode this is total_length * unit_tangent (so du=1 covers
        # the full curve length). In non-uniform mode it is the tangent with
        # respect to the user parameter: the native one over the knots' scale
        # (1 without knots).
        if self.uniform:
            scale = self.total_length

            def _velocity(t_native):
                norm = np.linalg.norm(t_native)
                if norm > 1e-14:
                    return scale * (t_native / norm)
                return np.zeros(n_dims, dtype=np.float64)
        else:

            def _velocity(t_native):
                return t_native / self._scale

        if out_high.any():
            u_end = np.array([self._max_param], dtype=np.float64)
            if self.use_numba:
                p_end = _evaluate_bspline_curve(
                    u_end, self._kv, self._cv_f64, self.degree
                )[0]
                t_end = _evaluate_bspline_curve_derivative(
                    u_end, self._kv, self._cv_f64, self.degree
                )[0]
            else:
                p_end = self._spl(u_end)[0]
                t_end = self._der(u_end)[0]

            v_end         = _velocity(t_end)
            idx           = np.where(out_high)[0]
            delta         = (u_user[idx] - u_high)[:, None]
            points[idx]   = p_end + delta * v_end
            tangents[idx] = t_end

        if out_low.any():
            u_start = np.array([0.0], dtype=np.float64)
            if self.use_numba:
                p_start = _evaluate_bspline_curve(
                    u_start, self._kv, self._cv_f64, self.degree
                )[0]
                t_start = _evaluate_bspline_curve_derivative(
                    u_start, self._kv, self._cv_f64, self.degree
                )[0]
            else:
                p_start = self._spl(u_start)[0]
                t_start = self._der(u_start)[0]

            v_start = _velocity(t_start)
            idx     = np.where(out_low)[0]
            # u_user[idx] is below the range -> moves opposite to the start
            # tangent, extending the curve backward in space.
            delta         = (u_user[idx] - u_low)[:, None]
            points[idx]   = p_start + delta * v_start
            tangents[idx] = t_start

        if was_1d:
            return points[0], tangents[0]
        return points, tangents

    def invalidate(self) -> None:
        """
        Invalidate all cached data.

        Call this after modifying control points to ensure
        subsequent computations use the updated geometry.

        Example
        -------
        >>> curve.points[0] = [1.0, 2.0, 3.0]
        >>> curve.invalidate()  # Forces full re-initialization on next use
        >>> points, tangents = curve.compute(u)
        """
        self._max_param        = None
        self._spl              = None
        self._der              = None
        self._kv               = None
        self._geometry         = None
        self._count            = None
        self._cv_f64           = None
        self._arc_length_table = None
        self._u_table          = None
        self._total_length     = None
        self._cp_params        = None
        self._offset           = None
        self._lo               = None
        self._hi               = None
        self._scale            = None

    def _refresh(self) -> None:
        """Rebuilds every cache from the current fields."""
        self.invalidate()
        self._init_bspline()

    def rebuild(
        self,
        count:    Optional[int]   = None,
        degree:   Optional[int]   = None,
        collapse: Optional[tuple] = None,
        knots:    str             = "best",
    ) -> float:
        """
        Rebuilds the curve with a new control point count, degree or
        stacked ends, as close as it can to its current shape, and returns
        how far it moved.

        Called bare, nothing changes: it refreshes every cache from the
        current control points, as after editing ``points`` by hand.

        The parameter range (``domain``) is kept, and so are the end
        points of an open curve. More control points at the same degree
        keep the exact shape: knots are inserted. Otherwise the curve is
        the closest one its new knots allow, measured the same way at
        every parameter, so points keep roughly the same parameter. Sets
        ``points``, ``knots`` and ``degree``. This isn't Maya's
        ``rebuildCurve``, which spaces its knots evenly over a new range.

        Parameters
        ----------
        count : int, optional
            Number of control points. None keeps the current count.
        degree : int, optional
            Degree of the new curve. None keeps the current degree.
        collapse : tuple of int, optional
            ``(start, end)``: how many control points sit stacked on each
            end of an open curve, 0 or 2 to degree each. Stacked control
            points are exactly equal, which leaves the curve no tangent at
            that end. None keeps the stacks the curve has, as
            ``get_collapsed_points()`` finds them at its ends, capped at
            the new degree. ``(0, 0)`` stops keeping them stacked; the
            closest shape may still leave them where they are.
        knots : str
            ``"best"`` (default): the closest shape. Exact when it can be;
            otherwise the closest of three knot placements: evenly
            spaced, following the current spacing, and denser where the
            curve bends. ``"even"``: evenly spaced knots over the current
            range, and the closest shape for them.

        Returns
        -------
        float
            The largest distance between the new curve and the old one,
            measured both ways on dense samples; 0.0 when nothing
            changes.

        Raises
        ------
        ValueError
            An empty curve given any argument; a count of -1 (only fit()
            picks a count) or too small for the degree and stacks; a
            degree below 1; a bad ``collapse`` or ``knots``; any stack on
            a periodic curve.

        Example
        -------
        >>> curve = BSplineData.from_edit_points(joint_positions, collapse=(2, 2))
        >>> moved = curve.rebuild(count=8)       # 8 CVs, still stacked
        >>> curve.rebuild(count=16) < 1e-12      # exact: knots inserted
        True
        """
        if knots not in ("best", "even"):
            raise ValueError(f'knots must be "best" or "even", got {knots!r}')

        # Nothing asked: refresh the caches only
        if count is None and degree is None and collapse is None and knots == "best":
            self._refresh()
            return 0.0

        if len(self.points) == 0:
            raise ValueError("an empty curve has nothing to rebuild; fit() it first")
        if count is not None and _as_int(count, "count") == -1:
            raise ValueError(
                "count=-1 (the count that passes through every point) only "
                "applies to fit()"
            )

        if self._max_param is None:
            self._init_bspline()

        old_degree = int(self.degree)
        new_degree = old_degree if degree is None else _check_degree(degree)
        stack      = self._end_stacks(collapse, new_degree)
        old_count  = self._count
        new_count  = old_count if count is None else _as_int(count, "count")
        _check_count(new_count, new_degree, self.periodic, stack, "pass a larger count")

        span  = int(self._max_param)
        spans = new_count if self.periodic else new_count - new_degree
        edges = self._kv[old_degree : old_degree + span + 1]
        same  = new_degree == old_degree and stack == self._end_stacks(None, old_degree)

        # Nothing changes: same curve, same knots
        even = np.linspace(0.0, span, spans + 1)
        if same and new_count == old_count:
            if knots == "best" or np.array_equal(even, edges):
                self._refresh()
                return 0.0

        # Candidate knots in the current curve's range [0, span]
        if knots == "even":
            candidates = [even]
        elif same and new_count > old_count:
            candidates = [_split_longest(edges, new_count - old_count)]
        else:
            follow     = np.interp(np.arange(spans + 1) * span / spans, np.arange(span + 1), edges)
            candidates = [even, follow, _bend_edges(self._spl, span, spans)]

        best = None
        for candidate in candidates:
            full   = _knots_from_edges(candidate / span * spans, new_degree, self.periodic)
            cv     = _project(self._spl, span, full, new_count, new_degree, self.periodic, stack)
            spline = BSpline(full, _wrap(cv, new_degree, self.periodic), new_degree)
            moved  = _curve_distance(self._spl, span, spline, spans)
            if best is None or moved < best[0]:
                best = (moved, cv, full)

        # Keep the parameter range, end knots exactly
        moved, cv, full = best
        lo, hi = self._lo, self._hi
        stored                         = lo + full[1:-1] / spans * (hi - lo)
        stored[new_degree - 1]         = lo
        stored[new_degree - 1 + spans] = hi
        if not self.periodic:
            stored[: new_degree - 1]     = lo
            stored[new_degree + spans :] = hi

        self.degree = new_degree
        self.points = cv
        self.knots  = stored
        self._refresh()
        return moved

    def get_collapsed_points(self, tol: float = 1e-6) -> list:
        """
        Groups of control points stacked on each other: every run of
        neighbouring control points within ``tol`` of the next one.

        Only reports; ``fit()`` and ``rebuild()`` keep the runs that start
        or finish an open curve. A periodic curve's run can cross the
        seam, from its last control points to its first.

        Parameters
        ----------
        tol : float
            Largest distance between two neighbouring control points that
            still counts as stacked.

        Returns
        -------
        list of list of int
            Control point indices of each run, in curve order, e.g.
            ``[[0, 1], [10, 11]]``. Empty when nothing is stacked.

        Example
        -------
        >>> curve = BSplineData.from_edit_points(joint_positions, collapse=(2, 2))
        >>> curve.get_collapsed_points()
        [[0, 1], [10, 11]]
        """
        cv    = np.asarray(self.points, dtype=np.float64)
        count = cv.shape[0]
        if count < 2:
            return []

        close = np.linalg.norm(np.diff(cv, axis=0), axis=1) <= tol
        if self.periodic and np.all(close) and np.linalg.norm(cv[-1] - cv[0]) <= tol:
            return [list(range(count))]

        groups = []
        run    = [0]
        for i, stacked in enumerate(close):
            if stacked:
                run.append(i + 1)
                continue
            if len(run) > 1:
                groups.append(run)
            run = [i + 1]
        if len(run) > 1:
            groups.append(run)

        # A periodic curve joins its last run to its first across the seam
        if self.periodic and np.linalg.norm(cv[-1] - cv[0]) <= tol:
            head = groups.pop(0) if groups and groups[0][0] == 0 else [0]
            tail = groups.pop() if groups and groups[-1][-1] == count - 1 else [count - 1]
            groups.append(tail + head)

        return groups

    def _end_stacks(self, collapse, degree):
        """
        Stacked control point counts ``(start, end)`` for ``fit()`` and
        ``rebuild()``: checked from ``collapse``, or found at the ends of
        the curve when it is None. A found run longer than the curve's
        degree, or covering every control point, isn't a stack (a curve
        with every control point at the origin has none); one longer than
        ``degree`` is capped to it.
        """
        if collapse is None:
            if self.periodic or len(self.points) == 0:
                return (0, 0)

            last  = len(self.points) - 1
            start = end = 0
            for group in self.get_collapsed_points():
                if len(group) > self.degree or len(group) > last:
                    continue
                if group[0] == 0:
                    start = len(group)
                if group[-1] == last:
                    end = len(group)

            start, end = min(start, degree), min(end, degree)
            return (start if start > 1 else 0, end if end > 1 else 0)

        try:
            start, end = collapse
            valid = all(int(n) == n and not isinstance(n, bool) for n in (start, end))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"collapse must be (start, end), how many control points are "
                f"stacked on each end, got {collapse!r}"
            )

        start, end = int(start), int(end)
        for n in (start, end):
            if n != 0 and not 2 <= n <= degree:
                raise ValueError(
                    f"collapse counts are 0 or 2 to the degree ({degree}), "
                    f"got {collapse!r}"
                )
        if self.periodic and (start or end):
            raise ValueError("a periodic curve has no ends to collapse; use (0, 0) or None")
        return (start, end)

    def rebuild_arc_length_table(self) -> None:
        """
        Rebuild cached spline and arc-length lookup table.

        Call this after modifying control points to update all
        internal caches. This method ALWAYS rebuilds the scipy
        BSpline instance, regardless of the uniform setting.

        Note: For complete cache invalidation, use invalidate() or rebuild().
        """
        if self._max_param is None:
            self._init_bspline()
            return

        # ALWAYS rebuild scipy spline AND cv cache (fixes stale cache issue
        # after direct mutation of self.points).
        self._cv_f64 = self.cv.astype(np.float64)
        self._spl    = BSpline(self._kv, self._cv_f64, self.degree)
        self._der    = self._spl.derivative()

        # Rebuild arc-length table if uniform parameterization is enabled
        if self.uniform:
            self._build_arc_length_table()

    def open(self):
        """
        Opens a periodic (closed) curve.

        Converts the curve from periodic to non-periodic by keeping
        all control points and reinitializing with clamped knot vector.
        Custom ``knots`` are dropped: the curve goes back to uniform knots.
        Topology change invalidates all caches (control_point_params,
        arc-length tables, etc).
        """
        if self._max_param is None:
            self._init_bspline()

        if self.periodic:
            self.periodic = False
            self.knots    = None
            # Topology changed: invalidate everything (cp_params,
            # arc-length table, total_length) and rebuild.
            self._refresh()

    def close(self):
        """
        Closes an open curve to make it periodic.

        Converts the curve from non-periodic to periodic by
        reinitializing with periodic knot vector and wrapped geometry.
        Custom ``knots`` are dropped: the curve goes back to uniform knots.
        Topology change invalidates all caches (control_point_params,
        arc-length tables, etc).
        """
        if self._max_param is None:
            self._init_bspline()

        if not self.periodic:
            self.periodic = True
            self.knots    = None
            # Topology changed: invalidate everything (cp_params,
            # arc-length table, total_length) and rebuild.
            self._refresh()

    def smooth(
        self,
        n:              int,
        polyorder:      int  = 3,
        lock_endpoints: bool = True,
        pad_endpoints:  bool = True,
    ):
        """
        smoothes the curve points via a savgol_filter

        n = half width of the sliding window
        polyorder = order of the polynomial used to fit the samples
        lock_endpoints = on open curves, locks the endpoints of the curve after smoothing
        pad_endpoints = on open curves, pads the curve with endpoints before and after smoothing

        After smoothing, all internal caches are invalidated since
        self.points has been replaced.
        """
        if self._max_param is None:
            self._init_bspline()

        n             = min(n, self.points.shape[0] - (not self.periodic))
        window_length = n * 2 + 1
        polyorder     = min(polyorder, window_length - 1)

        # open curve
        if not self.periodic:
            if pad_endpoints:
                points       = np.zeros((self.points.shape[0] + 2 * n, self.points.shape[1]))
                points[:n]   = self.points[0]
                points[-n:]  = self.points[-1]
                points[n:-n] = self.points
                points       = savgol_filter(points, window_length, polyorder, axis=0)[n:-n]
            else:
                points = savgol_filter(self.points, window_length, polyorder, axis=0)

            if lock_endpoints:
                # linearly adjust the values so the endpoints match
                delta0   = self.points[0] - points[0]
                delta1   = self.points[-1] - points[-1]
                linspace = np.linspace(0, 1, points.shape[0])[:, None]
                points += delta0 + (delta1 - delta0) * linspace

            self.points = points

        # closed curve
        else:
            end    = self._count - 1 - (np.arange(n) % self._count)[::-1]
            start  = np.arange(n + 1) % self._count

            end    = self.points[end]
            start  = self.points[start]
            points = np.concatenate((end, self.points, start))

            self.points = savgol_filter(points, window_length, polyorder, axis=0)[
                n : -(n + 1)
            ]

        # self.points was replaced; every cached value (scipy spline,
        # cv float64, arc-length table, cp_params, total_length) is now
        # stale. Invalidate so the next access rebuilds from fresh CPs.
        self.invalidate()

    @classmethod
    def from_edit_points(
        cls,
        points,
        degree:   int             = 3,
        periodic: bool            = False,
        collapse: Optional[tuple] = None,
        **fields,
    ) -> "BSplineData":
        """
        A new curve through the points, each one an edit point (where two
        spans join): the curve Maya's EP Curve Tool builds, and the spline
        IK curve ``ikHandle`` builds with ``simplifyCurve=False``.

        ``fit()`` with the count it picks (``count=-1``). For each point's
        parameter as well, or another control point count, fit an empty
        curve instead: ``curve = BSplineData(); u = curve.fit(points)``.

        Parameters
        ----------
        points : array-like
            Edit points, in order (N, dims). The curve passes through each,
            on a knot. On a periodic curve of even degree each sits mid-span
            instead (see ``fit()``).
        degree : int
            Degree of the curve.
        periodic : bool
            Whether the curve is closed.
        collapse : tuple of int, optional
            ``(start, end)``: how many control points sit stacked on each
            end of an open curve, 0 or 2 to degree each.
        **fields
            Any other field: ``uniform``, ``registered``, ``use_numba``,
            ``arc_length_samples``.

        Returns
        -------
        BSplineData
            The new curve.

        Raises
        ------
        TypeError
            ``knots`` given: the fit sets them.
        ValueError
            As ``fit()``: too few points, a point repeating the one before
            it, a bad degree or ``collapse``.

        Example
        -------
        >>> spine = BSplineData.from_edit_points(joint_positions, collapse=(2, 2))
        >>> spine.count == len(joint_positions) + 2
        True
        """
        if "knots" in fields:
            raise TypeError("from_edit_points() sets knots itself, from the points")

        curve = cls(degree=degree, periodic=periodic, **fields)
        curve.fit(points, count=-1, collapse=collapse)
        return curve

    def fit(
        self,
        points,
        count:    Optional[int]   = None,
        degree:   Optional[int]   = None,
        collapse: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Moves the control points so the curve passes through the points,
        and returns the curve parameter of each point. For a new curve in
        one line, use ``BSplineData.from_edit_points()``.

        With the count it picks (``count=-1``, or any count on an empty
        curve) it builds the curve Maya's EP Curve Tool builds, which is
        also the spline IK curve when ``ikHandle`` doesn't simplify it
        (``simplifyCurve=False``): each point becomes an edit point, where
        two spans join, and the spans are spaced by the distance between
        the points. The knots are Maya's too: the distance along the
        points, so ``kv`` and every parameter match Maya's curve. Sets
        ``points`` (to a new array), ``knots`` and ``degree``; keeps
        ``periodic``.

        Parameters
        ----------
        points : array-like
            Points to pass through, in order (N, dims). A periodic curve
            closes the loop itself, so a last point repeating the first
            is dropped.
        count : int, optional
            Number of control points. None (default) keeps the current
            count; an empty curve takes the count -1 picks. -1 picks the
            count that passes exactly through every point: Maya's,
            N + degree - 1 when open and N when periodic, plus one per
            stacked control point beyond what Maya's end rule needs (see
            Notes). More control points than that give the same curve,
            with knots inserted. Fewer still pass exactly through the
            points while there is one free control point per point, as
            smoothly as they can, and as close as they can below that;
            open curves keep their ends on the first and last point.
        degree : int, optional
            Degree of the curve. None keeps the current degree.
        collapse : tuple of int, optional
            ``(start, end)``: how many control points sit stacked on each
            end of an open curve, 0 or 2 to degree each, e.g. ``(2, 2)``
            for a spine whose ends don't bend. Stacked control points are
            exactly equal, which leaves the curve no tangent at that end.
            None keeps the stacks the curve has, as
            ``get_collapsed_points()`` finds them at its ends, capped at
            the degree; ``(0, 0)`` removes them.

        Returns
        -------
        np.ndarray
            The curve parameter of each point (N,), in the space
            ``compute()`` and ``sample()`` use: the distance along the
            points, from 0 to max_param (Maya's EP / spline IK parameter),
            or [0, 1] with ``uniform``; periodic curves wrap into the range,
            starting at the registration point when ``registered``. When the
            curve passes through the points, each sits exactly at its u,
            and a point on a knot gets that knot's value from ``kv``. On a
            best fit, u is the closest point on the point's own stretch of
            the curve, searched only half way to its neighbours. Either way
            the points keep their order. ``sample(points)`` gives the same
            values on smooth chains, but it searches the whole curve: on a
            tight fold, or a point the chain passes twice, it can pick
            another stretch.

        Raises
        ------
        ValueError
            A degree below 1; too few points (2 on an open curve, 3 and
            at least ``degree`` on a periodic one); a point at the same
            position as the one before it (stack end control points with
            ``collapse`` instead); a count too small for the degree and
            stacks; a bad ``collapse``; any stack on a periodic curve.

        Notes
        -----
        Open curves need degree - 1 extra conditions. As in Maya,
        derivatives 2 to 1 + ceil((degree - 1) / 2) are zero at the start
        and 2 to 1 + floor((degree - 1) / 2) at the end: a cubic's second
        derivative is zero at both ends, a quadratic's only at the start.
        A stacked end replaces that rule at its end: k stacked control
        points zero derivatives 1 to k - 1 there, so a cubic's pairs need
        no more control points than Maya's curve, and a triple one more.
        That end then differs from Maya's curve. Periodic curves need no
        extra conditions. On even degrees each point sits mid-span, since
        points on the span joins have no solution when their count is
        even. Maya's own periodic EP curve spaced by distance has clamped
        knots and a kink at the seam; this one closes smoothly.

        A high degree with barely one control point per point can
        overshoot between the points: exact interpolation leaves it no
        freedom to stay smooth.

        Example
        -------
        >>> curve = BSplineData()             # a new curve, and each point's u
        >>> u = curve.fit(joint_positions)
        >>> curve.count == len(joint_positions) + 2
        True
        >>> np.allclose(curve.compute(u)[0], joint_positions)
        True
        >>> spine = BSplineData.from_edit_points(joint_positions, collapse=(2, 2))
        >>> u = spine.fit(moved_joints)       # refit: same count and stacks
        >>> spine.get_collapsed_points()
        [[0, 1], [10, 11]]
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(
                f"fit points must be an (N, dims) array, got shape {points.shape}"
            )

        degree = _check_degree(self.degree if degree is None else degree)

        # Positions closer than this count as the same point
        tolerance = 1e-12 * max(1.0, float(np.abs(points).max(initial=0.0)))

        # A periodic curve closes the loop itself
        closing = (
            self.periodic
            and points.shape[0] > 1
            and np.linalg.norm(points[-1] - points[0]) <= tolerance
        )
        if closing:
            points = points[:-1]

        n       = points.shape[0]
        minimum = max(3, degree) if self.periodic else 2
        if n < minimum:
            kind = "a periodic" if self.periodic else "an open"
            raise ValueError(
                f"{kind} curve of degree {degree} needs at least {minimum} "
                f"fit points, got {n}"
            )

        # Stacked ends first: they set the count the points need
        stack = self._end_stacks(collapse, degree)
        exact = _exact_count(n, degree, self.periodic, stack)
        if count is None:
            count = len(self.points) or exact
        else:
            count = _as_int(count, "count")
            if count == -1:
                count = exact
        _check_count(count, degree, self.periodic, stack, "fit with count=-1")

        # The exact curve, or the best one with fewer control points
        fitted = min(count, exact)
        full, params, length = _fit_knots(
            points, fitted, degree, self.periodic, tolerance, stack
        )
        cv = _fit_points(
            points, full, params, fitted, degree, self.periodic, fitted == exact, stack
        )

        # More control points: the same curve with knots inserted
        if count > exact:
            cv, full, params = _insert_knots(
                cv, full, params, degree, self.periodic, count - exact, stack
            )

        # Maya's knots: the distance along the points, so kv and every
        # parameter match Maya's EP and spline IK curves
        spans       = count if self.periodic else count - degree
        self.degree = degree
        self.points = cv
        self.knots  = full[1:-1] * (length / spans)
        self._refresh()

        # Each point's parameter. Through the points, it is where the point
        # was fitted; one on a knot takes the knot's own value, so u matches
        # kv exactly.
        u       = params.copy()
        through = n if self.periodic else n + max(0, stack[0] - 1) + max(0, stack[1] - 1)
        if count >= through:
            edges = self._kv[degree : degree + spans + 1]
            near  = np.clip(np.searchsorted(edges, u), 1, spans)
            for side in (near - 1, near):
                on_knot = np.abs(edges[side] - u) <= 1e-12 * spans
                u       = np.where(on_knot, edges[side], u)
        else:
            u = self._best_fit_params(points, u, spans)

        # Same space as sample(): wrapped on periodic curves, then uniform
        # and registered applied
        if self.periodic:
            u = u % spans
            u = np.where(u >= spans, 0.0, u)
        u = self._denormalize_u(u)

        return np.append(u, u[:1]) if closing else u

    def _best_fit_params(self, points, u, spans):
        """
        Best fit point parameters for ``fit()``: each refined to its closest
        point on the curve without leaving its own partition (half way to
        its neighbours), so the points keep their order. Open curves keep
        their ends on the first and last point.

        Parameters
        ----------
        points : np.ndarray
            Fit points (N, dims).
        u : np.ndarray
            Native parameters the points were fitted at (N,), wrapped into
            [0, spans) on periodic curves.
        spans : int
            The curve's span count.

        Returns
        -------
        np.ndarray
            Refined native parameters (N,), not wrapped.
        """
        u = u.copy()
        if self.periodic:
            wraps = np.flatnonzero(np.diff(u) < 0)
            if wraps.size:
                u[wraps[0] + 1 :] += spans

        mids  = (u[:-1] + u[1:]) / 2.0
        u_min = np.concatenate([u[:1], mids])
        u_max = np.concatenate([mids, u[-1:]])
        mask  = np.ones(u.shape[0], dtype=bool)
        if self.periodic:
            gap       = u[0] + spans - u[-1]
            u_min[0]  = u[0] - gap / 2.0
            u_max[-1] = u[-1] + gap / 2.0
        else:
            # the ends sit exactly on the first and last point
            mask[0]  = False
            mask[-1] = False

        return self._partitioned_newton(points, u, u_min, u_max, mask)

    @property
    def count(self) -> int:
        if self._max_param is None:
            self._init_bspline()
        return self._count

    @property
    def kv(self) -> np.ndarray:
        """Unpadded knot vector.

        scipy.interpolate.BSpline of degree n with m control points has
        m + n + 1 knots, as defined in the NURBS book by Piegl and Tiller.
        Maya and other DCCs have curves defined as m + n - 1 knots.
        This is the layout the ``knots`` field takes, and with ``knots`` set
        it gives them back in their own range.
        """
        if self._max_param is None:
            self._init_bspline()
        if self.knots is not None:
            return self._lo + self._kv[1:-1] * self._scale
        return self._kv[1:-1]

    @property
    def geometry(self) -> np.ndarray:
        if self._max_param is None:
            self._init_bspline()
        return self._geometry

    @property
    def max_param(self) -> float:
        """
        End of the curve's parameter range: the end of the knots' range, or
        the span count without ``knots``. Ignores ``uniform`` and
        ``registered``.
        """
        if self._max_param is None:
            self._init_bspline()
        if self.knots is not None:
            return self._hi
        return self._max_param

    @property
    def min_param(self) -> float:
        """
        Start of the curve's parameter range: the start of the knots'
        range, or 0 without ``knots``. Ignores ``uniform`` and
        ``registered``.
        """
        if self._max_param is None:
            self._init_bspline()
        return self._lo

    @property
    def domain(self) -> tuple:
        """The curve's parameter range, (min_param, max_param), like Maya's knotDomain."""
        return (self.min_param, self.max_param)

    @property
    def cv(self) -> np.ndarray:
        """control vertices"""
        if self._max_param is None:
            self._init_bspline()

        return self.points[self.geometry]

    @property
    def length(self) -> float:
        """Returns the total arc length of the curve."""
        return self.get_length(fast=True)

    @property
    def total_length(self) -> float:
        """
        Returns the total arc length of the curve.

        If uniform=True, this is pre-computed and cached.
        Otherwise, computes it on demand.
        """
        if self._max_param is None:
            self._init_bspline()

        if self._total_length is not None:
            return self._total_length

        return self.get_length(fast=True)

    @property
    def control_point_params(self) -> np.ndarray:
        """
        Monotonically increasing parameter values for each control point,
        representing the closest point on the curve to each CP.

        Uses Greville abscissae (knot averages) as initial guesses and
        Newton-Raphson refinement constrained to non-overlapping partitions
        derived from midpoints between adjacent Greville points.

        For periodic curves, returns count + 1 values where the last
        value is the first plus the period (closing the loop).

        For open curves, the first and last values are clamped at
        min_param and max_param respectively.

        Returns
        -------
        np.ndarray
            Parameter values in the curve's parameter range ([0, 1] when
            uniform). Length is count for open curves, count + 1 for
            periodic.
        """
        if self._cp_params is not None:
            return self._cp_params

        if self._max_param is None:
            self._init_bspline()

        n  = self._count
        d  = self.degree
        kv = self._kv

        # Degree 1: each control point sits on a knot (an integer index on
        # uniform knots)
        if d <= 1:
            if self.knots is not None:
                self._cp_params = self._lo + kv[1 : n + 1 + int(self.periodic)] * self._scale
            elif self.periodic:
                self._cp_params = np.arange(n + 1, dtype=np.float64)
            else:
                self._cp_params = np.linspace(
                    0, float(self._max_param), n, dtype=np.float64
                )
            return self._cp_params

        # Greville abscissae: xi_i = mean(kv[i+1 : i+d+1])
        # These are the parameter "center of mass" of each basis function
        # and provide optimal initial guesses for closest-point search.
        cumsum   = np.concatenate([[0], np.cumsum(kv)])
        greville = (cumsum[d + 1 : d + 1 + n] - cumsum[1 : 1 + n]) / d

        # For open curves, clamp endpoints
        if not self.periodic:
            greville[0]  = 0.0
            greville[-1] = float(self._max_param)

        # Non-overlapping search bounds from midpoints of adjacent Greville points.
        # This partitions the parameter space so each CP's Newton search
        # cannot cross into a neighbor's region, guaranteeing monotonicity.
        mids  = (greville[:-1] + greville[1:]) / 2.0
        u_min = np.empty(n, dtype=np.float64)
        u_max = np.empty(n, dtype=np.float64)

        if self.periodic:
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
            u_max[-1]  = float(self._max_param)

        # Skip refinement for clamped endpoints on open curves
        mask = np.ones(n, dtype=bool)
        if not self.periodic:
            mask[0]  = False
            mask[-1] = False

        u = self._partitioned_newton(self.points[:n], greville.copy(), u_min, u_max, mask)

        if self.periodic:
            u = u % self._max_param
            u = self._denormalize_u(u)

            if self.uniform:
                start, period = 0.0, 1.0
            else:
                start, period = self._lo, self._hi - self._lo

            # For registered curves, CP 0 sits at the registration boundary.
            # If Newton landed slightly on the wrong side, it wraps to near
            # the end of the period instead of near its start. Correct this.
            if self.registered and u[0] - start > period / 2:
                u[0] -= period

            # Restore monotonicity in user space
            for i in range(1, n):
                if u[i] < u[i - 1]:
                    u[i:] += period
                    break

            u = np.append(u, u[0] + period)
        else:
            u = self._denormalize_u(u)

        self._cp_params = u
        return self._cp_params

    def _partitioned_newton(self, queries, u, u_min, u_max, mask):
        """
        Closest-point parameters by Newton-Raphson, each confined to its
        own partition.

        Every query only searches [u_min, u_max], so it cannot latch onto
        another part of the curve that happens to pass closer. Used by
        ``control_point_params`` and ``fit()``.

        Parameters
        ----------
        queries : np.ndarray
            Points to project (n, dims).
        u : np.ndarray
            Starting native parameters (n,); on periodic curves they may
            run past [0, spans], the curve is evaluated modulo it.
        u_min, u_max : np.ndarray
            Partition bounds (n,), native space.
        mask : np.ndarray
            Which queries to refine (n,), bool; the others keep their u.

        Returns
        -------
        np.ndarray
            Refined native parameters (n,), unwrapped like ``u``.
        """
        der1 = self._spl.derivative(1)
        der2 = self._spl.derivative(2) if self.degree >= 2 else None

        if mask.any():
            for _ in range(10):
                u_eval = u % self._max_param if self.periodic else u

                C     = self._spl(u_eval)
                Cp    = der1(u_eval)
                Cpp   = der2(u_eval) if der2 is not None else np.zeros_like(Cp)

                diff  = C - queries
                f     = np.einsum("ij,ij->i", diff, Cp)
                fp    = np.einsum("ij,ij->i", Cp, Cp) + np.einsum("ij,ij->i", diff, Cpp)
                fp    = np.where(np.abs(fp) < 1e-14, 1e-14, fp)

                du    = np.where(mask, -f / fp, 0.0)
                du    = np.clip(du, -0.5, 0.5)

                u_new = np.clip(u + du, u_min, u_max)

                if np.max(np.abs(u_new[mask] - u[mask])) < 1e-10:
                    u = u_new
                    break
                u = u_new

        return u

    def get_length(self, fast: bool = True, samples: int = 100) -> float:
        """
        Approximates the length of the curve.

        Parameters
        ----------
        fast : bool
            If True (default), uses fast approximation by sampling.
            If False, uses precise scipy integration.
        samples : int
            Number of samples per control point for fast mode.

        Returns
        -------
        float
            Approximate arc length of the curve.
        """
        if self._max_param is None:
            self._init_bspline()

        # If we already computed total length (from arc-length table), use it
        if self._total_length is not None:
            return self._total_length

        # Fast approximation by sampling and summing segment lengths
        if fast:
            n = (self._count + self.periodic) * samples
            u = np.linspace(0, self._max_param, n)
            p = self._spl(u)
            d = np.diff(p, axis=0)
            d = np.einsum("...i,...i", d, d) ** 0.5
            return np.sum(d)

        # Precise approximation via integration (~1e-8 error), but slower
        def _deriv(x):
            return self._spl.derivative()(x)

        def _curve_length(x):
            return np.linalg.norm(_deriv(x))

        return si.quad(_curve_length, 0, self._max_param)[0]

    def basis(self, u, collapse=False, use_numba=None, _native=False):
        """
        Return the B-spline basis functions at parameter u.

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s). By default, expected in user space:
            - If uniform=True: [0, 1] range
            - If uniform=False: [min_param, max_param] range
            - If registered=True: offset is applied internally
        collapse : bool
            If True and periodic, collapse the basis for skin weights.
        use_numba : bool or None
            If None, uses self.use_numba. Otherwise overrides.
        _native : bool
            Internal flag. If True, skip normalization (u is already
            in native parameter space). Used by sample() to avoid
            double-applying the registration offset.

        Returns
        -------
        np.ndarray
            Basis function values (n_samples, n_control_points).
        """
        if self._max_param is None:
            self._init_bspline()

        # Convert from user space to native parameter space
        # Skip if _native=True (caller already provides native params)
        if not _native:
            u = self._normalize_u(u)

            if self.periodic:
                u = u % self._max_param

        u = np.atleast_1d(u).astype(np.float64)

        # Choose computation method
        if use_numba is None:
            use_numba = self.use_numba

        if use_numba:
            b = _compute_basis_parallel(u, self._kv, self.cv.shape[0], self.degree)
        else:
            b = compute_basis(u, self._kv, self.cv.shape[0], self.degree)

        # reformat for skin weights
        if collapse and self.periodic:
            b_ = b[:, np.arange(self.cv.shape[0] - self.degree)]
            b_[:, : self.degree] += b[:, -self.degree :]
            return b_

        return b

    def sample(
        self,
        obj,
        initial_samples  = 20,
        max_newton_iters = 10,
        tolerance        = 1e-10,
    ):
        """
        Projects points onto the curve using Newton-Raphson refinement.

        Algorithm:
        1. Coarse sampling to find initial parameter guesses (single KD-tree)
        2. Newton-Raphson iteration to refine to true closest point

        The Newton-Raphson method solves for the parameter u where the
        vector from query point Q to curve point C(u) is perpendicular
        to the tangent C'(u):

            f(u) = (C(u) - Q) * C'(u) = 0

        This gives quadratic convergence (~3-5 iterations vs ~10+ for
        the previous interval refinement approach).

        When use_numba=True (default), the Newton-Raphson refinement is
        parallelized across all CPU cores using Numba, providing significant
        speedup for large numbers of query points.

        Parameters
        ----------
        obj : array-like or object with .points attribute
            Query points to project onto the curve.
        initial_samples : int
            Number of samples per control point for initial guess.
        max_newton_iters : int
            Maximum Newton-Raphson iterations per query point.
        tolerance : float
            Convergence tolerance for parameter change.

        Returns
        -------
        SampleData
            Contains projected points, tangents, distances, params, and basis.
            params are in [0, 1] if uniform=True, else [min_param, max_param].
        """
        if self._max_param is None:
            self._init_bspline()

        if hasattr(obj, "points"):
            queries = np.asarray(obj.points, dtype=np.float64)
        else:
            queries = np.asarray(obj, dtype=np.float64)

        # Ensure 2D array
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        # === Phase 1: Coarse sampling for initial guesses ===
        n_coarse = initial_samples * self._count
        u_coarse = np.linspace(0, self._max_param, n_coarse, endpoint=not self.periodic)

        # Evaluate curve at coarse samples
        pts_coarse = self._spl(u_coarse)

        # Build KD-tree for fast nearest neighbor (ONLY ONCE!)
        tree = cKDTree(pts_coarse)
        _, indices = tree.query(queries)

        # Initial parameter guesses
        u_init = u_coarse[indices].astype(np.float64)

        # === Phase 2: Newton-Raphson refinement ===
        if self.use_numba:
            # Use parallel Numba kernel
            u, points, tangents = _newton_closest_point_parallel(
                queries,
                u_init,
                self._kv.astype(np.float64),
                self._cv_f64,
                self.degree,
                float(self._max_param),
                self.periodic,
                max_newton_iters,
                tolerance,
            )
        else:
            # Fallback to scipy-based Newton-Raphson
            u, points, tangents = self._newton_scipy(
                queries, u_init, max_newton_iters, tolerance
            )

        # Compute distances
        distances = np.linalg.norm(queries - points, axis=1)

        # Convert to user parameter space
        # This handles uniform (returns [0,1]) and registered offset
        params = self._denormalize_u(u)

        # derivatives with respect to the knots' parameter, as in Maya
        if self.knots is not None:
            tangents = tangents / self._scale

        # Compute basis at native parameters (skip normalization to avoid
        # double-applying registration offset)
        basis = self.basis(u, collapse=True, use_numba=self.use_numba, _native=True)

        return SampleData(
            points    = points,
            tangents  = tangents,
            distances = distances,
            params    = params,
            basis     = basis,
        )

    def _newton_scipy(self, queries, u_init, max_iters, tolerance):
        """
        Newton-Raphson refinement using scipy BSpline evaluation.

        Fallback when use_numba=False.

        Parameters
        ----------
        queries : np.ndarray
            Query points (n, dims).
        u_init : np.ndarray
            Initial parameter guesses (n,).
        max_iters : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        tuple
            (u, points, tangents) arrays.
        """
        u = u_init.copy()

        # Get first and second derivatives
        der1 = self._spl.derivative(1)
        der2 = self._spl.derivative(2)

        for _ in range(max_iters):
            # Evaluate curve and derivatives at current u
            C              = self._spl(u)
            C_prime        = der1(u)
            C_double_prime = der2(u)

            # Vector from query to curve point
            diff = C - queries

            # f(u) = (C(u) - Q) * C'(u)
            f = np.einsum("ij,ij->i", diff, C_prime)

            # f'(u) = |C'(u)|^2 + (C(u) - Q) * C''(u)
            f_prime = np.einsum("ij,ij->i", C_prime, C_prime) + np.einsum(
                "ij,ij->i", diff, C_double_prime
            )

            # Avoid division by zero
            f_prime = np.where(np.abs(f_prime) < 1e-14, 1e-14, f_prime)

            # Newton step with trust region
            du = -f / f_prime
            du = np.clip(du, -0.5, 0.5)

            # Update parameters
            u_new = u + du

            # Handle bounds
            if self.periodic:
                u_new = u_new % self._max_param
                # Periodic distance: shortest signed step around the loop.
                actual_step = u_new - u
                actual_step = np.where(
                    actual_step > self._max_param / 2,
                    actual_step - self._max_param,
                    actual_step,
                )
                actual_step = np.where(
                    actual_step < -self._max_param / 2,
                    actual_step + self._max_param,
                    actual_step,
                )
            else:
                u_new       = np.clip(u_new, 0, self._max_param)
                actual_step = u_new - u

            # Check convergence on the ACTUAL change (post-clip / post-wrap),
            # not the requested du. Otherwise queries whose true closest
            # point lies outside the domain oscillate forever.
            if np.max(np.abs(actual_step)) < tolerance:
                u = u_new
                break

            u = u_new

        # Final evaluation
        points   = self._spl(u)
        tangents = der1(u)

        return u, points, tangents


# ------------------------------- Knots and fit ------------------------------- #


def _knot_vector(knots, count, degree, periodic):
    """
    Knot vector in scipy's layout from knots in Maya's layout, and the
    knots' own parameter range.

    The knot vector is rescaled to the curve's native range [0, spans], the
    one uniform knots have; the range returned is what users see. Periodic
    curves rebuild the knots outside that range from the ones inside it,
    so the seam stays smooth.

    Parameters
    ----------
    knots : array-like
        Knots in Maya's layout: count + degree - 1 values (open) or
        count + 2 * degree - 1 (periodic).
    count : int
        Number of control points, without the periodic wrap.
    degree : int
        Degree of the curve.
    periodic : bool
        Whether the curve is closed.

    Returns
    -------
    tuple
        (knot vector (count + degree + 1,) for open curves or
        (count + 2 * degree + 1,) for periodic ones, range start, range end).
    """
    knots = np.asarray(knots, dtype=np.float64).ravel()
    if degree < 1:
        raise ValueError(f"knots need a degree of 1 or more, got {degree}")

    spans = count if periodic else count - degree
    if spans < (degree if periodic else 1):
        kind = "periodic" if periodic else "open"
        raise ValueError(
            f"{count} control points are too few for a {kind} curve of "
            f"degree {degree}"
        )

    size = spans + 2 * degree - 1
    if knots.shape[0] != size:
        raise ValueError(
            f"{count} control points of degree {degree} take {size} knots, "
            f"got {knots.shape[0]}"
        )
    if np.any(np.diff(knots) < 0.0):
        raise ValueError("knots must never decrease")

    lo, hi = knots[degree - 1], knots[degree - 1 + spans]
    if not hi > lo:
        raise ValueError("knots must cover a parameter range longer than zero")
    if lo != 0.0 or hi != spans:
        knots = (knots - lo) / (hi - lo) * spans

    if periodic:
        full = _knots_from_edges(knots[degree - 1 : degree + spans], degree, True)
    else:
        full = np.concatenate([knots[:1], knots, knots[-1:]])
    return full, float(lo), float(hi)


def _fit_knots(points, count, degree, periodic, tolerance, stack=(0, 0)):
    """
    Knot vector (scipy's layout) and fit point parameters for ``fit()``,
    both in the curve's native range [0, spans], for at most the exact
    control point count (``_exact_count``).

    The points are spaced by the distance between them, like Maya's EP
    curves. At the exact count the knots land on the points; a stacked
    end that needs more control points than Maya's end rule splits its
    end span evenly, once per extra one. Periodic curves of even degree
    shift the knots half a span, so each point sits mid-span. Open
    curves with fewer control points, but at least one free control
    point per point, average each run of consecutive point parameters
    into a knot (the NURBS Book's averaging), each stacked end counting
    once per stacked control point; this keeps exact interpolation well
    conditioned. Fewer resample the spacing at evenly spaced fractional
    point indices.

    Parameters
    ----------
    points : np.ndarray
        Fit points (N, dims).
    count : int
        Number of control points, without the periodic wrap.
    degree : int
        Degree of the curve.
    periodic : bool
        Whether the curve is closed.
    tolerance : float
        Distance under which two consecutive points count as one.
    stack : tuple of int
        Stacked control points at the start and end, 0 for none.

    Returns
    -------
    tuple
        (knot vector, fit point parameters (N,), total distance along the
        points, which a periodic curve measures around its loop).
    """
    loop   = np.vstack([points, points[:1]]) if periodic else points
    chords = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    repeat = np.flatnonzero(chords <= tolerance)
    if repeat.size:
        first = int(repeat[0])
        last  = points.shape[0] - 2
        hint  = ""
        if not periodic and first in (0, last):
            stack = "(2, 0)" if first == 0 else "(0, 2)"
            hint  = f"; to stack the end control points, pass collapse={stack} instead"
        raise ValueError(
            f"fit points must not repeat: points {first} and "
            f"{(first + 1) % points.shape[0]} share a position{hint}"
        )
    along = np.concatenate([[0.0], np.cumsum(chords)])

    n     = points.shape[0]
    steps = chords.shape[0]
    spans = count if periodic else count - degree
    head, tail = stack

    if not periodic and count >= _exact_count(n, degree, False, stack):
        # On the points; a stack's extra conditions split its end span
        edges = _split_span(along, 0, 1 + max(0, head - 1 - degree // 2))
        edges = _split_span(
            edges, edges.shape[0] - 2, 1 + max(0, tail - 1 - (degree - 1) // 2)
        )
    elif not periodic and count >= n + max(0, head - 1) + max(0, tail - 1):
        # Each interior knot averages `width` consecutive interior sites;
        # a stacked end is a site once per stacked control point
        sites = np.concatenate(
            [np.repeat(along[:1], max(1, head)), along[1:-1], np.repeat(along[-1:], max(1, tail))]
        )
        width = sites.shape[0] + degree - count
        inner = sites[1:-1]
        if width > 1:
            inner = np.array(
                [inner[i : i + width].mean() for i in range(inner.shape[0] - width + 1)]
            )
        edges = np.concatenate([sites[:1], inner, sites[-1:]])
    else:
        shift = 0.5 if periodic and degree % 2 == 0 else 0.0
        where = (np.arange(spans + 1) + shift) * steps / spans

        if periodic:
            # A second lap, for the half-span shift
            laps  = np.concatenate([along, along[1:] + along[-1]])
            edges = np.interp(where, np.arange(2 * steps + 1), laps)
        else:
            edges = np.interp(where, np.arange(steps + 1), along)

    start  = edges[0]
    length = edges[-1] - start
    edges  = (edges - start) / length * spans
    params = (along[: points.shape[0]] - start) / length * spans

    if periodic:
        params = params % spans

    return _knots_from_edges(edges, degree, periodic), params, length


def _knots_from_edges(edges, degree, periodic):
    """
    Knot vector (scipy's layout) from the knots inside the parameter
    range: clamped on open curves, wrapped around on periodic ones.

    Parameters
    ----------
    edges : np.ndarray
        Knots from 0 to the span count, one per span join (spans + 1,).
    degree : int
        Degree of the curve.
    periodic : bool
        Whether the curve is closed.

    Returns
    -------
    np.ndarray
        Knot vector (spans + 2 * degree + 1,).
    """
    spans = edges.shape[0] - 1
    if periodic:
        return np.concatenate(
            [edges[spans - degree : spans] - spans, edges, edges[1 : degree + 1] + spans]
        )
    return np.concatenate([np.zeros(degree), edges, np.full(degree, float(spans))])


def _fit_points(points, full, params, count, degree, periodic, end_rule, stack=(0, 0)):
    """
    Control points for ``fit()``, solved in priority order: each level
    only uses the freedom the levels before it leave.

    0. Stacked control points are one unknown each stack, so they come
       out exactly equal.
    1. Open curves start and end on the first and last point.
    2. Pass through the points, as close as possible with too few
       control points.
    3. With ``end_rule``, Maya's end conditions on open curves, at each
       end that isn't stacked: derivatives 2 to 1 + ceil((degree - 1) / 2)
       are zero at the start, 2 to 1 + floor((degree - 1) / 2) at the end.
    4. Smoothest curve for whatever freedom is left: least squared second
       derivative (first on linear curves), integrated by Gauss-Legendre
       quadrature over every span.

    Parameters
    ----------
    points : np.ndarray
        Fit points (N, dims).
    full : np.ndarray
        Knot vector in scipy's layout.
    params : np.ndarray
        Parameter of each fit point (N,).
    count : int
        Number of control points, without the periodic wrap.
    degree : int
        Degree of the curve.
    periodic : bool
        Whether the curve is closed.
    end_rule : bool
        Whether to apply Maya's end conditions (level 3).
    stack : tuple of int
        Stacked control points at the start and end, 0 for none.

    Returns
    -------
    np.ndarray
        Control points (count, dims).
    """
    dims   = points.shape[1]
    spans  = count if periodic else count - degree
    levels = []

    if not periodic:
        levels.append(_end_pins(count, points[[0, -1]]))

    levels.append((_fit_basis(full, degree, count, periodic, params), points))

    if end_rule and not periodic:
        rows = []
        if not stack[0]:
            rows += [
                _fit_basis(full, degree, count, False, [0.0], order)
                for order in range(2, 2 + degree // 2)
            ]
        if not stack[1]:
            rows += [
                _fit_basis(full, degree, count, False, [float(spans)], order)
                for order in range(2, 2 + (degree - 1) // 2)
            ]
        if rows:
            levels.append((np.vstack(rows), np.zeros((len(rows), dims))))

    levels.append(_bending(full, degree, count, periodic, dims))
    return _layered_lstsq(levels, count, dims, _ties(count, stack))


def _insert_knots(cv, full, params, degree, periodic, extra, stack=(0, 0)):
    """
    The same curve with more control points: splits the longest span in
    two, ``extra`` times, then solves for the control points that
    reproduce the curve exactly (a refined knot vector holds it).

    Parameters
    ----------
    cv : np.ndarray
        Control points (count, dims), without the periodic wrap.
    full : np.ndarray
        Knot vector in scipy's layout.
    params : np.ndarray
        Curve parameters to carry over to the new parameter range.
    degree : int
        Degree of the curve.
    periodic : bool
        Whether the curve is closed.
    extra : int
        Number of knots to insert.
    stack : tuple of int
        Stacked control points at the start and end, kept exactly equal.

    Returns
    -------
    tuple
        (control points (count + extra, dims), knot vector, params in the
        new parameter range).
    """
    count = cv.shape[0]
    spans = count if periodic else count - degree
    edges = _split_longest(full[degree : degree + spans + 1], extra)

    # Rescale to the new parameter range [0, spans + extra]
    fine    = _knots_from_edges(edges / spans * (spans + extra), degree, periodic)
    spline  = BSpline(full, _wrap(cv, degree, periodic), degree)
    fine_cv = _project(spline, spans, fine, count + extra, degree, periodic, stack)
    return fine_cv, fine, params * ((spans + extra) / spans)


def _project(spline, span, full, count, degree, periodic, stack=(0, 0)):
    """
    Control points of the curve on knot vector ``full`` closest to
    ``spline``: the least squared distance between the two at the same
    relative parameter, integrated over the whole range. Open curves keep
    the spline's end points, and stacked control points come out exactly
    equal. Exact when the knots can hold the spline.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
        Curve to follow, on its native range [0, span].
    span : int
        Its span count.
    full : np.ndarray
        New knot vector in scipy's layout, on [0, spans].
    count : int
        Number of control points, without the periodic wrap.
    degree : int
        Degree of the new curve.
    periodic : bool
        Whether the curve is closed.
    stack : tuple of int
        Stacked control points at the start and end, 0 for none.

    Returns
    -------
    np.ndarray
        Control points (count, dims).
    """
    spans = count if periodic else count - degree
    scale = spans / span

    # Gauss nodes on every piece both curves are polynomial over, exact for
    # the squared distance
    breaks = np.unique(
        np.concatenate(
            [spline.t[spline.k : spline.k + span + 1] * scale, full[degree : degree + spans + 1]]
        )
    )
    at, weight = _span_quadrature(breaks, max(degree, spline.k))
    target = spline(at / scale)
    dims   = target.shape[1]

    levels = []
    if not periodic:
        levels.append(_end_pins(count, spline([0.0, float(span)])))
    basis = _fit_basis(full, degree, count, periodic, at)
    levels.append((basis * weight[:, None], target * weight[:, None]))
    levels.append(_bending(full, degree, count, periodic, dims))
    return _layered_lstsq(levels, count, dims, _ties(count, stack))


def _span_quadrature(edges, degree):
    """
    Gauss-Legendre nodes over every span longer than zero, exact for the
    polynomials a degree ``degree`` curve's squared derivatives make.

    Parameters
    ----------
    edges : np.ndarray
        Knots inside the parameter range (spans + 1,).
    degree : int
        Degree of the curve.

    Returns
    -------
    tuple
        (nodes, square roots of their weights), both (n,).
    """
    keep = edges[1:] > edges[:-1]
    lo, hi = edges[:-1][keep], edges[1:][keep]
    nodes, weights = np.polynomial.legendre.leggauss(degree + 1)
    half = (hi - lo)[:, None] / 2.0
    at   = (lo[:, None] + half * (nodes + 1.0)).ravel()
    return at, np.sqrt((half * weights).ravel())


def _fit_basis(full, degree, count, periodic, at, order=0):
    """
    Basis functions, or their derivatives, at parameter values.

    Periodic curves fold the wrapped control points back onto the ones
    they repeat, so there is one column per control point.

    Parameters
    ----------
    full : np.ndarray
        Knot vector in scipy's layout.
    degree : int
        Degree of the curve.
    count : int
        Number of control points, without the periodic wrap.
    periodic : bool
        Whether the curve is closed.
    at : array-like
        Parameter values (n,).
    order : int
        Derivative order, 0 for the basis functions themselves.

    Returns
    -------
    np.ndarray
        Values (n, count).
    """
    n_basis = full.shape[0] - degree - 1
    spline  = BSpline(full, np.eye(n_basis), degree)
    if order:
        spline = spline.derivative(order)

    values = spline(np.asarray(at, dtype=np.float64))
    if periodic:
        values[:, :degree] += values[:, count:]
        values = values[:, :count]
    return values


def _layered_lstsq(levels, size, dims, ties=None):
    """
    Least squares in priority order.

    Each level ``(a, b)`` minimises ``|a x - b|`` only over the solutions
    the levels before it left free, so a later level never spoils an
    earlier one. Whatever freedom is left after the last level takes the
    smallest solution.

    Parameters
    ----------
    levels : list of tuple
        ``(a, b)`` pairs: ``a`` (rows, size), ``b`` (rows, dims).
    size : int
        Number of unknowns.
    dims : int
        Number of right-hand sides, solved together.
    ties : np.ndarray, optional
        Which shared unknown each unknown is (size,), from ``_ties``:
        unknowns sharing one come out exactly equal.

    Returns
    -------
    np.ndarray
        Solution (size, dims).
    """
    if ties is not None:
        shared = np.eye(int(ties.max()) + 1)[ties]
        x      = _layered_lstsq([(a @ shared, b) for a, b in levels], shared.shape[1], dims)
        return x[ties]

    x    = np.zeros((size, dims))
    free = np.eye(size)
    for a, b in levels:
        if free.shape[1] == 0:
            break

        a_free = a @ free
        u, s, vt = np.linalg.svd(a_free, full_matrices=a_free.shape[0] < a_free.shape[1])
        if s.size == 0 or s[0] == 0.0:
            continue

        rank = int(np.sum(s > s[0] * max(a_free.shape) * np.finfo(np.float64).eps * 1e3))
        x    = x + free @ (vt[:rank].T @ ((u[:, :rank].T @ (b - a @ x)) / s[:rank, None]))
        free = free @ vt[rank:].T

    return x


def _as_int(value, name):
    """``value`` as an int, or ValueError when it isn't a whole number."""
    try:
        valid = int(value) == value and not isinstance(value, bool)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def _check_degree(degree):
    """``degree`` as an int of 1 or more, or ValueError."""
    degree = _as_int(degree, "degree")
    if degree < 1:
        raise ValueError(f"the degree must be 1 or more, got {degree}")
    return degree


def _check_count(count, degree, periodic, stack, hint):
    """ValueError when ``count`` control points are too few for the curve."""
    smallest = max(3, degree) if periodic else max(degree + 1, stack[0] + stack[1])
    if count < smallest:
        stacked = f" with {stack[0]} and {stack[1]} stacked" if any(stack) else ""
        raise ValueError(
            f"{count} control points are too few for this degree {degree} "
            f"curve{stacked}, it needs {smallest}; {hint}"
        )


def _exact_count(n, degree, periodic, stack):
    """
    Control points a curve needs to pass exactly through ``n`` points:
    Maya's EP count, n + degree - 1 open and n periodic, plus one per
    condition a stacked end adds beyond Maya's end rule. k stacked
    control points zero derivatives 1 to k - 1; the rule they replace
    zeroes ceil((degree - 1) / 2) derivatives at the start and
    floor((degree - 1) / 2) at the end.
    """
    if periodic:
        return n
    head, tail = stack
    extra = max(0, head - 1 - degree // 2) + max(0, tail - 1 - (degree - 1) // 2)
    return n + degree - 1 + extra


def _ties(count, stack):
    """
    Which shared unknown each control point is (count,): one per stacked
    end, one per other control point. None without stacks.
    """
    head, tail = stack
    if not head and not tail:
        return None
    ties = np.arange(count)
    if head:
        ties = np.maximum(ties - (head - 1), 0)
    if tail:
        ties[count - tail :] = ties[count - tail]
    return ties


def _end_pins(count, ends):
    """Least squares level holding the first and last control points on ``ends``."""
    pins        = np.zeros((2, count))
    pins[0, 0]  = 1.0
    pins[1, -1] = 1.0
    return pins, ends


def _bending(full, degree, count, periodic, dims):
    """
    Least squares level for the smoothest curve: squared second derivative
    (first on linear curves) integrated over every span.
    """
    spans = count if periodic else count - degree
    at, weight = _span_quadrature(full[degree : degree + spans + 1], degree)
    bending = _fit_basis(full, degree, count, periodic, at, min(2, degree))
    return bending * weight[:, None], np.zeros((at.shape[0], dims))


def _wrap(cv, degree, periodic):
    """Control points with a periodic curve's first ``degree`` repeated at the end."""
    if not periodic:
        return cv
    return cv[np.concatenate([np.arange(cv.shape[0]), np.arange(degree) % cv.shape[0]])]


def _split_span(edges, span, pieces):
    """Knots with span ``span`` split evenly into ``pieces``."""
    if pieces <= 1:
        return edges
    lo, hi = edges[span], edges[span + 1]
    inside = lo + (hi - lo) * np.arange(1, pieces) / pieces
    return np.concatenate([edges[: span + 1], inside, edges[span + 1 :]])


def _split_longest(edges, extra):
    """Knots with the longest span split in two, ``extra`` times."""
    edges = list(edges)
    for _ in range(extra):
        i = int(np.argmax(np.diff(edges)))
        edges.insert(i + 1, (edges[i] + edges[i + 1]) / 2.0)
    return np.array(edges)


def _bend_edges(spline, span, spans):
    """
    ``spans + 1`` knots on [0, span], denser where the curve bends: equal
    shares of sqrt(|C''| |C'|), the classic optimal knot density, plus a
    15% floor so straight stretches still get knots. A curve that never
    bends gets knots spaced by length.
    """
    u       = np.linspace(0.0, span, 64 * max(span, spans, 64) + 1)
    speed   = np.linalg.norm(spline.derivative(1)(u), axis=1)

    density = np.zeros_like(speed)
    if spline.k >= 2:
        density = np.sqrt(np.linalg.norm(spline.derivative(2)(u), axis=1) * speed)
    density = density + 0.15 * density.mean()
    if not density.mean() > 0.0:
        density = speed
    if not density.mean() > 0.0:
        return np.linspace(0.0, span, spans + 1)

    total     = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2.0 * np.diff(u))])
    edges     = np.interp(np.linspace(0.0, total[-1], spans + 1), total, u)
    edges[0]  = 0.0
    edges[-1] = float(span)
    return edges


def _curve_distance(a, a_span, b, b_span):
    """
    Largest distance between two curves, both ways (Hausdorff): dense
    samples of each, measured to the other's sampled polyline.

    Parameters
    ----------
    a, b : scipy.interpolate.BSpline
        The curves, on their native ranges.
    a_span, b_span : int
        Their span counts.

    Returns
    -------
    float
        The distance.
    """
    samples = 64 * max(a_span, b_span, 64) + 1
    pa      = a(np.linspace(0.0, a_span, samples))
    pb      = b(np.linspace(0.0, b_span, samples))
    return max(_polyline_distance(pa, pb), _polyline_distance(pb, pa))


def _polyline_distance(points, line):
    """Largest distance from ``points`` to the polyline through ``line``."""
    nearest = cKDTree(line).query(points)[1]
    best    = np.full(points.shape[0], np.inf)
    for segment in (nearest - 1, nearest):
        segment = np.clip(segment, 0, line.shape[0] - 2)
        start   = line[segment]
        along   = line[segment + 1] - start
        length  = np.einsum("ij,ij->i", along, along)
        t       = np.einsum("ij,ij->i", points - start, along) / np.where(length > 0.0, length, 1.0)
        t       = np.clip(t, 0.0, 1.0)
        best    = np.minimum(best, np.linalg.norm(points - (start + t[:, None] * along), axis=1))
    return float(best.max())
