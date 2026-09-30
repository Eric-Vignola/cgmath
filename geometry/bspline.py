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
        If False (default), uses native parameter space [0, max_param].
    use_numba : bool
        If True (default), uses optimized Numba kernels for evaluation.
        If False, uses scipy.interpolate.BSpline.
    registered : bool
        If True, u=0 on periodic curves aligns with the curve point nearest
        to points[0]. The offset is CV 0's Greville abscissa (the average
        of the knots under its basis function), which on uniform knots is
        offset = max_param - (degree - 1) / 2.
        Has no effect on non-periodic curves.
    arc_length_samples : int
        Number of samples used to build the arc-length lookup table.
        Higher values give more accurate arc-length parameterization.
        Default is 1000.
    knots : np.ndarray, optional
        Knot vector in Maya's layout, the one ``kv`` returns:
        count + degree - 1 values on open curves, count + 2 * degree - 1
        on periodic ones. None (default) gives uniform knots. The values
        are rescaled so the parameter range stays [0, max_param]; only
        their spacing shapes the curve. Periodic curves rebuild the knots
        outside that range from the ones inside it, so the seam stays
        smooth. Set by ``fit()``, cleared by ``open()`` and ``close()``;
        call ``invalidate()`` after changing it by hand.
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

        # Custom knots replace the uniform ones
        if self.knots is not None:
            self._kv = _knot_vector(self.knots, self._count, self.degree, self.periodic)

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
            Native parameter value(s) in [0, max_param].
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
            Native parameter value(s) in [0, max_param].

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

        If registered and periodic (non-uniform), offsets u so that
        u=0 aligns with the curve point nearest points[0].

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

        if self.registered and self.periodic and not self.uniform:
            u = u + self._offset

        if self.uniform:
            u = self._arc_length_to_param(u % 1.0 if self.periodic else u)

        return u

    def _denormalize_u(self, u):
        """
        Convert native parameter space to user parameter space.

        If uniform, converts native [0, max_param] to arc-length [0, 1].
        When also registered, native params are unwrapped to the table's
        domain [offset, offset+max_param] before lookup.

        If registered and periodic (non-uniform), removes the offset
        so that the returned values are relative to points[0], wrapped
        into [0, max_param).

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
        elif self.registered and self.periodic:
            # A value a hair below 0 wraps to max_param itself; that is 0
            u = (u - self._offset) % self._max_param
            u = np.where(u >= self._max_param, 0.0, u)

        return u

    def compute(self, u):
        """
        Computes points and tangents on curve at given u coordinate.

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s). If uniform=True, expected in [0, 1] range.
            If uniform=False, expected in [0, max_param] range.
            If registered=True on periodic curves, u=0 aligns with points[0].

            For open (non-periodic) curves, values outside the natural domain
            are linearly extrapolated along the endpoint tangent direction:
              - uniform=True: outside [0, 1]; one unit of u corresponds to one
                full curve length (`total_length`) of physical motion.
              - uniform=False: outside [0, max_param]; one unit of u corresponds
                to one unit of native-tangent magnitude.
            The returned tangent in the extrapolated region equals the endpoint
            native tangent (held constant during extrapolation).

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

        # For non-uniform open curves, clip native u to [0, max_param] so the
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

        return points, tangents

    def _apply_open_extrapolation(self, u_user, points, tangents):
        """
        Replace clamped endpoint values with linear extrapolation for open
        (non-periodic) curves when u_user is outside the natural domain.

        Natural domain depends on parameterization:
          - uniform=True:  [0, 1]
          - uniform=False: [0, max_param]

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
            u_high = 1.0
        else:
            u_high = float(self._max_param)

        out_low  = u_user < 0.0
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
        # the full curve length). In non-uniform mode, native u == user u, so
        # the velocity is just the native tangent itself.
        if self.uniform:
            scale = self.total_length

            def _velocity(t_native):
                norm = np.linalg.norm(t_native)
                if norm > 1e-14:
                    return scale * (t_native / norm)
                return np.zeros(n_dims, dtype=np.float64)
        else:

            def _velocity(t_native):
                return t_native

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
            # u_user[idx] is negative -> moves opposite to the start tangent,
            # extending the curve backward in space.
            delta         = u_user[idx][:, None]
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

    def rebuild(self) -> None:
        """
        Rebuild all cached data from current control points.

        Call this after modifying control points to update all
        internal caches immediately. This is equivalent to calling
        invalidate() followed by accessing any property.

        Example
        -------
        >>> curve.points[0] = [1.0, 2.0, 3.0]
        >>> curve.rebuild()  # Immediately rebuilds all caches
        """
        self.invalidate()
        self._init_bspline()

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
            self.rebuild()

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
            self.rebuild()

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

    def fit(self, points, resize: bool = True) -> np.ndarray:
        """
        Moves the control points so the curve passes through the points,
        and returns the curve parameter of each point.

        Builds the curve Maya's EP Curve Tool builds, which is also the
        spline IK curve when ``ikHandle`` doesn't simplify it
        (``simplifyCurve=False``): each point becomes an edit point, where
        two spans join, and the spans are spaced by the distance between
        the points. Sets ``points`` (to a new array) and ``knots``; keeps
        ``degree`` and ``periodic``.

        Parameters
        ----------
        points : array-like
            Points to pass through, in order (N, dims). A periodic curve
            closes the loop itself, so a last point repeating the first
            is dropped.
        resize : bool
            If True (default), the curve takes the control point count
            Maya uses, N + degree - 1 when open and N when periodic, and
            passes exactly through every point. If False, it keeps its
            count. With more control points it is still Maya's curve,
            with knots inserted: the same shape. With fewer it passes
            exactly through the points while it has at least one control
            point per point, as smoothly as it can, and as close as it
            can below that; open curves keep their ends on the first and
            last point. An empty curve is always resized.

        Returns
        -------
        np.ndarray
            The curve parameter of each point (N,), what ``sample(points)``
            returns: [0, max_param], or [0, 1] with ``uniform``; periodic
            curves wrap into the range, with u=0 at the registration
            point when ``registered``. The point sits exactly there when
            the curve passes through it; on a best fit it is the closest
            point, searched only half way to the neighbouring points so it
            can't latch onto another part of the curve.

        Raises
        ------
        ValueError
            A degree below 1; too few points (2 on an open curve, 3 and
            at least ``degree`` on a periodic one); two consecutive points
            at the same position; or, with resize=False, a count too
            small for the degree.

        Notes
        -----
        Open curves need degree - 1 extra conditions. As in Maya,
        derivatives 2 to 1 + ceil((degree - 1) / 2) are zero at the start
        and 2 to 1 + floor((degree - 1) / 2) at the end: a cubic's second
        derivative is zero at both ends, a quadratic's only at the start.
        Periodic curves need none. On even degrees each point sits
        mid-span, since points on the span joins have no solution when
        their count is even. Maya's own periodic EP curve spaced by
        distance has clamped knots and a kink at the seam; this one
        closes smoothly.

        With resize=False, a high degree with barely one control point
        per point can overshoot between the points: exact interpolation
        leaves it no freedom to stay smooth.

        Example
        -------
        >>> curve = BSplineData()
        >>> u = curve.fit(joint_positions)
        >>> curve.count == len(joint_positions) + 2
        True
        >>> np.allclose(curve.compute(u)[0], joint_positions)
        True
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(
                f"fit points must be an (N, dims) array, got shape {points.shape}"
            )

        degree = int(self.degree)
        if degree < 1:
            raise ValueError(f"fit needs a degree of 1 or more, got {degree}")

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

        # Maya's control point count
        maya = n if self.periodic else n + degree - 1

        if resize or len(self.points) == 0:
            count = maya
        else:
            count    = len(self.points)
            smallest = max(3, degree) if self.periodic else degree + 1
            if count < smallest:
                raise ValueError(
                    f"{count} control points are too few for this degree "
                    f"{degree} curve, it needs {smallest}; fit with resize=True"
                )

        # Maya's curve, or the best one with fewer control points
        fitted = min(count, maya)
        full, params = _fit_knots(points, fitted, degree, self.periodic, tolerance)
        cv = _fit_points(
            points, full, params, fitted, degree, self.periodic, end_rule=count >= maya
        )

        # More control points: Maya's curve with knots inserted, same shape
        if count > maya:
            cv, full, params = _insert_knots(
                cv, full, params, degree, self.periodic, count - maya
            )

        self.points = cv
        self.knots  = full[1:-1].copy()
        self.rebuild()

        # Each point's parameter, refined to its closest point on the curve
        # without leaving its own partition (half way to its neighbours).
        # It only moves on a best fit; elsewhere the curve passes through.
        spans = self._max_param
        u     = params.copy()
        if self.periodic:
            wraps = np.flatnonzero(np.diff(u) < 0)
            if wraps.size:
                u[wraps[0] + 1 :] += spans

        mids  = (u[:-1] + u[1:]) / 2.0
        u_min = np.concatenate([u[:1], mids])
        u_max = np.concatenate([mids, u[-1:]])
        mask  = np.ones(n, dtype=bool)
        if self.periodic:
            gap       = u[0] + spans - u[-1]
            u_min[0]  = u[0] - gap / 2.0
            u_max[-1] = u[-1] + gap / 2.0
        else:
            # the ends sit exactly on the first and last point
            mask[0]  = False
            mask[-1] = False

        u = self._partitioned_newton(points, u, u_min, u_max, mask)

        # Same space as sample(): wrapped on periodic curves, then uniform
        # and registered applied
        if self.periodic:
            u = u % spans
            u = np.where(u >= spans, 0.0, u)
        u = self._denormalize_u(u)

        return np.append(u, u[:1]) if closing else u

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
        This is the layout the ``knots`` field takes.
        """
        if self._max_param is None:
            self._init_bspline()
        return self._kv[1:-1]

    @property
    def geometry(self) -> np.ndarray:
        if self._max_param is None:
            self._init_bspline()
        return self._geometry

    @property
    def max_param(self) -> int:
        if self._max_param is None:
            self._init_bspline()
        return self._max_param

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
        Monotonically increasing native parameter values for each control
        point, representing the closest point on the curve to each CP.

        Uses Greville abscissae (knot averages) as initial guesses and
        Newton-Raphson refinement constrained to non-overlapping partitions
        derived from midpoints between adjacent Greville points.

        For periodic curves, returns count + 1 values where the last
        value is the first + max_param (closing the loop).

        For open curves, the first and last values are clamped at 0
        and max_param respectively.

        Returns
        -------
        np.ndarray
            Parameter values in native space. Length is count for open
            curves, count + 1 for periodic.
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
                self._cp_params = kv[1 : n + 1 + int(self.periodic)].copy()
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

            period = 1.0 if self.uniform else float(self._max_param)

            # For registered curves, CP 0 sits at the registration boundary.
            # If Newton landed slightly on the wrong side, it wraps to near
            # the end of the period instead of near 0. Correct this.
            if self.registered and u[0] > period / 2:
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
            run past [0, max_param], the curve is evaluated modulo it.
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
            - If uniform=False: [0, max_param] range
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
            params are in [0, 1] if uniform=True, else [0, max_param].
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
    Knot vector in scipy's layout from knots in Maya's layout.

    Rescales them so the parameter range is [0, max_param], as with
    uniform knots. Periodic curves rebuild the knots outside that range
    from the ones inside it, so the seam stays smooth.

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
    np.ndarray
        Knot vector (count + degree + 1,) for open curves,
        (count + 2 * degree + 1,) for periodic ones.
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
        return _knots_from_edges(knots[degree - 1 : degree + spans], degree, True)
    return np.concatenate([knots[:1], knots, knots[-1:]])


def _fit_knots(points, count, degree, periodic, tolerance):
    """
    Knot vector (scipy's layout) and fit point parameters for ``fit()``,
    both in the curve's parameter range [0, max_param], for at most the
    control point count Maya uses (points + degree - 1 open, points
    periodic).

    The points are spaced by the distance between them, like Maya's EP
    curves. At Maya's count the knots land on the points; periodic curves
    of even degree shift them half a span, so each point sits mid-span.
    Open curves with fewer control points, but at least one per point,
    average each run of consecutive point parameters into a knot (the
    NURBS Book's averaging), which keeps exact interpolation well
    conditioned. Fewer control points than points resample the spacing
    at evenly spaced fractional point indices.

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

    Returns
    -------
    tuple
        (knot vector, fit point parameters (N,)).
    """
    loop   = np.vstack([points, points[:1]]) if periodic else points
    chords = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    if np.any(chords <= tolerance):
        raise ValueError(
            "fit points must not repeat: two consecutive points share a position"
        )
    along = np.concatenate([[0.0], np.cumsum(chords)])

    n     = points.shape[0]
    steps = chords.shape[0]
    spans = count if periodic else count - degree

    if not periodic and count >= n:
        # Each interior knot averages `width` consecutive interior
        # parameters; a width of 1 puts the knots on the points
        width = n + degree - count
        inner = along[1:-1]
        if width > 1:
            inner = np.array(
                [inner[i : i + width].mean() for i in range(inner.shape[0] - width + 1)]
            )
        edges = np.concatenate([along[:1], inner, along[-1:]])
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

    return _knots_from_edges(edges, degree, periodic), params


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


def _fit_points(points, full, params, count, degree, periodic, end_rule):
    """
    Control points for ``fit()``, solved in priority order: each level
    only uses the freedom the levels before it leave.

    1. Open curves start and end on the first and last point.
    2. Pass through the points, as close as possible with too few
       control points.
    3. With ``end_rule``, Maya's end conditions on open curves:
       derivatives 2 to 1 + ceil((degree - 1) / 2) are zero at the start,
       2 to 1 + floor((degree - 1) / 2) at the end.
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

    Returns
    -------
    np.ndarray
        Control points (count, dims).
    """
    dims   = points.shape[1]
    spans  = count if periodic else count - degree
    levels = []

    if not periodic:
        pins        = np.zeros((2, count))
        pins[0, 0]  = 1.0
        pins[1, -1] = 1.0
        levels.append((pins, points[[0, -1]]))

    levels.append((_fit_basis(full, degree, count, periodic, params), points))

    if end_rule and not periodic:
        rows = [
            _fit_basis(full, degree, count, False, [0.0], order)
            for order in range(2, 2 + degree // 2)
        ]
        rows += [
            _fit_basis(full, degree, count, False, [float(spans)], order)
            for order in range(2, 2 + (degree - 1) // 2)
        ]
        if rows:
            levels.append((np.vstack(rows), np.zeros((len(rows), dims))))

    at, weight = _span_quadrature(full[degree : degree + spans + 1], degree)
    bending = _fit_basis(full, degree, count, periodic, at, min(2, degree))
    levels.append((bending * weight[:, None], np.zeros((at.shape[0], dims))))

    return _layered_lstsq(levels, count, dims)


def _insert_knots(cv, full, params, degree, periodic, extra):
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

    Returns
    -------
    tuple
        (control points (count + extra, dims), knot vector, params in the
        new parameter range).
    """
    count = cv.shape[0]
    spans = count if periodic else count - degree
    edges = list(full[degree : degree + spans + 1])
    for _ in range(extra):
        i = int(np.argmax(np.diff(edges)))
        edges.insert(i + 1, (edges[i] + edges[i + 1]) / 2.0)

    # Rescale to the new parameter range [0, spans + extra]
    scale = (spans + extra) / spans
    fine  = _knots_from_edges(np.array(edges) * scale, degree, periodic)

    # Periodic curves repeat their first control points at the end
    wrap = np.arange(count)
    if periodic:
        wrap = np.concatenate([wrap, np.arange(degree) % count])

    at, _ = _span_quadrature(fine[degree : degree + spans + extra + 1], degree)
    target  = BSpline(full, cv[wrap], degree)(at / scale)
    basis   = _fit_basis(fine, degree, count + extra, periodic, at)

    fine_cv = _layered_lstsq([(basis, target)], count + extra, cv.shape[1])
    return fine_cv, fine, params * scale


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


def _layered_lstsq(levels, size, dims):
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

    Returns
    -------
    np.ndarray
        Solution (size, dims).
    """
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
