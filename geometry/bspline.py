from dataclasses import dataclass, field

import numpy as np
import scipy.integrate as si
from cgmath.geometry._base import Data
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
        Control point positions (N, dims).
    degree : int
        Degree of the B-spline curve.
    periodic : bool
        Whether the curve is closed/periodic.
    uniform : bool
        If True, uses arc-length parameterization where equal spacing in
        parameter space [0, 1] gives equal spacing along the curve.
        If False (default), uses native parameter space [0, max_param].
    use_numba : bool
        If True (default), uses optimized Numba kernels for evaluation.
        If False, uses scipy.interpolate.BSpline.
    registered : bool
        If True, u=0 on periodic curves aligns with the curve point nearest
        to points[0]. Uses the formula: offset = max_param - (degree - 1) / 2.
        Has no effect on non-periodic curves.
    arc_length_samples : int
        Number of samples used to build the arc-length lookup table.
        Higher values give more accurate arc-length parameterization.
        Default is 1000.
    """

    points:             np.ndarray
    degree:             int
    periodic:           bool
    uniform:            bool = field(default=False)
    use_numba:          bool = field(default=True)
    registered:         bool = field(default=False)
    arc_length_samples: int = field(default=1000)

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

    __repr__ = Data.__repr__

    def __eq__(self, other):
        return np.all(
            [
                np.allclose(self.cv, other.cv),
                self.degree == other.degree,
                self.periodic == other.periodic,
                self.uniform == other.uniform,
                self.registered == other.registered,
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
            offset = self._max_param - (self.degree - 1) / 2.0
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
            offset = self._max_param - (self.degree - 1) / 2.0
            u      = u + offset

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
        so that the returned values are relative to points[0].

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
                offset = self._max_param - (self.degree - 1) / 2.0
                u      = np.where(u < offset, u + self._max_param, u)
            u = self._param_to_arc_length(u)
        elif self.registered and self.periodic:
            offset = self._max_param - (self.degree - 1) / 2.0
            u      = u - offset

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
        Topology change invalidates all caches (control_point_params,
        arc-length tables, etc).
        """
        if self._max_param is None:
            self._init_bspline()

        if self.periodic:
            self.periodic = False
            # Topology changed: invalidate everything (cp_params,
            # arc-length table, total_length) and rebuild.
            self.rebuild()

    def close(self):
        """
        Closes an open curve to make it periodic.

        Converts the curve from non-periodic to periodic by
        reinitializing with periodic knot vector and wrapped geometry.
        Topology change invalidates all caches (control_point_params,
        arc-length tables, etc).
        """
        if self._max_param is None:
            self._init_bspline()

        if not self.periodic:
            self.periodic = True
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

        # Degree 1: parameters map directly to integer indices
        if d <= 1:
            if self.periodic:
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

        # Vectorized Newton-Raphson refinement
        u       = greville.copy()
        queries = self.points[:n]

        der1 = self._spl.derivative(1)
        der2 = self._spl.derivative(2)

        # Skip refinement for clamped endpoints on open curves
        mask = np.ones(n, dtype=bool)
        if not self.periodic:
            mask[0]  = False
            mask[-1] = False

        if mask.any():
            for _ in range(10):
                u_eval = u % self._max_param if self.periodic else u

                C     = self._spl(u_eval)
                Cp    = der1(u_eval)
                Cpp   = der2(u_eval)

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