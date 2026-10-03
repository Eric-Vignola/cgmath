"""Direct tests for numba B-spline kernels, against scipy.

``_compute_basis_derivatives`` has no public wrapper, so it is checked
against scipy's derivatives of every basis function, on each kind of knot
vector a curve can carry. ``_patch_derivatives``, which the surface
closest-point search evaluates the patch with, is checked against scipy's
partial derivatives up to the second. ``_curve_derivatives``, which the
best-fit parameter repair evaluates the curve with, is checked against
scipy's curve derivatives. ``_bezier_pieces``, which cuts a curve into the
pieces sample() searches, is checked to end each piece exactly on the
curve's own spots, and ``_piece_closest``, which searches each piece, to
stop Newton's method at each root.
"""

import unittest
from unittest import mock

import numpy as np
from cgmath.geometry.utils._numba import _bspline
from cgmath.geometry.utils._numba._bspline import (
    _bezier_pieces,
    _compute_basis_derivatives,
    _curve_derivatives,
    _patch_derivatives,
)
from scipy.interpolate import BSpline

# Clamped cubic knot vectors
NON_UNIFORM = np.r_[[0.0] * 4, 0.3, 1.1, 1.25, 2.9, 4.0, [5.5] * 4]
DOUBLE_KNOT = np.r_[[0.0] * 4, 1.0, 2.0, 2.0, 3.5, [5.0] * 4]
TRIPLE_KNOT = np.r_[[0.0] * 4, 1.0, 2.0, 2.0, 2.0, 3.5, [5.0] * 4]


def uniform_clamped(d, spans):
    """unit-spaced knots over [0, spans], clamped at both ends"""
    return np.clip(np.arange(spans + 2 * d + 1.0) - d, 0, spans)


def periodic_style(d, n):
    """unit-spaced knots over [0, n] without clamping, so c = n + d"""
    return np.arange(-d, n + d + 1, dtype=float)


def inside_domain(kv, d, count=36):
    """evenly spaced samples strictly inside the domain"""
    c = len(kv) - d - 1
    return np.linspace(kv[d], kv[c], count + 2)[1:-1]


def domain_knots(kv, d):
    """every knot of the domain, both ends included"""
    c = len(kv) - d - 1
    return kv[d:c + 1]


def one_sided(kv, d, count, below):
    """
    every basis function's derivative of order nu at x: scipy's, right-sided
    on a knot, or with below the left-sided ones, from the mirrored spline
    """
    if not below:
        splines = BSpline(kv, np.eye(count), d)
        return lambda x, nu: splines(x, nu)
    mirror = BSpline(kv[0] + kv[-1] - kv[::-1], np.eye(count)[::-1], d)
    return lambda x, nu: (-1) ** nu * mirror(kv[0] + kv[-1] - x, nu)


class TestBasisDerivativeKernel(unittest.TestCase):
    def assert_matches_scipy(self, kv, d, orders, u=None):
        """checks each order against BSpline.derivative(order)"""
        c       = len(kv) - d - 1
        u       = inside_domain(kv, d) if u is None else u
        splines = BSpline(kv, np.eye(c), d)
        for order in orders:
            with self.subTest(order=order):
                np.testing.assert_allclose(
                    _compute_basis_derivatives(u, kv, c, d, order),
                    splines.derivative(order)(u),
                    atol=1e-9,
                )

    def test_uniform_clamped_cubic(self):
        self.assert_matches_scipy(uniform_clamped(3, 4), 3, (1, 2, 3))

    def test_periodic_style_uniform_knots(self):
        for d in range(1, 5):
            with self.subTest(d=d):
                kv = periodic_style(d, 5)
                self.assert_matches_scipy(kv, d, range(1, d + 1))

    def test_non_uniform_clamped_knots(self):
        self.assert_matches_scipy(NON_UNIFORM, 3, (1, 2, 3))

    def test_repeated_interior_knots(self):
        # derivative() only goes up to order d + 1 - m at a knot of
        # multiplicity m, the orders past it are checked below
        self.assert_matches_scipy(DOUBLE_KNOT, 3, (1, 2))
        self.assert_matches_scipy(TRIPLE_KNOT, 3, (1,))

    def test_orders_past_a_repeated_knots_smoothness(self):
        # scipy's direct evaluator still takes these orders
        d = 3
        for kv, orders in ((DOUBLE_KNOT, (3,)), (TRIPLE_KNOT, (2, 3))):
            c       = len(kv) - d - 1
            u       = np.r_[inside_domain(kv, d), domain_knots(kv, d)]
            splines = BSpline(kv, np.eye(c), d)
            for order in orders:
                with self.subTest(c=c, order=order):
                    np.testing.assert_allclose(
                        _compute_basis_derivatives(u, kv, c, d, order),
                        splines(u, nu=order),
                        atol=1e-9,
                    )

    def test_values_at_knots_and_domain_ends(self):
        cases = {
            "uniform clamped": (uniform_clamped(3, 4), (1, 2, 3)),
            "periodic-style":  (periodic_style(3, 5), (1, 2, 3)),
            "non-uniform":     (NON_UNIFORM, (1, 2, 3)),
            "double knot":     (DOUBLE_KNOT, (1, 2)),
            "triple knot":     (TRIPLE_KNOT, (1,)),
        }
        for name, (kv, orders) in cases.items():
            with self.subTest(name):
                self.assert_matches_scipy(kv, 3, orders, domain_knots(kv, 3))

    def test_order_zero_is_the_basis(self):
        u = inside_domain(NON_UNIFORM, 3)
        np.testing.assert_allclose(
            _compute_basis_derivatives(u, NON_UNIFORM, 9, 3, 0),
            BSpline(NON_UNIFORM, np.eye(9), 3)(u),
            atol=1e-9,
        )

    def test_orders_above_the_degree_give_zeros(self):
        kv     = uniform_clamped(3, 4)
        u      = inside_domain(kv, 3)
        result = _compute_basis_derivatives(u, kv, 7, 3, 4)
        self.assertEqual(result.shape, (u.shape[0], 7))
        self.assertFalse(result.any())

    def test_a_negative_order_raises(self):
        kv = uniform_clamped(3, 4)
        with self.assertRaises(ValueError):
            _compute_basis_derivatives(inside_domain(kv, 3), kv, 7, 3, -1)


def patch(d, spans_u, spans_v, periodic_u, seed=8):
    """
    A bumpy cubic patch: a tube around y when periodic in u, a grid
    otherwise. Returns the knot vectors, the control points (wrapped along
    u when periodic, as BSplinePatchData stores them) and the parameter
    ranges.
    """
    rng  = np.random.default_rng(seed)
    n_u  = spans_u if periodic_u else spans_u + d
    n_v  = spans_v + d
    rows = np.arange(n_v, dtype=float)[None, :] * np.ones((n_u, 1))
    if periodic_u:
        angle  = 2 * np.pi * np.arange(n_u) / n_u
        radius = 3 + rng.uniform(-0.5, 0.5, (n_u, n_v))
        x      = radius * np.cos(angle)[:, None]
        z      = radius * np.sin(angle)[:, None]
        cvs    = np.stack([x, rows, z], axis=-1)
        cvs    = cvs[np.r_[np.arange(n_u), np.arange(d) % n_u]]
        kv_u   = periodic_style(d, n_u)
        max_u  = float(n_u)
    else:
        cols  = np.arange(n_u, dtype=float)[:, None] * np.ones((1, n_v))
        cvs   = np.stack([cols, rows, rng.uniform(-0.6, 0.6, (n_u, n_v))], axis=-1)
        kv_u  = uniform_clamped(d, spans_u)
        max_u = float(spans_u)
    return kv_u, uniform_clamped(d, spans_v), cvs, max_u, float(spans_v)


class TestPatchDerivativesKernel(unittest.TestCase):
    """
    S and its partial derivatives up to the second match scipy's, inside
    the domain, on its knots and at both ends; with below, on a knot, they
    are the span below's (left-sided). The closest-point search takes its
    Newton steps from them; its first version divided S_uu and S_vv by a
    knot difference one index off (0.14 off after one step).
    """

    ORDERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))

    def assert_matches_scipy(self, d, periodic_u, below=False):
        kv_u, kv_v, cvs, _, _ = patch(d, 4, 3, periodic_u)
        if below:
            # every knot past the domain's start has a span below it
            u_all = domain_knots(kv_u, d)[1:]
            v_all = domain_knots(kv_v, d)[1:]
        else:
            u_all = np.r_[inside_domain(kv_u, d, 9), domain_knots(kv_u, d)]
            v_all = np.r_[inside_domain(kv_v, d, 7), domain_knots(kv_v, d)]
        bu = one_sided(kv_u, d, cvs.shape[0], below)
        bv = one_sided(kv_v, d, cvs.shape[1], below)

        work    = np.zeros(d + 1)
        basis_u = np.zeros((3, d + 1))
        basis_v = np.zeros((3, d + 1))
        out     = np.zeros((6, 3))
        for u in u_all:
            for v in v_all:
                _patch_derivatives(
                    u, v, kv_u, kv_v, cvs, d, d, 2, below, below, work, basis_u, basis_v, out
                )
                for row, (a, b) in enumerate(self.ORDERS):
                    want = np.einsum("i,j,ijd->d", bu(u, a), bv(v, b), cvs)
                    np.testing.assert_allclose(
                        out[row], want, rtol=1e-12, atol=1e-12, err_msg=f"{(a, b)} at {(u, v)}"
                    )

    def test_open_patch(self):
        for d in (1, 2, 3):
            with self.subTest(d=d):
                self.assert_matches_scipy(d, periodic_u=False)

    def test_periodic_patch(self):
        for d in (1, 2, 3):
            with self.subTest(d=d):
                self.assert_matches_scipy(d, periodic_u=True)

    def test_left_sided_on_knots(self):
        # S_u jumps across a degree-1 knot, S_uu across a degree-2 one
        for d in (1, 2, 3):
            for periodic_u in (False, True):
                with self.subTest(d=d, periodic_u=periodic_u):
                    self.assert_matches_scipy(d, periodic_u, below=True)


class TestCurveDerivativesKernel(unittest.TestCase):
    """
    _curve_derivatives runs scipy's own de Boor recursion: every derivative
    matches BSpline(x, nu), at knots and both domain ends included.
    """

    def assert_matches_scipy(self, kv, d, cv):
        curve = BSpline(kv, cv, d)
        work  = np.zeros((d + 2, d + 1))
        out   = np.zeros((d + 2, cv.shape[1]))
        for x in np.r_[inside_domain(kv, d), domain_knots(kv, d)]:
            _curve_derivatives(x, kv, cv, d, d + 1, work, out)
            for order in range(d + 1):
                np.testing.assert_allclose(out[order], curve(x, order), rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(out[d + 1], 0.0)

    def test_clamped_knots(self):
        rng = np.random.default_rng(0)
        for d in range(1, 6):
            kv = uniform_clamped(d, 4)
            with self.subTest(d=d):
                self.assert_matches_scipy(kv, d, rng.normal(size=(len(kv) - d - 1, 3)))
        for kv in (NON_UNIFORM, DOUBLE_KNOT, TRIPLE_KNOT):
            self.assert_matches_scipy(kv, 3, rng.normal(size=(len(kv) - 4, 2)))

    def test_periodic_knots(self):
        rng = np.random.default_rng(1)
        for d in (1, 2, 3, 5):
            kv = periodic_style(d, 6)
            cv = rng.normal(size=(6, 3))
            with self.subTest(d=d):
                self.assert_matches_scipy(kv, d, np.vstack([cv, cv[:d]]))


class TestBezierPiecesKernel(unittest.TestCase):
    """
    _bezier_pieces puts each piece's first and last Bezier control points
    exactly where _curve_derivatives puts the curve at the piece's start and
    end, so that sample() measures an open curve's ends as compute() does.
    """

    def test_piece_ends_are_the_curves_spots(self):
        rng   = np.random.default_rng(6)
        cases = [(uniform_clamped(d, 5), d, rng.normal(size=(5 + d, 3))) for d in range(1, 6)]
        cases.extend((kv, 3, rng.normal(size=(len(kv) - 4, 2)))
                     for kv in (NON_UNIFORM, DOUBLE_KNOT, TRIPLE_KNOT))
        for d in (1, 2, 3, 5):
            cv = rng.normal(size=(6, 3))
            cases.append((periodic_style(d, 6), d, np.vstack([cv, cv[:d]])))

        for kv, d, cv in cases:
            work = np.zeros((d + 2, d + 1))
            out  = np.zeros((d + 1, cv.shape[1]))
            for per_span in (1, 2, 3):
                windows, bezier = _bezier_pieces(kv, cv, d, per_span)[:2]
                for piece, (start, end) in enumerate(windows):
                    with self.subTest(kv=kv.tolist(), d=d, per_span=per_span, piece=piece):
                        _curve_derivatives(start, kv, cv, d, 0, work, out)
                        np.testing.assert_array_equal(bezier[piece, 0], out[0])
                        _curve_derivatives(end, kv, cv, d, 0, work, out)
                        np.testing.assert_array_equal(bezier[piece, d], out[0])


class TestPieceClosestKernel(unittest.TestCase):
    """
    _piece_closest finds each root of a piece's distance slope by Newton's
    method inside a bracket, and stops once a step is too small to move t:
    halving the bracket from there would only walk back to the same root,
    over dozens more slope evaluations.
    """

    def test_newton_stops_at_the_root(self):
        # run in Python, counting each piece's slope evaluations
        real  = _bspline._bernstein
        calls = []

        def counting(coef, t, work):
            calls.append(t)
            return real(coef, t, work)

        rng = np.random.default_rng(4)
        for d in (2, 3, 5):
            kv = uniform_clamped(d, 12 - d)
            cv = np.cumsum(rng.normal(size=(12, 3)), axis=0)
            windows, bezier, const, linear = _bezier_pieces(kv, cv, d, 2)[:4]

            queries = bezier[:, 0] + rng.normal(scale=0.3, size=bezier[:, 0].shape)
            scratch = (np.empty(2 * d), np.empty((96, 2 * d)), np.empty((96, 2)), np.empty(2 * d))
            most    = 0
            with mock.patch.object(_bspline, "_bernstein", counting):
                for point in queries:
                    for piece in range(len(windows)):
                        del calls[:]
                        _bspline._piece_closest.py_func(point, piece, bezier, const, linear, 0.0,
                                                        *scratch)
                        most = max(most, len(calls))
            with self.subTest(d=d):
                self.assertLessEqual(most, 12)


if __name__ == "__main__":
    unittest.main()
