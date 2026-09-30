"""Direct tests for numba B-spline kernels, against scipy.

``_compute_basis_derivatives`` has no public wrapper, so it is checked
against scipy's derivatives of every basis function, on each kind of knot
vector a curve can carry. The surface closest-point Newton kernel is checked
step by step against the same steps taken with scipy's exact second
derivatives.
"""

import unittest

import numpy as np
from cgmath.geometry.utils._numba._bspline import (
    _compute_basis_derivatives,
    _newton_closest_point_surface_parallel,
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


def exact_newton_steps(queries, u, v, kv_u, kv_v, cvs, d, max_u, max_v, periodic_u, steps):
    """
    The kernel's closest-point Newton steps, with exact second derivatives
    from scipy: same trust region, clamping and wrapping.
    """
    bu = BSpline(kv_u, np.eye(cvs.shape[0]), d)
    bv = BSpline(kv_v, np.eye(cvs.shape[1]), d)
    for _ in range(steps):
        eu = [bu(u), bu.derivative(1)(u), bu.derivative(2)(u)]
        ev = [bv(v), bv.derivative(1)(v), bv.derivative(2)(v)]

        def surface(i, j):
            return np.einsum("ni,nj,ijd->nd", eu[i], ev[j], cvs)

        def dot(a, b):
            return np.einsum("nd,nd->n", a, b)

        su, sv = surface(1, 0), surface(0, 1)
        r      = surface(0, 0) - queries
        f1     = dot(r, su)
        f2     = dot(r, sv)
        j11    = dot(su, su) + dot(r, surface(2, 0))
        j12    = dot(su, sv) + dot(r, surface(1, 1))
        j22    = dot(sv, sv) + dot(r, surface(0, 2))
        det    = j11 * j22 - j12 * j12
        det    = np.where(np.abs(det) < 1e-14, np.where(det >= 0, 1e-14, -1e-14), det)
        step_u = np.clip((-f1 * j22 + f2 * j12) / det, -0.5, 0.5)
        step_v = np.clip((f1 * j12 - f2 * j11) / det, -0.5, 0.5)

        u = (u + step_u) % max_u if periodic_u else np.clip(u + step_u, 0.0, max_u)
        v = np.clip(v + step_v, 0.0, max_v)
    return u, v


class TestSurfaceNewtonKernel(unittest.TestCase):
    """
    Each Newton step uses exact second derivatives; S_uu and S_vv once
    divided by a knot difference one index off (0.14 off after one step).
    """

    def assert_steps_exact(self, periodic_u):
        d = 3
        kv_u, kv_v, cvs, max_u, max_v = patch(d, 4, 3, periodic_u)

        rng = np.random.default_rng(9)
        u   = rng.uniform(0.05, max_u - 0.05, 40)
        v   = rng.uniform(0.05, max_v - 0.05, 40)
        on = np.einsum(
            "ni,nj,ijd->nd",
            BSpline(kv_u, np.eye(cvs.shape[0]), d)(u),
            BSpline(kv_v, np.eye(cvs.shape[1]), d)(v),
            cvs,
        )
        queries = on + rng.normal(scale=0.15, size=on.shape)
        u0      = u + rng.uniform(-0.25, 0.25, u.shape)
        u0      = u0 % max_u if periodic_u else np.clip(u0, 0.0, max_u)
        v0      = np.clip(v + rng.uniform(-0.25, 0.25, v.shape), 0.0, max_v)

        for steps in (1, 2):
            with self.subTest(steps=steps):
                found_u, found_v = _newton_closest_point_surface_parallel(
                    queries, u0, v0, kv_u, kv_v, cvs, d, d, max_u, max_v,
                    periodic_u, False, steps, 0.0,
                )[:2]
                exact_u, exact_v = exact_newton_steps(
                    queries, u0, v0, kv_u, kv_v, cvs, d, max_u, max_v, periodic_u, steps
                )
                gap_u = found_u - exact_u
                if periodic_u:
                    gap_u = (gap_u + max_u / 2) % max_u - max_u / 2
                np.testing.assert_allclose(gap_u, 0.0, atol=1e-12)
                np.testing.assert_allclose(found_v, exact_v, atol=1e-12)

    def test_open_patch(self):
        self.assert_steps_exact(periodic_u=False)

    def test_periodic_patch(self):
        self.assert_steps_exact(periodic_u=True)


if __name__ == "__main__":
    unittest.main()
