"""Direct tests for the numba B-spline basis derivative kernel.

``_compute_basis_derivatives`` has no public wrapper, so it is checked here
against scipy's derivatives of every basis function, on each kind of knot
vector a curve can carry.
"""

import unittest

import numpy as np
from cgmath.geometry.utils._numba._bspline import _compute_basis_derivatives
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


if __name__ == "__main__":
    unittest.main()
