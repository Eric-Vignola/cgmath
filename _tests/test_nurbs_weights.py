"""Weights on NurbsCurveData and NurbsSurfaceData: rational curves and surfaces.

A shape holding ``weights`` is evaluated on its control points, each times
its weight, and divided by the weights' own spline; its ``points`` stay the
control points themselves. The shapes here are exact conics, a quarter
circle and a quarter cylinder of radius 1, which no curve or surface
without weights is.
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
from cgmath.geometry import NurbsCurveData, NurbsSurfaceData

ARC = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
# the middle point's weight: the cosine of half the arc's angle
ARC_WEIGHTS  = np.array([1.0, np.sqrt(0.5), 1.0])
RING         = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
RING_WEIGHTS = np.array([1.0, 2.0, 1.0, 3.0])
PARAMS       = np.linspace(0.0, 1.0, 101)
STEP         = 1e-6


def arc(**more) -> NurbsCurveData:
    """a quarter circle of radius 1 about z"""
    return NurbsCurveData(points=ARC, degree=2, weights=ARC_WEIGHTS, **more)


def cylinder(**more) -> NurbsSurfaceData:
    """a quarter cylinder of radius 1 about z, 2 high: the arc along u"""
    fields = {
        "points":     np.stack([ARC, ARC + [0.0, 0.0, 2.0]], axis=1),
        "degree_u":   2,
        "degree_v":   1,
        "periodic_u": False,
        "periodic_v": False,
        "weights":    np.stack([ARC_WEIGHTS, ARC_WEIGHTS], axis=1),
    }
    fields.update(more)
    return NurbsSurfaceData(**fields)


def grid() -> tuple:
    """21 by 5 parameters over the cylinder, flat"""
    u, v = np.meshgrid(np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 5))
    return u.ravel(), v.ravel()


class _Told(unittest.TestCase):
    def told(self, message: str, call) -> None:
        with self.assertRaises(ValueError) as caught:
            call()
        self.assertEqual(str(caught.exception), message)

    def files(self, item) -> None:
        """the item through each file format, its weights bit for bit"""
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("error")
            for kind in ("json", "npz", "pkl"):
                path = os.path.join(folder, f"item.{kind}")
                item.save(path)
                back = type(item).load(path)
                self.assertEqual(back, item, kind)
                np.testing.assert_array_equal(back.weights, item.weights, err_msg=kind)
        np.testing.assert_array_equal(item.copy().weights, item.weights)


class TestCurve(_Told):
    def test_quarter_circle(self):
        for numba in (True, False):
            curve = arc(use_numba=numba)
            points, tangents = curve.compute(PARAMS)
            np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, rtol=0, atol=1e-12)
            # the tangent: the slope of the curve, and square to the radius
            inner = PARAMS[1:-1]
            slope = (curve.compute(inner + STEP)[0] - curve.compute(inner - STEP)[0]) / (2.0 * STEP)
            np.testing.assert_allclose(tangents[1:-1], slope, rtol=0, atol=1e-6)
            np.testing.assert_allclose((points * tangents).sum(axis=1), 0.0, rtol=0, atol=1e-12)
            np.testing.assert_allclose(curve.compute_fast(PARAMS)[0], points, rtol=0, atol=1e-12)
            self.assertAlmostEqual(curve.get_length(fast=False), np.pi / 2.0, places=12)

    def test_points_stay(self):
        """the points are the control points, not multiplied by their weights"""
        curve = arc()
        np.testing.assert_array_equal(np.asarray(curve.points), ARC)
        np.testing.assert_array_equal(np.asarray(curve.weights), ARC_WEIGHTS)
        self.assertEqual(np.asarray(curve.weights).dtype, np.float64)
        # without them the same points are a parabola, off the circle
        plain = NurbsCurveData(points=ARC, degree=2)
        self.assertIsNone(plain.weights)
        self.assertGreater(abs(np.linalg.norm(plain.compute(0.5)[0]) - 1.0), 0.05)

    def test_ones_are_none(self):
        ones = NurbsCurveData(points=ARC, degree=2, weights=[1, 1, 1])
        none = NurbsCurveData(points=ARC, degree=2)
        np.testing.assert_allclose(ones.compute(PARAMS)[0], none.compute(PARAMS)[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ones.compute(PARAMS)[1], none.compute(PARAMS)[1], rtol=0, atol=1e-12)
        self.assertEqual(ones, none)
        self.assertNotEqual(arc(), none)

    def test_basis(self):
        """the rational shares: they sum to 1 and weigh the points into the curve"""
        curve  = arc()
        shares = curve.basis(PARAMS)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(shares @ ARC, curve.compute(PARAMS)[0], rtol=0, atol=1e-12)

    def test_uniform(self):
        points = arc(uniform=True).compute(PARAMS)[0]
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, rtol=0, atol=1e-12)

    def test_periodic(self):
        """the weights wrap with the control points"""
        ring = NurbsCurveData(points=RING, periodic=True, weights=RING_WEIGHTS)
        ends = ring.compute(np.array([0.0, 4.0]))[0]
        np.testing.assert_array_equal(ends[0], ends[1])
        plain = NurbsCurveData(points=RING, periodic=True)
        self.assertGreater(np.abs(ring.compute(1.5)[0] - plain.compute(1.5)[0]).max(), 0.05)

    def test_sample(self):
        """the closest point of a weighted curve: on the circle, under the point"""
        given = np.array([[2.0, 2.0, 0.0], [0.5, 0.1, 0.0], [3.0, 0.2, 0.0]])
        found = arc().sample(given)
        want  = given / np.linalg.norm(given, axis=1)[:, None]
        np.testing.assert_allclose(found.points, want, rtol=0, atol=1e-9)
        np.testing.assert_allclose(found.params[0], 0.5, rtol=0, atol=1e-9)

    def test_control_point_params(self):
        np.testing.assert_allclose(arc().control_point_params, [0.0, 0.5, 1.0], rtol=0, atol=1e-9)

    def test_checked(self):
        self.told(
            "weights must hold one value per control point, shape (3,), got shape (2,)",
            lambda: NurbsCurveData(points=ARC, degree=2, weights=[1.0, 1.0]).compute(0.5),
        )
        for held in ([1.0, 0.0, 1.0], [1.0, -2.0, 1.0], [1.0, np.nan, 1.0], [1.0, np.inf, 1.0]):
            self.told(
                "weights must be finite and above zero",
                lambda: NurbsCurveData(points=ARC, degree=2, weights=held).compute(0.5),
            )

    def test_not_solved(self):
        for what, call in (("fit", lambda: arc().fit(np.eye(3))), ("rebuild", lambda: arc().rebuild(count=5))):
            self.told(
                f"NurbsCurveData.{what}: the curve holds weights, which {what}() does not solve with; set weights "
                f"to None to {what} it as a curve whose weights are all 1",
                call,
            )

    def test_files(self):
        self.files(arc())
        self.assertIsNone(NurbsCurveData.from_dict(NurbsCurveData(points=ARC, degree=2).to_dict()).weights)


class TestSurface(_Told):
    def test_quarter_cylinder(self):
        u, v = grid()
        for numba in (True, False):
            patch  = cylinder(use_numba=numba)
            points = patch.evaluate(u, v)
            np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1.0, rtol=0, atol=1e-12)
            np.testing.assert_allclose(points[:, 2], 2.0 * v, rtol=0, atol=1e-12)
            computed, along_u, along_v = patch.compute(u, v)
            np.testing.assert_allclose(computed, points, rtol=0, atol=1e-12)
            inner   = (u > 0.01) & (u < 0.99) & (v > 0.01) & (v < 0.99)
            slope_u = (patch.evaluate(u[inner] + STEP, v[inner]) - patch.evaluate(u[inner] - STEP, v[inner])) / (2.0 * STEP)
            slope_v = (patch.evaluate(u[inner], v[inner] + STEP) - patch.evaluate(u[inner], v[inner] - STEP)) / (2.0 * STEP)
            np.testing.assert_allclose(along_u[inner], slope_u, rtol=0, atol=1e-6)
            np.testing.assert_allclose(along_v[inner], slope_v, rtol=0, atol=1e-6)
            self.assertAlmostEqual(patch.get_length_u(fast=False), np.pi / 2.0, places=12)
            self.assertAlmostEqual(patch.get_length_v(fast=False), 2.0, places=12)
            # an odd count: Simpson's rule over the samples
            self.assertAlmostEqual(patch.get_area(samples=51), np.pi, places=6)

    def test_ones_are_none(self):
        u, v = grid()
        ones = cylinder(weights=np.ones((3, 2)))
        none = cylinder(weights=None)
        self.assertIsNone(none.weights)
        np.testing.assert_allclose(ones.evaluate(u, v), none.evaluate(u, v), rtol=0, atol=1e-12)
        self.assertEqual(ones, none)
        self.assertNotEqual(cylinder(), none)

    def test_basis(self):
        u, v = grid()
        patch  = cylinder()
        shares = patch.basis(u, v)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        flat = np.asarray(patch.points).reshape(-1, 3)
        np.testing.assert_allclose(shares @ flat, patch.evaluate(u, v), rtol=0, atol=1e-12)

    def test_uniform(self):
        u, v = grid()
        points = cylinder(uniform_u=True, uniform_v=True).evaluate(u, v)
        np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1.0, rtol=0, atol=1e-12)

    def test_periodic(self):
        """a periodic direction's weights wrap with its points: a row of the
        surface is the curve of that row"""
        tube = NurbsSurfaceData(
            points=np.stack([RING, RING + [0.0, 0.0, 1.0]], axis=1), degree_u=3, degree_v=1, periodic_u=True,
            periodic_v=False, weights=np.stack([RING_WEIGHTS, RING_WEIGHTS], axis=1),
        )
        seam = tube.evaluate(np.array([0.0, 4.0]), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(seam[0], seam[1])
        at    = np.linspace(0.0, 4.0, 9)
        curve = NurbsCurveData(points=RING, periodic=True, weights=RING_WEIGHTS)
        np.testing.assert_allclose(tube.evaluate(at, np.zeros(9)), curve.compute(at)[0], rtol=0, atol=1e-12)

    def test_checked(self):
        self.told(
            "weights must hold one value per control point, shape (3, 2), got shape (2, 3)",
            lambda: cylinder(weights=np.ones((2, 3))).evaluate(0.5, 0.5),
        )
        self.told("weights must be finite and above zero", lambda: cylinder(weights=-np.ones((3, 2))).evaluate(0.5, 0.5))

    def test_not_searched(self):
        for what, more in (("sample", ()), ("raycast", (np.ones((1, 3)),))):
            self.told(
                f"NurbsSurfaceData.{what}: the surface holds weights, which {what}() does not search; set weights "
                f"to None for the surface its control points give at weight 1",
                lambda: getattr(cylinder(), what)(np.zeros((1, 3)), *more),
            )

    def test_files(self):
        self.files(cylinder())
        self.assertIsNone(NurbsSurfaceData.from_dict(cylinder(weights=None).to_dict()).weights)


if __name__ == "__main__":
    unittest.main()
