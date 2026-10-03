"""
The curve and surface data round trip empty too, as strictly as
``test_roundtrip`` compares them filled in: an empty curve, a constraint with
nothing attached, and the results of sampling or casting nothing. The arrays
of a result share one length, so the round-trip test's filled-in builders
cannot hold an empty one.
"""

import os
import tempfile
import unittest

import numpy as np
from cgmath._tests._examples_curves import arm, sheet, strip, tube
from cgmath._tests.test_roundtrip import FORMATS, assert_same
from cgmath.constraints.procrustes import ProcrustesData
from cgmath.geometry.bspline import BSplineData
from cgmath.geometry.bspline_patch import BSplinePatchData

NOTHING = np.zeros((0, 3))


class TestEmptyRoundTrips(unittest.TestCase):
    def check(self, obj):
        """``obj`` back the same from every format, named and sniffed, and copy()"""
        with tempfile.TemporaryDirectory() as tmp:
            for ext in FORMATS:
                path = obj.save(os.path.join(tmp, f"data.{ext}"))
                for mode in (ext, None):
                    back = type(obj).load(path, mode=mode)
                    assert_same(self, obj, back, f"{type(obj).__name__}.{ext}")
        assert_same(self, obj, obj.copy(), f"{type(obj).__name__}.copy()")

    def test_empty_curve(self):
        curve = BSplineData(
            degree             = 2,
            periodic           = True,
            uniform            = True,
            registered         = True,
            use_numba          = False,
            arc_length_samples = 64,
        )
        self.assertEqual(curve.points.shape, (0, 3))
        self.check(curve)

    def test_nothing_sampled_on_a_curve(self):
        result = BSplineData.from_edit_points(arm()).sample(NOTHING)
        self.assertEqual(result.points.shape, (0, 3))
        self.assertEqual(result.basis.shape, (0, 8))
        self.check(result)

    def test_nothing_sampled_on_a_patch(self):
        surface = BSplinePatchData(
            points=tube(), degree_u=2, degree_v=3, periodic_u=True, periodic_v=False
        )
        result = surface.sample(NOTHING)
        self.assertEqual(result.tangents.shape, (0, 2, 3))
        self.assertEqual(result.basis.shape, (0, 30))
        self.check(result)

    def test_nothing_cast_at_a_patch(self):
        surface = BSplinePatchData(
            points=sheet(), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False
        )
        result = surface.raycast(NOTHING, NOTHING)
        self.assertEqual(result.params.shape, (0, 2))
        self.assertEqual(result.hit.shape, (0,))
        self.check(result)

    def test_nothing_sampled_on_a_mesh(self):
        result = strip().sample(NOTHING)
        self.assertEqual(result.weights.shape, (0, 4))
        self.assertEqual(result.geometry.shape, (0, 4))
        self.check(result)

    def test_nothing_cast_at_a_mesh(self):
        result = strip().raycast(NOTHING, NOTHING)
        self.assertEqual(result.uvs.shape, (0, 2))
        self.assertEqual(result.geometry.shape, (0, 4))
        self.check(result)

    def test_constraint_with_nothing_attached(self):
        constraint              = ProcrustesData(arm())
        constraint.scale_offset = False
        self.assertIsNone(constraint.transforms)
        self.check(constraint)


if __name__ == "__main__":
    unittest.main()
