"""Argument handling of the geometry/utils wrappers the deformers share."""

import unittest

import numpy as np
from cgmath.geometry.utils import bilinear_sample, compute_centroids

# the unit quad on XY, and the same quad with its third corner lifted, which
# takes Newton several steps to solve from (0.5, 0.5)
QUAD     = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
TWISTED  = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
GEOMETRY = np.array([[0, 1, 2, 3]], dtype=np.int32)
UP       = np.tile([0.0, 0.0, 1.0], (4, 1))


def sample_uv(points: np.ndarray, query, **kwargs) -> np.ndarray:
    """the uv of the closest point to query on the one face"""
    centroids, radiuses = compute_centroids(points, GEOMETRY, return_radiuses=True)
    _, _, _, uv, _ = bilinear_sample(
        np.array([query]),
        points,
        GEOMETRY,
        centroids,
        radiuses,
        centroid_distance_tolerance = 10.0,
        **kwargs,
    )
    return uv[0]


class TestBilinearSampleTolerance(unittest.TestCase):
    """The Newton solve stops once its step in (u, v) is under iteration_tolerance."""

    def test_a_tight_tolerance_converges(self):
        uv = sample_uv(QUAD, [0.7, 0.3, 0.5], iteration_tolerance=1e-12)
        np.testing.assert_allclose(uv, [0.7, 0.3], atol=1e-9)

    def test_a_fractional_tolerance_is_not_truncated_to_zero(self):
        # it was cast with np.int32, so any tolerance below 1 became 0 and the
        # solve always ran every iteration; 0.9 must stop after the first step
        query     = [0.7, 0.3, 0.8]
        one_step  = sample_uv(TWISTED, query, iteration_tolerance=0.9)
        converged = sample_uv(TWISTED, query, iteration_tolerance=0.0)
        self.assertGreater(np.abs(one_step - converged).max(), 1e-2)
        self.assertGreater(np.abs(one_step - 0.5).max(), 1e-2)  # it took a step


class TestBilinearSampleScale(unittest.TestCase):
    """The same face at any size gives the same uv: no test is in absolute units."""

    SCALES = (1e-6, 1e-4, 1e-3, 1.0, 1e3)

    def check(self, points, query, **kwargs):
        reference = sample_uv(points, query, **kwargs)
        for scale in self.SCALES:
            with self.subTest(scale=scale):
                uv = sample_uv(points * scale, np.multiply(query, scale), **kwargs)
                np.testing.assert_allclose(uv, reference, atol=1e-9)
        return reference

    def test_a_small_face_is_solved(self):
        # under 1e-3 the old absolute tests returned the face centre
        np.testing.assert_allclose(self.check(QUAD, [0.7, 0.3, 0.5]), [0.7, 0.3], atol=1e-9)
        self.check(TWISTED, [0.7, 0.3, 0.8])

    def test_a_face_with_no_area_gives_its_closest_edge_point(self):
        # corners on one line: the solve stops on parallel tangents, and the
        # answer comes from the edges, not the face centre
        line = np.outer([0.0, 40.0, 100.0, 70.0], [2.0, 1.0, 2.0]) / 3.0
        for scale in self.SCALES:
            with self.subTest(scale=scale):
                points   = line * scale
                across   = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)  # square to the line
                query    = points[2] * 0.95 + across * scale
                centroid = points.mean(0)
                proj, dist, _, _, _ = bilinear_sample(
                    np.array([query]), points, GEOMETRY, centroid[None], np.array([1e30]), 10.0
                )
                closest = points[2] * 0.95  # on the line, under the query
                self.assertLess(np.linalg.norm(proj[0] - closest), 1e-6 * scale)

    def test_a_small_bezier_face_is_solved(self):
        np.testing.assert_allclose(
            self.check(QUAD, [0.7, 0.3, 0.5], normals=UP), [0.7, 0.3], atol=1e-9
        )
        self.check(TWISTED, [0.7, 0.3, 0.8], normals=UP)


if __name__ == "__main__":
    unittest.main()
