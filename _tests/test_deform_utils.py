"""Argument handling of the geometry/utils wrappers the deformers share."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np
from cgmath.geometry.utils import bilinear_sample, compute_centroids
from cgmath.geometry.utils.main import _seeded_bilinear_sample, bezier_evaluate

# the unit quad on XY, and the same quad with its third corner lifted, which
# takes Newton several steps to solve from (0.5, 0.5)
QUAD     = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
TWISTED  = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
GEOMETRY = np.array([[0, 1, 2, 3]], dtype=np.int32)
UP       = np.tile([0.0, 0.0, 1.0], (4, 1))
PROJECTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        # solve always ran every iteration; 0.9 must stop after the first step,
        # of Newton and of the exact solve's root search alike
        query     = [0.7, 0.3, 0.8]
        one_step  = sample_uv(TWISTED, query, iteration_tolerance=0.9)
        converged = sample_uv(TWISTED, query, iteration_tolerance=0.0)
        self.assertGreater(np.abs(one_step - converged).max(), 1e-3)
        self.assertGreater(np.abs(one_step - 0.5).max(), 1e-2)  # it took a step

    def test_the_bezier_patch_honours_them_too(self):
        # the Bezier solve ran its own 20 iterations to 1e-6 whatever was asked
        query = [0.7, 0.3, 0.8]
        one_step = sample_uv(
            TWISTED, query, normals=UP, iteration_count=1, iteration_tolerance=0.1
        )
        converged = sample_uv(
            TWISTED, query, normals=UP, iteration_count=100, iteration_tolerance=1e-14
        )
        self.assertGreater(np.abs(one_step - converged).max(), 1e-2)


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


def sample_distance(points: np.ndarray, query, **kwargs) -> float:
    """the distance from query to its closest point on the one face"""
    centroids, radiuses = compute_centroids(points, GEOMETRY, return_radiuses=True)
    _, dist_sq, _, _, _ = bilinear_sample(
        np.array([query], dtype=np.float64),
        points,
        GEOMETRY,
        centroids,
        radiuses,
        centroid_distance_tolerance = 10.0,
        **kwargs,
    )
    return float(np.sqrt(dist_sq[0]))


def dense_distance(surface, query) -> float:
    """the distance from query to the closest of 401 x 401 points of a face"""
    grid = np.linspace(0.0, 1.0, 401)
    u, v = (x.ravel() for x in np.meshgrid(grid, grid, indexing="ij"))
    return float(np.sqrt(np.min(np.sum((surface(u, v) - query) ** 2, axis=1))))


def bilinear(points: np.ndarray):
    """S(u, v) of the bilinear face on 4 corners"""

    def surface(u, v):
        weights = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], 1)
        return weights @ points

    return surface


class TestBilinearSampleExact(unittest.TestCase):
    """The closest point is the closest anywhere on the face, not a local minimum."""

    # a strongly twisted face, and two plane faces that fold over themselves
    SADDLE = QUAD + np.outer([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    BOWTIE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    DART   = np.array([[0.0, 0.0], [2.0, 0.0], [0.4, 0.4], [0.0, 2.0]])

    def check(self, points, query, scale=1.0):
        points = points * scale
        query  = np.multiply(query, scale)
        found  = sample_distance(points, query)
        dense  = dense_distance(bilinear(points), query)
        self.assertLessEqual(found, dense + 1e-12 * scale)
        return found

    def test_a_twisted_face_far_from_its_centre(self):
        # Newton from the centre stopped half a face away from the closest corner
        for query in ([0.75, 0.75, 2.0], [0.7, 0.2, -1.0]):
            with self.subTest(query=query):
                self.check(self.SADDLE, query)

    def test_a_folded_uv_face_finds_the_query_on_it(self):
        # the query is on the face: Newton stopped 0.18 away from it
        self.assertLess(self.check(self.BOWTIE, [0.5, 0.75]), 1e-9)
        uv = sample_uv(self.BOWTIE, [0.5, 0.75])
        np.testing.assert_allclose(uv, [0.5, 0.75], atol=1e-9)

    def test_a_dart_uv_face(self):
        self.check(self.DART, [0.5, 1.5])

    def test_the_exact_solve_holds_at_any_scale(self):
        query     = [0.7, 0.2, -1.0]
        reference = sample_uv(self.SADDLE, query)
        for scale in TestBilinearSampleScale.SCALES:
            with self.subTest(scale=scale):
                self.check(self.SADDLE, query, scale)
                uv = sample_uv(self.SADDLE * scale, np.multiply(query, scale))
                np.testing.assert_allclose(uv, reference, atol=1e-9)

    def test_a_bezier_face_far_from_its_centre(self):
        # Newton from the centre stopped 0.3 farther than the closest point
        points = QUAD + np.outer([-0.4, 0.0, 0.2, 0.3], [0.0, 0.0, 1.0])
        normals = np.array(
            [[-0.4, -0.1, 1.0], [-0.1, -0.2, 1.0], [0.2, 0.0, 1.0], [-0.1, 0.3, 1.0]]
        )
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        query   = [0.1, 0.7, 1.1]

        def surface(u, v):
            uv = np.stack([u, v], 1)
            return bezier_evaluate(points, normals, np.repeat(GEOMETRY, len(uv), 0), uv)

        found = sample_distance(points, query, normals=normals)
        self.assertLessEqual(found, dense_distance(surface, query) + 1e-12)


def seed_answers():
    """
    (bezier, seed set, unseeded answer, seeded answer) for each output of
    sampling a twisted grid with each seed set. A function, so a fresh
    process can run it too.
    """
    rng = np.random.default_rng(2)
    x, y = np.meshgrid(np.arange(9) / 8.0, np.arange(9) / 8.0, indexing="ij")
    points   = np.stack([x.ravel(), y.ravel(), 0.1 * rng.normal(size=x.size)], 1)
    corner   = (np.arange(8)[:, None] * 9 + np.arange(8)[None]).ravel()
    geometry = np.stack([corner, corner + 9, corner + 10, corner + 1], 1).astype(np.int32)
    normals  = np.column_stack([0.2 * rng.normal(size=(x.size, 2)), np.ones(x.size)])
    normals  = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # queries around the grid, on its vertices and on its edge midpoints
    around = np.column_stack([rng.uniform(-0.2, 1.2, (300, 2)), 0.3 * rng.normal(size=300)])
    queries = np.vstack([
        around,
        points[::5],
        (points[geometry[:, 0]] + points[geometry[:, 1]]) / 2,
    ])
    centroids, radiuses = compute_centroids(points, geometry, return_radiuses=True)

    # a face of each query's nearest vertex, random faces, and no seed
    holder = np.full(len(points), -1, dtype=np.int32)
    np.put(holder, geometry.ravel(), np.repeat(np.arange(len(geometry)), 4))

    nearest = np.linalg.norm(queries[:, None] - points[None], axis=2).argmin(axis=1)
    seed_sets = {
        "nearest vertex": holder[nearest],
        "random":         rng.integers(-1, len(geometry), len(queries)),
        "none":           np.full(len(queries), -1),
    }
    for kind in (None, normals):
        plain = bilinear_sample(
            queries, points, geometry, centroids, radiuses, 10.0, normals=kind
        )
        for name, seeds in seed_sets.items():
            seeded = _seeded_bilinear_sample(
                queries, points, geometry, centroids, radiuses, 10.0,
                seed_faces = seeds,
                normals    = kind,
            )
            for a, b in zip(plain, seeded):
                yield kind is not None, name, a, b


class TestBilinearSampleSeedFaces(unittest.TestCase):
    """A seed only skips the faces beyond its closest point: any seed, the same answer.

    sample() seeds each query with a face of its nearest vertex, through the
    private _seeded_bilinear_sample; bilinear_sample is the unseeded call.
    """

    def test_any_seed_gives_the_same_answer(self):
        for bezier, seeds, a, b in seed_answers():
            with self.subTest(bezier=bezier, seeds=seeds):
                np.testing.assert_array_equal(a, b)

    def test_any_seed_gives_the_same_answer_whatever_the_cache_holds(self):
        # the seed's face solve, its bound left at inf, is another overload
        # than the loop's. With the seed's in numba's cache and not the
        # loop's, as two processes compiling at once can leave it, the seed
        # ran the copies of the solve's callees it was cached with and the
        # loop the ones this process compiled: under fast math 2 answers
        # moved by 3e-16, and 90 Bezier ones by up to 2e-9
        fill = textwrap.dedent(
            """
            import numpy as np
            from cgmath.geometry.utils._numba._bilinear import (
                _closest_point_on_bezier_face,
                _closest_point_on_face,
            )

            p = np.zeros(3)
            _closest_point_on_face(p, p, p, p, p, 1e-8, np.int32(100))
            _closest_point_on_bezier_face(np.zeros((4, 4, 3)), p, 1e-8, np.int32(100))
            """
        )
        check = textwrap.dedent(
            """
            import numpy as np
            from cgmath._tests.test_deform_utils import seed_answers

            moved = [
                (bezier, seeds)
                for bezier, seeds, a, b in seed_answers()
                if not np.array_equal(a, b)
            ]
            assert not moved, moved
            """
        )
        # a new cache that holds only the seed's overloads, then a new
        # process. The children import from where this process does
        # (mayapy's venv is only on sys.path once Maya is initialised)
        path = f"import sys\nsys.path[:0] = {sys.path!r}\n"
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, NUMBA_CACHE_DIR=cache)
            for script in (fill, check):
                result = subprocess.run(
                    [sys.executable, "-c", path + script],
                    cwd            = PROJECTS,
                    env            = env,
                    capture_output = True,
                    text           = True,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-1500:])

    def test_the_samplers_use_no_fast_math(self):
        # under fast math each compiled copy of a function rounds its own
        # way, and which copy a kernel runs depends on what numba's cache
        # held (the test above). Every function the samplers call, or read
        # control points from, rounds alike without it
        from cgmath.geometry.utils._numba import _bilinear

        todo = [
            _bilinear._bilinear_sample,
            _bilinear._bezier_sample,
            _bilinear._compute_all_pn_quad_cps,
            _bilinear._compute_bezier_face_bounds,
        ]
        seen = set()
        while todo:
            function = todo.pop()
            if function.__name__ in seen:
                continue
            seen.add(function.__name__)
            with self.subTest(function.__name__):
                self.assertFalse(function.targetoptions.get("fastmath", False))

            # the numba functions it calls
            for name in function.py_func.__code__.co_names:
                called = function.py_func.__globals__.get(name)
                if hasattr(called, "py_func") and hasattr(called, "targetoptions"):
                    todo.append(called)
        # the walk reaches the face solves and what they call
        for name in ("_poly_value", "_closest_point_on_bezier_face", "_compute_edge_cp"):
            self.assertIn(name, seen)

    def test_a_seed_that_is_no_face_is_refused(self):
        # the solver reads the seed's face unchecked. 2**32 wrapped onto face
        # 0 in the int32 cast, and -2 or a fraction passed as they were
        centroids, radiuses = compute_centroids(QUAD, GEOMETRY, return_radiuses=True)
        for seeds in ([1], [0, 0], [-2], [-7], [0.9], [2**32]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                _seeded_bilinear_sample(
                    np.array([[0.5, 0.5, 1.0]]), QUAD, GEOMETRY, centroids, radiuses, 10.0,
                    seed_faces = np.array(seeds),
                )


if __name__ == "__main__":
    unittest.main()
