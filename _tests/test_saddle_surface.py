import time
import unittest

import numpy as np
from cgmath.geometry import MeshData
from cgmath.geometry._saddle_surface import integrate, sample, SampleData

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestSaddleSurfaceIntegrate(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Create simple test geometries
        # Triangle
        self.triangle_points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.triangle_geometry = np.array([[0, 1, 2]])

        # Square (as quad)
        self.square_points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.square_geometry = np.array([[0, 1, 2, 3]])

        # Plane mesh for testing
        self.plane = MeshData(
            points=np.array(
                [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

    def test_integrate_triangle(self):
        """Test integrate() with a triangle"""
        # Triangle should be converted to quad internally
        area = integrate(self.triangle_points, self.triangle_geometry, samples=100)

        # Triangle with base=1, height=1 has area=0.5
        # The integration should approximate this value
        self.assertIsInstance(area, np.ndarray)
        self.assertGreater(area[0], 0.4)
        self.assertLess(area[0], 0.6)

    def test_integrate_square(self):
        """Test integrate() with a square"""
        area = integrate(self.square_points, self.square_geometry, samples=100)

        # Square with side=1 has area=1.0
        self.assertIsInstance(area, np.ndarray)
        self.assertTrue(allclose(area[0], 1.0, atol=0.01))

    def test_integrate_with_different_samples(self):
        """Test integrate() with different sample counts"""
        # More samples should give more accurate results
        area_low  = integrate(self.square_points, self.square_geometry, samples=10)
        area_high = integrate(self.square_points, self.square_geometry, samples=200)

        # Both should be close to 1.0, but high sample should be more accurate
        self.assertIsInstance(area_low, np.ndarray)
        self.assertIsInstance(area_high, np.ndarray)

        # High sample count should be closer to the true area of 1.0
        diff_low  = abs(area_low[0] - 1.0)
        diff_high = abs(area_high[0] - 1.0)
        self.assertLessEqual(diff_high, diff_low + 0.01)  # Allow small margin

    def test_integrate_with_2d_points(self):
        """Test integrate() handles 2D points correctly"""
        points_2d = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        # Should pad to 3D internally
        area = integrate(points_2d, self.square_geometry, samples=100)
        self.assertIsInstance(area, np.ndarray)
        self.assertTrue(allclose(area[0], 1.0, atol=0.01))

    def test_integrate_with_4d_points(self):
        """Test integrate() handles 4D points correctly"""
        points_4d = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )

        # Should use first 3 dimensions only
        area = integrate(points_4d, self.square_geometry, samples=100)
        self.assertIsInstance(area, np.ndarray)
        self.assertTrue(allclose(area[0], 1.0, atol=0.01))

    def test_integrate_with_ngons_raises_error(self):
        """Test integrate() raises error for ngons (>4 vertices)"""
        # Pentagon geometry
        pentagon_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        pentagon_geometry = np.array([[0, 1, 2, 3, 4]])

        with self.assertRaises(ValueError) as context:
            integrate(pentagon_points, pentagon_geometry, samples=100)
        self.assertIn("ngons detected", str(context.exception))


class TestSaddleSurfaceSample(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Create a simple plane for testing
        self.plane = MeshData(
            points=np.array(
                [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

        # Create face points and geometry
        self.face_points   = self.plane.points
        self.face_geometry = self.plane.f2v

    def test_sample_basic(self):
        """Test basic sampling functionality"""
        # Query point above the plane
        query_points = np.array([[0.0, 0.0, 1.5]])

        sample_data  = sample(query_points, self.face_points, self.face_geometry)

        # Verify return type
        self.assertIsInstance(sample_data, SampleData)

        # Verify properties
        self.assertEqual(sample_data.projections.shape,  query_points.shape)
        self.assertEqual(sample_data.distances.shape[0], query_points.shape[0])
        self.assertEqual(sample_data.weights.shape,      (query_points.shape[0], 4))
        self.assertEqual(sample_data.indices.shape[0],   query_points.shape[0])
        self.assertEqual(sample_data.uvs.shape,          (query_points.shape[0], 2))

        # The projection should be at the plane (z=0)
        self.assertTrue(allclose(sample_data.projections[0, :2], [0.0, 0.0]))
        self.assertTrue(allclose(sample_data.projections[0, 2], 0.0))

        # Distance should be 1.5
        self.assertTrue(allclose(sample_data.distances[0], 1.5))

    def test_sample_perfect_match(self):
        """Test sampling with perfect vertex match"""
        # Query point exactly at a vertex
        query_points = np.array([[0.5, 0.5, 0.0]])

        sample_data  = sample(query_points, self.face_points, self.face_geometry)

        # Should match perfectly
        self.assertTrue(allclose(sample_data.distances[0], 0.0))
        self.assertTrue(allclose(sample_data.projections[0], [0.5, 0.5, 0.0]))

    def test_a_query_on_a_vertex_gets_the_vertex_at_its_real_distance(self):
        # it got distance 0 and itself as its projection
        query       = np.array([[0.5, 0.5, 1e-7]])
        sample_data = sample(query, self.face_points, self.face_geometry)

        np.testing.assert_array_equal(sample_data.projections[0], [0.5, 0.5, 0.0])
        self.assertAlmostEqual(sample_data.distances[0], 1e-7, delta=1e-20)
        self.assertEqual(sample_data.weights[0].max(), 1.0)

    def test_the_vertex_match_is_relative_to_the_face_size(self):
        # 1e-6 from a vertex was a match on any face: on one 1e-4 wide that
        # is a hundredth of the face, so the query answered with distance 0
        for scale in (1e-4, 1e-6):
            with self.subTest(scale=scale):
                points      = self.face_points * scale
                query       = np.array([[0.5 * scale, 0.5 * scale, 5e-7]])
                sample_data = sample(query, points, self.face_geometry)
                self.assertAlmostEqual(sample_data.distances[0], 5e-7, delta=1e-15)
                np.testing.assert_allclose(
                    sample_data.projections[0], [0.5 * scale, 0.5 * scale, 0.0], atol=1e-15
                )

    def test_the_vertex_match_radius_is_the_smallest_face_around_the_vertex(self):
        # the radius came from the first face that holds the vertex: a face
        # 10 wide listed before a 0.01 wide one that shares the vertex
        # snapped a query lying on the small face, 7e-4 of it from the
        # vertex, to the vertex
        points = np.array([
            [0, 0, 0], [-10, 0, 0], [-10, -10, 0], [0, -10, 0],
            [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0.0],
        ])
        big   = [0, 3, 2, 1]
        small = [0, 4, 5, 6]
        query = np.array([[5e-6, 5e-6, 0.0]])
        for faces in ([big, small], [small, big]):
            with self.subTest(small_face=faces.index(small)):
                found = sample(query, points, np.array(faces))
                self.assertLess(found.distances[0], 1e-15)
                np.testing.assert_allclose(
                    found.projections[0], query[0], rtol=0, atol=1e-15
                )
                self.assertEqual(found.indices[0], faces.index(small))
                np.testing.assert_allclose(found.uvs[0], [5e-4, 5e-4], rtol=1e-9)

    def test_a_face_index_past_the_last_point_raises(self):
        # the vertex lookup writes per vertex, which numba does not check
        query = np.array([[0.2, 0.3, 1.0]])
        for faces in ([[0, 1, 2, 4]], [[0, 1, 5]]):
            with self.subTest(faces=faces):
                with self.assertRaises(IndexError):
                    sample(query, self.face_points, np.array(faces))

    def test_a_vertex_no_face_holds_is_never_matched(self):
        # a stray vertex has no face to answer with: a query on it, or near
        # enough to match it, read a face index that was never set and
        # crashed or answered a random face. A stray copy of a corner must
        # not change that corner's queries either, stored after the corner
        # or before it: before it, the copy took the queries near the corner
        # away from the vertex match
        points = self.face_points * 1000
        strays = np.array([points[1], [5000.0, 5000.0, 5000.0]])
        queries = np.array([
            [5000.0, 5000.0, 5000.0],
            [5000.0, 5000.0, 5000.0 + 1e-4],
            [500.0, 500.0, 0.0],
            [500.0, 500.0, 1e-5],
            [500.0 - 2e-4, 500.0 - 2e-4, 3e-4],
            [500.0 + 2e-4, 500.0 - 2e-4, 3e-4],
            [0.0, 0.0, 10.0],
        ])
        expected = sample(queries, points, self.face_geometry)
        for where in ("after", "before"):
            if where == "after":
                stored = np.vstack([points, strays])
                offset = 0
            else:
                stored = np.vstack([strays, points])
                offset = len(strays)

            found = sample(queries, stored, self.face_geometry + offset)
            for name in ("projections", "distances", "weights", "indices", "uvs"):
                with self.subTest(where=where, name=name):
                    np.testing.assert_array_equal(getattr(found, name), getattr(expected, name))

            with self.subTest(where=where, name="geometry"):
                np.testing.assert_array_equal(found.geometry, expected.geometry + offset)

            with self.subTest(where=where, name="MeshData.sample"):
                mesh = MeshData(points=stored, indices=np.arange(4) + offset, counts=np.array([4]))
                np.testing.assert_array_equal(mesh.sample(queries).distances, expected.distances)

    def test_sample_with_normals(self):
        """Test sampling with normal calculation"""
        query_points = np.array([[0.0, 0.0, 1.5]])

        # Calculate normals for the plane (pointing up in Z)
        normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )

        sample_data = sample(
            query_points, self.face_points, self.face_geometry, normals=normals
        )

        # Check that normals are computed
        self.assertEqual(sample_data.normals.shape, query_points.shape)

        # Check occlusion - points must be below the normal to be occluded
        self.assertFalse(sample_data.occluded[0])

    def test_sample_multiple_points(self):
        """Test sampling with multiple query points"""
        query_points = np.array(
            [[0.0, 0.0, 1.0], [0.25, 0.25, 0.5], [-0.25, -0.25, 2.0]]
        )

        sample_data = sample(query_points, self.face_points, self.face_geometry)

        # All points should be sampled
        self.assertEqual(sample_data.projections.shape[0], 3)
        self.assertEqual(sample_data.distances.shape[0],   3)
        self.assertEqual(sample_data.weights.shape[0],     3)

        # All distances should be positive
        self.assertTrue(np.all(sample_data.distances >= 0))

    def test_sample_with_triangles(self):
        """Test sampling works with triangular faces"""
        # Triangle mesh
        triangle_points   = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangle_geometry = np.array([[0, 1, 2]])

        query_points = np.array([[0.25, 0.25, 0.5]])

        sample_data  = sample(query_points, triangle_points, triangle_geometry)

        self.assertIsInstance(sample_data, SampleData)
        self.assertEqual(sample_data.projections.shape[0], 1)
        self.assertTrue(sample_data.distances[0] >= 0)

    def test_sample_with_ngons_raises_error(self):
        """Test sample() raises error for ngons"""
        # Pentagon
        pentagon_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        pentagon_geometry = np.array([[0, 1, 2, 3, 4]])
        query_points      = np.array([[0.5, 0.5, 0.5]])

        with self.assertRaises(ValueError) as context:
            sample(query_points, pentagon_points, pentagon_geometry)
        self.assertIn("ngons detected", str(context.exception))


class TestSaddleSurfaceSampleSpeed(unittest.TestCase):
    def test_a_few_queries_cost_about_what_the_trees_cost(self):
        # each call measured every face's shortest edge and found a face for
        # every vertex in numpy, 26 ms on 100k faces over the 30 ms of the
        # kd-trees and centroids the call needs anyway, so few queries on a
        # big mesh took 1.6-2x as long. One pass over the faces finds both
        from scipy.spatial import cKDTree
        from cgmath.geometry.utils import compute_centroids

        count = 316
        steps = np.linspace(0.0, 1.0, count + 1)
        x, y = np.meshgrid(steps, steps, indexing="ij")
        x        = x.ravel()
        y        = y.ravel()
        points   = np.stack([x, y, 0.1 * np.sin(6.0 * x) * np.cos(5.0 * y)], 1)
        first    = np.arange(count)
        corner   = (first[:, None] * (count + 1) + first).ravel()
        step     = count + 1
        geometry = np.stack([corner, corner + step, corner + step + 1, corner + 1], 1)
        query    = np.array([[0.4, 0.6, 0.3]])

        def needed():
            start = time.perf_counter()
            cKDTree(points).query(query)
            centroids, _ = compute_centroids(points, geometry, return_radiuses=True)
            cKDTree(centroids).query(query)
            return time.perf_counter() - start

        def sampled():
            start = time.perf_counter()
            sample(query, points, geometry)
            return time.perf_counter() - start

        # a warm-up, then the best of 7, the two taken in turn so a busy
        # machine slows both
        needed()
        sampled()
        times = [(needed(), sampled()) for _ in range(7)]
        best_needed, best_sampled = np.min(times, axis=0)
        self.assertLess(best_sampled, 1.5 * best_needed)


class TestSampleData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Create sample SampleData instance for 4 query points sampled from a quad
        # This simulates sampling 4 points from a quad mesh
        self.sample_data = SampleData(
            projections=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            distances=np.array([1.0, 0.5, 0.3, 0.7]),
            weights=np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],  # Equal weights
                    [0.5, 0.5, 0.0, 0.0],  # First two vertices
                    [0.0, 0.5, 0.5, 0.0],  # Middle two vertices
                    [0.0, 0.0, 0.5, 0.5],  # Last two vertices
                ]
            ),
            indices=np.array([0, 0, 0, 0]),
            normals=np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            occluded = np.array([False, True, False, True]),
            uvs      = np.array([[0.5, 0.5], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]),
            geometry = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
        )

    def test_sample_data_compute(self):
        """Test SampleData.compute() method"""
        # Create test values at vertices
        values = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )

        # Compute weighted average
        result = self.sample_data.compute(values)

        # Should return weighted values
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 4)  # Four query points
        self.assertEqual(result.shape[1], 3)  # Three dimensions

        # First point: equal weights, so should be average of all 4 vertices
        expected_first = np.mean(values, axis=0)
        self.assertTrue(allclose(result[0], expected_first))

    def test_sample_data_call(self):
        """Test SampleData.__call__() method"""
        values = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )

        # Should be able to call the object
        result = self.sample_data(values)

        # Should give same result as compute()
        expected = self.sample_data.compute(values)
        self.assertTrue(allclose(result, expected))

    def test_sample_data_remap(self):
        """Test SampleData.remap() method"""
        # Create source and destination meshes
        src_mesh = MeshData(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

        dst_mesh = src_mesh.copy()

        # Create simple UV data
        from cgmath.geometry import UVData

        dst_uv = UVData(
            points  = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

        # Remap the sample data
        remapped = self.sample_data.remap(src_mesh, dst_mesh, dst_uv)

        # Should return a new SampleData instance
        self.assertIsInstance(remapped, SampleData)

        # Should have same number of samples
        self.assertEqual(
            remapped.projections.shape[0], self.sample_data.projections.shape[0]
        )
        self.assertEqual(remapped.weights.shape[0], self.sample_data.weights.shape[0])

    def test_sample_data_copy(self):
        """Test SampleData.copy() creates independent copy"""
        copy = self.sample_data.copy()

        # Should be equal initially
        self.assertTrue(allclose(copy.projections, self.sample_data.projections))
        self.assertTrue(allclose(copy.distances, self.sample_data.distances))
        self.assertTrue(allclose(copy.weights, self.sample_data.weights))

        # Modify original
        self.sample_data.projections[0, 0] = 999.0

        # Copy should be unaffected
        self.assertFalse(allclose(copy.projections[0, 0], 999.0))