import unittest

import numpy as np
from cgmath.geometry.utils.main import (
    average_points,
    compare_normals,
    indices_replace,
    matrix_row_overlaps,
    matrix_to_stream,
    point_row_overlaps,
    pxr,
    stream_to_matrix,
    vector_angle_difference,
)

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=atol)


class TestPxrFunction(unittest.TestCase):
    """Test pxr() USD module accessor"""

    def test_pxr_import(self):
        """Test pxr() function"""
        # Note: USD may or may not be available in the test environment
        try:
            pxr_module = pxr()
            self.assertIsNotNone(pxr_module)
        except ImportError as e:
            # If USD is not available, should raise ImportError with specific message
            self.assertIn("USD not found", str(e))


class TestMatrixRowOverlaps(unittest.TestCase):
    """Test matrix_row_overlaps() function"""

    def test_basic_overlap_detection(self):
        """Test basic overlap detection with exact matches"""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]])

        unique, duplicates, originals = matrix_row_overlaps(matrix, exact=True)

        # Should find 3 unique rows (indices 0, 1, 3)
        self.assertEqual(len(unique), 3)

        # Row 2 is duplicate of row 0
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0],   2)
        self.assertEqual(originals[0],    0)

    def test_non_exact_overlap(self):
        """Test overlap detection with sorted comparison"""
        matrix = np.array([[3, 2, 1], [4, 5, 6], [1, 2, 3], [7, 8, 9]])

        unique, duplicates, originals = matrix_row_overlaps(matrix, exact=False)

        # Rows 0 and 2 should match when sorted
        self.assertEqual(len(duplicates), 1)

    def test_empty_matrix(self):
        """Test with empty matrix"""
        matrix = np.array([]).reshape(0, 3)

        result = matrix_row_overlaps(matrix, exact=True)
        self.assertTrue(allclose(result, matrix))

    def test_none_matrix(self):
        """Test with None matrix"""
        result = matrix_row_overlaps(None, exact=True)
        self.assertIsNone(result)

    def test_sorted_indices(self):
        """Test with sort_indices=True"""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3]])

        unique, duplicates, originals = matrix_row_overlaps(
            matrix, exact=True, sort_indices=True
        )

        # Should find duplicates and sort them
        self.assertEqual(len(duplicates), 2)
        # Originals should be sorted
        self.assertTrue(np.all(originals[:-1] <= originals[1:]))


class TestPointRowOverlaps(unittest.TestCase):
    """Test point_row_overlaps() function"""

    def test_basic_point_overlap(self):
        """Test basic point overlap detection"""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
        )

        unique, duplicates, originals = point_row_overlaps(points, tolerance=1e-6)

        # Should find 3 unique points (indices 0, 1, 3)
        self.assertEqual(len(unique), 3)

        # Point 2 is duplicate of point 0
        self.assertIn(2, duplicates)
        self.assertIn(0, originals)

    def test_tolerance_matching(self):
        """Test tolerance-based matching"""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1e-7],  # Very close to first point
                [2.0, 2.0, 2.0],
            ]
        )

        unique, duplicates, originals = point_row_overlaps(points, tolerance=1e-6)

        # Should match points within tolerance
        self.assertEqual(len(unique), 3)
        self.assertIn(2, duplicates)

    def test_empty_points(self):
        """Test with empty points matrix"""
        points = np.array([]).reshape(0, 3)

        result = point_row_overlaps(points, tolerance=1e-6)
        self.assertTrue(allclose(result, points))

    def test_none_points(self):
        """Test with None points"""
        result = point_row_overlaps(None, tolerance=1e-6)
        self.assertIsNone(result)

    def test_no_duplicates(self):
        """Test when there are no duplicate points"""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        unique, duplicates, originals = point_row_overlaps(points, tolerance=1e-6)

        # All points should be unique
        self.assertEqual(len(unique), 3)
        self.assertEqual(len(duplicates), 0)


class TestMatrixStreamConversion(unittest.TestCase):
    """Test matrix_to_stream() and stream_to_matrix() functions"""

    def test_matrix_to_stream(self):
        """Test matrix_to_stream() conversion"""
        matrix = np.array([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]])

        indices, counts = matrix_to_stream(matrix)

        # Check indices (should skip -1)
        expected_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(allclose(indices, expected_indices))

        # Check counts
        expected_counts = np.array([3, 2, 4])
        self.assertTrue(allclose(counts, expected_counts))

    def test_stream_to_matrix(self):
        """Test stream_to_matrix() conversion"""
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        counts  = np.array([3, 2, 4])

        matrix  = stream_to_matrix(indices, counts)

        # Check shape (3 rows, max count is 4)
        self.assertEqual(matrix.shape, (3, 4))

        # Check values
        self.assertTrue(allclose(matrix[0], [0, 1, 2, -1]))
        self.assertTrue(allclose(matrix[1], [3, 4, -1, -1]))
        self.assertTrue(allclose(matrix[2], [5, 6, 7, 8]))

    def test_round_trip_conversion(self):
        """Test round-trip matrix -> stream -> matrix"""
        original = np.array([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]])

        indices, counts = matrix_to_stream(original)
        reconstructed = stream_to_matrix(indices, counts)

        self.assertTrue(allclose(original, reconstructed))


class TestIndicesReplace(unittest.TestCase):
    """Test indices_replace() function"""

    def test_basic_replacement(self):
        """Test basic index replacement"""
        matrix       = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 4]])
        from_indices = np.array([0, 2, 4])
        to_indices   = np.array([10, 20, 40])

        result   = indices_replace(matrix, from_indices, to_indices)

        expected = np.array([[10, 1, 20], [3, 40, 5], [10, 20, 40]])
        self.assertTrue(allclose(result, expected))

    def test_replacement_with_collapse(self):
        """Test index replacement with collapse"""
        matrix       = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 4]])
        from_indices = np.array([0, 2, 4])
        to_indices   = np.array([10, 20, 40])

        result = indices_replace(matrix, from_indices, to_indices, collapse=True)

        # After collapse, indices should be remapped to 0-based sequential
        self.assertTrue(np.all(result >= 0))
        self.assertEqual(np.max(result), np.unique(result[result >= 0]).size - 1)

    def test_empty_matrix(self):
        """Test with empty matrix"""
        matrix       = np.array([]).reshape(0, 3)
        from_indices = np.array([0, 1])
        to_indices   = np.array([10, 20])

        result = indices_replace(matrix, from_indices, to_indices)
        self.assertTrue(allclose(result, matrix))

    def test_none_matrix(self):
        """Test with None matrix"""
        result = indices_replace(None, np.array([0]), np.array([10]))
        self.assertIsNone(result)

    def test_no_matching_indices(self):
        """Test when no indices match"""
        matrix       = np.array([[0, 1, 2], [3, 4, 5]])
        from_indices = np.array([10, 20])
        to_indices   = np.array([100, 200])

        result = indices_replace(matrix, from_indices, to_indices)

        # Matrix should remain unchanged
        self.assertTrue(allclose(result, matrix))


class TestAveragePoints(unittest.TestCase):
    """Test average_points() function"""

    def test_basic_averaging(self):
        """Test basic point averaging"""
        points = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
        geometry = np.array([[0, 1, 2, -1], [0, 1, 2, 3]])

        result   = average_points(points, geometry)

        # First row: average of first 3 points
        expected_first = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self.assertTrue(allclose(result[0], expected_first))

        # Second row: average of all 4 points
        expected_second = np.array([0.5, 0.5, 0.5])
        self.assertTrue(allclose(result[1], expected_second))

    def test_with_negative_indices(self):
        """Test averaging with -1 indices (should be ignored)"""
        points   = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        geometry = np.array([[0, 1, -1, -1]])

        result   = average_points(points, geometry)

        # Should average only indices 0 and 1
        expected = np.array([0.5, 0.5, 0.0])
        self.assertTrue(allclose(result[0], expected))

    def test_single_point_per_row(self):
        """Test with single point per row"""
        points   = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        geometry = np.array([[0, -1], [1, -1]])

        result   = average_points(points, geometry)

        # Each row should equal the single point
        self.assertTrue(allclose(result[0], points[0]))
        self.assertTrue(allclose(result[1], points[1]))


class TestVectorComparison(unittest.TestCase):
    """Test vector comparison functions"""

    def test_compare_normals(self):
        """Test compare_normals() function"""
        normal_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normal_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        angles   = compare_normals(normal_a, normal_b)

        # First pair: identical (0 degrees)
        self.assertTrue(allclose(angles[0], 0.0))

        # Second pair: perpendicular (90 degrees)
        self.assertTrue(allclose(angles[1], 90.0, atol=1e-5))

    def test_compare_opposite_normals(self):
        """Test comparing opposite normals (180 degrees)"""
        normal_a = np.array([[1.0, 0.0, 0.0]])
        normal_b = np.array([[-1.0, 0.0, 0.0]])

        angles   = compare_normals(normal_a, normal_b)

        # Should be 180 degrees
        self.assertTrue(allclose(angles[0], 180.0, atol=1e-5))

    def test_vector_angle_difference(self):
        """Test vector_angle_difference() function"""
        vector_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vector_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        angles   = vector_angle_difference(vector_a, vector_b)

        # First pair: identical (0 degrees)
        self.assertTrue(allclose(angles[0], 0.0))

        # Second pair: perpendicular (90 degrees)
        self.assertTrue(allclose(angles[1], 90.0, atol=1e-5))

    def test_vector_angle_opposite(self):
        """Test angle difference for opposite vectors"""
        vector_a = np.array([[1.0, 0.0, 0.0]])
        vector_b = np.array([[-1.0, 0.0, 0.0]])

        angles   = vector_angle_difference(vector_a, vector_b)

        # Should be 180 degrees
        self.assertTrue(allclose(angles[0], 180.0, atol=1e-5))
