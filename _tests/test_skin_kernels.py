"""Direct tests for the numba skin weight kernels.

Every kernel here is otherwise reached only through a public wrapper called
with default arguments, so the un-fused variants and the degenerate-row
branches have no coverage at all from the SkinData tests.
"""

import unittest

import numpy as np
from cgmath.geometry.utils._numba._skin_weights import (
    _normalize_weights_parallel,
    _prune_weights_parallel,
    _round_normalize_parallel,
    _set_max_influences_parallel,
    normalize_fast,
    prune_fast,
    round_normalize_fast,
    set_max_influences_fast,
)


def row(*values):
    return np.array([list(values)], dtype=np.float64)


def first_column(n_cols, index=0):
    """the degenerate-row answer every operation is expected to give"""
    expected = np.zeros(n_cols)
    expected[index] = 1.0
    return expected


class TestPruneKernel(unittest.TestCase):
    def test_threshold_zeroes_below_and_keeps_at_or_above(self):
        result = _prune_weights_parallel(row(0.6, 0.2, 0.15, 0.05), 0.15)
        np.testing.assert_allclose(result[0], [0.6, 0.2, 0.15, 0.0])

    def test_the_shape_is_never_changed(self):
        weights = np.full((7, 5), 0.2)
        self.assertEqual(_prune_weights_parallel(weights, 0.9).shape, (7, 5))

    def test_a_fully_pruned_row_keeps_its_max_at_one(self):
        result = _prune_weights_parallel(row(0.3, 0.4, 0.3, 0.0), 0.5)
        np.testing.assert_allclose(result[0], first_column(4, index=1))

    def test_an_all_zero_row_binds_the_first_column(self):
        """`0.0 >= 0.0` passes the threshold, so the rescue has to look for a
        non-zero survivor rather than a passing one."""
        result = _prune_weights_parallel(np.zeros((1, 4)), 0.0)
        np.testing.assert_allclose(result[0], first_column(4))

    def test_negatives_are_pruned_at_a_zero_threshold(self):
        result = _prune_weights_parallel(row(0.8, -0.2, 0.4, 0.0), 0.0)
        np.testing.assert_allclose(result[0], [0.8, 0.0, 0.4, 0.0])

    def test_an_all_negative_row_keeps_the_least_negative(self):
        result = _prune_weights_parallel(row(-0.3, -0.1, -0.5, -0.2), 0.0)
        np.testing.assert_allclose(result[0], first_column(4, index=1))

    def test_the_input_is_not_mutated(self):
        weights = row(0.6, 0.2, 0.1, 0.1)
        before  = weights.copy()
        _prune_weights_parallel(weights, 0.15)
        np.testing.assert_array_equal(weights, before)

    def test_the_wrapper_prunes_then_normalizes(self):
        """prune_fast composes the two pure kernels instead of fusing them"""
        cases = (
            (row(0.6, 0.2, 0.1, 0.1), 0.15),
            (row(0.3, 0.4, 0.3, 0.0), 0.5),
            (np.zeros((1, 4)), 0.0),
            (row(0.5, 0.5, 0.0, 0.0), 0.5),
        )
        for weights, threshold in cases:
            with self.subTest(threshold=threshold, weights=weights[0].tolist()):
                composed = _prune_weights_parallel(weights.copy(), threshold)
                normalize_fast(composed)
                np.testing.assert_allclose(
                    prune_fast(weights.copy(), threshold, normalize=True), composed
                )
                np.testing.assert_allclose(composed.sum(axis=1), 1.0)

    def test_the_wrapper_can_skip_normalizing(self):
        result = prune_fast(row(0.6, 0.2, 0.1, 0.1), 0.15, normalize=False)
        np.testing.assert_allclose(result[0], [0.6, 0.2, 0.0, 0.0])


class TestNormalizeKernel(unittest.TestCase):
    def test_a_row_is_scaled_to_sum_one(self):
        weights = row(3.0, 1.0, 0.0, 0.0)
        normalize_fast(weights)
        np.testing.assert_allclose(weights[0], [0.75, 0.25, 0.0, 0.0])

    def test_an_all_zero_row_binds_the_first_column(self):
        weights = np.zeros((1, 4))
        normalize_fast(weights)
        np.testing.assert_allclose(weights[0], first_column(4))

    def test_a_locked_column_keeps_its_value(self):
        weights      = row(0.5, 1.0, 1.0, 0.0)
        include_mask = np.ones(4, dtype=np.bool_)
        include_mask[0] = False
        _normalize_weights_parallel(
            weights, np.array([0], dtype=np.int32), include_mask
        )

        self.assertAlmostEqual(float(weights[0, 0]), 0.5)
        self.assertAlmostEqual(float(weights[0].sum()), 1.0)

    def test_the_catch_skips_a_locked_first_column(self):
        """with column 0 locked the unweighted row binds column 1 instead"""
        weights      = np.zeros((1, 4))
        include_mask = np.ones(4, dtype=np.bool_)
        include_mask[0] = False
        _normalize_weights_parallel(
            weights, np.array([0], dtype=np.int32), include_mask
        )
        np.testing.assert_allclose(weights[0], first_column(4, index=1))

    def test_indices_limits_which_rows_are_touched(self):
        """`normalize_fast` used to discard `indices` whenever nothing was
        locked, quietly normalizing the whole matrix instead."""
        weights = np.array([[3.0, 1.0], [2.0, 2.0], [9.0, 1.0]])
        normalize_fast(weights, indices=[0, 2])

        np.testing.assert_allclose(weights[0], [0.75, 0.25])
        np.testing.assert_array_equal(weights[1], [2.0, 2.0])
        np.testing.assert_allclose(weights[2], [0.9, 0.1])

    def test_indices_and_locks_together(self):
        weights = np.array([[0.5, 1.0, 1.0], [4.0, 4.0, 4.0]])
        normalize_fast(weights, locked_influences=[0], indices=[0])

        self.assertAlmostEqual(float(weights[0, 0]), 0.5)
        self.assertAlmostEqual(float(weights[0].sum()), 1.0)
        np.testing.assert_array_equal(weights[1], [4.0, 4.0, 4.0])

    def test_no_indices_normalizes_every_row(self):
        weights = np.array([[3.0, 1.0], [2.0, 2.0]])
        normalize_fast(weights)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2))


class TestRoundKernel(unittest.TestCase):
    def test_weights_are_quantized_to_n_digits(self):
        result = round_normalize_fast(row(0.114, 0.224, 0.662), 2)
        np.testing.assert_allclose(result[0], np.round(result[0], 2), atol=1e-12)

    def test_the_residual_lands_on_the_max_column(self):
        # 0.11 + 0.22 + 0.66 leaves 0.01 over, which the 0.66 column absorbs
        result = round_normalize_fast(row(0.114, 0.224, 0.662), 2)
        np.testing.assert_allclose(result[0], [0.11, 0.22, 0.67])

    def test_the_row_sums_to_one(self):
        rng     = np.random.default_rng(0)
        weights = rng.random((32, 6))
        weights /= weights.sum(axis=1, keepdims=True)
        result = round_normalize_fast(weights, 3)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(32))

    def test_an_unnormalized_row_is_normalized_first(self):
        """without step 1 the residual is the whole overshoot, which reads as
        unrepresentable and collapses the row instead of rounding it"""
        weights = row(6.0, 3.0, 2.0, 1.0)
        bare    = _round_normalize_parallel(weights.copy(), 2)
        wrapped = round_normalize_fast(weights.copy(), 2)

        np.testing.assert_allclose(bare[0], first_column(4))
        np.testing.assert_allclose(wrapped[0], [0.5, 0.25, 0.17, 0.08])

    def test_an_already_normalized_row_is_untouched_by_step_one(self):
        weights = np.full((1, 12), 1.0 / 12.0)
        np.testing.assert_array_equal(
            round_normalize_fast(weights.copy(), 3),
            _round_normalize_parallel(weights.copy(), 3),
        )

    def test_the_input_is_not_mutated(self):
        weights = row(3.0, 1.0, 0.0, 0.0)
        before  = weights.copy()
        round_normalize_fast(weights, 2)
        np.testing.assert_array_equal(weights, before)

    def test_an_unrepresentable_row_collapses(self):
        """twelve equal influences each round up to 0.1, overshooting 1.0 by
        more than the heaviest column carries"""
        for n_inf, precision in ((12, 1), (18, 2), (128, 2)):
            with self.subTest(influences=n_inf, precision=precision):
                result = round_normalize_fast(
                    np.full((1, n_inf), 1.0 / n_inf), precision
                )
                np.testing.assert_allclose(result[0], first_column(n_inf))

    def test_the_collapse_skips_a_column_that_rounds_away(self):
        weights = np.zeros((1, 13))
        weights[0, 0] = 0.001
        weights[0, 1:] = 1.0 / 12.0
        result = round_normalize_fast(weights, 1)
        np.testing.assert_allclose(result[0], first_column(13, index=1))

    def test_a_representable_row_is_not_collapsed(self):
        # 200 influences still fit at three decimals, where the cap is 1000
        result = round_normalize_fast(np.full((1, 200), 1.0 / 200), 3)
        self.assertEqual(int((result[0] != 0).sum()), 200)
        self.assertGreaterEqual(float(result.min()), 0.0)

    def test_only_the_unrepresentable_rows_collapse(self):
        weights = np.zeros((3, 12))
        weights[0] = 1.0 / 12.0
        weights[1, :3] = [0.5, 0.3, 0.2]
        weights[2, :2] = [0.9, 0.1]
        result = round_normalize_fast(weights, 1)

        self.assertEqual(int((result[0] != 0).sum()), 1)
        self.assertEqual(int((result[1] != 0).sum()), 3)
        self.assertEqual(int((result[2] != 0).sum()), 2)

    def test_no_weight_is_ever_negative(self):
        rng = np.random.default_rng(1)
        for n_inf in (4, 12, 32, 64):
            for precision in (1, 2, 3):
                with self.subTest(influences=n_inf, precision=precision):
                    weights = rng.random((16, n_inf))
                    weights /= weights.sum(axis=1, keepdims=True)
                    result = round_normalize_fast(weights, precision)
                    self.assertGreaterEqual(float(result.min()), 0.0)
                    np.testing.assert_allclose(result.sum(axis=1), np.ones(16))

    def test_an_all_zero_row_binds_the_first_column(self):
        np.testing.assert_allclose(
            round_normalize_fast(np.zeros((1, 4)), 2)[0], first_column(4)
        )


class TestMaxInfluencesKernel(unittest.TestCase):
    def test_only_the_top_k_survive(self):
        result = _set_max_influences_parallel(row(0.1, 0.5, 0.3, 0.1), 2)
        np.testing.assert_allclose(result[0], [0.0, 0.5, 0.3, 0.0])

    def test_the_shape_is_never_changed(self):
        weights = np.full((7, 5), 0.2)
        self.assertEqual(_set_max_influences_parallel(weights, 2).shape, (7, 5))

    def test_a_count_at_or_above_the_width_is_a_passthrough(self):
        weights = row(0.1, 0.5, 0.3, 0.1)
        np.testing.assert_array_equal(_set_max_influences_parallel(weights, 4), weights)

    def test_it_does_not_normalize(self):
        """the un-fused kernel is pure -- the caller normalizes afterwards"""
        result = _set_max_influences_parallel(row(0.1, 0.5, 0.3, 0.1), 2)
        self.assertAlmostEqual(float(result.sum()), 0.8)

    def test_the_input_is_not_mutated(self):
        weights = row(0.1, 0.5, 0.3, 0.1)
        before  = weights.copy()
        _set_max_influences_parallel(weights, 2)
        np.testing.assert_array_equal(weights, before)

    def test_the_wrapper_limits_then_normalizes(self):
        result = set_max_influences_fast(row(0.1, 0.5, 0.3, 0.1), 2, normalize=True)
        np.testing.assert_allclose(result[0], [0.0, 0.625, 0.375, 0.0])

    def test_the_wrapper_can_skip_normalizing(self):
        result = set_max_influences_fast(row(0.1, 0.5, 0.3, 0.1), 2, normalize=False)
        np.testing.assert_allclose(result[0], [0.0, 0.5, 0.3, 0.0])

    def test_the_wrapper_rescues_an_all_zero_row(self):
        """the fused kernel passed zeros straight through, leaving the row
        summing to 0 -- composing with normalize closes that hole"""
        result = set_max_influences_fast(np.zeros((1, 4)), 2, normalize=True)
        np.testing.assert_allclose(result[0], first_column(4))


class TestDegenerateRowPolicy(unittest.TestCase):
    """An unweighted vertex must come out on the first column, whichever
    operation ran last. Normalize is terminal in every wrapper composition,
    so an even split there silently overrides every other op's answer."""

    def test_prune_then_normalize(self):
        result = _prune_weights_parallel(np.zeros((1, 4)), 0.0)
        normalize_fast(result)
        np.testing.assert_allclose(result[0], first_column(4))

    def test_max_influences_then_normalize(self):
        result = _set_max_influences_parallel(np.zeros((1, 4)), 2)
        normalize_fast(result)
        np.testing.assert_allclose(result[0], first_column(4))

    def test_normalize_then_round(self):
        weights = np.zeros((1, 4))
        normalize_fast(weights)
        np.testing.assert_allclose(
            _round_normalize_parallel(weights, 2)[0], first_column(4)
        )


class TestKernelEdgeCases(unittest.TestCase):
    def test_a_single_column_matrix(self):
        np.testing.assert_allclose(_prune_weights_parallel(row(0.5), 0.0)[0],    [0.5])
        np.testing.assert_allclose(round_normalize_fast(row(0.5), 2)[0],         [1.0])
        np.testing.assert_allclose(_set_max_influences_parallel(row(0.5), 1)[0], [0.5])

        weights = row(0.5)
        normalize_fast(weights)
        np.testing.assert_allclose(weights[0], [1.0])

    def test_an_empty_matrix(self):
        empty = np.zeros((0, 4))
        self.assertEqual(_prune_weights_parallel(empty, 0.0).shape,    (0, 4))
        self.assertEqual(round_normalize_fast(empty, 2).shape,         (0, 4))
        self.assertEqual(_set_max_influences_parallel(empty, 2).shape, (0, 4))

        normalize_fast(empty)
        self.assertEqual(empty.shape, (0, 4))

    def test_rounding_to_zero_digits(self):
        result = round_normalize_fast(row(0.4, 0.6), 0)
        np.testing.assert_allclose(result[0], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()