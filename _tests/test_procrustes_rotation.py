"""Correctness of the batched Procrustes rotation solve.

``ProcrustesData._batch_rotation`` routes clusters through one of two
implementations depending on ``det(H)``, because the numba polar
decomposition is only the Procrustes optimum when the covariance is not a
reflection. On a reflection it still returns something orthonormal with
``det == +1``, so a wrong answer is invisible to every downstream check --
these tests compare against the objective the solve is supposed to maximise
rather than against the shape of the result.

For a covariance ``H`` with singular values ``s0 >= s1 >= s2`` the optimum of
``trace(R.T @ H)`` over rotations is ``s0 + s1 + s2`` when ``det(H) > 0`` and
``s0 + s1 - s2`` when ``det(H) < 0``. That closed form is the reference here;
it does not depend on either implementation being right.
"""

import unittest

import numpy as np
from cgmath.constraints.procrustes import ProcrustesData

SEED = 8675309


def objective(R, H):
    """The quantity a Procrustes solve maximises: ``trace(R.T @ H)``."""
    return np.einsum("bij,bij->b", R, H)


def optimal_objective(H):
    """Closed-form optimum, independent of how the rotation is computed."""
    s    = np.linalg.svd(H, compute_uv=False)
    sign = np.where(np.linalg.det(H) < 0.0, -1.0, 1.0)
    return s[:, 0] + s[:, 1] + sign * s[:, 2]


def covariances(count, seed=SEED, reflected=None):
    """Random 3x3 covariances, optionally forced to a given det sign."""
    rng = np.random.default_rng(seed)
    H   = rng.normal(size=(count, 3, 3))
    if reflected is None:
        return H
    wrong = (np.linalg.det(H) < 0.0) != bool(reflected)
    H[wrong, 2, :] *= -1.0
    return H


class TestRotationsAreValid(unittest.TestCase):
    def _assert_rotations(self, R):
        self.assertEqual(R.shape[1:], (3, 3))
        eye = np.broadcast_to(np.eye(3), R.shape)
        self.assertTrue(
            np.allclose(np.einsum("bij,bkj->bik", R, R), eye, atol=1e-9),
            "result is not orthonormal",
        )
        self.assertTrue(
            np.allclose(np.linalg.det(R), 1.0, atol=1e-9),
            "result contains a reflection, not a rotation",
        )

    def test_upright_covariances(self):
        self._assert_rotations(
            ProcrustesData._batch_rotation(covariances(200, reflected=False))
        )

    def test_reflected_covariances(self):
        self._assert_rotations(
            ProcrustesData._batch_rotation(covariances(200, reflected=True))
        )

    def test_mixed_batch(self):
        self._assert_rotations(ProcrustesData._batch_rotation(covariances(400)))


class TestObjectiveIsMaximised(unittest.TestCase):
    """The point of the whole exercise, and what the polar path gets wrong."""

    def test_upright_batch_reaches_the_optimum(self):
        H   = covariances(200, reflected=False)
        got = objective(ProcrustesData._batch_rotation(H), H)
        self.assertTrue(np.allclose(got, optimal_objective(H), atol=1e-9))

    def test_reflected_batch_reaches_the_optimum(self):
        # the regression guard: the polar path averaged 1.41 here against a
        # true optimum of 3.46, while still returning a valid rotation
        H   = covariances(200, reflected=True)
        got = objective(ProcrustesData._batch_rotation(H), H)
        self.assertTrue(np.allclose(got, optimal_objective(H), atol=1e-9))

    def test_mixed_batch_reaches_the_optimum(self):
        H = covariances(400)
        self.assertTrue((np.linalg.det(H) < 0).any(), "batch has no reflections")
        self.assertTrue((np.linalg.det(H) > 0).any(), "batch has no upright rows")

        got = objective(ProcrustesData._batch_rotation(H), H)
        self.assertTrue(np.allclose(got, optimal_objective(H), atol=1e-9))

    def test_no_rotation_beats_the_returned_one(self):
        """Brute force against many random rotations, as an independent check."""
        H    = covariances(32)
        best = objective(ProcrustesData._batch_rotation(H), H)

        rng  = np.random.default_rng(SEED + 1)
        for _ in range(400):
            A = rng.normal(size=(3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 2] *= -1
            trial = objective(np.broadcast_to(Q, H.shape), H)
            self.assertTrue(np.all(trial <= best + 1e-9))


class TestBothBranchesAgree(unittest.TestCase):
    def test_upright_branch_matches_kabsch(self):
        """Where the fast path is used it must equal the reference exactly."""
        H     = covariances(300, reflected=False)
        fast  = ProcrustesData._batch_rotation(H)
        exact = ProcrustesData._svd_rotation(H)
        self.assertTrue(np.allclose(fast, exact, atol=1e-9))

    def test_reflected_branch_is_kabsch(self):
        H = covariances(300, reflected=True)
        self.assertTrue(
            np.allclose(
                ProcrustesData._batch_rotation(H),
                ProcrustesData._svd_rotation(H),
                atol=1e-12,
            )
        )

    def test_mixed_batch_rows_match_their_branch(self):
        """Masked assembly must not scramble row order."""
        H     = covariances(400)
        mixed = ProcrustesData._batch_rotation(H)
        for i in range(0, len(H), 37):
            single = ProcrustesData._batch_rotation(H[i : i + 1])
            self.assertTrue(np.allclose(mixed[i], single[0], atol=1e-9), f"row {i}")


class TestSvdRotationIsClean(unittest.TestCase):
    def test_does_not_mutate_its_input(self):
        H      = covariances(64)
        before = H.copy()
        ProcrustesData._svd_rotation(H)
        self.assertTrue(np.array_equal(H, before))

    def test_batch_rotation_does_not_mutate_its_input(self):
        H      = covariances(64)
        before = H.copy()
        ProcrustesData._batch_rotation(H)
        self.assertTrue(np.array_equal(H, before))

    def test_accepts_a_list_and_non_contiguous_input(self):
        H = covariances(8)
        self.assertEqual(ProcrustesData._batch_rotation(H.tolist()).shape, (8, 3, 3))

        strided = np.repeat(H, 2, axis=0)[::2]
        self.assertFalse(strided.flags["C_CONTIGUOUS"])
        self.assertTrue(
            np.allclose(
                ProcrustesData._batch_rotation(strided),
                ProcrustesData._batch_rotation(H),
                atol=1e-12,
            )
        )


class TestKnownRotationRoundTrips(unittest.TestCase):
    def test_recovers_a_known_rotation(self):
        rng    = np.random.default_rng(SEED + 2)
        points = rng.normal(size=(64, 3))

        angle  = np.radians(37.0)
        cos, sin = np.cos(angle), np.sin(angle)
        R_true = np.array(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]
        )
        moved = points @ R_true.T

        H = (moved.T @ points)[None]
        self.assertGreater(np.linalg.det(H[0]), 0.0)

        R = ProcrustesData._batch_rotation(H)[0]
        self.assertTrue(np.allclose(R, R_true, atol=1e-9))
        self.assertTrue(np.allclose(points @ R.T, moved, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
