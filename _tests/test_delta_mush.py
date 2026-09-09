import unittest

import numpy as np
from cgmath.geometry.delta_mush import DeltaMushData
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.utils._numba._delta_mush import (
    _delta_mush_decode_ddm_numpy,
    _delta_mush_decode_numpy,
    _delta_mush_decode_procrustes_numpy,
    _delta_mush_decode_tbn_numpy,
    _delta_mush_encode_numpy,
    _delta_mush_encode_tbn_numpy,
    _polar_decompose_3x3,
)


def _make_grid_mesh(n: int = 8, jitter: float = 0.0, seed: int = 0) -> MeshData:
    """Build a quad grid mesh with optional Z jitter for tests."""
    rng  = np.random.default_rng(seed)
    xs   = np.linspace(0.0, 1.0, n)
    ys   = np.linspace(0.0, 1.0, n)
    grid = np.array([[x, y, 0.0] for y in ys for x in xs])
    if jitter > 0.0:
        grid[:, 2] += rng.normal(scale=jitter, size=grid.shape[0])

    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n + 1
            d = a + n
            indices.extend([a, b, c, d])
    counts = np.full((n - 1) ** 2, 4, dtype=np.int32)

    return MeshData(
        points  = grid,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = counts,
    )


class TestDeltaMushData(unittest.TestCase):
    """Behavioural tests that should pass for ALL three methods."""

    METHODS = ("TBN", "PROCRUSTES", "DDM")

    def test_default_method_is_tbn(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=0)
        mush = DeltaMushData(rest)
        self.assertEqual(mush.method, "TBN")

    def test_method_property_setter_normalises_case(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=0)
        mush = DeltaMushData(rest, method="procrustes")
        self.assertEqual(mush.method, "PROCRUSTES")
        mush.method = "ddm"
        self.assertEqual(mush.method, "DDM")
        mush.method = "tbn"
        self.assertEqual(mush.method, "TBN")

    def test_method_property_rejects_unknown(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=0)
        mush = DeltaMushData(rest)
        with self.assertRaises(ValueError):
            mush.method = "BOGUS"

    def test_method_change_invalidates_bind(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=0)
        mush = DeltaMushData(rest, smooth_iterations=2, method="TBN")
        mush.bind()
        self.assertTrue(mush.valid)
        mush.method = "PROCRUSTES"
        self.assertFalse(mush.valid)

    def test_bind_apply_identity_all_methods(self):
        """Apply on the rest pose should reconstruct rest exactly for any method."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=1)
        for method in self.METHODS:
            with self.subTest(method=method):
                mush = DeltaMushData(
                    rest, smooth_iterations=8, pin_borders=True, method=method
                )
                mush.bind()
                out = mush.apply(rest)
                np.testing.assert_allclose(out.points, rest.points, atol=1e-7)

    def test_borders_pinned_when_requested(self):
        """When pin_borders=True border vertices stay put after smoothing."""
        rest = _make_grid_mesh(n=8, jitter=0.0, seed=2)
        # add jitter only on borders so smoothing would normally move them
        border_idx = rest.get_border_vertices(flatten=True)
        rest.points[border_idx, 2] = 0.25

        mush = DeltaMushData(rest, smooth_iterations=20, pin_borders=True)
        mush.bind()

        np.testing.assert_allclose(
            mush.smoothed_points[border_idx], rest.points[border_idx], atol=1e-12
        )

    def test_borders_move_when_unpinned(self):
        """When pin_borders=False border vertices may smooth like the rest."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=3)
        mush = DeltaMushData(rest, smooth_iterations=20, pin_borders=False)
        mush.bind()

        # at least one border vertex should have moved
        border_idx = rest.get_border_vertices(flatten=True)
        delta      = np.abs(mush.smoothed_points[border_idx] - rest.points[border_idx])
        self.assertGreater(delta.max(), 1e-6)

    def test_smoothing_actually_smooths(self):
        """Variance of interior vertices should drop after smoothing."""
        rest = _make_grid_mesh(n=12, jitter=0.1, seed=4)
        interior = np.setdiff1d(
            np.arange(rest.point_count), rest.get_border_vertices(flatten=True)
        )
        rest_var = rest.points[interior, 2].var()

        mush     = DeltaMushData(rest, smooth_iterations=15, pin_borders=True)
        mush.bind()

        smooth_var = mush.smoothed_points[interior, 2].var()
        self.assertLess(smooth_var, rest_var * 0.5)

    def test_reconstruct_after_translation_all_methods(self):
        """A pure translation must be reproduced exactly for any method."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=5)
        for method in self.METHODS:
            with self.subTest(method=method):
                mush = DeltaMushData(
                    rest, smooth_iterations=10, pin_borders=True, method=method
                )
                mush.bind()
                deformed = rest.copy()
                deformed.points = deformed.points + np.array([1.5, -2.0, 0.7])
                out = mush.apply(deformed)
                np.testing.assert_allclose(out.points, deformed.points, atol=1e-6)

    def test_reconstruct_after_rotation_all_methods(self):
        """A pure rigid rotation must round-trip surface detail for any method."""
        rest = _make_grid_mesh(n=10, jitter=0.0, seed=6)
        rng  = np.random.default_rng(123)
        rest.points[:, 2] += rng.normal(scale=0.02, size=rest.point_count)

        # rotate around Y by 30deg
        angle = np.deg2rad(30)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

        for method in self.METHODS:
            with self.subTest(method=method):
                mush = DeltaMushData(
                    rest, smooth_iterations=10, pin_borders=True, method=method
                )
                mush.bind()
                deformed = rest.copy()
                deformed.points = deformed.points @ R.T
                out = mush.apply(deformed)
                # the reconstruction should match the rotated rest within
                # smoothing-frame discretisation tolerance
                np.testing.assert_allclose(out.points, deformed.points, atol=5e-3)

    def test_pointcloud_path_falls_back_from_tbn(self):
        """Bind/apply works with bare point cloud + neighbours; TBN auto-falls back to DDM."""
        rest = _make_grid_mesh(n=6, jitter=0.0, seed=7)
        pts  = rest.points.copy()
        nbrs = rest.get_edge_vertex_neighbors()
        # ask for TBN -- should silently switch to DDM since there is no mesh
        mush = DeltaMushData(
            pts,
            neighbors         = nbrs,
            smooth_iterations = 5,
            pin_borders       = False,
            method            = "TBN",
        )
        mush.bind()
        self.assertEqual(mush.method, "DDM")

        out = mush.apply(pts)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out, pts, atol=1e-7)

    def test_pointcloud_requires_neighbors(self):
        """Constructing on a raw cloud without neighbours must raise."""
        pts = np.zeros((5, 3))
        with self.assertRaises(ValueError):
            DeltaMushData(pts)

    def test_weight_zero_returns_smoothed_mesh(self):
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=8)
        mush = DeltaMushData(rest, smooth_iterations=10, pin_borders=True, weight=0.0)
        mush.bind()
        out = mush.apply(rest)
        np.testing.assert_allclose(out.points, mush.smoothed_points, atol=1e-9)

    def test_weight_blend_midpoint(self):
        rest = _make_grid_mesh(n=8, jitter=0.0, seed=9)
        mush = DeltaMushData(rest, smooth_iterations=10, pin_borders=True, weight=0.5)
        mush.bind()

        deformed = rest.copy()
        deformed.points = deformed.points + np.array([0.0, 0.0, 1.0])

        out = mush.apply(deformed)

        # 0.5 blend between deformed and full reconstruction.  Since for
        # a pure translation the full reconstruction equals deformed,
        # the smoothed deformed pose equals translated smooth_rest, and
        # the result equals deformed exactly.
        np.testing.assert_allclose(out.points, deformed.points, atol=1e-6)

    def test_weight_clamped(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=10)
        mush = DeltaMushData(rest, smooth_iterations=2, pin_borders=False)
        mush.weight = 5.0
        self.assertEqual(mush.weight, 1.0)
        mush.weight = -1.0
        self.assertEqual(mush.weight, 0.0)

    def test_smooth_iterations_invalidates_bind(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=11)
        mush = DeltaMushData(rest, smooth_iterations=2, pin_borders=False)
        mush.bind()
        self.assertTrue(mush.valid)
        mush.smooth_iterations = 5
        self.assertFalse(mush.valid)

    def test_zero_iterations_yields_zero_deltas_all_methods(self):
        rest = _make_grid_mesh(n=5, jitter=0.05, seed=12)
        for method in self.METHODS:
            with self.subTest(method=method):
                mush = DeltaMushData(
                    rest,
                    smooth_iterations = 0,
                    pin_borders       = False,
                    method            = method,
                )
                mush.bind()
                # smooth == rest, so deltas are zero in any frame
                np.testing.assert_allclose(mush.smoothed_points, rest.points)
                np.testing.assert_allclose(mush.local_deltas, 0.0, atol=1e-12)

    def test_apply_auto_binds_all_methods(self):
        """apply() should auto-bind when called before an explicit bind()."""
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=13)
        for method in self.METHODS:
            with self.subTest(method=method):
                mush = DeltaMushData(rest, smooth_iterations=2, method=method)
                self.assertFalse(mush.valid)
                result = mush.apply(rest)
                self.assertTrue(mush.valid)
                # identity: applying mush to the rest pose returns the rest pose
                np.testing.assert_allclose(result.points, rest.points, atol=1e-7)

    def test_target_shape_mismatch_raises(self):
        rest = _make_grid_mesh(n=4, jitter=0.0, seed=14)
        mush = DeltaMushData(rest, smooth_iterations=2)
        mush.bind()
        with self.assertRaises(ValueError):
            mush.apply(np.zeros((rest.point_count + 1, 3)))


class TestDeltaMushNumpyFallback(unittest.TestCase):
    """Direct tests for the numpy fallback paths used when numba fails."""

    def test_legacy_encode_decode_roundtrip(self):
        """Original (averaged-Laplacian-N) encode/decode round-trips on rest."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=21)
        mush = DeltaMushData(rest, smooth_iterations=10, pin_borders=True)
        mush.bind()

        nbrs   = np.ascontiguousarray(rest.get_edge_vertex_neighbors(), dtype=np.int32)
        deltas = _delta_mush_encode_numpy(rest.points, mush.smoothed_points, nbrs)
        out    = _delta_mush_decode_numpy(mush.smoothed_points, nbrs, deltas)
        np.testing.assert_allclose(out, rest.points, atol=1e-7)

    def test_tbn_numpy_round_trip(self):
        """TBN encode -> decode round-trips on the rest pose."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=31)
        mush = DeltaMushData(rest, smooth_iterations=10, method="TBN")
        mush.bind()

        deltas = _delta_mush_encode_tbn_numpy(
            rest.points, mush.smoothed_points, mush._first_nbrs
        )
        out = _delta_mush_decode_tbn_numpy(
            mush.smoothed_points, mush._first_nbrs, deltas
        )
        np.testing.assert_allclose(out, rest.points, atol=1e-7)

    def test_tbn_numpy_matches_numba(self):
        """Numpy and numba TBN paths produce the same encoded deltas."""
        rest = _make_grid_mesh(n=6, jitter=0.05, seed=32)
        mush = DeltaMushData(rest, smooth_iterations=8, method="TBN")
        mush.bind()
        np_deltas = _delta_mush_encode_tbn_numpy(
            rest.points, mush.smoothed_points, mush._first_nbrs
        )
        np.testing.assert_allclose(mush.local_deltas, np_deltas, atol=1e-9)

    def test_procrustes_numpy_round_trip(self):
        """Procrustes apply on the rest pose round-trips."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=41)
        mush = DeltaMushData(rest, smooth_iterations=10, method="PROCRUSTES")
        mush.bind()

        out = _delta_mush_decode_procrustes_numpy(
            mush.smoothed_points,
            mush.smoothed_points,  # smooth_def == smooth_rest for identity
            mush.neighbors,
            mush.local_deltas,
        )
        np.testing.assert_allclose(out, rest.points, atol=1e-7)

    def test_ddm_numpy_round_trip(self):
        """DDM apply on the rest pose round-trips."""
        rest = _make_grid_mesh(n=8, jitter=0.05, seed=51)
        mush = DeltaMushData(rest, smooth_iterations=10, method="DDM")
        mush.bind()

        out = _delta_mush_decode_ddm_numpy(
            mush.smoothed_points,
            mush.neighbors,
            mush._ddm_offsets,
            mush._ddm_weights,
            mush.local_deltas,
        )
        np.testing.assert_allclose(out, rest.points, atol=1e-7)


class TestPolarDecompose3x3(unittest.TestCase):
    """Direct tests for the Higham polar-decomposition kernel.

    Regression coverage for the rank-deficient guard: planar (XY-only)
    smoothed-rest neighbour offsets produce a cross-covariance whose
    z-row/column are zero, so ``det(M) == 0`` and Higham's iteration
    cannot proceed without regularisation.  The expected rotation in
    that case is the identity.
    """

    def _polar(self, M, max_iter=16):
        r = _polar_decompose_3x3(
            M[0, 0],
            M[0, 1],
            M[0, 2],
            M[1, 0],
            M[1, 1],
            M[1, 2],
            M[2, 0],
            M[2, 1],
            M[2, 2],
            max_iter,
        )
        return np.asarray(r, dtype=np.float64).reshape(3, 3)

    def test_identity_input(self):
        R = self._polar(np.eye(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_planar_diagonal_returns_identity(self):
        """Rank-2 SPSD diag(1, 1, 0) -> polar factor = identity."""
        M = np.diag([1.0, 1.0, 0.0])
        R = self._polar(M)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

    def test_planar_covariance_returns_identity(self):
        """Cross-covariance from XY-plane offsets -> polar factor = identity."""
        a = np.array(
            [
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [-0.2, 0.0, 0.0],
                [0.0, -0.2, 0.0],
            ]
        )
        M = a.T @ a
        # sanity: planar cross-covariance is rank-deficient
        self.assertAlmostEqual(np.linalg.det(M), 0.0, places=12)
        R = self._polar(M)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

    def test_full_rank_rotation_recovered(self):
        """Full-rank M = R * S round-trips: polar(M) ~= R."""
        # rotation around y by 30deg
        c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
        R_true = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        S      = np.diag([1.5, 0.8, 1.2])
        M      = R_true @ S
        R      = self._polar(M)
        np.testing.assert_allclose(R, R_true, atol=1e-6)


if __name__ == "__main__":
    unittest.main()