import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.patch_relax import PatchRelaxData
from cgmath.geometry.utils import (
    build_vertex_rings,
    compute_decal_maps,
    compute_span_weights,
)
from cgmath.geometry.utils._numba._delta_mush import _polar_decompose_3x3
from cgmath.geometry.utils._numba._patch_relax import _span_weight


def _make_grid_mesh(n: int = 9, height: float = 0.0, power: float = 1.0) -> MeshData:
    """Quad grid, optionally bumped in Z and/or graded along X.

    *height* adds a sine bump so the baseline carries detail worth
    restoring; *power* grades the column spacing so the patch has
    non-uniform edge spans.
    """
    t    = np.linspace(0.0, 1.0, n)
    xs   = t**power
    grid = np.array([[x, y, 0.0] for y in t for x in xs])
    if height:
        grid[:, 2] = height * np.sin(np.pi * grid[:, 0]) * np.sin(np.pi * grid[:, 1])

    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices.extend([a, a + 1, a + n + 1, a + n])

    return MeshData(
        points  = grid,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
    )


def _make_cube_mesh() -> MeshData:
    """Closed cube -- every vertex ring closes with valence 3."""
    points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return MeshData(
        points  = points,
        indices = np.asarray(faces, dtype=np.int32).ravel(),
        counts  = np.full(6, 4, dtype=np.int32),
    )


def _uniform_laplacian(mesh, points, iterations, pinned, step=0.5):
    """Reference umbrella-operator smoother, built independently of the
    implementation under test, to contrast against span-aware weights."""
    neighbors    = mesh.get_edge_vertex_neighbors()
    valid        = neighbors >= 0
    counts       = valid.sum(axis=1)
    out          = np.asarray(points, dtype=np.float64).copy()
    free         = np.ones(out.shape[0], dtype=bool)
    free[pinned] = False

    for _ in range(iterations):
        gathered         = out[np.where(valid, neighbors, 0)]
        gathered[~valid] = 0.0
        centroid         = gathered.sum(axis=1) / np.maximum(counts, 1)[:, None]
        delta            = np.where(counts[:, None] > 0, centroid - out, 0.0)
        out[free] += step * delta[free]

    return out


def _row_spans(points, n, row):
    """Horizontal edge lengths along one row of a grid."""
    idx = np.arange(row * n, row * n + n)
    return np.abs(np.diff(points[idx, 0]))


class TestVertexRings(unittest.TestCase):
    """Winding-ordered 1-ring adjacency."""

    def test_grid_valence_and_boundary_flags(self):
        n    = 5
        mesh = _make_grid_mesh(n)
        ring, valence, is_boundary = build_vertex_rings(
            mesh.indices, mesh.counts, n * n
        )

        expected_valence = np.array(
            [
                [2, 3, 3, 3, 2],
                [3, 4, 4, 4, 3],
                [3, 4, 4, 4, 3],
                [3, 4, 4, 4, 3],
                [2, 3, 3, 3, 2],
            ]
        )
        self.assertTrue(np.array_equal(valence.reshape(n, n), expected_valence))

        # only the interior 3x3 block closes its ring
        interior             = np.zeros((n, n), dtype=bool)
        interior[1:-1, 1:-1] = True
        self.assertTrue(np.array_equal(~is_boundary.reshape(n, n), interior))

        # padded slots stay marked
        for i in range(n * n):
            self.assertTrue(np.all(ring[i, valence[i] :] == -1))

    def test_ring_matches_edge_neighbors(self):
        mesh = _make_grid_mesh(5)
        ring, valence, _ = build_vertex_rings(mesh.indices, mesh.counts, 25)
        edge_neighbors = mesh.get_edge_vertex_neighbors()

        for i in range(25):
            ordered  = set(ring[i, : valence[i]].tolist())
            expected = set(edge_neighbors[i][edge_neighbors[i] >= 0].tolist())
            self.assertEqual(ordered, expected)

    def test_ring_is_walk_ordered(self):
        """Consecutive ring entries must share a face with the centre."""
        mesh = _make_grid_mesh(5)
        ring, valence, is_boundary = build_vertex_rings(mesh.indices, mesh.counts, 25)
        faces = {frozenset(f) for f in mesh.f2v}

        for i in range(25):
            val   = int(valence[i])
            steps = val if not is_boundary[i] else val - 1
            for k in range(steps):
                a = int(ring[i, k])
                b = int(ring[i, (k + 1) % val])
                self.assertTrue(
                    any({i, a, b} <= face for face in faces),
                    f"ring entries {a},{b} of vertex {i} do not share a face",
                )

    def test_closed_mesh_has_no_boundary(self):
        cube = _make_cube_mesh()
        _, valence, is_boundary = build_vertex_rings(cube.indices, cube.counts, 8)
        self.assertTrue(np.array_equal(valence, np.full(8, 3, dtype=np.int32)))
        self.assertFalse(np.any(is_boundary))

    def test_degenerate_faces_are_ignored(self):
        """A dangling two-corner face contributes no ring segment."""
        mesh    = _make_grid_mesh(3)
        indices = np.concatenate([mesh.indices, np.array([0, 8], dtype=np.int32)])
        counts  = np.concatenate([mesh.counts, np.array([2], dtype=np.int32)])

        ring, valence, _ = build_vertex_rings(indices, counts, 9)
        base_ring, base_valence, _ = build_vertex_rings(mesh.indices, mesh.counts, 9)
        self.assertTrue(np.array_equal(valence, base_valence))
        self.assertTrue(np.array_equal(ring[:, : base_ring.shape[1]], base_ring))

    def test_isolated_vertex_has_zero_valence(self):
        mesh   = _make_grid_mesh(3)
        points = np.vstack([mesh.points, np.array([[5.0, 5.0, 5.0]])])
        mesh   = MeshData(points=points, indices=mesh.indices, counts=mesh.counts)

        _, valence, is_boundary = build_vertex_rings(mesh.indices, mesh.counts, 10)
        self.assertEqual(valence[9], 0)
        self.assertTrue(is_boundary[9])

        # and it survives relaxation untouched
        relaxed = PatchRelaxData(mesh, iterations=5).relax(points)
        self.assertTrue(np.allclose(relaxed[9], points[9]))


class TestDecalMaps(unittest.TestCase):
    """Geodesic-polar stencil flattening (paper section 2)."""

    def setUp(self):
        self.n    = 5
        self.mesh = _make_grid_mesh(self.n)
        self.ring, self.valence, self.is_boundary = build_vertex_rings(
            self.mesh.indices, self.mesh.counts, self.n * self.n
        )
        self.decals = compute_decal_maps(
            self.mesh.points, self.ring, self.valence, self.is_boundary
        )

    def test_edge_lengths_are_preserved(self):
        for i in range(self.n * self.n):
            for k in range(int(self.valence[i])):
                j        = int(self.ring[i, k])
                expected = np.linalg.norm(self.mesh.points[j] - self.mesh.points[i])
                self.assertAlmostEqual(
                    float(np.linalg.norm(self.decals[i, k])), expected, places=9
                )

    def test_interior_stencil_spans_full_disk(self):
        """A regular interior stencil lands on 90 degree spokes."""
        center = 12
        d      = self.decals[center, : self.valence[center]]
        angles = np.sort(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0)
        self.assertTrue(np.allclose(angles, [0.0, 90.0, 180.0, 270.0], atol=1e-6))

    def test_boundary_stencil_spans_half_disk(self):
        """Boundary fans normalise to pi so they map to a half-disk."""
        for i in np.flatnonzero(self.is_boundary):
            val = int(self.valence[i])
            if val < 2:
                continue
            d     = self.decals[i, :val]
            first = np.arctan2(d[0, 1], d[0, 0])
            last  = np.arctan2(d[val - 1, 1], d[val - 1, 0])
            self.assertAlmostEqual(abs(last - first), np.pi, places=9)

    def test_padded_slots_are_zero(self):
        for i in range(self.n * self.n):
            self.assertTrue(np.all(self.decals[i, int(self.valence[i]) :] == 0.0))


class TestSpanWeights(unittest.TestCase):
    """Span-aware weighting (paper section 3)."""

    def test_weights_are_normalised(self):
        mesh = _make_grid_mesh(7, power=2.0)
        ring, valence, is_boundary = build_vertex_rings(mesh.indices, mesh.counts, 49)
        decals  = compute_decal_maps(mesh.points, ring, valence, is_boundary)
        weights = compute_span_weights(decals, valence)

        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
        self.assertTrue(np.all(weights >= 0.0))
        for i in range(49):
            self.assertTrue(np.all(weights[i, int(valence[i]) :] == 0.0))

    def test_regular_stencil_is_uniform(self):
        """On a grid patch the weights reduce to a bilinear blend."""
        mesh = _make_grid_mesh(5)
        ring, valence, is_boundary = build_vertex_rings(mesh.indices, mesh.counts, 25)
        decals  = compute_decal_maps(mesh.points, ring, valence, is_boundary)
        weights = compute_span_weights(decals, valence)
        self.assertTrue(np.allclose(weights[12, :4], 0.25))

    def test_zero_valence_stencil_is_zero(self):
        decals  = np.zeros((1, 4, 2))
        valence = np.zeros(1, dtype=np.int32)
        self.assertTrue(np.all(compute_span_weights(decals, valence) == 0.0))

    def test_degenerate_stencil_falls_back_to_uniform(self):
        """A collapsed decal map carries no span information."""
        decals  = np.zeros((1, 4, 2))
        valence = np.full(1, 4, dtype=np.int32)
        self.assertTrue(np.allclose(compute_span_weights(decals, valence), 0.25))

    def test_single_neighbour_edge_has_no_weight(self):
        """A lone spoke has no opposing edge to measure a span against."""
        decal_row = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(_span_weight(decal_row, 1, 0), 0.0)


class TestStencilRotation(unittest.TestCase):
    """The per-vertex Procrustes fit driving the section 4 displacement."""

    @staticmethod
    def _covariance(rest, posed):
        """The cross-covariance ``_relax_step`` accumulates per vertex."""
        return np.einsum("ki,kj->ij", posed, rest)

    def test_recovers_a_known_rotation(self):
        rest  = np.eye(3)
        theta = 0.6
        expected = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cov    = self._covariance(rest, rest @ expected.T)
        result = np.array(_polar_decompose_3x3(*cov.ravel(), 16)).reshape(3, 3)
        self.assertTrue(np.allclose(result, expected, atol=1e-8))

    def test_mirrored_stencil_yields_a_proper_rotation(self):
        """A folded-over patch must not resolve to a mirror transform."""
        rng      = np.random.default_rng(7)
        rest     = rng.normal(size=(6, 3))
        mirrored = rest * np.array([1.0, 1.0, -1.0])
        cov      = self._covariance(rest, mirrored)
        self.assertLess(np.linalg.det(cov), 0.0)  # the fit is improper

        result = np.array(_polar_decompose_3x3(*cov.ravel(), 16)).reshape(3, 3)
        self.assertGreater(np.linalg.det(result), 0.0)
        self.assertTrue(np.allclose(result @ result.T, np.eye(3), atol=1e-6))

    def test_planar_stencil_is_handled(self):
        """A flat patch gives a rank-deficient covariance."""
        rest = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
        cov  = self._covariance(rest, rest)
        self.assertAlmostEqual(np.linalg.det(cov), 0.0)

        result = np.array(_polar_decompose_3x3(*cov.ravel(), 16)).reshape(3, 3)
        self.assertTrue(np.allclose(result, np.eye(3), atol=1e-6))


class TestPatchRelaxData(unittest.TestCase):
    """End-to-end behaviour of the operator."""

    def test_rest_pose_is_a_fixed_point(self):
        mesh   = _make_grid_mesh(9, height=0.3)
        result = PatchRelaxData(mesh, iterations=25).apply(mesh)
        self.assertTrue(np.allclose(result.points, mesh.points, atol=1e-9))

    def test_apply_returns_a_mesh_copy(self):
        mesh    = _make_grid_mesh(5, height=0.2)
        relaxer = PatchRelaxData(mesh, iterations=3)
        result  = relaxer.apply(mesh)

        self.assertIsInstance(result, MeshData)
        self.assertIsNot(result, mesh)
        self.assertTrue(np.array_equal(result.indices, mesh.indices))

    def test_apply_accepts_a_point_array(self):
        mesh    = _make_grid_mesh(5, height=0.2)
        relaxer = PatchRelaxData(mesh, iterations=3)
        result  = relaxer.apply(mesh.points)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, mesh.points.shape)

    def test_restores_baseline_detail(self):
        """alpha=1 rebuilds detail the pose lost; alpha=0 leaves it flat."""
        mesh = _make_grid_mesh(13, height=0.35)
        flat = _make_grid_mesh(13, height=0.0).points

        restored = PatchRelaxData(mesh, iterations=300, alpha=1.0).relax(flat)
        smoothed = PatchRelaxData(mesh, iterations=300, alpha=0.0).relax(flat)

        peak = mesh.points[:, 2].max()
        self.assertGreater(restored[:, 2].max(), 0.9 * peak)
        self.assertLess(smoothed[:, 2].max(), 0.01 * peak)

        restored_err = np.linalg.norm(restored - mesh.points, axis=1).mean()
        smoothed_err = np.linalg.norm(smoothed - mesh.points, axis=1).mean()
        self.assertLess(restored_err, 0.1 * smoothed_err)

    def test_alpha_interpolates_monotonically(self):
        mesh  = _make_grid_mesh(9, height=0.35)
        flat  = _make_grid_mesh(9, height=0.0).points

        peaks = []
        for alpha in (0.0, 0.25, 0.5, 1.0):
            out = PatchRelaxData(mesh, iterations=200, alpha=alpha).relax(flat)
            peaks.append(out[:, 2].max())

        self.assertTrue(all(a < b for a, b in zip(peaks, peaks[1:])))

    def test_is_invariant_to_rigid_motion(self):
        """A rigidly moved pose is already relaxed -- it must not drift."""
        mesh  = _make_grid_mesh(9, height=0.3)
        theta = 0.7
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        moved  = mesh.points @ rot.T + np.array([5.0, -2.0, 1.0])

        result = PatchRelaxData(mesh, iterations=25).relax(moved)
        self.assertTrue(np.allclose(result, moved, atol=1e-8))

    def test_restores_detail_under_rotation(self):
        """The per-vertex rotation must transfer detail into the posed frame."""
        mesh  = _make_grid_mesh(9, height=0.35)
        theta = np.radians(40.0)
        rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ]
        )
        flat_rotated = _make_grid_mesh(9, height=0.0).points @ rot.T
        target       = mesh.points @ rot.T

        result = PatchRelaxData(mesh, iterations=300, alpha=1.0).relax(flat_rotated)
        self.assertLess(np.linalg.norm(result - target, axis=1).mean(), 0.01)

    def test_resolves_a_foldover(self):
        mesh   = _make_grid_mesh(5)
        folded = mesh.points.copy()
        folded[12] += np.array([0.45, 0.45, 0.0])

        result = PatchRelaxData(mesh, iterations=60).relax(folded)
        before = np.linalg.norm(folded[12] - mesh.points[12])
        after  = np.linalg.norm(result[12] - mesh.points[12])
        self.assertLess(after, 0.01 * before)

    def test_preserves_spans_where_laplacian_equalises_them(self):
        """The paper's central claim, against an independent umbrella smoother."""
        n      = 9
        mesh   = _make_grid_mesh(n, power=3.0)
        pinned = np.asarray(mesh.get_border_vertices(flatten=True), dtype=np.int64)

        baseline_spans = _row_spans(mesh.points, n, 4)
        baseline_ratio = baseline_spans.max() / baseline_spans.min()

        relaxed   = PatchRelaxData(mesh, iterations=200).relax(mesh.points)
        laplacian = _uniform_laplacian(mesh, mesh.points, 200, pinned)

        relaxed_ratio   = _row_spans(relaxed, n, 4)
        relaxed_ratio   = relaxed_ratio.max() / relaxed_ratio.min()
        laplacian_spans = _row_spans(laplacian, n, 4)
        laplacian_ratio = laplacian_spans.max() / laplacian_spans.min()

        # patch relaxation holds the graded layout; the umbrella operator
        # pulls it toward uniform spacing
        self.assertAlmostEqual(relaxed_ratio, baseline_ratio, delta=0.01)
        self.assertLess(laplacian_ratio, 0.5 * baseline_ratio)

    def test_surface_blend_stays_closer_to_the_surface(self):
        mesh  = _make_grid_mesh(11, height=0.35)
        rng   = np.random.default_rng(5)
        noisy = mesh.points + rng.normal(scale=0.02, size=mesh.points.shape)

        volume_free = PatchRelaxData(mesh, iterations=40, surface_blend=0.0).relax(
            noisy
        )
        volume_kept = PatchRelaxData(mesh, iterations=40, surface_blend=0.8).relax(
            noisy
        )

        self.assertTrue(np.all(np.isfinite(volume_kept)))
        target = mesh.points[:, 2].mean()
        self.assertLess(
            abs(volume_kept[:, 2].mean() - target),
            abs(volume_free[:, 2].mean() - target),
        )

    def test_pin_borders_holds_the_boundary(self):
        mesh    = _make_grid_mesh(7, height=0.25)
        rng     = np.random.default_rng(2)
        noisy   = mesh.points + rng.normal(scale=0.02, size=mesh.points.shape)
        borders = np.asarray(mesh.get_border_vertices(flatten=True), dtype=np.int64)

        relaxer = PatchRelaxData(mesh, iterations=20, pin_borders=True)
        pinned  = relaxer.relax(noisy)
        self.assertTrue(np.array_equal(pinned[borders], noisy[borders]))

        relaxer.pin_borders = False
        free                = relaxer.relax(noisy)
        self.assertFalse(np.array_equal(free[borders], noisy[borders]))

    def test_mask_scales_the_step(self):
        mesh  = _make_grid_mesh(7, height=0.25)
        rng   = np.random.default_rng(4)
        noisy = mesh.points + rng.normal(scale=0.02, size=mesh.points.shape)

        frozen = PatchRelaxData(
            mesh, iterations=20, mask=np.zeros(mesh.points.shape[0])
        ).relax(noisy)
        self.assertTrue(np.allclose(frozen, noisy))

        half = PatchRelaxData(
            mesh, iterations=20, mask=np.full(mesh.points.shape[0], 0.5)
        ).relax(noisy)
        full = PatchRelaxData(mesh, iterations=20).relax(noisy)
        self.assertFalse(np.allclose(half, noisy))
        self.assertFalse(np.allclose(half, full))

    def test_reassigned_mask_takes_effect(self):
        """The step scale must not be cached across a mask change."""
        mesh    = _make_grid_mesh(7, height=0.25)
        rng     = np.random.default_rng(6)
        noisy   = mesh.points + rng.normal(scale=0.02, size=mesh.points.shape)

        relaxer = PatchRelaxData(mesh, iterations=20)
        self.assertFalse(np.allclose(relaxer.relax(noisy), noisy))

        relaxer.mask = np.zeros(mesh.points.shape[0])
        self.assertTrue(np.allclose(relaxer.relax(noisy), noisy))

    def test_zero_iterations_is_a_noop(self):
        mesh  = _make_grid_mesh(5, height=0.2)
        noisy = mesh.points + 0.05
        self.assertTrue(
            np.array_equal(PatchRelaxData(mesh, iterations=0).relax(noisy), noisy)
        )

    def test_works_on_a_closed_mesh(self):
        cube   = _make_cube_mesh()
        rng    = np.random.default_rng(11)
        noisy  = cube.points + rng.normal(scale=0.05, size=cube.points.shape)

        result = PatchRelaxData(cube, iterations=30).relax(noisy)
        self.assertTrue(np.all(np.isfinite(result)))
        before = np.linalg.norm(noisy - cube.points, axis=1).mean()
        after  = np.linalg.norm(result - cube.points, axis=1).mean()
        self.assertLess(after, before)

    def test_bind_populates_the_cache(self):
        mesh    = _make_grid_mesh(5, height=0.2)
        relaxer = PatchRelaxData(mesh)
        self.assertFalse(relaxer.valid)

        relaxer.bind()
        self.assertTrue(relaxer.valid)
        self.assertEqual(relaxer.weights.shape[0], mesh.points.shape[0])
        self.assertEqual(relaxer.decal_maps.shape[-1], 2)
        self.assertIsNotNone(relaxer.points)

    def test_apply_auto_binds(self):
        mesh    = _make_grid_mesh(5, height=0.2)
        relaxer = PatchRelaxData(mesh, iterations=2)
        self.assertFalse(relaxer.valid)
        relaxer.apply(mesh)
        self.assertTrue(relaxer.valid)

    def test_runtime_properties_round_trip(self):
        mesh    = _make_grid_mesh(5)
        relaxer = PatchRelaxData(mesh)

        relaxer.iterations    = 7
        relaxer.alpha         = 0.25
        relaxer.surface_blend = 2.0  # clamped
        relaxer.step_size     = 0.75
        relaxer.pin_borders   = False
        self.assertEqual(relaxer.iterations,    7)
        self.assertEqual(relaxer.alpha,         0.25)
        self.assertEqual(relaxer.surface_blend, 1.0)
        self.assertEqual(relaxer.step_size,     0.75)
        self.assertFalse(relaxer.pin_borders)

        with self.assertRaises(ValueError):
            relaxer.iterations = -1

    def test_rejects_bad_input(self):
        mesh = _make_grid_mesh(5)

        with self.assertRaises(ValueError):
            PatchRelaxData(mesh.points)

        with self.assertRaises(ValueError):
            PatchRelaxData(mesh, mask=np.ones(3))

        with self.assertRaises(ValueError):
            PatchRelaxData(mesh, iterations=2).relax(np.zeros((3, 3)))