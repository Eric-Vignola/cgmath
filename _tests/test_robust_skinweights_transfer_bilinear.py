import os
import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.resample import MeshDataResampler, ResampleMode
from cgmath.geometry.robust_skinweights_transfer_bilinear import (
    find_matches_closest_surface,
    robust_skinweights_transfer_bilinear,
    RobustBilinearSkinTransferOptions,
)
from cgmath.geometry.skin_weights import SkinData

EPSILON = np.finfo(np.float32).eps


def quad_grid(rows: int, cols: int, spacing: float = 1.0, z: float = 0.0) -> MeshData:
    """an all-quad grid in the XY plane, wound CCW when seen from +Z"""
    xs, ys = np.meshgrid(
        np.arange(cols) * spacing, np.arange(rows) * spacing, indexing="xy"
    )
    points = np.column_stack(
        [xs.ravel(), ys.ravel(), np.full(xs.size, z, dtype=np.float64)]
    )

    indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            indices += [i0, i0 + 1, i0 + cols + 1, i0 + cols]

    return MeshData(
        indices = np.array(indices, dtype=np.int32),
        counts  = np.full((rows - 1) * (cols - 1), 4, dtype=np.int32),
        points  = points,
    )


def ramp_weights(mesh: MeshData) -> SkinData:
    """two influences ramping linearly along X, summing to 1"""
    x = mesh.points[:, 0]
    t = (x - x.min()) / (x.max() - x.min())
    return SkinData(weights=np.column_stack([1.0 - t, t]), influences=["a", "b"])


def ramp_at(mesh: MeshData) -> np.ndarray:
    """the analytic value of ramp_weights sampled at another mesh's points"""
    return ramp_weights(mesh).weights


class TestRobustBilinearSkinTransfer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # a coarse source and a finer, offset target covering the same domain
        self.src_mesh = quad_grid(5, 5, spacing=1.0)
        self.src_skin = ramp_weights(self.src_mesh)

        self.dst_mesh = quad_grid(7, 7, spacing=4.0 / 6.0)
        self.options  = RobustBilinearSkinTransferOptions()

    # ------------------------------- options -------------------------------- #

    def test_options_defaults(self):
        options = RobustBilinearSkinTransferOptions()
        self.assertEqual(options.max_influences,     4)
        self.assertEqual(options.search_radius,      0.05)
        self.assertEqual(options.normal_threshold,   30)
        self.assertEqual(options.inpaint_iterations, 10)
        self.assertEqual(options.smooth_iterations,  10)
        self.assertEqual(options.smooth_strength,    0.1)

    def test_resample_mode_is_a_distinct_member(self):
        self.assertEqual(ResampleMode.ROBUST_BILINEAR.name, "ROBUST_BILINEAR")
        self.assertIn(ResampleMode.ROBUST_BILINEAR, list(ResampleMode))

    # ------------------------------ matching -------------------------------- #

    def test_matches_are_found_on_a_coincident_surface(self):
        sample_data = self.src_mesh.sample(self.dst_mesh)
        matched, weights = find_matches_closest_surface(
            sample_data, self.dst_mesh, self.src_skin.weights, 1.0, 30.0
        )

        self.assertTrue(np.all(matched))
        self.assertEqual(weights.shape, (self.dst_mesh.point_count, 2))

    def test_face_interior_samples_four_source_vertices(self):
        # a point at a quad centre must draw on all four corners -- this is what
        # triangulating the source would have thrown away
        centre      = np.array([[0.5, 0.5, 0.0]])
        sample_data = self.src_mesh.sample(centre)

        self.assertEqual(int((sample_data.geometry[0] >= 0).sum()), 4)
        self.assertEqual(int((sample_data.weights[0] > EPSILON).sum()), 4)
        self.assertAlmostEqual(float(sample_data.weights[0].sum()), 1.0, places=12)

    def test_normal_threshold_rejects_opposed_normals(self):
        sample_data = self.src_mesh.sample(self.dst_mesh)
        flipped = -self.dst_mesh.get_vertex_normals(
            angle_weighted=True, area_weighted=False
        )

        matched, _ = find_matches_closest_surface(
            sample_data, self.dst_mesh, self.src_skin.weights, 1.0, 30.0
        )
        self.assertTrue(np.all(matched))

        # same geometry, opposed normals -- every vertex must now be rejected
        sample_data.normals = flipped
        matched, _ = find_matches_closest_surface(
            sample_data, self.dst_mesh, self.src_skin.weights, 1.0, 30.0
        )
        self.assertFalse(np.any(matched))

    def test_aligned_normals_are_not_lost_to_arccos_domain_error(self):
        # identical normals give a dot product that can land just above 1.0
        sample_data = self.src_mesh.sample(self.dst_mesh)
        sample_data.normals = self.dst_mesh.get_vertex_normals(
            angle_weighted=True, area_weighted=False
        )

        matched, _ = find_matches_closest_surface(
            sample_data, self.dst_mesh, self.src_skin.weights, 1.0, 0.0
        )
        self.assertTrue(np.all(matched))

    # ------------------------------ transfer -------------------------------- #

    def test_transfer_reproduces_a_linear_ramp(self):
        weights = robust_skinweights_transfer_bilinear(
            self.src_mesh, self.src_skin, self.dst_mesh, self.options
        )

        self.assertEqual(weights.shape, (self.dst_mesh.point_count, 2))
        self.assertTrue(np.allclose(weights, ramp_at(self.dst_mesh), atol=1e-6))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))

    def test_precomputed_sample_data_gives_the_same_result(self):
        internal = robust_skinweights_transfer_bilinear(
            self.src_mesh, self.src_skin, self.dst_mesh, self.options
        )
        supplied = robust_skinweights_transfer_bilinear(
            self.src_mesh,
            self.src_skin,
            self.dst_mesh,
            self.options,
            sample_data=self.src_mesh.sample(self.dst_mesh),
        )

        self.assertTrue(np.array_equal(internal, supplied))

    def test_default_options_are_used_when_none(self):
        explicit = robust_skinweights_transfer_bilinear(
            self.src_mesh,
            self.src_skin,
            self.dst_mesh,
            RobustBilinearSkinTransferOptions(),
        )
        implicit = robust_skinweights_transfer_bilinear(
            self.src_mesh, self.src_skin, self.dst_mesh
        )

        self.assertTrue(np.array_equal(explicit, implicit))

    # ------------------------------ inpainting ------------------------------ #

    def test_unmatched_vertex_is_inpainted_from_its_neighbours(self):
        # lift one interior vertex clear of the search radius
        dst_mesh = quad_grid(5, 5, spacing=1.0)
        lifted   = 12
        dst_mesh.points[lifted, 2] = 1.0

        weights = robust_skinweights_transfer_bilinear(
            self.src_mesh, self.src_skin, dst_mesh, self.options
        )

        neighbors = dst_mesh.get_edge_vertex_neighbors()[lifted]
        neighbors = neighbors[neighbors >= 0]
        lo        = weights[neighbors].min(axis=0)
        hi        = weights[neighbors].max(axis=0)

        # inpaint and blur are both convex combinations of neighbour values
        self.assertTrue(np.all(weights[lifted] >= lo - 1e-9))
        self.assertTrue(np.all(weights[lifted] <= hi + 1e-9))
        self.assertAlmostEqual(float(weights[lifted].sum()), 1.0, places=9)

    def test_output_is_finite_and_non_negative(self):
        dst_mesh = quad_grid(5, 5, spacing=1.0)
        dst_mesh.points[12, 2] = 1.0

        weights = robust_skinweights_transfer_bilinear(
            self.src_mesh, self.src_skin, dst_mesh, self.options
        )

        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))

    def test_smoothing_can_be_disabled(self):
        options  = RobustBilinearSkinTransferOptions(smooth_iterations=0)
        dst_mesh = quad_grid(5, 5, spacing=1.0)
        dst_mesh.points[12, 2] = 1.0

        weights = robust_skinweights_transfer_bilinear(
            self.src_mesh, self.src_skin, dst_mesh, options
        )
        self.assertTrue(np.all(np.isfinite(weights)))

    # -------------------------------- errors -------------------------------- #

    def test_all_unmatched_raises(self):
        far_mesh = quad_grid(5, 5, spacing=1.0, z=100.0)

        with self.assertRaises(ValueError):
            robust_skinweights_transfer_bilinear(
                self.src_mesh, self.src_skin, far_mesh, self.options
            )

    def test_unreferenced_source_vertices_raise(self):
        src_mesh = quad_grid(5, 5, spacing=1.0)
        src_mesh.points = np.vstack([src_mesh.points, [[9.0, 9.0, 9.0]]])
        src_skin = SkinData(
            weights=np.ones((src_mesh.points.shape[0], 1)), influences=["a"]
        )

        with self.assertRaises(ValueError):
            robust_skinweights_transfer_bilinear(
                src_mesh, src_skin, self.dst_mesh, self.options
            )

    def test_unreferenced_target_vertices_raise(self):
        dst_mesh = quad_grid(5, 5, spacing=1.0)
        dst_mesh.points = np.vstack([dst_mesh.points, [[9.0, 9.0, 9.0]]])

        with self.assertRaises(ValueError):
            robust_skinweights_transfer_bilinear(
                self.src_mesh, self.src_skin, dst_mesh, self.options
            )

    def test_ngons_are_rejected(self):
        pentagon = MeshData(
            indices = np.arange(5, dtype=np.int32),
            counts  = np.array([5], dtype=np.int32),
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 1.0, 0.0],
                    [0.5, 1.5, 0.0],
                    [-0.5, 1.0, 0.0],
                ]
            ),
        )
        skin = SkinData(weights=np.ones((5, 1)), influences=["a"])

        with self.assertRaises(NotImplementedError):
            robust_skinweights_transfer_bilinear(
                pentagon, skin, self.dst_mesh, self.options
            )


class TestRobustBilinearResampleMode(unittest.TestCase):
    def setUp(self):
        super().setUp()

        root, fname = os.path.split(__file__)
        assets = os.path.join(root, "test_assets")

        self.src_mesh = MeshData.load(
            os.path.join(assets, "test_resample.src_mesh_data.npz")
        )
        self.dst_mesh = MeshData.load(
            os.path.join(assets, "test_resample.dst_mesh_data.npz")
        )
        self.src_skin = SkinData.load(
            os.path.join(assets, "test_resample.src_skin_data.npz")
        )

        self.resampler = MeshDataResampler(self.src_mesh, self.dst_mesh)

    def test_resample_skin_weights_robust_bilinear(self):
        options = RobustBilinearSkinTransferOptions()
        result = self.resampler.resample_skin_weights(
            self.src_skin, mode=ResampleMode.ROBUST_BILINEAR, options=options
        )

        self.assertEqual(
            result.weights.shape,
            (self.dst_mesh.point_count, len(self.src_skin.weights[0])),
        )
        self.assertTrue(np.all(np.isfinite(result.weights)))
        self.assertTrue(np.allclose(result.weights.sum(axis=1), 1.0))
        self.assertLessEqual(result.get_max_influences(), options.max_influences)

    def test_resample_reuses_the_cached_spatial_sample_data(self):
        self.resampler.resample_skin_weights(
            self.src_skin, mode=ResampleMode.ROBUST_BILINEAR
        )

        from cgmath.geometry.mesh import SampleMethod

        self.assertIn(
            (ResampleMode.SPATIAL, SampleMethod.BILINEAR),
            self.resampler._sample_cache,
        )

    def test_morph_target_robust_bilinear_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.resampler.resample_morph_target(
                self.src_mesh, mode=ResampleMode.ROBUST_BILINEAR
            )