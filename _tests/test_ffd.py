import unittest

import numpy as np
from cgmath.geometry.ffd import FFDData
from cgmath.geometry.utils.main import (
    assign_cells,
    get_cell_corners,
    inverse_trilinear,
    trilinear,
)


class TestTrilinearPrimitives(unittest.TestCase):
    """Tests for the low-level trilinear interpolation helpers."""

    def _unit_cube_corners(self, n: int = 1) -> np.ndarray:
        """Return (n, 8, 3) corners of a unit cube [0,1]^3."""
        c = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        return np.tile(c[None], (n, 1, 1))

    def test_trilinear_at_corners(self):
        corners = self._unit_cube_corners(8)
        uvw = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        result = trilinear(uvw, corners)
        np.testing.assert_allclose(result, uvw, atol=1e-12)

    def test_trilinear_at_centre(self):
        corners = self._unit_cube_corners(1)
        uvw     = np.array([[0.5, 0.5, 0.5]])
        result  = trilinear(uvw, corners)
        np.testing.assert_allclose(result, [[0.5, 0.5, 0.5]], atol=1e-12)

    def test_inverse_trilinear_recovers_points(self):
        corners       = self._unit_cube_corners(5)
        rng           = np.random.default_rng(42)
        pts           = rng.uniform(0, 1, (5, 3))
        uvw           = inverse_trilinear(pts, corners)
        reconstructed = trilinear(uvw, corners)
        np.testing.assert_allclose(reconstructed, pts, atol=1e-9)

    def test_inverse_trilinear_non_uniform(self):
        """Verify Newton converges for a skewed hex cell."""
        corners = np.array(
            [
                [
                    [0, 0, 0],
                    [2, 0, 0],
                    [0.5, 3, 0],
                    [2.5, 3, 0],
                    [0, 0, 4],
                    [2, 0, 4],
                    [0.5, 3, 4],
                    [2.5, 3, 4],
                ]
            ],
            dtype=np.float64,
        )
        rng          = np.random.default_rng(7)
        uvw_expected = rng.uniform(0.05, 0.95, (10, 3))
        pts          = trilinear(uvw_expected, np.tile(corners, (10, 1, 1)))
        uvw          = inverse_trilinear(pts, np.tile(corners, (10, 1, 1)))
        np.testing.assert_allclose(uvw, uvw_expected, atol=1e-8)


class TestCellAssignment(unittest.TestCase):
    """Tests for assign_cells on uniform and non-uniform lattices."""

    @staticmethod
    def _uniform_lattice(d: int = 3) -> np.ndarray:
        axes = [np.linspace(0, 1, d + 1)] * 3
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def test_interior_points(self):
        lattice = self._uniform_lattice(3)
        pts     = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.1]])
        cells, uvw = assign_cells(pts, lattice)
        reconstructed = trilinear(uvw, get_cell_corners(lattice, cells))
        np.testing.assert_allclose(reconstructed, pts, atol=1e-8)

    def test_corner_points(self):
        lattice = self._uniform_lattice(2)
        pts     = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        cells, uvw = assign_cells(pts, lattice)
        reconstructed = trilinear(uvw, get_cell_corners(lattice, cells))
        np.testing.assert_allclose(reconstructed, pts, atol=1e-8)

    def test_outside_points_get_assigned(self):
        lattice = self._uniform_lattice(2)
        pts     = np.array([[-0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
        cells, uvw = assign_cells(pts, lattice)
        self.assertEqual(cells.shape, (2, 3))
        self.assertEqual(uvw.shape, (2, 3))


class TestFFDDataConstruction(unittest.TestCase):
    """Tests for FFDData __init__, from_mesh, and create_lattice."""

    def test_create_lattice_returns_meshdata(self):
        lattice_mesh = FFDData.create_lattice((4, 5, 6))
        self.assertEqual(lattice_mesh.points.shape, (4 * 5 * 6, 3))
        self.assertTrue(len(lattice_mesh.counts) > 0)
        self.assertTrue(np.all(lattice_mesh.counts == 4))  # all quads

    def test_create_lattice_with_bbox(self):
        lattice_mesh = FFDData.create_lattice(
            (3, 3, 3), bbox_min=[-1, -1, -1], bbox_max=[1, 1, 1]
        )
        self.assertAlmostEqual(lattice_mesh.points.min(), -1.0)
        self.assertAlmostEqual(lattice_mesh.points.max(), 1.0)

    def test_create_lattice_default_bounds(self):
        lattice_mesh = FFDData.create_lattice((4, 4, 4))
        self.assertAlmostEqual(lattice_mesh.points.min(), -0.5)
        self.assertAlmostEqual(lattice_mesh.points.max(), 0.5)

    def test_init_from_meshdata(self):
        lattice_mesh = FFDData.create_lattice((4, 4, 4))
        ffd          = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))
        np.testing.assert_array_equal(ffd.divisions, [4, 4, 4])

    def test_init_4d_array(self):
        lattice = FFDData.create_lattice((4, 4, 4))
        ffd     = FFDData.from_mesh(lattice, divisions=(4, 4, 4))
        np.testing.assert_array_equal(ffd.divisions, [4, 4, 4])

    def test_init_flat_array_with_divisions(self):
        lattice_mesh = FFDData.create_lattice((3, 4, 5))
        ffd          = FFDData(lattice_mesh.points, divisions=(3, 4, 5))
        self.assertEqual(ffd.lattice.shape, (3, 4, 5, 3))

    def test_init_flat_array_wrong_count_raises(self):
        pts = np.zeros((10, 3))
        with self.assertRaises(ValueError):
            FFDData(pts, divisions=(4, 4, 4))

    def test_init_flat_array_no_divisions_raises(self):
        pts = np.zeros((27, 3))
        with self.assertRaises(ValueError):
            FFDData(pts)

    def test_from_mesh_flat_array(self):
        lattice_mesh = FFDData.create_lattice((4, 4, 4))
        ffd          = FFDData.from_mesh(lattice_mesh.points, divisions=(4, 4, 4))
        self.assertEqual(ffd.lattice.shape, (4, 4, 4, 3))

    def test_from_mesh_with_points_attr(self):
        """Accept objects with a .points attribute (MeshData-like)."""
        lattice_mesh = FFDData.create_lattice((4, 4, 4))
        ffd          = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))
        self.assertEqual(ffd.lattice.shape, (4, 4, 4, 3))

    def test_from_mesh_wrong_count_raises(self):
        pts = np.zeros((10, 3))
        with self.assertRaises(ValueError):
            FFDData.from_mesh(pts, divisions=(4, 4, 4))


class TestFFDDataIdentity(unittest.TestCase):
    """Deforming with the rest lattice should return the original points."""

    def test_identity_trilinear(self):
        lattice_mesh = FFDData.create_lattice(
            (4, 4, 4), bbox_min=[-0.1, -0.1, -0.1], bbox_max=[1.1, 1.1, 1.1]
        )
        ffd = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))
        pts = np.random.default_rng(10).uniform(0, 1, (200, 3))
        ffd.bind(pts)
        ffd.update(ffd.lattice)
        np.testing.assert_allclose(ffd.points, pts, atol=1e-9)

    def test_identity_with_local_influence(self):
        lattice_mesh = FFDData.create_lattice(
            (4, 4, 4), bbox_min=[-0.1, -0.1, -0.1], bbox_max=[1.1, 1.1, 1.1]
        )
        ffd = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))
        ffd.local_influence = (4, 4, 4)
        pts = np.random.default_rng(20).uniform(0, 1, (200, 3))
        ffd.bind(pts)
        ffd.update(ffd.lattice)
        np.testing.assert_allclose(ffd.points, pts, atol=1e-9)


class TestFFDDataDeformation(unittest.TestCase):
    """Tests that deformation produces correct results."""

    def _make_ffd_and_points(self):
        lattice_mesh = FFDData.create_lattice(
            (2, 2, 2), bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]
        )
        ffd = FFDData.from_mesh(lattice_mesh, divisions=(2, 2, 2))
        pts = np.array(
            [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
            dtype=np.float64,
        )
        ffd.bind(pts)
        return ffd, pts, ffd.lattice.copy()

    def test_trilinear_z_shift(self):
        ffd, pts, lattice = self._make_ffd_and_points()
        deformed = lattice.copy()
        deformed[:, :, -1, 2] += 1.0  # shift top Z layer up by 1

        ffd.update(deformed)

        # Z displacement should scale linearly with original Z
        dz          = ffd.points[:, 2] - pts[:, 2]
        expected_dz = pts[:, 2] * 1.0  # proportional to Z
        np.testing.assert_allclose(dz, expected_dz, atol=1e-9)

        # X and Y should be unchanged
        np.testing.assert_allclose(ffd.points[:, :2], pts[:, :2], atol=1e-9)

    def test_uniform_translation(self):
        """Shifting all lattice CPs by the same offset shifts mesh identically."""
        ffd, pts, lattice = self._make_ffd_and_points()
        offset   = np.array([3.0, -2.0, 1.0])
        deformed = lattice.copy() + offset

        ffd.update(deformed)
        np.testing.assert_allclose(ffd.points, pts + offset, atol=1e-9)

    def test_update_without_bind_raises(self):
        lattice_mesh = FFDData.create_lattice((2, 2, 2))
        ffd          = FFDData.from_mesh(lattice_mesh, divisions=(2, 2, 2))
        with self.assertRaises(RuntimeError):
            ffd.update(ffd.lattice)


class TestFFDOutsideModes(unittest.TestCase):
    """Tests for extrapolate, freeze, and falloff outside modes."""

    def _make_ffd_with_outside_point(self):
        lattice_mesh = FFDData.create_lattice(
            (4, 4, 4), bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]
        )
        ffd = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))

        # points: one inside, one outside
        pts = np.array(
            [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
            dtype=np.float64,
        )
        return ffd, pts, ffd.lattice.copy()

    def test_extrapolate_deforms_outside(self):
        ffd, pts, lattice = self._make_ffd_with_outside_point()
        ffd.outside = "extrapolate"
        ffd.bind(pts)

        deformed = lattice.copy()
        deformed[:, :, :, 2] += 1.0
        ffd.update(deformed)

        # Both points should be shifted in Z
        np.testing.assert_allclose(ffd.points[:, 2], pts[:, 2] + 1.0, atol=1e-8)

    def test_freeze_does_not_deform_outside(self):
        ffd, pts, lattice = self._make_ffd_with_outside_point()
        ffd.outside = "freeze"
        ffd.bind(pts)

        deformed = lattice.copy()
        deformed[:, :, :, 2] += 1.0
        ffd.update(deformed)

        # Inside point should be shifted
        self.assertGreater(ffd.points[0, 2], pts[0, 2] + 0.5)
        # Outside point should stay put
        np.testing.assert_allclose(ffd.points[1], pts[1], atol=1e-9)

    def test_falloff_blends(self):
        ffd, pts, lattice = self._make_ffd_with_outside_point()
        ffd.outside        = "falloff"
        ffd.falloff_radius = 2.0
        ffd.bind(pts)

        deformed = lattice.copy()
        deformed[:, :, :, 2] += 1.0
        ffd.update(deformed)

        # Inside point: full deformation
        self.assertGreater(ffd.points[0, 2], pts[0, 2] + 0.5)
        # Outside point: partial deformation (not zero, not full)
        dz_outside = ffd.points[1, 2] - pts[1, 2]
        self.assertGreater(dz_outside, 0.0)
        self.assertLess(dz_outside, 1.0)

    def test_weights_extrapolate(self):
        ffd, pts, _ = self._make_ffd_with_outside_point()
        ffd.outside = "extrapolate"
        ffd.bind(pts)
        np.testing.assert_allclose(ffd.weights, [1.0, 1.0])

    def test_weights_freeze_inside_point(self):
        ffd, pts, _ = self._make_ffd_with_outside_point()
        ffd.outside = "freeze"
        ffd.bind(pts)
        self.assertAlmostEqual(ffd.weights[0], 1.0)
        self.assertAlmostEqual(ffd.weights[1], 0.0)


class TestFFDDegreeAndOutsideProperties(unittest.TestCase):
    """Test that outside/falloff_radius/local_influence properties work as setters."""

    def test_outside_setter_validation(self):
        ffd = FFDData.from_mesh(FFDData.create_lattice((2, 2, 2)), divisions=(2, 2, 2))
        for mode in ("extrapolate", "freeze", "falloff"):
            ffd.outside = mode
            self.assertEqual(ffd.outside, mode)
        with self.assertRaises(ValueError):
            ffd.outside = "invalid"

    def test_falloff_radius_setter(self):
        ffd = FFDData.from_mesh(FFDData.create_lattice((2, 2, 2)), divisions=(2, 2, 2))
        ffd.falloff_radius = 5.0
        self.assertEqual(ffd.falloff_radius, 5.0)

    def test_change_local_influence_without_rebind(self):
        """Changing local_influence after bind should affect next update."""
        lattice_mesh = FFDData.create_lattice(
            (4, 4, 4), bbox_min=[-0.1, -0.1, -0.1], bbox_max=[1.1, 1.1, 1.1]
        )
        ffd = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))
        pts = np.random.default_rng(99).uniform(0, 1, (50, 3))
        ffd.bind(pts)

        deformed = ffd.lattice.copy()
        deformed[1, 1, :, 0] += 0.5

        ffd.local_influence = (2, 2, 2)
        ffd.update(deformed)
        pts_local = ffd.points.copy()

        ffd.local_influence = (4, 4, 4)
        ffd.update(deformed)
        pts_smooth = ffd.points.copy()

        self.assertFalse(np.allclose(pts_local, pts_smooth))

    def test_change_outside_without_rebind(self):
        """Changing outside mode after bind should recompute weights."""
        ffd = FFDData.from_mesh(FFDData.create_lattice((4, 4, 4)), divisions=(4, 4, 4))
        pts = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        ffd.outside = "extrapolate"
        ffd.bind(pts)
        np.testing.assert_allclose(ffd.weights, [1.0, 1.0])

        ffd.outside = "freeze"
        self.assertAlmostEqual(ffd.weights[0], 1.0)
        self.assertAlmostEqual(ffd.weights[1], 0.0)


class TestFFDNonUniformLattice(unittest.TestCase):
    """Tests with non-uniform (warped) lattice geometry."""

    def test_tapered_lattice_identity(self):
        """Non-uniform lattice: identity deformation leaves points unchanged."""
        lattice = np.zeros((3, 3, 4, 3))
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    taper = 1.0 - 0.3 * (k / 3)
                    lattice[i, j, k] = [
                        (i - 1) * taper,
                        (j - 1) * taper,
                        k * 0.5,
                    ]

        ffd = FFDData(lattice)
        pts = np.array(
            [[0.0, 0.0, 0.5], [0.3, 0.3, 1.0], [-0.2, 0.1, 0.25]],
            dtype=np.float64,
        )
        ffd.bind(pts)
        ffd.update(lattice)  # identity
        np.testing.assert_allclose(ffd.points, pts, atol=1e-8)

    def test_tapered_lattice_deformation(self):
        """Non-uniform lattice deformation produces non-zero displacement."""
        lattice = np.zeros((3, 3, 4, 3))
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    taper = 1.0 - 0.3 * (k / 3)
                    lattice[i, j, k] = [(i - 1) * taper, (j - 1) * taper, k * 0.5]

        ffd = FFDData(lattice)
        ffd.local_influence = (3, 3, 4)
        pts = np.array([[0.0, 0.0, 0.5]], dtype=np.float64)
        ffd.bind(pts)

        deformed = lattice.copy()
        deformed[:, :, -1, 2] += 1.0
        ffd.update(deformed)

        self.assertGreater(ffd.points[0, 2], pts[0, 2])


class TestFFDValidProperty(unittest.TestCase):
    def test_valid_before_bind(self):
        ffd = FFDData.from_mesh(FFDData.create_lattice((2, 2, 2)), divisions=(2, 2, 2))
        self.assertFalse(ffd.valid)

    def test_valid_after_bind(self):
        ffd = FFDData.from_mesh(FFDData.create_lattice((2, 2, 2)), divisions=(2, 2, 2))
        ffd.bind(np.array([[0.5, 0.5, 0.5]]))
        self.assertTrue(ffd.valid)


class TestFFDToMesh(unittest.TestCase):
    """Tests for the to_mesh() round-trip."""

    def test_to_mesh_returns_meshdata(self):
        ffd  = FFDData.from_mesh(FFDData.create_lattice((4, 4, 4)), divisions=(4, 4, 4))
        mesh = ffd.to_mesh()
        self.assertTrue(hasattr(mesh, "points"))
        self.assertTrue(hasattr(mesh, "indices"))
        self.assertTrue(hasattr(mesh, "counts"))
        self.assertEqual(mesh.points.shape[0], 4 * 4 * 4)
        self.assertTrue(np.all(mesh.counts == 4))

    def test_to_mesh_points_match_lattice(self):
        lattice_mesh = FFDData.create_lattice(
            (4, 4, 4), bbox_min=[-1, -1, -1], bbox_max=[1, 1, 1]
        )
        ffd  = FFDData.from_mesh(lattice_mesh, divisions=(4, 4, 4))
        mesh = ffd.to_mesh()
        np.testing.assert_allclose(mesh.points, lattice_mesh.points, atol=1e-12)

    def test_to_mesh_with_deformed_lattice(self):
        ffd      = FFDData.from_mesh(FFDData.create_lattice((4, 4, 4)), divisions=(4, 4, 4))
        deformed = ffd.lattice.copy()
        deformed[:, :, -1, 2] += 5.0
        mesh = ffd.to_mesh(deformed)
        self.assertGreater(mesh.points[:, 2].max(), 4.0)

    def test_create_lattice_roundtrip(self):
        """create_lattice -> from_mesh -> to_mesh should round-trip points."""
        original     = FFDData.create_lattice((4, 5, 6))
        ffd          = FFDData.from_mesh(original, divisions=(4, 5, 6))
        roundtripped = ffd.to_mesh()
        np.testing.assert_allclose(roundtripped.points, original.points, atol=1e-12)