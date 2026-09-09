import unittest
import warnings

import numpy as np
from cgmath.geometry.deform import WrapData
from cgmath.geometry.deform.wrap import _COMPACT_KERNELS, cdist_euclidean
from cgmath.geometry.mesh import MeshData
from cgmath.rbf._kernels import get_kernel, KERNEL_REGISTRY


EPSILON = 1e-6


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=atol)


class TestCdistEuclidean(unittest.TestCase):
    """Test the Numba-accelerated pairwise Euclidean distance function."""

    def test_identity(self):
        """Diagonal of self-distance matrix should be zero."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        D   = cdist_euclidean(pts, pts)
        self.assertTrue(allclose(np.diag(D), 0.0))

    def test_known_distances(self):
        """Check distances against hand-computed values."""
        X = np.array([[0.0, 0.0, 0.0]])
        Y = np.array([[3.0, 4.0, 0.0]])
        D = cdist_euclidean(X, Y)
        self.assertTrue(allclose(D[0, 0], 5.0))

    def test_shape(self):
        """Output shape should be (m, n)."""
        X = np.random.default_rng(42).random((7, 3))
        Y = np.random.default_rng(99).random((11, 3))
        D = cdist_euclidean(X, Y)
        self.assertEqual(D.shape, (7, 11))

    def test_symmetry(self):
        """cdist(X, X) should be symmetric."""
        pts = np.random.default_rng(0).random((20, 3))
        D   = cdist_euclidean(pts, pts)
        self.assertTrue(allclose(D, D.T))


class TestKernelRegistry(unittest.TestCase):
    """Test kernel lookup and basic evaluation."""

    def test_all_registered(self):
        """Every kernel in the registry should be retrievable by name."""
        for name in KERNEL_REGISTRY:
            fn = get_kernel(name)
            self.assertTrue(callable(fn))

    def test_case_insensitive(self):
        """get_kernel should be case-insensitive."""
        fn1 = get_kernel("Gaussian")
        fn2 = get_kernel("GAUSSIAN")
        fn3 = get_kernel("gaussian")
        D   = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertTrue(allclose(fn1(D), fn2(D)))
        self.assertTrue(allclose(fn2(D), fn3(D)))

    def test_unknown_kernel_raises(self):
        """Unknown kernel name should raise ValueError."""
        with self.assertRaises(ValueError):
            get_kernel("nonexistent_kernel")

    def test_kernel_output_shape(self):
        """Every kernel should preserve the input shape."""
        D = np.random.default_rng(7).random((5, 5))
        for name in KERNEL_REGISTRY:
            fn = get_kernel(name)
            if name in _COMPACT_KERNELS:
                result = fn(D, r=10.0)
            else:
                result = fn(D)
            self.assertEqual(result.shape, D.shape, msg=f"Shape mismatch for '{name}'")

    def test_kernel_zero_distance(self):
        """Kernels evaluated at zero distance should return finite values."""
        D = np.zeros((2, 2))
        for name in KERNEL_REGISTRY:
            fn = get_kernel(name)
            if name in _COMPACT_KERNELS:
                result = fn(D, r=1.0)
            else:
                result = fn(D)
            self.assertTrue(
                np.all(np.isfinite(result)),
                msg=f"Non-finite at zero distance for '{name}'",
            )

    def test_compact_kernels_vanish_beyond_radius(self):
        """Compact kernels should return zero for distances > r."""
        D = np.array([[2.0]])
        for name in _COMPACT_KERNELS:
            fn     = get_kernel(name)
            result = fn(D, r=1.0)
            self.assertTrue(
                allclose(result, 0.0),
                msg=f"'{name}' did not vanish beyond support radius",
            )


class TestWrapDataBasic(unittest.TestCase):
    """Test WrapData with raw ndarray inputs (no mesh topology)."""

    def setUp(self):
        super().setUp()
        self.src = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )

    def test_identity_deformation(self):
        """When target == source, deformed points should match source."""
        wrap = WrapData()
        wrap.set_source(self.src)
        wrap.set_target(self.src)
        result = wrap.deform(self.src)
        self.assertTrue(allclose(result, self.src, atol=1e-5))

    def test_translation(self):
        """Uniform translation of targets should translate deformed output."""
        offset = np.array([5.0, 3.0, -2.0])
        wrap   = WrapData()
        wrap.set_source(self.src)
        wrap.set_target(self.src + offset)
        result = wrap.deform(self.src)
        self.assertTrue(allclose(result, self.src + offset, atol=1e-5))

    def test_set_kernel(self):
        """set_kernel should change the active kernel and rebuild the system."""
        wrap = WrapData()
        wrap.set_source(self.src)
        wrap.set_kernel("gaussian")
        self.assertEqual(wrap.kernel_name, "gaussian")

    def test_set_kernel_rebuilds(self):
        """Changing kernel after source is set should still produce valid output."""
        wrap = WrapData()
        wrap.set_source(self.src)
        wrap.set_kernel("cubic")
        wrap.set_target(self.src)
        result = wrap.deform(self.src)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_kernel_via_constructor(self):
        """Kernel specified in constructor should be used."""
        wrap = WrapData(kernel="cubic")
        self.assertEqual(wrap.kernel_name, "cubic")

    def test_deform_output_shape(self):
        """Deformed output should match input shape."""
        wrap = WrapData()
        wrap.set_source(self.src)
        wrap.set_target(self.src)
        query  = np.random.default_rng(0).random((10, 3))
        result = wrap.deform(query)
        self.assertEqual(result.shape, (10, 3))


class TestWrapDataWithMesh(unittest.TestCase):
    """Test WrapData with MeshData inputs (topology-aware features)."""

    def _make_strip(self):
        """Create a 1x4 quad strip lying flat on the XZ plane.

        Vertices:
            0--1--2--3--4
            |  |  |  |  |
            5--6--7--8--9

        Four quads, each 1x1 unit. Total length = 4 along X.
        """
        points = []
        for z in [0.5, -0.5]:
            for x in range(5):
                points.append([float(x), 0.0, z])
        points = np.array(points, dtype=np.float64)

        indices = np.array(
            [
                0,
                1,
                6,
                5,
                1,
                2,
                7,
                6,
                2,
                3,
                8,
                7,
                3,
                4,
                9,
                8,
            ],
            dtype=np.int32,
        )
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        return MeshData(points=points, indices=indices, counts=counts)

    def test_geodesic_radius_property(self):
        """geodesic_radius getter/setter should work."""
        wrap = WrapData()
        self.assertIsNone(wrap.geodesic_radius)
        wrap.set_geodesic_radius(2.5)
        self.assertEqual(wrap.geodesic_radius, 2.5)

    def test_radius_property(self):
        """radius getter/setter should work."""
        wrap = WrapData()
        self.assertIsNone(wrap.radius)
        wrap.set_radius(1.5)
        self.assertEqual(wrap.radius, 1.5)

    def test_geodesic_radius_masks_distant_pairs(self):
        """With geodesic_radius set, far-apart vertices should have inf distance in _cdist."""
        mesh = self._make_strip()
        wrap = WrapData()
        wrap.set_geodesic_radius(1.5)
        wrap.set_source(mesh)
        wrap.set_target(mesh.points)
        wrap._rebuild()

        # Vertices 0 and 4 are 4 edges apart
        self.assertTrue(np.isinf(wrap._distance_matrix[0, 4]))
        self.assertTrue(np.isinf(wrap._distance_matrix[4, 0]))

        # Vertices 0 and 1 are 1 edge apart, should remain finite
        self.assertTrue(np.isfinite(wrap._distance_matrix[0, 1]))

    def test_geodesic_radius_identity(self):
        """Geodesic-radius RBF with compact kernel should still reproduce identity."""
        mesh = self._make_strip()
        wrap = WrapData(kernel="wendland_c2")
        wrap.set_geodesic_radius(3.0)
        wrap.set_source(mesh)
        wrap.set_target(mesh.points)
        result = wrap.deform(mesh.points)
        self.assertTrue(allclose(result, mesh.points, atol=1e-4))

    def test_no_geodesic_radius_full_connectivity(self):
        """Without geodesic_radius, _cdist should have no inf entries (full connectivity)."""
        mesh = self._make_strip()
        wrap = WrapData()
        wrap.set_source(mesh)
        wrap.set_target(mesh.points)
        wrap._rebuild()
        self.assertTrue(np.all(np.isfinite(wrap._distance_matrix)))

    def test_ndarray_source_ignores_geodesic_radius(self):
        """geodesic_radius should have no effect when source is a plain ndarray."""
        pts  = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        wrap = WrapData()
        wrap.set_geodesic_radius(0.5)
        wrap.set_source(pts)
        wrap.set_target(pts)
        wrap._rebuild()
        self.assertTrue(np.all(np.isfinite(wrap._distance_matrix)))


class TestWrapDataRadiusSeparation(unittest.TestCase):
    """Test independent control of geodesic_radius and kernel radius."""

    def test_effective_radius_falls_back(self):
        """If radius is not set, _effective_radius should return geodesic_radius."""
        wrap = WrapData()
        wrap.set_geodesic_radius(5.0)
        self.assertEqual(wrap._effective_radius(), 5.0)

    def test_effective_radius_prefers_explicit(self):
        """If radius is explicitly set, _effective_radius should return it."""
        wrap = WrapData()
        wrap.set_geodesic_radius(5.0)
        wrap.set_radius(2.0)
        self.assertEqual(wrap._effective_radius(), 2.0)

    def test_effective_radius_none(self):
        """If neither is set, _effective_radius should return None."""
        wrap = WrapData()
        self.assertIsNone(wrap._effective_radius())

    def test_radius_rebuild_on_set(self):
        """Changing radius should trigger rebuild and produce valid output."""
        src    = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        offset = np.array([0.1, 0.2, 0.3])
        wrap   = WrapData(kernel="wendland_c2")
        wrap.set_radius(5.0)
        wrap.set_source(src)
        wrap.set_target(src + offset)
        result_before = wrap.deform(src).copy()
        self.assertTrue(np.all(np.isfinite(result_before)))

        wrap.set_radius(0.1)
        self.assertTrue(wrap._dirty)
        result_after = wrap.deform(src)
        self.assertTrue(np.all(np.isfinite(result_after)))


class TestWrapDataWarnings(unittest.TestCase):
    """Test configuration warnings emitted by _warn_configuration."""

    def setUp(self):
        super().setUp()
        self.src = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
        )

    def test_warn_noncompact_with_geodesic(self):
        """Using geodesic_radius with a non-compact kernel should warn."""
        wrap = WrapData(kernel="thin_plate_spline")
        wrap.set_geodesic_radius(2.0)
        wrap.set_source(self.src)
        wrap.set_target(self.src)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrap.deform(self.src)
            self.assertTrue(any("ill-conditioning" in str(x.message) for x in w))

    def test_warn_radius_exceeds_geodesic(self):
        """radius > geodesic_radius should emit a warning."""
        wrap = WrapData(kernel="wendland_c2")
        wrap.set_geodesic_radius(1.0)
        wrap.set_radius(5.0)
        wrap.set_source(self.src)
        wrap.set_target(self.src)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrap.deform(self.src)
            self.assertTrue(any("radius" in str(x.message).lower() for x in w))

    def test_no_warning_compact_with_geodesic(self):
        """Using geodesic_radius with a compact kernel should NOT warn about ill-conditioning."""
        wrap = WrapData(kernel="wendland_c2")
        wrap.set_geodesic_radius(2.0)
        wrap.set_source(self.src)
        wrap.set_target(self.src)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrap.deform(self.src)
            ill_warnings = [x for x in w if "ill-conditioning" in str(x.message)]
            self.assertEqual(len(ill_warnings), 0)


class TestWrapDataMultipleKernels(unittest.TestCase):
    """Verify deformation works across several representative kernels."""

    def setUp(self):
        super().setUp()
        self.src = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.offset = np.array([1.0, 2.0, 3.0])

    def test_translation_per_kernel(self):
        """Identity-like translation should hold for a representative set of kernels."""
        kernels = [
            "thin_plate_spline",
            "gaussian",
            "cubic",
            "linear",
            "multiquadric",
            "inverse_multiquadric",
            "wendland_c2",
        ]
        for name in kernels:
            wrap = WrapData(kernel=name)
            if name in _COMPACT_KERNELS:
                wrap.set_radius(10.0)
            wrap.set_source(self.src)
            wrap.set_target(self.src + self.offset)
            result = wrap.deform(self.src)
            self.assertTrue(
                allclose(result, self.src + self.offset, atol=1e-3),
                msg=f"Translation failed for kernel '{name}'",
            )


class TestWrapDataOrderIndependence(unittest.TestCase):
    """Test that configuration order does not affect results."""

    def _make_strip(self):
        """Create a 1x4 quad strip lying flat on the XZ plane."""
        points = []
        for z in [0.5, -0.5]:
            for x in range(5):
                points.append([float(x), 0.0, z])
        points = np.array(points, dtype=np.float64)

        indices = np.array(
            [0, 1, 6, 5, 1, 2, 7, 6, 2, 3, 8, 7, 3, 4, 9, 8],
            dtype=np.int32,
        )
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        return MeshData(points=points, indices=indices, counts=counts)

    def test_order_source_target_geodesic_kernel(self):
        """Both orderings should produce identical deform output."""
        mesh          = self._make_strip()
        offset        = np.array([0.1, 0.2, 0.3])
        target_points = mesh.points + offset

        wrap1         = WrapData()
        wrap1.set_geodesic_radius(3.0)
        wrap1.set_kernel("wendland_c2")
        wrap1.set_source(mesh)
        wrap1.set_target(target_points)
        result1 = wrap1.deform(mesh.points)

        wrap2   = WrapData()
        wrap2.set_source(mesh)
        wrap2.set_target(target_points)
        wrap2.set_kernel("wendland_c2")
        wrap2.set_geodesic_radius(3.0)
        result2 = wrap2.deform(mesh.points)

        self.assertTrue(allclose(result1, result2, atol=1e-6))

    def test_toggle_geodesic_radius(self):
        """Setting geodesic_radius should mark dirty and recompute on next deform."""
        mesh          = self._make_strip()
        offset        = np.array([0.1, 0.2, 0.3])
        target_points = mesh.points + offset

        wrap          = WrapData(kernel="wendland_c2")
        wrap.set_radius(1.5)
        wrap.set_source(mesh)
        wrap.set_target(target_points)
        result_no_geo = wrap.deform(mesh.points).copy()
        self.assertTrue(np.all(np.isfinite(result_no_geo)))
        self.assertFalse(wrap._dirty)

        wrap.set_geodesic_radius(1.5)
        self.assertTrue(wrap._dirty)
        result_with_geo = wrap.deform(mesh.points).copy()
        self.assertTrue(np.all(np.isfinite(result_with_geo)))
        self.assertFalse(wrap._dirty)

        wrap.set_geodesic_radius(None)
        self.assertTrue(wrap._dirty)
        result_none_again = wrap.deform(mesh.points)
        self.assertTrue(np.all(np.isfinite(result_none_again)))
        self.assertTrue(allclose(result_no_geo, result_none_again, atol=1e-6))

    def test_change_kernel_after_deform(self):
        """Changing kernel between deform calls should work correctly."""
        src = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        wrap = WrapData(kernel="thin_plate_spline")
        wrap.set_source(src)
        wrap.set_target(src)
        result_tps = wrap.deform(src).copy()

        wrap.set_kernel("gaussian")
        result_gaussian = wrap.deform(src)

        self.assertTrue(allclose(result_tps, src, atol=1e-4))
        self.assertTrue(allclose(result_gaussian, src, atol=1e-4))


class TestWrapDataSaveLoad(unittest.TestCase):
    """Test WrapData save/load round-trip preserves deformation output."""

    def setUp(self):
        super().setUp()
        import tempfile

        fd, self.temp_path = tempfile.mkstemp(suffix=".npz")
        import os

        os.close(fd)

    def tearDown(self):
        import os

        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
        super().tearDown()

    def _make_strip(self):
        """Create a 1x4 quad strip lying flat on the XZ plane."""
        points = []
        for z in [0.5, -0.5]:
            for x in range(5):
                points.append([float(x), 0.0, z])
        points = np.array(points, dtype=np.float64)

        indices = np.array(
            [0, 1, 6, 5, 1, 2, 7, 6, 2, 3, 8, 7, 3, 4, 9, 8],
            dtype=np.int32,
        )
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        return MeshData(points=points, indices=indices, counts=counts)

    def test_save_load_preserves_deformation(self):
        """WrapData saved and loaded should produce identical deformation."""
        mesh          = self._make_strip()
        offset        = np.array([0.1, 0.2, 0.3])
        target_points = mesh.points + offset

        wrap          = WrapData(kernel="wendland_c2")
        wrap.set_geodesic_radius(3.0)
        wrap.set_source(mesh)
        wrap.set_target(target_points)

        before = wrap.deform(mesh.points).copy()

        wrap.save(self.temp_path)

        loaded = WrapData.load(self.temp_path)

        self.assertEqual(loaded.kernel_name, "wendland_c2")
        self.assertEqual(loaded.geodesic_radius, 3.0)

        after = loaded.deform(mesh.points)

        self.assertTrue(
            allclose(before, after),
            msg="Deformation output differs after save/load round-trip",
        )