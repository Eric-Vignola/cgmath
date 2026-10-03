import os
import pickle
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath.geometry.deform.wrap import WrapData


def _make_grid_mesh(n: int = 6, height: float = 0.0):
    """Quad grid on XY, optionally bumped in Z.

    The bump matters for anything used as a cage: the interpolation system is
    augmented with the polynomial terms ``[1, x, y, z]``, so a perfectly planar
    control set leaves that block rank deficient and the solve carries no
    information about the off-plane direction.
    """
    from cgmath.geometry.mesh import MeshData

    t    = np.linspace(0.0, 1.0, n)
    grid = np.array([[x, y, 0.0] for y in t for x in t])
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


class TestWrapDeform(unittest.TestCase):
    def setUp(self):
        self.cage        = _make_grid_mesh(4, height=0.4)
        self.cage_points = np.asarray(self.cage.points)
        self.mesh        = np.asarray(_make_grid_mesh(9, height=0.15).points)

    def _bound(self):
        wrap = WrapData()
        wrap.set_source(self.cage)
        wrap.set_target(self.cage_points.copy())
        return wrap

    def test_unmoved_control_points_are_the_identity(self):
        deformed = self._bound().deform(self.mesh)
        np.testing.assert_allclose(deformed, self.mesh, atol=1e-9)

    def test_affine_control_motion_is_reproduced_exactly(self):
        """A thin plate spline carries a linear polynomial term."""
        moved = self.cage_points.copy()
        moved[:, 2] += 0.35 * moved[:, 0]

        wrap = self._bound()
        wrap.set_target(moved)

        expected = self.mesh.copy()
        expected[:, 2] += 0.35 * expected[:, 0]
        np.testing.assert_allclose(wrap.deform(self.mesh), expected, atol=1e-9)

    def test_deform_without_a_source_raises(self):
        wrap = WrapData()
        wrap.set_target(self.cage_points.copy())
        with self.assertRaises(ValueError):
            wrap.deform(self.mesh)

    def test_deform_without_a_target_raises(self):
        wrap = WrapData()
        wrap.set_source(self.cage)
        with self.assertRaises(ValueError):
            wrap.deform(self.mesh)


class TestWrapCaching(unittest.TestCase):
    """The system matrix depends on the source, the kernel and the radii.

    Moving the target changes only the right-hand side, so re-solving it is a
    matmul rather than another pseudo-inverse. These tests pin both halves of
    that: the inverse survives a moved target, and every other setter still
    invalidates it.
    """

    def setUp(self):
        self.cage        = _make_grid_mesh(4, height=0.4)
        self.cage_points = np.asarray(self.cage.points)
        self.mesh        = np.asarray(_make_grid_mesh(9, height=0.15).points)

        self.wrap = WrapData()
        self.wrap.set_source(self.cage)
        self.wrap.set_target(self.cage_points.copy())
        self.wrap.deform(self.mesh)

    def _moved(self, seed):
        rng = np.random.default_rng(seed)
        return self.cage_points + rng.normal(0.0, 0.2, self.cage_points.shape)

    def _fresh(self, target, **kwargs):
        wrap = WrapData(**kwargs)
        wrap.set_source(self.cage)
        for name, value in (
            ("set_radius", kwargs.pop("radius", None)),
            ("set_geodesic_radius", kwargs.pop("geodesic_radius", None)),
        ):
            if value is not None:
                getattr(wrap, name)(value)
        wrap.set_target(target)
        return wrap.deform(self.mesh)

    def test_set_target_reuses_the_pseudo_inverse(self):
        before = self.wrap._system_pinv
        self.wrap.set_target(self._moved(1))
        self.wrap.deform(self.mesh)
        self.assertIs(self.wrap._system_pinv, before)

    def test_set_kernel_rebuilds_the_pseudo_inverse(self):
        before = self.wrap._system_pinv
        self.wrap.set_kernel("wendland_c2")
        self.wrap.deform(self.mesh)
        self.assertIsNot(self.wrap._system_pinv, before)

    def test_set_source_rebuilds_the_pseudo_inverse(self):
        before = self.wrap._system_pinv
        other  = _make_grid_mesh(5, height=0.3)
        self.wrap.set_source(other)
        self.wrap.set_target(np.asarray(other.points).copy())
        self.wrap.deform(self.mesh)
        self.assertIsNot(self.wrap._system_pinv, before)

    def test_set_radius_rebuilds_the_pseudo_inverse(self):
        before = self.wrap._system_pinv
        self.wrap.set_radius(2.0)
        self.wrap.deform(self.mesh)
        self.assertIsNot(self.wrap._system_pinv, before)

    def test_set_geodesic_radius_rebuilds_the_pseudo_inverse(self):
        before = self.wrap._system_pinv
        self.wrap.set_geodesic_radius(0.8)
        self.wrap.deform(self.mesh)
        self.assertIsNot(self.wrap._system_pinv, before)

    def test_a_reused_object_matches_a_fresh_one(self):
        for seed in range(3):
            target = self._moved(seed)
            with self.subTest(seed=seed):
                self.wrap.set_target(target)
                np.testing.assert_array_equal(
                    self.wrap.deform(self.mesh), self._fresh(target)
                )

    def test_a_reused_object_matches_after_a_kernel_change(self):
        target = self._moved(9)
        self.wrap.set_kernel("wendland_c2")
        self.wrap.set_radius(2.0)
        self.wrap.set_target(target)

        fresh = WrapData(kernel="wendland_c2")
        fresh.set_source(self.cage)
        fresh.set_radius(2.0)
        fresh.set_target(target)

        np.testing.assert_array_equal(
            self.wrap.deform(self.mesh), fresh.deform(self.mesh)
        )

    def test_deforming_twice_without_a_change_is_stable(self):
        first = self.wrap.deform(self.mesh)
        np.testing.assert_array_equal(self.wrap.deform(self.mesh), first)

    def test_returning_to_a_pose_returns_the_same_answer(self):
        a = self._moved(4)
        self.wrap.set_target(a)
        first = self.wrap.deform(self.mesh)

        self.wrap.set_target(self._moved(5))
        self.wrap.deform(self.mesh)

        self.wrap.set_target(a)
        np.testing.assert_array_equal(self.wrap.deform(self.mesh), first)

    def test_the_target_dirty_flag_is_not_a_dataclass_field(self):
        """It must stay out of ``fields()``, and so out of ``to_dict()``.

        ``_dirty`` is a declared field, so its name is part of the serialized
        schema; the companion flag is deliberately unannotated so that adding
        it did not change what a ``WrapData`` serializes to.
        """
        from dataclasses import fields

        names = {x.name for x in fields(WrapData)}
        self.assertIn("_dirty", names)
        self.assertNotIn("_target_dirty", names)


class TestWrapFiles(unittest.TestCase):
    """What a save keeps: the inputs and the pseudo-inverse, nothing derived."""

    def setUp(self):
        self.cage   = _make_grid_mesh(4, height=0.4)
        self.mesh   = np.asarray(_make_grid_mesh(9, height=0.15).points)
        rng         = np.random.default_rng(8)
        self.target = self.cage.points + rng.normal(0.0, 0.1, self.cage.points.shape)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name

    def _bound(self) -> WrapData:
        wrap = WrapData()
        wrap.set_source(self.cage)
        wrap.set_target(self.target)
        wrap.deform(self.mesh)
        return wrap

    def _loaded(self, wrap: WrapData) -> dict:
        loaded = {"copy": wrap.copy(), "pickle": pickle.loads(pickle.dumps(wrap))}
        for ext in ("pkl", "npz", "json"):
            path        = wrap.save(os.path.join(self.folder, f"wrap.{ext}"))
            loaded[ext] = WrapData.load(path)
        return loaded

    def test_an_unbound_wrap_has_every_field(self):
        # _distance_matrix was a field that __init__ never set, so to_dict,
        # save, copy and pickle all raised AttributeError on a fresh WrapData
        wrap = WrapData()
        for field in fields(WrapData):
            getattr(wrap, field.name)

    def test_an_unbound_wrap_saves_and_loads(self):
        wrap = WrapData(kernel="cubic", name="unbound")
        for how, loaded in self._loaded(wrap).items():
            with self.subTest(how=how):
                self.assertEqual(loaded, wrap)
                self.assertEqual((loaded.name, loaded.kernel_name), ("unbound", "cubic"))

                loaded.set_source(self.cage)
                loaded.set_target(self.target)
                fresh = WrapData(kernel="cubic")
                fresh.set_source(self.cage)
                fresh.set_target(self.target)
                np.testing.assert_array_equal(
                    loaded.deform(self.mesh), fresh.deform(self.mesh)
                )

    def test_the_derived_matrices_are_not_saved(self):
        data = self._bound().to_dict()
        for name in ("_system_matrix", "_distance_matrix", "_weights", "_target_matrix"):
            self.assertNotIn(name, data)
        for name in (
            "src_points",
            "dst_points",
            "conn_matrix",
            "conn_distances",
            "_system_pinv",
        ):
            self.assertIn(name, data)

    def test_a_loaded_wrap_reuses_the_saved_pseudo_inverse(self):
        wrap     = self._bound()
        expected = wrap.deform(self.mesh)
        for how, loaded in self._loaded(wrap).items():
            with self.subTest(how=how):
                self.assertFalse(loaded._dirty)
                pinv = loaded._system_pinv
                np.testing.assert_array_equal(loaded.deform(self.mesh), expected)
                self.assertIs(loaded._system_pinv, pinv)

    def test_index_and_point_arrays_take_their_declared_dtypes(self):
        wrap = self._bound()
        self.assertEqual(wrap.conn_matrix.dtype, np.int32)
        for name in ("src_points", "dst_points", "conn_distances", "_system_pinv"):
            self.assertEqual(getattr(wrap, name).dtype, np.float64, name)


if __name__ == "__main__":
    unittest.main()