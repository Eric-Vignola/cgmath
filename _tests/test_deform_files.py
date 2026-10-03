"""Every deformer saves to pkl, npz and json and loads back as the same object.

The same type, every field with its dtype and shape, and -- the point of it --
the same deformation, without the mesh, skin or rig it was built from: those
are constructor arguments and are never saved. Each deformer is saved before
it deforms, so whatever it binds lazily is rebuilt from the file.
"""

import os
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath.geometry.deform import (
    DeltaMushData,
    FFDData,
    PatchRelaxData,
    SkinDeformData,
    WrapData,
)
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.skin_weights import SkinData


def _grid(n: int = 7, height: float = 0.3) -> MeshData:
    """Quad grid on XY with a sine bump in Z, so smoothing has work to do."""
    t            = np.linspace(0.0, 1.0, n)
    points       = np.array([[x, y, 0.0] for y in t for x in t])
    points[:, 2] = height * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices.extend([a, a + 1, a + n + 1, a + n])

    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
    )


class _RoundTrip(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name

        self.mesh   = _grid()
        rng         = np.random.default_rng(3)
        self.noisy  = self.mesh.points + rng.normal(scale=0.02, size=self.mesh.points.shape)

    def _loads(self, obj) -> dict:
        """``obj`` saved as every format and loaded back, and copied"""
        loads = {}
        for ext in ("pkl", "npz", "json"):
            path       = obj.save(os.path.join(self.folder, f"deformer.{ext}"))
            loads[ext] = type(obj).load(path)
        loads["copy"]  = obj.copy()
        loads["bytes"] = type(obj).from_bytes(obj.to_bytes())
        return loads

    def _assert_same_object(self, original, loaded):
        self.assertIs(type(loaded), type(original))
        for field in fields(original):
            if field.name in type(original).TRANSIENT_FIELDS:
                continue
            a = getattr(original, field.name)
            b = getattr(loaded, field.name)
            if isinstance(a, np.ndarray):
                self.assertIsInstance(b, np.ndarray, field.name)
                self.assertEqual(b.dtype, a.dtype, field.name)
                self.assertEqual(b.shape, a.shape, field.name)
                np.testing.assert_array_equal(b, a, err_msg=field.name)
            else:
                self.assertIs(type(b), type(a), field.name)
                self.assertEqual(b, a, field.name)
        self.assertEqual(loaded, original)
        self.assertEqual(original, loaded)

    def assert_round_trips(self, obj, deform):
        """``deform(loaded)`` is exactly ``deform(obj)``, for every way to load"""
        loads = self._loads(obj)
        for how, loaded in loads.items():
            with self.subTest(how=how):
                self._assert_same_object(obj, loaded)

        expected = deform(obj)
        for how, loaded in loads.items():
            with self.subTest(how=how):
                np.testing.assert_array_equal(deform(loaded), expected)


class TestDeltaMushFiles(_RoundTrip):
    def test_every_method_bound_or_not(self):
        for method in ("TBN", "PROCRUSTES", "DDM"):
            for bound in (False, True):
                with self.subTest(method=method, bound=bound):
                    mush = DeltaMushData(
                        self.mesh, smooth_iterations=6, smooth_step_size=0.4,
                        weight=0.7, method=method,
                    )
                    if bound:
                        mush.bind()
                    self.assert_round_trips(mush, lambda d: d.apply(self.noisy))

    def test_unpinned(self):
        mush = DeltaMushData(self.mesh, smooth_iterations=6, pin_borders=False)
        self.assertIsNone(mush.border_vertices)
        self.assert_round_trips(mush, lambda d: d.apply(self.noisy))

    def test_a_point_cloud(self):
        mush = DeltaMushData(
            self.mesh.points,
            neighbors         = self.mesh.get_edge_vertex_neighbors(),
            smooth_iterations = 4,
            pin_borders       = False,
        )
        self.assert_round_trips(mush, lambda d: d.apply(self.noisy))

    def test_the_settings_are_fields(self):
        mush = DeltaMushData(
            self.mesh, smooth_iterations=7, smooth_step_size=0.3, pin_borders=False,
            weight=0.25, method="ddm",
        )
        loaded = DeltaMushData.from_dict(mush.to_dict())
        self.assertEqual(
            (loaded.smooth_iterations, loaded.smooth_step_size, loaded.pin_borders,
             loaded.weight, loaded.method),
            (7, 0.3, False, 0.25, "DDM"),
        )

    def test_index_arrays_are_int32(self):
        mush = DeltaMushData(self.mesh)
        for name in ("neighbors", "border_vertices", "_first_nbrs"):
            self.assertEqual(getattr(mush, name).dtype, np.int32, name)


class TestPatchRelaxFiles(_RoundTrip):
    def test_bound_or_not(self):
        for bound in (False, True):
            with self.subTest(bound=bound):
                relaxer = PatchRelaxData(
                    self.mesh, iterations=12, alpha=0.8, surface_blend=0.3,
                    step_size=0.4, mask=np.linspace(0.0, 1.0, len(self.mesh.points)),
                )
                if bound:
                    relaxer.bind()
                self.assert_round_trips(relaxer, lambda d: d.apply(self.noisy))

    def test_unpinned_without_a_mask(self):
        relaxer = PatchRelaxData(self.mesh, iterations=5, pin_borders=False)
        self.assert_round_trips(relaxer, lambda d: d.apply(self.noisy))


class TestFFDFiles(_RoundTrip):
    def _ffd(self, outside: str) -> FFDData:
        lattice = FFDData.create_lattice(
            (3, 4, 3), bbox_min=[-0.1, -0.1, -0.5], bbox_max=[1.1, 1.1, 0.5]
        )
        ffd                 = FFDData.from_mesh(lattice, divisions=(3, 4, 3))
        ffd.local_influence = (3, 4, 2)
        ffd.outside         = outside
        ffd.falloff_radius  = 1.5
        return ffd

    def _deformed(self, ffd: FFDData) -> np.ndarray:
        lattice = ffd.lattice.copy()
        lattice[:, :, -1, 2] += 0.4
        lattice[1, 1, :, 0]  += 0.2
        return lattice

    def test_every_outside_mode(self):
        # one point outside the lattice, so freeze and falloff weigh it
        points = np.vstack([self.mesh.points, [[2.0, 0.5, 0.0]]])
        for outside in ("extrapolate", "freeze", "falloff"):
            with self.subTest(outside=outside):
                ffd = self._ffd(outside)
                ffd.bind(points)
                lattice = self._deformed(ffd)
                self.assert_round_trips(ffd, lambda d: d.update(lattice))

    def test_bound_to_a_mesh(self):
        # the mesh is not saved: a loaded FFD hands back points, the same points
        ffd = self._ffd("extrapolate")
        ffd.bind(self.mesh)
        lattice = self._deformed(ffd)

        def deform(d):
            d.update(lattice)
            return d.points

        self.assert_round_trips(ffd, deform)

    def test_unbound(self):
        ffd = self._ffd("falloff")
        self.assert_round_trips(ffd, lambda d: d.to_mesh().points)

    def test_local_influence_loads_as_an_int32_array(self):
        ffd = self._ffd("extrapolate")
        for how, loaded in self._loads(ffd).items():
            with self.subTest(how=how):
                self.assertEqual(loaded.local_influence.dtype, np.int32)
                np.testing.assert_array_equal(loaded.local_influence, [3, 4, 2])


class TestSkinDeformFiles(_RoundTrip):
    def _matrices(self) -> np.ndarray:
        matrices           = np.stack([np.eye(4), np.eye(4)])
        matrices[1, 3, :3] = [0.5, 1.0, 0.0]
        return matrices

    def _skin(self, count: int) -> SkinData:
        ramp = np.linspace(0.0, 1.0, count)
        return SkinData(weights=np.column_stack([ramp, 1.0 - ramp]), influences=["a", "b"])

    def test_every_method(self):
        for method in ("lbs", "dqs"):
            with self.subTest(method=method):
                deformer = SkinDeformData(
                    self.mesh,
                    self._skin(len(self.mesh.points)),
                    inverse_bind_matrices=np.stack([np.eye(4)] * 2),
                    method=method,
                    name="A:hero",
                )
                deformer.bind()
                self.assert_round_trips(deformer, lambda d: d.apply(self._matrices()))

    def test_meshless(self):
        deformer = SkinDeformData(
            None, self._skin(6), inverse_bind_matrices=np.stack([np.eye(4)] * 2)
        )
        deformer.bind()
        points = np.random.default_rng(1).normal(size=(6, 3))
        self.assert_round_trips(deformer, lambda d: d.apply(self._matrices(), points))


class TestWrapFiles(_RoundTrip):
    def setUp(self):
        super().setUp()
        self.cage  = _grid(4, height=0.4)
        rng        = np.random.default_rng(5)
        self.moved = self.cage.points + rng.normal(scale=0.05, size=self.cage.points.shape)

    def _wrap(self, kernel: str = "thin_plate_spline") -> WrapData:
        wrap = WrapData(kernel=kernel, name="cage_wrap")
        wrap.set_source(self.cage)
        wrap.set_target(self.moved)
        return wrap

    def test_bound(self):
        wrap = self._wrap()
        wrap.deform(self.mesh.points)
        self.assert_round_trips(wrap, lambda d: d.deform(self.mesh.points))

    def test_set_but_not_solved(self):
        self.assert_round_trips(self._wrap(), lambda d: d.deform(self.mesh.points))

    def test_geodesic_with_a_compact_kernel(self):
        wrap = self._wrap("wendland_c2")
        wrap.set_radius(0.9)
        wrap.set_geodesic_radius(1.0)
        wrap.deform(self.mesh.points)
        self.assert_round_trips(wrap, lambda d: d.deform(self.mesh.points))

    def test_dirty_after_a_kernel_change(self):
        wrap = self._wrap()
        wrap.deform(self.mesh.points)
        wrap.set_kernel("cubic")
        self.assert_round_trips(wrap, lambda d: d.deform(self.mesh.points))

    def test_unbound(self):
        def deform(d):
            d.set_source(self.cage)
            d.set_target(self.moved)
            return d.deform(self.mesh.points)

        self.assert_round_trips(WrapData(name="empty"), deform)


if __name__ == "__main__":
    unittest.main()
