"""
Array types and file round trips of the curve and surface data:
BSplineData, BSplinePatchData and their sample and raycast results, the
saddle surface's sample and raycast results, and ProcrustesData.
"""

import os
import pickle
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath.constraints import ProcrustesData
from cgmath.geometry._saddle_surface import RaycastData, SampleData, sample
from cgmath.geometry.bspline import BSplineData
from cgmath.geometry.bspline import SampleData as CurveSampleData
from cgmath.geometry.bspline_patch import (
    BSplinePatchData,
    PatchRaycastData,
    PatchSampleData,
)
from cgmath.geometry.mesh import MeshData, UVData

MODES = ("pkl", "npz", "json")


def round_trips(obj) -> dict:
    """``obj`` back from each file format, bytes, pickle and copy()"""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in MODES:
            filename = os.path.join(tmp, f"data.{mode}")
            obj.save(filename)
            out[mode] = type(obj).load(filename)
    out["bytes"]  = type(obj).from_bytes(obj.to_bytes())
    out["pickle"] = pickle.loads(pickle.dumps(obj))
    out["copy"]   = obj.copy()
    return out


class _Same(unittest.TestCase):
    def assertSame(self, a, b, tag=""):
        """same type, and every field the same type, dtype, shape and value"""
        self.assertIs(type(a), type(b), tag)
        for field in fields(a):
            x, y = getattr(a, field.name), getattr(b, field.name)
            where = f"{tag} {type(a).__name__}.{field.name}"
            if isinstance(x, np.ndarray):
                self.assertIsInstance(y, np.ndarray, where)
                self.assertEqual(x.dtype, y.dtype, where)
                self.assertEqual(x.shape, y.shape, where)
                self.assertTrue(
                    np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), where
                )
            else:
                self.assertIs(type(x), type(y), where)
                self.assertEqual(x, y, where)

    def assertRoundTrips(self, obj, tag=""):
        for mode, loaded in round_trips(obj).items():
            self.assertSame(obj, loaded, f"{tag} {mode}")

    def assertArray(self, value, dtype, shape, tag=""):
        self.assertIsInstance(value, np.ndarray, tag)
        self.assertEqual(value.dtype, np.dtype(dtype), tag)
        self.assertEqual(value.shape, shape, tag)


def chain(n=6, dims=3):
    """a gently bending chain of n points"""
    t      = np.linspace(0.0, 1.0, n)
    points = np.column_stack([t * 10.0, np.sin(t * 3.0), np.cos(t * 2.0), t * t])
    return points[:, :dims]


def grid(n=6):
    """an n x n wavy control point grid"""
    u, v = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n), indexing="ij")
    return np.stack([u, v, np.sin(u * 0.5) * np.cos(v * 0.5)], axis=-1)


# --------------------------------- CURVES ----------------------------------- #


class TestBSplineTypes(_Same):
    def test_points_and_knots_convert(self):
        curve = BSplineData(points=chain().astype(np.float32), knots=None)
        self.assertArray(curve.points, np.float64, (6, 3))

        curve = BSplineData(points=[[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]])
        self.assertArray(curve.points, np.float64, (4, 3))
        self.assertEqual(curve.count, 4)

        curve.knots = [0, 0, 0, 1, 1, 1]
        self.assertArray(curve.knots, np.float64, (6,))

    def test_free_dimension(self):
        for dims in (2, 3, 4):
            curve = BSplineData(points=chain(dims=dims))
            self.assertArray(curve.points, np.float64, (6, dims))
            points, _ = curve.compute(0.5)
            self.assertEqual(points.shape, (dims,))

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            BSplineData(points=np.zeros(6))
        with self.assertRaises(ValueError):
            BSplineData(points=chain(), knots=np.zeros((2, 4)))

    def test_empty_points_take_the_default_shape(self):
        # json keeps no shape for an empty array: (0,) comes back as (0, 3)
        self.assertArray(BSplineData(points=[]).points, np.float64, (0, 3))
        self.assertArray(BSplineData(points=np.zeros((0, 0))).points, np.float64, (0, 3))
        self.assertArray(BSplineData(points=np.zeros((0, 2))).points, np.float64, (0, 2))

        curve        = BSplineData()
        curve.points = np.zeros(0)
        self.assertArray(curve.points, np.float64, (0, 3))

    def test_defaults_are_not_shared(self):
        a = BSplineData()
        b = BSplineData()
        self.assertIsNot(a.points, b.points)
        a.fit(chain())
        self.assertEqual(b.points.shape, (0, 3))

    def test_round_trips(self):
        fitted = BSplineData()
        fitted.fit(chain(8))

        curves = {
            "empty": BSplineData(),
            "open":  BSplineData(points=chain()),
            "periodic": BSplineData(
                points             = chain(),
                periodic           = True,
                uniform            = True,
                registered         = True,
                use_numba          = False,
                arc_length_samples = 50,
            ),
            "fitted":  fitted,
            "2d":      BSplineData(points=chain(dims=2), degree=2),
            "float32": BSplineData(points=chain().astype(np.float32)),
        }
        for tag, curve in curves.items():
            self.assertRoundTrips(curve, tag)

    def test_caches_are_not_saved(self):
        curve = BSplineData(points=chain(), uniform=True)
        before, _ = curve.compute([0.1, 0.5, 0.9])
        self.assertIsNotNone(curve._kv)

        for mode, loaded in round_trips(curve).items():
            self.assertIsNone(loaded._kv, mode)
            self.assertNotIn("_kv", loaded.to_dict(), mode)
            after, _ = loaded.compute([0.1, 0.5, 0.9])
            self.assertTrue(np.allclose(before, after), mode)

    def test_sample_types(self):
        curve  = BSplineData(points=chain())
        result = curve.sample(np.array([[1, 0, 1], [5, 1, 0]]))
        self.assertIsInstance(result, CurveSampleData)
        self.assertArray(result.points,    np.float64, (2, 3))
        self.assertArray(result.tangents,  np.float64, (2, 3))
        self.assertArray(result.distances, np.float64, (2,))
        self.assertArray(result.params,    np.float64, (2,))
        self.assertArray(result.basis,     np.float64, (2, 6))
        self.assertRoundTrips(result, "sample")

        loop = BSplineData(points=chain(dims=2), periodic=True)
        self.assertRoundTrips(loop.sample(chain(dims=2) * 0.9), "periodic 2d sample")


# -------------------------------- SURFACES ---------------------------------- #


class TestBSplinePatchTypes(_Same):
    def test_points_convert(self):
        patch = BSplinePatchData(
            points     = grid().astype(np.float32),
            degree_u   = 3,
            degree_v   = 3,
            periodic_u = False,
            periodic_v = False,
        )
        self.assertArray(patch.points, np.float64, (6, 6, 3))

        with self.assertRaises(ValueError):
            BSplinePatchData(points=grid()[0], degree_u=3, degree_v=3, periodic_u=False, periodic_v=False)

    def test_round_trips(self):
        patches = {
            "open": BSplinePatchData(
                points=grid(), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False
            ),
            "periodic": BSplinePatchData(
                points       = grid(),
                degree_u     = 2,
                degree_v     = 3,
                periodic_u   = True,
                periodic_v   = False,
                uniform_v    = True,
                registered_u = True,
                use_numba    = False,
            ),
            "2d": BSplinePatchData(
                points=grid()[..., :2], degree_u=3, degree_v=3, periodic_u=False, periodic_v=True
            ),
        }
        for tag, patch in patches.items():
            self.assertRoundTrips(patch, tag)

    def test_sample_types(self):
        patch = BSplinePatchData(
            points=grid(), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False
        )
        result = patch.sample(np.array([[0, 0, 2], [1, -2, 1], [3, 3, -1]]))
        self.assertIsInstance(result, PatchSampleData)
        self.assertArray(result.points,    np.float64, (3, 3))
        self.assertArray(result.tangents,  np.float64, (3, 2, 3))
        self.assertArray(result.distances, np.float64, (3,))
        self.assertArray(result.params,    np.float64, (3, 2))
        self.assertArray(result.basis,     np.float64, (3, 36))
        self.assertRoundTrips(result, "sample")

        # a patch result is shaped as such
        with self.assertRaises(ValueError):
            result.params = np.zeros(3)

    def test_raycast_types(self):
        patch = BSplinePatchData(
            points=grid(), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False
        )
        origins = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [50.0, 50.0, 5.0]])
        result  = patch.raycast(origins, np.array([0.0, 0.0, -1.0]))
        self.assertIsInstance(result, PatchRaycastData)
        self.assertArray(result.points,    np.float64, (3, 3))
        self.assertArray(result.tangents,  np.float64, (3, 2, 3))
        self.assertArray(result.distances, np.float64, (3,))
        self.assertArray(result.params,    np.float64, (3, 2))
        self.assertArray(result.basis,     np.float64, (3, 36))
        self.assertArray(result.occluded,  np.bool_,   (3,))
        self.assertArray(result.hit,       np.bool_,   (3,))

        # the miss is NaN, which json writes as a bare token
        self.assertEqual(result.hit.tolist(), [True, True, False])
        self.assertTrue(np.isnan(result.distances[2]))
        self.assertRoundTrips(result, "raycast")


# ----------------------------- SADDLE SURFACE ------------------------------- #


def triangles(dims=3):
    """two triangles, every face a triangle"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    return points[:, :dims], np.array([0, 1, 2, 1, 3, 2]), np.array([3, 3])


class TestSaddleTypes(_Same):
    def setUp(self):
        super().setUp()
        self.quad = MeshData(
            points  = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float64),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

    def check_sample(self, result, n, dims, tag=""):
        self.assertIsInstance(result, SampleData, tag)
        self.assertArray(result.projections, np.float64, (n, dims), tag)
        self.assertArray(result.distances,   np.float64, (n,),      tag)
        self.assertArray(result.weights,     np.float64, (n, 4),    tag)
        self.assertArray(result.indices,     np.int32,   (n,),      tag)
        self.assertArray(result.normals,     np.float64, (n, dims), tag)
        self.assertArray(result.occluded,    np.bool_,   (n,),      tag)
        self.assertArray(result.uvs,         np.float64, (n, 2),    tag)
        self.assertArray(result.geometry,    np.int32,   (n, 4),    tag)

    def test_sample_types(self):
        queries = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0], [0.5, 0.2, -1.0]])
        for dtype in (np.float64, np.float32):
            self.check_sample(self.quad.sample(queries.astype(dtype)), 3, 3, str(dtype))

    def test_integer_queries_keep_their_weights(self):
        # bug: weights were stored in the query dtype, ints truncated them to 0
        floats = self.quad.sample(np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]))
        ints   = self.quad.sample(np.array([[1, 1, 1], [0, 1, 0]]))
        self.check_sample(ints, 2, 3)
        self.assertTrue(np.allclose(ints.weights, [[0.25] * 4, [0.5, 0, 0, 0.5]]))
        self.assertTrue(np.allclose(ints.weights, floats.weights))
        self.assertTrue(np.allclose(ints.distances, floats.distances))
        self.assertTrue(np.allclose(ints.uvs, floats.uvs))

    def test_triangles_matched_exactly(self):
        # bug: an all-triangle mesh whose queries all sit on its vertices kept
        # its 3-wide geometry, and the normal blend crashed on it
        points, indices, counts = triangles()
        mesh   = MeshData(points=points, indices=indices, counts=counts)
        result = mesh.sample(mesh.points)
        self.check_sample(result, 4, 3)
        self.assertTrue(np.allclose(result.projections, points))
        self.assertTrue(np.all(result.geometry[:, 3] == -1))
        self.assertTrue(np.allclose(result(points), points))
        self.assertRoundTrips(result, "triangles")

    def test_uv_triangles_matched_exactly(self):
        # a 2-D (UV) all-triangle map sampled at its own points: the geometry
        # is padded to 4 like the weights
        points, indices, counts = triangles(dims=2)
        uv     = UVData(points=points, indices=indices, counts=counts)
        result = uv.sample(uv.points)
        self.check_sample(result, 4, 2)
        self.assertTrue(np.allclose(result(points), points))

    def test_sample_function_pads_triangles(self):
        points, _, _ = triangles()
        geometry = np.array([[0, 1, 2], [1, 3, 2]])
        normals  = np.tile([0.0, 0.0, 1.0], (4, 1))
        result   = sample(points, points, geometry, normals=normals)
        self.check_sample(result, 4, 3)
        self.assertTrue(np.allclose(result.normals, normals))

        with self.assertRaises(ValueError):
            sample(points, points, np.array([[0, 1, 2, 3, 0]]))

    def test_remap_pads_triangles(self):
        points, indices, counts = triangles()
        mesh   = MeshData(points=points, indices=indices, counts=counts)
        uv     = UVData(points=points[:, :2].copy(), indices=indices, counts=counts)
        result = uv.sample(uv).remap(mesh, mesh, uv)
        self.check_sample(result, 4, 2)
        self.assertTrue(np.all(result.geometry[:, 3] == -1))

    def test_sample_round_trips(self):
        queries = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0], [0.5, 0.2, -1.0]])
        self.assertRoundTrips(self.quad.sample(queries), "sample")

    def test_raycast_types(self):
        origins = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, -1.0], [9.0, 9.0, 1.0]])
        dirs    = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        result  = self.quad.raycast(origins, dirs)
        self.assertIsInstance(result, RaycastData)
        self.assertArray(result.projections, np.float64, (3, 3))
        self.assertArray(result.distances,   np.float64, (3,))
        self.assertArray(result.weights,     np.float64, (3, 4))
        self.assertArray(result.indices,     np.int32,   (3,))
        self.assertArray(result.normals,     np.float64, (3, 3))
        self.assertArray(result.occluded,    np.bool_,   (3,))
        self.assertArray(result.uvs,         np.float64, (3, 2))
        self.assertArray(result.geometry,    np.int32,   (3, 4))
        self.assertArray(result.hit,         np.bool_,   (3,))

        # the miss: NaN distance, -1 face, which every format keeps
        self.assertEqual(result.hit.tolist(), [True, True, False])
        self.assertEqual(int(result.indices[2]), -1)
        self.assertTrue(np.isnan(result.distances[2]))
        self.assertRoundTrips(result, "raycast")

        points, indices, counts = triangles()
        mesh   = MeshData(points=points, indices=indices, counts=counts)
        result = mesh.raycast(origins.astype(np.float32), dirs.astype(np.float32))
        self.assertArray(result.projections, np.float64, (3, 3))
        self.assertArray(result.geometry, np.int32, (3, 4))


# -------------------------------- PROCRUSTES -------------------------------- #


class TestProcrustesTypes(_Same):
    def setUp(self):
        super().setUp()
        self.points = np.array(
            [
                [-0.5, -0.5, 0.5],
                [ 0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [ 0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [ 0.5, 0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [ 0.5, -0.5, -0.5],
            ]
        )
        self.transform        = np.eye(4)
        self.transform[3, :3] = [0.0, 0.0, 1.0]

    def constraint(self):
        constraint = ProcrustesData(self.points)
        constraint.attach(self.transform, [0, 1, 2, 3])
        constraint.attach(self.transform.tolist())
        return constraint

    def test_types(self):
        constraint = ProcrustesData(self.points.astype(np.float32))
        self.assertArray(constraint.points, np.float64, (8, 3))
        self.assertIsNone(constraint.transforms)
        self.assertIsNone(constraint.clusters)

        constraint.attach(self.transform, [3, 1, 1, 0, -1])
        self.assertArray(constraint.transforms, np.float64, (1, 4, 4))
        self.assertArray(constraint.clusters, np.int32, (1, 3))
        self.assertEqual(constraint.clusters.tolist(), [[0, 1, 3]])

        # a list is a matrix too
        constraint.attach(self.transform.tolist())
        self.assertArray(constraint.transforms, np.float64, (2, 4, 4))
        self.assertArray(constraint.clusters, np.int32, (2, 8))
        self.assertEqual(constraint.clusters[0].tolist(), [0, 1, 3, -1, -1, -1, -1, -1])

        with self.assertRaises(ValueError):
            constraint.attach(np.eye(3))

    def test_scale_offset_is_saved(self):
        constraint = self.constraint()
        self.assertTrue(constraint.scale_offset)

        constraint.scale_offset = False
        for mode, loaded in round_trips(constraint).items():
            self.assertFalse(loaded.scale_offset, mode)
            self.assertSame(constraint, loaded, mode)

    def test_scale_offset_before_attach(self):
        # nothing to compute yet: the flag is set, nothing raises
        constraint              = ProcrustesData(self.points)
        constraint.scale_offset = False
        self.assertFalse(constraint.scale_offset)

        constraint.attach(self.transform)
        constraint.update(self.points * 2.0)
        self.assertTrue(np.allclose(constraint.scale, 1.0 * 2.0))
        self.assertTrue(np.allclose(constraint.matrix[0, :3, :3], np.eye(3)))

    def test_round_trips(self):
        self.assertRoundTrips(ProcrustesData(self.points), "empty")

        constraint = self.constraint()
        moved      = self.points.copy()
        moved[:2, 2] += 1.0
        constraint.update(moved)

        for mode, loaded in round_trips(constraint).items():
            self.assertSame(constraint, loaded, mode)

            # the per-frame target and results are not saved
            self.assertIsNone(loaded.matrix, mode)
            loaded.update(moved)
            self.assertTrue(np.allclose(loaded.matrix, constraint.matrix), mode)


if __name__ == "__main__":
    unittest.main()
