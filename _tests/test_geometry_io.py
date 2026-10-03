"""
pkl, npz and json give back exactly the object that was saved: its class,
every field, each array's dtype and shape. Covers the mesh, UV, map, morph,
skin and SDF classes.
"""

import os
import pickle
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath.geometry._base import Data, DataList
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
from cgmath.geometry.morph_target import MorphData, MorphList
from cgmath.geometry.sdf import DMCField, SDFBox, SDFCylinder, SDFSphere
from cgmath.geometry.skin_weights import CompactSkinData, Patterns, SkinData, SkinList


def _box() -> MeshData:
    return MeshData(
        points=[
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
        ],
        indices = [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4],
        counts  = [4, 4, 4, 4, 4, 4],
        name    = "boxShape",
    )


def _skin() -> SkinData:
    weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
            [0.2, 0.3, 0.5],
        ]
    )
    return SkinData(weights=weights, influences=["root", "spine", "head"], name="skinCluster1")


class RoundTrip(unittest.TestCase):
    """the comparisons every round trip test uses"""

    def assertIdentical(self, a, b, path="obj"):
        """same class, every field, every array's dtype, shape and values"""
        self.assertIs(type(a), type(b), path)

        if isinstance(a, np.ndarray):
            self.assertEqual(a.dtype, b.dtype, path)
            self.assertEqual(a.shape, b.shape, path)
            self.assertTrue(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), path)
        elif isinstance(a, Data):
            for field in fields(a):
                if field.name not in type(a).TRANSIENT_FIELDS:
                    self.assertIdentical(
                        getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}"
                    )
        elif isinstance(a, DataList):
            self.assertEqual(len(a), len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assertIdentical(x, y, f"{path}[{i}]")
            for name in type(a).LIST_FIELDS:
                self.assertIdentical(getattr(a, name), getattr(b, name), f"{path}.{name}")
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys(), path)
            for key in a:
                self.assertIdentical(a[key], b[key], f"{path}[{key!r}]")
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assertIdentical(x, y, f"{path}[{i}]")
        else:
            self.assertEqual(a, b, path)

    def assertRoundTrips(self, obj):
        """pkl, npz, json, pickle and copy() all give ``obj`` back"""
        cls = type(obj)
        with tempfile.TemporaryDirectory() as temp_dir:
            for ext in ("pkl", "npz", "json"):
                with self.subTest(format=ext):
                    filename = os.path.join(temp_dir, f"data.{ext}")
                    obj.save(filename)
                    loaded = cls.load(filename)
                    self.assertIdentical(obj, loaded)
                    self.assertTrue(obj == loaded)

        with self.subTest(format="pickle"):
            self.assertIdentical(obj, pickle.loads(pickle.dumps(obj)))
        with self.subTest(format="copy"):
            self.assertIdentical(obj, obj.copy())


class TestMeshRoundTrip(RoundTrip):
    def test_mesh(self):
        self.assertRoundTrips(_box())

    def test_mesh_every_field(self):
        mesh              = _box()
        mesh.matrix       = np.arange(16, dtype=float).reshape(4, 4)
        mesh.hole_faces   = [1]
        mesh.hole_counts  = [3]
        mesh.hole_indices = [2, 3, 5]
        mesh.set_normals(hard_edge_angle=30)
        mesh.points[0, 0] = np.nan
        self.assertRoundTrips(mesh)

    def test_mesh_empty(self):
        mesh = MeshData(indices=[], counts=[], points=[])
        self.assertEqual(mesh.points.shape, (0, 3))
        self.assertRoundTrips(mesh)

    def test_uvs(self):
        box = _box()
        uv  = UVData(points=np.random.rand(8, 2), indices=box.indices, counts=box.counts)
        self.assertRoundTrips(uv)

        uv.resolution = (512, 256)
        uv.antialias  = True
        self.assertRoundTrips(uv)

    def test_lists(self):
        box       = _box()
        uv_a      = UVData(points=np.random.rand(8, 2), indices=box.indices, counts=box.counts)
        uv_b      = uv_a.copy()
        uv_b.name = "map2"

        self.assertRoundTrips(MeshList([box, box.copy()]))
        self.assertRoundTrips(UVList([uv_a, uv_b]))
        self.assertRoundTrips(MeshList())


class TestMapRoundTrip(RoundTrip):
    def test_map(self):
        self.assertRoundTrips(
            MapData(
                name          = "mask",
                indices       = [0, 2],
                values        = [0.5, 1.0],
                default_value = 0.25,
                categories    = ["a", "b"],
            )
        )
        self.assertRoundTrips(MapData(name="empty", indices=[], values=[]))

    def test_geom_subset(self):
        self.assertRoundTrips(GeomSubsetData(name="faces", indices=[1, 2, 3], category="mat"))
        self.assertRoundTrips(GeomSubsetData(name="cvs", indices=[[1, 2], [3, 4]]))


class TestMorphRoundTrip(RoundTrip):
    def test_morph(self):
        morph = MorphData(name="smile", offsets=np.random.rand(5, 3))
        self.assertRoundTrips(morph)
        self.assertRoundTrips(MorphData(name="sparse", offsets=[[1, 2, 3]], indices=[7]))
        self.assertRoundTrips(MorphData(name="empty", offsets=np.empty((0, 3))))
        self.assertRoundTrips(MorphList([morph, morph.copy()]))


class TestSkinRoundTrip(RoundTrip):
    def test_skin(self):
        skin = _skin()
        self.assertRoundTrips(skin)

        skin.patterns = Patterns.SHORT
        self.assertRoundTrips(skin)

    def test_skin_empty(self):
        # json keeps no shape for an empty array: the columns come back
        self.assertRoundTrips(SkinData(weights=np.zeros((0, 2)), influences=["a", "b"]))

    def test_compact_skin(self):
        self.assertRoundTrips(_skin().to_compact_skin_data())

    def test_skin_list(self):
        skin = _skin()
        self.assertRoundTrips(SkinList([skin, skin[:3]]))


class TestSDFRoundTrip(RoundTrip):
    def test_primitives(self):
        sphere = SDFSphere(radius=2.5, translate=[1, 2, 3], name="ball")
        box    = SDFBox(half_extents=[1, 2, 3], rotate=[10, 20, 30], name="crate")
        cyl    = SDFCylinder(radius=0.3, height=4, axis=2, scale=[1, 2, 1], name="pipe")

        # a primitive in a field round trips without it
        field = DMCField(resolution=8)
        field.add(sphere)
        field.subtract(box)
        field.intersect(cyl)

        for primitive in (sphere, box, cyl):
            with self.subTest(primitive=primitive.name):
                self.assertRoundTrips(primitive)

                # the shape settings are among the fields compared
                self.assertTrue(
                    np.allclose(primitive.copy().bounding_box(), primitive.bounding_box())
                )


if __name__ == "__main__":
    unittest.main()
