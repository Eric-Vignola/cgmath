import unittest

import numpy as np
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import MeshData

try:
    from pxr import Usd
except ImportError:
    Usd = None


def _triangle() -> MeshData:
    return MeshData(points=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], indices=[0, 1, 2], counts=[3])


class TestMapData(unittest.TestCase):
    def test_fields(self):
        data = MapData(name="mask", indices=[0, 2], values=[1, 0])
        self.assertEqual(data.indices.dtype, np.int32)
        self.assertEqual(data.values.dtype,  np.float64)
        self.assertEqual(data.default_value, 0.0)

    def test_to_dense_array_keeps_fractions(self):
        """the default value does not truncate the values to ints"""
        data  = MapData(name="mask", indices=[0, 2], values=[0.5, 1.0])
        dense = data.to_dense_array(3)

        self.assertEqual(dense.dtype, np.float64)
        self.assertTrue(np.array_equal(dense, [0.5, 0.0, 1.0]))

        # an int default given explicitly truncates nothing either
        data.default_value = 0
        self.assertTrue(np.array_equal(data.to_dense_array(3), [0.5, 0.0, 1.0]))

    def test_to_skin_data_keeps_fractions(self):
        data = MapData(name="mask", indices=[0, 2], values=[0.5, 1.0])
        skin = data.to_skin_data(_triangle())

        self.assertEqual(skin.weights.dtype, np.float64)
        self.assertTrue(np.array_equal(skin.weights[:, 0], [0.5, 0.0, 1.0]))

    @unittest.skipIf(Usd is None, "USD is not installed")
    def test_prim_round_trip(self):
        """to_prim writes what from_prim reads: default value, component type"""
        stage = Usd.Stage.CreateInMemory()
        for component_type in ("v", "f"):
            with self.subTest(component_type=component_type):
                data = MapData(
                    name           = f"mask_{component_type}",
                    indices        = [0, 2],
                    values         = [0.5, 1.0],
                    default_value  = 0.25,
                    component_type = component_type,
                )
                prim = stage.DefinePrim(f"/{data.name}")
                data.to_prim(prim)

                loaded = MapData.from_prim(prim)
                self.assertEqual(loaded.default_value, 0.25)
                self.assertEqual(loaded.component_type, component_type)
                self.assertTrue(loaded == data)


class TestGeomSubsetData(unittest.TestCase):
    def test_fields(self):
        data = GeomSubsetData(name="tag", indices=np.array([[1, 2], [3, 4]], dtype=np.int64))
        self.assertEqual(data.indices.dtype, np.int32)
        self.assertEqual(data.indices.shape, (2, 2))

    @unittest.skipIf(Usd is None, "USD is not installed")
    def test_prim_round_trip(self):
        """the category is the subset's family, not its element type"""
        stage = Usd.Stage.CreateInMemory()
        prim  = stage.DefinePrim("/mesh/faces")
        data = GeomSubsetData(
            name           = "faces",
            indices        = [0, 3],
            component_type = "f",
            category       = "materialBind",
        )
        data.to_prim(prim)

        loaded = GeomSubsetData.from_prim(prim)
        self.assertEqual(loaded.category, "materialBind")
        self.assertEqual(loaded.component_type, "f")
        self.assertTrue(np.array_equal(loaded.indices, [0, 3]))


if __name__ == "__main__":
    unittest.main()
