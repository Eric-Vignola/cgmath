"""Meshes and uv sets read from USD prims."""

import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry.mesh import MeshList, UVList

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD (pxr) is not installed")
class TestMeshFromPrim(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path  = os.path.join(tmp.name, "meshes.usda")
        self.stage = Usd.Stage.CreateNew(self.path)

    def quad(self, path: str):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreateFaceVertexCountsAttr().Set([4])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        return mesh

    def corners(self, width: float):
        return Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(width, 0, 0), Gf.Vec3f(width, 1, 0), Gf.Vec3f(0, 1, 0)])

    def test_a_texcoord3f_uv_set_keeps_u_and_v(self):
        mesh = self.quad("/uvw")
        mesh.CreatePointsAttr().Set(self.corners(1))
        st = mesh.GetPrim().CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord3fArray)
        st.Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 9), Gf.Vec3f(1, 0, 9), Gf.Vec3f(1, 1, 9), Gf.Vec3f(0, 1, 9)]))
        mesh.GetPrim().CreateAttribute("primvars:st:indices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
        self.stage.Save()

        uvs = UVList.load_usd(self.path)
        self.assertEqual(len(uvs), 1)
        np.testing.assert_array_equal(uvs[0].points, [[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(len(MeshList.load_usd(self.path)), 1)

    def test_points_kept_only_as_time_samples_read_their_first(self):
        mesh   = self.quad("/animated")
        points = mesh.CreatePointsAttr()
        points.Set(self.corners(2), 1)
        points.Set(self.corners(3), 2)
        self.stage.Save()

        meshes = MeshList.load_usd(self.path)
        np.testing.assert_array_equal(meshes[0].points, np.asarray(self.corners(2)))


if __name__ == "__main__":
    unittest.main()
