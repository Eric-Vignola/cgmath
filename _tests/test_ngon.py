import os
import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData, UVList, skimage
from cgmath.geometry.utils._numba._cdt import check_delaunay_violation


class TestNgon(unittest.TestCase):
    """Tests for n-gon triangulation and UV rasterization with hole support."""

    def setUp(self):
        super().setUp()
        root   = os.path.split(__file__)[0]
        assets = os.path.join(root, "test_assets")

        self.mesh_data_path = os.path.join(assets, "test_ngon.source_mesh_data.json")
        self.mesh_data      = MeshData.load(self.mesh_data_path)

        self.uv_list_path   = os.path.join(assets, "test_ngon.source_uv_list.json")
        self.uv_list        = UVList.load(self.uv_list_path)

        self.expected_mesh_tris_po_no_path = os.path.join(
            assets,
            "test_ngon.triangulate_mesh_data_preserve_order_ngons_only.npz",
        )
        self.expected_mesh_tris_po_no = MeshData.load(
            self.expected_mesh_tris_po_no_path
        )

        self.expected_mesh_tris_po_path = os.path.join(
            assets, "test_ngon.triangulate_mesh_data_preserve_order.npz"
        )
        self.expected_mesh_tris_po = MeshData.load(self.expected_mesh_tris_po_path)

        self.expected_mesh_tris_no_path = os.path.join(
            assets, "test_ngon.triangulate_mesh_data_ngons_only.npz"
        )
        self.expected_mesh_tris_no = MeshData.load(self.expected_mesh_tris_no_path)

        self.expected_mesh_tris_path = os.path.join(
            assets, "test_ngon.triangulate_mesh_data.npz"
        )
        self.expected_mesh_tris = MeshData.load(self.expected_mesh_tris_path)

        self.expected_uv_tris_po_no_path = os.path.join(
            assets,
            "test_ngon.triangulate_uv_list_preserve_order_ngons_only.npz",
        )
        self.expected_uv_tris_po_no = UVList.load(self.expected_uv_tris_po_no_path)

        self.expected_uv_tris_po_path = os.path.join(
            assets, "test_ngon.triangulate_uv_list_preserve_order.npz"
        )
        self.expected_uv_tris_po = UVList.load(self.expected_uv_tris_po_path)

        self.expected_uv_tris_no_path = os.path.join(
            assets, "test_ngon.triangulate_uv_list_ngons_only.npz"
        )
        self.expected_uv_tris_no = UVList.load(self.expected_uv_tris_no_path)

        self.expected_uv_tris_path = os.path.join(
            assets, "test_ngon.triangulate_uv_list.npz"
        )
        self.expected_uv_tris = UVList.load(self.expected_uv_tris_path)

        self.expected_buffer = np.load(
            os.path.join(assets, "test_ngon.bitmap_buffer.npz")
        )["buffer"]

    # --- Mesh triangulation ---

    def test_triangulate_preserve_order_ngons_only(self):
        data  = self.mesh_data.copy()
        rules = data.get_triangulate_rules(ngons_only=True)
        data.triangulate(rules, preserve_order=True)
        self.assertTrue(data == self.expected_mesh_tris_po_no)

    def test_triangulate_ngons_only(self):
        data  = self.mesh_data.copy()
        rules = data.get_triangulate_rules(ngons_only=True)
        data.triangulate(rules, preserve_order=False)
        self.assertTrue(data == self.expected_mesh_tris_no)

    def test_triangulate_preserve_order(self):
        data  = self.mesh_data.copy()
        rules = data.get_triangulate_rules(ngons_only=False)
        data.triangulate(rules, preserve_order=True)
        self.assertTrue(data == self.expected_mesh_tris_po)

    def test_triangulate(self):
        data  = self.mesh_data.copy()
        rules = data.get_triangulate_rules(ngons_only=False)
        data.triangulate(rules, preserve_order=False)
        self.assertTrue(data == self.expected_mesh_tris)

    # --- UV triangulation ---

    def test_uv_triangulate_preserve_order_ngons_only(self):
        data    = self.mesh_data.copy()
        rules   = data.get_triangulate_rules(ngons_only=True)
        uv_data = self.uv_list.copy()
        uv_data[0].triangulate(rules, preserve_order=True)
        self.assertTrue(uv_data == self.expected_uv_tris_po_no)

    def test_uv_triangulate_ngons_only(self):
        data    = self.mesh_data.copy()
        rules   = data.get_triangulate_rules(ngons_only=True)
        uv_data = self.uv_list.copy()
        uv_data[0].triangulate(rules, preserve_order=False)
        self.assertTrue(uv_data == self.expected_uv_tris_no)

    def test_uv_triangulate_preserve_order(self):
        data    = self.mesh_data.copy()
        rules   = data.get_triangulate_rules(ngons_only=False)
        uv_data = self.uv_list.copy()
        uv_data[0].triangulate(rules, preserve_order=True)

        self.assertTrue(uv_data == self.expected_uv_tris_po)

    def test_uv_triangulate(self):
        data    = self.mesh_data.copy()
        rules   = data.get_triangulate_rules(ngons_only=False)
        uv_data = self.uv_list.copy()
        uv_data[0].triangulate(rules, preserve_order=False)
        self.assertTrue(uv_data == self.expected_uv_tris)

    # --- UV bitmap rendering ---

    @unittest.skipIf(skimage is None, "scikit-image is not installed")
    def test_uv_bitmap_render(self):
        uv_data               = self.uv_list.copy()
        uv_data[0].renderer   = "skimage"
        uv_data[0].resolution = 512
        uv_data[0].draw_mask()
        self.assertTrue(np.allclose(uv_data[0].buffer, self.expected_buffer))


class TestDelaunayViolation(unittest.TestCase):
    """The edge flip test of the n-gon triangulation."""

    # four points on a circle of radius 1.5 (a 3-4-5 triangle's, scaled):
    # edge (0, 2) with (1, 3) across it. 0.9 and 1.2 are not exact in binary,
    # so the circle test of these cocircular points is rounding noise
    COCIRCULAR = np.array([[1.5, 0.0], [0.9, 1.2], [-0.9, 1.2], [-1.5, 0.0]])

    def test_cocircular_points_do_not_flip(self):
        """a tie is not a violation, whatever its rounding noise, at any scale or offset"""
        for scale in (1.0, 1e-3, 1e3):
            for offset in (0.0, 0.1, 10.7):
                with self.subTest(scale=scale, offset=offset):
                    points = self.COCIRCULAR * scale + offset
                    self.assertFalse(check_delaunay_violation(points, 0, 2, 1, 3))
                    self.assertFalse(check_delaunay_violation(points, 1, 3, 0, 2))

    def test_violations_still_flip(self):
        """a point inside the circle by more than the tolerance is a violation"""
        for inside in (0.5, 1e-6):
            for scale in (1.0, 1e-3, 1e3):
                with self.subTest(inside=inside, scale=scale):
                    points = self.COCIRCULAR * scale
                    points[3, 0] *= 1.0 - inside  # pulled in towards the centre
                    self.assertTrue(check_delaunay_violation(points, 0, 2, 1, 3))
