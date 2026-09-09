import os
import unittest

import numpy as np
from cgmath.geometry import MeshData
from cgmath.geometry.sdf import DMCField, SDFBox, SDFCylinder, SDFSphere


class TestSDFIgloo(unittest.TestCase):
    """Test SDF mesh generation produces consistent output."""

    def setUp(self):
        super().setUp()
        root = os.path.dirname(__file__)
        self.igloo_reference = MeshData.load(
            os.path.join(root, "test_assets", "test_sdf.igloo.npz")
        )

    def _create_igloo(self) -> MeshData:
        """Build a classic igloo with entrance tunnel and interior space."""
        field = DMCField(resolution=16)

        # === MAIN DOME ===
        dome_outer = SDFSphere(radius=1.5, name="dome_outer")
        field.add(dome_outer)

        dome_inner = SDFSphere(radius=1.3, name="dome_inner")
        field.subtract(dome_inner)

        ground_cutter = SDFBox(half_extents=[3.0, 1.5, 3.0], name="ground_cut")
        ground_cutter.translate = [0, -1.5, 0]
        field.subtract(ground_cutter)

        # === ENTRANCE TUNNEL ===
        tunnel_outer = SDFCylinder(radius=0.5, height=1.2, name="tunnel_outer")
        tunnel_outer.translate = [0, 0, 1.8]
        tunnel_outer.rotate    = [90, 0, 0]
        field.add(tunnel_outer, smoothing=0.15)

        tunnel_inner = SDFCylinder(radius=0.35, height=1.4, name="tunnel_inner")
        tunnel_inner.translate = [0, 0, 1.85]
        tunnel_inner.rotate    = [90, 0, 0]
        field.subtract(tunnel_inner)

        tunnel_floor = SDFBox(half_extents=[0.6, 0.25, 1.0], name="tunnel_floor")
        tunnel_floor.translate = [0, -0.25, 1.8]
        field.subtract(tunnel_floor)

        # === ENTRANCE ARCH ===
        doorway = SDFCylinder(radius=0.38, height=0.5, name="doorway")
        doorway.translate = [0, 0, 1.3]
        doorway.rotate    = [90, 0, 0]
        field.subtract(doorway)

        # === FLOOR ===
        interior_floor = SDFBox(half_extents=[1.2, 0.1, 1.2], name="interior_floor")
        interior_floor.translate = [0, -0.1, 0]
        field.subtract(interior_floor)

        # === VENTILATION HOLE ===
        vent_hole = SDFCylinder(radius=0.12, height=0.5, name="vent_hole")
        vent_hole.translate = [0, 1.35, 0]
        field.subtract(vent_hole)

        # === ICE BLOCK DETAILS ===
        num_base_blocks = 16
        for i in range(num_base_blocks):
            angle = (2 * np.pi * i) / num_base_blocks
            if abs(angle - np.pi / 2) < 0.4 or abs(angle - 3 * np.pi / 2) < 0.4:
                continue
            block = SDFBox(half_extents=[0.25, 0.12, 0.08], name=f"block_base_{i}")
            block.translate = [np.cos(angle) * 1.45, 0.15, np.sin(angle) * 1.45]
            block.rotate    = [0, -np.degrees(angle), 0]
            field.add(block, smoothing=0.08)

        for i in range(num_base_blocks):
            angle = (2 * np.pi * (i + 0.5)) / num_base_blocks
            if abs(angle - np.pi / 2) < 0.5:
                continue
            block            = SDFBox(half_extents=[0.22, 0.11, 0.07], name=f"block_mid_{i}")
            radius_at_height = np.sqrt(1.45**2 - 0.4**2)
            block.translate = [
                np.cos(angle) * radius_at_height,
                0.45,
                np.sin(angle) * radius_at_height,
            ]
            block.rotate = [8, -np.degrees(angle), 0]
            field.add(block, smoothing=0.06)

        for ring in range(2, 5):
            height = 0.3 + ring * 0.28
            if height > 1.2:
                break
            radius_at_height = np.sqrt(max(0, 1.5**2 - height**2)) * 0.97
            num_blocks       = max(6, num_base_blocks - ring * 3)

            for i in range(num_blocks):
                angle = (2 * np.pi * (i + ring * 0.3)) / num_blocks
                block = SDFBox(
                    half_extents = [0.18 - ring * 0.02, 0.09, 0.05],
                    name         = f"block_r{ring}_{i}",
                )
                block.translate = [
                    np.cos(angle) * radius_at_height,
                    height,
                    np.sin(angle) * radius_at_height,
                ]
                tilt = np.degrees(np.arctan2(height, radius_at_height))
                block.rotate = [tilt, -np.degrees(angle), 0]
                field.add(block, smoothing=0.05)

        # === SNOW MOUNDS ===
        snow_mound = SDFSphere(radius=0.4, name="snow_mound_1")
        snow_mound.translate = [1.2, -0.15, 0.8]
        snow_mound.scale     = [1.0, 0.3, 1.0]
        field.add(snow_mound, smoothing=0.2)

        snow_mound2 = SDFSphere(radius=0.35, name="snow_mound_2")
        snow_mound2.translate = [-1.0, -0.15, -0.9]
        snow_mound2.scale     = [1.2, 0.25, 0.8]
        field.add(snow_mound2, smoothing=0.2)

        return field.mesh_data

    def test_igloo_mesh_vertices(self):
        """Test that igloo mesh vertices match reference."""
        mesh = self._create_igloo()
        self.assertEqual(mesh.point_count, self.igloo_reference.point_count)
        np.testing.assert_allclose(
            mesh.points, self.igloo_reference.points, rtol=1e-5, atol=1e-7
        )

    def test_igloo_mesh_faces(self):
        """Test that igloo mesh faces match reference."""
        mesh = self._create_igloo()
        self.assertEqual(mesh.face_count, self.igloo_reference.face_count)
        np.testing.assert_array_equal(mesh.indices, self.igloo_reference.indices)

    def test_igloo_mesh_validity(self):
        """Test that generated mesh is valid."""
        mesh = self._create_igloo()
        self.assertTrue(mesh.valid)