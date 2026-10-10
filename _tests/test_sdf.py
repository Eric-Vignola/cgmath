import copy
import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath.geometry import MeshData
from cgmath.geometry.sdf import DMCField, SDFBox, SDFCylinder, SDFSphere
from cgmath.hierarchy import ClipData, HierarchyData, TransformData


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

        ground_cutter           = SDFBox(half_extents=[3.0, 1.5, 3.0], name="ground_cut")
        ground_cutter.translate = [0, -1.5, 0]
        field.subtract(ground_cutter)

        # === ENTRANCE TUNNEL ===
        tunnel_outer           = SDFCylinder(radius=0.5, height=1.2, name="tunnel_outer")
        tunnel_outer.translate = [0, 0, 1.8]
        tunnel_outer.rotate    = [90, 0, 0]
        field.add(tunnel_outer, smoothing=0.15)

        tunnel_inner           = SDFCylinder(radius=0.35, height=1.4, name="tunnel_inner")
        tunnel_inner.translate = [0, 0, 1.85]
        tunnel_inner.rotate    = [90, 0, 0]
        field.subtract(tunnel_inner)

        tunnel_floor           = SDFBox(half_extents=[0.6, 0.25, 1.0], name="tunnel_floor")
        tunnel_floor.translate = [0, -0.25, 1.8]
        field.subtract(tunnel_floor)

        # === ENTRANCE ARCH ===
        doorway           = SDFCylinder(radius=0.38, height=0.5, name="doorway")
        doorway.translate = [0, 0, 1.3]
        doorway.rotate    = [90, 0, 0]
        field.subtract(doorway)

        # === FLOOR ===
        interior_floor           = SDFBox(half_extents=[1.2, 0.1, 1.2], name="interior_floor")
        interior_floor.translate = [0, -0.1, 0]
        field.subtract(interior_floor)

        # === VENTILATION HOLE ===
        vent_hole           = SDFCylinder(radius=0.12, height=0.5, name="vent_hole")
        vent_hole.translate = [0, 1.35, 0]
        field.subtract(vent_hole)

        # === ICE BLOCK DETAILS ===
        num_base_blocks = 16
        for i in range(num_base_blocks):
            angle = (2 * np.pi * i) / num_base_blocks
            if abs(angle - np.pi / 2) < 0.4 or abs(angle - 3 * np.pi / 2) < 0.4:
                continue
            block           = SDFBox(half_extents=[0.25, 0.12, 0.08], name=f"block_base_{i}")
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
                tilt         = np.degrees(np.arctan2(height, radius_at_height))
                block.rotate = [tilt, -np.degrees(angle), 0]
                field.add(block, smoothing=0.05)

        # === SNOW MOUNDS ===
        snow_mound           = SDFSphere(radius=0.4, name="snow_mound_1")
        snow_mound.translate = [1.2, -0.15, 0.8]
        snow_mound.scale     = [1.0, 0.3, 1.0]
        field.add(snow_mound, smoothing=0.2)

        snow_mound2           = SDFSphere(radius=0.35, name="snow_mound_2")
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


class TestSDFPrimitiveSettings(unittest.TestCase):
    """a primitive's shape settings are data: copies and files keep them"""

    def setUp(self):
        super().setUp()
        self.primitives = [
            SDFSphere(radius=2.5, translate=[1, 0, 0], name="ball"),
            SDFBox(half_extents=[1, 2, 3], rotate=[0, 30, 0], name="crate"),
            SDFCylinder(radius=0.3, height=4, axis=2, scale=[1, 2, 1], name="pipe"),
        ]
        self.field = DMCField(resolution=8)
        for primitive in self.primitives:
            self.field.add(primitive)

        grid = np.linspace(-2, 2, 6)
        self.X, self.Y, self.Z = np.meshgrid(grid, grid, grid, indexing="ij")

    def assertSameShape(self, a, b):
        self.assertIs(type(a), type(b))
        for name in ("radius", "half_extents", "height", "axis"):
            if hasattr(a, name):
                self.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), name)
        grid = (self.X, self.Y, self.Z)
        self.assertTrue(np.allclose(a.sample(*grid), b.sample(*grid)))

    def test_copy(self):
        for primitive in self.primitives:
            with self.subTest(primitive=primitive.name):
                copy = primitive.copy()
                self.assertSameShape(primitive, copy)

                # a copy is not in the field: editing it leaves the field clean
                self.assertIsNone(copy._field)
                self.field.mesh_data
                copy.translate = [5, 5, 5]
                self.assertFalse(self.field._dirty)

    def test_pickle(self):
        for primitive in self.primitives:
            with self.subTest(primitive=primitive.name):
                self.assertSameShape(primitive, pickle.loads(pickle.dumps(primitive)))

    def test_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for primitive in self.primitives:
                for ext in ("pkl", "npz", "json"):
                    with self.subTest(primitive=primitive.name, format=ext):
                        f = os.path.join(temp_dir, f"{primitive.name}.{ext}")
                        primitive.save(f)
                        loaded = type(primitive).load(f)
                        self.assertSameShape(primitive, loaded)
                        self.assertTrue(loaded == primitive)

    def test_settings_still_invalidate(self):
        """the setters still mark the field dirty"""
        sphere, box, cylinder = self.primitives
        edits = [
            (sphere, "radius", 1.0),
            (box, "half_extents", [1, 1, 1]),
            (cylinder, "radius", 1.0),
            (cylinder, "height", 1.0),
            (cylinder, "axis", 0),
        ]
        for primitive, name, value in edits:
            with self.subTest(setting=f"{primitive.name}.{name}"):
                self.field.mesh_data
                self.assertFalse(self.field._dirty)
                setattr(primitive, name, value)
                self.assertTrue(self.field._dirty)

    def test_a_copied_field_still_follows_its_primitives(self):
        # its primitives are copies too: an edit must mark the copy dirty, not
        # leave it showing the old mesh
        self.field.mesh_data
        for label, clone in (
            ("deepcopy", copy.deepcopy(self.field)),
            ("pickle", pickle.loads(pickle.dumps(self.field))),
        ):
            with self.subTest(label):
                clone.mesh_data
                self.assertFalse(clone._dirty)
                clone._nodes[0].primitive.radius = 0.5
                self.assertTrue(clone._dirty)
                self.assertFalse(self.field._dirty)

        # a shallow copy shares the original's primitives, which stay its own
        shallow = copy.copy(self.field)
        self.assertIs(shallow._nodes[0].primitive._field, self.field)

    def test_settings_checked(self):
        with self.assertRaises(ValueError):
            SDFBox(half_extents=[1, 2])
        with self.assertRaises(ValueError):
            SDFCylinder(axis=3)
        with self.assertRaises(ValueError):
            self.primitives[2].axis = -1

    def test_settings_are_floats(self):
        sphere = SDFSphere(radius=1)
        self.assertIs(type(sphere.radius), float)
        self.assertEqual(SDFBox(half_extents=[1, 2, 3]).half_extents.dtype, np.float64)


class TestSDFFieldFollowsTransforms(unittest.TestCase):
    """moving, rotating or scaling a primitive, or a parent, marks its field dirty"""

    def setUp(self):
        super().setUp()
        self.primitives = [
            SDFSphere(radius=0.5, name="ball"),
            SDFBox(half_extents=[0.5, 0.25, 0.25], name="crate"),
            SDFCylinder(radius=0.25, height=1.0, name="pipe"),
        ]

    def test_a_transform_change_marks_the_field_dirty(self):
        edits = [
            ("translate", [0.5, 0.0, 0.0]),
            ("rotate", [0.0, 0.0, 30.0]),
            ("scale", [1.0, 2.0, 1.0]),
            ("translate_y", 0.25),
            ("rotate_order", 2),
        ]
        for primitive in self.primitives:
            field = DMCField(resolution=8).add(primitive)
            for name, value in edits:
                with self.subTest(primitive=primitive.name, channel=name):
                    field.mesh_data
                    self.assertFalse(field._dirty)
                    setattr(primitive, name, value)
                    self.assertTrue(field._dirty)

    def test_the_mesh_follows_a_moved_primitive(self):
        sphere = self.primitives[0]
        field  = DMCField(resolution=16).add(sphere)
        centre = field.mesh_data.points.mean(axis=0)
        self.assertTrue(np.allclose(centre, [0.0, 0.0, 0.0], atol=1e-3))

        # the old mesh, still centred on the origin, must not come back
        sphere.translate = [2.0, 0.0, 0.0]
        centre           = field.mesh_data.points.mean(axis=0)
        self.assertTrue(np.allclose(centre, [2.0, 0.0, 0.0], atol=1e-3))

    def test_a_small_move_far_from_the_origin_marks_the_field_dirty(self):
        # 0.0009 at x = 100 is below the change check's old tolerance, which
        # grew with the value, so the move counted as no change
        sphere           = self.primitives[0]
        sphere.translate = [100.0, 0.0, 0.0]
        field            = DMCField(resolution=8).add(sphere)
        field.mesh_data

        sphere.translate = [100.0009, 0.0, 0.0]
        self.assertTrue(field._dirty)

    def test_a_small_matrix_move_far_from_the_origin_marks_the_field_dirty(self):
        # the matrix setters had the same check, so the move was dropped and
        # the field kept its mesh
        sphere = self.primitives[0]
        field  = DMCField(resolution=8).add(sphere)
        for setter in ("matrix", "world_matrix"):
            with self.subTest(setter):
                sphere.translate = [100.0, 0.0, 0.0]
                field.mesh_data

                M       = getattr(sphere, setter)
                M[3, 0] = 100.0009
                setattr(sphere, setter, M)
                self.assertTrue(field._dirty)

    def test_an_unchanged_transform_keeps_the_mesh(self):
        sphere = self.primitives[0]
        field  = DMCField(resolution=8).add(sphere)
        mesh   = field.mesh_data

        sphere.translate = sphere.translate
        self.assertIs(field.mesh_data, mesh)

        # nor does a primitive the field no longer holds
        field.remove(sphere)
        mesh             = field.mesh_data
        sphere.translate = [1.0, 0.0, 0.0]
        self.assertIs(field.mesh_data, mesh)

    def test_a_parent_moving_marks_the_field_dirty(self):
        group = TransformData(name="group")
        rig   = HierarchyData([group])
        field = DMCField(resolution=8)
        for primitive in self.primitives:
            primitive.parent_uuid = group.uuid
            rig.append(primitive)
            field.add(primitive)

        edits = [
            ("translate", [1.0, 0.0, 0.0]),
            ("rotate", [0.0, 45.0, 0.0]),
            ("scale", [2.0, 2.0, 2.0]),
        ]
        for name, value in edits:
            with self.subTest(channel=name):
                field.mesh_data
                self.assertFalse(field._dirty)
                setattr(group, name, value)
                self.assertTrue(field._dirty)

    def test_a_clip_changing_frame_marks_the_field_dirty(self):
        clip           = ClipData([self.primitives[0]], frames=2)
        ball           = clip[0]
        clip.frame     = 1
        ball.translate = [1.0, 0.0, 0.0]

        clip.frame = 0
        field      = DMCField(resolution=8).add(ball)
        field.mesh_data
        clip.frame = 1
        self.assertTrue(field._dirty)

    def test_a_copied_field_follows_its_moved_primitives(self):
        field = DMCField(resolution=8)
        for primitive in self.primitives:
            field.add(primitive)
        field.mesh_data

        for label, clone in (
            ("deepcopy", copy.deepcopy(field)),
            ("pickle", pickle.loads(pickle.dumps(field))),
        ):
            with self.subTest(label):
                clone.mesh_data
                self.assertFalse(clone._dirty)
                clone.primitives[0].translate = [1.0, 0.0, 0.0]
                self.assertTrue(clone._dirty)
                self.assertFalse(field._dirty)
