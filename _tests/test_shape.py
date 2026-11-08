"""ShapeData: a uuid of its own, the transform it sits under, and the files that keep both"""

import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.geometry.skin_weights import SkinData
from cgmath.hierarchy import ClipData, HierarchyData, ShapeData, TransformData, generate_uuid


def quad(**kwargs) -> MeshData:
    return MeshData(
        indices = np.arange(4, dtype=np.int32),
        counts  = np.array([4], dtype=np.int32),
        points  = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64),
        **kwargs,
    )


class TestAShapeHasAUuid(unittest.TestCase):
    def test_a_uuid_is_made_when_none_is_given_and_kept_when_one_is(self):
        a, b = quad(), quad()
        self.assertIsInstance(a.uuid, str)
        self.assertNotEqual(a.uuid, b.uuid)
        given = generate_uuid()
        self.assertEqual(quad(uuid=given).uuid, given)
        self.assertIsNone(quad().parent_uuid)

    def test_the_ids_do_not_count_in_equality(self):
        under = TransformData(name="under")
        self.assertEqual(quad(), quad(parent_uuid=under.uuid or generate_uuid()))

    def test_a_name_is_no_uuid(self):
        mesh = quad()
        with self.assertRaisesRegex(ValueError, r"^MeshData\.parent_uuid takes a uuid or None, not 'body'$"):
            mesh.parent_uuid = "body"
        with self.assertRaisesRegex(ValueError, r"^MeshData\.uuid takes a uuid or None"):
            quad(uuid="not one")

    def test_every_shape_class_is_one(self):
        from cgmath.geometry.bspline import NurbsCurveData
        from cgmath.geometry.bspline_patch import NurbsSurfaceData

        self.assertTrue(issubclass(MeshData, ShapeData))
        self.assertTrue(issubclass(UVData, ShapeData))
        self.assertTrue(issubclass(NurbsCurveData, ShapeData))
        self.assertTrue(issubclass(NurbsSurfaceData, ShapeData))
        curve = NurbsCurveData(points=np.zeros((4, 3)))
        surface = NurbsSurfaceData(
            points=np.zeros((4, 4, 3)), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False
        )
        self.assertIsInstance(curve.uuid, str)
        self.assertIsInstance(surface.uuid, str)


class TestAShapeIsPlacedByItsTransform(unittest.TestCase):
    def setUp(self):
        self.root  = TransformData(name="root", translate=[0.0, 10.0, 0.0])
        self.child = TransformData(name="child", translate=[1.0, 0.0, 0.0], rotate=[0.0, 90.0, 0.0])
        self.rig   = HierarchyData([self.root, self.child])
        self.child.set_parent(self.root, world_space=False)  # its translate stays local

    def test_the_transform_and_its_world_matrix(self):
        mesh = quad(parent_uuid=self.child.uuid)
        self.assertIs(mesh.transform(self.rig), self.child)
        np.testing.assert_allclose(mesh.world_matrix(self.rig), self.child.world_matrix)
        np.testing.assert_allclose(mesh.world_matrix(self.rig)[3, :3], [1.0, 10.0, 0.0])

    def test_no_transform_is_identity(self):
        loose = quad()
        self.assertIsNone(loose.transform(self.rig))
        np.testing.assert_array_equal(loose.world_matrix(self.rig), np.eye(4))
        # one the hierarchy does not hold, and no hierarchy at all
        elsewhere = quad(parent_uuid=generate_uuid())
        self.assertIsNone(elsewhere.transform(self.rig))
        np.testing.assert_array_equal(elsewhere.world_matrix(self.rig), np.eye(4))
        np.testing.assert_array_equal(elsewhere.world_matrix(None), np.eye(4))

    def test_a_clip_places_it_frame_by_frame(self):
        clip = ClipData(self.rig, frames=3)
        for frame in range(3):
            clip.frame              = frame
            clip["child"].translate = [float(frame), 0.0, 0.0]
        mesh = quad(parent_uuid=self.rig["child"].uuid)
        for frame in range(3):
            clip.frame = frame
            np.testing.assert_allclose(mesh.world_matrix(clip)[3, :3], [float(frame), 10.0, 0.0])


class TestTheIdsSave(unittest.TestCase):
    def test_every_format_keeps_uuid_parent_uuid_and_mesh_uuid(self):
        under = TransformData(name="under")
        HierarchyData([under])
        mesh = quad(name="body", parent_uuid=under.uuid)
        uv = UVData(
            indices   = mesh.indices.copy(),
            counts    = mesh.counts.copy(),
            points    = np.zeros((4, 2)),
            mesh_uuid = mesh.uuid,
        )
        skin = SkinData(weights=np.ones((4, 1)), influences=["under"], mesh_uuid=mesh.uuid)
        with tempfile.TemporaryDirectory() as tmp:
            for ext in ("json", "npz", "pkl"):
                with self.subTest(ext=ext):
                    back = MeshData.load(mesh.save(os.path.join(tmp, f"mesh.{ext}")))
                    self.assertEqual((back.uuid, back.parent_uuid), (mesh.uuid, under.uuid))
                    back = UVData.load(uv.save(os.path.join(tmp, f"uv.{ext}")))
                    self.assertEqual((back.uuid, back.mesh_uuid, back.parent_uuid), (uv.uuid, mesh.uuid, None))
                    back = SkinData.load(skin.save(os.path.join(tmp, f"skin.{ext}")))
                    self.assertEqual(back.mesh_uuid, mesh.uuid)

    def test_a_file_saved_before_shapes_had_a_uuid_gets_one(self):
        tree = quad().to_dict()
        del tree["uuid"]
        back = MeshData.from_dict(tree)
        self.assertIsInstance(back.uuid, str)
        self.assertIsNone(back.parent_uuid)


if __name__ == "__main__":
    unittest.main()
