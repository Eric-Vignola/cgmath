"""CameraData and LightData: Maya's camera and light attributes as shapes."""

import math
import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry import CameraData, CameraList, LightData, LightList
from cgmath.geometry.camera import FILM_FITS
from cgmath.geometry.light import DECAY_RATES, LIGHT_KINDS
from cgmath.hierarchy import HierarchyData, ShapeData, TransformData


def rig() -> HierarchyData:
    """a group turned 90 about y over a child moved 5 along z"""
    rig = HierarchyData()
    rig.append(TransformData("grp", rotate=(0.0, 90.0, 0.0)))
    rig.append(TransformData("cam", translate=(0.0, 0.0, 5.0)))
    rig["cam"].parent_uuid = rig["grp"].uuid
    return rig


class TestCameraData(unittest.TestCase):
    def test_defaults_are_a_fresh_maya_camera(self):
        """cmds.camera() in Maya 2025 (measured): 35 mm, 1.41732 x 0.94488 in,
        horizontal fit, no offset, squeeze 1, scale 1, clips 0.1 and 10000,
        perspective, orthographic width 10"""
        camera = CameraData()
        self.assertIsInstance(camera, ShapeData)
        self.assertEqual(
            (camera.focal_length, camera.horizontal_film_aperture, camera.vertical_film_aperture, camera.film_fit),
            (35.0, 1.41732, 0.94488, "horizontal"),
        )
        np.testing.assert_array_equal(camera.film_offset, [0.0, 0.0])
        self.assertEqual((camera.lens_squeeze_ratio, camera.camera_scale), (1.0, 1.0))
        self.assertEqual((camera.near_clip_plane, camera.far_clip_plane),  (0.1, 10000.0))
        self.assertEqual((camera.orthographic, camera.orthographic_width), (False, 10.0))
        self.assertIsNone(camera.name)
        self.assertIsNotNone(camera.uuid)
        self.assertIsNone(camera.parent_uuid)

    def test_angle_of_view_as_maya_derives_it(self):
        """2 atan(aperture / 2 focal), the aperture in mm; Maya's angleOfView
        is the horizontal one: 54.432 degrees for the default camera. Maya's
        own query takes neither the camera scale nor the lens squeeze into
        account (measured in Maya 2025), so neither does this"""
        camera = CameraData()
        self.assertAlmostEqual(camera.angle_of_view, math.degrees(2.0 * math.atan(1.41732 * 25.4 / 70.0)), places=12)
        self.assertAlmostEqual(camera.angle_of_view, 54.4322, places=3)
        self.assertAlmostEqual(camera.vertical_angle_of_view, math.degrees(2.0 * math.atan(0.94488 * 25.4 / 70.0)), places=12)
        self.assertAlmostEqual(camera.film_aspect_ratio, 1.5, places=6)
        camera.focal_length       = 50.0
        camera.lens_squeeze_ratio = 2.0
        camera.camera_scale       = 0.5
        self.assertAlmostEqual(camera.horizontal_angle_of_view, math.degrees(2.0 * math.atan(1.41732 * 25.4 / 100.0)), places=12)
        self.assertAlmostEqual(camera.vertical_angle_of_view, math.degrees(2.0 * math.atan(0.94488 * 25.4 / 100.0)), places=12)

    def test_film_fit_is_one_of_mayas(self):
        camera = CameraData()
        for fit in FILM_FITS:
            camera.film_fit = fit
        with self.assertRaisesRegex(ValueError, r"film_fit is one of \['fill', 'horizontal', 'vertical', 'overscan'\], not 'wide'"):
            camera.film_fit = "wide"
        with self.assertRaises(ValueError):
            CameraData(film_fit="fill ")
        self.assertEqual(FILM_FITS.index("fill"), 0)  # Maya's filmFit enum order

    def test_placed_by_its_transform(self):
        nodes  = rig()
        camera = CameraData(name="shot", parent_uuid=nodes["cam"].uuid)
        self.assertIs(camera.transform(nodes), nodes["cam"])
        np.testing.assert_allclose(camera.world_matrix(nodes), nodes["cam"].world_matrix)
        np.testing.assert_allclose(camera.world_matrix(nodes)[3, :3], [5.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_array_equal(CameraData().world_matrix(nodes), np.eye(4))
        # a camera's own values do not count in ==, nor its ids
        self.assertEqual(camera, CameraData(name="other", focal_length=35.0))
        self.assertNotEqual(camera, CameraData(focal_length=50.0))

    def test_saved_and_loaded(self):
        camera = CameraData(
            name="shot", focal_length=50.0, horizontal_film_aperture=1.0, vertical_film_aperture=0.6, film_fit="vertical",
            film_offset=[0.1, -0.05], lens_squeeze_ratio=1.5, camera_scale=2.0, near_clip_plane=1.0, far_clip_plane=5000.0,
            orthographic=True, orthographic_width=50.0,
        )
        with tempfile.TemporaryDirectory() as folder:
            for ext in ("json", "npz", "pkl"):
                with self.subTest(ext):
                    back = CameraData.load(camera.save(os.path.join(folder, f"c.{ext}")))
                    self.assertEqual(back, camera)
                    self.assertEqual((back.name, back.uuid, back.film_fit, back.orthographic), ("shot", camera.uuid, "vertical", True))
                    np.testing.assert_array_equal(back.film_offset, [0.1, -0.05])
            clips = CameraList([camera, CameraData()])
            back  = CameraList.load(clips.save(os.path.join(folder, "cams.json")))
            self.assertEqual(back, clips)
            self.assertEqual([c.name for c in back], ["shot", None])


class TestLightData(unittest.TestCase):
    def test_defaults_are_a_fresh_maya_light(self):
        """Maya 2025's point, directional and spot lights (measured): white,
        intensity 1, no decay; a spot's cone 40, penumbra 0, dropoff 0"""
        light = LightData()
        self.assertIsInstance(light, ShapeData)
        self.assertEqual(light.kind, "point")
        np.testing.assert_array_equal(light.color, [1.0, 1.0, 1.0])
        self.assertEqual((light.intensity, light.decay_rate, light.decay), (1.0, 0, "none"))
        self.assertEqual((light.cone_angle, light.penumbra_angle, light.dropoff), (40.0, 0.0, 0.0))

    def test_kind_and_decay_are_mayas(self):
        light = LightData()
        for kind in LIGHT_KINDS:
            light.kind = kind
        with self.assertRaisesRegex(ValueError, r"kind is one of \['point', 'directional', 'spot'\], not 'ambient'"):
            light.kind = "ambient"
        for rate, name in enumerate(DECAY_RATES):
            light.decay_rate = rate
            self.assertEqual(light.decay, name)
        light.decay_rate = 2.0  # a whole number is taken as the int it is
        self.assertEqual(light.decay_rate, 2)
        self.assertIsInstance(light.decay_rate, int)
        for bad in (4, -1, 1.5, True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "decay_rate is Maya's decayRate, 0 to 3"):
                light.decay_rate = bad

    def test_placed_and_saved(self):
        nodes = rig()
        light = LightData(name="key", kind="spot", color=[0.25, 0.5, 1.0], intensity=2.5, decay_rate=2, cone_angle=50.0,
                          penumbra_angle=5.0, dropoff=3.0, parent_uuid=nodes["cam"].uuid)
        self.assertIs(light.transform(nodes), nodes["cam"])
        self.assertEqual(light, LightData(kind="spot", color=[0.25, 0.5, 1.0], intensity=2.5, decay_rate=2, cone_angle=50.0,
                                          penumbra_angle=5.0, dropoff=3.0))
        self.assertNotEqual(light, LightData())
        with tempfile.TemporaryDirectory() as folder:
            for ext in ("json", "npz", "pkl"):
                with self.subTest(ext):
                    back = LightData.load(light.save(os.path.join(folder, f"l.{ext}")))
                    self.assertEqual(back, light)
                    self.assertEqual((back.name, back.kind, back.decay_rate, back.parent_uuid), ("key", "spot", 2, nodes["cam"].uuid))
            lights = LightList([light, LightData(kind="directional")])
            back   = LightList.load(lights.save(os.path.join(folder, "lights.npz")))
            self.assertEqual(back, lights)
            self.assertEqual([x.kind for x in back], ["spot", "directional"])


if __name__ == "__main__":
    unittest.main()
