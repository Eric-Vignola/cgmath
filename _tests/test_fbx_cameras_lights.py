"""Cameras and lights through load_fbx and save_fbx: Maya's conventions, measured.

Maya's exporter (2025) turns a camera's node to fbx's camera axes with a post
rotation of (0, -90, 0) and a directional, spot or area light's node with
(-90, 0, 0); it writes the focal length and film back as they are
(ApertureMode eFocalLength), the film fit as the gate fit, the orthographic
width over 30 as the ortho zoom, a light's intensity times 100, its decay
rate as the decay type, a spot's cone as the inner angle and its cone plus
twice its penumbra as the outer; it drops the camera scale and the dropoff,
and writes an ambient light as a point light that casts no light. The fork
reads and writes the same. The SDK tests need the Autodesk SDK, the Maya ones
a Maya with its fbxmaya plug-in.
"""

import math
import os
import unittest
import warnings

import numpy as np
from cgmath._tests.test_skeleton_fbx import _Files, _InMaya, at_float32, counts
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.formats._load import _amatrix
from cgmath.formats._write import _enum
from cgmath.geometry import CameraData, CameraList, LightData, LightList, MeshData
from cgmath.hierarchy import ClipData, HierarchyData, TransformData

FBX = load_sdk()

FRAMES = 5
FPS    = 24.0


def views_rig() -> HierarchyData:
    """a turned group over the shot camera's node, an orthographic camera's
    node, three light nodes turned and moved, and a joint"""
    rig = HierarchyData()
    rig.append(TransformData("grp", rotate=(0.0, 45.0, 0.0)))
    rig.append(TransformData("shotCam", translate=(10.0, 20.0, 30.0), rotate=(10.0, 20.0, 30.0)))
    rig["shotCam"].parent_uuid = rig["grp"].uuid
    rig.append(TransformData("orthoCam", rotate=(-90.0, 0.0, 0.0)))
    rig.append(TransformData("key", translate=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0)))
    rig.append(TransformData("sun", rotate=(-45.0, 30.0, 0.0)))
    rig.append(TransformData("fill", translate=(0.0, 5.0, 0.0)))
    rig.append(TransformData("jnt", node_type="joint", translate=(0.0, 1.0, 0.0), joint_orient=(0.0, 0.0, 90.0)))
    return rig


def shot(rig) -> CameraData:
    return CameraData(
        name="shotShape", focal_length=50.0, horizontal_film_aperture=1.0, vertical_film_aperture=0.6, film_fit="vertical",
        film_offset=[0.1, -0.05], lens_squeeze_ratio=1.5, near_clip_plane=1.0, far_clip_plane=5000.0,
        parent_uuid=rig["shotCam"].uuid,
    )


def ortho(rig) -> CameraData:
    return CameraData(name="orthoShape", orthographic=True, orthographic_width=50.0, parent_uuid=rig["orthoCam"].uuid)


def views_lights(rig) -> LightList:
    return LightList([
        LightData(name="keyShape", kind="spot", color=[0.25, 0.5, 1.0], intensity=2.5, decay_rate=2, cone_angle=50.0,
                  penumbra_angle=5.0, parent_uuid=rig["key"].uuid),
        LightData(name="sunShape", kind="directional", intensity=0.5, parent_uuid=rig["sun"].uuid),
        LightData(name="fillShape", kind="point", color=[1.0, 0.9, 0.8], decay_rate=1, parent_uuid=rig["fill"].uuid),
    ])


def views_clip(rig) -> ClipData:
    """FRAMES frames: the shot camera turning about y, the key light about x"""
    clip                          = ClipData(rig, frames=FRAMES, start_frame=1001, fps=FPS, take="pan")
    f                             = np.arange(FRAMES, dtype=np.float64)[:, None]
    clip.frames["shotCam"].rotate = np.hstack([10.0 + f * 0.0, 20.0 + f * 15.0, 30.0 + f * 0.0])
    clip.frames["key"].rotate     = np.hstack([10.0 + f * 7.0, 20.0 + f * 0.0, 30.0 + f * 0.0])
    return clip


def same_views(test, got, want) -> None:
    """``got``, a CameraList or LightList read back, holds ``want``'s shapes
    in order, equal, named alike"""
    test.assertEqual(type(got), type(want))
    test.assertEqual(len(got), len(want))
    for a, b in zip(got, want):
        test.assertEqual(a, b)
        test.assertEqual(a.name, b.name)


def other_writer(path: str, keyed: bool = True) -> str:
    """an sdk file as another writer makes one: node cam (rotate 10, 20,
    30, no post rotation) holding a camera given by its horizontal field of
    view (60 degrees on a 1 by 0.5 inch back), its rotate y keyed 0 to 90
    over FRAMES frames at 24 fps when ``keyed``; node lens holding a camera
    whose focal length the take keys 35 to 50; node sun (rotate 30, 0, 0,
    no post rotation) holding a directional light; node lamp holding a
    point light whose intensity the take keys 100 to 300; nodes area, vol
    and amb holding an area light, a volume light and a point light that
    casts none"""
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene = FBX.FbxScene.Create(manager, "")
        scene.GetGlobalSettings().SetTimeMode(FBX.FbxTime.EMode.eFrames24)
        root = scene.GetRootNode()

        def node(name, rotate=(0.0, 0.0, 0.0), translate=(0.0, 0.0, 0.0)):
            made = FBX.FbxNode.Create(scene, name)
            made.SetRotationActive(True)
            made.SetTransformationInheritType(FBX.FbxTransform.EInheritType.eInheritRSrs)
            made.LclRotation.Set(FBX.FbxDouble3(*rotate))
            made.LclTranslation.Set(FBX.FbxDouble3(*translate))
            root.AddChild(made)
            return made

        cam    = node("cam", (10.0, 20.0, 30.0), (1.0, 2.0, 3.0))
        camera = FBX.FbxCamera.Create(scene, "camShape")
        camera.ApertureMode.Set(_enum(FBX.FbxCamera, "EApertureMode", "eHorizontal"))
        camera.FieldOfView.Set(60.0)
        camera.FilmWidth.Set(1.0)
        camera.FilmHeight.Set(0.5)
        cam.SetNodeAttribute(camera)
        lens = node("lens")
        zoom = FBX.FbxCamera.Create(scene, "lensShape")
        zoom.ApertureMode.Set(_enum(FBX.FbxCamera, "EApertureMode", "eFocalLength"))
        zoom.FocalLength.Set(35.0)
        lens.SetNodeAttribute(zoom)
        sun   = node("sun", (30.0, 0.0, 0.0))
        light = FBX.FbxLight.Create(scene, "sunShape")
        light.LightType.Set(_enum(FBX.FbxLight, "EType", "eDirectional"))
        sun.SetNodeAttribute(light)
        lamp  = node("lamp", translate=(0.0, 5.0, 0.0))
        point = FBX.FbxLight.Create(scene, "lampShape")
        point.LightType.Set(_enum(FBX.FbxLight, "EType", "ePoint"))
        point.Intensity.Set(100.0)
        lamp.SetNodeAttribute(point)
        for name, kind in (("area", "eArea"), ("vol", "eVolume"), ("amb", "ePoint")):
            held  = node(name)
            other = FBX.FbxLight.Create(scene, name + "Shape")
            other.LightType.Set(_enum(FBX.FbxLight, "EType", kind))
            if name == "amb":
                other.CastLight.Set(False)
            held.SetNodeAttribute(other)

        stack = FBX.FbxAnimStack.Create(scene, "take")
        layer = FBX.FbxAnimLayer.Create(scene, "BaseLayer")
        stack.AddMember(layer)
        times = []
        for frame in range(FRAMES):
            when = FBX.FbxTime()
            when.SetFrame(frame, FBX.FbxTime.EMode.eFrames24)
            times.append(when)
        stack.SetLocalTimeSpan(FBX.FbxTimeSpan(times[0], times[-1]))
        linear = FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear

        def key(curve, values):
            curve.KeyModifyBegin()
            for when, value in zip(times, values):
                index = curve.KeyAdd(when)[0]
                curve.KeySetValue(index, float(value))
                curve.KeySetInterpolation(index, linear)
            curve.KeyModifyEnd()

        if keyed:
            cam.LclRotation.GetCurveNode(layer, True)
            key(typed(cam.LclRotation.GetCurve(layer, "Y", True), FBX.FbxAnimCurve, FBX), np.linspace(0.0, 90.0, FRAMES))
        zoom.FocalLength.GetCurveNode(layer, True)
        key(typed(zoom.FocalLength.GetCurve(layer, True), FBX.FbxAnimCurve, FBX), np.linspace(35.0, 50.0, FRAMES))
        point.Intensity.GetCurveNode(layer, True)
        key(typed(point.Intensity.GetCurve(layer, True), FBX.FbxAnimCurve, FBX), np.linspace(100.0, 300.0, FRAMES))

        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


def fbx_nodes(path: str, frames=None) -> dict:
    """{node name: (post rotation, world (4, 4) at each of ``frames`` or
    static, the attribute's properties of note)} of a file, through the sdk"""
    out = {}
    with open_fbx(path, FBX) as scene:
        source = FBX.FbxNode.EPivotSet.eSourcePivot
        stack  = [scene.GetRootNode()]
        while stack:
            node = stack.pop()
            for i in range(node.GetChildCount()):
                stack.append(node.GetChild(i))
            if node is scene.GetRootNode():
                continue
            if frames is None:
                worlds = _amatrix(node.EvaluateGlobalTransform())[None]
            else:
                worlds = []
                for frame in frames:
                    when = FBX.FbxTime()
                    when.SetFrame(int(frame), FBX.FbxTime.EMode.eFrames24)
                    worlds.append(_amatrix(node.EvaluateGlobalTransform(when)))
                worlds = np.array(worlds)
            attribute = node.GetNodeAttribute()
            props     = {}
            if attribute is not None and attribute.GetAttributeType() == FBX.FbxNodeAttribute.EType.eCamera:
                camera = typed(attribute, FBX.FbxCamera, FBX)
                props = {
                    "FocalLength":  float(camera.FocalLength.Get()),
                    "FilmWidth":    float(camera.FilmWidth.Get()),
                    "FilmHeight":   float(camera.FilmHeight.Get()),
                    "GateFit":      int(getattr(camera.GateFit.Get(), "value", camera.GateFit.Get())),
                    "ApertureMode": int(getattr(camera.ApertureMode.Get(), "value", camera.ApertureMode.Get())),
                    "Projection":   int(getattr(camera.ProjectionType.Get(), "value", camera.ProjectionType.Get())),
                    "OrthoZoom":    float(camera.OrthoZoom.Get()),
                    "NearPlane":    float(camera.NearPlane.Get()),
                    "FarPlane":     float(camera.FarPlane.Get()),
                    "Interest":     tuple(camera.InterestPosition.Get())[:3],
                    "Up":           tuple(camera.UpVector.Get())[:3],
                }
            if attribute is not None and attribute.GetAttributeType() == FBX.FbxNodeAttribute.EType.eLight:
                light = typed(attribute, FBX.FbxLight, FBX)
                props = {
                    "LightType":  int(getattr(light.LightType.Get(), "value", light.LightType.Get())),
                    "Intensity":  float(light.Intensity.Get()),
                    "DecayType":  int(getattr(light.DecayType.Get(), "value", light.DecayType.Get())),
                    "InnerAngle": float(light.InnerAngle.Get()),
                    "OuterAngle": float(light.OuterAngle.Get()),
                    "Color":      tuple(light.Color.Get())[:3],
                }
            out[str(node.GetName())] = (tuple(node.GetPostRotation(source))[:3], worlds, props)
    return out


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestViews(_Files):
    """cameras and lights on their nodes, saved and read back"""

    def test_round_trip_in_a_rig(self):
        rig     = views_rig()
        cameras = CameraList([shot(rig), ortho(rig)])
        lights  = views_lights(rig)
        save_fbx(self.path("a.fbx"), rig, cameras=cameras, lights=lights)
        back = load_fbx(self.path("a.fbx"))
        self.assertSameSkeleton(back.hierarchy, rig)
        same_views(self, back.cameras, cameras)
        same_views(self, back.lights, lights)
        self.assertEqual([back.hierarchy[c.parent_uuid].name for c in back.cameras], ["shotCam", "orthoCam"])
        self.assertEqual([back.hierarchy[x.parent_uuid].name for x in back.lights], ["key", "sun", "fill"])
        self.assertEqual([x.kind for x in back.lights], ["spot", "directional", "point"])
        self.assertEqual((back.clip, back.meshes, back.skins, back.uvs, back.materials), (None, None, None, None, None))
        # the nodes are Maya's transforms, no rotate axis, their worlds as before
        np.testing.assert_array_equal(np.asarray(back.hierarchy.rotate_axis), 0.0)
        np.testing.assert_allclose(np.asarray(back.hierarchy.world_matrix), np.asarray(rig.world_matrix), rtol=0, atol=1e-12)
        # a second cycle, bit for bit
        save_fbx(self.path("b.fbx"), *back)
        again = load_fbx(self.path("b.fbx"))
        self.assertSameSkeleton(again.hierarchy, back.hierarchy)
        same_views(self, again.cameras, back.cameras)
        same_views(self, again.lights, back.lights)
        # the class routes read the nodes alone
        self.assertEqual(HierarchyData.load_fbx(self.path("a.fbx")).name, rig.name)

    def test_the_file_is_what_maya_writes(self):
        """the node turns and the properties Maya 2025's exporter writes
        (measured): post rotation (0, -90, 0) on a camera's node, (-90, 0, 0)
        on a spot's and a directional light's, none on a point light's; the
        focal length and film back as they are in focal length mode, the
        gate fit, ortho zoom = width / 30, intensity x 100, inner and outer
        angles; the camera's view down the file's +X, a light's down -Y"""
        rig     = views_rig()
        cameras = CameraList([shot(rig), ortho(rig)])
        lights  = views_lights(rig)
        save_fbx(self.path("a.fbx"), rig, cameras=cameras, lights=lights)
        found  = fbx_nodes(self.path("a.fbx"))
        worlds = {node.name: np.asarray(node.world_matrix) for node in rig}
        post, world, props = found["shotCam"]
        np.testing.assert_allclose(post, (0.0, -90.0, 0.0), rtol=0, atol=1e-9)
        self.assertEqual(props["FocalLength"], 50.0)
        self.assertEqual((props["FilmWidth"], props["FilmHeight"]), (1.0, 0.6))
        self.assertEqual(props["GateFit"], _enum(FBX.FbxCamera, "EGateFit", "eFitVertical"))
        self.assertEqual(props["ApertureMode"], _enum(FBX.FbxCamera, "EApertureMode", "eFocalLength"))
        self.assertEqual(props["Projection"], _enum(FBX.FbxCamera, "EProjectionType", "ePerspective"))
        self.assertEqual((props["NearPlane"], props["FarPlane"]), (1.0, 5000.0))
        # fbx's camera looks down its node's +X: our -Z; its up is our +Y
        np.testing.assert_allclose(world[0, 0, :3], -worlds["shotCam"][2, :3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(world[0, 1, :3], worlds["shotCam"][1, :3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(world[0, 3, :3], worlds["shotCam"][3, :3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(props["Interest"], worlds["shotCam"][3, :3] - 5.0 * worlds["shotCam"][2, :3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(props["Up"], worlds["shotCam"][1, :3], rtol=0, atol=1e-9)
        post, world, props = found["orthoCam"]
        self.assertEqual(props["Projection"], _enum(FBX.FbxCamera, "EProjectionType", "eOrthogonal"))
        self.assertAlmostEqual(props["OrthoZoom"], 50.0 / 30.0, places=12)
        for name, turned, kind, intensity, decay in (
            ("key", True, "eSpot", 250.0, "eQuadratic"), ("sun", True, "eDirectional", 50.0, "eNone"), ("fill", False, "ePoint", 100.0, "eLinear"),
        ):
            with self.subTest(name):
                post, world, props = found[name]
                np.testing.assert_allclose(post, (-90.0, 0.0, 0.0) if turned else (0.0, 0.0, 0.0), rtol=0, atol=1e-9)
                self.assertEqual(props["LightType"], _enum(FBX.FbxLight, "EType", kind))
                self.assertEqual(props["Intensity"], intensity)
                self.assertEqual(props["DecayType"], _enum(FBX.FbxLight, "EDecayType", decay))
                if turned:
                    # fbx's light shines down its node's -Y: our -Z
                    np.testing.assert_allclose(-world[0, 1, :3], -worlds[name][2, :3], rtol=0, atol=1e-9)
                np.testing.assert_allclose(world[0, 3, :3], worlds[name][3, :3], rtol=0, atol=1e-9)
        self.assertEqual((found["key"][2]["InnerAngle"], found["key"][2]["OuterAngle"]), (50.0, 60.0))
        np.testing.assert_allclose(found["fill"][2]["Color"], (1.0, 0.9, 0.8), rtol=0, atol=1e-12)

    def test_round_trip_in_a_clip(self):
        """the camera's node keyed: its rotate curves come back as written,
        the camera on it; the lights on theirs"""
        rig     = views_rig()
        clip    = views_clip(rig)
        cameras = CameraList([shot(rig), ortho(rig)])
        lights  = views_lights(rig)
        save_fbx(self.path("a.fbx"), clip=clip, cameras=cameras, lights=lights)
        back = load_fbx(self.path("a.fbx"))
        self.assertIsNone(back.hierarchy)
        got = back.clip.active
        self.assertEqual(got.take, "pan")
        np.testing.assert_array_equal(np.asarray(got.frames.rotate), at_float32(clip.frames.rotate))
        np.testing.assert_array_equal(np.asarray(got.frames.rotate_axis), 0.0)
        same_views(self, back.cameras, cameras)
        same_views(self, back.lights, lights)
        self.assertEqual([got[c.parent_uuid].name for c in back.cameras], ["shotCam", "orthoCam"])
        self.assertEqual(counts(self.path("a.fbx"))["stacks"], 1)
        save_fbx(self.path("b.fbx"), *back)
        again = load_fbx(self.path("b.fbx"))
        np.testing.assert_array_equal(np.asarray(again.clip.active.frames.rotate), np.asarray(got.frames.rotate))
        same_views(self, again.cameras, back.cameras)

    def test_another_writers_file(self):
        """a camera given by its field of view on a node with no post
        rotation: the focal length the view gives, the node turned so the
        camera looks down our -Z where the file's looks down +X, on every
        frame of the keyed take; a directional light likewise, our -Z where
        the file's shines down -Y; keyed attributes read at the first frame
        with a warning; an area, a volume and an ambient light skipped with
        one"""
        path = other_writer(self.path("other.fbx"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = load_fbx(path)
        messages = [str(w.message) for w in caught]
        self.assertEqual(len([m for m in messages if "reads point, directional and spot lights" in m]),   1)
        self.assertEqual(len([m for m in messages if "camera 'lensShape' keys ['FocalLength']" in m]),    1)
        self.assertEqual(len([m for m in messages if "point light 'lampShape' keys ['Intensity']" in m]), 1)
        skipped = [m for m in messages if "reads point, directional and spot lights" in m][0]
        for name in ("area light 'areaShape'", "volume light 'volShape'", "ambient light 'ambShape'"):
            self.assertIn(name, skipped)

        clip = back.clip.active
        self.assertEqual(clip.name, ["cam", "lens", "sun", "lamp", "area", "vol", "amb"])
        self.assertEqual([node.node_type for node in clip], ["transform"] * 7)
        cameras = {c.name: c for c in back.cameras}
        self.assertAlmostEqual(cameras["camShape"].focal_length, 1.0 * 25.4 / (2.0 * math.tan(math.radians(30.0))), places=9)
        self.assertEqual((cameras["camShape"].horizontal_film_aperture, cameras["camShape"].vertical_film_aperture), (1.0, 0.5))
        self.assertEqual(cameras["lensShape"].focal_length, 35.0)  # the first frame's
        lights = {x.name: x for x in back.lights}
        self.assertEqual(sorted(lights), ["lampShape", "sunShape"])
        self.assertEqual(lights["lampShape"].intensity, 1.0)  # 100 at the first frame

        # the file's views, frame by frame
        found = fbx_nodes(path, frames=range(FRAMES))
        ours  = np.asarray(clip.frames.world_matrix)
        cam   = clip.name.index("cam")
        sun   = clip.name.index("sun")
        np.testing.assert_allclose(-ours[:, cam, 2, :3], found["cam"][1][:, 0, :3],  rtol=0, atol=1e-6)  # our -Z, its +X
        np.testing.assert_allclose(ours[:, cam, 1, :3],  found["cam"][1][:, 1, :3],  rtol=0, atol=1e-6)  # our +Y, its +Y
        np.testing.assert_allclose(ours[:, cam, 3, :3],  found["cam"][1][:, 3, :3],  rtol=0, atol=1e-6)
        np.testing.assert_allclose(-ours[:, sun, 2, :3], -found["sun"][1][:, 1, :3], rtol=0, atol=1e-9)  # our -Z, its -Y
        np.testing.assert_array_equal(np.asarray(clip.frames.rotate_axis), 0.0)
        self.assertTrue(clip.keyed[cam])
        # the turned rotate is continuous along the frames: no branch flip
        # (a flip steps by 180 or 360); the composed eulers step faster than
        # the file's 22.5 a frame where the composition bends them
        self.assertLess(float(np.abs(np.diff(np.asarray(clip.frames.rotate)[:, cam], axis=0)).max()), 180.0)

    def test_refusals_and_warnings(self):
        rig = views_rig()

        def refused(message, **kwargs):
            with self.assertRaisesRegex(ValueError, message):
                save_fbx(self.path("refused.fbx"), rig, **kwargs)
            self.assertFalse(os.path.exists(self.path("refused.fbx")))

        refused("camera 'shotShape' sits on joint 'jnt'", cameras=CameraData(name="shotShape", parent_uuid=rig["jnt"].uuid))
        refused("spot light 'keyShape' sits on joint 'jnt'", lights=LightData(name="keyShape", kind="spot", parent_uuid=rig["jnt"].uuid))
        two = CameraList([shot(rig), CameraData(name="again", parent_uuid=rig["shotCam"].uuid)])
        refused("camera 'shotShape' and camera 'again' sit on node 'shotCam'; a node holds one shape", cameras=two)
        refused("camera 'shotShape' and the unnamed point light on node 'shotCam' sit on node 'shotCam'",
                cameras=shot(rig), lights=LightData(parent_uuid=rig["shotCam"].uuid))
        mesh = MeshData(indices=[0, 1, 2], counts=[3], points=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], name="tri", parent_uuid=rig["shotCam"].uuid)
        refused("mesh 'tri' and camera 'shotShape' sit on node 'shotCam'", meshes=mesh, cameras=shot(rig))
        refused("camera 'shotShape' sits on node 'grp', which has children", cameras=CameraData(name="shotShape", parent_uuid=rig["grp"].uuid))
        turned                    = views_rig()
        turned["key"].rotate_axis = (0.0, 5.0, 0.0)
        with self.assertRaisesRegex(ValueError, "spot light 'keyShape' sits on node 'key', which holds a rotate axis"):
            save_fbx(self.path("refused.fbx"), turned, lights=views_lights(turned)[0])
        with self.assertRaisesRegex(TypeError, "the cameras are a CameraList or a CameraData, not a LightData"):
            save_fbx(self.path("refused.fbx"), rig, cameras=LightData())
        with self.assertRaisesRegex(TypeError, "the lights are a LightList or a LightData, not a list"):
            save_fbx(self.path("refused.fbx"), rig, lights=[LightData()])

        # what fbx cannot hold: one warning each, the file written without it
        scaled              = shot(rig)
        scaled.camera_scale = 2.0
        drops               = views_lights(rig)
        drops[0].dropoff    = 3.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_fbx(self.path("warned.fbx"), rig, cameras=scaled, lights=drops)
        messages = [str(w.message) for w in caught]
        self.assertEqual(len([m for m in messages if "camera 'shotShape': fbx holds no camera scale" in m]), 1)
        self.assertEqual(len([m for m in messages if "spot light 'keyShape': fbx holds no dropoff" in m]), 1)
        back = load_fbx(self.path("warned.fbx"))
        self.assertEqual((back.cameras[0].camera_scale, back.lights[0].dropoff), (1.0, 0.0))

    def test_a_shape_naming_no_node_goes_on_a_root(self):
        camera = CameraData(name="free")
        light  = LightData(name="lamp", kind="directional")
        save_fbx(self.path("a.fbx"), cameras=camera, lights=light)
        back = load_fbx(self.path("a.fbx"))
        self.assertEqual(back.hierarchy.name, ["free", "lamp"])
        np.testing.assert_array_equal(np.asarray(back.hierarchy.world_matrix), np.broadcast_to(np.eye(4), (2, 4, 4)))
        self.assertEqual([back.hierarchy[c.parent_uuid].name for c in back.cameras], ["free"])
        self.assertEqual([back.hierarchy[x.parent_uuid].name for x in back.lights], ["lamp"])
        same_views(self, back.cameras, CameraList([camera]))
        same_views(self, back.lights, LightList([light]))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaViews(_InMaya):
    """Maya 2025's export of its cameras and lights read as Maya shows them,
    and Maya's import of ours"""

    CAMERA_ATTRS = ("focalLength", "horizontalFilmAperture", "verticalFilmAperture", "horizontalFilmOffset",
                    "verticalFilmOffset", "lensSqueezeRatio", "nearClipPlane", "farClipPlane", "orthographicWidth")

    def exported_views(self, name: str) -> str:
        cmds, mel = self.cmds, self.mel
        mel.eval("FBXResetExport")
        mel.eval("FBXExportConvertUnitString cm")
        mel.eval("FBXExportUpAxis y")
        mel.eval("FBXExportCameras -v true")
        mel.eval("FBXExportLights -v true")
        out = self.path(name).replace(os.sep, "/")
        mel.eval(f'FBXExport -f "{out}"')
        return out

    def maya_scene(self) -> dict:
        """the probe's scene: a tuned camera, an orthographic one, a fresh one
        under a turned group, every light kind turned and moved; {transform:
        shape}"""
        cmds = self.cmds
        cmds.file(new=True, force=True)
        cmds.currentUnit(time="film", linear="cm", angle="deg")
        made = {}
        tuned, shape = cmds.camera(name="tunedCam")
        for attr, value in (("focalLength", 50.0), ("horizontalFilmAperture", 1.0), ("verticalFilmAperture", 0.6),
                            ("filmFit", 2), ("horizontalFilmOffset", 0.1), ("verticalFilmOffset", -0.05),
                            ("lensSqueezeRatio", 1.5), ("cameraScale", 2.0), ("nearClipPlane", 1.0), ("farClipPlane", 5000.0)):
            cmds.setAttr(f"{shape}.{attr}", value)
        cmds.setAttr(f"{tuned}.translate", 10.0, 20.0, 30.0)
        cmds.setAttr(f"{tuned}.rotate", 10.0, 20.0, 30.0)
        made[tuned] = shape
        ortho, shape = cmds.camera(name="orthoCam")
        cmds.setAttr(f"{shape}.orthographic", True)
        cmds.setAttr(f"{shape}.orthographicWidth", 50.0)
        cmds.setAttr(f"{ortho}.rotate", -90.0, 0.0, 0.0)
        made[ortho] = shape
        plain, shape = cmds.camera(name="plainCam")
        group = cmds.group(empty=True, name="camGroup")
        cmds.setAttr(f"{group}.rotate", 0.0, 45.0, 0.0)
        cmds.parent(plain, group)
        made[plain] = shape
        for kind, make in (("point", cmds.pointLight), ("directional", cmds.directionalLight), ("spot", cmds.spotLight),
                           ("ambient", cmds.ambientLight)):
            node = make(name=f"{kind}LightShape")
            if cmds.nodeType(node) != "transform":
                node = cmds.listRelatives(node, parent=True)[0]
            node  = cmds.rename(node, f"{kind}Light")
            shape = cmds.listRelatives(node, shapes=True)[0]
            cmds.setAttr(f"{shape}.color", 0.25, 0.5, 1.0)
            cmds.setAttr(f"{shape}.intensity", 2.5)
            if kind in ("point", "spot"):
                cmds.setAttr(f"{shape}.decayRate", 2)
            if kind == "spot":
                cmds.setAttr(f"{shape}.coneAngle",     50.0)
                cmds.setAttr(f"{shape}.penumbraAngle", 5.0)
                cmds.setAttr(f"{shape}.dropoff",       3.0)
            cmds.setAttr(f"{node}.translate", 1.0, 2.0, 3.0)
            cmds.setAttr(f"{node}.rotate", 10.0, 20.0, 30.0)
            made[node] = shape
        for kind in ("areaLight", "volumeLight"):
            # shadingNode hands back the transform or the shape, by kind
            node = cmds.shadingNode(kind, asLight=True, name=f"{kind}Shape")
            if cmds.nodeType(node) != "transform":
                node = cmds.listRelatives(node, parent=True)[0]
            node = cmds.rename(node, kind)
            cmds.setAttr(f"{node}.rotate", 10.0, 20.0, 30.0)
            made[node] = cmds.listRelatives(node, shapes=True)[0]
        return made

    def test_mayas_export_reads_as_maya_shows_it(self):
        """every camera attribute as Maya holds it (the camera scale apart,
        which Maya's export drops: it reads 1), every light's, the kinds, the
        node worlds within 1e-6, the ambient, area and volume lights
        skipped with one warning"""
        cmds = self.cmds
        made = self.maya_scene()
        out  = self.exported_views("maya.fbx")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = load_fbx(out)
        self.assertEqual(len([w for w in caught if "reads point, directional and spot lights" in str(w.message)]), 1)
        rig = back.hierarchy if back.hierarchy is not None else back.clip.active
        for node, shape in made.items():
            with self.subTest(node):
                self.assertEqual(rig[node].node_type, "transform")
                world = np.array(cmds.xform(node, q=True, ws=True, m=True)).reshape(4, 4)
                np.testing.assert_allclose(np.asarray(rig[node].world_matrix), world, rtol=0, atol=1e-6)
        cameras = {rig[c.parent_uuid].name: c for c in back.cameras}
        self.assertEqual(sorted(cameras), ["orthoCam1", "plainCam1", "tunedCam1"])
        for node, camera in cameras.items():
            shape = made[node]
            with self.subTest(node):
                for attr, field in (("focalLength", "focal_length"), ("horizontalFilmAperture", "horizontal_film_aperture"),
                                    ("verticalFilmAperture", "vertical_film_aperture"), ("lensSqueezeRatio", "lens_squeeze_ratio"),
                                    ("nearClipPlane", "near_clip_plane"), ("farClipPlane", "far_clip_plane"),
                                    ("orthographicWidth", "orthographic_width")):
                    self.assertAlmostEqual(getattr(camera, field), cmds.getAttr(f"{shape}.{attr}"), places=9, msg=attr)
                self.assertEqual(camera.film_fit, ("fill", "horizontal", "vertical", "overscan")[cmds.getAttr(f"{shape}.filmFit")])
                np.testing.assert_allclose(camera.film_offset, [cmds.getAttr(f"{shape}.horizontalFilmOffset"), cmds.getAttr(f"{shape}.verticalFilmOffset")], rtol=0, atol=1e-12)
                self.assertEqual(camera.orthographic, bool(cmds.getAttr(f"{shape}.orthographic")))
                self.assertEqual(camera.camera_scale, 1.0)
        self.assertEqual(cmds.getAttr(f"{made['tunedCam1']}.cameraScale"), 2.0)  # what the file lost
        lights = {rig[x.parent_uuid].name: x for x in back.lights}
        self.assertEqual(sorted(lights), ["directionalLight", "pointLight", "spotLight"])
        for node, light in lights.items():
            shape = made[node]
            with self.subTest(node):
                self.assertEqual(light.kind, cmds.nodeType(shape)[: -len("Light")])
                np.testing.assert_allclose(light.color, cmds.getAttr(f"{shape}.color")[0], rtol=0, atol=1e-12)
                self.assertAlmostEqual(light.intensity, cmds.getAttr(f"{shape}.intensity"), places=9)
                self.assertEqual(light.decay_rate, cmds.getAttr(f"{shape}.decayRate"))
                if light.kind == "spot":
                    self.assertAlmostEqual(light.cone_angle, cmds.getAttr(f"{shape}.coneAngle"), places=9)
                    self.assertAlmostEqual(light.penumbra_angle, cmds.getAttr(f"{shape}.penumbraAngle"), places=9)
                    self.assertEqual(light.dropoff, 0.0)  # what the file lost
        self.assertEqual(cmds.getAttr(f"{made['spotLight']}.dropoff"), 3.0)

    def test_maya_imports_our_save_as_we_hold_it(self):
        """our cameras and lights, imported: Maya's camera and light shapes
        with our attributes, on transforms at our worlds within 1e-6"""
        cmds    = self.cmds
        rig     = views_rig()
        cameras = CameraList([shot(rig), ortho(rig)])
        lights  = views_lights(rig)
        save_fbx(self.path("a.fbx"), rig, cameras=cameras, lights=lights)
        self.imported(self.path("a.fbx"))
        worlds = {node.name: np.asarray(node.world_matrix) for node in rig}
        for node in rig:
            with self.subTest(node.name):
                world = np.array(cmds.xform(node.name, q=True, ws=True, m=True)).reshape(4, 4)
                np.testing.assert_allclose(world, worlds[node.name], rtol=0, atol=1e-6)
        for camera in cameras:
            node  = rig[camera.parent_uuid].name
            shape = cmds.listRelatives(node, shapes=True, type="camera")
            with self.subTest(node):
                self.assertEqual(len(shape or []), 1)
                shape = shape[0]
                for attr, field in (("focalLength", "focal_length"), ("horizontalFilmAperture", "horizontal_film_aperture"),
                                    ("verticalFilmAperture", "vertical_film_aperture"), ("lensSqueezeRatio", "lens_squeeze_ratio"),
                                    ("nearClipPlane", "near_clip_plane"), ("farClipPlane", "far_clip_plane")):
                    self.assertAlmostEqual(cmds.getAttr(f"{shape}.{attr}"), getattr(camera, field), places=6, msg=attr)
                self.assertEqual(cmds.getAttr(f"{shape}.filmFit"), ("fill", "horizontal", "vertical", "overscan").index(camera.film_fit))
                self.assertAlmostEqual(cmds.getAttr(f"{shape}.horizontalFilmOffset"), camera.film_offset[0], places=9)
                self.assertAlmostEqual(cmds.getAttr(f"{shape}.verticalFilmOffset"), camera.film_offset[1], places=9)
                self.assertEqual(bool(cmds.getAttr(f"{shape}.orthographic")), camera.orthographic)
                if camera.orthographic:
                    self.assertAlmostEqual(cmds.getAttr(f"{shape}.orthographicWidth"), camera.orthographic_width, places=6)
        for light in lights:
            node  = rig[light.parent_uuid].name
            shape = cmds.listRelatives(node, shapes=True)
            with self.subTest(node):
                self.assertEqual(len(shape or []), 1)
                shape = shape[0]
                self.assertEqual(cmds.nodeType(shape), f"{light.kind}Light")
                np.testing.assert_allclose(cmds.getAttr(f"{shape}.color")[0], light.color, rtol=0, atol=1e-6)
                self.assertAlmostEqual(cmds.getAttr(f"{shape}.intensity"), light.intensity, places=6)
                self.assertEqual(cmds.getAttr(f"{shape}.decayRate"), light.decay_rate)
                if light.kind == "spot":
                    self.assertAlmostEqual(cmds.getAttr(f"{shape}.coneAngle"), light.cone_angle, places=6)
                    self.assertAlmostEqual(cmds.getAttr(f"{shape}.penumbraAngle"), light.penumbra_angle, places=6)

    def test_maya_reads_another_writers_camera_as_we_do(self):
        """a camera and a directional light on nodes with no post rotation,
        as another writer leaves them: Maya's import puts their transforms
        where our turn puts them, within 1e-6, on every frame"""
        cmds = self.cmds
        path = other_writer(self.path("other.fbx"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = load_fbx(path).clip.active
        self.imported(path)
        cmds.currentUnit(time="film")
        ours = np.asarray(clip.frames.world_matrix)
        for name in ("cam", "sun"):
            column = clip.name.index(name)
            for f in range(FRAMES):
                world = np.array(cmds.getAttr(f"{name}.worldMatrix", time=f)).reshape(4, 4)
                np.testing.assert_allclose(world, ours[f, column], rtol=0, atol=1e-6, err_msg=(name, f))

    def test_angle_of_view_is_mayas(self):
        """CameraData's angle of view against Maya's own query on the tuned
        camera: squeeze 1.5, scale 2, 50 mm on a 1 by 0.6 inch back"""
        cmds  = self.cmds
        made  = self.maya_scene()
        shape = made["tunedCam1"]
        # Maya's query is within 1e-4 of a degree of 2 atan(aperture / 2 focal):
        # its own inch holds a few more digits of rounding (measured 5.5e-5
        # on 28.5035 degrees); neither the camera scale nor the squeeze shows
        camera = CameraData(focal_length=50.0, horizontal_film_aperture=1.0, vertical_film_aperture=0.6, lens_squeeze_ratio=1.5, camera_scale=2.0)
        self.assertAlmostEqual(camera.horizontal_angle_of_view, cmds.camera(shape, q=True, horizontalFieldOfView=True), delta=1e-3)
        self.assertAlmostEqual(camera.vertical_angle_of_view, cmds.camera(shape, q=True, verticalFieldOfView=True), delta=1e-3)
        plain = CameraData()
        self.assertAlmostEqual(plain.horizontal_angle_of_view, cmds.camera(made["plainCam1"], q=True, horizontalFieldOfView=True), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
