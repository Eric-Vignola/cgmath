"""The r17-skeleton fork reads every FBX in Maya's Y-up and cm, the axes and
unit every cgmath value is in, and writes every file labelled so.

A file whose up axis is X or Z is turned as Maya's importer turns it: the
truth is the sdk's own evaluation after ``FbxAxisSystem.MayaYUp.ConvertScene``,
the conversion Maya's FBX importer runs on such a file (measured,
``hierarchy.hierarchy._fbx_axis_turn``). A file whose up axis is Y, up or
down, whatever its front, is left in its own axes, as Maya's importer leaves
it. A left handed file is refused. The file's unit is converted to cm. The
fixtures are authored through the sdk, in the axis system and unit each test
names; the Maya tests import them with FBXImport.
"""

import itertools
import unittest

import numpy as np
from cgmath._tests.test_skeleton_fbx import START, TAKE, UNIT, _Files, _Recorded, body, bound, motion, near, weighed
from cgmath._tests.maya_fixtures import same_read
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, generate_uuid
from cgmath.hierarchy.hierarchy import _fbx_scene_axes

FBX = load_sdk()

FRAMES = 9
FPS    = 24.0

# the Z-up to Maya's Y-up turn, (x, y, z) to (x, z, -y): what a Z-up file's
# roots take, as Maya's importer puts it on them (measured)
Z_TO_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def axes(up: str, parity: str, handed: str = "eRightHanded"):
    system = FBX.FbxAxisSystem
    return system(
        getattr(system.EUpVector, up),
        getattr(system.EFrontVector, parity),
        getattr(system.ECoordSystem, handed),
    )


def m4(matrix) -> np.ndarray:
    return np.array([[matrix.Get(r, c) for c in range(4)] for r in range(4)])


def authored(path: str, system, unit=None, animated: bool = True, flip=()) -> str:
    """a joint chain under a joint root with a pre rotation, a group root in
    zxy scaled unevenly (a leaf) and a locator root in zyx holding a joint,
    keyed over FRAMES frames at 24 fps, in axis system ``system`` and unit
    ``unit`` (an FbxSystemUnit name, cm for None). ``flip`` names global
    settings signs to negate after the system is set, such as "UpAxisSign":
    the binding builds no axis system pointing down. The values are the
    same whatever the axes and unit: the file's label alone says where
    they are"""
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    scene = FBX.FbxScene.Create(manager, "scene")
    scene.GetGlobalSettings().SetAxisSystem(system)
    for sign in flip:
        stored = FBX.FbxPropertyInteger1(scene.GetGlobalSettings().FindProperty(sign))
        stored.Set(-stored.Get())
    if unit is not None:
        scene.GetGlobalSettings().SetSystemUnit(getattr(FBX.FbxSystemUnit, unit))
    scene.GetGlobalSettings().SetTimeMode(FBX.FbxTime.EMode.eFrames24)
    source = FBX.FbxNode.EPivotSet.eSourcePivot

    def null(name, look):
        attr = FBX.FbxNull.Create(scene, name)
        attr.Look.Set(look.value)
        return attr

    def node(name, attr, parent, translate, rotate, order=0, pre=None):
        made = FBX.FbxNode.Create(scene, name)
        made.SetNodeAttribute(attr)
        made.LclTranslation.Set(FBX.FbxDouble3(*translate))
        made.LclRotation.Set(FBX.FbxDouble3(*rotate))
        made.SetRotationActive(True)
        made.SetRotationOrder(source, list(FBX.EFbxRotationOrder)[order])
        if pre is not None:
            made.SetPreRotation(source, FBX.FbxVector4(*pre, 1.0))
        parent.AddChild(made)
        return made

    top   = scene.GetRootNode()
    bone  = FBX.FbxSkeleton.Create
    hip   = node("hip", bone(scene, "a"), top, (1.0, 2.0, 3.0), (10.0, 20.0, 30.0), pre=(5.0, -15.0, 25.0))
    knee  = node("knee", bone(scene, "b"), hip, (0.0, -4.0, 0.5), (-30.0, 5.0, 0.0))
    group = node("group", null("c", FBX.FbxNull.ELook.eNone), top, (-2.0, 1.0, 4.0), (35.0, -10.0, 50.0), 4)
    loc   = node("loc", null("e", FBX.FbxNull.ELook.eCross), top, (3.0, -1.0, 2.0), (0.0, 80.0, 0.0), 5)
    node("prop", bone(scene, "d"), loc, (0.5, 0.5, 0.0), (0.0, 0.0, 15.0))
    group.LclScaling.Set(FBX.FbxDouble3(1.0, 2.0, 0.5))

    if animated:
        stack = FBX.FbxAnimStack.Create(scene, "Take 001")
        layer = FBX.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)
        last   = FRAMES - 1
        linear = FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
        keys = (
            (hip.LclTranslation, "X", 1.0, 7.0),
            (hip.LclTranslation, "Y", 2.0, 102.0),
            (hip.LclTranslation, "Z", 3.0, 53.0),
            (hip.LclRotation, "X", 10.0, 70.0),
            (knee.LclRotation, "Z", 0.0, -45.0),
            (group.LclTranslation, "X", -2.0, 6.0),
            (group.LclRotation, "Z", 50.0, 450.0),  # 50 degrees a frame
            (group.LclRotation, "X", 35.0, 75.0),
            (loc.LclRotation, "Y", 80.0, 100.0),
        )
        for prop_of, axis, first, end in keys:
            curve = prop_of.GetCurve(layer, axis, True)
            curve.KeyModifyBegin()
            for frame in range(FRAMES):
                time = FBX.FbxTime()
                time.SetSecondDouble(frame / FPS)
                key = curve.KeyAdd(time)[0]
                curve.KeySetValue(key, first + (end - first) * frame / last)
                curve.KeySetInterpolation(key, linear)
            curve.KeyModifyEnd()
        first, last = FBX.FbxTime(), FBX.FbxTime()
        first.SetSecondDouble(0.0)
        last.SetSecondDouble((FRAMES - 1) / FPS)
        stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))

    exporter = FBX.FbxExporter.Create(manager, "")
    if not exporter.Initialize(path, -1, manager.GetIOSettings()):
        raise RuntimeError(f"cannot write {path}")
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
    return path


def turned_rig() -> HierarchyData:
    """the roots a turn lands on, as save_fbx writes them (limb node joints,
    no pre rotation on a transform): a joint root with a joint orient and a
    joint under it, a group root in zxy scaled unevenly (a leaf) and a group
    root in zyx holding a joint"""
    hips, knee, group, loc, prop = (generate_uuid() for _ in range(5))
    return HierarchyData(
        [
            TransformData("hips", node_type="joint", uuid=hips, translate=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), joint_orient=(5.0, -15.0, 25.0)),
            TransformData("knee", node_type="joint", uuid=knee, parent_uuid=hips, translate=(0.0, -4.0, 0.5), rotate=(-30.0, 5.0, 0.0), joint_orient=(0.0, 0.0, -90.0)),
            TransformData("group", uuid=group, translate=(-2.0, 1.0, 4.0), rotate=(35.0, -10.0, 50.0), scale=(1.0, 2.0, 0.5), rotate_order=4),
            TransformData("loc", uuid=loc, translate=(3.0, -1.0, 2.0), rotate=(0.0, 80.0, 0.0), rotate_order=5),
            TransformData("prop", node_type="joint", uuid=prop, parent_uuid=loc, translate=(0.5, 0.5, 0.0), rotate=(0.0, 0.0, 15.0)),
        ]
    )


def turned_clip(rig: HierarchyData) -> ClipData:
    """FRAMES frames from 0 at 30 fps (the Maya scene's rate: Maya solves a
    turned group root's rotate again at the scene's rate) over turned_rig():
    the joint root travels and turns about x, the group root travels and
    turns 50 degrees a frame about z, the locator root turns about y"""
    clip                           = ClipData(rig, frames=FRAMES, start_frame=0, fps=30.0, take="turn")
    f                              = np.linspace(0.0, 1.0, FRAMES)[:, None]
    clip.frames["hips"].translate  = np.hstack([1.0 + f * 6.0, 2.0 + f * 100.0, 3.0 + f * 50.0])
    clip.frames["hips"].rotate     = np.hstack([10.0 + f * 60.0, 20.0 + f * 0.0, 30.0 + f * 0.0])
    clip.frames["knee"].rotate     = np.hstack([-30.0 + f * 0.0, 5.0 + f * 0.0, f * -45.0])
    clip.frames["group"].translate = np.hstack([-2.0 + f * 8.0, 1.0 + f * 0.0, 4.0 + f * 0.0])
    clip.frames["group"].rotate    = np.hstack([35.0 + f * 40.0, -10.0 + f * 0.0, 50.0 + f * 400.0])
    clip.frames["loc"].rotate      = np.hstack([f * 0.0, 80.0 + f * 20.0, f * 0.0])
    clip.frame                     = 0
    return clip


def truth(path: str, convert: bool, cm: float = 1.0) -> tuple:
    """({name: world}, {name: (F, 4, 4)}): every node's world matrix as the
    sdk evaluates the file, after ConvertScene to Maya's Y-up when
    ``convert``, the translations times ``cm``"""
    manager = FBX.FbxManager.Create()
    try:
        manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
        importer = FBX.FbxImporter.Create(manager, "")
        importer.Initialize(path, -1, manager.GetIOSettings())
        scene = FBX.FbxScene.Create(manager, "")
        importer.Import(scene)
        importer.Destroy()
        if convert:
            FBX.FbxAxisSystem.MayaYUp.ConvertScene(scene)

        nodes, stack = {}, [scene.GetRootNode()]
        while stack:
            node = stack.pop()
            if node.GetNodeAttribute():
                nodes[str(node.GetName())] = node
            stack.extend(node.GetChild(i) for i in range(node.GetChildCount()))

        still, moving = {}, {}
        for name, node in nodes.items():
            world = m4(node.EvaluateGlobalTransform())
            world[3, :3] *= cm
            still[name] = world
            worlds      = []
            for frame in range(FRAMES):
                time = FBX.FbxTime()
                time.SetSecondDouble(frame / FPS)
                world = m4(node.EvaluateGlobalTransform(time))
                world[3, :3] *= cm
                worlds.append(world)
            moving[name] = np.array(worlds)
        return still, moving
    finally:
        manager.Destroy()


def relabelled(source: str, path: str, system, unit: str) -> str:
    """``source`` read with the sdk, its label alone changed to ``system`` and
    ``unit`` (an FbxSystemUnit name), the values left as they are, written
    to ``path``: a file that says its numbers are in those axes and that
    unit"""
    from cgmath.formats._fbx_io import import_scene, new_manager

    manager = new_manager(FBX)
    try:
        scene = import_scene(manager, source, FBX)
        scene.GetGlobalSettings().SetAxisSystem(system)
        scene.GetGlobalSettings().SetSystemUnit(getattr(FBX.FbxSystemUnit, unit))
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


def labelled(path: str) -> tuple:
    """(up axis, up sign, front axis, front sign, right handed, cm a unit) a file holds"""
    with open_fbx(path, FBX) as scene:
        return _fbx_scene_axes(scene, FBX) + (float(scene.GetGlobalSettings().GetSystemUnit().GetScaleFactor()),)


MAYA_Y_UP_CM = (1, 1, 2, 1, True, 1.0)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestAxes(_Files):
    """every file loads in Maya's Y-up: the worlds the sdk gives after
    ConvertScene for an X-up or Z-up file, the file's own for a Y-up one"""

    # the binding builds no axis pointing down: a sign is flipped on the
    # global settings, with the third axis's, which keeps the system right
    # handed (one sign alone would mirror it)
    def turned_systems(self) -> list:
        """(label, axis system, flips) of every right handed system whose up
        axis is x or z the binding builds, Unreal's Z-up among them, one
        pointing down and one whose front points the other way"""
        out = [("MayaZUp", FBX.FbxAxisSystem.MayaZUp, ())]
        for up, parity in itertools.product(("eXAxis", "eZAxis"), ("eParityEven", "eParityOdd")):
            out.append((f"{up} {parity}", axes(up, parity), ()))
        out.append(("-z up", axes("eZAxis", "eParityOdd"), ("UpAxisSign", "CoordAxisSign")))
        out.append(("x up, -front", axes("eXAxis", "eParityEven"), ("FrontAxisSign", "CoordAxisSign")))
        return out

    def y_up_systems(self) -> list:
        """(label, axis system, flips) of the Y-up systems: Maya's own, x
        front, -z front and -y up"""
        return [
            ("MayaYUp", FBX.FbxAxisSystem.MayaYUp, ()),
            ("y up, x front", axes("eYAxis", "eParityEven"), ()),
            ("y up, -z front", axes("eYAxis", "eParityOdd"), ("FrontAxisSign", "CoordAxisSign")),
            ("-y up", axes("eYAxis", "eParityOdd"), ("UpAxisSign", "CoordAxisSign")),
        ]

    def assertWorlds(self, got, want, atol=1e-9):
        self.assertEqual(sorted(got), sorted(want))
        for name in want:
            np.testing.assert_allclose(got[name], want[name], rtol=0, atol=atol, err_msg=name)

    @staticmethod
    def rig_worlds(rig) -> dict:
        return {node.name: np.asarray(node.world_matrix) for node in rig}

    @staticmethod
    def clip_worlds(clip) -> dict:
        stack = np.asarray(clip.frames.world_matrix)
        return {node.name: stack[:, i] for i, node in enumerate(clip)}

    def test_x_up_and_z_up_files_turn_to_maya_y_up(self):
        for label, system, flips in self.turned_systems():
            with self.subTest(label):
                path  = authored(self.path("axes.fbx"), system, flip=flips)
                still = HierarchyData.load_fbx(path)
                clip  = load_fbx(path).clip.active
                want_still, want_moving = truth(path, convert=True)
                self.assertWorlds(self.rig_worlds(still), want_still)
                self.assertWorlds(self.clip_worlds(clip), want_moving)
                # the turn sits on the roots: a joint root in its joint
                # orient, a group root in its rotate, solved again
                self.assertEqual([n.name for n in still if n.parent_name is None], ["hip", "group", "loc"])
                self.assertTrue(np.any(still["hip"].joint_orient != [5.0, -15.0, 25.0]))
                np.testing.assert_array_equal(still["knee"].rotate,    [-30.0, 5.0, 0.0])
                np.testing.assert_array_equal(still["knee"].translate, [0.0, -4.0, 0.5])
                np.testing.assert_array_equal(still["group"].scale,    [1.0, 2.0, 0.5])

    def test_a_group_root_rotate_is_solved_again_and_filtered(self):
        """its curve turns 50 degrees a frame: solved again frame by frame it
        would step by whole turns, so it goes through euler_filter, frame 0
        as solved; a joint root keeps its rotate curves"""
        from cgmath.transforms import euler_to_matrix, matrix_to_euler

        path   = authored(self.path("zup.fbx"), FBX.FbxAxisSystem.MayaZUp)
        clip   = load_fbx(path).clip.active
        rig    = HierarchyData.load_fbx(path)
        rotate = np.array(clip.frames["group"].rotate)
        self.assertLess(float(np.abs(np.diff(rotate, axis=0)).max()), 90.0)
        self.assertGreater(float(np.ptp(rotate, axis=0).max()), 300.0)
        order  = np.full(len(rotate), clip["group"].rotate_order)
        solved = np.degrees(matrix_to_euler(euler_to_matrix(np.radians(rotate), order), order))
        self.assertGreater(float(np.abs(rotate - solved).max()), 180.0)
        np.testing.assert_allclose(rotate[0], rig["group"].rotate, rtol=0, atol=1e-9)
        # hip, a joint root: its rotate curves as the file keys them, the
        # turn in its joint orient and translate
        hip = np.array(clip.frames["hip"].rotate)
        np.testing.assert_allclose(hip[:, 0], np.linspace(10.0, 70.0, FRAMES), rtol=0, atol=1e-12)
        np.testing.assert_allclose(hip[:, 1:], np.broadcast_to([20.0, 30.0], (FRAMES, 2)), rtol=0, atol=1e-12)
        translate = np.array(clip.frames["hip"].translate)
        file_side = np.stack([np.linspace(1.0, 7.0, FRAMES), np.linspace(2.0, 102.0, FRAMES), np.linspace(3.0, 53.0, FRAMES)], axis=1)
        np.testing.assert_allclose(translate, file_side @ Z_TO_Y, rtol=0, atol=1e-12)

    def test_y_up_files_keep_their_axes_whatever_their_front(self):
        for label, system, flips in self.y_up_systems():
            with self.subTest(label):
                path  = authored(self.path("yup.fbx"), system, flip=flips)
                still = HierarchyData.load_fbx(path)
                clip  = load_fbx(path).clip.active
                want_still, want_moving = truth(path, convert=False)
                self.assertWorlds(self.rig_worlds(still), want_still)
                self.assertWorlds(self.clip_worlds(clip), want_moving)
                np.testing.assert_array_equal(still["hip"].joint_orient, [5.0, -15.0, 25.0])
                np.testing.assert_array_equal(still["group"].rotate, [35.0, -10.0, 50.0])

    def test_a_left_handed_file_is_refused(self):
        path = authored(self.path("left.fbx"), axes("eYAxis", "eParityOdd", "eLeftHanded"))
        for loader in (load_fbx, HierarchyData.load_fbx):
            with self.assertRaisesRegex(ValueError, "left handed axis system, which no rotation turns into Maya's Y-up"):
                loader(path)

    def test_units_convert_to_cm(self):
        for unit, cm in (("m", 100.0), ("Inch", 2.54), ("mm", 0.1)):
            for label, system, convert in (("y up", FBX.FbxAxisSystem.MayaYUp, False), ("z up", FBX.FbxAxisSystem.MayaZUp, True)):
                with self.subTest(unit=unit, axes=label):
                    path  = authored(self.path("unit.fbx"), system, unit=unit)
                    still = HierarchyData.load_fbx(path)
                    clip  = load_fbx(path).clip.active
                    want_still, want_moving = truth(path, convert=convert, cm=cm)
                    self.assertWorlds(self.rig_worlds(still), want_still, atol=1e-9 * cm)
                    self.assertWorlds(self.clip_worlds(clip), want_moving, atol=1e-9 * cm)
                    # lengths times the unit, angles and scales as they are
                    np.testing.assert_allclose(still["knee"].translate, np.array([0.0, -4.0, 0.5]) * cm, rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(still["knee"].rotate, [-30.0, 5.0, 0.0])
                    np.testing.assert_array_equal(still["group"].scale, [1.0, 2.0, 0.5])
                    rows = np.array(clip.frames["knee"].translate)
                    np.testing.assert_allclose(rows, np.broadcast_to(np.array([0.0, -4.0, 0.5]) * cm, rows.shape), rtol=0, atol=1e-12)

    def test_a_save_is_labelled_maya_y_up_and_cm(self):
        rig = bound()
        save_fbx(self.path("rig.fbx"), rig)
        self.assertEqual(labelled(self.path("rig.fbx")), MAYA_Y_UP_CM)
        save_fbx(self.path("clip.fbx"), clip=motion(rig))
        self.assertEqual(labelled(self.path("clip.fbx")), MAYA_Y_UP_CM)
        save_fbx(self.path("mesh.fbx"), meshes=body())
        self.assertEqual(labelled(self.path("mesh.fbx")), MAYA_Y_UP_CM)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSkinnedZUp(_Files):
    """a skinned Z-up file in metres, as a game engine exports one: the mesh
    on a root at identity, the skeleton beside it; every root turned, the
    mesh's node among them, every length a hundred times its number, the
    bind pose and the clusters with them"""

    def skinned(self) -> tuple:
        """(the cm Y-up source save_fbx wrote, the same file relabelled Z-up
        and metres, save_fbx's arguments)"""
        rig    = bound()
        mesh   = body(rig["geo"].uuid)
        source = {"hierarchy": rig, "clip": motion(rig), "meshes": mesh, "skins": weighed(mesh)}
        save_fbx(self.path("y_cm.fbx"), **source)
        path = relabelled(self.path("y_cm.fbx"), self.path("z_m.fbx"), FBX.FbxAxisSystem.MayaZUp, "m")
        self.assertEqual(labelled(path)[:5], (2, 1, 1, -1, True))
        return self.path("y_cm.fbx"), path, source

    def test_the_read_turns_and_scales_everything(self):
        cm, path, source = self.skinned()
        rig           = source["hierarchy"]
        turn4         = np.eye(4)
        turn4[:3, :3] = Z_TO_Y
        got           = load_fbx(path)
        self.assertEqual([n.name for n in got.hierarchy], [n.name for n in rig])
        want = np.asarray(rig.world_matrix).copy()
        want[:, 3, :3] *= 100.0
        np.testing.assert_allclose(np.asarray(got.hierarchy.world_matrix), want @ turn4, rtol=0, atol=1e-9)
        # the take's keys are float32: the Y-up cm file's read, every length
        # a hundred times, turned, is the Z-up metres file's read
        base = load_fbx(cm)
        want = np.asarray(base.clip.active.frames.world_matrix).copy()
        want[..., 3, :3] *= 100.0
        np.testing.assert_allclose(np.asarray(got.clip.active.frames.world_matrix), want @ turn4, rtol=1e-12, atol=1e-9)
        # the mesh on geo, a root at identity in the file, turned: its
        # points in its node's space as they were, a hundred times
        self.assertEqual(got.hierarchy[got.meshes[0].parent_uuid].name, "geo")
        np.testing.assert_allclose(got.meshes[0].world_matrix(got.hierarchy), turn4, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.meshes[0].points, source["meshes"].points * 100.0, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(got.skins[0].weights, source["skins"].weights)
        self.assertEqual(list(got.skins[0].influences), list(source["skins"].influences))
        # the file holds every node as it was labelled: the turn is the read's
        with open_fbx(path, FBX) as scene:
            root = scene.GetRootNode()
            geo  = next(root.GetChild(i) for i in range(root.GetChildCount()) if str(root.GetChild(i).GetName()) == "geo")
            np.testing.assert_array_equal(tuple(geo.LclRotation.Get())[:3], (0.0, 0.0, 0.0))

    def test_the_save_of_the_read_is_maya_y_up_cm_and_settles_in_one_cycle(self):
        """a turned, scaled read holds float64 values no float32 key holds
        (a solved rotate, a key a hundred times): its save rounds them, as a
        first save of any data does, and from there the cycle is bit for bit"""
        _, path, _ = self.skinned()
        first = load_fbx(path)
        save_fbx(self.path("again.fbx"), *first)
        self.assertEqual(labelled(self.path("again.fbx")), MAYA_Y_UP_CM)
        second = load_fbx(self.path("again.fbx"))
        save_fbx(self.path("third.fbx"), *second)
        third = load_fbx(self.path("third.fbx"))
        for channel in ("translate", "rotate", "scale", "joint_orient", "rotate_axis"):
            np.testing.assert_allclose(getattr(second.hierarchy, channel), getattr(first.hierarchy, channel), rtol=1e-6, atol=1e-6, err_msg=channel)
            np.testing.assert_array_equal(getattr(third.hierarchy, channel), getattr(second.hierarchy, channel), err_msg=channel)
        self.assertEqual((second.clip.take, third.clip.take), (first.clip.take, first.clip.take))
        for name, block in first.clip.active._blocks.items():
            np.testing.assert_allclose(second.clip.active._blocks[name], block, rtol=1e-6, atol=1e-6, err_msg=name)
            np.testing.assert_array_equal(third.clip.active._blocks[name], second.clip.active._blocks[name], err_msg=name)
        np.testing.assert_array_equal(second.meshes[0].points, first.meshes[0].points)
        np.testing.assert_array_equal(second.skins[0].weights, first.skins[0].weights)
        self.assertEqual(second.hierarchy[second.meshes[0].parent_uuid].name, "geo")


def as_maya(world: np.ndarray, cm: float) -> np.ndarray:
    """``world``, a load_fbx world of a file in a unit of ``cm`` centimeters,
    as Maya's importer holds it: the unit as a scale of ``cm`` on the file's
    roots and compensating joints, the numbers as the file has them, so the
    position is the same and the 3x3 is ``cm`` times load_fbx's (measured)"""
    held = np.array(world, dtype=np.float64)
    held[:3, :3] *= cm
    return held


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaAxes(_Recorded):
    """Maya 2025's FBXImport of a Z-up file in metres puts every node where
    load_fbx reads it, the unit held as a scale on its roots where load_fbx
    holds it in the numbers (as_maya), and deforms a skinned one as cgmath
    does; as recorded"""

    def test_maya_imports_a_z_up_file_as_the_read(self):
        """a save relabelled Z-up, in cm and in metres: every node's world on
        every frame, the joint root's joint orient and the group root's
        rotate, solved again, as Maya's import holds them"""
        held = self.case("zup")
        save_fbx(self.path("y_cm.fbx"), clip=turned_clip(turned_rig()))
        same_read(self, load_fbx(self.path("y_cm.fbx")), load_fbx(self.fixture("y_cm.fbx")))
        for unit, cm in (("cm", 1.0), ("m", 100.0)):
            with self.subTest(unit=unit):
                path = relabelled(self.path("y_cm.fbx"), self.path(f"z_{unit}.fbx"), FBX.FbxAxisSystem.MayaZUp, unit)
                same_read(self, load_fbx(path), load_fbx(self.fixture(f"z_{unit}.fbx")))
                clip = load_fbx(self.fixture(f"z_{unit}.fbx")).clip.active
                self.assertEqual(held.value(f"{unit}/dagPoses"), 0)
                self.assertEqual(clip.start_frame, 0)
                wanted = np.asarray(clip.frames.world_matrix)
                for column, node in enumerate(clip):
                    worlds = held.array(f"{unit}/world/{node.name}")
                    for frame in range(FRAMES):
                        near(worlds[frame], as_maya(wanted[frame, column], cm), err_msg=(node.name, frame, unit))
                clip.frame = 0
                # the turn as Maya puts it: in a joint root's joint orient, in
                # a group root's rotate, solved again; the joint orients
                # under a root as they were
                self.assertEqual(held.value(f"{unit}/hips_type"), "joint")
                np.testing.assert_allclose(held.array(f"{unit}/hips_joint_orient"), clip["hips"].joint_orient, rtol=0, atol=1e-4)
                np.testing.assert_allclose(held.array(f"{unit}/knee_joint_orient"), (0.0, 0.0, -90.0),         rtol=0, atol=1e-4)
                np.testing.assert_allclose(held.array(f"{unit}/group_rotate"),      clip["group"].rotate,      rtol=0, atol=1e-4)
                np.testing.assert_allclose(held.array(f"{unit}/loc_rotate"),        clip["loc"].rotate,        rtol=0, atol=1e-4)
                self.assertTrue(np.any(np.abs(clip["hips"].joint_orient - [5.0, -15.0, 25.0]) > 1.0))
                # the unit: a scale on Maya's roots, in load_fbx's numbers
                np.testing.assert_allclose(held.array(f"{unit}/hips_scale"), (cm, cm, cm), rtol=0, atol=1e-6)
                np.testing.assert_allclose(held.array(f"{unit}/knee_translate"), (0.0, -4.0, 0.5), rtol=0, atol=1e-6)
                np.testing.assert_allclose(clip["knee"].translate, np.array((0.0, -4.0, 0.5)) * cm, rtol=0, atol=1e-9)
                np.testing.assert_array_equal(clip["hips"].scale, (1.0, 1.0, 1.0))

    def test_maya_imports_a_z_up_skinned_file_and_deforms_as_cgmath(self):
        """a skinned Z-up file in cm, as a game engine exports one: Maya turns
        its roots, the mesh's among them, binds the skin where load_fbx reads
        it and deforms as cgmath does"""
        from cgmath.geometry.deform import SkinDeformData

        held   = self.case("zup_skinned")
        rig    = bound()
        mesh   = body(rig["geo"].uuid)
        source = {"hierarchy": rig, "clip": motion(rig), "meshes": mesh, "skins": weighed(mesh)}
        save_fbx(self.path("y_cm.fbx"), **source)
        path = relabelled(self.path("y_cm.fbx"), self.path("z_cm.fbx"), FBX.FbxAxisSystem.MayaZUp, "cm")
        same_read(self, load_fbx(path), load_fbx(self.fixture("z_cm.fbx")))
        back = load_fbx(self.fixture("z_cm.fbx"))

        self.assertEqual(held.value("skinClusters"), 1)
        worlds = np.asarray(back.hierarchy.world_matrix)
        near(held.array("geomMatrix"), worlds[back.hierarchy.name.index("geo")], err_msg="geomMatrix")
        deformer = SkinDeformData(back.meshes[0], back.skins[0], bind_rig=back.hierarchy)
        wanted   = deformer.apply(back.clip.active.frames)
        for f in (0, TAKE // 2, TAKE - 1):
            near(held.array(f"world_points/{f}"), wanted[f], err_msg=f"frame {START + f}")
        # the deformation is no stand-still: the points move by whole units
        self.assertGreater(float(np.abs(wanted[-1] - wanted[0]).max()), 2.0 * UNIT)


if __name__ == "__main__":
    unittest.main()
