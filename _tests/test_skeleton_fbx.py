"""The r17-skeleton fork: one hierarchy or its takes, the meshes on its
nodes, saved, read back, and read by Maya's own FBXImport.

``cgmath.formats.save_fbx(path, hierarchy)`` writes a HierarchyData of
joints and transforms, ``save_fbx(path, clip=clips)`` a ClipList's nodes and
its takes (a ClipData as a list of one), ``meshes=`` each mesh as the shape
of the node its ``parent_uuid`` names; ``cgmath.formats.load_fbx(path)``
gives ``(hierarchy, clip, meshes, skins, uvs)`` back, ``clip`` a ClipList of
every take, None for what the file holds none of. A bind pose comes with
skins only. The FBX tests need the Autodesk SDK; the Maya ones read what
Maya 2025 did, recorded once (maya_fixtures), and need no Maya.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from cgmath._tests.maya_fixtures import recorded, same_read
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.geometry import mesh as mesh_module, MeshData, skin_weights as skin_module, SkinData
from cgmath.geometry.mesh import MeshList, UVData, UVList
from cgmath.geometry.skin_weights import SkinList
from cgmath.hierarchy import ClipData, ClipList, HierarchyData, TransformData, generate_uuid

FBX = load_sdk()

START  = 1001
FPS    = 30.0
FRAMES = 12

# frame numbers far outside the clip's range, and on its ends
FAR = (-100000, 0, 1000, 1001, 1012, 5000, 100000)

# (name, parent, node type, translate, rotate, scale, joint orient, rotate
# axis, rotate order, segment scale compensate), parents first
NODES = (
    ("base", None, "transform", (1.0, 2.0, 3.0), (0.0, 30.0, 0.0), (2.0, 2.0, 2.0), (0, 0, 0), (0, 0, 0), 0, False),
    ("hips", "base", "joint", (0.0, 90.0, 0.0), (5.0, 0.0, 0.0), (1, 1, 1), (0.0, 0.0, 90.0), (10.0, 0.0, 0.0), 1, False),
    ("spine", "hips", "joint", (10.0, 0.0, 0.0), (0.0, 0.0, 12.0), (1.5, 1.5, 1.5), (0.0, 0.0, -15.0), (0.0, 5.0, 0.0), 2, True),
    ("chest", "spine", "joint", (12.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1, 1, 1), (5.0, 0.0, 0.0), (0, 0, 0), 3, True),
    ("head", "chest", "joint", (15.0, 0.0, 0.0), (0, 0, 0), (1.0, 1.2, 0.8), (0, 0, 0), (0, 0, 0), 4, False),
    ("arm_grp", "chest", "transform", (0.0, 5.0, 3.0), (10.0, 20.0, 30.0), (1, 1, 1), (0, 0, 0), (0, 0, 0), 5, False),
    ("arm", "arm_grp", "joint", (2.0, 0.0, 0.0), (0, 0, 0), (0.5, 1.0, 1.0), (0.0, 90.0, 0.0), (0.0, 0.0, 10.0), 0, False),
    ("prop", "base", "transform", (0.0, 0.0, 5.0), (0, 0, 0), (1.0, 2.0, 3.0), (0, 0, 0), (0, 0, 0), 1, False),
)

# a user attribute of each numeric type, a vector and a text one, on hips:
# (name, cmds.addAttr type, value, enum names)
USER = (
    ("weight", "double", 0.5, None),
    ("ratio", "float", 0.25, None),
    ("twist", "doubleAngle", 45.0, None),
    ("reach", "doubleLinear", 12.5, None),
    ("count", "long", 3, None),
    ("level", "short", -2, None),
    ("tiny", "byte", 7, None),
    ("letter", "char", 65, None),
    ("on", "bool", True, None),
    ("mode", "enum", 5, "a=1:b=5:c"),
    ("label", "string", "hello", None),
)


def skeleton() -> HierarchyData:
    """five levels of joints and transforms, every rotate order, joint orients
    and rotate axes, segment scale compensate on and off, a uniform scale on
    parents, uneven ones on leaves, and every kind of user attribute"""
    uuids = {}
    nodes = []
    for name, parent, kind, t, r, s, jo, ra, order, ssc in NODES:
        node = TransformData(
            name,
            node_type=kind,
            uuid=generate_uuid(),
            parent_uuid=uuids.get(parent),
            translate=t,
            rotate=r,
            scale=s,
            joint_orient=jo,
            rotate_axis=ra,
            rotate_order=order,
            segment_scale_compensate = ssc,
        )
        uuids[name] = node.uuid
        nodes.append(node)
    rig  = HierarchyData(nodes)
    hips = rig["hips"]
    for name, kind, value, fields in USER:
        hips.add_user_attribute(name, value, attribute_type=kind, enum_names=fields)
    # a vector: a double3 of three doubles, as Maya's import makes one
    attrs        = hips.user_defined_attributes
    attrs["aim"] = {"attributeType": "double3", "numberOfChildren": 3, "keyable": True, "channel_box": False}
    for suffix, value in zip("012", (1.0, -2.0, 0.5)):
        attrs["aim" + suffix] = {
            "attributeType": "double",
            "value":         value,
            "parent":        "aim",
            "keyable":       True,
            "channel_box":   False,
        }
    return rig


def take(rig: HierarchyData) -> ClipData:
    """12 frames from 1001 at 30 fps, some channels animated, some constant"""
    clip = ClipData(rig, frames=FRAMES, start_frame=START, fps=FPS, take="walk")
    f    = np.arange(FRAMES, dtype=np.float64)[:, None]
    clip.frames["hips"].translate    = np.hstack([np.sin(f / 3.0), 90.0 + f * 0.25, -f * 0.1])
    clip.frames["hips"].rotate       = np.hstack([5.0 + f * 2.0, f * -1.5, np.cos(f) * 4.0])
    clip.frames["spine"].rotate      = np.hstack([f * 0.5, f * 0.0 + 1.0, 12.0 + f])
    clip.frames["chest"].rotate      = [[1.0, 2.0, 3.0]]
    clip.frames["head"].scale        = np.hstack([1.0 + f * 0.01, 1.2 - f * 0.02, 0.8 + f * 0.005])
    clip.frames["arm_grp"].translate = np.hstack([f * 0.3, 5.0 + f * 0.0, 3.0 - f * 0.2])
    clip.frames["hips"].weight       = 0.5 + f * 0.05
    clip.frames["hips"].count        = (f % 4.0) - 1.0
    clip.frames["hips"].on           = f % 2.0
    clip.frames["hips"].mode         = np.where(f % 3.0 == 0.0, 1.0, 5.0)
    clip.frames["hips"].aim          = np.hstack([f * 0.1, -2.0 + f * 0.0, f * f * 0.01])
    clip.frames["hips"].twist        = [[90.0]]
    clip.frame = 0
    return clip


def takes() -> ClipList:
    """three takes over skeleton(): walk, take(); run, a cut of 6 of its
    frames from 1003, every rotate turned 3 degrees; and a take of no name,
    3 frames from 1001; run active"""
    walk              = take(skeleton())
    run               = walk.frames[2:8]
    run.take          = "run"
    run.frames.rotate = np.asarray(run.frames.rotate) + 3.0
    blank             = walk.frames[0:3]
    blank.take        = ""
    clips             = ClipList([walk, run, blank])
    clips.active      = "run"
    return clips


def at_float32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def near(got, want, err_msg="") -> None:
    """``got`` within 1e-6 of the largest magnitude in ``want`` (at least 1):
    what Maya's float32 skinCluster, bind pose and export hold"""
    want = np.asarray(want, dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6 * max(1.0, float(np.abs(want).max())), err_msg=err_msg)


def user_values(node) -> dict:
    return {name: spec.get("value") for name, spec in node.user_defined_attributes.items()}


def user_types(node) -> dict:
    return {name: spec.get("attributeType", spec.get("dataType")) for name, spec in node.user_defined_attributes.items()}


def structure(nodes) -> list:
    """each node's name, parent, kind, rotate order, segment scale
    compensate and user attribute types"""
    return [
        (n.name, n.parent_name, n.node_type, n.rotate_order, n.segment_scale_compensate, user_types(n)) for n in nodes
    ]


def layout(clip: ClipData) -> dict:
    """{(node, channel): rows held}, every framed channel and user channel"""
    from cgmath.hierarchy.hierarchy import _user_channels

    out = {}
    for node in clip:
        axis = clip.frames[node.name]
        for channel in ("translate", "rotate", "scale", "joint_orient", "rotate_axis", "shear"):
            out[(node.name, channel)] = np.array(getattr(axis, channel))
        for name in _user_channels(node.user_defined_attributes):
            out[(node.name, name)] = np.array(getattr(axis, name))
    return out


def curve_count(clip: ClipData) -> int:
    """the curves a take of ``clip`` holds: three a transform channel, one a
    user channel, three a vector"""
    from cgmath.hierarchy.hierarchy import _user_channels

    found = 0
    for node in clip:
        found += 9 + sum(1 if kids is None else 3 for kids in _user_channels(node.user_defined_attributes).values())
    return found


def counts(path: str) -> dict:
    """the poses, takes, layers, curve nodes and curves a file holds, as the
    sdk reads it"""
    with open_fbx(path, FBX) as scene:

        def of(kind):
            return scene.GetSrcObjectCount(FBX.FbxCriteria.ObjectType(kind.ClassId))

        return {
            "poses":       scene.GetPoseCount(),
            "stacks":      of(FBX.FbxAnimStack),
            "layers":      of(FBX.FbxAnimLayer),
            "curve nodes": of(FBX.FbxAnimCurveNode),
            "curves":      of(FBX.FbxAnimCurve),
        }


class _Files(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="cgmath_skeleton_fbx_")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def path(self, name: str) -> str:
        return os.path.join(self.folder, name)

    def assertSameSkeleton(self, got, want, exact: bool = True, atol: float = 0.0):
        self.assertEqual([n.name for n in got], [n.name for n in want])
        self.assertEqual([n.node_type for n in got], [n.node_type for n in want])
        self.assertEqual([n.parent_name for n in got], [n.parent_name for n in want])
        self.assertEqual([n.rotate_order for n in got], [n.rotate_order for n in want])
        self.assertEqual([n.segment_scale_compensate for n in got], [n.segment_scale_compensate for n in want])
        for channel in ("translate", "rotate", "scale", "joint_orient", "rotate_axis", "shear"):
            a = np.array([getattr(n, channel) for n in got])
            b = np.array([getattr(n, channel) for n in want])
            if exact:
                np.testing.assert_array_equal(a, b, err_msg=channel)
            else:
                np.testing.assert_allclose(a, b, rtol=0, atol=atol, err_msg=channel)
        for a, b in zip(got, want):
            self.assertEqual(user_types(a), user_types(b), a.name)
            if exact:
                self.assertEqual(user_values(a), user_values(b), a.name)

    def assertSameClip(self, got, want):
        self.assertEqual((got.start_frame, got.frame_count, got.fps), (want.start_frame, want.frame_count, want.fps))
        self.assertEqual(structure(got), structure(want))
        a, b = layout(got), layout(want)
        self.assertEqual(list(a), list(b))
        for key in a:
            self.assertEqual(a[key].shape, b[key].shape, key)
            np.testing.assert_array_equal(a[key], b[key], err_msg=str(key))

    def assertSameClips(self, got, want):
        """two ClipLists: the same takes by name, in order, the same one
        active, each take the same clip"""
        self.assertEqual(got.take, want.take)
        self.assertEqual(got._active, want._active)
        for a, b in zip(got, want):
            self.assertSameClip(a, b)


class TestClipLayout(unittest.TestCase):
    """a channel holds one row, constant on every frame number, or F rows"""

    def test_one_row_is_constant_and_f_rows_animated(self):
        clip = take(skeleton())
        self.assertEqual(clip.frames["chest"].rotate.shape, (1, 3))
        self.assertEqual(clip.frames["hips"].rotate.shape,  (FRAMES, 3))
        self.assertEqual(clip.frames["hips"].twist.shape,   (1, 1))
        self.assertEqual(clip.frames["hips"].weight.shape,  (FRAMES, 1))
        self.assertEqual(clip.frames["hips"].aim.shape,     (FRAMES, 3))
        self.assertEqual(clip.frames.rotate.shape,          (FRAMES, len(NODES), 3))
        # equal values are no reason to be constant: F equal rows stay F rows
        clip.frames["prop"].translate = np.zeros((FRAMES, 3))
        self.assertEqual(clip.frames["prop"].translate.shape, (FRAMES, 3))
        clip.frames["prop"].translate = [[0.0, 0.0, 5.0]]
        self.assertEqual(clip.frames["prop"].translate.shape, (1, 3))

    def test_a_constant_channel_reads_its_value_at_every_frame_number(self):
        clip = take(skeleton())
        for frame in (0, 1000, 1001, 1012, 5000):
            pose = clip.at(frame)
            np.testing.assert_array_equal(pose["chest"].rotate, [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(pose["base"].translate, [1.0, 2.0, 3.0])
            self.assertEqual(pose["hips"].twist, 90.0)
            self.assertEqual(pose["hips"].level, -2)
        # an animated one: its first row before the clip, its last after it
        rows = clip.frames["hips"].rotate
        np.testing.assert_array_equal(clip.at(0)["hips"].rotate,    rows[0])
        np.testing.assert_array_equal(clip.at(1005)["hips"].rotate, rows[4])
        np.testing.assert_array_equal(clip.at(5000)["hips"].rotate, rows[-1])

    def test_a_write_at_the_playhead_animates_a_constant_channel(self):
        clip                 = take(skeleton())
        clip.frame           = 4
        clip["chest"].rotate = [7.0, 8.0, 9.0]
        rows                 = clip.frames["chest"].rotate
        self.assertEqual(rows.shape, (FRAMES, 3))
        np.testing.assert_array_equal(rows[4], [7.0, 8.0, 9.0])
        np.testing.assert_array_equal(np.delete(rows, 4, axis=0), np.tile([1.0, 2.0, 3.0], (FRAMES - 1, 1)))
        clip["hips"].twist = 10.0
        rows               = clip.frames["hips"].twist
        self.assertEqual(rows.shape, (FRAMES, 1))
        self.assertEqual(rows[4, 0], 10.0)
        self.assertEqual(rows[0, 0], 90.0)

    def test_the_layout_survives_copies_cuts_and_files(self):
        clip = take(skeleton())
        want = layout(clip)
        for label, other in (
            ("copy", clip.copy()),
            ("from_dict", ClipData.from_dict(clip.to_dict())),
        ):
            with self.subTest(label):
                got = layout(other)
                self.assertEqual(list(got), list(want))
                for key in want:
                    np.testing.assert_array_equal(got[key], want[key], err_msg=str(key))
        cut = clip.frames[2:5]
        self.assertEqual(cut.frames["chest"].rotate.shape, (1, 3))
        self.assertEqual(cut.frames["hips"].rotate.shape, (3, 3))
        np.testing.assert_array_equal(cut.frames["hips"].weight, clip.frames["hips"].weight[2:5])

    def test_the_clip_holds_no_rest_pose(self):
        clip = take(skeleton())
        self.assertFalse(hasattr(clip, "rest_pose"))
        self.assertNotIn("rest_pose", clip.to_dict())


class TestArguments(_Files):
    """save_fbx takes one skeleton or one clip, never both, nothing else;
    no sdk needed"""

    def refused(self, error, message, *args, **kwargs):
        with self.assertRaisesRegex(error, message):
            save_fbx(self.path("refused.fbx"), *args, **kwargs)
        self.assertFalse(os.path.exists(self.path("refused.fbx")))

    def test_a_skeleton_and_a_clip_together(self):
        self.refused(NotImplementedError, "comes with skins", skeleton(), take(skeleton()))
        self.refused(NotImplementedError, "comes with skins", hierarchy=skeleton(), clip=take(skeleton()))

    def test_neither(self):
        self.refused(TypeError, r"takes a hierarchy \(a HierarchyData\) or clip=")
        self.refused(TypeError, r"takes a hierarchy \(a HierarchyData\) or clip=", None, None)
        self.refused(TypeError, r"takes a hierarchy \(a HierarchyData\) or clip=", meshes=MeshList([]))

    def test_a_clip_as_the_hierarchy(self):
        self.refused(TypeError, "the hierarchy is a HierarchyData, not a ClipData, pass it as clip=", take(skeleton()))
        self.refused(TypeError, "not a ClipData, pass it as clip=", hierarchy=take(skeleton()))

    def test_anything_else(self):
        self.refused(TypeError, "the clip is a ClipData or a ClipList, not a HierarchyData", clip=skeleton())
        self.refused(TypeError, "the clip is a ClipData or a ClipList, not a list", clip=[take(skeleton())])
        self.refused(TypeError, "the hierarchy is a HierarchyData, not a ClipList, pass it as clip=", ClipList([take(skeleton())]))
        self.refused(TypeError, "the hierarchy is a HierarchyData, not a list",     list(skeleton()))
        self.refused(ValueError, "the ClipList holds no take", clip=ClipList())
        with self.assertRaises(TypeError):
            take(skeleton()).save_fbx(self.path("x.fbx"), hierarchy=skeleton())


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestClipAlone(_Files):
    """save_fbx(path, clip=clip): the clip's nodes at its first frame, and its
    take; load_fbx gives (None, clip)"""

    def test_save_load_gives_no_skeleton_and_the_clip_in_float32(self):
        clip = take(skeleton())
        save_fbx(self.path("a.fbx"), clip=clip)
        rig, clips = load_fbx(self.path("a.fbx"))[:2]
        self.assertIsNone(rig)
        self.assertIsInstance(clips, ClipList)
        self.assertEqual(clips.take, ["walk"])
        back = clips.active
        self.assertIs(back, clips[0])
        self.assertEqual(back.take, "walk")
        self.assertEqual((back.start_frame, back.frame_count, back.fps), (START, FRAMES, FPS))
        self.assertEqual(structure(back), structure(clip))
        self.assertEqual(back.keyed.tolist(), clip.keyed.tolist())  # a constant channel is one key in the file
        self.assertIn(False, clip.keyed.tolist())
        self.assertEqual(back["hips"].label, "hello")
        want = layout(clip)
        read = layout(back)
        self.assertEqual(list(read), list(want))
        for key in want:
            self.assertEqual(read[key].shape, want[key].shape, key)
            np.testing.assert_array_equal(read[key], at_float32(want[key]), err_msg=str(key))

    def test_a_second_save_and_load_is_bit_for_bit(self):
        save_fbx(self.path("a.fbx"), clip=take(skeleton()))
        first = load_fbx(self.path("a.fbx")).clip.active
        save_fbx(self.path("b.fbx"), clip=first)
        rig, clips = load_fbx(self.path("b.fbx"))[:2]
        self.assertIsNone(rig)
        second = clips.active
        self.assertEqual(second.take, first.take)
        self.assertSameClip(second, first)
        for frame in (START, START + 5, START + FRAMES - 1):
            self.assertSameSkeleton(second.at(frame), first.at(frame))

    def test_constant_channels_read_back_at_every_frame_number(self):
        save_fbx(self.path("a.fbx"), clip=take(skeleton()))
        clip = load_fbx(self.path("a.fbx")).clip.active
        for frame in FAR:
            pose = clip.at(frame)
            np.testing.assert_array_equal(pose["chest"].rotate, [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(pose["base"].translate, [1.0, 2.0, 3.0])
            self.assertEqual(pose["hips"].twist, 90.0)
            self.assertEqual(pose["hips"].level, -2)
        rows = clip.frames["hips"].rotate
        np.testing.assert_array_equal(clip.at(FAR[0])["hips"].rotate, rows[0])
        np.testing.assert_array_equal(clip.at(FAR[-1])["hips"].rotate, rows[-1])

    def test_the_nodes_hold_the_first_frame(self):
        """their own values, what Maya shows with animation off: the clip's
        pose at its first frame, wherever its playhead is"""
        clip       = take(skeleton())
        clip.frame = 7
        save_fbx(self.path("a.fbx"), clip=clip)
        self.assertSameSkeleton(HierarchyData.load_fbx(self.path("a.fbx")), clip.at(START))

    def test_the_file_holds_one_take_every_curve_and_no_pose(self):
        clip = take(skeleton())
        save_fbx(self.path("a.fbx"), clip=clip)
        found = counts(self.path("a.fbx"))
        self.assertEqual((found["poses"], found["stacks"], found["layers"]), (0, 1, 1))
        self.assertEqual(found["curves"], curve_count(clip))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestTakes(_Files):
    """save_fbx(path, clip=clips): every take of a ClipList, one animation
    stack each, the active take the file's; load_fbx gives the list back"""

    def test_every_take_round_trips_the_active_one_active(self):
        clips = takes()
        save_fbx(self.path("a.fbx"), clip=clips)
        found = counts(self.path("a.fbx"))
        self.assertEqual((found["poses"], found["stacks"], found["layers"]), (0, 3, 3))
        self.assertEqual(found["curves"], sum(curve_count(clip) for clip in clips))

        rig, back = load_fbx(self.path("a.fbx"))[:2]
        self.assertIsNone(rig)
        self.assertIsInstance(back, ClipList)
        self.assertEqual(back.take, ["walk", "run", "Take 003"])
        self.assertIs(back.active, back["run"])
        self.assertEqual((back.start_frame, back.frame_count, back.fps), ([START, START + 2, START], [FRAMES, 6, 3], [FPS] * 3))
        self.assertEqual(len({tuple(uuids) for uuids in back.uuid}), 1)  # one skeleton, one set of uuids
        for got, want in zip(back, clips):
            self.assertEqual(structure(got), structure(want))
            self.assertEqual(got.keyed.tolist(), want.keyed.tolist())
            a, b = layout(got), layout(want)
            self.assertEqual(list(a), list(b))
            for key in b:
                self.assertEqual(a[key].shape, b[key].shape, key)
                np.testing.assert_array_equal(a[key], at_float32(b[key]), err_msg=str(key))

        # the class routes: every take, the active take, the nodes at the
        # active take's first frame
        self.assertSameClips(ClipList.load_fbx(self.path("a.fbx")), back)
        self.assertSameClip(ClipData.load_fbx(self.path("a.fbx")), back["run"])
        self.assertSameSkeleton(HierarchyData.load_fbx(self.path("a.fbx")), clips["run"].at(START + 2))

        # a second cycle, bit for bit, through save_fbx and ClipList.save_fbx
        save_fbx(self.path("b.fbx"), clip=back)
        self.assertSameClips(load_fbx(self.path("b.fbx")).clip, back)
        back.save_fbx(self.path("c.fbx"))
        self.assertSameClips(load_fbx(self.path("c.fbx")).clip, back)

    def test_a_clip_is_written_as_a_list_of_one(self):
        clip = take(skeleton())
        save_fbx(self.path("a.fbx"), clip=clip)
        save_fbx(self.path("b.fbx"), clip=ClipList([clip]))
        one, other = load_fbx(self.path("a.fbx")).clip, load_fbx(self.path("b.fbx")).clip
        self.assertIsInstance(one, ClipList)
        self.assertEqual((one.take, one._active), (["walk"], 0))
        self.assertIs(one.active, one[0])
        self.assertSameClips(one, other)
        self.assertEqual(ClipList.load_fbx(self.path("a.fbx")).take, ["walk"])

    def test_a_file_of_no_take(self):
        skeleton().save_fbx(self.path("rig.fbx"))
        self.assertEqual(len(ClipList.load_fbx(self.path("rig.fbx"))), 0)
        self.assertIsNone(load_fbx(self.path("rig.fbx")).clip)

    def test_refused_before_anything_is_written(self):
        def refused(error, message, **kwargs):
            with self.assertRaisesRegex(error, message):
                save_fbx(self.path("refused.fbx"), **kwargs)
            self.assertFalse(os.path.exists(self.path("refused.fbx")))

        rates        = takes()
        rates[1].fps = 24.0
        refused(ValueError, r"the takes play at \[24\.0, 30\.0\] fps; a file holds one frame rate", clip=rates)
        twice         = takes()
        twice[2].take = "walk"
        refused(ValueError, r"two takes would be written as \['walk'\]; take names must differ", clip=twice)
        orient                                = takes()
        orient[0].frames["hips"].joint_orient = (0.0, 0.0, 91.0)
        refused(
            ValueError,
            r"node 'hips' channel 'joint_orient' is \[0\.0, 0\.0, 91\.0\] in take 0 \('walk'\) and \[0\.0, 0\.0, 90\.0\] "
            r"in the active take 1 \('run'\): a file holds one joint_orient a node, the active take's",
            clip=orient,
        )
        axis                                = takes()
        axis[2].frames["spine"].rotate_axis = (0.0, 6.0, 0.0)
        refused(ValueError, r"node 'spine' channel 'rotate_axis' is \[0\.0, 6\.0, 0\.0\] in take 2 and", clip=axis)
        moving                                = takes()
        moving[1].frames["hips"].joint_orient = np.array([[0.0, 0.0, 90.0 + f] for f in range(6)])
        refused(ValueError, "node 'hips' channel 'joint_orient' frame 1004", clip=moving)


def hidden(clip: ClipData) -> ClipData:
    """``clip`` with prop hidden on frames 3 to 5 and head hidden throughout"""
    shown                          = np.ones((clip.frame_count, 1), dtype=bool)
    shown[3:6]                     = False
    clip.frames["prop"].visibility = shown
    clip.frames["head"].visibility = False
    return clip


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestVisibilityAndLocators(_Files):
    """a node's visibility over the take, and locators, round trip"""

    def test_visibility_keyed_over_the_take(self):
        plain = take(skeleton())
        clip  = hidden(take(skeleton()))
        self.assertEqual(clip.keyed.tolist(), [k or n == "prop" for k, n in zip(plain.keyed.tolist(), plain.name)])
        save_fbx(self.path("plain.fbx"), clip=plain)
        save_fbx(self.path("a.fbx"), clip=clip)
        self.assertEqual(counts(self.path("a.fbx"))["curves"], counts(self.path("plain.fbx"))["curves"] + 1)  # prop's Visibility

        back = load_fbx(self.path("a.fbx")).clip.active
        np.testing.assert_array_equal(back.frames.visibility, clip.frames.visibility)
        self.assertEqual(back.keyed.tolist(), clip.keyed.tolist())
        self.assertFalse(back["head"].visibility)
        self.assertIsNotNone(back._visible)
        # the nodes alone: the first frame's values
        self.assertEqual([n.visibility for n in HierarchyData.load_fbx(self.path("a.fbx"))], [n.visibility for n in clip.at(START)])
        # a second cycle, bit for bit, visibility included
        save_fbx(self.path("b.fbx"), clip=back)
        again = load_fbx(self.path("b.fbx")).clip.active
        self.assertSameClip(again, back)
        np.testing.assert_array_equal(again.frames.visibility, back.frames.visibility)

    def test_a_locator_round_trips(self):
        rig                   = skeleton()
        rig["prop"].node_type = "locator"
        save_fbx(self.path("rig.fbx"), rig)
        back = load_fbx(self.path("rig.fbx")).hierarchy
        self.assertEqual(back["prop"].node_type, "locator")
        self.assertSameSkeleton(back, rig)
        clip = take(rig)
        save_fbx(self.path("clip.fbx"), clip=clip)
        got = load_fbx(self.path("clip.fbx")).clip.active
        self.assertEqual(got["prop"].node_type, "locator")
        self.assertEqual(structure(got), structure(clip))
        # a bare null stays a transform
        self.assertEqual(back["base"].node_type, "transform")


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSkeletonAlone(_Files):
    """save_fbx(path, skeleton): the nodes and their own values, no take;
    load_fbx gives (skeleton, None)"""

    def test_save_load_gives_the_skeleton_bit_for_bit_from_a_file_of_no_take_and_no_pose(self):
        rig = skeleton()
        save_fbx(self.path("a.fbx"), rig)
        self.assertEqual(
            counts(self.path("a.fbx")), {"poses": 0, "stacks": 0, "layers": 0, "curve nodes": 0, "curves": 0}
        )
        got, clip = load_fbx(self.path("a.fbx"))[:2]
        self.assertIsNone(clip)
        self.assertSameSkeleton(got, rig)
        save_fbx(self.path("b.fbx"), hierarchy=got)
        again, clip = load_fbx(self.path("b.fbx"))[:2]
        self.assertIsNone(clip)
        self.assertSameSkeleton(again, rig)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestClassRoutes(_Files):
    def test_hierarchy_routes_write_a_skeleton_and_read_any_file_s_nodes(self):
        from cgmath.hierarchy import TransformList

        rig, clip = skeleton(), take(skeleton())
        rig.save_fbx(self.path("rig.fbx"))
        self.assertIsNone(load_fbx(self.path("rig.fbx"))[1])
        self.assertSameSkeleton(HierarchyData.load_fbx(self.path("rig.fbx")), rig)
        # a file with a take: its nodes at their own values
        save_fbx(self.path("clip.fbx"), clip=clip)
        self.assertSameSkeleton(HierarchyData.load_fbx(self.path("clip.fbx")), clip.at(START))
        view = TransformList.load_fbx(self.path("clip.fbx"))
        self.assertNotIsInstance(view, HierarchyData)
        self.assertEqual(view.name, [name for name, *_ in NODES])
        np.testing.assert_array_equal(view.translate, clip.at(START).translate)
        # a view: its nodes as a skeleton
        rig.match("base", "prop").save_fbx(self.path("view.fbx"))
        self.assertEqual(load_fbx(self.path("view.fbx"))[0].name, ["base", "prop"])

    def test_clip_routes_write_and_read_the_take(self):
        clip = take(skeleton())
        clip.save_fbx(self.path("clip.fbx"))
        rig, back = load_fbx(self.path("clip.fbx"))[:2]
        self.assertIsNone(rig)
        self.assertSameClip(ClipData.load_fbx(self.path("clip.fbx")), back.active)
        self.assertSameClips(ClipList.load_fbx(self.path("clip.fbx")), back)
        skeleton().save_fbx(self.path("rig.fbx"))
        with self.assertRaisesRegex(ValueError, "holds no take"):
            ClipData.load_fbx(self.path("rig.fbx"))
        self.assertEqual(ClipList.load_fbx(self.path("rig.fbx")), ClipList())


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestRefusals(_Files):
    def refused(self, message, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, message):
            save_fbx(self.path("refused.fbx"), *args, **kwargs)
        self.assertFalse(os.path.exists(self.path("refused.fbx")))

    def test_shear(self):
        rig              = skeleton()
        rig["arm"].shear = [0.1, 0.0, 0.0]
        self.refused("node 'arm' channel 'shear'", rig)
        clip              = take(skeleton())
        shear             = np.zeros((FRAMES, len(NODES), 3))
        shear[5, 1, 2]    = 1e-3
        clip.frames.shear = shear
        self.refused("node 'hips' channel 'shear' frame 1006", clip=clip)

    def test_uneven_scale_on_a_parent(self):
        rig                = skeleton()
        rig["spine"].scale = [1.5, 1.5, 1.6]
        self.refused("node 'spine' channel 'scale'", rig)
        clip              = take(skeleton())
        scale             = np.array(clip.frames.scale)
        scale[7, 2]       = [1.5, 1.0, 1.5]
        clip.frames.scale = scale
        self.refused("node 'spine' channel 'scale' frame 1008", clip=clip)
        # on a leaf it is kept
        rig               = skeleton()
        rig["head"].scale = [1.0, 3.0, 0.1]
        save_fbx(self.path("leaf.fbx"), rig)
        clip                      = take(skeleton())
        clip.frames["head"].scale = np.tile([1.0, 3.0, 0.1], (FRAMES, 1))
        save_fbx(self.path("leaf_clip.fbx"), clip=clip)

    def test_a_joint_orient_or_rotate_axis_the_clip_changes(self):
        for channel in ("joint_orient", "rotate_axis"):
            with self.subTest(channel):
                clip = take(skeleton())
                held = np.array(getattr(clip.frames, channel))
                held[3, 2, 0] += 1.0
                setattr(clip.frames, channel, held)
                self.refused(f"node 'spine' channel '{channel}' frame 1004", clip=clip)

    def test_other_node_kinds(self):
        rig                   = skeleton()
        rig["prop"].node_type = "camera"  # a kind the writer does not know
        self.refused("node 'prop' is a 'camera'", rig)
        self.refused("node 'prop' is a 'camera'", clip=take(rig))
        # a locator is a transform with a look: no joint orient, no compensate
        rig                                  = skeleton()
        rig["prop"].node_type                = "locator"
        rig["prop"].segment_scale_compensate = True
        self.refused("node 'prop' channel 'segment_scale_compensate' is on a locator", rig)

    def written(self, name="sdk.fbx", parent_scale=(1.0, 1.0, 1.0), inherit=None, keys=None) -> str:
        """a file the sdk writes: a group p scaled ``parent_scale`` over a
        joint c turned (10, 20, 30); ``keys`` p's scale on frames 0, 1 ..."""
        return _written(self.path(name), parent_scale=parent_scale, inherit=inherit, keys=keys)

    def test_load_refuses_uneven_parents_and_inherit_types(self):
        rig, clip = load_fbx(self.written())[:2]
        self.assertIsNone(clip)
        self.assertEqual([n.name for n in rig], ["p", "c"])
        # c turned under p's uneven scale: a sheared world, refused at p
        with self.assertRaisesRegex(ValueError, "node 'p' channel 'scale'"):
            load_fbx(self.written(parent_scale=(1.0, 2.0, 1.0)))
        with self.assertRaisesRegex(ValueError, "node 'p' channel 'scale'"):
            HierarchyData.load_fbx(self.written(parent_scale=(1.0, 2.0, 1.0)))
        keyed = self.written(keys=[(1.0, 1.0, 1.0), (1.0, 1.5, 1.0), (1.0, 1.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "node 'p' channel 'scale' frame 1"):
            load_fbx(keyed)
        # the nodes alone, the take left unread
        self.assertEqual(HierarchyData.load_fbx(keyed).name, ["p", "c"])
        # RrSs under a uniformly scaled parent is the standard product: the
        # same rig as the RSrs file's
        standard = load_fbx(self.written("rsrs.fbx", parent_scale=(2.0, 2.0, 2.0))).hierarchy
        own_axes = load_fbx(self.written("rrss.fbx", parent_scale=(2.0, 2.0, 2.0), inherit=FBX.FbxTransform.EInheritType.eInheritRrSs)).hierarchy
        self.assertSameSkeleton(own_axes, standard)
        np.testing.assert_allclose(own_axes["c"].world_matrix, standard["c"].world_matrix, rtol=0, atol=1e-12)
        self.assertNotEqual(float(own_axes["c"].world_matrix[3, 0]), 1.0)  # p's scale reaches c
        # under an uneven parent the scale rule speaks first
        with self.assertRaisesRegex(ValueError, "node 'p' channel 'scale'"):
            load_fbx(self.written(parent_scale=(1.0, 2.0, 1.0), inherit=FBX.FbxTransform.EInheritType.eInheritRrSs))


def _written(path, parent_scale=(1.0, 1.0, 1.0), bind=None, inherit=None, keys=None) -> str:
    """a file the sdk writes: a group p scaled ``parent_scale`` over a joint c
    at (1, 2, 3) turned (10, 20, 30); ``bind`` a bind pose: "moved" lists c 5
    further along x and p scaled 3, "sheared" c sheared; ``keys`` p's scale
    on frames 0, 1 ... of a take at 30 fps"""
    manager = FBX.FbxManager.Create()
    try:
        scene = FBX.FbxScene.Create(manager, "")
        p     = FBX.FbxNode.Create(scene, "p")
        c     = FBX.FbxNode.Create(scene, "c")
        c.SetNodeAttribute(FBX.FbxSkeleton.Create(scene, ""))
        scene.GetRootNode().AddChild(p)
        p.AddChild(c)
        p.LclScaling.Set(FBX.FbxDouble3(*parent_scale))
        c.LclTranslation.Set(FBX.FbxDouble3(1.0, 2.0, 3.0))
        c.LclRotation.Set(FBX.FbxDouble3(10.0, 20.0, 30.0))
        for node in (p, c):
            node.SetRotationActive(True)
            node.SetTransformationInheritType(FBX.FbxTransform.EInheritType.eInheritRSrs)
        if inherit is not None:
            c.SetTransformationInheritType(inherit)
        if bind is not None:
            pose = FBX.FbxPose.Create(scene, "bind")
            pose.SetIsBindPose(True)
            top   = FBX.FbxMatrix(p.EvaluateGlobalTransform())
            world = FBX.FbxMatrix(c.EvaluateGlobalTransform())
            if bind == "sheared":
                world.Set(0, 1, world.Get(0, 1) + 0.25)
            else:
                world.Set(3, 0, world.Get(3, 0) + 5.0)
                for k in range(3):
                    top.Set(k, k, top.Get(k, k) * 3.0)
            pose.Add(p, top)
            pose.Add(c, world)
            scene.AddPose(pose)
        if keys is not None:
            mode = FBX.FbxTime.EMode.eFrames30
            scene.GetGlobalSettings().SetTimeMode(mode)
            stack = FBX.FbxAnimStack.Create(scene, "take")
            layer = FBX.FbxAnimLayer.Create(scene, "BaseLayer")
            stack.AddMember(layer)
            times = []
            for frame in range(len(keys)):
                when = FBX.FbxTime()
                when.SetFrame(frame, mode)
                times.append(when)
            p.LclScaling.GetCurveNode(layer, True)
            linear = FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
            for axis, component in enumerate("XYZ"):
                curve = typed(p.LclScaling.GetCurve(layer, component, True), FBX.FbxAnimCurve, FBX)
                key   = FBX.FbxAnimCurveKey()
                key.SetInterpolation(linear)
                curve.KeyModifyBegin()
                for when, value in zip(times, keys):
                    key.SetValue(float(value[axis]))
                    curve.KeyAdd(when, key)
                curve.KeyModifyEnd()
            stack.SetLocalTimeSpan(FBX.FbxTimeSpan(times[0], times[-1]))
            stack.SetReferenceTimeSpan(FBX.FbxTimeSpan(times[0], times[-1]))
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestPoses(_Files):
    """no save writes a pose; a file's bind pose places no node"""

    def test_no_save_writes_a_pose(self):
        for label, kwargs in (("hierarchy", {"hierarchy": skeleton()}), ("clip", {"clip": take(skeleton())})):
            for as_ascii in (False, True):
                with self.subTest(label, as_ascii=as_ascii):
                    save_fbx(self.path("a.fbx"), as_ascii=as_ascii, **kwargs)
                    self.assertEqual(counts(self.path("a.fbx"))["poses"], 0)
        skeleton().save_fbx(self.path("rig.fbx"))
        take(skeleton()).save_fbx(self.path("clip.fbx"))
        self.assertEqual([counts(self.path(x))["poses"] for x in ("rig.fbx", "clip.fbx")], [0, 0])

    def test_a_bind_pose_is_left_out(self):
        """a bind pose listing the nodes elsewhere, or sheared: the nodes'
        own values, with a take or without"""
        for bind in ("moved", "sheared"):
            with self.subTest(bind):
                path = _written(self.path(f"{bind}.fbx"), bind=bind)
                self.assertEqual(counts(path)["poses"], 1)
                for rig in (load_fbx(path)[0], HierarchyData.load_fbx(path)):
                    np.testing.assert_array_equal(rig["p"].scale,     [1.0, 1.0, 1.0])
                    np.testing.assert_array_equal(rig["c"].translate, [1.0, 2.0, 3.0])
                    np.testing.assert_array_equal(rig["c"].rotate,    [10.0, 20.0, 30.0])
                    np.testing.assert_array_equal(rig["c"].shear,     [0.0, 0.0, 0.0])
        path = _written(self.path("take.fbx"), bind="moved", keys=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)])
        rig, clips = load_fbx(path)[:2]
        self.assertIsNone(rig)
        clip = clips.active
        np.testing.assert_array_equal(clip.frames["c"].translate, [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(clip.frames["c"].rotate,    [[10.0, 20.0, 30.0]])
        np.testing.assert_array_equal(clip.frames["p"].scale,     [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])


# Maya's FBXImport makes a double of a float, doubleAngle or doubleLinear,
# an enum whose fields Maya numbers by their place in the list, and its
# FBXExport adds filmboxTypeID and writes a char as 0
MAYA_TYPES = {"float": "double", "doubleAngle": "double", "doubleLinear": "double"}


def maya_value(node, name: str, value) -> np.ndarray:
    """a user channel's value as Maya holds it: an enum's field place"""
    from cgmath.formats.fbx import _user_property_value

    spec  = node.user_defined_attributes[name]
    value = np.asarray(value, dtype=np.float64)
    if spec.get("attributeType") == "enum":
        return np.vectorize(lambda x: float(_user_property_value(spec, int(round(x)))))(value)
    return value


class _Recorded(_Files):
    """a test over a case Maya 2025 recorded once
    (tools/make_maya_fbx_fixtures.py, read through maya_fixtures): ``case()``
    loads its readings, ``fixture()`` names a file it holds, the files Maya
    exported and the ones our writer gave Maya, and ``assertFreshSave``
    checks a save of the same data today still reads as the file Maya was
    given, so the readings still apply"""

    def case(self, name: str):
        self.held = recorded(name)
        return self.held

    def fixture(self, name: str) -> str:
        return self.held.path(name)

    def assertFreshSave(self, stored: str, **data) -> None:
        path = self.path("fresh.fbx")
        save_fbx(path, **data)
        same_read(self, load_fbx(path), load_fbx(stored), f"{os.path.basename(stored)}: ")


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMaya(_Recorded):
    """Maya 2025's FBXImport of a save, and its FBXExport of that scene, as
    recorded"""

    def maya_node(self, node, meta: dict) -> None:
        """the imported node's name, type, parent and static attributes, as
        Maya held them"""
        self.assertEqual(meta["count"], 1, node.name)
        if node.node_type == "locator":
            # Maya's locator: a transform holding a locator shape
            self.assertEqual(meta["type"], "transform")
            self.assertEqual(meta["shapes"], ["locator"], node.name)
        else:
            self.assertEqual(meta["type"], node.node_type)
            self.assertEqual(meta["shapes"], [], node.name)
        self.assertEqual(meta["parent"], node.parent_name)
        self.assertEqual(meta["rotate_order"], node.rotate_order)
        np.testing.assert_allclose(meta["rotate_axis"], node.rotate_axis, rtol=0, atol=1e-6)
        if node.node_type == "joint":
            np.testing.assert_allclose(meta["joint_orient"], node.joint_orient, rtol=0, atol=1e-6)
            self.assertEqual(meta["ssc"], node.segment_scale_compensate)

    def maya_user(self, node, frame=None) -> None:
        from cgmath.hierarchy.hierarchy import _user_channels

        for name in _user_channels(node.user_defined_attributes):
            value = self.held.array(f"user/{node.name}/{name}")
            if frame is not None:
                value = value[frame]
            want = maya_value(node, name, np.atleast_1d(np.asarray(getattr(node, name), dtype=np.float64)))
            np.testing.assert_allclose(value, want, rtol=0, atol=1e-6, err_msg=(node.name, name, frame))

    def test_maya_imports_the_clip_as_an_animated_hierarchy(self):
        """every node, and on every frame its channels, user attributes and
        world matrix, within 1e-6 of load_fbx's read of the same file"""
        held = self.case("clip_import")
        self.assertFreshSave(self.fixture("ours.fbx"), clip=take(skeleton()))
        rig, clips = load_fbx(self.fixture("ours.fbx"))[:2]
        self.assertIsNone(rig)
        back = clips.active
        self.assertEqual(held.value("counts")["dagPose"], 0)
        self.assertEqual(held.value("counts")["animCurve"], curve_count(back))
        for node in back:
            self.maya_node(node, held.value("nodes")[node.name])

        wanted = np.asarray(back.frames.world_matrix)
        for f in range(back.frame_count):
            back.frame = f
            for column, node in enumerate(back):
                for attr in ("translate", "rotate", "scale"):
                    np.testing.assert_allclose(
                        held.array(f"{attr}/{node.name}")[f], getattr(node, attr), rtol=0, atol=1e-6, err_msg=(node.name, attr, f)
                    )
                np.testing.assert_allclose(held.array(f"world/{node.name}")[f], wanted[f, column], rtol=0, atol=1e-6, err_msg=(node.name, f))
                self.maya_user(node, f)

    def test_maya_imports_visibility_keys_and_a_locator_and_exports_them_back(self):
        """prop, a locator hidden on frames 3 to 5, head hidden throughout:
        Maya's import keys prop's visibility as ours on every frame, gives
        prop a locator shape, and its export reads back as the save"""
        held = self.case("visibility_locator")
        rig  = skeleton()
        rig["prop"].node_type = "locator"
        clip = hidden(take(rig))
        self.assertFreshSave(self.fixture("ours.fbx"), clip=clip)
        for node in clip:
            self.maya_node(node, held.value("nodes")[node.name])
        shown = np.asarray(clip.frames.visibility)
        for column, node in enumerate(clip):
            np.testing.assert_array_equal(held.array(f"visibility/{node.name}"), shown[:, column], err_msg=node.name)
        self.assertEqual(held.value("prop_visibility_curves"), 1)

        back = load_fbx(self.fixture("maya.fbx")).clip.active
        self.assertEqual(back["prop"].node_type, "locator")
        np.testing.assert_array_equal(np.asarray(back.frames.visibility), shown)
        self.assertFalse(back["head"].visibility)

    def test_maya_reads_rrss_under_a_uniform_parent_as_we_do(self):
        """an sdk file with eInheritRrSs under a parent scaled 2 uniformly:
        Maya's import and load_fbx agree on every world within 1e-6"""
        held = self.case("rrss")
        path = _written(self.path("rrss.fbx"), parent_scale=(2.0, 2.0, 2.0), inherit=FBX.FbxTransform.EInheritType.eInheritRrSs)
        same_read(self, load_fbx(path), load_fbx(self.fixture("source.fbx")))
        rig = load_fbx(self.fixture("source.fbx")).hierarchy
        for column, node in enumerate(rig):
            np.testing.assert_allclose(
                held.array(f"world/{node.name}"), np.asarray(rig.world_matrix)[column], rtol=0, atol=1e-6, err_msg=node.name
            )

    def test_maya_imports_each_take(self):
        """a file of three takes: FBXImport -t picks one by position, and
        gives that take's curves alone, every node's world on every frame of
        it within 1e-6 of load_fbx's read of the take"""
        held = self.case("takes")
        self.assertFreshSave(self.fixture("ours.fbx"), clip=takes())
        back = load_fbx(self.fixture("ours.fbx")).clip
        self.assertEqual(back.take, ["walk", "run", "Take 003"])
        for index, clip in enumerate(back):
            with self.subTest(take=clip.take):
                self.assertEqual(held.value(f"take{index}/animCurve"), curve_count(clip))
                wanted = np.asarray(clip.frames.world_matrix)
                for column, node in enumerate(clip):
                    np.testing.assert_allclose(
                        held.array(f"take{index}/world/{node.name}"), wanted[:, column], rtol=0, atol=1e-6, err_msg=(clip.take, node.name)
                    )

    def test_maya_imports_the_skeleton_as_a_still_hierarchy(self):
        """no curve; every node's attributes and world matrix within 1e-6 of
        the skeleton"""
        held = self.case("skeleton_import")
        rig  = skeleton()
        self.assertFreshSave(self.fixture("ours.fbx"), hierarchy=rig)
        self.assertEqual(counts(self.fixture("ours.fbx"))["poses"], 0)
        self.assertEqual(held.value("counts")["animCurve"],         0)
        self.assertEqual(held.value("counts")["dagPose"],           0)
        worlds = np.asarray(rig.world_matrix)
        for column, node in enumerate(rig):
            self.maya_node(node, held.value("nodes")[node.name])
            self.maya_user(node)
            np.testing.assert_allclose(held.array(f"world/{node.name}"), worlds[column], rtol=0, atol=1e-6, err_msg=node.name)
            for attr in ("translate", "rotate", "scale"):
                np.testing.assert_allclose(held.array(f"{attr}/{node.name}"), getattr(node, attr), rtol=0, atol=1e-6)

    def test_maya_export_reads_back_as_the_save(self):
        """Maya's FBXExport of the import: load, save, load gives the first
        load back bit for bit; the first is the source's take within 1e-6"""
        from cgmath.hierarchy.hierarchy import _user_channels

        self.case("export_reads_back")
        self.assertFreshSave(self.fixture("ours.fbx"), clip=take(skeleton()))
        out = self.fixture("maya.fbx")

        rig, clips = load_fbx(out)[:2]
        self.assertIsNone(rig)
        self.assertEqual(clips.take, ["Take 001"])  # Maya's export names the one take
        first = clips.active
        save_fbx(self.path("again.fbx"), clip=first)
        rig, again = load_fbx(self.path("again.fbx"))[:2]
        self.assertIsNone(rig)
        second = again.active
        self.assertSameClip(second, first)
        for frame in (START, START + FRAMES - 1):
            self.assertSameSkeleton(second.at(frame), first.at(frame))

        source = load_fbx(self.fixture("ours.fbx")).clip.active
        self.assertEqual((first.start_frame, first.frame_count, first.fps), (START, FRAMES, FPS))
        self.assertEqual([n.name for n in first], [n.name for n in source])
        for a, b in zip(first, source):
            self.assertEqual((a.node_type, a.parent_name, a.rotate_order), (b.node_type, b.parent_name, b.rotate_order))
            self.assertEqual(a.segment_scale_compensate, b.segment_scale_compensate)
            types = {k: MAYA_TYPES.get(t, t) for k, t in user_types(b).items()}
            held  = user_types(a)
            self.assertEqual({k: held.get(k) for k in types}, types, a.name)

        # Maya's own filmboxTypeID left out
        a, b = layout(first), layout(source)
        self.assertEqual([key for key in a if key[1] != "filmboxTypeID"], list(b))
        for (name, channel), rows in b.items():
            self.assertEqual(a[(name, channel)].shape, rows.shape, (name, channel))
            if channel == "letter":
                continue  # Maya 2025's FBXExport writes a char as 0 (measured)
            node = source[name]
            if channel in _user_channels(node.user_defined_attributes):
                rows = maya_value(node, channel, rows)
            np.testing.assert_allclose(a[(name, channel)], rows, rtol=0, atol=1e-6, err_msg=(name, channel))


# ----------------------------------------------------------------------------
# one mesh and one skin: a skeleton bound to a mesh, a clip moving it

# (name, parent, node type, translate, rotate, scale, joint orient, rotate
# axis, rotate order, segment scale compensate), parents first; lengths in
# UNIT, a character's size (about 90 units tall): Maya's skinCluster deforms
# in float32, so its points are compared within 1e-6 of the farthest
# (near())
UNIT = 15.0
BOUND = (
    ("root", None, "transform", (0.25, 0.5, -0.25), (0.0, 15.0, 0.0), (1.25, 1.25, 1.25), (0, 0, 0), (0, 0, 0), 0, False),
    ("hips", "root", "joint", (0.0, 1.0, 0.0), (5.0, 0.0, 0.0), (1, 1, 1), (0.0, 0.0, 90.0), (10.0, 0.0, 0.0), 1, False),
    ("spine", "hips", "joint", (1.0, 0.0, 0.0), (0.0, 0.0, 12.0), (1.1, 1.1, 1.1), (0.0, 0.0, -15.0), (0.0, 5.0, 0.0), 2, True),
    ("chest", "spine", "joint", (1.25, 0.0, 0.0), (0.0, 3.0, 0.0), (1, 1, 1), (5.0, 0.0, 0.0), (0, 0, 0), 3, True),
    ("head", "chest", "joint", (1.0, 0.0, 0.0), (0, 0, 0), (1.0, 1.2, 0.8), (0, 0, 0), (0, 0, 0), 4, False),
    ("arm_grp", "chest", "transform", (0.0, 0.5, 0.25), (10.0, 20.0, 30.0), (1, 1, 1), (0, 0, 0), (0, 0, 0), 5, False),
    ("arm", "arm_grp", "joint", (0.5, 0.0, 0.0), (0, 0, 0), (1, 1, 1), (0.0, 90.0, 0.0), (0.0, 0.0, 10.0), 0, False),
    ("hand", "arm", "joint", (0.75, 0.0, 0.0), (0, 0, 0), (0.9, 1.1, 1.0), (0, 0, 0), (0, 0, 0), 5, True),
    ("prop", "arm_grp", "transform", (0.0, 0.5, 0.0), (0.0, 10.0, 0.0), (1, 1, 1), (0, 0, 0), (0, 0, 0), 3, False),
    ("leg", "hips", "joint", (0.0, -0.5, 0.5), (0, 0, 0), (1, 1, 1), (0.0, 0.0, -90.0), (0, 0, 0), 4, False),
    ("foot", "leg", "joint", (1.0, 0.0, 0.0), (0, 0, 0), (1.3, 0.7, 1.0), (0, 0, 0), (0, 0, 0), 2, False),
    ("geo", None, "transform", (0, 0, 0), (0, 0, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0), 0, False),
)

# the skin's influences, leaves among them
INFLUENCES = ("hips", "spine", "chest", "head", "arm", "hand", "leg", "foot")
LEAVES     = ("head", "hand", "foot")
TAKE       = 15


def bound() -> HierarchyData:
    """joints and transforms, joint orients, rotate axes, every rotate order,
    segment scale compensate on and off, uneven scales on the leaves; geo, a
    root transform at identity, the skinned mesh's node; prop, a leaf
    transform under arm_grp, a mesh's node the clip moves"""
    uuids = {}
    nodes = []
    for name, parent, kind, t, r, s, jo, ra, order, ssc in BOUND:
        node = TransformData(
            name,
            node_type=kind,
            uuid=generate_uuid(),
            parent_uuid=uuids.get(parent),
            translate=tuple(v * UNIT for v in t),
            rotate=r,
            scale=s,
            joint_orient=jo,
            rotate_axis=ra,
            rotate_order=order,
            segment_scale_compensate = ssc,
        )
        uuids[name] = node.uuid
        nodes.append(node)
    rig = HierarchyData(nodes)
    rig["hips"].add_user_attribute("weight", 0.5, attribute_type="double")
    return rig


def placed() -> HierarchyData:
    """bound() and one more root transform, placed: off the origin, turned
    and scaled, still over motion(): a skinned mesh's node that is not at
    identity (one the clip moved would move Maya's skinned mesh twice)"""
    rig = bound()
    rig.append(
        TransformData(
            "placed",
            node_type = "transform",
            uuid      = generate_uuid(),
            translate = (3.0 * UNIT, 2.0 * UNIT, -1.0 * UNIT),
            rotate    = (0.0, 25.0, 0.0),
            scale     = (1.5, 1.5, 1.5),
        )
    )
    return rig


def motion(rig: HierarchyData) -> ClipData:
    """15 frames: uniform scale on root, spine and arm (each with children),
    uneven scale on the leaves head, hand, foot and prop, rotation and
    translation"""
    clip = ClipData(rig, frames=TAKE, start_frame=START, fps=FPS, take="bend")
    f    = np.arange(TAKE, dtype=np.float64)[:, None]
    one  = np.ones((1, 3))
    clip.frames["root"].translate    = np.hstack([0.25 + f * 0.025, 0.5 + f * 0.0, -0.25 - f * 0.01]) * UNIT
    clip.frames["root"].scale        = (1.25 + f * 0.02) * one
    clip.frames["hips"].translate    = np.hstack([np.sin(f / 3.0) * 0.15, 1.0 + f * 0.01, -f * 0.005]) * UNIT
    clip.frames["hips"].rotate       = np.hstack([5.0 + f * 2.0, f * -1.5, np.cos(f) * 4.0])
    clip.frames["spine"].rotate      = np.hstack([f * 0.5, f * 0.0 + 1.0, 12.0 + f])
    clip.frames["spine"].scale       = (1.1 - f * 0.015) * one
    clip.frames["arm_grp"].translate = np.hstack([f * 0.015, 0.5 + f * 0.0, 0.25 - f * 0.01]) * UNIT
    clip.frames["arm"].rotate        = np.hstack([f * 3.0, 10.0 + f * 0.0, -f * 2.0])
    clip.frames["arm"].scale         = (1.0 + f * 0.03) * one
    clip.frames["leg"].rotate        = np.hstack([f * -2.0, f * 1.0, f * 2.5])
    clip.frames["head"].scale        = np.hstack([1.0 + f * 0.01, 1.2 - f * 0.02, 0.8 + f * 0.005])
    clip.frames["hand"].scale        = np.hstack([0.9 + f * 0.02, 1.1 + f * 0.0, 1.0 - f * 0.03])
    clip.frames["foot"].scale        = np.hstack([1.3 - f * 0.01, 0.7 + f * 0.02, 1.0 + f * 0.01])
    # prop's first frame is its bind value: Maya's dagPose lists the
    # influences and their parents only, so Maya's FBXExport writes a leaf
    # transform at the playhead (measured)
    clip.frames["prop"].rotate = np.hstack([f * 4.0, 10.0 + f * 0.0, -f * 1.5])
    clip.frames["prop"].scale  = np.hstack([1.0 + f * 0.02, 1.0 - f * 0.01, 1.0 + f * 0.015])
    clip.frames["hips"].weight = 0.5 + f * 0.05
    clip.frame                 = 0
    return clip


def body(parent=None) -> MeshData:
    """a tube of 24 rings of 12 points up the skeleton, quads and, every
    third quad split in two, triangles; float32 points, as Maya holds them;
    each corner's normal its point's, a triangle's corners their face's; on
    the node of uuid ``parent``"""
    rings, around = 24, 12
    turn   = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    height = np.linspace(0.5, 6.0, rings)
    points = np.stack(
        [np.tile(np.cos(turn) * 0.75 + 0.25, rings), np.repeat(height, around), np.tile(np.sin(turn) * 0.75 - 0.25, rings)],
        axis=1,
    )
    radial  = np.stack([np.tile(np.cos(turn), rings), np.zeros(rings * around), np.tile(np.sin(turn), rings)], axis=1)
    counts  = []
    indices = []
    ids     = []
    extra   = []
    for i in range(rings - 1):
        for j in range(around):
            a, b = i * around + j, i * around + (j + 1) % around
            c, d = b + around, a + around
            if (i * around + j) % 3:
                counts.append(4)
                indices += [a, b, c, d]
                ids     += [a, b, c, d]
                continue
            for face in ((a, b, c), (a, c, d)):
                p      = points[list(face)]
                normal = np.cross(p[1] - p[0], p[2] - p[0])
                extra.append(normal / np.linalg.norm(normal))
                counts.append(3)
                indices += list(face)
                ids     += [len(points) + len(extra) - 1] * 3
    normals = np.vstack([radial, np.array(extra)])
    return MeshData(
        name           = "body",
        points         = at_float32(points * UNIT),
        counts         = np.array(counts, dtype=np.int32),
        indices        = np.array(indices, dtype=np.int32),
        normals        = at_float32(normals),
        normal_indices = np.array(ids, dtype=np.int32),
        parent_uuid    = parent,
    )


def tetra(parent=None) -> MeshData:
    """four triangles over four points, a normal a face, a tenth of UNIT
    across; on the node of uuid ``parent``"""
    points  = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64) * (UNIT / 10.0)
    faces   = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    normals = []
    for face in faces:
        p      = points[face]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        normals.append(normal / np.linalg.norm(normal))
    return MeshData(
        name           = "tetra",
        points         = at_float32(points),
        counts         = np.full(4, 3, dtype=np.int32),
        indices        = faces.ravel(),
        normals        = at_float32(np.array(normals)),
        normal_indices = np.repeat(np.arange(4, dtype=np.int32), 3),
        parent_uuid    = parent,
    )


def weighed(mesh: MeshData) -> SkinData:
    """1 to 4 influences a point, 0.1 and over before normalizing; zeros
    elsewhere; naming ``mesh`` by its uuid"""
    rng     = np.random.default_rng(7)
    weights = np.zeros((mesh.point_count, len(INFLUENCES)))
    for row in range(mesh.point_count):
        picked               = rng.choice(len(INFLUENCES), size=rng.integers(1, 5), replace=False)
        values               = 0.1 + rng.random(len(picked))
        weights[row, picked] = values / values.sum()
    return SkinData(weights=weights, influences=list(INFLUENCES), name="bodySkin", mesh_uuid=mesh.uuid)


def bundle() -> dict:
    """save_fbx's arguments: the bind pose, its clip, body() on geo and its
    skin"""
    rig  = bound()
    mesh = body(rig["geo"].uuid)
    return {"hierarchy": rig, "clip": motion(bound()), "meshes": mesh, "skins": weighed(mesh)}


def _named(scene, name):
    from cgmath.formats._load import _walk

    return next(node for node, _ in _walk(scene) if str(node.GetName()) == name)


def _edited(source: str, path: str, edit) -> str:
    """``source`` read with the sdk, changed by ``edit(scene)``, written to
    ``path``"""
    from cgmath.formats._fbx_io import import_scene, new_manager

    manager = new_manager(FBX)
    try:
        scene = import_scene(manager, source, FBX)
        edit(scene)
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


class _Skinned(_Files):
    def assertSameMesh(self, got, want):
        """``got`` holds ``want``'s name, points and faces bit for bit, and
        shades as it does. The normals ``got`` stores are ``want``'s, bit for
        bit. Where it stores none (load_fbx stores no normal Maya would
        derive from the points) its corners hold ``want``'s normals within
        1e-5 a component, broken at ``want``'s hard edges when ``want``
        stores none either"""
        self.assertEqual(got.name, want.name)
        fields = ["points", "counts", "indices"]
        if got.normals is not None or want.normals is None:
            fields += ["normals", "normal_indices", "hard_edges"]
        for field in fields:
            a, b = getattr(got, field), getattr(want, field)
            if a is None or b is None:
                self.assertIs(a, b, field)
                continue
            self.assertEqual(a.dtype, b.dtype, field)
            np.testing.assert_array_equal(a, b, err_msg=field)
        np.testing.assert_allclose(got.get_face_vertex_normals(), want.get_face_vertex_normals(), rtol=0, atol=1e-5, err_msg="normals")

    def assertOn(self, mesh, nodes, name):
        """``mesh`` names, by ``parent_uuid``, the node of ``nodes`` called ``name``"""
        self.assertEqual(nodes[mesh.parent_uuid].name, name)
        self.assertIs(mesh.transform(nodes), nodes[name])

    def assertSameSkin(self, got, want):
        self.assertEqual(got.name, want.name)
        self.assertEqual(list(got.influences), list(want.influences))
        np.testing.assert_array_equal(got.weights, want.weights)

    def assertFloat32Clip(self, got, want):
        self.assertEqual(got.take, want.take)
        self.assertEqual((got.start_frame, got.frame_count, got.fps), (want.start_frame, want.frame_count, want.fps))
        self.assertEqual(structure(got), structure(want))
        a, b = layout(got), layout(want)
        self.assertEqual(list(a), list(b))
        for key in b:
            self.assertEqual(a[key].shape, b[key].shape, key)
            np.testing.assert_array_equal(a[key], at_float32(b[key]), err_msg=str(key))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSkin(_Skinned):
    """save_fbx(path, hierarchy, clip, meshes, skins): the hierarchy the bind
    pose, the mesh on its node geo, one skin, one bind pose; load_fbx gives
    the four back"""

    def test_round_trip(self):
        source = bundle()
        # the motion: an even scale on root, spine and arm (each with
        # children), an uneven one on the leaves head, hand and foot
        scale  = np.asarray(source["clip"].frames.scale)
        column = {name: i for i, name in enumerate(source["clip"].name)}
        for name in ("root", "spine", "arm"):
            self.assertGreater(np.ptp(scale[:, column[name]]), 0.1, name)
            np.testing.assert_allclose(scale[:, column[name]], scale[:, column[name], :1].repeat(3, axis=1))
        for name in LEAVES:
            self.assertIn(name, INFLUENCES)
            self.assertNotIn(name, [node.parent_name for node in source["clip"]])
            self.assertTrue(np.all(np.ptp(scale[:, column[name]], axis=1) > 0.05), name)
            self.assertGreater(np.ptp(scale[:, column[name]], axis=0).max(), 0.1, name)

        save_fbx(self.path("a.fbx"), **source)
        found = counts(self.path("a.fbx"))
        self.assertEqual((found["poses"], found["stacks"], found["curves"]), (1, 1, curve_count(source["clip"])))
        first = load_fbx(self.path("a.fbx"))
        self.assertEqual(first._fields, ("hierarchy", "clip", "meshes", "skins", "uvs", "cameras", "lights", "materials", "morphs"))
        self.assertEqual((first.cameras, first.lights, first.materials, first.morphs), (None, None, None, None))
        self.assertSameSkeleton(first.hierarchy, source["hierarchy"])
        self.assertIsInstance(first.meshes, MeshList)
        self.assertIsInstance(first.skins, SkinList)
        self.assertEqual((len(first.meshes), len(first.skins), first.uvs), (1, 1, None))
        self.assertSameMesh(first.meshes[0], source["meshes"])
        self.assertSameSkin(first.skins[0], source["skins"])
        self.assertFloat32Clip(first.clip.active, source["clip"])
        # the mesh on geo, in the bind pose and in the clip alike; its skin
        # naming it
        self.assertOn(first.meshes[0], first.hierarchy, "geo")
        self.assertOn(first.meshes[0], first.clip.active, "geo")
        self.assertEqual(first.skins[0].mesh_uuid, first.meshes[0].uuid)

        save_fbx(self.path("b.fbx"), *first)
        second = load_fbx(self.path("b.fbx"))
        self.assertSameSkeleton(second.hierarchy, first.hierarchy)
        self.assertSameMesh(second.meshes[0], first.meshes[0])
        self.assertSameSkin(second.skins[0], first.skins[0])
        self.assertSameClips(second.clip, first.clip)
        self.assertOn(second.meshes[0], second.hierarchy, "geo")

    def test_the_file_holds_one_skin_its_clusters_and_one_bind_pose(self):
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)
        worlds = np.asarray(source["hierarchy"].world_matrix)
        place  = {name: i for i, name in enumerate(source["hierarchy"].name)}
        with open_fbx(self.path("a.fbx"), FBX) as scene:
            # the mesh is the shape of geo, named after the mesh
            node  = _named(scene, "geo")
            shape = typed(node.GetNodeAttribute(), FBX.FbxMesh, FBX)
            self.assertEqual(str(shape.GetName()), "body")
            kind = FBX.FbxDeformer.EDeformerType.eSkin
            self.assertEqual(shape.GetDeformerCount(kind), 1)
            skin = typed(shape.GetDeformer(0, kind), FBX.FbxSkin, FBX)
            self.assertEqual(str(skin.GetName()), "bodySkin")
            self.assertEqual(skin.GetClusterCount(), len(INFLUENCES))
            for k, name in enumerate(INFLUENCES):
                cluster = skin.GetCluster(k)
                self.assertEqual(str(cluster.GetLink().GetName()), name)
                column = source["skins"].weights[:, k]
                np.testing.assert_array_equal(cluster.GetControlPointIndices(), np.flatnonzero(column))
                np.testing.assert_array_equal(cluster.GetControlPointWeights(), column[column != 0.0])
                linked, placed = FBX.FbxAMatrix(), FBX.FbxAMatrix()
                cluster.GetTransformLinkMatrix(linked)
                cluster.GetTransformMatrix(placed)
                np.testing.assert_allclose(_matrix(linked), worlds[place[name]], rtol=0, atol=1e-12)
                # the cluster transform: the mesh's node, geo, at its world
                np.testing.assert_allclose(_matrix(placed), worlds[place["geo"]], rtol=0, atol=1e-12)
            self.assertEqual(scene.GetPoseCount(), 1)
            pose = scene.GetPose(0)
            self.assertTrue(pose.IsBindPose())
            self.assertEqual(pose.GetCount(), len(BOUND))
            held = {str(pose.GetNode(k).GetName()): _matrix(pose.GetMatrix(k)) for k in range(pose.GetCount())}
            self.assertEqual(sorted(held), sorted(place))
            for name, matrix in held.items():
                np.testing.assert_array_equal(matrix, worlds[place[name]], err_msg=name)
            # the nodes hold the hierarchy's own values, the take moves them
            for name, *_ in BOUND:
                np.testing.assert_array_equal(
                    tuple(_named(scene, name).LclTranslation.Get()), source["hierarchy"][name].translate
                )

    def test_a_skin_without_a_take(self):
        source = bundle()
        del source["clip"]
        save_fbx(self.path("a.fbx"), **source)
        self.assertEqual(counts(self.path("a.fbx"))["curves"], 0)
        rig, clip, meshes, skins, uvs, cameras, lights, materials, morphs = load_fbx(self.path("a.fbx"))
        self.assertEqual((clip, uvs, materials, morphs), (None, None, None, None))
        self.assertSameSkeleton(rig, source["hierarchy"])
        self.assertSameMesh(meshes[0], source["meshes"])
        self.assertSameSkin(skins[0], source["skins"])
        self.assertOn(meshes[0], rig, "geo")

    def test_the_bind_pose_places_nodes_their_own_values_do_not(self):
        """Maya's FBXExport writes the nodes at the playhead: the bind pose's
        matrices give the hierarchy back, the skinned mesh's node where the
        pose and the clusters' transform put it (identity, for geo)"""
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)

        def moved(scene):
            _named(scene, "spine").LclRotation.Set(FBX.FbxDouble3(40.0, -20.0, 7.0))
            _named(scene, "root").LclScaling.Set(FBX.FbxDouble3(2.0, 2.0, 2.0))
            _named(scene, "hand").LclScaling.Set(FBX.FbxDouble3(3.0, 1.0, 1.0))
            _named(scene, "geo").LclTranslation.Set(FBX.FbxDouble3(0.0, 1.0, 0.0))

        got = load_fbx(_edited(self.path("a.fbx"), self.path("moved.fbx"), moved))
        self.assertSameSkeleton(got.hierarchy, source["hierarchy"], exact=False, atol=1e-9)
        np.testing.assert_allclose(got.hierarchy.world_matrix, source["hierarchy"].world_matrix, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(got.meshes[0].world_matrix(got.hierarchy), np.eye(4))

    def test_mesh_alone_and_beside_a_clip_or_a_hierarchy(self):
        # a mesh on no node: the shape of a root node at identity named after it
        mesh = body()
        self.assertIsNone(mesh.parent_uuid)
        save_fbx(self.path("mesh.fbx"), meshes=mesh)
        self.assertEqual(counts(self.path("mesh.fbx")), {"poses": 0, "stacks": 0, "layers": 0, "curve nodes": 0, "curves": 0})
        got = load_fbx(self.path("mesh.fbx"))
        self.assertEqual((got.clip, got.skins, got.uvs), (None, None, None))
        self.assertEqual(got.hierarchy.name, ["body"])
        node = got.hierarchy["body"]
        self.assertEqual((node.node_type, node.parent_name), ("transform", None))
        np.testing.assert_array_equal(node.world_matrix, np.eye(4))
        self.assertSameMesh(got.meshes[0], mesh)
        self.assertOn(got.meshes[0], got.hierarchy, "body")
        np.testing.assert_array_equal(got.meshes[0].world_matrix(got.hierarchy), np.eye(4))
        save_fbx(self.path("again.fbx"), *got)
        again = load_fbx(self.path("again.fbx"))
        self.assertSameMesh(again.meshes[0], got.meshes[0])
        self.assertOn(again.meshes[0], again.hierarchy, "body")
        # the class routes
        mesh.save_fbx(self.path("route.fbx"))
        self.assertSameMesh(load_fbx(self.path("route.fbx")).meshes[0], mesh)
        MeshList([mesh, tetra()]).save_fbx(self.path("routes.fbx"))
        got = load_fbx(self.path("routes.fbx"))
        self.assertEqual((got.hierarchy.name, [m.name for m in got.meshes]), (["body", "tetra"], ["body", "tetra"]))

        # on a node of the clip
        clip = motion(bound())
        mesh = body(clip["geo"].uuid)
        save_fbx(self.path("clip.fbx"), clip=clip, meshes=mesh)
        found = counts(self.path("clip.fbx"))
        self.assertEqual((found["poses"], found["curves"]), (0, curve_count(clip)))
        rig, back, held, skins, uvs, cameras, lights, materials, morphs = load_fbx(self.path("clip.fbx"))
        self.assertEqual((cameras, lights, materials, morphs), (None, None, None, None))
        self.assertEqual((rig, skins, uvs), (None, None, None))
        self.assertFloat32Clip(back.active, clip)
        self.assertSameMesh(held[0], mesh)
        self.assertOn(held[0], back.active, "geo")
        self.assertEqual(ClipData.load_fbx(self.path("clip.fbx")).name, back.active.name)
        save_fbx(self.path("clip2.fbx"), clip=back, meshes=held)
        again = load_fbx(self.path("clip2.fbx"))
        self.assertSameClips(again.clip, back)
        self.assertSameMesh(again.meshes[0], held[0])
        self.assertOn(again.meshes[0], again.clip.active, "geo")

        # on a node of the hierarchy
        rig  = bound()
        mesh = body(rig["geo"].uuid)
        save_fbx(self.path("rig.fbx"), rig, meshes=mesh)
        got = load_fbx(self.path("rig.fbx"))
        self.assertEqual((got.clip, got.skins), (None, None))
        self.assertSameSkeleton(got.hierarchy, rig)
        self.assertSameMesh(got.meshes[0], mesh)
        self.assertOn(got.meshes[0], got.hierarchy, "geo")

    def test_two_meshes_on_two_nodes_one_animated(self):
        """body() on geo and tetra() on prop, a leaf transform the clip turns
        and scales unevenly: both back on their nodes, the second cycle bit
        for bit; a mesh naming a node the file does not hold goes on a root"""
        clip   = motion(bound())
        meshes = MeshList([body(clip["geo"].uuid), tetra(clip["prop"].uuid)])
        self.assertGreater(np.ptp(np.asarray(clip.frames.world_matrix)[:, clip.name.index("prop")]), UNIT)
        save_fbx(self.path("a.fbx"), clip=clip, meshes=meshes)
        first = load_fbx(self.path("a.fbx"))
        self.assertEqual((first.hierarchy, first.skins, first.uvs), (None, None, None))
        self.assertFloat32Clip(first.clip.active, clip)
        # the meshes come back in their nodes' order: prop ahead of geo
        self.assertEqual([mesh.name for mesh in first.meshes], ["tetra", "body"])
        for got, want, name in zip(first.meshes, meshes[::-1], ("prop", "geo")):
            self.assertSameMesh(got, want)
            self.assertOn(got, first.clip.active, name)
        # tetra()'s normals, one a face, are the ones Maya derives with every
        # edge hard: not stored, its six edges hard; body()'s, authored, stay
        self.assertIsNone(first.meshes[0].normals)
        self.assertIsNone(first.meshes[0].normal_indices)
        self.assertEqual(
            first.meshes[0].hard_edges.tolist(), [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        )
        self.assertEqual(len(first.meshes[1].normals), len(meshes[0].normals))
        save_fbx(self.path("b.fbx"), *first)
        second = load_fbx(self.path("b.fbx"))
        self.assertSameClips(second.clip, first.clip)
        for got, want, name in zip(second.meshes, first.meshes, ("prop", "geo")):
            self.assertSameMesh(got, want)
            self.assertOn(got, second.clip.active, name)
        # the node readers keep the meshes' nodes
        self.assertEqual(HierarchyData.load_fbx(self.path("a.fbx")).name, clip.name)
        self.assertEqual(ClipData.load_fbx(self.path("a.fbx")).name, clip.name)
        # a mesh naming a node of another rig: a root node of its own
        stray = MeshList([body(clip["geo"].uuid), tetra(bound()["prop"].uuid)])
        save_fbx(self.path("stray.fbx"), clip=clip, meshes=stray)
        got = load_fbx(self.path("stray.fbx"))
        self.assertEqual(got.clip.active.name, clip.name + ["tetra"])
        self.assertOn(got.meshes[0], got.clip.active, "geo")
        self.assertOn(got.meshes[1], got.clip.active, "tetra")
        np.testing.assert_array_equal(got.meshes[1].world_matrix(got.clip.active), np.eye(4))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSkinRefusals(_Skinned):
    def refused(self, error, message, **kwargs):
        with self.assertRaisesRegex(error, message):
            save_fbx(self.path("refused.fbx"), **kwargs)
        self.assertFalse(os.path.exists(self.path("refused.fbx")))

    def test_a_skin_needs_its_hierarchy_and_its_mesh(self):
        source = bundle()
        self.refused(TypeError, r"missing: hierarchy \(a HierarchyData, the bind pose\)$", meshes=source["meshes"], skins=source["skins"])
        self.refused(TypeError, r"missing: meshes=$", hierarchy=source["hierarchy"], skins=source["skins"])
        self.refused(TypeError, "missing: hierarchy .* and meshes=", clip=source["clip"], skins=source["skins"])
        self.refused(TypeError, "the meshes are a MeshList or a MeshData, not a list",   meshes=[source["meshes"]])
        self.refused(TypeError, "the meshes are a MeshList or a MeshData, not a UVList", meshes=maps(source["meshes"]))
        self.refused(TypeError, "the meshes are a MeshList or a MeshData, not a UVData", meshes=maps(source["meshes"])[0])
        self.refused(TypeError, "the skins are a SkinList or a SkinData, not a MeshData", hierarchy=source["hierarchy"], meshes=source["meshes"], skins=source["meshes"])
        self.refused(TypeError, "the skins are a SkinList or a SkinData, not a list", hierarchy=source["hierarchy"], meshes=source["meshes"], skins=[source["skins"]])
        # a hierarchy and a clip without a skin
        self.refused(NotImplementedError, "comes with skins", hierarchy=source["hierarchy"], clip=source["clip"], meshes=source["meshes"])
        self.refused(NotImplementedError, "comes with skins", hierarchy=source["hierarchy"], clip=source["clip"], meshes=source["meshes"], skins=SkinList([]))

    def test_influences_and_weights(self):
        source          = bundle()
        skin            = source["skins"]
        skin.influences = ["hips", "spine", "chest", "head", "arm", "hand", "leg", "toe"]
        self.refused(ValueError, r"skin 'bodySkin' names no node of the hierarchy: influences \['toe'\]", **source)
        source                  = bundle()
        source["skins"].weights = source["skins"].weights[:-1]
        self.refused(ValueError, "skin 'bodySkin' channel 'weights' holds 287 rows, mesh 'body' holds 288 points", **source)
        source                  = bundle()
        source["skins"].weights = source["skins"].weights[:, :-1]
        self.refused(ValueError, "holds 7 columns for its 8 influences", **source)

    def test_the_skin_s_mesh_and_node(self):
        source                    = bundle()
        source["skins"].mesh_uuid = None
        self.refused(ValueError, "skin 'bodySkin' names no mesh of meshes=: mesh_uuid None", **source)
        source                    = bundle()
        source["skins"].mesh_uuid = generate_uuid()
        self.refused(ValueError, "skin 'bodySkin' names no mesh of meshes=: mesh_uuid '", **source)
        # one skin a mesh
        source          = bundle()
        source["skins"] = SkinList([source["skins"], weighed(source["meshes"])])
        self.refused(ValueError, "skins 'bodySkin' and 'bodySkin' both name mesh 'body'; a mesh holds one skin", **source)
        # on no node: a root node at identity, bound
        source           = bundle()
        source["meshes"] = body()
        source["skins"]  = weighed(source["meshes"])
        save_fbx(self.path("rooted.fbx"), **source)
        got = load_fbx(self.path("rooted.fbx"))
        self.assertEqual(got.hierarchy.name, [row[0] for row in BOUND] + ["body"])
        self.assertOn(got.meshes[0], got.hierarchy, "body")
        self.assertSameSkin(got.skins[0], source["skins"])
        self.assertEqual(counts(self.path("rooted.fbx"))["poses"], 1)

    def test_a_skinned_mesh_on_a_node_anywhere(self):
        """body() on placed, a root transform off the origin, turned and
        scaled, still over the clip: every cluster's transform and the bind
        pose's entry are its world, the read puts the mesh back on it, the
        bind pose places the node whatever its own values say, and the
        second cycle is bit for bit"""
        rig    = placed()
        mesh   = body(rig["placed"].uuid)
        source = {"hierarchy": rig, "clip": motion(rig), "meshes": mesh, "skins": weighed(mesh)}
        world  = np.asarray(rig["placed"].world_matrix)
        self.assertGreater(float(np.abs(world - np.eye(4)).max()), 1.0)
        save_fbx(self.path("a.fbx"), **source)
        with open_fbx(self.path("a.fbx"), FBX) as scene:
            shape = typed(_named(scene, "placed").GetNodeAttribute(), FBX.FbxMesh, FBX)
            skin  = typed(shape.GetDeformer(0, FBX.FbxDeformer.EDeformerType.eSkin), FBX.FbxSkin, FBX)
            self.assertEqual(skin.GetClusterCount(), len(INFLUENCES))
            for k in range(skin.GetClusterCount()):
                bound_at = FBX.FbxAMatrix()
                skin.GetCluster(k).GetTransformMatrix(bound_at)
                np.testing.assert_allclose(_matrix(bound_at), world, rtol=0, atol=1e-9, err_msg=str(k))
            pose = scene.GetPose(0)
            held = {str(pose.GetNode(k).GetName()): _matrix(pose.GetMatrix(k)) for k in range(pose.GetCount())}
            np.testing.assert_allclose(held["placed"], world, rtol=0, atol=1e-12)

        first = load_fbx(self.path("a.fbx"))
        self.assertSameSkeleton(first.hierarchy, rig)
        self.assertOn(first.meshes[0], first.hierarchy, "placed")
        self.assertOn(first.meshes[0], first.clip.active, "placed")
        self.assertSameMesh(first.meshes[0], mesh)
        self.assertSameSkin(first.skins[0], source["skins"])
        np.testing.assert_allclose(first.meshes[0].world_matrix(first.hierarchy), world, rtol=0, atol=1e-12)

        def moved(scene):
            _named(scene, "placed").LclTranslation.Set(FBX.FbxDouble3(0.0, 1.0, 0.0))
            _named(scene, "chest").LclRotation.Set(FBX.FbxDouble3(40.0, -20.0, 7.0))

        got = load_fbx(_edited(self.path("a.fbx"), self.path("moved.fbx"), moved))
        self.assertSameSkeleton(got.hierarchy, rig, exact=False, atol=1e-9)
        np.testing.assert_allclose(got.meshes[0].world_matrix(got.hierarchy), world, rtol=0, atol=1e-9)

        save_fbx(self.path("b.fbx"), *first)
        second = load_fbx(self.path("b.fbx"))
        self.assertSameSkeleton(second.hierarchy, first.hierarchy)
        self.assertSameClips(second.clip, first.clip)
        self.assertSameMesh(second.meshes[0], first.meshes[0])
        self.assertSameSkin(second.skins[0], first.skins[0])
        self.assertOn(second.meshes[0], second.hierarchy, "placed")

        # on arm_grp, under the joints the clip moves: bound without a clip,
        # refused with one (measured: Maya's skinCluster applies the node's
        # own transform on top of the skin)
        source           = bundle()
        source["meshes"] = body(source["hierarchy"]["arm_grp"].uuid)
        source["skins"]  = weighed(source["meshes"])
        self.refused(ValueError, "the clip moves node 'arm_grp', which holds skinned mesh 'body', off its bind world on frame 1001", **source)
        del source["clip"]
        save_fbx(self.path("still.fbx"), **source)
        got = load_fbx(self.path("still.fbx"))
        self.assertOn(got.meshes[0], got.hierarchy, "arm_grp")
        self.assertSameSkeleton(got.hierarchy, source["hierarchy"])

    def test_the_mesh(self):
        mesh              = body()
        mesh.hole_faces   = np.array([0],       dtype=np.int32)
        mesh.hole_counts  = np.array([3],       dtype=np.int32)
        mesh.hole_indices = np.array([0, 1, 2], dtype=np.int32)
        self.refused(ValueError, "mesh 'body' holds 1 polygon holes", meshes=mesh)
        # a node holds one mesh, and a joint none (measured: Maya's FBXExport
        # and FBXImport both drop a mesh on a joint)
        rig = bound()
        self.refused(ValueError, "mesh 'body' and mesh 'tetra' sit on node 'geo'; a node holds one mesh", hierarchy=rig, meshes=MeshList([body(rig["geo"].uuid), tetra(rig["geo"].uuid)]))
        self.refused(ValueError, "mesh 'tetra' sits on joint 'hand'; Maya's FBX holds a mesh on a transform only", hierarchy=rig, meshes=tetra(rig["hand"].uuid))
        self.refused(ValueError, "the unnamed mesh on node 'hand' sits on joint 'hand'", clip=motion(rig), meshes=MeshData(points=tetra().points, counts=tetra().counts, indices=tetra().indices, parent_uuid=rig["hand"].uuid))
        # two meshes on no node are two root nodes
        save_fbx(self.path("two.fbx"), hierarchy=rig, meshes=MeshList([body(), tetra()]))
        self.assertEqual(load_fbx(self.path("two.fbx")).hierarchy.name, [row[0] for row in BOUND] + ["body", "tetra"])

    def test_a_clip_that_is_not_the_skeleton_s(self):
        def other(edit):
            source = bundle()
            rig    = bound()
            edit(rig)
            source["clip"] = motion(rig)
            return source

        for node, channel, edit in (
            ("chest", "rotate_order", lambda rig: setattr(rig["chest"], "rotate_order", 0)),
            ("chest", "joint_orient", lambda rig: setattr(rig["chest"], "joint_orient", [5.0, 0.0, 1.0])),
            ("chest", "rotate_axis", lambda rig: setattr(rig["chest"], "rotate_axis", [0.0, 1.0, 0.0])),
            ("chest", "segment_scale_compensate", lambda rig: setattr(rig["chest"], "segment_scale_compensate", False)),
            ("arm_grp", "node_type", lambda rig: setattr(rig["arm_grp"], "node_type", "joint")),
            ("chest", "user attributes", lambda rig: rig["chest"].add_user_attribute("extra", 1.0, attribute_type="double")),
            ("foot", "parent", lambda rig: setattr(rig["foot"], "parent_uuid", rig["hips"].uuid)),
        ):
            with self.subTest(channel):
                self.refused(ValueError, f"node '{node}' channel '{channel}'", **other(edit))
        source                     = bundle()
        source["clip"]["leg"].name = "thigh"
        self.refused(ValueError, "the clip's nodes .* are not the hierarchy's", **source)

    def write_bad(self, name: str, edit) -> str:
        save_fbx(self.path("good.fbx"), **bundle())
        return _edited(self.path("good.fbx"), self.path(name), edit)

    def test_load_refusals(self):
        def two_poses(scene):
            pose = FBX.FbxPose.Create(scene, "again")
            pose.SetIsBindPose(True)
            pose.Add(_named(scene, "geo"), FBX.FbxMatrix())
            scene.AddPose(pose)

        def two_skins(scene):
            shape   = typed(_named(scene, "geo").GetNodeAttribute(), FBX.FbxMesh, FBX)
            skin    = FBX.FbxSkin.Create(scene, "again")
            cluster = FBX.FbxCluster.Create(scene, "again")
            cluster.SetLink(_named(scene, "hips"))
            cluster.AddControlPointIndex(0, 1.0)
            skin.AddCluster(cluster)
            shape.AddDeformer(skin)

        def bound_off(scene):
            """the bind pose lists geo, the skinned mesh's node, elsewhere
            than the clusters' transform does"""
            old  = scene.GetPose(0)
            pose = FBX.FbxPose.Create(scene, "bind")
            pose.SetIsBindPose(True)
            for k in range(old.GetCount()):
                node   = old.GetNode(k)
                matrix = old.GetMatrix(k)
                if str(node.GetName()) == "geo":
                    matrix.Set(3, 1, 1.0)
                pose.Add(node, matrix)
            self.assertTrue(scene.RemovePose(0))
            scene.AddPose(pose)

        def cluster_off(scene):
            """one cluster's transform elsewhere than the others'"""
            shape = typed(_named(scene, "geo").GetNodeAttribute(), FBX.FbxMesh, FBX)
            skin  = typed(shape.GetDeformer(0, FBX.FbxDeformer.EDeformerType.eSkin), FBX.FbxSkin, FBX)
            moved = FBX.FbxAMatrix()
            moved.SetT(FBX.FbxVector4(0.0, 1.0, 0.0, 1.0))
            skin.GetCluster(2).SetTransformMatrix(moved)

        def offset(scene):
            _named(scene, "geo").SetGeometricRotation(FBX.FbxNode.EPivotSet.eSourcePivot, FBX.FbxVector4(0.0, 90.0, 0.0))

        for name, edit, message in (
            ("two_poses", two_poses, "mesh 'body' is listed by 2 bind poses"),
            ("two_skins", two_skins, "mesh 'body' holds 2 skin deformers"),
            ("bound_off", bound_off, "node 'geo' channel 'bind pose' is .* in the bind pose of mesh 'body', .* in skin 'bodySkin''s cluster transform"),
            ("cluster_off", cluster_off, "node 'geo' channel 'bind pose' is .* in the bind pose of mesh 'body', .* in skin 'bodySkin''s cluster transform"),
            ("offset", offset, "mesh 'body' channel 'geometric transform'"),
        ):
            with self.subTest(name):
                path = self.write_bad(f"{name}.fbx", edit)
                with self.assertRaisesRegex(ValueError, message):
                    load_fbx(path)
                # the clip and node readers leave the meshes and skins unread
                good = load_fbx(self.path("good.fbx"))
                self.assertSameClip(ClipData.load_fbx(path), good.clip.active)
                self.assertEqual(HierarchyData.load_fbx(path).name, [row[0] for row in BOUND])


def _matrix(matrix) -> np.ndarray:
    return np.array([[matrix.Get(row, column) for column in range(4)] for row in range(4)])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaSkin(_Recorded, _Skinned):
    """Maya 2025's FBXImport of a skinned save: one skinCluster and one bind
    pose deforming the mesh as cgmath's own skinning does, frame by frame,
    as recorded"""

    @staticmethod
    def skinned(on: str) -> dict:
        """save_fbx's arguments: body() on node ``on`` of placed(), geo at
        identity or placed off it, and its skin"""
        rig  = placed()
        mesh = body(rig[on].uuid)
        return {"hierarchy": rig, "clip": motion(rig), "meshes": mesh, "skins": weighed(mesh)}

    def test_maya_imports_one_skin_that_deforms_as_cgmath(self):
        self.deforms_as_cgmath("geo", "skin_geo")

    def test_maya_imports_a_skin_on_a_placed_node_that_deforms_as_cgmath(self):
        """the mesh on placed, off the origin, turned and scaled at the bind:
        Maya's skinCluster takes the bind placement as its geomMatrix and
        deforms as cgmath's, which takes it from the node"""
        self.deforms_as_cgmath("placed", "skin_placed")

    def deforms_as_cgmath(self, on: str, case: str) -> None:
        from cgmath.geometry.deform import SkinDeformData

        held   = self.case(case)
        source = self.skinned(on)
        self.assertFreshSave(self.fixture("ours.fbx"), **source)
        back = load_fbx(self.fixture("ours.fbx"))

        # the mesh is the node's shape
        self.assertEqual(held.value("skinClusters"), 1)
        self.assertTrue(held.value("history_has_cluster"))
        self.assertEqual(held.value("dagPoses"), 1)
        self.assertTrue(held.value("bindPose"))

        influences = held.value("influences")
        self.assertEqual(influences, list(source["skins"].influences))
        np.testing.assert_allclose(held.array("weights"), source["skins"].weights, rtol=0, atol=1e-6)
        worlds = np.asarray(source["hierarchy"].world_matrix)
        for k, name in enumerate(influences):
            expected = np.linalg.inv(worlds[source["hierarchy"].name.index(name)])
            near(held.array("bindPreMatrix")[k], expected, err_msg=name)
        # the bind placement: the mesh node's world as the cluster's geomMatrix
        near(held.array("geomMatrix"), worlds[source["hierarchy"].name.index(on)], err_msg="geomMatrix")
        self.assertOn(back.meshes[0], back.hierarchy, on)

        # THE SKIN DEFORMATION: cgmath's linear blend skinning of the re-read
        # mesh, by the re-read clip, bound to the re-read hierarchy, which
        # places the mesh's node
        deformer = SkinDeformData(back.meshes[0], back.skins[0], bind_rig=back.hierarchy)
        wanted   = deformer.apply(back.clip.active.frames)
        self.assertEqual(wanted.shape, (TAKE, source["meshes"].point_count, 3))
        points = held.array("points")
        for f in range(TAKE):
            near(points[f], wanted[f], err_msg=f"frame {START + f}")
        # the deformation is no stand-still: the points move by whole units
        self.assertGreater(float(np.abs(wanted[-1] - source["meshes"].points).max()), 2.0 * UNIT)

    def test_maya_imports_two_meshes_on_their_nodes(self):
        """body() on geo, still, and tetra() on prop, which the clip turns
        and scales unevenly: each the shape of its node, its points and
        normals as the mesh holds them, in the world where its node's frame
        puts them, within 1e-6 of load_fbx's read of the same file"""
        held   = self.case("two_meshes")
        clip   = motion(bound())
        meshes = MeshList([body(clip["geo"].uuid), tetra(clip["prop"].uuid)])
        self.assertFreshSave(self.fixture("ours.fbx"), clip=clip, meshes=meshes)
        back = load_fbx(self.fixture("ours.fbx"))
        self.assertEqual(held.value("counts")["skinCluster"], 0)
        self.assertEqual(held.value("counts")["dagPose"],     0)
        self.assertEqual(held.value("counts")["animCurve"],   curve_count(clip))
        worlds = np.asarray(back.clip.active.frames.world_matrix)
        column = {name: i for i, name in enumerate(back.clip.active.name)}
        self.assertEqual([mesh.name for mesh in back.meshes], ["tetra", "body"])
        for mesh, name in zip(meshes, ("geo", "prop")):
            self.assertIsNone(held.value(f"{name}/curves"))
            np.testing.assert_allclose(held.array(f"{name}/points"), mesh.points, rtol=0, atol=1e-6, err_msg=name)
            np.testing.assert_array_equal(held.array(f"{name}/normal_counts"), mesh.counts, err_msg=name)
            normals = held.array(f"{name}/normals")[held.array(f"{name}/normal_ids")]
            np.testing.assert_allclose(normals, mesh.normals[mesh.normal_indices], rtol=0, atol=1e-6, err_msg=name)
            for f in (0, TAKE - 1):
                want = (np.c_[mesh.points, np.ones(len(mesh.points))] @ worlds[f, column[name]])[:, :3]
                near(held.array(f"{name}/world_points/{f}"), want, err_msg=(name, START + f))
        # prop moves: its mesh is elsewhere on the last frame
        prop = worlds[:, column["prop"]]
        self.assertGreater(float(np.abs(prop[-1] - prop[0]).max()), 0.1 * UNIT)
        np.testing.assert_array_equal(worlds[:, column["geo"]], np.broadcast_to(np.eye(4), (TAKE, 4, 4)))

    def test_maya_export_of_the_skin_reads_back(self):
        self.export_reads_back("geo", "skin_geo")

    def test_maya_export_of_a_skin_on_a_placed_node_reads_back(self):
        self.export_reads_back("placed", "skin_placed")

    def export_reads_back(self, on: str, case: str) -> None:
        """Maya's FBXExport of the skinned import: load, save, load gives the
        first load back bit for bit; the first is the source within 1e-6,
        the mesh on its node"""
        self.case(case)
        source = self.skinned(on)
        out    = self.fixture("maya.fbx")

        first  = load_fbx(out)
        save_fbx(self.path("again.fbx"), *first)
        second = load_fbx(self.path("again.fbx"))
        self.assertSameSkeleton(second.hierarchy, first.hierarchy)
        self.assertSameClips(second.clip, first.clip)
        self.assertSameMesh(second.meshes[0], first.meshes[0])
        self.assertSameSkin(second.skins[0], first.skins[0])
        self.assertOn(first.meshes[0], first.hierarchy, on)
        self.assertOn(second.meshes[0], second.hierarchy, on)

        # Maya writes its bind pose in float32: the angles within 1e-6
        # radians
        rig = source["hierarchy"]
        self.assertEqual([n.name for n in first.hierarchy], [n.name for n in rig])
        for channel in ("translate", "rotate", "scale", "joint_orient", "rotate_axis"):
            got, want = np.asarray(getattr(first.hierarchy, channel)), np.asarray(getattr(rig, channel))
            if channel in ("rotate", "joint_orient", "rotate_axis"):
                got, want = np.radians(got), np.radians(want)
            near(got, want, err_msg=channel)
        near(first.hierarchy.world_matrix, rig.world_matrix)
        self.assertEqual(sorted(first.skins[0].influences), sorted(INFLUENCES))
        columns = [list(first.skins[0].influences).index(x) for x in INFLUENCES]
        np.testing.assert_allclose(first.skins[0].weights[:, columns], source["skins"].weights, rtol=0, atol=1e-6)
        near(first.meshes[0].points, source["meshes"].points)
        self.assertEqual((first.clip.start_frame, first.clip.frame_count, first.clip.fps), ([START], [TAKE], [FPS]))


# --------------------------------- UV sets --------------------------------- #


def maps(mesh: MeshData) -> UVList:
    """two UV sets on body(): "map1" a cylinder unrolled, a seam column
    where the tube closes, "lightmap" an island a face, its UVs shuffled;
    float64 UVs, which fbx holds as they are; each naming ``mesh`` by its
    uuid"""
    rings, around = 24, 12
    columns = around + 1
    grid = np.stack(
        [np.tile(np.arange(columns) / around, rings), np.repeat(np.linspace(0.0, 1.0, rings), columns)], axis=1
    ) + 1.0 / 3.0e5
    corners = (mesh.indices // around) * columns + mesh.indices % around
    # a face spanning the closing edge takes the seam column for its
    # corners on the first column
    start = 0
    for size in mesh.counts:
        face = corners[start : start + size]
        if (face % columns).max() == around - 1 and (face % columns).min() == 0:
            face[face % columns == 0] += around
        start += size
    first  = UVData(name="map1", points=grid, counts=mesh.counts.copy(), indices=corners.astype(np.int32), mesh_uuid=mesh.uuid)

    rng    = np.random.default_rng(11)
    total  = int(mesh.counts.sum())
    order  = rng.permutation(total)
    points = rng.random((total, 2)) * np.array([0.98, 0.97]) + 0.01
    second = UVData(name="lightmap", points=points, counts=mesh.counts.copy(), indices=order.astype(np.int32), mesh_uuid=mesh.uuid)
    return UVList([first, second])


def islands(mesh: MeshData) -> UVData:
    """one UV set "map1" on any mesh: an island a face, its UVs in corner
    order; naming ``mesh`` by its uuid"""
    total  = int(mesh.counts.sum())
    rng    = np.random.default_rng(13)
    points = rng.random((total, 2)) * 0.9 + 0.05
    return UVData(name="map1", points=points, counts=mesh.counts.copy(), indices=np.arange(total, dtype=np.int32), mesh_uuid=mesh.uuid)


def per_corner(uv: UVData) -> np.ndarray:
    """(corners, 2) the UV at each face corner"""
    return np.asarray(uv.points)[np.asarray(uv.indices)]


class _Mapped(_Skinned):
    def assertSameUVs(self, got, want, mesh):
        """``got`` the sets ``want`` holds, each naming ``mesh``, the loaded mesh, by its uuid"""
        self.assertIsInstance(got, UVList)
        self.assertEqual([uv.name for uv in got], [uv.name for uv in want])
        for a, b in zip(got, want):
            self.assertIsInstance(a, UVData)
            self.assertEqual(a.mesh_uuid, mesh.uuid)
            for field in ("points", "counts", "indices"):
                x, y = getattr(a, field), getattr(b, field)
                self.assertEqual(x.dtype, y.dtype, f"{a.name} {field}")
                np.testing.assert_array_equal(x, y, err_msg=f"{a.name} {field}")


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestShapeRoutes(_Mapped):
    """the mesh, UV and skin classes' fbx routes read what load_fbx reads"""

    def test_mesh_routes(self):
        path = self.path("bundle.fbx")
        save_fbx(path, **bundle())
        held = load_fbx(path)
        self.assertEqual(MeshList.load_fbx(path), held.meshes)
        self.assertEqual(MeshData.load_fbx(path), held.meshes[0])
        self.assertEqual(MeshData.load_fbx(path, name="body"), held.meshes[0])
        self.assertIsNotNone(MeshData.load_fbx(path).parent_uuid)  # its node's, in the hierarchy load_fbx gives
        with self.assertRaisesRegex(ValueError, r"^Mesh 'nope' not found; available: \['body'\]$"):
            MeshData.load_fbx(path, name="nope")
        pairs = mesh_module.load_fbx(path)
        self.assertEqual([mesh for mesh, _ in pairs], list(held.meshes))
        self.assertEqual(pairs[0][1], UVList())  # body() has no UV set
        # load() routes an fbx here
        self.assertEqual(MeshData.load(path), held.meshes[0])
        self.assertEqual(MeshList.load(path), held.meshes)

    def test_skin_routes(self):
        path = self.path("bundle.fbx")
        save_fbx(path, **bundle())
        held = load_fbx(path)
        self.assertEqual(SkinList.load_fbx(path), held.skins)
        self.assertEqual(skin_module.load_fbx(path), held.skins)
        self.assertEqual(SkinList.load(path), held.skins)
        self.assertEqual(SkinData.load_fbx(path), held.skins[0])
        self.assertEqual(SkinData.load_fbx(path, name="body"), held.skins[0])  # by the mesh's name
        self.assertEqual(SkinData.load_fbx(path).name, "bodySkin")             # the deformer's
        with self.assertRaisesRegex(ValueError, r"^Skin of mesh 'nope' not found; available: \['body'\]$"):
            SkinData.load_fbx(path, name="nope")

    def same_sets(self, got, want):
        """``got`` the sets ``want`` holds, all on one mesh (each load mints
        its own uuids, so the mesh is not compared across loads)"""
        self.assertIsInstance(got, UVList)
        self.assertEqual([uv.name for uv in got], [uv.name for uv in want])
        self.assertEqual(len({uv.mesh_uuid for uv in got}), 1 if len(got) else 0)
        for a, b in zip(got, want):
            for field in ("points", "counts", "indices"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=f"{a.name} {field}")

    def test_uv_routes_read_one_meshs_sets(self):
        mesh = body()
        path = self.path("mapped.fbx")
        save_fbx(path, meshes=mesh, uvs=maps(mesh))
        held = load_fbx(path)
        self.same_sets(UVList.load_fbx(path), held.uvs)
        self.same_sets(UVList.load_fbx(path, name="body"), held.uvs)
        self.same_sets(UVList([UVData.load_fbx(path)]), held.uvs[:1])
        self.assertEqual(UVData.load_fbx(path, channel=1).name, "lightmap")
        self.same_sets(UVList([UVData.load_fbx(path, channel=1)]), held.uvs[1:])
        with self.assertRaises(IndexError):
            UVData.load_fbx(path, channel=2)
        with self.assertRaisesRegex(ValueError, "^Mesh 'nope' not found"):
            UVList.load_fbx(path, name="nope")
        # the module reader pairs each mesh with its sets, from one load
        pair_mesh, pair_sets = mesh_module.load_fbx(path)[0]
        self.assertSameUVs(pair_sets, held.uvs, pair_mesh)

    def test_routes_on_a_file_of_no_mesh(self):
        path = self.path("rig.fbx")
        skeleton().save_fbx(path)
        self.assertEqual(MeshList.load_fbx(path),    MeshList())
        self.assertEqual(SkinList.load_fbx(path),    SkinList())
        self.assertEqual(mesh_module.load_fbx(path), [])
        with self.assertRaisesRegex(RuntimeError, "^No meshes found in FBX file$"):
            MeshData.load_fbx(path)
        with self.assertRaisesRegex(RuntimeError, "^No meshes found in FBX file$"):
            UVList.load_fbx(path)
        with self.assertRaisesRegex(RuntimeError, "^No skinned meshes found in FBX file$"):
            SkinData.load_fbx(path)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestUVs(_Mapped):
    """save_fbx(..., meshes, uvs=): each mesh's UV sets, a UVList or one
    UVData, naming their mesh; load_fbx gives them back as FbxContents.uvs,
    None for none"""

    def test_round_trip_beside_a_skin(self):
        source = bundle()
        uvs    = maps(source["meshes"])
        # the seam: map1 holds more UVs than the mesh points; lightmap's
        # indices are shuffled
        self.assertFalse(np.array_equal(uvs[0].indices, source["meshes"].indices))
        self.assertGreater(len(uvs[0].points), source["meshes"].point_count)
        self.assertFalse(np.array_equal(uvs[1].indices, np.arange(len(uvs[1].indices))))
        save_fbx(self.path("a.fbx"), **source, uvs=uvs)
        first = load_fbx(self.path("a.fbx"))
        self.assertSameUVs(first.uvs, uvs, first.meshes[0])
        self.assertSameMesh(first.meshes[0], source["meshes"])
        self.assertSameSkin(first.skins[0], source["skins"])
        self.assertEqual(first.skins[0].mesh_uuid, first.meshes[0].uuid)
        # the five fields are save_fbx's arguments in their order
        save_fbx(self.path("b.fbx"), *first)
        second = load_fbx(self.path("b.fbx"))
        self.assertSameUVs(second.uvs, first.uvs, second.meshes[0])

    def test_a_uvdata_is_one_set_and_none_reads_none(self):
        mesh = body()
        uv   = maps(mesh)[1]
        save_fbx(self.path("one.fbx"), meshes=mesh, uvs=uv)
        got = load_fbx(self.path("one.fbx"))
        self.assertSameUVs(got.uvs, UVList([uv]), got.meshes[0])
        for name, kwargs in (
            ("mesh", {"meshes": mesh}),
            ("empty", {"meshes": mesh, "uvs": UVList([])}),
            ("clip", {"clip": motion(bound())}),
            ("rig", {"hierarchy": bound()}),
        ):
            with self.subTest(name):
                save_fbx(self.path(f"{name}.fbx"), **kwargs)
                self.assertIsNone(load_fbx(self.path(f"{name}.fbx")).uvs)

    def test_sets_on_two_meshes(self):
        """body()'s two sets and tetra()'s one, "map1" on both meshes: back
        in the file's order, each naming its mesh"""
        meshes = MeshList([body(), tetra()])
        uvs    = UVList([islands(meshes[1]), *maps(meshes[0])])
        save_fbx(self.path("a.fbx"), meshes=meshes, uvs=uvs)
        got = load_fbx(self.path("a.fbx"))
        self.assertEqual([uv.name for uv in got.uvs], ["map1", "lightmap", "map1"])
        self.assertSameUVs(got.uvs[:2], maps(meshes[0]), got.meshes[0])
        self.assertSameUVs(got.uvs[2:], UVList([islands(meshes[1])]), got.meshes[1])
        save_fbx(self.path("b.fbx"), *got)
        again = load_fbx(self.path("b.fbx"))
        self.assertSameUVs(again.uvs[:2], got.uvs[:2], again.meshes[0])
        self.assertSameUVs(again.uvs[2:], got.uvs[2:], again.meshes[1])

    def refused(self, error, message, **kwargs):
        with self.assertRaisesRegex(error, message):
            save_fbx(self.path("refused.fbx"), **kwargs)
        self.assertFalse(os.path.exists(self.path("refused.fbx")))

    def test_save_refusals(self):
        mesh = body()
        good = maps(mesh)

        def edited(place, **fields):
            sets = [uv.copy() for uv in good]
            for key, value in fields.items():
                setattr(sets[place], key, value)
            return UVList(sets)

        fewer      = mesh.counts[:-1].copy()
        holes      = mesh.counts.copy()
        holes[5]   = 0
        outside    = good[0].indices.copy()
        outside[7] = len(good[0].points)
        self.refused(TypeError, "missing: meshes=", uvs=good, clip=motion(bound()))
        self.refused(TypeError, "a UVList or a UVData, not a list", meshes=mesh, uvs=list(good))
        self.refused(TypeError, "a UVList or a UVData, not a MeshData", meshes=mesh, uvs=mesh)
        self.refused(ValueError, "UV set 1 'lightmap' names no mesh of meshes=: mesh_uuid None", meshes=mesh, uvs=edited(1, mesh_uuid=None))
        self.refused(ValueError, "UV set 0 'map1' names no mesh of meshes=: mesh_uuid '", meshes=mesh, uvs=edited(0, mesh_uuid=generate_uuid()))
        self.refused(ValueError, "UV set 1 '' has no name", meshes=mesh, uvs=edited(1, name=""))
        self.refused(ValueError, "UV set 1 'map1' has the name of set 0 of mesh 'body'", meshes=mesh, uvs=edited(1, name="map1"))
        self.refused(ValueError, "not ASCII", meshes=mesh, uvs=edited(1, name="carteé"))
        self.refused(ValueError, "UV set 0 'map1' channel 'counts' holds", meshes=mesh, uvs=edited(0, counts=fewer))
        self.refused(ValueError, "leaves 1 faces without UVs", meshes=mesh, uvs=edited(0, counts=holes))
        self.refused(ValueError, "channel 'indices' names UVs", meshes=mesh, uvs=edited(0, indices=outside))

    def test_load_reads_no_empty_set_and_refuses_unmapped_corners(self):
        mesh = body()
        save_fbx(self.path("good.fbx"), meshes=mesh, uvs=maps(mesh))

        def shape_of(scene):
            return typed(_named(scene, "body").GetNodeAttribute(), FBX.FbxMesh, FBX)

        def empty(scene):
            element = shape_of(scene).CreateElementUV("empty")
            element.SetMappingMode(FBX.FbxLayerElement.EMappingMode.eByPolygonVertex)
            element.SetReferenceMode(FBX.FbxLayerElement.EReferenceMode.eIndexToDirect)

        def unmapped(scene):
            shape_of(scene).GetElementUV(1).GetIndexArray().SetAt(4, -1)

        got = load_fbx(_edited(self.path("good.fbx"), self.path("empty.fbx"), empty))
        self.assertSameUVs(got.uvs, maps(mesh), got.meshes[0])
        bad = _edited(self.path("good.fbx"), self.path("unmapped.fbx"), unmapped)
        with self.assertRaisesRegex(ValueError, "UV set 'lightmap' leaves 1 face corners without UVs"):
            load_fbx(bad)
        # a mesh whose one set is empty reads None
        save_fbx(self.path("bare.fbx"), meshes=mesh)
        path = _edited(self.path("bare.fbx"), self.path("only_empty.fbx"), empty)
        with open_fbx(path, FBX) as scene:
            self.assertEqual(shape_of(scene).GetElementUVCount(), 1)
        self.assertIsNone(load_fbx(path).uvs)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaUVs(_Recorded, _Mapped):
    """Maya 2025's FBXImport of a save's UV sets, and its FBXExport read
    back, as recorded"""

    def test_maya_imports_every_set_on_every_face(self):
        held = self.case("uvs_import")
        mesh = body()
        uvs  = maps(mesh)
        self.assertFreshSave(self.fixture("ours.fbx"), meshes=mesh, uvs=uvs)
        self.assertEqual(held.value("uv_sets"), [uv.name for uv in uvs])
        for uv in uvs:
            np.testing.assert_array_equal(held.array(f"uv/{uv.name}/sizes"), mesh.counts, err_msg=uv.name)
            near(held.array(f"uv/{uv.name}/corners"), per_corner(uv), err_msg=uv.name)

    def test_maya_export_reads_back(self):
        """Maya's FBXExport of the import: its sets within 1e-6 (Maya holds
        UVs in float32); load, save, load bit for bit"""
        self.case("uvs_export")
        source = bundle()
        uvs    = maps(source["meshes"])
        self.assertFreshSave(self.fixture("ours.fbx"), **source, uvs=uvs)
        first = load_fbx(self.fixture("maya.fbx"))
        self.assertEqual([uv.name for uv in first.uvs], [uv.name for uv in uvs])
        for got, want in zip(first.uvs, uvs):
            np.testing.assert_array_equal(got.counts, source["meshes"].counts, err_msg=got.name)
            near(per_corner(got), per_corner(want), err_msg=got.name)
            self.assertEqual(got.mesh_uuid, first.meshes[0].uuid)
        save_fbx(self.path("again.fbx"), *first)
        again = load_fbx(self.path("again.fbx"))
        self.assertSameUVs(again.uvs, first.uvs, again.meshes[0])

    def test_maya_export_of_a_mesh_with_no_uvs_reads_none(self):
        held = self.case("uvs_none")
        # a save of no UVs, imported
        self.assertFreshSave(self.fixture("ours.fbx"), meshes=body())
        self.assertTrue(all(count == 0 for count in held.value("body_sets").values()), held.value("body_sets"))
        self.assertIsNone(load_fbx(self.fixture("maya.fbx")).uvs)
        # a mesh Maya makes with no UVs: on its transform, of no name (Maya's
        # FBXExport leaves the FbxMesh unnamed; measured)
        self.assertTrue(all(count == 0 for count in held.value("box_sets").values()), held.value("box_sets"))
        got = load_fbx(self.fixture("box.fbx"))
        self.assertIsNone(got.meshes[0].name)
        self.assertOn(got.meshes[0], got.clip.active, "box")
        self.assertIsNone(got.uvs)
