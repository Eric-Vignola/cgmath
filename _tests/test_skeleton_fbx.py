"""The r17-skeleton fork: one skeleton or one take, saved, read back, and
read by Maya's own FBXImport.

``cgmath.formats.save_fbx(path, skeleton)`` writes a HierarchyData of joints
and transforms, ``save_fbx(path, clip=clip)`` a ClipData's nodes and its
take; ``cgmath.formats.load_fbx(path)`` gives ``(hierarchy, None)`` or
``(None, clip)`` back. No bind pose is written or read. The FBX tests need
the Autodesk SDK (mayapy), the Maya ones a Maya with its fbxmaya plug-in.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.geometry import MeshData, SkinData
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


def at_float32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


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
        self.refused(NotImplementedError, "comes with skins", skeleton=skeleton(), clip=take(skeleton()))

    def test_neither(self):
        self.refused(TypeError, r"takes a skeleton \(a HierarchyData\) or clip=")
        self.refused(TypeError, r"takes a skeleton \(a HierarchyData\) or clip=", None, None)

    def test_a_clip_as_the_skeleton(self):
        self.refused(TypeError, "the skeleton is a HierarchyData, not a ClipData, pass it as clip=", take(skeleton()))
        self.refused(TypeError, "not a ClipData, pass it as clip=", skeleton=take(skeleton()))

    def test_anything_else(self):
        self.refused(TypeError, "the clip is a ClipData, not a HierarchyData",     clip=skeleton())
        self.refused(TypeError, "the clip is a ClipData, not a ClipList",          clip=ClipList([take(skeleton())]))
        self.refused(TypeError, "the skeleton is a HierarchyData, not a ClipList", ClipList([take(skeleton())]))
        self.refused(TypeError, "the skeleton is a HierarchyData, not a list",     list(skeleton()))
        with self.assertRaisesRegex(NotImplementedError, "r17-skeleton"):
            ClipList([take(skeleton())]).save_fbx(self.path("x.fbx"))
        with self.assertRaises(TypeError):
            take(skeleton()).save_fbx(self.path("x.fbx"), skeleton=skeleton())


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestClipAlone(_Files):
    """save_fbx(path, clip=clip): the clip's nodes at its first frame, and its
    take; load_fbx gives (None, clip)"""

    def test_save_load_gives_no_skeleton_and_the_clip_in_float32(self):
        clip = take(skeleton())
        save_fbx(self.path("a.fbx"), clip=clip)
        rig, back = load_fbx(self.path("a.fbx"))[:2]
        self.assertIsNone(rig)
        self.assertEqual(back.take, "walk")
        self.assertEqual((back.start_frame, back.frame_count, back.fps), (START, FRAMES, FPS))
        self.assertEqual(structure(back), structure(clip))
        self.assertEqual(back["hips"].label, "hello")
        want = layout(clip)
        read = layout(back)
        self.assertEqual(list(read), list(want))
        for key in want:
            self.assertEqual(read[key].shape, want[key].shape, key)
            np.testing.assert_array_equal(read[key], at_float32(want[key]), err_msg=str(key))

    def test_a_second_save_and_load_is_bit_for_bit(self):
        save_fbx(self.path("a.fbx"), clip=take(skeleton()))
        first = load_fbx(self.path("a.fbx")).clip
        save_fbx(self.path("b.fbx"), clip=first)
        rig, second = load_fbx(self.path("b.fbx"))[:2]
        self.assertIsNone(rig)
        self.assertEqual(second.take, first.take)
        self.assertSameClip(second, first)
        for frame in (START, START + 5, START + FRAMES - 1):
            self.assertSameSkeleton(second.at(frame), first.at(frame))

    def test_constant_channels_read_back_at_every_frame_number(self):
        save_fbx(self.path("a.fbx"), clip=take(skeleton()))
        clip = load_fbx(self.path("a.fbx")).clip
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
        save_fbx(self.path("b.fbx"), skeleton=got)
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
        self.assertSameClip(ClipData.load_fbx(self.path("clip.fbx")), back)
        skeleton().save_fbx(self.path("rig.fbx"))
        with self.assertRaisesRegex(ValueError, "holds no take"):
            ClipData.load_fbx(self.path("rig.fbx"))


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
        rig["prop"].node_type = "locator"
        self.refused("node 'prop' is a 'locator'", rig)
        self.refused("node 'prop' is a 'locator'", clip=take(rig))

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
        with self.assertRaisesRegex(ValueError, "node 'c' channel 'inherit type'"):
            load_fbx(self.written(parent_scale=(2.0, 2.0, 2.0), inherit=FBX.FbxTransform.EInheritType.eInheritRrSs))


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
        for label, kwargs in (("skeleton", {"skeleton": skeleton()}), ("clip", {"clip": take(skeleton())})):
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
        rig, clip = load_fbx(path)[:2]
        self.assertIsNone(rig)
        np.testing.assert_array_equal(clip.frames["c"].translate, [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(clip.frames["c"].rotate,    [[10.0, 20.0, 30.0]])
        np.testing.assert_array_equal(clip.frames["p"].scale,     [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])


def _maya():
    """maya.cmds and maya.mel with the fbxmaya plug-in loaded, or None"""
    try:
        import maya.cmds as cmds
        import maya.mel as mel

        cmds.loadPlugin("fbxmaya", quiet=True)
    except Exception:  # not running in Maya, or no fbx plug-in
        return None
    return cmds, mel


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


class _InMaya(_Files):
    def setUp(self):
        super().setUp()
        maya = _maya()
        if maya is None:
            self.skipTest("needs Maya and its fbxmaya plug-in")
        self.cmds, self.mel = maya
        self.addCleanup(self.cmds.file, new=True, force=True)

    def imported(self, path: str) -> None:
        cmds, mel = self.cmds, self.mel
        cmds.file(new=True, force=True)
        cmds.currentUnit(time="ntsc", linear="cm", angle="deg")
        cmds.upAxis(axis="y")
        mel.eval("FBXResetImport")
        mel.eval(f'FBXImport -f "{path.replace(os.sep, "/")}"')

    def exported(self, name: str) -> str:
        """Maya's FBXExport of the scene at the clip's first frame"""
        cmds, mel = self.cmds, self.mel
        cmds.currentTime(START)
        mel.eval("FBXResetExport")
        # what FBXResetExport leaves as an earlier export set it
        mel.eval("FBXExportConvertUnitString cm")
        mel.eval("FBXExportUpAxis y")
        out = self.path(name).replace(os.sep, "/")
        mel.eval(f'FBXExport -f "{out}"')
        return out


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMaya(_InMaya):
    """Maya 2025's FBXImport of a save, and its FBXExport of that scene"""

    def maya_node(self, node) -> None:
        """the imported node's name, type, parent and static attributes"""
        cmds = self.cmds
        self.assertEqual(len(cmds.ls(node.name, long=True)), 1, node.name)
        self.assertEqual(cmds.nodeType(node.name), node.node_type)
        parent = cmds.listRelatives(node.name, parent=True)
        self.assertEqual(parent[0] if parent else None, node.parent_name)
        self.assertEqual(cmds.getAttr(f"{node.name}.rotateOrder"), node.rotate_order)
        np.testing.assert_allclose(cmds.getAttr(f"{node.name}.rotateAxis")[0], node.rotate_axis, rtol=0, atol=1e-6)
        if node.node_type == "joint":
            np.testing.assert_allclose(cmds.getAttr(f"{node.name}.jointOrient")[0], node.joint_orient, rtol=0, atol=1e-6)
            self.assertEqual(cmds.getAttr(f"{node.name}.segmentScaleCompensate"), node.segment_scale_compensate)

    def maya_user(self, node, frame=None) -> None:
        from cgmath.hierarchy.hierarchy import _user_channels

        cmds = self.cmds
        for name in _user_channels(node.user_defined_attributes):
            value = cmds.getAttr(f"{node.name}.{name}") if frame is None else cmds.getAttr(f"{node.name}.{name}", time=frame)
            value = np.ravel(value[0] if isinstance(value, list) else value).astype(np.float64)
            want  = maya_value(node, name, np.atleast_1d(np.asarray(getattr(node, name), dtype=np.float64)))
            np.testing.assert_allclose(value, want, rtol=0, atol=1e-6, err_msg=(node.name, name, frame))

    def test_maya_imports_the_clip_as_an_animated_hierarchy(self):
        """every node, and on every frame its channels, user attributes and
        world matrix, within 1e-6 of load_fbx's read of the same file"""
        cmds = self.cmds
        save_fbx(self.path("a.fbx"), clip=take(skeleton()))
        rig, back = load_fbx(self.path("a.fbx"))[:2]
        self.assertIsNone(rig)
        self.imported(self.path("a.fbx"))
        self.assertEqual(cmds.ls(type="dagPose"), [])
        self.assertEqual(len(cmds.ls(type="animCurve")), curve_count(back))
        for node in back:
            self.maya_node(node)

        wanted = np.asarray(back.frames.world_matrix)
        for f in range(back.frame_count):
            frame      = START + f
            back.frame = f
            for column, node in enumerate(back):
                for attr in ("translate", "rotate", "scale"):
                    value = cmds.getAttr(f"{node.name}.{attr}", time=frame)[0]
                    np.testing.assert_allclose(value, getattr(node, attr), rtol=0, atol=1e-6, err_msg=(node.name, attr, frame))
                world = np.array(cmds.getAttr(f"{node.name}.worldMatrix", time=frame)).reshape(4, 4)
                np.testing.assert_allclose(world, wanted[f, column], rtol=0, atol=1e-6, err_msg=(node.name, frame))
                self.maya_user(node, frame)

    def test_maya_imports_the_skeleton_as_a_still_hierarchy(self):
        """no curve; every node's attributes and world matrix within 1e-6 of
        the skeleton"""
        cmds = self.cmds
        rig  = skeleton()
        save_fbx(self.path("rig.fbx"), rig)
        self.assertEqual(counts(self.path("rig.fbx"))["poses"], 0)
        self.imported(self.path("rig.fbx"))
        self.assertEqual(cmds.ls(type="animCurve"), [])
        self.assertEqual(cmds.ls(type="dagPose"), [])
        worlds = np.asarray(rig.world_matrix)
        for column, node in enumerate(rig):
            self.maya_node(node)
            self.maya_user(node)
            world = np.array(cmds.getAttr(f"{node.name}.worldMatrix")).reshape(4, 4)
            np.testing.assert_allclose(world, worlds[column], rtol=0, atol=1e-6, err_msg=node.name)
            for attr in ("translate", "rotate", "scale"):
                np.testing.assert_allclose(cmds.getAttr(f"{node.name}.{attr}")[0], getattr(node, attr), rtol=0, atol=1e-6)

    def test_maya_export_reads_back_as_the_save(self):
        """Maya's FBXExport of the import: load, save, load gives the first
        load back bit for bit; the first is the source's take within 1e-6"""
        from cgmath.hierarchy.hierarchy import _user_channels

        cmds, mel = self.cmds, self.mel
        save_fbx(self.path("a.fbx"), clip=take(skeleton()))
        self.imported(self.path("a.fbx"))
        cmds.currentTime(START)
        mel.eval("FBXResetExport")
        # what FBXResetExport leaves as an earlier export set it
        mel.eval("FBXExportConvertUnitString cm")
        mel.eval("FBXExportUpAxis y")
        out = self.path("maya.fbx").replace(os.sep, "/")
        mel.eval(f'FBXExport -f "{out}"')

        rig, first = load_fbx(out)[:2]
        self.assertIsNone(rig)
        save_fbx(self.path("again.fbx"), clip=first)
        rig, second = load_fbx(self.path("again.fbx"))[:2]
        self.assertIsNone(rig)
        self.assertSameClip(second, first)
        for frame in (START, START + FRAMES - 1):
            self.assertSameSkeleton(second.at(frame), first.at(frame))

        source = load_fbx(self.path("a.fbx")).clip
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
# axis, rotate order, segment scale compensate), parents first; the worlds
# stay within 8 units: Maya's skinCluster deforms in float32, its error
# grows with the distance from the origin
BOUND = (
    ("root", None, "transform", (0.25, 0.5, -0.25), (0.0, 15.0, 0.0), (1.25, 1.25, 1.25), (0, 0, 0), (0, 0, 0), 0, False),
    ("hips", "root", "joint", (0.0, 1.0, 0.0), (5.0, 0.0, 0.0), (1, 1, 1), (0.0, 0.0, 90.0), (10.0, 0.0, 0.0), 1, False),
    ("spine", "hips", "joint", (1.0, 0.0, 0.0), (0.0, 0.0, 12.0), (1.1, 1.1, 1.1), (0.0, 0.0, -15.0), (0.0, 5.0, 0.0), 2, True),
    ("chest", "spine", "joint", (1.25, 0.0, 0.0), (0.0, 3.0, 0.0), (1, 1, 1), (5.0, 0.0, 0.0), (0, 0, 0), 3, True),
    ("head", "chest", "joint", (1.0, 0.0, 0.0), (0, 0, 0), (1.0, 1.2, 0.8), (0, 0, 0), (0, 0, 0), 4, False),
    ("arm_grp", "chest", "transform", (0.0, 0.5, 0.25), (10.0, 20.0, 30.0), (1, 1, 1), (0, 0, 0), (0, 0, 0), 5, False),
    ("arm", "arm_grp", "joint", (0.5, 0.0, 0.0), (0, 0, 0), (1, 1, 1), (0.0, 90.0, 0.0), (0.0, 0.0, 10.0), 0, False),
    ("hand", "arm", "joint", (0.75, 0.0, 0.0), (0, 0, 0), (0.9, 1.1, 1.0), (0, 0, 0), (0, 0, 0), 5, True),
    ("leg", "hips", "joint", (0.0, -0.5, 0.5), (0, 0, 0), (1, 1, 1), (0.0, 0.0, -90.0), (0, 0, 0), 4, False),
    ("foot", "leg", "joint", (1.0, 0.0, 0.0), (0, 0, 0), (1.3, 0.7, 1.0), (0, 0, 0), (0, 0, 0), 2, False),
)

# the skin's influences, leaves among them
INFLUENCES = ("hips", "spine", "chest", "head", "arm", "hand", "leg", "foot")
LEAVES     = ("head", "hand", "foot")
TAKE       = 15


def bound() -> HierarchyData:
    """joints and transforms, joint orients, rotate axes, every rotate order,
    segment scale compensate on and off, uneven scales on the leaves"""
    uuids = {}
    nodes = []
    for name, parent, kind, t, r, s, jo, ra, order, ssc in BOUND:
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
    rig = HierarchyData(nodes)
    rig["hips"].add_user_attribute("weight", 0.5, attribute_type="double")
    return rig


def motion(rig: HierarchyData) -> ClipData:
    """15 frames: uniform scale on root, spine and arm (each with children),
    uneven scale on the leaves head, hand and foot, rotation and
    translation"""
    clip = ClipData(rig, frames=TAKE, start_frame=START, fps=FPS, take="bend")
    f    = np.arange(TAKE, dtype=np.float64)[:, None]
    one  = np.ones((1, 3))
    clip.frames["root"].translate    = np.hstack([0.25 + f * 0.025, 0.5 + f * 0.0, -0.25 - f * 0.01])
    clip.frames["root"].scale        = (1.25 + f * 0.02) * one
    clip.frames["hips"].translate    = np.hstack([np.sin(f / 3.0) * 0.15, 1.0 + f * 0.01, -f * 0.005])
    clip.frames["hips"].rotate       = np.hstack([5.0 + f * 2.0, f * -1.5, np.cos(f) * 4.0])
    clip.frames["spine"].rotate      = np.hstack([f * 0.5, f * 0.0 + 1.0, 12.0 + f])
    clip.frames["spine"].scale       = (1.1 - f * 0.015) * one
    clip.frames["arm_grp"].translate = np.hstack([f * 0.015, 0.5 + f * 0.0, 0.25 - f * 0.01])
    clip.frames["arm"].rotate        = np.hstack([f * 3.0, 10.0 + f * 0.0, -f * 2.0])
    clip.frames["arm"].scale         = (1.0 + f * 0.03) * one
    clip.frames["leg"].rotate        = np.hstack([f * -2.0, f * 1.0, f * 2.5])
    clip.frames["head"].scale        = np.hstack([1.0 + f * 0.01, 1.2 - f * 0.02, 0.8 + f * 0.005])
    clip.frames["hand"].scale        = np.hstack([0.9 + f * 0.02, 1.1 + f * 0.0, 1.0 - f * 0.03])
    clip.frames["foot"].scale        = np.hstack([1.3 - f * 0.01, 0.7 + f * 0.02, 1.0 + f * 0.01])
    clip.frames["hips"].weight       = 0.5 + f * 0.05
    clip.frame = 0
    return clip


def body() -> MeshData:
    """a tube of 24 rings of 12 points up the skeleton, quads and, every
    third quad split in two, triangles; float32 points, as Maya holds them;
    each corner's normal its point's, a triangle's corners their face's"""
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
        points         = at_float32(points),
        counts         = np.array(counts, dtype=np.int32),
        indices        = np.array(indices, dtype=np.int32),
        normals        = at_float32(normals),
        normal_indices = np.array(ids, dtype=np.int32),
    )


def weighed(mesh: MeshData) -> SkinData:
    """1 to 4 influences a point, 0.1 and over before normalizing; zeros
    elsewhere"""
    rng     = np.random.default_rng(7)
    weights = np.zeros((mesh.point_count, len(INFLUENCES)))
    for row in range(mesh.point_count):
        picked               = rng.choice(len(INFLUENCES), size=rng.integers(1, 5), replace=False)
        values               = 0.1 + rng.random(len(picked))
        weights[row, picked] = values / values.sum()
    return SkinData(weights=weights, influences=list(INFLUENCES), name="bodySkin")


def bundle() -> dict:
    rig  = bound()
    mesh = body()
    return {"skeleton": rig, "clip": motion(bound()), "mesh": mesh, "skin": weighed(mesh)}


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
        self.assertEqual(got.name, want.name)
        for field in ("points", "counts", "indices", "normals", "normal_indices"):
            a, b = getattr(got, field), getattr(want, field)
            self.assertEqual(a.dtype, b.dtype, field)
            np.testing.assert_array_equal(a, b, err_msg=field)
        np.testing.assert_array_equal(got.matrix, np.eye(4))

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
    """save_fbx(path, skeleton, clip, mesh, skin): the skeleton the bind pose,
    one skin, one bind pose; load_fbx gives the four back"""

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
        self.assertEqual(first._fields, ("skeleton", "clip", "mesh", "skin"))
        self.assertSameSkeleton(first.skeleton, source["skeleton"])
        self.assertSameMesh(first.mesh, source["mesh"])
        self.assertSameSkin(first.skin, source["skin"])
        self.assertFloat32Clip(first.clip, source["clip"])

        save_fbx(self.path("b.fbx"), *first)
        second = load_fbx(self.path("b.fbx"))
        self.assertSameSkeleton(second.skeleton, first.skeleton)
        self.assertSameMesh(second.mesh, first.mesh)
        self.assertSameSkin(second.skin, first.skin)
        self.assertSameClip(second.clip, first.clip)

    def test_the_file_holds_one_skin_its_clusters_and_one_bind_pose(self):
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)
        worlds = np.asarray(source["skeleton"].world_matrix)
        place  = {name: i for i, name in enumerate(source["skeleton"].name)}
        with open_fbx(self.path("a.fbx"), FBX) as scene:
            node  = _named(scene, "body")
            shape = typed(node.GetNodeAttribute(), FBX.FbxMesh, FBX)
            kind  = FBX.FbxDeformer.EDeformerType.eSkin
            self.assertEqual(shape.GetDeformerCount(kind), 1)
            skin = typed(shape.GetDeformer(0, kind), FBX.FbxSkin, FBX)
            self.assertEqual(str(skin.GetName()), "bodySkin")
            self.assertEqual(skin.GetClusterCount(), len(INFLUENCES))
            for k, name in enumerate(INFLUENCES):
                cluster = skin.GetCluster(k)
                self.assertEqual(str(cluster.GetLink().GetName()), name)
                column = source["skin"].weights[:, k]
                np.testing.assert_array_equal(cluster.GetControlPointIndices(), np.flatnonzero(column))
                np.testing.assert_array_equal(cluster.GetControlPointWeights(), column[column != 0.0])
                linked, placed = FBX.FbxAMatrix(), FBX.FbxAMatrix()
                cluster.GetTransformLinkMatrix(linked)
                cluster.GetTransformMatrix(placed)
                np.testing.assert_allclose(_matrix(linked), worlds[place[name]], rtol=0, atol=1e-12)
                np.testing.assert_allclose(_matrix(placed), np.eye(4), rtol=0, atol=1e-12)
            self.assertEqual(scene.GetPoseCount(), 1)
            pose = scene.GetPose(0)
            self.assertTrue(pose.IsBindPose())
            self.assertEqual(pose.GetCount(), len(BOUND) + 1)
            held = {str(pose.GetNode(k).GetName()): _matrix(pose.GetMatrix(k)) for k in range(pose.GetCount())}
            np.testing.assert_array_equal(held.pop("body"), np.eye(4))
            for name, matrix in held.items():
                np.testing.assert_array_equal(matrix, worlds[place[name]], err_msg=name)
            # the nodes hold the skeleton's own values, the take moves them
            for name, *_ in BOUND:
                np.testing.assert_array_equal(
                    tuple(_named(scene, name).LclTranslation.Get()), source["skeleton"][name].translate
                )

    def test_a_skin_without_a_take(self):
        source = bundle()
        del source["clip"]
        save_fbx(self.path("a.fbx"), **source)
        self.assertEqual(counts(self.path("a.fbx"))["curves"], 0)
        rig, clip, mesh, skin = load_fbx(self.path("a.fbx"))
        self.assertIsNone(clip)
        self.assertSameSkeleton(rig, source["skeleton"])
        self.assertSameMesh(mesh, source["mesh"])
        self.assertSameSkin(skin, source["skin"])

    def test_the_bind_pose_places_nodes_their_own_values_do_not(self):
        """Maya's FBXExport writes the nodes at the playhead: the bind pose's
        matrices give the skeleton back"""
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)

        def moved(scene):
            _named(scene, "spine").LclRotation.Set(FBX.FbxDouble3(40.0, -20.0, 7.0))
            _named(scene, "root").LclScaling.Set(FBX.FbxDouble3(2.0, 2.0, 2.0))
            _named(scene, "hand").LclScaling.Set(FBX.FbxDouble3(3.0, 1.0, 1.0))

        rig = load_fbx(_edited(self.path("a.fbx"), self.path("moved.fbx"), moved)).skeleton
        self.assertSameSkeleton(rig, source["skeleton"], exact=False, atol=1e-9)
        np.testing.assert_allclose(rig.world_matrix, source["skeleton"].world_matrix, rtol=0, atol=1e-12)

    def test_mesh_alone_and_beside_a_clip_or_a_skeleton(self):
        mesh = body()
        save_fbx(self.path("mesh.fbx"), mesh=mesh)
        self.assertEqual(counts(self.path("mesh.fbx")), {"poses": 0, "stacks": 0, "layers": 0, "curve nodes": 0, "curves": 0})
        got = load_fbx(self.path("mesh.fbx"))
        self.assertEqual((got.skeleton, got.clip, got.skin), (None, None, None))
        self.assertSameMesh(got.mesh, mesh)
        save_fbx(self.path("again.fbx"), mesh=got.mesh)
        self.assertSameMesh(load_fbx(self.path("again.fbx")).mesh, got.mesh)
        # the class routes
        mesh.save_fbx(self.path("route.fbx"))
        self.assertSameMesh(load_fbx(self.path("route.fbx")).mesh, mesh)

        clip = motion(bound())
        save_fbx(self.path("clip.fbx"), clip=clip, mesh=mesh)
        found = counts(self.path("clip.fbx"))
        self.assertEqual((found["poses"], found["curves"]), (0, curve_count(clip)))
        rig, back, held, skin = load_fbx(self.path("clip.fbx"))
        self.assertEqual((rig, skin), (None, None))
        self.assertFloat32Clip(back, clip)
        self.assertSameMesh(held, mesh)
        self.assertEqual(ClipData.load_fbx(self.path("clip.fbx")).name, back.name)
        save_fbx(self.path("clip2.fbx"), clip=back, mesh=held)
        again = load_fbx(self.path("clip2.fbx"))
        self.assertSameClip(again.clip, back)
        self.assertSameMesh(again.mesh, held)

        rig = bound()
        save_fbx(self.path("rig.fbx"), rig, mesh=mesh)
        got = load_fbx(self.path("rig.fbx"))
        self.assertEqual((got.clip, got.skin), (None, None))
        self.assertSameSkeleton(got.skeleton, rig)
        self.assertSameMesh(got.mesh, mesh)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSkinRefusals(_Skinned):
    def refused(self, error, message, **kwargs):
        with self.assertRaisesRegex(error, message):
            save_fbx(self.path("refused.fbx"), **kwargs)
        self.assertFalse(os.path.exists(self.path("refused.fbx")))

    def test_a_skin_needs_its_skeleton_and_its_mesh(self):
        source = bundle()
        self.refused(TypeError, r"missing: skeleton \(a HierarchyData, the bind pose\)$", mesh=source["mesh"], skin=source["skin"])
        self.refused(TypeError, r"missing: mesh=$", skeleton=source["skeleton"], skin=source["skin"])
        self.refused(TypeError, "missing: skeleton .* and mesh=", clip=source["clip"], skin=source["skin"])
        self.refused(TypeError, "the mesh is a MeshData, not a list", mesh=[source["mesh"]])
        self.refused(TypeError, "the skin is a SkinData, not a MeshData", skeleton=source["skeleton"], mesh=source["mesh"], skin=source["mesh"])
        # a skeleton and a clip without a skin
        self.refused(NotImplementedError, "comes with skins", skeleton=source["skeleton"], clip=source["clip"], mesh=source["mesh"])

    def test_influences_and_weights(self):
        source          = bundle()
        skin            = source["skin"]
        skin.influences = ["hips", "spine", "chest", "head", "arm", "hand", "leg", "toe"]
        self.refused(ValueError, r"skin 'bodySkin' names no node of the skeleton: influences \['toe'\]", **source)
        source                 = bundle()
        source["skin"].weights = source["skin"].weights[:-1]
        self.refused(ValueError, "skin 'bodySkin' channel 'weights' holds 287 rows, mesh 'body' holds 288 points", **source)
        source                 = bundle()
        source["skin"].weights = source["skin"].weights[:, :-1]
        self.refused(ValueError, "holds 7 columns for its 8 influences", **source)

    def test_the_mesh(self):
        source                = bundle()
        matrix                = np.eye(4)
        matrix[3, 1]          = 1.0
        source["mesh"].matrix = matrix
        self.refused(ValueError, "mesh 'body' channel 'matrix'", **source)
        self.refused(ValueError, "mesh 'body' channel 'matrix'", mesh=source["mesh"])
        mesh              = body()
        mesh.hole_faces   = np.array([0],       dtype=np.int32)
        mesh.hole_counts  = np.array([3],       dtype=np.int32)
        mesh.hole_indices = np.array([0, 1, 2], dtype=np.int32)
        self.refused(ValueError, "mesh 'body' holds 1 polygon holes", mesh=mesh)

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
        self.refused(ValueError, "the clip's nodes .* are not the skeleton's", **source)

    def write_bad(self, name: str, edit) -> str:
        save_fbx(self.path("good.fbx"), **bundle())
        return _edited(self.path("good.fbx"), self.path(name), edit)

    def test_load_refusals(self):
        def two_meshes(scene):
            node  = FBX.FbxNode.Create(scene, "other")
            shape = FBX.FbxMesh.Create(scene, "otherShape")
            shape.InitControlPoints(3)
            for i in range(3):
                shape.SetControlPointAt(FBX.FbxVector4(float(i), float(i % 2), 0.0, 1.0), i)
            shape.BeginPolygon(-1, -1, -1, False)
            for i in range(3):
                shape.AddPolygon(i)
            shape.EndPolygon()
            node.SetNodeAttribute(shape)
            scene.GetRootNode().AddChild(node)

        def two_poses(scene):
            pose = FBX.FbxPose.Create(scene, "again")
            pose.SetIsBindPose(True)
            pose.Add(_named(scene, "body"), FBX.FbxMatrix())
            scene.AddPose(pose)

        def two_skins(scene):
            shape   = typed(_named(scene, "body").GetNodeAttribute(), FBX.FbxMesh, FBX)
            skin    = FBX.FbxSkin.Create(scene, "again")
            cluster = FBX.FbxCluster.Create(scene, "again")
            cluster.SetLink(_named(scene, "hips"))
            cluster.AddControlPointIndex(0, 1.0)
            skin.AddCluster(cluster)
            shape.AddDeformer(skin)

        def moved(scene):
            _named(scene, "body").LclTranslation.Set(FBX.FbxDouble3(0.0, 1.0, 0.0))

        def offset(scene):
            _named(scene, "body").SetGeometricRotation(FBX.FbxNode.EPivotSet.eSourcePivot, FBX.FbxVector4(0.0, 90.0, 0.0))

        for name, edit, error, message in (
            ("two_meshes", two_meshes, NotImplementedError, r"holds 2 meshes \['body', 'other'\]"),
            ("two_poses", two_poses, ValueError, "mesh 'body' is listed by 2 bind poses"),
            ("two_skins", two_skins, ValueError, "mesh 'body' holds 2 skin deformers"),
            ("moved", moved, ValueError, "mesh 'body' channel 'matrix'"),
            ("offset", offset, ValueError, "mesh 'body' channel 'matrix'"),
        ):
            with self.subTest(name):
                path = self.write_bad(f"{name}.fbx", edit)
                with self.assertRaisesRegex(error, message):
                    load_fbx(path)
                with self.assertRaisesRegex(error, message):
                    ClipData.load_fbx(path)


def _matrix(matrix) -> np.ndarray:
    return np.array([[matrix.Get(row, column) for column in range(4)] for row in range(4)])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaSkin(_InMaya, _Skinned):
    """Maya 2025's FBXImport of a skinned save: one skinCluster and one bind
    pose deforming the mesh as cgmath's own skinning does, frame by frame"""

    def mesh_path(self, name: str):
        import maya.api.OpenMaya as om

        shapes = self.cmds.listRelatives(name, shapes=True, noIntermediate=True, fullPath=True)
        self.assertEqual(len(shapes), 1)
        chosen = om.MSelectionList()
        chosen.add(shapes[0])
        return chosen.getDagPath(0)

    def test_maya_imports_one_skin_that_deforms_as_cgmath(self):
        import maya.api.OpenMaya as om
        import maya.api.OpenMayaAnim as oma
        from cgmath.geometry.deform import SkinDeformData

        cmds   = self.cmds
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)
        back = load_fbx(self.path("a.fbx"))
        self.imported(self.path("a.fbx"))

        path     = self.mesh_path("body")
        clusters = cmds.ls(type="skinCluster")
        self.assertEqual(len(clusters), 1)
        self.assertEqual(cmds.ls(cmds.listHistory(path.fullPathName()), type="skinCluster"), clusters)
        poses = cmds.ls(type="dagPose")
        self.assertEqual(len(poses), 1)
        self.assertTrue(cmds.getAttr(f"{poses[0]}.bindPose"))

        chosen = om.MSelectionList()
        chosen.add(clusters[0])
        skin       = oma.MFnSkinCluster(chosen.getDependNode(0))
        influences = [x.partialPathName() for x in skin.influenceObjects()]
        self.assertEqual(influences, list(source["skin"].influences))
        every = om.MFnSingleIndexedComponent()
        part  = every.create(om.MFn.kMeshVertComponent)
        every.setCompleteData(source["mesh"].point_count)
        weights, _ = skin.getWeights(path, part)
        weights = np.array(weights).reshape(source["mesh"].point_count, len(influences))
        np.testing.assert_allclose(weights, source["skin"].weights, rtol=0, atol=1e-6)
        worlds = np.asarray(source["skeleton"].world_matrix)
        for k, joint in enumerate(skin.influenceObjects()):
            index    = skin.indexForInfluenceObject(joint)
            prebind  = np.array(cmds.getAttr(f"{clusters[0]}.bindPreMatrix[{index}]")).reshape(4, 4)
            expected = np.linalg.inv(worlds[source["skeleton"].name.index(influences[k])])
            np.testing.assert_allclose(prebind, expected, rtol=0, atol=1e-6, err_msg=influences[k])

        # THE SKIN DEFORMATION: cgmath's linear blend skinning of the re-read
        # mesh, by the re-read clip, bound to the re-read skeleton
        deformer = SkinDeformData(back.mesh, back.skin, bind_rig=back.skeleton)
        wanted   = deformer.apply(back.clip.frames)
        self.assertEqual(wanted.shape, (TAKE, source["mesh"].point_count, 3))
        for f in range(TAKE):
            cmds.currentTime(START + f)
            got = np.array([tuple(p)[:3] for p in om.MFnMesh(path).getPoints(om.MSpace.kWorld)])
            np.testing.assert_allclose(got, wanted[f], rtol=0, atol=1e-5, err_msg=f"frame {START + f}")
        # the deformation is no stand-still: the points move by whole units
        self.assertGreater(float(np.abs(wanted[-1] - source["mesh"].points).max()), 2.0)

    def test_maya_imports_the_unskinned_mesh_at_identity_with_its_normals(self):
        import maya.api.OpenMaya as om

        cmds = self.cmds
        mesh = body()
        clip = motion(bound())
        save_fbx(self.path("a.fbx"), clip=clip, mesh=mesh)
        self.imported(self.path("a.fbx"))
        self.assertEqual(cmds.ls(type="skinCluster"),    [])
        self.assertEqual(cmds.ls(type="dagPose"),        [])
        self.assertEqual(len(cmds.ls(type="animCurve")), curve_count(clip))
        path = self.mesh_path("body")
        self.assertIsNone(cmds.listConnections("body", type="animCurve"))
        self.assertIsNone(cmds.listConnections(path.fullPathName(), type="animCurve"))
        for frame in (START, START + TAKE - 1):
            cmds.currentTime(frame)
            np.testing.assert_array_equal(np.array(cmds.xform("body", q=True, ws=True, matrix=True)).reshape(4, 4), np.eye(4))
        shape  = om.MFnMesh(path)
        points = np.array([tuple(p)[:3] for p in shape.getPoints(om.MSpace.kWorld)])
        np.testing.assert_allclose(points, mesh.points, rtol=0, atol=1e-6)
        counts_, ids = shape.getNormalIds()
        normals = np.array([tuple(n) for n in shape.getNormals()])
        np.testing.assert_array_equal(np.array(counts_), mesh.counts)
        np.testing.assert_allclose(normals[np.array(ids)], mesh.normals[mesh.normal_indices], rtol=0, atol=1e-6)

    def test_maya_export_of_the_skin_reads_back(self):
        """Maya's FBXExport of the skinned import: load, save, load gives the
        first load back bit for bit; the first is the source within 1e-6"""
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)
        self.imported(self.path("a.fbx"))
        out   = self.exported("maya.fbx")

        first = load_fbx(out)
        save_fbx(self.path("again.fbx"), *first)
        second = load_fbx(self.path("again.fbx"))
        self.assertSameSkeleton(second.skeleton, first.skeleton)
        self.assertSameClip(second.clip, first.clip)
        self.assertSameMesh(second.mesh, first.mesh)
        self.assertSameSkin(second.skin, first.skin)

        # Maya writes its bind pose in float32: the angles within 1e-6
        # radians
        rig = source["skeleton"]
        self.assertEqual([n.name for n in first.skeleton], [n.name for n in rig])
        for channel in ("translate", "rotate", "scale", "joint_orient", "rotate_axis"):
            got, want = np.asarray(getattr(first.skeleton, channel)), np.asarray(getattr(rig, channel))
            if channel in ("rotate", "joint_orient", "rotate_axis"):
                got, want = np.radians(got), np.radians(want)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6, err_msg=channel)
        np.testing.assert_allclose(first.skeleton.world_matrix, rig.world_matrix, rtol=0, atol=1e-6)
        self.assertEqual(sorted(first.skin.influences), sorted(INFLUENCES))
        columns = [list(first.skin.influences).index(x) for x in INFLUENCES]
        np.testing.assert_allclose(first.skin.weights[:, columns], source["skin"].weights, rtol=0, atol=1e-6)
        np.testing.assert_allclose(first.mesh.points, source["mesh"].points, rtol=0, atol=1e-6)
        self.assertEqual((first.clip.start_frame, first.clip.frame_count, first.clip.fps), (START, TAKE, FPS))
