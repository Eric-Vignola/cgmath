"""cgmath.formats.load_fbx: a file's rig at its bind, takes, meshes, UV sets
and skins in one object, which save_fbx writes back

What a save of cgmath's objects writes, a load reads back: the keys as the
float32 fbx keeps them, everything else to the bit; and a load saved and
loaded again reads the same, to the bit.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from cgmath import formats
from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
from cgmath.geometry.skin_weights import SkinData, SkinList
from cgmath.hierarchy import ClipData, ClipList, FBX, HierarchyData, TransformData

CHANNELS = ("translate", "rotate", "scale", "joint_orient", "rotate_axis", "shear")


def build():
    """``(rig, clips, meshes, uvs, skins)``: a group scaled unevenly over a
    branching chain of joints, oriented, in other rotate orders, some with a
    rotate axis or an uneven scale, a group holding a mesh of quads,
    triangles and a hexagon with two UV sets, skinned to four joints, and two
    takes of every node"""
    rng = np.random.default_rng(3)
    rig = HierarchyData()

    def add(name, parent, kind="joint", **values):
        node = TransformData(name, node_type=kind, **values)
        if parent is not None:
            node.parent_uuid = parent.uuid
        rig.append(node)
        return node

    top  = add("rig", None, "transform", translate=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), scale=(1.0, 1.5, 0.75))
    hips = add("hips", top, translate=(0.0, 5.0, 0.0), joint_orient=(0.0, 90.0, 0.0), rotate_order=2)
    spine = add("spine", hips, translate=(0.0, 3.0, 0.0), rotate=(5.0, -7.0, 12.0), rotate_axis=(15.0, 120.0, -30.0),
                scale=(1.2, 0.9, 1.1), segment_scale_compensate=True, rotate_order=5)
    add("head", spine, translate=(0.0, 2.0, 0.5), joint_orient=(-20.0, 0.0, 45.0), rotate_order=1)
    add("arm", spine, translate=(2.0, 1.0, 0.0), rotate=(0.0, 0.0, -80.0), joint_orient=(170.0, -60.0, 5.0),
        segment_scale_compensate=True, rotate_order=3)
    body = add("body", None, "transform", translate=(0.0, 0.5, 0.0), rotate=(0.0, 30.0, 0.0))

    xs, ys = np.meshgrid(np.arange(4.0), np.arange(3.0))
    points  = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1) * 0.5 + [0.25, 0.0, -0.125]
    faces   = [[0, 1, 5], [0, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [4, 5, 9, 8], [5, 6, 7, 11, 10, 9]]
    counts  = np.array([len(f) for f in faces], dtype=np.int32)
    indices = np.array([v for f in faces for v in f], dtype=np.int32)
    mesh    = MeshData(points=points, indices=indices, counts=counts, name=body.name)
    mesh.set_normals(hard_edge_angle=30.0)
    first = UVData(points=points[:, :2] / 4.0, indices=indices.copy(), counts=counts.copy(), name="map1")
    second = UVData(points=rng.integers(0, 256, size=(len(indices), 2)) / 256.0,
                    indices=np.arange(len(indices), dtype=np.int32), counts=counts.copy(), name="detail")
    weights = rng.uniform(0.0, 1.0, size=(len(points), 4)) * (rng.random((len(points), 4)) < 0.7)
    weights[:, 0] += 0.1
    weights       /= weights.sum(axis=1, keepdims=True)
    skin  = SkinData(weights=weights, influences=["hips", "spine", "arm", "head"], name=body.name)

    takes = []
    for name, count, start in (("idle", 5, 0), ("walk", 9, 10)):
        clip                  = ClipData(rig, frames=count, start_frame=start, fps=30.0, take=name)
        size                  = (count, len(rig), 3)
        steps                 = np.arange(count)[:, None, None]
        clip.frames.translate = np.asarray(clip.frames.translate) + rng.uniform(-0.3, 0.3, size) * steps
        clip.frames.rotate    = np.asarray(clip.frames.rotate) + rng.uniform(-9.0, 9.0, size) * steps
        clip.frames.scale     = np.asarray(clip.frames.scale) * (1.0 + rng.uniform(-0.05, 0.05, size) * steps)
        takes.append(clip)
    return rig, ClipList(takes), MeshList([mesh]), UVList([first, second]), SkinList([skin])


def parts(contents, keys=False) -> dict:
    """{name: value} of everything a load holds; the take channels keyed in
    a file at float32 when ``keys``"""
    out = {}
    if contents.rig is not None:
        names = {node.uuid: node.name for node in contents.rig}
        for node in contents.rig:
            out[f"rig {node.name}"] = (names.get(node.parent_uuid), node.node_type, int(node.rotate_order),
                                       bool(node.segment_scale_compensate))
            for channel in CHANNELS:
                out[f"rig {node.name} {channel}"] = np.array(getattr(node, channel))
        out["rig order"] = [node.name for node in contents.rig]
    for clip in contents.clips:
        out[f"take {clip.take}"] = (float(clip.fps), int(clip.start_frame), int(clip.frame_count), list(clip.name))
        for channel in CHANNELS:
            block = np.array(getattr(clip.frames, channel))
            if keys and channel in ("translate", "rotate", "scale"):
                block = block.astype(np.float32).astype(np.float64)
            out[f"take {clip.take} {channel}"] = block
    for mesh in contents.meshes:
        for field in ("points", "indices", "counts", "normals", "normal_indices", "matrix"):
            out[f"mesh {mesh.name} {field}"] = getattr(mesh, field)
    for uv in contents.uvs:
        for field in ("points", "indices", "counts"):
            out[f"uv {uv.mesh_name} {uv.name} {field}"] = np.array(getattr(uv, field))
    for skin in contents.skins:
        out[f"skin {skin.name}"] = (list(skin.influences), np.array(skin.weights))
    return out


class TestLoadFbx(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def assert_same(self, got: dict, want: dict):
        self.assertEqual(sorted(got), sorted(want))
        for key, value in want.items():
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], np.ndarray):
                self.assertEqual(got[key][0], value[0], key)
                np.testing.assert_array_equal(got[key][1], value[1], err_msg=key)
                self.assertEqual(got[key][1].dtype, value[1].dtype, key)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(got[key], value, err_msg=key)
                self.assertEqual(got[key].dtype, value.dtype, key)
                held = got[key][got[key] == 0.0] if got[key].dtype.kind == "f" else ()
                self.assertFalse(np.signbit(held).any(), key)
            else:
                self.assertEqual(got[key], value, key)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_round_trip(self):
        """save -> load: the keys float32 of the source, everything else as
        it was; save -> load again: the first load, to the bit"""
        rig, clips, meshes, uvs, skins = build()
        formats.save_fbx(self.path("first.fbx"), meshes, skins, clips, uvs, rig)
        first = formats.load_fbx(self.path("first.fbx"))
        self.assertIsInstance(first, formats.FbxContents)
        for uv in uvs:
            uv.mesh_name = "body"
        self.assert_same(parts(first), parts(formats.FbxContents(rig, clips, meshes, uvs, skins), keys=True))

        formats.save_fbx(self.path("again.fbx"), first)
        self.assert_same(parts(formats.load_fbx(self.path("again.fbx"))), parts(first))
        first.save_fbx(self.path("method.fbx"))
        self.assert_same(parts(formats.load_fbx(self.path("method.fbx"))), parts(first))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_the_rig_is_at_its_bind(self):
        """the rig is read at the bind pose a skin holds, whatever pose its
        nodes were left in; with no bind pose, at its nodes' own values"""
        from cgmath._tests.test_save_fbx import rewrite

        rig, clips, meshes, uvs, skins = build()
        formats.save_fbx(self.path("bound.fbx"), rig, meshes, skins)

        def bend(scene):
            for index in range(scene.GetNodeCount()):
                node = scene.GetNode(index)
                if node.GetName() == "spine":
                    node.LclRotation.Set(FBX.FbxDouble3(40.0, 50.0, 60.0))

        rewrite(self.path("bound.fbx"), self.path("bent.fbx"), bend)
        bent = formats.load_fbx(self.path("bent.fbx"))
        np.testing.assert_allclose(bent.rig["spine"].rotate, rig["spine"].rotate, rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(HierarchyData.load_fbx(self.path("bent.fbx"))["spine"].rotate),
                                   (40.0, 50.0, 60.0), rtol=0, atol=1e-12)

        formats.save_fbx(self.path("free.fbx"), rig, clips)
        free = formats.load_fbx(self.path("free.fbx"))
        self.assertEqual((len(free.meshes), len(free.uvs), len(free.skins), free.clips.take),
                         (0, 0, 0, ["idle", "walk"]))
        for node in rig:
            np.testing.assert_array_equal(free.rig[node.name].rotate, node.rotate)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_shear(self):
        """a bind pose that shears a group raises ValueError naming it, as a
        save of a sheared rig does; an uneven scale without shear loads"""
        from cgmath._tests.test_save_fbx import rewrite

        rig, clips, meshes, uvs, skins = build()
        formats.save_fbx(self.path("bound.fbx"), rig, meshes, skins)
        loaded = formats.load_fbx(self.path("bound.fbx"))
        self.assertFalse(np.asarray(loaded.rig.shear).any())
        self.assertGreater(np.ptp(np.asarray(loaded.rig["rig"].scale)), 0.5)

        def shear(scene):
            for index in range(scene.GetPoseCount()):
                pose = scene.GetPose(index)
                for k in range(pose.GetCount()):
                    node = pose.GetNode(k)
                    if node.GetName() != "body":
                        continue
                    held    = pose.GetMatrix(k)
                    rows    = np.array([[held.Get(r, c) for c in range(4)] for r in range(4)])
                    rows    = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ rows
                    sheared = FBX.FbxMatrix()
                    for r in range(4):
                        for c in range(4):
                            sheared.Set(r, c, float(rows[r, c]))
                    pose.Remove(k)
                    pose.Add(node, sheared)
                    return

        rewrite(self.path("bound.fbx"), self.path("sheared.fbx"), shear)
        self.assertGreater(np.abs(HierarchyData.load_fbx(self.path("sheared.fbx"), pose="bind")["body"].shear).max(), 0.1)
        with self.assertRaisesRegex(ValueError, r"load_fbx: fbx has no shear.*the rig's \['body'\]"):
            formats.load_fbx(self.path("sheared.fbx"))

        sheared               = rig.copy()
        sheared["body"].shear = (0.0, 0.25, 0.0)
        with self.assertRaisesRegex(ValueError, r"save_fbx: fbx has no shear.*the rig's \['body'\]"):
            formats.save_fbx(self.path("x.fbx"), formats.FbxContents(sheared, clips, meshes, uvs, skins))
        self.assertFalse(os.path.exists(self.path("x.fbx")))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_what_it_ignores(self):
        """a camera, a light, a material, a blend shape and a custom property
        load without an error, left out; the rest reads as it was"""
        from cgmath._tests.test_save_fbx import rewrite

        rig, clips, meshes, uvs, skins = build()
        formats.save_fbx(self.path("plain.fbx"), rig, clips, meshes, uvs, skins)

        def add(scene):
            root   = scene.GetRootNode()
            camera = FBX.FbxNode.Create(scene, "shot")
            camera.SetNodeAttribute(FBX.FbxCamera.Create(scene, "shotShape"))
            light = FBX.FbxNode.Create(scene, "key")
            light.SetNodeAttribute(FBX.FbxLight.Create(scene, "keyShape"))
            root.AddChild(camera)
            root.AddChild(light)
            for index in range(scene.GetNodeCount()):
                node = scene.GetNode(index)
                if node.GetName() != "body":
                    continue
                node.AddMaterial(FBX.FbxSurfacePhong.Create(scene, "skin"))
                prop = FBX.FbxProperty.Create(node, FBX.FbxDoubleDT, "custom", "custom")
                prop.ModifyFlag(FBX.FbxPropertyFlags.EFlags.eUserDefined, True)
                prop.Set(2.5)
                mesh  = node.GetMesh()
                blend = FBX.FbxBlendShape.Create(scene, "faces")
                track = FBX.FbxBlendShapeChannel.Create(scene, "smile")
                shape = FBX.FbxShape.Create(scene, "smileShape")
                shape.InitControlPoints(mesh.GetControlPointsCount())
                for i in range(mesh.GetControlPointsCount()):
                    point = mesh.GetControlPointAt(i)
                    shape.SetControlPointAt(FBX.FbxVector4(point[0], point[1] + 1.0, point[2], 1.0), i)
                track.AddTargetShape(shape)
                blend.AddBlendShapeChannel(track)
                mesh.AddDeformer(blend)

        rewrite(self.path("plain.fbx"), self.path("more.fbx"), add)
        plain, more = formats.load_fbx(self.path("plain.fbx")), formats.load_fbx(self.path("more.fbx"))
        want = {k: v for k, v in parts(plain).items() if not k.startswith(("rig", "take"))}
        self.assert_same({k: v for k, v in parts(more).items() if not k.startswith(("rig", "take"))}, want)
        for node in rig:
            for channel in CHANNELS:
                np.testing.assert_array_equal(getattr(more.rig[node.name], channel), getattr(plain.rig[node.name], channel))
        formats.save_fbx(self.path("again.fbx"), more)
        self.assert_same(parts(formats.load_fbx(self.path("again.fbx"))), parts(more))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_one_frame_take(self):
        """a take of one frame at frame 0 reads back with its own frame and
        values, alone, first or beside others, through ClipList too"""
        rig           = HierarchyData()
        a             = TransformData("a", node_type="joint", translate=(0.0, 1.0, 0.0))
        b             = TransformData("b", node_type="joint", translate=(0.0, 2.0, 0.0))
        b.parent_uuid = a.uuid
        rig.append(a)
        rig.append(b)
        ntsc = 30000.0 / 1001.0

        def take(name, count, start, fps):
            clip               = ClipData(rig, frames=count, start_frame=start, fps=fps, take=name)
            rotate             = np.array(clip.frames.rotate)
            rotate[:, :, 1]    = np.arange(count)[:, None] * 2.0 + 5.0 + start
            clip.frames.rotate = rotate
            return clip

        cases = (
            [("one", 1, 0, 24.0)],
            [("long", 200, 1001, ntsc), ("one", 1, 0, ntsc)],
            [("long", 5, 10, 24.0), ("one", 1, 0, 24.0)],
            [("one", 1, 0, 24.0), ("long", 5, 10, 24.0)],
            [("neg", 3, -5, 24.0), ("one", 1, 0, 24.0)],
            [("one", 1, 5, 24.0), ("two", 2, 0, 24.0)],
        )
        for index, specs in enumerate(cases):
            clips = ClipList([take(*spec) for spec in specs])
            formats.save_fbx(self.path(f"takes{index}.fbx"), rig, clips)
            clips.save_fbx(self.path(f"list{index}.fbx"))
            for read in (formats.load_fbx(self.path(f"takes{index}.fbx")).clips,
                         ClipList.load_fbx(self.path(f"list{index}.fbx"))):
                self.assertEqual(read.take, [clip.take for clip in clips], index)
                for got, want in zip(read, clips):
                    label = (index, want.take)
                    self.assertEqual((got.start_frame, got.frame_count), (want.start_frame, want.frame_count), label)
                    np.testing.assert_array_equal(np.asarray(got.frames.rotate),
                                                  np.asarray(want.frames.rotate).astype(np.float32), err_msg=str(label))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_visibility_radius_draw_style(self):
        """a node's visibility and a joint's radius and draw style read back
        as saved, held as Maya's exporter writes them: Visibility, Size =
        radius * 100 / 3, Show off for draw style None; a file's own nodes
        read as Maya's importer reads them"""
        from cgmath._tests.test_save_fbx import rewrite

        rig = HierarchyData()
        a = TransformData("a", node_type="joint", translate=(0.0, 1.0, 0.0), visibility=False, radius=2.5,
                          draw_style=2)
        b             = TransformData("b", node_type="joint", radius=0.1, draw_style=1)
        c             = TransformData("c", node_type="joint", radius=12.345, draw_style=3)
        g             = TransformData("g", visibility=False)
        spot          = TransformData("spot", node_type="locator", visibility=False)
        b.parent_uuid = a.uuid
        c.parent_uuid = b.uuid
        for node in (a, b, c, g, spot):
            rig.append(node)
        formats.save_fbx(self.path("look.fbx"), rig)
        back = formats.load_fbx(self.path("look.fbx")).rig
        for node in rig:
            got = back[node.name]
            self.assertEqual((got.visibility, got.radius, got.draw_style), (node.visibility, node.radius, node.draw_style),
                             node.name)

        held = {}

        def add(scene):
            for index in range(scene.GetNodeCount()):
                node = scene.GetNode(index)
                attr = node.GetNodeAttribute()
                size = attr.FindProperty("Size") if attr else None
                held[node.GetName()] = (float(node.Visibility.Get()), bool(node.Show.Get()),
                                        float(FBX.FbxPropertyDouble1(size).Get()) if size is not None and size.IsValid()
                                        else None)
            root = scene.GetRootNode()
            for name, joint, visibility, show, size in (("m_joint", True, 1.0, False, None), ("m_size", True, 1.0, True, 50.0),
                                                        ("m_vis", False, 0.3, True, None),
                                                        ("m_hide", False, 0.0, True, None)):
                node = FBX.FbxNode.Create(scene, name)
                attr = FBX.FbxSkeleton.Create(scene, "") if joint else FBX.FbxNull.Create(scene, "")
                if joint:
                    attr.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
                if size is not None:
                    attr.Size.Set(size)
                node.SetNodeAttribute(attr)
                node.Visibility.Set(visibility)
                node.Show.Set(show)
                root.AddChild(node)

        rewrite(self.path("look.fbx"), self.path("more.fbx"), add)
        third = 100.0 / 3.0
        self.assertEqual(held["a"],        (0.0, False, 2.5 * third))
        self.assertEqual(held["b"],        (1.0, True, 0.1 * third))
        self.assertEqual(held["c"],        (1.0, True, 12.345 * third))
        self.assertEqual(held["g"][:2],    (0.0, True))
        self.assertEqual(held["spot"][:2], (0.0, False))
        more = HierarchyData.load_fbx(self.path("more.fbx"))
        for name, want in (("m_joint", (True, 3.0, 2)), ("m_size", (True, 1.5, 0)), ("m_vis", (True, 1.0, 0)),
                           ("m_hide", (False, 1.0, 0)), ("a", (False, 2.5, 2)), ("c", (True, 12.345, 3))):
            got = more[name]
            self.assertEqual((got.visibility, got.radius, got.draw_style), want, name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_group_axes(self):
        """a group's or a locator's rotate axis reads back as saved, rig and
        takes; the file holds it as the post rotation, no pre rotation. A
        file's own group reads as Maya's importer reads it: its post
        rotation its rotate axis, its pre rotation folded into its rotate"""
        from cgmath._tests.test_save_fbx import rewrite
        from cgmath.formats.fbx import _post_rotation

        rig = HierarchyData()
        top = TransformData("top", translate=(1.0, 2.0, 3.0), rotate=(14.0, -25.0, 36.0),
                            rotate_axis=(-20.0, 95.0, -130.0), rotate_order=4)
        hips = TransformData("hips", node_type="joint", translate=(0.0, 5.0, 0.0), joint_orient=(0.0, 90.0, 0.0))
        spot = TransformData("spot", node_type="locator", translate=(0.0, 1.0, 0.0), rotate=(3.0, 2.0, 1.0),
                             rotate_axis=(179.0, -91.0, 0.5), rotate_order=2)
        bare             = TransformData("bare", rotate_axis=(0.0, 0.0, 45.0))
        hips.parent_uuid = top.uuid
        spot.parent_uuid = hips.uuid
        bare.parent_uuid = hips.uuid
        for node in (top, hips, spot, bare):
            rig.append(node)
        clip               = ClipData(rig, frames=4, start_frame=0, fps=24.0, take="move")
        clip.frames.rotate = np.asarray(clip.frames.rotate) + np.arange(4)[:, None, None] * [7.0, -3.0, 5.0]
        formats.save_fbx(self.path("axes.fbx"), rig, ClipList([clip]))
        first = formats.load_fbx(self.path("axes.fbx"))
        for node in rig:
            for channel in CHANNELS:
                np.testing.assert_array_equal(getattr(first.rig[node.name], channel), getattr(node, channel),
                                              err_msg=f"{node.name} {channel}")
        back = first.clips[0]
        for channel in CHANNELS:
            want = np.asarray(getattr(clip.frames, channel))
            if channel in ("translate", "rotate", "scale"):
                want = want.astype(np.float32).astype(np.float64)
            np.testing.assert_array_equal(np.asarray(getattr(back.frames, channel)), want, err_msg=channel)
        np.testing.assert_allclose(np.asarray(back.frames.world_matrix), np.asarray(clip.frames.world_matrix), rtol=0,
                                   atol=1e-5)
        formats.save_fbx(self.path("again.fbx"), first)
        self.assert_same(parts(formats.load_fbx(self.path("again.fbx"))), parts(first))

        source = FBX.FbxNode.EPivotSet.eSourcePivot
        held   = {}

        def add(scene):
            for index in range(scene.GetNodeCount()):
                node                 = scene.GetNode(index)
                held[node.GetName()] = (tuple(node.GetPreRotation(source))[:3], tuple(node.GetPostRotation(source))[:3])
            root = scene.GetRootNode()
            for name, pre, post in (("m_post", None, (-5.0, 15.0, 25.0)), ("m_both", (10.0, 20.0, 30.0), (-5.0, 15.0, 25.0)),
                                    ("m_pre", (10.0, 20.0, 30.0), None)):
                node = FBX.FbxNode.Create(scene, name)
                node.SetNodeAttribute(FBX.FbxNull.Create(scene, ""))
                node.SetRotationActive(True)
                if pre is not None:
                    node.SetPreRotation(source, FBX.FbxVector4(*pre))
                if post is not None:
                    node.SetPostRotation(source, FBX.FbxVector4(*post))
                node.LclRotation.Set(FBX.FbxDouble3(14.0, -25.0, 36.0))
                root.AddChild(node)

        rewrite(self.path("axes.fbx"), self.path("maya.fbx"), add)
        self.assertEqual(held["top"][0], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(held["spot"][1], tuple(_post_rotation(spot.rotate_axis))[:3], rtol=0, atol=1e-12)
        maya = HierarchyData.load_fbx(self.path("maya.fbx"))
        axis = (11.052367, -11.352436, -26.760598)
        for name, rotate, rotate_axis in (("m_post", (14.0, -25.0, 36.0), axis), ("m_both", (33.998415, -13.135141, 57.603958), axis),
                                          ("m_pre", (33.998415, -13.135141, 57.603958), (0.0, 0.0, 0.0))):
            got = maya[name]
            np.testing.assert_allclose(got.rotate, rotate, rtol=0, atol=1e-6, err_msg=name)
            np.testing.assert_allclose(got.rotate_axis, rotate_axis, rtol=0, atol=1e-6, err_msg=name)
            np.testing.assert_array_equal(got.joint_orient, (0.0, 0.0, 0.0), err_msg=name)

    def test_refusals(self):
        """no file: FileNotFoundError; no sdk: ImportError, before the file
        is looked for"""
        from cgmath.hierarchy import hierarchy

        if FBX is not None:
            with self.assertRaises(FileNotFoundError):
                formats.load_fbx(self.path("none.fbx"))
        with mock.patch.object(hierarchy, "FBX", None), self.assertRaisesRegex(ImportError, "fbx module not found"):
            formats.load_fbx(self.path("none.fbx"))


if __name__ == "__main__":
    unittest.main()
