"""A file's takes: the clip loaders, one take at a time and every take.

Its FBX files are built through the SDK (mayapy only), its GLB files through
pygltflib, here, keying each take apart, so a loader that read another take
than the one asked for shows.
"""

import copy
import hashlib
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import numpy as np
from cgmath.data import FORWARD_MARK
from cgmath.formats._fbx_io import load_sdk
from cgmath.formats.glb import load_model, pygltflib
from cgmath.hierarchy import ClipData, ClipList, HierarchyData, PerTake, TransformData
from cgmath.hierarchy.hierarchy import _take_names

try:
    import jedi
except ImportError:
    jedi = None

if pygltflib is not None:
    from pygltflib import (
        Accessor,
        Animation,
        AnimationChannel,
        AnimationChannelTarget,
        AnimationSampler,
        Buffer,
        BufferView,
        FLOAT,
        GLTF2,
        Node,
        Scene,
    )

# the Autodesk SDK, told by its own classes, or None
FBX = load_sdk()

# Maya's own walk cycle, where this interpreter is a Maya's: the clip loaders'
# literals were taken on it
WALK = os.path.join(
    os.path.dirname(os.path.dirname(sys.executable)), "Examples", "Animation", "Motion_Capture", "FBX", "Walk.fbx"
)

# a chain of three joints: (name, parent, translate, joint orient, rotate order)
CHAIN = (
    ("hips", None, (0.0, 90.0, 0.0), (0.0, 0.0, 0.0), 0),
    ("spine", "hips", (0.0, 12.0, 1.5), (10.0, 0.0, 5.0), 2),
    ("head", "spine", (0.0, 25.0, 0.0), (0.0, -15.0, 0.0), 4),
)

# three takes, each keying its own channels over its own frames at 24 fps,
# the first the node with no attribute above the chain, which the read folds
# into the hips every frame, the third from a key between two frames:
# (name, {node: {property: [(frame, (x, y, z))]}})
TAKES = (
    ("walk", {"hips": {"t": [(0, (0, 90, 0)), (12, (5, 92, 1)), (24, (10, 90, 0))]},
              "spine": {"r": [(0, (0, 0, 0)), (24, (15, -5, 3))]},
              "offset": {"r": [(0, (0, 30, 0)), (24, (0, 60, 10))]}}),
    ("idle", {"head": {"r": [(6, (0, 10, 0)), (14, (0, -10, 0)), (30, (0, 0, 0))]}}),
    ("idle", {"hips": {"s": [(0, (1, 1, 1)), (10, (1, 1.2, 1))]},
              "head": {"t": [(2.4, (0, 25, 0)), (9, (0, 26, 1))]}}),
)


def build_fbx(path, takes=TAKES, up_axis=1, unit=1.0, mixed=False):
    """an fbx of ``CHAIN`` under a node with no attribute, ``offset``, turned
    and moved, keyed by ``takes``, one stack each, at 24 fps; ``up_axis`` 2
    for a z-up file, ``unit`` the cm a unit holds; ``mixed``, a group
    between the spine and the head and a locator under the head"""
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene    = FBX.FbxScene.Create(manager, "")
        settings = scene.GetGlobalSettings()
        if up_axis == 2:
            settings.SetAxisSystem(FBX.FbxAxisSystem(FBX.FbxAxisSystem.EPreDefinedAxisSystem.eMax))
        settings.SetSystemUnit(FBX.FbxSystemUnit(unit))
        settings.SetTimeMode(FBX.FbxTime.EMode.eFrames24)
        pivot  = FBX.FbxNode.EPivotSet.eSourcePivot
        offset = FBX.FbxNode.Create(scene, "offset")
        offset.LclTranslation.Set(FBX.FbxDouble3(5.0, 0.0, 0.0))
        offset.LclRotation.Set(FBX.FbxDouble3(0.0, 30.0, 0.0))
        scene.GetRootNode().AddChild(offset)
        nodes = {"offset": offset}
        for name, parent, translate, orient, order in CHAIN:
            node = FBX.FbxNode.Create(scene, name)
            attr = FBX.FbxSkeleton.Create(scene, "")
            attr.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
            node.SetNodeAttribute(attr)
            node.LclTranslation.Set(FBX.FbxDouble3(*translate))
            node.SetRotationActive(True)
            node.SetPreRotation(pivot, FBX.FbxVector4(*orient))
            node.SetRotationOrder(pivot, FBX.EFbxRotationOrder(order))
            nodes[parent or "offset"].AddChild(node)
            nodes[name] = node
        if mixed:
            bare = FBX.FbxNull.Create(scene, "")
            bare.Look.Set(FBX.FbxNull.ELook.eNone.value)
            group = FBX.FbxNode.Create(scene, "grp")
            group.SetNodeAttribute(bare)
            group.LclTranslation.Set(FBX.FbxDouble3(0.0, 4.0, 0.0))
            group.LclRotation.Set(FBX.FbxDouble3(0.0, 0.0, 20.0))
            nodes["spine"].RemoveChild(nodes["head"])
            group.AddChild(nodes["head"])
            nodes["spine"].AddChild(group)
            look = FBX.FbxNull.Create(scene, "")
            look.Look.Set(FBX.FbxNull.ELook.eCross.value)
            locator = FBX.FbxNode.Create(scene, "loc")
            locator.SetNodeAttribute(look)
            locator.LclTranslation.Set(FBX.FbxDouble3(0.0, 3.0, 0.0))
            nodes["head"].AddChild(locator)

        props = {"t": "LclTranslation", "r": "LclRotation", "s": "LclScaling"}
        for take, keys in takes:
            stack = FBX.FbxAnimStack.Create(scene, take)
            layer = FBX.FbxAnimLayer.Create(scene, "BaseLayer")
            stack.AddMember(layer)
            frames = []
            for joint, channels in keys.items():
                for short, pairs in channels.items():
                    prop = getattr(nodes[joint], props[short])
                    for axis, component in enumerate("XYZ"):
                        curve = prop.GetCurve(layer, component, True)
                        curve.KeyModifyBegin()
                        for frame, values in pairs:
                            time = FBX.FbxTime()
                            time.SetSecondDouble(frame / 24.0)
                            index = curve.KeyAdd(time)[0]
                            curve.KeySetValue(index, float(values[axis]))
                            curve.KeySetInterpolation(
                                index, FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
                            )
                            frames.append(frame)
                        curve.KeyModifyEnd()
            first, last = FBX.FbxTime(), FBX.FbxTime()
            first.SetSecondDouble(min(frames) / 24.0)
            last.SetSecondDouble(max(frames) / 24.0)
            stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))

        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


# three animations of a three-node chain, the second unnamed, the first and
# third both named walk: (name, node, path, times, values)
ANIMATIONS = (
    ("walk", 0, "translation", (0.0, 0.5, 1.0), ((0, 1, 0), (2, 1.5, 0), (4, 1, 0))),
    (None, 1, "rotation", (0.0, 1.0), ((0, 0, 0, 1), (0, 0.38268343, 0, 0.9238795))),
    ("walk", 2, "scale", (0.25, 0.75), ((1, 1, 1), (1, 2, 1))),
)


def build_glb(path, animations=ANIMATIONS):
    """a glb of a three-node chain, one glTF animation per entry"""
    blob = b""
    views, accessors, gltf_animations = [], [], []

    def accessor(array, kind):
        nonlocal blob
        data = np.asarray(array, dtype=np.float32)
        views.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=data.nbytes))
        blob += data.tobytes()
        accessors.append(
            Accessor(
                bufferView    = len(views) - 1,
                componentType = FLOAT,
                count         = len(data),
                type          = kind,
                min           = [float(data.min())] if kind == "SCALAR" else None,
                max           = [float(data.max())] if kind == "SCALAR" else None,
            )
        )
        return len(accessors) - 1

    kinds = {"translation": "VEC3", "rotation": "VEC4", "scale": "VEC3"}
    for name, node, path_name, times, values in animations:
        sampler = AnimationSampler(input=accessor(times, "SCALAR"), output=accessor(values, kinds[path_name]))
        channel = AnimationChannel(sampler=0, target=AnimationChannelTarget(node=node, path=path_name))
        gltf_animations.append(Animation(name=name, samplers=[sampler], channels=[channel]))

    gltf       = GLTF2()
    gltf.scene = 0
    gltf.nodes = [
        Node(name="root", translation=[0.0, 1.0, 0.0], children=[1]),
        Node(name="mid", translation=[0.0, 1.0, 0.0], children=[2]),
        Node(name="tip", translation=[0.0, 1.0, 0.0]),
    ]
    gltf.scenes      = [Scene(nodes=[0])]
    gltf.buffers     = [Buffer(byteLength=len(blob))]
    gltf.bufferViews = views
    gltf.accessors   = accessors
    gltf.animations  = gltf_animations
    gltf.set_binary_blob(blob)
    gltf.save(path)
    return path


def fingerprint(clip) -> dict:
    """what the loaders' literals hold of a clip: its nodes, its timebase and
    a sha256 of each channel block, -0.0 written as 0.0, so the hash compares
    as np.array_equal does; a shear block all zero is left out, as a save
    leaves it out: the literals come from before a clip framed shear"""
    names  = "\n".join(str(node.name) for node in clip)
    blocks = {}
    for channel in sorted(clip._blocks):
        if channel == "_shear" and not np.any(clip._blocks[channel]):
            continue
        block           = np.ascontiguousarray(clip._blocks[channel], dtype=np.float64) + 0.0
        blocks[channel] = hashlib.sha256(block.tobytes()).hexdigest()
    return {
        "names":  hashlib.sha256(names.encode("utf-8")).hexdigest(),
        "start":  int(clip.start_frame),
        "fps":    float(clip.fps),
        "frames": int(clip.frame_count),
        "blocks": blocks,
    }


# ClipData.load_fbx and load_glb before their per-take bodies moved out
# (capture_clip_loaders.py, run on the code they moved from)
LITERALS = {'fbx 0': {'blocks': {'_joint_orient': 'd760653dad991759992423796bf2a1253f3105d1c4d28e81cfb0904c972f9804',
                      '_rotate':      'd55df23cf2640b6dd36d8d8bcfd5a766fdb1e663b073fe73b7aafc1ee4c9ed23',
                      '_rotate_axis': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_scale':       '39503975da8ecde4ca3e66183506c471ee10f9fc43d08d10026ce7dc4862cdae',
                      '_translate': '3f38b1f2243cf062be8445b49bd702e16626b4ff55d57f0ea957f806234fa07f'},
           'fps':    24.0,
           'frames': 25,
           'names':  '64154de79b90f7ad232c755012e1de9aae11cb45167ab71aa864e54396c12b6c',
           'start': 0},
 'fbx 1': {'blocks': {'_joint_orient': '0ef45a2f73830fe196ffe249b6105065d0b2e9ebb51b3f9830585eb66fee8f22',
                      '_rotate':      '266a2215586389e4230e66c30143015c9019cb00947abd2566c14f8302f69de2',
                      '_rotate_axis': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_scale':       '39503975da8ecde4ca3e66183506c471ee10f9fc43d08d10026ce7dc4862cdae',
                      '_translate': '030222a6e2b03421a996333cac2c6b5f82b843261b807a0fc49268e1bf99cf1e'},
           'fps':    24.0,
           'frames': 25,
           'names':  '64154de79b90f7ad232c755012e1de9aae11cb45167ab71aa864e54396c12b6c',
           'start': 6},
 'fbx 2': {'blocks': {'_joint_orient': '8ab368bb38a36c9fc73219ed9ae8e186576803c50cdd1b319e2fb0f5ab473be8',
                      '_rotate':      '52f033ec34066343ced3a1bed745100fe4efe6577154125a6d9483c6b482219a',
                      '_rotate_axis': '52f033ec34066343ced3a1bed745100fe4efe6577154125a6d9483c6b482219a',
                      '_scale':       'c1904c6ada04bffed12cf7b44ee0b2ece4ad2b92fcb1634d27a7850a3c099095',
                      '_translate': 'efeaf5f0a1c6823aa078ac28cf184c9999430990e7f1dc1829aae7222c6bbc77'},
           'fps':    24.0,
           'frames': 11,
           'names':  '64154de79b90f7ad232c755012e1de9aae11cb45167ab71aa864e54396c12b6c',
           'start': 0},
 'fbx default': {'blocks': {'_joint_orient': 'd760653dad991759992423796bf2a1253f3105d1c4d28e81cfb0904c972f9804',
                            '_rotate':      'd55df23cf2640b6dd36d8d8bcfd5a766fdb1e663b073fe73b7aafc1ee4c9ed23',
                            '_rotate_axis': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                            '_scale':       '39503975da8ecde4ca3e66183506c471ee10f9fc43d08d10026ce7dc4862cdae',
                            '_translate': '3f38b1f2243cf062be8445b49bd702e16626b4ff55d57f0ea957f806234fa07f'},
                 'fps':    24.0,
                 'frames': 25,
                 'names':  '64154de79b90f7ad232c755012e1de9aae11cb45167ab71aa864e54396c12b6c',
                 'start': 0},
 'fbx z-up': {'blocks': {'_joint_orient': 'b0a434ef068fcb517c51da57c9504a4db16e1a3837fa541d14d8f3687d486e51',
                         '_rotate':      'd55df23cf2640b6dd36d8d8bcfd5a766fdb1e663b073fe73b7aafc1ee4c9ed23',
                         '_rotate_axis': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                         '_scale':       '39503975da8ecde4ca3e66183506c471ee10f9fc43d08d10026ce7dc4862cdae',
                         '_translate': 'f0a7f8c55408d932740b9fd48c4584045b6e86eb193c846c2b2c226fa69b771a'},
              'fps':    24.0,
              'frames': 25,
              'names':  '64154de79b90f7ad232c755012e1de9aae11cb45167ab71aa864e54396c12b6c',
              'start': 0},
 'glb 0': {'blocks': {'_joint_orient': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_rotate':      '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_rotate_axis': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_scale':       '39503975da8ecde4ca3e66183506c471ee10f9fc43d08d10026ce7dc4862cdae',
                      '_translate': '8f5c403a10ec9c700ba023d7d130a2dec8633d4e0a11b97dc29d3bb554eb01e6'},
           'fps':    24.0,
           'frames': 25,
           'names':  '8b51a5d4241b9866b6eb7a5d19bbf03ddddd4d0ff67549127cbf585b916fc44b',
           'start': 0},
 'glb 1': {'blocks': {'_joint_orient': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_rotate':      'cef6b9c92847cb2816d705c1d1ece0d11d73fe27a05bc07fe7483fa660ef968a',
                      '_rotate_axis': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11',
                      '_scale':       '39503975da8ecde4ca3e66183506c471ee10f9fc43d08d10026ce7dc4862cdae',
                      '_translate': '1f7af240088ed426a5b5f66de4e66a0fb7945359c6046924d0abd6172faa1999'},
           'fps':    24.0,
           'frames': 25,
           'names':  '8b51a5d4241b9866b6eb7a5d19bbf03ddddd4d0ff67549127cbf585b916fc44b',
           'start': 0},
 'glb 2': {'blocks': {'_joint_orient': '4db2fd521b198ad08a6602141c428d23a3bf232e85fd7de8418439e89d7b3d04',
                      '_rotate':      '4db2fd521b198ad08a6602141c428d23a3bf232e85fd7de8418439e89d7b3d04',
                      '_rotate_axis': '4db2fd521b198ad08a6602141c428d23a3bf232e85fd7de8418439e89d7b3d04',
                      '_scale':       '2d753427171fda0ca859ca4215e46c172b6cc31fc9deac7f5c1d9b44e3ea16dc',
                      '_translate': '183cb4a809d24a00236f66966a2a7bd0723f0c3ba8dcab8b9f6067f09d6cafef'},
           'fps':    24.0,
           'frames': 13,
           'names':  '8b51a5d4241b9866b6eb7a5d19bbf03ddddd4d0ff67549127cbf585b916fc44b',
           'start': 0},
 'walk': {'blocks': {'_joint_orient': 'ebc372514e4acd657107fc7be8347943c85a2b986356ea9038bd8b01c04839e4',
                     '_rotate':      '5d24b4661541c4d238247559e7e6433ad1922f5caaf99bdb1118d66c55750553',
                     '_rotate_axis': '8b97f6a5d23d19e3f6db02755820c24f4b4a7caa001e48531ea13fd59022ead0',
                     '_scale':       'bab5bb48ed1c4118092664528d51e9469a221cb4023e18e6a1f5570bcd3795d4',
                     '_translate': '6c1635971713251daa8c0bb9de8b419ea42b2dd8e6d257e902b4b3bcb155c0b6'},
          'fps':    24.0,
          'frames': 122,
          'names':  'c2f31c6ed48f2d94e339bb37b051ce094b7200bda32e7897457b109c58568b57',
          'start': 0}}


def hand_built(take="", frames=6, namespace=""):
    """a clip of a two-joint rig, built by hand, its rotate moving"""
    rig = HierarchyData()
    rig.append(TransformData(f"{namespace}root", node_type="joint", translate=(0.0, 1.0, 0.0)))
    rig.append(TransformData(f"{namespace}tip", node_type="joint", translate=(0.0, 2.0, 0.0)))
    rig[1].parent_uuid = rig[0].uuid
    clip               = ClipData(rig, frames=frames, take=take)
    turns              = np.array(clip.frames.rotate)
    turns[:, :, 2] += np.arange(frames)[:, None] * 5.0
    clip.frames.rotate = turns
    return clip


class TestClipTake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_saved_only_when_set(self):
        plain = hand_built()
        self.assertEqual(plain.take, "")
        tree = plain.to_dict()
        self.assertNotIn("take", tree)
        self.assertNotIn("_take", tree)

        named = hand_built(take="idle")
        tree  = named.to_dict()
        self.assertEqual(tree["take"], "idle")
        self.assertNotIn("_take", tree)
        for ext in ("pkl", "npz", "json"):
            with self.subTest(ext):
                path = named.save(os.path.join(self.tmp.name, f"clip.{ext}"))
                self.assertEqual(ClipData.load(path).take, "idle")
        tree["_take"] = tree.pop("take")
        self.assertEqual(ClipData.from_dict(tree).take, "idle")
        self.assertNotEqual(named, hand_built(take="walk"))

    def test_a_take_is_a_str(self):
        with self.assertRaisesRegex(TypeError, "a take's name is a str"):
            ClipData(take=3)
        clip = hand_built()
        with self.assertRaisesRegex(TypeError, "a take's name is a str"):
            clip.take = None
        clip.take = "run"
        self.assertEqual(clip.take, "run")

    def test_carried(self):
        clip = hand_built(take="idle")
        pose = HierarchyData(node.copy() for node in clip)
        for label, made in (
            ("copy", lambda: clip.copy()),
            ("deepcopy", lambda: copy.deepcopy(clip)),
            ("pickle", lambda: pickle.loads(pickle.dumps(clip))),
            ("cut", lambda: clip.frames[2:5]),
            ("cut backwards", lambda: clip.frames[::-2]),
            ("delta against a pose", lambda: clip.get_delta(pose)),
            ("delta against a clip", lambda: clip.get_delta(hand_built())),
            ("one delta added", lambda: clip.add_delta(pose)),
            ("a clip of deltas added", lambda: clip.add_delta(hand_built())),
        ):
            with self.subTest(label):
                self.assertEqual(made().take, "idle")
        self.assertEqual(ClipData.from_poses([pose, pose]).take, "")

    def test_take_names(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(_take_names(["idle", "idle", "idle_1"]), ["idle", "idle_2", "idle_1"])
        self.assertEqual(len(caught), 1)
        self.assertEqual(
            str(caught[0].message), "takes that would read under one name are told apart: take 1 'idle' as 'idle_2'"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(_take_names(["walk", "idle", "idle"], read=[0]), ["walk", "idle", "idle_1"])
        self.assertEqual(caught, [])

    def test_blank_names(self):
        """a blank name, empty or whitespace only, reads
        by the take's position, Take 001, Take 002 ..., before two takes of
        one name are told apart; a glb's as an animation with no name, the
        warning naming the take as the file does"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(_take_names(["", " ", "idle"]), ["Take 001", "Take 002", "idle"])
            self.assertEqual(_take_names(["\t", "\u00a0\n", "a\tb"]), ["Take 001", "Take 002", "a\tb"])
        self.assertEqual(caught, [])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(_take_names(["idle", "", "Take 002"]), ["idle", "Take 002", "Take 002_1"])
            self.assertEqual(_take_names(["Take 002", "  "]), ["Take 002", "Take 002_1"])
        self.assertEqual(len(caught), 2)
        self.assertIn("take 2 'Take 002' as 'Take 002_1'", str(caught[0].message))
        self.assertIn("take 1 '  ' as 'Take 002_1'", str(caught[1].message))

        if pygltflib is not None:
            animations = [(name,) + ANIMATIONS[0][1:] for name in ("", None, "  ", "animation1")]
            path       = build_glb(os.path.join(self.tmp.name, "blank.glb"), animations=animations)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                names = [ClipData.load_glb(path, animation=i).take for i in range(4)]
            self.assertEqual(names, ["animation0", "animation1", "animation2", "animation1_1"])
            # the warning names the take as the file holds it
            animations = [(name,) + ANIMATIONS[0][1:] for name in ("animation1", "  ")]
            path       = build_glb(os.path.join(self.tmp.name, "spaces.glb"), animations=animations)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(ClipData.load_glb(path, animation=1).take, "animation1_1")
            self.assertEqual(
                [str(w.message) for w in caught],
                ["takes that would read under one name are told apart: take 1 '  ' as 'animation1_1'"],
            )
        if FBX is not None:
            stacks = [(name, TAKES[0][1]) for name in ("", "  ", "idle", "", "Take 002")]
            path   = build_fbx(os.path.join(self.tmp.name, "blank.fbx"), takes=stacks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                names = [ClipData.load_fbx(path, take=i).take for i in range(5)]
            self.assertEqual(names, ["Take 001", "Take 002", "idle", "Take 004", "Take 002_1"])

    @unittest.skipIf(pygltflib is None, "pygltflib is not installed")
    def test_glb_names(self):
        path = build_glb(os.path.join(self.tmp.name, "takes.glb"))
        self.assertEqual([x.name for x in load_model(path).animations], ["walk", None, "walk"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            names = [ClipData.load_glb(path, animation=i).take for i in range(3)]
        self.assertEqual(names, ["walk", "animation1", "walk_1"])
        self.assertEqual(len(caught), 1)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_fbx_names(self):
        path = build_fbx(os.path.join(self.tmp.name, "takes.fbx"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            names = [ClipData.load_fbx(path, take=i).take for i in range(3)]
        self.assertEqual(names, ["walk", "idle", "idle_1"])
        self.assertEqual(len(caught), 1)
        self.assertEqual(ClipData.load_fbx(path).take, "walk")


def two_takes():
    """takes of 6 and 4 frames, idle and walk, of a namespaced rig"""
    return ClipList([hand_built("idle", 6, "ns:"), hand_built("walk", 4, "ns:")])


def channels(clip) -> dict:
    """every block of a clip, copied"""
    return {name: np.array(block) for name, block in clip._blocks.items()}


def whole(clips) -> list:
    """every take as a refused write or edit must leave it, to the bit: its
    blocks and columns, timebase, record, name, bound frame and rest pose
    object, and its nodes in their order, the same objects, each with its
    own fields"""
    held = []
    for clip in clips:
        nodes = [
            (id(n), n.name, n.uuid, n.parent_uuid, n.node_type, int(n.rotate_order), bool(n.segment_scale_compensate),
             bool(n.visibility), float(n.radius), int(n.draw_style), dict(n.user_defined_attributes))
            for n in clip
        ]
        held.append((
            {name: (block.shape, block.tobytes()) for name, block in clip._blocks.items()},
            list(clip._columns),
            (clip.fps, clip.start_frame, clip.unit, clip.axis_system, id(clip.rest_pose), clip.take, clip.frame),
            nodes,
        ))
    return held


def oriented(take, count=3, frames=5, kind="joint"):
    """a clip of a chain of ``count`` nodes of ``kind``, each joint holding
    a joint orient, radius 2.5, draw_style 2 and segment_scale_compensate
    on, the second node a rotate axis, its rotate moving, a rest pose of its
    own, bound to frame 1"""
    rig = HierarchyData()
    for index in range(count):
        rig.append(TransformData(f"n{index}", node_type=kind, translate=(0.0, 1.0 + index, 0.5 * index)))
        if index:
            rig[index].parent_uuid = rig[index - 1].uuid
        if kind == "joint":
            rig[index].joint_orient             = (10.0 * index - 5.0, 15.0, 5.0 * index)
            rig[index].radius                   = 2.5
            rig[index].draw_style               = 2
            rig[index].segment_scale_compensate = True
    if count > 1:
        rig[1].rotate_axis = (3.0, 4.0, 5.0)
    clip  = ClipData(rig, frames=frames, take=take)
    turns = np.array(clip.frames.rotate)
    turns[:, :, 0] += np.arange(frames)[:, None] * 4.0
    turns[:, :, 2] += np.arange(frames)[:, None] * 7.0
    clip.frames.rotate = turns
    clip.rest_pose     = HierarchyData(node.copy() for node in clip)
    clip.frame         = 1
    return clip


def oriented_takes(counts=(3, 3, 3), kind="joint") -> ClipList:
    """takes a, b and c of :func:`oriented`, of 5, 6 and 4 frames"""
    return ClipList([oriented(take, count, frames, kind) for take, count, frames in zip("abc", counts, (5, 6, 4))])


class TestClipList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_built_copied_and_saved(self):
        clips = two_takes()
        self.assertIs(type(clips.list), list)
        for label, made in (
            ("copy.copy", lambda: copy.copy(clips)),
            ("deepcopy", lambda: copy.deepcopy(clips)),
            ("copy()", lambda: clips.copy()),
            ("pickle", lambda: pickle.loads(pickle.dumps(clips))),
            ("pkl", lambda: ClipList.load(clips.save(os.path.join(self.tmp.name, "c.pkl")))),
            ("npz", lambda: ClipList.load(clips.save(os.path.join(self.tmp.name, "c.npz")))),
            ("json", lambda: ClipList.load(clips.save(os.path.join(self.tmp.name, "c.json")))),
        ):
            with self.subTest(label):
                other = made()
                self.assertEqual(other.take, ["idle", "walk"])
                self.assertEqual(other, clips)
                self.assertIs(type(other.list), list)
        clips.take = PerTake(["x", "y"])
        self.assertEqual(clips.take, ["x", "y"])

    def test_a_misspelt_name_changes_no_take(self):
        clips  = two_takes()
        before = [channels(clip) for clip in clips]
        with self.assertRaisesRegex(AttributeError, "'ClipList' has no attribute 'tyop'"):
            clips.tyop = 1
        with self.assertRaises(AttributeError):
            clips.frames.tyop = 1
        with self.assertRaises(AttributeError):
            ClipList().tyop
        for clip, held in zip(clips, before):
            for name, block in channels(clip).items():
                np.testing.assert_array_equal(block, held[name])

    def test_lookup_by_take(self):
        clips = two_takes()
        self.assertIs(clips["walk"], clips[1])
        self.assertIn("idle", clips)
        self.assertNotIn("run", clips)
        self.assertIs(clips.get("walk"), clips[1])
        self.assertEqual(clips.get("run", 7), 7)
        self.assertEqual(clips.index("walk"), 1)
        with self.assertRaisesRegex(KeyError, r"no take named 'run'; the takes are \['idle', 'walk'\]"):
            clips["run"]
        with self.assertRaisesRegex(ValueError, "'run' is not a take of the list"):
            clips.index("run")
        self.assertEqual(clips.name, [["ns:root", "ns:tip"], ["ns:root", "ns:tip"]])

    def test_per_take_values(self):
        clips = two_takes()
        self.assertIsInstance(clips.fps, PerTake)
        self.assertEqual(clips.frame_count, [6, 4])
        values = PerTake([1.0, 2.0])
        self.assertEqual(values + 1, [2.0, 3.0])
        self.assertEqual(values * PerTake([3.0, 4.0]), [3.0, 8.0])
        self.assertEqual(np.float64(2) * values, [2.0, 4.0])
        self.assertEqual(-values, [-1.0, -2.0])
        self.assertEqual(abs(PerTake([-1.0])), [1.0])
        held = values
        values += 5
        self.assertEqual(held, [1.0, 2.0])  # the old one, element by element, as it was
        self.assertEqual(values, [6.0, 7.0])
        with self.assertRaisesRegex(ValueError, "2 takes against 3"):
            PerTake([1, 2]) + PerTake([1, 2, 3])

        rotate = [np.array(x) for x in clips.frames.rotate]
        clips.frames.rotate += 5  # read-only views: a new PerTake, then the setter
        for clip, before in zip(clips, rotate):
            np.testing.assert_allclose(np.asarray(clip.frames.rotate), np.asarray(before) + 5)

    def test_frames_write_every_frame_of_every_take(self):
        clips  = two_takes()
        before = [np.array(clip.frames.rotate) for clip in clips]
        clips.frames.match("ns:tip").rotate_y += 5
        for clip, held in zip(clips, before):
            turned = np.array(clip.frames.rotate)
            np.testing.assert_allclose(turned[:, 1, 1], held[:, 1, 1] + 5)  # tip, every frame
            np.testing.assert_array_equal(turned[:, 0], held[:, 0])         # root untouched
        cut = clips.frames[1:3]
        self.assertIsInstance(cut, ClipList)
        self.assertEqual((cut.take, cut.frame_count), (["idle", "walk"], [2, 2]))

    def test_match_writes_the_current_frame(self):
        clips       = two_takes()
        clips.frame = 2
        before      = [np.array(clip.frames.rotate) for clip in clips]
        clips.match("ns:tip").rotate_y += 5
        for clip, held in zip(clips, before):
            turned = np.array(clip.frames.rotate)
            held[2, 1, 1] += 5
            np.testing.assert_allclose(turned, held)

    def test_one_value_or_one_per_take(self):
        clips     = two_takes()
        clips.fps = 30.0
        self.assertEqual(clips.fps, [30.0, 30.0])
        clips.fps = PerTake([24.0, 25.0])
        self.assertEqual(clips.fps, [24.0, 25.0])

        before = [channels(clip) for clip in clips]
        for label, write in (
            ("a PerTake of 3", lambda: setattr(clips, "fps", PerTake([1.0, 2.0, 3.0]))),
            ("fits take 0 only", lambda: setattr(clips.frames, "rotate", np.zeros((6, 2, 3)))),
            ("one node's flat column", lambda: setattr(clips.frames["ns:tip"], "rotate_y", np.zeros(6))),
        ):
            with self.subTest(label), self.assertRaises(ValueError) as caught:
                write()
            if label != "a PerTake of 3":
                self.assertIn("take 1 ('walk')", str(caught.exception))
        for clip, held in zip(clips, before):
            for name, block in channels(clip).items():
                np.testing.assert_array_equal(block, held[name])

        for scrub in (lambda: setattr(clips, "frame", 5), lambda: clips.frames[5]):
            with self.assertRaisesRegex(IndexError, r"take 1 \('walk'\), which holds 4 frames; no take was scrubbed"):
                scrub()
            self.assertEqual(clips.frame, [0, 0])
        self.assertIs(clips.frames[3], clips)
        self.assertEqual(clips.frame, [3, 3])

    def test_a_refused_write_changes_no_take(self):
        """a write reaches every take or none, every member: a last value
        the last take's own setter refuses leaves every take as it was, the
        takes written before it put back, its rest pose the same object"""

        def made():
            clips = ClipList([hand_built("a", 6), hand_built("b", 6), hand_built("c", 4)])
            for clip in clips:
                clip.rest_pose = HierarchyData(node.copy() for node in clip)
            return clips

        def state(clips):
            return [
                (
                    {name: block.tobytes() for name, block in channels(clip).items()},
                    (clip.fps, clip.start_frame, clip.unit, clip.axis_system, id(clip.rest_pose), clip.take, clip.frame),
                    [(n.name, int(n.rotate_order), n.node_type, bool(n.segment_scale_compensate), n.visibility,
                      n.radius, n.draw_style) for n in clip],
                )
                for clip in clips
            ]

        def moved(value, name):
            """a value of the take's own kind other than the one it holds"""
            if isinstance(value, np.ndarray):
                return np.array(value) + (1 if value.dtype.kind in "fiu" else 0)
            if isinstance(value, bool):
                return not value
            if isinstance(value, (int, float)):
                return value + 1
            if isinstance(value, HierarchyData):
                return HierarchyData(node.copy() for node in value)
            if name == "axis_system":
                return "z up, -y front, right handed"
            return value

        refused = set()
        for owner in ("ClipList", "_Takes"):
            cls = ClipList if owner == "ClipList" else type(made().frames)
            for name, raw in list(vars(cls).items()):
                if not (isinstance(raw, property) and raw.fget.__qualname__.startswith("_per_take.") and raw.fset):
                    continue
                for bad in ("x", object()):
                    with self.subTest(member=f"{owner}.{name}", bad=type(bad).__name__):
                        clips  = made()
                        target = clips if owner == "ClipList" else clips.frames
                        held   = getattr(target, name)
                        before = state(clips)
                        try:
                            setattr(target, name, PerTake([moved(held[0], name), moved(held[1], name), bad]))
                        except (TypeError, ValueError):
                            refused.add(name)
                            self.assertEqual(state(clips), before)
        for name in ("fps", "start_frame", "unit", "rest_pose", "translate", "world_matrix", "rotate_order"):
            self.assertIn(name, refused)

        # takes 0 and 1 turn their joints into transforms, which
        # folds each joint orient into rotate and resets radius, draw_style
        # and segment_scale_compensate, and the last take's list is one
        # short: the whole of every take is put back
        clips  = oriented_takes()
        before = whole(clips)
        with self.assertRaisesRegex(ValueError, "got 2 node types for 3 nodes"):
            clips.node_type = PerTake([["transform"] * 3, ["transform"] * 3, ["transform"] * 2])
        self.assertEqual(whole(clips), before)
        self.assertEqual(clips.node_type, [["joint"] * 3] * 3)
        clips.node_type = "transform"  # one every take holds: as each take alone
        alone           = oriented_takes()
        for clip in alone:
            clip.node_type = "transform"
        for clip, own in zip(clips, alone):
            for name, block in clip._blocks.items():
                np.testing.assert_array_equal(block, own._blocks[name])

        # on groups a sheared matrix writes a shear, which the
        # put back takes back with the other channels
        shear       = np.eye(4)
        shear[1, 0] = 0.3
        for owner, name in (("ClipList", "matrix"), ("ClipList", "world_matrix"), ("_Takes", "matrix")):
            with self.subTest(member=f"{owner}.{name}", nodes="groups"):
                clips  = oriented_takes(kind="transform")
                target = clips if owner == "ClipList" else clips.frames
                held   = getattr(target, name)
                values = [np.asarray(value) @ shear if owner == "_Takes" else shear @ np.asarray(value) for value in held[:2]]
                before = whole(clips)
                with self.assertRaises((TypeError, ValueError)):
                    setattr(target, name, PerTake(values + ["x"]))
                self.assertEqual(whole(clips), before)
                self.assertFalse(any(np.any(clip._blocks["_shear"]) for clip in clips))

    def test_one_rest_pose_is_copied_for_each_take(self):
        """each take holds its own copy of the rest pose, one
        written whole too; a PerTake gives each its own as given"""
        clips           = two_takes()
        rig             = HierarchyData(node.copy() for node in clips[0])
        clips.rest_pose = rig
        rests           = list(clips.rest_pose)
        self.assertEqual(len({id(rest) for rest in rests + [rig]}), 3)
        for rest in rests:
            np.testing.assert_array_equal(np.asarray(rest.world_matrix), np.asarray(rig.world_matrix))
        rests[0][0].translate = (5.0, 6.0, 7.0)
        np.testing.assert_array_equal(np.asarray(rests[1].world_matrix), np.asarray(rig.world_matrix))
        np.testing.assert_array_equal(np.asarray(rig.world_matrix), np.asarray(clips[1].rest_pose.world_matrix))

        own             = PerTake([rig, HierarchyData(node.copy() for node in clips[1])])
        clips.rest_pose = own
        self.assertIs(clips[0].rest_pose, own[0])
        self.assertIs(clips[1].rest_pose, own[1])

    def test_edits_run_on_every_take(self):
        clips = two_takes()
        alone = [clip.copy() for clip in clips]
        clips.set_rotate_order("zyx")
        for clip in alone:
            clip.set_rotate_order("zyx")
        for clip, own in zip(clips, alone):
            np.testing.assert_allclose(np.asarray(clip.frames.rotate), np.asarray(own.frames.rotate))
            self.assertEqual(list(clip.rotate_order), list(own.rotate_order))

        copied = clips.copy()
        copied.frames.rotate += 1
        self.assertFalse(np.allclose(np.asarray(copied[0].frames.rotate), np.asarray(clips[0].frames.rotate)))

        clips.strip_namespace()
        self.assertEqual(clips.name, [["root", "tip"], ["root", "tip"]])

        clash = ClipList([hand_built("a", 3, "ns:"), hand_built("b", 3, "ns:")])
        for clip in clash:
            clip.rest_pose = HierarchyData(node.copy() for node in clip)
        clash[1].append(TransformData("tip", node_type="joint"))  # take 1 would hold two tips
        with self.assertRaises(ValueError):
            clash.strip_namespace()
        self.assertEqual(clash[0].name, ["ns:root", "ns:tip"])
        # the rest poses' names put back too, the stripped take's and the refused one's
        self.assertEqual([[node.name for node in clip.rest_pose] for clip in clash], [["ns:root", "ns:tip"]] * 2)

    def refused_edit(self, name, args, refused=()):
        """the rig edit ``name`` reaches every take or
        none. Each of ``refused``, (arguments, error), on takes of 3, 3 and 2
        nodes, which the last take refuses by its rig; and ``args`` on takes
        of one rig, the last take's own edit made to change it and then
        raise. Either way every take is as before, the refusing one
        included, the same objects; ``args``, which every take accepts,
        gives each take what the edit gives it alone"""
        from unittest import mock

        for given, error in refused:
            with self.subTest(arguments=given):
                clips  = oriented_takes((3, 3, 2))
                before = whole(clips)
                with self.assertRaisesRegex(ValueError, error):
                    getattr(clips, name)(*given)
                self.assertEqual(whole(clips), before)

        clips   = oriented_takes()
        before  = whole(clips)
        real    = getattr(ClipData, name)
        reached = []

        def edit(clip, *given, **keywords):
            real(clip, *given, **keywords)
            reached.append(whole([clip])[0] != before[len(reached)])
            if clip is clips[-1]:
                raise RuntimeError(f"{name} refused after its edit")

        with mock.patch.object(ClipData, name, edit):
            with self.assertRaisesRegex(RuntimeError, "refused after its edit"):
                getattr(clips, name)(*args)
        self.assertEqual(reached, [True, True, True])  # every take edited, the last then refusing
        self.assertEqual(whole(clips), before)

        alone = oriented_takes()
        for clip in alone:
            getattr(clip, name)(*args)
        getattr(clips, name)(*args)
        for clip, own in zip(clips, alone):
            names = {node.uuid: node.name for node in clip}
            mine  = {node.uuid: node.name for node in own}
            self.assertEqual([(n.name, names.get(n.parent_uuid), n.node_type, int(n.rotate_order)) for n in clip],
                             [(n.name, mine.get(n.parent_uuid), n.node_type, int(n.rotate_order)) for n in own])
            for channel, block in clip._blocks.items():
                np.testing.assert_array_equal(block, own._blocks[channel])

    def test_set_rotate_order_reaches_every_take_or_none(self):
        refused = (((["zyx", "yxz", "xzy"],), "got 3 rotate orders for 2 nodes"),)
        self.refused_edit("set_rotate_order", ("zyx",), refused)

    def test_swapaxes_reaches_every_take_or_none(self):
        self.refused_edit("swapaxes", (0, 1))

    def test_set_rotate_to_joint_orient_reaches_every_take_or_none(self):
        self.refused_edit("set_rotate_to_joint_orient", ())

    def test_set_joint_orient_to_rotate_reaches_every_take_or_none(self):
        self.refused_edit("set_joint_orient_to_rotate", ())

    def test_set_parent_reaches_every_take_or_none(self):
        refused = ((("n2",), "n2 is not in list"), (("n2", False), "n2 is not in list"))
        self.refused_edit("set_parent", ("n2",), refused)

    def test_an_item_assignment_reaches_every_take_or_none(self):
        """an item assignment on every take's frames
        reaches every take or none: a PerTake of another length changes no
        take, and so does a value the last take refuses, or the last take's
        own write made to change it and then raise, the refusing take put
        back too; a value every take holds gives each what it gives alone"""
        from unittest import mock

        def turned(clips, last=None):
            """each take's clip copied and turned, the last ``last`` if given"""
            values = []
            for clip in clips:
                value               = clip.copy()
                value.frames.rotate = np.asarray(value.frames.rotate) + 10.0
                values.append(value)
            if last is not None:
                values[-1] = last
            return values

        def posed(clips, last=None):
            """each take's pose moved, the last ``last`` if given"""
            values = []
            for clip in clips:
                value              = HierarchyData(node.copy() for node in clip)
                value[0].translate = (4.0, 5.0, 6.0)
                values.append(value)
            if last is not None:
                values[-1] = last
            return values

        short = oriented("x", 2, 4)  # a rig one node short, of the last take's 4 frames
        for label, query, value, error in (
            ("frames pasted", slice(None), lambda c: PerTake(turned(c, short)), "pose holds 2 nodes, clip holds 3"),
            ("a pose", 2, lambda c: PerTake(posed(c, HierarchyData(n.copy() for n in short))), "pose holds 2 nodes"),
            ("a PerTake of 2 for 3 takes", slice(None), lambda c: PerTake(turned(c)[:2]), "2 values for 3 takes"),
        ):
            with self.subTest(label):
                clips  = oriented_takes()
                before = whole(clips)
                with self.assertRaisesRegex(ValueError, error):
                    clips.frames[query] = value(clips)
                self.assertEqual(whole(clips), before)

        clips   = oriented_takes()
        before  = whole(clips)
        axis    = type(clips[0].frames)
        real    = axis.__setitem__
        reached = []

        def written(frames, query, value):
            real(frames, query, value)
            reached.append(whole([frames._clip])[0] != before[len(reached)])
            if frames._clip is clips[-1]:
                raise RuntimeError("refused after its write")

        with mock.patch.object(axis, "__setitem__", written):
            with self.assertRaisesRegex(RuntimeError, "refused after its write"):
                clips.frames[2] = PerTake(posed(clips))
        self.assertEqual(reached, [True, True, True])
        self.assertEqual(whole(clips), before)

        alone           = oriented_takes()
        values          = turned(clips)
        clips.frames[:] = PerTake(values)
        for clip, value in zip(alone, values):
            clip.frames[:] = value
        clips.frames[1] = PerTake(posed(clips))
        for clip, value in zip(alone, posed(alone)):
            clip.frames[1] = value
        for clip, own in zip(clips, alone):
            for channel, block in clip._blocks.items():
                np.testing.assert_array_equal(block, own._blocks[channel])

    def test_deltas_pair_by_take(self):
        clips = two_takes()
        rest  = ClipList([hand_built("walk", 4, "ns:"), hand_built("idle", 6, "ns:")])
        delta = clips.get_delta(rest)
        self.assertEqual(delta.take, ["idle", "walk"])
        np.testing.assert_allclose(np.asarray(delta[0].frames.rotate), np.asarray(clips[0].get_delta(rest[1]).frames.rotate))
        with self.assertRaisesRegex(ValueError, "holds no take 'walk'"):
            clips.get_delta(ClipList([hand_built("idle", 6, "ns:")]))
        twice = ClipList([hand_built("idle", 6, "ns:"), hand_built("idle", 6, "ns:")])
        with self.assertRaisesRegex(ValueError, r"takes named twice cannot be paired by name: \['idle'\]"):
            clips.get_delta(twice)
        unnamed = ClipList([hand_built("", 6), hand_built("", 4)])
        self.assertEqual(len(unnamed.get_delta(ClipList([hand_built("", 6), hand_built("", 4)]))), 2)
        pose  = HierarchyData(node.copy() for node in clips[0])
        added = clips.add_delta(pose)
        self.assertEqual(added.take, ["idle", "walk"])

    def test_shear_passes_on(self):
        """shear is a channel like the others: every take's at the frame it
        is on and over its every frame, by its long and short names"""
        clips = two_takes()
        clips.frames.shear_xz += 0.25
        clips.frames.shyz = 0.5
        for clip in clips:
            block = np.asarray(clip.frames.shear)
            np.testing.assert_array_equal(block, np.broadcast_to([0.0, 0.25, 0.5], block.shape))
        clips.shxy = 0.125
        self.assertEqual([np.asarray(held).tolist() for held in clips.shear],
                         [np.asarray(clip.shear).tolist() for clip in clips])
        self.assertEqual([np.asarray(held).tolist() for held in clips.sh],
                         [[[0.125, 0.25, 0.5]] * len(clip) for clip in clips])
        self.assertEqual([np.asarray(held).tolist() for held in clips.frames.shear_yz],
                         [[[0.5] * len(clip)] * clip.frame_count for clip in clips])
        self.assertEqual([np.asarray(held).tolist() for held in clips.frames.shear],
                         [np.asarray(clip.frames.shear).tolist() for clip in clips])

    def test_animate_plays_one_take(self):
        from cgmath.render.scene import Scene, _one_take

        clip = hand_built("idle")
        self.assertIs(_one_take(clip), clip)
        self.assertIs(_one_take(ClipList([clip])), clip)
        with self.assertRaisesRegex(ValueError, r"holds 2: \['idle', 'walk'\]; pick one, animate\(clips\['idle'\]\)"):
            Scene().animate(two_takes())
        with self.assertRaisesRegex(ValueError, "holds no take"):
            Scene().animate(ClipList())
        with self.assertRaisesRegex(TypeError, "not str"):
            Scene().animate("idle")

    def test_forwarded_members_are_listed(self):
        """every member a ClipList passes on is a line of its own
        class, in dir() and in REFERENCE.md"""
        lines = [name for name, raw in vars(ClipList).items()
                 if isinstance(raw, property) and raw.fget.__qualname__.startswith("_per_take.")
                 or getattr(raw, FORWARD_MARK, None) is not None]
        self.assertGreater(len(lines), 60)
        for name in ("strip_namespace", "fps", "t", "frames", "take", "set_rotate_order", "wm"):
            self.assertIn(name, dir(ClipList))
        reference = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REFERENCE.md")
        with open(reference, encoding="utf-8") as handle:
            text = handle.read()
        start   = text.index("ClipList(DataList)")
        section = text[start : text.index("\n#### ", start)]  # up to the next class
        listed  = section.splitlines()
        for name in lines:
            with self.subTest(name):
                self.assertRegex(section, rf"(?m)^{name}\b")
                raw = vars(ClipList)[name]
                if isinstance(raw, property):
                    # listed with its whole text, which its getter carries
                    self.assertEqual(raw.fget.__doc__, raw.__doc__)
                    self.assertRegex(section, rf"(?m)^{name} +``{raw.fget.__name__}`` of every take")
                    self.assertIn(f"{name:<34} {raw.__doc__}", listed)
                else:
                    # a rig edit: its signature, then its first line, every take or none
                    first = raw.__doc__.split("\n", 1)[0]
                    self.assertTrue(first.endswith("() on every take, every take or none"), first)
                    at = next(i for i, line in enumerate(listed) if line.startswith(f"{name}("))
                    self.assertEqual(listed[at + 1], f"    {first}")

    @unittest.skipIf(
        os.environ.get("CGMATH_JEDI_TEST") != "1" or jedi is None,
        "runs only with CGMATH_JEDI_TEST=1 and jedi installed",
    )
    def test_jedi_completes_a_clip_list(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.addCleanup(setattr, jedi.settings, "cache_directory", jedi.settings.cache_directory)
        jedi.settings.cache_directory = cache.name
        root    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        project = jedi.Project(path=root, added_sys_path=[root], smart_sys_path=False)
        source  = "from cgmath.hierarchy import ClipList\nclips = ClipList()\nclips."
        names   = {x.name for x in jedi.Script(source, project=project).complete(3, len("clips."))}
        for name in ("strip_namespace", "fps", "t"):
            self.assertIn(name, names)


class TestClipListLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def check(self, label, clip):
        with self.subTest(label):
            self.assertEqual(fingerprint(clip), LITERALS[label])

    def same_take(self, got, want):
        """a take of a ClipList equal to the ClipData read alone: its frames,
        timebase, name, labels and rest pose"""
        self.assertEqual(fingerprint(got), fingerprint(want))
        self.assertEqual((got.take, got.axis_system, got.unit), (want.take, want.axis_system, want.unit))
        if want.rest_pose is None:
            self.assertIsNone(got.rest_pose)
            return
        self.assertEqual([n.name for n in got.rest_pose], [n.name for n in want.rest_pose])
        np.testing.assert_array_equal(np.asarray(got.rest_pose.world_matrix), np.asarray(want.rest_pose.world_matrix))
        self.assertEqual((got.rest_pose.axis_system, got.rest_pose.unit), (want.rest_pose.axis_system, want.rest_pose.unit))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_load_fbx_every_stack(self):
        from unittest import mock

        path = build_fbx(os.path.join(self.tmp.name, "takes.fbx"))
        with warnings.catch_warnings(record=True) as caught, \
                mock.patch.object(FBX.FbxImporter, "Create", wraps=FBX.FbxImporter.Create) as opened:
            warnings.simplefilter("always")
            clips = ClipList.load_fbx(path)
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(clips.take, ["walk", "idle", "idle_1"])
        self.assertEqual(len([w for w in caught if "takes that would read under one name are told apart" in str(w.message)]), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, clip in enumerate(clips):
                with self.subTest(take=index):
                    self.same_take(clip, ClipData.load_fbx(path, take=index))
            self.assertEqual(len({clip[0].uuid for clip in clips}), 1)  # one rig's uuids

            # every take, no take argument
            for keyword in ("takes", "take"):
                with self.assertRaisesRegex(TypeError, rf"unexpected keyword argument '{keyword}'"):
                    ClipList.load_fbx(path, **{keyword: ["walk"]})

            clash = build_fbx(os.path.join(self.tmp.name, "clash.fbx"), takes=[(n, TAKES[0][1]) for n in ("idle", "idle", "idle_1")])
            self.assertEqual(ClipList.load_fbx(clash).take, ["idle", "idle_2", "idle_1"])

            # a z-up file in metres: each take turned and in cm, as alone
            zup = build_fbx(os.path.join(self.tmp.name, "zup.fbx"), up_axis=2, unit=100.0)
            for index, clip in enumerate(ClipList.load_fbx(zup)):
                self.same_take(clip, ClipData.load_fbx(zup, take=index))
            joints = ClipList.load_fbx(path, node_types=("joint",))
            for index, clip in enumerate(joints):
                self.same_take(clip, ClipData.load_fbx(path, take=index, node_types=("joint",)))

            # a rig holding a group and a locator: the joints alone, each take
            # as read alone with node_types, fewer nodes than the plain read
            mixed  = build_fbx(os.path.join(self.tmp.name, "mixed.fbx"), mixed=True)
            plain  = ClipList.load_fbx(mixed)
            joints = ClipList.load_fbx(mixed, node_types=("joint",))
            self.assertEqual(plain.name,         [["hips", "spine", "grp", "head", "loc"]] * 3)
            self.assertEqual(plain[0].node_type, ["joint", "joint", "transform", "joint", "locator"])
            self.assertEqual(joints.name,        [["hips", "spine", "head"]] * 3)
            for index, clip in enumerate(joints):
                self.same_take(clip, ClipData.load_fbx(mixed, take=index, node_types=("joint",)))

            # every keyword reaches every take, each read as alone with it
            plain = ClipList.load_fbx(zup)
            for keyword in ({"fps": 30.0}, {"scale_factor": 2.0}, {"axis_system": None}):
                with self.subTest(**keyword):
                    clips = ClipList.load_fbx(zup, **keyword)
                    for index, clip in enumerate(clips):
                        self.same_take(clip, ClipData.load_fbx(zup, take=index, **keyword))
                        self.assertNotEqual(fingerprint(clip), fingerprint(plain[index]))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_a_file_with_no_take(self):
        path = build_fbx(os.path.join(self.tmp.name, "still.fbx"), takes=())
        self.assertEqual(len(ClipList.load_fbx(path)), 0)
        self.assertEqual(ClipData.load_fbx(path).frame_count, 1)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_rest_pose_per_take(self):
        """every take holds its own copy of the file's rest pose"""
        path = build_fbx(os.path.join(self.tmp.name, "takes.fbx"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clips = ClipList.load_fbx(path)
            file  = ClipData.load_fbx(path).rest_pose
        rests = [clip.rest_pose for clip in clips]
        self.assertEqual(len({id(rest) for rest in rests}), 3)
        for rest in rests:
            np.testing.assert_array_equal(np.asarray(rest.world_matrix), np.asarray(file.world_matrix))
        rests[0]["hips"].translate = (1.0, 2.0, 3.0)
        np.testing.assert_array_equal(np.asarray(rests[1].world_matrix), np.asarray(file.world_matrix))
        saved = ClipList.load(clips.save(os.path.join(self.tmp.name, "takes.npz")))
        self.assertEqual(len({id(clip.rest_pose) for clip in saved}), 3)

    @unittest.skipIf(pygltflib is None, "pygltflib is not installed")
    def test_load_glb_every_animation(self):
        from unittest import mock

        from cgmath.formats import glb

        path = build_glb(os.path.join(self.tmp.name, "takes.glb"))
        with warnings.catch_warnings(record=True) as caught, \
                mock.patch.object(glb, "load_gltf", wraps=glb.load_gltf) as parse:
            warnings.simplefilter("always")
            clips = ClipList.load_glb(path)
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(clips.take, ["walk", "animation1", "walk_1"])
        self.assertEqual(len([w for w in caught if "takes that would read under one name are told apart" in str(w.message)]), 1)
        rig = HierarchyData.load_glb(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, clip in enumerate(clips):
                with self.subTest(take=index):
                    self.same_take(clip, ClipData.load_glb(path, animation=index))
                    # the file's node pose as the rest pose
                    np.testing.assert_array_equal(np.asarray(clip.rest_pose.world_matrix), np.asarray(rig.world_matrix))
        # each take its own copy, edited alone
        self.assertEqual(len({id(clip.rest_pose) for clip in clips}), 3)
        clips[0].rest_pose["root"].translate = (1.0, 2.0, 3.0)
        for clip in clips[1:]:
            np.testing.assert_array_equal(np.asarray(clip.rest_pose.world_matrix), np.asarray(rig.world_matrix))
        still = build_glb(os.path.join(self.tmp.name, "still.glb"), animations=())
        self.assertEqual(len(ClipList.load_glb(still)), 0)

    def test_blank_names_in_the_list(self):
        """the list names a blank take as ClipData's
        loaders name it, Take 001 ... by position in an fbx, animation<i> in
        a glb, before two takes of one name are told apart"""
        if pygltflib is not None:
            animations = [(name,) + ANIMATIONS[0][1:] for name in ("", "  ", "animation1", "walk")]
            path       = build_glb(os.path.join(self.tmp.name, "blank.glb"), animations=animations)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clips = ClipList.load_glb(path)
            self.assertEqual(clips.take, ["animation0", "animation1", "animation1_1", "walk"])
            self.assertEqual(len([w for w in caught if "take 2 'animation1' as 'animation1_1'" in str(w.message)]), 1)
            # the warning names a take as the file holds it
            animations = [(name,) + ANIMATIONS[0][1:] for name in ("animation1", "  ")]
            path       = build_glb(os.path.join(self.tmp.name, "spaces.glb"), animations=animations)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(ClipList.load_glb(path).take, ["animation1", "animation1_1"])
            self.assertEqual(
                [str(w.message) for w in caught],
                ["takes that would read under one name are told apart: take 1 '  ' as 'animation1_1'"],
            )
        if FBX is not None:
            stacks = [(name, TAKES[0][1]) for name in ("idle", "", "Take 002", " ")]
            path   = build_fbx(os.path.join(self.tmp.name, "blank.fbx"), takes=stacks)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clips = ClipList.load_fbx(path)
            self.assertEqual(clips.take, ["idle", "Take 002", "Take 002_1", "Take 004"])
            self.assertEqual(len([w for w in caught if "take 2 'Take 002' as 'Take 002_1'" in str(w.message)]), 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for index, clip in enumerate(clips):
                    self.same_take(clip, ClipData.load_fbx(path, take=index))

    @unittest.skipIf(pygltflib is None, "pygltflib is not installed")
    def test_load_glb_takes_no_take_argument(self):
        """ClipList.load_glb reads every
        animation, as ClipList.load_fbx every take; one alone is
        ClipData.load_glb's, by animation=i. takes= and take= are refused
        before the file is parsed"""
        from unittest import mock

        from cgmath.formats import glb

        path = build_glb(os.path.join(self.tmp.name, "takes.glb"))
        with mock.patch.object(glb, "load_gltf", wraps=glb.load_gltf) as parse:
            for keyword, value in (("takes", ["walk"]), ("takes", ()), ("take", 0)):
                with self.subTest(keyword=keyword, value=value):
                    with self.assertRaisesRegex(TypeError, rf"unexpected keyword argument '{keyword}'"):
                        ClipList.load_glb(path, **{keyword: value})
            for keyword in ("takes", "take"):
                with self.subTest(route=keyword):
                    with self.assertRaisesRegex(TypeError, rf"ClipList\.load_glb: .*'{keyword}'"):
                        ClipList.load(path, **{keyword: ["walk"]})
        self.assertEqual(parse.call_count, 0)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_fbx_grid_counts_whole_spans(self):
        """3.0.0's frame rule survives: 62 frames for frames 60-121 at 24 fps,
        14 for frames 1-14"""
        keys  = {"hips": {"t": [(60, (0, 90, 0)), (121, (5, 90, 0))]}}
        path  = build_fbx(os.path.join(self.tmp.name, "spans.fbx"), takes=[("long", keys), ("short", {"hips": {"t": [(1, (0, 90, 0)), (14, (1, 90, 0))]}})])
        clips = ClipList.load_fbx(path)
        self.assertEqual(clips.frame_count, [62, 14])
        self.assertEqual(clips.start_frame, [60, 1])

    def test_routes(self):
        """ClipList.load routes an fbx or glb by extension, its keywords
        checked before the file is read"""
        if pygltflib is not None:
            path = build_glb(os.path.join(self.tmp.name, "takes.glb"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(ClipList.load(path).take, ["walk", "animation1", "walk_1"])
            with self.assertRaisesRegex(TypeError, r"ClipList\.load_glb: .*'takes'"):
                ClipList.load(path, takes=["walk"])
        if FBX is not None:
            path = build_fbx(os.path.join(self.tmp.name, "takes.fbx"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(ClipList.load(path).take, ["walk", "idle", "idle_1"])
            # every take, no take argument, refused before the file is opened
            from unittest import mock

            with mock.patch.object(FBX.FbxImporter, "Create", wraps=FBX.FbxImporter.Create) as opened:
                with self.assertRaisesRegex(TypeError, r"ClipList\.load_fbx: .*'takes'"):
                    ClipList.load(path, takes=["idle"])
            self.assertEqual(opened.call_count, 0)

    @unittest.skipIf(pygltflib is None, "pygltflib is not installed")
    def test_load_glb_parses_once(self):
        """ClipData.load_glb reads the rig and the take from one parse"""
        from unittest import mock

        from cgmath.formats import glb

        path = build_glb(os.path.join(self.tmp.name, "takes.glb"))
        with mock.patch.object(glb, "load_gltf", wraps=glb.load_gltf) as parse:
            ClipData.load_glb(path, animation=1)
        self.assertEqual(parse.call_count, 1)

    @unittest.skipIf(pygltflib is None, "pygltflib is not installed")
    def test_load_glb_splits_a_node_matrix(self):
        """the rig ClipData.load_glb builds from its one parse splits a node
        given as a matrix as HierarchyData.load_glb splits it (3.2.0): its
        turn on joint_orient, its scale and translate, on every frame"""
        path  = build_glb(os.path.join(self.tmp.name, "matrix.glb"), animations=ANIMATIONS[:1])
        gltf  = GLTF2().load(path)
        turn  = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        local = np.eye(4)
        local[:3, :3]             = np.diag([1.0, 2.0, 0.5]) @ turn
        local[3, :3]              = (0.5, 1.0, -0.25)
        gltf.nodes[1].translation = None
        gltf.nodes[1].matrix      = [float(value) for value in local.ravel()]
        gltf.save(path)

        for factor in (1.0, 100.0):
            with self.subTest(scale_factor=factor):
                rig  = HierarchyData.load_glb(path, scale_factor=factor)
                clip = ClipData.load_glb(path, scale_factor=factor)
                # the matrix's own: its turn 90 about z, its scale, its place
                np.testing.assert_allclose(rig["mid"].joint_orient, (0.0, 0.0, 90.0), rtol=0, atol=1e-12)
                np.testing.assert_allclose(rig["mid"].scale, (1.0, 2.0, 0.5), rtol=0, atol=1e-12)
                np.testing.assert_allclose(rig["mid"].translate, np.array([0.5, 1.0, -0.25]) * factor, rtol=0, atol=1e-12)
                for channel in ("translate", "rotate", "joint_orient", "scale"):
                    np.testing.assert_array_equal(
                        np.asarray(getattr(clip.frames, channel))[:, 1],
                        np.broadcast_to(np.asarray(getattr(rig["mid"], channel)), (clip.frame_count, 3)),
                        err_msg=channel,
                    )

    def test_clip_loaders_bit_identical(self):
        """the clip loaders read what they read before their bodies moved:
        Maya's Walk.fbx and an SDK-built file of three takes (mayapy), and a
        glb of three animations, each take"""
        if pygltflib is not None:
            path = build_glb(os.path.join(self.tmp.name, "takes.glb"))
            for index in range(len(ANIMATIONS)):
                self.check(f"glb {index}", ClipData.load_glb(path, animation=index))
        if FBX is None:
            return
        path = build_fbx(os.path.join(self.tmp.name, "takes.fbx"))
        self.check("fbx default", ClipData.load_fbx(path))
        for index in range(len(TAKES)):
            self.check(f"fbx {index}", ClipData.load_fbx(path, take=index))
        path = build_fbx(os.path.join(self.tmp.name, "z_up.fbx"), up_axis=2)
        self.check("fbx z-up", ClipData.load_fbx(path))
        if os.path.exists(WALK):
            self.check("walk", ClipData.load_fbx(WALK))


# TAKES without the keys on the node with no attribute, which the read folds
# into the hips' joint orient every frame, a joint orient fbx can't key
PLAIN_TAKES = tuple((name, {k: v for k, v in keys.items() if k != "offset"}) for name, keys in TAKES)


def at_float32(values) -> np.ndarray:
    """values as fbx keys hold them"""
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def orienting_takes(orders=(0, 0, 0)):
    """two takes of one rig, a joint root, a group, a joint, a joint and a
    locator, whose joint orients and rotate axes move within a take and
    differ between the takes, every node turning: take "a", 6 frames, the
    mid joint's joint orient moving and its rotate z crossing 180, the tip
    joint's rotate axis moving from (15, 0, 25), the group's rotate axis
    held; take "b", 4 frames from 10, the mid joint's joint orient and the
    tip's rotate axis held off take a's first frame, the group's moving.
    ``orders`` the rotate orders of the mid joint, the tip joint and the
    group"""
    rig = HierarchyData()
    for name, kind, parent, place in (
        ("root", "joint", None, (0.0, 1.0, 0.0)),
        ("grp", "transform", "root", (1.0, 0.0, 0.0)),
        ("mid", "joint", "grp", (0.0, 2.0, 0.0)),
        ("tip", "joint", "mid", (0.0, 2.0, 0.5)),
        ("loc", "locator", "tip", (0.0, 1.0, 1.0)),
    ):
        rig.append(TransformData(name, node_type=kind, translate=place))
        if parent is not None:
            rig[name].parent_uuid = rig[parent].uuid
    for name, order in zip(("mid", "tip", "grp"), orders):
        rig[name].rotate_order = order
    takes = []
    for take, frames, start in (("a", 6, 0), ("b", 4, 10)):
        clip  = ClipData(rig, frames=frames, start_frame=start, take=take)
        f     = np.arange(frames, dtype=float)[:, None]
        turns = np.array(clip.frames.rotate)
        turns += np.array([3.0, -4.0, 5.0]) * f[:, :, None] + np.array([10.0, 20.0, -30.0])
        if take == "a":
            turns[:, 2, 2] += 180.0 + 9.0 * f[:, 0]  # 150 to 220: across 180
        clip.frames.rotate = turns
        orient             = np.array(clip.frames.joint_orient)
        axis               = np.array(clip.frames.rotate_axis)
        if take == "a":
            orient[:, 2] = np.hstack([10.0 + 3.0 * f, -20.0 + 7.0 * f, 5.0 * f])
            axis[:, 3]   = np.hstack([15.0 + 2.0 * f, 0.0 * f, 25.0 - 3.0 * f])
            axis[:, 1]   = (15.0, 0.0, 25.0)
        else:
            orient[:, 2] = (30.0, 10.0, -5.0)
            axis[:, 1]   = np.hstack([-6.0 * f, 2.0 * f, 0.0 * f])
        clip.frames.joint_orient = orient
        clip.frames.rotate_axis  = axis
        takes.append(clip)
    return ClipList(takes)


class TestClipListSaveFbx(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def assert_back(self, clips, back, exact_statics=False):
        """``back``, read from what ``clips`` wrote: the same takes, every
        channel the float32 of the one written"""
        self.assertEqual(back.take, [take.strip() or f"Take {i + 1:03d}" for i, take in enumerate(clips.take)])
        self.assertEqual((back.frame_count, back.start_frame, back.fps), (clips.frame_count, clips.start_frame, clips.fps))
        for a, b in zip(clips, back):
            self.assertEqual(list(a.name), list(b.name))
            for channel in ("translate", "rotate", "scale"):
                np.testing.assert_allclose(
                    np.asarray(getattr(b.frames, channel)), at_float32(getattr(a.frames, channel)), rtol=0, atol=1e-9
                )
            if exact_statics:
                for channel in ("joint_orient", "rotate_axis"):
                    np.testing.assert_array_equal(np.asarray(getattr(b.frames, channel)), np.asarray(getattr(a.frames, channel)))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_round_trip(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source = build_fbx(self.path("takes.fbx"), takes=PLAIN_TAKES)
            clips  = ClipList.load_fbx(source)
            with self.subTest("read from an fbx"):
                clips.save_fbx(self.path("back.fbx"))
                back = ClipList.load_fbx(self.path("back.fbx"))
                self.assert_back(clips, back)
                self.assertEqual((back.axis_system, back.unit), (clips.axis_system, clips.unit))

            with self.subTest("hand-built, random channels"):
                rng  = np.random.default_rng(7)
                made = ClipList([hand_built("idle", 6), hand_built("walk", 4)])
                for clip in made:
                    clip.frames.rotate    = rng.uniform(-170, 170, np.asarray(clip.frames.rotate).shape)
                    clip.frames.translate = rng.uniform(-50, 50, np.asarray(clip.frames.translate).shape)
                # one joint orient, static, on every take: fbx holds one
                made[0][1].joint_orient  = (10.0, 0.0, -5.0)
                made.frames.joint_orient = PerTake([np.broadcast_to(np.asarray(made[0].frames.joint_orient)[0], np.asarray(c.frames.joint_orient).shape).copy() for c in made])
                made.save_fbx(self.path("made.fbx"))
                self.assert_back(made, ClipList.load_fbx(self.path("made.fbx"), axis_system=None), exact_statics=True)

            with self.subTest("frames 60 to 121"):
                keys = {"hips": {"t": [(60, (0, 90, 0)), (121, (5, 90, 0))]}}
                long = ClipList.load_fbx(build_fbx(self.path("long.fbx"), takes=[("long", keys)]))
                long.save_fbx(self.path("long_back.fbx"))
                self.assertEqual(ClipList.load_fbx(self.path("long_back.fbx")).frame_count, [62])

            with self.subTest("ascii"):
                # an ascii file holds keys as the text its writer prints, about
                # 7 significant digits: 4.6e-7 of the value here
                clips.save_fbx(self.path("ascii.fbx"), as_ascii=True)
                back = ClipList.load_fbx(self.path("ascii.fbx"))
                self.assertEqual((back.take, back.frame_count), (clips.take, clips.frame_count))
                for a, b in zip(clips, back):
                    for channel in ("translate", "rotate", "scale"):
                        np.testing.assert_allclose(
                            np.asarray(getattr(b.frames, channel)), np.asarray(getattr(a.frames, channel)), rtol=1e-6, atol=1e-9
                        )

            with self.subTest("zero_root"):
                clips.save_fbx(self.path("zero.fbx"), zero_root=True)
                for take in ClipList.load_fbx(self.path("zero.fbx"), axis_system=None):
                    np.testing.assert_array_equal(np.asarray(take.frames.matrix)[:, 0], np.broadcast_to(np.eye(4), (take.frame_count, 4, 4)))

            with self.subTest("a take named ' idle '"):
                spaced      = clips[1:2].copy()
                spaced.take = " idle "
                spaced.save_fbx(self.path("spaced.fbx"))
                self.assertEqual(ClipList.load_fbx(self.path("spaced.fbx")).take, ["idle"])

            with self.subTest("a z-up file read in its own axes"):
                zup = ClipList.load_fbx(build_fbx(self.path("zup.fbx"), takes=PLAIN_TAKES, up_axis=2), axis_system=None)
                zup.save_fbx(self.path("zup_back.fbx"))
                back = ClipList.load_fbx(self.path("zup_back.fbx"), axis_system=None)
                self.assert_back(zup, back)
                self.assertEqual(back.axis_system, zup.axis_system)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_a_name_with_a_space(self):
        """a take is named as clip.take says, "Walk Forward" kept as it
        is; Take 001 ... only for a blank name"""
        clips = ClipList([hand_built("Walk Forward", 4), hand_built("Run", 3), hand_built(" ", 2)])
        clips.save_fbx(self.path("named.fbx"))
        back = ClipList.load_fbx(self.path("named.fbx"))
        self.assertEqual(back.take, ["Walk Forward", "Run", "Take 003"])
        self.assertEqual(ClipData.load_fbx(self.path("named.fbx"), take=0).take, "Walk Forward")
        self.assertEqual(back.frame_count, [4, 3, 2])

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_shear_on_any_frame_is_named_once(self):
        """fbx has no channel for shear: one warning names the nodes holding
        one on any frame of any take, parents first, as a rig's save names
        those of its frame; none where no take holds any"""
        from cgmath._tests.test_fbx_export import _no_fbx_warnings, scaled_rig

        clips = ClipList([ClipData(scaled_rig(), frames=3, take="a"), ClipData(scaled_rig(), frames=2, take="b")])
        with _no_fbx_warnings(self):
            clips.save_fbx(self.path("plain.fbx"))

        names = list(clips[0].name)
        for clip, frame, node in ((clips[1], 1, "sc_lcf"), (clips[0], 2, "sc_loc"), (clips[1], 0, "sc_kid")):
            shear                              = np.array(clip.frames.shear)
            shear[frame, names.index(node), 0] = 0.5
            clip.frames.shear                  = shear
        with self.assertLogs("cgmath.formats.fbx", "WARNING") as logged:
            clips.save_fbx(self.path("sheared.fbx"))
        said = [record.getMessage() for record in logged.records if "no shear" in record.getMessage()]
        self.assertEqual(len(said), 1, said)
        # parents first, as a rig's save names them: sc_loc is under sc_kid
        self.assertTrue(said[0].endswith(": sc_kid, sc_lcf, sc_loc"), said[0])
        rig = scaled_rig()
        for node in ("sc_lcf", "sc_loc", "sc_kid"):
            rig[node].shear = (0.5, 0.0, 0.0)
        with self.assertLogs("cgmath.formats.fbx", "WARNING") as logged:
            rig.save_fbx(self.path("rig.fbx"))
        self.assertEqual([record.getMessage() for record in logged.records if "no shear" in record.getMessage()], said)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_a_moving_joint_orient_is_keyed_in_rotate(self):
        """a joint orient or rotate axis that moves within a
        take, or differs between takes, saves: the file holds the first
        take's first frame's as a joint's pre and post rotation, a node that
        is no joint none, and every frame keys its difference in its rotate,
        so every node's world on every frame of every take reads back within
        float32 of the clip's"""
        both  = orienting_takes()
        still = orienting_takes()
        for clip in still:
            for channel in ("joint_orient", "rotate_axis"):
                block = np.array(getattr(clip.frames, channel))
                setattr(clip.frames, channel, np.broadcast_to(block[:1], block.shape).copy())
        bound       = orienting_takes()
        bound.frame = PerTake([3, 2])  # the rig written from take a's frame 3
        cases = (
            ("within a take", ClipList([both[0]]), [(0, 6)]),
            ("between takes", still, [(0, 6), (10, 4)]),
            ("both", both, [(0, 6), (10, 4)]),
            ("bound frames 3 and 2", bound, [(0, 6), (10, 4)]),
        )
        for label, clips, spans in cases:
            with self.subTest(label):
                clips.save_fbx(self.path(f"{label}.fbx"))
                back = ClipList.load_fbx(self.path(f"{label}.fbx"), axis_system=None)
                self.assertEqual(list(zip(back.start_frame, back.frame_count)), spans)
                for clip, read in zip(clips, back):
                    want = np.asarray(clip.frames.world_matrix)
                    got  = np.asarray(read.frames.world_matrix)
                    self.assertLessEqual(np.abs(got[..., :3, :3] - want[..., :3, :3]).max(), 2e-7, clip.take)
                    self.assertLessEqual(np.abs(got[..., 3, :3] - want[..., 3, :3]).max(), 4e-7, clip.take)
                    # the file's one pre and post rotation per joint: take a's first frame
                    orient = np.broadcast_to((10.0, -20.0, 0.0), (read.frame_count, 3))
                    axis   = np.broadcast_to((15.0, 0.0, 25.0), (read.frame_count, 3))
                    np.testing.assert_allclose(np.asarray(read.frames.joint_orient)[:, 2], orient, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(np.asarray(read.frames.rotate_axis)[:, 3], axis, rtol=0, atol=1e-12)
                    # the keys continuous: none a turn away from the frame before
                    steps = np.abs(np.diff(np.asarray(read.frames.rotate), axis=0))
                    self.assertLess(steps.max(), 180.0, clip.take)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_maya_reads_a_moving_joint_orient(self):
        """Maya 2025's FBXImport of each take (1-based -t) of a moving joint
        orient and rotate axis saved: every node's world on every frame within
        float32 of the clip's. Needs mayapy with the fbxmaya plug-in"""
        clips = orienting_takes()
        clips.save_fbx(self.path("orient.fbx"))
        for index, clip in enumerate(clips):
            with self.subTest(take=clip.take):
                frames = range(clip.start_frame, clip.start_frame + clip.frame_count)
                worlds = self.maya_frames(self.path("orient.fbx"), index + 1, frames)
                want   = np.asarray(clip.frames.world_matrix)
                for column, node in enumerate(clip):
                    got = worlds[node.name]
                    self.assertLessEqual(np.abs(got[:, :3, :3] - want[:, column, :3, :3]).max(), 2e-7, node.name)
                    self.assertLessEqual(np.abs(got[:, 3, :3] - want[:, column, 3, :3]).max(), 4e-7, node.name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_refusals(self):
        """each check raises before the sdk is reached, naming what fails"""
        def cases():
            yield "no take", ClipList(), "no take to save"
            empty = ClipList([hand_built("a"), ClipData(hand_built("b"), take="b")])
            yield "a take with no frame", empty, "take 1 ('b') holds no frame"
            renamed            = ClipList([hand_built("a"), hand_built("b")])
            renamed[1][1].name = "tap"
            yield "a renamed node", renamed, "node 1 is"
            moved                   = ClipList([hand_built("a"), hand_built("b")])
            moved[1][1].parent_uuid = None
            yield "a node reparented", moved, "node 1 is"
            ordered                    = ClipList([hand_built("a"), hand_built("b")])
            ordered[1][0].rotate_order = 3
            yield "another rotate order", ordered, "node 0 is"
            rates        = ClipList([hand_built("a"), hand_built("b")])
            rates[1].fps = 30.0
            yield "two frame rates", rates, "[24.0, 30.0] fps"
            odd     = ClipList([hand_built("a")])
            odd.fps = 17.5
            yield "17.5 fps", odd, (
                "no fbx time mode carries 17.5 fps; the rates a take may have are [23.976024, 24.0, 25.0, "
                "29.97003, 30.0, 48.0, 50.0, 59.94006, 60.0, 72.0, 96.0, 100.0, 119.88012, 120.0, 1000.0]"
            )
            twice = ClipList([hand_built("idle"), hand_built("idle")])
            yield "two takes named idle", twice, "['idle']"
            spaced = ClipList([hand_built("idle"), hand_built(" idle ")])
            yield "idle and ' idle '", spaced, "['idle']"
            labels         = ClipList([hand_built("a"), hand_built("b")])
            labels[1].unit = 100.0
            yield "takes in two units", labels, "in other axes or units"

        for label, clips, message in cases():
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    clips.save_fbx(self.path(f"{label}.fbx"))
                self.assertIn(message, str(caught.exception))
                self.assertFalse(os.path.exists(self.path(f"{label}.fbx")))

    def test_with_no_sdk_a_save_raises_import_error(self):
        """with no Autodesk SDK ClipList.save_fbx, ClipData.save_fbx of
        two frames or more and ClipList.save of an fbx raise ImportError, as
        HierarchyData.save_fbx and the readers do, before the takes are
        checked; nothing is written. The SDK's absence patched in where the
        SDK is installed"""
        from unittest import mock

        from cgmath.hierarchy import hierarchy

        clips = ClipList([hand_built("idle", 3)])
        with mock.patch.object(hierarchy, "FBX", None), mock.patch.object(hierarchy, "FbxExporter", None):
            for label, save in (
                ("ClipList.save_fbx", clips.save_fbx),
                ("ClipData.save_fbx of 3 frames", clips[0].save_fbx),
                ("ClipList.save", clips.save),
                ("a list with no take", ClipList().save_fbx),
            ):
                with self.subTest(label):
                    path = self.path(f"{label}.fbx")
                    with self.assertRaisesRegex(ImportError, "^fbx module not found$"):
                        save(path)
                    self.assertFalse(os.path.exists(path))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_a_path_from_the_home_folder(self):
        """~ is the home folder, as in every other save and read:
        ClipList.save_fbx and ClipData.save_fbx of two frames or more write
        there, and load_fbx of the same path reads the file back"""
        from unittest import mock

        home, cwd = self.path("home"), self.path("cwd")
        os.makedirs(home)
        os.makedirs(cwd)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(cwd)
        clips = ClipList([hand_built("idle", 3)])
        with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
            clips.save_fbx("~/list.fbx")
            clips[0].save_fbx("~/clip.fbx")
            for name in ("list.fbx", "clip.fbx"):
                with self.subTest(name):
                    self.assertTrue(os.path.isfile(os.path.join(home, name)))
                    self.assertEqual(ClipList.load_fbx(f"~/{name}").take, ["idle"])
        self.assertEqual(os.listdir(cwd), [])  # no folder named ~ where the save ran

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_frame_rate_table(self):
        from cgmath.hierarchy.hierarchy import _FBX_FRAME_RATES

        held = {name for _, name in _FBX_FRAME_RATES}
        for rate, name in _FBX_FRAME_RATES:
            with self.subTest(name):
                got = FBX.FbxTime.GetFrameRate(getattr(FBX.FbxTime.EMode, name))
                self.assertAlmostEqual(got / rate, 1.0, delta=1e-12)
        for name in dir(FBX.FbxTime.EMode):
            if not name.startswith("e") or name in ("eCustom", "eModesCount", "eDefaultMode"):
                continue
            rate = FBX.FbxTime.GetFrameRate(getattr(FBX.FbxTime.EMode, name))
            if rate > 0 and name not in held:
                self.assertTrue(any(abs(rate - r) <= 1e-9 * r for r, _ in _FBX_FRAME_RATES), name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_curves_are_linear_keys(self):
        from cgmath.formats._fbx_io import open_fbx

        clips = ClipList([hand_built("idle", 6), hand_built("walk", 4)])
        for name, zero_root in (("keys.fbx", False), ("zero.fbx", True)):
            clips.save_fbx(self.path(name), zero_root=zero_root)
            with open_fbx(self.path(name), FBX) as scene:
                criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimStack.ClassId)
                stacks   = [scene.GetSrcObject(criteria, i) for i in range(scene.GetSrcObjectCount(criteria))]
                self.assertEqual([x.GetName() for x in stacks], ["idle", "walk"])
                nodes        = {scene.GetRootNode().GetChild(0).GetName(): scene.GetRootNode().GetChild(0)}
                nodes["tip"] = nodes["root"].GetChild(0)
                for stack, frames in zip(stacks, (6, 4)):
                    layer = stack.GetMember(FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId), 0)
                    for node_name, node in nodes.items():
                        for prop in (node.LclTranslation, node.LclRotation, node.LclScaling):
                            for component in "XYZ":
                                curve = prop.GetCurve(layer, component, False)
                                if zero_root and node_name == "root":
                                    self.assertIsNone(curve)
                                    continue
                                self.assertEqual(curve.KeyGetCount(), frames)
                                for k in range(frames):
                                    self.assertAlmostEqual(curve.KeyGetTime(k).GetSecondDouble(), k / 24.0, places=9)
                                    self.assertEqual(
                                        curve.KeyGetInterpolation(k),
                                        FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear,
                                    )

    def maya_frames(self, path, take, frames):
        """{node: (F, 4, 4) world matrices} of Maya 2025's FBXImport of take
        ``take`` (1-based -t), on each of ``frames``; skips without Maya"""
        from cgmath._tests.test_fbx_labels import _maya

        maya = _maya()
        if maya is None:
            self.skipTest("needs Maya and its fbxmaya plug-in")
        cmds, mel = maya
        self.addCleanup(cmds.file, new=True, force=True)
        cmds.file(new=True, force=True)
        mel.eval("FBXResetImport")
        mel.eval(f'FBXImport -f "{path.replace(os.sep, "/")}" -t {take}')
        nodes  = cmds.ls(type="transform", long=True)
        worlds = {node.rsplit("|", 1)[-1]: [] for node in nodes}
        for frame in frames:
            cmds.currentTime(frame)
            for node in nodes:
                worlds[node.rsplit("|", 1)[-1]].append(cmds.xform(node, q=True, ws=True, m=True))
        return {name: np.array(rows).reshape(-1, 4, 4) for name, rows in worlds.items()}

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_keyed_scales_are_written_as_they_are(self):
        """QE (a): a keyed scale of 0 is written as 0, as Maya's own export
        writes it; only a static scale under 1e-12 is raised to 1e-12; Maya's
        import of it is finite on every frame"""
        clips                 = ClipList([hand_built("idle", 4)])
        scale                 = np.array(clips[0].frames.scale)
        scale[2, 0]           = 0.0
        clips[0].frames.scale = scale
        clips.save_fbx(self.path("scale.fbx"))
        back = ClipList.load_fbx(self.path("scale.fbx"), axis_system=None)
        self.assertEqual(float(np.asarray(back[0].frames.scale)[2, 0, 0]), 0.0)

        # Maya's evaluation holds the key of 0 as 1e-12: 9.85e-13, turned 10
        worlds = self.maya_frames(self.path("scale.fbx"), 1, range(4))
        self.assertLessEqual(abs(worlds["root"][2, 0, 0]), 1e-12)
        for name in ("root", "tip"):
            self.assertTrue(np.isfinite(worlds[name]).all(), name)

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_maya_reads_every_take(self):
        """Maya 2025's FBXImport of each take (1-based -t) holds each frame's
        channels within 1e-6 of the take's float32 values. Needs mayapy with
        the fbxmaya plug-in"""
        try:
            import maya.cmds as cmds
            import maya.mel as mel

            cmds.loadPlugin("fbxmaya", quiet=True)
        except Exception:  # not running in Maya, or no fbx plug-in
            self.skipTest("needs Maya and its fbxmaya plug-in")

        clips                     = ClipList([hand_built("idle", 6), hand_built("walk", 4)])
        clips[1].frames.translate = np.asarray(clips[1].frames.translate) + [0.0, 0.0, 2.5]
        clips.save_fbx(self.path("maya.fbx"))
        self.addCleanup(cmds.file, new=True, force=True)
        for index, clip in enumerate(clips):
            with self.subTest(take=clip.take):
                cmds.file(new=True, force=True)
                mel.eval("FBXResetImport")
                mel.eval(f'FBXImport -f "{self.path("maya.fbx").replace(os.sep, "/")}" -t {index + 1}')
                for column, node in enumerate(clip):
                    for channel, attr in (("rotate", "rotate"), ("translate", "translate")):
                        want = at_float32(getattr(clip.frames, channel))[:, column]
                        got  = np.array([cmds.getAttr(f"{node.name}.{attr}", time=f)[0] for f in range(clip.frame_count)])
                        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)

        # groups, locators, a joint and a space transform under a group
        # scaled (1, 3, 0.5) and turned, moving, where cgmath has them on every
        # frame (each node written with its inherit type)
        from cgmath._tests.test_fbx_export import gaps, scaled_rig

        moving = ClipList([ClipData(scaled_rig(), frames=4, take="a"), ClipData(scaled_rig(), frames=3, take="b")])
        for step, clip in zip((1.0, -2.0), moving):
            turns  = np.array(clip.frames.rotate)
            places = np.array(clip.frames.translate)
            turns[:, 1]     += np.arange(clip.frame_count)[:, None] * 10.0 * step
            places[:, 0, 0] += np.arange(clip.frame_count) * step
            clip.frames.rotate, clip.frames.translate = turns, places
        moving.save_fbx(self.path("scaled.fbx"))
        for index, clip in enumerate(moving):
            with self.subTest(take=clip.take, rig="scaled"):
                worlds = self.maya_frames(self.path("scaled.fbx"), index + 1, range(clip.frame_count))
                for column, node in enumerate(clip):
                    want = np.asarray(clip.frames.world_matrix)[:, column]
                    for frame in range(clip.frame_count):
                        axes, position = gaps(worlds[node.name][frame], want[frame])
                        self.assertLessEqual(axes, 1e-9, (node.name, frame))
                        # the largest measured on this file
                        self.assertLessEqual(position, 8.881784197001252e-16, (node.name, frame))

    @unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
    def test_routes(self):
        clips = ClipList([hand_built("idle", 6)])
        path  = clips.save(self.path("routed.fbx"))
        self.assertEqual(ClipList.load(path).take, ["idle"])
        with self.assertRaisesRegex(TypeError, "zeroroot"):
            clips.save(self.path("wrong.fbx"), zeroroot=True)
        self.assertFalse(os.path.exists(self.path("wrong.fbx")))


# the two files of a Maya import-dialog test (2026-11-01):
# two joints, the root's
# translate x keyed per take, (name, keys or None, span), at 24 fps
DIALOG_FILES = {
    "three_takes_middle_active": (
        (("first", [(0, 100.0), (10, 110.0)], (0, 10)),
         ("middle", [(20, 200.0), (30, 220.0)], (20, 30)),
         ("last", [(40, 300.0), (50, 330.0)], (40, 50))),
        "middle",
    ),
    "empty_first_take": (
        (("Take 001", None, (0, 48)),
         ("Walk Forward", [(0, 100.0), (10, 110.0)], (0, 10)),
         ("Run", [(0, 200.0), (20, 220.0)], (0, 20))),
        "Take 001",
    ),
}


def build_dialog_fbx(path, takes, active):
    """one of ``DIALOG_FILES``, written with the sdk"""
    def at(frame):
        time = FBX.FbxTime()
        time.SetSecondDouble(frame / 24.0)
        return time

    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene = FBX.FbxScene.Create(manager, "scene")
        scene.GetGlobalSettings().SetTimeMode(FBX.FbxTime.EMode.eFrames24)
        parent, nodes = scene.GetRootNode(), []
        for name, place in (("probe_root", (-7.0, 1.0, 0.0)), ("probe_tip", (10.0, 0.0, 0.0))):
            attr = FBX.FbxSkeleton.Create(scene, name + "Attr")
            attr.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
            node = FBX.FbxNode.Create(scene, name)
            node.SetNodeAttribute(attr)
            node.LclTranslation.Set(FBX.FbxDouble3(*place))
            parent.AddChild(node)
            nodes.append(node)
            parent = node
        stacks = {}
        for name, keys, span in takes:
            stack = FBX.FbxAnimStack.Create(scene, name)
            layer = FBX.FbxAnimLayer.Create(scene, "Base Layer")
            stack.AddMember(layer)
            stacks[name] = stack
            nodes[0].LclTranslation.GetCurveNode(layer, True)
            if keys:
                curve = nodes[0].LclTranslation.GetCurve(layer, "X", True)
                curve.KeyModifyBegin()
                for frame, value in keys:
                    index = curve.KeyAdd(at(frame))[0]
                    curve.KeySetValue(index, value)
                    curve.KeySetInterpolation(index, FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                curve.KeyModifyEnd()
            stack.SetLocalTimeSpan(FBX.FbxTimeSpan(at(span[0]), at(span[1])))
        scene.SetCurrentAnimationStack(stacks[active])
        scene.ActiveAnimStackName.Set(FBX.FbxString(active))
        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestLoadFbxTakeByName(unittest.TestCase):
    """ClipData.load_fbx(take="Walk Forward") loads the take of that
    name, as ClipList.load_fbx names it; the default and None keep 3.2.0's
    meaning, the first take holding a key, and an int its position"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files = {
            name: build_dialog_fbx(os.path.join(self.tmp.name, name + ".fbx"), *spec)
            for name, spec in DIALOG_FILES.items()
        }

    def keys(self, clip):
        """(start, end, translate x of the root on the first and last frame)"""
        x = np.asarray(clip.frames.translate)[:, 0, 0]
        return clip.start_frame, clip.end_frame, round(float(x[0]), 9), round(float(x[-1]), 9)

    def test_each_name_loads_its_take(self):
        three, empty = self.files["three_takes_middle_active"], self.files["empty_first_take"]
        for path, name, want in (
            (three, "first", (0, 10, 100.0, 110.0)),
            (three, "middle", (20, 30, 200.0, 220.0)),
            (three, "last", (40, 50, 300.0, 330.0)),
            (empty, "Walk Forward", (0, 10, 100.0, 110.0)),
            (empty, "Run", (0, 20, 200.0, 220.0)),
            (empty, "Take 001", (0, 48, -7.0, -7.0)),
            (empty, " Run ", (0, 20, 200.0, 220.0)),
        ):
            with self.subTest(name):
                clip = ClipData.load_fbx(path, take=name)
                self.assertEqual(self.keys(clip), want)
                self.assertEqual(clip.take, name.strip())
        # and through the route
        self.assertEqual(ClipData.load(empty, take="Run").take, "Run")

    def test_a_name_as_the_list_names_it(self):
        """two stacks of one name are idle and idle_1, as ClipList.load_fbx
        names them, the second read with the list's one warning"""
        path = build_fbx(os.path.join(self.tmp.name, "idle.fbx"), takes=[(n, TAKES[0][1]) for n in ("idle", "idle")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            second = ClipData.load_fbx(path, take="idle_1")
        self.assertEqual(second.take, "idle_1")
        self.assertEqual(len([w for w in caught if "takes that would read under one name are told apart" in str(w.message)]), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(fingerprint(second), fingerprint(ClipData.load_fbx(path, take=1)))
            self.assertEqual(ClipData.load_fbx(path, take="idle").take, "idle")

    def test_a_name_exactly_first(self):
        """the take the list names so first, else the
        first whose name matches outer spaces aside: of stacks ' idle ' and
        'idle', take='idle' reads 'idle' and take=' idle ' reads ' idle ';
        a blank second take reads as take="Take 002",
        ahead of a stack named Take 002 itself"""
        stacks = [(" idle ", TAKES[0][1]), ("idle", TAKES[1][1]), (" ", TAKES[2][1]), ("Take 002", TAKES[0][1])]
        path   = build_fbx(os.path.join(self.tmp.name, "names.fbx"), takes=stacks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clips = ClipList.load_fbx(path)
            self.assertEqual(clips.take, [" idle ", "idle", "Take 003", "Take 002"])
            # "idle " is no take's own name: the first that matches it outer spaces aside
            for name, index in (("idle", 1), (" idle ", 0), ("idle ", 0), ("Take 003", 2), ("Take 002", 3), (" Take 002 ", 3)):
                with self.subTest(name):
                    clip = ClipData.load_fbx(path, take=name)
                    self.assertEqual(fingerprint(clip), fingerprint(clips[index]))
                    self.assertEqual(clip.take, clips[index].take)
            names = ("idle", "", "Take 002")
            blank = build_fbx(os.path.join(self.tmp.name, "blank.fbx"), takes=[(n, k) for n, (_, k) in zip(names, TAKES)])
            clips = ClipList.load_fbx(blank)
            self.assertEqual(clips.take, ["idle", "Take 002", "Take 002_1"])
            for name, index in (("Take 002", 1), ("Take 002_1", 2)):
                with self.subTest(name):
                    self.assertEqual(fingerprint(ClipData.load_fbx(blank, take=name)), fingerprint(clips[index]))

    def test_an_unknown_take_names_the_takes(self):
        path  = self.files["three_takes_middle_active"]
        takes = r"\['first', 'middle', 'last'\]"
        with self.assertRaisesRegex(KeyError, rf"no take named 'Walk'.*{takes}"):
            ClipData.load_fbx(path, take="Walk")
        for index in (3, -4):
            with self.subTest(index):
                with self.assertRaisesRegex(IndexError, rf"take {index} out of range, the file holds 3: {takes}"):
                    ClipData.load_fbx(path, take=index)

        # a file with no take holds no name either; an
        # index and None keep 3.2.0's one frame clip, its static pose
        still = build_fbx(os.path.join(self.tmp.name, "still.fbx"), takes=())
        with self.assertRaisesRegex(KeyError, r"no take named 'Walk Forward'; its takes are \[\]"):
            ClipData.load_fbx(still, take="Walk Forward")
        for take in (None, 0, 5):
            with self.subTest(take=take):
                clip = ClipData.load_fbx(still, take=take)
                self.assertEqual((clip.frame_count, clip.take), (1, ""))

    def test_the_default_and_none_as_in_3_2_0(self):
        """the first take holding a key, not the active take nor the last:
        3.2.0's meaning, kept in 3.3.0 (4.0.0 changes it); an int by position"""
        three, empty = self.files["three_takes_middle_active"], self.files["empty_first_take"]
        for path, want in ((three, "first"), (empty, "Walk Forward")):
            with self.subTest(os.path.basename(path)):
                self.assertEqual(ClipData.load_fbx(path).take, want)
                self.assertEqual(ClipData.load_fbx(path, take=None).take, want)
        self.assertEqual(ClipData.load_fbx(three, take=-1).take,      "last")
        self.assertEqual(ClipData.load_fbx(empty, take=0).take,       "Take 001")
        self.assertEqual(self.keys(ClipData.load_fbx(three, take=1)), (20, 30, 200.0, 220.0))
        # every take a name: the list's names load the list's takes
        clips = ClipList.load_fbx(empty)
        for clip in clips:
            self.assertEqual(fingerprint(ClipData.load_fbx(empty, take=clip.take)), fingerprint(clip))


if __name__ == "__main__":
    unittest.main()
