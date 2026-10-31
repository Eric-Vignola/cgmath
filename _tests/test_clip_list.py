"""A file's takes: the clip loaders, one take at a time and every take.

Its FBX files are built through the SDK (mayapy only), its GLB files through
pygltflib, here, keying each take apart, so a loader that read another take
than the one asked for shows.
"""

import hashlib
import os
import sys
import tempfile
import unittest

import numpy as np
from cgmath.formats._fbx_io import load_sdk
from cgmath.formats.glb import pygltflib
from cgmath.hierarchy import ClipData

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


def build_fbx(path, takes=TAKES, up_axis=1, unit=1.0):
    """an fbx of ``CHAIN`` under a node with no attribute, ``offset``, turned
    and moved, keyed by ``takes``, one stack each, at 24 fps; ``up_axis`` 2
    for a z-up file, ``unit`` the cm a unit holds"""
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


class TestClipListLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def check(self, label, clip):
        with self.subTest(label):
            self.assertEqual(fingerprint(clip), LITERALS[label])

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


if __name__ == "__main__":
    unittest.main()
