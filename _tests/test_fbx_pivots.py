"""Pivots and offsets through load_fbx: a transform's rotate pivot, scale
pivot, rotate offset and scale offset (Maya's rotatePivot, scalePivot,
rotatePivotTranslate and scalePivotTranslate) fold into its translate, on
every frame of a take; its 3x3 stays as the channels hold it.

Measured in Maya 2025: its exporter writes them as the sdk's pivot and
offset properties, with the rotate axis as an inverted post rotation; on
such a node the sdk's EvaluateGlobalTransform and Maya's worldMatrix agree
to 1e-15 on every frame, so the sdk's evaluation stands in for Maya's in
the sdk tests; Maya's re-import keeps the pivots and refits the translate to
the same world. A joint's pivots Maya's exporter drops (it writes zeros, and
its re-import lands 0.5 off a joint whose pivot counted), so a joint
carrying them comes from another writer and is refused, as is a keyed
pivot. The sdk tests need the Autodesk SDK, the Maya ones a Maya with its
fbxmaya plug-in.
"""

import unittest

import numpy as np
from cgmath._tests.test_skeleton_fbx import _Files, _InMaya
from cgmath.formats import load_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.formats._load import _amatrix

FBX = load_sdk()

NAMES  = ("grp", "j", "leaf")
FRAMES = 3


def _vector(values):
    return FBX.FbxVector4(*[float(v) for v in values], 1.0)


def _frames(block, width) -> np.ndarray:
    """a clip's one-node block, (F, 1, ...), as (F, ...)"""
    return np.asarray(block, dtype=np.float64).reshape((-1,) + width)


def _keys(prop, layer, component, values, mode):
    """``values`` keyed on frames 0, 1 ... of ``prop``'s ``component``, linear"""
    linear = FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    prop.GetCurveNode(layer, True)
    curve = typed(prop.GetCurve(layer, component, True), FBX.FbxAnimCurve, FBX)
    key   = FBX.FbxAnimCurveKey()
    key.SetInterpolation(linear)
    curve.KeyModifyBegin()
    for frame, value in enumerate(values):
        when = FBX.FbxTime()
        when.SetFrame(frame, mode)
        key.SetValue(float(value))
        curve.KeyAdd(when, key)
    curve.KeyModifyEnd()


def pivoted(path: str, keyed: bool = False, joint: bool = False, keyed_pivot: bool = False) -> dict:
    """a file the sdk writes, and ``{name: (F, 4, 4)}`` the sdk's own world
    matrices of its nodes on frames 0 .. FRAMES-1

    ``grp``, a transform turned (10, 20, 30) in zxy about a rotate axis of
    (5, 0, 0), scaled 2, at (3, 4, 5), with every pivot and offset set; under
    it the joint ``j`` at (1, 2, 3) with a joint orient of (0, 15, 0) and the
    leaf transform ``leaf``, scaled (1, 2, 3) about pivots of its own.
    ``keyed`` keys grp's rotate z and translate x and leaf's scale z (its
    scale pivot sits along z, so its position follows) over FRAMES frames
    at 30 fps; ``joint`` adds a root joint ``pj`` turned 30 about a rotate
    pivot; ``keyed_pivot`` keys grp's rotate pivot.
    """
    manager = FBX.FbxManager.Create()
    try:
        scene  = FBX.FbxScene.Create(manager, "")
        source = FBX.FbxNode.EPivotSet.eSourcePivot
        mode   = FBX.FbxTime.EMode.eFrames30
        scene.GetGlobalSettings().SetTimeMode(mode)

        grp  = FBX.FbxNode.Create(scene, "grp")
        j    = FBX.FbxNode.Create(scene, "j")
        leaf = FBX.FbxNode.Create(scene, "leaf")
        j.SetNodeAttribute(FBX.FbxSkeleton.Create(scene, ""))
        scene.GetRootNode().AddChild(grp)
        grp.AddChild(j)
        grp.AddChild(leaf)
        nodes = {"grp": grp, "j": j, "leaf": leaf}
        for node in nodes.values():
            node.SetRotationActive(True)
            node.SetTransformationInheritType(FBX.FbxTransform.EInheritType.eInheritRSrs)

        grp.SetRotationOrder(source, FBX.EFbxRotationOrder.eEulerZXY)
        grp.SetPostRotation(source, _vector((-5.0, 0.0, 0.0)))
        grp.SetRotationPivot(source, _vector((1.0, 2.0, 3.0)))
        grp.SetScalingPivot(source, _vector((-1.0, 0.5, 2.0)))
        grp.SetRotationOffset(source, _vector((0.5, 0.0, 0.0)))
        grp.SetScalingOffset(source, _vector((0.0, 0.25, 0.0)))
        grp.LclTranslation.Set(FBX.FbxDouble3(3.0, 4.0, 5.0))
        grp.LclRotation.Set(FBX.FbxDouble3(10.0, 20.0, 30.0))
        grp.LclScaling.Set(FBX.FbxDouble3(2.0, 2.0, 2.0))

        j.SetPreRotation(source, _vector((0.0, 15.0, 0.0)))
        j.LclTranslation.Set(FBX.FbxDouble3(1.0, 2.0, 3.0))

        leaf.SetRotationPivot(source, _vector((2.0, 0.0, 0.0)))
        leaf.SetScalingPivot(source, _vector((0.0, 0.0, 1.0)))
        leaf.LclTranslation.Set(FBX.FbxDouble3(0.0, 1.0, 0.0))
        leaf.LclRotation.Set(FBX.FbxDouble3(0.0, 45.0, 0.0))
        leaf.LclScaling.Set(FBX.FbxDouble3(1.0, 2.0, 3.0))

        if joint:
            pj = FBX.FbxNode.Create(scene, "pj")
            pj.SetNodeAttribute(FBX.FbxSkeleton.Create(scene, ""))
            pj.SetRotationActive(True)
            pj.SetRotationPivot(source, _vector((1.0, 0.0, 0.0)))
            pj.LclTranslation.Set(FBX.FbxDouble3(0.0, 10.0, 0.0))
            pj.LclRotation.Set(FBX.FbxDouble3(0.0, 0.0, 30.0))
            scene.GetRootNode().AddChild(pj)
            nodes["pj"] = pj

        stack = None
        if keyed or keyed_pivot:
            stack = FBX.FbxAnimStack.Create(scene, "take")
            layer = FBX.FbxAnimLayer.Create(scene, "BaseLayer")
            stack.AddMember(layer)
            if keyed:
                _keys(grp.LclRotation,    layer, "Z", (0.0, 45.0, 90.0), mode)
                _keys(grp.LclTranslation, layer, "X", (3.0, 4.0, 5.0),   mode)
                _keys(leaf.LclScaling,    layer, "Z", (3.0, 4.0, 5.0),   mode)
            if keyed_pivot:
                # the sdk gives a pivot no curve until it is made animatable
                grp.RotationPivot.ModifyFlag(FBX.FbxPropertyFlags.EFlags.eAnimatable, True)
                _keys(grp.RotationPivot, layer, "X", (1.0, 2.0, 3.0), mode)
            first, last = FBX.FbxTime(), FBX.FbxTime()
            first.SetFrame(0, mode)
            last.SetFrame(FRAMES - 1, mode)
            stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))
            scene.SetCurrentAnimationStack(stack)

        worlds = {}
        for name, node in nodes.items():
            rows = []
            for frame in range(FRAMES):
                when = FBX.FbxTime()
                when.SetFrame(frame, mode)
                rows.append(_amatrix(node.EvaluateGlobalTransform(when)))
            worlds[name] = np.array(rows)

        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return worlds


def pivots_of(path: str) -> dict:
    """``{name: (4, 3)}`` rotate pivot, scale pivot, rotate offset, scale offset of every node of the file"""
    out = {}
    with open_fbx(path, FBX) as scene:
        source = FBX.FbxNode.EPivotSet.eSourcePivot
        stack  = [scene.GetRootNode()]
        while stack:
            node = stack.pop()
            stack.extend(node.GetChild(i) for i in range(node.GetChildCount()))
            out[str(node.GetName())] = np.array([
                tuple(getter(source))[:3]
                for getter in (node.GetRotationPivot, node.GetScalingPivot, node.GetRotationOffset, node.GetScalingOffset)
            ])
    return out


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestPivots(_Files):
    """pivots and offsets fold into the translate; the worlds are the sdk's"""

    def test_static_pivots_fold_into_translate(self):
        path   = self.path("pivots.fbx")
        worlds = pivoted(path)
        rig    = load_fbx(path).hierarchy
        self.assertEqual(sorted(rig.name), sorted(NAMES))
        for name in NAMES:
            np.testing.assert_allclose(rig[name].world_matrix, worlds[name][0], rtol=0.0, atol=1e-9, err_msg=name)
        grp = rig["grp"]
        np.testing.assert_allclose(grp.rotate,      [10.0, 20.0, 30.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(grp.scale,       [2.0, 2.0, 2.0],    rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(grp.rotate_axis, [5.0, 0.0, 0.0],    rtol=0.0, atol=1e-12)
        self.assertEqual(grp.node_type, "transform")
        self.assertGreater(float(np.abs(np.asarray(grp.translate) - [3.0, 4.0, 5.0]).max()), 0.1)
        np.testing.assert_allclose(rig["leaf"].scale, [1.0, 2.0, 3.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(rig["j"].translate, [1.0, 2.0, 3.0], rtol=0.0, atol=1e-12)

    def test_a_keyed_node_folds_on_every_frame(self):
        """grp's rotate z and translate x keyed, leaf's scale z: each frame's
        translate holds that frame's pivots, keyed as a curve"""
        path   = self.path("keyed.fbx")
        worlds = pivoted(path, keyed=True)
        held   = load_fbx(path)
        self.assertIsNone(held.hierarchy)
        clip = held.clip.active
        self.assertEqual(clip.frame_count, FRAMES)
        for name in NAMES:
            np.testing.assert_allclose(
                _frames(clip.frames[name].world_matrix, (4, 4)), worlds[name], rtol=0.0, atol=1e-9, err_msg=name
            )
        moved = _frames(clip.frames["leaf"].translate, (3,))
        self.assertGreater(float(np.abs(moved[1:] - moved[:1]).max()), 1e-3, "leaf's translate follows its keyed scale")
        np.testing.assert_allclose(_frames(clip.frames["leaf"].scale, (3,))[:, 2], [3.0, 4.0, 5.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(_frames(clip.frames["grp"].rotate, (3,))[:, 2], [0.0, 45.0, 90.0], rtol=0.0, atol=1e-9)
        still = _frames(clip.frames["j"].translate, (3,))  # a channel no key moves is one row
        np.testing.assert_allclose(still, np.broadcast_to([1.0, 2.0, 3.0], still.shape), rtol=0.0, atol=1e-12)

    def test_a_joint_with_pivots_is_refused(self):
        path = self.path("joint.fbx")
        pivoted(path, joint=True)
        with self.assertRaises(ValueError) as held:
            load_fbx(path)
        self.assertIn("joint 'pj' channel 'pivots'", str(held.exception))

    def test_a_keyed_pivot_is_refused(self):
        path = self.path("keyed_pivot.fbx")
        pivoted(path, keyed_pivot=True)
        with self.assertRaises(ValueError) as held:
            load_fbx(path)
        self.assertIn("'grp' keys its pivots", str(held.exception))


class TestMayaPivots(_InMaya):
    """Maya's export of a pivoted transform reads back at Maya's worlds"""

    def test_maya_export_reads_at_maya_worlds(self):
        cmds = self.cmds
        cmds.currentUnit(time="ntsc", linear="cm", angle="deg")
        grp = cmds.createNode("transform", name="grp")
        cmds.setAttr(grp + ".rotatePivot", 1.0, 2.0, 3.0)
        cmds.setAttr(grp + ".scalePivot", -1.0, 0.5, 2.0)
        cmds.setAttr(grp + ".rotatePivotTranslate", 0.5, 0.0, 0.0)
        cmds.setAttr(grp + ".scalePivotTranslate", 0.0, 0.25, 0.0)
        cmds.setAttr(grp + ".rotate", 10.0, 20.0, 30.0)
        cmds.setAttr(grp + ".rotateOrder", 2)
        cmds.setAttr(grp + ".rotateAxis", 5.0, 0.0, 0.0)
        cmds.setAttr(grp + ".scale",      2.0, 2.0, 2.0)
        cmds.setAttr(grp + ".translate",  3.0, 4.0, 5.0)
        j = cmds.createNode("joint", name="j", parent=grp)
        cmds.setAttr(j + ".translate", 1.0, 2.0, 3.0)
        cmds.setAttr(j + ".jointOrient", 0.0, 15.0, 0.0)
        leaf = cmds.createNode("transform", name="leaf", parent=grp)
        cmds.setAttr(leaf + ".scale",       1.0, 2.0,  3.0)
        cmds.setAttr(leaf + ".rotatePivot", 2.0, 0.0,  0.0)
        cmds.setAttr(leaf + ".scalePivot",  0.0, 0.0,  1.0)
        cmds.setAttr(leaf + ".rotate",      0.0, 45.0, 0.0)
        cmds.setAttr(leaf + ".translate",   0.0, 1.0,  0.0)
        for frame, (rz, tx, sz) in enumerate(((0.0, 3.0, 3.0), (45.0, 4.0, 4.0), (90.0, 5.0, 5.0))):
            cmds.setKeyframe(grp,  attribute="rotateZ",    time=frame, value=rz)
            cmds.setKeyframe(grp,  attribute="translateX", time=frame, value=tx)
            cmds.setKeyframe(leaf, attribute="scaleZ",     time=frame, value=sz)
        # Maya's export spans the time slider
        cmds.playbackOptions(minTime=0, maxTime=FRAMES - 1, animationStartTime=0, animationEndTime=FRAMES - 1)
        worlds = {
            name: np.array([cmds.getAttr(name + ".worldMatrix[0]", time=frame) for frame in range(FRAMES)]).reshape(FRAMES, 4, 4)
            for name in NAMES
        }

        path = self.exported("maya_pivots.fbx")
        np.testing.assert_array_equal(pivots_of(path)["grp"], [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.5, 0.0, 0.0], [0.0, 0.25, 0.0]])
        clip = load_fbx(path).clip.active
        self.assertEqual(clip.frame_count, FRAMES)
        for name in NAMES:
            np.testing.assert_allclose(
                _frames(clip.frames[name].world_matrix, (4, 4)), worlds[name], rtol=0.0, atol=1e-6, err_msg=name
            )


if __name__ == "__main__":
    unittest.main()
