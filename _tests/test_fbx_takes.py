"""A take's layers and span through load_fbx.

Layers holding no curve are passed over: an Unreal export (measured on the
Luna animations) carries a 'root_motion', an 'idle_pose' and a muted
'lean_NOT_4_EXPORT' layer with no curve beside its BaseLayer. Curves on two
layers, or on a muted one, blend, which the fork does not read. The clip
covers the stored span widened to the keys: an Unreal export stores the span
a frame short of its last key (measured: span 0-45 frames, keys to 46), a
Maya export can store one wider than its keys. Needs the Autodesk SDK.
"""

import unittest

import numpy as np
from cgmath._tests.test_skeleton_fbx import _Files
from cgmath.formats import load_fbx
from cgmath.formats._fbx_io import load_sdk, typed

FBX = load_sdk()


def stacked(path: str, layers=(("BaseLayer", True, False),), span=(0, 2), keys=(0, 1, 2)) -> str:
    """a file the sdk writes: a node ``n`` under one take at 30 fps whose
    layers are ``(name, keyed, muted)`` each, a keyed one holding n's
    translate x on the frames ``keys`` at their own frame numbers, the
    stack's span ``span`` in frames"""
    manager = FBX.FbxManager.Create()
    try:
        scene = FBX.FbxScene.Create(manager, "")
        mode  = FBX.FbxTime.EMode.eFrames30
        scene.GetGlobalSettings().SetTimeMode(mode)
        node = FBX.FbxNode.Create(scene, "n")
        node.SetRotationActive(True)
        scene.GetRootNode().AddChild(node)
        stack  = FBX.FbxAnimStack.Create(scene, "take")
        linear = FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
        for name, keyed, muted in layers:
            layer = FBX.FbxAnimLayer.Create(scene, name)
            layer.Mute.Set(bool(muted))
            stack.AddMember(layer)
            if not keyed:
                continue
            node.LclTranslation.GetCurveNode(layer, True)
            curve = typed(node.LclTranslation.GetCurve(layer, "X", True), FBX.FbxAnimCurve, FBX)
            key   = FBX.FbxAnimCurveKey()
            key.SetInterpolation(linear)
            curve.KeyModifyBegin()
            for frame in keys:
                when = FBX.FbxTime()
                when.SetFrame(int(frame), mode)
                key.SetValue(float(frame))
                curve.KeyAdd(when, key)
            curve.KeyModifyEnd()
        first, last = FBX.FbxTime(), FBX.FbxTime()
        first.SetFrame(int(span[0]), mode)
        last.SetFrame(int(span[1]), mode)
        stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestLayers(_Files):
    def test_layers_holding_no_curve_are_passed_over(self):
        """the Unreal layout: the curves on the first layer, three empty
        layers after it, one muted; and the curves on a later layer"""
        plain = load_fbx(stacked(self.path("one.fbx"))).clip.active
        for label, layers in (
            ("unreal", (("BaseLayer", True, False), ("root_motion", False, False), ("idle_pose", False, False), ("lean_NOT_4_EXPORT", False, True))),
            ("later", (("empty", False, False), ("keyed", True, False))),
        ):
            with self.subTest(label):
                clip = load_fbx(stacked(self.path(label + ".fbx"), layers=layers)).clip.active
                self.assertSameClip(clip, plain)
                np.testing.assert_array_equal(clip.frames["n"].translate[:, 0], [0.0, 1.0, 2.0])

    def test_curves_on_two_layers_are_refused(self):
        path = stacked(self.path("two.fbx"), layers=(("BaseLayer", True, False), ("add", True, False), ("empty", False, False)))
        with self.assertRaises(NotImplementedError) as held:
            load_fbx(path)
        self.assertIn("2 animation layers with curves", str(held.exception))

    def test_curves_on_a_muted_layer_are_refused(self):
        path = stacked(self.path("muted.fbx"), layers=(("BaseLayer", False, False), ("muted", True, True)))
        with self.assertRaises(NotImplementedError) as held:
            load_fbx(path)
        self.assertIn("muted layer", str(held.exception))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSpan(_Files):
    def test_the_clip_covers_the_span_and_every_key(self):
        """a span a frame short of the last key (Unreal), keys before the
        span, a span wider than the keys (Maya), a span that is the keys"""
        for label, span, keys, start, count in (
            ("short", (0, 4), (0, 1, 2, 3, 4, 5), 0, 6),
            ("before", (0, 4), (-2, 0, 4), -2, 7),
            ("wide", (0, 8), (0, 5), 0, 9),
            ("exact", (2, 5), (2, 5), 2, 4),
        ):
            with self.subTest(label):
                clip = load_fbx(stacked(self.path(label + ".fbx"), span=span, keys=keys)).clip.active
                self.assertEqual((clip.start_frame, clip.frame_count), (start, count))
                np.testing.assert_allclose(clip.frames.times, (start + np.arange(count)) / 30.0, rtol=0.0, atol=1e-12)
                frames = start + np.arange(count)
                held   = np.interp(frames, keys, keys)
                np.testing.assert_allclose(clip.frames["n"].translate[:, 0], held, rtol=0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
