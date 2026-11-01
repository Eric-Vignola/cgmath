"""Writes the fbx files of test_fbx_maya_import.py from their layouts.

Maya's import of these files is recorded in test_assets/fbx_maya_import.npz by
tools/make_fbx_maya_import_fixture.py, which holds the layouts and writes each
file with this one function before Maya imports it; the test writes them
again from the layouts the archive holds, so no fbx file ships. It needs the
Autodesk FBX SDK and nothing of cgmath: the code under test has no hand in
its own reference.

A layout is a dict: ``nodes``, a list of node dicts, parents first, ``axes``
("yup", "zup" or "xup"), ``span`` (the take's first and last frame) and
optionally ``fps`` (24 by default), ``cubic`` (bezier keys, linear by
default) and ``layers``, more animation layers as ``[mode, {node: keys}]``,
a third entry True on a muted one, ``keys`` as a node's own and optionally
``ckeys`` among them, one axis each.

A node dict holds ``name``, ``kind`` ("joint", "group", "locator", "mesh",
"camera", "light", "lod" or "none", no node attribute), ``parent`` (a name or
None), ``t``, ``r``, ``s``, ``order``, ``pre``, ``post``, ``inherit`` ("RSrs",
"RrSs" or "Rrs"), ``piv`` (rotate pivot, scale pivot, rotate offset, scale
offset, or None), ``keys`` (``{channel: [frames, values]}``, all three axes),
``ckeys`` (``[channel, axis, frames, values]``, one axis), ``points`` (a
mesh's points) and optionally ``skin``, the joint a mesh is skinned to,
``skin_empty``, its skin left with no cluster, ``repeats``, channels whose
curves repeat before their keys, with the count Maya's export writes, and
``active`` False, its rotation inactive.
"""
import numpy as np

try:
    import fbx as FBX
except ImportError:
    FBX = None


def _axes(name: str):
    if name == "zup":
        return FBX.FbxAxisSystem.MayaZUp
    if name == "xup":
        axes = FBX.FbxAxisSystem
        return axes(axes.EUpVector.eXAxis, axes.EFrontVector.eParityOdd, axes.ECoordSystem.eRightHanded)
    return FBX.FbxAxisSystem.MayaYUp


def add_curve(layer, prop, axis: int, frames, values, fps: float = 24, cubic: bool = False) -> None:
    """keys ``values`` on ``frames`` on one axis of ``prop``, in ``layer``"""
    kinds = FBX.FbxAnimCurveDef
    prop.GetCurveNode(layer, True)
    curve = prop.GetCurve(layer, "XYZ"[axis], True)
    curve.KeyModifyBegin()
    for frame, value in zip(frames, values):
        time = FBX.FbxTime()
        time.SetSecondDouble(frame / float(fps))
        key = curve.KeyAdd(time)[0]
        curve.KeySetValue(key, float(value))
        curve.KeySetInterpolation(
            key, kinds.EInterpolationType.eInterpolationCubic if cubic else kinds.EInterpolationType.eInterpolationLinear
        )
        if cubic:
            curve.KeySetTangentMode(key, kinds.ETangentMode.eTangentAuto)
    curve.KeyModifyEnd()


def _add_keys(layer, node, keys: dict, fps: float, cubic: bool) -> None:
    props = {"t": node.LclTranslation, "r": node.LclRotation, "s": node.LclScaling}
    for channel, (frames, values) in keys.items():
        values = np.asarray(values, dtype=float)
        for axis in range(3):
            add_curve(layer, props[channel], axis, frames, values[:, axis], fps, cubic)


def _attribute(scene, spec: dict):
    """the node attribute of a node dict, None for kind "none" """
    name, kind = spec["name"], spec["kind"]
    if kind == "joint":
        skeleton = FBX.FbxSkeleton.Create(scene, name)
        skeleton.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
        return skeleton
    if kind in ("group", "locator"):
        null = FBX.FbxNull.Create(scene, name)
        null.Look.Set((FBX.FbxNull.ELook.eCross if kind == "locator" else FBX.FbxNull.ELook.eNone).value)
        return null
    if kind == "camera":
        return FBX.FbxCamera.Create(scene, name)
    if kind == "light":
        return FBX.FbxLight.Create(scene, name)
    if kind == "lod":
        return FBX.FbxLODGroup.Create(scene, name)
    if kind == "none":
        return None

    mesh   = FBX.FbxMesh.Create(scene, name)
    points = spec["points"]
    mesh.InitControlPoints(len(points))
    for index, point in enumerate(points):
        mesh.SetControlPointAt(FBX.FbxVector4(*point, 1.0), index)
    for triangle in ((0, 1, 2), (0, 2, 3), (0, 3, 1)):
        mesh.BeginPolygon()
        for index in triangle:
            mesh.AddPolygon(index)
        mesh.EndPolygon()
    return mesh


def _skin(scene, node, spec: dict, joint) -> None:
    mesh = node.GetNodeAttribute()
    skin = FBX.FbxSkin.Create(scene, spec["name"] + "_skin")
    mesh.AddDeformer(skin)
    if spec.get("skin_empty"):
        return
    cluster = FBX.FbxCluster.Create(scene, spec["name"] + "_cl")
    cluster.SetLink(joint)
    cluster.SetLinkMode(FBX.FbxCluster.ELinkMode.eNormalize)
    for index in range(mesh.GetControlPointsCount()):
        cluster.AddControlPointIndex(index, 1.0)
    cluster.SetTransformMatrix(node.EvaluateGlobalTransform())
    cluster.SetTransformLinkMatrix(joint.EvaluateGlobalTransform())
    skin.AddCluster(cluster)


def write_layout(path: str, layout: dict) -> None:
    """writes the fbx file of ``layout`` to ``path``"""
    source  = FBX.FbxNode.EPivotSet.eSourcePivot
    inherit = FBX.FbxTransform.EInheritType
    types   = {"RrSs": inherit.eInheritRrSs, "RSrs": inherit.eInheritRSrs, "Rrs": inherit.eInheritRrs}
    rates   = {24: FBX.FbxTime.EMode.eFrames24, 30: FBX.FbxTime.EMode.eFrames30, 60: FBX.FbxTime.EMode.eFrames60}
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    scene = FBX.FbxScene.Create(manager, "scene")
    fps   = layout.get("fps", 24)
    cubic = layout.get("cubic", False)
    scene.GetGlobalSettings().SetAxisSystem(_axes(layout["axes"]))
    scene.GetGlobalSettings().SetTimeMode(rates[fps])
    stack = FBX.FbxAnimStack.Create(scene, "Take 001")
    layer = FBX.FbxAnimLayer.Create(scene, "Base Layer")
    stack.AddMember(layer)

    made = {}
    for spec in layout["nodes"]:
        name      = spec["name"]
        node      = FBX.FbxNode.Create(scene, name)
        attribute = _attribute(scene, spec)
        if attribute is not None:
            node.SetNodeAttribute(attribute)
        node.LclTranslation.Set(FBX.FbxDouble3(*spec["t"]))
        node.LclRotation.Set(FBX.FbxDouble3(*spec["r"]))
        node.LclScaling.Set(FBX.FbxDouble3(*spec["s"]))
        node.SetRotationActive(spec.get("active", True))
        node.SetRotationOrder(source, getattr(FBX.EFbxRotationOrder, "eEuler" + spec["order"].upper()))
        if spec["pre"] is not None:
            node.SetPreRotation(source, FBX.FbxVector4(*spec["pre"], 1.0))
        if spec["post"] is not None:
            node.SetPostRotation(source, FBX.FbxVector4(*spec["post"], 1.0))
        if spec["piv"] is not None:
            rotate, scale, rotate_offset, scale_offset = spec["piv"]
            node.SetRotationPivot(source, FBX.FbxVector4(*rotate, 1.0))
            node.SetScalingPivot(source, FBX.FbxVector4(*scale, 1.0))
            node.SetRotationOffset(source, FBX.FbxVector4(*rotate_offset, 1.0))
            node.SetScalingOffset(source, FBX.FbxVector4(*scale_offset, 1.0))
        node.SetTransformationInheritType(types[spec["inherit"]])
        (made[spec["parent"]] if spec["parent"] else scene.GetRootNode()).AddChild(node)
        _add_keys(layer, node, spec["keys"], fps, cubic)
        props = {"t": node.LclTranslation, "r": node.LclRotation, "s": node.LclScaling}
        for channel, axis, frames, values in spec["ckeys"]:
            add_curve(layer, props[channel], axis, frames, values, fps, cubic)
        for channel in spec.get("repeats", ()):
            for axis in "XYZ":
                curve = props[channel].GetCurve(layer, axis, False)
                curve.SetPreExtrapolation(FBX.FbxAnimCurveBase.EExtrapolationType.eRepetition)
                curve.SetPreExtrapolationCount(2**32 - 1)
        if spec.get("skin"):
            _skin(scene, node, spec, made[spec["skin"]])
        made[name] = node

    for mode, keyed, *muted in layout.get("layers", ()):
        extra = FBX.FbxAnimLayer.Create(scene, "Layer " + mode)
        stack.AddMember(extra)
        extra.BlendMode.Set({"additive": 0, "override": 1}[mode])
        extra.Mute.Set(bool(muted))
        for name, keys in keyed.items():
            keys  = dict(keys)
            props = {"t": made[name].LclTranslation, "r": made[name].LclRotation, "s": made[name].LclScaling}
            for channel, axis, frames, values in keys.pop("ckeys", ()):
                add_curve(extra, props[channel], axis, frames, values, fps, cubic)
            _add_keys(extra, made[name], keys, fps, cubic)

    first, last = FBX.FbxTime(), FBX.FbxTime()
    first.SetSecondDouble(layout["span"][0] / float(fps))
    last.SetSecondDouble(layout["span"][1] / float(fps))
    stack.SetLocalTimeSpan(FBX.FbxTimeSpan(first, last))

    exporter = FBX.FbxExporter.Create(manager, "")
    exporter.Initialize(path, -1, manager.GetIOSettings())
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
