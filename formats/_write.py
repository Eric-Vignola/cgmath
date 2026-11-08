"""``cgmath.formats.save_fbx``: one skeleton, or one take

The r17-skeleton fork writes a skeleton of joints and transforms, or a
clip's nodes and its take; no bind pose. Nothing here imports the fbx sdk,
nor the classes, until a file is written: the classes' modules import this
one.
"""

import logging
import os

import numpy as np

# the fbx writer's logger: its warnings and save_fbx's are one stream
LOGGER = logging.getLogger("cgmath.formats.fbx")

FORK   = "the r17-skeleton fork"

# a scale whose components differ by more than this, relative to the
# largest, is not uniform
UNIFORM_SCALE = 1e-6

# the node kinds a skeleton holds
NODE_TYPES = ("joint", "transform")

# the hidden property keeping each node's place in the skeleton's list
ORDER_PROPERTY = "cgmath:index"


def save_fbx(filename, skeleton=None, clip=None, as_ascii: bool = False) -> None:
    """writes one of ``skeleton``, a HierarchyData of joints and transforms,
    or ``clip``, a ClipData of joints and transforms; binary unless
    ``as_ascii``

    A skeleton alone: its nodes and their own values, no take. A clip alone:
    its nodes, their static values its pose at its first frame, and one take,
    named ``clip.take`` or "Take 001", spanning the clip's frames at its fps;
    every channel of every node has its curves, one a component: a constant
    channel one key on the clip's first frame, an animated one a key a
    frame. No bind pose is written: a skeleton given with a clip is a bind
    pose, which comes with skins (NotImplementedError).

    Refused, ValueError naming node, channel and frame, before anything is
    written: a node that is no joint or transform, a transform holding
    segment scale compensate or a joint orient, a shear, a scale that is not
    uniform on a node with children, in the skeleton or on any frame of the
    clip, a joint orient or rotate axis the clip changes over its frames.
    Neither given, a skeleton that is not a HierarchyData (a ClipData is
    passed as ``clip``), a clip that is not a ClipData: TypeError.
    """
    # imported here: the classes' modules import this one
    from cgmath.hierarchy import hierarchy

    if skeleton is not None and clip is not None:
        raise NotImplementedError(
            f"save_fbx on {FORK}: a skeleton given with a clip is a bind pose, which comes with skins; "
            f"pass clip= alone, or a skeleton alone"
        )
    if skeleton is None and clip is None:
        raise TypeError("save_fbx takes a skeleton (a HierarchyData) or clip= (a ClipData)")
    if skeleton is not None and type(skeleton) is not hierarchy.HierarchyData:
        hint = ", pass it as clip=" if isinstance(skeleton, hierarchy.ClipData) else ""
        raise TypeError(f"save_fbx: the skeleton is a HierarchyData, not a {type(skeleton).__name__}{hint}")
    if clip is not None and not isinstance(clip, hierarchy.ClipData):
        raise TypeError(f"save_fbx: the clip is a ClipData, not a {type(clip).__name__}")
    if hierarchy.FBX is None or hierarchy.FbxExporter is None:
        raise ImportError("fbx module not found")

    filename = os.path.expanduser(filename)
    if clip is None:
        _check_skeleton(skeleton)
        written = skeleton
    else:
        _check_clip(clip)
        written = clip.at(clip.start_frame)

    fbx = hierarchy.FBX
    with hierarchy.FbxExporter() as exporter:
        exporter.add_skeleton(written)
        nodes = [exporter._fbx_nodes[id(node)] for node in written]
        for index, fbx_node in enumerate(nodes):
            prop = fbx.FbxProperty.Create(fbx_node, fbx.FbxIntDT, ORDER_PROPERTY, "")
            prop.Set(index)
        if clip is not None:
            _write_take(exporter, nodes, clip)
        exporter.export(path=filename, as_ascii=as_ascii)


def nonuniform(scale: np.ndarray) -> np.ndarray:
    """(...) bool, the (..., 3) scales whose components differ by more than
    UNIFORM_SCALE of the largest"""
    scale  = np.asarray(scale, dtype=np.float64)
    spread = scale.max(axis=-1) - scale.min(axis=-1)
    return spread > UNIFORM_SCALE * np.abs(scale).max(axis=-1)


def parents_of(nodes) -> np.ndarray:
    """(N,) the position of each node's parent among ``nodes``, -1 for a root"""
    place = {node.uuid: i for i, node in enumerate(nodes)}
    return np.array([place.get(node.parent_uuid, -1) for node in nodes], dtype=np.intp)


def refuse_values(what: str, nodes, shear, scale, frames=None) -> None:
    """ValueError naming the first node, channel and frame holding a shear
    other than 0, or a scale that is not uniform on a node with children;
    ``shear`` and ``scale`` (F, N, 3), ``frames`` the frame numbers, None
    for one pose"""
    nodes   = list(nodes)
    parents = parents_of(nodes)
    held    = np.zeros(len(nodes), dtype=bool)
    held[parents[parents >= 0]] = True
    for channel, bad, why in (
        ("shear", np.any(np.asarray(shear) != 0.0, axis=-1), "a shear, which fbx and Maya's joints hold none of"),
        ("scale", nonuniform(scale) & held[None], "a scale that is not uniform on a node with children"),
    ):
        found = np.argwhere(bad)
        if len(found):
            frame, column = (int(x) for x in found[0])
            value = np.asarray(shear if channel == "shear" else scale)[frame, column]
            when  = "" if frames is None else f" frame {int(frames[frame])}"
            raise ValueError(
                f"{what}: node {nodes[column].name!r} channel {channel!r}{when} holds {why}: "
                f"{np.round(value, 12).tolist()} ({len(np.unique(found[:, 1]))} nodes, {len(found)} values)"
            )


def _check_kinds(nodes, orients) -> None:
    """a node that is no joint or transform, a transform holding segment
    scale compensate, or a joint orient (``orients`` (F, N, 3)): refused"""
    for column, node in enumerate(nodes):
        if node.node_type not in NODE_TYPES:
            raise ValueError(
                f"save_fbx: node {node.name!r} is a {node.node_type!r}; {FORK} writes joints and transforms"
            )
        if node.node_type == "transform" and node.segment_scale_compensate:
            raise ValueError(
                f"save_fbx: node {node.name!r} channel 'segment_scale_compensate' is on a transform, which Maya "
                f"holds on joints only"
            )
        if node.node_type == "transform" and np.any(orients[:, column] != 0.0):
            raise ValueError(
                f"save_fbx: node {node.name!r} channel 'joint_orient' is on a transform, which Maya holds on "
                f"joints only"
            )


def _check_skeleton(skeleton) -> None:
    nodes = list(skeleton)
    _check_kinds(nodes, np.array([node.joint_orient for node in nodes], dtype=np.float64).reshape(1, -1, 3))
    shear = np.array([node.shear for node in nodes], dtype=np.float64).reshape(1, -1, 3)
    scale = np.array([node.scale for node in nodes], dtype=np.float64).reshape(1, -1, 3)
    refuse_values("save_fbx, the skeleton", nodes, shear, scale)


def _check_clip(clip) -> None:
    from cgmath.hierarchy.hierarchy import _fbx_rate_mode

    if not clip.frame_count:
        raise ValueError("save_fbx: the clip holds no frame")
    if _fbx_rate_mode(float(clip.fps)) is None:
        raise ValueError(f"save_fbx: no fbx time mode carries {clip.fps} fps")

    nodes  = list(clip)
    frames = clip.start_frame + np.arange(clip.frame_count)
    _check_kinds(nodes, np.asarray(clip.frames.joint_orient, dtype=np.float64))

    # a file holds one joint orient and one rotate axis a node, unkeyed
    for channel in ("joint_orient", "rotate_axis"):
        held = np.asarray(getattr(clip.frames, channel), dtype=np.float64)
        bad  = np.argwhere(np.any(held != held[:1], axis=-1))
        if len(bad):
            frame, column = (int(x) for x in bad[0])
            raise ValueError(
                f"save_fbx, the clip: node {nodes[column].name!r} channel {channel!r} frame {int(frames[frame])} "
                f"is {held[frame, column].tolist()}, its first frame {held[0, column].tolist()}: a file holds one "
                f"{channel} a node, which no curve animates"
            )
    refuse_values("save_fbx, the clip", nodes, clip.frames.shear, clip.frames.scale, frames)


def _keys(curve, times, values, mode, fbx) -> None:
    """every key of ``curve`` replaced by one at each of ``times``"""
    key = fbx.FbxAnimCurveKey()
    key.SetInterpolation(mode)
    curve.KeyModifyBegin()
    try:
        curve.KeyClear()
        for when, value in zip(times, np.asarray(values, dtype=np.float64).tolist()):
            key.SetValue(value)
            curve.KeyAdd(when, key)
    finally:
        curve.KeyModifyEnd()


def _write_take(exporter, nodes, clip) -> None:
    """one take: a stack spanning the clip's frames, a BaseLayer, and the
    curves of every channel of every node"""
    from cgmath.formats._fbx_io import typed
    from cgmath.formats.fbx import _user_property_value
    from cgmath.hierarchy.hierarchy import FBX as fbx
    from cgmath.hierarchy.hierarchy import _fbx_rate_mode, _user_channels

    scene = exporter.scene
    mode  = getattr(fbx.FbxTime.EMode, _fbx_rate_mode(float(clip.fps)))
    scene.GetGlobalSettings().SetTimeMode(mode)

    frames = clip.start_frame + np.arange(clip.frame_count)
    times  = []
    for frame in frames.tolist():
        when = fbx.FbxTime()
        when.SetFrame(int(frame), mode)
        times.append(when)
    first, last = fbx.FbxTime(times[0].Get()), fbx.FbxTime(times[-1].Get())
    # the sdk writes no span (0, 0) and reads another in its place: one tick
    # past 0, which a read puts on frame 0
    if first.Get() == 0 and last.Get() == 0:
        last.Set(1)

    name  = clip.take.strip() or "Take 001"
    stack = fbx.FbxAnimStack.Create(scene, name)
    stack.SetLocalTimeSpan(fbx.FbxTimeSpan(first, last))
    stack.SetReferenceTimeSpan(fbx.FbxTimeSpan(first, last))
    layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)
    scene.ActiveAnimStackName.Set(fbx.FbxString(name))

    linear = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    clip._settle()
    blocks = {
        "_translate": "LclTranslation",
        "_rotate":    "LclRotation",
        "_scale":     "LclScaling",
    }
    for channel, prop_name in blocks.items():
        block = np.asarray(clip._blocks[channel], dtype=np.float64)
        keyed = clip._keyed(channel)
        for column, fbx_node in enumerate(nodes):
            prop = getattr(fbx_node, prop_name)
            prop.GetCurveNode(layer, True)
            rows = block[:, column] if keyed[column] else block[:1, column]
            for axis, component in enumerate("XYZ"):
                curve = typed(prop.GetCurve(layer, component, True), fbx.FbxAnimCurve, fbx)
                _keys(curve, times[: len(rows)], rows[:, axis], linear, fbx)

    # each animatable user property: one curve, or three for a vector
    flag = fbx.FbxPropertyFlags.EFlags.eAnimatable
    for node, fbx_node in zip(clip, nodes):
        attrs = node.user_defined_attributes
        keys  = clip._user_keys.get(node.uuid, {})
        for attr, children in _user_channels(attrs).items():
            prop = fbx_node.FindProperty(attr)
            if not prop.IsValid():
                continue
            prop.ModifyFlag(flag, True)
            rows = keys.get(attr)
            if rows is None:
                rows = np.array([[float(attrs[x]["value"]) for x in ((attr,) if children is None else children)]])
            rows = np.asarray(rows, dtype=np.float64)
            prop.GetCurveNode(layer, True)
            if children is None:
                # an enum keys its field's place in the list, as its property holds it
                if attrs[attr].get("attributeType") == "enum":
                    rows = np.array([[float(_user_property_value(attrs[attr], int(x)))] for x in rows[:, 0]])
                curve = typed(prop.GetCurve(layer, True), fbx.FbxAnimCurve, fbx)
                _keys(curve, times[: len(rows)], rows[:, 0], linear, fbx)
                continue
            for axis, component in enumerate("XYZ"):
                curve = typed(prop.GetCurve(layer, component, True), fbx.FbxAnimCurve, fbx)
                _keys(curve, times[: len(rows)], rows[:, axis], linear, fbx)
