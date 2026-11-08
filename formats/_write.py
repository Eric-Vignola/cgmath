"""``cgmath.formats.save_fbx``: one skeleton, and one take animating it

The r17-skeleton fork writes a skeleton of joints and transforms at its bind
pose and, with a clip, one take of it. Nothing here imports the fbx sdk, nor
the classes, until a file is written: the classes' modules import this one.
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


def save_fbx(filename, skeleton, clip=None, as_ascii: bool = False) -> None:
    """writes ``skeleton``, a HierarchyData of joints and transforms, at its
    bind pose, and ``clip``, a ClipData of its nodes, as one take; binary
    unless ``as_ascii``

    The file's nodes hold the skeleton's own values: static Lcl Translation,
    Rotation and Scaling, the joint orient as the pre rotation, the rotate
    axis as the post rotation (its inverse), the rotate order, segment scale
    compensate as the inherit type (eInheritRrs on, eInheritRSrs off), the
    user defined attributes as user properties; and one bind pose (FbxPose)
    lists every node at its world matrix. The take, named ``clip.take`` or
    "Take 001", spans the clip's frames at its fps; every channel of every
    node has its curves, one a component: a constant channel one key on the
    clip's first frame, an animated one a key a frame.

    Refused, ValueError naming node, channel and frame, before anything is
    written: a node that is no joint or transform, a transform holding
    segment scale compensate, a shear, a scale that is not uniform on a node
    with children, in the skeleton or on any frame of the clip, and a clip
    whose nodes are not the skeleton's (names, parents, order, node types,
    joint orient, rotate axis, rotate order, segment scale compensate, user
    attributes and their types). Any other object: NotImplementedError.
    """
    # imported here: the classes' modules import this one
    from cgmath.hierarchy import hierarchy

    if hierarchy.FBX is None or hierarchy.FbxExporter is None:
        raise ImportError("fbx module not found")
    if not isinstance(skeleton, hierarchy.HierarchyData) or isinstance(skeleton, hierarchy.ClipData):
        raise NotImplementedError(
            f"save_fbx on {FORK} writes a HierarchyData skeleton and a ClipData clip, "
            f"not a {type(skeleton).__name__} skeleton"
        )
    if clip is not None and not isinstance(clip, hierarchy.ClipData):
        raise NotImplementedError(
            f"save_fbx on {FORK} writes one ClipData take, not a {type(clip).__name__}"
        )

    filename = os.path.expanduser(filename)
    _check_skeleton(skeleton)
    if clip is not None:
        _check_clip(skeleton, clip)

    fbx = hierarchy.FBX
    with hierarchy.FbxExporter() as exporter:
        exporter.add_skeleton(skeleton)
        nodes = [exporter._fbx_nodes[id(node)] for node in skeleton]
        for index, fbx_node in enumerate(nodes):
            prop = fbx.FbxProperty.Create(fbx_node, fbx.FbxIntDT, ORDER_PROPERTY, "")
            prop.Set(index)

        # the bind pose: every node at its world matrix
        pose = fbx.FbxPose.Create(exporter.scene, "BindPose")
        pose.SetIsBindPose(True)
        for fbx_node, world in zip(nodes, np.asarray(skeleton.world_matrix, dtype=np.float64)):
            matrix = fbx.FbxMatrix()
            for row in range(4):
                for column in range(4):
                    matrix.Set(row, column, float(world[row, column]))
            pose.Add(fbx_node, matrix)
        exporter.scene.AddPose(pose)

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


def _check_skeleton(skeleton) -> None:
    for node in skeleton:
        if node.node_type not in NODE_TYPES:
            raise ValueError(
                f"save_fbx: node {node.name!r} is a {node.node_type!r}; a skeleton on {FORK} holds joints and "
                f"transforms"
            )
        if node.node_type == "transform" and node.segment_scale_compensate:
            raise ValueError(
                f"save_fbx: node {node.name!r} channel 'segment_scale_compensate' is on a transform, which Maya "
                f"holds on joints only"
            )
        if node.node_type == "transform" and np.any(np.asarray(node.joint_orient) != 0.0):
            raise ValueError(
                f"save_fbx: node {node.name!r} channel 'joint_orient' is on a transform, which Maya holds on "
                f"joints only"
            )
    shear = np.array([node.shear for node in skeleton], dtype=np.float64).reshape(1, -1, 3)
    scale = np.array([node.scale for node in skeleton], dtype=np.float64).reshape(1, -1, 3)
    refuse_values("save_fbx, the skeleton", skeleton, shear, scale)


def _user_types(node) -> dict:
    from cgmath.hierarchy.hierarchy import _user_attr_type

    return {name: _user_attr_type(spec) for name, spec in node.user_defined_attributes.items()}


def _check_clip(skeleton, clip) -> None:
    from cgmath.hierarchy.hierarchy import _fbx_rate_mode

    if not clip.frame_count:
        raise ValueError("save_fbx: the clip holds no frame")
    if _fbx_rate_mode(float(clip.fps)) is None:
        raise ValueError(f"save_fbx: no fbx time mode carries {clip.fps} fps")
    if len(clip) != len(skeleton):
        raise ValueError(f"save_fbx: the clip holds {len(clip)} nodes, the skeleton {len(skeleton)}")

    mine   = parents_of(list(skeleton))
    theirs = parents_of(list(clip))
    for column, (node, held) in enumerate(zip(skeleton, clip)):
        checks = (
            ("name", node.name, held.name),
            ("parent", int(mine[column]), int(theirs[column])),
            ("node_type", node.node_type, held.node_type),
            ("rotate_order", node.rotate_order, held.rotate_order),
            ("segment_scale_compensate", bool(node.segment_scale_compensate), bool(held.segment_scale_compensate)),
            ("user_defined_attributes", _user_types(node), _user_types(held)),
        )
        for channel, want, got in checks:
            if want != got:
                raise ValueError(
                    f"save_fbx: the clip's node {column} {held.name!r} channel {channel!r} differs from the "
                    f"skeleton's {node.name!r}: {got!r}, the skeleton {want!r}"
                )

    frames = clip.start_frame + np.arange(clip.frame_count)
    for channel in ("joint_orient", "rotate_axis"):
        want = np.array([getattr(node, channel) for node in skeleton], dtype=np.float64)
        got  = np.asarray(getattr(clip.frames, channel), dtype=np.float64)
        bad  = np.argwhere(np.any(got != want[None], axis=-1))
        if len(bad):
            frame, column = (int(x) for x in bad[0])
            raise ValueError(
                f"save_fbx: the clip's node {clip[column].name!r} channel {channel!r} frame {int(frames[frame])} "
                f"is {got[frame, column].tolist()}, the skeleton's {want[column].tolist()}"
            )
    refuse_values("save_fbx, the clip", clip, clip.frames.shear, clip.frames.scale, frames)


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
