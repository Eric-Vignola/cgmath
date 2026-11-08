"""``cgmath.formats.load_fbx``: one skeleton, or one take

The r17-skeleton fork reads every node of a file as a joint (a skeleton
node) or a transform (any other: a group, or a transform holding a shape, a
mesh, a camera, a light or a locator, the shape left out), at their own
values, a bind pose left out, or the active take animating them.
"""

import os

import numpy as np

from cgmath.formats._write import FORK, ORDER_PROPERTY, refuse_values

# a parent scale this close to 1 leaves an inherit type nothing to change
UNIT_SCALE = 1e-6


def load_fbx(filename) -> tuple:
    """``(None, clip)`` for an fbx file holding a take: a ClipData of its
    active take; ``(hierarchy, None)`` for one holding none: a HierarchyData
    of its nodes, joints and transforms, at their own values

    The clip is the active take (``ActiveAnimStackName``, else the first):
    its span and frame rate, each channel constant where it has no curve
    (the node's own value) or a curve of one key, animated where a curve
    holds more, sampled on every frame of the span. A bind pose is left
    out: no node takes its values from one.

    Refused, ValueError naming node, channel and frame: a shear, a scale
    that is not uniform on a node with children (the nodes' own values and
    any frame), an inherit type a joint or transform can not hold where the
    parent's scale is not 1, a pre rotation on a transform, pivots and
    offsets. Several animation layers: NotImplementedError.
    """
    read = _open(filename, True)
    skeleton, order = _build(read)
    if read["take"] is None:
        return skeleton, None
    return None, _build_clip(read, skeleton, order)


def load_nodes(filename):
    """the HierarchyData of an fbx file's nodes at their own values, whether
    it holds a take or not (the take left unread)"""
    return _build(_open(filename, False))[0]


def _open(filename, take: bool) -> dict:
    # imported here: the classes' modules import this one
    from cgmath.formats._fbx_io import open_fbx
    from cgmath.hierarchy.hierarchy import FBX

    if FBX is None:
        raise ImportError("fbx module not found")
    filename = os.path.expanduser(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with open_fbx(filename, FBX) as scene:
        return _read(scene, FBX, take)


def _walk(scene) -> list:
    """[(node, parent position or -1)] of every node under the scene root,
    depth first, each parent ahead of its children, in the file's order"""
    found   = []
    root    = scene.GetRootNode()
    pending = [(root.GetChild(i), -1) for i in reversed(range(root.GetChildCount()))]
    while pending:
        node, parent = pending.pop()
        found.append((node, parent))
        here = len(found) - 1
        for i in reversed(range(node.GetChildCount())):
            pending.append((node.GetChild(i), here))
    return found


def _read(scene, fbx, take: bool = True) -> dict:
    """everything the build needs, out of the scene as python and numpy; the
    active take's curves unless not ``take``"""
    from cgmath.formats._fbx_io import typed
    from cgmath.hierarchy.hierarchy import (
        _FBX_ROTATE_ORDERS,
        _FBX_VISIBLE,
        _axis_words,
        _fbx_frame_rate,
        _fbx_joint_look,
        _fbx_negative_zeros,
        _fbx_scene_axes,
        _fbx_user_attributes,
        _fbx_written_axis,
    )

    source  = fbx.FbxNode.EPivotSet.eSourcePivot
    inherit = fbx.FbxTransform.EInheritType
    found   = _walk(scene)
    nodes   = []
    for node, parent in found:
        name      = str(node.GetName())
        attribute = node.GetNodeAttribute()
        joint     = attribute is not None and attribute.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton
        active    = bool(node.GetRotationActive())
        stored    = node.GetRotationOrder(source)
        stored    = int(getattr(stored, "value", stored))
        signs     = _fbx_negative_zeros(node)

        translate = np.array(tuple(node.LclTranslation.Get())[:3], dtype=np.float64)
        rotate    = np.array(tuple(node.LclRotation.Get())[:3],    dtype=np.float64)
        scale     = np.array(tuple(node.LclScaling.Get())[:3],     dtype=np.float64)
        orient    = np.zeros(3)
        axis      = np.zeros(3)
        if active:
            orient = np.array(tuple(node.GetPreRotation(source))[:3], dtype=np.float64)
            post   = fbx.FbxAMatrix()
            post.SetR(node.GetPostRotation(source))
            axis = np.array(_fbx_written_axis(node, tuple(x + 0.0 for x in tuple(post.Inverse().GetR())[:3])))
        for channel, value in (("translate", translate), ("rotate", rotate), ("joint_orient", orient)):
            if channel in signs and not value.any():
                value[...] = np.where(signs[channel], -0.0, value)

        if not joint and np.any(orient != 0.0):
            raise ValueError(
                f"load_fbx: node {name!r} channel 'joint_orient' is a pre rotation on a transform, which Maya "
                f"folds into its rotate and {FORK} does not read"
            )
        pivots = [
            tuple(getter(source))[:3]
            for getter in (node.GetRotationPivot, node.GetScalingPivot, node.GetRotationOffset, node.GetScalingOffset)
        ]
        if np.any(np.abs(np.array(pivots)) > 0.0):
            raise ValueError(f"load_fbx: node {name!r} channel 'pivots' holds pivots or offsets, which {FORK} does not read")

        kind = node.GetTransformationInheritType()
        if kind == inherit.eInheritRrs:
            held = "Rrs"
        elif kind == inherit.eInheritRrSs:
            held = "RrSs"
        else:
            held = "RSrs"

        entry = {
            "name":         name,
            "parent":       parent,
            "node_type":    "joint" if joint else "transform",
            "translate":    translate,
            "rotate":       rotate,
            "scale":        scale,
            "rotate_order": _FBX_ROTATE_ORDERS[stored if active else 0],
            "joint_orient": orient,
            "rotate_axis":  axis,
            "inherit":      held,
            "visibility":   float(node.Visibility.Get()) > _FBX_VISIBLE,
            "attrs":        _fbx_user_attributes(node),
            "index":        None,
            "id":           node.GetUniqueID(),
        }
        if joint:
            entry["radius"], entry["draw_style"] = _fbx_joint_look(node)
        kept = node.FindProperty(ORDER_PROPERTY)
        if kept.IsValid():
            entry["index"] = int(fbx.FbxPropertyInteger1(kept).Get())
        nodes.append(entry)

    # a bind pose (FbxPose) is left out: no node takes its values from one
    settings = scene.GetGlobalSettings()
    record   = (_axis_words(*_fbx_scene_axes(scene, fbx)), float(settings.GetSystemUnit().GetScaleFactor()))
    read     = {"nodes": nodes, "record": record, "take": None}

    # the active take, else the first
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
    stacks   = [typed(scene.GetSrcObject(criteria, i), fbx.FbxAnimStack, fbx) for i in range(scene.GetSrcObjectCount(criteria))]
    if not take or not stacks:
        return read
    active = str(scene.ActiveAnimStackName.Get())
    stack  = next((x for x in stacks if str(x.GetName()) == active), stacks[0])

    mode   = settings.GetTimeMode()
    fps    = _fbx_frame_rate(mode)
    if fps <= 0.0:
        fps = float(settings.GetCustomFrameRate())
    second = fbx.FbxTime()
    second.SetSecondDouble(1.0)
    per_second = int(second.Get())
    span       = stack.GetLocalTimeSpan()
    first      = int(round(span.GetStart().Get() * fps / per_second))
    last       = int(round(span.GetStop().Get() * fps / per_second))
    if last < first:
        first, last = last, first
    frames = np.arange(first, last + 1)
    ticks  = np.round(frames * (per_second / fps)).astype(np.int64)

    layers = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
    count  = stack.GetMemberCount(layers)
    if count > 1:
        raise NotImplementedError(f"load_fbx: the take {str(stack.GetName())!r} holds {count} animation layers; {FORK} reads one")
    layer  = typed(stack.GetMember(layers, 0), fbx.FbxAnimLayer, fbx) if count else None

    curves = []
    for (node, _), entry in zip(found, nodes):
        held = {}
        for channel, prop in (("translate", node.LclTranslation), ("rotate", node.LclRotation), ("scale", node.LclScaling)):
            held[channel] = [
                typed(prop.GetCurve(layer, component, False), fbx.FbxAnimCurve, fbx) if layer is not None else None
                for component in "XYZ"
            ]
        users = {}
        for name, spec in entry["attrs"].items():
            prop = node.FindProperty(name)
            if layer is None or not prop.IsValid():
                continue
            if spec.get("attributeType") in ("double3", "float3"):
                users[name] = [typed(prop.GetCurve(layer, component, False), fbx.FbxAnimCurve, fbx) for component in "XYZ"]
            elif "numberOfChildren" not in spec:
                users[name] = [typed(prop.GetCurve(layer, False), fbx.FbxAnimCurve, fbx)]
        curves.append((held, users))

    read["take"] = {
        "name":   str(stack.GetName()),
        "fps":    fps,
        "frames": frames,
        "values": _sampled(curves, ticks, per_second),
    }
    return read


def _sampled(curves, ticks, per_second) -> list:
    """each node's ({channel: [component value or (F,) values]}, {user
    attribute: [...]}), None for a component with no curve; a curve of one
    key its value, of more its values on every frame"""
    from cgmath.hierarchy.hierarchy import _fbx_sample_curves

    many, where = [], []
    out = []
    for n, (held, users) in enumerate(curves):
        done = ({}, {})
        for side, group in enumerate((held, users)):
            for channel, components in group.items():
                values = []
                for k, curve in enumerate(components):
                    keys = curve.KeyGetCount() if curve is not None else 0
                    if keys == 0:
                        values.append(None)
                    elif keys == 1:
                        values.append(float(curve.KeyGetValue(0)))
                    else:
                        values.append(None)
                        many.append(curve)
                        where.append((n, side, channel, k))
                done[side][channel] = values
        out.append(done)
    if many:
        sampled = _fbx_sample_curves(many, ticks, per_second)
        for row, (n, side, channel, k) in zip(sampled, where):
            out[n][side][channel][k] = np.array(row, dtype=np.float64)
    return out


def _build(read: dict) -> tuple:
    """(the HierarchyData of the file's nodes at their own values, the
    file's place of each of its nodes)"""
    from cgmath.hierarchy.hierarchy import HierarchyData, TransformData, generate_uuid

    entries = read["nodes"]
    uuids   = [generate_uuid() for _ in entries]
    made    = []
    for i, entry in enumerate(entries):
        joint  = entry["node_type"] == "joint"
        parent = entry["parent"]
        node = TransformData(
            entry["name"],
            node_type=entry["node_type"],
            uuid=uuids[i],
            parent_uuid=uuids[parent] if parent >= 0 else None,
            scale=entry["scale"],
            rotate=entry["rotate"],
            translate=entry["translate"],
            rotate_order=entry["rotate_order"],
            rotate_axis=entry["rotate_axis"],
            joint_orient=entry["joint_orient"],
            visibility=entry["visibility"],
            segment_scale_compensate = joint and entry["inherit"] == "Rrs",
            radius=entry.get("radius"),
            draw_style=entry.get("draw_style"),
            user_defined_attributes=entry["attrs"] or None,
        )
        made.append(node)

    # the skeleton's order: the one save_fbx kept, else the file's
    order = list(range(len(made)))
    kept  = [entry["index"] for entry in entries]
    if made and None not in kept and sorted(kept) == order:
        order = sorted(order, key=lambda i: kept[i])
    skeleton = HierarchyData([made[i] for i in order])
    skeleton._axis_system, skeleton._unit = read["record"]

    parents  = np.array([entry["parent"] for entry in entries], dtype=np.intp)
    in_order = [made[i] for i in order]
    refuse_values(
        "load_fbx, the nodes",
        in_order,
        np.array([node.shear for node in in_order]).reshape(1, -1, 3),
        np.array([node.scale for node in in_order]).reshape(1, -1, 3),
    )
    back = np.argsort(order)
    _refuse_inherit(
        entries,
        parents,
        np.asarray(skeleton.world_matrix)[back][None],
        np.array([node.scale for node in made], dtype=np.float64)[None],
        None,
    )
    return skeleton, order


def _build_clip(read: dict, skeleton, order):
    """the ClipData of the file's take over ``skeleton``, its nodes"""
    from cgmath.hierarchy.hierarchy import ClipData, _enum_fields, _user_channels

    entries = read["nodes"]
    parents = np.array([entry["parent"] for entry in entries], dtype=np.intp)
    take    = read["take"]
    frames  = take["frames"]
    count   = len(frames)
    clip    = ClipData(skeleton, frames=count, start_frame=int(frames[0]), fps=take["fps"], take=take["name"])
    clip._realign()
    for channel in ("translate", "rotate", "scale"):
        field = "_" + channel
        block = clip._blocks[field]
        keyed = np.zeros(len(clip), dtype=bool)
        for column, i in enumerate(order):
            values = take["values"][i][0][channel]
            static = entries[i][channel]
            rows   = np.empty((count, 3))
            for k in range(3):
                value      = values[k]
                rows[:, k] = static[k] if value is None else value
            keyed[column]    = any(isinstance(value, np.ndarray) for value in values) and count > 1
            block[:, column] = rows
        clip._animated[field] = keyed

    # the user channels: a key's value as the attribute holds it, an enum's
    # field place read back as its index
    for column, i in enumerate(order):
        node    = clip[column]
        attrs   = node.user_defined_attributes
        sampled = take["values"][i][1]
        for name, children in _user_channels(attrs).items():
            values = sampled.get(name)
            if values is None:
                continue
            width = 1 if children is None else 3
            names = (name,) if children is None else children
            rows  = np.empty((count, width))
            for k in range(width):
                value = values[k] if k < len(values) else None
                if value is None:
                    rows[:, k] = float(attrs[names[k]]["value"])
                    continue
                value = np.asarray(value, dtype=np.float64)
                if attrs[names[k]].get("attributeType") == "enum":
                    fields = _enum_fields(attrs[names[k]].get("enumName") or "")
                    place  = np.round(value).astype(int)
                    value  = np.array([fields[p][1] if 0 <= p < len(fields) else p for p in np.atleast_1d(place)], dtype=np.float64)
                    value  = value.reshape(np.shape(np.asarray(values[k])))
                rows[:, k] = value
            for k, child in enumerate(names):
                kind = attrs[child].get("attributeType")
                if kind == "bool":
                    rows[:, k] = (rows[:, k] != 0.0).astype(np.float64)
                elif kind in ("long", "short", "byte", "char", "enum"):
                    rows[:, k] = np.round(rows[:, k])
            if any(isinstance(value, np.ndarray) for value in values) and count > 1:
                clip._user_keys.setdefault(node.uuid, {})[name] = rows
            else:
                from cgmath.hierarchy.hierarchy import _user_put

                _user_put(attrs, name, children, rows[0])
    clip._bind(0)

    shown = clip.start_frame + np.arange(count)
    refuse_values("load_fbx, the take", clip, clip.frames.shear, clip.frames.scale, shown)
    back = np.argsort(order)
    _refuse_inherit(
        entries, parents, np.asarray(clip.frames.world_matrix)[:, back], np.asarray(clip.frames.scale)[:, back], shown
    )
    return clip


def _scales(worlds: np.ndarray) -> np.ndarray:
    """(F, N, 3) the scale of each world matrix: its rows' lengths"""
    return np.linalg.norm(np.asarray(worlds)[..., :3, :3], axis=-1)


def _refuse_inherit(entries, parents, worlds, scales, frames) -> None:
    """ValueError for an inherit type a joint or transform can not hold
    where the parent's scale is not 1: eInheritRrSs where the parent's world
    scale is not, eInheritRrs on a transform where the parent's own is not;
    ``worlds`` (F, N, 4, 4) and ``scales`` (F, N, 3) in the file's order"""
    for i, entry in enumerate(entries):
        parent = int(parents[i])
        if parent < 0 or entry["inherit"] == "RSrs" or (entry["inherit"] == "Rrs" and entry["node_type"] == "joint"):
            continue
        if entry["inherit"] == "RrSs":
            scale = _scales(worlds[:, parent])
            what  = "eInheritRrSs under a parent whose world scale"
        else:
            scale = np.asarray(scales[:, parent], dtype=np.float64)
            what  = "eInheritRrs on a transform under a parent whose scale"
        off = np.flatnonzero(np.any(np.abs(scale - 1.0) > UNIT_SCALE, axis=-1))
        if len(off):
            when = "" if frames is None else f" frame {int(frames[off[0]])}"
            raise ValueError(
                f"load_fbx: node {entry['name']!r} channel 'inherit type'{when} is {what} is not 1, "
                f"{np.round(scale[off[0]], 9).tolist()}, which a joint or transform can not hold"
            )
