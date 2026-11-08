"""``cgmath.formats.load_fbx``: one skeleton, or one take

The r17-skeleton fork reads every node of a file as a joint (a skeleton
node) or a transform (any other: a group, or a transform holding a shape, a
mesh, a camera, a light or a locator, the shape left out), at their own
values, a bind pose left out, or the active take animating them.
"""

import os
from collections import namedtuple

import numpy as np

from cgmath.formats._write import FORK, ORDER_PROPERTY, refuse_values

# a parent scale this close to 1 leaves an inherit type nothing to change
UNIT_SCALE = 1e-6

# a node whose own values place it this close to its bind matrix keeps them
BIND_AGREE = 1e-8

# a mesh node whose world is this close to identity is at identity
MESH_IDENTITY = 1e-12

# a bind matrix the solved channels place a node this far from: refused
BIND_SOLVED = 1e-6

# a skin's matrices and its bind pose this far apart, relative to the
# largest element, disagree; Maya's FBXExport writes the cluster's in
# float32
SKIN_AGREE  = 1e-5

FbxContents = namedtuple("FbxContents", ("skeleton", "clip", "mesh", "skin"))


def load_fbx(filename) -> FbxContents:
    """``(skeleton, clip, mesh, skin)``, None where the file holds none

    Without a skin: ``(None, clip, mesh, None)`` for a file holding a take,
    ``(hierarchy, None, mesh, None)`` for one holding none. With a skin: the
    skeleton is its bind pose, the clip the take if any.
    """
    read = _open(filename, True, True)
    skeleton, order = _build(read)
    clip = None if read["take"] is None else _build_clip(read, skeleton, order)
    mesh = _build_mesh(read["mesh"])
    if read["skin"] is not None:
        bind, order = _build(read)
        _settle_bind(read, bind, order)
        return FbxContents(bind, clip, mesh, _build_skin(read))
    if clip is not None:
        return FbxContents(None, clip, mesh, None)
    return FbxContents(skeleton if len(skeleton) else None, None, mesh, None)


def load_nodes(filename):
    """the HierarchyData of an fbx file's nodes at their own values, whether
    it holds a take or not (the take left unread), a mesh node a transform"""
    return _build(_open(filename, False))[0]


def _open(filename, take: bool, meshes: bool = False) -> dict:
    # imported here: the classes' modules import this one
    from cgmath.formats._fbx_io import open_fbx
    from cgmath.hierarchy.hierarchy import FBX

    if FBX is None:
        raise ImportError("fbx module not found")
    filename = os.path.expanduser(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with open_fbx(filename, FBX) as scene:
        return _read(scene, FBX, take, meshes)


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


def _mesh_apart(found, fbx) -> tuple:
    """(``found`` less its one mesh node, that node or None);
    NotImplementedError for several, ValueError for one holding children"""
    kind = fbx.FbxNodeAttribute.EType.eMesh
    shapes = [
        i for i, (node, _) in enumerate(found)
        if node.GetNodeAttribute() is not None and node.GetNodeAttribute().GetAttributeType() == kind
    ]
    if not shapes:
        return found, None
    if len(shapes) > 1:
        names = [str(found[i][0].GetName()) for i in shapes]
        raise NotImplementedError(f"load_fbx: the file holds {len(shapes)} meshes {names}; {FORK} reads one")
    node = found[shapes[0]][0]
    if node.GetChildCount():
        raise ValueError(f"load_fbx: mesh {str(node.GetName())!r} holds children, which {FORK} does not read")
    kept  = [i for i in range(len(found)) if i != shapes[0]]
    place = {old: new for new, old in enumerate(kept)}
    return [(found[i][0], place.get(found[i][1], -1)) for i in kept], node


def _amatrix(matrix) -> np.ndarray:
    """an FbxAMatrix or FbxMatrix as its (4, 4) row-vector layout"""
    return np.array([[matrix.Get(row, column) for column in range(4)] for row in range(4)], dtype=np.float64)


def _read_mesh(node, fbx) -> dict:
    """the mesh node's points, faces and normals, one a face corner;
    ValueError for one off identity, its geometric transform included, or
    holding holes"""
    from cgmath.formats._fbx_io import typed
    from cgmath.geometry.mesh import (
        _fbx_control_points,
        _fbx_direct_array_to_ndarray,
        _fbx_face_varying_indices,
        _fbx_polygon_streams,
    )

    name   = str(node.GetName())
    source = fbx.FbxNode.EPivotSet.eSourcePivot
    offset = fbx.FbxAMatrix(node.GetGeometricTranslation(source), node.GetGeometricRotation(source), node.GetGeometricScaling(source))
    world  = _amatrix(node.EvaluateGlobalTransform() * offset)
    if np.abs(world - np.eye(4)).max() > MESH_IDENTITY:
        raise ValueError(
            f"load_fbx: mesh {name!r} channel 'matrix' is {np.round(world, 12).tolist()} (its world, the geometric "
            f"transform included): {FORK} reads a mesh at identity"
        )
    shape = typed(node.GetNodeAttribute(), fbx.FbxMesh, fbx)
    holes = [i for i in range(shape.GetPolygonCount()) if shape.GetPolyHoleInfo(i)]
    if holes:
        raise ValueError(f"load_fbx: mesh {name!r} holds {len(holes)} polygon holes, which {FORK} does not read")
    indices, counts = _fbx_polygon_streams(shape)
    normals, ids = None, None
    if shape.GetElementNormalCount():
        element = shape.GetElementNormal(0)
        normals = _fbx_direct_array_to_ndarray(element.GetDirectArray(), dim=3)
        ids     = _fbx_face_varying_indices(shape, element, counts)
    return {
        "name":           name,
        "points":         _fbx_control_points(shape),
        "counts":         counts,
        "indices":        indices,
        "normals":        normals,
        "normal_indices": ids,
        "shape":          shape,
    }


def _read_skin(scene, node, shape, found, fbx) -> dict:
    """the one skin on the mesh: its clusters' nodes (places in ``found``),
    weights and matrices, and the bind pose listing the mesh; None for no
    skin, ValueError for several skins or several bind poses"""
    from cgmath.formats._fbx_io import typed

    name  = str(node.GetName())
    kind  = fbx.FbxDeformer.EDeformerType.eSkin
    count = shape.GetDeformerCount(kind)
    if count == 0:
        return None
    if count > 1:
        raise ValueError(f"load_fbx: mesh {name!r} holds {count} skin deformers; {FORK} reads one")
    place    = {entry.GetUniqueID(): i for i, (entry, _) in enumerate(found)}
    deformer = typed(shape.GetDeformer(0, kind), fbx.FbxSkin, fbx)
    clusters = []
    for k in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(k)
        link    = cluster.GetLink()
        if link is None:
            continue
        at = place.get(link.GetUniqueID())
        if at is None:
            raise ValueError(
                f"load_fbx: skin {str(deformer.GetName())!r} cluster {k} links {str(link.GetName())!r}, no node of the "
                f"skeleton"
            )
        linked, placed = fbx.FbxAMatrix(), fbx.FbxAMatrix()
        cluster.GetTransformLinkMatrix(linked)
        cluster.GetTransformMatrix(placed)
        clusters.append(
            {
                "node":    at,
                "indices": np.asarray(cluster.GetControlPointIndices(), dtype=np.int64),
                "weights": np.asarray(cluster.GetControlPointWeights(), dtype=np.float64),
                "link":    _amatrix(linked),
                "mesh":    _amatrix(placed),
            }
        )

    poses = [scene.GetPose(i) for i in range(scene.GetPoseCount())]
    poses = [pose for pose in poses if pose.IsBindPose() and pose.Find(node) >= 0]
    if len(poses) > 1:
        raise ValueError(f"load_fbx: mesh {name!r} is listed by {len(poses)} bind poses; {FORK} reads one")
    binds, mesh_bind = {}, None
    for pose in poses:
        for k in range(pose.GetCount()):
            target = pose.GetNode(k)
            if target is None:
                continue
            matrix = _amatrix(pose.GetMatrix(k))
            if target.GetUniqueID() == node.GetUniqueID():
                mesh_bind = matrix
            elif target.GetUniqueID() in place:
                binds[place[target.GetUniqueID()]] = matrix
    return {
        "name":      str(deformer.GetName()),
        "mesh":      name,
        "count":     shape.GetControlPointsCount(),
        "clusters":  clusters,
        "binds":     binds,
        "mesh_bind": mesh_bind,
    }


def _read(scene, fbx, take: bool = True, meshes: bool = False) -> dict:
    """everything the build needs, out of the scene as python and numpy; the
    active take's curves unless not ``take``; the one mesh and its skin set
    apart from the nodes with ``meshes``, else a mesh node is a transform"""
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
    shape   = None
    mesh    = None
    skin    = None
    if meshes:
        found, shape = _mesh_apart(found, fbx)
    if shape is not None:
        mesh = _read_mesh(shape, fbx)
        skin = _read_skin(scene, shape, mesh.pop("shape"), found, fbx)
    nodes = []
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
    read     = {"nodes": nodes, "record": record, "take": None, "mesh": mesh, "skin": skin}

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
    layer = typed(stack.GetMember(layers, 0), fbx.FbxAnimLayer, fbx) if count else None

    # the mesh stays at identity: a key on it elsewhere is refused
    if shape is not None and layer is not None:
        for channel, prop, still in (
            ("translate", shape.LclTranslation, 0.0),
            ("rotate", shape.LclRotation, 0.0),
            ("scale", shape.LclScaling, 1.0),
        ):
            for component in "XYZ":
                curve = typed(prop.GetCurve(layer, component, False), fbx.FbxAnimCurve, fbx)
                keys  = [curve.KeyGetValue(k) for k in range(curve.KeyGetCount())] if curve is not None else []
                if any(abs(value - still) > MESH_IDENTITY for value in keys):
                    raise ValueError(
                        f"load_fbx: mesh {mesh['name']!r} channel {channel!r} is keyed off identity in the take; "
                        f"{FORK} reads a mesh at identity"
                    )

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


def _build_mesh(read):
    """the MeshData of the file's one mesh, or None"""
    from cgmath.geometry.mesh import MeshData

    if read is None:
        return None
    return MeshData(
        name           = read["name"],
        points         = read["points"],
        counts         = read["counts"],
        indices        = read["indices"],
        normals        = read["normals"],
        normal_indices = read["normal_indices"],
    )


def _build_skin(read):
    """the SkinData of the mesh's skin: its points by its influences, in
    cluster order, a node reached by two clusters one column"""
    from cgmath.geometry.skin_weights import SkinData

    skin    = read["skin"]
    entries = read["nodes"]
    places  = []
    for cluster in skin["clusters"]:
        if cluster["node"] not in places:
            places.append(cluster["node"])
    influences = [entries[at]["name"] for at in places]
    shared     = sorted({x for x in influences if influences.count(x) > 1})
    if shared:
        raise ValueError(f"load_fbx: skin {skin['name']!r} links several nodes named {shared}")
    weights = np.zeros((skin["count"], len(places)), dtype=np.float64)
    for cluster in skin["clusters"]:
        np.add.at(weights, (cluster["indices"], places.index(cluster["node"])), cluster["weights"])
    return SkinData(weights=weights, influences=influences, name=skin["name"] or None)


def _near(a, b, within: float) -> bool:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= within


def _settle_bind(read, bind, order) -> None:
    """``bind``, the file's nodes at their own values, moved to the bind
    pose: a node the bind pose (or its cluster) places where its own values
    do keeps them, any other its channels solved from the bind matrix,
    parents first. ValueError for a skin whose clusters and bind pose
    disagree, a mesh bound off identity, or a bind matrix no channels hold"""
    skin    = read["skin"]
    entries = read["nodes"]
    label   = f"skin {skin['name']!r}"
    eye     = np.eye(4)
    links   = {}
    for cluster in skin["clusters"]:
        links.setdefault(cluster["node"], cluster["link"])
        if not _near(cluster["mesh"], eye, SKIN_AGREE):
            raise ValueError(
                f"load_fbx: {label} binds mesh {skin['mesh']!r} at {np.round(cluster['mesh'], 9).tolist()}: {FORK} "
                f"reads a mesh bound at identity"
            )
    if skin["mesh_bind"] is not None and not _near(skin["mesh_bind"], eye, BIND_AGREE):
        raise ValueError(
            f"load_fbx: the bind pose places mesh {skin['mesh']!r} at {np.round(skin['mesh_bind'], 9).tolist()}: "
            f"{FORK} reads a mesh bound at identity"
        )
    for at, matrix in links.items():
        held = skin["binds"].get(at)
        if held is not None and not _near(held, matrix, SKIN_AGREE * max(1.0, float(np.abs(held).max()))):
            raise ValueError(
                f"load_fbx: node {entries[at]['name']!r} channel 'bind pose' is {np.round(held, 9).tolist()} in the "
                f"bind pose, {np.round(matrix, 9).tolist()} in {label}'s cluster"
            )
    binds = dict(links)
    binds.update(skin["binds"])
    column = {at: i for i, at in enumerate(order)}
    # the file's order is depth first, each parent ahead of its children
    for at in range(len(entries)):
        matrix = binds.get(at)
        if matrix is None:
            continue
        node = bind[column[at]]
        if _near(node.world_matrix, matrix, BIND_AGREE):
            continue
        node.world_matrix = matrix
        # a transform takes the shear a matrix holds: a float32 one's
        # rounding is none
        if np.any(node.shear != 0.0) and np.abs(node.shear).max() <= BIND_SOLVED:
            node.shear = np.zeros(3)
        if not _near(node.world_matrix, matrix, BIND_SOLVED * max(1.0, float(np.abs(matrix).max()))):
            raise ValueError(
                f"load_fbx: node {node.name!r} channel 'bind pose' is {np.round(matrix, 9).tolist()}, which its "
                f"channels can not hold (a shear)"
            )
    refuse_values(
        "load_fbx, the bind pose",
        bind,
        np.array([node.shear for node in bind]).reshape(1, -1, 3),
        np.array([node.scale for node in bind]).reshape(1, -1, 3),
    )
