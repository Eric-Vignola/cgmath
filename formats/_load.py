"""``cgmath.formats.load_fbx``: a hierarchy or a take, the meshes on its nodes

The r17-skeleton fork reads every node of a file as a joint (a skeleton
node) or a transform (any other: a group, or a node holding a shape, a
mesh, a camera, a light or a locator), at their own values, a bind pose left
out, or the active take animating them; each mesh as a MeshData on its node,
the mesh's UV sets and its skin naming it by uuid.
"""

import os
from collections import namedtuple

import numpy as np

from cgmath.formats._write import FORK, ORDER_PROPERTY, refuse_values

# a parent scale this close to 1 leaves an inherit type nothing to change
UNIT_SCALE = 1e-6

# a node whose own values place it this close to its bind matrix keeps them
BIND_AGREE = 1e-8

# a mesh's geometric transform this close to identity is none
MESH_IDENTITY = 1e-12

# a bind matrix the solved channels place a node this far from: refused
BIND_SOLVED = 1e-6

# a skin's matrices and its bind pose this far apart, relative to the
# largest element, disagree; Maya's FBXExport writes the cluster's in
# float32
SKIN_AGREE  = 1e-5

FbxContents = namedtuple("FbxContents", ("hierarchy", "clip", "meshes", "skins", "uvs"))


def load_fbx(filename) -> FbxContents:
    """``(hierarchy, clip, meshes, skins, uvs)``, None where the file holds none

    ``hierarchy`` every node of the file at its own values, a mesh's node
    among them; None for a file holding a take, whose nodes ``clip`` holds at
    the take's frames. With a skin, ``hierarchy`` is the bind pose and
    ``clip`` the take, if any. ``meshes`` a MeshList of the file's meshes in
    their nodes' order, each ``parent_uuid`` the uuid of its node in
    ``hierarchy`` or ``clip``; ``skins`` a SkinList, one a skinned mesh, in
    that order; ``uvs`` a UVList of every mesh's UV sets, mesh by mesh in the
    file's order; each skin and set names its mesh by ``mesh_uuid``. A mesh
    is named as its FbxMesh is, None for one unnamed (Maya's FBXExport
    leaves every FbxMesh unnamed). A set of no UVs, as Maya's FBXExport
    writes an empty one, is no set; ValueError for a set leaving faces
    without UVs. A node holding a shape that is no mesh is a transform, the
    shape left out; a joint is a joint whatever else it holds (Maya's FBX
    holds no mesh on one).
    """
    read = _open(filename, True, True)
    hierarchy, order = _build(read)
    # Maya's FBXExport writes a take whether anything moves or not: a take
    # over no node is no clip
    clip = None if read["take"] is None or not len(hierarchy) else _build_clip(read, hierarchy, order)
    # the meshes in their nodes' order, the hierarchy's
    rank = {at: column for column, at in enumerate(order)}
    read["meshes"].sort(key=lambda held: rank[held["node"]])
    meshes = _build_meshes(read)
    uvs    = _build_uvs(read, meshes)
    if any(held["skin"] is not None for held in read["meshes"]):
        bind, order = _build(read)
        _settle_bind(read, bind, order)
        return FbxContents(bind, clip, meshes, _build_skins(read, meshes), uvs)
    if clip is not None:
        return FbxContents(None, clip, meshes, None, uvs)
    return FbxContents(hierarchy if len(hierarchy) else None, None, meshes, None, uvs)


def load_nodes(filename):
    """the HierarchyData of an fbx file's nodes at their own values, whether
    it holds a take or not (the take left unread), a mesh's node among them,
    its shape left out"""
    return _build(_open(filename, False, False))[0]


def load_take(filename):
    """the ClipData of an fbx file's active take over every node, a mesh's
    among them, its shape and skin left out; None for a file holding no take"""
    read = _open(filename, True, False)
    if read["take"] is None:
        return None
    hierarchy, order = _build(read)
    return _build_clip(read, hierarchy, order)


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


def _amatrix(matrix) -> np.ndarray:
    """an FbxAMatrix or FbxMatrix as its (4, 4) row-vector layout"""
    return np.array([[matrix.Get(row, column) for column in range(4)] for row in range(4)], dtype=np.float64)


def _read_mesh(node, at: int, fbx) -> dict:
    """the mesh on the node at place ``at`` of the file's nodes: its name,
    points, faces, normals, one a face corner, and UV sets; ValueError for a
    node holding a geometric transform, which no field holds, or a mesh
    holding holes"""
    from cgmath.formats._fbx_io import typed
    from cgmath.geometry.mesh import (
        _fbx_control_points,
        _fbx_direct_array_to_ndarray,
        _fbx_face_varying_indices,
        _fbx_polygon_streams,
    )

    shape  = typed(node.GetNodeAttribute(), fbx.FbxMesh, fbx)
    name   = str(shape.GetName()) or None
    label  = f"mesh {name!r}" if name else f"the mesh on node {str(node.GetName())!r}"
    source = fbx.FbxNode.EPivotSet.eSourcePivot
    offset = fbx.FbxAMatrix(node.GetGeometricTranslation(source), node.GetGeometricRotation(source), node.GetGeometricScaling(source))
    offset = _amatrix(offset)
    if np.abs(offset - np.eye(4)).max() > MESH_IDENTITY:
        raise ValueError(
            f"load_fbx: {label} channel 'geometric transform' is {np.round(offset, 12).tolist()} on node "
            f"{str(node.GetName())!r}, which no field holds; {FORK} reads a mesh at its node"
        )
    holes = [i for i in range(shape.GetPolygonCount()) if shape.GetPolyHoleInfo(i)]
    if holes:
        raise ValueError(f"load_fbx: {label} holds {len(holes)} polygon holes, which {FORK} does not read")
    indices, counts = _fbx_polygon_streams(shape)
    normals, ids = None, None
    if shape.GetElementNormalCount():
        element = shape.GetElementNormal(0)
        normals = _fbx_direct_array_to_ndarray(element.GetDirectArray(), dim=3)
        ids     = _fbx_face_varying_indices(shape, element, counts)
    return {
        "uvs":            _read_uvs(label, shape, counts, fbx),
        "name":           name,
        "label":          label,
        "node":           at,
        "points":         _fbx_control_points(shape),
        "counts":         counts,
        "indices":        indices,
        "normals":        normals,
        "normal_indices": ids,
        "shape":          shape,
    }


def _read_uvs(label, shape, counts, fbx) -> list:
    """the mesh's UV sets, (name, points, indices) each, in the file's
    order, one index a face corner; a set of no UVs left out, ValueError for
    one leaving faces without UVs"""
    from cgmath.geometry.mesh import _fbx_direct_array_to_ndarray, _fbx_face_varying_indices

    mapping = fbx.FbxLayerElement.EMappingMode
    refs    = fbx.FbxLayerElement.EReferenceMode
    total   = int(np.sum(counts))
    found   = []
    for place in range(shape.GetElementUVCount()):
        element = shape.GetElementUV(place)
        if element is None:
            continue
        set_name = str(element.GetName()) or f"map{place + 1}"
        points   = _fbx_direct_array_to_ndarray(element.GetDirectArray(), dim=2)
        if not len(points):
            continue
        held = element.GetIndexArray().GetCount()
        if (
            element.GetMappingMode() == mapping.eByPolygonVertex
            and element.GetReferenceMode() == refs.eIndexToDirect
            and held != total
        ):
            raise ValueError(
                f"load_fbx: {label} UV set {set_name!r} maps {held} face corners of {total}; {FORK} reads a set "
                f"mapping every face"
            )
        indices = _fbx_face_varying_indices(shape, element, counts)
        if indices.size and (indices.min() < 0 or indices.max() >= len(points)):
            unmapped = int(np.count_nonzero(indices < 0))
            why      = f"leaves {unmapped} face corners without UVs" if unmapped else "names UVs it does not hold"
            raise ValueError(f"load_fbx: {label} UV set {set_name!r} {why}; {FORK} reads a set mapping every face")
        found.append((set_name, points, indices))
    return found


def _read_skin(scene, node, at: int, label: str, shape, found, fbx) -> dict:
    """the one skin on the mesh (``label``) of the node at place ``at``: its
    clusters' nodes (places in ``found``), weights and matrices, and the
    bind pose listing the mesh's node; None for no skin, ValueError for
    several skins or several bind poses"""
    from cgmath.formats._fbx_io import typed

    kind  = fbx.FbxDeformer.EDeformerType.eSkin
    count = shape.GetDeformerCount(kind)
    if count == 0:
        return None
    if count > 1:
        raise ValueError(f"load_fbx: {label} holds {count} skin deformers; {FORK} reads one")
    place    = {entry.GetUniqueID(): i for i, (entry, _) in enumerate(found)}
    deformer = typed(shape.GetDeformer(0, kind), fbx.FbxSkin, fbx)
    clusters = []
    for k in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(k)
        link    = cluster.GetLink()
        if link is None:
            continue
        linked_at = place.get(link.GetUniqueID())
        if linked_at is None:
            raise ValueError(
                f"load_fbx: skin {str(deformer.GetName())!r} cluster {k} links {str(link.GetName())!r}, no node of the "
                f"hierarchy"
            )
        linked, placed = fbx.FbxAMatrix(), fbx.FbxAMatrix()
        cluster.GetTransformLinkMatrix(linked)
        cluster.GetTransformMatrix(placed)
        clusters.append(
            {
                "node":    linked_at,
                "indices": np.asarray(cluster.GetControlPointIndices(), dtype=np.int64),
                "weights": np.asarray(cluster.GetControlPointWeights(), dtype=np.float64),
                "link":    _amatrix(linked),
                "mesh":    _amatrix(placed),
            }
        )

    poses = [scene.GetPose(i) for i in range(scene.GetPoseCount())]
    poses = [pose for pose in poses if pose.IsBindPose() and pose.Find(node) >= 0]
    if len(poses) > 1:
        raise ValueError(f"load_fbx: {label} is listed by {len(poses)} bind poses; {FORK} reads one")
    binds = {}
    for pose in poses:
        for k in range(pose.GetCount()):
            target = pose.GetNode(k)
            if target is not None and target.GetUniqueID() in place:
                binds[place[target.GetUniqueID()]] = _amatrix(pose.GetMatrix(k))
    return {
        "name":     str(deformer.GetName()),
        "mesh":     label,
        "node":     at,
        "count":    shape.GetControlPointsCount(),
        "clusters": clusters,
        "binds":    binds,
    }


def _read(scene, fbx, take: bool = True, meshes: bool = False) -> dict:
    """everything the build needs, out of the scene as python and numpy: every
    node, one uuid each, the active take's curves unless not ``take``, and
    with ``meshes`` each mesh and its skin, else the meshes left unread"""
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
        generate_uuid,
    )

    source  = fbx.FbxNode.EPivotSet.eSourcePivot
    inherit = fbx.FbxTransform.EInheritType
    found   = _walk(scene)
    shapes  = []
    if meshes:
        kind = fbx.FbxNodeAttribute.EType.eMesh
        for at, (node, _) in enumerate(found):
            attribute = node.GetNodeAttribute()
            if attribute is None or attribute.GetAttributeType() != kind:
                continue
            mesh         = _read_mesh(node, at, fbx)
            mesh["skin"] = _read_skin(scene, node, at, mesh["label"], mesh.pop("shape"), found, fbx)
            shapes.append(mesh)
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
            "uuid":         generate_uuid(),
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
    read     = {"nodes": nodes, "record": record, "take": None, "meshes": shapes}

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
    """(the HierarchyData of the file's nodes at their own values, each the
    uuid the read gave it, the file's place of each of its nodes)"""
    from cgmath.hierarchy.hierarchy import HierarchyData, TransformData

    entries = read["nodes"]
    made    = []
    for entry in entries:
        joint  = entry["node_type"] == "joint"
        parent = entry["parent"]
        node = TransformData(
            entry["name"],
            node_type=entry["node_type"],
            uuid=entry["uuid"],
            parent_uuid=entries[parent]["uuid"] if parent >= 0 else None,
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

    # the hierarchy's order: the one save_fbx kept, else the file's
    order = list(range(len(made)))
    kept  = [entry["index"] for entry in entries]
    if made and None not in kept and sorted(kept) == order:
        order = sorted(order, key=lambda i: kept[i])
    hierarchy = HierarchyData([made[i] for i in order])
    hierarchy._axis_system, hierarchy._unit = read["record"]

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
        np.asarray(hierarchy.world_matrix)[back][None],
        np.array([node.scale for node in made], dtype=np.float64)[None],
        None,
    )
    return hierarchy, order


def _build_clip(read: dict, hierarchy, order):
    """the ClipData of the file's take over ``hierarchy``, its nodes"""
    from cgmath.hierarchy.hierarchy import ClipData, _enum_fields, _user_channels

    entries = read["nodes"]
    parents = np.array([entry["parent"] for entry in entries], dtype=np.intp)
    take    = read["take"]
    frames  = take["frames"]
    count   = len(frames)
    clip    = ClipData(hierarchy, frames=count, start_frame=int(frames[0]), fps=take["fps"], take=take["name"])
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


def _build_meshes(read: dict):
    """the MeshList of the file's meshes, as ``read`` orders them, each on
    its node by uuid; None for none"""
    from cgmath.geometry.mesh import MeshData, MeshList

    entries = read["nodes"]
    if not read["meshes"]:
        return None
    return MeshList(
        [
            MeshData(
                name           = held["name"],
                points         = held["points"],
                counts         = held["counts"],
                indices        = held["indices"],
                normals        = held["normals"],
                normal_indices = held["normal_indices"],
                parent_uuid    = entries[held["node"]]["uuid"],
            )
            for held in read["meshes"]
        ]
    )


def _build_uvs(read: dict, meshes):
    """the UVList of every mesh's UV sets, in the file's order, each naming
    its mesh by uuid; None for no set"""
    from cgmath.geometry.mesh import UVData, UVList

    if meshes is None:
        return None
    sets = []
    for held, mesh in zip(read["meshes"], meshes):
        for set_name, points, indices in held["uvs"]:
            sets.append(
                UVData(
                    name      = set_name,
                    points    = points,
                    counts    = np.asarray(held["counts"], dtype=np.int32).copy(),
                    indices   = indices,
                    mesh_uuid = mesh.uuid,
                )
            )
    return UVList(sets) if sets else None


def _build_skins(read: dict, meshes):
    """the SkinList of the skinned meshes' skins in the file's order, each
    naming its mesh by uuid; None for none"""
    from cgmath.geometry.skin_weights import SkinList

    skins = [
        _build_skin(held["skin"], read["nodes"], mesh)
        for held, mesh in zip(read["meshes"], meshes)
        if held["skin"] is not None
    ]
    return SkinList(skins) if skins else None


def _build_skin(skin: dict, entries: list, mesh):
    """the SkinData of a mesh's skin, naming the mesh by its uuid: its
    points by its influences, in cluster order, a node reached by two
    clusters one column"""
    from cgmath.geometry.skin_weights import SkinData

    places = []
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
    return SkinData(weights=weights, influences=influences, name=skin["name"] or None, mesh_uuid=mesh.uuid)


def _near(a, b, within: float) -> bool:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= within


def _agree(held: dict, at: int, matrix, where: str, entries: list) -> None:
    """``held[at]`` set to (``matrix``, ``where``); ValueError for one held
    already, from another skin or pose, that disagrees"""
    other = held.get(at)
    if other is None:
        held[at] = (matrix, where)
        return
    if not _near(other[0], matrix, SKIN_AGREE * max(1.0, float(np.abs(other[0]).max()))):
        raise ValueError(
            f"load_fbx: node {entries[at]['name']!r} channel 'bind pose' is {np.round(other[0], 9).tolist()} in "
            f"{other[1]}, {np.round(matrix, 9).tolist()} in {where}"
        )


def _settle_bind(read: dict, bind, order) -> None:
    """``bind``, the file's nodes at their own values, moved to the bind
    pose: a node the bind pose (or a cluster) places where its own values
    do keeps them, any other its channels solved from the bind matrix,
    parents first. A skinned mesh's node is placed by its skin's cluster
    transform, the mesh's world at the bind (Maya's geomMatrix), one on
    every cluster, which the bind pose agrees with. ValueError for skins
    whose clusters and bind poses disagree, or a bind matrix no channels
    hold"""
    entries = read["nodes"]
    links   = {}
    binds   = {}
    for held in read["meshes"]:
        skin = held["skin"]
        if skin is None:
            continue
        label = f"skin {skin['name']!r}"
        # the pose first: its matrices are written element by element, a
        # cluster's through the sdk's affine matrix, within 1e-15 of them
        for at, matrix in skin["binds"].items():
            _agree(binds, at, matrix, f"the bind pose of {skin['mesh']}", entries)
        for cluster in skin["clusters"]:
            _agree(links, cluster["node"], cluster["link"], f"{label}'s cluster", entries)
            _agree(binds, skin["node"], cluster["mesh"], f"{label}'s cluster transform", entries)
    for at, (matrix, where) in links.items():
        held = binds.get(at)
        if held is not None and not _near(held[0], matrix, SKIN_AGREE * max(1.0, float(np.abs(held[0]).max()))):
            raise ValueError(
                f"load_fbx: node {entries[at]['name']!r} channel 'bind pose' is {np.round(held[0], 9).tolist()} in "
                f"{held[1]}, {np.round(matrix, 9).tolist()} in {where}"
            )
    matrices = {at: matrix for at, (matrix, _) in links.items()}
    matrices.update({at: matrix for at, (matrix, _) in binds.items()})
    column = {at: i for i, at in enumerate(order)}
    # the file's order is depth first, each parent ahead of its children
    for at in range(len(entries)):
        matrix = matrices.get(at)
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
