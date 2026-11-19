"""``FbxFile.save`` of a changed layout: the nodes and records added to,
taken off or moved in the open scene, before their values are written

A record's values are written by :mod:`cgmath.formats._fbx_edit`, which
compares the file as it reads with the records to save. This module first
gives the file the records' layout: a node the records add is made bare, one
they leave out is taken off with what belongs to it, one under another
parent is moved. The file is then read again, and the values written into
it as into any file of that layout.
"""

import json
import urllib.parse

import numpy as np

from cgmath.formats import _load

_ELSE = "write the records with save_fbx"

_NODE_FIELDS  = ("hierarchy", "clip")
_SHAPE_FIELDS = ("meshes", "cameras", "lights", "curves", "surfaces")


def _held(value):
    """a field's value, None for one the reader refuses"""
    return None if type(value).__name__ == "_Refused" else value


# ---------------------------------------------------------------- the nodes


def _node_lists(held: dict) -> list:
    """[(what a message calls it, its nodes)] of the records holding nodes:
    the hierarchy, then each take"""
    found     = []
    hierarchy = _held(held["hierarchy"])
    clip      = _held(held["clip"])
    if hierarchy is not None:
        found.append(("the hierarchy", hierarchy.list))
    if clip is not None:
        found.extend((f"take {take.take!r}", take.list) for take in clip.list)
    return found


def _links(nodes) -> dict:
    """{uuid: (its parent's uuid, its kind)} of a list of nodes"""
    return {node.uuid: (node.parent_uuid, node.node_type) for node in nodes}


def _wanted(state: dict):
    """(the nodes the records to save hold, their links), off the first
    record holding nodes; ValueError where another holds other nodes, a node
    under another parent or of another kind: a file holds one set of nodes"""
    lists = _node_lists(state)
    if not lists:
        return None, {}
    first, nodes = lists[0]
    links = _links(nodes)
    names = {node.uuid: node.name for node in nodes}
    for label, other in lists[1:]:
        held = _links(other)
        if held == links:
            continue
        names.update({node.uuid: node.name for node in other})
        odd  = next(uuid for uuid in list(links) + list(held) if links.get(uuid) != held.get(uuid))
        what = "holds" if odd in links and odd in held else "lacks" if odd in links else "adds"
        raise ValueError(
            f"FbxFile.save: {label} {what} node {names[odd]!r} "
            f"{'under another parent, or as another kind, than' if what == 'holds' else 'of'} {first}; a file holds "
            f"one set of nodes: give every record holding nodes the same ones, a node added to several under one uuid "
            f"(TransformData(name, uuid=...))"
        )
    return nodes, links


def check_nodes(state: dict) -> None:
    """ValueError where the records to save do not hold one set of nodes"""
    _wanted(state)


def nodes_differ(base: dict, state: dict) -> bool:
    """whether the records to save hold other nodes, or a node under another
    parent, than the file: a change of layout this module writes. A file or
    a save holding no node at all is left to the caller's refusal"""
    was = _node_lists(base)
    now = _node_lists(state)
    if not was or not now:
        return False
    held = _links(was[0][1])
    return any(_links(nodes) != held for _, nodes in now)


def _bare(edit, record):
    """a node of ``record``'s name and kind, holding nothing: its values are
    written once the file is read with it"""
    fbx  = edit.fbx
    name = str(record.name)
    node = fbx.FbxNode.Create(edit.scene, name)
    zero = fbx.FbxVector4(0.0, 0.0, 0.0, 1.0)
    node.SetRotationActive(True)
    node.SetRotationOrder(edit.source, fbx.EFbxRotationOrder.eEulerXYZ)
    node.SetPreRotation(edit.source, zero)
    node.SetPostRotation(edit.source, zero)
    node.SetTransformationInheritType(fbx.FbxTransform.EInheritType.eInheritRSrs)
    if record.node_type == "joint":
        shape = fbx.FbxSkeleton.Create(edit.scene, "")
        shape.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
    elif record.node_type in ("transform", "locator"):
        shape = fbx.FbxNull.Create(edit.scene, "")
        look  = fbx.FbxNull.ELook.eCross if record.node_type == "locator" else fbx.FbxNull.ELook.eNone
        shape.Look.Set(look.value)
    else:
        raise ValueError(f"FbxFile.save: node {name!r} is a {record.node_type}, a kind of node an fbx file is not given")
    node.SetNodeAttribute(shape)
    return node


def _curve_nodes(edit, owner) -> list:
    """the curve nodes animating ``owner``'s properties, in every take"""
    fbx      = edit.fbx
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
    found    = []
    prop     = owner.GetFirstProperty()
    while prop.IsValid():
        for at in range(prop.GetSrcObjectCount(criteria)):
            found.append(edit.typed(prop.GetSrcObject(criteria, at), fbx.FbxAnimCurveNode, fbx))
        prop = owner.GetNextProperty(prop)
    return found


def strip_curves(edit, owner) -> None:
    """every key of ``owner`` taken off the file: its curves and the curve
    nodes holding them"""
    fbx = edit.fbx
    for curve_node in _curve_nodes(edit, owner):
        for channel in range(curve_node.GetChannelsCount()):
            for at in reversed(range(curve_node.GetCurveCount(channel))):
                curve = edit.typed(curve_node.GetCurve(channel, at), fbx.FbxAnimCurve, fbx)
                if curve is not None:
                    curve.Destroy()
        curve_node.Destroy()


# the classes a node is tied to as a matter of course, which go or stay with
# it by themselves
_OWN_TIES = ("FbxNode", "FbxScene", "FbxDocument", "FbxAnimCurveNode", "FbxCluster", "FbxAnimLayer", "FbxAnimStack")


def _ties(edit, node) -> list:
    """[(what a message calls it, the object)] of what the file holds that
    refers to ``node`` and no record holds: a constraint, a character, a
    display layer, a selection set ..."""
    found = {}
    held  = [node.GetDstObject(at) for at in range(node.GetDstObjectCount())]
    for at in range(node.GetDstPropertyCount()):
        owner = node.GetDstProperty(at).GetFbxObject()
        if owner is not None:
            held.append(owner)
    for other in held:
        if other is None:
            continue
        kind = str(other.GetClassId().GetName())
        if kind in _OWN_TIES or kind.startswith("FbxNodeAttribute"):
            continue
        found[int(other.GetUniqueID())] = (f"the {kind[3:] if kind.startswith('Fbx') else kind} {str(other.GetName())!r}", other)
    return list(found.values())


def _off_poses(edit, node) -> None:
    """``node`` taken off every pose that lists it; a pose left listing
    nothing goes"""
    scene = edit.scene
    for at in reversed(range(scene.GetPoseCount())):
        pose  = scene.GetPose(at)
        place = pose.Find(node)
        if place is None or place < 0:
            continue
        pose.Remove(place)
        if pose.GetCount() == 0:
            scene.RemovePose(at)
            pose.Destroy()


def _take_off(edit, node, name: str) -> None:
    """a node the records leave out taken off the file, with what is its
    own: its keys in every take, its place in the poses, the clusters that
    link it, its shape. What refers to it and no record holds is refused, or
    with ``force`` cut loose from it and named among what the save dropped"""
    fbx = edit.fbx
    for label, other in _ties(edit, node):

        def remove(other=other):
            other.DisconnectSrcObject(node)
            for at in reversed(range(node.GetDstPropertyCount())):
                prop = node.GetDstProperty(at)
                if prop.GetFbxObject() is not None and prop.GetFbxObject().GetUniqueID() == other.GetUniqueID():
                    prop.DisconnectSrcObject(node)

        edit.unread(f"{label}, which refers to node {name!r}", f"node {name!r} is taken off the file", remove)

    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxCluster.ClassId)
    for at in reversed(range(node.GetDstObjectCount(criteria))):
        cluster = edit.typed(node.GetDstObject(criteria, at), fbx.FbxCluster, fbx)
        if cluster is not None:
            cluster.Destroy()
    _off_poses(edit, node)
    strip_curves(edit, node)
    for at in reversed(range(node.GetNodeAttributeCount())):
        shape = node.GetNodeAttributeByIndex(at)
        if shape is None:
            continue
        strip_curves(edit, shape)
        node.RemoveNodeAttributeByIndex(at)
        shape.Destroy()
    parent = node.GetParent()
    if parent is not None:
        parent.RemoveChild(node)
    node.Destroy()


def _children(node) -> list:
    return [node.GetChild(at) for at in range(node.GetChildCount())]


def _in_order(edit, uuid_of: dict, order: list) -> list:
    """the file's nodes put in the records' order, and the uuid of each in
    the file's own: every parent's children in the order the records list
    them, and where the file's order (parents first, depth first) is still
    another, the place of each in the records kept on it, as ``save_fbx``
    keeps it"""
    from cgmath.formats._write import ORDER_PROPERTY

    fbx  = edit.fbx
    rank = {uuid: at for at, uuid in enumerate(order)}
    root = edit.scene.GetRootNode()
    for parent in [root] + [node for node, _ in _load._walk(edit.scene)]:
        held = _children(parent)
        want = sorted(held, key=lambda child: rank[uuid_of[int(child.GetUniqueID())]])
        if [int(child.GetUniqueID()) for child in held] == [int(child.GetUniqueID()) for child in want]:
            continue
        for child in held:
            parent.RemoveChild(child)
        for child in want:
            parent.AddChild(child)

    found  = [node for node, _ in _load._walk(edit.scene)]
    uuids  = [uuid_of[int(node.GetUniqueID())] for node in found]
    marked = [node.FindProperty(ORDER_PROPERTY).IsValid() for node in found]
    if uuids != order or any(marked):
        for node, uuid in zip(found, uuids):
            prop = node.FindProperty(ORDER_PROPERTY)
            if not prop.IsValid():
                prop = fbx.FbxProperty.Create(node, fbx.FbxIntDT, ORDER_PROPERTY, "")
            prop.Set(int(rank[uuid]))
    return uuids


def write_nodes(edit, base: dict, state: dict, undo: list) -> set:
    """the file given the nodes of the records to save: one they add made
    bare under its parent, one under another parent moved, one they leave
    out taken off (:func:`_take_off`); the file's nodes then in the records'
    order, and ``file._uuids`` the uuid of each. Returns the uuids of the
    nodes added. ValueError for a node of another kind than the file's, and
    for one left out that still carries a shape the records hold"""
    file = edit.file
    nodes, links = _wanted(state)
    was   = _node_lists(base)[0][1]
    old   = _links(was)
    names = {node.uuid: node.name for node in was}
    names.update({node.uuid: node.name for node in nodes})

    for uuid in links:
        if uuid in old and old[uuid][1] != links[uuid][1]:
            raise ValueError(
                f"FbxFile.save: node {names[uuid]!r} is a {links[uuid][1]} where the file holds a {old[uuid][1]}; "
                f"a node of another kind is another node: take it off and add one, or {_ELSE}"
            )
    added   = [node for node in nodes if node.uuid not in old]
    removed = [uuid for uuid in old if uuid not in links]
    moved   = [uuid for uuid in links if uuid in old and old[uuid][0] != links[uuid][0]]
    for field in _SHAPE_FIELDS:
        shapes = _held(state[field])
        for shape in shapes.list if shapes is not None else ():
            if shape.parent_uuid in removed:
                raise ValueError(
                    f"FbxFile.save: node {names[shape.parent_uuid]!r} is taken off the file and still carries one of "
                    f"its {field}; take that one off too, or give it another node"
                )

    by_uuid = dict(edit.nodes)
    root    = edit.scene.GetRootNode()
    for record in added:
        by_uuid[record.uuid] = _bare(edit, record)
    for uuid in moved:
        node   = by_uuid[uuid]
        parent = node.GetParent()
        if parent is not None:
            parent.RemoveChild(node)
    for uuid in [record.uuid for record in added] + moved:
        above = links[uuid][0]
        (root if above is None else by_uuid[above]).AddChild(by_uuid[uuid])
    # the deepest first: a node goes once nothing hangs from it
    for uuid in reversed([uuid for uuid in file._uuids if uuid in removed]):
        _take_off(edit, by_uuid.pop(uuid), names[uuid])

    uuid_of = {int(node.GetUniqueID()): uuid for uuid, node in by_uuid.items()}
    uuids   = _in_order(edit, uuid_of, [node.uuid for node in nodes])
    undo.append((file, "_uuids", file._uuids))
    file._uuids = uuids
    return {record.uuid for record in added}


# ------------------------------------------------- user defined attributes


def _user_props(edit, node) -> dict:
    """{name: the property} of ``node``'s user defined properties"""
    flag  = edit.fbx.FbxPropertyFlags.EFlags.eUserDefined
    found = {}
    prop  = node.GetFirstProperty()
    while prop.IsValid():
        if prop.GetFlag(flag):
            try:
                found[str(prop.GetName())] = prop
            except UnicodeDecodeError:
                pass
        prop = node.GetNextProperty(prop)
    return found


def write_user(edit, node, was, now, same) -> None:
    """a node's user defined attributes as its record holds them: one it
    adds made, one it leaves out taken off with its keys, one of another
    value or kind written; an attribute left as it was is not touched. The
    spec of each one written is kept beside it, as ``save_fbx`` keeps it, so
    it reads back as given. ValueError for an attribute an fbx file does not
    take (a kind with no fbx type, a multi, a text that is not plain ASCII)"""
    from cgmath.formats.fbx import FbxExporter, _ascii_json, _ascii_text
    from cgmath.hierarchy.hierarchy import _fbx_written_specs

    fbx   = edit.fbx
    flags = fbx.FbxPropertyFlags.EFlags
    old   = was.user_defined_attributes or {}
    new   = now.user_defined_attributes or {}
    moved = {name for name in set(old) | set(new) if name not in old or name not in new or not same(old[name], new[name])}
    if not moved:
        return

    for name in moved:
        spec = new.get(name)
        if spec is not None and not (isinstance(spec, dict) and ("attributeType" in spec or "dataType" in spec)):
            raise ValueError(
                f"FbxFile.save: node {now.name!r} holds the user defined attribute {name!r} as {spec!r}; an attribute "
                f"is a dict naming its kind, as {{'attributeType': 'bool', 'value': True}} or "
                f"{{'dataType': 'string', 'value': 'text'}}"
            )
    skipped = {}
    props, specs = FbxExporter._user_properties(now, skipped)
    if skipped:
        raise ValueError(
            f"FbxFile.save: node {now.name!r} holds user defined attributes an fbx file does not take "
            f"({', '.join(sorted(skipped))}); leave them out"
        )
    specs  = json.loads(urllib.parse.unquote(specs)) if specs else {}
    wanted = {entry[0]: entry for entry in props}
    held   = _user_props(edit, node)

    # an attribute the record leaves out, or holds as part of another now
    for name in moved:
        if name in held and name not in wanted:
            strip_curves_of(edit, held[name])
            held.pop(name).Destroy()

    written = dict(_fbx_written_specs(node) or {})
    for name in [name for name in written if name not in new]:
        del written[name]
    for name, kind, value, fields, keyable in props:
        # a vector's children are written with it
        parts = [name] + [child for child, spec in new.items() if spec.get("parent") == name]
        if not any(part in moved for part in parts):
            continue
        if isinstance(value, str) and _ascii_text(value) != value:
            raise ValueError(
                f"FbxFile.save: node {now.name!r} holds the user defined attribute {name!r}, a text the FBX SDK does "
                f"not take as it is; {_ELSE}"
            )
        prop = held.get(name)
        if prop is not None and (fields or prop.GetPropertyDataType().GetType() != kind.GetType()):
            strip_curves_of(edit, prop)
            prop.Destroy()
            prop = None
        if prop is None:
            prop = fbx.FbxProperty.Create(node, kind, name, name)
            for field in fields:
                prop.AddEnumValue(field)
            prop.ModifyFlag(flags.eUserDefined, True)
        prop.Set(value)
        prop.ModifyFlag(flags.eAnimatable, keyable)
        for part in parts:
            if part in specs:
                written[part] = specs[part]

    prop = node.FindProperty(FbxExporter._USER_ATTR_SPECS)
    if not written:
        if prop.IsValid():
            prop.Destroy()
        return
    if not prop.IsValid():
        prop = fbx.FbxProperty.Create(node, fbx.FbxStringDT, FbxExporter._USER_ATTR_SPECS, "")
    prop.Set(_ascii_json(written))


def strip_curves_of(edit, prop) -> None:
    """every key of one property taken off the file"""
    fbx      = edit.fbx
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
    for at in reversed(range(prop.GetSrcObjectCount(criteria))):
        curve_node = edit.typed(prop.GetSrcObject(criteria, at), fbx.FbxAnimCurveNode, fbx)
        for channel in range(curve_node.GetChannelsCount()):
            for k in reversed(range(curve_node.GetCurveCount(channel))):
                curve = edit.typed(curve_node.GetCurve(channel, k), fbx.FbxAnimCurve, fbx)
                if curve is not None:
                    curve.Destroy()
        curve_node.Destroy()


# --------------------------------------------------------------- the meshes

# the records that belong to a mesh, naming it by ``mesh_uuid``
_MESH_RECORDS = ("uvs", "skins", "materials", "morphs")

# a mesh's layer elements no record holds: none outlives its faces
_FACE_BOUND = (
    ("Tangent", "tangents"),
    ("Binormal", "binormals"),
    ("PolygonGroup", "polygon groups"),
    ("EdgeCrease", "edge creases"),
    ("VertexCrease", "vertex creases"),
    ("Visibility", "face visibilities"),
    ("UserData", "user data layers"),
)


class _Shim:
    """what ``save_fbx``'s builders ask of their exporter: the scene"""

    def __init__(self, scene):
        self.scene = scene

    def _check_open(self) -> None:
        pass


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _faces_differ(old, new) -> bool:
    """whether two meshes are of other faces or another count of points"""
    return (
        np.shape(old.points) != np.shape(new.points)
        or not _same(old.counts, new.counts)
        or not _same(old.indices, new.indices)
    )


def _by_node(meshes) -> dict:
    """{the uuid of its node: the mesh}; ValueError for two on one node"""
    found = {}
    for mesh in meshes.list if meshes is not None else ():
        if mesh.parent_uuid in found:
            raise ValueError(
                f"FbxFile.save: meshes holds two meshes on one node ({mesh.name!r} and "
                f"{found[mesh.parent_uuid].name!r}); a node carries one"
            )
        found[mesh.parent_uuid] = mesh
    return found


def _of(records, mesh) -> list:
    """the records of ``records`` that name ``mesh``"""
    if records is None or mesh is None:
        return []
    return [record for record in records.list if record.mesh_uuid == mesh.uuid]


def settle(file, base: dict, state: dict, undo: list) -> dict:
    """the records to save as a file lists them, ``state`` changed in place
    and {field: its new list} returned for the handle to hold once saved:

    - the one material of a mesh that lists every face of it covers the
      mesh whole (``faces`` None, ``undo`` taking the faces it listed), as
      a read holds it;

    - a list holding nothing is None;
    - the meshes in their nodes' order, and each mesh's UV sets, skin,
      materials and blend shapes in the meshes' order, as a read lists them;
    - the records of a mesh taken off the file (its node carries none in
      ``meshes`` any more) taken off with it;
    - the hierarchy of a file holding takes None once no skin is left: it
      is the pose the skins are bound in, and the takes hold the nodes

    ValueError for two meshes of one uuid, which the records naming a mesh
    could not tell apart"""
    replaced = {}
    nodes    = _node_lists(state)
    seen     = {}
    for mesh in (_held(state["meshes"]).list if _held(state["meshes"]) is not None else ()):
        if mesh.uuid in seen and seen[mesh.uuid] is not mesh:
            raise ValueError(
                f"FbxFile.save: two meshes of meshes share one uuid ({mesh.name!r} and {seen[mesh.uuid].name!r}); a "
                f"mesh made by copy() or from_faces() keeps its source's: give it one of its own "
                f"(mesh.uuid = generate_uuid()) before any record names it"
            )
        seen[mesh.uuid] = mesh
    rank   = {node.uuid: at for at, node in enumerate(nodes[0][1])} if nodes else {}
    meshes = _held(state["meshes"])
    if meshes is not None:
        held = sorted(meshes.list, key=lambda mesh: rank.get(mesh.parent_uuid, len(rank)))
        if not held:
            replaced["meshes"] = None
        elif [id(mesh) for mesh in held] != [id(mesh) for mesh in meshes.list]:
            replaced["meshes"] = type(meshes)(held)
        meshes = replaced.get("meshes", meshes)

    live  = {mesh.uuid: at for at, mesh in enumerate(meshes.list)} if meshes is not None else {}
    nodes = {mesh.parent_uuid for mesh in meshes.list} if meshes is not None else set()
    known = dict(file._issued)
    was   = _held(base["meshes"])
    known.update({mesh.uuid: mesh.parent_uuid for mesh in was.list} if was is not None else {})
    for field in _MESH_RECORDS:
        records = _held(state[field])
        if records is None:
            continue
        kept = [
            record for record in records.list
            if record.mesh_uuid in live or record.mesh_uuid not in known or known[record.mesh_uuid] in nodes
        ]
        kept.sort(key=lambda record: live.get(record.mesh_uuid, len(live)))
        if not kept:
            replaced[field] = None
        elif [id(record) for record in kept] != [id(record) for record in records.list]:
            replaced[field] = type(records)(kept)
    for field in ("cameras", "lights", "curves", "surfaces"):
        records = _held(state[field])
        if records is not None and not records.list:
            replaced[field] = None
    materials = replaced["materials"] if "materials" in replaced else _held(state["materials"])
    for mesh in meshes.list if meshes is not None and materials is not None else ():
        held = [record for record in materials.list if record.mesh_uuid == mesh.uuid]
        if len(held) == 1 and held[0].faces is not None and len(np.unique(held[0].faces)) == len(mesh.counts):
            undo.append((held[0], "faces", held[0].faces))
            held[0].faces = None
    skins = replaced["skins"] if "skins" in replaced else _held(state["skins"])
    if skins is None and _held(base["skins"]) is not None and _held(state["hierarchy"]) is not None and _held(state["clip"]) is not None:
        _wanted(state)
        replaced["hierarchy"] = None
    state.update(replaced)
    return replaced


def _shape_of(edit, node):
    """the FbxMesh ``node`` carries, None for none"""
    fbx   = edit.fbx
    shape = node.GetNodeAttribute()
    if shape is None or shape.GetAttributeType() != fbx.FbxNodeAttribute.EType.eMesh:
        return None
    return edit.typed(shape, fbx.FbxMesh, fbx)


def _skin_of(edit, shape):
    """the FbxSkin of a mesh, None for none"""
    fbx  = edit.fbx
    kind = fbx.FbxDeformer.EDeformerType.eSkin
    if not shape.GetDeformerCount(kind):
        return None
    return edit.typed(shape.GetDeformer(0, kind), fbx.FbxSkin, fbx)


def _clusters(edit, deformer) -> list:
    fbx = edit.fbx
    return [edit.typed(deformer.GetCluster(at), fbx.FbxCluster, fbx) for at in range(deformer.GetClusterCount())]


def _label(mesh, node) -> str:
    return f"mesh {mesh.name!r}" if mesh.name else f"the mesh on node {str(node.GetName())!r}"


def _weighed(skin, hierarchy) -> set:
    """the uuids of the nodes a skin's influences name in ``hierarchy``, a
    name it does not hold as it is"""
    names = {}
    for record in hierarchy.list if hierarchy is not None else ():
        names.setdefault(record.name, record.uuid)
    return {names.get(str(name), str(name)) for name in skin.influences}


def _file_world(edit, matrix) -> np.ndarray:
    """a world matrix of the records as the file holds one: in its axes and
    its unit, what the reader's turn and scale of a bind matrix undo"""
    held = np.array(matrix, dtype=np.float64)
    if edit.turn is not None:
        turn         = np.eye(4)
        turn[:3, :3] = edit.turn
        held         = held @ turn.T
    held[3, :3] /= edit.cm
    return held


def _amatrix(matrix) -> np.ndarray:
    """(4, 4) of an FbxAMatrix, row by row"""
    return np.array([[matrix.Get(row, column) for column in range(4)] for row in range(4)], dtype=np.float64)


def _drop_mesh(edit, node, loose: dict) -> None:
    """the mesh of ``node`` taken off the file with what is its own: its
    skin and clusters, its blend shapes and their keys, its UV sets, its
    node's materials, which ``loose`` takes ({id: material}) for
    :func:`_drop_loose`; the node stays, an empty transform"""
    fbx   = edit.fbx
    kinds = fbx.FbxDeformer.EDeformerType
    shape = _shape_of(edit, node)
    if shape is None:
        return
    for at in reversed(range(shape.GetDeformerCount())):
        deformer = shape.GetDeformer(at)
        if deformer.GetDeformerType() == kinds.eSkin:
            for cluster in _clusters(edit, edit.typed(deformer, fbx.FbxSkin, fbx)):
                cluster.Destroy()
        elif deformer.GetDeformerType() == kinds.eBlendShape:
            blend = edit.typed(deformer, fbx.FbxBlendShape, fbx)
            for c in reversed(range(blend.GetBlendShapeChannelCount())):
                channel = blend.GetBlendShapeChannel(c)
                strip_curves(edit, channel)
                for s in reversed(range(channel.GetTargetShapeCount())):
                    channel.GetTargetShape(s).Destroy()
                channel.Destroy()
        deformer.Destroy()
    for at in range(node.GetMaterialCount()):
        held                           = node.GetMaterial(at)
        loose[int(held.GetUniqueID())] = held
    node.RemoveAllMaterials()
    strip_curves(edit, shape)
    node.RemoveNodeAttribute(shape)
    shape.Destroy()
    empty = fbx.FbxNull.Create(edit.scene, "")
    empty.Look.Set(fbx.FbxNull.ELook.eNone.value)
    node.SetNodeAttribute(empty)
    flags = fbx.FbxPropertyFlags.EFlags
    for prop in (node.LclTranslation, node.LclRotation, node.LclScaling):
        for member in (flags.eLockedMember0, flags.eLockedMember1, flags.eLockedMember2):
            prop.ModifyFlag(member, False)


def _drop_loose(edit, loose: dict) -> None:
    """the materials a save left on no node taken off the file, each with
    the textures nothing else reads and their videos"""
    fbx    = edit.fbx
    nodes  = fbx.FbxCriteria.ObjectType(fbx.FbxNode.ClassId)
    maps   = fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId)
    videos = fbx.FbxCriteria.ObjectType(fbx.FbxVideo.ClassId)
    for shader in loose.values():
        if shader.GetDstObjectCount(nodes):
            continue
        textures = {}
        prop     = shader.GetFirstProperty()
        while prop.IsValid():
            for at in range(prop.GetSrcObjectCount(maps)):
                held                              = prop.GetSrcObject(maps, at)
                textures[int(held.GetUniqueID())] = held
            prop = shader.GetNextProperty(prop)
        shader.Destroy()
        for texture in textures.values():
            if texture.GetDstPropertyCount() or texture.GetDstObjectCount(maps):
                continue
            clips = [texture.GetSrcObject(videos, at) for at in range(texture.GetSrcObjectCount(videos))]
            texture.Destroy()
            for clip in clips:
                if clip is not None and not clip.GetDstObjectCount(maps):
                    clip.Destroy()


def _smooth_edges(edit, shape, mesh) -> bool:
    """the smoothing of a mesh given other faces written anew, an edge at a
    time: hard where the record breaks its normals; False where the record
    does not say (normals stored with no hard edges)"""
    fbx = edit.fbx
    if mesh.hard_edges is None and mesh.normals is not None:
        return False
    hard = set()
    if mesh.hard_edges is not None:
        for a, b in np.asarray(mesh.hard_edges, dtype=np.int64).reshape(-1, 2).tolist():
            hard.add((min(a, b), max(a, b)))
    shape.BuildMeshEdgeArray()
    if shape.GetLayerCount() == 0:
        shape.CreateLayer()
    element = fbx.FbxLayerElementSmoothing.Create(shape, "")
    shape.GetLayer(0).SetSmoothing(element)
    element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByEdge)
    element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
    direct = element.GetDirectArray()
    for at in range(shape.GetMeshEdgeCount()):
        a, b = tuple(shape.GetMeshEdgeVertices(at))[-2:]
        direct.Add(0 if (min(a, b), max(a, b)) in hard else 1)
    return True


def _reshape(edit, node, old, mesh):
    """the FbxMesh of ``mesh`` on ``node``, built as ``save_fbx`` builds one,
    in the file's unit: a mesh added (``old`` None), or one given other
    faces, whose FbxMesh it replaces. The skin of the one replaced moves to
    it, its user defined properties are copied, its smoothing is written
    anew; its other data no record holds is of the faces it had: refused, or
    with ``force`` dropped. ValueError for a mesh holding blend shapes, and
    for a node that carries another shape"""
    from cgmath.formats import _fbx_edit, _write

    fbx    = edit.fbx
    kinds  = fbx.FbxNodeAttribute.EType
    label  = _label(mesh, node)
    held   = _shape_of(edit, node) if old is not None else None
    why    = f"{label} is given other faces"
    kept   = []
    props  = []
    smooth = False
    if held is not None:
        if _fbx_edit._targets(edit, held):
            raise ValueError(
                f"FbxFile.save: {label} is given other faces and holds blend shapes, which an open file keeps on the "
                f"faces they were made for; {_ELSE}"
            )
        for kind, plural in _FACE_BOUND:
            count = getattr(held, f"GetElement{kind}Count", None)
            if count is not None and count():
                edit.unread(f"the {plural} of {label}", why, lambda: None)
        if held.GetElementVertexColorCount() > 1:
            edit.unread(f"the colour sets of {label} after its first", why, lambda: None)
        smooth = bool(held.GetElementSmoothingCount())
        skin   = fbx.FbxDeformer.EDeformerType.eSkin
        for at in reversed(range(held.GetDeformerCount())):
            deformer = held.GetDeformer(at)
            if deformer.GetDeformerType() == skin:
                kept.append(held.RemoveDeformer(at))
            else:
                edit.unread(f"the deformer {str(deformer.GetName())!r} of {label}", why, lambda: None)
        prop = held.GetFirstProperty()
        while prop.IsValid():
            if prop.GetFlag(fbx.FbxPropertyFlags.EFlags.eUserDefined):
                props.append(prop)
            prop = held.GetNextProperty(prop)
    else:
        carried = node.GetNodeAttribute()
        if carried is not None and carried.GetAttributeType() != kinds.eNull:
            raise ValueError(
                f"FbxFile.save: {label} is put on node {str(node.GetName())!r}, which carries another shape or is a "
                f"joint; give the mesh a transform of its own"
            )
        if carried is not None:
            node.RemoveNodeAttribute(carried)
            carried.Destroy()

    proxy = mesh
    if edit.cm != 1.0:
        proxy        = mesh.copy()
        proxy.points = np.asarray(mesh.points, dtype=np.float64) / edit.cm
    _, shape = _write._write_mesh(_Shim(edit.scene), proxy, node)
    if held is not None:
        for deformer in reversed(kept):
            shape.AddDeformer(deformer)
        for prop in props:
            made = fbx.FbxProperty.Create(shape, prop.GetPropertyDataType(), str(prop.GetName()), str(prop.GetLabel()))
            made.CopyValue(prop)
            made.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eUserDefined, True)
        if smooth and not _smooth_edges(edit, shape, mesh):
            edit.unread(f"the smoothing groups of {label}", why, lambda: None)
        for at in reversed(range(node.GetNodeAttributeCount())):
            other = node.GetNodeAttributeByIndex(at)
            if other is not None and int(other.GetUniqueID()) == int(held.GetUniqueID()):
                node.RemoveNodeAttributeByIndex(at)
        held.Destroy()
        node.SetNodeAttribute(shape)
    return shape


def _sync_uvs(edit, shape, sets: list, label: str, rebuilt: bool) -> None:
    """the UV sets of a mesh as its records list them: every set of the
    file's mesh taken off and each record's written, as ``save_fbx`` writes
    one"""
    from cgmath.formats import _fbx_edit
    from cgmath.formats.fbx import FbxExporter

    if not rebuilt:
        _fbx_edit._stale_elements(edit, shape, label, "uvs", f"{label} is given other UV sets")
    while shape.GetElementUVCount():
        shape.RemoveElementUV(shape.GetElementUV(0))
    if sets:
        FbxExporter.add_uvs(_Shim(edit.scene), shape, sets)


def _link_world(edit, node):
    """the bind matrix a cluster of the file already holds for ``node``,
    None where none links it"""
    fbx      = edit.fbx
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxCluster.ClassId)
    for at in range(node.GetDstObjectCount(criteria)):
        cluster = edit.typed(node.GetDstObject(criteria, at), fbx.FbxCluster, fbx)
        if cluster is not None and cluster.GetLink() is not None:
            held = fbx.FbxAMatrix()
            cluster.GetTransformLinkMatrix(held)
            return held
    return None


def _pose_of(edit, node):
    """the bind pose that lists ``node``, None for none"""
    scene = edit.scene
    for at in range(scene.GetPoseCount()):
        pose  = scene.GetPose(at)
        place = pose.Find(node)
        if pose.IsBindPose() and place is not None and place >= 0:
            return pose
    return None


def _sync_skin(edit, node, shape, mesh, skin, hierarchy, label: str) -> None:
    """the skin of a mesh as its record holds it: the file's skin taken off
    for a mesh given none; one made for a mesh the file binds to nothing; a
    cluster made for each influence the file's skin lacks, bound where the
    hierarchy places its joint, and listed in the mesh's bind pose; each
    influence's weights, the clusters in the record's order, one of an
    influence the record leaves out taken off"""
    from cgmath.formats import _fbx_edit, _write

    fbx      = edit.fbx
    deformer = _skin_of(edit, shape)
    if skin is None:
        if deformer is not None:
            for cluster in _clusters(edit, deformer):
                cluster.Destroy()
            deformer.Destroy()
        return
    if hierarchy is None:
        raise ValueError(f"FbxFile.save: {label} is given a skin in a file holding no hierarchy to bind it to; {_ELSE}")
    if len(set(skin.influences)) != len(skin.influences):
        raise ValueError(f"FbxFile.save: the skin of {label} names an influence twice")
    weights = np.asarray(skin.weights, dtype=np.float64)
    if len(weights) != len(mesh.points):
        raise ValueError(
            f"FbxFile.save: the skin of {label} weighs {len(weights)} points, the mesh holds {len(mesh.points)}; give "
            f"the skin the mesh's points (skin.subdivide(mesh) before mesh.subdivide())"
        )
    by_name = {}
    by_uuid = {}
    for record in hierarchy.list:
        by_name.setdefault(record.name, record)
        by_uuid[record.uuid] = record
    worlds = {}

    def world(record) -> np.ndarray:
        if record.uuid not in worlds:
            worlds[record.uuid] = _file_world(edit, record.world_matrix)
        return worlds[record.uuid]

    made = deformer is None
    if made:
        deformer = fbx.FbxSkin.Create(edit.scene, skin.name or "")
        deformer.SetSkinningType(fbx.FbxSkin.EType.eLinear)
        shape.AddDeformer(deformer)
        flags = fbx.FbxPropertyFlags.EFlags
        for prop in (node.LclTranslation, node.LclRotation, node.LclScaling):
            for member in (flags.eLockedMember0, flags.eLockedMember1, flags.eLockedMember2):
                prop.ModifyFlag(member, True)

    # an influence is its node, whatever a save in progress renames it
    uuid_of = {int(found.GetUniqueID()): uuid for uuid, found in edit.nodes.items()}
    groups = {
        uuid_of.get(int(group[0].GetLink().GetUniqueID())): group for group in _fbx_edit._linked(edit, deformer)
    }
    bound = None
    for group in groups.values():
        bound = fbx.FbxAMatrix()
        group[0].GetTransformMatrix(bound)
        break
    if bound is None:
        placed = by_uuid.get(mesh.parent_uuid)
        bound  = _write._fbx_affine(np.eye(4) if placed is None else world(placed), fbx)

    fresh = []
    order = []
    for column, influence in enumerate(skin.influences):
        record = by_name.get(str(influence))
        if record is None or record.uuid not in edit.nodes:
            raise ValueError(
                f"FbxFile.save: the skin of {label} names the influence {influence!r}, which the hierarchy does not hold"
            )
        group = groups.get(record.uuid)
        if group is None:
            link    = edit.nodes[record.uuid]
            held    = _link_world(edit, link)
            cluster = fbx.FbxCluster.Create(edit.scene, str(influence))
            cluster.SetLink(link)
            cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)
            cluster.SetTransformMatrix(bound)
            cluster.SetTransformLinkMatrix(_write._fbx_affine(world(record), fbx) if held is None else held)
            group = [cluster]
            fresh.append(record)
        _fbx_edit._set_cluster(group[0], weights[:, column])
        for cluster in group[1:]:
            cluster.SetControlPointIWCount(0)
        order.append(group)

    wanted = {int(cluster.GetUniqueID()) for group in order for cluster in group}
    for cluster in _clusters(edit, deformer):
        deformer.RemoveCluster(cluster)
        if int(cluster.GetUniqueID()) not in wanted:
            cluster.Destroy()
    for group in order:
        for cluster in group:
            deformer.AddCluster(cluster)

    # the bind pose: Maya's importer poses the joints by it
    pose = _pose_of(edit, node)
    if pose is None and made:
        pose = fbx.FbxPose.Create(edit.scene, skin.name or "BindPose")
        pose.SetIsBindPose(True)
        pose.Add(node, _write._fbx_matrix(_amatrix(bound), fbx))
        edit.scene.AddPose(pose)
    if pose is None:
        return
    for record in fresh:
        member = record
        while member is not None:
            target = edit.nodes.get(member.uuid)
            place  = None if target is None else pose.Find(target)
            if target is not None and (place is None or place < 0):
                held   = _link_world(edit, target)
                matrix = world(member) if held is None else _amatrix(held)
                pose.Add(target, _write._fbx_matrix(matrix, fbx))
            member = by_uuid.get(member.parent_uuid)


def _shader_table(edit, base: dict):
    """{a material record's uuid: the file's FbxSurfaceMaterial}, off the
    order the handle read its materials in; None for a handle that read
    none"""
    from cgmath.formats import _fbx_edit

    uuids = edit.file._material_uuids
    if uuids is None:
        return None
    held = _fbx_edit._shaders(edit, base)
    if len(held) != len(uuids):
        raise ValueError(f"FbxFile.save: the file's materials are not the ones read; {_ELSE}")
    return {uuid: shader for uuid, (shader, _, _, _) in zip(uuids, held)}


def _sync_materials(edit, node, shape, mesh, records: list, shaders: dict, rebuilt: bool, loose: dict) -> None:
    """the materials of a mesh as its records list them: each record's
    material on the node, in the records' order, the file's own where the
    record is of one it holds (one uuid a material), else one made as
    ``save_fbx`` makes it; the faces of each in the mesh's material element,
    written anew. A mesh whose records the file already lists so, on the
    faces it had, is left to the values' writer"""
    from cgmath.formats import _write
    from cgmath.formats._load import _read_materials

    fbx    = edit.fbx
    label  = _label(mesh, node)
    wraps  = {mode: _write._enum(fbx.FbxTexture, "EWrapMode", name) for mode, name in (("repeat", "eRepeat"), ("clamp", "eClamp"))}
    wanted = []
    for record in records:
        held = None if record.uuid is None else shaders.get(record.uuid)
        if held is None:
            if record.texture is not None or record.normal_map is not None:
                raise ValueError(
                    f"FbxFile.save: material {record.name!r} of {label} is one the file does not hold and brings a "
                    f"texture or a normal map; an open file keeps the maps it holds; {_ELSE}"
                )
            held = _write._surface(edit.scene, record, None, set(), wraps, fbx)
            if record.uuid is not None:
                shaders[record.uuid] = held
        wanted.append(held)

    if not rebuilt:
        entries = _read_materials(node, shape, mesh.counts, label, edit.file._filename, fbx, pixels=False)
        same    = [entry["id"] for entry in entries] == [int(held.GetUniqueID()) for held in wanted]
        whole   = [entry["faces"] is None for entry in entries] == [record.faces is None for record in records]
        if same and whole:
            return

    for at in range(node.GetMaterialCount()):
        held                           = node.GetMaterial(at)
        loose[int(held.GetUniqueID())] = held
    node.RemoveAllMaterials()
    for held in wanted:
        node.AddMaterial(held)
    while shape.GetElementMaterialCount():
        shape.RemoveElementMaterial(shape.GetElementMaterial(0))
    if not records:
        return
    if shape.GetLayerCount() == 0:
        shape.CreateLayer()
    element = fbx.FbxLayerElementMaterial.Create(shape, "")
    shape.GetLayer(0).SetMaterials(element)
    element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
    indices = element.GetIndexArray()
    if len(records) == 1 and records[0].faces is None:
        element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eAllSame)
        indices.Add(0)
        return
    element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygon)
    owner = np.zeros(len(mesh.counts), dtype=np.int64)
    for place, record in enumerate(records):
        owner[np.asarray(record.faces, dtype=np.int64)] = place
    for index in owner.tolist():
        indices.Add(index)


def shapes_differ(file, base: dict, state: dict) -> bool:
    """whether the records to save hold other meshes than the file, a mesh
    of other faces, or other records of a mesh (UV sets, a skin, its
    influences, materials) than the file lists for it"""
    was, now = _held(base["meshes"]), _held(state["meshes"])
    old = {mesh.parent_uuid: mesh for mesh in was.list} if was is not None else {}
    new = {mesh.parent_uuid: mesh for mesh in now.list} if now is not None else {}
    if set(old) != set(new):
        return True
    for uuid, mesh in new.items():
        if _faces_differ(old[uuid], mesh):
            return True
        if len(_of(_held(base["uvs"]), old[uuid])) != len(_of(_held(state["uvs"]), mesh)):
            return True
        before, after = _of(_held(base["skins"]), old[uuid]), _of(_held(state["skins"]), mesh)
        if len(before) != len(after):
            return True
        if after and _weighed(after[0], _held(state["hierarchy"])) - _weighed(before[0], _held(base["hierarchy"])):
            return True
    if file._material_uuids is None or "materials" not in file._values:
        return False
    before, after = _held(base["materials"]), _held(state["materials"])
    if before is None or after is None:
        return (before is None) != (after is None)
    if len(before.list) != len(after.list) or len(file._material_uuids) != len(before.list):
        return True
    for uuid, read, record in zip(file._material_uuids, before.list, after.list):
        if record.uuid != uuid or record.mesh_uuid != read.mesh_uuid or (record.faces is None) != (read.faces is None):
            return True
    return False


def write_shapes(edit, base: dict, state: dict) -> None:
    """the file given the meshes of the records to save, and each mesh the
    records that name it: a mesh the records leave out taken off
    (:func:`_drop_mesh`), one they add or give other faces built
    (:func:`_reshape`), and each one's UV sets, skin and materials listed as
    its records list them. ValueError for a mesh on no node of the file"""
    was       = _by_node(_held(base["meshes"]))
    now       = _by_node(_held(state["meshes"]))
    hierarchy = _held(state["hierarchy"])
    for uuid, mesh in now.items():
        if uuid not in edit.nodes:
            raise ValueError(
                f"FbxFile.save: mesh {mesh.name!r} sits on no node of the file; a mesh added to an open file names "
                f"the node that carries it: append a TransformData to the hierarchy and give the mesh its uuid as "
                f"parent_uuid"
            )
    loose   = {}
    shaders = _shader_table(edit, base)
    for uuid in was:
        if uuid not in now:
            _drop_mesh(edit, edit.nodes[uuid], loose)
    for uuid, mesh in now.items():
        node    = edit.nodes[uuid]
        old     = was.get(uuid)
        label   = _label(mesh, node)
        rebuilt = old is None or _faces_differ(old, mesh)
        shape   = _reshape(edit, node, old, mesh) if rebuilt else _shape_of(edit, node)

        sets = _of(_held(state["uvs"]), mesh)
        if rebuilt or len(sets) != len(_of(_held(base["uvs"]), old)):
            _sync_uvs(edit, shape, sets, label, rebuilt)

        before = _of(_held(base["skins"]), old)
        after  = _of(_held(state["skins"]), mesh)
        if len(after) > 1:
            raise ValueError(f"FbxFile.save: {label} is given {len(after)} skins; a mesh takes one")
        added = after and before and _weighed(after[0], hierarchy) - _weighed(before[0], _held(base["hierarchy"]))
        if rebuilt or len(before) != len(after) or added:
            _sync_skin(edit, node, shape, mesh, after[0] if after else None, hierarchy, label)

        records = _of(_held(state["materials"]), mesh)
        if shaders is not None:
            _sync_materials(edit, node, shape, mesh, records, shaders, rebuilt, loose)
        elif old is None and records:
            _sync_materials(edit, node, shape, mesh, records, {}, True, loose)
    _drop_loose(edit, loose)
