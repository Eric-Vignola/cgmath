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


def write_nodes(edit, base: dict, state: dict, undo: list) -> None:
    """the file given the nodes of the records to save: one they add made
    bare under its parent, one under another parent moved, one they leave
    out taken off (:func:`_take_off`); the file's nodes then in the records'
    order, and ``file._uuids`` the uuid of each. ValueError for a node of
    another kind than the file's, and for one left out that still carries a
    shape the records hold"""
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
