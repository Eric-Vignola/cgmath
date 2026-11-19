"""``FbxFile.save`` of changed records: each change written into the open
scene, in the file's own axes, units and pivots

The scene is read once before a change is written and once after: the first
read tells what changed, record by record; the second must hold the records
as they are saved, or the save is refused and the scene put back. A change
this module writes wrong is so a refusal, never a wrong file.
"""

import dataclasses
import os

import numpy as np

from cgmath.formats import _fbx_layout, _load

FIELDS = ("hierarchy", "clip", "meshes", "skins", "uvs", "cameras", "lights", "materials", "morphs", "curves", "surfaces")

# the fields a node's own values are held by; every other one is a shape's
_NODE_FIELDS = ("hierarchy", "clip")

# how near a value read back is to the one saved, (rtol, atol): doubles
# through the file's own conversion; a take's keys through the 32-bit floats
# an fbx curve holds
_NEAR  = (1e-9, 1e-9)
_CLOSE = {"clip": (1e-5, 1e-5)}
_SAME  = (0.0, 0.0)

_ELSE  = "write the records with save_fbx"


class _Refused:
    """a field the reader refuses, held in its place"""

    def __init__(self, error):
        self.error = error


def _held(value):
    """a field's value, None for a refused one"""
    return None if isinstance(value, _Refused) else value


# ------------------------------------------------------------------ reading


def _shadow(file, changed, like) -> dict:
    """every field of ``file``'s open scene as it reads now, off a handle of
    its own sharing the scene and the nodes' uuids: {field: value}, a
    refused one a :class:`_Refused`. Each mesh takes the uuid of ``like``'s
    mesh on the same node, so the records naming a mesh name the same one in
    every read"""
    from cgmath.formats._fbx_file import _REFUSALS, FbxFile

    shadow                 = FbxFile.__new__(FbxFile)
    shadow._fbx            = file._fbx
    shadow._manager        = file._manager
    shadow._scene          = file._scene
    shadow._media          = None
    shadow._mesh_raw       = None
    shadow._reads          = {}
    shadow._refused        = {}
    shadow._filename       = file._filename
    shadow._textures       = file._textures if "materials" in changed else False
    shadow._verbose        = False
    shadow._uuids          = file._uuids
    shadow._refusals       = {}
    shadow._values         = {}
    shadow._digests        = {}
    shadow._warned         = set()
    shadow._hearing        = False
    shadow._issued         = {}
    shadow._texture_dir    = None
    shadow._wrote_media    = False
    shadow._settled        = {}
    shadow._material_uuids = None

    values = {}
    try:
        for field in ("meshes",) + tuple(name for name in FIELDS if name != "meshes"):
            try:
                value = getattr(shadow, field)
            except _REFUSALS as error:
                value = _Refused(error)
            if field == "meshes" and _held(value) is not None and _held(like) is not None:
                uuids = {}
                for mesh in like:
                    uuids.setdefault(mesh.parent_uuid, mesh.uuid)
                for mesh in value:
                    uuid = uuids.pop(mesh.parent_uuid, None)
                    if uuid is not None:
                        mesh.uuid = uuid
            values[field] = value
    finally:
        # the scene is the file's: this handle lets go of it, unfreed
        shadow._manager  = None
        shadow._scene    = None
        shadow._mesh_raw = None
        shadow._reads.clear()
        shadow._refused.clear()
    return values


def _state(file, base: dict, changed) -> dict:
    """the fields to save: a changed one as the handle holds it, any other
    as the file reads; the nodes' fields as held whenever they are, a copy
    of the read otherwise, since a change of one node field is carried to
    the other"""
    state = {}
    for field in FIELDS:
        if field in file._values and (field in changed or field in _NODE_FIELDS):
            state[field] = file._values[field]
        elif field in _NODE_FIELDS and _held(base[field]) is not None:
            state[field] = base[field].copy()
        else:
            state[field] = base[field]
    return state


# ---------------------------------------------------------------- comparing


def _differs(a, b, near: tuple, where: str):
    """None where ``a`` and ``b`` hold the same, floats within ``near``
    (rtol, atol); else where they first differ, in words. A shape's own
    ``uuid`` is passed over: a read makes one anew"""
    from cgmath.data import DataList
    from cgmath.hierarchy.hierarchy import ClipData, ClipList, TransformData

    if isinstance(a, _Refused) or isinstance(b, _Refused):
        return None if isinstance(a, _Refused) and isinstance(b, _Refused) else where
    if a is None or b is None:
        return None if a is None and b is None else where
    if isinstance(a, ClipData) and isinstance(b, ClipData):
        return _take_differs(a, b, near, where)
    if isinstance(a, DataList) and isinstance(b, DataList):
        if type(a) is not type(b):
            return where
        if len(a.list) != len(b.list):
            return f"{where} ({len(a.list)} records against {len(b.list)})"
        for at, (one, other) in enumerate(zip(a.list, b.list)):
            text = _differs(one, other, near, f"{where}[{at}]")
            if text is not None:
                return text
        if isinstance(a, ClipList):
            places = [next((at for at, take in enumerate(held.list) if take is held.active), -1) for held in (a, b)]
            return None if places[0] == places[1] else f"{where}.active"
        return None
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        if type(a) is not type(b):
            return where
        shaded = type(a).__name__ == "MeshData"
        for field in dataclasses.fields(a):
            name = field.name.lstrip("_")
            if name == "uuid" and not isinstance(a, TransformData):
                continue
            if shaded and name in _SHADING:
                continue
            one   = getattr(a, name) if hasattr(a, name) else getattr(a, field.name)
            other = getattr(b, name) if hasattr(b, name) else getattr(b, field.name)
            text  = _differs(one, other, near, f"{where}.{name}")
            if text is not None:
                return text
        return _shading_differs(a, b, near, where) if shaded else None
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return where
        for key in a:
            text = _differs(a[key], b[key], near, f"{where}[{key!r}]")
            if text is not None:
                return text
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return where
        for at, (one, other) in enumerate(zip(a, b)):
            text = _differs(one, other, near, f"{where}[{at}]")
            if text is not None:
                return text
        return None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return None if _same_array(a, b, near) else where
    if isinstance(a, float) or isinstance(b, float):
        try:
            return None if _same_array(float(a), float(b), near) else where
        except (TypeError, ValueError):
            return where
    return None if a == b else where


# the fields that say how a mesh is shaded: its normals, or the edges the
# derived ones break at
_SHADING = ("normals", "normal_indices", "hard_edges")


def _shading_differs(a, b, near: tuple, where: str):
    """:func:`_differs` of how two meshes of the same points and faces are
    shaded: the same fields, else the same normal at every face corner,
    within the reader's own tolerance for derived normals. Hard edges two
    flat faces share shade as soft ones, and so read back"""
    from cgmath.geometry.mesh import _corner_normals

    if all(_differs(getattr(a, name), getattr(b, name), near, where) is None for name in _SHADING):
        return None
    if near == _SAME:
        return f"{where}.normals"
    try:
        corners = [normals[np.asarray(ids)] for normals, ids in (_corner_normals(mesh, always=True, force=True) for mesh in (a, b))]
    except (ValueError, RuntimeError):
        return f"{where}.normals"
    loose = (near[0], max(near[1], _load.NORMALS_DERIVED))
    return None if _same_array(corners[0], corners[1], loose) else f"{where}.normals"


def _same_array(a, b, near: tuple) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        try:
            return bool(np.allclose(a, b, rtol=near[0], atol=near[1], equal_nan=True))
        except TypeError:
            return False
    return bool(np.array_equal(a, b))


def _take_differs(a, b, near: tuple, where: str):
    """:func:`_differs` of two takes: their span, their nodes, and each
    node's channels, keys and visibility, node by node whatever the order
    of the columns"""
    for name in ("take", "start_frame", "frame_count"):
        if getattr(a, name) != getattr(b, name):
            return f"{where}.{name}"
    if not _same_array(float(a.fps), float(b.fps), near):
        return f"{where}.fps"
    a._settle()
    b._settle()
    if len(a.list) != len(b.list):
        return f"{where} ({len(a.list)} nodes against {len(b.list)})"
    for at, (one, other) in enumerate(zip(a.list, b.list)):
        text = _differs(one, other, near, f"{where}[{at}]")
        if text is not None:
            return text
    place, other = a._column_place(), b._column_place()
    if set(place) != set(other):
        return where
    uuids = sorted(place, key=place.get)
    here  = [place[uuid] for uuid in uuids]
    there = [other[uuid] for uuid in uuids]
    for channel in sorted(set(a._blocks) | set(b._blocks)):
        if channel not in a._blocks or channel not in b._blocks:
            return f"{where}.{channel.lstrip('_')}"
        if not _same_array(np.asarray(a._blocks[channel])[:, here], np.asarray(b._blocks[channel])[:, there], near):
            return f"{where}.{channel.lstrip('_')}"
        if not np.array_equal(np.asarray(a._keyed(channel))[here], np.asarray(b._keyed(channel))[there]):
            return f"{where}.{channel.lstrip('_')} (which nodes it keys)"
    if not np.array_equal(a._shown()[:, here], b._shown()[:, there]):
        return f"{where}.visibility"
    return _differs(a._user_keys, b._user_keys, near, f"{where}.user_keys")


# ----------------------------------------------------------------- checking


def _layout(base: dict, state: dict) -> None:
    """ValueError unless ``state`` holds the records the file holds: the
    same nodes under the same parents, the same takes, as many of each
    shape"""
    for field in FIELDS:
        was, now = base[field], state[field]
        if isinstance(was, _Refused) or isinstance(now, _Refused):
            continue
        if (was is None) != (now is None):
            raise ValueError(
                f"FbxFile.save: {field} is {'None' if now is None else 'given'} where the file holds "
                f"{'none' if was is None else 'some'}; {_ELSE}"
            )
        if was is None:
            continue
        if len(was.list) != len(now.list):
            raise ValueError(
                f"FbxFile.save: {field} holds {len(now.list)} records where the file holds {len(was.list)}; {_ELSE}"
            )
        takes = zip(was.list, now.list) if field == "clip" else [(was, now)] if field == "hierarchy" else []
        for held, given in takes:
            links = [{node.uuid: (node.parent_uuid, node.node_type) for node in nodes.list} for nodes in (held, given)]
            if links[0] != links[1]:
                raise ValueError(
                    f"FbxFile.save: {field} holds other nodes than the file's, or a node under another parent or of "
                    f"another kind; {_ELSE}"
                )


def _checked(state: dict) -> None:
    """``save_fbx``'s own checks of the records to save"""
    from cgmath.formats import _write

    held      = {field: _held(value) for field, value in state.items()}
    hierarchy = held["hierarchy"]
    clip      = held["clip"]
    meshes    = held["meshes"]
    written   = hierarchy
    if hierarchy is not None:
        _write._check_hierarchy(hierarchy)
    if clip is not None:
        _write._check_takes(clip)
        if hierarchy is None:
            written = clip.active.at(clip.active.start_frame)
        else:
            for take in clip:
                _write._check_same(hierarchy, take)
    if meshes is not None:
        _write._check_meshes(meshes, written)
    if held["uvs"] is not None:
        _write._check_uvs(held["uvs"], meshes)
    if held["skins"] is not None:
        _write._check_skins(held["skins"], meshes, hierarchy)
        if clip is not None:
            for take in clip:
                _write._check_still(held["skins"], meshes, hierarchy, take)
    if held["cameras"] is not None or held["lights"] is not None:
        _write._check_views(held["cameras"], held["lights"], meshes, written)
    if held["materials"] is not None:
        _write._check_materials(held["materials"], meshes)
    if held["morphs"] is not None:
        _write._check_morphs(held["morphs"], meshes)
    if held["curves"] is not None or held["surfaces"] is not None:
        _write._check_nurbs(held["curves"], held["surfaces"], meshes, held["cameras"], held["lights"], written)


def _validate(base: dict, state: dict) -> None:
    """the records to save pass ``save_fbx``'s checks, or fail one the
    file's own records fail the same way: a file ``save_fbx`` would not
    write still saves what it held"""
    try:
        _checked(state)
    except (TypeError, ValueError, NotImplementedError) as error:
        try:
            _checked(base)
        except (TypeError, ValueError, NotImplementedError) as before:
            if type(before) is type(error) and str(before) == str(error):
                return
        raise type(error)(str(error).replace("save_fbx", "FbxFile.save", 1)) from None


# ------------------------------------------------------------------ writing


def apply(file, changed, force: bool = False, began=None) -> list:
    """the ``changed`` fields of ``file`` written into its open scene;
    returns what ``force`` dropped of the file, each in words. ValueError,
    TypeError or NotImplementedError for a change that cannot be: the
    records are then as they were given, and where ``began`` (a list) was
    appended to, the scene is part written and the caller's to put back"""
    undo  = []
    began = [] if began is None else began
    try:
        return _apply(file, changed, force, undo, began)
    except BaseException:
        for held, name, value in reversed(undo):
            setattr(held, name, value)
        raise


def _apply(file, changed, force: bool, undo: list, began: list) -> list:
    base    = _shadow(file, changed, file._values.get("meshes"))
    state   = _state(file, base, changed)
    settled = _fbx_layout.settle(file, base, state, undo)
    dropped = []
    fresh   = set()
    _spread(file, base, state, undo)
    nodes = _fbx_layout.nodes_differ(base, state)
    takes = _fbx_layout.takes_differ(base, state)
    if nodes or takes or _fbx_layout.shapes_differ(file, base, state):
        # the file given the records' layout first, then read as it stands
        _fbx_layout.check_nodes(state)
        _validate(base, state)
        began.append(True)
        edit      = _Edit(file, force)
        edit.undo = undo
        if nodes:
            fresh = _fbx_layout.write_nodes(edit, base, state, undo)
            edit, dropped = _Edit(file, force), edit.dropped
            edit.dropped = dropped
        _fbx_layout.write_shapes(edit, base, state)
        if takes:
            _fbx_layout.write_takes(edit, base, state)
        dropped = edit.dropped
        base    = _shadow(file, changed, state["meshes"])
    undo.append((file, "_settled", file._settled))
    file._settled = {field: value for field, value in settled.items() if field in file._values}
    _layout(base, state)
    _spread(file, base, state, undo)
    _validate(base, state)

    began.append(True)
    edit      = _Edit(file, force)
    edit.undo = undo
    edit.dropped.extend(dropped)
    edit.fresh = fresh
    _write_nodes(edit, base, state)
    _write_clip(edit, base, state, force)
    for field in FIELDS:
        if field in _NODE_FIELDS or _differs(base[field], state[field], _SAME, field) is None:
            continue
        writer = _WRITERS.get(field)
        if writer is None:
            raise ValueError(f"FbxFile.save: a change of {field} is not written into an open file; {_ELSE}")
        writer(edit, _held(base[field]), _held(state[field]), base, state)

    after = _shadow(file, changed, state["meshes"])
    for field in FIELDS:
        text = _differs(after[field], state[field], _CLOSE.get(field, _NEAR), field)
        if text is not None:
            raise ValueError(
                f"FbxFile.save: the change of {' and '.join(changed)} cannot be written into this file: {text} would "
                f"read back otherwise; {_ELSE}"
            )
    return edit.dropped


# a shape field -> what writes its change into the scene
_WRITERS = {}


# ---------------------------------------------------------------- the scene


class _Place:
    """how the file holds one node: a joint or not, a root or not, the
    camera or light it carries, the rotate axis it holds and the pivots the
    reader folds into its translate"""

    def __init__(self, edit, node, root: bool):
        from cgmath.hierarchy.hierarchy import _fbx_written_axis

        fbx       = edit.fbx
        kinds     = fbx.FbxNodeAttribute.EType
        attribute = node.GetNodeAttribute()
        kind      = None if attribute is None else attribute.GetAttributeType()
        view      = None
        if kind == kinds.eCamera:
            view = "camera"
        elif kind == kinds.eLight:
            view = _load._light_kind(edit.typed(attribute, fbx.FbxLight, fbx), fbx)

        self.node      = node
        self.root      = root
        self.attribute = attribute
        self.joint     = kind == kinds.eSkeleton
        self.view      = view if view in _load._VIEW_AXES else None
        self.axis      = np.zeros(3)
        if node.GetRotationActive():
            post = fbx.FbxAMatrix()
            post.SetR(node.GetPostRotation(edit.source))
            self.axis = np.array(
                _fbx_written_axis(node, tuple(x + 0.0 for x in tuple(post.Inverse().GetR())[:3])), dtype=np.float64
            )
        pivots = np.array(
            [
                tuple(getter(edit.source))[:3]
                for getter in (node.GetRotationPivot, node.GetScalingPivot, node.GetRotationOffset, node.GetScalingOffset)
            ],
            dtype=np.float64,
        )
        self.pivots = pivots if np.any(pivots != 0.0) else None


class _Edit:
    """the open scene of a file being changed: its nodes by uuid, its axes
    and unit, its takes"""

    def __init__(self, file, force: bool = False):
        from cgmath.formats._fbx_io import typed

        self.file   = file
        self.force  = force
        self.fbx    = file._fbx
        self.scene  = file._scene
        self.typed  = typed
        self.source = self.fbx.FbxNode.EPivotSet.eSourcePivot
        found       = _load._walk(self.scene)
        self.nodes  = {uuid: node for uuid, (node, _) in zip(file._uuids, found)}
        self.roots  = {uuid for uuid, (_, parent) in zip(file._uuids, found) if parent < 0}
        self.turn, self.cm = _load._axes(self.scene, self.fbx, file._filename)
        self.single  = len(self.stacks()) < 2
        self.dropped = []     # what force dropped of the file, each in words
        self.fresh   = set()  # the uuids of the nodes this save adds or makes another kind
        self.taken   = set()  # the names of the map files this save writes
        self.undo    = []     # (record, field, value before) of each record the save touches
        self.named   = {}     # a new material's uuid -> the names its maps were written under
        self._maps   = None   # the folder it writes them to
        self._places = {}

    def place(self, uuid) -> _Place:
        if uuid not in self._places:
            self._places[uuid] = _Place(self, self.nodes[uuid], uuid in self.roots)
        return self._places[uuid]

    def texture_folder(self) -> str:
        """a folder of this save's own for the maps it writes, under the
        handle's: a map of a name the file already holds does not meet it"""
        import tempfile

        if self._maps is None:
            self._maps             = tempfile.mkdtemp(prefix="save_", dir=self.file._texture_root())
            self.file._wrote_media = True
        return self._maps

    def unread(self, what: str, why: str, remove) -> None:
        """``what`` of the file, which no record holds and the change
        leaves stale: refused, or with ``force`` dropped (:func:`_unread`)"""
        _unread(self, what, why, remove)

    def stacks(self) -> list:
        """the scene's takes, as the reader lists them"""
        fbx      = self.fbx
        criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        return [
            self.typed(self.scene.GetSrcObject(criteria, at), fbx.FbxAnimStack, fbx)
            for at in range(self.scene.GetSrcObjectCount(criteria))
        ]

    def layer(self, stack, make: bool = False):
        """the layer of ``stack`` that holds its keys: the first with any,
        else its first; None for a stack with none, unless ``make``"""
        fbx      = self.fbx
        criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        layers = [
            self.typed(stack.GetMember(criteria, at), fbx.FbxAnimLayer, fbx)
            for at in range(stack.GetMemberCount(criteria))
        ]
        for layer in layers:
            if _load._layer_keys(layer, fbx) is not None:
                return layer
        if layers:
            return layers[0]
        if not make:
            return None
        layer = fbx.FbxAnimLayer.Create(self.scene, "BaseLayer")
        stack.AddMember(layer)
        return layer


# ------------------------------------------------- the reader's way, undone


def _rotation(angles, order: int) -> np.ndarray:
    """(F, 3, 3) of (F, 3) degrees in ``order``"""
    from cgmath.transforms import euler_to_matrix

    return euler_to_matrix(np.radians(np.asarray(angles, dtype=np.float64).reshape(-1, 3)), order)[:, :3, :3]


def _solved(matrices, order: int) -> np.ndarray:
    """(F, 3) degrees of (F, 3, 3) in ``order``"""
    from cgmath.transforms import matrix_to_euler

    whole            = np.zeros((len(matrices), 4, 4))
    whole[:, :3, :3] = matrices
    whole[:, 3, 3]   = 1.0
    return np.degrees(matrix_to_euler(whole, order)).reshape(-1, 3)


def _to_file(edit, place, name: str, order, translate, rotate, scale, orient, axis) -> tuple:
    """a node's values as its file holds them: ``(translate, rotate, scale,
    orient, axis)``, the first three (F, 3) and the last two (3,)

    What the reader does on the way in, undone in its reverse order: the
    turn of a root of a file in other axes, the file's unit, the turn of a
    camera's or light's node and the fold of the pivots into the translate.
    """
    from cgmath.hierarchy.hierarchy import _filtered, _fbx_turn_roots

    order  = int(order)
    blocks = [np.asarray(block, dtype=np.float64).reshape(-1, 3) for block in (translate, rotate, scale)]
    frames = max(len(block) for block in blocks)
    translate, rotate, scale = (np.broadcast_to(block, (frames, 3)).copy() for block in blocks)
    orient = np.array(orient, dtype=np.float64).reshape(3)
    axis   = np.array(axis, dtype=np.float64).reshape(3)

    if place.root and edit.turn is not None:
        translate, rotate, turned = _fbx_turn_roots(
            edit.turn.T, [place.joint], [order], translate[:, None], rotate[:, None],
            np.broadcast_to(orient, (frames, 1, 3)),
        )
        translate, rotate, orient = translate[:, 0], rotate[:, 0], turned[0, 0]
    translate = translate / edit.cm

    if place.view is not None:
        if np.any(axis != 0.0):
            raise ValueError(
                f"FbxFile.save: node {name!r} holds a {place.view if place.view == 'camera' else place.view + ' light'}, "
                f"whose rotate axis the file keeps for itself; leave it at zero"
            )
        axis0 = np.array(_load._VIEW_AXES[place.view], dtype=np.float64)
        if not np.allclose(place.axis, axis0, rtol=0, atol=1e-9):
            turn   = _rotation(-axis0, 0)[0] @ _rotation(place.axis, 0)[0]
            rotate = _solved(turn.T @ _rotation(rotate, order), order)
            if frames > 1:
                rotate = _filtered(rotate[:, None], [order])[:, 0]
            rotate = rotate + 0.0
        axis = place.axis.copy()

    if place.pivots is not None:
        translate = translate - _load._folded(np.zeros((frames, 3)), rotate, scale, orient, axis, order, place.pivots)
    return translate, rotate, scale, orient, axis


# ---------------------------------------------------------------- the nodes

# what a file holds once for a node, whichever records name it
_LABELS = ("name", "rotate_order", "segment_scale_compensate", "radius", "draw_style")


def _pairs(base: dict, state: dict) -> dict:
    """{uuid: [(the record read, the record to save)]} over the hierarchy
    and every take, in that order"""
    pairs = {}
    for field in _NODE_FIELDS:
        was, now = _held(base[field]), _held(state[field])
        if was is None or now is None:
            continue
        for held, given in zip(was.list, now.list) if field == "clip" else [(was, now)]:
            before = {node.uuid: node for node in held.list}
            for node in given.list:
                # a node the save adds is in no read yet
                if node.uuid in before:
                    pairs.setdefault(node.uuid, []).append((before[node.uuid], node))
    return pairs


def _spread(file, base: dict, state: dict, undo: list) -> None:
    """a node's labels made the same in every record of it: a file holds
    each once, so the one changed in the hierarchy or in a take is carried
    to the others. ValueError for two records giving two new values. A
    joint's new name is carried to the skins that weigh it. ``undo`` takes
    ``(record, field, value before)`` for each record touched"""
    renamed = {}
    for found in _pairs(base, state).values():
        for label in _LABELS:
            fresh = []
            for was, now in found:
                value = getattr(now, label, None)
                if _differs(getattr(was, label, None), value, _SAME, label) is None:
                    continue
                if all(_differs(value, other, _SAME, label) is not None for other in fresh):
                    fresh.append(value)
            if not fresh:
                continue
            if len(fresh) > 1:
                raise ValueError(
                    f"FbxFile.save: node {found[0][0].name!r} is given two values of {label}, {fresh[0]!r} and "
                    f"{fresh[1]!r}, by the records that hold it; a file holds one"
                )
            for _, now in found:
                if hasattr(now, label) and _differs(getattr(now, label), fresh[0], _SAME, label) is not None:
                    undo.append((now, label, getattr(now, label)))
                    setattr(now, label, fresh[0])
            if label == "name":
                renamed[found[0][0].name] = fresh[0]
    if not renamed:
        return
    seen = []
    for skins in (_held(state["skins"]), _held(file._values.get("skins"))):
        if skins is None or any(skins is other for other in seen):
            continue
        seen.append(skins)
        for skin in skins.list:
            names = [renamed.get(name, name) for name in skin.influences]
            if names != list(skin.influences):
                undo.append((skin, "influences", skin.influences))
                skin.influences = names


_MOVES = ("translate", "rotate", "scale", "joint_orient", "rotate_axis")


def _hidden(fbx, node, name: str, kind, value) -> None:
    """the hidden property ``name`` of ``node`` set to ``value``, made where
    it lacks; dropped for ``kind`` None"""
    prop = node.FindProperty(name)
    if kind is None:
        if prop.IsValid():
            prop.Destroy()
        return
    if not prop.IsValid():
        prop = fbx.FbxProperty.Create(node, kind, name, "")
    prop.Set(value)


def _active(edit, node) -> None:
    """``node``'s rotate order, pre and post rotation switched on, holding
    nothing where they were off: what an inactive node held is not read"""
    if node.GetRotationActive():
        return
    fbx  = edit.fbx
    zero = fbx.FbxVector4(0.0, 0.0, 0.0, 1.0)
    node.SetPreRotation(edit.source, zero)
    node.SetPostRotation(edit.source, zero)
    node.SetRotationOrder(edit.source, fbx.EFbxRotationOrder.eEulerXYZ)
    node.SetRotationActive(True)


def _write_orient(edit, place, node, name: str, was: tuple, now: tuple) -> None:
    """a node's joint orient and rotate axis, each ``(orient, axis)`` as the
    file holds them, written where ``now`` is not ``was``"""
    from cgmath.formats.fbx import FbxExporter, _post_rotation

    fbx = edit.fbx
    if not np.array_equal(was[0], now[0]):
        if not place.joint:
            raise ValueError(f"FbxFile.save: node {name!r} is no joint; a file holds a joint orient on a joint only")
        _active(edit, node)
        node.SetPreRotation(edit.source, fbx.FbxVector4(*[float(x) for x in now[0]], 1.0))
    if not np.array_equal(was[1], now[1]):
        _active(edit, node)
        node.SetPostRotation(edit.source, _post_rotation([float(x) for x in now[1]]))
        kept = bool(np.any(np.signbit(now[1]) | (now[1] != 0.0)))
        _hidden(
            fbx, node, FbxExporter._ROTATE_AXIS, fbx.FbxDouble3DT if kept else None,
            fbx.FbxDouble3(*[float(x) for x in now[1]]),
        )


def _write_static(edit, place, node, was, now) -> None:
    """a node's own values, off the records of a file holding no take: each
    component of its translate, rotate and scale the change moves, its
    joint orient, rotate axis and visibility"""
    from cgmath.formats.fbx import FbxExporter, _negative_zeros
    from cgmath.hierarchy.hierarchy import _fbx_maya_scale

    fbx = edit.fbx
    held = [
        _to_file(
            edit, place, record.name, record.rotate_order, record.translate, record.rotate, record.scale,
            record.joint_orient, record.rotate_axis,
        )
        for record in (was, now)
    ]
    for at, prop in enumerate((node.LclTranslation, node.LclRotation, node.LclScaling)):
        before, after = held[0][at][0], held[1][at][0]
        moved = before != after
        if not moved.any():
            continue
        if at == 1 and _resolved(edit, place):
            # another triple in the file: its three components stand together
            moved = np.ones(3, dtype=bool)
        if at == 2:
            after = _fbx_maya_scale(after)
        value        = np.array(tuple(prop.Get())[:3], dtype=np.float64)
        value[moved] = after[moved]
        prop.Set(fbx.FbxDouble3(*[float(x) for x in value]))
    _write_orient(edit, place, node, now.name, held[0][3:], held[1][3:])

    signs = _negative_zeros(now)
    if signs != _negative_zeros(was):
        _hidden(fbx, node, FbxExporter._NEGATIVE_ZEROS, fbx.FbxStringDT if signs else None, signs)
    if bool(was.visibility) != bool(now.visibility):
        node.Visibility.Set(1.0 if now.visibility else 0.0)
        if now.node_type == "locator":
            node.Show.Set(bool(now.visibility))


def _write_nodes(edit, base: dict, state: dict) -> None:
    """every node's change written: its labels, whichever record holds
    them, and its own values off the hierarchy of a file holding no take"""
    from cgmath.formats.fbx import FbxExporter, _ascii_text

    fbx = edit.fbx
    for uuid, found in _pairs(base, state).items():
        was, now = found[0]
        node  = edit.nodes[uuid]
        place = edit.place(uuid)
        for before, after in found:
            if _differs(before.user_defined_attributes, after.user_defined_attributes, _SAME, "") is not None:
                _fbx_layout.write_user(edit, node, before, after, lambda a, b: _differs(a, b, _SAME, "") is None)
                break
        if now.name != was.name:
            node.SetName(_ascii_text(now.name))
        if int(now.rotate_order) != int(was.rotate_order):
            _active(edit, node)
            node.SetRotationOrder(edit.source, FbxExporter._ROTATE_ORDER_MAP[int(now.rotate_order)])
        if bool(now.segment_scale_compensate) != bool(was.segment_scale_compensate):
            inherit = fbx.FbxTransform.EInheritType
            node.SetTransformationInheritType(
                inherit.eInheritRrs if now.segment_scale_compensate else inherit.eInheritRSrs
            )
        if place.joint and (float(now.radius) != float(was.radius) or int(now.draw_style) != int(was.draw_style)):
            radius = float(now.radius)
            edit.typed(place.attribute, fbx.FbxSkeleton, fbx).Size.Set(radius * (100.0 / 3.0))
            node.Show.Set(int(now.draw_style) != 2)
            lossy = radius * (100.0 / 3.0) * 3.0 / 100.0 != radius
            _hidden(fbx, node, FbxExporter._RADIUS, fbx.FbxDoubleDT if lossy else None, radius)
            odd = int(now.draw_style) not in (0, 2)
            _hidden(fbx, node, FbxExporter._DRAW_STYLE, fbx.FbxIntDT if odd else None, int(now.draw_style))

    was, now = _held(base["hierarchy"]), _held(state["hierarchy"])
    if was is None or now is None:
        return
    takes  = _held(state["clip"]) is not None
    before = {record.uuid: record for record in was.list}
    for record in now.list:
        read  = before[record.uuid]
        moved = [name for name in _MOVES if not np.array_equal(getattr(read, name), getattr(record, name))]
        # a node this save adds, or makes another kind, is given its own values
        if takes and moved and record.uuid not in edit.fresh:
            raise ValueError(
                f"FbxFile.save: node {record.name!r} is given another {moved[0]} in the hierarchy of a file holding takes, "
                f"where the hierarchy is the pose its takes start from and its skins are bound in; change the node in "
                f"the clip, or {_ELSE}"
            )
        if moved or int(read.rotate_order) != int(record.rotate_order) or bool(read.visibility) != bool(record.visibility):
            _write_static(edit, edit.place(record.uuid), edit.nodes[record.uuid], read, record)


# ---------------------------------------------------------------- the takes

_CHANNELS = (("_translate", "LclTranslation"), ("_rotate", "LclRotation"), ("_scale", "LclScaling"))


def _resolved(edit, place) -> bool:
    """whether the file holds this node's rotate as another triple than the
    records': a root that is no joint in a file of other axes, a camera's
    or light's node holding another axis than its kind's own"""
    if place.root and edit.turn is not None and not place.joint:
        return True
    if place.view is None:
        return False
    return not np.allclose(place.axis, np.array(_load._VIEW_AXES[place.view], dtype=np.float64), rtol=0, atol=1e-9)


def _times(edit, take) -> list:
    """the FbxTime of each frame of ``take``, where the reader looks for it"""
    fbx    = edit.fbx
    second = fbx.FbxTime()
    second.SetSecondDouble(1.0)
    frames = int(take.start_frame) + np.arange(int(take.frame_count))
    ticks  = np.round(frames * (int(second.Get()) / float(take.fps))).astype(np.int64)
    return [fbx.FbxTime(int(tick)) for tick in ticks.tolist()]


def _columns(edit, take) -> dict:
    """{uuid: a node's values over ``take``'s frames, as the file holds them
    (:func:`_to_file`)}"""
    take._settle()
    place = take._column_place()
    blocks = {
        name: np.asarray(take._blocks[name], dtype=np.float64)
        for name in ("_translate", "_rotate", "_scale", "_joint_orient", "_rotate_axis")
    }
    out = {}
    for record in take.list:
        column = place[record.uuid]
        out[record.uuid] = _to_file(
            edit, edit.place(record.uuid), record.name, record.rotate_order, blocks["_translate"][:, column],
            blocks["_rotate"][:, column], blocks["_scale"][:, column], blocks["_joint_orient"][0, column],
            blocks["_rotate_axis"][0, column],
        )
    return out


def _curve(edit, prop, layer, component=None, make: bool = False):
    """the curve of ``prop`` in ``layer``, of its ``component`` X, Y or Z
    where it holds three; None where it holds none, unless ``make``"""
    fbx = edit.fbx
    if make:
        prop.GetCurveNode(layer, True)
    found = prop.GetCurve(layer, make) if component is None else prop.GetCurve(layer, component, make)
    return edit.typed(found, fbx.FbxAnimCurve, fbx)


def _keyed_at(edit, curve, times, values, done: set) -> None:
    """``curve`` holding ``values`` at ``times`` and nothing else, linear"""
    from cgmath.formats._write import _keys

    fbx = edit.fbx
    _keys(curve, times[: len(values)], values, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear, fbx)
    done.add(int(curve.GetUniqueID()))


def _write_moves(edit, layer, place, before, after, keyed: tuple, times, retime: bool, loose: bool, done: set) -> None:
    """one node's translate, rotate and scale in one take, ``before`` and
    ``after`` as the file holds them: each component the change moves, a
    whole channel where the take is given other frames (``retime``), where
    the channel turns keyed or still (``keyed``: per channel, (was, is)),
    or where its rotate is another triple in the file

    A component holding a curve keeps it, a key a frame where the channel
    is keyed, else one; a keyed one holding none is given one; a still one
    holding none takes the node's own value where nothing else reads it
    (``loose``), else a curve of one key"""
    from cgmath.hierarchy.hierarchy import _fbx_maya_scale

    fbx  = edit.fbx
    node = place.node
    for at, (_, prop_name) in enumerate(_CHANNELS):
        old, new = before[at], after[at]
        was, now = keyed[at]
        moved = np.any(old != new, axis=0) if old.shape == new.shape else np.ones(3, dtype=bool)
        whole = retime or was != now or (at == 1 and moved.any() and _resolved(edit, place))
        prop  = getattr(node, prop_name)
        own   = None
        for k, component in enumerate("XYZ"):
            if not (whole or moved[k]):
                continue
            rows  = new[:, k] if now else new[:1, k]
            curve = _curve(edit, prop, layer, component)
            if curve is not None and curve.KeyGetCount():
                _keyed_at(edit, curve, times, rows, done)
            elif now and (was != now or not np.all(rows == rows[0])):
                _keyed_at(edit, _curve(edit, prop, layer, component, make=True), times, rows, done)
            elif not moved[k]:
                continue
            elif loose:
                own    = np.array(tuple(prop.Get())[:3], dtype=np.float64) if own is None else own
                own[k] = _fbx_maya_scale(rows[:1])[0] if at == 2 else rows[0]
            else:
                _keyed_at(edit, _curve(edit, prop, layer, component, make=True), times, rows[:1], done)
        if own is not None:
            prop.Set(fbx.FbxDouble3(*[float(x) for x in own]))


def _write_shown(edit, layer, node, record, old, new, times, retime: bool, loose: bool, done: set) -> None:
    """one node's visibility in one take, ``old`` and ``new`` (F,) bools: a
    stepped key a frame where it changes over the take, one key where a
    curve holds it still, else the node's own value where nothing else
    reads it (``loose``), else a curve of one key"""
    from cgmath.formats._write import _visibility_keys

    curve = _curve(edit, node.Visibility, layer)
    count = curve.KeyGetCount() if curve is not None else 0
    same  = old.shape == new.shape and bool(np.array_equal(old, new))
    if same and not (retime and count):
        return
    varies = not bool(np.all(new == new[0]))
    if not (varies or count) and loose:
        node.Visibility.Set(1.0 if new[0] else 0.0)
        if record.node_type == "locator":
            node.Show.Set(bool(new[0]))
        return
    rows = new if varies else new[:1]
    _visibility_keys(node, layer, times[: len(rows)], rows, edit.fbx)
    done.add(int(_curve(edit, node.Visibility, layer).GetUniqueID()))


def _write_user(edit, layer, node, record, old: dict, new: dict, times, retime: bool, done: set) -> None:
    """one node's keyed user attributes in one take, {attribute: (F, 1) or
    (F, 3)}: each one the change moves, every one where the take is given
    other frames"""
    from cgmath.formats.fbx import _user_property_value
    from cgmath.hierarchy.hierarchy import _user_channels

    fbx   = edit.fbx
    attrs = record.user_defined_attributes
    for attr, children in _user_channels(attrs).items():
        rows = new.get(attr)
        if rows is None:
            continue
        rows = np.asarray(rows, dtype=np.float64)
        held = old.get(attr)
        if not retime and held is not None and np.shape(held) == rows.shape and np.array_equal(held, rows):
            continue
        prop = node.FindProperty(attr)
        if not prop.IsValid():
            raise ValueError(
                f"FbxFile.save: node {record.name!r} keys the user defined attribute {attr!r}, which its file does not "
                f"hold; {_ELSE}"
            )
        prop.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
        if children is None:
            # an enum keys its field's place in the list, as its property holds it
            if attrs[attr].get("attributeType") == "enum":
                rows = np.array([[float(_user_property_value(attrs[attr], int(x)))] for x in rows[:, 0]])
            _keyed_at(edit, _curve(edit, prop, layer, make=True), times, rows[:, 0], done)
            continue
        for axis, component in enumerate("XYZ"):
            _keyed_at(edit, _curve(edit, prop, layer, component, make=True), times, rows[:, axis], done)


def _curve_name(curve_node) -> str:
    """what a curve node animates, as a message names it: object.property"""
    try:
        prop = curve_node.GetDstProperty(0)
        return f"{prop.GetFbxObject().GetName()}.{prop.GetName()}"
    except Exception:
        return str(curve_node.GetName())


def _sweep(edit, layer, take, times, force: bool, done: set) -> None:
    """a take given other frames: every curve of its layer left unwritten.
    One of a single key is moved to the take's first frame. One of more is
    of something the records do not hold (a camera's focal length, a blend
    shape's weight), which no record says how to move: ValueError, or with
    ``force`` its keys dropped, the first one's value held at the take's
    first frame, each named in ``edit.dropped``"""
    fbx      = edit.fbx
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
    unread   = []
    for j in range(layer.GetSrcObjectCount(criteria)):
        curve_node = edit.typed(layer.GetSrcObject(criteria, j), fbx.FbxAnimCurveNode, fbx)
        for channel in range(curve_node.GetChannelsCount()):
            for k in range(curve_node.GetCurveCount(channel)):
                curve = edit.typed(curve_node.GetCurve(channel, k), fbx.FbxAnimCurve, fbx)
                count = curve.KeyGetCount() if curve is not None else 0
                if not count or int(curve.GetUniqueID()) in done:
                    continue
                if count == 1:
                    _keyed_at(edit, curve, times, [float(curve.KeyGetValue(0))], done)
                else:
                    unread.append((_curve_name(curve_node), curve))
    if not unread:
        return
    names = sorted({name for name, _ in unread})
    if not force:
        raise ValueError(
            f"FbxFile.save: take {take.take!r} is given other frames, and its file keys {', '.join(names)} over the "
            f"frames it had, which the records do not hold; save with force=True to drop those keys, or {_ELSE}"
        )
    for _, curve in unread:
        _keyed_at(edit, curve, times, [float(curve.KeyGetValue(0))], done)
    edit.dropped.extend(f"the keys of {name} in take {take.take!r}, held at their first value" for name in names)


def _write_clip(edit, base: dict, state: dict, force: bool) -> None:
    """every take's change written: its name and span, the file's frame
    rate and active take, and each node's channels, visibility and keyed
    user attributes. A changed component's keys are linear, one a frame"""
    from cgmath.formats.fbx import _ascii_text
    from cgmath.hierarchy.hierarchy import _fbx_rate_mode

    was, now = _held(base["clip"]), _held(state["clip"])
    if was is None or now is None:
        return
    fbx    = edit.fbx
    stacks = edit.stacks()
    rated  = float(now.list[0].fps) != float(was.list[0].fps)
    if rated:
        mode = _fbx_rate_mode(float(now.list[0].fps))
        if mode is None:
            raise ValueError(f"FbxFile.save: no fbx time mode carries {now.list[0].fps} fps")
        edit.scene.GetGlobalSettings().SetTimeMode(getattr(fbx.FbxTime.EMode, mode))

    # a still component holding no curve reads the node's own value, which
    # every take and a skin's bind pose read too
    loose   = edit.single and _held(base["hierarchy"]) is None
    renamed = False
    for stack, read, take in zip(stacks, was.list, now.list):
        if _take_differs(read, take, _SAME, "") is None:
            continue
        retime  = rated or (read.start_frame, read.frame_count) != (take.start_frame, take.frame_count)
        layer   = edit.layer(stack, make=True)
        times   = _times(edit, take)
        done    = set()
        columns = _columns(edit, read), _columns(edit, take)
        places  = read._column_place(), take._column_place()
        shown   = read._shown(), take._shown()
        flags   = [(np.asarray(read._keyed(name)), np.asarray(take._keyed(name))) for name, _ in _CHANNELS]
        for record in take.list:
            uuid  = record.uuid
            node  = edit.nodes[uuid]
            place = edit.place(uuid)
            a, b = places[0][uuid], places[1][uuid]
            before, after = columns[0][uuid], columns[1][uuid]
            keyed = tuple((bool(old[a]), bool(new[b])) for old, new in flags)
            _write_orient(edit, place, node, record.name, before[3:], after[3:])
            _write_moves(edit, layer, place, before, after, keyed, times, retime, loose, done)
            _write_shown(edit, layer, node, record, shown[0][:, a], shown[1][:, b], times, retime, loose, done)
            _write_user(
                edit, layer, node, record, read._user_keys.get(uuid, {}), take._user_keys.get(uuid, {}), times, retime,
                done,
            )
        if retime:
            _sweep(edit, layer, take, times, force, done)
            first, last = fbx.FbxTime(times[0].Get()), fbx.FbxTime(times[-1].Get())
            # the sdk writes no span (0, 0) and reads another in its place
            if first.Get() == 0 and last.Get() == 0:
                last.Set(1)
            stack.SetLocalTimeSpan(fbx.FbxTimeSpan(first, last))
            stack.SetReferenceTimeSpan(fbx.FbxTimeSpan(first, last))
        if take.take != read.take:
            stack.SetName(_ascii_text(take.take))
            renamed = True

    at = [next((i for i, take in enumerate(held.list) if take is held.active), 0) for held in (was, now)]
    if at[0] != at[1] or renamed:
        stack = stacks[at[1]]
        edit.scene.SetCurrentAnimationStack(stack)
        edit.scene.ActiveAnimStackName.Set(fbx.FbxString(stack.GetName()))


# ------------------------------------------------------------------- shapes


def _paired(edit, was, now, field: str) -> list:
    """[(the file's record, the one to save, the FbxNode both sit on)],
    place by place; ValueError for a record on another node than the
    file's"""
    pairs = []
    for at, (old, new) in enumerate(zip(was.list, now.list)):
        if old.parent_uuid != new.parent_uuid or new.parent_uuid not in edit.nodes:
            raise ValueError(f"FbxFile.save: {field}[{at}] sits on another node than the file's; {_ELSE}")
        pairs.append((old, new, edit.nodes[new.parent_uuid]))
    return pairs


def _carried(edit, node, kind: str):
    """the shape ``node`` carries, as the sdk class named ``kind``"""
    return edit.typed(node.GetNodeAttribute(), getattr(edit.fbx, kind), edit.fbx)


def _mesh_shapes(edit, state: dict) -> dict:
    """{a mesh's uuid: (its FbxMesh, what a message calls it)}"""
    found  = {}
    meshes = _held(state["meshes"])
    for mesh in meshes.list if meshes is not None else ():
        node = edit.nodes.get(mesh.parent_uuid)
        if node is not None:
            found[mesh.uuid] = (_carried(edit, node, "FbxMesh"), _label(mesh, node))
    return found


def _label(mesh, node) -> str:
    return f"mesh {mesh.name!r}" if mesh.name else f"the mesh on node {str(node.GetName())!r}"


def _same(a, b) -> bool:
    """two fields of a record hold the same, to the bit"""
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _filled(array, values) -> None:
    """a layer element's array holding ``values`` alone"""
    array.Clear()
    for value in values:
        array.Add(value)


def _unread(edit, what: str, why: str, remove) -> None:
    """``what`` of the file, which no record holds and the change leaves
    stale (``why``): ValueError naming it, or with ``force`` taken off the
    file by ``remove()`` and listed among what the save dropped"""
    if not edit.force:
        raise ValueError(
            f"FbxFile.save: {why}, and the file holds {what}, which no record holds and the change leaves stale; "
            f"save with force=True to drop them, or {_ELSE}"
        )
    remove()
    if what not in edit.dropped:
        edit.dropped.append(what)


def _targets(edit, shape) -> list:
    """the blend shape targets of a mesh, in the reader's order:
    [(deformer, channel, place in the channel, FbxShape)]"""
    fbx   = edit.fbx
    kind  = fbx.FbxDeformer.EDeformerType.eBlendShape
    found = []
    for d in range(shape.GetDeformerCount(kind)):
        deformer = edit.typed(shape.GetDeformer(d, kind), fbx.FbxBlendShape, fbx)
        for c in range(deformer.GetBlendShapeChannelCount()):
            channel = deformer.GetBlendShapeChannel(c)
            for s in range(channel.GetTargetShapeCount()):
                found.append((deformer, channel, s, channel.GetTargetShape(s)))
    return found


def _stale_targets(edit, targets: list, label: str, why: str) -> None:
    """the normals the targets hold, which no record does, refused or with
    ``force`` dropped: they are of the points the targets had"""
    held = [target for _, _, _, target in targets if target.GetElementNormalCount()]
    if not held:
        return

    def remove():
        for target in held:
            while target.GetElementNormalCount():
                target.RemoveElementNormal(target.GetElementNormal(0))

    _unread(edit, f"the normals of the blend shape targets of {label}", why, remove)


# a mesh's layer elements no record holds, and what leaves each stale
_DERIVED = (
    ("Tangent", "tangents", ("points", "normals", "uvs")),
    ("Binormal", "binormals", ("points", "normals", "uvs")),
    ("Smoothing", "smoothing groups", ("normals",)),
)


def _stale_elements(edit, shape, label: str, moved: str, why: str) -> None:
    """the tangents, binormals and smoothing groups of a mesh a change of
    its ``moved`` (points, normals or uvs) leaves stale, refused or with
    ``force`` dropped"""
    for kind, plural, causes in _DERIVED:
        if moved not in causes or not getattr(shape, f"GetElement{kind}Count")():
            continue

        def remove(kind=kind):
            while getattr(shape, f"GetElement{kind}Count")():
                getattr(shape, f"RemoveElement{kind}")(getattr(shape, f"GetElement{kind}")(0))

        _unread(edit, f"the {plural} of {label}", why, remove)


def _set_points(edit, shape, old, new, label: str) -> None:
    """the points of a mesh that moved, in the file's unit; each blend shape
    target follows, its offsets kept"""
    fbx   = edit.fbx
    moved = np.flatnonzero(np.any(np.asarray(old.points) != np.asarray(new.points), axis=1))
    if not moved.size:
        return
    targets = _targets(edit, shape)
    _stale_targets(edit, targets, label, f"{label} is given other points")
    rows = (np.asarray(new.points, dtype=np.float64)[moved] / edit.cm).tolist()
    for at, row in zip(moved.tolist(), rows):
        held = shape.GetControlPointAt(at)
        for _, _, _, target in targets:
            point = target.GetControlPointAt(at)
            target.SetControlPointAt(
                fbx.FbxVector4(*[row[k] + (point[k] - held[k]) for k in range(3)], point[3]), at
            )
        shape.SetControlPointAt(fbx.FbxVector4(*row, held[3]), at)


def _set_normals(edit, shape, new) -> None:
    """the normals of a mesh as its record shades it: the file's first
    normal element written anew, one a face corner through their indices;
    a mesh storing none keeps an element the file holds, of the normals
    derived from its points, and is given none where the file holds none"""
    from cgmath.geometry.mesh import _corner_normals

    fbx  = edit.fbx
    held = shape.GetElementNormalCount()
    normals, ids = _corner_normals(new, always=bool(held), force=True)
    if normals is None:
        return
    if held:
        element = shape.GetElementNormal(0)
    else:
        # on the first layer by hand, as save_fbx places a colour set
        if shape.GetLayerCount() == 0:
            shape.CreateLayer()
        element = fbx.FbxLayerElementNormal.Create(shape, "")
        shape.GetLayer(0).SetNormals(element)
    element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
    element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
    _filled(element.GetDirectArray(), [fbx.FbxVector4(*row, 0.0) for row in np.asarray(normals, dtype=np.float64).tolist()])
    _filled(element.GetIndexArray(), np.asarray(ids, dtype=np.int64).tolist())


def _set_colors(edit, shape, new, label: str) -> None:
    """the colours of a mesh: the file's first colour set written anew, one
    a vertex or a face corner as the record holds them; a set made for a
    mesh holding none, named as save_fbx names it; the set taken off a mesh
    given none, the file's other sets with it (``force``)"""
    from cgmath.formats._write import COLOR_SET
    from cgmath.geometry.mesh import _corner_colors

    fbx     = edit.fbx
    mapping = fbx.FbxLayerElement.EMappingMode
    held    = [shape.GetElementVertexColor(k) for k in range(shape.GetElementVertexColorCount())]
    mapped  = [element for element in held if element.GetMappingMode() != mapping.eNone]
    colors, ids = _corner_colors(new)
    if colors is None:
        if len(mapped) > 1:
            names = ", ".join(repr(str(element.GetName())) for element in mapped[1:])
            _unread(edit, f"the colour sets {names} of {label}", f"{label} is given no colours", lambda: None)
        for element in mapped:
            shape.RemoveElementVertexColor(element)
        return
    if mapped:
        element = mapped[0]
    else:
        if shape.GetLayerCount() == 0:
            shape.CreateLayer()
        element = fbx.FbxLayerElementVertexColor.Create(shape, COLOR_SET)
        shape.GetLayer(0).SetVertexColors(element)
    per_vertex = new.color_indices is None
    element.SetMappingMode(mapping.eByControlPoint if per_vertex else mapping.eByPolygonVertex)
    element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
    _filled(element.GetDirectArray(), [fbx.FbxColor(*row) for row in np.asarray(colors, dtype=np.float64).tolist()])
    _filled(element.GetIndexArray(), range(len(colors)) if per_vertex else np.asarray(ids, dtype=np.int64).tolist())


def _write_meshes(edit, was, now, base: dict, state: dict) -> None:
    """each mesh's change written into its FbxMesh: its name, the points
    that moved, its normals and its colours. One given other faces or
    another count of points is another mesh: ValueError"""
    from cgmath.formats.fbx import _ascii_text

    for old, new, node in _paired(edit, was, now, "meshes"):
        shape = _carried(edit, node, "FbxMesh")
        label = _label(new, node)
        if (
            np.shape(old.points) != np.shape(new.points)
            or not _same(old.counts, new.counts)
            or not _same(old.indices, new.indices)
        ):
            raise ValueError(
                f"FbxFile.save: {label} is given other faces or another count of points than the file's; {_ELSE}"
            )
        if old.name != new.name:
            shape.SetName(_ascii_text(new.name or ""))
        moved  = not _same(old.points, new.points)
        shaded = not all(_same(getattr(old, name), getattr(new, name)) for name in ("normals", "normal_indices", "hard_edges"))
        if moved:
            _stale_elements(edit, shape, label, "points", f"{label} is given other points")
            _set_points(edit, shape, old, new, label)
        if shaded:
            _stale_elements(edit, shape, label, "normals", f"{label} is given other normals")
        if shaded or (moved and new.normals is None):
            _set_normals(edit, shape, new)
        if not (_same(old.colors, new.colors) and _same(old.color_indices, new.color_indices)):
            _set_colors(edit, shape, new, label)


_WRITERS["meshes"] = _write_meshes


def _uv_elements(edit, base: dict) -> list:
    """the file's UV sets in the reader's order, a set of no UVs left out:
    [(FbxMesh, what a message calls its mesh, the element)]"""
    found  = []
    meshes = _held(base["meshes"])
    for mesh in meshes.list if meshes is not None else ():
        node  = edit.nodes[mesh.parent_uuid]
        shape = _carried(edit, node, "FbxMesh")
        for place in range(shape.GetElementUVCount()):
            element = shape.GetElementUV(place)
            if element is not None and element.GetDirectArray().GetCount():
                found.append((shape, _label(mesh, node), element))
    return found


def _write_uvs(edit, was, now, base: dict, state: dict) -> None:
    """each UV set's change written into its element: its name, its UVs
    and their indices, one a face corner. A set of another mesh or of
    other faces than the file's: ValueError"""
    from cgmath.formats.fbx import _ascii_text

    fbx      = edit.fbx
    elements = _uv_elements(edit, base)
    if len(elements) != len(was.list):
        raise ValueError(f"FbxFile.save: the file's UV sets are not the ones read; {_ELSE}")
    for at, (old, new, (shape, label, element)) in enumerate(zip(was.list, now.list, elements)):
        if old.mesh_uuid != new.mesh_uuid:
            raise ValueError(f"FbxFile.save: uvs[{at}] is of another mesh than the file's; {_ELSE}")
        if not _same(old.counts, new.counts) or np.shape(old.indices) != np.shape(new.indices):
            raise ValueError(f"FbxFile.save: uvs[{at}] is of other faces than the file's; {_ELSE}")
        if old.name != new.name:
            element.SetName(_ascii_text(new.name or ""))
        if _same(old.points, new.points) and _same(old.indices, new.indices):
            continue
        _stale_elements(edit, shape, label, "uvs", f"{label} is given other UVs")
        element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
        element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
        _filled(element.GetDirectArray(), [fbx.FbxVector2(*row) for row in np.asarray(new.points, dtype=np.float64).tolist()])
        _filled(element.GetIndexArray(), np.asarray(new.indices, dtype=np.int64).tolist())


_WRITERS["uvs"] = _write_uvs


def _linked(edit, deformer) -> list:
    """the clusters of a skin by the node each links, in the reader's
    order: [[the clusters of one node]], a cluster linking none left out"""
    order, groups = [], {}
    for k in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(k)
        link    = cluster.GetLink()
        if link is None:
            continue
        key = link.GetUniqueID()
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(cluster)
    return [groups[key] for key in order]


def _set_cluster(cluster, column) -> None:
    """a cluster weighing the points ``column`` weighs other than 0"""
    rows = np.flatnonzero(column)
    cluster.SetControlPointIWCount(0)
    for vertex, weight in zip(rows.tolist(), np.asarray(column, dtype=np.float64)[rows].tolist()):
        cluster.AddControlPointIndex(vertex, weight)


def _write_skins(edit, was, now, base: dict, state: dict) -> None:
    """each skin's change written into its FbxSkin: its name, the weights
    of the influences that changed, influences taken off, their order. An
    influence the file's skin does not hold: ValueError"""
    from cgmath.formats.fbx import _ascii_text

    fbx    = edit.fbx
    kind   = fbx.FbxDeformer.EDeformerType.eSkin
    shapes = _mesh_shapes(edit, state)
    for at, (old, new) in enumerate(zip(was.list, now.list)):
        if old.mesh_uuid != new.mesh_uuid or new.mesh_uuid not in shapes:
            raise ValueError(f"FbxFile.save: skins[{at}] is of another mesh than the file's; {_ELSE}")
        shape, label = shapes[new.mesh_uuid]
        deformer = edit.typed(shape.GetDeformer(0, kind), fbx.FbxSkin, fbx)
        groups   = _linked(edit, deformer)
        before   = np.asarray(old.weights)
        after    = np.asarray(new.weights)
        if len(groups) != len(old.influences) or len(after) != len(before):
            raise ValueError(f"FbxFile.save: skins[{at}] weighs other points than the file's skin of {label}; {_ELSE}")
        if len(set(new.influences)) != len(new.influences):
            raise ValueError(f"FbxFile.save: skins[{at}] names an influence twice")
        if old.name != new.name:
            deformer.SetName(_ascii_text(new.name or ""))
        # a node the save renames is named as the hierarchy now names it
        named = [str(group[0].GetLink().GetName()) for group in groups]
        kept  = []
        for column, influence in enumerate(new.influences):
            if influence in named:
                place = named.index(influence)
            elif influence in old.influences:
                raise ValueError(
                    f"FbxFile.save: skins[{at}] names the influence {influence!r}, a node the save renames "
                    f"{named[old.influences.index(influence)]!r}; name it so"
                )
            else:
                raise ValueError(
                    f"FbxFile.save: skins[{at}] names the influence {influence!r}, which the file's skin of {label} "
                    f"does not hold; {_ELSE}"
                )
            kept.append(place)
            if not np.array_equal(before[:, place], after[:, column]):
                _set_cluster(groups[place][0], after[:, column])
                for cluster in groups[place][1:]:
                    cluster.SetControlPointIWCount(0)
        if kept == list(range(len(groups))):
            continue
        for group in groups:
            for cluster in group:
                deformer.RemoveCluster(cluster)
        for place in kept:
            for cluster in groups[place]:
                deformer.AddCluster(cluster)
        for place in sorted(set(range(len(groups))) - set(kept)):
            for cluster in groups[place]:
                cluster.Destroy()


_WRITERS["skins"] = _write_skins


def _set_target(edit, shape, target, morph) -> None:
    """a blend shape target at its mesh's points moved by its record's
    offsets, in the file's unit; the points it holds already are left"""
    from cgmath.geometry.mesh import _fbx_control_points

    fbx    = edit.fbx
    points = _fbx_control_points(shape)
    held   = _fbx_control_points(target)
    points[np.asarray(morph.indices, dtype=np.int64)] += np.asarray(morph.offsets, dtype=np.float64) / edit.cm
    for row in np.flatnonzero(np.any(points != held, axis=1)).tolist():
        target.SetControlPointAt(fbx.FbxVector4(*points[row].tolist(), target.GetControlPointAt(row)[3]), row)


def _one(held: list, field: str, what: str):
    """the one value the records of ``held``, ``[(place, old, new, ...)]``,
    give ``field``; ValueError for two, which one ``what`` of a file does
    not hold"""
    values = []
    for entry in held:
        value = getattr(entry[2], field)
        if value not in values:
            values.append(value)
    if len(values) > 1:
        places = ", ".join(f"morphs[{entry[0]}]" for entry in held)
        raise ValueError(
            f"FbxFile.save: {places} are the targets of one {what} of the file and are given two values of {field}, "
            f"{values[0]!r} and {values[1]!r}; a {what} holds one; {_ELSE}"
        )
    return values[0]


def _write_morphs(edit, was, now, base: dict, state: dict) -> None:
    """each blend shape target's change written into its FbxShape, its
    channel and its deformer: its name, its deformer's, its weight, the
    weight it is reached at and its offsets. A target of another mesh than
    the file's, or targets of one channel put in another order: ValueError"""
    from cgmath.formats._load import WEIGHT_UNIT
    from cgmath.formats.fbx import _ascii_text

    shapes = _mesh_shapes(edit, state)
    meshes = _held(state["meshes"])
    found  = []
    for mesh in meshes.list if meshes is not None else ():
        if mesh.uuid in shapes:
            shape, label = shapes[mesh.uuid]
            found += [(mesh.uuid, shape, label) + entry for entry in _targets(edit, shape)]
    if len(found) != len(was.list):
        raise ValueError(f"FbxFile.save: the file's blend shape targets are not the ones read; {_ELSE}")

    channels, deformers = {}, {}
    for at, (old, new, entry) in enumerate(zip(was.list, now.list, found)):
        uuid, shape, label, deformer, channel, _, target = entry
        if old.mesh_uuid != new.mesh_uuid or new.mesh_uuid != uuid:
            raise ValueError(f"FbxFile.save: morphs[{at}] is of another mesh than the file's; {_ELSE}")
        channels.setdefault(channel.GetUniqueID(), (channel, deformer, []))[2].append((at, old, new, target))
        deformers.setdefault(deformer.GetUniqueID(), (deformer, []))[1].append((at, old, new))
        if not (_same(old.offsets, new.offsets) and _same(old.indices, new.indices)):
            _stale_targets(edit, [entry[3:]], label, f"morphs[{at}] is given other offsets")
            _set_target(edit, shape, target, new)

    for channel, deformer, held in channels.values():
        if any(old.name != new.name for _, old, new, _ in held):
            name   = _one(held, "name", "channel")
            dname  = str(deformer.GetName())
            prefix = f"{dname}." if dname and str(channel.GetName()).startswith(dname + ".") else ""
            channel.SetName(_ascii_text(prefix + name))
            for _, _, new, target in held:
                reach = float(new.at)
                target.SetName(_ascii_text(name if reach == 1.0 else f"{name}_{round(reach * WEIGHT_UNIT):g}"))
        if any(float(old.weight) != float(new.weight) for _, old, new, _ in held):
            channel.DeformPercent.Set(float(_one(held, "weight", "channel")) * WEIGHT_UNIT)
        if any(float(old.at) != float(new.at) for _, old, new, _ in held):
            reached = [float(new.at) for _, _, new, _ in held]
            if any(b <= a for a, b in zip(reached, reached[1:])):
                places = ", ".join(f"morphs[{entry[0]}]" for entry in held)
                raise ValueError(
                    f"FbxFile.save: {places}, the targets of one channel of the file, are reached at {reached}, "
                    f"another order than the file's; {_ELSE}"
                )
            # the channel's targets taken off and put back, each at its weight
            for _, _, _, target in held:
                channel.RemoveTargetShape(target)
            for (_, _, _, target), reach in zip(held, reached):
                channel.AddTargetShape(target, reach * WEIGHT_UNIT)

    for deformer, held in deformers.values():
        if all(old.deformer == new.deformer for _, old, new in held):
            continue
        before = str(deformer.GetName())
        after  = _one(held, "deformer", "deformer") or ""
        deformer.SetName(_ascii_text(after))
        for c in range(deformer.GetBlendShapeChannelCount()):
            channel = deformer.GetBlendShapeChannel(c)
            cname   = str(channel.GetName())
            if before and cname.startswith(before + "."):
                channel.SetName(_ascii_text((f"{after}." if after else "") + cname[len(before) + 1:]))


_WRITERS["morphs"] = _write_morphs


def _keyed(edit, prop) -> bool:
    """a take of the file keys ``prop``"""
    fbx      = edit.fbx
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
    for stack in edit.stacks():
        for at in range(stack.GetMemberCount(criteria)):
            layer = edit.typed(stack.GetMember(criteria, at), fbx.FbxAnimLayer, fbx)
            held  = prop.GetCurveNode(layer, False)
            if held is not None and any(held.GetCurveCount(c) for c in range(held.GetChannelsCount())):
                return True
    return False


def _given(edit, prop, value, place: str, field: str) -> None:
    """``prop`` holding ``value``; ValueError where a take keys it: the
    reader takes the take's word, and no record holds those keys"""
    if _keyed(edit, prop):
        raise ValueError(
            f"FbxFile.save: {place} is given another {field}, which a take of the file keys, and no record holds "
            f"those keys; {_ELSE}"
        )
    prop.Set(value)


def _moved(old, new, *fields) -> bool:
    """one of ``fields`` differs between two records"""
    return any(not _same(getattr(old, field), getattr(new, field)) for field in fields)


_FITS = {"fill": "eFitFill", "horizontal": "eFitHorizontal", "vertical": "eFitVertical", "overscan": "eFitOverscan"}


def _write_cameras(edit, was, now, base: dict, state: dict) -> None:
    """each camera's change written into its FbxCamera: its name, lens,
    film back, fit, offsets, squeeze, clip planes in the file's unit, and
    its projection. A lens or film back given to a file that holds the view
    makes the lens the file's word (``ApertureMode``). A value a take keys:
    ValueError"""
    from cgmath.formats._load import ORTHO_UNIT
    from cgmath.formats._write import _enum
    from cgmath.formats.fbx import _ascii_text

    fbx = edit.fbx
    for at, (old, new, node) in enumerate(_paired(edit, was, now, "cameras")):
        shape = _carried(edit, node, "FbxCamera")
        place = f"cameras[{at}]"
        if old.name != new.name:
            shape.SetName(_ascii_text(new.name or ""))
        if _moved(old, new, "focal_length", "horizontal_film_aperture", "vertical_film_aperture"):
            if _keyed(edit, shape.FieldOfView) or _keyed(edit, shape.FieldOfViewX):
                raise ValueError(
                    f"FbxFile.save: {place} is given another lens, and a take of the file keys its field of view, "
                    f"which no record holds; {_ELSE}"
                )
            if float(shape.FocalLength.Get()) != float(new.focal_length):
                _given(edit, shape.FocalLength, float(new.focal_length), place, "focal_length")
            shape.ApertureMode.Set(_enum(fbx.FbxCamera, "EApertureMode", "eFocalLength"))
        if _moved(old, new, "horizontal_film_aperture", "vertical_film_aperture"):
            width, height = float(new.horizontal_film_aperture), float(new.vertical_film_aperture)
            _given(edit, shape.FilmWidth, width, place, "horizontal_film_aperture")
            _given(edit, shape.FilmHeight, height, place, "vertical_film_aperture")
            shape.FilmAspectRatio.Set(width / height)
        if _moved(old, new, "film_fit"):
            shape.GateFit.Set(_enum(fbx.FbxCamera, "EGateFit", _FITS[new.film_fit]))
        if _moved(old, new, "film_offset"):
            _given(edit, shape.FilmOffsetX, float(new.film_offset[0]), place, "film_offset")
            _given(edit, shape.FilmOffsetY, float(new.film_offset[1]), place, "film_offset")
        if _moved(old, new, "lens_squeeze_ratio"):
            _given(edit, shape.FilmSqueezeRatio, float(new.lens_squeeze_ratio), place, "lens_squeeze_ratio")
        if _moved(old, new, "near_clip_plane"):
            _given(edit, shape.NearPlane, float(new.near_clip_plane) / edit.cm, place, "near_clip_plane")
        if _moved(old, new, "far_clip_plane"):
            _given(edit, shape.FarPlane, float(new.far_clip_plane) / edit.cm, place, "far_clip_plane")
        if _moved(old, new, "orthographic"):
            kind = "eOrthogonal" if new.orthographic else "ePerspective"
            shape.ProjectionType.Set(_enum(fbx.FbxCamera, "EProjectionType", kind))
        if new.orthographic and _moved(old, new, "orthographic", "orthographic_width"):
            width = float(new.orthographic_width) / edit.cm / ORTHO_UNIT
            _given(edit, shape.OrthoZoom, width, place, "orthographic_width")


_WRITERS["cameras"] = _write_cameras

_DECAYS = {0: "eNone", 1: "eLinear", 2: "eQuadratic", 3: "eCubic"}


def _write_lights(edit, was, now, base: dict, state: dict) -> None:
    """each light's change written into its FbxLight: its name, colour,
    intensity, decay, and a spot's cone and penumbra. A light given another
    kind, which may shine down another axis of its node, or a value a take
    keys: ValueError"""
    from cgmath.formats._load import INTENSITY_UNIT
    from cgmath.formats._write import _enum
    from cgmath.formats.fbx import _ascii_text

    fbx = edit.fbx
    for at, (old, new, node) in enumerate(_paired(edit, was, now, "lights")):
        shape = _carried(edit, node, "FbxLight")
        place = f"lights[{at}]"
        if old.kind != new.kind:
            raise ValueError(
                f"FbxFile.save: {place} is a {old.kind} light given the kind {new.kind!r}; a light keeps its kind in "
                f"an open file; {_ELSE}"
            )
        if old.name != new.name:
            shape.SetName(_ascii_text(new.name or ""))
        if _moved(old, new, "color"):
            _given(edit, shape.Color, fbx.FbxDouble3(*[float(x) for x in new.color]), place, "color")
        if _moved(old, new, "intensity"):
            _given(edit, shape.Intensity, float(new.intensity) * INTENSITY_UNIT, place, "intensity")
        if _moved(old, new, "decay_rate"):
            shape.DecayType.Set(_enum(fbx.FbxLight, "EDecayType", _DECAYS[int(new.decay_rate)]))
        if new.kind == "spot" and _moved(old, new, "cone_angle", "penumbra_angle"):
            cone = float(new.cone_angle)
            _given(edit, shape.InnerAngle, cone, place, "cone_angle")
            _given(edit, shape.OuterAngle, cone + 2.0 * float(new.penumbra_angle), place, "penumbra_angle")


_WRITERS["lights"] = _write_lights


class _Scribe:
    """what the shape writers of ``_write`` ask of an exporter, over the
    scene of an open file"""

    def __init__(self, edit):
        self.manager = edit.file._manager
        self.scene   = edit.scene

    def _get_acsii_type(self):
        from cgmath.formats.fbx import FbxExporter

        return FbxExporter._get_acsii_type(self)


def _in_file(edit, shape):
    """a copy of a curve or surface, its points in the file's unit"""
    held        = shape.copy()
    held.points = np.asarray(shape.points, dtype=np.float64) / edit.cm
    return held


def _rows(shape) -> np.ndarray:
    """(..., 4): a curve's or surface's points, each with its weight, 1 for
    a shape that holds none"""
    points  = np.asarray(shape.points, dtype=np.float64)
    weights = np.ones(points.shape[:-1]) if shape.weights is None else np.asarray(shape.weights, dtype=np.float64)
    return np.concatenate([points, weights[..., None]], axis=-1)


def _set_rows(edit, shape, was, now, places, weighed: bool = True) -> None:
    """the control points of ``shape`` at ``places`` holding the rows of
    ``now``, x, y, z and a weight, in the file's unit; a row that is its
    ``was`` is left as the file holds it, and so is each weight of a shape
    that is not ``weighed``"""
    fbx = edit.fbx
    was = np.asarray(was, dtype=np.float64).reshape(-1, 4)
    now = np.asarray(now, dtype=np.float64).reshape(-1, 4)
    for row in np.flatnonzero(np.any(was != now, axis=1)).tolist():
        place  = int(places[row])
        weight = float(now[row, 3]) if weighed else shape.GetControlPointAt(place)[3]
        shape.SetControlPointAt(fbx.FbxVector4(*(now[row, :3] / edit.cm).tolist(), weight), place)


def _rebuilt(edit, node, kind: str, why: str, write) -> None:
    """the shape of ``node`` made again by ``write()``, under the name of
    the file's own, which is taken off its node and freed: what it held
    beyond the record's fields goes with it, so ValueError without
    ``force``"""
    old  = node.GetNodeAttribute()
    name = str(old.GetName())
    what = f"the settings of the {kind} on node {str(node.GetName())!r} beyond its points, knots and form"
    _unread(edit, what, f"{why}, which makes its shape anew", lambda: None)
    write()
    made = node.GetNodeAttribute()
    if made.GetUniqueID() != old.GetUniqueID():
        node.RemoveNodeAttribute(old)
        old.Destroy()
    made.SetName(name)


def _line_places(shape) -> list:
    """the control points each line of an FbxLine runs through, as the
    reader takes them"""
    indices = [int(shape.GetPointIndexAt(i)) for i in range(shape.GetIndexArraySize())]
    indices = indices or list(range(shape.GetControlPointsCount()))
    ends    = sorted(int(shape.GetEndPointAt(i)) for i in range(shape.GetEndPointCount()))
    if indices and (not ends or ends[-1] != len(indices) - 1):
        ends.append(len(indices) - 1)
    found, start = [], 0
    for end in ends:
        found.append(indices[start : end + 1])
        start = end + 1
    return found


def _curve_kept(old, new) -> bool:
    """a curve keeps its degree, form, knots and point count"""
    return (
        old.degree == new.degree
        and bool(old.periodic) == bool(new.periodic)
        and _same(old.knots, new.knots)
        and np.shape(old.points) == np.shape(new.points)
    )


def _write_curves(edit, was, now, base: dict, state: dict) -> None:
    """each curve's change written into the shape of its node: points and
    weights moved in place, in the file's unit; another degree, form, knots
    or point count makes the shape again as save_fbx writes it, which loses
    what the file's shape held beyond them, so it takes ``force``"""
    from cgmath.formats._write import _is_line, _write_lines, _write_nurbs

    fbx    = edit.fbx
    kinds  = fbx.FbxNodeAttribute.EType
    groups = {}
    for at, (old, new, node) in enumerate(_paired(edit, was, now, "curves")):
        groups.setdefault(new.parent_uuid, (node, []))[1].append((at, old, new))

    for node, held in groups.values():
        lined  = node.GetNodeAttribute().GetAttributeType() == kinds.eLine
        places = ", ".join(f"curves[{at}]" for at, _, _ in held)
        kept   = all(_curve_kept(old, new) for _, old, new in held)
        if kept and lined and all(_is_line(new) for _, _, new in held):
            shape = _carried(edit, node, "FbxLine")
            lines = _line_places(shape)
            if [len(line) for line in lines] == [len(new.points) for _, _, new in held]:
                for line, (_, old, new) in zip(lines, held):
                    _set_rows(edit, shape, _rows(old), _rows(new), line, weighed=False)
                continue
        elif kept and not lined and len(held) == 1:
            _, old, new = held[0]
            shape = _carried(edit, node, "FbxNurbsCurve")
            count = int(shape.GetControlPointsCount())
            # a closed form leaves out its last point, the first again
            if count in (len(new.points), len(new.points) - 1):
                _set_rows(edit, shape, _rows(old)[:count], _rows(new)[:count], range(count))
                continue

        shapes = [_in_file(edit, new) for _, _, new in held]
        why    = f"{places} is given another degree, form, knots or point count"
        if all(_is_line(shape) for shape in shapes):
            _rebuilt(edit, node, "curve", why, lambda: _write_lines(_Scribe(edit), shapes, node, fbx))
        elif len(shapes) == 1:
            _rebuilt(edit, node, "curve", why, lambda: _write_nurbs(_Scribe(edit), [("curve", shapes[0], node)], fbx))
        else:
            raise ValueError(
                f"FbxFile.save: {places} sit on one node, which holds them as the lines of one shape, each of degree "
                f"1 holding no knots and no weights; {_ELSE}"
            )


_WRITERS["curves"] = _write_curves


def _surface_kept(old, new) -> bool:
    """a surface keeps its degrees, forms, knots and point counts"""
    return (
        old.degree_u == new.degree_u
        and old.degree_v == new.degree_v
        and bool(old.periodic_u) == bool(new.periodic_u)
        and bool(old.periodic_v) == bool(new.periodic_v)
        and _same(old.knots_u, new.knots_u)
        and _same(old.knots_v, new.knots_v)
        and np.shape(old.points) == np.shape(new.points)
    )


def _write_surfaces(edit, was, now, base: dict, state: dict) -> None:
    """each surface's change written into the FbxNurbsSurface of its node:
    points and weights moved in place, in the file's unit, the file's point
    ``v * count_u + u`` ours ``[u][v]``; another degree, form, knots or
    point count makes the shape again as save_fbx writes it, which loses
    what the file's shape held beyond them, so it takes ``force``"""
    from cgmath.formats._write import _write_nurbs

    fbx = edit.fbx
    for at, (old, new, node) in enumerate(_paired(edit, was, now, "surfaces")):
        if _surface_kept(old, new):
            shape = _carried(edit, node, "FbxNurbsSurface")
            u, v = int(shape.GetUCount()), int(shape.GetVCount())
            held = np.shape(new.points)
            # a closed form leaves out its last row, the first again
            if u in (held[0], held[0] - 1) and v in (held[1], held[1] - 1):
                places = (np.arange(v)[None, :] * u + np.arange(u)[:, None]).reshape(-1)
                _set_rows(edit, shape, _rows(old)[:u, :v], _rows(new)[:u, :v], places)
                continue

        held = _in_file(edit, new)
        why  = f"surfaces[{at}] is given another degree, form, knots or point count"
        _rebuilt(edit, node, "surface", why, lambda: _write_nurbs(_Scribe(edit), [("surface", held, node)], fbx))


_WRITERS["surfaces"] = _write_surfaces


_SHADES = (
    "color",
    "diffuse",
    "ambient_color",
    "incandescence",
    "transparency",
    "specular_color",
    "cosine_power",
    "reflectivity",
    "bump_depth",
)
_LOOKS = ("uv_set", "wrap_u", "wrap_v")
_MAPS  = ("texture", "texture_file", "normal_map", "normal_map_file", "normal_map_kind")


def _shaders(edit, state: dict) -> list:
    """what the file holds for each material record, in the records' order:
    (its FbxSurfaceMaterial, what the reader makes of it, its place among
    its node's materials, its mesh's FbxMesh)"""
    from cgmath.formats._load import _read_materials

    fbx    = edit.fbx
    found  = []
    meshes = _held(state["meshes"])
    for mesh in meshes.list if meshes is not None else ():
        node = edit.nodes.get(mesh.parent_uuid)
        if node is None:
            continue
        shape  = _carried(edit, node, "FbxMesh")
        places = {}
        for at in range(node.GetMaterialCount()):
            places.setdefault(int(node.GetMaterial(at).GetUniqueID()), at)
        label = _label(mesh, node)
        for entry in _read_materials(node, shape, mesh.counts, label, edit.file._filename, fbx, pixels=False):
            at = places[entry["id"]]
            found.append((edit.typed(node.GetMaterial(at), fbx.FbxSurfaceMaterial, fbx), entry, at, shape))
    return found


def _tinted(edit, colour, factor, value, place: str, field: str) -> None:
    """a colour the reader multiplies by its factor, holding ``value`` over
    the file's factor, which a 0 turns 1"""
    scale = float(factor.Get()) or 1.0
    _given(edit, factor, scale, place, field)
    held = np.asarray(value, dtype=np.float64) / scale
    _given(edit, colour, edit.fbx.FbxDouble3(*held.tolist()), place, field)


def _looked(edit, shader, new, fields) -> None:
    """a material's UV set and wraps written into its maps: the one the
    reader takes them from, its colour map, else its normal map, and its
    other map where that held the same"""
    from cgmath.formats._load import COLOUR_SLOT, DEFAULT_UV_SET, _enum_int, _file_texture
    from cgmath.formats._write import _enum
    from cgmath.formats.fbx import _ascii_text

    fbx  = edit.fbx
    maps = []
    for slot in (COLOUR_SLOT, "Bump", "NormalMap"):
        prop = shader.FindProperty(slot)
        if prop.IsValid() and prop.GetSrcObjectCount():
            file = _file_texture(prop, fbx)[0]
            if file is not None:
                maps.append(file)
    if not maps:
        return
    wraps = {mode: _enum(fbx.FbxTexture, "EWrapMode", name) for mode, name in (("repeat", "eRepeat"), ("clamp", "eClamp"))}
    for field in fields:
        value = getattr(new, field)
        name  = {"uv_set": "UVSet", "wrap_u": "WrapModeU", "wrap_v": "WrapModeV"}[field]
        if field == "uv_set":
            value = fbx.FbxString(_ascii_text(value or DEFAULT_UV_SET))
            read  = lambda texture: str(getattr(texture, name).Get())
        else:
            value = wraps[value]
            read  = lambda texture: _enum_int(getattr(texture, name).Get())
        before = read(maps[0])
        for texture in maps:
            if read(texture) == before:
                getattr(texture, name).Set(value)


def _map_made(edit, record, pixels, file, fallback):
    """an FbxFileTexture of ``pixels``, written as a PNG the export embeds,
    as ``save_fbx`` makes a material's map: reading the record's UV set,
    wrapping as it says"""
    from cgmath.formats._write import _enum, _write_png

    fbx     = edit.fbx
    path    = _write_png(pixels, file, fallback, edit.texture_folder(), edit.taken)
    texture = fbx.FbxFileTexture.Create(edit.scene, os.path.splitext(os.path.basename(path))[0])
    texture.SetFileName(path)
    texture.SetRelativeFileName(os.path.basename(path))
    texture.SetTextureUse(fbx.FbxTexture.ETextureUse.eStandard)
    texture.SetMappingType(fbx.FbxTexture.EMappingType.eUV)
    texture.SetMaterialUse(fbx.FbxFileTexture.EMaterialUse.eModelMaterial)
    if record.uv_set:
        texture.UVSet.Set(fbx.FbxString(record.uv_set))
    texture.WrapModeU.Set(_enum(fbx.FbxTexture, "EWrapMode", "eRepeat" if record.wrap_u == "repeat" else "eClamp"))
    texture.WrapModeV.Set(_enum(fbx.FbxTexture, "EWrapMode", "eRepeat" if record.wrap_v == "repeat" else "eClamp"))
    return texture


def _map_off(edit, prop, place: str) -> None:
    """the file texture feeding ``prop`` taken off it, and off the file
    with the media it was read from once nothing else reads them. A map
    that is no plain file texture (layered, procedural) is refused"""
    from cgmath.formats._load import _file_texture

    fbx = edit.fbx
    texture, other = _file_texture(prop, fbx)
    if other or (texture is None and prop.GetSrcObjectCount()):
        raise ValueError(
            f"FbxFile.save: {place} is given another map where the file's is no plain file texture; {_ELSE}"
        )
    if texture is None:
        return
    if not prop.DisconnectSrcObject(texture):
        raise ValueError(f"FbxFile.save: {place} is given another map where the file's is layered; {_ELSE}")
    if texture.GetDstPropertyCount() or texture.GetDstObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId)):
        return
    videos = fbx.FbxCriteria.ObjectType(fbx.FbxVideo.ClassId)
    clips  = [texture.GetSrcObject(videos, at) for at in range(texture.GetSrcObjectCount(videos))]
    texture.Destroy()
    for clip in clips:
        if clip is not None and not clip.GetDstObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId)):
            clip.Destroy()


def _map_given(edit, prop, record, pixels, file, fallback, place: str) -> None:
    """``prop`` fed the image ``pixels``: the file texture it holds reads
    the new file, its own settings kept, the media it was read from taken
    off; a slot holding none is given a texture made as ``save_fbx`` makes
    one"""
    from cgmath.formats._load import _file_texture
    from cgmath.formats._write import _write_png

    fbx = edit.fbx
    texture, other = _file_texture(prop, fbx)
    if other:
        raise ValueError(f"FbxFile.save: {place} is given another map where the file's is no plain file texture; {_ELSE}")
    if texture is None:
        made = _map_made(edit, record, pixels, file, fallback)
        prop.ConnectSrcObject(made)
        return os.path.basename(str(made.GetFileName()))
    path = _write_png(pixels, file, fallback, edit.texture_folder(), edit.taken)
    texture.SetFileName(path)
    texture.SetRelativeFileName(os.path.basename(path))
    videos = fbx.FbxCriteria.ObjectType(fbx.FbxVideo.ClassId)
    for at in reversed(range(texture.GetSrcObjectCount(videos))):
        clip = texture.GetSrcObject(videos, at)
        texture.DisconnectSrcObject(clip)
        if not clip.GetDstObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId)):
            clip.Destroy()
    return os.path.basename(path)


def named_as_written(edit, records, names: dict) -> None:
    """each of ``records`` given the file names its maps were written
    under, {field: name}: a map is written as a PNG, named after the
    record's file name, its material's, or apart from another of that
    name, and reads back under that name. ``edit.undo`` takes the names the
    records held"""
    for record in records:
        for field, name in names.items():
            if getattr(record, field) != name:
                edit.undo.append((record, field, getattr(record, field)))
                setattr(record, field, name)


def _remapped(edit, shader, old, new, place: str) -> dict:
    """a material's colour map and normal map as its record holds them: an
    image given in the place of another written into the file's texture, a
    map the record drops taken off, one it brings made, a normal map of
    another kind moved to that kind's slot (NormalMap for tangent space
    normals, Bump for a bump map)"""
    from cgmath.formats._load import COLOUR_SLOT

    def slot(kind: str):
        return shader.FindProperty("NormalMap" if kind == "tangent" else "Bump")

    names  = {}
    colour = shader.FindProperty(COLOUR_SLOT)
    if _moved(old, new, "texture", "texture_file"):
        if new.texture is None:
            _map_off(edit, colour, place)
        else:
            names["texture_file"] = _map_given(edit, colour, new, new.texture, new.texture_file, new.name, place)
    if _moved(old, new, "normal_map", "normal_map_file", "normal_map_kind"):
        fallback = f"{new.name or 'texture'}_normal"
        if old.normal_map is not None and (new.normal_map is None or old.normal_map_kind != new.normal_map_kind):
            _map_off(edit, slot(old.normal_map_kind), place)
        if new.normal_map is not None:
            names["normal_map_file"] = _map_given(
                edit, slot(new.normal_map_kind), new, new.normal_map, new.normal_map_file, fallback, place
            )
    return names


def _refaced(edit, shape, records: list) -> None:
    """the faces of a mesh's materials written into its material element:
    each face given to another material names that one's place on the
    node; ``records`` [(the record's place in materials=, its material's
    place on the node, the file's record, the one to save)]"""
    fbx     = edit.fbx
    mapping = fbx.FbxLayerElement.EMappingMode
    found   = [shape.GetElementMaterial(k) for k in range(shape.GetElementMaterialCount())]
    element = next((e for e in found if e.GetMappingMode() != mapping.eNone), None)
    whole   = any(old.faces is None or new.faces is None for _, _, old, new in records)
    if whole or element is None or element.GetMappingMode() != mapping.eByPolygon:
        at = next(at for at, _, old, new in records if _moved(old, new, "faces"))
        raise ValueError(
            f"FbxFile.save: materials[{at}] is given other faces, and a material covering its whole mesh keeps it in "
            f"an open file; {_ELSE}"
        )
    indices = element.GetIndexArray()
    for _, index, old, new in records:
        for face in np.setdiff1d(new.faces, old.faces).tolist():
            indices.SetAt(int(face), int(index))


def _write_materials(edit, was, now, base: dict, state: dict) -> None:
    """each material's change written into its FbxSurfaceMaterial: its
    name, its colours and factors as the reader takes them (a colour times
    its factor: the colour moved, the file's factor kept), its bump depth,
    the UV set and wraps of its maps; faces given to another material of
    their mesh move in the mesh's material element. A material given
    another kind, mesh, texture or normal map, one that is neither a
    lambert nor a phong given other colours, the records of one material
    given different values, or a value a take keys: ValueError"""
    from cgmath.formats.fbx import _ascii_text

    fbx   = edit.fbx
    held  = _shaders(edit, state)
    pairs = list(zip(was.list, now.list))
    if len(held) != len(pairs):
        raise ValueError(f"FbxFile.save: materials= holds {len(pairs)} records, the file's meshes {len(held)}; {_ELSE}")

    first   = {}  # a material of the file -> the first record of it
    owners  = {}  # a record's uuid -> the material of the file it is
    faced   = {}  # an FbxMesh -> its records
    written = {}  # a material of the file -> the names its maps were written under
    for at, ((old, new), (shader, entry, index, shape)) in enumerate(zip(pairs, held)):
        place = f"materials[{at}]"
        if old.mesh_uuid != new.mesh_uuid:
            raise ValueError(f"FbxFile.save: {place} names another mesh than the file's; {_ELSE}")
        if old.kind != new.kind:
            raise ValueError(
                f"FbxFile.save: {place} is a {old.kind} given the kind {new.kind!r}; a material keeps its kind in an "
                f"open file; {_ELSE}"
            )
        other = first.setdefault(entry["id"], (at, new))
        split = new.uuid != other[1].uuid or owners.setdefault(new.uuid, entry["id"]) != entry["id"]
        if split or _moved(other[1], new, "name", *_SHADES, *_LOOKS, *_MAPS):
            raise ValueError(
                f"FbxFile.save: {place} and materials[{other[0]}] no longer agree on which records are one material, "
                f"or on its values: the file holds one material a uuid, on every mesh it is on; {_ELSE}"
            )
        faced.setdefault(int(shape.GetUniqueID()), (shape, []))[1].append((at, index, old, new))

        if other[0] == at and _moved(old, new, *_MAPS):
            # once a material, off its first record: the others agree with it
            if entry["bare"]:
                raise ValueError(
                    f"FbxFile.save: {place} is given another texture or normal map, and the file's material is "
                    f"neither a lambert nor a phong: it reads as a default lambert, whatever it holds; {_ELSE}"
                )
            written[entry["id"]] = _remapped(edit, shader, old, new, place)
        changed = [field for field in _SHADES if _moved(old, new, field)]
        if entry["bare"] and changed:
            raise ValueError(
                f"FbxFile.save: {place} is given another {changed[0]}, and the file's material is neither a lambert "
                f"nor a phong: it reads as a default lambert, whatever it holds; {_ELSE}"
            )
        if old.name != new.name:
            shader.SetName(_ascii_text(new.name or ""))
        _looked(edit, shader, new, [field for field in _LOOKS if _moved(old, new, field)])
        if not changed:
            continue
        phong = new.kind == "phong"
        made  = edit.typed(shader, fbx.FbxSurfacePhong if phong else fbx.FbxSurfaceLambert, fbx)
        plain = [("color", made.Diffuse), ("diffuse", made.DiffuseFactor), ("bump_depth", made.BumpFactor)]
        tints = [
            ("ambient_color", made.Ambient, made.AmbientFactor),
            ("incandescence", made.Emissive, made.EmissiveFactor),
            ("transparency", made.TransparentColor, made.TransparencyFactor),
        ]
        if phong:
            plain += [("cosine_power", made.Shininess), ("reflectivity", made.ReflectionFactor)]
            tints += [("specular_color", made.Specular, made.SpecularFactor)]
        for field, prop in plain:
            if field in changed:
                value = getattr(new, field)
                value = fbx.FbxDouble3(*[float(x) for x in value]) if field == "color" else float(value)
                _given(edit, prop, value, place, field)
        for field, colour, factor in tints:
            if field in changed:
                _tinted(edit, colour, factor, getattr(new, field), place, field)

    for shape, records in faced.values():
        if any(_moved(old, new, "faces") for _, _, old, new in records):
            _refaced(edit, shape, records)
    # every record of a material whose maps were written, under the names they were written as
    for (_, new), (_, entry, _, _) in zip(pairs, held):
        if entry["id"] in written:
            named_as_written(edit, [new], written[entry["id"]])


_WRITERS["materials"] = _write_materials


