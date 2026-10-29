from __future__ import annotations

import copy
import itertools
import bisect
import json
import numbers
import os
import re
import sys
import threading
import urllib.parse
import uuid
import warnings
import weakref
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, List, Optional, overload, Union

import numpy as np
from cgmath.data import (
    ITEMS_KEY,
    Array,
    Data,
    DataList,
    ImmutableArray as numpy_array,
    _class_attribute,
    _decode,
    _from_tagged,
    _holds_numbered,
    _load_derived,
    _mask_to_indices,
    _NO_DEFAULT,
    _reached_twice,
    _refusal,
    _save_shares,
)
from cgmath.formats._fbx_io import open_fbx, typed
from cgmath.formats.glb import load_model
from cgmath.transforms import (
    euler_filter,
    euler_reorder,
    euler_to_matrix,
    matrix_local as local_matrix,
    matrix_multiply,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_conjugate as q_conjugate,
    quaternion_multiply as q_multiply,
    quaternion_normalize as q_normalize,
    quaternion_slerp as q_slerp,
    quaternion_to_euler,
    quaternion_to_matrix,
)


try:
    from pygltflib import GLTF2
except ImportError:
    GLTF2 = None

try:
    import fbx as FBX
    from cgmath.formats.fbx import FbxExporter, _ascii_read_back, _user_property_value
except ImportError:
    FBX                  = None
    FbxExporter          = None
    _ascii_read_back     = None
    _user_property_value = None


__all__ = [
    "ClipData",
    "HierarchyData",
    "TransformData",
    "TransformList",
    "MAYA_ATTRIBUTE_MAP",
    "SUPPORTED_NODE_TYPES",
    "generate_uuid",
    "load",
    "load_fbx",
    "load_glb",
    "validate_uuid",
]


# attribute names mapped against Autodesk Maya
MAYA_ATTRIBUTE_MAP = {
    "_scale": "scale",
    "_rotate": "rotate",
    "_translate": "translate",
    "_rotate_order": "rotateOrder",
    "_rotate_axis": "rotateAxis",
    "_joint_orient": "jointOrient",
    "_segment_scale_compensate": "segmentScaleCompensate",
    "_radius": "radius",
    "_visibility": "visibility",
    "_draw_style": "drawStyle",
}

# the node types a TransformData can describe.
# anything else (constraints, ikHandles, etc.) is only a transform by inheritance,
# it can't be rebuilt nor exported from this data and is skipped along with its children.
SUPPORTED_NODE_TYPES = ("transform", "joint", "locator", "space_transform")

# ---------------------------------------------------------------------------- #


def _fbx_enum(owner, scope: str, name: str):
    """reads an fbx sdk enum through either of its two spellings

    Newer bindings nest the values in a scope class, ``FbxNodeAttribute.EType
    .eSkeleton``, where older ones hang them straight off the owner.
    """
    return getattr(getattr(owner, scope, owner), name)


def _duplicate_leaf(node, nodes) -> bool:
    """whether a glb node is a childless copy of its own parent

    three.js writes every joint of a skinned rig twice: the joint itself, and
    a childless node of the same name parented straight underneath it. The two
    collide by name in a dcc, which cannot then say which one a parent
    reference means, and the copy is never the one an animation targets.

    ``parent_index`` is set on every node by ``order_nodes_root_first``, which
    ``load_model`` runs before it returns.
    """
    if node.children or node.parent_index < 0:
        return False

    return bool(node.name) and node.name == nodes[node.parent_index].name


def _glb_node_type(node, nodes) -> Optional[str]:
    """the node type ``load_glb`` reads a glb node as, None for one it skips

    A momentum file says what each node is, and only its joints and locators
    are kept. Anything else is read style2: every node is a joint but one
    holding a mesh. That node is told by its mesh, not its name: a namespaced
    name such as "ns:body" says nothing about what the node holds.
    """
    if (
        not node.name
        or node.skin is not None
        or node.name == "RootNode"
        or _duplicate_leaf(node, nodes)
    ):
        return None

    # momentum skeleton, only keep joints and locators
    if node.extensions and "FB_momentum" in node.extensions:
        momentum = node.extensions["FB_momentum"]
        if "limitOrigin" in momentum:
            return None
        if momentum["type"] == "skeleton_joint":
            return "joint"
        if momentum["type"] == "locator":
            return "locator"
        return None

    # style2, everything's a joint but a node holding a mesh
    return "joint" if node.mesh is None else None


def _glb_columns(model, rig, filename: str) -> dict:
    """pairs a glb's node indexes with the rows of a rig read from it

    A file addresses nodes by index, and it can hold two nodes under the same
    name -- a rig exported once per mesh carries the whole skeleton twice --
    so the pairing is positional rather than by name. The rig is built from
    the same parent first walk of the file and only drops the nodes
    ``_glb_node_type`` skips, so walking that order over the nodes it keeps
    pairs them, and names are read to confirm the walk stayed in step rather
    than to do the matching. A skipped node never takes a row: a mesh named
    like a joint would otherwise hand that joint its animation.
    """
    column    = {}
    cursor    = 0
    rig_names = list(rig.name)

    for index in model.ordered_node_indexes:
        node = model.nodes[index]
        if _glb_node_type(node, model.nodes) is None:
            continue

        if cursor < len(rig_names) and node.name == rig_names[cursor]:
            column[index] = cursor
            cursor += 1

    if cursor != len(rig_names):
        raise ValueError(
            f"{filename!r} does not line up with the rig read from it: "
            f"matched {cursor} of {len(rig_names)} nodes"
        )

    return column


def _sample_gltf(sampler, times: np.ndarray, width: int) -> np.ndarray:
    """resamples one gltf sampler onto ``times``

    glTF keys every property on its own timeline, so a frame grid has to be
    interpolated rather than read off. Outside the authored range the first
    and last keys hold.

    A held (STEP) key shows from the frame it was keyed for. glTF stores
    key times as float32 seconds, and the rounding of a key and of the
    grid's first key can put a key on a frame up to one float32 spacing
    after that frame's time, so a time within one spacing before a key, the
    spacing taken at the key time, counts as on the key; of two keys on one
    time that way, the later shows. A key exactly on a time, on the time's
    own float32 value whichever side of the time float32 put it, keeps it,
    as glTF's own sampler gives for a key at or before the time, though a
    second key sits a spacing after it, and of two keys on that value the
    later shows. A key further off, half a frame say, shows from the next
    time on. Linear and cubic samplers are read at the times as given.
    """
    keys   = np.asarray(sampler.keyframe_times, dtype=float).ravel()
    values = np.asarray(sampler.keyframe_values, dtype=float)
    cubic  = sampler.interpolation == "CUBICSPLINE"

    # a cubic key is three values wide: in tangent, value, out tangent
    values = values.reshape(len(keys), 3, width) if cubic else values.reshape(-1, width)

    if len(keys) == 1:
        single = values[0, 1] if cubic else values[0]
        return np.repeat(single[None, :], len(times), axis=0)

    clamped = np.clip(times, keys[0], keys[-1])
    right   = np.clip(np.searchsorted(keys, clamped, side="right"), 1, len(keys) - 1)
    left    = right - 1

    span    = keys[right] - keys[left]
    safe    = np.where(span > 0.0, span, 1.0)
    weight  = np.where(span > 0.0, (clamped - keys[left]) / safe, 0.0)

    if sampler.interpolation == "STEP":
        # a held key owns the instant it lands on, including the last one,
        # so this indexes the key rather than the segment ahead of it, and
        # the times up to one float32 spacing of the key time before it.
        # The times go in unclamped: one before the first key shows that
        # key, even with a second key within a spacing of it. A key on a
        # time's own float32 value is exactly on it, whichever side of the
        # time float32 put it, and keeps it, though a second key sits a
        # spacing after it
        early = keys - np.spacing(np.abs(keys).astype(np.float32)).astype(float)
        snapped = np.clip(
            np.searchsorted(early, times, side="right") - 1, 0, len(keys) - 1
        )
        exact = times.astype(np.float32).astype(float)
        plain = np.clip(
            np.searchsorted(keys, exact, side="right") - 1, 0, len(keys) - 1
        )
        return values[np.where(keys[plain] == exact, plain, snapped)]

    if cubic:
        # the spec scales the tangents by the segment length
        u   = weight[:, None]
        uu  = u * u
        uuu = uu * u
        return (
            (2.0 * uuu - 3.0 * uu + 1.0) * values[left, 1]
            + (uuu - 2.0 * uu + u) * (span[:, None] * values[left, 2])
            + (-2.0 * uuu + 3.0 * uu) * values[right, 1]
            + (uuu - uu) * (span[:, None] * values[right, 0])
        )

    if width == 4:
        # a component wise lerp of two quaternions is not a rotation
        return q_slerp(values[left], values[right], weight)

    return values[left] + (values[right] - values[left]) * weight[:, None]


def _glb_frame_count(first: float, last: float, fps: float) -> int:
    """the frames a glTF animation from ``first`` to ``last`` seconds spans at
    ``fps``, both ends included

    glTF stores key times as float32 seconds, so keys on whole frames read
    back a hair off them: frames 0 and 1 at 24 fps are 1.00000003 frames
    apart, which a plain ceil counts as 3 frames. A span within one float32
    spacing of a whole number of frames, the spacing taken at the larger in
    size of the first and last key times (the last key's for any clip from
    0 on), is that whole number, and its last whole frame can fall a hair
    before the last key: on two frames or more the caller samples that
    frame on the key itself, unless a held channel's own key sits exactly
    on the frame, on its float32 time, and keys within one spacing of each
    other give one frame, on the first. Any other span rounds up, so its
    last key falls inside the grid rather than past its end. The spacing
    grows with the time, so the snap reaches further on a key minutes in
    than on one near zero.
    """
    span  = float(last - first) * fps
    whole = round(span)
    grain = float(np.spacing(np.float32(max(abs(first), abs(last))))) * fps

    if abs(span - whole) <= grain:
        return whole + 1
    return int(np.ceil(span)) + 1


def _is_sequence(obj, ndim: int = 1) -> bool:
    """tests if given input is a sequence matching the desired dimension number"""
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict, set)):
        obj = np.asarray(obj)
        return obj.ndim == ndim
    return False


def _rotation(matrix: np.ndarray) -> np.ndarray:
    """returns a copy of 4x4 matrices holding only their rotation

    scale is divided out of the rotation block and the translation row is
    zeroed, so the result composes as a pure orientation.
    """
    matrix              = np.array(matrix, dtype=float)
    block               = matrix[..., :3, :3]
    magnitude           = np.einsum("...ij,...ij->...i", block, block) ** 0.5
    matrix[..., :3, :3] = block / magnitude[..., None]
    matrix[..., 3, :3]  = 0.0
    return matrix


def _scale_and_rotation(block: np.ndarray, current: np.ndarray) -> tuple:
    """splits a 3x3 [S][Q] block into its signed scale S and rotation Q

    each row of the block is a scale times a row of Q, so the row lengths
    are the scale. a mirror (a negative determinant) goes on scale z, the
    way Maya splits a matrix. a zero scale flattens its row, so the block
    says nothing of where that row points: it comes from ``current``, the
    orientation the node has now, squared to the rows the block does set.
    """
    scale = np.einsum("...i,...i", block, block) ** 0.5
    held  = scale > 1e-12 * scale.max()

    if held.all():
        if np.linalg.det(block) < 0:
            scale[2] = -scale[2]
        return scale, block / scale[:, None]

    rotation       = np.array(current, dtype=float)
    rotation[held] = block[held] / scale[held, None]

    if held.sum() == 2:
        # one zero scale: its row closes the right handed frame
        i           = np.flatnonzero(~held)[0]
        rotation[i] = np.cross(rotation[(i + 1) % 3], rotation[(i + 2) % 3])

    elif held.sum() == 1:
        # two zero scales: the current row that leans least on the held one
        # is squared to it, and the last row closes the right handed frame
        i = np.flatnonzero(held)[0]
        j, k = (i + 1) % 3, (i + 2) % 3
        u = rotation[i]
        if abs(np.dot(rotation[j], u)) <= abs(np.dot(rotation[k], u)):
            v           = rotation[j] - np.dot(rotation[j], u) * u
            rotation[j] = v / np.linalg.norm(v)
            rotation[k] = np.cross(u, rotation[j])
        else:
            v           = rotation[k] - np.dot(rotation[k], u) * u
            rotation[k] = v / np.linalg.norm(v)
            rotation[j] = np.cross(rotation[k], u)

    # three zero scales leave the current orientation as it is
    return scale, rotation


def _same_transform(A: np.ndarray, B: np.ndarray) -> bool:
    """whether two 4x4 matrices hold the same transform, up to round-off

    a matrix that came through a product or an inverse differs in its last
    bits from the one the channels compute. taking that for a change would
    rebuild the channels from it, and rotate can come back as other angles
    for the same rotation. the band is a fixed size: np.allclose's default
    band grows with the value, so a small move far from the origin fell
    inside it and was dropped.
    """
    return np.allclose(A, B, rtol=0.0, atol=1e-8)


def _axis_permutation(axis0: int, axis1: int, negate: bool) -> np.ndarray:
    """4x4 signed permutation swapping two axes, with a determinant of +1

    swapping two rows on its own flips the determinant, so exactly one row
    has to change sign to keep the frame right handed. with ``negate`` that
    row is ``axis0``, which leaves the third axis pointing where it was;
    without it the third axis flips, which makes the swap its own inverse.
    """
    for axis in (axis0, axis1):
        if axis not in (0, 1, 2):
            raise ValueError(f"axes must be 0, 1 or 2, got {axis}")
    if axis0 == axis1:
        raise ValueError(f"axes must differ, got {axis0} twice")

    P               = np.eye(4)
    other           = 3 - axis0 - axis1
    P[axis0, axis0] = P[axis1, axis1] = 0.0
    P[axis0, axis1] = -1.0 if negate else 1.0
    P[axis1, axis0] = 1.0
    P[other, other] = 1.0 if negate else -1.0
    return P


def _component_property(channel: str, axis: str) -> property:
    """builds a vectorized accessor for one component of a vector channel

    there are fifteen of these (scale_x .. rotate_axis_z) and they only
    differ by which TransformData property they forward to, so they are
    generated rather than written out fifteen times.
    """
    name = f"{channel}_{axis}"

    def getter(self) -> np.ndarray:
        return np.array([getattr(node, name) for node in self])

    def setter(self, value) -> None:
        if _is_sequence(value):
            for i, node in enumerate(self):
                setattr(node, name, value[i])
        else:
            for node in self:
                setattr(node, name, value)

    getter.__name__ = name
    setter.__name__ = name
    return property(getter, setter, doc=f"vectorized {name}")


# Maya's short attribute names, as cmds.attributeQuery(<long>, node=<joint>,
# shortName=True) gives them in Maya 2025, to the name each stands for. A
# short name reads and writes its long name itself, so it does whatever the
# class does with it: its conversions and errors, a clip's frame, an SDF
# primitive's radius. Never saved: a file keeps the long names. Maya's
# inverseScale is "is", a Python keyword, so it has none here
_SHORT_NAMES = {
    "t":    "translate",
    "tx":   "translate_x",
    "ty":   "translate_y",
    "tz":   "translate_z",
    "r":    "rotate",
    "rx":   "rotate_x",
    "ry":   "rotate_y",
    "rz":   "rotate_z",
    "s":    "scale",
    "sx":   "scale_x",
    "sy":   "scale_y",
    "sz":   "scale_z",
    "jo":   "joint_orient",
    "jox":  "joint_orient_x",
    "joy":  "joint_orient_y",
    "joz":  "joint_orient_z",
    "ra":   "rotate_axis",
    "rax":  "rotate_axis_x",
    "ray":  "rotate_axis_y",
    "raz":  "rotate_axis_z",
    "ro":   "rotate_order",
    "ssc":  "segment_scale_compensate",
    "v":    "visibility",
    "radi": "radius",
    "ds":   "draw_style",
    "m":    "matrix",
    "wm":   "world_matrix",
    "pm":   "parent_matrix",
    "im":   "inverse_matrix",
    "wim":  "world_inverse_matrix",
    "pim":  "parent_inverse_matrix",
}

# the names that became built in with the short names, the four matrices
# that came with them included: a file's user attribute carrying one loads
# all the same, hidden by the built-in, and is warned of
_TAKEN_NAMES = frozenset(_SHORT_NAMES) | {
    "parent_matrix",
    "inverse_matrix",
    "world_inverse_matrix",
    "parent_inverse_matrix",
}


def _short_name(short: str, long: str, writable: bool) -> property:
    """Maya's short name ``short`` for the property ``long``: reads and,
    when ``writable``, writes ``long`` itself, by name, so a subclass's own
    ``long`` is the one it reads"""

    def getter(self):
        return getattr(self, long)

    def setter(self, value) -> None:
        setattr(self, long, value)

    getter.__name__ = setter.__name__ = short
    getter.__doc__  = f"Maya's short name for {long}"
    return property(getter, setter if writable else None)


def _add_short_names(cls) -> None:
    """gives ``cls`` the short name of every long name it has as a
    property, read only where the long name is"""
    for short, long in _SHORT_NAMES.items():
        held = getattr(cls, long, None)
        if isinstance(held, property):
            alias = _short_name(short, long, held.fset is not None)
            setattr(cls, short, alias)
            # named as a class body names it, for the error a write to a read
            # only one raises; python 3.9's property has no __set_name__
            named = getattr(alias, "__set_name__", None)
            if named is not None:
                named(cls, short)


def _inverse(matrices: np.ndarray, what: str, names) -> np.ndarray:
    """the inverse of each (4, 4) of ``matrices``, (4, 4), (n, 4, 4) or
    (F, n, 4, 4), ``what`` matrices of the n nodes ``names()`` names

    A true inverse, as Maya's: matrix_inverse takes the rows for a scale
    times a rotation, which a non-uniform scale under a rotation shears.

    Raises
    ------
    numpy.linalg.LinAlgError
        A matrix has no inverse, a zero scale: the first such is named.
    """
    try:
        return np.linalg.inv(matrices)
    except np.linalg.LinAlgError:
        flat     = np.asarray(matrices).reshape(-1, 4, 4)
        singular = np.flatnonzero(np.linalg.det(flat) == 0.0)
        if not len(singular):
            raise
        place = np.unravel_index(int(singular[0]), np.shape(matrices)[:-2] or (1,))
        frame = f" on frame {place[0]}" if len(place) == 2 else ""
        raise np.linalg.LinAlgError(
            f"the {what} of {names()[place[-1]]!r}{frame} has no inverse: a zero scale flattens it"
        ) from None


class _HiddenUserAttrs(threading.local):
    """for each load of nodes running in this thread, the user attributes it
    read that a built-in name hides, each attribute's name to the (uuid,
    name) of the nodes carrying it, for one warning a name once the
    outermost is done; and the copies running, which warn of nothing, the
    copy of a node a load warned of already"""

    def __init__(self):
        self.loads  = []
        self.copies = 0


_HIDDEN = _HiddenUserAttrs()


class _NodesLoad:
    """a load of nodes, a hierarchy or a list: the user attributes a built-in
    name hides that its nodes carry warn once a name, at its end; one inside
    another hands them on, a node the outer load counted already, by its
    uuid, counting once, as a clip's rest pose holds the clip's nodes again"""

    def __enter__(self) -> None:
        _HIDDEN.loads.append({})

    def __exit__(self, kind, error, trace) -> None:
        found = _HIDDEN.loads.pop()
        if found and kind is None:
            _hidden_found(found, nested=True)


class _Copying:
    """a copy: its load warns of no user attribute a built-in name hides"""

    def __enter__(self) -> None:
        _HIDDEN.copies += 1

    def __exit__(self, kind, error, trace) -> None:
        _HIDDEN.copies -= 1


def _hidden_noted(node: "TransformData") -> "TransformData":
    """``node``, just loaded, its user attributes a built-in name hides
    noted for the warning: for the load of nodes running to give, or now
    for a node loaded on its own; not for a copy"""
    attrs = node.__dict__.get("user_defined_attributes")
    if attrs and not _TAKEN_NAMES.isdisjoint(attrs) and not _HIDDEN.copies:
        _hidden_found({name: [(node.uuid, node.name)] for name in attrs if name in _TAKEN_NAMES})
    return node


def _hidden_found(found: dict, nested: bool = False) -> None:
    """``found``, the (uuid, name) of the nodes a load read with a user
    attribute a built-in name hides, by attribute: handed to the load of
    nodes running it is part of, or warned of, one warning a name, filed
    under the first caller outside cgmath. ``nested``, the nodes of a load
    of nodes inside that one: a node whose uuid it counted for that name
    already is not counted again"""
    loads = _HIDDEN.loads
    if loads:
        for name, nodes in found.items():
            counted = loads[-1].setdefault(name, [])
            if nested and counted:
                seen  = {uuid for uuid, _ in counted if uuid is not None}
                nodes = [node for node in nodes if node[0] not in seen]
            counted.extend(nodes)
        return

    level, frame = 2, sys._getframe(1)
    while frame is not None and not _outside_cgmath(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    for name, nodes in found.items():
        long  = _SHORT_NAMES.get(name)
        what  = f"Maya's short name for {long}" if long else "a built-in"
        other = len(nodes) - 1
        more  = f" and {other} other node{'s' * (other > 1)}" if other else ""
        warnings.warn(
            f"user attribute {name!r} of {nodes[0][1]!r}{more} loaded into "
            f"user_defined_attributes[{name!r}]: node.{name} is {what}, which hides it",
            stacklevel=level,
        )


def generate_uuid() -> str:
    """generates a new unique uuid"""
    value = str(uuid.uuid4()).upper()

    # a uuid by construction: a parent_uuid written with it is not checked
    # (_is_uuid), as a loop chaining nodes just made writes each one once
    if len(_UUIDS_SEEN) >= _UUIDS_KEPT:
        _UUIDS_SEEN.clear()
    _UUIDS_SEEN.add(value)
    return value


def validate_uuid(uuid_string: str) -> bool:
    """validates a uuid string"""
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


# user defined attribute types, by the cmds.addAttr() name, grouped by how a
# value converts to them
_FLOAT_TYPES  = ("double", "float", "doubleAngle", "doubleLinear")
_INT_TYPES    = ("long", "short", "byte", "char")
_NUMBER_TYPES = _FLOAT_TYPES + _INT_TYPES

# the types add_user_attribute() creates, as attributeType then as dataType
_NEW_ATTRIBUTE_TYPES = _NUMBER_TYPES + ("bool", "enum")
_NEW_DATA_TYPES = (
    "string",
    "stringArray",
    "doubleArray",
    "floatArray",
    "Int32Array",
    "vectorArray",
    "pointArray",
)


def _enum_fields(enum_name: str) -> list:
    """Splits a Maya enumName ("a=1:b=5:c") into (field, index) pairs.

    A field without an explicit index follows the previous one, the way
    Maya numbers them. As in Maya 2025's addAttr, a token is a field and
    its index only when its "=" splits it into a name and a whole number,
    spaces or tabs around it; a run of "=", or one at either end, splits
    nothing, so "k==2" and "=k=2" are field k at index 2. Any other token
    is the field's whole name: "size=small", "a=1=2", "=3".
    """
    fields = []
    index  = -1
    for token in enum_name.split(":"):
        # Maya splits at "=" the way C's strtok does, skipping empty pieces
        pieces = [piece for piece in token.split("=") if piece]
        if len(pieces) == 2 and re.fullmatch(r"[ \t]*[+-]?[0-9]+[ \t]*", pieces[1]):
            name, index = pieces[0], int(pieces[1])
        else:
            name, index = token, index + 1
        fields.append((name, index))
    return fields


def _user_attr_type(spec: dict) -> str:
    """the cmds.addAttr() type name of a user defined attribute spec"""
    return spec.get("attributeType", spec.get("dataType"))


def _to_float(value, label: str) -> float:
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"{label} takes a number, got {value!r}")


def _to_int(value, label: str) -> int:
    if isinstance(value, (bool, int, np.bool_, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"{label} takes a whole number, got {value!r}")
    raise TypeError(f"{label} takes a whole number, got {value!r}")


def _to_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{label} takes True, False, 0 or 1, got {value!r}")
    raise TypeError(f"{label} takes True, False, 0 or 1, got {value!r}")


def _to_enum(value, enum_name: Optional[str], label: str) -> int:
    fields = _enum_fields(enum_name) if enum_name else []
    if isinstance(value, str):
        for field, index in fields:
            if field == value:
                return index
        raise ValueError(
            f"{label} has no field {value!r}, its fields are {enum_name!r}"
        )
    index = _to_int(value, label)
    if fields and index not in {i for _, i in fields}:
        raise ValueError(f"{label} has no field {index}, its fields are {enum_name!r}")
    return index


def _to_list(value, label: str) -> list:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    raise TypeError(f"{label} takes a list, got {value!r}")


def _to_rows(value, size: int, label: str) -> list:
    rows = []
    for item in _to_list(value, label):
        row = [_to_float(x, label) for x in _to_list(item, label)]
        if len(row) != size:
            raise ValueError(f"{label} takes rows of {size} numbers, got {item!r}")
        rows.append(row)
    return rows


def _convert_one(kind: str, spec: dict, value, label: str):
    """``value`` converted to one value of a user defined attribute of ``kind``;
    a kind without a rule is stored as given"""
    if kind in _FLOAT_TYPES:
        return _to_float(value, label)
    if kind in _INT_TYPES:
        return _to_int(value, label)
    if kind == "bool":
        return _to_bool(value, label)
    if kind == "enum":
        return _to_enum(value, spec.get("enumName"), label)
    if kind == "string":
        if isinstance(value, str):
            return str(value)
        raise TypeError(f"{label} takes a string, got {value!r}")
    if kind == "stringArray":
        items = _to_list(value, label)
        if all(isinstance(x, str) for x in items):
            return [str(x) for x in items]
        raise TypeError(f"{label} takes a list of strings, got {value!r}")
    if kind in ("doubleArray", "floatArray"):
        return [_to_float(x, label) for x in _to_list(value, label)]
    if kind == "Int32Array":
        return [_to_int(x, label) for x in _to_list(value, label)]
    if kind == "vectorArray":
        return _to_rows(value, 3, label)
    if kind == "pointArray":
        return _to_rows(value, 4, label)
    if kind == "matrix":
        rows = _to_list(value, label)
        if len(rows) == 4:
            rows = [x for row in _to_rows(rows, 4, label) for x in row]
        if len(rows) != 16:
            raise ValueError(f"{label} takes 16 numbers or 4 rows of 4, got {value!r}")
        return [_to_float(x, label) for x in rows]
    return value


def _convert_user_value(spec: dict, value, label: str):
    """``value`` converted to what the user defined attribute ``spec`` stores; a
    multi attribute stores ``[[index, value], ...]``"""
    kind = _user_attr_type(spec)
    if not spec.get("multi"):
        return _convert_one(kind, spec, value, label)

    pairs = []
    for pair in _to_list(value, label):
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise TypeError(f"{label} takes [[index, value], ...], got {value!r}")
        index = _to_int(pair[0], label)
        if index < 0:
            raise ValueError(f"{label} takes indices from 0, got {index}")
        pairs.append([index, _convert_one(kind, spec, pair[1], label)])
    return pairs


def _compound_children(attrs: dict, name: str) -> list:
    """the names of a compound user attribute's children, in order"""
    return [child for child, spec in attrs.items() if spec.get("parent") == name]


def _user_value(attrs: dict, name: str):
    """a user defined attribute's value; a compound parent's comes from its
    children, as an array when they are all numbers, else as a tuple"""
    spec = attrs[name]
    if "numberOfChildren" not in spec:
        return spec.get("value")

    children = _compound_children(attrs, name)
    values   = [_user_value(attrs, child) for child in children]
    numeric = all(
        _user_attr_type(attrs[child]) in _NUMBER_TYPES
        and "numberOfChildren" not in attrs[child]
        and not attrs[child].get("multi")
        for child in children
    )
    return np.array(values) if children and numeric else tuple(values)


def _user_writes(attrs: dict, name: str, value, owner: str) -> list:
    """the (spec, converted value) pairs that setting ``name`` to ``value``
    stores, all converted before anything is written; a compound parent takes
    one value per child"""
    spec  = attrs[name]
    label = f"{owner}.{name}"
    if "numberOfChildren" not in spec:
        return [(spec, _convert_user_value(spec, value, label))]

    children = _compound_children(attrs, name)
    items    = _to_list(value, label)
    if len(items) != len(children):
        raise ValueError(
            f"{label} takes {len(children)} values, one per child, got {len(items)}"
        )
    writes = []
    for child, item in zip(children, items):
        writes.extend(_user_writes(attrs, child, item, owner))
    return writes


def _has_user_attr(node: "TransformData", name: str) -> bool:
    """whether ``node`` has a user attribute ``name`` a built-in does not hide"""
    attrs = node.user_defined_attributes
    return name in attrs and not node._is_builtin_attr(name)


def _infer_user_attr_type(value) -> str:
    """the type a new user attribute takes from its first value"""
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "long"
    if isinstance(value, (float, np.floating)):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple, np.ndarray)) and len(value):
        items   = list(value)
        numbers = (int, float, np.integer, np.floating)
        flags   = (bool, np.bool_)
        if all(isinstance(x, str) for x in items):
            return "stringArray"
        if all(isinstance(x, numbers) and not isinstance(x, flags) for x in items):
            return "doubleArray"
    raise TypeError(
        f"cannot tell a user attribute type from {value!r}, pass attribute_type"
    )


def _new_user_attr(
    value, attribute_type: Optional[str], keyable: bool, enum_names, label: str
) -> dict:
    """the spec of a new user defined attribute, in the form rig serializes"""
    kind = attribute_type or _infer_user_attr_type(value)
    if kind in _NEW_ATTRIBUTE_TYPES:
        spec = {"attributeType": kind}
    elif kind in _NEW_DATA_TYPES:
        spec = {"dataType": kind}
    else:
        raise ValueError(f"add_user_attribute cannot create a {kind!r} attribute")

    if kind == "enum":
        if not enum_names:
            raise ValueError(
                "an enum user attribute needs enum_names, e.g. 'a=1:b=5:c'"
            )
        spec["enumName"] = enum_names

    # Maya can not key strings or arrays
    spec["value"]       = _convert_user_value(spec, value, label)
    spec["keyable"]     = bool(keyable) and "attributeType" in spec
    spec["channel_box"] = False
    return spec


def _prefixed(name: str, prefix: str) -> str:
    """``name`` with ``prefix`` on its own part, after its namespace"""
    head, colon, short = name.rpartition(":")
    return head + colon + prefix + short


# which node is whose child follows every node's parent_uuid and uuid, and
# what each hierarchy lists. A hierarchy counts the edits to those among its
# own nodes: the index it keeps (HierarchyData._rig_index) and its nodes'
# cached world matrices hold the count they were built at, so an edit
# retires them in its own rig and nowhere else. Every count is drawn from
# one sequence, so no two rigs, or two edits, ever share one. A node added
# or given another parent_uuid patches a current index instead, and drops
# only the world matrices it moves (a node inserted ahead of the end, only
# when a lookup fetched the index since the last such insert). A name links
# no node to another: a rename changes the index's name table alone
_structure_ticks = itertools.count(1)


def _structure_changed(owner, index_only: bool = False) -> None:
    """an edit may have given some node of ``owner`` another parent or
    other children; a node no hierarchy owns has neither, so None counts
    nothing. ``index_only``: an edit that moves no node retires the index
    alone"""
    if owner is not None:
        tick                 = next(_structure_ticks)
        owner._index_version = tick
        if not index_only:
            owner._structure_version = tick


def _worlds_moved(owner) -> None:
    """an edit the index was patched for may have moved nodes it did not
    look for: the cached world matrices of ``owner`` go, the index stays"""
    # straight to the instance, as _index_patched writes: a loop of
    # parent_uuid writes runs this once a node
    owner.__dict__["_structure_version"] = next(_structure_ticks)


def _kept(owner) -> Optional["_RigIndex"]:
    """the index ``owner`` keeps while it is current, for an edit to patch;
    None when it keeps none, or none that is current"""
    kept = getattr(owner, "_kept_index", None)
    return None if kept is None else kept()


def _listed_by(node, rig) -> None:
    """``rig``, a HierarchyData, lists ``node`` without owning it, as an item
    assignment lists one, or may: a rig popping a node it listed twice, or
    losing it to another rig, lists it still in the other place. A uuid
    written on the node retires the index and the uuid counts of ``rig``
    too, as it retires its owner's. Held weakly, so a node keeps no rig
    alive"""
    listers = node._listers
    for lister in listers:
        if lister() is rig:
            return
    if listers:
        listers = tuple(lister for lister in listers if lister() is not None)
    node.__dict__["_listers"] = listers + (weakref.ref(rig),)


def _uuids_counted(counts: dict, nodes, step: int) -> dict:
    """``counts``, the count of each uuid a list holds, with the uuids of
    ``nodes`` put in (``step`` 1) or taken out (-1)"""
    for node in nodes:
        uuid = getattr(node, "uuid", None)
        if isinstance(uuid, str):
            left = counts.get(uuid, 0) + step
            if left > 0:
                counts[uuid] = left
            else:
                counts.pop(uuid, None)
    return counts


# a uuid as uuid.UUID writes it, in either case: what generate_uuid gives,
# and every uuid a file reader writes
_UUID_TEXT = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)

# the texts _is_uuid has found to be uuids, and the uuids generate_uuid
# made, so that a loop writing one parent_uuid over and over checks it
# once; emptied past _UUIDS_KEPT
_UUIDS_SEEN = set()
_UUIDS_KEPT = 1 << 16


def _is_uuid(value) -> bool:
    """validate_uuid(value) for a string, False for anything else

    The usual form is matched without a parse, as a loop of parent_uuid
    writes checks every one, and a text found to be a uuid is kept for the
    next check; any other form is parsed, from 32 characters on, the hex
    digits a uuid takes.
    """
    try:
        if value in _UUIDS_SEEN:
            return True
    except TypeError:
        return False  # unhashable, no string
    if not isinstance(value, str):
        return False
    if not (
        (len(value) == 36 and _UUID_TEXT.fullmatch(value) is not None)
        or (len(value) >= 32 and validate_uuid(value))
    ):
        return False
    if len(_UUIDS_SEEN) >= _UUIDS_KEPT:
        _UUIDS_SEEN.clear()
    _UUIDS_SEEN.add(value)
    return True


def _not_a_parent_uuid(cls, value) -> ValueError:
    """the error a parent_uuid that is not a uuid raises, naming set_parent
    for a name"""
    hint = f"; set_parent({value!r}) parents by name" if isinstance(value, str) else ""
    return ValueError(f"{cls.__name__}.parent_uuid takes a uuid or None, not {value!r}{hint}")


# cgmath's folder, and its tests' in it: a warning is filed under the first
# frame outside cgmath, its tests being outside
_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS_DIR   = os.path.join(_PACKAGE_DIR, "_tests")

# _outside_cgmath's answer for each file met: a warning asks it of each frame
# it walks, some 0.9 us of path work a frame, 0.05 once the answer is kept
_OUTSIDE = {}


def _outside_cgmath(filename: str) -> bool:
    """whether code in ``filename`` is the caller's rather than cgmath's"""
    outside = _OUTSIDE.get(filename)
    if outside is None:
        path               = os.path.normcase(os.path.abspath(filename))
        outside            = not path.startswith(_PACKAGE_DIR + os.sep) or path.startswith(_TESTS_DIR + os.sep)
        _OUTSIDE[filename] = outside
    return outside


def _uuid_of(parent) -> str:
    """``parent``'s uuid, as a child's parent_uuid takes it

    Raises
    ------
    ValueError
        ``parent`` carries no uuid: a node no hierarchy gave one, or whose
        uuid was written as something else.
    """
    uuid = parent.uuid
    if not _is_uuid(uuid):
        raise ValueError(f"{parent.name!r} can not be a parent: its uuid {uuid!r} is not a uuid")
    return uuid


def _parent_written(node, ref) -> None:
    """``node``'s parent_uuid is about to become ``ref``

    When its hierarchy keeps a current index the index is patched rather
    than retired, and only what the new parent moves is dropped: nothing
    when ``ref`` names the parent the old one named, the node's own cached
    world matrix when nothing hangs below it, which is how a rig is built
    node by node; a node with a branch retires the rig's cached world
    matrices, rather than walking the branch. Without such an index the
    structure count retires both, as any other edit does.
    """
    owner = node._hierarchy
    if owner is None:
        return

    # owner._kept_index() and index.position(node) written out, the rows
    # moved and the parents compared in place: a loop of parent_uuid writes
    # runs this once a node
    nodes = owner.list
    cache = owner._rig_index_cache
    if (
        cache is None
        or cache[3] is None
        or cache[0] != owner._index_version
        or cache[1] is not nodes
        or cache[2] != len(nodes)
    ):
        _structure_changed(owner)
        return

    index = cache[3]
    uuids = index.uuids
    uuid  = node.uuid
    try:
        at = uuids.get(uuid)
    except TypeError:
        at = None

    # an index that does not hold the node where it is, or does not have it
    # holding its parent_uuid -- one written past __setattr__ -- is out of
    # date: retired
    old  = node.parent_uuid
    refs = index.refs
    if at is None or nodes[at] is not node or not _remove_row(refs, old, at):
        _structure_changed(owner)
        return
    _insort_row(refs, ref, at)

    # the parent rows read from the index go; the list is as long as it was
    owner.__dict__["_parent_rows_cache"] = None

    # the positions of the parents named before and after, None for none or
    # one the hierarchy does not hold
    if (None if old is None else uuids.get(old)) == (None if ref is None else uuids.get(ref)):
        return
    if uuid is not None and uuid in refs:
        _worlds_moved(owner)
    else:
        # its world matrix alone, as the structure count drops it: a
        # parent_uuid written never called _transform_changed, which a
        # subclass extends (an SDF primitive marks its field dirty)
        node.__dict__["_world_matrix"] = None


def _renamed(owner, index: "_RigIndex", node, value) -> None:
    """``node`` of ``owner``, which keeps ``index`` (current or not), is
    about to be named ``value``

    A name links no node to another, so no node changes parent: the index
    the rig keeps stands, each node's unique_name is worked out again on
    its next read, and the name table, once a lookup by name has built one
    (:meth:`_RigIndex.named`), follows the rename of the one node carrying
    a name, the new name's first node being known. A rename of a node whose
    name other nodes may carry, or one the table is out of step with (a
    name written past __setattr__), drops the table for the next lookup by
    name to build again. A name the index can not key on retires it.
    TransformData.__setattr__ writes the common cases out.
    """
    if value.__class__ is not str and not isinstance(value, _INDEX_KEYS):
        _structure_changed(owner, index_only=True)
        return

    index._holders = None
    names          = index.names
    if names is None:
        return

    # an index an edit has retired is patched too, which does no harm, as
    # the next lookup builds another. The table has the node at its position
    # when it is the first node of its name: the table follows a rename of
    # the one node carrying a name. Of a name other nodes may carry, the node
    # left first is not known; nor is the node renamed when the table has
    # another first of its name -- an earlier node of a repeated name, or a
    # name written past __setattr__ -- or none: the table goes
    old   = node.name
    nodes = index.nodes
    at    = names.get(old)
    if at is None or at >= len(nodes) or nodes[at] is not node or old in index.repeated:
        index.names = None
        return
    del names[old]

    first = names.get(value)
    if first is None:
        names[value] = at
    else:
        index.repeated.add(value)
        if at < first:
            names[value] = at


class _LegacyParents(threading.local):
    """for each HierarchyData load running in this thread, the nodes it read
    with a parent named under "parent_node", a file older than parent_uuid:
    (node, name) pairs, given a uuid once every node is in"""

    def __init__(self):
        self.loads = []


_LEGACY_PARENTS = _LegacyParents()


def _structure_stamp(node) -> int:
    """the structure count ``node``'s world matrix holds while it is
    current: its hierarchy's, 0 for a node no hierarchy owns"""
    owner = node._hierarchy
    return 0 if owner is None else owner._structure_version


@dataclass(repr=False, eq=False)
class TransformData(Data):
    # name, node_type, uuid, and parent_uuid: the uuid of the parent node,
    # None for a root. Only a uuid or None: set_parent() parents by name
    name:        str = None
    node_type:   Optional[str] = "transform"
    uuid:        Optional[str] = None
    parent_uuid: Optional[str] = None

    # general transform attributes
    _scale:        Array(np.float64, 3) = numpy_array([1.0, 1.0, 1.0])
    _rotate:       Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    _translate:    Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    _rotate_order: Optional[int] = 0  # xyz, yzx, zxy, xzy, yxz, zyx
    _rotate_axis:  Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])

    # attributes specific to joints as defined by Maya
    _joint_orient:             Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    _visibility:               Optional[bool] = True
    _segment_scale_compensate: Optional[bool] = False
    _radius:                   Optional[float] = 1.0
    _draw_style:               Optional[int] = 0

    # user defined attributes, each node its own dict
    user_defined_attributes: dict = field(default_factory=dict)

    # --- cached attributes --- #
    _hierarchy     = None  # HierarchyData reference pointer
    _world_matrix  = None  # cached world matrix
    _world_version = None  # its rig's structure count it was cached at, and its link
    _listers       = ()    # weak references to the rigs that may list it, not owning it

    # the private state a node holds beside its fields: the caches above
    _STATE = ("_hierarchy", "_world_matrix", "_world_version", "_listers")

    def __init__(
        self,
        name:                     str,
        node_type:                Optional[str]        = "transform",
        uuid:                     Optional[str]        = None,
        parent_uuid:              Optional[str]        = None,
        scale:                    Optional[np.ndarray] = None,
        rotate:                   Optional[np.ndarray] = None,
        translate:                Optional[np.ndarray] = None,
        rotate_order:             Optional[int]        = None,
        rotate_axis:              Optional[np.ndarray] = None,
        joint_orient:             Optional[np.ndarray] = None,
        visibility:               Optional[bool]       = None,
        segment_scale_compensate: Optional[bool]       = None,
        radius:                   Optional[float]      = None,
        draw_style:               Optional[int]        = None,
        user_defined_attributes:  Optional[dict]       = None,
    ):
        """a node named ``name``; ``parent_uuid`` is the uuid of its parent,
        None for a root. A uuid.UUID, given as either uuid, is kept as its
        text

        Raises
        ------
        ValueError
            ``parent_uuid`` is not a uuid: a name is parented through
            :meth:`set_parent` once the node is in its hierarchy.
        """
        # set the variables
        self.name = str(name)

        if node_type is not None:
            self.node_type = str(node_type)

        if uuid is not None:
            self.uuid = str(uuid)

        if parent_uuid is not None:
            self.parent_uuid = parent_uuid

        # each node owns its channels: a channel write is in place, so a node
        # holding the caller's array would write through to it
        if scale is not None:
            self._scale = np.array(scale, dtype=float)
        else:
            self._scale = np.ones(3, dtype=float)

        if rotate is not None:
            self._rotate = np.array(rotate, dtype=float)
        else:
            self._rotate = np.zeros(3, dtype=float)

        if translate is not None:
            self._translate = np.array(translate, dtype=float)
        else:
            self._translate = np.zeros(3, dtype=float)

        if rotate_order is not None:
            self._rotate_order = int(rotate_order)
        else:
            self._rotate_order = 0  # xyz default

        if rotate_axis is not None:
            self._rotate_axis = np.array(rotate_axis, dtype=float)
        else:
            self._rotate_axis = np.zeros(3, dtype=float)

        if joint_orient is not None:
            self._joint_orient = np.array(joint_orient, dtype=float)
        else:
            self._joint_orient = np.zeros(3, dtype=float)

        if visibility is not None:
            self._visibility = bool(visibility)
        else:
            self._visibility = True

        if segment_scale_compensate is not None:
            self._segment_scale_compensate = bool(segment_scale_compensate)
        else:
            self._segment_scale_compensate = False

        if radius is not None:
            self._radius = float(radius)
        else:
            self._radius = 1.0

        if draw_style is not None:
            self._draw_style = int(draw_style)
        else:
            self._draw_style = 0

        if user_defined_attributes is not None:
            # its own, as the channels are: a write to one node's attribute
            # would otherwise reach every node built from the caller's dict
            self.user_defined_attributes = copy.deepcopy(user_defined_attributes)
        else:
            self.user_defined_attributes = {}

    def _write_channel(self, name: str, value: np.ndarray) -> None:
        """writes a vector channel into the storage it already holds

        The write is in place, so a channel bound to a view keeps writing
        through it. The class level default is one array the class shares,
        so a node ever left holding it takes its own copy first.

        A value is written the way numpy writes into a (3,) array: one number
        sets all three components, and a (1, 3) row, the shape a transforms
        function returns for one matrix, writes its row. A value numpy can
        not write there raises ValueError naming the channel, as the field
        does, rather than numpy's broadcast error.
        """
        cls   = type(self)
        spec  = cls._array_specs()[name]
        label = cls._field_label(name)
        try:
            value = np.asarray(value, dtype=float)
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from None

        shaped = np.empty(spec.shape)
        try:
            shaped[...] = value
        except ValueError:
            raise ValueError(f"{label} must have shape {spec.describe()}, got {value.shape}") from None

        value   = shaped
        current = getattr(self, name)

        # any difference is a change: a tolerance relative to the value would
        # miss a small move far from the origin and keep a stale world matrix
        if not np.array_equal(current, value):
            self._reset_branch_world_matrices()

        if current is getattr(type(self), name, None):
            setattr(self, name, np.array(value, dtype=float))
        else:
            current[...] = value

    @property
    def scale(self) -> np.ndarray:
        return np.array(self._scale, dtype=float)

    @property
    def scale_x(self) -> float:
        return self.scale[0]

    @property
    def scale_y(self) -> float:
        return self.scale[1]

    @property
    def scale_z(self) -> float:
        return self.scale[2]

    @scale.setter
    def scale(self, S: np.ndarray) -> None:
        self._write_channel("_scale", S)

    @scale_x.setter
    def scale_x(self, S: float) -> None:
        s          = self.scale
        s[0]       = S
        self.scale = s

    @scale_y.setter
    def scale_y(self, S: float) -> None:
        s          = self.scale
        s[1]       = S
        self.scale = s

    @scale_z.setter
    def scale_z(self, S: float) -> None:
        s          = self.scale
        s[2]       = S
        self.scale = s

    @property
    def rotate(self) -> np.ndarray:
        return np.array(self._rotate, dtype=float)

    @property
    def rotate_x(self) -> float:
        return self.rotate[0]

    @property
    def rotate_y(self) -> float:
        return self.rotate[1]

    @property
    def rotate_z(self) -> float:
        return self.rotate[2]

    @rotate.setter
    def rotate(self, R: np.ndarray) -> None:
        self._write_channel("_rotate", R)

    @rotate_x.setter
    def rotate_x(self, R: float) -> None:
        r           = self.rotate
        r[0]        = R
        self.rotate = r

    @rotate_y.setter
    def rotate_y(self, R: float) -> None:
        r           = self.rotate
        r[1]        = R
        self.rotate = r

    @rotate_z.setter
    def rotate_z(self, R: float) -> None:
        r           = self.rotate
        r[2]        = R
        self.rotate = r

    @property
    def translate(self) -> np.ndarray:
        return np.array(self._translate, dtype=float)

    @property
    def translate_x(self) -> float:
        return self.translate[0]

    @property
    def translate_y(self) -> float:
        return self.translate[1]

    @property
    def translate_z(self) -> float:
        return self.translate[2]

    @translate.setter
    def translate(self, T: np.ndarray) -> None:
        self._write_channel("_translate", T)

    @translate_x.setter
    def translate_x(self, T: float) -> None:
        t              = self.translate
        t[0]           = T
        self.translate = t

    @translate_y.setter
    def translate_y(self, T: float) -> None:
        t              = self.translate
        t[1]           = T
        self.translate = t

    @translate_z.setter
    def translate_z(self, T: float) -> None:
        t              = self.translate
        t[2]           = T
        self.translate = t

    @property
    def rotate_order(self) -> int:
        """the rotate order, 0..5: xyz, yzx, zxy, xzy, yxz, zyx

        Setting it changes the order alone, as Maya's setAttr does: rotate
        keeps its three angles, which now read in the new order, so the node
        turns and its children move with it. :meth:`set_rotate_order` keeps
        the pose instead.
        """
        return int(self._rotate_order)

    @rotate_order.setter
    def rotate_order(self, order: int) -> None:
        order = int(order)
        if self._rotate_order != order:
            self._reset_branch_world_matrices()
        self._rotate_order = order

    @property
    def rotate_axis(self) -> np.ndarray:
        return np.array(self._rotate_axis, dtype=float)

    @property
    def rotate_axis_x(self) -> float:
        return self.rotate_axis[0]

    @property
    def rotate_axis_y(self) -> float:
        return self.rotate_axis[1]

    @property
    def rotate_axis_z(self) -> float:
        return self.rotate_axis[2]

    @rotate_axis.setter
    def rotate_axis(self, axis: np.ndarray) -> None:
        self._write_channel("_rotate_axis", axis)

    @rotate_axis_x.setter
    def rotate_axis_x(self, R: float) -> None:
        r                = self.rotate_axis
        r[0]             = R
        self.rotate_axis = r

    @rotate_axis_y.setter
    def rotate_axis_y(self, R: float) -> None:
        r                = self.rotate_axis
        r[1]             = R
        self.rotate_axis = r

    @rotate_axis_z.setter
    def rotate_axis_z(self, R: float) -> None:
        r                = self.rotate_axis
        r[2]             = R
        self.rotate_axis = r

    @property
    def joint_orient(self) -> np.ndarray:
        return np.array(self._joint_orient, dtype=float)

    @property
    def joint_orient_x(self) -> float:
        return self.joint_orient[0]

    @property
    def joint_orient_y(self) -> float:
        return self.joint_orient[1]

    @property
    def joint_orient_z(self) -> float:
        return self.joint_orient[2]

    @joint_orient.setter
    def joint_orient(self, JO: np.ndarray) -> None:
        self._write_channel("_joint_orient", JO)

    @joint_orient_x.setter
    def joint_orient_x(self, JO: float) -> None:
        jo                = self.joint_orient
        jo[0]             = JO
        self.joint_orient = jo

    @joint_orient_y.setter
    def joint_orient_y(self, JO: float) -> None:
        jo                = self.joint_orient
        jo[1]             = JO
        self.joint_orient = jo

    @joint_orient_z.setter
    def joint_orient_z(self, JO: float) -> None:
        jo                = self.joint_orient
        jo[2]             = JO
        self.joint_orient = jo

    @property
    def segment_scale_compensate(self) -> bool:
        return bool(self._segment_scale_compensate)

    @segment_scale_compensate.setter
    def segment_scale_compensate(self, SSC: bool) -> None:
        SSC = bool(SSC)
        if self._segment_scale_compensate != SSC:
            self._reset_branch_world_matrices()
        self._segment_scale_compensate = SSC

    @property
    def parent_scale_inverse(self) -> np.ndarray:
        # an unresolved parent counts as no parent, matching get_parent_matrix
        parent = self.get_parent()
        if isinstance(parent, TransformData):
            return parent.scale

        return np.ones(3, dtype=float)

    @property
    def visibility(self) -> bool:
        return bool(self._visibility)

    @visibility.setter
    def visibility(self, value: bool) -> None:
        self._visibility = bool(value)

    @property
    def radius(self) -> float:
        return float(self._radius)

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def draw_style(self) -> int:
        return int(self._draw_style)

    @draw_style.setter
    def draw_style(self, value: int) -> None:
        self._draw_style = int(value)

    # --- methods --- #
    def _transform_changed(self) -> None:
        """this node's transform, or a parent's, has changed: drops the
        cached world matrix

        A subclass that keeps something else built from its transform
        extends this to drop that too.
        """
        self._world_matrix = None

    def _reset_branch_world_matrices(self):
        """resets the world matrix of this node and all its decendents, and
        anything else they built from their transform

        The branch is walked through the index the hierarchy keeps. When an
        edit has retired it since, one scan first tells whether the node has
        children at all: a leaf written after every edit that retires it
        would otherwise rebuild the index on each write.
        """
        hierarchy = self._hierarchy
        if hierarchy is not None:
            current = getattr(hierarchy, "_rig_index_is_current", None)
            if (current is None or not current()) and not self._scan_children():
                self._transform_changed()
                return

        for node in _walk((self,), _Lookups()):
            node._transform_changed()

    def __eq__(self, other):
        """use assigned uuid for equality"""

        if validate_uuid(self.uuid):
            if isinstance(other, str):
                if validate_uuid(other):
                    return self.uuid == other

                # if we're given a name, retrieve the uuid from the hierarchy
                if self._hierarchy is not None:
                    other = self._hierarchy[other]

            return self.uuid == other.uuid

        # use parent class equality test
        return super().__eq__(other)

    def to_attributes(self, mapping: dict = MAYA_ATTRIBUTE_MAP) -> dict:
        """returns a dict mapped against expected DCC node attributes

        Each field ``mapping`` names goes under the attribute name it maps
        to. ``mapping`` names a field as a file saves it, ``"translate"``, or
        by its own name, ``"_translate"``. On an SDFSphere or SDFCylinder the
        two spellings of a radius differ: ``"radius"`` is the primitive's
        radius, as its file saves it, and only ``"_radius"`` names the joint
        radius. A public name that is no field of the node maps nothing.

        Raises
        ------
        ValueError
            ``mapping`` names one field both ways.
        """
        cls     = type(self)
        mapping = cls._as_fields(mapping, "the mapping")
        return {
            mapping[key]: value if not isinstance(value, np.ndarray) else value.tolist()
            for key, value in self.__dict__.items()
            if key in mapping
        }

    def get_parent(self) -> Union["TransformData", str, None]:
        """returns this TransformData's parent TransformData, by its parent_uuid

        The first node of its hierarchy carrying the parent_uuid; None for a
        root and for a node no hierarchy owns. A parent_uuid no node of the
        hierarchy carries -- the parent was taken out with del or remove(),
        or the uuid was written so -- comes back as the uuid.
        """
        owner = self._hierarchy
        if owner is None:
            return None  # raise RuntimeError("undefined hierarchy")

        ref = self.parent_uuid
        if ref is None:
            return None

        # read from the index the hierarchy keeps while it is current, where
        # the scan below costs the hierarchy's size; with none, one is built
        # once a few lookups have scanned (HierarchyData._index_for_lookup)
        if isinstance(ref, str):
            index = _kept(owner)
            if index is None:
                built = getattr(owner, "_index_for_lookup", None)
                index = None if built is None else built()
            if index is not None:
                return index.resolve(ref)

        for node in owner.list:
            if node.uuid == ref:
                return node
        return ref

    @property
    def parent_name(self) -> Optional[str]:
        """the name of the node's parent, None for a root; read only

        ``get_parent().name``, None too for a node no hierarchy owns and for a
        parent_uuid no node of its hierarchy carries. Never saved: parent_uuid
        is the link, and set_parent() parents by name.
        """
        parent = self.get_parent()
        return parent.name if isinstance(parent, TransformData) else None

    def get_parent_matrix(self) -> np.ndarray:
        """returns the parent's worldspace matrix

        A node whose parent_uuid no node of its hierarchy carries -- the
        parent was taken out with del or remove() -- is treated as a root,
        the same answer a node with no hierarchy at all already gets.
        ``get_parent`` still hands back the uuid there, so the unresolved
        reference stays visible.
        """
        parent = self.get_parent()
        if isinstance(parent, TransformData):
            return parent.world_matrix

        return np.eye(4)

    def set_rotate_to_joint_orient(self) -> None:
        """sets rotate and rotate_axis to 0 and applies all rotation to joint_orient

        On a clip every frame is solved at once and joint_orient's curves
        go through euler_filter, frame 0 kept.
        """
        if self.node_type == "joint":
            frames, row = _clip_row(self)
            if frames is not None:
                _orient_rows(frames, [row], into_rotate=False)
                return

            # quaternion captures the full RO * R * JO orientation, so rotate
            # AND rotate_axis must both be zeroed or RO would be double-applied
            Q                 = self.quaternion
            self.rotate       = np.zeros(3, dtype=float)
            self.rotate_axis  = np.zeros(3, dtype=float)
            self.joint_orient = np.degrees(quaternion_to_euler(Q, 0)[0])

    def set_joint_orient_to_rotate(self) -> None:
        """sets joint_orient and rotate_axis to 0 and applies all rotation to rotate

        On a clip every frame is solved at once and rotate's curves go
        through euler_filter in this node's rotate order, frame 0 kept.
        """
        if self.node_type == "joint":
            frames, row = _clip_row(self)
            if frames is not None:
                _orient_rows(frames, [row], into_rotate=True)
                return

            # quaternion captures the full RO * R * JO orientation, so
            # joint_orient AND rotate_axis must both be zeroed or RO would be
            # double-applied
            Q                 = self.quaternion
            self.joint_orient = np.zeros(3, dtype=float)
            self.rotate_axis  = np.zeros(3, dtype=float)
            self.rotate       = np.degrees(quaternion_to_euler(Q, self.rotate_order)[0])

    def set_rotate_order(self, order: Union[int, str]) -> None:
        """
        Changes the rotate order and keeps the pose.

        As Maya's ``xform -rotateOrder <order> -preserve`` does: ``order`` is
        0..5 or the axes as a string, eg: ``"zyx"``, and only rotate changes.
        It is re-expressed in the new order, picking the solution Maya
        picks, so no world matrix moves. joint_orient and rotate_axis
        are built in xyz whatever the rotate order, so they are kept.
        Setting the order it already has changes nothing.

        On a clip every frame is re-expressed at once and the curves go
        through euler_filter in the new order, frame 0 kept.

        Assigning ``rotate_order`` directly changes the order alone, as
        Maya's setAttr does, and the node turns.
        """
        order = _rotate_order_index(order)
        frames, row = _clip_row(self)
        if frames is not None:
            _reorder_rows(frames, [row], [order])
            return

        if order != self.rotate_order:
            rotate            = _reorder_like_maya(self.rotate, self.rotate_order, order)
            self.rotate       = rotate
            self.rotate_order = order

    def set_node_type(self, node_type: str) -> None:
        """
        Sets the node type.

        A joint that becomes another type sheds its joint-only attributes
        and keeps its pose: joint_orient is folded into rotate (rotate_axis
        is kept), segment_scale_compensate is turned off, radius and
        draw_style go back to 1.0 and 0. Turning segment_scale_compensate
        off moves the node when its parent is scaled, since it no longer
        cancels that scale. Any other change only sets the type.

        On a clip every frame with a joint_orient is folded at once, and the
        rotate curves rebuilt go through euler_filter, frame 0 kept.

        Assigning ``node_type`` directly sets the type and nothing else.
        """
        self._set_node_type(node_type, fold=True)

    def _set_node_type(self, node_type: str, fold: bool) -> None:
        """set_node_type; ``fold`` False when a clip's joint_orient curves
        were already folded, every node of a selection in one batch"""
        node_type = str(node_type)
        if self.node_type == "joint" and node_type != "joint":
            frames, row = _clip_row(self) if fold else (None, None)
            if frames is not None:
                _fold_orient_rows(frames, [row])

            elif fold and np.any(self.joint_orient != 0.0):
                # the orientation chain is RO * R * JO, so R * JO as the new
                # rotate leaves the local matrix as it was once JO is zero
                RJ                = np.dot(self.rotate_matrix, self.joint_orient_matrix)
                self.rotate       = np.degrees(matrix_to_euler(RJ, self.rotate_order))[0]
                self.joint_orient = np.zeros(3, dtype=float)

            self.segment_scale_compensate = False
            self.radius                   = 1.0
            self.draw_style               = 0

        self.node_type = node_type

    def match_translate(
        self,
        other: "TransformData" | np.ndarray,
        x:     bool                         = True,
        y:     bool                         = True,
        z:     bool                         = True,
    ):
        """matches the world position of a TransformData object

        A node of a rig that records other axes or units than this node's
        rig raises ValueError, before anything moves
        (:attr:`HierarchyData.axis_system`); a position is taken as it is.
        """
        _check_matched(self, other)
        M0 = self.world_matrix

        # other is a xyz translation
        if _is_sequence(other, ndim=1):
            M1        = M0.copy()
            M1[3, :3] = other
        else:
            M1 = other.world_matrix

        # match the translate
        axes              = np.array([x, y, z, False], dtype=bool)
        M0[3, axes]       = M1[3, axes]
        self.world_matrix = M0

    def match_rotate(self, other: "TransformData" | np.ndarray):
        """matches the world rotation of a TransformData object

        A node of a rig that records other axes or units than this node's
        rig raises ValueError, before anything moves, as
        :meth:`match_translate` says; a rotation is taken as it is.
        """
        _check_matched(self, other)

        # convert the input to world matrix
        if _is_sequence(other, ndim=1):
            # set world matrix from euler rotation expressed in degrees
            if len(other) == 3:
                M = euler_to_matrix(np.radians(other), self.rotate_order)[0]

            # set world matrix from quaternion rotation
            elif len(other) == 4:
                M = quaternion_to_matrix(other)[0]

            # anything else is not a valid rotation
            else:
                raise ValueError(
                    "match_rotate expects a length-3 euler (degrees) or "
                    f"length-4 quaternion sequence, got length {len(other)}"
                )

        # get the world matrix
        else:
            M = other.world_matrix

            # normalize the matrix
            mag = np.einsum("ij,ij->i", M[:3, :3], M[:3, :3]) ** 0.5
            M[:3, :3] /= mag[:, None]

        # the world orientation is RO * R * JO * IS * P with the scale
        # divided out, so the pure rotate is R = inv(RO) * M * inv(JO * IS * P).
        # leaving RO out applies rotate_axis twice, as the quaternion setter
        # explains; leaving IS out lets a compensated parent scale skew it
        P  = self.get_parent_matrix()
        RO = self.rotate_axis_matrix
        JO = matrix_multiply(
            np.dot(self.joint_orient_matrix, self.parent_scale_inverse_matrix), P
        )
        L = local_matrix(np.dot(np.linalg.inv(RO), M), JO)

        # set the rotate value
        self.rotate = np.degrees(matrix_to_euler(L, self.rotate_order))[0]

    def match_matrix(self, other: "TransformData"):
        """matches the world matrix of a TransformData object

        A node of a rig that records other axes or units than this node's
        rig raises ValueError, before anything moves, as
        :meth:`match_translate` says.
        """
        _check_matched(self, other)
        M                 = other.world_matrix
        self.world_matrix = M

    def swapaxes(self, axis0: int, axis1: int, negate: bool = False) -> None:
        """swaps two of this node's local axes without moving its children

        ``axis0`` takes ``axis1``'s place and ``axis1`` takes ``axis0``'s, so
        ``swapaxes(0, 1)`` on a joint aiming down +X leaves it aiming down +Y.
        one axis has to flip sign to keep the frame right handed: with
        ``negate`` that is ``axis0``, a 90 degree roll leaving the third axis
        alone; without it the third axis flips and the swap undoes itself.

        the position never moves and direct children keep their world matrix,
        so only this node's frame changes. the swap lands in joint_orient, so
        an unposed joint keeps a rotate of zero.

        one exception: a child is put back through the matrix setter, which
        puts a mirror on scale z, the way Maya splits a matrix. a child whose
        parent chain mirrors another axis comes back with its scale signs
        moved, so a segment scale compensated child of it, and the nodes
        below, move with it.

        on a clip every frame is solved at once, and every curve rebuilt --
        this node's rotate, rotate_axis and joint_orient, its children's
        rotate -- goes through euler_filter, frame 0 kept.
        """
        P = _axis_permutation(axis0, axis1, negate)
        frames, row = _clip_row(self)
        if frames is not None:
            _swap_rows(frames, [row], axis0, axis1, negate)
            return

        Pi = P.T  # a signed permutation is orthogonal, so the inverse is the transpose

        # children are positioned relative to this frame, so snapshot their
        # world matrices and put them back once the frame has changed
        children = self.get_children()
        snapshot = [child.world_matrix for child in children]

        # scale permutes exactly, because P is an axis permutation and
        # therefore P * S is S with its components swapped, times P
        scale                 = np.array(self.scale, dtype=float)
        scale[[axis0, axis1]] = scale[[axis1, axis0]]

        # the orientation chain is [RO][R][JO]. conjugating RO and R and
        # left multiplying JO reproduces P * [RO][R][JO] exactly, which keeps
        # the rest orientation in joint_orient and the pose in rotate
        RO = self.rotate_axis_matrix
        R  = self.rotate_matrix
        JO = self.joint_orient_matrix

        self.scale       = scale
        self.rotate_axis = np.degrees(matrix_to_euler(np.dot(np.dot(P, RO), Pi), 0))[0]
        self.rotate = np.degrees(
            matrix_to_euler(np.dot(np.dot(P, R), Pi), self.rotate_order)
        )[0]
        self.joint_orient = np.degrees(matrix_to_euler(np.dot(P, JO), 0))[0]

        for child, world in zip(children, snapshot):
            child.world_matrix = world

    @property
    def unique_name(self):
        """returns True if no other node has this name"""

        if self._hierarchy is None:
            raise RuntimeError("undefined hierarchy")

        for x in self._hierarchy:
            if x.name == self.name and x.uuid != self.uuid:
                return False
        return True

    @property
    def index(self) -> int:
        """return's the node's index in the assigned hierarchy"""
        if self._hierarchy is None:
            raise ValueError(f"TransformData {self} not part of NodeList class")

        return self._hierarchy.index(self)

    def _cached_world(self) -> Optional[np.ndarray]:
        """the cached world matrix, None when there is none or when the
        rig's structure has changed since it was cached

        A channel write clears its branch's caches, but a structural edit --
        a parent_uuid or uuid written, a node added, removed, replaced or
        reordered -- can give any node of its rig another parent, and
        finding the nodes it moves would cost a walk of the rig per edit. The
        cache holds its rig's structure version instead, so one built before
        such an edit in its own rig is not trusted. An edit in another rig,
        or on a node no rig owns, leaves it alone, and so does a rename,
        which moves no node. An edit that patches the rig's index finds the
        nodes it moves there, and drops those alone when it can: a node
        added that no node names, a new parent_uuid on a node with nothing
        below it. A node below one its rig lists but does not own, which an
        edit of another rig or of no rig's node moves, holds that node's
        world matrix too, and is trusted while that node has it cached
        (_linked_world).
        """
        # _structure_stamp, written out: this is every world read's first step
        owner   = self._hierarchy
        version = self._world_version
        if version != (0 if owner is None else owner._structure_version):
            return None if version.__class__ is not tuple else _linked_world(self)
        return self._world_matrix

    @property
    def world_matrix(self):
        """compute a world matrix

        Up the parents to the first one whose world matrix is cached, or to
        a root, then back down, caching each node's on the way: a read after
        a structural edit fills its chain once, and a chain deeper than the
        recursion limit reads as any other. Below a node its rig lists but
        does not own (``rig[i] = node``, another rig's or no rig's), a
        cache stands while that node's does, so what moves that node shows.
        """
        world = self._cached_world()
        if world is None:
            world = _world_down(self)
        return world.copy()

    @world_matrix.setter
    def world_matrix(self, M: np.ndarray) -> None:
        """sets the world matrix, solving the local matrix under the parent

        Scale, rotate and translate are solved; rotate_axis, joint_orient
        and segment_scale_compensate are kept as they are, so a world matrix
        the node can hold (one without shear) reads back as written. A
        mirror goes on scale z, the way Maya splits a matrix, and what a
        zero scale leaves of the rotation is kept from the current one.
        """
        M      = np.asarray(M, dtype=float).reshape(4, 4)
        cached = self._cached_world()
        if cached is None or not _same_transform(cached, M):
            W = self.get_parent_matrix()
            # a true inverse is required: a scaled parent's world matrix has
            # non-orthogonal rows, and matrix_local's fast affine inverse
            # (which assumes scale*rotation) would compute the wrong local
            self.matrix = np.dot(M, np.linalg.inv(W))

    @property
    def rotate_axes(self) -> str:
        """returns the rotate order as a string"""
        return ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"][self.rotate_order]

    @property
    def rotate_axis_matrix(self) -> np.ndarray:
        return euler_to_matrix(np.radians(self.rotate_axis), 0)[0]

    @property
    def rotate_matrix(self) -> np.ndarray:
        return euler_to_matrix(np.radians(self.rotate), self.rotate_order)[0]

    @property
    def quaternion(self) -> np.ndarray:
        """rotation as a quaterion"""
        # matrix = [S] * [RO] * [R] * [JO] * [IS] * [T]
        RO = self.rotate_axis_matrix
        R  = self.rotate_matrix
        JO = self.joint_orient_matrix

        M  = np.eye(4)
        for mat in [RO, R, JO]:
            M = np.dot(M, mat)
        return matrix_to_quaternion(M)[0]

    @quaternion.setter
    def quaternion(self, Q: np.ndarray) -> None:
        """sets rotation as a quaternion, preserve rotate_axis and joint_orient"""
        Q = np.asarray(Q, dtype=float)
        M = quaternion_to_matrix(Q)[0]

        # the orientation chain is RO * R * JO, so solve for the pure rotate
        # R = inv(RO) * M * inv(JO); ignoring RO leaves rotate_axis double-applied
        RO          = self.rotate_axis_matrix
        JO          = self.joint_orient_matrix
        L           = local_matrix(np.dot(np.linalg.inv(RO), M), JO)
        self.rotate = np.degrees(matrix_to_euler(L, self.rotate_order))[0]

        # reset the branch cached world_matrix
        self._reset_branch_world_matrices()

    @property
    def joint_orient_matrix(self) -> np.ndarray:
        return euler_to_matrix(np.radians(self.joint_orient), 0)[0]

    @property
    def scale_matrix(self) -> np.ndarray:
        S = np.eye(4)
        S[:3, :3] *= self.scale
        return S

    @property
    def parent_scale_inverse_matrix(self) -> np.ndarray:
        IS = np.eye(4)
        if self.segment_scale_compensate:
            IS[:3, :3] /= self.parent_scale_inverse
        return IS

    @property
    def translate_matrix(self) -> np.ndarray:
        T        = np.eye(4)
        T[3, :3] = self.translate
        return T

    @property
    def matrix(self) -> np.ndarray:
        """computes the local transform matrix"""

        # matrix = [S] * [RO] * [R] * [JO] * [IS] * [T]
        S  = self.scale_matrix
        RO = self.rotate_axis_matrix
        R  = self.rotate_matrix
        JO = self.joint_orient_matrix
        IS = self.parent_scale_inverse_matrix
        T  = self.translate_matrix

        M  = np.eye(4)
        for mat in [S, RO, R, JO, IS, T]:
            M = np.dot(M, mat)

        return M

    @matrix.setter
    def matrix(self, M: np.ndarray) -> None:
        """sets the local transform matrix

        Scale, rotate and translate are solved; rotate_axis, joint_orient
        and segment_scale_compensate are kept as they are, so a matrix the
        node can hold (one without shear) reads back as written. A mirror
        goes on scale z, the way Maya splits a matrix, and what a zero scale
        leaves of the rotation is kept from the current one.
        """
        M = np.asarray(M, dtype=float).reshape(4, 4)

        # trigger the update only if the matrix has changed
        if not _same_transform(self.matrix, M):
            # the local matrix is [S][RO][R][JO][IS][T]; strip the joint
            # orient AND parent-scale-inverse factors (JO * IS) from the right
            # to recover the [S][RO][R] block.
            # ignoring IS here corrupts the decomposition whenever the parent
            # has non-unit scale and this node is segment-scale-compensated.
            # NOTE: a true inverse is required -- IS carries a scale, so JO * IS
            # has non-orthogonal rows and matrix_local's fast affine inverse
            # (which assumes scale*rotation) would invert it incorrectly.
            JO = self.joint_orient_matrix
            IS = self.parent_scale_inverse_matrix
            B  = np.dot(M, np.linalg.inv(np.dot(JO, IS)))

            # split the [S][RO][R] block into the signed scale and the
            # orientation [RO][R]; the current one fills what a zero scale
            # leaves out
            RO      = self.rotate_axis_matrix
            current = np.dot(RO, self.rotate_matrix)[:3, :3]
            scale, rotation = _scale_and_rotation(B[:3, :3], current)
            self.scale = scale

            # rotate: rotate_axis is kept, so R = inv(RO) * [RO][R], the way
            # the quaternion setter solves it
            RR          = np.eye(4)
            RR[:3, :3]  = rotation
            R           = np.dot(np.linalg.inv(RO), RR)
            self.rotate = np.degrees(matrix_to_euler(R, self.rotate_order))[0]

            # translate
            self.translate = M[3, :3]

            # reset the branch cached world_matrix
            self._reset_branch_world_matrices()

    # --- Maya's parent and inverse matrices, read only --- #
    @property
    def parent_matrix(self) -> np.ndarray:
        """the parent's world matrix, Maya's parentMatrix; read only

        get_parent_matrix() as a property: identity for a root, and for a
        node whose parent its hierarchy does not hold.
        """
        return self.get_parent_matrix()

    @property
    def inverse_matrix(self) -> np.ndarray:
        """the inverse of matrix, Maya's inverseMatrix; read only

        A true inverse. A zero scale has none: numpy.linalg.LinAlgError
        names the node (Maya holds a zero scale at 1e-12, and inverts that).
        """
        return _inverse(self.matrix, "matrix", lambda: [self.name])

    @property
    def world_inverse_matrix(self) -> np.ndarray:
        """the inverse of world_matrix, Maya's worldInverseMatrix; read only

        A true inverse, as inverse_matrix.
        """
        return _inverse(self.world_matrix, "world_matrix", lambda: [self.name])

    @property
    def parent_inverse_matrix(self) -> np.ndarray:
        """the inverse of parent_matrix, Maya's parentInverseMatrix; read only

        Identity for a root. A true inverse, as inverse_matrix.
        """
        return _inverse(self.parent_matrix, "parent_matrix", lambda: [self.name])

    # --- hierarchy queries --- #

    def get_root(self) -> "TransformList":
        """returns a NodeList of the branch from root to self"""
        branch_list = [self]
        while True:
            node = branch_list[-1].get_parent()
            if node is None:
                break
            branch_list.append(node)

        return TransformList(branch_list[::-1])

    def get_children(self) -> "TransformList":
        """returns the node's children, the nodes whose parent_uuid is its uuid

        They come in hierarchy order, from the index the hierarchy keeps
        until its structure changes, not from a scan of every node, so a
        branch walk, and with it every channel write, costs the size of the
        branch.
        """
        return TransformList(_Lookups().children(self))

    def _scan_children(self) -> list:
        """:meth:`get_children` by a scan of every node of the hierarchy: for
        a hierarchy the index can not stand for, and for a reset whose index
        an edit has just retired"""
        owner, uuid = self._hierarchy, self.uuid
        if owner is None or uuid is None:
            return []

        # the plain list rather than the hierarchy's item access: a reset runs
        # this on a node whose index an edit has just retired
        return [node for node in owner.list if node.parent_uuid == uuid]

    def get_branch(self) -> "TransformList":
        """returns a recursive NodeList from of all decendents

        This node, then each child followed by that child's own branch.
        Walked with a stack rather than by recursion, so a chain deeper than
        Python's recursion limit is walked too, and a node is listed once
        even when a cyclic parent_uuid leads back to it.
        """
        return TransformList(_walk((self,), _Lookups()))

    def set_parent(
        self,
        parent:      Union["TransformData", str, None],
        world_space: bool                              = True,
    ) -> None:
        """reparents node to a given parent and resets internal SRTs

        ``parent`` is a node of this node's hierarchy, its uuid, or a name:
        the first node of the hierarchy carrying it. None puts the node at
        the world. Whichever way it is given, the node keeps the parent's
        uuid in ``parent_uuid``, so a later rename moves nothing.

        With ``world_space`` the node keeps its world matrix: scale, rotate
        and translate are solved under the new parent. On a clip that holds
        for every frame, all solved at once, and a rotate curve the solve
        rebuilds goes through euler_filter, frame 0 kept; one that already
        held its world on every frame keeps its keys as they are. Without it
        the channels are left as they are and the node moves with its new
        parent. A parent in the node's own branch, the node itself included,
        would close a loop: it raises ValueError before anything moves, and
        so does a parent the hierarchy does not hold.
        """

        # raise error if _hierarchy is missing
        if self._hierarchy is None:
            raise RuntimeError("undefined hierarchy")

        # an unknown parent, or one in this node's own branch, raises before
        # anything moves
        above = ref = None
        if parent is not None:
            above = _held_node(self._hierarchy, parent)
            if above is self:
                raise ValueError(f"{self.name!r} can not be its own parent")
            _refuse_a_cycle([self], above)
            ref = _uuid_of(above)

        # grab the original world_matrix ahead of time, every frame of a clip
        frames, row = _clip_row(self) if world_space else (None, None)
        if frames is not None:
            M      = frames.world([row])
            before = frames.read("_rotate", [row])
        else:
            M = self.world_matrix

        # set new parent_uuid
        self.parent_uuid = ref

        # --- recompute matrices if world_space is True --- #
        if frames is not None:
            up = -1 if above is None else frames.row_of(above)
            _write_solved(frames, [row], _solve_under(frames, [row], M, -1 if up is None else up))
            _filter_rotate(frames, [row], before)
            frames.finish()
        elif world_space:
            self._world_matrix = None
            self.world_matrix  = M
        else:
            # parent_uuid changed without restoring world space, so any
            # cached world matrices on this node and its branch are now
            # stale and must be invalidated
            self._reset_branch_world_matrices()

        self._to_the_end()

    def _to_the_end(self) -> None:
        """moves the node and all its decendents to the end of the list"""
        _to_the_end(self._hierarchy, [self])

    def add_prefix(self, prefix: str) -> None:
        """adds a prefix to the node's name

        The prefix goes on the node's own name, after its namespace: ``"L_"``
        turns ``A:B:root`` into ``A:B:L_root``.
        """
        self.name = _prefixed(self.name, prefix)

    def add_suffix(self, suffix: str) -> None:
        """adds a suffix to the node's name"""
        self.name = self.name + suffix

    def _namespace_peers(self):
        """the node's hierarchy: a stripped name must not take its nodes' names"""
        return self._hierarchy if self._hierarchy is not None else ()

    # --- user defined attributes, as properties --- #

    def _is_builtin_attr(self, name: str) -> bool:
        """whether ``name`` is a field, property or method of the class"""
        return hasattr(type(self), name) or name in self._dataclass_field_names()

    def __getattr__(self, name: str):
        """a user defined attribute's value, read as ``node.heroHeight``

        Only reached when normal lookup fails, so built-in names always win.
        """
        attrs = self.__dict__.get("user_defined_attributes")
        if attrs and not name.startswith("_") and name in attrs:
            return _user_value(attrs, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value) -> None:
        """sets a user defined attribute the node has, as ``node.heroHeight =
        3.1416``, converted to its type; anything else goes to Data, which
        raises on names the class does not declare. ``user_defined_attributes
        = None`` stores an empty dict. ``parent_uuid`` takes a uuid or None,
        and raises ValueError on anything else, a name included."""
        # a rename: the name table of the index the hierarchy keeps follows
        # it, and nothing else does, as a name links no node to another. It
        # comes first, as a loop of renames runs it once a node; name is a
        # plain field, which Data.__setattr__ would hand to object.__setattr__
        # too, and no user attribute carries it
        if name == "name":
            owner = self._hierarchy
            if owner is not None:
                cache = owner._rig_index_cache
                index = None if cache is None else cache[3]
                if index is not None:
                    # _renamed's common cases written out: no name table
                    # built, or a name no other node carries, before the
                    # rename or after
                    names = index.names
                    if value.__class__ is not str:
                        _renamed(owner, index, self, value)
                    elif names is None:
                        index._holders = None
                    elif not index.spare:
                        # renamed through its spare with no lookup by name
                        # between: the table lapses, for the next to build
                        index.names    = None
                        index._holders = None
                    else:
                        index.spare -= 1
                        old   = self.name
                        at    = names.get(old)
                        nodes = index.nodes
                        if (
                            at is not None
                            and at < len(nodes)
                            and nodes[at] is self
                            and old not in index.repeated
                            and value not in names
                        ):
                            index._holders = None
                            del names[old]
                            names[value] = at
                        else:
                            _renamed(owner, index, self, value)
            object.__setattr__(self, "name", value)
            return

        # a new parent_uuid or uuid can move a node of the rig to another
        # parent, so the rig's index of its children and its cached world
        # matrices have to go: a new parent_uuid patches the index
        # (_parent_written), a new uuid retires both. Written past
        # Data.__setattr__ too, as a loop of parent_uuid writes runs it once
        # a node. A uuid.UUID is kept as its text, as the uuid keyword is
        if name == "parent_uuid":
            if value is not None and not _is_uuid(value):
                if not isinstance(value, uuid.UUID):
                    raise _not_a_parent_uuid(type(self), value)
                value = str(value)
            _parent_written(self, value)
            object.__setattr__(self, "parent_uuid", value)
            return

        if not name.startswith("_"):
            if name == "uuid":
                _structure_changed(self._hierarchy)
                # and each rig listing the node without owning it
                for lister in self._listers:
                    _structure_changed(lister())

            attrs = self.__dict__.get("user_defined_attributes")
            if attrs and name in attrs and not self._is_builtin_attr(name):
                for spec, converted in _user_writes(attrs, name, value, self.name):
                    spec["value"] = converted
                return

        # no attributes is an empty dict, never None: every node holds its own
        if name == "user_defined_attributes" and value is None:
            value = {}

        try:
            super().__setattr__(name, value)
        except AttributeError as error:
            # a user attribute is public, and never a name the class has
            if name.startswith("_") or self._is_builtin_attr(name):
                raise
            raise AttributeError(
                f"{error}; add_user_attribute() creates a user attribute"
            ) from None

    @classmethod
    def _from_state(cls, data: dict) -> "TransformData":
        # a file older than parent_uuid saved it as "parent_node", a uuid or
        # a name. A uuid is read as parent_uuid; a name is left to the
        # hierarchy loading the node (HierarchyData._from_state), which gives
        # it the uuid of the first node of that name, and a node loaded on
        # its own has no hierarchy to look in: a root. A user attribute a
        # built-in name hides is warned of (_hidden_noted)
        if "parent_node" not in data:
            return _hidden_noted(super()._from_state(data))

        legacy = data.pop("parent_node")
        if "parent_uuid" in data:
            raise ValueError(
                f"{cls.__name__}.parent_uuid is given twice in the data, as "
                f"'parent_node' and 'parent_uuid'"
            )
        named = legacy is not None and not _is_uuid(legacy)
        if legacy is not None and not named:
            data["parent_uuid"] = legacy

        node    = super()._from_state(data)
        pending = _LEGACY_PARENTS.loads
        if named and pending:
            pending[-1].append((node, legacy))
        return _hidden_noted(node)

    def copy(self) -> "TransformData":
        """returns a deep copy of self: the object a save and load gives back

        But for the load's warnings: the node's own load gave them.
        """
        with _Copying():
            return super().copy()

    def __dir__(self):
        attrs = self.__dict__.get("user_defined_attributes") or {}
        return sorted(set(super().__dir__()) | set(attrs))

    def _check_new_user_attr(self, name: str) -> None:
        """raises ValueError when ``name`` can not be a new user attribute"""
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"{name!r} is not a valid user attribute name")
        if self._is_builtin_attr(name):
            long = _SHORT_NAMES.get(name)
            hint = f", Maya's short name for {long}" if long else ""
            raise ValueError(
                f"{name!r} is a built-in attribute of {type(self).__name__}{hint}"
            )
        if name in self.user_defined_attributes:
            raise ValueError(f"{self.name} already has a user attribute {name!r}")

    def add_user_attribute(
        self,
        name:           str,
        value:          Any,
        attribute_type: Optional[str] = None,
        keyable:        bool          = True,
        enum_names:     Optional[str] = None,
    ) -> None:
        """creates a user defined attribute, then read and set as ``node.name``

        ``attribute_type`` is a cmds.addAttr() type: double, float, doubleAngle,
        doubleLinear, long, short, byte, char, bool, enum, string, stringArray,
        doubleArray, floatArray, Int32Array, vectorArray or pointArray. Without
        one it comes from ``value``: bool, long, double, string, stringArray or
        doubleArray. An enum needs ``enum_names``, e.g. ``"a=1:b=5:c"``, read
        as Maya reads it: a "=" gives an index only when it splits the field
        into a name and a whole number, and a run of "=", or one at either
        end, splits nothing, so ``"size=small"`` and ``"=3"`` are one field
        each and ``"k==2"`` is field k at index 2. String and array
        attributes are never keyable, as in Maya.

        Raises ValueError when the name exists, is built in (Maya's short
        names, ``tx`` or ``wm``, included) or is not a valid attribute name,
        and TypeError or ValueError when the value does not fit the type.
        """
        self._check_new_user_attr(name)
        spec = _new_user_attr(
            value, attribute_type, keyable, enum_names, f"{self.name}.{name}"
        )
        self.user_defined_attributes[name] = spec


# Maya's short names, t for translate to wm for world_matrix
_add_short_names(TransformData)


def _fbx_user_attributes(node) -> dict:
    """the user defined attributes of an fbx node, as specs, parents first

    A node FbxExporter wrote comes back with the specs it had, kept in a
    hidden property, its values read from the properties, or kept there too
    for a value a property can't give back. Any other property reads the way
    Maya's own FBX importer reads it.

    A property whose name, string or enum field is not UTF-8, which the fbx
    binding can't read, is left out, and so is an enum with a field an
    enumName can't hold, with a warning.
    """
    user_defined = FBX.FbxPropertyFlags.EFlags.eUserDefined
    written      = _fbx_written_specs(node)
    attrs        = {}

    prop = node.GetFirstProperty()
    while prop.IsValid():
        if prop.GetFlag(user_defined):
            # an older exporter's Latin-1 text loses this property only, not
            # the whole load, nor the clip and skin loads that walk it too
            try:
                name = str(prop.GetName())
                attrs.update(
                    _fbx_written_attr(prop, name, written)
                    or _fbx_inferred_attr(prop, name, node)
                )
            except UnicodeDecodeError:
                pass
        prop = node.GetNextProperty(prop)

    return attrs


def _fbx_written_specs(node) -> dict:
    """the specs FbxExporter kept on a node, by name; {} for one it did not write"""
    prop = node.FindProperty(FbxExporter._USER_ATTR_SPECS)
    if not prop.IsValid():
        return {}

    # FbxExporter %-escapes the json, see _ascii_json
    try:
        specs = json.loads(urllib.parse.unquote(str(FBX.FbxPropertyString(prop).Get())))
    except ValueError:
        return {}
    if not isinstance(specs, dict):
        return {}
    return {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, dict) and isinstance(_user_attr_type(spec), str)
    }


def _fbx_value(prop, kind: str):
    """a user defined fbx property's value, as a ``kind`` attribute holds it;
    raises UnicodeDecodeError for a string that is not UTF-8"""
    if kind in _FLOAT_TYPES:
        return float(FBX.FbxPropertyDouble1(prop).Get())
    if kind == "bool":
        return bool(FBX.FbxPropertyBool1(prop).Get())
    if kind in ("string", "stringArray"):
        return str(FBX.FbxPropertyString(prop).Get())
    if kind in ("double3", "float3"):
        return [float(x) for x in FBX.FbxPropertyDouble3(prop).Get()]

    # whole numbers, and an enum's position in its field list
    return int(FBX.FbxPropertyInteger1(prop).Get())


def _fbx_string_array(text: str) -> Optional[list]:
    """the list a stringArray's string holds, None for any other string"""
    mark = FbxExporter._STRING_ARRAY_MARK
    if not text.startswith(mark):
        return None

    # FbxExporter %-escapes the json, see _string_array_text
    try:
        items = json.loads(urllib.parse.unquote(text[len(mark) :]))
    except ValueError:
        return None
    if isinstance(items, list) and all(isinstance(x, str) for x in items):
        return items
    return None


def _fbx_written_attr(prop, name: str, written: dict) -> Optional[dict]:
    """a property FbxExporter wrote, as ``{name: spec}``, a double3 or float3
    with its three children; None when its spec is missing or no longer fits"""
    spec = written.get(name)
    kind = _user_attr_type(spec) if spec else None
    want = FbxExporter._USER_ATTR_TYPE_MAP.get(kind)
    if want is None or want.GetType() != prop.GetPropertyDataType().GetType():
        return None

    raw = _fbx_value(prop, kind)
    if kind not in ("double3", "float3"):
        attr = _fbx_kept_attr(spec, raw)
        return None if attr is None else {name: attr}

    children = [child for child, entry in written.items() if entry.get("parent") == name]
    if len(children) != 3 or any(
        _user_attr_type(written[child]) not in _FLOAT_TYPES for child in children
    ):
        return None

    attrs = {name: dict(spec)}
    for child, number in zip(children, raw):
        attrs[child] = _fbx_kept_attr(written[child], number)
    if "value" in spec:
        attrs[name]["value"] = [attrs[child]["value"] for child in children]
    return attrs


def _fbx_kept_attr(spec: dict, raw) -> Optional[dict]:
    """a spec FbxExporter kept, with its value from a property read as ``raw``;
    None for a stringArray whose string is not one

    A value the property can't give back is kept in the spec as ``[value]``,
    and taken while the property still holds what FbxExporter wrote for it,
    as a binary or an ASCII .fbx gives it back: one changed since wins.
    """
    kept = spec.get("value")
    if isinstance(kept, list) and len(kept) == 1:
        if _fbx_holds(spec, kept[0], raw):
            return {**spec, "value": kept[0]}
    elif "value" not in spec and _fbx_holds(spec, None, raw):
        return dict(spec)  # a string Maya has not set, still empty

    kind  = _user_attr_type(spec)
    value = raw
    if kind == "stringArray":
        value = _fbx_string_array(raw)
        if value is None:
            return None
    elif kind == "enum":
        # fbx keeps the position in the field list, Maya its own index
        enum_name = spec.get("enumName")
        fields    = _enum_fields(enum_name) if isinstance(enum_name, str) else []
        if 0 <= raw < len(fields):
            value = fields[raw][1]
    return {**spec, "value": value}


def _fbx_holds(spec: dict, value, raw) -> bool:
    """whether a property read as ``raw`` holds what FbxExporter wrote for the
    user defined attr ``spec`` holding ``value``, binary or ASCII"""
    held = _user_property_value(spec, value)
    return raw in (held, _ascii_read_back(held, _user_attr_type(spec)))


def _fbx_inferred_attr(prop, name: str, node) -> dict:
    """a user defined property of ``node`` with no spec, as ``{name: spec}`` the
    way Maya's own FBX importer reads it; {} for a type it makes no attribute
    of, and, with a warning, for an enum with a field an enumName can't hold"""
    kinds = {
        "eFbxDouble":  "double",
        "eFbxFloat":   "float",
        "eFbxInt":     "long",
        "eFbxShort":   "short",
        "eFbxUChar":   "byte",
        "eFbxChar":    "char",
        "eFbxBool":    "bool",
        "eFbxEnum":    "enum",
        "eFbxString":  "string",
        "eFbxDouble3": "double3",
    }
    kinds = {
        _fbx_enum(FBX, "EFbxType", fbx_type): attr_type
        for fbx_type, attr_type in kinds.items()
    }

    data_type = prop.GetPropertyDataType()
    kind      = kinds.get(data_type.GetType())
    if kind is None:
        return {}

    keyable = prop.GetFlag(FBX.FbxPropertyFlags.EFlags.eAnimatable)
    value   = _fbx_value(prop, kind)

    def spec(attr_type, held, parent=None, **extra):
        """a spec laid out the way rig serializes one"""
        if attr_type in _NEW_DATA_TYPES:
            made = {"dataType": attr_type}
        else:
            made = {"attributeType": attr_type}
        if held is not None:
            made["value"] = held
        made.update(extra)
        if parent is not None:
            made["parent"] = parent
        made["keyable"]     = keyable
        made["channel_box"] = False
        return made

    if kind == "string":
        items = _fbx_string_array(value)
        if items is not None:
            return {name: spec("stringArray", items)}
    elif kind == "enum":
        fields = [str(prop.GetEnumValue(i)) for i in range(prop.GetEnumCount())]

        # an enumName splits fields at ":" and reads "k=2", or "k==2", as field
        # k, index 2: a file's field like that would come back as other fields,
        # so its enum is left out. Maya's importer keeps such a field, but its
        # own enumName of it reads back as other fields too
        lost = [field for field in fields if _enum_fields(field) != [(field, 0)]]
        if lost:
            warnings.warn(
                f"user defined enum {node.GetName()}.{name} left out: its field "
                f"{lost[0]!r} holds ':' or reads as a field and its index, which "
                "an enumName can't hold"
            )
            return {}
        return {name: spec("enum", value, enumName=":".join(fields))}
    elif kind == "double3":
        # a color is a float3 of R, G and B, any other vector a double3 of 0, 1, 2
        if str(data_type.GetName()) == "Color":
            kind, child, suffixes = "float3", "float", "RGB"
        else:
            kind, child, suffixes = "double3", "double", "012"

        attrs = {name: spec(kind, None, numberOfChildren=3)}
        for suffix, number in zip(suffixes, value):
            attrs[name + suffix] = spec(child, number, parent=name)
        return attrs

    return {name: spec(kind, value)}


# the sdk's euler orders (xyz, xzy, yzx, yxz, zxy, zyx) as rotate_order
_FBX_ROTATE_ORDERS = {0: 0, 2: 1, 4: 2, 1: 3, 3: 4, 5: 5}


# the names node_types takes, the kinds of node the fbx loaders can keep: an
# eSkeleton is a joint, an eMarker or a null drawn as a cross a locator, any
# other null a transform (a group), and an eMesh a mesh, read as a transform
_FBX_NODE_TYPES = ("joint", "locator", "transform", "mesh")


def _fbx_check_node_types(node_types) -> Optional[frozenset]:
    """the kinds ``node_types`` keeps (:data:`_FBX_NODE_TYPES`), None for
    every one of them; refuses anything else"""
    if node_types is None:
        return None
    if isinstance(node_types, str) or not isinstance(node_types, Iterable):
        raise TypeError(
            f"node_types takes a tuple of names, such as ('joint',), or None, not {node_types!r}"
        )

    names = tuple(node_types)
    for name in names:
        if not (isinstance(name, str) and name in _FBX_NODE_TYPES):
            known = ", ".join(repr(x) for x in _FBX_NODE_TYPES)
            raise ValueError(f"node_types names {known}, not {name!r}")
    if not names:
        raise ValueError("node_types keeps no node; None keeps every kind")
    return frozenset(names)


class _FbxTree:
    """what :func:`_fbx_fold` reads of a scene: the nodes :func:`_walk_fbx`
    keeps and every node above one of them, the scene root included, in the
    order the walk met them, so a parent comes before its children

    Per node, (n,) or (n, 3) arrays: ``parent``, its parent's position, -1
    for the scene root; ``kept``, its column among the walk's dicts, -1 for
    a node the walk drops; ``joint``, whether it is a joint, and
    ``compensate``, whether it is a joint that divides its parent's scale
    out (``eInheritRrs``, segment scale compensate); ``orders``, its rotate
    order as rotate_order, the one it turns in: xyz where the read takes no
    pre rotation (below), as the sdk turns a node whose rotation is
    inactive; ``blend``, the fbx rotate order the sdk blends a rotation by
    layer in (:func:`_fbx_blend`): the stored one, rotation active or not,
    but xyz at a root of the file, kept or dropped, that the read turns in
    xyz where it turns the file into Maya's axes, as the sdk's ConvertScene
    turns such a root's rotation on in xyz (a node kept under a dropped
    root is none); its own ``translate`` (times the scale factor),
    ``rotate`` and ``scale``; ``pre``, its pre rotation, and ``axis``, the
    inverse of its post rotation, as euler angles in xyz, 0 where the sdk
    applies none (rotation inactive) and on a light, whose post rotation is
    fbx's view axis; on a camera, what its post rotation turns beyond fbx's
    view axis, (0, -90, 0), which Maya's camera holds as its rotate axis
    (:data:`_FBX_CAMERA_AXIS`), all of it where Maya reads no post rotation.
    A camera's or a light's pre rotation and rotate order, and a camera's
    post rotation, are read as Maya's importer reads them, rotation active
    or not, but for a root with its rotation inactive in a file whose up
    axis is x or z, which reads none of them; and ``pivots`` (n, 4, 3), its
    rotate pivot, scale pivot, rotate offset and scale offset, times the
    scale factor. ``nodes`` holds each one's ``FbxNode``, ``columns`` the
    position of each kept column's node and ``above`` each kept column's
    nearest kept ancestor's column, -1 for a root.

    The scale factor is the node's own (:class:`_FbxUnits`): ``lengths``
    (n,) holds what its translate, curves included, is multiplied by, the
    file's unit in cm, but 1 under a unit root, and ``unit_scale`` (n, 3)
    what its scale, curves included, is divided by, the unit root's own
    scale on that root and on each node under it that does not inherit its
    parent's scale, 1 elsewhere; ``unit_roots`` (n,) says which nodes are
    unit roots, whose curves only a take's first layer converts
    (:func:`_fbx_sample_take`). ``translate``, ``scale`` and ``pivots``
    hold them applied, a unit root's pivots as :func:`_walk_fbx` reads them.
    """

    __slots__ = (
        "nodes",
        "parent",
        "kept",
        "columns",
        "above",
        "joint",
        "compensate",
        "orders",
        "blend",
        "translate",
        "rotate",
        "scale",
        "pre",
        "axis",
        "pivots",
        "lengths",
        "unit_scale",
        "unit_roots",
    )


# the post rotation Maya's exporter writes on a camera, (0, -90, 0), fbx's
# view axis, as a row vector matrix: Maya's importer keeps what a camera's
# post rotation P turns beyond it as the camera's rotate axis, this @ P^-1,
# which a camera's children follow (probe_views.py: post (0, 0, 0) gives a
# rotate axis of (0, -90, 0), (0, -90, 20) one of (-20, 0, 0))
_FBX_CAMERA_AXIS = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _walk_fbx(
    scene,
    units:      "_FbxUnits",
    node_types: Optional[frozenset] = None,
    turned:     bool                = False,
):
    """walks an open fbx scene's nodes into TransformData dicts

    Keeps the joints, locators, groups and meshes (a mesh read as a
    transform), or the kinds ``node_types`` names
    (:func:`_fbx_check_node_types`), and drops every other node: one of
    another kind, one with no node attribute, and the scene root. A kept
    node's parent is its nearest kept ancestor. What the nodes dropped
    between them hold, and the pivots, offsets and the pre and post rotation
    of a node that is no joint, which its channels can't hold, are folded
    into its channels (:func:`_fbx_fold`), so its world matrix is the
    file's. ``turned`` says the read turns the file into Maya's axes
    (:func:`_fbx_axis_turn`), which only the orders a take's layers blend
    in follow (``blend``, :class:`_FbxTree`). ``units`` (:func:`_fbx_units`)
    says what each node's lengths are multiplied by and its scale divided
    by. A unit root's pivots are read through the scale it holds, 1/s of
    the one it loads with: the rotate pivot and both offsets are in the
    file's unit, the scale pivot in the root's own space, in cm, and the
    scale offset takes what the scale pivot, applied on either side of that
    scale, moves, ``so s + sp (s - 1)``, so its local matrix is the file's
    with the unit taken out.

    Hands back the dicts, the ``FbxNode`` behind each uuid, because an
    animated read has to sample those same objects and fbx lets two nodes
    share a name, so pairing them up afterwards by name would be ambiguous,
    and the :class:`_FbxTree` the fold read, which a clip folds again on
    every frame.

    The nodes live only as long as the ``open_fbx`` block the scene came from.
    """
    skeleton     = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eSkeleton")
    marker       = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eMarker")
    null         = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eNull")
    mesh         = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eMesh")
    camera       = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eCamera")
    light        = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eLight")
    cross        = _fbx_enum(FBX.FbxNull,          "ELook",        "eCross")
    source_pivot = _fbx_enum(FBX.FbxNode,          "EPivotSet",    "eSourcePivot")
    inherit_rrs  = _fbx_enum(FBX.FbxTransform,     "EInheritType", "eInheritRrs")

    # Maya's importer reads a camera's or a light's pre rotation and rotate
    # order, and a camera's post rotation, whether its rotation is active or
    # not, but on a root of a file whose up axis is x or z: it turns that
    # file into its Y-up with the sdk's ConvertScene, which drops a root's
    # pre and post rotation where its rotation is inactive, and turns that
    # root in xyz (view_rules.py, convert_views.py)
    up, _ = scene.GetGlobalSettings().GetAxisSystem().GetUpVector()
    converts = int(getattr(up, "value", up)) != 2

    # every node, parents first, in the order the walk has always met them:
    # a node, then each of its children's branches in turn
    found, parents, kinds, views = [], [], [], []
    stack = [(scene.GetRootNode(), -1)]
    while stack:
        node, parent = stack.pop()
        attr = node.GetNodeAttribute() if parent >= 0 else None
        kind = None
        view = None
        if attr:
            attr_type = attr.GetAttributeType()
            if attr_type == skeleton:
                kind = "joint"
            elif attr_type == marker:
                kind = "locator"
            # a null drawn as a cross is a locator, a bare one is a group,
            # which is how Maya's own FBX plugin tells them apart
            elif attr_type == null:
                kind = "locator" if attr.Look.Get() == cross else "transform"
            elif attr_type == mesh:
                kind = "mesh"
            elif attr_type == camera:
                view = "camera"
            elif attr_type == light:
                view = "light"

        found.append(node)
        parents.append(parent)
        kinds.append(kind)
        views.append(view)
        here = len(found) - 1
        stack.extend((node.GetChild(i), here) for i in reversed(range(node.GetChildCount())))

    keeps = [kind is not None and (node_types is None or kind in node_types) for kind in kinds]

    # the tree: the kept nodes and every node above one, renumbered
    needed = list(keeps)
    for index in reversed(range(len(found))):
        if needed[index] and parents[index] >= 0:
            needed[parents[index]] = True
    place = np.cumsum(needed) - 1
    rows  = [index for index in range(len(found)) if needed[index]]

    tree            = _FbxTree()
    tree.nodes      = [found[index] for index in rows]
    tree.parent     = np.array([place[parents[i]] if parents[i] >= 0 else -1 for i in rows], dtype=np.intp)
    tree.kept       = np.full(len(rows), -1, dtype=np.intp)
    tree.joint      = np.array([kinds[i] == "joint" for i in rows], dtype=bool)
    tree.compensate = np.zeros(len(rows), dtype=bool)
    tree.orders     = np.zeros(len(rows), dtype=np.intp)
    tree.blend      = np.zeros(len(rows), dtype=np.intp)
    tree.translate  = np.zeros((len(rows), 3))
    tree.rotate     = np.zeros((len(rows), 3))
    tree.scale      = np.ones((len(rows), 3))
    tree.pre        = np.zeros((len(rows), 3))
    tree.axis       = np.zeros((len(rows), 3))
    tree.pivots     = np.zeros((len(rows), 4, 3))
    tree.lengths    = np.full(len(rows), units.factor)
    tree.unit_scale = np.ones((len(rows), 3))
    tree.unit_roots = np.zeros(len(rows), dtype=bool)

    # the nodes inside a unit root hold cm already: the lengths under them
    # are read as they are, and the scale of each that holds the unit is
    # divided by the root's
    ids    = [node.GetUniqueID() for node in found] if units.inner else None
    inside = [uid in units.inner for uid in ids] if units.inner else None

    hierarchy_data = {}
    nodes          = {}
    uuids          = {}  # a tree position's uuid, for the kept nodes
    above          = []

    for position, index in enumerate(rows):
        node = found[index]
        if inside is not None:
            if parents[index] >= 0 and inside[parents[index]]:
                tree.lengths[position] = 1.0
            if ids[index] in units.reset:
                tree.unit_scale[position] = units.reset[ids[index]]
        length = tree.lengths[position]

        # fbx ignores the rotate order and the pre and post rotation entirely
        # unless rotationActive is on: the sdk turns such a node in xyz.
        # Maya's importer reads a view's order and pre rotation as if it were
        # on, and so does the read. On any other node it drops the pre and
        # post rotation too, but turns the node by neither order: it keeps
        # the stored order on the xyz angles of that order's turn
        # (probe_rotactive_rule.py), where the read keeps the sdk's xyz
        active = node.GetRotationActive() or (views[index] is not None and not (converts and parents[index] == 0))

        # rotate_order, which newer bindings hand back as an enum
        stored                = node.GetRotationOrder(source_pivot)
        stored                = int(getattr(stored, "value", stored))
        tree.orders[position] = _FBX_ROTATE_ORDERS[stored if active else 0]

        # the sdk blends a rotation by layer in the stored order, rotation
        # active or not; but where the read turns the file into Maya's axes
        # it stands for ConvertScene, which turns on, in xyz, the rotation of
        # a root of the file whose rotation is inactive, kept or dropped (a
        # node kept under a dropped root is none), in every axis system but
        # Maya's Y-up, and the sdk then blends that root as an xyz node
        # (probe_axes.py). A view whose order the read keeps blends in it
        tree.blend[position] = 0 if turned and not active and parents[index] == 0 else stored

        # get SRT
        tree.scale[position]     = np.array(tuple(node.LclScaling.Get())[:3]) / tree.unit_scale[position]
        tree.rotate[position]    = tuple(node.LclRotation.Get())[:3]
        tree.translate[position] = np.array(tuple(node.LclTranslation.Get())[:3]) * length

        # pivots and offsets turn and scale the node about other points than
        # its origin, rotation active or not
        for k, pivot in enumerate(
            (
                node.GetRotationPivot(source_pivot),
                node.GetScalingPivot(source_pivot),
                node.GetRotationOffset(source_pivot),
                node.GetScalingOffset(source_pivot),
            )
        ):
            tree.pivots[position, k] = np.array(tuple(pivot)[:3]) * length

        # a unit root's scale pivot is in its own space, in cm, and applied
        # on either side of the 1/s scale the file holds: its scale offset
        # takes what that moves (_walk_fbx)
        if inside is not None and ids[index] in units.roots:
            tree.unit_roots[position] = True
            sp                        = np.array(tuple(node.GetScalingPivot(source_pivot))[:3])
            tree.pivots[position, 1]  = sp
            tree.pivots[position, 3]  = tree.pivots[position, 3] + sp * (length - 1.0)

        # the pre rotation where the read takes it, as above
        if active:
            tree.pre[position] = tuple(node.GetPreRotation(source_pivot))[:3]

        # fbx composes the post rotation as its inverse. A light's is fbx's
        # view axis, which Maya places its children without; a camera's turns
        # them by what it adds to that view axis, as Maya's importer reads it,
        # all of it where it reads no post rotation
        if (active and not views[index]) or views[index] == "camera":
            post = FBX.FbxAMatrix()
            if active:
                post.SetR(node.GetPostRotation(source_pivot))
            axis = tuple(post.Inverse().GetR())[:3]
            if views[index] == "camera":
                turn = _FBX_CAMERA_AXIS @ _eulers_to_matrices(axis, 0)[:3, :3]
                axis = 0.0 if np.abs(turn - np.eye(3)).max() <= _FBX_FOLD_BAND else _matrices_to_eulers(turn, 0)
            tree.axis[position] = axis

        if kinds[index] == "joint":
            tree.compensate[position] = node.GetTransformationInheritType() == inherit_rrs

        if not keeps[index]:
            continue

        # a node the walk drops has no uuid to hand down, so its children take
        # the nearest ancestor it kept rather than falling to the world
        up = parents[index]
        while up >= 0 and not keeps[up]:
            up = parents[up]
        above.append(tree.kept[place[up]] if up >= 0 else -1)

        # basics
        unique_id    = generate_uuid()
        data         = {}
        data["name"] = node.GetName()
        data["uuid"] = unique_id
        if up >= 0:
            data["parent_uuid"] = uuids[place[up]]

        data["rotate_order"] = int(tree.orders[position])
        data["scale"]        = tree.scale[position].copy()
        data["rotate"]       = tree.rotate[position].copy()
        data["translate"]    = tree.translate[position].copy()

        # if joint, get joint_orient
        if kinds[index] == "joint":
            data["node_type"] = "joint"
            if node.GetRotationActive():
                data["joint_orient"] = tree.pre[position].copy()
                data["rotate_axis"]  = tree.axis[position].copy()
            data["segment_scale_compensate"] = bool(tree.compensate[position])
        elif kinds[index] == "locator":
            data["node_type"] = "locator"
        else:
            data["node_type"] = "transform"

        attrs = _fbx_user_attributes(node)
        if attrs:
            data["user_defined_attributes"] = attrs

        tree.kept[position]       = len(hierarchy_data)
        uuids[position]           = unique_id
        hierarchy_data[unique_id] = data
        nodes[unique_id]          = node

    tree.columns = np.flatnonzero(tree.kept >= 0)
    tree.above   = np.array(above, dtype=np.intp)

    # what the file holds that the dicts can't: folded in, on the static values
    translate, rotate, scale, joint_orient, changed = _fbx_fold(
        tree, tree.translate[None], tree.rotate[None], tree.scale[None]
    )
    for column, data in enumerate(hierarchy_data.values()):
        if not changed[column]:
            continue
        data["translate"] = translate[0, column]
        data["rotate"]    = rotate[0, column]
        data["scale"]     = scale[0, column]
        if data["node_type"] == "joint":
            data["joint_orient"] = joint_orient[0, column]

    return hierarchy_data, nodes, tree


# a 3x3 block this close to a rotation, or to a rotation times one scale, as
# a fraction of its size, is one: the products of rotations a fold forms are
# a few ulps off one
_FBX_FOLD_BAND = 1e-12


def _fbx_fold_split(block: np.ndarray) -> tuple:
    """(F, 3, 3) blocks as ``(U, s, uneven)``: where a block is a rotation U
    times one positive scale s, those, s exactly 1 for a rotation; anywhere
    else (``uneven`` (F,)) the rotation nearest the block and s 1, and NaN
    for a block holding a value that is not finite, which has no nearest
    rotation (a file can carry one)"""
    finite = np.isfinite(block).all(axis=(-2, -1))
    with np.errstate(invalid="ignore"):
        gram  = block @ np.swapaxes(block, -1, -2)
        size  = np.trace(gram, axis1=-2, axis2=-1) / 3.0
        eye   = np.eye(3)
        turns = _det3(block) > 0.0
        even  = turns & (np.abs(gram - size[:, None, None] * eye).max(axis=(-2, -1)) <= _FBX_FOLD_BAND * size)
        exact = turns & (np.abs(gram - eye).max(axis=(-2, -1)) <= _FBX_FOLD_BAND)

    s     = np.where(exact, 1.0, np.sqrt(np.where(even, size, 1.0)))
    U     = block / s[:, None, None]
    solve = ~even & finite
    if solve.any():
        # the rotation nearest an uneven block, a mirror in it left to the scale
        w, _, vt = np.linalg.svd(block[solve])
        w[..., 2] *= np.sign(_det3(w @ vt))[:, None]
        U[solve] = w @ vt
    U[~finite] = np.nan
    return U, s, ~even


def _fbx_fold(tree: _FbxTree, translate, rotate, scale) -> tuple:
    """the kept nodes' channels, (F, k, 3) each, from those of every node of
    ``tree`` as the file holds them, (F, n, 3) each, every frame at once:
    ``(translate, rotate, scale, joint_orient, changed)``, ``changed`` (k,)
    saying which kept nodes the fold moved off the file's values

    Row vectors. The sdk composes a node's local matrix as Sp^-1 S Sp Soff
    Rp^-1 RA R JO Rp Roff T: Sp and Rp the scale and rotate pivots, Soff and
    Roff the offsets, RA the inverse of the post rotation and JO the pre
    rotation, both only with rotation active, and a joint that compensates
    its parent's scale divides it out of its 3x3 block. The pivots and
    offsets come to a translation, translate + Roff + Rp + (Soff + Sp - Rp -
    Sp S) Q with Q = RA R JO, which a kept node takes in its translate, on
    every frame. A kept node's local matrix under its nearest kept ancestor
    is its own times those of the nodes dropped between, D: its translate
    becomes t D + d, d D's translation, and E, the 3x3 block D leaves (the
    compensated scale the fbx parent and the kept parent divide out traded),
    goes on the rotation:

    - a joint keeps its scale, rotate axis and rotate, and takes E in its
      joint orient, JO E, the way the turn to Maya's axes goes on a joint
      root, so its rotate curves stay as authored;
    - any other node, which the walk gives no rotate axis or joint orient,
      takes RA R JO E in its rotate, solved again in its rotate order;
    - E a rotation U times one scale s puts s on the node's scale; a block
      no node can hold under a turn, scales uneven, takes its nearest
      rotation in place of U, and the scale and the rotate are split out
      of the rest as the matrix setter splits them, shear dropped. What it
      drops, the block it should hold times the inverse of the one it
      holds (:func:`_fbx_fold_shear`), goes on the kept nodes under it, so
      they stay where the file puts them, under a scale under 1e-12 too: a
      joint that compensates its parent's scale takes it in its block as a
      dropped node's matrix, split again; any other node keeps the
      channels it holds without it, takes it in its translate alone, and
      hands on what it should hold times the inverse of what it holds
      (:func:`_fbx_hand_on`). On a frame the node is scaled exactly 0
      along an axis the shear has no inverse and stays out of them. The
      split keeps the length of each row of the block, so a node's axes
      are the file's length where its kept parent's world scales evenly;
      elsewhere which way they point and how long they are is the split's.

    A scale under 1e-12 in size is read as 1e-12, as Maya's evaluation holds
    it (:func:`_fbx_maya_scale`), wherever the fold multiplies or divides by
    one Maya holds: a dropped node's own, and the scale of a dropped parent
    that a compensating joint divides out, with the kept parent's it is then
    traded for. So a joint that compensates a dropped node scaled to 0 keeps
    its axes and sits on that node's position, and a node that does not
    collapses onto it, its axes 1e-12 long, as Maya places them; the sdk's
    own evaluation gives NaN there. A kept node's own scale keeps the file's
    value, 0 included, and a joint compensating it, dropped or kept, divides
    out that value, which world_matrix multiplies by, 0 dividing nothing; a
    kept one is traded for the kept node's scale out of the fold as it is,
    so what the fold put on that scale, a dropped node's above it, stays in
    the joint, as in Maya, where the kept node's held scale cancels. So the
    joints and nodes under a kept node scaled under 1e-12 sit where Maya
    puts them, down to about 1e-154, under which the fold's products
    overflow; along an axis it is scaled exactly 0, those under a dropped
    joint collapse onto it (Maya holds 1e-12 there, which the kept node does
    not) and a kept joint's world_matrix is NaN. A rotation solved again
    goes through euler_filter along the frames, frame 0 as solved. A kept
    node nothing moves keeps the file's values to the bit, and so does every
    node of a file whose dropped nodes, pivots and offsets hold the
    identity. The kept nodes go parents first, so a compensating joint
    divides out its kept parent's scale as the fold leaves it.
    """
    frames  = len(translate)
    kept    = tree.kept
    parent  = tree.parent
    columns = tree.columns
    count   = len(columns)
    eye     = np.eye(3)

    out_translate = translate[:, columns].copy()
    out_rotate    = rotate[:, columns].copy()
    out_scale     = scale[:, columns].copy()
    out_orient    = np.repeat(tree.pre[columns][None], frames, axis=0)
    changed       = np.zeros(count, dtype=bool)

    pivoted = np.any(tree.pivots != 0.0, axis=(1, 2))
    turned  = np.any(tree.pre != 0.0, axis=1) | np.any(tree.axis != 0.0, axis=1)

    # the scale a node's own block divides out: its parent's, when it
    # compensates, held as Maya holds a scale, so a joint compensating a
    # node scaled to 0 divides out 1e-12 and keeps its axes, as in Maya. One
    # under a kept node, kept or dropped, divides out that node's own scale,
    # which the kept node keeps and world_matrix multiplies by, 0 dividing
    # nothing: the 1e-12 would cancel Maya's held scale, which no kept node
    # holds
    divided           = np.ones((frames, len(kept), 3))
    rows              = np.flatnonzero(tree.compensate & (parent >= 0))
    divided[:, rows]  = _fbx_maya_scale(scale[:, parent[rows]])
    under             = rows[kept[parent[rows]] >= 0]
    own               = scale[:, parent[under]]
    divided[:, under] = np.where(own == 0.0, 1.0, own)
    maya_scale        = _fbx_maya_scale(scale)

    # a dropped node that holds the identity on every frame moves nothing
    still = (
        (kept < 0)
        & ~pivoted
        & ~turned
        & np.all(translate == 0.0, axis=(0, 2))
        & np.all(rotate == 0.0, axis=(0, 2))
        & np.all(scale == 1.0, axis=(0, 2))
        & np.all(divided == 1.0, axis=(0, 2))
    )

    def orientations(i: int) -> np.ndarray:
        """(F, 3, 3) RA R JO of node i"""
        R = _eulers_to_matrices(rotate[:, i], tree.orders[i])[:, :3, :3]
        if not turned[i]:
            return R
        RA, JO = _eulers_to_matrices(np.array([tree.axis[i], tree.pre[i]]), 0)[:, :3, :3]
        return RA @ R @ JO

    def offset(i: int, Q: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """(F, 3) node i's translate with its pivots and offsets in it, for
        its scale ``sizes``"""
        if not pivoted[i]:
            return translate[:, i]
        rp, sp, ro, so = tree.pivots[i]
        moved = (so + sp - rp) - sp * sizes[:, i]
        return translate[:, i] + ((ro + rp) + np.einsum("fi,fij->fj", moved, Q))

    def local(i: int) -> np.ndarray:
        """(F, 4, 4) dropped node i's local matrix, its parent's scale
        divided out, its scales as Maya holds them"""
        Q                 = orientations(i)
        matrix            = np.zeros((frames, 4, 4))
        matrix[:, :3, :3] = maya_scale[:, i, :, None] * Q / divided[:, i, None, :]
        matrix[:, 3, :3]  = offset(i, Q, maya_scale)
        matrix[:, 3, 3]   = 1.0
        return matrix

    # each dropped node's matrix times those of the dropped nodes above it,
    # up to the nearest kept one: None for the identity
    chain = {}
    for i in np.flatnonzero(kept < 0).tolist():
        up    = parent[i]
        upper = chain.get(up) if up >= 0 and kept[up] < 0 else None
        if still[i]:
            chain[i] = upper
        else:
            chain[i] = local(i) if upper is None else local(i) @ upper

    # the shear a kept node's split drops, per column, (F, 4, 4), on the
    # frames it drops any: the kept nodes under it take it as they take a
    # dropped node's matrix, so they stay where the file puts them
    sheared = {}

    for column, i in enumerate(columns.tolist()):
        up       = parent[i]
        ancestor = tree.above[column]
        D        = chain.get(up) if up >= 0 and kept[up] < 0 else None

        # the shear the kept parent's split dropped: a compensating joint
        # takes it in its block, split again; any other node keeps the
        # channels it holds without it, takes it in its translate alone and
        # hands it on, so its rotate and scale stay the file's
        lost = sheared.get(ancestor)
        if lost is not None and tree.compensate[i]:
            D, lost = (lost if D is None else D @ lost), None
        moved = D if lost is None else (lost if D is None else D @ lost)
        block = None if D is None else D[:, :3, :3]

        # a compensating joint divides its fbx parent's scale out, which the
        # dicts trade for its kept parent's, as the fold leaves it. Where the
        # fbx parent is that kept parent, as it is, 0 as 1: the joint divides
        # out the parent's own scale, as world_matrix multiplies by it, and
        # what the fold put on that scale, a dropped node's above it, stays
        # in the joint, as in Maya, where the parent's held scale cancels;
        # under a dropped fbx parent, as Maya holds it
        if tree.compensate[i]:
            if ancestor < 0:
                held = np.ones((frames, 3))
            elif kept[up] >= 0:
                held = np.where(out_scale[:, ancestor] == 0.0, 1.0, out_scale[:, ancestor])
            else:
                held = _fbx_maya_scale(out_scale[:, ancestor])
            if block is not None or not np.array_equal(divided[:, i], held):
                block = (eye if block is None else block) / divided[:, i, :, None] * held[:, None, :]

        level = block is None or bool(np.all(block == eye))
        if not (pivoted[i] or moved is not None or not level or (turned[i] and not tree.joint[i])):
            continue

        Q = orientations(i)
        t = offset(i, Q, scale)
        if moved is not None:
            t = np.einsum("fi,fij->fj", t, moved[:, :3, :3]) + moved[:, 3, :3]
        out_translate[:, column] = t
        changed[column]          = True

        if level and (tree.joint[i] or not turned[i]):
            if lost is not None:
                _fbx_hand_on(sheared, column, scale[:, i, :, None] * Q @ moved[:, :3, :3], scale[:, i], Q)
            continue

        U, s, uneven = _fbx_fold_split(np.broadcast_to(eye, (frames, 3, 3)) if block is None else block)
        order = tree.orders[i]
        if uneven.any():
            whole = (scale[uneven, i, :, None] * Q[uneven]) @ block[uneven]
        if tree.joint[i]:
            # the joint orient takes the turn, the scale its size
            JO                    = _eulers_to_matrices(tree.pre[i], 0)[:3, :3] @ U
            out_orient[:, column] = _filtered(_matrices_to_eulers(JO, 0)[:, None], [0])[:, 0]
            out_scale[:, column]  = scale[:, i] * s[:, None]
            if uneven.any():
                RA      = _eulers_to_matrices(tree.axis[i], 0)[:3, :3]
                current = RA @ _eulers_to_matrices(rotate[uneven, i], order)[:, :3, :3]
                whole   = whole @ np.swapaxes(JO[uneven], -1, -2)
                sized, turn = _scales_and_rotations(whole, current)
                out_scale[uneven, column]  = sized
                out_rotate[uneven, column] = _matrices_to_eulers(RA.T @ turn, order)
                out_rotate[:, column]      = _filtered(out_rotate[:, column][:, None], [order])[:, 0]
        else:
            # the rotate takes the pre and post rotation and the turn
            solved = _matrices_to_eulers(Q @ U, order)
            sized  = scale[:, i] * s[:, None]
            if uneven.any():
                sized[uneven], turn = _scales_and_rotations(whole, Q[uneven])
                solved[uneven] = _matrices_to_eulers(turn, order)
            out_scale[:, column]  = sized
            out_rotate[:, column] = _filtered(solved[:, None], [order])[:, 0]

        if lost is not None:
            # the rotation the node holds, rotate axis and joint orient in
            holds = _eulers_to_matrices(out_rotate[:, column], order)[:, :3, :3]
            if tree.joint[i]:
                RA    = _eulers_to_matrices(tree.axis[i], 0)[:3, :3]
                holds = RA @ holds @ _eulers_to_matrices(out_orient[:, column], 0)[:, :3, :3]
            _fbx_hand_on(sheared, column, scale[:, i, :, None] * Q @ moved[:, :3, :3], out_scale[:, column], holds)
        elif uneven.any():
            # the rotation the node holds, its joint orient left out
            holds = _eulers_to_matrices(out_rotate[uneven, column], order)[:, :3, :3]
            if tree.joint[i]:
                holds = RA @ holds
            lost = _fbx_fold_shear(whole, out_scale[uneven, column], holds, frames, uneven)
            if lost is not None:
                sheared[column] = lost

    return out_translate, out_rotate, out_scale, out_orient, changed


def _fbx_hand_on(sheared: dict, column: int, whole, sized, holds) -> None:
    """puts in ``sheared[column]`` what a kept node under a sheared one hands
    on to the nodes under it, on every frame, where any frame hands on any:
    ``whole`` (F, 3, 3), the block it should hold under its kept parent,
    times the inverse of the one its channels hold, ``sized`` (F, 3) times
    the rotation ``holds`` (F, 3, 3) (:func:`_fbx_fold_shear`)"""
    frames = len(whole)
    lost   = _fbx_fold_shear(whole, sized, holds, frames, np.ones(frames, dtype=bool))
    if lost is not None:
        sheared[column] = lost


# the least size Maya's evaluation gives a scale: one under it, 0 among them,
# is held as this, positive, so a matrix always has an inverse and a joint
# compensating a node scaled to 0 keeps its axes (maya_zero_clamp.py: 0,
# -0, 1e-13, -1e-13 and 5e-13 are held as 1e-12; 1e-12, -1e-12 and 2e-12 as
# they are)
_FBX_LEAST_SCALE = 1e-12


def _fbx_maya_scale(scale: np.ndarray) -> np.ndarray:
    """``scale`` as Maya's evaluation holds it: a value under
    :data:`_FBX_LEAST_SCALE` in size as that, positive"""
    return np.where(np.abs(scale) < _FBX_LEAST_SCALE, _FBX_LEAST_SCALE, scale)


def _fbx_fold_shear(whole, sized, turn, frames: int, uneven) -> Optional[np.ndarray]:
    """(F, 4, 4) the shear a split dropped, ``whole @ inv(sized turn)`` on
    the ``uneven`` frames, the identity on the others; None where every
    frame drops none, within :data:`_FBX_FOLD_BAND`

    ``whole`` (u, 3, 3) is the block the node should hold, its joint orient
    taken out, and ``sized`` and ``turn`` the scale and the rotation its
    channels hold; a frame where a scale is 0, or a value is not finite,
    has no inverse, what it would drop not finite, and keeps the
    identity. A scale however small has one, and keeps its shear: a
    joint under the node that compensates it divides that scale out
    again.
    """
    eye = np.eye(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        lost = whole @ np.swapaxes(turn, -1, -2) / sized[:, None, :]
    ok = np.isfinite(lost).all(axis=(-2, -1))
    ok &= np.abs(lost - eye).max(axis=(-2, -1)) > _FBX_FOLD_BAND
    if not ok.any():
        return None

    out               = np.repeat(np.eye(4)[None], frames, axis=0)
    rows              = np.flatnonzero(uneven)[ok]
    out[rows, :3, :3] = lost[ok]
    return out


def _fbx_has_curves(stack) -> bool:
    """whether an animation stack holds any curve node, with or without a curve

    The criteria have to be built here rather than at import: they match
    nothing unless an FbxManager already exists.
    """
    layers = FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId)
    curves = FBX.FbxCriteria.ObjectType(FBX.FbxAnimCurveNode.ClassId)

    for index in range(stack.GetSrcObjectCount(layers)):
        layer = typed(stack.GetSrcObject(layers, index), FBX.FbxAnimLayer, FBX)
        if layer.GetSrcObjectCount(curves):
            return True

    return False


def _fbx_scale_factor(scene, scale_factor: Optional[float]) -> float:
    """what a loader multiplies a scene's lengths by: ``scale_factor`` when
    the caller gives one, otherwise the file's unit in centimeters, 1.0 for
    a file in cm and 100.0 for one in meters"""
    if scale_factor is not None:
        return float(scale_factor)
    return float(scene.GetGlobalSettings().GetSystemUnit().GetScaleFactor())


# a child of the scene root scaled evenly within this fraction of 1/s, s the
# file's unit in cm, holds that unit (_fbx_units)
_FBX_UNIT_BAND = 1e-9


class _FbxUnits:
    """what a read multiplies a scene's lengths by (:func:`_fbx_units`):
    ``factor``, the file's unit in cm or the caller's number; ``inner``,
    the unique ids of the nodes whose own space holds cm already, each unit
    root and every node under one; ``roots``, the unit roots'; and
    ``reset``, ``{unique id: (3,) scale}``, the nodes whose scale holds the
    unit, each unit root and every node under one that does not inherit its
    parent's scale, with what that scale is divided by, the root's own
    scale"""

    __slots__ = ("factor", "inner", "roots", "reset")

    def __init__(self, factor: float, inner=frozenset(), roots=frozenset(), reset=None):
        self.factor = float(factor)
        self.inner  = inner
        self.roots  = roots
        self.reset  = {} if reset is None else reset

    def space(self, node) -> float:
        """what the lengths ``node``'s own space holds, its children's
        translates and a mesh's points, are multiplied by: 1 inside a unit
        root, the factor elsewhere"""
        return 1.0 if self.inner and node.GetUniqueID() in self.inner else self.factor


def _fbx_units(scene, scale_factor: Optional[float], sdk=None) -> _FbxUnits:
    """the :class:`_FbxUnits` a loader reads ``scene`` with, for the
    ``scale_factor`` it was given (:func:`_fbx_scale_factor`)

    The sdk's ``FbxSystemUnit.ConvertScene`` and Maya's unit converting
    export (``FBXExportConvertUnitString``) write a scene in another unit
    without converting its values: they multiply by 1/s, s the cm one unit
    of the new file holds, the scale of each child of the scene root and of
    every node that does not inherit its parent's scale (``eInheritRrs``, a
    joint's segment scale compensate), the curves of each take's first
    layer with it. They convert each child of the scene root's translate
    and pivots, its translate curves on that first layer, and every matrix
    of the bind poses and the skin clusters as a world matrix, and leave
    every other value as it was, in cm for a Maya scene: the meshes'
    points, the translates under the roots, the curves of every later
    layer. (probe_sdk_convert.py, probe_maya_convert.py and
    probe_layers_convert.py, under the sdk and Maya 2025.)

    So with ``scale_factor`` None a child of the scene root whose scale is
    1/s on all three axes, within :data:`_FBX_UNIT_BAND` of it, holds the
    file's unit, a unit root: its own translate is in the file's unit, every
    length under it already in cm, and its scale, and that of each node
    under it that does not inherit its parent's scale, is divided by the
    root's own, so the root's is 1. A file only labelled in a unit, its
    values in that unit, holds no unit root and has every length multiplied
    by s; so does a root the export found scaled other than 1, and a number
    given for ``scale_factor`` reads none at all. The one risk: a root whose
    own scale is exactly 1/s is read as the unit, and every node under it
    that does not inherit its parent's scale as holding it too. ``sdk`` is
    the fbx module the scene was read with, this module's by default.
    """
    factor = _fbx_scale_factor(scene, scale_factor)
    if scale_factor is not None or factor == 1.0:
        return _FbxUnits(factor)

    sdk   = FBX if sdk is None else sdk
    rrs   = _fbx_enum(sdk.FbxTransform, "EInheritType", "eInheritRrs")
    inner = set()
    roots = set()
    reset = {}
    top   = scene.GetRootNode()
    for index in range(top.GetChildCount()):
        root  = top.GetChild(index)
        scale = np.array(tuple(root.LclScaling.Get())[:3], dtype=float)
        if not np.all(np.abs(scale * factor - 1.0) <= _FBX_UNIT_BAND):
            continue

        # fbx lets two nodes share a name, and the binding's == holds for
        # any two: a node is known by its unique id
        stack = [root]
        while stack:
            node = stack.pop()
            uid  = node.GetUniqueID()
            inner.add(uid)
            if node.GetTransformationInheritType() == rrs:
                reset[uid] = scale
            stack.extend(node.GetChild(i) for i in range(node.GetChildCount()))
        roots.add(root.GetUniqueID())
        reset[root.GetUniqueID()] = scale

    return _FbxUnits(factor, frozenset(inner), frozenset(roots), reset)


def _fbx_check_axis_system(axis_system) -> None:
    """refuses an ``axis_system`` the fbx loaders don't know"""
    if not (axis_system is None or (isinstance(axis_system, str) and axis_system == "maya_y_up")):
        raise ValueError(f"axis_system must be 'maya_y_up' or None, got {axis_system!r}")


def _fbx_axis_turn(scene, axis_system, filename, sdk=None) -> Optional[np.ndarray]:
    """the (3, 3) rotation from a scene's axes to ``axis_system``'s, its rows
    the file's x, y and z in Maya's Y-up; None when there is nothing to turn:
    ``axis_system`` None, or a file already in Maya's Y-up

    Derived from the file's ``FbxAxisSystem``: its up vector goes to Maya's
    y, its front vector to z, and the third axis, up x front in a right
    handed system, to x. That is the turn ``FbxAxisSystem.MayaYUp
    .ConvertScene`` gives a scene, measured on all 24 right handed axis
    systems a file can hold. Maya's FBX importer gives the same turn to the
    16 whose up axis is x or z, and none to the 8 whose up axis is y, either
    way up: it leaves those in their own axes. A left handed one raises
    ValueError: no rotation takes it to Maya's axes, only a mirror would.
    ``sdk`` is the fbx module the scene was read with, this module's by
    default.
    """
    if axis_system is None:
        return None

    up_axis, up_sign, front_axis, front_sign, right = _fbx_scene_axes(scene, sdk)
    if not right:
        # the remedy names load_fbx; load(), which passes axis_system on to
        # it, reads the file that way too
        raise ValueError(
            f"{filename!r} is in a left handed axis system, which no rotation turns "
            "into Maya's Y-up (it takes a mirror); load it with "
            "load_fbx(filename, axis_system=None) to keep the file's own axes"
        )

    upward              = np.zeros(3)
    forward             = np.zeros(3)
    upward[up_axis]     = float(up_sign)
    forward[front_axis] = float(front_sign)

    turn = np.array([np.cross(upward, forward), upward, forward]).T
    return None if np.array_equal(turn, np.eye(3)) else turn


def _fbx_scene_axes(scene, sdk=None) -> tuple:
    """a scene's axis system, ``(up axis, up sign, front axis, front sign,
    right handed)``, the axes 0, 1 and 2 for x, y and z and the signs 1 or
    -1, as its ``FbxAxisSystem`` holds them. ``sdk`` is the fbx module the
    scene was read with, this module's by default."""
    sdk  = FBX if sdk is None else sdk
    axes = scene.GetGlobalSettings().GetAxisSystem()

    # the up vector is x, y or z (1, 2, 3), the front vector the first or the
    # second of the other two (parity even 1, odd 2), each with its sign
    up, up_sign = axes.GetUpVector()
    front, front_sign = axes.GetFrontVector()
    up_axis    = int(getattr(up, "value", up)) - 1
    others     = [axis for axis in range(3) if axis != up_axis]
    front_axis = others[int(getattr(front, "value", front)) - 1]
    left       = _fbx_enum(sdk.FbxAxisSystem, "ECoordSystem", "eLeftHanded")
    return (
        up_axis,
        1 if up_sign > 0 else -1,
        front_axis,
        1 if front_sign > 0 else -1,
        axes.GetCoorSystem() != left,
    )


# ------------------------- WHAT A RIG'S VALUES ARE IN ------------------------ #
#
# A rig and a clip record the axis system and the unit their values are in,
# axis_system and unit: the fbx loaders set them to what they produced, and
# save_fbx labels the file with them. Hand-built data, and a file saved
# before there were any, hold Maya's Y-up and cm.

# the axis_system of Maya's Y-up, the fbx loaders' default
_MAYA_Y_UP = "maya_y_up"

# any other axis system, as a record spells it: "z up, -y front, right handed"
_AXIS_WORDS = re.compile(r"(-?)([xyz]) up, (-?)([xyz]) front, (right|left) handed")


def _axis_words(up_axis: int, up_sign: int, front_axis: int, front_sign: int, right: bool) -> str:
    """the axis_system an axis system is recorded as (:func:`_fbx_scene_axes`'s
    tuple): "maya_y_up" for Maya's Y-up, any other its up and front axes and
    its handedness in words: "z up, -y front, right handed" for Unreal's"""
    if (up_axis, up_sign, front_axis, front_sign, right) == (1, 1, 2, 1, True):
        return _MAYA_Y_UP

    def axis(index, sign):
        return ("-" if sign < 0 else "") + "xyz"[index]

    hand = "right" if right else "left"
    return f"{axis(up_axis, up_sign)} up, {axis(front_axis, front_sign)} front, {hand} handed"


def _axis_record(value) -> str:
    """``value`` as a rig records an axis_system: "maya_y_up", or an axis
    system in words (:func:`_axis_words`), Maya's Y-up in words read as
    "maya_y_up"

    Raises TypeError for anything but a str, and ValueError for a str that
    names no axis system, up and front on one axis among them.
    """
    if not isinstance(value, str):
        raise TypeError(f"axis_system must be a str, got {type(value).__name__}")
    if value == _MAYA_Y_UP:
        return value

    match = _AXIS_WORDS.fullmatch(value)
    if match is None or match.group(2) == match.group(4):
        raise ValueError(
            "axis_system must be 'maya_y_up' or an axis system in words, its up and "
            "front two of x, y and z, either one negated, and its hand, such as "
            f"'z up, -y front, right handed'; got {value!r}"
        )
    return _axis_words(*_axis_record_axes(value))


def _axis_record_axes(record: str) -> tuple:
    """the ``(up axis, up sign, front axis, front sign, right handed)`` of a
    recorded axis_system, as :func:`_fbx_scene_axes` gives a scene's"""
    if record == _MAYA_Y_UP:
        return 1, 1, 2, 1, True
    up_sign, up, front_sign, front, hand = _AXIS_WORDS.fullmatch(record).groups()
    return (
        "xyz".index(up),
        -1 if up_sign else 1,
        "xyz".index(front),
        -1 if front_sign else 1,
        hand == "right",
    )


def _unit_record(value) -> float:
    """``value`` as a rig records a unit, the centimeters one unit of its
    lengths holds: a positive finite number

    Raises TypeError for anything but a real number, a bool among them, and
    ValueError for one that is not positive and finite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"unit must be a number of centimeters, got {type(value).__name__}")
    unit = float(value)
    if not (np.isfinite(unit) and unit > 0.0):
        raise ValueError(f"unit must be a positive number of centimeters, got {value!r}")
    return unit


def _fbx_check_scale_factor(scale_factor) -> None:
    """refuses a ``scale_factor`` that leaves the read's lengths in no unit:
    anything but None and a positive finite number"""
    if scale_factor is None:
        return
    try:
        factor = float(scale_factor)
    except (TypeError, ValueError):
        factor = float("nan")
    if not (np.isfinite(factor) and factor > 0.0):
        raise ValueError(
            "scale_factor must be None, which converts the file's unit to "
            f"centimeters, or a positive number, got {scale_factor!r}"
        )


def _fbx_record(scene, axis_system: Optional[str], scale_factor: Optional[float]) -> tuple:
    """``(axis_system, unit)`` a load of ``scene`` records: "maya_y_up" when
    it loads in Maya's Y-up, the file's own axes in words when it keeps
    them; 1.0, cm, when it converts the file's unit, and the file's unit in
    cm over the caller's factor when it is given one, the unit the lengths
    it multiplies by that factor are in"""
    axes = _MAYA_Y_UP if axis_system is not None else _axis_words(*_fbx_scene_axes(scene))
    if scale_factor is None:
        return axes, 1.0
    return axes, _unit_record(_fbx_scale_factor(scene, None) / float(scale_factor))


def _recorded(data, record: tuple):
    """``data``, the rig, clip or view a load made, recording ``record``,
    the ``(axis_system, unit)`` of :func:`_fbx_record`"""
    if isinstance(data, HierarchyData):
        data._axis_system, data._unit = record
    else:
        data._record = record
    return data


def _records_listed(records) -> str:
    """``(axis_system, unit)`` records as a refusal names them, each once"""
    return "; ".join(f"axis_system {axes!r}, unit {unit!r}" for axes, unit in sorted(set(records)))


def _refuse_records(what: str, records) -> None:
    """raises ValueError for values recorded in two axis systems or units
    meeting in one rig, clip or pose algebra: no one axis_system and unit
    says where they are. ``what`` names them, ``records`` holds their
    ``(axis_system, unit)``"""
    raise ValueError(
        f"{what} are in other axes or units ({_records_listed(records)}): no one "
        "axis_system and unit says where they are; give them one first"
    )


def _owner_records(nodes) -> list:
    """the ``(axis_system, unit)`` the rigs owning ``nodes`` record, each
    once, sorted; none for nodes no rig owns"""
    found = {}
    for node in nodes:
        owner = getattr(node, "_hierarchy", None)
        if owner is not None and id(owner) not in found:
            found[id(owner)] = (owner._axis_system, owner._unit)
    return sorted(set(found.values()))


def _records_of(nodes) -> tuple:
    """``(axis_system, unit)`` of nodes given as a rig, a clip, a view or a
    plain sequence: a TransformList's own (``_records``), or what the rigs
    owning a sequence's nodes record, Maya's Y-up and cm when no rig owns
    one. ValueError when they record two"""
    if isinstance(nodes, TransformList):
        return nodes._records()
    records = _owner_records(nodes)
    if len(records) > 1:
        _refuse_records("the nodes", records)
    return records[0] if records else (_MAYA_Y_UP, 1.0)


def _check_operands(data, other) -> None:
    """refuses the pose algebra between values in two axis systems or units,
    ``data``'s and ``other``'s: a delta taken between them, or applied in
    other axes or units than its own, mixes them"""
    records = (data._records(), other._records())
    if records[0] != records[1]:
        _refuse_records("the pose algebra's operands", records)


def _held_records(values) -> list:
    """the ``(axis_system, unit)`` records the nodes of a match_* or paste
    operand or receiver hold, each once, sorted: a frame axis's clip's, a
    rig's or clip's own and the rigs' of the nodes it lists without owning
    them (an item assignment lists such a node), a view's nodes'
    (``TransformList._held_records``), a node's rig's and a sequence's
    nodes' rigs'. A node no rig owns holds none, as it holds none joining a
    rig, but in a view :meth:`TransformList.load_fbx` gave, or one taken
    from it, which holds its load's"""
    if isinstance(values, _FrameAxis):
        values = values._clip
    if isinstance(values, TransformList):
        return values._held_records()
    if isinstance(values, TransformData):
        values = (values,)
    return _owner_records(values)


def _check_matched(data, other, what: str = "the nodes and the nodes they match") -> None:
    """refuses matching or pasting into ``data`` the values of nodes that
    hold another record than ``data``'s: the positions, rotations or
    channels taken from them would land in ``data`` under a label that does
    not describe them. Values, and nodes that hold no record, pass"""
    # a node matching a node, which a view's match_* does node by node, read
    # off their rigs without the sets
    if isinstance(other, TransformData) and isinstance(data, TransformData):
        theirs, mine = other._hierarchy, data._hierarchy
        if theirs is None or mine is None or theirs is mine:
            return
        records = (mine._records(), theirs._records())
        if records[0] != records[1]:
            _refuse_records(what, records)
        return

    if not _holds_nodes(other):
        return
    found = _held_records(other)
    if not found:
        return
    found = set(found) | set(_held_records(data))
    if len(found) > 1:
        _refuse_records(what, found)


def _fbx_turn_roots(turn, joints, orders, translate, rotate, joint_orient) -> tuple:
    """(F, r, 3) channels of a scene's roots with ``turn`` put on them, every
    frame at once: each root's world matrix becomes its world times ``turn``

    ``joints`` (r,) says which roots are joints and ``orders`` (r,) gives each
    one's rotate order. A joint takes the turn in its joint orient, JO turn,
    and keeps its rotate, so its curves are not solved again; any other root,
    which the walk reads without a joint orient, has its rotate solved again
    for R turn in its own rotate order, through euler_filter along the
    frames, frame 0 as solved. Every root's translate turns with it. Hands
    back new (translate, rotate, joint_orient).
    """
    joints       = np.asarray(joints, dtype=bool)
    orders       = np.asarray(orders, dtype=np.intp)
    frames       = len(translate)
    translate    = np.asarray(translate, dtype=float) @ turn
    rotate       = np.array(rotate, dtype=float)
    joint_orient = np.array(joint_orient, dtype=float)

    # joint orient is xyz, a rotate in the node's own order
    for chosen, block, axes in (
        (joints, joint_orient, np.zeros(len(orders), dtype=np.intp)),
        (~joints, rotate, orders),
    ):
        if not chosen.any():
            continue

        order               = np.tile(axes[chosen], frames)
        matrices            = euler_to_matrix(np.radians(block[:, chosen].reshape(-1, 3)), order)
        matrices[:, :3, :3] = matrices[:, :3, :3] @ turn
        solved              = np.degrees(matrix_to_euler(matrices, order)).reshape(frames, -1, 3)
        block[:, chosen]    = _filtered(solved, axes[chosen])

    return translate, rotate, joint_orient


def _fbx_turn_walk(hierarchy_data: dict, turn) -> None:
    """puts ``turn`` on the roots of the dicts :func:`_walk_fbx` read, in
    place: :func:`_fbx_turn_roots` on their static values"""
    roots = [data for data in hierarchy_data.values() if "parent_uuid" not in data]
    if not roots:
        return

    joints = [data["node_type"] == "joint" for data in roots]
    orders = [data["rotate_order"] for data in roots]
    channels = [
        np.array([[data.get(channel, np.zeros(3)) for data in roots]], dtype=float)
        for channel in ("translate", "rotate", "joint_orient")
    ]
    translate, rotate, joint_orient = _fbx_turn_roots(turn, joints, orders, *channels)

    for i, data in enumerate(roots):
        data["translate"] = translate[0, i]
        data["rotate"]    = rotate[0, i]
        if joints[i]:
            data["joint_orient"] = joint_orient[0, i]


def _fbx_check_pose(pose) -> None:
    """refuses a ``pose`` HierarchyData.load_fbx doesn't know"""
    if not (isinstance(pose, str) and pose in ("node", "bind")):
        raise ValueError(f"pose must be 'node' or 'bind', got {pose!r}")


# two bind poses that list one node agree on it when their matrices are this
# close: one export writes both from one scene state
_FBX_BIND_AGREE = 1e-6


def _fbx_bind_matrices(scene, tree: _FbxTree, units: _FbxUnits, filename) -> dict:
    """``{position: (matrix, local)}`` of each node of ``tree`` (the nodes
    :func:`_walk_fbx` kept and every node above one) that a bind pose
    lists: the pose's (4, 4) matrix, row vectors, in cm (``units``,
    :func:`_fbx_units`), and whether the pose holds it local to the parent
    rather than as the world

    A matrix's translation is multiplied by the unit, and on a node inside
    a unit root its 3x3 too: the export scales every matrix of a pose by
    1/s, the world of a node it scaled by 1/s with it.

    A bind pose is an ``FbxPose`` whose ``IsBindPose()`` is true. When two
    list one node the first wins, and a later one more than
    :data:`_FBX_BIND_AGREE` off it, in the loaded unit, warns, naming the
    node. Raises ValueError when no bind pose lists a node of ``tree``: a
    pose that lists a node above the kept ones, and none of them, puts them
    at its bind (:func:`_fbx_kept_binds`).
    """
    positions = {node.GetUniqueID(): position for position, node in enumerate(tree.nodes)}
    inner     = np.array([uid in units.inner for uid in positions], dtype=bool)
    found     = {}  # position: (matrix, local, the pose's name)
    for index in range(scene.GetPoseCount()):
        pose = typed(scene.GetPose(index), FBX.FbxPose, FBX)
        if not pose.IsBindPose():
            continue

        for k in range(pose.GetCount()):
            node     = typed(pose.GetNode(k), FBX.FbxNode, FBX)
            position = None if node is None else positions.get(node.GetUniqueID())
            if position is None:
                continue

            source = pose.GetMatrix(k)
            matrix = np.array([[source.Get(r, c) for c in range(4)] for r in range(4)])
            local  = bool(pose.IsLocalMatrix(k))
            if inner[position]:
                matrix[:, :3] *= units.factor
            else:
                matrix[3, :3] *= units.factor
            if position not in found:
                found[position] = (matrix, local, pose.GetName())
                continue

            # a local matrix and a world one can't be held side by side
            first, held, name = found[position]
            gap = float(np.abs(matrix - first).max())
            if local == held and gap > _FBX_BIND_AGREE:
                warnings.warn(
                    f"{filename!r}: bind poses {name!r} and {pose.GetName()!r} disagree on "
                    f"{node.GetName()!r} by {gap:.3g}; the first, {name!r}, is read"
                )

    if not found:
        raise ValueError(
            f"{filename!r} holds no bind pose that lists a node the read keeps or one above it; "
            "pose='node' reads the nodes' own values"
        )
    return {position: (matrix, local) for position, (matrix, local, _) in found.items()}


def _fbx_kept_binds(tree: _FbxTree, listed: dict, uuids: list) -> dict:
    """``{uuid: (matrix, local)}`` of the kept nodes for
    :func:`_fbx_pose_binds`, from ``listed``, what :func:`_fbx_bind_matrices`
    read, ``uuids`` the kept nodes' uuids by column

    A kept node a bind pose lists takes its bind, as before. A dropped node
    one lists is folded in at its bind, the way it would be read if it were
    kept: a kept node no pose lists that sits under it, with no kept node a
    pose lists between them, takes the world the whole tree gives it, every
    node of it kept, each node a pose lists at its bind and every other at
    its own values under its parent (a joint compensating its parent's
    scale as the bind leaves that scale). The other kept nodes are left to
    that same rule as :func:`_fbx_pose_binds` applies it. The matrices are
    in the file's axes.
    """
    kept = tree.kept
    out  = {uuids[kept[p]]: entry for p, entry in listed.items() if kept[p] >= 0}

    # the kept nodes a listed dropped node moves: under one, with no kept
    # node a pose lists between them
    over = np.zeros(len(kept), dtype=bool)
    for p, up in enumerate(tree.parent.tolist()):
        if up >= 0:
            over[p] = (up in listed and kept[up] < 0) or (bool(over[up]) and up not in listed)
    moved = [p for p in np.flatnonzero(over & (kept >= 0)).tolist() if p not in listed]
    if not moved:
        return out

    # the whole tree, every node of it kept, so the fold folds into each one
    # only what its own channels can't hold
    whole = _FbxTree()
    for name in _FbxTree.__slots__:
        setattr(whole, name, getattr(tree, name))
    whole.kept    = np.arange(len(kept), dtype=np.intp)
    whole.columns = whole.kept
    whole.above   = tree.parent
    translate, rotate, scale, joint_orient, _ = _fbx_fold(
        whole, tree.translate[None], tree.rotate[None], tree.scale[None]
    )

    ids  = [generate_uuid() for _ in range(len(kept))]
    data = {}
    for p, up in enumerate(tree.parent.tolist()):
        node = {
            "name":         f"node{p}",
            "uuid":         ids[p],
            "rotate_order": int(tree.orders[p]),
            "translate":    translate[0, p],
            "rotate":       rotate[0, p],
            "scale":        scale[0, p],
            "node_type":    "joint" if tree.joint[p] else "transform",
        }
        if up >= 0:
            node["parent_uuid"] = ids[up]
        if tree.joint[p]:
            node["joint_orient"]             = joint_orient[0, p]
            node["rotate_axis"]              = tree.axis[p].copy()
            node["segment_scale_compensate"] = bool(tree.compensate[p])
        data[ids[p]] = node

    # only the binds above a moved node say where it goes
    needed = set()
    for p in moved:
        while p >= 0 and p not in needed:
            needed.add(p)
            p = int(tree.parent[p])
    rig = HierarchyData.from_dict(data)
    _fbx_pose_binds(rig, {ids[p]: entry for p, entry in listed.items() if p in needed}, None)

    for p in moved:
        out[uuids[kept[p]]] = (np.asarray(rig[f"node{p}"].world_matrix), False)
    return out


def _fbx_pose_binds(rig, binds: dict, turn) -> None:
    """puts each node of ``rig`` that ``binds`` lists (:func:`_fbx_bind_matrices`)
    at its bind matrix, in place

    The matrix is the node's world in the file, times ``turn``, as the read
    turns every world; one the pose holds local to the parent is composed
    with the parent's world, as the read gives it, or with ``turn`` on a
    root. The world matrices are written parents first, so a node no bind
    pose lists keeps its own values under its parent's bind. Each write
    solves scale, rotate (in the node's rotate order) and translate, the
    rotate axis and joint orient kept; a node the matrix leaves where it
    is keeps its own values.
    """
    K = np.eye(4)
    if turn is not None:
        K[:3, :3] = turn

    for index in rig._parent_first():
        node  = rig.list[index]
        entry = binds.get(node.uuid)
        if entry is None:
            continue

        matrix, local = entry
        parent            = node.get_parent() if local else None
        node.world_matrix = matrix @ (K if parent is None else parent.world_matrix)


def _fbx_frame_rate(mode) -> float:
    """frames per second for a time mode, or 0.0 if the sdk cannot say"""
    rate = float(FBX.FbxTime.GetFrameRate(mode))
    if rate > 0.0:
        return rate

    # the sdk reports eFrames30Drop as 0.0 even though it names the same rate
    # as eNTSCDropFrame, which it reports correctly
    if mode == _fbx_enum(FBX.FbxTime, "EMode", "eFrames30Drop"):
        return float(
            FBX.FbxTime.GetFrameRate(_fbx_enum(FBX.FbxTime, "EMode", "eNTSCDropFrame"))
        )

    return 0.0


# curves x frames entries one batch of fbx curves is evaluated over: a few
# arrays that size are alive at once, so a long take goes a batch at a time
_FBX_SAMPLE_BUDGET = 2**22


def _fbx_ticks_per_second() -> int:
    """the sdk's time unit: ``FbxTime`` counts in ticks, and how many make a
    second depends on the sdk version"""
    second = FBX.FbxTime()
    second.SetSecondDouble(1.0)
    return int(second.Get())


# a time this close to a whole frame, in frames, is that frame: the float
# noise in a take's span or in a key's time adds no frame to a clip
_FBX_FRAME_SNAP = 1e-6


def _fbx_whole_frames(ticks, fps: float, per_second: int) -> np.ndarray:
    """fbx times, in ticks, as frames at ``fps``, each within
    :data:`_FBX_FRAME_SNAP` of a whole frame put on it"""
    frames  = np.asarray(ticks, dtype=np.float64) * float(fps) / float(per_second)
    nearest = np.round(frames)
    return np.where(np.abs(frames - nearest) <= _FBX_FRAME_SNAP, nearest, frames)


def _fbx_key_ticks(stack) -> Optional[tuple]:
    """the first and the last key time, in ticks, over every curve on every
    layer of a take, None for a take that holds no key

    A curve keeps its keys in time order, so its first and last keys are its
    ends.
    """
    layers      = FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId)
    curve_nodes = FBX.FbxCriteria.ObjectType(FBX.FbxAnimCurveNode.ClassId)

    first, last = None, None
    for i in range(stack.GetSrcObjectCount(layers)):
        layer = typed(stack.GetSrcObject(layers, i), FBX.FbxAnimLayer, FBX)
        for j in range(layer.GetSrcObjectCount(curve_nodes)):
            curve_node = typed(layer.GetSrcObject(curve_nodes, j), FBX.FbxAnimCurveNode, FBX)
            for channel in range(curve_node.GetChannelsCount()):
                for k in range(curve_node.GetCurveCount(channel)):
                    curve = typed(curve_node.GetCurve(channel, k), FBX.FbxAnimCurve, FBX)
                    count = curve.KeyGetCount() if curve is not None else 0
                    if not count:
                        continue
                    lo    = int(curve.KeyGetTime(0).Get())
                    hi    = int(curve.KeyGetTime(count - 1).Get())
                    first = lo if first is None else min(first, lo)
                    last  = hi if last is None else max(last, hi)

    return None if first is None else (first, last)


def _fbx_ticks(seconds, per_second: int) -> np.ndarray:
    """times in seconds as fbx ticks, rounded the way ``FbxTime.SetSecondDouble``
    rounds them

    That is toward zero once the product is pushed 0.005 of a tick away from
    zero, which no plain rounding reproduces. It is what lands a frame time a
    hair short of a whole tick on that tick, so keys baked on frames are hit
    exactly. Checked against the sdk on 75,000 times, negative ones included.
    """
    scaled = np.asarray(seconds, dtype=np.float64) * float(per_second)
    return np.trunc(scaled + np.copysign(0.005, scaled)).astype(np.int64)


# the extrapolation count at which keep slope carries the end key's slope
# on rather than holding the key, as any other count does: the one Maya
# writes for every infinity it exports
_FBX_KEEP_SLOPE_ON = 2**32 - 1


def _fbx_agrees_past(curve, tick: int, expected: float) -> bool:
    """whether the sdk's own evaluation of ``curve`` at ``tick``, past its
    keys, is ``expected``: the check on the keep slope model, which another
    sdk need not follow"""
    time = FBX.FbxTime()
    time.Set(int(tick))
    got = curve.Evaluate(time)
    got = got[0] if isinstance(got, tuple) else got
    return float(got) == float(expected)


def _fbx_extrapolated(ticks, first: int, last: int, before, after) -> tuple:
    """the tick inside a curve's keys, ``first`` to ``last``, that the sdk
    reads for each of ``ticks``, the tick itself between them, the number of
    periods each one's value is shifted by, and whether the sdk reads it as 0

    ``before`` and ``after`` are ``(kind, count)``, the curve's pre and post
    extrapolation, each ``"repeat"``, ``"mirror"``, ``"offset"``, ``"zero"``
    or None for the end key's value. A repetition reads the keys again,
    period after period, a mirror reads them forward and backward in turn;
    past ``count`` periods the end key's value holds. Worked out in whole
    ticks, as the sdk reads them: a tick exactly a whole number of periods
    past the last key starts the next period, one as far before the first
    key ends the previous one.

    ``"offset"`` is a repetition that climbs (the sdk's relative repetition,
    Maya's cycle with offset): a tick ``n`` periods away reads the keys as a
    repetition does, its value shifted by ``n`` times the last key's value
    less the first's, ``-n`` before the keys; past ``count`` periods the end
    key holds, shifted ``count`` times. The shifts come back as the second
    array, 0 for every tick no offset moves.

    ``"zero"`` is how the sdk reads an extrapolation number it does not
    know: 0 for ``count`` periods, then the end key. The third array flags
    the ticks it reads as 0.
    """
    ticks   = np.asarray(ticks, dtype=np.int64)
    out     = np.clip(ticks, first, last)
    periods = np.zeros(len(ticks), dtype=np.int64)
    zeroed  = np.zeros(len(ticks), dtype=bool)
    span    = last - first
    if span <= 0:
        return out, periods, zeroed

    for side, (kind, count) in (("before", before), ("after", after)):
        if kind is None or count < 1:
            continue
        rows = np.flatnonzero(ticks < first if side == "before" else ticks > last)
        if not len(rows):
            continue

        reach  = first - ticks[rows] if side == "before" else ticks[rows] - last
        period = reach // span + 1            # 1 for the period next to the keys
        offset = reach - (period - 1) * span  # 0 <= offset < span
        if side == "before":
            # read back from the last key, or, mirrored, forward from the first
            ahead = first + offset
            back  = last - offset
            held  = first
        else:
            ahead = last - offset
            back  = first + offset
            held  = last
        if kind == "mirror":
            u = np.where(period % 2 == 1, ahead, back)
        else:
            u = back
        out[rows] = np.where(period <= count, u, held)
        if kind == "offset":
            periods[rows] = np.minimum(period, count) * (-1 if side == "before" else 1)
        if kind == "zero":
            out[rows]    = held
            zeroed[rows] = period <= count
    return out, periods, zeroed


def _fbx_sample_curves(curves, ticks, per_second: int) -> np.ndarray:
    """fbx animation curves evaluated on a grid of ticks, every frame at once,
    as (curves, frames) values

    :func:`_fbx_sample_span` evaluates the grid; a curve that repeats or
    mirrors past its keys is evaluated again where the extrapolation sends
    each tick (:func:`_fbx_extrapolated`), every curve sent the same way in
    one more batch. A side of it that does not repeat keeps what the first
    pass gave it: the end key, or a keep slope carried on.
    """
    ticks = np.asarray(ticks, dtype=np.int64)
    values, extrapolated = _fbx_sample_span(curves, ticks, per_second)

    groups = {}
    for row, (spec, ends) in extrapolated.items():
        groups.setdefault(spec, []).append((row, ends))

    for (first, last, before, after), members in groups.items():
        rows = [row for row, _ in members]
        read, periods, zeroed = _fbx_extrapolated(ticks, first, last, before, after)
        grid, back = np.unique(read, return_inverse=True)
        again   = _fbx_sample_curves([curves[r] for r in rows], grid, per_second)[:, back]
        shifted = np.flatnonzero(periods)
        if len(shifted):
            # an offset repetition climbs by the last key less the first a
            # period, added in single precision as the sdk adds it
            ends              = np.array([pair for _, pair in members], dtype=np.float32)
            rise              = (ends[:, 1] - ends[:, 0])[:, None]
            climbed           = again[:, shifted].astype(np.float32) + periods[shifted].astype(np.float32) * rise
            again[:, shifted] = climbed
        again[:, zeroed] = 0.0
        kept             = ((ticks < first) & (before[0] is None)) | ((ticks > last) & (after[0] is None))
        values[rows]     = np.where(kept, values[rows], again)
    return values


def _fbx_sample_span(curves, ticks, per_second: int):
    """fbx animation curves evaluated on a grid of ticks, every frame at once

    Each curve's key times and values are read once; derivatives, weights and
    interpolation only for the keys that start a span some grid tick falls
    strictly inside, which is none at all for a take baked on its frames.
    The sdk has no batch evaluator, and reading a key costs a fraction of
    the few microseconds one ``EvaluateLocal*`` call costs.

    Hands back ``values`` (curves, frames) and ``extrapolated``: for each
    curve that repeats, mirrors or climbs past its keys where some tick
    falls, its row, ``(first key, last key, (kind, count) before, (kind,
    count) after)`` and its first and last key values, for
    :func:`_fbx_sample_curves` to evaluate again.

    The sdk's own evaluation, as measured against ``EvaluateLocalTranslation``:

    - on a key's tick, that key's value, whatever its interpolation;
    - before the first key and after the last, the end key's value, unless
      the curve repeats, mirrors or repeats with an offset there, for as
      many periods as its count says (:func:`_fbx_extrapolated`). An
      extrapolation number the sdk does not know reads as 0 for as many
      periods. Keep slope holds the end key's value too, unless its count is
      2**32 - 1 (:data:`_FBX_KEEP_SLOPE_ON`, the count Maya writes): then
      the end key's value carries on along its slope, the first key's left
      derivative before the keys and the last key's right one after them,
      the product rounded to single precision and added in it, as the sdk
      adds it. The curve's own ``Evaluate`` at the frame farthest past its
      keys checks that, once a curve and side, and an sdk that evaluates it
      otherwise is refused with a ValueError rather than sampled frame by
      frame;
    - between two keys, by the first key's interpolation. Constant holds its
      value, or the next key's under ``eConstantNext``. Linear blends the
      two. Cubic is a bezier whose handles sit ``weight * span`` along the
      key's right derivative and the next key's left one. ``KeyGet*Derivative``
      already resolves auto, tcb, clamped and broken tangents; tangent
      velocity changes nothing. An unweighted handle sits at a third, which
      makes the bezier a hermite; a weighted one needs its time solved, done
      here by bisection. An interpolation the sdk does not know (a file can
      carry one, read back as 0) gives 0, as the sdk gives it.

    The sdk works in single precision between keys, so values there agree to
    its rounding rather than to the bit, a few times 1e-6 of the curve's
    range at most, but on a weighted span whose two handles crowd one end:
    one weight at most about 0.01 and the other about 0.88 or more, the
    smaller the one, the lower the other can be (the sdk holds a weight to
    0.0001 at least and 0.99 at most). There the sdk's own solve for the
    bezier's time stops short of the exact time solved here, by up to 2e-5
    of the curve's range at weights 0.01 and 0.99, 3e-5 at 0.0001 and 0.88,
    4e-4 at 0.0001 and 0.89, 3e-3 at 0.0001 and 0.9 and 0.55 at 0.0001 and
    0.99. Maya's own import of such a span gives the values solved here, to
    a few times 1e-15 of the range, and they are kept: on a crowded span the
    loader follows Maya, not the sdk. On keys they are the same values.
    """
    constant, linear, cubic = 0, 1, 2
    interpolation = [
        (_fbx_enum(FBX.FbxAnimCurveDef, "EInterpolationType", name), code)
        for name, code in (
            ("eInterpolationConstant", constant),
            ("eInterpolationLinear", linear),
            ("eInterpolationCubic", cubic),
        )
    ]
    # how a curve goes on past its keys, by the sdk's number for it, which a
    # file carries: these bindings leave relative repetition (5) unnamed, and
    # a file can carry a number the sdk does not know
    def number(member) -> int:
        return int(getattr(member, "value", member))

    extrapolation = "EExtrapolationType"
    flat_past     = number(_fbx_enum(FBX.FbxAnimCurve, extrapolation, "eConstant"))
    slope_past    = number(_fbx_enum(FBX.FbxAnimCurve, extrapolation, "eKeepSlope"))
    repetition    = number(_fbx_enum(FBX.FbxAnimCurve, extrapolation, "eRepetition"))
    mirror        = number(_fbx_enum(FBX.FbxAnimCurve, extrapolation, "eMirrorRepetition"))
    repeats       = {repetition: "repeat", mirror: "mirror", 5: "offset"}
    next_key      = _fbx_enum(FBX.FbxAnimCurveDef, "EConstantMode", "eConstantNext")

    ticks  = np.asarray(ticks, dtype=np.int64)
    counts = np.array([curve.KeyGetCount() for curve in curves], dtype=np.int64)
    starts = np.cumsum(counts) - counts
    owner  = np.repeat(np.arange(len(curves)), counts)

    # one sdk call per key and field, looped in C by map() and np.fromiter
    tick_of = FBX.FbxTime.Get
    pairs   = list(zip(curves, counts.tolist()))
    when = np.concatenate(
        [
            np.fromiter(map(tick_of, map(curve.KeyGetTime, range(count))), np.int64, count)
            for curve, count in pairs
        ]
    )
    value = np.concatenate(
        [np.fromiter(map(curve.KeyGetValue, range(count)), np.float64, count) for curve, count in pairs]
    )

    # the last key at or before each grid tick, for every curve at once. Each
    # key goes in the cell of the first grid tick at or after it, the last
    # key of a cell wins, and a running maximum along each curve's row hands
    # every later tick the latest key behind it; -1 is before the first key
    width             = len(ticks) + 1
    cell              = owner * width + np.searchsorted(ticks, when, side="left")
    wins              = np.append(cell[1:] != cell[:-1], True)
    index             = np.full(len(curves) * width, -1, dtype=np.int64)
    index[cell[wins]] = np.flatnonzero(wins)
    index             = np.maximum.accumulate(index.reshape(len(curves), width), axis=1)[:, :-1]

    first   = starts[:, None]
    last    = (starts + counts - 1)[:, None]
    before  = index < 0
    index   = np.maximum(index, first)
    on_key  = when[index] == ticks
    after   = (index == last) & ~on_key & ~before
    between = ~(before | on_key | after)

    # on a key it is that key, and either side of the keys the end key holds
    values = value[index]

    # past the keys: a curve that repeats, mirrors or climbs there is read
    # again where that sends each tick, and one extrapolated by a number the
    # sdk does not know reads 0 there; constant holds the end key's value,
    # and so does keep slope, or carries its slope on, once the curve's own
    # evaluation agrees
    extrapolated = {}
    edges        = np.flatnonzero(before.any(axis=1) | after.any(axis=1))
    for row in edges.tolist():
        curve = curves[row]
        ends  = (starts[row], starts[row] + counts[row] - 1)
        sides = []
        for past, kind, count, end, far in (
            (before[row], curve.GetPreExtrapolation(), curve.GetPreExtrapolationCount(), ends[0], 0),
            (after[row], curve.GetPostExtrapolation(), curve.GetPostExtrapolationCount(), ends[1], -1),
        ):
            sides.append((None, 0))
            kind = number(kind)
            if not past.any() or kind == flat_past:
                continue
            if kind == slope_past:
                beyond = np.flatnonzero(past)
                if count == _FBX_KEEP_SLOPE_ON:
                    # the end key's slope, in single precision as the sdk
                    # carries it on
                    if far == 0:
                        slope = curve.KeyGetLeftDerivative(0)
                    else:
                        slope = curve.KeyGetRightDerivative(int(counts[row]) - 1)
                    seconds             = (ticks[beyond] - when[end]) / per_second
                    values[row, beyond] = np.float32(value[end]) + (slope * seconds).astype(np.float32)

                # the frame farthest from the keys, where a slope would show most
                farthest = beyond[far]
                if not _fbx_agrees_past(curve, ticks[farthest], values[row, farthest]):
                    raise ValueError(
                        "this fbx sdk evaluates a keep slope curve past its keys "
                        "otherwise than load_fbx does: the sdk it was measured "
                        "against holds the end key, or carries its slope on "
                        "when the count is 2**32 - 1"
                    )
                continue
            sides[-1] = (repeats.get(kind, "zero"), count) if count > 0 else (None, 0)

        bounds = (int(when[ends[0]]), int(when[ends[1]]))
        if bounds[1] > bounds[0] and (sides[0][0] or sides[1][0]):
            extrapolated[row] = (bounds + tuple(sides), (value[ends[0]], value[ends[1]]))

    # the grid ticks strictly inside a span, as flat indexes into values
    inside = np.flatnonzero(between)
    if not len(inside):
        return values, extrapolated

    # each span's first key, read once however many grid ticks fall inside it
    at          = index.ravel()[inside]
    needed      = np.zeros(len(when), dtype=bool)
    needed[at]  = True
    key         = np.flatnonzero(needed)
    lookup      = np.zeros(len(when), dtype=np.intp)
    lookup[key] = np.arange(len(key))
    slot        = lookup[at]
    curve_of    = owner[key]
    local       = key - starts[curve_of]

    def read(getter, spans, step=0):
        """one sdk getter over the first key of ``spans`` (or the key ``step``
        after it), a C loop per curve: the spans are sorted, so each curve's
        are one run"""
        got   = []
        whose = curve_of[spans]
        keys  = (local[spans] + step).tolist()
        cuts  = (np.flatnonzero(np.diff(whose)) + 1).tolist()
        for lo, hi in zip([0, *cuts], [*cuts, len(keys)]):
            got.extend(map(getattr(curves[whose[lo]], getter), keys[lo:hi]))
        return got

    # enum members, compared by identity rather than hashed one by one
    kinds = np.array(read("KeyGetInterpolation", np.arange(len(key))), dtype=object)
    # an interpolation the sdk does not know stays -1, and gives 0 as there
    kind = np.full(len(key), -1, dtype=np.int8)
    for member, code in interpolation:
        kind[kinds == member] = code

    # each span as a cubic in its own parameter u, 0 to 1 across the span:
    # value = ((c3 u + c2) u + c1) u + c0, which holds constant and linear too
    v0           = value[key]
    v1           = value[key + 1]
    coefficients = np.zeros((len(key), 4))

    held = np.flatnonzero(kind == constant)
    if len(held):
        to_next               = np.array([mode == next_key for mode in read("KeyGetConstantMode", held)])
        coefficients[held, 0] = np.where(to_next, v1[held], v0[held])

    blend                  = np.flatnonzero(kind == linear)
    coefficients[blend, 0] = v0[blend]
    coefficients[blend, 1] = v1[blend] - v0[blend]

    # a cubic span is a bezier, its handles weight * span along the key's
    # right derivative and the next key's left one. Unweighted, the weights
    # are a third and the bezier is this hermite
    bent    = np.flatnonzero(kind == cubic)
    seconds = (when[key + 1] - when[key]) / per_second
    out_arm = np.zeros(len(key))
    in_arm  = np.zeros(len(key))
    w_right = np.full(len(key), 1.0 / 3.0)
    w_left  = np.full(len(key), 1.0 / 3.0)
    if len(bent):
        out_arm[bent] = np.array(read("KeyGetRightDerivative", bent)) * seconds[bent]
        in_arm[bent]  = np.array(read("KeyGetLeftDerivative", bent, 1)) * seconds[bent]
        w_right[bent] = read("KeyGetRightTangentWeight", bent)
        w_left[bent]  = read("KeyGetLeftTangentWeight", bent, 1)

        a, b = v0[bent], v1[bent]
        coefficients[bent, 0] = a
        coefficients[bent, 1] = out_arm[bent]
        coefficients[bent, 2] = 3.0 * (b - a) - 2.0 * out_arm[bent] - in_arm[bent]
        coefficients[bent, 3] = 2.0 * (a - b) + out_arm[bent] + in_arm[bent]

    first_tick = when[key][slot]
    span_ticks = (when[key + 1] - when[key])[slot]
    u          = (ticks[inside % len(ticks)] - first_tick) / span_ticks
    c          = coefficients[slot]
    result     = ((c[:, 3] * u + c[:, 2]) * u + c[:, 1]) * u + c[:, 0]

    weighted = (kind == cubic) & (
        (np.abs(w_right - 1.0 / 3.0) > 1e-6) | (np.abs(w_left - 1.0 / 3.0) > 1e-6)
    )
    points = np.flatnonzero(weighted[slot])
    if len(points):
        # the bezier's time runs 0, wr, 1 - wl, 1: solve it for each u. Both
        # weights lie in 0..1, which keeps it rising
        spans = slot[points]
        wr, wl = w_right[spans], w_left[spans]
        x  = u[points]
        lo = np.zeros(len(points))
        hi = np.ones(len(points))
        for _ in range(60):
            mid   = 0.5 * (lo + hi)
            rest  = 1.0 - mid
            below = 3.0 * rest * mid * (rest * wr + mid * (1.0 - wl)) + mid**3 < x
            lo    = np.where(below, mid, lo)
            hi    = np.where(below, hi, mid)

        t    = 0.5 * (lo + hi)
        rest = 1.0 - t
        a, b = v0[spans], v1[spans]
        p1 = a + wr * out_arm[spans]
        p2 = b - wl * in_arm[spans]

        result[points] = rest**3 * a + 3.0 * rest * t * (rest * p1 + t * p2) + t**3 * b

    values.ravel()[inside] = result
    return values, extrapolated


def _fbx_euler_axes(order: int) -> tuple:
    """the axes ``(i, j, k)`` and the parity the sdk's euler code (Shoemake's,
    from Graphics Gems IV) gives an fbx rotate order. The sdk hands it the x,
    y and z angles as they are, whatever the order, so a rotation blended by
    layer comes out as an xyz node's blend for an even order and as a zyx
    node's for an odd one, whose additive layer turns on the other side of
    what is below (M(below) @ M(layer), both matrices of zyx angles, where
    Maya gives M(layer) @ M(below) in the node's own order); only the
    rounding, and so which way a tie breaks, follows the order"""
    first = (0, 0, 1, 1, 2, 2)[order]
    odd   = order % 2
    after = (1, 2, 0, 1)
    return first, after[first + odd], after[first + 1 - odd], odd


def _fbx_turns(euler: np.ndarray, order: int) -> np.ndarray:
    """(..., 4) quaternions, x y z w, of (..., 3) euler angles in degrees, as
    the sdk builds them to blend rotations a layer at a time: the matrix of
    the angles in the order's axes (:func:`_fbx_euler_axes`), then its
    quaternion, solved for w when the trace is positive and otherwise for
    the axis of the largest diagonal entry, the first of a tie; the
    component solved for comes out positive. That sign is what sends an
    additive layer's turn the long way round or not, so it is worked out
    on the same terms the sdk works it out on"""
    i, j, k, odd = _fbx_euler_axes(order)
    angles = np.radians(np.asarray(euler, dtype=np.float64))
    if odd:
        angles = -angles
    cos, sin = np.cos(angles), np.sin(angles)
    ci, cj, ch = cos[..., 0], cos[..., 1], cos[..., 2]
    si, sj, sh = sin[..., 0], sin[..., 1], sin[..., 2]
    cc, cs = ci * ch, ci * sh
    sc, ss = si * ch, si * sh

    m            = np.empty(angles.shape[:-1] + (3, 3))
    m[..., i, i] = cj * ch
    m[..., i, j] = sj * sc - cs
    m[..., i, k] = sj * cc + ss
    m[..., j, i] = cj * sh
    m[..., j, j] = sj * ss + cc
    m[..., j, k] = sj * cs - sc
    m[..., k, i] = -sj
    m[..., k, j] = cj * si
    m[..., k, k] = cj * ci

    diagonal = np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], axis=-1)
    trace    = diagonal[..., 0] + diagonal[..., 1] + diagonal[..., 2]
    largest  = np.where(diagonal[..., 1] > diagonal[..., 0], 1, 0)
    reached  = np.take_along_axis(diagonal, largest[..., None], axis=-1)[..., 0]
    largest  = np.where(diagonal[..., 2] > reached, 2, largest)

    # every branch is worked out everywhere, and each frame keeps its own:
    # a branch another frame takes can take a root of a negative number
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(trace + 1.0)
        half = 0.5 / root
        q = np.stack(
            [
                (m[..., 2, 1] - m[..., 1, 2]) * half,
                (m[..., 0, 2] - m[..., 2, 0]) * half,
                (m[..., 1, 0] - m[..., 0, 1]) * half,
                root * 0.5,
            ],
            axis=-1,
        )
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            root = np.sqrt((m[..., a, a] - (m[..., b, b] + m[..., c, c])) + 1.0)
            half = 0.5 / root
            turn = np.empty_like(q)

            turn[..., a] = root * 0.5
            turn[..., b] = (m[..., a, b] + m[..., b, a]) * half
            turn[..., c] = (m[..., c, a] + m[..., a, c]) * half
            turn[..., 3] = (m[..., c, b] - m[..., b, c]) * half
            q            = np.where(((trace <= 0.0) & (largest == a))[..., None], turn, q)
    return q


def _fbx_turn_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """the quaternion product a b, x y z w: b turns first, then a"""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _fbx_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray, shortest: bool) -> np.ndarray:
    """the sdk's spherical blend from ``a`` to ``b`` (x y z w) by ``t``, which
    need not lie in 0..1; ``shortest`` turns ``b`` round when it is the long
    way. The dot is summed as the sdk sums it, w first: when the two lie a
    half turn apart its rounding is what picks the way round"""
    dot = ((a[..., 3] * b[..., 3] + a[..., 0] * b[..., 0]) + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]
    if shortest:
        b   = np.where((dot < 0.0)[..., None], -b, b)
        dot = np.abs(dot)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sine  = np.sin(angle)
    t     = np.broadcast_to(t, dot.shape)
    flat  = sine < 1e-12
    safe  = np.where(flat, 1.0,     sine)
    out   = np.where(flat, 1.0 - t, np.sin((1.0 - t) * angle) / safe)
    into  = np.where(flat, t,       np.sin(t * angle) / safe)
    return out[..., None] * a + into[..., None] * b


# how far from 0 the cosine of the middle angle has to be for the sdk to
# split the turn between the first and the last angle: 16 double epsilons
_FBX_GIMBAL = 16.0 * np.finfo(np.float64).eps


def _fbx_turns_to_euler(q: np.ndarray, order: int) -> np.ndarray:
    """(..., 3) euler angles in degrees of quaternions, x y z w, as the sdk
    reads a blended rotation back: from its matrix, in the order's axes
    (:func:`_fbx_euler_axes`), the middle angle within -90..90. At gimbal,
    where that angle's cosine is within :data:`_FBX_GIMBAL` of 0, the last
    angle is 0 and the first takes the whole turn, as the sdk splits it:
    measured on the sdk, a cosine of 3.3e-15 splits and one of 4.0e-15
    does not. Short of it the first and last angles lose a few times 1e-16
    / cos(middle angle) radians to the rounding (6.4 times at most,
    measured), in the sdk as here, so a hair off gimbal the two agree only
    to that. Within about 1e-15 of the split each side's rounding of the
    cosine decides the branch -- the sdk's 4.1e-15 where this one is 3.2e-15
    -- and there the two can be whole rotations apart"""
    i, j, k, odd = _fbx_euler_axes(order)
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    norm = x * x + y * y + z * z + w * w
    s    = np.where(norm > 0.0, 2.0 / np.where(norm > 0.0, norm, 1.0), 0.0)
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    m            = np.empty(x.shape + (3, 3))
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    cy     = np.sqrt(m[..., i, i] * m[..., i, i] + m[..., j, i] * m[..., j, i])
    locked = cy <= _FBX_GIMBAL
    first  = np.where(locked, np.arctan2(-m[..., j, k], m[..., j, j]), np.arctan2(m[..., k, j], m[..., k, k]))
    middle = np.arctan2(-m[..., k, i], cy)
    last   = np.where(locked, 0.0, np.arctan2(m[..., j, i], m[..., i, i]))
    euler  = np.degrees(np.stack([first, middle, last], axis=-1))
    return -euler if odd else euler


def _fbx_held(curve_node, curve, axis: int) -> float:
    """the value read for a channel with a curve node but no keys: the curve
    node's own value, rounded to single precision when a curve is there at all.
    It is what the sdk takes where the channel has no curve. For a curve with
    no key the sdk's evaluator takes the curve's own default instead, which
    this does not read, and so does Maya where the curve is in the file's
    active take and the last take does not key its channel: under a curve node
    holding 55, a curve with no key whose default is 99 reads 55 here, 99 in
    the sdk and, there, in Maya"""
    held = curve_node.GetChannelValue(axis, 0.0)
    if curve is not None:
        held = float(np.float32(held))
    return held


def _fbx_layer_channel(layer, props, ticks, per_second: int) -> tuple:
    """one layer's values of one channel (translation, rotation or scaling)
    of every node, every frame at once: ``present`` (nodes,), whether the
    layer holds a curve node for the node's channel, ``owned`` (nodes, 3),
    the axes that curve node holds a channel for (an exporter can write a
    curve node with fewer than three), and ``values`` (frames, nodes, 3), 0
    where there is no curve node or no channel"""
    present = np.zeros(len(props), dtype=bool)
    owned   = np.zeros((len(props), 3), dtype=bool)
    values  = np.zeros((len(ticks), len(props), 3))
    curves  = []
    targets = []
    for column, prop in enumerate(props):
        curve_node = typed(prop.GetCurveNode(layer), FBX.FbxAnimCurveNode, FBX)
        if curve_node is None:
            continue

        present[column]                                = True
        owned[column, : curve_node.GetChannelsCount()] = True
        for axis in range(3):
            curve = typed(curve_node.GetCurve(axis), FBX.FbxAnimCurve, FBX)
            if curve is not None and curve.KeyGetCount():
                curves.append(curve)
                targets.append((column, axis))
            else:
                values[:, column, axis] = _fbx_held(curve_node, curve, axis)

    # as many curves at a time as keep the working arrays, curves x frames,
    # inside the budget however long the take
    targets = np.array(targets, dtype=np.intp).reshape(-1, 2)
    step    = max(1, _FBX_SAMPLE_BUDGET // len(ticks))
    for begin in range(0, len(curves), step):
        batch   = targets[begin : begin + step]
        sampled = _fbx_sample_curves(curves[begin : begin + step], ticks, per_second)
        # each curve's frames to its node and axis
        values[:, batch[:, 0], batch[:, 1]] = sampled.T
    return present, owned, values


def _fbx_layer_weight(layer, ticks, per_second: int) -> np.ndarray:
    """(frames,) a layer's weight, in percent: its own curve, the one its
    Weight property holds in the layer itself, or its value"""
    curve_node = typed(layer.Weight.GetCurveNode(layer), FBX.FbxAnimCurveNode, FBX)
    if curve_node is None:
        return np.full(len(ticks), float(layer.Weight.Get()))

    curve = typed(curve_node.GetCurve(0), FBX.FbxAnimCurve, FBX)
    if curve is not None and curve.KeyGetCount():
        return _fbx_sample_curves([curve], ticks, per_second)[0]
    return np.full(len(ticks), _fbx_held(curve_node, curve, 0))


def _fbx_blend(
    layer,
    channel: int,
    below:   np.ndarray,
    values:  np.ndarray,
    weight:  np.ndarray,
    orders:  np.ndarray,
    owned:   np.ndarray,
) -> np.ndarray:
    """a layer's ``values`` (frames, nodes, 3) of one channel over what the
    layers ``below`` it hold, as the sdk blends them

    ``weight`` is the layer's (frames,) in percent and need not lie in
    0..100. Additive adds weight x values, override takes weight x values,
    passthrough mixes the two by the weight. Scaling multiplies by default:
    additive by values ** weight, override takes values ** weight,
    passthrough below ** (1 - weight) x values ** weight, every term
    unsigned, as the sdk takes them. Rotation by layer (the default) blends
    quaternions built as the sdk builds them for each node's fbx rotate
    order (``orders``, see :func:`_fbx_turns`), its stored one, rotation
    active or not, but xyz at a root whose rotation ConvertScene turns on
    in xyz (:class:`_FbxTree`'s ``blend``), which blend as an xyz node's
    for an even order and as a zyx node's for an odd one, an additive layer
    then turning on the other side of what is below
    (:func:`_fbx_euler_axes`): additive turns by the layer's rotation scaled
    by the weight, the long way round when that is how its quaternion comes
    out, after what is below; passthrough is the shortest spherical blend;
    override takes the layer's rotation, whatever the weight. The blend is
    read back as the sdk reads it, gimbal included
    (:func:`_fbx_turns_to_euler`). By channel, rotation blends angle by
    angle. A layer whose blend mode is bypassed for double3 values
    overrides.

    An axis the layer's curve node holds no channel for (``owned`` (nodes,
    3) False) keeps what is below, as the sdk keeps it, except in a
    rotation blended by layer, which reads it as 0 as the sdk does.
    """
    mode = int(layer.BlendMode.Get())
    if layer.GetBlendModeBypass(_fbx_enum(FBX, "EFbxType", "eFbxDouble3")):
        mode = 1
    additive, override = mode == 0, mode == 1
    w = (weight / 100.0)[:, None, None]

    if channel == 1 and int(layer.RotationAccumulationMode.Get()) == 0:
        # one pass per rotate order the nodes hold, every frame at once
        blended = np.empty_like(values)
        for order in np.unique(orders).tolist():
            nodes = orders == order
            under = _fbx_turns(below[:, nodes], order)
            turns = _fbx_turns(values[:, nodes], order)
            if override:
                turned = turns
            elif additive:
                identity         = np.zeros_like(turns)
                identity[..., 3] = 1.0
                scaled           = _fbx_slerp(identity, turns, w[..., 0], shortest=False)
                turned           = _fbx_turn_product(under, scaled)
            else:
                turned = _fbx_slerp(under, turns, w[..., 0], shortest=True)
            blended[:, nodes] = _fbx_turns_to_euler(turned, order)
        return blended

    if channel == 2 and int(layer.ScaleAccumulationMode.Get()) == 0:
        # a zero scaling to a negative power is infinite, in the sdk too
        with np.errstate(divide="ignore", invalid="ignore"):
            if additive:
                out = np.abs(below) * np.abs(values) ** w
            elif override:
                out = np.abs(values) ** w
            else:
                out = np.abs(below) ** (1.0 - w) * np.abs(values) ** w
    elif additive:
        out = below + w * values
    elif override:
        out = w * values
    else:
        out = (1.0 - w) * below + w * values
    return np.where(owned[None], out, below)


def _fbx_sample_take(stack, sampled, ticks, tree: _FbxTree, blocks) -> None:
    """the current take's local channels of ``sampled`` ((node, column) pairs,
    a column a position of ``tree``) on the ``ticks`` grid, written into
    ``blocks`` (translate, rotate, scale, each frames x nodes x 3), the
    translate times the column's ``lengths`` and the scale divided by its
    ``unit_scale``, as the walk reads the static values; ``tree.orders``
    holds each column's rotate order as the read turns it (rotate_order) and
    ``tree.blend`` the fbx rotate order the sdk blends its rotation by layer
    in (:class:`_FbxTree`)

    A unit conversion (:func:`_fbx_units`) converts the curves of each
    take's first layer alone, muted or not, and leaves every later layer's
    as they were: on a unit root, the translate and the scale of that layer
    alone are converted, before the layers are blended, and on a node under
    one that does not inherit its parent's scale, the scale of that layer.

    The curves of every node are evaluated together, every frame at once,
    and the take's animation layers blended over them as the sdk blends
    them: from the first layer that is not muted and holds anything (a
    muted or empty one below it does not count), each later layer that is
    not muted over it, by its weight, blend mode and accumulation modes
    (:func:`_fbx_blend`); solo changes nothing. A node's channel the first
    layer holds no curve node for starts from 0 under the later ones. A
    channel no layer holds a curve node for keeps the rest pose ``blocks``
    already hold, the same value the sdk hands back.

    A rotation blended by layer is rebuilt from quaternions, each frame on
    its own, as the sdk rebuilds it, in the node's stored rotate order
    whether its rotation is active or not, but in xyz at a root whose
    rotation is inactive in a file the read turns into Maya's axes, whose
    rotation ConvertScene turns on in xyz (``blend``), so its curve goes
    through euler_filter in the order the read turns the node in
    (``orders``: xyz where the rotation is inactive, as the sdk turns it),
    frame 0 as the sdk gives it, where the sdk's curve can step by a whole
    turn or flip near gimbal. A curve no such layer touches keeps the
    branches the file holds.
    """
    criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId)
    layers   = [stack.GetSrcObject(criteria, i) for i in range(stack.GetSrcObjectCount(criteria))]
    layers   = [typed(layer, FBX.FbxAnimLayer, FBX) for layer in layers]
    first    = layers[0].GetUniqueID() if layers else None
    layers   = [layer for layer in layers if not layer.Mute.Get()]
    while layers and not layers[0].GetMemberCount():
        layers.pop(0)
    if not layers or not sampled:
        return

    per_second = _fbx_ticks_per_second()
    columns    = np.array([index for _, index in sampled], dtype=np.intp)
    weights    = [_fbx_layer_weight(layer, ticks, per_second) for layer in layers[1:]]
    blend      = tree.blend[columns]
    orders     = tree.orders

    # the lengths: a unit root's translate is multiplied on the take's first
    # layer before the blend, every other node's on the blend after it; the
    # scale that holds the unit is divided on that first layer
    roots     = tree.unit_roots[columns]
    converted = layers[0].GetUniqueID() == first
    after     = np.where(roots, 1.0, tree.lengths[columns])
    lengths   = np.where(roots, tree.lengths[columns], 1.0)[None, :, None]
    sizes     = tree.unit_scale[columns][None]

    for channel, name in enumerate(("LclTranslation", "LclRotation", "LclScaling")):
        props = [getattr(node, name) for node, _ in sampled]
        keyed, _, blended = _fbx_layer_channel(layers[0], props, ticks, per_second)
        if converted and channel == 0 and roots.any():
            blended = blended * lengths
        elif converted and channel == 2 and np.any(sizes != 1.0):
            blended = blended / sizes
        rebuilt = np.zeros(len(sampled), dtype=bool)
        for layer, weight in zip(layers[1:], weights):
            present, owned, values = _fbx_layer_channel(layer, props, ticks, per_second)
            if present.any():
                blended[:, present] = _fbx_blend(
                    layer,
                    channel,
                    blended[:, present],
                    values[:, present],
                    weight,
                    blend[present],
                    owned[present],
                )
                keyed |= present
                if channel == 1 and int(layer.RotationAccumulationMode.Get()) == 0:
                    rebuilt |= present

        if rebuilt.any():
            blended[:, rebuilt] = _filtered(blended[:, rebuilt], orders[columns[rebuilt]])
        if not keyed.any():
            continue

        rows = columns[keyed]
        if channel == 0:
            blocks[0][:, rows] = blended[:, keyed] * after[keyed][None, :, None]
        else:
            blocks[channel][:, rows] = blended[:, keyed]


# a uuid, name or parent_uuid the rig index can key on
_INDEX_KEYS = (str, type(None))


# the rows of a key this many or more become a deque the first time one is
# put in or taken out ahead of the end: a loop reparenting the children of
# one parent takes each from the front, which in a list moves every later
# row, the square of the children over the loop. A build keeps lists, the
# most compact for the one row most keys hold
_DEQUE_ROWS = 32

# the renames a name table takes past a quarter of the rig's nodes, with no
# lookup by name between, before it lapses for the next lookup to build
# (_RigIndex.named). A rename patching it costs some 0.12 us more than one
# with none to patch, and a build 0.05 us a node (2000 nodes, one thread):
# a loop renaming every node pays a quarter of the patches, a loop looking
# a name up after each rename builds nothing, and the costliest, a lookup
# after each quarter of the nodes renamed, pays under three times the
# patches it would have made
_RENAMES_SPARED = 16


def _insort_row(table: dict, key, row: int) -> None:
    """puts ``row`` in order among the rows ``table`` holds for ``key``"""
    rows = table.get(key)
    if rows is None:
        table[key] = [row]
    elif rows[-1] < row:
        rows.append(row)
    else:
        if len(rows) >= _DEQUE_ROWS and type(rows) is list:
            rows = table[key] = deque(rows)
        bisect.insort(rows, row)


def _remove_row(table: dict, key, row: int) -> bool:
    """takes ``row`` out of the rows ``table`` holds for ``key``; False, and
    nothing changed, when it does not hold it there"""
    rows = table.get(key)
    if rows is None:
        return False

    # the first or the last, as a loop over the children takes them, or the
    # second: a loop over the roots in list order, chaining them, keeps the
    # first root and takes each one after it
    if rows[0] == row:
        at = 0
    elif rows[-1] == row:
        at = len(rows) - 1
    elif len(rows) > 2 and rows[1] == row:
        at = 1
    else:
        at = bisect.bisect_left(rows, row)
        if at == len(rows) or rows[at] != row:
            return False
    if at < len(rows) - 1 and len(rows) >= _DEQUE_ROWS and type(rows) is list:
        rows = table[key] = deque(rows)
    del rows[at]
    if not rows:
        del table[key]
    return True


class _RigIndex:
    """One pass over a hierarchy's nodes, so that get_parent, get_children,
    index, unique_name and a node looked up by name read dicts rather than
    scanning the nodes

    ``uuids`` holds the position of the first node carrying each uuid and
    ``refs`` the positions of the nodes holding each ``parent_uuid``, in
    order. Every answer follows the node method's rule:

    * a ``parent_uuid`` names the first node carrying that uuid; one no node
      carries comes back as the string, as get_parent returns it;
    * a node's children hold its uuid, and come in hierarchy order;
    * a node's index is the position of the first node with its uuid;
    * a name finds the first node carrying it, as ``index`` does.

    ``names``, the position of the first node carrying each name, is built
    on the first lookup by name (:meth:`named`), with ``repeated``, every
    name two or more nodes may carry, and kept with the rest. A rename of
    the one node carrying its name patches it, the new name's first node
    being known; a rename of a node whose name other nodes may carry can
    not tell which node is first of it after, and drops the name table for
    the next lookup by name to build again. So do renames past ``spare``,
    a quarter of the nodes and _RENAMES_SPARED, with no lookup by name
    between: a loop renaming the rig stops patching a table no lookup
    reads. The rows of ``refs`` are lists, a long one made a deque when a
    patch puts a row in or takes one out ahead of its end
    (:func:`_remove_row`).

    :meth:`build` gives None for a hierarchy it cannot stand for: a class
    that answers these lookups its own way, or a node whose uuid, name or
    parent_uuid is not a string or None. Its callers ask the nodes instead.

    The hierarchy patches its index in place for the edits that change a
    few entries (:meth:`add`, :func:`_parent_written`, :func:`_renamed`):
    a node appended, given another parent_uuid or renamed costs a few dict
    writes, not the pass over every node a build costs. A node inserted
    ahead of the end moves every position after it up one (:meth:`_shift`),
    a pass over every entry that costs a tenth to two fifths of a build,
    the more the nearer the front: the hierarchy makes it only for an index
    a lookup has fetched since the last one (``fetched``), and otherwise
    retires the index, which the next lookup builds again.
    """

    __slots__ = ("nodes", "uuids", "refs", "names", "repeated", "spare", "fetched", "_holders")

    def __init__(self, nodes: list, uuids: dict, refs: dict):
        self.nodes    = nodes
        self.uuids    = uuids
        self.refs     = refs
        self.names    = None   # name -> its first position, on the first lookup by name
        self.repeated = None   # the names two or more nodes may carry, with names
        self.spare    = 0      # the renames names takes before it lapses, set by named
        self.fetched  = False  # a lookup read it since an insert last shifted it
        self._holders = None   # name -> the uuids carrying it, on first use

    @classmethod
    def build(cls, hierarchy) -> Optional["_RigIndex"]:
        kind = type(hierarchy)
        if not (
            kind.__contains__ is TransformList.__contains__
            and kind.index is TransformList.index
            and kind.__getitem__ is TransformList.__getitem__
        ):
            return None

        nodes = hierarchy.list
        uuids, refs = {}, {}
        for position, node in enumerate(nodes):
            uuid, ref = node.uuid, node.parent_uuid
            if not (
                isinstance(uuid, _INDEX_KEYS)
                and isinstance(node.name, _INDEX_KEYS)
                and isinstance(ref, _INDEX_KEYS)
            ):
                return None
            uuids.setdefault(uuid, position)
            refs.setdefault(ref, []).append(position)

        return cls(nodes, uuids, refs)

    def resolve(self, ref: str):
        """the node a ``parent_uuid`` names, the uuid when no node carries
        it, as get_parent answers"""
        position = self.uuids.get(ref)
        return ref if position is None else self.nodes[position]

    def parent(self, node):
        """``node.get_parent()`` for a ``parent_uuid`` that is a string"""
        return self.resolve(node.parent_uuid)

    def position(self, node) -> Optional[int]:
        """``node``'s position, None when the index does not hold it there: a
        node no longer listed, or one whose uuid an earlier node carries"""
        uuid = node.uuid
        if isinstance(uuid, _INDEX_KEYS):
            position = self.uuids.get(uuid)
            if position is not None and self.nodes[position] is node:
                return position
        return None

    def named(self, name: str) -> Optional[int]:
        """the position of the first node carrying ``name``, None for none;
        the name table is built on the first call, and each call spares it
        for a quarter of the nodes renamed (``spare``)"""
        names = self.names
        if names is None:
            names, repeated = {}, set()
            for position, node in enumerate(self.nodes):
                if names.setdefault(node.name, position) != position:
                    repeated.add(node.name)
            self.names, self.repeated = names, repeated
        self.spare = (len(self.nodes) >> 2) + _RENAMES_SPARED
        return names.get(name)

    # --- patches, each for an edit just made to the list or to one node --- #
    def add(self, at: int, node) -> bool:
        """``node`` was just put at ``at`` in the list. False, and nothing
        patched, when a uuid, name or parent_uuid is not a key it can hold"""
        uuid, name, ref = node.uuid, node.name, node.parent_uuid
        if not (
            isinstance(uuid, _INDEX_KEYS)
            and isinstance(name, _INDEX_KEYS)
            and isinstance(ref, _INDEX_KEYS)
        ):
            return False

        if at < len(self.nodes) - 1:
            self._shift(at)
        first = self.uuids.get(uuid)
        if first is None or at < first:
            self.uuids[uuid] = at
        if self.names is not None:
            self._take_name(name, at)
        _insort_row(self.refs, ref, at)
        if self._holders is not None:
            self._holders.setdefault(name, set()).add(uuid)
        return True

    def _take_name(self, name, at: int) -> None:
        """the node at ``at`` carries ``name``, which the name table has"""
        first = self.names.get(name)
        if first is None:
            self.names[name] = at
        else:
            self.repeated.add(name)
            if at < first:
                self.names[name] = at

    def _shift(self, at: int) -> None:
        """every position from ``at`` on moves up one, for a node inserted"""
        for table in (self.uuids, self.names or {}):
            for key, position in table.items():
                if position >= at:
                    table[key] = position + 1

        # rows are in order: only their tail moves. Most hold one row, a
        # node's single child, written without a loop
        for rows in self.refs.values():
            last = rows[-1]
            if last >= at:
                if len(rows) == 1:
                    rows[0] = last + 1
                else:
                    for k in range(bisect.bisect_left(rows, at), len(rows)):
                        rows[k] += 1

    def children(self, node) -> list:
        """``node.get_children()``, as a list"""
        uuid = node.uuid
        if uuid is None:
            return []
        return [self.nodes[position] for position in self.refs.get(uuid, ())]

    def unique_name(self, node) -> bool:
        """``node.unique_name`` for a node of this hierarchy"""
        if self._holders is None:
            self._holders = {}
            for member in self.nodes:
                self._holders.setdefault(member.name, set()).add(member.uuid)

        # shared when a node of another uuid carries the name too
        return all(uuid == node.uuid for uuid in self._holders.get(node.name, ()))


class _Lookups:
    """get_parent, get_children, index and unique_name for many nodes at once

    A node's get_parent, index and unique_name scan its hierarchy, so asking
    them of every node of a rig costs the rig's size squared. Here each
    owning hierarchy's index (:class:`_RigIndex`) is fetched the first time
    one of its nodes is asked about, and held for the length of the call: a
    pass that edits as it goes reads the rig as it was when it started. A
    HierarchyData keeps its index between calls, until its structure
    changes (:meth:`HierarchyData._rig_index`). A node whose hierarchy can
    not be indexed is asked itself.
    """

    __slots__ = ("_indexes",)

    def __init__(self):
        self._indexes = {}

    def index_of(self, hierarchy) -> Optional[_RigIndex]:
        """the index of ``hierarchy``, fetched on first use"""
        entry = self._indexes.get(id(hierarchy))
        if entry is None:
            # a HierarchyData keeps its index until its structure changes
            kept  = getattr(hierarchy, "_rig_index", None)
            index = kept() if kept is not None else _RigIndex.build(hierarchy)
            entry = (hierarchy, index)
            self._indexes[id(hierarchy)] = entry
        return entry[1]

    def parent(self, node):
        """``node.get_parent()``"""
        hierarchy, ref = node._hierarchy, node.parent_uuid
        if hierarchy is None or ref is None:
            return None

        index = self.index_of(hierarchy)
        if index is None or not isinstance(ref, str):
            return node.get_parent()
        return index.resolve(ref)

    def children(self, node) -> list:
        """``node.get_children()``, as a list"""
        hierarchy = node._hierarchy
        if hierarchy is None:
            return []

        index = self.index_of(hierarchy)
        if index is None or not isinstance(node.uuid, _INDEX_KEYS):
            return node._scan_children()
        return index.children(node)

    def index(self, node) -> int:
        """``node.index``"""
        hierarchy = node._hierarchy
        index     = None if hierarchy is None else self.index_of(hierarchy)
        if index is not None and isinstance(node.uuid, str):
            position = index.uuids.get(node.uuid)
            if position is not None:
                return position

        # not listed, or no hierarchy: raises as it always has
        return node.index

    def unique_name(self, node) -> bool:
        """``node.unique_name``"""
        hierarchy = node._hierarchy
        index     = None if hierarchy is None else self.index_of(hierarchy)
        if index is None or not isinstance(node.name, str):
            return node.unique_name
        return index.unique_name(node)


def _walk(roots, lookups: _Lookups) -> list:
    """each of ``roots`` not met yet, followed by its branch: every node
    before its children, the children in hierarchy order, each node once

    Walked with a stack rather than by recursion, so a chain deeper than
    Python's recursion limit is walked too, and a cyclic parent_uuid ends
    where the walk meets a node again.
    """
    walked, seen = [], set()
    for root in roots:
        if id(root) in seen:
            continue
        seen.add(id(root))
        walked.append(root)

        stack = [iter(lookups.children(root))]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
            elif id(child) not in seen:
                seen.add(id(child))
                walked.append(child)
                stack.append(iter(lookups.children(child)))

    return walked


def _world_down(node) -> np.ndarray:
    """``node``'s world matrix, worked out from the nearest ancestor whose
    world matrix is cached, or from a root, every node on the way cached

    A loop rather than each node asking its parent: a chain deeper than the
    recursion limit reads too. The parents come from the index the rig
    keeps, one build after a structural edit, then a dict read a level,
    where each node's get_parent scanned the rig twice. A cyclic
    parent_uuid has no world matrix: RecursionError, as the recursion gave.

    A node whose parent its rig lists but does not own -- another rig's
    node, or no rig's, put in by an item assignment -- is moved by what
    moves that parent there: a channel written on it or above it, an edit
    of the rig owning it, which walk that rig's branches and count on that
    rig alone. Its cache, and that of every node of its rig below it, holds
    the link it rests on as well (_linked_world): that parent, weakly, and
    the world matrix the parent had cached, which any such edit drops or
    retires.
    """
    lookups, chain, seen = _Lookups(), [], set()
    parent_world = None
    while parent_world is None:
        chain.append(node)
        seen.add(id(node))
        parent = lookups.parent(node)
        if not isinstance(parent, TransformData):
            # a root, or a parent the hierarchy does not hold
            parent       = None
            parent_world = np.eye(4)
        elif id(parent) in seen:
            raise RecursionError(f"the parent_uuid of {parent.name!r} leads back to it")
        else:
            parent_world = parent._cached_world()
            node         = parent

    # the link each node rests on, None for none: a cached parent of its own
    # rig hands its own down, a parent of another rig or of none makes one
    world, above, link = parent_world, parent, None
    if above is not None and above._world_version.__class__ is tuple:
        link = above._world_version[1:]
    for node in reversed(chain):
        if above is not None and above._hierarchy is not node._hierarchy:
            link = (weakref.ref(above), world)
        world               = np.dot(node.matrix, world)
        stamp               = _structure_stamp(node)
        node._world_matrix  = world
        node._world_version = stamp if link is None else (stamp,) + link
        above               = node
    return world


def _linked_world(node) -> Optional[np.ndarray]:
    """``node``'s cached world matrix when it rests on links (_world_down)
    that all hold, else None

    Each link names the nearest parent above of another rig, or of none,
    and the world matrix that parent had cached: the parent must have it
    cached still, at its own rig's structure count, and so on up each link
    in turn. A loop, a link a step, so a chain of links deeper than the
    recursion limit reads too.
    """
    world, version = node._world_matrix, node._world_version
    if world is None:
        return None
    while version.__class__ is tuple:
        stamp, held, seen = version
        if stamp != _structure_stamp(node):
            return None
        node = held()
        if node is None or node._world_matrix is not seen:
            return None
        version = node._world_version
    return world if version == _structure_stamp(node) else None


def _chain_depths(parents) -> np.ndarray:
    """the number of ancestors above each entry of a parent array

    ``parents[i]`` is the position of i's parent, negative for none. The
    count is the one a walk up the array makes, every distinct entry met on
    the way up, stopping at a root or at an entry met already. It is found
    once per entry rather than once per entry per level, so a chain N deep
    costs N steps instead of N squared. A cycle counts its own members, as
    the walk does, so a cyclic parent_uuid still orders the same.
    """
    parents = [int(p) for p in parents]
    depth   = [-1] * len(parents)
    cyclic  = [False] * len(parents)

    for start in range(len(parents)):
        path, on_path, i = [], {}, start
        while i > -1 and depth[i] < 0 and i not in on_path:
            on_path[i] = len(path)
            path.append(i)
            i = parents[i]

        # the walk came back on itself: every member of the loop meets the
        # whole loop, itself included, on its way up
        if i > -1 and i in on_path:
            loop = path[on_path[i] :]
            for j in loop:
                depth[j], cyclic[j] = len(loop), True
            path = path[: on_path[i]]

        for j in reversed(path):
            p        = parents[j]
            depth[j] = 0 if p < 0 else depth[p] + (not cyclic[p])

    return np.array(depth, dtype=int)


def _reset_branches(nodes) -> None:
    """drops the cached world matrices of ``nodes`` and of every node below
    them, as _reset_branch_world_matrices does for one

    The branches are walked together, each node once, through one index
    of each hierarchy: a branch shared by several of ``nodes`` is not
    walked again for each.
    """
    for node in _walk(nodes, _Lookups()):
        node._transform_changed()


class TransformList(DataList[TransformData]):
    """Non-owning ordered view over ``TransformData`` nodes.

    A ``TransformList`` is a *selection* of nodes (e.g. the result of
    :meth:`match`, :meth:`get_children`, slicing, or fancy indexing). It never
    takes ownership of the nodes it contains: it does NOT set their
    ``_hierarchy`` back-pointer. That back-pointer keeps referencing the owning
    :class:`HierarchyData`, so queries that depend on it (``get_parent``,
    ``world_matrix``, ``index``, ``unique_name``, name-based ``__eq__`` ...)
    keep returning data relative to the real hierarchy even when a node is
    reached through a view. Use :class:`HierarchyData` when you need an owning
    container that mutates the hierarchy graph.

    Every channel settable on a node is settable here, applying to the whole
    selection at once, and the pose algebra (:meth:`get_delta` /
    :meth:`add_delta`) is defined here so a delta can be taken over a
    selection. Because a delta has to own its nodes, ``get_delta`` returns a
    :class:`HierarchyData` even when called on a view. So does a saved or
    pickled view: it saves as its :meth:`copy`, a :class:`HierarchyData`,
    which ``pickle.loads``, ``HierarchyData.load()``, ``Data.load()`` or
    ``cgmath.hierarchy.load()`` gives back; ``TransformList.load()`` refuses
    its file.
    """

    # the private state a view holds beside its list: the uuids extend()
    # keeps while it adds (_holds), and the record of a load (_record)
    _STATE = ("_held", "_record")

    # (axis_system, unit) TransformList.load_fbx records on the view it gives,
    # whose nodes no rig owns, and every view taken from that one carries
    # (_view); None on any other view. It stands for the nodes no rig owns:
    # a node a rig owns is in what that rig records (_records)
    _record = None

    def __init__(self, iterable=None):
        super().__init__(iterable)
        # a view of the one load_fbx gives holds its load's record too
        record = iterable._record if isinstance(iterable, TransformList) else None
        if record is not None and not isinstance(self, HierarchyData):
            self._record = record

    def _view(self, nodes) -> "TransformList":
        """a view over ``nodes``, taken from this one: a view of the one
        :meth:`load_fbx` gives carries its load's record (``_record``)"""
        view = TransformList(nodes)
        if self._record is not None:
            view._record = self._record
        return view

    @property
    def axis_system(self) -> str:
        """the axis system the values are in, as the nodes' rig records it

        :attr:`HierarchyData.axis_system`, read only. A view whose nodes no
        rig owns says "maya_y_up", as hand-built data does, but for the one
        :meth:`load_fbx` gives, and every view taken from it (a slice or an
        index of it, its :meth:`match`, :meth:`get_roots`, ``TransformList``
        of it), which say what its load produced until a rig owns their
        nodes (``HierarchyData(view)``), and then what that rig records. A
        view holding nodes of rigs that record other axes or units raises
        ValueError, and so do its :meth:`copy` and :meth:`save_fbx`; so
        does :meth:`load_fbx`'s view, or one taken from it, that holds
        nodes no rig owns beside nodes of a rig that records another.
        """
        return self._records()[0]

    @property
    def unit(self) -> float:
        """the cm one unit of the values' lengths holds, as their rig records it

        :attr:`HierarchyData.unit`, read only; a view whose nodes no rig
        owns says 1.0, as :attr:`axis_system` says.
        """
        return self._records()[1]

    def _held_records(self) -> list:
        """the ``(axis_system, unit)`` records the nodes hold, each once,
        sorted: those the rigs that own them record, and the load's of a
        view :meth:`load_fbx` gave for its nodes no rig owns, or for the view
        left empty; none for a node no rig owns of any other view"""
        records = _owner_records(self.list)
        if self._record is not None and (
            not self.list or any(getattr(node, "_hierarchy", None) is None for node in self.list)
        ):
            records = sorted(set(records) | {self._record})
        return records

    def _records(self) -> tuple:
        """``(axis_system, unit)`` of the values: those the rigs that own the
        nodes record, for the nodes no rig owns the load's of a view
        :meth:`load_fbx` gave, and Maya's Y-up and cm for nodes no rig owns
        of any other view

        A rig that owns the nodes of a load's view, ``HierarchyData(view)``,
        holds them in its record, and a relabel of it relabels them: the
        rig is asked first, and the load's record stands only for nodes no
        rig owns, or for the view left empty.

        Raises ValueError when the nodes hold two of them.
        """
        records = self._held_records()
        if len(records) > 1:
            raise ValueError(
                "the view holds nodes whose values are in other axes or units "
                f"({_records_listed(records)}): no one axis_system and unit says "
                "where they are; give them one first"
            )
        return records[0] if records else (_MAYA_Y_UP, 1.0)

    # TODO scale_factor is prolly not the best name, think of a better way to handle unit conversions
    @classmethod
    def load_glb(
        cls, filename: str, scale_factor: float = 100.0, pose_frame: None | int = None
    ):
        """primitive glb joint loader that supports style2 and momentum glbs

        A node holding a mesh is not read as a joint, whatever its name, and
        ``pose_frame`` leaves its channels out: a joint named like it keeps
        its own pose.
        """

        # make sure pygltflib is available
        if GLTF2 is None:
            raise ImportError("pygltflib not found")

        # convert the glb to dict
        filename = os.path.expanduser(filename)

        # parse the nodes and look for style2 or momentum joints
        data = load_model(filename)

        # None stands for a node skipped; _glb_columns skips the same ones
        node_list = []
        for node in data.nodes:
            node_list.append(None)

            node_type = _glb_node_type(node, data.nodes)
            if node_type is not None:
                new_node              = {"name": node.name}
                new_node["node_type"] = node_type
                new_node["uuid"]      = generate_uuid()

                if node.scale:
                    new_node["scale"] = node.scale

                if node.rotation:
                    new_node["rotate"] = node.rotation

                if node.translation:
                    new_node["translate"] = node.translation

                if node.children:
                    new_node["children"] = node.children

                node_list[-1] = new_node

        # convert the dict so it matches the TransformData class properties
        hierarchy_data = {}

        # load_model already walked the file's tree parent first. The node
        # array itself carries no such promise -- exporters write it leaf
        # first often enough -- and a dcc has to create a parent before it
        # can parent anything to it.
        for position in data.ordered_node_indexes:
            node = node_list[position]
            if node is not None:
                # convert children to parent_uuid
                if "children" in node:
                    for index in node.pop("children", None):
                        if node_list[index] is not None:
                            node_list[index]["parent_uuid"] = node["uuid"]

                # convert rotation to joint_orient
                if "rotate" in node:
                    Q                    = node.pop("rotate", None)
                    node["joint_orient"] = np.degrees(quaternion_to_euler(Q, 0)[0])

                # convert translation to translate
                if "translate" in node:
                    node["translate"] = np.array(node["translate"]) * scale_factor

                hierarchy_data[node["uuid"]] = node

        # set the pose_frame if requested
        hierarchy_data                          = cls.from_dict(hierarchy_data)
        hierarchy_data.segment_scale_compensate = False

        if pose_frame is not None and data.animations:
            animation = data.animations[0]
            for channel in animation.channels:
                # a node skipped has no row, and its name would pose a joint
                # that shares it: a mesh named like a joint, say
                if node_list[channel.node] is None:
                    continue

                node_name = data.nodes[channel.node].name
                if node_name in hierarchy_data:
                    attribute = channel.path
                    sampler   = animation.samplers[channel.sampler]
                    values    = sampler.keyframe_values
                    values    = values[pose_frame]

                    if attribute in ["scale", "translation", "rotation"]:
                        if attribute == "translation":
                            hierarchy_data[node_name].translate = values * scale_factor
                        elif attribute == "rotation":
                            q0    = hierarchy_data[node_name].quaternion
                            q1    = values
                            delta = q_multiply(q_conjugate(q0), q1)
                            hierarchy_data[node_name].rotate = np.degrees(
                                quaternion_to_euler(
                                    delta, hierarchy_data[node_name].rotate_order
                                )[0]
                            )
                        else:
                            hierarchy_data[node_name].scale = values

        return hierarchy_data

    @classmethod
    def load_fbx(
        cls,
        filename:     str,
        scale_factor: Optional[float] = None,
        axis_system:  Optional[str]   = "maya_y_up",
        pose:         str             = "node",
        node_types:   Optional[tuple] = None,
    ):
        """primitive fbx joint loader

        ``scale_factor`` None, the default, converts the file's unit to
        centimeters: translations are multiplied by the centimeters one unit of
        the file holds (``GetSystemUnit().GetScaleFactor()``), s, 1.0 for a
        file in cm, 100.0 for one in meters. Only lengths are converted;
        rotations, scales and user defined attributes keep the file's values,
        but for a unit held as a scale. The sdk's
        ``FbxSystemUnit.ConvertScene`` and Maya's unit converting export
        (``FBXExportConvertUnitString``) write a scene in another unit by
        scaling each child of the scene root by 1/s, and each node that does
        not inherit its parent's scale (a joint's segment scale compensate) by
        1/s as well, over values left in the unit they were in, cm for a Maya
        scene. A child of the scene root scaled 1/s on all three axes, within
        1e-9 relative, is read as that unit, whether the read keeps it or drops
        it (``node_types`` below): its scale is reset to 1 and divided out of
        each node under it that does not inherit its parent's scale, and the
        lengths under it are read as they are, on the rig, on every frame of a
        clip, in the bind pose and by the mesh and skin readers. So such a file
        loads as the cm file it was written from. A file only labelled in a
        unit, its values in that unit, has every length multiplied as above,
        and so does a root the export found scaled other than 1: its nodes load
        as the sdk places them, where the cm file has them, scaled 1/s, and the
        points of a mesh at or under it s times larger, right once posed. But
        the export leaves the curves of a take's later layers in cm, and on
        such a root and the nodes under it the read takes them as the sdk
        places them. A translate the root keys on one is multiplied by s with
        the rest: on a clip the root and every node under it are off by s - 1
        times that layer's own translate times its weight, on each channel
        its curve node holds: what an additive layer adds, an override or
        passthrough layer's whole value, a channel it holds unkeyed too
        (Maya's meters export of a group scaled 2 whose translate x and z an
        override animLayer keys: 2475 cm off on frame 5, and 198 along y,
        which the layer holds at the group's own value). A scale an override
        or passthrough layer keys on the root, or on a node under it that does
        not inherit its parent's scale, is read as the cm value it is, s
        times what the rest of the read gives it, and every node under it is
        off (that group's scale x on an override animLayer: its joints 792
        and 1780 cm off); an additive layer's scale, which multiplies, loads
        right. And a node there that compensates its parent's scale under a
        parent that is neither the root nor compensating, as the second joint
        of a Maya skeleton does when its first joint sits under such a root, is
        scaled 1/s once more by the export, and every node under it with it.
        Read by their own values (``pose`` "node", and a clip on every frame
        and its rest pose), that node loads at its cm position scaled 1/s^2,
        the nodes under it off their cm positions, and a mesh on or under it,
        or skinned to it, is off once posed. Under ``pose`` "bind" a node a
        bind pose lists takes its bind matrix, which the export converts as a
        world matrix: it loads where the cm file has it, scaled 1/s like the
        rest, and so do the nodes under it and the meshes on them or skinned to
        them, posed at the bind. A node no bind pose lists keeps what its own
        values give it under its parent's bind: the node the export scales once
        more, and a compensating node under it, still loads scaled 1/s^2, and
        the nodes under them that no bind pose lists are off, but for those
        under a node one lists, which load right, as that node does; any other
        loads scaled 1/s like the rest, as the joints of a skeleton whose root
        joint is scaled other than 1 do, the end joints Maya's export leaves
        out of the bind pose among them. Maya's export lists every skin
        influence, the joints between them and the nodes above them. Maya's
        importer scales every root by s whatever its scale, and every
        compensating node, which brings all of these back but puts a file only
        labelled in meters at scale 100. The one risk: a root authored at
        exactly 1/s in a file labelled in that unit is read as the unit. A
        number is the caller's own factor, and neither the file's unit nor a
        unit held as a scale is read.

        ``axis_system`` "maya_y_up", the default, loads the scene in Maya's
        Y-up axes as ``FbxAxisSystem.MayaYUp.ConvertScene`` converts it:
        every node's world matrix is the file's times the turn from the
        file's axis system to Maya's. Maya's FBX importer puts a file whose
        up axis is x or z, an Unreal file among them, in the same place, but
        for a skeleton node at the root of the file, eRoot or eEffector, that
        holds no eLimbNode child, which it imports as a transform, 1.41 off
        ConvertScene's in its 3x3 in a Z-up file and 1.28 in an X-up one, the
        nodes under it up to 2.98 and 5.0 off in translate, where the read
        follows ConvertScene: one that holds an eLimbNode, as an Unreal file's
        skeleton root does, imports as a joint where ConvertScene puts it,
        and save_fbx writes every joint as an eLimbNode. Maya's importer also
        parts from the read on a node below the roots that is no camera or
        light whose rotation is inactive, in another order than xyz, which it
        turns by neither order (``node_types`` below). At the file's rate, with
        FBXImportResamplingRateSource "File", or from a scene whose rate is a
        whole multiple of the file's but 90 fps, Maya's import and the read are
        up to 2.0 apart in the 3x3 on a node of world scale 1: on keys within
        whole turns of the angles Maya's xyz solve returns (y between -90 and
        90), less the less the node turns, up to about the square of that turn
        in radians on turns under 20 degrees, and on keys off that branch, as
        (180, 170, 180) is of (0, 10, 0), up to about twice the turn in radians
        in xzy, yzx, yxz and zxy, zyx keeping the square; and on a take up to
        3.9e-4 more where Maya holds the rotate it solves level
        (:meth:`ClipData.load_fbx`). Against the same node under the same
        parents with every scale 1, a scale multiplies that by the node's world
        scale, by the largest axis of an uneven one at most, where the scales
        above the node are even; by up to the product of the largest axes of
        its own scale and those above it where no parent scaled unevenly is
        turned or sits under a turned one; and otherwise by up to 1.19 times
        that product on the files measured (a node scaled (3, 1, 1) under a
        group scaled (1, 1, 3) and turned), 1.46 times it in a numpy scan of
        the rule Maya follows and never more than 2.13 times it. Such a node
        turns about its rotate pivot: a node under it is off in translate by up
        to twice its world distance from that pivot, and so is the node itself
        where it has one, times the largest over the smallest singular value of
        the node's parent's world 3x3 (1 where the scales above it are even,
        and that 3x3's longest row over its shortest only where no turned
        parent sits under one scaled unevenly), and a node on the pivot agrees.
        On the files measured, up to 3.0e-4 on turns of 1.1 degrees, 3.3e-2 on
        11, 0.32 on 35, 1.1 on 72 and 2.0 on turns past 110, and off the branch
        3.6e-2 on turns of 1.4 degrees, 7.1e-2 on 3 and 0.35 on 10; on turns of
        up to 111 degrees, 5.1 on a node scaled 3, 4.9 on one scaled (1, 2, 3)
        and 3.4 under one scaled 2; on turns of 11 degrees, a node scaled
        (3, 1, 1) under one scaled (1, 1, 3) 8.2 times its gap at scale 1,
        where the largest axis of its world 3x3 is 3.1 and the product of the
        two largest axes 9; a locator 1.9 away up to 3.6 off, one 5.6 away
        10.6, and one 0.1 away under a group scaled (1, 0.1, 1) up to 10 times
        that distance, as much with a group turned 45 degrees between them,
        where that 3x3's rows give 2.8; with rotate pivots 0.5 to 7.1 away, the
        node itself up to 1.99 times that distance off (3.94 with a pivot 2
        away) and 3.96 times it under a group scaled (1, 0.5, 2), and a node on
        the pivot within 5.9e-14. From another rate Maya keys such a node at
        that rate (:meth:`ClipData.load_fbx`), and on the file's frames between
        those keys it parts from the read by the gap above and by a share of
        its own, the two adding or partly cancelling: that share grows with how
        far the angles Maya solves bend between that rate's frames, not with
        the gap, and is largest where they jump, as they do by 180 degrees
        where the node's turn passes y = 90 or -90 in xyz. On the files
        measured, on nodes of world scale 1, from 24, 25 and 48 fps scenes for
        a 30 fps file and 30 and 25 fps ones for a 24 fps file: a smooth turn
        about one axis, which agrees at the file's rate, 1.0e-2 to 6.1e-2
        apart; keys that turn back each frame, 4.1e-6 apart at the file's rate
        on a 0.14 degree turn and 3.1e-4 to 6.4e-4 from those scenes, 4.2e-4
        and 3.2e-3 to 6.6e-3 on a 1.4 degree one, 1.1e-2 and 1.6e-2 to 3.9e-2
        on a 7 degree one; keys linear in time, at their largest within 2
        percent of the gap at the file's rate, but Maya's turn parting from its
        own at the file's rate by up to 1.3e-3 on turns up to 45 degrees,
        4.3e-2 on turns to 111 and 1.9 on frames where the angles it solves
        jump. A scale multiplies the share as it does the gap: from a 25 fps
        scene for a 24 fps file, the keys whose share is 1.85 at scale 1 have
        one of 7.42 on a node scaled (1, 2, 3) under one scaled (2, 2, 2). On a
        file whose up axis is x or z a clip parts from Maya's import also where
        a layer blends the rotation of a root that is no joint, and in a file
        of any axes where Maya's import keys what it solves or merges at
        another frame rate than the file's (:meth:`ClipData.load_fbx`). Maya's
        importer leaves a file whose up axis is y, pointing up or down, in its
        own axes, so a y-up file whose front is not +z, or a -y-up one, loads
        here turned 90 or 180 degrees from where Maya shows it. The turn comes
        from the file's ``FbxAxisSystem`` (its up vector, front vector and
        handedness) and goes on each root the read keeps, the way ConvertScene
        puts it on the scene root's children: a joint root takes it in its
        joint orient and its translate, keeping its rotate, any other root has
        its rotate solved again in its rotate order and its translate turned.
        An Unreal file, Z-up, turns (x, y, z) to (x, z, -y). A file already in
        Maya's Y-up, such as one save_fbx wrote from a default read, loads as
        it is. A left handed file raises ValueError, since no rotation turns it
        into Maya's axes. ``axis_system=None`` keeps the file's own axes. Which
        way the character faces is left as the file has it.

        ``pose`` "node", the default, reads the nodes' own values. "bind"
        reads the skin bind pose a skinned file stores, which the node
        values can differ from: they hold whatever pose the scene was in at
        export. Every node the read keeps that a bind pose (an ``FbxPose``
        whose ``IsBindPose()`` is true) lists takes that pose's matrix as its
        world matrix, in the unit and the axes above; a node no bind pose
        lists keeps its own values under its parent, so it follows the
        parent's bind. The world matrices are written parents first, each
        solved into scale, rotate (in the node's rotate order) and
        translate, the joint orient and rotate axis kept, and a node its
        bind leaves where it is keeps its own values: a file at its bind
        pose reads the same either way. A pose's matrices are world
        matrices, but for one the pose holds local to the parent
        (``IsLocalMatrix``), which is composed with the parent's world as
        the read gives it. When two bind poses list one node the first wins,
        and a later one more than 1e-6 off it warns, naming the node. A file
        with no bind pose listing a node the read keeps, or one above a kept
        node (which the read drops, ``node_types`` below), raises
        ValueError.
        ``TransformList.load_fbx``, whose nodes no rig owns, solves each
        one's channels under its parent all the same, so a ``HierarchyData``
        of them reads at the bind ``HierarchyData.load_fbx`` gives.

        ``node_types`` None, the default, keeps the kinds of node the read
        has always kept: joints, locators (an ``eMarker``, or a null drawn as
        a cross), groups (any other null) and meshes, a mesh read as a
        transform. A tuple of those names, "joint", "locator", "transform"
        and "mesh", keeps those kinds alone: ``("joint",)`` reads a skeleton
        without its meshes and groups. A node of a kind left out, one of a
        kind the read can't keep (a camera, a light, an LOD group), one with
        no node attribute and the scene root are dropped, and each node kept
        goes under its nearest kept ancestor. A string, or anything that is
        no collection of names, raises TypeError; an unknown name, or no
        name at all, raises ValueError.

        Every kept node's world matrix is the file's all the same: what the
        nodes dropped between it and its kept parent hold is folded into its
        channels, on every frame for a clip, the way the turn above goes on
        a root. A joint takes their turn in its joint orient and their move
        in its translate, keeping its rotate (but for the uneven scale
        below); any other node has its rotate solved again in its rotate
        order. A uniform scale among them goes on the node's scale, and on
        the scale of each joint under it that compensates its parent's scale
        (segment scale compensate), which a joint compensating a dropped
        parent divides out instead. An uneven one under a node turned in it
        would take shear, which no node holds: that node's position is the
        file's, its axes squared, its rotate solved again from the matrix,
        a joint's too, and the length of each of its axes is the file's
        where its kept parent's world scales evenly, as under a joint that
        compensates the scale. The shear it drops goes on the kept nodes
        under it, so they stay where the file puts them, under a scale
        under 1e-12 too: a joint that compensates its scale takes it in its
        channels, split again, and any other node keeps the rotate and
        scale it holds without it, the file's where nothing else moves it,
        takes the shear in its translate alone and hands it on. Their
        positions are the file's, but on a frame the node is scaled exactly
        0 along an axis, which leaves the shear no inverse; their axes are
        the split's, or their own under the parent the read gives them. A
        scale under 1e-12 in size on a dropped node is read as 1e-12, as
        Maya's evaluation holds a scale: a joint that compensates a dropped
        node scaled to 0 keeps its axes and sits on that node's position,
        and a node that does not collapses onto it, its axes 1e-12 long,
        the 1e-12 in its scale, where Maya 2025 puts them; the sdk's own
        evaluation gives NaN there. A kept node's own scale keeps the
        file's value, 0 included, which cgmath's evaluation divides by as
        it is: ``world_matrix`` is NaN on a joint that compensates a kept
        node scaled to 0 and on every node under it, its channels finite,
        where Maya keeps that joint's axes. A joint that compensates a kept
        node, kept or dropped, divides out that node's own scale, as
        ``world_matrix`` multiplies by it, and a kept one keeps the scale
        of a node dropped above that kept node, as Maya does: a joint under
        a locator scaled 5e-13 under a dropped node scaled 2 takes that 2
        in its scale. So the joints and nodes kept under a node scaled
        under 1e-12 sit where Maya puts them, down to about 1e-154, under
        which the read overflows; along an axis it is scaled exactly 0,
        those under a joint the read drops collapse onto it: Maya holds
        1e-12 there, which a kept node does not. Under a kept node the read
        scales under 1e-12, its own scale times what its split takes from
        the uneven dropped nodes above it (a locator scaled 1.5e-11 under a
        dropped node scaled 0.05 reads 7.5e-13), the translates that place
        the nodes below it can reach the inverse of that scale, which
        ``world_matrix`` multiplies back but Maya, holding 1e-12, does not:
        such a frame written with ``save_fbx`` imports into Maya away from
        where Maya imports the file (a locator under a kept locator split
        under (1, 1, -1e-13): its translate 1.6e12, 1.34 off; under that
        locator read 7.5e-13, 0.059 off). Folded in the same way is
        what a node's own channels can't hold: its rotate and scale pivots
        and offsets, in its translate, rotation active or not, and, on a
        node that is no joint, its pre and post rotation, which the sdk
        reads only with rotation active, in its rotate (a joint holds them
        as its joint orient and rotate axis). A node whose rotation is
        inactive turns in xyz, as the sdk turns it, its rotate_order 0
        whatever order the file stores, and its pre and post rotation left
        out. Maya's importer turns such a node by neither order, keeping the
        stored order on the xyz angles of that order's turn, but on a root
        of a file whose up axis is x or z, whose rotation the sdk's
        ConvertScene turns on in xyz; Maya's exporter writes every node with
        rotation active, as save_fbx does. A Maya group whose pivot is
        centred on its children loads where Maya shows it. A light's post
        rotation is fbx's view axis, which Maya places its children without,
        and so does the read. A camera's turns its children by what it adds
        to fbx's view axis, (0, -90, 0), the post rotation Maya's exporter
        writes, which Maya's importer keeps as the camera's rotate axis:
        nothing on Maya's own camera, (0, -90, 0) on one with no post
        rotation. Maya's importer reads a camera's or a light's pre rotation
        and rotate order, and a camera's post rotation, whether its rotation
        is active or not, and so does the read, but on a root of a file whose
        up axis is x or z with its rotation inactive: the sdk's ConvertScene,
        which turns that file into Maya's Y-up for Maya's importer, drops
        them and turns that root in xyz, so such a camera turns its children
        by (0, -90, 0) whatever its post rotation.
        The sdk's own evaluation puts those children elsewhere. A node
        nothing moves keeps the file's values to the bit, so a file whose
        dropped nodes, pivots and offsets hold the identity, as Unreal's and
        Maya's skeleton exports do, reads as it did. Under ``pose="bind"`` a
        node the read drops follows the bind rule above as a kept one would:
        it is folded in at its bind when a bind pose lists it, so the nodes
        under it follow that bind, and at its own values otherwise. A bind
        read that keeps fewer kinds puts each node it keeps where a full bind
        read puts it.

        A node's inherit type is taken on a joint alone, beyond a unit root's
        scale: eInheritRrs, which leaves the parent's own scale out, as segment
        scale compensate, where the sdk and Maya's importer put it. Every other
        node is read as eInheritRSrs, whatever the file stores: one that is no
        joint stored eInheritRrSs, which scales it along its own axes by its
        parent's world scale where eInheritRSrs shears it, is off where that
        scale is uneven, and one stored eInheritRrs wherever its parent's own
        scale is other than 1, even or not, the nodes under them with them. A
        unit root's scale counts as 1 there: the read divides its 1/s out of
        every node under it stored eInheritRrs, joint or not, so one directly
        under it is where the sdk puts it. A node turned (20, 40, 60) and
        scaled (1, 1.5, 0.75) under a group turned (0, 30, 0) and scaled
        (1, 2, 3) is 1.11 off the sdk in the 3x3 either way, a locator at
        (0.5, 1, -2) under it 2.34 and 1.75 in translate; under one scaled
        (2, 2, 2) the eInheritRrSs node is within 8.9e-16, the eInheritRrs one
        0.99 off, its locator 2.03. Maya's importer puts both where the sdk
        does, within 1.1e-15 in the 3x3 and 2.3e-15 in translate, baking the
        inherit type into the node's own scale and shear. A joint stored
        eInheritRrSs is read as Maya's importer reads it, as eInheritRSrs, and
        is as far off the sdk as such a group. On the files measured Maya's
        export writes eInheritRrs on joints alone, and every eInheritRrSs node,
        Unreal's, sits under scales of 1, where the read is right.

        The rig records what the read produced, which :meth:`save_fbx`
        labels a file with: :attr:`HierarchyData.axis_system`, "maya_y_up",
        or with ``axis_system=None`` the file's own axes in words ("z up, -y
        front, right handed" for an Unreal file), and
        :attr:`HierarchyData.unit`, 1.0 for cm, or with a number for
        ``scale_factor`` the file's unit in cm over that number (100.0, meters,
        for a cm file read with 0.01). ``scale_factor`` must be None or a
        positive number, or ValueError is raised. The view
        ``TransformList.load_fbx`` gives, whose nodes no rig owns, holds the
        record as well (:attr:`axis_system`, :attr:`unit`), and so does a
        view taken from it, a slice or an index of it, its :meth:`match`,
        :meth:`get_roots` or ``TransformList(view)``; a
        :class:`HierarchyData` built from either takes it.

        User defined attributes come back as the specs save_fbx wrote; those
        of a file from Maya's own FBX plugin, the way its importer reads them.
        One whose name, string or enum field is not UTF-8, which the fbx
        binding can't read, is left out, and so is an enum from another tool
        with a field an enumName can't hold (one holding ":", or reading as a
        field and its index, "k=2" or "k==2"), with a warning.
        """
        if FBX is None:
            raise ImportError("fbx module not found")
        _fbx_check_scale_factor(scale_factor)
        _fbx_check_axis_system(axis_system)
        _fbx_check_pose(pose)
        kinds = _fbx_check_node_types(node_types)

        with open_fbx(filename, FBX) as scene:
            turn   = _fbx_axis_turn(scene, axis_system, filename)
            units  = _fbx_units(scene, scale_factor)
            record = _fbx_record(scene, axis_system, scale_factor)
            hierarchy_data, _, tree = _walk_fbx(scene, units, kinds, turn is not None)
            listed = _fbx_bind_matrices(scene, tree, units, filename) if pose == "bind" else None

        # each kept node's bind, a dropped node a bind pose lists folded in
        # at its bind, in the file's axes
        binds = None if listed is None else _fbx_kept_binds(tree, listed, list(hierarchy_data))

        if turn is not None:
            _fbx_turn_walk(hierarchy_data, turn)

        # build the TransformList
        if binds is None:
            return _recorded(cls.from_dict(hierarchy_data), record)

        # a bind is a world matrix, solved under the node's parent, which a
        # view's nodes, owned by no rig, can't find: they are posed in a rig
        # that owns them, and the view holds copies, owned by none as a read
        # gives them. One load of nodes, posed inside it: a bind that can't
        # be solved under its parent's, one a zero scale flattens, raises,
        # and the nodes warn of nothing
        owns = issubclass(cls, HierarchyData)
        with _NodesLoad():
            rig = (cls if owns else HierarchyData).from_dict(hierarchy_data)
            _fbx_pose_binds(rig, binds, turn)
        return _recorded(rig if owns else cls(node.copy() for node in rig), record)

    def save_fbx(self, filename, zero_root=False, as_ascii=False):
        """saves the data to a .fbx file, binary unless as_ascii is set

        Each node is written after its parent, whatever order the list holds
        them in. A node whose parent the file does not hold -- left out of a
        view, or a parent_uuid no node carries -- is a root of the file, as a
        view's copy() roots it; siblings, the roots among them, keep the
        list's order. zero_root writes the first joint the file holds as a
        root with an identity transform.

        The file is labelled with the axis system and the unit the values
        are in, as the rig records them (:attr:`axis_system`, :attr:`unit`;
        a view's are its rig's), the unit set explicitly, and holds the
        values as they are. A rig :meth:`load_fbx` read with its defaults is
        saved in Maya's Y-up and cm, one read with ``axis_system=None`` in
        the file's own axes, and one read with a number for ``scale_factor``
        in the unit that number leaves its lengths in (a cm file read with
        0.01, in meters). So a save loads with :meth:`load_fbx`'s defaults
        where the file the rig was read from does, in cm and Maya's Y-up,
        read as it is (``axis_system=None, scale_factor=1.0``) gives the
        values back, and Maya's FBXImport, and the sdk's ``ConvertScene`` to
        Maya's Y-up and cm, put it where they put that file. Raises
        ValueError, before anything is written, for a view holding nodes of
        rigs that record other axes or units, and for :meth:`load_fbx`'s
        view, or one taken from it, holding nodes no rig owns beside nodes
        of a rig that records another.

        Where the read and Maya part, they part on the save too. A file
        whose up axis is y and whose front is not +z, which the read turns
        into Maya's Y-up and Maya's importer leaves in its own axes, saved
        from a default read imports as the read turned it. A save in another
        unit than cm holds its lengths in that unit, as a file authored in
        it does, which Maya's importer and the sdk's cm conversion bring in
        with its top nodes scaled by the unit (100 for meters), at the
        positions the values say; :meth:`load_fbx` reads it at scale 1, in
        cm. Maya's importer, as the sdk's ConvertScene, converts a left
        handed file on its top nodes through their translate and pre
        rotation alone, so a top joint the read folded a dropped node's turn
        into, as its joint orient, written as its pre rotation, imports
        elsewhere than the file it was read from does (20 cm off on a
        fixture), the other nodes where they did. And a file holding its
        unit as a scale on its top nodes, read with a number other than 1.0
        for ``scale_factor``, keeps that scale, 1/s, which the unit it
        records, s over the number, does not explain: its save loads and
        imports off under those nodes.

        No geometry is written: a mesh's node, which :meth:`load_fbx` reads
        as a transform, is written as a group at its node's transform. A
        mesh holds no record of its own. The fbx mesh readers keep its
        points in its node's object space, which no axis system changes, in
        the unit their ``scale_factor`` gives, the unit a rig read with the
        same ``scale_factor`` records: under its node's world from that rig,
        or from Maya's import of the rig's save, it is where its file puts
        it. Every joint is written as a limb node, as Maya's exporter writes
        them: Maya's importer brings a root written as a skeleton root
        (``eRoot``) with no child in as a plain transform, without its
        joint orient.

        rotate_axis and joint_orient are written for joints only. A
        transform or locator holding a rotate_axis (a matrix write or
        set_parent() keeps it) loses it in the file; fold it into rotate
        first by writing node.matrix back with rotate_axis zeroed. Every
        node is written with rotation active, as Maya's exporter writes it:
        fbx turns a node without it in xyz, whatever its rotate order. A
        file an earlier save_fbx wrote holds a group or a locator with its
        rotation inactive, so one in another order than xyz now loads where
        the sdk puts it, in xyz, rotate_order 0, not where it was saved (a
        zxy group 0.49 off, a joint under it 1.26): load it with the version
        that wrote it, save() it, then load that and save_fbx it with this
        one.

        User defined numbers (double, float, doubleAngle, doubleLinear, long,
        short, byte, char), bools, enums, strings, stringArrays and double3 or
        float3 vectors of three numbers come back from load_fbx as they went
        in, binary or ASCII. Any other (a doubleArray or other array, a
        matrix, a multi, another compound) is skipped with a logged warning,
        the children of a skipped compound written on their own. Raises
        ValueError, before anything is written, when a string attribute's
        value is not plain ASCII, the only text the fbx sdk takes. In an ASCII
        file a string or enum field list over 4096 bytes holding a quote, line
        break, tab, semicolon, brace, & or run of spaces is %-escaped, and
        Maya reads the escaped text: the fbx sdk's ASCII reader changes it
        otherwise, or loses the whole scene for a brace.
        """

        if FbxExporter is None:
            raise ImportError("fbx module not found")

        # expand ~, consistent with the other IO paths
        filename = os.path.expanduser(filename)

        with FbxExporter() as exporter:
            exporter.add_skeleton(self)

            exporter.export(
                path      = filename,
                as_ascii  = as_ascii,
                zero_root = zero_root,
            )

    # --------- vectorized properties and methods --------- #
    def set_rotate_to_joint_orient(self) -> None:
        """sets rotate to 0 and applies all rotation to joint_orient

        On a clip every node and frame is solved at once.
        """
        self._orient(into_rotate=False)

    def set_joint_orient_to_rotate(self) -> None:
        """sets joint_orient to 0 and applies all rotation to rotate

        On a clip every node and frame is solved at once.
        """
        self._orient(into_rotate=True)

    def _orient(self, into_rotate: bool) -> None:
        for owner, nodes in _by_owner(self[:]):
            frames = _clip_frames(owner)
            if frames is not None:
                _orient_rows(frames, frames.rows_of(nodes), into_rotate)
            elif into_rotate:
                for node in nodes:
                    node.set_joint_orient_to_rotate()
            else:
                for node in nodes:
                    node.set_rotate_to_joint_orient()

    def swapaxes(self, axis0: int, axis1: int, negate: bool = False) -> None:
        """swaps two local axes on every node in the view

        every node is solved at once, over every frame of a clip: each swapped
        node's world becomes the permutation times what it was and each of its
        children keeps its world, so the result does not depend on the order
        the view happens to be in. The result is the per node method's run
        parent first, the exception it makes included: under a parent chain
        that mirrors an axis other than z, a restored child's scale signs
        move, and its segment scale compensated children with them.
        """
        _axis_permutation(axis0, axis1, negate)  # bad axes raise before anything moves
        for owner, nodes in _by_owner(self[:]):
            if owner is None:
                for node in nodes:
                    node.swapaxes(axis0, axis1, negate=negate)
                continue

            frames = _RigFrames(owner)
            _swap_rows(frames, frames.rows_of(nodes), axis0, axis1, negate)

    def set_rotate_order(self, order) -> None:
        """
        Changes every node's rotate order and keeps the pose.

        ``order`` is one order for every node or one per node, each 0..5 or
        the axes as a string, eg: ``"zyx"``. Every node is solved at once,
        over every frame of a clip; see :meth:`TransformData.set_rotate_order`.
        """
        if _is_sequence(order):
            orders = [_rotate_order_index(o) for o in order]
            if len(orders) != len(self):
                raise ValueError(f"got {len(orders)} rotate orders for {len(self)} nodes")
        else:
            orders = [_rotate_order_index(order)] * len(self)

        wanted = {id(node): o for node, o in zip(self, orders)}
        for owner, nodes in _by_owner(self[:]):
            if owner is None:
                for node in nodes:
                    node.set_rotate_order(wanted[id(node)])
                continue

            frames = _RigFrames(owner)
            rows   = frames.rows_of(nodes)
            _reorder_rows(frames, rows, [wanted[id(n)] for n in nodes])

    def get_roots(self) -> "TransformList":
        """returns all contained nodes parented to the world"""
        lookups = _Lookups()
        return self._view([obj for obj in self if lookups.parent(obj) is None])

    def get_parents(self) -> np.ndarray:
        """returns each node's parent index in its owning hierarchy, -1 for none

        A parent_uuid no node of the hierarchy carries gives -1 too, as
        get_parent_matrix reads such a node as a root.
        """
        lookups = _Lookups()
        parents = []
        for node in self:
            parent = lookups.parent(node)
            if isinstance(parent, TransformData):
                parents.append(lookups.index(parent))
            else:
                parents.append(-1)
        return np.array(parents, dtype=int)

    def set_parent(
        self,
        parent:      Union["TransformData", str, None],
        world_space: bool                              = True,
    ) -> None:
        """sets all the nodes under a given parent

        ``parent`` is a node, its uuid or a name, as TransformData.set_parent
        takes it, and each node keeps the parent's uuid. The result is the
        one a node at a time gives: every node to the world, then each under
        ``parent``, resolved once the first pass has moved the nodes, the
        parent itself left at the world when it is one of them. Each node's
        branch moves to the end of the list, where its last move puts it. A
        parent that hangs below one of the nodes raises ValueError before
        anything moves.

        Nodes of one hierarchy are done together, in one pass over it. In
        world space every node keeps its world matrix, solved for every node
        at once, over every frame of a clip, and each rotate curve the solve
        rebuilds goes through euler_filter, frame 0 kept; a node that already
        held its world on every frame keeps its keys as they are. Where
        solving one node would move a later one before its turn -- a shear
        its channels can not hold, the matrix setter's keep test holding a
        node's channels, or a new scale reaching a segment scale compensated
        child -- the nodes are solved one at a time instead, each over every
        frame at once, the list still reordered once.
        """
        nodes  = self[:]
        owners = _by_owner(nodes)
        if len(owners) == 1 and isinstance(owners[0][0], HierarchyData):
            _reparent(owners[0][0], list(nodes), parent, world_space)
            return

        # nodes of several hierarchies, or of none: one at a time
        for node in nodes:
            node.set_parent(None, world_space=world_space)

        # parent to new node if parent is not None
        if parent is not None:
            for node in nodes:
                # ignore self
                if node != parent:
                    node.set_parent(parent, world_space=world_space)

    def add_prefix(self, prefix: str) -> None:
        """adds a prefix to the node's name"""
        for node in self.list:
            node.add_prefix(prefix)

    def add_suffix(self, suffix: str) -> None:
        """adds a suffix to the node's name"""
        for node in self.list:
            node.add_suffix(suffix)

    # --- user defined attributes, as properties --- #

    def __getattr__(self, name: str):
        """a user defined attribute read on every node of the view, as
        ``h.heroHeight``: one value per node, None where a node has none"""
        nodes = None if name.startswith("_") else self.__dict__.get("list")
        if nodes and any(_has_user_attr(node, name) for node in nodes):
            return [
                _user_value(node.user_defined_attributes, name)
                if _has_user_attr(node, name)
                else None
                for node in nodes
            ]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value) -> None:
        """sets a user defined attribute on every node of the view that has it,
        as ``h.heroHeight = 1.8``: every value is converted before any is
        written. A name the class declares is set on the list, as
        DataList.__setattr__ sets it; any other the class has -- a method, a
        class setting, a read-only property -- or a private one raises, and
        so does a name no node has, instead of landing on the list."""
        cls      = type(self)
        writable = cls.__dict__.get("_WRITABLE_CACHE")
        if writable is None:
            writable = cls._writable()
        if name in writable:
            object.__setattr__(self, name, value)
            return

        # a user attribute is public, and never a name the class has; such a
        # name is looked for once more on the class as it is now
        if name.startswith("_") or _class_attribute(cls, name) is not _NO_DEFAULT:
            if name not in cls._writable(fresh=True):
                raise _refusal(cls, name)
            object.__setattr__(self, name, value)
            return

        nodes = [n for n in self.__dict__.get("list", []) if _has_user_attr(n, name)]
        if not nodes:
            raise AttributeError(
                f"{type(self).__name__!r} has no attribute {name!r} and no node has "
                f"a user attribute of that name; add_user_attribute() creates one"
            )
        writes = []
        for node in nodes:
            attrs = node.user_defined_attributes
            writes.extend(_user_writes(attrs, name, value, node.name))
        for spec, converted in writes:
            spec["value"] = converted

    def __dir__(self):
        names = set(super().__dir__())
        for node in self.__dict__.get("list", []):
            names.update(node.user_defined_attributes)
        return sorted(names)

    def add_user_attribute(
        self,
        name:           str,
        value:          Any,
        attribute_type: Optional[str] = None,
        keyable:        bool          = True,
        enum_names:     Optional[str] = None,
    ) -> None:
        """add_user_attribute() on every node of the view, all or nothing"""
        specs = []
        for node in self:
            node._check_new_user_attr(name)
            label = f"{node.name}.{name}"
            spec  = _new_user_attr(value, attribute_type, keyable, enum_names, label)
            specs.append((node, spec))
        for node, spec in specs:
            node.user_defined_attributes[name] = spec

    def _parent_first(self) -> List[int]:
        """view indices ordered so an ancestor is always written before a child

        setters that solve against the parent's world matrix -- world_matrix,
        matrix and the match_* methods -- give a different answer depending on
        write order: a child written before its parent solves against the old
        parent and is then dragged when the parent moves. depth is measured in
        the owning hierarchy, so a view that skips levels still orders right.
        """
        return TransformList._owning_parent_first(self)

    @staticmethod
    def _owning_parent_first(nodes, within: bool = False) -> List[int]:
        """:meth:`_parent_first` for any sequence of nodes, eg: a pairing

        Depth counts every ancestor in the owning hierarchy, the ones left
        out of ``nodes`` too. Each node met on the way up is resolved once,
        through one index of the rig, into one parent array over all of
        them, so the cost is a pass over the rig rather than a walk to the
        root from every node. ``within``: only the ancestors ``nodes`` holds
        count, a node whose parent it leaves out being a root, as in a file
        written from them: their roots keep the order ``nodes`` lists them in.
        """
        lookups = _Lookups()
        met, place = [], {}

        def slot(node) -> int:
            if id(node) not in place:
                place[id(node)] = len(met)
                met.append(node)
            return place[id(node)]

        rows    = [slot(node) for node in nodes]
        parents = []
        while len(parents) < len(met):
            # a parent outside the hierarchy resolves to a string, which
            # ends the chain; _chain_depths guards a cyclic parent_uuid
            parent = lookups.parent(met[len(parents)])
            if not isinstance(parent, TransformData):
                parents.append(-1)
            elif within:
                parents.append(place.get(id(parent), -1))
            else:
                parents.append(slot(parent))

        depth = _chain_depths(parents)[rows]
        return list(np.argsort(depth, kind="stable"))

    def _match_operands(self, other, ndim: int) -> list:
        """resolves a match_* operand into one value per node in the view

        ``ndim`` is the dimension a per-node sequence would have, so a value
        one dimension shallower broadcasts to every node instead.
        """
        if isinstance(other, TransformData):
            return [other] * len(self)

        if isinstance(other, (TransformList, list, tuple)) and all(
            isinstance(x, TransformData) for x in other
        ):
            if len(other) != len(self):
                raise ValueError(
                    f"expected {len(self)} nodes to match against, got {len(other)}"
                )
            return list(other)

        if _is_sequence(other, ndim=ndim):
            if len(other) != len(self):
                raise ValueError(
                    f"expected {len(self)} values to match against, got {len(other)}"
                )
            return list(np.asarray(other))

        return [other] * len(self)

    def get_children(self) -> "TransformList":
        """returns the children of every node in the view, without duplicates"""
        lookups = _Lookups()
        children, seen = [], set()
        for node in self:
            for child in lookups.children(node):
                if child.uuid not in seen:
                    seen.add(child.uuid)
                    children.append(child)

        return self._view(children)

    def get_branch(self) -> "TransformList":
        """returns every node in the view and all of their decendents

        Each node's branch in the order its own get_branch() lists it, parent
        before children. A node already walked is not walked again, which
        also ends a cyclic parent_uuid.
        """
        branch, seen = [], set()
        for member in _walk(self, _Lookups()):
            if member.uuid not in seen:
                seen.add(member.uuid)
                branch.append(member)

        return self._view(branch)

    def get_parent_matrix(self) -> np.ndarray:
        """returns every node's parent worldspace matrix"""
        return np.array([node.get_parent_matrix() for node in self])

    def to_attributes(self, mapping: dict = MAYA_ATTRIBUTE_MAP) -> List[dict]:
        """returns every node's attributes as a list of dicts

        ``mapping`` is read as TransformData.to_attributes reads it: a field
        named ``"translate"`` or ``"_translate"``. Where SDF primitives sit
        among other nodes, on which ``"radius"`` is the joint radius too,
        ``"_sdf_radius"`` names the primitives' radius.
        """
        return [node.to_attributes(mapping) for node in self]

    def match_translate(
        self, other, x: bool = True, y: bool = True, z: bool = True
    ) -> None:
        """matches the world position of a node, a list of nodes or a translation

        Nodes in other axes or units than these, as their rigs record
        them, raise ValueError before anything moves, as
        :meth:`TransformData.match_translate` says.
        """
        _check_matched(self, other)
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "translate", other, axes=(x, y, z)):
            return

        others = self._match_operands(other, ndim=2)
        for i in self._parent_first():
            self[i].match_translate(others[i], x=x, y=y, z=z)

    def match_rotate(self, other) -> None:
        """matches the world rotation of a node, a list of nodes or a rotation

        Nodes in other axes or units than these, as their rigs record
        them, raise ValueError before anything moves, as
        :meth:`TransformData.match_translate` says.
        """
        _check_matched(self, other)
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "rotate", other):
            return

        others = self._match_operands(other, ndim=2)
        for i in self._parent_first():
            self[i].match_rotate(others[i])

    def match_matrix(self, other) -> None:
        """matches the world matrix of a node or a list of nodes

        Nodes in other axes or units than these, as their rigs record
        them, raise ValueError before anything moves, as
        :meth:`TransformData.match_translate` says.
        """
        _check_matched(self, other)
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "match", other):
            return

        others = self._match_operands(other, ndim=3)
        for i in self._parent_first():
            self[i].match_matrix(others[i])

    @property
    def uuid(self) -> List[str]:
        """returns a list of all node uuids"""
        return [x.uuid for x in self]

    @property
    def indices(self) -> np.ndarray:
        """returns each node's index in its owning hierarchy

        named ``indices`` rather than ``index`` because ``DataList.index`` is
        already the name-lookup method.
        """
        lookups = _Lookups()
        return np.array([lookups.index(node) for node in self])

    @property
    def unique_name(self) -> np.ndarray:
        """whether each node's name is its own in its owning hierarchy"""
        lookups = _Lookups()
        return np.array([lookups.unique_name(node) for node in self])

    @property
    def matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.matrix)
        return np.array(values)

    @matrix.setter
    def matrix(self, M: np.ndarray) -> None:
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "local", M):
            return

        M = np.asarray(M)

        # decomposing a local matrix reads the parent's scale through
        # parent_scale_inverse, so parents have to be written first
        order = self._parent_first()
        if _is_sequence(M, ndim=3):
            for i in order:
                self[i].matrix = M[i]
        else:
            for i in order:
                self[i].matrix = M

    @property
    def world_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.world_matrix)
        return np.array(values)

    @world_matrix.setter
    def world_matrix(self, M: np.ndarray) -> None:
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "world", M):
            return

        M = np.asarray(M)

        # solving a local matrix reads the parent's world matrix, so a child
        # written before its parent lands on the wrong spot
        order = self._parent_first()
        if _is_sequence(M, ndim=3):
            for i in order:
                self[i].world_matrix = M[i]
        else:
            for i in order:
                self[i].world_matrix = M

    @property
    def scale(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.scale)
        return np.array(values)

    @scale.setter
    def scale(self, value: np.ndarray) -> None:
        value = np.asarray(value)

        if _is_sequence(value, ndim=2):
            for i, node in enumerate(self):
                node.scale = value[i]
        else:
            for node in self:
                node.scale = value

    @property
    def rotate(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.rotate)
        return np.array(values)

    @rotate.setter
    def rotate(self, value: np.ndarray) -> None:
        value = np.asarray(value)

        if _is_sequence(value, ndim=2):
            for i, node in enumerate(self):
                node.rotate = value[i]
        else:
            for node in self:
                node.rotate = value

    @property
    def translate(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.translate)
        return np.array(values)

    @translate.setter
    def translate(self, value: np.ndarray) -> None:
        value = np.asarray(value)

        if _is_sequence(value, ndim=2):
            for i, node in enumerate(self):
                node.translate = value[i]
        else:
            for node in self:
                node.translate = value

    @property
    def rotate_order(self) -> np.ndarray:
        """every node's rotate order, 0..5

        Setting it, one order for all or one per node, changes the orders
        alone, as Maya's setAttr does, so the nodes turn.
        :meth:`set_rotate_order` keeps the pose instead.
        """
        values = []
        for node in self:
            values.append(node.rotate_order)
        return np.array(values)

    @rotate_order.setter
    def rotate_order(self, value) -> None:
        if _is_sequence(value):
            for i, node in enumerate(self):
                node.rotate_order = value[i]
        else:
            for node in self:
                node.rotate_order = value

    @property
    def rotate_axes(self) -> List[str]:
        """returns each node's rotate order as a string"""
        return [node.rotate_axes for node in self]

    @property
    def rotate_axis(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.rotate_axis)
        return np.array(values)

    @rotate_axis.setter
    def rotate_axis(self, value: np.ndarray) -> None:
        value = np.asarray(value)

        if _is_sequence(value, ndim=2):
            for i, node in enumerate(self):
                node.rotate_axis = value[i]
        else:
            for node in self:
                node.rotate_axis = value

    @property
    def parent_scale_inverse(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.parent_scale_inverse)
        return np.array(values)

    @property
    def joint_orient(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.joint_orient)
        return np.array(values)

    @joint_orient.setter
    def joint_orient(self, value: np.ndarray) -> None:
        value = np.asarray(value)

        if _is_sequence(value, ndim=2):
            for i, node in enumerate(self):
                node.joint_orient = value[i]
        else:
            for node in self:
                node.joint_orient = value

    @property
    def segment_scale_compensate(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.segment_scale_compensate)
        return np.array(values)

    @segment_scale_compensate.setter
    def segment_scale_compensate(self, value) -> None:
        if _is_sequence(value):
            for i, node in enumerate(self):
                node.segment_scale_compensate = value[i]
        else:
            for node in self:
                node.segment_scale_compensate = value

    @property
    def visibility(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.visibility)
        return np.array(values)

    @visibility.setter
    def visibility(self, value: bool) -> None:
        if _is_sequence(value):
            for i, node in enumerate(self):
                node.visibility = value[i]
        else:
            for node in self:
                node.visibility = value

    @property
    def radius(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.radius)
        return np.array(values)

    @radius.setter
    def radius(self, value: float) -> None:
        if _is_sequence(value):
            for i, node in enumerate(self):
                node.radius = value[i]
        else:
            for node in self:
                node.radius = value

    @property
    def draw_style(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.draw_style)
        return np.array(values)

    @draw_style.setter
    def draw_style(self, value: int) -> None:
        if _is_sequence(value):
            for i, node in enumerate(self):
                node.draw_style = value[i]
        else:
            for node in self:
                node.draw_style = value

    @property
    def node_type(self) -> List[str]:
        values = []
        for node in self:
            values.append(node.node_type)
        return values

    @node_type.setter
    def node_type(self, value) -> None:
        """
        Sets the type of every node, one value for all or one per node.
        Joints that become another type shed their joint-only attributes,
        see ``TransformData.set_node_type``; on a clip their joint_orient
        curves are folded into rotate for every node and frame at once.
        """
        if _is_sequence(value):
            values = [str(v) for v in value]
            if len(values) != len(self):
                raise ValueError(
                    f"got {len(values)} node types for {len(self)} nodes"
                )
        else:
            values = [str(value)] * len(self)

        wanted = {id(node): v for node, v in zip(self, values)}
        folded = set()
        for owner, nodes in _by_owner(self[:]):
            frames = _clip_frames(owner)
            if frames is not None:
                shed = [
                    n for n in nodes if n.node_type == "joint" and wanted[id(n)] != "joint"
                ]
                _fold_orient_rows(frames, frames.rows_of(shed))
                folded.update(id(n) for n in shed)

        for node, node_type in zip(self, values):
            node._set_node_type(node_type, fold=id(node) not in folded)

    @property
    def scale_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.scale_matrix)
        return np.array(values)

    @property
    def rotate_axis_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.rotate_axis_matrix)
        return np.array(values)

    @property
    def rotate_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.rotate_matrix)
        return np.array(values)

    @property
    def quaternion(self) -> np.ndarray:
        # the head of a multi-frame clip reads its frame, every node at once
        clip = _framed_clip(self)
        if clip is not None:
            clip._realign()
            frame = clip.frame
            span  = slice(frame, frame + 1)
            return _span_quaternions(clip, _clip_columns(clip, self), span)[0]

        values = []
        for node in self:
            values.append(node.quaternion)
        return np.array(values)

    @quaternion.setter
    def quaternion(self, Q: np.ndarray) -> None:
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "quaternion", Q):
            return

        Q = np.asarray(Q)

        if _is_sequence(Q, ndim=2):
            for i, node in enumerate(self):
                node.quaternion = Q[i]
        else:
            for node in self:
                node.quaternion = Q

    @property
    def joint_orient_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.joint_orient_matrix)
        return np.array(values)

    @property
    def parent_scale_inverse_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.parent_scale_inverse_matrix)
        return np.array(values)

    @property
    def translate_matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.translate_matrix)
        return np.array(values)

    # --- Maya's parent and inverse matrices, read only --- #
    # each owning rig evaluated once, every node of it a read needs together,
    # as a clip's frames are (_pose_matrices), and every inverse in one call;
    # on a clip, of the frame it is bound to. A node whose parents loop
    # raises the RecursionError its world_matrix raises
    @property
    def parent_matrix(self) -> np.ndarray:
        """every node's parent world matrix, Maya's parentMatrix; read only

        (N, 4, 4), identity for a root.
        """
        return _pose_matrices(self, strict=True)[2]

    @property
    def inverse_matrix(self) -> np.ndarray:
        """every node's inverse local matrix, Maya's inverseMatrix; read only

        (N, 4, 4), a true inverse, as TransformData.inverse_matrix.
        """
        return _inverse(_pose_matrices(self, worlds=False)[0], "matrix", lambda: self.name)

    @property
    def world_inverse_matrix(self) -> np.ndarray:
        """every node's inverse world matrix, Maya's worldInverseMatrix; read only

        (N, 4, 4), a true inverse.
        """
        return _inverse(_pose_matrices(self, strict=True)[1], "world_matrix", lambda: self.name)

    @property
    def parent_inverse_matrix(self) -> np.ndarray:
        """every node's inverse parent matrix, Maya's parentInverseMatrix; read only

        (N, 4, 4), identity for a root, a true inverse.
        """
        return _inverse(_pose_matrices(self, strict=True)[2], "parent_matrix", lambda: self.name)

    # --- per axis channel components --- #
    scale_x  = _component_property("scale",        "x")
    scale_y  = _component_property("scale",        "y")
    scale_z  = _component_property("scale",        "z")

    rotate_x = _component_property("rotate",       "x")
    rotate_y = _component_property("rotate",       "y")
    rotate_z = _component_property("rotate",       "z")

    translate_x = _component_property("translate",    "x")
    translate_y = _component_property("translate",    "y")
    translate_z = _component_property("translate",    "z")

    joint_orient_x = _component_property("joint_orient", "x")
    joint_orient_y = _component_property("joint_orient", "y")
    joint_orient_z = _component_property("joint_orient", "z")

    rotate_axis_x  = _component_property("rotate_axis",  "x")
    rotate_axis_y  = _component_property("rotate_axis",  "y")
    rotate_axis_z  = _component_property("rotate_axis",  "z")

    # --- pose algebra ----------------------------------------------------- #
    # `a - b` captures how `a`'s joint frames differ from `b`'s, such that
    # `(a - b) + b == a`. The delta holds a per-joint frame offset rather than
    # a local rotation difference, so `+` can re-apply it to ANY pose of the
    # source rig: a local difference is only valid at the pose it was measured
    # in, which makes it useless for converting a posed rig onto a skeleton
    # whose joint axes differ. Segment-scale-compensated scale is handled so
    # that addition is the exact inverse of subtraction.
    #
    # `by="index"` is the default: joints pair on position, names are ignored,
    # and both sides must hold the same count, order and parenting. That is
    # the usual case -- the same skeleton twice -- and it composes with
    # slicing. `by="name"` opts in to pairing on the intersection of the two
    # name sets, which is what lets a delta cover only PART of a skeleton (eg:
    # an upper body against a full rig).
    #
    # Neither pairing is safe in every case. Index pairing accepts symmetric
    # limbs exported in the opposite sibling order -- both arms share a parent
    # index, so the structure check passes and left pairs to right. Name
    # pairing only raises when NOTHING matches: a partial overlap is accepted
    # silently, converting the joints it recognises and passing the rest
    # through untouched.
    #
    # Mind the receiver: `get_delta` is called on the DESTINATION rig, and
    # `delta + rig` is spelled `rig.add_delta(delta)` -- the operator only
    # ever reads the delta from its LEFT operand, so `rig + delta` silently
    # computes something else.
    #
    # This lives on TransformList, not HierarchyData, so a delta can be taken
    # over any selection -- `rig.match("L_*")`, `rig[1:]`, `node.get_branch()`.
    # A delta must own its nodes, so `get_delta` always returns a HierarchyData
    # whatever the receiver was -- a ClipData of every frame when the other
    # side is a clip of several frames, as `add_delta` returns for a clip of
    # deltas.
    #
    # `translate=False` transfers orientation only: the delta keeps its own
    # bone offsets rather than a position difference, so the result holds the
    # source rig's proportions at any pose. The SAME value must go to both
    # calls -- the two modes store different things in the one translate
    # channel, and mixing them silently produces nonsense. It cannot be
    # inferred at apply time, which is why it is not a property of the delta.
    #
    # Two limits are inherent to carrying a delta in a hierarchy: scale is an
    # additive delta and is NOT permuted by the frame offset (`F * S * inv(F)`
    # is diagonal only when `F` is an axis permutation), so a non-unit scale
    # under a rotated frame is approximate; and a matched node whose parent is
    # absent from the subset is assumed to already be in the target frame.

    def copy(self) -> "TransformList":
        """returns a deep copy, owning the nodes it holds.

        A view owns nothing: it points every node's ``_hierarchy`` back at the
        real hierarchy, which is what lets ``get_parent``, ``world_matrix``,
        ``index`` and ``unique_name`` keep answering relative to the whole rig
        when a node is reached through a selection.

        A copy has no rig to point at. Its nodes are new objects that no
        hierarchy holds, so leaving them unowned does not preserve the link --
        there is no link -- it drops every one of those queries on the floor,
        silently: ``get_parent`` returns None for the whole copy and a caller
        rebuilding from it gets a flat list. A copy therefore has to own what
        it holds, the same reason ``get_delta`` hands back a
        :class:`HierarchyData` when it is called on a view. An object two
        nodes hold, a mesh two Objects instance, is one object in the copy
        too, as in any copy().

        Subclasses already own their nodes and keep their own class. A copy
        warns of nothing its load would: the original's load did.
        """
        if type(self) is TransformList:
            return self._owned_copy()

        with _Copying():
            return super().copy()

    def __copy__(self):
        """
        A view: a new view over the same nodes. A hierarchy owns its nodes,
        and a node belongs to one hierarchy, so a copy that shared them would
        break the other on its first edit: it gets copy()'s own nodes.
        """
        if type(self) is TransformList:
            return super().__copy__()
        return self.copy()

    def __deepcopy__(self, memo):
        # a view copies into a hierarchy of its own, as copy() explains
        if type(self) is TransformList:
            new            = self._owned_copy()
            memo[id(self)] = new
            return new
        return super().__deepcopy__(memo)

    def _owned_copy(self) -> "HierarchyData":
        """Copy these nodes into an owning hierarchy of their own.

        A copied node keeps its ``parent_uuid``. When that parent is not part
        of the copy -- a selection, a slice, a partial rig -- the reference
        would dangle: the copy joins its hierarchy as any node does, a root
        of the result (:meth:`HierarchyData._join`).

        Nodes that share nothing are copied one by one. When two of them hold
        one object, a mesh two Objects instance, they are copied together, as
        one save and load, so that object is one object in the copy too.
        """
        if _reached_twice(self):
            return self._owned([copy.copy(node) for node in self]).copy()
        return self._owned([node.copy() for node in self])

    def _owned(self, nodes: list) -> "HierarchyData":
        """
        ``nodes``, copies of these, in an owning hierarchy of their own, a
        node whose parent is not among them made a root, as any node joining
        a hierarchy is. A shallow copy is over the values its node holds: the
        nodes themselves keep their owner and parent, and an object they
        share is still one object. It records the axes and unit these record
        (:meth:`_records`), taken before the copies join it.
        """
        record = self._records()
        rig    = HierarchyData(nodes)
        rig._axis_system, rig._unit = record
        return rig

    def _matched_subset(self, other, by: str = "index"):
        """Return ``(nodes_self, nodes_other, parents)`` for paired nodes.

        ``parents`` holds, for each paired node, the index of its parent
        *within the pairing* (-1 when the parent is a root or is absent from
        it), so the scale math indexes the subset arrays correctly.
        """
        if by == "name":
            return self._pair_by_name(other)
        if by == "index":
            return self._pair_by_index(other)
        raise ValueError(f"unknown pairing {by!r}, expected 'name' or 'index'")

    def _pair_by_name(self, other):
        """Pair nodes on the intersection of their names.

        Only nodes present in both hierarchies take part, which is what lets
        a delta cover part of a skeleton -- eg: an upper body against a full
        rig. Raises on duplicate names or an empty intersection rather than
        silently producing garbage.
        """
        self_index = {}
        for i, node in enumerate(self):
            if node.name in self_index:
                raise ValueError(f"duplicate node name in self: {node.name!r}")
            self_index[node.name] = i

        other_index = {}
        for j, node in enumerate(other):
            if node.name in other_index:
                raise ValueError(f"duplicate node name in other: {node.name!r}")
            other_index[node.name] = j

        matched = [node.name for node in self if node.name in other_index]
        if not matched:
            raise ValueError("hierarchies share no matching node names")

        nodes_self  = [self[self_index[name]] for name in matched]
        nodes_other = [other[other_index[name]] for name in matched]

        subset_pos  = {name: k for k, name in enumerate(matched)}
        parents     = np.full(len(matched), -1, dtype=int)
        lookups     = _Lookups()
        for k, node in enumerate(nodes_self):
            parent = lookups.parent(node)
            if parent is not None and not isinstance(parent, str):
                parents[k] = subset_pos.get(parent.name, -1)

        return nodes_self, nodes_other, parents

    @staticmethod
    def _parent_indices(nodes) -> np.ndarray:
        """Parent of each node as a position within ``nodes``, -1 for a root."""

        # positions within THIS pairing, never `node.index`: that is the
        # position in the OWNING hierarchy, which for a selection or a
        # partial rig points at the wrong node -- or at the node itself
        position = {node.uuid: i for i, node in enumerate(nodes)}
        indices  = np.full(len(nodes), -1, dtype=int)
        lookups  = _Lookups()
        for i, node in enumerate(nodes):
            parent = lookups.parent(node)

            # a parent left out of the pairing resolves to a uuid string
            # or is simply absent; either way the node is a subset root
            if isinstance(parent, TransformData):
                indices[i] = position.get(parent.uuid, -1)
        return indices

    def _pair_by_index(self, other):
        """Pair nodes positionally, ignoring their names.

        The two hierarchies must hold the same number of nodes, in the same
        order, with the same parenting -- but NOT the same names, which is
        what lets a delta be taken between rigs that name their joints
        differently. The topology is checked rather than assumed: pairing
        unrelated joints would produce a plausible-looking wrong result
        instead of an error.
        """

        if not len(self) or len(self) != len(other):
            raise ValueError(
                f"index pairing needs equal non-zero lengths, "
                f"got {len(self)} and {len(other)}"
            )

        parents       = self._parent_indices(list(self))
        other_parents = self._parent_indices(list(other))

        mismatched = np.flatnonzero(parents != other_parents)
        if len(mismatched):
            i = int(mismatched[0])
            raise ValueError(
                f"hierarchies differ in structure at index {i}: "
                f"{self[i].name!r} has parent index {parents[i]}, "
                f"{other[i].name!r} has parent index {other_parents[i]}"
            )

        return list(self), list(other), parents

    @staticmethod
    def _parent_first_order(parents: np.ndarray) -> list:
        """Subset indices ordered so every parent precedes its children.

        Takes the parent map built by the pairing, so the order is relative to
        the pairing. :meth:`_parent_first` is the equivalent for a plain view
        and measures depth in the owning hierarchy instead.
        """
        return list(np.argsort(_chain_depths(parents), kind="stable"))

    def __isub__(self, other, by: str = "index", translate: bool = True):
        """In-place subtraction: ``self`` becomes the delta ``self - other``."""
        if not isinstance(other, TransformList):
            return NotImplemented
        _check_operands(self, other)

        nodes_self, nodes_other, parents = self._matched_subset(other, by=by)

        # snapshot every read up front: writing to a node invalidates the
        # cached world matrices of its whole branch
        s_self     = np.array([n.scale for n in nodes_self])
        s_other    = np.array([n.scale for n in nodes_other])
        w_self     = np.array([n.world_matrix for n in nodes_self])
        w_other    = np.array([n.world_matrix for n in nodes_other])
        compensate = np.array([n.segment_scale_compensate for n in nodes_self])

        # scale: additive delta around 1, divided by the parent's delta for
        # segment-scale-compensated joints (the inverse of __iadd__)
        base_delta = 1 + (s_self - s_other)
        scale      = base_delta.copy()
        for i in range(len(nodes_self)):
            if compensate[i] and parents[i] > -1:
                scale[i] = base_delta[i] / base_delta[parents[i]]

        # rotate: the frame offset F where `W_self == F * W_other`. F is fixed
        # to the bone, so it stays valid for every pose of the source
        rotate = matrix_to_quaternion(
            local_matrix(_rotation(w_self), _rotation(w_other))
        )

        # translate: a world-space offset, so __iadd__ restores a position
        # difference without re-reading it in a rotated parent frame. with
        # `translate` off it is left alone, and the delta keeps this rig's own
        # bone offsets for __iadd__ to reuse
        offset = w_self[:, 3, :3] - w_other[:, 3, :3]

        for i, node in enumerate(nodes_self):
            node.scale      = scale[i]
            node.quaternion = rotate[i]
            if translate:
                node.translate = offset[i]

        return self

    def __sub__(self, other):
        return self.get_delta(other)

    def get_delta(
        self, other, by: str = "index", translate: bool = True
    ) -> "HierarchyData":
        """Return the delta ``self - other``, ie: what ``other`` must gain to
        become ``self``.

        The receiver is the DESTINATION: pass the rig you have and call it on
        the rig you want. The inverse is :meth:`add_delta`, so for any pair
        ``b.add_delta(a.get_delta(b)) == a``.

        Either side may be a selection rather than a whole rig -- a
        ``match``, a slice, a branch. The result always owns its nodes, so it
        is a :class:`HierarchyData` whatever the receiver was.

        Args:
            other: the hierarchy or selection to subtract.
            by: ``"index"`` pairs positionally and ignores names, requiring
                the same joint count, order and parenting on both sides.
                ``"name"`` pairs on the intersection of the two name sets
                instead, so the delta may cover only part of a skeleton.
            translate: when False the delta carries this rig's own bone
                offsets instead of a position difference, so applying it
                keeps this rig's proportions. :meth:`add_delta` MUST be
                passed the same value -- the two store different things in
                the same channel.

        A :class:`ClipData` is the pose of the frame it is bound to, like
        any other. Its frame axis, ``clip.frames``, is every frame: the
        delta is taken against each, all solved at once, and comes back as
        a :class:`ClipData` on that clip's timebase, bound to its first
        frame. The axis of a clip of one frame, or none, is one pose.

        Both sides record one axis system and unit (:attr:`axis_system`,
        :attr:`unit`), which the delta records; values of two raise
        ValueError, since no one axis_system and unit says where their
        difference is.
        """
        # clip.frames stands for every frame of its clip: this pose, repeated
        # down the clip, is taken against each
        other, every = _framed_operand(other)
        if every:
            new = ClipData._adopt(
                self._owned_copy(), other.frame_count, other.start_frame, other.fps
            )
            new._subtract_all_frames(other, by=by, translate=translate)
            new._bind(0)
            return new

        new = self._owned_copy()
        return new.__isub__(other, by=by, translate=translate)

    def __iadd__(self, other, by: str = "index", translate: bool = True):
        """In-place addition: ``self`` (a delta) becomes ``self + other``."""
        if not isinstance(other, TransformList):
            return NotImplemented
        _check_operands(self, other)

        nodes_self, nodes_other, parents = self._matched_subset(other, by=by)

        # snapshot every read up front: the loop below writes as it goes, and
        # each write invalidates the cached world matrices of a whole branch
        s_self     = np.array([n.scale for n in nodes_self])
        s_other    = np.array([n.scale for n in nodes_other])
        f_self     = quaternion_to_matrix(np.array([n.quaternion for n in nodes_self]))
        t_self     = np.array([n.translate for n in nodes_self])
        w_other    = np.array([n.world_matrix for n in nodes_other])
        compensate = np.array([n.segment_scale_compensate for n in nodes_self])

        # scale: reconstruct the uncompensated base delta parent-first, then
        # undo the additive-around-1 offset (the inverse of __isub__)
        base_delta = s_self.copy()
        for i in self._parent_first_order(parents):
            if compensate[i] and parents[i] > -1:
                base_delta[i] = s_self[i] * base_delta[parents[i]]
        scale = s_other - 1 + base_delta

        # target world transform: the source's orientation carried into this
        # hierarchy's frame by F, at the source position plus the stored
        # offset. with `translate` off only the orientation is targeted and
        # each node keeps its own bone offset, so proportions are preserved
        target = matrix_multiply(f_self, _rotation(w_other))
        if translate:
            target[:, 3, :3] = w_other[:, 3, :3] + t_self

        # parent first, so every node solves against a parent already
        # converted into the target frame. the order comes from the OWNING
        # hierarchy, not the pairing: under a name pairing a node whose
        # parent is unpaired is a subset root there, and could be solved
        # before a paired grandparent that later moves and drags it
        for i in self._owning_parent_first(nodes_self):
            node = nodes_self[i]

            # a true inverse is required: a scaled parent's world matrix has
            # non-orthogonal rows, and matrix_local's fast affine inverse
            # (which assumes scale*rotation) would compute the wrong local
            local = np.dot(target[i], np.linalg.inv(node.get_parent_matrix()))

            node.scale      = scale[i]
            node.quaternion = matrix_to_quaternion(_rotation(local))[0]
            if translate:
                node.translate = local[3, :3]

        return self

    def __add__(self, other):
        # the operator takes the delta on the LEFT while add_delta takes it as
        # the argument, so `delta + rig` is spelled `rig.add_delta(delta)`
        if not isinstance(other, TransformList):
            return NotImplemented
        return other.add_delta(self)

    def add_delta(
        self, delta: "HierarchyData", by: str = "index", translate: bool = True
    ) -> "HierarchyData":
        """Return ``self`` with ``delta`` applied, as a new hierarchy.

        The result is built from ``delta``, so it carries the names, joint
        orients and rotate orders of the hierarchy the delta was taken from
        -- posing a flipped rig through a delta hands back the original rig,
        not the flipped one. The inverse is :meth:`get_delta`, so for any pair
        ``b.add_delta(a.get_delta(b)) == a``.

        Args:
            delta: a hierarchy produced by :meth:`get_delta`.
            by: the pairing used to build ``delta``; see :meth:`get_delta`.
            translate: pass the same value used to build ``delta``. False
                transfers orientation only, so every joint keeps the bone
                offset it already has and the rig holds its proportions at
                any pose of ``self``; a joint that moves rather than rotates
                (a root, an IK offset) will not follow.

        A :class:`ClipData` of deltas is the delta of the frame it is bound
        to. Its frame axis, ``deltas.frames``, is a delta per frame: each
        poses ``self``, all solved at once, and the result is a
        :class:`ClipData` on the delta clip's timebase, bound to its first
        frame. The axis of a clip of one frame, or none, is one delta.

        ``delta`` records the axis system and unit ``self`` does
        (:attr:`axis_system`, :attr:`unit`), which the result records; a
        delta of another raises ValueError, as :meth:`get_delta` of two.
        """
        # deltas.frames stands for every frame: a copy of the clip takes this
        # pose on every frame at once
        delta, every = _framed_operand(delta)
        if every:
            new = delta._rooted_copy()
            new._add_all_frames(self, by=by, translate=translate)
            new._bind(0)
            return new

        new = delta._owned_copy()
        return new.__iadd__(self, by=by, translate=translate)

    # --- overloaded methods --- #
    @overload
    def __getitem__(self, query: Union[int, str]) -> TransformData: ...

    @overload
    def __getitem__(self, query: Union[slice, List[int], np.ndarray]) -> TransformList: ...

    def __getitem__(self, query):
        """Index access that never transfers node ownership.

        Single-item access (by name or int) returns the node itself.
        Slice / sequence (fancy-index) access returns a NON-owning
        ``TransformList`` view over the same node objects -- it must NOT
        build ``self.__class__``, because a ``HierarchyData`` subclass
        would re-claim every node's ``_hierarchy`` back-pointer onto the
        throwaway result, silently corrupting the source hierarchy.
        """
        # lookup by name
        if isinstance(query, str):
            return self.list[self.index(query)]

        # fancy index (indices or boolean mask) -> non-owning view
        if _is_sequence(query):
            query = _mask_to_indices(query, len(self.list))
            return self._view([self.list[i] for i in query])

        # slice -> non-owning view; plain int -> the node itself
        result = self.list[query]
        if isinstance(query, slice):
            return self._view(result)
        return result

    def __contains__(self, obj):
        if isinstance(obj, self.DATA_LIST_CLASS):
            obj = obj.uuid

        elif isinstance(obj, str) and not validate_uuid(obj):
            return super().__contains__(obj)

        for shape in self.list:
            if shape.uuid == obj:
                return True

        return False

    def index(self, obj, start=0, stop=None) -> int:
        """
        The position of a node, given as the node, its uuid or its name,
        searched from ``start`` up to ``stop`` the way ``list.index``
        searches: a negative bound counts from the end. ValueError when no
        node matches.
        """
        if isinstance(obj, self.DATA_LIST_CLASS):
            obj = obj.uuid

        elif isinstance(obj, str) and not validate_uuid(obj):
            return super().index(obj, start=start, stop=stop)

        start, stop, _ = slice(start, stop).indices(len(self))
        for i, shape in enumerate(self.list[start:stop], start):
            if shape.uuid == obj:
                return i

        raise ValueError("Object is not in NodeList")

    def append(self, node: "TransformData") -> None:
        # reject wrong-typed input up front (matches DataList's contract)
        if self.DATA_LIST_CLASS is not None and not isinstance(
            node, self.DATA_LIST_CLASS
        ):
            raise TypeError(f"{node} is not a valid object.")

        # if the TransformData doesn't have a unique id, assign one and
        # append it to self
        if node.uuid is None:
            while True:
                new = generate_uuid()
                if not self._holds(new):
                    node.uuid = new
                    break
            self.list.append(node)

        # otherwise make sure TransformData not already in self
        elif self._holds(node.uuid):
            raise RuntimeError("TransformData is already in NodeList")

        # append the node to the list
        else:
            self.list.append(node)

        self._listed(node)

    def extend(self, iterable) -> None:
        """appends every node in turn, exactly as append() would

        append() looks for the node's uuid through the whole list, so
        building an N node hierarchy one append at a time is quadratic. For
        the length of this call that check reads a set of the uuids the list
        holds instead, kept up to date as the nodes go in, and dropped when
        the call returns: nothing outlives it to go stale.
        """
        # snapshot first, as DataList.extend does; a few nodes are checked
        # quicker by append's own scan than by building the set
        values = list(iterable)
        if len(values) < 4 or "_held" in self.__dict__:
            return super().extend(values)

        held = set()
        for node in self.list:
            uuid = getattr(node, "uuid", None)
            if not isinstance(uuid, str):
                return super().extend(values)
            held.add(uuid)

        self._held = held
        try:
            super().extend(values)
        finally:
            self.__dict__.pop("_held", None)

    def _holds(self, uuid) -> bool:
        """whether a node of this list carries ``uuid``, compared as uuids"""
        held = self.__dict__.get("_held")
        if held is not None and isinstance(uuid, str):
            return uuid in held

        for shape in self.list:
            if shape.uuid == uuid:
                return True
        return False

    def _listed(self, node: "TransformData") -> None:
        """records a node just added in the set extend() keeps, if any"""
        held = self.__dict__.get("_held")
        if held is not None:
            if isinstance(node.uuid, str):
                held.add(node.uuid)
            else:
                # a uuid the set cannot stand for: back to scanning the list
                del self.__dict__["_held"]

    def insert(self, index: int, node: "TransformData") -> None:
        # reject wrong-typed input up front (matches DataList's contract)
        if self.DATA_LIST_CLASS is not None and not isinstance(
            node, self.DATA_LIST_CLASS
        ):
            raise TypeError(f"{node} is not a valid object.")

        # NOTE: a TransformList is a non-owning view, so it never claims
        # _hierarchy -- only HierarchyData (the owning container) does.

        # if the TransformData doesn't have a unique id, assign
        # one and append to the self
        if node.uuid is None:
            while True:
                new = generate_uuid()
                if not self._holds(new):
                    node.uuid = new
                    break

            self.list.insert(index, node)

        # otherwise make sure TransformData not already in self
        elif self._holds(node.uuid):
            raise RuntimeError("TransformData is already in NodeList")

        # insert the node to the list
        else:
            self.list.insert(index, node)

        self._listed(node)

    def to_dict(self) -> dict:
        """
        The list as a tree. A view saves as the hierarchy :meth:`copy`
        gives: its own nodes, owned, a node whose parent the view left out
        made a root. A view on its own has no hierarchy to point its nodes
        at, so loading one back as a view would read every node as a root.
        An object two of its nodes hold, a mesh two Objects instance, is
        saved once, as in any tree. Inside another save that reaches an
        object they hold twice, a mesh the view and a field both hold, the
        view saves its nodes over the very objects they hold, not over
        copies, so that object is saved once too.
        """
        if type(self) is TransformList:
            if _save_shares(self):
                return self._owned([copy.copy(node) for node in self]).to_dict()
            return self._owned_copy().to_dict()

        return super().to_dict()

    def _saved_parts(self) -> list:
        # a view saves copies of its nodes: a save reaches what its nodes
        # hold, never the nodes themselves
        if type(self) is TransformList:
            return [part for node in self for part in node._saved_parts()]
        return super()._saved_parts()

    @classmethod
    def _from_state(cls, data: dict) -> "TransformList":
        # one load of nodes: a user attribute a built-in name hides warns
        # once a name, not once a node
        with _NodesLoad():
            return super()._from_state(data)

    def match(
        self, *args, exclude: bool = False, exact: bool = True
    ) -> "TransformList":
        """returns a TransformList object with matching names"""
        new = self._view(())

        for shape in self.list:
            if exclude:
                if not shape.match(*args, exact=exact):
                    new.append(shape)
            elif shape.match(*args, exact=exact):
                new.append(shape)

        return new


# Maya's short names, vectorised as their long names are
_add_short_names(TransformList)


# the nodes a rig holds from which an append with no current index builds
# one (HierarchyData.append), and a parent looked up after a few scans
# (HierarchyData._index_for_lookup): a scan of fewer costs less than the build
_BUILT_FOR_APPEND = 64

# the lookups of one node's parent that scan a rig at one structure count
# before the next builds the index: a build of 2000 nodes costs about as
# much as 14 such scans of a chain, each finding the parent halfway down
# the list on average, or 7 scans of the whole list
_SCANS_BEFORE_BUILD = 8


class HierarchyData(TransformList):
    """Owning container for a hierarchy of ``TransformData`` nodes.

    Unlike its :class:`TransformList` base (a non-owning view), a
    ``HierarchyData`` *owns* the nodes it holds: every node added through
    :meth:`append`/:meth:`insert` (and :meth:`extend`, the constructor and a
    load, which append) has its ``_hierarchy`` back-pointer claimed via
    :meth:`_claim` (detaching it from any prior owner first), and
    :meth:`pop` clears it. A node belongs to exactly one ``HierarchyData`` at
    a time, which is what lets the ``_hierarchy`` back-pointer always resolve
    to a single, correct source of truth. The pose algebra (``a - b`` / ``a
    + b``, :meth:`get_delta` / :meth:`add_delta`) is inherited from
    :class:`TransformList` so it works over selections too.

    A node joining a hierarchy keeps its parent only when the hierarchy
    holds it: one whose ``parent_uuid`` no node of the hierarchy carries
    becomes a root (``parent_uuid = None``), and a parent added after does
    not take it back. Nodes that join together -- the constructor,
    :meth:`extend`, a load -- are checked once every one of them is in, so
    a child may come before its parent. An item assignment, which claims
    nothing, holds a node no hierarchy owns to the same rule.
    """

    # the private state a rig holds beside its list: its structure counts,
    # the index and parent rows kept with them, and the bookkeeping append,
    # get_parent, extend and an item assignment keep between calls
    _STATE = (
        "_index_version",
        "_structure_version",
        "_rig_index_cache",
        "_parent_rows_cache",
        "_append_builds",
        "_lookup_scans",
        "_joining",
        "_uuid_counts",
    )

    # saved beside the nodes: the axis system and the unit their values are
    # in, which a file names as their properties, axis_system and unit. A
    # file saved before there were any loads them as hand-built data holds
    # them, Maya's Y-up and cm
    LIST_FIELDS = ("_axis_system", "_unit")

    # hand-built data's record; an empty rig takes the one the nodes of
    # another rig, clip or view joining it bring (_join_records)
    _axis_system = _MAYA_Y_UP
    _unit        = 1.0

    @property
    def axis_system(self) -> str:
        """the axis system the values are in, "maya_y_up" or one in words

        "maya_y_up" is Maya's Y-up; any other is its up and front axes,
        either one negated, and its hand in words, "z up, -y front, right
        handed" for Unreal's. :meth:`load_fbx` and :meth:`ClipData.load_fbx`
        record what they produced: "maya_y_up" by default, and with
        ``axis_system=None`` the file's own axes. :meth:`save_fbx` labels
        the file with it, so values saved in a file's own axes are labelled
        in them, and load, and import into Maya, where that file does.
        Hand-built data, and a file saved before there was a record, hold
        "maya_y_up". A rig or clip built from another
        (``HierarchyData(view)``, ``ClipData(rig, frames=...)``, a copy, a
        cut of frames, a delta) takes its record. Values of two records
        have no one label, and where they would meet ValueError is raised:
        nodes of a rig recording another joining this one (:meth:`append`,
        :meth:`extend`, :meth:`insert`, an item assignment), unless this one
        is empty, which takes theirs; the pose algebra between two
        (:meth:`get_delta`, :meth:`add_delta`); matching or pasting the
        values of nodes of another (``match_translate``, ``match_rotate`` and
        ``match_matrix`` of a node, a view, a rig, a clip or ``clip.frames``,
        given nodes, a view, a rig, a clip or a frame axis, and a pose or
        frames pasted into ``clip.frames``), where values pass as they are,
        and so do nodes no rig owns, but in a view
        :meth:`TransformList.load_fbx` gave or one taken from it, which hold
        its load's; :meth:`ClipData.from_poses` of poses of two; a clip's
        :attr:`ClipData.rest_pose` of another.
        Setting it relabels the values and moves none of them, a clip's rest
        pose with the clip; Maya's Y-up in words is held as "maya_y_up".
        """
        return self._axis_system

    @axis_system.setter
    def axis_system(self, value: str) -> None:
        self._axis_system = _axis_record(value)

    @property
    def unit(self) -> float:
        """the centimeters one unit of the values' lengths holds, 1.0 for cm

        100.0 for meters, 2.54 for inches. :meth:`load_fbx` and
        :meth:`ClipData.load_fbx` record what they produced: 1.0 when they
        convert the file's unit to cm, the default, and with a number for
        ``scale_factor`` the file's unit in cm over that number, the unit
        the file's lengths times it are in (a cm file read with 0.01 records
        100.0, meters). :meth:`save_fbx` writes the file in it, so the file
        loads at the size the file it was read from does, and Maya's
        importer puts its nodes at the positions it puts that file's, its
        top nodes scaled by the unit (:meth:`save_fbx` says why). Hand-built
        data, and a file saved before there was a record, hold 1.0. It is
        passed on, checked and set as :attr:`axis_system` is: setting it
        relabels the values, scaling none of them.
        """
        return self._unit

    @unit.setter
    def unit(self, value: float) -> None:
        self._unit = _unit_record(value)

    def _records(self) -> tuple:
        return self._axis_system, self._unit

    def _held_records(self) -> list:
        # a node an item assignment lists here that another rig owns is in
        # that rig's record, and a match or a paste writes or reads its
        # values with the rest: its record counts, as a view's nodes' do, so
        # one of another record is refused before the first node moves
        records = {self._records()}
        for node in self.list:
            owner = getattr(node, "_hierarchy", None)
            if owner is not None and owner is not self:
                records.add(owner._records())
        return sorted(records)

    def _join_records(self, nodes, given=None) -> None:
        """
        Checks that ``nodes``, about to join this rig, are in its axes and
        unit: a node another rig owns in what that rig records, and a node
        no rig owns of the view :meth:`TransformList.load_fbx` gave, or of
        one taken from it (``given``, the view they come in), in what its
        load records. Any other node holds no record. An empty rig takes the
        one they hold.

        Raises
        ------
        ValueError
            The nodes hold two records, or another than this rig's.
        """
        loaded = given._record if isinstance(given, TransformList) else None
        found  = set()
        for node in nodes:
            owner = getattr(node, "_hierarchy", None)
            if owner is None:
                if loaded is not None:
                    found.add(loaded)
            elif owner is not self:
                found.add(owner._records())
        if not found:
            return

        empty = not self.list
        if not empty:
            found.add(self._records())
        if len(found) > 1:
            _refuse_records("the nodes joining the rig" if empty else "the rig and the nodes joining it", found)
        if empty:
            self.axis_system, self.unit = found.pop()

    def _post_load(self) -> None:
        """
        Checks the record a file holds, as the setters check one.

        Raises
        ------
        TypeError, ValueError
            The file's axis_system or unit is none the setters take.
        """
        super()._post_load()
        self._axis_system = _axis_record(self._axis_system)
        self._unit        = _unit_record(self._unit)

    def _claim(self, node: "TransformData") -> None:
        """Take ownership of ``node`` for this hierarchy.

        A node belongs to exactly one HierarchyData at a time. If it was
        owned by another hierarchy, detach it from that one first so the
        same object never lives in two lists with a single back-pointer.
        """
        prior = node._hierarchy
        if prior is not None and prior is not self:
            for i, existing in enumerate(prior.list):
                if existing is node:
                    prior.list.pop(i)
                    _structure_changed(prior)
                    break
            # a node an item assignment listed there twice stays listed in
            # the other place, which no longer owns it
            _listed_by(node, prior)
        node._hierarchy = self

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        # a list replaced can give any node other children: cached world
        # matrices are not trusted past it
        if name == "list":
            _structure_changed(self)

    # the structure counts of this rig (_structure_changed): the last edit
    # of any kind, which retires the index, and the last that can give a
    # node another parent, which retires the cached world matrices too. -1
    # until the list is first set, never a count a node with no rig holds
    _index_version     = -1
    _structure_version = -1

    # (index version, list, length) the index was built at, and the index
    _rig_index_cache = None

    def _rig_index(self) -> Optional[_RigIndex]:
        """the index :class:`_Lookups` reads this hierarchy's parents,
        children, positions and unique names from, kept between calls

        A branch walk, which every channel write makes to clear the cached
        world matrices below the node, asks each node of the branch for its
        children: an index built for each walk would cost the hierarchy's
        size every write. It is kept until this rig's structure count moves:
        a uuid written on one of its nodes, or on a node it lists without
        owning it (_listed_by), or an edit to its list through
        its methods that takes nodes out or reorders them. While it is
        current, a node appended, renamed or given another parent_uuid
        patches it in place instead; with none current, such an edit has
        none to patch, and a rename has nothing to retire. A node inserted
        ahead of the end shifts it in place when it was fetched here since
        the last such insert, and retires it otherwise (:meth:`_added`). A
        rig built node by node, a world read or written on the way, builds
        it at the first read: from there on each node costs the same. A
        ``list`` replaced, or grown or shrunk in place, is caught as well; an
        assignment, sort or reverse made on the raw ``list`` is not, and
        nothing in cgmath makes one.
        """
        if self._rig_index_is_current():
            index = self._rig_index_cache[3]
        else:
            nodes                 = self.list
            index                 = _RigIndex.build(self)
            self._rig_index_cache = (self._index_version, nodes, len(nodes), index)
        if index is not None:
            index.fetched = True
        return index

    # (the index they were read from, each node's parent position)
    _parent_rows_cache = None

    def _parent_rows(self) -> np.ndarray:
        """each node's parent as a position in the list, -1 for a root or a
        parent the hierarchy does not hold (:meth:`TransformList._parent_indices`),
        kept with the index: read only, a rig edit copies it to write"""
        index = self._rig_index()
        cache = self._parent_rows_cache
        if index is not None and cache is not None and cache[0] is index:
            return cache[1]

        rows = TransformList._parent_indices(self.list)
        rows.setflags(write=False)
        self._parent_rows_cache = (index, rows)
        return rows

    def _rig_index_is_current(self) -> bool:
        """whether the index kept is the one the structure would build now"""
        cache = self._rig_index_cache
        return (
            cache is not None
            and cache[0] == self._index_version
            and cache[1] is self.list
            and cache[2] == len(self.list)
        )

    def _kept_index(self) -> Optional[_RigIndex]:
        """the index kept, while it is current, for an edit to patch in place
        or a lookup to read; None when there is none"""
        cache = self._rig_index_cache
        if (
            cache is not None
            and cache[0] == self._index_version
            and cache[1] is self.list
            and cache[2] == len(self.list)
        ):
            return cache[3]
        return None

    def _index_patched(self) -> None:
        """the index kept was just patched for an edit: it stands for the list
        as it now is, and the parent rows read from it go"""
        cache = self._rig_index_cache
        # straight to the instance: these are written on every node added
        self.__dict__["_rig_index_cache"]   = (cache[0], cache[1], len(self.list), cache[3])
        self.__dict__["_parent_rows_cache"] = None

    def _holds(self, uuid) -> bool:
        # the index kept answers without a scan of the list
        if isinstance(uuid, str) and "_held" not in self.__dict__:
            index = self._kept_index()
            if index is not None:
                return uuid in index.uuids
        return super()._holds(uuid)

    def _added(self, index: Optional[_RigIndex], at: int, node: "TransformData") -> None:
        """``node`` was just put at ``at``; ``index`` is the one kept before

        The index is patched to hold the node, and the nodes it now parents
        -- those whose parent_uuid named its uuid, which no node carried
        before -- are the only ones it can move: when there are none, every
        cached world matrix of the rig stands. An index that went out of
        date on the way, or a node it cannot hold, retires both, as any edit
        does.

        A node inserted ahead of the end moves every later position up one,
        a pass over the whole index (:meth:`_RigIndex._shift`). It is made
        only when a lookup fetched the index since the last such insert:
        inserts with nothing read between retire it instead, as every insert
        did before the index was patched. The cached world matrices stand
        then, as for a patch, unless a node names the one put in.
        """
        cache = self._rig_index_cache
        if (
            index is None
            or cache[0] != self._index_version
            or cache[1] is not self.list
            or cache[2] + 1 != len(self.list)
            or self.list[at] is not node
        ):
            _structure_changed(self)
            return

        # the nodes naming the new one's uuid are the ones it moves
        uuid = node.uuid
        if not isinstance(uuid, _INDEX_KEYS):
            _structure_changed(self)
            return
        named = uuid is not None and uuid in index.refs
        ahead = at < len(self.list) - 1
        if ahead and not index.fetched:
            _structure_changed(self, index_only=not named)
            return
        if not index.add(at, node):
            _structure_changed(self)
            return
        if ahead:
            index.fetched = False

        # the node's own cached world matrix holds the count of another rig,
        # or 0 for none, never this rig's: it is not trusted here
        self._index_patched()
        if named:
            _worlds_moved(self)

    def _join_one(self, node: "TransformData", index: Optional[_RigIndex]) -> None:
        """``node``, just listed and claimed, joins alone: a root unless a
        node of this hierarchy carries its parent_uuid. ``index`` is the
        one kept before it was listed, which does not hold it yet"""
        ref = node.parent_uuid
        if ref is None or ref == node.uuid:
            return
        if index is not None:
            held = ref in index.uuids
        else:
            held = any(member.uuid == ref for member in self.list)
        if not held:
            # past __setattr__: the node is in no index yet, which _added
            # patches or retires next
            object.__setattr__(node, "parent_uuid", None)

    def _join(self, nodes) -> None:
        """``nodes``, every one of them listed, joined together: each whose
        parent_uuid no node of this hierarchy carries becomes a root"""
        linked = [node for node in nodes if node.parent_uuid is not None]
        if not linked:
            return

        index = self._kept_index()
        if index is not None:
            held = index.uuids
        else:
            held = {node.uuid for node in self.list if isinstance(node.uuid, str)}
        for node in linked:
            if node._hierarchy is self and node.parent_uuid not in held:
                node.parent_uuid = None

    def _index_for_append(self) -> Optional[_RigIndex]:
        """the index an append to a rig of _BUILT_FOR_APPEND nodes or more
        with none current builds, for the check that the uuid is new and the
        join to read where they would scan the list; None, for the append to
        scan, while such builds go to waste

        A loop appending, naming and parenting nodes, nothing read, builds
        once and patches the index from there on. An edit that retires the
        index between appends -- a remove, a del, a pop, a sort, a uuid
        written -- would make every append build, which costs several scans:
        an append that finds the index it built last retired, no append
        having found it current, scans for twice as many appends as it did
        the last time (1, 2, 4, ...) before building again, so such a loop
        builds a number of times the log of its length. An append that finds
        the index current starts over (:meth:`append`). The build is known by
        the structure count it was made at, not held.
        """
        built, wait, left = self.__dict__.get("_append_builds", (None, 0, 0))
        cache = self._rig_index_cache
        if built is not None and cache is not None and cache[0] == built:
            wait = left = max(1, 2 * wait)
        if left:
            self.__dict__["_append_builds"] = (None, wait, left - 1)
            return None

        # a build no lookup has read yet: an insert ahead of the end retires
        # it rather than shifting it, and the next append finds it retired
        index                           = self._rig_index()
        self.__dict__["_append_builds"] = (self._rig_index_cache[0], wait, 0)
        if index is not None:
            index.fetched = False
        return index

    def _index_for_lookup(self) -> Optional[_RigIndex]:
        """the index a node's get_parent, and parent_name, reads when none is
        current; None, for it to scan the list

        A rig of _BUILT_FOR_APPEND nodes or more builds one once
        _SCANS_BEFORE_BUILD lookups have scanned it at one structure count:
        a loop reading every node's parent on a rig just made or loaded
        scans a few times and reads the index from there on, where it
        scanned at each node. A loop with an edit retiring the index at each
        step scans at each, as building at each would cost more.
        """
        version = self._index_version
        scans   = self.__dict__.get("_lookup_scans")
        count   = 1 if scans is None or scans[0] != version else scans[1] + 1
        if count > _SCANS_BEFORE_BUILD and len(self.list) >= _BUILT_FOR_APPEND:
            return self._rig_index()
        self.__dict__["_lookup_scans"] = (version, count)
        return None

    def append(self, node: "TransformData") -> None:
        # validate + add first (raises before we claim the node), then
        # take ownership (detaching from any prior hierarchy). Alone, it
        # joins now; in a batch, once every node is in (extend). A rig of
        # _BUILT_FOR_APPEND nodes or more with no current index builds one
        # for an append outside a batch, unless such builds go to waste
        # (_index_for_append). A node of another rig joins in that rig's
        # axes and unit (_join_records), which a batch checks once
        owner = getattr(node, "_hierarchy", None)
        if owner is not None and owner is not self and "_joining" not in self.__dict__:
            self._join_records((node,))
        index = self._kept_index()
        if index is not None:
            if "_append_builds" in self.__dict__:
                del self.__dict__["_append_builds"]
        elif "_held" not in self.__dict__:
            if len(self.list) >= _BUILT_FOR_APPEND:
                index = self._index_for_append()
            if index is None and "_joining" not in self.__dict__ and self._appended_by_scan(node):
                self._added(None, len(self.list) - 1, node)
                return
        super().append(node)
        self._claim(node)
        if "_joining" not in self.__dict__:
            self._join_one(node, index)
        self._added(index, len(self.list) - 1, node)

    def _appended_by_scan(self, node: "TransformData") -> bool:
        """append() with no index to read, of a node naming a parent: one
        pass over the list checks that its uuid is new and looks for the
        parent, where each scanned the list. The pass runs from the end, the
        parent compared until it is found: a node appended under the node
        appended before it, as a rig is built or a branch moved to the end,
        costs one comparison a node, as a root does. False, nothing done, for
        any other node, which append() takes on
        """
        kind = self.DATA_LIST_CLASS
        if not isinstance(node, TransformData) or (kind is not None and not isinstance(node, kind)):
            return False
        ref, uuid = node.parent_uuid, node.uuid
        if ref is None or not (uuid is None or isinstance(uuid, str)) or ref == uuid:
            return False

        # a uuid made for a node that has none, made again in the unlikely
        # case a node holds it, as TransformList.append makes it
        fresh = uuid is None
        while True:
            if fresh:
                uuid = generate_uuid()
            taken   = held = False
            members = reversed(self.list)
            for member in members:
                other = member.uuid
                if other == uuid:
                    taken = True
                    break
                if other == ref:
                    held = True
                    break
            if held:
                for member in members:
                    if member.uuid == uuid:
                        taken = True
                        break
            if not taken:
                break
            if not fresh:
                raise RuntimeError("TransformData is already in NodeList")

        if fresh:
            node.uuid = uuid
        self.list.append(node)
        self._claim(node)
        if not held:
            # past __setattr__, as _join_one writes it
            object.__setattr__(node, "parent_uuid", None)
        return True

    def extend(self, iterable) -> None:
        """appends every node in turn, as append() would, joining them together

        A node is made a root only when no node of the hierarchy carries its
        parent_uuid once every node is in, so the nodes may come in any
        order.
        """
        values = list(iterable)
        self._join_records(values, iterable)
        if "_joining" in self.__dict__:
            return super().extend(values)

        self.__dict__["_joining"] = True
        try:
            super().extend(values)
        finally:
            del self.__dict__["_joining"]
            self._join([v for v in values if isinstance(v, TransformData)])

    def insert(self, index: int, node: "TransformData") -> None:
        self._join_records((node,))
        kept  = self._kept_index()
        count = len(self.list)
        super().insert(index=index, node=node)
        self._claim(node)
        self._join_one(node, kept)

        # where list.insert puts it
        at = int(index)
        if at < 0:
            at = max(at + count, 0)
        self._added(kept, min(at, count), node)

    def __setitem__(self, index, value) -> None:
        # an item assignment lists a node without claiming it, as it always
        # has: a clip skins such a node from its own channels, which its
        # blocks know nothing of. A node no hierarchy owns joins the list as
        # an appended one joins, a root unless its parent is here; a node of
        # another hierarchy keeps the parent it has there, and one listed
        # here already, as an item swap moves them, is no new member. A node
        # this rig does not own is told the rig lists it (_listed_by), so a
        # uuid written on it retires the rig's index and uuid counts. The
        # list is read only for a node no hierarchy owns that names a
        # parent: a swap, reverse() or a loop putting nodes back makes no
        # pass over it. A node of another rig is listed in that rig's axes
        # and unit, and a node no rig owns of TransformList.load_fbx's view
        # in its load's, as one joining is (_join_records)
        if isinstance(index, slice):
            given = value
            value = list(value)
            self._join_records(value, given)
            counts = self._uuid_counts_kept()
            gone   = self.list[index] if counts is not None else ()
            super().__setitem__(index, value)
            _structure_changed(self)
            loose = []
            for node in value:
                if isinstance(node, TransformData) and node._hierarchy is not self:
                    _listed_by(node, self)
                    if node._hierarchy is None and node.parent_uuid is not None:
                        loose.append(node)
            if counts is not None:
                _uuids_counted(counts, gone, -1)
                _uuids_counted(counts, value, 1)
            elif loose:
                counts = _uuids_counted({}, self.list, 1)
            else:
                return
            for node in loose:
                if node.parent_uuid not in counts:
                    node.parent_uuid = None
            self.__dict__["_uuid_counts"] = (self._index_version, self.list, len(self.list), counts)
            return

        # one node: the count of each uuid the list holds, made for such a
        # node, is kept for the next assignment, which patches it for the
        # node taken out and the node put in, while nothing else edits the
        # list or writes a uuid on a node it holds between
        # (_uuid_counts_kept). A loop assigning such nodes -- each node's
        # copy put back in its place -- counts the list once, where a count
        # at each ran it in the square of its size
        owner = getattr(value, "_hierarchy", None)
        if owner is not None and owner is not self:
            self._join_records((value,))
        if "_uuid_counts" not in self.__dict__:
            super().__setitem__(index, value)
            _structure_changed(self)
            if not isinstance(value, TransformData) or value._hierarchy is self:
                return
            _listed_by(value, self)
            if value._hierarchy is not None or value.parent_uuid is None:
                return
            counts, loose = None, True
        else:
            counts = self._uuid_counts_kept()
            if counts is not None:
                try:
                    old = self.list[index]
                except (IndexError, TypeError):
                    counts = None  # the assignment raises
            super().__setitem__(index, value)
            _structure_changed(self)
            loose = False
            if isinstance(value, TransformData) and value._hierarchy is not self:
                _listed_by(value, self)
                loose = value._hierarchy is None and value.parent_uuid is not None

        if counts is not None:
            gone, new = getattr(old, "uuid", None), getattr(value, "uuid", None)
            if not (isinstance(gone, str) and isinstance(new, str) and gone == new):
                if isinstance(gone, str):
                    left = counts.get(gone, 0) - 1
                    if left > 0:
                        counts[gone] = left
                    else:
                        counts.pop(gone, None)
                if isinstance(new, str):
                    counts[new] = counts.get(new, 0) + 1
        elif loose:
            counts = _uuids_counted({}, self.list, 1)
        else:
            return

        if loose and value.parent_uuid not in counts:
            value.parent_uuid = None
        self.__dict__["_uuid_counts"] = (self._index_version, self.list, len(self.list), counts)

    def _uuid_counts_kept(self) -> Optional[dict]:
        """the count of each uuid the list holds, which the last item
        assignment kept, taken from the rig while it stands for the list;
        None once another edit came between, a uuid written on a node the
        list holds included: its owner's count moves, or a rig's listing it
        without owning it (_listed_by)"""
        kept = self.__dict__.pop("_uuid_counts", None)
        if (
            kept is None
            or kept[0] != self._index_version
            or kept[1] is not self.list
            or kept[2] != len(self.list)
        ):
            return None
        return kept[3]

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        _structure_changed(self)

    def sort(self, key=None, reverse=False) -> None:
        """sorts the nodes in place, by name unless ``key`` says otherwise"""
        # a sort whose comparison raises can leave the list part sorted, so
        # the index is retired whatever happens
        try:
            super().sort(key=key, reverse=reverse)
        finally:
            _structure_changed(self)

    def pop(self, index=-1) -> TransformData:
        # retrieves a node and removes pointer
        if index < 0:
            index += len(self.list)
        if index < 0 or index >= len(self.list):
            raise IndexError("pop index out of range")

        # a node listed here, owned by another rig or by none, as an item
        # assignment lists one, leaves this list alone: its owner keeps it,
        # and its children there their parent. Listed twice, it stays listed
        # in the other place, as the edit that listed it told it (_listed_by)
        target = self.list[index]
        if target._hierarchy is not self:
            node = self.list.pop(index)
            _structure_changed(self)
            return node

        # its children are found while it is still listed. The previous pop
        # retired the index, so a loop of pops (clear(), say) would rebuild
        # it every time: a scan finds them as cheaply
        if self._rig_index_is_current():
            children = _Lookups().children(target)
        else:
            children = target._scan_children()

        # extract the node and remove pointer. A node an item assignment
        # listed twice stays listed in the other place, which no longer owns
        # it: told so, as an item assignment tells such a node
        node            = self.list.pop(index)
        node._hierarchy = None
        _listed_by(node, self)
        _structure_changed(self)

        # any node in the list which had this node as a parent is now at
        # root. all of them first, so the resets walk one index
        for obj in children:
            obj.parent_uuid = None
        _reset_branches(children)

        return node

    @classmethod
    def _from_state(cls, data: dict) -> "HierarchyData":
        # every node built first, then added as one batch, so that a node
        # saved ahead of its parent keeps it (the joining rule). A file older
        # than parent_uuid names a parent under "parent_node": a uuid is read
        # as parent_uuid (TransformData._from_state), and a name is the first
        # node of that name once every node is in, no node of it a root
        if ITEMS_KEY in data:
            items = data.pop(ITEMS_KEY)
        else:
            # a dict of item dicts, as the asset readers build them
            items, data = list(data.values()), {}

        # one load of nodes, a clip's rest pose with them, which warns of a
        # user attribute a built-in name hides once a name, once it is done
        with _NodesLoad():
            pending = _LEGACY_PARENTS.loads
            pending.append([])
            try:
                # a loop, not a comprehension, which would cost each nested
                # object one more frame
                nodes = []
                for item in items:
                    nodes.append(_from_tagged(item, cls.DATA_LIST_CLASS))
            finally:
                named = pending.pop()

            obj = cls()
            obj.extend(nodes)
            if named:
                _parent_by_name(obj, named)

            # in the order the tree holds them, as DataList._from_state reads
            # them. A part the class no longer declares is left out, but an
            # object first saved there is built still: a later part may refer
            # to it
            for name, value in data.items():
                if name in cls.LIST_FIELDS:
                    setattr(obj, name, _decode(value))
                elif _holds_numbered(value):
                    _decode(value)

            obj._post_load()
        return obj


def _parent_by_name(owner, named) -> None:
    """each (node, name) of ``named``, a node of ``owner`` a file older than
    parent_uuid parented by name, given the uuid of the first node of
    ``owner`` carrying that name; a node when none does is a root"""
    first = {}
    for node in owner.list:
        if isinstance(node.name, str):
            first.setdefault(node.name, node)

    for node, name in named:
        parent = first.get(name) if isinstance(name, str) else None
        if parent is not None and parent is not node and _is_uuid(parent.uuid):
            node.parent_uuid = parent.uuid


def _diagonal(values: np.ndarray) -> np.ndarray:
    """(..., 3) factors as (..., 4, 4) diagonal matrices"""
    out                  = np.zeros(values.shape[:-1] + (4, 4), dtype=float)
    axes                 = np.arange(3)
    out[..., axes, axes] = values
    out[..., 3, 3]       = 1.0
    return out


def _translation(values: np.ndarray) -> np.ndarray:
    """(..., 3) offsets as (..., 4, 4) row vector translation matrices"""
    out                  = np.zeros(values.shape[:-1] + (4, 4), dtype=float)
    axes                 = np.arange(4)
    out[..., axes, axes] = 1.0
    out[..., 3, :3]      = values
    return out


def _euler_matrices(degrees: np.ndarray, orders: Optional[np.ndarray]) -> np.ndarray:
    """(F, N, 3) euler angles as (F, N, 4, 4) rotation matrices

    ``orders`` is per node, so the batch is split by the orders actually
    present rather than solved one node at a time.
    """
    frames, count = degrees.shape[:2]
    flat = np.radians(degrees).reshape(-1, 3)

    if orders is None:
        return euler_to_matrix(flat, 0).reshape(frames, count, 4, 4)

    out   = np.empty((frames * count, 4, 4), dtype=float)
    tiled = np.tile(np.asarray(orders, dtype=int), frames)
    for order in np.unique(tiled):
        rows      = tiled == order
        out[rows] = euler_to_matrix(flat[rows], int(order))

    return out.reshape(frames, count, 4, 4)


def _matrices_to_euler(matrices: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """(F, N, 4, 4) rotations as (F, N, 3) euler degrees, ``orders`` per node

    The inverse of :func:`_euler_matrices`, batched the same way: one solve
    per rotate order actually present.
    """
    frames, count = matrices.shape[:2]
    out = np.empty((frames, count, 3), dtype=float)
    for order in np.unique(orders):
        cols = np.flatnonzero(orders == order)
        flat = np.ascontiguousarray(matrices[:, cols]).reshape(-1, 4, 4)
        out[:, cols] = np.degrees(matrix_to_euler(flat, int(order))).reshape(
            frames, len(cols), 3
        )
    return out


def _scales_and_rotations(block: np.ndarray, current) -> tuple:
    """:func:`_scale_and_rotation` over a stack of (..., 3, 3) blocks at once

    The same split, element for element: the row lengths are the signed
    scale, a mirror goes on scale z, and a zero scale takes what its row
    leaves out from ``current``, the orientation the node holds now.
    ``current`` may be a callable building that stack; it is only called
    when some scale is zero.
    """
    scale = np.einsum("...ij,...ij->...i", block, block) ** 0.5
    held  = scale > 1e-12 * scale.max(axis=-1, keepdims=True)
    kept  = held.sum(axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rotation = block / np.where(held, scale, 1.0)[..., None]

    mirror              = (kept == 3) & (_det3(block) < 0)
    scale[mirror, 2]    = -scale[mirror, 2]
    rotation[mirror, 2] = -rotation[mirror, 2]
    if (kept == 3).all():
        return scale, rotation

    current  = np.asarray(current() if callable(current) else current, dtype=float)
    rotation = np.where(held[..., None], rotation, current)

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3

        # one zero scale: its row closes the right handed frame
        one = (kept == 2) & ~held[..., i]
        if one.any():
            rotation[one, i] = np.cross(rotation[one, j], rotation[one, k])

        # two zero scales: the current row that leans least on the held one
        # is squared to it, and the last row closes the right handed frame
        two = (kept == 1) & held[..., i]
        if two.any():
            rows = rotation[two]
            u    = rows[:, i]
            dj   = np.einsum("mi,mi->m", rows[:, j], u)
            dk   = np.einsum("mi,mi->m", rows[:, k], u)
            lean = (np.abs(dj) <= np.abs(dk))[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                vj = rows[:, j] - dj[:, None] * u
                vj = vj / np.linalg.norm(vj, axis=-1, keepdims=True)
                vk = rows[:, k] - dk[:, None] * u
                vk = vk / np.linalg.norm(vk, axis=-1, keepdims=True)
            rows[:, j]    = np.where(lean, vj, np.cross(vk, u))
            rows[:, k]    = np.where(lean, np.cross(u, vj), vk)
            rotation[two] = rows

    # three zero scales leave the current orientation as it is
    return scale, rotation


def _same_transforms(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """:func:`_same_transform` per matrix of two (..., 4, 4) stacks"""
    return np.all(np.abs(A - B) <= 1e-8, axis=(-2, -1))


def _eulers_to_matrices(degrees: np.ndarray, orders) -> np.ndarray:
    """(..., 3) euler degrees as (..., 4, 4) rotations, ``orders`` broadcast
    over the leading axes

    One kernel call whatever orders are present: a call into the parallel
    kernels costs far more than the work in it once the machine is busy, so
    a solve makes as few of them as it can.
    """
    degrees = np.asarray(degrees, dtype=float)
    lead    = degrees.shape[:-1]
    if not int(np.prod(lead)):
        return np.zeros(lead + (4, 4), dtype=float)
    axes = np.broadcast_to(np.asarray(orders, dtype=np.int32), lead).reshape(-1)
    flat = np.radians(degrees).reshape(-1, 3)
    return euler_to_matrix(flat, axes).reshape(lead + (4, 4))


def _matrices_to_eulers(matrices: np.ndarray, orders) -> np.ndarray:
    """(..., 4, 4) or (..., 3, 3) rotations as (..., 3) euler degrees,
    ``orders`` broadcast over the leading axes, in one kernel call"""
    matrices = np.asarray(matrices, dtype=float)
    lead     = matrices.shape[:-2]
    if not int(np.prod(lead)):
        return np.zeros(lead + (3,), dtype=float)
    flat            = np.zeros((int(np.prod(lead)), 4, 4), dtype=float)
    flat[:, :3, :3] = matrices[..., :3, :3].reshape(-1, 3, 3)
    flat[:, 3, 3]   = 1.0
    axes            = np.broadcast_to(np.asarray(orders, dtype=np.int32), lead).reshape(-1)
    return np.degrees(matrix_to_euler(flat, axes)).reshape(lead + (3,))


def _chain_matrices(rotate_axis, rotate, joint_orient, orders) -> tuple:
    """RO, R and JO for (F, n, 3) channel blocks, ``orders`` per node of
    ``rotate``, from one kernel call"""
    first, count = rotate_axis.shape[1], rotate.shape[1]
    every = np.concatenate([rotate_axis, rotate, joint_orient], axis=1)
    axes = np.concatenate(
        [np.zeros(first, dtype=int), orders, np.zeros(joint_orient.shape[1], dtype=int)]
    )
    M = _eulers_to_matrices(every, axes[None])
    return M[:, :first], M[:, first:first + count], M[:, first + count:]


def _det3(block: np.ndarray) -> np.ndarray:
    """the determinants of a stack of (..., 3, 3) blocks"""
    return np.einsum(
        "...i,...i->...", block[..., 0, :], np.cross(block[..., 1, :], block[..., 2, :])
    )


def _rotation_local(M: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """the 3x3 block of ``matrix_local(M, parent)`` from their 3x3 blocks

    M times the inverse ``matrix_inverse`` computes, which takes the parent
    for scale times rotation: its transpose, each column divided by its
    source row's squared length. The block is all the euler and quaternion
    solves read of a local matrix.
    """
    lengths = np.einsum("...ij,...ij->...i", parent, parent)
    return M @ (np.swapaxes(parent, -1, -2) / lengths[..., None, :])


def _local_matrices(scale, RO, R, JO, inverse, translate) -> np.ndarray:
    """(..., 4, 4) local matrices from their factors

    S * RO * R * JO * IS * T, as TransformData.matrix composes them, with
    the diagonal and translation factors applied as the row and column
    scales and the row they are; ``inverse`` is the parent scale a
    compensated node divides out, 1 for any other.
    """
    block            = RO[..., :3, :3] @ R[..., :3, :3] @ JO[..., :3, :3]
    out              = np.zeros(block.shape[:-2] + (4, 4), dtype=float)
    out[..., :3, :3] = block * scale[..., :, None] * (1.0 / inverse)[..., None, :]
    out[..., 3, :3]  = translate
    out[..., 3, 3]   = 1.0
    return out


def _span_world(clip: "ClipData", span: slice) -> np.ndarray:
    """the world matrices of every node over the frames in ``span``

    :func:`_evaluate_local` and :func:`_evaluate_world` for a run of frames
    rather than all of them, so reading the frame a clip is bound to costs
    one frame, not the whole clip.
    """
    clip._realign()
    blocks     = clip._blocks
    nodes      = clip.list
    orders     = np.array([node.rotate_order for node in nodes], dtype=int)
    compensate = np.array([bool(n.segment_scale_compensate) for n in nodes], dtype=bool)
    parents    = clip._parent_indices(nodes)

    scale   = blocks["_scale"][span]
    inverse = np.where(compensate[:, None], _parent_scale(clip, parents, span), 1.0)

    RO, R, JO = _chain_matrices(
        blocks["_rotate_axis"][span],
        blocks["_rotate"][span],
        blocks["_joint_orient"][span],
        orders,
    )
    local = _local_matrices(scale, RO, R, JO, inverse, blocks["_translate"][span])
    return _evaluate_world(local, parents)


def _span_quaternions(clip: "ClipData", columns: np.ndarray, span: slice) -> np.ndarray:
    """(F, n, 4) orientations of ``columns`` over the frames in ``span``

    TransformData.quaternion for every frame at once: RO * R * JO.
    """
    blocks = clip._blocks
    orders = np.array([clip.list[c].rotate_order for c in columns], dtype=int)
    RO, R, JO = _chain_matrices(
        blocks["_rotate_axis"][span][:, columns],
        blocks["_rotate"][span][:, columns],
        blocks["_joint_orient"][span][:, columns],
        orders,
    )
    M = RO @ R @ JO
    return matrix_to_quaternion(M.reshape(-1, 4, 4)).reshape(M.shape[:2] + (4,))


def _filtered(degrees: np.ndarray, orders) -> np.ndarray:
    """(F, n, 3) euler degrees through euler_filter, column k in orders[k]

    Every later frame takes the solution nearest the one before it, the way
    Maya's Euler Filter keeps the first key. Only what the filter changes is
    added back, in degrees: a key it leaves alone, frame 0 always, comes
    back bit for bit as computed, where a round trip through radians would
    move some of them by an ulp.
    """
    degrees = np.asarray(degrees, dtype=float)
    if degrees.shape[0] < 2 or degrees.shape[1] == 0:
        return degrees.copy()

    radians = np.radians(degrees)
    turned  = euler_filter(radians, np.asarray(orders, dtype=int)) - radians
    out     = degrees + np.degrees(turned)
    out[0]  = degrees[0]
    return out


# a rotation whose middle euler angle is this close to +-90 degrees, its
# cosine below sin(0.01 degrees), is next to gimbal lock: matrix_to_euler
# snaps there, so the eulers hold a slightly different rotation
_GIMBAL_COSINE = np.sin(np.radians(1e-2))


def _solve_frames(clip, columns, span: slice, op: str, target, axes=None) -> None:
    """Writes a pose into nodes of a clip, every frame of ``span`` at once.

    The TransformData setters batched over frames, step for step, so each
    frame's channels come out as the single pose setter writes them:

    * ``"world"``: ``target`` holds (F, n, 4, 4) world matrices, as the
      world_matrix setter and match_matrix take them;
    * ``"local"``: (F, n, 4, 4) local matrices, the matrix setter;
    * ``"translate"``: (F, n, 3) world positions, match_translate, taking
      the components ``axes`` names;
    * ``"rotate"``: (F, n, 4, 4) world orientations, match_rotate;
    * ``"quaternion"``: (F, n, 4), the quaternion setter.

    ``columns`` are positions in the clip's node list, each once, one per
    target column. A node solves against its parent's final world, so nodes
    go parent first, one depth level at a time, the frames of a level
    together. An element whose local matrix already holds its target, to
    the setters' 1e-8, keeps its channels, as the matrix setter keeps them;
    a world target is tested on the local matrix it solves to, as a node
    with no world matrix cached tests it, so a node a fresh copy of the
    frame would solve is solved here too. Rotate rebuilt from a matrix picks
    its euler branch one frame at a time, so every node with a rebuilt
    rotate goes through euler_filter with its rotate order, the first frame
    kept as computed. The blocks are written, and the clip rebound, once.

    A rebuilt rotation is carried down the levels as a matrix and read off
    as eulers once at the end. Only one the eulers can not hold exactly --
    sheared, mirrored, or within 0.01 degrees of gimbal lock, where
    matrix_to_euler snaps -- makes the round trip on its own level, so its
    children solve under what the channels will hold, as they do frame by
    frame. The inverses of RO and of JO * IS are their transposes, scaled
    for IS; a parent's world takes a true inverse.
    """
    from cgmath.transforms._numba._euler import MAYA_EA

    clip._realign()
    if not clip._blocks:
        raise ValueError("the clip holds no frames to write")

    blocks  = clip._blocks
    nodes   = clip.list
    columns = np.asarray(columns, dtype=int)
    frames  = blocks["_scale"][span].shape[0]
    count   = len(columns)

    if op == "quaternion":
        # rotate alone, under the node's own rotate_axis and joint_orient:
        # no node solves against another, so this is one batched solve
        orders = np.array([nodes[c].rotate_order for c in columns], dtype=int)
        RO, _, JO = _chain_matrices(
            blocks["_rotate_axis"][span][:, columns],
            np.zeros((frames, 0, 3)),
            blocks["_joint_orient"][span][:, columns],
            orders[:0],
        )
        M = quaternion_to_matrix(np.reshape(target, (-1, 4))).reshape(frames, count, 4, 4)
        L = _rotation_local(
            np.swapaxes(RO[..., :3, :3], -1, -2) @ M[..., :3, :3], JO[..., :3, :3]
        )
        rotate                           = _matrices_to_eulers(L, orders)
        blocks["_rotate"][span, columns] = _filtered(rotate, orders)
        clip._bind(clip._frame)
        return

    # the nodes taking part: the targets and, for a world target, every
    # ancestor, whose world the targets solve under
    parents         = clip._parent_indices(nodes)
    depth           = _chain_depths(parents)
    taking          = np.zeros(len(nodes), dtype=bool)
    taking[columns] = True
    if op != "local":
        up = parents[columns]
        for _ in range(len(nodes)):
            up = np.unique(up[up >= 0])
            up = up[~taking[up]]
            if not len(up):
                break
            taking[up] = True
            up         = parents[up]

    rel         = np.flatnonzero(taking)
    slot        = np.full(len(nodes), -1, dtype=int)
    slot[rel]   = np.arange(len(rel))
    tpos        = slot[columns]
    which       = np.full(len(rel), -1, dtype=int)
    which[tpos] = np.arange(count)

    pcol   = parents[rel]
    prel   = np.where(pcol >= 0, slot[np.maximum(pcol, 0)], -1)
    orders = np.array([nodes[i].rotate_order for i in rel], dtype=int)
    compensate = np.array(
        [bool(nodes[i].segment_scale_compensate) for i in rel], dtype=bool
    )

    # working copies of the channels written; rotate_axis and joint_orient
    # are kept, as every setter keeps them
    scale  = blocks["_scale"][span][:, rel]
    rotate = blocks["_rotate"][span][:, rel]
    offset = blocks["_translate"][span][:, rel]
    RO, R, JO = _chain_matrices(
        blocks["_rotate_axis"][span][:, rel],
        rotate,
        blocks["_joint_orient"][span][:, rel],
        orders,
    )
    R = np.array(R)
    RO3, JO3 = RO[..., :3, :3], JO[..., :3, :3]

    # the parent scale a compensated node divides out: read as solved for a
    # parent taking part, as the block holds it for any other
    fixed = np.ones((frames, len(rel), 3), dtype=float)
    still = compensate & (pcol >= 0) & (prel < 0)
    if still.any():
        fixed[:, still] = blocks["_scale"][span][:, pcol[still]]
    live = compensate & (prel >= 0)

    eye     = np.eye(4)
    world   = None if op == "local" else np.empty((frames, len(rel), 4, 4))
    rebuilt = np.zeros(len(rel), dtype=bool)
    unread  = np.zeros((frames, len(rel)), dtype=bool)

    def rebuild(M, t, mask):
        # the rotations M (F, len(t), 3, 3) rotate of the nodes t is read off,
        # where mask says. a proper one is carried as M itself and read once
        # at the end; any other is whatever the eulers make of it, which its
        # children have to solve under, so it is read now
        block  = M / np.linalg.norm(M, axis=-1, keepdims=True)
        drift  = np.abs(block @ np.swapaxes(block, -1, -2) - np.eye(3)).max(axis=(-2, -1))
        proper = (drift <= 1e-12) & (_det3(block) > 0)

        # |sin| of the middle euler angle is element [a0, a2] of the block,
        # the first and last axes of the rotate order
        ea     = MAYA_EA[orders[t]]
        sine   = block[:, np.arange(len(t)), ea[:, 0], ea[:, 2]]
        proper = proper & (1.0 - sine * sine > _GIMBAL_COSINE**2)

        take              = proper & mask
        held              = R[:, t]
        held[..., :3, :3] = np.where(take[..., None, None], block, held[..., :3, :3])
        R[:, t]           = held
        unread[:, t]      = unread[:, t] | take

        bad = mask & ~proper
        if bad.any():
            f, j = np.nonzero(bad)
            n            = t[j]
            eulers       = _matrices_to_eulers(M[f, j], orders[n])
            rotate[f, n] = eulers
            R[f, n]      = _eulers_to_matrices(eulers, orders[n])
            unread[f, n] = False

        rebuilt[t[mask.any(axis=0)]] = True

    for level in np.unique(depth[rel]):
        lvl     = np.flatnonzero(depth[rel] == level)
        inverse = fixed[:, lvl]
        moved   = live[lvl]
        if moved.any():
            inverse[:, moved] = scale[:, prel[lvl[moved]]]

        local = _local_matrices(
            scale[:, lvl], RO[:, lvl], R[:, lvl], JO[:, lvl], inverse, offset[:, lvl]
        )
        if world is not None:
            PW           = np.broadcast_to(eye, (frames, len(lvl), 4, 4)).copy()
            under        = prel[lvl] >= 0
            PW[:, under] = world[:, prel[lvl[under]]]

        hit = np.flatnonzero(which[lvl] >= 0)
        if len(hit):
            t    = lvl[hit]
            k    = which[t]
            RO3t = np.swapaxes(RO3[:, t], -1, -2)

            if op == "rotate":
                # match_rotate: R = inv(RO) * M * inv(JO * IS * P), rotate only
                JP = (JO3[:, t] * (1.0 / inverse[:, hit])[..., None, :]) @ PW[:, hit, :3, :3]
                L  = _rotation_local(RO3t @ target[:, k, :3, :3], JP)
                rebuild(L, t, np.ones(L.shape[:2], dtype=bool))

            else:
                current = local[:, hit]
                if op == "local":
                    goal = np.asarray(target[:, k], dtype=float)
                    same = _same_transforms(goal, current)
                else:
                    parent = PW[:, hit]
                    now    = current @ parent
                    if op == "world":
                        aim = np.asarray(target[:, k], dtype=float)
                    else:
                        aim             = now.copy()
                        aim[..., 3, :3] = np.where(axes, target[:, k], now[..., 3, :3])

                    # a true inverse: a scaled parent's world is not orthogonal
                    goal = aim @ np.linalg.inv(parent)
                    same = _same_transforms(goal, current)

                todo = ~same
                if todo.any():
                    # the matrix setter, step for step: strip JO * IS from the
                    # right, split the [S][RO][R] block, rotate = inv(RO) * [RO][R]
                    B = goal[..., :3, :3] @ (
                        inverse[:, hit][..., :, None] * np.swapaxes(JO3[:, t], -1, -2)
                    )
                    s, rotation = _scales_and_rotations(
                        B, lambda: RO3[:, t] @ R[:, t, :3, :3]
                    )
                    scale[:, t]  = np.where(todo[..., None], s, scale[:, t])
                    offset[:, t] = np.where(todo[..., None], goal[..., 3, :3], offset[:, t])
                    rebuild(RO3t @ rotation, t, todo)

            # the solved nodes' local from what their channels now hold, for
            # the world their children solve under
            local[:, hit] = _local_matrices(
                scale[:, t], RO[:, t], R[:, t], JO[:, t], inverse[:, hit], offset[:, t]
            )

        if world is not None:
            world[:, lvl] = local @ PW

    f, n = np.nonzero(unread)
    if len(f):
        rotate[f, n] = _matrices_to_eulers(R[f, n], orders[n])

    # every frame picked its euler branch on its own: filter the curves
    done = tpos[rebuilt[tpos]]
    if len(done):
        rotate[:, done] = _filtered(rotate[:, done], orders[done])

    solved = {"_scale": scale, "_rotate": rotate, "_translate": offset}
    for channel in ("_rotate",) if op == "rotate" else solved:
        blocks[channel][span, columns] = solved[channel][:, tpos]
    clip._bind(clip._frame)


def _framed_clip(nodes) -> Optional["ClipData"]:
    """the multi-frame clip every one of ``nodes`` is the playback head of

    A node taken out of its clip with del or remove() still points at it,
    but has no column there: such a node is not a head, and the single pose
    route takes it.
    """
    nodes = list(nodes)
    clip  = None
    for node in nodes:
        owner = getattr(node, "_hierarchy", None)
        if clip is None:
            if not isinstance(owner, ClipData) or owner.frame_count < 2:
                return None
            clip = owner
        elif owner is not clip:
            return None

    # listed by identity: list membership would compare uuids node by node
    if clip is not None:
        listed = {id(node) for node in clip.list}
        if any(id(node) not in listed for node in nodes):
            return None
    return clip


def _framed_operand(other) -> tuple:
    """an operand of the pose algebra as ``(operand, every frame)``

    A clip's frame axis, ``clip.frames``, stands for every frame of its
    clip: it comes back as the clip, and True when the clip holds more than
    one frame. Anything else -- a clip itself included, the pose of the
    frame it is bound to -- is one pose and comes back as it is, with False.
    """
    if isinstance(other, _FrameAxis):
        if other._uuids is not None:
            raise TypeError(
                "the pose algebra takes a whole clip.frames, not one scoped to some nodes"
            )
        return other._clip, other._clip.frame_count > 1
    return other, False


def _holds_nodes(other) -> bool:
    """whether a match_* operand is nodes, a clip or a frame axis, rather
    than values"""
    if isinstance(other, (TransformData, TransformList, _FrameAxis)):
        return True
    return (
        isinstance(other, (list, tuple))
        and len(other) > 0
        and all(isinstance(item, TransformData) for item in other)
    )


def _clip_columns(clip: "ClipData", nodes) -> np.ndarray:
    """each node's position in the clip's node list"""
    place = {id(node): i for i, node in enumerate(clip.list)}
    return np.array([place[id(node)] for node in nodes], dtype=int)


def _last_of_each(columns: np.ndarray) -> tuple:
    """the columns without repeats, and which entry each one keeps: the last,
    which is what writing them in turn would leave"""
    columns = np.asarray(columns, dtype=int)
    _, first = np.unique(columns[::-1], return_index=True)
    keep = np.sort(len(columns) - 1 - first)
    return columns[keep], keep


def _orientations(world: np.ndarray) -> np.ndarray:
    """world matrices with their rows divided by their length, the
    orientation match_rotate takes from a node; the position stays"""
    world              = np.array(world, dtype=float)
    block              = world[..., :3, :3]
    world[..., :3, :3] = block / np.einsum("...ij,...ij->...i", block, block)[..., None] ** 0.5
    return world


def _rotation_targets(values: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """(F, n, 3) eulers in degrees in each node's rotate order, or (F, n, 4)
    quaternions, as (F, n, 4, 4) rotations: what match_rotate takes"""
    if values.shape[-1] == 3:
        return _eulers_to_matrices(values, np.asarray(orders)[None])
    frames, count = values.shape[:2]
    return quaternion_to_matrix(values.reshape(-1, 4)).reshape(frames, count, 4, 4)


def _clip_setter(nodes, op: str, value, axes=None) -> bool:
    """Runs a TransformList setter on the playback head of a multi-frame clip
    as a clip solve.

    The nodes of a clip are bound to its current frame, so a setter there
    writes that frame. Rather than solving node by node, the frame is solved
    the way ``clip.frames`` solves every frame, all the nodes in one solve
    and the clip rebound once, with the operand taken the way the
    TransformList setter takes it. Returns False, having done nothing, when
    the nodes are not all heads of one multi-frame clip, or when the operand
    is one only the single pose setter knows what to make of.

    A single node's own setters keep the single pose route: a write there
    clears the node's branch alone, where a solve rebinds the whole clip.
    """
    nodes = list(nodes)
    clip  = _framed_clip(nodes)
    if clip is None:
        return False

    count = len(nodes)
    if op in ("world", "local", "quaternion"):
        shape  = (4,) if op == "quaternion" else (4, 4)
        values = np.asarray(value, dtype=float)
        if _is_sequence(values, ndim=len(shape) + 1):
            values = values.reshape((count,) + shape)
        else:
            values = np.broadcast_to(values.reshape(shape), (count,) + shape)

    else:
        ndim     = 3 if op == "match" else 2
        operands = TransformList(nodes)._match_operands(value, ndim=ndim)
        if all(isinstance(o, TransformData) for o in operands):
            if (
                isinstance(value, ClipData)
                and value._blocks
                and len(operands) == len(value)
                and all(a is b for a, b in zip(operands, value.list))
            ):
                # a clip read at its own frame, every node at once
                frame = value.frame
                world = _span_world(value, slice(frame, frame + 1))[0]
            else:
                world = np.array([o.world_matrix for o in operands]).reshape(-1, 4, 4)
            if op == "rotate":
                values = _orientations(world)
            elif op == "translate":
                values = world[:, 3, :3]
            else:
                values = world

        elif op == "translate" and all(_is_sequence(o, ndim=1) for o in operands):
            values = np.array([np.asarray(o, dtype=float) for o in operands]).reshape(count, 3)

        elif op == "rotate" and all(_is_sequence(o, ndim=1) for o in operands):
            for o in operands:
                if len(o) not in (3, 4):
                    raise ValueError(
                        "match_rotate expects a length-3 euler (degrees) or "
                        f"length-4 quaternion sequence, got length {len(o)}"
                    )
            if len({len(o) for o in operands}) > 1:
                return False
            orders = np.array([node.rotate_order for node in nodes], dtype=int)
            raw    = np.array([np.asarray(o, dtype=float) for o in operands])
            values = _rotation_targets(raw[None], orders)[0]

        else:
            return False

    clip._realign()
    columns, keep = _last_of_each(_clip_columns(clip, nodes))
    frame = clip.frame
    _solve_frames(
        clip,
        columns,
        slice(frame, frame + 1),
        "world" if op == "match" else op,
        np.asarray(values)[keep][None],
        axes=None if axes is None else np.asarray(axes, dtype=bool),
    )
    return True


def _parent_scale(
    clip: "ClipData", parents: np.ndarray, span: slice = slice(None)
) -> np.ndarray:
    """every frame's parent scale as (F, N, 3), ones for a root

    What TransformData.parent_scale_inverse reads: the parent's scale,
    whether or not the node compensates for it. ``parents`` holds each
    node's parent as a position in the clip, -1 for a root; ``span`` picks
    the frames.
    """
    scale    = clip._blocks["_scale"][span]
    out      = np.ones(scale.shape, dtype=float)
    parented = parents >= 0
    if parented.any():
        out[:, parented] = scale[:, parents[parented]]
    return out


class _PoseFrame:
    """the pose some nodes hold, as the one frame clip the clip evaluators
    read: their ``list`` and ``(1, N, 3)`` blocks of the framed channels"""

    __slots__ = ("list", "_blocks")

    def __init__(self, nodes: list):
        self.list = nodes
        self._blocks = {
            channel: np.array(
                [getattr(node, channel) for node in nodes], dtype=float
            ).reshape(1, len(nodes), 3)
            for channel in ClipData.FRAMED_CHANNELS
        }


def _local_factors(
    clip: "ClipData", parents: Optional[np.ndarray] = None
) -> tuple:
    """every frame's local matrix factors, and each node's parent

    Returns S, RO, R, JO, IS and T, each (F, N, 4, 4) and built as the
    TransformData property of that factor builds it, then each node's
    parent as a position in the clip, -1 for a root. ``parents`` is the
    clip's parent map, when the caller already has it.
    """
    blocks = clip._blocks
    nodes  = clip.list

    orders = np.array([node.rotate_order for node in nodes], dtype=int)
    compensate = np.array(
        [bool(node.segment_scale_compensate) for node in nodes], dtype=bool
    )
    if parents is None:
        parents = clip._parent_indices(nodes)

    # the parent's scale is per frame, the flag that applies it is structural
    inverse = np.where(compensate[:, None], _parent_scale(clip, parents), 1.0)

    return _factors(blocks, orders, inverse) + (parents,)


def _factors(channels: dict, orders: np.ndarray, inverse: np.ndarray) -> tuple:
    """S, RO, R, JO, IS and T for (F, N, 3) channels, each (F, N, 4, 4)

    Each built as the TransformData property of that factor builds it.
    ``inverse`` is the parent scale each node's IS divides by, ones where
    segment scale compensation is off or there is no parent.
    """
    S  = _diagonal(channels["_scale"])
    RO = _euler_matrices(channels["_rotate_axis"],  None)
    R  = _euler_matrices(channels["_rotate"],       orders)
    JO = _euler_matrices(channels["_joint_orient"], None)
    IS = _diagonal(1.0 / inverse)
    T  = _translation(channels["_translate"])
    return S, RO, R, JO, IS, T


def _compose_local(channels: dict, orders: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """(F, N, 3) channels as (F, N, 4, 4) local matrices

    Mirrors TransformData.matrix factor for factor: S * RO * R * JO * IS * T,
    as :func:`_evaluate_local` does for a whole clip.
    """
    S, RO, R, JO, IS, T = _factors(channels, orders, inverse)
    return S @ RO @ R @ JO @ IS @ T


def _evaluate_local(
    clip: "ClipData", parents: Optional[np.ndarray] = None
) -> np.ndarray:
    """every frame's local matrices as (F, N, 4, 4)

    Mirrors TransformData.matrix factor for factor: S * RO * R * JO * IS * T.
    ``parents`` is the clip's parent map, when the caller already has it.
    """
    S, RO, R, JO, IS, T, _ = _local_factors(clip, parents)
    return S @ RO @ R @ JO @ IS @ T


def _evaluate_world(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """accumulates local matrices down the hierarchy, all frames at once"""
    world = np.empty_like(local)
    for i in TransformList._parent_first_order(parents):
        parent = parents[i]
        if parent < 0:
            world[:, i] = local[:, i]
        else:
            world[:, i] = local[:, i] @ world[:, parent]
    return world


def _looped(parents: np.ndarray) -> np.ndarray:
    """whether each entry of a parent array, ``parents[i]`` the position of
    i's parent, negative for none, reaches a loop going up: one on a loop,
    or below one

    Every entry at once, by pointer doubling: each pass links an entry to
    what its link linked to, twice as far up, so an entry k steps below a
    root links to none after k.bit_length() passes, and one on or below a
    loop never does. n entries take log2(n) passes at most, rounded up.
    """
    up = np.asarray(parents, dtype=int)
    up = np.where(up < 0, -1, up)
    for _ in range(max(len(up) - 1, 0).bit_length()):
        linked = up > -1
        if not linked.any():
            break
        up = np.where(linked, up[np.maximum(up, 0)], -1)
    return up > -1


def _pose_members(picked: list, owner, worlds: bool) -> Optional[tuple]:
    """``(members, parents)``: the nodes a pose read of ``picked``, nodes of
    ``owner``'s rig, evaluates, and each one's parent as a position among
    them, -1 for none. Each of ``picked`` and, with ``worlds``, every node
    above it, whose world matrices its own is made of, or without, its
    parent, whose scale it divides by when it compensates, a parent whose
    own parent is not read. A parent is the node get_parent gives, placed
    by the node itself, never by its uuid: of two nodes of one uuid, the
    first the rig lists; above another rig's node the rig lists, that rig's

    None, for the whole rig (:func:`_rig_members`), when ``picked`` hold
    half its nodes or more, where the walk up costs about what it saves,
    and when the rig has no index to walk up by.
    """
    lookups = _Lookups()
    index   = lookups.index_of(owner)
    if index is None or 2 * len(picked) >= len(index.nodes):
        return None

    # with worlds, a node met has every node above it met too: the walk up
    # stops at the first one met, a loop's included
    members, above, position = [], [], {}
    for node in picked:
        while isinstance(node, TransformData) and id(node) not in position:
            position[id(node)] = len(members)
            members.append(node)
            node = lookups.parent(node)
            above.append(node if isinstance(node, TransformData) else None)
            if not worlds:
                break
    if not worlds:
        for node in above[:]:
            if node is not None and id(node) not in position:
                position[id(node)] = len(members)
                members.append(node)
                above.append(None)

    parents = [-1 if node is None else position.get(id(node), -1) for node in above]
    return members, np.array(parents, dtype=int)


def _rig_members(picked: list, owner) -> tuple:
    """``(members, parents)`` for a pose read of ``picked``, nodes of
    ``owner``'s rig, evaluating the whole rig: every node it lists, each of
    ``picked`` it no longer lists, and every node above them it does not
    list, another rig's above that rig's node it lists; and each one's
    parent as a position among them, -1 for none

    A parent is the node get_parent gives, placed by the node itself, as
    :func:`_pose_members` places it; TransformList._parent_indices places
    one by its uuid, the last node carrying it.
    """
    lookups  = _Lookups()
    members  = [] if owner is None else list(owner.list)
    position = {id(node): i for i, node in enumerate(members)}
    for node in picked:
        if id(node) not in position:
            position[id(node)] = len(members)
            members.append(node)

    # members grows as it is read: a parent it does not hold joins it
    parents = []
    for node in members:
        parent = lookups.parent(node)
        if not isinstance(parent, TransformData):
            parents.append(-1)
            continue
        at = position.get(id(parent))
        if at is None:
            at = position[id(parent)] = len(members)
            members.append(parent)
        parents.append(at)
    return members, np.array(parents, dtype=int)


def _pose_matrices(nodes, worlds: bool = True, strict: bool = False) -> tuple:
    """``(local, world, parent)``: the ``(n, 4, 4)`` local, world and parent
    world matrices of ``nodes`` at the pose they hold, a clip's nodes at its
    frame; ``world`` and ``parent`` None without ``worlds``

    Each owning hierarchy is evaluated once, every node of it the read
    needs together, through the evaluators a clip's frames go through, so
    the matrices are the ones ``matrix``, ``world_matrix`` and
    ``get_parent_matrix()`` read node by node. Each node sits on the parent
    its ``get_parent`` gives, so a selection reads what it reads node by
    node: one under half its rig evaluates the part of the rig its matrices
    are made of (:func:`_pose_members`), a larger one the whole rig
    (:func:`_rig_members`). A node no hierarchy owns is a root, as
    ``get_parent`` has it, a parent the hierarchy does not hold is none,
    another rig's node a rig lists brings the nodes above it in that rig,
    and a node its owner no longer lists sits on the parent its
    ``get_parent`` gives, as any node does. With ``strict``, a node whose
    parents loop raises RecursionError, as its world_matrix does; without,
    as a clip's frames evaluate, its matrices are not worked out.
    """
    nodes  = list(nodes)
    count  = len(nodes)
    local  = np.empty((count, 4, 4), dtype=float)
    world  = np.empty((count, 4, 4), dtype=float) if worlds else None
    parent = np.empty((count, 4, 4), dtype=float) if worlds else None
    owners = {}
    for i, node in enumerate(nodes):
        owners.setdefault(id(node._hierarchy), []).append(i)

    for rows in owners.values():
        owner  = nodes[rows[0]]._hierarchy
        picked = [nodes[i] for i in rows]
        found  = None if owner is None else _pose_members(picked, owner, worlds)
        members, parents = _rig_members(picked, owner) if found is None else found

        position    = {id(node): j for j, node in enumerate(members)}
        at          = np.array([position[id(node)] for node in picked], dtype=int)
        evaluated   = _evaluate_local(_PoseFrame(members), parents)
        local[rows] = evaluated[0, at]
        if worlds and strict:
            # a node whose parents loop has no world matrix: its own read
            # raises the RecursionError naming the loop
            looped = _looped(parents)[at]
            if looped.any():
                picked[int(np.flatnonzero(looped)[0])].world_matrix
        if worlds:
            evaluated   = _evaluate_world(evaluated, parents)[0]
            above       = parents[at]
            world[rows] = evaluated[at]
            parent[rows] = np.where(
                (above > -1)[:, None, None], evaluated[np.maximum(above, 0)], np.eye(4)
            )

    return local, world, parent


# --- rig edits, every frame at once --------------------------------------- #
# A rig edit -- swapaxes, the orient helpers, set_parent in world space, a
# joint becoming another type, set_rotate_order -- rebuilds channels out of
# matrices. On a clip every frame is solved in one batch, through the same
# steps the single pose methods take, and each rebuilt euler curve then goes
# through euler_filter in its own rotate order, frame 0 kept as computed,
# the way Maya's Euler Filter keeps the first key. Solving frames one at a
# time instead lets each pick its euler branch on its own: the poses stay
# exact while the curves step by 180 and 360 degrees.

_ROTATE_ORDERS = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")


def _rotate_order_index(order) -> int:
    """a rotate order as 0..5, given as that number or as its axes, eg: "zyx" """
    if isinstance(order, str):
        if order.lower() in _ROTATE_ORDERS:
            return _ROTATE_ORDERS.index(order.lower())

    elif not isinstance(order, (bool, np.bool_)):
        try:
            value = int(order)
        except (TypeError, ValueError):
            value = None
        if value is not None and value == order and 0 <= value < len(_ROTATE_ORDERS):
            return value

    raise ValueError(
        f"a rotate order is 0..5 or one of {', '.join(_ROTATE_ORDERS)}, got {order!r}"
    )


def _reorder_like_maya(rotate: np.ndarray, before, after) -> np.ndarray:
    """(..., 3) euler degrees in rotate order ``before``, re-expressed in ``after``

    A rotation has two euler solutions in any order, plus whole turns on any
    axis. This picks the one Maya's ``xform -rotateOrder <order> -preserve``
    picks: euler_reorder's solution with every angle in [-180, 180), or its
    alternate -- the first and last axes applied turned by 180, the middle
    one mirrored to 180 minus itself -- when that sums to fewer degrees, the
    first one on a tie. ``before`` and ``after`` broadcast against the
    leading axes of ``rotate``.
    """
    from cgmath.transforms._numba._euler import MAYA_EA

    rotate    = np.asarray(rotate, dtype=float)
    shape     = rotate.shape
    flat      = rotate.reshape(-1, 3)
    before    = np.broadcast_to(np.asarray(before, dtype=int), shape[:-1]).reshape(-1)
    after     = np.broadcast_to(np.asarray(after, dtype=int), shape[:-1]).reshape(-1)

    standard  = np.degrees(euler_reorder(np.radians(flat), before, after))
    standard  = (standard + 180.0) % 360.0 - 180.0

    axes      = MAYA_EA[after]
    rows      = np.arange(len(flat))
    alternate = standard.copy()
    alternate[rows, axes[:, 0]] += 180.0
    alternate[rows, axes[:, 2]] += 180.0
    alternate[rows, axes[:, 1]] = 180.0 - standard[rows, axes[:, 1]]

    shorter = np.abs(alternate).sum(axis=-1) < np.abs(standard).sum(axis=-1)
    return np.where(shorter[:, None], alternate, standard).reshape(shape)


def _solve_local(target: np.ndarray, channels: dict, orders, inverse: np.ndarray, gap: bool = False) -> tuple:
    """TransformData.matrix's setter for (F, n) frames and nodes at once

    ``target`` holds the (F, n, 4, 4) local matrices to reach, ``channels``
    the (F, n, 3) channels the nodes hold now, and ``inverse`` the parent
    scale each one's IS divides by. Scale, rotate and translate are solved
    and rotate_axis, joint_orient and segment_scale_compensate kept, step for
    step as the setter does, so a frame whose channels already make the
    target, to round-off, keeps them as the setter would.

    Returns the new (scale, rotate, translate), each (F, n, 3), and with
    ``gap`` the largest entry the keep test found between the channels'
    matrix and the target, (F, n).
    """
    shape = target.shape[:-2]
    S, RO, R, JO, IS, T = _factors(channels, orders, inverse)
    composed = S @ RO @ R @ JO @ IS @ T
    same     = _same_transforms(composed, target)

    # strip JO * IS from the right, split [S][RO][R] into scale and [RO][R],
    # then R = inv(RO) * [RO][R]: rotate_axis is kept
    B = target @ np.linalg.inv(JO @ IS)
    scale, rotation = _scales_and_rotations(
        B[..., :3, :3], lambda: (RO @ R)[..., :3, :3]
    )
    RR              = np.zeros(shape + (4, 4), dtype=float)
    RR[..., :3, :3] = rotation
    RR[..., 3, 3]   = 1.0
    rotate          = _matrices_to_euler(np.linalg.inv(RO) @ RR, np.asarray(orders))

    keep = same[..., None]
    solved = (
        np.where(keep, channels["_scale"], scale),
        np.where(keep, channels["_rotate"], rotate),
        np.where(keep, channels["_translate"], target[..., 3, :3]),
    )
    if gap:
        return solved + (np.abs(composed - target).max(axis=(-2, -1)),)
    return solved


def _store_channel(node: "TransformData", name: str, value: np.ndarray) -> None:
    """writes a (3,) channel the way _write_channel does, minus the cache reset

    A batched edit drops every cached world matrix once at the end, instead
    of once per write over each node's whole branch.
    """
    current = getattr(node, name)
    if current is getattr(type(node), name, None):
        setattr(node, name, np.array(value, dtype=float))
    else:
        current[...] = value


class _RowValues:
    """one value per row of a :class:`_RigFrames`, read off its node the
    first time it is asked for, so an edit of a few nodes reads a few"""

    __slots__ = ("_read", "_held", "_dtype")

    def __init__(self, read, dtype):
        self._read  = read
        self._held  = {}
        self._dtype = dtype

    def __getitem__(self, rows):
        if np.ndim(rows) == 0:
            row   = int(rows)
            value = self._held.get(row)
            if value is None:
                value = self._held[row] = self._read(row)
            return value
        return np.array([self[row] for row in np.asarray(rows, dtype=int).tolist()], dtype=self._dtype)

    def __setitem__(self, rows, values) -> None:
        rows   = np.atleast_1d(np.asarray(rows, dtype=int))
        values = np.broadcast_to(np.asarray(values, dtype=self._dtype), rows.shape)
        for row, value in zip(rows.tolist(), values.tolist()):
            self._held[row] = value


class _RigFrames:
    """every frame of one hierarchy's framed channels, for a batched rig edit

    A clip's (F, N, 3) blocks are read and written in place through their
    uuid columns, so nothing is realigned or copied wholesale; a hierarchy
    without frames is the same thing with one frame, read from and written
    to the nodes. Rows are positions in the owner's node list.

    An edit pays for the rows it touches, not for the rig: a node's row and
    its parent come from the index the hierarchy keeps, its rotate order and
    compensation are read when asked for, its column from the map the clip
    keeps, and :meth:`finish` clears the branches of the rows written, so a
    loop of one node edits costs the nodes it edits. ``snapshot`` keeps the
    rows of the list as it is now, for an edit that reorders it.
    """

    def __init__(self, owner: "HierarchyData", snapshot: bool = False):
        self.owner   = owner
        self.nodes   = list(owner.list) if snapshot else owner.list
        self.frames  = owner.frame_count if getattr(owner, "_blocks", None) else 1
        self.lookups = _Lookups()
        self.orders  = _RowValues(lambda row: int(self.nodes[row].rotate_order), int)
        self.compensate = _RowValues(
            lambda row: bool(self.nodes[row].segment_scale_compensate), bool
        )
        self._index     = self.lookups.index_of(owner)
        self._place     = None   # id(node) -> row, built only when the index can not say
        self._parents   = None   # every row's parent, once the whole map is asked for
        self._shared    = False  # whether _parents is the owner's, to copy before a write
        self._parent_of = {}
        self._touched   = set()
        if snapshot:
            self.parents  # the rows of the list as it is now

    # --- rows --- #
    def row_of(self, node) -> Optional[int]:
        """the row of ``node``, None when the owner does not list it"""
        index = self._index
        if index is not None and isinstance(node.uuid, str):
            row = index.uuids.get(node.uuid)
            if row is not None and row < len(self.nodes) and self.nodes[row] is node:
                return row

        if self._place is None:
            self._place = {id(n): i for i, n in enumerate(self.nodes)}
        return self._place.get(id(node))

    def rows_of(self, nodes) -> list:
        """the rows of ``nodes``, every one of which the owner lists"""
        return [self.row_of(node) for node in nodes]

    @property
    def parents(self) -> np.ndarray:
        """every row's parent row, -1 for a root"""
        if self._parents is None:
            owner = self.owner
            if isinstance(owner, HierarchyData) and owner.list is self.nodes:
                self._parents, self._shared = owner._parent_rows(), True
            else:
                self._parents = TransformList._parent_indices(self.nodes)
        return self._parents

    def parent_row(self, row: int) -> int:
        """the parent row of one row, -1 for a root"""
        if self._parents is not None:
            return int(self._parents[row])

        up = self._parent_of.get(row)
        if up is None:
            parent = self.lookups.parent(self.nodes[row])
            found  = self.row_of(parent) if isinstance(parent, TransformData) else None
            up     = self._parent_of[row] = -1 if found is None else found
        return up

    def set_parents(self, rows, parent: int) -> None:
        """the rows now hang under row ``parent``, -1 for none"""
        parents = self.parents
        if self._shared:
            self._parents, self._shared = parents.copy(), False
        self._parents[np.asarray(rows, dtype=int)] = parent

    def children_rows(self, row: int) -> list:
        """the rows of a row's children"""
        found = (self.row_of(child) for child in self.lookups.children(self.nodes[row]))
        return [child for child in found if child is not None]

    # --- channels --- #
    def _columns_of(self, rows: np.ndarray) -> np.ndarray:
        owner = self.owner
        place = owner._column_place()
        nodes = self.nodes
        try:
            return np.array([place[nodes[r].uuid] for r in rows.tolist()], dtype=int)
        except KeyError:
            # a node added since the blocks were last lined up has no column
            owner._realign()
            place = owner._column_place()
            return np.array([place[nodes[r].uuid] for r in rows.tolist()], dtype=int)

    def read(self, channel: str, rows) -> np.ndarray:
        """(F, n, 3) copy of one channel"""
        rows = np.asarray(rows, dtype=int)
        if getattr(self.owner, "_blocks", None):
            return self.owner._blocks[channel][:, self._columns_of(rows)]

        values = [getattr(self.nodes[i], channel) for i in rows]
        return np.array(values, dtype=float).reshape(1, len(rows), 3)

    def channels(self, rows) -> dict:
        return {c: self.read(c, rows) for c in ClipData.FRAMED_CHANNELS}

    def write(self, channel: str, rows, values: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=int)
        self._touched.update(rows.tolist())
        if getattr(self.owner, "_blocks", None):
            self.owner._blocks[channel][:, self._columns_of(rows)] = values
        else:
            for k, i in enumerate(rows):
                _store_channel(self.nodes[i], channel, values[0, k])

    def inverse(self, rows) -> np.ndarray:
        """(F, n, 3): the parent scale each row's IS divides by"""
        rows    = np.asarray(rows, dtype=int)
        parents = np.array([self.parent_row(r) for r in rows.tolist()], dtype=int)
        out     = np.ones((self.frames, len(rows), 3), dtype=float)
        on      = (parents > -1) & self.compensate[rows]
        if on.any():
            out[:, on] = self.read("_scale", parents[on])
        return out

    def world(self, rows) -> np.ndarray:
        """(F, n, 4, 4) world matrices, evaluated over the rows' chains alone"""
        rows  = [int(r) for r in rows]
        need  = []
        depth = {}
        for r in rows:
            chain = []
            while r > -1 and r not in depth and len(chain) <= len(self.nodes):
                chain.append(r)
                r = self.parent_row(r)
            base = depth.get(r, -1)
            for count, member in enumerate(reversed(chain), 1):
                depth[member] = base + count
            need.extend(chain)

        need  = sorted(need, key=depth.__getitem__)
        at    = {r: m for m, r in enumerate(need)}

        local = _compose_local(self.channels(need), self.orders[need], self.inverse(need))
        world = np.empty_like(local)
        for m, r in enumerate(need):
            parent = self.parent_row(r)
            if parent in at:
                world[:, m] = local[:, m] @ world[:, at[parent]]
            else:
                world[:, m] = local[:, m]

        return world[:, [at[r] for r in rows]]

    def finish(self) -> None:
        """drops the cached world matrices of every row written and of every
        node below them: the edit may have moved them"""
        _reset_branches([self.nodes[r] for r in sorted(self._touched)])
        self._touched.clear()


def _clip_frames(owner) -> Optional[_RigFrames]:
    """the frames of a clip, for an edit that has to cover all of them;
    None for anything that holds one pose"""
    if owner is not None and getattr(owner, "_blocks", None):
        return _RigFrames(owner)
    return None


def _clip_row(node) -> tuple:
    """(frames, row) of a node of a clip, for an edit that covers every
    frame; (None, None) for a node of one pose, and for a node its clip no
    longer lists (taken out with del or remove()), which the single pose
    route takes"""
    frames = _clip_frames(node._hierarchy)
    if frames is not None:
        row = frames.row_of(node)
        if row is not None:
            return frames, row
    return None, None


def _held_node(owner, query) -> "TransformData":
    """``owner[owner.index(query)]``, the node given as a node, a uuid or a
    name (the first node carrying it), read from the index ``owner`` keeps
    while it is current, where the scan costs the hierarchy's size.
    ValueError when no node matches"""
    index = _kept(owner)
    if index is not None:
        position = None
        if isinstance(query, TransformData):
            if isinstance(query.uuid, str):
                position = index.uuids.get(query.uuid)
        elif isinstance(query, str):
            position = index.uuids.get(query) if _is_uuid(query) else index.named(query)
        if position is not None:
            return index.nodes[position]
    return owner[owner.index(query)]


def _refuse_a_cycle(nodes, above) -> None:
    """raises ValueError when ``above``, once ``nodes`` are taken to the
    world, would hang below one of them: parenting that node under it
    would close a loop. Nothing hangs below nodes without children, which
    a rig built node by node parents: no climb"""
    chosen = {id(node) for node in nodes}
    if id(above) in chosen:
        return

    lookups = _Lookups()
    if not any(lookups.children(node) for node in nodes):
        return

    seen = set()
    up   = lookups.parent(above)
    while isinstance(up, TransformData) and id(up) not in seen:
        if id(up) in chosen:
            raise ValueError(
                f"{up.name!r} can not go under {above.name!r}, which hangs below it"
            )
        seen.add(id(up))
        up = lookups.parent(up)


# _to_the_end's ref when the nodes keep the parent_uuid they hold
_KEEP_PARENT = object()


def _to_the_end(owner, nodes, ref=_KEEP_PARENT) -> list:
    """moves the branch of each of ``nodes`` to the end of ``owner``'s list,
    in one pass, where moving one node after another would put it

    ``ref``, when given, is first written as each node's parent_uuid. A
    node lands where its last move puts it and its branch follows it in
    walk order, every node before its children; the nodes the moves leave
    alone keep their order. Returns the nodes moved, in their new order.

    Branches already at the end, in that order -- a node added and then
    parented, as a rig is built node by node -- leave the list as it is,
    and the index the hierarchy keeps with it.
    """
    index = _kept(owner)
    if index is None or any(index.position(node) is None for node in nodes):
        listed = {id(node) for node in owner.list}
        nodes  = [node for node in nodes if id(node) in listed]
    if ref is not _KEEP_PARENT:
        for node in nodes:
            node.parent_uuid = ref

    # each node where its last move puts it: in the order of its last
    # occurrence, its branch walked as it stands once every link is written
    last  = {id(node): (i, node) for i, node in enumerate(nodes)}
    roots = [node for _, node in sorted(last.values(), key=lambda pair: pair[0])]
    moved = _walk(roots, _Lookups())

    # a node listed here but owned by another rig has its children looked
    # up in that rig: only the nodes this list holds move
    if any(node._hierarchy is not owner for node in moved):
        listed = {id(node) for node in owner.list}
        moved  = [node for node in moved if id(node) in listed]

    start = len(owner.list) - len(moved)
    if start >= 0 and all(a is b for a, b in zip(owner.list[start:], moved)):
        return moved

    gone          = {id(node) for node in moved}
    owner.list[:] = [node for node in owner.list if id(node) not in gone] + moved
    _structure_changed(owner)
    return moved


def _named_after_the_move(owner, nodes, parent):
    """the node ``parent`` names once ``nodes`` are taken to the world and
    their branches moved to the end of the list, as :func:`_to_the_end`
    moves them, worked out without moving anything

    A uuid, or a name only one node carries, names the same node whatever
    the order. A name several nodes carry names the first of them: the
    first one no branch moves, or when every one of them moves, the first
    to land.
    """
    position = owner.index(parent)  # an unknown parent raises here
    named    = owner[position]
    if not isinstance(parent, str) or validate_uuid(parent):
        return named
    carriers = [node for node in owner.list if node.name == parent]
    if len(carriers) < 2:
        return named

    # the branches as they will walk once every node is a root: a node's
    # chosen child is walked as a root of its own
    listed  = {id(node) for node in owner.list}
    chosen  = {id(node) for node in nodes}
    lookups = _Lookups()
    last    = {id(node): (i, node) for i, node in enumerate(nodes) if id(node) in listed}
    order, seen = [], set()
    for _, root in sorted(last.values(), key=lambda pair: pair[0]):
        if id(root) in seen:
            continue
        seen.add(id(root))
        order.append(root)
        stack = [iter(lookups.children(root))]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
            elif id(child) not in seen and id(child) not in chosen:
                seen.add(id(child))
                order.append(child)
                stack.append(iter(lookups.children(child)))

    for node in carriers:
        if id(node) not in seen:
            return node
    return next(node for node in order if node.name == parent)


def _reparent(owner, nodes, parent, world_space: bool) -> None:
    """TransformList.set_parent for nodes of one hierarchy, all at once

    The result one node at a time gives, in one pass over the rig: every
    node to the world, the list reordered once; ``parent`` resolved then,
    as each later call resolves it; every other node under it, the list
    reordered once more. In world space a frame per pose of a hierarchy
    without frames, every frame of a clip, solved at once.
    """
    named = None
    if parent is not None:
        named = _named_after_the_move(owner, nodes, parent)
        _refuse_a_cycle(nodes, named)
        _uuid_of(named)  # a parent with no uuid to take raises before anything moves

    frames = _RigFrames(owner, snapshot=True) if world_space else None
    if frames is not None:
        rows    = np.array(frames.rows_of(nodes), dtype=int)
        parents = frames.parents.copy()
        rotate  = frames.read("_rotate", rows)
        above   = -1 if named is None else frames.row_of(named)

        # every node to the world
        worlds = frames.world(rows)
        first  = _solve_under(frames, rows, worlds, -1, gap=True)
        first, gap = first[:3], first[3]
        if _cascades(frames, rows, parents, worlds, first, above, gap):
            # one at a time, each reading what the ones before it wrote
            frames.set_parents(rows, parents[rows])
            _to_the_world_in_turn(frames, rows, parents, worlds)
        else:
            _write_solved(frames, rows, first)

    moved = _to_the_end(owner, nodes, None)

    # then under the parent, as the list now resolves it, the parent itself
    # left at the world
    if parent is not None:
        above = owner[owner.index(parent)]
        _refuse_a_cycle(nodes, above)
        rest = [node for node in nodes if node is not above]
        if rest:
            if frames is not None:
                under  = np.array(frames.rows_of(rest), dtype=int)
                solved = _solve_under(frames, under, frames.world(under), frames.row_of(above))
                _write_solved(frames, under, solved)
            moved = moved + _to_the_end(owner, rest, above.uuid)

    if frames is not None:
        _filter_rotate(frames, rows, rotate)
        frames.finish()
    else:
        _reset_branches(moved)


def _by_owner(nodes) -> list:
    """``nodes`` grouped by the hierarchy that owns them, in first-seen order:
    [(owner, nodes)], owner None for nodes no hierarchy lists"""
    groups, members = {}, {}
    for node in nodes:
        owner = node._hierarchy
        if owner is not None:
            # by identity: list membership would compare uuids node by node
            if id(owner) not in members:
                members[id(owner)] = {id(n) for n in owner.list}
            if id(node) not in members[id(owner)]:
                owner = None

        key = None if owner is None else id(owner)
        groups.setdefault(key, (owner, []))[1].append(node)
    return list(groups.values())


def _swap_rows(frames: _RigFrames, targets, axis0: int, axis1: int, negate: bool) -> None:
    """TransformData.swapaxes on every target row, every frame at once

    A swapped node's world becomes P * W and each direct child keeps its
    world, whatever order the nodes would have been swapped in. A child that
    is swapped too is put back first and then swapped, as a parent-first pass
    does it: its channels are those of the single pose methods to round-off.
    """
    P                    = _axis_permutation(axis0, axis1, negate)
    Pi                   = P.T  # a signed permutation is orthogonal, so the inverse is the transpose
    swap                 = np.arange(3)
    swap[[axis0, axis1]] = [axis1, axis0]

    targets = np.unique(np.asarray(targets, dtype=int))
    under = np.array(
        sorted({child for t in targets.tolist() for child in frames.children_rows(t)}),
        dtype=int,
    )

    rows   = np.union1d(targets, under)
    at     = {r: m for m, r in enumerate(rows)}
    t      = np.array([at[r] for r in targets], dtype=int)
    k      = np.array([at[r] for r in under], dtype=int)
    orders = frames.orders[rows]
    xyz    = np.zeros(len(t), dtype=int)
    ch     = frames.channels(rows)
    world  = frames.world(rows)

    moved       = world.copy()
    moved[:, t] = P @ world[:, t]

    # children first: each keeps its world under the swapped parent, solved
    # as world_matrix's setter solves it, the parent's IS scale swapped too
    if len(k):
        up      = np.array([at[frames.parent_row(rows[i])] for i in k], dtype=int)
        inverse = np.ones((frames.frames, len(k), 3), dtype=float)
        on      = frames.compensate[rows[k]]
        if on.any():
            inverse[:, on] = ch["_scale"][:, up[on]][..., swap]

        local = world[:, k] @ np.linalg.inv(moved[:, up])
        kids  = {c: v[:, k] for c, v in ch.items()}
        S, R, T = _solve_local(local, kids, orders[k], inverse)
        ch["_scale"][:, k], ch["_rotate"][:, k], ch["_translate"][:, k] = S, R, T

    # the swap: scale permutes exactly, RO and R are conjugated and JO left
    # multiplied, which reproduces P * [RO][R][JO]
    RO = _euler_matrices(ch["_rotate_axis"][:, t],  None)
    R  = _euler_matrices(ch["_rotate"][:, t],       orders[t])
    JO = _euler_matrices(ch["_joint_orient"][:, t], None)

    ch["_scale"][:, t]        = ch["_scale"][:, t][..., swap]
    ch["_rotate_axis"][:, t]  = _filtered(_matrices_to_euler(P @ RO @ Pi, xyz), xyz)
    ch["_joint_orient"][:, t] = _filtered(_matrices_to_euler(P @ JO, xyz), xyz)
    ch["_rotate"][:, t]       = _matrices_to_euler(P @ R @ Pi, orders[t])
    ch["_rotate"]             = _filtered(ch["_rotate"], orders)

    for channel in ("_scale", "_rotate", "_translate"):
        frames.write(channel, rows, ch[channel])
    frames.write("_rotate_axis", targets, ch["_rotate_axis"][:, t])
    frames.write("_joint_orient", targets, ch["_joint_orient"][:, t])
    frames.finish()


def _orient_rows(frames: _RigFrames, rows, into_rotate: bool) -> None:
    """set_joint_orient_to_rotate (``into_rotate``) or set_rotate_to_joint_orient
    on every joint row, every frame at once"""
    rows = np.array([r for r in rows if frames.nodes[r].node_type == "joint"], dtype=int)
    if not len(rows):
        return

    count = len(rows)
    ch    = frames.channels(rows)
    chain = (
        _euler_matrices(ch["_rotate_axis"], None)
        @ _euler_matrices(ch["_rotate"], frames.orders[rows])
        @ _euler_matrices(ch["_joint_orient"], None)
    )
    # through a quaternion, as TransformData.quaternion reads the chain
    Q     = matrix_to_quaternion(chain.reshape(-1, 4, 4))
    zeros = np.zeros_like(ch["_rotate"])

    if into_rotate:
        orders = frames.orders[rows]
        euler  = quaternion_to_euler(Q, np.tile(orders, frames.frames))
        rotate = _filtered(np.degrees(euler).reshape(-1, count, 3), orders)
        frames.write("_joint_orient", rows, zeros)
        frames.write("_rotate_axis",  rows, zeros)
        frames.write("_rotate",       rows, rotate)
    else:
        xyz    = np.zeros(count, dtype=int)
        euler  = quaternion_to_euler(Q, 0)
        orient = _filtered(np.degrees(euler).reshape(-1, count, 3), xyz)
        frames.write("_rotate",       rows, zeros)
        frames.write("_rotate_axis",  rows, zeros)
        frames.write("_joint_orient", rows, orient)

    frames.finish()


def _fold_orient_rows(frames: _RigFrames, rows) -> None:
    """a joint becoming another type: joint_orient folded into rotate on
    every frame that has one, every frame at once"""
    rows   = np.asarray(rows, dtype=int)
    orient = frames.read("_joint_orient", rows)
    folded = np.any(orient != 0.0, axis=-1)
    if not folded.any():
        return

    orders = frames.orders[rows]
    rotate = frames.read("_rotate", rows)
    RJ     = _euler_matrices(rotate, orders) @ _euler_matrices(orient, None)
    rotate = np.where(folded[..., None], _matrices_to_euler(RJ, orders), rotate)

    # only a curve with something folded into it was rebuilt
    hit            = folded.any(axis=0)
    rotate[:, hit] = _filtered(rotate[:, hit], orders[hit])

    frames.write("_rotate", rows, rotate)
    frames.write("_joint_orient", rows, np.zeros_like(orient))
    frames.finish()


def _solve_under(frames: _RigFrames, rows, worlds: np.ndarray, parent: int, gap: bool = False) -> tuple:
    """set_parent in world space, every frame at once: ``rows`` now hang under
    row ``parent`` (-1 for none), and the (scale, rotate, translate) that keep
    their ``worlds`` (F, n, 4, 4) there are solved as world_matrix's setter
    solves them, and with ``gap`` the keep test's gap (:func:`_solve_local`).
    Nothing is written."""
    rows = np.asarray(rows, dtype=int)
    frames.set_parents(rows, parent)

    target = worlds
    if parent > -1:
        # a true inverse: a scaled parent's world is not orthogonal
        target = worlds @ np.linalg.inv(frames.world([parent]))

    channels = frames.channels(rows)
    return _solve_local(target, channels, frames.orders[rows], frames.inverse(rows), gap)


def _to_the_world_in_turn(frames: _RigFrames, rows, parents: np.ndarray, worlds: np.ndarray) -> None:
    """set_parent(None) in world space on ``rows`` one at a time, as the
    single pose method does it, every frame at once: each row keeps the
    world it has when its turn comes, which the rows before it may have
    moved. ``parents`` is the parenting before the edit, ``worlds`` (F, n,
    4, 4) the rows' worlds then.

    A row's world at its turn is its chain up to w, the lowest ancestor
    solved before it, which is a root by then: the locals from the row up
    to c, w's child on the way, as they were (nothing between was solved
    yet), c's local under w's new scale (c may compensate it), and w's new
    local. Without such an ancestor it is the world it had. So nothing is
    climbed twice, however the selection runs: one walk down the rig finds
    every row's w, the solved ancestors on the way kept in the order they
    are solved, and the products of the locals as they were, from each row
    up to its c, come out of a union-find with path compression as the
    walk leaves each c. Then the rows are solved in turn, each over every
    frame at once.
    """
    rows  = [int(r) for r in rows]
    first = {}
    for turn, row in enumerate(rows):
        first.setdefault(row, turn)

    # the rig the rows hang in: their chains to the root, each walked once
    held = set()
    for row in first:
        node = row
        while node > -1 and node not in held:
            held.add(node)
            node = int(parents[node])
    below = {}
    for node in sorted(held):
        below.setdefault(int(parents[node]), []).append(node)

    # their locals as they are. A new parent scale changes only the IS a
    # compensated child divides by, so the factors around it are kept for
    # the children that need them once the walk below has named them
    order = sorted(held)
    place = {node: k for k, node in enumerate(order)}
    S, RO, R, JO, IS, T = _factors(frames.channels(order), frames.orders[order], frames.inverse(order))
    head  = S @ RO @ R @ JO
    local = head @ IS @ T
    del S, RO, R, JO, IS

    # every row's w and c, walking down: the solved ancestors on the way, in
    # the order they are solved, each one leaving out the ones above it that
    # are solved after it (a row below them meets it first). turns[:top] and
    # kept[:top] are that list; a node pushed on it is taken off as the walk
    # leaves the node
    anchor  = {}  # row -> (w, c)
    asked   = {}  # c -> the rows whose product up to c is wanted
    product = {}  # row -> the locals as they were from the row up to its c, c left out
    turns   = []
    kept    = []
    top     = 0
    depth   = {}
    path    = []
    link    = {}  # union-find: node -> the node its product runs up to
    reach   = {}  # node -> the product of locals from it up to link[node], that one left out

    stack   = [(root, None) for root in reversed(below.get(-1, []))]
    while stack:
        node, undo = stack.pop()
        if undo is not None:
            # the walk leaves the node: its children's chains now run up to
            # it, so every product wanted up to it is complete
            for child in below.get(node, []):
                link[child]  = node
                reach[child] = local[:, place[child]]
            for row in asked.get(node, []):
                if row != node:
                    _compress(link, reach, row)
                    product[row] = reach[row]
            path.pop()
            at, was, top = undo
            if was is not None:
                turns[at], kept[at] = was
            continue

        depth[node] = len(path)
        path.append(node)
        undo = (0, None, top)
        if node in first:
            turn = first[node]
            at   = bisect.bisect_left(turns, turn, 0, top)
            if at:
                w            = kept[at - 1]
                c            = path[depth[w] + 1]
                anchor[node] = (w, c)
                asked.setdefault(c, []).append(node)
            if at < len(turns):
                undo = (at, (turns[at], kept[at]), top)
                turns[at], kept[at] = turn, node
            else:
                undo = (at, None, top)
                turns.append(turn)
                kept.append(node)
            top = at + 1
        stack.append((node, undo))
        stack.extend((child, None) for child in reversed(below.get(node, [])))

    # the factors of each compensated child of a solved node, the full
    # arrays let go
    around = {
        c: (head[:, place[c]].copy(), T[:, place[c]].copy())
        for _, c in anchor.values()
        if frames.compensate[c]
    }
    del head, T

    # the rows in turn, each solved over every frame at once
    solved_local = {}
    solved_scale = {}
    for row in rows:
        if row in solved_local:
            # solved before: a root holding the local it was given
            world = solved_local[row]
        elif row in anchor:
            w, c = anchor[row]
            under = local[:, place[c]]
            if c in around:
                left, right = around[c]
                under = left @ _diagonal(1.0 / solved_scale[w]) @ right
            world = under if row == c else product[row] @ under
            world = world @ solved_local[w]
        else:
            world = worlds[:, first[row]]

        solved = _solve_under(frames, [row], world[:, None], -1)
        _write_solved(frames, [row], solved)

        # its local as a root: no IS, rotate_axis and joint_orient as they were
        scale             = _diagonal(solved[0][:, 0])
        axis              = _euler_matrices(frames.read("_rotate_axis", [row]), None)[:, 0]
        rotate            = _euler_matrices(solved[1], frames.orders[[row]])[:, 0]
        orient            = _euler_matrices(frames.read("_joint_orient", [row]), None)[:, 0]
        solved_scale[row] = solved[0][:, 0]
        solved_local[row] = scale @ axis @ rotate @ orient @ _translation(solved[2][:, 0])


def _compress(link: dict, reach: dict, node) -> None:
    """the union-find step of :func:`_to_the_world_in_turn`: points ``node``
    and every node on its way at the top of its chain, the node no link
    leaves, each one's ``reach`` then the product of the locals from it up
    to there, that one left out. A later walk through any of them takes one
    step, so however many rows ask, each chain is walked once."""
    chain = []
    while link.get(node, node) != node:
        chain.append(node)
        node = link[node]
    for lower, upper in zip(reversed(chain[:-1]), reversed(chain[1:])):
        reach[lower] = reach[lower] @ reach[upper]
        link[lower]  = node


def _write_solved(frames: _RigFrames, rows, solved: tuple) -> None:
    for channel, values in zip(("_scale", "_rotate", "_translate"), solved):
        frames.write(channel, rows, values)


def _filter_rotate(frames: _RigFrames, rows, before: np.ndarray) -> None:
    """puts the rotate curves of ``rows`` a solve rebuilt through euler_filter:
    those that differ from ``before`` (F, n, 3), the curves as they were, on
    some frame. A curve the solve kept on every frame keeps its keys"""
    rows    = np.asarray(rows, dtype=int)
    rotate  = frames.read("_rotate", rows)
    rebuilt = (rotate != before).any(axis=(0, 2))
    if rebuilt.any():
        rows = rows[rebuilt]
        frames.write("_rotate", rows, _filtered(rotate[:, rebuilt], frames.orders[rows]))


def _in_own_axes(axes: np.ndarray, *changes: np.ndarray, parts: bool = False) -> tuple:
    """for each of ``changes``, the largest entry of inv(axes) @ change, one
    per (..., 3, 3) pair: a change of a row's world 3x3 in the row's own
    axes, the factor it moves a point below the row by, per unit of the
    point's distance from it

    With ``parts`` the first change also gives the largest entry of its
    turn and of the rest, the skew and the symmetric part of that matrix,
    after the sizes: everything below the row takes the change on the right
    of its world, so the skew part turns it and the rest is what a euler
    split below can make more of (:func:`_split_gain`).

    0 where nothing changes; inf where ``axes`` has no inverse and something
    does. Worked entry by entry over the whole stack: a stack of 3x3
    inverses and products costs far more than the arithmetic in it.
    """
    (a, b, c), (d, e, f), (g, h, i) = [[axes[..., r, k] for k in range(3)] for r in range(3)]
    adjugate = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    det = a * adjugate[0][0] + b * adjugate[1][0] + c * adjugate[2][0]

    def scaled(size, change):
        with np.errstate(divide="ignore", invalid="ignore"):
            size = size / np.abs(det)
        size = np.where(np.isfinite(size), size, np.inf)
        return np.where((change == 0.0).all(axis=(-2, -1)), 0.0, size)

    sizes, split = [], []
    for n, change in enumerate(changes):
        size    = np.zeros(det.shape)
        entries = [[None] * 3 for _ in range(3)]
        for r in range(3):
            for k in range(3):
                entry = adjugate[r][0] * change[..., 0, k] + adjugate[r][1] * change[..., 1, k]
                entry = entry + adjugate[r][2] * change[..., 2, k]
                size  = np.maximum(size, np.abs(entry))
                if parts and not n:
                    entries[r][k] = entry
        sizes.append(scaled(size, change))
        if parts and not n:
            turn = np.zeros(det.shape)
            rest = np.abs(entries[0][0])
            for r, k in ((0, 1), (0, 2), (1, 2)):
                turn = np.maximum(turn, 0.5 * np.abs(entries[r][k] - entries[k][r]))
                rest = np.maximum(rest, 0.5 * np.abs(entries[r][k] + entries[k][r]))
            rest  = np.maximum(rest, np.maximum(np.abs(entries[1][1]), np.abs(entries[2][2])))
            split = [scaled(turn, change), scaled(rest, change)]
    return tuple(sizes + split)


def _branches(parents, tops) -> tuple:
    """(inside, walk): which rows hang in the branches of ``tops``, those
    rows included, and every row in an order that puts parents first"""
    walk   = np.argsort(_chain_depths(parents), kind="stable").tolist()
    inside = np.zeros(len(parents), dtype=bool)

    inside[np.asarray(tops, dtype=int)] = True
    for row in walk:
        if not inside[row] and parents[row] > -1 and inside[parents[row]]:
            inside[row] = True
    return inside, walk


def _branch_worlds(frames: _RigFrames, rows, parents, worlds: np.ndarray, inside, walk) -> np.ndarray:
    """(F, m, 4, 4): the world of every row ``inside`` on each frame.
    ``rows`` and ``worlds`` are the picked rows and their worlds as they
    were, ``parents`` the parenting before the edit; the worlds of the rows
    between are put together from their locals, every row once"""
    known = {}
    for k, row in enumerate(np.asarray(rows, dtype=int).tolist()):
        if inside[row]:
            known.setdefault(row, worlds[:, k])
    order = [row for row in walk if inside[row] and row not in known]
    if order:
        local = _compose_local(frames.channels(order), frames.orders[order], frames.inverse(order))
        for m, row in enumerate(order):
            known[row] = local[:, m] @ known[int(parents[row])]
    return np.stack(list(known.values()), axis=1)


def _branch_bound(frames: _RigFrames, rows, parents, worlds: np.ndarray, inside, walk) -> np.ndarray:
    """(F, 1): an upper bound of what :func:`_weigh` gives on
    :func:`_branch_worlds`, from the translations and scales alone: a row
    that is not picked sits from its parent no farther than its translation
    times the stretch of its parent's world, and a world stretches no more
    than its parent's times the row's largest scale, divided by the parent's
    smallest one where the row compensates it; a rotation stretches
    nothing. Every row once, a few numbers a frame each"""
    rows  = np.asarray(rows, dtype=int)
    held  = inside[rows]
    known = set(rows[held].tolist())
    order = [row for row in walk if inside[row] and row not in known]
    norm  = {}  # row -> (F,) the stretch of its world
    reach = {}  # row -> (F,) how far it can sit from the picked row above it
    for k in np.flatnonzero(held).tolist():
        norm[int(rows[k])]  = np.sqrt((worlds[:, k, :3, :3] ** 2).sum(axis=(-2, -1)))
        reach[int(rows[k])] = np.zeros(worlds.shape[0])

    up      = np.array([int(parents[row]) for row in order], dtype=int)
    largest = np.abs(frames.read("_scale", order)).max(axis=-1)
    length  = np.sqrt((frames.read("_translate", order) ** 2).sum(axis=-1))
    divide  = np.ones_like(largest)
    on      = np.asarray(frames.compensate[order], dtype=bool)
    if on.any():
        with np.errstate(divide="ignore"):
            divide[:, on] = 1.0 / np.abs(frames.read("_scale", up[on])).min(axis=-1)
    for m, row in enumerate(order):
        above      = int(up[m])
        reach[row] = reach[above] + length[:, m] * norm[above]
        norm[row]  = norm[above] * largest[:, m] * divide[:, m]

    places  = worlds[:, held, 3, :3]
    spread  = (places.max(axis=1) - places.min(axis=1)).max(axis=-1)
    far     = np.max(np.stack(list(reach.values()), axis=-1), axis=-1)
    stretch = np.max(np.stack(list(norm.values()), axis=-1), axis=-1)
    with np.errstate(invalid="ignore"):
        bound = np.maximum(spread + 2.0 * far, stretch)[:, None]
    return np.where(np.isfinite(bound), np.maximum(1.0, bound), np.inf)


def _weigh(worlds: np.ndarray) -> np.ndarray:
    """(F, 1): what a turn or a new scale above (F, m, 4, 4) worlds is
    weighed by, at least 1: how far apart they sit on the widest axis, as a
    point below moves by the change times its distance, or the largest
    stretch of their 3x3s (the Frobenius norm) where that is more, as the
    axes below turn by the change times their own scale"""
    places  = worlds[..., 3, :3]
    spread  = (places.max(axis=1) - places.min(axis=1)).max(axis=-1)
    stretch = np.sqrt((worlds[..., :3, :3] ** 2).sum(axis=(-2, -1))).max(axis=1)
    weight  = np.maximum(spread, stretch)[:, None]
    return np.where(np.isfinite(weight), np.maximum(1.0, weight), np.inf)


def _carried(frames: _RigFrames, rows, parents, moved, resized, gains=None) -> bool:
    """whether a picked row hangs below a ``moved`` row, or below a
    ``resized`` one through a compensated child of it, far enough for its
    answer to differ: each row's answer is its parent's, or the link to its
    parent, worked out once a row however many rows climb through it

    ``moved`` and ``resized`` are each row's (F, n) weighed change, a turn
    or a new scale. ``gains``, when given, is ((turn, rest), (on_turn,
    on_rest), floor, (by_turn, by_rest)), each (F, n): ``moved`` split into
    its turn and the rest, a new scale being all rest, what each picked row
    makes of a unit of each (:func:`_cascades`), and of its split's
    round-off, and how far its first pass split turns its world per unit of
    each short of a step, past the turn passing as it is; each is taken at
    its largest over the frames. What reaches a row is the largest change
    above it, of each kind, and a picked row counts once the change passes
    1e-11 alone, once its turn and its rest times their gains add up past
    that, or, where its floor passes it, once any change reaches it at all.
    A picked row on the way down passes on, as a turn, how far its split
    turns its world by what reaches it: one at a time every row below it is
    solved from a world turned by that much. A change that can carry the
    row across a step counts the row itself.
    """
    moved   = moved.max(axis=0).tolist()
    resized = resized.max(axis=0).tolist()
    if not (max(moved) > 0.0 or max(resized) > 0.0):
        return False
    turn = rest = on_turn = on_rest = floor = by_turn = by_rest = [0.0] * len(moved)
    if gains is not None:
        (turn, rest), (on_turn, on_rest), floor, (by_turn, by_rest) = gains
        turn, rest = turn.max(axis=0).tolist(), rest.max(axis=0).tolist()
        on_turn, on_rest = on_turn.max(axis=0).tolist(), on_rest.max(axis=0).tolist()
        by_turn, by_rest = by_turn.max(axis=0).tolist(), by_rest.max(axis=0).tolist()
        floor = floor.max(axis=0).tolist()

    def through(k, turned, other):
        # how far picked row k's split turns its world as a turn and a rest
        # reach it, past the turn itself
        return (turned * by_turn[k] if turned > 0.0 else 0.0) + (other * by_rest[k] if other > 0.0 else 0.0)

    at    = {int(r): k for k, r in enumerate(rows)}
    below = {}
    for row in rows.tolist():
        path, node = [], int(row)
        while node not in below:
            above = int(parents[node])
            if above < 0 or len(path) > len(parents):
                below[node] = (0.0, 0.0, 0.0)
                break
            path.append(node)
            node = above

        reach = below[node]
        for child in reversed(path):
            k = at.get(int(parents[child]))
            if k is not None:
                grown = resized[k] if frames.compensate[child] else 0.0
                reach = (
                    max(reach[0], moved[k], grown),
                    max(reach[1], turn[k], through(k, reach[1], reach[2])),
                    max(reach[2], rest[k], grown),
                )
            below[child] = reach

        k = at[int(row)]
        whole, turned, other = below[int(row)]
        gained = (turned * on_turn[k] if turned > 0.0 else 0.0) + (other * on_rest[k] if other > 0.0 else 0.0)
        if whole > 1e-11 or gained > 1e-11 or (whole > 0.0 and floor[k] > 1e-11):
            return True
    return False


def _split_input(target: np.ndarray, channels: dict, orders, inverse: np.ndarray) -> np.ndarray:
    """(F, n, 3, 3): the rotations :func:`_solve_local` hands matrix_to_euler
    to reach ``target`` (F, n, 4, 4), step for step to round-off: JO * IS
    stripped from the right, the rows normalized, rotate_axis taken out"""
    RO = _euler_matrices(channels["_rotate_axis"], None)
    JO = _euler_matrices(channels["_joint_orient"], None)
    B  = target @ _diagonal(inverse) @ np.swapaxes(JO, -1, -2)
    _, rotation = _scales_and_rotations(
        B[..., :3, :3], lambda: (RO @ _euler_matrices(channels["_rotate"], orders))[..., :3, :3]
    )
    return np.swapaxes(RO[..., :3, :3], -1, -2) @ rotation


def _split_gain(matrices: np.ndarray, orders) -> tuple:
    """(g, snap), each (F, n): how far matrix_to_euler can move the rotation
    it reads off each of ``matrices`` (F, n, 3, 3) in ``orders``

    The split normalizes each row and reads each angle off two entries with
    atan2, which moves it by a change in them over their length. A turn of
    a rotation moves the result by the turn alone; any other change by up
    to ``g`` times itself, round-off included, 1 over the shortest row times
    the shortest such pair: on a rotation near gimbal, the cosine of the
    middle angle. Where the split snaps (that cosine at most float32's eps)
    it reads the first angle off another pair and loses nothing; a matrix
    within 1e-9 of the snap can land on either side of it, about the snap
    apart, so ``snap`` is 1e-9 over its distance from there
    (:func:`_gain`).
    """
    from cgmath.transforms._numba._euler import MAYA_EA

    eps     = float(np.finfo(np.float32).eps)
    lengths = np.sqrt((matrices ** 2).sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        N = matrices / lengths[..., None]

    i, j, k = MAYA_EA[np.asarray(orders, dtype=int)].T
    column = np.arange(len(i))

    def pair(a, b, c, d):
        return np.hypot(N[:, column, a, b], N[:, column, c, d])

    # the first and last angles' pairs, the first's another once the split
    # snaps; the middle one is read off a whole row, which is never short
    yy    = pair(i, i, i, j)
    least = np.where(yy > eps, np.minimum(pair(j, k, k, k), yy), pair(k, j, j, j))
    with np.errstate(divide="ignore", invalid="ignore"):
        g    = np.maximum(1.0, 1.0 / (lengths.min(axis=-1) * least))
        snap = 1e-9 / np.abs(yy - eps)
    return np.where(np.isnan(g), np.inf, g), snap


def _off_square(matrices: np.ndarray) -> np.ndarray:
    """(F, n): how far each of ``matrices`` (F, n, 3, 3), a euler split's
    input (:func:`_split_input`), is from square, at most 1: the largest
    entry of M @ M.T less the identity. A turn of a square input turns what
    the split reads off it by as much; one that is not square takes what
    is not square along, so up to this much of the turn reaches the split
    as a change that is not one (:func:`_gain`)"""
    with np.errstate(invalid="ignore"):
        off = np.abs(matrices @ np.swapaxes(matrices, -1, -2) - np.eye(3)).max(axis=(-2, -1))
    return np.where(np.isnan(off), 1.0, np.minimum(off, 1.0))


def _gain(g: np.ndarray, snap: np.ndarray, turn, other) -> np.ndarray:
    """how far a split of gain ``g`` and ``snap`` (:func:`_split_gain`) can
    move a row per unit of a change made of up to ``turn`` times as much of
    a turn and ``other`` times as much of a change that is not one"""
    with np.errstate(invalid="ignore", over="ignore"):
        gain = np.maximum(turn + other * g, (turn + other) * snap)
    return np.where(np.isnan(gain), np.inf, gain)


def _keep_gain(gap: np.ndarray) -> np.ndarray:
    """how far the matrix setter's keep test can move a row per unit of a
    change in what it compares: ``gap``, the largest entry between the
    row's channels' matrix and the one it is solved to

    The setter keeps the channels while every entry of the two is within
    1e-8 (:func:`_same_transforms`), a step about 1e-8 tall. A row whose
    gap m sits within a change of 1e-8 can land either side of it, so it
    weighs as 1e-8 over |m - 1e-8|, much as the split's snap does
    (:func:`_split_gain`); a row far from the step weighs nothing.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = 1e-8 / np.abs(gap - 1e-8)
    return np.where(np.isnan(gain), np.inf, gain)


def _under_gain(frames: _RigFrames, rows, local: np.ndarray, solved: tuple, kept: dict, above: int, reached) -> tuple:
    """(gain, g, moved), how far the second pass of set_parent can move the
    rows put under row ``above``: ``gain`` (F, n), 1 but for the
    ``reached`` ones of ``rows``, per unit of a change in the world the
    first pass gave a row, and ``g`` (F, n), 0 but for them, per unit of
    the round-off of its split (:func:`_split_gain`); ``moved``, when the
    first pass moves the parent itself, the two per unit of the change in
    its world, (F,) each, the most any row put under it makes of that, else
    None

    The first pass leaves every row a root, ``local`` (F, n, 4, 4) its world
    and ``solved`` its channels, ``kept`` its rotate_axis and joint_orient.
    The new local is that world times the inverse of the parent's, so a
    change of the world reaches it conjugated by the parent's: up to the
    parent's condition number times as large, all but a turn's worth of it
    no turn at all. A change of the scale alone the split takes out whole.
    A parent the first pass moves reaches every row put under it, by the
    change in its world times its condition number, none of which need be
    a turn. The keep test weighs in too (:func:`_keep_gain`): a change of
    a row's world moves what it compares by up to three times the change
    times 1 or the largest entry of the row's IS, whichever is more, plus
    the largest entry of the parent's inverse; a change of the parent's
    world, by up to three times the change times the parent's condition
    number and the largest entry of the row's new local, or 1.
    """
    rows  = np.asarray(rows, dtype=int)
    count = frames.frames, len(rows)
    gain  = np.ones(count)
    g     = np.zeros(count)
    at    = np.flatnonzero(rows == above)
    moved = False
    if len(at):
        parent, scale = local[:, at[0]], solved[0][:, at[0]]
        moved = bool(np.isin(at[0], reached))
    else:
        parent, scale = frames.world([above])[:, 0], frames.read("_scale", [above])[:, 0]
    under = np.arange(len(rows)) if moved else np.asarray(reached, dtype=int)
    under = under[rows[under] != above]
    if not len(under):
        return gain, g, None

    sizes = np.linalg.svd(parent[:, :3, :3], compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        uneven = sizes[:, 0] / sizes[:, -1] - 1.0
    usable   = np.isfinite(uneven)[:, None]
    parent   = np.where(usable[:, :, None], parent, np.eye(4))
    turn     = 1.0 + uneven[:, None]

    flip     = np.linalg.inv(parent)
    target   = local[:, under] @ flip[:, None]
    channels = {name: values[:, under] for name, values in kept.items()}
    channels.update(_scale=solved[0][:, under], _rotate=solved[1][:, under], _translate=solved[2][:, under])
    on             = np.asarray(frames.compensate[rows[under]], dtype=bool)
    inverse        = np.ones(target.shape[:2] + (3,))
    inverse[:, on] = scale[:, None]
    orders         = frames.orders[rows[under]]
    split, snap = _split_gain(_split_input(target, channels, orders, inverse), orders)

    # the keep test: the channels the first pass solved, with IS, make that
    # pass's world with its columns over the parent's scale, which moves
    # with the row's world through the parent's inverse and the row's IS,
    # and with the parent's world through the row's new local
    composed = local[:, under].copy()
    composed[..., :3, :3] /= inverse[..., None, :]
    keep = _keep_gain(np.abs(composed - target).max(axis=(-2, -1)))
    with np.errstate(divide="ignore", over="ignore"):
        through = np.maximum(1.0, (1.0 / np.abs(inverse)).max(axis=-1))
        keeps   = 3.0 * keep * (through + np.abs(flip[:, None, :3, :3]).max(axis=(-2, -1)))

    own            = np.isin(under, reached)
    gain[:, under] = np.where(
        own, np.where(usable, np.maximum(_gain(split, snap, turn, turn - 1.0), keeps), np.inf), 1.0
    )
    g[:, under] = np.where(own, np.where(usable, split, np.inf), 0.0)
    if not moved:
        return gain, g, None
    with np.errstate(invalid="ignore", over="ignore"):
        carry = keep * 3.0 * turn * np.maximum(1.0, np.abs(target).max(axis=(-2, -1)))
    whole = np.where(usable, np.maximum(_gain(split, snap, turn, turn), carry), np.inf).max(axis=1)
    return gain, g, (whole, np.where(usable[:, 0], split.max(axis=1), np.inf))


def _cascades(
    frames: _RigFrames,
    rows,
    parents,
    worlds: np.ndarray,
    solved: tuple,
    above:  int                  = -1,
    gap:    Optional[np.ndarray] = None,
) -> bool:
    """whether solving ``rows`` to the world one at a time could move a row
    below one of them before its turn, so a batch would not give the same
    answer

    A row solved to the world keeps its world matrix unless that matrix
    holds a shear its channels can not, or the solve keeps the channels the
    row has (within 1e-8 of its world, the matrix setter's keep test), and
    then everything below it moves. A row whose scale changes resizes a
    segment scale compensated child, and everything below that. ``parents``
    is the parenting before the edit; a row below such a row counts whatever
    the order, so this errs towards one at a time.

    Only a row a later picked row carries comes out otherwise in a batch:
    one in the branch of a picked row that another picked row hangs below. A
    turn the channels leave out moves it by that turn, taken in the moving
    row's own axes, times its distance from that row; a new scale over a
    compensated child by the change in the same axes times its distance from
    the child. Its axes turn by the same change times their own world scale.
    Each is weighed by the spread of every row in the moving rows' branches,
    or by the largest stretch of their worlds where that is more, and by the
    gains below, and held to 1e-11, so the batch stays within 1e-9 of one at
    a time however far the rig sits from the origin, however far out a row
    that is not picked hangs, however small or large the scale above,
    however near gimbal or its keep test's step a picked row it reaches
    sits, and however unevenly the new parent stretches. A translation the
    solve leaves out moves everything below by as much, and any counts: a
    solve that does not keep copies the world's translation to the bit.

    A picked row so reached is solved from a world the change has moved. The
    change is a turn, its skew part in the moving row's own axes, and the
    rest. The row's euler split passes the turn as it is, its world turning
    by that alone (near gimbal its first and last angles can trade by the
    turn over the middle angle's cosine), but for what the turn takes along
    of a matrix far from square (:func:`_off_square`), and can make far more
    of the rest, near gimbal or on a matrix far from square
    (:func:`_split_gain`); either can carry the row across the matrix
    setter's keep test, which then keeps its channels one way and solves
    them the other (:func:`_keep_gain`). When the rows then go under row
    ``above`` (-1 for none), what the first pass left them reaches their new
    locals through the parent's world, and that pass's split and keep test
    gain on it too (:func:`_under_gain`). Each picked row's gains over both
    passes, at their largest over the frames, weigh the largest turn and the
    largest rest that reach it from above; up to 10 of a gain spends a tenth
    of the factor of 100 between 1e-11 and 1e-9, so the weight grows with a
    tenth of the gain past that. A split that gains that much gains on
    round-off as well, so two solves of a world a bit apart part by up to
    the gain times eps whatever the change: any change counts once a tenth
    of that does. What the row's first pass split makes of the rest, and of
    what the turn takes along, short of the snap and the keep test's step,
    turns its world one at a time, and so reaches the rows below it as a
    turn. The gains are only worked out when the change alone does not
    settle the call. ``gap`` is each row's keep test gap in the first pass's
    solve (:func:`_solve_local`), (F, n), worked out again when not given.
    """
    rows         = np.asarray(rows, dtype=int)
    count        = len(parents)
    picked       = np.zeros(count, dtype=bool)
    picked[rows] = True

    # the picked rows some other picked row hangs below, each chain climbed
    # once: whatever is above a row met before is marked already
    lower = np.zeros(count, dtype=bool)
    met   = np.zeros(count, dtype=bool)
    for row in rows.tolist():
        if met[row]:
            continue
        met[row] = True
        node     = int(parents[row])
        steps    = 0
        while node > -1 and steps <= count:
            lower[node] |= picked[node]
            if met[node]:
                break
            met[node] = True
            node      = int(parents[node])
            steps += 1
    if not lower[rows].any():
        return False

    before = frames.channels(rows)
    after  = dict(before, _scale=solved[0], _rotate=solved[1], _translate=solved[2])
    local  = _compose_local(after, frames.orders[rows], np.ones_like(solved[0]))
    axes   = worlds[..., :3, :3]
    scale  = before["_scale"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(solved[0] == scale, 0.0, scale / solved[0] - 1.0)

    turned, grown, skew, rest = _in_own_axes(axes, local[..., :3, :3] - axes, ratio[..., None] * axes, parts=True)
    shifted = (local[..., 3, :3] != worlds[..., 3, :3]).any(axis=-1)
    tops    = lower[rows] & ((turned > 0.0) | (grown > 0.0) | shifted).any(axis=0)
    if not tops.any():
        return False

    def carried(weight, gains=None):
        with np.errstate(invalid="ignore", over="ignore"):
            moved   = np.where(shifted, np.inf, turned * weight)
            resized = grown * weight
            if gains is None:
                return _carried(frames, rows, parents, moved, resized)
            parts = np.where(shifted, np.inf, skew * weight), np.where(shifted, np.inf, rest * weight)
            gain, floor, by = gains
            return _carried(frames, rows, parents, moved, resized, (parts, gain, floor * weight, by))

    def gains():
        # each picked row the change can reach, below a moving row: solved
        # from a world the change shears, then put under ``above``. A turn
        # passes the split as it is and the rest by its gain; either can
        # carry a row across the snap or the keep test
        up      = np.asarray(parents[rows])
        reached = np.flatnonzero((up > -1) & inside[np.maximum(up, 0)])
        turn    = np.ones(turned.shape)
        other   = np.ones(turned.shape)
        g       = np.ones(turned.shape)
        by_turn = np.zeros(turned.shape)
        by_rest = np.zeros(turned.shape)
        first   = {name: values[:, reached] for name, values in before.items()}
        orders  = frames.orders[rows[reached]]
        ones    = np.ones_like(first["_scale"])
        split   = _split_input(worlds[:, reached], first, orders, ones)
        g1, snap = _split_gain(split, orders)
        gaps = gap
        if gaps is None:
            gaps = np.abs(_compose_local(first, orders, ones) - worlds[:, reached]).max(axis=(-2, -1))
        else:
            gaps = gaps[:, reached]
        keep              = _keep_gain(gaps)
        off               = _off_square(split)
        turn[:, reached]  = np.maximum(_gain(g1, snap, 1.0, off), keep)
        other[:, reached] = np.maximum(_gain(g1, snap, 0.0, 1.0), keep)
        g[:, reached]     = g1

        # short of the snap and the keep test's step, either of which counts
        # the row itself before a change can carry it across, the first pass
        # split turns the row's world by the turn that reaches it, which
        # reaches the rows below anyway, and by what it makes of the rest
        # and of what the turn takes along: one at a time the rows below it
        # are solved from worlds turned so
        none                = np.zeros_like(snap)
        by_turn[:, reached] = _gain(g1, none, 0.0, off)
        by_rest[:, reached] = _gain(g1, none, 0.0, 1.0)
        if above > -1:
            kept = {name: before[name] for name in ("_rotate_axis", "_joint_orient")}
            gain2, g2, moved = _under_gain(frames, rows, local, solved, kept, above, reached)
            g     = g * gain2 + g2
            turn  = turn * gain2
            other = other * gain2
            if moved is not None:
                # the parent's own change reaches every row put under it
                k           = int(np.flatnonzero(rows == above)[0])
                g[:, k]     = g[:, k] * moved[0] + moved[1]
                turn[:, k]  = turn[:, k] * moved[0]
                other[:, k] = other[:, k] * moved[0]
        # 1e-11 is a hundredth of the 1e-9 the batch is held to: a gain of
        # up to 10 spends a tenth of that room, past it the weight grows
        # with the gain, round-off's included
        return (turn / 10.0, other / 10.0), np.finfo(float).eps * g / 10.0, (by_turn, by_rest)

    # weighed first by the picked rows in those branches, whose worlds are
    # known: they sit no farther apart and stretch no more than the whole
    # branches, so a row carried then is carried anyway, and most calls are
    # settled, the change alone first
    inside, walk = _branches(parents, rows[tops])
    held   = inside[rows]
    weight = _weigh(worlds[:, held])
    if carried(weight):
        return True
    gained = gains()
    if carried(weight, gained):
        return True
    if inside.sum() == len(set(rows[held].tolist())):
        return False
    # then by a bound of the whole branches, which settles the rest but for
    # a turn or a new scale near the limit; the branches' worlds last
    if not carried(_branch_bound(frames, rows, parents, worlds, inside, walk), gained):
        return False
    return carried(_weigh(_branch_worlds(frames, rows, parents, worlds, inside, walk)), gained)


def _reorder_rows(frames: _RigFrames, rows, orders) -> None:
    """set_rotate_order on every row, every frame at once: rotate is
    re-expressed in the new order, so no world matrix moves"""
    rows   = np.asarray(rows, dtype=int)
    orders = np.asarray(orders, dtype=int)
    change = orders != frames.orders[rows]
    rows, orders = rows[change], orders[change]
    if not len(rows):
        return

    # frame 0 the way Maya's xform -preserve picks it, then the filter
    rotate = _reorder_like_maya(frames.read("_rotate", rows), frames.orders[rows], orders)
    frames.write("_rotate", rows, _filtered(rotate, orders))

    for i, order in zip(rows, orders):
        frames.nodes[i]._rotate_order = int(order)
    frames.orders[rows] = orders
    frames._touched.update(rows.tolist())
    frames.finish()


def _frame_property(channel: str) -> property:
    """builds the read/write accessor for one framed channel"""

    def getter(self) -> np.ndarray:
        # read only: an in place edit would land in the block without
        # rebinding the nodes, so half the write would be visible
        columns = self._columns
        block   = self._clip._blocks[channel]
        view    = block.view() if columns is None else block[:, columns]
        view.setflags(write=False)
        return view

    def setter(self, value: np.ndarray) -> None:
        clip    = self._clip
        columns = self._columns
        value   = np.asarray(value, dtype=float)
        if columns is None:
            clip._blocks[channel][...] = value
        else:
            clip._blocks[channel][:, columns] = value
        clip._bind(clip._frame)

    return property(getter, setter)


def _frame_component(channel: str, axis: int) -> property:
    """builds the x / y / z accessor for one framed channel

    A read is a copy, a new array every time, so ``+=`` works on it and a
    write to it changes nothing until it is assigned back. One node reads
    back as (F, 1), and a flat list of F values writes it the same way.
    """

    def getter(self) -> np.ndarray:
        # always a copy: the channel read is read only (on the whole axis a
        # view of the clip, on a scoped axis a copy), and its slice is
        # contiguous already for one frame of one node or for no node, and
        # can be on a block not in C order (one node over every frame of a
        # Fortran-ordered block, say), so a copy made only when the slice is
        # not would hand it back read only
        return getattr(self, channel)[..., axis].copy()

    def setter(self, value) -> None:
        current = np.array(getattr(self, channel))
        value   = np.asarray(value, dtype=float)

        frames, count = current.shape[:2]
        if count == 1 and value.shape == (frames,):
            value = value[:, None]

        current[..., axis] = value
        setattr(self, channel, current)

    return property(getter, setter)


class _FrameAxis:
    """the frame addressed face of a ClipData

    An integer scrubs the clip and hands it back, so ``clip.frames[12]`` and
    ``clip.frame = 12`` are the same operation and the nodes stay a single
    set. Holding two frames at once therefore is not possible; take a
    ``copy()`` for that.

    Everything else narrows instead of scrubbing. A name, a list of names or
    indices, or :meth:`match` gives back an axis scoped to those columns, and
    a channel read or write on it covers every frame. A slice cuts a new
    clip out of the frame range.

    The two do not combine: a scoped axis addresses channels, so slicing one
    or writing a pose into one raises rather than quietly widening back to
    the whole clip.

    Only what the axis defines can be written. Each ``clip.frames`` is a new
    axis, so a write to anything else, ``clip.frames.rotate_order = 0`` say,
    would land on a throwaway object and change nothing; it raises instead.

    A channel read, ``rotate`` say, is read only: on the whole axis it is a
    view of the clip, which an edit in place would change without rebinding
    the nodes, and on a scoped axis a copy, which would lose the edit. A
    component read, ``rotate_y``, is a copy of its own, so ``+=`` works on a
    component on any clip and any scope and writes back through it.

    An axis can be held across an edit that reorders the clip's nodes: it
    keeps its nodes' uuids, not their positions, and lines the clip up with
    them on every read and write.
    """

    __slots__ = ("_clip", "_uuids")

    def __init__(self, clip: "ClipData", columns=None):
        # ``columns`` are positions in the clip's node list as it is now
        self._clip  = clip
        self._uuids = None
        if columns is not None:
            nodes       = clip.list
            positions   = np.asarray(columns, dtype=int).reshape(-1)
            self._uuids = [nodes[i].uuid for i in positions]

    @property
    def _columns(self) -> Optional[np.ndarray]:
        """the positions in the clip of the nodes this axis covers, None for
        every node, worked out when asked

        The clip is realigned first, its block columns put back in the order
        of its nodes: an edit since the axis was taken -- set_parent moving a
        branch to the end of the list, a sort, a node added or taken out --
        would otherwise pair one node's columns with another's rotate order
        and parent. A scoped axis finds its nodes by uuid, and raises when the
        clip no longer holds one of them; a clip with no frames has nothing to
        read or write and raises too.
        """
        clip = self._clip
        clip._realign()
        if not clip._blocks:
            raise ValueError("the clip holds no frames")
        if self._uuids is None:
            return None

        place = {node.uuid: i for i, node in enumerate(clip.list)}
        for uuid in self._uuids:
            if uuid not in place:
                raise ValueError(f"the clip no longer holds the node {uuid!r} this axis covers")
        return np.array([place[uuid] for uuid in self._uuids], dtype=int)

    def __setattr__(self, name: str, value) -> None:
        if not hasattr(type(self), name):
            raise AttributeError(
                f"clip.frames has no {name!r} to write; a value every frame "
                f"shares, such as rotate_order, is set on the clip itself"
            )
        object.__setattr__(self, name, value)

    # the rig edits change what every frame shares, so they belong to the
    # clip and cover every frame wherever they are called
    _RIG_EDITS = frozenset(
        (
            "swapaxes",
            "set_rotate_to_joint_orient",
            "set_joint_orient_to_rotate",
            "set_rotate_order",
            "set_parent",
            "set_node_type",
            "node_type",
        )
    )

    def __getattr__(self, name: str):
        if name in _FrameAxis._RIG_EDITS:
            raise AttributeError(
                f"clip.frames has no {name}: a rig edit changes what every frame "
                f"shares and covers every frame wherever it is called; use "
                f"clip.{name}, or clip['node'].{name} for one node"
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __len__(self) -> int:
        return self._clip.frame_count

    def __iter__(self):
        for frame in range(len(self)):
            yield self[frame]

    def __getitem__(self, query):
        """str or sequence scopes to nodes, slice cuts frames, int scrubs"""
        clip = self._clip

        if isinstance(query, str):
            return _FrameAxis(clip, [clip.index(query)])

        if _is_sequence(query):
            return _FrameAxis(clip, [self._column(item) for item in query])

        if self._uuids is not None:
            raise TypeError("cut the frame range before scoping to nodes")

        if isinstance(query, slice):
            return clip._subclip(query)

        clip.frame = query
        return clip

    def match(self, *args, exclude: bool = False, exact: bool = True) -> "_FrameAxis":
        """the same name matching as TransformList, scoped over every frame"""
        clip  = self._clip
        nodes = clip.match(*args, exclude=exclude, exact=exact)
        return _FrameAxis(clip, [clip.index(node) for node in nodes])

    def _column(self, item) -> int:
        return self._clip.index(item) if isinstance(item, str) else int(item)

    def __setitem__(self, query, value) -> None:
        """int writes one pose, slice writes a run of them

        A run is written in one go. A clip's blocks are copied across whole,
        without scrubbing it, so its current frame stays where it was. A
        single pose is read once and repeated down the range. Either way the
        nodes pair by position, as they do for one pose. A clip or a pose in
        other axes or units than this clip, as their rigs record them,
        raises ValueError before anything is written
        (:attr:`HierarchyData.axis_system`).
        """
        clip = self._clip

        if self._uuids is not None:
            raise TypeError("a node scoped axis writes channels, not poses")

        # a pose given as an iterator of nodes is read once, for its record
        # and for its values
        if isinstance(value, Iterable) and not isinstance(value, (DataList, _FrameAxis)):
            value = list(value)
        _check_matched(clip, value, "the clip and the frames or pose pasted into it")

        # this axis may have been taken before the nodes were reordered, and
        # pairing by position needs column i to be node i
        clip._realign()

        paste = isinstance(query, slice) and isinstance(value, ClipData)
        if isinstance(query, slice):
            span     = query
            selected = len(range(*query.indices(clip.frame_count)))
        else:
            span     = clip._resolve_frame(query)
            selected = 1

        if paste and value.frame_count != selected:
            raise ValueError(f"{selected} frames selected, {value.frame_count} supplied")

        if selected and paste:
            rows = self._clip_rows(value)
        elif selected:
            rows = self._pose_rows(value, len(clip), clip._blocks)
        else:
            rows = {}

        # a slice takes a pose's (N, 3) rows down the whole range at once
        for channel, values in rows.items():
            clip._blocks[channel][span] = values

        clip._bind(clip._frame)

    def _clip_rows(self, source: "ClipData") -> dict:
        """every frame of ``source``, as ``(F, N, 3)`` blocks per channel

        ``source`` is realigned, so column ``i`` is its node ``i``. The
        blocks themselves are handed over: numpy copies a source that
        overlaps the rows it is written into, so a clip pasted into itself
        reads every frame before any is written over.
        """
        clip = self._clip
        source._realign()
        if len(source) != len(clip):
            raise ValueError(f"pose holds {len(source)} nodes, clip holds {len(clip)}")

        return {channel: source._blocks[channel] for channel in clip._blocks}

    @staticmethod
    def _pose_rows(pose, count: int, channels) -> dict:
        """one pose's ``channels`` as ``(count, 3)`` rows, each node read once

        A clip given as the pose gives the frame it is bound to.
        """
        nodes = pose.list if isinstance(pose, DataList) else list(pose)
        if len(nodes) != count:
            raise ValueError(f"pose holds {len(nodes)} nodes, clip holds {count}")

        return {
            channel: np.array(
                [getattr(node, channel) for node in nodes], dtype=float
            ).reshape(count, 3)
            for channel in channels
        }

    def __repr__(self) -> str:
        return f"<frames 0..{max(len(self) - 1, 0)} of {self._clip}>"

    @property
    def times(self) -> np.ndarray:
        """the time of each frame in seconds"""
        clip = self._clip
        return (clip.start_frame + np.arange(len(self))) / clip.fps

    @property
    def matrix(self) -> np.ndarray:
        """every frame's local matrices, (F, N, 4, 4)"""
        columns = self._columns
        return self._scoped(_evaluate_local(self._clip), columns)

    @staticmethod
    def _scoped(values: np.ndarray, columns) -> np.ndarray:
        """``values`` over every node, cut to ``columns`` (None for all)"""
        return values if columns is None else values[:, columns]

    @property
    def world_matrix(self) -> np.ndarray:
        """every frame's world matrices, (F, N, 4, 4)"""
        columns = self._columns
        clip    = self._clip
        parents = clip._parent_indices(clip.list)
        world   = _evaluate_world(_evaluate_local(clip, parents), parents)
        return self._scoped(world, columns)

    # --- solved writes: every frame at once --- #
    @matrix.setter
    def matrix(self, value) -> None:
        """writes local matrices as TransformData.matrix does, every frame at
        once: (4, 4) for every node and frame, (n, 4, 4) per node, or
        (F, n, 4, 4) frame for frame"""
        self._solve("local", self._values(value, (4, 4)))

    @world_matrix.setter
    def world_matrix(self, value) -> None:
        """writes world matrices as TransformData.world_matrix does, every
        frame at once: (4, 4), (n, 4, 4) or (F, n, 4, 4)"""
        self._solve("world", self._values(value, (4, 4)))

    @property
    def quaternion(self) -> np.ndarray:
        """every frame's orientation RO * R * JO as quaternions, (F, n, 4)"""
        return _span_quaternions(self._clip, self._positions(), slice(None))

    @quaternion.setter
    def quaternion(self, value) -> None:
        """writes orientations as TransformData.quaternion does, every frame
        at once: (4,), (n, 4) or (F, n, 4)"""
        self._solve("quaternion", self._values(value, (4,)))

    def match_translate(self, other, x: bool = True, y: bool = True, z: bool = True) -> None:
        """matches world positions on every frame at once

        ``other`` is a clip or a frame axis, paired frame for frame; a node,
        a list of nodes or a pose, held on every frame; or positions: (3,),
        (n, 3) or (F, n, 3). Nodes in other axes or units than the clip, as
        their rigs record them, raise ValueError before anything moves
        (:attr:`HierarchyData.axis_system`).
        """
        _check_matched(self, other)
        if _holds_nodes(other):
            target = self._worlds(other)[..., 3, :3]
        else:
            target = self._values(other, (3,))
        self._solve("translate", target, axes=np.array([x, y, z], dtype=bool))

    def match_rotate(self, other) -> None:
        """matches world orientations on every frame at once

        ``other`` is what :meth:`match_translate` takes, or rotations: eulers
        in degrees in each node's rotate order, (3,), (n, 3) or (F, n, 3), or
        quaternions, (4,), (n, 4) or (F, n, 4). Nodes in other axes or units
        than the clip raise ValueError, as :meth:`match_translate` says.
        """
        _check_matched(self, other)
        if _holds_nodes(other):
            target = _orientations(self._worlds(other))
        else:
            values = np.asarray(other, dtype=float)
            if values.ndim == 0 or values.shape[-1] not in (3, 4):
                raise ValueError(
                    "match_rotate expects a length-3 euler (degrees) or "
                    f"length-4 quaternion sequence, got shape {values.shape}"
                )
            values = self._values(values, values.shape[-1:])
            orders = np.array([node.rotate_order for node in self._nodes()], dtype=int)
            target = _rotation_targets(np.ascontiguousarray(values), orders)
        self._solve("rotate", target)

    def match_matrix(self, other) -> None:
        """matches world matrices on every frame at once

        ``other`` is a clip or a frame axis, paired frame for frame, or a
        node, a list of nodes or a pose, held on every frame. Nodes in other
        axes or units than the clip raise ValueError, as
        :meth:`match_translate` says.
        """
        _check_matched(self, other)
        if not _holds_nodes(other):
            raise TypeError(
                "match_matrix matches nodes, clips or frame axes; "
                "set world_matrix to write matrices"
            )
        self._solve("world", self._worlds(other))

    def _positions(self) -> np.ndarray:
        """the positions in the clip of the nodes this axis covers, the clip
        lined up with its nodes first"""
        columns = self._columns
        if columns is None:
            return np.arange(len(self._clip))
        return columns

    def _values(self, value, shape: tuple) -> np.ndarray:
        """``value`` as (F, n) + ``shape``: one value for every node and frame,
        one per node held on every frame, or one per node and frame"""
        frames, count = self._clip.frame_count, len(self._positions())
        value = np.asarray(value, dtype=float)
        lead  = value.ndim - len(shape)
        if not 0 <= lead <= 2 or value.shape[lead:] != tuple(shape):
            raise ValueError(
                f"expected {tuple(shape)}, (n,) + {tuple(shape)} or (F, n) + "
                f"{tuple(shape)} values, got shape {value.shape}"
            )
        value = value.reshape((1,) * (2 - lead) + value.shape)
        try:
            return np.broadcast_to(value, (frames, count) + tuple(shape))
        except ValueError:
            raise ValueError(
                f"values of shape {value.shape} do not fit {frames} frames "
                f"of {count} nodes"
            ) from None

    def _worlds(self, other) -> np.ndarray:
        """(F, n, 4, 4) world matrices to match: a clip or a frame axis pairs
        frame for frame, a node, a list of nodes or a pose holds on every
        frame, and so does a clip of one frame"""
        frames, count = self._clip.frame_count, len(self._positions())

        if isinstance(other, _FrameAxis) or (isinstance(other, ClipData) and other._blocks):
            source = other._clip if isinstance(other, _FrameAxis) else other
            if source.frame_count not in (frames, 1):
                raise ValueError(f"clips hold {frames} and {source.frame_count} frames")
            world = np.asarray(other.world_matrix if isinstance(other, _FrameAxis)
                               else other.frames.world_matrix)
        elif isinstance(other, TransformData):
            world = other.world_matrix[None, None]
        else:
            world = np.array([node.world_matrix for node in other]).reshape(1, -1, 4, 4)

        if world.shape[1] not in (1, count):
            raise ValueError(f"expected {count} nodes to match against, got {world.shape[1]}")
        return np.broadcast_to(world, (frames, count, 4, 4))

    # --- the pose algebra, every frame at once --- #
    # clip.frames stands for every frame where the clip itself stands for
    # the frame it is bound to: ``clip -= pose`` is that frame alone,
    # ``clip.frames -= pose`` every frame
    def _every(self, op: str) -> "ClipData":
        """the clip, for ``op`` over every frame of it"""
        if self._uuids is not None:
            raise TypeError(f"{op} covers every node of a clip: use clip.frames, not a scoped axis")
        if not self._clip._blocks:
            raise ValueError("the clip holds no frames")
        return self._clip

    def __isub__(self, other, by: str = "index", translate: bool = True):
        """``clip.frames -= other``: every frame becomes its delta against
        ``other``, all solved at once, in place, as :meth:`ClipData.get_delta`
        solves them; the clip stays on its frame. ``other`` is one pose, or a
        clip of the same length, or its frame axis, paired frame for frame."""
        clip = self._every("-=")
        other, _ = _framed_operand(other)
        if not isinstance(other, TransformList):
            return NotImplemented

        clip._subtract_all_frames(other, by=by, translate=translate)
        clip._bind(clip._frame)
        return self

    def __iadd__(self, other, by: str = "index", translate: bool = True):
        """``deltas.frames += other``: every frame's delta poses ``other``,
        all solved at once, in place, as :meth:`ClipData.add_delta` solves
        them; the clip stays on its frame. ``other`` is one pose, or a clip of
        the same length, or its frame axis, paired frame for frame."""
        clip = self._every("+=")
        other, _ = _framed_operand(other)
        if not isinstance(other, TransformList):
            return NotImplemented

        clip._add_all_frames(other, by=by, translate=translate)
        clip._bind(clip._frame)
        return self

    def __sub__(self, other):
        """``clip.frames - other``: :meth:`ClipData.get_delta`, every frame"""
        return self._every("-").get_delta(other)

    def __add__(self, other):
        """``deltas.frames + rig``: ``rig.add_delta(deltas.frames)``, every frame"""
        if not isinstance(other, TransformList):
            return NotImplemented
        return other.add_delta(self)

    def _solve(self, op: str, target: np.ndarray, axes=None) -> None:
        # a node listed twice takes its last value, as writing in turn would
        columns, keep = _last_of_each(self._positions())
        _solve_frames(self._clip, columns, slice(None), op, target[:, keep], axes=axes)

    scale          = _frame_property("_scale")
    rotate         = _frame_property("_rotate")
    translate      = _frame_property("_translate")
    rotate_axis    = _frame_property("_rotate_axis")
    joint_orient   = _frame_property("_joint_orient")

    scale_x        = _frame_component("scale",        0)
    scale_y        = _frame_component("scale",        1)
    scale_z        = _frame_component("scale",        2)
    rotate_x       = _frame_component("rotate",       0)
    rotate_y       = _frame_component("rotate",       1)
    rotate_z       = _frame_component("rotate",       2)
    translate_x    = _frame_component("translate",    0)
    translate_y    = _frame_component("translate",    1)
    translate_z    = _frame_component("translate",    2)
    rotate_axis_x  = _frame_component("rotate_axis",  0)
    rotate_axis_y  = _frame_component("rotate_axis",  1)
    rotate_axis_z  = _frame_component("rotate_axis",  2)
    joint_orient_x = _frame_component("joint_orient", 0)
    joint_orient_y = _frame_component("joint_orient", 1)
    joint_orient_z = _frame_component("joint_orient", 2)

    # --- what the nodes build from their channels, every frame at once ---- #
    # each equals the TransformData read of the same name taken frame by
    # frame, scoped the way the channels are
    def _nodes(self) -> list:
        """the nodes this axis covers, in column order"""
        columns = self._columns
        nodes   = self._clip.list
        return nodes if columns is None else [nodes[i] for i in columns]

    @property
    def scale_matrix(self) -> np.ndarray:
        """every frame's scale matrices, (F, N, 4, 4)"""
        return _diagonal(self.scale)

    @property
    def rotate_axis_matrix(self) -> np.ndarray:
        """every frame's rotate axis matrices, (F, N, 4, 4)"""
        return _euler_matrices(self.rotate_axis, None)

    @property
    def rotate_matrix(self) -> np.ndarray:
        """every frame's rotate matrices, (F, N, 4, 4), in each node's order"""
        orders = np.array([node.rotate_order for node in self._nodes()], dtype=int)
        return _euler_matrices(self.rotate, orders)

    @property
    def joint_orient_matrix(self) -> np.ndarray:
        """every frame's joint orient matrices, (F, N, 4, 4)"""
        return _euler_matrices(self.joint_orient, None)

    @property
    def parent_scale_inverse(self) -> np.ndarray:
        """every frame's parent scale, (F, N, 3), ones for a root

        As on a node, this is the parent's scale whether or not the node
        compensates for it; parent_scale_inverse_matrix applies the flag.
        """
        columns = self._columns
        clip    = self._clip
        return self._scoped(_parent_scale(clip, clip._parent_indices(clip.list)), columns)

    @property
    def parent_scale_inverse_matrix(self) -> np.ndarray:
        """every frame's parent scale inverse matrices, (F, N, 4, 4)

        Identity for a node that does not compensate segment scale.
        """
        compensate = np.array(
            [bool(node.segment_scale_compensate) for node in self._nodes()], dtype=bool
        )
        inverse = np.where(compensate[:, None], self.parent_scale_inverse, 1.0)
        return _diagonal(1.0 / inverse)

    @property
    def translate_matrix(self) -> np.ndarray:
        """every frame's translate matrices, (F, N, 4, 4)"""
        return _translation(self.translate)

    @property
    def parent_matrix(self) -> np.ndarray:
        """every frame's parent world matrices, (F, N, 4, 4)

        What get_parent_matrix gives a node on each frame: identity for a
        root, and for a node whose parent is not in the clip.
        """
        columns = self._columns
        clip    = self._clip
        parents = clip._parent_indices(clip.list)
        world   = _evaluate_world(_evaluate_local(clip, parents), parents)
        if columns is not None:
            parents = parents[columns]

        out      = np.empty((len(world), len(parents), 4, 4), dtype=float)
        out[...] = np.eye(4)

        parented = parents >= 0
        if parented.any():
            out[:, parented] = world[:, parents[parented]]
        return out

    def get_parent_matrix(self) -> np.ndarray:
        """every frame's parent world matrices, the same as parent_matrix"""
        return self.parent_matrix

    @property
    def inverse_matrix(self) -> np.ndarray:
        """every frame's inverse local matrices, (F, N, 4, 4), in one call"""
        return _inverse(self.matrix, "matrix", self._names)

    @property
    def world_inverse_matrix(self) -> np.ndarray:
        """every frame's inverse world matrices, (F, N, 4, 4), in one call"""
        return _inverse(self.world_matrix, "world_matrix", self._names)

    @property
    def parent_inverse_matrix(self) -> np.ndarray:
        """every frame's inverse parent world matrices, (F, N, 4, 4):
        identity for a root; in one call"""
        return _inverse(self.parent_matrix, "parent_matrix", self._names)

    def _names(self) -> list:
        """the names of the nodes this axis covers, in column order"""
        return [node.name for node in self._nodes()]

    def to_attributes(self, mapping: dict = MAYA_ATTRIBUTE_MAP) -> List[dict]:
        """every node's attributes as a list of dicts, covering every frame

        What TransformData.to_attributes gives on each frame, gathered: a
        framed channel maps to one [x, y, z] per frame, anything else to
        the single value every frame shares. ``mapping`` names a field as
        each node's own to_attributes takes it, ``"translate"`` or
        ``"_translate"``.
        """
        # one (n, F, 3) list per framed channel, rather than a tolist per node.
        # A channel only needs its field's name here: each node's class reads
        # the whole mapping below, and refuses one naming its field both ways
        columns  = self._columns
        field_of = self._clip.DATA_LIST_CLASS._saved_names()[1]
        channels = {field_of.get(key, key) for key in mapping}
        framed = {
            channel: np.swapaxes(self._scoped(block, columns), 0, 1).tolist()
            for channel, block in self._clip._blocks.items()
            if channel in channels
        }

        # the mapping by field, as each node's class reads it
        by_class = {}
        out      = []
        for column, node in enumerate(self._nodes()):
            kind = type(node)
            if kind not in by_class:
                by_class[kind] = kind._as_fields(mapping, "the mapping")
            names      = by_class[kind]
            attributes = {}
            for key, value in node.__dict__.items():
                if key not in names:
                    continue
                if key in framed:
                    attributes[names[key]] = framed[key][column]
                elif isinstance(value, np.ndarray):
                    attributes[names[key]] = value.tolist()
                else:
                    attributes[names[key]] = value
            out.append(attributes)
        return out


# Maya's short names of what the axis has, every frame as their long names:
# no ro, ssc, v, radi or ds, which every frame shares
_add_short_names(_FrameAxis)


def _check_rest_pose(value, clip: "ClipData") -> None:
    """refuses a rest pose of ``clip`` that is not None or one pose: a
    :class:`HierarchyData`, not a view and not a clip, recording the
    clip's axes and unit"""
    if value is None:
        return
    if not isinstance(value, HierarchyData) or isinstance(value, ClipData):
        raise TypeError(
            f"a clip's rest_pose is a HierarchyData or None, not a {type(value).__name__}"
        )
    if value._records() != clip._records():
        _refuse_records("the clip and its rest pose", (clip._records(), value._records()))


class ClipData(HierarchyData):
    """A HierarchyData holding F poses rather than one.

    The nodes are the playback head. Each framed channel is bound to one row
    of a dense ``(F, N, 3)`` block, so reading a channel reads the current
    frame and writing one writes it -- there is no copy and nothing to flush.
    Scrubbing rebinds:

        clip = ClipData(rig, frames=300, start_frame=1001, fps=24.0)

        clip.frame = 12
        clip.match("L_*").rotate_y += 5      # writes frame 12

        clip.frames[12].match("L_*").rotate_y += 5   # the same thing
        clip.frames.translate                # (300, N, 3), read only
        clip.frames.translate = values       # whole block

    Scoping the frame axis to nodes writes every frame at once, which is the
    difference between keying a channel and keying a pose:

        clip.frames.match("L_*").rotate_y += 5   # all 300 frames
        clip.frames["hips"].translate_y = 3.0    # one node, all frames
        clip.frames[10:20]                       # a new 10 frame clip
        clip.frames[10:20] = other               # paste 10 frames in

    The pose setters work the same way: on the frame axis they solve every
    frame at once, and a rotate rebuilt from a matrix is then euler filtered
    along the frames, while on the clip they write the frame it is bound to:

        clip.frames.world_matrix = W             # (300, N, 4, 4), one solve
        clip.frames.match_matrix(other_clip)     # frame for frame
        clip.frames["hips"].match_rotate(pose)   # one pose on every frame
        clip.world_matrix = W[12]                # frame 12, the one bound above

    So does the pose algebra. The clip stands for the frame it is bound to,
    its frame axis for every frame:

        clip.frames -= rest                      # every frame, in place
        rest.get_delta(clip.frames)              # a clip: rest against each frame
        clip -= rest                             # frame 12 alone
        rest.get_delta(clip)                     # one pose: rest against frame 12

    A clip's own get_delta and add_delta answer for every frame.

    Only the channels a pose setting path writes are framed. Everything else
    -- names, uuids, parenting, rotate_order, segment_scale_compensate,
    radius, draw_style -- is structural and shared by every frame.

    The rig level edits -- ``swapaxes``, ``set_rotate_to_joint_orient``,
    ``set_joint_orient_to_rotate``, ``set_parent`` in world space, a joint
    becoming another type and ``set_rotate_order`` -- change what every
    frame shares, so they cover every frame whether they are called on the
    clip, a selection or one node, all solved at once, with the result the
    single pose methods give frame by frame:

        clip.set_rotate_order("zyx")             # every node, every frame
        clip["hips"].swapaxes(0, 1)              # one node, every frame
        clip.match("L_*").set_parent("root")     # each keeps its world

    They rebuild channels out of matrices. Solved a frame at a time, each
    frame would pick its euler branch on its own, and the curves would step
    by 180 and 360 degrees while every pose stayed exact. So each rebuilt
    euler curve -- rotate in the node's own rotate order, rotate_axis and
    joint_orient in xyz -- goes through euler_filter, frame 0 kept as
    computed, the way Maya's Euler Filter keeps the first key.
    """

    FRAMED_CHANNELS = (
        "_scale",
        "_rotate",
        "_translate",
        "_rotate_axis",
        "_joint_orient",
    )

    # saved beside the nodes: the timebase, the loaded frame, one (F, N, 3)
    # block per framed channel, keyed by channel, the rest pose, and a rig's
    # record of the axes and unit. A file names them as their properties,
    # start_frame, fps, frame, rest_pose, axis_system and unit, and the
    # blocks, which have none, "blocks", each keyed by its channel's
    # property, "translate". A clip with no rest pose saves no "rest_pose"
    LIST_FIELDS = (
        "_start_frame",
        "_fps",
        "_frame",
        "_blocks",
        "_rest_pose",
        "_axis_system",
        "_unit",
    )
    SAVED_NAMES = {"_blocks": "blocks"}

    # the private state a clip holds beside its list, its LIST_FIELDS and a
    # rig's: the uuid each block column stands for, and each uuid's column
    _STATE = ("_columns", "_column_place_cache")

    def __init__(
        self,
        iterable                   = None,
        frames:      Optional[int] = None,
        start_frame: int           = 0,
        fps:         float         = 24.0,
    ):
        self._blocks      = {}
        self._columns     = []
        self._frame       = 0
        self._start_frame = int(start_frame)
        self._fps         = float(fps)
        self._rest_pose   = None

        # a clip owns its nodes: adopting the caller's rig would empty it.
        # The record of a rig, clip or view comes with its nodes, and so
        # does the one the rigs owning a sequence's nodes record
        if iterable is not None and not isinstance(iterable, TransformList):
            iterable = list(iterable)
        record = None if iterable is None else _records_of(iterable)
        nodes  = [] if iterable is None else [node.copy() for node in iterable]
        super().__init__(nodes)
        if record is not None:
            self._axis_system, self._unit = record

        if frames is not None:
            self._allocate(int(frames))

    # --- storage --- #
    def _resolve_frame(self, frame: int) -> int:
        count = self.frame_count
        frame = int(frame)
        if frame < 0:
            frame += count
        if not count or frame < 0 or frame >= count:
            raise IndexError(f"frame {frame} out of range for {count} frames")
        return frame

    def _pose(self, channel: str) -> np.ndarray:
        values = [np.asarray(getattr(node, channel), dtype=float) for node in self.list]
        return np.array(values, dtype=float).reshape(len(self.list), 3)

    def _allocate(self, frames: int) -> None:
        """seeds every block with the pose the nodes currently hold"""
        for channel in self.FRAMED_CHANNELS:
            self._blocks[channel] = np.repeat(
                self._pose(channel)[None, ...], frames, axis=0
            )
        self._columns = [node.uuid for node in self.list]
        self._bind(0)

    # (the column list, each uuid's column in it)
    _column_place_cache = None

    def _column_place(self) -> dict:
        """each uuid's block column, kept until the column list is replaced"""
        cache = self._column_place_cache
        if cache is None or cache[0] is not self._columns:
            place = {uuid: j for j, uuid in enumerate(self._columns)}
            cache = self._column_place_cache = (self._columns, place)
        return cache[1]

    def _realign(self) -> None:
        """keeps the block columns lined up with the node list

        A column belongs to a uuid, not to a position. Base class operations
        reorder nodes by removing and re-adding them -- set_parent sorts the
        new parent ahead of its children -- so alignment is repaired on
        demand rather than maintained by append and pop. A node with no
        column yet is seeded from the pose it already holds.
        """
        wanted = [node.uuid for node in self.list]
        if not self._blocks or wanted == self._columns:
            self._columns = wanted
            return

        source = {uuid: i for i, uuid in enumerate(self._columns)}
        frames = self.frame_count

        for channel, block in self._blocks.items():
            rebuilt = np.empty((frames, len(wanted), 3), dtype=float)
            for i, node in enumerate(self.list):
                column = source.get(node.uuid)
                if column is None:
                    rebuilt[:, i] = np.asarray(getattr(node, channel), dtype=float)
                else:
                    rebuilt[:, i] = block[:, column]
            self._blocks[channel] = rebuilt

        self._columns = wanted
        self._bind(min(self._frame, max(frames - 1, 0)))

    def _bind(self, frame: int) -> None:
        """points every node's channels at one row of the blocks"""
        if self._blocks:
            frame = self._resolve_frame(frame)
            for channel, block in self._blocks.items():
                row = block[frame]
                # each row is already the channel's float64 (3,): set it as
                # the field directly, without the check every frame repeats
                for i, node in enumerate(self.list):
                    node.__dict__[channel] = row[i]
            self._frame = frame

        for node in self.list:
            node._transform_changed()

    # --- timebase --- #
    @property
    def frame_count(self) -> int:
        for block in self._blocks.values():
            return int(block.shape[0])
        return 0

    @property
    def frame(self) -> int:
        return self._frame

    @frame.setter
    def frame(self, value: int) -> None:
        self._realign()
        self._bind(value)

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @start_frame.setter
    def start_frame(self, value: int) -> None:
        self._start_frame = int(value)

    @property
    def end_frame(self) -> int:
        return self._start_frame + self.frame_count - 1

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, value: float) -> None:
        self._fps = float(value)

    @property
    def frames(self) -> _FrameAxis:
        self._realign()
        return _FrameAxis(self)

    @frames.setter
    def frames(self, value) -> None:
        # ``clip.frames -= pose`` solves every frame in place and hands the
        # axis back to be assigned here: nothing is left to do
        if not (isinstance(value, _FrameAxis) and value._clip is self and value._uuids is None):
            raise AttributeError(
                "clip.frames is not assigned; write through it, eg: "
                "clip.frames[:] = other or clip.frames.rotate = values"
            )

    # --- rest pose --- #
    @property
    def rest_pose(self) -> Optional[HierarchyData]:
        """the pose the clip's file holds outside its take, or None

        A :class:`HierarchyData`, which :meth:`load_fbx` sets: the nodes'
        own values, with the clip's names and node order and loaded as
        :meth:`HierarchyData.load_fbx` loads the file, from the same read.
        For an Unreal animation that is the skeleton's bind pose. It
        records the clip's axes and unit (:attr:`axis_system`,
        :attr:`unit`). It is saved with the clip, and a copy or a cut of
        frames (``clip.frames[10:20]``) holds a copy of it. It is None on a
        clip the constructor, :meth:`from_poses` or :meth:`load_glb` builds, on
        one the pose algebra builds (:meth:`get_delta`, :meth:`add_delta`),
        which holds new animation, and on a saved clip that held none.
        ``strip_namespace()`` on the clip strips its names too, and setting
        the clip's :attr:`axis_system` or :attr:`unit` gives the clip a copy
        of it relabelled with the clip, leaving the one set as it was; any
        other edit of the clip leaves it as it is. A :class:`HierarchyData`
        set here is held as it is, not copied, and must record the clip's
        axes and unit, or ValueError is raised: when it is set, and when a
        clip whose rest pose was relabelled since is saved, pickled, copied
        or cut, before anything is written, as it is for a saved clip whose
        rest pose records others.
        """
        return self._rest_pose

    @rest_pose.setter
    def rest_pose(self, value: Optional[HierarchyData]) -> None:
        _check_rest_pose(value, self)
        self._rest_pose = value

    # --- the record: the rest pose is relabelled with the clip --- #
    # The rest pose set is held as it is, and may be another clip's or the
    # caller's rig: a relabel gives the clip a relabelled copy of it, and
    # leaves the one set as it was
    @HierarchyData.axis_system.setter
    def axis_system(self, value: str) -> None:
        HierarchyData.axis_system.fset(self, value)
        rest = self._rest_pose
        if rest is not None and rest._axis_system != self._axis_system:
            rest              = rest.copy()
            rest._axis_system = self._axis_system
            self._rest_pose   = rest

    @HierarchyData.unit.setter
    def unit(self, value: float) -> None:
        HierarchyData.unit.fset(self, value)
        rest = self._rest_pose
        if rest is not None and rest._unit != self._unit:
            rest            = rest.copy()
            rest._unit      = self._unit
            self._rest_pose = rest

    # --- construction --- #
    @classmethod
    def from_poses(cls, poses, start_frame: int = 0, fps: float = 24.0) -> "ClipData":
        """builds a clip from a sequence of HierarchyData poses

        The nodes are copies of the first pose's, and every pose pairs with
        them by position. Each pose is read once, straight into the blocks,
        and the nodes are bound once, to frame 0. The poses record one axis
        system and unit, which the clip records (:attr:`axis_system`,
        :attr:`unit`); poses of two raise ValueError.
        """
        # a pose given as a sequence of nodes is read twice: for its record,
        # then for its values
        poses = [pose if isinstance(pose, DataList) else list(pose) for pose in poses]
        if not poses:
            raise ValueError("from_poses needs at least one pose")
        records = [_records_of(pose) for pose in poses]
        if len(set(records)) > 1:
            _refuse_records("the poses", records)

        clip     = cls(poses[0], start_frame=start_frame, fps=fps)
        count    = len(clip)
        channels = cls.FRAMED_CHANNELS
        blocks   = {c: np.empty((len(poses), count, 3), dtype=float) for c in channels}

        for i, pose in enumerate(poses):
            for channel, values in _FrameAxis._pose_rows(pose, count, channels).items():
                blocks[channel][i] = values

        clip._blocks  = blocks
        clip._columns = [node.uuid for node in clip.list]
        clip._bind(0)
        return clip

    @classmethod
    def _adopt(
        cls, hierarchy: HierarchyData, frames: int, start_frame: int, fps: float
    ) -> "ClipData":
        """a clip of ``frames`` frames over the very nodes of ``hierarchy``,
        which is left empty

        The constructor copies every node, so a clip never takes a caller's
        rig. A hierarchy built only to become a clip -- a delta's working
        copy -- is handed over instead of being copied a second time, with
        its record of the axes and unit.
        """
        clip              = cls.__new__(cls)
        clip._blocks      = {}
        clip._columns     = []
        clip._frame       = 0
        clip._start_frame = int(start_frame)
        clip._fps         = float(fps)
        clip._rest_pose   = None
        clip._axis_system = hierarchy._axis_system
        clip._unit        = hierarchy._unit
        clip.list         = hierarchy.list
        hierarchy.list    = []
        for node in clip.list:
            node._hierarchy = clip

        clip._allocate(int(frames))
        return clip

    def _rooted_copy(self) -> "ClipData":
        """a copy in which a node whose parent the clip lacks is a root

        What an owned copy of a selection does, so what the single pose
        algebra starts from: a delta taken off a clip of part of a rig makes
        such a node a root. A copy's nodes join it as any node joins a
        hierarchy, so the copy is one already. It holds no rest pose: what
        the algebra builds on it is new animation, which the file's rest
        pose does not describe.
        """
        return self._copy_frames()

    # --- pose algebra --- #
    @staticmethod
    def _pose_world(nodes) -> np.ndarray:
        """``(n, 4, 4)`` world matrices of ``nodes`` at the pose they hold,
        each owning hierarchy evaluated once (:func:`_pose_matrices`)"""
        return _pose_matrices(nodes)[1]

    def _operand(self, other, nodes, world: bool = True) -> dict:
        """every frame of ``nodes``, taken from ``other``

        ``other`` is either a clip of the same length, or its frame axis,
        paired frame for frame, or one pose for every frame of this clip: a
        hierarchy, a clip of one frame or none, or a selection or view --
        even one of this clip, read at the frame it is bound to. Against a
        clip of one frame, a longer clip is read at its bound frame too.
        ``nodes`` are the nodes to read, in order: other's own, or copies of
        them, so a clip finds their columns by uuid. Returns the framed channels as ``(F, n, 3)`` blocks and,
        with ``world``, the ``(F, n, 4, 4)`` world matrices: a clip's from
        one evaluation of every frame, a pose's from one of the pose
        (:meth:`_pose_world`). A pose is read once, with a frame axis of 1
        rather than F that broadcasts against the frames, so whatever is
        built from it alone is built once.
        """
        frames = self.frame_count
        other, _ = _framed_operand(other)

        framed = isinstance(other, ClipData) and other.frame_count > 1 and frames > 1
        if framed:
            if other.frame_count != frames:
                raise ValueError(
                    f"clips hold {frames} and {other.frame_count} frames"
                )

            other._realign()
            place = {uuid: j for j, uuid in enumerate(other._columns)}
            cols  = [place[node.uuid] for node in nodes]
            out   = {c: other._blocks[c][:, cols] for c in self.FRAMED_CHANNELS}
            if world:
                out["world"] = other.frames.world_matrix[:, cols]
            return out

        # one pose: read once, as a single frame
        count = len(nodes)
        out   = {}
        for channel in self.FRAMED_CHANNELS:
            pose         = np.array([getattr(n, channel) for n in nodes], dtype=float)
            out[channel] = pose.reshape(1, count, 3)
        if world:
            out["world"] = self._pose_world(nodes)[None]
        return out

    def get_delta(self, other, by: str = "index", translate: bool = True) -> "ClipData":
        """The delta of every frame against ``other``, as a clip.

        ``other`` is either one pose, compared against every frame, or another
        clip of the same length, or its frame axis, compared frame for frame.
        A selection or view, even one of this clip, is one pose, read at the
        frame it is bound to; so is a clip of one frame or none. This clip
        holding one frame or none is one pose too, and is compared against
        every frame of a longer clip, as ``pose.get_delta(clip.frames)``.
        Every frame is solved at once, and ``other`` keeps its current frame.
        Rotate is rebuilt from matrices, so each paired node's rotate curve
        goes through ``euler_filter`` with its rotate order, frame 0 kept as
        solved. Both record one axis system and unit, or ValueError is
        raised, as :meth:`HierarchyData.get_delta` says.
        """
        other, _ = _framed_operand(other)
        if self.frame_count < 2 and isinstance(other, ClipData) and other.frame_count > 1:
            return TransformList.get_delta(self, other.frames, by=by, translate=translate)

        if not self._blocks:
            return HierarchyData.get_delta(self, other, by=by, translate=translate)

        clip = self._rooted_copy()
        clip._subtract_all_frames(other, by=by, translate=translate)
        clip._bind(0)
        return clip

    def _subtract_all_frames(self, other, by: str, translate: bool) -> None:
        """:meth:`HierarchyData.__isub__` on every frame at once, in place.

        ``__isub__`` reads only world matrices, scales and the segment scale
        compensate flags, and writes each node on its own -- no node solves
        against another -- so every frame of every node is one batched solve.
        The steps follow ``__isub__`` and the ``TransformData.quaternion``
        setter it writes through, so the channels match a frame by frame
        solve to rounding. That solve picks each frame's euler branch on its
        own, so the rebuilt rotate curves then go through ``euler_filter``
        with each node's rotate order, frame 0 kept. Nodes left out of the
        pairing keep their channels. Everything is read before anything is
        written; the caller rebinds.
        """
        _check_operands(self, other)
        self._realign()
        frames = self.frame_count

        nodes_self, nodes_other, parents = self._matched_subset(other, by=by)
        count      = len(nodes_self)
        mine       = self._operand(self, nodes_self)
        theirs     = self._operand(other, nodes_other)
        orders     = np.array([n.rotate_order for n in nodes_self], dtype=int)
        compensate = np.array([bool(n.segment_scale_compensate) for n in nodes_self])

        # scale: additive delta around 1, divided by the parent's delta for
        # segment-scale-compensated joints (the inverse of __iadd__)
        base              = 1 + (mine["_scale"] - theirs["_scale"])
        scale             = base.copy()
        divided           = compensate & (parents > -1)
        scale[:, divided] = base[:, divided] / base[:, parents[divided]]

        # rotate: the frame offset F where `W_self == F * W_other`, written
        # through the quaternion setter: the chain is RO * R * JO, solved for R
        shape = (frames, count, 4, 4)
        flat  = (frames * count, 4, 4)
        offset = local_matrix(
            _rotation(mine["world"]).reshape(flat),
            np.broadcast_to(_rotation(theirs["world"]), shape).reshape(flat),
        )
        M      = quaternion_to_matrix(matrix_to_quaternion(offset)).reshape(shape)
        RO     = _euler_matrices(mine["_rotate_axis"], None)
        JO     = _euler_matrices(mine["_joint_orient"], None)
        L      = local_matrix((np.linalg.inv(RO) @ M).reshape(flat), JO.reshape(flat))
        rotate = _matrices_to_euler(L.reshape(shape), orders)

        # each frame picked its euler branch on its own
        rotate = _filtered(rotate, orders)

        place  = {uuid: j for j, uuid in enumerate(self._columns)}
        cols   = [place[node.uuid] for node in nodes_self]

        # the paired columns take the solved scale and rotate
        self._blocks["_scale"][:, cols]  = scale
        self._blocks["_rotate"][:, cols] = rotate

        # translate: a world-space offset; with `translate` off this clip's
        # own bone offsets stay, for add_delta to reuse
        if translate:
            self._blocks["_translate"][:, cols] = (
                mine["world"][..., 3, :3] - theirs["world"][..., 3, :3]
            )

    def add_delta(self, delta, by: str = "index", translate: bool = True) -> "ClipData":
        """Every frame with ``delta`` applied, as a clip.

        ``delta`` is either one delta applied to every frame -- the retarget
        case, where the offset is fixed to the bone -- or a clip of deltas of
        the same length, or its frame axis, paired frame for frame. A
        selection or view, even one of this clip, is one delta, read at the
        frame it is bound to; so is a clip of one frame or none. This clip
        holding one frame or none is one pose, posed by every delta of a
        longer clip of them, as ``pose.add_delta(deltas.frames)``. Every
        frame is solved at once, and a clip of deltas keeps its current
        frame. Rotate is rebuilt from matrices, so each paired node's rotate
        curve goes through ``euler_filter`` with its rotate order, frame 0
        kept as solved. Both record one axis system and unit, or ValueError
        is raised, as :meth:`HierarchyData.add_delta` says.
        """
        delta, _ = _framed_operand(delta)
        if self.frame_count < 2 and isinstance(delta, ClipData) and delta.frame_count > 1:
            return TransformList.add_delta(self, delta.frames, by=by, translate=translate)

        if not self._blocks:
            return HierarchyData.add_delta(self, delta, by=by, translate=translate)

        return self._add_delta_all_frames(delta, by=by, translate=translate)

    def _add_delta_all_frames(self, delta, by: str, translate: bool) -> "ClipData":
        """:meth:`HierarchyData.add_delta` against every frame, vectorised.

        The result is built from the delta, as ``HierarchyData.add_delta``
        is, on this clip's timebase; :meth:`_add_solve` poses it by this
        clip. A clip of deltas brings one delta per frame. A single delta is
        read once, so what is built from it alone -- its orientation
        offsets, rotate axes and joint orients -- is built once and
        broadcast over the frames.
        """
        _check_operands(self, delta)
        self._realign()
        frames = self.frame_count

        # the result is built from the delta, as HierarchyData.add_delta is
        new   = delta._owned_copy()
        nodes = list(new.list)

        # the delta's channels: one row per frame for a clip of deltas, one
        # row for a single delta, which broadcasts against the frames
        d = self._operand(delta, nodes, world=False)
        scale, rotate, offset = self._add_solve(new, d, self, by, translate)

        # the result takes over `new`, which nothing else holds, rather than
        # copying every node of it a second time
        clip = ClipData._adopt(new, frames, self._start_frame, self._fps)

        # block columns belong to uuids; line the solved rows up through them
        # rather than trusting the clip to keep `new`'s order
        place = {uuid: j for j, uuid in enumerate(clip._columns)}
        cols  = [place[node.uuid] for node in nodes]

        # the solved channels, and the delta's own orientation channels
        clip._blocks["_scale"][:, cols]        = scale
        clip._blocks["_rotate"][:, cols]       = rotate
        clip._blocks["_translate"][:, cols]    = offset
        clip._blocks["_rotate_axis"][:, cols]  = d["_rotate_axis"]
        clip._blocks["_joint_orient"][:, cols] = d["_joint_orient"]
        clip._bind(0)
        return clip

    def _add_all_frames(self, source, by: str, translate: bool) -> None:
        """:meth:`HierarchyData.__iadd__` on every frame at once, in place.

        This clip holds the deltas and ``source`` is what they pose: one
        pose, or a clip of the same length, solved by :meth:`_add_solve`.
        Nodes the pairing leaves out keep their channels. Everything is read
        before anything is written; the caller rebinds.
        """
        _check_operands(self, source)
        self._realign()
        blocks = self._blocks
        scale, rotate, offset = self._add_solve(self, blocks, source, by, translate)

        blocks["_scale"][...]     = scale
        blocks["_rotate"][...]    = rotate
        blocks["_translate"][...] = offset

    def _add_solve(self, deltas, d: dict, source, by: str, translate: bool) -> tuple:
        """every frame of the deltas ``deltas`` holds, posed by ``source``

        ``d`` holds the deltas' framed channels in ``deltas``' order, one row
        per frame of this clip or one row for every frame; ``source`` is one
        pose, read once, or a clip of this clip's length. Frame by frame,
        ``HierarchyData.__iadd__`` copies the whole rig and solves it node by
        node. Here the same algebra -- step for step, so the channels come
        out the same to rounding -- runs once per level of the deltas'
        hierarchy, batched over the frame axis and the level's nodes: the
        source's world matrices come from one evaluation, and a level solves
        against the final world matrices of the level above. That solve
        picks each frame's euler branch on its own, so the rebuilt rotate
        curves then go through ``euler_filter`` with each node's rotate
        order, frame 0 kept. Returns the ``(F, N, 3)`` scale, rotate and
        translate in ``deltas``' order; a node the pairing leaves out holds
        its delta's own channels, as it does per frame.
        """
        frames = self.frame_count

        nodes_new, nodes_source, pairing_parents = deltas._matched_subset(source, by=by)

        nodes   = list(deltas.list)
        count   = len(nodes)
        parents = self._parent_indices(nodes)
        row     = {id(node): i for i, node in enumerate(nodes)}

        # rows[k]: the node at pairing position k; paired[i]: node i's place
        # in the pairing, -1 when unpaired
        rows         = np.array([row[id(node)] for node in nodes_new], dtype=int)
        paired       = np.full(count, -1, dtype=int)
        paired[rows] = np.arange(len(rows))

        orders       = np.array([n.rotate_order for n in nodes], dtype=int)
        compensate   = np.array([bool(n.segment_scale_compensate) for n in nodes])
        RO           = _euler_matrices(d["_rotate_axis"], None)  # (F or 1, N, 4, 4)
        RO_inverse   = np.linalg.inv(RO)
        JO           = _euler_matrices(d["_joint_orient"], None)
        R            = _euler_matrices(d["_rotate"][:, rows], orders[rows])

        # F: each paired node's orientation chain RO * R * JO, through a
        # quaternion as TransformData.quaternion reads it
        chain = RO[:, rows] @ R @ JO[:, rows]
        f_delta = quaternion_to_matrix(
            matrix_to_quaternion(chain.reshape(-1, 4, 4))
        ).reshape(chain.shape)

        # scale: the uncompensated base delta, rebuilt over the pairing as
        # HierarchyData.__iadd__ does, then offset by the source scale
        s_pair     = d["_scale"][:, rows]
        base_delta = s_pair.copy()
        for k in self._parent_first_order(pairing_parents):
            if compensate[rows[k]] and pairing_parents[k] > -1:
                base_delta[:, k] = s_pair[:, k] * base_delta[:, pairing_parents[k]]

        # the source in pairing order: one row per frame for a clip, one row
        # for a single pose, which broadcasts against the frames below
        src      = self._operand(source, nodes_source)
        w_source = src["world"]   # (F or 1, P, 4, 4)
        s_source = src["_scale"]  # (F or 1, P, 3)

        # output blocks, seeded with the delta's channels for the unpaired nodes
        shape  = (frames, count, 3)
        scale  = np.array(np.broadcast_to(d["_scale"], shape))
        rotate = np.array(np.broadcast_to(d["_rotate"], shape))
        offset = np.array(np.broadcast_to(d["_translate"], shape))
        world  = np.empty((frames, count, 4, 4), dtype=float)

        # one level of the deltas' hierarchy at a time, every node of it
        # together, against the final world matrices of the level above.
        # `parents` covers all of `deltas`, so its depths are the owning
        # hierarchy's: the order HierarchyData.__iadd__ solves in
        depth = _chain_depths(parents)
        for level in range(int(depth.max()) + 1):
            idx    = np.flatnonzero(depth == level)
            above  = parents[idx]
            rooted = above > -1

            parent_world            = np.empty((frames, len(idx), 4, 4), dtype=float)
            parent_world[:]         = np.eye(4)
            parent_world[:, rooted] = world[:, above[rooted]]

            # i: the level's paired nodes, k: their places in the pairing
            solved = paired[idx] > -1
            if solved.any():
                i    = idx[solved]
                k    = paired[i]
                many = (frames, len(i), 4, 4)
                flat = (frames * len(i), 4, 4)

                # target world: the source orientation carried into this rig's
                # frame by F, at the source position plus the stored offset.
                # A pose is laid out frame by frame as a clip's world is: a
                # copy of a broadcast keeps the frame axis innermost, and
                # sums over it round differently
                w_other = np.ascontiguousarray(np.broadcast_to(w_source[:, k], many))
                offsets = np.ascontiguousarray(np.broadcast_to(f_delta[:, k], many))
                target = matrix_multiply(
                    offsets.reshape(flat), _rotation(w_other).reshape(flat)
                ).reshape(many)
                if translate:
                    target[..., 3, :3] = w_other[..., 3, :3] + d["_translate"][:, i]

                # a true inverse: a scaled parent's world is not orthogonal
                local = target @ np.linalg.inv(parent_world[:, solved])

                # TransformData.quaternion's setter, step for step: the chain
                # is RO * R * JO, solved for the pure rotate R
                Q      = matrix_to_quaternion(_rotation(local).reshape(flat))
                M      = quaternion_to_matrix(Q).reshape(many)
                orient = np.ascontiguousarray(np.broadcast_to(JO[:, i], many))
                L = local_matrix(
                    (RO_inverse[:, i] @ M).reshape(flat), orient.reshape(flat)
                )
                rotate[:, i] = _matrices_to_euler(L.reshape(many), orders[i])
                scale[:, i]  = s_source[:, k] - 1 + base_delta[:, k]
                if translate:
                    offset[:, i] = local[..., 3, :3]

            # the level's world for the children below it, factor for factor
            # S * RO * R * JO * IS * T as TransformData.matrix composes it
            inverse                 = np.ones((frames, len(idx), 3))
            compensated             = rooted & compensate[idx]
            inverse[:, compensated] = scale[:, above[compensated]]

            local_world = (
                _diagonal(scale[:, idx])
                @ RO[:, idx]
                @ _euler_matrices(rotate[:, idx], orders[idx])
                @ JO[:, idx]
                @ _diagonal(1.0 / inverse)
                @ _translation(offset[:, idx])
            )
            world[:, idx] = local_world @ parent_world

        # each frame picked its euler branch on its own; the levels below
        # were solved under the rotate as solved, as they are frame by frame
        rotate[:, rows] = _filtered(rotate[:, rows], orders[rows])
        return scale, rotate, offset

    # --- serialization --- #
    def _subclip(self, span: slice) -> "ClipData":
        """a new clip holding only the frames in ``span``

        The cut always copies. A plain stride would give a numpy view but a
        negative one would not, and a result that sometimes writes back
        through to the parent is worse than one that never does. The cut
        holds a copy of the rest pose, which every frame of the file shares.
        """
        self._realign()
        first = span.indices(self.frame_count)[0]

        # the clip itself, not its list: its record is the clip's, whoever
        # owns a node it lists
        new = self.__class__(
            self, start_frame=self._start_frame + first, fps=self._fps
        )
        new._blocks    = {c: np.array(b[span]) for c, b in self._blocks.items()}
        new._columns   = list(self._columns)
        new._rest_pose = self._rest_pose_copy()
        new._bind(0)
        return new

    def copy(self) -> "ClipData":
        new            = self._copy_frames()
        new._rest_pose = self._rest_pose_copy()
        return new

    def _copy_frames(self) -> "ClipData":
        """a copy of the nodes, the timebase, every frame and the record of
        the axes and unit, with no rest pose"""
        self._realign()
        # the clip itself, not its list, as a cut of frames
        new         = self.__class__(self, start_frame=self._start_frame, fps=self._fps)
        new._blocks = {c: b.copy() for c, b in self._blocks.items()}
        # the blocks are installed behind __init__, so the column list has to
        # come with them: a copy left claiming no columns would have its next
        # realign reseed every frame from the one pose the nodes are bound to
        new._columns = list(self._columns)
        new._bind(self._frame)
        return new

    def _rest_pose_copy(self) -> Optional[HierarchyData]:
        """a copy of the rest pose, for a copy or a cut of the clip; None when
        the clip holds none. ValueError for one recording other axes or
        units than the clip, relabelled after it was set"""
        _check_rest_pose(self._rest_pose, self)
        return None if self._rest_pose is None else self._rest_pose.copy()

    def __deepcopy__(self, memo):
        # the blocks, columns and bound frame come only through copy()
        new            = self.copy()
        memo[id(self)] = new
        return new

    @classmethod
    def load_glb(
        cls,
        filename:     str,
        scale_factor: float = 100.0,
        animation:    int   = 0,
        fps:          float = 24.0,
        start_frame:  int   = 0,
    ) -> "ClipData":
        """builds a clip from a glb's animation

        glTF keys every property on its own timeline and stores rotation as
        quaternions, so there are no frames to read: the samplers are
        resampled onto a uniform grid at ``fps`` and converted once per
        frame. That conversion picks an euler branch per frame, so the
        result goes through ``euler_filter``.

        The grid starts on the earliest key of the channels read. Keys on
        whole frames give exactly that many frames, both ends included --
        keys at frames 0 and 24 give 25 -- although glTF stores key times
        as float32 seconds, a hair off the frames: a span within one
        float32 spacing of a whole number of frames, the spacing taken at
        the larger in size of the first and last key times (the last key's
        for any clip from 0 on), counts as that number. On a clip of two
        frames or more, when the last key sits a hair after the last frame,
        that frame is sampled on the key itself, so a held (STEP) last key
        still shows, unless a held channel's own key sits exactly on that
        frame, which then keeps it. A last key further between frames adds
        the frame after it. Either way a clip of two frames or more ends on
        or past its last key, so the tail of the animation is inside the
        grid, save a held key a hair after a key of its channel exactly on
        the last frame: that key shows on no frame. Keys all within one
        float32 spacing of each other give one frame, on the earliest key.

        A held (STEP) key shows from the frame it was keyed for. float32
        can store it a hair after that frame's time, so a key up to one
        float32 spacing after a frame, the spacing taken at the key's own
        time, counts as on that frame, and of two keys on one frame that
        way the later shows. A key exactly on a frame keeps it, the last
        frame too, though a second key sits a spacing after it, so a one
        frame clip of two such keys shows the first. Exactly on a frame
        means on the grid's time rounded to float32, whichever side of that
        time float32 put the key: float32 stores a third of the frame times
        at 24 fps a hair early, frame 5 as 0.2083333284 s for
        0.2083333333 s, and such a key keeps its frame too. The grid starts
        on the earliest key's float32 time; when that time rounded up and a
        key's own time rounded down, the key can sit half a spacing or more
        before the grid's frame, off its float32 value, and a second key a
        spacing after it, on that value, then shows there: 14 of 1,621
        generated test files hold such a pair, the key 0.5 to 0.93 of its
        spacings before the frame. The grid's first key is float32 too, so
        a key truly between frames is taken in up to about two spacings
        after a frame: in clips up to 40 minutes in, 0.01 frame after one
        at 60 fps from 17 minutes in and at any rate from 34, never 0.03
        frame or more. An exporter that computes key times in float32
        arithmetic can store a key more than a spacing after its frame, so
        such a held key can still show a frame late, as such a span can
        still count a frame long. A key further after a frame, half a frame
        say, shows from the next frame on. Linear and cubic channels are
        read at the frames' times as they are, with no snap, their last
        frame on the last key as above.

        A file with no animation gives a one frame clip rather than an
        error -- it is an accurate reading of a rig that does not move.
        A channel on a node the rig does not read, such as a mesh, animates
        nothing, even when a joint shares that node's name.
        """
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")

        rig   = HierarchyData.load_glb(filename, scale_factor=scale_factor)
        model = load_model(os.path.expanduser(filename))

        animations = model.animations or []
        if animation and not -len(animations) <= animation < len(animations):
            raise IndexError(
                f"animation {animation} out of range, the file holds {len(animations)}"
            )

        column   = _glb_columns(model, rig, filename)

        channels = []
        for channel in animations[animation].channels if animations else []:
            if channel.path not in ("scale", "rotation", "translation"):
                continue
            if channel.node not in column:
                continue

            channels.append((channel, column[channel.node]))

        # the grid spans the channels actually used: a sampler belonging to a
        # node the rig dropped would otherwise stretch it
        spans = [
            np.asarray(
                animations[animation].samplers[channel.sampler].keyframe_times,
                dtype=float,
            ).ravel()
            for channel, _ in channels
        ]
        spans = [keys for keys in spans if keys.size]

        if not spans:
            return cls(rig, frames=1, start_frame=start_frame, fps=fps)

        first  = min(keys[0] for keys in spans)
        last   = max(keys[-1] for keys in spans)

        frames = _glb_frame_count(first, last, fps)
        grid   = first + np.arange(frames) / fps
        times  = grid.copy()
        if frames > 1:
            # the last frame can fall a hair before the last key: float32
            # stored the key a hair late, or float64 rounded the frame's time
            # a hair short of it. That frame samples the key itself, so a
            # held last key still shows, unless a held channel's own key sits
            # exactly on the frame's time (below)
            times[-1] = max(times[-1], last)

        clip = cls(rig, frames=frames, start_frame=start_frame, fps=fps)

        # seeded with the rest pose, which is what unanimated nodes keep
        scale     = np.array(clip.frames.scale)
        rotate    = np.array(clip.frames.rotate)
        translate = np.array(clip.frames.translate)

        for channel, index in channels:
            sampler = animations[animation].samplers[channel.sampler]

            # a held key exactly on the last frame's time, its float32 value,
            # keeps that frame, as on any other frame, though a second key
            # sits a spacing after it: that channel reads the last frame on
            # the grid
            at = times
            if sampler.interpolation == "STEP":
                held = np.asarray(sampler.keyframe_times, dtype=float)
                if np.any(held == float(np.float32(grid[-1]))):
                    at = grid

            if channel.path == "translation":
                translate[:, index] = _sample_gltf(sampler, at, 3) * scale_factor
            elif channel.path == "scale":
                scale[:, index] = _sample_gltf(sampler, at, 3)
            else:
                # the rest branch puts the node's orientation in joint_orient,
                # so rotate carries the delta from it and nothing else
                node = clip[index]
                delta = q_multiply(
                    q_conjugate(node.quaternion),
                    q_normalize(_sample_gltf(sampler, at, 4)),
                )
                rotate[:, index] = np.degrees(
                    quaternion_to_euler(delta, node.rotate_order)
                )

        orders                = np.asarray(clip.rotate_order, dtype=np.intp)
        clip.frames.scale     = scale
        clip.frames.rotate    = np.degrees(euler_filter(np.radians(rotate), orders))
        clip.frames.translate = translate
        clip.frame            = 0
        return clip

    @classmethod
    def load_fbx(
        cls,
        filename:     str,
        scale_factor: Optional[float] = None,
        take:         Optional[int]   = None,
        fps:          Optional[float] = None,
        start_frame:  Optional[int]   = None,
        end_frame:    Optional[int]   = None,
        axis_system:  Optional[str]   = "maya_y_up",
        node_types:   Optional[tuple] = None,
    ) -> "ClipData":
        """builds a clip from an fbx's animation take

        ``scale_factor`` None, the default, converts the file's unit to
        centimeters, a number is the caller's own factor, as in
        :meth:`HierarchyData.load_fbx`: the static translations and the
        translation curves are multiplied by it, rotations and scales are
        not, and a file that holds its unit as a scale on the scene root's
        children loads as the cm file it was written from, on every frame.
        The export converts the curves of each take's first layer alone, so
        on a root that holds the unit only that layer's translate is
        multiplied, and only that layer's scale divided by the unit on the
        root and on each node under it that does not inherit its parent's
        scale, before the layers are blended; the later layers' curves are
        read as they are. A root the export found scaled other than 1 holds
        no unit: its translate is multiplied after the blend, the later
        layers' cm values with the first layer's, so the root and every node
        under it are off by s - 1 times each later layer's own translate
        times its weight, s the cm one unit of the file holds, on each
        channel its curve node holds: what an additive layer adds, an
        override or passthrough layer's whole value; and a scale an override
        or passthrough layer keys on that root, or on a node under it that
        does not inherit its parent's scale, is read as the cm value it is,
        every node under it off, as :meth:`HierarchyData.load_fbx` says.

        ``axis_system`` "maya_y_up", the default, loads the take in Maya's
        Y-up axes, as :meth:`HierarchyData.load_fbx` loads the rig: every
        node's world matrix, on every frame, is the file's times the turn
        from the file's axis system to Maya's, the one ConvertScene gives,
        root travel included. Maya's FBX importer agrees on a file whose up
        axis is x or z, but on a skeleton root it imports as a transform and on
        a node below the roots that is no camera or light whose rotation is
        inactive, in another order than xyz, as :meth:`HierarchyData.load_fbx`
        says, and where the list below says, a root that is no joint whose
        rotation a layer blends among them, and leaves one whose up axis is y
        in its own axes. In a file of any axes it also parts from the clip
        where it keys at another frame rate than the file's, as the list says.
        The clip parts from both on a root a layer above the base translates,
        additive, override or passthrough: it keeps that root's translate in
        the file's world turned, the motion as the file authors it, where
        ConvertScene leaves that layer's translate in the file's axes, and so
        does Maya's importer wherever the base layer holds that translate
        (within 7.7e-7 of ConvertScene; where the base layer holds no curve
        node for it, Maya parts from ConvertScene as the first live layer item
        below says). The clip sits 2.8 to 49 off Maya's import of such a root
        on the files measured, a gap that grows with how far the layer moves
        it; Maya's import of a Z-up scene it exported, a group root an additive
        layer translates, sits 45 off that scene turned, the clip 1.5 off it,
        as Maya's import of the same scene exported Y-up does, and puts that
        group's 3x3, in xyz, 1.41 off ConvertScene's and the clip's, as the
        list below says. A root whose rotation is inactive still blends its
        rotation, blended by layer, as ConvertScene's xyz node, as Maya's
        importer does a group's, a locator's or a joint's: a zyx group root the
        layer also translates and turns has its 3x3 1.70 off the file's world
        turned and within 1.7e-7 of Maya's FBXImport into a scene at the file's
        frame rate, and its translate the file's world turned, 49 off Maya's
        import in a Z-up file and 30 in an X-up one. A joint root takes the
        turn in its joint orient and its translate curves, keeping its rotate
        curves as authored; any other root has its rotate solved again on every
        frame in its rotate order, through euler_filter along the frames, frame
        0 as solved, and its translate turned. A file already in Maya's Y-up
        loads as it is, a left handed one raises ValueError, and
        ``axis_system=None`` keeps the file's own axes.

        ``node_types`` keeps the kinds of node :meth:`HierarchyData.load_fbx`
        keeps, and what that read folds into the nodes it keeps, the nodes
        it drops and the pivots, offsets and pre and post rotations a node's
        channels can't hold, is folded in on every frame: the dropped nodes
        are sampled with the kept ones, every frame at once. A joint under a
        dropped node the take moves keeps its rotate curves as authored and
        takes that node in a joint orient that moves with it, but on a frame
        the dropped nodes scale it unevenly under a turn, where its rotate
        is solved again from the matrix and its whole rotate curve then goes
        through euler_filter; any other node has its rotate solved again on
        every frame; a node whose pivot its rotate or scale moves takes it in
        its translate on every frame.

        A node's inherit type is read as :meth:`HierarchyData.load_fbx` reads
        it, on every frame: a node that is no joint stored eInheritRrSs or
        eInheritRrs, read as eInheritRSrs, parts from the sdk and from Maya's
        import under a parent scaled as that says, a unit root's scale counting
        as 1, and the nodes under it with it. Scaled (1, 1.5, 0.75) and keyed
        (20, 40, 60), (35, -10, 80), (-25, 50, 10), (60, 20, -40) and
        (5, -70, 30) over 5 frames under a group turned (0, 30, 0) and scaled
        (1, 2, 3), the eInheritRrSs node is 1.63 off in the 3x3 and the
        eInheritRrs one 2.11, a locator at (0.5, 1, -2) under each 2.98 and
        2.88 in translate; under one scaled (2, 2, 2) the eInheritRrs node
        1.35, its locator 2.13. A joint stored eInheritRrSs is read as Maya's
        importer reads it, the two within 1.8e-15 and both 1.63 off the sdk
        there.

        The clip's :attr:`rest_pose` is the file's static pose, the nodes'
        own values outside the take (the world matrices
        ``FbxNode.EvaluateGlobalTransform`` gives with no time), taken from
        the same read: a :class:`HierarchyData` with the clip's names and
        node order, loaded as :meth:`HierarchyData.load_fbx` loads the file,
        in the same unit and axes. For an Unreal animation it is the
        skeleton's bind pose, which frame 0 can be far from. The clip and
        its rest pose record the axes and unit the read produced, as
        :meth:`HierarchyData.load_fbx`'s rig does (:attr:`axis_system`,
        :attr:`unit`), which :meth:`save_fbx` labels a frame's file with.

        The three local channels are sampled separately rather than
        decomposed out of ``EvaluateLocalTransform``: that matrix has already
        folded in joint orient, rotate axis and the inherit type, so pulling
        euler angles back out of it would not give the curves the file holds.

        The curves the file holds are not euler filtered, unlike the glb
        path. Fbx stores euler curves, so the branches are the ones the file
        was authored with and rewriting them would change the animation. A
        rotation an animation layer blends by layer is rebuilt from
        quaternions frame by frame, by the sdk as here, so that curve goes
        through euler_filter in the node's rotate order, frame 0 as the sdk
        gives it. So does the rotate of a root that is not a joint, solved
        again for Maya's axes (``axis_system``), the rotate of any node that
        is no joint, solved again for a dropped node above it or its own
        pre and post rotation, a joint's orient, solved again for a dropped
        node above it, in xyz, and the rotate of a joint a dropped node
        above it scales unevenly under a turn, solved again on those frames
        (``node_types``).

        Every frame is sampled at once: each curve's keys are read once and
        evaluated on the frame grid the way the sdk evaluates them, the same
        values on keys and to the sdk's single precision rounding between
        them, past them too where a curve repeats, mirrors or repeats with
        an offset (Maya's cycle with offset); keep slope holds the end key,
        or carries its slope on when its count is 2**32 - 1, as Maya writes
        it, as this sdk evaluates it, which one evaluation of the curve
        checks (an sdk that evaluates it otherwise raises ValueError). The
        one exception between keys is a weighted span whose two handles
        crowd one end (one weight at most about 0.01, the other about 0.88
        or more, the smaller the one the lower the other), where the sdk's
        own time solve stops short of the exact one taken here, by up to
        0.55 of the curve's range; Maya's own import gives the values taken
        here, and there the loader follows Maya, not the sdk. The take's
        animation layers are blended here as the sdk blends them: from the
        first layer that is not muted and holds anything, a channel it holds
        no curve node for starting from 0, scaling too, then each later one
        that is not muted by its weight (animated or not), blend mode,
        rotation and scaling accumulation; a channel no live layer holds
        keeps the rest pose. A rotation blended a hair off gimbal agrees to
        a few times 1e-16 / cos(middle angle) radians, the rounding either
        side loses there; within about 1e-15 of the cosine the sdk splits
        the turn at, each side's rounding picks the branch, and the two can
        be whole rotations apart. No frame is asked of the sdk.

        Maya's own evaluation of a take differs from the sdk's, and so from
        the clip, where:

        - a rotation is blended by layer on a node whose rotate order is not
          xyz, as the file stores it, rotation active or not, though a node
          whose rotation is inactive loads with rotate_order 0 (a root of the
          file whose rotation is inactive, kept or dropped, blends as an xyz
          node in a file the clip turns into Maya's axes, as ConvertScene
          turns its rotation on in xyz; a node kept under a dropped root is no
          root of the file): the sdk hands the x, y and z angles to its euler
          code as they are, so an even order (yzx, zxy) blends as an xyz node
          does and an odd one (xzy, yxz, zyx) as the sdk blends a zyx node,
          whose additive layer turns on the other side of what is below:
          M(below) @ M(layer), both matrices of zyx angles, where Maya gives
          M(layer) @ M(below) in the node's own order. Maya blends every node
          in its own order the way it blends xyz. Additive layers agree only
          on xyz (but for the sign cases below), passthrough layers on xyz and
          zyx; the others are up to 2.0 apart as matrices;
        - Maya's FBXImport merges the layers, as it does by default, on a node
          that turns in xzy, yzx, yxz or zxy, a group or a joint, root or not
          (but a root of the next item), in a Y-up and a Z-up file alike: on
          some frames, by channel or by layer, the merged turn parts from the
          sdk's, up to 2.0 apart as matrices. By channel such a frame holds the
          sdk's angles with x and z turned by 180 degrees and y by 180 - y, the
          same turn in xyz and zyx alone (a frame the sdk turns by
          (-100, 68, -25), after one at (120, -54, 205), merges to
          (80, 112, 155), 1.77 to 1.99 apart), where the layers kept agree; by
          layer, keys of 120 to 128 degrees about y part on every frame, 1.23
          to 1.99 apart. Merged, a node in xyz or zyx agrees, and so did every
          order on the smaller turns measured, by layer too, which kept parts
          as the item above says, but for the single precision values the
          curves it merges, bakes or solves hold, the rotate's as well as the
          scale's and the translate's, which part from the sdk's the more the
          larger they are. The angles that count are, by channel, the sums of
          the keys, which Maya rounds to single precision, or the angles its
          curve holds where those are larger: it starts the curve on the take's
          first frame at the angles nearest 0 that turn the node as the sums
          do, whole turns off and on either branch (sums of (330, 89.7, 330)
          held as (-30, 89.7, -30), of (3500, 89.7, 3500) as (80, 90.3, 80)),
          and its continuity can carry the curve on past the sums where the
          keys jump (random keys summing to under 245 degrees: curves holding
          up to 1210); by layer, and where Maya solves the rotate (a root of
          the next item, a node whose pre rotation it bakes), the angles it
          solves, and more where that continuity carries them on. On the files
          measured, as matrices on a node of world scale 1, within 3.7e-7 on
          angles up to 250 degrees, 5.7e-7 up to 500, 1.3e-6 up to 1000 and
          4.3e-6 up to 3600, by channel within the single precision spacing of
          the largest such angle in radians (2.642e-7 on sums of 205 degrees
          and 4.228e-6 on sums of 3505, 0.992 of it, where single precision
          rounds the x and z sums by almost half of it each near y = 90), and
          within 2.2e-15 on keys in whole degrees, whose sums single precision
          holds. A scale on the node or above it multiplies that, on the files
          measured, by up to the largest axis of the node's world scale (scales
          of 2 to 10 and (1, 2, 3): a node scaled 3.5, or under one scaled 3.5,
          7.0e-7 by channel and 8.2e-7 by layer on angles up to 250; one whose
          scale is keyed to 10, 9.0e-7 on angles to 193); a node under it, and
          with a rotate pivot the node itself, is off in translate by about
          that gap at scale 1 times its world distance from the node's rotate
          pivot, up to 1.7 times that on the files measured and 2.9 under a
          group scaled (1, 2, 3) (a locator 19.7 from a node at scale 1, 3.7e-6
          off on angles up to 250, the one under the node keyed to 10, 1.7e-6);
          a translate both layers key is up to 6.0e-8 (2^-24) of its size off
          (3.8e-6 at 71). But a frame whose blended y angle is 90 or -90, or a
          whole number of turns from either (Maya holds 270 as -90 and -270 as
          90), merges by channel to angles that hold its turn in xyz alone,
          x - z kept at 90 and x + z at -90: those nearest the key before's
          (the frame before's at the file's rate), and on a take's first frame,
          which has none before it, those nearest 0 ((25, 90, 30) merges to
          (-2.5, 90, 2.5)). On the files measured, y from -810 to 810, 8.5e-3
          to 0.89 apart in xzy, yzx, yxz and zxy and 1.4e-2 to 1.4 in zyx, the
          more the further x and z move, and up to 2.0 where the merged angles
          also sit a half turn off the keys as above, x and z turned by 180
          degrees and y by 180 - y, as the frames after such a first frame can
          go on doing (a take whose first frame, (-150, -90, 170), merges to
          (10, -90, 10): 1.58 to 1.97 apart in those four orders on every
          frame). A node that is no joint with a pre rotation parts as the item
          on it below says;
        - a layer blends the rotation of a root that is no joint in a file
          whose up axis is x or z, which Maya's importer turns into its Y-up:
          it can part from ConvertScene there in any rotate order, by channel
          or by layer, the layers kept or merged, up to 2.0 apart as matrices
          (Maya's own Z-up export of a group root in xyz, its default order,
          under an additive animLayer, which it writes by channel: 1.41 apart,
          the locator under it 2.25 off). On files the sdk wrote, by channel
          0.40 to 1.96 apart, a root whose rotation is inactive as a root in
          xyz, which ConvertScene makes it, in orders that hang on the file's
          axes and on the turns: on the smaller turns measured and on a turn of
          120 to 128 degrees about y, xyz, xzy, zxy and yxz in Maya's Z-up axes
          and zyx, xzy, yzx and zxy in an X-up file whose front is z, the other
          two agreeing; in a Z-up file whose front is x, in an X-up one whose
          front is y, and on keys that turn far from one frame to the next (5
          frames, up to 230 degrees apart on an axis), every order. By layer,
          in every order but xyz, 0.40 to 1.93 apart, merged on the smaller
          turns too, but yxz in that X-up file, and kept as the first item
          says. On every file and turn measured, a root in xyz, or one whose
          rotation is inactive, agrees by layer but for the single precision of
          the curves Maya solves, as the second item says, and where it holds
          one level (the next item); and a joint root parts only where any node
          does (the two items above). The clip keeps ConvertScene's values
          there, within 7.1e-15;
        - Maya's FBXImport keys the curves it solves or merges at the rate its
          FBXImportResamplingRateSource option names, by default "Scene", the
          rate the scene runs at when the import starts; the import then sets
          the scene to the file's rate. It solves the rotate of a root that is
          no joint in a file whose up axis is x or z; the rotate of a node that
          is no joint with a pre rotation it reads (rotation active, or on a
          camera or a light), which it bakes into that rotate, in a y-up file
          too; and the rotate of a node that is no camera or light whose
          rotation is inactive, in another order than xyz, but a root of a file
          whose up axis is x or z, which it turns by neither order
          (``node_types``). It merges a take's layers, as it does by default,
          into new curves on the nodes they hold, a joint's rotate and
          translate too. The other curves keep the file's keys, of a single
          layer or of kept layers: a joint's, its pre rotation kept as its
          joint orient, a camera's or a light's with no pre rotation, a node's
          whose post rotation it keeps as its rotate axis, translates, pivots
          and offsets. From a scene at another rate, the keys it makes sit on
          the frames of that rate, not the file's, and on the file's frames
          between them Maya parts from the clip by amounts that hang on both
          rates. On the files measured, a 30 fps file imported from Maya's
          default 24 fps scene, its keys 1.25 of the file's frames apart: a
          root that is no joint of an X- or Z-up file up to 0.17 apart as
          matrices, the node under it up to 0.29 off; a node with a pre
          rotation up to 6.9e-2, the locator under it up to 0.11; the nodes of
          a merged take up to 1.6e-2, the locators under them up to 3.9e-2.
          From a 25 fps scene: a group root of a Z-up file 0.16 apart, the
          locator under it 0.19; the nodes of a merged take up to 6.4e-2, the
          locators under them 9.8e-2. A node it turns by neither order parts
          from the clip at another rate by its gap at the file's rate and by a
          share of its own, which grows with how far the angles Maya solves
          bend or jump between that rate's frames, not with that gap, as
          :meth:`HierarchyData.load_fbx` says (a smooth turn about one axis,
          which agrees at the file's rate, 1.0e-2 to 6.1e-2 apart on the files
          measured). With that option set to "File", or from a scene at the
          file's rate or at a whole multiple of it, it keys every frame of the
          file (on the files measured, from 60, 120, 150 and 240 fps scenes for
          a 30 fps file and 48, 120 and 240 fps ones for a 24 fps file) and
          agrees on them but for the single precision of the curves it solves
          or merges, as the second item says (as matrices within 3.7e-7 on a
          node of world scale 1 on angles up to 250 degrees, more on larger
          angles, under a scale and on the nodes under it), and where a curve
          it bakes or solves moves by less than about 1e-2 degrees a frame of
          the rate it keys at, which it can hold level over runs of frames, a
          faster scene holding more: on the files measured, moving by 1e-4 to
          6e-3 degrees a frame, up to 3.7e-4 apart as matrices on takes of 11
          and 101 frames, and from 120 to 240 fps scenes up to 3.0e-4 more than
          at the file's rate on curves moving by about 1e-2 degrees a frame of
          the file and 1.1e-4 on a root of an X-up file; and from a 60 fps
          scene one curve of the 3,372 it made lost its key on the take's last
          frame, 1.3e-4 apart there. From a 90 fps scene, though, every curve
          it made held keys on the take's first and last frames alone: on the
          30 and 24 fps files measured up to 2.0 apart as matrices on nodes of
          world scale 1, the nodes under them up to 3.6 off in translate. On a
          node it turns by neither order it is as far apart at the file's rate
          as :meth:`HierarchyData.load_fbx` says, and it parts where another
          item of this list says;
        - a layer above the base turns a node that is no joint whose pre
          rotation Maya's FBXImport bakes into its rotate (the item above),
          leaving the layer's own angles as they are. An additive layer adds
          them to the baked ones by channel, the layers kept or merged, where
          the sdk adds them to the node's own under the pre rotation: that
          parts in every rotate order, but for a pre rotation about one axis
          alone in the two orders that turn about that axis last (xyz and yxz
          for z, yzx and zyx for x, xzy and zxy for y), which agree on the
          smaller turns (on the files measured within 1.2e-15 in nine such
          cases of ten, the others, on the larger turns, 9.5e-2 to 2.0 apart).
          By layer it parts in every order but xyz, merged too, but for yxz
          under a pre rotation about z alone merged on the smaller turns
          measured (within 1.6e-7), where the same node with no pre rotation
          agrees merged on the turns measured. On the files measured, pre
          rotations about mixed axes that turn 7 to 161 degrees, by channel
          4.3e-3 to 1.99 apart as matrices, by layer merged 6.5e-3 to 2.0, more
          as the pre rotation and the turns grow; by layer, a node in xyz
          agrees, kept or merged, but for the single precision of the curve
          Maya bakes and where it holds that curve level (the item above):
          within 4.5e-7 on keys of up to 330 degrees and 4.5e-6 on keys to 3500
          on a node of world scale 1, a scale multiplying these as the second
          item says, and 1.2e-4 on a baked curve held level over three frames;
          a post rotation, which Maya keeps as the rotate axis, parts only
          where a node with none does. An override layer's angles take the
          place of the baked ones in every order, by layer merged at every
          weight measured or kept at 100, and by channel at 100, and so do a
          passthrough layer's at weight 100: the pre rotation is lost where the
          layer holds the rotate, Maya turning the node by the layer's angles
          alone where the sdk turns it by them under the pre rotation, apart by
          about what the pre rotation turns it. By channel under 100 the gap
          stays near that one; kept by layer under 100 Maya weighs the
          override, as the item on that says, so the gap holds both. On the
          files measured, weights 10 to 100: by layer merged, at 100 and by
          channel, 8.7e-2 to 0.12 as matrices for a pre rotation of 7 degrees,
          0.42 to 0.68 for 36 to 43, 0.75 to 0.98 for 60 to 66 and 1.35 to 1.72
          for 120 to 161, by channel under 100 within 0.21 of the gap the
          layer's angles alone give; kept by layer under 100, 0.11 to 1.99, and
          0.11 to 1.92 for 7 degrees. Under 100 a passthrough layer's angles
          mix with the baked ones, which hold the pre rotation, so part of it
          stays, the more the lower the weight: on the smaller pre rotations
          and turns the gap is about the weight times the gap at 100 (by
          channel, a pre rotation of 7 degrees on the smaller keys, 2.1e-2 to
          2.6e-2 at 25, 4.5e-2 to 5.1e-2 at 50 and 6.9e-2 to 7.5e-2 at 75), and
          over both modes and every pre rotation measured 2.1e-2 to 2.0 at 25,
          4.5e-2 to 2.0 at 50 and 6.9e-2 to 2.0 at 75, where the same node with
          no pre rotation agrees but where another item of this list says. In
          every mode measured such a node turns about its rotate pivot, so a
          node under it, and with a pivot the node itself, is off in translate
          by up to twice its world distance from that pivot times the largest
          over the smallest singular value of the parent's world 3x3, as
          :meth:`HierarchyData.load_fbx` says, and a node on the pivot agrees:
          on the files measured, under even scales, up to 2.0 times that
          distance (3.72 at 1.87 and 38.9 at 19.7; with pivots 0.5 to 7.1 away,
          the node itself 4.00 off with one 2 away), under a group scaled
          (1, 0.5, 2) up to 3.48 times it, and a node on the pivot within
          5.7e-14;
        - an additive layer blends a rotation by layer at a weight other than
          0 or 100, and the quaternion Maya builds from the layer's angles
          (qz qy qx of the half angles on an xyz node) has the other sign
          from the one the sdk solves from its matrix: the scaled turn goes
          round the other way (up to 2.0 apart as matrices, in every rotate
          order, xyz included). The sdk's w is positive on every turn under
          120 degrees, so such a turn parts whenever Maya's w is negative:
          an angle keyed past 180 (a 20 degree turn keyed as (380, 0, 0) is
          1.9 apart at weight 50) or all three angles near +-180 (a 40 degree
          turn Maya keyed as (150.4, 154.1, -176.7) is 1.5 apart). Over 120
          degrees either sign can come out;
        - a rotation is overridden by layer at a weight under 100: the sdk
          takes the layer's rotation whatever the weight, and Maya weighs it
          (0.86 apart as matrices at weight 50);
        - a negative scale is multiplied by a layer: the sdk takes every
          term unsigned, and Maya keeps the sign (1.6 apart in scale on
          Maya's own export);
        - the first live layer holds no curve node for a channel a later layer
          holds: the sdk starts it from 0, and Maya does not (0.82 units apart
          in translation, 1.94 in scale);
        - a layer is soloed: solo changes nothing here, and Maya honours it;
        - the base layer is muted: it is skipped here, but Maya's exporter
          does not write that mute, so a Maya take whose base layer is muted
          loads with it playing;
        - a curve has weighted tangents: Maya's exporter does not write them
          as the scene holds them, so between keys a Maya export loads up to
          about 2.5e-2 of the curve's range off what Maya showed (over 5e-3 on
          7 of 72 test exports, whatever the infinity); Maya's own import of
          the file gives the clip's values;
        - the curve has linear infinity: past the keys the clip carries on
          along the end key's derivative as the file holds it, where Maya
          follows the key's outer tangent. Unweighted spline, clamped, flat,
          stepped, plateau and auto tangents agree; linear, fixed and every
          weighted tangent do not (up to 158 units 48 frames out on Maya's
          own exports). A weighted one on no layer is the export again:
          Maya's own import of it carries on past the keys as the clip does,
          but for linear and stepped tangents.

        ``take`` defaults to the first one that holds a key rather than to
        index 0, then, in a file with no key at all, to the first that holds
        a curve node, with or without a curve, then to index 0. A file can
        carry takes holding no curves, or curve nodes and curves with no key
        on them (an empty "Take 001" ahead of the animation), and reading one
        back gives that take's own static values on every frame, the rest
        pose where its curve nodes hold the node's own -- a silent answer
        that looks like a broken loader rather than an empty take. A curve with
        no key reads at its curve node's value, where the sdk's evaluator takes
        the curve's own default, and so does Maya where the curve is in the
        file's active take and the last take does not key its channel (a curve
        with no key in any other take, the last included, Maya does not read):
        under a curve node holding Y at 55, a Y curve with no key whose default
        is 99 reads 55 in the clip, 99 in the sdk's evaluator and, there, in
        Maya. The frames left out default to the take picked. Pass an index to
        override, including onto an empty take.

        Maya 2025's FBXImport, with its default settings, picks otherwise. It
        loads its curves from the file's last take, keyed or not, so an empty
        last take imports no curve. That is the last take the file's Takes
        section lists, not the last take object, which ``take=-1`` loads; the
        two are the same take wherever the file lists its takes in the order it
        holds them, as the sdk and Maya write them, and only a file edited by
        hand, its takes reordered or one left out of its Takes section, parts
        them (of the test files with two or more takes imported in Maya 2025,
        the 148 written with the sdk and the 6 exported by Maya list them in
        that order, and the 23 that do not were edited by hand); in a file with
        no Takes section Maya loads the last take object's curves, the take
        ``take=-1`` loads. Its playback range is the span of the file's active
        take: the take the file's ``ActiveAnimStackName`` names, or the first
        take object, the one ``take=0`` loads, when that name is empty, missing
        or names no take. A channel the last take does not key holds the value
        the active take gives it at frame 0. So the clip's curves are Maya's
        where the first take holding a key is the last take, or where no take
        holds one, though its frames, and a channel no take keys, can still
        differ there (Maya holds the active take's value at frame 0);
        ``take=-1`` loads the last take's curves, over that take's own frames,
        with a channel it does not key at that take's own value for it (its
        curve node's, the rest pose where it holds none). The clip's frames are
        the picked take's, where Maya's playback range is the active take's. On
        5 of 32 test files imported in Maya 2025, 3.0.1 moved the frames off
        Maya's playback range: cgmath 3.0.0 loaded the empty first take, the
        rest pose over Maya's playback range, and 3.0.1 loads the keyed take's
        curves, as Maya does, over that take's frames:

        - an empty first take over frames 0-48, holding a curve node and
          recorded as the active take, then a take keyed over frames 0-10:
          Maya 0-48, the clip 0-10;
        - the same file, with no active take recorded: Maya 0-48, the clip
          0-10;
        - the same file, its empty take over frames 0-40: Maya 0-40, the clip
          0-10;
        - the same file, its empty take holding a curve with no key on it:
          Maya 0-48, the clip 0-10;
        - Maya's own export, split into an empty take over frames 100-110 and
          a take keyed over frames 0-10 with its original take deleted, which
          records "Take 001", a name neither take has, as active: Maya
          100-110, the clip 0-10.

        ``start_frame`` and ``end_frame`` are frame numbers on the file's own
        timeline, at the clip's frame rate. Each one left out covers both the
        take's stored span and every key of every curve on every layer of the
        take: ``start_frame`` is the earlier of the span's start and the first
        key, ``end_frame`` the later of the span's end and the last key, out
        to the whole frame where either falls between two. Unreal stores the
        span a frame short of its last key, and a Maya export can store one
        wider than its keys; the clip keeps both. A time within 1e-6 of a
        frame of a whole frame is that frame, so float noise in a span or a
        key adds no frame. A take with no key covers its stored span. Each
        end defaults on its own: a ``start_frame`` passed alone ends the clip
        on the take's last frame, not the take's length after it, and one past
        that frame raises ValueError naming it; an ``end_frame`` passed alone
        before the take's first frame raises the same way. A file with no
        animation gives a one frame clip rather than an error.
        """
        if fps is not None and fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")

        if FBX is None:
            raise ImportError("fbx module not found")
        _fbx_check_scale_factor(scale_factor)
        _fbx_check_axis_system(axis_system)
        kinds = _fbx_check_node_types(node_types)

        # everything read from the scene dies with it: copy out inside the
        # block. One load of nodes, the whole read: the rest pose and the rig
        # the clip is built from, the same nodes twice, warn of a user
        # attribute a built-in name hides once a name, once the file is
        # read, and a read that raises, a take out of range or a frame past
        # or before the take's, warns of nothing
        with _NodesLoad(), open_fbx(filename, FBX) as scene:
            turn   = _fbx_axis_turn(scene, axis_system, filename)
            units  = _fbx_units(scene, scale_factor)
            record = _fbx_record(scene, axis_system, scale_factor)
            hierarchy_data, nodes, tree = _walk_fbx(scene, units, kinds, turn is not None)

            # the rest pose, the file's static pose from this same read, as
            # HierarchyData.load_fbx loads it: turned on a copy of the dicts,
            # since the clip's rig is built from them in the file's own axes
            rest = copy.deepcopy(hierarchy_data)
            if turn is not None:
                _fbx_turn_walk(rest, turn)

            criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimStack.ClassId)
            count    = scene.GetSrcObjectCount(criteria)

            # the rest pose and, with a take, the rig the clip is built from,
            # in the file's own axes, turned once every frame is sampled
            rest = _recorded(HierarchyData.from_dict(rest), record)
            rig  = HierarchyData.from_dict(hierarchy_data) if count else None

            if not count:
                clip            = cls(rest, frames=1, start_frame=start_frame or 0, fps=fps or 24.0)
                clip._rest_pose = rest
                return _recorded(clip, record)

            # typed: the binding can hand back a live wrapper of an earlier
            # scene's object for one that sits where it did
            stacks = [typed(scene.GetSrcObject(criteria, index), FBX.FbxAnimStack, FBX) for index in range(count)]

            # the first take holding a key, then the first holding a curve
            # node, with or without a curve: a take can hold curve nodes,
            # curves even, and not one key, and reads back as its own static
            # values on every frame, the rest pose where its curve nodes hold
            # the node's own (a curve with no key at its curve node's value,
            # where the sdk takes the curve's own default, and so does Maya
            # where the curve is in the file's active take and the last take
            # does not key its channel: _fbx_held)
            if take is None:
                stack = next((x for x in stacks if _fbx_key_ticks(x) is not None), None)
                if stack is None:
                    stack = next((x for x in stacks if _fbx_has_curves(x)), stacks[0])
            else:
                if not -count <= take < count:
                    raise IndexError(f"take {take} out of range, the file holds {count}")

                stack = stacks[take % count]

            # the evaluator reads whichever stack is current, not the one we hold
            scene.SetCurrentAnimationStack(stack)

            if fps is None:
                fps = _fbx_frame_rate(scene.GetGlobalSettings().GetTimeMode())
                if fps <= 0.0:
                    raise ValueError(
                        f"{filename!r} uses a time mode the fbx sdk reports no "
                        "frame rate for, so the frame rate has to be given: "
                        "load_fbx(..., fps=...)"
                    )

            # the stored span and every key of the take: Unreal stores the
            # span a frame short of its last key, and a Maya export can store
            # one wider than its keys
            per_second = _fbx_ticks_per_second()
            span       = stack.GetLocalTimeSpan()
            bounds     = [int(span.GetStart().Get()), int(span.GetStop().Get())]
            keys       = _fbx_key_ticks(stack)
            if keys is not None:
                bounds = [min(bounds[0], keys[0]), max(bounds[1], keys[1])]
            first, last = _fbx_whole_frames(bounds, fps, per_second).tolist()

            # out to whole frames, so both ends are inside the grid rather
            # than part of a frame past it
            start = int(np.floor(first)) if start_frame is None else int(start_frame)
            stop  = int(np.ceil(last)) if end_frame is None else int(end_frame)

            # each end defaults on its own, so one passed alone can fall past
            # the take's other end: the error names the end the take gave
            if stop < start:
                if end_frame is None:
                    raise ValueError(
                        f"start_frame {start} is past the take's last frame {stop}; "
                        "pass end_frame too"
                    )
                if start_frame is None:
                    raise ValueError(
                        f"end_frame {stop} is before the take's first frame {start}; "
                        "pass start_frame too"
                    )
                raise ValueError(f"end_frame {stop} is before start_frame {start}")

            clip         = cls(rig, frames=stop - start + 1, start_frame=start, fps=fps)
            column       = {uuid: index for index, uuid in enumerate(clip.uuid)}
            scale        = np.array(clip.frames.scale)
            rotate       = np.array(clip.frames.rotate)
            translate    = np.array(clip.frames.translate)
            joint_orient = np.array(clip.frames.joint_orient)

            # every node of the tree, the kept ones through the objects the
            # walk handed back and the dropped ones above them, sampled as the
            # file holds them, then folded every frame at once, as the walk
            # folds their static values
            uuids = list(hierarchy_data)
            sampled = [
                (nodes[uuids[kept]] if kept >= 0 else tree.nodes[position], position)
                for position, kept in enumerate(tree.kept.tolist())
            ]
            held = [
                np.repeat(values[None], clip.frame_count, axis=0)
                for values in (tree.translate, tree.rotate, tree.scale)
            ]

            # the frame grid in the sdk's ticks, every frame sampled at once
            ticks = _fbx_ticks((start + np.arange(clip.frame_count)) / fps, per_second)
            _fbx_sample_take(stack, sampled, ticks, tree, held)

        # a joint's orient as the fold leaves it, every frame: a node a
        # dropped node above moves only at its static values took the fold
        # in the rig the clip was built from
        folded = _fbx_fold(tree, *held)
        put    = np.array([column[uuid] for uuid in uuids], dtype=np.intp)
        joints = tree.joint[tree.columns]
        translate[:, put], rotate[:, put], scale[:, put] = folded[:3]
        joint_orient[:, put[joints]] = folded[3][:, joints]

        if turn is not None:
            roots  = [index for index, node in enumerate(clip) if node.parent_uuid is None]
            joints = [clip[index].node_type == "joint" for index in roots]
            orders = [clip[index].rotate_order for index in roots]

            turned = _fbx_turn_roots(
                turn, joints, orders, translate[:, roots], rotate[:, roots], joint_orient[:, roots]
            )
            translate[:, roots], rotate[:, roots], joint_orient[:, roots] = turned

        clip.frames.joint_orient = joint_orient
        clip.frames.scale        = scale
        clip.frames.rotate       = rotate
        clip.frames.translate    = translate
        clip.frame               = 0
        clip._rest_pose          = rest
        return _recorded(clip, record)

    def save_fbx(self, filename, zero_root=False, as_ascii=False):
        """saves the bound frame to a .fbx file, binary unless as_ascii is set

        The exporter writes one pose per node and never opens an animation
        stack, so a clip holding more than one frame has to say which frame
        it meant rather than quietly shipping whichever one is loaded. The
        file is labelled with the clip's :attr:`axis_system` and
        :attr:`unit`, as :meth:`HierarchyData.save_fbx` labels a rig's.
        """
        if self.frame_count > 1:
            raise ValueError(
                "save_fbx writes one pose but this clip holds "
                f"{self.frame_count} frames; cut the frame out first, "
                f"clip.frames[{self.frame}:{self.frame + 1}].save_fbx(...), "
                "or save it as npz, json or pkl to keep the whole clip"
            )

        super().save_fbx(filename, zero_root=zero_root, as_ascii=as_ascii)

    @classmethod
    def _channel_name(cls, channel: str) -> str:
        """a framed channel as a file and an error name it: its property's name"""
        return cls.DATA_LIST_CLASS._saved_names()[0].get(channel, channel)

    def to_dict(self) -> dict:
        # a rest pose relabelled after it was set refuses the save, and the
        # pickle, before anything is written: the load would refuse the file
        _check_rest_pose(self._rest_pose, self)
        # the saved block columns follow the saved node order
        self._realign()
        tree   = super().to_dict()
        blocks = tree.get("blocks")
        if isinstance(blocks, dict):
            tree["blocks"] = {self._channel_name(c): block for c, block in blocks.items()}
        # a clip with no rest pose saves as one did before it had any
        if tree.get("rest_pose", ...) is None:
            del tree["rest_pose"]
        return tree

    @classmethod
    def _from_state(cls, data: dict) -> "ClipData":
        # each channel back to its field's own name, "translate" to
        # "_translate"; the blocks hold only numbers: one array each now,
        # rather than a walk over every json row looking for objects to rebuild
        blocks = data.get("_blocks")
        if isinstance(blocks, dict):
            blocks = cls.DATA_LIST_CLASS._as_fields(blocks, "the blocks")
            arrays = {}
            for channel, block in blocks.items():
                try:
                    arrays[channel] = np.asarray(block, dtype=np.float64)
                except (TypeError, ValueError):
                    arrays[channel] = block  # _post_load names what is wrong with it
            data["_blocks"] = arrays
        return super()._from_state(data)

    def _post_load(self) -> None:
        """
        Checks the loaded blocks and binds the nodes to the saved frame. A
        node's channels are views of one row of the blocks, which no file
        keeps, so they are rebound here.

        Raises
        ------
        ValueError
            The blocks are not one (F, N, 3) array per framed channel, all
            of the same F, N the node count.
        TypeError
            The rest pose is not a :class:`HierarchyData`, or is a clip.
        ValueError
            The rest pose records other axes or units than the clip.
        """
        super()._post_load()
        _check_rest_pose(self._rest_pose, self)

        # a channel is named as a file names it: "translate"
        count  = len(self.list)
        blocks = {}
        for channel, block in (self._blocks or {}).items():
            name = self._channel_name(channel)
            if channel not in self.FRAMED_CHANNELS:
                raise ValueError(f"ClipData has no framed channel {name!r}")

            try:
                block = np.asarray(block, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(
                    f"ClipData block {name} must have shape (F, {count}, 3), "
                    f"got rows of different lengths"
                ) from None
            if block.size == 0 and block.ndim:
                # json keeps no shape for an empty row: (F, 0) is (F, 0, 3)
                block = block.reshape(len(block), 0, 3)
            if block.ndim != 3 or not len(block) or block.shape[1:] != (count, 3):
                raise ValueError(
                    f"ClipData block {name} must have shape (F, {count}, 3) "
                    f"with F at least 1, got {block.shape}"
                )
            blocks[channel] = block

        if blocks and set(blocks) != set(self.FRAMED_CHANNELS):
            missing = sorted(self._channel_name(x) for x in set(self.FRAMED_CHANNELS) - set(blocks))
            raise ValueError(f"ClipData is missing the blocks {missing}")
        if len({block.shape[0] for block in blocks.values()}) > 1:
            raise ValueError("ClipData blocks hold different frame counts")

        self._blocks      = blocks
        self._columns     = [node.uuid for node in self.list]
        self._start_frame = int(self._start_frame)
        self._fps         = float(self._fps)
        self._frame       = int(self._frame) if blocks else 0
        self._bind(self._frame)

    def strip_namespace(self, namespace: Optional[str] = None) -> None:
        """
        ``strip_namespace()`` on every node and the rest pose's, all or nothing.

        When either would give one name to two nodes, ValueError lists them
        and neither is renamed.
        """
        rest = self._rest_pose
        if rest is None:
            return super().strip_namespace(namespace)

        names = [(node, node.name) for node in rest.list]
        rest.strip_namespace(namespace)
        try:
            super().strip_namespace(namespace)
        except BaseException:
            for node, name in names:
                node.name = name
            raise

    def __eq__(self, other) -> bool:
        """the same nodes, timebase, loaded frame, blocks and rest pose"""
        if not isinstance(other, ClipData):
            return False

        # a column belongs to a uuid: line both clips up with their nodes
        self._realign()
        other._realign()
        return super().__eq__(other)

    __hash__ = HierarchyData.__hash__


# ------------------------------- FILE LOADERS ------------------------------- #


def load(filename: str, mode: Optional[str] = None, **kwargs) -> "HierarchyData":
    """
    Whatever hierarchy the file holds, read as ``mode`` says or as the file
    reads (fbx, glb, or cgmath's own npz / json / pickle). An fbx or a glb
    gives the rig, a :class:`HierarchyData`, ``kwargs`` going to
    ``HierarchyData.load_fbx`` / ``load_glb``: ``load("rig.fbx",
    pose="bind")``. A cgmath file takes none; it gives the class it holds,
    a :class:`HierarchyData` or one deriving from it: a saved
    :class:`ClipData` comes back a clip, where ``HierarchyData.load()``
    refuses one. Any other class, a lone node, raises TypeError.
    """
    return _load_derived(HierarchyData, filename, mode, kwargs)


def load_fbx(
    filename:     str,
    scale_factor: Optional[float] = None,
    axis_system:  Optional[str]   = "maya_y_up",
    pose:         str             = "node",
    node_types:   Optional[tuple] = None,
) -> "HierarchyData":
    """The fbx's node hierarchy, names as in the file: ``HierarchyData.load_fbx()``."""
    return HierarchyData.load_fbx(filename, scale_factor, axis_system, pose, node_types)


def load_glb(
    filename:     str,
    scale_factor: float         = 100.0,
    pose_frame:   Optional[int] = None,
) -> "HierarchyData":
    """The glb's joint hierarchy, names as in the file: ``HierarchyData.load_glb()``."""
    return HierarchyData.load_glb(filename, scale_factor, pose_frame)
