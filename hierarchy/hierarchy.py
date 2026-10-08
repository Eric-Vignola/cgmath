from __future__ import annotations

import copy
import itertools
import json
import os
import re
import urllib.parse
import uuid
import warnings
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
from cgmath.formats._fbx_io import open_fbx
from cgmath.formats.glb import load_model
from cgmath.geometry import Data, DataList, ImmutableArray as numpy_array
from cgmath.geometry._base import Array, _mask_to_indices, _reached_twice, _save_shares
from cgmath.transforms import (
    euler_filter,
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
        # so this indexes the key rather than the segment ahead of it
        held = np.clip(
            np.searchsorted(keys, clamped, side="right") - 1, 0, len(keys) - 1
        )
        return values[held]

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


def generate_uuid() -> str:
    """generates a new unique uuid"""
    return str(uuid.uuid4()).upper()


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


def _declares(cls: type, name: str) -> bool:
    """whether ``cls`` or one of its parents annotates ``name`` at class level"""
    return any(name in vars(c).get("__annotations__", {}) for c in cls.__mro__)


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


# which node is whose child follows every node's parent_node, name and
# uuid, and what each hierarchy lists. A hierarchy counts the edits to those
# among its own nodes: the index it keeps (HierarchyData._rig_index) and its
# nodes' cached world matrices hold the count they were built at, so an edit
# retires them in its own rig and nowhere else. Every count is drawn from
# one sequence, so no two rigs, or two edits, ever share one
_STRUCTURAL_FIELDS = frozenset(("parent_node", "name", "uuid"))
_structure_ticks   = itertools.count(1)


def _structure_changed(owner, renamed: bool = False) -> None:
    """an edit may have given some node of ``owner`` another parent or
    other children; a node no hierarchy owns has neither, so None counts
    nothing. ``renamed``: a rename, whose moved children _set_names has
    seen to, retires the index alone"""
    if owner is not None:
        tick                 = next(_structure_ticks)
        owner._index_version = tick
        if not renamed:
            owner._structure_version = tick


def _structure_stamp(node) -> int:
    """the structure count ``node``'s world matrix holds while it is
    current: its hierarchy's, 0 for a node no hierarchy owns"""
    owner = node._hierarchy
    return 0 if owner is None else owner._structure_version


@dataclass(repr=False, eq=False)
class TransformData(Data):
    # name, node_type, uuid, and parent_node
    name:        str = None
    node_type:   Optional[str] = "transform"
    uuid:        Optional[str] = None
    parent_node: Optional[str] = None

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
    _world_version = None  # its rig's structure count it was cached at

    def __init__(
        self,
        name:                     str,
        node_type:                Optional[str]        = "transform",
        uuid:                     Optional[str]        = None,
        parent_node:              Optional[str]        = None,
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
        # set the variables
        self.name = str(name)

        if node_type is not None:
            self.node_type = str(node_type)

        if uuid is not None:
            self.uuid = str(uuid)

        if parent_node is not None:
            self.parent_node = str(parent_node)

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
        children at all: a leaf written after every node added, or every
        parent changed, would otherwise rebuild the index on each write.
        """
        hierarchy = self._hierarchy
        if hierarchy is not None:
            current = getattr(hierarchy, "_rig_index_is_current", None)
            if (current is None or not current()) and not self._scan_children():
                self._transform_changed()
                return

        for node in _walk((self,), _Lookups()):
            node._transform_changed()

    def _reject_animated(self, what: str) -> None:
        """refuses a rig level edit that an animated clip cannot hold

        These rebuild rotate from a matrix, and a clip would have to do it
        one frame at a time. Each frame then picks its euler branch on its
        own, so the poses stay exact while the curves come out with 180 and
        360 degree steps in them. Nothing here can filter that back out, so
        the edit belongs on the rest pose, before the clip is built.

        Duck typed on frame_count rather than on ClipData, which is defined
        further down the module.
        """
        frames = getattr(self._hierarchy, "frame_count", 0)
        if frames > 1:
            raise RuntimeError(
                f"{what} on {self.name!r} rebuilds rotation a frame at a "
                f"time and would break the curves across {frames} frames; "
                f"apply it to the rest pose before building the clip"
            )

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
        """returns a dict mapped against expected DCC node attributes"""
        return {
            mapping[key]: value if not isinstance(value, np.ndarray) else value.tolist()
            for key, value in self.__dict__.items()
            if key in mapping
        }

    def get_parent(self) -> Union["TransformData", None]:
        """returns this TransformData's parent TransformData"""

        if self._hierarchy is None:
            return None  # raise RuntimeError("undefined hierarchy")

        if self.parent_node is None:
            return None

        # return the Node
        if self.parent_node in self._hierarchy:
            parent_index = self._hierarchy.index(self.parent_node)
            return self._hierarchy[parent_index]

        # return the string
        return self.parent_node

    def get_parent_matrix(self) -> np.ndarray:
        """returns the parent's worldspace matrix

        A node whose parent is not in its hierarchy -- a copied branch, a
        selection -- is treated as a root, the same answer a node with no
        hierarchy at all already gets. ``get_parent`` still hands back the
        raw uuid there, so the unresolved reference stays visible.
        """
        parent = self.get_parent()
        if isinstance(parent, TransformData):
            return parent.world_matrix

        return np.eye(4)

    def set_rotate_to_joint_orient(self) -> None:
        """sets rotate and rotate_axis to 0 and applies all rotation to joint_orient"""
        if self.node_type == "joint":
            self._reject_animated("set_rotate_to_joint_orient()")
            # quaternion captures the full RO * R * JO orientation, so rotate
            # AND rotate_axis must both be zeroed or RO would be double-applied
            Q                 = self.quaternion
            self.rotate       = np.zeros(3, dtype=float)
            self.rotate_axis  = np.zeros(3, dtype=float)
            self.joint_orient = np.degrees(quaternion_to_euler(Q, 0)[0])

    def set_joint_orient_to_rotate(self) -> None:
        """sets joint_orient and rotate_axis to 0 and applies all rotation to rotate"""
        if self.node_type == "joint":
            self._reject_animated("set_joint_orient_to_rotate()")
            # quaternion captures the full RO * R * JO orientation, so
            # joint_orient AND rotate_axis must both be zeroed or RO would be
            # double-applied
            Q                 = self.quaternion
            self.joint_orient = np.zeros(3, dtype=float)
            self.rotate_axis  = np.zeros(3, dtype=float)
            self.rotate       = np.degrees(quaternion_to_euler(Q, self.rotate_order)[0])

    def set_node_type(self, node_type: str) -> None:
        """
        Sets the node type.

        A joint that becomes another type sheds its joint-only attributes
        and keeps its pose: joint_orient is folded into rotate (rotate_axis
        is kept), segment_scale_compensate is turned off, radius and
        draw_style go back to 1.0 and 0. Turning segment_scale_compensate
        off moves the node when its parent is scaled, since it no longer
        cancels that scale. Any other change only sets the type.

        Assigning ``node_type`` directly sets the type and nothing else.

        Raises
        ------
        RuntimeError
            A joint with a joint_orient in an animated clip: its rotate
            would have to be rebuilt one frame at a time.
        """
        node_type = str(node_type)
        if self.node_type == "joint" and node_type != "joint":
            if np.any(self.joint_orient != 0.0):
                self._reject_animated("set_node_type()")
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
        """matches the world position of a TransformData object"""
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
        """matches the world rotation of a TransformData object"""

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
        """matches the world matrix of a TransformData object"""
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
        """
        self._reject_animated("swapaxes()")

        P  = _axis_permutation(axis0, axis1, negate)
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
        a parent_node, name or uuid written, a node added, removed, replaced
        or reordered -- can give any node of its rig another parent, and
        finding the nodes it moves would cost a walk of the rig per edit. The
        cache holds its rig's structure version instead, so one built before
        such an edit in its own rig is not trusted. An edit in another rig,
        or on a node no rig owns, leaves it alone.
        """
        # _structure_stamp, written out: this is every world read's first step
        owner = self._hierarchy
        if self._world_version != (0 if owner is None else owner._structure_version):
            return None
        return self._world_matrix

    @property
    def world_matrix(self):
        """compute a world matrix

        Up the parents to the first one whose world matrix is cached, or to
        a root, then back down, caching each node's on the way: a read after
        a structural edit fills its chain once, and a chain deeper than the
        recursion limit reads as any other.
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
        """returns the node's children

        A child is matched the way :meth:`get_parent` resolves it: its
        ``parent_node`` holds this node's uuid, or this node's name when this
        is the first node carrying that name. Matching uuids alone left a rig
        parented by name with no children below any node, so an edit never
        cleared a descendant's cached world matrix.

        The children come from the index the hierarchy keeps until its
        structure changes, not from a scan of every node, so a branch walk,
        and with it every channel write, costs the size of the branch.
        """
        return TransformList(_Lookups().children(self))

    def _scan_children(self) -> list:
        """:meth:`get_children` by a scan of every node of the hierarchy: for
        a hierarchy the index can not stand for, and for a reset whose index
        an edit has just retired"""
        children = []
        if self._hierarchy is not None:
            # read once, and walk the plain list rather than the hierarchy's
            # item access: a reset runs this on a node whose index an edit
            # has just retired
            uuid, name = self.uuid, self.name
            by_name = None  # whether this node is the one its name resolves to
            for node in self._hierarchy.list:
                ref = node.parent_node
                if ref == uuid:
                    children.append(node)

                elif ref == name and ref is not None and not validate_uuid(ref):
                    # get() rather than index(): a node taken out with del or
                    # remove() still points here, and is no longer listed
                    if by_name is None:
                        by_name = self._hierarchy.get(name) is self
                    if by_name:
                        children.append(node)

        return children

    def get_branch(self) -> "TransformList":
        """returns a recursive NodeList from of all decendents

        This node, then each child followed by that child's own branch.
        Walked with a stack rather than by recursion, so a chain deeper than
        Python's recursion limit is walked too, and a node is listed once
        even when a cyclic parent_node leads back to it.
        """
        return TransformList(_walk((self,), _Lookups()))

    def set_parent(self, parent: Union[str, None], world_space: bool = True) -> None:
        """reparents node to a given parent and resets internal SRTs"""

        # raise error if _hierarchy is missing
        if self._hierarchy is None:
            raise RuntimeError("undefined hierarchy")

        # world_space=False writes no pose at all, so it is safe on a clip
        if world_space:
            self._reject_animated("set_parent(world_space=True)")

        # grab the original world_matrix ahead of time
        M = self.world_matrix

        # set new parent_node and retrieve TransformData parent object
        if parent is None:
            self.parent_node = None
        else:
            index            = self._hierarchy.index(parent)
            self.parent_node = self._hierarchy[index].uuid

        # --- recompute matrices if world_space is True --- #
        if world_space:
            self._world_matrix = None
            self.world_matrix  = M
        else:
            # parent_node changed without restoring world space, so any
            # cached world matrices on this node and its branch are now
            # stale and must be invalidated
            self._reset_branch_world_matrices()

        # move the node and all its decendents to the end of the list. a name
        # resolves to the first node carrying it, so a link inside the branch
        # whose name another node shares would follow that node once the
        # branch lands behind it: those links are pinned to the uuid first
        branch = self.get_branch()
        self._pin_shared_names(branch)
        affected = [self._hierarchy.list.pop(node.index) for node in branch]
        self._hierarchy.extend(affected)

    def _pin_shared_names(self, nodes) -> None:
        """points each of ``nodes`` whose ``parent_node`` is a name other
        nodes carry too at its parent's uuid, so that a reorder cannot hand
        it to another node of that name"""
        refs = [n.parent_node for n in nodes]
        if not any(isinstance(ref, str) and not validate_uuid(ref) for ref in refs):
            return

        counts = Counter(n.name for n in self._hierarchy.list)
        for node, ref in zip(nodes, refs):
            if isinstance(ref, str) and counts[ref] > 1 and not validate_uuid(ref):
                parent = node.get_parent()
                if isinstance(parent, TransformData):
                    node.parent_node = parent.uuid

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
        = None`` stores an empty dict."""
        if not name.startswith("_"):
            # a new parent_node or uuid can move a node of the rig to another
            # parent, so the rig's index of its children and its cached world
            # matrices have to go. A rename is _set_names' to count
            if name in _STRUCTURAL_FIELDS and name != "name":
                _structure_changed(self._hierarchy)

            attrs = self.__dict__.get("user_defined_attributes")
            if attrs and name in attrs and not self._is_builtin_attr(name):
                for spec, converted in _user_writes(attrs, name, value, self.name):
                    spec["value"] = converted
                return

        # no attributes is an empty dict, never None: every node holds its own
        if name == "user_defined_attributes" and value is None:
            value = {}

        # a rename inside a hierarchy keeps the children that name this node
        if name == "name" and self._hierarchy is not None and value != self.name:
            self._set_names([(self, value)])
            return

        try:
            super().__setattr__(name, value)
        except AttributeError as error:
            if self._is_builtin_attr(name):
                raise
            raise AttributeError(
                f"{error}; add_user_attribute() creates a user attribute"
            ) from None

    @staticmethod
    def _set_names(pairs) -> None:
        """sets the name of each ``(node, name)`` of ``pairs``, the last one
        given for a node listed twice

        A child whose ``parent_node`` holds a renamed node's old name would be
        cut loose, or would follow another node of that name: it is pointed
        at that node's uuid instead. A child whose name now finds another
        node -- it named a node the hierarchy lacks, or a later node of a
        name a renamed node took -- has the cached world matrices of its
        branch cleared.

        The nodes of one hierarchy are renamed together, against the names
        they all held before: two nodes trading names, or one taking a name
        another gives up, keep their own children. That is one pass over the
        hierarchy whatever the number of nodes renamed, where a node at a
        time would scan it once per node, and the branches to clear are
        walked together.

        A rename retires the hierarchy's index, whose names it changes, but
        not its cached world matrices: the children it would move are the
        ones pinned or cleared here, so every other node keeps its parent.
        """
        set_raw = Data.__setattr__  # past the rename and count in __setattr__
        owners  = {}
        for node, value in pairs:
            owner = node._hierarchy
            if owner is None:
                set_raw(node, "name", value)
            else:
                _, renamed = owners.setdefault(id(owner), (owner, {}))
                renamed[id(node)] = (node, value)

        for hierarchy, renamed in owners.values():
            changes = [(n, value) for n, value in renamed.values() if value != n.name]
            moved   = {id(n) for n, _ in changes}
            names   = {name for n, value in changes for name in (n.name, value)}
            names.discard(None)

            # one pass: the children naming an old or a new name, the only
            # ones that can change parent, and the nodes carrying those names
            linked, carriers = [], []
            for node in hierarchy.list:
                ref = node.parent_node
                if isinstance(ref, str) and ref in names and not validate_uuid(ref):
                    linked.append(node)
                if node.name in names:
                    carriers.append(node)

            # the first node carrying each name, the one get_parent finds;
            # before the renames and after, every such node is a carrier
            def first_named():
                found = {}
                for n in carriers:
                    found.setdefault(n.name, n)
                return found

            before = first_named() if linked else {}
            for n, value in changes:
                set_raw(n, "name", value)
            if changes:
                _structure_changed(hierarchy, renamed=True)

            if not linked:
                continue

            # a link pinned keeps its parent: written past __setattr__, it
            # retires the index alone
            after = first_named()
            stale = []
            for child in linked:
                parent = before.get(child.parent_node)
                if parent is not None and id(parent) in moved:
                    set_raw(child, "parent_node", parent.uuid)
                elif after.get(child.parent_node) is not parent:
                    stale.append(child)
            _structure_changed(hierarchy, renamed=True)

            # once every link is pinned, so that the walk finds the renamed
            # nodes' children too
            if stale:
                _reset_branches(stale)

    def __dir__(self):
        attrs = self.__dict__.get("user_defined_attributes") or {}
        return sorted(set(super().__dir__()) | set(attrs))

    def _check_new_user_attr(self, name: str) -> None:
        """raises ValueError when ``name`` can not be a new user attribute"""
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"{name!r} is not a valid user attribute name")
        if self._is_builtin_attr(name):
            raise ValueError(
                f"{name!r} is a built-in attribute of {type(self).__name__}"
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

        Raises ValueError when the name exists, is built in or is not a valid
        attribute name, and TypeError or ValueError when the value does not
        fit the type.
        """
        self._check_new_user_attr(name)
        spec = _new_user_attr(
            value, attribute_type, keyable, enum_names, f"{self.name}.{name}"
        )
        self.user_defined_attributes[name] = spec


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


def _walk_fbx(scene, scale_factor: float):
    """walks an open fbx scene's joints into TransformData dicts

    Hands back the ``FbxNode`` behind each uuid alongside the dicts, because
    an animated read has to sample those same objects and fbx lets two nodes
    share a name, so pairing them up afterwards by name would be ambiguous.

    The nodes live only as long as the ``open_fbx`` block the scene came from.
    """
    skeleton       = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eSkeleton")
    marker         = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eMarker")
    null           = _fbx_enum(FBX.FbxNodeAttribute, "EType",        "eNull")
    cross          = _fbx_enum(FBX.FbxNull,          "ELook",        "eCross")
    source_pivot   = _fbx_enum(FBX.FbxNode,          "EPivotSet",    "eSourcePivot")
    inherit_rrs    = _fbx_enum(FBX.FbxTransform,     "EInheritType", "eInheritRrs")

    hierarchy_data = {}
    nodes          = {}

    # traverse the scene and build a dict compatible with TransformData
    def traverse(node, parent_uuid=None):
        unique_id = generate_uuid()
        kept      = False

        attr = node.GetNodeAttribute()
        if attr:
            # TODO: for now lets consider meshes as transforms
            # supported node types for now are joints, locators and transforms
            attr_type = attr.GetAttributeType()
            if attr_type in (
                skeleton,
                marker,
                null,
                _fbx_enum(FBX.FbxNodeAttribute, "EType", "eMesh"),
            ):
                # basics
                data         = {}
                data["name"] = node.GetName()
                data["uuid"] = unique_id
                if parent_uuid is not None:
                    data["parent_node"] = parent_uuid

                # rotate_order, which newer bindings hand back as an enum
                rotate_order = node.GetRotationOrder(source_pivot)
                rotate_order = getattr(rotate_order, "value", rotate_order)
                data["rotate_order"] = {0: 0, 2: 1, 4: 2, 1: 3, 3: 4, 5: 5}[
                    rotate_order
                ]

                # get SRT
                data["scale"]  = np.array(tuple(x for x in node.LclScaling.Get()))
                data["rotate"] = np.array(tuple(x for x in node.LclRotation.Get()))
                data["translate"] = (
                    np.array(tuple(x for x in node.LclTranslation.Get())) * scale_factor
                )

                # if joint, get joint_orient
                if attr_type == skeleton:
                    data["node_type"] = "joint"

                    # fbx ignores pre and post rotation entirely unless
                    # rotationActive is on
                    if node.GetRotationActive():
                        data["joint_orient"] = np.array(
                            tuple(x for x in node.GetPreRotation(source_pivot))
                        )[:3]

                        # and it composes post rotation as its inverse
                        post = FBX.FbxAMatrix()
                        post.SetR(node.GetPostRotation(source_pivot))
                        data["rotate_axis"] = np.array(
                            tuple(x for x in post.Inverse().GetR())
                        )[:3]

                    data["segment_scale_compensate"] = (
                        node.GetTransformationInheritType() == inherit_rrs
                    )
                elif attr_type == marker:
                    data["node_type"] = "locator"
                # a null drawn as a cross is a locator, a bare one is a group,
                # which is how Maya's own FBX plugin tells them apart
                elif attr_type == null and attr.Look.Get() == cross:
                    data["node_type"] = "locator"
                else:
                    data["node_type"] = "transform"

                attrs = _fbx_user_attributes(node)
                if attrs:
                    data["user_defined_attributes"] = attrs

                hierarchy_data[unique_id] = data
                nodes[unique_id]          = node
                kept                      = True

        # a node we skipped has no uuid to hand down, so its children take
        # the nearest ancestor we did keep rather than falling to the world
        child_uuid = unique_id if kept else parent_uuid
        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i), parent_uuid=child_uuid)

    traverse(scene.GetRootNode())

    # remove the root node and any invalid parent nodes
    uuids = list(hierarchy_data.keys())
    for node in hierarchy_data.values():
        # use .get(): a root-level node may carry no parent_node key
        if node.get("parent_node") not in uuids:
            node.pop("parent_node", None)

    return hierarchy_data, nodes


def _fbx_has_curves(stack) -> bool:
    """whether an animation stack holds any curves at all

    The criteria have to be built here rather than at import: they match
    nothing unless an FbxManager already exists.
    """
    layers = FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId)
    curves = FBX.FbxCriteria.ObjectType(FBX.FbxAnimCurveNode.ClassId)

    for index in range(stack.GetSrcObjectCount(layers)):
        if stack.GetSrcObject(layers, index).GetSrcObjectCount(curves):
            return True

    return False


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


# a uuid, name or parent_node the rig index can key on
_INDEX_KEYS = (str, type(None))


class _RigIndex:
    """One pass over a hierarchy's nodes, so that get_parent, get_children,
    index and unique_name read dicts rather than scanning the nodes

    ``uuids`` and ``names`` hold the position of the first node carrying
    each uuid and each name, ``refs`` the positions of the nodes holding
    each ``parent_node`` value, in order. Every answer follows the node
    method's rule:

    * a ``parent_node`` shaped like a uuid names a node by uuid only; any
      other string names the first node carrying it as a name; one that
      matches nothing comes back as the string, as get_parent returns it;
    * a node's children hold its uuid, or its name when it is the first
      node carrying that name and the name is not shaped like a uuid, and
      come in hierarchy order;
    * a node's index is the position of the first node with its uuid.

    :meth:`build` gives None for a hierarchy it cannot stand for: a class
    that answers these lookups its own way, or a node whose uuid, name or
    parent_node is not a string. Its callers ask the nodes instead.
    """

    __slots__ = ("nodes", "uuids", "names", "refs", "_shapes", "_holders")

    def __init__(self, nodes: list, uuids: dict, names: dict, refs: dict):
        self.nodes    = nodes
        self.uuids    = uuids
        self.names    = names
        self.refs     = refs
        self._shapes  = {}    # text -> whether it is shaped like a uuid
        self._holders = None  # name -> the uuids carrying it, on first use

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
        uuids, names, refs = {}, {}, {}
        for position, node in enumerate(nodes):
            uuid, name, ref = node.uuid, node.name, node.parent_node
            if not (
                isinstance(uuid, _INDEX_KEYS)
                and isinstance(name, _INDEX_KEYS)
                and isinstance(ref, _INDEX_KEYS)
            ):
                return None
            uuids.setdefault(uuid, position)
            names.setdefault(name, position)
            refs.setdefault(ref, []).append(position)

        return cls(nodes, uuids, names, refs)

    def is_uuid(self, text: str) -> bool:
        """validate_uuid(text), worked out once per text"""
        shape = self._shapes.get(text)
        if shape is None:
            shape = self._shapes[text] = validate_uuid(text)
        return shape

    def parent(self, node):
        """``node.get_parent()`` for a ``parent_node`` that is a string"""
        ref      = node.parent_node
        position = self.uuids.get(ref) if self.is_uuid(ref) else self.names.get(ref)
        return ref if position is None else self.nodes[position]

    def children(self, node) -> list:
        """``node.get_children()``, as a list"""
        uuid, name = node.uuid, node.name
        rows = self.refs.get(uuid, ())

        # a child naming this node counts when this is the first node of its
        # name; a name shaped like a uuid is left to the uuid rule
        if isinstance(name, str) and name != uuid:
            named = self.refs.get(name)
            if named and not self.is_uuid(name):
                first = self.names.get(name)
                if first is not None and self.nodes[first] is node:
                    rows = sorted(list(rows) + named)

        return [self.nodes[position] for position in rows]

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

    A node's get_parent, index and unique_name scan its hierarchy,
    get_parent twice, so asking them of every node of a rig costs the rig's
    size squared. Here each owning hierarchy's index (:class:`_RigIndex`) is
    fetched the first time one of its nodes is asked about, and held for the
    length of the call: a pass that edits as it goes reads the rig as it was
    when it started. A HierarchyData keeps its index between calls, until
    its structure changes (:meth:`HierarchyData._rig_index`). A node whose
    hierarchy can not be indexed is asked itself.
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
        hierarchy, ref = node._hierarchy, node.parent_node
        if hierarchy is None or ref is None:
            return None

        index = self.index_of(hierarchy)
        if index is None or not isinstance(ref, str):
            return node.get_parent()
        return index.parent(node)

    def children(self, node) -> list:
        """``node.get_children()``, as a list"""
        hierarchy = node._hierarchy
        if hierarchy is None:
            return []

        index = self.index_of(hierarchy)
        if (
            index is None
            or not isinstance(node.uuid, _INDEX_KEYS)
            or not isinstance(node.name, _INDEX_KEYS)
        ):
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
    Python's recursion limit is walked too, and a cyclic parent_node ends
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
    parent_node has no world matrix: RecursionError, as the recursion gave.
    """
    lookups, chain, seen = _Lookups(), [], set()
    parent_world = None
    while parent_world is None:
        chain.append(node)
        seen.add(id(node))
        parent = lookups.parent(node)
        if not isinstance(parent, TransformData):
            # a root, or a parent the hierarchy does not hold
            parent_world = np.eye(4)
        elif id(parent) in seen:
            raise RecursionError(f"the parent_node of {parent.name!r} leads back to it")
        else:
            parent_world = parent._cached_world()
            node         = parent

    world = parent_world
    for node in reversed(chain):
        world               = np.dot(node.matrix, world)
        node._world_matrix  = world
        node._world_version = _structure_stamp(node)
    return world


def _chain_depths(parents) -> np.ndarray:
    """the number of ancestors above each entry of a parent array

    ``parents[i]`` is the position of i's parent, negative for none. The
    count is the one a walk up the array makes, every distinct entry met on
    the way up, stopping at a root or at an entry met already. It is found
    once per entry rather than once per entry per level, so a chain N deep
    costs N steps instead of N squared. A cycle counts its own members, as
    the walk does, so a cyclic parent_node still orders the same.
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


class TransformList(DataList):
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
    :class:`HierarchyData` even when called on a view. So does loading a saved
    or pickled view: it saves as its :meth:`copy`.
    """

    DATA_LIST_CLASS = TransformData

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
                # convert children to parent_node
                if "children" in node:
                    for index in node.pop("children", None):
                        if node_list[index] is not None:
                            node_list[index]["parent_node"] = node["uuid"]

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

    # TODO scale_factor is prolly not the best name, think of a better way to handle unit conversions
    @classmethod
    def load_fbx(cls, filename: str, scale_factor: float = 1.0):
        """primitive fbx joint loader

        User defined attributes come back as the specs save_fbx wrote; those
        of a file from Maya's own FBX plugin, the way its importer reads them.
        One whose name, string or enum field is not UTF-8, which the fbx
        binding can't read, is left out, and so is an enum from another tool
        with a field an enumName can't hold (one holding ":", or reading as a
        field and its index, "k=2" or "k==2"), with a warning.
        """
        if FBX is None:
            raise ImportError("fbx module not found")

        with open_fbx(filename, FBX) as scene:
            hierarchy_data, _ = _walk_fbx(scene, scale_factor)

        # build the TransformList
        return cls.from_dict(hierarchy_data)

    def save_fbx(self, filename, zero_root=False, as_ascii=False):
        """saves the data to a .fbx file, binary unless as_ascii is set

        rotate_axis and joint_orient are written for joints only. A
        transform or locator holding a rotate_axis (a matrix write or
        set_parent() keeps it) loses it in the file; fold it into rotate
        first by writing node.matrix back with rotate_axis zeroed.

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
        """sets rotate to 0 and applies all rotation to joint_orient"""
        for node in self[:]:
            node.set_rotate_to_joint_orient()

    def set_joint_orient_to_rotate(self) -> None:
        """sets joint_orient to 0 and applies all rotation to rotate"""
        for node in self[:]:
            node.set_joint_orient_to_rotate()

    def swapaxes(self, axis0: int, axis1: int, negate: bool = False) -> None:
        """swaps two local axes on every node in the view

        each node restores its own children, so the result does not depend on
        the order the view happens to be in -- no parent-first pass needed.
        """
        for node in self[:]:
            node.swapaxes(axis0, axis1, negate=negate)

    def get_roots(self) -> "TransformList":
        """returns all contained nodes parented to the world"""
        lookups = _Lookups()
        return TransformList([obj for obj in self if lookups.parent(obj) is None])

    def get_parents(self) -> np.ndarray:
        """returns each node's parent index in its owning hierarchy, -1 for none

        A parent_node naming a node the hierarchy does not hold, by uuid or
        by name, gives -1 too, as get_parent_matrix reads such a node as a
        root.
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

    def set_parent(self, parent: Union[str, None], world_space: bool = True) -> None:
        """sets all the nodes under a given parent"""

        # first parent everything to world
        for node in self[:]:
            node.set_parent(None, world_space=world_space)

        # parent to new node if parent is not None
        if parent is not None:
            for node in self[:]:
                # ignore self
                if node != parent:
                    node.set_parent(parent, world_space=world_space)

    def add_prefix(self, prefix: str) -> None:
        """adds a prefix to the node's name"""
        self._rename_each(lambda name: _prefixed(name, prefix))

    def add_suffix(self, suffix: str) -> None:
        """adds a suffix to the node's name"""
        self._rename_each(lambda name: name + suffix)

    def _rename_each(self, rename) -> None:
        """renames every node to ``rename(its name)``, all of a hierarchy at
        once (TransformData._set_names); a node listed twice is renamed twice"""
        renamed = {}
        for node in self.list:
            name              = renamed.get(id(node), (node, node.name))[1]
            renamed[id(node)] = (node, rename(name))

        TransformData._set_names(renamed.values())

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
        written. A name no node has raises instead of landing on the list,
        unless the class declares it."""
        own = name.startswith("_") or name == "list" or name in self.__dict__
        if own or hasattr(type(self), name) or _declares(type(self), name):
            super().__setattr__(name, value)
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
    def _owning_parent_first(nodes) -> List[int]:
        """:meth:`_parent_first` for any sequence of nodes, eg: a pairing

        Depth counts every ancestor in the owning hierarchy, the ones left
        out of ``nodes`` too. Each node met on the way up is resolved once,
        through one index of the rig, into one parent array over all of
        them, so the cost is a pass over the rig rather than a walk to the
        root from every node.
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
            # ends the chain; _chain_depths guards a cyclic parent_node
            parent = lookups.parent(met[len(parents)])
            parents.append(slot(parent) if isinstance(parent, TransformData) else -1)

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

        return TransformList(children)

    def get_branch(self) -> "TransformList":
        """returns every node in the view and all of their decendents

        Each node's branch in the order its own get_branch() lists it, parent
        before children. A node already walked is not walked again, which
        also ends a cyclic parent_node.
        """
        branch, seen = [], set()
        for member in _walk(self, _Lookups()):
            if member.uuid not in seen:
                seen.add(member.uuid)
                branch.append(member)

        return TransformList(branch)

    def get_parent_matrix(self) -> np.ndarray:
        """returns every node's parent worldspace matrix"""
        return np.array([node.get_parent_matrix() for node in self])

    def to_attributes(self, mapping: dict = MAYA_ATTRIBUTE_MAP) -> List[dict]:
        """returns every node's attributes as a list of dicts"""
        return [node.to_attributes(mapping) for node in self]

    def match_translate(
        self, other, x: bool = True, y: bool = True, z: bool = True
    ) -> None:
        """matches the world position of a node, a list of nodes or a translation"""
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "translate", other, axes=(x, y, z)):
            return

        others = self._match_operands(other, ndim=2)
        for i in self._parent_first():
            self[i].match_translate(others[i], x=x, y=y, z=z)

    def match_rotate(self, other) -> None:
        """matches the world rotation of a node, a list of nodes or a rotation"""
        # the head of a multi-frame clip writes its frame through the clip solve
        if _clip_setter(self, "rotate", other):
            return

        others = self._match_operands(other, ndim=2)
        for i in self._parent_first():
            self[i].match_rotate(others[i])

    def match_matrix(self, other) -> None:
        """matches the world matrix of a node or a list of nodes"""
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
        see ``TransformData.set_node_type``. Every node is checked before
        any changes, so a refused change leaves them all as they were.
        """
        if _is_sequence(value):
            values = [str(v) for v in value]
            if len(values) != len(self):
                raise ValueError(
                    f"got {len(values)} node types for {len(self)} nodes"
                )
        else:
            values = [str(value)] * len(self)

        for node, node_type in zip(self, values):
            if (
                node.node_type == "joint"
                and node_type != "joint"
                and np.any(node.joint_orient != 0.0)
            ):
                node._reject_animated("node_type")

        for node, node_type in zip(self, values):
            node.set_node_type(node_type)

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
    # whatever the receiver was.
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

        Subclasses already own their nodes and keep their own class.
        """
        if type(self) is TransformList:
            return self._owned_copy()

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

        A copied node keeps its ``parent_node`` uuid. When that parent is not
        part of the copy -- a selection, a slice, a partial rig -- the
        reference dangles: ``get_parent`` hands back the raw uuid string and
        ``world_matrix`` raises on it. Those references are cleared, so the
        node becomes a root of the result instead.

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
        node whose parent is not among them made a root. A shallow copy is
        over the values its node holds: the nodes themselves keep their owner
        and parent, and an object they share is still one object.
        """
        new     = HierarchyData(nodes)
        lookups = _Lookups()
        for node in new.list:
            if isinstance(lookups.parent(node), str):
                node.parent_node = None

        return new

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
        """
        new = self._owned_copy()
        return new.__isub__(other, by=by, translate=translate)

    def __iadd__(self, other, by: str = "index", translate: bool = True):
        """In-place addition: ``self`` (a delta) becomes ``self + other``."""
        if not isinstance(other, TransformList):
            return NotImplemented

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
        """
        new = delta._owned_copy()
        return new.__iadd__(self, by=by, translate=translate)

    # --- overloaded methods --- #
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
            return TransformList([self.list[i] for i in query])

        # slice -> non-owning view; plain int -> the node itself
        result = self.list[query]
        if isinstance(query, slice):
            return TransformList(result)
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

    def match(
        self, *args, exclude: bool = False, exact: bool = True
    ) -> "TransformList":
        """returns a TransformList object with matching names"""
        new = TransformList()

        for shape in self.list:
            if exclude:
                if not shape.match(*args, exact=exact):
                    new.append(shape)
            elif shape.match(*args, exact=exact):
                new.append(shape)

        return new


class HierarchyData(TransformList):
    """Owning container for a hierarchy of ``TransformData`` nodes.

    Unlike its :class:`TransformList` base (a non-owning view), a
    ``HierarchyData`` *owns* the nodes it holds: every node added through
    :meth:`append`/:meth:`insert` has its ``_hierarchy`` back-pointer claimed
    via :meth:`_claim` (detaching it from any prior owner first), and
    :meth:`pop` clears it. A node belongs to exactly one ``HierarchyData`` at a
    time, which is what lets the ``_hierarchy`` back-pointer always resolve to a
    single, correct source of truth. The pose algebra (``a - b`` / ``a + b``,
    :meth:`get_delta` / :meth:`add_delta`) is inherited from
    :class:`TransformList` so it works over selections too.
    """

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
        node._hierarchy = self

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        # a list replaced can hand a shared name to another node, which moves
        # the children naming it: cached world matrices are not trusted past it
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
        a parent_node, name or uuid written on one of its nodes, or any edit
        to its list through its methods. A ``list`` replaced, or grown
        or shrunk in place, is caught as well; an assignment, sort or
        reverse made on the raw ``list`` is not, and nothing in cgmath makes
        one.
        """
        if self._rig_index_is_current():
            return self._rig_index_cache[3]

        nodes                 = self.list
        index                 = _RigIndex.build(self)
        self._rig_index_cache = (self._index_version, nodes, len(nodes), index)
        return index

    def _rig_index_is_current(self) -> bool:
        """whether the index kept is the one the structure would build now"""
        cache = self._rig_index_cache
        return (
            cache is not None
            and cache[0] == self._index_version
            and cache[1] is self.list
            and cache[2] == len(self.list)
        )

    def append(self, node: "TransformData") -> None:
        # validate + add first (raises before we claim the node), then
        # take ownership (detaching from any prior hierarchy)
        super().append(node)
        self._claim(node)
        _structure_changed(self)

    def insert(self, index: int, node: "TransformData") -> None:
        super().insert(index=index, node=node)
        self._claim(node)
        _structure_changed(self)

    def __setitem__(self, index, value) -> None:
        super().__setitem__(index, value)
        _structure_changed(self)

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        _structure_changed(self)

    def sort(self, key=None, reverse=False) -> None:
        """sorts the nodes in place, by name unless ``key`` says otherwise"""
        # the order decides which node a shared name resolves to. a sort
        # whose comparison raises can leave the list part sorted, so the
        # index is retired whatever happens
        try:
            super().sort(key=key, reverse=reverse)
        finally:
            _structure_changed(self)

    def pop(self, index=-1):
        # retrieves a node and removes pointer
        if index < 0:
            index += len(self.list)
        if index < 0 or index >= len(self.list):
            raise IndexError("pop index out of range")

        # its children are found while it is still listed: a child may name
        # it by uuid or by name, and get_children resolves both. the
        # previous pop retired the index, so a loop of pops (clear(), say)
        # would rebuild it every time: a scan finds them as cheaply
        target = self.list[index]
        if self._rig_index_is_current():
            children = _Lookups().children(target)
        else:
            children = target._scan_children()

        # extract the node and remove pointer
        node            = self.list.pop(index)
        node._hierarchy = None
        _structure_changed(self)

        # any node in the list which had this node as a parent is now at
        # root. all of them first, so the resets walk one index
        for obj in children:
            obj.parent_node = None
        _reset_branches(children)

        return node


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
    scale  = blocks["_scale"]
    nodes  = clip.list

    orders = np.array([node.rotate_order for node in nodes], dtype=int)
    compensate = np.array(
        [bool(node.segment_scale_compensate) for node in nodes], dtype=bool
    )
    if parents is None:
        parents = clip._parent_indices(nodes)

    # the parent's scale is per frame, the flag that applies it is structural
    inverse = np.where(compensate[:, None], _parent_scale(clip, parents), 1.0)

    S  = _diagonal(scale)
    RO = _euler_matrices(blocks["_rotate_axis"],  None)
    R  = _euler_matrices(blocks["_rotate"],       orders)
    JO = _euler_matrices(blocks["_joint_orient"], None)
    IS = _diagonal(1.0 / inverse)
    T  = _translation(blocks["_translate"])

    return S, RO, R, JO, IS, T, parents


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

    One node reads back as (F, 1), and a flat list of F values writes it
    the same way.
    """

    def getter(self) -> np.ndarray:
        return np.ascontiguousarray(getattr(self, channel)[..., axis])

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
        nodes pair by position, as they do for one pose.
        """
        clip = self._clip

        if self._uuids is not None:
            raise TypeError("a node scoped axis writes channels, not poses")

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
        (n, 3) or (F, n, 3).
        """
        if _holds_nodes(other):
            target = self._worlds(other)[..., 3, :3]
        else:
            target = self._values(other, (3,))
        self._solve("translate", target, axes=np.array([x, y, z], dtype=bool))

    def match_rotate(self, other) -> None:
        """matches world orientations on every frame at once

        ``other`` is what :meth:`match_translate` takes, or rotations: eulers
        in degrees in each node's rotate order, (3,), (n, 3) or (F, n, 3), or
        quaternions, (4,), (n, 4) or (F, n, 4).
        """
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
        node, a list of nodes or a pose, held on every frame.
        """
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
        frame for frame, a node, a list of nodes or a pose holds on every frame"""
        frames, count = self._clip.frame_count, len(self._positions())

        if isinstance(other, _FrameAxis) or (isinstance(other, ClipData) and other._blocks):
            source = other._clip if isinstance(other, _FrameAxis) else other
            if source.frame_count != frames:
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

    def to_attributes(self, mapping: dict = MAYA_ATTRIBUTE_MAP) -> List[dict]:
        """every node's attributes as a list of dicts, covering every frame

        What TransformData.to_attributes gives on each frame, gathered: a
        framed channel maps to one [x, y, z] per frame, anything else to
        the single value every frame shares.
        """
        # one (n, F, 3) list per framed channel, rather than a tolist per node
        columns = self._columns
        framed = {
            channel: np.swapaxes(self._scoped(block, columns), 0, 1).tolist()
            for channel, block in self._clip._blocks.items()
            if channel in mapping
        }

        out = []
        for column, node in enumerate(self._nodes()):
            attributes = {}
            for key, value in node.__dict__.items():
                if key not in mapping:
                    continue
                if key in framed:
                    attributes[mapping[key]] = framed[key][column]
                elif isinstance(value, np.ndarray):
                    attributes[mapping[key]] = value.tolist()
                else:
                    attributes[mapping[key]] = value
            out.append(attributes)
        return out


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

    Only the channels a pose setting path writes are framed. Everything else
    -- names, uuids, parenting, rotate_order, segment_scale_compensate,
    radius, draw_style -- is structural and shared by every frame.

    The rig level edits refuse to run once there is more than one frame:
    ``swapaxes``, ``set_rotate_to_joint_orient``, ``set_joint_orient_to_rotate``
    and ``set_parent`` in world space all rebuild rotate out of a matrix. A
    clip would have to do that a frame at a time, and each frame would pick
    its euler branch on its own, so every pose stays exact while the curves
    come out with 180 and 360 degree steps in them. Do them on the rest pose
    and build the clip afterwards. ``set_parent(world_space=False)`` writes no
    pose, so it stays allowed.
    """

    FRAMED_CHANNELS = (
        "_scale",
        "_rotate",
        "_translate",
        "_rotate_axis",
        "_joint_orient",
    )

    # saved beside the nodes: the timebase, the loaded frame and one (F, N, 3)
    # block per framed channel, keyed by channel
    LIST_FIELDS = ("_start_frame", "_fps", "_frame", "_blocks")

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

        # a clip owns its nodes: adopting the caller's rig would empty it
        nodes = [] if iterable is None else [node.copy() for node in iterable]
        super().__init__(nodes)

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

    # --- construction --- #
    @classmethod
    def from_poses(cls, poses, start_frame: int = 0, fps: float = 24.0) -> "ClipData":
        """builds a clip from a sequence of HierarchyData poses

        The nodes are copies of the first pose's, and every pose pairs with
        them by position. Each pose is read once, straight into the blocks,
        and the nodes are bound once, to frame 0.
        """
        poses = list(poses)
        if not poses:
            raise ValueError("from_poses needs at least one pose")

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
        copy -- is handed over instead of being copied a second time.
        """
        clip              = cls.__new__(cls)
        clip._blocks      = {}
        clip._columns     = []
        clip._frame       = 0
        clip._start_frame = int(start_frame)
        clip._fps         = float(fps)
        clip.list         = hierarchy.list
        hierarchy.list    = []
        for node in clip.list:
            node._hierarchy = clip

        clip._allocate(int(frames))
        return clip

    # --- pose algebra --- #
    def _operand(self, other, nodes, world: bool = True) -> dict:
        """every frame of ``nodes``, taken from ``other``

        ``other`` is either a clip of the same length, paired frame for
        frame, or one pose for every frame of this clip: a hierarchy, a clip
        with no frames, or a selection or view -- even one of this clip, read
        at the frame it is bound to. ``nodes`` are the nodes to read, in
        order: other's own, or copies of them, so a clip finds their columns
        by uuid. Returns the framed channels as ``(F, n, 3)`` blocks and,
        with ``world``, the ``(F, n, 4, 4)`` world matrices. A pose is read
        once, with a frame axis of 1 rather than F that broadcasts against
        the frames, so whatever is built from it alone is built once.
        """
        frames = self.frame_count

        if isinstance(other, ClipData) and other._blocks:
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
            pose         = np.array([n.world_matrix for n in nodes], dtype=float)
            out["world"] = pose.reshape(1, count, 4, 4)
        return out

    def get_delta(self, other, by: str = "index", translate: bool = True) -> "ClipData":
        """The delta of every frame against ``other``, as a clip.

        ``other`` is either one pose, compared against every frame, or another
        clip of the same length, compared frame for frame. A selection or
        view, even one of this clip, is one pose, read at the frame it is
        bound to; so is a clip with no frames. Every frame is solved at once,
        and ``other`` keeps its current frame.
        """
        if not self._blocks:
            return HierarchyData.get_delta(self, other, by=by, translate=translate)

        return self._get_delta_all_frames(other, by=by, translate=translate)

    def _get_delta_all_frames(self, other, by: str, translate: bool) -> "ClipData":
        """:meth:`HierarchyData.get_delta` for every frame, vectorised.

        ``__isub__`` reads only world matrices, scales and the segment scale
        compensate flags, and writes each node on its own -- no node solves
        against another -- so every frame of every node is one batched solve.
        The steps follow ``__isub__`` and the ``TransformData.quaternion``
        setter it writes through, so the channels match a frame by frame
        solve to rounding. Nodes left out of the pairing keep this clip's own
        channels, as they do per frame.
        """
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
        base  = 1 + (mine["_scale"] - theirs["_scale"])
        scale = base.copy()
        for i in range(count):
            if compensate[i] and parents[i] > -1:
                scale[:, i] = base[:, i] / base[:, parents[i]]

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

        # a copy keeps a parent the clip lacks; the per frame route started
        # from an owned copy, which makes such a node a root
        clip    = self.copy()
        lookups = _Lookups()
        for node in clip.list:
            if isinstance(lookups.parent(node), str):
                node.parent_node = None

        place = {uuid: j for j, uuid in enumerate(clip._columns)}
        cols  = [place[node.uuid] for node in nodes_self]

        # the paired columns take the solved scale and rotate
        clip._blocks["_scale"][:, cols]  = scale
        clip._blocks["_rotate"][:, cols] = rotate

        # translate: a world-space offset; with `translate` off this clip's
        # own bone offsets stay, for add_delta to reuse
        if translate:
            clip._blocks["_translate"][:, cols] = (
                mine["world"][..., 3, :3] - theirs["world"][..., 3, :3]
            )

        clip._bind(0)
        return clip

    def add_delta(self, delta, by: str = "index", translate: bool = True) -> "ClipData":
        """Every frame with ``delta`` applied, as a clip.

        ``delta`` is either one delta applied to every frame -- the retarget
        case, where the offset is fixed to the bone -- or a clip of deltas of
        the same length, paired frame for frame. A selection or view, even
        one of this clip, is one delta, read at the frame it is bound to; so
        is a clip with no frames. Every frame is solved at once, and a clip
        of deltas keeps its current frame.
        """
        if not self._blocks:
            return HierarchyData.add_delta(self, delta, by=by, translate=translate)

        return self._add_delta_all_frames(delta, by=by, translate=translate)

    def _add_delta_all_frames(self, delta, by: str, translate: bool) -> "ClipData":
        """:meth:`HierarchyData.add_delta` against every frame, vectorised.

        Frame by frame, this copies the whole rig and solves it node by node
        once per frame. Here the same algebra -- step for step, so the
        channels come out the same to rounding -- runs once per level of the
        delta's hierarchy, batched over the frame axis and the level's nodes:
        the source's world matrices come from one evaluation of the clip, and
        a level solves against the final world matrices of the level above.
        A clip of deltas brings one delta per frame. A single delta is read
        once, so what is built from it alone -- its orientation offsets,
        rotate axes and joint orients -- is built once and broadcast over the
        frames. Nodes of ``delta`` the clip does not pair keep the delta's
        own channels, as they do per frame.
        """
        self._realign()
        frames = self.frame_count

        # the result is built from the delta, as HierarchyData.add_delta is
        new = delta._owned_copy()
        nodes_new, nodes_clip, pairing_parents = new._matched_subset(self, by=by)

        nodes   = list(new.list)
        count   = len(nodes)
        parents = self._parent_indices(nodes)
        row     = {id(node): i for i, node in enumerate(nodes)}
        column  = {id(node): j for j, node in enumerate(self.list)}

        # paired[i]: position of node i within the pairing, -1 when unpaired
        paired = np.full(count, -1, dtype=int)
        for k, node in enumerate(nodes_new):
            paired[row[id(node)]] = k
        source = np.array([column[id(node)] for node in nodes_clip], dtype=int)
        rows   = [row[id(n)] for n in nodes_new]

        # the delta's channels: one row per frame for a clip of deltas, one
        # row for a single delta, which broadcasts against the frames below
        d          = self._operand(delta, nodes, world=False)
        orders     = np.array([n.rotate_order for n in nodes], dtype=int)
        compensate = np.array([bool(n.segment_scale_compensate) for n in nodes])
        RO         = _euler_matrices(d["_rotate_axis"], None)  # (F or 1, N, 4, 4)
        RO_inverse = np.linalg.inv(RO)
        JO         = _euler_matrices(d["_joint_orient"], None)
        R          = _euler_matrices(d["_rotate"][:, rows], orders[rows])

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

        w_source = self.frames.world_matrix  # (F, Nc, 4, 4)
        s_source = self._blocks["_scale"]    # (F, Nc, 3)

        # output blocks, seeded with the delta's channels for the unpaired nodes
        shape  = (frames, count, 3)
        scale  = np.array(np.broadcast_to(d["_scale"], shape))
        rotate = np.array(np.broadcast_to(d["_rotate"], shape))
        offset = np.array(np.broadcast_to(d["_translate"], shape))
        world  = np.empty((frames, count, 4, 4), dtype=float)

        # one level of the delta's hierarchy at a time, every node of it
        # together, against the final world matrices of the level above.
        # `parents` covers all of `new`, so its depths are the owning
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
                # frame by F, at the source position plus the stored offset
                w_other = w_source[:, source[k]]
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
                scale[:, i]  = s_source[:, source[k]] - 1 + base_delta[:, k]
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

    # --- serialization --- #
    def _subclip(self, span: slice) -> "ClipData":
        """a new clip holding only the frames in ``span``

        The cut always copies. A plain stride would give a numpy view but a
        negative one would not, and a result that sometimes writes back
        through to the parent is worse than one that never does.
        """
        self._realign()
        first = span.indices(self.frame_count)[0]

        new = self.__class__(
            self.list, start_frame=self._start_frame + first, fps=self._fps
        )
        new._blocks  = {c: np.array(b[span]) for c, b in self._blocks.items()}
        new._columns = list(self._columns)
        new._bind(0)
        return new

    def copy(self) -> "ClipData":
        self._realign()
        new         = self.__class__(self.list, start_frame=self._start_frame, fps=self._fps)
        new._blocks = {c: b.copy() for c, b in self._blocks.items()}
        # the blocks are installed behind __init__, so the column list has to
        # come with them: a copy left claiming no columns would have its next
        # realign reseed every frame from the one pose the nodes are bound to
        new._columns = list(self._columns)
        new._bind(self._frame)
        return new

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

        first = min(keys[0] for keys in spans)
        last  = max(keys[-1] for keys in spans)

        # ceil, so the tail of the animation is inside the grid rather than
        # half a frame past its end
        frames = int(np.ceil((last - first) * fps)) + 1
        times  = first + np.arange(frames) / fps

        clip   = cls(rig, frames=frames, start_frame=start_frame, fps=fps)

        # seeded with the rest pose, which is what unanimated nodes keep
        scale     = np.array(clip.frames.scale)
        rotate    = np.array(clip.frames.rotate)
        translate = np.array(clip.frames.translate)

        for channel, index in channels:
            sampler = animations[animation].samplers[channel.sampler]

            if channel.path == "translation":
                translate[:, index] = _sample_gltf(sampler, times, 3) * scale_factor
            elif channel.path == "scale":
                scale[:, index] = _sample_gltf(sampler, times, 3)
            else:
                # the rest branch puts the node's orientation in joint_orient,
                # so rotate carries the delta from it and nothing else
                node = clip[index]
                delta = q_multiply(
                    q_conjugate(node.quaternion),
                    q_normalize(_sample_gltf(sampler, times, 4)),
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
        scale_factor: float           = 1.0,
        take:         Optional[int]   = None,
        fps:          Optional[float] = None,
        start_frame:  Optional[int]   = None,
        end_frame:    Optional[int]   = None,
    ) -> "ClipData":
        """builds a clip from an fbx's animation take

        The three local channels are sampled separately rather than
        decomposed out of ``EvaluateLocalTransform``: that matrix has already
        folded in joint orient, rotate axis and the inherit type, so pulling
        euler angles back out of it would not give the curves the file holds.

        Nothing is euler filtered here, unlike the glb path. Fbx stores euler
        curves, so the branches are the ones the file was authored with and
        rewriting them would change the animation.

        ``take`` defaults to the first one that was actually keyed rather than
        to index 0, because a file can carry takes holding no curves at all
        and reading one back gives the rest pose on every frame -- a silent
        answer that looks like a broken loader rather than an empty take. Pass
        an index to override, including onto an empty take.

        ``start_frame`` and ``end_frame`` are frame numbers on the file's own
        timeline and default to the take's range. A file with no animation
        gives a one frame clip rather than an error.
        """
        if fps is not None and fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")

        if FBX is None:
            raise ImportError("fbx module not found")

        # everything read from the scene dies with it: copy out inside the block
        with open_fbx(filename, FBX) as scene:
            hierarchy_data, nodes = _walk_fbx(scene, scale_factor)

            rig = HierarchyData.from_dict(hierarchy_data)

            criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimStack.ClassId)
            count    = scene.GetSrcObjectCount(criteria)

            if not count:
                return cls(rig, frames=1, start_frame=start_frame or 0, fps=fps or 24.0)

            stacks = [scene.GetSrcObject(criteria, index) for index in range(count)]

            if take is None:
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

            span  = stack.GetLocalTimeSpan()
            first = span.GetStart().GetSecondDouble()
            last  = span.GetStop().GetSecondDouble()

            # ceil, so the tail of the take is inside the grid rather than half a
            # frame past its end
            start = round(first * fps) if start_frame is None else int(start_frame)
            stop = (
                start + int(np.ceil((last - first) * fps))
                if end_frame is None
                else int(end_frame)
            )

            if stop < start:
                raise ValueError(f"end_frame {stop} is before start_frame {start}")

            clip      = cls(rig, frames=stop - start + 1, start_frame=start, fps=fps)
            column    = {uuid: index for index, uuid in enumerate(clip.uuid)}
            sampled   = [(nodes[uuid], column[uuid]) for uuid in nodes if uuid in column]

            scale     = np.array(clip.frames.scale)
            rotate    = np.array(clip.frames.rotate)
            translate = np.array(clip.frames.translate)

            for frame in range(clip.frame_count):
                time = FBX.FbxTime()
                time.SetSecondDouble((start + frame) / fps)

                for node, index in sampled:
                    values = node.EvaluateLocalTranslation(time)
                    translate[frame, index] = (
                        values[0] * scale_factor,
                        values[1] * scale_factor,
                        values[2] * scale_factor,
                    )

                    values               = node.EvaluateLocalRotation(time)
                    rotate[frame, index] = (values[0], values[1], values[2])

                    values               = node.EvaluateLocalScaling(time)
                    scale[frame, index]  = (values[0], values[1], values[2])

        clip.frames.scale     = scale
        clip.frames.rotate    = rotate
        clip.frames.translate = translate
        clip.frame            = 0
        return clip

    def save_fbx(self, filename, zero_root=False, as_ascii=False):
        """saves the bound frame to a .fbx file, binary unless as_ascii is set

        The exporter writes one pose per node and never opens an animation
        stack, so a clip holding more than one frame has to say which frame
        it meant rather than quietly shipping whichever one is loaded.
        """
        if self.frame_count > 1:
            raise ValueError(
                "save_fbx writes one pose but this clip holds "
                f"{self.frame_count} frames; cut the frame out first, "
                f"clip.frames[{self.frame}:{self.frame + 1}].save_fbx(...), "
                "or use save() to keep the whole clip"
            )

        super().save_fbx(filename, zero_root=zero_root, as_ascii=as_ascii)

    def to_dict(self) -> dict:
        # the saved block columns follow the saved node order
        self._realign()
        return super().to_dict()

    @classmethod
    def _from_state(cls, data: dict) -> "ClipData":
        # the blocks hold only numbers: one array each now, rather than a
        # walk over every json row looking for objects to rebuild
        blocks = data.get("_blocks")
        if isinstance(blocks, dict):
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
        """
        super()._post_load()

        count  = len(self.list)
        blocks = {}
        for channel, block in (self._blocks or {}).items():
            if channel not in self.FRAMED_CHANNELS:
                raise ValueError(f"ClipData has no framed channel {channel!r}")

            try:
                block = np.asarray(block, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(
                    f"ClipData block {channel} must have shape (F, {count}, 3), "
                    f"got rows of different lengths"
                ) from None
            if block.size == 0 and block.ndim:
                # json keeps no shape for an empty row: (F, 0) is (F, 0, 3)
                block = block.reshape(len(block), 0, 3)
            if block.ndim != 3 or not len(block) or block.shape[1:] != (count, 3):
                raise ValueError(
                    f"ClipData block {channel} must have shape (F, {count}, 3) "
                    f"with F at least 1, got {block.shape}"
                )
            blocks[channel] = block

        if blocks and set(blocks) != set(self.FRAMED_CHANNELS):
            missing = sorted(set(self.FRAMED_CHANNELS) - set(blocks))
            raise ValueError(f"ClipData is missing the blocks {missing}")
        if len({block.shape[0] for block in blocks.values()}) > 1:
            raise ValueError("ClipData blocks hold different frame counts")

        self._blocks      = blocks
        self._columns     = [node.uuid for node in self.list]
        self._start_frame = int(self._start_frame)
        self._fps         = float(self._fps)
        self._frame       = int(self._frame) if blocks else 0
        self._bind(self._frame)

    def __eq__(self, other) -> bool:
        """the same nodes, timebase, loaded frame and blocks"""
        if not isinstance(other, ClipData):
            return False

        # a column belongs to a uuid: line both clips up with their nodes
        self._realign()
        other._realign()
        return super().__eq__(other)

    __hash__ = HierarchyData.__hash__


# ------------------------------- FILE LOADERS ------------------------------- #


def load(filename: str, mode: Optional[str] = None) -> "HierarchyData":
    """
    The file's hierarchy: ``HierarchyData.load()``, which picks the reader
    from ``mode`` or from the file (fbx, glb, or cgmath's own npz / json /
    pickle). A cgmath file loads as the class it names, a :class:`ClipData`
    as a clip; an fbx or a glb gives the rig, a :class:`HierarchyData`.
    """
    return HierarchyData.load(filename, mode)


def load_fbx(filename: str, scale_factor: float = 1.0) -> "HierarchyData":
    """The fbx's node hierarchy, names as in the file: ``HierarchyData.load_fbx()``."""
    return HierarchyData.load_fbx(filename, scale_factor)


def load_glb(
    filename:     str,
    scale_factor: float         = 100.0,
    pose_frame:   Optional[int] = None,
) -> "HierarchyData":
    """The glb's joint hierarchy, names as in the file: ``HierarchyData.load_glb()``."""
    return HierarchyData.load_glb(filename, scale_factor, pose_frame)
