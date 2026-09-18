from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from cgmath.formats.glb import load_gltf, load_model
from cgmath.geometry import Data, DataList, ImmutableArray as numpy_array
from cgmath.geometry._base import _mask_to_indices
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
    from cgmath.formats.fbx import FbxExporter
except ImportError:
    FBX         = None
    FbxExporter = None


__all__ = [
    "ClipData",
    "HierarchyData",
    "TransformData",
    "TransformList",
    "MAYA_ATTRIBUTE_MAP",
    "SUPPORTED_NODE_TYPES",
    "generate_uuid",
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


def _glb_columns(model, rig, filename: str) -> dict:
    """pairs a glb's node indexes with the rows of a rig read from it

    A file addresses nodes by index, and it can hold two nodes under the same
    name -- a rig exported once per mesh carries the whole skeleton twice --
    so the pairing is positional rather than by name. The rig is built from
    the same parent first walk of the file and only drops nodes, so walking
    that order pairs them, and names are read to confirm the walk stayed in
    step rather than to do the matching.
    """
    column    = {}
    cursor    = 0
    rig_names = list(rig.name)

    for index in model.ordered_node_indexes:
        if cursor < len(rig_names) and model.nodes[index].name == rig_names[cursor]:
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


@dataclass(repr=False, eq=False)
class TransformData(Data):
    # name, node_type, uuid, and parent_node
    name:        str = None
    node_type:   Optional[str] = "transform"
    uuid:        Optional[str] = None
    parent_node: Optional[str] = None

    # general transform attributes
    _scale:        Optional[np.ndarray] = numpy_array([1.0, 1.0, 1.0])
    _rotate:       Optional[np.ndarray] = numpy_array([0.0, 0.0, 0.0])
    _translate:    Optional[np.ndarray] = numpy_array([0.0, 0.0, 0.0])
    _rotate_order: Optional[int] = 0  # xyz, yzx, zxy, xzy, yxz, zyx
    _rotate_axis:  Optional[np.ndarray] = numpy_array([0.0, 0.0, 0.0])

    # attributes specific to joints as defined by Maya
    _joint_orient:             Optional[np.ndarray] = numpy_array([0.0, 0.0, 0.0])
    _visibility:               Optional[bool] = True
    _segment_scale_compensate: Optional[bool] = True  # Maya default for joints :(
    _radius:                   Optional[float] = 1.0
    _draw_style:               Optional[int] = 0

    # user defined attributes
    user_defined_attributes: Optional[dict] = None

    # --- cached attributes --- #
    _hierarchy    = None  # HierarchyData reference pointer
    _world_matrix = None  # cached world matrix

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

        if scale is not None:
            self._scale = np.asarray(scale, dtype=float)
        else:
            self._scale = np.ones(3, dtype=float)

        if rotate is not None:
            self._rotate = np.asarray(rotate, dtype=float)
        else:
            self._rotate = np.zeros(3, dtype=float)

        if translate is not None:
            self._translate = np.asarray(translate, dtype=float)
        else:
            self._translate = np.zeros(3, dtype=float)

        if rotate_order is not None:
            self._rotate_order = int(rotate_order)
        else:
            self._rotate_order = 0  # xyz default

        if rotate_axis is not None:
            self._rotate_axis = np.asarray(rotate_axis, dtype=float)
        else:
            self._rotate_axis = np.zeros(3, dtype=float)

        if joint_orient is not None:
            self._joint_orient = np.asarray(joint_orient, dtype=float)
        else:
            self._joint_orient = np.zeros(3, dtype=float)

        if visibility is not None:
            self._visibility = bool(visibility)
        else:
            self._visibility = True

        if segment_scale_compensate is not None:
            self._segment_scale_compensate = bool(segment_scale_compensate)
        else:
            self._segment_scale_compensate = True

        if radius is not None:
            self._radius = float(radius)
        else:
            self._radius = 1.0

        if draw_style is not None:
            self._draw_style = int(draw_style)
        else:
            self._draw_style = 0

        if user_defined_attributes is not None:
            self.user_defined_attributes = user_defined_attributes
        else:
            self.user_defined_attributes = {}

    def _write_channel(self, name: str, value: np.ndarray) -> None:
        """writes a vector channel into the storage it already holds

        The write is in place, so a channel bound to a view keeps writing
        through it. The class level default is one array shared by every
        instance, so a node still holding it takes its own copy first.
        """
        value   = np.asarray(value, dtype=float)
        current = getattr(self, name)

        if not np.allclose(current, value):
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
    def _reset_branch_world_matrices(self):
        """resets the world matrix of this node and all its decendents"""
        for node in self.get_branch():
            node._world_matrix = None

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

        # preserve the joint orient value if any
        P  = self.get_parent_matrix()
        JO = matrix_multiply(self.joint_orient_matrix, P)
        L  = local_matrix(M, JO)

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

    @property
    def world_matrix(self):
        """compute a world matrix"""
        if self._world_matrix is None:
            P                  = self.get_parent_matrix()
            self._world_matrix = np.dot(self.matrix, P)
        return self._world_matrix.copy()

    @world_matrix.setter
    def world_matrix(self, M: np.ndarray) -> None:
        """set worldspace_matrix to this value, this will reset joint orient"""
        M = np.asarray(M, dtype=float).reshape(4, 4)
        if self._world_matrix is None or not np.allclose(self._world_matrix, M):
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
        """sets the local transform matrix, this will reset joint orient"""
        M = np.asarray(M, dtype=float).reshape(4, 4)

        # trigger the update only if the matrix has changed
        if not np.allclose(self.matrix, M):
            # the local matrix is [S][RO][R][JO][IS][T]; rotate_axis (RO) is
            # reset below, so strip the joint orient AND parent-scale-inverse
            # factors (JO * IS) from the right to recover the pure [S][R] block.
            # ignoring IS here corrupts the decomposition whenever the parent
            # has non-unit scale and this node is segment-scale-compensated.
            # NOTE: a true inverse is required -- IS carries a scale, so JO * IS
            # has non-orthogonal rows and matrix_local's fast affine inverse
            # (which assumes scale*rotation) would invert it incorrectly.
            JO = self.joint_orient_matrix
            IS = self.parent_scale_inverse_matrix
            B  = np.dot(M, np.linalg.inv(np.dot(JO, IS)))

            # compute the scale from the [S][R] block (rotation preserves the
            # row norms, so the row magnitudes are the scale)
            scale = np.einsum("...i,...i", B[:3, :3], B[:3, :3]) ** 0.5

            # compute the expected rotation matrix with no scale
            B_ = euler_to_matrix(matrix_to_euler(B))[0]

            # compute the sign to determine negative scale
            sign = np.einsum("...i,...i", B[:3, :3], B_[:3, :3]) < 0
            scale[sign] *= -1
            self.scale = scale

            # rotate (joint orient and parent scale are already stripped)
            self.rotate = np.degrees(matrix_to_euler(B, self.rotate_order))[0]

            # translate
            self.translate = M[3, :3]

            # reset joint_axis
            # TODO: conserve rotate_axis
            self.rotate_axis = np.zeros(3)

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
        """returns the node's children"""
        children = []
        if self._hierarchy is not None:
            for node in self._hierarchy:
                if node.parent_node == self.uuid:
                    children.append(node)

        return TransformList(children)

    def get_branch(self) -> "TransformList":
        """returns a recursive NodeList from of all decendents"""

        def _recurse(leaf):
            for child in leaf.get_children():
                branch.append(child)
                _recurse(child)

        # recurse through the hierarchy
        branch = [self]
        _recurse(self)

        return TransformList(branch)

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

        # move the node and all its decendents to the end of the list
        affected = [self._hierarchy.list.pop(node.index) for node in self.get_branch()]
        self._hierarchy.extend(affected)

    def add_prefix(self, prefix: str) -> None:
        """adds a prefix to the node's name"""
        self.name = prefix + self.name

    def add_suffix(self, suffix: str) -> None:
        """adds a suffix to the node's name"""
        self.name = self.name + suffix


def _read_fbx(filename: str, scale_factor: float):
    """opens an fbx and walks its joints into TransformData dicts

    Hands back the ``FbxNode`` behind each uuid alongside the dicts, because
    an animated read has to sample those same objects and fbx lets two nodes
    share a name, so pairing them up afterwards by name would be ambiguous.

    The manager comes back too and has to be held for as long as the scene is
    used: it owns everything reachable from it.
    """
    if FBX is None:
        raise ImportError("fbx module not found")

    manager = FBX.FbxManager.Create()
    ios     = FBX.FbxIOSettings.Create(manager, FBX.IOSROOT)
    manager.SetIOSettings(ios)
    importer = FBX.FbxImporter.Create(manager, "")

    filename = os.path.expanduser(filename)
    if not importer.Initialize(filename, -1, manager.GetIOSettings()):
        raise RuntimeError("Unable to open file!")

    scene = FBX.FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()

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

    return manager, scene, hierarchy_data, nodes


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
    :class:`HierarchyData` even when called on a view.
    """

    DATA_LIST_CLASS = TransformData

    # TODO scale_factor is prolly not the best name, think of a better way to handle unit conversions
    @classmethod
    def load_glb(
        cls, filename: str, scale_factor: float = 100.0, pose_frame: None | int = None
    ):
        """primitive glb joint loader that supports style2 and momentum glbs"""

        # make sure pygltflib is available
        if GLTF2 is None:
            raise ImportError("pygltflib not found")

        # convert the glb to dict
        filename = os.path.expanduser(filename)

        # parse the nodes and look for style2 or momentum joints
        data = load_model(filename)
        gltf_data = load_gltf(
            filename
        )  # TODO this is a hack to weed out meshes, need to fix this in load_model

        node_list = []
        for node in data.nodes:
            node_list.append(None)

            if (
                node.name
                and node.skin is None
                and node.name != "RootNode"
                and not _duplicate_leaf(node, data.nodes)
            ):
                new_node = {}

                # momentum skeleton, only keep joints and locators
                if node.extensions and "FB_momentum" in node.extensions:
                    if "limitOrigin" not in node.extensions["FB_momentum"]:
                        if node.extensions["FB_momentum"]["type"] == "skeleton_joint":
                            new_node              = {"name": node.name}
                            new_node["node_type"] = "joint"

                        elif node.extensions["FB_momentum"]["type"] == "locator":
                            new_node              = {"name": node.name}
                            new_node["node_type"] = "locator"

                # style2, everything's a joint
                else:
                    # Make sure this is not a mesh
                    # TODO: this is a hack, need to fix this in load_model
                    found = False
                    for gltf_node in gltf_data.nodes:
                        if gltf_node.mesh is not None and gltf_node.name is not None:
                            if node.name == gltf_node.name.split(":")[0]:
                                found = True
                                break
                    if not found:
                        new_node              = {"name": node.name}
                        new_node["node_type"] = "joint"

                if "name" in new_node:
                    new_node["uuid"] = generate_uuid()
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
        """primitive fbx joint loader"""

        _, _, hierarchy_data, _ = _read_fbx(filename, scale_factor)

        # build the TransformList
        return cls.from_dict(hierarchy_data)

    def save_fbx(self, filename, zero_root=False, as_ascii=False):
        """saves the data to a .fbx file, binary unless as_ascii is set"""

        if FbxExporter is None:
            raise ImportError("fbx module not found")

        # expand ~, consistent with the other IO paths
        filename = os.path.expanduser(filename)

        exporter = FbxExporter()
        exporter.add_skeleton(self)

        exporter.export(
            path      = filename,
            as_ascii  = as_ascii,
            zero_root = zero_root,
        )

    # --------- gltf and fbx methods --------- #
    @classmethod
    def load(cls, filename: str, mode: Optional[str] = None):
        """loads the data from a file"""

        filename = os.path.expanduser(filename)

        # check to see if this is a gltf
        if mode is None:
            with open(filename, "rb") as file:
                if file.read(4) == b"glTF":
                    mode = "gltf"
                else:
                    file.seek(0)
                    if b" FBX " in file.read(18):
                        mode = "fbx"

        # load gltf
        if mode in ("gltf", "glb"):
            return cls.load_glb(filename)

        # load fbx
        elif mode == "fbx":
            return cls.load_fbx(filename)

        # load using parent class
        else:
            return super().load(filename, mode)

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
        roots = []
        for obj in self:
            if obj.get_parent() is None:
                roots.append(obj)

        return TransformList(roots)

    def get_parents(self) -> np.ndarray:
        """returns the node's parent index"""
        return np.array(
            [-1 if x.get_parent() is None else x.get_parent().index for x in self]
        )

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
        for node in self:
            node.add_prefix(prefix)

    def add_suffix(self, suffix: str) -> None:
        """adds a suffix to the node's name"""
        for node in self:
            node.add_suffix(suffix)

    def _parent_first(self) -> List[int]:
        """view indices ordered so an ancestor is always written before a child

        setters that solve against the parent's world matrix -- world_matrix,
        matrix and the match_* methods -- give a different answer depending on
        write order: a child written before its parent solves against the old
        parent and is then dragged when the parent moves. depth is measured in
        the owning hierarchy, so a view that skips levels still orders right.
        """
        depth = np.zeros(len(self), dtype=int)
        for i, node in enumerate(self):
            seen, parent, count = set(), node.get_parent(), 0

            # a parent outside the hierarchy resolves to a uuid string, which
            # ends the walk; the seen set guards a cyclic parent_node
            while isinstance(parent, TransformData) and id(parent) not in seen:
                seen.add(id(parent))
                count += 1
                parent = parent.get_parent()

            depth[i] = count

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
        children, seen = [], set()
        for node in self:
            for child in node.get_children():
                if child.uuid not in seen:
                    seen.add(child.uuid)
                    children.append(child)

        return TransformList(children)

    def get_branch(self) -> "TransformList":
        """returns every node in the view and all of their decendents"""
        branch, seen = [], set()
        for node in self:
            for member in node.get_branch():
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
        others = self._match_operands(other, ndim=2)
        for i in self._parent_first():
            self[i].match_translate(others[i], x=x, y=y, z=z)

    def match_rotate(self, other) -> None:
        """matches the world rotation of a node, a list of nodes or a rotation"""
        others = self._match_operands(other, ndim=2)
        for i in self._parent_first():
            self[i].match_rotate(others[i])

    def match_matrix(self, other) -> None:
        """matches the world matrix of a node or a list of nodes"""
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
        return np.array([node.index for node in self])

    @property
    def name(self) -> List[str]:
        """returns a list of all names"""
        return [x.name for x in self]

    @property
    def unique_name(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.unique_name)
        return np.array(values)

    @property
    def matrix(self) -> np.ndarray:
        values = []
        for node in self:
            values.append(node.matrix)
        return np.array(values)

    @matrix.setter
    def matrix(self, M: np.ndarray) -> None:
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
        values = []
        for node in self:
            values.append(node.quaternion)
        return np.array(values)

    @quaternion.setter
    def quaternion(self, Q: np.ndarray) -> None:
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
        :class:`HierarchyData` when it is called on a view.

        Subclasses already own their nodes and keep their own class.
        """
        if type(self) is TransformList:
            return self._owned_copy()

        return super().copy()

    def _owned_copy(self) -> "HierarchyData":
        """Copy these nodes into an owning hierarchy of their own.

        A copied node keeps its ``parent_node`` uuid. When that parent is not
        part of the copy -- a selection, a slice, a partial rig -- the
        reference dangles: ``get_parent`` hands back the raw uuid string and
        ``world_matrix`` raises on it. Those references are cleared, so the
        node becomes a root of the result instead.
        """
        new = HierarchyData([node.copy() for node in self])
        for node in new:
            if isinstance(node.get_parent(), str):
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
        for k, node in enumerate(nodes_self):
            parent = node.get_parent()
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
        for i, node in enumerate(nodes):
            parent = node.get_parent()

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
        depth = np.zeros(len(parents), dtype=int)
        for i in range(len(parents)):
            d, p, seen = 0, parents[i], set()
            while p > -1 and p not in seen:
                seen.add(p)
                d += 1
                p = parents[p]
            depth[i] = d
        return list(np.argsort(depth, kind="stable"))

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
        # converted into the target frame
        for i in self._parent_first_order(parents):
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
        if isinstance(obj, self.DATA_LIST_CLASS):
            obj = obj.uuid

        elif isinstance(obj, str) and not validate_uuid(obj):
            return super().index(obj, start=start, stop=stop)

        if stop is None:
            stop = len(self)
        for i, shape in enumerate(self.list[start:stop]):
            if shape.uuid == obj:
                return i + start

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
                if new not in self:
                    node.uuid = new
                    break
            self.list.append(node)

        # otherwise make sure TransformData not already in self
        elif node in self:
            raise RuntimeError("TransformData is already in NodeList")

        # append the node to the list
        else:
            self.list.append(node)

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
                if new not in self:
                    node.uuid = new
                    break

            self.list.insert(index, node)

        # otherwise make sure TransformData not already in self
        elif node in self:
            raise RuntimeError("TransformData is already in NodeList")

        # insert the node to the list
        else:
            self.list.insert(index, node)

    def to_dict(self) -> dict:
        """set keys as uuid in the case of duplicate names"""
        shape_dict = {}
        if self.DATA_LIST_CLASS is not None:
            for node in self.list:
                shape_dict[node.uuid] = node.to_dict()

        return shape_dict

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
                    break
        node._hierarchy = self

    def append(self, node: "TransformData") -> None:
        # validate + add first (raises before we claim the node), then
        # take ownership (detaching from any prior hierarchy)
        super().append(node)
        self._claim(node)

    def insert(self, index: int, node: "TransformData") -> None:
        super().insert(index=index, node=node)
        self._claim(node)

    def pop(self, index=-1):
        # retrieves a node and removes pointer
        if index < 0:
            index += len(self.list)
        if index < 0 or index >= len(self.list):
            raise IndexError("pop index out of range")

        # extract the node and remove pointer
        node            = self.list.pop(index)
        node._hierarchy = None

        # any node in the list which had this node as a parent is now at root
        for obj in self.list:
            if obj.parent_node == node.uuid:
                obj.parent_node = None

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


def _evaluate_local(clip: "ClipData") -> np.ndarray:
    """every frame's local matrices as (F, N, 4, 4)

    Mirrors TransformData.matrix factor for factor: S * RO * R * JO * IS * T.
    """
    blocks = clip._blocks
    scale  = blocks["_scale"]
    nodes  = clip.list

    orders = np.array([node.rotate_order for node in nodes], dtype=int)
    compensate = np.array(
        [bool(node.segment_scale_compensate) for node in nodes], dtype=bool
    )
    parents = clip._parent_indices(nodes)

    # the parent's scale is per frame, the flag that applies it is structural
    inverse  = np.ones(scale.shape, dtype=float)
    parented = (parents >= 0) & compensate
    if parented.any():
        inverse[:, parented] = scale[:, parents[parented]]

    S  = _diagonal(scale)
    RO = _euler_matrices(blocks["_rotate_axis"],  None)
    R  = _euler_matrices(blocks["_rotate"],       orders)
    JO = _euler_matrices(blocks["_joint_orient"], None)
    IS = _diagonal(1.0 / inverse)
    T  = _translation(blocks["_translate"])

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
        block = self._clip._blocks[channel]
        view  = block.view() if self._columns is None else block[:, self._columns]
        view.setflags(write=False)
        return view

    def setter(self, value: np.ndarray) -> None:
        clip  = self._clip
        value = np.asarray(value, dtype=float)
        if self._columns is None:
            clip._blocks[channel][...] = value
        else:
            clip._blocks[channel][:, self._columns] = value
        clip._bind(clip._frame)

    return property(getter, setter)


def _frame_component(channel: str, axis: int) -> property:
    """builds the x / y / z accessor for one framed channel"""

    def getter(self) -> np.ndarray:
        return np.ascontiguousarray(getattr(self, channel)[..., axis])

    def setter(self, value) -> None:
        current            = np.array(getattr(self, channel))
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
    """

    def __init__(self, clip: "ClipData", columns=None):
        self._clip    = clip
        self._columns = None if columns is None else np.asarray(columns, dtype=int)

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

        if self._columns is not None:
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
        """int writes one pose, slice writes a run of them"""
        clip = self._clip

        if self._columns is not None:
            raise TypeError("a node scoped axis writes channels, not poses")

        if isinstance(query, slice):
            frames = range(*query.indices(clip.frame_count))
            paste  = isinstance(value, ClipData)

            if paste and value.frame_count != len(frames):
                raise ValueError(
                    f"{len(frames)} frames selected, {value.frame_count} supplied"
                )

            for i, frame in enumerate(frames):
                # value.frames[i] scrubs value and hands back the same object,
                # so each pose is consumed before the next one is asked for
                self._write_pose(frame, value.frames[i] if paste else value)
        else:
            self._write_pose(clip._resolve_frame(query), value)

        clip._bind(clip._frame)

    def _write_pose(self, frame: int, pose) -> None:
        clip = self._clip
        if len(pose) != len(clip):
            raise ValueError(f"pose holds {len(pose)} nodes, clip holds {len(clip)}")

        for channel, block in clip._blocks.items():
            block[frame] = [
                np.asarray(getattr(node, channel), dtype=float) for node in pose
            ]

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
        return self._scoped(_evaluate_local(self._clip))

    def _scoped(self, values: np.ndarray) -> np.ndarray:
        return values if self._columns is None else values[:, self._columns]

    @property
    def world_matrix(self) -> np.ndarray:
        """every frame's world matrices, (F, N, 4, 4)"""
        clip  = self._clip
        world = _evaluate_world(_evaluate_local(clip), clip._parent_indices(clip.list))
        return self._scoped(world)

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
    CLIP_KEY = "__clip__"

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
                for i, node in enumerate(self.list):
                    setattr(node, channel, row[i])
            self._frame = frame

        for node in self.list:
            node._world_matrix = None

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
        """builds a clip from a sequence of HierarchyData poses"""
        poses = list(poses)
        if not poses:
            raise ValueError("from_poses needs at least one pose")

        clip = cls(poses[0], frames=len(poses), start_frame=start_frame, fps=fps)
        for i, pose in enumerate(poses):
            clip.frames[i] = pose

        clip.frame = 0
        return clip

    # --- pose algebra --- #
    def _other_pose(self, other, frame: int):
        """the pose of ``other`` to pair with ``frame`` of this clip"""
        if not isinstance(other, ClipData):
            return other

        if other.frame_count != self.frame_count:
            raise ValueError(
                f"clips hold {self.frame_count} and {other.frame_count} frames"
            )

        return other.frames[frame]

    def _per_frame(self, action) -> "ClipData":
        """runs a pose algebra call on every frame and collects the results"""
        self._realign()
        resume = self._frame

        poses  = []
        for frame in range(self.frame_count):
            self._bind(frame)
            poses.append(action(frame))

        self._bind(resume)
        return ClipData.from_poses(poses, start_frame=self._start_frame, fps=self._fps)

    def get_delta(self, other, by: str = "index", translate: bool = True) -> "ClipData":
        """The delta of every frame against ``other``, as a clip.

        ``other`` is either one pose, compared against every frame, or another
        clip of the same length, compared frame for frame.
        """
        if not self._blocks:
            return HierarchyData.get_delta(self, other, by=by, translate=translate)

        return self._per_frame(
            lambda frame: HierarchyData.get_delta(
                self, self._other_pose(other, frame), by=by, translate=translate
            )
        )

    def add_delta(self, delta, by: str = "index", translate: bool = True) -> "ClipData":
        """Every frame with ``delta`` applied, as a clip.

        ``delta`` is either one delta applied to every frame -- the retarget
        case, where the offset is fixed to the bone -- or a clip of deltas.
        """
        if not self._blocks:
            return HierarchyData.add_delta(self, delta, by=by, translate=translate)

        return self._per_frame(
            lambda frame: HierarchyData.add_delta(
                self, self._other_pose(delta, frame), by=by, translate=translate
            )
        )

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

        # manager goes unread, but it owns the scene and the nodes sampled
        # below, so the name has to stay bound for the rest of the call
        manager, scene, hierarchy_data, nodes = _read_fbx(
            filename, scale_factor
        )

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
        self._realign()
        data = super().to_dict()
        clip = {
            "start_frame": self._start_frame,
            "fps":         self._fps,
            "frame":       self._frame,
        }
        clip.update(self._blocks)
        data[self.CLIP_KEY] = clip
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ClipData":
        data = dict(data)
        clip = data.pop(cls.CLIP_KEY, None)

        obj  = cls()
        for key in data:
            obj.append(cls.DATA_LIST_CLASS.from_dict(data[key]))

        if clip is not None:
            obj._start_frame = int(clip["start_frame"])
            obj._fps         = float(clip["fps"])
            for channel in cls.FRAMED_CHANNELS:
                if channel in clip:
                    obj._blocks[channel] = np.asarray(clip[channel], dtype=float)
            obj._columns = [node.uuid for node in obj.list]
            obj._bind(int(clip.get("frame", 0)))

        return obj

    def __eq__(self, other) -> bool:
        if not isinstance(other, ClipData):
            return False

        self._realign()
        other._realign()

        if self._start_frame != other._start_frame or self._fps != other._fps:
            return False

        if set(self._blocks) != set(other._blocks):
            return False

        for channel, block in self._blocks.items():
            mine, theirs = block, other._blocks[channel]
            if mine.shape != theirs.shape or not np.allclose(mine, theirs):
                return False

        return super().__eq__(other)

    __hash__ = HierarchyData.__hash__