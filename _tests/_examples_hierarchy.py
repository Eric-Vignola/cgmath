"""Filled-in examples of the hierarchy classes, for test_roundtrip.

Each builder returns a new object, built the way a user builds one, with
every saved field away from its default where that is valid:

- TransformData: a joint saved on its own, every channel and joint setting
  changed, a parent uuid outside it, and a user attribute of every type.
- HierarchyData: a namespaced rig of joints, a locator, a group, a space
  transform and the three SDF primitives (TransformData subclasses that are
  not render classes), parented by uuid, one joint carrying a user attribute
  of every type and the group a user attribute a built-in hides, its values
  recorded in Unreal's axes and in meters.
- TransformList: a selection of that rig, one node's parent left out of it.
- ClipData: the rig over four frames, every framed channel moving, its own
  timebase, and a node reparented after the blocks were laid out, so the save
  has to line the columns up.

A ClipData is no Data class: it is a thin object over its storage, a private
HierarchyData subclass, and a pick of its nodes one over a selection of the
storage's nodes, of a private class too. clip_data(), clip_view() and
clip_list() give what a caller holds; clip_storage() and clip_view_storage()
what is behind the first two, which is what the tests of every Data class
build.

Within the lists, the hip carries every field; the other nodes leave the
fields a real node of their kind would at their defaults, which checks that
a field left out of the file loads back as its default.

uuids are fixed (uuid5 of the name): a node appended without one gets a
random uuid4.
"""

from __future__ import annotations

import uuid

import numpy as np
from cgmath.hierarchy import ClipData, ClipList, HierarchyData, ShapeData, TransformData, TransformList

_NAMESPACE = uuid.UUID("6f1c2a9e-3b7d-4e55-9a0c-2d8e4f6b1a37")


def _uuid(name: str) -> str:
    """a fixed uuid for ``name``, upper case as generate_uuid() writes them"""
    return str(uuid.uuid5(_NAMESPACE, name)).upper()


def add_every_user_attribute(node: TransformData) -> None:
    """
    A user attribute of every type on ``node``: each one add_user_attribute()
    creates (every number type, bool, enum, string and array data type, with
    an empty string, empty arrays, NaN and the infinities), then the kinds it
    cannot create, written as rig serializes them and set through the node:
    a matrix, a double3 and a mixed compound, and a sparse multi.
    """
    # numbers, the type inferred or named
    node.add_user_attribute("heroHeight", 1.8)  # double
    node.add_user_attribute("maxReach",   float("inf"))
    node.add_user_attribute("restLength", float("nan"))
    node.add_user_attribute("stiffness",  0.75, attribute_type="float")
    node.add_user_attribute("twistLimit", 45.0, attribute_type="doubleAngle")
    node.add_user_attribute("boneLength", 2.54, attribute_type="doubleLinear")
    node.add_user_attribute("segments", 12)     # long
    node.add_user_attribute("priority", -300, attribute_type="short")
    node.add_user_attribute("lod", 3, attribute_type="byte", keyable=False)
    node.add_user_attribute("sideCode", 76, attribute_type="char")
    node.add_user_attribute("mirrored", True)   # bool
    node.add_user_attribute("pinned", False, keyable=False)
    node.add_user_attribute(
        "space", "world", attribute_type="enum", enum_names="local=1:parent=5:world"
    )

    # strings and arrays, never keyable
    # a string holds any text: json must write it whatever the locale
    node.add_user_attribute("label", "hip L / hanche gauche / 左股関節")              # string
    node.add_user_attribute("note", "", attribute_type="string")
    node.add_user_attribute("tags", ["deform", "twist", ""])                      # stringArray
    node.add_user_attribute("noTags", [], attribute_type="stringArray")
    node.add_user_attribute("weights", [0.25, float("nan"), -float("inf"), 1.0])  # doubleArray
    node.add_user_attribute("falloff", [1.0, 0.5, 0.0], attribute_type="floatArray")
    node.add_user_attribute("noFalloff", [], attribute_type="floatArray")
    node.add_user_attribute("vertexIds", [4, 8, 15, 16, 23, 42], attribute_type="Int32Array")
    node.add_user_attribute("noIds", [], attribute_type="Int32Array")
    node.add_user_attribute(
        "aimVectors", [[0, 1, 0], [1.0, 0.0, float("nan")]], attribute_type="vectorArray"
    )
    node.add_user_attribute("noVectors", [], attribute_type="vectorArray")
    node.add_user_attribute(
        "guideCvs", [[0, 0, 0, 1], [1.5, 2.0, -3.0, 1.0]], attribute_type="pointArray"
    )

    # what add_user_attribute() cannot create, as rig serializes it, then set
    # through the node so the value takes its type
    attrs = node.user_defined_attributes
    attrs["offsetMatrix"] = {
        "attributeType": "matrix",
        "value":         [float(x) for x in np.eye(4).ravel()],
        "keyable":       False,
        "channel_box":   False,
    }
    attrs["pivot"]      = {"attributeType": "double3", "numberOfChildren": 3, "keyable": True}
    attrs["pivotX"]     = {"attributeType": "double", "value": 0.0, "parent": "pivot", "keyable": True}
    attrs["pivotY"]     = {"attributeType": "double", "value": 0.0, "parent": "pivot", "keyable": True}
    attrs["pivotZ"]     = {"attributeType": "double", "value": 0.0, "parent": "pivot", "keyable": True}
    attrs["limits"]     = {"attributeType": "compound", "numberOfChildren": 3}
    attrs["limitsOn"]   = {"attributeType": "bool", "value": False, "parent": "limits"}
    attrs["limitsMax"]  = {"attributeType": "doubleAngle", "value": 0.0, "parent": "limits"}
    attrs["limitsName"] = {"dataType": "string", "value": "", "parent": "limits"}
    attrs["driverWeights"] = {
        "attributeType": "double",
        "multi":         True,
        "value":         [],
        "keyable":       True,
        "channel_box":   True,
    }

    translate          = np.eye(4)
    translate[3, :3]   = (0.5, -1.25, 2.0)
    node.offsetMatrix  = translate  # 4 rows, stored as 16 floats
    node.pivot         = [0.5, 0.0, -0.5]
    node.limits        = [True, 90.0, "knee"]
    node.driverWeights = [[0, 1.5], [3, -0.25], [7, 0.0]]


def transform_data() -> TransformData:
    """a joint saved on its own: every channel and joint setting changed, its
    parent a uuid outside it, a user attribute of every type"""
    node = TransformData(
        "VLR:hip_L",
        node_type="joint",
        uuid=_uuid("VLR:hip_L"),
        parent_uuid=_uuid("VLR:pelvis"),
        scale=[1.0, 1.25, 0.8],
        rotate=[12.5, -40.0, 3.75],
        translate=[9.5, -2.25, 0.125],
        rotate_order=4,
        rotate_axis=[0.0, 5.0, -2.5],
        joint_orient=[-90.0, 0.0, 178.5],
        visibility=False,
        segment_scale_compensate=True,
        radius=2.5,
        draw_style=2,
        shear=[0.125, -0.25, 0.5],
    )
    add_every_user_attribute(node)
    return node


def rig() -> HierarchyData:
    """
    A namespaced rig: a joint chain, a locator, a group and a space
    transform, parented by uuid. The hip carries a user attribute of every
    type; the group a user attribute named like a built-in, which the
    built-in hides but a save keeps; the space transform an enum. Its values
    are recorded in Unreal's axes and in meters.
    """
    root = TransformData(
        "VLR:root",
        node_type    = "joint",
        uuid         = _uuid("VLR:root"),
        translate    = [0.0, 95.0, 1.5],
        joint_orient = [0.0, 90.0, 0.0],
        rotate_order = 3,
        radius       = 4.0,
    )
    pelvis = TransformData(
        "VLR:pelvis",
        node_type="joint",
        uuid=_uuid("VLR:pelvis"),
        parent_uuid=root.uuid,
        rotate=[0.0, 0.0, -7.5],
        translate=[2.0, 0.0, 0.0],
        segment_scale_compensate=True,
        draw_style=2,
    )
    hip             = transform_data()
    hip.parent_uuid = pelvis.uuid

    aim = TransformData(
        "VLR:hip_L_aim",
        node_type   = "locator",
        uuid        = _uuid("VLR:hip_L_aim"),
        parent_uuid = hip.uuid,
        translate   = [0.0, 0.0, 25.0],
        visibility  = False,
    )
    group = TransformData(
        "VLR:ctrl_grp",
        uuid        = _uuid("VLR:ctrl_grp"),
        scale       = [2.0, 2.0, 2.0],
        rotate_axis = [0.0, 0.0, 15.0],
        user_defined_attributes={
            # a transform may carry a user "radius": TransformData's built-in
            # hides it, a save keeps it
            "radius": {"attributeType": "double", "value": 9.0, "keyable": True},
        },
    )
    space = TransformData(
        "VLR:world_space",
        node_type    = "transform",
        uuid         = _uuid("VLR:world_space"),
        rotate       = [0.0, 180.0, 0.0],
        rotate_order = 5,
    )

    # a hierarchy holds transforms: a shape, an SDF primitive, is its own
    # record, placed by the transform it names
    space.add_user_attribute(
        "collisionLayer", "limbs", attribute_type="enum", enum_names="body:limbs:props"
    )

    return HierarchyData([root, pelvis, hip, aim, group, space])


def hierarchy_data() -> HierarchyData:
    return rig()


def shape_data() -> ShapeData:
    """the base of every shape, on its own: its uuid and the transform it sits under"""
    return ShapeData(uuid=_uuid("VLR:shape"), parent_uuid=_uuid("VLR:hip_L"))


def transform_list() -> TransformList:
    """a selection of the rig: the hip, its locator and the space transform,
    the hip's parent left out of it"""
    view = TransformList(list(rig().match("VLR:hip_L*", "VLR:world_space")))
    assert type(view) is TransformList and len(view) == 3 and not view.is_view
    return view


def hierarchy_view() -> HierarchyData:
    """the same selection as the rig gives it: a view, of the rig's class"""
    view = rig().match("VLR:hip_L*", "VLR:world_space")
    assert view.is_view and isinstance(view, HierarchyData) and len(view) == 3
    return view


def clip_view() -> ClipData:
    """a selection of a clip's nodes: a view that keeps the clip's frames"""
    clip = clip_data()
    view = clip[: max(1, len(clip) - 1)]
    assert view.is_view and isinstance(view, ClipData) and view.frame_count == clip.frame_count
    return view


def frame_pose() -> HierarchyData:
    """one frame of the clip, live: a view of the clip's pose of that frame"""
    clip = clip_data()
    pose = clip.at(1002)
    assert pose.is_view and pose.owner is clip and isinstance(pose, HierarchyData)
    return pose


def frame_node() -> TransformData:
    """a node of that pose: the clip's node on that frame"""
    return frame_pose()["VLR:root"]


def clip_data() -> ClipData:
    """the rig over four frames from 1001 at 30 fps, every framed channel
    moving"""
    clip   = ClipData(rig(), frames=4, start_frame=1001, fps=30.0, take="wave")
    frames = clip.frames
    ramp   = np.arange(4, dtype=float)[:, None, None]

    frames.rotate       = frames.rotate + ramp * [5.0, -2.5, 1.25]
    frames.translate    = frames.translate + ramp * [0.0, 0.5, -0.25]
    frames.scale        = frames.scale * (1.0 + ramp * [0.0, 0.125, 0.0])
    frames.rotate_axis  = frames.rotate_axis + ramp * [0.5, 0.0, 0.0]
    frames.joint_orient = frames.joint_orient + ramp * [0.0, 0.0, 2.0]
    frames.shear        = frames.shear + ramp * [0.125, 0.0, -0.25]

    # one node's channel reads one value a frame, (F,), and takes as many
    clip["VLR:root"].translate_y = np.array([95.0, 96.5, 98.0, 99.5])

    # reparenting without a pose write is allowed on a clip; it moves the
    # group to the end of the list, so the block columns are out of step with
    # the nodes until the save lines them up
    clip["VLR:ctrl_grp"].set_parent("VLR:root", world_space=False)
    return clip


def clip_list() -> ClipList:
    """two takes of the rig: clip_data()'s wave, and an idle cut of two of
    its frames"""
    wave      = clip_data()
    idle      = wave.frames[1:3].copy()  # a cut of frames is a view: a take owns its nodes
    idle.take = "idle"
    return ClipList([wave, idle])


def clip_storage():
    """what clip_data()'s ClipData keeps its values in: a list of the private
    class behind every ClipData, a HierarchyData subclass, which is what
    saves and loads"""
    return clip_data()._engine


def clip_view_storage():
    """what clip_view()'s ClipData selects with: a view of the storage's
    nodes, of a private class of its own"""
    return clip_view()._sel


# by class. ClipData is no Data class, so no test of every Data class reads
# its key: it is here for the tests that build what a caller holds, a view of
# one, clip_view(), being of the same class
EXAMPLES = {
    TransformData: transform_data,
    TransformList: transform_list,
    HierarchyData: hierarchy_data,
    ClipData:      clip_data,
    ClipList:      clip_list,
    ShapeData:     shape_data,
}

# a selection is of a private class of its own, behind the public one it is
# an instance of: read off a selection
EXAMPLES[type(hierarchy_view())] = hierarchy_view

# and so are the storage of a clip and a selection of its nodes: the Data
# classes behind a ClipData, read off each
EXAMPLES[type(clip_storage())]      = clip_storage
EXAMPLES[type(clip_view_storage())] = clip_view_storage

# a TransformList is a view: it owns no nodes, so it saves as the hierarchy
# its copy() gives (its nodes copied and owned, the hip's parent, outside the
# view, cleared), and loads back as that HierarchyData (TransformList docstring)
SAVES_AS = {
    TransformList: lambda view: view.copy(),
}
SAVES_AS[type(hierarchy_view())] = lambda view: view.copy()

# a selection of a clip's storage saves as the storage its copy() gives
SAVES_AS[type(clip_view_storage())] = lambda view: view.copy()

# the pose of one frame of a clip and a node of it are views too, of the
# frame: each saves as its copy, a HierarchyData and a TransformData
EXAMPLES[type(frame_pose())] = frame_pose
EXAMPLES[type(frame_node())] = frame_node
SAVES_AS[type(frame_pose())] = lambda pose: pose.copy()
SAVES_AS[type(frame_node())] = lambda node: node.copy()
