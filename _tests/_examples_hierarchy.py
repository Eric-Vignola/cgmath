"""Filled-in examples of the hierarchy classes, for test_roundtrip.

Each builder returns a new object, built the way a user builds one, with
every saved field away from its default where that is valid:

- TransformData: a joint saved on its own, every channel and joint setting
  changed, a parent uuid outside it, and a user attribute of every type.
- HierarchyData: a namespaced rig of joints, a locator, a group, a space
  transform and the three SDF primitives (TransformData subclasses that are
  not render classes), parented by uuid, one joint carrying a user attribute
  of every type and the group a user attribute a built-in hides.
- TransformList: a selection of that rig, one node's parent left out of it.
- ClipData: the rig over four frames, every framed channel moving, a frame
  other than the first loaded, its own timebase, and a node reparented after
  the blocks were laid out, so the save has to line the columns up.

Within the lists, the hip carries every field; the other nodes leave the
fields a real node of their kind would at their defaults, which checks that
a field left out of the file loads back as its default.

uuids are fixed (uuid5 of the name): a node appended without one gets a
random uuid4.
"""

from __future__ import annotations

import uuid

import numpy as np
from cgmath.geometry.sdf import SDFBox, SDFCylinder, SDFSphere
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList

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
        parent_node=_uuid("VLR:pelvis"),
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
    )
    add_every_user_attribute(node)
    return node


def rig() -> HierarchyData:
    """
    A namespaced rig: a joint chain, a locator, a group, a space transform,
    and an SDF sphere, box and cylinder used as colliders, parented by uuid.
    The hip carries a user attribute of every type; the group a user
    attribute named like a built-in, which the built-in hides but a save
    keeps.
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
        parent_node=root.uuid,
        rotate=[0.0, 0.0, -7.5],
        translate=[2.0, 0.0, 0.0],
        segment_scale_compensate=True,
        draw_style=2,
    )
    hip             = transform_data()
    hip.parent_node = pelvis.uuid

    aim = TransformData(
        "VLR:hip_L_aim",
        node_type   = "locator",
        uuid        = _uuid("VLR:hip_L_aim"),
        parent_node = hip.uuid,
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
        node_type    = "space_transform",
        uuid         = _uuid("VLR:world_space"),
        rotate       = [0.0, 180.0, 0.0],
        rotate_order = 5,
    )

    head            = SDFSphere(radius=11.5, translate=[0.0, 70.0, 0.0], name="VLR:head_collider")
    head.uuid       = _uuid("VLR:head_collider")
    head.visibility = False

    foot = SDFBox(
        half_extents = [4.0, 1.5, 12.0],
        translate    = [10.0, -90.0, 5.0],
        rotate       = [0.0, 15.0, 0.0],
        name         = "VLR:foot_collider",
    )
    foot.uuid         = _uuid("VLR:foot_collider")
    foot.rotate_order = 2

    thigh = SDFCylinder(
        radius = 7.5,
        height = 42.0,
        axis   = 0,
        scale  = [1.0, 1.1, 0.9],
        name   = "VLR:thigh_collider",
    )
    thigh.uuid = _uuid("VLR:thigh_collider")
    thigh.add_user_attribute(
        "collisionLayer", "limbs", attribute_type="enum", enum_names="body:limbs:props"
    )

    h = HierarchyData([root, pelvis, hip, aim, group, space, head, foot, thigh])
    head.set_parent("VLR:pelvis", world_space=False)
    thigh.set_parent("VLR:hip_L", world_space=False)
    return h


def hierarchy_data() -> HierarchyData:
    return rig()


def transform_list() -> TransformList:
    """a selection of the rig: the hip, its locator and its collider, the
    hip's parent left out of it"""
    view = rig().match("VLR:hip_L*", "VLR:thigh_*")
    assert type(view) is TransformList and len(view) == 3
    return view


def clip_data() -> ClipData:
    """the rig over four frames from 1001 at 30 fps, every framed channel
    moving, frame 2 loaded"""
    clip   = ClipData(rig(), frames=4, start_frame=1001, fps=30.0)
    frames = clip.frames
    ramp   = np.arange(4, dtype=float)[:, None, None]

    frames.rotate       = frames.rotate + ramp * [5.0, -2.5, 1.25]
    frames.translate    = frames.translate + ramp * [0.0, 0.5, -0.25]
    frames.scale        = frames.scale * (1.0 + ramp * [0.0, 0.125, 0.0])
    frames.rotate_axis  = frames.rotate_axis + ramp * [0.5, 0.0, 0.0]
    frames.joint_orient = frames.joint_orient + ramp * [0.0, 0.0, 2.0]

    # one node over every frame reads (F, 1), and writes the same shape
    clip.frames["VLR:root"].translate_y = np.array([[95.0], [96.5], [98.0], [99.5]])
    clip.frame                          = 2

    # reparenting without a pose write is allowed on a clip; it moves the
    # group to the end of the list, so the block columns are out of step with
    # the nodes until the save lines them up
    clip["VLR:ctrl_grp"].set_parent("VLR:root", world_space=False)
    return clip


EXAMPLES = {
    TransformData: transform_data,
    TransformList: transform_list,
    HierarchyData: hierarchy_data,
    ClipData:      clip_data,
}

# a TransformList is a view: it owns no nodes, so it saves as the hierarchy
# its copy() gives (its nodes copied and owned, the hip's parent, outside the
# view, cleared), and loads back as that HierarchyData (TransformList docstring)
SAVES_AS = {
    TransformList: lambda view: view.copy(),
}
