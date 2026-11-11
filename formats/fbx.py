import json
import logging
import math
import pathlib
import re
import urllib.parse
import warnings

import numpy as np
from cgmath.formats._fbx_io import load_sdk, new_manager, typed

LOGGER = logging.getLogger(__name__)

# The Autodesk FBX SDK ships as a separate download and is NOT on PyPI, so it
# cannot be a wheel dependency: naming it would make `pip install cgmath` fail
# on every machine that does not already have it, taking OBJ, GLB and USD down
# with it. Importing this module therefore has to survive its absence -- warn,
# and let the failure land only if FBX is actually used.
#
# The stand-in resolves attribute CHAINS and raises only when something is
# CALLED. That split is what the module needs: class bodies below name FBX
# constants at import time (`FbxExporter` maps rotation orders to
# `fbx.EFbxRotationOrder.*`), so a stand-in that raised on attribute access
# would fail at import -- the very thing this is here to prevent. Every real
# FBX operation goes through a call, so those still fail, with one actionable
# message instead of `NameError: fbx` or `AttributeError: 'NoneType' object
# has no attribute 'FbxManager'`.
FBX_SDK_REQUIRED = (
    "The Autodesk FBX SDK (the `fbx` module) is not installed, so cgmath's FBX "
    "support is unavailable. Autodesk distributes it separately from PyPI: "
    "install the FBX Python SDK matching your Python version from "
    "https://aps.autodesk.com/developer/overview/fbx-sdk . Every other format "
    "cgmath reads and writes -- OBJ, GLB, USD -- works without it."
)

# a module that imports as fbx but is not the SDK is taken for none (load_sdk)
fbx = load_sdk()
if fbx is None:

    class _FbxSdkMissing:
        """Stands in for the absent ``fbx`` module and explains itself."""

        def __getattr__(self, name):
            return self  # so `fbx.EFbxRotationOrder.eEulerXYZ` still imports

        def __call__(self, *args, **kwargs):
            raise RuntimeError(FBX_SDK_REQUIRED)

        def __bool__(self):
            return False

        def __repr__(self):
            return "<Autodesk FBX SDK not installed>"

    fbx = _FbxSdkMissing()
    # `warnings` rather than LOGGER: it is the channel a library is supposed to
    # use for a degraded-capability notice, it prints once per location by
    # default, and it stays silenceable per-caller. Emitting on both channels
    # only doubles the noise on a plain import.
    warnings.warn(FBX_SDK_REQUIRED, UserWarning, stacklevel=2)


def _ascii_json(value) -> str:
    """
    ``value`` as json any .fbx keeps as it is, however long: every character
    escaped to ASCII, then all but _PLAIN_TEXT %-escaped. It is read back by
    unquoting it, then json.
    """
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=_json_plain)
    return urllib.parse.quote(text, safe=_PLAIN_TEXT)


def _json_plain(value):
    """a numpy number or array in a spec, as the python value json writes"""
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise TypeError(f"a {type(value).__name__} does not go to json")


# what a long string keeps as it is in any .fbx, the rest is %-escaped:
# printable ASCII but the spaces, quote, semicolon and braces the FBX SDK's
# ASCII reader can break (see _ASCII_TEXT_LIMIT), the & it unescapes, the
# backslash Maya's FBX importer drops, and the %
_PLAIN_TEXT = "".join(chr(x) for x in range(33, 127) if chr(x) not in '"\\&%{};')


def _string_array_text(items) -> str:
    """a stringArray as the string FbxExporter writes: the mark, then the list
    as _ascii_json"""
    return FbxExporter._STRING_ARRAY_MARK + _ascii_json([str(x) for x in items])


# an ASCII .fbx writes a quote and the line breaks as these, and its reader
# turns them back, typed ones too: a typed &quot; comes back as a quote
_ASCII_ESCAPES = {"quot": '"', "lf": "\n", "cr": "\r"}

# the longest string, in bytes as an ASCII .fbx writes it, its reader gives
# back as it is whatever it holds. past about 6,100 it reads a string in
# pieces, and a quote, line break, tab, semicolon or &-entity where two meet
# comes back changed, as does a run of spaces across it; a brace there loses
# the whole scene
_ASCII_TEXT_LIMIT = 4096


def _ascii_text(text: str) -> str:
    """
    ``text`` as FbxExporter writes it to an ASCII .fbx: as it is, but for a
    string over _ASCII_TEXT_LIMIT bytes holding a character the reader can
    break, whose every character but _PLAIN_TEXT is then %-escaped. What it
    can't break: _PLAIN_TEXT, the % and backslash, and single spaces.
    """
    # a quote is written as &quot;, a line break as &lf; or &cr;
    size = len(text.encode("utf-8")) + 5 * text.count('"')
    size += 3 * (text.count("\n") + text.count("\r"))
    if size <= _ASCII_TEXT_LIMIT:
        return text
    if "  " not in text and set(text) <= set(_PLAIN_TEXT + "%\\ "):
        return text
    return urllib.parse.quote(text, safe=_PLAIN_TEXT)


def _set_enum_fields(prop, fields) -> None:
    """an enum property's field list replaced by ``fields``, every field
    removed then added again: the SDK's SetEnumValue leaves a field as it was
    when its new text is in the list already, and AddEnumValue skips a field
    listed already, which Maya allows"""
    for i in reversed(range(prop.GetEnumCount())):
        prop.RemoveEnumValue(i)
    for field in fields:
        prop.AddEnumValue(field)


def _ascii_read_back(value, kind: str):
    """
    ``value`` as an ASCII .fbx gives it back, written by FbxExporter to a
    ``kind`` user defined property: a string as _ascii_text writes it, with a
    typed &quot;, &lf; or &cr; read as the quote or line break; a double to 15
    significant digits; the smallest short or long as the largest. A binary
    .fbx gives back every value as it is, but for -inf and a double under
    1e-307, which it reads as 0 too.
    """
    # imported here, cgmath.hierarchy imports this module
    from cgmath.hierarchy.hierarchy import _FLOAT_TYPES

    if isinstance(value, str):
        text = _ascii_text(value)
        return re.sub("&(quot|lf|cr);", lambda found: _ASCII_ESCAPES[found[1]], text)

    if kind in _FLOAT_TYPES and isinstance(value, (int, float, np.number)):
        text = f"{float(value):.15g}"
        back = float(text)

        # the SDK reads -inf, what the digits overflow below it and a power of
        # ten under -307 as 0
        if back == -math.inf or int(text.partition("e")[2] or 0) < -307:
            return 0.0
        return back

    # and the smallest short or long as the largest
    if kind == "short" and value == -(2**15):
        return 2**15 - 1
    if kind in ("long", "int", "enum") and value == -(2**31):
        return 2**31 - 1
    return value


def _user_property_value(spec: dict, value):
    """
    what FbxExporter writes to the property of the user defined attr ``spec``
    for ``value``: an empty string for a string with none, as Maya leaves one
    it has not set; a stringArray as _string_array_text; an enum value as its
    position in the field list, Maya's indices can be explicit ("a=1:b=5:c");
    the rest as it is.
    """
    # imported here, cgmath.hierarchy imports this module
    from cgmath.hierarchy.hierarchy import _enum_fields, _user_attr_type

    kind = _user_attr_type(spec)
    if value is None:
        return ""
    if kind == "stringArray":
        return _string_array_text(value)
    if kind == "enum":
        enum_name = spec.get("enumName", "")
        fields    = _enum_fields(enum_name if isinstance(enum_name, str) else "")
        return {index: i for i, (_, index) in enumerate(fields)}.get(value, value)
    return value


def _kept_spec(spec: dict, keep: bool) -> dict:
    """``spec`` as _USER_ATTR_SPECS keeps it: its value None, left to the
    property, or as ``[value]`` when ``keep``"""
    return {
        key: ([entry] if keep else None) if key == "value" else entry
        for key, entry in spec.items()
    }


# the channels whose zeros _negative_zeros notes, as the file holds them
NEGATIVE_ZERO_CHANNELS = ("translate", "rotate", "joint_orient")


def _negative_zeros(item) -> str:
    """``"name x y z;..."`` of each channel of ``item`` holding zeros only,
    -0.0 among them, 1 for each -0.0; "" where none does"""
    held = []
    for name in NEGATIVE_ZERO_CHANNELS:
        value = np.asarray(getattr(item, name), dtype=np.float64)
        if np.all(value == 0.0) and np.signbit(value).any():
            held.append(" ".join([name] + [str(int(x)) for x in np.signbit(value)]))
    return ";".join(held)


def _fbx_times(seconds) -> list:
    """an ``fbx.FbxTime`` per time in seconds"""
    made = []
    for value in np.asarray(seconds, dtype=np.float64).ravel().tolist():
        fbx_time = fbx.FbxTime()
        fbx_time.SetSecondDouble(value)
        made.append(fbx_time)
    return made


def _set_keys(curve, fbx_times, values, mode) -> None:
    """every key of ``curve`` replaced by one per ``fbx.FbxTime`` of
    ``fbx_times``, ``values`` (an array) as they are, ``mode`` an
    EInterpolationType: the times made once serve every curve of a take"""
    add = curve.KeyAdd
    # one key, its interpolation set once, added at each time with its
    # value: the binding converts an enum argument through python's enum
    # machinery, which a KeySet a key would pay each time (half the write)
    key = fbx.FbxAnimCurveKey()
    key.SetInterpolation(mode)
    value_of = key.SetValue
    curve.KeyModifyBegin()
    try:
        curve.KeyClear()
        for fbx_time, value in zip(fbx_times, np.asarray(values).tolist()):
            value_of(value)
            add(fbx_time, key)
    finally:
        curve.KeyModifyEnd()


def _key_visibility(node, layer, times, shown) -> None:
    """``node``'s Visibility keyed in ``layer``, at each ``fbx.FbxTime`` of
    ``times``, 1.0 where ``shown``, else 0.0, constant interpolation,
    constant standard: each key holds to the next, as Maya keys it"""
    node.Visibility.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
    curve    = typed(node.Visibility.GetCurve(layer, True), fbx.FbxAnimCurve, fbx)
    constant = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant
    standard = fbx.FbxAnimCurveDef.EConstantMode.eConstantStandard
    curve.KeyModifyBegin()
    try:
        curve.KeyClear()
        for fbx_time, value in zip(times, np.asarray(shown, dtype=bool).tolist()):
            key = curve.KeyAdd(fbx_time)
            key = key[0] if isinstance(key, tuple) else key
            curve.KeySet(key, fbx_time, 1.0 if value else 0.0, constant)
            curve.KeySetConstantMode(key, standard)
    finally:
        curve.KeyModifyEnd()


def _post_rotation(rotate_axis):
    """the post rotation a joint of ``rotate_axis`` is written with: fbx
    composes it as its inverse"""
    post = fbx.FbxAMatrix()
    post.SetR(fbx.FbxVector4(*rotate_axis, 1.0))
    return post.Inverse().GetR()


class FbxExporter:
    """
    Export fbx files from pipeline components

    The FBX SDK's memory is freed by :meth:`close`: use it as a context
    manager, ``with FbxExporter() as exporter:``.
    """

    _ROTATE_ORDER_MAP = {
        0: fbx.EFbxRotationOrder.eEulerXYZ,
        1: fbx.EFbxRotationOrder.eEulerYZX,
        2: fbx.EFbxRotationOrder.eEulerZXY,
        3: fbx.EFbxRotationOrder.eEulerXZY,
        4: fbx.EFbxRotationOrder.eEulerYXZ,
        5: fbx.EFbxRotationOrder.eEulerZYX,
    }
    """dict: mapping HierarchyData rotation order to fbx rotation order"""

    _USER_ATTR_TYPE_MAP = {
        "string":       fbx.FbxStringDT,
        "stringArray":  fbx.FbxStringDT,
        "bool":         fbx.FbxBoolDT,
        "short":        fbx.FbxShortDT,
        "int":          fbx.FbxIntDT,
        "long":         fbx.FbxIntDT,
        "byte":         fbx.FbxUCharDT,
        "char":         fbx.FbxCharDT,
        "enum":         fbx.FbxEnumDT,
        "float":        fbx.FbxDoubleDT,
        "double":       fbx.FbxDoubleDT,
        "doubleAngle":  fbx.FbxDoubleDT,
        "doubleLinear": fbx.FbxDoubleDT,
        "double3":      fbx.FbxDouble3DT,
        "float3":       fbx.FbxDouble3DT,
    }
    """dict: mapping Maya user defined attr types to fbx data types, the same types
    Maya's own FBX plugin writes: a double3 or float3 is one vector holding its three
    children. A stringArray, which that plugin skips, is a string holding
    _STRING_ARRAY_MARK and the list as json, %-escaped to plain ASCII. A string with
    no value, as Maya leaves one it has not set, is written empty. Types it does not
    write (doubleArray, time, other compounds, etc.) are left out on purpose, those
    attrs are skipped on export."""

    _STRING_ARRAY_MARK = "cgmath:stringArray:"
    """str: what a stringArray's string starts with, ahead of the list as json"""

    _USER_ATTR_SPECS = "cgmath:userAttributes"

    _RADIUS          = "cgmath:radius"
    _DRAW_STYLE      = "cgmath:drawStyle"
    _NEGATIVE_ZEROS  = "cgmath:negativeZeros"

    _ROTATE_AXIS = "cgmath:rotateAxis"
    """str: the hidden property on a joint keeping its rotate axis as given:
    the post rotation holds its inverse, which reads back as other angles
    for the same turn, or as the same angles a few ulps off. The readers
    take it where it turns as the post rotation does."""
    """str: the hidden property on each node keeping its user defined attr specs,
    less the values the properties hold: fbx types alone can't tell a float from a
    double, nor keep an enum's own indices, a double3's children or the channel box
    state. A value its property can't give back is kept there as ``[value]``: a
    string with none, an enum value that is no field's index, what an ASCII .fbx
    changes (see _ascii_read_back). It is not flagged user defined, so Maya's FBX
    importer leaves it alone."""

    def __init__(self):
        self._skeleton  = None
        self._meshes    = set()
        self._fbx_nodes = {}
        self._roots     = []  # the skeleton's joints the file writes as roots

        self._manager   = None
        self._scene     = None
        self._closed    = False

        # (property, text) of the long strings an ASCII export %-escapes, and
        # (property, fields) of the long enum field lists
        self._long_texts = []
        self._long_enums = []

    def _check_open(self):
        if self._closed:
            raise RuntimeError("this FbxExporter is closed; make a new one")

    @property
    def manager(self):
        """fbx.FbxManager: the fbx manager that handles the memory"""
        self._check_open()

        if self._manager is None:
            # import/export requires IO settings
            self._manager = new_manager(fbx)
        return self._manager

    @property
    def scene(self):
        """fbx.FbxScene: the fbx scene to export, labelled Maya's Y-up and cm,
        the axes and unit every cgmath value is in"""
        self._check_open()

        if self._scene is None:
            self._scene = fbx.FbxScene.Create(self.manager, "")
            settings    = self._scene.GetGlobalSettings()
            settings.SetAxisSystem(fbx.FbxAxisSystem.MayaYUp)
            settings.SetSystemUnit(fbx.FbxSystemUnit.cm)
        return self._scene

    def close(self) -> None:
        """frees the manager, and with it the scene and every node added"""
        if self._manager is not None:
            self._manager.Destroy()
        self._manager = None
        self._scene   = None
        self._closed  = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def add_skeleton(self, skeleton_component) -> None:
        """Add a skeleton component to the scene.

        Each node is added after its parent, whatever order the component
        lists them in; a node whose parent it leaves out hangs from the scene
        root, a root of the file. Every joint is written as a limb node
        (``eLimbNode``), as Maya's own exporter writes every joint: Maya's
        importer brings a root skeleton node (``eRoot``) with no child in as
        a plain transform, without its joint orient.

        Every node takes the inherit type its ``segment_scale_compensate``
        says, ``eInheritRrs`` when on and ``eInheritRSrs`` when off, a group
        and a locator too, as Maya's own exporter writes a group or a
        locator: the sdk's default, ``eInheritRrSs``, which an earlier
        version left on them, scales a node along its own axes by its
        parent's world scale, so the sdk and Maya's importer put a group
        under an unevenly scaled parent, and the nodes under it, elsewhere
        than the component has them.

        The scene is labelled Maya's Y-up and cm, the axes and unit every
        cgmath value is in: the values are written as they are, but for a
        scale under 1e-12 in size, 0 and -0 among them, written as 1e-12,
        positive, as Maya's own exporter writes a static one and as Maya's
        evaluation holds it, and for a node's shear: fbx has no channel for
        it, so scale, rotate and translate are written as they are
        (``cgmath.formats.save_fbx`` refuses a shear over 1e-9).

        Raises ValueError, before anything is added, when a user defined string
        attr is not plain ASCII, the only text the FBX SDK takes, or a node's user
        defined attr specs do not go to json.

        Notes:
            Only a single skeleton is supported.
        """
        self._check_open()
        if self._skeleton is not None:
            raise ValueError(f"Skeleton already set: {self._skeleton}!")

        # user defined attrs with no fbx data type, skipped and reported once
        skipped = {}

        # nodes that can't be exported (constraints, ikHandles, etc.), skipped
        # along with what is under them, the same way Maya's own FBX plugin
        # never writes them as scene nodes.
        # imported here, cgmath.hierarchy imports this module.
        from cgmath.hierarchy import SUPPORTED_NODE_TYPES
        from cgmath.hierarchy.hierarchy import TransformList, _fbx_maya_scale

        skipped_nodes = {}
        dropped       = set()

        # every parent ahead of its children, whatever order the list holds
        # them in -- a rig sorted, or built or loaded child first: a node's
        # parent is made, or dropped, before the node is. Depth counts the
        # parents the export holds, a node whose parent it leaves out a root,
        # so the file's roots, as siblings, keep the list's order
        members = list(skeleton_component)
        items   = [members[i] for i in TransformList._owning_parent_first(members, within=True)]

        # every node's user defined attrs are worked out first, so one that
        # can't be written raises before anything is added
        kept = []
        for item in items:
            supported = item.node_type in SUPPORTED_NODE_TYPES
            if not supported or id(item.get_parent()) in dropped:
                dropped.add(id(item))
                skipped_nodes[item.node_type] = skipped_nodes.get(item.node_type, 0) + 1
                continue
            kept.append((item, self._user_properties(item, skipped)))

        self._skeleton = skeleton_component

        # the fbx node of every item, keyed by item identity.
        # parents can't be found by name, names may be duplicated across branches.
        fbx_nodes = self._fbx_nodes = {}

        # the joints hung from the scene root, the file's skeleton roots
        roots = set()

        for item, (props, specs) in kept:
            node = fbx.FbxNode.Create(self.manager, item.name)

            fbx_nodes[id(item)] = node

            node.LclTranslation.Set(fbx.FbxDouble3(*item.translate))
            node.LclRotation.Set(fbx.FbxDouble3(*item.rotate))
            # a scale under 1e-12 in size as 1e-12, positive, as Maya's
            # exporter writes a static one: Maya's evaluation holds such a
            # scale as 1e-12, so load_fbx and Maya's import of the file then
            # place the nodes under it alike
            node.LclScaling.Set(fbx.FbxDouble3(*_fbx_maya_scale(item.scale)))

            node.SetRotationOrder(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                self._ROTATE_ORDER_MAP[item.rotate_order],
            )

            # rotation active on every node, as Maya's exporter writes it: fbx
            # ignores the rotate order of a node without it, turning it in xyz
            node.SetRotationActive(True)

            # the inherit type on every node, a group and a locator too: the
            # sdk's default, eInheritRrSs, scales a node along its own axes by
            # its parent's world scale, which no node does in cgmath or in
            # Maya, whose exporter writes eInheritRSrs on a group or locator
            inherit_type = (
                fbx.FbxTransform.EInheritType.eInheritRrs
                if item.segment_scale_compensate
                else fbx.FbxTransform.EInheritType.eInheritRSrs
            )
            node.SetTransformationInheritType(inherit_type)

            # a parent the export leaves out -- outside the view exported, or
            # a parent_uuid no node carries -- makes the node a root, as a
            # view's copy() does: never a node found by its name
            parent_node = fbx_nodes.get(id(item.get_parent()))
            rooted      = parent_node is None
            if rooted:
                parent_node = self.scene.GetRootNode()

            parent_node.AddChild(node)

            # pre and post rotation, which fbx reads with rotation active:
            # Maya's exporter writes a group's or a locator's rotate axis as
            # its post rotation too, which Maya's importer reads back as its
            # rotate axis; a joint's joint orient is its pre rotation, and a
            # node that is no joint holds none (save_fbx refuses one)
            orient = item.joint_orient if item.node_type == "joint" else (0.0, 0.0, 0.0)
            node.SetPreRotation(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                fbx.FbxVector4(*orient, 1.0),
            )
            # fbx composes post rotation as its inverse
            node.SetPostRotation(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                _post_rotation(item.rotate_axis),
            )

            if item.node_type == "joint":
                if rooted:
                    roots.add(id(item))
                # a limb node at the top too, as Maya writes one: Maya reads
                # an eRoot with no child as a transform, its orient dropped
                node_attr = fbx.FbxSkeleton.Create(self.manager, "")
                node_attr.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
            # a null's look is what tells a locator from a group, the same way
            # Maya's own FBX plugin writes them. the binding's Look.Set() only
            # takes the enum's integer value.
            elif item.node_type == "transform":
                node_attr = fbx.FbxNull.Create(self.manager, "")
                node_attr.Look.Set(fbx.FbxNull.ELook.eNone.value)
            elif item.node_type == "locator":
                node_attr = fbx.FbxNull.Create(self.manager, "")
                node_attr.Look.Set(fbx.FbxNull.ELook.eCross.value)

            # set user defined attrs
            for name, fbx_att, value, fields, keyable in props:
                prop = fbx.FbxProperty.Create(node, fbx_att, name, name)
                for field in fields:
                    prop.AddEnumValue(field)
                if isinstance(value, str) and _ascii_text(value) != value:
                    self._long_texts.append((prop, value))

                # an ASCII .fbx writes the field list as one string, "~" between
                listed = "~".join(fields)
                if _ascii_text(listed) != listed:
                    self._long_enums.append((prop, fields))
                prop.Set(value)

                # flagged the way Maya's importer expects a custom attr
                prop.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eUserDefined, True)
                prop.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, keyable)

            # not flagged user defined: Maya's importer makes no attr of it
            if specs is not None:
                prop = fbx.FbxProperty.Create(node, fbx.FbxStringDT, self._USER_ATTR_SPECS, "")
                prop.Set(specs)
            if np.any(np.signbit(item.rotate_axis) | (np.asarray(item.rotate_axis) != 0.0)):
                prop = fbx.FbxProperty.Create(node, fbx.FbxDouble3DT, self._ROTATE_AXIS, "")
                prop.Set(fbx.FbxDouble3(*[float(x) for x in item.rotate_axis]))
            # the sdk leaves out a vector of zeros, -0.0 among them, which
            # then reads 0.0: which of them are -0.0
            signs = _negative_zeros(item)
            if signs:
                prop = fbx.FbxProperty.Create(node, fbx.FbxStringDT, self._NEGATIVE_ZEROS, "")
                prop.Set(signs)

            # visibility, a joint's radius and draw style as Maya's exporter
            # writes them: Visibility, the skeleton's Size (radius * 100 / 3)
            # and Show (off for draw style None), a locator's Show its
            # visibility; what Maya's reading of them can't give back kept
            # in a hidden property
            node.Visibility.Set(1.0 if item.visibility else 0.0)
            if item.node_type == "joint":
                radius = float(item.radius)
                node_attr.Size.Set(radius * (100.0 / 3.0))
                node.Show.Set(int(item.draw_style) != 2)
                if radius * (100.0 / 3.0) * 3.0 / 100.0 != radius:
                    prop = fbx.FbxProperty.Create(node, fbx.FbxDoubleDT, self._RADIUS, "")
                    prop.Set(radius)
                if int(item.draw_style) not in (0, 2):
                    prop = fbx.FbxProperty.Create(node, fbx.FbxIntDT, self._DRAW_STYLE, "")
                    prop.Set(int(item.draw_style))
            elif item.node_type == "locator":
                node.Show.Set(bool(item.visibility))

            node.SetNodeAttribute(node_attr)

        # in the list's order, for zero_root to zero the first
        self._roots = [item for item in members if id(item) in roots]

        if skipped_nodes:
            counts = ", ".join(f"{v} {k}" for k, v in sorted(skipped_nodes.items()))
            LOGGER.warning(f"skipped nodes that can't be exported: {counts}")

        if skipped:
            counts = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
            LOGGER.warning(f"skipped user defined attrs with no fbx type: {counts}")
        LOGGER.info(f"added skeleton: {skeleton_component}")

    @classmethod
    def _user_properties(cls, item, skipped) -> tuple:
        """A node's user defined attrs as fbx properties, nothing written yet.

        Returns ``([(name, fbx type, value, enum fields, keyable), ...], specs)``,
        ``specs`` the json of the attrs' specs for _USER_ATTR_SPECS, or None.
        What has no fbx type is counted in ``skipped``.

        Raises ValueError when a string is not plain ASCII, or a spec does not
        go to json.
        """
        # imported here, cgmath.hierarchy imports this module
        from cgmath.hierarchy.hierarchy import _FLOAT_TYPES, _compound_children, _enum_fields

        attrs = item.user_defined_attributes or {}

        # a double3 or float3 is one vector of its three children, as Maya's own
        # FBX plugin writes it; any other compound's children go on their own
        vectors = {}
        for name, spec in attrs.items():
            if spec.get("attributeType") not in ("double3", "float3") or spec.get("multi"):
                continue
            children = _compound_children(attrs, name)
            if len(children) == 3 and all(
                attrs[child].get("attributeType") in _FLOAT_TYPES
                and attrs[child].get("value") is not None
                and not attrs[child].get("multi")
                for child in children
            ):
                vectors[name] = children
        held  = {child for children in vectors.values() for child in children}

        props = []
        specs = {}
        for name, spec in attrs.items():
            if name in held:
                continue

            if "attributeType" in spec:
                attr_type = spec["attributeType"]
            else:
                attr_type = spec["dataType"]

            fbx_att = cls._USER_ATTR_TYPE_MAP.get(attr_type)
            value   = spec.get("value")
            if attr_type in ("double3", "float3") and name not in vectors:
                fbx_att = None

            # skip what Maya's own FBX plugin does not write either: unmapped
            # types, multi attrs and compound parents (no value). a string
            # with no value, as Maya leaves one it has not set, is written empty
            if spec.get("multi"):
                attr_type = f"{attr_type}[]"
            unset = value is None and name not in vectors
            if (
                fbx_att is None
                or spec.get("multi")
                or (unset and attr_type not in ("string", "stringArray"))
            ):
                skipped[attr_type] = skipped.get(attr_type, 0) + 1
                continue
            if isinstance(value, str) and not value.isascii():
                # the first character that is not, the string can be long
                at = re.search(r"[^\x00-\x7f]", value).start()
                raise ValueError(
                    f"user defined attr {item.name}.{name} can't be written to fbx: "
                    f"its character {value[at]!r} (U+{ord(value[at]):04X}) at index "
                    f"{at} is not plain ASCII, the only text the FBX SDK takes"
                )

            # the spec comes back as it is, its value from the property, but
            # for one the property can't give back: no value, an enum value that
            # is no field's index, what an ASCII .fbx changes
            fields = []
            keep   = unset
            if name in vectors:
                written = fbx.FbxDouble3(*[attrs[child]["value"] for child in vectors[name]])
            else:
                written = _user_property_value(spec, value)
                keep    = keep or _ascii_read_back(written, attr_type) != written
            if attr_type == "enum":
                pairs  = _enum_fields(spec.get("enumName", ""))
                fields = [field for field, _ in pairs]
                keep   = keep or value not in [index for _, index in pairs]

            props.append((name, fbx_att, written, fields, bool(spec.get("keyable"))))

            # a child of a compound left out comes back on its own
            specs[name] = _kept_spec(spec, keep)
            specs[name].pop("parent", None)
            for child in vectors.get(name, ()):
                number       = attrs[child]["value"]
                kind         = attrs[child]["attributeType"]
                specs[child] = _kept_spec(attrs[child], _ascii_read_back(number, kind) != number)

        if not specs:
            return props, None
        try:
            return props, _ascii_json(specs)
        except TypeError as error:
            raise ValueError(
                f"the user defined attrs of {item.name} can't be written to fbx: {error}"
            ) from None

    def export(self, path: pathlib.Path, as_ascii=False, zero_root=False) -> None:
        """Export the scene to the given path, binary unless as_ascii is set.

        zero_root writes the skeleton's first joint hung from the scene root,
        in the order the skeleton lists them, with an identity transform;
        ValueError when there is none.

        The FBX SDK's ASCII reader changes a long string holding a quote, line
        break, tab, semicolon, &-entity or run of spaces, and loses the whole
        scene for a brace, so in an ASCII file a user defined string, or an
        enum's field list, over _ASCII_TEXT_LIMIT bytes holding one is
        %-escaped, with a warning. load_fbx reads it back as it was; Maya reads
        the escaped text.
        """
        file_format = self.manager.GetIOPluginRegistry().GetNativeWriterFormat()
        if as_ascii:
            file_format = self._get_acsii_type()

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        # escaped while this file is written only, a binary one keeps them as they are
        escaped = self._long_texts if as_ascii else []
        enums   = self._long_enums if as_ascii else []
        for prop, text in escaped:
            prop.Set(_ascii_text(text))
        for prop, fields in enums:
            _set_enum_fields(prop, [urllib.parse.quote(x, safe=_PLAIN_TEXT) for x in fields])

        exporter = fbx.FbxExporter.Create(self.manager, "")
        try:
            self._export(exporter, path, file_format, zero_root)
        finally:
            exporter.Destroy()
            for prop, text in escaped:
                prop.Set(text)
            for prop, fields in enums:
                _set_enum_fields(prop, fields)

        if escaped or enums:
            LOGGER.warning(
                "%-escaped user defined strings and enum field lists over "
                f"{_ASCII_TEXT_LIMIT} bytes, which an ASCII file can't keep as they are "
                "(load_fbx reads them back as they were, a binary file keeps them as "
                f"they are): {len(escaped) + len(enums)}"
            )
        LOGGER.info(f"exported to: {path}")

    def _export(self, exporter, path, file_format, zero_root) -> None:
        if not exporter.Initialize(
            path.as_posix(), file_format, self.manager.GetIOSettings()
        ):
            raise Exception(
                f"Failed to initialize exporter: {exporter.GetStatus().GetErrorString()}"
            )

        # ensure parent dir exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # zero-out root translations and rotations
        if zero_root:
            skel_root = self._zero_root()
            skel_root.LclTranslation.Set(fbx.FbxDouble3(0, 0, 0))
            skel_root.LclRotation.Set(fbx.FbxDouble3(0, 0, 0))
            skel_root.LclScaling.Set(fbx.FbxDouble3(1, 1, 1))

            skel_root.SetRotationActive(True)
            skel_root.SetPreRotation(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                fbx.FbxVector4(0, 0, 0, 1.0),
            )
            skel_root.SetPostRotation(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                fbx.FbxVector4(0, 0, 0, 1.0),
            )

        if not exporter.Export(self.scene):
            raise Exception(f"Failed to export to: {path}!")

    def _zero_root(self):
        """the FbxNode zero_root zeroes: the first joint the file writes as a
        root, one at the world or one whose parent the export leaves out, a
        view's top node (meshes and locators can sit at the world too)"""
        roots = self._roots
        if not roots:
            raise ValueError(
                "zero_root needs a joint parented to the world, or whose "
                "parent the export leaves out"
            )
        root_name = roots[0].name

        # by identity: names may repeat across branches
        LOGGER.debug(f"Zero-ing Root: {root_name}")
        skel_root = self._fbx_nodes.get(id(roots[0]))
        if skel_root is None:
            raise RuntimeError(f"Could not find {root_name} to reset!")
        return skel_root

    def add_takes(self, clips, names, time_mode, zero_root: bool = False, columns=None, active=None, ranges=None) -> None:
        """One animation stack per clip, after :meth:`add_skeleton`

        ``clips`` are takes of the skeleton added, their nodes in its order,
        or, with ``columns``, the clip column each skeleton node is keyed
        from, None for a node left unkeyed (a rig the takes animate by name);
        ``names`` each one's stack name and ``time_mode`` the
        ``fbx.FbxTime.EMode`` of their frame rate, which the scene takes.
        Each stack spans its clip's frames and holds a BaseLayer in which
        every node the skeleton wrote, the root ``zero_root`` zeroes aside,
        has its Lcl Translation, Rotation and Scaling X, Y and Z keyed on
        every frame, linear, at ``(start_frame + f) / fps``: the values each
        frame holds, keyed scales as they are, as Maya's own export writes
        them. The curves are made on each node itself, never by name.
        """
        self.scene.GetGlobalSettings().SetTimeMode(time_mode)
        skeleton = list(self._skeleton) if self._skeleton is not None else []
        skipped  = self._zero_root() if zero_root else None
        columns  = range(len(skeleton)) if columns is None else columns
        looked   = [(column, self._fbx_nodes.get(id(node)), node) for column, node in zip(columns, skeleton)]
        looked   = [(column, node, item) for column, node, item in looked if column is not None and node is not None]
        written  = [(column, node) for column, node, _ in looked if node is not skipped]

        ranges   = [None] * len(clips) if ranges is None else ranges
        second   = fbx.FbxTime()
        second.SetSecondDouble(1.0)
        for clip, name, held in zip(clips, names, ranges):
            stack = fbx.FbxAnimStack.Create(self.scene, name)
            self.scene.AddMember(stack)
            first, last = fbx.FbxTime(), fbx.FbxTime()
            first.SetSecondDouble(clip.start_frame / clip.fps)
            last.SetSecondDouble(clip.end_frame / clip.fps)
            if held is not None:
                # the span the take was read with, in its frames
                first.Set(int(round(held[0] * second.Get() / clip.fps)))
                last.Set(int(round(held[1] * second.Get() / clip.fps)))
            # the sdk writes no span (0, 0) and reads another in its place:
            # one tick past 0, which a read puts on frame 0
            if first.Get() == 0 and last.Get() == 0:
                last.Set(1)
            stack.SetLocalTimeSpan(fbx.FbxTimeSpan(first, last))
            stack.SetReferenceTimeSpan(fbx.FbxTimeSpan(first, last))
            layer = fbx.FbxAnimLayer.Create(self.scene, "BaseLayer")
            stack.AddMember(layer)
            times  = _fbx_times((clip.start_frame + np.arange(clip.frame_count)) / clip.fps)
            mode   = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear

            frames = clip.frames
            blocks = (
                ("Lcl Translation", np.asarray(frames.translate)),
                ("Lcl Rotation", np.asarray(frames.rotate)),
                ("Lcl Scaling", np.asarray(frames.scale)),
            )
            for column, fbx_node in written:
                for channel, block in blocks:
                    prop = fbx_node.FindProperty(channel)
                    for axis, component in enumerate("XYZ"):
                        # GetCurve makes the curve node and its connections
                        _set_keys(prop.GetCurve(layer, component, True), times, block[:, column, axis], mode)

            # a node's visibility other than the file's on any frame of the
            # take: a Visibility curve keyed on every frame, stepped (constant,
            # standard), 0 or 1, as Maya's FBXExport keys one
            shown = np.asarray(frames.visibility)
            for column, fbx_node, item in looked:
                if np.any(shown[:, column] != bool(item.visibility)):
                    _key_visibility(fbx_node, layer, times, shown[:, column])

        # the take Maya's import opens on: ``active`` where it names one of
        # them, else the first
        names = list(names)
        if names:
            self.scene.ActiveAnimStackName.Set(fbx.FbxString(active if active in names else names[0]))

    def add_mesh(self, mesh, node=None):
        """Adds a ``MeshData`` as an ``FbxMesh`` and returns it.

        The points are written as they are, in the unit and axes the scene
        is labelled with, the faces in their order with their vertex counts
        (ngons kept). Normals are written as the mesh holds them: through
        ``normal_indices``, indexed, one a face corner (``eByPolygonVertex``,
        ``eIndexToDirect``); else one a vertex (``eByControlPoint``, as
        Maya's exporter writes smooth normals) or one a corner; none when it
        has none, unless ``normal_indices`` holds groups of corners, which
        write the normals derived inside them. ``node`` is the ``FbxNode``
        that takes the mesh, a group the
        skeleton wrote, whose own transform it keeps; None makes one at the
        scene root, named after the mesh, at identity: a mesh holds no
        transform of its own. Raises ValueError for normals that fit neither
        the vertices nor the corners, or a mesh holding polygon holes,
        which this writer does not write.
        """
        # imported here: cgmath.geometry imports this module's package
        from cgmath.geometry.mesh import _corner_normals
        from cgmath.hierarchy.hierarchy import TransformData, _fbx_maya_scale

        self._check_open()
        name  = mesh.name or "mesh"
        label = f"mesh {name!r}"
        if mesh.hole_faces is not None and len(mesh.hole_faces):
            raise ValueError(f"{label} holds polygon holes, which save_fbx does not write")
        normals, ids = _corner_normals(mesh)

        if node is None:
            # a node at identity, its channels a transform's defaults
            own  = TransformData(name)
            node = fbx.FbxNode.Create(self.manager, name)
            node.LclTranslation.Set(fbx.FbxDouble3(*own.translate))
            node.LclRotation.Set(fbx.FbxDouble3(*own.rotate))
            node.LclScaling.Set(fbx.FbxDouble3(*_fbx_maya_scale(own.scale)))
            node.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, self._ROTATE_ORDER_MAP[own.rotate_order])
            node.SetRotationActive(True)
            node.SetTransformationInheritType(fbx.FbxTransform.EInheritType.eInheritRSrs)
            self.scene.GetRootNode().AddChild(node)

        shape  = fbx.FbxMesh.Create(self.manager, f"{name}Shape")
        points = np.asarray(mesh.points, dtype=np.float64).tolist()
        shape.InitControlPoints(len(points))
        for index, point in enumerate(points):
            shape.SetControlPointAt(fbx.FbxVector4(*point, 1.0), index)

        indices = np.asarray(mesh.indices, dtype=np.int64).tolist()
        start   = 0
        for size in np.asarray(mesh.counts, dtype=np.int64).tolist():
            shape.BeginPolygon(-1, -1, -1, False)
            for vertex in indices[start : start + size]:
                shape.AddPolygon(vertex)
            shape.EndPolygon()
            start += size

        if normals is not None:
            mapping = fbx.FbxLayerElement.EMappingMode
            refs    = fbx.FbxLayerElement.EReferenceMode
            element = shape.CreateElementNormal()
            if mesh.normal_indices is not None:
                element.SetMappingMode(mapping.eByPolygonVertex)
                element.SetReferenceMode(refs.eIndexToDirect)
                index_array = element.GetIndexArray()
                for index in np.asarray(ids, dtype=np.int64).tolist():
                    index_array.Add(index)
            else:
                # one a vertex, or one a corner: the ids _corner_normals gave
                by_point = normals.shape[0] == len(points)
                element.SetMappingMode(mapping.eByControlPoint if by_point else mapping.eByPolygonVertex)
                element.SetReferenceMode(refs.eDirect)
            direct = element.GetDirectArray()
            for normal in normals.tolist():
                direct.Add(fbx.FbxVector4(*normal, 0.0))

        node.SetNodeAttribute(shape)
        LOGGER.info(f"added mesh: {name}")
        return shape

    def add_uvs(self, shape, uvs) -> None:
        """Adds UV sets, ``UVData`` each, to an ``FbxMesh`` :meth:`add_mesh`
        returned, in their order, each named after its set: its UVs as they
        are, one a face corner through its ``indices`` (``eByPolygonVertex``,
        ``eIndexToDirect``, as Maya's exporter writes a UV set), each set an
        element of its own. The sets' faces are the mesh's, which the caller
        checks (``cgmath.formats.save_fbx`` pairs them)."""
        self._check_open()
        mapping = fbx.FbxLayerElement.EMappingMode
        refs    = fbx.FbxLayerElement.EReferenceMode
        for uv in uvs:
            element = shape.CreateElementUV(uv.name)
            element.SetMappingMode(mapping.eByPolygonVertex)
            element.SetReferenceMode(refs.eIndexToDirect)
            direct = element.GetDirectArray()
            for point in np.asarray(uv.points, dtype=np.float64).tolist():
                direct.Add(fbx.FbxVector2(*point))
            index_array = element.GetIndexArray()
            for index in np.asarray(uv.indices, dtype=np.int64).tolist():
                index_array.Add(index)

    def add_skin(self, shape, skin, links) -> None:
        """Adds a ``SkinData`` to an ``FbxMesh`` :meth:`add_mesh` returned:
        an ``FbxSkin``, one ``FbxCluster`` an influence, in the skin's order,
        linked to the ``FbxNode`` of ``links`` in that order (nodes the
        skeleton wrote), and a bind pose listing the mesh's node and the
        links, as Maya's exporter writes one a skin.

        The nodes as written are the bind pose: each cluster's link matrix
        is its node's world and its transform matrix the mesh node's, as the
        sdk evaluates them from the channels written (a static scale under
        1e-12 as 1e-12), and the bind pose holds the same worlds. Each
        cluster lists the vertices its influence weighs other than 0, with
        their weights as stored (float64), normalized link mode, as Maya's
        exporter writes it; an influence weighing none still has its
        cluster."""
        self._check_open()
        node    = typed(shape.GetNode(), fbx.FbxNode, fbx)
        name    = node.GetName()
        worlds  = [link.EvaluateGlobalTransform() for link in links]
        placed  = node.EvaluateGlobalTransform()
        weights = np.asarray(skin.weights, dtype=np.float64)
        made    = fbx.FbxSkin.Create(self.manager, f"{name}Skin")
        for column, (influence, link) in enumerate(zip(skin.influences, links)):
            cluster = fbx.FbxCluster.Create(self.manager, influence)
            cluster.SetLink(link)
            cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)
            rows = np.flatnonzero(weights[:, column])
            for vertex, weight in zip(rows.tolist(), weights[rows, column].tolist()):
                cluster.AddControlPointIndex(vertex, weight)
            cluster.SetTransformMatrix(placed)
            cluster.SetTransformLinkMatrix(worlds[column])
            made.AddCluster(cluster)
        shape.AddDeformer(made)

        pose = fbx.FbxPose.Create(self.manager, f"{name}BindPose")
        pose.SetIsBindPose(True)
        pose.Add(node, fbx.FbxMatrix(placed))
        for link, world in zip(links, worlds):
            pose.Add(link, fbx.FbxMatrix(world))
        self.scene.AddPose(pose)

        # Maya's FBXExport locks a skinned mesh's translate, rotate and scale,
        # members 0 to 2, and its import leaves a locked member where it is
        flags = fbx.FbxPropertyFlags.EFlags
        for prop in (node.LclTranslation, node.LclRotation, node.LclScaling):
            for member in (flags.eLockedMember0, flags.eLockedMember1, flags.eLockedMember2):
                prop.ModifyFlag(member, True)
        LOGGER.info(f"added skin: {name}, {len(links)} influences")

    def _get_acsii_type(self):
        plg_rego     = self.manager.GetIOPluginRegistry()
        file_format  = plg_rego.GetNativeWriterFormat()
        format_count = plg_rego.GetWriterFormatCount()
        for format_index in range(format_count):
            if plg_rego.WriterIsFBX(format_index):
                desc = plg_rego.GetWriterFormatDescription(format_index)
                if "ascii" in desc:
                    file_format = format_index
                    break
        return file_format