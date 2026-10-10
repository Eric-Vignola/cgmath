"""
FbxExporter unit tests, skeleton export.

These need the Autodesk FBX SDK (the `fbx` module), they are skipped without it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import unittest
import urllib.parse
import uuid
import warnings

import numpy as np

from cgmath.hierarchy import HierarchyData, TransformData, TransformList

try:
    import fbx
except ImportError:
    fbx = None


def _joint(name, parent=None, **kwargs):
    """Returns a joint TransformData with a uuid, parented by uuid."""
    return TransformData(
        name        = name,
        node_type   = "joint",
        uuid        = str(uuid.uuid4()).upper(),
        parent_uuid = parent.uuid if parent is not None else None,
        **kwargs,
    )


@contextlib.contextmanager
def _no_fbx_warnings(test):
    """
    ``test.assertNoLogs("cgmath.formats.fbx", level="WARNING")`` done by hand,
    as assertNoLogs is new in python 3.10: the logger's records at WARNING and
    above are caught for the block, and the test fails if there are any.
    """
    logger           = logging.getLogger("cgmath.formats.fbx")
    records          = []
    handler          = logging.Handler(logging.WARNING)
    handler.emit     = records.append
    saved            = logger.handlers[:], logger.level, logger.propagate
    logger.handlers  = [handler]
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.handlers, logger.propagate = saved[0], saved[2]
        logger.setLevel(saved[1])
    test.assertEqual([], [record.getMessage() for record in records])


class TestNoFbxWarnings(unittest.TestCase):
    """the stand-in for assertNoLogs runs without the FBX SDK, on any python"""

    def test_a_warning_fails_and_the_logger_is_restored(self):
        logger = logging.getLogger("cgmath.formats.fbx")
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel(logging.ERROR)  # a warning is caught all the same
        before = logger.handlers[:], logging.ERROR, logger.propagate

        with self.assertRaises(AssertionError) as caught:
            with _no_fbx_warnings(self):
                logger.warning("1 matrix skipped")
        self.assertIn("1 matrix skipped", str(caught.exception))

        with _no_fbx_warnings(self):
            logger.info("not a warning")
        self.assertEqual((logger.handlers, logger.level, logger.propagate), before)


@unittest.skipIf(fbx is None, "the Autodesk FBX SDK is required")
class TestFbxExportSkeleton(unittest.TestCase):
    def setUp(self):
        self.tmp  = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "skeleton.fbx")

    def tearDown(self):
        self.tmp.cleanup()

    def _export(self, hierarchy):
        # save_fbx() rather than a bare FbxExporter, it is what flags
        # the user defined properties as such.
        hierarchy.save_fbx(self.path)

    def _walk(self, read_node):
        """Returns {node path: read_node(fbx node)} for every node in the file."""
        manager  = fbx.FbxManager.Create()
        settings = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
        manager.SetIOSettings(settings)
        importer = fbx.FbxImporter.Create(manager, "")
        self.assertTrue(importer.Initialize(self.path, -1, settings))
        scene = fbx.FbxScene.Create(manager, "")
        importer.Import(scene)

        result = {}

        def _recurse(node, path):
            path         = f"{path}|{node.GetName()}"
            result[path] = read_node(node)
            for i in range(node.GetChildCount()):
                _recurse(node.GetChild(i), path)

        root = scene.GetRootNode()
        for i in range(root.GetChildCount()):
            _recurse(root.GetChild(i), "")

        # release the file, or the temp directory can't be removed on Windows
        importer.Destroy()
        manager.Destroy()
        return result

    def _read(self):
        """Returns {node path: {user defined property name: fbx data type name}}."""
        user_defined = fbx.FbxPropertyFlags.EFlags.eUserDefined

        def _props(node):
            props = {}
            prop  = node.GetFirstProperty()
            while prop.IsValid():
                if prop.GetFlag(user_defined):
                    props[str(prop.GetName())] = str(prop.GetPropertyDataType().GetName())
                prop = node.GetNextProperty(prop)
            return props

        return self._walk(_props)

    def _read_flags(self):
        """Returns {node path: {user defined property name: (user defined, animatable)}}."""
        flags = fbx.FbxPropertyFlags.EFlags

        def _flags(node):
            props = {}
            prop  = node.GetFirstProperty()
            while prop.IsValid():
                if prop.GetFlag(flags.eUserDefined):
                    props[str(prop.GetName())] = (True, prop.GetFlag(flags.eAnimatable))
                prop = node.GetNextProperty(prop)
            return props

        return self._walk(_flags)

    def _read_enum(self, name):
        """Returns ([field names], value) of a user defined enum property on the root."""

        def _enum(node):
            prop = node.FindProperty(name)
            if not prop.IsValid():
                return None
            fields = [prop.GetEnumValue(i) for i in range(prop.GetEnumCount())]
            return (fields, fbx.FbxPropertyInteger1(prop).Get())

        return self._walk(_enum)["|root"]

    def _read_attributes(self):
        """Returns {node path: (node attribute class name, null look name or None)}."""

        def _attribute(node):
            attr = node.GetNodeAttribute()
            name = attr.GetClassId().GetName()
            look = attr.Look.Get().name if name == "FbxNull" else None
            return (name, look)

        return self._walk(_attribute)

    def test_duplicate_names_keep_their_parent(self):
        root  = _joint("root")
        nodes = [root]
        for side in ("L", "R"):
            arm  = _joint(side,   root)
            hand = _joint("hand", arm)
            tip  = _joint("tip",  hand)
            nodes.extend([arm, hand, tip])

        self._export(HierarchyData(nodes))

        expected = [
            "|root",
            "|root|L",
            "|root|L|hand",
            "|root|L|hand|tip",
            "|root|R",
            "|root|R|hand",
            "|root|R|hand|tip",
        ]
        self.assertEqual(sorted(self._read()), expected)

    def test_a_parent_left_out_makes_a_root(self):
        # a view without the parent, a parent taken out with del: the node
        # is a root of the file and a skeleton root, never hung from a node
        # written before it that carries the parent's name. 938ae44 looked
        # the parent up by name: the second view hung knee from the other
        # hip, and the others raised AttributeError
        def chain():
            root = _joint("root")
            hip  = _joint("hip", root)
            knee = _joint("knee", hip)
            return HierarchyData([root, hip, knee])

        def skeleton_types():
            return self._walk(lambda node: node.GetNodeAttribute().GetSkeletonType())

        rig = chain()
        rig.append(_joint("hip"))  # another hip, a root
        for view, expected in (
            (rig.match("knee"), ["|knee"]),
            (TransformList([rig.list[3], rig["knee"]]), ["|hip", "|knee"]),
        ):
            with self.subTest(expected):
                self._export(view)
                self.assertEqual(sorted(self._read()), expected)
                self.assertEqual(skeleton_types()["|knee"], fbx.FbxSkeleton.EType.eRoot)

        rig = chain()
        del rig[0]
        self.assertIsInstance(rig["hip"].get_parent(), str)  # the uuid of no node
        self._export(rig)
        self.assertEqual(sorted(self._read()), ["|hip", "|hip|knee"])
        types = skeleton_types()
        self.assertEqual(types["|hip"], fbx.FbxSkeleton.EType.eRoot)
        self.assertEqual(types["|hip|knee"], fbx.FbxSkeleton.EType.eLimbNode)

    def test_a_rig_listed_child_first_keeps_its_hierarchy(self):
        # each node is written after its parent, whatever order the list
        # holds them in: a rig sorted, one built child first. Written in list
        # order, a node listed ahead of its parent found no fbx node for it
        # and hung from the scene root, a skeleton root at the world, with no
        # error (938ae44 and 1.0.4 raised AttributeError there), and a node
        # listed ahead of a skipped parent was written
        def chain():
            root = _joint("root", translate=(0, 1, 0))
            hip  = _joint("hip", root, translate=(1, 0, 0))
            knee = _joint("knee", hip, translate=(0, -2, 0))
            return root, hip, knee

        def skeleton_types():
            return self._walk(lambda node: node.GetNodeAttribute().GetSkeletonType())

        def sorted_rig():
            rig = HierarchyData(chain())
            rig.sort()  # hip, knee, root
            return rig

        limb = fbx.FbxSkeleton.EType.eLimbNode
        for label, rig in (("sorted", sorted_rig()), ("built child first", HierarchyData(chain()[::-1]))):
            with self.subTest(label):
                self.assertEqual(rig.get_parents().tolist()[:2], [2, 0] if label == "sorted" else [1, 2])
                worlds = {node.name: node.world_matrix[3, :3] for node in rig}
                self._export(rig)
                self.assertEqual(sorted(self._read()), ["|root", "|root|hip", "|root|hip|knee"])
                types = skeleton_types()
                self.assertEqual(types["|root"], fbx.FbxSkeleton.EType.eRoot)
                self.assertEqual((types["|root|hip"], types["|root|hip|knee"]), (limb, limb))
                for node in HierarchyData.load_fbx(self.path):
                    np.testing.assert_allclose(node.world_matrix[3, :3], worlds[node.name], atol=1e-9)

        # a child listed ahead of a constraint the export skips goes with it
        root, hip, knee = chain()
        con           = _joint("con", knee)
        below         = _joint("below", con)
        con.node_type = "parentConstraint"
        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            self._export(HierarchyData([below, con, knee, hip, root]))
        self.assertIn("1 joint", logs.output[0])
        self.assertIn("1 parentConstraint", logs.output[0])
        self.assertEqual(sorted(self._read()), ["|root", "|root|hip", "|root|hip|knee"])

    def test_null_look_tells_a_locator_from_a_group(self):
        # the same way Maya's own FBX plugin writes them
        root  = _joint("root")
        group = _joint("group", root)
        loc   = _joint("loc", root)

        group.node_type = "transform"
        loc.node_type   = "locator"

        self._export(HierarchyData([root, group, loc]))

        expected = {
            "|root":       ("FbxSkeleton", None),
            "|root|group": ("FbxNull", "eNone"),
            "|root|loc":   ("FbxNull", "eCross"),
        }
        self.assertEqual(self._read_attributes(), expected)

        # and the reader tells them apart again
        loaded = HierarchyData.load_fbx(self.path)
        self.assertEqual(
            [(x.name, x.node_type) for x in loaded],
            [("root", "joint"), ("group", "transform"), ("loc", "locator")],
        )

    def test_unsupported_node_types_are_skipped(self):
        # a constraint can't be exported, it is skipped along with what is under it
        weight = {"w0": {"attributeType": "double", "value": 1.0, "keyable": True}}
        root   = _joint("root")
        kept   = _joint("kept", root)
        con    = _joint("con", root, user_defined_attributes=weight)
        below  = _joint("below", con)

        con.node_type = "parentConstraint"

        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            self._export(HierarchyData([root, kept, con, below]))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("1 joint", logs.output[0])
        self.assertIn("1 parentConstraint", logs.output[0])
        self.assertEqual(sorted(self._read()), ["|root", "|root|kept"])

    def test_user_attr_types(self):
        # the fbx data types Maya's own FBX plugin writes for each attr type
        attrs = {
            "a_string": {"dataType": "string", "value": "abc", "keyable": False},
            "a_bool":   {"attributeType": "bool", "value": True, "keyable": True},
            "a_short":  {"attributeType": "short", "value": 3, "keyable": True},
            "a_long":   {"attributeType": "long", "value": 3, "keyable": True},
            "a_byte":   {"attributeType": "byte", "value": 3, "keyable": True},
            "a_char":   {"attributeType": "char", "value": 3, "keyable": True},
            "a_enum":   {"attributeType": "enum", "value": 1, "keyable": True},
            "a_float":  {"attributeType": "float", "value": 1.5, "keyable": True},
            "a_double": {"attributeType": "double", "value": 1.5, "keyable": True},
            "a_angle":  {"attributeType": "doubleAngle", "value": 1.5, "keyable": True},
            "a_linear": {"attributeType": "doubleLinear", "value": 1.5, "keyable": True},
        }
        root = _joint("root", user_defined_attributes=attrs)

        with _no_fbx_warnings(self):
            self._export(HierarchyData([root]))

        expected = {
            "a_string": "KString",
            "a_bool":   "Bool",
            "a_short":  "Short",
            "a_long":   "Integer",
            "a_byte":   "UByte",
            "a_char":   "Byte",
            "a_enum":   "Enum",
            "a_float":  "Number",
            "a_double": "Number",
            "a_angle":  "Number",
            "a_linear": "Number",
        }
        self.assertEqual(self._read()["|root"], expected)

        # flagged as custom attrs, animatable when keyable
        flags = self._read_flags()["|root"]
        self.assertEqual(
            flags, {name: (True, spec["keyable"]) for name, spec in attrs.items()}
        )

    def test_ascii_and_binary(self):
        root = _joint("root", user_defined_attributes={
            "a_double": {"attributeType": "double", "value": 1.5, "keyable": True},
        })

        # binary by default
        self._export(HierarchyData([root]))
        with open(self.path, "rb") as io:
            self.assertTrue(io.read(18).startswith(b"Kaydara FBX Binary"))
        self.assertEqual(self._read()["|root"], {"a_double": "Number"})

        HierarchyData([root]).save_fbx(self.path, as_ascii=True)
        with open(self.path, "rb") as io:
            self.assertTrue(io.read(5).startswith(b"; FBX"))
        self.assertEqual(self._read()["|root"], {"a_double": "Number"})

    def test_enum_fields_and_multi(self):
        attrs = {
            # Maya's explicit indices become fbx positions
            "a_enum":  {"attributeType": "enum", "value": 5, "keyable": True, "enumName": "a=1:b=5:c"},
            "a_multi": {"attributeType": "double", "value": [[0, 1.5], [2, 3.5]], "keyable": True, "multi": True},
        }
        root = _joint("root", user_defined_attributes=attrs)

        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            self._export(HierarchyData([root]))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("1 double[]", logs.output[0])
        self.assertEqual(self._read()["|root"], {"a_enum": "Enum"})
        self.assertEqual(self._read_enum("a_enum"), (["a", "b", "c"], 1))

    def test_user_attr_without_fbx_type_is_skipped(self):
        attrs = {
            "a_maps":   {"dataType": "doubleArray", "value": [1.0, 2.0], "keyable": False},
            "a_matrix": {"attributeType": "matrix", "value": [0.0] * 16, "keyable": False},
            "a_uuids":  {"dataType": "stringArray", "value": ["a", "b"], "keyable": False},
            "a_double": {"attributeType": "double", "value": 1.5, "keyable": True},
        }
        root  = _joint("root", user_defined_attributes=attrs)
        child = _joint("child", root, user_defined_attributes={"a_maps": attrs["a_maps"]})

        # the export completes, the skipped attrs are reported once
        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            self._export(HierarchyData([root, child]))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("2 doubleArray", logs.output[0])
        self.assertIn("1 matrix", logs.output[0])
        self.assertNotIn("stringArray", logs.output[0])

        # a stringArray is a string, which Maya's own FBX plugin does not write
        result = self._read()
        self.assertEqual(result["|root"], {"a_uuids": "KString", "a_double": "Number"})
        self.assertEqual(result["|root|child"], {})

    def test_a_user_attr_set_as_a_property_is_exported(self):
        attrs = {"heroHeight": {"attributeType": "double", "value": 1.8, "keyable": True}}
        root  = _joint("root", user_defined_attributes=attrs)

        root.heroHeight = 3.1416
        self._export(HierarchyData([root]))

        def _value(node):
            return fbx.FbxPropertyDouble1(node.FindProperty("heroHeight")).Get()

        self.assertAlmostEqual(self._walk(_value)["|root"], 3.1416)

    def test_a_double3_is_one_vector(self):
        # as Maya's own FBX plugin writes it, which its importer reads back
        # as a double3 of three children
        attrs = {
            "pivot":  {"attributeType": "double3", "numberOfChildren": 3, "keyable": True},
            "pivotX": {"attributeType": "double", "value": 0.5, "parent": "pivot", "keyable": True},
            "pivotY": {"attributeType": "double", "value": 1.0, "parent": "pivot", "keyable": True},
            "pivotZ": {"attributeType": "double", "value": 2.0, "parent": "pivot", "keyable": True},
        }
        root = _joint("root", user_defined_attributes=attrs)

        with _no_fbx_warnings(self):
            self._export(HierarchyData([root]))

        self.assertEqual(self._read()["|root"], {"pivot": "Vector"})

        def _value(node):
            return tuple(fbx.FbxPropertyDouble3(node.FindProperty("pivot")).Get())

        self.assertEqual(self._walk(_value)["|root"], (0.5, 1.0, 2.0))

    def test_a_non_ascii_string_raises_before_writing(self):
        # the SDK takes ASCII strings only; it used to fail with a bare
        # SystemError half way through the scene, and write no file
        from cgmath.formats.fbx import FbxExporter

        attrs = {"label": {"dataType": "string", "value": "hanche gauche \u00e9", "keyable": False}}
        root  = _joint("root")
        hip   = _joint("hip", root, user_defined_attributes=attrs)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                with self.assertRaisesRegex(ValueError, r"hip\.label .* not plain ASCII"):
                    HierarchyData([root, hip]).save_fbx(self.path, as_ascii=as_ascii)
                self.assertFalse(os.path.exists(self.path))

        # the exporter is left empty, and takes the skeleton once it is fixed
        with FbxExporter() as exporter:
            with self.assertRaises(ValueError):
                exporter.add_skeleton(HierarchyData([root, hip]))
            self.assertEqual(exporter.scene.GetRootNode().GetChildCount(), 0)

            hip.label = "hanche gauche"
            exporter.add_skeleton(HierarchyData([root, hip]))
            exporter.export(self.path)
        self.assertEqual(HierarchyData.load_fbx(self.path)[1].label, "hanche gauche")

    def test_a_non_ascii_string_error_names_the_character(self):
        # it held the whole string instead: 200 KB of message for this one,
        # and no word of which character was not ASCII
        attrs = {"blob": {"dataType": "string", "value": "x" * 200_000 + "\u00e9", "keyable": False}}
        root  = _joint("root", user_defined_attributes=attrs)

        with self.assertRaises(ValueError) as caught:
            HierarchyData([root]).save_fbx(self.path)
        message = str(caught.exception)

        self.assertIn("root.blob", message)
        self.assertIn("'\u00e9' (U+00E9) at index 200000 is not plain ASCII", message)
        self.assertLess(len(message), 200)
        self.assertFalse(os.path.exists(self.path))

    def test_a_spec_that_does_not_go_to_json_raises_before_writing(self):
        from cgmath.formats.fbx import FbxExporter

        attrs = {"twist": {"attributeType": "double", "value": 0.5, "tags": {"a", "b"}}}
        root  = _joint("root")
        hip   = _joint("hip", root, user_defined_attributes=attrs)

        with self.assertRaisesRegex(ValueError, r"attrs of hip can't be written .* a set"):
            HierarchyData([root, hip]).save_fbx(self.path)
        self.assertFalse(os.path.exists(self.path))

        # the exporter is left empty, and takes the skeleton once it is fixed
        with FbxExporter() as exporter:
            with self.assertRaises(ValueError):
                exporter.add_skeleton(HierarchyData([root, hip]))
            self.assertEqual(exporter.scene.GetRootNode().GetChildCount(), 0)

            hip.user_defined_attributes["twist"]["tags"] = ["a", "b"]
            exporter.add_skeleton(HierarchyData([root, hip]))
            exporter.export(self.path)
        self.assertEqual(HierarchyData.load_fbx(self.path)[1].twist, 0.5)


def _spec(kind, value=None, keyable=True, channel_box=False, **extra):
    """a user attribute spec laid out the way rig serializes one"""
    key  = "dataType" if kind in ("string", "stringArray") else "attributeType"
    spec = {key: kind}
    if value is not None:
        spec["value"] = value
    spec.update(extra)
    spec["keyable"]     = keyable
    spec["channel_box"] = channel_box
    return spec


@unittest.skipIf(fbx is None, "the Autodesk FBX SDK is required")
class TestFbxUserAttributes(unittest.TestCase):
    """user defined attributes come back from save_fbx as they went in"""

    def setUp(self):
        self.tmp  = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "attrs.fbx")

    def tearDown(self):
        self.tmp.cleanup()

    def assert_same_specs(self, read, written):
        """the same specs, attrs and keys in the same order"""
        self.assertEqual(read, written)
        self.assertEqual(list(read), list(written))
        for name, spec in written.items():
            self.assertEqual(list(read[name]), list(spec), name)

    def test_every_kind(self):
        attrs = {
            "a_double": _spec("double", 1.5),
            "a_float":  _spec("float", 2.25, keyable=False, channel_box=True),
            "a_angle":  _spec("doubleAngle", 90.0),
            "a_linear": _spec("doubleLinear", -3.5),
            "a_long":   _spec("long", -7),
            "a_int":    {"attributeType": "int", "value": 9},
            "a_short":  _spec("short", 5),
            "a_byte":   _spec("byte", 250),
            "a_char":   _spec("char", -5),
            "a_bool":   _spec("bool", True),
            "a_enum":   _spec("enum", 5, enumName="a=1:b=5:c"),
            "a_string": _spec("string", 'say "hi" \\ 100% & more', keyable=False),
            "a_strings": _spec(
                "stringArray",
                ["a b", "", "caf\u00e9", 'back\\slash "q" 100% & ;', "line\nbreak", "&quot;"],
                keyable=False,
            ),
            "a_empty": _spec("stringArray", [], keyable=False),
            # older rig files store a stringArray as an attributeType
            "a_legacy": {"attributeType": "stringArray", "value": ["87BB9BD4"], "keyable": False},
            "pivot":    _spec("double3", numberOfChildren=3),
            "pivotX":   _spec("double", 0.5, parent="pivot"),
            "pivotY":   _spec("double", 0.0, parent="pivot", channel_box=True),
            "pivotZ":   _spec("double", -0.5, parent="pivot"),
            "tint":     _spec("float3", numberOfChildren=3, keyable=False),
            "tintR":    _spec("float", 0.25, parent="tint", keyable=False),
            "tintG":    _spec("float", 0.5, parent="tint", keyable=False),
            "tintB":    _spec("float", 1.0, parent="tint", keyable=False),
        }
        root = TransformData("root", node_type="joint", user_defined_attributes=attrs)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                HierarchyData([root]).save_fbx(self.path, as_ascii=as_ascii)
                node = HierarchyData.load_fbx(self.path)[0]

                self.assert_same_specs(node.user_defined_attributes, attrs)
                self.assertEqual(node.a_enum,         5)
                self.assertEqual(node.a_strings[3],   'back\\slash "q" 100% & ;')
                self.assertEqual(node.pivot.tolist(), [0.5, 0.0, -0.5])
                self.assertEqual(node.tint.tolist(),  [0.25, 0.5, 1.0])

    def test_only_a_string_value_has_to_be_ascii(self):
        # the docs say so: an enum field, a stringArray item or a nice name
        # that is not plain ASCII comes back as it was
        attrs = {
            "side": _spec("enum", 1, enumName="gauche:café"),
            "tags": _spec("stringArray", ["café", "b"], keyable=False),
            "x":    _spec("double", 1.5, niceName="café"),
        }
        root = TransformData("root", node_type="joint", user_defined_attributes=attrs)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                HierarchyData([root]).save_fbx(self.path, as_ascii=as_ascii)
                node = HierarchyData.load_fbx(self.path)[0]
                self.assert_same_specs(node.user_defined_attributes, attrs)

    def test_a_rig_skeleton(self):
        # the shape rig serializes a Maya skeleton in: namespaced names, a
        # constraint under a joint, which is not exported
        def node(name, parent=None, node_type="joint", **attrs):
            return {
                "name": name,
                "node_type": node_type,
                "uuid": str(uuid.uuid4()).upper(),
                "parent_uuid": parent["uuid"] if parent else None,
                "user_defined_attributes": attrs,
            }

        root = node(
            "BAS_RIG:SK:root",
            sourceFile           = _spec("string", "C:/2k/assets/SK_Hero.mb"),
            lockInfluenceWeights = _spec("bool", False),
            filmboxTypeID        = _spec("short", 5),
            sSubRigRootUuids={
                "attributeType": "stringArray",
                "value":         ["87BB9BD4-4FC6-0772-2ACF-1C9F4095E874"],
                "keyable":       False,
                "channel_box":   False,
            },
            fov=_spec("float", 70.0, keyable=False),
        )
        hip = node("BAS_RIG:SK:hip_L", root, liw=_spec("bool", True))
        con = node(
            "BAS_RIG:hip_L_parentConstraint1",
            hip,
            node_type = "parentConstraint",
            conTargW0 = _spec("double", 1.0),
        )
        knee = node("BAS_RIG:SK:knee_L", hip)
        rig  = HierarchyData.from_dict({x["uuid"]: x for x in (root, hip, con, knee)})

        rig.save_fbx(self.path)
        back = HierarchyData.load_fbx(self.path)

        self.assertEqual(back.name, ["BAS_RIG:SK:root", "BAS_RIG:SK:hip_L", "BAS_RIG:SK:knee_L"])
        for read, written in zip(back, (root, hip, knee)):
            self.assert_same_specs(
                read.user_defined_attributes, written["user_defined_attributes"]
            )

    def test_many_attrs_on_one_node_in_an_ascii_file(self):
        # the kept specs are one long string, and the FBX SDK's ASCII reader
        # lost the whole scene, every node of it, once a long string held a
        # brace, which json always does: 45 doubles on one node did it
        attrs         = {f"a{i}": _spec("double", float(i)) for i in range(60)}
        attrs["pick"] = _spec("enum", 1, enumName=":".join(f"choice{i}" for i in range(200)))
        root          = _joint("root")
        spine         = _joint("spine", root, user_defined_attributes=attrs)
        head          = _joint("head", spine, user_defined_attributes={"a0": _spec("double", 1.0)})
        rig           = HierarchyData([root, spine, head])

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                rig.save_fbx(self.path, as_ascii=as_ascii)
                back = HierarchyData.load_fbx(self.path)

                self.assertEqual(back.name, ["root", "spine", "head"])
                for read, written in zip(back, rig):
                    self.assert_same_specs(
                        read.user_defined_attributes, written.user_defined_attributes
                    )

    def test_a_string_array_with_braces_in_an_ascii_file(self):
        # a long one lost the whole scene the same way
        items = ["{k: v%d}" % i for i in range(100)] * 3 + ["}{", "{"]
        attrs = {"meta": _spec("stringArray", items, keyable=False)}
        root  = _joint("root", user_defined_attributes=attrs)
        child = _joint("child", root)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                HierarchyData([root, child]).save_fbx(self.path, as_ascii=as_ascii)
                back = HierarchyData.load_fbx(self.path)

                self.assertEqual(back.name, ["root", "child"])
                self.assert_same_specs(back[0].user_defined_attributes, attrs)

    def raw_string(self, name):
        """the text the file holds for the first node's string property ``name``"""
        from cgmath.formats._fbx_io import open_fbx

        with open_fbx(self.path, fbx) as scene:
            prop = scene.GetRootNode().GetChild(0).FindProperty(name)
            return str(fbx.FbxPropertyString(prop).Get())

    def test_a_string_an_ascii_file_changes_comes_back_as_it_was(self):
        # an ASCII .fbx reads a typed &quot;, &lf; or &cr; back as the quote or
        # line break it writes that way, and loses the whole scene when a long
        # string holds a brace
        from cgmath.formats.fbx import _PLAIN_TEXT, FbxExporter

        blob = json.dumps({f"key{i}": [i, 'say "hi"'] for i in range(300)}, indent=1)
        attrs = {
            "quot":  _spec("string", 'a &quot; b "c"', keyable=False),
            "lf":    _spec("string", "a &lf; b\nc &cr; d\r", keyable=False),
            "blob":  _spec("string", blob, keyable=False),
            "short": _spec("string", "{k: v}", keyable=False),
        }
        root  = _joint("root", user_defined_attributes=attrs)
        child = _joint("child", root)
        rig   = HierarchyData([root, child])

        rig.save_fbx(self.path)
        self.assertEqual(self.raw_string("blob"), blob)
        self.assert_same_specs(HierarchyData.load_fbx(self.path)[0].user_defined_attributes, attrs)

        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            rig.save_fbx(self.path, as_ascii=True)
        self.assertIn("ASCII file", logs.output[0])
        self.assertTrue(logs.output[0].endswith(": 1"), logs.output[0])

        # Maya reads the escaped text
        escaped = urllib.parse.quote(blob, safe=_PLAIN_TEXT)
        self.assertEqual(self.raw_string("blob"), escaped)
        self.assertEqual(self.raw_string("short"), "{k: v}")

        back = HierarchyData.load_fbx(self.path)
        self.assertEqual(back.name, ["root", "child"])
        self.assert_same_specs(back[0].user_defined_attributes, attrs)

        # escaped for the ASCII file only
        with FbxExporter() as exporter:
            exporter.add_skeleton(rig)
            with self.assertLogs("cgmath.formats.fbx", level="WARNING"):
                exporter.export(self.path, as_ascii=True)
            exporter.export(self.path)
        self.assertEqual(self.raw_string("blob"), blob)

    def test_a_kept_value_changed_since_comes_from_its_property(self):
        # the spec keeps a value its property can't give back, the property decides
        from cgmath.formats.fbx import FbxExporter, _ascii_json, _kept_spec

        quot  = _spec("string", "a &quot; b", keyable=False)
        third = _spec("double", 1 / 3)
        enum  = _spec("enum", 0, enumName="a=1:b=5")
        unset = _spec("string", keyable=False)

        # as a binary file holds it, as an ASCII one does, and changed since
        for spec, data_type, held, read in (
            (quot, fbx.FbxStringDT, "a &quot; b", "a &quot; b"),
            (quot, fbx.FbxStringDT, 'a " b', "a &quot; b"),
            (quot, fbx.FbxStringDT, "changed", "changed"),
            (third, fbx.FbxDoubleDT, 1 / 3, 1 / 3),
            (third, fbx.FbxDoubleDT, 0.333333333333333, 1 / 3),
            (third, fbx.FbxDoubleDT, 0.5, 0.5),
            (enum, fbx.FbxEnumDT, 0, 0),
            (enum, fbx.FbxEnumDT, 1, 5),
            (unset, fbx.FbxStringDT, "", None),
            (unset, fbx.FbxStringDT, "changed", "changed"),
        ):
            kept = _ascii_json({"attr": _kept_spec(spec, True)})

            def populate(node):
                self.add(node, data_type, "attr", held, fields=["a", "b"] if spec is enum else ())
                prop = fbx.FbxProperty.Create(
                    node, fbx.FbxStringDT, FbxExporter._USER_ATTR_SPECS, ""
                )
                prop.Set(kept)

            with self.subTest(spec=spec, held=held):
                self.write_by_hand(populate)
                attrs = HierarchyData.load_fbx(self.path)[0].user_defined_attributes
                self.assertEqual(attrs["attr"].get("value"), read)

    def test_long_strings_in_an_ascii_file_come_back(self):
        # past about 6,100 bytes the SDK's ASCII reader reads a string in
        # pieces, and a quote, line break, tab, semicolon or entity where two
        # meet comes back changed. the json quotes did it: a stringArray of 750
        # uuids came back with 624, and the specs of 300 floats were lost
        from cgmath.formats.fbx import _string_array_text

        uuids           = [str(uuid.UUID(int=i * 7919 + 1)).upper() for i in range(750)]
        blob            = json.dumps({f"key{i}": [i, 'say "hi"'] for i in range(2000)}, indent=1)
        floats          = {f"attr{i}": _spec("float", float(i)) for i in range(300)}
        floats["space"] = _spec("enum", 5, enumName="local=1:world=5")
        attrs = {
            "uuids":  _spec("stringArray", uuids, keyable=False),
            "blob":   _spec("string", blob, keyable=False),
            "ratio":  _spec("float", 0.5),
            "spaced": _spec("string", "a;\tb" * 2000 + " " * 7000 + "&amp;" * 2000, keyable=False),
        }
        root  = _joint("root", user_defined_attributes=attrs)
        spine = _joint("spine", root, user_defined_attributes=floats)
        rig   = HierarchyData([root, spine])

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                rig.save_fbx(self.path, as_ascii=as_ascii)
                back = HierarchyData.load_fbx(self.path)

                self.assertEqual(back.name, ["root", "spine"])
                self.assert_same_specs(back[0].user_defined_attributes, attrs)
                self.assert_same_specs(back[1].user_defined_attributes, floats)

                # Maya reads the same stringArray text from either file
                self.assertEqual(self.raw_string("uuids"), _string_array_text(uuids))

        # no &quot; reaches the file: the long strings with quotes are escaped,
        # the stringArray and the kept specs hold none
        with open(self.path, "rb") as f:
            self.assertNotIn(b"&quot;", f.read())

    def raw_fields(self, name):
        """the field list the file holds for the first node's enum property ``name``"""
        from cgmath.formats._fbx_io import open_fbx

        with open_fbx(self.path, fbx) as scene:
            prop = scene.GetRootNode().GetChild(0).FindProperty(name)
            return [str(prop.GetEnumValue(i)) for i in range(prop.GetEnumCount())]

    def test_a_long_enum_field_list_in_an_ascii_file(self):
        # an ASCII .fbx writes an enum's fields as one string, which its reader
        # takes in pieces as it does a long user string: a brace in a list of
        # 800 fields lost the whole scene
        from cgmath.formats.fbx import _PLAIN_TEXT, FbxExporter

        lists = {
            "braces": [f"opt{{{i:04d}}}" for i in range(800)],
            "quotes": [f'opt "{i:04d}";' for i in range(800)],
            "plain":  [f"option{i:04d}" for i in range(800)],
            # Maya takes a field twice, which the SDK lists once, and a field
            # escaped can be another's text
            "twice": ["twice", "twice", "a{", "a%7B"] + [f"opt{{{i:04d}}}" for i in range(800)],
        }
        attrs = {
            name: _spec("enum", 3, enumName=":".join(fields)) for name, fields in lists.items()
        }
        root  = _joint("root", user_defined_attributes=attrs)
        child = _joint("child", root)
        rig   = HierarchyData([root, child])

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                rig.save_fbx(self.path, as_ascii=as_ascii)
                back = HierarchyData.load_fbx(self.path)

                self.assertEqual(back.name, ["root", "child"])
                self.assert_same_specs(back[0].user_defined_attributes, attrs)

        # Maya reads the escaped fields, and a plain list as it is
        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            rig.save_fbx(self.path, as_ascii=True)
        self.assertIn("enum field lists", logs.output[0])
        self.assertTrue(logs.output[0].endswith(": 3"), logs.output[0])

        listed = {name: list(dict.fromkeys(fields)) for name, fields in lists.items()}
        for name in ("braces", "quotes", "twice"):
            escaped = [urllib.parse.quote(x, safe=_PLAIN_TEXT) for x in listed[name]]
            self.assertEqual(self.raw_fields(name), escaped)
        self.assertEqual(self.raw_fields("plain"), listed["plain"])

        # escaped for the ASCII file only
        with FbxExporter() as exporter:
            exporter.add_skeleton(rig)
            with self.assertLogs("cgmath.formats.fbx", level="WARNING"):
                exporter.export(self.path, as_ascii=True)
            exporter.export(self.path)
        for name in ("braces", "twice"):
            self.assertEqual(self.raw_fields(name), listed[name])

    def test_an_enum_value_that_is_no_field_index_comes_back(self):
        # Maya's default 0 for "local=1:world=5" came back as 1, and 2 for
        # "a=1:b=5:c" as 6: the value was written as a position in the list
        attrs = {
            "space": _spec("enum", 0, enumName="local=1:world=5"),
            "pick":  _spec("enum", 2, enumName="a=1:b=5:c"),
            "far":   _spec("enum", 7, enumName="a=1:b=5:c"),
            "field": _spec("enum", 5, enumName="a=1:b=5:c"),
        }
        root = _joint("root", user_defined_attributes=attrs)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                HierarchyData([root]).save_fbx(self.path, as_ascii=as_ascii)
                node = HierarchyData.load_fbx(self.path)[0]
                self.assert_same_specs(node.user_defined_attributes, attrs)

    def test_numbers_a_file_changes_come_back(self):
        # an ASCII .fbx keeps 15 significant digits of a double, and reads the
        # smallest short or long back as the largest; a binary one too reads
        # -inf and a double under 1e-307 as 0
        attrs = {
            "third":  _spec("double", 1 / 3),
            "ratio":  _spec("float", 2 / 3),
            "angle":  _spec("doubleAngle", 1 / 7),
            "tiny":   _spec("double", 3e-308),
            "low":    _spec("double", -math.inf),
            "short":  _spec("short", -(2**15)),
            "long":   _spec("long", -(2**31)),
            "int":    {"attributeType": "int", "value": -(2**31)},
            "char":   _spec("char", -128),
            "byte":   _spec("byte", 255),
            "pivot":  _spec("double3", numberOfChildren=3),
            "pivotX": _spec("double", 1 / 3, parent="pivot"),
            "pivotY": _spec("double", -0.1 - 0.2, parent="pivot"),
            "pivotZ": _spec("double", 0.5, parent="pivot"),
        }
        root = _joint("root", user_defined_attributes=attrs)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                HierarchyData([root]).save_fbx(self.path, as_ascii=as_ascii)
                node = HierarchyData.load_fbx(self.path)[0]
                self.assert_same_specs(node.user_defined_attributes, attrs)

    def test_a_string_with_no_value_comes_back(self):
        # Maya's getAttr gives None for a string or stringArray it has not
        # set; they were skipped as having no fbx type. Maya reads an empty
        # string
        attrs = {
            "note":    _spec("string", keyable=False),
            "label":   {"dataType": "string", "value": None, "keyable": False, "channel_box": False},
            "tags":    _spec("stringArray", keyable=False),
            "targets": {"dataType": "stringArray", "value": None, "keyable": False},
        }
        root = _joint("root", user_defined_attributes=attrs)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                with _no_fbx_warnings(self):
                    HierarchyData([root]).save_fbx(self.path, as_ascii=as_ascii)
                for name in attrs:
                    self.assertEqual(self.raw_string(name), "")

                node = HierarchyData.load_fbx(self.path)[0]
                self.assert_same_specs(node.user_defined_attributes, attrs)

    def test_other_compounds_come_back_as_their_children(self):
        # Maya's own FBX plugin writes the children of any other compound on
        # their own, and its importer reads them back so; the compound is
        # reported as skipped
        attrs = {
            "limits":    _spec("compound", numberOfChildren=2),
            "limitsOn":  _spec("bool", True, parent="limits"),
            "limitsMax": _spec("doubleAngle", 90.0, parent="limits"),
            "steps":     _spec("long3", numberOfChildren=3),
            "stepsX":    _spec("long", 1, parent="steps"),
            "stepsY":    _spec("long", 2, parent="steps"),
            "stepsZ":    _spec("long", 3, parent="steps"),
        }
        root = TransformData("root", node_type="joint", user_defined_attributes=attrs)

        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            HierarchyData([root]).save_fbx(self.path)
        self.assertIn("1 compound, 1 long3", logs.output[0])

        loose = {
            name: {key: value for key, value in spec.items() if key != "parent"}
            for name, spec in attrs.items()
            if "parent" in spec
        }
        node = HierarchyData.load_fbx(self.path)[0]
        self.assert_same_specs(node.user_defined_attributes, loose)

    def write_by_hand(self, populate, as_ascii=False):
        """an fbx holding one joint, ``root``, whose properties populate(node) adds"""
        from cgmath.formats._fbx_io import new_manager

        manager = new_manager(fbx)
        try:
            scene = fbx.FbxScene.Create(manager, "")
            node  = fbx.FbxNode.Create(manager, "root")
            node.SetNodeAttribute(fbx.FbxSkeleton.Create(manager, ""))
            scene.GetRootNode().AddChild(node)
            populate(node)

            file_format = -1
            if as_ascii:
                registry = manager.GetIOPluginRegistry()
                file_format = next(
                    i
                    for i in range(registry.GetWriterFormatCount())
                    if registry.WriterIsFBX(i)
                    and "ascii" in registry.GetWriterFormatDescription(i)
                )

            exporter = fbx.FbxExporter.Create(manager, "")
            self.assertTrue(exporter.Initialize(self.path, file_format, manager.GetIOSettings()))
            self.assertTrue(exporter.Export(scene))
            exporter.Destroy()
        finally:
            manager.Destroy()

    @staticmethod
    def add(node, data_type, name, value, animatable=True, fields=()):
        """a user defined property, flagged the way Maya's own FBX plugin does"""
        prop = fbx.FbxProperty.Create(node, data_type, name, name)
        for field in fields:
            prop.AddEnumValue(field)
        prop.Set(value)
        prop.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eUserDefined, True)
        prop.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, animatable)
        return prop

    def test_a_file_from_maya_reads_as_maya_reads_it(self):
        # what Maya 2025's FBX plugin writes, and the attrs its importer makes
        def populate(node):
            self.add(node, fbx.FbxDoubleDT, "a_double", 1.5)
            self.add(node, fbx.FbxIntDT,    "a_long",   -7)
            self.add(node, fbx.FbxShortDT,  "a_short",  5)
            self.add(node, fbx.FbxUCharDT,  "a_byte",   250)
            self.add(node, fbx.FbxCharDT,   "a_char",   -5)
            self.add(node, fbx.FbxBoolDT,   "a_bool",   True)
            self.add(node, fbx.FbxEnumDT, "a_enum", 5, fields=["0", "a", "2", "3", "4", "b"])
            self.add(node, fbx.FbxStringDT, "a_string", "hello", animatable=False)
            self.add(node, fbx.FbxDouble3DT, "a_vec", fbx.FbxDouble3(1.0, 2.0, 3.0))
            self.add(node, fbx.FbxColor3DT, "a_col", fbx.FbxDouble3(0.25, 0.5, 1.0))

        self.write_by_hand(populate)
        node = HierarchyData.load_fbx(self.path)[0]

        expected = {
            "a_double": _spec("double", 1.5),
            "a_long":   _spec("long", -7),
            "a_short":  _spec("short", 5),
            "a_byte":   _spec("byte", 250),
            "a_char":   _spec("char", -5),
            "a_bool":   _spec("bool", True),
            "a_enum":   _spec("enum", 5, enumName="0:a:2:3:4:b"),
            "a_string": _spec("string", "hello", keyable=False),
            "a_vec":    _spec("double3", numberOfChildren=3),
            "a_vec0":   _spec("double", 1.0, parent="a_vec"),
            "a_vec1":   _spec("double", 2.0, parent="a_vec"),
            "a_vec2":   _spec("double", 3.0, parent="a_vec"),
            "a_col":    _spec("float3", numberOfChildren=3),
            "a_colR":   _spec("float", 0.25, parent="a_col"),
            "a_colG":   _spec("float", 0.5, parent="a_col"),
            "a_colB":   _spec("float", 1.0, parent="a_col"),
        }
        self.assert_same_specs(node.user_defined_attributes, expected)

    def test_a_file_from_maya_with_a_non_ascii_string(self):
        # Maya writes a string as UTF-8, which load_fbx reads; saving it again
        # raises ValueError naming it, the SDK taking plain ASCII only
        label = "hanche gauche \u00e9"

        def populate(node):
            text = fbx.FbxString(label.encode("utf-8"))
            self.add(node, fbx.FbxStringDT, "label", text, animatable=False)

        self.write_by_hand(populate)
        rig = HierarchyData.load_fbx(self.path)
        self.assertEqual(rig[0].label, label)

        again = os.path.join(self.tmp.name, "again.fbx")
        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                with self.assertRaisesRegex(ValueError, r"root\.label .* not plain ASCII"):
                    rig.save_fbx(again, as_ascii=as_ascii)
                self.assertFalse(os.path.exists(again))

    def test_a_string_that_is_not_utf8_is_left_out(self):
        # an older exporter's Latin-1 text, which the binding can't read: it
        # failed the whole load with UnicodeDecodeError
        from cgmath.formats.fbx import FbxExporter, _ascii_json

        for kept in (None, _ascii_json({"label": _spec("string", keyable=False)})):

            def populate(node):
                text = fbx.FbxString(b"caf\xe9")
                self.add(node, fbx.FbxStringDT, "label", text, animatable=False)
                self.add(node, fbx.FbxDoubleDT, "a_double", 1.5)
                if kept is not None:
                    prop = fbx.FbxProperty.Create(
                        node, fbx.FbxStringDT, FbxExporter._USER_ATTR_SPECS, ""
                    )
                    prop.Set(kept)

            with self.subTest(kept=kept is not None):
                self.write_by_hand(populate)
                node = HierarchyData.load_fbx(self.path)[0]
                self.assertEqual(node.user_defined_attributes, {"a_double": _spec("double", 1.5)})

    def test_a_name_or_enum_field_that_is_not_utf8_is_left_out(self):
        # an older exporter's Latin-1 text in a property's name or an enum
        # field failed the whole load, and the clip and skin loads with it,
        # where in a string it was left out
        from cgmath.geometry import skin_weights
        from cgmath.hierarchy import ClipData

        def populate(node):
            self.add(node, fbx.FbxDoubleDT, "cafX", 0.5)
            self.add(node, fbx.FbxEnumDT, "side", 0, fields=["gaucheX", "droite"])
            self.add(node, fbx.FbxDoubleDT, "a_double", 1.5)

        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                self.write_by_hand(populate, as_ascii=as_ascii)

                # the binding writes ASCII only: the Latin-1 goes in after,
                # the same length a binary file counts
                with open(self.path, "rb") as f:
                    data = f.read()
                self.assertIn(b"cafX", data)
                self.assertIn(b"gaucheX", data)
                data = data.replace(b"cafX", b"caf\xe9").replace(b"gaucheX", b"gauche\xe9")
                with open(self.path, "wb") as f:
                    f.write(data)

                expected = {"a_double": _spec("double", 1.5)}
                rig      = HierarchyData.load_fbx(self.path)
                clip     = ClipData.load_fbx(self.path)

                self.assertEqual(rig.name, ["root"])
                self.assertEqual(rig[0].user_defined_attributes, expected)
                self.assertEqual(clip[0].user_defined_attributes, expected)
                self.assertEqual(len(skin_weights.load_fbx(self.path)), 0)

    def test_an_enum_field_from_another_tool(self):
        # another exporter's enum field can hold "=" or ":", which Maya's
        # addAttr can't make. "size=small" loaded, then failed save_fbx and any
        # set with a bare int() ValueError; "a:b" read back as fields a and b,
        # and "LOD=0" reads as field LOD at index 0 in an enumName, as does
        # "k==2" as field k at index 2, which loaded with no warning. "a==" and
        # "==2" are whole fields in Maya and come back
        def populate(node):
            self.add(node, fbx.FbxEnumDT, "mode",  1, fields=["size=small", "size=large"])
            self.add(node, fbx.FbxEnumDT, "split", 1, fields=["a:b", "c"])
            self.add(node, fbx.FbxEnumDT, "lod",   1, fields=["LOD=0", "LOD=1"])
            self.add(node, fbx.FbxEnumDT, "runs",  1, fields=["k==2", "b"])
            self.add(node, fbx.FbxEnumDT, "ends",  1, fields=["a==", "==2"])
            self.add(node, fbx.FbxDoubleDT, "a_double", 1.5)

        expected = {
            "mode":     _spec("enum", 1, enumName="size=small:size=large"),
            "ends":     _spec("enum", 1, enumName="a==:==2"),
            "a_double": _spec("double", 1.5),
        }
        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                self.write_by_hand(populate, as_ascii=as_ascii)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    rig = HierarchyData.load_fbx(self.path)
                left = [str(w.message) for w in caught if "user defined enum" in str(w.message)]

                self.assert_same_specs(rig[0].user_defined_attributes, expected)
                self.assertEqual(len(left), 3, left)
                self.assertIn("root.split left out: its field 'a:b'", left[0])
                self.assertIn("root.lod left out: its field 'LOD=0'", left[1])
                self.assertIn("root.runs left out: its field 'k==2'", left[2])

                rig[0].mode = "size=small"
                self.assertEqual(rig[0].mode, 0)
                rig[0].mode = 1

                # written back with the fields the other tool wrote
                rig.save_fbx(self.path, as_ascii=as_ascii)
                self.assertEqual(self.raw_fields("mode"), ["size=small", "size=large"])
                self.assertEqual(self.raw_fields("ends"), ["a==", "==2"])
                node = HierarchyData.load_fbx(self.path)[0]
                self.assert_same_specs(node.user_defined_attributes, expected)

    def test_a_kept_spec_that_does_not_fit_is_passed_over(self):
        from cgmath.formats.fbx import FbxExporter

        for kept in ('{"a_double": {"dataType": "string"}}', "not json", "[1, 2]"):

            def populate(node):
                self.add(node, fbx.FbxDoubleDT, "a_double", 1.5)
                prop = fbx.FbxProperty.Create(
                    node, fbx.FbxStringDT, FbxExporter._USER_ATTR_SPECS, ""
                )
                prop.Set(kept)

            with self.subTest(kept=kept):
                self.write_by_hand(populate)
                node = HierarchyData.load_fbx(self.path)[0]
                self.assertEqual(node.user_defined_attributes, {"a_double": _spec("double", 1.5)})


if __name__ == "__main__":
    unittest.main()
