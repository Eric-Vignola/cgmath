"""
FbxExporter unit tests, skeleton export.

These need the Autodesk FBX SDK (the `fbx` module), they are skipped without it.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid

from cgmath.hierarchy import HierarchyData, TransformData

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
        parent_node = parent.uuid if parent is not None else None,
        **kwargs,
    )


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

        with self.assertNoLogs("cgmath.formats.fbx", level="WARNING"):
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
            "a_uuids":  {"dataType": "stringArray", "value": ["a", "b"], "keyable": False},
            "a_maps":   {"dataType": "doubleArray", "value": [1.0, 2.0], "keyable": False},
            "a_double": {"attributeType": "double", "value": 1.5, "keyable": True},
        }
        root  = _joint("root", user_defined_attributes=attrs)
        child = _joint("child", root, user_defined_attributes={"a_uuids": attrs["a_uuids"]})

        # the export completes, the skipped attrs are reported once
        with self.assertLogs("cgmath.formats.fbx", level="WARNING") as logs:
            self._export(HierarchyData([root, child]))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("2 stringArray", logs.output[0])
        self.assertIn("1 doubleArray", logs.output[0])

        result = self._read()
        self.assertEqual(result["|root"], {"a_double": "Number"})
        self.assertEqual(result["|root|child"], {})


if __name__ == "__main__":
    unittest.main()
