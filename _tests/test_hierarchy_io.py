import copy
import os
import pickle
import tempfile
import unittest
from dataclasses import fields

import numpy as np
from cgmath import hierarchy
from cgmath.geometry import Data
from cgmath.hierarchy import HierarchyData, TransformData, TransformList


CHANNELS = ("_scale", "_rotate", "_translate", "_rotate_axis", "_joint_orient")


def rig():
    """a three node rig parented by uuid, with every kind of field set"""
    h = HierarchyData()
    root = TransformData(
        "ns:root",
        node_type="joint",
        translate=[1.0, 2.0, 3.0],
        joint_orient=[0.0, 90.0, 0.0],
        rotate_order=3,
        segment_scale_compensate=True,
        radius=2.5,
        draw_style=2,
    )
    hip = TransformData(
        "hip",
        node_type   = "joint",
        rotate      = [10.5, -3.0, 7.0],
        scale       = [1.0, 2.0, 1.0],
        rotate_axis = [1.0, 2.0, 3.0],
        visibility  = False,
    )
    loc = TransformData("loc", node_type="locator", translate=[0.0, 5.0, 0.0])
    for node in (root, hip, loc):
        h.append(node)
    hip.set_parent("ns:root", world_space=False)
    loc.set_parent("hip", world_space=False)

    hip.add_user_attribute("heroHeight", 2.5)
    hip.add_user_attribute("tags", ["a", "b"])
    hip.add_user_attribute("mode", "b", attribute_type="enum", enum_names="a=1:b=5:c")
    hip.add_user_attribute("points", [[1, 2, 3], [4, 5, 6]], attribute_type="vectorArray")
    hip.add_user_attribute("empty", [], attribute_type="doubleArray")
    return h


class TestHierarchyFiles(unittest.TestCase):
    """pkl, npz and json all give back exactly what was saved"""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def loaded(self, obj, name):
        """``obj`` back from every file format, pickle and deepcopy"""
        out = {
            "pickle":   pickle.loads(pickle.dumps(obj)),
            "deepcopy": copy.deepcopy(obj),
        }
        for mode in ("json", "npz", "pkl"):
            path      = obj.save(os.path.join(self.tmp.name, f"{name}.{mode}"))
            out[mode] = type(obj).load(path)
        return out

    def assertSameValue(self, a, b, where):
        """equal, and of the same type, dtype and shape all the way down"""
        if isinstance(a, np.ndarray):
            self.assertIsInstance(b, np.ndarray, where)
            self.assertEqual(a.dtype, b.dtype, where)
            self.assertEqual(a.shape, b.shape, where)
            self.assertTrue(np.array_equal(a, b), where)
        elif isinstance(a, dict):
            self.assertIsInstance(b, dict, where)
            self.assertEqual(a.keys(), b.keys(), where)
            for key in a:
                self.assertSameValue(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, (list, tuple)):
            self.assertIs(type(a), type(b), where)
            self.assertEqual(len(a), len(b), where)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assertSameValue(x, y, f"{where}[{i}]")
        else:
            self.assertIs(type(a), type(b), where)
            self.assertEqual(a, b, where)

    def assertSameNode(self, a, b, where):
        self.assertIs(type(a), type(b), where)
        for field in fields(a):
            self.assertSameValue(getattr(a, field.name), getattr(b, field.name), f"{where}.{field.name}")

        # TransformData's own == only compares uuids
        self.assertTrue(Data.__eq__(a, b), where)

    def test_a_node_round_trips(self):
        node = rig()["hip"]
        for how, loaded in self.loaded(node, "node").items():
            with self.subTest(how):
                self.assertSameNode(node, loaded, how)
                self.assertIsNone(loaded._hierarchy)

    def test_a_rig_round_trips_with_its_parenting(self):
        source = rig()
        for how, loaded in self.loaded(source, "rig").items():
            with self.subTest(how):
                self.assertIs(type(loaded), HierarchyData)
                self.assertEqual(loaded.name, source.name)
                for a, b in zip(source, loaded):
                    self.assertSameNode(a, b, f"{how}:{a.name}")
                    self.assertIs(b._hierarchy, loaded)

                self.assertIsNone(loaded["ns:root"].get_parent())
                self.assertIs(loaded["hip"].get_parent(), loaded["ns:root"])
                self.assertIs(loaded["loc"].get_parent(), loaded["hip"])
                self.assertTrue(np.allclose(loaded.world_matrix, source.world_matrix))
                self.assertEqual(loaded["hip"].heroHeight, 2.5)

    def test_the_module_loader_reads_every_format(self):
        source = rig()
        for mode in ("json", "npz", "pkl"):
            with self.subTest(mode):
                path   = source.save(os.path.join(self.tmp.name, f"rig.{mode}"))
                loaded = hierarchy.load(path)

                self.assertIs(type(loaded), HierarchyData)
                self.assertEqual(loaded, source)

    def test_a_file_names_its_class(self):
        tree = rig().to_dict()

        self.assertEqual(tree["__class__"], "cgmath.hierarchy.hierarchy.HierarchyData")
        self.assertEqual(len(tree["__items__"]), 3)
        self.assertEqual(
            tree["__items__"][0]["__class__"], "cgmath.hierarchy.hierarchy.TransformData"
        )

    def test_a_view_saves_as_its_copy(self):
        # a view owns nothing, so it saves as the hierarchy copy() gives: its
        # own nodes, a node whose parent it left out made a root
        source = rig()
        view   = source[1:]
        for how, loaded in self.loaded(view, "view").items():
            with self.subTest(how):
                self.assertIs(type(loaded), HierarchyData)
                self.assertEqual(loaded.name, ["hip", "loc"])
                self.assertEqual(loaded.uuid, view.uuid)
                self.assertIsNone(loaded["hip"].get_parent())
                self.assertIsNone(loaded["hip"].parent_node)
                self.assertIs(loaded["loc"].get_parent(), loaded["hip"])
                self.assertTrue(all(node._hierarchy is loaded for node in loaded))

        # the view and its rig are left as they were
        self.assertIs(type(view), TransformList)
        self.assertIs(source["hip"].get_parent(), source["ns:root"])
        self.assertTrue(all(node._hierarchy is source for node in source))

    def test_an_empty_rig_round_trips(self):
        for how, loaded in self.loaded(HierarchyData(), "empty").items():
            with self.subTest(how):
                self.assertIs(type(loaded), HierarchyData)
                self.assertEqual(len(loaded), 0)

    def test_the_asset_readers_node_dicts_still_read(self):
        # load_fbx and load_glb hand from_dict a dict of node dicts keyed by
        # uuid, written with the constructor's names
        root_id = hierarchy.generate_uuid()
        hip_id  = hierarchy.generate_uuid()
        nodes = {
            root_id: {"name": "root", "uuid": root_id, "rotate_order": 2},
            hip_id: {
                "name": "hip",
                "uuid": hip_id,
                "parent_node": root_id,
                "translate": np.array([0, 10, 0]),
                "segment_scale_compensate": True,
            },
        }
        for cls in (HierarchyData, TransformList):
            with self.subTest(cls.__name__):
                loaded = cls.from_dict(nodes)

                self.assertIs(type(loaded), cls)
                self.assertEqual(loaded.name, ["root", "hip"])
                self.assertEqual(loaded["root"].rotate_order, 2)
                self.assertIs(loaded["hip"].segment_scale_compensate, True)
                self.assertEqual(loaded["hip"]._translate.dtype, np.float64)
                self.assertTrue(np.allclose(loaded["hip"].translate, [0.0, 10.0, 0.0]))

        loaded = HierarchyData.from_dict(nodes)
        self.assertTrue(np.allclose(loaded["hip"].world_matrix[3, :3], [0.0, 10.0, 0.0]))


class TestTransformChannels(unittest.TestCase):
    """the channels are float64 (3,) arrays, each node its own"""

    def test_a_file_of_whole_numbers_still_loads_floats(self):
        # a json channel written as [0, 5, 0] used to load as an int64 array,
        # and the in place write that followed truncated 2.75 to 2
        tree               = TransformData("a").to_dict()
        tree["_translate"] = [0, 5, 0]
        tree["_scale"]     = [1, 2, 1]

        node               = TransformData.from_dict(tree)
        node.translate_y   = 2.75

        self.assertEqual(node._translate.dtype, np.float64)
        self.assertEqual(node._scale.dtype,     np.float64)
        self.assertEqual(node.translate_y,      2.75)

    def test_a_channel_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            TransformData("a", translate=[1.0, 2.0])
        with self.assertRaises(ValueError):
            TransformData("a", scale=[[1.0, 1.0, 1.0]])

        node = TransformData("a")
        with self.assertRaises(ValueError):
            node._rotate = np.zeros(4)

        tree            = node.to_dict()
        tree["_rotate"] = [1.0, 2.0]
        with self.assertRaises(ValueError):
            TransformData.from_dict(tree)

    def test_the_constructor_copies_the_channels_it_is_given(self):
        # a channel write is in place, so two nodes built from one array
        # used to share it and a write to one moved the other
        pose = np.array([1.0, 2.0, 3.0])
        a    = TransformData("a", translate=pose, rotate=pose, scale=pose)
        b    = TransformData("b", translate=pose, rotate=pose, scale=pose)

        a.translate = [9.0, 9.0, 9.0]
        a.rotate_x  = 45.0
        a.scale_z   = 4.0

        self.assertTrue(np.allclose(pose, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(b.translate, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(b.rotate, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(b.scale, [1.0, 2.0, 3.0]))

    def test_a_loaded_node_owns_its_defaults(self):
        a = TransformData.from_dict({"name": "a"})
        b = TransformData.from_dict({"name": "b"})

        for channel in CHANNELS:
            self.assertIsNot(getattr(a, channel), getattr(b, channel))
            self.assertIsNot(getattr(a, channel), getattr(TransformData, channel))

        self.assertEqual(a.user_defined_attributes, {})
        self.assertIsNot(a.user_defined_attributes, b.user_defined_attributes)

        a.add_user_attribute("heroHeight", 2.5)
        self.assertEqual(b.user_defined_attributes, {})
        self.assertNotIn("user_defined_attributes", TransformData("c").to_dict())


if __name__ == "__main__":
    unittest.main()
