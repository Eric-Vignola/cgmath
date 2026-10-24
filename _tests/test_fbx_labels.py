"""
What a rig and a clip record of the axes and unit their values are in,
``axis_system`` and ``unit``, and the label ``save_fbx`` writes from them: a
load then a save lands where the values say, in cgmath's own loaders, in the
sdk's conversion to Maya's Y-up and cm, and in Maya's FBXImport.

The fbx tests need the Autodesk FBX SDK, the Maya ones mayapy with its
fbxmaya plug-in; both are skipped without them.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import os
import pickle
import tempfile
import unittest

import numpy as np

from cgmath.hierarchy import FBX, ClipData, HierarchyData, TransformData, TransformList
from cgmath.hierarchy import generate_uuid

UNREAL = "z up, -y front, right handed"

# the records other than hand-built data's, ("maya_y_up", 1.0), the record
# refusals are tested with: another unit, other axes, and both, so a check
# that compares one half of a record alone lets one of them through
_OTHERS = ((UNREAL, 100.0), ("maya_y_up", 100.0), (UNREAL, 1.0))


def _chain() -> HierarchyData:
    """a joint root over a joint, beside a group root over a locator"""
    ids = [generate_uuid() for _ in range(4)]
    return HierarchyData([
        TransformData("hip", node_type="joint", uuid=ids[0], translate=(1.0, 2.0, 3.0),
                      rotate=(10.0, 0.0, 0.0), joint_orient=(0.0, 0.0, 90.0)),
        TransformData("knee", node_type="joint", uuid=ids[1], parent_uuid=ids[0],
                      translate=(40.0, 0.0, 0.0), rotate=(0.0, 0.0, -30.0)),
        TransformData("grp", uuid=ids[2], translate=(10.0, 20.0, 30.0), rotate=(0.0, 0.0, 45.0)),
        TransformData("loc", node_type="locator", uuid=ids[3], parent_uuid=ids[2],
                      translate=(-20.0, 0.0, 15.0), rotate=(15.0, 25.0, 35.0)),
    ])


def _worlds(rig) -> dict:
    return {node.name: np.asarray(node.world_matrix) for node in rig}


def _records(data) -> tuple:
    return data.axis_system, data.unit


def _joined(rig, method: str, nodes):
    """``rig`` once ``nodes`` joined it by ``method``, append or extend"""
    getattr(rig, method)(nodes)
    return rig


def _names_both(test, caught, record) -> None:
    """the ValueError ``caught`` names ``record`` and hand-built data's"""
    test.assertIn(f"axis_system {record[0]!r}, unit {record[1]!r}", str(caught.exception))
    test.assertIn("axis_system 'maya_y_up', unit 1.0", str(caught.exception))


class TestTheRecord(unittest.TestCase):
    """the record itself: its defaults, its spelling, how it is saved and
    which rigs, clips and views carry it"""

    def test_hand_built_data_is_in_maya_y_up_and_cm(self):
        for data in (HierarchyData(), _chain(), ClipData(), ClipData(_chain(), frames=3)):
            with self.subTest(type(data).__name__):
                self.assertEqual((data.axis_system, data.unit), ("maya_y_up", 1.0))
        view = TransformList([TransformData("loose")])
        self.assertEqual((view.axis_system, view.unit), ("maya_y_up", 1.0))

    def test_the_setters_check_it_and_spell_it_one_way(self):
        rig = _chain()
        for given, held in (
            (UNREAL, UNREAL),
            ("-x up, z front, left handed", "-x up, z front, left handed"),
            ("y up, z front, right handed", "maya_y_up"),  # Maya's Y-up in words
            ("y up, -z front, right handed", "y up, -z front, right handed"),
            ("maya_y_up", "maya_y_up"),
        ):
            with self.subTest(given):
                rig.axis_system = given
                self.assertEqual(rig.axis_system, held)
        for given, error in (
            ("y up, y front, right handed", ValueError),  # up and front on one axis
            ("+z up, -y front, right handed", ValueError),  # no plus sign
            ("Z up, -y front, right handed", ValueError),
            ("z up, -y front", ValueError),
            ("maya_z_up", ValueError),
            (None, TypeError),
            (2, TypeError),
        ):
            with self.subTest(given=given):
                rig.axis_system = UNREAL
                with self.assertRaises(error):
                    rig.axis_system = given
                self.assertEqual(rig.axis_system, UNREAL)

        rig.unit = 100
        self.assertEqual(rig.unit, 100.0)
        self.assertIs(type(rig.unit), float)
        rig.unit = np.float32(2.5)
        self.assertEqual(rig.unit, 2.5)
        self.assertIs(type(rig.unit), float)
        for given, error in (
            (0, ValueError),
            (-1.0, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (True, TypeError),
            ("100", TypeError),
            (None, TypeError),
        ):
            with self.subTest(unit=given):
                rig.unit = 100.0
                with self.assertRaises(error):
                    rig.unit = given
                self.assertEqual(rig.unit, 100.0)

    def test_it_is_saved_under_its_public_names(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rig = _chain()
        rig.axis_system, rig.unit = UNREAL, 2.54
        clip = ClipData(rig, frames=3)
        for data in (rig, clip):
            tree = data.to_dict()
            self.assertEqual((tree["axis_system"], tree["unit"]), (UNREAL, 2.54))
            self.assertNotIn("_axis_system", tree)
            self.assertNotIn("_unit", tree)
            for ext in ("npz", "json", "pkl"):
                with self.subTest(type(data).__name__, ext=ext):
                    path = data.save(os.path.join(tmp.name, f"{type(data).__name__}.{ext}"))
                    back = type(data).load(path)
                    self.assertEqual((back.axis_system, back.unit), (UNREAL, 2.54))
                    self.assertIs(type(back.unit), float)
                    self.assertEqual(back, data)
            back = pickle.loads(pickle.dumps(data))
            self.assertEqual((back.axis_system, back.unit), (UNREAL, 2.54))

    def test_a_file_saved_before_there_was_one_reads_as_maya_y_up_and_cm(self):
        rig = _chain()
        rig.axis_system, rig.unit = UNREAL, 100.0
        for data in (rig, ClipData(rig, frames=2)):
            tree = data.to_dict()
            del tree["axis_system"], tree["unit"]
            back = type(data).from_dict(tree)
            self.assertEqual((back.axis_system, back.unit), ("maya_y_up", 1.0))

        # the files test_hierarchy keeps from before parent_uuid
        assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assets")
        for name, cls in (
            ("test_hierarchy.legacy_parent_node.json", HierarchyData),
            ("test_hierarchy.legacy_parent_node.npz", HierarchyData),
            ("test_hierarchy.legacy_parent_node_clip.json", ClipData),
        ):
            with self.subTest(name):
                back = cls.load(os.path.join(assets, name))
                self.assertEqual((back.axis_system, back.unit), ("maya_y_up", 1.0))

    def test_a_file_whose_record_the_setters_refuse_raises(self):
        for key, value, error in (
            ("unit", -1.0, ValueError),
            ("unit", "cm", TypeError),
            ("axis_system", "up", ValueError),
            ("axis_system", 3, TypeError),
        ):
            with self.subTest(key=key, value=value):
                tree      = _chain().to_dict()
                tree[key] = value
                with self.assertRaises(error):
                    HierarchyData.from_dict(tree)

    def test_what_is_built_from_a_rig_or_a_clip_takes_its_record(self):
        rig = _chain()
        rig.axis_system, rig.unit = UNREAL, 100.0
        clip = ClipData(rig, frames=4)
        built = {
            "rig.copy()":            lambda: rig.copy(),
            "copy.copy(rig)":        lambda: copy.copy(rig),
            "copy.deepcopy(rig)":    lambda: copy.deepcopy(rig),
            "a view's copy()":       lambda: rig.match("knee", "loc").copy(),
            "a view's tree":         lambda: HierarchyData.from_dict(rig.match("knee").to_dict()),
            "HierarchyData(view)":   lambda: HierarchyData(rig.copy().match("knee")),
            "HierarchyData(rig)":    lambda: HierarchyData(rig.copy()),
            "ClipData(rig)":         lambda: ClipData(rig, frames=2),
            "ClipData(view)":        lambda: ClipData(rig.match("hip", "knee"), frames=2),
            "ClipData.from_poses":   lambda: ClipData.from_poses([rig, rig]),
            "clip.copy()":           lambda: clip.copy(),
            "copy.deepcopy(clip)":   lambda: copy.deepcopy(clip),
            "clip.frames[1:3]":      lambda: clip.frames[1:3],
            "rig.get_delta(rig)":    lambda: rig.get_delta(rig.copy()),
            "rig.add_delta(delta)":  lambda: rig.copy().add_delta(rig.get_delta(rig.copy())),
            "clip.get_delta(rig)":   lambda: clip.get_delta(rig),
            "rig.get_delta(frames)": lambda: rig.get_delta(clip.frames),
            "clip.add_delta(delta)": lambda: clip.add_delta(rig.get_delta(rig.copy())),
            "rig.add_delta(frames)": lambda: rig.add_delta(clip.get_delta(rig).frames),
        }
        for label, build in built.items():
            with self.subTest(label):
                data = build()
                self.assertIsInstance(data, HierarchyData)
                self.assertEqual((data.axis_system, data.unit), (UNREAL, 100.0))

        # a clip's copy and cut hold its record whoever owns a node it
        # lists, as its frames hold the values in it; an empty clip has
        # only its record to pass on
        listed = _chain()
        listed.axis_system, listed.unit = UNREAL, 100.0
        clip[1]     = listed[1]  # listed, still owned by the other rig
        listed.unit = 2.54
        empty       = ClipData()
        empty.axis_system, empty.unit = UNREAL, 100.0
        for label, build in (
            ("a listed node's clip.copy()", clip.copy),
            ("a listed node's clip.frames[0:2]", lambda: clip.frames[0:2]),
            ("an empty clip's copy()", empty.copy),
            ("an empty clip's deepcopy", lambda: copy.deepcopy(empty)),
        ):
            with self.subTest(label):
                self.assertEqual(_records(build()), (UNREAL, 100.0))
        self.assertEqual(empty.copy(), empty)

    def test_a_view_reads_its_rigs_record_and_takes_no_write(self):
        rig = _chain()
        rig.axis_system, rig.unit = UNREAL, 100.0
        clip = ClipData(rig, frames=2)
        for view in (
            rig.match("knee"),
            rig[1:3],
            clip.match("hip"),
            TransformList([rig["loc"], TransformData("loose")]),  # a node no rig owns has none
        ):
            with self.subTest(view.name):
                self.assertIs(type(view), TransformList)
                self.assertEqual((view.axis_system, view.unit), (UNREAL, 100.0))
                with self.assertRaises(AttributeError):
                    view.axis_system = "maya_y_up"
                with self.assertRaises(AttributeError):
                    view.unit = 1.0

    def test_a_loads_view_reads_the_rig_that_owns_its_nodes(self):
        """the view TransformList.load_fbx gives, nodes no rig owns recording
        what the load produced, and every view taken from it: once a rig
        owns their nodes, they record the rig's, and its relabels"""
        from cgmath.hierarchy.hierarchy import _recorded

        def loaded():
            return _recorded(TransformList([node.copy() for node in _chain()]), (UNREAL, 100.0))

        view = loaded()
        rig  = HierarchyData(view)
        self.assertEqual(_records(rig), (UNREAL, 100.0))
        rig.axis_system, rig.unit = "maya_y_up", 2.54
        for label, taken in (
            ("the view", view),
            ("a slice", view[0:]),
            ("an index", view[[0, 1, 2, 3]]),
            ("a mask", view[np.ones(4, dtype=bool)]),
            ("match", view.match("*", exact=False)),
            ("get_roots", view.get_roots()),
            ("get_children", view.get_children()),
            ("get_branch", view.get_branch()),
            ("TransformList", TransformList(view)),
        ):
            with self.subTest(label):
                self.assertTrue(all(node._hierarchy is rig for node in taken))
                delta = rig.match(*taken.name).get_delta(taken)  # the same nodes
                for data in (taken, taken.copy(), ClipData(taken, frames=2), delta):
                    self.assertEqual(_records(data), ("maya_y_up", 2.54))

        # a node no rig owns is in the load's record still: with the rig's
        # another, the view holds two; the view left empty holds the load's
        view = loaded()
        rig  = HierarchyData(view[0:2])
        self.assertEqual(_records(view), (UNREAL, 100.0))
        rig.unit = 2.54
        for label, call in (("unit", lambda: view.unit), ("copy()", view.copy), ("ClipData", lambda: ClipData(view))):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn("unit 2.54", str(caught.exception))
                self.assertIn("unit 100.0", str(caught.exception))
        rig.unit = 100.0
        self.assertEqual(_records(view), (UNREAL, 100.0))
        self.assertEqual(_records(view.match("none")), (UNREAL, 100.0))

    def test_a_view_over_rigs_that_disagree_has_no_record(self):
        rig, other = _chain(), _chain()
        rig.unit = 100.0
        mixed    = TransformList([rig["hip"], other["grp"]])
        for label, call in (
            ("axis_system", lambda: mixed.axis_system),
            ("unit", lambda: mixed.unit),
            ("copy()", mixed.copy),
            ("HierarchyData(view)", lambda: HierarchyData(mixed)),
            ("ClipData(view)", lambda: ClipData(mixed, frames=2)),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn("unit 100.0", str(caught.exception))
        self.assertEqual(rig.name, ["hip", "knee", "grp", "loc"])  # nothing moved

        other.unit = 100.0
        self.assertEqual((mixed.axis_system, mixed.unit), ("maya_y_up", 100.0))

    def test_nodes_of_a_rig_recording_another_join_no_rig(self):
        def pair(record=(UNREAL, 100.0)):
            rig, other = _chain(), _chain()
            other.axis_system, other.unit = record
            return rig, other

        def refused(label, call, rig, other, record):
            names = (rig.name, other.name)
            with self.subTest(label, record=record):
                with self.assertRaises(ValueError) as caught:
                    call()
                _names_both(self, caught, record)
                self.assertEqual((rig.name, other.name), names)  # nothing moved
                self.assertEqual(_records(rig), ("maya_y_up", 1.0))

        for record in _OTHERS:
            for label, join in (
                ("extend", lambda rig, other: rig.extend(other)),
                ("extend(a view)", lambda rig, other: rig.extend(other.match("grp", "loc"))),
                ("append", lambda rig, other: rig.append(other["loc"])),
                ("insert", lambda rig, other: rig.insert(0, other["loc"])),
                ("an item", lambda rig, other: rig.__setitem__(0, other["loc"])),
                ("a slice", lambda rig, other: rig.__setitem__(slice(0, 1), [other["loc"]])),
            ):
                rig, other = pair(record)
                refused(label, lambda: join(rig, other), rig, other, record)
            rig, other = pair(record)
            for label, call in (
                ("HierarchyData(nodes)", lambda: HierarchyData([rig["hip"], other["grp"]])),
                ("ClipData(nodes)", lambda: ClipData([rig["hip"], other["grp"]], frames=2)),
                ("ClipData.from_poses", lambda: ClipData.from_poses([rig, other])),
                ("ClipData.from_poses(nodes)", lambda: ClipData.from_poses([rig, list(other)])),
            ):
                refused(label, call, rig, other, record)

        # an empty rig takes theirs
        for label, build in (
            ("HierarchyData(nodes)", lambda other: HierarchyData([other["hip"], other["knee"]])),
            ("an empty rig's extend", lambda other: _joined(HierarchyData(), "extend", other.match("grp", "loc"))),
            ("an empty rig's append", lambda other: _joined(HierarchyData(), "append", other["hip"])),
            ("ClipData(nodes)", lambda other: ClipData(list(other), frames=2)),
            ("ClipData.from_poses(nodes)", lambda other: ClipData.from_poses([list(other), other])),
        ):
            with self.subTest(label):
                self.assertEqual(_records(build(pair()[1])), (UNREAL, 100.0))

        # nodes given as an iterator are read once, for their record and
        # their values
        other = pair()[1]
        clip  = ClipData((node for node in other), frames=2)
        self.assertEqual((_records(clip), clip.name), ((UNREAL, 100.0), other.name))
        clip = ClipData.from_poses([iter(other.list), iter(other.list)])
        self.assertEqual((_records(clip), clip.name, clip.frame_count), ((UNREAL, 100.0), other.name, 2))

        # the nodes no rig owns of the view TransformList.load_fbx gives are
        # in its load's record, joining by extend or by a slice alike
        from cgmath.hierarchy.hierarchy import _recorded

        def loaded(record=(UNREAL, 100.0)):
            return _recorded(TransformList([node.copy() for node in _chain()]), record)

        for record in _OTHERS:
            for label, join in (
                ("extend(a load's view)", lambda rig, view: rig.extend(view)),
                ("a slice at the end", lambda rig, view: rig.__setitem__(slice(len(rig), None), view)),
                ("a slice at 0", lambda rig, view: rig.__setitem__(slice(0, 0), view)),
                ("a slice over every node", lambda rig, view: rig.__setitem__(slice(None), view)),
            ):
                rig, view = _chain(), loaded(record)
                refused(label, lambda: join(rig, view), rig, view, record)
            clip, view = ClipData(_chain(), frames=2), loaded(record)
            refused("a clip's slice", lambda: clip.__setitem__(slice(len(clip), None), view), clip, view, record)
        for label, join in (
            ("an empty rig's extend", lambda rig, view: rig.extend(view)),
            ("an empty rig's slice", lambda rig, view: rig.__setitem__(slice(None), view)),
        ):
            with self.subTest(label):
                rig, view = HierarchyData(), loaded()
                join(rig, view)
                self.assertEqual((_records(rig), rig.name), ((UNREAL, 100.0), view.name))

        # a rig of the same record and a node no rig owns join as they always have
        other, same = pair()[1], pair()[1]
        other.extend(same)
        other.append(TransformData("loose"))
        self.assertEqual((_records(other), len(other)), ((UNREAL, 100.0), 9))

    def test_a_clips_rest_pose_takes_the_clips_record(self):
        rig = _chain()
        rig.axis_system, rig.unit = UNREAL, 100.0
        clip             = ClipData(rig, frames=2)
        clip.rest_pose   = rig.copy()

        clip.axis_system = "maya_y_up"
        self.assertEqual(_records(clip.rest_pose), ("maya_y_up", 100.0))
        clip.unit = 2.54
        self.assertEqual(_records(clip.rest_pose), ("maya_y_up", 2.54))
        with self.assertRaises(ValueError):
            clip.unit = -1.0
        self.assertEqual(_records(clip.rest_pose), ("maya_y_up", 2.54))  # a refused value relabels nothing

        held = clip.rest_pose
        for label, record in (("axes", (UNREAL, 2.54)), ("unit", ("maya_y_up", 100.0))):
            with self.subTest(label):
                pose = _chain()
                pose.axis_system, pose.unit = record
                with self.assertRaises(ValueError) as caught:
                    clip.rest_pose = pose
                self.assertIn("rest pose", str(caught.exception))
                self.assertIs(clip.rest_pose, held)
        clip.rest_pose = None
        clip.rest_pose = held

        # a saved clip whose rest pose records others raises on load
        tree = clip.to_dict()
        self.assertEqual(_records(ClipData.from_dict(tree).rest_pose), ("maya_y_up", 2.54))
        tree["rest_pose"]["unit"] = 1.0
        with self.assertRaises(ValueError):
            ClipData.from_dict(tree)

        # Maya's Y-up in words is "maya_y_up" for the rest pose too
        clip             = ClipData(rig, frames=2)
        clip.rest_pose   = rig.copy()
        clip.axis_system = "y up, z front, right handed"
        want             = rig.copy()
        want.axis_system = "maya_y_up"
        self.assertEqual((clip.axis_system, clip.rest_pose.axis_system), ("maya_y_up", "maya_y_up"))
        self.assertEqual(clip.rest_pose, want)

        # the rest pose set is held as it is: a relabel gives the clip a
        # relabelled copy of it, leaving the one set, another clip's too
        for label, name, value, record in (
            ("axes", "axis_system", "maya_y_up", ("maya_y_up", 100.0)),
            ("unit", "unit", 2.54, (UNREAL, 2.54)),
        ):
            with self.subTest(label):
                shared = rig.copy()
                first, second = ClipData(rig, frames=2), ClipData(rig, frames=2)
                first.rest_pose, second.rest_pose = shared, shared
                setattr(first, name, value)
                self.assertEqual([_records(x) for x in (first.rest_pose, second.rest_pose)], [record, (UNREAL, 100.0)])
                self.assertIs(second.rest_pose, shared)
                self.assertEqual(ClipData.from_dict(second.to_dict()), second)

        # one relabelled after it was set: the clip is not saved, pickled,
        # copied or cut, and no file is written
        second.rest_pose.unit = 1.0
        out                   = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        for ext in ("npz", "json", "pkl"):
            path = os.path.join(out.name, f"clip.{ext}")
            with self.subTest(ext):
                with self.assertRaises(ValueError) as caught:
                    second.save(path)
                self.assertIn("rest pose", str(caught.exception))
                self.assertFalse(os.path.exists(path))
        for label, call in (
            ("pickle", lambda: pickle.dumps(second)),
            ("copy()", second.copy),
            ("deepcopy", lambda: copy.deepcopy(second)),
            ("a cut", lambda: second.frames[0:1]),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn("rest pose", str(caught.exception))

    def test_the_pose_algebra_takes_one_record(self):
        rig                  = _chain()
        posed                = rig.copy()
        posed["knee"].rotate = (0.0, 0.0, -45.0)
        delta                = posed.get_delta(rig)
        deltas               = ClipData(posed, frames=3).get_delta(rig)
        for record in _OTHERS:
            other = posed.copy()
            other.axis_system, other.unit = record
            clip = ClipData(other, frames=3)
            for label, call in (
                ("get_delta", lambda: other.get_delta(rig)),
                ("get_delta of a view", lambda: other.match("hip", "knee").get_delta(rig.match("hip", "knee"))),
                ("add_delta", lambda: other.add_delta(delta)),
                ("rig - rig", lambda: other - rig),
                ("delta + rig", lambda: delta + other),
                ("get_delta(frames)", lambda: rig.get_delta(clip.frames)),
                ("add_delta(frames)", lambda: other.add_delta(deltas.frames)),
                ("clip.get_delta", lambda: clip.get_delta(rig)),
                ("clip.add_delta", lambda: clip.add_delta(delta)),
                ("clip.add_delta(frames)", lambda: clip.add_delta(deltas.frames)),
                ("rig -= rig", lambda: other.copy().__isub__(rig)),
                ("delta += rig", lambda: delta.copy().__iadd__(other)),
                ("frames -= rig", lambda: clip.copy().frames.__isub__(rig)),
                ("frames += rig", lambda: deltas.copy().frames.__iadd__(other)),
            ):
                with self.subTest(label, record=record):
                    with self.assertRaises(ValueError) as caught:
                        call()
                    _names_both(self, caught, record)
            self.assertEqual(_records(other.add_delta(other.get_delta(other.copy()))), record)

    def test_matching_or_pasting_takes_one_record(self):
        """the values of nodes of another record matched or pasted into a
        node, a view, a rig, a clip or its frames are refused before anything
        moves; of the same record they are not, nor values or nodes no rig
        owns"""
        from cgmath.hierarchy.hierarchy import _recorded

        def build(record) -> dict:
            rig                    = _chain()
            rig["knee"].rotate     = (0.0, 0.0, -45.0)
            other                  = _chain()
            other["hip"].translate = (5.0, 6.0, 7.0)
            other["loc"].rotate    = (0.0, 90.0, 0.0)
            other.axis_system, other.unit = record
            data = {
                "rig":   rig,
                "other": other,
                # TransformList.load_fbx's view: nodes no rig owns, in its load's record
                "loaded": _recorded(TransformList([node.copy() for node in other]), record),
                "clip":   ClipData(rig, frames=3),
                "clips":  ClipData(other, frames=3),
            }
            data["clip"].frame = 1
            return data

        def state(data) -> dict:
            """every world matrix a call could move: the rigs', the load's
            view's, the clip's bound frame's and every frame's"""
            keys = ("rig", "other", "loaded", "clip")
            return dict(
                {key: np.array([node.world_matrix for node in data[key]]) for key in keys},
                frames=np.array(data["clip"].frames.world_matrix),
            )

        def paste(where, what):
            return lambda d: d["clip"].frames.__setitem__(where, what(d))

        calls = {
            "node.match_translate(node)": lambda d: d["rig"]["knee"].match_translate(d["other"]["knee"]),
            "node.match_rotate(node)": lambda d: d["rig"]["knee"].match_rotate(d["other"]["loc"]),
            "node.match_matrix(node)": lambda d: d["rig"]["knee"].match_matrix(d["other"]["loc"]),
            "its node.match_matrix(node)": lambda d: d["other"]["loc"].match_matrix(d["rig"]["knee"]),
            "rig.match_translate(rig)": lambda d: d["rig"].match_translate(d["other"]),
            "rig.match_rotate(nodes)": lambda d: d["rig"].match_rotate(list(d["other"])),
            "rig.match_matrix(rig)": lambda d: d["rig"].match_matrix(d["other"]),
            "its rig.match_rotate(rig)": lambda d: d["other"].match_rotate(d["rig"]),
            "view.match_matrix(view)": lambda d: d["rig"][0:2].match_matrix(d["other"][0:2]),
            "view.match_translate(node)": lambda d: d["rig"][0:2].match_translate(d["other"]["loc"]),
            "rig.match_matrix(a load's view)": lambda d: d["rig"].match_matrix(d["loaded"]),
            "a load's view.match_matrix(rig)": lambda d: d["loaded"].match_matrix(d["rig"]),
            "clip.match_translate(rig)": lambda d: d["clip"].match_translate(d["other"]),
            "clip.match_rotate(node)": lambda d: d["clip"].match_rotate(d["other"]["loc"]),
            "clip.match_matrix(rig)": lambda d: d["clip"].match_matrix(d["other"]),
            "frames.match_translate(rig)": lambda d: d["clip"].frames.match_translate(d["other"]),
            "frames.match_rotate(clip)": lambda d: d["clip"].frames.match_rotate(d["clips"]),
            "frames.match_matrix(frames)": lambda d: d["clip"].frames.match_matrix(d["clips"].frames),
            "frames.match_matrix(load view)": lambda d: d["clip"].frames.match_matrix(d["loaded"]),
            "frames['knee'].match_matrix": lambda d: d["clip"].frames["knee"].match_matrix(d["other"]["loc"]),
            "frames[0:3] = a clip's frames": paste(slice(0, 3), lambda d: d["clips"].frames[0:3]),
            "frames[1] = rig": paste(1, lambda d: d["other"]),
            "frames[:] = rig": paste(slice(None), lambda d: d["other"]),
            "frames[0] = nodes": paste(0, lambda d: list(d["other"])),
            "frames[0] = an iterator": paste(0, lambda d: iter(d["other"].list)),
            "frames[0] = a load's view": paste(0, lambda d: d["loaded"]),
        }
        for record in _OTHERS:
            for label, call in calls.items():
                with self.subTest(label, record=record):
                    data   = build(record)
                    before = state(data)
                    with self.assertRaises(ValueError) as caught:
                        call(data)
                    _names_both(self, caught, record)
                    for key, values in state(data).items():
                        np.testing.assert_array_equal(values, before[key], err_msg=key)  # nothing moved

        # a rig or a clip listing a node another rig owns, as an item
        # assignment lists one, holds that rig's record for it too: once the
        # owner or the lister is relabelled, a match or a paste is refused
        # before the first node moves, by every route, where the rig's own
        # record alone let it through to the listed node, the nodes before it
        # moved, and clip.frames took it whole
        def moved():
            """a chain whose every node sits and turns elsewhere"""
            rig = _chain()
            for node in rig:
                node.translate = np.asarray(node.translate) + 5.0
                node.rotate    = np.asarray(node.rotate) + 5.0
            return rig

        listed = {
            "rig.match_matrix(rig)": lambda a, c: a.match_matrix(c),
            "rig.match_translate(rig)": lambda a, c: a.match_translate(c),
            "rig.match_rotate(rig)": lambda a, c: a.match_rotate(c),
            "its rig.match_matrix(the lister)": lambda a, c: c.match_matrix(a),
            "clip.match_matrix(rig)": lambda a, c: a.match_matrix(c),
            "clip.frames.match_matrix(rig)": lambda a, c: a.frames.match_matrix(c),
            "clip.frames[0] = rig": lambda a, c: a.frames.__setitem__(0, c),
        }
        for relabelled in ("the owner", "the lister"):
            for label, call in listed.items():
                with self.subTest(label, relabelled=relabelled):
                    a    = ClipData(_chain(), frames=3) if label.startswith("clip") else _chain()
                    b    = _chain()
                    c    = moved()
                    a[3] = b["loc"]
                    if relabelled == "the owner":
                        b.unit = 100.0
                    else:
                        a.unit = c.unit = 100.0
                    before = [np.array([node.world_matrix for node in x]) for x in (a, b, c)]
                    if isinstance(a, ClipData):
                        before.append(np.array(a.frames.world_matrix))
                    with self.assertRaises(ValueError) as caught:
                        call(a, c)
                    self.assertIn("unit 100.0", str(caught.exception))
                    self.assertIn("unit 1.0", str(caught.exception))
                    after = [np.array([node.world_matrix for node in x]) for x in (a, b, c)]
                    if isinstance(a, ClipData):
                        after.append(np.array(a.frames.world_matrix))
                    for got, want in zip(after, before):
                        np.testing.assert_array_equal(got, want)  # nothing moved

        # of the same record each one runs, and moves its receiver
        for label, call in calls.items():
            with self.subTest(label, record="the same"):
                data   = build(("maya_y_up", 1.0))
                before = state(data)
                call(data)
                after = state(data)
                self.assertTrue(any(not np.allclose(after[key], before[key]) for key in before))

        # values hold no record, and nor does a node no rig owns
        data   = build((UNREAL, 100.0))
        loose  = TransformData("loose", translate=(1.0, 2.0, 3.0), rotate=(0.0, 30.0, 0.0))
        node   = data["other"]["loc"]
        frames = data["clips"].frames
        for label, call in (
            ("node.match_translate(a position)", lambda: node.match_translate((1.0, 2.0, 3.0))),
            ("rig.match_rotate(rotations)", lambda: data["other"].match_rotate(np.zeros((4, 3)))),
            ("frames.match_translate(positions)", lambda: frames.match_translate((1.0, 2.0, 3.0))),
            ("node.match_matrix(a node no rig owns)", lambda: node.match_matrix(loose)),
            ("a node no rig owns' match_matrix(node)", lambda: loose.copy().match_matrix(node)),
            ("frames['loc'].match_matrix(a node no rig owns)", lambda: frames["loc"].match_matrix(loose)),
            ("frames[0] = nodes no rig owns", lambda: frames.__setitem__(0, [n.copy() for n in _chain()])),
        ):
            with self.subTest(label):
                call()
        self.assertEqual(_records(data["other"]), (UNREAL, 100.0))

    def test_two_rigs_recording_other_axes_or_units_differ(self):
        a = _chain()
        b = a.copy()
        self.assertEqual(a, b)
        b.unit = 100.0
        self.assertNotEqual(a, b)
        b.unit, b.axis_system = 1.0, UNREAL
        self.assertNotEqual(a, b)


# ------------------------------------------------------------------ fbx files

# the six global settings an axis system is stored in, and those of the
# axis systems the tests write: (up, its sign, front, its sign, the third
# axis, its sign), the axes 0, 1 and 2 for x, y and z
_AXES_PROPERTIES = ("UpAxis", "UpAxisSign", "FrontAxis", "FrontAxisSign", "CoordAxis", "CoordAxisSign")
_AXES = {
    "maya_y_up": (1, 1, 2, 1, 0, 1),
    UNREAL: (2, 1, 1, -1, 0, 1),
    "x up, y front, right handed": (0, 1, 1, 1, 2, 1),
    "y up, x front, right handed": (1, 1, 0, 1, 2, -1),
    "y up, z front, left handed": (1, 1, 2, 1, 0, -1),
    "-y up, z front, right handed": (1, -1, 2, 1, 0, -1),
}

# (axes, the file's unit in cm, what its lengths are multiplied by, and
# whether the sdk's FbxSystemUnit.m.ConvertScene writes it, the unit then a
# scale on its roots) of each file the tests write
_SOURCES = {
    "yup_cm":      ("maya_y_up", 1.0, 1.0, False),
    "zup_cm":      (UNREAL, 1.0, 1.0, False),
    "xup_cm":      ("x up, y front, right handed", 1.0, 1.0, False),
    "yfx_cm":      ("y up, x front, right handed", 1.0, 1.0, False),
    "zup_m":       (UNREAL, 100.0, 0.01, False),          # only labelled in meters
    "yup_in":      ("maya_y_up", 2.54, 1 / 2.54, False),  # only labelled in inches
    "zup_cm_to_m": (UNREAL, 1.0, 1.0, True),
    "left_cm":     ("y up, z front, left handed", 1.0, 1.0, False),
}
_ONLY_LABELLED = ("zup_m", "yup_in")
_FRAMES        = 4

# the reads of every file but the left handed one
_MODES = (
    ("default", {}),
    ("its own axes", {"axis_system": None}),
    ("as it is", {"axis_system": None, "scale_factor": 1.0}),
    ("by 0.01", {"scale_factor": 0.01}),
    ("by 2", {"scale_factor": 2.0}),
)


def _cases():
    """(file, read, options) of each read a save lands where its values say
    of, every file but the left handed one in every mode: a file holding its
    unit as a scale on its roots, read with a number other than 1, keeps
    that scale, which the unit it records no longer explains (save_fbx's
    docstring says so)"""
    for label, (mode, options) in itertools.product(_SOURCES, _MODES):
        if label == "left_cm":
            continue
        if _SOURCES[label][3] and options.get("scale_factor") not in (None, 1.0):
            continue
        yield label, mode, options


def _recorded(label, options) -> tuple:
    """(axis_system, unit) a read of the file ``label`` with ``options`` records"""
    axes, unit, _, convert = _SOURCES[label]
    unit = 100.0 if convert else unit  # the file's own unit, meters once converted
    if options.get("axis_system", "maya_y_up") is not None:
        axes = "maya_y_up"
    number = options.get("scale_factor")
    return axes, 1.0 if number is None else unit / number


def _author(path, axes, unit, factor, convert, bind=False) -> str:
    """a group root moved and turned over a joint chain (the hip with a pre
    rotation) and a mesh, a locator root, a mesh root, and a node with no
    attribute over a joint, which the read drops and folds, keyed over 4
    frames at 30 fps, in ``axes`` and ``unit``, every length times
    ``factor``; written in meters by FbxSystemUnit.m.ConvertScene when
    ``convert``. With ``bind``, a bind pose lists the hip, the knee and the
    spine where they stand, and the knee and the spine then turn off it"""
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    scene    = FBX.FbxScene.Create(manager, "scene")
    settings = scene.GetGlobalSettings()
    for name, value in zip(_AXES_PROPERTIES, _AXES[axes]):
        FBX.FbxPropertyInteger1(settings.FindProperty(name)).Set(value)
    settings.SetSystemUnit(FBX.FbxSystemUnit(unit))
    settings.SetTimeMode(FBX.FbxTime.EMode.eFrames30)

    def cube(name, side):
        mesh    = FBX.FbxMesh.Create(scene, name)
        corners = [(x, y, z) for x in (0, side) for y in (0, side) for z in (0, side)]
        mesh.InitControlPoints(len(corners))
        for index, corner in enumerate(corners):
            mesh.SetControlPointAt(FBX.FbxVector4(*corner, 1.0), index)
        for face in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            mesh.BeginPolygon()
            for index in face:
                mesh.AddPolygon(index)
            mesh.EndPolygon()
        return mesh

    def node(name, kind, parent, translate, rotate=(0, 0, 0), scale=(1, 1, 1)):
        made = FBX.FbxNode.Create(scene, name)
        if kind == "joint":
            attr = FBX.FbxSkeleton.Create(scene, name)
            attr.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
            made.SetNodeAttribute(attr)
        elif kind in ("group", "locator"):
            attr = FBX.FbxNull.Create(scene, name)
            look = FBX.FbxNull.ELook.eCross if kind == "locator" else FBX.FbxNull.ELook.eNone
            attr.Look.Set(look.value)
            made.SetNodeAttribute(attr)
        elif kind == "mesh":
            made.SetNodeAttribute(cube(name, 10.0 * factor))
        made.LclTranslation.Set(FBX.FbxDouble3(*[x * factor for x in translate]))
        made.LclRotation.Set(FBX.FbxDouble3(*rotate))
        made.LclScaling.Set(FBX.FbxDouble3(*scale))
        made.SetRotationActive(True)
        parent.AddChild(made)
        return made

    top = scene.GetRootNode()
    grp = node("grp", "group", top, (10, 20, 30), (0, 0, 45))
    hip = node("hip", "joint", grp, (0, 90, 0), (10, 0, 0))
    hip.SetPreRotation(FBX.FbxNode.EPivotSet.eSourcePivot, FBX.FbxVector4(0, 0, 90, 0))
    knee = node("knee", "joint", hip, (40, 0, 0), (0, 0, -30))
    node("foot", "joint", knee, (40, 0, 0))
    node("box", "mesh", grp, (5, 0, 0), (0, 30, 0), (1, 2, 1))
    node("loc", "locator", top, (-20, 0, 15), (15, 25, 35))
    node("prop", "mesh", top, (0, 0, -40), (0, 45, 0))
    bare  = node("bare", None, top, (3, 4, 5), (5, 10, 15))
    spine = node("spine", "joint", bare, (0, 10, 0), (0, 20, 0))

    if bind:
        pose = FBX.FbxPose.Create(scene, "bindPose")
        pose.SetIsBindPose(True)
        for listed in (hip, knee, spine):
            pose.Add(listed, FBX.FbxMatrix(listed.EvaluateGlobalTransform()))
        scene.AddPose(pose)
        knee.LclRotation.Set(FBX.FbxDouble3(-50.0, 5.0, 0.0))
        spine.LclRotation.Set(FBX.FbxDouble3(0.0, 0.0, 35.0))

    stack = FBX.FbxAnimStack.Create(scene, "take")
    layer = FBX.FbxAnimLayer.Create(scene, "base")
    stack.AddMember(layer)
    last = (_FRAMES - 1) / 30.0
    for prop, axis, values in (
        (hip.LclRotation, "X", (10.0, 70.0)),
        (grp.LclTranslation, "X", (10.0 * factor, 60.0 * factor)),
        (grp.LclTranslation, "Y", (20.0 * factor, -10.0 * factor)),
        (knee.LclRotation, "Z", (-30.0, -80.0)),
    ):
        curve = prop.GetCurve(layer, axis, True)
        curve.KeyModifyBegin()
        for seconds, value in zip((0.0, last), values):
            time = FBX.FbxTime()
            time.SetSecondDouble(seconds)
            index = curve.KeyAdd(time)[0]
            curve.KeySetValue(index, value)
            curve.KeySetInterpolation(index, FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
        curve.KeyModifyEnd()
    start, stop = FBX.FbxTime(), FBX.FbxTime()
    stop.SetSecondDouble(last)
    stack.SetLocalTimeSpan(FBX.FbxTimeSpan(start, stop))

    if convert:
        FBX.FbxSystemUnit.m.ConvertScene(scene)
    exporter = FBX.FbxExporter.Create(manager, "")
    exporter.Initialize(path, -1, manager.GetIOSettings())
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
    return path


def _author_all(folder, bind=False) -> dict:
    """{label: path} of every file of _SOURCES, written to ``folder``, each
    holding a bind pose when ``bind``"""
    name = "{}_bind.fbx" if bind else "{}.fbx"
    return {
        label: _author(os.path.join(folder, name.format(label)), *source, bind=bind)
        for label, source in _SOURCES.items()
    }


@contextlib.contextmanager
def _sdk_scene(path, converted=False):
    """the sdk's scene of ``path``, converted to Maya's Y-up and cm, as
    Maya's importer converts a file, when ``converted``"""
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    importer = FBX.FbxImporter.Create(manager, "")
    importer.Initialize(path, -1, manager.GetIOSettings())
    scene = FBX.FbxScene.Create(manager, "")
    importer.Import(scene)
    importer.Destroy()
    try:
        if converted:
            FBX.FbxAxisSystem.MayaYUp.ConvertScene(scene)
            FBX.FbxSystemUnit.cm.ConvertScene(scene)
        yield scene
    finally:
        manager.Destroy()


def _sdk_nodes(scene) -> list:
    """every node of ``scene`` holding a node attribute"""
    out, stack = [], [scene.GetRootNode()]
    while stack:
        node = stack.pop()
        stack.extend(node.GetChild(i) for i in range(node.GetChildCount()))
        if node.GetNodeAttribute() is not None:
            out.append(node)
    return out


def _m4(matrix) -> np.ndarray:
    return np.array([[matrix.Get(r, c) for c in range(4)] for r in range(4)])


def _label(path) -> tuple:
    """(the six axis properties, the unit in cm) the sdk reads from ``path``"""
    with _sdk_scene(path) as scene:
        settings = scene.GetGlobalSettings()
        axes     = tuple(FBX.FbxPropertyInteger1(settings.FindProperty(x)).Get() for x in _AXES_PROPERTIES)
        return axes, settings.GetSystemUnit().GetScaleFactor()


def _sdk_worlds(path) -> dict:
    """every node's world matrix as the sdk gives it once the file is
    converted to Maya's Y-up and cm, as Maya's importer converts it"""
    with _sdk_scene(path, converted=True) as scene:
        return {node.GetName(): _m4(node.EvaluateGlobalTransform()) for node in _sdk_nodes(scene)}


def _sdk_points(path) -> dict:
    """each mesh's control points under its node's world, where the sdk puts
    them once the file is converted to Maya's Y-up and cm"""
    kind = FBX.FbxNodeAttribute.EType.eMesh
    out  = {}
    with _sdk_scene(path, converted=True) as scene:
        for node in _sdk_nodes(scene):
            if node.GetNodeAttribute().GetAttributeType() != kind:
                continue
            mesh  = node.GetMesh()
            world = _m4(node.EvaluateGlobalTransform())
            points = np.array(
                [[mesh.GetControlPointAt(i)[k] for k in range(3)] for i in range(mesh.GetControlPointsCount())]
            )
            out[node.GetName()] = points @ world[:3, :3] + world[3, :3]
    return out


def _axes_only(matrix) -> np.ndarray:
    """``matrix`` with each axis made unit length: its position and axes"""
    matrix = np.array(matrix, dtype=float)
    matrix[:3, :3] /= np.linalg.norm(matrix[:3, :3], axis=1, keepdims=True)
    return matrix


@unittest.skipIf(FBX is None, "the Autodesk FBX SDK is required")
class TestSaveFbxLabels(unittest.TestCase):
    """save_fbx labels the file with the record of the values it writes, so
    every save loads with load_fbx's defaults where the file the values were
    read from loads, reads as it is to the values themselves, and sits where
    the sdk's conversion to Maya's Y-up and cm puts that file (_author's
    files)"""

    @classmethod
    def setUpClass(cls):
        cls.tmp   = tempfile.TemporaryDirectory()
        cls.files = _author_all(cls.tmp.name)
        cls.binds = _author_all(cls.tmp.name, bind=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        self.out = out.name

    def assertPlaced(self, got, want, label, scale=True):
        """every node of ``got`` where ``want`` puts it: the full 4x4 to 1e-9
        of its largest number, or its position and axes alone, its axes'
        lengths left out, when not ``scale``"""
        for name, matrix in got.items():
            other = np.asarray(want[name])
            if not scale:
                matrix, other = _axes_only(matrix), _axes_only(other)
            size = max(1.0, float(np.abs(other).max()))
            gap  = float(np.abs(np.asarray(matrix) - other).max())
            self.assertLessEqual(gap, 1e-9 * size, f"{label}: {name} {gap:.2e} off")

    def test_a_load_records_what_it_produced(self):
        for label, mode, options in itertools.chain(_cases(), [("left_cm", "its own axes", {"axis_system": None})]):
            path = self.files[label]
            want = _recorded(label, options)
            with self.subTest(file=label, mode=mode):
                clip = ClipData.load_fbx(path, **options)
                view = TransformList.load_fbx(path, **options)
                for data in (HierarchyData.load_fbx(path, **options), clip, clip.rest_pose, view, HierarchyData(view)):
                    self.assertEqual((data.axis_system, data.unit), want)
                    self.assertIs(type(data.unit), float)

    def test_a_bind_read_records_what_it_produced(self):
        """pose="bind", a return of its own in each loader: the rig and the
        view record what the read produced, and the rig's save loads with
        the defaults where the file's default bind read puts it"""
        nodes = _worlds(HierarchyData.load_fbx(self.binds["yup_cm"]))
        binds = _worlds(HierarchyData.load_fbx(self.binds["yup_cm"], pose="bind"))
        self.assertGreater(float(np.abs(nodes["knee"] - binds["knee"]).max()), 0.1)  # the bind is no node's values

        for label, mode, options in _cases():
            path = self.binds[label]
            want = _recorded(label, options)
            with self.subTest(file=label, mode=mode):
                rig  = HierarchyData.load_fbx(path, pose="bind", **options)
                view = TransformList.load_fbx(path, pose="bind", **options)
                for data in (rig, view, HierarchyData(view)):
                    self.assertEqual(_records(data), want)
                out = os.path.join(self.out, f"{label}_bind.fbx")
                rig.save_fbx(out)
                truth = _worlds(HierarchyData.load_fbx(path, pose="bind"))
                self.assertPlaced(_worlds(HierarchyData.load_fbx(out)), truth, "bind")

    def test_a_view_taken_from_a_loads_view_keeps_its_record(self):
        """a slice, an index, a match, the roots, a branch or a TransformList
        of the view TransformList.load_fbx gives, whose nodes no rig owns,
        record its load's, and so does a rig built from it, whose save
        loads with the defaults where its file does; the view's nodes join
        a rig in that record"""
        cuts = (
            ("a slice", lambda view: view[0:]),
            ("an index", lambda view: view[list(range(len(view)))]),
            ("a mask", lambda view: view[np.ones(len(view), dtype=bool)]),
            ("match", lambda view: view.match("*", exact=False)),
            ("get_roots", lambda view: view.get_roots()),
            ("get_branch", lambda view: view.get_branch()),
            ("TransformList", lambda view: TransformList(view)),
        )
        for label, mode, options in _cases():
            if label not in ("zup_cm", "zup_m", "yup_in") or not options:
                continue
            path  = self.files[label]
            want  = _recorded(label, options)
            truth = _worlds(HierarchyData.load_fbx(path))
            for cut, take in cuts:
                with self.subTest(file=label, mode=mode, cut=cut):
                    taken = take(TransformList.load_fbx(path, **options))
                    self.assertIs(type(taken), TransformList)
                    self.assertEqual(_records(taken), want)
                    rig = HierarchyData(taken)
                    self.assertEqual(_records(rig), want)
                    out = os.path.join(self.out, f"{label}_cut.fbx")
                    rig.save_fbx(out)
                    self.assertPlaced(_worlds(HierarchyData.load_fbx(out)), truth, cut)

            # its nodes join a rig in its load's record: an empty one takes
            # it, one recording another refuses them
            with self.subTest(file=label, mode=mode, cut="extend"):
                view = TransformList.load_fbx(path, **options)
                self.assertEqual(_records(_joined(HierarchyData(), "extend", view)), want)
                if want != ("maya_y_up", 1.0):
                    other = _chain()
                    with self.assertRaises(ValueError):
                        other.extend(TransformList.load_fbx(path, **options))
                    self.assertEqual(len(other), 4)

    def test_a_loads_view_saves_as_the_rig_that_owns_its_nodes(self):
        """once a rig owns the nodes of TransformList.load_fbx's view, which
        HierarchyData(view) makes, and is relabelled, the view, every view
        taken from it and a clip of one record the rig's and save where the
        rig's own save loads, labelled as it is; the rig's pose algebra
        takes them"""
        view = TransformList.load_fbx(self.files["yup_in"], axis_system=None, scale_factor=1.0)
        rig  = HierarchyData(view)
        self.assertEqual(_records(rig), ("maya_y_up", 2.54))
        rig.unit = 1.0
        out      = os.path.join(self.out, "rig.fbx")
        rig.save_fbx(out)
        truth = _worlds(HierarchyData.load_fbx(out))
        for label, taken in (
            ("the view", view),
            ("a slice", view[0:]),
            ("an index", view[list(range(len(view)))]),
            ("a mask", view[np.ones(len(view), dtype=bool)]),
            ("match", view.match("*", exact=False)),
            ("get_roots", view.get_roots()),
            ("get_branch", view.get_branch()),
            ("TransformList", TransformList(view)),
        ):
            with self.subTest(label):
                self.assertEqual(_records(taken), ("maya_y_up", 1.0))
                for kind, data in (("view", taken), ("clip", ClipData(taken, frames=1))):
                    saved = os.path.join(self.out, f"{kind}.fbx")
                    data.save_fbx(saved)
                    self.assertEqual(_label(saved), _label(out))
                    self.assertPlaced(_worlds(HierarchyData.load_fbx(saved)), truth, f"{label}, {kind}")
        self.assertEqual(_records(rig.get_delta(view[0:])), ("maya_y_up", 1.0))

    def test_a_scale_factor_that_gives_no_unit_raises(self):
        for loader in (HierarchyData.load_fbx, ClipData.load_fbx, TransformList.load_fbx):
            for number in (0.0, -1.0, float("nan"), float("inf"), "x"):
                with self.subTest(loader=loader.__qualname__, number=number):
                    with self.assertRaises(ValueError) as caught:
                        loader(self.files["yup_cm"], scale_factor=number)
                    self.assertIn("scale_factor", str(caught.exception))

    def test_save_fbx_labels_the_file_with_the_record(self):
        rig  = _chain()
        path = os.path.join(self.out, "labelled.fbx")
        for axes, unit in itertools.product(_AXES, (1.0, 100.0, 2.54, 0.4)):
            with self.subTest(axes=axes, unit=unit):
                rig.axis_system, rig.unit = axes, unit
                rig.save_fbx(path)
                self.assertEqual(_label(path), (_AXES[axes], unit))

                back = HierarchyData.load_fbx(path, axis_system=None, scale_factor=1.0)
                self.assertEqual((back.axis_system, back.unit), (axes, unit))
                self.assertPlaced(_worlds(back), _worlds(rig), "as it is")

    def test_a_save_loads_where_the_values_say(self):
        for label, mode, options in _cases():
            path = self.files[label]
            with self.subTest(file=label, mode=mode):
                data = HierarchyData.load_fbx(path, **options)
                out  = os.path.join(self.out, f"{label}.fbx")
                data.save_fbx(out)
                self.assertEqual(_label(out)[1], data.unit)

                # with the defaults, where the file it was read from loads
                self.assertPlaced(_worlds(HierarchyData.load_fbx(out)), _worlds(HierarchyData.load_fbx(path)), "default")
                # as it is, the values themselves
                kept = HierarchyData.load_fbx(out, axis_system=None, scale_factor=1.0)
                self.assertPlaced(_worlds(kept), _worlds(data), "as it is")
                # where the sdk's conversion to Maya's Y-up and cm puts that
                # file; one in another unit than cm holds its lengths in it,
                # which the conversion takes with its roots scaled by it
                got   = _sdk_worlds(out)
                whole = data.unit == 1.0 and label not in _ONLY_LABELLED
                self.assertPlaced({name: got[name] for name in data.name}, _sdk_worlds(path), "sdk", scale=whole)

    def test_a_left_handed_file_saves_in_its_own_axes(self):
        data = HierarchyData.load_fbx(self.files["left_cm"], axis_system=None)
        out  = os.path.join(self.out, "left.fbx")
        data.save_fbx(out)
        self.assertEqual(_label(out), (_AXES["y up, z front, left handed"], 1.0))
        self.assertPlaced(_worlds(HierarchyData.load_fbx(out, axis_system=None)), _worlds(data), "its own axes")
        with self.assertRaises(ValueError) as caught:
            HierarchyData.load_fbx(out)
        self.assertIn("axis_system=None", str(caught.exception))

    def test_a_frame_and_the_rest_pose_save_as_the_rig_does(self):
        for label, mode, options in _cases():
            path = self.files[label]
            with self.subTest(file=label, mode=mode):
                clip    = ClipData.load_fbx(path, **options)
                default = ClipData.load_fbx(path)
                for name, data, want in (
                    ("frame", clip.frames[2:3], default.frames[2:3]),
                    ("rest pose", clip.rest_pose, default.rest_pose),
                ):
                    out = os.path.join(self.out, f"{label}_{name}.fbx")
                    data.save_fbx(out)
                    self.assertEqual(_label(out)[1], data.unit)
                    self.assertPlaced(_worlds(HierarchyData.load_fbx(out)), _worlds(want), name)

    def test_a_view_saves_with_its_rigs_record(self):
        rig  = HierarchyData.load_fbx(self.files["zup_cm"], axis_system=None, scale_factor=0.01)
        out  = os.path.join(self.out, "view.fbx")
        kept = ("grp", "hip", "knee", "foot")
        rig.match(*kept).save_fbx(out)
        self.assertEqual(_label(out), (_AXES[UNREAL], 100.0))
        full = _worlds(HierarchyData.load_fbx(self.files["zup_cm"]))
        self.assertPlaced(_worlds(HierarchyData.load_fbx(out)), {name: full[name] for name in kept}, "view")

        mixed   = TransformList([rig["hip"], _chain()["grp"]])
        missing = os.path.join(self.out, "mixed.fbx")
        with self.assertRaises(ValueError):
            mixed.save_fbx(missing)
        self.assertFalse(os.path.exists(missing))

    def test_a_mesh_sits_under_its_saved_node(self):
        """no geometry is written, but a mesh's node is, as a group: a mesh
        the readers keep in its node's object space, in cm, sits under that
        node's world in the save's default read where the sdk puts the
        file's mesh in cm and Maya's Y-up, whatever the rig was read with"""
        from cgmath.geometry.mesh import load_fbx as mesh_load_fbx

        for label, mode, options in _cases():
            path = self.files[label]
            with self.subTest(file=label, mode=mode):
                out = os.path.join(self.out, f"{label}_mesh.fbx")
                HierarchyData.load_fbx(path, **options).save_fbx(out)
                worlds = _worlds(HierarchyData.load_fbx(out))
                truth  = _sdk_points(path)
                meshes = mesh_load_fbx(path)
                self.assertEqual(sorted(mesh.name for mesh, _ in meshes), ["box", "prop"])
                for mesh, _ in meshes:
                    world  = worlds[mesh.name]
                    points = np.asarray(mesh.points) @ world[:3, :3] + world[3, :3]
                    self.assertLess(float(np.abs(points - truth[mesh.name]).max()), 1e-9, mesh.name)

    def test_every_joint_is_written_as_a_limb_node(self):
        out = os.path.join(self.out, "limbs.fbx")
        _chain().match("hip").save_fbx(out)  # a root joint with no child
        with _sdk_scene(out) as scene:
            hip = scene.GetRootNode().GetChild(0)
            self.assertEqual(hip.GetName(), "hip")
            self.assertEqual(hip.GetNodeAttribute().GetSkeletonType(), FBX.FbxSkeleton.EType.eLimbNode)


def _maya():
    """maya.cmds and maya.mel with the fbxmaya plug-in loaded, or None"""
    try:
        import maya.cmds as cmds
        import maya.mel as mel

        cmds.loadPlugin("fbxmaya", quiet=True)
    except Exception:  # not running in Maya, or no fbx plug-in
        return None
    return cmds, mel


@unittest.skipIf(FBX is None, "the Autodesk FBX SDK is required")
class TestSaveFbxInMaya(unittest.TestCase):
    """Maya's FBXImport of a save puts every node where it puts the file the
    values were read from, and a mesh, placed under the group its node is
    saved as, where Maya puts the file's mesh. Needs mayapy with the fbxmaya
    plug-in"""

    @classmethod
    def setUpClass(cls):
        cls.maya = _maya()
        if cls.maya is None:
            raise unittest.SkipTest("needs Maya and its fbxmaya plug-in")
        cls.tmp   = tempfile.TemporaryDirectory()
        cls.files = _author_all(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.cmds, self.mel = self.maya
        self.addCleanup(self.cmds.file, new=True, force=True)
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        self.out = out.name

    def imported(self, path) -> tuple:
        """({node: world matrix}, {mesh's node: world vertices}) of Maya's FBXImport"""
        cmds = self.cmds
        cmds.file(new=True, force=True)
        self.mel.eval("FBXResetImport")
        self.mel.eval(f'FBXImport -f "{path.replace(os.sep, "/")}"')
        worlds = {
            node.rsplit("|", 1)[-1]: np.array(cmds.xform(node, q=True, ws=True, m=True)).reshape(4, 4)
            for node in cmds.ls(type="transform", long=True)
        }
        points = {}
        for shape in cmds.ls(type="mesh", noIntermediate=True, long=True):
            count          = cmds.polyEvaluate(shape, vertex=True)
            flat           = cmds.xform(f"{shape}.vtx[0:{count - 1}]", q=True, ws=True, t=True)
            parent         = cmds.listRelatives(shape, parent=True)[0]
            points[parent] = np.array(flat).reshape(-1, 3)
        return worlds, points

    def test_a_save_imports_where_the_file_it_was_read_from_does(self):
        from cgmath.geometry.mesh import load_fbx as mesh_load_fbx

        for label, mode, options in (
            ("yup_cm", "default", {}),
            ("zup_cm", "its own axes", {"axis_system": None}),
            ("xup_cm", "its own axes", {"axis_system": None}),
            ("yfx_cm", "its own axes", {"axis_system": None}),
            ("yup_cm", "by 0.01", {"scale_factor": 0.01}),
            ("zup_cm", "by 2, its own axes", {"axis_system": None, "scale_factor": 2.0}),
            ("zup_m", "as it is", {"axis_system": None, "scale_factor": 1.0}),
            ("zup_cm_to_m", "as it is", {"axis_system": None, "scale_factor": 1.0}),
            ("zup_cm_to_m", "default", {}),
        ):
            path = self.files[label]
            with self.subTest(file=label, mode=mode):
                data = HierarchyData.load_fbx(path, **options)
                out  = os.path.join(self.out, f"{label}.fbx")
                data.save_fbx(out)
                want, vertices = self.imported(path)
                got, _ = self.imported(out)
                # a file in another unit than cm holds its lengths in it, which
                # Maya's importer takes with its top nodes scaled by the unit
                # Maya's importer keeps a translate in single precision
                whole = data.unit == 1.0 and label not in _ONLY_LABELLED
                for name in data.name:
                    a, b = (got[name], want[name]) if whole else (_axes_only(got[name]), _axes_only(want[name]))
                    self.assertLess(float(np.abs(a - b).max()), 1e-6 * max(1.0, float(np.abs(b).max())), name)
                # a mesh read with the rig's scale_factor, under its group
                for mesh, _ in mesh_load_fbx(path, options.get("scale_factor")):
                    world  = got[mesh.name]
                    points = np.asarray(mesh.points) @ world[:3, :3] + world[3, :3]
                    size   = max(1.0, float(np.abs(vertices[mesh.name]).max()))
                    self.assertLess(float(np.abs(points - vertices[mesh.name]).max()), 1e-6 * size, mesh.name)

    def test_a_root_joint_with_no_child_imports_as_a_joint_with_its_orient(self):
        rig = _chain()
        out = os.path.join(self.out, "hip.fbx")
        rig.match("hip").save_fbx(out)
        worlds, _ = self.imported(out)
        self.assertEqual(self.cmds.nodeType("hip"), "joint")
        self.assertTrue(np.allclose(self.cmds.getAttr("hip.jointOrient")[0], (0.0, 0.0, 90.0)))
        self.assertLess(float(np.abs(worlds["hip"] - np.asarray(rig["hip"].world_matrix)).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
