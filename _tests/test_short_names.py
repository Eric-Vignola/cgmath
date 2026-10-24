"""Maya's short attribute names, and the four matrices that came with them.

Each of the 31 short names reads and writes what its long name does, on a
node, a rig, a clip's bound frame and the clip's frame axis, and raises what
it raises. parent_matrix, inverse_matrix, world_inverse_matrix and
parent_inverse_matrix are checked against matrices composed here with
numpy, to 1e-12, and against Maya itself when the suite runs in mayapy.
add_user_attribute refuses every one of the names; a file whose user
attribute carries one loads it, hidden, with one warning a name.
"""

from __future__ import annotations

import copy
import keyword
import os
import pickle
import rlcompleter
import shutil
import sys
import tempfile
import unittest
import uuid
import warnings
from unittest import mock

import numpy as np
from cgmath import hierarchy
from cgmath.geometry import Data
from cgmath.geometry.sdf import SDFSphere
from cgmath.hierarchy import FBX, ClipData, HierarchyData, TransformData, TransformList
from cgmath.hierarchy import hierarchy as hierarchy_mod
from cgmath.hierarchy.hierarchy import _SHORT_NAMES, _looped

# (short, cgmath's long name, Maya's long name): the short names Maya 2025's
# cmds.attributeQuery(<Maya's long name>, node=<joint>, shortName=True) gives
MAYA_NAMES = (
    ("t", "translate", "translate"),
    ("tx", "translate_x", "translateX"),
    ("ty", "translate_y", "translateY"),
    ("tz", "translate_z", "translateZ"),
    ("r", "rotate", "rotate"),
    ("rx", "rotate_x", "rotateX"),
    ("ry", "rotate_y", "rotateY"),
    ("rz", "rotate_z", "rotateZ"),
    ("s", "scale", "scale"),
    ("sx", "scale_x", "scaleX"),
    ("sy", "scale_y", "scaleY"),
    ("sz", "scale_z", "scaleZ"),
    ("jo", "joint_orient", "jointOrient"),
    ("jox", "joint_orient_x", "jointOrientX"),
    ("joy", "joint_orient_y", "jointOrientY"),
    ("joz", "joint_orient_z", "jointOrientZ"),
    ("ra", "rotate_axis", "rotateAxis"),
    ("rax", "rotate_axis_x", "rotateAxisX"),
    ("ray", "rotate_axis_y", "rotateAxisY"),
    ("raz", "rotate_axis_z", "rotateAxisZ"),
    ("ro", "rotate_order", "rotateOrder"),
    ("ssc", "segment_scale_compensate", "segmentScaleCompensate"),
    ("v", "visibility", "visibility"),
    ("radi", "radius", "radius"),
    ("ds", "draw_style", "drawStyle"),
    ("m", "matrix", "matrix"),
    ("wm", "world_matrix", "worldMatrix"),
    ("pm", "parent_matrix", "parentMatrix"),
    ("im", "inverse_matrix", "inverseMatrix"),
    ("wim", "world_inverse_matrix", "worldInverseMatrix"),
    ("pim", "parent_inverse_matrix", "parentInverseMatrix"),
)
SHORT = {short: long for short, long, _ in MAYA_NAMES}

VECTORS    = ("translate", "rotate", "scale", "joint_orient", "rotate_axis")
COMPONENTS = tuple(f"{vector}_{axis}" for vector in VECTORS for axis in "xyz")
READ_ONLY  = ("parent_matrix", "inverse_matrix", "world_inverse_matrix", "parent_inverse_matrix")

# what every frame shares: a clip's frame axis has none of these
SHARED = ("rotate_order", "segment_scale_compensate", "visibility", "radius", "draw_style")

# four joints, every channel away from its default, four rotate orders, and
# two compensated joints, one under a non-uniformly scaled parent
POSES = (
    ("root", None, dict(translate=(1, 2, 3), rotate=(10, 20, 30), scale=(1, 2, 3), rotate_order=0)),
    ("hip", "root", dict(
        translate=(4, 5, 6),
        rotate=(40, -50, 60),
        scale=(0.5, 1.5, 1),
        joint_orient=(5, 15, -25),
        rotate_axis=(7, -8, 9),
        rotate_order=3,
        segment_scale_compensate=True,
    )),
    ("knee", "hip", dict(
        translate    = (0, 7, -2),
        rotate       = (-15, 25, 95),
        scale        = (1.2, 0.8, 2.0),
        joint_orient = (-30, 10, 45),
        rotate_axis  = (0, 12, 0),
        rotate_order = 5,
        visibility   = False,
    )),
    ("foot", "knee", dict(
        translate=(1, -3, 2),
        rotate=(0, 0, 33),
        rotate_order=1,
        segment_scale_compensate=True,
        radius=2.5,
        draw_style=2,
    )),
)
FRAMES = 4
BOUND  = 2


def _uuid(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "cgmath/short/" + name)).upper()


def _rig(**user_attributes) -> HierarchyData:
    """the POSES rig, the same uuids on every build; ``user_attributes`` maps
    a node name to the user_defined_attributes it carries"""
    nodes = []
    for name, parent, pose in POSES:
        nodes.append(
            TransformData(
                name,
                node_type="joint",
                uuid=_uuid(name),
                parent_uuid=None if parent is None else _uuid(parent),
                user_defined_attributes=user_attributes.get(name),
                **pose,
            )
        )
    return HierarchyData(nodes)


def _clip() -> ClipData:
    """the POSES rig over FRAMES frames, every framed channel moving, bound
    to frame BOUND"""
    clip  = ClipData(_rig(), frames=FRAMES)
    rng   = np.random.default_rng(7)
    shape = (FRAMES, len(POSES), 3)
    clip.frames.translate    = clip.frames.translate + rng.uniform(-2.0, 2.0, shape)
    clip.frames.rotate       = clip.frames.rotate + rng.uniform(-30.0, 30.0, shape)
    clip.frames.scale        = clip.frames.scale * rng.uniform(0.7, 1.4, shape)
    clip.frames.joint_orient = clip.frames.joint_orient + rng.uniform(-10.0, 10.0, shape)
    clip.frames.rotate_axis  = clip.frames.rotate_axis + rng.uniform(-10.0, 10.0, shape)
    clip.frame               = BOUND
    return clip


def _value(long: str, lead: tuple, seed: int):
    """a value to write ``long`` with, shaped ``lead`` + the channel's shape:
    () for a node, (N,) for a list, (F, N) for a frame axis"""
    rng = np.random.default_rng(seed)
    if long in VECTORS:
        return rng.uniform(0.5, 2.0, lead + (3,))
    if long in COMPONENTS:
        return rng.uniform(0.5, 2.0, lead) if lead else float(rng.uniform(0.5, 2.0))
    if long in ("matrix", "world_matrix") or long in READ_ONLY:
        count = int(np.prod(lead)) if lead else 1
        matrices = [
            TransformData(
                "pose",
                translate = rng.uniform(-5.0, 5.0, 3),
                rotate    = rng.uniform(-90.0, 90.0, 3),
                scale     = rng.uniform(0.5, 2.0, 3),
            ).matrix
            for _ in range(count)
        ]
        return np.array(matrices).reshape(lead + (4, 4))

    picks = {
        "rotate_order": lambda: rng.integers(0, 6, lead).tolist(),
        "segment_scale_compensate": lambda: (rng.integers(0, 2, lead) == 1).tolist(),
        "visibility": lambda: (rng.integers(0, 2, lead) == 1).tolist(),
        "radius": lambda: rng.uniform(0.5, 4.0, lead).tolist(),
        "draw_style": lambda: rng.integers(0, 3, lead).tolist(),
    }
    picked = picks[long]()
    return picked if lead else {"rotate_order": 4, "segment_scale_compensate": True,
                                "visibility": False, "radius": 3.5, "draw_style": 1}[long]


def _same(test, a, b, what: str) -> None:
    """``a`` and ``b`` are the same value, of the same type, bit for bit"""
    test.assertIs(type(a), type(b), what)
    test.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)), what)


# the targets each name is read and written on: a node of a rig, the rig,
# a clip at its bound frame, and the clip's frame axis
TARGETS = {
    "node":   (lambda: _rig(), lambda owner: owner["knee"], ()),
    "rig":    (lambda: _rig(), lambda owner: owner, (len(POSES),)),
    "view":   (lambda: _rig(), lambda owner: owner[1:3], (2,)),
    "clip":   (lambda: _clip(), lambda owner: owner, (len(POSES),)),
    "frames": (lambda: _clip(), lambda owner: owner.frames, (FRAMES, len(POSES))),
}


class TestTheNames(unittest.TestCase):
    def test_they_are_maya_2025s(self):
        self.assertEqual(_SHORT_NAMES, SHORT)
        self.assertEqual(len(SHORT), 31)

    def test_is_has_none(self):
        # inverseScale's short name is a Python keyword: node.is would not parse
        self.assertTrue(keyword.iskeyword("is"))
        self.assertNotIn("is", _SHORT_NAMES)
        self.assertFalse(hasattr(TransformData, "is"))
        with self.assertRaises(AttributeError):
            getattr(TransformData("a"), "is")

    def test_each_class_has_what_its_long_names_allow(self):
        for cls in (TransformData, TransformList, HierarchyData, ClipData, type(_clip().frames)):
            for short, long in SHORT.items():
                with self.subTest(cls=cls.__name__, short=short):
                    held = getattr(cls, long, None)
                    if not isinstance(held, property):
                        self.assertFalse(hasattr(cls, short))
                        continue
                    alias = getattr(cls, short)
                    self.assertIsInstance(alias, property)
                    self.assertEqual(alias.fset is None, held.fset is None)
                    self.assertEqual(alias.fget.__doc__, f"Maya's short name for {long}")

    def test_dir_and_tab_completion_show_them(self):
        clip = _clip()
        for label, target in (
            ("node", clip["knee"]), ("rig", _rig()), ("clip", clip), ("frames", clip.frames),
        ):
            names = set(dir(target))
            for short, long in SHORT.items():
                with self.subTest(target=label, short=short):
                    self.assertEqual(short in names, hasattr(type(target), long))

        node      = _rig()["knee"]
        completer = rlcompleter.Completer({"node": node})
        found     = set()
        for state in range(200):
            match = completer.complete("node.t", state)
            if match is None:
                break
            found.add(match.rstrip("("))
        self.assertTrue({"node.t", "node.tx", "node.ty", "node.tz", "node.translate"} <= found, found)


class TestReadsAndWrites(unittest.TestCase):
    def test_reads_are_the_long_names(self):
        for label, (make, target, _) in TARGETS.items():
            owner = make()
            for short, long in SHORT.items():
                with self.subTest(target=label, short=short):
                    on = target(owner)
                    if not hasattr(type(on), long):
                        # a frame axis has no rotate_order: nor ro, the same error
                        with self.assertRaises(AttributeError) as was:
                            getattr(on, long)
                        with self.assertRaises(AttributeError) as now:
                            getattr(on, short)
                        self.assertEqual(str(now.exception), str(was.exception).replace(repr(long), repr(short)))
                        continue
                    _same(self, getattr(on, short), getattr(on, long), f"{label}.{short}")

    def test_writes_are_the_long_names(self):
        # the same write through each name leaves two equal builds equal,
        # every node, every frame and every saved field compared
        for label, (make, target, lead) in TARGETS.items():
            for seed, (short, long) in enumerate(SHORT.items()):
                with self.subTest(target=label, short=short):
                    by_long, by_short = make(), make()
                    value = _value(long, lead, seed)
                    if not hasattr(type(target(by_long)), long) or long in READ_ONLY:
                        with self.assertRaises(AttributeError) as was:
                            setattr(target(by_long), long, value)
                        with self.assertRaises(AttributeError) as now:
                            setattr(target(by_short), short, value)
                        self.assertEqual(
                            str(now.exception).replace(repr(short), "NAME"),
                            str(was.exception).replace(repr(long), "NAME"),
                        )
                        continue

                    setattr(target(by_long), long, value)
                    setattr(target(by_short), short, value)
                    self.assertEqual(by_short.to_json(), by_long.to_json())
                    if label in ("clip", "frames"):
                        self.assertTrue(np.array_equal(by_short.frames.world_matrix, by_long.frames.world_matrix))
                    else:
                        self.assertTrue(np.array_equal(by_short.world_matrix, by_long.world_matrix))

    def test_a_clip_writes_its_bound_frame_and_the_axis_every_frame(self):
        clip, before = _clip(), _clip()
        clip.tx = [9.0, 8.0, 7.0, 6.0]
        moved   = np.flatnonzero(np.any(clip.frames.translate != before.frames.translate, axis=(1, 2)))
        self.assertEqual(moved.tolist(), [BOUND])
        self.assertEqual(clip.frames.tx[BOUND].tolist(), [9.0, 8.0, 7.0, 6.0])

        clip.frames.ry = 11.0
        self.assertTrue(np.all(clip.frames.rotate_y == 11.0))
        self.assertEqual(clip.ry.tolist(), [11.0] * len(POSES))

    def test_errors_are_the_long_names(self):
        clip = _clip()
        for label, target, long, bad in (
            ("node", _rig()["knee"], "translate", (1.0, 2.0)),
            ("node", _rig()["knee"], "rotate_order", "abc"),
            ("node", _rig()["knee"], "matrix", np.zeros(5)),
            ("node", _rig()["knee"], "world_matrix", "abc"),
            ("node", _rig()["knee"], "scale_x", "abc"),
            ("rig", _rig(), "rotate", np.zeros((4, 2))),
            ("rig", _rig(), "joint_orient", "abc"),
            ("clip", clip, "matrix", np.zeros((4, 3, 3))),
            ("frames", clip.frames, "translate", np.zeros(2)),
            ("frames", clip.frames, "world_matrix", np.zeros((4, 4, 3))),
        ):
            short = {value: key for key, value in SHORT.items()}[long]
            with self.subTest(target=label, short=short):
                with self.assertRaises(Exception) as was:
                    setattr(target, long, bad)
                with self.assertRaises(Exception) as now:
                    setattr(target, short, bad)
                self.assertIs(type(now.exception), type(was.exception))
                self.assertEqual(str(now.exception), str(was.exception))

    def test_an_sdf_primitive_reads_its_own_radius(self):
        # radi reads whatever radius means on the class: a sphere's radius
        sphere      = SDFSphere(radius=2.0, name="ball")
        sphere.radi = 3.0
        self.assertEqual((sphere.radius, sphere.radi), (3.0, 3.0))

    def test_files_keep_the_long_names(self):
        rig             = _rig()
        rig["knee"].t   = (1.0, 2.0, 3.0)
        rig.ro          = [1, 2, 3, 4]
        rig["hip"].radi = 4.0
        clip            = _clip()
        clip.frames.ra  = 3.0
        for tree in (rig.to_dict(), clip.to_dict(), rig["knee"].to_dict()):
            keys  = set()
            stack = [tree]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    keys.update(item)
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
            self.assertEqual(keys & set(SHORT), set())
        self.assertEqual(np.asarray(rig["knee"].to_dict()["translate"]).tolist(), [1.0, 2.0, 3.0])
        self.assertIn("rotate_axis", clip.to_dict()["blocks"])


def _rotation(degrees, order: int) -> np.ndarray:
    """a row vector rotation matrix built here, one axis at a time, in Maya's
    rotate order ``order``: xyz applies x first"""
    out = np.eye(4)
    for axis in ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")[order]:
        k = "xyz".index(axis)
        c, s = np.cos(np.radians(degrees[k])), np.sin(np.radians(degrees[k]))
        i, j = [a for a in range(3) if a != k]
        turn = np.eye(4)
        turn[i, i], turn[i, j], turn[j, i], turn[j, j] = c, s, -s, c
        if k == 1:
            turn[i, j], turn[j, i] = -s, s
        out = out @ turn
    return out


def _composed(channels: dict, nodes) -> tuple:
    """(local, world, parent world) of ``nodes``, (N, 4, 4) each, composed
    here from their (N, 3) ``channels``: S RO R JO IS T, then down the chain"""
    index  = {node.uuid: i for i, node in enumerate(nodes)}
    local  = np.empty((len(nodes), 4, 4))
    world  = np.empty((len(nodes), 4, 4))
    parent = np.empty((len(nodes), 4, 4))
    for i, node in enumerate(nodes):
        up = index.get(node.parent_uuid)
        S  = np.diag(list(channels["scale"][i]) + [1.0])
        IS = np.eye(4)
        if node.segment_scale_compensate and up is not None:
            IS = np.diag(list(1.0 / channels["scale"][up]) + [1.0])
        T         = np.eye(4)
        T[3, :3]  = channels["translate"][i]
        RO        = _rotation(channels["rotate_axis"][i],  0)
        R         = _rotation(channels["rotate"][i],       node.rotate_order)
        JO        = _rotation(channels["joint_orient"][i], 0)
        local[i]  = S @ RO @ R @ JO @ IS @ T
        parent[i] = np.eye(4) if up is None else world[up]
        world[i]  = local[i] @ parent[i]
    return local, world, parent


def _expected(local, world, parent) -> dict:
    return {
        "matrix": local,
        "world_matrix": world,
        "parent_matrix": parent,
        "inverse_matrix": np.linalg.inv(local),
        "world_inverse_matrix": np.linalg.inv(world),
        "parent_inverse_matrix": np.linalg.inv(parent),
    }


class TestTheFourMatrices(unittest.TestCase):
    def assert_close(self, got, want, what):
        self.assertEqual(np.shape(got), np.shape(want), what)
        self.assertLessEqual(float(np.abs(got - want).max()), 1e-12, what)

    def test_a_node_and_a_rig_against_numpy(self):
        rig      = _rig()
        channels = {name: getattr(rig, name) for name in VECTORS}
        want     = _expected(*_composed(channels, rig.list))
        for long, matrices in want.items():
            short = {value: key for key, value in SHORT.items()}[long]
            for i, node in enumerate(rig):
                self.assert_close(getattr(node, long), matrices[i], f"{node.name}.{long}")
                self.assert_close(getattr(node, short), matrices[i], f"{node.name}.{short}")
            self.assert_close(getattr(rig, long), matrices, f"rig.{long}")
            self.assert_close(getattr(rig[1:3], short), matrices[1:3], f"rig[1:3].{short}")

        # a root's parent matrices are identity, exactly
        self.assertTrue(np.array_equal(rig["root"].pm, np.eye(4)))
        self.assertTrue(np.array_equal(rig["root"].pim, np.eye(4)))

    def test_a_clip_against_numpy_every_frame(self):
        clip = _clip()
        for frame in range(FRAMES):
            channels = {name: getattr(clip.frames, name)[frame] for name in VECTORS}
            want     = _expected(*_composed(channels, clip.list))
            for long, matrices in want.items():
                short = {value: key for key, value in SHORT.items()}[long]
                self.assert_close(getattr(clip.frames, long)[frame],          matrices,      f"frames.{long}[{frame}]")
                self.assert_close(getattr(clip.frames, short)[frame],         matrices,      f"frames.{short}[{frame}]")
                self.assert_close(getattr(clip.frames["knee"], short)[frame], matrices[2:3], f"frames['knee'].{short}")
                if frame == BOUND:
                    self.assert_close(getattr(clip, short), matrices, f"clip.{short}")
                    self.assert_close(getattr(clip["foot"], short), matrices[3], f"clip['foot'].{short}")

    def test_a_parent_outside_a_view_or_none_at_all(self):
        rig  = _rig()
        view = rig[[3, 1]]
        self.assert_close(view.pm, np.array([rig["knee"].wm, rig["root"].wm]), "view.pm")
        self.assert_close(view.pim, np.linalg.inv(view.pm), "view.pim")

        loose = TransformList([TransformData("loose", translate=(1.0, 2.0, 3.0), scale=(2.0, 2.0, 2.0))])
        self.assertTrue(np.array_equal(loose.pm, np.eye(4)[None]))
        self.assert_close(loose.wim, np.linalg.inv(loose[0].matrix)[None], "loose.wim")
        self.assertEqual(TransformList().im.shape, (0, 4, 4))

    def test_a_parent_loop_raises_as_world_matrix_does(self):
        # parent_uuids written in place can loop: a node there, or below,
        # has no world matrix, read by itself or in a list
        rig                    = _rig()
        rig["hip"].parent_uuid = rig["foot"].uuid
        with self.assertRaises(RecursionError) as was:
            rig.world_matrix
        for short in ("wm", "pm", "wim", "pim"):
            with self.subTest(short=short):
                with self.assertRaises(RecursionError) as now:
                    getattr(rig, short)
                self.assertEqual(str(now.exception), str(was.exception))
                with self.assertRaises(RecursionError):
                    getattr(rig[2:], short)

        # the local matrices need no parent's world, and a node outside the
        # loop reads as it did
        self.assert_close(rig.im, np.linalg.inv([node.matrix for node in rig]), "rig.im")
        self.assert_close(rig[[0]].wim, rig["root"].wim[None], "rig[[0]].wim")

    def test_loops_are_found_for_every_node_at_once(self):
        # _looped, numpy pointer doubling, against a walk up from each entry:
        # on a loop or below one, for trees, chains and random arrays
        def walked(parents) -> list:
            looped = []
            for start in range(len(parents)):
                seen, i = set(), start
                while i > -1 and i not in seen:
                    seen.add(i)
                    i = int(parents[i])
                looped.append(i > -1)
            return looped

        rng    = np.random.default_rng(5)
        arrays = [np.zeros(0, dtype=int), np.array([-1]), np.array([0]), np.array([-7, 0, 1])]
        for size in (2, 3, 5, 8, 33, 300):
            for _ in range(12):
                # random entries, a forest, and the forest with one entry
                # given another parent, often a loop
                tree    = np.array([int(rng.integers(-1, i)) if i else -1 for i in range(size)])
                cut     = tree.copy()
                at      = int(rng.integers(0, size))
                cut[at] = int(rng.integers(0, size))
                arrays.extend([rng.integers(-1, size, size), tree, cut])
        chain = np.arange(-1, 3999)
        arrays.append(chain)
        arrays.append(np.concatenate([[3999], chain[1:]]))  # the root's parent is the last

        # the array is never iterated
        class Unwalked(np.ndarray):
            def __iter__(self):
                raise AssertionError("walked entry by entry")

        for parents in arrays:
            with self.subTest(parents=parents.tolist()[:12]):
                self.assertEqual(_looped(parents.view(Unwalked)).tolist(), walked(parents))

        # no Python step an entry: the lines Python runs inside _looped, and
        # in what it calls, are a few a pass, log2(n) passes, where a step an
        # entry on the 4000 deep chain runs 4000 at least
        def lines(parents) -> int:
            count = [0]

            def trace(frame, event, arg):
                if event == "line":
                    count[0] += 1
                return trace

            previous = sys.gettrace()
            sys.settrace(trace)
            try:
                _looped(parents)
            finally:
                sys.settrace(previous)
            return count[0]

        for parents in arrays[-2:]:
            with self.subTest(looped=bool(_looped(parents)[0])):
                self.assertLess(lines(parents), len(parents) // 4)

    def test_a_small_view_evaluates_what_its_matrices_are_made_of(self):
        # a view under half its rig evaluates its nodes and every node above
        # them, or for inverse_matrix its nodes and their parents, whose
        # scale a compensating node divides by; it reads what the whole rig
        # evaluated reads, bit for bit, and the matrices composed with numpy
        rng, nodes = np.random.default_rng(11), []
        for i in range(40):
            nodes.append(
                TransformData(
                    f"n{i}",
                    node_type="joint",
                    uuid=_uuid(f"n{i}"),
                    parent_uuid=_uuid(f"n{int(rng.integers(0, i))}") if i else None,
                    translate=rng.uniform(-3.0, 3.0, 3),
                    rotate=rng.uniform(-90.0, 90.0, 3),
                    scale=rng.uniform(0.5, 2.0, 3),
                    joint_orient=rng.uniform(-30.0, 30.0, 3),
                    rotate_axis=rng.uniform(-30.0, 30.0, 3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(i % 2),
                )
            )
        rig      = HierarchyData(nodes)
        channels = {name: getattr(rig, name) for name in VECTORS}
        want     = _expected(*_composed(channels, rig.list))

        sizes = []

        def whole(picked, owner, worlds):
            return None

        def needed(picks, worlds: bool) -> int:
            # walked here: each pick and every node above it, or each pick
            # and its parent
            met = set()
            for i in picks:
                node = rig[i]
                for _ in range(40 if worlds else 2):
                    if not isinstance(node, TransformData):
                        break
                    met.add(node.name)
                    node = node.get_parent()
            return len(met)

        self.assertGreater(needed([39], True), 3)  # n39 is deep in the rig
        self._counted(sizes)
        for picks in ([39], [39, 0], [7, 3, 1, 0], rng.choice(40, 19, replace=False).tolist()):
            view = rig[picks]
            for long in READ_ONLY:
                with self.subTest(picks=picks, long=long):
                    sizes.clear()
                    got = getattr(view, long)
                    self.assertEqual(sizes, [needed(picks, long != "inverse_matrix")])
                    with mock.patch.object(hierarchy_mod, "_pose_members", whole):
                        self.assertTrue(np.array_equal(got, getattr(view, long)))
                    self.assertEqual(sizes[-1], 40)
                    self.assert_close(got, want[long][picks], f"rig[{picks}].{long}")

        # half the rig or more evaluates the whole rig; a clip's view reads
        # its bound frame
        sizes.clear()
        rig[:20].wim
        self.assertEqual(sizes, [40])
        clip               = ClipData(rig, frames=3)
        clip.frame         = 1
        clip.frames.rotate = clip.frames.rotate + rng.uniform(-20.0, 20.0, (3, 40, 3))
        for long in READ_ONLY:
            self.assertTrue(np.array_equal(getattr(clip[[39, 5]], long), getattr(clip.frames, long)[1, [39, 5]]))

        # a small view on a loop raises what its node's world_matrix raises,
        # and reads its inverse_matrix; one above the loop reads
        top = rig["n39"]
        while top.get_parent().name != "n0":
            top = top.get_parent()
        top.parent_uuid = rig["n39"].uuid
        for name in ("n39", top.name):
            with self.assertRaises(RecursionError) as was:
                rig[name].world_matrix
            for short in ("pm", "wim", "pim"):
                with self.subTest(name=name, short=short):
                    with self.assertRaises(RecursionError) as now:
                        getattr(TransformList([rig[name]]), short)
                    self.assertEqual(str(now.exception), str(was.exception))
            alone = TransformList([rig[name]]).im
            self.assert_close(alone, rig[name].im[None], f"TransformList([rig[{name!r}]]).im")
        self.assert_close(rig[[0]].wim, want["world_inverse_matrix"][:1], "rig[[0]].wim")

    def _assert_node_by_node(self, view, sizes, counted: dict):
        """each of the four matrices of ``view`` reads what its nodes read,
        bit for bit, evaluating ``counted[long]`` nodes a rig"""
        for long in READ_ONLY:
            with self.subTest(view=len(view), long=long):
                sizes.clear()
                got = getattr(view, long)
                self.assertEqual(sizes, counted[long])
                self.assertTrue(np.array_equal(got, np.array([getattr(node, long) for node in view])))

    def _counted(self, sizes: list):
        """_PoseFrame counting the nodes each evaluation is made of"""

        class Counted(hierarchy_mod._PoseFrame):
            __slots__ = ()

            def __init__(self, members):
                sizes.append(len(members))
                super().__init__(members)

        patch = mock.patch.object(hierarchy_mod, "_PoseFrame", Counted)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_rig_listing_another_rigs_node(self):
        # a rig listing another rig's node in its list, rig[i] = node, with a
        # child under it: above that node are its own rig's nodes, as
        # get_parent has them. A small view evaluates its nodes and those
        # above them, the other rig's included, the rig evaluates its own
        # nodes and the other rig's above them, and every read is what the
        # nodes read
        def chain(tag: str) -> HierarchyData:
            return HierarchyData([
                TransformData(
                    f"{tag}{i}",
                    uuid        = _uuid(f"{tag}{i}"),
                    parent_uuid = _uuid(f"{tag}{i - 1}") if i else None,
                    translate   = (1.0 + i, 0.5, 0.0),
                    rotate      = (0.0, 10.0 * i, 3.0),
                    scale       = (1.0, 1.0 + 0.1 * i, 1.0),
                )
                for i in range(12)
            ])

        rig, other = chain("a"), chain("b")
        rig[5]                = other[3]
        rig["a6"].parent_uuid = other["b3"].uuid
        self.assertIs(rig["a6"].get_parent(), other["b3"])
        self.assertIs(other["b3"].get_parent(), other["b2"])

        sizes = []
        self._counted(sizes)
        # a7 a6 b3 b2 b1 b0 a9 a8, or a7 a9 and their parents a6 a8
        self._assert_node_by_node(
            rig[[7, 9]], sizes, {long: [4 if long == "inverse_matrix" else 8] for long in READ_ONLY}
        )
        # rig's own 11 nodes: the 12 it lists and b2 b1 b0; b3 in other:
        # b3 b2 b1 b0, or b3 and its parent b2
        self._assert_node_by_node(
            rig, sizes, {long: [15, 2 if long == "inverse_matrix" else 4] for long in READ_ONLY}
        )
        self.assertLessEqual(float(np.abs(rig.wim - np.linalg.inv(rig.wm)).max()), 1e-12)
        self.assertTrue(np.array_equal(rig.pm, rig.get_parent_matrix()))
        # the world matrices clip pose algebra reads, the whole rig's and a
        # few nodes'
        worlds = np.array([node.world_matrix for node in rig])
        self.assertTrue(np.array_equal(rig.world_matrix, worlds))
        self.assertTrue(np.array_equal(ClipData._pose_world(list(rig)), worlds))
        self.assertTrue(np.array_equal(ClipData._pose_world([rig[7], rig[9]]), worlds[[7, 9]]))

    def test_two_nodes_of_one_uuid(self):
        # a uuid written in place onto a second node, listed before the one
        # it was: a parent_uuid naming it names the first, as get_parent has
        # it, whose scale a compensating child divides by. A small view
        # evaluates its nodes and those above them, and every read, and the
        # world matrices clip pose algebra reads, are what the nodes read
        nodes = [
            TransformData(
                f"n{i}",
                node_type="joint",
                uuid=_uuid(f"n{i}"),
                parent_uuid=_uuid(f"n{i - 1}") if i else None,
                translate=(1.0 + i, 0.5, 0.0),
                rotate=(0.0, 10.0 * i, 3.0),
                scale=(1.0, 1.0 + 0.1 * i, 1.0),
                segment_scale_compensate=True,
            )
            for i in range(12)
        ]
        nodes.insert(2, TransformData("twin", node_type="joint", translate=(9.0, 9.0, 9.0), scale=(3.0, 2.0, 1.0)))
        rig              = HierarchyData(nodes)
        rig["twin"].uuid = rig["n4"].uuid
        self.assertIs(rig["n5"].get_parent(), rig["twin"])
        self.assertEqual(rig["n5"].parent_scale_inverse.tolist(), [3.0, 2.0, 1.0])

        sizes = []
        self._counted(sizes)
        self.assertEqual(rig[[6, 8, 11]].name, ["n5", "n7", "n10"])
        # n5 twin, or n5 and its parent twin
        self._assert_node_by_node(rig[[6]], sizes, {long: [2] for long in READ_ONLY})
        # n7 n6 n5 twin n10 n9 n8, or n7 n10 and their parents n6 n9
        self._assert_node_by_node(
            rig[[8, 11]], sizes, {long: [4 if long == "inverse_matrix" else 7] for long in READ_ONLY}
        )
        # both carriers: n5 twin n4 n3 n2 n1 n0, or n5 n4 and their parents
        # twin n3
        self._assert_node_by_node(
            rig[[6, 5]], sizes, {long: [4 if long == "inverse_matrix" else 7] for long in READ_ONLY}
        )
        self._assert_node_by_node(rig, sizes, {long: [13] for long in READ_ONLY})
        self.assertTrue(np.array_equal(ClipData._pose_world(list(rig)), rig.world_matrix))

    def test_a_list_or_a_clip_inverts_in_one_batch(self):
        # no node's own matrix read, one inverse call for the whole list or
        # every frame of a clip
        def unread(self):
            raise AssertionError("a node's own matrix was read")

        rig, clip = _rig(), _clip()
        patched = [
            mock.patch.object(TransformData, name, property(unread))
            for name in ("matrix", "world_matrix", "parent_matrix", "inverse_matrix",
                         "world_inverse_matrix", "parent_inverse_matrix")
        ] + [mock.patch.object(TransformData, "get_parent_matrix", unread)]
        for patch in patched:
            patch.start()
            self.addCleanup(patch.stop)

        for label, target in (("rig", rig), ("view", rig[1:]), ("clip", clip), ("frames", clip.frames)):
            for short in ("pm", "im", "wim", "pim"):
                with self.subTest(target=label, short=short):
                    with mock.patch.object(np.linalg, "inv", wraps=np.linalg.inv) as inverse:
                        getattr(target, short)
                    self.assertEqual(inverse.call_count, 0 if short == "pm" else 1)

    def test_a_zero_scale_has_no_inverse(self):
        rig                = _rig()
        rig["hip"].scale_y = 0.0
        for target, short, named in (
            (rig["hip"], "im", "the matrix of 'hip'"),
            (rig["hip"], "wim", "the world_matrix of 'hip'"),
            (rig["knee"], "pim", "the parent_matrix of 'knee'"),
            (rig, "im", "the matrix of 'hip'"),
            (rig[2:], "pim", "the parent_matrix of 'knee'"),
        ):
            with self.subTest(target=str(target), short=short):
                with self.assertRaises(np.linalg.LinAlgError) as caught:
                    getattr(target, short)
                self.assertEqual(str(caught.exception), f"{named} has no inverse: a zero scale flattens it")

        clip                       = _clip()
        clip.frames["hip"].scale_x = [1.0, 1.0, 1.0, 0.0]
        with self.assertRaises(np.linalg.LinAlgError) as caught:
            clip.frames.im
        self.assertEqual(
            str(caught.exception), "the matrix of 'hip' on frame 3 has no inverse: a zero scale flattens it"
        )
        clip.frames.match("root", "knee", "foot").im  # the others invert

    def test_maya_itself(self):
        """Maya 2025's own joints, posed as POSES with inverseScale wired:
        the short names Maya's attributeQuery gives, every channel read by
        its short name, and m, wm, pm, im, wim and pim, to 1e-12. Needs
        mayapy"""
        try:
            import maya.cmds as cmds

            cmds.createNode("transform", name="cgmath_probe")
        except Exception:  # not running in Maya
            self.skipTest("needs Maya")
        self.addCleanup(cmds.file, new=True, force=True)
        cmds.file(new=True, force=True)

        rig = _rig()
        for name, parent, pose in POSES:
            joint = cmds.createNode("joint", name=name, **({"parent": parent} if parent else {}))
            if parent:
                cmds.connectAttr(parent + ".scale", joint + ".inverseScale")
            node = rig[name]
            for attr in ("t", "r", "s", "jo", "ra"):
                cmds.setAttr(f"{joint}.{attr}", *getattr(node, attr).tolist())
            for attr in ("ro", "ssc", "v", "radi", "ds"):
                cmds.setAttr(f"{joint}.{attr}", getattr(node, attr))

        for short, long, maya_long in MAYA_NAMES + (("is", None, "inverseScale"),):
            with self.subTest(short=short):
                self.assertEqual(cmds.attributeQuery(maya_long, node="knee", shortName=True), short)

        plugs = {"m": "m", "wm": "wm[0]", "pm": "pm[0]", "im": "im", "wim": "wim[0]", "pim": "pim[0]"}
        for name, _, _ in POSES:
            node = rig[name]
            for short, long, _ in MAYA_NAMES:
                with self.subTest(node=name, short=short):
                    maya = np.array(cmds.getAttr(f"{name}.{plugs.get(short, short)}"), dtype=float)
                    mine = np.asarray(getattr(node, short), dtype=float)
                    self.assertLessEqual(float(np.abs(mine - maya.reshape(mine.shape)).max()), 1e-12)

    def test_maya_wires_inverse_scale_from_a_joint_alone(self):
        """parent_scale_inverse is the parent's scale, whatever the parent;
        Maya's inverseScale holds it where Maya wires it from the parent
        joint. A joint drawn under a transform, or made with createNode -p
        under a joint, holds (1, 1, 1): compensating there, it has Maya's
        matrices with segment_scale_compensate off. One moved under a
        transform with cmds.parent holds, unwired, the scale of the joint it
        left, which Maya's m divides by: knee, moved under group, scaled
        unlike hip, lands under a transform Maya inserts, where neither
        setting gives Maya's m, and wrist, moved right under sleeve, scaled
        as elbow, the joint it left, has Maya's matrices either way. With
        relative=True a joint lands right under a transform or a joint,
        still wired from the joint it left; an absolute cmds.parent wires it
        where it leaves it right under a joint, and under a joint elsewhere
        puts a transform between them, unwired. Needs mayapy"""
        try:
            import maya.cmds as cmds

            cmds.createNode("transform", name="cgmath_probe")
        except Exception:  # not running in Maya
            self.skipTest("needs Maya")
        self.addCleanup(cmds.file, new=True, force=True)
        cmds.file(new=True, force=True)

        cmds.createNode("transform", name="group")
        cmds.select("group")
        cmds.joint(name="hip")
        cmds.joint(name="knee")
        cmds.setAttr("group.s", 2.0, 3.0, 4.0)
        cmds.setAttr("hip.s", 5.0, 6.0, 7.0)
        self.assertEqual(cmds.listRelatives("hip", parent=True), ["group"])
        self.assertEqual(cmds.listRelatives("knee", parent=True), ["hip"])
        self.assertIsNone(cmds.listConnections("hip.is", source=True, destination=False))
        self.assertEqual(cmds.getAttr("hip.is")[0], (1.0, 1.0, 1.0))
        self.assertEqual(
            cmds.listConnections("knee.is", source=True, destination=False, plugs=True), ["hip.scale"]
        )
        self.assertEqual(cmds.getAttr("knee.is")[0], (5.0, 6.0, 7.0))
        self.assertTrue(cmds.getAttr("hip.ssc"))

        rig = HierarchyData([TransformData("group", scale=(2.0, 3.0, 4.0))])
        rig.append(TransformData("hip", node_type="joint", scale=(5.0, 6.0, 7.0), parent_uuid=rig["group"].uuid))
        rig.append(TransformData("knee", node_type="joint", parent_uuid=rig["hip"].uuid))
        self.assertEqual(rig["hip"].parent_scale_inverse.tolist(), [2.0, 3.0, 4.0])
        self.assertEqual(rig["knee"].parent_scale_inverse.tolist(), [5.0, 6.0, 7.0])

        def gap(name: str, short: str) -> float:
            maya = np.array(cmds.getAttr(f"{name}.{short}" + ("[0]" if short != "m" else "")))
            return float(np.abs(getattr(rig[name], short) - maya.reshape(4, 4)).max())

        rig.ssc = True
        self.assertGreater(gap("hip", "m"), 0.5)
        rig["hip"].ssc = False
        for name in ("hip", "knee"):
            for short in ("m", "wm", "pm", "wim"):
                with self.subTest(name=name, short=short):
                    self.assertLessEqual(gap(name, short), 1e-12)

        # a joint made with createNode -p under a joint: unwired, at 1, so
        # compensating it has Maya's matrices with segment_scale_compensate
        # off, as under a transform
        cmds.createNode("joint", name="shin", parent="hip")
        self.assertIsNone(cmds.listConnections("shin.is", source=True, destination=False))
        self.assertEqual(cmds.getAttr("shin.is")[0], (1.0, 1.0, 1.0))
        self.assertTrue(cmds.getAttr("shin.ssc"))
        rig.append(TransformData("shin", node_type="joint", parent_uuid=rig["hip"].uuid))
        rig["shin"].ssc = True
        self.assertGreater(gap("shin", "m"), 0.5)
        rig["shin"].ssc = False
        for short in ("m", "wm", "pm", "wim"):
            with self.subTest(name="shin", short=short):
                self.assertLessEqual(gap("shin", short), 1e-12)

        # knee moved under the transform: Maya puts it under a transform,
        # unwires its inverseScale and leaves hip's scale there, which its
        # m divides by, compensating
        cmds.parent("knee", "group")
        above = cmds.listRelatives("knee", parent=True)
        self.assertEqual(cmds.nodeType(above[0]), "transform")
        self.assertEqual(cmds.listRelatives(above[0], parent=True), ["group"])
        self.assertIsNone(cmds.listConnections("knee.is", source=True, destination=False))
        self.assertEqual(cmds.getAttr("knee.is")[0], (5.0, 6.0, 7.0))
        self.assertTrue(cmds.getAttr("knee.ssc"))
        maya = np.array(cmds.getAttr("knee.m")).reshape(4, 4)[:3, :3]
        self.assertLessEqual(float(np.abs(maya - np.diag([1 / 5, 1 / 6, 1 / 7])).max()), 1e-12)
        moved = HierarchyData([TransformData("group", scale=(2.0, 3.0, 4.0))])
        moved.append(TransformData("knee", node_type="joint", parent_uuid=moved["group"].uuid))
        for compensate in (True, False):
            with self.subTest(compensate=compensate):
                moved["knee"].ssc = compensate
                self.assertGreater(float(np.abs(moved["knee"].m[:3, :3] - maya).max()), 0.25)

        # moved with cmds.parent -relative, a joint lands right under its new
        # parent, a transform or a joint, wired as it was: from the joint it
        # left, or, a root, not at all
        for name, drawn, under, wired in (
            ("heel", "hip", "group", ["hip.scale"]),
            ("toe", "hip", "shin", ["hip.scale"]),
            ("stray", None, "shin", None),
        ):
            with self.subTest(name=name, under=under):
                cmds.select(clear=True)
                if drawn:
                    cmds.select(drawn)
                cmds.joint(name=name)
                cmds.parent(name, under, relative=True)
                self.assertEqual(cmds.listRelatives(name, parent=True), [under])
                self.assertEqual(
                    cmds.listConnections(name + ".is", source=True, destination=False, plugs=True), wired
                )

        # an absolute cmds.parent wires it where it leaves a joint right under
        # a joint, as under ankle, a root joint unscaled, and under a joint
        # elsewhere puts a transform between them, unwired, as under hip,
        # scaled
        for name in ("ankle", "loose"):
            cmds.select(clear=True)
            cmds.joint(name=name)
        cmds.parent("loose", "ankle")
        self.assertEqual(cmds.listRelatives("loose", parent=True), ["ankle"])
        self.assertEqual(
            cmds.listConnections("loose.is", source=True, destination=False, plugs=True), ["ankle.scale"]
        )
        cmds.parent("loose", "hip")
        above = cmds.listRelatives("loose", parent=True)
        self.assertEqual(cmds.nodeType(above[0]), "transform")
        self.assertEqual(cmds.listRelatives(above[0], parent=True), ["hip"])
        self.assertIsNone(cmds.listConnections("loose.is", source=True, destination=False))

        # and under a transform it can leave the joint right under it,
        # unwired, holding the scale of the joint it left: wrist, drawn under
        # elbow, moved under sleeve, both scaled (5, 6, 7), where cgmath's
        # parent_scale_inverse is sleeve's scale, what Maya's inverseScale
        # holds, and its m and wm, compensating or not, are Maya's
        cmds.select(clear=True)
        cmds.joint(name="elbow")
        cmds.joint(name="wrist", position=(1.0, 2.0, 3.0))
        cmds.setAttr("elbow.s", 5.0, 6.0, 7.0)
        cmds.createNode("transform", name="sleeve")
        cmds.setAttr("sleeve.s", 5.0, 6.0, 7.0)
        cmds.parent("wrist", "sleeve")
        self.assertEqual(cmds.listRelatives("wrist", parent=True), ["sleeve"])
        self.assertIsNone(cmds.listConnections("wrist.is", source=True, destination=False))
        self.assertEqual(cmds.getAttr("wrist.is")[0], (5.0, 6.0, 7.0))
        sleeve = HierarchyData([TransformData("sleeve", scale=(5.0, 6.0, 7.0))])
        wrist = TransformData(
            "wrist",
            node_type    = "joint",
            translate    = cmds.getAttr("wrist.t")[0],
            rotate       = cmds.getAttr("wrist.r")[0],
            scale        = cmds.getAttr("wrist.s")[0],
            joint_orient = cmds.getAttr("wrist.jo")[0],
            parent_uuid  = sleeve["sleeve"].uuid,
        )
        sleeve.append(wrist)
        self.assertEqual(sleeve["wrist"].parent_scale_inverse.tolist(), [5.0, 6.0, 7.0])
        for compensate in (True, False):
            cmds.setAttr("wrist.ssc", compensate)
            sleeve["wrist"].ssc = compensate
            for short in ("m", "wm"):
                with self.subTest(compensate=compensate, short=short):
                    maya = np.array(cmds.getAttr("wrist." + short + ("[0]" if short != "m" else "")))
                    mine = getattr(sleeve["wrist"], short)
                    self.assertLessEqual(float(np.abs(mine - maya.reshape(4, 4)).max()), 1e-12)


def _spec(value: float) -> dict:
    return {"attributeType": "double", "value": value, "keyable": True, "channel_box": False}


class TestUserAttributes(unittest.TestCase):
    """a user attribute can not take a name a built-in took; one a file
    carries anyway loads, hidden, with one warning a name"""

    TAKEN = tuple(SHORT) + READ_ONLY

    def test_add_user_attribute_refuses_every_name(self):
        rig = _rig()
        for name in self.TAKEN:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    rig["knee"].add_user_attribute(name, 1.0)
                hint = f", Maya's short name for {SHORT[name]}" if name in SHORT else ""
                self.assertEqual(str(caught.exception), f"{name!r} is a built-in attribute of TransformData{hint}")
                with self.assertRaises(ValueError):
                    rig.add_user_attribute(name, 1.0)
                with self.assertRaises(ValueError):
                    rig[1:].add_user_attribute(name, 1.0)
        self.assertEqual([node.user_defined_attributes for node in rig], [{}] * len(POSES))

    def folder(self) -> str:
        """a folder of this test's own, gone once it ends"""
        made = tempfile.mkdtemp(prefix="short_names_")
        self.addCleanup(shutil.rmtree, made, ignore_errors=True)
        return made

    def hidden(self, caught) -> list:
        """the messages of the warnings in ``caught`` of a hidden user
        attribute, each checked to be filed under this file"""
        found = [w for w in caught if "user_defined_attributes[" in str(w.message)]
        for w in found:
            self.assertIs(w.category, UserWarning)
            self.assertEqual(os.path.basename(w.filename), os.path.basename(__file__))
        return [str(w.message) for w in found]

    WARNED = [
        "user attribute 'tx' of 'hip' and 1 other node loaded into user_defined_attributes['tx']: "
        "node.tx is Maya's short name for translate_x, which hides it",
        "user attribute 'wm' of 'knee' and 1 other node loaded into user_defined_attributes['wm']: "
        "node.wm is Maya's short name for world_matrix, which hides it",
        "user attribute 'parent_matrix' of 'foot' loaded into user_defined_attributes['parent_matrix']: "
        "node.parent_matrix is a built-in, which hides it",
    ]

    def carrying(self) -> HierarchyData:
        """the rig, two nodes carrying a user 'tx', two a 'wm', one a
        'parent_matrix' and a 'wm_extra', which nothing hides, and the root
        a user attribute of its own"""
        return _rig(
            root = {"heroHeight": _spec(1.8)},
            hip  = {"tx": _spec(7.0)},
            knee = {"wm": _spec(1.0), "tx": _spec(8.0)},
            foot = {"wm": _spec(2.0), "parent_matrix": _spec(3.0), "wm_extra": _spec(4.0)},
        )

    def test_a_saved_rig_loads_them_with_one_warning_a_name(self):
        rig                                      = self.carrying()
        rig["hip"].user_defined_attributes["wm"] = _spec(5.0)
        folder                                   = self.folder()
        for mode in ("json", "npz", "pkl"):
            path = rig.save(os.path.join(folder, "rig." + mode))
            for loader in (HierarchyData.load, Data.load, hierarchy.load):
                with self.subTest(mode=mode, loader=loader.__qualname__):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        loaded = loader(path)
                    self.assertEqual(
                        self.hidden(caught),
                        [
                            self.WARNED[0],
                            self.WARNED[1].replace("'knee' and 1 other node", "'hip' and 2 other nodes"),
                            self.WARNED[2],
                        ],
                    )

                    knee = loaded["knee"]
                    self.assertEqual(knee.tx, knee.translate_x)
                    self.assertTrue(np.array_equal(knee.wm, knee.world_matrix))
                    self.assertEqual(knee.user_defined_attributes["tx"]["value"], 8.0)
                    self.assertTrue(np.array_equal(loaded.tx, loaded.translate_x))
                    self.assertEqual(loaded.heroHeight, [1.8, None, None, None])

                    # node.tx writes translate_x, never the user attribute,
                    # and a save keeps the user attribute as it came
                    knee.tx = 12.5
                    self.assertEqual(knee.translate[0], 12.5)
                    self.assertEqual(knee.user_defined_attributes["tx"]["value"], 8.0)
                    self.assertEqual(
                        loaded.to_dict()["__items__"][2]["user_defined_attributes"]["tx"]["value"], 8.0
                    )

    def test_a_clip_a_list_and_a_lone_node_warn_once_a_name(self):
        rig    = self.carrying()
        clip   = ClipData(rig, frames=3)
        folder = self.folder()
        for label, load in (
            ("clip", lambda: ClipData.load(clip.save(os.path.join(folder, "clip.json")))),
            ("pickle", lambda: pickle.loads(pickle.dumps(rig))),
            ("list", lambda: TransformList.from_dict({node.uuid: node.to_dict() for node in rig})),
            ("from_bytes", lambda: HierarchyData.from_bytes(rig.to_bytes())),
        ):
            with self.subTest(load=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    load()
                self.assertEqual(self.hidden(caught), self.WARNED)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            node = TransformData.load(rig["knee"].save(os.path.join(folder, "knee.json")))
        self.assertEqual(
            self.hidden(caught),
            [
                "user attribute 'wm' of 'knee' loaded into user_defined_attributes['wm']: "
                "node.wm is Maya's short name for world_matrix, which hides it",
                "user attribute 'tx' of 'knee' loaded into user_defined_attributes['tx']: "
                "node.tx is Maya's short name for translate_x, which hides it",
            ],
        )
        self.assertEqual(node.tx, node.translate_x)

    def test_a_warning_finds_its_caller_once_a_file(self):
        # the warning is filed under the first caller outside cgmath, known by
        # the file each frame runs in: the answer for each file is kept, so a
        # loop of loads that warn works each out once. Each count starts with
        # no file known, whatever ran before it
        absolute = os.path.abspath
        answers  = hierarchy_mod._OUTSIDE

        def made(count):
            paths = []

            def counted(path):
                paths.append(path)
                return absolute(path)

            with warnings.catch_warnings(), mock.patch.object(os.path, "abspath", counted), mock.patch.dict(
                answers, clear=True
            ):
                warnings.simplefilter("ignore")
                for _ in range(count):
                    node = TransformData.from_dict({"name": "x", "user_defined_attributes": {"tx": _spec(1.0)}})
            self.assertEqual(node.user_defined_attributes["tx"]["value"], 1.0)
            return len(paths)

        small, large = made(4), made(16)
        self.assertGreater(small, 0)
        self.assertEqual(small, large)

    def test_a_clip_with_a_rest_pose_warns_once_a_name(self):
        # the rest pose loads inside the clip's load, and its nodes, the
        # clip's again by their uuids, count once: the warnings a clip with
        # no rest pose gives
        rig            = self.carrying()
        clip           = ClipData(rig, frames=3)
        clip.rest_pose = rig.copy()
        folder         = self.folder()
        loaders        = {"ClipData.load": ClipData.load, "Data.load": Data.load, "hierarchy.load": hierarchy.load}
        loads          = [("pickle", lambda: pickle.loads(pickle.dumps(clip)))]
        loads.append(("from_bytes", lambda: ClipData.from_bytes(clip.to_bytes())))
        for mode in ("json", "npz", "pkl"):
            path = clip.save(os.path.join(folder, "clip." + mode))
            for name, loader in loaders.items():
                loads.append((f"{mode} {name}", lambda path=path, loader=loader: loader(path)))
        for label, load in loads:
            with self.subTest(load=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded = load()
                self.assertEqual(self.hidden(caught), self.WARNED)
                knee = loaded.rest_pose["knee"]
                self.assertEqual(knee.tx, knee.translate_x)
                self.assertEqual(knee.user_defined_attributes["tx"]["value"], 8.0)

        # a rest pose node carrying one the clip's node does not counts
        clip.rest_pose["root"].user_defined_attributes["tx"] = _spec(9.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ClipData.from_bytes(clip.to_bytes())
        self.assertEqual(
            self.hidden(caught),
            [self.WARNED[0].replace("1 other node", "2 other nodes")] + self.WARNED[1:],
        )

    def test_a_load_that_raises_warns_of_nothing(self):
        # the warnings come once the load is done: a clip whose blocks are
        # the wrong shape raises, and its nodes warn of nothing
        tree = ClipData(self.carrying(), frames=3).to_dict()
        # a translate block of 2 nodes, where the clip has 4
        tree["blocks"]["translate"] = np.zeros((3, 2, 3)).tolist()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                ClipData.from_dict(tree)
        self.assertEqual(self.hidden(caught), [])

    def test_a_copy_warns_of_nothing(self):
        rig  = self.carrying()
        clip = ClipData(rig, frames=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copies = [
                rig.copy(), rig[1:].copy(), rig["knee"].copy(), copy.deepcopy(rig), copy.copy(rig),
                clip.copy(), ClipData(rig, frames=2), clip.frames[0:2], rig.get_delta(_rig()),
            ]
        self.assertEqual(self.hidden(caught), [])
        self.assertEqual(copies[0]["knee"].user_defined_attributes["tx"]["value"], 8.0)

    def test_a_built_rig_warns_of_nothing(self):
        # only a load warns: a rig built in code, the dicts handed over, does not
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rig = self.carrying()
        self.assertEqual(self.hidden(caught), [])
        self.assertEqual(rig["hip"].tx, 4.0)

    @staticmethod
    def rewritten(static: str, path: str, edit) -> str:
        """``static``, an fbx file, written to ``path`` once ``edit(scene)``
        has changed its scene"""
        from cgmath.formats._fbx_io import import_scene, new_manager

        manager = new_manager(FBX)
        try:
            scene = import_scene(manager, static, FBX)
            edit(scene)
            exporter = FBX.FbxExporter.Create(manager, "")
            if not exporter.Initialize(path, -1, manager.GetIOSettings()):
                raise RuntimeError(f"cannot write {path}")
            exporter.Export(scene)
            exporter.Destroy()
        finally:
            manager.Destroy()
        return path

    @classmethod
    def keyed(cls, static: str, path: str) -> str:
        """``static``, an fbx file, written to ``path`` with a take keying
        hip's rotate y over 4 frames"""
        from cgmath.formats._fbx_io import typed

        def edit(scene) -> None:
            hip   = typed(scene.FindNodeByName("hip"), FBX.FbxNode, FBX)
            stack = FBX.FbxAnimStack.Create(scene, "Take 001")
            layer = FBX.FbxAnimLayer.Create(scene, "Base Layer")
            stack.AddMember(layer)
            hip.LclRotation.GetCurveNode(layer, True)
            curve = hip.LclRotation.GetCurve(layer, "Y", True)
            curve.KeyModifyBegin()
            for frame in range(4):
                time = FBX.FbxTime()
                time.SetSecondDouble(frame / 24.0)
                curve.KeySetValue(curve.KeyAdd(time)[0], 10.0 * frame)
            curve.KeyModifyEnd()

        return cls.rewritten(static, path, edit)

    @classmethod
    def bound(cls, static: str, path: str, scale: float) -> str:
        """``static``, an fbx file, written to ``path`` with a bind pose
        listing hip at a world scaled by ``scale`` and knee at one unscaled,
        both at (1, 2, 3)"""
        from cgmath.formats._fbx_io import typed

        def edit(scene) -> None:
            pose = FBX.FbxPose.Create(scene, "bindPose1")
            pose.SetIsBindPose(True)
            for name, scaled in (("hip", scale), ("knee", 1.0)):
                world        = np.diag([scaled, scaled, scaled, 1.0])
                world[3, :3] = (1.0, 2.0, 3.0)
                matrix       = FBX.FbxMatrix()
                for row in range(4):
                    matrix.SetRow(row, FBX.FbxVector4(*world[row].tolist()))
                pose.Add(typed(scene.FindNodeByName(name), FBX.FbxNode, FBX), matrix)
            scene.AddPose(pose)

        return cls.rewritten(static, path, edit)

    @unittest.skipIf(FBX is None, "needs the Autodesk FBX SDK")
    def test_an_fbx_file_loads_them_with_one_warning_a_name(self):
        # a clip's rig and rest pose, both built from the file's nodes, are
        # one load, with a take as without one, and a bind read poses the
        # nodes inside their load
        rig    = _rig(hip={"tx": _spec(7.0)}, knee={"tx": _spec(8.0), "v": _spec(0.5)})
        folder = self.folder()
        path   = os.path.join(folder, "rig.fbx")
        rig.save_fbx(path)
        taken = self.keyed(path, os.path.join(folder, "take.fbx"))
        bound = self.bound(path, os.path.join(folder, "bind.fbx"), 1.0)
        bind  = {"pose": "bind"}
        loads = [(f"{owner.__name__}.load_fbx", owner.load_fbx, path, {}) for owner in (HierarchyData, TransformList)]
        loads.append(("hierarchy.load_fbx", hierarchy.load_fbx, path, {}))
        loads.append(("ClipData.load_fbx", ClipData.load_fbx, path, {}))
        loads.append(("ClipData.load_fbx, a take", ClipData.load_fbx, taken, {}))
        for owner in (HierarchyData, TransformList):
            loads.append((f"{owner.__name__}.load_fbx, a bind pose", owner.load_fbx, bound, bind))
        for label, loader, file, kwargs in loads:
            with self.subTest(loader=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded = loader(file, **kwargs)
                self.assertEqual(
                    self.hidden(caught),
                    [
                        "user attribute 'tx' of 'hip' and 1 other node loaded into "
                        "user_defined_attributes['tx']: node.tx is Maya's short name for "
                        "translate_x, which hides it",
                        "user attribute 'v' of 'knee' loaded into user_defined_attributes['v']: "
                        "node.v is Maya's short name for visibility, which hides it",
                    ],
                )
                rigs = [loaded] + ([loaded.rest_pose] if isinstance(loaded, ClipData) else [])
                for knee in (each[2] for each in rigs):
                    self.assertEqual(knee.name, "knee")
                    self.assertEqual(knee.tx, knee.translate_x)
                    self.assertIs(knee.v, knee.visibility)
                    self.assertEqual(knee.user_defined_attributes["tx"]["value"], 8.0)
                    self.assertEqual(knee.user_defined_attributes["v"]["value"], 0.5)
                if file is taken:
                    self.assertGreater(loaded.frame_count, 1)

        # the warnings come once the file is read and its nodes posed: a
        # take out of range, a frame past or before the take's, or a bind
        # under hip's, which a zero scale flattens, raises, and the nodes
        # warn of nothing
        flat = self.bound(path, os.path.join(folder, "flat.fbx"), 0.0)
        for loader, file, kwargs, error, message in (
            (ClipData.load_fbx, taken, {"take": 5}, IndexError, "take 5 out of range"),
            (ClipData.load_fbx, taken, {"take": -2}, IndexError, "take -2 out of range"),
            (ClipData.load_fbx, taken, {"start_frame": 100}, ValueError, "start_frame 100 is past"),
            (ClipData.load_fbx, taken, {"end_frame": -100}, ValueError, "end_frame -100 is before"),
            (HierarchyData.load_fbx, flat, bind, np.linalg.LinAlgError, ""),
            (TransformList.load_fbx, flat, bind, np.linalg.LinAlgError, ""),
        ):
            label = f"{loader.__self__.__name__}.load_fbx, {os.path.basename(file)}"
            with self.subTest(loader=label, **kwargs):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaisesRegex(error, message):
                        loader(file, **kwargs)
                self.assertEqual(self.hidden(caught), [])


if __name__ == "__main__":
    unittest.main()
