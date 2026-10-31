"""TransformData's shear: Maya's shearXY, shearXZ and shearYZ as a channel.

A node holds it, a list reads and writes it for every node, a clip keeps one
(F, N, 3) block of it, files save it under its public name and leave a zero
one out, and every matrix in the package composes S * Sh * RO * R * JO * IS
* T, one node at a time or batched. test_shear_maya.py holds those matrices
to what Maya recorded. A rig holding no shear skips every product.
"""
from __future__ import annotations

import copy
import inspect
import os
import pickle
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock

import numpy as np
from cgmath._tests.test_clip import _DeltaRigs
from cgmath._tests.test_skin_deform import branching_clip, pose_at, skinned_mesh
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.sdf import SDFBox, SDFCylinder, SDFSphere
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList, generate_uuid
from cgmath.hierarchy.hierarchy import _diagonal, _pose_matrices, _scale_shears, _span_world
from cgmath.render.scene import Camera, Light, Object, Scene

ASSETS = os.path.join(os.path.dirname(__file__), "test_assets")
SHEAR  = (0.5, -0.25, 0.75)

# a gate on shear reads all three entries: neither xy nor their sum says
# what a node holds
PARTIAL_SHEARS = ((0.0, 0.5, 0.0), (0.0, 0.0, 0.75), (0.0, 0.5, 0.75), (0.25, 0.5, -0.75))
READS = (
    "matrix",
    "world_matrix",
    "parent_matrix",
    "inverse_matrix",
    "world_inverse_matrix",
    "parent_inverse_matrix",
)


def _rig(sheared=None, seed: int = 11, count: int = 9) -> HierarchyData:
    """a branching rig of joints, transforms and locators, every channel away
    from its default, half the joints compensating their parent's scale.
    ``sheared`` lists the rows holding shear, every row when None"""
    rng   = np.random.default_rng(seed)
    uuids = [generate_uuid() for _ in range(count)]
    nodes = []
    for i in range(count):
        kind  = ("joint", "transform", "joint", "locator")[i % 4]
        joint = kind == "joint"
        shear = rng.uniform(-0.8, 0.8, 3)
        held  = dict(shear=shear) if sheared is None or i in sheared else {}
        nodes.append(
            TransformData(
                f"n{i}",
                node_type=kind,
                uuid=uuids[i],
                parent_uuid=uuids[int(rng.integers(0, i))] if i else None,
                translate=rng.uniform(-3.0, 3.0, 3),
                rotate=rng.uniform(-150.0, 150.0, 3),
                scale=rng.uniform(0.5, 2.0, 3),
                rotate_order=int(rng.integers(0, 6)),
                rotate_axis=rng.uniform(-30.0, 30.0, 3),
                joint_orient=rng.uniform(-40.0, 40.0, 3) if joint else None,
                segment_scale_compensate=bool(joint and i and rng.random() < 0.5),
                **held,
            )
        )
    return HierarchyData(nodes)


def _animated(rig: HierarchyData, frames: int = 5, seed: int = 21) -> ClipData:
    """``rig`` over ``frames`` frames, bound to the first, every framed channel
    but shear moving"""
    rng   = np.random.default_rng(seed)
    clip  = ClipData(rig, frames=frames)
    shape = clip.frames.rotate.shape
    clip.frames.rotate       = clip.frames.rotate + rng.normal(size=shape) * 20.0
    clip.frames.translate    = clip.frames.translate + rng.normal(size=shape) * 0.3
    clip.frames.scale        = clip.frames.scale * rng.uniform(0.8, 1.2, shape)
    clip.frames.rotate_axis  = clip.frames.rotate_axis + rng.normal(size=shape) * 5.0
    clip.frames.joint_orient = clip.frames.joint_orient + rng.normal(size=shape) * 5.0
    return clip


def _sheared_clip(rig: HierarchyData, frames: int = 5, seed: int = 22) -> ClipData:
    """:func:`_animated`, shear moving on every frame too"""
    clip              = _animated(rig, frames)
    rng               = np.random.default_rng(seed)
    clip.frames.shear = rng.uniform(-0.8, 0.8, clip.frames.shear.shape)
    return clip


def _gap(a, b) -> float:
    """the largest entry of ``a - b``, relative to ``b``'s largest"""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def _sheared_rows(rows) -> np.ndarray:
    """(xy, xz, yz) as the 4x4 Sh this file spells out, the reference"""
    xy, xz, yz = rows
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [xy, 1.0, 0.0, 0.0],
            [xz, yz, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _spelled(node) -> np.ndarray:
    """``node``'s matrix with Sh written out, the reference"""
    return (
        node.scale_matrix
        @ _sheared_rows(node.shear)
        @ node.rotate_axis_matrix
        @ node.rotate_matrix
        @ node.joint_orient_matrix
        @ node.parent_scale_inverse_matrix
        @ node.translate_matrix
    )


def _spelled_worlds(rig: HierarchyData) -> np.ndarray:
    """every world of ``rig`` down from :func:`_spelled`, in ``rig``'s order"""
    worlds = {}
    for row in rig._parent_first():
        node              = rig[row]
        worlds[node.uuid] = _spelled(node) @ worlds.get(node.parent_uuid, np.eye(4))
    return np.array([worlds[node.uuid] for node in rig.list])


class TestShearChannel(unittest.TestCase):
    def test_a_node_starts_at_zero_and_keeps_its_own(self):
        a, b = TransformData("a"), TransformData("b")
        self.assertEqual(a.shear.dtype, np.float64)
        self.assertEqual(a.shear.tolist(), [0.0, 0.0, 0.0])

        a.shear = SHEAR
        self.assertEqual(a.shear.tolist(), list(SHEAR))
        self.assertEqual(b.shear.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(TransformData("c", shear=SHEAR).shear.tolist(), list(SHEAR))

        given    = np.array(SHEAR)  # the node copies what it is built from
        c        = TransformData("c", shear=given)
        given[0] = 9.0
        self.assertEqual(c.shear.tolist(), list(SHEAR))

        held    = a.shear  # a copy, as every channel's getter hands back
        held[0] = 9.0
        self.assertEqual(a.shear.tolist(), list(SHEAR))

    def test_it_takes_what_a_channel_takes(self):
        node       = TransformData("a")
        node.shear = 2.0
        self.assertEqual(node.shear.tolist(), [2.0, 2.0, 2.0])
        node.shear = [[1.0, 2.0, 3.0]]
        self.assertEqual(node.shear.tolist(), [1.0, 2.0, 3.0])
        node.shear = np.array(SHEAR)
        self.assertEqual(node.shear.tolist(), list(SHEAR))

        with self.assertRaisesRegex(ValueError, r"^TransformData\.shear must have shape \(3,\), got \(2,\)$"):
            node.shear = (1.0, 2.0)
        self.assertEqual(node.shear.tolist(), list(SHEAR))

    def test_its_components_are_xy_xz_yz(self):
        node = TransformData("a", shear=SHEAR)
        self.assertEqual((node.shear_xy, node.shear_xz, node.shear_yz), SHEAR)

        node.shear_xy = 1.0
        node.shear_xz = 2.0
        node.shear_yz = 3.0
        self.assertEqual(node.shear.tolist(), [1.0, 2.0, 3.0])
        self.assertFalse(hasattr(node, "shear_x"))

    def test_a_write_moves_a_childs_cached_world(self):
        parent = TransformData("p", uuid=generate_uuid(), translate=(1.0, 2.0, 3.0), scale=(1.0, 2.0, 3.0))
        child  = TransformData("c", uuid=generate_uuid(), parent_uuid=parent.uuid, translate=(1.0, 1.0, 1.0))
        rig    = HierarchyData([parent, child])
        before = rig["c"].world_matrix

        rig["p"].shear = SHEAR
        want           = child.matrix @ np.diag([1.0, 2.0, 3.0, 1.0]) @ _sheared_rows(SHEAR) @ parent.translate_matrix
        self.assertGreater(np.abs(rig["c"].world_matrix - before).max(), 0.1)
        self.assertLessEqual(_gap(rig["c"].world_matrix, want), 1e-14)

    def test_it_is_the_last_parameter(self):
        # every positional call written before it stays as it was
        self.assertEqual(list(inspect.signature(TransformData.__init__).parameters)[-1], "shear")

    def test_a_list_reads_and_writes_every_node(self):
        rig = _rig(sheared=())
        self.assertEqual(rig.shear.shape, (len(rig), 3))
        self.assertFalse(rig.shear.any())

        rig.shear = SHEAR  # one value, every node
        self.assertTrue(np.array_equal(rig.shear, np.tile(SHEAR, (len(rig), 1))))

        values    = np.arange(len(rig) * 3, dtype=float).reshape(-1, 3)
        rig.shear = values  # one row a node
        self.assertTrue(np.array_equal(rig.shear, values))
        self.assertTrue(np.array_equal(rig[2:5].shear, values[2:5]))
        self.assertTrue(np.array_equal(rig.shear_xz, values[:, 1]))

        rig[1:3].shear_yz = [7.0, 8.0]
        self.assertEqual(rig.shear_yz[1:3].tolist(), [7.0, 8.0])
        self.assertEqual([n.shear_yz for n in rig.list[1:3]], [7.0, 8.0])

    def test_shear_matrix_on_a_node_a_list_and_a_clip(self):
        node = TransformData("a", shear=SHEAR)
        self.assertTrue(np.array_equal(node.shear_matrix, _sheared_rows(SHEAR)))

        rig = _rig()
        self.assertTrue(np.array_equal(rig.shear_matrix, np.array([_sheared_rows(s) for s in rig.shear])))

        clip = _sheared_clip(_rig())
        got  = clip.frames.shear_matrix
        self.assertEqual(got.shape, clip.frames.shear.shape[:2] + (4, 4))
        for frame, row in np.ndindex(got.shape[:2]):
            self.assertTrue(np.array_equal(got[frame, row], _sheared_rows(clip.frames.shear[frame, row])))

    def test_a_clip_frames_it(self):
        rig   = _rig()
        clip  = _sheared_clip(rig, frames=4)
        shape = (4, len(rig))
        self.assertEqual(clip.frames.shear.shape, shape + (3,))
        self.assertEqual(clip.frames.shear_xy.shape, shape)
        for k, component in enumerate(("shear_xy", "shear_xz", "shear_yz")):
            self.assertTrue(np.array_equal(getattr(clip.frames, component), clip.frames.shear[..., k]))

        # the bound frame reads and writes its row, and no other frame's
        before     = np.array(clip.frames.shear)
        clip.frame = 2
        self.assertTrue(np.array_equal(clip.shear, before[2]))
        self.assertTrue(np.array_equal(clip["n3"].shear, before[2, 3]))

        clip["n3"].shear = SHEAR
        after            = clip.frames.shear
        self.assertEqual(after[2, 3].tolist(), list(SHEAR))
        same       = np.ones(shape, dtype=bool)
        same[2, 3] = False
        self.assertTrue(np.array_equal(after[same], before[same]))

        clip.frames.shear_xy = 4.0
        self.assertTrue((clip.frames.shear[..., 0] == 4.0).all())

    def test_maya_short_names_read_and_write_it_and_stay_out_of_files(self):
        clip = _sheared_clip(_rig())
        rig  = _rig()
        node = rig["n1"]
        for short, long in (("sh", "shear"), ("shxy", "shear_xy"), ("shxz", "shear_xz"), ("shyz", "shear_yz")):
            for target in (node, rig, clip, clip.frames):
                with self.subTest(short=short, target=type(target).__name__):
                    self.assertTrue(np.array_equal(getattr(target, short), getattr(target, long)))

        node.sh = SHEAR
        self.assertEqual(node.shear.tolist(), list(SHEAR))
        node.shxz = 9.0
        self.assertEqual(node.shear_xz, 9.0)
        rig.shyz = 8.0
        self.assertTrue((rig.shear_yz == 8.0).all())
        clip.frames.sh = 7.0
        self.assertTrue((clip.frames.shear == 7.0).all())

        for tree in (node.to_dict(), rig.to_dict(), clip.to_dict()):
            stack, keys = [tree], set()
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    keys.update(item)
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
            self.assertEqual(keys & {"sh", "shxy", "shxz", "shyz", "shear_xy", "shear_matrix", "_shear"}, set())

    def test_to_attributes_gives_it_on_every_node(self):
        # as it gives rotateAxis, zero included: a create path sets it
        self.assertEqual(TransformData("a").to_attributes()["shear"], [0.0, 0.0, 0.0])
        self.assertEqual(TransformData("a", node_type="joint", shear=SHEAR).to_attributes()["shear"], list(SHEAR))

        clip = _sheared_clip(_rig(), frames=2)
        got  = clip.frames.to_attributes()
        self.assertEqual(got[3]["shear"], clip.frames.shear[:, 3].tolist())


class TestNoShear(unittest.TestCase):
    """the fast path: every read below holds on 3.0.3 as well, and is here
    for what a change to the channel can do to a rig that holds none"""

    def test_a_node_is_the_six_factor_product_to_the_bit(self):
        for node in _rig(sheared=()):
            with self.subTest(node=node.name):
                M = np.eye(4)
                for factor in (
                    node.scale_matrix,
                    node.rotate_axis_matrix,
                    node.rotate_matrix,
                    node.joint_orient_matrix,
                    node.parent_scale_inverse_matrix,
                    node.translate_matrix,
                ):
                    M = np.dot(M, factor)
                self.assertTrue(np.array_equal(node.matrix, M))

    def test_a_rig_and_a_clip_are_the_product_of_their_own_factors(self):
        rig  = _rig(sheared=())
        clip = _animated(rig)
        got  = clip.frames.matrix
        want = (
            clip.frames.scale_matrix
            @ clip.frames.rotate_axis_matrix
            @ clip.frames.rotate_matrix
            @ clip.frames.joint_orient_matrix
            @ clip.frames.parent_scale_inverse_matrix
            @ clip.frames.translate_matrix
        )
        self.assertTrue(np.array_equal(got, want))

    def test_no_tree_and_no_block_names_it(self):
        rig  = _rig(sheared=())
        clip = _animated(rig)
        self.assertNotIn("shear", rig["n1"].to_dict())
        self.assertNotIn("shear", rig.to_json())
        self.assertNotIn("shear", clip.to_dict()["blocks"])
        self.assertNotIn("shear", clip.to_json())

    def test_a_node_holding_none_builds_no_sh(self):
        with mock.patch.object(TransformData, "shear_matrix", new_callable=mock.PropertyMock, create=True) as built:
            for node in _rig(sheared=()):
                node.matrix
        built.assert_not_called()


class TestShearBatched(unittest.TestCase, _DeltaRigs):
    """every batched read against one node at a time, on rigs and clips of
    mixed node types"""

    def _rigs(self):
        yield "every node", _rig()
        yield "one node of many", _rig(sheared=(4,))
        yield "the last node of many", _rig(sheared=(8,))

    def _clips(self):
        yield "every frame", _sheared_clip(_rig())
        clip = _animated(_rig(sheared=()))

        # one node on one frame: neither the first node nor the first frame
        # says what the block holds
        block             = np.zeros(clip.frames.shear.shape)
        block[3, 4]       = SHEAR
        clip.frames.shear = block
        yield "one node on one frame", clip

    def test_a_rig_reads_what_its_nodes_read(self):
        for label, rig in self._rigs():
            for read in READS:
                with self.subTest(label, read=read):
                    want = np.array([getattr(node, read) for node in rig.list])
                    self.assertLessEqual(_gap(getattr(rig, read), want), 1e-12)

    def test_a_selection_reads_through_the_pose_evaluators(self):
        for label, rig in self._rigs():
            with self.subTest(label):
                local, world, parent = _pose_matrices(rig.list)
                self.assertLessEqual(_gap(local, np.array([n.matrix for n in rig.list])),         1e-12)
                self.assertLessEqual(_gap(world, np.array([n.world_matrix for n in rig.list])),   1e-12)
                self.assertLessEqual(_gap(parent, np.array([n.parent_matrix for n in rig.list])), 1e-12)

    def test_a_scale_that_is_not_finite_stays_out_of_a_stack_with_no_shear(self):
        # inf * 0 is NaN
        scale = np.array([[[1.0, np.inf, 2.0], [np.inf, 1.0, np.inf]]])
        got   = _scale_shears(scale, np.zeros_like(scale))
        self.assertTrue(np.array_equal(got, _diagonal(scale)))

    def test_a_clip_reads_what_its_nodes_read_on_every_frame(self):
        for label, clip in self._clips():
            for read in READS:
                got = getattr(clip.frames, read)
                for frame in range(clip.frame_count):
                    with self.subTest(label, read=read, frame=frame):
                        clip.frame = frame
                        want       = np.array([getattr(node, read) for node in clip.list])
                        self.assertLessEqual(_gap(got[frame], want), 1e-12)

    def test_a_span_of_frames_evaluates_what_the_frames_read(self):
        # the route a clip's world target takes, its rows built one by one
        for label, clip in self._clips():
            with self.subTest(label):
                self.assertLessEqual(_gap(_span_world(clip, slice(1, 4)), clip.frames.world_matrix[1:4]), 1e-12)

    def test_a_frame_axis_solve_is_each_pose_solved(self):
        # each node solves under its parent's final world, which _solve_frames
        # composes from the shear a joint keeps
        clip   = _sheared_clip(_rig())
        target = _animated(_rig(sheared=(), seed=5)).frames.world_matrix

        want   = []
        for frame in range(clip.frame_count):
            pose              = pose_at(clip.copy(), frame)
            pose.world_matrix = target[frame]
            want.append((pose.world_matrix, pose.shear))

        clip.frames.world_matrix = target
        self.assertLessEqual(_gap(clip.frames.world_matrix, np.array([w for w, _ in want])), 1e-9)
        self.assertLessEqual(_gap(clip.frames.shear, np.array([h for _, h in want])), 1e-9)

    def test_skinning_a_sheared_clip_is_skinning_its_poses(self):
        rig, clip = branching_clip()
        mesh, skin = skinned_mesh(rig)
        plain = clip.copy()

        rng               = np.random.default_rng(31)
        clip.frames.shear = rng.uniform(-0.5, 0.5, clip.frames.shear.shape)
        for method in ("lbs", "dqs"):
            with self.subTest(method):
                d    = SkinDeformData(mesh, skin, bind_rig=rig, method=method)
                want = np.stack([d.apply(pose_at(clip.copy(), f)) for f in range(clip.frame_count)])
                got  = d.apply(clip.frames)
                self.assertLessEqual(np.abs(got - want).max(), 1e-9)
                self.assertGreater(np.abs(got - d.apply(plain.frames)).max(), 0.01)

    def test_a_clip_of_deltas_whose_shear_moves(self):
        names  = [f"j{i}" for i in range(6)]
        ups    = [None, "j0", "j1", "j1", "j3", "j4"]
        rig    = self._rig(names, ups, 7, scale=True)
        clip   = self._animated(rig.copy(), seed=8)
        deltas = self._animated(self._rig(names, ups, 7, scale=True), seed=9)

        rng                 = np.random.default_rng(32)
        deltas.frames.shear = rng.uniform(-0.5, 0.5, deltas.frames.shear.shape)
        got                 = clip.add_delta(deltas)
        self.assertFalse(np.array_equal(got.frames.shear[0], got.frames.shear[1]))
        self._assert_same(got, self._per_frame("add_delta", clip, deltas))

    def test_one_delta_holding_shear(self):
        names       = [f"j{i}" for i in range(6)]
        ups         = [None, "j0", "j1", "j1", "j3", "j4"]
        rig         = self._rig(names, ups, 7, scale=True)
        clip        = self._animated(rig.copy(), seed=8)
        delta       = self._rig(names, ups, 7, scale=True)
        delta.shear = SHEAR
        self._assert_same(clip.add_delta(delta), self._per_frame("add_delta", clip, delta))

    def test_a_node_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            with self.subTest(shear=shear):
                node = TransformData("a", shear=shear, scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))
                self.assertTrue(np.array_equal(node.shear_matrix, _sheared_rows(shear)))
                self.assertLessEqual(_gap(node.matrix, _spelled(node)), 1e-14)

    def test_a_rig_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            for label, rows in (("one node", (4,)), ("every node", range(9))):
                with self.subTest(label, shear=shear):
                    rig = _rig(sheared=())
                    for row in rows:
                        rig.list[row].shear = shear
                    local  = np.array([_spelled(node) for node in rig.list])
                    worlds = _spelled_worlds(rig)
                    self.assertLessEqual(_gap(rig.matrix, local), 1e-12)
                    self.assertLessEqual(_gap(rig.world_matrix, worlds), 1e-12)
                    self.assertLessEqual(_gap(_pose_matrices(rig.list)[0], local), 1e-12)
                    self.assertLessEqual(_gap(_pose_matrices(rig.list)[1], worlds), 1e-12)

    def test_a_clip_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            for label, at in (("one node on one frame", (3, 4)), ("every node", Ellipsis)):
                with self.subTest(label, shear=shear):
                    clip              = _animated(_rig(sheared=()))
                    block             = np.zeros(clip.frames.shear.shape)
                    block[at]         = shear
                    clip.frames.shear = block

                    poses  = [pose_at(clip.copy(), f) for f in range(clip.frame_count)]
                    local  = np.array([[_spelled(node) for node in pose.list] for pose in poses])
                    worlds = np.array([_spelled_worlds(pose) for pose in poses])
                    self.assertLessEqual(_gap(clip.frames.matrix, local), 1e-12)
                    self.assertLessEqual(_gap(clip.frames.world_matrix, worlds), 1e-12)
                    self.assertLessEqual(_gap(_span_world(clip, slice(1, 4)), worlds[1:4]), 1e-12)


class TestShearSolvesDrop(unittest.TestCase):
    """a solve of a target that holds no shear: a node that is no joint ends
    with none and reads back as written, a joint keeps its own"""

    TARGET = dict(scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), translate=(4.0, 5.0, 6.0))

    def _target(self) -> np.ndarray:
        return TransformData("t", **self.TARGET).matrix

    def test_a_node_that_is_no_joint(self):
        for kind in ("transform", "locator"):
            for write in ("matrix", "world_matrix"):
                with self.subTest(kind=kind, write=write):
                    node = TransformData("a", node_type=kind, shear=SHEAR, scale=(2.0, 2.0, 2.0))
                    setattr(node, write, self._target())
                    self.assertEqual(node.shear.tolist(), [0.0, 0.0, 0.0])
                    self.assertLessEqual(_gap(node.matrix, self._target()), 1e-12)

    def test_a_node_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            for write in ("matrix", "world_matrix"):
                with self.subTest(shear=shear, write=write):
                    node = TransformData("a", shear=shear, scale=(2.0, 2.0, 2.0))
                    setattr(node, write, self._target())
                    self.assertEqual(node.shear.tolist(), [0.0, 0.0, 0.0])
                    self.assertLessEqual(_gap(node.matrix, self._target()), 1e-12)

    def test_a_joint_keeps_its_own(self):
        for write in ("matrix", "world_matrix"):
            with self.subTest(write=write):
                node = TransformData("a", node_type="joint", shear=SHEAR, scale=(2.0, 2.0, 2.0))
                setattr(node, write, self._target())
                self.assertEqual(node.shear.tolist(), list(SHEAR))
                self.assertTrue(np.allclose(node.scale, self.TARGET["scale"], rtol=0.0, atol=1e-12))

    def test_a_list_and_a_rig(self):
        rig        = _rig()
        joints     = np.array([n.node_type == "joint" for n in rig.list])
        before     = rig.shear
        rig.matrix = np.tile(self._target(), (len(rig), 1, 1))
        self.assertTrue(np.array_equal(rig.shear[joints], before[joints]))
        self.assertFalse(rig.shear[~joints].any())
        self.assertLessEqual(_gap(rig.matrix[~joints], self._target()), 1e-12)

    def test_a_clips_frames(self):
        clip   = _sheared_clip(_rig())
        joints = np.array([n.node_type == "joint" for n in clip.list])
        before = np.array(clip.frames.shear)
        shape  = before.shape[:2] + (4, 4)

        clip.frames.matrix = np.broadcast_to(self._target(), shape).copy()
        self.assertTrue(np.array_equal(clip.frames.shear[:, joints], before[:, joints]))
        self.assertFalse(clip.frames.shear[:, ~joints].any())

    def test_a_clip_keeps_a_frame_already_there(self):
        # a frame whose channels make the target keeps them, shear included
        clip               = _sheared_clip(_rig(), frames=3)
        before             = np.array(clip.frames.shear)
        target             = clip.frames.matrix.copy()
        target[1, 1]       = self._target()
        clip.frames.matrix = target
        after              = clip.frames.shear
        self.assertFalse(after[1, 1].any())
        self.assertTrue(np.array_equal(after[0], before[0]))
        self.assertTrue(np.array_equal(after[2], before[2]))

    def test_a_list_keeps_what_a_node_that_stays_where_it_is_holds(self):
        # a root going to the world solves nothing: the channels it holds
        # make its target, shear among them, as on a clip's kept frame
        rig    = HierarchyData([TransformData("a", shear=SHEAR, scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))])
        before = rig["a"].world_matrix
        TransformList([rig["a"]]).set_parent(None)
        self.assertEqual(rig["a"].shear.tolist(), list(SHEAR))
        self.assertEqual(rig["a"].world_matrix.tolist(), before.tolist())

    def test_a_child_put_back_by_a_swap_is_one_node_at_a_time(self):
        # the list writes what the child's setter writes
        uuids = [generate_uuid() for _ in range(2)]
        rig = HierarchyData(
            [
                TransformData("a", uuid=uuids[0], scale=(1.2, 0.8, 1.5), rotate=(10.0, 20.0, 30.0)),
                TransformData("b", uuid=uuids[1], parent_uuid=uuids[0], translate=(1.0, 2.0, 3.0), shear=SHEAR),
            ]
        )
        turn = rig.copy()
        turn["a"].swapaxes(0, 1)
        batch = rig.copy()
        TransformList([batch["a"]]).swapaxes(0, 1)
        self.assertLessEqual(_gap(batch["b"].shear, turn["b"].shear), 1e-12)
        self.assertLessEqual(_gap(batch.world_matrix, turn.world_matrix), 1e-12)

    def _picked(self, rig, parent):
        # the list's route, and the same nodes one at a time: each to the
        # world, then each under the parent
        names = [node.name for node in rig.list[2:7]]
        batch = rig.copy()
        TransformList([batch[name] for name in names]).set_parent(parent)
        turn = rig.copy()
        for name in names:
            turn[name].set_parent(None)
        if parent is not None:
            for name in names:
                turn[name].set_parent(parent)
        return batch, turn

    def test_set_parent_runs_on_a_rig_holding_shear(self):
        for label, parent in (("the world", None), ("a node", "n0")):
            with self.subTest(label):
                for seed in (11, 12, 13):
                    rig = _rig(seed=seed)
                    batch, turn = self._picked(rig, parent)
                    self.assertTrue(np.isfinite(batch.world_matrix).all())
                    self.assertLessEqual(_gap(batch.world_matrix, turn.world_matrix), 1e-9)

    def test_set_parent_keeps_a_joints_shear(self):
        rig    = _rig()
        names  = [node.name for node in rig.list[2:7] if node.node_type == "joint"]
        before = [rig[name].shear for name in names]
        batch, turn = self._picked(rig, None)
        for name, held in zip(names, before):
            self.assertEqual(batch[name].shear.tolist(), held.tolist())
            self.assertEqual(turn[name].shear.tolist(), held.tolist())


class TestShearFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def _held(self, obj):
        return obj.frames.shear if isinstance(obj, ClipData) else obj.shear

    def _objects(self):
        rig = _rig()
        yield "node", TransformData("a", node_type="joint", shear=SHEAR), TransformData
        yield "rig", rig, HierarchyData
        yield "clip", _sheared_clip(_rig()), ClipData

    def test_every_format_keeps_it(self):
        for label, obj, cls in self._objects():
            for ext in ("json", "npz", "pkl"):
                with self.subTest(label, ext=ext):
                    path = obj.save(os.path.join(self.tmp, f"{label}.{ext}"))
                    back = cls.load(path)
                    self.assertIsInstance(back, cls)
                    self.assertTrue(np.array_equal(self._held(back), self._held(obj)))

    def test_a_pickle_and_a_copy_keep_it(self):
        for label, obj, _ in self._objects():
            for how, make in (
                ("pickle", lambda x: pickle.loads(pickle.dumps(x))),
                ("deepcopy", copy.deepcopy),
                ("copy", lambda x: x.copy()),
            ):
                with self.subTest(label, how=how):
                    self.assertTrue(np.array_equal(self._held(make(obj)), self._held(obj)))

    def test_its_public_name_is_saved(self):
        node = TransformData("a", shear=SHEAR)
        tree = node.to_dict()
        self.assertNotIn("_shear", tree)
        self.assertEqual(np.asarray(tree["shear"]).tolist(), list(SHEAR))

        clip   = _sheared_clip(_rig())
        blocks = clip.to_dict()["blocks"]
        self.assertNotIn("_shear", blocks)
        self.assertTrue(np.array_equal(np.asarray(blocks["shear"]), clip.frames.shear))

    def test_an_all_zero_block_is_left_out_and_one_non_zero_frame_keeps_it(self):
        clip = _animated(_rig(sheared=()), frames=4)
        self.assertNotIn("shear", clip.to_dict()["blocks"])

        # on a frame the clip is not bound to
        clip.frame        = 0
        block             = np.zeros(clip.frames.shear.shape)
        block[2, 3, 1]    = 0.5
        clip.frames.shear = block
        clip.frame        = 0
        tree              = clip.to_dict()
        self.assertTrue(np.array_equal(np.asarray(tree["blocks"]["shear"]), block))
        self.assertTrue(np.array_equal(ClipData.from_dict(tree).frames.shear, block))

    def test_a_clip_tree_with_no_shear_block_loads_zeros(self):
        clip           = _sheared_clip(_rig())
        tree           = clip.to_dict()
        tree["blocks"] = {key: block for key, block in tree["blocks"].items() if key != "shear"}
        back           = ClipData.from_dict(tree)
        self.assertEqual(back.frames.shear.shape, clip.frames.shear.shape)
        self.assertFalse(back.frames.shear.any())
        self.assertFalse(back.shear.any())

    def test_a_tree_missing_another_block_still_raises_naming_only_it(self):
        tree           = _animated(_rig(sheared=())).to_dict()
        tree["blocks"] = {key: block for key, block in tree["blocks"].items() if key != "rotate_axis"}
        with self.assertRaisesRegex(ValueError, r"missing the blocks \['rotate_axis'\]"):
            ClipData.from_dict(tree)

    def test_files_older_than_it_load_with_shear_zero(self):
        for ext in ("json", "npz", "pkl"):
            with self.subTest(ext=ext):
                rig = HierarchyData.load(os.path.join(ASSETS, f"test_hierarchy.legacy_parent_node.{ext}"))
                self.assertGreater(len(rig), 0)
                self.assertFalse(rig.shear.any())

        clip = ClipData.load(os.path.join(ASSETS, "test_hierarchy.legacy_parent_node_clip.json"))
        self.assertEqual(clip.frames.shear.shape, clip.frames.scale.shape)
        self.assertFalse(clip.frames.shear.any())

    def test_files_saved_by_2_0_0_load_with_shear_zero(self):
        with zipfile.ZipFile(os.path.join(ASSETS, "test_data_module.saved_by_2_0_0.zip")) as z:
            z.extractall(self.tmp)
        seen = 0
        for where, _, names in os.walk(self.tmp):
            for name in names:
                if name.startswith("hierarchy") and name.endswith((".json", ".npz", ".pkl")):
                    with self.subTest(name):
                        loaded = HierarchyData.load(os.path.join(where, name))
                        self.assertFalse(loaded.shear.any())
                        seen += 1
        self.assertGreater(seen, 0)

    def test_a_user_attribute_with_its_name_loads_hidden_with_a_warning(self):
        for name, what in (
            ("shear", "a built-in"),
            ("shear_xy", "a built-in"),
            ("shear_xz", "a built-in"),
            ("shear_yz", "a built-in"),
            ("shear_matrix", "a built-in"),
            ("sh", "Maya's short name for shear"),
            ("shxy", "Maya's short name for shear_xy"),
            ("shxz", "Maya's short name for shear_xz"),
            ("shyz", "Maya's short name for shear_yz"),
        ):
            with self.subTest(name):
                tree = TransformData("b").to_dict()
                tree["user_defined_attributes"] = {
                    name: {"attributeType": "double", "value": 1.5, "keyable": False, "channel_box": False}
                }
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    node = TransformData.from_dict(tree)

                self.assertEqual(node.shear.tolist(), [0.0, 0.0, 0.0])
                self.assertEqual(node.user_defined_attributes[name]["value"], 1.5)
                self.assertEqual(
                    [str(w.message) for w in caught],
                    [
                        f"user attribute {name!r} of 'b' loaded into user_defined_attributes[{name!r}]: "
                        f"node.{name} is {what}, which hides it"
                    ],
                )

    def test_add_user_attribute_refuses_its_names(self):
        node = TransformData("a")
        for name in ("shear", "sh", "shxy", "shear_xz", "shear_matrix"):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, f"'{name}' is a built-in attribute of TransformData"):
                    node.add_user_attribute(name, 1.5)

    def _primitives(self):
        for cls in (SDFSphere, SDFBox, SDFCylinder):
            plain      = cls(translate=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), scale=(1.0, 2.0, 3.0))
            held       = plain.copy()
            held.shear = SHEAR
            yield cls, plain, held

    def test_an_sdf_primitive_saves_it(self):
        for cls, plain, held in self._primitives():
            with self.subTest(cls.__name__):
                self.assertEqual(pickle.loads(pickle.dumps(held)).shear.tolist(), list(SHEAR))
                self.assertEqual(copy.deepcopy(held).shear.tolist(), list(SHEAR))

                back = HierarchyData.from_dict(HierarchyData([held]).to_dict())[0]
                self.assertIsInstance(back, cls)
                self.assertEqual(back.shear.tolist(), list(SHEAR))
                self.assertGreater(np.abs(held.matrix - plain.matrix).max(), 0.1)

    def test_an_sdf_primitive_does_not_evaluate_it(self):
        # as it does not evaluate rotate_axis, joint_orient or a parent
        X, Y, Z = np.meshgrid(*[np.linspace(-2.0, 2.0, 5)] * 3, indexing="ij")
        for cls, plain, held in self._primitives():
            with self.subTest(cls.__name__):
                self.assertTrue(np.array_equal(plain.sample(X, Y, Z), held.sample(X, Y, Z)))
                for a, b in zip(plain.bounding_box(), held.bounding_box()):
                    self.assertTrue(np.array_equal(a, b))


class TestShearInARender(unittest.TestCase):
    """the render caches key on the raw channel: a write past the setter
    leaves the cached world as it was, so only the raw key sees it"""

    def _object(self) -> Object:
        return Object("o", scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))

    def test_every_render_node_takes_it(self):
        for cls in (Object, Camera, Light):
            with self.subTest(cls.__name__):
                self.assertEqual(cls(name="a", shear=SHEAR).shear.tolist(), list(SHEAR))

    def test_an_objects_snapshot(self):
        node = self._object()
        node.world_matrix  # cached
        before = node._capture_transform_state()

        node._shear[...] = SHEAR
        self.assertNotEqual(node._capture_transform_state(), before)

        node       = self._object()
        before     = node._capture_transform_state()
        node.shear = SHEAR
        self.assertNotEqual(node._capture_transform_state(), before)

    def test_a_scenes_signature(self):
        scene = Scene()
        scene.append(Camera("c"))
        scene.append(self._object())
        node = scene["o"]
        node.world_matrix  # cached
        before = scene._compute_render_signature()

        node._shear[...] = SHEAR
        self.assertNotEqual(scene._compute_render_signature(), before)

        node.shear = (0.0, 0.0, 0.0)
        before     = scene._compute_render_signature()
        node.shear = SHEAR
        self.assertNotEqual(scene._compute_render_signature(), before)


if __name__ == "__main__":
    unittest.main()
