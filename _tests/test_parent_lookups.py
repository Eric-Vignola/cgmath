"""Whole-rig parent lookups: one index of the rig per call, never a scan of
the rig per node, and the same answers each node's own method gives.

The scale tests count string comparisons rather than time: a uuid, name or
parent_uuid that counts how often it is compared tells a pass over the rig
(a few per node) from a scan of the rig for every node (tens of thousands
at 300 nodes), whatever else the machine is doing.
"""

import sys
import unittest
from unittest import mock

import numpy as np
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList
from cgmath.hierarchy import generate_uuid
from cgmath.transforms import euler_filter


class _Counted(str):
    """a string that counts how often it is compared, so a test can tell one
    pass over a rig from a scan of the rig for every node"""

    compared = 0

    def __eq__(self, other):
        _Counted.compared += 1
        return str.__eq__(self, other)

    def __ne__(self, other):
        _Counted.compared += 1
        return str.__ne__(self, other)

    __hash__ = str.__hash__


def _counted_nodes(count: int) -> list:
    """a branching rig's nodes, not yet in a hierarchy, whose uuids, names
    and parent_uuids are counted strings, each parent_uuid its own object so
    no lookup gets away with an identity check"""
    rng   = np.random.default_rng(3)
    nodes = []
    for i in range(count):
        node      = TransformData(f"j{i}")
        node.uuid = _Counted(generate_uuid())
        node.name = _Counted(f"j{i}")
        if i:
            parent           = nodes[int(rng.integers(max(0, i - 4), i))]
            node.parent_uuid = _Counted(str(parent.uuid))
        nodes.append(node)
    return nodes


class TestParentLookupsScale(unittest.TestCase):
    """whole-rig lookups index the rig once rather than scanning it for every
    node: the comparisons they make grow with the node count, not its square"""

    COUNT = 300

    def assertLinear(self, call, what):
        _Counted.compared = 0
        call()
        # a scan per node makes tens of thousands here
        self.assertLess(_Counted.compared, 5 * self.COUNT, what)

    def test_building_a_hierarchy_checks_each_uuid_once(self):
        # the joining rule included: each parent_uuid is looked up once
        nodes = _counted_nodes(self.COUNT)
        self.assertLinear(lambda: HierarchyData(nodes), "HierarchyData(nodes)")
        self.assertEqual(sum(node.parent_uuid is not None for node in nodes), self.COUNT - 1)

    def test_parents_resolve_in_one_pass(self):
        rig = HierarchyData(_counted_nodes(self.COUNT))
        self.assertLinear(lambda: rig._parent_indices(rig.list), "_parent_indices")
        self.assertLinear(lambda: rig._parent_first(), "_parent_first")
        self.assertLinear(
            lambda: TransformList._owning_parent_first(rig[::3]),
            "_owning_parent_first",
        )
        self.assertLinear(lambda: rig.get_roots(), "get_roots")
        self.assertLinear(lambda: rig.get_parents(), "get_parents")

    def test_children_and_branches_resolve_in_one_pass(self):
        rig = HierarchyData(_counted_nodes(self.COUNT))
        self.assertLinear(lambda: rig.get_children(),   "get_children")
        self.assertLinear(lambda: rig.get_branch(),     "get_branch")
        self.assertLinear(lambda: rig[:1].get_branch(), "get_branch of the root")

    def test_a_name_pairing_resolves_parents_in_one_pass(self):
        # get_delta / add_delta by="name" pair the nodes, then parent each
        # pair within the pairing
        rig   = HierarchyData(_counted_nodes(self.COUNT))
        other = HierarchyData(_counted_nodes(self.COUNT))
        self.assertLinear(lambda: rig._pair_by_name(other), "_pair_by_name")

    def test_indices_and_unique_names_resolve_in_one_pass(self):
        rig = HierarchyData(_counted_nodes(self.COUNT))
        self.assertLinear(lambda: rig.indices, "indices")
        self.assertLinear(lambda: rig.unique_name, "unique_name")

    def test_a_chain_deeper_than_the_recursion_limit(self):
        # each node's own get_branch recurses once per level; the list level
        # walk must not, and must order the chain parent first
        count = sys.getrecursionlimit() + 500
        uuids = [generate_uuid() for _ in range(count)]
        rig = HierarchyData(
            [
                TransformData(f"j{i}", uuid=uuids[i], parent_uuid=uuids[i - 1] if i else None)
                for i in range(count)
            ]
        )
        self.assertEqual(rig.get_branch().name, rig.name)
        self.assertEqual(rig[:1].get_branch().name, rig.name)
        np.testing.assert_array_equal(rig.get_parents(), np.arange(-1, count - 1))
        self.assertEqual([int(i) for i in rig._parent_first()], list(range(count)))
        self.assertEqual(
            [int(i) for i in rig[::-1]._parent_first()], list(range(count))[::-1]
        )


class TestParentLookupsMatchTheNodes(unittest.TestCase):
    """the list-level lookups answer what each node's own method answers"""

    def _awkward(self) -> HierarchyData:
        """a shared name, a node named like a uuid, a child listed before its
        parent, nodes that joined with a parent the hierarchy does not hold
        (roots), and one whose parent_uuid was written so since"""
        ids = {name: generate_uuid() for name in ("root", "twin", "twin_b", "odd", "a", "b")}
        odd = TransformData(generate_uuid(), uuid=ids["odd"])  # named like a uuid
        rig = HierarchyData(
            [
                TransformData("f", parent_uuid=ids["b"]),  # a child listed first
                TransformData("root", uuid=ids["root"]),
                TransformData("twin", uuid=ids["twin"]),
                TransformData("twin", uuid=ids["twin_b"]),
                odd,
                TransformData("a", uuid=ids["a"], parent_uuid=ids["root"]),
                TransformData("b", uuid=ids["b"], parent_uuid=ids["twin"]),  # the first twin
                TransformData("g", parent_uuid=ids["twin_b"]),  # the second
                TransformData("c", parent_uuid=odd.name),  # no node carries it: a root
                TransformData("d", parent_uuid=generate_uuid()),  # not held: a root
                TransformData("e"),
                TransformData("h", parent_uuid=ids["a"]),
            ]
        )
        rig["e"].parent_uuid = generate_uuid()  # a parent the rig does not hold
        return rig

    def _views(self, rig):
        return [rig, rig[::-1], rig[[11, 0, 5, 3, 7]]]

    @staticmethod
    def _walk_depth(node) -> int:
        """ancestors up get_parent, the walk the old _parent_first made"""
        seen, parent = set(), node.get_parent()
        while isinstance(parent, TransformData) and id(parent) not in seen:
            seen.add(id(parent))
            parent = parent.get_parent()
        return len(seen)

    def test_parents_match_get_parent(self):
        rig = self._awkward()
        for view in self._views(rig):
            nodes    = list(view)
            position = {node.uuid: i for i, node in enumerate(nodes)}
            expected = [
                position.get(p.uuid, -1) if isinstance(p, TransformData) else -1
                for p in (node.get_parent() for node in nodes)
            ]
            np.testing.assert_array_equal(view._parent_indices(nodes), expected)

            self.assertEqual(
                [node.name for node in view.get_roots()],
                [node.name for node in view if node.get_parent() is None],
            )

            depth = [self._walk_depth(node) for node in view]
            self.assertEqual(
                [int(i) for i in view._parent_first()],
                [int(i) for i in np.argsort(depth, kind="stable")],
            )

    def test_get_parents_reads_a_parent_it_does_not_hold_as_a_root(self):
        rig     = self._awkward()
        parents = dict(zip(rig.name, rig.get_parents()))
        self.assertEqual(rig.get_parents().dtype.kind, "i")
        self.assertEqual(parents["b"], 2)               # the first twin
        self.assertEqual(parents["g"], 3)               # the second twin
        self.assertEqual(parents["f"], rig.index("b"))  # listed before its parent
        for name in ("c", "d", "e", "root"):
            self.assertEqual(parents[name], -1, name)
        self.assertIsNone(rig["c"].parent_uuid)
        self.assertIsNone(rig["d"].parent_uuid)
        self.assertIsInstance(rig["e"].get_parent(), str)

        view = rig[[11, 5]]
        np.testing.assert_array_equal(view.get_parents(), [5, 1])  # owning positions

    def test_children_and_branches_match_the_nodes(self):
        rig = self._awkward()
        for view in self._views(rig):
            expected, seen = [], set()
            for node in view:
                for child in node.get_children():
                    if child.uuid not in seen:
                        seen.add(child.uuid)
                        expected.append(child.name)
            self.assertEqual(view.get_children().name, expected)

            expected, seen = [], set()
            for node in view:
                for member in node.get_branch():
                    if member.uuid not in seen:
                        seen.add(member.uuid)
                        expected.append(member.name)
            self.assertEqual(view.get_branch().name, expected)

        self.assertEqual(rig[[2]].get_branch().name, ["twin", "b", "f"])

    def test_indices_and_unique_names_match_the_nodes(self):
        rig = self._awkward()
        for view in self._views(rig):
            np.testing.assert_array_equal(view.indices, [node.index for node in view])
            np.testing.assert_array_equal(
                view.unique_name, [node.unique_name for node in view]
            )

        loose = TransformList([TransformData("loose")])
        with self.assertRaises(ValueError):
            loose.indices
        with self.assertRaises(RuntimeError):
            loose.unique_name

    def test_a_cyclic_parent_ends_every_walk(self):
        ids = [generate_uuid() for _ in range(3)]
        rig = HierarchyData(
            [
                TransformData("a", uuid=ids[0], parent_uuid=ids[1]),
                TransformData("b", uuid=ids[1], parent_uuid=ids[0]),
                TransformData("c", uuid=ids[2], parent_uuid=ids[0]),
            ]
        )
        self.assertEqual(rig.get_branch().name, ["a", "b", "c"])
        self.assertEqual(len(rig.get_roots()), 0)
        self.assertEqual([int(i) for i in rig._parent_first()], [0, 1, 2])
        order = TransformList._parent_first_order(np.array([1, 0, 0, -1]))
        self.assertEqual([int(i) for i in order], [3, 0, 1, 2])

    def test_building_still_refuses_a_node_twice(self):
        node = TransformData("a", uuid=generate_uuid())
        twin = TransformData("b", uuid=node.uuid)
        with self.assertRaises(RuntimeError):
            HierarchyData([node, twin])

        rig = HierarchyData([TransformData(f"n{i}") for i in range(5)])
        self.assertEqual(len(set(rig.uuid)), 5)
        with self.assertRaises(RuntimeError):
            rig.extend([TransformData("x", uuid=rig[2].uuid)])
        with self.assertRaises(RuntimeError):
            rig.insert(0, TransformData("y", uuid=rig[4].uuid))
        self.assertNotIn("_held", vars(rig))


# a name shaped like a uuid: a parent_uuid is a uuid, never a name, so a
# child given it as its parent_uuid is a root
_UUID_SHAPED_NAME = "0F0F0F0F-0000-4000-8000-000000000001"


class TestClipParentLookups(unittest.TestCase):
    """the clip evaluator and delta solve read parents through one index of
    the rig; on an awkward rig they still match single poses frame by frame"""

    def _rig(self, seed: int, shared: bool) -> HierarchyData:
        """a node named like a uuid, parents the rig does not hold, two
        levels of branching; with ``shared`` two nodes share a name, and the
        children of the first hang under it by its uuid"""
        rng = np.random.default_rng(seed)
        names = [
            "root", "spine", "twin", "twin" if shared else "twin2", _UUID_SHAPED_NAME,
            "arm", "hand", "leg", "foot", "toe", "loose", "stray", "tip", "nub",
        ]
        uuids = [generate_uuid() for _ in names]
        parents = [
            None, uuids[0], uuids[1], uuids[1], uuids[2], uuids[4], uuids[5], uuids[0],
            uuids[7], uuids[8], generate_uuid(), generate_uuid(), uuids[2], _UUID_SHAPED_NAME,
        ]
        return HierarchyData(
            [
                TransformData(
                    name=name,
                    node_type="joint",
                    uuid=uuid,
                    parent_uuid=parent,
                    translate=rng.normal(size=3),
                    rotate=rng.uniform(-90.0, 90.0, size=3),
                    scale=rng.uniform(0.8, 1.25, size=3),
                    joint_orient=rng.uniform(-30.0, 30.0, size=3),
                    rotate_axis=rng.uniform(-20.0, 20.0, size=3),
                    rotate_order=int(rng.integers(0, 6)),
                    segment_scale_compensate=bool(rng.integers(0, 2)),
                )
                for name, uuid, parent in zip(names, uuids, parents)
            ]
        )

    def _animated(self, rig: HierarchyData, frames: int = 7, seed: int = 9) -> ClipData:
        rng                   = np.random.default_rng(seed)
        clip                  = ClipData(rig, frames=frames, start_frame=1001, fps=30.0)
        shape                 = clip.frames.rotate.shape
        clip.frames.rotate    = clip.frames.rotate + rng.normal(size=shape) * 20.0
        clip.frames.translate = clip.frames.translate + rng.normal(size=shape) * 0.3
        clip.frames.scale     = clip.frames.scale * rng.uniform(0.8, 1.2, size=shape)
        return clip

    @staticmethod
    def _pose(clip: ClipData, frame: int) -> HierarchyData:
        """one frame of ``clip`` as a plain single pose rig of its own"""
        clip.frame = frame
        return TransformList._owned_copy(clip)

    def test_the_rig_resolves_its_awkward_parents(self):
        rig     = self._rig(5, shared=True)
        parents = dict(zip(range(len(rig)), rig.get_parents()))
        self.assertEqual(parents[4], 2)  # "twin" is the first twin
        self.assertEqual(parents[12], 2)
        for root in (0, 10, 11, 13):  # absent, or named like a uuid
            self.assertEqual(parents[root], -1, root)

    def test_world_and_local_matrices_match_single_poses(self):
        for shared in (True, False):
            clip  = self._animated(self._rig(5, shared))
            world = clip.frames.world_matrix
            local = clip.frames.matrix
            for frame in range(clip.frame_count):
                pose = self._pose(clip, frame)
                np.testing.assert_allclose(
                    world[frame], pose.world_matrix, rtol=0, atol=1e-9
                )
                np.testing.assert_allclose(local[frame], pose.matrix, rtol=0, atol=1e-9)

    def test_add_delta_matches_single_poses(self):
        # by index the shared name is fine; a name pairing refuses it
        for by, shared in (("index", True), ("name", False)):
            source = self._rig(5, shared)
            delta  = self._rig(17, shared).get_delta(source, by=by)
            clip   = self._animated(source)
            fast   = clip.add_delta(delta, by=by)
            slow = [
                self._pose(clip, frame).add_delta(delta, by=by)
                for frame in range(clip.frame_count)
            ]

            # rotate compared after euler_filter on both sides, frame 0 kept
            orders = np.asarray(fast.rotate_order, dtype=np.intp)
            np.testing.assert_allclose(
                euler_filter(np.radians(fast.frames.rotate), orders),
                euler_filter(np.radians([pose.rotate for pose in slow]), orders),
                rtol=0, atol=1e-9, err_msg=by,
            )
            world = fast.frames.world_matrix
            for frame, pose in enumerate(slow):
                np.testing.assert_allclose(
                    world[frame], pose.world_matrix, rtol=0, atol=1e-9, err_msg=by
                )

    def test_building_a_clip_checks_each_uuid_once(self):
        count = 300
        uuids = [generate_uuid() for _ in range(count)]
        rig = HierarchyData(
            [
                TransformData(f"j{i}", uuid=uuids[i], parent_uuid=uuids[(i - 1) // 2] if i else None)
                for i in range(count)
            ]
        )
        clip = ClipData(rig, frames=4)
        real = TransformData.copy

        def counted_copy(node):
            new      = real(node)
            new.uuid = _Counted(new.uuid)
            return new

        builds = {
            "ClipData(rig, frames)": lambda: ClipData(rig, frames=4),
            "clip.copy()":           clip.copy,
            "clip.frames[1:3]":      lambda: clip.frames[1:3],
        }
        with mock.patch.object(TransformData, "copy", counted_copy):
            for what, build in builds.items():
                _Counted.compared = 0
                built             = build()
                self.assertLess(_Counted.compared, 5 * count, what)
                self.assertEqual(built.name, rig.name, what)
