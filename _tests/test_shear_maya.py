"""TransformData's shear against Maya.

test_assets/maya_shear.npz holds what Maya 2025 did, recorded by
tools/make_maya_shear_fixture.py (its docstring lists every array). Nothing
here needs Maya: TestParentAsMaya reads what Maya did, recorded too.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
from cgmath.hierarchy import ClipData, HierarchyData, TransformData

FIXTURE = os.path.join(os.path.dirname(__file__), "test_assets", "maya_shear.npz")


def load_fixture() -> dict:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def relative(a, b) -> float:
    """the largest entry of ``a - b``, over the largest of ``b``, at least 1:
    a child of a near zero scale holds entries near 1e12"""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def compose_rig(data: dict) -> list:
    """the compose records as TransformData, each of the chains they make"""
    nodes = []
    for i in range(len(data["compose_node_type"])):
        up = int(data["compose_parent"][i])
        nodes.append(
            TransformData(
                f"n{i}",
                node_type="joint" if data["compose_node_type"][i] else "transform",
                uuid=f"00000000-0000-4000-8000-{i:012d}",
                parent_uuid=f"00000000-0000-4000-8000-{up:012d}" if up > -1 else None,
                scale=data["compose_scale"][i],
                shear=data["compose_shear"][i],
                rotate=data["compose_rotate"][i],
                translate=data["compose_translate"][i],
                rotate_order=int(data["compose_rotate_order"][i]),
                rotate_axis=data["compose_rotate_axis"][i],
                joint_orient=data["compose_joint_orient"][i],
                segment_scale_compensate=bool(data["compose_segment_scale_compensate"][i]),
            )
        )
    return nodes


class TestCompose(unittest.TestCase):
    """node.matrix = S Sh RO R JO IS T and worldMatrix as Maya evaluates them,
    every rotate order, every scale and shear kind"""

    @classmethod
    def setUpClass(cls):
        cls.data  = load_fixture()
        cls.nodes = compose_rig(cls.data)
        cls.rig   = HierarchyData(cls.nodes)

    def test_matrix_is_mayas(self):
        self.assertEqual(len(self.nodes), 216)
        worst = max(
            relative(node.matrix, self.data["compose_matrix"][i]) for i, node in enumerate(self.nodes)
        )
        self.assertLessEqual(worst, 1e-14)

    def test_world_matrix_is_mayas(self):
        worst = max(
            relative(node.world_matrix, self.data["compose_world_matrix"][i]) for i, node in enumerate(self.nodes)
        )
        self.assertLessEqual(worst, 1e-14)

    def test_a_clip_reads_mayas_on_every_node(self):
        # the batched evaluators, which build S * Sh on their own
        clip = ClipData(self.rig, frames=2)
        self.assertLessEqual(relative(clip.frames.matrix[1], self.data["compose_matrix"]), 1e-14)
        self.assertLessEqual(relative(clip.frames.world_matrix[1], self.data["compose_world_matrix"]), 1e-14)

    def test_the_factors_multiply_to_it(self):
        for node in self.nodes:
            with self.subTest(node=node.name):
                product = (
                    node.scale_matrix
                    @ node.shear_matrix
                    @ node.rotate_axis_matrix
                    @ node.rotate_matrix
                    @ node.joint_orient_matrix
                    @ node.parent_scale_inverse_matrix
                    @ node.translate_matrix
                )
                self.assertLessEqual(relative(product, node.matrix), 1e-14)

    def test_sh_is_one_matrix_with_no_cross_term(self):
        # xy goes in row y and yz in row z: row z of Sh holds xz as it is,
        # where a product of three shears would give xz + xy * yz
        node    = TransformData("a", shear=(0.5, -0.25, 0.75))
        rows    = node.matrix
        sheared = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [-0.25, 0.75, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(np.array_equal(rows, sheared))
        self.assertTrue(np.array_equal(node.shear_matrix, sheared))

    def test_scale_comes_after_the_shear(self):
        # S Sh: a row's whole shear takes that row's scale
        got = TransformData("a", scale=(1.0, 2.0, 3.0), shear=(0.5, -0.25, 0.75)).matrix
        self.assertEqual(got[1, :3].tolist(), [1.0, 2.0, 0.0])
        self.assertEqual(got[2, :3].tolist(), [-0.75, 2.25, 3.0])


class TestSplit(unittest.TestCase):
    """the matrix setters against xform -m and -ws -m. Maya's Euler branch is
    not asserted: rotate is compared as matrices"""

    @classmethod
    def setUpClass(cls):
        cls.data  = load_fixture()
        cls.kinds = [str(kind) for kind in cls.data["scale_kinds"]]

    def lossy(self, i: int) -> bool:
        # a scale below about 1e-10 is lossy in Maya's split
        return self.kinds[self.data["split_scale_kind"][i]] == "near_zero_1e-12"

    def solved(self, i: int) -> TransformData:
        """record ``i``'s node, under a transform holding its parent's world,
        after the target of that record went through the setter"""
        data   = self.data
        kind   = "joint" if data["split_node_type"][i] else "transform"
        parent = None
        nodes  = []
        if data["split_has_parent"][i]:
            parent        = TransformData("p", uuid="00000000-0000-4000-8000-000000000001")
            parent.matrix = data["split_parent_world"][i]
            nodes.append(parent)
        node = TransformData(
            "n",
            node_type=kind,
            uuid="00000000-0000-4000-8000-000000000002",
            parent_uuid=None if parent is None else parent.uuid,
            scale=data["split_before_scale"][i],
            shear=data["split_before_shear"][i],
            rotate=data["split_before_rotate"][i],
            translate=data["split_before_translate"][i],
            rotate_order=int(data["split_rotate_order"][i]),
            rotate_axis=data["split_before_rotate_axis"][i],
            joint_orient=data["split_before_joint_orient"][i],
            segment_scale_compensate=bool(data["split_segment_scale_compensate"][i]),
        )
        nodes.append(node)
        HierarchyData(nodes)
        if data["split_space"][i]:
            node.world_matrix = data["split_target"][i]
        else:
            node.matrix = data["split_target"][i]
        return node

    def rotate_gap(self, i: int, node: TransformData) -> float:
        data = self.data
        maya = TransformData("m", rotate=data["split_after_rotate"][i], rotate_order=int(data["split_rotate_order"][i]))
        return float(np.abs(node.rotate_matrix - maya.rotate_matrix).max())

    def test_a_transform_is_xform_m(self):
        data = self.data
        seen = 0
        for i in np.flatnonzero(data["split_node_type"] == 0).tolist():
            if self.lossy(i):
                continue
            seen += 1
            node = self.solved(i)
            with self.subTest(record=i, kind=self.kinds[data["split_scale_kind"][i]]):
                self.assertLessEqual(relative(node.matrix, data["split_matrix_after"][i]),    1e-12)
                self.assertLessEqual(np.abs(node.scale - data["split_after_scale"][i]).max(), 1e-12)
                self.assertLessEqual(np.abs(node.shear - data["split_after_shear"][i]).max(), 1e-12)
                self.assertTrue(np.array_equal(node.rotate_axis, data["split_before_rotate_axis"][i]))
                self.assertLessEqual(self.rotate_gap(i, node), 1e-10)
        self.assertEqual(seen, 63)

    def test_a_transform_lands_where_mayas_split_is_lossy(self):
        # Maya's own split misses these targets, 0.75 from what it wrote
        data = self.data
        for i in np.flatnonzero(data["split_node_type"] == 0).tolist():
            if self.lossy(i):
                node = self.solved(i)
                got  = node.world_matrix if data["split_space"][i] else node.matrix
                with self.subTest(record=i):
                    self.assertLessEqual(relative(got, data["split_target"][i]), 1e-11)

    def test_a_joint_is_xform_m(self):
        # a compensated joint under a parent, 24 records, is a fact and no
        # target: Maya misses a target it can hold there, cgmath strips IS
        # from the right
        data = self.data
        seen = 0
        for i in np.flatnonzero(data["split_node_type"] == 1).tolist():
            compensated = data["split_segment_scale_compensate"][i] and data["split_has_parent"][i]
            if self.lossy(i) or compensated:
                continue
            seen += 1
            node = self.solved(i)
            with self.subTest(record=i, kind=self.kinds[data["split_scale_kind"][i]]):
                self.assertLessEqual(relative(node.matrix, data["split_matrix_after"][i]), 1e-12)
                self.assertLessEqual(np.abs(node.scale - data["split_after_scale"][i]).max(), 1e-12)
                self.assertTrue(np.array_equal(node.shear, data["split_before_shear"][i]))
                self.assertTrue(np.array_equal(node.joint_orient, data["split_before_joint_orient"][i]))
                self.assertTrue(np.array_equal(node.rotate_axis, data["split_before_rotate_axis"][i]))
        self.assertEqual(seen, 44)


class TestParentAsMaya(unittest.TestCase):
    """set_parent against cmds.parent on a transform: twelve random trials,
    as recorded (tools/make_maya_fbx_fixtures.py, shear_parent)"""

    def test_set_parent_is_cmds_parent_on_a_transform(self):
        from cgmath._tests.maya_fixtures import recorded

        case = recorded("shear_parent")
        rng = np.random.default_rng(13)
        for trial in range(12):
            parent = TransformData(
                "p",
                uuid      = "00000000-0000-4000-8000-000000000001",
                scale     = rng.uniform(0.4, 2.2, 3),
                rotate    = rng.uniform(-170.0, 170.0, 3),
                translate = rng.uniform(-3.0, 3.0, 3),
                shear     = rng.uniform(-1.0, 1.0, 3) if trial % 2 else None,
            )
            child = TransformData(
                "c",
                uuid         = "00000000-0000-4000-8000-000000000002",
                scale        = rng.uniform(0.4, 2.2, 3),
                rotate       = rng.uniform(-170.0, 170.0, 3),
                translate    = rng.uniform(-3.0, 3.0, 3),
                rotate_order = int(rng.integers(0, 6)),
                rotate_axis  = rng.uniform(-30.0, 30.0, 3),
            )
            HierarchyData([parent, child])
            child.set_parent("p")

            matrix = case.array(f"trial{trial}/m")
            with self.subTest(trial=trial):
                self.assertLessEqual(np.abs(child.scale - case.array(f"trial{trial}/s")).max(),  1e-12)
                self.assertLessEqual(np.abs(child.shear - case.array(f"trial{trial}/sh")).max(), 1e-12)
                self.assertLessEqual(np.abs(child.matrix[:3, :3] - matrix[:3, :3]).max(),        1e-12)
                self.assertLessEqual(np.abs(child.matrix[3, :3] - matrix[3, :3]).max(),          1e-12)


if __name__ == "__main__":
    unittest.main()
