"""TransformData's shear against Maya.

test_assets/maya_shear.npz holds what Maya 2025 did, recorded by
tools/make_maya_shear_fixture.py (its docstring lists every array). Nothing
here needs Maya.
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


if __name__ == "__main__":
    unittest.main()
