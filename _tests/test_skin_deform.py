import gc
import os
import pickle
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from cgmath.geometry.deform import DeformMethod, SkinDeformData
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.skin_weights import SkinData
from cgmath.geometry.utils._numba import _skin_deform as kernels
from cgmath.geometry.utils._numba._skin_deform import (
    _dqs_compact_numpy,
    _dqs_decompose,
    _dqs_decompose_frames,
    _lbs_compact_numpy,
    dqs_compact_fast,
    lbs_compact_fast,
)
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, generate_uuid
from cgmath.transforms import matrix_point_multiply


EPSILON = 1e-9


def translation(t) -> np.ndarray:
    """Row-vector translation matrix -- translation lives in the LAST ROW."""
    m        = np.eye(4, dtype=np.float64)
    m[3, :3] = t
    return m


def scale(s) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def rotation(axis, degrees, translate=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Row-vector rigid matrix: ``p @ rotation(...)`` turns p about *axis*.

    Written out rather than routed through ``euler_to_matrix`` because that
    takes RADIANS, and a helper that silently reinterprets 170 as radians
    produces a 20 degree rotation and a test that proves nothing.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    cross = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])

    m = np.eye(4, dtype=np.float64)
    # row-vector is the transpose of the usual column-vector construction
    m[:3, :3] = (c * np.eye(3) + s * cross + (1.0 - c) * np.outer(a, a)).T
    m[3, :3]  = translate
    return m


def ring_points(count: int = 32) -> np.ndarray:
    """unit circle in the xy plane -- radius is what a bend test measures"""
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta), np.zeros(count)], axis=1)


def grid_mesh(n: int = 6) -> MeshData:
    xs, zs = np.meshgrid(
        np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij"
    )
    points  = np.stack([xs.ravel(), np.zeros(n * n), zs.ravel()], axis=1)

    indices = []
    for i in range(n - 1):
        for j in range(n - 1):
            indices += [i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1]

    return MeshData(
        name    = "grid",
        points  = points,
        indices = np.array(indices),
        counts  = np.full((n - 1) * (n - 1), 4, dtype=np.int64),
    )


def two_joint_skin(n_points: int, split: float = 1.0) -> SkinData:
    """`split` of each vertex on joint ``a``, the remainder on ``b``."""
    weights       = np.zeros((n_points, 2), dtype=np.float64)
    weights[:, 0] = split
    weights[:, 1] = 1.0 - split
    return SkinData(weights=weights, influences=["a", "b"])


def deformer(points, skin, ibm=None, **kwargs) -> SkinDeformData:
    if ibm is None:
        ibm = np.stack([np.eye(4)] * len(skin.influences))
    return SkinDeformData(mesh=points, skin=skin, inverse_bind_matrices=ibm, **kwargs)


def skin_matrices(count: int, frames: int, seed: int) -> np.ndarray:
    """``(frames, count, 4, 4)`` rotations with a non-uniform scale and an
    offset, the shape a posed rig's skin matrices take"""
    rng = np.random.default_rng(seed)
    out = np.empty((frames, count, 4, 4))
    for f in range(frames):
        for j in range(count):
            out[f, j] = scale(rng.uniform(0.7, 1.3, 3)) @ rotation(
                rng.normal(size=3), rng.uniform(-170.0, 170.0), rng.normal(size=3)
            )
    return out


# --- a clip to skin ------------------------------------------------------ #


def branching_clip(joints: int = 12, frames: int = 9, seed: int = 23):
    """``(rig, clip)``: a branching rig and a clip animating every framed
    channel of it.

    Every joint has a random joint orient, rotate axis, rotate order and
    scale, and half of them compensate segment scale.
    """
    rng   = np.random.default_rng(seed)
    nodes = []
    for i in range(joints):
        parent = None
        if i:
            p      = i - 1 if rng.random() < 0.7 else int(rng.integers(0, i))
            parent = nodes[p].uuid
        nodes.append(
            TransformData(
                name=f"j{i:02d}",
                uuid=generate_uuid(),
                parent_uuid=parent,
                translate=rng.normal(size=3),
                rotate=rng.uniform(-60.0, 60.0, 3),
                scale=rng.uniform(0.8, 1.25, 3),
                joint_orient=rng.uniform(-90.0, 90.0, 3),
                rotate_axis=rng.uniform(-30.0, 30.0, 3),
                rotate_order=int(rng.integers(0, 6)),
                segment_scale_compensate=bool(rng.random() < 0.5),
            )
        )
    rig   = HierarchyData(nodes)

    clip  = ClipData(rig, frames=frames, start_frame=3, fps=24.0)
    shape = (frames, joints, 3)
    for channel, step in (
        ("rotate", 8.0),
        ("translate", 0.1),
        ("joint_orient", 3.0),
        ("rotate_axis", 3.0),
    ):
        walk = np.cumsum(rng.normal(size=shape) * step, axis=0)
        setattr(clip.frames, channel, getattr(clip.frames, channel) + walk)
    clip.frames.scale = clip.frames.scale * rng.uniform(0.9, 1.1, shape)
    clip.frame        = 0
    return rig, clip


def skinned_mesh(rig, seed: int = 29):
    """``(mesh, skin)``: a small tetrahedron around every joint but the last.

    Each vertex weighs its own joint, its parent and one more joint. The
    skin lists its influences in a shuffled order, so the joint columns are
    looked up rather than lined up, and the last joint is left out of it: a
    rig may carry joints a skin does not use.
    """
    rng    = np.random.default_rng(seed)
    names  = list(rig.name)
    world  = np.asarray(rig.world_matrix)
    used   = names[:-1]
    order  = [str(x) for x in rng.permutation(used)]
    column = {name: k for k, name in enumerate(order)}
    tetra  = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])

    points, faces, rows = [], [], []
    for j, name in enumerate(used):
        parent = rig[j].get_parent()
        for _ in range(4):
            row               = np.zeros(len(order))
            row[column[name]] = rng.uniform(0.4, 1.0)
            if parent is not None:
                row[column[parent.name]] = rng.uniform(0.1, 0.5)
            row[column[used[rng.integers(len(used))]]] += rng.uniform(0.0, 0.3)
            rows.append(row / row.sum())
            points.append(world[j, 3, :3] + rng.normal(size=3) * 0.3)
        faces.append(tetra + 4 * j)

    faces = np.concatenate(faces)
    mesh = MeshData(
        name    = "tetras",
        points  = np.array(points),
        indices = faces.ravel(),
        counts  = np.full(len(faces), 3, dtype=np.int64),
    )
    return mesh, SkinData(weights=np.array(rows), influences=order)


def pose_at(clip, frame: int) -> HierarchyData:
    """frame *frame* of *clip* as a plain single-pose HierarchyData: the
    frame-by-frame reference every clip result is checked against"""
    clip.frame = frame
    return HierarchyData([node.copy() for node in clip])


class TestConvention(unittest.TestCase):
    """The package is row-vector: p' = p @ M, translation in M[3, :3]."""

    def test_single_influence_matches_the_package_point_transform(self):
        # ground truth is the library's own transform, not a hand-rolled one
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0], [0.0, 0.0, 0.0]])
        m      = translation([5.0, -2.0, 1.0]) @ scale([2.0, 3.0, 4.0])

        d   = deformer(points, two_joint_skin(3, split=1.0))
        got = d.deform(points, np.stack([m, np.eye(4)]))

        expected = matrix_point_multiply(points, np.tile(m[None], (3, 1, 1)))
        self.assertTrue(np.allclose(got, expected, atol=EPSILON))

    def test_translation_is_read_from_the_last_row(self):
        # a column-vector reading would move the points by (0, 0, 0)
        points = np.zeros((3, 3))
        d      = deformer(points, two_joint_skin(3, split=1.0))
        got    = d.deform(points, np.stack([translation([7.0, 8.0, 9.0]), np.eye(4)]))
        self.assertTrue(np.allclose(got, [7.0, 8.0, 9.0], atol=EPSILON))


class TestBlend(unittest.TestCase):
    def test_identity_pose_returns_the_bind_pose(self):
        points = grid_mesh().points
        d      = deformer(points, two_joint_skin(len(points), split=1.0))
        got    = d.deform(points, np.stack([np.eye(4), np.eye(4)]))
        self.assertTrue(np.allclose(got, points, atol=EPSILON))

    def test_a_fifty_fifty_blend_lands_on_the_midpoint(self):
        points   = np.zeros((4, 3))
        d        = deformer(points, two_joint_skin(4, split=0.5))

        matrices = np.stack([translation([10.0, 0, 0]), translation([0, 20.0, 0])])
        got      = d.deform(points, matrices)

        self.assertTrue(np.allclose(got, [5.0, 10.0, 0.0], atol=EPSILON))

    def test_weights_partition_between_two_rigid_results(self):
        points = np.array([[1.0, 1.0, 1.0]])
        a      = translation([10.0, 0, 0])
        b      = translation([0, 0, 30.0])

        for split in (0.0, 0.25, 0.5, 0.75, 1.0):
            d   = deformer(points, two_joint_skin(1, split=split))
            got = d.deform(points, np.stack([a, b]))
            expected = split * (points + [10, 0, 0]) + (1 - split) * (
                points + [0, 0, 30]
            )
            self.assertTrue(np.allclose(got, expected, atol=EPSILON), f"split={split}")

    def test_a_zero_weight_influence_contributes_nothing(self):
        # the kernel short-circuits w == 0, so a non-finite matrix behind a
        # zero weight must not leak in. The second vertex is split so that
        # compaction keeps K=2 and the zero column actually survives to the
        # kernel -- with an all-[1, 0] fixture, k collapses to 1 and the
        # garbage matrix is never referenced, making the test vacuous.
        points = np.zeros((2, 3))
        skin = SkinData(
            weights=np.array([[1.0, 0.0], [0.5, 0.5]]), influences=["a", "b"]
        )
        d = deformer(points, skin)
        d.bind()
        self.assertEqual(d.influence_indices.shape[1], 2, "fixture collapsed to K=1")

        for bad in (1e12, np.nan, np.inf):
            got = d.deform(
                points,
                np.stack([translation([1.0, 2.0, 3.0]), np.full((4, 4), bad)]),
            )
            self.assertTrue(
                np.allclose(got[0], [1.0, 2.0, 3.0], atol=EPSILON), f"bad={bad}"
            )

    def test_the_numpy_fallback_also_ignores_a_zero_weighted_nan(self):
        # the two backends are pinned as equivalent, so they must agree here
        points   = np.zeros((1, 3))
        indices  = np.array([[0, 1]], dtype=np.int32)
        weights  = np.array([[1.0, 0.0]])
        matrices = np.stack([translation([1.0, 2.0, 3.0]), np.full((4, 4), np.nan)])

        fast = lbs_compact_fast(points, indices, weights, matrices)
        slow = _lbs_compact_numpy(points, indices, weights, matrices)
        self.assertTrue(np.allclose(fast, [1.0, 2.0, 3.0], atol=EPSILON))
        self.assertTrue(np.allclose(slow, fast, atol=EPSILON))

    def test_negative_weights_are_refused_rather_than_dropped(self):
        # gather_top_k picks the largest SIGNED values, so a zero column would
        # outrank a negative weight and silently evict it
        points = np.zeros((1, 3))
        skin = SkinData(
            weights=np.array([[0.5, 0.7, -0.2, 0.0, 0.0]]), influences=list("abcde")
        )
        with self.assertRaisesRegex(ValueError, "negative skin weights"):
            deformer(points, skin).bind()


class TestKernelParity(unittest.TestCase):
    def test_numba_and_numpy_fallback_agree(self):
        rng     = np.random.default_rng(0)
        points  = rng.normal(size=(97, 3))
        indices = rng.integers(0, 3, size=(97, 4)).astype(np.int32)
        weights = rng.random((97, 4))
        weights /= weights.sum(axis=1, keepdims=True)
        matrices = rng.normal(size=(3, 4, 4))

        fast = lbs_compact_fast(points, indices, weights, matrices)
        slow = _lbs_compact_numpy(points, indices, weights, matrices)
        self.assertTrue(np.allclose(fast, slow, atol=1e-10))

    def test_compact_and_dense_skin_agree(self):
        rng     = np.random.default_rng(7)
        points  = rng.normal(size=(40, 3))
        weights = rng.random((40, 5))
        weights /= weights.sum(axis=1, keepdims=True)

        dense    = SkinData(weights=weights, influences=list("abcde"))
        compact  = dense.to_compact_skin_data()

        matrices = np.stack([translation(rng.normal(size=3)) for _ in range(5)])

        got_dense   = deformer(points, dense).deform(points, matrices)
        got_compact = deformer(points, compact).deform(points, matrices)
        self.assertTrue(np.allclose(got_dense, got_compact, atol=1e-8))

    def test_the_input_points_are_not_mutated(self):
        points   = grid_mesh().points
        original = points.copy()
        d        = deformer(points, two_joint_skin(len(points), split=1.0))
        d.deform(points, np.stack([translation([3.0, 0, 0]), np.eye(4)]))
        self.assertTrue(np.array_equal(points, original))

    def test_a_sequence_matches_the_numpy_fallback(self):
        """the batched kernels against the independent numpy expression,
        which takes the same frame axis and agrees with itself frame by frame"""
        rng     = np.random.default_rng(3)
        points  = rng.normal(size=(61, 3)) * 4.0
        indices = rng.integers(0, 5, size=(61, 3)).astype(np.int32)
        weights = rng.random((61, 3))
        weights[rng.random((61, 3)) < 0.2] = 0.0
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        matrices = skin_matrices(5, 7, seed=4)

        slow = _lbs_compact_numpy(points, indices, weights, matrices)
        for f in range(7):
            np.testing.assert_allclose(
                slow[f],
                _lbs_compact_numpy(points, indices, weights, matrices[f]),
                atol = 1e-12,
                rtol = 0,
            )
        np.testing.assert_allclose(
            lbs_compact_fast(points, indices, weights, matrices), slow, atol=1e-10
        )

        quats   = np.empty((7, 5, 4))
        duals   = np.empty((7, 5, 4))
        stretch = np.empty((7, 5, 3, 3))
        _dqs_decompose_frames(matrices, quats, duals, stretch)
        for f in range(7):
            q, d, s = np.empty((5, 4)), np.empty((5, 4)), np.empty((5, 3, 3))
            _dqs_decompose(matrices[f], q, d, s)
            np.testing.assert_array_equal(quats[f],   q)
            np.testing.assert_array_equal(duals[f],   d)
            np.testing.assert_array_equal(stretch[f], s)

        slow = _dqs_compact_numpy(points, indices, weights, quats, duals, stretch)
        for f in range(7):
            np.testing.assert_allclose(
                slow[f],
                _dqs_compact_numpy(
                    points, indices, weights, quats[f], duals[f], stretch[f]
                ),
                atol = 1e-12,
                rtol = 0,
            )
        np.testing.assert_allclose(
            dqs_compact_fast(points, indices, weights, matrices), slow, atol=1e-10
        )


class TestMapping(unittest.TestCase):
    def test_joint_order_is_honoured_over_influence_order(self):
        # skin lists a, b -- joints list b, a. The remap must follow joints.
        points = np.zeros((1, 3))
        skin   = SkinData(weights=np.array([[1.0, 0.0]]), influences=["a", "b"])

        d = SkinDeformData(
            mesh=points,
            skin=skin,
            joints=["b", "a"],
            inverse_bind_matrices=np.stack([np.eye(4), np.eye(4)]),
        )
        # column 0 is now joint "b"; the weight belongs to "a" at column 1
        got = d.deform(
            points, np.stack([translation([99.0, 0, 0]), translation([1.0, 0, 0])])
        )
        self.assertTrue(np.allclose(got, [1.0, 0.0, 0.0], atol=EPSILON))

    def test_an_influence_missing_from_joints_raises(self):
        points = np.zeros((1, 3))
        skin   = SkinData(weights=np.array([[1.0, 0.0]]), influences=["a", "ghost"])
        with self.assertRaises(ValueError):
            SkinDeformData(
                mesh=points,
                skin=skin,
                joints=["a", "b"],
                inverse_bind_matrices=np.stack([np.eye(4)] * 2),
            ).bind()

    def test_duplicate_joint_names_raise(self):
        points = np.zeros((1, 3))
        skin   = SkinData(weights=np.array([[1.0, 0.0]]), influences=["a", "b"])
        with self.assertRaises(ValueError):
            SkinDeformData(
                mesh=points,
                skin=skin,
                joints=["a", "a"],
                inverse_bind_matrices=np.stack([np.eye(4)] * 2),
            ).bind()

    def test_a_matrix_count_that_misses_the_joints_raises(self):
        with self.assertRaises(ValueError):
            SkinDeformData(
                mesh=np.zeros((1, 3)),
                skin=two_joint_skin(1),
                inverse_bind_matrices=np.stack([np.eye(4)]),
            )

    def test_no_bind_source_raises(self):
        with self.assertRaises(ValueError):
            SkinDeformData(mesh=np.zeros((1, 3)), skin=two_joint_skin(1))


class TestRig(unittest.TestCase):
    def _rig(self, offset=(0.0, 0.0, 0.0), rotate=None) -> HierarchyData:
        rig = HierarchyData(
            [
                TransformData(name="a", translate=offset, rotate=rotate),
                TransformData(name="b", translate=(10.0, 0.0, 0.0)),
            ]
        )
        rig["b"].parent_uuid = rig["a"].uuid
        return rig

    def test_pose_indices_follows_the_joint_list(self):
        d = deformer(np.zeros((1, 3)), two_joint_skin(1), joints=["b", "a"])
        self.assertTrue(np.array_equal(d.pose_indices(self._rig()), [1, 0]))

    def test_a_joint_missing_from_the_rig_raises(self):
        d = deformer(np.zeros((1, 3)), two_joint_skin(1), joints=["a", "nope"])
        with self.assertRaises(ValueError):
            d.pose_indices(self._rig())

    def test_a_bind_rig_reproduces_the_bind_pose(self):
        rig    = self._rig()
        points = np.array([[10.0, 0.0, 0.0]])
        skin   = SkinData(weights=np.array([[0.0, 1.0]]), influences=["a", "b"])

        d   = SkinDeformData(mesh=points, skin=skin, bind_rig=rig)
        got = d.apply(rig)
        self.assertTrue(np.allclose(got, points, atol=1e-8))

    def test_posing_the_root_carries_the_child_bound_vertex(self):
        bind   = self._rig()
        points = np.array([[10.0, 0.0, 0.0]])
        skin   = SkinData(weights=np.array([[0.0, 1.0]]), influences=["a", "b"])

        d = SkinDeformData(mesh=points, skin=skin, bind_rig=bind)

        posed = self._rig(offset=(0.0, 5.0, 0.0))
        got   = d.apply(posed)
        self.assertTrue(np.allclose(got, [[10.0, 5.0, 0.0]], atol=1e-8))

    def test_a_rotated_pose_pins_the_skin_matrix_order(self):
        # THE convention test. Every other rig case here is translation-only,
        # and translations commute -- so `world @ ibm` would pass them all.
        # A rotated root makes the two orders differ by ~5 units.
        bind   = self._rig()
        points = np.array([[10.0, 0.0, 0.0], [10.0, 2.0, -3.0]])
        skin = SkinData(
            weights=np.array([[0.0, 1.0], [0.0, 1.0]]), influences=["a", "b"]
        )
        d = SkinDeformData(mesh=points, skin=skin, bind_rig=bind)

        posed = self._rig(rotate=(0.0, 0.0, 30.0))
        got   = d.apply(posed)

        # ground truth built here, not via _skin_matrices: p @ inv(bind) @ posed
        cols = d.pose_indices(posed)
        expected = matrix_point_multiply(
            points,
            np.tile(
                (
                    np.linalg.inv(np.asarray(bind.world_matrix)[cols][1])
                    @ np.asarray(posed.world_matrix)[cols][1]
                )[None],
                (2, 1, 1),
            ),
        )
        self.assertTrue(np.allclose(got, expected, atol=1e-8))

        swapped = np.asarray(posed.world_matrix)[cols] @ d.inverse_bind_matrices
        self.assertGreater(
            np.abs(d.deform(points, swapped) - expected).max(),
            1.0,
            "fixture is degenerate -- the swapped order must differ here",
        )

    def test_authored_matrices_win_over_a_bind_rig(self):
        # an IBM that bakes a transform the bind pose cannot reproduce
        rig    = self._rig()
        points = np.array([[10.0, 0.0, 0.0]])
        skin   = SkinData(weights=np.array([[0.0, 1.0]]), influences=["a", "b"])

        baked  = np.stack([np.eye(4), translation([0.0, 0.0, 100.0])])
        d = SkinDeformData(
            mesh=points, skin=skin, bind_rig=rig, inverse_bind_matrices=baked
        )
        self.assertTrue(np.allclose(d.inverse_bind_matrices, baked, atol=EPSILON))


class TestApply(unittest.TestCase):
    def test_a_mesh_target_returns_a_copy_not_the_original(self):
        mesh   = grid_mesh()
        before = mesh.points.copy()
        d      = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))

        out = d.apply(np.stack([translation([0.0, 4.0, 0.0]), np.eye(4)]), mesh)

        self.assertIsNot(out, mesh)
        self.assertTrue(np.array_equal(mesh.points, before))
        self.assertTrue(np.allclose(out.points, before + [0, 4, 0], atol=EPSILON))

    def test_the_returned_mesh_carries_no_stale_bvh(self):
        mesh = grid_mesh(12)
        mesh.raycast(
            np.array([[5.5, 10.0, 5.5]]),
            np.array([[0.0, -1.0, 0.0]]),
            forward_only = True,
            twosided     = True,
        )
        self.assertIsNotNone(mesh._bvh_bilinear)

        d   = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        out = d.apply(np.stack([translation([100.0, 0.0, 0.0]), np.eye(4)]), mesh)

        self.assertIsNone(out._bvh_bilinear)
        hit = out.raycast(
            np.array([[105.5, 10.0, 5.5]]),
            np.array([[0.0, -1.0, 0.0]]),
            forward_only = True,
            twosided     = True,
        )
        self.assertTrue(bool(hit.hit[0]))

    def test_the_returned_mesh_drops_bind_pose_normals(self):
        mesh         = grid_mesh()
        mesh.normals = np.tile([0.0, 1.0, 0.0], (len(mesh.points), 1))

        d   = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        out = d.apply(np.stack([translation([0.0, 0.0, 3.0]), np.eye(4)]), mesh)
        self.assertIsNone(out.normals)

    def test_no_target_deforms_the_bind_points(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        out  = d.apply(np.stack([translation([1.0, 0.0, 0.0]), np.eye(4)]))
        self.assertTrue(np.allclose(out, mesh.points + [1, 0, 0], atol=EPSILON))

    def test_a_batched_pose_returns_one_array_per_frame(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))

        frames = np.stack(
            [np.stack([translation([f, 0.0, 0.0]), np.eye(4)]) for f in range(5)]
        )
        out = d.apply(frames)

        self.assertEqual(out.shape, (5, len(mesh.points), 3))
        self.assertTrue(np.allclose(out[3], mesh.points + [3, 0, 0], atol=EPSILON))

    def test_a_batched_pose_with_a_mesh_target_raises(self):
        mesh   = grid_mesh()
        d      = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        frames = np.stack([np.stack([np.eye(4), np.eye(4)])] * 3)
        with self.assertRaises(ValueError):
            d.apply(frames, mesh)

    def test_a_pose_with_the_wrong_joint_count_raises(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        with self.assertRaises(ValueError):
            d.apply(np.stack([np.eye(4)] * 5))


class TestMethodSelector(unittest.TestCase):
    def test_the_field_stores_a_string_not_an_enum(self):
        d = deformer(np.zeros((1, 3)), two_joint_skin(1))
        self.assertIsInstance(d.method, str)
        self.assertEqual(d.method, "lbs")
        self.assertIs(d.method_enum, DeformMethod.LBS)

    def test_the_constructor_takes_an_enum_or_its_value(self):
        for method in (DeformMethod.LBS, "lbs"):
            d = deformer(np.zeros((1, 3)), two_joint_skin(1), method=method)
            self.assertEqual(d.method, "lbs")

    def test_an_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            deformer(np.zeros((1, 3)), two_joint_skin(1), method="wobble")

    def test_changing_method_keeps_the_bind(self):
        d = deformer(np.zeros((1, 3)), two_joint_skin(1))
        d.bind()
        indices       = d.influence_indices
        d.method_enum = DeformMethod.LBS
        self.assertIs(d.influence_indices, indices)

    def test_dqs_is_selectable_by_enum_or_value(self):
        for method in (DeformMethod.DQS, "dqs"):
            d = deformer(np.zeros((1, 3)), two_joint_skin(1), method=method)
            self.assertEqual(d.method, "dqs")
            self.assertIs(d.method_enum, DeformMethod.DQS)


class TestDualQuaternion(unittest.TestCase):
    """DQS agrees with LBS wherever blending is a no-op, and diverges only
    where LBS is the one that is wrong.
    """

    def setUp(self):
        rng         = np.random.default_rng(7)
        self.points = rng.normal(size=(64, 3)) * 3.0

    def both(self, points, indices, weights, matrices):
        return (
            lbs_compact_fast(points, indices, weights, matrices),
            dqs_compact_fast(points, indices, weights, matrices),
        )

    # -- cases where the two must agree exactly --------------------------- #

    def test_a_single_rigid_influence_matches_lbs(self):
        """with one influence there is nothing to blend, so any disagreement
        is the decomposition or the sandwich being wrong
        """
        matrices = np.stack(
            [rotation([0.4, -0.2, 0.9], 77.0, [3.0, -1.0, 0.5]), np.eye(4)]
        )
        indices       = np.zeros((64, 2), dtype=np.int32)
        weights       = np.zeros((64, 2))
        weights[:, 0] = 1.0

        lbs, dqs = self.both(self.points, indices, weights, matrices)
        self.assertTrue(np.allclose(lbs, dqs, atol=1e-10))

    def test_influences_sharing_one_rotation_match_lbs(self):
        shared        = rotation([0.3, 0.9, -0.2], 44.0, [-2.0, 5.0, 1.0])
        matrices      = np.stack([shared, shared.copy()])
        indices       = np.zeros((64, 2), dtype=np.int32)
        indices[:, 1] = 1
        weights       = np.empty((64, 2))
        weights[:, 0] = 0.3
        weights[:, 1] = 0.7

        lbs, dqs = self.both(self.points, indices, weights, matrices)
        self.assertTrue(np.allclose(lbs, dqs, atol=1e-10))

    def test_pure_translation_matches_lbs(self):
        """translations commute, so a translation-only rig is a case where
        dual quaternion blending and linear blending coincide
        """
        matrices = np.stack(
            [translation([4.0, -2.0, 1.0]), translation([0.0, 3.0, 0.0])]
        )
        indices       = np.zeros((64, 2), dtype=np.int32)
        indices[:, 1] = 1
        weights       = np.full((64, 2), 0.5)

        lbs, dqs = self.both(self.points, indices, weights, matrices)
        self.assertTrue(np.allclose(lbs, dqs, atol=1e-10))

    def test_all_zero_weights_match_lbs(self):
        matrices = np.stack([rotation([0.1, 0.5, 0.8], 30.0), np.eye(4)])
        indices  = np.zeros((64, 2), dtype=np.int32)
        weights  = np.zeros((64, 2))

        lbs, dqs = self.both(self.points, indices, weights, matrices)
        self.assertTrue(np.allclose(lbs, dqs, atol=1e-12))

    # -- the split that makes scale survive ------------------------------- #

    def test_pure_scale_is_not_dropped(self):
        """a dual quaternion cannot carry scale; without the split this
        silently returns the unscaled point
        """
        matrices = np.stack([scale([2.0, 3.0, 0.5]), np.eye(4)])
        indices  = np.zeros((64, 1), dtype=np.int32)
        weights  = np.ones((64, 1))

        got = dqs_compact_fast(self.points, indices, weights, matrices)
        self.assertTrue(np.allclose(got, self.points * [2.0, 3.0, 0.5], atol=1e-9))

    def test_rotation_and_scale_together_match_lbs_for_one_influence(self):
        matrices = np.stack(
            [scale([2.0, 0.5, 1.5]) @ rotation([0.0, 0.0, 1.0], 40.0, [1.0, 2.0, 3.0])]
        )
        indices = np.zeros((64, 1), dtype=np.int32)
        weights = np.ones((64, 1))

        lbs, dqs = self.both(self.points, indices, weights, matrices)
        self.assertTrue(np.allclose(lbs, dqs, atol=1e-9))

    def test_the_decomposition_reconstructs_the_matrix(self):
        matrices = np.stack(
            [
                scale([2.0, 0.5, 1.5])
                @ rotation([0.2, -0.6, 0.7], 51.0, [1.0, 2.0, 3.0]),
                rotation([0.0, 1.0, 0.0], 90.0, [-4.0, 0.0, 2.0]),
                scale([-1.0, 1.0, 1.0]),
            ]
        )
        quats   = np.empty((3, 4))
        duals   = np.empty((3, 4))
        stretch = np.empty((3, 3, 3))
        _dqs_decompose(matrices, quats, duals, stretch)

        # rebuild R through the package's own converter rather than a
        # hand-written formula, so the test cannot agree with a wrong
        # convention just because both sides share my arithmetic
        from cgmath.transforms._numba._quaternion import _quaternion_to_matrix

        rebuilt = _quaternion_to_matrix(quats)

        for index in range(3):
            rot = rebuilt[index][:3, :3]

            self.assertAlmostEqual(np.linalg.norm(quats[index]), 1.0, places=12)
            self.assertGreater(np.linalg.det(rot), 0.0)
            self.assertTrue(
                np.allclose(stretch[index] @ rot, matrices[index, :3, :3], atol=1e-9),
                f"S @ R != A for matrix {index}",
            )

    def test_a_mirrored_joint_stays_finite(self):
        """a negative determinant polar-decomposes to a reflection, whose
        quaternion is meaningless unless the flip is folded into the stretch
        """
        matrices = np.stack([scale([-1.0, 1.0, 1.0]), np.eye(4)])
        indices  = np.zeros((64, 1), dtype=np.int32)
        weights  = np.ones((64, 1))

        got = dqs_compact_fast(self.points, indices, weights, matrices)
        self.assertTrue(np.isfinite(got).all())
        self.assertTrue(np.allclose(got, self.points * [-1.0, 1.0, 1.0], atol=1e-9))

    # -- where DQS is supposed to beat LBS -------------------------------- #

    def test_a_bend_holds_volume_where_lbs_collapses(self):
        ring          = ring_points()
        matrices      = np.stack([np.eye(4), rotation([0.0, 0.0, 1.0], 120.0)])
        indices       = np.zeros((len(ring), 2), dtype=np.int32)
        indices[:, 1] = 1
        weights       = np.full((len(ring), 2), 0.5)

        lbs, dqs = self.both(ring, indices, weights, matrices)

        self.assertLess(np.linalg.norm(lbs, axis=1).mean(), 0.99)
        self.assertAlmostEqual(np.linalg.norm(dqs, axis=1).mean(), 1.0, places=9)

    def test_negating_an_influence_quaternion_changes_nothing(self):
        """q and -q name the same rotation, and so do (q, d) and (-q, -d).
        The blend must not be able to tell them apart. This is the property
        the pivot sign fix exists to preserve, tested directly on the kernel
        because it is the definition rather than a consequence.
        """
        from cgmath.geometry.utils._numba._skin_deform import _dqs_compact

        matrices = np.stack(
            [
                rotation([1.0, 0.0, 0.0], 100.0, [1.0, 2.0, 3.0]),
                rotation([0.0, 1.0, 0.3], 140.0, [-2.0, 0.0, 1.0]),
            ]
        )
        quats   = np.empty((2, 4))
        duals   = np.empty((2, 4))
        stretch = np.empty((2, 3, 3))
        _dqs_decompose(matrices, quats, duals, stretch)

        indices       = np.zeros((len(self.points), 2), dtype=np.int32)
        indices[:, 1] = 1
        weights       = np.full((len(self.points), 2), 0.5)

        straight = np.empty_like(self.points)
        _dqs_compact(self.points, indices, weights, quats, duals, stretch, straight)

        quats[1] = -quats[1]
        duals[1] = -duals[1]

        flipped  = np.empty_like(self.points)
        _dqs_compact(self.points, indices, weights, quats, duals, stretch, flipped)

        self.assertTrue(np.allclose(straight, flipped, atol=1e-12))

    def test_an_antipodal_pair_reachable_from_real_matrices(self):
        """the extractor half-canonicalizes signs, but a search over random
        rotation pairs still finds dot(q0, q1) < 0 about a quarter of the
        time, so this is an ordinary case rather than a corner
        """
        matrices = np.stack(
            [
                rotation([-1.282556, 0.945063, 1.388134], -178.093),
                rotation([0.710477, -0.396075, -0.704789], -169.131),
            ]
        )
        quats   = np.empty((2, 4))
        duals   = np.empty((2, 4))
        stretch = np.empty((2, 3, 3))
        _dqs_decompose(matrices, quats, duals, stretch)

        # the fixture is only meaningful while it stays antipodal
        self.assertLess(float(np.dot(quats[0], quats[1])), 0.0)

        indices       = np.zeros((len(self.points), 2), dtype=np.int32)
        indices[:, 1] = 1
        weights       = np.full((len(self.points), 2), 0.5)

        got = dqs_compact_fast(self.points, indices, weights, matrices)

        self.assertTrue(np.isfinite(got).all())

        # a rigid blend cannot move two points closer together or further
        # apart -- an unsigned blend does exactly that
        source = np.linalg.norm(self.points[:-1] - self.points[1:], axis=1)
        result = np.linalg.norm(got[:-1] - got[1:], axis=1)
        self.assertTrue(np.allclose(source, result, atol=1e-9))

    def test_a_bend_is_rigid_for_uniformly_weighted_points(self):
        """every vertex here shares one set of weights, so they all undergo
        the same rigid transform and every pairwise distance is preserved
        """
        ring = ring_points()
        matrices = np.stack(
            [
                rotation([0.0, 0.0, 1.0], 55.0, [2.0, 0.0, 0.0]),
                rotation([1.0, 1.0, 0.0], 130.0, [0.0, -3.0, 1.0]),
            ]
        )
        indices       = np.zeros((len(ring), 2), dtype=np.int32)
        indices[:, 1] = 1
        weights       = np.full((len(ring), 2), 0.5)

        got = dqs_compact_fast(ring, indices, weights, matrices)

        source = np.linalg.norm(ring[:, None] - ring[None], axis=-1)
        result = np.linalg.norm(got[:, None] - got[None], axis=-1)
        self.assertTrue(np.allclose(source, result, atol=1e-9))

    # -- parity and plumbing ---------------------------------------------- #

    def test_kernel_matches_the_numpy_fallback(self):
        rng = np.random.default_rng(11)
        matrices = np.stack(
            [
                rotation([0.4, -0.2, 0.9], 77.0, [3.0, -1.0, 0.5]),
                rotation([-0.9, 0.3, 0.1], 64.0, [-1.0, 2.0, 4.0]),
                scale([1.5, 0.8, 1.0]) @ rotation([1.0, 1.0, 1.0], 10.0),
            ]
        )
        indices = rng.integers(0, 3, size=(64, 3)).astype(np.int32)
        weights = rng.random((64, 3))
        weights /= weights.sum(axis=1, keepdims=True)

        quats   = np.empty((3, 4))
        duals   = np.empty((3, 4))
        stretch = np.empty((3, 3, 3))
        _dqs_decompose(matrices, quats, duals, stretch)

        fast = dqs_compact_fast(self.points, indices, weights, matrices)
        slow = _dqs_compact_numpy(self.points, indices, weights, quats, duals, stretch)
        self.assertTrue(np.allclose(fast, slow, atol=1e-10))

    def test_dqs_runs_through_apply_with_a_rig(self):
        mesh = grid_mesh()
        skin = two_joint_skin(len(mesh.points), split=0.5)
        d    = deformer(mesh, skin, method="dqs")

        rig = HierarchyData(
            [
                TransformData(name="a", rotate=(0.0, 0.0, 30.0)),
                TransformData(name="b", translate=(2.0, 0.0, 0.0)),
            ]
        )

        got = d.apply(rig)

        self.assertEqual(got.shape, mesh.points.shape)
        self.assertTrue(np.isfinite(got).all())

    def test_the_bind_pose_is_a_fixed_point(self):
        mesh = grid_mesh()
        skin = two_joint_skin(len(mesh.points), split=0.5)
        d    = deformer(mesh, skin, method="dqs")

        identity = np.stack([np.eye(4), np.eye(4)])

        self.assertTrue(np.allclose(d.apply(identity), mesh.points, atol=1e-9))

    def test_the_input_points_are_not_mutated(self):
        points  = self.points.copy()
        indices = np.zeros((64, 1), dtype=np.int32)
        weights = np.ones((64, 1))

        dqs_compact_fast(
            self.points, indices, weights, np.stack([rotation([1.0, 1.0, 1.0], 9.0)])
        )

        self.assertTrue(np.array_equal(self.points, points))


class TestPersistence(unittest.TestCase):
    """Rigs, meshes and skins are constructor args precisely so this works."""

    def _bound(self) -> SkinDeformData:
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=0.5), name="hero")
        d.bind()
        return d

    def _assert_same(self, a: SkinDeformData, b: SkinDeformData):
        self.assertTrue(np.allclose(a.rest_points, b.rest_points))
        self.assertEqual(list(a.joints), list(b.joints))
        self.assertTrue(np.allclose(a.inverse_bind_matrices, b.inverse_bind_matrices))
        self.assertTrue(np.array_equal(a.influence_indices, b.influence_indices))
        self.assertTrue(np.allclose(a.influence_weights, b.influence_weights))
        self.assertEqual(a.method, b.method)

    def test_copy(self):
        d = self._bound()
        self._assert_same(d, d.copy())

    def test_to_dict_keeps_every_field(self):
        data = self._bound().to_dict()
        for field in (
            "rest_points",
            "joints",
            "inverse_bind_matrices",
            "influence_indices",
            "influence_weights",
        ):
            self.assertIn(field, data, f"{field} was dropped by to_dict")

    def test_bytes(self):
        d = self._bound()
        self._assert_same(d, SkinDeformData.from_bytes(d.to_bytes()))

    def test_pickle(self):
        d = self._bound()
        self._assert_same(d, pickle.loads(pickle.dumps(d)))

    def test_json(self):
        d = self._bound()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "skin_deform.json")
            d.save_json(path)
            self._assert_same(d, SkinDeformData.load_json(path))

    def test_a_copy_still_deforms_without_its_skin(self):
        # the bound state lives in fields, so a reload never needs the SkinData
        d = self._bound().copy()
        self.assertIsNone(d._skin)
        self.assertTrue(d.valid)

        matrices = np.stack([translation([2.0, 0, 0]), translation([0, 6.0, 0])])
        got      = d.apply(matrices)
        self.assertTrue(np.allclose(got, d.rest_points + [1.0, 3.0, 0.0], atol=1e-8))

    def test_the_method_selector_survives_every_path(self):
        """the enum is stored as its .value for exactly this reason -- an Enum
        field breaks three of these four
        """
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=0.5), method="dqs")
        d.bind()

        self.assertEqual(d.copy().method, "dqs")
        self.assertEqual(SkinDeformData.from_bytes(d.to_bytes()).method, "dqs")
        self.assertEqual(pickle.loads(pickle.dumps(d)).method, "dqs")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "skin_deform.json")
            d.save_json(path)
            self.assertEqual(SkinDeformData.load_json(path).method, "dqs")

    def test_the_deformer_does_not_keep_the_bind_mesh(self):
        # only the mesh's points are used; the mesh itself was held, unread
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=0.5))
        ref  = weakref.ref(mesh)

        del mesh
        gc.collect()
        self.assertIsNone(ref())
        self.assertTrue(np.allclose(d.apply(np.stack([np.eye(4)] * 2)), grid_mesh().points))

    def test_editing_the_mesh_later_does_not_move_the_bind_pose(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=0.5))

        mesh.points[:] += 5.0
        self.assertTrue(np.allclose(d.rest_points, grid_mesh().points))

    def test_a_restored_dqs_deformer_still_deforms(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=0.5), method="dqs")
        d.bind()

        restored = d.copy()
        matrices = np.stack([rotation([0.0, 0.0, 1.0], 25.0), np.eye(4)])

        self.assertTrue(
            np.allclose(restored.apply(matrices), d.apply(matrices), atol=1e-12)
        )


class TestGuards(unittest.TestCase):
    """deform() is public and feeds an unchecked njit kernel."""

    def _three_joint(self) -> SkinDeformData:
        points  = np.zeros((4, 3))
        w       = np.zeros((4, 3))
        w[:, 2] = 1.0
        skin    = SkinData(weights=w, influences=["a", "b", "c"])
        d = SkinDeformData(
            mesh=points, skin=skin, inverse_bind_matrices=np.stack([np.eye(4)] * 3)
        )
        d.bind()
        return d

    def test_a_short_matrix_stack_raises_instead_of_reading_out_of_bounds(self):
        # numba does not bounds-check: this used to return uninitialized heap
        d = self._three_joint()
        with self.assertRaisesRegex(ValueError, "matrices has 1 entries"):
            d.deform(np.zeros((4, 3)), np.stack([np.eye(4)]))

    def test_a_batched_stack_passed_to_deform_raises(self):
        d      = self._three_joint()
        frames = np.stack([np.stack([np.eye(4)] * 3)] * 2)
        with self.assertRaisesRegex(ValueError, r"matrices must be \(3, 4, 4\)"):
            d.deform(np.zeros((4, 3)), frames)

    def test_a_bare_4x4_matrix_raises(self):
        d = self._three_joint()
        with self.assertRaisesRegex(ValueError, r"matrices must be \(3, 4, 4\)"):
            d.deform(np.zeros((4, 3)), np.eye(4))

    def test_a_point_count_mismatch_raises(self):
        d = self._three_joint()
        with self.assertRaisesRegex(ValueError, "bound weights"):
            d.deform(np.zeros((9, 3)), np.stack([np.eye(4)] * 3))

    def test_a_bare_4x4_pose_raises_value_error_not_index_error(self):
        # shape[-3] on a 2-tuple was an IndexError, violating the documented
        # contract for the likeliest caller slip
        d = self._three_joint()
        with self.assertRaisesRegex(ValueError, "pose must be"):
            d.apply(np.eye(4))

    def test_a_single_joint_pose_raises(self):
        # J=1 broadcasts cleanly in numpy, so only the explicit guard catches it
        d = self._three_joint()
        with self.assertRaisesRegex(ValueError, "pose must be"):
            d.apply(np.stack([np.eye(4)]))


class TestMeshlessDeformer(unittest.TestCase):
    """mesh=None is a supported configuration -- it must still bind."""

    def _meshless(self) -> SkinDeformData:
        skin = two_joint_skin(6, split=0.5)
        d = SkinDeformData(
            mesh=None, skin=skin, inverse_bind_matrices=np.stack([np.eye(4)] * 2)
        )
        d.bind()
        return d

    def test_it_reports_as_bound(self):
        self.assertTrue(self._meshless().valid)

    def test_apply_does_not_rebind_every_call(self):
        d     = self._meshless()
        calls = {"n": 0}
        real  = type(d)._compact

        def counting(self):
            calls["n"] += 1
            return real(self)

        type(d)._compact = counting
        try:
            for _ in range(5):
                d.apply(np.stack([np.eye(4)] * 2), np.zeros((6, 3)))
        finally:
            type(d)._compact = real

        self.assertEqual(calls["n"], 0)

    def test_a_copy_still_deforms(self):
        d = self._meshless().copy()
        self.assertTrue(d.valid)
        got = d.apply(
            np.stack([translation([4.0, 0, 0]), translation([0, 8.0, 0])]),
            np.zeros((6, 3)),
        )
        self.assertTrue(np.allclose(got, [2.0, 4.0, 0.0], atol=EPSILON))


class TestAliasing(unittest.TestCase):
    def test_the_returned_mesh_does_not_alias_the_deformer_cache(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        out  = d.apply(np.stack([translation([1.0, 0, 0]), np.eye(4)]), mesh)

        self.assertFalse(np.shares_memory(out.points, d.points))
        before = out.points.copy()
        d.apply(np.stack([translation([50.0, 0, 0]), np.eye(4)]), mesh)
        self.assertTrue(np.array_equal(out.points, before))

    def test_a_batched_apply_updates_the_points_property(self):
        mesh = grid_mesh()
        d    = deformer(mesh, two_joint_skin(len(mesh.points), split=1.0))
        frames = np.stack(
            [np.stack([translation([f, 0.0, 0.0]), np.eye(4)]) for f in range(4)]
        )
        out = d.apply(frames)
        self.assertTrue(np.allclose(d.points, out[-1], atol=EPSILON))


class TestCompactPadding(unittest.TestCase):
    def test_a_padded_compact_skin_round_trips(self):
        # max_influences < len(influences), so real padding exists and the
        # -1 sentinel path is exercised
        rng   = np.random.default_rng(11)
        dense = np.zeros((20, 6))
        for i in range(20):
            cols           = rng.choice(6, size=2, replace=False)
            w              = rng.random(2)
            dense[i, cols] = w / w.sum()

        skin    = SkinData(weights=dense, influences=list("abcdef"))
        compact = skin.to_compact_skin_data()
        self.assertLess(compact.max_influences, 6)

        points   = rng.normal(size=(20, 3))
        matrices = np.stack([translation(rng.normal(size=3)) for _ in range(6)])

        got_dense   = deformer(points, skin).deform(points, matrices)
        got_compact = deformer(points, compact).deform(points, matrices)
        self.assertTrue(np.allclose(got_dense, got_compact, atol=1e-8))


class TestBatchedPoses(unittest.TestCase):
    """``(F, J, 4, 4)`` world matrices deform every frame in one kernel call,
    each frame as a ``(J, 4, 4)`` pose on its own would."""

    def setUp(self):
        rng     = np.random.default_rng(17)
        points  = rng.normal(size=(40, 3)) * 3.0
        weights = rng.random((40, 4))
        weights[rng.random((40, 4)) < 0.3] = 0.0
        weights[:, 0] += 0.05

        self.points = points
        self.skin = SkinData(
            weights    = weights / weights.sum(axis=1, keepdims=True),
            influences = list("abcd"),
        )
        self.world = skin_matrices(4, 6, seed=19)

    def make(self, method):
        ibm = np.linalg.inv(skin_matrices(4, 1, seed=21)[0])
        return deformer(self.points, self.skin, ibm=ibm, method=method)

    def test_a_sequence_matches_one_pose_at_a_time(self):
        for method in ("lbs", "dqs"):
            with self.subTest(method):
                d        = self.make(method)
                expected = np.stack([d.apply(w) for w in self.world])
                got      = d.apply(self.world)
                self.assertEqual(got.shape, (6, 40, 3))
                np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(d.points, got[-1])

    def test_a_sequence_is_one_kernel_call(self):
        """one call per frame was the old route: F kernel launches, each far
        too small to fill the cores"""
        for method, name in (("lbs", "lbs_compact_fast"), ("dqs", "dqs_compact_fast")):
            with self.subTest(method):
                d    = self.make(method)
                real = getattr(kernels, name)
                with patch.object(kernels, name, wraps=real) as spy:
                    d.apply(self.world)
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(spy.call_args.args[3].shape, (6, 4, 4, 4))

    def test_a_numba_failure_splits_every_frame_at_once(self):
        """with numba failing, DQS splits the joints in numpy, every frame
        at once: the per joint helper is never called from python and the
        numpy split runs once a call, whatever the frame count, giving what
        the kernels give. The fallback split one joint of one frame at a
        time through the kernel's python body, F x J calls"""
        from numba.core.errors import NumbaError

        class Failing:
            """a kernel numba failed to compile: a call raises, its python
            body is still there"""

            def __init__(self, kernel):
                self.py_func = kernel.py_func

            def __call__(self, *args):
                raise NumbaError("forced")

        d = self.make("dqs")
        for frames in (1, 10, 40):
            world = skin_matrices(4, frames, seed=frames + 1)
            for label, pose in (("sequence", world), ("one pose", world[0])):
                with self.subTest(label, frames=frames):
                    want = d.apply(pose)
                    with patch.object(
                        kernels, "_dqs_decompose_frames", Failing(kernels._dqs_decompose_frames)
                    ), patch.object(
                        kernels, "_dqs_decompose", Failing(kernels._dqs_decompose)
                    ), patch.object(
                        kernels, "_polar_rotation", wraps=kernels._polar_rotation
                    ) as per_joint, patch.object(
                        kernels,
                        "_dqs_decompose_numpy",
                        wraps  = getattr(kernels, "_dqs_decompose_numpy", None),
                        create = True,
                    ) as split:
                        got = d.apply(pose)
                    self.assertEqual(per_joint.call_count, 0)
                    self.assertEqual(split.call_count,     1)
                    self.assertEqual(got.shape,            want.shape)
                    np.testing.assert_allclose(got, want, atol=1e-10, rtol=0)

    def test_the_numpy_split_matches_the_kernels(self):
        """the fallback's split against _dqs_decompose_frames and
        _dqs_decompose, a singular, a mirrored and a near zero joint among
        scaled and rotated ones"""
        from cgmath.geometry.utils._numba._skin_deform import _dqs_decompose_numpy

        world               = skin_matrices(6, 5, seed=29)
        world[0, 0, :3, :3] = 0.0
        world[0, 1, :3, :3] = np.diag([1.0, 2.0, -1.0])
        world[0, 2, :3, :3] = np.eye(3) * 1e-13
        world[1, 3, :3, :3] = np.diag([-0.5, 1.5, 1.0]) @ world[1, 3, :3, :3]

        quats, duals, stretch = np.empty((5, 6, 4)), np.empty((5, 6, 4)), np.empty((5, 6, 3, 3))
        _dqs_decompose_frames(world, quats, duals, stretch)
        for got, want in zip(_dqs_decompose_numpy(world), (quats, duals, stretch)):
            np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(_dqs_decompose_numpy(world)[0][0, 0], [0.0, 0.0, 0.0, 1.0])

        one = (np.empty((6, 4)), np.empty((6, 4)), np.empty((6, 3, 3)))
        _dqs_decompose(world[0], *one)
        for got, want in zip(_dqs_decompose_numpy(world[0]), one):
            self.assertEqual(got.shape, want.shape)
            np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)

    def test_an_empty_sequence_returns_no_frames(self):
        d   = self.make("lbs")
        got = d.apply(np.empty((0, 4, 4, 4)))
        self.assertEqual(got.shape, (0, 40, 3))
        self.assertIsNone(d.points)

    def test_a_sequence_with_the_wrong_joint_count_raises(self):
        d = self.make("lbs")
        with self.assertRaisesRegex(ValueError, "pose must be"):
            d.apply(self.world[:, :3])


class TestClipPoses(unittest.TestCase):
    """A :class:`ClipData` is skinned from its channel blocks.

    The reference is the frame-by-frame route: the same frame copied into a
    plain HierarchyData, skinned through its single-pose world matrices.
    """

    @classmethod
    def setUpClass(cls):
        cls.rig, cls.source = branching_clip()
        cls.mesh, cls.skin = skinned_mesh(cls.rig)

    def setUp(self):
        self.clip = self.source.copy()

    def make(self, method="lbs", bind_rig=None):
        rig = self.rig if bind_rig is None else bind_rig
        return SkinDeformData(self.mesh, self.skin, bind_rig=rig, method=method)

    def reference(self, d, frames=None):
        clip   = self.clip.copy()
        frames = range(clip.frame_count) if frames is None else frames
        return np.stack([d.apply(pose_at(clip, f)) for f in frames])

    def test_a_clip_deforms_at_the_frame_it_is_on(self):
        for method in ("lbs", "dqs"):
            d = self.make(method)
            for frame in (0, 4, self.clip.frame_count - 1):
                with self.subTest(method=method, frame=frame):
                    self.clip.frame = frame
                    np.testing.assert_allclose(
                        d.apply(self.clip), self.reference(d, [frame])[0], atol=1e-9
                    )

    def test_a_clip_deforms_a_mesh_target_at_its_frame(self):
        d               = self.make()
        self.clip.frame = 6
        got             = d.apply(self.clip, self.mesh)
        self.assertIsInstance(got, MeshData)
        np.testing.assert_allclose(got.points, self.reference(d, [6])[0], atol=1e-9)

    def test_every_frame_of_a_clip_deforms_in_one_call(self):
        for method in ("lbs", "dqs"):
            with self.subTest(method):
                d   = self.make(method)
                got = d.apply(self.clip.frames)
                self.assertEqual(
                    got.shape, (self.clip.frame_count, len(self.mesh.points), 3)
                )
                np.testing.assert_allclose(got, self.reference(d), atol=1e-9)
                np.testing.assert_array_equal(d.points, got[-1])

    def test_a_node_scoped_frame_axis_finds_the_joints_among_its_nodes(self):
        d     = self.make()
        names = list(self.clip.name)
        for scope in (names[::-1], names[:-1], self.skin.influences):
            with self.subTest(scope=scope[:3]):
                got = d.apply(self.clip.frames[list(scope)])
                np.testing.assert_allclose(got, self.reference(d), atol=1e-9)

    def test_a_frame_axis_without_a_joint_raises(self):
        d = self.make()
        with self.assertRaisesRegex(ValueError, "absent from the rig"):
            d.apply(self.clip.frames[list(self.clip.name)[:3]])

    def test_every_frame_of_a_clip_cannot_fill_one_mesh(self):
        with self.assertRaisesRegex(ValueError, "batched pose"):
            self.make().apply(self.clip.frames, self.mesh)

    def test_the_clip_keeps_its_frame_and_its_channels(self):
        d               = self.make()
        self.clip.frame = 5
        before          = np.array(self.clip.frames.rotate)

        d.apply(self.clip.frames)
        d.apply(self.clip)
        self.make(bind_rig=self.clip)

        self.assertEqual(self.clip.frame, 5)
        np.testing.assert_array_equal(self.clip.frames.rotate, before)

    def test_an_edit_on_the_bound_frame_is_seen(self):
        d               = self.make()
        self.clip.frame = 2
        names           = list(self.clip.name)

        self.clip[names[3]].rotate    = (10.0, 20.0, 30.0)
        self.clip[names[0]].translate = (1.0, -2.0, 0.5)

        expected = d.apply(HierarchyData([node.copy() for node in self.clip]))
        np.testing.assert_allclose(d.apply(self.clip), expected, atol=1e-9)

    def test_a_clip_without_frames_is_one_pose(self):
        d = self.make()
        np.testing.assert_allclose(
            d.apply(ClipData(self.rig)), d.apply(self.rig), atol=1e-12
        )

    def test_a_clip_binds_at_the_frame_it_is_on(self):
        self.clip.frame = 4
        got             = self.make(bind_rig=self.clip)
        ref             = self.make(bind_rig=pose_at(self.clip.copy(), 4))
        np.testing.assert_allclose(
            got.inverse_bind_matrices, ref.inverse_bind_matrices, atol=1e-9
        )
        # that frame is then the bind pose
        np.testing.assert_allclose(got.apply(self.clip), self.mesh.points, atol=1e-9)

    def test_every_frame_of_a_clip_is_not_a_bind_pose(self):
        with self.assertRaisesRegex(ValueError, "one pose"):
            self.make(bind_rig=self.clip.frames)

    def test_reading_a_frame_hands_numba_its_threads_back(self):
        """the bound frame is read on one thread, and only while it is read,
        even when the read fails. The count is set here rather than read:
        a test before this one that kept the thread taken left 1 to compare
        against, and a regression passed"""
        import numba
        from cgmath.geometry.deform import skin_deform

        if numba.config.NUMBA_NUM_THREADS < 2:
            self.skipTest("numba runs on one thread here")
        self.addCleanup(numba.set_num_threads, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(2)

        self.make().apply(self.clip)
        self.assertEqual(numba.get_num_threads(), 2)

        with patch.object(
            skin_deform, "_BoundFrame", side_effect=RuntimeError("boom")
        ), self.assertRaisesRegex(RuntimeError, "boom"):
            self.make().apply(self.clip)
        self.assertEqual(numba.get_num_threads(), 2)

    def test_a_frame_axis_held_across_a_reorder_skins_its_own_joints(self):
        # an axis read block columns in the old order against names and
        # parents in the new: 6 to 16 off the reference, or an IndexError
        # once a node had been added or taken out
        unbound = [n for n in self.clip.name if n not in self.skin.influences]
        reorders = {
            "sort":             lambda c: c.sort(key=lambda node: node.name[::-1]),
            "reverse":          lambda c: c.reverse(),
            "set_parent local": lambda c: c["j05"].set_parent("j08", world_space=False),
            "a node added":     lambda c: c.append(TransformData("extra", translate=(1, 0, 0))),
            "a node taken out": lambda c: c.remove(c[unbound[0]]),
        }
        for method in ("lbs", "dqs"):
            for label, reorder in reorders.items():
                for scoped in (False, True):
                    with self.subTest(method=method, edit=label, scoped=scoped):
                        clip      = self.source.copy()
                        self.clip = clip
                        d         = self.make(method)
                        axis      = clip.frames[list(self.skin.influences)] if scoped else clip.frames
                        reorder(clip)

                        got = d.apply(axis)  # before anything lines the clip up
                        np.testing.assert_allclose(got, self.reference(d), atol=1e-9)

    def test_a_node_put_back_from_elsewhere_is_read_at_the_bound_frame(self):
        # a node assigned into the clip keeps its own channels and is not the
        # clip's: the blocks know nothing of it, the nodes do
        for method in ("lbs", "dqs"):
            with self.subTest(method):
                self.clip       = self.source.copy()
                self.clip.frame = 3
                d               = self.make(method)
                node            = self.clip[0].copy()
                node.rotate     = (11.0, 22.0, 33.0)
                self.clip[0]    = node
                self.assertIsNot(node._hierarchy, self.clip)

                expected = HierarchyData([n.copy() for n in self.clip])
                np.testing.assert_allclose(d.apply(self.clip), d.apply(expected), atol=1e-9)
                np.testing.assert_allclose(
                    self.make(method, bind_rig=self.clip).inverse_bind_matrices,
                    self.make(method, bind_rig=expected).inverse_bind_matrices,
                    atol=1e-9,
                )

    def test_the_world_matrices_are_the_nodes_bit_for_bit(self):
        # the bound frame goes through the evaluator that mirrors
        # TransformData.matrix factor for factor, so a skinned frame is the
        # frame skinned through its nodes, to the bit: the clip's own one
        # span read lands 1e-15 away
        d     = self.make()
        every = [d._posed_world(pose_at(self.clip.copy(), f)) for f in range(self.clip.frame_count)]
        for frame in (0, 4, self.clip.frame_count - 1):
            with self.subTest(frame=frame):
                self.clip.frame = frame
                np.testing.assert_array_equal(d._posed_world(self.clip), every[frame])
        np.testing.assert_array_equal(d._posed_world(self.clip.frames), np.stack(every))

    def test_a_clip_is_read_from_its_blocks_not_node_by_node(self):
        """node by node, a world matrix costs ~30 ms a frame on a few hundred
        joints; the blocks evaluate a frame in a fraction of that"""
        d        = self.make()
        expected = self.reference(d)

        def refuse(node):
            raise AssertionError(f"{node.name}.world_matrix was read")

        self.clip.frame = 3
        with patch.object(TransformData, "world_matrix", property(refuse)):
            one   = d.apply(self.clip)
            every = d.apply(self.clip.frames)
            bound = self.make(bind_rig=self.clip)

        np.testing.assert_allclose(one, expected[3], atol=1e-9)
        np.testing.assert_allclose(every, expected, atol=1e-9)
        np.testing.assert_allclose(bound.apply(self.clip), self.mesh.points, atol=1e-9)


if __name__ == "__main__":
    unittest.main()