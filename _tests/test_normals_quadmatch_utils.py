"""Tests for the new face-varying-normal / quadrangulate utility wrappers
in ``cgmath.geometry.utils.main``.
"""

import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.utils import (
    build_tri_expand_map,
    build_uv_to_normal_map,
    quad_match_greedy,
    split_at_hard_edges,
)


def _open_plane(n=3):
    """An (n-1) x (n-1) quad grid in the XY plane."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    points  = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
    indices = []
    counts  = []
    for j in range(n - 1):
        for i in range(n - 1):
            v0 = j * n + i
            v1 = j * n + (i + 1)
            v2 = (j + 1) * n + (i + 1)
            v3 = (j + 1) * n + i
            indices.extend([v0, v1, v2, v3])
            counts.append(4)
    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    )


def _unit_cube_quads():
    """Closed unit cube of 6 quads -- every edge is a 90deg hard edge."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    quads = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 4, 7, 3),
    ]
    indices = []
    counts  = []
    for q in quads:
        indices.extend(q)
        counts.append(4)
    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    )


# ------------------------ split_at_hard_edges -------------------------


class TestSplitAtHardEdges(unittest.TestCase):
    def test_no_hard_edges_returns_one_normal_per_vertex(self):
        """All edges smooth -> unique normal slot per mesh vertex."""
        mesh = _open_plane(3)
        normal_indices, num_normals = split_at_hard_edges(
            mesh.indices,
            mesh.counts,
            mesh.e2v,
            mesh.e2f,
            np.zeros(0, dtype=np.int64),
        )
        self.assertEqual(normal_indices.shape, mesh.indices.shape)
        # 3x3 vertex grid -> at most 9 unique slots.
        self.assertLessEqual(num_normals, 9)
        # Same mesh-vertex within smooth neighborhood -> same slot.
        for vi in np.unique(mesh.indices):
            slots = np.unique(normal_indices[mesh.indices == vi])
            self.assertEqual(slots.size, 1, f"vertex {vi} got slots {slots}")

    def test_all_hard_edges_per_face_unique_normals(self):
        """Every edge hard -> each face-vertex position is its own slot."""
        mesh    = _unit_cube_quads()
        n_edges = mesh.e2f.shape[0]
        result = split_at_hard_edges(
            mesh.indices,
            mesh.counts,
            mesh.e2v,
            mesh.e2f,
            np.arange(n_edges, dtype=np.int64),
        )
        num_normals = result[1]
        self.assertEqual(num_normals, mesh.indices.size)
        # 8 mesh vertices x 3 incident faces = 24 face-vertex slots.
        self.assertEqual(num_normals, 24)

    def test_dtype_preserved(self):
        """Output dtype matches input ``indices`` dtype."""
        mesh = _open_plane(2)
        for dt in (np.int32, np.int64):
            ind = mesh.indices.astype(dt)
            out, _ = split_at_hard_edges(
                ind,
                mesh.counts,
                mesh.e2v,
                mesh.e2f,
                np.zeros(0, dtype=np.int64),
            )
            self.assertEqual(out.dtype, dt)

    def test_empty_indices(self):
        """Empty face-vertex stream returns empty output."""
        out, num = split_at_hard_edges(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.int64),
        )
        self.assertEqual(out.size, 0)
        self.assertEqual(num, 0)

    def test_hard_edges_none(self):
        """``hard_edges=None`` is equivalent to an empty array."""
        mesh = _open_plane(3)
        out, num = split_at_hard_edges(
            mesh.indices, mesh.counts, mesh.e2v, mesh.e2f, None
        )
        self.assertEqual(out.shape, mesh.indices.shape)
        self.assertGreater(num, 0)


# ------------------------ build_uv_to_normal_map ----------------------


class TestBuildUvToNormalMap(unittest.TestCase):
    def test_first_write_wins(self):
        """When a UV vertex appears multiple times, the FIRST occurrence wins."""
        uv_indices     = np.array([0, 1, 0, 2, 1])
        normal_indices = np.array([10, 20, 30, 40, 50])
        out            = build_uv_to_normal_map(uv_indices, normal_indices, 3)
        np.testing.assert_array_equal(out, [10, 20, 40])

    def test_unreferenced_uv_verts_default_to_zero(self):
        """UV verts not in the stream get the default zero."""
        uv_indices     = np.array([0, 2])
        normal_indices = np.array([7, 9])
        out            = build_uv_to_normal_map(uv_indices, normal_indices, 4)
        np.testing.assert_array_equal(out, [7, 0, 9, 0])

    def test_empty_input(self):
        """Empty indices returns a zero output of the requested length."""
        out = build_uv_to_normal_map(
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            5,
        )
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0])

    def test_zero_length_output(self):
        """num_uv_verts == 0 returns an empty array."""
        out = build_uv_to_normal_map(
            np.array([], dtype=int),
            np.array([], dtype=int),
            0,
        )
        self.assertEqual(out.size, 0)


# ------------------------ build_tri_expand_map ------------------------


class TestBuildTriExpandMap(unittest.TestCase):
    def test_pure_triangles(self):
        """All-tri mesh: identity mapping."""
        counts = np.array([3, 3, 3], dtype=np.int32)
        rules  = np.zeros(3, dtype=np.int32)
        out    = build_tri_expand_map(counts, rules)
        np.testing.assert_array_equal(out, np.arange(9))

    def test_quad_split_rule_0(self):
        """Single quad with rule 0: (1,2,0) + (0,2,3) face-vertex remap."""
        counts = np.array([4], dtype=np.int32)
        rules  = np.array([0], dtype=np.int32)
        out    = build_tri_expand_map(counts, rules)
        np.testing.assert_array_equal(out, [1, 2, 0, 0, 2, 3])

    def test_quad_split_rule_1(self):
        """Single quad with rule 1: (0,1,3) + (3,1,2) face-vertex remap."""
        counts = np.array([4], dtype=np.int32)
        rules  = np.array([1], dtype=np.int32)
        out    = build_tri_expand_map(counts, rules)
        np.testing.assert_array_equal(out, [0, 1, 3, 3, 1, 2])

    def test_mixed_topology(self):
        """Tri then quad: pos offsets correctly cumulate across faces."""
        counts = np.array([3, 4], dtype=np.int32)
        rules  = np.array([0, 0], dtype=np.int32)
        out    = build_tri_expand_map(counts, rules)
        # First face: tri at positions 0..2.
        # Second face: quad at positions 3..6, rule 0.
        np.testing.assert_array_equal(out, [0, 1, 2, 4, 5, 3, 3, 5, 6])

    def test_empty_counts(self):
        """Empty face list returns empty mapping."""
        out = build_tri_expand_map(
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        )
        self.assertEqual(out.size, 0)


# --------------------------- quad_match_greedy ------------------------


class TestQuadMatchGreedy(unittest.TestCase):
    def test_picks_highest_score_first(self):
        """Best edge picked, conflicting edges skipped."""
        # 4 candidate edges across 4 faces (0..3).
        # cand 0: faces (0, 1) score 0.9 -- best
        # cand 1: faces (1, 2) score 0.8 -- conflicts with cand 0 on face 1
        # cand 2: faces (2, 3) score 0.7 -- face 2 free, face 3 free -> pick
        # cand 3: faces (0, 3) score 0.5 -- both already used -> skip
        score = np.array([0.9, 0.8, 0.7, 0.5])
        fa    = np.array([0, 1, 2, 0])
        fb    = np.array([1, 2, 3, 3])
        pick  = quad_match_greedy(score, fa, fb, n_faces=4)
        np.testing.assert_array_equal(pick, [True, False, True, False])

    def test_skips_non_finite_scores(self):
        """Once we hit -inf, the loop breaks (descending sort)."""
        score = np.array([1.0, -np.inf, 0.5])
        fa    = np.array([0, 2, 4])
        fb    = np.array([1, 3, 5])
        pick  = quad_match_greedy(score, fa, fb, n_faces=6)
        # Sorted desc: 1.0, 0.5, -inf -> first two picks are independent.
        # But the original loop breaks on first non-finite, so 0.5 still picks.
        np.testing.assert_array_equal(pick, [True, False, True])

    def test_all_negative_infinity(self):
        """All -inf scores -> no picks."""
        score = np.full(3, -np.inf)
        fa    = np.array([0, 1, 2])
        fb    = np.array([3, 4, 5])
        pick  = quad_match_greedy(score, fa, fb, n_faces=6)
        self.assertFalse(pick.any())

    def test_empty(self):
        """Empty input returns empty output."""
        pick = quad_match_greedy(
            np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int), n_faces=0
        )
        self.assertEqual(pick.size, 0)

    def test_each_face_used_at_most_once(self):
        """Postcondition -- no face appears in two picked candidates."""
        rng     = np.random.default_rng(seed=0)
        n_faces = 50
        n_cand  = 100
        score   = rng.random(n_cand)
        fa      = rng.integers(0, n_faces, size=n_cand)
        fb      = rng.integers(0, n_faces, size=n_cand)
        # Avoid self-loops for sanity.
        for i in range(n_cand):
            if fa[i] == fb[i]:
                fb[i] = (fa[i] + 1) % n_faces
        pick = quad_match_greedy(score, fa, fb, n_faces)

        used = np.zeros(n_faces, dtype=bool)
        for k in np.where(pick)[0]:
            self.assertFalse(used[fa[k]])
            self.assertFalse(used[fb[k]])
            used[fa[k]] = True
            used[fb[k]] = True