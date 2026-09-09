import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData, UVData


def _triangulated_cube():
    """Returns a MeshData of the unit cube split into 12 triangles.

    Each of the 6 quad faces is split along its (q0, q2) diagonal so that
    quadrangulate() should recover all 6 original quads.
    """
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=float,
    )
    quads = [
        (0, 3, 2, 1),  # bottom (-Z)
        (4, 5, 6, 7),  # top    (+Z)
        (0, 1, 5, 4),  # front  (-Y)
        (2, 3, 7, 6),  # back   (+Y)
        (1, 2, 6, 5),  # right  (+X)
        (0, 4, 7, 3),  # left   (-X)
    ]
    indices = []
    counts  = []
    for q in quads:
        indices.extend([q[0], q[1], q[2]])
        indices.extend([q[0], q[2], q[3]])
        counts.extend([3, 3])
    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    ), quads


def _triangulated_grid(n=3):
    """Returns a triangulated planar (n x n) grid of quads.

    n=3 produces a 2x2 grid (4 quads, 8 triangles) on a 3x3 vertex lattice.
    Each quad is split along its (0, 2) diagonal.
    """
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="xy")
    zs      = np.zeros_like(xs)
    points  = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)

    indices = []
    counts  = []
    for j in range(n - 1):
        for i in range(n - 1):
            v0 = j * n + i
            v1 = j * n + (i + 1)
            v2 = (j + 1) * n + (i + 1)
            v3 = (j + 1) * n + i
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
            counts.extend([3, 3])
    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    )


def _quad_mesh():
    """A pure-quad mesh (no triangles). quadrangulate() must be a no-op."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return MeshData(
        points  = points,
        indices = np.array([0, 1, 2, 3], dtype=np.int64),
        counts  = np.array([4], dtype=np.int64),
    )


class TestQuadrangulate(unittest.TestCase):
    def _quad_set(self, mesh):
        """Returns the set of (frozenset, count) pairs for a mesh's faces."""
        out    = set()
        cursor = 0
        for c in mesh.counts:
            verts = frozenset(int(v) for v in mesh.indices[cursor : cursor + c])
            out.add((verts, int(c)))
            cursor += c
        return out

    # ----- get_quadrangulate_rules -----

    def test_rules_recovers_six_pairs_on_cube(self):
        mesh, _ = _triangulated_cube()
        rules = mesh.get_quadrangulate_rules()
        self.assertEqual(rules.shape, (6, 4))
        # Each row is (fa, fb, oa_slot, ob_slot).  fa and fb must be
        # in-range face indices to triangles, slots must be in [0, 3).
        for fa, fb, oa_slot, ob_slot in rules:
            self.assertGreaterEqual(fa, 0)
            self.assertGreaterEqual(fb, 0)
            self.assertEqual(mesh.counts[fa], 3)
            self.assertEqual(mesh.counts[fb], 3)
            self.assertIn(int(oa_slot), (0, 1, 2))
            self.assertIn(int(ob_slot), (0, 1, 2))

    def test_rules_empty_on_pure_quad_mesh(self):
        mesh  = _quad_mesh()
        rules = mesh.get_quadrangulate_rules()
        self.assertEqual(rules.shape, (0, 4))

    def test_rules_min_score_gates_all_merges(self):
        # min_score above the maximum possible total weight kills every merge.
        mesh, _ = _triangulated_cube()
        rules = mesh.get_quadrangulate_rules(min_score=10.0)
        self.assertEqual(rules.shape, (0, 4))

    def test_rules_require_convex_false_still_runs(self):
        # All cube candidates are convex so result should match the default.
        mesh, _ = _triangulated_cube()
        with_convex    = mesh.get_quadrangulate_rules(require_convex=True)
        without_convex = mesh.get_quadrangulate_rules(require_convex=False)
        self.assertEqual(with_convex.shape, without_convex.shape)

    def test_rules_custom_weights_run_without_error(self):
        mesh, _ = _triangulated_cube()
        rules = mesh.get_quadrangulate_rules(
            planarity_weight  = 2.0,
            squareness_weight = 0.0,
            diagonal_weight   = 0.0,
        )
        self.assertEqual(rules.shape, (6, 4))

    # ----- quadrangulate -----

    def test_quadrangulate_recovers_cube_quads(self):
        mesh, original_quads = _triangulated_cube()
        mesh.quadrangulate()

        # 6 quads, 0 tris.
        self.assertEqual(int(np.sum(mesh.counts == 4)), 6)
        self.assertEqual(int(np.sum(mesh.counts == 3)), 0)
        self.assertEqual(mesh.indices.size,             24)

        # The recovered face vertex sets must equal the originals.
        recovered = set()
        cursor    = 0
        for c in mesh.counts:
            recovered.add(frozenset(int(v) for v in mesh.indices[cursor : cursor + c]))
            cursor += c
        expected = {frozenset(q) for q in original_quads}
        self.assertEqual(recovered, expected)

    def test_quadrangulate_recovers_grid_quads(self):
        mesh = _triangulated_grid(n=3)
        mesh.quadrangulate()
        self.assertEqual(int(np.sum(mesh.counts == 4)), 4)
        self.assertEqual(int(np.sum(mesh.counts == 3)), 0)

    def test_quadrangulate_no_op_on_pure_quad_mesh(self):
        mesh   = _quad_mesh()
        before = self._quad_set(mesh)
        mesh.quadrangulate()
        after = self._quad_set(mesh)
        self.assertEqual(before, after)

    def test_quadrangulate_no_op_on_empty_rules(self):
        mesh, _ = _triangulated_cube()
        before = (mesh.indices.copy(), mesh.counts.copy())
        mesh.quadrangulate(rules=np.empty((0, 4), dtype=int))
        np.testing.assert_array_equal(mesh.indices, before[0])
        np.testing.assert_array_equal(mesh.counts, before[1])

    def test_quadrangulate_explicit_rules(self):
        mesh, _ = _triangulated_cube()
        rules = mesh.get_quadrangulate_rules()
        mesh.quadrangulate(rules=rules)
        self.assertEqual(int(np.sum(mesh.counts == 4)), 6)

    def test_quadrangulate_leaves_unmerged_tris_alone(self):
        # Build a mesh with a single triangle that has no triangle neighbors:
        # one quad + one triangle attached to its border.
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.5, 0.0],  # apex
            ]
        )
        mesh = MeshData(
            points  = points,
            indices = np.array([0, 1, 2, 3, 1, 4, 2], dtype=np.int64),
            counts  = np.array([4, 3], dtype=np.int64),
        )
        mesh.quadrangulate()
        # The triangle has no triangle neighbor so it must remain a triangle.
        self.assertEqual(int(np.sum(mesh.counts == 3)), 1)
        self.assertEqual(int(np.sum(mesh.counts == 4)), 1)

    def test_quadrangulate_resets_cached_data(self):
        mesh, _ = _triangulated_cube()
        # Force a cache populate.
        _ = mesh.f2v
        _ = mesh.ue2f
        mesh.quadrangulate()
        # After mutation, caches must be cleared.
        self.assertIsNone(mesh._f2v)
        self.assertIsNone(mesh._ue2f)
        # Re-accessing them should rebuild correctly.
        self.assertEqual(mesh.f2v.shape[0], 6)

    def test_quadrangulate_rejects_non_planar_pair(self):
        # A 90-degree fold made of 4 triangles: two pairs share an edge but
        # each pair lies in a different plane. The fold edge has dihedral 0,
        # so the across-fold candidate should score very low.  We use a high
        # min_score to enforce: only well-aligned pairs survive.
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Triangles share edge (0,1): one in XY plane, the other in XZ plane.
        mesh = MeshData(
            points  = points,
            indices = np.array([0, 1, 2, 0, 3, 1], dtype=np.int64),
            counts  = np.array([3, 3], dtype=np.int64),
        )
        # Default min_score=0 with planarity term in [0,1] still allows the
        # 90-degree merge.  Bump min_score to verify the gate rejects it.
        rules = mesh.get_quadrangulate_rules(min_score=1.5)
        self.assertEqual(rules.shape, (0, 4))

    # ----- cross-data-type application: rules apply to UVData too -----

    def test_rules_apply_to_seamed_uvdata(self):
        # Two adjacent quads (sharing edge 1-2) triangulated.  In MESH the
        # two quads share vertices (1, 2).  In UV, a SEAM separates them,
        # so the second quad's UV vertices are entirely fresh (4-7).
        #
        # Within each individual quad, both UV triangles still share the
        # diagonal vertices (that's how triangulation always works), so
        # quadrangulate can recover both quads in UV using the same
        # face-local rules generated from the mesh.
        #
        #   mesh quad A: [0,1,2,3]   mesh quad B: [1,5,6,2]
        #   UV  quad A: [0,1,2,3]   UV  quad B: [4,5,6,7]   <-- seam
        mesh = MeshData(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            # Quad A split (0-2): [0,1,2] + [0,2,3]
            # Quad B split (1-2): [1,4,2] + [4,5,2]
            indices = np.array([0, 1, 2, 0, 2, 3, 1, 4, 2, 4, 5, 2], dtype=np.int64),
            counts  = np.array([3, 3, 3, 3], dtype=np.int64),
        )
        uv = UVData(
            points=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [2.0, 0.0],  # seam-side fresh UVs
                    [3.0, 0.0],
                    [3.0, 1.0],
                    [2.0, 1.0],
                ]
            ),
            # Same per-face structure; quad B uses fresh UV vertices.
            indices = np.array([0, 1, 2, 0, 2, 3, 4, 5, 7, 5, 6, 7], dtype=np.int64),
            counts  = np.array([3, 3, 3, 3], dtype=np.int64),
        )

        rules = mesh.get_quadrangulate_rules()
        self.assertEqual(rules.shape[0], 2)

        # Same rules drive both - no get_face_normals call on UVData.
        mesh.quadrangulate(rules)
        uv.quadrangulate(rules)

        self.assertEqual(int(np.sum(mesh.counts == 4)), 2)
        self.assertEqual(int(np.sum(uv.counts == 4)),   2)
        self.assertEqual(int(np.sum(mesh.counts == 3)), 0)
        self.assertEqual(int(np.sum(uv.counts == 3)),   0)

        # The two UV quads must use the seamed indices: one from {0,1,2,3}
        # and the other from {4,5,6,7}.  Mesh quads share vertices 1 and 2.
        cursor       = 0
        uv_face_sets = []
        for c in uv.counts:
            uv_face_sets.append({int(v) for v in uv.indices[cursor : cursor + c]})
            cursor += c
        self.assertEqual(
            sorted([sorted(s) for s in uv_face_sets]),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )

    def test_get_rules_works_on_uvdata_2d(self):
        # UVData has 2D points.  get_quadrangulate_rules() must NOT touch
        # face_normals (which is undefined for 2D) and must return rules
        # that are valid to apply back to the corresponding MeshData.
        mesh = MeshData(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            counts  = np.array([3, 3], dtype=np.int64),
        )
        uv = UVData(
            points=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            counts  = np.array([3, 3], dtype=np.int64),
        )

        # Rules generated FROM the 2D UV data...
        rules = uv.get_quadrangulate_rules()
        self.assertEqual(rules.shape, (1, 4))

        # ...applied back to BOTH 3D mesh and 2D uv.
        mesh.quadrangulate(rules)
        uv.quadrangulate(rules)
        self.assertEqual(int(np.sum(mesh.counts == 4)), 1)
        self.assertEqual(int(np.sum(uv.counts == 4)), 1)

    def test_quadrangulate_preserves_winding(self):
        # Original quad (0, 3, 2, 1) split into triangles (0, 3, 2) +
        # (0, 2, 1).  The recovered quad should be a cyclic rotation of
        # the source winding (i.e. visit the same vertices in the same
        # rotational order).
        mesh = MeshData(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            indices = np.array([0, 3, 2, 0, 2, 1], dtype=np.int64),
            counts  = np.array([3, 3], dtype=np.int64),
        )
        mesh.quadrangulate()

        recovered = [int(v) for v in mesh.indices[:4]]
        # Original is (0, 3, 2, 1); valid CCW rotations:
        valid = {
            (0, 3, 2, 1),
            (3, 2, 1, 0),
            (2, 1, 0, 3),
            (1, 0, 3, 2),
        }
        self.assertIn(tuple(recovered), valid)

    # ----- n-gon behaviour -----

    def test_quadrangulate_passes_ngons_through_unchanged(self):
        # 5-gon (face 0) + 4 triangles forming 2 quads.  After
        # quadrangulate the 5-gon must be preserved as a 5-gon, untouched
        # by the merging that happens to the surrounding triangles.
        mesh = MeshData(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    # quad 1
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    # quad 2
                    [5.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [6.0, 1.0, 0.0],
                    [5.0, 1.0, 0.0],
                ]
            ),
            indices=np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,  # face 0: 5-gon
                    5,
                    6,
                    7,
                    5,
                    7,
                    8,  # faces 1-2: tris of quad1
                    9,
                    10,
                    11,
                    9,
                    11,
                    12,  # faces 3-4: tris of quad2
                ],
                dtype=np.int64,
            ),
            counts=np.array([5, 3, 3, 3, 3], dtype=np.int64),
        )
        mesh.quadrangulate()
        self.assertEqual(int(np.sum(mesh.counts == 5)), 1)
        self.assertEqual(int(np.sum(mesh.counts == 4)), 2)
        self.assertEqual(int(np.sum(mesh.counts == 3)), 0)
        # The 5-gon was emitted first (input order preserved for unmerged
        # faces) so it stays at face 0 with its original vertex set.
        self.assertEqual(int(mesh.counts[0]), 5)
        self.assertEqual(
            sorted(int(v) for v in mesh.indices[: mesh.counts[0]]),
            [0, 1, 2, 3, 4],
        )

    def test_quadrangulate_remaps_hole_faces(self):
        # 4 triangles (faces 0-3) forming 2 quads, then a 5-gon (face 4)
        # that owns a hole.  After merging the triangles, the 5-gon
        # SHIFTS from input face 4 to output face 2 - hole_faces must
        # be remapped accordingly.
        mesh = MeshData(
            points=np.array(
                [
                    # quad 1
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    # quad 2
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    # 5-gon
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [6.0, 0.5, 0.0],
                    [5.0, 1.0, 0.0],
                    [4.0, 1.0, 0.0],
                ]
            ),
            indices=np.array(
                [
                    0,
                    1,
                    2,
                    0,
                    2,
                    3,  # faces 0-1: tris of quad 1
                    4,
                    5,
                    6,
                    4,
                    6,
                    7,  # faces 2-3: tris of quad 2
                    8,
                    9,
                    10,
                    11,
                    12,  # face 4: 5-gon
                ],
                dtype=np.int64,
            ),
            counts       = np.array([3, 3, 3, 3, 5], dtype=np.int64),
            hole_faces   = np.array([4], dtype=np.int64),
            hole_counts  = np.array([3], dtype=np.int64),
            hole_indices = np.array([8, 9, 10], dtype=np.int64),
        )
        mesh.quadrangulate()
        self.assertEqual(int(np.sum(mesh.counts == 4)), 2)
        self.assertEqual(int(np.sum(mesh.counts == 5)), 1)
        # The 5-gon was input face 4; after the two pairs collapse, it
        # should now be output face 2.
        ngon_pos = int(np.where(mesh.counts == 5)[0][0])
        self.assertEqual(ngon_pos, 2)
        # hole_faces must point at the new position of the 5-gon.
        self.assertEqual(int(mesh.hole_faces[0]), ngon_pos)
        # hole_counts and hole_indices reference vertex/hole data, not
        # face indices - they must be untouched.
        np.testing.assert_array_equal(mesh.hole_counts, np.array([3]))
        np.testing.assert_array_equal(mesh.hole_indices, np.array([8, 9, 10]))