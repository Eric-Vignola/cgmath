import unittest

import numpy as np
from cgmath.geometry.mesh import UVData


def _make_quad_island(origin, size=0.4):
    """Create a single quad island (4 verts, 1 face) at the given origin."""
    ox, oy = origin
    points = np.array([
        [ox,        oy       ],
        [ox + size, oy       ],
        [ox + size, oy + size],
        [ox,        oy + size],
    ])
    indices = np.array([0, 1, 2, 3])
    counts  = np.array([4])
    return points, indices, counts


def _make_two_island_uv():
    """Return a UVData with two disconnected quad islands."""
    p0, i0, c0 = _make_quad_island((0.0, 0.0), size=0.4)
    p1, i1, c1 = _make_quad_island((2.0, 2.0), size=0.4)
    return UVData(
        points  = np.vstack([p0, p1]),
        indices = np.concatenate([i0, i1 + len(p0)]),
        counts  = np.concatenate([c0, c1]),
    )


def _make_three_island_uv():
    """Return a UVData with three disconnected quad islands of different sizes."""
    p0, i0, c0 = _make_quad_island((0.0, 0.0), size=0.5)
    p1, i1, c1 = _make_quad_island((3.0, 3.0), size=0.3)
    p2, i2, c2 = _make_quad_island((6.0, 0.0), size=0.2)
    return UVData(
        points  = np.vstack([p0, p1, p2]),
        indices = np.concatenate([i0, i1 + len(p0), i2 + len(p0) + len(p1)]),
        counts  = np.concatenate([c0, c1, c2]),
    )


class TestUVDataPack(unittest.TestCase):
    """Tests for UVData.pack (rasterized bitmap skyline UV packing)."""

    # ------------------------------------------------------------------ #
    #  Edge cases
    # ------------------------------------------------------------------ #
    def test_pack_empty(self):
        """Empty UVData (no faces) should be a no-op."""
        uv = UVData(
            points  = np.empty((0, 2)),
            indices = np.array([], dtype=int),
            counts  = np.array([], dtype=int),
        )
        uv.pack()
        self.assertEqual(uv.face_count, 0)
        self.assertEqual(uv.point_count, 0)

    def test_pack_single_island(self):
        """A single island should be normalized to [0, 1]^2."""
        pts, idx, cts = _make_quad_island((5.0, 5.0), size=2.0)
        uv = UVData(points=pts, indices=idx, counts=cts)
        uv.pack()

        self.assertTrue(np.all(uv.points >= -1e-7), f"points below 0:\n{uv.points}")
        self.assertTrue(
            np.all(uv.points <= 1.0 + 1e-7), f"points above 1:\n{uv.points}"
        )

    # ------------------------------------------------------------------ #
    #  Core functionality
    # ------------------------------------------------------------------ #
    def test_pack_two_islands_in_unit_square(self):
        """Two islands should be packed within [0, 1]^2."""
        uv = _make_two_island_uv()
        uv.pack()

        self.assertTrue(np.all(uv.points >= -1e-7), f"points below 0:\n{uv.points}")
        self.assertTrue(
            np.all(uv.points <= 1.0 + 1e-7), f"points above 1:\n{uv.points}"
        )

    def test_pack_three_islands_in_unit_square(self):
        """Three islands of varying size should be packed within [0, 1]^2."""
        uv = _make_three_island_uv()
        uv.pack()

        self.assertTrue(np.all(uv.points >= -1e-7))
        self.assertTrue(np.all(uv.points <= 1.0 + 1e-7))

    def test_pack_preserves_topology(self):
        """Packing must not change face_count or point_count."""
        uv        = _make_two_island_uv()
        fc_before = uv.face_count
        pc_before = uv.point_count

        uv.pack()

        self.assertEqual(uv.face_count, fc_before)
        self.assertEqual(uv.point_count, pc_before)

    def test_pack_preserves_island_shape(self):
        """Each island should keep its internal relative geometry (up to uniform scale + rotation)."""
        uv = _make_two_island_uv()

        # record per-island edge lengths before packing
        edges_before = []
        for shell in uv.shell_faces:
            verts = np.unique(uv.f2v[shell][uv.f2v[shell] >= 0])
            pts   = uv.points[verts]
            dists = np.linalg.norm(np.diff(np.sort(pts, axis=0), axis=0), axis=1)
            edges_before.append(dists)

        uv.pack()

        edges_after = []
        for shell in uv.shell_faces:
            verts = np.unique(uv.f2v[shell][uv.f2v[shell] >= 0])
            pts   = uv.points[verts]
            dists = np.linalg.norm(np.diff(np.sort(pts, axis=0), axis=0), axis=1)
            edges_after.append(dists)

        # ratios within each island should be preserved
        for before, after in zip(edges_before, edges_after):
            if before.size == 0:
                continue
            ratios_before = before / before.max()
            ratios_after  = after / after.max()
            np.testing.assert_allclose(
                ratios_before,
                ratios_after,
                atol    = 0.05,
                err_msg = "Island internal proportions changed after packing",
            )

    def test_pack_islands_do_not_overlap(self):
        """Packed island bounding boxes should not overlap."""
        uv = _make_three_island_uv()
        uv.pack(padding=2)

        bboxes = []
        for shell in uv.shell_faces:
            verts = np.unique(uv.f2v[shell][uv.f2v[shell] >= 0])
            pts   = uv.points[verts]
            bboxes.append((pts.min(axis=0), pts.max(axis=0)))

        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                mn_i, mx_i = bboxes[i]
                mn_j, mx_j = bboxes[j]
                overlap_x = mn_i[0] < mx_j[0] and mn_j[0] < mx_i[0]
                overlap_y = mn_i[1] < mx_j[1] and mn_j[1] < mx_i[1]
                self.assertFalse(
                    overlap_x and overlap_y,
                    f"Islands {i} and {j} bounding boxes overlap: "
                    f"{bboxes[i]} vs {bboxes[j]}",
                )

    # ------------------------------------------------------------------ #
    #  Parameters
    # ------------------------------------------------------------------ #
    def test_pack_rotation_disabled(self):
        """rotations=1 should disable rotation search and still pack correctly."""
        uv = _make_two_island_uv()
        uv.pack(rotations=1)

        self.assertTrue(np.all(uv.points >= -1e-7))
        self.assertTrue(np.all(uv.points <= 1.0 + 1e-7))

    def test_pack_high_resolution(self):
        """Higher resolution should still produce valid packing."""
        uv = _make_two_island_uv()
        uv.pack(resolution=2048)

        self.assertTrue(np.all(uv.points >= -1e-7))
        self.assertTrue(np.all(uv.points <= 1.0 + 1e-7))

    def test_pack_zero_padding(self):
        """padding=0 should still produce valid packing."""
        uv = _make_two_island_uv()
        uv.pack(padding=0)

        self.assertTrue(np.all(uv.points >= -1e-7))
        self.assertTrue(np.all(uv.points <= 1.0 + 1e-7))

    def test_pack_large_padding(self):
        """Large padding should still produce valid packing (islands fit in [0,1]^2)."""
        uv = _make_two_island_uv()
        uv.pack(padding=20)

        self.assertTrue(np.all(uv.points >= -1e-7))
        self.assertTrue(np.all(uv.points <= 1.0 + 1e-7))

    # ------------------------------------------------------------------ #
    #  Idempotency
    # ------------------------------------------------------------------ #
    def test_pack_idempotent(self):
        """Packing twice should produce results still within [0, 1]^2."""
        uv = _make_two_island_uv()
        uv.pack()
        uv.pack()

        self.assertTrue(np.all(uv.points >= -1e-7))
        self.assertTrue(np.all(uv.points <= 1.0 + 1e-7))