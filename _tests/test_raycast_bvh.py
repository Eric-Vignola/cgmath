"""Tests for the BVH-accelerated raycast path on MeshData."""

import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.utils._numba._bilinear import (
    _compute_face_aabbs,
    _intersect_mesh_parallel,
    _intersect_mesh_parallel_bvh,
)
from cgmath.geometry.utils._numba._bvh import build_bvh


def _make_subdivided_grid(n=20):
    """Returns a MeshData of an (n x n) flat grid of quads triangulated as
    2 tris per quad, lying in the XZ plane.

    n=20 -> 400 quads -> 800 triangles, large enough to exercise the BVH
    multi-leaf split paths but small enough to test fast.
    """
    xs = np.linspace(-1.0, 1.0, n + 1)
    zs = np.linspace(-1.0, 1.0, n + 1)
    points = np.array(
        [(x, 0.0, z) for z in zs for x in xs], dtype=float
    )
    indices, counts = [], []
    stride = n + 1
    for j in range(n):
        for i in range(n):
            v0 = j * stride + i
            v1 = v0 + 1
            v2 = v0 + stride + 1
            v3 = v0 + stride
            indices.extend([v0, v1, v2, v0, v2, v3])
            counts.extend([3, 3])
    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    )


def _make_random_rays(seed, n, mesh):
    rng = np.random.default_rng(seed)
    mn, mx = mesh.points.min(0), mesh.points.max(0)
    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn)) + 1.0
    o      = center + rng.normal(scale=extent, size=(n, 3))
    p      = center + rng.normal(scale=0.3 * extent, size=(n, 3))
    d      = p - o
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o.astype(np.float64), d.astype(np.float64)


class TestBVHRaycast(unittest.TestCase):
    def test_bvh_kernel_matches_brute_force(self):
        """The BVH-accelerated kernel must produce identical (t, uv, face)
        results to the brute-force linear-scan kernel for every ray."""
        mesh = _make_subdivided_grid(n=20)

        # Pad geometry to width-4 just like the public wrappers do.
        geom = np.asarray(mesh.geometry, dtype=np.int32)
        if geom.shape[1] < 4:
            pad                     = np.full((geom.shape[0], 4), -1, dtype=np.int32)
            pad[:, : geom.shape[1]] = geom
            geom                    = pad
        pts = np.ascontiguousarray(mesh.points, dtype=np.float64)

        aabb_min, aabb_max = _compute_face_aabbs(geom, pts)
        bvh = build_bvh(aabb_min, aabb_max, leaf_size=8)

        o, d = _make_random_rays(seed=42, n=2000, mesh=mesh)

        t_bf, uv_bf, fi_bf = _intersect_mesh_parallel(
            geom, pts, aabb_min, aabb_max, o, d, True
        )
        t_bvh, uv_bvh, fi_bvh = _intersect_mesh_parallel_bvh(
            geom,
            pts,
            bvh.node_aabb_min,
            bvh.node_aabb_max,
            bvh.node_left,
            bvh.node_right,
            bvh.node_first,
            bvh.node_count,
            bvh.face_perm,
            o,
            d,
            True,
        )

        # Hit/miss masks must match exactly.
        np.testing.assert_array_equal(np.isnan(t_bf), np.isnan(t_bvh))
        # Face indices must match exactly.
        np.testing.assert_array_equal(fi_bf, fi_bvh)
        # t and uv values must agree to within float tolerance for hits.
        both = ~np.isnan(t_bf) & ~np.isnan(t_bvh)
        if both.any():
            np.testing.assert_allclose(t_bf[both], t_bvh[both], atol=1e-9)
            np.testing.assert_allclose(uv_bf[both], uv_bvh[both], atol=1e-9)

    def test_meshdata_bvh_lazy_caching(self):
        """The bvh_bilinear property must build on first access and reuse
        on subsequent accesses (same object)."""
        mesh = _make_subdivided_grid(n=20)
        self.assertIsNone(mesh._bvh_bilinear)

        bvh1 = mesh.bvh_bilinear
        self.assertIsNotNone(bvh1)
        self.assertIsNotNone(mesh._bvh_bilinear)

        bvh2 = mesh.bvh_bilinear
        self.assertIs(bvh1, bvh2, "bvh_bilinear must reuse the cached object")

    def test_meshdata_bvh_skipped_for_small_mesh(self):
        """Tiny meshes (<=64 faces) should fall back to the brute-force
        path - bvh_bilinear returns None, raycast still works."""
        # A unit cube has 6 quads = 12 triangles, well under the threshold.
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
        idx = []
        cnt = []
        for q in [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 4, 7, 3),
        ]:
            idx.extend([q[0], q[1], q[2], q[0], q[2], q[3]])
            cnt.extend([3, 3])
        mesh = MeshData(
            points  = points,
            indices = np.asarray(idx, dtype=np.int64),
            counts  = np.asarray(cnt, dtype=np.int64),
        )
        self.assertIsNone(mesh.bvh_bilinear)
        # raycast must still work and return valid hits.
        o  = np.array([[0.5, 0.5, -2.0]])
        d  = np.array([[0.0, 0.0, 1.0]])
        rd = mesh.raycast(o, d)
        self.assertTrue(bool(rd.hit[0]))

    def test_meshdata_bvh_invalidated_on_topology_change(self):
        """reset_cached_data() (called by triangulate/quadrangulate/etc.)
        must clear the cached BVHs since they reference stale topology."""
        mesh = _make_subdivided_grid(n=20)
        _    = mesh.bvh_bilinear
        self.assertIsNotNone(mesh._bvh_bilinear)
        mesh.reset_cached_data()
        self.assertIsNone(mesh._bvh_bilinear)
        self.assertIsNone(mesh._bvh_bezier)

    def test_meshdata_raycast_matches_with_and_without_bvh(self):
        """Calling MeshData.raycast() (which uses cached BVH for big
        meshes) must agree with the no-BVH path for the same inputs."""
        mesh = _make_subdivided_grid(n=20)
        o, d = _make_random_rays(seed=7, n=500, mesh=mesh)

        # Use cached BVH path (default - mesh has 800 tris > 64 threshold).
        rd_bvh = mesh.raycast(o, d, method="bilinear")

        # Force the brute-force path by clearing the cache and bypassing
        # the property: monkey-patch a sentinel that tells the property to
        # return None, simulating the "small mesh" fallback.
        mesh._bvh_bilinear = None
        original_threshold = MeshData._BVH_MIN_FACES
        try:
            MeshData._BVH_MIN_FACES = 100_000  # force fallback
            rd_bf                   = mesh.raycast(o, d, method="bilinear")
        finally:
            MeshData._BVH_MIN_FACES = original_threshold

        np.testing.assert_array_equal(rd_bvh.hit, rd_bf.hit)
        np.testing.assert_array_equal(rd_bvh.indices, rd_bf.indices)
        both = ~np.isnan(rd_bvh.distances) & ~np.isnan(rd_bf.distances)
        if both.any():
            np.testing.assert_allclose(
                rd_bvh.distances[both], rd_bf.distances[both], atol=1e-9
            )