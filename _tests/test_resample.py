import os
import pickle
import unittest
from unittest import mock

import numpy as np
from cgmath.geometry import MeshData, MorphList, SkinData, UVList
from cgmath.geometry._saddle_surface import sample as sample_patches
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import cv2, SampleMethod, skimage, UVData
from cgmath.geometry.resample import MeshDataResampler, ResampleMode
from cgmath.geometry.utils.main import compute_samples

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestHierarchy(unittest.TestCase):
    def setUp(self):
        super().setUp()

        root, fname = os.path.split(__file__)
        fname, ext = os.path.splitext(fname)

        self.src_mesh_data = os.path.join(root, "test_assets", fname + ".src_mesh_data.npz")
        self.dst_mesh_data = os.path.join(root, "test_assets", fname + ".dst_mesh_data.npz")
        self.src_uv_data   = os.path.join(root, "test_assets", fname + ".src_uv_data.npz")
        self.dst_uv_data   = os.path.join(root, "test_assets", fname + ".dst_uv_data.npz")
        self.src_bs_data   = os.path.join(root, "test_assets", fname + ".src_bs_data.npz")
        self.dst_bs_data   = os.path.join(root, "test_assets", fname + ".dst_bs_data.npz")
        self.src_skin_data = os.path.join(root, "test_assets", fname + ".src_skin_data.npz")
        self.dst_skin_data = os.path.join(root, "test_assets", fname + ".dst_skin_data.npz")
        self.overlap_data  = os.path.join(root, "test_assets", fname + ".overlap.pkl")

        self.src_mesh_data = MeshData.load(self.src_mesh_data)
        self.dst_mesh_data = MeshData.load(self.dst_mesh_data)
        self.src_uv_data   = UVList.load(self.src_uv_data)
        self.dst_uv_data   = UVList.load(self.dst_uv_data)
        self.src_bs_data   = MorphList.load(self.src_bs_data)
        self.dst_bs_data   = MorphList.load(self.dst_bs_data)
        self.src_skin_data = SkinData.load(self.src_skin_data)
        self.dst_skin_data = SkinData.load(self.dst_skin_data)

        with open(self.overlap_data, "rb") as io:
            self.overlap_data = pickle.load(io)


    def test_resample(self):

        # construct a resampler
        resampler = MeshDataResampler(
            self.src_mesh_data, self.dst_mesh_data, src_uv=self.src_uv_data[0], dst_uv=self.dst_uv_data[0]
        )


        # resample the blendshape data
        new_bs_data = MorphList()
        for shape in self.src_bs_data:
            data = resampler.resample_morph_target(shape, mode=ResampleMode.UV)
            new_bs_data.append(data)
        self.assertTrue(new_bs_data == self.dst_bs_data)

        # resample the skin data
        new_skin_data = resampler.resample_skin_weights(self.src_skin_data, mode=ResampleMode.UV)
        new_skin_data.set_max_influences(4, sorting_method="stable") # sorting_method="stable" is slower but backwards compatible
        new_skin_data.remove_unused(tolerance=0)
        self.assertTrue(new_skin_data == self.dst_skin_data)



    @unittest.skipIf(cv2 is None and skimage is None, "needs cv2 or scikit-image")
    def test_overlap(self):
        indices, result1, result2 = self.overlap_data

        overlap = self.dst_uv_data[0].get_overlap_faces(self.src_uv_data[0],
                                                        indices    = indices,
                                                        resolution = 2048,
                                                        tolerance=0.0)
        self.assertTrue(allclose(overlap, result1))

        overlap = self.dst_uv_data[0].get_overlap_faces(self.src_uv_data[0],
                                                        indices    = indices,
                                                        resolution = 2048,
                                                        tolerance=0.2)
        self.assertTrue(allclose(overlap, result2))

    def test_resample_mesh_spatial(self):
        """Test resample_mesh() method with spatial mode"""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        # Test resampling without providing mesh_data (uses src_mesh points)
        resampled_mesh = resampler.resample_mesh(mode=ResampleMode.SPATIAL)

        # Should return a mesh with dst_mesh topology
        self.assertEqual(resampled_mesh.point_count, self.dst_mesh_data.point_count)
        self.assertEqual(resampled_mesh.face_count, self.dst_mesh_data.face_count)

        # Should have resampled points (not identical to dst_mesh)
        self.assertFalse(np.allclose(resampled_mesh.points, self.dst_mesh_data.points))

        # Should be valid mesh
        self.assertTrue(resampled_mesh.valid)

    def test_resample_mesh_with_mesh_data(self):
        """Test resample_mesh() method with explicit mesh_data parameter"""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        # Create a modified version of src_mesh with offset points
        modified_mesh        = self.src_mesh_data.copy()
        modified_mesh.points = modified_mesh.points + np.array([0.1, 0.1, 0.1])

        # Test resampling with explicit mesh_data parameter
        resampled_mesh = resampler.resample_mesh(
            mesh_data=modified_mesh, mode=ResampleMode.SPATIAL
        )

        # Should return a mesh with dst_mesh topology
        self.assertEqual(resampled_mesh.point_count, self.dst_mesh_data.point_count)
        self.assertEqual(resampled_mesh.face_count, self.dst_mesh_data.face_count)

        # Should have resampled the modified mesh points
        # (should not be identical to resampling the original src_mesh)
        resampled_original = resampler.resample_mesh(mode=ResampleMode.SPATIAL)
        self.assertFalse(np.allclose(resampled_mesh.points, resampled_original.points))

        # Should be valid mesh
        self.assertTrue(resampled_mesh.valid)

    def test_resample_mesh_uv(self):
        """Test resample_mesh() method with UV mode"""
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )

        # Test resampling in UV space
        resampled_mesh = resampler.resample_mesh(mode=ResampleMode.UV)

        # Should return a mesh with dst_mesh topology
        self.assertEqual(resampled_mesh.point_count, self.dst_mesh_data.point_count)
        self.assertEqual(resampled_mesh.face_count, self.dst_mesh_data.face_count)

        # Should have resampled points
        self.assertFalse(np.allclose(resampled_mesh.points, self.dst_mesh_data.points))

        # Should be valid mesh
        self.assertTrue(resampled_mesh.valid)

    def test_resample_map_data(self):
        """Test resample_map() method with MapData"""
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )

        # Create a MapData object from skin data
        # MapData requires a single weight value
        skin_data            = self.src_skin_data.copy()
        skin_data.influences = [skin_data.influences[0]]
        rng                  = np.random.default_rng(seed=42)
        skin_data.weights    = rng.random((skin_data.weights.shape[0],1))
        map_data = MapData.from_skin_data(
            skin_data,
            mesh_data      = self.src_mesh_data,
            name           = "test_map",
            component_type = "v",
        )

        # Resample the map data
        resampled_map = resampler.resample_map(map_data, mode=ResampleMode.UV)

        # Should return MapData instance
        self.assertIsInstance(resampled_map, MapData)

        # Should have the same name
        self.assertEqual(resampled_map.name, map_data.name)

        # Should have the same component type
        self.assertEqual(resampled_map.component_type, map_data.component_type)

        # Should have dst_mesh face count
        self.assertEqual(len(resampled_map.indices), self.dst_mesh_data.point_count)

    def test_resample_geom_subset_data(self):
        """Test resample_map() method with GeomSubsetData"""
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )

        # Create a GeomSubsetData object from skin data
        # GeomSubsetData requires a single weight value
        skin_data            = self.src_skin_data.copy()
        skin_data.influences = [skin_data.influences[0]]
        rng                  = np.random.default_rng(seed=42)
        skin_data.weights    = rng.random((skin_data.weights.shape[0],1))
        geom_subset = GeomSubsetData.from_skin_data(
            skin_data,
            mesh_data      = self.src_mesh_data,
            name           = "test_subset",
            component_type = "f",
        )

        # Resample the geom subset data
        resampled_subset = resampler.resample_map(geom_subset, mode=ResampleMode.UV)

        # Should return GeomSubsetData instance
        self.assertIsInstance(resampled_subset, GeomSubsetData)

        # Should have the same name
        self.assertEqual(resampled_subset.name, geom_subset.name)

        # Should have the same component type
        self.assertEqual(resampled_subset.component_type, geom_subset.component_type)

        # Should have dst_mesh face count
        self.assertEqual(len(resampled_subset.indices), self.dst_mesh_data.face_count)

    def test_resampler_properties(self):
        """Test MeshDataResampler properties"""
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )

        # Test src_mesh property
        self.assertIs(resampler.src_mesh, self.src_mesh_data)
        self.assertEqual(resampler.src_mesh.point_count, self.src_mesh_data.point_count)

        # Test dst_mesh property
        self.assertIs(resampler.dst_mesh, self.dst_mesh_data)
        self.assertEqual(resampler.dst_mesh.point_count, self.dst_mesh_data.point_count)

        # Test src_uv property
        self.assertIs(resampler.src_uv, self.src_uv_data[0])

        # Test dst_uv property
        self.assertIs(resampler.dst_uv, self.dst_uv_data[0])

        # Test spatial_sample_data property (lazy initialization)
        spatial_sample = resampler.spatial_sample_data
        self.assertIsNotNone(spatial_sample)
        # Should return same instance on subsequent calls (cached)
        self.assertIs(resampler.spatial_sample_data, spatial_sample)

        # Test uv_sample_data property (lazy initialization)
        uv_sample = resampler.uv_sample_data
        self.assertIsNotNone(uv_sample)
        # Should return same instance on subsequent calls (cached)
        self.assertIs(resampler.uv_sample_data, uv_sample)

    def test_resampler_without_uv(self):
        """Test MeshDataResampler properties when UV is not provided"""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        # src_uv and dst_uv should be None
        self.assertIsNone(resampler.src_uv)
        self.assertIsNone(resampler.dst_uv)

        # Accessing uv_sample_data without UV should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            _ = resampler.uv_sample_data
        self.assertIn("UV data not supplied", str(context.exception))

    def test_resample_mesh_maintain_offset_with_orient(self):
        """orient_offset=True recomputes normals from mesh_data for the offset direction."""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        resampled = resampler.resample_mesh(
            maintain_offset=True, orient_offset=True,
        )
        self.assertEqual(resampled.point_count, self.dst_mesh_data.point_count)

        # With maintain_offset the result should differ from without it
        resampled_no_offset = resampler.resample_mesh(maintain_offset=False)
        self.assertFalse(np.allclose(resampled.points, resampled_no_offset.points))

    def test_resample_mesh_maintain_offset_without_orient(self):
        """orient_offset=False uses sample_data.normals (original behaviour)."""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        resampled = resampler.resample_mesh(
            maintain_offset=True, orient_offset=False,
        )
        self.assertEqual(resampled.point_count, self.dst_mesh_data.point_count)

        resampled_no_offset = resampler.resample_mesh(maintain_offset=False)
        self.assertFalse(np.allclose(resampled.points, resampled_no_offset.points))

    def test_orient_offset_differs_from_no_orient_on_distorted_mesh(self):
        """orient_offset=True and False should produce different results on a distorted mesh."""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        # Create a distorted version of the source mesh
        distorted        = self.src_mesh_data.copy()
        rng              = np.random.default_rng(seed=123)
        distorted.points = distorted.points + rng.normal(scale=0.05, size=distorted.points.shape)

        resampled_orient = resampler.resample_mesh(
            mesh_data=distorted, maintain_offset=True, orient_offset=True,
        )
        resampled_no_orient = resampler.resample_mesh(
            mesh_data=distorted, maintain_offset=True, orient_offset=False,
        )

        # Recomputed normals from the distorted mesh should differ from the
        # stored source normals, so the two results must not match.
        self.assertFalse(np.allclose(resampled_orient.points, resampled_no_orient.points))

    def test_orient_offset_no_effect_without_maintain_offset(self):
        """orient_offset should have no effect when maintain_offset=False."""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        resampled_a = resampler.resample_mesh(
            maintain_offset=False, orient_offset=True,
        )
        resampled_b = resampler.resample_mesh(
            maintain_offset=False, orient_offset=False,
        )
        self.assertTrue(np.allclose(resampled_a.points, resampled_b.points))

    def test_orient_offset_without_mesh_data(self):
        """orient_offset should work when mesh_data is None (falls back to src_mesh)."""
        resampler = MeshDataResampler(self.src_mesh_data, self.dst_mesh_data)

        resampled = resampler.resample_mesh(
            mesh_data=None, maintain_offset=True, orient_offset=True,
        )
        self.assertEqual(resampled.point_count, self.dst_mesh_data.point_count)
        self.assertTrue(resampled.valid)

    def test_maintain_offset_is_rejected_in_uv_mode(self):
        """UV sample data has 2-D projections, so the offset math cannot run.

        Without the guard this dies inside numpy as
        ``operands could not be broadcast together with shapes (n,3) (n,2)``,
        which says nothing about which argument was wrong.
        """
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )
        with self.assertRaises(ValueError) as context:
            resampler.resample_mesh(mode=ResampleMode.UV, maintain_offset=True)
        self.assertIn("maintain_offset", str(context.exception))
        self.assertIn("UV mode", str(context.exception))

    def test_uv_mode_without_maintain_offset_is_unaffected(self):
        """The guard must not fire on the ordinary UV path."""
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )
        resampled = resampler.resample_mesh(mode=ResampleMode.UV)
        self.assertEqual(resampled.point_count, self.dst_mesh_data.point_count)

    def test_maintain_offset_still_works_in_spatial_mode(self):
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )
        resampled = resampler.resample_mesh(
            mode=ResampleMode.SPATIAL, maintain_offset=True
        )
        self.assertEqual(resampled.point_count, self.dst_mesh_data.point_count)

    def test_bezier_is_honoured_in_uv_mode(self):
        """BEZIER is NOT a no-op in UV mode, despite appearances on a regular grid.

        A destination whose vertices coincide with source vertices evaluates the
        patch at its control points, where the PN-quad correction vanishes and
        BEZIER and BILINEAR agree exactly. That is a property of the fixture,
        not of the code, and it previously led to UV+BEZIER being written off as
        a silent no-op. This pins the real behaviour.
        """
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
            dst_uv = self.dst_uv_data[0],
        )
        bilinear = resampler.resample_mesh(
            mode=ResampleMode.UV, method=SampleMethod.BILINEAR
        )
        bezier = resampler.resample_mesh(
            mode=ResampleMode.UV, method=SampleMethod.BEZIER
        )
        self.assertEqual(bezier.point_count, bilinear.point_count)
        self.assertFalse(np.allclose(bilinear.points, bezier.points))

    def test_resample_uv_on_fixture_meshes(self):
        """resample_uv() runs on the production fixture and keeps dst topology."""
        resampler = MeshDataResampler(
            self.src_mesh_data,
            self.dst_mesh_data,
            src_uv = self.src_uv_data[0],
        )
        for raycast in (False, True):
            with self.subTest(raycast=raycast):
                uv = resampler.resample_uv(raycast=raycast)
                self.assertTrue(np.array_equal(uv.counts, self.dst_mesh_data.counts))
                self.assertEqual(uv.indices.size, self.dst_mesh_data.indices.size)
                self.assertTrue(np.all(np.isfinite(uv.points)))
                self.assertEqual(uv.name, self.src_uv_data[0].name)

    def test_resample_uv_of_the_fixture_onto_itself(self):
        """The fixture onto itself gives back every UV but those on a cut inside a shell."""
        mesh, src_uv = self.src_mesh_data, self.src_uv_data[0]
        shell = np.zeros(mesh.face_count, dtype=np.int64)
        for k, faces in enumerate(src_uv.shell_faces):
            shell[faces] = k

        # a vertex with two UVs in one shell sits on a cut, which never splits
        key   = mesh.indices * len(src_uv.shell_faces) + np.repeat(shell, mesh.counts)
        pairs = np.unique(np.stack([key, src_uv.indices], axis=1), axis=0)
        keys, n = np.unique(pairs[:, 0], return_counts=True)
        kept = ~np.isin(key, keys[n > 1])
        want = src_uv.points[src_uv.indices]

        resampler = MeshDataResampler(mesh, mesh, src_uv=src_uv)
        for method in SampleMethod:
            for raycast in (False, True):
                with self.subTest(method=method, raycast=raycast):
                    uv    = resampler.resample_uv(method=method, raycast=raycast)
                    error = np.abs(uv.points[uv.indices] - want).max(axis=1)
                    self.assertLess(error[kept].max(), 1e-9)


# --- synthetic meshes for resample_uv: two UV shells with a known answer


def _grid(n, lift=0.0, low=0.0, high=1.0):
    """n x n quad grid over x, z in [low, high], curved in y: y = 0.2 sin(pi x)."""
    xs = np.linspace(low, high, n + 1)
    x, z = np.meshgrid(xs, xs, indexing="ij")
    x, z = x.ravel(), z.ravel()
    pts = np.stack([x, 0.2 * np.sin(np.pi * x) + lift, z], axis=1)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v     = (i * (n + 1) + j).ravel()
    quads = np.stack([v, v + n + 1, v + n + 2, v + 1], axis=1)
    return MeshData(indices=quads.ravel(), counts=np.full(n * n, 4), points=pts)


def _split_uvs(mesh):
    """Two UV shells split at x = 0.5: uv = (x, z) left of it, (x + 1, z) right."""
    right = np.repeat(mesh.points[mesh.geometry, 0].mean(axis=1) >= 0.5, 4)
    ids   = mesh.indices + right * mesh.point_count
    pts   = np.concatenate([mesh.points[:, [0, 2]], mesh.points[:, [0, 2]] + [1.0, 0.0]])
    used, inverse = np.unique(ids, return_inverse=True)
    return UVData(indices=inverse, counts=mesh.counts.copy(), points=pts[used])


def _left_shell_only(mesh):
    """_split_uvs with the faces of the right shell left without UVs."""
    full   = _split_uvs(mesh)
    left   = mesh.points[mesh.geometry, 0].mean(axis=1) < 0.5
    counts = np.where(left, full.counts, 0).astype(np.int32)
    keep   = np.repeat(left, full.counts)
    return UVData(indices=full.indices[keep], counts=counts, points=full.points)


def _with_hexagon(mesh, face):
    """An n x n _grid with quad ``face`` and the next one in z merged into a hexagon."""
    n     = int(round(np.sqrt(mesh.face_count)))
    faces = [list(quad) for quad in mesh.indices.reshape(-1, 4)]
    v     = faces[face][0]
    # the two quads share the edge (v + 1, v + n + 2): drop it
    faces[face : face + 2] = [[v, v + n + 1, v + n + 2, v + n + 3, v + 2, v + 1]]
    return MeshData(
        indices = np.concatenate(faces),
        counts  = np.array([len(f) for f in faces]),
        points  = mesh.points.copy(),
    )


def _centroids(mesh):
    """Each face's centroid over its own corners, triangles included."""
    valid   = mesh.geometry >= 0
    corners = np.where(valid, mesh.geometry, 0)
    return (mesh.points[corners] * valid[..., None]).sum(axis=1) / mesh.counts[:, None]


def _block_uvs(mesh, cuts):
    """UV shells cut at every x and z in ``cuts``: uv = (x, z), moved 1.5 per shell."""
    centroids = _centroids(mesh)
    shell     = np.searchsorted(cuts, centroids[:, 0]) + (len(cuts) + 1) * np.searchsorted(cuts, centroids[:, 2])
    ids       = mesh.indices + np.repeat(shell, mesh.counts) * mesh.point_count
    pts       = np.concatenate([mesh.points[:, [0, 2]] + [1.5 * k, 0.0] for k in range((len(cuts) + 1) ** 2)])
    used, inverse = np.unique(ids, return_inverse=True)
    return UVData(indices=inverse, counts=mesh.counts.copy(), points=pts[used])


def _face_shells(mesh):
    """A UV shell per face: uv = (x, z), face k's moved 2 k along u."""
    ids = mesh.indices + np.repeat(np.arange(mesh.face_count), mesh.counts) * mesh.point_count
    used, inverse = np.unique(ids, return_inverse=True)
    pts = mesh.points[used % mesh.point_count][:, [0, 2]] + (used // mesh.point_count)[:, None] * [2.0, 0.0]
    return UVData(indices=inverse.reshape(-1), counts=mesh.counts.copy(), points=pts)


def _mixed(n, **kwargs):
    """_grid(n, ...) with every other quad split into two triangles."""
    quads = _grid(n, **kwargs)
    faces = []
    for k, quad in enumerate(quads.indices.reshape(-1, 4)):
        if k % 2:
            faces += [quad[[0, 1, 2]], quad[[0, 2, 3]]]
        else:
            faces.append(quad)
    return MeshData(
        indices = np.concatenate(faces),
        counts  = np.array([len(f) for f in faces]),
        points  = quads.points.copy(),
    )


def _box(n, thickness):
    """A 2 x 2 x thickness box, n x n quads per side, one UV shell per side."""
    sides = [(0, 1, 2, -1.0), (0, 1, 2, 1.0), (0, 2, 1, -1.0), (0, 2, 1, 1.0), (1, 2, 0, -1.0), (1, 2, 0, 1.0)]
    points, quads, uvs = [], [], []
    for side, (a, b, c, sign) in enumerate(sides):
        g = np.linspace(-1.0, 1.0, n + 1)
        u, v = np.meshgrid(g, g, indexing="ij")
        p = np.zeros((n + 1, n + 1, 3))
        p[..., a], p[..., b], p[..., c] = u, v, sign
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        w = (i * (n + 1) + j).ravel()
        q = np.stack([w, w + n + 1, w + n + 2, w + 1], axis=1)
        quads.append((q[:, ::-1] if sign < 0 else q) + sum(len(x) for x in points))
        points.append(p.reshape(-1, 3) * [1.0, 1.0, thickness / 2])
        uvs.append(np.stack([(u.ravel() + 1) / 2 + 1.2 * side, (v.ravel() + 1) / 2], axis=1))
    points = np.concatenate(points)
    quads  = np.concatenate(quads)
    welded, inverse = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    mesh = MeshData(indices=inverse.reshape(-1)[quads].ravel(), counts=np.full(len(quads), 4), points=welded)
    uv   = UVData(indices=quads.ravel(), counts=np.full(len(quads), 4), points=np.concatenate(uvs))
    return mesh, uv


def _expected_uvs(dst):
    """Per face-vertex UVs: each corner in the shell under its face's centroid."""
    side = np.repeat(_centroids(dst)[:, 0] >= 0.5, dst.counts)
    return dst.points[dst.indices][:, [0, 2]] + side[:, None] * [1.0, 0.0]


def _torn_faces(uv):
    """Faces whose corners span the 1.0 gap between the two shells."""
    valid = uv.geometry >= 0
    u     = uv.points[uv.geometry][..., 0]
    span  = np.where(valid, u, -np.inf).max(axis=1) - np.where(valid, u, np.inf).min(axis=1)
    return int(np.sum(span > 0.6))


class TestResampleUV(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.src    = _grid(10)
        self.src_uv = _split_uvs(self.src)

    def resample_uv(self, dst, **kwargs):
        return MeshDataResampler(self.src, dst, src_uv=self.src_uv).resample_uv(**kwargs)

    def test_seam_on_destination_edges(self):
        """The source seam runs along dst edges: split there, nothing stretched."""
        dst = _grid(8)
        for raycast in (False, True):
            for method in SampleMethod:
                with self.subTest(raycast=raycast, method=method):
                    uv = self.resample_uv(dst, raycast=raycast, method=method)
                    self.assertEqual(_torn_faces(uv), 0)
                    self.assertEqual(uv.point_count, 81 + 9)  # one seam column doubled
                    error = np.abs(uv.points[uv.indices] - _expected_uvs(dst)).max()
                    self.assertLess(error, 1e-3)

    def test_seam_through_destination_faces(self):
        """The seam cuts dst faces: their far corners clamp, none tears."""
        dst = _grid(7)
        for raycast in (False, True):
            with self.subTest(raycast=raycast):
                uv = self.resample_uv(dst, raycast=raycast)
                self.assertEqual(_torn_faces(uv), 0)

    def test_one_uv_per_vertex_would_tear(self):
        """Guards the fixture: a seamless transfer tears it, so the tests above bite."""
        dst      = _grid(8)
        samples  = self.src.sample(dst)
        uvs      = compute_samples(self.src_uv.points, samples.weights, self.src_uv.geometry[samples.indices])
        seamless = UVData(indices=dst.indices.copy(), counts=dst.counts.copy(), points=uvs)
        self.assertGreater(_torn_faces(seamless), 0)

    def test_destination_with_triangles(self):
        """Triangles and quads: each face reads the shell under the centroid of its own corners."""
        dst = _mixed(8)  # 64 triangles, 32 quads
        for raycast in (False, True):
            with self.subTest(raycast=raycast):
                uv = self.resample_uv(dst, raycast=raycast)
                self.assertTrue(np.array_equal(uv.counts, dst.counts))
                self.assertEqual(uv.point_count, 81 + 9)
                self.assertEqual(_torn_faces(uv), 0)
                self.assertLess(np.abs(uv.points[uv.indices] - _expected_uvs(dst)).max(), 1e-3)

    def test_raycast_follows_stored_normals(self):
        """Straight-down stored normals hit the vertices under a lifted copy."""
        dst         = _grid(10, lift=0.1)
        dst.normals = np.tile([0.0, -1.0, 0.0], (dst.point_count, 1))
        stored      = dst.normals.copy()

        uv    = self.resample_uv(dst, raycast=True, forward_only=True)
        error = np.abs(uv.points[uv.indices] - _expected_uvs(dst)).max()
        self.assertLess(error, 1e-6)
        self.assertTrue(np.array_equal(dst.normals, stored))

        computed       = self.resample_uv(dst, raycast=True, use_mesh_normals=False)
        computed_error = np.abs(computed.points[computed.indices] - _expected_uvs(dst)).max()
        self.assertGreater(computed_error, 10 * error + 1e-3)

    def test_forward_only_false_hits_behind(self):
        """Stored normals pointing away from src: only the backward cast hits."""
        dst         = _grid(10, lift=0.1)
        dst.normals = np.tile([0.0, 1.0, 0.0], (dst.point_count, 1))
        closest     = self.resample_uv(dst)

        both = self.resample_uv(dst, raycast=True)
        self.assertLess(np.abs(both.points[both.indices] - _expected_uvs(dst)).max(), 1e-6)
        ahead = self.resample_uv(dst, raycast=True, forward_only=True)  # every ray misses
        self.assertTrue(np.allclose(ahead.points, closest.points))
        self.assertGreater(np.abs(ahead.points - both.points).max(), 1e-2)

    def test_twosided_false_rejects_back_faces(self):
        """Straight-down rays meet src's back faces: twosided=False misses them all."""
        dst         = _grid(10, lift=0.1)
        dst.normals = np.tile([0.0, -1.0, 0.0], (dst.point_count, 1))
        closest     = self.resample_uv(dst)

        front = self.resample_uv(dst, raycast=True, forward_only=True)
        back  = self.resample_uv(dst, raycast=True, forward_only=True, twosided=False)
        self.assertTrue(np.allclose(back.points, closest.points))
        self.assertGreater(np.abs(front.points - back.points).max(), 1e-2)

    def test_twosided_false_reaches_the_centroid_rays(self):
        """Tilted rays that meet only back faces: twosided=False misses them all, centroids too."""
        dst         = _grid(8, lift=0.1)
        dst.normals = np.tile(np.array([-1.0, -1.0, 0.0]) / np.sqrt(2.0), (dst.point_count, 1))
        closest     = self.resample_uv(dst)
        back        = self.resample_uv(dst, raycast=True, twosided=False)
        self.assertTrue(np.array_equal(back.indices, closest.indices))
        self.assertTrue(np.allclose(back.points, closest.points, rtol=0, atol=1e-9))

    def test_reversed_normals_build_the_same_uv_set(self):
        """forward_only=False: n and -n are one line, so one UV set, the hits', seam-cut faces too."""
        dst  = _grid(7, lift=-0.08, low=0.02, high=0.96)  # under src, the seam through column 3
        tilt = np.array([0.3, 1.0, 0.4]) / np.linalg.norm([0.3, 1.0, 0.4])

        # each corner reads its vertex's hit, clamped to the shell its face's centroid ray hits
        hit      = self.src.raycast(dst.points, tilt).projections[dst.indices]
        centroid = self.src.raycast(dst.points[dst.geometry].mean(axis=1), tilt).projections
        right    = np.repeat(centroid[:, 0] >= 0.5, 4)
        u        = np.clip(hit[:, 0], 0.5 * right, 0.5 + 0.5 * right) + right
        expected = np.stack([u, hit[:, 2]], axis=1)
        self.assertEqual(int(np.sum((hit[:, 0] >= 0.5) != right)), 14)  # corners past the seam

        for method, tolerance in ((SampleMethod.BILINEAR, 1e-9), (SampleMethod.BEZIER, 2e-3)):
            with self.subTest(method=method):
                built = []
                for sign in (1.0, -1.0):  # rays hit forward, then backward
                    dst.normals = np.tile(sign * tilt, (dst.point_count, 1))
                    built.append(self.resample_uv(dst, raycast=True, method=method))
                    error = np.abs(built[-1].points[built[-1].indices] - expected).max()
                    self.assertLess(error, tolerance)
                self.assertTrue(np.array_equal(built[0].indices, built[1].indices))
                self.assertTrue(np.allclose(built[0].points, built[1].points, rtol=0, atol=1e-9))

    def test_faces_pick_their_shell_along_the_centroid_ray(self):
        """Tilted rays: a face takes the shell its centroid's ray hits, not the closest one's."""
        dst         = _grid(8, lift=0.1)
        tilt        = np.array([-1.0, -1.0, 0.0]) / np.sqrt(2.0)
        dst.normals = np.tile(tilt, (dst.point_count, 1))
        uv          = self.resample_uv(dst, raycast=True, forward_only=True)

        centroids   = dst.points[dst.geometry].mean(axis=1)
        hit         = self.src.raycast(centroids, tilt).projections[:, 0] >= 0.5
        closest     = self.src.sample(centroids).projections[:, 0] >= 0.5
        self.assertEqual(int(np.sum(hit != closest)), 8)  # column 4 moves left

        u = uv.points[uv.geometry][..., 0]
        self.assertTrue(np.all(np.where(hit[:, None], u >= 1.5 - 1e-6, u <= 0.5 + 1e-6)))

    def test_a_zero_stored_normal_leaves_its_faces_to_their_other_corners(self):
        """Tilted rays, a few zero normals on the seam: their faces still follow the others' rays."""
        dst               = _grid(8, lift=0.1)
        tilt              = np.array([-1.0, -1.0, 0.0]) / np.sqrt(2.0)
        dst.normals       = np.tile(tilt, (dst.point_count, 1))
        zero              = np.isclose(dst.points[:, 0], 0.5) & (np.arange(dst.point_count) % 2 == 0)
        dst.normals[zero] = 0.0
        uv                = self.resample_uv(dst, raycast=True, forward_only=True)

        hit = self.src.raycast(dst.points[dst.geometry].mean(axis=1), tilt).projections[:, 0] >= 0.5
        u   = uv.points[uv.geometry][..., 0]
        self.assertEqual(int(zero.sum()), 5)
        self.assertTrue(np.all(np.where(hit[:, None], u >= 1.5 - 1e-6, u <= 0.5 + 1e-6)))

    def test_triangle_centroid_ray_ignores_padding(self):
        """A triangle's centroid ray sums its own 3 corners' normals, not vertex 0's."""
        dst            = _mixed(8, lift=0.3)
        dst.normals    = np.tile([0.0, -1.0, 0.0], (dst.point_count, 1))
        dst.normals[0] = [-1.0, 0.0, 0.0]  # the padding of a triangle's 4th corner reads vertex 0
        uv             = self.resample_uv(dst, raycast=True, forward_only=True)
        error          = np.abs(uv.points[uv.indices] - _expected_uvs(dst))
        self.assertLess(error[dst.indices != 0].max(), 1e-6)

    def test_bezier_differs_from_bilinear(self):
        """method reaches the projection: BEZIER and BILINEAR give different UVs."""
        dst = _grid(8)
        for raycast in (False, True):
            with self.subTest(raycast=raycast):
                bilinear = self.resample_uv(dst, raycast=raycast, method=SampleMethod.BILINEAR)
                bezier   = self.resample_uv(dst, raycast=raycast, method=SampleMethod.BEZIER)
                self.assertGreater(np.abs(bilinear.points - bezier.points).max(), 1e-5)

    def test_method_given_as_a_string(self):
        """method='bezier' is SampleMethod.BEZIER, not a silent BILINEAR; an unknown name raises."""
        dst = _grid(8)
        for raycast in (False, True):
            with self.subTest(raycast=raycast):
                named    = self.resample_uv(dst, raycast=raycast, method="bezier")
                bezier   = self.resample_uv(dst, raycast=raycast, method=SampleMethod.BEZIER)
                bilinear = self.resample_uv(dst, raycast=raycast, method=SampleMethod.BILINEAR)
                named, bezier, bilinear = (uv.points[uv.indices] for uv in (named, bezier, bilinear))
                self.assertTrue(np.allclose(named, bezier, rtol=0, atol=1e-9))
                self.assertGreater(np.abs(named - bilinear).max(), 1e-5)
        with self.assertRaises(ValueError):
            self.resample_uv(dst, method="nearest")

    def test_a_mesh_onto_itself_gives_back_its_uvs(self):
        """A thin box onto itself, one shell per side: every UV comes back, BEZIER too."""
        src, src_uv = _box(4, 0.1)
        for method in SampleMethod:
            for raycast in (False, True):
                with self.subTest(method=method, raycast=raycast):
                    uv    = MeshDataResampler(src, src, src_uv=src_uv).resample_uv(method=method, raycast=raycast)
                    error = np.abs(uv.points[uv.indices] - src_uv.points[src_uv.indices]).max()
                    self.assertLess(error, 1e-9)

    def test_a_shell_per_face_reads_each_shell_on_its_own(self):
        """A shell per face onto itself: every UV back, each shell sampled on its own vertices, all strays read at once."""
        src    = _grid(12)
        src_uv = _face_shells(src)
        given  = []

        def sample(query, face_points, face_geometry, **kwargs):
            given.append((len(face_points), np.unique(face_geometry[face_geometry >= 0]).size))
            return sample_patches(query, face_points, face_geometry, **kwargs)

        resampler = MeshDataResampler(src, src, src_uv=src_uv)
        with mock.patch("cgmath.geometry.resample.sample_patches", side_effect=sample), mock.patch(
            "cgmath.geometry.resample.compute_samples", side_effect=compute_samples
        ) as reads:
            uv = resampler.resample_uv()
        self.assertLess(np.abs(uv.points[uv.indices] - src_uv.points[src_uv.indices]).max(), 1e-9)
        # the whole mesh twice, vertices and centroids, then a shell per pass
        self.assertGreater(len(given), 100)
        self.assertTrue(all(points == held for points, held in given))
        # the corners, then every stray corner in one go
        self.assertEqual(reads.call_count, 2)

    def test_bezier_surface_normals_computed_once(self):
        """BEZIER reads the source's vertex normals once a call, not once per shell."""
        src, src_uv = _box(4, 1.0)
        dst = src.copy()
        dst.points *= 0.9
        resampler = MeshDataResampler(src, dst, src_uv=src_uv)
        with mock.patch.object(
            MeshData, "get_vertex_normals", autospec=True, side_effect=MeshData.get_vertex_normals
        ) as counted:
            resampler.resample_uv(method=SampleMethod.BEZIER)
        on_src = [call for call in counted.call_args_list if call.args[0] is src]
        self.assertEqual(len(on_src), 1)

    def test_closest_bezier_matches_mesh_sample(self):
        """Closest point, BEZIER, one shell: the UVs are what MeshData.sample(BEZIER) reads."""
        src = _grid(10)
        src.points += np.random.default_rng(4).normal(scale=0.02, size=src.points.shape)
        planar   = UVData(indices=src.indices.copy(), counts=src.counts.copy(), points=src.points[:, [0, 2]])
        dst      = _grid(7, lift=0.03, low=0.05, high=0.95)
        uv       = MeshDataResampler(src, dst, src_uv=planar).resample_uv(method=SampleMethod.BEZIER)
        samples  = src.sample(dst, method=SampleMethod.BEZIER)
        expected = compute_samples(planar.points, samples.weights, planar.geometry[samples.indices])
        self.assertLess(np.abs(uv.points[uv.indices] - expected[dst.indices]).max(), 1e-9)

    def test_bezier_reads_a_stray_corner_off_its_shells_patches(self):
        """BEZIER: a corner past a seam reads its face's shell at the nearest point of its PN-quad patches."""
        src, src_uv = _box(4, 1.0)
        dst = src.copy()
        dst.points *= 0.9
        uv = MeshDataResampler(src, dst, src_uv=src_uv).resample_uv(method=SampleMethod.BEZIER)

        shell = np.zeros(src.face_count, dtype=np.int64)
        for k, faces in enumerate(src_uv.shell_faces):
            shell[faces] = k
        hits    = src.sample(dst, method=SampleMethod.BEZIER)
        wanted  = np.repeat(shell[src.sample(_centroids(dst)).indices], dst.counts)
        stray   = np.flatnonzero(shell[hits.indices[dst.indices]] != wanted)
        normals = src.get_vertex_normals(angle_weighted=True, area_weighted=False)
        self.assertGreater(stray.size, 0)
        for corner in stray:
            faces = src_uv.shell_faces[wanted[corner]]
            point = hits.projections[dst.indices[corner]][None]
            near  = sample_patches(point, src.points, src.geometry[faces], surface_normals=normals)
            want  = compute_samples(src_uv.points, near.weights, src_uv.geometry[faces][near.indices])
            self.assertLess(np.abs(uv.points[uv.indices[corner]] - want[0]).max(), 1e-9)

    def test_face_varying_normals(self):
        """Indexed and unindexed face-varying normals are summed per vertex."""
        dst      = _grid(10, lift=0.1)
        down     = np.array([0.0, -1.0, 0.0])
        expected = _expected_uvs(dst)

        layouts = {
            "unindexed": (np.tile(down, (dst.indices.size, 1)), None),
            "indexed":   (down[None, :].copy(), np.zeros(dst.indices.size, dtype=np.int32)),
        }
        for name, (normals, normal_indices) in layouts.items():
            with self.subTest(layout=name):
                dst.normals        = normals
                dst.normal_indices = normal_indices
                uv                 = self.resample_uv(dst, raycast=True, forward_only=True)
                self.assertLess(np.abs(uv.points[uv.indices] - expected).max(), 1e-6)

    def test_varying_face_varying_normals_are_summed(self):
        """Each vertex's ray is the sum of its corners' normals, through normal_indices."""
        dst       = _grid(6)
        resampler = MeshDataResampler(self.src, dst, src_uv=self.src_uv)
        rng       = np.random.default_rng(1)
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                if indexed:
                    normals            = rng.normal(size=(7, 3))
                    dst.normals        = normals
                    dst.normal_indices = rng.integers(0, 7, dst.indices.size).astype(np.int32)
                    corner_normals     = normals[dst.normal_indices]
                else:
                    corner_normals     = rng.normal(size=(dst.indices.size, 3))
                    dst.normals        = corner_normals
                    dst.normal_indices = None
                summed = np.zeros((dst.point_count, 3))
                np.add.at(summed, dst.indices, corner_normals)
                summed /= np.linalg.norm(summed, axis=1, keepdims=True)
                actual = resampler._get_ray_directions(True, True, True)
                self.assertTrue(np.allclose(actual, summed))

    def test_normal_indices_win_over_a_vertex_count_match(self):
        """As many indexed normals as vertices: normal_indices still maps them to the corners."""
        dst                = _grid(6)
        rng                = np.random.default_rng(2)
        dst.normals        = rng.normal(size=(dst.point_count, 3))
        dst.normal_indices = rng.integers(0, dst.point_count, dst.indices.size).astype(np.int32)
        summed             = np.zeros((dst.point_count, 3))
        np.add.at(summed, dst.indices, dst.normals[dst.normal_indices])
        summed /= np.linalg.norm(summed, axis=1, keepdims=True)
        resampler = MeshDataResampler(self.src, dst, src_uv=self.src_uv)
        self.assertTrue(np.allclose(resampler._get_ray_directions(True, True, True), summed))

    def test_raycast_without_stored_normals_leaves_mesh_alone(self):
        """No stored normals: computed ones are used and never written back."""
        dst = _grid(8, lift=0.05)
        self.assertIsNone(dst.normals)
        for angle_weighted in (True, False):
            for area_weighted in (True, False):
                with self.subTest(angle_weighted=angle_weighted, area_weighted=area_weighted):
                    uv = self.resample_uv(
                        dst,
                        raycast        = True,
                        angle_weighted = angle_weighted,
                        area_weighted  = area_weighted,
                    )
                    self.assertEqual(_torn_faces(uv), 0)
                    self.assertIsNone(dst.normals)
                    self.assertIsNone(dst.normal_indices)

    def test_ray_directions_match_get_vertex_normals(self):
        """Computed directions are get_vertex_normals() with the given weighting."""
        dst = _grid(6)
        dst.points += np.random.default_rng(3).normal(scale=0.02, size=dst.points.shape)
        resampler  = MeshDataResampler(self.src, dst, src_uv=self.src_uv)
        directions = []
        for angle_weighted in (True, False):
            for area_weighted in (True, False):
                expected = dst.get_vertex_normals(angle_weighted=angle_weighted, area_weighted=area_weighted)
                expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
                actual   = resampler._get_ray_directions(False, angle_weighted, area_weighted)
                self.assertTrue(np.allclose(actual, expected))
                directions.append(actual)
        for i in range(4):  # the jitter makes every weighting count
            for j in range(i):
                self.assertGreater(np.abs(directions[i] - directions[j]).max(), 1e-3)

    def test_computed_normals_default_to_angle_and_area_weighting(self):
        """No stored normals: the default rays are get_vertex_normals()' own defaults."""
        dst = _grid(8, lift=0.05)
        dst.points += np.random.default_rng(3).normal(scale=0.02, size=dst.points.shape)
        resampler = MeshDataResampler(self.src, dst, src_uv=self.src_uv)
        default   = resampler.resample_uv(raycast=True)
        default   = default.points[default.indices]
        for angle_weighted in (True, False):
            for area_weighted in (True, False):
                with self.subTest(angle_weighted=angle_weighted, area_weighted=area_weighted):
                    uv = resampler.resample_uv(
                        raycast        = True,
                        angle_weighted = angle_weighted,
                        area_weighted  = area_weighted,
                    )
                    error = np.abs(uv.points[uv.indices] - default).max()
                    if angle_weighted and area_weighted:
                        self.assertLess(error, 1e-12)
                    else:
                        self.assertGreater(error, 1e-4)

    def test_mismatched_stored_normals_raise(self):
        dst         = _grid(8)
        dst.normals = np.tile([0.0, 1.0, 0.0], (5, 1))
        with self.assertRaises(ValueError) as context:
            self.resample_uv(dst, raycast=True)
        self.assertIn("use_mesh_normals=False", str(context.exception))

    def test_missed_rays_fall_back_to_closest_point(self):
        """dst overhangs src: rays past the edge miss and clamp to the shell border."""
        dst         = _grid(12, lift=0.1, low=-0.3, high=1.3)
        dst.normals = np.tile([0.0, -1.0, 0.0], (dst.point_count, 1))
        uv          = self.resample_uv(dst, raycast=True, forward_only=True)
        u, v = uv.points[:, 0], uv.points[:, 1]
        eps = 1e-6

        self.assertTrue(np.all(np.isfinite(uv.points)))
        self.assertTrue(np.all((v >= -eps) & (v <= 1 + eps)))
        self.assertTrue(np.all(((u >= -eps) & (u <= 0.5 + eps)) | ((u >= 1.5 - eps) & (u <= 2 + eps))))
        self.assertTrue(np.any(np.isclose(u, 0.0)) and np.any(np.isclose(u, 2.0)))

    def test_faces_without_uvs_are_skipped(self):
        """Only the left shell has UVs: everything reads it, the right side clamps to its border."""
        dst         = _grid(8, lift=0.1)
        dst.normals = np.tile([0.0, -1.0, 0.0], (dst.point_count, 1))
        resampler   = MeshDataResampler(self.src, dst, src_uv=_left_shell_only(self.src))
        right       = np.repeat(dst.points[dst.geometry, 0].mean(axis=1) > 0.5, 4)
        x           = dst.points[dst.indices, 0]
        for raycast in (False, True):
            with self.subTest(raycast=raycast):
                uv = resampler.resample_uv(raycast=raycast, forward_only=True)
                u  = uv.points[uv.indices][:, 0]
                self.assertEqual(uv.point_count, dst.point_count)
                self.assertTrue(np.all((u >= -1e-6) & (u <= 0.5 + 1e-6)))
                self.assertTrue(np.allclose(u[right], 0.5))
                if raycast:  # straight down: the left side reads its own x
                    self.assertLess(np.abs(u[~right] - x[~right]).max(), 1e-6)

    def test_subset_raycast_passes_its_options(self):
        """UVs on the left shell only: the faces clear of the seam read as with the whole set."""
        up           = _grid(8, lift=0.1)
        up.normals   = np.tile([0.0, 1.0, 0.0], (up.point_count, 1))
        down         = _grid(8, lift=0.1)
        down.normals = np.tile([0.0, -1.0, 0.0], (down.point_count, 1))
        cases = {
            "forward_only=False": (up, dict(forward_only=False)),
            "twosided=False":     (down, dict(forward_only=True, twosided=False)),
            "BEZIER":             (down, dict(forward_only=True, method=SampleMethod.BEZIER)),
        }
        left_only = _left_shell_only(self.src)
        for name, (dst, options) in cases.items():
            with self.subTest(case=name):
                part  = MeshDataResampler(self.src, dst, src_uv=left_only).resample_uv(raycast=True, **options)
                whole = self.resample_uv(dst, raycast=True, **options)
                clear = np.repeat(dst.points[dst.geometry, 0].max(axis=1) <= 0.4 + 1e-9, 4)
                error = np.abs(part.points[part.indices] - whole.points[whole.indices])[clear].max()
                self.assertLess(error, 1e-9)

    def test_ngon_source_faces(self):
        """An n-gon with UVs raises in both modes; one without UVs is never sampled."""
        src    = _with_hexagon(_grid(10), 44)
        planar = UVData(indices=src.indices.copy(), counts=src.counts.copy(), points=src.points[:, [0, 2]])
        ngon   = src.counts > 4
        uvless = UVData(
            indices = src.indices[np.repeat(~ngon, src.counts)],
            counts  = np.where(ngon, 0, src.counts),
            points  = planar.points,
        )
        dst = _grid(8)
        for raycast in (False, True):
            for method in SampleMethod:
                with self.subTest(raycast=raycast, method=method):
                    resampler = MeshDataResampler(src, dst)
                    with self.assertRaisesRegex(NotImplementedError, r"resample_uv\(\) does not support n-gon"):
                        resampler.resample_uv(uv_data=planar, raycast=raycast, method=method)
                    uv    = resampler.resample_uv(uv_data=uvless, raycast=raycast, method=method)
                    error = np.abs(uv.points[uv.indices] - dst.points[dst.indices][:, [0, 2]]).max()
                    self.assertLess(error, 1e-3)

    def test_uv_data_argument_overrides_src_uv(self):
        dst       = _grid(8)
        resampler = MeshDataResampler(self.src, dst)
        with self.assertRaises(RuntimeError):
            resampler.resample_uv()
        uv = resampler.resample_uv(uv_data=self.src_uv)
        self.assertEqual(_torn_faces(uv), 0)

    def test_foreign_uv_data_raise(self):
        dst       = _grid(8)
        resampler = MeshDataResampler(self.src, dst)
        with self.assertRaisesRegex(ValueError, "their faces differ"):
            resampler.resample_uv(uv_data=_split_uvs(_grid(4)))

        counts = self.src_uv.counts.copy()
        counts[0], counts[1] = 3, 5  # as many faces, two of other sizes
        other_sizes = UVData(indices=self.src_uv.indices, counts=counts, points=self.src_uv.points)
        with self.assertRaisesRegex(ValueError, "their faces differ"):
            resampler.resample_uv(uv_data=other_sizes)

        counts[1] = 4  # one face smaller, none larger
        smaller   = UVData(indices=self.src_uv.indices[1:], counts=counts, points=self.src_uv.points)
        with self.assertRaisesRegex(ValueError, "their faces differ"):
            resampler.resample_uv(uv_data=smaller)

        counts[0] = 5  # one face larger, none smaller
        indices   = np.concatenate([self.src_uv.indices[:4], [0], self.src_uv.indices[4:]])
        larger    = UVData(indices=indices, counts=counts, points=self.src_uv.points)
        with self.assertRaisesRegex(ValueError, "their faces differ"):
            resampler.resample_uv(uv_data=larger)

    def test_uv_set_without_uvs_raise(self):
        empty = UVData(
            indices = np.zeros(0, dtype=np.int32),
            counts  = np.zeros(self.src.face_count, dtype=np.int32),
            points  = np.zeros((0, 2)),
        )
        with self.assertRaisesRegex(ValueError, "no faces with UVs"):
            MeshDataResampler(self.src, _grid(8)).resample_uv(uv_data=empty)

    def test_round_trip_through_resample_mesh(self):
        """resample_uv() then resample_mesh(UV) rebuilds dst from its UVs alone."""
        dst             = _grid(8)
        dst_uv          = self.resample_uv(dst)
        blank           = dst.copy()
        blank.points[:] = 0.0

        resampler = MeshDataResampler(self.src, blank, src_uv=self.src_uv, dst_uv=dst_uv)
        back      = resampler.resample_mesh(mode=ResampleMode.UV)
        self.assertLess(np.linalg.norm(back.points - dst.points, axis=1).max(), 0.002)

    def test_round_trip_beside_a_seam_through_faces(self):
        """The seam cuts dst faces: the way back is exact but beside it, where a vertex can land on the seam."""
        dst             = _grid(7, low=0.02)  # faces 0.14 wide; column 3, x 0.44 to 0.58, reads the right shell
        dst_uv          = self.resample_uv(dst)
        blank           = dst.copy()
        blank.points[:] = 0.0

        resampler = MeshDataResampler(self.src, blank, src_uv=self.src_uv, dst_uv=dst_uv)
        back      = resampler.resample_mesh(mode=ResampleMode.UV)
        error     = np.linalg.norm(back.points - dst.points, axis=1)
        beside    = np.isclose(dst.points[:, 0], 0.44)
        on_seam   = np.isclose(back.points[:, 0], 0.5)
        self.assertLess(error[~beside].max(), 0.002)
        self.assertTrue(np.all(on_seam[beside] | (error[beside] < 0.002)))
        self.assertLess(error.max(), 0.07 + 0.002)  # half a face at most

    def test_round_trip_where_seams_cross_inside_a_face(self):
        """Seams cross inside dst faces: a vertex comes back past half a face, within half its diagonal."""
        src_uv          = _block_uvs(self.src, [0.3, 0.7])  # 9 shells, their corners inside dst faces
        dst             = _grid(8)                          # faces 0.125 wide
        dst_uv          = MeshDataResampler(self.src, dst, src_uv=src_uv).resample_uv()
        blank           = dst.copy()
        blank.points[:] = 0.0

        resampler = MeshDataResampler(self.src, blank, src_uv=src_uv, dst_uv=dst_uv)
        back      = resampler.resample_mesh(mode=ResampleMode.UV)
        error     = np.linalg.norm(back.points - dst.points, axis=1).max()
        self.assertGreater(error, 0.125 / 2)
        self.assertLess(error, 0.125 * np.sqrt(2.0) / 2)

    def test_resample_mesh_uv_accuracy(self):
        """resample_mesh(UV) places every dst vertex where its UV sits on src."""
        for n in (7, 8):
            truth = _grid(n)
            x, z = truth.points[:, 0], truth.points[:, 2]
            dst_uv = UVData(
                indices = truth.indices.copy(),
                counts  = truth.counts.copy(),
                points  = np.stack([x + (x > 0.5 + 1e-9), z], axis=1),
            )
            blank           = truth.copy()
            blank.points[:] = 0.0  # UV mode must not read dst positions

            resampler = MeshDataResampler(self.src, blank, src_uv=self.src_uv, dst_uv=dst_uv)
            for method, tolerance in ((SampleMethod.BILINEAR, 0.003), (SampleMethod.BEZIER, 0.001)):
                with self.subTest(n=n, method=method):
                    back  = resampler.resample_mesh(mode=ResampleMode.UV, method=method)
                    error = np.linalg.norm(back.points - truth.points, axis=1).max()
                    self.assertLess(error, tolerance)