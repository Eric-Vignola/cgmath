import os
import pickle
import unittest

import numpy as np
from cgmath.geometry import MeshData, MorphList, SkinData, UVList
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import SampleMethod
from cgmath.geometry.resample import MeshDataResampler, ResampleMode

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
        modified_mesh = self.src_mesh_data.copy()
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
        skin_data = self.src_skin_data.copy()
        skin_data.influences = [skin_data.influences[0]]
        rng = np.random.default_rng(seed=42)
        skin_data.weights = rng.random((skin_data.weights.shape[0],1))
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
        skin_data = self.src_skin_data.copy()
        skin_data.influences = [skin_data.influences[0]]
        rng = np.random.default_rng(seed=42)
        skin_data.weights = rng.random((skin_data.weights.shape[0],1))
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
        distorted = self.src_mesh_data.copy()
        rng       = np.random.default_rng(seed=123)
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