"""Tests for BILINEAR vs BEZIER surface fitting in MeshDataResampler.

Covers:
    - String-to-enum coercion of the ``method`` parameter
    - ``bezier_evaluate`` producing positions on the Bezier surface
    - ``bezier_vectors`` producing PN Quad tangent vectors
    - ``resample_mesh`` yielding different results for BILINEAR vs BEZIER
    - ``MeshData.sample()`` working with meshes that have face-varying normals
"""

import unittest

import numpy as np
from cgmath.geometry.mesh import MeshData, SampleMethod
from cgmath.geometry.resample import MeshDataResampler
from cgmath.geometry.utils.main import (
    bezier_evaluate,
    bezier_vectors,
    bilinear_vectors,
)



def _make_two_quad_mesh(center_height: float = 0.0) -> MeshData:
    """Two quads sharing a center edge.  ``center_height`` lifts verts 2,3.

    Vertex layout (top view, Y up):

        0 --- 2 --- 4
        |     |     |
        1 --- 3 --- 5

    Face 0: [0, 1, 3, 2]
    Face 1: [2, 3, 5, 4]
    """
    points = np.array([
        [-0.5,  0.0,  1.0],
        [ 0.5,  0.0,  1.0],
        [-0.5,  center_height,  0.0],
        [ 0.5,  center_height,  0.0],
        [-0.5,  0.0, -1.0],
        [ 0.5,  0.0, -1.0],
    ], dtype=np.float64)

    indices = np.array([0, 1, 3, 2, 2, 3, 5, 4], dtype=np.int32)
    counts  = np.array([4, 4], dtype=np.int32)
    return MeshData(indices=indices, counts=counts, points=points)


def _make_subdivided_box() -> MeshData:
    """A 4x4 grid of quads (5x5 = 25 vertices) covering the same area as the two-quad deformer."""
    nz, nx = 4, 4
    zs     = np.linspace(1.0, -1.0, nz + 1)
    xs     = np.linspace(-0.5, 0.5, nx + 1)
    points = []
    for z in zs:
        for x in xs:
            points.append([x, 0.0, z])
    points       = np.array(points, dtype=np.float64)

    indices_list = []
    counts_list  = []
    for j in range(nz):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)
            indices_list.extend([v0, v1, v2, v3])
            counts_list.append(4)

    return MeshData(
        indices = np.array(indices_list, dtype=np.int32),
        counts  = np.array(counts_list, dtype=np.int32),
        points  = points,
    )


class TestBezierResample(unittest.TestCase):
    """Tests for the BEZIER surface fitting path in MeshDataResampler."""

    def setUp(self):
        super().setUp()
        self.original  = _make_two_quad_mesh(center_height=0.0)
        self.deformed  = _make_two_quad_mesh(center_height=0.75)
        self.mesh      = _make_subdivided_box()
        self.resampler = MeshDataResampler(self.original, self.mesh)

    # ---- string coercion ------------------------------------------------

    def test_method_string_coercion(self):
        """Passing method as string should produce the same result as the enum."""
        result_str = self.resampler.resample_mesh(
            mesh_data=self.deformed, method="bezier",
        )
        result_enum = self.resampler.resample_mesh(
            mesh_data=self.deformed, method=SampleMethod.BEZIER,
        )
        np.testing.assert_allclose(result_str.points, result_enum.points)

    def test_method_string_bilinear(self):
        """String 'bilinear' should match SampleMethod.BILINEAR."""
        result_str = self.resampler.resample_mesh(
            mesh_data=self.deformed, method="bilinear",
        )
        result_enum = self.resampler.resample_mesh(
            mesh_data=self.deformed, method=SampleMethod.BILINEAR,
        )
        np.testing.assert_allclose(result_str.points, result_enum.points)

    # ---- bilinear vs bezier differ --------------------------------------

    def test_resample_mesh_bilinear_vs_bezier_differ(self):
        """BILINEAR and BEZIER must produce different positions on a wedge deformer."""
        bilinear = self.resampler.resample_mesh(
            mesh_data=self.deformed, method=SampleMethod.BILINEAR,
        )
        bezier = self.resampler.resample_mesh(
            mesh_data=self.deformed, method=SampleMethod.BEZIER,
        )
        self.assertFalse(
            np.allclose(bilinear.points, bezier.points, atol=1e-6),
            "BILINEAR and BEZIER should produce different results on a wedge deformer.",
        )

    def test_resample_mesh_with_offset_bilinear_vs_bezier_differ(self):
        """With maintain_offset=True the difference should also be present."""
        bilinear = self.resampler.resample_mesh(
            mesh_data=self.deformed, maintain_offset=True, method=SampleMethod.BILINEAR,
        )
        bezier = self.resampler.resample_mesh(
            mesh_data=self.deformed, maintain_offset=True, method=SampleMethod.BEZIER,
        )
        self.assertFalse(
            np.allclose(bilinear.points, bezier.points, atol=1e-6),
            "BILINEAR and BEZIER with maintain_offset should differ on a wedge deformer.",
        )

    # ---- bezier_evaluate vs bilinear interpolation ----------------------

    def test_bezier_evaluate_differs_from_bilinear(self):
        """bezier_evaluate at interior (u,v) on a curved face must differ from bilinear."""
        sample_data = self.original.sample(self.mesh)
        source_normals = self.deformed.get_vertex_normals(
            angle_weighted=True, area_weighted=False,
        )

        bilinear_pts = sample_data(self.deformed.points)
        bezier_pts = bezier_evaluate(
            self.deformed.points,
            source_normals,
            sample_data.geometry,
            sample_data.uvs,
        )
        self.assertFalse(
            np.allclose(bilinear_pts, bezier_pts, atol=1e-6),
            "bezier_evaluate should produce different positions than bilinear interpolation.",
        )

    # ---- bezier_vectors vs bilinear_vectors -----------------------------

    def test_bezier_vectors_differ_from_bilinear(self):
        """Bezier tangent vectors should differ from bilinear on a curved mesh."""
        sample_data = self.original.sample(self.mesh)
        source_normals = self.deformed.get_vertex_normals(
            angle_weighted=True, area_weighted=False,
        )
        U_bl, V_bl = bilinear_vectors(
            self.deformed.points, sample_data.geometry, sample_data.uvs,
        )
        U_bz, V_bz = bezier_vectors(
            self.deformed.points, source_normals, sample_data.geometry, sample_data.uvs,
        )
        normals_bl = np.cross(U_bl, V_bl)
        normals_bz = np.cross(U_bz, V_bz)
        self.assertFalse(
            np.allclose(normals_bl, normals_bz, atol=1e-6),
            "Bezier surface normals should differ from bilinear surface normals.",
        )

    # ---- MeshData.sample() with face-varying normals --------------------

    def test_sample_with_face_varying_normals(self):
        """MeshData.sample() must not crash when the mesh carries face-varying normals."""
        mesh_with_normals = self.original.copy()
        n                 = np.array([[0, 1, 0]] * 6, dtype=np.float64)
        ni                = np.array([0, 1, 2, 3, 3, 2, 4, 5], dtype=np.int32)
        mesh_with_normals.normals        = n
        mesh_with_normals.normal_indices = ni

        sample_data = mesh_with_normals.sample(self.mesh)
        self.assertEqual(sample_data.projections.shape[0], self.mesh.point_count)

    # ---- topology preservation ------------------------------------------

    def test_bezier_resample_preserves_topology(self):
        """Bezier-resampled mesh should keep the destination topology."""
        result = self.resampler.resample_mesh(
            mesh_data=self.deformed, method=SampleMethod.BEZIER,
        )
        self.assertEqual(result.point_count, self.mesh.point_count)
        self.assertEqual(result.face_count, self.mesh.face_count)
        self.assertTrue(result.valid)

    # ---- flat mesh: bilinear == bezier ----------------------------------

    def test_flat_mesh_bilinear_equals_bezier(self):
        """On a flat (undeformed) source mesh, BILINEAR and BEZIER should agree."""
        bilinear = self.resampler.resample_mesh(
            mesh_data=self.original, method=SampleMethod.BILINEAR,
        )
        bezier = self.resampler.resample_mesh(
            mesh_data=self.original, method=SampleMethod.BEZIER,
        )
        np.testing.assert_allclose(
            bilinear.points, bezier.points, atol=1e-10,
            err_msg="On a flat mesh BILINEAR and BEZIER should produce identical results.",
        )
