import unittest

import numpy as np
from cgmath.geometry import MeshData
from cgmath.geometry._saddle_surface import raycast, RaycastData

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=1e-4):
    return np.allclose(x, y, atol=atol)


class TestRaycast(unittest.TestCase):
    """Tests for the raycast() function in _saddle_surface.py"""

    def setUp(self):
        super().setUp()

        # Unit cube: 6 quad faces, centered at origin, side length 1
        # Face 0 (front +Z): vertices 0,1,3,2
        # Face 1 (top +Y):   vertices 2,3,5,4
        # Face 2 (back -Z):  vertices 4,5,7,6
        # Face 3 (bottom -Y):vertices 6,7,1,0
        # Face 4 (right +X): vertices 1,7,5,3
        # Face 5 (left -X):  vertices 6,0,2,4
        self.cube_points = np.array(
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
            ],
            dtype=np.float64,
        )
        self.cube_indices = np.array(
            [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
        )
        self.cube_counts = np.array([4, 4, 4, 4, 4, 4])
        self.cube = MeshData(
            points  = self.cube_points,
            indices = self.cube_indices,
            counts  = self.cube_counts,
        )
        self.cube_geometry = self.cube.f2v

        # Single quad plane at z=0 with normal pointing +Z
        # Winding: p0(-0.5,-0.5,0) p1(0.5,-0.5,0) p2(0.5,0.5,0) p3(-0.5,0.5,0)
        self.plane_points = np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
            dtype=np.float64,
        )
        self.plane_geometry = np.array([[0, 1, 2, 3]], dtype=np.int32)

        # Triangle mesh
        self.tri_points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
        )
        self.tri_geometry = np.array([[0, 1, 2]], dtype=np.int32)

    # -------------------- basic hit/miss -------------------- #

    def test_forward_hit_center(self):
        """Ray from +Z toward origin hits the front face at center"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertIsInstance(result, RaycastData)
        self.assertTrue(allclose(result.distances[0], 1.5))
        self.assertTrue(allclose(result.projections[0], [0.0, 0.0, 0.5]))
        self.assertEqual(result.indices[0], 0)

    def test_hit_flag_on_hit(self):
        """hit is True for successful intersections"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(result.hit[0])

    def test_miss(self):
        """Ray pointing away from the cube produces a miss"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, 1.0]])  # pointing away

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertEqual(result.indices[0], -1)
        self.assertFalse(result.hit[0])

    def test_parallel_miss(self):
        """Ray parallel to a face misses the mesh"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[1.0, 0.0, 0.0]])  # parallel to front face

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertEqual(result.indices[0], -1)

    # -------------------- UV coordinates -------------------- #

    def test_uv_at_center(self):
        """Ray through face center gives UV (0.5, 0.5)"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(allclose(result.uvs[0], [0.5, 0.5]))

    def test_uv_at_corners(self):
        """Rays through exact face corners give expected UV values"""
        # Front face (0,1,3,2): p0=(-0.5,-0.5,0.5) p1=(0.5,-0.5,0.5)
        #                       p2=(0.5,0.5,0.5)   p3=(-0.5,0.5,0.5)
        corners = [
            ([-0.5, -0.5, 2.0], [0.0, 0.0]),  # p0 -> UV(0,0)
            ([0.5, -0.5, 2.0], [1.0, 0.0]),  # p1 -> UV(1,0)
            ([0.5, 0.5, 2.0], [1.0, 1.0]),  # p2 -> UV(1,1)
            ([-0.5, 0.5, 2.0], [0.0, 1.0]),  # p3 -> UV(0,1)
        ]
        for origin, expected_uv in corners:
            origins    = np.array([origin])
            directions = np.array([[0.0, 0.0, -1.0]])

            result = raycast(
                self.cube_points,
                self.cube_points,
                self.cube_geometry,
                origins,
                directions,
            )
            self.assertTrue(
                allclose(result.uvs[0], expected_uv),
                f"Expected UV {expected_uv} at origin {origin}, got {result.uvs[0]}",
            )

    # -------------------- multiple rays -------------------- #

    def test_multiple_rays(self):
        """Batch of rays all hitting the cube"""
        origins = np.array(
            [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
        )
        directions = np.array(
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
        )

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertEqual(result.distances.shape[0], 3)
        self.assertTrue(np.all(~np.isnan(result.distances)))
        self.assertTrue(np.all(result.indices >= 0))
        # all hits at distance 1.5 from face center
        self.assertTrue(allclose(result.distances, [1.5, 1.5, 1.5]))

    def test_mixed_hits_and_misses(self):
        """Batch with some hits and some misses"""
        origins    = np.array([[0.0, 0.0, 2.0], [10.0, 10.0, 10.0]], dtype=np.float64)
        directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        # first ray hits
        self.assertTrue(result.hit[0])
        self.assertGreaterEqual(result.indices[0], 0)
        # second ray misses
        self.assertFalse(result.hit[1])
        self.assertEqual(result.indices[1], -1)

    def test_miss_projections_are_origins(self):
        """Missed rays have projections set to their origins"""
        origins    = np.array([[10.0, 10.0, 10.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertFalse(result.hit[0])
        self.assertTrue(allclose(result.projections[0], origins[0]))

    # -------------------- forward_only -------------------- #

    def test_forward_only_true(self):
        """forward_only=True only finds forward hits"""
        # ray from origin pointing +Z should hit back of cube at z=0.5
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            forward_only=True,
        )
        self.assertTrue(allclose(result.distances[0], 0.5))

    def test_forward_only_false(self):
        """forward_only=False also considers backward hits and keeps the closer"""
        # ray from origin pointing +Z: forward hit at z=0.5, backward hit at z=-0.5
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            forward_only=False,
        )
        # both forward and backward hit at t=0.5, so distance should be 0.5
        self.assertTrue(allclose(result.distances[0], 0.5))
        self.assertGreaterEqual(result.indices[0], 0)

    # -------------------- twosided -------------------- #

    def test_twosided_true_hits_from_both_sides(self):
        """twosided=True (default) hits the plane from either side"""
        # front side: ray from +Z toward plane
        origins_front = np.array([[0.0, 0.0, 1.0]])
        dirs_front    = np.array([[0.0, 0.0, -1.0]])

        result_front = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins_front,
            dirs_front,
            twosided=True,
        )
        self.assertFalse(np.isnan(result_front.distances[0]))

        # back side: ray from -Z toward plane
        origins_back = np.array([[0.0, 0.0, -1.0]])
        dirs_back    = np.array([[0.0, 0.0, 1.0]])

        result_back = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins_back,
            dirs_back,
            twosided=True,
        )
        self.assertFalse(np.isnan(result_back.distances[0]))

    def test_twosided_false_front_face_hit(self):
        """twosided=False accepts front-face hits (ray opposes normal)"""
        # plane normal is +Z, ray comes from +Z in -Z direction -> front face
        origins    = np.array([[0.0, 0.0, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins,
            directions,
            twosided=False,
        )
        self.assertFalse(np.isnan(result.distances[0]))
        self.assertTrue(allclose(result.distances[0], 1.0))

    def test_twosided_false_back_face_rejected(self):
        """twosided=False rejects back-face hits (ray aligns with normal)"""
        # plane normal is +Z, ray comes from -Z in +Z direction -> back face
        origins    = np.array([[0.0, 0.0, -1.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins,
            directions,
            twosided=False,
        )
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertEqual(result.indices[0], -1)

    def test_twosided_false_cube_from_outside(self):
        """twosided=False still hits cube faces from outside (all front-facing)"""
        origins = np.array(
            [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0]], dtype=np.float64
        )
        directions = np.array(
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            twosided=False,
        )
        self.assertTrue(np.all(~np.isnan(result.distances)))
        self.assertTrue(allclose(result.distances, [1.5, 1.5, 1.5]))

    def test_twosided_false_cube_from_inside(self):
        """twosided=False rejects all hits when casting from inside the cube
        because all faces' normals point outward (same direction as the ray)"""
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            twosided=False,
        )
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertEqual(result.indices[0], -1)

    # -------------------- triangles -------------------- #

    def test_triangle_hit(self):
        """Ray hits a triangle face"""
        origins    = np.array([[0.4, 0.3, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.tri_points,
            self.tri_points,
            self.tri_geometry,
            origins,
            directions,
        )
        self.assertFalse(np.isnan(result.distances[0]))
        self.assertTrue(allclose(result.distances[0], 1.0))
        self.assertEqual(result.indices[0], 0)

    def test_triangle_weights_sum_to_one(self):
        """Weights on a triangle face sum to 1 with w3 == 0"""
        origins    = np.array([[0.4, 0.3, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.tri_points,
            self.tri_points,
            self.tri_geometry,
            origins,
            directions,
        )
        self.assertTrue(allclose(np.sum(result.weights[0]), 1.0))
        self.assertTrue(allclose(result.weights[0, 3], 0.0))

    # -------------------- weights -------------------- #

    def test_quad_weights_sum_to_one(self):
        """Bilinear weights on a quad face sum to 1"""
        origins    = np.array([[0.1, -0.2, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(allclose(np.sum(result.weights[0]), 1.0))

    def test_weights_at_center(self):
        """Weights at face center should be equal (0.25 each)"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(allclose(result.weights[0], [0.25, 0.25, 0.25, 0.25]))

    # -------------------- normals / occluded -------------------- #

    def test_normals_computed_with_vertex_normals(self):
        """When vertex normals are provided, interpolated normals are returned"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        normals    = np.zeros_like(self.cube_points)
        normals[:, 2] = 1.0  # all normals pointing +Z

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            normals=normals,
        )
        # interpolated normal should point in +Z
        self.assertTrue(allclose(result.normals[0], [0.0, 0.0, 1.0]))

    def test_occluded_flag(self):
        """Occluded is True when ray direction opposes the surface normal"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        normals    = np.zeros_like(self.cube_points)
        normals[:, 2] = 1.0

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            normals=normals,
        )
        # ray direction (-Z) * normal (+Z) = -1 < 0 -> occluded = True
        self.assertTrue(result.occluded[0])

    # -------------------- geometry field -------------------- #

    def test_geometry_field_for_hits(self):
        """Hit results carry the correct face vertex indices"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        face_idx     = result.indices[0]
        expected_geo = self.cube_geometry[face_idx]
        self.assertTrue(np.array_equal(result.geometry[0], expected_geo))

    def test_geometry_field_for_misses(self):
        """Miss results have -1 geometry"""
        origins    = np.array([[10.0, 10.0, 10.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
        )
        self.assertTrue(np.all(result.geometry[0] == -1))


class TestRaycastData(unittest.TestCase):
    """Tests for the RaycastData dataclass"""

    def setUp(self):
        super().setUp()

        self.data = RaycastData(
            projections=np.array(
                [[0.0, 0.0, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64
            ),
            distances = np.array([1.5, 1.5]),
            weights   = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.25, 0.125, 0.125]]),
            indices   = np.array([0, 0], dtype=np.int32),
            normals   = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            occluded  = np.array([True, True]),
            uvs       = np.array([[0.5, 0.5], [0.75, 0.25]]),
            geometry  = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32),
            hit       = np.array([True, True]),
        )

    def test_compute(self):
        """RaycastData.compute() produces weighted values"""
        values = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64
        )
        result = self.data.compute(values)
        self.assertEqual(result.shape, (2, 2))
        # first point: equal weights -> mean of all 4
        self.assertTrue(allclose(result[0], [0.5, 0.5]))

    def test_call(self):
        """__call__ delegates to compute"""
        values = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64
        )
        self.assertTrue(allclose(self.data(values), self.data.compute(values)))

    def test_copy(self):
        """copy() creates an independent deep copy"""
        copy = self.data.copy()
        self.assertTrue(allclose(copy.distances, self.data.distances))

        self.data.distances[0] = 999.0
        self.assertNotAlmostEqual(copy.distances[0], 999.0)


class TestMeshDataRaycast(unittest.TestCase):
    """Tests for MeshData.raycast() method"""

    def setUp(self):
        super().setUp()

        self.cube = MeshData(
            points=np.array(
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                ],
                dtype=np.float64,
            ),
            indices=np.array(
                [
                    0,
                    1,
                    3,
                    2,
                    2,
                    3,
                    5,
                    4,
                    4,
                    5,
                    7,
                    6,
                    6,
                    7,
                    1,
                    0,
                    1,
                    7,
                    5,
                    3,
                    6,
                    0,
                    2,
                    4,
                ]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

    def test_basic_raycast(self):
        """MeshData.raycast returns RaycastData"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result     = self.cube.raycast(origins, directions)
        self.assertIsInstance(result, RaycastData)
        self.assertTrue(allclose(result.distances[0], 1.5))

    def test_forward_only_parameter(self):
        """forward_only parameter is passed through"""
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result     = self.cube.raycast(origins, directions, forward_only=False)
        self.assertFalse(np.isnan(result.distances[0]))

    def test_twosided_parameter(self):
        """twosided parameter is passed through"""
        # from inside, twosided=False should miss (normals face outward)
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result     = self.cube.raycast(origins, directions, twosided=False)
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertEqual(result.indices[0], -1)

    def test_normals_auto_computed(self):
        """MeshData.raycast automatically computes vertex normals"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result     = self.cube.raycast(origins, directions)
        # normal at hit point should roughly face +Z
        self.assertGreater(result.normals[0, 2], 0.5)


class TestBezierRaycast(unittest.TestCase):
    """Tests for the Bezier (PN Quad) raycast path in _saddle_surface.py"""

    def setUp(self):
        super().setUp()

        self.cube_points = np.array(
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
            ],
            dtype=np.float64,
        )
        self.cube_indices = np.array(
            [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
        )
        self.cube_counts = np.array([4, 4, 4, 4, 4, 4])
        self.cube = MeshData(
            points  = self.cube_points,
            indices = self.cube_indices,
            counts  = self.cube_counts,
        )
        self.cube_geometry = self.cube.f2v
        self.cube_normals = self.cube.get_vertex_normals(
            angle_weighted=True, area_weighted=False
        )

        self.plane_points = np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
            dtype=np.float64,
        )
        self.plane_geometry = np.array([[0, 1, 2, 3]], dtype=np.int32)
        self.plane_normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self.tri_points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
        )
        self.tri_geometry = np.array([[0, 1, 2]], dtype=np.int32)
        self.tri_normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def test_forward_hit_center(self):
        """Bezier ray from +Z toward origin hits the front face"""
        origins    = np.array([[0.0, 0.0, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins,
            directions,
            normals         = self.plane_normals,
            surface_normals = self.plane_normals,
        )
        self.assertIsInstance(result, RaycastData)
        self.assertTrue(result.hit[0])
        self.assertTrue(allclose(result.distances[0], 1.0, atol=0.05))
        self.assertTrue(allclose(result.projections[0, 2], 0.0, atol=0.05))
        self.assertEqual(result.indices[0], 0)

    def test_miss(self):
        """Bezier ray pointing away produces a miss"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            normals         = self.cube_normals,
            surface_normals = self.cube_normals,
        )
        self.assertFalse(result.hit[0])
        self.assertTrue(np.isnan(result.distances[0]))

    def test_weights_sum_to_one(self):
        """Bezier hit weights sum to 1"""
        origins    = np.array([[0.1, -0.2, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            normals         = self.cube_normals,
            surface_normals = self.cube_normals,
        )
        self.assertTrue(result.hit[0])
        self.assertTrue(allclose(np.sum(result.weights[0]), 1.0))

    def test_backward_cast(self):
        """Bezier forward_only=False finds backward hits"""
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            normals         = self.cube_normals,
            surface_normals = self.cube_normals,
            forward_only    = False,
        )
        self.assertTrue(result.hit[0])
        self.assertFalse(np.isnan(result.distances[0]))

    def test_twosided_false_rejects_backface(self):
        """Bezier twosided=False rejects back-face hits"""
        origins    = np.array([[0.0, 0.0, -1.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins,
            directions,
            normals         = self.plane_normals,
            surface_normals = self.plane_normals,
            twosided        = False,
        )
        self.assertFalse(result.hit[0])

    def test_twosided_true_hits_backface(self):
        """Bezier twosided=True (default) hits from either side"""
        origins_front = np.array([[0.0, 0.0, 1.0]])
        dirs_front    = np.array([[0.0, 0.0, -1.0]])
        origins_back  = np.array([[0.0, 0.0, -1.0]])
        dirs_back     = np.array([[0.0, 0.0, 1.0]])

        result_front = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins_front,
            dirs_front,
            normals         = self.plane_normals,
            surface_normals = self.plane_normals,
        )
        result_back = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins_back,
            dirs_back,
            normals         = self.plane_normals,
            surface_normals = self.plane_normals,
        )
        self.assertTrue(result_front.hit[0])
        self.assertTrue(result_back.hit[0])

    def test_triangle_hit(self):
        """Bezier raycast hits a triangle face"""
        origins    = np.array([[0.4, 0.3, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result = raycast(
            self.tri_points,
            self.tri_points,
            self.tri_geometry,
            origins,
            directions,
            normals         = self.tri_normals,
            surface_normals = self.tri_normals,
        )
        self.assertTrue(result.hit[0])
        self.assertTrue(allclose(result.projections[0, 2], 0.0))

    def test_batch_hits_and_misses(self):
        """Bezier batch with hits and misses"""
        origins    = np.array([[0.0, 0.0, 2.0], [10.0, 10.0, 10.0]], dtype=np.float64)
        directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        result = raycast(
            self.cube_points,
            self.cube_points,
            self.cube_geometry,
            origins,
            directions,
            normals         = self.cube_normals,
            surface_normals = self.cube_normals,
        )
        self.assertTrue(result.hit[0])
        self.assertFalse(result.hit[1])

    def test_matches_bilinear_on_flat_faces(self):
        """On a flat plane, Bezier and bilinear results should agree closely"""
        origins    = np.array([[0.0, 0.0, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result_bilinear = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins,
            directions,
            normals=self.plane_normals,
        )
        result_bezier = raycast(
            self.plane_points,
            self.plane_points,
            self.plane_geometry,
            origins,
            directions,
            normals         = self.plane_normals,
            surface_normals = self.plane_normals,
        )
        self.assertTrue(result_bilinear.hit[0])
        self.assertTrue(result_bezier.hit[0])
        self.assertTrue(allclose(result_bilinear.distances, result_bezier.distances))
        self.assertTrue(
            allclose(result_bilinear.projections, result_bezier.projections)
        )


class TestMeshDataBezierRaycast(unittest.TestCase):
    """Tests for MeshData.raycast(method='bezier')"""

    def setUp(self):
        super().setUp()

        self.cube = MeshData(
            points=np.array(
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                ],
                dtype=np.float64,
            ),
            indices=np.array(
                [
                    0,
                    1,
                    3,
                    2,
                    2,
                    3,
                    5,
                    4,
                    4,
                    5,
                    7,
                    6,
                    6,
                    7,
                    1,
                    0,
                    1,
                    7,
                    5,
                    3,
                    6,
                    0,
                    2,
                    4,
                ]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

    def test_bezier_raycast(self):
        """MeshData.raycast(method='bezier') returns RaycastData"""
        plane = MeshData(
            points=np.array(
                [
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                ],
                dtype=np.float64,
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )
        origins    = np.array([[0.0, 0.0, 1.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result     = plane.raycast(origins, directions, method="bezier")
        self.assertIsInstance(result, RaycastData)
        self.assertTrue(result.hit[0])
        self.assertTrue(allclose(result.distances[0], 1.0, atol=0.05))

    def test_bezier_forward_only(self):
        """method='bezier' with forward_only=False finds backward hits"""
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result = self.cube.raycast(
            origins, directions, method="bezier", forward_only=False
        )
        self.assertTrue(result.hit[0])

    def test_bezier_twosided(self):
        """method='bezier' with twosided=False rejects inside-out hits"""
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        result     = self.cube.raycast(origins, directions, method="bezier", twosided=False)
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertEqual(result.indices[0], -1)

    def test_bezier_normals(self):
        """method='bezier' automatically computes normals"""
        origins    = np.array([[0.0, 0.0, 2.0]])
        directions = np.array([[0.0, 0.0, -1.0]])

        result     = self.cube.raycast(origins, directions, method="bezier")
        self.assertGreater(result.normals[0, 2], 0.5)

    def test_method_enum_and_string(self):
        """method accepts both SampleMethod enum and string"""
        from cgmath.geometry.mesh import SampleMethod

        origins     = np.array([[0.0, 0.0, 2.0]])
        directions  = np.array([[0.0, 0.0, -1.0]])

        result_str  = self.cube.raycast(origins, directions, method="bezier")
        result_enum = self.cube.raycast(origins, directions, method=SampleMethod.BEZIER)
        self.assertTrue(result_str.hit[0])
        self.assertTrue(result_enum.hit[0])
        self.assertTrue(allclose(result_str.distances, result_enum.distances))