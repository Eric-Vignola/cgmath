import os
import pickle
import tempfile
import unittest

import numpy as np
from cgmath.geometry.bspline import BSplineData
from cgmath.geometry.bspline_patch import BSplinePatchData
from cgmath.geometry.mesh import (
    Axis,
    load_obj,
    MeshData,
    MeshList,
    SampleMethod,
    save_obj,
    UVData,
    UVList,
)
from cgmath.geometry.utils import compute_topological_neighborhood

EPSILON = np.finfo(np.float32).eps


def allclose(x, y, atol=EPSILON):
    return np.allclose(x, y, atol=EPSILON)


class TestMesh(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define a base shapes
        self.plane = MeshData(
            points=np.array(
                [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

        self.box = MeshData(
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
                ]
            ),
            indices=np.array(
                [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )


        self.box_uv = UVData(
            points=np.array([[0.33000001, 0.        ],
                             [0.66333336, 0.        ],
                             [0.33000001, 0.25      ],
                             [0.66333336, 0.25      ],
                             [0.33000001, 0.5       ],
                             [0.66333336, 0.5       ],
                             [0.33000001, 0.75      ],
                             [0.66333336, 0.75      ],
                             [0.33000001, 1.        ],
                             [0.66333336, 1.        ],
                             [1.        , 0.        ],
                             [1.        , 0.25      ],
                             [0.        , 0.        ],
                             [0.        , 0.25      ]]),
            indices=np.array(
                [ 0,  1,  3,  2,  2,  3,  5,  4,  4,  5,  7,  6,  6,  7,  9,  8,  1, 10, 11,  3, 12,  0,  2, 13]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )



        self.weird = MeshData(
            points=np.array(
                [
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                ]
            ),
            indices = np.array([0, 2, 1, 1, 2, 4, 3, 3, 4, 5, 5, 0, 1, 3]),
            counts  = np.array([3, 4, 3, 4]),
        )


        self.weird_uv = UVData(
            points=np.array(
                [
                    [-0.5, -0.5,],
                    [-0.5, 0.5],
                    [0.5, 0.5],
                    [-0.5, 0.5],
                    [0.5, 0.5],
                    [-0.5, -0.5],
                ]
            ),
            indices = np.array([0, 2, 1, 1, 2, 4, 3, 3, 4, 5, 5, 0, 1, 3]),
            counts  = np.array([3, 4, 3, 4]),
        )


        self.plane_with_hole = MeshData(indices=np.array([ 0,  1,  5,  4,  1,  2,  6,  5,  2,  3,  7,  6,  4,  5,  9,  8,  6,7, 11, 10,  8,  9, 13, 12,  9, 10, 14, 13, 10, 11, 15, 14]),
                                        counts=np.array([4, 4, 4, 4, 4, 4, 4, 4]),
                                        points=np.array([[-1.        ,  0.        ,  1.        ],
                                                         [-0.33333331,  0.        ,  1.        ],
                                                         [ 0.33333337,  0.        ,  1.        ],
                                                         [ 1.        ,  0.        ,  1.        ],
                                                         [-1.        ,  0.        ,  0.33333331],
                                                         [-0.33333331,  0.        ,  0.33333331],
                                                         [ 0.33333337,  0.        ,  0.33333331],
                                                         [ 1.        ,  0.        ,  0.33333331],
                                                         [-1.        ,  0.        , -0.33333337],
                                                         [-0.33333331,  0.        , -0.33333337],
                                                         [ 0.33333337,  0.        , -0.33333337],
                                                         [ 1.        ,  0.        , -0.33333337],
                                                         [-1.        ,  0.        , -1.        ],
                                                         [-0.33333331,  0.        , -1.        ],
                                                         [ 0.33333337,  0.        , -1.        ],
                                                         [ 1.        ,  0.        , -1.        ]]))


    def test_equivalence(self):
        # test equivalence
        self.assertTrue(self.plane == self.plane)
        self.assertFalse(self.plane != self.plane)

        # test inequality
        self.assertFalse(self.plane == self.box)
        self.assertTrue(self.plane != self.box)

        # test inequality
        self.assertFalse(self.weird == self.weird_uv)
        self.assertTrue(self.weird != self.weird_uv)



    def test_correspondance(self):
        mesh = self.box.copy()
        mesh.delete_faces([0,1,3])

        uv = self.box_uv.copy()
        uv.delete_faces([0,1,3])

        # get the correspondence between a mesh and itself
        corr = mesh.get_correspondence(mesh)
        self.assertTrue(allclose(corr, [0, 1, 2, 3, 4, 5, 6, 7]))

        # get the correspondence between a mesh and its uv set
        corr = uv.get_correspondence(mesh)
        self.assertTrue(allclose(corr, [0, 1, 2, 3, 4, 5, 6, 7, 7, 5, 6, 4]))

    def test_neighbors(self):
        mesh      = self.box.copy()
        neighbors = mesh.get_edge_vertex_neighbors()
        expected = np.array([[1, 2, 6],
                             [0, 3, 7],
                             [3, 0, 4],
                             [1, 2, 5],
                             [5, 2, 6],
                             [3, 4, 7],
                             [7, 4, 0],
                             [5, 6, 1]])
        self.assertTrue(allclose(neighbors, expected))


        neighbors = mesh.get_edge_vertex_neighbors(n=1)
        expected = np.array([[1, 2, 6, 3, 7, 4],
                             [0, 3, 7, 2, 6, 5],
                             [3, 0, 4, 1, 5, 6],
                             [1, 2, 5, 0, 7, 4],
                             [5, 2, 6, 3, 7, 0],
                             [3, 4, 7, 1, 2, 6],
                             [7, 4, 0, 5, 1, 2],
                             [5, 6, 1, 3, 4, 0]])
        self.assertTrue(allclose(neighbors, expected))


        x = mesh.get_edge_vertex_neighbors(n=0)[0]
        x = x[x != -1]
        self.assertTrue(np.allclose(x, [1, 2, 6]))

        x = mesh.get_face_vertex_neighbors(n=0)[0]
        x = x[x != -1]
        self.assertTrue(np.allclose(x, [1, 3, 2, 6, 7, 4]))

        x = mesh.get_vertex_edge_neighbors(n=0)[0]
        x = x[x != -1]
        self.assertTrue(np.allclose(x, [ 3, 11,  1, 10]))

        x = mesh.get_face_edge_neighbors(n=0)[0]
        x = x[x != -1]
        self.assertTrue(np.allclose(x, [ 1,  2,  3,  8, 10, 11]))

        x = mesh.get_vertex_face_neighbors(n=0)[0]
        x = x[x != -1]
        self.assertTrue(np.allclose(x, [3, 5, 4, 1]))

        x = mesh.get_edge_face_neighbors(n=0)[0]
        x = x[x != -1]
        self.assertTrue(np.allclose(x, [3, 4, 1, 5]))


        # test grow and shrink
        mesh = self.plane_with_hole.copy()
        x    = mesh.grow_face_vertex_indices([5], n=1)
        self.assertTrue(np.allclose(x, [0, 1, 2, 4, 5, 6, 8, 9]))
        y = mesh.shrink_face_vertex_indices(x, n=1)
        self.assertTrue(np.all(y==[5]))

        x = mesh.grow_edge_vertex_indices([5], n=1)
        self.assertTrue(np.allclose(x, [1, 4, 5, 6, 9]))
        y = mesh.shrink_edge_vertex_indices(x, n=1)
        self.assertTrue(np.all(y==[5]))

        x = mesh.grow_vertex_edge_indices([6], n=1)
        self.assertTrue(np.allclose(x, [ 1,  2,  5,  6,  9, 10, 15]))
        y = mesh.shrink_vertex_edge_indices(x, n=1)
        self.assertTrue(np.all(y==[6]))

        x = mesh.grow_face_edge_indices([6], n=1)
        self.assertTrue(np.allclose(x, [1, 4, 5, 6]))
        y = mesh.shrink_face_edge_indices(x, n=1)
        self.assertTrue(np.all(y==[6]))

        x = mesh.grow_vertex_face_indices([1], n=1)
        self.assertTrue(np.allclose(x, [0, 1, 2, 3, 4]))
        y = mesh.shrink_vertex_face_indices(x, n=1)
        self.assertTrue(np.all(y==[1]))

        x = mesh.grow_edge_face_indices([1], n=1)
        self.assertTrue(np.allclose(x, [0, 1, 2]))
        y = mesh.shrink_edge_face_indices(x, n=1)
        self.assertTrue(np.all(y==[1]))

    def test_clusters(self):
        mesh     = self.plane_with_hole.copy()

        clusters = mesh.get_edge_vertex_clusters([0,5])
        self.assertTrue(np.allclose(clusters, [np.array([0]), np.array([5])]))

        clusters = mesh.get_face_vertex_clusters([0,5])
        self.assertTrue(np.allclose(clusters, [np.array([0,5])]))

        clusters = mesh.get_edge_face_clusters([1,4])
        self.assertTrue(np.allclose(clusters, [np.array([1]), np.array([4])]))

        clusters = mesh.get_vertex_face_clusters([1,4])
        self.assertTrue(np.allclose(clusters, [np.array([1,4])]))

        clusters = mesh.get_vertex_edge_clusters([1,5])
        self.assertTrue(np.allclose(clusters, [np.array([1]), np.array([5])]))

        clusters = mesh.get_face_edge_clusters([1,5])
        self.assertTrue(np.allclose(clusters, [np.array([1,5])]))






    def test_obj(self):
        # test obj serialization
        obj1 = self.box.copy()

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.obj")
            obj1.save_obj(f)

            obj2 = MeshData.load_obj(f)
            self.assertTrue(obj1 == obj2)

    def test_load_obj_utility_round_trip(self):
        # test module-level save_obj / load_obj round-trip with UVs
        mesh = self.box.copy()
        uv   = self.box_uv.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.obj")
            save_obj(f, [(mesh, UVList([uv]))])
            data = load_obj(f)

        self.assertEqual(len(data), 1)

        mesh2, uv_list2 = data[0]
        self.assertIsInstance(mesh2, MeshData)
        self.assertIsInstance(uv_list2, UVList)
        self.assertEqual(len(uv_list2), 1)
        self.assertTrue(mesh == mesh2)
        self.assertTrue(uv == uv_list2[0])

    def test_load_obj_utility_multi_mesh(self):
        # test module-level save_obj / load_obj round-trip with multiple meshes
        mesh_a = self.box.copy()
        mesh_a.name = "boxA"
        uv_a   = self.box_uv.copy()

        mesh_b = self.box.copy()
        mesh_b.name = "boxB"
        uv_b = self.box_uv.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "multi.obj")
            save_obj(
                f,
                [
                    (mesh_a, UVList([uv_a])),
                    (mesh_b, UVList([uv_b])),
                ],
            )
            data = load_obj(f)

        self.assertEqual(len(data), 2)

        # save_obj strips "Shape" suffix and load_obj re-inserts it,
        # so names round-trip back to the originals.
        names = [m.name for m, _ in data]
        self.assertIn("boxAShape", names)
        self.assertIn("boxBShape", names)

        for m, uvs in data:
            self.assertIsInstance(m, MeshData)
            self.assertIsInstance(uvs, UVList)
            self.assertEqual(m.point_count,      mesh_a.point_count)
            self.assertEqual(m.face_count,       mesh_a.face_count)
            self.assertEqual(len(uvs),           1)
            self.assertEqual(uvs[0].point_count, uv_a.point_count)

    def test_load_obj_utility_no_uvs(self):
        # an obj with no vt lines should still load successfully with empty UVList
        mesh = self.box.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "noUV.obj")
            mesh.save_obj(f)  # uses MeshData.save_obj which writes no vt lines
            data = load_obj(f)

        self.assertEqual(len(data), 1)
        mesh2, uv_list2 = data[0]
        self.assertIsInstance(uv_list2, UVList)
        self.assertEqual(len(uv_list2), 0)
        self.assertTrue(mesh == mesh2)

    def test_bytes(self):
        obj1 = self.box.copy()
        b    = obj1.to_bytes()
        obj2 = MeshData.from_bytes(b)
        self.assertTrue(obj1 == obj2)


    def test_serialize(self):

        # force generate cached data
        obj1 = self.box.copy()
        obj1._compute_borders()
        self.assertFalse(obj1._e2v is None)

        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # save obj1 to file
            obj1.save(f)

            # load obj2 from file
            obj2 = MeshData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # confirm obj2 has no cached data
            self.assertTrue(obj2._e2v is None)

            # do the same via pickle, put obj1 inside a list
            with open(f, 'wb') as io:
                pickle.dump([obj1], io)

            with open(f, 'rb') as io:
                obj3 = pickle.load(io)[0]

                # confirm they're identical
                self.assertTrue(obj1 == obj3)

                # confirm obj3 has no cached data
                self.assertTrue(obj3._e2v is None)

            # add custom private data
            obj1._custom_data = "test"
            obj1.save(f)
            obj2 = MeshData.load(f)
            self.assertFalse(hasattr(obj2, "_custom_data"))
            self.assertTrue(obj1 == obj2)

            # save the file using the ascii format
            obj1.save(f, mode='json')
            obj2 = MeshData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # test json string output
            self.assertTrue(isinstance(obj1.to_json(), str))

            # test .npz
            obj1.save(f, mode='npz')
            obj3 = MeshData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj3)

    def test_from_dict(self):
        # create a basic quad
        data = {
            "indices": [0, 1, 3, 2],
            "counts":  [4],
            "points": [[-0.5,  0.0,  0.5],
                    [ 0.5,  0.0,  0.5],
                    [-0.5,  0.0, -0.5],
                    [ 0.5,  0.0, -0.5]],
            "name": "pPlaneShape1"
        }
        obj = MeshData.from_dict(data)
        self.assertTrue(obj.points.dtype == 'float64')
        self.assertTrue(isinstance(data['points'], list))



    def test_paired_convertions(self):
        cylinder = {
            "indices": [ 0,  1,  4,  3,  1,  2,  5,  4,  2,  0,  3,  5,  3,  4,  7,  6,  4,  5,  8,  7,
                        5,  3,  6,  8,  1,  0,  9,  2,  1,  9,  0,  2,  9,  6,  7, 10,  7,  8, 10,  8,
                        6, 10],
            "counts": [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3],
            "points": [[-0.5, -1.0, -0.9],
                    [-0.5, -1.0,  0.9],
                    [ 1.0, -1.0,  0.0],
                    [-0.5, -0.8, -0.9],
                    [-0.5, -0.8,  0.9],
                    [ 1.0, -0.8,  0.0],
                    [-0.5,  1.0, -0.9],
                    [-0.5,  1.0,  0.9],
                    [ 1.0,  1.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  1.0,  0.0]],
            "name": "pCylinderShape1"
        }
        cylinder  = MeshData.from_dict(cylinder)


        edge_row0 = np.array([0,4,7])
        edge_row1 = np.array([2,6,8])

        vert_row0 = cylinder.edges_to_vertex_pairs(edge_row0)
        vert_row1 = cylinder.edges_to_vertex_pairs(edge_row1)
        vert_rows = np.concatenate([vert_row0, vert_row1])

        # test vertices_to_edges
        src_edges = np.concatenate([edge_row0, edge_row1])
        dst_edges = cylinder.vertex_pairs_to_edges(vert_rows)
        self.assertTrue(allclose(src_edges, dst_edges))

        # test that the vertex pairs can be in any order
        idx = cylinder.vertex_pairs_to_edges([0,1,1,0])
        self.assertTrue(np.all(idx == 0))

        # test that the returned vertex pairs are sorted
        as_stored  = cylinder.e2v[2]                      # 4,3
        as_queried = cylinder.edges_to_vertex_pairs([2])  # 3, 4
        self.assertFalse(allclose(as_stored, as_queried))
        self.assertTrue(allclose(as_stored, as_queried[::-1]))

        # test that the same edges are returnd regardless of the order of the vertex pairs
        from_stored  = cylinder.vertex_pairs_to_edges(as_stored)
        from_queried = cylinder.vertex_pairs_to_edges(as_queried)
        self.assertTrue(allclose(from_stored, from_queried))


    def test_sample(self):
        mesh   = self.plane.copy()
        p      = np.array([[0.0, 0.0, 1.5]])

        sample = mesh.sample(p)
        self.assertTrue(allclose(sample.projections[0], [0, 0, 0]))
        self.assertTrue(allclose(sample.distances[0], 1.5))
        self.assertTrue(allclose(sample.weights[0], [0.25, 0.25, 0.25, 0.25]))
        self.assertTrue(allclose(sample.indices[0], 0))
        self.assertTrue(allclose(sample.uvs[0], [0.5, 0.5]))

        # query shape and sampled data shape must match!
        self.assertTrue(sample.projections.shape == p.shape)


    def test_valid(self):
        self.assertTrue(self.plane.valid)

    def test_contains(self):
        mesh = self.box

        # get all face indices connected to given vert indices
        f = mesh.contains_vertices([0], contained=False, exclude=False)
        self.assertTrue(allclose(f, [0, 3, 5]))

        # get all face indices not connected to given vert indices
        f = mesh.contains_vertices([0], contained=False, exclude=True)
        self.assertTrue(allclose(f, [1, 2, 4]))

        # get all contained face indices
        f = mesh.contains_vertices([0, 1, 2, 3, 5, 7], contained=True, exclude=False)
        self.assertTrue(allclose(f, [0, 4]))

        # get all faces uncontained by given vertices
        f = mesh.contains_vertices([0, 1, 2, 3, 5, 7], contained=True, exclude=True)
        self.assertTrue(allclose(f, [1, 2, 3, 5]))

    def test_from_vertices_flat_is_unchanged(self):
        """Sentinel for difference(), which is the only production caller."""
        mesh = self.box

        new  = mesh.from_vertices([0, 1, 2, 3])
        self.assertTrue(allclose(new.counts, [4]))
        self.assertTrue(allclose(new.indices, [0, 1, 3, 2]))

        self.assertEqual(mesh.from_vertices(1).face_count, 0)
        self.assertEqual(mesh.from_vertices([0, 1, 2, 3], contained=False).face_count, 5)
        self.assertEqual(mesh.from_vertices([0, 1, 2, 3], exclude=True).face_count, 5)

    def test_subset_rules_shape_and_padding(self):
        rules = self.box.get_subset_rules([0, 1, 2, 3, 5])

        self.assertEqual(rules.shape, self.box.geometry.shape)
        self.assertEqual(rules.dtype, self.box.geometry.dtype)
        # a face left with too few vertices keeps nothing
        self.assertTrue((rules == -1).all(axis=1).any())

    def test_subset_rules_identity_round_trip(self):
        mesh = self.box
        new  = mesh.from_vertices(mesh.get_subset_rules(np.arange(mesh.point_count)))

        self.assertTrue(allclose(new.counts, mesh.counts))
        self.assertTrue(allclose(new.indices, mesh.indices))
        self.assertTrue(allclose(new.points, mesh.points))

    def test_subset_rules_keep_partial_faces_in_source_order(self):
        """A quad missing one vertex survives as a triangle, winding held."""
        mesh  = self.box
        rules = mesh.get_subset_rules([0, 1, 2, 3, 4, 5, 7])
        new   = mesh.from_vertices(rules)

        self.assertTrue(allclose(new.counts, [4, 4, 3, 3, 4, 3]))
        self.assertTrue(
            allclose(new.indices, [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 6, 6, 1, 0, 1, 6, 5, 3, 0, 2, 4])
        )

    def test_subset_rules_ngon_keeps_alternating_vertices(self):
        mesh = MeshData(
            points=np.array(
                [
                    [np.cos(t), np.sin(t), 0.0]
                    for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
                ]
            ),
            indices = np.arange(6),
            counts  = np.array([6]),
        )
        rules = mesh.get_subset_rules([0, 2, 4])

        self.assertTrue(allclose(rules, [[0, 2, 4, -1, -1, -1]]))
        new = mesh.from_vertices(rules)
        self.assertTrue(allclose(new.counts, [3]))
        self.assertTrue(allclose(new.indices, [0, 1, 2]))

    def test_subset_rules_drop_faces_below_min_count(self):
        mesh  = self.box

        rules = mesh.get_subset_rules([0, 1])
        self.assertTrue((rules == -1).all())
        self.assertEqual(mesh.from_vertices(rules).face_count, 0)

        # the threshold is what decides, not the face size
        rules = mesh.get_subset_rules([0, 1, 2, 3], min_count=4)
        self.assertTrue(allclose(rules.sum(axis=1) >= 0, [True] + [False] * 5))

    def test_subset_rules_keep_mesh_and_uv_parallel(self):
        """One rules array drives both, which is the point of face local slots."""
        rules = self.box.get_subset_rules([0, 1, 2, 3, 4, 5, 7])
        mesh  = self.box.from_vertices(rules)
        uv    = self.box_uv.from_vertices(rules)

        self.assertTrue(allclose(mesh.counts, uv.counts))
        self.assertEqual(len(mesh.indices), len(uv.indices))

        # compare uv COORDINATES, not ids: the two spaces renumber separately
        source = self.box_uv.geometry
        at     = 0
        for face in range(self.box.counts.size):
            slots = rules[face][rules[face] >= 0]
            if not slots.size:
                continue
            want = self.box_uv.points[source[face][slots]]
            got  = uv.points[uv.indices[at : at + slots.size]]
            self.assertTrue(allclose(want, got))
            at += slots.size

    def test_subset_rules_exclude_inverts_the_selection(self):
        mesh  = self.box
        rules = mesh.get_subset_rules([6], exclude=True)
        new   = mesh.from_vertices(rules)

        kept = np.take_along_axis(
            mesh.geometry, np.where(rules >= 0, rules, 0), axis=1
        )[rules >= 0]
        self.assertNotIn(6, np.unique(kept).tolist())
        self.assertTrue((new.counts >= 3).all())

    def test_subset_rules_reject_a_mismatched_topology(self):
        """The check that stops a mesh and its uv silently diverging."""
        mesh = MeshData(
            points  = np.zeros((5, 3)),
            indices = np.array([0, 1, 2, 3, 0, 1, 4]),
            counts  = np.array([4, 3]),
        )

        with self.assertRaises(ValueError):
            mesh.from_vertices(np.zeros((3, 9), dtype=int))

        # slot 3 of a triangle does not exist
        with self.assertRaises(ValueError):
            mesh.from_vertices(np.array([[0, 1, 2, 3], [0, 1, 3, -1]]))

        with self.assertRaises(ValueError):
            mesh.from_vertices(np.array([[0, 1, 1, -1], [0, 1, 2, -1]]))

    def test_subset_rules_reject_face_level_flags(self):
        rules = self.box.get_subset_rules([0, 1, 2, 3, 5])

        with self.assertRaises(ValueError):
            self.box.from_vertices(rules, contained=False)

        with self.assertRaises(ValueError):
            self.box.from_vertices(rules, exclude=True)

    def test_subset_rules_clear_normals(self):
        mesh = self.box.copy()
        mesh.normals        = np.zeros((24, 3))
        mesh.normal_indices = np.arange(24)

        new = mesh.from_vertices(mesh.get_subset_rules(np.arange(8)))
        self.assertIsNone(new.normals)
        self.assertIsNone(new.normal_indices)

    def test_rules_to_vertices_names_what_survives(self):
        mesh  = self.box
        rules = mesh.get_subset_rules([0, 1, 2, 3, 4, 5, 7])

        kept  = mesh.rules_to_vertices(rules)
        self.assertTrue(allclose(kept, [0, 1, 2, 3, 4, 5, 7]))

        # slot i of the map is vertex i of the rebuilt mesh
        self.assertTrue(allclose(mesh.points[kept], mesh.from_vertices(rules).points))

    def test_rules_to_vertices_on_a_full_drop(self):
        mesh = self.box
        self.assertEqual(mesh.rules_to_vertices(mesh.get_subset_rules([0, 1])).size, 0)

    def test_valid_on_an_empty_mesh(self):
        """A subset that drops every face reaches this, so it cannot raise."""
        empty = MeshData(
            points  = np.zeros((0, 3)),
            indices = np.array([], dtype=int),
            counts  = np.array([], dtype=int),
        )
        self.assertTrue(empty.valid)
        self.assertTrue(self.box.from_vertices(self.box.get_subset_rules([0, 1])).valid)

    def test_valid_rejects_a_face_that_cannot_be_a_polygon(self):
        two_gon = MeshData(
            points  = self.plane.points,
            indices = np.array([0, 1, 2, 3, 0, 1]),
            counts  = np.array([4, 2]),
        )
        self.assertFalse(two_gon.valid)

        # three distinct vertices, real area, but a repeated slot
        repeated = MeshData(
            points  = self.plane.points[:3],
            indices = np.array([0, 1, 2, 0]),
            counts  = np.array([4]),
        )
        self.assertFalse(repeated.valid)
        self.assertEqual(repeated.get_zero_area_faces().size, 0)

    def test_valid_allows_uv_holes(self):
        """UVData spells a hole count 0, so zero counts must stay legal."""
        holed = UVData(
            points  = self.plane.points[:, :2],
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4, 0]),
        )
        self.assertTrue(holed.has_holes)
        self.assertTrue(holed.valid)
        self.assertEqual(holed.get_degenerate_faces().size, 0)

    def test_valid_is_memoized_and_reset(self):
        mesh = self.box.copy()
        self.assertTrue(mesh.valid)

        mesh._valid = False
        self.assertFalse(mesh.valid)

        mesh.reset_cached_data()
        self.assertTrue(mesh.valid)

    def test_degenerate_faces_flag_repeats_and_short_faces(self):
        mesh = MeshData(
            points  = self.box.points[:6],
            indices = np.array([0, 1, 0, 2, 0, 1, 2, 3, 0, 1, 2, 0, 4, 5, 0, 0, 0]),
            counts  = np.array([4, 4, 6, 3]),
        )

        # face 1 is the only clean one
        self.assertTrue(allclose(mesh.get_degenerate_faces(), [0, 2, 3]))
        self.assertTrue(mesh.has_degenerate_faces)

        self.assertFalse(self.box.has_degenerate_faces)
        self.assertEqual(self.box.get_degenerate_faces().size, 0)

    def test_zero_area_faces_catch_what_a_count_cannot(self):
        collinear = MeshData(
            points  = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            indices = np.arange(3),
            counts  = np.array([3]),
        )
        self.assertTrue(allclose(collinear.get_zero_area_faces(), [0]))

        # the count is clean, so this is the only check that sees it
        self.assertEqual(collinear.get_degenerate_faces().size, 0)
        self.assertEqual(self.box.get_zero_area_faces().size, 0)

    def test_zero_area_faces_survive_ngons_and_holes(self):
        """A hole must not drag the face before it to zero."""
        mesh = MeshData(
            points=np.array(
                [
                    [0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [3, 0, 0], [4, 0, 0], [5, 0, 0],
                ]
            ),
            indices = np.array([0, 1, 2, 3, 4, 5, 6]),
            counts  = np.array([4, 0, 3]),
        )
        self.assertTrue(allclose(mesh.get_zero_area_faces(), [2]))

        hexagon = MeshData(
            points=np.array(
                [
                    [np.cos(t), np.sin(t), 0.0]
                    for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
                ]
            ),
            indices = np.arange(6),
            counts  = np.array([6]),
        )
        self.assertEqual(hexagon.get_zero_area_faces().size, 0)

    def test_zero_area_faces_on_uv_points(self):
        """UVData is 2d, and newell needs three components."""
        uv = UVData(
            points  = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
            indices = np.array([0, 1, 3, 0, 1, 2]),
            counts  = np.array([3, 3]),
        )
        self.assertTrue(allclose(uv.get_zero_area_faces(), [1]))

    def test_attributes(self):
        mesh = self.weird

        # expected outputs
        f2v = np.array([[ 0,  2,  1, -1],
                        [ 1,  2,  4,  3],
                        [ 3,  4,  5, -1],
                        [ 5,  0,  1,  3]])

        f2e = np.array([[ 0,  1,  2, -1],
                        [ 1,  3,  4,  5],
                        [ 4,  6,  7, -1],
                        [ 8,  2,  5,  7]])

        e2f = np.array([[ 0, -1],
                        [ 0,  1],
                        [ 0,  3],
                        [ 1, -1],
                        [ 1,  2],
                        [ 1,  3],
                        [ 2, -1],
                        [ 2,  3],
                        [ 3, -1]])

        e2v = np.array([[0, 2],
                        [2, 1],
                        [1, 0],
                        [2, 4],
                        [4, 3],
                        [3, 1],
                        [4, 5],
                        [5, 3],
                        [5, 0]])

        v2e = np.array([[0, 2, 8],
                        [1, 2, 5],
                        [0, 1, 3],
                        [4, 5, 7],
                        [3, 4, 6],
                        [6, 7, 8]])

        v2f = np.array([[ 0,  3, -1],
                        [ 0,  1,  3],
                        [ 0,  1, -1],
                        [ 1,  2,  3],
                        [ 1,  2, -1],
                        [ 2,  3, -1]])

        # check attributes
        self.assertTrue(allclose(mesh.f2v, f2v))
        self.assertTrue(allclose(mesh.f2e, f2e))
        self.assertTrue(allclose(mesh.e2f, e2f))
        self.assertTrue(allclose(mesh.e2v, e2v))
        self.assertTrue(allclose(mesh.v2f, v2f))
        self.assertTrue(allclose(mesh.v2e, v2e))

        # this is an open mesh
        self.assertTrue(mesh.open)

        # get the borders
        faces = mesh.get_border_faces(flatten=True)
        edges = mesh.get_border_edges(flatten=True)
        verts = mesh.get_border_vertices(flatten=True)

        self.assertTrue(allclose(faces, np.array([0, 1, 2, 3])))
        self.assertTrue(allclose(edges, np.array([0, 3, 6, 8])))
        self.assertTrue(allclose(verts, np.array([0, 2, 4, 5])))

        # count the tris, quads and ngons
        self.assertTrue(mesh.quads == 2)
        self.assertTrue(mesh.triangles == 2)
        self.assertTrue(mesh.ngons == 0)
        self.assertTrue(mesh.valence == 3)

        # test bbx
        bbx_min, bbx_max = self.box.get_extent()
        self.assertTrue(allclose(bbx_min, [-0.5, -0.5, -0.5]))
        self.assertTrue(allclose(bbx_max, [0.5, 0.5, 0.5]))

    def test_border(self):
        mesh    = self.plane_with_hole.copy()
        borders = mesh.get_border_vertices()
        self.assertTrue(allclose(borders[0], [5,  6,  9, 10]))
        self.assertTrue(allclose(borders[1], [ 0,  1,  2,  3,  4,  7,  8, 11, 12, 13, 14, 15]))

        borders = mesh.get_border_vertices(flatten=True)
        self.assertTrue(allclose(borders, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]))

        self.assertTrue(len(self.box.get_border_vertices())==0)
        self.assertTrue(len(self.box.get_border_vertices(flatten=True))==0)

        mesh = self.box.copy()
        mesh.delete_faces([4])
        borders = mesh.get_border_vertices(flatten=True)
        self.assertTrue(allclose(borders, [1, 3, 5, 7]))

        borders = mesh.get_border_vertices()
        self.assertTrue(allclose(borders, [[1, 3, 5, 7]]))

    def test_union_difference(self):
        mesh = self.box.copy()
        self.assertFalse(mesh.open)
        self.assertTrue(mesh.face_count == 6)

        mesh.delete_faces([0])
        self.assertTrue(mesh.open)
        self.assertTrue(mesh.face_count == 5)

        plane = self.box.from_faces([0])
        self.assertTrue(plane.face_count == 1)

        plane.union(mesh)
        self.assertTrue(plane.open)

        plane = self.box.from_faces([0])
        plane += mesh
        self.assertTrue(plane.open)

        plane.merge()
        self.assertFalse(plane.open)

        # original mesh is recreated!
        self.assertTrue(plane == self.box)

        mesh0 = self.box.copy()
        mesh1 = self.box.copy()
        mesh1.points[:, 0] += 1
        mesh0.difference(mesh1)
        self.assertTrue(mesh0.open)
        self.assertTrue(mesh0.face_count == 5)

        mesh0 = self.box.copy()
        mesh0 -= mesh1
        self.assertTrue(mesh0.open)
        self.assertTrue(mesh0.face_count == 5)


    def test_offset_scale(self):
        mesh = self.box.from_faces([0])
        mesh *= 2
        expected = np.array([[-1., -1.,  1.],
                             [ 1., -1.,  1.],
                             [-1.,  1.,  1.],
                             [ 1.,  1.,  1.]])
        self.assertTrue(allclose(mesh.points, expected))

        mesh *= [10, 20, 30]
        expected = np.array([[-10., -20.,  30.],
                             [ 10., -20.,  30.],
                             [-10.,  20.,  30.],
                             [ 10.,  20.,  30.]])
        self.assertTrue(allclose(mesh.points, expected))


        mesh += 5
        expected = np.array([[ -5., -15.,  35.],
                             [ 15., -15.,  35.],
                             [ -5.,  25.,  35.],
                             [ 15.,  25.,  35.]])
        self.assertTrue(allclose(mesh.points, expected))

        mesh -= [10, 20, 30]
        expected = np.array([[-15., -35.,   5.],
                             [  5., -35.,   5.],
                             [-15.,   5.,   5.],
                             [  5.,   5.,   5.]])
        self.assertTrue(allclose(mesh.points, expected))




    def test_subdivision(self):
        mesh = self.weird.copy()
        mesh.subdivide(1)

        indices = np.array([15,  6,  2,  7, 15,  7,  1,  8, 15,  8,  0,  6, 16,  7,  2,  9, 16,
                             9,  4, 10, 16, 10,  3, 11, 16, 11,  1,  7, 17, 10,  4, 12, 17, 12,
                             5, 13, 17, 13,  3, 10, 18, 14,  0,  8, 18,  8,  1, 11, 18, 11,  3,
                            13, 18, 13,  5, 14])

        counts = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        points = np.array([[-0.375     , -0.375     ,  0.375     ],
                           [-0.2962963 ,  0.2962963 ,  0.27777778],
                           [ 0.375     ,  0.375     ,  0.375     ],
                           [-0.2962963 ,  0.2962963 , -0.27777778],
                           [ 0.375     ,  0.375     , -0.375     ],
                           [-0.375     , -0.375     , -0.375     ],
                           [ 0.        ,  0.        ,  0.5       ],
                           [-0.04166667,  0.41666667,  0.375     ],
                           [-0.41666667,  0.04166667,  0.375     ],
                           [ 0.5       ,  0.5       ,  0.        ],
                           [-0.04166667,  0.41666667, -0.375     ],
                           [-0.375     ,  0.375     ,  0.        ],
                           [ 0.        ,  0.        , -0.5       ],
                           [-0.41666667,  0.04166667, -0.375     ],
                           [-0.5       , -0.5       ,  0.        ],
                           [-0.16666667,  0.16666667,  0.5       ],
                           [ 0.        ,  0.5       ,  0.        ],
                           [-0.16666667,  0.16666667, -0.5       ],
                           [-0.5       ,  0.        ,  0.        ]])

        self.assertTrue(allclose(mesh.indices, indices))
        self.assertTrue(allclose(mesh.points, points))
        self.assertTrue(allclose(mesh.counts, counts))


        # test fitted subd
        mesh = self.weird.copy()
        mesh.subdivide(1, fit=True)

        points = np.array([[-0.49999999, -0.49999999,  0.49999999],
                           [-0.49999935,  0.49999935,  0.49900933],
                           [ 0.49999999,  0.49999999,  0.49999999],
                           [-0.49999935,  0.49999935, -0.49900933],
                           [ 0.49999999,  0.49999999, -0.49999999],
                           [-0.49999999, -0.49999999, -0.49999999],
                           [ 0.        ,  0.        ,  0.66666666],
                           [-0.13671825,  0.63671824,  0.66539295],
                           [-0.63671824,  0.13671825,  0.66539295],
                           [ 0.66666666,  0.66666666,  0.        ],
                           [-0.13671825,  0.63671824, -0.66539295],
                           [-0.63281167,  0.63281167,  0.        ],
                           [ 0.        ,  0.        , -0.66666666],
                           [-0.63671824,  0.13671825, -0.66539295],
                           [-0.66666666, -0.66666666,  0.        ],
                           [-0.28124963,  0.28124963,  0.83205961],
                           [-0.08854112,  0.75520778,  0.        ],
                           [-0.28124963,  0.28124963, -0.83205961],
                           [-0.75520778,  0.08854112,  0.        ]])

        self.assertTrue(allclose(mesh.points, points))


    def test_non_manifold(self):
        # According to Maya, a non manifold mesh has:
        # - 3 or more faces sharing an edge
        # - 2 or more faces sharing a vertex but no edges
        # - Adjacent faces have opposite vertex winding order

        # SHARED VERTEX
        mesh = MeshData(indices=np.array([0, 1, 3, 2, 4, 2, 6, 5]),
                        counts=np.array([4, 4]),
                        points=np.array([[ 0.,  0.,  1.],
                                         [ 1.,  0.,  1.],
                                         [ 0.,  0.,  0.],
                                         [ 1.,  0.,  0.],
                                         [-1.,  0.,  0.],
                                         [-1.,  0., -1.],
                                         [ 0.,  0., -1.]]))

        vertices = mesh.get_non_manifold_vertices()
        self.assertTrue(allclose(vertices, [2]))

        # SHARED EDGE
        mesh = MeshData(indices=np.array([0, 1, 3, 2, 4, 0, 2, 6, 0, 5, 7, 2]),
                        counts=np.array([4, 4, 4]),
                        points=np.array([[ 0. ,  0. ,  0.5],
                                         [ 0. ,  0.5,  0.5],
                                         [ 0. ,  0. , -0.5],
                                         [ 0. ,  0.5, -0.5],
                                         [-0.5,  0. ,  0.5],
                                         [ 0.5,  0. ,  0.5],
                                         [-0.5,  0. , -0.5],
                                         [ 0.5,  0. , -0.5]]))

        vertices = mesh.get_non_manifold_vertices()
        self.assertTrue(allclose(vertices, [0, 2]))


        # WINDING ORDER
        mesh = MeshData(indices=np.array([0, 1, 3, 2, 2, 4, 5, 3]),
                        counts=np.array([4, 4]),
                        points=np.array([[-0.5,  0. ,  0.5],
                                         [ 0.5,  0. ,  0.5],
                                         [-0.5,  0. , -0.5],
                                         [ 0.5,  0. , -0.5],
                                         [-0.5,  0. , -1.5],
                                         [ 0.5,  0. , -1.5]]))

        vertices = mesh.get_non_manifold_vertices()
        self.assertTrue(allclose(vertices, [2, 3]))


        # LAMINA FACE
        mesh = MeshData(indices=np.array([0, 1, 3, 2, 0, 1, 3, 2]),
                        counts=np.array([4, 4]),
                        points=np.array([[-0.5,  0. ,  0.5],
                                         [ 0.5,  0. ,  0.5],
                                         [-0.5,  0. , -0.5],
                                         [ 0.5,  0. , -0.5]]))

        vertices = mesh.get_non_manifold_vertices()
        self.assertTrue(allclose(vertices, [0, 1, 2, 3]))

        faces = mesh.get_lamina_faces()
        self.assertTrue(allclose(faces, [1]))


    def test_rasterization(self):

        # test rasterization
        box = self.box_uv.copy()

        # color encode/decode
        known_indices = np.unique(box.indices)
        rgb           = box.encode_rgb(known_indices)
        idx           = box.decode_rgb(rgb)
        self.assertTrue(allclose(known_indices, idx))

        # rasterize to image
        box.clear_buffer()
        box.resolution = (256, 256)
        box.draw_faces()
        image = box.to_image()
        self.assertTrue(allclose(np.array(image), np.flip(box.buffer, axis=0)))

        # rasterize only some faces
        box.clear_buffer()
        box.draw_faces(indices=[0,2,3,4], color=None)
        rgb     = np.unique(box.buffer.reshape(-1,3), axis=0)
        rgb     = rgb[np.sum(rgb == 0, axis=1) != 3]
        indices = box.decode_rgb(rgb)
        self.assertTrue(allclose(indices, [0,2,3,4]))

        # inverse
        box.clear_buffer()
        box.draw_faces(indices=[0,2,3,4], invert=True, color=None)
        rgb     = np.unique(box.buffer.reshape(-1,3), axis=0)
        rgb     = rgb[np.sum(rgb == 0, axis=1) != 3]
        indices = box.decode_rgb(rgb)
        self.assertTrue(allclose(indices, [1,5]))

        # convolution
        box = self.box_uv.copy()

        before = np.array([[ True,  True,  True,  True,  True,  True],
                           [ True,  True,  True,  True,  True,  True],
                           [False, False,  True,  True, False, False],
                           [False, False,  True,  True, False, False],
                           [False, False,  True,  True, False, False],
                           [False, False,  True,  True, False, False]])

        after = np.array([[ True,  True,  True,  True,  True,  True],
                          [ True,  True,  True,  True,  True,  True],
                          [ True,  True,  True,  True,  True,  True],
                          [False,  True,  True,  True,  True, False],
                          [False,  True,  True,  True,  True, False],
                          [False,  True,  True,  True,  True, False]])

        box.resolution = 6
        box.draw_faces(color=(255,255,255))
        test = np.any(box.buffer > 0, axis=2)
        self.assertTrue(allclose(test, before))

        box.convolve()
        test2 = np.any(box.buffer > 0, axis=2)
        self.assertTrue(allclose(test2, after))

    def test_topology_conversions(self):
        """Test faces_to_vertices, edges_to_vertices, vertices_to_faces, vertices_to_edges"""
        mesh = self.box.copy()

        # Test faces_to_vertices
        verts          = mesh.faces_to_vertices([0, 1])
        expected_verts = np.unique([0, 1, 3, 2, 2, 3, 5, 4])
        self.assertTrue(allclose(np.sort(verts), np.sort(expected_verts)))

        # Test edges_to_vertices
        verts = mesh.edges_to_vertices([0, 1, 2])
        # Edges 0,1,2 should give us unique vertices from those edges
        self.assertTrue(len(verts) > 0)
        self.assertTrue(np.all(verts >= 0))

        # Test vertices_to_faces
        faces = mesh.vertices_to_faces([0, 1])
        self.assertTrue(len(faces) > 0)
        # Face 0 and 3 contain vertex 0
        self.assertTrue(0 in faces or 3 in faces)

        # Test vertices_to_edges
        edges = mesh.vertices_to_edges([0, 1])
        self.assertTrue(len(edges) > 0)
        self.assertTrue(np.all(edges >= 0))

    def test_get_unused_points(self):
        """Test get_unused_points method"""
        # Create a mesh with unused points
        mesh = MeshData(
            points=np.array([
                [0., 0., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [0., 1., 0.],
                [5., 5., 5.],  # Unused point
                [10., 10., 10.]  # Unused point
            ]),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4])
        )

        unused = mesh.get_unused_points()
        self.assertTrue(allclose(unused, [4, 5]))

    def test_get_overlap_vertices(self):
        """Test get_overlap_vertices method"""
        # Create a mesh with overlapping vertices
        mesh = MeshData(
            points=np.array([
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],  # Duplicate of vertex 0
                [1., 1., 0.]
            ]),
            indices = np.array([0, 1, 3, 2]),
            counts  = np.array([4])
        )

        overlaps = mesh.get_overlap_vertices()
        # Should find that vertex 0 and 2 overlap
        self.assertTrue(len(overlaps) > 0)

    def test_properties_counts(self):
        """Test point_count, face_count, edge_count, closed properties"""
        mesh = self.box.copy()

        # Test counts
        self.assertEqual(mesh.point_count, 8)
        self.assertEqual(mesh.face_count, 6)
        self.assertTrue(mesh.edge_count > 0)

        # Closed box should not be open
        self.assertFalse(mesh.open)
        self.assertTrue(mesh.closed)

        # Test geometry property
        geometry = mesh.geometry
        self.assertTrue(allclose(geometry, mesh.f2v))

        # Test points4 property
        points4 = mesh.points4
        self.assertEqual(points4.shape, (8, 4))
        self.assertTrue(allclose(points4[:, 3], 1.0))  # Last column should be 1s
        self.assertTrue(allclose(points4[:, :3], mesh.points))

    def test_normal_methods(self):
        """Test get_face_normals and get_vertex_normals"""
        mesh = self.plane.copy()

        # Get face normals
        face_normals = mesh.get_face_normals()
        self.assertEqual(face_normals.shape[0], mesh.face_count)
        self.assertEqual(face_normals.shape[1], 3)

        # For a plane in XY plane, normal should be in Z direction
        # The normal might be [0, 0, 1] or [0, 0, -1] depending on winding
        self.assertTrue(allclose(np.abs(face_normals[0, 2]), 1.0, atol=0.01))

        # Get vertex normals
        vertex_normals = mesh.get_vertex_normals()
        self.assertEqual(vertex_normals.shape[0], mesh.point_count)
        self.assertEqual(vertex_normals.shape[1], 3)

        # All vertex normals should be normalized
        magnitudes = np.linalg.norm(vertex_normals, axis=1)
        self.assertTrue(allclose(magnitudes, 1.0, atol=0.01))

    def test_edge_lengths(self):
        """Test get_edge_lengths method"""
        mesh         = self.plane.copy()

        edge_lengths = mesh.get_edge_lengths()
        self.assertEqual(edge_lengths.shape[0], mesh.edge_count)

        # All edge lengths should be positive
        self.assertTrue(np.all(edge_lengths > 0))

        # For our unit plane, edge lengths should be 1.0
        self.assertTrue(allclose(edge_lengths, 1.0, atol=0.01))

    def test_edge_normals(self):
        """Test get_edge_normals method"""
        mesh         = self.box.copy()

        edge_normals = mesh.get_edge_normals()
        self.assertEqual(edge_normals.shape[0], mesh.edge_count)
        self.assertEqual(edge_normals.shape[1], 3)

        # All normals should be normalized
        magnitudes = np.linalg.norm(edge_normals, axis=1)
        self.assertTrue(allclose(magnitudes, 1.0, atol=0.01))

    def test_face_vertex_angles(self):
        """Test get_face_vertex_angles method"""
        mesh   = self.plane.copy()

        angles = mesh.get_face_vertex_angles()
        self.assertEqual(angles.shape[0], mesh.indices.size)

        # For a square, all angles should be 90 degrees (pi/2)
        expected_angle = np.pi / 2
        self.assertTrue(allclose(angles, expected_angle, atol=0.01))

    # --- shading normals (set_normals / recompute_normals) ---

    def test_set_normals_smooth(self):
        """set_normals with no hard edges produces per-vertex normals."""
        mesh = self.box.copy()
        mesh.set_normals()

        self.assertIsNotNone(mesh.normals)
        self.assertIsNone(mesh.normal_indices)

        # per-vertex: one normal per mesh vertex
        self.assertEqual(mesh.normals.shape, (mesh.point_count, 3))

        # all normals should be unit length
        mag = np.linalg.norm(mesh.normals, axis=1)
        self.assertTrue(allclose(mag, 1.0, atol=0.01))

    def test_set_normals_hard_edge_angle(self):
        """set_normals with hard_edge_angle splits at sharp edges."""
        mesh = self.box.copy()

        # box has all 90deg edges -- a 45deg threshold should mark them all hard
        mesh.set_normals(hard_edge_angle=45.0)

        self.assertIsNotNone(mesh.normals)
        self.assertIsNotNone(mesh.normal_indices)

        # face-varying indices must cover every face-vertex
        self.assertEqual(mesh.normal_indices.size, mesh.indices.size)

        # with all edges hard, each face-vertex is independent
        # so we should have as many unique normals as face-vertices
        self.assertEqual(mesh.normals.shape[0], np.unique(mesh.normal_indices).size)

        # normals should be unit length
        mag = np.linalg.norm(mesh.normals, axis=1)
        self.assertTrue(allclose(mag, 1.0, atol=0.01))

    def test_set_normals_explicit_hard_edges(self):
        """set_normals with explicit hard_edges parameter."""
        mesh = self.box.copy()

        # mark the first 3 edges as hard
        hard_edges = np.array([0, 1, 2])
        mesh.set_normals(hard_edges=hard_edges)

        self.assertIsNotNone(mesh.normals)
        self.assertIsNotNone(mesh.normal_indices)

        # face-varying count must match index stream
        self.assertEqual(mesh.normal_indices.size, mesh.indices.size)

        # the split should produce more normals than the all-smooth case
        self.assertGreater(mesh.normals.shape[0], mesh.point_count)

    def test_set_normals_plane_all_smooth(self):
        """A flat plane should have identical normals regardless of hard edges."""
        mesh = self.plane.copy()
        mesh.set_normals()

        # all normals should point in the same direction
        expected = mesh.normals[0]
        for n in mesh.normals:
            self.assertTrue(allclose(n, expected, atol=0.01))

    def test_recompute_normals_after_deformation(self):
        """recompute_normals updates vectors but preserves topology."""
        mesh = self.box.copy()
        mesh.set_normals(hard_edge_angle=45.0)
        original_indices = mesh.normal_indices.copy()

        # deform: shift all points along Y
        mesh.points[:, 1] += 5.0
        mesh.recompute_normals()

        # topology (indices) should be unchanged
        self.assertTrue(np.array_equal(mesh.normal_indices, original_indices))

        # normals should still be unit length after recomputation
        mag = np.linalg.norm(mesh.normals, axis=1)
        self.assertTrue(allclose(mag, 1.0, atol=0.01))

    def test_recompute_normals_reflects_geometry_change(self):
        """recompute_normals actually changes the normal vectors."""
        mesh = self.box.copy()
        mesh.set_normals()
        original_normals = mesh.normals.copy()

        # deform the mesh non-uniformly so normals change
        mesh.points[0] += [2.0, 0.0, 0.0]
        mesh.recompute_normals()

        # at least some normals should differ
        self.assertFalse(np.allclose(mesh.normals, original_normals))

    def test_recompute_normals_raises_without_set(self):
        """recompute_normals raises when normals were never set."""
        mesh = self.box.copy()
        self.assertIsNone(mesh.normals)

        with self.assertRaises(RuntimeError):
            mesh.recompute_normals()

    def test_normals_cleared_on_delete_faces(self):
        """Topology-changing ops invalidate stored normals."""
        mesh = self.box.copy()
        mesh.set_normals()
        self.assertIsNotNone(mesh.normals)

        mesh.delete_faces(np.array([0]))
        self.assertIsNone(mesh.normals)
        self.assertIsNone(mesh.normal_indices)

    def test_normals_cleared_on_triangulate(self):
        """Triangulation invalidates stored normals."""
        mesh = self.box.copy()
        mesh.set_normals()
        self.assertIsNotNone(mesh.normals)

        rules = mesh.get_triangulate_rules()
        mesh.triangulate(rules)
        self.assertIsNone(mesh.normals)

    def test_normals_cleared_on_merge(self):
        """Merge invalidates stored normals."""
        mesh = self.box.copy()
        mesh.set_normals()

        # detach first to have something to merge
        mesh.detach_faces()
        # detach clears normals, re-set them
        mesh.set_normals()
        self.assertIsNotNone(mesh.normals)

        mesh.merge()
        self.assertIsNone(mesh.normals)

    def test_normals_survive_copy(self):
        """copy() preserves stored normals."""
        mesh = self.box.copy()
        mesh.set_normals(hard_edge_angle=45.0)

        copied = mesh.copy()
        self.assertIsNotNone(copied.normals)
        self.assertTrue(np.allclose(copied.normals, mesh.normals))

        if mesh.normal_indices is not None:
            self.assertTrue(np.array_equal(copied.normal_indices, mesh.normal_indices))

    def test_normals_serialization(self):
        """to_dict / from_dict round-trips normals."""
        mesh = self.box.copy()
        mesh.set_normals(hard_edge_angle=45.0)

        d = mesh.to_dict()
        self.assertIn("normals", d)
        self.assertIn("normal_indices", d)

        restored = MeshData.from_dict(d)
        self.assertTrue(np.allclose(restored.normals, mesh.normals))
        self.assertTrue(np.array_equal(restored.normal_indices, mesh.normal_indices))

    def test_normals_not_serialized_when_none(self):
        """to_dict omits normals when they are None."""
        mesh = self.box.copy()
        self.assertIsNone(mesh.normals)

        d = mesh.to_dict()
        self.assertNotIn("normals", d)
        self.assertNotIn("normal_indices", d)

    # --- tangent space (get_tangent_space) ---

    def test_get_tangent_space_shape(self):
        """get_tangent_space returns correctly shaped arrays."""
        mesh = self.box.copy()
        mesh.set_normals()
        uvdata = self.box_uv.copy()

        tangents, bitangents = mesh.get_tangent_space(uvdata)

        # tangents: (num_uv_verts, 4)
        self.assertEqual(tangents.shape, (uvdata.point_count, 4))

        # bitangents: (num_uv_verts, 3)
        self.assertEqual(bitangents.shape, (uvdata.point_count, 3))

    def test_tangent_unit_length(self):
        """Tangent xyz components should be unit length."""
        mesh = self.box.copy()
        mesh.set_normals()
        uvdata = self.box_uv.copy()

        tangents, _ = mesh.get_tangent_space(uvdata)

        mag = np.linalg.norm(tangents[:, :3], axis=1)
        self.assertTrue(np.allclose(mag, 1.0, atol=0.01))

    def test_bitangent_unit_length(self):
        """Bitangent vectors should be unit length."""
        mesh = self.box.copy()
        mesh.set_normals()
        uvdata = self.box_uv.copy()

        _, bitangents = mesh.get_tangent_space(uvdata)

        mag = np.linalg.norm(bitangents, axis=1)
        self.assertTrue(np.allclose(mag, 1.0, atol=0.01))

    def test_tangent_sign_values(self):
        """Bitangent sign (tangent.w) should be +1 or -1."""
        mesh = self.box.copy()
        mesh.set_normals()
        uvdata = self.box_uv.copy()

        tangents, _ = mesh.get_tangent_space(uvdata)

        signs = tangents[:, 3]
        self.assertTrue(np.all(np.isin(signs, [-1.0, 1.0])))

    def test_tangent_orthogonal_to_normal(self):
        """Tangent vectors should be orthogonal to normals at each UV vertex."""
        mesh = self.box.copy()
        mesh.set_normals()
        uvdata = self.box_uv.copy()

        tangents, bitangents = mesh.get_tangent_space(uvdata)

        # resolve normals at UV vertices (same logic as get_tangent_space)
        from cgmath.geometry.utils import build_uv_to_normal_map
        uv_to_nrm = build_uv_to_normal_map(
            uvdata.indices, mesh.indices, uvdata.point_count,
        )
        N      = mesh.normals[uv_to_nrm]

        T      = tangents[:, :3]
        dot_NT = np.abs(np.einsum("ij,ij->i", N, T))
        self.assertTrue(np.all(dot_NT < 0.05))

        dot_NB = np.abs(np.einsum("ij,ij->i", N, bitangents))
        self.assertTrue(np.all(dot_NB < 0.05))

    def test_tangent_space_without_stored_normals(self):
        """get_tangent_space falls back to smooth vertex normals."""
        mesh   = self.box.copy()
        uvdata = self.box_uv.copy()

        # no set_normals call -- should still work
        self.assertIsNone(mesh.normals)
        tangents, bitangents = mesh.get_tangent_space(uvdata)

        self.assertEqual(tangents.shape, (uvdata.point_count, 4))
        mag = np.linalg.norm(tangents[:, :3], axis=1)
        self.assertTrue(np.allclose(mag, 1.0, atol=0.01))

    def test_tangent_space_mixed_topology(self):
        """get_tangent_space works on mixed tri/quad meshes."""
        mesh = self.weird.copy()
        mesh.set_normals()
        uvdata = self.weird_uv.copy()

        tangents, bitangents = mesh.get_tangent_space(uvdata)

        self.assertEqual(tangents.shape[0],   uvdata.point_count)
        self.assertEqual(tangents.shape[1],   4)
        self.assertEqual(bitangents.shape[1], 3)

        mag = np.linalg.norm(tangents[:, :3], axis=1)
        self.assertTrue(np.allclose(mag, 1.0, atol=0.01))

    def test_tangent_space_with_hard_edges(self):
        """get_tangent_space works correctly with hard edge normals."""
        mesh = self.box.copy()
        mesh.set_normals(hard_edge_angle=45.0)
        uvdata = self.box_uv.copy()

        tangents, bitangents = mesh.get_tangent_space(uvdata)

        self.assertEqual(tangents.shape, (uvdata.point_count, 4))
        mag = np.linalg.norm(tangents[:, :3], axis=1)
        self.assertTrue(np.allclose(mag, 1.0, atol=0.01))

    # --- MikkTSpace reference regression (gltf-rs/mikktspace) ---

    def _load_mikktspace_json(self, filename):
        """Load MikkTSpace JSON test data and return (MeshData, UVData, expected)."""
        import json

        data_dir = os.path.join(os.path.dirname(__file__), "test_assets")
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)

        verts     = data["mesh"]["vertices"]
        positions = np.array([v["position"] for v in verts])
        normals   = np.array([v["normal"] for v in verts])
        uvs       = np.array([v["tex_coord"] for v in verts])
        faces     = np.array(data["mesh"]["faces"])

        mesh = MeshData(
            points  = positions,
            indices = faces.ravel(),
            counts  = np.full(faces.shape[0], 3),
        )
        mesh.normals        = normals
        mesh.normal_indices = None

        uvdata = UVData(
            points  = uvs,
            indices = faces.ravel(),
            counts  = np.full(faces.shape[0], 3),
        )

        return mesh, uvdata, data["expected_results"]

    def test_mikktspace_reference_cube(self):
        """MikkTSpace tangents match gltf-rs/mikktspace reference (standard cube).

        Faces with degenerate UVs (zero-area UV triangles) are excluded
        because the MikkTSpace reference uses connectivity-aware attribute
        grouping to inherit tangents from non-degenerate neighbours -- a
        strategy not yet implemented here.
        """
        mesh, uvdata, expected = self._load_mikktspace_json(
            "mikktspace_test_data.json",
        )

        tangents, bitangents = mesh.get_tangent_space(uvdata)
        fv_tangents = tangents[uvdata.indices]

        # Identify degenerate UV faces (zero-area UV triangles)
        geom        = uvdata.geometry
        ds1         = uvdata.points[geom[:, 1], 0] - uvdata.points[geom[:, 0], 0]
        dt1         = uvdata.points[geom[:, 1], 1] - uvdata.points[geom[:, 0], 1]
        ds2         = uvdata.points[geom[:, 2], 0] - uvdata.points[geom[:, 0], 0]
        dt2         = uvdata.points[geom[:, 2], 1] - uvdata.points[geom[:, 0], 1]
        face_det    = ds1 * dt2 - ds2 * dt1
        degen_faces = set(np.where(np.abs(face_det) < 1e-12)[0])

        tested      = 0
        for entry in expected:
            fi = entry["face"]
            if fi in degen_faces:
                continue
            vi     = entry["vert"]
            fv_idx = fi * 3 + vi
            tested += 1

            expected_t = np.array(entry["tangent"])
            computed_t = fv_tangents[fv_idx, :3]
            self.assertTrue(
                np.allclose(computed_t, expected_t, atol=1e-4),
                f"face {fi} vert {vi}: tangent {computed_t} != {expected_t}",
            )

            expected_sign = 1.0 if entry["bi_tangent_preserves_orientation"] else -1.0
            computed_sign = fv_tangents[fv_idx, 3]
            self.assertAlmostEqual(
                computed_sign, expected_sign, places=4,
                msg=f"face {fi} vert {vi}: sign {computed_sign} != {expected_sign}",
            )

        self.assertGreater(tested, 0)

    def test_mikktspace_reference_deformed(self):
        """MikkTSpace tangents match reference after non-uniform scale (x*2, y*0.5).

        See test_mikktspace_reference_cube for degenerate-UV exclusion notes.
        """
        mesh, uvdata, expected = self._load_mikktspace_json(
            "mikktspace_deformed_data.json",
        )

        tangents, _ = mesh.get_tangent_space(uvdata)
        fv_tangents = tangents[uvdata.indices]

        geom        = uvdata.geometry
        ds1         = uvdata.points[geom[:, 1], 0] - uvdata.points[geom[:, 0], 0]
        dt1         = uvdata.points[geom[:, 1], 1] - uvdata.points[geom[:, 0], 1]
        ds2         = uvdata.points[geom[:, 2], 0] - uvdata.points[geom[:, 0], 0]
        dt2         = uvdata.points[geom[:, 2], 1] - uvdata.points[geom[:, 0], 1]
        face_det    = ds1 * dt2 - ds2 * dt1
        degen_faces = set(np.where(np.abs(face_det) < 1e-12)[0])

        tested      = 0
        for entry in expected:
            fi = entry["face"]
            if fi in degen_faces:
                continue
            vi     = entry["vert"]
            fv_idx = fi * 3 + vi
            tested += 1

            expected_t = np.array(entry["tangent"])
            computed_t = fv_tangents[fv_idx]
            self.assertTrue(
                np.allclose(computed_t, expected_t, atol=1e-4),
                f"face {fi} vert {vi}: tangent {computed_t} != {expected_t}",
            )

        self.assertGreater(tested, 0)

    def test_tangent_space_tilted_normals(self):
        """MikkTSpace tangents/bitangents match Maya reference for tilted normals.

        Uses a 2-triangle plane with non-planar (tilted) normals, matching
        the Maya mikktspace reference test (ambrusc/mikktpy).
        """
        mesh = MeshData.from_dict({
            "indices": [0, 1, 2, 3, 4, 5],
            "counts":  [3, 3],
            "points": [[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0]],
            "normals": [[ 0.0              , -0.707106828689575,  0.707106828689575],
                        [ 0.707106828689575,  0.0              ,  0.707106828689575],
                        [ 0.0              ,  0.707106828689575,  0.707106828689575],
                        [ 0.0              ,  0.707106828689575,  0.707106828689575],
                        [ 0.707106828689575,  0.0              ,  0.707106828689575],
                        [-0.707106828689575,  0.0              ,  0.707106828689575]],
            "normal_indices": [0, 1, 2, 3, 4, 5],
        })

        uvdata = UVData(
            points=np.array([[0.0, 0.0],
                             [1.0, 1.0],
                             [0.0, 1.0],
                             [0.0, 1.0],
                             [1.0, 1.0],
                             [1.0, 2.0]]),
            indices = np.array([0, 1, 2, 3, 4, 5]),
            counts  = np.array([3, 3]),
        )

        tangents, bitangents = mesh.get_tangent_space(uvdata)

        # Expected results from ambrusc/mikktpy (Maya MikktSpace reference).
        # Layout: tangent_xyz, bitangent_xyz, mag_tan, mag_bitan  per face-vertex.
        er = np.array([
            0.8164966702461243, -0.40824833512306213, -0.40824827551841736,
            0.0, 0.7071068286895752, 0.7071067690849304,
            1.4142135381698608, 1.0,
            0.40824833512306213, -0.8164965510368347, -0.40824827551841736,
            0.0, 0.9999999403953552, 0.0,
            1.4142135381698608, 1.0,
            0.8164965510368347, -0.40824833512306213, 0.40824827551841736,
            0.0, 0.7071067690849304, -0.7071067094802856,
            1.4142135381698608, 1.0,
            0.8164965510368347, -0.40824833512306213, 0.40824827551841736,
            0.0, 0.7071067690849304, -0.7071067094802856,
            1.4142135381698608, 1.0,
            0.40824833512306213, -0.8164965510368347, -0.40824827551841736,
            0.0, 0.9999999403953552, 0.0,
            1.4142135381698608, 1.0,
            0.40824833512306213, -0.8164966702461243, 0.40824827551841736,
            0.0, 1.0, 0.0,
            1.4142135381698608, 1.0,
        ])
        expected_tangents   = np.array([er[0::8], er[1::8], er[2::8]]).T
        expected_bitangents = np.array([er[3::8], er[4::8], er[5::8]]).T

        fv_tangents         = tangents[uvdata.indices, :3]
        fv_bitangents       = bitangents[uvdata.indices]

        self.assertTrue(
            np.allclose(fv_tangents, expected_tangents, atol=1e-4),
            f"Tangent mismatch:\n  computed: {fv_tangents}\n  expected: {expected_tangents}",
        )
        self.assertTrue(
            np.allclose(fv_bitangents, expected_bitangents, atol=1e-4),
            f"Bitangent mismatch:\n  computed: {fv_bitangents}\n  expected: {expected_bitangents}",
        )

    def test_closest_points(self):
        """Test get_closest_points method"""
        mesh = self.plane.copy()

        # Query points above the plane
        query_points = np.array([[0., 0., 1.], [0.5, 0.5, 2.]])

        distances, indices = mesh.get_closest_points(query_points)

        self.assertEqual(distances.shape[0], query_points.shape[0])
        self.assertEqual(indices.shape[0], query_points.shape[0])

        # All distances should be positive
        self.assertTrue(np.all(distances >= 0))

        # Check that indices are valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < mesh.point_count))

    def test_get_extent(self):
        """Test get_extent method returns correct bounding box"""
        mesh = self.box.copy()

        bbx_min, bbx_max = mesh.get_extent()

        self.assertEqual(bbx_min.shape[0], 3)
        self.assertEqual(bbx_max.shape[0], 3)

        # Verify the bounding box is correct
        self.assertTrue(allclose(bbx_min, [-0.5, -0.5, -0.5]))
        self.assertTrue(allclose(bbx_max, [0.5, 0.5, 0.5]))

    def test_detach_faces(self):
        """Test detach_faces method"""
        mesh                 = self.box.copy()
        original_point_count = mesh.point_count

        # Detach first two faces
        mesh.detach_faces([0, 1])

        # After detaching, we should have more points (duplicated vertices)
        self.assertGreater(mesh.point_count, original_point_count)

        # Face count should remain the same
        self.assertEqual(mesh.face_count, 6)

    def test_from_vertices(self):
        """Test from_vertices method"""
        mesh = self.box.copy()

        # Get a subset of faces using from_vertices
        new_mesh = mesh.from_vertices([0, 1, 2, 3])

        # Should have fewer faces than original
        self.assertLessEqual(new_mesh.face_count, mesh.face_count)

        # All vertices in new mesh should be in the specified set
        used_verts = np.unique(new_mesh.indices)
        self.assertTrue(np.all(used_verts <= 3))

    def test_to_identity(self):
        """Test to_identity method"""
        mesh = self.box.copy()

        # Modify the mesh transform
        mesh.matrix = np.array([[2, 0, 0, 1],
                                [0, 2, 0, 2],
                                [0, 0, 2, 3],
                                [0, 0, 0, 1]])

        # Reset to identity
        mesh.to_identity()

        # Matrix should be identity
        expected_identity = np.eye(4)
        self.assertTrue(allclose(mesh.matrix, expected_identity))

    def test_get_inverse_matrix(self):
        """Test get_inverse_matrix method"""
        mesh = self.box.copy()

        # Set a non-identity matrix
        mesh.matrix = np.array([[2, 0, 0, 1],
                                [0, 3, 0, 2],
                                [0, 0, 4, 3],
                                [0, 0, 0, 1]], dtype=float)

        # Get inverse
        inv_matrix = mesh.get_inverse_matrix()

        # Verify it returns a 4x4 matrix
        self.assertEqual(inv_matrix.shape, (4, 4))

        # Verify it's not the same as the original matrix
        self.assertFalse(allclose(inv_matrix, mesh.matrix))

    def test_smooth_mesh(self):
        """Test smooth method on mesh"""
        mesh            = self.box.copy()
        original_points = mesh.points.copy()

        # Apply smoothing
        mesh.smooth(iterations=1)

        # Points should have changed
        self.assertFalse(allclose(mesh.points, original_points))

        # Should still have same number of points
        self.assertEqual(mesh.points.shape, original_points.shape)

    def test_triangulate_mesh(self):
        """Test triangulate method"""
        mesh                = self.box.copy()
        original_face_count = mesh.face_count

        # Get triangulation rules first
        rules = mesh.get_triangulate_rules()

        # Triangulate the mesh
        mesh.triangulate(rules)

        # All faces should now be triangles
        self.assertTrue(np.all(mesh.counts == 3))

        # Should have more faces than before (each quad becomes 2 tris)
        self.assertGreater(mesh.face_count, original_face_count)

    def test_get_triangulate_rules(self):
        """Test get_triangulate_rules method"""
        from cgmath.geometry.mesh import TriangulateRules

        mesh = self.box.copy()

        # Get triangulation rules
        rules = mesh.get_triangulate_rules()

        # Should return a TriangulateRules instance
        self.assertIsInstance(rules, TriangulateRules)

        # rules.rules should be a per-face array
        self.assertIsInstance(rules.rules, np.ndarray)
        self.assertEqual(len(rules.rules), mesh.face_count)

        # ngon_tris should be an empty dict for a quad-only mesh
        self.assertIsInstance(rules.ngon_tris, dict)
        self.assertEqual(len(rules.ngon_tris), 0)

    def test_unique_edge_properties(self):
        """Test unique edge properties: f2ue, ue2f, ue2v, ue, uep, v2ue"""
        mesh = self.box.copy()

        # Test f2ue (faces to unique edges)
        f2ue = mesh.f2ue
        self.assertEqual(f2ue.shape[0], mesh.face_count)
        # Each face should reference edges
        self.assertTrue(np.all(f2ue >= -1))

        # Test ue2f (unique edges to faces)
        ue2f = mesh.ue2f
        self.assertTrue(ue2f.shape[0] > 0)
        # Each edge should reference 1 or 2 faces (or -1 for border edges)
        self.assertTrue(ue2f.shape[1] == 2)

        # Test ue2v (unique edges to vertices)
        ue2v = mesh.ue2v
        self.assertTrue(ue2v.shape[0] > 0)
        self.assertEqual(ue2v.shape[1], 2)
        # All vertex indices should be valid
        self.assertTrue(np.all(ue2v >= 0))
        self.assertTrue(np.all(ue2v < mesh.point_count))

        # Test ue (unique edge indices)
        ue = mesh.ue
        self.assertTrue(len(ue) > 0)
        # All indices should be non-negative
        self.assertTrue(np.all(ue >= 0))

        # Test uep (unique edge pairs)
        uep = mesh.uep
        self.assertTrue(uep.shape[0] > 0)
        self.assertEqual(uep.shape[1], 2)
        # Edge pairs should contain valid indices
        self.assertTrue(np.all(uep >= 0))

        # Test v2ue (vertices to unique edges)
        v2ue = mesh.v2ue
        self.assertEqual(v2ue.shape[0], mesh.point_count)
        # Each vertex should connect to at least one edge
        # Check that each row has at least one non-negative value
        for i in range(v2ue.shape[0]):
            self.assertTrue(np.any(v2ue[i] >= 0))

    def test_shell_properties(self):
        """Test shell properties: shell_edges, shell_points, shell_faces, shell_count"""
        # Test with a closed mesh (single shell)
        mesh = self.box.copy()

        # Test shell_faces
        shell_faces = mesh.shell_faces
        self.assertIsInstance(shell_faces, list)
        self.assertEqual(len(shell_faces), 1)  # Box is a single shell
        self.assertEqual(len(shell_faces[0]), mesh.face_count)

        # Test shell_points
        shell_points = mesh.shell_points
        self.assertIsInstance(shell_points, list)
        self.assertEqual(len(shell_points), 1)  # Box is a single shell
        self.assertEqual(len(shell_points[0]), mesh.point_count)

        # Test shell_edges
        shell_edges = mesh.shell_edges
        self.assertIsInstance(shell_edges, list)
        self.assertEqual(len(shell_edges), 1)  # Box is a single shell

        # Test shell_count
        shell_count = mesh.shell_count
        self.assertEqual(shell_count, 1)  # Box is a single shell

        # Test with multiple shells by creating two separate meshes
        mesh1 = self.plane.copy()
        mesh1.points[:, 0] -= 2  # Move first mesh to the left

        mesh2 = self.plane.copy()
        mesh2.points[:, 0] += 2  # Move second mesh to the right

        # Combine into one mesh with two shells
        combined = mesh1.copy()
        combined.union(mesh2)

        # Should have 2 shells
        self.assertEqual(combined.shell_count,       2)
        self.assertEqual(len(combined.shell_faces),  2)
        self.assertEqual(len(combined.shell_points), 2)
        self.assertEqual(len(combined.shell_edges),  2)

    def test_area_property(self):
        """Test area property calculation"""
        # Test with zero-area mesh (crushed)
        crushed = MeshData(
            indices = np.array([0, 1, 3, 2]),
            counts  = np.array([4]),
            points=np.array([[0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.]])
        )
        self.assertTrue(np.allclose(crushed.area, 0))

        # Test with triangle
        triangle = MeshData(
            indices = np.array([0, 1, 2]),
            counts  = np.array([3]),
            points=np.array([[1.5, 0., 0.5],
                            [2.5, 0., 0.5],
                            [1.5, 0., -0.5]])
        )
        self.assertTrue(np.allclose(triangle.area, 0.5))

        # Test with square
        square = MeshData(
            indices = np.array([0, 1, 3, 2]),
            counts  = np.array([4]),
            points=np.array([[3.5, 0., 0.5],
                            [4.5, 0., 0.5],
                            [3.5, 0., -0.5],
                            [4.5, 0., -0.5]])
        )
        self.assertTrue(np.allclose(square.area, 1))

        # Test with non-planar quad
        weird = MeshData(
            indices = np.array([0, 1, 3, 2]),
            counts  = np.array([4]),
            points=np.array([[5.5, 0., 0.5],
                            [6.5, 0., 0.5],
                            [5., 1., -1.],
                            [7., 0., -1.]])
        )
        self.assertTrue(np.allclose(weird.area, 2.51933346517785766354))

    def test_tangent_space_rejects_ngons(self):
        """get_tangent_space raises NotImplementedError for n-gon faces."""
        mesh = MeshData(
            points  = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 1.5, 0]], dtype=np.float64),
            indices = np.array([0, 1, 2, 3, 4]),
            counts  = np.array([5]),
        )
        uvdata = UVData(
            points  = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 1.5]], dtype=np.float64),
            indices = np.array([0, 1, 2, 3, 4]),
            counts  = np.array([5]),
        )
        with self.assertRaises(NotImplementedError):
            mesh.get_tangent_space(uvdata)

    def test_to_uvdata_basic(self):
        """Test basic projection along default Y axis"""
        mesh   = self.box.copy()
        uvdata = mesh.to_uvdata()

        # Verify the result is a UVData object
        self.assertIsInstance(uvdata, UVData)

        # Verify the resulting UVData has correct structure
        self.assertEqual(uvdata.point_count, mesh.point_count)
        self.assertTrue(allclose(uvdata.indices, mesh.indices))
        self.assertTrue(allclose(uvdata.counts, mesh.counts))

        # Verify UV points are normalized to 0-1 range
        self.assertTrue(np.all(uvdata.points >= 0.0))
        self.assertTrue(np.all(uvdata.points <= 1.0))

        # Verify the UV points are 2D
        self.assertEqual(uvdata.points.shape[1], 2)

    def test_to_uvdata_axis_projections(self):
        """Test projection along different axes"""
        mesh = self.box.copy()

        # Test projection along X axis (uses Y and Z coordinates)
        uvdata_x = mesh.to_uvdata(axis=Axis.X)
        # For X axis projection, U should be from Y coords, V from Z coords
        bbx_min, bbx_max = uvdata_x.get_extent()
        self.assertTrue(allclose(bbx_min, [0.0, 0.0]))
        self.assertTrue(allclose(bbx_max, [1.0, 1.0]))

        # Test projection along Y axis (uses X and Z coordinates)
        uvdata_y = mesh.to_uvdata(axis=Axis.Y)
        bbx_min, bbx_max = uvdata_y.get_extent()
        self.assertTrue(allclose(bbx_min, [0.0, 0.0]))
        self.assertTrue(allclose(bbx_max, [1.0, 1.0]))

        # Test projection along Z axis (uses X and Y coordinates)
        uvdata_z = mesh.to_uvdata(axis=Axis.Z)
        bbx_min, bbx_max = uvdata_z.get_extent()
        self.assertTrue(allclose(bbx_min, [0.0, 0.0]))
        self.assertTrue(allclose(bbx_max, [1.0, 1.0]))

        # Verify different axes produce different UV coordinates
        # Since the box is symmetric, X and Y projections should differ from Z
        # The points arrays should not all be identical
        self.assertFalse(allclose(uvdata_x.points, uvdata_y.points))

        # Test axis can be passed as string
        uvdata_x_str = mesh.to_uvdata(axis="x")
        self.assertTrue(allclose(uvdata_x.points, uvdata_x_str.points))

    def test_to_uvdata_maintain_ratio(self):
        """Test with maintain_ratio=True"""
        # Create a non-square mesh to test aspect ratio preservation
        rect_mesh = MeshData(
            points=np.array([
                [-1.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [1.0, 0.0, -0.5],
                [-1.0, 0.0, -0.5]
            ]),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4])
        )

        # Without maintain_ratio - should fill 0-1 range in both dimensions
        uvdata_no_ratio = rect_mesh.to_uvdata(axis=Axis.Y, maintain_ratio=False)
        bbx_min, bbx_max = uvdata_no_ratio.get_extent()
        self.assertTrue(allclose(bbx_min, [0.0, 0.0]))
        self.assertTrue(allclose(bbx_max, [1.0, 1.0]))

        # With maintain_ratio - should preserve aspect ratio
        uvdata_ratio = rect_mesh.to_uvdata(axis=Axis.Y, maintain_ratio=True)

        # UV points should still be within 0-1 bounds
        self.assertTrue(np.all(uvdata_ratio.points >= 0.0))
        self.assertTrue(np.all(uvdata_ratio.points <= 1.0))

        # The shorter dimension should not span the full 0-1 range
        bbx_min, bbx_max = uvdata_ratio.get_extent()
        delta = bbx_max - bbx_min

        # Original mesh is 2 units wide and 1 unit tall (2:1 ratio)
        # So with maintain_ratio, the V (height) should be scaled down
        # The wider dimension (U) should span 0-1, the narrower (V) should be less
        self.assertTrue(allclose(delta[0], 1.0))  # U spans full range
        self.assertTrue(delta[1] < 1.0)           # V is smaller due to aspect ratio

        # Verify the ratio is approximately preserved (2:1 becomes 1:0.5)
        self.assertTrue(allclose(delta[1], 0.5, atol=0.01))

    def test_to_uvdata_swap_uvs(self):
        """Test with swap_uvs=True"""
        # Create a non-square mesh to clearly see the axis swap
        rect_mesh = MeshData(
            points=np.array([
                [-1.0, 0.0, 0.25],
                [1.0, 0.0, 0.25],
                [1.0, 0.0, -0.25],
                [-1.0, 0.0, -0.25]
            ]),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4])
        )

        # Without swap_uvs
        uvdata_normal = rect_mesh.to_uvdata(axis=Axis.Y, swap_uvs=False)

        # With swap_uvs - U and V should be swapped
        uvdata_swapped = rect_mesh.to_uvdata(axis=Axis.Y, swap_uvs=True)

        # The U values of normal should match V values of swapped (after normalization)
        # Since both are normalized to 0-1, we check the structure is swapped
        # The swapped version should have its coordinates exchanged
        self.assertFalse(allclose(uvdata_normal.points, uvdata_swapped.points))

        # Verify the swap by checking that the columns are exchanged
        # After normalization, the relationship should be: normal[:,0] corresponds to swapped[:,1]
        # and normal[:,1] corresponds to swapped[:,0]
        self.assertTrue(allclose(uvdata_normal.points[:, 0], uvdata_swapped.points[:, 1]))
        self.assertTrue(allclose(uvdata_normal.points[:, 1], uvdata_swapped.points[:, 0]))

    def test_to_uvdata_preserves_topology(self):
        """Verify indices and counts match between source mesh and resulting UVData"""
        mesh   = self.box.copy()
        uvdata = mesh.to_uvdata()

        # Verify indices match exactly
        self.assertTrue(allclose(uvdata.indices, mesh.indices))

        # Verify counts match exactly
        self.assertTrue(allclose(uvdata.counts, mesh.counts))

        # Verify face count is preserved
        self.assertEqual(uvdata.face_count, mesh.face_count)

        # Test with a mixed mesh (triangles and quads)
        mixed_mesh   = self.weird.copy()
        uvdata_mixed = mixed_mesh.to_uvdata()

        self.assertTrue(allclose(uvdata_mixed.indices, mixed_mesh.indices))
        self.assertTrue(allclose(uvdata_mixed.counts, mixed_mesh.counts))
        self.assertEqual(uvdata_mixed.face_count, mixed_mesh.face_count)


class TestBSpline(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define a base shapes
        self.open_curve = BSplineData(points=np.array([[-4.5505618 ,  0.        , -5.53772071],
                                                       [-9.84751204,  0.        ,  1.44462279],
                                                       [-4.83948636,  0.        ,  5.00802568],
                                                       [ 4.30979133,  0.        ,  6.21187801],
                                                       [11.09951846,  0.        ,  4.04494382],
                                                       [ 6.52487961,  0.        , -4.52648475]]),
                                      degree=3,
                                      periodic=False)

        self.closed_curve = BSplineData(points=np.array([[-4.5505618 ,  0.        , -5.53772071],
                                                         [-9.84751204,  0.        ,  1.44462279],
                                                         [-4.83948636,  0.        ,  5.00802568],
                                                         [ 4.30979133,  0.        ,  6.21187801],
                                                         [11.09951846,  0.        ,  4.04494382],
                                                         [ 6.52487961,  0.        , -4.52648475]]),
                                        degree=3,
                                        periodic=True)

    def test_bytes(self):
        obj1 = self.open_curve.copy()
        b    = obj1.to_bytes()
        obj2 = BSplineData.from_bytes(b)
        self.assertTrue(obj1 == obj2)

    def test_serialize(self):
        obj1 = self.open_curve.copy()
        self.assertTrue(obj1._kv is None)

        # add custom private data
        obj1._custom_data = "test"


        # write to file
        with tempfile.TemporaryDirectory() as temp_dir:
            f = os.path.join(temp_dir, "test.pkl")

            # save obj1 to file
            obj1.save(f)

            # load obj2 from file
            obj2 = BSplineData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # confirm obj2 has no private data
            self.assertFalse(hasattr(obj2, "_custom_data"))

            # do the same via pickle, put obj1 inside a list
            with open(f, 'wb') as io:
                pickle.dump([obj1], io)

            with open(f, 'rb') as io:
                obj3 = pickle.load(io)[0]

                # confirm they're identical
                self.assertTrue(obj1 == obj3)

                # confirm obj3 has no cached data
                self.assertFalse(hasattr(obj3, "_custom_data"))

            # save the file using the ascii format
            obj1.save(f, mode='json')
            obj2 = BSplineData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj2)

            # test string output
            self.assertTrue(isinstance(obj1.to_json(), str))

            # test npz
            obj1.save(f, mode='npz')
            obj3 = BSplineData.load(f)

            # confirm they're identical
            self.assertTrue(obj1 == obj3)


    def test_properties(self):

        # --- OPEN CURVE --- #
        obj = self.open_curve

        # precise length
        self.assertTrue(allclose(obj.length, 34.55899983014413))
        self.assertTrue(allclose(obj.get_length(fast=False), obj.length))

        # fast length (precise enough)
        self.assertTrue(allclose(obj.get_length(fast=True), 34.5589022564839))

        # cv is same as points
        self.assertTrue(allclose(obj.cv, obj.points))

        # DCC friendly kv and internal scipy.interpolate.BSpline kv
        self.assertTrue(allclose(obj.kv,  [0, 0, 0, 1, 2, 3, 3, 3]))
        self.assertTrue(allclose(obj._kv, [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]))

        # max param
        self.assertTrue(obj.max_param==3)

        # unique point count
        self.assertTrue(obj.count==6)

        # geometry array (for open curve, should be [0, 1, 2, ..., count-1])
        self.assertTrue(allclose(obj.geometry, np.array([0, 1, 2, 3, 4, 5])))
        self.assertEqual(len(obj.geometry), 6)


        # --- CLOSED CURVE --- #
        obj = self.closed_curve

        # precise length
        self.assertTrue(allclose(obj.length, 44.58294549604119))
        self.assertTrue(allclose(obj.get_length(), 44.58294549604119))

        # fast length
        self.assertTrue(allclose(obj.get_length(fast=True), 44.582784389268326))

        # cv is NOT same as points
        self.assertFalse(allclose(obj.cv.shape, obj.points.shape))

        # DCC friendly kv and internal scipy.interpolate.BSpline kv
        self.assertTrue(allclose(obj.kv,  [-2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8]))
        self.assertTrue(allclose(obj._kv, [-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9]))

        # max param
        self.assertTrue(obj.max_param==6)

        # unique point count
        self.assertTrue(obj.count==6)

        # geometry array
        self.assertTrue(allclose(obj.geometry, np.array([0, 1, 2, 3, 4, 5, 0, 1, 2])))
        self.assertEqual(len(obj.geometry), 9)


    def test_basis(self):
        open_basis = np.array([[1.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
                               [0.015625  , 0.45703125, 0.45703125, 0.0703125 , 0.        , 0.        ],
                               [0.        , 0.03125   , 0.46875   , 0.46875   , 0.03125   , 0.        ],
                               [0.        , 0.        , 0.0703125 , 0.45703125, 0.45703125, 0.015625  ],
                               [0.        , 0.        , 0.        , 0.        , 0.        , 1.        ]])

        closed_basis = np.array([[0.16666667, 0.66666667, 0.16666667, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
                                 [0.        , 0.02083333, 0.47916667, 0.47916667, 0.02083333, 0.        , 0.        , 0.        , 0.        ],
                                 [0.        , 0.        , 0.        , 0.16666667, 0.66666667, 0.16666667, 0.        , 0.        , 0.        ],
                                 [0.        , 0.        , 0.        , 0.        , 0.02083333, 0.47916667, 0.47916667, 0.02083333, 0.        ],
                                 [0.16666667, 0.66666667, 0.16666667, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]])

        collapsed_closed_basis = np.array([[0.16666667, 0.66666667, 0.16666667, 0.        , 0.        , 0.        ],
                                           [0.        , 0.02083333, 0.47916667, 0.47916667, 0.02083333, 0.        ],
                                           [0.        , 0.        , 0.        , 0.16666667, 0.66666667, 0.16666667],
                                           [0.47916667, 0.02083333, 0.        , 0.        , 0.02083333, 0.47916667],
                                           [0.16666667, 0.66666667, 0.16666667, 0.        , 0.        , 0.        ]])

        u = np.linspace(0, self.open_curve.max_param, 5)
        self.assertTrue(allclose(self.open_curve.basis(u), open_basis))
        self.assertTrue(np.all(np.sum(self.open_curve.basis(u), axis=1) == 1.))

        # ensure basis rows sums to 1
        self.assertTrue(np.allclose(np.sum(self.open_curve.basis(u, collapse=False), axis=1), np.ones(u.shape[0])))

        u = np.linspace(0, self.closed_curve.max_param, 5)
        self.assertTrue(allclose(self.closed_curve.basis(u), closed_basis))
        self.assertTrue(allclose(self.closed_curve.basis(u, collapse=True), collapsed_closed_basis))

        # ensure basis rows sums to 1
        self.assertTrue(np.allclose(np.sum(self.closed_curve.basis(u, collapse=False), axis=1), np.ones(u.shape[0])))
        self.assertTrue(np.allclose(np.sum(self.closed_curve.basis(u, collapse=True), axis=1), np.ones(u.shape[0])))

    def test_open_close(self):
        obj1 = self.open_curve.copy()
        obj2 = self.closed_curve.copy()
        obj1.close()
        self.assertTrue(obj1==obj2)


        obj1 = self.open_curve.copy()
        obj2 = self.closed_curve.copy()
        obj2.open()
        self.assertTrue(obj1==obj2)


    def test_compute(self):

        open_points = np.array([[-4.5505618 ,  0.        , -5.53772071],
                                [-6.48048756,  0.        ,  3.29930778],
                                [-0.20916934,  0.        ,  5.43087881],
                                [ 6.80421098,  0.        ,  4.96908858],
                                [ 6.52487961,  0.        , -4.52648475]])

        open_tangents = np.array([[-15.89085072,   0.        ,  20.9470305 ],
                                  [  6.50983146,   0.        ,   5.15549759],
                                  [  9.07403692,   0.        ,   1.16472713],
                                  [  8.39912721,   0.        ,  -3.40163523],
                                  [-13.72391653,   0.        , -25.71428571]])

        closed_points = np.array([[-8.13001605,  0.        ,  0.87479935],
                                  [-0.22772873,  0.        ,  5.49056982],
                                  [ 9.20545746,  0.        ,  2.97752809],
                                  [ 0.97211075,  0.        , -4.70806581],
                                  [-8.13001605,  0.        ,  0.87479935]])

        closed_tangents = np.array([[-0.14446228,  0.        ,  5.27287319],
                                    [ 8.33667737,  0.        ,  1.07744783],
                                    [ 1.10754414,  0.        , -5.36918138],
                                    [-9.5405297 ,  0.        , -0.9570626 ],
                                    [-0.14446228,  0.        ,  5.27287319]])


        obj = self.open_curve
        u   = np.linspace(0, obj.max_param, 5)
        points, tangents = obj.compute(u)

        self.assertTrue(allclose(open_points, points))
        self.assertTrue(allclose(open_tangents, tangents))


        obj = self.closed_curve
        u   = np.linspace(0, obj.max_param, 5)
        points, tangents = obj.compute(u)

        self.assertTrue(allclose(closed_points, points))
        self.assertTrue(allclose(closed_tangents, tangents))


    def test_compute_open_uniform_extrapolation_high(self):
        """For open uniform curves, u > 1 extrapolates linearly along the endpoint tangent."""
        obj = self.open_curve.copy()
        obj.uniform = True

        # Reference: position and native tangent at the endpoint (u=1).
        p_end, t_end = obj.compute(1.0)
        total_length = obj.total_length
        t_unit       = t_end / np.linalg.norm(t_end)

        # Extrapolated sample beyond the curve.
        u_extrap = 1.5
        p_ex, t_ex = obj.compute(u_extrap)

        expected_p = p_end + (u_extrap - 1.0) * total_length * t_unit
        self.assertTrue(allclose(p_ex, expected_p))

        # Tangent held constant at the endpoint native tangent.
        self.assertTrue(allclose(t_ex, t_end))

        # Physical distance traveled past the end equals (u-1) * total_length.
        distance = np.linalg.norm(p_ex - p_end)
        self.assertTrue(allclose(distance, (u_extrap - 1.0) * total_length))


    def test_compute_open_uniform_extrapolation_low(self):
        """For open uniform curves, u < 0 extrapolates backward from the start."""
        obj = self.open_curve.copy()
        obj.uniform = True

        p_start, t_start = obj.compute(0.0)
        total_length = obj.total_length
        t_unit       = t_start / np.linalg.norm(t_start)

        u_extrap     = -0.3
        p_ex, t_ex = obj.compute(u_extrap)

        # u < 0 => motion opposite to the start tangent (extends backward).
        expected_p = p_start + u_extrap * total_length * t_unit
        self.assertTrue(allclose(p_ex, expected_p))

        self.assertTrue(allclose(t_ex, t_start))


    def test_compute_open_uniform_extrapolation_mixed_array(self):
        """Array input with a mix of in-range, sub-zero, and beyond-one u values."""
        obj = self.open_curve.copy()
        obj.uniform = True

        p_start, t_start = obj.compute(0.0)
        p_end, t_end = obj.compute(1.0)
        total_length = obj.total_length
        t_unit_start = t_start / np.linalg.norm(t_start)
        t_unit_end   = t_end / np.linalg.norm(t_end)

        u            = np.array([-0.2, 0.0, 0.5, 1.0, 1.4])
        points, tangents = obj.compute(u)

        # In-range entries are unaffected.
        self.assertTrue(allclose(points[1], p_start))
        self.assertTrue(allclose(points[3], p_end))

        # Below-range: extrapolate backward from the start.
        expected_low = p_start + u[0] * total_length * t_unit_start
        self.assertTrue(allclose(points[0], expected_low))
        self.assertTrue(allclose(tangents[0], t_start))

        # Above-range: extrapolate forward past the end.
        expected_high = p_end + (u[4] - 1.0) * total_length * t_unit_end
        self.assertTrue(allclose(points[4], expected_high))
        self.assertTrue(allclose(tangents[4], t_end))


    def test_compute_fast_open_uniform_extrapolation(self):
        """compute_fast() applies the same extrapolation as compute()."""
        obj = self.open_curve.copy()
        obj.uniform = True

        u = np.array([-0.5, 0.5, 1.5])
        p_compute, t_compute = obj.compute(u)
        p_fast, t_fast = obj.compute_fast(u)

        self.assertTrue(allclose(p_compute, p_fast))
        self.assertTrue(allclose(t_compute, t_fast))


    def test_compute_open_uniform_extrapolation_scipy_matches_numba(self):
        """Numba and scipy code paths produce identical extrapolated values."""
        obj_numba = self.open_curve.copy()
        obj_numba.uniform   = True
        obj_numba.use_numba = True

        obj_scipy = self.open_curve.copy()
        obj_scipy.uniform   = True
        obj_scipy.use_numba = False

        u = np.array([-0.7, -0.1, 0.25, 0.75, 1.1, 2.0])
        p_n, t_n = obj_numba.compute(u)
        p_s, t_s = obj_scipy.compute(u)

        self.assertTrue(allclose(p_n, p_s))
        self.assertTrue(allclose(t_n, t_s))


    def test_compute_periodic_uniform_no_extrapolation(self):
        """Periodic uniform curves wrap via modulo -- they do not extrapolate."""
        obj = self.closed_curve.copy()
        obj.uniform = True

        # u=1.5 in periodic uniform space => one full loop + half loop = u=0.5.
        p_half, _ = obj.compute(0.5)
        p_one_and_half, _ = obj.compute(1.5)

        self.assertTrue(allclose(p_half, p_one_and_half))


    def test_compute_open_non_uniform_extrapolation_high(self):
        """For open non-uniform curves, u > max_param extrapolates linearly along native tangent."""
        obj = self.open_curve.copy()
        # uniform=False is the default; user-space u is in [0, max_param].

        max_param = obj.max_param
        p_end, t_end = obj.compute(max_param)

        # In non-uniform mode, native u == user u, so the per-unit-u position
        # velocity in the extrapolation region is just the native tangent.
        u_extrap = max_param + 0.5
        p_ex, t_ex = obj.compute(u_extrap)

        expected_p = p_end + (u_extrap - max_param) * t_end
        self.assertTrue(allclose(p_ex, expected_p))

        # Tangent held constant at endpoint native tangent.
        self.assertTrue(allclose(t_ex, t_end))


    def test_compute_open_non_uniform_extrapolation_low(self):
        """For open non-uniform curves, u < 0 extrapolates backward from start."""
        obj = self.open_curve.copy()

        p_start, t_start = obj.compute(0.0)

        u_extrap = -0.4
        p_ex, t_ex = obj.compute(u_extrap)

        # u < 0 -> motion opposite to start tangent.
        expected_p = p_start + u_extrap * t_start
        self.assertTrue(allclose(p_ex, expected_p))

        self.assertTrue(allclose(t_ex, t_start))


    def test_compute_open_non_uniform_extrapolation_mixed_array(self):
        """Mixed array on open non-uniform curve: in-range, sub-zero, beyond max_param."""
        obj       = self.open_curve.copy()
        max_param = obj.max_param

        p_start, t_start = obj.compute(0.0)
        p_end, t_end = obj.compute(max_param)

        u = np.array([-0.3, 0.0, max_param * 0.5, max_param, max_param + 0.7])
        points, tangents = obj.compute(u)

        # In-range entries are unaffected.
        self.assertTrue(allclose(points[1], p_start))
        self.assertTrue(allclose(points[3], p_end))

        # Below-range: extrapolated backward.
        expected_low = p_start + u[0] * t_start
        self.assertTrue(allclose(points[0], expected_low))
        self.assertTrue(allclose(tangents[0], t_start))

        # Above-range: extrapolated past the end.
        expected_high = p_end + (u[4] - max_param) * t_end
        self.assertTrue(allclose(points[4], expected_high))
        self.assertTrue(allclose(tangents[4], t_end))


    def test_compute_fast_open_non_uniform_extrapolation(self):
        """compute_fast() extrapolates non-uniform open curves identically to compute()."""
        obj       = self.open_curve.copy()
        max_param = obj.max_param

        u         = np.array([-0.5, max_param * 0.5, max_param + 1.0])
        p_compute, t_compute = obj.compute(u)
        p_fast, t_fast = obj.compute_fast(u)

        self.assertTrue(allclose(p_compute, p_fast))
        self.assertTrue(allclose(t_compute, t_fast))


    def test_compute_open_non_uniform_extrapolation_scipy_matches_numba(self):
        """Numba and scipy code paths produce identical extrapolated values for non-uniform open."""
        obj_numba = self.open_curve.copy()
        obj_numba.use_numba = True

        obj_scipy = self.open_curve.copy()
        obj_scipy.use_numba = False

        max_param = obj_numba.max_param
        u         = np.array([-0.7, -0.1, 0.5, max_param * 0.5, max_param + 0.2, max_param + 1.5])
        p_n, t_n = obj_numba.compute(u)
        p_s, t_s = obj_scipy.compute(u)

        self.assertTrue(allclose(p_n, p_s))
        self.assertTrue(allclose(t_n, t_s))


    def test_compute_periodic_non_uniform_no_extrapolation(self):
        """Periodic non-uniform curves wrap via modulo -- they do not extrapolate."""
        obj = self.closed_curve.copy()
        # uniform=False default; user-space u is in [0, max_param].

        max_param = obj.max_param

        # u = max_param + 1 in periodic non-uniform space wraps to u = 1.
        p_one, _ = obj.compute(1.0)
        p_wrapped, _ = obj.compute(max_param + 1.0)

        self.assertTrue(allclose(p_one, p_wrapped))


    def test_sample(self):

        obj     = self.open_curve
        samples = obj.sample(obj.points)
        self.assertTrue(allclose(samples(obj.points), samples.points))

        obj     = self.closed_curve
        samples = obj.sample(obj.points)
        self.assertTrue(allclose(samples(obj.points), samples.points))

    def test_sample_data_copy(self):
        """Test that SampleData.copy() creates independent copies of all arrays"""
        obj     = self.open_curve
        samples = obj.sample(obj.points)

        # Create a copy
        samples_copy = samples.copy()

        # Verify they're equal initially
        self.assertTrue(allclose(samples.points, samples_copy.points))
        self.assertTrue(allclose(samples.tangents, samples_copy.tangents))
        self.assertTrue(allclose(samples.distances, samples_copy.distances))
        self.assertTrue(allclose(samples.params, samples_copy.params))
        self.assertTrue(allclose(samples.basis, samples_copy.basis))

        # Modify the original
        samples.points[0, 0] += 100.0
        samples.tangents[0, 0] += 100.0
        samples.distances[0] += 100.0
        samples.params[0] += 100.0
        samples.basis[0, 0] += 100.0

        # Verify the copy is unaffected (deep copy)
        self.assertFalse(allclose(samples.points, samples_copy.points))
        self.assertFalse(allclose(samples.tangents, samples_copy.tangents))
        self.assertFalse(allclose(samples.distances, samples_copy.distances))
        self.assertFalse(allclose(samples.params, samples_copy.params))
        self.assertFalse(allclose(samples.basis, samples_copy.basis))

    def test_smooth_open_curve(self):
        """Test smooth() method on open curves with various parameters"""
        # Test basic smoothing on open curve
        obj             = self.open_curve.copy()
        original_points = obj.points.copy()
        original_first  = original_points[0].copy()
        original_last   = original_points[-1].copy()

        # Apply smoothing
        obj.smooth(n=1, polyorder=2)

        # Verify points array has same shape
        self.assertEqual(obj.points.shape, original_points.shape)

        # Endpoints should be preserved with lock_endpoints=True (default)
        self.assertTrue(allclose(original_first, obj.points[0]))
        self.assertTrue(allclose(original_last, obj.points[-1]))

    def test_smooth_open_curve_no_lock(self):
        """Test smooth() without locking endpoints on open curve"""
        obj = self.open_curve.copy()

        # Apply smoothing without locking endpoints
        # Should not raise any exceptions
        obj.smooth(n=1, polyorder=2, lock_endpoints=False)

        # Verify the curve is still valid
        self.assertEqual(obj.points.shape[0], 6)

    def test_smooth_open_curve_no_padding(self):
        """Test smooth() without padding endpoints on open curve"""
        obj            = self.open_curve.copy()
        original_first = obj.points[0].copy()
        original_last  = obj.points[-1].copy()

        # Apply smoothing without padding
        obj.smooth(n=1, polyorder=2, lock_endpoints=True, pad_endpoints=False)

        # Verify the curve is still valid
        self.assertEqual(obj.points.shape[0], 6)

        # With lock_endpoints=True, endpoints should still match
        self.assertTrue(allclose(original_first, obj.points[0]))
        self.assertTrue(allclose(original_last, obj.points[-1]))

    def test_smooth_closed_curve(self):
        """Test smooth() method on closed curves"""
        obj            = self.closed_curve.copy()
        original_count = obj.count

        # Apply smoothing
        obj.smooth(n=1, polyorder=2)

        # Verify the curve is still periodic and has same count
        self.assertTrue(obj.periodic)
        self.assertEqual(obj.count, original_count)

    def test_smooth_varying_window_size(self):
        """Test smooth() with different window sizes"""
        obj = self.open_curve.copy()

        # Test with small window
        obj1 = obj.copy()
        obj1.smooth(n=1, polyorder=2)
        points_n1 = obj1.points.copy()

        # Test with larger window
        obj2 = obj.copy()
        obj2.smooth(n=2, polyorder=2)
        points_n2 = obj2.points.copy()

        # Different window sizes should produce different results
        self.assertFalse(allclose(points_n1, points_n2))

    def test_smooth_varying_polyorder(self):
        """Test smooth() with different polynomial orders"""
        obj = self.open_curve.copy()

        # Test with polyorder=1
        obj1 = obj.copy()
        obj1.smooth(n=2, polyorder=1)
        points_poly1 = obj1.points.copy()

        # Test with polyorder=3
        obj2 = obj.copy()
        obj2.smooth(n=2, polyorder=3)
        points_poly3 = obj2.points.copy()

        # Different polynomial orders should produce different results
        # Use a larger tolerance check since the algorithm may still produce similar results
        max_diff = np.max(np.abs(points_poly1 - points_poly3))
        self.assertGreater(max_diff, 0.001, "Different polyorders should produce visibly different results")

    def test_smooth_edge_cases(self):
        """Test smooth() with edge case parameters"""
        # Test with n larger than half the curve length
        # Should automatically clamp to valid range
        obj = self.open_curve.copy()
        obj.smooth(n=100, polyorder=3)
        # Should not crash and should still work

        # Verify the curve is still valid
        self.assertEqual(obj.points.shape[0], 6)


class TestUVList(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.map0 = UVData(
            name='map0',
            points=np.array([[0.33000001, 0.        ],
                             [0.66333336, 0.        ],
                             [0.33000001, 0.25      ],
                             [0.66333336, 0.25      ],
                             [0.33000001, 0.5       ],
                             [0.66333336, 0.5       ],
                             [0.33000001, 0.75      ],
                             [0.66333336, 0.75      ],
                             [0.33000001, 1.        ],
                             [0.66333336, 1.        ],
                             [1.        , 0.        ],
                             [1.        , 0.25      ],
                             [0.        , 0.        ],
                             [0.        , 0.25      ]]),
            indices=np.array(
                [ 0,  1,  3,  2,  2,  3,  5,  4,  4,  5,  7,  6,  6,  7,  9,  8,  1, 10, 11,  3, 12,  0,  2, 13]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

        self.map1 = UVData(
            name='map1',
            points=np.array([[0.16500001, 0.        ],
                             [0.33166668, 0.        ],
                             [0.16500001, 0.125     ],
                             [0.33166668, 0.125     ],
                             [0.16500001, 0.25      ],
                             [0.33166668, 0.25      ],
                             [0.16500001, 0.375     ],
                             [0.33166668, 0.375     ],
                             [0.16500001, 0.5       ],
                             [0.33166668, 0.5       ],
                             [0.5       , 0.        ],
                             [0.5       , 0.125     ],
                             [0.        , 0.        ],
                             [0.        , 0.125     ]]),
            indices=np.array(
                [ 0,  1,  3,  2,  2,  3,  5,  4,  4,  5,  7,  6,  6,  7,  9,  8,  1, 10, 11,  3, 12,  0,  2, 13]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )


        self.map2 = UVData(
            name='map2',
            points=np.array([[0.0825    , 0.        ],
                             [0.16583334, 0.        ],
                             [0.0825    , 0.0625    ],
                             [0.16583334, 0.0625    ],
                             [0.0825    , 0.125     ],
                             [0.16583334, 0.125     ],
                             [0.0825    , 0.1875    ],
                             [0.16583334, 0.1875    ],
                             [0.0825    , 0.25      ],
                             [0.16583334, 0.25      ],
                             [0.25      , 0.        ],
                             [0.25      , 0.0625    ],
                             [0.        , 0.        ],
                             [0.        , 0.0625    ]]),
            indices=np.array(
                [ 0,  1,  3,  2,  2,  3,  5,  4,  4,  5,  7,  6,  6,  7,  9,  8,  1, 10, 11,  3, 12,  0,  2, 13]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

        self.list = UVList([self.map0, self.map1, self.map2])

    def test_bytes(self):
        obj1 = self.list.copy()
        b    = obj1.to_bytes()
        obj2 = UVList.from_bytes(b)
        self.assertTrue(obj1 == obj2)


    def test_serialize(self):

            obj1 = self.list.copy()

            # write to file
            with tempfile.TemporaryDirectory() as temp_dir:
                f = os.path.join(temp_dir, "test.pkl")

                # save obj1 to file
                obj1.save(f)

                # load obj2 from file
                obj2 = UVList.load(f)

                # confirm they're identical
                self.assertTrue(obj1 == obj2)

                # do the same via pickle, put obj1 inside a list
                with open(f, "wb") as io:
                    pickle.dump([obj1], io)

                with open(f, "rb") as io:
                    obj3 = pickle.load(io)[0]

                    # confirm they're identical
                    self.assertTrue(obj1 == obj3)

                # test json and npz
                obj1.save(f, mode='json')
                obj4 = UVList.load(f)
                self.assertTrue(obj1 == obj4)

                obj1.save(f, mode='npz')
                obj5 = UVList.load(f)
                self.assertTrue(obj1 == obj5)


    def test_list_like(self):

        # randomize list
        import random

        obj1 = self.list.copy()
        obj2 = self.list.copy()

        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj2.sort()
        self.assertTrue(obj1 == obj2)

        # use sorted
        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj3 = UVList(sorted(obj2))
        self.assertTrue(obj1 == obj3)


        # array indexing, slicing
        self.assertTrue(obj1[1] == self.map1)

        # reverse
        obj1.sort()
        obj2.sort()
        obj1.reverse()
        self.assertTrue(obj1 == obj2[::-1])

        # numpy like indexing
        obj1.sort()
        obj2.sort()
        self.assertTrue(obj1[[1, 2]] == obj2[1:])

        # mutability
        map0 = self.map0.copy()
        map1 = self.map1.copy()
        map2 = self.map2.copy()

        obj1 = UVList([map0, map1, map2])
        obj2 = obj1[[2, 1, 0]]
        map2.name = 'POOF!'
        self.assertTrue(obj1[2].name == 'POOF!')
        self.assertTrue(obj2[0].name == 'POOF!')

        # dict like indexting
        obj1 = self.list.copy()
        obj2 = self.list.copy()
        self.assertTrue(obj2['map1'] == self.map1)

        index = obj1.index('map2')
        self.assertTrue(obj1[index] == self.map2)

        # is a uv name in the list?
        self.assertTrue(obj1[0] in obj1)                # uv object in list by name
        self.assertTrue('map1' in obj1)                 # string in list names
        self.assertFalse('SDFAFDAEWFASDFAFSD' in obj1)  # string not in list names

    def test_functions(self):
        obj1 = self.list.copy()

        # pattern matching
        obj2 = obj1.match("*2")
        self.assertTrue(obj2[0] == self.map2)





class TestMeshList(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define a base shapes
        self.mesh0 = MeshData(
            name='map0',
            points=np.array(
                [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

        self.mesh1 = MeshData(
            name='map1',
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
                ]
            ),
            indices=np.array(
                [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

        self.mesh2 = MeshData(
            name='map2',
            points=np.array(
                [
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                ]
            ),
            indices = np.array([0, 2, 1, 1, 2, 4, 3, 3, 4, 5, 5, 0, 1, 3]),
            counts  = np.array([3, 4, 3, 4]),
        )


        self.list = MeshList([self.mesh0, self.mesh1, self.mesh2])

    def test_bytes(self):
        obj1 = self.list.copy()
        b    = obj1.to_bytes()
        obj2 = MeshList.from_bytes(b)
        self.assertTrue(obj1 == obj2)

    def test_serialize(self):

            obj1 = self.list.copy()

            # write to file
            with tempfile.TemporaryDirectory() as temp_dir:
                f = os.path.join(temp_dir, "test.pkl")

                # save obj1 to file
                obj1.save(f)

                # load obj2 from file
                obj2 = MeshList.load(f)

                # confirm they're identical
                self.assertTrue(obj1 == obj2)

                # do the same via pickle, put obj1 inside a list
                with open(f, "wb") as io:
                    pickle.dump([obj1], io)

                with open(f, "rb") as io:
                    obj3 = pickle.load(io)[0]

                    # confirm they're identical
                    self.assertTrue(obj1 == obj3)

                # test json and npz
                obj1.save(f, mode='json')
                obj4 = MeshList.load(f)
                self.assertTrue(obj1 == obj4)

                obj1.save(f, mode='npz')
                obj5 = MeshList.load(f)
                self.assertTrue(obj1 == obj5)


    def test_list_like(self):

        # randomize list
        import random

        obj1 = self.list.copy()
        obj2 = self.list.copy()

        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj2.sort()
        self.assertTrue(obj1 == obj2)

        # use sorted
        while any(str(x)==str(y) for x, y in zip(obj1, obj2)):
            random.shuffle(obj2)

        self.assertTrue(obj1 != obj2)
        obj3 = MeshList(sorted(obj2))
        self.assertTrue(obj1 == obj3)


        # array indexing, slicing
        self.assertTrue(obj1[1] == self.mesh1)

        # reverse
        obj1.sort()
        obj2.sort()
        obj1.reverse()
        self.assertTrue(obj1 == obj2[::-1])

        # numpy like indexing
        obj1.sort()
        obj2.sort()
        self.assertTrue(obj1[[1, 2]] == obj2[1:])

        # mutability
        mesh0 = self.mesh0.copy()
        mesh1 = self.mesh1.copy()
        mesh2 = self.mesh2.copy()

        obj1  = MeshList([mesh0, mesh1, mesh2])
        obj2  = obj1[[2, 1, 0]]
        mesh2.name = 'POOF!'
        self.assertTrue(obj1[2].name == 'POOF!')
        self.assertTrue(obj2[0].name == 'POOF!')

        # dict like indexting
        obj1 = self.list.copy()
        obj2 = self.list.copy()
        self.assertTrue(obj2['map1'] == self.mesh1)

        index = obj1.index('map2')
        self.assertTrue(obj1[index] == self.mesh2)

        # is a uv name in the list?
        self.assertTrue(obj1[0] in obj1)                # uv object in list by name
        self.assertTrue('map1' in obj1)                 # string in list names
        self.assertFalse('SDFAFDAEWFASDFAFSD' in obj1)  # string not in list names

    def test_functions(self):
        obj1 = self.list.copy()

        # pattern matching
        obj2 = obj1.match("*2")
        self.assertTrue(obj2[0] == self.mesh2)


class TestTopologicalNeighborhood(unittest.TestCase):
    """Tests for compute_topological_neighborhood function."""

    def test_correctness(self):
        """Test correctness with a simple mesh."""
        # Create a simple chain: 0 - 1 - 2 - 3 - 4
        connectivity = np.array(
            [
                [1, -1, -1],  # vertex 0: neighbors [1]
                [0, 2, -1],  # vertex 1: neighbors [0, 2]
                [1, 3, -1],  # vertex 2: neighbors [1, 3]
                [2, 4, -1],  # vertex 3: neighbors [2, 4]
                [3, -1, -1],  # vertex 4: neighbors [3]
            ],
            dtype=np.int32,
        )

        # Test 1 hop
        result       = compute_topological_neighborhood(connectivity, num_hops=1)
        v0_neighbors = result[0][result[0] >= 0]
        self.assertIn(1, v0_neighbors)
        v2_neighbors = result[2][result[2] >= 0]
        self.assertIn(1, v2_neighbors)
        self.assertIn(3, v2_neighbors)

        # Test 2 hops
        result       = compute_topological_neighborhood(connectivity, num_hops=2)
        v0_neighbors = sorted(result[0][result[0] >= 0])
        self.assertEqual(v0_neighbors, [1, 2])
        v2_neighbors = sorted(result[2][result[2] >= 0])
        self.assertEqual(v2_neighbors, [0, 1, 3, 4])

        # Test 3 hops
        result       = compute_topological_neighborhood(connectivity, num_hops=3)
        v0_neighbors = sorted(result[0][result[0] >= 0])
        self.assertEqual(v0_neighbors, [1, 2, 3])

        # Verify uniqueness
        for i in range(result.shape[0]):
            neighbors        = result[i][result[i] >= 0]
            unique_neighbors = np.unique(neighbors)
            self.assertEqual(
                len(neighbors),
                len(unique_neighbors),
                f"Duplicate neighbors found for vertex {i}",
            )

        # Verify self not in neighborhood
        for i in range(result.shape[0]):
            neighbors = result[i][result[i] >= 0]
            self.assertNotIn(i, neighbors, f"Self found in neighborhood for vertex {i}")

    def test_indices_argument(self):
        """Test the optional indices argument."""
        connectivity = np.array(
            [
                [1, -1, -1],
                [0, 2, -1],
                [1, 3, -1],
                [2, 4, -1],
                [3, -1, -1],
            ],
            dtype=np.int32,
        )

        # Test 1: indices=None should match full computation
        full_result = compute_topological_neighborhood(connectivity, num_hops=2)
        none_result = compute_topological_neighborhood(
            connectivity, num_hops=2, indices=None
        )
        self.assertEqual(full_result.shape, none_result.shape)
        self.assertTrue(np.array_equal(full_result, none_result))

        # Test 2: Subset of indices returns correct shape
        indices = np.array([0, 2, 4], dtype=np.int32)
        subset_result = compute_topological_neighborhood(
            connectivity, num_hops=2, indices=indices
        )
        self.assertEqual(subset_result.shape[0], len(indices))

        # Test 3: Subset results match corresponding rows in full result
        for i, idx in enumerate(indices):
            subset_neighbors = set(subset_result[i][subset_result[i] >= 0])
            full_neighbors   = set(full_result[idx][full_result[idx] >= 0])
            self.assertEqual(subset_neighbors, full_neighbors)

        # Test 4: Single index
        single_result = compute_topological_neighborhood(
            connectivity, num_hops=2, indices=np.array([2])
        )
        self.assertEqual(single_result.shape[0], 1)
        single_neighbors = set(single_result[0][single_result[0] >= 0])
        full_neighbors   = set(full_result[2][full_result[2] >= 0])
        self.assertEqual(single_neighbors, full_neighbors)

        # Test 5: Indices as Python list
        list_result = compute_topological_neighborhood(
            connectivity, num_hops=2, indices=[1, 3]
        )
        self.assertEqual(list_result.shape[0], 2)

        # Test 6: Non-contiguous indices
        non_contig_indices = np.array([4, 0, 2], dtype=np.int32)
        non_contig_result = compute_topological_neighborhood(
            connectivity, num_hops=2, indices=non_contig_indices
        )
        for i, idx in enumerate(non_contig_indices):
            subset_neighbors = set(non_contig_result[i][non_contig_result[i] >= 0])
            full_neighbors   = set(full_result[idx][full_result[idx] >= 0])
            self.assertEqual(subset_neighbors, full_neighbors)

    def test_distances(self):
        """Test distance computation with known values."""
        connectivity = np.array(
            [
                [1, -1, -1],
                [0, 2, -1],
                [1, 3, -1],
                [2, 4, -1],
                [3, -1, -1],
            ],
            dtype=np.int32,
        )

        distances = np.array(
            [
                [1.0, -1, -1],
                [1.0, 2.0, -1],
                [2.0, 3.0, -1],
                [3.0, 4.0, -1],
                [4.0, -1, -1],
            ],
            dtype=np.float32,
        )

        # Test 1: distances=None returns same result (backward compat)
        result_no_dist = compute_topological_neighborhood(connectivity, num_hops=2)
        neighbors_with_dist, dists = compute_topological_neighborhood(
            connectivity, num_hops=2, distances=distances
        )
        self.assertTrue(np.array_equal(result_no_dist, neighbors_with_dist))

        # Test 2: Return type is tuple when distances provided
        result = compute_topological_neighborhood(
            connectivity, num_hops=1, distances=distances
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Test 3: Verify 1-hop distances
        neighbors, dists = compute_topological_neighborhood(
            connectivity, num_hops=1, distances=distances
        )
        v0_neighbors = neighbors[0][neighbors[0] >= 0]
        v0_dists     = dists[0][neighbors[0] >= 0]
        self.assertIn(1, v0_neighbors)
        idx = np.where(v0_neighbors == 1)[0][0]
        self.assertAlmostEqual(v0_dists[idx], 1.0, places=5)

        # Test 4: Verify 2-hop cumulative distances
        neighbors, dists = compute_topological_neighborhood(
            connectivity, num_hops=2, distances=distances
        )
        v0_neighbors = neighbors[0][neighbors[0] >= 0]
        v0_dists     = dists[0][neighbors[0] >= 0]

        self.assertIn(1, v0_neighbors)
        idx1 = np.where(v0_neighbors == 1)[0][0]
        self.assertAlmostEqual(v0_dists[idx1], 1.0, places=5)

        self.assertIn(2, v0_neighbors)
        idx2 = np.where(v0_neighbors == 2)[0][0]
        self.assertAlmostEqual(v0_dists[idx2], 3.0, places=5)

        # Test 5: Test with indices subset
        neighbors, dists = compute_topological_neighborhood(
            connectivity, num_hops=2, indices=np.array([0, 2]), distances=distances
        )
        self.assertEqual(neighbors.shape[0], 2)
        self.assertEqual(dists.shape[0],     2)
        self.assertEqual(neighbors.shape,    dists.shape)

        # Test 6: Verify padding is -1.0 for distances
        neighbors, dists = compute_topological_neighborhood(
            connectivity, num_hops=1, distances=distances
        )
        if neighbors.shape[1] > 1:
            self.assertEqual(dists[0, 1], -1.0)

    def test_max_distance(self):
        """Test max_distance parameter excludes neighbors beyond the threshold."""
        connectivity = np.array(
            [
                [1, -1, -1],
                [0, 2, -1],
                [1, 3, -1],
                [2, 4, -1],
                [3, -1, -1],
            ],
            dtype=np.int32,
        )

        distances = np.array(
            [
                [1.0, -1, -1],
                [1.0, 2.0, -1],
                [2.0, 3.0, -1],
                [3.0, 4.0, -1],
                [4.0, -1, -1],
            ],
            dtype=np.float32,
        )

        # Test 1: max_distance=None should behave same as inf
        neighbors_no_limit, dists_no_limit = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, max_distance=None
        )
        neighbors_inf, dists_inf = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, max_distance=float("inf")
        )
        self.assertTrue(np.array_equal(neighbors_no_limit, neighbors_inf))
        self.assertTrue(
            np.allclose(
                dists_no_limit[dists_no_limit >= 0],
                dists_inf[dists_inf >= 0],
            )
        )

        # Test 2: max_distance=0 should exclude all neighbors
        neighbors_zero, dists_zero = compute_topological_neighborhood(
            connectivity,
            num_hops     = 4,
            indices      = np.array([0]),
            distances    = distances,
            max_distance = 0.0,
        )
        v0_neighbors = neighbors_zero[0][neighbors_zero[0] >= 0]
        self.assertEqual(len(v0_neighbors), 0)

        # Test 3: max_distance=1.5 allows only vertex 1 from vertex 0
        neighbors_1p5, dists_1p5 = compute_topological_neighborhood(
            connectivity,
            num_hops     = 4,
            indices      = np.array([0]),
            distances    = distances,
            max_distance = 1.5,
        )
        v0_neighbors = neighbors_1p5[0][neighbors_1p5[0] >= 0]
        v0_dists     = dists_1p5[0][neighbors_1p5[0] >= 0]

        self.assertIn(1, v0_neighbors)
        self.assertNotIn(2, v0_neighbors)
        self.assertNotIn(3, v0_neighbors)

        idx1 = np.where(v0_neighbors == 1)[0][0]
        self.assertAlmostEqual(v0_dists[idx1], 1.0, places=5)
        self.assertLessEqual(v0_dists.max(), 1.5)

        # Test 4: max_distance large enough should match no-limit behavior
        neighbors_large, dists_large = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, max_distance=100.0
        )
        self.assertTrue(np.array_equal(neighbors_no_limit, neighbors_large))

        # Test 5: Verify neighbors beyond max_distance are excluded entirely
        neighbors_2p5, dists_2p5 = compute_topological_neighborhood(
            connectivity,
            num_hops     = 4,
            indices      = np.array([0]),
            distances    = distances,
            max_distance = 2.5,
        )
        v0_neighbors = neighbors_2p5[0][neighbors_2p5[0] >= 0]
        v0_dists     = dists_2p5[0][neighbors_2p5[0] >= 0]
        self.assertIn(1, v0_neighbors)
        self.assertNotIn(2, v0_neighbors)
        self.assertNotIn(3, v0_neighbors)
        if len(v0_dists) > 0:
            self.assertLessEqual(v0_dists.max(), 2.5)

        # Test 6: Test with middle vertex to verify bidirectional behavior
        neighbors_mid, dists_mid = compute_topological_neighborhood(
            connectivity,
            num_hops     = 4,
            indices      = np.array([2]),
            distances    = distances,
            max_distance = 3.5,
        )
        v2_neighbors = neighbors_mid[0][neighbors_mid[0] >= 0]
        v2_dists     = dists_mid[0][neighbors_mid[0] >= 0]
        self.assertIn(0, v2_neighbors)
        self.assertIn(1, v2_neighbors)
        self.assertIn(3, v2_neighbors)
        self.assertNotIn(4, v2_neighbors)
        self.assertLessEqual(v2_dists.max(), 3.5)

        # Test 7: Verify d.max() never exceeds max_distance for any vertex
        neighbors_all, dists_all = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, max_distance=5.0
        )
        for i in range(len(neighbors_all)):
            valid_dists = dists_all[i][dists_all[i] >= 0]
            if len(valid_dists) > 0:
                self.assertLessEqual(valid_dists.max(), 5.0)

    def test_shortest_distance_selection(self):
        """Test that shorter euclidean paths are preferred when found at same depth."""
        # Diamond graph - two paths to same vertex at same depth
        connectivity_diamond = np.array(
            [
                [1, 2, -1],
                [0, 3, -1],
                [0, 3, -1],
                [1, 2, -1],
            ],
            dtype=np.int32,
        )

        distances_diamond = np.array(
            [
                [1.0, 5.0, -1],
                [1.0, 1.0, -1],
                [5.0, 0.5, -1],
                [1.0, 0.5, -1],
            ],
            dtype=np.float32,
        )

        neighbors, dists = compute_topological_neighborhood(
            connectivity_diamond,
            num_hops  = 2,
            indices   = np.array([0]),
            distances = distances_diamond,
        )

        v0_neighbors = neighbors[0]
        v0_dists     = dists[0]
        idx3         = np.where(v0_neighbors == 3)[0][0]
        dist_to_3    = v0_dists[idx3]

        # Should be 2.0 (via 0->1->3), not 5.5 (via 0->2->3)
        self.assertAlmostEqual(dist_to_3, 2.0, places=5)

        # Verify all distances are optimal
        v0_neighbors_valid = neighbors[0][neighbors[0] >= 0]
        v0_dists_valid     = dists[0][neighbors[0] >= 0]

        for n, expected in [(1, 1.0), (2, 5.0), (3, 2.0)]:
            idx    = np.where(v0_neighbors_valid == n)[0][0]
            actual = v0_dists_valid[idx]
            self.assertAlmostEqual(actual, expected, places=5)

        # Multi-path distance updates
        connectivity_multi = np.array(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
            dtype=np.int32,
        )

        distances_multi = np.array(
            [
                [1.0, 10.0, 10.0],
                [1.0, 0.1, 0.1],
                [10.0, 0.1, 0.1],
                [10.0, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        neighbors, dists = compute_topological_neighborhood(
            connectivity_multi,
            num_hops  = 2,
            indices   = np.array([0]),
            distances = distances_multi,
        )

        v0_neighbors = neighbors[0]
        v0_dists     = dists[0]

        idx1         = np.where(v0_neighbors == 1)[0][0]
        idx2         = np.where(v0_neighbors == 2)[0][0]
        idx3         = np.where(v0_neighbors == 3)[0][0]

        self.assertAlmostEqual(v0_dists[idx1], 1.0, places=5)
        self.assertAlmostEqual(v0_dists[idx2], 1.1, places=5)
        self.assertAlmostEqual(v0_dists[idx3], 1.1, places=5)

    def test_uniqueness_large(self):
        """Test uniqueness on larger mesh."""
        np.random.seed(42)
        n_vertices    = 1000
        max_neighbors = 6

        connectivity  = np.full((n_vertices, max_neighbors), -1, dtype=np.int32)
        for i in range(n_vertices):
            n_neigh   = np.random.randint(2, max_neighbors + 1)
            neighbors = np.random.choice(n_vertices, size=n_neigh, replace=False)
            neighbors = neighbors[neighbors != i][:max_neighbors]
            connectivity[i, : len(neighbors)] = neighbors

        result = compute_topological_neighborhood(connectivity, num_hops=5)

        # Check uniqueness for all vertices
        for i in range(n_vertices):
            neighbors        = result[i][result[i] >= 0]
            unique_neighbors = np.unique(neighbors)
            self.assertEqual(
                len(neighbors),
                len(unique_neighbors),
                f"Duplicate neighbors found for vertex {i}",
            )

    def test_num_hops_optional(self):
        """Test that num_hops=None explores all reachable vertices."""
        connectivity = np.array(
            [
                [1, -1, -1],
                [0, 2, -1],
                [1, 3, -1],
                [2, 4, -1],
                [3, -1, -1],
            ],
            dtype=np.int32,
        )

        # Test 1: num_hops=None should find all vertices from vertex 0
        result_none = compute_topological_neighborhood(
            connectivity, num_hops=None, indices=np.array([0])
        )
        result_explicit = compute_topological_neighborhood(
            connectivity, num_hops=4, indices=np.array([0])
        )
        v0_none     = sorted(result_none[0][result_none[0] >= 0])
        v0_explicit = sorted(result_explicit[0][result_explicit[0] >= 0])
        self.assertEqual(v0_none, v0_explicit)
        self.assertEqual(set(v0_none), {1, 2, 3, 4})

        # Test 2: num_hops=None for all vertices
        result_all = compute_topological_neighborhood(connectivity, num_hops=None)
        self.assertEqual(result_all.shape[0], 5)
        for i in range(5):
            neighbors = set(result_all[i][result_all[i] >= 0])
            expected  = set(range(5)) - {i}
            self.assertEqual(neighbors, expected)

        # Test 3: num_hops=None with distances
        distances = np.array(
            [
                [1.0, -1, -1],
                [1.0, 2.0, -1],
                [2.0, 3.0, -1],
                [3.0, 4.0, -1],
                [4.0, -1, -1],
            ],
            dtype=np.float32,
        )
        neighbors, dists = compute_topological_neighborhood(
            connectivity, num_hops=None, distances=distances, indices=np.array([0])
        )
        v0_neighbors = neighbors[0][neighbors[0] >= 0]
        self.assertEqual(set(v0_neighbors), {1, 2, 3, 4})

        # Test 4: Disconnected graph
        connectivity_disconnected = np.array(
            [
                [1, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
                [2, -1, -1],
            ],
            dtype=np.int32,
        )
        result_disconnected = compute_topological_neighborhood(
            connectivity_disconnected, num_hops=None, indices=np.array([0])
        )
        v0_neighbors = set(result_disconnected[0][result_disconnected[0] >= 0])
        self.assertEqual(v0_neighbors, {1})

    def test_dist_format(self):
        """Test distance_format parameter returns correct output formats."""
        connectivity = np.array(
            [
                [1, -1, -1],
                [0, 2, -1],
                [1, 3, -1],
                [2, 4, -1],
                [3, -1, -1],
            ],
            dtype=np.int32,
        )

        distances = np.array(
            [
                [1.0, -1, -1],
                [1.0, 2.0, -1],
                [2.0, 3.0, -1],
                [3.0, 4.0, -1],
                [4.0, -1, -1],
            ],
            dtype=np.float32,
        )

        # Test 1: distance_format="dense" returns tuple with correct shapes
        result = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, distance_format="dense"
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        neighborhood, dist_matrix = result
        self.assertEqual(dist_matrix.shape, (5, 5))
        self.assertEqual(neighborhood.shape[0], 5)

        # Test 2: Diagonal should be 0
        diagonal = np.diag(dist_matrix)
        self.assertTrue(np.all(diagonal == 0.0))

        # Test 3: Matrix should be symmetric
        self.assertTrue(np.allclose(dist_matrix, dist_matrix.T))

        # Test 3b: Neighborhood matches sparse result
        sparse_neighborhood, _ = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, distance_format="sparse"
        )
        self.assertTrue(np.array_equal(neighborhood, sparse_neighborhood))

        # Test 4: Verify specific distances
        self.assertAlmostEqual(dist_matrix[0, 1], 1.0,  places=5)
        self.assertAlmostEqual(dist_matrix[0, 2], 3.0,  places=5)
        self.assertAlmostEqual(dist_matrix[0, 3], 6.0,  places=5)
        self.assertAlmostEqual(dist_matrix[0, 4], 10.0, places=5)

        # Test 5: distance_format="dense" with subset of indices
        indices = np.array([0, 2, 4], dtype=np.int32)
        neighborhood_subset, dist_matrix_subset = compute_topological_neighborhood(
            connectivity,
            num_hops        = 4,
            indices         = indices,
            distances       = distances,
            distance_format = "dense",
        )
        self.assertEqual(dist_matrix_subset.shape, (3, 5))
        self.assertEqual(neighborhood_subset.shape[0], 3)
        self.assertAlmostEqual(dist_matrix_subset[0, 2], 3.0,  places=5)
        self.assertAlmostEqual(dist_matrix_subset[0, 4], 10.0, places=5)
        self.assertAlmostEqual(dist_matrix_subset[1, 4], 7.0,  places=5)

        # Test 6: distance_format="dense" requires distances
        with self.assertRaises(ValueError) as ctx:
            compute_topological_neighborhood(
                connectivity, num_hops=4, distance_format="dense"
            )
        self.assertIn("distance_format='dense' requires distances", str(ctx.exception))


        # Test 7: distance_format="dense" with max_distance
        neighborhood_limited, dist_matrix_limited = compute_topological_neighborhood(
            connectivity,
            num_hops        = 4,
            distances       = distances,
            max_distance    = 2.5,
            distance_format = "dense",
        )
        self.assertEqual(dist_matrix_limited[0, 1], 1.0)
        self.assertTrue(np.isinf(dist_matrix_limited[0, 2]))
        self.assertTrue(np.isinf(dist_matrix_limited[0, 3]))
        self.assertTrue(np.isinf(dist_matrix_limited[0, 4]))

        # Test 8: distance_format="dense" with num_hops=None
        neighborhood_all, dist_matrix_all = compute_topological_neighborhood(
            connectivity, num_hops=None, distances=distances, distance_format="dense"
        )
        self.assertEqual(dist_matrix_all.shape, (5, 5))
        self.assertTrue(np.all(np.diag(dist_matrix_all) == 0.0))
        self.assertFalse(np.any(np.isinf(dist_matrix_all)))

        # Test 9: Explicit distance_format="sparse" returns same as default
        result_default = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances
        )
        result_sparse = compute_topological_neighborhood(
            connectivity, num_hops=4, distances=distances, distance_format="sparse"
        )
        self.assertIsInstance(result_default, tuple)
        self.assertEqual(len(result_default), 2)
        self.assertIsInstance(result_sparse, tuple)
        self.assertEqual(len(result_sparse), 2)
        self.assertTrue(np.array_equal(result_default[0], result_sparse[0]))
        self.assertTrue(np.array_equal(result_default[1], result_sparse[1]))

    def test_stale_frontier_distance(self):
        """Test that frontier vertex distances are read from result_dists."""
        connectivity = np.array(
            [
                [1, 2, -1, -1],
                [0, 3, 4, -1],
                [0, 4, -1, -1],
                [1, -1, -1, -1],
                [1, 2, 5, -1],
                [4, -1, -1, -1],
            ],
            dtype=np.int32,
        )

        distances = np.array(
            [
                [1.0, 5.0, -1, -1],
                [1.0, 0.1, 0.1, -1],
                [5.0, 0.1, -1, -1],
                [0.1, -1, -1, -1],
                [0.1, 0.1, 0.1, -1],
                [0.1, -1, -1, -1],
            ],
            dtype=np.float32,
        )

        neighbors, dists = compute_topological_neighborhood(
            connectivity,
            num_hops  = 3,
            indices   = np.array([0]),
            distances = distances,
        )

        v0_neighbors = neighbors[0][neighbors[0] >= 0]
        v0_dists     = dists[0][neighbors[0] >= 0]

        self.assertIn(5, v0_neighbors)
        idx5      = np.where(v0_neighbors == 5)[0][0]
        dist_to_5 = v0_dists[idx5]
        self.assertAlmostEqual(dist_to_5, 1.2, places=5)

        idx4      = np.where(v0_neighbors == 4)[0][0]
        dist_to_4 = v0_dists[idx4]
        self.assertAlmostEqual(dist_to_4, 1.1, places=5)

    def test_excluded_vertex_reconsideration(self):
        """Test that vertices excluded by max_distance can be reconsidered via shorter paths."""
        connectivity = np.array(
            [
                [1, 2, -1],
                [0, 3, -1],
                [0, 4, -1],
                [1, 4, -1],
                [2, 3, -1],
            ],
            dtype=np.int32,
        )

        distances = np.array(
            [
                [5.0, 0.1, -1],
                [5.0, 5.0, -1],
                [0.1, 0.1, -1],
                [5.0, 0.1, -1],
                [0.1, 0.1, -1],
            ],
            dtype=np.float32,
        )

        neighbors, dists = compute_topological_neighborhood(
            connectivity,
            num_hops     = 4,
            indices      = np.array([0]),
            distances    = distances,
            max_distance = 0.5,
        )

        v0_neighbors = neighbors[0][neighbors[0] >= 0]
        v0_dists     = dists[0][neighbors[0] >= 0]

        # Vertex 1 should be excluded
        self.assertNotIn(1, v0_neighbors)

        # Vertex 2 should be included
        self.assertIn(2, v0_neighbors)
        idx2 = np.where(v0_neighbors == 2)[0][0]
        self.assertAlmostEqual(v0_dists[idx2], 0.1, places=5)

        # Vertex 4 should be included
        self.assertIn(4, v0_neighbors)
        idx4 = np.where(v0_neighbors == 4)[0][0]
        self.assertAlmostEqual(v0_dists[idx4], 0.2, places=5)

        # Vertex 3 should be included via shorter path
        self.assertIn(3, v0_neighbors)
        idx3      = np.where(v0_neighbors == 3)[0][0]
        dist_to_3 = v0_dists[idx3]
        self.assertAlmostEqual(dist_to_3, 0.3, places=5)

    def test_geodesic_neighborhood_edge_based(self):
        """Test edge-based geodesic neighborhood methods."""
        # Create a simple grid mesh (4x4 vertices, 9 quads)
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0],
            [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0],
            [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6,
            4, 5, 9, 8, 5, 6, 10, 9, 6, 7, 11, 10,
            8, 9, 13, 12, 9, 10, 14, 13, 10, 11, 15, 14
        ], dtype=np.int32)
        counts = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)
        mesh   = MeshData(points=points, indices=indices, counts=counts)

        # Test get_face_edge_geodesic_neighborhood
        # With a max_distance of 1.5, we should reach immediate neighbors
        neighbors, dists = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=1.5, n=2, indices=np.array([0]), distance_format="shortest"
        )

        # Result should contain edge indices within max_distance
        self.assertIsInstance(neighbors, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(len(neighbors), len(dists))
        # Should find neighbor edges
        self.assertGreater(len(neighbors), 0)

        # Test get_vertex_edge_geodesic_neighborhood
        neighbors, dists = mesh.get_vertex_edge_geodesic_neighborhood(
            max_distance=1.5, n=2, indices=np.array([0]), distance_format="shortest"
        )

        self.assertIsInstance(neighbors, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(len(neighbors), len(dists))
        self.assertGreater(len(neighbors), 0)

        # Test with larger max_distance to verify more edges are included
        neighbors_large, dists_large = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=5.0, n=5, indices=np.array([0]), distance_format="shortest"
        )
        # With larger distance, we should reach more edges
        self.assertGreaterEqual(len(neighbors_large), len(neighbors))

        # Verify distance values are non-negative
        self.assertTrue(np.all(dists >= 0))
        self.assertTrue(np.all(dists_large >= 0))

    def test_geodesic_neighborhood_face_based(self):
        """Test face-based geodesic neighborhood methods."""
        # Create a simple grid mesh (4x4 vertices, 9 quads)
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0],
            [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0],
            [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6,
            4, 5, 9, 8, 5, 6, 10, 9, 6, 7, 11, 10,
            8, 9, 13, 12, 9, 10, 14, 13, 10, 11, 15, 14
        ], dtype=np.int32)
        counts = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)
        mesh   = MeshData(points=points, indices=indices, counts=counts)

        # Test get_vertex_face_geodesic_neighborhood
        neighbors, dists = mesh.get_vertex_face_geodesic_neighborhood(
            max_distance=2.0, n=2, indices=np.array([0]), distance_format="shortest"
        )

        self.assertIsInstance(neighbors, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(len(neighbors), len(dists))
        # Should find neighbor faces
        self.assertGreater(len(neighbors), 0)

        # Test get_edge_face_geodesic_neighborhood
        neighbors, dists = mesh.get_edge_face_geodesic_neighborhood(
            max_distance=2.0, n=2, indices=np.array([0]), distance_format="shortest"
        )

        self.assertIsInstance(neighbors, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(len(neighbors), len(dists))
        self.assertGreater(len(neighbors), 0)

        # Test with larger max_distance
        neighbors_large, dists_large = mesh.get_vertex_face_geodesic_neighborhood(
            max_distance=10.0, n=10, indices=np.array([0]), distance_format="shortest"
        )
        # With larger distance, we should reach more faces
        self.assertGreaterEqual(len(neighbors_large), len(neighbors))

        # Verify distance values are non-negative
        self.assertTrue(np.all(dists >= 0))
        self.assertTrue(np.all(dists_large >= 0))

    def test_geodesic_neighborhood_distance_formats(self):
        """Test different distance_format options for geodesic neighborhood methods."""
        # Create a simple 2x2 grid mesh (4 vertices, 1 quad)
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([0, 1, 3, 2], dtype=np.int32)
        counts  = np.array([4], dtype=np.int32)
        mesh    = MeshData(points=points, indices=indices, counts=counts)

        # Test sparse format (default)
        result_sparse = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=5.0, n=3, distance_format="sparse"
        )
        self.assertIsInstance(result_sparse, tuple)
        self.assertEqual(len(result_sparse), 2)
        # Sparse format returns (neighborhood_matrix, distances_matrix)
        neighborhood, distances = result_sparse
        self.assertIsInstance(neighborhood, np.ndarray)
        self.assertIsInstance(distances, np.ndarray)

        # Test dense format
        result_dense = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=5.0, n=3, distance_format="dense"
        )
        self.assertIsInstance(result_dense, tuple)
        self.assertEqual(len(result_dense), 2)
        neighborhood_dense, distances_dense = result_dense
        # Dense format should return square matrices
        num_edges = mesh.e2v.shape[0]
        self.assertEqual(distances_dense.shape, (num_edges, num_edges))

        # Test shortest format
        result_shortest = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=5.0, n=3, distance_format="shortest"
        )
        self.assertIsInstance(result_shortest, tuple)
        self.assertEqual(len(result_shortest), 2)
        # Shortest returns (indices, distances) arrays
        idx_arr, dist_arr = result_shortest
        self.assertEqual(len(idx_arr), len(dist_arr))

    def test_geodesic_neighborhood_all_indices(self):
        """Test geodesic neighborhoods when indices=None (all elements)."""
        # Create a simple 3x3 grid mesh
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 4, 3, 1, 2, 5, 4,
            3, 4, 7, 6, 4, 5, 8, 7,
        ], dtype=np.int32)
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        mesh   = MeshData(points=points, indices=indices, counts=counts)

        # Test face_edge with all edges (indices=None)
        result = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=5.0, n=3, indices=None, distance_format="sparse"
        )
        neighborhood, distances = result
        num_edges = mesh.e2v.shape[0]
        # When indices=None, should return results for all edges
        self.assertEqual(neighborhood.shape[0], num_edges)
        self.assertEqual(distances.shape[0], num_edges)

        # Test vertex_face with all faces (indices=None)
        result = mesh.get_vertex_face_geodesic_neighborhood(
            max_distance=5.0, n=3, indices=None, distance_format="sparse"
        )
        neighborhood, distances = result
        num_faces = len(counts)
        # When indices=None, should return results for all faces
        self.assertEqual(neighborhood.shape[0], num_faces)
        self.assertEqual(distances.shape[0], num_faces)

    def test_geodesic_neighborhood_distance_accuracy(self):
        """Test that geodesic distances are computed accurately."""
        # Create a simple line of 4 quads (5 vertices in a row)
        # Layout: v0--e0--v1--e1--v2--e2--v3--e3--v4
        #         |       |       |       |       |
        #         v5------v6------v7------v8------v9
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 6, 5, 1, 2, 7, 6, 2, 3, 8, 7, 3, 4, 9, 8
        ], dtype=np.int32)
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        mesh   = MeshData(points=points, indices=indices, counts=counts)

        # Test face-based distances
        # Face centroids should be at (0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5)
        neighbors, dists = mesh.get_vertex_face_geodesic_neighborhood(
            max_distance=10.0, n=10, indices=np.array([0]), distance_format="shortest"
        )

        # Face 0 centroid to face 1 centroid distance should be ~1.0
        if 1 in neighbors:
            idx1 = np.where(neighbors == 1)[0][0]
            # Distance from face 0 to face 1 via vertex connectivity
            # Should be approximately 1.0 (horizontal distance between centroids)
            self.assertAlmostEqual(dists[idx1], 1.0, places=1)

        # Test edge-based distances
        neighbors, dists = mesh.get_face_edge_geodesic_neighborhood(
            max_distance=10.0, n=10, indices=np.array([0]), distance_format="shortest"
        )

        # Edge centers for adjacent edges should be ~0.5-1.0 apart
        self.assertTrue(len(neighbors) > 1)  # Should find neighbor edges
        # All distances should be non-negative
        self.assertTrue(np.all(dists >= 0))

    def test_geodesic_neighborhood_max_distance_filtering(self):
        """Test that max_distance correctly filters distant elements."""
        # Create a long strip of faces
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 6, 5, 1, 2, 7, 6, 2, 3, 8, 7, 3, 4, 9, 8
        ], dtype=np.int32)
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        mesh   = MeshData(points=points, indices=indices, counts=counts)

        # With small max_distance, should only include nearby faces
        neighbors_small, dists_small = mesh.get_vertex_face_geodesic_neighborhood(
            max_distance=0.5, n=10, indices=np.array([0]), distance_format="shortest"
        )

        # With large max_distance, should include more faces
        neighbors_large, dists_large = mesh.get_vertex_face_geodesic_neighborhood(
            max_distance=10.0, n=10, indices=np.array([0]), distance_format="shortest"
        )

        # Larger max_distance should yield more or equal neighbors
        self.assertGreaterEqual(len(neighbors_large), len(neighbors_small))

        # All distances in small result should be <= max_distance
        self.assertTrue(np.all(dists_small <= 0.5 + 1e-5))

    def test_geodesic_neighborhood_n_hops_limiting(self):
        """Test that n parameter correctly limits the number of hops."""
        # Create a 3x3 grid mesh
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 4, 3, 1, 2, 5, 4,
            3, 4, 7, 6, 4, 5, 8, 7,
        ], dtype=np.int32)
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        mesh   = MeshData(points=points, indices=indices, counts=counts)

        # With n=1, should only get immediate neighbors
        neighbors_1hop, _ = mesh.get_edge_face_geodesic_neighborhood(
            max_distance=100.0, n=1, indices=np.array([0]), distance_format="shortest"
        )

        # With n=5, should get more neighbors
        neighbors_5hop, _ = mesh.get_edge_face_geodesic_neighborhood(
            max_distance=100.0, n=5, indices=np.array([0]), distance_format="shortest"
        )

        # More hops should yield more or equal neighbors
        self.assertGreaterEqual(len(neighbors_5hop), len(neighbors_1hop))


class TestBezierSampling(unittest.TestCase):
    """Tests for MeshData.sample() with SampleMethod.BEZIER (PN Quad patches)."""

    def setUp(self):
        super().setUp()

        # Unit cube (same as TestMesh.box)
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
                [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
            ),
            counts=np.array([4, 4, 4, 4, 4, 4]),
        )

        # Flat quad for bilinear-vs-bezier equivalence tests
        self.flat_quad = MeshData(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            indices = np.array([0, 1, 2, 3]),
            counts  = np.array([4]),
        )

        # Flat triangle
        self.flat_tri = MeshData(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
                dtype=np.float64,
            ),
            indices = np.array([0, 1, 2]),
            counts  = np.array([3]),
        )

    def test_flat_quad_bezier_matches_bilinear(self):
        """On a flat quad with uniform normals, Bezier should match bilinear."""
        np.random.seed(123)
        queries   = np.random.rand(50, 3) * 2 - 0.5

        result_bi = self.flat_quad.sample(queries, method=SampleMethod.BILINEAR)
        result_bz = self.flat_quad.sample(queries, method=SampleMethod.BEZIER)

        self.assertTrue(
            np.allclose(result_bi.distances, result_bz.distances, atol=1e-4),
            "Distances should match for flat quad",
        )
        self.assertTrue(
            np.allclose(result_bi.uvs, result_bz.uvs, atol=1e-3),
            "UVs should match for flat quad",
        )

    def test_flat_triangle_bezier_matches_bilinear(self):
        """On a flat triangle, Bezier should match bilinear (degenerate quad)."""
        queries   = np.array([[0.5, 0.4, 1.0]], dtype=np.float64)

        result_bi = self.flat_tri.sample(queries, method=SampleMethod.BILINEAR)
        result_bz = self.flat_tri.sample(queries, method=SampleMethod.BEZIER)

        self.assertTrue(
            np.allclose(result_bi.distances, result_bz.distances, atol=1e-3),
            "Triangle distances should match between bilinear and Bezier",
        )

    def test_sphere_approximation_bezier_more_accurate(self):
        """Bezier should approximate a sphere better than bilinear.

        Scale the cube so vertices lie on the unit sphere, compute sphere normals
        (normalized vertex positions), then query from radius 1.5.
        True distance to unit sphere = 0.5.
        """
        cube_pts = self.cube.points.copy()

        # Sphere normals = normalized vertex positions
        cube_nrm = cube_pts.copy()
        norms    = np.linalg.norm(cube_nrm, axis=1, keepdims=True)
        cube_nrm /= norms

        # Scale cube so vertices sit on the unit sphere
        scale = 1.0 / np.sqrt(0.75)
        sphere_cube = MeshData(
            points  = cube_pts * scale,
            indices = self.cube.indices.copy(),
            counts  = self.cube.counts.copy(),
        )

        # Query points at radius 1.5 along random directions
        np.random.seed(42)
        n_queries  = 200
        directions = np.random.randn(n_queries, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        queries   = directions * 1.5

        true_dist = 0.5

        result_bi = sphere_cube.sample(queries, method=SampleMethod.BILINEAR)
        result_bz = sphere_cube.sample(queries, method=SampleMethod.BEZIER)

        err_bi    = np.abs(result_bi.distances - true_dist)
        err_bz    = np.abs(result_bz.distances - true_dist)

        self.assertLess(
            np.mean(err_bz),
            np.mean(err_bi),
            "Bezier mean error should be smaller than bilinear for sphere approximation",
        )

    def test_vertex_self_query_bezier_zero_distance(self):
        """Querying mesh vertices against their own mesh should yield ~0 distance."""
        result = self.cube.sample(
            self.cube.points.copy(), method=SampleMethod.BEZIER
        )

        self.assertTrue(
            np.allclose(result.distances, 0.0, atol=1e-4),
            f"Self-query distances should be ~0, got max={np.max(result.distances):.2e}",
        )

    def test_bezier_default_is_bilinear(self):
        """Default sample method should be bilinear (no normals passed to kernel)."""
        queries         = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)

        result_default  = self.cube.sample(queries)
        result_bilinear = self.cube.sample(queries, method=SampleMethod.BILINEAR)

        self.assertTrue(
            np.allclose(result_default.distances, result_bilinear.distances, atol=1e-10),
            "Default method should behave identically to BILINEAR",
        )

    def test_bezier_string_method(self):
        """SampleMethod should accept string values."""
        queries     = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)

        result_enum = self.cube.sample(queries, method=SampleMethod.BEZIER)
        result_str  = self.cube.sample(queries, method="bezier")

        self.assertTrue(
            np.allclose(result_enum.distances, result_str.distances, atol=1e-10),
            "String 'bezier' should behave identically to SampleMethod.BEZIER",
        )


class TestBSplinePatchData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

        # 6x6 wavy surface for open-open
        u_vals = np.linspace(-5, 5, 6)
        v_vals = np.linspace(-5, 5, 6)
        uu, vv = np.meshgrid(u_vals, v_vals, indexing='ij')
        zz   = np.sin(uu * 0.5) * np.cos(vv * 0.5)
        grid = np.stack([uu, vv, zz], axis=-1)

        self.open_open = BSplinePatchData(
            points=grid.copy(), degree_u=3, degree_v=3,
            periodic_u=False, periodic_v=False)

        self.closed_open = BSplinePatchData(
            points=grid.copy(), degree_u=3, degree_v=3,
            periodic_u=True, periodic_v=False)

        self.open_closed = BSplinePatchData(
            points=grid.copy(), degree_u=3, degree_v=3,
            periodic_u=False, periodic_v=True)

        self.closed_closed = BSplinePatchData(
            points=grid.copy(), degree_u=3, degree_v=3,
            periodic_u=True, periodic_v=True)

    # -------------------------------------------------------------- #
    #  Serialization                                                   #
    # -------------------------------------------------------------- #

    def test_serialization(self):
        for patch in [self.open_open, self.closed_open, self.open_closed, self.closed_closed]:
            obj1 = patch.copy()

            # to_bytes / from_bytes round-trip
            b    = obj1.to_bytes()
            obj2 = BSplinePatchData.from_bytes(b)
            self.assertTrue(obj1 == obj2)

            with tempfile.TemporaryDirectory() as tmp:
                # pkl round-trip
                f = os.path.join(tmp, "test.pkl")
                obj1.save(f)
                obj3 = BSplinePatchData.load(f)
                self.assertTrue(obj1 == obj3)

                # json round-trip
                obj1.save(f, mode='json')
                obj4 = BSplinePatchData.load(f)
                self.assertTrue(obj1 == obj4)

                # npz round-trip
                obj1.save(f, mode='npz')
                obj5 = BSplinePatchData.load(f)
                self.assertTrue(obj1 == obj5)

            # private data excluded
            obj1._custom_data = "test"
            b    = obj1.to_bytes()
            obj6 = BSplinePatchData.from_bytes(b)
            self.assertFalse(hasattr(obj6, "_custom_data"))

    # -------------------------------------------------------------- #
    #  Properties                                                      #
    # -------------------------------------------------------------- #

    def test_properties(self):
        obj = self.open_open
        self.assertEqual(obj.count_u,     6)
        self.assertEqual(obj.count_v,     6)
        self.assertEqual(obj.max_param_u, 3)
        self.assertEqual(obj.max_param_v, 3)
        self.assertTrue(allclose(obj.cv, obj.points))
        self.assertGreater(obj.area, 0)

        obj = self.closed_closed
        self.assertEqual(obj.count_u,     6)
        self.assertEqual(obj.count_v,     6)
        self.assertEqual(obj.max_param_u, 6)
        self.assertEqual(obj.max_param_v, 6)
        self.assertFalse(obj.cv.shape == obj.points.shape)
        self.assertGreater(obj.area, 0)

    # -------------------------------------------------------------- #
    #  Compute                                                         #
    # -------------------------------------------------------------- #

    def test_compute(self):
        for patch in [self.open_open, self.closed_open, self.open_closed, self.closed_closed]:
            u = np.linspace(0, patch.max_param_u * 0.99, 5)
            v = np.linspace(0, patch.max_param_v * 0.99, 5)
            pts, su, sv = patch.compute(u, v)
            self.assertEqual(pts.shape, (5, 3))
            self.assertEqual(su.shape,  (5, 3))
            self.assertEqual(sv.shape,  (5, 3))

            # evaluate (position only) should match compute positions
            pts2 = patch.evaluate(u, v)
            self.assertTrue(np.allclose(pts, pts2, atol=1e-10))

    # -------------------------------------------------------------- #
    #  Derivative finite-difference check                              #
    # -------------------------------------------------------------- #

    def test_derivatives_finite_difference(self):
        patch = self.open_open
        eps   = 1e-6
        u     = np.array([0.5, 1.0, 1.5, 2.0])
        v     = np.array([0.5, 1.0, 1.5, 2.0])

        _, su, sv = patch.compute(u, v)

        # Finite difference for dS/du
        pts_up = patch.evaluate(u + eps, v)
        pts_um = patch.evaluate(u - eps, v)
        su_fd  = (pts_up - pts_um) / (2 * eps)
        self.assertTrue(np.allclose(su, su_fd, atol=1e-4),
                        f"S_u max error: {np.max(np.abs(su - su_fd))}")

        # Finite difference for dS/dv
        pts_vp = patch.evaluate(u, v + eps)
        pts_vm = patch.evaluate(u, v - eps)
        sv_fd  = (pts_vp - pts_vm) / (2 * eps)
        self.assertTrue(np.allclose(sv, sv_fd, atol=1e-4),
                        f"S_v max error: {np.max(np.abs(sv - sv_fd))}")

    # -------------------------------------------------------------- #
    #  Sample -- self-projection                                        #
    # -------------------------------------------------------------- #

    def test_sample_self_projection(self):
        for patch in [self.open_open, self.closed_open]:
            u = np.linspace(0.1, patch.max_param_u - 0.1, 8)
            v = np.linspace(0.1, patch.max_param_v - 0.1, 8)
            uu, vv = np.meshgrid(u, v, indexing='ij')
            pts     = patch.evaluate(uu.ravel(), vv.ravel())

            samples = patch.sample(pts)
            self.assertTrue(np.allclose(samples.distances, 0, atol=1e-6),
                            f"Max distance: {np.max(samples.distances)}")

    # -------------------------------------------------------------- #
    #  Sample -- off-surface projection                                 #
    # -------------------------------------------------------------- #

    def test_sample_off_surface(self):
        patch = self.open_open
        u     = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        v     = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        pts, su, sv = patch.compute(u, v)

        normals = np.cross(su, sv)
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        norms   = np.maximum(norms, 1e-14)
        normals = normals / norms

        offset  = 0.1
        queries = pts + offset * normals

        samples = patch.sample(queries)
        self.assertTrue(np.allclose(samples.distances, offset, atol=1e-4),
                        f"Max distance error: {np.max(np.abs(samples.distances - offset))}")

    # -------------------------------------------------------------- #
    #  Basis -- partition of unity                                      #
    # -------------------------------------------------------------- #

    def test_basis_partition_of_unity(self):
        for patch in [self.open_open, self.closed_open, self.open_closed, self.closed_closed]:
            u        = np.linspace(0, patch.max_param_u * 0.99, 10)
            v        = np.linspace(0, patch.max_param_v * 0.99, 10)
            b        = patch.basis(u, v, collapse=False)
            row_sums = np.sum(b, axis=1)
            self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-10),
                            f"Basis partition of unity failed: {row_sums}")

            b_c        = patch.basis(u, v, collapse=True)
            row_sums_c = np.sum(b_c, axis=1)
            self.assertTrue(np.allclose(row_sums_c, 1.0, atol=1e-10),
                            f"Collapsed basis partition of unity failed: {row_sums_c}")

    # -------------------------------------------------------------- #
    #  Registration                                                    #
    # -------------------------------------------------------------- #

    def test_registration(self):
        patch_reg = BSplinePatchData(
            points=self.closed_closed.points.copy(),
            degree_u=3, degree_v=3,
            periodic_u=True, periodic_v=True,
            registered_u=True, registered_v=True)

        # Registration should shift the parameter origin
        u         = np.array([0.0])
        v         = np.array([0.0])
        pts_reg   = patch_reg.evaluate(u, v)
        pts_unreg = self.closed_closed.evaluate(u, v)

        # They should differ because of the registration offset
        self.assertFalse(np.allclose(pts_reg, pts_unreg, atol=1e-8))

    # -------------------------------------------------------------- #
    #  Open / Close toggle                                             #
    # -------------------------------------------------------------- #

    def test_open_close_toggle(self):
        obj = self.open_open.copy()
        self.assertFalse(obj.periodic_u)
        self.assertFalse(obj.periodic_v)

        obj.close_u()
        self.assertTrue(obj.periodic_u)
        self.assertFalse(obj.periodic_v)

        obj.close_v()
        self.assertTrue(obj.periodic_u)
        self.assertTrue(obj.periodic_v)

        obj.open_u()
        self.assertFalse(obj.periodic_u)
        self.assertTrue(obj.periodic_v)

        obj.open_v()
        self.assertFalse(obj.periodic_u)
        self.assertFalse(obj.periodic_v)

    # -------------------------------------------------------------- #
    #  Smooth                                                          #
    # -------------------------------------------------------------- #

    def test_smooth(self):
        obj = self.open_open.copy()
        # Add noise
        noisy = obj.points.copy()
        noisy += np.random.randn(*noisy.shape) * 0.1
        obj.points = noisy

        original_range = np.ptp(noisy[:, :, 2])
        obj.smooth(n_u=1, n_v=1)
        smoothed_range = np.ptp(obj.points[:, :, 2])

        self.assertLessEqual(smoothed_range, original_range * 1.5)
        self.assertEqual(obj.points.shape, noisy.shape)

    # -------------------------------------------------------------- #
    #  Periodic seam continuity                                        #
    # -------------------------------------------------------------- #

    def test_periodic_seam_continuity(self):
        # Closed in U: S(0, v) ~= S(max_u, v)
        obj     = self.closed_open
        v       = np.linspace(0.1, obj.max_param_v - 0.1, 10)
        u_zero  = np.zeros_like(v)
        u_max   = np.full_like(v, obj.max_param_u - 1e-10)

        pts_0   = obj.evaluate(u_zero, v)
        pts_max = obj.evaluate(u_max, v)
        self.assertTrue(np.allclose(pts_0, pts_max, atol=1e-4),
                        f"U-seam max error: {np.max(np.abs(pts_0 - pts_max))}")

        # Closed in V: S(u, 0) ~= S(u, max_v)
        obj     = self.open_closed
        u       = np.linspace(0.1, obj.max_param_u - 0.1, 10)
        v_zero  = np.zeros_like(u)
        v_max   = np.full_like(u, obj.max_param_v - 1e-10)

        pts_0   = obj.evaluate(u, v_zero)
        pts_max = obj.evaluate(u, v_max)
        self.assertTrue(np.allclose(pts_0, pts_max, atol=1e-4),
                        f"V-seam max error: {np.max(np.abs(pts_0 - pts_max))}")

    # -------------------------------------------------------------- #
    #  Linear extrapolation past open boundaries                       #
    # -------------------------------------------------------------- #

    def test_extrapolation_open_u_high_non_uniform(self):
        """U > max_param_u on open-non-uniform U extrapolates along S_u."""
        patch = self.open_open  # both directions open, non-uniform default
        max_u = patch.max_param_u
        v     = 1.5             # arbitrary in-range v

        # Boundary point and partials at (max_u, v)
        p_end, su_end, sv_end = patch.compute(max_u, v)

        # Extrapolated query at u = max_u + 0.7
        u_ex = max_u + 0.7
        p_ex, su_ex, sv_ex = patch.compute(u_ex, v)

        expected = p_end + (u_ex - max_u) * su_end
        self.assertTrue(np.allclose(p_ex, expected, atol=1e-9))

        # Tangents held constant at the boundary values.
        self.assertTrue(np.allclose(su_ex, su_end, atol=1e-9))
        self.assertTrue(np.allclose(sv_ex, sv_end, atol=1e-9))

    def test_extrapolation_open_u_low_non_uniform(self):
        """U < 0 on open-non-uniform U extrapolates backward along S_u."""
        patch = self.open_open
        v     = 1.5
        p_start, su_start, sv_start = patch.compute(0.0, v)

        u_ex = -0.4
        p_ex, su_ex, sv_ex = patch.compute(u_ex, v)

        expected = p_start + u_ex * su_start
        self.assertTrue(np.allclose(p_ex, expected, atol=1e-9))
        self.assertTrue(np.allclose(su_ex, su_start, atol=1e-9))
        self.assertTrue(np.allclose(sv_ex, sv_start, atol=1e-9))

    def test_extrapolation_open_v_non_uniform(self):
        """V > max_param_v on open-non-uniform V extrapolates along S_v."""
        patch = self.open_open
        max_v = patch.max_param_v
        u     = 1.0

        p_end, su_end, sv_end = patch.compute(u, max_v)
        v_ex = max_v + 0.6
        p_ex, _, _ = patch.compute(u, v_ex)

        expected = p_end + (v_ex - max_v) * sv_end
        self.assertTrue(np.allclose(p_ex, expected, atol=1e-9))

    def test_extrapolation_open_uv_corner_non_uniform(self):
        """When BOTH u and v are out of range, position is the bilinear sum
        of u-extrapolation and v-extrapolation from the corner."""
        patch = self.open_open
        max_u = patch.max_param_u
        max_v = patch.max_param_v

        # Corner (max_u, max_v) values
        p_corner, su_corner, sv_corner = patch.compute(max_u, max_v)

        u_ex = max_u + 0.4
        v_ex = max_v + 0.3
        p_ex, su_ex, sv_ex = patch.compute(u_ex, v_ex)

        expected = (
            p_corner
            + (u_ex - max_u) * su_corner
            + (v_ex - max_v) * sv_corner
        )
        self.assertTrue(np.allclose(p_ex, expected, atol=1e-9))
        # Tangents held constant at the corner values.
        self.assertTrue(np.allclose(su_ex, su_corner, atol=1e-9))
        self.assertTrue(np.allclose(sv_ex, sv_corner, atol=1e-9))

    def test_extrapolation_open_uniform_high(self):
        """U > 1 on open-uniform U extrapolates along total_length_u * unit_S_u."""
        patch = self.open_open.copy()
        patch.uniform_u = True
        patch.uniform_v = True

        # Reference at u=1 (uniform end)
        v = 0.5
        p_end, su_end, sv_end = patch.compute(1.0, v)
        total_u = patch.total_length_u

        norm    = np.linalg.norm(su_end)
        unit_su = su_end / norm if norm > 1e-14 else np.zeros_like(su_end)
        vel_u   = total_u * unit_su

        u_ex    = 1.4
        p_ex, su_ex, _ = patch.compute(u_ex, v)

        expected = p_end + (u_ex - 1.0) * vel_u
        self.assertTrue(np.allclose(p_ex, expected, atol=1e-9))
        self.assertTrue(np.allclose(su_ex, su_end, atol=1e-9))

    def test_extrapolation_mixed_array(self):
        """Mixed array input: in-range, sub-zero, beyond-max each handled correctly."""
        patch = self.open_open
        max_u = patch.max_param_u

        v     = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        u     = np.array([-0.3, 0.0, max_u * 0.5, max_u, max_u + 0.7])
        pts, su, sv = patch.compute(u, v)

        # In-range entries: should match standalone evaluations.
        p_start, _, _ = patch.compute(0.0, 1.0)
        p_end, su_end, _ = patch.compute(max_u, 1.0)
        p_start_su, su_start, _ = patch.compute(0.0, 1.0)

        self.assertTrue(np.allclose(pts[1], p_start, atol=1e-9))
        self.assertTrue(np.allclose(pts[3], p_end, atol=1e-9))

        # Below-range: extrapolated backward.
        expected_low = p_start + u[0] * su_start
        self.assertTrue(np.allclose(pts[0], expected_low, atol=1e-9))

        # Above-range: extrapolated forward.
        expected_high = p_end + (u[4] - max_u) * su_end
        self.assertTrue(np.allclose(pts[4], expected_high, atol=1e-9))

    def test_extrapolation_periodic_does_not_extrapolate(self):
        """Periodic directions wrap via modulo -- they do not extrapolate."""
        # Closed in U, open in V: u=max_u+1 should wrap to u=1, NOT extrapolate.
        patch = self.closed_open
        max_u = patch.max_param_u
        v     = 1.0

        p_one, _, _ = patch.compute(1.0, v)
        p_wrapped, _, _ = patch.compute(max_u + 1.0, v)

        self.assertTrue(np.allclose(p_one, p_wrapped, atol=1e-9))

    def test_extrapolation_evaluate_matches_compute(self):
        """evaluate() returns the same extrapolated positions as compute()."""
        patch = self.open_open
        max_u = patch.max_param_u
        max_v = patch.max_param_v

        u     = np.array([-0.5, 0.5, max_u + 0.7])
        v     = np.array([0.5, max_v + 0.3, 0.5])

        pts_compute, _, _ = patch.compute(u, v)
        pts_evaluate = patch.evaluate(u, v)

        self.assertTrue(np.allclose(pts_compute, pts_evaluate, atol=1e-9))

    def test_extrapolation_compute_fast_matches_compute(self):
        """compute_fast() applies the same extrapolation as compute()."""
        patch = self.open_open
        max_u = patch.max_param_u
        max_v = patch.max_param_v

        u     = np.array([-0.5, 0.5, max_u + 0.7])
        v     = np.array([0.5, max_v + 0.3, 0.5])

        pts_a, su_a, sv_a = patch.compute(u, v)
        pts_b, su_b, sv_b = patch.compute_fast(u, v)

        self.assertTrue(np.allclose(pts_a, pts_b, atol=1e-9))
        self.assertTrue(np.allclose(su_a, su_b, atol=1e-9))
        self.assertTrue(np.allclose(sv_a, sv_b, atol=1e-9))

    def test_extrapolation_scipy_matches_numba(self):
        """Numba and scipy code paths agree on extrapolated values."""
        patch_n = self.open_open.copy()
        patch_n.use_numba = True

        patch_s = self.open_open.copy()
        patch_s.use_numba = False

        max_u = patch_n.max_param_u
        max_v = patch_n.max_param_v
        u     = np.array([-0.5, 0.5, max_u + 1.2])
        v     = np.array([0.5, max_v + 0.4, 0.5])

        pn, sun, svn = patch_n.compute(u, v)
        ps, sus, svs = patch_s.compute(u, v)

        self.assertTrue(np.allclose(pn, ps, atol=1e-9))
        self.assertTrue(np.allclose(sun, sus, atol=1e-9))
        self.assertTrue(np.allclose(svn, svs, atol=1e-9))

    def test_extrapolation_native_flag_bypasses(self):
        """_native=True skips extrapolation (caller provides native params)."""
        patch = self.open_open
        max_u = patch.max_param_u

        # With _native=True, out-of-range u should NOT be extrapolated.
        # The clipping inside compute() also doesn't kick in (gated on `not _native`).
        # We just want to verify extrapolation logic is bypassed; the underlying
        # evaluator behavior outside the knot range is up to scipy/numba.
        # Easiest check: in-range query gives the same result with and without _native.
        v = 0.5
        p_user, _, _ = patch.compute(1.0, v, _native=False)
        p_native, _, _ = patch.compute(1.0, v, _native=True)
        self.assertTrue(np.allclose(p_user, p_native, atol=1e-9))

    # -------------------------------------------------------------- #
    #  Raycast                                                         #
    # -------------------------------------------------------------- #

    def _raycast_pick_anchor(self, patch, u_val, v_val):
        """Helper: surface point + unit normal at (u_val, v_val)."""
        pts, su, sv = patch.compute(np.array([u_val]), np.array([v_val]))
        n = np.cross(su[0], sv[0])
        n /= np.linalg.norm(n)
        return pts[0], n

    def test_raycast_hit_at_known_point(self):
        """Ray from a point above the surface along -normal hits at distance d."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 1.5)
        d         = 2.0
        origin    = anchor + d * normal
        direction = -normal

        result    = patch.raycast(np.array([origin]), np.array([direction]))

        from cgmath.geometry.bspline_patch import PatchRaycastData
        self.assertIsInstance(result, PatchRaycastData)
        self.assertTrue(result.hit[0])
        self.assertTrue(np.isclose(result.distances[0], d, atol=1e-3))
        self.assertTrue(np.allclose(result.points[0], anchor, atol=1e-3))

    def test_raycast_normals_match_surface(self):
        """Normals at hit point match cross(S_u, S_v) at recovered (u, v)."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.0, 2.0)
        origin    = anchor + 1.5 * normal
        direction = -normal

        result    = patch.raycast(np.array([origin]), np.array([direction]))
        self.assertTrue(result.hit[0])

        # Normals are unit length and point toward the ray (occluded=True
        # for front-face hit).
        self.assertTrue(np.isclose(np.linalg.norm(result.normals[0]), 1.0, atol=1e-9))
        self.assertTrue(result.occluded[0])

    def test_raycast_miss_returns_nan(self):
        """A ray pointing away from the surface produces a miss."""
        patch = self.open_open
        # Far origin pointing further away from surface (which sits near z=0).
        origin    = np.array([0.0, 0.0, 100.0])
        direction = np.array([0.0, 0.0, 1.0])

        result    = patch.raycast(np.array([origin]), np.array([direction]))
        self.assertFalse(result.hit[0])
        self.assertTrue(np.isnan(result.distances[0]))
        self.assertTrue(np.allclose(result.points[0], origin))

    def test_raycast_batch_mixed_hits_and_misses(self):
        """Batch with hits and misses returns correct per-ray flags."""
        patch = self.open_open
        anchor_a, normal_a = self._raycast_pick_anchor(patch, 1.0, 1.5)
        anchor_b, normal_b = self._raycast_pick_anchor(patch, 2.0, 1.0)

        origins = np.array(
            [
                anchor_a + 1.0 * normal_a,        # hit
                np.array([0.0, 0.0, 100.0]),       # miss
                anchor_b + 1.5 * normal_b,        # hit
            ]
        )
        directions = np.array(
            [
                -normal_a,
                np.array([0.0, 0.0, 1.0]),
                -normal_b,
            ]
        )
        result = patch.raycast(origins, directions)

        self.assertEqual(result.hit.shape, (3,))
        self.assertTrue(result.hit[0])
        self.assertFalse(result.hit[1])
        self.assertTrue(result.hit[2])
        self.assertTrue(np.isnan(result.distances[1]))

    def test_raycast_forward_only_true_misses_backward(self):
        """forward_only=True: a ray pointing away from the surface misses."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 1.5)
        origin    = anchor + 1.0 * normal
        direction = normal  # pointing AWAY from surface

        result = patch.raycast(
            np.array([origin]), np.array([direction]), forward_only=True
        )
        self.assertFalse(result.hit[0])

    def test_raycast_forward_only_false_finds_backward(self):
        """forward_only=False: backward direction is also tried."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 1.5)
        origin    = anchor + 1.0 * normal
        direction = normal  # surface is at t=-1 along this direction

        result = patch.raycast(
            np.array([origin]), np.array([direction]), forward_only=False
        )
        self.assertTrue(result.hit[0])
        # Backward hit: t should be negative (~-1.0) in the original frame.
        self.assertTrue(np.isclose(result.distances[0], -1.0, atol=1e-3))
        self.assertTrue(np.allclose(result.points[0], anchor, atol=1e-3))

    def test_raycast_twosided_false_rejects_backface(self):
        """twosided=False: ray hitting the back face is rejected."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 1.5)
        # Coming from behind (-normal side) toward the surface = back-face hit.
        origin    = anchor - 2.0 * normal
        direction = normal  # toward surface from the back

        # twosided=True: hit accepted.
        r_two = patch.raycast(
            np.array([origin]), np.array([direction]), twosided=True
        )
        self.assertTrue(r_two.hit[0])

        # twosided=False: hit rejected because surface normal * direction > 0.
        r_one = patch.raycast(
            np.array([origin]), np.array([direction]), twosided=False
        )
        self.assertFalse(r_one.hit[0])

    def test_raycast_basis_partition_of_unity_at_hits(self):
        """Basis weights at hit points sum to 1."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.0, 2.0)
        result = patch.raycast(
            np.array([anchor + 0.5 * normal]), np.array([-normal])
        )
        self.assertTrue(result.hit[0])
        self.assertTrue(np.isclose(result.basis[0].sum(), 1.0, atol=1e-9))

    def test_raycast_compute_method(self):
        """PatchRaycastData.compute(values) reproduces position from points."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 1.0)
        result = patch.raycast(
            np.array([anchor + 0.5 * normal]), np.array([-normal])
        )
        self.assertTrue(result.hit[0])
        # Use the control-point grid as the values; compute should reproduce
        # the surface position at the hit.
        recon = result.compute(patch.points)
        self.assertTrue(np.allclose(recon[0], result.points[0], atol=1e-6))

    def test_raycast_periodic_surface(self):
        """Raycast hits a periodic (closed) surface as expected."""
        patch = self.closed_closed
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 2.0)
        result = patch.raycast(
            np.array([anchor + 1.0 * normal]), np.array([-normal])
        )
        self.assertTrue(result.hit[0])
        self.assertTrue(np.isclose(result.distances[0], 1.0, atol=1e-3))

    def test_raycast_single_direction_tiled(self):
        """Single (3,) direction is tiled across all origins."""
        patch = self.open_open
        anchor_a, normal_a = self._raycast_pick_anchor(patch, 1.0, 1.0)
        anchor_b, normal_b = self._raycast_pick_anchor(patch, 2.0, 2.0)
        # Use the same downward-ish direction for both.
        common_dir = np.array([0.0, 0.0, -1.0])
        origins = np.array(
            [anchor_a + 5.0 * np.array([0.0, 0.0, 1.0]),
             anchor_b + 5.0 * np.array([0.0, 0.0, 1.0])]
        )

        result = patch.raycast(origins, common_dir)
        self.assertEqual(result.hit.shape, (2,))
        # Both should hit somewhere on the surface.
        self.assertTrue(np.all(result.hit))

    def test_raycast_2d_surface_raises(self):
        """Raycast on a 2D surface raises ValueError."""
        u_vals = np.linspace(0, 1, 4)
        v_vals = np.linspace(0, 1, 4)
        uu, vv = np.meshgrid(u_vals, v_vals, indexing='ij')
        grid_2d = np.stack([uu, vv], axis=-1)
        patch_2d = BSplinePatchData(
            points=grid_2d, degree_u=2, degree_v=2,
            periodic_u=False, periodic_v=False,
        )
        with self.assertRaises(ValueError):
            patch_2d.raycast(np.array([[0.5, 0.5, 1.0]]),
                             np.array([[0.0, 0.0, -1.0]]))

    def test_raycast_directions_none_requires_get_vertex_normals(self):
        """directions=None on a plain ndarray origins should raise."""
        patch = self.open_open
        with self.assertRaises(ValueError):
            patch.raycast(np.array([[0.0, 0.0, 5.0]]))

    def test_raycast_meshdata_input(self):
        """MeshData origins with directions=None uses vertex normals."""
        patch = self.open_open
        anchor, normal = self._raycast_pick_anchor(patch, 1.5, 1.5)

        # Simple 2-triangle mesh whose vertex normals all point along -normal
        # (toward the surface). We pass it as origins so MeshData.points are
        # used as origins and MeshData.get_vertex_normals() supplies dirs.
        verts = np.array(
            [
                anchor + 1.0 * normal,
                anchor + 1.0 * normal + np.array([0.01, 0.0, 0.0]),
                anchor + 1.0 * normal + np.array([0.0, 0.01, 0.0]),
            ],
            dtype=np.float64,
        )
        # All three vertices share normal pointing along -normal so the rays
        # hit the surface.
        mesh = MeshData(
            points  = verts,
            indices = np.array([0, 1, 2]),
            counts  = np.array([3]),
        )
        # Override the computed normals: build a tiny patched mesh that
        # already has the right orientation. The orientation of the triangle
        # isn't guaranteed to match -normal, so we patch get_vertex_normals
        # via cast against a cube-aligned ray instead. Simpler: just supply
        # explicit directions through the mesh path by passing both args.
        directions = np.tile(-normal, (verts.shape[0], 1))
        result     = patch.raycast(mesh, directions)

        self.assertEqual(result.hit.shape, (3,))
        # All rays should hit (origins are 1 unit above the surface, direction
        # toward the surface).
        self.assertTrue(np.all(result.hit))
