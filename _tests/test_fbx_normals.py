"""Shading normals as Maya derives them: its rule, measured.

Where a mesh holds no authored normal, Maya 2025 derives one for each face
corner. The faces around the corner's vertex that soft edges join are a fan;
each face of the fan gives its Newell vector (its normal, as long as twice
the area of the face's outline seen along it) times the angle of its corner
at the vertex, and the sum is made unit. A hard edge parts two fans; the
edges of a border are marked hard and part nothing. A locked normal is the
one authored, whatever the faces say.

MeshData computes the same normals: ``get_vertex_normals()`` with every edge
soft, ``set_normals(hard_edges=...)`` inside the groups the hard edges
leave. The tests read what Maya 2025 held on 15 meshes, recorded once
(maya_fixtures, case derived_normals), and need no Maya.
"""

import unittest

import numpy as np
from cgmath._tests.test_skeleton_fbx import _Recorded
from cgmath.geometry import MeshData

# a component; Maya hands its normals out as floats
TOLERANCE = 1e-5


def held_mesh(case, name: str) -> MeshData:
    """the recorded mesh: Maya's points, face counts and corner vertices, no normals"""
    return MeshData(points=case.array(f"{name}/points"), indices=case.array(f"{name}/indices"), counts=case.array(f"{name}/counts"))


def hard_edges(case, name: str, mesh: MeshData) -> np.ndarray:
    """the edges Maya marked hard, as the mesh's own edge ids"""
    ours   = {tuple(pair): edge for edge, pair in enumerate(np.sort(np.asarray(mesh.e2v), axis=1).tolist())}
    marked = np.sort(case.array(f"{name}/edges")[~case.array(f"{name}/smooth")], axis=1)
    return np.array([ours[tuple(pair)] for pair in marked.tolist()], dtype=np.int32)


class TestMayaDerivedNormals(_Recorded):
    """the normals Maya 2025 derives where none is authored, as recorded,
    against the ones MeshData computes"""

    def meshes(self, kind: str):
        case = self.case("derived_normals")
        info = case.value("info")
        return case, info, [name for name in case.value("meshes") if info[name]["kind"] == kind]

    def test_every_edge_soft(self):
        """one normal a vertex: get_vertex_normals() gives Maya's"""
        case, info, names = self.meshes("soft")
        self.assertEqual(len(names), 5)
        for name in names:
            with self.subTest(name):
                mesh    = held_mesh(case, name)
                indices = np.asarray(mesh.indices)
                maya    = case.array(f"{name}/normals")
                self.assertEqual(info[name]["locked_corners"], 0)
                self.assertEqual(info[name]["normals"], len(mesh.points))
                got = mesh.get_vertex_normals()[indices]
                np.testing.assert_allclose(got, maya, rtol=0, atol=TOLERANCE)
                # the faces bend: weighed by get_face_areas(), the area of
                # the surface and not of the outline, the normals are others
                by_surface = mesh.get_vertex_normals(_areas=mesh.get_face_areas(), force=True)[indices]
                self.assertGreater(np.abs(by_surface - maya).max(), 1e-3)

    def test_border_edges_are_marked_hard_and_part_nothing(self):
        """an open sheet softened all over: Maya marks its 18 border edges
        hard and still holds one normal a vertex"""
        case, info, names = self.meshes("soft")
        self.assertEqual([name for name in names if info[name]["hard_edges"]], ["softSheet"])
        mesh  = held_mesh(case, "softSheet")
        edges = hard_edges(case, "softSheet", mesh)
        self.assertEqual(edges.size, 18)
        faces = np.asarray(mesh.indices).reshape(-1, 4)
        pairs = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
        used  = {}
        for pair in pairs.tolist():
            used[tuple(pair)] = used.get(tuple(pair), 0) + 1
        marked = np.sort(case.array("softSheet/edges")[~case.array("softSheet/smooth")], axis=1)
        self.assertEqual(sorted(tuple(pair) for pair in marked.tolist()), sorted(pair for pair, count in used.items() if count == 1))
        mesh.set_normals(hard_edges=edges)
        slots = np.asarray(mesh.indices) if mesh.normal_indices is None else np.asarray(mesh.normal_indices)
        self.assertEqual(len(mesh.normals), len(mesh.points))
        np.testing.assert_allclose(np.asarray(mesh.normals)[slots], case.array("softSheet/normals"), rtol=0, atol=TOLERANCE)

    def test_hard_edges(self):
        """set_normals(hard_edges=...) on the edges Maya marked hard gives
        Maya's normals, in Maya's groups of corners"""
        case, info, names = self.meshes("hard")
        self.assertEqual(len(names), 9)
        for name in names:
            with self.subTest(name):
                mesh  = held_mesh(case, name)
                maya  = case.array(f"{name}/normals")
                ids   = case.array(f"{name}/normal_ids")
                edges = hard_edges(case, name, mesh)
                self.assertEqual(info[name]["locked_corners"], 0)
                self.assertEqual(edges.size, info[name]["hard_edges"])
                mesh.set_normals(hard_edges=edges)
                slots = np.asarray(mesh.normal_indices)
                np.testing.assert_allclose(np.asarray(mesh.normals)[slots], maya, rtol=0, atol=TOLERANCE)
                # two corners share a normal here where they share one in Maya
                self.assertEqual(len(mesh.normals), info[name]["normals"])
                self.assertEqual(len(set(zip(ids.tolist(), slots.tolist()))), info[name]["normals"])
                # with every edge soft the normals are others
                soft = held_mesh(case, name).get_vertex_normals()[np.asarray(mesh.indices)]
                self.assertGreater(np.abs(soft - maya).max(), 1e-3)

    def test_groups_of_several_faces_follow_the_rule(self):
        """the jittered meshes softened by an angle hold groups of several
        bent faces: their normals are not their faces' own"""
        case, info, names = self.meshes("hard")
        several = [name for name in names if info[name]["jittered"] and info[name]["normals"] < case.array(f"{name}/indices").size]
        self.assertEqual(several, ["canJittered", "coneJittered", "ballSoft30", "ringSoft45", "sheetSoft20"])
        for name in several:
            with self.subTest(name):
                mesh = held_mesh(case, name)
                flat = np.repeat(mesh.get_face_normals(), np.asarray(mesh.counts), axis=0)
                self.assertGreater(np.abs(flat - case.array(f"{name}/normals")).max(), 0.1)

    def test_a_sphere_as_maya_makes_it_is_all_hard(self):
        """Maya's polySphere: every edge hard, a normal a corner, each its
        face's own"""
        case, info, names = self.meshes("hard")
        self.assertIn("ballDefault", names)
        mesh = held_mesh(case, "ballDefault")
        self.assertFalse(case.array("ballDefault/smooth").any())
        self.assertEqual(info["ballDefault"]["normals"], np.asarray(mesh.indices).size)
        flat = np.repeat(mesh.get_face_normals(), np.asarray(mesh.counts), axis=0)
        np.testing.assert_allclose(flat, case.array("ballDefault/normals"), rtol=0, atol=TOLERANCE)

    def test_a_locked_normal_is_the_one_authored(self):
        """a soft cube with vertex 0's normal set to +Y: its three corners
        are locked and hold +Y, every other corner follows the rule"""
        case, info, names = self.meshes("locked")
        self.assertEqual(names, ["lockedCube"])
        mesh    = held_mesh(case, "lockedCube")
        indices = np.asarray(mesh.indices)
        maya    = case.array("lockedCube/normals")
        locked  = case.array("lockedCube/locked")
        self.assertEqual(info["lockedCube"]["hard_edges"], 0)
        self.assertEqual((int(locked.sum()), int((indices == 0).sum())), (3, 3))
        self.assertTrue(np.all(indices[locked] == 0))
        np.testing.assert_allclose(maya[locked], [[0.0, 1.0, 0.0]] * 3, rtol=0, atol=1e-7)
        got = mesh.get_vertex_normals()[indices]
        np.testing.assert_allclose(got[~locked], maya[~locked], rtol=0, atol=TOLERANCE)
        self.assertGreater(np.abs(got[locked] - maya[locked]).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
