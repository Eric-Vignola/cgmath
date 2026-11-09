"""Tests for the skins a Scene carries from a glb, and for posing them.

Two things here are worth more than the rest.

The first is that a posed mesh is always deformed from the BIND pose. The
scene never holds a deformed mesh: ``placed_meshes`` hands out a copy moved
to the pose, so a sequence never compounds and the meshes stay as loaded.

The second is how a mesh gets paired with its weights: a glb is paired
positionally because trimesh renames a duplicate ('part', 'part_1') while
the skin reader does not. A name lookup would bind a mesh to some other
mesh's weights, which looks like geometry and reads like a rig bug, so the
wrong pairing is covered explicitly.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import unittest
import warnings

import numpy as np
from cgmath.geometry.skin_weights import SkinData
from cgmath.hierarchy import ClipData, HierarchyData
from cgmath.render.scene import Scene

try:
    import pygltflib
    from pygltflib import Mesh, Node, Skin, TRIANGLE_FAN
except ImportError:
    pygltflib = None

try:
    import trimesh
except ImportError:
    trimesh = None

if pygltflib is not None:
    from cgmath._tests.test_skin_glb import Builder, one_hot, triangle


HAVE_GLB = pygltflib is not None and trimesh is not None


# ------------------------------- glb fixtures ------------------------------ #


def _joint_nodes():
    return [
        Node(name="root", children=[1]),
        Node(name="spine", children=[2]),
        Node(name="head"),
    ]


def write_glb(path, mesh_names, skinned=True, modes=None, geo_names=None):
    """one mesh per name, each a triangle bound one vertex per joint"""
    builder = Builder()
    meshes  = []
    for i, mesh_name in enumerate(mesh_names):
        mode = None if modes is None else modes[i]
        if skinned:
            joints, weights = one_hot([0, 1, 2])
            args = (triangle(3.0 * i), joints, weights)
        else:
            args = (triangle(3.0 * i),)
        primitive = (
            builder.primitive(*args)
            if mode is None
            else builder.primitive(*args, mode=mode)
        )
        meshes.append(Mesh(primitives=[primitive], name=mesh_name))

    nodes = _joint_nodes()
    for i, mesh_name in enumerate(mesh_names):
        geo = mesh_name if geo_names is None else geo_names[i]
        nodes.append(Node(name=f"{geo}_geo", mesh=i, skin=i if skinned else None))

    binds = builder.identity_binds(3)
    skins = (
        [Skin(joints=[0, 1, 2], inverseBindMatrices=binds) for _ in mesh_names]
        if skinned
        else []
    )
    return builder.write(path, meshes, nodes, skins)


def _row_translate(x, y, z):
    matrix        = np.eye(4, dtype=np.float64)
    matrix[3, :3] = (x, y, z)
    return matrix


def write_bound_glb(path, bind_y, node_y):
    """one triangle, one vertex per joint, with the bind given separately

    'spine' (and 'head' under it) bind *bind_y* up but their nodes sit
    *node_y* up, so the file's authored binds and its node transforms
    disagree exactly the way an animated file's do.
    """
    builder = Builder()
    joints, weights = one_hot([0, 1, 2])
    primitive = builder.primitive(triangle(), joints, weights)
    meshes    = [Mesh(primitives=[primitive], name="body")]

    nodes = [
        Node(name="root", children=[1]),
        Node(name="spine", children=[2], translation=[0.0, node_y, 0.0]),
        Node(name="head"),
        Node(name="body_geo", mesh=0, skin=0),
    ]

    world = [
        np.eye(4),
        _row_translate(0.0, bind_y, 0.0),
        _row_translate(0.0, bind_y, 0.0),
    ]
    binds = np.asarray([np.linalg.inv(x).ravel() for x in world], dtype=np.float32)
    skins = [
        Skin(joints=[0, 1, 2], inverseBindMatrices=builder.add(binds, kind="MAT4"))
    ]
    return builder.write(path, meshes, nodes, skins)


class GlbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name="hero.glb"):
        return os.path.join(self.tmp.name, name)

    def skinned(self, names=("body",), **kwargs):
        path = self.path()
        write_glb(path, list(names), **kwargs)
        return path

    def bent_rig(self, path, degrees=90.0):
        """the file's rig with 'spine' rotated, so posing actually moves points"""
        rig               = HierarchyData.load_glb(path)
        index             = list(rig.name).index("spine")
        rig[index].rotate = np.array([0.0, 0.0, degrees])
        return rig

    @staticmethod
    def posed(scene, rig):
        """the scene's first mesh placed by ``rig``: the scene's hierarchy
        swapped for it, as a pose, the bind pose kept as the deformer's"""
        points = scene._deformer(scene.meshes[0], scene.skins[0]).apply(rig, scene.meshes[0].points)
        return np.asarray(points, dtype=np.float64)


# ------------------------------- attachment -------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbAttachment(GlbCase):
    def test_load_glb_attaches_a_skin_and_the_bind_pose(self):
        scene = Scene.load_glb(self.skinned())

        self.assertEqual(len(scene.skins), 1)
        self.assertIsInstance(scene.skins[0], SkinData)
        self.assertEqual(scene.skins[0].influences, ["root", "spine", "head"])
        self.assertEqual(scene.skins[0].mesh_uuid, scene.meshes[0].uuid)
        # the file's joints, the bind pose; the glb reader keeps no mesh node
        self.assertEqual(list(scene.hierarchy.name), ["root", "spine", "head"])

    def test_load_skin_false_leaves_it_unset(self):
        scene = Scene.load_glb(self.skinned(), load_skin=False)

        self.assertIsNone(scene.skins)
        self.assertIsNone(scene.hierarchy)

    def test_an_unskinned_glb_is_not_a_warning(self):
        path = self.path()
        write_glb(path, ["body"], skinned=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene = Scene.load_glb(path)

        self.assertIsNone(scene.skins)
        # most glbs are unskinned; warning on all of them would be noise
        self.assertEqual([str(w.message) for w in caught], [])

    def test_an_unskinned_glb_trimesh_thins_out_is_still_not_a_warning(self):
        """no skins anywhere means nothing can be mis-bound"""
        path = self.path()
        write_glb(path, ["m0", "m1"], skinned=False, modes=[None, TRIANGLE_FAN])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene = Scene.load_glb(path)

        self.assertIsNone(scene.skins)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_scene_load_glb_attaches_one_skin_per_mesh(self):
        scene = Scene.load_glb(self.skinned(names=("torso", "limb")))

        self.assertEqual(len(scene.meshes), 2)
        self.assertEqual([skin.mesh_uuid for skin in scene.skins], [mesh.uuid for mesh in scene.meshes])

    def test_scene_load_skin_false_leaves_them_unset(self):
        scene = Scene.load_glb(self.skinned(names=("torso", "limb")), load_skin=False)

        self.assertIsNone(scene.skins)


# -------------------------------- pairing ---------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbPairing(GlbCase):
    def test_duplicate_names_still_pair_by_position(self):
        """trimesh renames the second mesh, the skin reader does not"""
        path = self.path()
        write_glb(path, ["part", "part"], geo_names=["a", "b"])

        scene = Scene.load_glb(path)

        self.assertEqual([m.name for m in scene.meshes], ["part", "part_1"])
        self.assertEqual([s.mesh_uuid for s in scene.skins], [m.uuid for m in scene.meshes])
        # each mesh keeps its OWN geometry, so the pairing is positional
        self.assertAlmostEqual(float(scene.meshes[0].points[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(scene.meshes[1].points[:, 0].min()), 300.0)

    def test_a_static_mesh_gets_no_skin_and_the_other_still_does(self):
        """the glb reader holds a None slot, so positions stay in step"""
        builder = Builder()
        joints, weights = one_hot([0, 1, 2])
        meshes = [
            Mesh(
                primitives = [builder.primitive(triangle(0.0), joints, weights)],
                name       = "skinned",
            ),
            Mesh(primitives=[builder.primitive(triangle(3.0))], name="static"),
        ]
        nodes = _joint_nodes() + [
            Node(name="geo0", mesh=0, skin=0),
            Node(name="geo1", mesh=1),
        ]
        binds = builder.identity_binds(3)
        path  = self.path()
        builder.write(
            path, meshes, nodes, [Skin(joints=[0, 1, 2], inverseBindMatrices=binds)]
        )

        scene   = Scene.load_glb(path)
        by_name = {m.name: m for m in scene.meshes}

        self.assertEqual([s.mesh_uuid for s in scene.skins], [by_name["skinned"].uuid])

    def test_a_primitive_trimesh_drops_warns_instead_of_mis_binding(self):
        """trimesh skips the TRIANGLE_FAN, the skin reader keeps it: 2 meshes
        against 3 skins, so index 1 would pair 'm2' with 'm1''s weights"""
        path = self.path()
        write_glb(path, ["m0", "m1", "m2"], modes=[None, TRIANGLE_FAN, None])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene = Scene.load_glb(path)

        self.assertEqual([m.name for m in scene.meshes], ["m0", "m2"])
        self.assertIsNone(scene.skins)
        self.assertTrue(
            any("cannot be paired up" in str(w.message) for w in caught),
            f"expected a pairing warning, got {[str(w.message) for w in caught]}",
        )


# --------------------------------- posing ---------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestPose(GlbCase):
    def test_a_clip_poses_the_mesh(self):
        """the hierarchy is the bind pose; a clip over the rig poses it"""
        path  = self.skinned()
        scene = Scene.load_glb(path)
        rest  = np.array(scene.meshes[0].points, copy=True)

        scene.clip = ClipData(self.bent_rig(path), frames=1)
        placed, world = scene.placed_meshes()[0]

        self.assertFalse(np.allclose(rest, placed.points))
        np.testing.assert_array_equal(world, np.eye(4))  # a skinned mesh's points are world points
        np.testing.assert_allclose(placed.points, self.posed(scene, self.bent_rig(path)), atol=1e-9)

    def test_the_bind_pose_is_a_fixed_point(self):
        path   = self.skinned()
        scene  = Scene.load_glb(path)
        before = np.array(scene.meshes[0].points, copy=True)

        placed, _ = scene.placed_meshes()[0]

        np.testing.assert_allclose(before, placed.points, atol=1e-9)

    def test_posing_twice_does_not_compound(self):
        """the scene's mesh is never the posed one: the same pose twice lands
        in the same place, and the mesh stays as loaded"""
        path  = self.skinned()
        scene = Scene.load_glb(path)
        rest  = np.array(scene.meshes[0].points, copy=True)

        scene.clip = ClipData(self.bent_rig(path), frames=1)
        once       = np.array(scene.placed_meshes()[0][0].points, copy=True)
        twice      = scene.placed_meshes()[0][0].points

        np.testing.assert_allclose(once, twice, atol=1e-9)
        np.testing.assert_array_equal(scene.meshes[0].points, rest)

    def test_a_sequence_is_order_independent(self):
        path    = self.skinned()
        rig_a   = self.bent_rig(path, 30.0)
        rig_b   = self.bent_rig(path, 75.0)
        scene   = Scene.load_glb(path)

        direct  = self.posed(scene, rig_b)
        after_a = self.posed(scene, rig_a)
        after_b = self.posed(scene, rig_b)

        self.assertFalse(np.allclose(direct, after_a))
        np.testing.assert_allclose(direct, after_b, atol=1e-9)

    def test_a_render_caches_by_the_pose(self):
        """the render signature holds every joint's world when a mesh is
        skinned, so a frame kept at one pose goes when a joint moves, in
        the bind pose or in the clip that poses it"""
        path        = self.skinned()
        scene       = Scene.load_glb(path)
        scene.frame = object()
        self.assertIsNotNone(scene.frame)
        scene.hierarchy["spine"].rotate = np.array([0.0, 0.0, 45.0])
        self.assertIsNone(scene.frame)

        scene.clip  = ClipData(self.bent_rig(path), frames=1)
        scene.frame = object()
        self.assertIsNotNone(scene.frame)
        scene.clip.active["spine"].rotate = np.array([0.0, 0.0, 10.0])
        self.assertIsNone(scene.frame)

    def test_a_scene_without_a_hierarchy_cannot_pose_a_skin(self):
        scene           = Scene.load_glb(self.skinned())
        scene.hierarchy = None
        with self.assertRaisesRegex(ValueError, "is skinned, which takes the bind pose"):
            scene.placed_meshes()


# ------------------------------- persistence ------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestPersistence(GlbCase):
    def round_trips(self, scene):
        yield "to_dict", Scene.from_dict(scene.to_dict())
        yield "pickle", pickle.loads(pickle.dumps(scene))
        yield "copy", scene.copy()

    def test_the_skin_survives_every_round_trip(self):
        scene = Scene.load_glb(self.skinned())
        for label, restored in self.round_trips(scene):
            with self.subTest(label):
                self.assertIsInstance(restored.skins[0], SkinData)
                self.assertEqual(restored.skins[0].influences, ["root", "spine", "head"])
                self.assertEqual(restored.skins[0].mesh_uuid, restored.meshes[0].uuid)

    def test_a_restored_scene_still_poses(self):
        path     = self.skinned()
        scene    = Scene.load_glb(path)
        expected = self.posed(scene, self.bent_rig(path))

        for label, restored in self.round_trips(scene):
            with self.subTest(label):
                np.testing.assert_allclose(expected, self.posed(restored, self.bent_rig(path)), atol=1e-9)


# -------------------------------- the bind --------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbBindPose(GlbCase):
    """the bind pose is the file's inverse bind matrices, not its nodes, in
    the unit the points were read in"""

    def test_bind_pose_comes_from_the_file_not_the_nodes(self):
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=3.0)

        scene = Scene.load_glb(path, scale_factor=1.0)

        # the file says the rig bound 1 up; the nodes sit 3 up because that
        # is where the take left them
        np.testing.assert_allclose(scene.hierarchy["spine"].world_matrix[3, :3], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(HierarchyData.load_glb(path, scale_factor=1.0)["spine"].world_matrix[3, :3], [0.0, 3.0, 0.0], atol=1e-6)

    def test_posing_moves_by_the_bind_to_node_difference(self):
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=3.0)

        scene = Scene.load_glb(path, scale_factor=1.0)
        rest  = np.asarray(scene.meshes[0].points, dtype=float)
        moved = self.posed(scene, HierarchyData.load_glb(path, scale_factor=1.0)) - rest

        # sorted because trimesh owes us no particular vertex order: the
        # 'root' vertex holds still and the other two travel the 2 unit gap
        np.testing.assert_allclose(sorted(moved[:, 1]), [0.0, 2.0, 2.0], atol=1e-6)

    def test_the_bind_is_scaled_with_the_mesh(self):
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=1.0)

        scene = Scene.load_glb(path, scale_factor=100.0)

        np.testing.assert_allclose(scene.hierarchy["spine"].world_matrix[3, :3], [0.0, 100.0, 0.0], atol=1e-4)

    def test_posing_at_the_bind_pose_does_not_move_anything(self):
        """bind and pose are the same skeleton here, so every skin matrix has
        to come out identity; binds left in the file's units would miss by
        the scale factor"""
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=1.0)

        scene = Scene.load_glb(path, scale_factor=100.0)
        rest  = np.asarray(scene.meshes[0].points, dtype=float).copy()

        np.testing.assert_allclose(self.posed(scene, HierarchyData.load_glb(path, scale_factor=100.0)), rest, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
