"""Tests for the skin an ``Object`` carries, and for ``Object.pose``.

Two things here are worth more than the rest.

The first is that ``pose`` must deform the BIND pose every frame.  Feeding
the previous frame's result back in still produces geometry -- it just
compounds, so a limb drifts further every frame and nothing raises.  The
sequence tests below pin that down by posing twice and demanding the answer
not move.

The second is how a mesh gets paired with its weights, which the two formats
have to do in opposite ways.  A glb is paired positionally because trimesh
renames a duplicate ('part', 'part_1') while the skin reader does not, and an
fbx is paired by name because its skin reader drops unskinned meshes and the
positions shift.  Doing either one the other way round binds a mesh to some
other mesh's weights, which looks like geometry and reads like a rig bug, so
both wrong pairings are covered explicitly.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import unittest
import warnings

import numpy as np
from cgmath.geometry.deform import SkinDeformData
from cgmath.render.scene import Object, Scene
from cgmath.hierarchy import HierarchyData

try:
    import pygltflib
    from pygltflib import Mesh, Node, Skin, TRIANGLE_FAN
except ImportError:
    pygltflib = None

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import fbx
except ImportError:
    fbx = None

if pygltflib is not None:
    from cgmath._tests.test_skin_glb import Builder, one_hot, triangle

if fbx is not None:
    from cgmath._tests.test_skin_fbx import add_joint, add_mesh, bind, build_scene, export


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

    # 'head' hangs off 'spine' with no offset of its own, so the two share a
    # world matrix
    world = [
        np.eye(4),
        _row_translate(0.0, bind_y, 0.0),
        _row_translate(0.0, bind_y, 0.0),
    ]
    # glTF stores a matrix column-major, so a row-vector matrix goes out
    # exactly as it sits in memory
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


# ------------------------------- attachment -------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbAttachment(GlbCase):
    def test_load_glb_attaches_a_skin(self):
        obj = Object.load_glb(self.skinned())

        self.assertIsInstance(obj.skin, SkinDeformData)
        self.assertEqual(obj.skin.joints, ["root", "spine", "head"])

    def test_load_skin_false_leaves_it_unset(self):
        obj = Object.load_glb(self.skinned(), load_skin=False)

        self.assertIsNone(obj.skin)

    def test_an_unskinned_glb_is_not_a_warning(self):
        path = self.path()
        write_glb(path, ["body"], skinned=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = Object.load_glb(path)

        self.assertIsNone(obj.skin)
        # most glbs are unskinned; warning on all of them would be noise
        self.assertEqual([str(w.message) for w in caught], [])

    def test_an_unskinned_glb_trimesh_thins_out_is_still_not_a_warning(self):
        """no skins anywhere means nothing can be mis-bound

        Trimesh drops the TRIANGLE_FAN and the skin reader keeps it, so the
        two lists disagree on length -- but with nothing skinned there is no
        pairing to get wrong, and warning here would fire on ordinary files.
        """
        path = self.path()
        write_glb(path, ["m0", "m1"], skinned=False, modes=[None, TRIANGLE_FAN])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene = Scene.load_glb(path)

        self.assertTrue(all(o.skin is None for o in scene.objects))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_an_explicit_skin_wins_over_the_file(self):
        path = self.skinned()
        mine = Object.load_glb(path).skin

        obj  = Object.load_glb(path, skin=mine)

        self.assertIs(obj.skin, mine)

    def test_scene_load_glb_attaches_one_skin_per_object(self):
        scene   = Scene.load_glb(self.skinned(names=("torso", "limb")))

        objects = list(scene.objects)
        self.assertEqual(len(objects), 2)
        for obj in objects:
            self.assertIsInstance(obj.skin, SkinDeformData)

    def test_scene_load_skin_false_leaves_them_unset(self):
        scene = Scene.load_glb(self.skinned(names=("torso", "limb")), load_skin=False)

        self.assertTrue(all(o.skin is None for o in scene.objects))


# -------------------------------- pairing ---------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbPairing(GlbCase):
    def test_duplicate_names_still_pair_by_position(self):
        """trimesh renames the second mesh, the skin reader does not

        Reading these back gives ['part', 'part_1'] on the mesh side and
        ['part', 'part'] on the skin side, so a name lookup would miss the
        second mesh entirely and hand the first one's weights to both.
        """
        path = self.path()
        write_glb(path, ["part", "part"], geo_names=["a", "b"])

        scene   = Scene.load_glb(path)

        objects = list(scene.objects)
        self.assertEqual([o.name for o in objects], ["part", "part_1"])
        self.assertTrue(all(o.skin is not None for o in objects))
        # each mesh keeps its OWN geometry, so the pairing is positional
        self.assertAlmostEqual(float(objects[0].mesh.points[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(objects[1].mesh.points[:, 0].min()), 300.0)

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

        by_name = {o.name: o for o in Scene.load_glb(path).objects}

        self.assertIsNotNone(by_name["skinned"].skin)
        self.assertIsNone(by_name["static"].skin)

    def test_a_primitive_trimesh_drops_warns_instead_of_mis_binding(self):
        """trimesh skips the TRIANGLE_FAN, the skin reader keeps it

        That leaves 2 meshes against 3 skins, so index 1 would pair 'm2'
        with 'm1''s weights.  Refusing beats guessing.
        """
        path = self.path()
        write_glb(path, ["m0", "m1", "m2"], modes=[None, TRIANGLE_FAN, None])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene = Scene.load_glb(path)

        objects = list(scene.objects)
        self.assertEqual([o.name for o in objects], ["m0", "m2"])
        self.assertTrue(all(o.skin is None for o in objects))
        self.assertTrue(
            any("cannot be paired up" in str(w.message) for w in caught),
            f"expected a pairing warning, got {[str(w.message) for w in caught]}",
        )


# --------------------------------- posing ---------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestPose(GlbCase):
    def test_posing_moves_the_mesh(self):
        path   = self.skinned()
        obj    = Object.load_glb(path)
        before = np.array(obj.mesh.points, copy=True)

        obj.pose(self.bent_rig(path))

        self.assertFalse(np.allclose(before, obj.mesh.points))

    def test_pose_returns_self_for_chaining(self):
        path = self.skinned()
        obj  = Object.load_glb(path)

        self.assertIs(obj.pose(self.bent_rig(path)), obj)

    def test_the_bind_pose_is_a_fixed_point(self):
        path   = self.skinned()
        obj    = Object.load_glb(path)
        before = np.array(obj.mesh.points, copy=True)

        obj.pose(HierarchyData.load_glb(path))

        np.testing.assert_allclose(before, obj.mesh.points, atol=1e-9)

    def test_posing_twice_does_not_compound(self):
        """the whole reason ``pose`` keeps a bind mesh

        Deforming the CURRENT mesh instead of the bind mesh also produces
        geometry, it just drifts further every frame.  Same pose twice has
        to land in the same place.
        """
        path = self.skinned()
        obj  = Object.load_glb(path)
        rig  = self.bent_rig(path)

        obj.pose(rig)
        once = np.array(obj.mesh.points, copy=True)
        obj.pose(rig)

        np.testing.assert_allclose(once, obj.mesh.points, atol=1e-9)

    def test_a_sequence_is_order_independent(self):
        """frame N must not depend on which frames were rendered before it"""
        path    = self.skinned()
        rig_a   = self.bent_rig(path, 30.0)
        rig_b   = self.bent_rig(path, 75.0)

        direct  = Object.load_glb(path).pose(rig_b)
        after_a = Object.load_glb(path).pose(rig_a).pose(rig_b)

        np.testing.assert_allclose(direct.mesh.points, after_a.mesh.points, atol=1e-9)

    def test_returning_to_the_bind_pose_restores_the_points(self):
        path = self.skinned()
        obj  = Object.load_glb(path)
        rest = np.array(obj.mesh.points, copy=True)

        obj.pose(self.bent_rig(path)).pose(HierarchyData.load_glb(path))

        np.testing.assert_allclose(rest, obj.mesh.points, atol=1e-9)

    def test_posing_hands_back_a_new_mesh(self):
        """the render cache keys on id(), so a new object is what invalidates"""
        path   = self.skinned()
        obj    = Object.load_glb(path)
        before = obj.mesh

        obj.pose(self.bent_rig(path))

        self.assertIsNot(before, obj.mesh)

    def test_posing_leaves_the_deformer_rest_points_alone(self):
        path = self.skinned()
        obj  = Object.load_glb(path)
        rest = np.array(obj.skin.rest_points, copy=True)

        obj.pose(self.bent_rig(path))

        np.testing.assert_allclose(rest, obj.skin.rest_points)

    def test_swapping_the_mesh_rebuilds_the_bind_pose(self):
        """a mesh assignment drops the cached bind pose

        Without that, the cache would keep serving the old topology and the
        newly assigned mesh would be silently ignored.
        """
        path = self.skinned()
        obj  = Object.load_glb(path)
        obj.pose(self.bent_rig(path))
        self.assertIsNotNone(obj._bind_mesh)

        obj.mesh = obj.mesh.copy()

        self.assertIsNone(obj._bind_mesh)

    def test_assigning_a_skin_drops_the_bind_pose(self):
        path = self.skinned()
        obj  = Object.load_glb(path)
        obj.pose(self.bent_rig(path))

        obj.skin = obj.skin

        self.assertIsNone(obj._bind_mesh)

    def test_pose_without_a_skin_raises(self):
        obj = Object.load_glb(self.skinned(), load_skin=False)

        with self.assertRaises(ValueError):
            obj.pose(HierarchyData.load_glb(self.skinned()))

    def test_pose_without_a_mesh_raises(self):
        path     = self.skinned()
        obj      = Object.load_glb(path)
        obj.mesh = None

        with self.assertRaises(ValueError):
            obj.pose(HierarchyData.load_glb(path))


# ------------------------------ render cache ------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestRenderCache(GlbCase):
    def test_attaching_a_skin_is_not_a_render_change(self):
        """a deformer alone changes nothing on screen until pose() runs

        Two separate claims: the skin stays out of the scene signature, and
        the skin setter does not drop an already rendered frame.  The frame
        half needs the private ``_frame`` because clearing it is the whole
        observable effect of ``_mark_render_dirty`` -- the signature does
        not move when it fires.
        """
        path  = self.skinned()
        obj   = Object.load_glb(path, load_skin=False)
        scene = Scene("s")
        scene.append(obj)
        before    = scene._compute_render_signature()
        sentinel  = object()
        obj.frame = sentinel

        obj.skin  = Object.load_glb(path).skin

        self.assertEqual(before, scene._compute_render_signature())
        self.assertIs(obj.frame, sentinel)

    def test_posing_does_drop_a_cached_render(self):
        """the counterpart: pose() changes the mesh, so the frame must go"""
        path      = self.skinned()
        obj       = Object.load_glb(path)
        obj.frame = object()

        obj.pose(self.bent_rig(path))

        self.assertIsNone(obj.frame)

    def test_posing_invalidates_the_scene_signature(self):
        path  = self.skinned()
        obj   = Object.load_glb(path)
        scene = Scene("s")
        scene.append(obj)
        before = scene._compute_render_signature()

        obj.pose(self.bent_rig(path))

        self.assertNotEqual(before, scene._compute_render_signature())


# ------------------------------- persistence ------------------------------- #


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestPersistence(GlbCase):
    def round_trips(self):
        path = self.skinned()
        obj  = Object.load_glb(path)
        yield "to_dict", Object.from_dict(obj.to_dict())
        yield "pickle", pickle.loads(pickle.dumps(obj))
        yield "copy", obj.copy()

    def test_the_skin_survives_every_round_trip(self):
        """``Data.to_dict`` only writes a field that differs from its default,
        and ``Data`` compares equal to a foreign type, so a nested ``Data``
        is dropped unless it is handled by hand."""
        for label, restored in self.round_trips():
            with self.subTest(label):
                self.assertIsInstance(restored.skin, SkinDeformData)
                self.assertEqual(restored.skin.joints, ["root", "spine", "head"])

    def test_a_restored_object_still_poses(self):
        path     = self.skinned()
        expected = Object.load_glb(path).pose(self.bent_rig(path))

        for label, restored in self.round_trips():
            with self.subTest(label):
                restored.pose(self.bent_rig(path))
                np.testing.assert_allclose(
                    expected.mesh.points, restored.mesh.points, atol=1e-9
                )

    def test_a_restored_object_poses_from_the_bind_pose(self):
        """the bind mesh is a cache and does not survive the trip; it has to
        be rebuilt from ``rest_points``, which does"""
        path  = self.skinned()
        obj   = Object.load_glb(path).pose(self.bent_rig(path))
        posed = np.array(obj.mesh.points, copy=True)

        restored = Object.from_dict(obj.to_dict())
        self.assertIsNone(restored._bind_mesh)
        restored.pose(self.bent_rig(path))

        # saved POSED, so a naive rebuild would deform the posed points
        np.testing.assert_allclose(posed, restored.mesh.points, atol=1e-9)

    def test_an_object_without_a_skin_round_trips_unchanged(self):
        obj = Object.load_glb(self.skinned(), load_skin=False)

        self.assertIsNone(Object.from_dict(obj.to_dict()).skin)


# --------------------------------- the fbx --------------------------------- #


def _fbx_transform(translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0)):
    matrix = fbx.FbxAMatrix()
    matrix.SetT(fbx.FbxVector4(*translate, 0.0))
    matrix.SetR(fbx.FbxVector4(*rotate, 0.0))
    return matrix


class FbxCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, populate):
        manager = fbx.FbxManager.Create()
        try:
            scene = build_scene(manager)
            populate(scene)
            return export(manager, scene, os.path.join(self.tmp.name, name))
        finally:
            manager.Destroy()


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestFbx(FbxCase):
    def test_load_fbx_attaches_a_skin(self):
        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            bind(scene, mesh, [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})])

        obj = Object.load_fbx(self.write("body.fbx", populate))

        self.assertIsInstance(obj.skin, SkinDeformData)
        self.assertEqual(obj.skin.joints, ["root", "tip"])

    def test_load_skin_false_leaves_it_unset(self):
        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            bind(scene, mesh, [(root, {0: 1.0})])

        obj = Object.load_fbx(self.write("body.fbx", populate), load_skin=False)

        self.assertIsNone(obj.skin)

    def test_a_static_mesh_does_not_shift_the_pairing(self):
        """the fbx skin reader drops unskinned meshes, so positions lie

        'prop_geo' is mesh 1 but there is only ONE skin entry, so pairing by
        position would hand 'prop_geo' the body's weights.  Matching on the
        node name is what keeps this honest.
        """

        def populate(scene):
            add_mesh(scene, "prop_geo", 4)
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            bind(scene, mesh, [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})])

        path    = self.write("mixed.fbx", populate)
        names   = [Object.load_fbx(path, index=i).name for i in range(2)]
        skins   = [Object.load_fbx(path, index=i).skin for i in range(2)]

        by_name = dict(zip(names, skins))
        self.assertIsNone(by_name["prop_geo"])
        self.assertIsNotNone(by_name["body_geo"])

    def test_an_unskinned_fbx_is_not_a_warning(self):
        def populate(scene):
            add_mesh(scene, "prop_geo", 4)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = Object.load_fbx(self.write("static.fbx", populate))

        self.assertIsNone(obj.skin)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_two_skinned_meshes_sharing_a_name_warn_instead_of_guessing(self):
        """fbx lets two nodes share a name, and then the name stops
        identifying anything"""

        def populate(scene):
            root = add_joint(scene, "root")
            for _ in range(2):
                _, mesh = add_mesh(scene, "body_geo", 4)
                bind(scene, mesh, [(root, {0: 1.0})])

        path = self.write("twins.fbx", populate)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = Object.load_fbx(path)

        self.assertIsNone(obj.skin)
        self.assertTrue(
            any("cannot be told apart" in str(w.message) for w in caught),
            f"expected an ambiguity warning, got {[str(w.message) for w in caught]}",
        )


# -------------------------------- bind pose -------------------------------- #


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestFbxBindPose(FbxCase):
    """the bind pose has to come off the clusters, not off the joint nodes

    An animated fbx leaves its joint nodes wherever the take put them, so
    reading a bind pose back off them yields some arbitrary frame of the
    animation rather than the pose the mesh was actually bound in.  Every
    vertex is then displaced by the difference between the two, which reads
    as limbs stretching away from the body.  The file's own answer is stored
    per cluster, so that is what has to be read.
    """

    def bent(self, name="bent.fbx", bind_y=5.0, node_y=10.0):
        """'tip' binds at bind_y while its node sits at node_y"""

        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            tip.LclTranslation.Set(fbx.FbxDouble3(0.0, node_y, 0.0))
            bind(
                scene,
                mesh,
                [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})],
                transform_link={
                    "root": _fbx_transform(),
                    "tip":  _fbx_transform((0.0, bind_y, 0.0)),
                },
            )

        return self.write(name, populate)

    def test_the_deformer_holds_the_authored_inverse_binds(self):
        obj = Object.load_fbx(self.bent())

        expected           = np.tile(np.eye(4), (2, 1, 1))
        expected[1, 3, :3] = (0.0, -5.0, 0.0)
        np.testing.assert_allclose(
            np.asarray(obj.skin.inverse_bind_matrices), expected, atol=1e-6
        )

    def test_posing_moves_by_the_bind_to_node_difference(self):
        path = self.bent()
        obj  = Object.load_fbx(path)
        rest = np.asarray(obj.mesh.points, dtype=float).copy()

        obj.pose(HierarchyData.load_fbx(path))
        moved = np.asarray(obj.mesh.points, dtype=float) - rest

        # 'tip' binds 5 up and its node sits 10 up, so the vertices it owns
        # travel the 5 unit difference.  Deriving the bind from the nodes
        # instead makes both ends agree and nothing moves at all.
        np.testing.assert_allclose(moved[[0, 1]], 0.0, atol=1e-9)
        np.testing.assert_allclose(
            moved[[2, 3]], np.tile([0.0, 5.0, 0.0], (2, 1)), atol=1e-6
        )

    def test_posing_at_the_bind_pose_does_not_move_anything(self):
        path = self.bent(bind_y=7.0, node_y=7.0)
        obj  = Object.load_fbx(path)
        rest = np.asarray(obj.mesh.points, dtype=float).copy()

        obj.pose(HierarchyData.load_fbx(path))

        np.testing.assert_allclose(
            np.asarray(obj.mesh.points, dtype=float), rest, atol=1e-6
        )

    def test_the_first_cluster_wins_when_a_joint_is_reached_twice(self):
        """two clusters may link one joint, and they may disagree

        The weights reader already folds the repeat onto its first column,
        so the bind matrix has to be picked the same way round -- otherwise
        the weights come from one cluster and the bind from the other.
        """

        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            skin = fbx.FbxSkin.Create(scene, "skin")
            for index, (weights, bind_y) in enumerate(
                (({0: 1.0, 1: 1.0}, 5.0), ({2: 1.0, 3: 1.0}, 9.0))
            ):
                cluster = fbx.FbxCluster.Create(scene, f"cl{index}")
                cluster.SetLink(root)
                for vertex, weight in weights.items():
                    cluster.AddControlPointIndex(vertex, weight)
                cluster.SetTransformLinkMatrix(_fbx_transform((0.0, bind_y, 0.0)))
                skin.AddCluster(cluster)
            mesh.AddDeformer(skin)

        obj = Object.load_fbx(self.write("twoclusters.fbx", populate))

        self.assertEqual(obj.skin.joints, ["root"])
        np.testing.assert_allclose(
            np.asarray(obj.skin.inverse_bind_matrices)[0][3, :3],
            [0.0, -5.0, 0.0],
            atol=1e-6,
        )

    def test_the_mesh_bind_matrix_is_applied_before_the_joint(self):
        """``MeshBind @ inv(LinkBind)``, in that order

        A cluster stores the mesh's own place at bind time as well as the
        joint's, and the two only commute while both are pure translation --
        which is the case in most files, and in this file's other fixtures,
        so a swapped or dropped mesh bind hides there.  Rotating it pulls
        the two apart.
        """

        def populate(scene):
            _, mesh = add_mesh(scene, "body_geo", 4)
            root = add_joint(scene, "root")
            tip  = add_joint(scene, "tip", parent=root)
            tip.LclTranslation.Set(fbx.FbxDouble3(0.0, 10.0, 0.0))
            skin = bind(
                scene,
                mesh,
                [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})],
                transform_link={
                    "root": _fbx_transform(),
                    "tip":  _fbx_transform((0.0, 5.0, 0.0)),
                },
            )
            for index in range(skin.GetClusterCount()):
                skin.GetCluster(index).SetTransformMatrix(
                    _fbx_transform(rotate=(0.0, 0.0, 90.0))
                )

        path = self.write("meshbind.fbx", populate)
        obj  = Object.load_fbx(path)
        obj.pose(HierarchyData.load_fbx(path))

        # control points sit at (i, 0, 0).  The mesh bind turns +X into +Y,
        # then 'tip' adds the 5 unit gap between its bind and its node:
        #   root: (i, 0, 0) @ Rz90                 -> (0, i, 0)
        #   tip : (i, 0, 0) @ Rz90 @ T(0, 5, 0)    -> (0, i + 5, 0)
        np.testing.assert_allclose(
            np.asarray(obj.mesh.points, dtype=float),
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0]],
            atol=1e-6,
        )


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbBindPose(GlbCase):
    """same rule for glb, plus the unit conversion the fbx path does not need

    ``MeshData.load_glb`` scales the points it reads while the file's binds
    stay in the file's own units, so the two only line up once the binds are
    brought into the same units.
    """

    def test_bind_pose_comes_from_the_file_not_the_nodes(self):
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=3.0)

        obj = Object.load_glb(path, scale_factor=1.0)

        # the file says the rig bound 1 up; the nodes sit 3 up because that
        # is where the take left them
        np.testing.assert_allclose(
            np.asarray(obj.skin.inverse_bind_matrices)[1][3, :3],
            [0.0, -1.0, 0.0],
            atol=1e-6,
        )

    def test_posing_moves_by_the_bind_to_node_difference(self):
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=3.0)

        obj  = Object.load_glb(path, scale_factor=1.0)
        rest = np.asarray(obj.mesh.points, dtype=float).copy()
        obj.pose(HierarchyData.load_glb(path, scale_factor=1.0))
        moved = np.asarray(obj.mesh.points, dtype=float) - rest

        # sorted because trimesh owes us no particular vertex order: the
        # 'root' vertex holds still and the other two travel the 2 unit gap
        np.testing.assert_allclose(sorted(moved[:, 1]), [0.0, 2.0, 2.0], atol=1e-6)

    def test_inverse_binds_are_scaled_with_the_mesh(self):
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=1.0)

        obj = Object.load_glb(path, scale_factor=100.0)

        np.testing.assert_allclose(
            np.asarray(obj.skin.inverse_bind_matrices)[1][3, :3],
            [0.0, -100.0, 0.0],
            atol=1e-4,
        )

    def test_posing_at_the_bind_pose_does_not_move_anything(self):
        """the sharpest statement of the units problem

        Bind and pose are the same skeleton here, so every skin matrix has
        to come out identity.  Leave the binds in the file's units while the
        points get scaled and they miss by the scale factor instead, which
        flings the mesh 99 units off.
        """
        path = self.path()
        write_bound_glb(path, bind_y=1.0, node_y=1.0)

        obj  = Object.load_glb(path, scale_factor=100.0)
        rest = np.asarray(obj.mesh.points, dtype=float).copy()

        obj.pose(HierarchyData.load_glb(path, scale_factor=100.0))

        np.testing.assert_allclose(
            np.asarray(obj.mesh.points, dtype=float), rest, atol=1e-6
        )