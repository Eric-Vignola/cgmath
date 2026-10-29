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
from cgmath.geometry.mesh import MeshData, MeshList, UVList, load as mesh_load, load_fbx as mesh_load_fbx
from cgmath.geometry.skin_weights import SkinList, _read_skins_fbx
from cgmath.formats._fbx_io import open_fbx
from cgmath.render.scene import Object, Scene
from cgmath.hierarchy import ClipData, HierarchyData

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
        """the skin is nested data: every route back must rebuild it as a
        ``SkinDeformData``, not drop it or leave a dict"""
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


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestFbxPosesInTheRigsUnitAndAxes(FbxCase):
    """Object.load_fbx reads the mesh and its inverse binds in the rig
    loaders' unit, cm by default, and turned into Maya's Y-up as they turn a
    rig's roots, the Object having no node to carry the turn, so a rig or a
    clip read from the same file poses the mesh where the file has it; the
    mesh readers read the same unit and leave the axes alone

    At the file's bind pose the rig leaves the bind mesh where it is. Posed
    off it, the mesh is the file's own posed mesh, read raw
    (``scale_factor=1.0, axis_system=None``), converted the same way.
    """

    # Unreal's Z-up into Maya's Y-up: (x, y, z) -> (x, z, -y)
    TURN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    RAW  = {"scale_factor": 1.0, "axis_system": None}

    def skinned(self, name, unit=None, axes=None, knee=20.0, moved_body=False):
        """a hip and a knee bound to a quad with the knee at 20 degrees;
        ``knee`` is the knee's rotate in the file's node pose, off the bind
        for any other value. ``moved_body`` moves and turns the mesh's own
        node, its bind matrix with it. ``unit`` names the file's
        FbxSystemUnit and ``axes`` is its FbxAxisSystem, cm and Maya's Y-up
        when None"""

        def populate(scene):
            if unit is not None:
                scene.GetGlobalSettings().SetSystemUnit(getattr(fbx.FbxSystemUnit, unit))
            if axes is not None:
                scene.GetGlobalSettings().SetAxisSystem(axes)

            hip  = add_joint(scene, "hip")
            shin = add_joint(scene, "knee", parent=hip)
            hip.LclTranslation.Set(fbx.FbxDouble3(0.0, 1.0, 0.5))
            hip.LclRotation.Set(fbx.FbxDouble3(0.0, 0.0, 10.0))
            shin.LclTranslation.Set(fbx.FbxDouble3(0.0, -0.5, 0.1))
            shin.LclRotation.Set(fbx.FbxDouble3(20.0, 0.0, 0.0))

            corners = ((-0.2, 1.2, 0.0), (0.2, 1.2, 0.0), (0.2, 0.4, 0.1), (-0.2, 0.4, 0.1))
            mesh    = fbx.FbxMesh.Create(scene, "body_geoShape")
            mesh.InitControlPoints(len(corners))
            for index, corner in enumerate(corners):
                mesh.SetControlPointAt(fbx.FbxVector4(*corner, 1.0), index)
            mesh.BeginPolygon()
            for index in range(len(corners)):
                mesh.AddPolygon(index)
            mesh.EndPolygon()
            normals = mesh.CreateElementNormal()
            normals.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
            normals.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
            for _ in corners:
                normals.GetDirectArray().Add(fbx.FbxVector4(0.0, 0.6, 0.8, 0.0))

            body = fbx.FbxNode.Create(scene, "body_geo")
            body.SetNodeAttribute(mesh)
            if moved_body:
                body.LclTranslation.Set(fbx.FbxDouble3(0.3, 0.0, -0.2))
                body.LclRotation.Set(fbx.FbxDouble3(0.0, 15.0, 0.0))
            scene.GetRootNode().AddChild(body)

            weights = [(hip, {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5}), (shin, {2: 0.5, 3: 0.5})]
            skin    = bind(scene, mesh, weights)
            for index in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(index)
                cluster.SetTransformMatrix(body.EvaluateGlobalTransform())
                cluster.SetTransformLinkMatrix(cluster.GetLink().EvaluateGlobalTransform())
            shin.LclRotation.Set(fbx.FbxDouble3(knee, 0.0, 0.0))

        return self.write(name, populate)

    @staticmethod
    def points(obj):
        return np.array(obj.mesh.points, dtype=float)

    def cases(self):
        """(label, unit, axis system, cm per unit, turn) of the files these
        tests read"""
        zup = fbx.FbxAxisSystem.MayaZUp
        return (
            ("m", "m", None, 100.0, np.eye(3)),
            ("Inch", "Inch", None, 2.54, np.eye(3)),
            ("cm", "cm", None, 1.0, np.eye(3)),
            ("cm Z-up", "cm", zup, 1.0, self.TURN),
            ("m Z-up", "m", zup, 100.0, self.TURN),
        )

    def test_at_its_bind_pose_the_rig_leaves_the_mesh_where_it_is(self):
        """a meters file posed by its own rig at rest moved the mesh 99 units
        when the mesh stayed in meters and the rig loaded in cm, and a Z-up
        file 1.2 units on a mesh 0.8 across when the rig alone turned"""
        for label, unit, axes, _, _ in self.cases():
            with self.subTest(file=label):
                path = self.skinned(f"bind {label}.fbx", unit=unit, axes=axes)
                obj  = Object.load_fbx(path)
                bind = self.points(obj)
                for loader in (HierarchyData.load_fbx, ClipData.load_fbx):
                    obj.pose(loader(path))
                    np.testing.assert_allclose(self.points(obj), bind, rtol=0.0, atol=1e-9)
                    obj.restore_bind_pose()

    def test_the_mesh_loads_in_centimeters_and_the_object_in_maya_axes(self):
        """the points scale with the file's unit; the mesh readers keep the
        file's axes, and the Object turns its points and normals"""
        for label, unit, axes, cm, turn in self.cases():
            with self.subTest(file=label):
                path = self.skinned(f"mesh {label}.fbx", unit=unit, axes=axes)
                raw  = MeshData.load_fbx(path, scale_factor=1.0)
                read = (
                    (MeshData.load_fbx(path), np.eye(3)),
                    (MeshList.load_fbx(path)[0], np.eye(3)),
                    (mesh_load_fbx(path)[0][0], np.eye(3)),
                    (Object.load_fbx(path).mesh, turn),
                )
                for mesh, axes_turn in read:
                    want = np.asarray(raw.points) * cm @ axes_turn
                    np.testing.assert_allclose(mesh.points, want, rtol=0.0, atol=1e-12)
                    want = np.asarray(raw.normals) @ axes_turn
                    np.testing.assert_allclose(mesh.normals, want, rtol=0.0, atol=1e-12)

                # a number is the caller's own factor on the list reader too
                np.testing.assert_array_equal(MeshList.load_fbx(path, scale_factor=1.0)[0].points, raw.points)
                np.testing.assert_array_equal(Object.load_fbx(path, **self.RAW).mesh.points, raw.points)

    def test_posed_off_its_bind_the_mesh_is_the_files_posed_mesh_converted(self):
        for label, unit, axes, cm, turn in self.cases():
            with self.subTest(file=label):
                path  = self.skinned(f"posed {label}.fbx", unit=unit, axes=axes, knee=50.0, moved_body=True)
                posed = Object.load_fbx(path).pose(HierarchyData.load_fbx(path))
                raw   = Object.load_fbx(path, **self.RAW)
                bind  = self.points(raw)
                raw.pose(HierarchyData.load_fbx(path, **self.RAW))

                self.assertGreater(float(np.abs(self.points(raw) - bind).max()), 0.1)
                want = self.points(raw) * cm @ turn
                np.testing.assert_allclose(self.points(posed), want, rtol=0.0, atol=1e-9)

                # a number is the caller's own factor, on the rig and the mesh alike
                three = Object.load_fbx(path, scale_factor=3.0)
                three.pose(HierarchyData.load_fbx(path, scale_factor=3.0))
                want = self.points(raw) * 3.0 @ turn
                np.testing.assert_allclose(self.points(three), want, rtol=0.0, atol=1e-9)

    def test_a_left_handed_file_raises_on_the_object_unless_its_axes_are_kept(self):
        """no rotation turns it into Maya's axes, so the Object, which turns
        its points, raises; the mesh, weight and UV readers keep each mesh's
        own space and load it as it is, load() too"""
        system = fbx.FbxAxisSystem
        left   = system(system.EUpVector.eYAxis, system.EFrontVector.eParityOdd, system.ECoordSystem.eLeftHanded)
        path   = self.skinned("left.fbx", axes=left)
        raw    = MeshData.load_fbx(path)

        with self.assertRaises(ValueError) as caught:
            Object.load_fbx(path)
        self.assertIn("load_fbx(filename, axis_system=None)", str(caught.exception))
        kept = Object.load_fbx(path, axis_system=None)
        self.assertIsNotNone(kept.skin)
        np.testing.assert_array_equal(kept.mesh.points, raw.points)

        for label, loader in (
            ("MeshList.load_fbx", lambda x: MeshList.load_fbx(x)[0]),
            ("mesh.load_fbx", lambda x: mesh_load_fbx(x)[0][0]),
            ("MeshData.load", MeshData.load),
            ("mesh.load", lambda x: mesh_load(x)[0][0]),
        ):
            with self.subTest(loader=label):
                np.testing.assert_array_equal(loader(path).points, raw.points)

        self.assertEqual([skin.name for skin in SkinList.load_fbx(path)], ["body_geo"])
        self.assertEqual(len(UVList.load_fbx(path)), 0)

    def test_an_unknown_axis_system_is_refused(self):
        """by the Object; the mesh readers take none"""
        path = self.skinned("axes.fbx")

        with self.assertRaises(ValueError) as caught:
            Object.load_fbx(path, axis_system="maya_z_up")
        self.assertIn("'maya_y_up' or None", str(caught.exception))

        for loader in (MeshData.load_fbx, MeshList.load_fbx, mesh_load_fbx):
            with self.subTest(loader=loader.__qualname__):
                with self.assertRaises(TypeError):
                    loader(path, axis_system="maya_y_up")


def _rows(matrix):
    """an FbxAMatrix as a row-vector 4x4, the translation in row 3"""
    return np.array([[matrix.Get(row, col) for col in range(4)] for row in range(4)])


def _find(scene, name):
    """the scene's node called ``name``"""
    stack = [scene.GetRootNode()]
    while stack:
        node = stack.pop()
        if node.GetName() == name:
            return node
        stack.extend(node.GetChild(i) for i in range(node.GetChildCount()))
    raise KeyError(name)


def _control_points(mesh):
    """an FbxMesh's control points, (N, 3)"""
    return np.array([list(mesh.GetControlPointAt(i))[:3] for i in range(mesh.GetControlPointsCount())])


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestFbxMeshesInTheirOwnSpace(FbxCase):
    """the FBX mesh readers keep each mesh in its own object space, as the
    file, the sdk and Maya keep it, so under its node's world from a rig read
    from the same file a mesh is where the sdk puts it

    The rig loaders turn a Z-up file into Maya's Y-up on its roots, and a
    mesh takes that turn through its node. Mesh readers that turned the
    points too put every mesh under its node turned twice: 2-unreal.fbx's
    LODs 216 to 229 cm off. render's Object.load_fbx, a mesh with no node
    above it, takes the turn on its points instead.

    Each file holds a rigid mesh at the top, moved, turned and scaled
    (``prop``), one under a moved group (``sub``), one under a moved node
    with no attribute, which the rig read drops and folds into the mesh's
    node (``held``), one at the top with no transform (``flat``), and a mesh
    skinned to a hip and a knee, its node moved and turned and the knee
    posed 30 degrees off its bind (``body``). The truth is the sdk's own: a
    mesh node's EvaluateGlobalTransform after FbxAxisSystem.MayaYUp
    .ConvertScene, and the linear blend of a skin's clusters in the file's
    axes, turned, both times the cm one unit of the file holds.

    The same file written in meters and in inches by the sdk's
    FbxSystemUnit.ConvertScene holds the unit as a scale of 1/s on the
    scene root's children over points left in cm: a mesh under such a root
    reads its points as they are, and ``prop``, which the export found
    scaled other than 1, holds no unit, its points read s times larger.
    Under its node's world every mesh is where the sdk puts it, at true
    size.
    """

    # Maya's Z-up into its Y-up: (x, y, z) -> (x, z, -y)
    TURN   = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    MESHES = ("prop", "sub", "held", "flat", "body")

    # a cube off its own origin, so a turn moves every corner
    CORNERS = np.array(
        [
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
        ]
    ) + (0.5, 1.0, 1.5)
    FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3))
    NORMALS = (
        (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    )

    def cases(self):
        """(label, FbxSystemUnit name, axis system, cm per unit, turn, the
        FbxSystemUnit name ConvertScene writes the file in or None)"""
        system = fbx.FbxAxisSystem
        return (
            ("cm Y-up", "cm", system.MayaYUp, 1.0, np.eye(3), None),
            ("cm Z-up", "cm", system.MayaZUp, 1.0, self.TURN, None),
            ("m Z-up", "m", system.MayaZUp, 100.0, self.TURN, None),
            ("cm Z-up in m", "cm", system.MayaZUp, 100.0, self.TURN, "m"),
            ("cm Y-up in Inch", "cm", system.MayaYUp, 2.54, np.eye(3), "Inch"),
        )

    @staticmethod
    def factor(name, cm, convert):
        """what the readers multiply ``name``'s control points by: the unit,
        but in a converted file, whose meshes keep their points in cm under
        a root that holds the unit, all but ``prop``"""
        return cm if convert is None or name == "prop" else 1.0

    def scene_file(self, name, unit, axes, convert=None):
        """the file the class docstring describes, in ``unit`` and ``axes``,
        written in ``convert`` by the sdk's ConvertScene when it is given"""

        def place(
            scene,
            node_name,
            attribute,
            parent,
            translate = (0.0, 0.0, 0.0),
            rotate    = (0.0, 0.0, 0.0),
            scale     = (1.0, 1.0, 1.0),
        ):
            node = fbx.FbxNode.Create(scene, node_name)
            if attribute is not None:
                node.SetNodeAttribute(attribute)
            node.LclTranslation.Set(fbx.FbxDouble3(*translate))
            node.LclRotation.Set(fbx.FbxDouble3(*rotate))
            node.LclScaling.Set(fbx.FbxDouble3(*scale))
            parent.AddChild(node)
            return node

        def cube(scene, node_name, parent, *transform):
            mesh = fbx.FbxMesh.Create(scene, f"{node_name}Shape")
            mesh.InitControlPoints(len(self.CORNERS))
            for index, corner in enumerate(self.CORNERS):
                mesh.SetControlPointAt(fbx.FbxVector4(*corner, 1.0), index)
            for face in self.FACES:
                mesh.BeginPolygon()
                for index in face:
                    mesh.AddPolygon(index)
                mesh.EndPolygon()
            normals = mesh.CreateElementNormal()
            normals.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
            normals.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
            for normal, face in zip(self.NORMALS, self.FACES):
                for _ in face:
                    normals.GetDirectArray().Add(fbx.FbxVector4(*normal, 0.0))
            return place(scene, node_name, mesh, parent, *transform), mesh

        def populate(scene):
            settings = scene.GetGlobalSettings()
            settings.SetSystemUnit(getattr(fbx.FbxSystemUnit, unit))
            settings.SetAxisSystem(axes)
            top = scene.GetRootNode()

            cube(scene, "prop", top, (10.0, 2.0, 5.0), (15.0, 30.0, -20.0), (1.0, 2.0, 1.5))
            null  = fbx.FbxNull.Create(scene, "grpAttr")
            group = place(scene, "grp", null, top, (0.0, 0.0, 10.0), (0.0, 0.0, 45.0))
            cube(scene, "sub", group, (2.0, 0.0, 0.0), (10.0, 0.0, 0.0))
            bare = place(scene, "bare", None, top, (4.0, -3.0, 2.0), (20.0, 0.0, 35.0))
            cube(scene, "held", bare, (1.0, 1.0, 0.0), (0.0, 15.0, 0.0))
            cube(scene, "flat", top)

            hip  = add_joint(scene, "hip")
            knee = add_joint(scene, "knee", parent=hip)
            hip.LclTranslation.Set(fbx.FbxDouble3(0.0, 1.0, 0.5))
            hip.LclRotation.Set(fbx.FbxDouble3(0.0, 0.0, 10.0))
            knee.LclTranslation.Set(fbx.FbxDouble3(0.0, -0.5, 0.1))
            knee.LclRotation.Set(fbx.FbxDouble3(20.0, 0.0, 0.0))
            body, mesh = cube(scene, "body", top, (0.3, 0.0, -0.2), (0.0, 15.0, 0.0))
            weights = [
                (hip, {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5, 4: 1.0, 5: 0.25}),
                (knee, {2: 0.5, 3: 0.5, 5: 0.75, 6: 1.0, 7: 1.0}),
            ]
            skin = bind(scene, mesh, weights)
            for index in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(index)
                cluster.SetTransformMatrix(body.EvaluateGlobalTransform())
                cluster.SetTransformLinkMatrix(cluster.GetLink().EvaluateGlobalTransform())
            knee.LclRotation.Set(fbx.FbxDouble3(50.0, 0.0, 0.0))
            if convert is not None:
                getattr(fbx.FbxSystemUnit, convert).ConvertScene(scene)

        return self.write(name, populate)

    def sdk_points(self, path, cm):
        """{mesh: (its control points, the sdk's world points times ``cm``)},
        after FbxAxisSystem.MayaYUp.ConvertScene"""
        out = {}
        with open_fbx(path, fbx) as scene:
            fbx.FbxAxisSystem.MayaYUp.ConvertScene(scene)
            for name in self.MESHES:
                node      = _find(scene, name)
                points    = _control_points(node.GetNodeAttribute())
                world     = _rows(node.EvaluateGlobalTransform())
                out[name] = (points, (points @ world[:3, :3] + world[3, :3]) * cm)
        return out

    def sdk_blend(self, path, cm, turn):
        """the body as the sdk's linear blend of its clusters puts it, in the
        file's axes, then turned and times ``cm``"""
        with open_fbx(path, fbx) as scene:
            mesh   = _find(scene, "body").GetNodeAttribute()
            points = np.c_[_control_points(mesh), np.ones(mesh.GetControlPointsCount())]
            skin   = mesh.GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
            blend  = np.zeros_like(points)
            for index in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(index)
                link_bind, mesh_bind = fbx.FbxAMatrix(), fbx.FbxAMatrix()
                cluster.GetTransformLinkMatrix(link_bind)
                cluster.GetTransformMatrix(mesh_bind)
                link     = _rows(cluster.GetLink().EvaluateGlobalTransform())
                skinning = _rows(mesh_bind) @ np.linalg.inv(_rows(link_bind)) @ link
                for vertex, weight in zip(cluster.GetControlPointIndices(), cluster.GetControlPointWeights()):
                    blend[vertex] += weight * (points[vertex] @ skinning)
        return blend[:, :3] @ turn * cm

    def test_the_points_are_the_files_own_in_centimeters(self):
        """every reader gives each mesh's control points times the unit, as
        they are under a root that holds the unit, and the file's own
        normals"""
        for label, unit, axes, cm, _, convert in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"points {label}.fbx", unit, axes, convert)
                truth = self.sdk_points(path, cm)
                read  = {mesh.name: mesh for mesh in MeshList.load_fbx(path)}
                pairs = {mesh.name: mesh for mesh, _ in mesh_load_fbx(path)}
                self.assertEqual(sorted(read), sorted(self.MESHES))

                normals = np.repeat(self.NORMALS, [len(face) for face in self.FACES], axis=0)
                for name in self.MESHES:
                    for mesh in (read[name], pairs[name], MeshData.load_fbx(path, name=name)):
                        np.testing.assert_array_equal(mesh.points, truth[name][0] * self.factor(name, cm, convert))
                        np.testing.assert_array_equal(np.asarray(mesh.normals)[mesh.normal_indices], normals)

    def test_under_its_nodes_world_a_mesh_is_where_the_sdk_puts_it(self):
        """each mesh's node, from HierarchyData.load_fbx and ClipData.load_fbx
        of the same file, places its points where the sdk does; points turned
        as well land elsewhere in a Z-up file"""
        for label, unit, axes, cm, turn, convert in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"worlds {label}.fbx", unit, axes, convert)
                truth = self.sdk_points(path, cm)
                read  = {mesh.name: np.asarray(mesh.points) for mesh in MeshList.load_fbx(path)}

                for loader in (HierarchyData.load_fbx, ClipData.load_fbx):
                    rig = loader(path)
                    for name in self.MESHES:
                        world = np.asarray(rig[name].world_matrix)
                        got   = read[name] @ world[:3, :3] + world[3, :3]
                        np.testing.assert_allclose(got, truth[name][1], rtol=0.0, atol=1e-12)

                        if turn is not self.TURN:
                            continue
                        twice = read[name] @ turn @ world[:3, :3] + world[3, :3]
                        self.assertGreater(float(np.abs(twice - truth[name][1]).max()), 1.0)

    def test_a_skinned_mesh_posed_by_the_rig_is_the_sdks_blend(self):
        """the reader's points and _read_skins_fbx's inverse binds, both in the
        mesh's own space, deformed by the rig or the clip of the same file; and
        the Object, whose points and binds turn together"""
        for label, unit, axes, cm, turn, convert in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"skin {label}.fbx", unit, axes, convert)
                want  = self.sdk_blend(path, cm, turn)
                mesh  = MeshData.load_fbx(path, name="body")
                skins = {skin.name: (skin, binds) for skin, binds in _read_skins_fbx(path, bind_matrices=True)}

                deformer = SkinDeformData(mesh, skins["body"][0], inverse_bind_matrices=skins["body"][1])
                index    = [x.name for x in MeshList.load_fbx(path)].index("body")
                obj      = Object.load_fbx(path, index=index, extract_texture=False)
                for loader in (HierarchyData.load_fbx, ClipData.load_fbx):
                    rig = loader(path)
                    np.testing.assert_allclose(deformer.apply(rig), want, rtol=0.0, atol=1e-12)
                    np.testing.assert_allclose(obj.pose(rig).mesh.points, want, rtol=0.0, atol=1e-12)
                    obj.restore_bind_pose()

                # the knee is off its bind, so the blend moved the mesh
                world = np.asarray(HierarchyData.load_fbx(path)["body"].world_matrix)
                rest  = np.asarray(mesh.points) @ world[:3, :3] + world[3, :3]
                self.assertGreater(float(np.abs(want - rest).max()), 0.1 * self.factor("body", cm, convert))

    def test_the_object_turns_a_mesh_with_no_node(self):
        """Object.load_fbx shows each mesh's own points and normals turned,
        which is where the sdk puts a mesh whose node has no transform and
        nothing above it"""
        for label, unit, axes, cm, turn, convert in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"object {label}.fbx", unit, axes, convert)
                truth = self.sdk_points(path, cm)

                for index, mesh in enumerate(MeshList.load_fbx(path)):
                    obj = Object.load_fbx(path, index=index, extract_texture=False, load_skin=False)
                    self.assertEqual(obj.name, mesh.name)
                    np.testing.assert_array_equal(obj.mesh.points, np.asarray(mesh.points) @ turn)
                    np.testing.assert_array_equal(obj.mesh.normals, np.asarray(mesh.normals) @ turn)
                    if mesh.name == "flat":
                        np.testing.assert_allclose(obj.mesh.points, truth["flat"][1], rtol=0.0, atol=1e-12)


def _polar(matrix):
    """the orthogonal factor of a (3, 3) matrix's polar decomposition: its
    rotation part, a mirror with it when its determinant is negative"""
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def _off_face_normals():
    """(24, 3) unit face-vertex normals for the cube's 24 face corners, each
    face's own tipped a different way, so a rule that turns them by the
    rotation alone and one that takes the scale too part on every one"""
    k      = np.arange(24.0)
    tipped = np.repeat(TestFbxMeshesInTheirOwnSpace.NORMALS, 4, axis=0)
    tipped = tipped + 0.4 * np.c_[np.sin(k), np.cos(k), np.sin(2.0 * k)]
    return tipped / np.linalg.norm(tipped, axis=1, keepdims=True)


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class TestFbxGeometricTransform(FbxCase):
    """the FBX mesh readers bake a node's geometric transform, the offset
    that moves its mesh alone and not its children, into the mesh, as Maya's
    FBXImport does; 3.0.0 left it out, the meshes with an offset in this
    file 2.00 to 5.50 off Maya's object space

    The points take the offset as the sdk's FbxAMatrix(T, R, S) composes it,
    scaled, then turned in xyz whatever the node's rotate order, then moved,
    its translation times the unit, so under their node's world they are
    where the sdk puts them. The normals take the offset's rotation alone,
    flipped when it mirrors, at unit length, as Maya's import turns them:
    ``sign(det G) * n @ U``, U the rotation part of the offset's 3x3 G. So
    under an uneven offset scale they are not perpendicular to the stretched
    faces, as in Maya. The faces keep the file's order, a mirror's too, and
    a skinned mesh posed by its rig is the sdk's blend, which applies the
    offset before the cluster's matrices, with the inverse binds as read.
    An offset whose scale is 0 on an axis flattens the mesh, and the readers
    drop its normals, as Maya's import does; a scale near 0 but not 0 does
    not flatten it, and its normals follow the rule, as Maya keeps them.

    Each file holds, under a group moved, turned and evenly scaled
    (``grp``): ``geo``, offset by (0, 3, 0), (0, 0, 30) and an uneven
    (1.5, 0.5, 2); ``turned``, offset by a turn (25, -40, 60) alone on a
    node whose rotate order is zxy; ``mirror``, offset by a mirroring
    (-1.5, 0.5, 2); ``plain``, no offset; ``body``, offset and skinned to a
    hip and a knee posed off its bind; and ``scaled``, offset by an uneven
    scale (0.5, 2, 1) alone. The normals are authored off the faces' own,
    and longer or shorter than 1 on the meshes with an offset. The file of
    flattening offsets holds ``flat``, offset by (0.5, 1, 0), (10, 20, 30)
    and a scale (1, 0, 1); ``edge``, by a scale (-0, 1, 1) alone; and
    ``thin``, offset as ``flat`` but by a scale (1, 1e-30, 1).
    """

    # Maya's Z-up into its Y-up: (x, y, z) -> (x, z, -y)
    TURN    = TestFbxMeshesInTheirOwnSpace.TURN
    CORNERS = TestFbxMeshesInTheirOwnSpace.CORNERS
    FACES   = TestFbxMeshesInTheirOwnSpace.FACES
    OFFSETS = {
        "geo":    ((0.0, 3.0, 0.0), (0.0, 0.0, 30.0), (1.5, 0.5, 2.0)),
        "turned": ((0.0, 0.0, 0.0), (25.0, -40.0, 60.0), (1.0, 1.0, 1.0)),
        "mirror": ((1.0, 2.0, 3.0), (0.0, 0.0, 30.0), (-1.5, 0.5, 2.0)),
        "plain":  None,
        "body":   ((0.5, 1.0, 0.0), (0.0, 0.0, 30.0), (1.0, 2.0, 1.0)),
        "scaled": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 2.0, 1.0)),
    }
    # offsets that flatten the mesh, a scale of 0 on an axis
    FLAT = {
        "flat": ((0.5, 1.0, 0.0), (10.0, 20.0, 30.0), (1.0, 0.0, 1.0)),
        "edge": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.0, 1.0, 1.0)),
    }
    # one that nearly flattens it, a scale near 0 but not 0: Maya keeps its
    # normals, turned by the rule
    THIN = {"thin": ((0.5, 1.0, 0.0), (10.0, 20.0, 30.0), (1.0, 1e-30, 1.0))}

    def cases(self):
        """(label, FbxSystemUnit name, axis system, cm per unit, turn)"""
        system = fbx.FbxAxisSystem
        return (
            ("cm Y-up", "cm", system.MayaYUp, 1.0, np.eye(3)),
            ("cm Z-up", "cm", system.MayaZUp, 1.0, self.TURN),
            ("m Z-up", "m", system.MayaZUp, 100.0, self.TURN),
        )

    def cube(self, scene, node_name, offset):
        """a node of the cube with its normals authored, offset by
        ``offset``, (T, R, S), or None; returns (node, mesh)"""
        normals = _off_face_normals()
        lengths = 0.5 + 0.1 * np.arange(len(normals))

        mesh = fbx.FbxMesh.Create(scene, f"{node_name}Shape")
        mesh.InitControlPoints(len(self.CORNERS))
        for index, corner in enumerate(self.CORNERS):
            mesh.SetControlPointAt(fbx.FbxVector4(*corner, 1.0), index)
        for face in self.FACES:
            mesh.BeginPolygon()
            for index in face:
                mesh.AddPolygon(index)
            mesh.EndPolygon()
        element = mesh.CreateElementNormal()
        element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
        element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
        authored = normals if offset is None else normals * lengths[:, None]
        for normal in authored:
            element.GetDirectArray().Add(fbx.FbxVector4(*normal, 0.0))

        node = fbx.FbxNode.Create(scene, node_name)
        node.SetNodeAttribute(mesh)
        if offset is not None:
            pivot = fbx.FbxNode.EPivotSet.eSourcePivot
            node.SetGeometricTranslation(pivot, fbx.FbxVector4(*offset[0], 1.0))
            node.SetGeometricRotation(pivot, fbx.FbxVector4(*offset[1], 1.0))
            node.SetGeometricScaling(pivot, fbx.FbxVector4(*offset[2], 1.0))
        return node, mesh

    def scene_file(self, name, unit, axes):
        """the file the class docstring describes, in ``unit`` and ``axes``"""

        def populate(scene):
            settings = scene.GetGlobalSettings()
            settings.SetSystemUnit(getattr(fbx.FbxSystemUnit, unit))
            settings.SetAxisSystem(axes)

            group = fbx.FbxNode.Create(scene, "grp")
            group.SetNodeAttribute(fbx.FbxNull.Create(scene, "grpAttr"))
            group.LclTranslation.Set(fbx.FbxDouble3(1.0, 2.0, -3.0))
            group.LclRotation.Set(fbx.FbxDouble3(10.0, 20.0, 30.0))
            group.LclScaling.Set(fbx.FbxDouble3(1.5, 1.5, 1.5))
            scene.GetRootNode().AddChild(group)

            nodes = {}
            for index, (node_name, offset) in enumerate(self.OFFSETS.items()):
                node, mesh = self.cube(scene, node_name, offset)
                node.LclTranslation.Set(fbx.FbxDouble3(4.0 * index, -1.0, 2.0))
                node.LclRotation.Set(fbx.FbxDouble3(0.0, 45.0, 0.0))
                group.AddChild(node)
                nodes[node_name] = node, mesh

            turned = nodes["turned"][0]
            turned.SetRotationActive(True)
            turned.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.EFbxRotationOrder.eEulerZXY)
            turned.LclRotation.Set(fbx.FbxDouble3(10.0, 0.0, 20.0))

            hip  = add_joint(scene, "hip")
            knee = add_joint(scene, "knee", parent=hip)
            hip.LclTranslation.Set(fbx.FbxDouble3(0.0, 1.0, 0.5))
            hip.LclRotation.Set(fbx.FbxDouble3(0.0, 0.0, 10.0))
            knee.LclTranslation.Set(fbx.FbxDouble3(0.0, -0.5, 0.1))
            knee.LclRotation.Set(fbx.FbxDouble3(20.0, 0.0, 0.0))
            body, mesh = nodes["body"]
            weights = [
                (hip, {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5, 4: 1.0, 5: 0.25}),
                (knee, {2: 0.5, 3: 0.5, 5: 0.75, 6: 1.0, 7: 1.0}),
            ]
            skin = bind(scene, mesh, weights)
            for index in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(index)
                cluster.SetTransformMatrix(body.EvaluateGlobalTransform())
                cluster.SetTransformLinkMatrix(cluster.GetLink().EvaluateGlobalTransform())
            knee.LclRotation.Set(fbx.FbxDouble3(50.0, 0.0, 0.0))

        return self.write(name, populate)

    @staticmethod
    def offset(node):
        """the node's geometric transform as the sdk composes it,
        FbxAMatrix(T, R, S), a row-vector 4x4"""
        pivot = fbx.FbxNode.EPivotSet.eSourcePivot
        return _rows(
            fbx.FbxAMatrix(
                node.GetGeometricTranslation(pivot),
                node.GetGeometricRotation(pivot),
                node.GetGeometricScaling(pivot),
            )
        )

    def sdk_truth(self, path, cm, names=None):
        """{mesh: (its points with the offset, times ``cm``; the file's
        face-vertex normals; the offset's 3x3; the sdk's world points, the
        offset's then the node's world after FbxAxisSystem.MayaYUp
        .ConvertScene, times ``cm``)}, for the meshes ``names`` or the
        class's"""
        names = self.OFFSETS if names is None else names
        out   = {}
        with open_fbx(path, fbx) as scene:
            for name in names:
                node      = _find(scene, name)
                mesh      = node.GetNodeAttribute()
                points    = _control_points(mesh)
                offset    = self.offset(node)
                array     = mesh.GetElementNormal(0).GetDirectArray()
                normals   = np.array([list(array.GetAt(i))[:3] for i in range(array.GetCount())])
                local     = (points @ offset[:3, :3] + offset[3, :3]) * cm
                out[name] = [local, normals, offset[:3, :3]]
            fbx.FbxAxisSystem.MayaYUp.ConvertScene(scene)
            for name in names:
                node   = _find(scene, name)
                points = _control_points(node.GetNodeAttribute())
                world  = self.offset(node) @ _rows(node.EvaluateGlobalTransform())
                out[name].append((points @ world[:3, :3] + world[3, :3]) * cm)
        return out

    @staticmethod
    def turned(normals, matrix):
        """the normals as Maya's import turns them by an offset whose 3x3 is
        ``matrix``: at unit length, by its rotation part, flipped when it
        mirrors"""
        unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return np.sign(np.linalg.det(matrix)) * unit @ _polar(matrix)

    def sdk_blend(self, path, cm, turn):
        """the body as the sdk's linear blend of its clusters puts it, the
        offset applied first (the sdk's ViewScene sample, DrawScene.cxx), in
        the file's axes, then turned and times ``cm``"""
        with open_fbx(path, fbx) as scene:
            node   = _find(scene, "body")
            mesh   = node.GetNodeAttribute()
            points = np.c_[_control_points(mesh), np.ones(mesh.GetControlPointsCount())]
            skin   = mesh.GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
            blend  = np.zeros_like(points)
            for index in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(index)
                link_bind, mesh_bind = fbx.FbxAMatrix(), fbx.FbxAMatrix()
                cluster.GetTransformLinkMatrix(link_bind)
                cluster.GetTransformMatrix(mesh_bind)
                link     = _rows(cluster.GetLink().EvaluateGlobalTransform())
                skinning = self.offset(node) @ _rows(mesh_bind) @ np.linalg.inv(_rows(link_bind)) @ link
                for vertex, weight in zip(cluster.GetControlPointIndices(), cluster.GetControlPointWeights()):
                    blend[vertex] += weight * (points[vertex] @ skinning)
        return blend[:, :3] @ turn * cm

    def faces(self, mesh):
        """a MeshData's faces, as lists of point indices"""
        ends = np.cumsum(mesh.counts)
        return [list(np.asarray(mesh.indices)[end - count:end]) for count, end in zip(mesh.counts, ends)]

    def test_the_points_and_normals_take_the_offset(self):
        """every reader: the points with the offset baked, times the unit;
        the normals turned by its rotation alone, flipped in a mirror, unit
        length; the faces in the file's order; a mesh with no offset as the
        file holds it, to the bit"""
        for label, unit, axes, cm, _ in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"points {label}.fbx", unit, axes)
                truth = self.sdk_truth(path, cm)
                read  = {mesh.name: mesh for mesh in MeshList.load_fbx(path)}
                pairs = {mesh.name: mesh for mesh, _ in mesh_load_fbx(path)}
                self.assertEqual(sorted(read), sorted(self.OFFSETS))

                for name in self.OFFSETS:
                    local, normals, matrix, _ = truth[name]
                    for mesh in (read[name], pairs[name], MeshData.load_fbx(path, name=name)):
                        got = np.asarray(mesh.normals)[mesh.normal_indices]
                        self.assertEqual(self.faces(mesh), [list(face) for face in self.FACES])
                        if self.OFFSETS[name] is None:
                            np.testing.assert_array_equal(mesh.points, local)
                            np.testing.assert_array_equal(got, normals)
                            continue
                        np.testing.assert_allclose(mesh.points, local, rtol=0.0, atol=1e-9)
                        np.testing.assert_allclose(got, self.turned(normals, matrix), rtol=0.0, atol=1e-9)
                        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, rtol=0.0, atol=1e-12)

    def test_the_normals_are_not_the_faces_own_under_an_uneven_scale(self):
        """as in Maya: turned by the offset's rotation alone, the normals of
        ``geo`` are not what its inverse transpose, which keeps a normal
        perpendicular to a stretched face, makes of them; ``mirror``'s are
        flipped, its faces kept, so they agree with the face order"""
        def unit(rows):
            return rows / np.linalg.norm(rows, axis=1, keepdims=True)

        path    = self.scene_file("normals.fbx", "cm", fbx.FbxAxisSystem.MayaYUp)
        truth   = self.sdk_truth(path, 1.0)
        read    = {mesh.name: mesh for mesh in MeshList.load_fbx(path)}
        normals = {name: np.asarray(read[name].normals)[read[name].normal_indices] for name in read}

        _, authored, matrix, _ = truth["geo"]
        square = unit(authored @ np.linalg.inv(matrix).T)
        self.assertGreater(float(np.abs(normals["geo"] - square).max()), 0.3)

        _, authored, matrix, _ = truth["mirror"]
        self.assertLess(np.linalg.det(matrix), 0.0)
        unflipped = unit(authored) @ _polar(matrix)
        np.testing.assert_allclose(normals["mirror"], -unflipped, rtol=0.0, atol=1e-9)
        self.assertEqual(self.faces(read["mirror"]), [list(face) for face in self.FACES])

    def test_under_its_nodes_world_a_mesh_is_where_the_sdk_puts_it(self):
        """each mesh's node, from HierarchyData.load_fbx and ClipData.load_fbx
        of the same file, places its points where the sdk does, the offset
        and the node's world"""
        for label, unit, axes, cm, _ in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"worlds {label}.fbx", unit, axes)
                truth = self.sdk_truth(path, cm)
                read  = {mesh.name: np.asarray(mesh.points) for mesh in MeshList.load_fbx(path)}

                for loader in (HierarchyData.load_fbx, ClipData.load_fbx):
                    rig = loader(path)
                    for name in self.OFFSETS:
                        world = np.asarray(rig[name].world_matrix)
                        got   = read[name] @ world[:3, :3] + world[3, :3]
                        np.testing.assert_allclose(got, truth[name][3], rtol=0.0, atol=1e-9)

    def test_a_skinned_mesh_posed_by_the_rig_is_the_sdks_blend(self):
        """the reader's points, the offset baked, and _read_skins_fbx's
        inverse binds as read, deformed by the rig or the clip of the same
        file; and the Object, whose points and binds turn together"""
        for label, unit, axes, cm, turn in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"skin {label}.fbx", unit, axes)
                want  = self.sdk_blend(path, cm, turn)
                mesh  = MeshData.load_fbx(path, name="body")
                skins = {skin.name: (skin, binds) for skin, binds in _read_skins_fbx(path, bind_matrices=True)}

                deformer = SkinDeformData(mesh, skins["body"][0], inverse_bind_matrices=skins["body"][1])
                index    = [x.name for x in MeshList.load_fbx(path)].index("body")
                obj      = Object.load_fbx(path, index=index, extract_texture=False)
                for loader in (HierarchyData.load_fbx, ClipData.load_fbx):
                    rig = loader(path)
                    np.testing.assert_allclose(deformer.apply(rig), want, rtol=0.0, atol=1e-9)
                    np.testing.assert_allclose(obj.pose(rig).mesh.points, want, rtol=0.0, atol=1e-9)
                    obj.restore_bind_pose()

    def test_the_object_turns_the_mesh_with_its_offset(self):
        """Object.load_fbx shows each mesh as MeshList.load_fbx reads it, the
        offset baked, then turned into Maya's Y-up, points and normals"""
        for label, unit, axes, cm, turn in self.cases():
            with self.subTest(file=label):
                path  = self.scene_file(f"object {label}.fbx", unit, axes)
                truth = self.sdk_truth(path, cm)
                for index, mesh in enumerate(MeshList.load_fbx(path)):
                    obj = Object.load_fbx(path, index=index, extract_texture=False, load_skin=False)
                    self.assertEqual(obj.name, mesh.name)
                    np.testing.assert_array_equal(obj.mesh.points, np.asarray(mesh.points) @ turn)
                    np.testing.assert_array_equal(obj.mesh.normals, np.asarray(mesh.normals) @ turn)

                    local, normals, matrix, _ = truth[mesh.name]
                    got = np.asarray(obj.mesh.normals)[obj.mesh.normal_indices]
                    np.testing.assert_allclose(obj.mesh.points, local @ turn, rtol=0.0, atol=1e-9)
                    np.testing.assert_allclose(got, self.turned(normals, matrix) @ turn, rtol=0.0, atol=1e-9)

    def flat_file(self, name, unit, axes):
        """a file of the cubes ``FLAT`` and ``THIN`` offset, under the scene
        root"""

        def populate(scene):
            settings = scene.GetGlobalSettings()
            settings.SetSystemUnit(getattr(fbx.FbxSystemUnit, unit))
            settings.SetAxisSystem(axes)
            for node_name, offset in {**self.FLAT, **self.THIN}.items():
                node, _ = self.cube(scene, node_name, offset)
                scene.GetRootNode().AddChild(node)

        return self.write(name, populate)

    def test_an_offset_that_flattens_the_mesh_drops_its_normals(self):
        """an offset whose scale is 0 on an axis flattens the mesh: Maya's
        FBXImport drops the file's normals and rebuilds its own from the flat
        faces, so every reader reads the mesh with no normals, its points
        flattened where the sdk puts them; a scale near 0 but not 0 does not
        flatten it, and every reader keeps its normals, turned by the rule,
        as Maya keeps them"""
        offsets = {**self.FLAT, **self.THIN}
        for label, unit, axes, cm, turn in self.cases():
            with self.subTest(file=label):
                path   = self.flat_file(f"flat {label}.fbx", unit, axes)
                truth  = self.sdk_truth(path, cm, offsets)
                meshes = MeshList.load_fbx(path)
                pairs  = {mesh.name: mesh for mesh, _ in mesh_load_fbx(path)}
                self.assertEqual(sorted(mesh.name for mesh in meshes), sorted(offsets))

                for index, mesh in enumerate(meshes):
                    local, normals, matrix, _ = truth[mesh.name]
                    self.assertEqual(len(normals), len(self.FACES) * 4)
                    obj   = Object.load_fbx(path, index=index, extract_texture=False, load_skin=False)
                    reads = (mesh, pairs[mesh.name], MeshData.load_fbx(path, name=mesh.name), obj.mesh)
                    for read, into in zip(reads, (np.eye(3), np.eye(3), np.eye(3), turn)):
                        if mesh.name not in self.THIN:
                            self.assertIsNone(read.normals)
                            self.assertIsNone(read.normal_indices)
                            continue
                        self.assertIsNotNone(read.normals)
                        got = np.asarray(read.normals)[read.normal_indices]
                        np.testing.assert_allclose(got, self.turned(normals, matrix) @ into, rtol=0.0, atol=1e-9)
                    np.testing.assert_allclose(mesh.points, local, rtol=0.0, atol=1e-9)
                    np.testing.assert_allclose(obj.mesh.points, local @ turn, rtol=0.0, atol=1e-9)

    def test_maya_imports_the_mesh_where_the_readers_put_it(self):
        """Maya's own FBXImport: its object-space points are the reader's,
        over the cm one unit of the file holds (Maya keeps a file's unit in
        its object space and scales the node), and under their node's world
        from HierarchyData.load_fbx the points are where Maya puts the mesh,
        both within the gap between Maya's points and the sdk's (Maya holds
        a mesh's points in single precision) plus 1e-9; its face-vertex
        normals are the reader's within 2e-7 (single precision: at most
        1.2e-7 measured), its faces in the file's order. Needs mayapy with
        the fbxmaya plug-in"""
        try:
            import maya.api.OpenMaya as om
            import maya.cmds as cmds
            import maya.mel as mel

            cmds.loadPlugin("fbxmaya", quiet=True)
        except Exception:  # not running in Maya, or no fbx plug-in
            self.skipTest("needs Maya and its fbxmaya plug-in")

        def shape(name):
            selection = om.MSelectionList()
            selection.add(name)
            path = selection.getDagPath(0)
            path.extendToShape()
            return om.MFnMesh(path)

        def rows(values):
            return np.array([[value.x, value.y, value.z] for value in values])

        self.addCleanup(cmds.file, new=True, force=True)
        for label, unit, axes, cm, _ in self.cases():
            path  = self.scene_file(f"maya {label}.fbx", unit, axes)
            truth = self.sdk_truth(path, cm)
            read  = {mesh.name: mesh for mesh in MeshList.load_fbx(path)}
            rig   = HierarchyData.load_fbx(path)
            cmds.file(new=True, force=True)
            mel.eval("FBXResetImport")
            mel.eval(f'FBXImport -f "{path.replace(os.sep, "/")}"')

            for name in ("geo", "turned", "mirror", "plain", "scaled"):
                with self.subTest(file=label, mesh=name):
                    mesh  = read[name]
                    maya  = shape(name)
                    faces = [list(maya.getPolygonVertices(face)) for face in range(maya.numPolygons)]
                    local = rows(maya.getPoints(om.MSpace.kObject))
                    world = rows(maya.getPoints(om.MSpace.kWorld))
                    normals = rows(
                        normal
                        for face in range(maya.numPolygons)
                        for normal in maya.getFaceVertexNormals(face, om.MSpace.kObject)
                    )
                    self.assertEqual(faces, self.faces(mesh))

                    gap = float(np.abs(local - truth[name][0] / cm).max())
                    self.assertLess(gap, 1e-6)
                    got = np.asarray(mesh.points) / cm
                    self.assertLessEqual(float(np.abs(got - local).max()), gap + 1e-9)

                    gap = float(np.abs(world - truth[name][3]).max())
                    self.assertLess(gap, 1e-6 * max(1.0, float(np.abs(world).max())))
                    node = np.asarray(rig[name].world_matrix)
                    got  = np.asarray(mesh.points) @ node[:3, :3] + node[3, :3]
                    self.assertLessEqual(float(np.abs(got - world).max()), gap + 1e-9)

                    got = np.asarray(mesh.normals)[mesh.normal_indices]
                    self.assertLess(float(np.abs(got - normals).max()), 2e-7)


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestGlbBindPose(GlbCase):
    """same rule for glb, with the unit conversion the fbx path also does

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