"""Tests for the render Scene: what an fbx holds and the render settings,
placed by the nodes, shaded by the materials and rendered by the raytracer."""

import base64
import io
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
from cgmath.geometry import CameraData, LightData, MaterialData, MorphData, SkinData
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, generate_uuid
from cgmath.render import aim, look_at, render, Scene
from cgmath.render.scene import (
    _channel_offsets,
    _default_lighting_rig_for,
    _RENDER_CONFIG_KEYS,
    _rotation_matrix_col,
    _SHADING_KEYS,
)

# PIL is optional throughout the rest of the suite; expose under a stable
# alias so tests that need it can opt in (encoder round-trips, etc.).
try:
    from PIL import Image as _PIL_Image_module

    Image = _PIL_Image_module
except ImportError:
    Image = None

# cv2 / skimage detection -- the wireframe-bake path needs at least one
# rasterizer.  When neither is available (e.g., mayapy without optional
# deps), wireframe tests should self-skip rather than error out deep in
# UVData.draw_edges.
try:
    import cv2 as _cv2_module

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
try:
    import skimage as _skimage_module

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
_HAS_RASTERIZER = _HAS_CV2 or _HAS_SKIMAGE

INCH = 25.4


def _make_textured_cube():
    """a triangulated unit cube and its UV set (one [0, 1]^2 island per
    face), the set naming the mesh, at a small bake resolution"""
    points = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
            [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        ],
        dtype=float,
    )
    quads = [
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (2, 3, 7, 6),
        (1, 2, 6, 5), (0, 4, 7, 3),
    ]
    indices, counts = [], []
    for q in quads:
        indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]])
        counts.extend([3, 3])
    mesh = MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    )
    uv_pts, uv_idx = [], []
    for fi in range(6):
        base = fi * 4
        uv_pts.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        uv_idx.extend([base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])
    uv = UVData(
        points    = np.asarray(uv_pts, dtype=float),
        indices   = np.asarray(uv_idx, dtype=np.int64),
        counts    = np.asarray(counts, dtype=np.int64),
        mesh_uuid = mesh.uuid,
        name      = "map1",
    )
    uv.resolution = (64, 64)
    return mesh, uv


def _make_checker(n=64, squares=4):
    tex = np.zeros((n, n, 3), dtype=np.uint8)
    s   = n // squares
    for j in range(squares):
        for i in range(squares):
            tex[j * s : (j + 1) * s, i * s : (i + 1) * s] = (
                (220, 220, 220) if (i + j) % 2 == 0 else (40, 40, 40)
            )
    return tex


def _node(name, **channels):
    """a node with its uuid already, so a shape can name it before it joins
    a scene (a node appended without one gets its uuid then)"""
    return TransformData(name, uuid=generate_uuid(), **channels)


def _camera(angle_of_view=35.0, **fields):
    """a CameraData whose vertical angle of view is ``angle_of_view``"""
    aperture = CameraData.vertical_film_aperture * INCH
    return CameraData(focal_length=aperture / (2.0 * math.tan(math.radians(angle_of_view) / 2.0)), **fields)


def _under(shape, node):
    """``shape`` placed by ``node``"""
    shape.parent_uuid = node.uuid
    return shape


def _material(mesh, texture=None, color=(1.0, 1.0, 1.0), **fields):
    """a material covering every face of ``mesh``, its colour as given (the
    diffuse factor 1, so the colour is the base colour)"""
    return MaterialData(mesh_uuid=mesh.uuid, texture=texture, color=color, diffuse=1.0, **fields)


def _scene(*items, **settings):
    return Scene("s", **settings).add(*items)


def _hits(frame):
    """the pixels that are not the frame's top-left corner, the background"""
    return np.any(frame.array != frame.array[0, 0], axis=-1)


def _looking(translate, target=(0.0, 0.0, 0.0), angle_of_view=35.0, **fields):
    """a camera node at ``translate`` aimed at ``target``, and a camera on
    it with ``angle_of_view``"""
    node = _node("cam", translate=list(translate))
    aim(node, target)
    return node, _camera(angle_of_view, parent_uuid=node.uuid, **fields)


class TestSceneBuilding(unittest.TestCase):
    """add(), the typed lists, camera(), pose(), aim() and the placement"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_add_puts_each_item_in_its_list(self):
        node   = _node("cube")
        camera = _camera(parent_uuid=node.uuid)
        scene  = Scene("s")
        self.assertIs(scene.add(node, _under(self.mesh, node), self.uv, camera, LightData(name="key")), scene)
        scene.add(
            _material(self.mesh),
            MorphData("puff", np.zeros((1, 3)), [0], mesh_uuid=self.mesh.uuid),
            SkinData(weights=np.ones((8, 1)), influences=["cube"], mesh_uuid=self.mesh.uuid),
        )
        held = (scene.hierarchy, scene.meshes, scene.uvs, scene.cameras, scene.lights, scene.materials, scene.morphs, scene.skins)
        self.assertEqual(
            [type(x).__name__ for x in held],
            ["HierarchyData", "MeshList", "UVList", "CameraList", "LightList", "MaterialList", "MorphList", "SkinList"],
        )
        self.assertEqual([len(x) for x in held], [1] * 8)
        self.assertIs(scene.hierarchy[0], node)
        self.assertEqual(scene.meshes[0].uuid, self.mesh.uuid)
        self.assertEqual(scene.uvs[0].mesh_uuid, self.mesh.uuid)

    def test_a_list_a_hierarchy_or_a_clip_is_taken_whole(self):
        node  = _node("root")
        scene = Scene("s")
        scene.add([node, self.mesh, self.uv])
        self.assertEqual((len(scene.hierarchy), len(scene.meshes), len(scene.uvs)), (1, 1, 1))

        other = HierarchyData([_node("a"), _node("b")])
        scene.add(other)
        self.assertEqual(scene.hierarchy.name, ["root", "a", "b"])

        clip = ClipData(HierarchyData([_node("c")]), frames=2)
        scene.add(clip)
        self.assertEqual(len(scene.clip), 1)
        self.assertEqual(scene.clip.active.frame_count, 2)

    def test_add_refuses_anything_else(self):
        with self.assertRaisesRegex(TypeError, "Scene.add takes"):
            Scene("s").add({"not": "a shape"})
        with self.assertRaises(TypeError):
            Scene("s").add(42)

    def test_the_lists_hold_one_kind(self):
        scene        = Scene("s")
        scene.meshes = self.mesh
        self.assertEqual(type(scene.meshes).__name__, "MeshList")
        scene.meshes = [self.mesh]
        self.assertEqual(len(scene.meshes), 1)
        with self.assertRaisesRegex(TypeError, "Scene.meshes holds MeshData"):
            scene.meshes = [self.uv]  # a UVData is a MeshData, and goes in uvs
        with self.assertRaisesRegex(TypeError, "Scene.lights holds LightData"):
            scene.lights = [_camera()]
        with self.assertRaisesRegex(TypeError, "pass a ClipData as clip="):
            scene.hierarchy = ClipData(HierarchyData([_node("a")]), frames=1)
        with self.assertRaisesRegex(TypeError, "Scene.clip is a ClipList or a ClipData"):
            scene.clip = HierarchyData()
        scene.hierarchy = _node("solo")
        self.assertEqual(scene.hierarchy.name, ["solo"])
        scene.clip = ClipData(HierarchyData([_node("a")]), frames=3)
        self.assertEqual(type(scene.clip).__name__, "ClipList")

    def test_camera_picks_the_name_the_default_or_the_first(self):
        node_a, node_b = _node("a_node"), _node("b_node")
        a     = _camera(name="A", parent_uuid=node_a.uuid)
        b     = _camera(name="B", parent_uuid=node_b.uuid)
        scene = _scene(node_a, node_b, a, b)
        self.assertEqual(scene.camera(), (a, node_a.uuid))
        self.assertEqual(scene.camera("B"), (b, node_b.uuid))
        scene.default_camera = "B"
        self.assertEqual(scene.camera()[0].name, "B")
        self.assertEqual(Scene("empty").camera(), (None, None))
        self.assertIsNone(_scene(CameraData(name="loose")).camera()[1])

    def test_pose_is_the_hierarchy_or_the_active_takes_frame(self):
        node  = _node("root")
        scene = _scene(node)
        self.assertIs(scene.pose(), scene.hierarchy)

        clip                          = ClipData(scene.hierarchy, frames=3, start_frame=10)
        clip.frames["root"].translate = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        scene.clip                    = clip
        np.testing.assert_allclose(scene.pose()["root"].translate, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(scene.pose(11)["root"].translate, [1.0, 0.0, 0.0])
        self.assertIsNone(Scene("bare").pose())

    def test_aim_turns_a_node_to_look_down_its_minus_z(self):
        node = _node("cam", translate=[5.0, 5.0, 5.0], rotate_axis=[0.0, 0.0, 20.0])
        self.assertIs(aim(node, (0.0, 0.0, 0.0)), node)
        # row 2 of a world matrix is the node's Z axis: a camera looks down -Z
        forward  = -node.world_matrix[2, :3]
        expected = -np.array([5.0, 5.0, 5.0]) / np.linalg.norm([5.0, 5.0, 5.0])
        np.testing.assert_allclose(forward, expected, atol=1e-9)
        np.testing.assert_allclose(node.world_matrix, look_at((5.0, 5.0, 5.0), (0.0, 0.0, 0.0)).T, atol=1e-9)
        np.testing.assert_allclose(node.rotate_axis, [0.0, 0.0, 20.0])  # the channels solve around it

    def test_union_aabb_places_each_mesh_by_its_node(self):
        far = _node("far", translate=[10.0, 0.0, 0.0])
        other, _ = _make_textured_cube()
        scene = _scene(far, self.mesh, _under(other, far))
        mn, mx = scene.union_aabb()
        np.testing.assert_allclose(mn, [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(mx, [10.5, 0.5, 0.5])
        self.assertEqual(scene.union_points().shape, (16, 3))
        self.assertEqual(Scene("empty").union_points().shape, (0, 3))
        np.testing.assert_array_equal(Scene("empty").union_aabb(), np.zeros((2, 3)))

    def test_placed_meshes_leaves_hidden_nodes_out(self):
        group = _node("group", visibility=False)
        child = _node("child")
        loose, _ = _make_textured_cube()
        scene = _scene(group, child, _under(self.mesh, child), loose)
        child.set_parent(group)

        placed = scene.placed_meshes()
        self.assertEqual([m.uuid for m, _ in placed], [loose.uuid])  # under a hidden parent: hidden too
        np.testing.assert_array_equal(placed[0][1], np.eye(4))       # a mesh on no node sits at identity

        group.visibility = True
        placed           = scene.placed_meshes()
        self.assertEqual(len(placed), 2)
        self.assertIs(placed[0][0], scene.meshes[0])  # an unskinned mesh is itself, not a copy
        np.testing.assert_allclose(placed[0][1], child.world_matrix)


class TestSceneRendering(unittest.TestCase):
    """the raytracer through a Scene"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()
        self.tex = _make_checker()

    def test_a_scene_renders_through_render_and_its_own_method(self):
        scene = _scene(self.mesh, self.uv, _material(self.mesh, self.tex))
        img   = render(scene=scene, resolution=(64, 48), default_light=False)
        self.assertEqual(img.shape, (48, 64, 4))
        self.assertEqual(img.dtype, np.uint8)
        # the cube is in front of the auto-built camera, so most pixels hit
        self.assertGreater(int(_hits(img).sum()), 100)
        own = scene.render(resolution=(64, 48), default_light=False)
        np.testing.assert_array_equal(own, img)
        self.assertIs(scene.frame, own)

    def test_the_scenes_camera_matches_the_camera_matrix_path(self):
        """the same content through a CameraData on a node and through the
        legacy camera_matrix= argument: the pixels agree exactly"""
        cam_node, camera = _looking((2.0, 2.0, 2.0))
        scene     = _scene(cam_node, self.mesh, self.uv, _material(self.mesh, self.tex), camera)
        img_scene = render(scene=scene, resolution=(64, 48), default_light=False)
        img_legacy = render(
            self.mesh,
            self.uv,
            self.tex,
            camera_matrix = look_at(eye=(2.0, 2.0, 2.0), target=(0.0, 0.0, 0.0)),
            angle_of_view = camera.vertical_angle_of_view,
            resolution    = (64, 48),
            ambient       = scene.ambient,
            point_light   = None,
            default_light = False,
        )
        np.testing.assert_array_equal(img_scene, img_legacy)

    def test_moving_a_mesh_node_shifts_its_silhouette(self):
        frames = []
        for x in (0.0, 1.0):
            mesh, uv = _make_textured_cube()
            node = _node("cube", translate=[x, 0.0, 0.0])
            cam_node, camera = _looking((0.0, 0.0, 5.0))
            scene = _scene(node, cam_node, _under(mesh, node), uv, _material(mesh, self.tex), camera)
            frames.append(render(scene=scene, resolution=(64, 48), default_light=False))
        self.assertFalse(np.array_equal(frames[0], frames[1]))
        # the cube moved to +x shows up further right (larger pixel column)
        centres = [float(np.where(_hits(f))[1].mean()) for f in frames]
        self.assertGreater(centres[1], centres[0])

    def test_two_lights_add(self):
        def lit(count):
            mesh, uv = _make_textured_cube()
            cam_node, camera = _looking((0.0, 0.0, 5.0))
            scene = _scene(cam_node, mesh, uv, _material(mesh), camera, ambient=0.0)
            for i in range(count):
                node = _node(f"L{i}", translate=[3.0, 3.0, 5.0])
                scene.add(node, LightData(name=f"L{i}", kind="point", intensity=10.0, decay_rate=2, parent_uuid=node.uuid))
            return render(scene=scene, resolution=(48, 36), default_light=False)

        one, two = lit(1), lit(2)
        mask = _hits(one) & _hits(two)
        self.assertTrue(mask.any())
        self.assertGreater(float(two.array[mask][:, :3].mean()), float(one.array[mask][:, :3].mean()))

    def test_the_default_rig_lights_an_unlit_scene(self):
        """with no light, default_light adds the 3-point rig and the faces
        shade apart; without it the raytracer draws the flat colour (its
        no-light convention)"""
        scene     = _scene(self.mesh, self.uv, _material(self.mesh, color=(0.7, 0.7, 0.7)), ambient=0.0)
        lit       = scene.render(resolution=(48, 36), default_light=True)
        flat      = scene.render(resolution=(48, 36), default_light=False)
        lit_hits  = lit.array[_hits(lit)][:, :3]
        flat_hits = flat.array[_hits(flat)][:, :3]
        self.assertGreater(int(lit_hits.max()), 30)
        self.assertGreater(len(np.unique(lit_hits, axis=0)), 1)  # shaded
        self.assertEqual(len(np.unique(flat_hits, axis=0)), 1)   # one flat colour, 0.7 grey
        self.assertIn(int(flat_hits[0, 0]), (178, 179))


class TestMaterials(unittest.TestCase):
    """each face is shaded by the material covering it; the colour is the
    material's colour times its diffuse, the texture read through the UV
    set it names, a face no material covers the scene's base colour"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()
        # the default 3/4 view sees the +x, +y and +z faces: triangles 8 and
        # 9, 6 and 7, 2 and 3 of the cube

    def seen(self, scene, resolution=(64, 64)):
        """the colours of the hit pixels, as RGB triples"""
        frame = scene.render(resolution=resolution, default_light=False)
        return {tuple(x) for x in frame.array[_hits(frame)][:, :3].tolist()}

    def test_each_material_colours_the_faces_it_covers(self):
        scene = _scene(
            self.mesh,
            MaterialData(mesh_uuid=self.mesh.uuid, faces=[2, 3], color=(1.0, 0.0, 0.0), diffuse=1.0),
            MaterialData(mesh_uuid=self.mesh.uuid, faces=[6, 7], color=(0.0, 1.0, 0.0), diffuse=1.0),
            MaterialData(mesh_uuid=self.mesh.uuid, faces=[8, 9], color=(0.0, 0.0, 1.0), diffuse=1.0),
            ambient    = 1.0,
            base_color = (1.0, 1.0, 1.0),
        )
        seen = self.seen(scene)
        for colour in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
            self.assertIn(colour, seen)
        self.assertLessEqual(seen, {(255, 0, 0), (0, 255, 0), (0, 0, 255)})

    def test_an_uncovered_face_takes_the_scenes_base_colour(self):
        scene = _scene(
            self.mesh,
            MaterialData(mesh_uuid=self.mesh.uuid, faces=[2, 3], color=(1.0, 0.0, 0.0), diffuse=1.0),
            ambient    = 1.0,
            base_color = (0.0, 1.0, 0.0),
        )
        seen = self.seen(scene)
        self.assertIn((255, 0, 0), seen)
        self.assertIn((0, 255, 0), seen)

    def test_the_colour_is_the_materials_colour_times_its_diffuse(self):
        scene = _scene(self.mesh, MaterialData(mesh_uuid=self.mesh.uuid, color=(1.0, 1.0, 1.0), diffuse=0.5), ambient=1.0)
        seen  = self.seen(scene, (32, 32))
        self.assertTrue(seen and seen <= {(127, 127, 127), (128, 128, 128)}, seen)

    def test_a_textured_material_reads_the_uv_set_it_names(self):
        other        = self.uv.copy()
        other.name   = "flat"
        other.points = np.full_like(other.points, 0.0)  # every corner on one dark texel
        checker      = _make_checker()
        scene = _scene(
            self.mesh, self.uv, other,
            MaterialData(mesh_uuid=self.mesh.uuid, texture=checker, uv_set="map1", color=(1.0, 1.0, 1.0), diffuse=1.0),
            ambient=1.0,
        )
        seen = self.seen(scene)
        self.assertTrue(any(c[0] >= 200 for c in seen))
        self.assertTrue(any(c[0] <= 60 for c in seen))

        scene.materials[0].uv_set = "flat"
        self.assertTrue(all(c[0] <= 60 for c in self.seen(scene)))

        # no set named: the mesh's first
        scene.materials[0].uv_set = None
        self.assertTrue(any(c[0] >= 200 for c in self.seen(scene)))

    def test_a_texture_without_a_uv_set_raises(self):
        scene = _scene(self.mesh, _material(self.mesh, _make_checker()))
        with self.assertRaisesRegex(ValueError, "has a texture but no UV set"):
            scene.render(resolution=(8, 8))

    def test_the_entries_carry_each_materials_faces(self):
        red       = MaterialData(mesh_uuid=self.mesh.uuid, faces=[0, 1], color=(1.0, 0.0, 0.0), diffuse=0.5)
        scene     = _scene(self.mesh, self.uv, red, MaterialData(mesh_uuid=generate_uuid(), color=(0.0, 0.0, 1.0)))
        entry     = scene._render_inputs().entries[0]
        materials = entry["materials"]
        self.assertEqual(len(materials), 1)  # another mesh's material is not this mesh's
        np.testing.assert_array_equal(materials[0]["faces"], [True, True] + [False] * 10)
        np.testing.assert_allclose(materials[0]["base_color"], [0.5, 0.0, 0.0])
        self.assertEqual(materials[0]["uv"].uuid, self.uv.uuid)
        self.assertIsNone(materials[0]["tex"])
        self.assertEqual(materials[0]["wrap"], "repeat")
        np.testing.assert_allclose(entry["base_color"], scene.base_color)
        np.testing.assert_array_equal(entry["world_matrix"], np.eye(4))

        # no material: one of the scene's base colour over every face
        bare = _scene(self.mesh)._render_inputs().entries[0]["materials"]
        self.assertEqual(len(bare), 1)
        self.assertIsNone(bare[0]["faces"])
        np.testing.assert_allclose(bare[0]["base_color"], [0.7, 0.7, 0.7])


class TestMorphTargets(unittest.TestCase):
    """blend shape targets move a mesh's points by their weights, Maya's way"""

    def test_one_target_is_scaled_by_the_weight(self):
        target = MorphData("puff", [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], [1, 3])
        np.testing.assert_allclose(
            _channel_offsets([target], 0.5, 4),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(_channel_offsets([target], 2.0, 4)[3], [0.0, 0.0, 4.0])

    def test_an_in_between_is_a_knot_the_weight_blends_through(self):
        full = MorphData("puff", [[0.0, 0.0, 1.0]], [0])
        half = MorphData("puff", [[0.0, 0.0, 0.8]], [0], at=0.5)  # off the straight line
        for weight, z in ((0.0, 0.0), (0.25, 0.4), (0.5, 0.8), (0.75, 0.9), (1.0, 1.0), (1.5, 1.2)):
            with self.subTest(weight=weight):
                # in either order
                for knots in ([full, half], [half, full]):
                    np.testing.assert_allclose(_channel_offsets(knots, weight, 1), [[0.0, 0.0, z]], atol=1e-12)

    def test_placed_meshes_moves_the_points_by_the_weight(self):
        mesh, _ = _make_textured_cube()
        morph = MorphData("puff", [[0.0, 0.0, 0.25]] * 4, [4, 5, 6, 7], mesh_uuid=mesh.uuid, weight=1.0)
        scene = _scene(mesh, morph)
        placed, world = scene.placed_meshes()[0]
        self.assertIsNot(placed, mesh)
        np.testing.assert_allclose(placed.points[4:, 2], 0.75)
        np.testing.assert_allclose(placed.points[:4],    mesh.points[:4])
        np.testing.assert_allclose(mesh.points[4:, 2],   0.5)  # the mesh itself is not moved
        np.testing.assert_array_equal(world, np.eye(4))

        morph.weight = 0.0
        self.assertIs(scene.placed_meshes()[0][0], mesh)

    def test_channels_add_and_deformers_apart_are_separate_channels(self):
        mesh, _ = _make_textured_cube()
        scene = _scene(
            mesh,
            MorphData("a", [[1.0, 0.0, 0.0]], [0], mesh_uuid=mesh.uuid, weight=1.0, deformer="face"),
            MorphData("b", [[0.0, 1.0, 0.0]], [0], mesh_uuid=mesh.uuid, weight=1.0, deformer="face"),
            MorphData("a", [[0.0, 0.0, 1.0]], [0], mesh_uuid=mesh.uuid, weight=1.0, deformer="body"),
        )
        placed, _ = scene.placed_meshes()[0]
        np.testing.assert_allclose(placed.points[0] - mesh.points[0], [1.0, 1.0, 1.0])

    def test_a_weight_change_drops_the_kept_frame(self):
        mesh, _ = _make_textured_cube()
        morph        = MorphData("puff", [[0.0, 0.0, 0.25]], [4], mesh_uuid=mesh.uuid, weight=0.3)
        scene        = _scene(mesh, morph)
        scene.frame  = object()
        morph.weight = 0.7
        self.assertIsNone(scene.frame)


class TestLights(unittest.TestCase):
    """the lights as the raytracer takes them"""

    def lights(self, *items):
        return _scene(*items)._render_inputs().lights

    def test_a_point_light_sits_at_its_node(self):
        node = _node("key", translate=[1.0, 2.0, 3.0])
        [light] = self.lights(node, LightData(kind="point", color=(1.0, 0.5, 0.25), intensity=7.0, parent_uuid=node.uuid))
        self.assertEqual(light["kind"], "point")
        np.testing.assert_allclose(light["position"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(light["color"], [1.0, 0.5, 0.25])
        self.assertEqual(light["intensity"], 7.0)
        self.assertIs(light["falloff"], False)  # Maya's decay none

    def test_a_decay_rate_gives_falloff(self):
        [light] = self.lights(LightData(kind="point", decay_rate=2))
        self.assertIs(light["falloff"], True)
        np.testing.assert_allclose(light["position"], [0.0, 0.0, 0.0])  # on no node: the origin

    def test_a_directional_light_is_infinite_down_its_nodes_minus_z(self):
        [light] = self.lights(LightData(kind="directional"))
        self.assertEqual(light["kind"], "infinite")
        np.testing.assert_allclose(light["direction"], [0.0, 0.0, -1.0])

        node = _node("sun", rotate=[-50.0, 30.0, 10.0])
        [light] = self.lights(node, LightData(kind="directional", parent_uuid=node.uuid))
        axis = -node.world_matrix[2, :3]
        np.testing.assert_allclose(light["direction"], axis / np.linalg.norm(axis), atol=1e-12)

    def test_a_spot_light_is_a_point_light_with_a_warning(self):
        with self.assertWarnsRegex(UserWarning, "rendered as point lights"):
            [light] = self.lights(LightData(name="spot1", kind="spot"))
        self.assertEqual(light["kind"], "point")

    def test_a_light_on_a_hidden_node_is_left_out(self):
        hidden = _node("hidden", visibility=False)
        self.assertEqual(self.lights(hidden, LightData(parent_uuid=hidden.uuid)), [])


class TestSkinnedMesh(unittest.TestCase):
    """a skinned mesh is deformed from the bind pose the hierarchy holds"""

    def setUp(self):
        self.root = _node("root", node_type="joint")
        self.tri = MeshData(
            points  = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            indices = np.arange(3),
            counts  = np.array([3]),
            name    = "tri",
        )
        self.skin = SkinData(weights=np.ones((3, 1)), influences=["root"], mesh_uuid=self.tri.uuid)

    def test_the_pose_deforms_a_copy_in_world_space(self):
        scene = _scene(self.root, self.tri, self.skin)
        placed, world = scene.placed_meshes()[0]
        np.testing.assert_allclose(placed.points, self.tri.points, atol=1e-12)  # the bind pose is a fixed point
        np.testing.assert_array_equal(world, np.eye(4))

        clip                          = ClipData(scene.hierarchy, frames=2)
        clip.frames["root"].translate = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        scene.clip                    = clip
        placed, _ = scene.placed_meshes(clip.start_frame + 1)[0]
        self.assertIsNot(placed, self.tri)
        np.testing.assert_allclose(placed.points, self.tri.points + [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(self.tri.points[:, 0], [0.0, 1.0, 0.0])  # the mesh stays at the bind

    def test_a_skin_without_the_bind_pose_raises(self):
        scene = _scene(self.tri, self.skin)
        with self.assertRaisesRegex(ValueError, "is skinned, which takes the bind pose"):
            scene.placed_meshes()

    def test_the_deformer_is_kept_while_its_inputs_stay(self):
        scene = _scene(self.root, self.tri, self.skin)
        first = scene._deformer(self.tri, self.skin)
        self.assertIs(scene._deformer(self.tri, self.skin), first)
        other = SkinData(weights=np.ones((3, 1)), influences=["root"], mesh_uuid=self.tri.uuid)
        self.assertIsNot(scene._deformer(self.tri, other), first)


class TestOrthographic(unittest.TestCase):
    """an orthographic CameraData: its width over the frame's aspect"""

    def test_ortho_renders_image(self):
        mesh, uv = _make_textured_cube()
        cam_node, camera = _looking((0.0, 0.0, 5.0), orthographic=True, orthographic_width=2.5 * 64 / 48)
        scene = _scene(cam_node, mesh, uv, camera)

        img, depth = render(
            scene         = scene,
            resolution    = (64, 48),
            default_light = False,
            return_depth  = True,
        )
        self.assertEqual(img.shape, (48, 64, 4))
        # parallel rays: the depth is the same across the cube's front face
        hit_depth = depth[~np.isnan(depth)]
        self.assertGreater(hit_depth.size, 10)
        self.assertLess(hit_depth.std(), 0.6)

    def test_the_width_sets_how_much_is_seen(self):
        mesh, uv = _make_textured_cube()
        hits = []
        for width in (1.5, 6.0):
            cam_node, camera = _looking((0.0, 0.0, 5.0), orthographic=True, orthographic_width=width)
            frame = _scene(cam_node, mesh, uv, camera).render(resolution=(32, 32), default_light=False)
            hits.append(int(_hits(frame).sum()))
        self.assertGreater(hits[0], hits[1])


class TestScaledNode(unittest.TestCase):
    """a mesh on a scaled node renders as the mesh scaled in world space would"""

    def test_a_scaled_node_depth_matches_a_scaled_mesh(self):
        """rays cast in object space are not unit length: the depth is still the world one"""
        for scale in (2.0, 0.5):
            with self.subTest(scale=scale):
                depths = []
                for by_node in (True, False):
                    mesh, uv = _make_textured_cube()
                    scene = Scene("s")
                    if by_node:
                        node = _node("c", scale=[scale] * 3)
                        scene.add(node, _under(mesh, node), uv)
                    else:
                        mesh.points = mesh.points * scale
                        scene.add(mesh, uv)
                    cam_node, camera = _looking((0.0, 0.0, 5.0))
                    scene.add(cam_node, camera)
                    _, depth = render(scene=scene, resolution=(32, 24), default_light=False, return_depth=True)
                    depths.append(depth)
                hit = ~np.isnan(depths[1])
                self.assertGreater(int(hit.sum()), 10)
                self.assertTrue(np.array_equal(np.isnan(depths[0]), ~hit))
                self.assertTrue(np.allclose(depths[0][hit], depths[1][hit], rtol=0, atol=1e-9))


@unittest.skipIf(Image is None, "PIL is not installed")
class TestTurntable(unittest.TestCase):
    """Scene.turntable() short loop."""

    def test_turntable_writes_frames(self):
        mesh, uv = _make_textured_cube()
        key   = _node("key", translate=[3.0, 3.0, 3.0])
        scene = _scene(mesh, uv, _material(mesh, _make_checker()), key)
        scene.add(LightData(name="key", kind="point", intensity=50.0, decay_rate=2, parent_uuid=key.uuid))

        with tempfile.TemporaryDirectory() as tmp:
            pattern = os.path.join(tmp, "turn.{frame:03d}.png")
            files = scene.turntable(
                pattern,
                n_frames      = 4,
                resolution    = (48, 36),
                default_light = False,
            )
            self.assertEqual(len(files), 4)
            for f in files:
                self.assertTrue(os.path.exists(f))

            # frame 0 and frame 2 (180 degrees apart) look different
            img0 = np.asarray(Image.open(files[0]))
            img2 = np.asarray(Image.open(files[2]))
            self.assertFalse(np.array_equal(img0, img2))

    def test_turntable_preserves_bvh_cache(self):
        """the mesh's cached BVH is not rebuilt across turntable frames"""
        # a mesh over the size threshold, so a BVH is built at all
        n      = 12
        xs     = np.linspace(-1.0, 1.0, n + 1)
        zs     = np.linspace(-1.0, 1.0, n + 1)
        points = np.array([(x, 0.0, z) for z in zs for x in xs], dtype=float)
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
        mesh = MeshData(
            points  = points,
            indices = np.asarray(indices, dtype=np.int64),
            counts  = np.asarray(counts, dtype=np.int64),
        )

        scene      = _scene(mesh)
        _          = render(scene=scene, resolution=(32, 24), default_light=False)
        bvh_before = mesh._bvh_bilinear
        self.assertIsNotNone(bvh_before)

        with tempfile.TemporaryDirectory() as tmp:
            scene.turntable(
                os.path.join(tmp, "turn.{frame:03d}.png"),
                n_frames      = 3,
                resolution    = (32, 24),
                default_light = False,
            )

        self.assertIs(bvh_before, mesh._bvh_bilinear, "BVH cache should persist across turntable frames")


class TestTurntableGlobalPivot(unittest.TestCase):
    """the camera orbits the world axis through the scene's centroid; the
    nodes are not touched; the fit holds every sampled angle"""

    def _scene_with_three(self):
        """3 cubes spread along X at -3 / 0 / +3."""
        scene = Scene("multi")
        for name, x in (("A", 0.0), ("B", 3.0), ("C", -3.0)):
            node = _node(name, translate=[x, 0.0, 0.0])
            mesh, uv = _make_textured_cube()
            scene.add(node, _under(mesh, node), uv)
        return scene

    def test_no_targets_kwarg(self):
        """the API has no ``target=`` keyword: passing it errors"""
        scene = self._scene_with_three()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                scene.turntable(
                    target         = "A",
                    output_pattern = os.path.join(tmp, "x.{frame:03d}.png"),
                    n_frames       = 2,
                    resolution     = (32, 24),
                    default_light  = False,
                )

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_the_nodes_are_not_mutated(self):
        """the turntable rotates the camera, not the nodes: every node's
        world matrix is byte for byte what it was"""
        scene = self._scene_with_three()
        a, b, c = scene.hierarchy
        a.rotate_y = 17.0
        b.rotate_x = 33.0
        c.rotate   = [11.0, 22.0, 33.0]
        before     = [node.world_matrix.copy() for node in (a, b, c)]

        with tempfile.TemporaryDirectory() as tmp:
            scene.turntable(
                output_pattern = os.path.join(tmp, "spin.{frame:03d}.png"),
                n_frames       = 4,
                resolution     = (48, 36),
                default_light  = False,
            )

        for node, held in zip((a, b, c), before):
            np.testing.assert_array_equal(node.world_matrix, held)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_camera_orbits_world_axis(self):
        """frames at 0 and 90 degrees look distinct"""
        scene = self._scene_with_three()
        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "orbit.{frame:03d}.png"),
                n_frames       = 4,  # frames at 0, 90, 180, 270
                resolution     = (64, 48),
                default_light  = True,
            )
            img0 = np.asarray(Image.open(files[0]))
            img1 = np.asarray(Image.open(files[1]))  # 90 degrees away
            self.assertFalse(np.array_equal(img0, img1), "0 and 90 degree frames should differ -- camera should orbit")

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_translated_meshes_stay_in_frame(self):
        """every shown mesh stays in frame across the rotation"""
        scene = self._scene_with_three()  # A/B/C at -3 / 0 / +3 along X
        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "all.{frame:03d}.png"),
                n_frames       = 4,
                resolution     = (120, 60),
                default_light  = True,
            )
            # over the frames, the hit columns span most of the width
            cols: set = set()
            for path in files:
                img  = np.asarray(Image.open(path))
                mask = np.any(img != 0, axis=-1)
                for c in np.where(mask.any(axis=0))[0]:
                    cols.add(int(c))
            self.assertGreater(len(cols), 0, "no hit pixels in any frame")
            span = max(cols) - min(cols)
            self.assertGreater(span, 70, f"translated meshes aren't framed correctly (span={span} of 120)")

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_fit_static_uses_existing_camera_unchanged(self):
        """fit='static' does not autofit: the scene's camera is the orbit
        base as it is, and its node is left as it was"""
        scene = self._scene_with_three()
        cam_node, camera = _looking((0.0, 0.0, 10.0))
        wm_before = cam_node.world_matrix.copy()
        scene.add(cam_node, camera)

        with tempfile.TemporaryDirectory() as tmp:
            scene.turntable(
                output_pattern = os.path.join(tmp, "static.{frame:03d}.png"),
                n_frames       = 2,
                resolution     = (32, 24),
                fit            = "static",
                default_light  = False,
            )

        np.testing.assert_array_equal(cam_node.world_matrix, wm_before)

    def test_fit_invalid_value_raises(self):
        scene = self._scene_with_three()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                scene.turntable(
                    output_pattern = os.path.join(tmp, "x.{frame:03d}.png"),
                    n_frames       = 2,
                    resolution     = (32, 24),
                    fit            = "bogus",
                    default_light  = False,
                )

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_no_off_center_clipping_on_translated_scene(self):
        """with three cubes along X (an asymmetric silhouette per angle), the
        bounding-box framing keeps the silhouette inside the frame at every
        sampled angle: no pixel touches an edge"""
        scene = self._scene_with_three()

        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "no_clip.{frame:03d}.png"),
                n_frames       = 16,
                resolution     = (120, 80),
                default_light  = True,
            )
            W, H = 120, 80
            for path in files:
                img  = np.asarray(Image.open(path))
                mask = np.any(img != 0, axis=-1)
                if not mask.any():
                    self.fail(f"no hits at {os.path.basename(path)}")
                cols = np.where(mask.any(axis=0))[0]
                rows = np.where(mask.any(axis=1))[0]
                name = os.path.basename(path)
                self.assertGreater(int(cols.min()), 0, f"{name}: silhouette clipped at the left edge")
                self.assertLess(int(cols.max()), W - 1, f"{name}: silhouette clipped at the right edge")
                self.assertGreater(int(rows.min()), 0, f"{name}: silhouette clipped at the top edge")
                self.assertLess(int(rows.max()), H - 1, f"{name}: silhouette clipped at the bottom edge")

    def test_camera_orbits_around_centroid_not_world_origin(self):
        """the camera path is a circle around the scene's centroid in the
        rotation plane, not around the world origin: an off-centre scene
        (one cube at x = 10) puts the centroid far from the origin, and
        the per-frame camera positions are equidistant from the centroid"""
        from cgmath.render.camera import _default_camera_for_points

        far = _node("far", translate=[10.0, 0.0, 0.0])
        mesh, uv = _make_textured_cube()
        scene = _scene(far, _under(mesh, far), uv)

        # what the turntable computes
        all_pts  = scene.union_points()
        centroid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
        np.testing.assert_allclose(centroid, [10.0, 0.0, 0.0], atol=1e-9)

        base_cam_col = _default_camera_for_points(all_pts)
        R_cam        = base_cam_col[:3, :3]

        # the bounding-box framing, over 8 angles
        pts_centered = all_pts - centroid
        n_samples    = 8
        bbx_min      = np.full(3, np.inf)
        bbx_max      = np.full(3, -np.inf)
        for i in range(n_samples):
            angle     = 360.0 * i / n_samples
            R_world_3 = _rotation_matrix_col("y", angle)[:3, :3]
            rotated   = pts_centered @ R_world_3.T @ R_cam
            bbx_min   = np.minimum(bbx_min, rotated.min(axis=0))
            bbx_max   = np.maximum(bbx_max, rotated.max(axis=0))
        half_aov_v = 0.5 * np.deg2rad(35.0)
        half_aov_h = np.arctan(np.tan(half_aov_v) * (160 / 120))
        half_w     = 0.5 * (bbx_max[0] - bbx_min[0])
        half_h     = 0.5 * (bbx_max[1] - bbx_min[1])
        cx         = 0.5 * (bbx_min[0] + bbx_max[0])
        cy         = 0.5 * (bbx_min[1] + bbx_max[1])
        cz = float(bbx_max[2]) + max(
            half_w / np.tan(half_aov_h),
            half_h / np.tan(half_aov_v),
        )
        cam_world_pos_chosen = centroid + R_cam @ np.array([cx, cy, cz])

        # 8 per-frame camera positions, on a circle of one radius about the
        # centroid, not about world 0
        radii_to_centroid     = []
        radii_to_world_origin = []
        for fi in range(8):
            angle                 = 360.0 * fi / 8
            R_full                = _rotation_matrix_col("y", angle)
            T_to_origin           = np.eye(4)
            T_to_origin[:3, 3]    = -centroid
            T_back                = np.eye(4)
            T_back[:3, 3]         = centroid
            base_with_pos         = np.eye(4)
            base_with_pos[:3, :3] = R_cam
            base_with_pos[:3, 3]  = cam_world_pos_chosen
            frame_cam             = T_back @ R_full @ T_to_origin @ base_with_pos
            eye                   = frame_cam[:3, 3]
            radii_to_centroid.append(float(np.linalg.norm(eye - centroid)))
            radii_to_world_origin.append(float(np.linalg.norm(eye)))

        rs = np.asarray(radii_to_centroid)
        self.assertLess(float(rs.max() - rs.min()), 1e-6, f"camera not equidistant from centroid: r = {rs}")
        rs_origin = np.asarray(radii_to_world_origin)
        self.assertGreater(
            float(rs_origin.max() - rs_origin.min()),
            1.0,
            f"camera unexpectedly equidistant from world origin too (centroid is at (10, 0, 0)): r = {rs_origin}",
        )

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_silhouette_centered_in_frame(self):
        """at every sampled angle the silhouette's midpoint is near the
        image centre: the bounding-box framing centres every angle on the
        optical axis"""
        scene = self._scene_with_three()

        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "centred.{frame:03d}.png"),
                n_frames       = 8,
                resolution     = (200, 120),
                default_light  = True,
            )
            W, H = 200, 120
            cx_img, cy_img = (W - 1) / 2, (H - 1) / 2
            for path in files:
                img  = np.asarray(Image.open(path))
                mask = np.any(img != 0, axis=-1)
                ys, xs = np.where(mask)
                if xs.size == 0:
                    self.fail(f"no hits at {os.path.basename(path)}")
                mid_x = 0.5 * (xs.min() + xs.max())
                mid_y = 0.5 * (ys.min() + ys.max())
                # up to ~10% off centre: the scene is not symmetric in Y
                self.assertLess(abs(mid_x - cx_img), 0.10 * W, f"{os.path.basename(path)}: X centre {mid_x:.1f}")
                self.assertLess(abs(mid_y - cy_img), 0.20 * H, f"{os.path.basename(path)}: Y centre {mid_y:.1f}")

    def test_no_shown_mesh_raises(self):
        scene = Scene("empty")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no shown mesh"):
                scene.turntable(
                    output_pattern = os.path.join(tmp, "x.{frame:03d}.png"),
                    n_frames       = 2,
                    resolution     = (32, 24),
                    default_light  = False,
                )

    def test_rotation_matrix_y_round_trip(self):
        """the Y rotation follows the right-hand rule: (1, 0, 0) turned 90
        degrees is (0, 0, -1)... as a column-major matrix applied to a column,
        (1, 0, 0) goes to (0, 0, 1)"""
        R       = _rotation_matrix_col("y", 90.0)
        rotated = R @ np.array([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotated[:3], [0.0, 0.0, 1.0], atol=1e-9)
        with self.assertRaises(ValueError):
            _rotation_matrix_col("w", 1.0)


class TestDefaultRig(unittest.TestCase):
    """the 3-point rig a scene without lights is lit by, and render()'s
    one-call conveniences"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_default_lighting_rig_three_lights(self):
        """key / fill / rim at full, 40% and 30%"""
        rig = _default_lighting_rig_for(self.mesh.points)
        self.assertEqual(len(rig), 3)
        self.assertEqual([light["kind"] for light in rig], ["point"] * 3)
        self.assertAlmostEqual(rig[1]["intensity"] / rig[0]["intensity"], 0.4, places=5)
        self.assertAlmostEqual(rig[2]["intensity"] / rig[0]["intensity"], 0.3, places=5)
        # the key in the +X +Y +Z octant of the centroid (the origin)
        self.assertGreater(rig[0]["position"][0], 0)
        self.assertGreater(rig[0]["position"][1], 0)
        self.assertGreater(rig[0]["position"][2], 0)
        # the rim behind, in -Z
        self.assertLess(rig[2]["position"][2], 0)
        self.assertTrue(all(light["falloff"] for light in rig))

    def test_default_lighting_rig_scales_with_extent(self):
        """a 100x bigger scene gets a 100x further rig"""
        small     = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=float)
        rig_small = _default_lighting_rig_for(small)
        rig_big   = _default_lighting_rig_for(small * 100.0)
        d_small   = float(np.linalg.norm(rig_small[0]["position"]))
        d_big     = float(np.linalg.norm(rig_big[0]["position"]))
        self.assertAlmostEqual(d_big / d_small, 100.0, places=4)

    def test_default_lighting_rig_empty_points(self):
        """no points: one fallback light, not pitch black"""
        rig = _default_lighting_rig_for(np.empty((0, 3)))
        self.assertEqual(len(rig), 1)
        self.assertEqual(rig[0]["intensity"], 1.0)

    def test_the_rig_is_passed_to_the_raytracer_and_not_added_to_the_scene(self):
        import cgmath.render.raytracer as raytracer

        scene = _scene(self.mesh, resolution=(16, 12))
        seen  = {}
        real  = raytracer.render

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        with patch.object(raytracer, "render", spy):
            scene.render(default_light=True)
        self.assertEqual(len(seen["point_light"]), 3)
        self.assertIs(seen["default_light"], False)
        self.assertIsNone(scene.lights)

        # a scene with a light of its own gets no rig
        scene.add(LightData(name="key"))
        seen.clear()
        with patch.object(raytracer, "render", spy):
            scene.render(default_light=True)
        self.assertNotIn("point_light", seen)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_render_saves_to_output(self):
        scene = _scene(self.mesh, self.uv)
        with tempfile.TemporaryDirectory() as tmp:
            path  = os.path.join(tmp, "preview.png")
            frame = scene.render(output=path, resolution=(64, 64))
            self.assertTrue(os.path.exists(path))
            self.assertEqual(frame.shape, (64, 64, 4))

    def test_render_default_lighting_is_not_black(self):
        scene = _scene(self.mesh, self.uv, _material(self.mesh, color=(0.7, 0.7, 0.7)), ambient=0.0)
        frame = scene.render(resolution=(64, 64))
        self.assertGreater(int(frame.array.max()), 30, "default rig produced a near-black preview")

    def test_render_forwards_kwargs(self):
        scene = _scene(self.mesh, self.uv)
        frame = scene.render(
            resolution    = (48, 48),
            background    = (1.0, 0.0, 0.0),  # red background
            default_light = False,
        )
        self.assertEqual(frame.shape, (48, 48, 4))
        corner = frame.array[0, 0]
        self.assertGreater(int(corner[0]), 200, "expected a red background")
        self.assertLess(int(corner[1]), 50)
        self.assertLess(int(corner[2]), 50)

    def test_render_does_not_mutate_the_nodes(self):
        node  = _node("cube", translate=[5.0, 0.0, 0.0])
        scene = _scene(node, _under(self.mesh, node), self.uv)
        held  = node.world_matrix.copy()
        scene.render(resolution=(32, 32))
        np.testing.assert_array_equal(node.world_matrix, held)
        self.assertEqual(scene.hierarchy.name, ["cube"])


class TestFrameWireframe(unittest.TestCase):
    """Frame.wireframe() post-process overlay."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()
        self.scene = _scene(self.mesh, self.uv)
        # a small base frame to overlay onto
        self.frame = self.scene.render(resolution=(96, 96), default_light=True)

    def test_render_attaches_camera_matrix_and_aov(self):
        """a frame from render() carries camera_matrix and angle_of_view so
        wireframe() can project edges without being passed them"""
        self.assertIsNotNone(self.frame.camera_matrix)
        self.assertEqual(self.frame.camera_matrix.shape, (4, 4))
        self.assertIsNotNone(self.frame.angle_of_view)
        self.assertGreater(float(self.frame.angle_of_view), 0.0)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_modifies_pixels(self):
        wired = self.frame.wireframe(self.scene)
        self.assertEqual(wired.shape, self.frame.shape)
        diff = int(np.any(self.frame.array != wired.array, axis=-1).sum())
        self.assertGreater(diff, 0, "wireframe drew zero pixels")

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_color_kwarg(self):
        red = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=2)
        red_mask = (
            (red.array[..., 0] > 200)
            & (red.array[..., 1] < 50)
            & (red.array[..., 2] < 50)
        )
        self.assertGreater(int(red_mask.sum()), 20)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_width_increases_pixel_count(self):
        def red_count(f):
            a    = f.array
            mask = (a[..., 0] > 200) & (a[..., 1] < 50) & (a[..., 2] < 50)
            return int(mask.sum())

        thin  = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1)
        thick = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=3)
        self.assertGreater(red_count(thick), red_count(thin))

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_accepts_a_scene_and_a_mesh(self):
        for source in (self.scene, self.mesh):
            with self.subTest(type(source).__name__):
                wired = self.frame.wireframe(source)
                self.assertEqual(wired.shape, self.frame.shape)

    def test_wireframe_bare_frame_raises(self):
        """a Frame without camera_matrix cannot project edges"""
        from cgmath.render import Frame

        bare = Frame(array=np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            bare.wireframe(self.scene)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_bad_source_raises(self):
        with self.assertRaises(TypeError):
            self.frame.wireframe(42)
        with self.assertRaises(TypeError):
            self.frame.wireframe("not a scene")

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_preserves_camera_metadata(self):
        wired = self.frame.wireframe(self.scene)
        np.testing.assert_array_equal(wired.camera_matrix, self.frame.camera_matrix)
        self.assertEqual(wired.angle_of_view, self.frame.angle_of_view)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_chainable(self):
        first  = self.frame.wireframe(self.scene, color=(0, 0, 0, 200), width=1)
        second = first.wireframe(self.mesh, color=(0, 255, 0, 255), width=1)
        green_mask = (
            (second.array[..., 1] > 200)
            & (second.array[..., 0] < 50)
            & (second.array[..., 2] < 50)
        )
        self.assertGreater(int(green_mask.sum()), 0)

    def test_wireframe_unique_vs_per_face_edges(self):
        """unique_edges=True yields fewer edges than False on a triangulated
        cube, where every quad has a shared diagonal"""
        from cgmath.render.frame import _iter_wireframe_primitives

        unique_pairs   = next(_iter_wireframe_primitives(self.mesh, unique=True))["edge_pairs"]
        per_face_pairs = next(_iter_wireframe_primitives(self.mesh, unique=False))["edge_pairs"]
        # 6 faces x 2 triangles x 3 edges = 36 per-face edges, 18 unique
        self.assertLess(unique_pairs.shape[0], per_face_pairs.shape[0])

    def test_wireframe_primitives_place_a_scenes_meshes(self):
        """a scene's primitives are its meshes in world space"""
        from cgmath.render.frame import _iter_wireframe_primitives

        node = _node("far", translate=[10.0, 0.0, 0.0])
        mesh, uv = _make_textured_cube()
        scene = _scene(node, _under(mesh, node), uv)
        [prim] = list(_iter_wireframe_primitives(scene, unique=True))
        np.testing.assert_allclose(prim["world_pts"], mesh.points + [10.0, 0.0, 0.0])

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_cull_backfaces_default_on(self):
        culled = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1)
        xray   = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1, cull_backfaces=False)
        self.assertFalse(np.array_equal(culled.array, xray.array), "cull_backfaces=True should differ from X-ray")

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_cull_reduces_drawn_pixels(self):
        def red_count(f):
            a    = f.array
            mask = (a[..., 0] > 200) & (a[..., 1] < 50) & (a[..., 2] < 50)
            return int(mask.sum())

        culled = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1)
        xray   = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1, cull_backfaces=False)
        c, x = red_count(culled), red_count(xray)
        self.assertGreater(x, 0)
        self.assertLess(c, x, f"cull should remove pixels (culled={c}, xray={x})")

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_cull_works_for_a_placed_mesh(self):
        """the cull uses world-space face normals, so a mesh on a moved and
        turned node still culls"""
        node = _node("cube", translate=[5.0, 0.0, 0.0], rotate=[0.0, 30.0, 0.0])
        mesh, uv = _make_textured_cube()
        scene = _scene(node, _under(mesh, node), uv)
        frame = scene.render(resolution=(96, 96), default_light=True)

        def red_count(f):
            a    = f.array
            mask = (a[..., 0] > 200) & (a[..., 1] < 50) & (a[..., 2] < 50)
            return int(mask.sum())

        culled = frame.wireframe(scene, color=(255, 0, 0, 255), width=1)
        xray   = frame.wireframe(scene, color=(255, 0, 0, 255), width=1, cull_backfaces=False)
        self.assertGreater(red_count(culled), 0, "culled pass drew nothing")
        self.assertLess(red_count(culled), red_count(xray), "culling should reduce pixel count")

    def test_wireframe_cull_data_attached_to_primitives(self):
        from cgmath.render.frame import _iter_wireframe_primitives

        prim = next(_iter_wireframe_primitives(self.mesh, unique=True, with_cull_data=True))
        for key in ("world_pts", "edge_pairs", "face_normals", "face_centres", "e2f"):
            self.assertIn(key, prim)
            self.assertIsNotNone(prim[key], f"{key!r} should be populated")

        prim_no_cull = next(_iter_wireframe_primitives(self.mesh, unique=True, with_cull_data=False))
        self.assertIsNone(prim_no_cull.get("face_normals"))

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_wireframe_cull_faceless_orphan_edges_keep_drawn(self):
        out = self.frame.wireframe(self.mesh, cull_backfaces=True)
        self.assertEqual(out.shape, self.frame.shape)


class TestToImageCache(unittest.TestCase):
    """Scene.to_image() and the kept frame"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_frame_and_buffer_start_none(self):
        scene = _scene(self.mesh)
        self.assertIsNone(scene.frame)
        self.assertIsNone(scene.buffer)

    def test_render_keeps_the_frame(self):
        scene = _scene(self.mesh)
        f     = scene.render(resolution=(48, 48), default_light=True)
        self.assertIs(scene.frame, f)
        self.assertEqual(scene.buffer.shape, (48, 48, 4))
        np.testing.assert_array_equal(scene.buffer, scene.frame.array)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_to_image_renders_at_the_resolution_when_nothing_is_kept(self):
        scene = _scene(self.mesh)
        img   = scene.to_image(resolution=(40, 40))
        self.assertIsNotNone(scene.frame)
        self.assertEqual(scene.buffer.shape, (40, 40, 4))
        self.assertEqual(img.size, (40, 40))

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_to_image_default_resolution_is_500(self):
        scene = _scene(self.mesh)
        img   = scene.to_image()
        self.assertEqual(scene.buffer.shape, (500, 500, 4))
        self.assertEqual(img.size, (500, 500))

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_to_image_returns_a_pil_image_of_the_kept_pixels(self):
        from PIL.Image import Image as _PILImage

        scene = _scene(self.mesh)
        f     = scene.render(resolution=(32, 32))
        img   = scene.to_image()
        self.assertIsInstance(img, _PILImage)
        self.assertIs(scene.frame, f)  # no second render
        self.assertEqual(img.size, (32, 32))
        np.testing.assert_array_equal(np.asarray(img), scene.buffer)


class TestSceneRenderConfig(unittest.TestCase):
    """the settings kept on a Scene and their merge into render / turntable"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_default_render_config(self):
        """resolution has a concrete default, (500, 500); the other render
        settings are None, the renderer's own defaults"""
        scene = Scene("s")
        np.testing.assert_array_equal(scene.resolution, (500, 500))
        for k in _RENDER_CONFIG_KEYS:
            if k != "resolution":
                self.assertIsNone(getattr(scene, k), f"{k} should default to None")
        self.assertEqual(
            [getattr(scene, k) if not isinstance(getattr(scene, k), np.ndarray) else tuple(getattr(scene, k)) for k in _SHADING_KEYS],
            [(0.7, 0.7, 0.7), "bilinear", 0.8, True, True, False, (0, 0, 0), 1],
        )

    def test_init_kwargs_set_render_config(self):
        scene = Scene(
            "s",
            samples_per_pixel = 4,
            resolution        = (640, 480),
            background        = (0.1, 0.2, 0.3),
            autofit           = False,
        )
        self.assertEqual(scene.samples_per_pixel, 4)
        np.testing.assert_array_equal(scene.resolution, (640, 480))
        np.testing.assert_array_equal(scene.background, (0.1, 0.2, 0.3, 1.0))  # RGB taken opaque
        self.assertFalse(scene.autofit)

    def test_configure_chainable_and_skips_none(self):
        scene = Scene("s")
        ret   = scene.configure(samples_per_pixel=4, resolution=None, ambient=0.2)
        self.assertIs(ret, scene)
        self.assertEqual(scene.samples_per_pixel, 4)
        self.assertEqual(scene.ambient, 0.2)
        np.testing.assert_array_equal(scene.resolution, (500, 500), err_msg="None values are skipped")

    def test_configure_unknown_attribute_raises(self):
        scene = Scene("s")
        with self.assertRaises(AttributeError):
            scene.configure(bogus_attr=42)
        with self.assertRaises(AttributeError):
            scene.bogus_attr = 42

    def test_render_uses_scene_attrs_when_no_kwargs(self):
        scene = _scene(self.mesh, resolution=(64, 48))
        f     = scene.render(default_light=True)
        self.assertEqual(f.shape, (48, 64, 4))

    def test_render_kwarg_overrides_scene_attr(self):
        scene = _scene(self.mesh, resolution=(64, 48))
        f     = scene.render(resolution=(40, 30), default_light=True)
        self.assertEqual(f.shape, (30, 40, 4))

    def test_merge_leaves_none_out(self):
        scene = Scene("s", resolution=None)
        self.assertEqual(scene._merge_render_config({}), {})
        self.assertEqual(scene._merge_render_config({"resolution": (8, 8)}), {"resolution": (8, 8)})

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_turntable_uses_scene_attrs(self):
        scene = _scene(self.mesh, resolution=(48, 48))
        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "spin.{frame:03d}.png"),
                n_frames       = 2,
                default_light  = True,
            )
            with Image.open(files[0]) as img:
                self.assertEqual(img.size, (48, 48))


@unittest.skipUnless(_HAS_RASTERIZER, "wireframe bake needs cv2 or scikit-image for UVData.draw_edges")
class TestSceneWireframe(unittest.TestCase):
    """the scene-wide wireframe: edges baked over each material's colour map
    (or its colour) on the mesh's UV set, kept while the inputs stay"""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()
        self.uv.resolution = (32, 32)

    def textured(self, **settings):
        material = _material(self.mesh, np.full((64, 64, 3), 200, dtype=np.uint8))
        return _scene(self.mesh, self.uv, material, **settings), material

    def test_wireframe_defaults(self):
        scene = Scene("s")
        self.assertFalse(scene.wireframe)
        np.testing.assert_array_equal(scene.wireframe_color, (0, 0, 0))
        self.assertEqual(scene.wireframe_thickness, 1)

    def test_wireframe_off_returns_the_plain_colour_map(self):
        scene, material = self.textured()
        tex = scene._texture(material, self.uv)
        self.assertEqual(tex.dtype, np.float32)
        self.assertEqual(tex.shape, (64, 64, 3))
        self.assertEqual(int((tex.sum(axis=-1) < 0.1).sum()), 0)  # all bright grey

    def test_wireframe_on_bakes_dark_texels(self):
        scene, material = self.textured(wireframe=True)
        tex = scene._texture(material, self.uv)
        self.assertGreater(int((tex.sum(axis=-1) < 0.1).sum()), 0, "wireframe bake produced no dark texels")

    def test_the_bake_is_kept_for_the_same_inputs(self):
        scene, material = self.textured(wireframe=True)
        self.assertIs(scene._texture(material, self.uv), scene._texture(material, self.uv))

    def test_a_colour_or_thickness_change_bakes_again(self):
        scene, material = self.textured(wireframe=True)
        first                 = scene._texture(material, self.uv)
        scene.wireframe_color = (255, 0, 0)
        second                = scene._texture(material, self.uv)
        self.assertIsNot(second, first)
        self.assertFalse(np.array_equal(second, first))
        scene.wireframe_thickness = 3
        self.assertIsNot(scene._texture(material, self.uv), second)

    def test_a_new_texture_bakes_again(self):
        scene, material = self.textured(wireframe=True)
        first            = scene._texture(material, self.uv)
        material.texture = np.full((64, 64, 3), 100, dtype=np.uint8)
        second           = scene._texture(material, self.uv)
        self.assertIsNot(second, first)
        self.assertLess(float(second.mean()), float(first.mean()))

    def test_wireframe_off_after_a_bake_returns_the_plain_map(self):
        scene, material = self.textured(wireframe=True)
        scene._texture(material, self.uv)
        scene.wireframe = False
        tex             = scene._texture(material, self.uv)
        self.assertEqual(int((tex.sum(axis=-1) < 0.1).sum()), 0, "should fall back to the plain map")

    def test_a_render_shows_the_wires(self):
        scene, _ = self.textured(wireframe=True, wireframe_thickness=2)
        f    = scene.render(resolution=(96, 96))
        dark = (f.array.sum(axis=-1) < 50).sum()
        self.assertGreater(int(dark), 100, "expected visible wireframe pixels in the render")

    def test_without_a_texture_the_wires_go_over_the_materials_colour(self):
        bakes = []
        for colour in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
            mesh, uv = _make_textured_cube()
            uv.resolution = (32, 32)
            material      = _material(mesh, color=colour)
            scene         = _scene(mesh, uv, material, wireframe=True, wireframe_color=(0, 0, 0))
            baked         = scene._texture(material, uv)
            self.assertEqual((baked.dtype, baked.ndim, baked.shape[2]), (np.float32, 3, 3))
            self.assertGreaterEqual(float(baked.min()), 0.0)
            self.assertLessEqual(float(baked.max()), 1.0)
            bakes.append(baked)
        dark, bright = bakes
        # black wires over black, and over white: the background shows
        self.assertLess(float(dark.mean()), 0.05)
        self.assertGreater(float(bright.mean()), 0.7)

    def test_a_mesh_without_a_material_bakes_over_the_scenes_base_colour(self):
        scene = _scene(self.mesh, self.uv, wireframe=True, base_color=(1.0, 1.0, 1.0))
        baked = scene._texture(None, self.uv)
        self.assertIsNotNone(baked)
        self.assertGreater(float(baked.mean()), 0.7)
        self.assertIsNotNone(scene._render_inputs().entries[0]["tex"])

    def test_without_a_uv_set_or_wires_there_is_no_map(self):
        material = _material(self.mesh)
        scene    = _scene(self.mesh, material, wireframe=True)
        self.assertIsNone(scene._texture(material, None))  # no UV set: nothing to bake on
        scene.wireframe = False
        self.assertIsNone(scene._texture(material, self.uv))  # wireframe is opt-in


class TestSceneCacheInvalidation(unittest.TestCase):
    """Scene.frame is dropped when anything the render depends on changes
    since the frame was kept: a setting, a shape's field, a node's
    transform or visibility. A sentinel stands in for a rendered frame."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def _populated_scene(self):
        cube = _node("cube")
        cam  = _node("cam", translate=[0.0, 0.0, 5.0])
        key  = _node("key", translate=[2.0, 2.0, 2.0])
        scene = _scene(
            cube, cam, key,
            _under(self.mesh, cube), self.uv,
            _material(self.mesh),
            _camera(parent_uuid=cam.uuid),
            LightData(name="key", kind="point", intensity=100.0, decay_rate=2, parent_uuid=key.uuid),
            MorphData("puff", [[0.0, 0.0, 1.0]], [0], mesh_uuid=self.mesh.uuid, weight=0.2),
        )
        return scene

    def _seed_frame(self, scene):
        sentinel    = object()
        scene.frame = sentinel
        return sentinel

    def test_setter_snapshots_signature(self):
        scene = self._populated_scene()
        self.assertIsNone(scene._render_signature)
        self._seed_frame(scene)
        self.assertIsNotNone(scene._render_signature)
        scene.frame = None
        self.assertIsNone(scene._render_signature)

    def test_no_change_keeps_cache(self):
        scene    = self._populated_scene()
        sentinel = self._seed_frame(scene)
        self.assertIs(scene.frame, sentinel)
        self.assertIs(scene.frame, sentinel)
        hash(scene._signature())

    def test_a_change_drops_the_frame(self):
        changes = {
            "the mesh node's rotate": lambda s: setattr(s.hierarchy["cube"], "rotate", [0.0, 45.0, 0.0]),
            "the mesh node's translate": lambda s: setattr(s.hierarchy["cube"], "translate", [10.0, 0.0, 0.0]),
            "the mesh node's scale": lambda s: setattr(s.hierarchy["cube"], "scale", [2.0, 2.0, 2.0]),
            "the mesh node's visibility": lambda s: setattr(s.hierarchy["cube"], "visibility", False),
            "the camera node's translate": lambda s: setattr(s.hierarchy["cam"], "translate", [5.0, 5.0, 5.0]),
            "the light node's translate": lambda s: setattr(s.hierarchy["key"], "translate", [10.0, 0.0, 0.0]),
            "the camera's focal length": lambda s: setattr(s.cameras[0], "focal_length", 50.0),
            "the camera orthographic": lambda s: setattr(s.cameras[0], "orthographic", True),
            "the camera's width": lambda s: setattr(s.cameras[0], "orthographic_width", 5.0),
            "the light's intensity": lambda s: setattr(s.lights[0], "intensity", 200.0),
            "the light's colour": lambda s: setattr(s.lights[0], "color", (1.0, 0.0, 0.0)),
            "the light's kind": lambda s: setattr(s.lights[0], "kind", "directional"),
            "the light's decay": lambda s: setattr(s.lights[0], "decay_rate", 0),
            "the material's colour": lambda s: setattr(s.materials[0], "color", (1.0, 0.0, 0.0)),
            "the material's diffuse": lambda s: setattr(s.materials[0], "diffuse", 0.5),
            "the material's texture": lambda s: setattr(s.materials[0], "texture", np.zeros((4, 4, 3), dtype=np.float32)),
            "the material's faces": lambda s: setattr(s.materials[0], "faces", [0, 1]),
            "the material's wrap": lambda s: setattr(s.materials[0], "wrap_u", "clamp"),
            "the morph's weight": lambda s: setattr(s.morphs[0], "weight", 0.7),
            "the mesh's points": lambda s: setattr(s.meshes[0], "points", s.meshes[0].points * 2.0),
            "a mesh added": lambda s: s.add(_make_textured_cube()[0]),
            "a light added": lambda s: s.add(LightData(name="fill", kind="point", intensity=50.0)),
            "a material added": lambda s: s.add(MaterialData(mesh_uuid=s.meshes[0].uuid, faces=[0])),
            "the default camera": lambda s: setattr(s, "default_camera", "other"),
            "resolution": lambda s: setattr(s, "resolution", (640, 480)),
            "samples_per_pixel": lambda s: setattr(s, "samples_per_pixel", 4),
            "background": lambda s: setattr(s, "background", (0.5, 0.5, 0.5)),
            "angle_of_view": lambda s: setattr(s, "angle_of_view", 60.0),
            "base_color": lambda s: setattr(s, "base_color", (1.0, 0.0, 0.0)),
            "sample_method": lambda s: setattr(s, "sample_method", "bezier"),
            "ambient": lambda s: setattr(s, "ambient", 0.1),
            "twosided": lambda s: setattr(s, "twosided", False),
            "wireframe": lambda s: setattr(s, "wireframe", True),
            "wireframe_color": lambda s: setattr(s, "wireframe_color", (255, 255, 255)),
            "wireframe_thickness": lambda s: setattr(s, "wireframe_thickness", 3),
        }
        for label, change in changes.items():
            with self.subTest(label):
                scene = self._populated_scene()
                self._seed_frame(scene)
                change(scene)
                self.assertIsNone(scene.frame)

    def test_the_same_value_set_again_keeps_the_frame(self):
        scene                             = self._populated_scene()
        sentinel                          = self._seed_frame(scene)
        scene.base_color                  = (0.7, 0.7, 0.7)
        scene.resolution                  = [500, 500]
        scene.sample_method               = "bilinear"
        scene.hierarchy["cube"].translate = [0.0, 0.0, 0.0]
        scene.materials[0].color          = (1.0, 1.0, 1.0)
        self.assertIs(scene.frame, sentinel)

    def test_a_parents_transform_drops_the_frame(self):
        scene  = self._populated_scene()
        parent = _node("parent")
        scene.add(parent)
        scene.hierarchy["cube"].set_parent(parent)
        self._seed_frame(scene)
        parent.translate = [10.0, 0.0, 0.0]
        self.assertIsNone(scene.frame)

    def test_the_frame_rendered_is_part_of_the_signature(self):
        node  = _node("root")
        scene = _scene(node, _under(self.mesh, node), resolution=(8, 8), samples_per_pixel=1)
        clip  = ClipData(scene.hierarchy, frames=2)
        clip.frames["root"].translate = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        scene.clip                    = clip
        first = scene.render(frame=clip.start_frame, default_light=False)
        self.assertIs(scene.frame, first)
        self.assertNotEqual(scene._signature(clip.start_frame), scene._signature(clip.start_frame + 1))


# -- GLB texture extraction tests ----------------------------------------------


def _png_bytes(color=(255, 0, 0), size=(2, 2)) -> bytes:
    """Encode a flat-color RGB image as PNG bytes via PIL."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_glb(json_dict: dict, bin_data: bytes = b"") -> bytes:
    """Pack a glTF JSON dict + binary buffer into a valid GLB blob.

    Implements the binary-glTF 2.0 container format: 12-byte header,
    JSON chunk, optional BIN chunk; both chunk payloads are padded to
    4-byte alignment.
    """
    json_bytes = json.dumps(json_dict).encode("utf-8")
    json_pad   = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad

    bin_pad   = (4 - (len(bin_data) % 4)) % 4
    bin_bytes = bin_data + b"\x00" * bin_pad

    total_length = 12 + 8 + len(json_bytes)
    if bin_data:
        total_length += 8 + len(bin_bytes)

    header     = struct.pack("<III", 0x46546C67, 2, total_length)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    if not bin_data:
        return header + json_chunk
    bin_chunk = struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    return header + json_chunk + bin_chunk


def _write_temp_glb(content: bytes) -> str:
    """Write GLB bytes to a temp file and return its path.  Caller cleans up."""
    fd, path = tempfile.mkstemp(suffix=".glb")
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path


def _minimal_glb_with_color(png: bytes) -> bytes:
    """Build a GLB whose first material exposes a baseColorTexture pointing
    at ``png`` embedded in the BIN chunk."""
    json_dict = {
        "asset": {"version": "2.0"},
        "materials": [
            {
                "name": "test_mat",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }
        ],
        "textures": [{"source": 0}],
        "images":   [{"bufferView": 0, "mimeType": "image/png"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(png)},
        ],
        "buffers": [{"byteLength": len(png)}],
    }
    return _build_glb(json_dict, bin_data=png)


class TestGlbTextureExtraction(unittest.TestCase):
    """``_extract_glb_textures`` and the materials Scene.load_glb makes of it"""

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_extract_glb_textures_finds_base_color(self):
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes((255, 0, 0))))
        try:
            textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)

        self.assertIn("base_color", textures)
        np.testing.assert_array_equal(textures["base_color"][:, :, 0], np.full((2, 2), 255, dtype=np.uint8))
        np.testing.assert_array_equal(textures["base_color"][:, :, 1], np.zeros((2, 2), dtype=np.uint8))

    def test_extract_glb_textures_invalid_magic_returns_empty(self):
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(b"not a glb at all, definitely not 12 bytes")
        try:
            self.assertEqual(_extract_glb_textures(path), {})
        finally:
            os.unlink(path)

    def test_extract_glb_textures_short_file_returns_empty(self):
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(b"tiny")
        try:
            self.assertEqual(_extract_glb_textures(path), {})
        finally:
            os.unlink(path)

    def test_extract_glb_textures_no_materials_returns_empty(self):
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(_build_glb({"asset": {"version": "2.0"}}))
        try:
            self.assertEqual(_extract_glb_textures(path), {})
        finally:
            os.unlink(path)

    def test_extract_glb_textures_material_index_out_of_range(self):
        from cgmath.render.scene import _extract_glb_textures

        json_dict = {
            "asset":     {"version": "2.0"},
            "materials": [{"name": "only_one"}],
        }
        path = _write_temp_glb(_build_glb(json_dict))
        try:
            self.assertEqual(_extract_glb_textures(path, material_index=5), {})
        finally:
            os.unlink(path)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_extract_glb_textures_pil_unavailable(self):
        from cgmath.render import scene as scene_mod
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes()))
        try:
            with patch.object(scene_mod, "Image", None):
                textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)
        # without PIL nothing decodes: base_color is left out
        self.assertEqual(textures, {})

    def test_extract_glb_textures_external_uri_skipped(self):
        from cgmath.render.scene import _extract_glb_textures

        json_dict = {
            "asset":     {"version": "2.0"},
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "textures":  [{"source": 0}],
            "images":    [{"uri": "external_image.png"}],
        }
        path = _write_temp_glb(_build_glb(json_dict))
        try:
            self.assertEqual(_extract_glb_textures(path), {})
        finally:
            os.unlink(path)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_extract_glb_textures_data_uri(self):
        from cgmath.render.scene import _extract_glb_textures

        png = _png_bytes(color=(0, 255, 0))
        b64 = base64.b64encode(png).decode("ascii")
        json_dict = {
            "asset":     {"version": "2.0"},
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "textures":  [{"source": 0}],
            "images":    [{"uri": f"data:image/png;base64,{b64}"}],
        }
        path = _write_temp_glb(_build_glb(json_dict))
        try:
            textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)
        self.assertIn("base_color", textures)
        np.testing.assert_array_equal(textures["base_color"][:, :, 1], np.full((2, 2), 255, dtype=np.uint8))

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_extract_glb_textures_extracts_all_pbr_slots(self):
        from cgmath.render.scene import _extract_glb_textures

        red      = _png_bytes((255, 0, 0))
        green    = _png_bytes((0, 255, 0))
        blue     = _png_bytes((0, 0, 255))
        bin_blob = red + green + blue
        json_dict = {
            "asset": {"version": "2.0"},
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorTexture":         {"index": 0},
                        "metallicRoughnessTexture": {"index": 1},
                    },
                    "normalTexture": {"index": 2},
                }
            ],
            "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
            "images": [
                {"bufferView": 0, "mimeType": "image/png"},
                {"bufferView": 1, "mimeType": "image/png"},
                {"bufferView": 2, "mimeType": "image/png"},
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(red)},
                {"buffer": 0, "byteOffset": len(red), "byteLength": len(green)},
                {"buffer": 0, "byteOffset": len(red) + len(green), "byteLength": len(blue)},
            ],
            "buffers": [{"byteLength": len(bin_blob)}],
        }
        path = _write_temp_glb(_build_glb(json_dict, bin_data=bin_blob))
        try:
            textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)
        self.assertEqual(set(textures.keys()), {"base_color", "metallic_roughness", "normal"})

    def test_extract_glb_textures_bad_buffer_view_index_skipped(self):
        from cgmath.render.scene import _extract_glb_textures

        json_dict = {
            "asset":       {"version": "2.0"},
            "materials":   [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "textures":    [{"source": 0}],
            "images":      [{"bufferView": 99}],  # out of range
            "bufferViews": [],
        }
        path = _write_temp_glb(_build_glb(json_dict, bin_data=b"some bytes"))
        try:
            self.assertEqual(_extract_glb_textures(path), {})
        finally:
            os.unlink(path)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_scene_load_glb_makes_a_material_of_the_base_colour_map(self):
        from cgmath.render import scene as scene_mod

        mesh, uv = _make_textured_cube()
        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes((10, 20, 30))))
        try:
            with patch.object(scene_mod, "_load_glb", return_value=[(mesh, [uv])]):
                scene = Scene.load_glb(path, load_skin=False)
        finally:
            os.unlink(path)

        self.assertEqual(len(scene.materials), 1)
        material = scene.materials[0]
        self.assertEqual(material.mesh_uuid, scene.meshes[0].uuid)
        self.assertEqual(material.uv_set, "map1")
        self.assertIsInstance(material.texture, np.ndarray)
        np.testing.assert_array_equal(material.texture[0, 0], [10, 20, 30])
        self.assertEqual(scene.uvs[0].mesh_uuid, scene.meshes[0].uuid)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_scene_load_glb_extract_texture_false(self):
        from cgmath.render import scene as scene_mod

        mesh, uv = _make_textured_cube()
        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes()))
        try:
            with patch.object(scene_mod, "_load_glb", return_value=[(mesh, [uv])]):
                scene = Scene.load_glb(path, extract_texture=False, load_skin=False)
        finally:
            os.unlink(path)

        self.assertIsNone(scene.materials)
        self.assertEqual(len(scene.meshes), 1)


class TestSceneObj(unittest.TestCase):
    """Scene.load_obj: every mesh of the file with its UV sets, on no node"""

    def _save_cube_obj(self, path: str, name: str = "cube") -> tuple:
        from cgmath.geometry.mesh import save_obj, UVList

        mesh, uv = _make_textured_cube()
        mesh.name = name
        save_obj(path, [(mesh, UVList([uv]))])
        return mesh, uv

    def _save_two_cube_obj(self, path: str) -> None:
        from cgmath.geometry.mesh import save_obj, UVList

        mesh_a, uv_a = _make_textured_cube()
        mesh_a.name = "cubeA"
        mesh_b, uv_b = _make_textured_cube()
        mesh_b.name = "cubeB"
        save_obj(path, [(mesh_a, UVList([uv_a])), (mesh_b, UVList([uv_b]))])

    def test_scene_load_obj_single_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "cube.obj")
            self._save_cube_obj(f)
            scene = Scene.load_obj(f)
        self.assertIsInstance(scene, Scene)
        self.assertEqual(scene.name,           "cube")
        self.assertEqual(len(scene.meshes),    1)
        self.assertEqual(scene.meshes[0].name, "cubeShape")  # save_obj strips Shape, load_obj puts it back
        self.assertIsNone(scene.meshes[0].parent_uuid)
        self.assertEqual(len(scene.uvs), 1)
        self.assertEqual(scene.uvs[0].mesh_uuid, scene.meshes[0].uuid)
        self.assertIsNone(scene.hierarchy)

    def test_scene_load_obj_multi_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "two.obj")
            self._save_two_cube_obj(f)
            scene = Scene.load_obj(f, name="my_scene")
        self.assertEqual(scene.name, "my_scene")
        self.assertEqual(sorted(m.name for m in scene.meshes), ["cubeAShape", "cubeBShape"])
        self.assertEqual([uv.mesh_uuid for uv in scene.uvs], [m.uuid for m in scene.meshes])

    def test_scene_load_obj_no_uv(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "noUV.obj")
            mesh, _ = _make_textured_cube()
            mesh.name = "cube"
            mesh.save_obj(f)
            scene = Scene.load_obj(f)
        self.assertEqual(len(scene.meshes), 1)
        self.assertIsNone(scene.uvs)


# ============================================================================
# Tests for ffmpeg / gifski subprocess helpers
# ============================================================================


class TestFrameSubprocessHelpers(unittest.TestCase):
    """Tests for :func:`cgmath.render.frame._spawn_or_raise` and
    :func:`_run_or_raise` -- thin wrappers that convert the low-level
    ``FileNotFoundError`` raised by ``subprocess`` when the binary at
    ``cmd[0]`` is missing from PATH into a friendly :class:`RuntimeError`.
    """

    def test_spawn_or_raise_returns_popen_when_binary_found(self):
        from cgmath.render.frame import _spawn_or_raise

        fake_proc = MagicMock(name="popen")
        with patch("cgmath.render.frame.subprocess.Popen", return_value=fake_proc) as popen:
            result = _spawn_or_raise(["echo", "hi"], stdout=subprocess.PIPE)
        self.assertIs(result, fake_proc)
        popen.assert_called_once_with(["echo", "hi"], stdout=subprocess.PIPE)

    def test_spawn_or_raise_translates_file_not_found_to_runtime_error(self):
        from cgmath.render.frame import _spawn_or_raise

        with patch("cgmath.render.frame.subprocess.Popen", side_effect=FileNotFoundError("execvp")):
            with self.assertRaises(RuntimeError) as ctx:
                _spawn_or_raise(["definitely-not-a-binary-xyz"])
        msg = str(ctx.exception)
        self.assertIn("definitely-not-a-binary-xyz", msg)
        self.assertIn("PATH", msg)

    def test_run_or_raise_returns_completed_process(self):
        from cgmath.render.frame import _run_or_raise

        fake_cp = MagicMock(name="completed-process", returncode=0)
        with patch("cgmath.render.frame.subprocess.run", return_value=fake_cp) as run:
            result = _run_or_raise(["echo", "hi"], capture_output=True)
        self.assertIs(result, fake_cp)
        run.assert_called_once_with(["echo", "hi"], capture_output=True)

    def test_run_or_raise_translates_file_not_found_to_runtime_error(self):
        from cgmath.render.frame import _run_or_raise

        with patch("cgmath.render.frame.subprocess.run", side_effect=FileNotFoundError("execvp")):
            with self.assertRaises(RuntimeError) as ctx:
                _run_or_raise(["definitely-not-a-binary-xyz"])
        msg = str(ctx.exception)
        self.assertIn("definitely-not-a-binary-xyz", msg)
        self.assertIn("PATH", msg)


class TestFrameEncoderMissingBinary(unittest.TestCase):
    """Encoder-level integration: when ffmpeg / gifski are absent from
    PATH, the user-facing error is a :class:`RuntimeError` (not a bare
    :class:`FileNotFoundError` from ``subprocess``)."""

    @staticmethod
    def _single_frame():
        return [np.zeros((4, 4, 3), dtype=np.uint8)]

    def test_encode_mp4_raises_runtime_error_when_ffmpeg_missing(self):
        from cgmath.render import frame as frame_mod

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.mp4")
            with patch("cgmath.render.frame.subprocess.Popen", side_effect=FileNotFoundError("execvp")):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_mp4(self._single_frame(), out)
        self.assertIn("ffmpeg", str(ctx.exception))
        self.assertIn("PATH", str(ctx.exception))

    def test_encode_avi_raises_runtime_error_when_ffmpeg_missing(self):
        from cgmath.render import frame as frame_mod

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.avi")
            with patch("cgmath.render.frame.subprocess.Popen", side_effect=FileNotFoundError("execvp")):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_avi(self._single_frame(), out)
        self.assertIn("ffmpeg", str(ctx.exception))

    def test_encode_gif_use_gifski_true_raises_when_gifski_missing(self):
        from cgmath.render import frame as frame_mod

        if Image is None:
            self.skipTest("PIL is required for encode_gif")

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.gif")
            # use_gifski=True forces the gifski path even when shutil.which
            # would have answered "no". The subprocess call then surfaces
            # the friendly RuntimeError from _run_or_raise.
            with patch("cgmath.render.frame.subprocess.run", side_effect=FileNotFoundError("execvp")):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_gif(self._single_frame(), out, use_gifski=True)
        self.assertIn("gifski", str(ctx.exception))

    def test_encode_gif_falls_back_to_ffmpeg_when_gifski_absent(self):
        """With auto-detect and gifski missing, encode_gif takes the
        ffmpeg two-pass branch. We don't need ffmpeg installed for the
        assertion -- a failing subprocess produces a clear error pointing
        at the ffmpeg branch (palettegen), not gifski."""
        from cgmath.render import frame as frame_mod

        if Image is None:
            self.skipTest("PIL is required for encode_gif")

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.gif")
            with (
                patch("cgmath.render.frame.shutil.which", return_value=None),
                patch("cgmath.render.frame.subprocess.run", side_effect=FileNotFoundError("execvp")),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_gif(self._single_frame(), out)
        self.assertIn("ffmpeg", str(ctx.exception))


# ============================================================================
# Tests for the unified RGBA-background renderer pipeline
# ============================================================================


class TestRendererRGBABackground(unittest.TestCase):
    """Renderer always emits ``(H, W, 4)`` now; ``background`` is RGBA."""

    def _scene(self):
        m, u = _make_textured_cube()
        return _scene(m, u)

    def test_default_bg_is_transparent_black_4ch(self):
        from cgmath.render.raytracer import render

        f = render(scene=self._scene(), resolution=(64, 64))
        self.assertEqual(f.array.shape, (64, 64, 4))
        # the top-left corner misses the centred cube
        np.testing.assert_array_equal(f.array[0, 0], [0, 0, 0, 0])

    def test_opaque_bg_fills_misses_with_color(self):
        from cgmath.render.raytracer import render

        f = render(scene=self._scene(), resolution=(64, 64), background=(1.0, 1.0, 1.0, 1.0))
        np.testing.assert_array_equal(f.array[0, 0], [255, 255, 255, 255])

    def test_semitransparent_bg_passes_through_at_misses(self):
        from cgmath.render.raytracer import render

        f = render(scene=self._scene(), resolution=(64, 64), background=(1.0, 0.0, 0.0, 0.5))
        # a miss is the background verbatim (not premultiplied); 0.5 * 255 = 127.5 -> 127
        np.testing.assert_array_equal(f.array[0, 0], [255, 0, 0, 127])

    def test_hit_pixel_alpha_is_one(self):
        from cgmath.render.raytracer import render

        f = render(scene=self._scene(), resolution=(64, 64), background=(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(int(f.array[32, 32, 3]), 255)

    def test_invalid_bg_length_raises(self):
        from cgmath.render.raytracer import render

        with self.assertRaisesRegex(ValueError, "RGBA 4-tuple"):
            render(scene=self._scene(), resolution=(8, 8), background=(0.0, 0.0, 0.0))

    def test_invalid_bg_range_raises(self):
        from cgmath.render.raytracer import render

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            render(scene=self._scene(), resolution=(8, 8), background=(0.0, 0.0, 0.0, 1.5))

    def test_rgba_kwarg_removed(self):
        from cgmath.render.raytracer import render

        with self.assertRaises(TypeError):
            render(scene=self._scene(), resolution=(8, 8), rgba=True)

    def test_composite_over_helper_porter_duff(self):
        from cgmath.render.raytracer import _composite_over

        src = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],  # fully covered
                [1.0, 0.0, 0.0, 0.5],  # half-covered
                [1.0, 0.0, 0.0, 0.0],  # missed
            ],
            dtype=np.float32,
        )
        out = _composite_over(src, (0.0, 1.0, 0.0, 1.0))  # over opaque green
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out[2], [0.0, 1.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(out[1, 3], 1.0, places=6)


# ============================================================================
# Tests for Scene.background normalization
# ============================================================================


class TestBackgroundProperty(unittest.TestCase):
    """RGB taken opaque, RGBA kept, refused outside [0, 1]; a change drops the frame"""

    def test_none_stays_none(self):
        self.assertIsNone(Scene("s").background)

    def test_rgb_promoted_to_opaque_rgba(self):
        s = Scene("s", background=(0.2, 0.3, 0.4))
        np.testing.assert_array_equal(s.background, (0.2, 0.3, 0.4, 1.0))

    def test_rgba_kept_as_is(self):
        s = Scene("s", background=(0.2, 0.3, 0.4, 0.5))
        np.testing.assert_array_equal(s.background, (0.2, 0.3, 0.4, 0.5))

    def test_invalid_length_raises(self):
        with self.assertRaisesRegex(ValueError, "Scene.background"):
            Scene("s", background=(0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "Scene.background"):
            Scene("s", background=(0.5, 0.5, 0.5, 0.5, 0.5))

    def test_out_of_range_raises(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            Scene("s", background=(1.5, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            Scene("s", background=(0.0, 0.0, 0.0, -0.1))

    def test_setter_normalizes_rgb(self):
        s            = Scene("s")
        s.background = (0.1, 0.2, 0.3)
        np.testing.assert_array_equal(s.background, (0.1, 0.2, 0.3, 1.0))

    def test_scene_background_change_invalidates_cache(self):
        mesh, uv = _make_textured_cube()
        scene = _scene(mesh, uv, _material(mesh, _make_checker()))
        scene.render(resolution=(32, 32))
        self.assertIsNotNone(scene.frame)
        scene.background = (0.5, 0.5, 0.5)
        self.assertIsNone(scene.frame)

    def test_the_background_reaches_the_renderer(self):
        from cgmath.render import raytracer as rt

        mesh, uv = _make_textured_cube()
        s        = _scene(mesh, uv, background=(0.0, 1.0, 0.0))
        captured = {}

        def fake_render(**kw):
            captured.update(kw)
            from cgmath.render.frame import Frame

            return Frame(array=np.zeros((4, 4, 4), dtype=np.uint8))

        with patch.object(rt, "render", side_effect=fake_render):
            s.render(resolution=(8, 8))
        np.testing.assert_array_equal(captured["background"], (0.0, 1.0, 0.0, 1.0))

        # a per-call RGB background is taken opaque too
        with patch.object(rt, "render", side_effect=fake_render):
            s.render(resolution=(8, 8), background=(1.0, 0.0, 0.0))
        np.testing.assert_array_equal(captured["background"], (1.0, 0.0, 0.0, 1.0))

    def test_scene_render_works_without_background(self):
        from cgmath.render import raytracer as rt

        mesh, uv = _make_textured_cube()
        s = _scene(mesh, uv)
        self.assertIsNone(getattr(s, "background"))

        def fake_render(**kw):
            from cgmath.render.frame import Frame

            self.assertNotIn("background", kw)
            return Frame(array=np.zeros((4, 4, 4), dtype=np.uint8))

        with patch.object(rt, "render", side_effect=fake_render):
            s.render(resolution=(8, 8))


# ============================================================================
# Tests for encoder background plumbing (mp4 flatten + avi alpha + dispatcher)
# ============================================================================


class TestEncoderBackgroundPlumbing(unittest.TestCase):
    """Frame.encode_mp4 background kwarg, encode_avi alpha preservation,
    encode dispatcher recognition of .avi, Scene.turntable plumbing."""

    def test_encode_mp4_validates_background_length(self):
        from cgmath.render.frame import Frame

        with self.assertRaisesRegex(ValueError, "RGB 3-tuple"):
            Frame.encode_mp4(
                iter([np.zeros((8, 8, 3), dtype=np.uint8)]),
                "/tmp/x.mp4",
                background=(0.0, 0.0, 0.0, 1.0),  # 4-tuple, not 3
            )

    def test_encode_mp4_validates_background_range(self):
        from cgmath.render.frame import Frame

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            Frame.encode_mp4(
                iter([np.zeros((8, 8, 3), dtype=np.uint8)]),
                "/tmp/x.mp4",
                background=(1.5, 0.0, 0.0),
            )

    def test_encode_mp4_accepts_rgba_input(self):
        from cgmath.render.frame import Frame

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

        red_rgba = np.tile(np.array([[255, 0, 0, 128]], dtype=np.uint8), (16, 16, 1))
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            t.close()
            try:
                Frame.encode_mp4(
                    iter([red_rgba, red_rgba, red_rgba]),
                    t.name,
                    fps        = 4,
                    background = (1.0, 0.0, 0.0),
                )
                self.assertGreater(os.path.getsize(t.name), 0)
            finally:
                os.unlink(t.name)

    def test_encode_dispatcher_recognises_avi(self):
        from cgmath.render.frame import Frame

        frames = [np.zeros((8, 8, 4), dtype=np.uint8)]
        with (
            patch.object(Frame, "encode_mp4") as mp4,
            patch.object(Frame, "encode_avi") as avi,
            patch.object(Frame, "encode_gif") as gif,
        ):
            mp4.return_value = "out.mp4"
            avi.return_value = "out.avi"
            gif.return_value = "out.gif"
            Frame.encode(iter(frames), "/tmp/x.mp4")
            Frame.encode(iter(frames), "/tmp/x.avi")
            Frame.encode(iter(frames), "/tmp/x.gif")
            mp4.assert_called_once()
            avi.assert_called_once()
            gif.assert_called_once()

    def test_encode_avi_round_trip_preserves_alpha(self):
        from cgmath.render.frame import Frame

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        if Image is None:
            self.skipTest("PIL not installed")

        half_red = np.tile(np.array([[255, 0, 0, 128]], dtype=np.uint8), (16, 16, 1))
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as t:
            t.close()
            try:
                Frame.encode_avi(iter([half_red] * 4), t.name, fps=4)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as p:
                    p.close()
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", t.name, "-frames:v", "1", p.name],
                        check          = True,
                        capture_output = True,
                    )
                    decoded = np.asarray(Image.open(p.name).convert("RGBA"))
                    os.unlink(p.name)
                self.assertEqual(decoded.shape[-1], 4)
                self.assertLess(abs(int(decoded[8, 8, 3]) - 128), 3)
            finally:
                os.unlink(t.name)

    def test_scene_turntable_mp4_uses_scene_background(self):
        from cgmath.render.frame import Frame

        m, u = _make_textured_cube()
        s = _scene(m, u, background=(0.0, 1.0, 0.0))
        with patch.object(Frame, "encode_mp4", return_value="out.mp4") as mock:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
                t.close()
                try:
                    s.turntable(t.name, n_frames=2, resolution=(8, 8))
                finally:
                    os.unlink(t.name)
        self.assertEqual(mock.call_args.kwargs["background"], (0.0, 1.0, 0.0))

    def test_scene_turntable_explicit_background_overrides_scene(self):
        from cgmath.render.frame import Frame

        m, u = _make_textured_cube()
        s = _scene(m, u, background=(0.0, 0.0, 0.0, 0.0))
        with patch.object(Frame, "encode_mp4", return_value="out.mp4") as mock:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
                t.close()
                try:
                    s.turntable(t.name, n_frames=2, resolution=(8, 8), background=(0.5, 0.5, 0.5))
                finally:
                    os.unlink(t.name)
        self.assertEqual(mock.call_args.kwargs["background"], (0.5, 0.5, 0.5))

    def test_scene_turntable_avi_routes_to_encode_avi(self):
        from cgmath.render.frame import Frame

        m, u = _make_textured_cube()
        s = _scene(m, u)
        with patch.object(Frame, "encode_avi", return_value="out.avi") as mock:
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as t:
                t.close()
                try:
                    s.turntable(t.name, n_frames=2, resolution=(8, 8))
                finally:
                    os.unlink(t.name)
        mock.assert_called_once()


# ============================================================================
# Tests for samples_per_pixel validation
# ============================================================================


class TestSamplesPerPixelValidation(unittest.TestCase):
    """an invalid sample count fails at assignment, not at render time"""

    def test_bare_scene_render_does_not_crash(self):
        m, u = _make_textured_cube()
        frame = _scene(m, u).render(resolution=(16, 16))
        self.assertEqual(frame.array.shape, (16, 16, 4))

    def test_setter_rejects_non_squares(self):
        scene = Scene("x")
        for value in (2, 3):
            with self.assertRaisesRegex(ValueError, "perfect square"):
                scene.samples_per_pixel = value

    def test_setter_rejects_zero_and_negative(self):
        scene = Scene("x")
        with self.assertRaisesRegex(ValueError, ">= 1"):
            scene.samples_per_pixel = 0
        with self.assertRaisesRegex(ValueError, ">= 1"):
            scene.samples_per_pixel = -4

    def test_setter_accepts_valid_values(self):
        scene = Scene("x")
        for v in (1, 4, 9, 16, 25, 100, None):
            scene.samples_per_pixel = v
            self.assertEqual(scene.samples_per_pixel, v)

    def test_setter_rejects_non_int(self):
        scene = Scene("x")
        with self.assertRaisesRegex(ValueError, "must be an int"):
            scene.samples_per_pixel = "four"

    def test_constructor_validates(self):
        with self.assertRaisesRegex(ValueError, "perfect square"):
            Scene("x", samples_per_pixel=2)
        for v in (None, 1, 4, 9, 16):
            self.assertEqual(Scene("x", samples_per_pixel=v).samples_per_pixel, v)


if __name__ == "__main__":
    unittest.main()
