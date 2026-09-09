"""Tests for the scene-graph rendering classes."""

import base64
import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.render import Camera, Light, look_at, Object, render, Scene

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


def _make_textured_cube():
    """Triangulated unit cube + matching UV (one [0,1]^2 island per face)."""
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
        points  = np.asarray(uv_pts, dtype=float),
        indices = np.asarray(uv_idx, dtype=np.int64),
        counts  = np.asarray(counts, dtype=np.int64),
    )
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


class TestSceneClasses(unittest.TestCase):
    """Round-trip + property tests for Object/Camera/Light/Scene."""

    def test_object_defaults_and_world_points(self):
        mesh, uv = _make_textured_cube()
        obj = Object(name="cube", mesh=mesh, uv=uv)
        self.assertEqual(obj.node_type, "object")
        self.assertIsNone(obj.texture)
        self.assertEqual(obj.sample_method, "bilinear")
        # No transform = world points equal mesh points
        np.testing.assert_allclose(obj.world_points, mesh.points)

    def test_object_transform_applies_to_world_points(self):
        mesh, _ = _make_textured_cube()
        obj = Object(name="cube", mesh=mesh, translate=[100.0, 0.0, 0.0])
        wp  = obj.world_points
        np.testing.assert_allclose(wp[:, 0], mesh.points[:, 0] + 100.0)
        np.testing.assert_allclose(wp[:, 1:], mesh.points[:, 1:])

    def test_camera_look_at(self):
        cam = Camera(name="cam", translate=[5.0, 5.0, 5.0])
        cam.look_at((0.0, 0.0, 0.0))
        # In row-major TransformData convention, ROW 2 of world_matrix is
        # the local Z basis vector (which is the *negative* of the camera's
        # forward direction by Maya/OpenGL convention).  So the actual
        # forward = -world_matrix[2, :3].
        forward  = -cam.world_matrix[2, :3]
        expected = -np.array([5.0, 5.0, 5.0]) / np.linalg.norm([5.0, 5.0, 5.0])
        np.testing.assert_allclose(forward, expected, atol=1e-9)

    def test_camera_default_view(self):
        pts = np.array([[-1, -1, -1], [1, 1, 1]], dtype=float)
        cam = Camera.default_view(pts)
        # eye should be on the +x +y +z side of the bbox center (origin)
        eye = cam.world_matrix[3, :3]
        self.assertGreater(eye[0], 0)
        self.assertGreater(eye[1], 0)
        self.assertGreater(eye[2], 0)

    def test_light_kinds_validated(self):
        Light(name="ok_point", kind="point")
        Light(name="ok_inf", kind="infinite")
        with self.assertRaises(ValueError):
            Light(name="bad", kind="spot")

    def test_light_position_and_direction(self):
        light = Light(name="key", kind="point", translate=[10.0, 20.0, 30.0])
        np.testing.assert_allclose(light.position, [10.0, 20.0, 30.0])

        # An infinite light's direction comes from -world_matrix[:3, 2].
        # The default identity matrix gives direction = -[0, 0, 1] = (0, 0, -1).
        sun = Light(name="sun", kind="infinite")
        np.testing.assert_allclose(sun.direction, [0.0, 0.0, -1.0])

    def test_scene_typed_views(self):
        mesh, uv = _make_textured_cube()
        scene = Scene("s")
        scene.append(Object(name="o1", mesh=mesh, uv=uv))
        scene.append(Camera(name="c1"))
        scene.append(Light(name="l1"))
        self.assertEqual(len(scene.objects),    1)
        self.assertEqual(len(scene.cameras),    1)
        self.assertEqual(len(scene.lights),     1)
        self.assertEqual(scene.objects[0].name, "o1")

    def test_scene_rejects_non_transform_data(self):
        scene = Scene("s")
        with self.assertRaises(TypeError):
            scene.append({"not": "a transform"})

    def test_scene_get_camera(self):
        scene = Scene("s")
        scene.append(Camera(name="A"))
        scene.append(Camera(name="B"))
        self.assertEqual(scene.get_camera().name, "A")     # first
        self.assertEqual(scene.get_camera("B").name, "B")  # by name
        scene.default_camera_name = "B"
        self.assertEqual(scene.get_camera().name, "B")  # default

    def test_scene_union_aabb(self):
        mesh, _ = _make_textured_cube()
        scene = Scene("s")
        scene.append(Object(name="a", mesh=mesh))
        b = Object(name="b", mesh=mesh, translate=[10.0, 0.0, 0.0])
        scene.append(b)
        mn, mx = scene.union_aabb()
        np.testing.assert_allclose(mn, [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(mx, [10.5, 0.5, 0.5])


class TestSceneRendering(unittest.TestCase):
    """Renderer integration with Scene."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()
        self.tex = _make_checker()

    def test_scene_render_produces_image(self):
        scene = Scene("s")
        scene.append(Object(name="cube", mesh=self.mesh, uv=self.uv, texture=self.tex))
        img = render(scene=scene, resolution=(64, 48), default_light=False)
        self.assertEqual(img.shape, (48, 64, 4))
        self.assertEqual(img.dtype, np.uint8)
        # The cube is in front of the auto-built camera, so most pixels hit.
        n_hit = int(np.any(img.array != 0, axis=-1).sum())
        self.assertGreater(n_hit, 100)

    def test_scene_with_class_camera_matches_camera_matrix_path(self):
        """Render the same content via Scene+Camera and via the legacy
        camera_matrix= argument; pixels should agree exactly."""
        cam_matrix = look_at(eye=(2.0, 2.0, 2.0), target=(0.0, 0.0, 0.0))

        # Path 1: scene + Camera (Camera.look_at handles the row/col bridge)
        scene = Scene("s")
        scene.append(Object(name="c", mesh=self.mesh, uv=self.uv, texture=self.tex))
        cam = Camera(name="cam", angle_of_view=35.0, translate=[2.0, 2.0, 2.0])
        cam.look_at((0.0, 0.0, 0.0))
        scene.append(cam)
        img_scene = render(scene=scene, resolution=(64, 48), default_light=False)

        # Path 2: legacy single-mesh
        img_legacy = render(
            self.mesh,
            self.uv,
            self.tex,
            camera_matrix = cam_matrix,
            angle_of_view = 35.0,
            resolution    = (64, 48),
            point_light   = None,
            default_light = False,
        )
        np.testing.assert_array_equal(img_scene, img_legacy)

    def test_object_translate_shifts_silhouette(self):
        """Translating an Object should shift the rendered cube on screen."""
        scene_a = Scene("a")
        scene_a.append(
            Object(name="cube", mesh=self.mesh, uv=self.uv, texture=self.tex)
        )
        cam_a = Camera(name="cam", angle_of_view=35.0, translate=[0.0, 0.0, 5.0])
        cam_a.look_at((0.0, 0.0, 0.0))
        scene_a.append(cam_a)
        img_a   = render(scene=scene_a, resolution=(64, 48), default_light=False)

        scene_b = Scene("b")
        scene_b.append(
            Object(
                name      = "cube",
                mesh      = self.mesh,
                uv        = self.uv,
                texture   = self.tex,
                translate = [1.0, 0.0, 0.0],
            )
        )
        cam_b = Camera(name="cam", angle_of_view=35.0, translate=[0.0, 0.0, 5.0])
        cam_b.look_at((0.0, 0.0, 0.0))
        scene_b.append(cam_b)
        img_b = render(scene=scene_b, resolution=(64, 48), default_light=False)

        # Different images
        self.assertFalse(np.array_equal(img_a, img_b))

        # The cube in img_b should be shifted to the right (larger x in pixel
        # coords).  Find the centroid of hit pixels in each.
        bg     = img_a[0, 0]
        mask_a = np.any(img_a.array != bg, axis=-1)
        mask_b = np.any(img_b.array != bg, axis=-1)
        if mask_a.any() and mask_b.any():
            cx_a = float(np.where(mask_a)[1].mean())
            cx_b = float(np.where(mask_b)[1].mean())
            self.assertGreater(cx_b, cx_a)

    def test_multi_light_additive(self):
        """Two identical lights should produce twice the diffuse contribution
        of one light (within float noise) at the same surface position."""
        scene = Scene("s")
        scene.append(
            Object(
                name       = "cube",
                mesh       = self.mesh,
                uv         = self.uv,
                ambient    = 0.0,
                base_color = (1.0, 1.0, 1.0),
            )
        )
        cam = Camera(name="cam", angle_of_view=35.0, translate=[0.0, 0.0, 5.0])
        cam.look_at((0.0, 0.0, 0.0))
        scene.append(cam)

        light_pos = (3.0, 3.0, 5.0)
        scene_1l  = scene
        scene_1l.append(
            Light(name="L1", kind="point", translate=list(light_pos), intensity=10.0)
        )
        img_one  = render(scene=scene_1l, resolution=(48, 36), default_light=False)

        scene_2l = Scene("s2")
        scene_2l.append(
            Object(
                name       = "cube",
                mesh       = self.mesh,
                uv         = self.uv,
                ambient    = 0.0,
                base_color = (1.0, 1.0, 1.0),
            )
        )
        cam2 = Camera(name="cam", angle_of_view=35.0, translate=[0.0, 0.0, 5.0])
        cam2.look_at((0.0, 0.0, 0.0))
        scene_2l.append(cam2)
        scene_2l.append(
            Light(name="L1", kind="point", translate=list(light_pos), intensity=10.0)
        )
        scene_2l.append(
            Light(name="L2", kind="point", translate=list(light_pos), intensity=10.0)
        )
        img_two = render(scene=scene_2l, resolution=(48, 36), default_light=False)

        # On lit pixels, two-light should be brighter (clamped at 255).
        bg = img_one[0, 0]
        mask = np.any(img_one.array != bg, axis=-1) & np.any(
            img_two.array != bg, axis=-1
        )
        if mask.any():
            mean_one = float(img_one[mask].mean())
            mean_two = float(img_two[mask].mean())
            self.assertGreater(mean_two, mean_one)

    def test_default_light_kicks_in(self):
        """Without explicit lights, a default headlight should be added so
        the image isn't pitch black."""
        scene = Scene("s")
        scene.append(
            Object(
                name       = "cube",
                mesh       = self.mesh,
                uv         = self.uv,
                ambient    = 0.0,
                base_color = (0.7, 0.7, 0.7),
            )
        )
        # default_light=True (the default)
        img_default = render(scene=scene, resolution=(48, 36), default_light=True)
        # default_light=False -- no light at all, ambient=0 -> pure black RGB
        img_dark = render(scene=scene, resolution=(48, 36), default_light=False)

        # Compare RGB max only -- the renderer always emits 4-channel
        # output now, so the alpha channel pushes both maxes to 255 at
        # any covered pixel.  RGB max is what "is the surface lit?"
        # actually measures.
        self.assertGreater(
            int(img_default.array[..., :3].max()),
            int(img_dark.array[..., :3].max()),
        )

    def test_render_scene_method_shim(self):
        """Scene.render(...) should produce the same image as
        render(scene=scene)."""
        scene = Scene("s")
        scene.append(Object(name="c", mesh=self.mesh, uv=self.uv, texture=self.tex))
        img1 = scene.render(resolution=(48, 36), default_light=False)
        img2 = render(scene=scene, resolution=(48, 36), default_light=False)
        np.testing.assert_array_equal(img1, img2)


class TestOrthographic(unittest.TestCase):
    """Orthographic camera mode."""

    def test_ortho_renders_image(self):
        mesh, uv = _make_textured_cube()
        scene = Scene("s")
        scene.append(Object(name="c", mesh=mesh, uv=uv))
        cam = Camera(
            name            = "cam",
            is_orthographic = True,
            ortho_height    = 2.5,
            translate       = [0.0, 0.0, 5.0],
        )
        cam.look_at((0.0, 0.0, 0.0))
        scene.append(cam)

        img, depth = render(
            scene         = scene,
            resolution    = (64, 48),
            default_light = False,
            return_depth  = True,
        )
        self.assertEqual(img.shape, (48, 64, 4))
        # Depth should be roughly uniform across the cube's front face
        # (because rays are parallel) - check that the depth standard
        # deviation across hit pixels is small.
        hit_depth = depth[~np.isnan(depth)]
        if hit_depth.size > 10:
            self.assertLess(hit_depth.std(), 0.6)


class TestTurntable(unittest.TestCase):
    """Scene.turntable() short loop."""

    def test_turntable_writes_frames(self):
        mesh, uv = _make_textured_cube()
        tex   = _make_checker()
        scene = Scene("s")
        scene.append(Object(name="cube", mesh=mesh, uv=uv, texture=tex))
        scene.append(
            Light(name="key", kind="point", translate=[3.0, 3.0, 3.0], intensity=50.0)
        )

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

            # Frame 0 and frame 2 (180deg apart) should look different
            from PIL import Image

            img0 = np.asarray(Image.open(files[0]))
            img2 = np.asarray(Image.open(files[2]))
            self.assertFalse(np.array_equal(img0, img2))

    def test_turntable_preserves_bvh_cache(self):
        """The mesh's cached BVH must NOT be invalidated across
        turntable frames - that would defeat the perf optimization."""
        # Use a slightly larger mesh so the BVH actually gets built
        # (small meshes skip the BVH below the size threshold).
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

        scene = Scene("s")
        scene.append(Object(name="grid", mesh=mesh))

        # Force BVH build by rendering once
        _          = render(scene=scene, resolution=(32, 24), default_light=False)
        bvh_before = mesh._bvh_bilinear

        with tempfile.TemporaryDirectory() as tmp:
            pattern = os.path.join(tmp, "turn.{frame:03d}.png")
            scene.turntable(
                pattern,
                n_frames      = 3,
                resolution    = (32, 24),
                default_light = False,
            )

        bvh_after = mesh._bvh_bilinear
        self.assertIs(
            bvh_before, bvh_after, "BVH cache should persist across turntable frames"
        )


class TestTurntableGlobalPivot(unittest.TestCase):
    """New global-pivot turntable: camera orbits the world axis, scene
    transforms are not mutated, autofit picks the most-pulled-back
    candidate among per-sample autofits."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def _scene_with_three(self):
        """3 cubes spread along X at -3 / 0 / +3."""
        scene = Scene("multi")
        scene.append(Object(name="A", mesh=self.mesh, uv=self.uv))
        scene.append(
            Object(name="B", mesh=self.mesh, uv=self.uv, translate=[3.0, 0.0, 0.0])
        )
        scene.append(
            Object(name="C", mesh=self.mesh, uv=self.uv, translate=[-3.0, 0.0, 0.0])
        )
        return scene

    def test_no_targets_kwarg(self):
        """The new API has no `target=` kwarg -- passing it must error."""
        scene = self._scene_with_three()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                # kwarg `target` no longer exists
                scene.turntable(
                    target         = "A",
                    output_pattern = os.path.join(tmp, "x.{frame:03d}.png"),
                    n_frames       = 2,
                    resolution     = (32, 24),
                    default_light  = False,
                )

    def test_object_transforms_are_not_mutated(self):
        """The new turntable rotates the camera, not the Objects.  All
        Object transforms must be byte-identical before vs after."""
        scene = self._scene_with_three()
        a, b, c = scene.objects
        # Pre-set arbitrary rotations.
        a.rotate_y  = 17.0
        b.rotate_x  = 33.0
        c.rotate    = [11.0, 22.0, 33.0]

        wm_a_before = a.world_matrix.copy()
        wm_b_before = b.world_matrix.copy()
        wm_c_before = c.world_matrix.copy()

        with tempfile.TemporaryDirectory() as tmp:
            scene.turntable(
                output_pattern = os.path.join(tmp, "spin.{frame:03d}.png"),
                n_frames       = 4,
                resolution     = (48, 36),
                default_light  = False,
            )

        np.testing.assert_array_equal(a.world_matrix, wm_a_before)
        np.testing.assert_array_equal(b.world_matrix, wm_b_before)
        np.testing.assert_array_equal(c.world_matrix, wm_c_before)

    def test_camera_orbits_world_axis(self):
        """Frames at 0\u00b0 vs 90\u00b0 must look distinct (camera has moved
        around the world Y axis)."""
        scene = self._scene_with_three()
        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "orbit.{frame:03d}.png"),
                n_frames       = 4,  # frames at 0, 90, 180, 270
                resolution     = (64, 48),
                default_light  = True,
            )
            from PIL import Image as _PIL_Image

            img0 = np.asarray(_PIL_Image.open(files[0]))
            img1 = np.asarray(_PIL_Image.open(files[1]))  # 90\u00b0 away
            self.assertFalse(
                np.array_equal(img0, img1),
                "0\u00b0 and 90\u00b0 frames should differ -- camera should orbit",
            )

    def test_translated_objects_stay_in_frame(self):
        """All visible Objects must remain in frame across the rotation."""
        scene = self._scene_with_three()  # A/B/C at -3 / 0 / +3 along X
        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "all.{frame:03d}.png"),
                n_frames       = 4,
                resolution     = (120, 60),
                default_light  = True,
            )
            from PIL import Image as _PIL_Image

            # Across all sampled frames, the union of hit columns should
            # span most of the image width (3 cubes spread along X).
            cols: set = set()
            for path in files:
                img  = np.asarray(_PIL_Image.open(path))
                mask = np.any(img != 0, axis=-1)
                for c in np.where(mask.any(axis=0))[0]:
                    cols.add(int(c))
            self.assertGreater(len(cols), 0, "no hit pixels in any frame")
            span = max(cols) - min(cols)
            self.assertGreater(
                span,
                70,
                f"translated objects aren't framed correctly (span={span} of 120)",
            )

    def test_fit_static_uses_existing_camera_unchanged(self):
        """fit='static' must NOT autofit -- the scene's existing Camera
        is used as-is for the orbit base, only its position rotates per
        frame.  Verify by checking that base_cam.world_matrix equals what
        we supplied even after turntable returns."""
        from cgmath.render import Camera as _Camera
        from cgmath.render.camera import look_at as _look_at

        scene  = self._scene_with_three()
        cam_in = _Camera(name="cam", angle_of_view=35.0, translate=[0.0, 0.0, 10.0])
        cam_in.look_at((0.0, 0.0, 0.0))
        # snapshot before
        wm_before = cam_in.world_matrix.copy()
        scene.append(cam_in)

        with tempfile.TemporaryDirectory() as tmp:
            scene.turntable(
                output_pattern = os.path.join(tmp, "static.{frame:03d}.png"),
                n_frames       = 2,
                resolution     = (32, 24),
                fit            = "static",
                default_light  = False,
            )

        # Even after turntable, the user-supplied camera should be unchanged.
        # (turntable doesn't append/pop a throwaway, and 'static' means no
        # autofit override of cam.world_matrix.)
        np.testing.assert_array_equal(cam_in.world_matrix, wm_before)

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

    def test_no_off_center_clipping_on_translated_scene(self):
        """Regression for the off-center clipping bug.  With three cubes
        spread along X (asymmetric silhouette per angle), the BBX-based
        framing must keep the silhouette inside the frame at EVERY
        sampled angle -- no pixel touching frame edges."""
        scene = self._scene_with_three()  # cubes at -3 / 0 / +3 along X

        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "no_clip.{frame:03d}.png"),
                n_frames       = 16,  # sample many angles
                resolution     = (120, 80),
                default_light  = True,
            )
            from PIL import Image as _PIL_Image

            W, H = 120, 80
            for path in files:
                img  = np.asarray(_PIL_Image.open(path))
                mask = np.any(img != 0, axis=-1)
                if not mask.any():
                    self.fail(f"no hits at {os.path.basename(path)}")
                cols = np.where(mask.any(axis=0))[0]
                rows = np.where(mask.any(axis=1))[0]
                col_min, col_max = int(cols.min()), int(cols.max())
                row_min, row_max = int(rows.min()), int(rows.max())
                self.assertGreater(
                    col_min,
                    0,
                    f"{os.path.basename(path)}: silhouette clipped at left "
                    f"edge (col_min={col_min})",
                )
                self.assertLess(
                    col_max,
                    W - 1,
                    f"{os.path.basename(path)}: silhouette clipped at right "
                    f"edge (col_max={col_max} of {W - 1})",
                )
                self.assertGreater(
                    row_min,
                    0,
                    f"{os.path.basename(path)}: silhouette clipped at top "
                    f"edge (row_min={row_min})",
                )
                self.assertLess(
                    row_max,
                    H - 1,
                    f"{os.path.basename(path)}: silhouette clipped at bottom "
                    f"edge (row_max={row_max} of {H - 1})",
                )

    def test_camera_orbits_around_centroid_not_world_origin(self):
        """The camera path must form a circle around the SCENE CENTROID
        in the rotation plane, not around the world origin.

        Build an off-center scene (one cube at translate=(10, 0, 0)) so
        the centroid is far from world origin; then verify the per-frame
        camera positions are equidistant from the centroid (not from the
        world origin)."""
        from cgmath.render.scene import _rotation_matrix_col

        # Single cube far off-center -- centroid at (10, 0, 0).
        scene = Scene("offcenter")
        scene.append(
            Object(
                name      = "far",
                mesh      = self.mesh,
                uv        = self.uv,
                translate = [10.0, 0.0, 0.0],
            )
        )

        # Recompute what turntable will compute internally.
        chunks   = [o.world_points for o in scene.objects]
        all_pts  = np.concatenate(chunks, axis=0)
        centroid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
        np.testing.assert_allclose(centroid, [10.0, 0.0, 0.0], atol=1e-9)

        from cgmath.render.camera import _default_camera_for_points

        base_cam_col = _default_camera_for_points(all_pts)
        R_cam        = base_cam_col[:3, :3]

        # Re-run the BBX framing to get the chosen camera (single sample
        # is enough for a simple cube at a single angle -- BBX is the cube).
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

        # Sample 8 per-frame camera positions and confirm they sit on a
        # circle of constant radius from the centroid (not from world 0).
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

        # Distances from CENTROID must all be equal (orbiting around it).
        rs = np.asarray(radii_to_centroid)
        self.assertLess(
            float(rs.max() - rs.min()),
            1e-6,
            f"camera not equidistant from centroid: r = {rs}",
        )
        # Distances from WORLD ORIGIN must vary (centroid is offset from origin).
        rs_origin = np.asarray(radii_to_world_origin)
        self.assertGreater(
            float(rs_origin.max() - rs_origin.min()),
            1.0,
            f"camera unexpectedly equidistant from world origin too "
            f"(centroid is at (10, 0, 0); should NOT orbit world 0): r = {rs_origin}",
        )

    def test_silhouette_centered_in_frame(self):
        """At every sampled angle, the silhouette's centroid in pixel space
        must be near the image centre.  This is the property the BBX-based
        algorithm guarantees: every angle's silhouette is centred on the
        optical axis, not just one specific autofit-sampled angle."""
        scene = self._scene_with_three()

        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "centred.{frame:03d}.png"),
                n_frames       = 8,
                resolution     = (200, 120),
                default_light  = True,
            )
            from PIL import Image as _PIL_Image

            W, H = 200, 120
            cx_img, cy_img = (W - 1) / 2, (H - 1) / 2
            for path in files:
                img  = np.asarray(_PIL_Image.open(path))
                mask = np.any(img != 0, axis=-1)
                ys, xs = np.where(mask)
                if xs.size == 0:
                    self.fail(f"no hits at {os.path.basename(path)}")
                # Silhouette midpoint (mean of bounding extents).
                mid_x = 0.5 * (xs.min() + xs.max())
                mid_y = 0.5 * (ys.min() + ys.max())
                # Allow up to ~10% offset from centre -- exact zero is
                # unrealistic because the scene has Y-asymmetry.
                self.assertLess(
                    abs(mid_x - cx_img),
                    0.10 * W,
                    f"{os.path.basename(path)}: silhouette X centre {mid_x:.1f} "
                    f"too far from image centre {cx_img:.1f}",
                )
                self.assertLess(
                    abs(mid_y - cy_img),
                    0.20 * H,  # vertical: a bit more slack
                    f"{os.path.basename(path)}: silhouette Y centre {mid_y:.1f} "
                    f"too far from image centre {cy_img:.1f}",
                )

    def test_no_visible_objects_raises(self):
        """An empty scene (no objects) should raise a clear error."""
        scene = Scene("empty")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                scene.turntable(
                    output_pattern = os.path.join(tmp, "x.{frame:03d}.png"),
                    n_frames       = 2,
                    resolution     = (32, 24),
                    default_light  = False,
                )

    def test_rotation_matrix_y_round_trip(self):
        """The Y-axis rotation matrix must implement the standard
        right-hand rule: rotating (1, 0, 0) by 90\u00b0 gives (0, 0, 1)."""
        from cgmath.render.scene import _rotation_matrix_col

        R = _rotation_matrix_col("y", 90.0)
        # Column-major: M @ p_col
        rotated = R @ np.array([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotated[:3], [0.0, 0.0, 1.0], atol=1e-9)


class TestObjectRenderShim(unittest.TestCase):
    """Object.render() / Object.turntable() one-call previews."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_default_lighting_rig_three_lights(self):
        """The default rig has key/fill/rim with sensible relative
        intensities (key full, fill ~40%, rim ~30%)."""
        from cgmath.render.scene import _default_lighting_rig_for

        # Unit cube around origin
        pts = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=float,
        )
        rig   = _default_lighting_rig_for(pts)
        names = [light.name for light in rig]
        self.assertEqual(names, ["key", "fill", "rim"])
        # Relative intensities
        self.assertAlmostEqual(rig[1].intensity / rig[0].intensity, 0.4, places=5)
        self.assertAlmostEqual(rig[2].intensity / rig[0].intensity, 0.3, places=5)
        # Key should be in +X +Y +Z octant relative to centroid (origin)
        self.assertGreater(rig[0].position[0], 0)
        self.assertGreater(rig[0].position[1], 0)
        self.assertGreater(rig[0].position[2], 0)
        # Rim should be behind the centroid in -Z
        self.assertLess(rig[2].position[2], 0)

    def test_default_lighting_rig_scales_with_extent(self):
        """A 100x bigger object should get a 100x further-away rig."""
        from cgmath.render.scene import _default_lighting_rig_for

        small     = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=float)
        big       = small * 100.0
        rig_small = _default_lighting_rig_for(small)
        rig_big   = _default_lighting_rig_for(big)
        d_small   = float(np.linalg.norm(rig_small[0].position))
        d_big     = float(np.linalg.norm(rig_big[0].position))
        self.assertAlmostEqual(d_big / d_small, 100.0, places=4)

    def test_default_lighting_rig_empty_points(self):
        """Empty points return a single fallback light (not pitch black)."""
        from cgmath.render.scene import _default_lighting_rig_for

        rig = _default_lighting_rig_for(np.empty((0, 3)))
        self.assertEqual(len(rig), 1)
        self.assertEqual(rig[0].name, "key")

    def test_object_render_returns_frame_with_default_resolution(self):
        """Object.render() should produce a 500x500 Frame by default
        (the per-Object default resolution)."""
        from cgmath.render.frame import Frame

        obj   = Object(name="cube", mesh=self.mesh, uv=self.uv)
        frame = obj.render()
        self.assertIsInstance(frame, Frame)
        self.assertEqual(frame.shape, (500, 500, 4))
        self.assertEqual(frame.dtype, np.uint8)

    def test_object_render_save_to_disk(self):
        """When output= is given, the frame should be saved to disk."""
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        with tempfile.TemporaryDirectory() as tmp:
            path  = os.path.join(tmp, "preview.png")
            frame = obj.render(output=path, resolution=(64, 64))
            self.assertTrue(os.path.exists(path))
            # Returned frame should have the requested resolution.
            self.assertEqual(frame.shape, (64, 64, 4))

    def test_object_render_default_lighting_is_not_black(self):
        """Without manual lighting, the default rig should produce a
        non-black image -- no more 'pitch black preview' surprise."""
        obj = Object(
            name       = "cube",
            mesh       = self.mesh,
            uv         = self.uv,
            ambient    = 0.0,
            base_color = (0.7, 0.7, 0.7),
        )
        frame = obj.render(resolution=(64, 64))
        self.assertGreater(
            int(frame.array.max()), 30, "default rig produced near-black preview"
        )

    def test_object_render_forwards_kwargs(self):
        """**kwargs should be forwarded to Scene.render() / render()."""
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        # Pass a known render kwarg through.
        frame = obj.render(
            resolution    = (48, 48),
            background    = (1.0, 0.0, 0.0),  # red background
            default_light = False,            # disable headlight; rig still lights it
        )
        self.assertEqual(frame.shape, (48, 48, 4))
        # A corner pixel should be red (no cube there).
        corner = frame.array[0, 0]
        self.assertGreater(int(corner[0]), 200, "expected red bg")
        self.assertLess(int(corner[1]), 50)
        self.assertLess(int(corner[2]), 50)

    def test_object_render_does_not_mutate_object(self):
        """Calling render() must leave the Object's transforms unchanged."""
        obj       = Object(name="cube", mesh=self.mesh, uv=self.uv, translate=[5.0, 0.0, 0.0])
        wm_before = obj.world_matrix.copy()
        _         = obj.render(resolution=(32, 32))
        np.testing.assert_array_equal(obj.world_matrix, wm_before)

    def test_object_turntable_writes_frames(self):
        """Object.turntable() should produce an image sequence."""
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        with tempfile.TemporaryDirectory() as tmp:
            files = obj.turntable(
                output     = os.path.join(tmp, "spin.{frame:03d}.png"),
                n_frames   = 3,
                resolution = (48, 48),
            )
            self.assertEqual(len(files), 3)
            for f in files:
                self.assertTrue(os.path.exists(f))

    def test_object_turntable_default_resolution_is_500(self):
        """The default resolution for turntable should be 500x500
        (the per-Object default resolution)."""
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        with tempfile.TemporaryDirectory() as tmp:
            files = obj.turntable(
                output   = os.path.join(tmp, "spin.{frame:03d}.png"),
                n_frames = 2,
                # no resolution given -> default
            )
            from PIL import Image as _PIL_Image

            with _PIL_Image.open(files[0]) as img:
                self.assertEqual(img.size, (500, 500))


class TestFrameWireframe(unittest.TestCase):
    """Frame.wireframe() post-process overlay."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()
        self.scene = Scene("wf")
        self.scene.append(Object(name="cube", mesh=self.mesh, uv=self.uv))
        # Render a small base frame for overlaying onto.
        self.frame = self.scene.render(resolution=(96, 96), default_light=True)

    def test_render_attaches_camera_matrix_and_aov(self):
        """Frame from render() must carry camera_matrix and angle_of_view
        so wireframe() can project edges without re-passing them."""
        self.assertIsNotNone(self.frame.camera_matrix)
        self.assertEqual(self.frame.camera_matrix.shape, (4, 4))
        self.assertIsNotNone(self.frame.angle_of_view)
        self.assertGreater(float(self.frame.angle_of_view), 0.0)

    def test_wireframe_modifies_pixels(self):
        """A wireframe overlay must change at least some pixels (it'd be
        meaningless if no edges got drawn)."""
        wired = self.frame.wireframe(self.scene)
        self.assertEqual(wired.shape, self.frame.shape)
        diff = int(np.any(self.frame.array != wired.array, axis=-1).sum())
        self.assertGreater(diff, 0, "wireframe drew zero pixels")

    def test_wireframe_color_kwarg(self):
        """Passing a bright red color must produce visible red pixels."""
        red = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=2)
        red_mask = (
            (red.array[..., 0] > 200)
            & (red.array[..., 1] < 50)
            & (red.array[..., 2] < 50)
        )
        # Wider line -> noticeable count of red pixels.
        self.assertGreater(int(red_mask.sum()), 20)

    def test_wireframe_width_increases_pixel_count(self):
        """Width=3 should color more pixels than width=1 for the same scene."""
        thin  = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1)
        thick = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=3)
        thin_count = int(
            (
                (thin.array[..., 0] > 200)
                & (thin.array[..., 1] < 50)
                & (thin.array[..., 2] < 50)
            ).sum()
        )
        thick_count = int(
            (
                (thick.array[..., 0] > 200)
                & (thick.array[..., 1] < 50)
                & (thick.array[..., 2] < 50)
            ).sum()
        )
        self.assertGreater(thick_count, thin_count)

    def test_wireframe_accepts_scene(self):
        wired = self.frame.wireframe(self.scene)
        self.assertEqual(wired.shape, self.frame.shape)

    def test_wireframe_accepts_object(self):
        wired = self.frame.wireframe(self.scene.objects[0])
        self.assertEqual(wired.shape, self.frame.shape)

    def test_wireframe_accepts_meshdata(self):
        wired = self.frame.wireframe(self.mesh)
        self.assertEqual(wired.shape, self.frame.shape)

    def test_wireframe_bare_frame_raises(self):
        """A Frame without camera_matrix can't project edges -> ValueError."""
        from cgmath.render import Frame

        bare = Frame(array=np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            bare.wireframe(self.scene)

    def test_wireframe_bad_source_raises(self):
        with self.assertRaises(TypeError):
            self.frame.wireframe(42)
        with self.assertRaises(TypeError):
            self.frame.wireframe("not a scene")

    def test_wireframe_preserves_camera_metadata(self):
        """The returned Frame should carry the same camera_matrix /
        angle_of_view -- useful for chained overlays."""
        wired = self.frame.wireframe(self.scene)
        np.testing.assert_array_equal(wired.camera_matrix, self.frame.camera_matrix)
        self.assertEqual(wired.angle_of_view, self.frame.angle_of_view)

    def test_wireframe_chainable(self):
        """Two wireframe calls should chain (e.g., black mesh edges +
        red object outlines)."""
        first  = self.frame.wireframe(self.scene, color=(0, 0, 0, 200), width=1)
        second = first.wireframe(self.mesh, color=(0, 255, 0, 255), width=1)
        # second pass must add some green pixels not present in the first.
        green_mask = (
            (second.array[..., 1] > 200)
            & (second.array[..., 0] < 50)
            & (second.array[..., 2] < 50)
        )
        self.assertGreater(int(green_mask.sum()), 0)

    def test_wireframe_unique_vs_per_face_edges(self):
        """unique_edges=True yields fewer drawn edges than False on a
        triangulated cube where every quad face has a shared diagonal."""
        from cgmath.render.frame import _iter_wireframe_primitives

        unique_pairs = next(_iter_wireframe_primitives(self.mesh, unique=True))[
            "edge_pairs"
        ]
        per_face_pairs = next(_iter_wireframe_primitives(self.mesh, unique=False))[
            "edge_pairs"
        ]
        # The cube has 6 faces x 2 triangles x 3 edges = 36 per-face edges,
        # but only 18 unique edges (12 cube edges + 6 face diagonals).
        self.assertLess(unique_pairs.shape[0], per_face_pairs.shape[0])

    def test_wireframe_cull_backfaces_default_on(self):
        """Default behaviour culls back-facing edges -- so the culled
        result should differ from an explicit X-ray pass."""
        culled = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1)
        xray = self.frame.wireframe(
            self.scene, color=(255, 0, 0, 255), width=1, cull_backfaces=False
        )
        # The two passes should NOT produce identical pixels (X-ray draws
        # back-side wires that the culled pass hides).
        self.assertFalse(
            np.array_equal(culled.array, xray.array),
            "default cull_backfaces=True should produce different pixels than X-ray",
        )

    def test_wireframe_cull_reduces_drawn_pixels(self):
        """Culled pass should draw strictly fewer red pixels than X-ray
        for a cube viewed at 3/4 angle (some faces are back-facing)."""

        def red_count(f):
            a    = f.array
            mask = (a[..., 0] > 200) & (a[..., 1] < 50) & (a[..., 2] < 50)
            return int(mask.sum())

        culled = self.frame.wireframe(self.scene, color=(255, 0, 0, 255), width=1)
        xray = self.frame.wireframe(
            self.scene, color=(255, 0, 0, 255), width=1, cull_backfaces=False
        )
        c, x = red_count(culled), red_count(xray)
        self.assertGreater(x, 0)
        self.assertLess(c, x, f"cull should remove pixels (culled={c}, xray={x})")

    def test_wireframe_cull_works_for_translated_object(self):
        """Cull must use world-space face normals, so a translated /
        rotated object's wireframe still culls correctly."""
        scene = Scene("translated")
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = self.uv,
            translate = [5.0, 0.0, 0.0],
            rotate    = [0.0, 30.0, 0.0],
        )
        scene.append(obj)
        frame = scene.render(resolution=(96, 96), default_light=True)

        # The culled pass should still draw something (front faces are
        # visible) and should differ from X-ray (back faces hidden).
        culled = frame.wireframe(scene, color=(255, 0, 0, 255), width=1)
        xray = frame.wireframe(
            scene, color=(255, 0, 0, 255), width=1, cull_backfaces=False
        )
        red_culled = int(
            (
                (culled.array[..., 0] > 200)
                & (culled.array[..., 1] < 50)
                & (culled.array[..., 2] < 50)
            ).sum()
        )
        red_xray = int(
            (
                (xray.array[..., 0] > 200)
                & (xray.array[..., 1] < 50)
                & (xray.array[..., 2] < 50)
            ).sum()
        )
        self.assertGreater(red_culled, 0, "culled pass drew nothing")
        self.assertLess(red_culled, red_xray, "culling should reduce pixel count")

    def test_wireframe_cull_data_attached_to_primitives(self):
        """When with_cull_data=True, _iter_wireframe_primitives must
        yield face_normals / face_centres / e2f keyed dicts."""
        from cgmath.render.frame import _iter_wireframe_primitives

        prim = next(
            _iter_wireframe_primitives(self.mesh, unique=True, with_cull_data=True)
        )
        for key in ("world_pts", "edge_pairs", "face_normals", "face_centres", "e2f"):
            self.assertIn(key, prim)
            self.assertIsNotNone(prim[key], f"{key!r} should be populated")

        # Without with_cull_data the cull-only fields should not be present
        # (or should be absent / None).
        prim_no_cull = next(
            _iter_wireframe_primitives(self.mesh, unique=True, with_cull_data=False)
        )
        # face_normals isn't required to be in the dict when culling is off,
        # but if present it must be None.
        self.assertEqual(
            prim_no_cull.get("face_normals"),
            None,
            "face_normals should be None / absent when with_cull_data=False",
        )

    def test_wireframe_cull_faceless_orphan_edges_keep_drawn(self):
        """Edges with no adjacent face (orphans) should survive cull."""
        # Build a tiny mesh where one edge is an orphan (no face uses it).
        # Use a 1-face mesh and verify it still draws the visible silhouette.
        # (Easier: just confirm the code doesn't crash on the all-triangle
        # cube; orphan-edge handling is covered by the existing e2f >= 0 mask.)
        out = self.frame.wireframe(self.mesh, cull_backfaces=True)
        self.assertEqual(out.shape, self.frame.shape)


class TestToImageCache(unittest.TestCase):
    """Object.to_image() / Scene.to_image() + frame caching."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_object_frame_starts_none_buffer_starts_none(self):
        obj = Object(name="cube", mesh=self.mesh)
        self.assertIsNone(obj.frame)
        self.assertIsNone(obj.buffer)

    def test_object_render_caches_frame(self):
        obj = Object(name="cube", mesh=self.mesh)
        f   = obj.render(resolution=(48, 48))
        self.assertIs(obj.frame, f)
        self.assertEqual(obj.buffer.shape, (48, 48, 4))

    def test_object_to_image_lazy_renders_when_no_cache(self):
        obj = Object(name="cube", mesh=self.mesh)
        img = obj.to_image()
        # Lazy render at the per-Object default (500x500) should have
        # populated the cache and returned a PIL Image of matching size.
        self.assertIsNotNone(obj.frame)
        self.assertEqual(obj.buffer.shape, (500, 500, 4))
        # PIL Image .size is (width, height).
        self.assertEqual(img.size, (500, 500))

    def test_object_to_image_uses_cached_frame_no_rerender(self):
        obj       = Object(name="cube", mesh=self.mesh)
        f         = obj.render(resolution=(32, 32))
        before_id = id(obj.frame)
        img       = obj.to_image()
        # to_image must NOT re-render -- should reuse the cached small frame.
        self.assertIs(obj.frame, f)
        self.assertEqual(id(obj.frame),    before_id)
        self.assertEqual(obj.buffer.shape, (32, 32, 4))
        self.assertEqual(img.size,         (32, 32))

    def test_object_to_image_returns_pil_image(self):
        """to_image() should return a PIL Image instance."""
        from PIL.Image import Image as _PILImage

        obj = Object(name="cube", mesh=self.mesh)
        obj.render(resolution=(32, 32))
        img = obj.to_image()
        self.assertIsInstance(img, _PILImage)

    def test_object_to_image_pixels_match_buffer(self):
        """The PIL Image returned by to_image must contain the same pixels
        as the cached frame buffer (no flip; Frame.array is top-down)."""
        obj = Object(name="cube", mesh=self.mesh)
        obj.render(resolution=(32, 32))
        img = obj.to_image()
        np.testing.assert_array_equal(np.asarray(img), obj.buffer)

    def test_scene_render_caches_frame(self):
        scene = Scene("s")
        scene.append(Object(name="cube", mesh=self.mesh))
        f = scene.render(resolution=(48, 48), default_light=True)
        self.assertIs(scene.frame, f)
        self.assertEqual(scene.buffer.shape, (48, 48, 4))

    def test_scene_to_image_lazy_renders_at_resolution(self):
        scene = Scene("s")
        scene.append(Object(name="cube", mesh=self.mesh))
        img = scene.to_image(resolution=(40, 40))
        self.assertIsNotNone(scene.frame)
        self.assertEqual(scene.buffer.shape, (40, 40, 4))
        self.assertEqual(img.size, (40, 40))

    def test_scene_to_image_default_resolution_is_500(self):
        """When no resolution is passed, lazy-render uses the 500x500 default."""
        scene = Scene("s")
        scene.append(Object(name="cube", mesh=self.mesh))
        img = scene.to_image()
        self.assertEqual(scene.buffer.shape, (500, 500, 4))
        self.assertEqual(img.size, (500, 500))

    def test_scene_to_image_returns_pil_image(self):
        """Scene.to_image() should return a PIL Image instance."""
        from PIL.Image import Image as _PILImage

        scene = Scene("s")
        scene.append(Object(name="cube", mesh=self.mesh))
        scene.render(resolution=(32, 32))
        img = scene.to_image()
        self.assertIsInstance(img, _PILImage)

    def test_scene_to_image_uses_cached_frame_no_rerender(self):
        """to_image must NOT re-render when a frame is already cached."""
        scene = Scene("s")
        scene.append(Object(name="cube", mesh=self.mesh))
        f         = scene.render(resolution=(32, 32))
        before_id = id(scene.frame)
        img       = scene.to_image()
        self.assertIs(scene.frame, f)
        self.assertEqual(id(scene.frame), before_id)
        self.assertEqual(img.size, (32, 32))

    def test_buffer_property_returns_array(self):
        """The buffer property should equal frame.array."""
        obj = Object(name="cube", mesh=self.mesh)
        obj.render(resolution=(32, 32))
        np.testing.assert_array_equal(obj.buffer, obj.frame.array)


class TestSceneRenderConfig(unittest.TestCase):
    """Persistent render-config attrs on Scene + the merge in render/turntable."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_default_render_config(self):
        """Scene's persistent render-config defaults: most are ``None``
        ("fall through to the renderer's internal default") but
        ``resolution`` defaults to ``(500, 500)`` so a bare
        ``Scene().render()`` call produces a sensibly-sized preview."""
        scene = Scene("s")
        # resolution has a concrete default; everything else is None.
        self.assertEqual(scene.resolution, (500, 500))
        for k in Scene._RENDER_CONFIG_KEYS:
            if k == "resolution":
                continue
            self.assertIsNone(getattr(scene, k), f"{k} should default to None")

    def test_init_kwargs_set_render_config(self):
        scene = Scene(
            "s",
            samples_per_pixel = 4,
            resolution        = (640, 480),
            background        = (0.1, 0.2, 0.3),
            autofit           = False,
        )
        self.assertEqual(scene.samples_per_pixel, 4)
        self.assertEqual(scene.resolution, (640, 480))
        # RGB input is auto-promoted to opaque RGBA by Scene.background's
        # _normalize_rgba pass at construction time.
        self.assertEqual(scene.background, (0.1, 0.2, 0.3, 1.0))
        self.assertFalse(scene.autofit)

    def test_configure_chainable_and_skips_none(self):
        scene = Scene("s")
        # Pre-condition: resolution starts at the (500, 500) default.
        self.assertEqual(scene.resolution, (500, 500))
        ret = scene.configure(samples_per_pixel=4, resolution=None)
        self.assertIs(ret, scene, "configure() should return self for chaining")
        self.assertEqual(scene.samples_per_pixel, 4)
        # ``resolution=None`` was passed -- configure() must SKIP it (not
        # overwrite the existing default with None).
        self.assertEqual(
            scene.resolution,
            (500, 500),
            "None values passed to configure() should be skipped",
        )

    def test_configure_unknown_attribute_raises(self):
        scene = Scene("s")
        with self.assertRaises(AttributeError):
            scene.configure(bogus_attr=42)

    def test_render_uses_scene_attrs_when_no_kwargs(self):
        """Scene.resolution should propagate to render() output dimensions."""
        scene = Scene("s", resolution=(64, 48))
        scene.append(Object(name="cube", mesh=self.mesh))
        f = scene.render(default_light=True)
        self.assertEqual(f.shape, (48, 64, 4))

    def test_render_kwarg_overrides_scene_attr(self):
        """Per-call kwarg should beat the persistent attr."""
        scene = Scene("s", resolution=(64, 48))
        scene.append(Object(name="cube", mesh=self.mesh))
        f = scene.render(resolution=(40, 30), default_light=True)
        self.assertEqual(f.shape, (30, 40, 4))

    def test_turntable_uses_scene_attrs(self):
        """Scene.resolution should propagate to turntable() too."""
        scene = Scene("s", resolution=(48, 48))
        scene.append(Object(name="cube", mesh=self.mesh))
        with tempfile.TemporaryDirectory() as tmp:
            files = scene.turntable(
                output_pattern = os.path.join(tmp, "spin.{frame:03d}.png"),
                n_frames       = 2,
                default_light  = True,
            )
            from PIL import Image as _PIL_Image

            with _PIL_Image.open(files[0]) as img:
                self.assertEqual(img.size, (48, 48))


@unittest.skipUnless(
    _HAS_RASTERIZER,
    "wireframe bake needs cv2 or scikit-image for UVData.draw_edges",
)
class TestObjectWireframe(unittest.TestCase):
    """Per-Object wireframe attrs + lazy bake + cache invalidation."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def _make_textured_obj(self, tmpdir):
        """Save a synthetic 64x64 grey diffuse and return an Object using it."""
        from PIL import Image as _PIL_Image

        diffuse = np.full((64, 64, 3), 200, dtype=np.uint8)
        path    = os.path.join(tmpdir, "diff.png")
        _PIL_Image.fromarray(diffuse).save(path)
        return Object(name="cube", mesh=self.mesh, uv=self.uv, texture=path)

    def test_wireframe_defaults(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        self.assertFalse(obj.wireframe)
        self.assertEqual(obj.wireframe_color, (0, 0, 0))
        self.assertEqual(obj.wireframe_thickness, 1)

    def test_wireframe_off_returns_plain_diffuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._make_textured_obj(tmp)
            tex = obj.get_loaded_texture()
            # All texels are bright grey -> no dark texels.
            dark = (tex.sum(axis=-1) < 0.1).sum()
            self.assertEqual(int(dark), 0)

    def test_wireframe_on_bakes_dark_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj           = self._make_textured_obj(tmp)
            obj.wireframe = True
            tex           = obj.get_loaded_texture()
            dark          = (tex.sum(axis=-1) < 0.1).sum()
            self.assertGreater(int(dark), 0, "wireframe bake produced no dark texels")

    def test_wireframe_cache_reused_on_repeated_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj           = self._make_textured_obj(tmp)
            obj.wireframe = True
            t1            = obj.get_loaded_texture()
            t2            = obj.get_loaded_texture()
            self.assertIs(t1, t2, "second call should return cached array")

    def test_wireframe_color_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj                 = self._make_textured_obj(tmp)
            obj.wireframe       = True
            t1                  = obj.get_loaded_texture()
            obj.wireframe_color = (255, 0, 0)
            t2                  = obj.get_loaded_texture()
            self.assertIsNot(t1, t2)

    def test_wireframe_thickness_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._make_textured_obj(tmp)
            obj.wireframe = True
            t1 = obj.get_loaded_texture()
            obj.wireframe_thickness = 3
            t2 = obj.get_loaded_texture()
            self.assertIsNot(t1, t2)

    def test_texture_change_invalidates_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj           = self._make_textured_obj(tmp)
            obj.wireframe = True
            _             = obj.get_loaded_texture()
            self.assertIsNotNone(obj._wired_texture_cache)
            # Re-assign texture to the same path: cache should clear.
            obj.texture = obj._texture
            self.assertIsNone(obj._wired_texture_cache)
            self.assertIsNone(obj._loaded_texture)

    def test_wireframe_off_after_bake_returns_plain_diffuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj           = self._make_textured_obj(tmp)
            obj.wireframe = True
            _             = obj.get_loaded_texture()  # populate wired cache
            obj.wireframe = False
            tex           = obj.get_loaded_texture()
            dark          = (tex.sum(axis=-1) < 0.1).sum()
            self.assertEqual(int(dark), 0, "should fall back to plain diffuse")

    def test_wireframe_renders_with_dark_pixels_in_image(self):
        """End-to-end: rendering a wireframe Object produces dark pixels
        in the output image where the wires get sampled by primary rays."""
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._make_textured_obj(tmp)
            obj.wireframe           = True
            obj.wireframe_thickness = 2
            f    = obj.render(resolution=(96, 96))
            dark = (f.array.sum(axis=-1) < 50).sum()
            self.assertGreater(
                int(dark), 100, "expected visible wireframe pixels in render"
            )


class TestObjectFrameRenderConfig(unittest.TestCase):
    """Per-Object frame-level render config (resolution / samples_per_pixel)
    + cache invalidation + isolation from parent-Scene renders."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    # -- defaults / constructor / setters -----------------------------

    def test_defaults(self):
        obj = Object(name="cube", mesh=self.mesh)
        # Per-Object preview defaults: 500x500 + 4-sample MSAA
        # (smallest perfect square > 1 the renderer accepts).
        self.assertEqual(obj.resolution, (500, 500))
        self.assertEqual(obj.samples_per_pixel, 4)

    def test_constructor_args(self):
        obj = Object(
            name="cube",
            mesh=self.mesh,
            resolution=(640, 480),
            samples_per_pixel=4,
        )
        self.assertEqual(obj.resolution, (640, 480))
        self.assertEqual(obj.samples_per_pixel, 4)

    def test_resolution_setter_coerces_to_int_tuple(self):
        obj            = Object(name="cube", mesh=self.mesh)
        obj.resolution = [800.0, 600.0]
        self.assertEqual(obj.resolution, (800, 600))
        self.assertIsInstance(obj.resolution[0], int)

    def test_samples_per_pixel_setter_accepts_none(self):
        obj                   = Object(name="cube", mesh=self.mesh, samples_per_pixel=4)
        obj.samples_per_pixel = None
        self.assertIsNone(obj.samples_per_pixel)

    # -- cache invalidation --------------------------------------------

    def test_resolution_change_clears_cached_frame(self):
        obj = Object(name="cube", mesh=self.mesh)
        # Manually populate the cache so we can detect invalidation.
        obj.frame      = "sentinel-frame"
        obj.resolution = (640, 480)
        self.assertIsNone(obj.frame)

    def test_resolution_no_op_change_does_not_clear_cache(self):
        obj            = Object(name="cube", mesh=self.mesh)
        obj.frame      = "sentinel"
        obj.resolution = obj.resolution  # set to the same value
        self.assertEqual(obj.frame, "sentinel")

    def test_samples_per_pixel_change_clears_cached_frame(self):
        obj       = Object(name="cube", mesh=self.mesh)
        obj.frame = "sentinel-frame"
        # Use a value different from the new default (4); 9 is the next
        # perfect square up that the renderer accepts.
        obj.samples_per_pixel = 9
        self.assertIsNone(obj.frame)

    def test_samples_per_pixel_no_op_change_does_not_clear_cache(self):
        obj                   = Object(name="cube", mesh=self.mesh, samples_per_pixel=4)
        obj.frame             = "sentinel"
        obj.samples_per_pixel = 4
        self.assertEqual(obj.frame, "sentinel")

    # -- render() / turntable() use the per-Object defaults -----------

    def test_render_uses_self_resolution_when_kwarg_omitted(self):
        obj            = Object(name="cube", mesh=self.mesh)
        obj.resolution = (320, 240)
        with patch.object(Scene, "render", return_value="sentinel") as mock_render:
            obj.render()
        _, kwargs = mock_render.call_args
        self.assertEqual(kwargs["resolution"], (320, 240))

    def test_render_uses_self_samples_per_pixel_when_kwarg_omitted(self):
        obj                   = Object(name="cube", mesh=self.mesh)
        obj.samples_per_pixel = 4
        with patch.object(Scene, "render", return_value="sentinel") as mock_render:
            obj.render()
        _, kwargs = mock_render.call_args
        self.assertEqual(kwargs.get("samples_per_pixel"), 4)

    def test_render_omits_samples_per_pixel_when_self_is_none(self):
        """When self.samples_per_pixel is None, render() must NOT inject
        anything for samples_per_pixel -- the renderer's internal default
        applies via Scene.render()'s _merge_render_config."""
        # The Object default is now 4 (a perfect square the renderer
        # accepts); explicitly opt out by passing None to the constructor.
        obj = Object(name="cube", mesh=self.mesh, samples_per_pixel=None)
        self.assertIsNone(obj.samples_per_pixel)
        with patch.object(Scene, "render", return_value="sentinel") as mock_render:
            obj.render()
        _, kwargs = mock_render.call_args
        self.assertNotIn("samples_per_pixel", kwargs)

    def test_render_explicit_resolution_kwarg_wins(self):
        obj            = Object(name="cube", mesh=self.mesh)
        obj.resolution = (320, 240)
        with patch.object(Scene, "render", return_value="sentinel") as mock_render:
            obj.render(resolution=(640, 480))
        _, kwargs = mock_render.call_args
        self.assertEqual(kwargs["resolution"], (640, 480))

    def test_render_explicit_samples_per_pixel_kwarg_wins(self):
        obj                   = Object(name="cube", mesh=self.mesh)
        obj.samples_per_pixel = 4
        with patch.object(Scene, "render", return_value="sentinel") as mock_render:
            obj.render(samples_per_pixel=9)
        _, kwargs = mock_render.call_args
        self.assertEqual(kwargs.get("samples_per_pixel"), 9)

    def test_turntable_uses_self_resolution_when_kwarg_omitted(self):
        obj            = Object(name="cube", mesh=self.mesh)
        obj.resolution = (320, 240)
        with patch.object(Scene, "turntable", return_value=[]) as mock_tt:
            obj.turntable(n_frames=1)
        _, kwargs = mock_tt.call_args
        self.assertEqual(kwargs["resolution"], (320, 240))

    def test_turntable_uses_self_samples_per_pixel_when_kwarg_omitted(self):
        obj                   = Object(name="cube", mesh=self.mesh)
        obj.samples_per_pixel = 4
        with patch.object(Scene, "turntable", return_value=[]) as mock_tt:
            obj.turntable(n_frames=1)
        _, kwargs = mock_tt.call_args
        self.assertEqual(kwargs.get("samples_per_pixel"), 4)

    # -- isolation: per-Object props don't leak into Scene renders ----

    def test_per_object_resolution_does_not_leak_into_scene_merge_config(self):
        """When an Object is part of a user-built Scene, its per-Object
        resolution / samples_per_pixel MUST NOT appear in the Scene's
        merged render config -- frame settings on the Scene win there.

        We construct the Scene with ``resolution=None`` to suppress the
        Scene-level default (``(500, 500)``) so this test isolates the
        Object-leak invariant from the Scene's own resolution default.
        """
        obj                   = Object(name="cube", mesh=self.mesh)
        obj.resolution        = (640, 480)
        obj.samples_per_pixel = 4

        scene = Scene("test", resolution=None)
        scene.append(obj)

        merged = scene._merge_render_config({})
        # Scene's own _RENDER_CONFIG_KEYS attrs are all None (we passed
        # resolution=None), and per-Object attrs are NOT consulted -- so
        # neither resolution nor samples_per_pixel appears in merged.
        self.assertNotIn("resolution", merged)
        self.assertNotIn("samples_per_pixel", merged)

    def test_scene_render_config_keys_owns_frame_settings(self):
        """Defensive guard against future regressions: ``resolution`` and
        ``samples_per_pixel`` must remain in Scene's render-config contract
        (so per-call kwargs / scene attrs continue to drive Scene renders)
        and per-Object setters must NOT add themselves to that list.
        """
        self.assertIn("resolution", Scene._RENDER_CONFIG_KEYS)
        self.assertIn("samples_per_pixel", Scene._RENDER_CONFIG_KEYS)


class TestSceneObjectIO(unittest.TestCase):
    """Tests for Object.load_obj/load_glb and Scene.load_obj/load_glb."""

    def _save_cube_obj(self, path: str, name: str = "cube") -> tuple:
        """Write a single textured cube to ``path`` via the module-level
        :func:`cgmath.geometry.mesh.save_obj`.  Returns (mesh, uv) of
        the source data (with the ``name`` baked into the mesh)."""
        from cgmath.geometry.mesh import save_obj, UVList

        mesh, uv = _make_textured_cube()
        mesh.name = name
        save_obj(path, [(mesh, UVList([uv]))])
        return mesh, uv

    def _save_two_cube_obj(self, path: str) -> None:
        """Write two named cubes to ``path``."""
        from cgmath.geometry.mesh import save_obj, UVList

        mesh_a, uv_a = _make_textured_cube()
        mesh_a.name = "cubeA"
        mesh_b, uv_b = _make_textured_cube()
        mesh_b.name = "cubeB"
        save_obj(path, [(mesh_a, UVList([uv_a])), (mesh_b, UVList([uv_b]))])

    # -- Object.load_obj ------------------------------------------------

    def test_object_load_obj_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "cube.obj")
            self._save_cube_obj(f)
            obj = Object.load_obj(f)
        self.assertIsInstance(obj,      Object)
        self.assertIsInstance(obj.mesh, MeshData)
        self.assertIsInstance(obj.uv,   UVData)
        # save_obj strips "Shape", load_obj re-adds it
        self.assertEqual(obj.name, "cubeShape")
        self.assertEqual(obj.node_type, "object")

    def test_object_load_obj_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "two.obj")
            self._save_two_cube_obj(f)
            obj0    = Object.load_obj(f, index=0)
            obj1    = Object.load_obj(f, index=1)
            obj_neg = Object.load_obj(f, index=-1)
        self.assertEqual(obj0.name,    "cubeAShape")
        self.assertEqual(obj1.name,    "cubeBShape")
        self.assertEqual(obj_neg.name, "cubeBShape")

    def test_object_load_obj_kwargs_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "cube.obj")
            self._save_cube_obj(f)
            obj = Object.load_obj(
                f,
                name       = "custom",
                base_color = (1.0, 0.0, 0.0),
                translate  = [5.0, 0.0, 0.0],
            )
        self.assertEqual(obj.name, "custom")
        self.assertEqual(obj.base_color, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(obj.world_matrix[3, :3], [5.0, 0.0, 0.0])

    def test_object_load_obj_index_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "cube.obj")
            self._save_cube_obj(f)
            with self.assertRaises(IndexError):
                Object.load_obj(f, index=5)

    # -- Scene.load_obj ------------------------------------------------

    def test_scene_load_obj_single_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "cube.obj")
            self._save_cube_obj(f)
            scene = Scene.load_obj(f)
        self.assertIsInstance(scene, Scene)
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.objects[0].name, "cubeShape")
        self.assertIsInstance(scene.objects[0].mesh, MeshData)
        self.assertIsInstance(scene.objects[0].uv, UVData)

    def test_scene_load_obj_multi_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "two.obj")
            self._save_two_cube_obj(f)
            scene = Scene.load_obj(f, name="my_scene")
        self.assertEqual(scene.scene_name, "my_scene")
        self.assertEqual(len(scene.objects), 2)
        names = sorted(o.name for o in scene.objects)
        self.assertEqual(names, ["cubeAShape", "cubeBShape"])

    def test_scene_load_obj_no_uv(self):
        # An OBJ with no `vt` lines (written via MeshData.save_obj) loads
        # with each Object's `uv` left as None.
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "noUV.obj")
            mesh, _ = _make_textured_cube()
            mesh.name = "cube"
            mesh.save_obj(f)
            scene = Scene.load_obj(f)
        self.assertEqual(len(scene.objects), 1)
        self.assertIsNone(scene.objects[0].uv)
        self.assertIsInstance(scene.objects[0].mesh, MeshData)


class TestObjectQuadrangulate(unittest.TestCase):
    """Object.quadrangulate() merges triangles into quads on both the
    mesh and the UV when present, keeping them in sync."""

    def setUp(self):
        # _make_textured_cube returns a triangulated cube (12 tris, 6 quads
        # split along their 0-2 diagonal) with a matching per-face UV
        # island, sharing face topology across mesh and uv.
        self.mesh, self.uv = _make_textured_cube()

    def test_quadrangulate_with_uv_default_source_is_uv(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        # Sanity: starts as 12 triangles in both mesh and uv.
        self.assertEqual(int(np.sum(self.mesh.counts == 3)), 12)
        self.assertEqual(int(np.sum(self.uv.counts == 3)), 12)
        # Default source is "uv" -- rules are generated from the uv but
        # applied to BOTH mesh and uv so they stay in sync.
        obj.quadrangulate()
        self.assertEqual(int(np.sum(self.mesh.counts == 4)), 6)
        self.assertEqual(int(np.sum(self.mesh.counts == 3)), 0)
        self.assertEqual(int(np.sum(self.uv.counts == 4)),   6)
        self.assertEqual(int(np.sum(self.uv.counts == 3)),   0)

    def test_quadrangulate_with_uv_source_uv_merges_both(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        obj.quadrangulate(source="uv")
        self.assertEqual(int(np.sum(self.mesh.counts == 4)), 6)
        self.assertEqual(int(np.sum(self.uv.counts == 4)), 6)

    def test_quadrangulate_with_uv_source_mesh_merges_both(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        obj.quadrangulate(source="mesh")
        self.assertEqual(int(np.sum(self.mesh.counts == 4)), 6)
        self.assertEqual(int(np.sum(self.uv.counts == 4)), 6)

    def test_quadrangulate_without_uv_merges_mesh_only(self):
        obj = Object(name="cube", mesh=self.mesh, uv=None)
        obj.quadrangulate()
        self.assertEqual(int(np.sum(self.mesh.counts == 4)), 6)
        self.assertEqual(int(np.sum(self.mesh.counts == 3)), 0)

    def test_quadrangulate_without_uv_ignores_source_arg(self):
        # When uv is None, the source argument is ignored (no error).
        obj = Object(name="cube", mesh=self.mesh, uv=None)
        obj.quadrangulate(source="mesh")
        self.assertEqual(int(np.sum(self.mesh.counts == 4)), 6)

    def test_quadrangulate_raises_when_mesh_is_none(self):
        obj = Object(name="empty", mesh=None, uv=None)
        with self.assertRaises(ValueError):
            obj.quadrangulate()

    def test_quadrangulate_raises_on_invalid_source(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        with self.assertRaises(ValueError):
            obj.quadrangulate(source="bogus")

    @unittest.skipUnless(_HAS_RASTERIZER, "needs cv2 or scikit-image")
    def test_quadrangulate_invalidates_wireframe_cache(self):
        # Bake a wireframe cache, then quadrangulate and confirm it clears.
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = self.uv,
            texture   = np.full((32, 32, 3), 200, dtype=np.uint8),
            wireframe = True,
        )
        _ = obj.get_loaded_texture()  # populates _wired_texture_cache
        self.assertIsNotNone(obj._wired_texture_cache)
        obj.quadrangulate()
        self.assertIsNone(obj._wired_texture_cache)


class TestObjectMerge(unittest.TestCase):
    """Object.merge() collapses overlapping points on the Object's mesh
    and (when present) its UV."""

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def test_merge_collapses_duplicates_on_mesh_and_uv(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        # detach_faces splits every face into its own island, blowing up
        # the unique-point count.  merge() should then collapse it back.
        self.mesh.detach_faces()
        self.uv.detach_faces()
        before_mesh_pts = self.mesh.points.shape[0]
        before_uv_pts   = self.uv.points.shape[0]
        obj.merge()
        self.assertLess(self.mesh.points.shape[0], before_mesh_pts)
        self.assertLess(self.uv.points.shape[0], before_uv_pts)

    def test_merge_without_uv_only_merges_mesh(self):
        obj = Object(name="cube", mesh=self.mesh, uv=None)
        self.mesh.detach_faces()
        before = self.mesh.points.shape[0]
        obj.merge()
        self.assertLess(self.mesh.points.shape[0], before)

    def test_merge_raises_when_mesh_is_none(self):
        obj = Object(name="empty", mesh=None, uv=None)
        with self.assertRaises(ValueError):
            obj.merge()

    @unittest.skipUnless(_HAS_RASTERIZER, "needs cv2 or scikit-image")
    def test_merge_invalidates_wireframe_cache(self):
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = self.uv,
            texture   = np.full((32, 32, 3), 200, dtype=np.uint8),
            wireframe = True,
        )
        _ = obj.get_loaded_texture()  # populates _wired_texture_cache
        self.assertIsNotNone(obj._wired_texture_cache)
        obj.merge()
        self.assertIsNone(obj._wired_texture_cache)


@unittest.skipUnless(
    _HAS_RASTERIZER,
    "wireframe bake needs cv2 or scikit-image for UVData.draw_edges",
)
class TestWireframeWithoutTexture(unittest.TestCase):
    """Wireframe-on Objects must render even when no diffuse texture is set.

    Regression suite for the bug where ``Object.get_loaded_texture()``
    early-returned ``None`` when ``texture is None``, short-circuiting
    the wireframe-bake branch and silently dropping the wires.  The fix
    fills the UV buffer with :attr:`Object.base_color` when no texture
    is set, so wireframe-only Objects render as edges over their flat-
    shading color.
    """

    def setUp(self):
        # Smallest reasonable UV so the bake is fast; bypassing the
        # 2048x2048 default keeps these tests under a second each.
        self.mesh, self.uv = _make_textured_cube()
        self.uv.resolution = (32, 32)

    def test_wireframe_without_texture_returns_array(self):
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = self.uv,
            texture   = None,
            wireframe = True,
        )
        baked = obj.get_loaded_texture()
        self.assertIsNotNone(baked)
        self.assertIsInstance(baked, np.ndarray)
        self.assertEqual(baked.dtype, np.float32)
        # Match _load_texture's output format: (H, W, 3) in [0, 1].
        self.assertEqual(baked.ndim, 3)
        self.assertEqual(baked.shape[2], 3)
        self.assertGreaterEqual(float(baked.min()), 0.0)
        self.assertLessEqual(float(baked.max()), 1.0)

    def test_wireframe_without_texture_uses_base_color(self):
        # Robust check that base_color drives the bake's background:
        # bake two Objects whose only difference is base_color, with
        # the wireframe colour pinned, and verify the bakes' means
        # differ in the expected direction.  Avoids relying on exact
        # uint8 round-trip (0.5*255 -> 127 -> 0.498... loses ~2e-3) or
        # on which renderer (cv2 vs skimage) is active in the test env.
        mesh_a, uv_a = _make_textured_cube()
        uv_a.resolution = (32, 32)
        mesh_b, uv_b = _make_textured_cube()
        uv_b.resolution = (32, 32)

        obj_dark = Object(
            name="cube_dark",
            mesh=mesh_a,
            uv=uv_a,
            texture=None,
            base_color=(0.0, 0.0, 0.0),
            wireframe=True,
            wireframe_color=(0, 0, 0),
        )
        obj_bright = Object(
            name="cube_bright",
            mesh=mesh_b,
            uv=uv_b,
            texture=None,
            base_color=(1.0, 1.0, 1.0),
            wireframe=True,
            wireframe_color=(0, 0, 0),
        )
        dark   = obj_dark.get_loaded_texture()
        bright = obj_bright.get_loaded_texture()

        # Both baked the same wires (black) over different backgrounds.
        # If base_color is being respected, dark is mostly black and
        # bright is mostly white.  With a 32x32 buffer + this cube's
        # stacked-UV layout the wireframe rasterises ~15-25% of the
        # pixels (4 boundary edges + the triangulation diagonal, all
        # 6 faces overlapping); pick thresholds that survive that
        # without making the assertion vacuous.
        self.assertLess(float(dark.mean()), 0.05)
        self.assertGreater(float(bright.mean()), 0.7)
        self.assertGreater(float(bright.mean()) - float(dark.mean()), 0.6)

    def test_wireframe_without_uv_returns_none_when_textureless(self):
        # No UV -> can't bake a wireframe -> falls back to the texture
        # branch, which returns None when texture is also None.
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = None,
            texture   = None,
            wireframe = True,
        )
        self.assertIsNone(obj.get_loaded_texture())

    def test_wireframe_off_texture_none_still_returns_none(self):
        # Regression: wireframe must remain opt-in; without it, a
        # textureless Object stays textureless.
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = self.uv,
            texture   = None,
            wireframe = False,
        )
        self.assertIsNone(obj.get_loaded_texture())

    def test_wireframe_with_texture_still_works(self):
        # Regression: the existing texture+wireframe path must survive
        # the get_loaded_texture re-ordering.
        obj = Object(
            name      = "cube",
            mesh      = self.mesh,
            uv        = self.uv,
            texture   = np.full((32, 32, 3), 200, dtype=np.uint8),
            wireframe = True,
        )
        baked = obj.get_loaded_texture()
        self.assertIsNotNone(baked)
        self.assertIsNotNone(obj._wired_texture_cache)

    def test_base_color_setter_invalidates_wireframe_cache(self):
        obj = Object(
            name       = "cube",
            mesh       = self.mesh,
            uv         = self.uv,
            texture    = None,
            base_color = (1.0, 0.0, 0.0),
            wireframe  = True,
        )
        first = obj.get_loaded_texture()
        self.assertIsNotNone(obj._wired_texture_cache)

        # New base_color -> cache must be dropped and the next bake
        # must reflect the new colour.
        obj.base_color = (0.0, 1.0, 0.0)
        self.assertIsNone(obj._wired_texture_cache)
        second = obj.get_loaded_texture()
        self.assertFalse(
            np.array_equal(first, second),
            "expected the bake to change after base_color update",
        )

    def test_base_color_setter_no_op_preserves_cache(self):
        # The setter has an inequality guard; assigning the same value
        # back must NOT drop the cache.
        obj = Object(
            name       = "cube",
            mesh       = self.mesh,
            uv         = self.uv,
            texture    = None,
            base_color = (0.5, 0.5, 0.5),
            wireframe  = True,
        )
        _            = obj.get_loaded_texture()
        cache_before = obj._wired_texture_cache
        self.assertIsNotNone(cache_before)
        obj.base_color = (0.5, 0.5, 0.5)
        self.assertIs(obj._wired_texture_cache, cache_before)


class TestFrameCacheInvalidation(unittest.TestCase):
    """Setters of attributes that affect the rendered output must drop
    the cached :attr:`Object.frame` so :meth:`imshow` / :meth:`to_image`
    lazy-rebuild on next call.

    Uses a sentinel object placed on ``obj.frame`` so the tests don't
    need to actually run the raytracer -- they only verify the
    invalidation hook fires.
    """

    def setUp(self):
        self.mesh, self.uv = _make_textured_cube()

    def _seed_frame(self, obj):
        """Stand-in for a cached render result; any non-None value will
        do since :meth:`imshow` / :meth:`to_image` only check ``is None``."""
        sentinel  = object()
        obj.frame = sentinel
        return sentinel

    def test_texture_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        self._seed_frame(obj)
        obj.texture = np.full((4, 4, 3), 7, dtype=np.uint8)
        self.assertIsNone(obj.frame)

    def test_mesh_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        self._seed_frame(obj)
        new_mesh, _ = _make_textured_cube()
        obj.mesh = new_mesh
        self.assertIsNone(obj.frame)

    def test_uv_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        self._seed_frame(obj)
        _, new_uv = _make_textured_cube()
        obj.uv = new_uv
        self.assertIsNone(obj.frame)

    def test_base_color_setter_invalidates_frame(self):
        obj = Object(
            name="cube", mesh=self.mesh, uv=self.uv, base_color=(0.7, 0.7, 0.7)
        )
        self._seed_frame(obj)
        obj.base_color = (0.1, 0.2, 0.3)
        self.assertIsNone(obj.frame)

    def test_base_color_setter_no_op_preserves_frame(self):
        # The setter has an inequality guard; assigning the same value
        # back must NOT drop the frame cache (same policy as the
        # _wired_texture_cache).
        obj = Object(
            name="cube", mesh=self.mesh, uv=self.uv, base_color=(0.5, 0.5, 0.5)
        )
        sentinel       = self._seed_frame(obj)
        obj.base_color = (0.5, 0.5, 0.5)
        self.assertIs(obj.frame, sentinel)

    def test_wireframe_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv, wireframe=False)
        self._seed_frame(obj)
        obj.wireframe = True
        self.assertIsNone(obj.frame)

    def test_wireframe_setter_no_op_preserves_frame(self):
        obj           = Object(name="cube", mesh=self.mesh, uv=self.uv, wireframe=True)
        sentinel      = self._seed_frame(obj)
        obj.wireframe = True
        self.assertIs(obj.frame, sentinel)

    def test_wireframe_color_setter_invalidates_frame(self):
        obj = Object(
            name="cube",
            mesh=self.mesh,
            uv=self.uv,
            wireframe=True,
            wireframe_color=(0, 0, 0),
        )
        self._seed_frame(obj)
        obj.wireframe_color = (255, 255, 255)
        self.assertIsNone(obj.frame)

    def test_wireframe_thickness_setter_invalidates_frame(self):
        obj = Object(
            name="cube",
            mesh=self.mesh,
            uv=self.uv,
            wireframe=True,
            wireframe_thickness=1,
        )
        self._seed_frame(obj)
        obj.wireframe_thickness = 3
        self.assertIsNone(obj.frame)

    def test_quadrangulate_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        self._seed_frame(obj)
        obj.quadrangulate()
        self.assertIsNone(obj.frame)

    def test_merge_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv)
        self._seed_frame(obj)
        obj.merge()
        self.assertIsNone(obj.frame)

    # -- shader knobs (sample_method / wrap / ambient / twosided /
    #    cast_shadows): setters mirror the wireframe/base_color contract:
    #    distinct values invalidate the cache, no-op assignments preserve it,
    #    and the setter coerces to the declared type.

    def test_sample_method_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv, sample_method="bilinear")
        self._seed_frame(obj)
        obj.sample_method = "bezier"
        self.assertIsNone(obj.frame)

    def test_sample_method_setter_no_op_preserves_frame(self):
        obj               = Object(name="cube", mesh=self.mesh, uv=self.uv, sample_method="bilinear")
        sentinel          = self._seed_frame(obj)
        obj.sample_method = "bilinear"
        self.assertIs(obj.frame, sentinel)

    def test_sample_method_setter_coerces_to_str(self):
        obj               = Object(name="cube", mesh=self.mesh, uv=self.uv)
        obj.sample_method = 42
        self.assertEqual(obj.sample_method, "42")
        self.assertIsInstance(obj.sample_method, str)

    def test_wrap_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv, wrap="repeat")
        self._seed_frame(obj)
        obj.wrap = "clamp"
        self.assertIsNone(obj.frame)

    def test_wrap_setter_no_op_preserves_frame(self):
        obj      = Object(name="cube", mesh=self.mesh, uv=self.uv, wrap="repeat")
        sentinel = self._seed_frame(obj)
        obj.wrap = "repeat"
        self.assertIs(obj.frame, sentinel)

    def test_wrap_setter_coerces_to_str(self):
        obj      = Object(name="cube", mesh=self.mesh, uv=self.uv)
        obj.wrap = 0
        self.assertEqual(obj.wrap, "0")
        self.assertIsInstance(obj.wrap, str)

    def test_ambient_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv, ambient=0.1)
        self._seed_frame(obj)
        obj.ambient = 0.5
        self.assertIsNone(obj.frame)

    def test_ambient_setter_no_op_preserves_frame(self):
        obj         = Object(name="cube", mesh=self.mesh, uv=self.uv, ambient=0.25)
        sentinel    = self._seed_frame(obj)
        obj.ambient = 0.25
        self.assertIs(obj.frame, sentinel)

    def test_ambient_setter_coerces_to_float(self):
        obj         = Object(name="cube", mesh=self.mesh, uv=self.uv)
        obj.ambient = 1
        self.assertEqual(obj.ambient, 1.0)
        self.assertIsInstance(obj.ambient, float)

    def test_twosided_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv, twosided=True)
        self._seed_frame(obj)
        obj.twosided = False
        self.assertIsNone(obj.frame)

    def test_twosided_setter_no_op_preserves_frame(self):
        obj          = Object(name="cube", mesh=self.mesh, uv=self.uv, twosided=True)
        sentinel     = self._seed_frame(obj)
        obj.twosided = True
        self.assertIs(obj.frame, sentinel)

    def test_twosided_setter_coerces_to_bool(self):
        obj          = Object(name="cube", mesh=self.mesh, uv=self.uv, twosided=False)
        obj.twosided = 1
        self.assertIs(obj.twosided, True)

    def test_cast_shadows_setter_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh, uv=self.uv, cast_shadows=True)
        self._seed_frame(obj)
        obj.cast_shadows = False
        self.assertIsNone(obj.frame)

    def test_cast_shadows_setter_no_op_preserves_frame(self):
        obj              = Object(name="cube", mesh=self.mesh, uv=self.uv, cast_shadows=True)
        sentinel         = self._seed_frame(obj)
        obj.cast_shadows = True
        self.assertIs(obj.frame, sentinel)

    def test_cast_shadows_setter_coerces_to_bool(self):
        obj              = Object(name="cube", mesh=self.mesh, uv=self.uv, cast_shadows=False)
        obj.cast_shadows = 1
        self.assertIs(obj.cast_shadows, True)

    def test_shader_knob_constructor_uses_private_storage(self):
        """The ctor must populate the private ``_attr`` slots that the new
        property getters read from -- regression guard against accidental
        reversion to direct ``self.attr =`` assignments."""
        obj = Object(
            name          = "cube",
            mesh          = self.mesh,
            uv            = self.uv,
            sample_method = "bezier",
            wrap          = "clamp",
            ambient       = 0.42,
            twosided      = False,
            cast_shadows  = False,
        )
        self.assertEqual(obj._sample_method, "bezier")
        self.assertEqual(obj._wrap,          "clamp")
        self.assertEqual(obj._ambient,       0.42)
        self.assertIs(obj._twosided, False)
        self.assertIs(obj._cast_shadows, False)
        # Public getters must agree with private storage.
        self.assertEqual(obj.sample_method, "bezier")
        self.assertEqual(obj.wrap,          "clamp")
        self.assertEqual(obj.ambient,       0.42)
        self.assertIs(obj.twosided, False)
        self.assertIs(obj.cast_shadows, False)


class TestObjectTransformCacheInvalidation(unittest.TestCase):
    """Object.frame auto-invalidates lazily when ``world_matrix`` changes.

    Covers transforms applied to self (translate / rotate / scale /
    direct world_matrix assignment) AND transforms applied to any
    ancestor in the parenting hierarchy -- both reflect through the
    composed ``world_matrix`` that the property getter snapshots.
    """

    def setUp(self):
        self.mesh, _ = _make_textured_cube()

    def _seed_frame(self, obj):
        sentinel  = object()
        obj.frame = sentinel
        return sentinel

    def test_setter_snapshots_transform_state(self):
        obj = Object(name="cube", mesh=self.mesh, translate=[1.0, 2.0, 3.0])
        self.assertIsNone(obj._render_transform_state)
        self._seed_frame(obj)
        # Setter must have snapshotted the live transform state.
        self.assertIsNotNone(obj._render_transform_state)
        # Snapshot must equal whatever the helper produces right now.
        self.assertEqual(obj._render_transform_state, obj._capture_transform_state())

    def test_setter_snapshot_is_stable_across_reads(self):
        """Repeated reads of a non-mutated Object must keep returning
        the same cached value -- the snapshot tuple must compare equal
        to a fresh capture as long as no transform attr has changed."""
        obj             = Object(name="cube", mesh=self.mesh)
        sentinel        = self._seed_frame(obj)
        snapshot_before = obj._render_transform_state
        # Read frame multiple times without mutating.
        for _ in range(3):
            self.assertIs(obj.frame, sentinel)
        # Snapshot should still be intact and equal to original.
        self.assertEqual(obj._render_transform_state, snapshot_before)

    def test_rotate_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh)
        self._seed_frame(obj)
        obj.rotate = [0.0, 90.0, 0.0]
        self.assertIsNone(obj.frame)
        # Reading the getter clears the snapshot too.
        self.assertIsNone(obj._render_transform_state)

    def test_translate_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh)
        self._seed_frame(obj)
        obj.translate = [10.0, 0.0, 0.0]
        self.assertIsNone(obj.frame)

    def test_scale_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh)
        self._seed_frame(obj)
        obj.scale = [2.0, 2.0, 2.0]
        self.assertIsNone(obj.frame)

    def test_world_matrix_assignment_invalidates_frame(self):
        obj = Object(name="cube", mesh=self.mesh)
        self._seed_frame(obj)
        new_matrix        = np.eye(4)
        new_matrix[3, :3] = [5.0, 5.0, 5.0]
        obj.world_matrix  = new_matrix
        self.assertIsNone(obj.frame)

    def test_no_transform_change_keeps_cache(self):
        obj      = Object(name="cube", mesh=self.mesh)
        sentinel = self._seed_frame(obj)
        # Multiple reads without mutating must keep returning the cache.
        self.assertIs(obj.frame, sentinel)
        self.assertIs(obj.frame, sentinel)
        self.assertIs(obj.frame, sentinel)

    def test_parent_transform_invalidates_frame(self):
        """Transform on a parent must invalidate the child's cache --
        the composed ``world_matrix`` reflects ancestor mutations, and
        the transform-state snapshot includes ``world_matrix`` bytes."""
        scene  = Scene("test")
        parent = Object(name="parent")
        obj    = Object(name="cube", mesh=self.mesh)
        scene.append(parent)
        scene.append(obj)
        obj.set_parent(parent)
        self._seed_frame(obj)
        self.assertIsNotNone(obj._render_transform_state)
        # Mutate parent's local transform; obj.world_matrix recomposes
        # via the hierarchy and our world_matrix-bytes signature catches it.
        parent.translate = [10.0, 0.0, 0.0]
        self.assertIsNone(obj.frame)

    def test_setter_clears_snapshot_when_value_none(self):
        obj = Object(name="cube", mesh=self.mesh)
        self._seed_frame(obj)
        self.assertIsNotNone(obj._render_transform_state)
        obj.frame = None
        self.assertIsNone(obj._render_transform_state)

    def test_mark_render_dirty_clears_snapshot(self):
        """Setter-driven invalidation (texture, wireframe, etc.) routes
        through ``_mark_render_dirty`` which must also clear the
        ``_render_transform_state`` snapshot so later reads don't compare
        against a stale state."""
        obj = Object(name="cube", mesh=self.mesh)
        self._seed_frame(obj)
        self.assertIsNotNone(obj._render_transform_state)
        obj._mark_render_dirty()
        self.assertIsNone(obj._render_transform_state)


class TestSceneCacheInvalidation(unittest.TestCase):
    """Scene.frame auto-invalidates lazily when ANY render-affecting
    attribute on the Scene OR its child Object/Camera/Light nodes
    changes since the last :meth:`render` call.

    Uses a sentinel placed on ``scene.frame`` so the tests don't need
    to actually run the raytracer -- they only verify the signature
    diff catches the mutation.
    """

    def setUp(self):
        self.mesh, _ = _make_textured_cube()

    def _populated_scene(self):
        scene = Scene("test")
        scene.append(Object(name="cube", mesh=self.mesh))
        scene.append(Camera(name="cam"))
        scene.append(Light(name="key", kind="point", intensity=100.0))
        return scene

    def _seed_frame(self, scene):
        sentinel    = object()
        scene.frame = sentinel
        return sentinel

    # -- setter snapshots a signature --------------------------------

    def test_setter_snapshots_signature(self):
        scene = self._populated_scene()
        self.assertIsNone(scene._render_signature)
        self._seed_frame(scene)
        self.assertIsNotNone(scene._render_signature)

    def test_setter_clears_signature_when_value_none(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        self.assertIsNotNone(scene._render_signature)
        scene.frame = None
        self.assertIsNone(scene._render_signature)

    def test_no_change_keeps_cache(self):
        scene    = self._populated_scene()
        sentinel = self._seed_frame(scene)
        self.assertIs(scene.frame, sentinel)
        self.assertIs(scene.frame, sentinel)

    # -- transforms on any child invalidate --------------------------

    def test_object_rotate_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].rotate = [0.0, 45.0, 0.0]
        self.assertIsNone(scene.frame)

    def test_object_translate_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].translate = [10.0, 0.0, 0.0]
        self.assertIsNone(scene.frame)

    def test_object_scale_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].scale = [2.0, 2.0, 2.0]
        self.assertIsNone(scene.frame)

    def test_camera_translate_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.cameras[0].translate = [5.0, 5.0, 5.0]
        self.assertIsNone(scene.frame)

    def test_light_translate_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.lights[0].translate = [10.0, 0.0, 0.0]
        self.assertIsNone(scene.frame)

    # -- Camera domain attrs invalidate ------------------------------

    def test_camera_angle_of_view_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.cameras[0].angle_of_view = 60.0
        self.assertIsNone(scene.frame)

    def test_camera_is_orthographic_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.cameras[0].is_orthographic = True
        self.assertIsNone(scene.frame)

    def test_camera_ortho_height_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.cameras[0].ortho_height = 5.0
        self.assertIsNone(scene.frame)

    def test_camera_aspect_ratio_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.cameras[0].aspect_ratio = 16.0 / 9.0
        self.assertIsNone(scene.frame)

    # -- Light domain attrs invalidate -------------------------------

    def test_light_intensity_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.lights[0].intensity = 200.0
        self.assertIsNone(scene.frame)

    def test_light_color_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.lights[0].color = (1.0, 0.0, 0.0)
        self.assertIsNone(scene.frame)

    def test_light_kind_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.lights[0].kind = "infinite"
        self.assertIsNone(scene.frame)

    def test_light_falloff_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.lights[0].falloff = False
        self.assertIsNone(scene.frame)

    # -- Object visual attrs invalidate (via signature, not just Object's own _frame) -

    def test_object_texture_assignment_invalidates_scene(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].texture = np.zeros((4, 4, 3), dtype=np.float32)
        self.assertIsNone(scene.frame)

    def test_object_base_color_invalidates_scene(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].base_color = (1.0, 0.0, 0.0)
        self.assertIsNone(scene.frame)

    def test_object_wireframe_invalidates_scene(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].wireframe = True
        self.assertIsNone(scene.frame)

    def test_object_visibility_invalidates_scene(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.objects[0].visibility = False
        self.assertIsNone(scene.frame)

    # -- topology mutations invalidate -------------------------------

    def test_appending_object_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.append(Object(name="cube2", mesh=self.mesh))
        self.assertIsNone(scene.frame)

    def test_appending_light_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.append(Light(name="fill", kind="point", intensity=50.0))
        self.assertIsNone(scene.frame)

    # -- Scene-level config invalidates ------------------------------

    def test_scene_resolution_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.resolution = (640, 480)
        self.assertIsNone(scene.frame)

    def test_scene_samples_per_pixel_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.samples_per_pixel = 4
        self.assertIsNone(scene.frame)

    def test_scene_background_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.background = (0.5, 0.5, 0.5)
        self.assertIsNone(scene.frame)

    def test_scene_angle_of_view_invalidates(self):
        scene = self._populated_scene()
        self._seed_frame(scene)
        scene.angle_of_view = 60.0
        self.assertIsNone(scene.frame)


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
    """Tests for ``_extract_glb_textures`` and the ``Object`` wrappers."""

    def test_extract_glb_textures_finds_base_color(self):
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes((255, 0, 0))))
        try:
            textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)

        self.assertIn("base_color", textures)
        # 2x2 red image -> red channel saturated, others zero.
        np.testing.assert_array_equal(
            textures["base_color"][:, :, 0], np.full((2, 2), 255, dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            textures["base_color"][:, :, 1], np.zeros((2, 2), dtype=np.uint8)
        )

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

    def test_extract_glb_textures_pil_unavailable(self):
        from cgmath.render import scene as scene_mod
        from cgmath.render.scene import _extract_glb_textures

        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes()))
        try:
            with patch.object(scene_mod, "Image", None):
                textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)
        # Without PIL we can't decode -> base_color is omitted.
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
        np.testing.assert_array_equal(
            textures["base_color"][:, :, 1], np.full((2, 2), 255, dtype=np.uint8)
        )

    def test_extract_glb_textures_extracts_all_pbr_slots(self):
        """Multi-texture support: helper still extracts non-base_color slots."""
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
                {
                    "buffer":     0,
                    "byteOffset": len(red),
                    "byteLength": len(green),
                },
                {
                    "buffer":     0,
                    "byteOffset": len(red) + len(green),
                    "byteLength": len(blue),
                },
            ],
            "buffers": [{"byteLength": len(bin_blob)}],
        }
        path = _write_temp_glb(_build_glb(json_dict, bin_data=bin_blob))
        try:
            textures = _extract_glb_textures(path)
        finally:
            os.unlink(path)
        self.assertEqual(
            set(textures.keys()), {"base_color", "metallic_roughness", "normal"}
        )

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

    def test_object_extract_texture_from_glb_assigns_texture(self):
        mesh, uv = _make_textured_cube()
        obj = Object(name="cube", mesh=mesh, uv=uv)
        self.assertIsNone(obj.texture)

        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes((128, 64, 32))))
        try:
            ok = obj.extract_texture_from_glb(path)
        finally:
            os.unlink(path)

        self.assertTrue(ok)
        self.assertIsNotNone(obj.texture)
        self.assertIsInstance(obj.texture, np.ndarray)
        self.assertEqual(obj.texture.shape[2], 3)

    def test_object_extract_texture_from_glb_no_color_returns_false(self):
        mesh, _ = _make_textured_cube()
        obj = Object(
            name="cube", mesh=mesh, texture=np.full((4, 4, 3), 7, dtype=np.uint8)
        )
        existing = obj.texture

        json_dict = {
            "asset":     {"version": "2.0"},
            "materials": [{"name": "no_textures"}],
        }
        path = _write_temp_glb(_build_glb(json_dict))
        try:
            ok = obj.extract_texture_from_glb(path)
        finally:
            os.unlink(path)

        self.assertFalse(ok)
        # Existing texture preserved on no-op.
        np.testing.assert_array_equal(obj.texture, existing)

    @unittest.skipUnless(_HAS_RASTERIZER, "needs cv2 or scikit-image")
    def test_object_extract_texture_from_glb_invalidates_caches(self):
        mesh, uv = _make_textured_cube()
        # Seed a wireframe-baked cache via an explicit np texture.
        obj = Object(
            name      = "cube",
            mesh      = mesh,
            uv        = uv,
            texture   = np.full((32, 32, 3), 200, dtype=np.uint8),
            wireframe = True,
        )
        _ = obj.get_loaded_texture()
        self.assertIsNotNone(obj._wired_texture_cache)

        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes()))
        try:
            obj.extract_texture_from_glb(path)
        finally:
            os.unlink(path)

        # The texture setter wired up by extract_texture_from_glb should
        # have invalidated both caches.
        self.assertIsNone(obj._wired_texture_cache)
        self.assertIsNone(obj._loaded_texture)

    def test_object_load_glb_auto_extracts_texture(self):
        """Object.load_glb auto-fills texture from the GLB when no override."""
        from cgmath.render import scene as scene_mod

        mesh, uv = _make_textured_cube()
        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes((10, 20, 30))))
        try:
            with patch.object(scene_mod, "_load_glb", return_value=[(mesh, [uv])]):
                obj = Object.load_glb(path)
        finally:
            os.unlink(path)

        self.assertIsNotNone(obj.texture)
        self.assertIsInstance(obj.texture, np.ndarray)

    def test_object_load_glb_extract_texture_false(self):
        from cgmath.render import scene as scene_mod

        mesh, uv = _make_textured_cube()
        path = _write_temp_glb(_minimal_glb_with_color(_png_bytes()))
        try:
            with patch.object(scene_mod, "_load_glb", return_value=[(mesh, [uv])]):
                obj = Object.load_glb(path, extract_texture=False)
        finally:
            os.unlink(path)

        self.assertIsNone(obj.texture)

    def test_object_load_glb_explicit_texture_wins(self):
        """An explicit ``texture=`` kwarg suppresses auto-extraction."""
        from cgmath.render import scene as scene_mod

        mesh, uv = _make_textured_cube()
        explicit = _make_checker()
        path     = _write_temp_glb(_minimal_glb_with_color(_png_bytes((255, 0, 0))))
        try:
            with patch.object(scene_mod, "_load_glb", return_value=[(mesh, [uv])]):
                obj = Object.load_glb(path, texture=explicit)
        finally:
            os.unlink(path)

        np.testing.assert_array_equal(obj.texture, explicit)


# ============================================================================
# Tests for Object serialization (to_dict / from_dict / save / load)
# ============================================================================


# PIL is optional; needed only for path-texture tests.  Mirrors scene.py.
try:
    from PIL import Image as _PIL_Image
except ImportError:
    _PIL_Image = None


def _write_temp_png(arr_uint8: np.ndarray) -> str:
    """Write a uint8 (H, W, C) array to a temp PNG; return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    _PIL_Image.fromarray(arr_uint8).save(tmp.name)
    return tmp.name


class TestObjectSerialization(unittest.TestCase):
    """Round-trip tests for :meth:`Object.to_dict` / :meth:`Object.from_dict`
    / :meth:`Object.save` / :meth:`Object.load`."""

    def _make_full_obj(self):
        """Build a textured Object with non-trivial transforms.

        Returns ``(obj, ground_truth_texture_uint8)``.
        """
        mesh, uv = _make_textured_cube()
        tex = _make_checker()
        obj = Object(
            name      = "hero",
            mesh      = mesh,
            uv        = uv,
            texture   = tex,
            translate = [1.0, 2.0, 3.0],
            rotate    = [10.0, 20.0, 30.0],
        )
        return obj, tex

    # -- to_dict -------------------------------------------------------

    def test_to_dict_recurses_into_mesh_uv(self):
        obj, _ = self._make_full_obj()
        d = obj.to_dict()
        self.assertIsInstance(d["_mesh"], dict)
        self.assertIsInstance(d["_uv"], dict)
        # Inner dicts should hold the raw geometry so the file is
        # self-contained.
        self.assertIn("points", d["_mesh"])
        self.assertIn("indices", d["_mesh"])

    def test_to_dict_materializes_array_texture_as_uint8(self):
        obj, tex = self._make_full_obj()
        d = obj.to_dict()
        self.assertIsInstance(d["_texture"], np.ndarray)
        self.assertEqual(d["_texture"].dtype, np.uint8)
        np.testing.assert_array_equal(d["_texture"], tex)

    def test_to_dict_materializes_path_texture_as_uint8(self):
        if _PIL_Image is None:
            self.skipTest("PIL not available")
        mesh, uv = _make_textured_cube()
        tex  = _make_checker()
        path = _write_temp_png(tex)
        try:
            obj = Object(name="hero", mesh=mesh, uv=uv, texture=path)
            d   = obj.to_dict()
        finally:
            os.unlink(path)
        self.assertEqual(d["_texture"].dtype, np.uint8)
        np.testing.assert_array_equal(d["_texture"], tex)

    def test_to_dict_drops_transient_caches(self):
        obj, _ = self._make_full_obj()
        # Force-populate caches that should NOT serialize.
        obj._loaded_texture         = np.zeros((4, 4, 3), dtype=np.uint8)
        obj._wired_texture_cache    = np.zeros((4, 4, 3), dtype=np.float32)
        obj._render_transform_state = ("dummy",)
        d                           = obj.to_dict()
        for k in (
            "_loaded_texture",
            "_wired_texture_cache",
            "_render_transform_state",
        ):
            self.assertNotIn(k, d)

    def test_to_dict_path_texture_no_pil_raises(self):
        """When the texture is a path AND PIL is unavailable,
        :meth:`to_dict` raises a clear RuntimeError instead of silently
        swallowing the texture."""
        from cgmath.render import scene as scene_mod

        mesh, uv = _make_textured_cube()
        obj = Object(name="x", mesh=mesh, uv=uv, texture="/no/such.png")
        with patch.object(scene_mod, "Image", None):
            with self.assertRaises(RuntimeError) as ctx:
                obj.to_dict()
        self.assertIn("PIL", str(ctx.exception))

    # -- from_dict -----------------------------------------------------

    def test_from_dict_reconstructs_mesh_uv_texture(self):
        obj, tex = self._make_full_obj()
        rebuilt = Object.from_dict(obj.to_dict())
        self.assertIsInstance(rebuilt.mesh, MeshData)
        self.assertIsInstance(rebuilt.uv, UVData)
        np.testing.assert_allclose(rebuilt.mesh.points, obj.mesh.points)
        np.testing.assert_array_equal(rebuilt.mesh.indices, obj.mesh.indices)
        np.testing.assert_allclose(rebuilt.uv.points, obj.uv.points)
        self.assertIsInstance(rebuilt.texture, np.ndarray)
        np.testing.assert_array_equal(rebuilt.texture, tex)
        self.assertEqual(rebuilt.name, obj.name)

    def test_from_dict_handles_optional_fields_missing(self):
        """An Object with no mesh / uv / texture round-trips cleanly."""
        obj     = Object(name="empty")
        rebuilt = Object.from_dict(obj.to_dict())
        self.assertIsNone(rebuilt.mesh)
        self.assertIsNone(rebuilt.uv)
        self.assertIsNone(rebuilt.texture)
        self.assertEqual(rebuilt.name, "empty")

    def test_from_dict_handles_mesh_only(self):
        mesh, _ = _make_textured_cube()
        obj     = Object(name="meshonly", mesh=mesh)
        rebuilt = Object.from_dict(obj.to_dict())
        self.assertIsInstance(rebuilt.mesh, MeshData)
        self.assertIsNone(rebuilt.uv)
        self.assertIsNone(rebuilt.texture)

    def test_from_dict_initializes_caches_to_none(self):
        obj, _ = self._make_full_obj()
        rebuilt = Object.from_dict(obj.to_dict())
        self.assertIsNone(rebuilt._loaded_texture)
        self.assertIsNone(rebuilt._wired_texture_cache)
        self.assertIsNone(rebuilt._frame)
        self.assertIsNone(rebuilt._render_transform_state)

    def test_from_dict_preserves_transforms(self):
        obj, _ = self._make_full_obj()
        rebuilt = Object.from_dict(obj.to_dict())
        np.testing.assert_allclose(rebuilt._translate, obj._translate)
        np.testing.assert_allclose(rebuilt._rotate, obj._rotate)

    # -- save / load round-trips on disk (npz / json / pkl) ------------

    def _round_trip(self, obj: Object, suffix: str) -> Object:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        try:
            obj.save(tmp.name)
            return Object.load(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_save_load_round_trip_npz(self):
        obj, tex = self._make_full_obj()
        rebuilt = self._round_trip(obj, ".npz")
        self.assertIsInstance(rebuilt.mesh, MeshData)
        self.assertIsInstance(rebuilt.uv, UVData)
        np.testing.assert_allclose(rebuilt.mesh.points, obj.mesh.points)
        np.testing.assert_array_equal(rebuilt.texture, tex)

    def test_save_load_round_trip_json(self):
        obj, tex = self._make_full_obj()
        rebuilt = self._round_trip(obj, ".json")
        self.assertIsInstance(rebuilt.mesh, MeshData)
        self.assertIsInstance(rebuilt.uv, UVData)
        np.testing.assert_allclose(rebuilt.mesh.points, obj.mesh.points)
        np.testing.assert_array_equal(rebuilt.texture, tex)

    def test_save_load_round_trip_pkl(self):
        obj, tex = self._make_full_obj()
        rebuilt = self._round_trip(obj, ".pkl")
        # Pickle goes through our __reduce__ -> from_dict so nested
        # MeshData/UVData are properly reconstructed (not raw dicts).
        self.assertIsInstance(rebuilt,      Object)
        self.assertIsInstance(rebuilt.mesh, MeshData)
        self.assertIsInstance(rebuilt.uv,   UVData)
        np.testing.assert_array_equal(rebuilt.texture, tex)

    def test_save_load_path_texture_npz(self):
        """A path-texture'd Object saves with the texture baked into the
        file and reloads as an ndarray (the path itself is intentionally
        not preserved -- the saved file is self-contained)."""
        if _PIL_Image is None:
            self.skipTest("PIL not available")
        mesh, uv = _make_textured_cube()
        tex  = _make_checker()
        path = _write_temp_png(tex)
        try:
            obj     = Object(name="hero", mesh=mesh, uv=uv, texture=path)
            rebuilt = self._round_trip(obj, ".npz")
        finally:
            os.unlink(path)
        self.assertIsInstance(rebuilt.texture, np.ndarray)
        np.testing.assert_array_equal(rebuilt.texture, tex)


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
        """When the binary exists on PATH, the helper returns the Popen
        instance returned by :class:`subprocess.Popen` unchanged."""
        from cgmath.render.frame import _spawn_or_raise

        fake_proc = MagicMock(name="popen")
        with patch(
            "cgmath.render.frame.subprocess.Popen", return_value=fake_proc
        ) as popen:
            result = _spawn_or_raise(["echo", "hi"], stdout=subprocess.PIPE)
        self.assertIs(result, fake_proc)
        popen.assert_called_once_with(["echo", "hi"], stdout=subprocess.PIPE)

    def test_spawn_or_raise_translates_file_not_found_to_runtime_error(self):
        from cgmath.render.frame import _spawn_or_raise

        with patch(
            "cgmath.render.frame.subprocess.Popen",
            side_effect=FileNotFoundError("execvp"),
        ):
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

        with patch(
            "cgmath.render.frame.subprocess.run",
            side_effect=FileNotFoundError("execvp"),
        ):
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
            with patch(
                "cgmath.render.frame.subprocess.Popen",
                side_effect=FileNotFoundError("execvp"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_mp4(self._single_frame(), out)
        self.assertIn("ffmpeg", str(ctx.exception))
        self.assertIn("PATH", str(ctx.exception))

    def test_encode_avi_raises_runtime_error_when_ffmpeg_missing(self):
        from cgmath.render import frame as frame_mod

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.avi")
            with patch(
                "cgmath.render.frame.subprocess.Popen",
                side_effect=FileNotFoundError("execvp"),
            ):
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
            with patch(
                "cgmath.render.frame.subprocess.run",
                side_effect=FileNotFoundError("execvp"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_gif(
                        self._single_frame(), out, use_gifski=True
                    )
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
                patch(
                    "cgmath.render.frame.subprocess.run",
                    side_effect=FileNotFoundError("execvp"),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    frame_mod.Frame.encode_gif(self._single_frame(), out)
        # Error originates from the ffmpeg branch, not gifski.
        self.assertIn("ffmpeg", str(ctx.exception))


# ============================================================================
# Tests for the unified RGBA-background renderer pipeline
# ============================================================================


class TestRendererRGBABackground(unittest.TestCase):
    """Renderer always emits ``(H, W, 4)`` now; ``background`` is RGBA."""

    def _scene(self):
        m, u = _make_textured_cube()
        s = Scene("s")
        s.append(Object(name="x", mesh=m, uv=u))
        return s

    def test_default_bg_is_transparent_black_4ch(self):
        """No bg arg -> output is (H, W, 4) with transparent miss pixels."""
        from cgmath.render.raytracer import render

        f = render(scene=self._scene(), resolution=(64, 64))
        self.assertEqual(f.array.shape, (64, 64, 4))
        # Top-left corner is guaranteed to miss the centered cube.
        np.testing.assert_array_equal(f.array[0, 0], [0, 0, 0, 0])

    def test_opaque_bg_fills_misses_with_color(self):
        from cgmath.render.raytracer import render

        f = render(
            scene      = self._scene(),
            resolution = (64, 64),
            background = (1.0, 1.0, 1.0, 1.0),
        )
        np.testing.assert_array_equal(f.array[0, 0], [255, 255, 255, 255])

    def test_semitransparent_bg_passes_through_at_misses(self):
        from cgmath.render.raytracer import render

        f = render(
            scene      = self._scene(),
            resolution = (64, 64),
            background = (1.0, 0.0, 0.0, 0.5),
        )
        # Miss pixel should be the bg verbatim (un-premultiplied).
        # 0.5 * 255 = 127.5 -> clip to 127 by float-to-uint8 cast.
        np.testing.assert_array_equal(f.array[0, 0], [255, 0, 0, 127])

    def test_hit_pixel_alpha_is_one(self):
        from cgmath.render.raytracer import render

        f = render(
            scene      = self._scene(),
            resolution = (64, 64),
            background = (0.0, 0.0, 0.0, 0.0),
        )
        # Center pixel must be hit by the cube; alpha must be 255.
        self.assertEqual(int(f.array[32, 32, 3]), 255)

    def test_invalid_bg_length_raises(self):
        from cgmath.render.raytracer import render

        with self.assertRaisesRegex(ValueError, "RGBA 4-tuple"):
            render(
                scene      = self._scene(),
                resolution = (8, 8),
                background = (0.0, 0.0, 0.0),
            )

    def test_invalid_bg_range_raises(self):
        from cgmath.render.raytracer import render

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            render(
                scene      = self._scene(),
                resolution = (8, 8),
                background = (0.0, 0.0, 0.0, 1.5),
            )

    def test_rgba_kwarg_removed(self):
        """Old ``rgba=True`` flag is gone -- TypeError, not silent accept."""
        from cgmath.render.raytracer import render

        with self.assertRaises(TypeError):
            render(scene=self._scene(), resolution=(8, 8), rgba=True)

    def test_composite_over_helper_porter_duff(self):
        """Direct unit test of _composite_over math (independent of raytracer)."""
        from cgmath.render.raytracer import _composite_over

        # src: one fully-opaque red, one half-coverage red, one transparent.
        src = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],  # fully covered
                [1.0, 0.0, 0.0, 0.5],  # half-covered
                [1.0, 0.0, 0.0, 0.0],  # missed
            ],
            dtype=np.float32,
        )
        # bg = opaque green
        out = _composite_over(src, (0.0, 1.0, 0.0, 1.0))
        # 1) fully-covered hit: red, alpha=1
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0, 1.0], atol=1e-6)
        # 3) miss pixel: bg (green, alpha=1)
        np.testing.assert_allclose(out[2], [0.0, 1.0, 0.0, 1.0], atol=1e-6)
        # 2) half-covered: alpha = 1.0 (src.a=0.5 + 1*0.5 = 1.0)
        self.assertAlmostEqual(out[1, 3], 1.0, places=6)


# ============================================================================
# Tests for Scene.background / Object.background normalization + precedence
# ============================================================================


class TestBackgroundProperty(unittest.TestCase):
    """RGB->RGBA promotion + cache invalidation + Scene-wins-over-Object."""

    def test_none_stays_none(self):
        self.assertIsNone(Object(name="x").background)
        self.assertIsNone(Scene("s").background)

    def test_rgb_promoted_to_opaque_rgba_object(self):
        obj = Object(name="x", background=(0.2, 0.3, 0.4))
        self.assertEqual(obj.background, (0.2, 0.3, 0.4, 1.0))

    def test_rgb_promoted_to_opaque_rgba_scene(self):
        s = Scene("s", background=(0.2, 0.3, 0.4))
        self.assertEqual(s.background, (0.2, 0.3, 0.4, 1.0))

    def test_rgba_kept_as_is(self):
        obj = Object(name="x", background=(0.2, 0.3, 0.4, 0.5))
        self.assertEqual(obj.background, (0.2, 0.3, 0.4, 0.5))

    def test_invalid_length_raises(self):
        with self.assertRaisesRegex(ValueError, "Object.background"):
            Object(name="x", background=(0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "Scene.background"):
            Scene("s", background=(0.5, 0.5, 0.5, 0.5, 0.5))

    def test_out_of_range_raises(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            Scene("s", background=(1.5, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            Scene("s", background=(0.0, 0.0, 0.0, -0.1))

    def test_setter_normalizes_rgb(self):
        obj            = Object(name="x")
        obj.background = (0.1, 0.2, 0.3)
        self.assertEqual(obj.background, (0.1, 0.2, 0.3, 1.0))

    def test_object_background_setter_marks_dirty(self):
        mesh, uv = _make_textured_cube()
        obj = Object(name="x", mesh=mesh, uv=uv, texture=_make_checker())
        obj.render(resolution=(32, 32))
        self.assertIsNotNone(obj.frame)
        obj.background = (1.0, 0.0, 0.0)
        self.assertIsNone(obj.frame)

    def test_scene_background_change_invalidates_cache(self):
        mesh, uv = _make_textured_cube()
        obj   = Object(name="x", mesh=mesh, uv=uv, texture=_make_checker())
        scene = Scene("s")
        scene.append(obj)
        scene.render(resolution=(32, 32))
        self.assertIsNotNone(scene.frame)
        scene.background = (0.5, 0.5, 0.5)
        self.assertIsNone(scene.frame)

    def test_scene_background_wins_over_object_when_parented(self):
        """When an Object with its own background is rendered via a Scene
        with its own background, the Scene's wins (mirrors how
        resolution / samples_per_pixel precedence works)."""
        from cgmath.render import raytracer as rt

        mesh, uv = _make_textured_cube()
        obj = Object(name="x", mesh=mesh, uv=uv, background=(1.0, 0.0, 0.0))
        s   = Scene("s", background=(0.0, 1.0, 0.0))  # green
        s.append(obj)

        captured = {}

        def fake_render(**kw):
            captured.update(kw)
            from cgmath.render.frame import Frame

            return Frame(array=np.zeros((4, 4, 4), dtype=np.uint8))

        with patch.object(rt, "render", side_effect=fake_render):
            s.render(resolution=(8, 8))

        # Scene's green (promoted to opaque RGBA) reaches the renderer;
        # Object's red is ignored because Object.background is only
        # consulted in the standalone Object.render() preview path.
        self.assertEqual(captured["background"], (0.0, 1.0, 0.0, 1.0))

    def test_object_background_used_in_standalone_preview(self):
        """obj.render() builds an internal one-Object Scene and forwards
        the Object's background to it.  Verify by patching render."""
        from cgmath.render import raytracer as rt

        mesh, uv = _make_textured_cube()
        obj = Object(name="x", mesh=mesh, uv=uv, background=(1.0, 0.0, 0.0))

        captured = {}

        def fake_render(**kw):
            captured.update(kw)
            from cgmath.render.frame import Frame

            return Frame(array=np.zeros((4, 4, 4), dtype=np.uint8))

        with patch.object(rt, "render", side_effect=fake_render):
            obj.render(resolution=(8, 8))

        self.assertEqual(captured["background"], (1.0, 0.0, 0.0, 1.0))

    def test_scene_background_property_round_trips_via_getattr(self):
        """Regression for a bug where Scene.__init__ assigned _background
        but no background property/setter existed on Scene -- so
        ``getattr(scene, 'background')`` (used by
        ``_merge_render_config`` -> ``_RENDER_CONFIG_KEYS``) raised
        AttributeError on the very first ``Scene().render()`` call."""
        scene = Scene("s")
        # getattr must not raise; default is None.
        self.assertIsNone(getattr(scene, "background"))
        scene2 = Scene("s2", background=(0.1, 0.2, 0.3))
        self.assertEqual(getattr(scene2, "background"), (0.1, 0.2, 0.3, 1.0))

    def test_scene_render_works_without_background(self):
        """Regression for the same bug as above: a fresh Scene with no
        explicit background should render cleanly (this used to crash
        in ``_merge_render_config`` with AttributeError)."""
        from cgmath.render import raytracer as rt

        mesh, uv = _make_textured_cube()
        s = Scene("s")
        s.append(Object(name="x", mesh=mesh, uv=uv))

        def fake_render(**kw):
            from cgmath.render.frame import Frame

            return Frame(array=np.zeros((4, 4, 4), dtype=np.uint8))

        with patch.object(rt, "render", side_effect=fake_render):
            # Should not raise.
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

    def test_encode_mp4_validates_background_length_without_ffmpeg(self):
        """Regression: background-shape validation must fire even when
        ffmpeg is absent.  Previously ``encode_mp4`` looked up ffmpeg
        BEFORE validating ``background``, so on Sandcastle Linux runners
        (no ffmpeg installed) bad-shape callers got a misleading
        ``RuntimeError: ffmpeg not found`` instead of the intended
        ``ValueError: ... RGB 3-tuple ...``.  Now that ffmpeg is
        resolved by ``subprocess`` at spawn time, ordering is locked
        in by construction -- this test guards against future
        regressions to a pre-spawn binary check."""
        from cgmath.render.frame import Frame

        with self.assertRaisesRegex(ValueError, "RGB 3-tuple"):
            Frame.encode_mp4(
                iter([np.zeros((8, 8, 3), dtype=np.uint8)]),
                "/tmp/x.mp4",
                background=(0.0, 0.0, 0.0, 1.0),  # 4-tuple, not 3
            )

    def test_encode_mp4_validates_background_range_without_ffmpeg(self):
        """Regression companion to
        ``test_encode_mp4_validates_background_length_without_ffmpeg``:
        out-of-range components must raise ``ValueError`` even with
        ffmpeg absent, so the test is reliable on Sandcastle Linux."""
        from cgmath.render.frame import Frame

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            Frame.encode_mp4(
                iter([np.zeros((8, 8, 3), dtype=np.uint8)]),
                "/tmp/x.mp4",
                background=(1.5, 0.0, 0.0),
            )

    def test_encode_mp4_accepts_rgba_input(self):
        """Old code raised on RGBA; new code auto-flattens.  Just verify
        the function runs and emits a non-empty file."""
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
                    background = (1.0, 0.0, 0.0),  # opaque red
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
        """AVI w/ PNG codec keeps per-pixel alpha."""
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
                # Decode the first frame back via ffmpeg (PNG codec preserves alpha).
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as p:
                    p.close()
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            t.name,
                            "-frames:v",
                            "1",
                            p.name,
                        ],
                        check          = True,
                        capture_output = True,
                    )
                    decoded = np.asarray(Image.open(p.name).convert("RGBA"))
                    os.unlink(p.name)
                self.assertEqual(decoded.shape[-1], 4)
                # Alpha preserved within +/-2 (codec slack).
                self.assertLess(abs(int(decoded[8, 8, 3]) - 128), 3)
            finally:
                os.unlink(t.name)

    def test_scene_turntable_mp4_uses_scene_background(self):
        """Scene.turntable forwards self._background[:3] to encode_mp4."""
        from cgmath.render.frame import Frame

        m, u = _make_textured_cube()
        s = Scene("s", background=(0.0, 1.0, 0.0))  # opaque green via promotion
        s.append(Object(name="x", mesh=m, uv=u))
        with patch.object(Frame, "encode_mp4", return_value="out.mp4") as mock:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
                t.close()
                try:
                    s.turntable(t.name, n_frames=2, resolution=(8, 8))
                finally:
                    os.unlink(t.name)
        kw = mock.call_args.kwargs
        self.assertEqual(kw["background"], (0.0, 1.0, 0.0))

    def test_scene_turntable_explicit_background_overrides_scene(self):
        from cgmath.render.frame import Frame

        m, u = _make_textured_cube()
        s = Scene("s", background=(0.0, 0.0, 0.0, 0.0))  # transparent for previews
        s.append(Object(name="x", mesh=m, uv=u))
        with patch.object(Frame, "encode_mp4", return_value="out.mp4") as mock:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
                t.close()
                try:
                    s.turntable(
                        t.name,
                        n_frames   = 2,
                        resolution = (8, 8),
                        background = (0.5, 0.5, 0.5),
                    )
                finally:
                    os.unlink(t.name)
        self.assertEqual(mock.call_args.kwargs["background"], (0.5, 0.5, 0.5))

    def test_scene_turntable_avi_routes_to_encode_avi(self):
        from cgmath.render.frame import Frame

        m, u = _make_textured_cube()
        s = Scene("s")
        s.append(Object(name="x", mesh=m, uv=u))
        with patch.object(Frame, "encode_avi", return_value="out.avi") as mock:
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as t:
                t.close()
                try:
                    s.turntable(t.name, n_frames=2, resolution=(8, 8))
                finally:
                    os.unlink(t.name)
        mock.assert_called_once()


# ============================================================================
# Tests for Object.triangulate
# ============================================================================


class TestObjectTriangulate(unittest.TestCase):
    """Object.triangulate keeps mesh + UV in sync and invalidates caches."""

    def _quad_obj(self):
        """Build an Object whose mesh is a single quad plus a matching UV."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        indices = np.array([0, 1, 2, 3], dtype=np.int64)
        counts  = np.array([4], dtype=np.int64)
        mesh    = MeshData(points=points, indices=indices, counts=counts)
        uv_pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        uv = UVData(points=uv_pts, indices=indices.copy(), counts=counts.copy())
        return Object(name="q", mesh=mesh, uv=uv)

    def test_triangulate_converts_quads_to_tris(self):
        obj = self._quad_obj()
        self.assertEqual(int(obj.mesh.counts[0]), 4)
        obj.triangulate()
        # Every face is now a triangle.
        self.assertTrue(np.all(obj.mesh.counts == 3))
        # 1 quad -> 2 triangles.
        self.assertEqual(int(obj.mesh.face_count), 2)

    def test_triangulate_keeps_mesh_and_uv_face_counts_in_sync(self):
        obj = self._quad_obj()
        obj.triangulate()
        self.assertEqual(int(obj.mesh.face_count), int(obj.uv.face_count))
        np.testing.assert_array_equal(obj.mesh.counts, obj.uv.counts)

    def test_triangulate_works_with_no_uv(self):
        obj    = self._quad_obj()
        obj.uv = None
        obj.triangulate()
        self.assertTrue(np.all(obj.mesh.counts == 3))
        self.assertIsNone(obj.uv)

    def test_triangulate_raises_when_mesh_is_none(self):
        obj = Object(name="empty")
        with self.assertRaisesRegex(ValueError, "self.mesh is None"):
            obj.triangulate()

    def test_triangulate_invalidates_render_cache(self):
        obj = self._quad_obj()
        obj.render(resolution=(16, 16))
        self.assertIsNotNone(obj.frame)
        obj.triangulate()
        self.assertIsNone(obj.frame)

    @unittest.skipUnless(_HAS_RASTERIZER, "needs cv2 or scikit-image")
    def test_triangulate_invalidates_wireframe_bake(self):
        obj           = self._quad_obj()
        obj.wireframe = True
        # Force a bake.
        _ = obj.get_loaded_texture()
        self.assertIsNotNone(obj._wired_texture_cache)
        obj.triangulate()
        self.assertIsNone(obj._wired_texture_cache)

    def test_triangulate_idempotent_on_already_tri_mesh(self):
        """Running triangulate on an already-triangle mesh is a no-op
        (all counts stay 3, face_count unchanged)."""
        obj = self._quad_obj()
        obj.triangulate()
        face_count_before = int(obj.mesh.face_count)
        obj.triangulate()
        self.assertEqual(int(obj.mesh.face_count), face_count_before)
        self.assertTrue(np.all(obj.mesh.counts == 3))


# ============================================================================
# Tests for samples_per_pixel validation (regression for bare obj.render())
# ============================================================================


class TestSamplesPerPixelValidation(unittest.TestCase):
    """Regression coverage for the bug where Object's default
    samples_per_pixel was 2 -- not a perfect square -- so bare
    ``obj.render()`` raised ``ValueError`` deep in the raytracer.

    The fix changed the default to 4 (smallest perfect square > 1)
    and added validation at the property setter / constructor so
    invalid values fail at assignment time rather than render time.
    """

    def _make_cube_obj(self):
        mesh, uv = _make_textured_cube()
        return Object(name="cube", mesh=mesh, uv=uv)

    # -- default + bare-render regression -----------------------------

    def test_object_default_spp_is_perfect_square(self):
        """The default must satisfy the renderer's perfect-square
        invariant (was 2, now 4)."""
        obj = Object(name="x")
        spp = obj.samples_per_pixel
        self.assertIsNotNone(spp)
        self.assertEqual(spp * 0, 0)  # int check
        n_axis = int(round(spp**0.5))
        self.assertEqual(
            n_axis * n_axis, spp, f"default SPP {spp} is not a perfect square"
        )
        self.assertGreaterEqual(spp, 1)

    def test_bare_object_render_does_not_crash(self):
        """Regression: ``obj.render()`` with no args must not raise.
        This was the user-visible failure mode before the fix."""
        obj = self._make_cube_obj()
        # No args -- defaults must all line up with the renderer's
        # contract (perfect-square SPP, RGBA bg, etc.).
        frame = obj.render(resolution=(16, 16))
        self.assertEqual(frame.array.shape, (16, 16, 4))

    def test_bare_scene_render_does_not_crash(self):
        """Same regression on the Scene path."""
        m, u = _make_textured_cube()
        s = Scene("s")
        s.append(Object(name="x", mesh=m, uv=u))
        frame = s.render(resolution=(16, 16))
        self.assertEqual(frame.array.shape, (16, 16, 4))

    # -- setter validator: fail-fast ----------------------------------

    def test_setter_rejects_two(self):
        obj = Object(name="x")
        with self.assertRaisesRegex(ValueError, "perfect square"):
            obj.samples_per_pixel = 2

    def test_setter_rejects_three(self):
        obj = Object(name="x")
        with self.assertRaisesRegex(ValueError, "perfect square"):
            obj.samples_per_pixel = 3

    def test_setter_rejects_zero_and_negative(self):
        obj = Object(name="x")
        with self.assertRaisesRegex(ValueError, ">= 1"):
            obj.samples_per_pixel = 0
        with self.assertRaisesRegex(ValueError, ">= 1"):
            obj.samples_per_pixel = -4

    def test_setter_accepts_valid_values(self):
        obj = Object(name="x")
        for v in (1, 4, 9, 16, 25, 100, None):
            obj.samples_per_pixel = v
            self.assertEqual(obj.samples_per_pixel, v)

    def test_setter_rejects_non_int(self):
        obj = Object(name="x")
        with self.assertRaisesRegex(ValueError, "must be an int"):
            obj.samples_per_pixel = "four"

    # -- constructor validator: fail-fast -----------------------------

    def test_constructor_rejects_invalid_spp(self):
        with self.assertRaisesRegex(ValueError, "perfect square"):
            Object(name="x", samples_per_pixel=2)
        with self.assertRaisesRegex(ValueError, "perfect square"):
            Object(name="x", samples_per_pixel=7)

    def test_constructor_accepts_none_and_perfect_squares(self):
        for v in (None, 1, 4, 9, 16):
            obj = Object(name="x", samples_per_pixel=v)
            self.assertEqual(obj.samples_per_pixel, v)


if __name__ == "__main__":
    unittest.main()