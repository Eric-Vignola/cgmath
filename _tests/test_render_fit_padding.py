"""Tests for the autofit fit_padding safety margin.

Bezier-sampled meshes evaluate surfaces that bulge OUTSIDE the
control-point hull, so a tight autofit clips the silhouette.  The
``fit_padding`` parameter scales the autofit point cloud about its AABB
center to add a margin.  These tests cover:

* ``_inflate_points_about_center`` (camera.py) -- the geometric primitive.
* ``_scene_uses_bezier`` / ``_resolve_fit_padding`` (raytracer.py) -- the
  per-call resolution policy used by ``render()``.
* ``_resolve_turntable_fit_padding`` (scene.py) -- the same policy on the
  turntable path, from the scene's sample method.
* ``Scene`` config plumbing (the ``fit_padding`` setting + ``_RENDER_CONFIG_KEYS``).
* End-to-end: a bezier render's autofit camera sits further back than
  the same scene rendered with bilinear sampling.
"""

import unittest
from typing import Optional, Tuple

import numpy as np
from cgmath.geometry.mesh import MeshData, SampleMethod, UVData
from cgmath.render import render, Scene
from cgmath.render.camera import _inflate_points_about_center
from cgmath.render.raytracer import _resolve_fit_padding, _scene_uses_bezier
from cgmath.render.scene import _RENDER_CONFIG_KEYS, _resolve_turntable_fit_padding


def _make_unit_cube() -> Tuple[MeshData, UVData]:
    """Centered unit cube (8 verts, 6 quads) + a trivial per-corner UV."""
    points = np.array(
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
    indices = np.array(
        [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
    )
    counts = np.array([4, 4, 4, 4, 4, 4])
    mesh   = MeshData(points=points, indices=indices, counts=counts)

    # Bare-minimum UV: 4 corners shared across all faces.  The renderer
    # will only invoke UV interpolation when a texture is set; we leave
    # texture None so the UV is just along for the validation ride.
    uv_points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    uv_indices = np.array(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    )
    uv = UVData(points=uv_points, indices=uv_indices, counts=counts, mesh_uuid=mesh.uuid)
    return mesh, uv


# -- camera._inflate_points_about_center -------------------------------------


class TestInflatePointsAboutCenter(unittest.TestCase):
    def test_factor_one_returns_input(self):
        pts = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        out = _inflate_points_about_center(pts, 1.0)
        # factor=1.0 fast-paths to identity; no copy expected.
        self.assertIs(out, pts)

    def test_empty_input_is_passthrough(self):
        pts = np.zeros((0, 3))
        out = _inflate_points_about_center(pts, 2.0)
        self.assertEqual(out.shape, (0, 3))

    def test_scales_outward_from_aabb_center(self):
        # AABB center of {[0,0,0], [2,4,6]} is [1,2,3].  Scaling by 2.0
        # sends them to [-1,-2,-3] and [3,6,9] respectively.
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
        out = _inflate_points_about_center(pts, 2.0)
        np.testing.assert_allclose(out, [[-1.0, -2.0, -3.0], [3.0, 6.0, 9.0]])

    def test_preserves_aabb_center(self):
        rng           = np.random.default_rng(0)
        pts           = rng.standard_normal((100, 3))
        center_before = 0.5 * (pts.min(0) + pts.max(0))
        out           = _inflate_points_about_center(pts, 1.5)
        center_after  = 0.5 * (out.min(0) + out.max(0))
        np.testing.assert_allclose(center_after, center_before, atol=1e-12)

    def test_aabb_extent_scales_by_factor(self):
        rng           = np.random.default_rng(1)
        pts           = rng.standard_normal((50, 3))
        extent_before = pts.max(0) - pts.min(0)
        out           = _inflate_points_about_center(pts, 1.25)
        extent_after  = out.max(0) - out.min(0)
        np.testing.assert_allclose(extent_after, extent_before * 1.25, atol=1e-12)


# -- raytracer._scene_uses_bezier --------------------------------------------


class TestSceneUsesBezier(unittest.TestCase):
    def test_empty_returns_false(self):
        self.assertFalse(_scene_uses_bezier([]))

    def test_missing_sample_method_returns_false(self):
        self.assertFalse(_scene_uses_bezier([{"mesh": object()}]))

    def test_bilinear_string_returns_false(self):
        self.assertFalse(_scene_uses_bezier([{"sample_method": "bilinear"}]))

    def test_bezier_string_returns_true(self):
        self.assertTrue(_scene_uses_bezier([{"sample_method": "bezier"}]))

    def test_bezier_string_case_insensitive(self):
        self.assertTrue(_scene_uses_bezier([{"sample_method": "BEZIER"}]))
        self.assertTrue(_scene_uses_bezier([{"sample_method": "Bezier"}]))

    def test_bezier_enum_returns_true(self):
        self.assertTrue(_scene_uses_bezier([{"sample_method": SampleMethod.BEZIER}]))

    def test_bilinear_enum_returns_false(self):
        self.assertFalse(_scene_uses_bezier([{"sample_method": SampleMethod.BILINEAR}]))

    def test_any_bezier_in_mixed_scene_returns_true(self):
        entries = [
            {"sample_method": "bilinear"},
            {"sample_method": "bezier"},
            {"sample_method": "bilinear"},
        ]
        self.assertTrue(_scene_uses_bezier(entries))

    def test_a_scenes_entries_carry_its_sample_method(self):
        mesh, uv = _make_unit_cube()
        scene = Scene("s", sample_method="bezier").add(mesh, uv)
        self.assertTrue(_scene_uses_bezier(scene._render_inputs().entries))
        scene.sample_method = "bilinear"
        self.assertFalse(_scene_uses_bezier(scene._render_inputs().entries))


# -- raytracer._resolve_fit_padding ------------------------------------------


class TestResolveFitPadding(unittest.TestCase):
    def test_explicit_value_wins_over_default(self):
        self.assertEqual(_resolve_fit_padding(2.5, [{"sample_method": "bezier"}]), 2.5)

    def test_explicit_one_disables_default(self):
        # An explicit 1.0 must override the bezier auto-bump (caller knows
        # what they're doing -- no surprise inflation).
        self.assertEqual(_resolve_fit_padding(1.0, [{"sample_method": "bezier"}]), 1.0)

    def test_default_is_one_for_bilinear(self):
        self.assertEqual(
            _resolve_fit_padding(None, [{"sample_method": "bilinear"}]), 1.0
        )

    def test_default_is_one_point_one_five_for_bezier(self):
        self.assertAlmostEqual(
            _resolve_fit_padding(None, [{"sample_method": "bezier"}]), 1.15
        )

    def test_default_is_one_for_empty_scene(self):
        self.assertEqual(_resolve_fit_padding(None, []), 1.0)


# -- scene._resolve_turntable_fit_padding ------------------------------------


class TestResolveTurntableFitPadding(unittest.TestCase):
    def test_pops_explicit_kwarg(self):
        scene  = Scene("s", sample_method="bezier")
        kwargs = {"fit_padding": 1.4, "other": "untouched"}
        pad    = _resolve_turntable_fit_padding(kwargs, scene)
        self.assertEqual(pad, 1.4)
        # Must be popped so per-frame render() doesn't re-apply.
        self.assertNotIn("fit_padding", kwargs)
        self.assertEqual(kwargs, {"other": "untouched"})

    def test_default_one_for_bilinear_scene(self):
        kwargs: dict = {}
        self.assertEqual(_resolve_turntable_fit_padding(kwargs, Scene("s")), 1.0)

    def test_default_one_one_five_for_bezier_scene(self):
        kwargs: dict = {}
        self.assertAlmostEqual(_resolve_turntable_fit_padding(kwargs, Scene("s", sample_method="bezier")), 1.15)
        self.assertAlmostEqual(_resolve_turntable_fit_padding({}, Scene("s", sample_method=SampleMethod.BEZIER)), 1.15)

    def test_explicit_one_overrides_bezier_default(self):
        kwargs: dict = {"fit_padding": 1.0}
        self.assertEqual(_resolve_turntable_fit_padding(kwargs, Scene("s", sample_method="bezier")), 1.0)


# -- Scene config plumbing --------------------------------------------------


class TestSceneFitPaddingConfig(unittest.TestCase):
    def test_init_accepts_fit_padding(self):
        scene = Scene("s", fit_padding=1.3)
        self.assertEqual(scene.fit_padding, 1.3)

    def test_default_fit_padding_is_none(self):
        scene = Scene("s")
        self.assertIsNone(scene.fit_padding)

    def test_fit_padding_in_render_config_keys(self):
        self.assertIn("fit_padding", _RENDER_CONFIG_KEYS)

    def test_configure_sets_fit_padding(self):
        scene = Scene("s")
        scene.configure(fit_padding=1.2)
        self.assertEqual(scene.fit_padding, 1.2)

    def test_merge_render_config_propagates_fit_padding(self):
        scene  = Scene("s", fit_padding=1.4)
        merged = scene._merge_render_config({})
        self.assertEqual(merged.get("fit_padding"), 1.4)

    def test_per_call_fit_padding_wins_over_scene_attr(self):
        scene  = Scene("s", fit_padding=1.4)
        merged = scene._merge_render_config({"fit_padding": 1.7})
        self.assertEqual(merged.get("fit_padding"), 1.7)


# -- End-to-end: bezier render pulls camera further back --------------------


class TestBezierAutofitPullsCameraBack(unittest.TestCase):
    """Integration test: rendering with bezier sampling should produce a
    camera that sits further from the scene than the bilinear baseline.

    We don't need the bezier evaluator to do anything sophisticated -- as
    long as ``render()`` resolves ``fit_padding`` to a value > 1.0 when
    bezier is in play, the resulting Frame's ``camera_matrix`` translation
    must have a strictly larger magnitude than the bilinear baseline.
    """

    def setUp(self):
        self.mesh, self.uv = _make_unit_cube()

    def _render_and_get_camera_distance(
        self,
        sample_method: str,
        fit_padding:   Optional[float] = None,
    ) -> float:
        kwargs = {
            "mesh": self.mesh,
            "sample_method": sample_method,
            "resolution": (64, 64),
            "samples_per_pixel": 1,
        }
        if fit_padding is not None:
            kwargs["fit_padding"] = fit_padding
        frame = render(**kwargs)
        # Frame.camera_matrix is column-major OpenGL -> position in [:3, 3].
        cam_pos = frame.camera_matrix[:3, 3]
        return float(np.linalg.norm(cam_pos))

    def _scene_camera_distance(self, sample_method: str) -> float:
        scene = Scene("s", sample_method=sample_method, resolution=(64, 64), samples_per_pixel=1).add(self.mesh)
        frame = scene.render(default_light=False)
        return float(np.linalg.norm(frame.camera_matrix[:3, 3]))

    def test_bezier_default_pulls_back_more_than_bilinear(self):
        d_bilinear = self._render_and_get_camera_distance("bilinear")
        d_bezier   = self._render_and_get_camera_distance("bezier")
        # Bezier auto-defaults to 1.15 -> camera ~15% further back.
        self.assertGreater(d_bezier, d_bilinear)

    def test_a_bezier_scene_pulls_back_too(self):
        self.assertGreater(self._scene_camera_distance("bezier"), self._scene_camera_distance("bilinear"))

    def test_explicit_padding_one_disables_inflation(self):
        # Explicitly setting fit_padding=1.0 must match the bilinear path
        # exactly (no surprise margin even though sample_method=bezier).
        d_bilinear = self._render_and_get_camera_distance("bilinear")
        d_bezier_no_pad = self._render_and_get_camera_distance(
            "bezier", fit_padding=1.0
        )
        np.testing.assert_allclose(d_bezier_no_pad, d_bilinear, rtol=1e-9)

    def test_larger_padding_pulls_camera_back_further(self):
        d_small = self._render_and_get_camera_distance("bilinear", fit_padding=1.1)
        d_large = self._render_and_get_camera_distance("bilinear", fit_padding=1.5)
        self.assertGreater(d_large, d_small)
