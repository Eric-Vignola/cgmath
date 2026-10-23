"""render() renders a frame in bands of pixel rows on a thread pool.

Every banded frame is byte for byte the whole frame (raytracer._WHOLE_FRAME,
today's path, kept as the reference): image, alpha and depth, for every call
style, size, sample count, background and ``threads=``. The band sizes are
patched small so that the frames here split into many bands.
"""

from __future__ import annotations

import ast
import functools
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from concurrent.futures import Future
from unittest import mock

import numba
import numpy as np
from cgmath.geometry.mesh import MeshData, UVData, UVList
from cgmath.render import _tiles, raytracer, render
from cgmath.render.raytracer import _generate_camera_rays, look_at
from cgmath.render.scene import Camera, Light, Object, Scene

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pygltflib
    import trimesh
except ImportError:
    pygltflib = trimesh = None

ASSETS   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assets")
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_MAYA  = "maya" in sys.modules
BLAS_OPS = {"dot", "matmul", "inner", "vdot", "tensordot", "multi_dot"}


def _layer():
    """numba's threading layer, launched first (numba.get_num_threads()
    launches it): a test run on its own has rendered nothing yet"""
    numba.get_num_threads()
    return numba.threading_layer()


def _side_by_side(threads):
    """whether bands run side by side with *threads*"""
    return _layer() in ("omp", "tbb") and _tiles.thread_budget(threads) > 1


# -- scenes ------------------------------------------------------------------------


def _checker(n=128, cells=8):
    y, x = np.mgrid[0:n, 0:n]
    c           = ((x // (n // cells)) + (y // (n // cells))) % 2
    tex         = np.zeros((n, n, 3), dtype=np.uint8)
    tex[..., 0] = np.where(c, 230, 40)
    tex[..., 1] = (x * 255 // n).astype(np.uint8)
    tex[..., 2] = (y * 255 // n).astype(np.uint8)
    return tex


def _cube():
    """12 triangles (no BVH: 64 faces or fewer), one UV square per side."""
    points = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ]
    )
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    indices, counts, uv_points, uv_indices = [], [], [], []
    for side, q in enumerate(quads):
        indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]])
        counts.extend([3, 3])
        base = len(uv_points)
        uv_points.extend([[0, 0], [1, 0], [1, 1], [0, 1]])
        uv_indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    mesh = MeshData(
        indices = np.array(indices, np.int32),
        counts  = np.array(counts, np.int32),
        points  = points,
    )
    uv = UVData(
        indices = np.array(uv_indices, np.int32),
        counts  = np.array(counts, np.int32),
        points  = np.array(uv_points, float),
    )
    return mesh, uv


@functools.lru_cache(maxsize=None)
def _assets():
    """the 15k-face textured character (BVH), and the igloo"""
    head = MeshData.load(os.path.join(ASSETS, "test_resample.src_mesh_data.npz"))
    uv   = UVList.load(os.path.join(ASSETS, "test_resample.src_uv_data.npz"))[0]
    igl  = MeshData.load(os.path.join(ASSETS, "test_sdf.igloo.npz"))
    return head, uv, igl


@functools.lru_cache(maxsize=None)
def _scene(name):
    head, head_uv, igl = _assets()
    tex = _checker()
    s   = Scene(name)
    if name == "cube":
        mesh, uv = _cube()
        s.append(Object(name="cube", mesh=mesh, uv=uv, texture=tex))
    elif name in ("head", "bezier"):
        o = Object(name="head", mesh=head, uv=head_uv, texture=tex)
        if name == "bezier":
            o.sample_method = "bezier"
        s.append(o)
    elif name == "multi":
        # transformed Objects (rotation, non-uniform scale, translation),
        # a point and an infinite light, a Camera, an opaque background
        size = float(np.max(head.points.max(0) - head.points.min(0)))
        isz  = float(np.max(igl.points.max(0) - igl.points.min(0)))
        crate, crate_uv = _cube()
        s.append(
            Object(
                name      = "head",
                mesh      = head,
                uv        = head_uv,
                texture   = tex,
                rotate    = (0.0, -20.0, 5.0),
                translate = (0.0, 0.1 * size, 0.0),
            )
        )
        igloo = Object(
            name      = "igloo",
            mesh      = igl,
            translate = (size * 1.1, 0.0, 0.0),
            scale     = (size / isz, 0.6 * size / isz, 1.3 * size / isz),
            rotate    = (0.0, 30.0, 0.0),
        )
        igloo.base_color = (0.3, 0.6, 0.9)
        s.append(igloo)
        s.append(
            Object(
                name      = "crate",
                mesh      = crate,
                uv        = crate_uv,
                texture   = tex,
                translate = (-size * 0.9, 0.0, size * 0.2),
                scale     = (size * 0.5, size * 0.3, size * 0.7),
                rotate    = (15.0, 40.0, 0.0),
            )
        )
        s.append(
            Light(
                name      = "key",
                kind      = "point",
                translate = (size * 3, size * 3, size * 3),
                intensity = (size * 5) ** 2,
            )
        )
        s.append(Light(name="sun", kind="infinite", rotate=(-45.0, 30.0, 0.0), intensity=0.6))
        pts = np.concatenate([head.points, head.points + (size * 1.1, 0, 0)])
        s.append(Camera.default_view(pts, name="cam", angle_of_view=40.0))
        s.background = (0.1, 0.1, 0.12, 1.0)
    elif name == "ortho":
        s.append(Object(name="head", mesh=head, uv=head_uv, texture=tex))
        cam                 = Camera.default_view(head.points, name="ortho_cam")
        cam.is_orthographic = True
        s.append(cam)
    else:
        raise KeyError(name)
    return s


# -- banded vs whole frame -----------------------------------------------------------


class _Bands:
    """patches the band sizes to *samples*, and records each frame's bands"""

    def __init__(self, samples=None):
        self.samples = samples
        self.frames  = []

    def __enter__(self):
        self._patches = [mock.patch.object(_tiles, "_stats_hook", self.frames.append)]
        if self.samples is not None:
            self._patches += [
                mock.patch.object(_tiles, "_MIN_BAND", self.samples),
                mock.patch.object(_tiles, "_MAX_BAND", self.samples),
                mock.patch.object(_tiles, "_ONE_BAND_BELOW", 0),
            ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


def _whole(call):
    with mock.patch.object(raytracer, "_WHOLE_FRAME", True):
        return call()


class BandCase(unittest.TestCase):
    def assertSameFrame(self, frame, ref, what):
        self.assertEqual(frame.array.dtype, ref.array.dtype, what)
        self.assertEqual(frame.array.shape, ref.array.shape, what)
        if frame.array.tobytes() != ref.array.tobytes():
            n = int(np.count_nonzero(frame.array != ref.array))
            self.fail(f"{what}: {n} image elements differ")
        if ref.depth is None:
            self.assertIsNone(frame.depth, what)
        elif frame.depth.tobytes() != ref.depth.tobytes():
            n = int(np.count_nonzero(frame.depth.view(np.int64) != ref.depth.view(np.int64)))
            self.fail(f"{what}: {n} depth elements differ")
        np.testing.assert_array_equal(frame.camera_matrix, ref.camera_matrix)
        self.assertEqual(frame.angle_of_view, ref.angle_of_view)

    def check(self, call, configs, what):
        """*call(**kw)* banded, for each ``(samples, threads)`` of *configs*,
        against the whole frame. Returns the band counts."""
        ref     = _whole(call)
        n_bands = []
        for samples, threads in configs:
            with _Bands(samples) as bands:
                frame = call(threads=threads)
            self.assertSameFrame(frame, ref, f"{what} bands={samples} threads={threads}")
            n_bands.append(len(bands.frames[-1]["bands"]))
        return n_bands


class TestIdentity(BandCase):
    """T1: banded == whole frame"""

    CONFIGS = [
        ((333, 217), 4, (0.0, 0.0, 0.0, 0.0), [(None, None), (997, 3), (37, 1)]),
        ((64, 48), 16, (0.2, 0.4, 0.6, 0.5), [(997, None), (37, 3)]),
        ((7, 5), 9, (0.0, 0.0, 0.0, 1.0), [(37, None), (1, 3)]),
        ((1, 1), 1, (0.0, 0.0, 0.0, 0.0), [(37, None)]),
        ((1, 9), 4, (0.2, 0.4, 0.6, 0.5), [(4, None), (37, 1)]),
        # the smallest band: one row of two samples
        ((2, 7), 1, (1.0, 1.0, 1.0, 1.0), [(2, None), (2, 1)]),
        # one sample wide: one band, whatever the band size (a one-sample
        # band rounds its matmuls differently)
        ((1, 217), 1, (0.2, 0.4, 0.6, 0.5), [(1, None), (1, 1)]),
    ]

    def run_scene(self, name, configs=None):
        s = _scene(name)
        for resolution, spp, background, bands in configs or self.CONFIGS:
            call = functools.partial(
                s.render,
                resolution        = resolution,
                samples_per_pixel = spp,
                background        = background,
                return_depth      = True,
            )
            n_bands = self.check(call, bands, f"{name} {resolution} spp{spp} bg{background}")
            if resolution == (333, 217):
                self.assertGreater(min(n_bands), 1)
            if resolution == (1, 1):
                self.assertEqual(n_bands, [1])
            if resolution == (2, 7):
                self.assertEqual(n_bands, [7, 7])
            if resolution == (1, 217):
                self.assertEqual(n_bands, [1, 1])

    def test_cube_no_bvh(self):
        self.assertIsNone(_scene("cube").objects[0].mesh.bvh_bilinear)
        self.run_scene("cube")

    def test_textured_character_bvh(self):
        self.run_scene("head")
        self.run_scene(
            "head", [((160, 120), 1, (1.0, 1.0, 1.0, 1.0), [(None, None), (997, 1)])]
        )

    def test_bezier(self):
        self.run_scene("bezier", self.CONFIGS[:3])

    def test_transformed_objects_lights_camera(self):
        self.run_scene("multi")

    def test_orthographic_camera(self):
        self.run_scene("ortho", self.CONFIGS[:3])

    def test_default_band_sizes_split_a_frame(self):
        call = functools.partial(
            _scene("head").render, resolution=(200, 150), samples_per_pixel=4, return_depth=True
        )
        self.assertGreater(self.check(call, [(None, None), (None, 2)], "default bands")[0], 1)

    def test_legacy_calls(self):
        head, head_uv, igl = _assets()
        tex    = _checker()
        pts    = head.points
        centre = 0.5 * (pts.min(0) + pts.max(0))
        size   = float(np.max(pts.max(0) - pts.min(0)))
        cam    = look_at(tuple(centre + np.array([0.3, 0.4, 2.5]) * size), tuple(centre))
        lights = [
            {"kind": "point", "position": tuple(centre + size * 2), "intensity": size * size * 4},
            {"kind": "infinite", "direction": (0.0, -1.0, -1.0), "intensity": 0.5},
        ]
        for spp in (1, 4):
            calls = {
                "single mesh, point_light list": functools.partial(
                    render,
                    head,
                    head_uv,
                    tex,
                    camera_matrix     = cam,
                    resolution        = (200, 150),
                    samples_per_pixel = spp,
                    return_depth      = True,
                    point_light       = lights,
                ),
                "list of dicts, a bezier entry": functools.partial(
                    render,
                    scene=[
                        {"mesh": head, "uv": head_uv, "texture": tex},
                        {"mesh": igl, "base_color": (0.2, 0.8, 0.3), "sample_method": "bezier"},
                    ],
                    resolution        = (210, 140),
                    samples_per_pixel = spp,
                    return_depth      = True,
                ),
                "no light, clamp, one-sided": functools.partial(
                    render,
                    head,
                    resolution=(120, 90),
                    samples_per_pixel=spp,
                    default_light=False,
                    return_depth=True,
                    wrap="clamp",
                    twosided=False,
                ),
            }
            for what, call in calls.items():
                self.check(call, [(997, None), (37, 1)], f"{what} spp{spp}")

    def test_no_depth_and_threads_above_the_cores(self):
        call = functools.partial(
            _scene("multi").render, resolution=(96, 64), samples_per_pixel=4
        )
        self.check(call, [(997, 10 ** 6), (997, numba.config.NUMBA_NUM_THREADS + 1)], "capped")
        with _Bands(997):
            self.assertIsNone(call().depth)

    def test_coincident_objects_tie_to_the_first(self):
        # one mesh twice at one place, two colours: every hit is an exact
        # tie of depth, which the first Object wins; at the world's origin,
        # and both through the same transform
        mesh, uv = _cube()
        for moved in (False, True):
            s = Scene("tie")
            for name, colour in (("red", (0.9, 0.1, 0.1)), ("blue", (0.1, 0.1, 0.9))):
                if moved:
                    o = Object(
                        name      = name,
                        mesh      = mesh,
                        uv        = uv,
                        translate = (0.1, -0.2, 0.3),
                        rotate    = (10.0, 20.0, 30.0),
                        scale     = (1.0, 1.3, 0.8),
                    )
                else:
                    o = Object(name=name, mesh=mesh, uv=uv)
                o.base_color = colour
                s.append(o)
            call = functools.partial(
                s.render, resolution=(120, 90), samples_per_pixel=4, return_depth=True
            )
            ref = _whole(call)
            # both colours are in the scene; only the first in the frame
            self.assertGreater(int(np.count_nonzero(ref.array[..., 3])), 1000)
            self.assertEqual(int(np.count_nonzero(ref.array[..., 2] > ref.array[..., 0])), 0)
            self.check(call, [(997, None), (37, 1), (None, None)], f"tie moved={moved}")

    def test_a_mesh_with_no_faces_renders_as_nothing(self):
        # beside the cube: the cube's frame. The whole frame raises
        # IndexError on it (in _saddle_surface.raycast); the bands cast and
        # post-process the rays that hit, and none hits a mesh with no face
        mesh, uv = _cube()
        empty = MeshData(
            indices = np.zeros(0, np.int32),
            counts  = np.zeros(0, np.int32),
            points  = np.zeros((0, 3)),
        )
        points_only = MeshData(
            indices = np.zeros(0, np.int32),
            counts  = np.zeros(0, np.int32),
            points  = np.array([[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]]),
        )
        kw = dict(
            camera_matrix     = look_at((1.2, 1.5, 3.0), (0.0, 0.0, 0.0)),
            resolution        = (96, 64),
            samples_per_pixel = 4,
            return_depth      = True,
        )
        ref   = _whole(functools.partial(render, scene=[{"mesh": mesh}], **kw))
        alone = Scene("cube")
        alone.append(Object(name="cube", mesh=mesh, uv=uv))
        beside = Scene("beside")
        beside.append(Object(name="cube", mesh=mesh, uv=uv))
        beside.append(Object(name="empty", mesh=empty))
        beside.append(Object(name="points", mesh=points_only))
        scene_kw  = dict(resolution=(96, 64), samples_per_pixel=4, return_depth=True)
        scene_ref = _whole(functools.partial(alone.render, **scene_kw))
        for samples, threads in ((None, None), (None, 1), (37, None), (37, 1)):
            what = f"bands={samples} threads={threads}"
            with _Bands(samples), warnings.catch_warnings():
                # the points-only mesh's vertex normals: 0 / 0, as before
                warnings.simplefilter("ignore", RuntimeWarning)
                for other in (empty, points_only):
                    frame = render(scene=[{"mesh": mesh}, {"mesh": other}], threads=threads, **kw)
                    self.assertSameFrame(frame, ref, f"{what} {other.points.shape}")
                frame = beside.render(threads=threads, **scene_kw)
                self.assertSameFrame(frame, scene_ref, f"{what} Scene")
        self.assertGreater(int(np.count_nonzero(ref.array[..., 3])), 500)


class TestBandSizes(BandCase):
    """the band sizes of render()'s docstring, at the real constants"""

    def sizes(self, width, height, spp, budget):
        """the samples in each band: whole rows, in order, the whole frame"""
        bands = _tiles._bands(width, height, spp, budget)
        rows  = [r for band in bands for r in range(*band)]
        self.assertEqual(rows, list(range(height)))
        return [(r1 - r0) * width * spp for r0, r1 in bands]

    def test_a_small_or_one_sample_wide_frame_is_one_band(self):
        for budget in (1, 4, 24):
            self.assertEqual(self.sizes(64, 48, 4, budget), [12288])
            self.assertEqual(self.sizes(1, 100000, 1, budget), [100000])
            # 16384 samples split
            self.assertEqual(self.sizes(64, 64, 4, budget), [8192, 8192])

    def test_two_bands_a_thread_of_8192_to_65536_samples(self):
        # one thread: bands as big as 65536 allows, in whole rows
        self.assertEqual(self.sizes(400, 300, 4, 1), [64000] * 7 + [32000])
        # two bands a thread
        self.assertEqual(self.sizes(250, 100, 4, 2), [25000] * 4)
        # but no smaller than 8192
        self.assertEqual(self.sizes(128, 64, 4, 4), [8192] * 4)
        # a row of more than 65536 samples is a band of its own
        self.assertEqual(self.sizes(20000, 3, 4, 1), [80000] * 3)

    def test_render_uses_them(self):
        s = _scene("cube")
        for resolution, threads, want in (((64, 48), None, 1), ((400, 300), 1, 8)):
            call = functools.partial(
                s.render, resolution=resolution, samples_per_pixel=4, return_depth=True
            )
            self.assertEqual(self.check(call, [(None, threads)], f"{resolution}"), [want])


# -- the one-row rule -------------------------------------------------------------------

W, H, AOV = 96, 72, 35.0
CAM = look_at((0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
EYE = CAM[:3, 3]

# a rotation about a skew axis times a non-uniform scale, a translation with
# every digit used: matmuls whose rounding shows
_Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))
SKEW  = _Q * np.array([1.37, 0.71, 1.13])
SHIFT = np.array([0.1234567, -0.2345678, 0.3456789])


def _sample_dir(px, py, spp):
    dirs = _generate_camera_rays(CAM, AOV, W, H, int(round(spp ** 0.5)))[1]
    return dirs[(py * W + px) * spp]


def _boxes(name, pixels, spp):
    """ONE Object: a tiny box on the first sample ray of each of *pixels*,
    5 units out, its points in object space under SKEW and SHIFT"""
    cube, cube_uv = _cube()
    pts, idx, cnt, upts, uidx = [], [], [], [], []
    for k, (px, py) in enumerate(pixels):
        world = EYE + _sample_dir(px, py, spp) * 5.0 + cube.points * 0.004
        idx.append(cube.indices + k * 8)
        uidx.append(cube_uv.indices + k * len(cube_uv.points))
        pts.append((world - SHIFT) @ np.linalg.inv(SKEW).T)
        upts.append(cube_uv.points)
        cnt.append(cube.counts)
    mesh = MeshData(
        indices = np.concatenate(idx).astype(np.int32),
        counts  = np.concatenate(cnt),
        points  = np.concatenate(pts),
    )
    uv = UVData(
        indices = np.concatenate(uidx).astype(np.int32),
        counts  = np.concatenate(cnt),
        points  = np.concatenate(upts),
    )
    M              = np.eye(4)
    M[:3, :3]      = SKEW
    M[:3, 3]       = SHIFT
    o              = Object(name=name, mesh=mesh, uv=uv, texture=_checker())
    o.world_matrix = M.T  # row-major for TransformData
    return o


# a case: the pixels each Object is hit in. In the last, the first Object's
# band is redone, and the second's one hit there stays padded: the frame
# hits it twice
SINGLE_HIT_CASES = {
    "one hit in the frame":            [[(40, 30)]],
    "one hit each, two bands":         [[(10, 5)], [(80, 66)]],
    "one hit and a big box":           [[(70, 12)]],
    "twelve objects, one hit each":    [[(7 + 7 * k, 3 + 6 * k)] for k in range(12)],
    "one mesh, one hit in many bands": [[(7 + 7 * k, 3 + 6 * k) for k in range(12)]],
    "one hit beside a mesh hit twice": [[(39, 7)], [(12, 4), (89, 38)]],
}


class TestSingleHit(BandCase):
    """T2: a band, or the whole frame, hitting a mesh once"""

    def test_single_hits(self):
        lights = [
            {"kind": "point", "position": (3.0, 4.0, 9.0), "intensity": 80.0},
            {"kind": "infinite", "direction": (0.3, -0.5, -1.0), "intensity": 0.7},
        ]
        padded = redone = 0
        real   = _tiles._band
        calls  = []

        def band(job, index, pads):
            calls.append((index, numba.get_num_threads()))
            return real(job, index, pads)

        for name, objects in SINGLE_HIT_CASES.items():
            for spp in (1, 4):
                sc = Scene(name)
                for i, pixels in enumerate(objects):
                    sc.append(_boxes(f"boxes{i}", pixels, spp))
                if name == "one hit and a big box":
                    box, box_uv = _cube()
                    sc.append(
                        Object(name="box", mesh=box, uv=box_uv, translate=(-1.5, -0.5, 0.0))
                    )
                call = functools.partial(
                    render,
                    scene             = sc,
                    camera_matrix     = CAM,
                    angle_of_view     = AOV,
                    resolution        = (W, H),
                    samples_per_pixel = spp,
                    return_depth      = True,
                    point_light       = lights,
                )
                ref = _whole(call)
                # bands of 997 samples, and the real band sizes, where the
                # frame at spp 1 (6912 samples) is one band: its single
                # hits shaded as one row each, as the whole frame shades them
                for samples in (997, None):
                    for threads in (None, 1):
                        what = f"{name} spp{spp} bands={samples} threads={threads}"
                        del calls[:]
                        with mock.patch.object(_tiles, "_band", band), _Bands(samples) as bands:
                            frame = call(threads=threads)
                        self.assertSameFrame(frame, ref, what)
                        stats  = bands.frames[-1]
                        counts = np.array(stats["counts"])
                        totals = counts.sum(axis=0)
                        if samples is None and spp == 1:
                            self.assertEqual(len(stats["bands"]), 1, what)
                            self.assertEqual(stats["redone"], [], what)
                            continue
                        self.assertGreater(len(stats["bands"]), 1, what)
                        padded += int(np.any((counts == 1) & (totals > 1)[None, :]))
                        redone += len(stats["redone"])
                        self.assertEqual(
                            len(stats["redone"]) > 0, bool(np.any(totals == 1)), what
                        )
                        # a redone band: after the others, with the whole budget
                        again = calls[len(stats["bands"]):]
                        self.assertEqual([i for i, _ in again], stats["redone"], what)
                        self.assertEqual({n for _, n in again} - {stats["budget"]}, set(), what)
        # the cases did hit the rule: a band hit once by a mesh the frame
        # hits more, and a mesh the whole frame hits once
        self.assertGreater(padded, 0)
        self.assertGreater(redone, 0)


# -- the cast for hits alone ----------------------------------------------------------------


class TestCastHits(unittest.TestCase):
    """T3: _tiles._cast_hits is MeshData.raycast's hit rows, bit for bit"""

    @staticmethod
    def rays(mesh, n=1500, seed=11):
        rng = np.random.default_rng(seed)
        lo, hi = mesh.points.min(0), mesh.points.max(0)
        centre = 0.5 * (lo + hi)
        size   = float(np.max(hi - lo))
        # from outside toward the mesh, with lengths 1e-3 to 1e3
        out_o  = centre + rng.standard_normal((n, 3)) * size * 2.0
        target = lo + rng.random((n, 3)) * (hi - lo)
        out_d  = (target - out_o) * 10.0 ** rng.uniform(-3, 3, (n, 1))
        # from inside outward: back faces
        in_o = centre + rng.standard_normal((n // 2, 3)) * size * 0.05
        in_d = rng.standard_normal((n // 2, 3))
        # pointing away: misses
        away_d = out_o[: n // 4] - centre
        return (
            np.concatenate([out_o, in_o, out_o[: n // 4]]),
            np.concatenate([out_d, in_d, away_d]),
        )

    def test_cast_hits_is_raycast(self):
        head = _assets()[0]
        cube = _cube()[0]
        for mesh, label in ((cube, "cube"), (head, "character")):
            origins, directions = self.rays(mesh)
            for method in ("bilinear", "bezier"):
                for twosided in (True, False):
                    entry = {
                        "mesh":          mesh,
                        "uv":            None,
                        "sample_method": method,
                        "twosided":      twosided,
                        "world_matrix":  np.eye(4),
                    }
                    what = f"{label} {method} twosided={twosided}"
                    prep = _tiles._prepare(entry)
                    hit, hits = _tiles._cast_hits(prep, origins, directions)
                    ref = mesh.raycast(
                        origins, directions, method=method, forward_only=True, twosided=twosided
                    )
                    np.testing.assert_array_equal(hit, ref.hit, what)
                    self.assertTrue(0 < hit.sum() < hit.size, what)
                    indices, weights, projections, normals, distances = hits
                    for got, want in (
                        (indices, ref.indices[hit]),
                        (weights, ref.weights[hit]),
                        (projections, ref.projections[hit]),
                        (normals, ref.normals[hit]),
                        (distances, ref.distances[hit]),
                    ):
                        self.assertEqual(got.dtype, want.dtype, what)
                        self.assertEqual(got.tobytes(), want.tobytes(), what)
            if label == "cube":
                self.assertIsNone(prep.bvh)
            else:
                self.assertIsNotNone(prep.bvh)

    def test_nothing_hit(self):
        mesh = _cube()[0]
        entry = {
            "mesh":          mesh,
            "uv":            None,
            "sample_method": "bilinear",
            "twosided":      True,
            "world_matrix":  np.eye(4),
        }
        hit, hits = _tiles._cast_hits(
            _tiles._prepare(entry), np.full((5, 3), 10.0), np.ones((5, 3))
        )
        self.assertFalse(hit.any())
        self.assertIsNone(hits)


# -- BLAS only under the lock ---------------------------------------------------------------


def _blas_calls_node(node):
    """``(line, what)`` if *node* itself is a BLAS call: ``@``, ``dot`` and
    its kin, a vector norm (no ``axis``: a dot), anything of ``np.linalg``
    but a norm, an einsum allowed to optimise (it calls tensordot)"""
    if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
        return [(node.lineno, "@")]
    if not isinstance(node, ast.Call):
        return []
    name     = getattr(node.func, "attr", getattr(node.func, "id", None))
    dotted   = ast.unparse(node.func)
    keywords = {k.arg for k in node.keywords}
    if name in BLAS_OPS:
        return [(node.lineno, name)]
    if name == "norm" and "axis" not in keywords:
        return [(node.lineno, "norm without axis")]
    if "linalg" in dotted and name != "norm":
        return [(node.lineno, dotted)]
    if name == "einsum" and "optimize" in keywords:
        return [(node.lineno, "einsum(optimize=)")]
    return []


def _blas_calls(tree):
    """every BLAS call in *tree*"""
    return [found for node in ast.walk(tree) for found in _blas_calls_node(node)]


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def _locked(tree):
    """the nodes inside a ``with _BLAS`` block"""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            getattr(item.context_expr, "id", None) == "_BLAS" for item in node.items
        ):
            for child in node.body:
                inside.update(id(n) for n in ast.walk(child))
    return inside


class TestBlasGuard(unittest.TestCase):
    """T4: every BLAS call a band makes holds _tiles._BLAS"""

    def test_tiles_calls_blas_only_through_the_lock(self):
        tree    = ast.parse(inspect.getsource(_tiles))
        locked  = _locked(tree)
        allowed = set()
        for fn in ("_matmul", "_prepare"):
            allowed.update(id(n) for n in ast.walk(_function(tree, fn)))
        outside = [
            found
            for node in ast.walk(tree)
            if id(node) not in allowed
            for found in _blas_calls_node(node)
        ]
        self.assertEqual(outside, [])
        # _matmul and _prepare make theirs under the lock
        for fn in ("_matmul", "_prepare"):
            calls = [n for n in ast.walk(_function(tree, fn)) if _blas_calls_node(n)]
            self.assertTrue(calls, fn)
            for n in calls:
                self.assertIn(id(n), locked, f"{fn} line {n.lineno}")

    def test_raytracer_functions_the_bands_call(self):
        tree = ast.parse(inspect.getsource(_tiles))
        used = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and getattr(n.value, "id", None) == "_rt"
        }
        free = {
            "_interpolate_uvs",
            "_sample_texture_bilinear",
            "_resolve_msaa",
            "_resolve_msaa_depth",
            "_composite_over",
            "_to_uint8_image",
        }
        self.assertEqual(used, free | {"_shade_one_light"})
        for name in free:
            source = textwrap.dedent(inspect.getsource(getattr(raytracer, name)))
            self.assertEqual(_blas_calls(ast.parse(source)), [], name)
        from cgmath.render._numba._texture import _sample_texture_bilinear_fast

        source = textwrap.dedent(inspect.getsource(_sample_texture_bilinear_fast.py_func))
        self.assertEqual(_blas_calls(ast.parse(source)), [])

        # _rt._shade_one_light: only in the wrapper, an infinite light's
        # under the lock; a point light's makes no BLAS call
        wrapper = _function(tree, "_shade_one_light")
        inside  = {id(n) for n in ast.walk(wrapper)}
        locked  = _locked(tree)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_shade_one_light"
        ]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(id(n) in inside for n in calls))
        self.assertEqual(sum(id(n) in locked for n in calls), 1)
        source = textwrap.dedent(inspect.getsource(raytracer._shade_one_light))
        start  = source.rindex("\n", 0, source.index("# default: point light")) + 1
        point  = textwrap.dedent(source[start:])
        self.assertIn("np.einsum", point)
        self.assertEqual(_blas_calls(ast.parse(point)), [])
        # nothing in _tiles calls the unlocked _shade_lights
        self.assertNotIn("_shade_lights", used)

    def test_the_intersection_kernels_make_no_blas_call(self):
        from cgmath.geometry.utils._numba import _bilinear

        tree = ast.parse(inspect.getsource(_bilinear))
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        pending = [
            "_intersect_mesh_parallel",
            "_intersect_mesh_parallel_bvh",
            "_intersect_bezier_mesh_parallel",
            "_intersect_bezier_mesh_parallel_bvh",
        ]
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            self.assertEqual(_blas_calls(defs[name]), [], name)
            for n in ast.walk(defs[name]):
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) in defs:
                    pending.append(n.func.id)
        self.assertGreater(len(seen), 4)


# -- many threads, many times ------------------------------------------------------------------


class TestStress(BandCase):
    """T5"""

    def test_twenty_renders_on_every_core(self):
        call = functools.partial(
            _scene("multi").render, resolution=(160, 120), samples_per_pixel=4, return_depth=True
        )
        ref = _whole(call)
        with _Bands(997) as bands:
            for i in range(20):
                self.assertSameFrame(call(threads=max(4, os.cpu_count() or 1)), ref, f"render {i}")
        self.assertGreater(len(bands.frames[-1]["bands"]), 50)


# -- threads=1, the calling thread, numba's thread count ----------------------------------------------


class TestThreads(BandCase):
    """T6 and T7"""

    def call(self, resolution=(120, 80), **kw):
        return _scene("head").render(
            resolution=resolution, samples_per_pixel=4, return_depth=True, **kw
        )

    def test_one_thread_is_the_calling_thread(self):
        ref = _whole(self.call)
        with mock.patch.object(_tiles, "_get_pool", side_effect=AssertionError("pool")):
            with _Bands(997) as bands:
                frame = self.call(threads=1)
        self.assertSameFrame(frame, ref, "threads=1")
        stats = bands.frames[-1]
        self.assertGreater(len(stats["bands"]), 1)
        self.assertEqual(set(stats["idents"]), {threading.get_ident()})
        self.assertEqual(stats["runners"], 1)

    def test_several_threads_share_the_bands(self):
        with _Bands(997) as bands:
            self.call(threads=4)
        stats = bands.frames[-1]
        self.assertEqual(
            stats["budget"], min(4, numba.config.NUMBA_NUM_THREADS, os.cpu_count() or 1)
        )
        self.assertLessEqual(len(set(stats["idents"])), stats["runners"])
        self.assertIn(threading.get_ident(), stats["idents"])

    def bands_seen(self, threads, barrier=None, samples=997, n_bands=40, resolution=(120, 80)):
        """renders *n_bands* bands with *threads*; returns the frame's
        bookkeeping and, per band, the thread that ran it and its numba
        thread count. With *barrier*, the first ``barrier.parties`` bands
        wait for each other: they only pass when that many threads run
        bands at once"""
        real = _tiles._band
        seen = {}

        def band(job, index, pads):
            if index not in seen:  # not a redone band
                seen[index] = (threading.get_ident(), numba.get_num_threads())
                if barrier is not None and index < barrier.parties:
                    barrier.wait()
            return real(job, index, pads)

        with mock.patch.object(_tiles, "_band", band), _Bands(samples) as bands:
            self.call(resolution=resolution, threads=threads)
        stats = bands.frames[-1]
        self.assertEqual(len(stats["bands"]), n_bands)
        self.assertEqual(sorted(seen), list(range(n_bands)))
        return stats, seen

    def test_one_thread_casts_with_one_numba_thread(self):
        stats, seen = self.bands_seen(1)
        self.assertEqual(stats["runners"], 1)
        self.assertEqual(set(seen.values()), {(threading.get_ident(), 1)})

    def test_fewer_bands_than_threads_share_the_whole_budget(self):
        # one band (12288 samples at the real band sizes): the calling
        # thread, with every numba thread of the budget
        for threads in (None, 4):
            budget = _tiles.thread_budget(threads)
            stats, seen = self.bands_seen(threads, samples=None, n_bands=1, resolution=(64, 48))
            self.assertEqual(stats["runners"], 1, threads)
            self.assertEqual(seen, {0: (threading.get_ident(), budget)}, threads)
        # four bands on eight threads: four at once, two numba threads each
        if _layer() not in ("omp", "tbb") or _tiles.thread_budget(8) < 8:
            self.skipTest("bands run one after another, or fewer than 8 CPU threads")
        stats, seen = self.bands_seen(8, threading.Barrier(4, timeout=60), samples=9600, n_bands=4)
        self.assertEqual(stats["runners"], 4)
        self.assertEqual(len({ident for ident, _ in seen.values()}), 4)
        self.assertEqual({n for _, n in seen.values()}, {2})

    def test_the_bands_run_side_by_side_within_the_budget(self):
        if _layer() not in ("omp", "tbb"):
            self.skipTest("bands run one after another under workqueue")
        for threads in (4, None):
            budget = _tiles.thread_budget(threads)
            if budget < 2:
                self.skipTest("one CPU thread")
            runners = min(budget, 40)
            stats, seen = self.bands_seen(threads, threading.Barrier(runners, timeout=60))
            self.assertEqual(stats["budget"], budget, threads)
            self.assertEqual(stats["runners"], runners, threads)
            # the first bands ran at once, each in its own thread
            self.assertEqual(len({seen[i][0] for i in range(runners)}), runners, threads)
            # runners times the numba threads of each, within the budget
            per_band = {n for _, n in seen.values()}
            self.assertEqual(per_band, {max(1, budget // runners)}, threads)
            self.assertLessEqual(runners * max(per_band), budget, threads)

    def test_numba_thread_count_is_restored(self):
        before = numba.get_num_threads()
        try:
            numba.set_num_threads(min(5, numba.config.NUMBA_NUM_THREADS))
            mine = numba.get_num_threads()
            for threads in (1, 3, None):
                with _Bands(997):
                    self.call(threads=threads)
                self.assertEqual(numba.get_num_threads(), mine, threads)
            # None: the calling thread's count is the budget
            with _Bands(997) as bands:
                self.call()
            self.assertEqual(bands.frames[-1]["budget"], mine)
        finally:
            numba.set_num_threads(before)

    def test_an_unknown_threading_layer_runs_band_0_alone_first(self):
        ref   = _whole(self.call)
        layer = _layer()
        calls = []
        real  = _tiles._band
        nbs   = {}

        def unknown_once():
            calls.append(1)
            return None if len(calls) == 1 else layer

        def band(job, index, pads):
            nbs.setdefault(index, numba.get_num_threads())
            return real(job, index, pads)

        with mock.patch.object(_tiles, "_threading_layer", side_effect=unknown_once):
            with mock.patch.object(_tiles, "_band", band), _Bands(997) as bands:
                frame = self.call(threads=4)
        self.assertSameFrame(frame, ref, "layer unknown")
        self.assertEqual(bands.frames[-1]["idents"][0], threading.get_ident())
        # alone, with the whole budget
        self.assertEqual(nbs[0], _tiles.thread_budget(4))
        # the rest side by side
        if layer in ("omp", "tbb") and _tiles.thread_budget(4) > 1:
            self.assertGreater(bands.frames[-1]["runners"], 1)

    def test_a_forked_child_starts_a_fresh_pool_and_locks(self):
        _tiles._get_pool()
        saved = (_tiles._BLAS, _tiles._POOL, _tiles._POOL_LOCK)
        saved[0].acquire()  # held by another thread when the process forks
        try:
            _tiles._after_fork_in_child()
            self.assertIsNone(_tiles._POOL)
            self.assertFalse(_tiles._BLAS.locked())
            self.assertFalse(_tiles._POOL_LOCK.locked())
            self.assertIsNot(_tiles._BLAS, saved[0])
        finally:
            saved[0].release()
            _tiles._BLAS, _tiles._POOL, _tiles._POOL_LOCK = saved

    def test_bad_threads(self):
        for bad in (0, -1, 1.5, True, False, "4", np.bool_(True)):
            with self.assertRaises(ValueError, msg=repr(bad)) as raised:
                self.call(threads=bad)
            self.assertIn("threads must be None or a positive int", str(raised.exception))
        self.assertSameFrame(self.call(threads=np.int64(2)), _whole(self.call), "np.int64")

    @unittest.skipIf(IN_MAYA, "a fresh python, not mayapy")
    def test_no_pool_with_one_thread_in_a_fresh_process(self):
        out = _child(
            """
            with mock.patch.object(_tiles, "_MIN_BAND", 997), \\
                 mock.patch.object(_tiles, "_MAX_BAND", 997):
                render(cube, resolution=(64, 48), samples_per_pixel=4, threads=1)
                render(cube, resolution=(400, 300), samples_per_pixel=1, threads=1)
            names = [t.name for t in threading.enumerate()]
            print("POOL", _tiles._POOL is None, any(n.startswith("cgmath-render") for n in names))
            """
        )
        self.assertIn("POOL True False", out)


# -- errors -----------------------------------------------------------------------------------------


class TestErrors(BandCase):
    """T8: the whole frame's errors, raised from bands"""

    def test_errors(self):
        mesh = _cube()[0]
        cases = {
            "point light without a position": dict(point_light={"kind": "point"}),
            "infinite light without a direction": dict(point_light={"kind": "infinite"}),
            "background not RGBA": dict(background=(0.0, 0.0, 0.0)),
            "background out of range": dict(background=(2.0, 0.0, 0.0, 1.0)),
        }
        for what, kw in cases.items():
            call = functools.partial(render, mesh, resolution=(64, 48), samples_per_pixel=4, **kw)
            with self.assertRaises(ValueError) as ref:
                _whole(call)
            for threads in (None, 1):
                with _Bands(37):
                    with self.assertRaises(type(ref.exception)) as got:
                        call(threads=threads)
                self.assertEqual(str(got.exception), str(ref.exception), what)

    def test_the_lowest_failing_band_raises(self):
        mesh = _cube()[0]
        boom = []
        real = _tiles._band

        def band(job, index, pads):
            if index in (3, 5, 9):
                boom.append(index)
                raise RuntimeError(f"band {index}")
            return real(job, index, pads)

        with mock.patch.object(_tiles, "_band", side_effect=band):
            for threads in (None, 1):
                with _Bands(37):
                    with self.assertRaises(RuntimeError) as got:
                        render(mesh, resolution=(64, 48), threads=threads)
                self.assertEqual(str(got.exception), "band 3")
        # one thread stops at the first
        self.assertEqual(boom[-1], 3)

    def side_by_side(self):
        if not _side_by_side(4):
            self.skipTest("bands run one after another")

    def test_the_lower_of_two_failing_bands_raises(self):
        # band 5 fails while band 3 runs, then band 3 fails
        self.side_by_side()
        real = _tiles._band
        five = threading.Event()

        def band(job, index, pads):
            if index == 5:
                five.set()
                raise RuntimeError("band 5")
            if index == 3:
                if not five.wait(60):
                    raise AssertionError("band 5 did not run beside band 3")
                raise RuntimeError("band 3")
            return real(job, index, pads)

        with mock.patch.object(_tiles, "_band", band), _Bands(37):
            with self.assertRaises(RuntimeError) as got:
                render(_cube()[0], resolution=(64, 48), threads=4)
        self.assertEqual(str(got.exception), "band 3")

    def test_no_band_starts_after_a_failure(self):
        # band 0 fails while the others run; they finish, and stop there
        self.side_by_side()
        real   = _tiles._band
        failed = threading.Event()
        taken  = []
        fail   = _tiles._Queue.fail

        def recorded(queue, index, exc):
            fail(queue, index, exc)
            failed.set()

        def band(job, index, pads):
            taken.append(index)
            if index == 0:
                raise RuntimeError("band 0")
            if not failed.wait(60):
                raise AssertionError("band 0 did not fail")
            return real(job, index, pads)

        with mock.patch.object(_tiles, "_band", band), mock.patch.object(
            _tiles._Queue, "fail", recorded
        ), _Bands(37) as bands:
            with self.assertRaises(RuntimeError) as got:
                render(_cube()[0], resolution=(64, 48), threads=4)
        self.assertEqual(str(got.exception), "band 0")
        # one band each at most, of 48
        self.assertLessEqual(len(taken), _tiles.thread_budget(4))

    def test_several_bad_inputs(self):
        # the background first, then each entry's own checks in scene
        # order, then the lights (render()'s Raises)
        cube   = _cube()[0]
        points = [[0, 0, 0], [1, 0, 0], [1.3, 1, 0], [0.5, 1.6, 0], [-0.3, 1, 0]]
        ngon = MeshData(
            indices = np.arange(5, dtype=np.int32),
            counts  = np.array([5], np.int32),
            points  = np.array(points, float),
        )
        light = {"kind": "point"}
        bad   = (2.0, 0.0, 0.0, 1.0)
        later = [{"mesh": cube}, {"mesh": cube, "sample_method": "x"}]
        cases = [
            (dict(mesh=cube, point_light=light, background=bad), ValueError, "background"),
            (dict(mesh=ngon, background=bad), ValueError, "background"),
            (
                dict(scene=[{"mesh": cube}, {"mesh": ngon}], point_light=light),
                NotImplementedError,
                "n-gon",
            ),
            (dict(scene=later, point_light=light), ValueError, "SampleMethod"),
        ]
        for kw, kind, text in cases:
            for threads in (None, 1):
                # one band, and many
                for samples in (None, 37):
                    with _Bands(samples), self.assertRaises(kind) as got:
                        render(resolution=(64, 48), threads=threads, **kw)
                    self.assertIn(text, str(got.exception), (kw, threads, samples))

    def test_a_singular_world_matrix(self):
        # an Object scaled to 0 raises the whole frame's error, numpy's
        # LinAlgError (a ValueError only under numpy 2), and before its own
        # n-gon faces (render()'s Raises)
        cube, uv = _cube()
        points = [[0, 0, 0], [1, 0, 0], [1.3, 1, 0], [0.5, 1.6, 0], [-0.3, 1, 0]]
        ngon = MeshData(
            indices = np.arange(5, dtype=np.int32),
            counts  = np.array([5], np.int32),
            points  = np.array(points, float),
        )
        for other in (cube, ngon):
            s = Scene("singular")
            s.append(Object(name="cube", mesh=cube, uv=uv))
            s.append(Object(name="flat", mesh=other, scale=(0.0, 1.0, 1.0)))
            call = functools.partial(s.render, resolution=(64, 48), samples_per_pixel=4)
            with self.assertRaises(np.linalg.LinAlgError) as ref:
                _whole(call)
            for threads in (None, 1):
                for samples in (None, 37):
                    with _Bands(samples), self.assertRaises(np.linalg.LinAlgError) as got:
                        call(threads=threads)
                    what = f"{other.counts[0]}-gons threads={threads} bands={samples}"
                    self.assertEqual(str(got.exception), str(ref.exception), what)

    def test_a_bad_input_while_a_ctrl_c_is_handled(self):
        # a render in an except KeyboardInterrupt: or a finally: block: its
        # own error, not a Ctrl-C (that KeyboardInterrupt is the error's
        # __context__, in the calling thread)
        cube   = _cube()[0]
        points = [[0, 0, 0], [1, 0, 0], [1.3, 1, 0], [0.5, 1.6, 0], [-0.3, 1, 0]]
        ngon = MeshData(
            indices = np.arange(5, dtype=np.int32),
            counts  = np.array([5], np.int32),
            points  = np.array(points, float),
        )
        cases = {
            "a light without a position": dict(mesh=cube, point_light={"kind": "point"}),
            "n-gon faces": dict(mesh=ngon),
            "a bad sample_method": dict(scene=[{"mesh": cube, "sample_method": "nearest"}]),
        }

        def raised(call, where):
            """what *call* raises while a KeyboardInterrupt is handled"""
            try:
                try:
                    raise KeyboardInterrupt
                except KeyboardInterrupt:
                    if where == "except":
                        call()
                    raise
                finally:
                    if where == "finally":
                        call()
            except BaseException as exc:
                return exc
            return None

        for what, kw in cases.items():
            call = functools.partial(render, resolution=(400, 300), **kw)
            ref  = raised(lambda: _whole(call), "except")
            self.assertNotIsInstance(ref, KeyboardInterrupt, what)
            for threads in (None, 1):
                for samples in (None, 37):
                    for where in ("except", "finally"):
                        with _Bands(samples):
                            got = raised(functools.partial(call, threads=threads), where)
                        msg = f"{what} threads={threads} bands={samples} in {where}"
                        self.assertIs(type(got), type(ref), msg)
                        self.assertEqual(str(got), str(ref), msg)


class TestFloatingPoint(BandCase):
    """the caller's numpy floating-point policy, in every band"""

    def test_the_callers_errstate_holds_in_every_band(self):
        # a point light on the face the camera sees: the frame's centre
        # sample hits it, and its shading divides by zero. A pool thread
        # starts at numpy's default policy, "warn"
        call = functools.partial(
            render,
            _cube()[0],
            camera_matrix = look_at((0.0, 0.0, 3.0), (0.0, 0.0, 0.0)),
            resolution    = (401, 301),
            point_light   = {"kind": "point", "position": (0.0, 0.0, 0.5)},
            return_depth  = True,
        )
        with np.errstate(all="raise"), self.assertRaises(FloatingPointError) as ref:
            _whole(call)
        with np.errstate(all="ignore"):
            whole = _whole(call)
        # all="call": the same errors reach the caller's callback
        reported = set()
        with np.errstate(all="call", call=lambda err, flag: reported.add(err)):
            _whole(call)
        self.assertIn("divide by zero", reported)
        for samples in (None, 997):
            for threads in (None, 1):
                what = f"bands={samples} threads={threads}"
                with _Bands(samples) as bands:
                    with np.errstate(all="raise"), self.assertRaises(FloatingPointError) as got:
                        call(threads=threads)
                    self.assertEqual(str(got.exception), str(ref.exception), what)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with np.errstate(all="ignore"):
                            frame = call(threads=threads)
                    seen = set()
                    with np.errstate(all="call", call=lambda err, flag: seen.add(err)):
                        call(threads=threads)
                self.assertSameFrame(frame, whole, what)
                self.assertEqual([str(w.message) for w in caught], [], what)
                self.assertEqual(seen, reported, what)
                self.assertGreater(len(bands.frames[-1]["bands"]), 1, what)


# -- Ctrl-C, interpreter shutdown --------------------------------------------------------------------


def _numba_interrupted(*args, **kwargs):
    """what a numba kernel raises when a Ctrl-C lands in Python code it
    calls back into (numba's _numba_unpickle)"""
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt as exc:
        raise SystemError("returned a result with an exception set") from exc


class TestInterrupt(BandCase):
    """T11: Ctrl-C raises KeyboardInterrupt; a render at exit renders"""

    def test_a_ctrl_c_numba_turned_into_a_system_error(self):
        call = functools.partial(
            _scene("head").render, resolution=(120, 80), samples_per_pixel=4, return_depth=True
        )
        ref  = _whole(call)
        real = _tiles._band

        def band(job, index, pads):
            if index == 2:
                _numba_interrupted()
            return real(job, index, pads)

        for threads in (None, 1):
            for where, patch in (
                ("a band", mock.patch.object(_tiles, "_band", band)),
                ("the per-mesh work", mock.patch.object(_tiles, "_prepare", _numba_interrupted)),
            ):
                with patch, _Bands(997):
                    with self.assertRaises(KeyboardInterrupt, msg=where) as got:
                        call(threads=threads)
                self.assertIsInstance(got.exception.__cause__, SystemError, where)
                # and the next render is whole
                with _Bands(997):
                    self.assertSameFrame(call(threads=threads), ref, f"{where} {threads}")

    def test_a_ctrl_c_while_the_runners_are_submitted(self):
        # the pool starts its threads in submit, so a Ctrl-C can land there:
        # the runners queued already stop at their next band, and render()
        # raises once the bands in flight end. Also when the interrupted
        # submit had queued its runner (its future lost), which is in its
        # 2nd band: no future to wait for, only the count of bands in flight
        if not _side_by_side(4) or _tiles.thread_budget(4) < 3:
            self.skipTest("bands run one after another, or two at most")
        pool    = _tiles._get_pool()
        real    = _tiles._band
        lock    = threading.Lock()
        running = set()
        started = []
        lost    = []

        def band(job, index, pads):
            with lock:
                running.add(index)
                started.append(threading.get_ident())
            try:
                # the lost runner's band outlasts the others
                time.sleep(0.5 if threading.get_ident() in lost else 0.05)
                return real(job, index, pads)
            finally:
                with lock:
                    running.discard(index)

        def until(ready, what):
            deadline = time.monotonic() + 60
            while not ready():
                if time.monotonic() > deadline:
                    raise AssertionError(what)
                time.sleep(0.001)

        class Interrupted:
            """the 2nd submit raises KeyboardInterrupt once the 1st runner
            took a band; with *queue_it*, once the runner it queued is in
            its 2nd band"""

            def __init__(self, queue_it):
                self.queue_it = queue_it
                self.n        = 0

            def submit(self, fn, *args):
                self.n += 1
                if self.n != 2:
                    return pool.submit(fn, *args)
                until(lambda: started, "the 1st runner took no band")
                if self.queue_it:

                    def runner():
                        lost.append(threading.get_ident())
                        fn(*args)

                    pool.submit(runner)
                    until(
                        lambda: lost and started.count(lost[0]) == 2,
                        "the queued runner took no 2nd band",
                    )
                raise KeyboardInterrupt

        for queue_it in (False, True):
            del started[:], lost[:]
            with mock.patch.object(_tiles, "_band", band), mock.patch.object(
                _tiles, "_get_pool", lambda: Interrupted(queue_it)
            ), _Bands(997):
                with self.assertRaises(KeyboardInterrupt, msg=queue_it):
                    _scene("head").render(resolution=(120, 80), samples_per_pixel=4, threads=4)
                with lock:
                    left, n = set(running), len(started)
                time.sleep(0.3)
            self.assertEqual(left, set(), f"bands running after render() raised, {queue_it}")
            self.assertEqual(len(started), n, f"bands started after render() raised, {queue_it}")
            self.assertGreater(n, 0, queue_it)

    def test_a_ctrl_c_in_the_calling_threads_band_stops_the_runners(self):
        # a Ctrl-C landing in the calling thread's own band (Python or numpy
        # code, not a numba callback): the pool's runners stop at their next
        # band, and render() raises once the bands in flight end
        if not _side_by_side(4) or _tiles.thread_budget(4) < 3:
            self.skipTest("bands run one after another, or two at most")
        real    = _tiles._band
        caller  = threading.get_ident()
        lock    = threading.Lock()
        running = set()
        started = []
        mine    = []
        pooled  = []

        def band(job, index, pads):
            if threading.get_ident() == caller:
                mine.append(index)
                if len(mine) == 3:
                    # once a runner of the pool took a band
                    deadline = time.monotonic() + 60
                    while not pooled:
                        if time.monotonic() > deadline:
                            raise AssertionError("no runner of the pool took a band")
                        time.sleep(0.001)
                    raise KeyboardInterrupt
            else:
                pooled.append(index)
            with lock:
                started.append(index)
                running.add(index)
            try:
                return real(job, index, pads)
            finally:
                with lock:
                    running.discard(index)

        with mock.patch.object(_tiles, "_band", band), _Bands(997):
            n_bands = len(_tiles._bands(120, 80, 4, _tiles.thread_budget(4)))
            with self.assertRaises(KeyboardInterrupt):
                _scene("head").render(resolution=(120, 80), samples_per_pixel=4, threads=4)
            with lock:
                left, n = set(running), len(started)
            time.sleep(0.3)
        self.assertEqual(left, set(), "bands running after render() raised")
        self.assertEqual(len(started), n, "bands started after render() raised")
        # the runners did not render the rest of the frame first
        self.assertLess(n, n_bands // 2)

    def test_runners_the_pool_never_starts_are_not_waited_for(self):
        # other renders hold every thread of the pool: the runners this
        # render queues there do not start. The calling thread renders every
        # band, then returns, or raises a Ctrl-C landing in its 3rd band,
        # without waiting for those renders to end (here, 30 s on)
        if not _side_by_side(None):
            self.skipTest("bands run one after another")
        call = functools.partial(
            _scene("head").render, resolution=(120, 80), samples_per_pixel=4, return_depth=True
        )
        ref  = _whole(call)
        real = _tiles._band

        class Busy:
            """a pool whose threads other renders hold for 30 s; then its
            threads run the runners queued, as ThreadPoolExecutor does"""

            def __init__(self):
                self.queued = []
                self.ran    = []
                self.timer  = threading.Timer(30, self.free)

            def submit(self, fn, *args):
                future = Future()
                self.queued.append((future, fn, args))
                return future

            def free(self):
                for future, fn, args in self.queued:
                    if future.done() or future.running():
                        continue
                    if future.set_running_or_notify_cancel():
                        self.ran.append(future)
                        try:
                            future.set_result(fn(*args))
                        except BaseException as exc:
                            future.set_exception(exc)

        def ctrl_c(job, index, pads):
            if index == 2:
                raise KeyboardInterrupt
            return real(job, index, pads)

        for interrupted in (False, True):
            busy = Busy()
            band = mock.patch.object(_tiles, "_band", ctrl_c if interrupted else real)
            pool = mock.patch.object(_tiles, "_get_pool", lambda: busy)
            busy.timer.start()
            try:
                with band, pool, _Bands(997) as bands:
                    if interrupted:
                        with self.assertRaises(KeyboardInterrupt):
                            call()
                    else:
                        frame = call()
            finally:
                busy.timer.cancel()
                busy.timer.join()
                busy.free()
            self.assertEqual(busy.ran, [], f"render() waited for the other renders, {interrupted}")
            self.assertGreater(len(busy.queued), 0, interrupted)
            self.assertTrue(all(f.cancelled() for f, _, _ in busy.queued), interrupted)
            if not interrupted:
                self.assertSameFrame(frame, ref, "the runners queued")
                self.assertEqual(bands.frames[-1]["runners"], 1)
                self.assertEqual(set(bands.frames[-1]["idents"]), {threading.get_ident()})

    @unittest.skipIf(IN_MAYA, "a fresh python, not mayapy")
    def test_ctrl_c_during_a_render(self):
        # _thread.interrupt_main() is what Ctrl-C does; a render whose
        # bands make many short kernel calls gets most of them inside one
        out = _child(
            """
            import _thread, time
            from cgmath._tests.test_render_threads import _assets, _checker
            head, uv, _ = _assets()
            tex = _checker()
            outcomes = []
            for threads, resolution in ((None, (640, 480)), (1, (320, 240))):
                kw = dict(resolution=resolution, samples_per_pixel=4, return_depth=True,
                          threads=threads)
                ref  = render(head, uv, tex, **kw)
                full = float("inf")
                for _ in range(2):
                    t0 = time.perf_counter()
                    render(head, uv, tex, **kw)
                    full = min(full, time.perf_counter() - t0)
                for k in range(8):
                    timer = threading.Timer(full * (0.1 + 0.1 * k), _thread.interrupt_main)
                    try:
                        timer.start()
                        render(head, uv, tex, **kw)
                        try:
                            time.sleep(full + 1.0)  # the render won: it lands here
                        except KeyboardInterrupt:
                            pass
                        outcomes.append((threads, "finished first"))
                    except KeyboardInterrupt:
                        outcomes.append((threads, "KeyboardInterrupt"))
                    except BaseException as exc:
                        outcomes.append((threads, type(exc).__name__ + ": " + str(exc)[:200]))
                    finally:
                        timer.cancel()
                again = render(head, uv, tex, **kw)
                outcomes.append((threads, again.array.tobytes() == ref.array.tobytes()
                                 and again.depth.tobytes() == ref.depth.tobytes()))
            print("OUTCOMES", outcomes)
            """
        )
        line     = [x for x in out.splitlines() if x.startswith("OUTCOMES ")][-1]
        outcomes = ast.literal_eval(line[len("OUTCOMES "):])
        for threads in (None, 1):
            got = [o for t, o in outcomes if t == threads]
            self.assertTrue(got[-1], f"threads={threads}: the next render differs")
            self.assertLessEqual(
                set(got[:-1]), {"KeyboardInterrupt", "finished first"}, f"threads={threads}"
            )
            self.assertIn("KeyboardInterrupt", got, f"threads={threads}")

    @unittest.skipIf(IN_MAYA, "a fresh python, not mayapy")
    def test_a_render_from_an_atexit_handler(self):
        # concurrent.futures takes no new work once the interpreter shuts
        # down: the calling thread renders the bands
        out = _child(
            """
            import atexit
            kw    = dict(resolution=(320, 240), samples_per_pixel=1, return_depth=True)
            stats = []
            _tiles._stats_hook = stats.append
            ref = render(cube, **kw)
            print("SCRIPT", len(stats[-1]["bands"]), stats[-1]["runners"], numba.threading_layer())

            @atexit.register
            def last_frame():
                try:
                    # the calling thread alone, with the whole budget
                    real, nbs = _tiles._band, set()

                    def band(job, index, pads):
                        nbs.add(numba.get_num_threads())
                        return real(job, index, pads)

                    _tiles._band = band
                    frame = render(cube, **kw)
                    print("AT EXIT", len(stats[-1]["bands"]), stats[-1]["runners"],
                          frame.array.tobytes() == ref.array.tobytes(),
                          frame.depth.tobytes() == ref.depth.tobytes(),
                          nbs == {stats[-1]["budget"]}, flush=True)
                except BaseException as exc:
                    print("AT EXIT", type(exc).__name__, exc, flush=True)
            """
        )
        script = [x.split() for x in out.splitlines() if x.startswith("SCRIPT ")][-1]
        if script[3] in ("omp", "tbb") and _tiles.thread_budget(None) > 1:
            self.assertGreater(int(script[2]), 1, out)  # the pool rendered in the script
        at_exit = [x for x in out.splitlines() if x.startswith("AT EXIT ")]
        self.assertEqual(at_exit, [f"AT EXIT {script[1]} 1 True True True"], out)
        self.assertGreater(int(script[1]), 1, out)


# -- numba's workqueue layer ---------------------------------------------------------------------


@unittest.skipIf(IN_MAYA, "a fresh python, not mayapy")
class TestWorkqueue(unittest.TestCase):
    """T9"""

    def test_workqueue_renders_its_bands_serially(self):
        out = _child(
            """
            with mock.patch.object(_tiles, "_MIN_BAND", 997), \\
                 mock.patch.object(_tiles, "_MAX_BAND", 997):
                frame = render(cube, resolution=(96, 64), samples_per_pixel=4, return_depth=True)
            with mock.patch.object(raytracer, "_WHOLE_FRAME", True):
                ref = render(cube, resolution=(96, 64), samples_per_pixel=4, return_depth=True)
            print("LAYER", numba.threading_layer())
            print("SAME", frame.array.tobytes() == ref.array.tobytes(),
                  frame.depth.tobytes() == ref.depth.tobytes(), _tiles._POOL is None)
            """,
            NUMBA_THREADING_LAYER="workqueue",
        )
        self.assertIn("LAYER workqueue", out)
        self.assertIn("SAME True True True", out)


def _child(body, **env):
    """runs *body* in a fresh python holding this cgmath, the cube built;
    returns its stdout"""
    code = textwrap.dedent(
        f"""
        import sys, threading
        sys.path.insert(0, {ROOT!r})
        from unittest import mock
        import numba
        from cgmath.render import _tiles, raytracer, render
        from cgmath._tests.test_render_threads import _cube
        cube = _cube()[0]
        """
    ) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text           = True,
        env            = dict(os.environ, **env),
        timeout        = 600,
    )
    if proc.returncode != 0:
        raise AssertionError(f"child failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return proc.stdout


# -- the sequence cache --------------------------------------------------------------------------


@unittest.skipIf(Image is None, "PIL is not installed")
class TestSequenceCache(unittest.TestCase):
    """T10: threads= is not part of a sequence's cache key"""

    def test_turntable_on_one_thread_replays_for_every_core(self):
        mesh, uv = _cube()
        obj = Object(name="cube", mesh=mesh, uv=uv, texture=_checker())
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "t.{frame:04d}.png")
            kw  = dict(n_frames=2, resolution=(32, 32), samples_per_pixel=1)
            obj.turntable(out, threads=1, **kw)
            original = Scene.render
            with mock.patch.object(Scene, "render", autospec=True, side_effect=original) as spy:
                obj.turntable(out, **kw)
                obj.turntable(out, threads=3, **kw)
            self.assertEqual(spy.call_count, 0)
            # a bad threads raises, kept frames or not
            with self.assertRaises(ValueError):
                obj.turntable(out, threads=0, **kw)

    @unittest.skipIf(pygltflib is None or trimesh is None, "pygltflib / trimesh are not installed")
    def test_an_animation_on_one_thread_replays_for_every_core(self):
        from cgmath._tests.test_object_skin import write_glb
        from cgmath.hierarchy import ClipData, HierarchyData

        with tempfile.TemporaryDirectory() as tmp:
            glb = os.path.join(tmp, "hero.glb")
            write_glb(glb, ["body"])
            rig   = HierarchyData.load_glb(glb)
            clip  = ClipData(rig, frames=2, fps=24.0)
            spine = list(rig.name).index("spine")
            for index in range(clip.frame_count):
                clip.frame         = index
                clip[spine].rotate = np.array([0.0, 0.0, 15.0 * index])
            scene = Scene("s")
            scene.append(Object.load_glb(glb))
            out = os.path.join(tmp, "a.{frame:04d}.png")
            kw  = dict(resolution=(32, 32), samples_per_pixel=1)
            scene.animate(clip, out, threads=1, **kw)
            original = Scene.render
            with mock.patch.object(Scene, "render", autospec=True, side_effect=original) as spy:
                scene.animate(clip, out, **kw)
                scene.animate(clip, out, threads=3, **kw)
            self.assertEqual(spy.call_count, 0)
            # a bad threads raises, kept frames or not
            with self.assertRaises(ValueError):
                scene.animate(clip, out, threads=0, **kw)


if __name__ == "__main__":
    unittest.main()
