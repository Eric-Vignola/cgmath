"""Tests for rendering a ``ClipData`` through ``Scene.animate``.

``animate`` is the complement of ``turntable``: one moves the scene and holds
the camera, the other moves the camera and holds the scene. They now share
their camera fitting and their encoder dispatch, so the turntable tests in
``test_scene`` are load bearing for this file too.

The thing most worth pinning here is that ``animate`` puts the scene back.
It is the only render entry point that has to deform anything, so it is the
only one that can leave your Objects in a state you did not ask for -- and
a half-restored scene shows up later as a second render quietly disagreeing
with the first, not as an error.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cgmath.geometry.deform import SkinDeformData
from cgmath.render.scene import (
    _clear_render_cache,
    _encode_frame_stream,
    _pose_sample_frames_for,
    Object,
    Scene,
)
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath._tests.test_skin_deform import branching_clip, pose_at, skinned_mesh

try:
    import pygltflib
except ImportError:
    pygltflib = None

try:
    import trimesh
except ImportError:
    trimesh = None

# rendered frames go through PIL (Frame.image); it is a separate extra from
# pygltflib / trimesh, so HAVE_GLB alone does not cover these tests
try:
    from PIL import Image
except ImportError:
    Image = None

if pygltflib is not None:
    from cgmath._tests.test_object_skin import write_glb


HAVE_GLB = pygltflib is not None and trimesh is not None

RES = {"resolution": (32, 32), "samples_per_pixel": 1}


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class AnimateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name="hero.glb"):
        return os.path.join(self.tmp.name, name)

    def out(self, name="anim.{frame:04d}.png"):
        return os.path.join(self.tmp.name, name)

    def skinned_glb(self, names=("body",)):
        path = self.path()
        write_glb(path, list(names))
        return path

    def clip(self, path, frames=5, fps=24.0, degrees_per_frame=15.0):
        """a clip that swings 'spine' a little further every frame"""
        rig   = HierarchyData.load_glb(path)
        clip  = ClipData(rig, frames=frames, fps=fps)
        spine = list(rig.name).index("spine")
        for index in range(clip.frame_count):
            clip.frame         = index
            clip[spine].rotate = np.array([0.0, 0.0, degrees_per_frame * index])
        return clip

    def scene_with(self, path):
        scene = Scene("s")
        scene.append(Object.load_glb(path))
        return scene


class TestAnimate(AnimateCase):
    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_writes_one_frame_per_clip_frame(self):
        path    = self.skinned_glb()
        scene   = self.scene_with(path)

        written = scene.animate(self.clip(path, frames=5), self.out(), **RES)

        self.assertEqual(len(written), 5)
        self.assertTrue(all(os.path.exists(p) for p in written))

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_the_frames_are_not_all_the_same_picture(self):
        """the whole point -- turntable renders the bind pose N times"""
        path    = self.skinned_glb()
        scene   = self.scene_with(path)

        written = scene.animate(self.clip(path), self.out(), **RES)

        blobs   = {open(p, "rb").read() for p in written}
        self.assertGreater(len(blobs), 1)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_the_scene_is_handed_back_undeformed(self):
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        obj   = list(scene.objects)[0]
        rest  = np.array(obj.mesh.points, copy=True)

        scene.animate(self.clip(path), self.out(), **RES)

        np.testing.assert_allclose(rest, obj.mesh.points, atol=1e-9)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_animating_twice_renders_the_same_thing(self):
        """follows from the restore, and is what would break silently"""
        path   = self.skinned_glb()
        scene  = self.scene_with(path)
        clip   = self.clip(path)

        first  = scene.animate(clip, self.out("a.{frame:04d}.png"), **RES)
        second = scene.animate(clip, self.out("b.{frame:04d}.png"), **RES)

        for a, b in zip(first, second):
            self.assertEqual(open(a, "rb").read(), open(b, "rb").read())

    def test_the_scene_is_restored_even_when_a_render_blows_up(self):
        """a failure part way through must not leave the scene deformed"""
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        obj   = list(scene.objects)[0]
        rest  = np.array(obj.mesh.points, copy=True)

        calls = {"n": 0}
        real  = Scene.render

        def explode(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("boom")
            return real(self, *args, **kwargs)

        with patch.object(Scene, "render", explode):
            with self.assertRaises(RuntimeError):
                scene.animate(self.clip(path), self.out(), **RES)

        np.testing.assert_allclose(rest, obj.mesh.points, atol=1e-9)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_frame_n_shows_pose_n(self):
        """'the frames differ' is also satisfied by a clip played backwards

        Nothing else in this file ties an output frame to a particular pose,
        so without this a reversed or frozen clip renders perfectly and reads
        as correct until someone watches it.
        """
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        obj   = list(scene.objects)[0]
        clip  = self.clip(path)

        seen  = []
        real  = Scene.render

        def record(self, *args, **kwargs):
            seen.append(np.array(obj.mesh.points, copy=True))
            return real(self, *args, **kwargs)

        with patch.object(Scene, "render", record):
            scene.animate(clip, self.out(), **RES)

        world     = np.asarray(clip.frames.world_matrix)
        column    = obj.skin.pose_indices(clip)
        reference = Object.load_glb(path)

        self.assertEqual(len(seen), clip.frame_count)
        for index, points in enumerate(seen):
            reference.pose(world[index, column])
            np.testing.assert_allclose(
                points,
                reference.mesh.points,
                atol    = 1e-9,
                err_msg = f"frame {index} was rendered at the wrong pose",
            )

    def test_the_scene_is_restored_when_the_fit_pass_blows_up(self):
        """the fit deforms to sample poses before a single frame is rendered,
        so it needs its own restore -- the render loop's never runs"""
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        obj   = list(scene.objects)[0]
        rest  = np.array(obj.mesh.points, copy=True)

        with patch.object(Scene, "_base_camera", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                scene.animate(self.clip(path), self.out(), **RES)

        np.testing.assert_allclose(rest, obj.mesh.points, atol=1e-9)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_an_unskinned_object_rides_along_as_a_prop(self):
        path      = self.skinned_glb()
        scene     = self.scene_with(path)
        prop      = Object.load_glb(path, load_skin=False)
        prop.name = "prop"
        scene.append(prop)
        prop_rest = np.array(prop.mesh.points, copy=True)

        written   = scene.animate(self.clip(path), self.out(), **RES)

        self.assertEqual(len(written), 5)
        np.testing.assert_allclose(prop_rest, prop.mesh.points, atol=1e-9)

    def test_fps_defaults_to_the_clips_own(self):
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        clip  = self.clip(path, fps=48.0)

        with patch(
            "cgmath.render.scene._encode_frame_stream", return_value=[]
        ) as encode:
            scene.animate(clip, self.out(), **RES)

        self.assertEqual(encode.call_args.kwargs["fps"], 48)

    def test_an_explicit_fps_beats_the_clip(self):
        path  = self.skinned_glb()
        scene = self.scene_with(path)

        with patch(
            "cgmath.render.scene._encode_frame_stream", return_value=[]
        ) as encode:
            scene.animate(self.clip(path, fps=48.0), self.out(), fps=12, **RES)

        self.assertEqual(encode.call_args.kwargs["fps"], 12)


class TestAnimateGuards(AnimateCase):
    def test_a_scene_with_nothing_skinned_raises(self):
        path  = self.skinned_glb()
        scene = Scene("s")
        scene.append(Object.load_glb(path, load_skin=False))

        with self.assertRaises(ValueError) as caught:
            scene.animate(self.clip(path), self.out(), **RES)

        self.assertIn("no visible skinned Objects", str(caught.exception))

    def test_an_empty_clip_raises(self):
        """``ClipData`` will not build with 0 frames, so the guard is reached
        by faking the count rather than by constructing one"""
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        clip  = self.clip(path)

        with patch.object(type(clip), "frame_count", property(lambda self: 0)):
            with self.assertRaises(ValueError) as caught:
                scene.animate(clip, self.out(), **RES)

        self.assertIn("no frames", str(caught.exception))

    def test_a_clip_missing_the_joints_raises_by_name(self):
        """a clip for a different rig is the realistic version of this"""
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        other = HierarchyData()
        other.append(TransformData(name="totally_different"))
        clip = ClipData(other, frames=3)

        with self.assertRaises(ValueError) as caught:
            scene.animate(clip, self.out(), **RES)

        message = str(caught.exception)
        self.assertIn("skinned to joints the clip does not hold", message)
        self.assertIn("body", message)

    def test_an_unknown_fit_raises(self):
        path  = self.skinned_glb()
        scene = self.scene_with(path)

        with self.assertRaises(ValueError) as caught:
            scene.animate(self.clip(path), self.out(), fit="wobble", **RES)

        self.assertIn("fit must be", str(caught.exception))

    def test_a_hidden_skinned_object_does_not_count(self):
        """asserting on the MESSAGE, not just ValueError: an animate that
        ignores visibility still raises here, just from the camera fit
        further down, so the type alone proves nothing"""
        path  = self.skinned_glb()
        scene = self.scene_with(path)
        list(scene.objects)[0].visibility = False

        with self.assertRaises(ValueError) as caught:
            scene.animate(self.clip(path), self.out(), **RES)

        self.assertIn("no visible skinned Objects", str(caught.exception))


@unittest.skipIf(Image is None, "PIL is not installed")
class TestFraming(AnimateCase):
    def test_static_and_auto_choose_different_cameras(self):
        path = self.skinned_glb()
        clip = self.clip(path)

        auto = self.scene_with(path).animate(
            clip, self.out("auto.{frame:04d}.png"), fit="auto", **RES
        )
        static = self.scene_with(path).animate(
            clip, self.out("static.{frame:04d}.png"), fit="static", **RES
        )

        self.assertEqual(len(auto), len(static))
        self.assertNotEqual(open(auto[0], "rb").read(), open(static[0], "rb").read())

    def test_an_orbit_changes_the_frames(self):
        path = self.skinned_glb()
        clip = self.clip(path)

        still = self.scene_with(path).animate(
            clip, self.out("still.{frame:04d}.png"), **RES
        )
        orbit = self.scene_with(path).animate(
            clip, self.out("orbit.{frame:04d}.png"), end_angle=360.0, **RES
        )

        self.assertNotEqual(open(still[-1], "rb").read(), open(orbit[-1], "rb").read())

    def test_sampling_more_poses_does_not_change_the_frame_count(self):
        path    = self.skinned_glb()
        scene   = self.scene_with(path)

        written = scene.animate(self.clip(path), self.out(), n_pose_samples=2, **RES)

        self.assertEqual(len(written), 5)


@unittest.skipIf(Image is None, "PIL is not installed")
class TestAnimateBatched(unittest.TestCase):
    """``animate`` deforms many frames per deformer call: the fit pass's
    samples in one call, the render loop's frames a chunk at a time.

    Every rendered frame must still show what a single-pose rig of that
    frame gives, seen from the camera the frame-by-frame fit would choose.
    Two skinned Objects, one moved off the origin, and a static prop.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        rig, self.clip = branching_clip(joints=8, frames=7, seed=31)
        mesh, skin = skinned_mesh(rig, seed=37)
        self.joints = len(skin.influences)

        self.scene  = Scene("batched")
        for name, offset in (("a", (0.0, 0.0, 0.0)), ("b", (4.0, 1.0, -2.0))):
            self.scene.append(
                Object(
                    name      = name,
                    mesh      = mesh.copy(),
                    skin      = SkinDeformData(mesh, skin, bind_rig=rig),
                    translate = offset,
                )
            )
        self.scene.append(Object(name="prop", mesh=mesh.copy(), translate=(-3.0, 0.0, 0.0)))
        self.skinned = [o for o in self.scene.objects if o.skin is not None]

    def out(self):
        return os.path.join(self.tmp.name, "anim.{frame:04d}.png")

    def reference_cloud(self, samples):
        """the frame-by-frame fit cloud: pose to each sample, gather, restore"""
        clip   = self.clip.copy()
        clouds = []
        for sample in samples:
            pose = pose_at(clip, sample)
            for obj in self.skinned:
                obj.pose(pose)
            clouds.append(self.scene._visible_world_points("animate"))
        for obj in self.skinned:
            obj.restore_bind_pose()
        return np.concatenate(clouds)

    def reference_points(self, frame):
        pose   = pose_at(self.clip.copy(), frame)
        points = [np.array(obj.pose(pose).mesh.points) for obj in self.skinned]
        for obj in self.skinned:
            obj.restore_bind_pose()
        return points

    def animate(self, **kwargs):
        """animate, keeping what every render saw: each skinned mesh's points
        and the camera; and the shape of every pose handed to a deformer"""
        seen, poses = [], []
        render  = Scene.render
        apply   = SkinDeformData.apply
        skinned = self.skinned

        def spy_render(scene, *args, **kw):
            points = [np.array(obj.mesh.points) for obj in skinned]
            seen.append((points, np.array(kw["camera_matrix"])))
            return render(scene, *args, **kw)

        def spy_apply(deformer, pose, target=None):
            poses.append(np.shape(pose) if isinstance(pose, np.ndarray) else pose)
            return apply(deformer, pose, target)

        _clear_render_cache(self.scene)
        with patch.object(Scene, "render", spy_render), patch.object(
            SkinDeformData, "apply", spy_apply
        ):
            self.scene.animate(self.clip, self.out(), **RES, **kwargs)
        return seen, poses

    def test_the_fit_cloud_is_every_sampled_pose(self):
        samples = _pose_sample_frames_for(self.clip.frame_count, None)
        skinned, columns = self.scene._skinned_objects(self.clip)
        world = np.asarray(self.clip.frames.world_matrix)
        rest  = [np.array(obj.mesh.points) for obj in skinned]

        got   = self.scene._posed_cloud(skinned, columns, world[samples])

        np.testing.assert_allclose(got, self.reference_cloud(samples), atol=1e-9)
        for obj, points in zip(skinned, rest):
            np.testing.assert_array_equal(obj.mesh.points, points)

    def test_every_frame_shows_its_pose_from_the_reference_camera(self):
        seen, _ = self.animate(end_angle=90.0)

        self.assertEqual(len(seen), self.clip.frame_count)
        for frame, (points, _) in enumerate(seen):
            for got, expected in zip(points, self.reference_points(frame)):
                np.testing.assert_allclose(
                    got, expected, atol=1e-9, err_msg=f"frame {frame}"
                )

        # the same animation with its fit fed the frame-by-frame cloud
        samples = _pose_sample_frames_for(self.clip.frame_count, None)
        cloud   = self.reference_cloud(samples)
        with patch.object(Scene, "_posed_cloud", lambda *args: cloud):
            expected, _ = self.animate(end_angle=90.0)

        for frame, ((_, got), (_, camera)) in enumerate(zip(seen, expected)):
            np.testing.assert_allclose(got, camera, atol=1e-9, err_msg=f"frame {frame}")

    def test_samples_and_frames_deform_in_one_call_each(self):
        """one deformer call per Object for the fit, one for the frames;
        posing frame by frame made (samples + frames) calls per Object"""
        frames  = self.clip.frame_count
        samples = len(_pose_sample_frames_for(frames, None))
        _, poses = self.animate()

        fit    = (samples, self.joints, 4, 4)
        render = (frames, self.joints, 4, 4)
        self.assertEqual(poses, [fit, fit, render, render])

    def test_a_small_chunk_budget_still_shows_every_pose(self):
        frames   = self.clip.frame_count
        samples  = len(_pose_sample_frames_for(frames, None))
        vertices = sum(len(obj.mesh.points) for obj in self.skinned)

        # room for three frames of every skinned Object's points
        with patch("cgmath.render.scene._ANIMATE_CHUNK_BYTES", vertices * 3 * 8 * 3):
            seen, poses = self.animate()

        shape = lambda count: (count, self.joints, 4, 4)  # noqa: E731
        self.assertEqual(
            poses,
            [shape(samples)] * 2 + [shape(3)] * 2 + [shape(3)] * 2 + [shape(1)] * 2,
        )
        self.assertEqual(len(seen), frames)
        for frame, (points, _) in enumerate(seen):
            for got, expected in zip(points, self.reference_points(frame)):
                np.testing.assert_allclose(
                    got, expected, atol=1e-9, err_msg=f"frame {frame}"
                )

    def test_without_an_orbit_the_fit_sweeps_one_angle(self):
        """a camera that holds still sees the cloud from one angle; sweeping
        frame_count copies of that angle gave the same box frame_count times"""
        from cgmath.render import scene as module

        calls = []
        real  = module._fit_camera_over_orbit

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        with patch.object(module, "_fit_camera_over_orbit", spy):
            seen, _ = self.animate()

        (args, kwargs), = calls
        self.assertEqual(args[10], 1)

        swept    = real(*args[:10], self.clip.frame_count, **kwargs)
        expected = module._orbit_camera_col(swept, args[3], args[7], args[8])
        for _, camera in seen:
            np.testing.assert_array_equal(camera, expected)

    def test_the_scene_is_handed_back_undeformed(self):
        rest = [np.array(obj.mesh.points) for obj in self.skinned]
        self.animate()
        for obj, points in zip(self.skinned, rest):
            np.testing.assert_allclose(obj.mesh.points, points, atol=1e-12)

    def test_an_object_poses_to_a_clip_at_its_frame(self):
        """read from the clip's blocks, not node by node"""
        obj      = self.skinned[1]
        expected = self.reference_points(5)[1]

        def refuse(node):
            raise AssertionError(f"{node.name}.world_matrix was read")

        clip       = self.clip.copy()
        clip.frame = 5
        with patch.object(TransformData, "world_matrix", property(refuse)):
            obj.pose(clip)

        np.testing.assert_allclose(obj.mesh.points, expected, atol=1e-9)
        self.assertEqual(clip.frame, 5)


class TestPoseSampling(unittest.TestCase):
    """the default is a cost control, so the arithmetic is worth pinning"""

    def test_a_short_clip_is_sampled_whole(self):
        np.testing.assert_array_equal(_pose_sample_frames_for(4, None), [0, 1, 2, 3])

    def test_a_medium_clip_gets_ten_samples(self):
        self.assertEqual(len(_pose_sample_frames_for(60, None)), 10)

    def test_a_long_clip_scales_to_a_tenth(self):
        self.assertEqual(len(_pose_sample_frames_for(300, None)), 30)

    def test_the_samples_span_the_whole_clip(self):
        picked = _pose_sample_frames_for(300, None)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 299)

    def test_an_explicit_count_wins(self):
        self.assertEqual(len(_pose_sample_frames_for(300, 3)), 3)

    def test_an_explicit_count_is_clamped_to_the_clip(self):
        """np.unique would bound this anyway -- the clamp is there to stop the
        intermediate linspace being built at the requested size, which no
        assertion on the output can observe"""
        self.assertEqual(len(_pose_sample_frames_for(4, 99)), 4)

    def test_zero_still_samples_one_frame(self):
        self.assertEqual(len(_pose_sample_frames_for(10, 0)), 1)


class TestEncodeFrameStream(unittest.TestCase):
    """the dispatch turntable and animate now share"""

    class FakeFrame:
        def __init__(self, tag):
            self.tag = tag

        def save(self, path):
            with open(path, "w") as handle:
                handle.write(self.tag)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def frames(self, count=3):
        return (self.FakeFrame(f"frame{i}") for i in range(count))

    def test_a_missing_placeholder_is_invented(self):
        out = os.path.join(self.tmp.name, "shot.png")

        written = _encode_frame_stream(self.frames(), out)

        self.assertEqual(
            [os.path.basename(p) for p in written],
            ["shot.0000.png", "shot.0001.png", "shot.0002.png"],
        )

    def test_an_explicit_placeholder_is_honoured(self):
        out = os.path.join(self.tmp.name, "shot_{frame:02d}.png")

        written = _encode_frame_stream(self.frames(2), out)

        self.assertEqual(
            [os.path.basename(p) for p in written], ["shot_00.png", "shot_01.png"]
        )

    def test_a_missing_directory_is_created(self):
        out = os.path.join(self.tmp.name, "nested", "deeper", "shot.png")

        written = _encode_frame_stream(self.frames(1), out)

        self.assertTrue(os.path.exists(written[0]))

    def test_the_stream_is_consumed_lazily(self):
        """a render-per-frame caller must never hold the whole sequence"""
        live = []

        def counted():
            for i in range(3):
                live.append(i)
                yield self.FakeFrame(f"frame{i}")
                # by the time frame i+1 is asked for, frame i is written
                self.assertEqual(len(live), i + 1)

        out = os.path.join(self.tmp.name, "lazy.png")
        self.assertEqual(len(_encode_frame_stream(counted(), out)), 3)

    def test_video_extensions_route_to_the_encoder(self):
        for ext, method in (
            (".mp4", "encode_mp4"),
            (".mov", "encode_mp4"),
            (".m4v", "encode_mp4"),
            (".avi", "encode_avi"),
            (".gif", "encode_gif"),
        ):
            with self.subTest(ext):
                out = os.path.join(self.tmp.name, f"clip{ext}")
                with patch(f"cgmath.render.frame.Frame.{method}") as encoder:
                    encoder.return_value = out
                    self.assertEqual(_encode_frame_stream(self.frames(), out), out)
                encoder.assert_called_once()


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestObjectAnimate(AnimateCase):
    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_it_renders_without_a_scene(self):
        path = self.skinned_glb()
        obj  = Object.load_glb(path)

        written = obj.animate(self.clip(path), output=self.out(), **RES)

        self.assertEqual(len(written), 5)

    @unittest.skipIf(Image is None, "PIL is not installed")
    def test_it_leaves_the_object_undeformed(self):
        path = self.skinned_glb()
        obj  = Object.load_glb(path)
        rest = np.array(obj.mesh.points, copy=True)

        obj.animate(self.clip(path), output=self.out(), **RES)

        np.testing.assert_allclose(rest, obj.mesh.points, atol=1e-9)

    def test_without_a_skin_it_says_so(self):
        path = self.skinned_glb()
        obj  = Object.load_glb(path, load_skin=False)

        with self.assertRaises(ValueError) as caught:
            obj.animate(self.clip(path), output=self.out(), **RES)

        self.assertIn("no skin to animate", str(caught.exception))


@unittest.skipIf(not HAVE_GLB, "pygltflib / trimesh are not installed")
class TestRestoreBindPose(AnimateCase):
    def test_it_undoes_a_pose(self):
        path = self.skinned_glb()
        obj  = Object.load_glb(path)
        rest = np.array(obj.mesh.points, copy=True)
        rig  = HierarchyData.load_glb(path)
        rig[list(rig.name).index("spine")].rotate = np.array([0.0, 0.0, 90.0])

        obj.pose(rig).restore_bind_pose()

        np.testing.assert_allclose(rest, obj.mesh.points, atol=1e-9)

    def test_it_does_not_alias_the_bind_pose(self):
        """assigning the cached bind mesh itself would let a later pose
        write through ``mesh`` into the bind pose"""
        path = self.skinned_glb()
        obj  = Object.load_glb(path)

        obj.restore_bind_pose()

        self.assertIsNot(obj.mesh, obj._bind_mesh)

    def test_it_is_a_no_op_without_a_skin(self):
        path = self.skinned_glb()
        obj  = Object.load_glb(path, load_skin=False)
        rest = np.array(obj.mesh.points, copy=True)

        obj.restore_bind_pose()

        np.testing.assert_allclose(rest, obj.mesh.points)