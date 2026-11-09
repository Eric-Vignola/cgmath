"""A Scene's turntables and animations keep their frames for the next encode.

A turntable or animation renders once; encoding the same frames again (a
.gif, then an .mp4) reads them back from a scratch folder. Anything the frames
depend on changing renders them again.
"""

from __future__ import annotations

import filecmp
import gc
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cgmath.geometry import MeshData
from cgmath.hierarchy import ClipData, generate_uuid, HierarchyData, TransformData
from cgmath.render import scene as scene_module
from cgmath.render.scene import Scene

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pygltflib
    import trimesh
except ImportError:
    pygltflib = trimesh = None

RES = {"resolution": (32, 32), "samples_per_pixel": 1}


def triangle():
    return MeshData(
        points  = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        indices = np.arange(3),
        counts  = np.array([3]),
        name    = "tri",
    )


@unittest.skipIf(Image is None, "PIL is not installed")
class RenderCacheCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def out(self, stem):
        return os.path.join(self.tmp.name, stem + ".{frame:04d}.png")

    def renders(self, call):
        """how many frames *call* rendered"""
        original = Scene.render
        with patch.object(Scene, "render", autospec=True, side_effect=original) as spy:
            call()
        return spy.call_count

    def scene(self):
        """a triangle on a node"""
        node             = TransformData("tri", uuid=generate_uuid())
        mesh             = triangle()
        mesh.parent_uuid = node.uuid
        return Scene("s").add(node, mesh)


class TestTurntable(RenderCacheCase):
    def test_a_setting_passed_as_its_own_value_reads_the_frames_back(self):
        # the stored resolution is an int32 array, the argument a tuple
        scene            = self.scene()
        scene.resolution = (32, 32)
        scene.turntable(self.out("a"), n_frames=2, samples_per_pixel=1)
        again = lambda: scene.turntable(self.out("b"), n_frames=2, **RES)
        self.assertEqual(self.renders(again), 0)

    def test_a_second_encode_reads_the_frames_back(self):
        scene = self.scene()
        first = scene.turntable(self.out("a"), n_frames=4, **RES)
        self.assertEqual(self.renders(lambda: scene.turntable(self.out("b"), n_frames=4, **RES)), 0)

        second = [self.out("b").format(frame=i) for i in range(4)]
        self.assertTrue(all(filecmp.cmp(x, y, shallow=False) for x, y in zip(first, second)))

    def test_a_change_renders_again(self):
        scene = self.scene()
        scene.turntable(self.out("a"), n_frames=4, **RES)

        changes = [
            {"n_frames": 5, **RES},
            {"n_frames": 4, "resolution": (40, 40), "samples_per_pixel": 1},
            {"n_frames": 4, "resolution": (32, 32), "samples_per_pixel": 4},
            {"n_frames": 4, "end_angle": 180.0, **RES},
        ]
        for settings in changes:
            with self.subTest(settings=settings):
                self.assertGreater(self.renders(lambda: scene.turntable(self.out("x"), **settings)), 0)

    def test_moving_a_node_renders_again(self):
        scene = self.scene()
        scene.turntable(self.out("a"), n_frames=3, **RES)
        scene.hierarchy[0].translate = [0.5, 0.0, 0.0]
        self.assertEqual(self.renders(lambda: scene.turntable(self.out("b"), n_frames=3, **RES)), 3)

    def test_a_shading_setting_renders_again(self):
        scene = self.scene()
        scene.turntable(self.out("a"), n_frames=3, **RES)
        scene.base_color = (1.0, 0.0, 0.0)
        self.assertEqual(self.renders(lambda: scene.turntable(self.out("b"), n_frames=3, **RES)), 3)

    def test_clear_render_cache(self):
        scene = self.scene()
        scene.turntable(self.out("a"), n_frames=3, **RES)
        scene.clear_render_cache()
        self.assertEqual(self.renders(lambda: scene.turntable(self.out("b"), n_frames=3, **RES)), 3)

    def test_an_interrupted_encode_is_not_reused(self):
        scene = self.scene()
        saved = []

        def save(frame, path, **kwargs):
            if len(saved) == 2:
                raise OSError("disk full")
            saved.append(path)

        with patch("cgmath.render.frame.Frame.save", autospec=True, side_effect=save):
            with self.assertRaises(OSError):
                scene.turntable(self.out("a"), n_frames=4, **RES)
        self.assertEqual(self.renders(lambda: scene.turntable(self.out("b"), n_frames=4, **RES)), 4)

    def test_the_folder_goes_with_its_owner(self):
        scene = self.scene()
        scene.turntable(self.out("a"), n_frames=2, **RES)
        folder = scene_module._CACHE_DIRS[id(scene)]
        self.assertTrue(os.listdir(folder))

        del scene
        gc.collect()
        self.assertFalse(os.path.exists(folder))

    def test_the_scratch_folder_goes_at_exit(self):
        # the child imports from where this process does (mayapy's venv is
        # only on sys.path once Maya is initialised)
        code = (
            "import sys; sys.path[:0] = {path!r}; import numpy as np; "
            "from cgmath.geometry import MeshData; "
            "from cgmath.render.scene import Scene; "
            "import cgmath.render.scene as S; "
            "s = Scene('s'); "
            "s.add(MeshData(points=np.eye(3), indices=np.arange(3), counts=np.array([3]), name='t')); "
            "s.turntable({out!r}, n_frames=2, resolution=(16, 16), samples_per_pixel=1); "
            "print(S._CACHE_ROOT)"
        ).format(path=sys.path, out=self.out("child"))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        lines  = result.stdout.strip().splitlines()
        self.assertTrue(lines, result.stderr)
        self.assertTrue(os.path.basename(lines[-1]).startswith("cgmath_render_"), result.stderr)
        self.assertFalse(os.path.exists(lines[-1]))


@unittest.skipIf(pygltflib is None, "pygltflib / trimesh are not installed")
class TestAnimate(RenderCacheCase):
    def setUp(self):
        super().setUp()
        from cgmath._tests.test_scene_skin import write_glb

        self.glb = os.path.join(self.tmp.name, "hero.glb")
        write_glb(self.glb, ["body"])

    def clip(self, degrees_per_frame=15.0):
        rig   = HierarchyData.load_glb(self.glb)
        clip  = ClipData(rig, frames=4, fps=24.0)
        spine = list(rig.name).index("spine")
        for index in range(clip.frame_count):
            clip.frame         = index
            clip[spine].rotate = np.array([0.0, 0.0, degrees_per_frame * index])
        return clip

    def test_a_second_encode_reads_the_frames_back(self):
        # posing places a deformed copy every frame: the key still matches
        scene = Scene.load_glb(self.glb)
        clip  = self.clip()
        scene.animate(self.out("a"), clip=clip, **RES)
        bind = scene.meshes[0].points.copy()

        self.assertEqual(self.renders(lambda: scene.animate(self.out("b"), clip=clip, **RES)), 0)
        np.testing.assert_array_equal(scene.meshes[0].points, bind)

        self.assertEqual(self.renders(lambda: scene.animate(self.out("c"), clip=self.clip(30.0), **RES)), 4)

    def test_the_scenes_own_clip_keeps_its_frames_too(self):
        scene      = Scene.load_glb(self.glb)
        scene.clip = self.clip()
        scene.animate(self.out("a"), **RES)
        self.assertEqual(self.renders(lambda: scene.animate(self.out("b"), **RES)), 0)


if __name__ == "__main__":
    unittest.main()
