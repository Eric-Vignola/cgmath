"""Every FBX SDK manager cgmath creates is destroyed.

The SDK frees what it read only when ``FbxManager.Destroy()`` is called; the
Python wrapper going away frees nothing. A reader that forgets leaks the whole
scene on every call (about 18 MB per read of a 117-mesh character).

Counting managers is deterministic where measuring memory is not: the tests
patch ``FbxManager.Create`` and ``FbxManager.Destroy`` at CLASS level and
compare the counts. Never per instance: SIP hands back the same Python wrapper
for a new manager allocated where a destroyed one lived, so per-instance
patches chain and over-count.

The SDK is an optional binary dependency that is not installed in CI. Tests
needing it skip without it; the fake-SDK and source tests run everywhere.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from cgmath.formats import _fbx_io
from cgmath.geometry import mesh as mesh_mod, skin_weights as skin_mod
from cgmath.hierarchy import hierarchy as hierarchy_mod
from cgmath._tests.test_fbx_import import (
    _make_fake_fbx_module,
    _make_fake_root,
    _make_real_file,
    _wire_importer,
)

try:
    import fbx
except ImportError:
    fbx = None

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextmanager
def balanced(test):
    """asserts every FbxManager created inside the block was destroyed in it"""
    count   = {"created": 0, "destroyed": 0}
    create  = fbx.FbxManager.Create
    destroy = fbx.FbxManager.Destroy

    def counting_create():
        count["created"] += 1
        return create()

    def counting_destroy(self):
        count["destroyed"] += 1
        return destroy(self)

    with patch.object(fbx.FbxManager, "Create", counting_create), patch.object(
        fbx.FbxManager, "Destroy", counting_destroy
    ):
        yield count

    test.assertGreater(count["created"], 0, "nothing made a manager: the patch missed")
    test.assertEqual(count["created"], count["destroyed"], f"leaked managers: {count}")


def scratch_folders():
    """the cgmath_fbm_ folders in the temp dir"""
    root = tempfile.gettempdir()
    return {x for x in os.listdir(root) if x.startswith("cgmath_fbm_")}


def write_fixture(path, populate, embed=False):
    """populate(scene) builds the file; the manager dies with the export"""
    from cgmath._tests.test_skin_fbx import build_scene

    manager = fbx.FbxManager.Create()
    try:
        scene = build_scene(manager)
        populate(scene)
        ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
        ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, embed)
        manager.SetIOSettings(ios)
        exporter = fbx.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, ios):
            raise RuntimeError(exporter.GetStatus().GetErrorString())
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


def skinned_rig(scene):
    """4 verts on root and tip, and one take keying root's translate x"""
    from cgmath._tests.test_skin_fbx import add_joint, add_mesh, bind

    _, mesh = add_mesh(scene, "body_geo", 4)
    root = add_joint(scene, "root")
    tip  = add_joint(scene, "tip", parent=root)
    bind(scene, mesh, [(root, {0: 1.0, 1: 1.0}), (tip, {2: 1.0, 3: 1.0})])

    stack = fbx.FbxAnimStack.Create(scene, "take")
    layer = fbx.FbxAnimLayer.Create(scene, "base")
    stack.AddMember(layer)
    root.LclTranslation.GetCurveNode(layer, True)
    curve = root.LclTranslation.GetCurve(layer, "X", True)
    curve.KeyModifyBegin()
    for second in (0.0, 1.0):
        time = fbx.FbxTime()
        time.SetSecondDouble(second)
        curve.KeySetValue(curve.KeyAdd(time)[0], second * 10.0)
    curve.KeyModifyEnd()


def static_rig(scene):
    """two joints, no take"""
    from cgmath._tests.test_skin_fbx import add_joint

    add_joint(scene, "tip", parent=add_joint(scene, "root"))


def textured_quad(png):
    """a quad whose one material's diffuse map is ``png``"""

    def populate(scene):
        mesh = fbx.FbxMesh.Create(scene, "quadShape")
        mesh.InitControlPoints(4)
        for index, point in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]):
            mesh.SetControlPointAt(fbx.FbxVector4(*point, 1.0), index)

        uv = mesh.CreateElementUV("map1")
        uv.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByControlPoint)
        uv.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
        for point in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            uv.GetDirectArray().Add(fbx.FbxVector2(*point))

        mesh.BeginPolygon()
        for vertex in range(4):
            mesh.AddPolygon(vertex)
        mesh.EndPolygon()

        node = fbx.FbxNode.Create(scene, "quad")
        node.SetNodeAttribute(mesh)
        scene.GetRootNode().AddChild(node)

        texture = fbx.FbxFileTexture.Create(scene, "tex")
        texture.SetFileName(png)
        texture.SetTextureUse(fbx.FbxTexture.ETextureUse.eStandard)
        texture.SetMappingType(fbx.FbxTexture.EMappingType.eUV)
        material = fbx.FbxSurfacePhong.Create(scene, "mat")
        material.Diffuse.ConnectSrcObject(texture)
        node.AddMaterial(material)

    return populate


@unittest.skipIf(fbx is None, "Autodesk FBX Python SDK is not installed")
class FbxFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp    = tempfile.TemporaryDirectory()
        cls.rig    = write_fixture(cls.file("rig.fbx"), skinned_rig)
        cls.static = write_fixture(cls.file("static.fbx"), static_rig)
        cls.out    = cls.file("out.fbx")

        cls.bad = cls.file("garbage.fbx")
        with open(cls.bad, "wb") as f:
            f.write(b"not an fbx" * 64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def file(cls, name):
        return os.path.join(cls.tmp.name, name)


class TestReaders(FbxFixtures):
    def test_hierarchy_loaders(self):
        from cgmath import hierarchy

        loaders = (
            hierarchy_mod.HierarchyData.load_fbx,
            hierarchy_mod.TransformList.load_fbx,
            hierarchy_mod.HierarchyData.load,
            hierarchy.load_fbx,
        )
        for load in loaders:
            with self.subTest(load=load), balanced(self):
                self.assertEqual(load(self.rig).name, ["body_geo", "root", "tip"])

    def test_clip(self):
        with balanced(self):
            clip = hierarchy_mod.ClipData.load_fbx(self.rig, fps=4.0)
        self.assertEqual(clip.frame_count, 5)

    def test_clip_without_a_take(self):
        with balanced(self):
            clip = hierarchy_mod.ClipData.load_fbx(self.static)
        self.assertEqual(clip.frame_count, 1)

    def test_clip_raising_after_the_read(self):
        cases = (
            (IndexError, {"take": 9}),
            (ValueError, {"fps": 4.0, "start_frame": 5, "end_frame": 1}),
        )
        for error, kwargs in cases:
            with self.subTest(**kwargs), balanced(self), self.assertRaises(error):
                hierarchy_mod.ClipData.load_fbx(self.rig, **kwargs)

    def test_mesh_loaders(self):
        loaders = (
            mesh_mod.load_fbx,
            mesh_mod.MeshList.load_fbx,
            mesh_mod.MeshData.load_fbx,
            mesh_mod.UVList.load_fbx,
        )
        for load in loaders:
            with self.subTest(load=load), balanced(self):
                load(self.rig)

        # the fixture has no uv channel
        with balanced(self), self.assertRaises(IndexError):
            mesh_mod.UVData.load_fbx(self.rig)

    def test_skin_loaders(self):
        loaders = (skin_mod.load_fbx, skin_mod.SkinList.load_fbx, skin_mod.SkinData.load_fbx)
        for load in loaders:
            with self.subTest(load=load), balanced(self):
                load(self.rig)

    def test_render_object(self):
        from cgmath.render import Object

        with balanced(self):
            obj = Object.load_fbx(self.rig)
        self.assertIsNotNone(obj.skin)

    def test_save(self):
        rig = hierarchy_mod.HierarchyData.load_fbx(self.rig)
        with balanced(self):
            rig.save_fbx(self.out)
        self.assertEqual(hierarchy_mod.HierarchyData.load_fbx(self.out).name, rig.name)

    def test_unreadable_file(self):
        loaders = (
            hierarchy_mod.HierarchyData.load_fbx,
            hierarchy_mod.ClipData.load_fbx,
            skin_mod.SkinList.load_fbx,
            mesh_mod.load_fbx,
        )
        for load in loaders:
            with self.subTest(load=load), balanced(self):
                with self.assertRaises(_fbx_io.FbxReadError):
                    load(self.bad)

    def test_missing_file(self):
        missing = self.file("missing.fbx")
        loaders = (
            hierarchy_mod.HierarchyData.load_fbx,
            hierarchy_mod.TransformList.load_fbx,
            hierarchy_mod.ClipData.load_fbx,
            skin_mod.SkinList.load_fbx,
            mesh_mod.load_fbx,
        )
        for load in loaders:
            with self.subTest(load=load), self.assertRaises(FileNotFoundError):
                load(missing)

    def test_texture_misses(self):
        from cgmath.render import scene as scene_mod

        for kwargs in ({}, {"mesh_index": 99}):
            with self.subTest(**kwargs), balanced(self):
                self.assertEqual(scene_mod._extract_fbx_textures(self.rig, **kwargs), {})

    def test_nothing_read_holds_an_fbx_object(self):
        from cgmath.render import Object

        results = [
            hierarchy_mod.HierarchyData.load_fbx(self.rig),
            hierarchy_mod.ClipData.load_fbx(self.rig, fps=4.0),
            mesh_mod.load_fbx(self.rig),
            skin_mod._read_skins_fbx(self.rig, bind_matrices=True),
            Object.load_fbx(self.rig),
        ]

        seen  = set()
        stack = list(results)
        while stack:
            value = stack.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            self.assertNotEqual(type(value).__module__, "fbx", repr(value))
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                stack.extend(value)
            elif hasattr(value, "__dict__") and not isinstance(value, type):
                stack.extend(vars(value).values())


@unittest.skipIf(Image is None, "PIL is not installed")
class TestEmbeddedMedia(FbxFixtures):
    """an fbx carrying its texture inside: read without writing beside it"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.folder = cls.file("embedded")
        os.makedirs(cls.folder)
        png = os.path.join(cls.folder, "green.png")
        Image.fromarray(np.full((8, 8, 3), (0, 255, 0), np.uint8)).save(png)
        cls.quad = write_fixture(
            os.path.join(cls.folder, "quad.fbx"), textured_quad(png), embed=True
        )
        # the texture can now only come from inside the file
        os.remove(png)

    def assert_nothing_written(self, before):
        self.assertEqual(os.listdir(self.folder), ["quad.fbx"])
        self.assertEqual(scratch_folders(), before)

    def test_texture_is_read_from_inside_the_file(self):
        from cgmath.render import scene as scene_mod

        before = scratch_folders()
        with balanced(self):
            textures = scene_mod._extract_fbx_textures(self.quad)
        np.testing.assert_array_equal(textures["base_color"][0, 0, :3], [0, 255, 0])
        self.assert_nothing_written(before)

    def test_object(self):
        from cgmath.render import Object

        before = scratch_folders()
        self.assertIsNotNone(Object.load_fbx(self.quad).texture)
        self.assert_nothing_written(before)

    def test_readers_extract_nothing(self):
        before = scratch_folders()
        hierarchy_mod.HierarchyData.load_fbx(self.quad)
        mesh_mod.load_fbx(self.quad)
        self.assert_nothing_written(before)

    def test_a_failed_copy_puts_the_scene_back(self):
        from cgmath.formats import fbx as fbx_mod
        from cgmath.render import scene as scene_mod

        broken = self.file("broken")
        later  = self.file("later")
        os.makedirs(broken)
        os.makedirs(later)

        with fbx_mod.SceneData(self.quad) as data:
            with patch.object(fbx_mod.shutil, "copy2", side_effect=PermissionError("read only")):
                with self.assertRaises(PermissionError):
                    data.save(os.path.join(broken, "copy.fbx"), embed_media=False)

            # the next save still finds the textures where the scene keeps them
            saved = data.save(os.path.join(later, "copy.fbx"), embed_media=False)

        textures = scene_mod._extract_fbx_textures(saved)
        np.testing.assert_array_equal(textures["base_color"][0, 0, :3], [0, 255, 0])

    def test_a_failed_export_copies_nothing(self):
        from cgmath.formats.fbx import SceneData

        target = self.file("taken")
        os.makedirs(os.path.join(target, "copy.fbx"))  # a folder where the file goes
        with SceneData(self.quad) as data, self.assertRaises(RuntimeError):
            data.save(os.path.join(target, "copy.fbx"), embed_media=False)
        self.assertEqual(os.listdir(target), ["copy.fbx"])

    def test_a_non_ascii_temp_folder_falls_back_beside_the_file(self):
        from cgmath.formats.fbx import SceneData
        from cgmath.render import scene as scene_mod

        # the fbx binding cannot hand a non-ASCII path to the SDK
        folder = self.file("fallback")
        os.makedirs(folder)
        source = os.path.join(folder, "quad.fbx")
        with open(self.quad, "rb") as f, open(source, "wb") as g:
            g.write(f.read())

        mkdtemp = tempfile.mkdtemp

        def unicode_temp(prefix="", **kwargs):
            return mkdtemp(prefix=prefix + "\u00e9_", **kwargs)

        with patch.object(_fbx_io.tempfile, "mkdtemp", unicode_temp):
            with self.assertWarnsRegex(RuntimeWarning, "ASCII"):
                textures = scene_mod._extract_fbx_textures(source)
            with self.assertWarnsRegex(RuntimeWarning, "ASCII"), SceneData(source) as data:
                self.assertEqual(data.name, "quad")

        np.testing.assert_array_equal(textures["base_color"][0, 0, :3], [0, 255, 0])
        self.assertEqual(sorted(os.listdir(folder)), ["quad.fbm", "quad.fbx"])

    def test_an_identical_read_only_texture_is_left_alone(self):
        import stat

        from cgmath.formats.fbx import SceneData

        folder = self.file("read_only")
        os.makedirs(folder)
        out     = os.path.join(folder, "copy.fbx")
        texture = os.path.join(folder, "copy.fbm", "green.png")

        with SceneData(self.quad) as data:
            data.save(out, embed_media=False)
        os.chmod(texture, stat.S_IREAD)  # as a submitted file is
        try:
            with SceneData(self.quad) as data:
                data.create_take("second")
                data.save(out, embed_media=False)
        finally:
            os.chmod(texture, stat.S_IWRITE | stat.S_IREAD)

    def test_a_relative_save_path_is_made_absolute(self):
        from cgmath.formats.fbx import SceneData
        from cgmath.render import scene as scene_mod

        folder = self.file("relative")
        decoy  = self.file("decoy")
        os.makedirs(os.path.join(folder, "sub"))
        os.makedirs(os.path.join(decoy, "sub", "copy.fbm"))
        red = np.full((8, 8, 3), (255, 0, 0), np.uint8)
        Image.fromarray(red).save(os.path.join(decoy, "sub", "copy.fbm", "green.png"))

        here = os.getcwd()
        try:
            os.chdir(folder)
            with SceneData(self.quad) as data:
                data.save(os.path.join("sub", "copy.fbx"), embed_media=False)

            # read from somewhere holding the same relative layout
            os.chdir(decoy)
            textures = scene_mod._extract_fbx_textures(os.path.join(folder, "sub", "copy.fbx"))
        finally:
            os.chdir(here)
        np.testing.assert_array_equal(textures["base_color"][0, 0, :3], [0, 255, 0])

    def test_a_wrapper_of_the_wrong_type_is_re_typed(self):
        from cgmath.formats.fbx import SceneData
        from cgmath.render import scene as scene_mod

        folder = self.file("re_typed")
        os.makedirs(folder)
        with SceneData(self.quad) as data:
            # the binding can hand back a wrapper typed for whatever lived at
            # the object's address before; a base FbxObject stands in for it
            scene      = data.scene
            get_object = scene.GetSrcObject
            handed     = []

            def base_typed(*args):
                handed.append(fbx.cast(get_object(*args), fbx.FbxObject))
                return handed[-1]

            scene.GetSrcObject = base_typed
            saved              = data.save(os.path.join(folder, "copy.fbx"), embed_media=False)

        self.assertTrue(handed and not isinstance(handed[0], fbx.FbxFileTexture))

        # wrappers of the closed scene, dropped before the next read
        del handed, scene, get_object, base_typed
        textures = scene_mod._extract_fbx_textures(saved)
        np.testing.assert_array_equal(textures["base_color"][0, 0, :3], [0, 255, 0])

    def test_typed(self):
        manager = fbx.FbxManager.Create()
        try:
            scene   = fbx.FbxScene.Create(manager, "")
            texture = fbx.FbxFileTexture.Create(scene, "tex")
            texture.SetFileName("C:/a/b.png")
            base    = fbx.cast(texture, fbx.FbxObject)
            self.assertNotIsInstance(base, fbx.FbxFileTexture)

            again = _fbx_io.typed(base, fbx.FbxFileTexture, fbx)
            self.assertIsInstance(again, fbx.FbxFileTexture)
            self.assertEqual(again.GetUniqueID(), texture.GetUniqueID())
            self.assertEqual(again.GetFileName(), "C:/a/b.png")
            self.assertIs(_fbx_io.typed(texture, fbx.FbxFileTexture, fbx), texture)
            self.assertIsNone(_fbx_io.typed(None, fbx.FbxFileTexture, fbx))
        finally:
            manager.Destroy()

    def test_a_save_without_embedding_brings_its_textures(self):
        from cgmath.formats.fbx import SceneData
        from cgmath.render import scene as scene_mod

        plain    = self.file("plain")
        embedded = self.file("embedded_again")
        os.makedirs(plain)
        os.makedirs(embedded)

        before = scratch_folders()
        with SceneData(self.quad) as data:
            linked = data.save(os.path.join(plain, "copy.fbx"), embed_media=False)
            packed = data.save(os.path.join(embedded, "copy.fbx"))

        # the textures sit beside the file that points at them
        self.assertEqual(sorted(os.listdir(plain)), ["copy.fbm", "copy.fbx"])
        self.assert_nothing_written(before)

        # and the scene was pointed back: the next save still embeds them
        for path in (linked, packed):
            with self.subTest(path=path):
                textures = scene_mod._extract_fbx_textures(path)
                np.testing.assert_array_equal(textures["base_color"][0, 0, :3], [0, 255, 0])


class TestSceneData(FbxFixtures):
    def test_with_block(self):
        from cgmath.formats.fbx import SceneData

        with balanced(self):
            with SceneData(self.rig) as data:
                self.assertEqual(data.name, "rig")
                media = data._media
                self.assertTrue(os.path.isdir(media))

        self.assertEqual(data.name, "No Scene")
        self.assertFalse(os.path.exists(media))
        data.close()  # twice is fine

    def test_unreadable_file(self):
        from cgmath.formats.fbx import SceneData

        before = scratch_folders()
        with balanced(self), self.assertRaises(_fbx_io.FbxReadError):
            SceneData(self.bad)
        self.assertEqual(scratch_folders(), before)

    def test_failed_manager_leaves_no_folder(self):
        from cgmath.formats import fbx as fbx_mod

        before = scratch_folders()
        with patch.object(fbx_mod, "new_manager", side_effect=RuntimeError("no sdk")):
            with self.assertRaisesRegex(RuntimeError, "no sdk"):
                fbx_mod.SceneData(self.rig)
        self.assertEqual(scratch_folders(), before)

    def test_reading_after_close_raises(self):
        from cgmath.formats.fbx import SceneData

        data   = SceneData(self.rig)
        takes  = data.takes
        take   = takes[0]
        layers = take.layers
        layer  = layers[0]
        self.assertEqual(take.name, "take")
        data.close()

        for read in (take, layer):
            with self.subTest(read=type(read).__name__):
                with self.assertRaisesRegex(RuntimeError, "destroyed"):
                    read.name
                self.assertIn("<destroyed>", repr(read))

        # a repr never raises, so the lists still print
        for items in (takes, layers):
            self.assertIn("<destroyed>", repr(items))

    def test_reload_frees_the_previous_scene(self):
        from cgmath.formats.fbx import SceneData

        with balanced(self):
            data = SceneData(self.rig)
            take = data.takes[0]
            data.load(self.static)
            with self.assertRaises(RuntimeError):
                take.name
            self.assertEqual(data.name, "static")
            data.close()

    def test_failed_reload_keeps_the_scene(self):
        from cgmath.formats.fbx import SceneData

        with balanced(self), SceneData(self.rig) as data:
            take = data.takes[0]
            with self.assertRaises(_fbx_io.FbxReadError):
                data.load(self.bad)
            self.assertEqual(take.name, "take")
            self.assertEqual(data.name, "rig")


class TestDelete(FbxFixtures):
    """deleting takes, layers and curves, and the wrappers left holding them"""

    def test_a_take_goes_with_its_layers_and_keys(self):
        from cgmath.formats.fbx import SceneData

        with SceneData(self.rig) as data:
            take  = data.takes[0]
            layer = take.layers[0]
            curve = layer.curves[0]
            self.assertEqual(curve.key_count, 2)

            data.takes.delete("take")
            self.assertEqual(len(data.takes), 0)

            # every other wrapper of what went raises, rather than read freed memory
            reads = ((take, lambda: take.name), (layer, lambda: layer.name),
                     (curve, lambda: curve.key_count))
            for stale, read in reads:
                with self.subTest(stale=type(stale).__name__):
                    with self.assertRaisesRegex(RuntimeError, "deleted"):
                        read()
                    self.assertIn("<destroyed>", repr(stale))

            saved = data.save(self.out)

        with SceneData(saved) as back:
            self.assertNotIn("take", [x.name for x in back.takes])

    def test_a_layer_and_a_curve(self):
        from cgmath.formats.fbx import SceneData

        with SceneData(self.rig) as data:
            layers = data.takes[0].layers
            curves = layers[0].curves
            curves.delete(curves[0].name)
            self.assertEqual(len(layers[0].curves), 0)
            saved = data.save(self.out)

            layers.delete(layers[0].name)
            self.assertEqual(len(data.takes[0].layers), 0)

        with SceneData(saved) as back:
            self.assertEqual(len(back.takes["take"].layers[0].curves), 0)

    def test_a_list_holding_a_deleted_item_still_finds_the_rest(self):
        from cgmath.formats.fbx import SceneData

        with SceneData(self.rig) as data:
            data.create_take("second")
            kept = data.takes
            data.takes.delete("take")

            self.assertIn("second", kept)
            self.assertEqual(kept["second"].name, "second")
            kept.delete("second")
            self.assertEqual(len(data.takes), 0)

    def test_a_wrapper_built_by_hand_is_tracked_too(self):
        from cgmath.formats.fbx import SceneData, TakeData, TakeList

        with SceneData(self.rig) as data:
            take  = data.takes[0]
            stack = take._data
            TakeList([TakeData(data.scene, stack)]).delete("take")
            with self.assertRaisesRegex(RuntimeError, "deleted"):
                take.name

            data.create_take("second")
            by_hand = TakeData(data.scene, data.takes[0]._data)

        with self.assertRaisesRegex(RuntimeError, "destroyed"):
            by_hand.name


class TestFbxExporter(FbxFixtures):
    def test_with_block(self):
        from cgmath.formats.fbx import FbxExporter

        rig = hierarchy_mod.HierarchyData.load_fbx(self.rig)
        with balanced(self), FbxExporter() as exporter:
            exporter.add_skeleton(rig)
            exporter.export(self.out)

        self.assertIsNone(exporter._manager)
        self.assertIsNone(exporter._scene)
        exporter.close()  # twice is fine

    def test_zero_root(self):
        from cgmath.hierarchy import HierarchyData, TransformData

        rig = HierarchyData([
            TransformData("root", node_type="joint", translate=(5, 0, 0), rotate=(0, 30, 0)),
            TransformData("spine", parent_node="root", translate=(0, 10, 0), node_type="joint"),
        ])
        with balanced(self):
            rig.save_fbx(self.out, zero_root=True)

        back = hierarchy_mod.HierarchyData.load_fbx(self.out)
        root, spine = back.get("root"), back.get("spine")
        np.testing.assert_allclose(root.translate,  [0, 0, 0],  atol=1e-9)
        np.testing.assert_allclose(root.rotate,     [0, 0, 0],  atol=1e-9)
        np.testing.assert_allclose(root.scale,      [1, 1, 1],  atol=1e-9)
        np.testing.assert_allclose(spine.translate, [0, 10, 0], atol=1e-9)

    def test_zero_root_leaves_a_mesh_at_the_world_alone(self):
        rig = hierarchy_mod.HierarchyData.load_fbx(self.rig)  # body_geo comes first
        rig.get("body_geo").translate = (1, 2, 3)
        rig.get("root").translate = (5, 0, 7)
        rig.save_fbx(self.out, zero_root=True)

        back = hierarchy_mod.HierarchyData.load_fbx(self.out)
        np.testing.assert_allclose(back.get("root").translate, [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(back.get("body_geo").translate, [1, 2, 3], atol=1e-9)

    def test_zero_root_with_a_name_used_twice(self):
        from cgmath.hierarchy import HierarchyData, TransformData

        group  = TransformData("grp")
        nested = TransformData("root", parent_node=group.uuid, translate=(1, 2, 3))
        joint  = TransformData("root", node_type="joint", translate=(5, 0, 7))
        HierarchyData([group, nested, joint]).save_fbx(self.out, zero_root=True)

        back  = hierarchy_mod.HierarchyData.load_fbx(self.out)
        roots = {x.node_type: x for x in back if x.name == "root"}
        np.testing.assert_allclose(roots["joint"].translate, [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(roots["transform"].translate, [1, 2, 3], atol=1e-9)

    def test_zero_root_without_a_root_joint(self):
        from cgmath.hierarchy import HierarchyData, TransformData

        with self.assertRaisesRegex(ValueError, "joint"):
            HierarchyData([TransformData("grp")]).save_fbx(self.out, zero_root=True)

    def test_closed_exporter_refuses_work(self):
        from cgmath.formats.fbx import FbxExporter

        rig = hierarchy_mod.HierarchyData.load_fbx(self.rig)
        with FbxExporter() as exporter:
            exporter.add_skeleton(rig)

        # rather than start over on an empty scene with a manager nobody frees
        for work in (
            lambda: exporter.export(self.out),
            lambda: exporter.add_skeleton(rig),
            lambda: exporter.scene,
        ):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                work()


class TestFakeSdk(unittest.TestCase):
    """the failure paths, through a fake SDK, so they run without one"""

    def setUp(self):
        self.path = _make_real_file()
        self.addCleanup(os.unlink, self.path)

    def check(self, module, attribute, load):
        for wiring in ({"initialize_ok": False}, {"import_ok": False}):
            fake = _make_fake_fbx_module()
            with self.subTest(**wiring), patch.object(module, attribute, fake):
                manager = _wire_importer(
                    fake, root_node=_make_fake_root([]), error="bad header", **wiring
                )
                with self.assertRaisesRegex(_fbx_io.FbxReadError, "bad header"):
                    load(self.path)
                manager.Destroy.assert_called_once()

    def test_hierarchy(self):
        self.check(hierarchy_mod, "FBX", hierarchy_mod.HierarchyData.load_fbx)

    def test_clip(self):
        self.check(hierarchy_mod, "FBX", hierarchy_mod.ClipData.load_fbx)

    def test_skin(self):
        self.check(skin_mod, "fbx", skin_mod.SkinList.load_fbx)

    def test_mesh(self):
        self.check(mesh_mod, "fbx", mesh_mod.load_fbx)


class TestOpenFbx(unittest.TestCase):
    def setUp(self):
        self.path = _make_real_file()
        self.addCleanup(os.unlink, self.path)
        self.fake = _make_fake_fbx_module()

    def test_without_the_sdk(self):
        with self.assertRaises(ImportError):
            with _fbx_io.open_fbx(self.path, None):
                pass

    def test_missing_file_makes_no_manager(self):
        with self.assertRaises(FileNotFoundError):
            with _fbx_io.open_fbx(self.path + ".missing", self.fake):
                pass
        self.fake.FbxManager.Create.assert_not_called()

    def test_failed_settings_destroy_the_manager(self):
        manager                                    = _wire_importer(self.fake, root_node=_make_fake_root([]))
        self.fake.FbxIOSettings.Create.side_effect = RuntimeError("no settings")
        with self.assertRaises(RuntimeError):
            with _fbx_io.open_fbx(self.path, self.fake):
                pass
        manager.Destroy.assert_called_once()

    def test_failed_import_destroys_scene_and_importer(self):
        _wire_importer(self.fake, root_node=_make_fake_root([]), import_ok=False)
        scene    = self.fake.FbxScene.Create.return_value
        importer = self.fake.FbxImporter.Create.return_value
        with self.assertRaises(_fbx_io.FbxReadError):
            with _fbx_io.open_fbx(self.path, self.fake):
                pass
        scene.Destroy.assert_called_once()
        importer.Destroy.assert_called_once()

    def test_media_settings(self):
        for media in (False, True):
            fake    = _make_fake_fbx_module()
            manager = _wire_importer(fake, root_node=_make_fake_root([]))
            ios     = fake.FbxIOSettings.Create.return_value
            with self.subTest(media=media):
                with _fbx_io.open_fbx(self.path, fake, media=media):
                    pass
                ios.SetBoolProp.assert_called_once_with(_fbx_io.EXTRACT_MEDIA, media)
                self.assertEqual(ios.SetStringProp.called, media)
                manager.Destroy.assert_called_once()

    def test_media_folder_goes_on_failure(self):
        _wire_importer(self.fake, root_node=_make_fake_root([]), initialize_ok=False)
        mkdtemp = tempfile.mkdtemp
        made    = []

        def record(**kwargs):
            made.append(mkdtemp(**kwargs))
            return made[-1]

        with patch.object(_fbx_io.tempfile, "mkdtemp", record):
            with self.assertRaises(_fbx_io.FbxReadError):
                with _fbx_io.open_fbx(self.path, self.fake, media=True):
                    pass
        self.assertEqual(len(made), 1)
        self.assertFalse(os.path.exists(made[0]))


class TestOneManagerSite(unittest.TestCase):
    """readers can only open fbx files through formats._fbx_io"""

    def test_only_fbx_io_creates_managers(self):
        found = []
        for folder, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in ("_tests", "__pycache__")]
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as f:
                    if "FbxManager.Create" in f.read():
                        found.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
        self.assertEqual(found, ["formats/_fbx_io.py"])

    def test_the_old_reader_is_gone(self):
        from cgmath import hierarchy

        self.assertFalse(hasattr(hierarchy, "_read_fbx"))


if __name__ == "__main__":
    unittest.main()
