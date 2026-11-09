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
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
from cgmath.formats import _fbx_io
from cgmath.geometry import mesh as mesh_mod, skin_weights as skin_mod
from cgmath.hierarchy import hierarchy as hierarchy_mod

fbx = _fbx_io.load_sdk()

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- a synthetic SDK, for the failure paths -----------------------------------


def _make_fake_fbx_module() -> types.SimpleNamespace:
    """a ``types.SimpleNamespace`` shaped like the Autodesk FBX SDK, holding
    only what ``_fbx_io.open_fbx`` touches"""
    fake                               = types.SimpleNamespace()
    fake.IOSROOT                       = "IOSROOT"
    fake.IMP_FBX_EXTRACT_EMBEDDED_DATA = "Import|ExtractEmbeddedData"
    fake.FbxManager                    = types.SimpleNamespace(Create=MagicMock())
    fake.FbxIOSettings                 = types.SimpleNamespace(Create=MagicMock())
    fake.FbxImporter                   = types.SimpleNamespace(Create=MagicMock())
    fake.FbxScene                      = types.SimpleNamespace(Create=MagicMock())
    return fake


def _wire_importer(
    fake_fbx:      types.SimpleNamespace,
    *,
    root_node,
    initialize_ok: bool                  = True,
    import_ok:     bool                  = True,
    error:         str                   = "fbx error",
) -> MagicMock:
    """wires the factory ``Create`` calls to mocks that give ``root_node``
    when the scene is asked; returns the manager mock, for its ``Destroy``"""
    manager  = MagicMock(name="manager")
    ios      = MagicMock(name="ios")
    importer = MagicMock(name="importer")
    scene    = MagicMock(name="scene")

    importer.Initialize.return_value                            = initialize_ok
    importer.Import.return_value                                = import_ok
    importer.GetStatus.return_value.GetErrorString.return_value = error
    scene.GetRootNode.return_value                              = root_node
    manager.GetIOSettings.return_value                          = ios

    unit                             = scene.GetGlobalSettings.return_value.GetSystemUnit.return_value
    unit.GetScaleFactor.return_value = 1.0

    fake_fbx.FbxManager.Create.return_value    = manager
    fake_fbx.FbxIOSettings.Create.return_value = ios
    fake_fbx.FbxImporter.Create.return_value   = importer
    fake_fbx.FbxScene.Create.return_value      = scene
    return manager


def _make_fake_root(children):
    """a root node exposing ``children``"""
    root                               = MagicMock(name="root")
    root.GetNodeAttribute.return_value = None
    root.GetChildCount.return_value    = len(children)
    root.GetChild.side_effect          = lambda i, _c=children: _c[i]
    return root


def _make_real_file(suffix: str = ".fbx") -> str:
    """a real file on disk, so the readers' existence check passes"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


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


def base_color(path):
    """the first material's colour map, (H, W, C) uint8 as formats.load_fbx
    reads it; None for no material or no map"""
    from cgmath.formats import load_fbx

    materials = load_fbx(path).materials
    if not materials or materials[0].texture is None:
        return None
    return np.asarray(materials[0].texture)


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
                # the node readers keep the mesh's node, its shape left out
                self.assertEqual(load(self.rig).name, ["body_geo", "root", "tip"])

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

    def test_formats_and_render(self):
        from cgmath.formats import load_fbx
        from cgmath.render import Scene

        with balanced(self):
            held = load_fbx(self.rig)
        self.assertIsNotNone(held.skins)
        with balanced(self):
            scene = Scene.load_fbx(self.rig)
        self.assertIsNotNone(scene.skins)

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

    def test_a_mesh_without_a_material_has_no_colour_map(self):
        with balanced(self):
            self.assertIsNone(base_color(self.rig))


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
        before = scratch_folders()
        with balanced(self):
            texture = base_color(self.quad)
        np.testing.assert_array_equal(texture[0, 0, :3], [0, 255, 0])
        self.assert_nothing_written(before)

    def test_scene(self):
        from cgmath.render import Scene

        before = scratch_folders()
        self.assertIsNotNone(Scene.load_fbx(self.quad).materials[0].texture)
        self.assert_nothing_written(before)

    def test_readers_extract_nothing(self):
        before = scratch_folders()
        hierarchy_mod.HierarchyData.load_fbx(self.quad)
        mesh_mod.load_fbx(self.quad)
        self.assert_nothing_written(before)

    def test_a_non_ascii_temp_folder_falls_back_beside_the_file(self):
        # the fbx binding cannot hand a non-ASCII path to the SDK
        folder = self.file("fallback")
        os.makedirs(folder)
        source = os.path.join(folder, "quad.fbx")
        with open(self.quad, "rb") as f, open(source, "wb") as g:
            g.write(f.read())

        mkdtemp = tempfile.mkdtemp

        def unicode_temp(prefix="", **kwargs):
            return mkdtemp(prefix=prefix + "\u00e9_", **kwargs)

        # on a volume keeping no short (8.3) names, which new folders here
        # may well have anyway
        def no_short_name(path):
            return path

        with patch.object(_fbx_io.tempfile, "mkdtemp", unicode_temp), patch.object(
            _fbx_io, "short_path", no_short_name, create=True
        ):
            with self.assertWarnsRegex(RuntimeWarning, "ASCII"):
                texture = base_color(source)

        np.testing.assert_array_equal(texture[0, 0, :3], [0, 255, 0])
        self.assertEqual(sorted(os.listdir(folder)), ["quad.fbm", "quad.fbx"])

    @unittest.skipUnless(os.name == "nt", "short (8.3) names and junctions are Windows'")
    def test_a_non_ascii_temp_folder_is_reached_by_its_short_name(self):
        import _winapi
        import warnings

        # a temp folder by a non-ASCII path, and an ASCII path to the same
        # folder standing in for its short name: giving it a real one takes
        # an administrator
        folder = self.file("short_name")
        temp   = os.path.join(folder, "temp \u00e9")
        alias  = os.path.join(folder, "TEMP~1")
        os.makedirs(temp)
        _winapi.CreateJunction(temp, alias)
        self.addCleanup(os.rmdir, alias)  # the junction, not what it points at

        source = os.path.join(folder, "quad.fbx")
        with open(self.quad, "rb") as f, open(source, "wb") as g:
            g.write(f.read())

        mkdtemp = tempfile.mkdtemp

        def in_temp(prefix="", **kwargs):
            return mkdtemp(prefix=prefix, dir=temp)

        def short_name(path):
            return path.replace(temp, alias)

        with patch.object(_fbx_io.tempfile, "mkdtemp", in_temp), patch.object(
            _fbx_io, "short_path", short_name, create=True
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            texture = base_color(source)

        self.assertEqual([str(x.message) for x in caught if "ASCII" in str(x.message)], [])
        np.testing.assert_array_equal(texture[0, 0, :3], [0, 255, 0])

        # nothing beside the file, and the media folders are gone
        self.assertEqual(sorted(os.listdir(folder)), ["TEMP~1", "quad.fbx", "temp \u00e9"])
        self.assertEqual(os.listdir(temp), [])

    def test_typed(self):
        manager = fbx.FbxManager.Create()
        try:
            scene   = fbx.FbxScene.Create(manager, "")
            texture = fbx.FbxFileTexture.Create(scene, "tex")
            texture.SetFileName("C:/a/b.png")
            base = fbx.cast(texture, fbx.FbxObject)
            self.assertNotIsInstance(base, fbx.FbxFileTexture)

            again = _fbx_io.typed(base, fbx.FbxFileTexture, fbx)
            self.assertIsInstance(again, fbx.FbxFileTexture)
            self.assertEqual(again.GetUniqueID(), texture.GetUniqueID())
            self.assertEqual(again.GetFileName(), "C:/a/b.png")
            self.assertIs(_fbx_io.typed(texture, fbx.FbxFileTexture, fbx), texture)
            self.assertIsNone(_fbx_io.typed(None, fbx.FbxFileTexture, fbx))
        finally:
            manager.Destroy()


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
    """the failure paths, through a fake SDK, so they run without one: every
    reader opens its file through formats._fbx_io with the sdk
    cgmath.hierarchy.hierarchy holds"""

    def setUp(self):
        self.path = _make_real_file()
        self.addCleanup(os.unlink, self.path)

    def check(self, load):
        for wiring in ({"initialize_ok": False}, {"import_ok": False}):
            fake = _make_fake_fbx_module()
            with self.subTest(**wiring), patch.object(hierarchy_mod, "FBX", fake):
                manager = _wire_importer(
                    fake, root_node=_make_fake_root([]), error="bad header", **wiring
                )
                with self.assertRaisesRegex(_fbx_io.FbxReadError, "bad header"):
                    load(self.path)
                manager.Destroy.assert_called_once()

    def test_hierarchy(self):
        self.check(hierarchy_mod.HierarchyData.load_fbx)

    def test_clip(self):
        self.check(hierarchy_mod.ClipData.load_fbx)

    def test_skin(self):
        self.check(skin_mod.SkinList.load_fbx)

    def test_mesh(self):
        self.check(mesh_mod.load_fbx)

    def test_formats(self):
        from cgmath.formats import load_fbx

        self.check(load_fbx)


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

    def test_short_path(self):
        # a folder comes back by its short (8.3) name when it has one, else
        # as given, as a path that does not exist does
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (here, os.environ.get("ProgramFiles", here)):
            with self.subTest(path=path):
                short = _fbx_io.short_path(path)
                self.assertTrue(os.path.samefile(short, path))
                self.assertTrue(short == path or "~" in short, short)

        missing = os.path.join(here, "no such folder \u00e9")
        self.assertEqual(_fbx_io.short_path(missing), missing)


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

    def test_the_old_readers_are_gone(self):
        from cgmath import hierarchy
        from cgmath.formats import fbx as fbx_mod

        self.assertFalse(hasattr(hierarchy, "_read_fbx"))
        for name in ("SceneData", "TakeData", "LayerData", "CurveData", "TakeList", "LayerList", "CurveList"):
            self.assertFalse(hasattr(fbx_mod, name), name)
        self.assertFalse(hasattr(mesh_mod, "_load_fbx"))
        self.assertFalse(hasattr(skin_mod, "_read_skins_fbx"))


if __name__ == "__main__":
    unittest.main()
