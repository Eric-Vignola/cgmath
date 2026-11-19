"""FbxFile: an fbx file held open, read a kind at a time, saved back as it was read.

Each property equals load_fbx's field of the same name, in any order the
properties are read, the nodes' uuids shared by all of them; a property
reads what it depends on alone. A save of the untouched file writes the
SDK's own scene: everything the SDK reads is kept, the fork's kinds and what
the fork does not read (a bind pose, an LOD group, user properties) alike,
in the file's own axes and unit, its embedded media embedded again. A record
added, removed or moved under another node refuses the save (the values of
a record changed: test_fbx_edit). The tests need the Autodesk SDK; the Maya
files were written by Maya 2025, recorded once (maya_fixtures), and need no
Maya.
"""

import collections
import copy
import os
import pickle
import unittest
import warnings

import numpy as np
from cgmath._tests.maya_fixtures import recorded, same_read
from cgmath._tests.test_fbx_materials import box, materials, plane, plane_uvs
from cgmath._tests.test_fbx_morphs import face_morphs
from cgmath._tests.test_fbx_nurbs import sheet, wave
from cgmath._tests.test_skeleton_fbx import _Files, bundle
from cgmath.formats import FbxContents, FbxFile, load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.geometry import (
    CameraData,
    CameraList,
    LightData,
    LightList,
    MeshList,
    NurbsCurveData,
    NurbsCurveList,
    NurbsSurfaceData,
    NurbsSurfaceList,
)

FBX = load_sdk()

FIELDS = FbxContents._fields

# Maya's files, every kind and quirk the reader takes among them: skinned,
# Z-up in cm and in metres, embedded and linked textures, keyed blend
# shapes, cameras and lights (an ambient and an area light warned on), a
# camera keyed by another writer, NURBS shapes, pivots, vertex colours,
# several takes, no UVs, a keyed visibility
MAYA_FILES = (
    ("skin_geo", "maya.fbx"),
    ("uvs_export", "maya.fbx"),
    ("zup_skinned", "z_cm.fbx"),
    ("zup", "z_m.fbx"),
    ("materials_export", "maya_embedded.fbx"),
    ("materials_export", "maya_linked.fbx"),
    ("normals_export", "maya_embedded.fbx"),
    ("morphs_export", "maya.fbx"),
    ("views_export", "maya.fbx"),
    ("other_writer", "other.fbx"),
    ("nurbs_export", "maya_rational_surface.fbx"),
    ("pivots_export", "maya.fbx"),
    ("colors_export", "shared.fbx"),
    ("takes_each", "ours.fbx"),
    ("uvs_none", "maya.fbx"),
    ("visibility_locator", "maya.fbx"),
    ("two_meshes", "ours.fbx"),
    ("rrss", "source.fbx"),
)

USER = None if FBX is None else FBX.FbxPropertyFlags.EFlags.eUserDefined


def maya_file(case: str, name: str) -> str:
    return recorded(case).path(name)


def every_kind(path: str) -> str:
    """a file of every kind the fork reads: bundle()'s bind pose, its clip,
    the skinned body and its skin; the materials test's plane and box, the
    plane's UV set, its textured lambert (an embedded texture) and phong,
    the box's phong, and the morphs test's targets; a camera, a spot light,
    a curve and a surface. Every shape the bind pose holds no node for goes
    on a root node of its own"""
    data   = bundle()
    shapes = MeshList([plane(), box()])
    held   = MeshList([data["meshes"], shapes[0], shapes[1]])
    save_fbx(
        path,
        hierarchy = data["hierarchy"],
        clip      = data["clip"],
        meshes    = held,
        skins     = data["skins"],
        uvs       = plane_uvs(shapes[0]),
        cameras   = CameraList([CameraData(name="shotShape", focal_length=50.0)]),
        lights    = LightList([LightData(name="keyShape", kind="spot", intensity=2.0)]),
        materials = materials(shapes),
        morphs    = face_morphs(shapes),
        curves    = NurbsCurveList([NurbsCurveData(points=wave(6), degree=3)]),
        surfaces  = NurbsSurfaceList([NurbsSurfaceData(points=sheet(4, 4), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False)]),
    )
    return path


def extended(source: str, path: str) -> str:
    """``source`` with what the fork does not read added through the SDK,
    written by the handle's own save: an LOD group on a new root node, a
    user property on it, and a user property on the first node"""
    with FbxFile(source, verbose=False) as f:
        scene = f._scene
        group = FBX.FbxLODGroup.Create(scene, "lodShape")
        node  = FBX.FbxNode.Create(scene, "lod")
        node.SetNodeAttribute(group)
        scene.GetRootNode().AddChild(node)
        for owner, name in ((node, "lodNote"), (scene.GetRootNode().GetChild(0), "rigNote")):
            prop = FBX.FbxProperty.Create(owner, FBX.FbxDoubleDT, name)
            prop.ModifyFlag(USER, True)
            FBX.FbxPropertyDouble1(prop).Set(2.5)
        f.save(path)
    return path


def census(path: str) -> tuple:
    """what the SDK reads of a file: its objects by class, its properties,
    its user properties, its curve keys, and its poses"""
    kinds = collections.Counter()
    props = user = keys = 0
    with open_fbx(path, FBX) as scene:
        for i in range(scene.GetSrcObjectCount()):
            held = scene.GetSrcObject(i)
            kinds[str(held.GetClassId().GetName())] += 1
            prop = held.GetFirstProperty()
            while prop.IsValid():
                props += 1
                user  += bool(prop.GetFlag(USER))
                prop = held.GetNextProperty(prop)
        wanted = FBX.FbxCriteria.ObjectType(FBX.FbxAnimCurve.ClassId)
        for i in range(scene.GetSrcObjectCount(wanted)):
            keys += typed(scene.GetSrcObject(wanted, i), FBX.FbxAnimCurve, FBX).KeyGetCount()
        poses = scene.GetPoseCount()
    return dict(kinds), props, user, keys, poses


def read(f: FbxFile, order) -> FbxContents:
    """the handle's properties read in ``order``, as one load_fbx result"""
    for field in order:
        getattr(f, field)
    return FbxContents(*(getattr(f, field) for field in FIELDS))


def one(field: str, value):
    """a load_fbx-like result of one field, for same_read"""
    return collections.namedtuple("One", (field,))(value)


def node_uuids(held) -> list:
    """the uuids of the nodes a read's shapes sit on: the hierarchy's,
    else the clip's"""
    if held.hierarchy is not None:
        return [node.uuid for node in held.hierarchy]
    if held.clip is not None:
        return [node.uuid for node in held.clip[0]]
    return []


class _Handles(_Files):
    def assertAgrees(self, path: str, label: str = "", **options) -> None:
        """the handle's properties equal load_fbx's fields, read forwards,
        backwards and each alone, every link naming a record of its own
        handle"""
        want = load_fbx(path, verbose=False, **options)
        for name, order in (("forwards", FIELDS), ("backwards", FIELDS[::-1])):
            with FbxFile(path, verbose=False, **options) as f:
                got = read(f, order)
            same_read(self, got, want, f"{label}{name}: ")
            self.assertLinked(got, f"{label}{name}: ")
        for field in FIELDS:
            with FbxFile(path, verbose=False, **options) as f:
                same_read(self, one(field, getattr(f, field)), one(field, getattr(want, field)), f"{label}alone: ")

    def assertLinked(self, held, label: str = "") -> None:
        """every shape on a node of the read, every record on a mesh a mesh
        of it; the bind pose's nodes the clip's, uuid for uuid"""
        nodes  = set(node_uuids(held))
        meshes = {mesh.uuid for mesh in held.meshes or ()}
        for field in ("meshes", "cameras", "lights", "curves", "surfaces"):
            for item in getattr(held, field) or ():
                self.assertIn(item.parent_uuid, nodes, f"{label}{field}")
        for field in ("skins", "uvs", "materials", "morphs"):
            for item in getattr(held, field) or ():
                self.assertIn(item.mesh_uuid, meshes, f"{label}{field}")
        if held.hierarchy is not None and held.clip is not None:
            for take in held.clip:
                self.assertEqual([node.uuid for node in take], [node.uuid for node in held.hierarchy], f"{label}clip")


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestAgrees(_Handles):
    """each property is load_fbx's field, in any read order"""

    def test_every_kind(self):
        path = every_kind(self.path("every.fbx"))
        held = load_fbx(path, verbose=False)
        self.assertEqual([field for field in FIELDS if getattr(held, field) is None], [])
        self.assertAgrees(path)

    def test_maya_files(self):
        for case, name in MAYA_FILES:
            with self.subTest(case=case, name=name):
                self.assertAgrees(maya_file(case, name), f"{case}/{name} ")

    def test_textures_off(self):
        """no pixels decoded, each texture's file named, as load_fbx(...,
        textures=False) reads it"""
        path = maya_file("materials_export", "maya_embedded.fbx")
        self.assertAgrees(path, textures=False)
        with FbxFile(path, textures=False, verbose=False) as f:
            self.assertTrue(f.materials)
            self.assertTrue(all(material.texture is None for material in f.materials))
            self.assertTrue(any(material.texture_file for material in f.materials))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestReadsAlone(_Handles):
    """a property reads what it depends on, nothing more"""

    def test_hierarchy(self):
        """the bind pose off a read of the skins alone: no mesh read, no
        record built but the nodes and the clip they ask after"""
        with FbxFile(maya_file("skin_geo", "maya.fbx"), verbose=False) as f:
            self.assertIsNotNone(f.hierarchy)
            self.assertTrue(f._reads)
            self.assertFalse([kinds for kinds in f._reads if "meshes" in kinds])
            self.assertLessEqual(set(f._values), {"hierarchy", "clip"})

    def test_meshes(self):
        """the meshes build no material"""
        with FbxFile(maya_file("materials_export", "maya_embedded.fbx"), verbose=False) as f:
            self.assertTrue(f.meshes)
            self.assertNotIn("materials", f._values)
            self.assertFalse([kinds for kinds in f._reads if "materials" in kinds])

    def test_curves(self):
        with FbxFile(maya_file("nurbs_export", "maya_rational_surface.fbx"), verbose=False) as f:
            self.assertTrue(f.curves)
            self.assertEqual(list(f._reads), [frozenset(("curves", "surfaces"))])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSave(_Handles):
    """an untouched save writes the SDK's own scene back"""

    def assertKept(self, source: str, saved: str) -> None:
        self.assertEqual(census(saved), census(source))
        same_read(self, load_fbx(saved, verbose=False), load_fbx(source, verbose=False), "saved: ")

    def test_untouched(self):
        """a bind pose, an LOD group and user properties, which the fork
        does not read, kept by a save with nothing read and by one after
        every property"""
        every  = every_kind(self.path("every.fbx"))
        source = extended(every, self.path("extended.fbx"))
        kinds, _, user, _, poses = census(source)
        self.assertEqual((kinds.get("FbxLODGroup"), user, poses), (1, census(every)[2] + 2, 1))
        with FbxFile(source, verbose=False) as f:
            self.assertEqual(f.save(self.path("unread.fbx")), self.path("unread.fbx"))
            read(f, FIELDS)
            f.save(self.path("read.fbx"))
        for name in ("unread.fbx", "read.fbx"):
            with self.subTest(name=name):
                self.assertKept(source, self.path(name))

    def test_embedded_media(self):
        """a texture the file embeds is embedded again, its pixels the
        file's, whether the handle decoded them or not; no media folder is
        left beside the file"""
        for case, name in (("materials_export", "maya_embedded.fbx"), ("normals_export", "maya_embedded.fbx")):
            source = maya_file(case, name)
            want   = load_fbx(source, verbose=False).materials
            for textures in (True, False):
                with self.subTest(case=case, textures=textures):
                    out = self.path(f"{case}_{textures}.fbx")
                    with FbxFile(source, textures=textures, verbose=False) as f:
                        f.materials
                        f.save(out)
                    got = load_fbx(out, verbose=False).materials
                    self.assertEqual(len(got), len(want))
                    for x, y in zip(got, want):
                        self.assertEqual(x, y)
                    self.assertTrue(any(x.texture is not None or x.normal_map is not None for x in got))
                    self.assertEqual(sorted(os.listdir(self.folder)), sorted(x for x in os.listdir(self.folder) if x.endswith(".fbx")))

    def test_writer_adds_a_video(self):
        """the SDK's writer adds one empty FbxVideo (20 properties) to a
        file holding a material's binding table (Maya's standardSurface)
        beside a file texture; nothing else differs, the read is the
        file's, and a save of the save adds none"""
        for name in ("maya_embedded.fbx", "maya_linked.fbx"):
            with self.subTest(name=name):
                source = maya_file("materials_export", name)
                first  = self.path(f"first_{name}")
                again  = self.path(f"again_{name}")
                with FbxFile(source, verbose=False) as f:
                    f.save(first)
                with FbxFile(first, verbose=False) as f:
                    f.save(again)
                kinds, props, user, keys, poses = census(source)
                kinds["FbxVideo"] = kinds.get("FbxVideo", 0) + 1
                self.assertEqual(census(first), (kinds, props + 20, user, keys, poses))
                self.assertEqual(census(again), census(first))
                same_read(self, load_fbx(first, verbose=False), load_fbx(source, verbose=False), "saved: ")

    def test_axes_and_unit(self):
        """a Z-up file in metres comes back Z-up in metres"""
        source = maya_file("zup", "z_m.fbx")
        out    = self.path("z_m.fbx")
        with FbxFile(source, verbose=False) as f:
            f.clip
            f.save(out)

        def axes(path):
            with open_fbx(path, FBX) as scene:
                settings = scene.GetGlobalSettings()
                return tuple(settings.GetAxisSystem().GetUpVector()), settings.GetSystemUnit().GetScaleFactor()

        self.assertEqual(axes(out), axes(source))
        self.assertEqual(axes(out)[1], 100.0)
        self.assertKept(source, out)

    def test_ascii(self):
        """every object kept; the SDK's ASCII writer prints a double to
        within 1e-13 of it (measured: 4e-14 on a bind pose's worlds)"""
        source = maya_file("skin_geo", "maya.fbx")
        out    = self.path("ascii.fbx")
        with FbxFile(source, verbose=False) as f:
            f.save(out, as_ascii=True)
        with open(out, "rb") as handle:
            self.assertTrue(handle.read(64).lstrip().startswith(b";"))
        self.assertEqual(census(out), census(source))
        got, want = load_fbx(out, verbose=False), load_fbx(source, verbose=False)
        self.assertEqual([node.name for node in got.hierarchy], [node.name for node in want.hierarchy])
        np.testing.assert_allclose(np.asarray(got.hierarchy.world_matrix), np.asarray(want.hierarchy.world_matrix), rtol=0, atol=1e-13)
        np.testing.assert_allclose(np.asarray(got.meshes[0].points), np.asarray(want.meshes[0].points), rtol=0, atol=1e-13)

    def test_new_folder(self):
        source = maya_file("rrss", "source.fbx")
        out    = os.path.join(self.folder, "made", "here", "rrss.fbx")
        with FbxFile(source, verbose=False) as f:
            self.assertEqual(f.save(out), out)
        self.assertKept(source, out)

    def test_layout_refused(self):
        """a shape added on no node of the file, and a node moved under
        another in the hierarchy alone of a file whose takes hold it too,
        refuse the save, naming what to do; the target is left as it was
        and nothing else is written"""

        def reparented(f):
            f.hierarchy.list[2].parent_uuid = f.hierarchy.list[0].uuid

        source = every_kind(self.path("every.fbx"))
        target = self.path("target.fbx")
        for change, said in (
            (lambda f: f.cameras.append(CameraData(name="extraShape")), "sits on no node of the file"),
            (reparented, "under another parent, or as another kind, than the hierarchy; a file holds one set of nodes"),
        ):
            with self.subTest(said=said):
                with open(target, "wb") as handle:
                    handle.write(b"before")
                with FbxFile(source, verbose=False) as f:
                    change(f)
                    with self.assertRaises(ValueError) as caught:
                        f.save(target)
                self.assertIn(said, str(caught.exception))
                self.assertTrue(str(caught.exception).startswith("FbxFile.save: "), str(caught.exception))
                with open(target, "rb") as handle:
                    self.assertEqual(handle.read(), b"before")
                self.assertEqual(sorted(os.listdir(self.folder)), ["every.fbx", "target.fbx"])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestHandle(_Handles):
    """open, closed, warned, neither pickled nor copied"""

    def test_closed(self):
        """the values read stay; an unread property and a save raise; the
        extracted media go"""
        f = FbxFile(maya_file("materials_export", "maya_embedded.fbx"), verbose=False)
        with f:
            meshes = f.meshes
            media  = f._media
            self.assertTrue(os.listdir(media))
            self.assertFalse(f.closed)
        self.assertTrue(f.closed)
        self.assertFalse(os.path.exists(media))
        self.assertIs(f.meshes, meshes)
        with self.assertRaisesRegex(ValueError, "is closed"):
            f.uvs
        with self.assertRaisesRegex(ValueError, "is closed"):
            f.save(self.path("closed.fbx"))
        f.close()
        self.assertIn("closed", repr(f))
        self.assertFalse(os.listdir(self.folder))

    def test_not_pickled_or_copied(self):
        with FbxFile(maya_file("rrss", "source.fbx"), verbose=False) as f:
            for call in (pickle.dumps, copy.copy, copy.deepcopy):
                with self.assertRaisesRegex(TypeError, "neither pickled nor copied"):
                    call(f)
            self.assertEqual(pickle.loads(pickle.dumps(f.hierarchy)), f.hierarchy)

    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            FbxFile(self.path("none.fbx"))

    def test_warnings(self):
        """the warnings load_fbx gives, each once a handle; none quiet"""
        path = maya_file("other_writer", "other.fbx")
        with warnings.catch_warnings(record=True) as want:
            warnings.simplefilter("always")
            load_fbx(path)
        self.assertTrue(want)
        with FbxFile(path) as f:
            with warnings.catch_warnings(record=True) as got:
                warnings.simplefilter("always")
                read(f, FIELDS)
            with warnings.catch_warnings(record=True) as again:
                warnings.simplefilter("always")
                read(f, FIELDS[::-1])
        self.assertEqual(sorted(str(x.message) for x in got), sorted(str(x.message) for x in want))
        self.assertEqual(again, [])
        with FbxFile(path, verbose=False) as f:
            with warnings.catch_warnings(record=True) as quiet:
                warnings.simplefilter("always")
                read(f, FIELDS)
        self.assertEqual(quiet, [])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestRefused(_Handles):
    """a file load_fbx refuses opens and saves whole"""

    def test_trimmed_surface(self):
        """the surfaces raise, every time; the rest is load_fbx's read with
        the surfaces off; the save keeps the trimmed surface"""
        source = maya_file("nurbs_export", "maya_trimmed.fbx")
        with self.assertRaisesRegex(ValueError, "trimmed"):
            load_fbx(source, verbose=False)
        want = load_fbx(source, surfaces=False, verbose=False)
        out  = self.path("trimmed.fbx")
        with FbxFile(source, verbose=False) as f:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "trimmed"):
                    f.surfaces
            got = FbxContents(*(None if field == "surfaces" else getattr(f, field) for field in FIELDS))
            f.save(out)
        same_read(self, got, want)
        self.assertTrue(got.curves)
        self.assertEqual(census(out), census(source))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestScene(_Handles):
    """Scene.load_fbx reads a handle as it reads the file"""

    def test_scene(self):
        from cgmath.render.scene import _FBX_FIELDS, Scene

        path = maya_file("uvs_export", "maya.fbx")
        want = Scene.load_fbx(path, verbose=False)
        with FbxFile(path, verbose=False) as f:
            got = Scene.load_fbx(f)
            self.assertEqual(got.name, want.name)
            self.assertIs(got.meshes, f.meshes)
            # the scene holds the handle's records, unchanged: the save goes
            f.save(self.path("scene.fbx"))
            with self.assertRaisesRegex(TypeError, "textures"):
                Scene.load_fbx(f, textures=False)

        def fields(scene):
            return FbxContents(*(getattr(scene, field) if field in _FBX_FIELDS else None for field in FIELDS))

        same_read(self, fields(got), fields(want))



@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestUnread(_Handles):
    """what a file holds that no record does: load_fbx says so, FbxFile
    lists it and keeps it"""

    WANT = {
        "LOD group": 1,
        "display layer": 1,
        "selection sets": 2,
        "constraint": 1,
        "HumanIK character": 1,
        "rest pose": 1,
        "user properties on shapes, materials or other objects that are no nodes": 2,
    }

    def built(self) -> str:
        """every kind, and one or two of each thing the fork does not read"""
        source = every_kind(self.path("every.fbx"))
        path   = self.path("extras.fbx")
        with FbxFile(source, verbose=False) as f:
            scene = f._scene
            node  = FBX.FbxNode.Create(scene, "lod")
            node.SetNodeAttribute(FBX.FbxLODGroup.Create(scene, "lodShape"))
            scene.GetRootNode().AddChild(node)
            FBX.FbxDisplayLayer.Create(scene, "shown")
            FBX.FbxSelectionSet.Create(scene, "one")
            FBX.FbxSelectionSet.Create(scene, "two")
            FBX.FbxConstraintPosition.Create(scene, "pinned")
            FBX.FbxCharacter.Create(scene, "hero")
            pose = FBX.FbxPose.Create(scene, "rest")
            pose.SetIsBindPose(False)
            scene.AddPose(pose)
            for kind in (FBX.FbxMesh, FBX.FbxSurfaceMaterial):
                owner = scene.GetSrcObject(FBX.FbxCriteria.ObjectType(kind.ClassId), 0)
                prop  = FBX.FbxProperty.Create(owner, FBX.FbxDoubleDT, "note")
                prop.ModifyFlag(USER, True)
            f.save(path)
        return path

    def notices(self, call) -> list:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        return [str(warning.message) for warning in caught if "does not read and save_fbx would not write" in str(warning.message)]

    def test_load_fbx_says_so(self):
        """one warning a read of every kind, naming each thing and how
        many; none with verbose off, for a read of some kinds, or for a
        file holding none"""
        path = self.built()
        told = self.notices(lambda: load_fbx(path))
        self.assertEqual(len(told), 1, told)
        for what, count in self.WANT.items():
            self.assertIn(f"{count} {what}", told[0])
        self.assertIn("FbxFile keeps them", told[0])
        self.assertEqual(self.notices(lambda: load_fbx(path, textures=False)),   told)
        self.assertEqual(self.notices(lambda: load_fbx(path, verbose=False)),    [])
        self.assertEqual(self.notices(lambda: MeshList.load_fbx(path)),          [])
        self.assertEqual(self.notices(lambda: load_fbx(path, cameras=False)),    [])
        self.assertEqual(self.notices(lambda: load_fbx(self.path("every.fbx"))), [])

    def test_handle_lists_and_keeps(self):
        """FbxFile.unread() is the list, in words; a save of the handle
        keeps every item, a property read and changed or not"""
        path = self.built()
        with FbxFile(path, verbose=False) as f:
            self.assertEqual(f.unread(), self.WANT)
            f.hierarchy
            f.meshes[0].name = "renamed"
            f.save(self.path("saved.fbx"))
            self.assertEqual(f.unread(), self.WANT)
        with FbxFile(self.path("saved.fbx"), verbose=False) as f:
            self.assertEqual(f.unread(), self.WANT)
        with FbxFile(self.path("every.fbx"), verbose=False) as f:
            self.assertEqual(f.unread(), {})
        with self.assertRaises(ValueError):
            f.unread()


if __name__ == "__main__":
    unittest.main()
