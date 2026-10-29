"""load() and save() route a file to the class's own load_<format> / save_<format>.

The format is the one ``mode`` names, else, for a load, the type the file
reads as (get_file_type), for a save, its extension; an extension no route
has is refused, naming the routes there are. Arguments past the file name
and the mode go to the route, which checks them before anything is read or
written. A route a class inherits but refuses (a _NoRoute) raises with its
reason; an asset's reader gives back the class itself. The obj writers:
MeshData.save_obj writes the normals a mesh has, Object.save_obj its mesh
where the Object puts it, with its UVs and normals, both in utf-8, which the
obj readers read first.
"""

import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from json import JSONDecodeError
from unittest import mock

import numpy as np
from cgmath import hierarchy
from cgmath.data import Data, DataList, LegacyFileError, get_file_type
from cgmath.formats.glb import GlbData
from cgmath.geometry import mesh as mesh_module
from cgmath.geometry import skin_weights
from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
from cgmath.geometry.skin_weights import SkinList
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath.render.scene import Object, Scene

try:
    import fbx
except ImportError:
    fbx = None

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import charset_normalizer  # trimesh's reader of text in another encoding
except ImportError:
    charset_normalizer = None

try:
    import pygltflib
    from cgmath._tests.test_object_skin import write_glb
except ImportError:
    pygltflib = None

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None


@dataclass
class _Quad(MeshData):
    """a MeshData subclass with no reader of its own"""


@dataclass
class _Stl(MeshData):
    """a MeshData subclass reading and writing a format of its own"""

    @classmethod
    def load_stl(cls, filename, scale=1.0):
        with open(filename) as f:
            return cls(points=np.loadtxt(f, ndmin=2) * scale, indices=np.arange(3), counts=np.array([3]))

    def save_stl(self, filename):
        np.savetxt(filename, self.points)


@dataclass
class _Loose(MeshData):
    """a MeshData subclass whose stl reader gives back a class deriving from it"""

    @classmethod
    def load_stl(cls, filename):
        return _Looser(points=np.eye(3), indices=np.arange(3), counts=np.array([3]))


@dataclass
class _Looser(_Loose):
    """derives from _Loose"""


def quad(name="quadShape"):
    return MeshData(
        points  = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        indices = np.arange(4),
        counts  = np.array([4]),
        name    = name,
    )


def cube(name="cubeShape"):
    """a unit cube of six quads, its points not all on the grid"""
    points = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=float,
    ) * 0.3
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    return MeshData(points=points, indices=np.array(quads).ravel(), counts=np.full(6, 4), name=name)


def cube_uv():
    """one UV island a face, 24 UVs, the faces naming them last first"""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) * 0.15
    points  = np.concatenate([corners + [0.16 * face, 0.1] for face in range(6)])
    return UVData(points=points[::-1], indices=np.arange(24)[::-1], counts=np.full(6, 4), name="map1")


def quad_uv(name="map1", offset=0.0):
    """one UV a corner of quad()"""
    return UVData(points=np.zeros((4, 2)) + offset, indices=np.arange(4), counts=np.array([4]), name=name)


def corner_normals(mesh):
    """the normal each face corner has"""
    normals = np.asarray(mesh.normals)
    if mesh.normal_indices is not None:
        return normals[mesh.normal_indices]
    if len(normals) == mesh.point_count:
        return normals[mesh.indices]
    return normals


class _Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)


class TestInheritedRoutes(unittest.TestCase):
    """a class reaches its base's asset routes by getattr: each one must fit it, or be refused"""

    # the inherited asset routes known to fit: TransformList's readers and
    # writer build ``cls`` and know nothing a HierarchyData adds
    FIT = {
        ("HierarchyData", "load_fbx"),
        ("HierarchyData", "load_glb"),
        ("HierarchyData", "save_fbx"),
    }

    def test_every_inherited_asset_route_fits_or_is_refused(self):
        from cgmath.data import _CLASSES, _FORMATS, _OWN_FORMATS, _route_of

        found = set()
        for tag, cls in _CLASSES.items():
            if not tag.startswith("cgmath.") or "._tests" in tag:
                continue
            for verb in ("load", "save"):
                for fmt in set(_FORMATS) - set(_OWN_FORMATS):
                    name = f"{verb}_{fmt}"
                    if _route_of(cls, verb, fmt) is not None and name not in vars(cls):
                        found.add((cls.__name__, name))
        self.assertEqual(found, self.FIT)

    @unittest.skipIf("maya" in sys.modules, "starts a plain interpreter, not a second mayapy")
    def test_the_factory_names_readers_it_has_not_imported(self):
        code = (
            "import cgmath; from cgmath.data import _readers_of; "
            "print(cgmath.__file__); print(','.join(_readers_of('fbx')))"
        )
        root  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out   = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
        lines = out.stdout.splitlines()
        self.assertEqual(len(lines), 2, out.stderr[-500:])
        self.assertTrue(os.path.abspath(lines[0]).startswith(root), lines[0])
        self.assertEqual(
            lines[1].split(","),
            ["ClipData", "HierarchyData", "MeshData", "MeshList", "Object", "SkinData", "SkinList",
             "TransformList", "UVData", "UVList"],
        )


class TestOwnFormats(_Files):
    """a format a class reads or writes itself is routed as cgmath's are"""

    def test_a_class_routes_its_own_format(self):
        triangle = _Stl(points=np.eye(3), indices=np.arange(3), counts=np.array([3]))
        path     = triangle.save(self.path("t.STL"))
        np.testing.assert_array_equal(_Stl.load(path, scale=2.0).points, 2.0 * np.eye(3))
        with self.assertRaisesRegex(TypeError, r"_Stl\.load_stl: .*'size'.*takes scale$"):
            _Stl.load(path, size=2.0)
        with self.assertRaisesRegex(ValueError, "_Stl has no reader for ply files.*loads fbx, glb, json, npz, obj, pickle, stl, usd$"):
            _Stl.load(self.path("t.ply"), mode="ply")
        with self.assertRaisesRegex(ValueError, "MeshData has no reader for stl files"):
            MeshData.load(path)

    def test_a_user_attribute_is_no_route(self):
        node = TransformData("node")
        node.add_user_attribute("save_obj", 1.0)
        self.assertEqual(node.save_obj, 1.0)
        with self.assertRaisesRegex(ValueError, "TransformData has no writer for obj files.*saves json, npz, pickle$"):
            node.save(self.path("node.obj"))
        self.assertFalse(os.path.exists(self.path("node.obj")))


class TestSave(_Files):
    def test_an_obj_goes_to_save_obj_and_back(self):
        path = quad().save(self.path("q.obj"))
        self.assertEqual(path, self.path("q.obj"))
        np.testing.assert_allclose(MeshData.load(path).points, quad().points)

    def test_mode_names_the_format_whatever_the_extension(self):
        path = quad().save(self.path("q.dat"), mode="obj")
        np.testing.assert_allclose(MeshData.load(path, mode="obj").points, quad().points)
        path = quad().save(self.path("q.OBJ"))
        self.assertEqual(MeshData.load(path), quad())

    def test_arguments_go_to_the_writer(self):
        path = quad().save(self.path("q.npz"), compression=zipfile.ZIP_STORED)
        with zipfile.ZipFile(path) as archive:
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist()))

    def test_an_argument_the_writer_does_not_take_raises_before_writing(self):
        with self.assertRaisesRegex(TypeError, r"MeshData\.save_json: .*'compression'.*takes no argument"):
            quad().save(self.path("q.json"), compression=zipfile.ZIP_STORED)
        self.assertFalse(os.path.exists(self.path("q.json")))

    def test_a_format_without_a_writer_names_those_there_are(self):
        with self.assertRaisesRegex(ValueError, "no writer for fbx files.*saves json, npz, obj, pickle$"):
            quad().save(self.path("q.fbx"))
        with self.assertRaisesRegex(ValueError, "no writer for stl files"):
            quad().save(self.path("q.stl"))
        with self.assertRaisesRegex(ValueError, "no writer for a file without an extension"):
            quad().save(self.path("q"))
        with self.assertRaisesRegex(ValueError, "MeshList has no writer for obj files.*saves json, npz, pickle$"):
            MeshList([quad()]).save(self.path("q.obj"))
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_pickle_names_one_format(self):
        for name in ("q.pkl", "q.pickle"):
            self.assertIsInstance(MeshData.load(quad().save(self.path(name))), MeshData)

    def test_a_refused_route_raises_with_its_reason(self):
        uv = UVData(points=np.zeros((4, 2)), indices=np.arange(4), counts=np.array([4]), name="map1")
        with self.assertRaisesRegex(ValueError, r"UVData has no writer for obj files \(.*2D.*saves json, npz, pickle$"):
            uv.save(self.path("uv.obj"))
        with self.assertRaisesRegex(TypeError, r"^UVData\.save_obj: "):
            uv.save_obj(self.path("uv.obj"))
        self.assertFalse(os.path.exists(self.path("uv.obj")))

    def test_a_clip_of_many_frames_names_the_files_that_keep_it(self):
        # save_fbx writes one pose; save() reaches it now, so the refusal
        # names the formats that keep the whole clip, not save()
        clip = ClipData(HierarchyData([TransformData("root", node_type="joint")]), frames=3)
        with self.assertRaisesRegex(ValueError, "or save it as npz, json or pkl to keep the whole clip$"):
            clip.save(self.path("clip.fbx"))
        self.assertFalse(os.path.exists(self.path("clip.fbx")))


class TestLoad(_Files):
    def test_mode_still_goes_by_position(self):
        path = quad().save(self.path("q.npz"))
        self.assertIsInstance(MeshData.load(path, "npz"), MeshData)
        self.assertIsInstance(hierarchy.load(HierarchyData().save(self.path("h.npz")), "npz"), HierarchyData)

    def test_an_own_file_takes_no_argument(self):
        path = quad().save(self.path("q.npz"))
        with self.assertRaisesRegex(TypeError, r"MeshData\.load_npz: .*'name'.*takes no argument"):
            MeshData.load(path, name="x")
        # in the module loaders too, whichever of cgmath's three files
        with self.assertRaisesRegex(TypeError, "^an npz file takes no argument but its name: got pose$"):
            hierarchy.load(HierarchyData().save(self.path("h.npz")), pose="bind")
        with self.assertRaisesRegex(TypeError, "^a json file takes no argument but its name: got pose$"):
            hierarchy.load(HierarchyData().save(self.path("h.json")), pose="bind")
        with self.assertRaisesRegex(TypeError, "^a pickle file takes no argument but its name: got bind_matrices$"):
            skin_weights.load(SkinList().save(self.path("s.pkl")), bind_matrices=True)

    def test_route_arguments_go_by_keyword_only(self):
        path = quad().save(self.path("q.obj"))
        with self.assertRaises(TypeError):
            MeshData.load(path, "obj", "quad")

    def test_arguments_go_to_the_reader(self):
        path = quad().save(self.path("q.obj"))
        self.assertEqual(Object.load(path, index=0, name="hero").name, "hero")
        self.assertIsInstance(Scene.load(path, name="set"), Scene)
        with self.assertRaisesRegex(TypeError, r"Scene\.load_obj: .*'nme'.*takes name$"):
            Scene.load(path, nme="set")

    # what Scene's refusal of an fbx file says to read one with today
    SCENE_FBX = "Scene reads FBX files from 4.0.0; today use Object.load_fbx(path, index=i) per mesh, or Scene.load_glb"

    def test_a_scene_says_what_reads_an_fbx_today(self):
        """Scene reads no fbx file before 4.0.0: its refusal, a direct
        call's TypeError and a load's ValueError alike, names what reads an
        fbx's meshes today; the refusal to write one is as it was"""
        path = self.path("x.fbx")
        open(path, "wb").close()
        with self.assertRaises(TypeError) as caught:
            Scene.load_fbx(path)
        self.assertEqual(str(caught.exception), f"Scene.load_fbx: {self.SCENE_FBX}")
        with self.assertRaises(ValueError) as caught:
            Scene.load(path)
        refusal = str(caught.exception).split("; it loads ")[0]
        self.assertEqual(refusal, f"Scene has no reader for fbx files ({self.SCENE_FBX}): {path}")
        with self.assertRaises(TypeError) as caught:
            Scene().save_fbx(self.path("y.fbx"))
        self.assertEqual(
            str(caught.exception),
            "Scene.save_fbx: HierarchyData's fbx writer writes joints only: a Scene's meshes would be lost",
        )

    def test_the_factory_refuses_an_asset_naming_the_readers(self):
        path = quad().save(self.path("q.obj"))
        for factory in (Data, DataList):
            with self.assertRaisesRegex(
                ValueError, r"is an obj file, which names none: read it with the class wanted \(MeshData, MeshList, Object, Scene"
            ):
                factory.load(path)

    def test_a_class_without_that_reader_names_those_there_are(self):
        path = quad().save(self.path("q.obj"))
        with self.assertRaisesRegex(ValueError, "no reader for obj files.*loads fbx, glb, json, npz, pickle$"):
            SkinList.load(path)

    def test_an_unknown_extension_is_refused_naming_the_readers(self):
        # a pickle under any other name is no longer read as one
        pickled = quad().save(self.path("q.pkl"))
        for name, kind in (("q.dat", "dat files"), ("q", "a file without an extension")):
            with open(pickled, "rb") as source, open(self.path(name), "wb") as copy:
                copy.write(source.read())
            self.assertEqual(get_file_type(self.path(name)), name.partition(".")[2])
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"MeshData has no reader for {kind}.*loads fbx, glb, json, npz, obj, pickle, usd$"):
                    MeshData.load(self.path(name))
                with self.assertRaisesRegex(ValueError, f"Data has no reader for {kind}.*loads json, npz, pickle$"):
                    Data.load(self.path(name))
                with self.assertRaisesRegex(ValueError, f"no reader for {kind}"):
                    hierarchy.load(self.path(name))
                # mode names its format whatever the name
                self.assertEqual(MeshData.load(self.path(name), mode="pkl"), quad())
        # cgmath's own files go by their header, whatever their name
        os.replace(quad().save(self.path("q.npz")), self.path("q.bin"))
        self.assertEqual(MeshData.load(self.path("q.bin")), quad())

    def test_a_glb_an_fbx_and_a_json_go_by_their_first_bytes_whatever_their_name(self):
        # under a name that is not their format's, so the extension cannot
        # decide: the header does
        cases = [
            ("a.bin", b"glTF\x02\x00\x00\x00", "glb"),
            ("a.dat", b"Kaydara FBX Binary  \x00", "fbx"),
            ("a.dat", b"; FBX 7.7.0 project file", "fbx"),
            ("a.dat", b'  \n{"a": 1}', "json"),
            ("a.dat", b" \n" * 20 + b'{"a": 1}', "json"),  # blanks past the first 18 bytes
            ("a.dat", b" " * 5000 + b'{"a": 1}', "json"),  # and past a further 4096
            ("a.dat", b'\xef\xbb\xbf{"a": 1}', "json"),  # utf-8's byte order mark first
            # utf-16, little and big endian, with its byte order mark and without
            ("a.dat", b"\xff\xfe" + ' {"a": 1}'.encode("utf-16-le"), "json"),
            ("a.dat", b"\xfe\xff" + ' {"a": 1}'.encode("utf-16-be"), "json"),
            ("a.dat", '{"a": 1}'.encode("utf-16-le"), "json"),
            ("a.dat", '[{"a": 1}]'.encode("utf-16-be"), "json"),
            # after blanks past its first 18 bytes, and past a further 4096
            ("a.dat", b"\xff\xfe" + (" " * 8 + '{"a": 1}').encode("utf-16-le"), "json"),
            ("a.dat", (" \r\n" * 1000 + '{"a": 1}').encode("utf-16-be"), "json"),
            # a no-break space is no blank, in utf-8 or in utf-16
            ("a.dat", b"\xc2\xa0{}", "dat"),
            ("a.dat", b"\xff\xfe\xa0\x00{\x00}\x00", "dat"),
            ("a", b"PK\x03\x04", "npz"),
        ]
        for name, data, expected in cases:
            with self.subTest(name=name, header=data[:8], size=len(data)):
                with open(self.path(name), "wb") as f:
                    f.write(data)
                self.assertEqual(get_file_type(self.path(name)), expected)

    @unittest.skipIf(trimesh is None or pygltflib is None, "trimesh and pygltflib are required")
    def test_a_glb_reads_under_any_name(self):
        # trimesh's readers and pygltflib's go by the extension; cgmath's go
        # by the first bytes under any name, .gltf too
        glb = write_glb(self.path("hero.glb"), ["body"])
        for name in ("hero.dat", "hero", "hero.gltf"):
            path = shutil.copyfile(glb, self.path(name))
            with self.subTest(name=name):
                self.assertEqual(get_file_type(path),             "glb")
                self.assertEqual(MeshData.load(path),             MeshData.load(glb))
                self.assertEqual(MeshData.load(path, mode="glb"), MeshData.load(glb))
                self.assertEqual(Object.load(path).mesh,          Object.load(glb).mesh)
                self.assertEqual(Object.load(path).skin,          Object.load(glb).skin)
                rig, named = HierarchyData.load(path), HierarchyData.load(glb)  # each load its own uuids
                self.assertEqual(rig.name, named.name)
                np.testing.assert_array_equal(rig.world_matrix, named.world_matrix)
                self.assertEqual(SkinList.load(path),          SkinList.load(glb))
                self.assertEqual(GlbData(path).gltf.to_json(), GlbData(glb).gltf.to_json())
                self.assertEqual(GlbData(path).mesh_list,      GlbData(glb).mesh_list)
        # a gltf, json beside its buffer, under another name, a .glb too:
        # mode= names it
        os.makedirs(self.path("text"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pygltflib's note that the buffer goes beside it
            gltf = write_glb(os.path.join(self.path("text"), "hero.gltf"), ["body"])
        for name in ("hero.txt", "hero.glb"):
            text = shutil.copyfile(gltf, os.path.join(self.path("text"), name))
            with self.subTest(gltf_as=name):
                self.assertEqual(MeshData.load(text, mode="glb"), MeshData.load(glb))
                self.assertEqual(MeshData.load_glb(text), MeshData.load(glb))
                self.assertEqual(Object.load(text, mode="glb").mesh, Object.load(glb).mesh)
                self.assertEqual(HierarchyData.load(text, mode="glb").name, named.name)
                self.assertEqual(HierarchyData.load_glb(text).name, named.name)
                self.assertEqual(SkinList.load(text, mode="glb"), SkinList.load(glb))
                self.assertEqual(GlbData(text).gltf.to_json(), GlbData(gltf).gltf.to_json())
                self.assertEqual(GlbData(text).mesh_list, GlbData(glb).mesh_list)
        # blanks before its json, however many, are read past, under its own
        # name and another
        with open(gltf, "rb") as f:
            blanked = b" \n" * 20 + f.read()
        for name in ("hero.gltf", "hero.glb"):
            path = os.path.join(self.path("text"), "blanks_" + name)
            with open(path, "wb") as f:
                f.write(blanked)
            with self.subTest(blanks_as=name):
                self.assertEqual(MeshData.load_glb(path),           MeshData.load(glb))
                self.assertEqual(HierarchyData.load_glb(path).name, named.name)
                self.assertEqual(GlbData(path).mesh_list,           GlbData(glb).mesh_list)

    @unittest.skipIf(trimesh is None or pygltflib is None, "trimesh and pygltflib are required")
    def test_a_file_that_starts_as_neither_a_glb_nor_a_gltf_is_no_glb(self):
        # a failed download saved as a web page, a Git LFS pointer, zeros:
        # the first bytes start neither, so the glb readers read a glb and
        # say it is none, under any name, as they did under a .glb name; not
        # json's error, nor trimesh's search for a model.gltf beside it
        neither = (
            ("web page", b"<!DOCTYPE html><html><body>404 Not Found</body></html>\n"),
            ("lfs pointer", b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"4d7a" * 16 + b"\nsize 12345\n"),
            ("zeros", bytes(64)),
        )
        trimesh_says   = (ValueError, "^incorrect header on GLB file$")
        pygltflib_says = (OSError, "^Unable to load binary gltf file. Header does not appear to be valid glb format")
        reads = (
            ("MeshData.load_glb", MeshData.load_glb, trimesh_says),
            ("mesh.load_glb", mesh_module.load_glb, trimesh_says),
            ("Object.load_glb", Object.load_glb, trimesh_says),
            ("HierarchyData.load_glb", HierarchyData.load_glb, pygltflib_says),
            ("SkinList.load_glb", SkinList.load_glb, pygltflib_says),
            ("GlbData", GlbData, pygltflib_says),
        )
        # load() takes a .glb or a .gltf name to them too
        routed = (
            ("MeshData.load", MeshData.load, trimesh_says),
            ("Object.load", Object.load, trimesh_says),
            ("HierarchyData.load", HierarchyData.load, pygltflib_says),
        )
        for name in ("hero.glb", "hero.gltf", "hero.dat", "hero"):
            for kind, data in neither:
                with open(self.path(name), "wb") as f:
                    f.write(data)
                for label, read, (error, says) in reads + (routed if name in ("hero.glb", "hero.gltf") else ()):
                    with self.subTest(name=name, file=kind, read=label):
                        with self.assertRaisesRegex(error, says):
                            read(self.path(name))

    @unittest.skipIf(trimesh is None or pygltflib is None, "trimesh and pygltflib are required")
    def test_a_gltf_trimesh_cannot_read_gets_json_s_error(self):
        # a gltf cut short or saved again with utf-8's byte order mark, and
        # one in utf-16 or an older cgmath's json in the locale's codepage
        # where trimesh has no charset_normalizer to read them with: json's
        # own error through every trimesh reader, under any name, not
        # trimesh's search for a model.gltf in it, as in a zip; pygltflib,
        # which GlbData reads with first, raises its own json error
        os.makedirs(self.path("text"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pygltflib's note that the buffer goes beside it
            gltf = write_glb(os.path.join(self.path("text"), "hero.gltf"), ["body"])
        with open(gltf, "rb") as f:
            text = f.read()
        older = quad("qu\u00e9").to_json().encode("cp1252")
        broken = (
            ("cut short", text[: len(text) // 2], JSONDecodeError, ""),
            ("utf-8 mark", b"\xef\xbb\xbf" + text, JSONDecodeError, r"^Unexpected UTF-8 BOM"),
            ("utf-16", b"\xff\xfe" + text.decode("utf-8").encode("utf-16-le"), UnicodeDecodeError, "0xff in position 0"),
            ("cp1252", older, UnicodeDecodeError, "0xe9"),
        )
        reads = (
            ("mesh.load_glb", mesh_module.load_glb),
            ("MeshData.load_glb", MeshData.load_glb),
            ("MeshList.load_glb", MeshList.load_glb),
            ("Object.load_glb", Object.load_glb),
            ("mode=glb", lambda path: MeshData.load(path, mode="glb")),
        )
        with mock.patch.dict(sys.modules, {"charset_normalizer": None}):
            for name in ("hero.gltf", "hero.glb", "hero.dat"):
                path = os.path.join(self.path("text"), "broken_" + name)
                for kind, data, error, says in broken:
                    with open(path, "wb") as f:
                        f.write(data)
                    for label, read in reads:
                        with self.subTest(name=name, file=kind, read=label):
                            with self.assertRaisesRegex(error, says):
                                read(path)
                    with self.subTest(name=name, file=kind, read="GlbData"):
                        with self.assertRaises(ValueError):
                            GlbData(path)
        # with charset_normalizer, trimesh reads the gltf in utf-16 itself
        if charset_normalizer is not None:
            path = os.path.join(self.path("text"), "utf16.dat")
            with open(path, "wb") as f:
                f.write(broken[2][1])
            self.assertEqual(MeshData.load_glb(path), MeshData.load_glb(gltf))
        # a gltf that json reads, its buffer missing, keeps trimesh's error
        # naming the buffer
        os.makedirs(self.path("alone"))
        for name in ("hero.gltf", "hero.dat"):
            path = shutil.copyfile(gltf, os.path.join(self.path("alone"), name))
            for label, read in reads[:2] + (("GlbData", GlbData),):
                with self.subTest(alone=name, read=label):
                    with self.assertRaisesRegex(FileNotFoundError, r"hero\.bin"):
                        read(path)

    def test_a_file_that_does_not_start_as_its_name_says_is_refused_naming_mode(self):
        # a pickle under a .json or an .npz name goes to that reader, which
        # says it is no such file: not an older cgmath's file, nor zipfile's
        # error; mode="pkl" reads it
        pickled = quad().save(self.path("q.pkl"))
        for ext, article in (("json", "a"), ("npz", "an")):
            path = shutil.copyfile(pickled, self.path(f"q.{ext}"))
            refused = (
                rf"q\.{ext} does not start as {article} {ext} file does; "
                r'load\(\) reads it with mode= naming its type, mode="pkl" for a pickle$'
            )
            loads = (
                ("MeshData.load", MeshData.load),
                ("Data.load", Data.load),
                ("hierarchy.load", hierarchy.load),
                (f"MeshData.load_{ext}", getattr(MeshData, f"load_{ext}")),
            )
            for label, load in loads:
                with self.subTest(ext=ext, load=label):
                    with self.assertRaisesRegex(ValueError, refused) as caught:
                        load(path)
                    self.assertNotIsInstance(caught.exception, LegacyFileError)
            self.assertEqual(MeshData.load(path, mode="pkl"), quad())
        # an npz read as json, a json read as npz: neither starts as the
        # other, so neither is an older cgmath's file nor a broken zip
        swapped = (
            ("json", shutil.copyfile(quad().save(self.path("z.npz")), self.path("z.json")), MeshData.load_json),
            ("npz", shutil.copyfile(quad().save(self.path("j.json")), self.path("j.npz")), MeshData.load_npz),
        )
        for ext, path, load in swapped:
            reads = ((f"MeshData.load_{ext}", load), (f"mode={ext}", lambda path: MeshData.load(path, mode=ext)))
            for label, read in reads:
                with self.subTest(read_as=ext, load=label):
                    with self.assertRaisesRegex(ValueError, rf"does not start as an? {ext} file does; load\(\) reads it with mode=") as caught:
                        read(path)
                    self.assertNotIsInstance(caught.exception, LegacyFileError)
        # a pickle save() writes under a .json name, as mode= allows
        path = quad().save(self.path("x.json"), mode="pkl")
        with self.assertRaisesRegex(ValueError, r"x\.json does not start as a json file does"):
            MeshData.load(path)
        # a text file too: an obj under a .json name reads with mode="obj"
        path = shutil.copyfile(quad().save(self.path("o.obj")), self.path("o.json"))
        with self.assertRaisesRegex(ValueError, r"o\.json does not start as a json file does"):
            MeshData.load(path)
        self.assertEqual(MeshData.load(path, mode="obj"), quad())
        # a json or an npz that starts as one but breaks off is json's or
        # zipfile's own error
        with open(self.path("cut.json"), "w") as f:
            f.write('{"__class__": ')
        with self.assertRaisesRegex(ValueError, "^Expecting value: line 1"):
            MeshData.load(self.path("cut.json"))
        with open(quad().save(self.path("whole.npz")), "rb") as f:
            start = f.read(64)
        with open(self.path("cut.npz"), "wb") as f:
            f.write(start)
        with self.assertRaises(zipfile.BadZipFile):
            MeshData.load(self.path("cut.npz"))
        # a json an editor saved again with utf-8's byte order mark starts as
        # one: json's own error names the mark, where mode= would not help
        with open(quad().save(self.path("plain.json")), "rb") as f:
            text = f.read()
        with open(self.path("bom.json"), "wb") as f:
            f.write(b"\xef\xbb\xbf" + text)
        reads = (
            ("MeshData.load", MeshData.load),
            ("MeshData.load_json", MeshData.load_json),
            ("mode=json", lambda path: MeshData.load(path, mode="json")),
            ("Data.load", Data.load),
            ("hierarchy.load", hierarchy.load),
        )
        for label, read in reads:
            with self.subTest(bom=label):
                with self.assertRaisesRegex(JSONDecodeError, r"^Unexpected UTF-8 BOM \(decode using utf-8-sig\)"):
                    read(self.path("bom.json"))
        # a json Windows PowerShell 5.1 passed through Out-File or > is
        # utf-16, little endian after its byte order mark, its lines CRLF:
        # an error of its own says so, where mode= would loop back to it;
        # after blanks too, however many: 8 spaces, past its first 18 bytes,
        # 4 blank lines, and 3000 characters of blank lines, past a further
        # 4096 bytes
        lines = text.decode("utf-8").replace("\n", "\r\n") + "\r\n"
        utf16 = []
        for blank_kind, blanks in (
            ("none", ""),
            ("8 spaces", " " * 8),
            ("4 lines", "\r\n" * 4),
            ("1500 lines", "\r\n" * 1500),
        ):
            utf16 += (
                (("le, mark", blank_kind), b"\xff\xfe" + (blanks + lines).encode("utf-16-le")),  # Out-File, >
                (("be, mark", blank_kind), b"\xfe\xff" + (blanks + lines).encode("utf-16-be")),  # Out-File -Encoding BigEndianUnicode
                (("le", blank_kind), (blanks + lines).encode("utf-16-le")),
                (("be", blank_kind), (blanks + lines).encode("utf-16-be")),
            )
        for (kind, blank_kind), data in utf16:
            with open(self.path("utf16.json"), "wb") as f:
                f.write(data)
            for label, read in reads:
                with self.subTest(utf16=kind, blanks=blank_kind, load=label):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"utf16\.json is a json file in utf-16, as Windows PowerShell 5\.1's Out-File "
                        r"and > write one; cgmath reads a json file in utf-8, without a byte order mark",
                    ) as caught:
                        read(self.path("utf16.json"))
                    self.assertIs(type(caught.exception), ValueError)

    def test_a_reader_that_builds_cls_serves_a_subclass(self):
        self.assertIsInstance(_Quad.load(quad().save(self.path("q.obj"))), _Quad)

    def test_a_reader_giving_a_class_derived_from_cls_is_refused(self):
        # the class itself, exactly: not one deriving from it
        with open(self.path("t.stl"), "w") as f:
            f.write("0 0 0\n")
        with self.assertRaisesRegex(TypeError, r"^_Loose\.load_stl gave a _Looser: "):
            _Loose.load(self.path("t.stl"))

    def test_the_module_loaders_pass_arguments(self):
        path = quad().save(self.path("q.obj"))
        with self.assertRaisesRegex(TypeError, r"mesh\.load_obj: .*'name'"):
            mesh_module.load(path, name="x")
        self.assertEqual(len(mesh_module.load(path)), 1)
        with self.assertRaisesRegex(ValueError, "mesh.load reads fbx, glb, obj and usd files, not stl"):
            mesh_module.load(path, mode="stl")
        # skin_weights.load hands its arguments on too: a cgmath file takes none
        skins = SkinList().save(self.path("s.npz"))
        self.assertIsInstance(skin_weights.load(skins), SkinList)
        with self.assertRaisesRegex(TypeError, "an npz file takes no argument but its name: got bind_matrices"):
            skin_weights.load(skins, bind_matrices=True)

    @unittest.skipIf(trimesh is None, "trimesh is required")
    def test_glb_arguments_and_the_subclass_rule(self):
        glb = self.path("q.glb")
        trimesh.Trimesh(vertices=quad().points, faces=[[0, 1, 2], [0, 2, 3]], process=False).export(glb)
        np.testing.assert_allclose(MeshData.load(glb, scale_factor=1.0).points, quad().points)
        with self.assertRaisesRegex(TypeError, r"load_glb: .*'scale'.*takes scale_factor, name"):
            MeshData.load(glb, scale=1.0)
        # MeshData.load_glb builds a MeshData, never a _Quad
        with self.assertRaisesRegex(TypeError, r"_Quad\.load_glb gave a MeshData"):
            _Quad.load(glb)
        self.assertEqual(len(MeshList.load(self.path("q.glb"), mode="gltf", scale_factor=1.0)), 1)

    @unittest.skipIf(Usd is None, "pxr is required")
    def test_a_mesh_reads_a_usd_file(self):
        for ext in ("usda", "usdc"):
            path  = self.path(f"q.{ext}")
            stage = Usd.Stage.CreateNew(path)
            prim  = UsdGeom.Mesh.Define(stage, "/quad").GetPrim()
            quad().to_prim(prim)
            UVData(points=np.zeros((4, 2)), indices=np.arange(4), counts=np.array([4]), name="map1").to_prim(0, prim)
            stage.GetRootLayer().Save()
            self.assertEqual(MeshData.load(path).name, "quad")
            self.assertEqual(MeshData.load(path, name="quad").points.shape, (4, 3))
            self.assertEqual(UVData.load(path, channel=0).points.shape, (4, 2))


class TestUsdReaders(_Files):
    """MeshData.load_usd and UVData.load_usd pick one mesh and one UV channel
    of what the module's usd reader gives, a stand-in here: no pxr needed"""

    def setUp(self):
        super().setUp()
        pairs = [
            (quad("a"), UVList([quad_uv("map1"), quad_uv("st", 0.5)])),
            (quad("bb"), UVList([quad_uv("map1", 0.25)])),
        ]
        reader = mock.patch.object(mesh_module, "load_usd", lambda filename: pairs)
        reader.start()
        self.addCleanup(reader.stop)

    def test_the_first_mesh_or_the_one_named(self):
        path = self.path("x.usda")
        self.assertEqual(MeshData.load(path).name, "a")
        self.assertEqual(MeshData.load(path, name="bb").name, "bb")
        with self.assertRaisesRegex(ValueError, r"^Mesh 'zz' not found; available: \['a', 'bb'\]$"):
            MeshData.load(path, name="zz")

    def test_one_uv_channel(self):
        path = self.path("x.usdc")
        self.assertEqual(UVData.load(path).name, "map1")
        self.assertEqual(UVData.load(path, channel=1).name, "st")
        self.assertEqual(UVData.load(path, name="bb").points.tolist(), [[0.25, 0.25]] * 4)
        with self.assertRaisesRegex(TypeError, r"UVData\.load_usd: .*'chanel'.*takes name, channel$"):
            UVData.load(path, chanel=1)

    def test_every_usd_name_routes_there(self):
        # mode names usd by any of its names, an extension by any case
        for mode in ("usd", "usda", "usdc", "usdz", "USDZ"):
            with self.subTest(mode=mode):
                self.assertEqual(MeshData.load(self.path("x.dat"), mode=mode).name, "a")
        self.assertEqual(UVData.load(self.path("x.USDZ"), channel=1).name, "st")


@unittest.skipIf(fbx is None, "the Autodesk FBX SDK is required")
class TestFbx(_Files):
    def rig(self):
        root = TransformData("root", node_type="joint", uuid=str(uuid.uuid4()).upper())
        hip = TransformData("hip", node_type="joint", uuid=str(uuid.uuid4()).upper(),
                            parent_uuid=root.uuid, translate=[0.0, 10.0, 0.0])
        return HierarchyData([root, hip])

    def test_a_rig_saves_and_loads_through_the_fbx_routes(self):
        path = self.rig().save(self.path("rig.fbx"), as_ascii=True)
        with open(path, "rb") as f:
            head = f.read(64)
        # an ASCII fbx starts with its comment line, "; FBX <version> project file"
        self.assertNotIn(b"Kaydara FBX Binary", head)
        self.assertTrue(head.lstrip().startswith(b"; FBX "), head)
        rig = HierarchyData.load(path, pose="node", scale_factor=1.0)
        self.assertEqual(rig.name, ["root", "hip"])
        clip = ClipData.load(path, scale_factor=2.0)
        self.assertIsInstance(clip, ClipData)
        np.testing.assert_allclose(np.asarray(clip.frames.translate)[0, 1], [0.0, 20.0, 0.0])
        self.assertEqual(hierarchy.load(path, pose="node").name, ["root", "hip"])
        with self.assertRaisesRegex(TypeError, r"HierarchyData\.load_fbx: .*'pse'"):
            HierarchyData.load(path, pse="bind")
        with self.assertRaisesRegex(TypeError, r"SkinList\.load_fbx: .*'pse'.*takes no argument"):
            skin_weights.load(path, pse="bind")
        with self.assertRaisesRegex(TypeError, r"HierarchyData\.save_fbx: .*'ascii'"):
            self.rig().save(self.path("again.fbx"), ascii=True)
        self.assertFalse(os.path.exists(self.path("again.fbx")))

    def test_a_scene_refuses_the_fbx_routes(self):
        path  = self.rig().save(self.path("rig.fbx"))
        today = re.escape(TestLoad.SCENE_FBX)
        with self.assertRaisesRegex(ValueError, rf"^Scene has no reader for fbx files \({today}\): "):
            Scene.load(path)
        with self.assertRaisesRegex(ValueError, "Scene has no writer for fbx files .*meshes"):
            Scene().save(self.path("scene.fbx"))
        with self.assertRaisesRegex(TypeError, rf"^Scene\.load_fbx: {today}$"):
            Scene.load_fbx(path)
        self.assertFalse(os.path.exists(self.path("scene.fbx")))

    def test_the_factory_refuses_an_fbx(self):
        path = self.rig().save(self.path("rig.fbx"))
        with self.assertRaisesRegex(ValueError, r"is an fbx file, which names none.*ClipData, HierarchyData"):
            Data.load(path)


class TestMeshObj(_Files):
    """MeshData.save_obj writes the mesh's normals, every face corner naming its own"""

    def test_a_mesh_without_normals_writes_as_before(self):
        path = quad().save_obj(self.path("q.obj"))
        with open(path, "rb") as f:
            self.assertEqual(
                f.read(),
                b"# OBJ file\ng quad\ns 1\nv 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3 4\n\n",
            )

    def round_trip(self, mesh):
        path = mesh.save(self.path("mesh.obj"))
        back = MeshData.load(path)
        np.testing.assert_array_equal(back.points,          mesh.points)
        np.testing.assert_array_equal(back.indices,         mesh.indices)
        np.testing.assert_array_equal(back.counts,          mesh.counts)
        np.testing.assert_array_equal(corner_normals(back), corner_normals(mesh))  # every digit
        (pair,) = mesh_module.load_obj(path)
        np.testing.assert_array_equal(corner_normals(pair[0]), corner_normals(mesh))
        return back

    def test_face_varying_normals_come_back_as_they_went(self):
        mesh = cube()
        mesh.set_normals(hard_edge_angle=30.0)
        self.assertEqual((len(mesh.normals), len(mesh.normal_indices)), (24, 24))
        self.assertEqual(self.round_trip(mesh), mesh)

    def test_normals_one_a_vertex(self):
        mesh = cube()
        mesh.set_normals()
        self.assertIsNone(mesh.normal_indices)
        back = self.round_trip(mesh)
        # read back face-varying, each corner naming its vertex's normal
        np.testing.assert_array_equal(back.normals, mesh.normals)
        np.testing.assert_array_equal(back.normal_indices, mesh.indices)

    def test_normals_one_a_vertex_where_the_vertices_are_as_many_as_the_corners(self):
        # two triangles sharing no vertex, the second naming its own out of
        # order: one normal a vertex and one a corner both fit, and a mesh's
        # normals read as one a vertex, as resample_uv reads them
        points  = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=float)
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=float)
        indices = np.array([0, 1, 2, 5, 3, 4])
        mesh    = MeshData(points=points, indices=indices, counts=np.array([3, 3]), normals=normals)
        back    = self.round_trip(mesh)
        np.testing.assert_array_equal(np.asarray(back.normals)[back.normal_indices], normals[mesh.indices])

    def test_normals_one_a_corner_without_indices(self):
        mesh         = cube()
        mesh.normals = np.repeat(np.eye(3), 8, axis=0)
        self.round_trip(mesh)

    def test_normals_that_fit_nothing_raise_before_writing(self):
        mesh         = cube()
        mesh.normals = np.ones((5, 3))
        with self.assertRaisesRegex(ValueError, "5 normals, which fit neither its 8 vertices nor its 24 face corners"):
            mesh.save(self.path("bad.obj"))
        for count in (23, 25):
            mesh.normal_indices = np.zeros(count, dtype=np.int32)
            with self.assertRaisesRegex(ValueError, f"{count} normal_indices for its 24 face corners"):
                mesh.save(self.path("bad.obj"))
        mesh.normal_indices = np.full(24, 5, dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "name normals it does not hold"):
            mesh.save(self.path("bad.obj"))
        # a negative one too, which would write the corner's normal as 0, no obj index
        mesh.normal_indices = np.r_[np.zeros(23), -1].astype(np.int32)
        with self.assertRaisesRegex(ValueError, "name normals it does not hold"):
            mesh.save(self.path("bad.obj"))
        self.assertFalse(os.path.exists(self.path("bad.obj")))

    # the names are compared as ascii() gives them, so a failure prints on
    # any console, a cp1252 one too

    def test_a_name_past_ascii_comes_back(self):
        # utf-8 whatever the locale's codepage (cp1252 on Windows): an e acute,
        # and a Cyrillic IO, whose utf-8 bytes cp1252 cannot read
        for name in (f"corps{chr(0xE9)}Shape", f"{chr(0x401)}Shape"):
            with self.subTest(name=ascii(name)):
                group = f"\ng {name[:-len('Shape')]}\n".encode("utf-8")
                path  = quad(name).save(self.path("q.obj"))
                with open(path, "rb") as f:
                    self.assertIn(group, f.read())
                self.assertEqual(ascii(MeshData.load(path).name), ascii(name))
                self.assertEqual(ascii(mesh_module.load_obj(path)[0][0].name), ascii(name))

                path = Object(name=name[:-len("Shape")], mesh=quad()).save(self.path("o.obj"))
                self.assertEqual(ascii(Object.load(path).name), ascii(name))

                # the module's writer of (MeshData, UVList) pairs too
                path = self.path("pairs.obj")
                mesh_module.save_obj(path, [(quad(name), UVList([quad_uv()]))])
                with open(path, "rb") as f:
                    self.assertIn(group, f.read().replace(b"\r\n", b"\n"))
                self.assertEqual(ascii(mesh_module.load_obj(path)[0][0].name), ascii(name))

    def test_a_file_in_a_codepage_still_reads(self):
        # cgmath wrote its obj files in the locale's codepage before: cp1252
        # on Windows, where an oe ligature is the one byte 0x9c and an e acute
        # 0xe9, neither of them utf-8. A byte the codepage has no character
        # for reads as latin-1 reads it, so the file still loads
        path = self.path("old.obj")
        text = b"# OBJ file\r\ng c\x9cur\xe9\r\ns 1\r\nv 0.0 0.0 0.0\r\nv 1.0 0.0 0.0\r\nv 1.0 1.0 0.0\r\nf 1 2 3\r\n\r\n"
        with open(path, "wb") as f:
            f.write(text)
        with mock.patch.object(locale, "getpreferredencoding", return_value="cp1252"):
            name = f"c{chr(0x153)}ur{chr(0xE9)}Shape"
            self.assertEqual(ascii(MeshData.load(path).name), ascii(name))
            self.assertEqual(ascii(mesh_module.load_obj(path)[0][0].name), ascii(name))
            with open(path, "wb") as f:
                f.write(text.replace(b"\x9c", b"\x81"))
            self.assertEqual(ascii(MeshData.load(path).name), ascii(f"c{chr(0x81)}ur{chr(0xE9)}Shape"))
            self.assertEqual(mesh_module.load_obj(path)[0][0].points.shape, (3, 3))


class TestObjectObj(_Files):
    """Object.save_obj writes its mesh where the Object puts it, with its UVs and normals"""

    def hero(self, **transform):
        """a cube turned off the axes, its normals of 24 lengths, under a uv"""
        mesh = cube()
        turn = np.array([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, -0.8, 0.6]]
        )
        mesh.points = mesh.points @ turn
        mesh.set_normals(hard_edge_angle=30.0)
        mesh.normals = mesh.normals * np.linspace(0.5, 2.0, 24)[:, None]
        return Object(name="hero", mesh=mesh, uv=cube_uv(), **transform)

    def placed(self):
        """an Object under a parent, turned and scaled unevenly: its points, corner normals and the Object"""
        scene = Scene("s")
        group = Object(name="group", translate=(3.0, -1.0, 2.0), rotate=(10.0, 20.0, 30.0))
        hero  = self.hero(translate=(0.5, 1.5, -2.0), rotate=(30.0, 45.0, -60.0), scale=(2.0, 0.5, 1.5))
        scene.extend([group, hero])
        hero.set_parent(group, world_space=False)
        world = np.asarray(hero.world_matrix)
        np.testing.assert_allclose(world, hero.matrix @ group.matrix)  # the parent's transform applies
        return hero, world

    def test_points_uvs_and_normals_in_the_objects_world(self):
        hero, world = self.placed()
        path = hero.save(self.path("hero.obj"))
        back = Object.load(path)
        mesh = hero.mesh
        np.testing.assert_allclose(back.mesh.points, mesh.points @ world[:3, :3] + world[3, :3], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(back.mesh.counts, mesh.counts)
        np.testing.assert_array_equal(back.uv.points,   hero.uv.points)
        np.testing.assert_array_equal(back.uv.indices,  hero.uv.indices)

        # each normal turned by the inverse transpose, keeping its length,
        # so it stays at right angles to the faces the points now make
        normals = corner_normals(back.mesh)
        before  = corner_normals(mesh)
        turned  = before @ np.linalg.inv(world[:3, :3]).T
        lengths = np.linalg.norm(before, axis=1, keepdims=True) / np.linalg.norm(turned, axis=1, keepdims=True)
        np.testing.assert_allclose(normals, turned * lengths, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), np.linalg.norm(before, axis=1), rtol=1e-15)
        faces = back.mesh.points[back.mesh.indices].reshape(6, 4, 3)
        edges = np.concatenate([faces[:, 1] - faces[:, 0], faces[:, 3] - faces[:, 0]])
        self.assertLess(np.abs(np.einsum("ij,ij->i", edges, np.concatenate([normals[::4], normals[::4]]))).max(), 1e-14)

        # the Object read back sits at the origin, its transform in the points
        np.testing.assert_array_equal(back.world_matrix, np.eye(4))
        self.assertEqual(back.name, "heroShape")

    def test_every_obj_reader_reads_it(self):
        hero, _ = self.placed()
        path = hero.save_obj(self.path("hero.obj"))
        self.assertEqual(path, self.path("hero.obj"))
        loaded = Object.load_obj(path)
        for mesh in (MeshData.load(path), Scene.load(path).objects[0].mesh, mesh_module.load(path)[0][0]):
            np.testing.assert_array_equal(mesh.points, loaded.mesh.points)
            np.testing.assert_array_equal(corner_normals(mesh), corner_normals(loaded.mesh))
        self.assertEqual(UVData.load(path), loaded.uv)
        with open(path, "rb") as f:
            text = f.read()
        self.assertNotIn(b"\r", text)
        self.assertEqual(text.count(b"\nf "), 6)
        self.assertIn(b"\ng hero\n", text)

    def test_an_object_at_the_origin_writes_its_mesh_as_it_is(self):
        hero = self.hero()
        back = Object.load(hero.save(self.path("hero.obj")))
        np.testing.assert_array_equal(back.mesh.points,         hero.mesh.points)
        np.testing.assert_array_equal(back.mesh.normals,        hero.mesh.normals)
        np.testing.assert_array_equal(back.mesh.normal_indices, hero.mesh.normal_indices)
        self.assertEqual(back.uv, hero.uv)

        # no uv, no normals: points and faces
        plain = Object(name="plain", mesh=cube())
        back  = Object.load(plain.save(self.path("plain.obj")))
        self.assertIsNone(back.uv)
        self.assertIsNone(back.mesh.normals)
        np.testing.assert_array_equal(back.mesh.points, plain.mesh.points)

    def test_a_uv_without_normals(self):
        # each corner names its point and its UV, f v/vt
        uvonly = Object(name="uvonly", mesh=cube(), uv=cube_uv())
        path   = uvonly.save(self.path("uvonly.obj"))
        with open(path, "rb") as f:
            faces = [line for line in f.read().split(b"\n") if line.startswith(b"f ")]
        self.assertEqual(faces[0], b"f 1/24 4/23 3/22 2/21")
        back = Object.load(path)
        self.assertIsNone(back.mesh.normals)
        np.testing.assert_array_equal(back.mesh.points, uvonly.mesh.points)
        np.testing.assert_array_equal(back.mesh.indices, uvonly.mesh.indices)
        self.assertEqual(back.uv, cube_uv())
        np.testing.assert_array_equal(back.uv.points[back.uv.indices], cube_uv().points[cube_uv().indices])

    def test_a_zero_normal_stays_zero(self):
        # a normal of no length has no direction to turn: it is written as it is, never nan
        normals      = np.repeat(np.eye(3), 8, axis=0)
        normals[0]   = 0.0
        mesh         = cube()
        mesh.normals = normals
        dim          = Object(name="dim", mesh=mesh, rotate=(30.0, 0.0, 0.0), scale=(2.0, 1.0, 1.0))
        path         = dim.save(self.path("dim.obj"))
        with open(path, "rb") as f:
            self.assertNotIn(b"nan", f.read())
        back = corner_normals(Object.load(path).mesh)
        np.testing.assert_array_equal(back[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(back[1:], axis=1), 1.0, rtol=1e-15)

    def test_what_it_cannot_write_raises_before_writing(self):
        with self.assertRaisesRegex(ValueError, "Object 'empty' has no mesh to write"):
            Object(name="empty").save(self.path("x.obj"))
        triangles = UVData(points=np.zeros((3, 2)), indices=np.arange(3), counts=np.array([3]))
        with self.assertRaisesRegex(ValueError, r"Object 'hero': its uv's 1 faces are not its mesh's 6"):
            Object(name="hero", mesh=cube(), uv=triangles).save(self.path("x.obj"))
        with self.assertRaisesRegex(ValueError, "world matrix cannot be inverted"):
            self.hero(scale=(1.0, 0.0, 1.0)).save(self.path("x.obj"))
        self.assertFalse(os.path.exists(self.path("x.obj")))
        # without normals a flat world matrix has nothing to turn
        flat = Object(name="flat", mesh=cube(), scale=(1.0, 0.0, 1.0)).save(self.path("flat.obj"))
        self.assertEqual(MeshData.load(flat).points[:, 1].tolist(), [0.0] * 8)

    def test_maya_imports_it(self):
        """Maya's OBJ import gives the points, UVs and normals written, in
        float, the normals made unit, and keeps them as locked user
        normals. Needs mayapy with the objExport plug-in"""
        try:
            import maya.cmds as cmds
            from maya.api import OpenMaya

            cmds.loadPlugin("objExport", quiet=True)
        except Exception:  # not running in Maya, or no objExport plug-in
            self.skipTest("needs Maya and its objExport plug-in")
        self.addCleanup(cmds.file, new=True, force=True)

        hero, world = self.placed()
        path = hero.save(self.path("hero.obj"))
        cmds.file(new=True, force=True)
        nodes  = cmds.file(path, i=True, type="OBJ", returnNewNodes=True, ignoreVersion=True, options="mo=1")
        shapes = cmds.ls(nodes, type="mesh", long=True)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(cmds.listRelatives(shapes[0], parent=True)[0], "hero")
        selection = OpenMaya.MSelectionList()
        selection.add(shapes[0])
        fn = OpenMaya.MFnMesh(selection.getDagPath(0))

        points = np.array([[p.x, p.y, p.z] for p in fn.getPoints(OpenMaya.MSpace.kWorld)])
        np.testing.assert_allclose(points, hero.mesh.points @ world[:3, :3] + world[3, :3], rtol=0, atol=1e-6)
        counts, indices = fn.getVertices()
        np.testing.assert_array_equal(np.array(counts), hero.mesh.counts)
        np.testing.assert_array_equal(np.array(indices), hero.mesh.indices)

        # Maya numbers the UVs as the faces first name them: each corner's UV
        us, vs = fn.getUVs()
        corners = [(face, k) for face, count in enumerate(counts) for k in range(count)]
        uv_ids  = [fn.getPolygonUVid(face, k) for face, k in corners]
        self.assertEqual(fn.numUVs(), len(hero.uv.points))
        np.testing.assert_allclose(
            np.column_stack([us, vs])[uv_ids], hero.uv.points[hero.uv.indices], rtol=0, atol=1e-7
        )

        normals = [fn.getFaceVertexNormal(face, fn.getPolygonVertices(face)[k], OpenMaya.MSpace.kWorld) for face, k in corners]
        turned  = corner_normals(hero.mesh) @ np.linalg.inv(world[:3, :3]).T
        np.testing.assert_allclose(
            np.array([[n.x, n.y, n.z] for n in normals]),
            turned / np.linalg.norm(turned, axis=1, keepdims=True),
            rtol = 0,
            atol = 1e-6,
        )
        self.assertEqual((fn.numNormals, sum(fn.isNormalLocked(i) for i in range(fn.numNormals))), (24, 24))


if __name__ == "__main__":
    unittest.main()
