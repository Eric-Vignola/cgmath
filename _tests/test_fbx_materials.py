"""Materials and textures through load_fbx and save_fbx: Maya's conventions, measured.

Maya's exporter (2025) writes a lambert as an FbxSurfaceLambert and a phong
(a blinn too) as an FbxSurfacePhong: the colour as DiffuseColor and the
diffuse as DiffuseFactor, the ambient colour, incandescence and transparency
as AmbientColor, EmissiveColor and TransparentColor with factors of 1, the
specular colour as SpecularColor with a factor of 1, the cosine power as
ShininessExponent (a blinn's eccentricity converted), the reflectivity as
ReflectionFactor with a ReflectionColor of 0; a standardSurface as a bare
FbxSurfaceMaterial with no shading model; a file texture on the colour as an
FbxFileTexture feeding DiffuseColor, the colour itself white, naming the UV
set and the wrap modes; two shading groups on a mesh as a by-polygon material
element, one as all-same; Embed Media as the file's bytes, extracted on read.
The fork reads and writes the same. The SDK tests need the Autodesk SDK; the
Maya ones read what Maya 2025 did, recorded once (maya_fixtures), and need no
Maya.
"""

import os
import re
import shutil
import unittest
import warnings
from unittest import mock

import numpy as np
from cgmath._tests.test_skeleton_fbx import _Files, _Recorded, _edited, _named, bundle
from cgmath.formats import _load as _load_module, load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.formats._write import _enum
from cgmath.geometry import CameraData, CameraList, LightData, LightList, MaterialData, MaterialList, MeshData, MeshList, UVData, UVList
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, generate_uuid
from cgmath.hierarchy.hierarchy import _is_uuid
from cgmath.render import Scene

FBX = load_sdk()

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is a dependency
    Image = None


def gradient() -> np.ndarray:
    """an 8 by 4 RGB gradient: red along u, green along v, blue 128"""
    pixels         = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[..., 0] = np.linspace(0, 255, 8, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.linspace(0, 255, 4, dtype=np.uint8)[:, None]
    pixels[..., 2] = 128
    return pixels


def plane(parent_uuid=None) -> MeshData:
    """a 2 by 2 plane of four quads in xz, 2 units wide"""
    points = [(x, 0.0, z) for z in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
    return MeshData(
        indices=[0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], counts=[4, 4, 4, 4], points=points, name="planeShape",
        parent_uuid=parent_uuid,
    )


def plane_uvs(mesh) -> UVData:
    points = np.asarray(mesh.points)
    return UVData(
        name="map1", points=np.stack([(points[:, 0] + 1.0) / 2.0, (points[:, 2] + 1.0) / 2.0], axis=1), counts=mesh.counts,
        indices=mesh.indices, mesh_uuid=mesh.uuid,
    )


def box(parent_uuid=None) -> MeshData:
    """a cube of six quads"""
    points = [(x, y, z) for z in (-1.0, 1.0) for y in (-1.0, 1.0) for x in (-1.0, 1.0)]
    faces  = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return MeshData(indices=[i for face in faces for i in face], counts=[4] * 6, points=points, name="boxShape", parent_uuid=parent_uuid)


def rig() -> HierarchyData:
    nodes = HierarchyData()
    nodes.append(TransformData("plane"))
    nodes.append(TransformData("box", translate=(3.0, 0.0, 0.0)))
    return nodes


def meshes(nodes) -> MeshList:
    return MeshList([plane(nodes["plane"].uuid), box(nodes["box"].uuid)])


def skin_lambert(mesh) -> MaterialData:
    """a textured lambert on the plane's first two faces, every attribute
    off its default"""
    return MaterialData(
        name="skinLambert", mesh_uuid=mesh.uuid, faces=[0, 1], color=[1.0, 1.0, 1.0], transparency=[0.1, 0.2, 0.3],
        ambient_color=[0.05, 0.1, 0.15], incandescence=[0.2, 0.0, 0.1], diffuse=0.6, texture=gradient(), texture_file="grad.png",
        uv_set="map1", wrap_u="repeat", wrap_v="clamp",
    )


def shiny_phong(mesh) -> MaterialData:
    return MaterialData(
        name="shinyPhong", mesh_uuid=mesh.uuid, faces=[2, 3], kind="phong", color=[0.9, 0.2, 0.1], specular_color=[0.3, 0.4, 0.5],
        cosine_power=35.0, reflectivity=0.25,
    )


def box_phong(mesh) -> MaterialData:
    return MaterialData(name="boxPhong", mesh_uuid=mesh.uuid, kind="phong", color=[0.1, 0.8, 0.2], reflectivity=0.1)


def materials(held) -> MaterialList:
    return MaterialList([skin_lambert(held[0]), shiny_phong(held[0]), box_phong(held[1])])


def same_materials(test, got, want, got_meshes, want_meshes) -> None:
    """``got``, a MaterialList read back, holds ``want``'s materials in
    order: every field equal, the texture to the byte, each on the mesh of
    ``got_meshes`` at the place its original has in ``want_meshes``"""
    test.assertEqual(type(got), MaterialList)
    test.assertEqual(len(got), len(want))
    for a, b in zip(got, want):
        test.assertEqual(a, b, a.name)
        test.assertEqual((a.name, a.kind, a.texture_file, a.uv_set, a.wrap_u, a.wrap_v), (b.name, b.kind, b.texture_file, b.uv_set, b.wrap_u, b.wrap_v))
        if b.faces is None:
            test.assertIsNone(a.faces, a.name)
        else:
            np.testing.assert_array_equal(a.faces, b.faces, err_msg=a.name)
        if b.texture is None:
            test.assertIsNone(a.texture, a.name)
        else:
            test.assertEqual(a.texture.dtype, b.texture.dtype, a.name)
            np.testing.assert_array_equal(a.texture, b.texture, err_msg=a.name)
        place = [mesh.uuid for mesh in want_meshes].index(b.mesh_uuid)
        test.assertEqual(a.mesh_uuid, got_meshes[place].uuid, a.name)


def fbx_materials(path: str) -> dict:
    """{node name: {materials: [(name, class, shading model, {property:
    value})], mapping, indices, textures: {material: (file exists, file
    name, uv set, wrap u, wrap v, slot)}}} of a file's mesh nodes, through
    the sdk, the embedded media extracted"""
    out = {}
    with open_fbx(path, FBX, media=True) as scene:
        stack = [scene.GetRootNode()]
        while stack:
            node = stack.pop()
            for i in range(node.GetChildCount()):
                stack.append(node.GetChild(i))
            attribute = node.GetNodeAttribute()
            if attribute is None or attribute.GetAttributeType() != FBX.FbxNodeAttribute.EType.eMesh:
                continue
            mesh  = typed(attribute, FBX.FbxMesh, FBX)
            entry = {"materials": [], "mapping": None, "indices": [], "textures": {}, "elements": mesh.GetElementMaterialCount(), "layer": None}
            if mesh.GetElementMaterialCount():
                element          = mesh.GetElementMaterial(0)
                held             = element.GetIndexArray()
                entry["mapping"] = int(getattr(element.GetMappingMode(), "value", element.GetMappingMode()))
                entry["indices"] = [int(held.GetAt(k)) for k in range(held.GetCount())]
                entry["layer"]   = next(i for i in range(mesh.GetLayerCount()) if mesh.GetLayer(i).GetMaterials() is not None)
            for m in range(node.GetMaterialCount()):
                material = typed(node.GetMaterial(m), FBX.FbxSurfaceMaterial, FBX)
                cid      = material.GetClassId()
                props    = {}
                if cid.Is(FBX.FbxSurfaceLambert.ClassId):
                    lambert = typed(material, FBX.FbxSurfacePhong if cid.Is(FBX.FbxSurfacePhong.ClassId) else FBX.FbxSurfaceLambert, FBX)
                    for name in ("Diffuse", "Ambient", "Emissive", "TransparentColor"):
                        props[name] = tuple(round(float(x), 9) for x in tuple(getattr(lambert, name).Get())[:3])
                    for name in ("DiffuseFactor", "AmbientFactor", "EmissiveFactor", "TransparencyFactor"):
                        props[name] = float(getattr(lambert, name).Get())
                    if cid.Is(FBX.FbxSurfacePhong.ClassId):
                        props["Specular"]         = tuple(round(float(x), 9) for x in tuple(lambert.Specular.Get())[:3])
                        props["Reflection"]       = tuple(round(float(x), 9) for x in tuple(lambert.Reflection.Get())[:3])
                        props["SpecularFactor"]   = float(lambert.SpecularFactor.Get())
                        props["Shininess"]        = float(lambert.Shininess.Get())
                        props["ReflectionFactor"] = float(lambert.ReflectionFactor.Get())
                entry["materials"].append((str(material.GetName()), str(cid.GetName()), str(material.ShadingModel.Get()), props))
                prop = material.GetFirstProperty()
                while prop.IsValid():
                    for t in range(prop.GetSrcObjectCount(FBX.FbxCriteria.ObjectType(FBX.FbxFileTexture.ClassId))):
                        tex = typed(prop.GetSrcObject(FBX.FbxCriteria.ObjectType(FBX.FbxFileTexture.ClassId), t), FBX.FbxFileTexture, FBX)
                        # the sdk points an extracted file's names at the extraction folder: its own name is kept
                        entry["textures"][str(material.GetName())] = (
                            os.path.isfile(str(tex.GetFileName())), os.path.basename(str(tex.GetRelativeFileName())), str(tex.UVSet.Get()),
                            int(getattr(tex.WrapModeU.Get(), "value", tex.WrapModeU.Get())),
                            int(getattr(tex.WrapModeV.Get(), "value", tex.WrapModeV.Get())), str(prop.GetName()),
                        )
                    prop = material.GetNextProperty(prop)
            out[str(node.GetName())] = entry
    return out


def other_writer(folder: str) -> str:
    """an sdk file as another writer makes one, beside its textures: node
    quad, a 2 by 2 plane, with a phong "fx" (factors other than 1, a
    texture on its specular slot and a bump map) on faces 0 and 1 and a bare
    material "pbr" (shading model "Unknown") on faces 2 and 3; node tri, a
    triangle, with a lambert "layered" whose colour a layered texture feeds,
    a file texture inside it, its UV set the default; node lost, a triangle,
    with a lambert "lost" whose texture file does not exist; node two, a
    triangle holding two materials under an all-same element"""
    path = os.path.join(folder, "other.fbx")
    spec = os.path.join(folder, "spec.png")
    grad = os.path.join(folder, "grad.png")
    Image.fromarray(gradient()).save(grad)
    Image.fromarray(gradient()[:, :, ::-1].copy()).save(spec)
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene = FBX.FbxScene.Create(manager, "")
        root  = scene.GetRootNode()

        def mesh_node(name, points, faces):
            node = FBX.FbxNode.Create(scene, name)
            root.AddChild(node)
            shape = FBX.FbxMesh.Create(scene, name + "Shape")
            shape.InitControlPoints(len(points))
            for i, point in enumerate(points):
                shape.SetControlPointAt(FBX.FbxVector4(*point, 1.0), i)
            for face in faces:
                shape.BeginPolygon(-1, -1, -1, False)
                for v in face:
                    shape.AddPolygon(v)
                shape.EndPolygon()
            node.SetNodeAttribute(shape)
            return node, shape

        def element(shape, mode, indices):
            made = shape.CreateElementMaterial()
            made.SetMappingMode(mode)
            made.SetReferenceMode(FBX.FbxLayerElement.EReferenceMode.eIndexToDirect)
            for index in indices:
                made.GetIndexArray().Add(index)

        def file_texture(name, filename, uv_set=None):
            tex = FBX.FbxFileTexture.Create(scene, name)
            tex.SetFileName(filename)
            tex.SetRelativeFileName(os.path.basename(filename))
            tex.SetTextureUse(FBX.FbxTexture.ETextureUse.eStandard)
            tex.SetMappingType(FBX.FbxTexture.EMappingType.eUV)
            if uv_set is not None:
                tex.UVSet.Set(FBX.FbxString(uv_set))
            return tex

        grid  = [(x, 0.0, z) for z in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
        quads = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)]
        node, shape = mesh_node("quad", grid, quads)
        fx = FBX.FbxSurfacePhong.Create(scene, "fx")
        fx.Diffuse.Set(FBX.FbxDouble3(0.9, 0.2, 0.1))
        fx.DiffuseFactor.Set(0.8)
        fx.Ambient.Set(FBX.FbxDouble3(0.5, 0.5, 0.5))
        fx.AmbientFactor.Set(0.5)
        fx.Emissive.Set(FBX.FbxDouble3(0.1, 0.1, 0.1))
        fx.EmissiveFactor.Set(2.0)
        fx.TransparentColor.Set(FBX.FbxDouble3(1.0, 1.0, 1.0))
        fx.TransparencyFactor.Set(0.25)
        fx.Specular.Set(FBX.FbxDouble3(1.0, 1.0, 1.0))
        fx.SpecularFactor.Set(0.5)
        fx.Shininess.Set(12.0)
        fx.ReflectionFactor.Set(0.75)
        fx.Specular.ConnectSrcObject(file_texture("specMap", spec, "map1"))
        fx.Bump.ConnectSrcObject(file_texture("bumpMap", spec, "map1"))
        pbr = FBX.FbxSurfaceMaterial.Create(scene, "pbr")
        pbr.ShadingModel.Set(FBX.FbxString("Unknown"))
        node.AddMaterial(fx)
        node.AddMaterial(pbr)
        element(shape, FBX.FbxLayerElement.EMappingMode.eByPolygon, (0, 0, 1, 1))

        tri = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        node, shape = mesh_node("tri", tri, [(0, 1, 2)])
        layered = FBX.FbxSurfaceLambert.Create(scene, "layered")
        stack   = FBX.FbxLayeredTexture.Create(scene, "stack")
        stack.ConnectSrcObject(file_texture("gradMap", grad))
        layered.Diffuse.ConnectSrcObject(stack)
        node.AddMaterial(layered)
        element(shape, FBX.FbxLayerElement.EMappingMode.eAllSame, (0,))

        node, shape = mesh_node("lost", tri, [(0, 1, 2)])
        lost = FBX.FbxSurfaceLambert.Create(scene, "lost")
        lost.Diffuse.ConnectSrcObject(file_texture("lostMap", os.path.join(folder, "gone", "lost.png"), "map1"))
        node.AddMaterial(lost)
        element(shape, FBX.FbxLayerElement.EMappingMode.eAllSame, (0,))

        node, shape = mesh_node("two", tri, [(0, 1, 2)])
        node.AddMaterial(FBX.FbxSurfaceLambert.Create(scene, "first"))
        node.AddMaterial(FBX.FbxSurfacePhong.Create(scene, "second"))
        element(shape, FBX.FbxLayerElement.EMappingMode.eAllSame, (0,))

        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


def mapped_file(folder: str, mode: str, indices, uv_sets: int = 0) -> str:
    """an sdk file whose triangle's material element is mapped ``mode`` with
    ``indices``, one lambert on the node; ``uv_sets`` UV sets made first, a
    layer each, so the sdk's CreateElementMaterial takes the last layer and
    pads the ones under it with empty eNone elements"""
    path    = os.path.join(folder, f"{mode}_{uv_sets}uv.fbx")
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene = FBX.FbxScene.Create(manager, "")
        node  = FBX.FbxNode.Create(scene, "tri")
        scene.GetRootNode().AddChild(node)
        shape = FBX.FbxMesh.Create(scene, "triShape")
        shape.InitControlPoints(3)
        for i, point in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]):
            shape.SetControlPointAt(FBX.FbxVector4(*point, 1.0), i)
        for face in ((0, 1, 2), (2, 1, 0)):
            shape.BeginPolygon(-1, -1, -1, False)
            for v in face:
                shape.AddPolygon(v)
            shape.EndPolygon()
        node.SetNodeAttribute(shape)
        for k in range(uv_sets):
            uv = shape.CreateElementUV(f"map{k + 1}")
            uv.SetMappingMode(FBX.FbxLayerElement.EMappingMode.eByPolygonVertex)
            uv.SetReferenceMode(FBX.FbxLayerElement.EReferenceMode.eIndexToDirect)
            for u, v in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
                uv.GetDirectArray().Add(FBX.FbxVector2(u, v))
            for corner in (0, 1, 2, 2, 1, 0):
                uv.GetIndexArray().Add(corner)
        node.AddMaterial(FBX.FbxSurfaceLambert.Create(scene, "lam"))
        made = shape.CreateElementMaterial()
        made.SetMappingMode(getattr(FBX.FbxLayerElement.EMappingMode, mode))
        made.SetReferenceMode(FBX.FbxLayerElement.EReferenceMode.eIndexToDirect)
        for index in indices:
            made.GetIndexArray().Add(index)
        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMaterials(_Files):
    """materials on their meshes, saved and read back"""

    def test_round_trip_and_what_the_file_holds(self):
        """every field back, the texture to the byte, from a binary and an
        ASCII file, both embedding it (measured: the sdk's ASCII writer
        embeds too); the file as Maya writes one: a lambert and two phongs,
        the colour as the diffuse colour and the diffuse as its factor, the
        other colours with factors of 1, the plane's element by polygon [0,
        0, 1, 1], the box's all-same [0], the texture on DiffuseColor naming
        its UV set and wraps; a second cycle bit for bit"""
        nodes = rig()
        held  = meshes(nodes)
        given = materials(held)
        for as_ascii in (False, True):
            with self.subTest(as_ascii=as_ascii):
                path = self.path("ascii.fbx" if as_ascii else "binary.fbx")
                save_fbx(path, nodes, meshes=held, uvs=plane_uvs(held[0]), materials=given, as_ascii=as_ascii)
                # the file moved to a folder of its own: its texture is inside it
                moved = os.path.join(self.folder, "moved")
                os.makedirs(moved, exist_ok=True)
                path = shutil.copy(path, os.path.join(moved, os.path.basename(path)))
                self.assertEqual(sorted(os.listdir(moved)), [os.path.basename(path)])
                back = load_fbx(path)
                self.assertSameSkeleton(back.hierarchy, nodes)
                self.assertEqual([mesh.name for mesh in back.meshes], ["planeShape", "boxShape"])
                same_materials(self, back.materials, given, back.meshes, held)
                self.assertEqual([back.hierarchy[mesh.parent_uuid].name for mesh in back.meshes], ["plane", "box"])
                # a second cycle, bit for bit
                save_fbx(self.path("again.fbx"), *back)
                again = load_fbx(self.path("again.fbx"))
                same_materials(self, again.materials, back.materials, again.meshes, back.meshes)
                os.remove(path)

        save_fbx(self.path("a.fbx"), nodes, meshes=held, materials=given)
        found = fbx_materials(self.path("a.fbx"))
        self.assertEqual(sorted(found), ["box", "plane"])
        self.assertEqual((found["plane"]["mapping"], found["plane"]["indices"]), (_enum(FBX.FbxLayerElement, "EMappingMode", "eByPolygon"), [0, 0, 1, 1]))
        self.assertEqual((found["box"]["mapping"], found["box"]["indices"]), (_enum(FBX.FbxLayerElement, "EMappingMode", "eAllSame"), [0]))
        (lam, phong) = found["plane"]["materials"]
        self.assertEqual(lam[:3], ("skinLambert", "FbxSurfaceLambert", "Lambert"))
        self.assertEqual(phong[:3], ("shinyPhong", "FbxSurfacePhong", "Phong"))
        self.assertEqual(found["box"]["materials"][0][:3], ("boxPhong", "FbxSurfacePhong", "Phong"))
        props = lam[3]
        self.assertEqual((props["Diffuse"], props["DiffuseFactor"]), ((1.0, 1.0, 1.0), 0.6))
        self.assertEqual((props["Ambient"], props["AmbientFactor"]), ((0.05, 0.1, 0.15), 1.0))
        self.assertEqual((props["Emissive"], props["EmissiveFactor"]), ((0.2, 0.0, 0.1), 1.0))
        self.assertEqual((props["TransparentColor"], props["TransparencyFactor"]), ((0.1, 0.2, 0.3), 1.0))
        self.assertNotIn("Specular", props)
        props = phong[3]
        self.assertEqual((props["Diffuse"], props["DiffuseFactor"]), ((0.9, 0.2, 0.1), 0.8))
        self.assertEqual((props["Specular"], props["SpecularFactor"], props["Shininess"]), ((0.3, 0.4, 0.5), 1.0, 35.0))
        self.assertEqual((props["Reflection"], props["ReflectionFactor"]), ((0.0, 0.0, 0.0), 0.25))
        self.assertEqual(found["plane"]["textures"], {"skinLambert": (True, "grad.png", "map1", 0, 1, "DiffuseColor")})
        self.assertEqual(found["box"]["textures"], {})

    def test_a_texture_of_each_channel_count(self):
        """grey, grey and alpha, RGB and RGBA maps: each back as it was"""
        nodes = rig()
        held  = meshes(nodes)
        maps  = [np.random.default_rng(seed).integers(0, 256, size=(3, 5, channels), dtype=np.uint8) for seed, channels in enumerate((1, 2, 3, 4))]
        given = MaterialList(
            [MaterialData(name=f"map{i}", mesh_uuid=held[0].uuid, faces=[i], texture=pixels, texture_file=f"map{i}.png") for i, pixels in enumerate(maps)]
        )
        save_fbx(self.path("a.fbx"), nodes, meshes=held, materials=given)
        back = load_fbx(self.path("a.fbx"))
        same_materials(self, back.materials, given, back.meshes, held)
        # the embedded file is a PNG named after the texture file, else the
        # material, else "texture", a name taken twice told apart by a number
        bare = MaterialList([MaterialData(mesh_uuid=held[1].uuid, texture=maps[2]), MaterialData(name="dup", mesh_uuid=held[0].uuid, faces=[0, 1], texture=maps[2], texture_file="same.jpg"),
                             MaterialData(name="dup", mesh_uuid=held[0].uuid, faces=[2, 3], texture=maps[3], texture_file="same.jpg"),
                             MaterialData(name="named", mesh_uuid=held[1].uuid, texture=maps[0])])
        with self.assertRaisesRegex(ValueError, "covers the whole mesh 'boxShape' while material 'named' covers it too"):
            save_fbx(self.path("b.fbx"), nodes, meshes=held, materials=bare)
        bare[3].mesh_uuid = held[0].uuid
        bare[3].faces     = [1]
        bare[1].faces     = [0]
        save_fbx(self.path("b.fbx"), nodes, meshes=held, materials=bare)
        back = load_fbx(self.path("b.fbx"))
        self.assertEqual(sorted(m.texture_file for m in back.materials), ["named.png", "same.png", "same_1.png", "texture.png"])

    def test_another_writers_file(self):
        """colours times their factors; a bare material a default lambert
        with one warning; textures on other slots left out with one warning
        naming slot and material; a layered texture read through to its
        file, its UV set None for the default; a missing file None with one
        warning, the name kept; two materials under an all-same element the
        first alone"""
        path = other_writer(self.folder)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = load_fbx(path)
        messages = [str(w.message) for w in caught]
        self.assertEqual(len([m for m in messages if "reads lambert and phong materials" in m]), 1)
        # the sdk writes no shading model for a bare material, as Maya's export of a standardSurface holds none
        self.assertIn("material 'pbr' (shading model '')", [m for m in messages if "reads lambert and phong materials" in m][0])
        left = [m for m in messages if "textures on slots" in m]
        self.assertEqual(len(left), 1)
        self.assertIn("material 'fx' SpecularColor", left[0])
        self.assertIn("material 'fx' Bump", left[0])
        missing = [m for m in messages if "texture files not found" in m]
        self.assertEqual(len(missing), 1)
        self.assertIn("material 'lost'", missing[0])
        self.assertIn("lost.png", missing[0])

        got = {m.name: m for m in back.materials}
        self.assertEqual([m.name for m in back.materials], ["fx", "pbr", "layered", "lost", "first"])
        fx = got["fx"]
        self.assertEqual(fx.kind, "phong")
        np.testing.assert_allclose(fx.color, [0.9, 0.2, 0.1], rtol=0, atol=1e-12)
        self.assertEqual(fx.diffuse, 0.8)
        np.testing.assert_allclose(fx.ambient_color,  [0.25, 0.25, 0.25], rtol=0, atol=1e-12)
        np.testing.assert_allclose(fx.incandescence,  [0.2, 0.2, 0.2],    rtol=0, atol=1e-12)
        np.testing.assert_allclose(fx.transparency,   [0.25, 0.25, 0.25], rtol=0, atol=1e-12)
        np.testing.assert_allclose(fx.specular_color, [0.5, 0.5, 0.5],    rtol=0, atol=1e-12)
        self.assertEqual((fx.cosine_power, fx.reflectivity, fx.texture), (12.0, 0.75, None))
        np.testing.assert_array_equal(fx.faces, [0, 1])
        self.assertEqual(got["pbr"], MaterialData(faces=[2, 3]))  # a default lambert on its faces
        np.testing.assert_array_equal(got["pbr"].faces, [2, 3])
        layered = got["layered"]
        np.testing.assert_array_equal(layered.texture, gradient())
        self.assertEqual((layered.texture_file, layered.uv_set, layered.faces), ("grad.png", None, None))
        lost = got["lost"]
        self.assertEqual((lost.texture, lost.texture_file, lost.uv_set), (None, "lost.png", "map1"))
        self.assertEqual(got["first"].kind, "lambert")
        self.assertEqual([m.mesh_uuid for m in back.materials], [back.meshes[i].uuid for i in (0, 0, 1, 2, 3)])

    def test_the_colour_map_without_pil(self):
        """with no PIL the texture is None, its name kept, with one warning;
        a save with a texture raises ImportError"""
        nodes = rig()
        held  = meshes(nodes)
        save_fbx(self.path("a.fbx"), nodes, meshes=held, materials=materials(held))
        with mock.patch("cgmath.formats._write._pil", return_value=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                back = load_fbx(self.path("a.fbx"))
            self.assertEqual(len([w for w in caught if "PIL (pillow), which is not installed" in str(w.message)]), 1)
            self.assertEqual((back.materials[0].texture, back.materials[0].texture_file, back.materials[0].uv_set), (None, "grad.png", "map1"))
            with self.assertRaisesRegex(ImportError, "material 'skinLambert' holds a texture, written as a PNG through PIL"):
                save_fbx(self.path("b.fbx"), nodes, meshes=held, materials=materials(held))

    def test_refusals(self):
        nodes = rig()
        held  = meshes(nodes)

        def refused(message, given, kind=ValueError):
            with self.assertRaisesRegex(kind, message):
                save_fbx(self.path("refused.fbx"), nodes, meshes=held, materials=given)
            self.assertFalse(os.path.exists(self.path("refused.fbx")))

        refused("material 'skinLambert' names no mesh of meshes=: mesh_uuid None", MaterialData(name="skinLambert"))
        refused("the unnamed material on mesh 'planeShape' channel 'faces' names faces mesh 'planeShape' of 4 faces does not hold",
                MaterialData(mesh_uuid=held[0].uuid, faces=[3, 4]))
        refused("material 'twice' channel 'faces' names a face twice", MaterialData(name="twice", mesh_uuid=held[0].uuid, faces=[1, 1]))
        refused("material 'a' and material 'b' both cover face 1 of mesh 'planeShape'; a face holds one material",
                MaterialList([MaterialData(name="a", mesh_uuid=held[0].uuid, faces=[0, 1]), MaterialData(name="b", mesh_uuid=held[0].uuid, faces=[1, 2, 3])]))
        refused("material 'a', material 'b' leave 1 faces of mesh 'planeShape' without a material; a mesh is covered whole or not at all",
                MaterialList([MaterialData(name="a", mesh_uuid=held[0].uuid, faces=[0, 1]), MaterialData(name="b", mesh_uuid=held[0].uuid, faces=[2])]))
        refused("material 'a' leaves 2 faces of mesh 'planeShape' without a material", MaterialData(name="a", mesh_uuid=held[0].uuid, faces=[0, 1]))
        refused("material 'whole' covers the whole mesh 'planeShape' while material 'part' covers it too; a face holds one material",
                MaterialList([MaterialData(name="part", mesh_uuid=held[0].uuid, faces=[0]), MaterialData(name="whole", mesh_uuid=held[0].uuid)]))
        refused("material 'hdr' channel 'texture' is uint16; a .fbx embeds an 8 bit PNG",
                MaterialData(name="hdr", mesh_uuid=held[0].uuid, texture=gradient().astype(np.uint16)))
        refused(r"material 'wide' channel 'texture' is \(2, 2, 5\), not \(H, W, C\) with 1 to 4 channels",
                MaterialData(name="wide", mesh_uuid=held[0].uuid, texture=np.zeros((2, 2, 5), dtype=np.uint8)))
        refused("the materials are a MaterialList or a MaterialData, not a list", [MaterialData()], TypeError)
        with self.assertRaisesRegex(TypeError, "materials are written on their mesh; missing: meshes="):
            save_fbx(self.path("refused.fbx"), nodes, materials=MaterialData(mesh_uuid=held[0].uuid))
        # a mesh with no material writes none, an empty list none
        save_fbx(self.path("none.fbx"), nodes, meshes=held, materials=MaterialList())
        self.assertIsNone(load_fbx(self.path("none.fbx")).materials)
        self.assertEqual(fbx_materials(self.path("none.fbx"))["plane"]["materials"], [])

        # what the reader refuses: another mapping, faces without a material
        with self.assertRaisesRegex(ValueError, "mesh 'triShape' channel 'materials' is mapped eByControlPoint, which the r17-skeleton fork does not read"):
            load_fbx(mapped_file(self.folder, "eByControlPoint", (0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "mesh 'triShape' channel 'materials' leaves 1 faces without a material"):
            load_fbx(mapped_file(self.folder, "eByPolygon", (0, -1)))
        with self.assertRaisesRegex(ValueError, "mesh 'triShape' channel 'materials' names material 3 of its node's 1"):
            load_fbx(mapped_file(self.folder, "eByPolygon", (0, 3)))
        # a by-polygon element giving one material every face: the whole mesh
        back = load_fbx(mapped_file(self.folder, "eByPolygon", (0, 0)))
        self.assertEqual((len(back.materials), back.materials[0].faces, back.materials[0].name), (1, None, "lam"))
        # the sdk's own placement, two UV layers first: the material element
        # on the last layer, the layer under it padded with an empty eNone
        # element (measured); the mapped element is read, a lone eNone refused
        padded = mapped_file(self.folder, "eAllSame", (0,), uv_sets=2)
        self.assertEqual((fbx_materials(padded)["tri"]["elements"], fbx_materials(padded)["tri"]["mapping"]), (2, _enum(FBX.FbxLayerElement, "EMappingMode", "eNone")))
        back = load_fbx(padded)
        self.assertEqual((len(back.materials), back.materials[0].faces, back.materials[0].name), (1, None, "lam"))
        with self.assertRaisesRegex(ValueError, "mesh 'triShape' channel 'materials' is mapped eNone, which the r17-skeleton fork does not read"):
            load_fbx(mapped_file(self.folder, "eNone", (0,)))

    def test_the_material_element_goes_on_the_first_layer_whatever_the_uv_sets(self):
        """a mesh with two UV sets (a layer each) and a material writes one
        material element, on layer 0, as Maya's exporter does: the sdk's
        CreateElementMaterial would take the last layer and pad layer 0 with
        an empty eNone element, which the reader met first (the UV step's
        Maya test, whose import gives the mesh a Default_Material)"""
        nodes  = rig()
        held   = meshes(nodes)
        first  = plane_uvs(held[0])
        second = UVData(name="lightmap", points=np.asarray(first.points) * 0.5, counts=first.counts, indices=first.indices, mesh_uuid=held[0].uuid)
        given  = MaterialList([MaterialData(name="flat", mesh_uuid=held[0].uuid)])
        path   = self.path("two_sets.fbx")
        save_fbx(path, nodes, meshes=held, uvs=UVList([first, second]), materials=given)
        found = fbx_materials(path)["plane"]
        self.assertEqual((found["elements"], found["layer"], found["mapping"], found["indices"]), (1, 0, _enum(FBX.FbxLayerElement, "EMappingMode", "eAllSame"), [0]))
        back = load_fbx(path)
        same_materials(self, back.materials, given, back.meshes, held)
        self.assertEqual([uv.name for uv in back.uvs], ["map1", "lightmap"])
        # a mesh with no UV set, no layer yet: the element still goes on layer 0
        save_fbx(self.path("no_sets.fbx"), nodes, meshes=held, materials=given)
        found = fbx_materials(self.path("no_sets.fbx"))["plane"]
        self.assertEqual((found["elements"], found["layer"]), (1, 0))


def shared_texture(folder: str) -> str:
    """an sdk file beside its textures: nodes a and b, triangles, both
    holding the one phong "fx", its colour fed by grad.png on UV set map1
    with V clamped, and a bump map, spec.png, on a slot the fork does not
    read"""
    path = os.path.join(folder, "shared.fbx")
    grad = os.path.join(folder, "grad.png")
    spec = os.path.join(folder, "spec.png")
    Image.fromarray(gradient()).save(grad)
    Image.fromarray(gradient()[:, :, ::-1].copy()).save(spec)
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene = FBX.FbxScene.Create(manager, "")
        fx    = FBX.FbxSurfacePhong.Create(scene, "fx")
        for prop, name, filename in ((fx.Diffuse, "colorMap", grad), (fx.Bump, "bumpMap", spec)):
            tex = FBX.FbxFileTexture.Create(scene, name)
            tex.SetFileName(filename)
            tex.SetRelativeFileName(os.path.basename(filename))
            tex.SetTextureUse(FBX.FbxTexture.ETextureUse.eStandard)
            tex.SetMappingType(FBX.FbxTexture.EMappingType.eUV)
            tex.UVSet.Set(FBX.FbxString("map1"))
            tex.WrapModeV.Set(_enum(FBX.FbxTexture, "EWrapMode", "eClamp"))
            prop.ConnectSrcObject(tex)
        for name in ("a", "b"):
            node = FBX.FbxNode.Create(scene, name)
            scene.GetRootNode().AddChild(node)
            shape = FBX.FbxMesh.Create(scene, name + "Shape")
            shape.InitControlPoints(3)
            for i, point in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
                shape.SetControlPointAt(FBX.FbxVector4(*point, 1.0), i)
            shape.BeginPolygon(-1, -1, -1, False)
            for v in (0, 1, 2):
                shape.AddPolygon(v)
            shape.EndPolygon()
            node.SetNodeAttribute(shape)
            node.AddMaterial(fx)
            made = shape.CreateElementMaterial()
            made.SetMappingMode(FBX.FbxLayerElement.EMappingMode.eAllSame)
            made.SetReferenceMode(FBX.FbxLayerElement.EReferenceMode.eIndexToDirect)
            made.GetIndexArray().Add(0)
        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestReadOptions(_Files):
    """load_fbx's flags: a kind left unread is None, a texture named by many
    materials is decoded once and shared, verbose False silences the read"""

    def everything(self) -> str:
        """our writer's file: the nodes, two meshes, a UV set, three
        materials (one textured), a camera and a light"""
        nodes = rig()
        nodes.append(TransformData("cam"))
        nodes.append(TransformData("lamp"))
        held = meshes(nodes)
        path = self.path("all.fbx")
        save_fbx(
            path, nodes, meshes=held, uvs=UVList([plane_uvs(held[0])]), materials=materials(held),
            cameras = CameraList([CameraData(name="cam", focal_length=50.0, parent_uuid=nodes["cam"].uuid)]),
            lights  = LightList([LightData(name="lamp", kind="point", intensity=2.0, parent_uuid=nodes["lamp"].uuid)]),
        )
        return path

    def test_a_kind_left_unread_is_none(self):
        path  = self.everything()
        every = load_fbx(path)
        #                                                  hierarchy clip meshes skins uvs cameras lights materials morphs
        self.assertEqual([x is not None for x in every], [True, False, True, False, True, True, True, True, False])
        self.assertEqual(every.hierarchy.name, ["plane", "box", "cam", "lamp"])

        no_meshes = load_fbx(path, meshes=False)
        self.assertEqual([x is not None for x in no_meshes], [True, False, False, False, False, True, True, False, False])
        self.assertEqual(no_meshes.hierarchy.name, every.hierarchy.name)

        some = load_fbx(path, uvs=False, materials=False, cameras=False)
        self.assertEqual([x is not None for x in some], [True, False, True, False, False, False, True, False, False])
        self.assertEqual(some.meshes.name, every.meshes.name)
        self.assertEqual(some.lights.name, ["lamp"])

        bare = load_fbx(path, textures=False).materials
        self.assertEqual([m.name for m in bare], ["skinLambert", "shinyPhong", "boxPhong"])
        self.assertEqual(
            (bare[0].texture, bare[0].texture_file, bare[0].uv_set, bare[0].wrap_u, bare[0].wrap_v), (None, "grad.png", "map1", "repeat", "clamp")
        )
        np.testing.assert_array_equal(every.materials[0].texture, gradient())
        self.assertEqual(bare[1], every.materials[1])

    def test_takes_off_gives_the_nodes_at_their_own_values(self):
        clip                      = ClipData(rig(), frames=3, fps=24.0, take="spin")
        clip.frames["box"].rotate = np.array([[0.0, 0.0, 0.0], [0.0, 45.0, 0.0], [0.0, 90.0, 0.0]])
        path                      = self.path("spin.fbx")
        save_fbx(path, clip=clip)
        with_takes = load_fbx(path)
        self.assertIsNone(with_takes.hierarchy)
        self.assertEqual([take.take for take in with_takes.clip], ["spin"])
        without = load_fbx(path, takes=False)
        self.assertIsNone(without.clip)
        self.assertEqual(without.hierarchy.name, ["plane", "box"])
        np.testing.assert_allclose(without.hierarchy["box"].translate, [3.0, 0.0, 0.0], rtol=0, atol=1e-12)

    def test_skins_off_gives_the_nodes_their_own_values_not_the_bind_pose(self):
        source = bundle()
        save_fbx(self.path("a.fbx"), **source)

        def moved(scene):
            _named(scene, "spine").LclRotation.Set(FBX.FbxDouble3(40.0, -20.0, 7.0))

        path  = _edited(self.path("a.fbx"), self.path("moved.fbx"), moved)
        bound = load_fbx(path)
        self.assertIsNotNone(bound.skins)
        np.testing.assert_allclose(bound.hierarchy.world_matrix, source["hierarchy"].world_matrix, rtol=0, atol=1e-9)
        own = load_fbx(path, skins=False, takes=False)
        self.assertIsNone(own.skins)
        self.assertEqual(own.hierarchy.name, bound.hierarchy.name)
        np.testing.assert_allclose(own.hierarchy["spine"].rotate, [40.0, -20.0, 7.0], rtol=0, atol=1e-9)
        self.assertIsNone(load_fbx(path, skins=False).hierarchy)  # the takes hold the nodes

    def test_a_texture_file_is_decoded_once_and_each_case_warned_once(self):
        """one phong on two nodes: its colour map one array both materials
        share, decoded once; its bump map named once in the one warning"""
        path = shared_texture(self.folder)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch("cgmath.formats._load._pixels", wraps=_load_module._pixels) as decoded:
                back = load_fbx(path)
        self.assertEqual(decoded.call_count, 1)
        self.assertEqual([m.name for m in back.materials], ["fx", "fx"])
        self.assertEqual([m.mesh_uuid for m in back.materials], [mesh.uuid for mesh in back.meshes])
        self.assertIs(back.materials[0].texture, back.materials[1].texture)
        np.testing.assert_array_equal(back.materials[0].texture, gradient())
        self.assertEqual((back.materials[0].uv_set, back.materials[0].wrap_u, back.materials[0].wrap_v), ("map1", "repeat", "clamp"))
        messages = [str(w.message) for w in caught]
        left     = [m for m in messages if "textures on slots" in m]
        self.assertEqual(len(left), 1, messages)
        self.assertEqual(left[0].count("material 'fx' Bump"), 1, left[0])
        self.assertEqual([m for m in messages if "textures on slots" not in m], [])

    def test_verbose_false_silences_the_read(self):
        path = shared_texture(self.folder)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loud  = load_fbx(path)
            quiet = load_fbx(path, verbose=False)
            scene = Scene.load_fbx(path, verbose=False, textures=False)
            mesh  = MeshData.load_fbx(path, verbose=False)
        self.assertEqual(len(caught), 1, [str(w.message) for w in caught])  # the loud read's
        self.assertEqual(quiet.materials[0], loud.materials[0])
        self.assertEqual((scene.materials[0].texture, scene.materials[0].texture_file), (None, "grad.png"))
        self.assertEqual(mesh.name, "aShape")
        # the filter is the read's alone
        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            load_fbx(path)
        self.assertEqual(len(again), 1)


def shader_count(path: str) -> int:
    """the FbxSurfaceMaterial objects a file holds"""
    with open_fbx(path, FBX, media=False) as scene:
        return int(scene.GetMaterialCount())


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestSharedMaterials(_Files):
    """a material on several meshes: one record a mesh sharing one uuid, read
    from one shader and written back as one"""

    def test_one_shader_on_two_nodes_reads_as_two_records_of_one_uuid(self):
        back = load_fbx(shared_texture(self.folder), verbose=False)
        a, b = back.materials
        self.assertTrue(_is_uuid(a.uuid))
        self.assertEqual(a.uuid, b.uuid)
        self.assertEqual([m.mesh_uuid for m in (a, b)], [mesh.uuid for mesh in back.meshes])
        other = load_fbx(other_writer(self.folder), verbose=False).materials
        self.assertEqual(len({m.uuid for m in other}), 5)  # five shaders, five uuids

    def test_records_sharing_a_uuid_are_written_as_one_shader(self):
        nodes          = rig()
        held           = meshes(nodes)
        sets           = UVList([plane_uvs(held[0]), plane_uvs(held[1])])
        paint          = MaterialData(name="paint", mesh_uuid=held[0].uuid, color=[0.2, 0.4, 0.6], texture=gradient(), texture_file="grad.png", uv_set="map1")
        twin           = paint.copy()
        twin.mesh_uuid = held[1].uuid
        path           = self.path("shared.fbx")
        save_fbx(path, nodes, meshes=held, uvs=sets, materials=MaterialList([paint, twin]))
        self.assertEqual(shader_count(path), 1)
        back = load_fbx(path, verbose=False)
        self.assertEqual([m.name for m in back.materials], ["paint", "paint"])
        self.assertEqual(back.materials[0].uuid, back.materials[1].uuid)
        self.assertEqual([m.mesh_uuid for m in back.materials], [mesh.uuid for mesh in back.meshes])
        self.assertIs(back.materials[0].texture, back.materials[1].texture)  # one file, decoded once
        np.testing.assert_array_equal(back.materials[0].texture, gradient())
        self.assertEqual(back.materials[1], paint)
        # two uuids: two shaders, two files, as before
        twin.uuid = generate_uuid()
        save_fbx(path, nodes, meshes=held, uvs=sets, materials=MaterialList([paint, twin]))
        self.assertEqual(shader_count(path), 2)
        back = load_fbx(path, verbose=False)
        self.assertNotEqual(back.materials[0].uuid, back.materials[1].uuid)
        self.assertEqual((back.materials[0].texture_file, back.materials[1].texture_file), ("grad.png", "grad_1.png"))

    def test_records_sharing_a_uuid_must_agree_but_for_their_faces(self):
        nodes = rig()
        held  = meshes(nodes)
        paint = MaterialData(name="paint", mesh_uuid=held[0].uuid, color=[0.2, 0.4, 0.6])
        twin  = paint.copy()
        twin.mesh_uuid, twin.kind, twin.color = held[1].uuid, "phong", [1.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            ValueError,
            r"save_fbx: material 'paint' on mesh 'boxShape' and material 'paint' on mesh 'planeShape' share uuid "
            r"'[0-9A-F-]+' but differ in kind, color; one material is written per uuid",
        ):
            save_fbx(self.path("x.fbx"), nodes, meshes=held, materials=MaterialList([paint, twin]))
        self.assertFalse(os.path.exists(self.path("x.fbx")))
        faces       = paint.copy()
        faces.faces = [0, 1]
        other       = paint.copy()
        other.faces = [2, 3]
        save_fbx(self.path("y.fbx"), nodes, meshes=held, materials=MaterialList([faces, other]))  # the faces may differ
        self.assertEqual(shader_count(self.path("y.fbx")), 1)
        self.assertEqual([list(m.faces) for m in load_fbx(self.path("y.fbx")).materials], [[0, 1], [2, 3]])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaMaterials(_Recorded):
    """Maya 2025's export of its shaders read as Maya holds them, and Maya's
    import of ours; as recorded"""

    def test_mayas_export_reads_as_maya_holds_it(self):
        """every lambert and phong attribute as Maya holds it, the blinn a
        phong with the shininess Maya converts its eccentricity to, the
        standardSurface a default lambert with one warning, the textured
        colour white as Maya writes it, the texture's pixels, UV set and
        wraps, the shading groups' faces; embedded and linked alike"""
        case = self.case("materials_export")
        made    = case.value("made")
        shaders = case.value("shaders")
        for embed in (True, False):
            with self.subTest(embed=embed):
                out = self.fixture(f"maya_{'embedded' if embed else 'linked'}.fbx")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    back = load_fbx(out)
                self.assertEqual(len([w for w in caught if "reads lambert and phong materials" in str(w.message)]), 1)
                rig    = back.hierarchy if back.hierarchy is not None else back.clip.active
                meshes = {rig[mesh.parent_uuid].name: mesh for mesh in back.meshes}
                got    = {material.name: material for material in back.materials}
                self.assertEqual(sorted(got), ["dullBlinn", "pbr", "shinyPhong", "skinLambert"])
                for shader, (node, faces) in made.items():
                    material = got[shader]
                    reading  = shaders[shader]
                    with self.subTest(shader):
                        self.assertEqual(material.mesh_uuid, meshes[node].uuid)
                        if faces is None:
                            self.assertIsNone(material.faces)
                        else:
                            np.testing.assert_array_equal(material.faces, faces)
                        kind = reading["type"]
                        if kind == "standardSurface":
                            self.assertEqual(material, MaterialData())
                            continue
                        self.assertEqual(material.kind, "lambert" if kind == "lambert" else "phong")
                        if material.texture is None:
                            np.testing.assert_allclose(material.color, reading["color"], rtol=0, atol=1e-7)
                        else:
                            np.testing.assert_array_equal(material.color, [1.0, 1.0, 1.0])  # Maya writes a textured colour white
                        for attr, field in (("transparency", "transparency"), ("ambientColor", "ambient_color"), ("incandescence", "incandescence")):
                            np.testing.assert_allclose(getattr(material, field), reading[attr], rtol=0, atol=1e-7, err_msg=attr)
                        self.assertAlmostEqual(material.diffuse, reading["diffuse"], places=7)
                        if kind == "phong":
                            np.testing.assert_allclose(material.specular_color, reading["specularColor"], rtol=0, atol=1e-7)
                            self.assertAlmostEqual(material.cosine_power, reading["cosinePower"], places=7)
                            self.assertAlmostEqual(material.reflectivity, reading["reflectivity"], places=7)
                        if kind == "blinn":
                            np.testing.assert_allclose(material.specular_color, reading["specularColor"], rtol=0, atol=1e-7)
                            self.assertAlmostEqual(material.reflectivity, reading["reflectivity"], places=7)
                            # Maya 2025's conversion of eccentricity 0.45 (measured)
                            self.assertAlmostEqual(material.cosine_power, 2.6872277431955554, places=6)
                skin = got["skinLambert"]
                np.testing.assert_array_equal(skin.texture, gradient())
                # Maya's export writes both wrap modes as repeat whatever the
                # place2dTexture holds (measured: wrapV off reads as repeat)
                self.assertEqual((skin.texture_file, skin.uv_set, skin.wrap_u, skin.wrap_v), ("grad.png", "map1", "repeat", "repeat"))
                self.assertEqual(case.value("gradPlace_wrapV"), False)  # what the file lost
                self.assertEqual([m.texture for m in back.materials if m.name != "skinLambert"], [None, None, None])

    def test_maya_imports_our_save_as_we_hold_it(self):
        """our materials, imported: Maya's lambert and phong shaders with our
        attributes, the texture on a file node feeding the colour with our
        pixels and wraps, each shading group on our faces"""
        case = self.case("materials_import")
        nodes = rig()
        held  = meshes(nodes)
        given = materials(held)
        self.assertFreshSave(self.fixture("ours.fbx"), hierarchy=nodes, meshes=held, uvs=plane_uvs(held[0]), materials=given)
        readings = case.value("materials")
        for material in given:
            node  = nodes[[mesh.uuid for mesh in held].index(material.mesh_uuid)].name
            count = len(held[[mesh.uuid for mesh in held].index(material.mesh_uuid)].counts)
            entry = readings[material.name]
            with self.subTest(material.name):
                self.assertEqual(entry["shader_count"], 1)
                if material.texture is None:
                    np.testing.assert_allclose(entry["color"], material.color, rtol=0, atol=1e-6)
                else:
                    self.assertEqual(entry["files"], 1)
                    np.testing.assert_array_equal(case.array(f"texture/{material.name}"), material.texture)
                    # Maya's import leaves the place2dTexture's wraps on whatever
                    # the file's wrap modes say (measured: WrapModeV clamp reads as on)
                    self.assertEqual(entry["places"],       1)
                    self.assertEqual(tuple(entry["wraps"]), (True, True))
                    self.assertEqual(material.wrap_v,       "clamp")  # what Maya does not take
                np.testing.assert_allclose(entry["transparency"],  material.transparency,  rtol=0, atol=1e-6)
                np.testing.assert_allclose(entry["ambientColor"],  material.ambient_color, rtol=0, atol=1e-6)
                np.testing.assert_allclose(entry["incandescence"], material.incandescence, rtol=0, atol=1e-6)
                self.assertAlmostEqual(entry["diffuse"], material.diffuse, places=6)
                if material.kind == "phong":
                    np.testing.assert_allclose(entry["specularColor"], material.specular_color, rtol=0, atol=1e-6)
                    self.assertAlmostEqual(entry["cosinePower"], material.cosine_power, places=6)
                    self.assertAlmostEqual(entry["reflectivity"], material.reflectivity, places=6)
                self.assertEqual(entry["groups"], 1)
                members = entry["members"]
                faces   = sorted(int(m.group(1)) for member in members for m in [re.search(r"\.f\[(\d+)\]$", member)] if m)
                whole   = [member for member in members if ".f[" not in member]
                if material.faces is None:
                    self.assertTrue(whole or faces == list(range(count)), members)
                    for member in whole:
                        self.assertIn(node, member)
                else:
                    self.assertEqual(faces, sorted(material.faces.tolist()), members)
                    for member in members:
                        self.assertIn(node, member)


if __name__ == "__main__":
    unittest.main()
