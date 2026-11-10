"""Vertex colours through load_fbx and save_fbx: Maya's conventions, measured.

Maya's exporter (2025) writes a colour set as an FbxGeometryElementVertexColor
named as the set, by polygon vertex (a shared, per-vertex set too), index to
direct, one RGBA colour a face corner (16 for a 2 by 2 plane, none shared),
alpha 1 for an RGB set, on the mesh's first layer beside its first UV set; a
second set goes on the next layer. Its importer reads a by-control-point
element and a by-polygon-vertex one alike, to the value, keeps two sets with
their names, and names an unnamed element "colorSet0". The fork writes one
set, "colorSet1", Maya's own name for a first set: by control point for a
colour a vertex, by polygon vertex otherwise, index to direct as Maya writes
both, on the first layer by hand (the sdk's CreateElementVertexColor takes
the last layer and pads the ones under it); it reads a file's first set,
warning of the others. The SDK tests need the Autodesk SDK; the Maya ones
read what Maya 2025 did, recorded once (maya_fixtures), and need no Maya.
"""

import unittest

import numpy as np
from cgmath._tests.test_fbx_materials import mapped_file
from cgmath._tests.test_skeleton_fbx import _Files, _Recorded
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, new_manager, open_fbx, typed
from cgmath.geometry import MeshData, UVData, UVList
from cgmath.geometry.mesh import _corner_colors

FBX = load_sdk()


def grid() -> MeshData:
    """a 2 by 2 plane: 9 points, 4 quads, 16 face corners, as Maya's polyPlane"""
    points = [(x, 0.0, z) for z in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
    return MeshData(
        points=points, indices=[0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], counts=[4, 4, 4, 4], name="gridShape",
    )


VERTEX_COLORS = np.column_stack([np.arange(9) / 8.0, 1.0 - np.arange(9) / 8.0, np.full(9, 0.25), 0.5 + np.arange(9) / 16.0])
CORNER_COLORS = np.column_stack([np.arange(16) / 15.0, np.zeros(16), 1.0 - np.arange(16) / 15.0, np.ones(16)])
PALETTE       = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.75], [0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.25]])
PALETTE_IDS   = np.repeat(np.arange(4, dtype=np.int32), 4)


def colored(form: str) -> MeshData:
    """the grid with a colour a vertex, a colour a corner, or a colour a face
    through four the corners name"""
    mesh = grid()
    if form == "vertex":
        mesh.colors = VERTEX_COLORS
    elif form == "corner":
        mesh.colors, mesh.color_indices = CORNER_COLORS, np.arange(16, dtype=np.int32)
    else:
        mesh.colors, mesh.color_indices = PALETTE, PALETTE_IDS
    return mesh


def per_corner(mesh) -> np.ndarray:
    colors, ids = _corner_colors(mesh)
    return colors[ids]


def maps(mesh) -> UVList:
    """two UV sets on the mesh: a layer each in the file"""
    points = np.asarray(mesh.points)
    uv     = np.stack([(points[:, 0] + 1.0) / 2.0, (points[:, 2] + 1.0) / 2.0], axis=1)
    return UVList([
        UVData(name="map1", points=uv, counts=mesh.counts, indices=mesh.indices, mesh_uuid=mesh.uuid),
        UVData(name="lightmap", points=uv * 0.5, counts=mesh.counts, indices=mesh.indices, mesh_uuid=mesh.uuid),
    ])


def fbx_colors(path: str) -> list:
    """every mesh's vertex colour elements, through the sdk: (name, mapping,
    reference, (K, 4) direct colours, index list, layer)"""
    out = []
    with open_fbx(path, FBX) as scene:
        stack = [scene.GetRootNode()]
        while stack:
            node = stack.pop()
            for i in range(node.GetChildCount()):
                stack.append(node.GetChild(i))
            attribute = node.GetNodeAttribute()
            if attribute is None or attribute.GetAttributeType() != FBX.FbxNodeAttribute.EType.eMesh:
                continue
            mesh = typed(attribute, FBX.FbxMesh, FBX)
            for k in range(mesh.GetElementVertexColorCount()):
                element = mesh.GetElementVertexColor(k)
                direct  = element.GetDirectArray()
                index   = element.GetIndexArray()
                layer = next(
                    (i for i in range(mesh.GetLayerCount())
                     if mesh.GetLayer(i).GetVertexColors() is not None and mesh.GetLayer(i).GetVertexColors().GetName() == element.GetName()),
                    None,
                )
                out.append((
                    str(element.GetName()),
                    str(element.GetMappingMode()).rsplit(".", 1)[-1],
                    str(element.GetReferenceMode()).rsplit(".", 1)[-1],
                    np.array([[c.mRed, c.mGreen, c.mBlue, c.mAlpha] for c in (direct.GetAt(i) for i in range(direct.GetCount()))]).reshape(-1, 4),
                    [index.GetAt(i) for i in range(index.GetCount())],
                    layer,
                ))
    return out


def sdk_file(path: str, elements, by_hand: bool = True, uv_sets: int = 0) -> str:
    """a quad with the vertex colour ``elements``, (name, mapping, reference,
    colours, indices) each, placed on the first free layer by hand, or as
    the sdk's CreateElementVertexColor places them; ``uv_sets`` UV sets
    first, a layer each"""
    manager = new_manager(FBX)
    try:
        scene = FBX.FbxScene.Create(manager, "scene")
        node  = FBX.FbxNode.Create(scene, "geo")
        mesh  = FBX.FbxMesh.Create(scene, "geoShape")
        mesh.InitControlPoints(4)
        for i, (x, y) in enumerate(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
            mesh.SetControlPointAt(FBX.FbxVector4(x, y, 0.0), i)
        mesh.BeginPolygon(-1, -1, -1, False)
        for i in range(4):
            mesh.AddPolygon(i)
        mesh.EndPolygon()
        node.SetNodeAttribute(mesh)
        scene.GetRootNode().AddChild(node)
        for k in range(uv_sets):
            uv = mesh.CreateElementUV(f"map{k + 1}")
            uv.SetMappingMode(FBX.FbxLayerElement.EMappingMode.eByPolygonVertex)
            uv.SetReferenceMode(FBX.FbxLayerElement.EReferenceMode.eIndexToDirect)
            for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
                uv.GetDirectArray().Add(FBX.FbxVector2(u, v))
            for corner in range(4):
                uv.GetIndexArray().Add(corner)
        for name, mapping, reference, colors, indices in elements:
            if by_hand:
                element = FBX.FbxLayerElementVertexColor.Create(mesh, name)
                if mesh.GetLayerCount() == 0:
                    mesh.CreateLayer()
                layer = next((mesh.GetLayer(i) for i in range(mesh.GetLayerCount()) if mesh.GetLayer(i).GetVertexColors() is None), None)
                if layer is None:
                    layer = mesh.GetLayer(mesh.CreateLayer())
                layer.SetVertexColors(element)
            else:
                element = mesh.CreateElementVertexColor()
                element.SetName(name)
            element.SetMappingMode(getattr(FBX.FbxLayerElement.EMappingMode, mapping))
            element.SetReferenceMode(getattr(FBX.FbxLayerElement.EReferenceMode, reference))
            for color in colors:
                element.GetDirectArray().Add(FBX.FbxColor(*color))
            for index in indices:
                element.GetIndexArray().Add(int(index))
        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


FOUR = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.75), (0.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.25)]


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestColors(_Files):
    """colours on their meshes, saved and read back; the sdk's own files read"""

    def test_round_trip_and_what_the_file_holds(self):
        """every form back as it was, from a binary file to the bit and from
        an ASCII one to the 15 digits it prints a double with, a second cycle
        bit for bit; the file holding one set, colorSet1, on layer 0, by
        control point for a colour a vertex, by polygon vertex otherwise,
        index to direct, with two UV sets on their layers too"""
        for form in ("vertex", "corner", "palette"):
            mesh = colored(form)
            for as_ascii in (False, True):
                with self.subTest(form=form, as_ascii=as_ascii):
                    path = self.path(f"{form}_{'ascii' if as_ascii else 'binary'}.fbx")
                    save_fbx(path, meshes=mesh, uvs=maps(mesh), as_ascii=as_ascii)
                    back = load_fbx(path).meshes[0]
                    np.testing.assert_allclose(back.colors, mesh.colors, rtol=0.0, atol=1e-15 if as_ascii else 0.0)
                    if form == "vertex":
                        self.assertIsNone(back.color_indices)
                    else:
                        np.testing.assert_array_equal(back.color_indices, mesh.color_indices)
                    np.testing.assert_allclose(per_corner(back), per_corner(mesh), rtol=0.0, atol=1e-15 if as_ascii else 0.0)
                    save_fbx(self.path("again.fbx"), meshes=back, uvs=maps(back), as_ascii=as_ascii)
                    self.assertEqual(load_fbx(self.path("again.fbx")).meshes[0], back)
            held = fbx_colors(self.path(f"{form}_binary.fbx"))
            self.assertEqual(len(held), 1)
            name, mapping, reference, direct, index, layer = held[0]
            self.assertEqual((name, reference, layer), ("colorSet1", "eIndexToDirect", 0))
            self.assertEqual(mapping, "eByControlPoint" if form == "vertex" else "eByPolygonVertex")
            np.testing.assert_array_equal(direct, mesh.colors)
            self.assertEqual(index, list(range(9)) if form == "vertex" else mesh.color_indices.tolist())
        # the class route reads the same
        np.testing.assert_array_equal(MeshData.load_fbx(self.path("vertex_binary.fbx")).colors, VERTEX_COLORS)
        # a mesh without colours writes no element
        save_fbx(self.path("none.fbx"), meshes=grid())
        self.assertEqual(fbx_colors(self.path("none.fbx")), [])
        self.assertIsNone(load_fbx(self.path("none.fbx")).meshes[0].colors)

    def test_the_sdks_own_files_read(self):
        """by control point, direct or indexed, is a colour a vertex; by
        polygon vertex, direct or indexed, a colour a corner; an element the
        sdk placed itself past two UV layers, padding the lower ones with
        eNone elements, reads as the mapped one; two sets give the first with
        a warning; another mapping is refused"""
        back = load_fbx(sdk_file(self.path("point_direct.fbx"), [("colorSet1", "eByControlPoint", "eDirect", FOUR, [])])).meshes[0]
        np.testing.assert_array_equal(back.colors, FOUR)
        self.assertIsNone(back.color_indices)
        back = load_fbx(sdk_file(self.path("point_indexed.fbx"), [("c", "eByControlPoint", "eIndexToDirect", FOUR[:2], [1, 0, 1, 0])])).meshes[0]
        np.testing.assert_array_equal(back.colors, [FOUR[1], FOUR[0], FOUR[1], FOUR[0]])
        self.assertIsNone(back.color_indices)
        back = load_fbx(sdk_file(self.path("corner_direct.fbx"), [("c", "eByPolygonVertex", "eDirect", FOUR, [])])).meshes[0]
        np.testing.assert_array_equal(per_corner(back), FOUR)
        self.assertIsNotNone(back.color_indices)
        back = load_fbx(sdk_file(self.path("corner_indexed.fbx"), [("c", "eByPolygonVertex", "eIndexToDirect", FOUR[:2], [0, 1, 1, 0])])).meshes[0]
        np.testing.assert_array_equal(per_corner(back), [FOUR[0], FOUR[1], FOUR[1], FOUR[0]])
        padded = sdk_file(self.path("padded.fbx"), [("c", "eByControlPoint", "eIndexToDirect", FOUR, [0, 1, 2, 3])], by_hand=False, uv_sets=2)
        self.assertEqual([(mapping, layer) for _, mapping, _, _, _, layer in fbx_colors(padded)], [("eNone", 0), ("eByControlPoint", 1)])
        np.testing.assert_array_equal(load_fbx(padded).meshes[0].colors, FOUR)
        two = sdk_file(
            self.path("two.fbx"),
            [("colorSet1", "eByControlPoint", "eDirect", FOUR, []), ("masks", "eByPolygonVertex", "eDirect", [(0.5, 0.5, 0.5, 1.0)] * 4, [])],
        )
        with self.assertWarnsRegex(UserWarning, r"mesh 'geoShape' holds the colour sets \['masks'\] beside 'colorSet1'; the r17-skeleton fork reads the first"):
            back = load_fbx(two).meshes[0]
        np.testing.assert_array_equal(back.colors, FOUR)
        with self.assertRaisesRegex(ValueError, "mesh 'geoShape' channel 'colors' is mapped eByPolygon, which the r17-skeleton fork does not read"):
            load_fbx(sdk_file(self.path("face.fbx"), [("c", "eByPolygon", "eDirect", FOUR[:1], [])]))
        # a material file's own mesh, no colour element: None, as before
        self.assertIsNone(load_fbx(mapped_file(self.folder, "eAllSame", (0,))).meshes[0].colors)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaColors(_Recorded):
    """Maya 2025's export of its colour sets read as Maya holds them, and
    Maya's import of ours; as recorded"""

    def test_mayas_export_reads_as_maya_holds_it(self):
        """a shared RGBA set and an RGB set on a 2 by 2 plane: Maya writes a
        colour a face corner, read back to the value, alpha 1 for RGB"""
        case = self.case("colors_export")
        self.assertEqual(case.value("shared/sets"), ["colorSet1"])
        back = load_fbx(self.fixture("shared.fbx")).meshes[0]
        self.assertIsNotNone(back.color_indices)  # Maya writes by polygon vertex, a shared set too
        self.assertEqual(back.colors.shape, (16, 4))
        np.testing.assert_allclose(per_corner(back), case.array("shared/colors"), atol=1e-6)
        self.assertEqual([x[0] for x in fbx_colors(self.fixture("shared.fbx"))], ["colorSet1"])

        self.assertEqual(case.value("rgb/sets"), ["tint"])
        back = load_fbx(self.fixture("rgb.fbx")).meshes[0]
        np.testing.assert_allclose(per_corner(back)[:, :3], case.array("rgb/colors")[:, :3], atol=1e-6)
        np.testing.assert_array_equal(per_corner(back)[:, 3], 1.0)

    def test_maya_imports_our_save_as_we_hold_it(self):
        """a colour a vertex and a colour a corner: one set, colorSet1, every
        corner's colour ours to the value"""
        case = self.case("colors_import")
        for form in ("vertex", "corner", "palette"):
            with self.subTest(form=form):
                mesh = colored(form)
                self.assertFreshSave(self.fixture(f"ours_{form}.fbx"), meshes=mesh)
                self.assertEqual(case.value(f"{form}/sets"), ["colorSet1"])
                np.testing.assert_allclose(case.array(f"{form}/colors"), per_corner(mesh), atol=1e-6)
