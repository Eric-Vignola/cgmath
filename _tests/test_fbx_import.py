"""Tests for FBX mesh / UV / texture import.

The Autodesk FBX Python SDK is an optional binary dependency that is not
installed in CI. Every test in this module therefore patches a synthetic
``fbx`` module into ``cgmath.geometry.mesh`` (and ``cgmath.render.scene``)
that mimics just the API surface the production code calls.
"""

from __future__ import annotations

import io
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from cgmath.geometry import mesh as mesh_mod
from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
from cgmath.render import scene as scene_mod
from cgmath.render.scene import Object

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None


# -- synthetic FBX SDK builder ------------------------------------------------


def _make_fake_fbx_module() -> types.SimpleNamespace:
    """Build a ``types.SimpleNamespace`` shaped like the Autodesk FBX SDK.

    Only the attributes / classes / enums referenced by ``mesh.py`` and
    ``scene.py`` are populated. ``FbxFileTexture.ClassId`` and
    ``FbxLayeredTexture.ClassId`` are sentinel strings used purely for
    identity comparison by the production code.
    """
    fake = types.SimpleNamespace()

    # ---- enums --------------------------------------------------------------
    attr_etype = types.SimpleNamespace(eMesh="eMesh")
    fake.FbxNodeAttribute = types.SimpleNamespace(EType=attr_etype)

    mapping = types.SimpleNamespace(
        eByPolygonVertex = "ByPolygonVertex",
        eByControlPoint  = "ByControlPoint",
    )
    reference = types.SimpleNamespace(
        eDirect        = "Direct",
        eIndexToDirect = "IndexToDirect",
    )
    fake.FbxLayerElement = types.SimpleNamespace(
        EMappingMode   = mapping,
        EReferenceMode = reference,
    )

    # ---- class ids (used as opaque identity tokens) ------------------------
    fake.FbxFileTexture    = types.SimpleNamespace(ClassId="FbxFileTexture")
    fake.FbxLayeredTexture = types.SimpleNamespace(ClassId="FbxLayeredTexture")

    # ---- IOSettings sentinels ----------------------------------------------
    fake.IOSROOT                       = "IOSROOT"
    fake.IMP_FBX_EXTRACT_EMBEDDED_DATA = "Import|ExtractEmbeddedData"

    # ---- factory shims -----------------------------------------------------
    fake.FbxManager    = types.SimpleNamespace(Create=MagicMock())
    fake.FbxIOSettings = types.SimpleNamespace(Create=MagicMock())
    fake.FbxImporter   = types.SimpleNamespace(Create=MagicMock())
    fake.FbxScene      = types.SimpleNamespace(Create=MagicMock())
    return fake


def _wire_importer(
    fake_fbx:      types.SimpleNamespace,
    *,
    root_node,
    initialize_ok: bool                  = True,
    import_ok:     bool                  = True,
    error:         str                   = "fbx error",
) -> MagicMock:
    """Wire the FBX factory ``Create`` calls to return mocks that produce
    ``root_node`` when the scene is queried. Returns the manager mock so
    callers can inspect ``Destroy()`` calls.
    """
    manager  = MagicMock(name="manager")
    ios      = MagicMock(name="ios")
    importer = MagicMock(name="importer")
    scene    = MagicMock(name="scene")

    importer.Initialize.return_value                            = initialize_ok
    importer.Import.return_value                                = import_ok
    importer.GetStatus.return_value.GetErrorString.return_value = error
    scene.GetRootNode.return_value                              = root_node
    manager.GetIOSettings.return_value                          = ios

    fake_fbx.FbxManager.Create.return_value                     = manager
    fake_fbx.FbxIOSettings.Create.return_value                  = ios
    fake_fbx.FbxImporter.Create.return_value                    = importer
    fake_fbx.FbxScene.Create.return_value                       = scene
    return manager


# -- fake FBX node / mesh / element factories --------------------------------


def _make_fake_control_point(xyz):
    """An object indexable via ``[0]/[1]/[2]`` like ``FbxVector4``."""
    cp = MagicMock()
    cp.__getitem__.side_effect = lambda i: xyz[i]
    return cp


def _make_fake_direct_array(values):
    """A ``GetCount``/``GetAt`` array of fixed-dimension vectors."""
    arr = MagicMock()
    arr.GetCount.return_value = len(values)
    arr.GetAt.side_effect     = lambda i, _vals=values: _make_fake_control_point(_vals[i])
    return arr


def _make_fake_index_array(indices):
    arr = MagicMock()
    arr.GetCount.return_value = len(indices)
    arr.GetAt.side_effect     = lambda i, _ix=indices: int(_ix[i])
    return arr


def _make_fake_uv_element(
    fake_fbx,
    *,
    points,
    indices   = None,
    mapping   = None,
    reference = None,
    name      = "map1",
):
    mapping   = mapping or fake_fbx.FbxLayerElement.EMappingMode.eByPolygonVertex
    reference = reference or fake_fbx.FbxLayerElement.EReferenceMode.eIndexToDirect
    element   = MagicMock()
    element.GetDirectArray.return_value   = _make_fake_direct_array(points)
    element.GetIndexArray.return_value    = _make_fake_index_array(indices or [])
    element.GetMappingMode.return_value   = mapping
    element.GetReferenceMode.return_value = reference
    element.GetName.return_value          = name
    return element


def _make_fake_normal_element(fake_fbx, *, points, indices):
    element = MagicMock()
    element.GetDirectArray.return_value = _make_fake_direct_array(points)
    element.GetIndexArray.return_value  = _make_fake_index_array(indices)
    element.GetMappingMode.return_value = (
        fake_fbx.FbxLayerElement.EMappingMode.eByPolygonVertex
    )
    element.GetReferenceMode.return_value = (
        fake_fbx.FbxLayerElement.EReferenceMode.eIndexToDirect
    )
    return element


def _make_fake_mesh(
    fake_fbx,
    *,
    control_points,
    polygons,
    uv_elements    = None,
    normal_element = None,
):
    """Build a fake FbxMesh attribute.

    ``polygons`` is a list of vertex-id tuples, one per face.
    """
    mesh = MagicMock(name="fbx-mesh")
    mesh.GetControlPointsCount.return_value = len(control_points)
    mesh.GetControlPointAt.side_effect = lambda i, _cps=control_points: (
        _make_fake_control_point(_cps[i])
    )
    mesh.GetPolygonCount.return_value = len(polygons)
    mesh.GetPolygonSize.side_effect   = lambda i, _p=polygons: len(_p[i])
    mesh.GetPolygonVertex.side_effect = lambda face, v, _p=polygons: int(_p[face][v])

    uv_elements = uv_elements or []
    mesh.GetElementUVCount.return_value = len(uv_elements)
    mesh.GetElementUV.side_effect       = lambda i, _uv=uv_elements: _uv[i]

    if normal_element is not None:
        mesh.GetElementNormalCount.return_value = 1
        mesh.GetElementNormal.return_value      = normal_element
    else:
        mesh.GetElementNormalCount.return_value = 0
    return mesh


def _make_fake_mesh_node(fake_fbx, *, name, fbx_mesh, materials=None):
    """Build a fake FbxNode whose attribute is ``fbx_mesh``."""
    node = MagicMock(name=f"node:{name}")
    node.GetName.return_value = name

    attr = MagicMock()
    attr.GetAttributeType.return_value = fake_fbx.FbxNodeAttribute.EType.eMesh
    node.GetNodeAttribute.return_value = attr
    # Make _walk_fbx_mesh_nodes treat this node as a leaf.
    node.GetChildCount.return_value = 0

    materials = materials or []
    node.GetMaterialCount.return_value = len(materials)
    node.GetMaterial.side_effect       = lambda i, _m=materials: _m[i]

    # Bind the mesh onto the node so production code finds it via
    # attribute.* lookups -- some callsites use ``node.GetNodeAttribute()``
    # then treat the result as the mesh itself.
    node.GetNodeAttribute.return_value     = fbx_mesh
    fbx_mesh.GetAttributeType.return_value = fake_fbx.FbxNodeAttribute.EType.eMesh
    return node


def _make_fake_root(children):
    """Synthetic root node that exposes the given children list."""
    root = MagicMock(name="root")
    root.GetNodeAttribute.return_value = None
    root.GetChildCount.return_value    = len(children)
    root.GetChild.side_effect          = lambda i, _c=children: _c[i]
    return root


# -- unit-cube fixtures ------------------------------------------------------


def _unit_triangle_mesh(fake_fbx, *, with_normals=False, with_uvs=True, name="hero"):
    """Single-triangle mesh: 3 verts, 1 tri, 1 UV channel, optional normals.

    Returns the wrapped FbxNode ready to drop into a fake root.
    """
    control_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    polygons       = [(0, 1, 2)]
    uv_elements    = []
    if with_uvs:
        uv_elements = [
            _make_fake_uv_element(
                fake_fbx,
                points  = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                indices = [0, 1, 2],
            )
        ]
    normal_element = None
    if with_normals:
        normal_element = _make_fake_normal_element(
            fake_fbx,
            points  = [(0, 0, 1), (0, 0, 1), (0, 0, 1)],
            indices = [0, 1, 2],
        )

    fbx_mesh = _make_fake_mesh(
        fake_fbx,
        control_points = control_points,
        polygons       = polygons,
        uv_elements    = uv_elements,
        normal_element = normal_element,
    )
    return _make_fake_mesh_node(fake_fbx, name=name, fbx_mesh=fbx_mesh)


def _make_real_file(suffix: str = ".fbx") -> str:
    """Touch a real on-disk file so the production existence check passes."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# -----------------------------------------------------------------------------
# load_fbx
# -----------------------------------------------------------------------------


class TestLoadFbx(unittest.TestCase):
    """Coverage for ``cgmath.geometry.mesh.load_fbx`` (module-level)."""

    def test_raises_when_fbx_sdk_missing(self):
        with patch.object(mesh_mod, "fbx", None):
            with self.assertRaises(ImportError):
                mesh_mod.load_fbx("anything.fbx")

    def test_raises_when_file_missing(self):
        fake = _make_fake_fbx_module()
        with patch.object(mesh_mod, "fbx", fake):
            with self.assertRaises(FileNotFoundError):
                mesh_mod.load_fbx("/no/such/path/missing_file.fbx")

    def test_returns_one_mesh_with_uvs(self):
        fake = _make_fake_fbx_module()
        node = _unit_triangle_mesh(fake, with_normals=False)
        root = _make_fake_root([node])

        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                manager = _wire_importer(fake, root_node=root)
                result  = mesh_mod.load_fbx(path)
        finally:
            os.unlink(path)

        self.assertEqual(len(result), 1)
        mesh_data, uv_list = result[0]
        self.assertIsInstance(mesh_data, MeshData)
        self.assertEqual(mesh_data.point_count,  3)
        self.assertEqual(mesh_data.face_count,   1)
        self.assertEqual(mesh_data.name,         "hero")
        self.assertEqual(len(uv_list),           1)
        self.assertEqual(uv_list[0].point_count, 3)
        # Manager always destroyed via try/finally.
        manager.Destroy.assert_called_once()

    def test_initialize_failure_raises_runtime_error(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])

        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(
                    fake, root_node=root, initialize_ok=False, error="bad header"
                )
                with self.assertRaises(RuntimeError) as ctx:
                    mesh_mod.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertIn("bad header", str(ctx.exception))

    def test_import_failure_raises_runtime_error(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])

        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(
                    fake, root_node=root, import_ok=False, error="corrupt scene"
                )
                with self.assertRaises(RuntimeError) as ctx:
                    mesh_mod.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertIn("corrupt scene", str(ctx.exception))

    def test_scene_without_mesh_nodes_returns_empty(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])

        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                result = mesh_mod.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertEqual(result, [])

    def test_normals_are_extracted_when_authored(self):
        fake = _make_fake_fbx_module()
        node = _unit_triangle_mesh(fake, with_normals=True)
        root = _make_fake_root([node])

        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                result = mesh_mod.load_fbx(path)
        finally:
            os.unlink(path)
        mesh_data, _ = result[0]
        # Normals are stored directly on the mesh as ``self.normals`` /
        # ``self.normal_indices`` -- the FBX loader populates them via the
        # ``MeshData(normals=..., normal_indices=...)`` constructor kwargs.
        self.assertIsNotNone(mesh_data.normals)
        self.assertEqual(np.asarray(mesh_data.normals).shape, (3, 3))
        self.assertIsNotNone(mesh_data.normal_indices)

    def test_uv_by_control_point_direct_mapping(self):
        """ByControlPoint + eDirect projects per-vertex UVs onto a
        face-varying index stream (one entry per face-vertex)."""
        fake = _make_fake_fbx_module()
        # 4 verts in a quad fan: triangle (0,1,2), triangle (0,2,3)
        # UVs authored once per control point.
        cps   = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        polys = [(0, 1, 2), (0, 2, 3)]
        uv_element = _make_fake_uv_element(
            fake,
            points    = [(0, 0), (1, 0), (1, 1), (0, 1)],
            indices   = [],  # eDirect ignores the index array
            mapping   = fake.FbxLayerElement.EMappingMode.eByControlPoint,
            reference = fake.FbxLayerElement.EReferenceMode.eDirect,
        )
        fbx_mesh = _make_fake_mesh(
            fake, control_points=cps, polygons=polys, uv_elements=[uv_element]
        )
        node = _make_fake_mesh_node(fake, name="quad", fbx_mesh=fbx_mesh)
        root = _make_fake_root([node])

        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                result = mesh_mod.load_fbx(path)
        finally:
            os.unlink(path)
        _, uv_list = result[0]
        # 6 face-vertex entries (3 + 3) all pointing at one of 4 UV points.
        self.assertEqual(uv_list[0].indices.shape, (6,))
        self.assertEqual(uv_list[0].points.shape, (4, 2))


# -----------------------------------------------------------------------------
# MeshData.load_fbx / UVData.load_fbx
# -----------------------------------------------------------------------------


def _two_mesh_loader_result(fake_fbx):
    """Helper: a load_fbx-style return list with two named meshes."""
    n0   = _unit_triangle_mesh(fake_fbx, name="hero")
    n1   = _unit_triangle_mesh(fake_fbx, name="prop")
    root = _make_fake_root([n0, n1])
    return root


class TestMeshDataLoadFbx(unittest.TestCase):
    def test_returns_first_mesh_when_name_is_none(self):
        fake = _make_fake_fbx_module()
        root = _two_mesh_loader_result(fake)
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                mesh_data = MeshData.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertEqual(mesh_data.name, "hero")

    def test_returns_named_mesh(self):
        fake = _make_fake_fbx_module()
        root = _two_mesh_loader_result(fake)
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                mesh_data = MeshData.load_fbx(path, name="prop")
        finally:
            os.unlink(path)
        self.assertEqual(mesh_data.name, "prop")

    def test_unknown_name_raises_value_error_with_available_names(self):
        fake = _make_fake_fbx_module()
        root = _two_mesh_loader_result(fake)
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                with self.assertRaises(ValueError) as ctx:
                    MeshData.load_fbx(path, name="ghost")
        finally:
            os.unlink(path)
        msg = str(ctx.exception)
        self.assertIn("ghost", msg)
        self.assertIn("hero",  msg)
        self.assertIn("prop",  msg)

    def test_empty_file_raises_runtime_error(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                with self.assertRaises(RuntimeError):
                    MeshData.load_fbx(path)
        finally:
            os.unlink(path)


class TestUVDataLoadFbx(unittest.TestCase):
    def test_returns_default_channel(self):
        fake = _make_fake_fbx_module()
        node = _unit_triangle_mesh(fake)
        root = _make_fake_root([node])
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                uv = UVData.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertIsInstance(uv, UVData)
        self.assertEqual(uv.point_count, 3)

    def test_channel_out_of_range_raises_index_error(self):
        fake = _make_fake_fbx_module()
        node = _unit_triangle_mesh(fake)
        root = _make_fake_root([node])
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                with self.assertRaises(IndexError):
                    UVData.load_fbx(path, channel=5)
        finally:
            os.unlink(path)


# -----------------------------------------------------------------------------
# MeshList.load_fbx / UVList.load_fbx
# -----------------------------------------------------------------------------


class TestMeshListLoadFbx(unittest.TestCase):
    def test_returns_all_meshes(self):
        fake = _make_fake_fbx_module()
        root = _two_mesh_loader_result(fake)
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                meshes = MeshList.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(meshes), 2)
        self.assertEqual([m.name for m in meshes], ["hero", "prop"])


class TestUVListLoadFbx(unittest.TestCase):
    def test_returns_all_channels_for_first_mesh(self):
        fake = _make_fake_fbx_module()
        node = _unit_triangle_mesh(fake)
        root = _make_fake_root([node])
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                uvs = UVList.load_fbx(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(uvs), 1)

    def test_returns_channels_for_named_mesh(self):
        fake = _make_fake_fbx_module()
        root = _two_mesh_loader_result(fake)
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                uvs = UVList.load_fbx(path, name="prop")
        finally:
            os.unlink(path)
        self.assertEqual(len(uvs), 1)

    def test_unknown_name_raises_value_error(self):
        fake = _make_fake_fbx_module()
        root = _two_mesh_loader_result(fake)
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                with self.assertRaises(ValueError):
                    UVList.load_fbx(path, name="ghost")
        finally:
            os.unlink(path)

    def test_empty_file_raises_runtime_error(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])
        path = _make_real_file()
        try:
            with patch.object(mesh_mod, "fbx", fake):
                _wire_importer(fake, root_node=root)
                with self.assertRaises(RuntimeError):
                    UVList.load_fbx(path)
        finally:
            os.unlink(path)


# -----------------------------------------------------------------------------
# Object.load_fbx
# -----------------------------------------------------------------------------


def _stub_object_loader(monkey_patch_target, *, mesh_data, uv):
    """Stub ``scene._load_fbx`` so Object.load_fbx skips the real SDK path."""
    return patch.object(
        monkey_patch_target, "_load_fbx", return_value=[(mesh_data, UVList([uv]))]
    )


def _tiny_mesh_and_uv():
    mesh = MeshData(
        indices = np.array([0, 1, 2], dtype=np.int64),
        counts  = np.array([3], dtype=np.int64),
        points=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        ),
        name="hero",
    )
    uv = UVData(
        indices = np.array([0, 1, 2], dtype=np.int64),
        counts  = np.array([3], dtype=np.int64),
        points  = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        name    = "map1",
    )
    return mesh, uv


class TestObjectLoadFbx(unittest.TestCase):
    def test_load_fbx_auto_extracts_texture(self):
        mesh, uv = _tiny_mesh_and_uv()
        tex = np.full((4, 4, 3), 200, dtype=np.uint8)

        with (
            _stub_object_loader(scene_mod, mesh_data=mesh, uv=uv),
            patch.object(
                scene_mod, "_extract_fbx_textures", return_value={"base_color": tex}
            ),
        ):
            obj = Object.load_fbx("fake.fbx")
        self.assertIsInstance(obj, Object)
        self.assertIs(obj.mesh, mesh)
        self.assertIs(obj.uv, uv)
        np.testing.assert_array_equal(obj.texture, tex)

    def test_load_fbx_extract_texture_false_skips_extraction(self):
        mesh, uv = _tiny_mesh_and_uv()
        with (
            _stub_object_loader(scene_mod, mesh_data=mesh, uv=uv),
            patch.object(scene_mod, "_extract_fbx_textures") as extract,
        ):
            obj = Object.load_fbx("fake.fbx", extract_texture=False)
        self.assertIsNone(obj.texture)
        extract.assert_not_called()

    def test_load_fbx_explicit_texture_wins(self):
        mesh, uv = _tiny_mesh_and_uv()
        explicit = np.full((2, 2, 3), 50, dtype=np.uint8)
        with (
            _stub_object_loader(scene_mod, mesh_data=mesh, uv=uv),
            patch.object(scene_mod, "_extract_fbx_textures") as extract,
        ):
            obj = Object.load_fbx("fake.fbx", texture=explicit)
        np.testing.assert_array_equal(obj.texture, explicit)
        # Auto-extraction is suppressed when a caller-supplied texture wins.
        extract.assert_not_called()

    def test_load_fbx_no_meshes_raises(self):
        with patch.object(scene_mod, "_load_fbx", return_value=[]):
            with self.assertRaises(ValueError):
                Object.load_fbx("empty.fbx")

    def test_load_fbx_index_out_of_range_raises(self):
        mesh, uv = _tiny_mesh_and_uv()
        with _stub_object_loader(scene_mod, mesh_data=mesh, uv=uv):
            with self.assertRaises(IndexError):
                Object.load_fbx("fake.fbx", index=7)

    def test_load_fbx_negative_index_supported(self):
        mesh, uv = _tiny_mesh_and_uv()
        with (
            _stub_object_loader(scene_mod, mesh_data=mesh, uv=uv),
            patch.object(scene_mod, "_extract_fbx_textures", return_value={}),
        ):
            obj = Object.load_fbx("fake.fbx", index=-1)
        self.assertIs(obj.mesh, mesh)


# -----------------------------------------------------------------------------
# _extract_fbx_textures + Object.extract_texture_from_fbx
# -----------------------------------------------------------------------------


def _png_bytes(color=(128, 64, 32), size=(2, 2)) -> bytes:
    img = _PILImage.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_fake_property(*, src_objects, valid=True):
    prop = MagicMock()
    prop.IsValid.return_value           = valid
    prop.GetSrcObjectCount.return_value = len(src_objects)
    prop.GetSrcObject.side_effect       = lambda i, _o=src_objects: _o[i]
    return prop


def _make_fake_file_texture(fake_fbx, *, filename: str, relative: str = ""):
    tex = MagicMock()
    tex.GetClassId.return_value          = fake_fbx.FbxFileTexture.ClassId
    tex.GetFileName.return_value         = filename
    tex.GetRelativeFileName.return_value = relative
    return tex


def _make_fake_layered_texture(fake_fbx, *, children):
    layered = MagicMock()
    layered.GetClassId.return_value        = fake_fbx.FbxLayeredTexture.ClassId
    layered.GetSrcObjectCount.return_value = len(children)
    layered.GetSrcObject.side_effect       = lambda i, _c=children: _c[i]
    return layered


def _write_png_at(path: str, color=(7, 8, 9)):
    img = _PILImage.new("RGB", (3, 3), color)
    img.save(path, format="PNG")


@unittest.skipIf(_PILImage is None, "PIL is required for texture decoding")
class TestExtractFbxTextures(unittest.TestCase):
    def test_returns_empty_when_fbx_sdk_missing(self):
        with patch.object(scene_mod, "_fbx", None):
            self.assertEqual(scene_mod._extract_fbx_textures("anything.fbx"), {})

    def test_returns_empty_when_pil_missing(self):
        fake = _make_fake_fbx_module()
        with (
            patch.object(scene_mod, "_fbx", fake),
            patch.object(scene_mod, "Image", None),
        ):
            self.assertEqual(scene_mod._extract_fbx_textures("anything.fbx"), {})

    def test_returns_empty_when_file_missing(self):
        fake = _make_fake_fbx_module()
        with patch.object(scene_mod, "_fbx", fake):
            self.assertEqual(scene_mod._extract_fbx_textures("/no/such/file.fbx"), {})

    def test_returns_empty_when_initialize_fails(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])
        path = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root, initialize_ok=False)
                self.assertEqual(scene_mod._extract_fbx_textures(path), {})
        finally:
            os.unlink(path)

    def test_returns_empty_when_no_mesh_nodes(self):
        fake = _make_fake_fbx_module()
        root = _make_fake_root([])
        path = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                self.assertEqual(scene_mod._extract_fbx_textures(path), {})
        finally:
            os.unlink(path)

    def test_returns_empty_when_material_index_out_of_range(self):
        fake     = _make_fake_fbx_module()
        fbx_mesh = _make_fake_mesh(fake, control_points=[(0, 0, 0)], polygons=[])
        node     = _make_fake_mesh_node(fake, name="x", fbx_mesh=fbx_mesh, materials=[])
        root     = _make_fake_root([node])
        path     = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                self.assertEqual(scene_mod._extract_fbx_textures(path), {})
        finally:
            os.unlink(path)

    def test_extracts_diffuse_color_texture(self):
        fake = _make_fake_fbx_module()

        # Drop a real PNG on disk for the texture path resolver.
        tex_path = _make_real_file(suffix=".png")
        _write_png_at(tex_path, color=(11, 22, 33))

        file_tex     = _make_fake_file_texture(fake, filename=tex_path)
        diffuse_prop = _make_fake_property(src_objects=[file_tex])
        invalid_prop = _make_fake_property(src_objects=[], valid=False)

        material     = MagicMock()

        def _find_property(name):
            if name == "DiffuseColor":
                return diffuse_prop
            return invalid_prop

        material.FindProperty.side_effect = _find_property

        fbx_mesh = _make_fake_mesh(fake, control_points=[(0, 0, 0)], polygons=[])
        node = _make_fake_mesh_node(
            fake, name="hero", fbx_mesh=fbx_mesh, materials=[material]
        )
        root = _make_fake_root([node])

        path = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                textures = scene_mod._extract_fbx_textures(path)
        finally:
            os.unlink(path)
            os.unlink(tex_path)

        self.assertIn("base_color", textures)
        self.assertIsInstance(textures["base_color"], np.ndarray)
        # Only base_color was authored -- other slots stay out.
        self.assertEqual(set(textures.keys()), {"base_color"})

    def test_normal_map_wins_over_bump(self):
        """NormalMap and Bump both map to ``normal``; the first one
        encountered (NormalMap, by ``_FBX_TEXTURE_SLOTS`` order) wins via
        ``dict.setdefault``."""
        fake        = _make_fake_fbx_module()

        normal_path = _make_real_file(suffix=".png")
        bump_path   = _make_real_file(suffix=".png")
        _write_png_at(normal_path, color=(255, 0, 0))
        _write_png_at(bump_path, color=(0, 255, 0))

        normal_tex   = _make_fake_file_texture(fake, filename=normal_path)
        bump_tex     = _make_fake_file_texture(fake, filename=bump_path)

        normal_prop  = _make_fake_property(src_objects=[normal_tex])
        bump_prop    = _make_fake_property(src_objects=[bump_tex])
        invalid_prop = _make_fake_property(src_objects=[], valid=False)

        material     = MagicMock()
        material.FindProperty.side_effect = lambda name: {
            "NormalMap": normal_prop,
            "Bump":      bump_prop,
        }.get(name, invalid_prop)

        fbx_mesh = _make_fake_mesh(fake, control_points=[(0, 0, 0)], polygons=[])
        node = _make_fake_mesh_node(
            fake, name="hero", fbx_mesh=fbx_mesh, materials=[material]
        )
        root = _make_fake_root([node])

        path = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                textures = scene_mod._extract_fbx_textures(path)
        finally:
            os.unlink(path)
            os.unlink(normal_path)
            os.unlink(bump_path)

        # The red NormalMap image should have won the ``normal`` slot.
        np.testing.assert_array_equal(textures["normal"][0, 0, :3], [255, 0, 0])

    def test_handles_layered_texture(self):
        fake     = _make_fake_fbx_module()
        tex_path = _make_real_file(suffix=".png")
        _write_png_at(tex_path, color=(1, 2, 3))

        file_tex     = _make_fake_file_texture(fake, filename=tex_path)
        layered      = _make_fake_layered_texture(fake, children=[file_tex])
        diffuse_prop = _make_fake_property(src_objects=[layered])
        invalid_prop = _make_fake_property(src_objects=[], valid=False)

        material     = MagicMock()
        material.FindProperty.side_effect = lambda name: (
            diffuse_prop if name == "DiffuseColor" else invalid_prop
        )
        fbx_mesh = _make_fake_mesh(fake, control_points=[(0, 0, 0)], polygons=[])
        node = _make_fake_mesh_node(
            fake, name="x", fbx_mesh=fbx_mesh, materials=[material]
        )
        root = _make_fake_root([node])

        path = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                textures = scene_mod._extract_fbx_textures(path)
        finally:
            os.unlink(path)
            os.unlink(tex_path)
        self.assertIn("base_color", textures)

    def test_resolves_relative_path_against_fbx_directory(self):
        fake = _make_fake_fbx_module()

        # Put the texture next to the (fake) FBX file.
        tmpdir = tempfile.mkdtemp()
        try:
            fbx_path     = os.path.join(tmpdir, "hero.fbx")
            tex_filename = "diffuse.png"
            tex_path     = os.path.join(tmpdir, tex_filename)
            with open(fbx_path, "w") as f:
                f.write("")
            _write_png_at(tex_path, color=(4, 5, 6))

            # FBX SDK returns an empty absolute path but a valid relative one.
            file_tex     = _make_fake_file_texture(fake, filename="", relative=tex_filename)
            diffuse_prop = _make_fake_property(src_objects=[file_tex])
            invalid_prop = _make_fake_property(src_objects=[], valid=False)

            material     = MagicMock()
            material.FindProperty.side_effect = lambda name: (
                diffuse_prop if name == "DiffuseColor" else invalid_prop
            )
            fbx_mesh = _make_fake_mesh(fake, control_points=[(0, 0, 0)], polygons=[])
            node = _make_fake_mesh_node(
                fake, name="x", fbx_mesh=fbx_mesh, materials=[material]
            )
            root = _make_fake_root([node])

            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                textures = scene_mod._extract_fbx_textures(fbx_path)
            self.assertIn("base_color", textures)
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_returns_empty_when_texture_path_unresolvable(self):
        fake = _make_fake_fbx_module()
        file_tex = _make_fake_file_texture(
            fake, filename="/no/such/diffuse.png", relative=""
        )
        diffuse_prop = _make_fake_property(src_objects=[file_tex])
        invalid_prop = _make_fake_property(src_objects=[], valid=False)
        material     = MagicMock()
        material.FindProperty.side_effect = lambda name: (
            diffuse_prop if name == "DiffuseColor" else invalid_prop
        )
        fbx_mesh = _make_fake_mesh(fake, control_points=[(0, 0, 0)], polygons=[])
        node = _make_fake_mesh_node(
            fake, name="x", fbx_mesh=fbx_mesh, materials=[material]
        )
        root = _make_fake_root([node])
        path = _make_real_file()
        try:
            with patch.object(scene_mod, "_fbx", fake):
                _wire_importer(fake, root_node=root)
                textures = scene_mod._extract_fbx_textures(path)
        finally:
            os.unlink(path)
        # Unresolvable path -> base_color is omitted (no entry, not None).
        self.assertNotIn("base_color", textures)


class TestObjectExtractTextureFromFbx(unittest.TestCase):
    def test_assigns_base_color(self):
        mesh, uv = _tiny_mesh_and_uv()
        obj = Object(name="hero", mesh=mesh, uv=uv)
        self.assertIsNone(obj.texture)
        tex = np.full((4, 4, 3), 99, dtype=np.uint8)
        with patch.object(
            scene_mod, "_extract_fbx_textures", return_value={"base_color": tex}
        ):
            ok = obj.extract_texture_from_fbx("fake.fbx")
        self.assertTrue(ok)
        np.testing.assert_array_equal(obj.texture, tex)

    def test_returns_false_when_no_base_color(self):
        mesh, uv = _tiny_mesh_and_uv()
        existing = np.full((2, 2, 3), 17, dtype=np.uint8)
        obj      = Object(name="hero", mesh=mesh, uv=uv, texture=existing)
        with patch.object(scene_mod, "_extract_fbx_textures", return_value={}):
            ok = obj.extract_texture_from_fbx("fake.fbx")
        self.assertFalse(ok)
        # Existing texture preserved on no-op.
        np.testing.assert_array_equal(obj.texture, existing)

    def test_returns_false_when_only_other_slots_extracted(self):
        """Normal-only result should not clobber base_color or claim success."""
        mesh, uv = _tiny_mesh_and_uv()
        obj = Object(name="hero", mesh=mesh, uv=uv)
        with patch.object(
            scene_mod,
            "_extract_fbx_textures",
            return_value={"normal": np.zeros((2, 2, 3), dtype=np.uint8)},
        ):
            ok = obj.extract_texture_from_fbx("fake.fbx")
        self.assertFalse(ok)
        self.assertIsNone(obj.texture)