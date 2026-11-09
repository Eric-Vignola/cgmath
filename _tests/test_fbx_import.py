"""Tests for FBX mesh / UV / texture import.

The Autodesk FBX Python SDK is an optional binary dependency that is not
installed in CI. Every test in this module therefore patches a synthetic
``fbx`` module into ``cgmath.geometry.mesh`` that mimics just the API
surface the production code calls.
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
    attr_etype            = types.SimpleNamespace(eMesh="eMesh")
    fake.FbxNodeAttribute = types.SimpleNamespace(EType=attr_etype)
    pivot_set             = types.SimpleNamespace(eSourcePivot="eSourcePivot")
    fake.FbxNode          = types.SimpleNamespace(EPivotSet=pivot_set)

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

    # a scene in cm; the mesh reader reads no axis system, so the fake has
    # none: a reader that turned the points would fail here
    unit                             = scene.GetGlobalSettings.return_value.GetSystemUnit.return_value
    unit.GetScaleFactor.return_value = 1.0

    fake_fbx.FbxManager.Create.return_value    = manager
    fake_fbx.FbxIOSettings.Create.return_value = ios
    fake_fbx.FbxImporter.Create.return_value   = importer
    fake_fbx.FbxScene.Create.return_value      = scene
    return manager


# -- fake FBX node / mesh / element factories --------------------------------


def _make_fake_control_point(xyz):
    """An object indexable via ``[0]/[1]/[2]`` like ``FbxVector4``."""
    cp                         = MagicMock()
    cp.__getitem__.side_effect = lambda i: xyz[i]
    return cp


def _make_fake_direct_array(values):
    """A ``GetCount``/``GetAt`` array of fixed-dimension vectors."""
    arr                       = MagicMock()
    arr.GetCount.return_value = len(values)
    arr.GetAt.side_effect     = lambda i, _vals=values: _make_fake_control_point(_vals[i])
    return arr


def _make_fake_index_array(indices):
    arr                       = MagicMock()
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
    element                             = MagicMock()
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
    mesh                                    = MagicMock(name="fbx-mesh")
    mesh.GetControlPointsCount.return_value = len(control_points)
    mesh.GetControlPointAt.side_effect = lambda i, _cps=control_points: (
        _make_fake_control_point(_cps[i])
    )
    mesh.GetPolygonCount.return_value = len(polygons)
    mesh.GetPolygonSize.side_effect   = lambda i, _p=polygons: len(_p[i])
    mesh.GetPolygonVertex.side_effect = lambda face, v, _p=polygons: int(_p[face][v])

    uv_elements                         = uv_elements or []
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
    node                      = MagicMock(name=f"node:{name}")
    node.GetName.return_value = name

    attr                               = MagicMock()
    attr.GetAttributeType.return_value = fake_fbx.FbxNodeAttribute.EType.eMesh
    node.GetNodeAttribute.return_value = attr
    # Make _walk_fbx_mesh_nodes treat this node as a leaf.
    node.GetChildCount.return_value = 0
    # no geometric transform, the offset a node gives its mesh alone
    node.GetGeometricTranslation.return_value = (0.0, 0.0, 0.0, 1.0)
    node.GetGeometricRotation.return_value    = (0.0, 0.0, 0.0, 1.0)
    node.GetGeometricScaling.return_value     = (1.0, 1.0, 1.0, 1.0)

    materials                          = materials or []
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
    root                               = MagicMock(name="root")
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




if __name__ == "__main__":
    unittest.main()
