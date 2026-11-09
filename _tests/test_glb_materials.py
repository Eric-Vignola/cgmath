"""Each mesh loaded from a glb gets its own primitive's material.

Scene.load_glb used to give every mesh material 0's texture, whatever its
primitive pointed at. The fixture crosses them: the first mesh uses material
1 and the second material 0, so material-0-for-all and an off-by-one pairing
both fail.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from cgmath.render.scene import _glb_material_indices, Scene

try:
    import pygltflib
except ImportError:
    pygltflib = None

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from PIL import Image
except ImportError:
    Image = None

RED  = (255, 0, 0)
BLUE = (0, 0, 255)


def png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), rgb, np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def write_glb(path, materials, palette=(RED, BLUE), modes=None):
    """one textured triangle per entry of ``materials``: the index of the
    material it uses, one per ``palette`` colour, or None for no material.
    ``modes`` gives each primitive's glTF mode (4, triangles, by default)."""
    gltf = pygltflib.GLTF2()
    blob = bytearray()

    def view(data):
        while len(blob) % 4:
            blob.append(0)
        gltf.bufferViews.append(
            pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data))
        )
        blob.extend(data)
        return len(gltf.bufferViews) - 1

    def accessor(array, kind, component):
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView    = view(array.tobytes()),
                componentType = component,
                count         = len(array),
                type          = kind,
                max           = array.max(axis=0).tolist() if array.ndim > 1 else None,
                min           = array.min(axis=0).tolist() if array.ndim > 1 else None,
            )
        )
        return len(gltf.accessors) - 1

    for index, rgb in enumerate(palette):
        gltf.images.append(pygltflib.Image(bufferView=view(png(rgb)), mimeType="image/png"))
        gltf.textures.append(pygltflib.Texture(source=index))
        gltf.materials.append(
            pygltflib.Material(
                name=f"mat{index}",
                pbrMetallicRoughness = pygltflib.PbrMetallicRoughness(
                    baseColorTexture = pygltflib.TextureInfo(index=index)
                ),
            )
        )

    uvs     = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    indices = np.array([0, 1, 2], np.uint32)
    for index, (material, mode) in enumerate(zip(materials, modes or [4] * len(materials))):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        points[:, 0] += 2 * index
        primitive = pygltflib.Primitive(
            attributes = pygltflib.Attributes(
                POSITION   = accessor(points, "VEC3", pygltflib.FLOAT),
                TEXCOORD_0 = accessor(uvs, "VEC2", pygltflib.FLOAT),
            ),
            indices  = accessor(indices, "SCALAR", pygltflib.UNSIGNED_INT),
            material = material,
            mode     = mode,
        )
        gltf.meshes.append(pygltflib.Mesh(name=f"part{index}", primitives=[primitive]))
        gltf.nodes.append(pygltflib.Node(name=f"part{index}", mesh=index))

    gltf.scenes = [pygltflib.Scene(nodes=list(range(len(materials))))]
    gltf.scene  = 0
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(blob)))
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(path)
    return path


def color(material):
    """the first texel of the material's texture, as 0-255 RGB"""
    texture = np.asarray(material.texture)
    if texture.dtype.kind == "f":
        texture = np.round(texture * 255)
    return tuple(int(x) for x in texture[0, 0, :3])


def by_mesh(scene):
    """each mesh's material, by the mesh's name, None for a mesh with none"""
    held = {material.mesh_uuid: material for material in scene.materials or []}
    return {mesh.name: held.get(mesh.uuid) for mesh in scene.meshes}


@unittest.skipIf(
    pygltflib is None or trimesh is None or Image is None,
    "pygltflib, trimesh and PIL are needed",
)
class TestGlbMaterials(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = write_glb(os.path.join(tmp.name, "parts.glb"), [1, 0, None])

    def test_scene(self):
        scene     = Scene.load_glb(self.path)
        materials = by_mesh(scene)
        self.assertEqual([x.name for x in scene.meshes], ["part0", "part1", "part2"])
        self.assertEqual(color(materials["part0"]),      BLUE)
        self.assertEqual(color(materials["part1"]),      RED)
        self.assertIsNone(materials["part2"])
        self.assertEqual(len(scene.materials), 2)
        # each material reads the mesh's own UV set
        sets = {uv.mesh_uuid: uv.name for uv in scene.uvs}
        for material in scene.materials:
            self.assertEqual(material.uv_set, sets[material.mesh_uuid])

    def test_pairing(self):
        self.assertEqual(_glb_material_indices(self.path, 3), [1, 0, None])

        # trimesh dropped a primitive: no guessing between two materials
        with self.assertWarnsRegex(UserWarning, "cannot be paired"):
            self.assertEqual(_glb_material_indices(self.path, 2), [None, None])

    def test_a_primitive_trimesh_skips_is_skipped_here_too(self):
        # a triangle fan: trimesh reads only triangles and strips
        path = write_glb(self.path.replace("parts", "fan"), [1, 0, 0], modes=[4, 6, 4])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scene = Scene.load_glb(path)
        materials = by_mesh(scene)
        self.assertEqual([x.name for x in scene.meshes], ["part0", "part2"])
        self.assertEqual([color(materials[x]) for x in ("part0", "part2")], [BLUE, RED])

    def test_no_textures_asked_for_reads_no_materials(self):
        path = write_glb(self.path.replace("parts", "fan"), [1, 0, 0], modes=[4, 6, 4])
        with patch("cgmath.render.scene._glb_material_indices") as pairing:
            scene = Scene.load_glb(path, extract_texture=False)
        pairing.assert_not_called()
        self.assertIsNone(scene.materials)
        self.assertEqual(len(scene.meshes), 2)

    def test_one_material_is_not_given_to_a_primitive_without_one(self):
        path = write_glb(
            self.path.replace("parts", "single_fan"), [0, 0, None], palette=(RED,), modes=[4, 6, 4]
        )
        materials = by_mesh(Scene.load_glb(path))
        self.assertEqual(color(materials["part0"]), RED)
        self.assertIsNone(materials["part2"])

    def test_a_single_material_still_pairs(self):
        path = write_glb(self.path.replace("parts", "single"), [0, 0], palette=(RED,))
        self.assertEqual(_glb_material_indices(path, 5), [0] * 5)


if __name__ == "__main__":
    unittest.main()
