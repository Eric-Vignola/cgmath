"""Tests for reading skin weights out of a glb.

Fixtures are synthesized with pygltflib rather than shipped as binaries, the
way ``test_glb.py`` does it, so every index in play is visible in the test.

The load bearing case is the ``skin.joints`` indirection: a vertex's
``JOINTS_0`` entry indexes the skin's own joint list, not the file's nodes.
Reading it as a node index still runs and still produces plausible looking
weights, so the fixtures below put decoy nodes exactly where a missing
indirection would land.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from cgmath.geometry import SkinData
from cgmath.geometry.skin_weights import load_glb

try:
    import pygltflib
    from pygltflib import (
        Accessor,
        Attributes,
        Buffer,
        BufferView,
        GLTF2,
        Mesh,
        Node,
        Primitive,
        Scene,
        Skin,
    )
except ImportError:
    pygltflib = None

try:
    import trimesh
except ImportError:
    trimesh = None


FLOAT        = 5126
USHORT       = 5123
TRIANGLES    = 4
TRIANGLE_FAN = 6


class Builder:
    """accumulates accessors into one buffer and writes a glb"""

    def __init__(self):
        self.blob      = b""
        self.views     = []
        self.accessors = []

    def add(self, array, kind="VEC3", ctype=FLOAT):
        raw = array.tobytes()
        self.views.append(
            BufferView(buffer=0, byteOffset=len(self.blob), byteLength=len(raw))
        )
        self.blob += raw + b"\x00" * ((-len(raw)) % 4)

        accessor = Accessor(
            bufferView    = len(self.views) - 1,
            componentType = ctype,
            count         = array.shape[0],
            type          = kind,
        )
        if kind == "VEC3":
            accessor.min = array.min(axis=0).tolist()
            accessor.max = array.max(axis=0).tolist()

        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def primitive(self, points, joints=None, weights=None, mode=TRIANGLES):
        attributes = {"POSITION": self.add(np.asarray(points, dtype=np.float32))}
        if joints is not None:
            attributes["JOINTS_0"] = self.add(
                np.asarray(joints, dtype=np.uint16), kind="VEC4", ctype=USHORT
            )
            attributes["WEIGHTS_0"] = self.add(
                np.asarray(weights, dtype=np.float32), kind="VEC4", ctype=FLOAT
            )

        indices = np.arange(len(points), dtype=np.uint16)
        return Primitive(
            attributes = Attributes(**attributes),
            indices    = self.add(indices, kind="SCALAR", ctype=USHORT),
            mode       = mode,
        )

    def identity_binds(self, count):
        return self.add(
            np.tile(np.eye(4, dtype=np.float32).ravel(), (count, 1)),
            kind  = "MAT4",
            ctype = FLOAT,
        )

    def write(self, path, meshes, nodes, skins, scene_nodes=None):
        gltf = GLTF2()
        gltf.buffers     = [Buffer(byteLength=len(self.blob))]
        gltf.bufferViews = self.views
        gltf.accessors   = self.accessors
        gltf.meshes      = meshes
        gltf.nodes       = nodes
        gltf.skins       = skins
        gltf.scenes = [
            Scene(
                nodes=scene_nodes
                if scene_nodes is not None
                else list(range(len(nodes)))
            )
        ]
        gltf.scene = 0
        gltf.set_binary_blob(self.blob)
        gltf.save(path)
        return path


def triangle(offset=0.0):
    return np.array(
        [[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]], dtype=np.float32
    )


def one_hot(slots):
    """(N, 4) joints/weights pinning vertex v entirely to slot ``slots[v]``"""
    joints  = np.zeros((len(slots), 4), dtype=np.uint16)
    weights = np.zeros((len(slots), 4), dtype=np.float32)
    joints[:, 0] = slots
    weights[:, 0] = 1.0
    return joints, weights


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class GlbSkinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def decoy_file(self, name="decoy.glb"):
        """skin.joints is [0, 3, 4], with decoys parked at nodes 1 and 2

        Reading JOINTS_0 as a node index yields ['root', 'geo', 'decoy'].
        Reading it correctly yields ['root', 'spine', 'head'].
        """
        builder = Builder()
        joints, weights = one_hot([0, 1, 2])
        primitive = builder.primitive(triangle(), joints, weights)
        binds     = builder.identity_binds(3)

        return builder.write(
            self.path(name),
            [Mesh(primitives=[primitive], name="body")],
            [
                Node(name="root", children=[3]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="decoy"),
                Node(name="spine", children=[4]),
                Node(name="head"),
            ],
            [Skin(joints=[0, 3, 4], inverseBindMatrices=binds)],
            scene_nodes=[0, 1, 2],
        )


class TestIndirection(GlbSkinTest):
    def test_joints_0_is_read_through_the_skins_joint_list(self):
        """the whole point -- a node index read would give ['root','geo','decoy']"""
        data = load_glb(self.decoy_file())

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].influences, ["root", "spine", "head"])

    def test_each_vertex_lands_on_the_joint_its_slot_names(self):
        skin = load_glb(self.decoy_file())[0]

        self.assertEqual(skin.weights.shape, (3, 3))
        np.testing.assert_allclose(skin.weights, np.eye(3))

    def test_fractional_weights_survive(self):
        builder = Builder()
        joints  = np.zeros((3, 4), dtype=np.uint16)
        joints[:, 0] = [0, 1, 2]
        joints[:, 1] = [1, 2, 0]
        weights = np.zeros((3, 4), dtype=np.float32)
        weights[:, 0] = 0.75
        weights[:, 1] = 0.25

        primitive = builder.primitive(triangle(), joints, weights)
        binds     = builder.identity_binds(3)
        path = builder.write(
            self.path("fractional.glb"),
            [Mesh(primitives=[primitive], name="body")],
            [
                Node(name="root", children=[2]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="spine", children=[3]),
                Node(name="head"),
            ],
            [Skin(joints=[0, 2, 3], inverseBindMatrices=binds)],
            scene_nodes=[0, 1],
        )

        skin     = load_glb(path)[0]

        expected = np.array([[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]])
        np.testing.assert_allclose(skin.weights, expected, atol=1e-7)

    def test_a_joint_repeated_across_slots_accumulates(self):
        """glTF lets two slots name the same joint; the scatter must add, not overwrite"""
        builder = Builder()
        joints  = np.zeros((1, 4), dtype=np.uint16)
        weights = np.zeros((1, 4), dtype=np.float32)
        joints[0] = [1, 1, 0, 0]
        weights[0] = [0.5, 0.5, 0.0, 0.0]

        primitive = builder.primitive(
            np.array([[0, 0, 0]], dtype=np.float32), joints, weights
        )
        binds = builder.identity_binds(2)
        path = builder.write(
            self.path("repeat.glb"),
            [Mesh(primitives=[primitive], name="body")],
            [
                Node(name="root", children=[2]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="spine"),
            ],
            [Skin(joints=[0, 2], inverseBindMatrices=binds)],
            scene_nodes=[0, 1],
        )

        skin = load_glb(path)[0]

        np.testing.assert_allclose(skin.weights, [[0.0, 1.0]])


class TestNaming(GlbSkinTest):
    def test_influences_resolve_against_a_rig_from_the_same_file(self):
        from cgmath.hierarchy import HierarchyData

        path = self.decoy_file()
        skin = load_glb(path)[0]
        rig  = [str(x) for x in HierarchyData.load_glb(path).name]

        for influence in skin.influences:
            self.assertIn(influence, rig)

    def test_a_joint_the_rig_drops_raises(self):
        """'RootNode' is filtered out at hierarchy.py:1262, so skinning to it
        cannot produce a name anything downstream could resolve
        """
        builder = Builder()
        joints, weights = one_hot([0, 1, 1])
        primitive = builder.primitive(triangle(), joints, weights)
        binds     = builder.identity_binds(2)
        path = builder.write(
            self.path("rootnode.glb"),
            [Mesh(primitives=[primitive], name="body")],
            [
                Node(name="RootNode", children=[2]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="Neck"),
            ],
            [Skin(joints=[0, 2], inverseBindMatrices=binds)],
            scene_nodes=[0, 1],
        )

        with self.assertRaises(ValueError):
            load_glb(path)

    def test_the_skin_carries_the_mesh_name(self):
        self.assertEqual(load_glb(self.decoy_file())[0].name, "body")


class TestPrimitives(GlbSkinTest):
    def multi(self, name, modes):
        builder    = Builder()
        primitives = []
        for index, mode in enumerate(modes):
            joints, weights = one_hot([0, 1, 0])
            primitives.append(
                builder.primitive(triangle(index * 10.0), joints, weights, mode=mode)
            )

        binds = builder.identity_binds(2)
        return builder.write(
            self.path(name),
            [Mesh(primitives=primitives, name="body")],
            [
                Node(name="root", children=[2]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="spine"),
            ],
            [Skin(joints=[0, 2], inverseBindMatrices=binds)],
            scene_nodes=[0, 1],
        )

    def test_one_entry_per_primitive(self):
        data = load_glb(self.multi("two.glb", [TRIANGLES, TRIANGLES]))

        self.assertEqual(len(data), 2)
        for skin in data:
            self.assertEqual(skin.influences, ["root", "spine"])

    def test_a_non_triangle_primitive_still_yields_weights(self):
        """trimesh drops these, so MeshData.load_glb would return fewer entries.
        This reader is parallel to load_model, which keeps them.
        """
        data = load_glb(self.multi("fan.glb", [TRIANGLES, TRIANGLE_FAN, TRIANGLES]))

        self.assertEqual(len(data), 3)
        self.assertTrue(all(x is not None for x in data))

    @unittest.skipIf(trimesh is None, "trimesh is not installed")
    def test_the_reader_diverges_from_meshdata_on_a_non_triangle_file(self):
        """pins the documented caveat rather than pretending the two always pair"""
        from cgmath.geometry import MeshData

        path = self.multi("fan_pairing.glb", [TRIANGLES, TRIANGLE_FAN, TRIANGLES])

        self.assertEqual(len(load_glb(path)), 3)
        self.assertEqual(len(MeshData.load_glb(path)), 2)

    def test_an_unskinned_primitive_becomes_a_hole(self):
        builder = Builder()
        joints, weights = one_hot([0, 1, 0])
        skinned = builder.primitive(triangle(), joints, weights)
        bare    = builder.primitive(triangle(10.0))
        binds   = builder.identity_binds(2)

        path = builder.write(
            self.path("hole.glb"),
            [
                Mesh(primitives=[skinned], name="body"),
                Mesh(primitives=[bare], name="prop"),
            ],
            [
                Node(name="root", children=[3]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="prop_node", mesh=1),
                Node(name="spine"),
            ],
            [Skin(joints=[0, 3], inverseBindMatrices=binds)],
            scene_nodes=[0, 1, 2],
        )

        data = load_glb(path)

        self.assertEqual(len(data), 2)
        self.assertIsNotNone(data[0])
        self.assertIsNone(data[1])

    def test_joints_0_at_accessor_index_zero_is_not_dropped(self):
        """regression for the truthiness guard in glb.py: accessor 0 is falsy,
        and the old `if primitive.attributes.JOINTS_0:` skipped joints AND
        weights without raising
        """
        builder = Builder()
        joints  = np.zeros((3, 4), dtype=np.uint16)
        joints[:, 0] = [0, 1, 0]

        # burn accessor 0 on JOINTS_0 before anything else is added
        attributes = {
            "JOINTS_0": builder.add(joints, kind="VEC4", ctype=USHORT),
            "POSITION": builder.add(triangle()),
        }
        self.assertEqual(attributes["JOINTS_0"], 0)

        weights = np.zeros((3, 4), dtype=np.float32)
        weights[:, 0] = 1.0
        attributes["WEIGHTS_0"] = builder.add(weights, kind="VEC4", ctype=FLOAT)

        primitive = Primitive(
            attributes=Attributes(**attributes),
            indices=builder.add(
                np.arange(3, dtype=np.uint16), kind="SCALAR", ctype=USHORT
            ),
            mode=TRIANGLES,
        )
        binds = builder.identity_binds(2)
        path = builder.write(
            self.path("accessor_zero.glb"),
            [Mesh(primitives=[primitive], name="body")],
            [
                Node(name="root", children=[2]),
                Node(name="geo", mesh=0, skin=0),
                Node(name="spine"),
            ],
            [Skin(joints=[0, 2], inverseBindMatrices=binds)],
            scene_nodes=[0, 1],
        )

        data = load_glb(path)

        self.assertIsNotNone(data[0])
        self.assertEqual(data[0].influences, ["root", "spine"])


class TestClassMethod(GlbSkinTest):
    def test_first_skinned_primitive_by_default(self):
        self.assertEqual(
            SkinData.load_glb(self.decoy_file()).influences,
            ["root", "spine", "head"],
        )

    def test_selection_by_name(self):
        self.assertEqual(SkinData.load_glb(self.decoy_file(), name="body").name, "body")

    def test_an_unknown_name_lists_what_is_there(self):
        with self.assertRaises(ValueError) as caught:
            SkinData.load_glb(self.decoy_file(), name="nope")

        self.assertIn("available", str(caught.exception))
        self.assertIn("body", str(caught.exception))

    def test_a_file_with_no_skin_raises(self):
        builder   = Builder()
        primitive = builder.primitive(triangle())
        path = builder.write(
            self.path("bare.glb"),
            [Mesh(primitives=[primitive], name="prop")],
            [Node(name="prop_node", mesh=0)],
            [],
        )

        with self.assertRaises(RuntimeError):
            SkinData.load_glb(path)


class TestDeformRoundTrip(GlbSkinTest):
    def test_the_loaded_pieces_bind_and_deform(self):
        """the reason this reader exists -- it has to feed SkinDeformData"""
        from cgmath.geometry.deform import SkinDeformData
        from cgmath.hierarchy import HierarchyData

        path     = self.decoy_file()
        skin     = load_glb(path)[0]
        rig      = HierarchyData.load_glb(path)
        points   = triangle().astype(np.float64)

        deformer = SkinDeformData(mesh=points, skin=skin, bind_rig=rig)
        deformer.bind()

        self.assertTrue(deformer.valid)

        # the bind pose reproduces the rest points
        np.testing.assert_allclose(deformer.apply(rig), points, atol=1e-9)


if __name__ == "__main__":
    unittest.main()