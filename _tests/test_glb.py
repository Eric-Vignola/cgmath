import json
import os
import struct
import tempfile
import unittest

import numpy as np
from cgmath.formats.glb import (
    get_dtype_cnt,
    load_accessor_data,
    load_gltf,
    pygltflib,
    repair_null_skin_joints,
)

if pygltflib is not None:
    from pygltflib import (
        Accessor,
        Buffer,
        BufferView,
        FLOAT,
        GLTF2,
        SHORT,
        UNSIGNED_BYTE,
    )


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestLoadAccessorData(unittest.TestCase):
    """a buffer view can carry more than one accessor, and an accessor can
    start part way into it, be shorter than it, or be interleaved with its
    neighbours
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, blob, views, accessors):
        """write a glb holding raw bytes and read it back through GLTF2"""
        gltf             = GLTF2()
        gltf.buffers     = [Buffer(byteLength=len(blob))]
        gltf.bufferViews = [BufferView(buffer=0, **view) for view in views]
        gltf.accessors   = [Accessor(**accessor) for accessor in accessors]
        gltf.set_binary_blob(blob)

        path = os.path.join(self.tmp.name, "fixture.glb")
        gltf.save(path)
        return GLTF2().load(path)

    def testTightlyPackedIsUnchanged(self):
        """the common case has to stay byte for byte what it always was"""
        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        gltf = self.build(
            values.tobytes(),
            [{"byteOffset": 0, "byteLength": 36}],
            [{"bufferView": 0, "componentType": FLOAT, "count": 3, "type": "VEC3"}],
        )

        got = load_accessor_data(gltf, gltf.accessors[0])

        self.assertEqual(got.shape, (3, 3))
        self.assertTrue(np.array_equal(got, values))

    def testTwoAccessorsSharingOneView(self):
        """the times used to come back with the values appended to them"""
        times  = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        blob   = times.tobytes() + values.tobytes()

        gltf = self.build(
            blob,
            [{"byteOffset": 0, "byteLength": len(blob)}],
            [
                {
                    "bufferView":    0,
                    "componentType": FLOAT,
                    "count":         3,
                    "type":          "SCALAR",
                },
                {
                    "bufferView":    0,
                    "byteOffset":    12,
                    "componentType": FLOAT,
                    "count":         3,
                    "type":          "VEC3",
                },
            ],
        )

        self.assertTrue(
            np.array_equal(load_accessor_data(gltf, gltf.accessors[0]), times)
        )
        self.assertTrue(
            np.array_equal(load_accessor_data(gltf, gltf.accessors[1]), values)
        )

    def testAccessorShorterThanItsView(self):
        values = np.arange(12, dtype=np.float32).reshape(4, 3)
        gltf = self.build(
            values.tobytes(),
            [{"byteOffset": 0, "byteLength": 48}],
            [{"bufferView": 0, "componentType": FLOAT, "count": 2, "type": "VEC3"}],
        )

        got = load_accessor_data(gltf, gltf.accessors[0])

        self.assertEqual(got.shape, (2, 3))
        self.assertTrue(np.array_equal(got, values[:2]))

    def testInterleavedAccessors(self):
        """position and uv packed into one 20 byte stride"""
        positions = np.arange(9, dtype=np.float32).reshape(3, 3)
        uvs       = (np.arange(6, dtype=np.float32) * 0.25).reshape(3, 2)
        blob      = b"".join(positions[i].tobytes() + uvs[i].tobytes() for i in range(3))

        gltf = self.build(
            blob,
            [{"byteOffset": 0, "byteLength": len(blob), "byteStride": 20}],
            [
                {"bufferView": 0, "componentType": FLOAT, "count": 3, "type": "VEC3"},
                {
                    "bufferView":    0,
                    "byteOffset":    12,
                    "componentType": FLOAT,
                    "count":         3,
                    "type":          "VEC2",
                },
            ],
        )

        self.assertTrue(
            np.array_equal(load_accessor_data(gltf, gltf.accessors[0]), positions)
        )
        self.assertTrue(
            np.array_equal(load_accessor_data(gltf, gltf.accessors[1]), uvs)
        )

    def testNormalizedShortIsDequantized(self):
        """a quantized rotation used to arrive scaled by 32767"""
        quat = np.array([[0, 0, 0, 32767]], dtype=np.int16)
        gltf = self.build(
            quat.tobytes(),
            [{"byteOffset": 0, "byteLength": 8}],
            [
                {
                    "bufferView":    0,
                    "componentType": SHORT,
                    "count":         1,
                    "type":          "VEC4",
                    "normalized":    True,
                }
            ],
        )

        got = load_accessor_data(gltf, gltf.accessors[0])

        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.allclose(got, [[0.0, 0.0, 0.0, 1.0]], atol=1e-6))

    def testNormalizedByteIsDequantized(self):
        weights = np.array([[255, 128, 64, 0]], dtype=np.uint8)
        gltf = self.build(
            weights.tobytes(),
            [{"byteOffset": 0, "byteLength": 4}],
            [
                {
                    "bufferView":    0,
                    "componentType": UNSIGNED_BYTE,
                    "count":         1,
                    "type":          "VEC4",
                    "normalized":    True,
                }
            ],
        )

        got = load_accessor_data(gltf, gltf.accessors[0])

        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.allclose(got, weights / 255.0, atol=1e-6))

    def testUnnormalizedIntegersAreLeftAlone(self):
        """joint indices are indices, not fractions"""
        joints = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        gltf = self.build(
            joints.tobytes(),
            [{"byteOffset": 0, "byteLength": 4}],
            [
                {
                    "bufferView":    0,
                    "componentType": UNSIGNED_BYTE,
                    "count":         1,
                    "type":          "VEC4",
                }
            ],
        )

        got = load_accessor_data(gltf, gltf.accessors[0])

        self.assertEqual(got.dtype, np.uint8)
        self.assertTrue(np.array_equal(got, joints))

    def testMatrixAccessorKnowsItsWidth(self):
        """without a MAT4 count the honoured accessor.count would read nothing"""
        matrices = np.arange(32, dtype=np.float32).reshape(2, 16)
        gltf = self.build(
            matrices.tobytes(),
            [{"byteOffset": 0, "byteLength": 128}],
            [{"bufferView": 0, "componentType": FLOAT, "count": 2, "type": "MAT4"}],
        )

        self.assertEqual(get_dtype_cnt(gltf.accessors[0]), (np.float32, 16))

        got = load_accessor_data(gltf, gltf.accessors[0])

        self.assertEqual(got.shape, (2, 16))
        self.assertTrue(np.array_equal(got, matrices))
        # load_model reshapes inverse bind matrices by hand, still a no-op
        self.assertTrue(np.array_equal(got.reshape((-1, 16)), matrices))


@unittest.skipIf(pygltflib is None, "pygltflib is not installed")
class TestRepairNullSkinJoints(unittest.TestCase):
    """three.js writes skins[].joints full of nulls when it could not resolve
    the skeleton, and glTF says those have to be integers, so the parser
    refuses a whole file over a section nothing here reads back
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, document, blob=b"\x00" * 16):
        """packs a document into a real glb container"""
        raw = json.dumps(document).encode("utf-8")
        raw  += b" " * (-len(raw) % 4)
        blob += b"\x00" * (-len(blob) % 4)

        chunks = struct.pack("<II", len(raw), 0x4E4F534A) + raw
        chunks += struct.pack("<II", len(blob), 0x004E4942) + blob

        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
        return path

    def document(self, joints, **extra):
        document = {
            "asset":  {"version": "2.0"},
            "scene":  0,
            "scenes": [{"nodes": [0]}],
            "nodes":  [{"name": "root", "children": [1]}, {"name": "child"}],
            "skins":  [{"joints": joints}],
        }
        document.update(extra)
        return document

    def test_a_file_with_no_nulls_is_left_alone(self):
        path = self.write("clean.glb", self.document([0, 1]))

        with open(path, "rb") as handle:
            self.assertIsNone(repair_null_skin_joints(handle.read()))

    def test_all_null_joints_become_an_empty_array(self):
        path = self.write("allnull.glb", self.document([None, None, None]))

        with open(path, "rb") as handle:
            original = handle.read()
        repaired = repair_null_skin_joints(original)

        # length preserving, so the chunk sizes in the header stay correct
        self.assertEqual(len(repaired), len(original))

        gltf = GLTF2.load_from_bytes(repaired)
        self.assertEqual(gltf.skins[0].joints, [])

    def test_a_null_among_real_indexes_leaves_the_rest(self):
        """blanking a null has to take a comma with it or the array is ragged"""
        path = self.write("some.glb", self.document([1, None, 2, None]))

        with open(path, "rb") as handle:
            repaired = repair_null_skin_joints(handle.read())

        gltf = GLTF2.load_from_bytes(repaired)
        self.assertEqual(gltf.skins[0].joints, [1, 2])

    def test_load_gltf_reads_a_file_with_null_joints(self):
        """the repair has to land however the decoder reacts to the null

        dataclasses-json 0.6 and up raise int(None) out of the decoder while
        0.5 passes the null through into the list, so asserting that
        GLTF2().load refuses would pin this to a version rather than to the
        behaviour being bought.
        """
        path = self.write("broken.glb", self.document([None, None]))

        gltf = load_gltf(path)

        self.assertEqual(len(gltf.nodes),        2)
        self.assertEqual(gltf.nodes[0].children, [1])
        self.assertEqual(gltf.skins[0].joints,   [])

    def test_the_binary_chunk_survives(self):
        """the repair must not move the bin chunk, or every accessor shifts"""
        blob = bytes(range(64))
        path = self.write("withbin.glb", self.document([None, None]), blob=blob)

        gltf = load_gltf(path)

        self.assertEqual(gltf.binary_blob(), blob)

    def test_a_null_outside_skins_is_not_repaired(self):
        """dropping one of these changes the hierarchy, so it has to fail"""
        document                         = self.document([0, 1])
        document["nodes"][0]["children"] = [1, None]
        path                             = self.write("badchild.glb", document)

        with open(path, "rb") as handle:
            self.assertIsNone(repair_null_skin_joints(handle.read()))

        with self.assertRaises(TypeError):
            load_gltf(path)

    def test_a_null_in_both_places_still_fails(self):
        """repairing the skins does not excuse the one in the hierarchy"""
        document                         = self.document([None, None])
        document["nodes"][0]["children"] = [1, None]
        path                             = self.write("both.glb", document)

        # the skins half IS repairable, so this only fails if the result is
        # looked at again after the repair rather than trusted
        with open(path, "rb") as handle:
            self.assertIsNotNone(repair_null_skin_joints(handle.read()))

        with self.assertRaises(TypeError):
            load_gltf(path)