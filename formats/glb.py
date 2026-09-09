import io
import os
import re
import zipfile
from collections import namedtuple
from contextlib import suppress
from pathlib import Path

import numpy as np

try:
    import pygltflib  # used to raise an error if pygltflib is not installed
    from pygltflib import (
        Accessor,
        Buffer,
        BufferView,
        BYTE,
        FLOAT,
        GLTF2,
        SCALAR,
        SHORT,
        UNSIGNED_BYTE,
        UNSIGNED_INT,
        UNSIGNED_SHORT,
    )
except ImportError:
    pygltflib = None

from cgmath.geometry import MeshData, MeshList

# trimesh
try:
    import trimesh
except ImportError:
    trimesh = None

Model = namedtuple("Model", "name nodes ordered_node_indexes meshes animations skins")
Mesh  = namedtuple("Mesh", "name primitives")
Primitive = namedtuple(
    "Primitive", "name material triangles vertices normals uvs joints weights"
)
AnimationSampler = namedtuple(
    "AnimationSampler", "interpolation keyframe_times keyframe_values"
)
AnimationChannel = namedtuple("AnimationChannel", "sampler node path")
Animation        = namedtuple("Animation",        "samplers channels, duration")
Skin             = namedtuple("Skin",             "joints inverse_bind_matrices")


def order_nodes_root_first(nodes):
    """
    Returns the nodes sorted so that the parents come first. This helps make transforming bone chain hierarchies trivial.
    """
    for node in nodes:
        node.parent_index = -1
    for i, node in enumerate(nodes):
        for child_node_index in node.children:
            nodes[child_node_index].parent_index = i
    ordered_parent_indexes = {}
    for i in range(len(nodes)):

        def add_node(j):
            if j in ordered_parent_indexes:
                return
            parent_index = nodes[j].parent_index
            if parent_index >= 0:
                add_node(parent_index)
            ordered_parent_indexes[j] = 1

        add_node(i)
    return list(ordered_parent_indexes)


def get_dtype_cnt(accessor):
    # tables rather than branches: the component types alone are enough to
    # push this past the complexity the linter allows
    if pygltflib is None:
        raise RuntimeError("pygltflib is not installed")

    dtypes = {
        BYTE:           np.int8,
        UNSIGNED_BYTE:  np.uint8,
        SHORT:          np.int16,
        UNSIGNED_SHORT: np.uint16,
        UNSIGNED_INT:   np.uint32,
        FLOAT:          np.float32,
    }
    counts = {"MAT4": 16, "VEC4": 4, "VEC3": 3, "VEC2": 2, "SCALAR": 1}

    return dtypes.get(accessor.componentType), counts.get(accessor.type, 0)


def load_accessor_data(gltf, accessor):
    if pygltflib is None:
        raise RuntimeError("pygltflib is not installed")

    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer      = gltf.buffers[buffer_view.buffer]
    data        = gltf.get_data_from_buffer_uri(buffer.uri)
    dtype, cnt = get_dtype_cnt(accessor)
    item_size = np.dtype(dtype).itemsize * cnt
    stride    = buffer_view.byteStride or item_size
    start     = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

    # a buffer view can hold several accessors, so the view alone does not
    # say where this one starts or how far it runs
    if stride == item_size:
        elements = np.frombuffer(
            data, dtype=dtype, count=accessor.count * cnt, offset=start
        )
    else:
        raw = np.frombuffer(
            data,
            dtype  = np.uint8,
            count  = (accessor.count - 1) * stride + item_size,
            offset = start,
        )
        rows     = np.arange(accessor.count)[:, None] * stride + np.arange(item_size)
        elements = np.frombuffer(raw[rows].tobytes(), dtype=dtype)

    if cnt > 1:
        elements = np.reshape(elements, (-1, cnt))

    if accessor.normalized and np.issubdtype(dtype, np.integer):
        elements = np.maximum(
            elements / np.float32(np.iinfo(dtype).max), np.float32(-1.0)
        ).astype(np.float32)

    return elements


def _blank_null_joints(match):
    """replaces every null in one joints array with spaces of the same width

    Each null takes a neighbouring comma with it so the array stays
    well formed: ``[1,null,2]`` becomes ``[1,     2]``, which reads back as
    two entries rather than three with a hole.
    """
    return re.sub(
        rb",[ \t\r\n]*null|null[ \t\r\n]*,|null",
        lambda hit: b" " * len(hit.group(0)),
        match.group(0),
    )


def repair_null_skin_joints(data):
    """blanks out null entries in a glb's ``skins[].joints``, or None

    Some exporters, three.js among them, write these as null when the
    skeleton was not resolvable, and glTF says they have to be integers, so
    the parser refuses the whole file over a section nothing here reads back.

    Only skins are repaired. A null in ``nodes[].children`` or
    ``scenes[].nodes`` changes the hierarchy, and dropping one quietly would
    be a worse answer than refusing the file.

    The edit is length preserving, so the chunk sizes in the glb header stay
    correct and the binary chunk does not move.
    """
    if data[:4] != b"glTF":
        return None

    # only the json chunk, so a byte pattern cannot be hit inside the mesh data
    length = int.from_bytes(data[12:16], "little")
    start, stop = 20, 20 + length
    if stop > len(data):
        return None

    body = data[start:stop]

    # "joints" is lower case only on a skin; mesh attributes spell it JOINTS_0
    repaired = re.sub(
        rb'"joints"[ \t\r\n]*:[ \t\r\n]*\[[^]]*\]', _blank_null_joints, body
    )
    if repaired == body:
        return None

    return data[:start] + repaired + data[stop:]


def _null_index_locations(gltf):
    """names every array that still holds a null where an index belongs

    Only ever finds anything on a dataclasses-json old enough to hand a null
    straight back through a ``List[int]``; the newer ones raise out of the
    decoder long before this. Both have to be catered for, so the null is
    looked for rather than inferred from whether something was thrown.
    """
    found = []
    for index, skin in enumerate(gltf.skins or []):
        if any(x is None for x in skin.joints or []):
            found.append(f"skins[{index}].joints")
    for index, node in enumerate(gltf.nodes or []):
        if any(x is None for x in node.children or []):
            found.append(f"nodes[{index}].children")
    for index, scene in enumerate(gltf.scenes or []):
        if any(x is None for x in scene.nodes or []):
            found.append(f"scenes[{index}].nodes")
    return found


def load_gltf(fname):
    """reads a gltf or glb, repairing null skin joints if that is what stops it

    A null where an index belongs reaches the two dataclasses-json
    generations differently: 0.6 and up raise ``int(None)`` from deep inside
    the json decoder, while 0.5 passes the null through into the list. Taking
    the exception as the signal would leave the repair dead on the older one,
    so a load that SUCCEEDS is inspected too.
    """
    if pygltflib is None:
        raise RuntimeError("pygltflib is not installed")

    refused = None
    try:
        gltf = GLTF2().load(fname)
    except TypeError as exc:
        gltf, refused = None, exc

    if gltf is not None and not _null_index_locations(gltf):
        return gltf

    with open(fname, "rb") as handle:
        repaired = repair_null_skin_joints(handle.read())

    if repaired is not None:
        patched = GLTF2.load_from_bytes(repaired)

        # load_binary sets these after the fact and external buffers need them
        patched._path = Path(fname).parent
        patched._name = Path(fname).name

        # checked again: repairing the skins does not excuse a null left
        # somewhere the repair deliberately will not touch
        if not _null_index_locations(patched):
            return patched
        gltf = patched

    # nothing repairable, so hand back whatever the decoder said if it spoke
    # at all -- it may have refused for a reason that has nothing to do with
    # nulls, and its own message is the better one in that case
    if refused is not None:
        raise refused

    raise TypeError(
        f"{fname}: null where a node index belongs, in "
        f"{', '.join(_null_index_locations(gltf))} -- dropping one would "
        f"change the hierarchy"
    )


def load_model(fname):
    if pygltflib is None:
        raise RuntimeError("pygltflib is not installed")

    gltf   = load_gltf(fname)

    meshes = []
    for mesh in gltf.meshes:
        primitives = []
        for primitive in mesh.primitives:
            triangles = vertices = normals = uvs = joints = weights = None

            with suppress(TypeError):
                triangles = load_accessor_data(gltf, gltf.accessors[primitive.indices])
            with suppress(TypeError):
                vertices = load_accessor_data(
                    gltf, gltf.accessors[primitive.attributes.POSITION]
                )
            with suppress(TypeError):
                normals = load_accessor_data(
                    gltf, gltf.accessors[primitive.attributes.NORMAL]
                )
            with suppress(TypeError):
                uvs = load_accessor_data(
                    gltf, gltf.accessors[primitive.attributes.TEXCOORD_0]
                )

            if primitive.attributes.JOINTS_0 is not None:
                with suppress(TypeError):
                    joints = load_accessor_data(
                        gltf, gltf.accessors[primitive.attributes.JOINTS_0]
                    )
                with suppress(TypeError):
                    weights = load_accessor_data(
                        gltf, gltf.accessors[primitive.attributes.WEIGHTS_0]
                    )

            if joints is not None:
                joints = np.array(joints, dtype=np.uint16)

            primitives.append(
                Primitive(
                    mesh.name,
                    primitive.material,
                    triangles,
                    vertices,
                    normals,
                    uvs,
                    joints,
                    weights,
                )
            )
        meshes.append(Mesh(mesh.name, primitives))

    animations = []
    for anim in gltf.animations:
        samplers = []
        channels = []
        duration = 0
        for sampler in anim.samplers:
            keyframe_times  = load_accessor_data(gltf, gltf.accessors[sampler.input])
            keyframe_values = load_accessor_data(gltf, gltf.accessors[sampler.output])
            s               = AnimationSampler(sampler.interpolation, keyframe_times, keyframe_values)
            samplers.append(s)
            if keyframe_times[-1] > duration:
                duration = keyframe_times[-1]

        for chnl in anim.channels:
            c = AnimationChannel(chnl.sampler, chnl.target.node, chnl.target.path)
            channels.append(c)

        a = Animation(samplers, channels, duration)
        animations.append(a)

    skins = []
    for skin in gltf.skins:
        joints = skin.joints
        inverse_bind_matrices = load_accessor_data(
            gltf, gltf.accessors[skin.inverseBindMatrices]
        )
        inverse_bind_matrices = inverse_bind_matrices.reshape((-1, 16))
        assert inverse_bind_matrices.dtype == np.float32
        skins.append(Skin(joints, inverse_bind_matrices))

    ordered_node_indices = order_nodes_root_first(gltf.nodes)

    name                 = Path(fname).stem
    return Model(name, gltf.nodes, ordered_node_indices, meshes, animations, skins)


def get_binary(gltf, name="runtime_rig_retargeting.zip"):
    """gets a binary blob from the gltf"""
    if pygltflib is None:
        raise RuntimeError("pygltflib is not installed")

    for i, x in enumerate(gltf.accessors):
        if x.name == name:
            accessor    = gltf.accessors[i]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer      = gltf.buffers[buffer_view.buffer]
            data        = gltf.get_data_from_buffer_uri(buffer.uri)

            start       = buffer_view.byteOffset
            end         = buffer_view.byteOffset + buffer_view.byteLength
            return data[start:end]


def set_binary(gltf, binary_data, name="runtime_rig_retargeting.zip"):
    """sets/adds a binary blob to the gltf"""
    if pygltflib is None:
        raise RuntimeError("pygltflib is not installed")

    # Check if an accessor with the given name already exists
    for accessor in gltf.accessors:
        if accessor.name == name:
            # Update the existing accessor's buffer view with the new data
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer      = gltf.buffers[buffer_view.buffer]

            # Update the buffer's byte length if necessary
            existing_data = gltf.binary_blob()
            start         = buffer_view.byteOffset
            end           = start + buffer_view.byteLength
            updated_data  = existing_data[:start] + binary_data + existing_data[end:]

            # Update buffer view and accessor properties
            buffer_view.byteLength = len(binary_data)
            accessor.count         = len(binary_data)

            # Update the binary blob
            gltf.set_binary_blob(updated_data)

            return

    # If no existing accessor is found, append the new data
    if not gltf.buffers:
        gltf.buffers.append(Buffer(byteLength=0))

    buffer         = gltf.buffers[0]
    current_length = buffer.byteLength
    buffer.byteLength += len(binary_data)
    new_buffer_view = BufferView()
    new_buffer_view.buffer     = 0
    new_buffer_view.byteOffset = current_length
    new_buffer_view.byteLength = len(binary_data)
    gltf.bufferViews.append(new_buffer_view)

    new_accessor = Accessor()
    new_accessor.bufferView    = len(gltf.bufferViews) - 1
    new_accessor.byteOffset    = 0
    new_accessor.componentType = BYTE
    new_accessor.count         = len(binary_data)
    new_accessor.type          = SCALAR
    new_accessor.name          = name
    gltf.accessors.append(new_accessor)

    existing_data = gltf.binary_blob()
    updated_data  = existing_data + binary_data
    gltf.set_binary_blob(updated_data)


class GlbData:
    """A class to hold the data from a gltf file"""

    def __init__(self, fname: str):
        if pygltflib is None:
            raise RuntimeError("pygltflib is not installed")

        self.gltf       = None
        self._mesh_list = None
        self.load(fname)

    def load(self, fname: str):
        fname = os.path.expanduser(fname)
        self.gltf = GLTF2().load(fname)

        if trimesh is not None:
            scene     = trimesh.load(fname)
            mesh_list = []
            for name, data in scene.geometry.items():
                points  = np.array(data.vertices) * 100.0
                indices = np.array(data.faces)
                counts  = np.ones(indices.shape[0], dtype=int) * 3

                new = MeshData(
                    points=points, indices=indices.ravel(), counts=counts, name=name
                )
                mesh_list.append(new)

            self._mesh_list = MeshList(mesh_list)

    def save(self, fname: str):
        fname = os.path.expanduser(fname)
        self.gltf.save(fname)

    def get_binary(self, name="runtime_rig_retargeting.zip"):
        return get_binary(self.gltf, name=name)

    def set_binary(self, data, name="runtime_rig_retargeting.zip"):
        set_binary(self.gltf, data, name=name)

    def extract_payload(self, path: str, name="runtime_rig_retargeting.zip"):
        """extracts the zipped payload from the glb to disk"""

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        bytes_io = io.BytesIO(self.get_binary(name=name))
        with zipfile.ZipFile(bytes_io, "r") as zip_ref:
            zip_ref.extractall(path)

    def inject_payload(self, path: str, name="runtime_rig_retargeting.zip"):
        """injects the zipped payload from disk into the glb"""

        path       = os.path.expanduser(path)
        zip_stream = io.BytesIO()

        with zipfile.ZipFile(zip_stream, "w") as zip_file:
            for root, _, files in os.walk(path):
                rel_root = os.path.relpath(root, path)

                for file in files:
                    file_path = os.path.join(root, file)
                    rel_file_path = (
                        os.path.join(rel_root, file) if rel_root != "." else file
                    )
                    zip_file.write(file_path, rel_file_path)

        data = zip_stream.getvalue()
        self.set_binary(data, name=name)

    @property
    def animations(self):
        return self.gltf.animations

    @animations.setter
    def animations(self, animations: list):
        self.gltf.animations = animations

    def clear_animations(self):
        self.gltf.animations = []

    @property
    def mesh_list(self):
        return self._mesh_list