# `cgmath.formats` Cheatsheet

Copy-paste recipes for the FBX, GLB and USD wrappers. Every `python` block below
runs, in order, as one script — the fixtures are built inline because no
`.glb` / `.fbx` assets ship with this repo. For the overview see
[`README.md`](README.md); for the rest of the library see
[`../CHEATSHEET.md`](../CHEATSHEET.md).

`cgmath.formats` exports `load_fbx`, `save_fbx` and `FbxContents`; the glb
and usd helpers are imported from their submodules.

- [Setup](#setup)
- [`glb` — build a fixture](#glb--build-a-fixture)
- [`glb` — load_gltf](#glb--load_gltf)
- [`glb` — load_model and the Model namedtuple](#glb--load_model-and-the-model-namedtuple)
- [`glb` — Mesh and Primitive](#glb--mesh-and-primitive)
- [`glb` — Animation, AnimationSampler, AnimationChannel](#glb--animation-animationsampler-animationchannel)
- [`glb` — Skin](#glb--skin)
- [`glb` — order_nodes_root_first](#glb--order_nodes_root_first)
- [`glb` — get_dtype_cnt](#glb--get_dtype_cnt)
- [`glb` — load_accessor_data](#glb--load_accessor_data)
- [`glb` — repair_null_skin_joints](#glb--repair_null_skin_joints)
- [`glb` — get_binary / set_binary](#glb--get_binary--set_binary)
- [`glb` — GlbData](#glb--glbdata)
- [`fbx` — load_fbx / save_fbx](#fbx--load_fbx--save_fbx)
- [`usd.stage` — stages, caches, sublayers](#usdstage--stages-caches-sublayers)
- [`usd.prim` — search, duplicate, compose](#usdprim--search-duplicate-compose)
- [`fbx` — FbxExporter](#fbx--fbxexporter)
- [The class routes slice the same read](#the-class-routes-slice-the-same-read)
- [Gotchas](#gotchas)

---

## Setup

One scratch directory and numpy, reused by every block below.

```python
import os
import tempfile

import numpy as np

WORK = tempfile.mkdtemp(prefix="rl_math_formats_")
```

---

## `glb` — build a fixture

Needs `pygltflib`. A tiny accumulator that packs numpy arrays into one glTF
buffer and hands back accessor indices — the same shape the package tests
use.

```python
from pygltflib import Accessor, BufferView, FLOAT, UNSIGNED_SHORT


class Accum:
    """Packs arrays into one buffer; add() returns the accessor index."""

    def __init__(self):
        self.blob      = b""
        self.views     = []
        self.accessors = []

    def add(self, array, kind="VEC3", ctype=FLOAT):
        raw = array.tobytes()
        self.views.append(
            BufferView(buffer=0, byteOffset=len(self.blob), byteLength=len(raw))
        )
        self.blob += raw + b"\x00" * (-len(raw) % 4)   # 4-byte alignment
        self.accessors.append(
            Accessor(
                bufferView    = len(self.views) - 1,
                componentType = ctype,
                count         = array.shape[0],
                type          = kind,
            )
        )
        return len(self.accessors) - 1
```

A unit cube, skinned to two joints, with one translation animation.

```python
from pygltflib import (
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Attributes,
    Buffer,
    GLTF2,
    Material,
    Mesh,
    Node,
    Primitive,
    Scene,
    Skin,
)

CUBE_POINTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float32)

CUBE_TRIS = np.array([
    0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,
    0, 1, 5,  0, 5, 4,  1, 2, 6,  1, 6, 5,
    2, 3, 7,  2, 7, 6,  3, 0, 4,  3, 4, 7,
], dtype=np.uint16)

acc   = Accum()
i_pos = acc.add(CUBE_POINTS)
i_nrm = acc.add(np.tile([[0, 0, 1]], (8, 1)).astype(np.float32))
i_uv  = acc.add(np.zeros((8, 2), np.float32), kind="VEC2")

joints        = np.zeros((8, 4), np.uint16)
joints[:, 0]  = [0, 0, 0, 0, 1, 1, 1, 1]        # bottom face -> slot 0, top -> slot 1
weights       = np.zeros((8, 4), np.float32)
weights[:, 0] = 1.0
i_joints      = acc.add(joints, kind="VEC4", ctype=UNSIGNED_SHORT)
i_weights     = acc.add(weights, kind="VEC4")

i_tris   = acc.add(CUBE_TRIS, kind="SCALAR", ctype=UNSIGNED_SHORT)
i_binds  = acc.add(np.tile(np.eye(4, dtype=np.float32).ravel(), (2, 1)), kind="MAT4")
i_times  = acc.add(np.array([0.0, 0.5, 1.0], np.float32), kind="SCALAR")
i_values = acc.add(np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], np.float32))

gltf             = GLTF2()
gltf.buffers     = [Buffer(byteLength=len(acc.blob))]
gltf.bufferViews = acc.views
gltf.accessors   = acc.accessors
gltf.materials   = [Material(name="grey")]
gltf.meshes = [Mesh(name="cube", primitives=[Primitive(
    attributes=Attributes(
        POSITION=i_pos, NORMAL=i_nrm, TEXCOORD_0=i_uv,
        JOINTS_0=i_joints, WEIGHTS_0=i_weights,
    ),
    indices  = i_tris,
    material = 0,
)])]
gltf.nodes = [
    Node(name="root", children=[1]),
    Node(name="spine", children=[2]),
    Node(name="geo", mesh=0, skin=0),
]
gltf.skins = [Skin(joints=[0, 1], inverseBindMatrices=i_binds)]
gltf.animations = [Animation(
    name     = "bob",
    samplers = [AnimationSampler(input=i_times, output=i_values, interpolation="LINEAR")],
    channels=[AnimationChannel(
        sampler=0, target=AnimationChannelTarget(node=0, path="translation")
    )],
)]
gltf.scenes = [Scene(nodes=[0])]
gltf.scene  = 0
gltf.set_binary_blob(acc.blob)

CUBE_GLB = os.path.join(WORK, "cube.glb")
gltf.save(CUBE_GLB)
```

---

## `glb` — load_gltf

`load_gltf(fname)` is `GLTF2().load` plus one repair: a `.glb` whose
`skins[].joints` holds `null` is patched in place rather than refused.
Everything else raises exactly what the decoder raised.

```python
from cgmath.formats.glb import load_gltf

doc = load_gltf(CUBE_GLB)                    # -> pygltflib.GLTF2
print(len(doc.nodes), len(doc.meshes), len(doc.accessors))
print([n.name for n in doc.nodes])
```

---

## `glb` — load_model and the Model namedtuple

`load_model(fname)` decodes every accessor eagerly and returns plain numpy.

```python
from cgmath.formats.glb import load_model

model = load_model(CUBE_GLB)
print(model._fields)
print(model.name)                  # file stem, not the glTF scene name
print(model.ordered_node_indexes)  # parent-first order into model.nodes
```

| `Model` field | Type | Notes |
|---|---|---|
| `name` | `str` | `Path(fname).stem` |
| `nodes` | `list[pygltflib.Node]` | raw glTF nodes, each stamped with `parent_index` |
| `ordered_node_indexes` | `list[int]` | index order with parents before children |
| `meshes` | `list[Mesh]` | the `glb.Mesh` namedtuple, not pygltflib's |
| `animations` | `list[Animation]` | |
| `skins` | `list[Skin]` | |

Walk the hierarchy parent-first:

```python
for index in model.ordered_node_indexes:
    node   = model.nodes[index]
    parent = model.nodes[node.parent_index].name if node.parent_index >= 0 else None
    print(f"{node.name:8s} parent={parent}")
```

---

## `glb` — Mesh and Primitive

`Mesh(name, primitives)`. Every `Primitive` attribute is `None` when the
file did not author it.

```python
mesh = model.meshes[0]
print(mesh.name, len(mesh.primitives))

prim = mesh.primitives[0]
print(prim._fields)
print(prim.vertices.shape)                   # (8, 3)  float32 POSITION
print(prim.triangles.shape)                  # (36,)   flat index stream, NOT reshaped
print(prim.normals.shape)                    # (8, 3)
print(prim.uvs.shape)                        # (8, 2)  TEXCOORD_0 only
print(prim.joints.shape, prim.joints.dtype)  # (8, 4) uint16, always cast
print(prim.weights.shape)                    # (8, 4)
print(prim.material)                         # index into gltf.materials, or None
print(prim.name)                             # the MESH name, repeated onto each primitive
```

| `Primitive` field | Source | Missing → |
|---|---|---|
| `name` | parent mesh's name | — |
| `material` | `primitive.material` | `None` |
| `triangles` | `primitive.indices` | `None` |
| `vertices` | `POSITION` | `None` |
| `normals` | `NORMAL` | `None` |
| `uvs` | `TEXCOORD_0` | `None` |
| `joints` | `JOINTS_0`, cast to `uint16` | `None` |
| `weights` | `WEIGHTS_0` | `None` (also `None` when `JOINTS_0` is absent) |

Triangles come back flat — reshape yourself:

```python
faces = prim.triangles.reshape(-1, 3)
print(faces.shape, faces[0])
```

---

## `glb` — Animation, AnimationSampler, AnimationChannel

```python
anim = model.animations[0]
print(anim._fields)   # ('samplers', 'channels', 'duration', 'name')
print(anim.duration)  # max last keyframe time across samplers
print(anim.name)      # the animation's name, None when the file gives none

sampler = anim.samplers[0]
print(sampler.interpolation)          # 'LINEAR' | 'STEP' | 'CUBICSPLINE'
print(sampler.keyframe_times)         # (K,) seconds
print(sampler.keyframe_values.shape)  # (K, 3) here; (K, 4) for rotation

channel = anim.channels[0]
print(channel.sampler, channel.node, channel.path)   # 0 0 translation
```

`channel.node` is an index into `model.nodes`, `channel.sampler` an index
into `anim.samplers`. Resolve a channel to a named node:

```python
for channel in anim.channels:
    sampler = anim.samplers[channel.sampler]
    target  = model.nodes[channel.node].name
    print(f"{target}.{channel.path}: {len(sampler.keyframe_times)} keys")
```

---

## `glb` — Skin

`Skin(joints, inverse_bind_matrices)`. `joints` is a list of **node**
indices; a vertex's `JOINTS_0` slot indexes *this list*, not the nodes.

```python
skin = model.skins[0]
print(skin.joints)                       # [0, 1] -> node indices
print(skin.inverse_bind_matrices.shape)  # (2, 16) float32, always reshaped
```

Resolve a vertex's influences correctly:

```python
vertex = 4
for slot, weight in zip(prim.joints[vertex], prim.weights[vertex]):
    if weight > 0:
        print(model.nodes[skin.joints[slot]].name, float(weight))
```

---

## `glb` — order_nodes_root_first

glTF does not promise parent-first node order. This returns an index order
that is, and stamps `parent_index` on every node as a side effect.

```python
from pygltflib import Node
from cgmath.formats.glb import order_nodes_root_first

nodes = [Node(name="hand"), Node(name="root", children=[2]), Node(name="arm", children=[0])]
order = order_nodes_root_first(nodes)
print(order)                                      # [1, 2, 0]
print([nodes[i].name for i in order])             # ['root', 'arm', 'hand']
print([(n.name, n.parent_index) for n in nodes])  # -1 marks a root
```

---

## `glb` — get_dtype_cnt

Maps a glTF accessor to `(numpy dtype, components per element)`.

```python
from cgmath.formats.glb import get_dtype_cnt

print(get_dtype_cnt(doc.accessors[i_pos]))    # (float32, 3)
print(get_dtype_cnt(doc.accessors[i_binds]))  # (float32, 16)
print(get_dtype_cnt(doc.accessors[i_tris]))   # (uint16, 1)
```

| `accessor.type` | count | | `componentType` | dtype |
|---|---|---|---|---|
| `MAT4` | 16 | | `BYTE` | `int8` |
| `VEC4` | 4 | | `UNSIGNED_BYTE` | `uint8` |
| `VEC3` | 3 | | `SHORT` | `int16` |
| `VEC2` | 2 | | `UNSIGNED_SHORT` | `uint16` |
| `SCALAR` | 1 | | `UNSIGNED_INT` | `uint32` |
| anything else | 0 | | `FLOAT` | `float32` |

An unknown `componentType` yields `None` for the dtype, which blows up
downstream — check it if you accept arbitrary files.

---

## `glb` — load_accessor_data

Decode one accessor to numpy. Honours `byteOffset` on both the view and the
accessor, honours `byteStride` for interleaved data, and de-quantizes
`normalized` integers.

```python
from cgmath.formats.glb import load_accessor_data

points = load_accessor_data(doc, doc.accessors[i_pos])
print(points.shape, points.dtype)                            # (8, 3) float32
print(load_accessor_data(doc, doc.accessors[i_tris]).shape)  # (36,) — SCALAR stays 1-D
```

Two accessors sharing one buffer view, the second offset into it:

```python
from pygltflib import Buffer

times  = np.array([0.0, 0.5, 1.0], np.float32)
values = np.arange(9, dtype=np.float32).reshape(3, 3)
blob   = times.tobytes() + values.tobytes()

shared             = GLTF2()
shared.buffers     = [Buffer(byteLength=len(blob))]
shared.bufferViews = [BufferView(buffer=0, byteOffset=0, byteLength=len(blob))]
shared.accessors = [
    Accessor(bufferView=0, componentType=FLOAT, count=3, type="SCALAR"),
    Accessor(bufferView=0, byteOffset=12, componentType=FLOAT, count=3, type="VEC3"),
]
shared.set_binary_blob(blob)
shared_path = os.path.join(WORK, "shared.glb")
shared.save(shared_path)

shared = GLTF2().load(shared_path)
print(load_accessor_data(shared, shared.accessors[0]))        # the times
print(load_accessor_data(shared, shared.accessors[1]).shape)  # (3, 3)
```

A `normalized` `SHORT` quaternion comes back as `float32` in `[-1, 1]`:

```python
from pygltflib import SHORT

quat           = np.array([[0, 0, 0, 32767]], dtype=np.int16)
qg             = GLTF2()
qg.buffers     = [Buffer(byteLength=8)]
qg.bufferViews = [BufferView(buffer=0, byteOffset=0, byteLength=8)]
qg.accessors = [Accessor(
    bufferView=0, componentType=SHORT, count=1, type="VEC4", normalized=True
)]
qg.set_binary_blob(quat.tobytes())
qpath = os.path.join(WORK, "quat.glb")
qg.save(qpath)

decoded = load_accessor_data(GLTF2().load(qpath), GLTF2().load(qpath).accessors[0])
print(decoded.dtype, decoded)                   # float32 [[0. 0. 0. 1.]]
```

Un-normalized integers are left alone — joint indices are indices, not
fractions.

---

## `glb` — repair_null_skin_joints

Some exporters (three.js among them) write `skins[].joints` full of `null`
when the skeleton did not resolve. glTF says those must be integers, so a
strict parser refuses the whole file. This blanks the nulls, length
preserving, so the chunk header stays valid and the binary chunk does not
move. Returns `None` when there is nothing to repair.

```python
import json
import struct

from cgmath.formats.glb import repair_null_skin_joints


def write_glb(path, document, blob=b"\x00" * 16):
    raw = json.dumps(document).encode("utf-8")
    raw  += b" " * (-len(raw) % 4)
    blob += b"\x00" * (-len(blob) % 4)
    chunks = struct.pack("<II", len(raw), 0x4E4F534A) + raw
    chunks += struct.pack("<II", len(blob), 0x004E4942) + blob
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
    return path


broken = write_glb(os.path.join(WORK, "nulljoints.glb"), {
    "asset":  {"version": "2.0"},
    "scene":  0,
    "scenes": [{"nodes": [0]}],
    "nodes":  [{"name": "root", "children": [1]}, {"name": "child"}],
    "skins":  [{"joints": [None, None]}],
})

raw      = open(broken, "rb").read()
repaired = repair_null_skin_joints(raw)
print(repaired is not None, len(repaired) == len(raw))   # True True
```

`load_gltf` calls it for you, so this is usually invisible:

```python
recovered = load_gltf(broken)
print(recovered.skins[0].joints)  # [] — the nulls are gone
print(len(recovered.nodes))       # 2 — the hierarchy survived
```

A clean file returns `None`, and a `null` outside `skins` is **not**
repaired — dropping one would silently change the hierarchy, so `load_gltf`
raises instead.

```python
print(repair_null_skin_joints(open(CUBE_GLB, "rb").read()))   # None

bad_child = write_glb(os.path.join(WORK, "badchild.glb"), {
    "asset":  {"version": "2.0"},
    "scene":  0,
    "scenes": [{"nodes": [0]}],
    "nodes":  [{"name": "root", "children": [1, None]}, {"name": "child"}],
    "skins":  [{"joints": [0, 1]}],
})
try:
    load_gltf(bad_child)
except TypeError as error:
    print("refused:", str(error)[:40])
```

---

## `glb` — get_binary / set_binary

Stash an arbitrary blob inside a `.glb` behind a named accessor. Used to
ship a rig-retargeting zip alongside the geometry; the name defaults to
`"runtime_rig_retargeting.zip"`.

```python
from cgmath.formats.glb import get_binary, set_binary

doc2 = GLTF2().load(CUBE_GLB)
print(get_binary(doc2))                       # None — nothing stashed yet

set_binary(doc2, b"hello world")              # default name
print(get_binary(doc2))                       # b'hello world'

set_binary(doc2, b"notes", name="notes.txt")  # any number of named blobs
print(get_binary(doc2, name="notes.txt"))
```

It survives a save/load round trip:

```python
stashed = os.path.join(WORK, "stashed.glb")
doc2.save(stashed)
reloaded = GLTF2().load(stashed)
print(get_binary(reloaded), get_binary(reloaded, name="notes.txt"))
```

Calling `set_binary` a second time under the same name overwrites in place:

```python
set_binary(doc2, b"replaced", name="notes.txt")
print(get_binary(doc2, name="notes.txt"))
```

`get_binary` returns `None` for an unknown name — it does not raise.

---

## `glb` — GlbData

The object wrapper. Loads on construction, a gltf or a glb by its first
bytes under any name, as the glb readers do; `pygltflib` is required,
`trimesh` is optional.

```python
from cgmath.formats.glb import GlbData

glb = GlbData(CUBE_GLB)          # ~ is expanded
print(type(glb.gltf).__name__)   # GLTF2 — the raw document is always reachable
```

| Member | Does |
|---|---|
| `.gltf` | the underlying `pygltflib.GLTF2` |
| `.load(fname)` | re-load in place |
| `.save(fname)` | write out |
| `.get_binary(name=...)` | as the free function, on `self.gltf` |
| `.set_binary(data, name=...)` | as the free function |
| `.extract_payload(path, name=...)` | unzip the named blob to a directory |
| `.inject_payload(path, name=...)` | zip a directory into the named blob |
| `.animations` | get/set `gltf.animations` |
| `.clear_animations()` | set it to `[]` |
| `.mesh_list` | `MeshList` via trimesh, or `None` (read-only) |

Animations:

```python
print([a.name for a in glb.animations])
glb.clear_animations()
print(glb.animations)                       # []
glb.animations = GLTF2().load(CUBE_GLB).animations    # settable
print([a.name for a in glb.animations])
```

Meshes, if `trimesh` is installed. Points arrive **scaled ×100** (metres to
centimetres) and every face is a triangle.

```python
if glb.mesh_list is not None:
    for mesh in glb.mesh_list:
        print(mesh.name, mesh.points.shape, mesh.counts[:3], mesh.points.max())
    print(glb.mesh_list["cube"].name)        # MeshList indexes by name too
```

Payload round trip — a directory in, a directory out:

```python
payload_src = os.path.join(WORK, "payload_src")
os.makedirs(payload_src, exist_ok=True)
with open(os.path.join(payload_src, "rig.json"), "w") as handle:
    handle.write('{"joints": 2}')

glb.inject_payload(payload_src)
payload_dst = os.path.join(WORK, "payload_dst")
glb.extract_payload(payload_dst)        # creates the dir if missing
print(sorted(os.listdir(payload_dst)))  # ['rig.json']

glb.save(os.path.join(WORK, "glbdata_out.glb"))
```

---

## `fbx` — load_fbx / save_fbx

Needs the Autodesk FBX Python SDK. Without it `import cgmath.formats.fbx`
still works, with one `UserWarning`; the FBX readers and `save_fbx` raise
`ImportError` at call time, and an `FbxExporter`'s methods (all but `close`)
and any call on the module's placeholder `fbx` object raise `RuntimeError`,
with the warning's message. A module that imports as `fbx` but holds not
all of the SDK's five classes (`FbxManager`, `FbxScene`, `FbxImporter`,
`FbxExporter`, `FbxNode`), such as a folder named `fbx` on the path, counts
as no SDK.

One read and one write of everything cgmath has a class for: a hierarchy at
rest or its takes, the meshes on the nodes with their colours (one colour
set, `colorSet1`, one a vertex or one a face corner), their UV sets, skins,
materials and blend shape targets, the cameras and lights, the NURBS curves
and surfaces. Everything is in Maya's
Y-up and cm; a file's unit and axes are converted at the boundary. Each
shape names its node or its mesh by uuid, so build the nodes first. A
material on several meshes is one record a mesh, sharing one `uuid`, and
is written back as one shader.

```python
from cgmath.formats import FbxContents, load_fbx, save_fbx
from cgmath.geometry import CameraData, CameraList, LightData, LightList, MaterialData, MaterialList, MeshData, UVData
from cgmath.hierarchy import ClipData, HierarchyData, TransformData

rig = HierarchyData([TransformData("grp"), TransformData("cam"), TransformData("lamp")])
quad = MeshData(
    points      = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    indices     = np.arange(4),
    counts      = np.array([4]),
    name        = "quad",
    parent_uuid = rig["grp"].uuid,            # the node it sits on
)
uvs  = UVData(points=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), indices=np.arange(4), counts=np.array([4]), name="map1", mesh_uuid=quad.uuid)
mat  = MaterialData(name="paint", mesh_uuid=quad.uuid, color=(0.8, 0.2, 0.1), diffuse=1.0)
cam  = CameraData(name="cam", focal_length=50.0, parent_uuid=rig["cam"].uuid)
lamp = LightData(name="lamp", kind="point", intensity=2.0, parent_uuid=rig["lamp"].uuid)

DEMO_FBX = os.path.join(WORK, "demo.fbx")
save_fbx(DEMO_FBX, hierarchy=rig, meshes=quad, uvs=uvs, materials=MaterialList([mat]), cameras=CameraList([cam]), lights=LightList([lamp]))

held = load_fbx(DEMO_FBX)
print(FbxContents._fields)
print(held.hierarchy.name, held.meshes.name, held.uvs.name, held.materials.name, held.cameras.name, held.lights.name)
print(held.hierarchy[held.meshes[0].parent_uuid].name, held.clip, held.skins, held.morphs)   # grp None None None
```

The fields are `save_fbx`'s arguments, in order, so `save_fbx(b, *load_fbx(a))`
writes a file back. Each kind is a flag, `True` by default (`takes`,
`meshes`, `uvs`, `skins`, `materials`, `textures`, `morphs`, `cameras`,
`lights`, `curves`, `surfaces`); a kind left unread is None and costs nothing, the nodes are
always read. `textures=False` keeps each material's `texture_file`, UV set
and wraps and decodes no pixels; a texture file many materials name is
decoded once, the one array shared. `verbose=False` silences the read's
warnings.

```python
some = load_fbx(DEMO_FBX, materials=False, cameras=False, lights=False)
print(some.hierarchy.name, some.meshes.name, some.uvs.name, some.materials, some.cameras, some.lights)
print(load_fbx(DEMO_FBX, meshes=False).meshes, load_fbx(DEMO_FBX, textures=False).materials[0].texture)   # None None
```

A mesh stores the least normals that shade it as its file does. Where the
file's normals are the ones Maya derives from the points (within 1e-5 a
component) the mesh stores none, `hard_edges` holding the edges they break
at, a pair of vertices each (None when every edge is soft); any other
normals are stored as read, each face corner naming its own.
`get_face_vertex_normals()` gives the normal of every face corner either
way:

```python
flat = quad.copy()
flat.set_normals()  # the derived normals, stored
tilt         = quad.copy()
tilt.normals = np.tile([0.0, 0.6, 0.8], (4, 1))  # authored: not what the points give
for mesh in (flat, tilt):
    save_fbx(os.path.join(WORK, "normals.fbx"), hierarchy=rig, meshes=mesh)
    back = load_fbx(os.path.join(WORK, "normals.fbx")).meshes[0]
    print(back.normals is None, back.get_face_vertex_normals()[0])  # True [0. 0. 1.], then False [0.  0.6 0.8]
```

A take is a `ClipData` over the nodes; a file holding takes comes back
with `hierarchy` None and the nodes in `clip` (`takes=False` reads the
nodes at their own values instead, the take left unread):

```python
clip                      = ClipData(rig, frames=3, fps=24.0, take="spin")
clip.frames["grp"].rotate = np.array([[0.0, 0.0, 0.0], [0.0, 45.0, 0.0], [0.0, 90.0, 0.0]])
save_fbx(os.path.join(WORK, "spin.fbx"), clip=clip)

back = load_fbx(os.path.join(WORK, "spin.fbx"))
print(back.hierarchy, [take.take for take in back.clip], back.clip.active.frame_count)   # None ['spin'] 3
```

A take covers its stored span widened to its keys (an Unreal export stores
the span a frame short of its last key); its layers holding no curve are
passed over, curves on two layers, or on a muted one, are refused. A
transform's rotate pivot, scale pivot and the two offsets are folded into
its translate, as the matrix Maya composes holds them, on every frame
where translate, rotate or scale is keyed; its 3x3 is untouched. Pivots on
a joint, or keyed pivots, are refused.

NURBS curves and surfaces are the `curves` and `surfaces` fields, a
`NurbsCurveList` and a `NurbsSurfaceList`, each shape in its node's space
and naming the node by `parent_uuid`:

```python
from cgmath.geometry import NurbsCurveData, NurbsCurveList, NurbsSurfaceData, NurbsSurfaceList

nodes = HierarchyData([TransformData("arc"), TransformData("edges"), TransformData("patch")])
along = np.arange(6.0)
arc   = NurbsCurveData(points=np.stack([along, np.sin(along), np.zeros(6)], axis=1), degree=3, parent_uuid=nodes["arc"].uuid)
edge  = NurbsCurveData(points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), degree=1, parent_uuid=nodes["edges"].uuid)
rung  = NurbsCurveData(points=np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]), degree=1, parent_uuid=nodes["edges"].uuid)
u, v = np.meshgrid(np.arange(4.0), np.arange(5.0), indexing="ij")
patch = NurbsSurfaceData(points=np.stack([u, 0.1 * u * v, v], axis=2), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False, parent_uuid=nodes["patch"].uuid)

NURBS_FBX = os.path.join(WORK, "nurbs.fbx")
save_fbx(NURBS_FBX, hierarchy=nodes, curves=NurbsCurveList([arc, edge, rung]), surfaces=NurbsSurfaceList([patch]))

back = load_fbx(NURBS_FBX)
print([(curve.degree, curve.knots, back.hierarchy[curve.parent_uuid].name) for curve in back.curves])      # [(3, None, 'arc'), (1, None, 'edges'), (1, None, 'edges')]
print(back.surfaces[0].points.shape, back.surfaces[0] == patch, load_fbx(NURBS_FBX, curves=False).curves)  # (4, 5, 3) True None
```

| In cgmath | In the file | Note |
|---|---|---|
| A curve of degree 1, not periodic, `knots` None | a line of an `FbxLine` | any number on one node, as Maya writes them; a read gives `knots` None |
| Any other curve | an `FbxNurbsCurve` | one a node; a periodic curve holds its points once |
| A surface | an `FbxNurbsSurface` | one a node; `points` are `[u][v]`, the file runs u fastest |
| `knots`, `knots_u`, `knots_v` | Maya's, plus one at each end | None on a read where they are the ones the shape gives itself |
| A shape whose `parent_uuid` names no node | a new root node, `curve` or `surface` | at identity |
| A closed curve of a file | read open | its first point again at the end |

Refused, on a read: a rational curve or surface (a control point's weight
that is not 1) and a trimmed surface, which no field holds; `curves=False`
or `surfaces=False` reads the file without them. On a save: two shapes on
one node, other than lines; a curve or surface on a joint, or on a node
holding a mesh, a camera or a light; points that are not 3D. Maya writes
two curves of degree 2 or more under one transform as an empty node, and
reads the first of two the file holds on one, so a control of several such
curves is a node a curve. A file holds no `uniform` or `registered` flag: a
curve that stored the knots it would give itself reads back with `knots`
None.

The measured conventions, the refusals (shear, non-uniform scale off a
leaf, a geometric transform on a mesh node, two deformers on one mesh ...)
and each shape's fields are in the hierarchy cheatsheet's "Importers and
exporters" and the geometry cheatsheet's `MaterialData` and `MorphData`
sections.

---

## `fbx` — FbxExporter

Writes a `HierarchyData` skeleton out as an FBX. One skeleton per exporter.
Each node is written after its parent, whatever order the list holds them
in.

| Member | Type | Notes |
|---|---|---|
| `.manager` | `fbx.FbxManager` | lazily created, with IO settings |
| `.close()` | — | frees the manager and scene; `with` calls it; safe twice |
| `.scene` | `fbx.FbxScene` | lazily created, labelled Maya's Y-up and cm |
| `.add_skeleton(component)` | — | `ValueError` if one is already set |
| `.export(path, as_ascii=False, zero_root=False)` | — | `path` may be a `str` or `Path` |
| `._ROTATE_ORDER_MAP` | `dict` | `HierarchyData` rotate-order → `EFbxRotationOrder` |

```python
from cgmath.formats.fbx import FbxExporter
from cgmath.hierarchy import HierarchyData, TransformData

skeleton = HierarchyData([
    TransformData("root", node_type="joint"),
    TransformData("spine", translate=(0, 10, 0), node_type="joint"),
    TransformData("head", translate=(0, 10, 0), node_type="joint"),
])
skeleton["spine"].set_parent("root", world_space=False)
skeleton["head"].set_parent("spine", world_space=False)

skel_path = os.path.join(WORK, "skeleton.fbx")

with FbxExporter() as exporter:
    print(type(exporter.manager).__name__, type(exporter.scene).__name__)
    exporter.add_skeleton(skeleton)
    exporter.export(skel_path)
print(os.path.exists(skel_path))
```

Read it straight back:

```python
print(load_fbx(skel_path).hierarchy.name)   # ['root', 'spine', 'head']
```

ASCII output, and the one-skeleton rule:

```python
with FbxExporter() as ascii_exporter:
    ascii_exporter.add_skeleton(skeleton)
    ascii_exporter.export(os.path.join(WORK, "skeleton_ascii.fbx"), as_ascii=True)

    try:
        ascii_exporter.add_skeleton(skeleton)
    except ValueError as error:
        print(str(error)[:30])
```

Each `TransformData` maps onto FBX like this:

| `node_type` | FBX attribute | Inherit type | Extras applied |
|---|---|---|---|
| `"joint"` | `FbxSkeleton`, `eLimbNode` at the top too, as Maya writes every joint (Maya's importer reads an `eRoot` with no child as a plain transform) | `eInheritRrs` with `segment_scale_compensate` on, `eInheritRSrs` with it off | `joint_orient` → pre-rotation, `rotate_axis` → inverted post-rotation |
| `"transform"` | `FbxNull`, look `eNone` | the same | — |
| `"locator"` | `FbxNull`, look `eCross`, as Maya writes a locator | the same | — |
| anything else | not written, nor any node under it; one warning names how many of each type were skipped | — | — |

FBX has no shear channel: a save of a node holding a shear over 1e-9
raises ValueError naming it, before anything is written.

```python
sheared = HierarchyData([TransformData("grp", shear=(0.5, 0.0, 0.0)), TransformData("loc")])
try:
    sheared.save_fbx(os.path.join(WORK, "sheared.fbx"))
    raise AssertionError("a shear saved")
except ValueError as error:
    assert "'grp'" in str(error)
```

Every node is written with rotation active, as Maya's exporter writes it:
FBX turns a node without it in xyz, whatever its `rotate_order`. A group
or a locator gets the inherit type a joint gets, `eInheritRSrs` unless
`segment_scale_compensate` is on, as Maya's exporter writes a group or a
locator: FBX's default, `eInheritRrSs`, which earlier versions left on
them, scales a node along its own axes by its parent's world scale, so
under an unevenly scaled parent the SDK and Maya's importer put it, and
the nodes under it, elsewhere than the rig has them (a group at (1, 1, 0)
turned (40, 10, -20) under a group turned 30 about z and scaled
(1, 3, 0.5): 1.58 off in its 3x3, a locator at (0, 2, 1) under it 3.17 in
position). `HierarchyData.save_fbx` says where Maya puts a group or a
locator with `segment_scale_compensate` on.

The scene is labelled Maya's Y-up and cm, the axes and unit every cgmath
value is in, and the values are written as they are, but for a scale under 1e-12 in size, 0
and -0 among them, written as 1e-12, positive, as Maya's own exporter
writes a static one and as Maya's evaluation holds it:
`HierarchyData.save_fbx` says where that puts a save on a load and in
Maya.

`user_defined_attributes` become real FBX properties, typed the way Maya's
own FBX plug-in types them, and `load_fbx` reads them back as the same
specs, binary or ASCII. A number, `bool`, `enum` or `string` is one
property, and a string Maya has not set (its value `None`) an empty one; a
`double3` or `float3` is one vector holding its three children; a
`stringArray`, which Maya's plug-in skips, is a string holding the list.
What the FBX types can't say (a `float` from a `double`, an enum's own
indices, `channel_box`), and a value the file can't give back (an enum
value that is no field's index, what an ASCII file changes), rides along
in a hidden property Maya's importer leaves alone. Arrays, matrices,
multis and other compounds are skipped with one warning, and the children
of a skipped compound come back on their own.

```python
tagged = HierarchyData([TransformData("root", node_type="joint")])
tagged[0].add_user_attribute("heroHeight", 1.8)
tagged[0].add_user_attribute("tags", ["deform", "twist"])
tagged[0].add_user_attribute(
    "space", 5, attribute_type="enum", enum_names="local=1:world=5"
)
tagged.save_fbx(os.path.join(WORK, "tagged.fbx"))

back = HierarchyData.load_fbx(os.path.join(WORK, "tagged.fbx"))
print(back[0].heroHeight, back[0].tags, back[0].space)   # 1.8 ['deform', 'twist'] 5
assert back[0].user_defined_attributes == tagged[0].user_defined_attributes
```

An ASCII file (`as_ascii=True`) changes some values, and `load_fbx` still
gives them back as they went in: it keeps 15 significant digits of a
double, and reads the smallest `short` or `long` back as the largest and a
typed `&quot;` as a quote. Past about 6,100 bytes its reader changes a
string holding a quote, line break, tab, semicolon, `&` or run of spaces,
and loses the whole scene for a brace, so a string, or an enum's field
list, over 4,096 bytes holding one is %-escaped there, with a warning. Maya
reads those as the file holds them.

```python
tagged[0].add_user_attribute("note", "a typed &quot; stays")
tagged[0].add_user_attribute("third", 1 / 3)
tagged[0].add_user_attribute("blob", '{"say": "hi"}\n' * 1000)
tagged.save_fbx(os.path.join(WORK, "label.fbx"), as_ascii=True)

back = HierarchyData.load_fbx(os.path.join(WORK, "label.fbx"))
print(back[0].note, back[0].third)       # a typed &quot; stays 0.3333333333333333
assert back[0].user_defined_attributes == tagged[0].user_defined_attributes
```

The FBX SDK takes plain ASCII text only: a string that is not raises
`ValueError`, naming the node and attribute, before anything is written.

```python
tagged[0].add_user_attribute("label", "hanche gauche \u00e9")
try:
    tagged.save_fbx(os.path.join(WORK, "label.fbx"))
except ValueError as error:
    print(str(error)[:45])               # user defined attr root.label can't be written
```

`export(path, zero_root=True)` resets the root joint's translate, rotate and
scale before writing, so the file carries the skeleton at the origin. The
root joint is the first joint the file writes as a root: one at the world,
or a view's top joint, whose parent the view leaves out:

```python
with FbxExporter() as exporter:
    exporter.add_skeleton(skeleton)
    exporter.export(os.path.join(WORK, "skeleton_zeroed.fbx"), zero_root=True)
```

It takes a `HierarchyData` or a view of one, or a pipeline skeleton
component holding one as `.data`.

---

## The class routes slice the same read

`cgmath.formats.load_fbx` hands back every class at once; each class's own
`load_fbx` is one field of it, read the same way, its kinds alone:
`MeshData.load_fbx` reads the nodes and the meshes, `UVList.load_fbx` the
UV sets too, `SkinList.load_fbx` the skins, `HierarchyData.load_fbx` the
nodes, `ClipList.load_fbx` the nodes and the takes. Every route takes
`verbose=False`; `Scene.load_fbx` takes every flag
(`Scene.load_fbx(path, textures=False)`) and leaves the curves and
surfaces unread. GLB has no such bundle: its
readers live in `geometry` and `hierarchy` (`GlbData.mesh_list` is the one
`MeshData` this package gives).

<!-- notest: illustrative import map; the loaders need real asset files -->
```python
from cgmath.geometry.mesh import load_fbx, load_glb          # [(MeshData, UVList), ...]
from cgmath.geometry.skin_weights import load_fbx as skin_fbx
from cgmath.geometry.skin_weights import load_glb as skin_glb
from cgmath.hierarchy import load_fbx as rig_fbx
from cgmath.render import Scene

load_fbx("hero.fbx")        # the meshes field, each with its UV sets
load_glb("hero.glb", scale_factor=100.0)
skin_fbx("hero.fbx")        # -> SkinList, the skins field
skin_glb("hero.glb")        # -> SkinList
rig_fbx("hero.fbx")         # -> HierarchyData, the hierarchy field (a take's nodes for a file of takes)
Scene.load_fbx("hero.fbx")  # -> every field, ready to render
```

---

## Gotchas

| Gotcha | What to do |
|---|---|
| `import cgmath.formats.fbx` fails without the SDK | Wrap the import; `glb` only fails at call time |
| `FbxExporter.save` / `.add_scene` do not exist | `.export(path)` / `.add_skeleton(component)` |
| `load_fbx` gives `hierarchy` None | the file holds takes: the nodes are in `clip`, every take's; `takes=False` gives the nodes |
| A material's `texture` is None | read with `textures=False`, the file not found, or no PIL: `texture_file` still names it |
| A loaded mesh's `normals` is None | the file's were the ones Maya derives from the points, so none is stored: `mesh.get_face_vertex_normals()` gives them, `hard_edges` the edges they break at |
| `load_fbx` raises on a rational curve or a trimmed surface | no field holds a weight or a trim: `curves=False` / `surfaces=False` reads the file without them |
| A loaded curve's `knots` is None | the file's were the ones the curve gives itself; a line of a polyline holds none |
| A shape's `parent_uuid` / `mesh_uuid` matches nothing | it came from another load: each read mints its own uuids |
| `set_binary` on a `GLTF2()` with no blob | `TypeError` — `binary_blob()` is `None`; load a real file first |
| `GlbData.mesh_list` is `None` | `trimesh` is not installed |
| `GlbData` mesh points look 100× too big | They are — trimesh metres are scaled to centimetres |
| `Primitive.triangles` is 1-D | `.reshape(-1, 3)` yourself |
| `JOINTS_0` slots look like node indices | They index `Skin.joints`, which then indexes the nodes |
| `Model.nodes` order surprises you | Use `Model.ordered_node_indexes` |
| `FbxExporter` holds memory | `.close()` it, or use a `with` block |

---

## `usd.stage` — stages, caches, sublayers

Everything takes and returns raw `pxr` objects. Stages register with USD's
default stage cache unless you say otherwise.

```python
import os
import tempfile

from pxr import Usd, UsdGeom

from cgmath.formats.usd import prim as usd_prim
from cgmath.formats.usd import stage as usd_stage

usd_root = tempfile.mkdtemp()

mem  = usd_stage.new_stage_in_memory()               # default cache
mem2 = usd_stage.new_stage_in_memory(no_cache=True)  # throw-away cache, cleared on exit
own  = Usd.StageCache()
mem3 = usd_stage.new_stage_in_memory(cache=own)      # your cache
print(own.Contains(mem3), own.Contains(mem))
```

```python
with usd_stage.no_stage_cache():                         # the context behind no_cache=True
    scratch = Usd.Stage.CreateInMemory()
print(scratch.GetRootLayer().anonymous)
```

`new_stage` creates the file, or opens and clears it when it already exists.
`write_stage` picks `Save` when the stage is bound to that path and `Export`
otherwise.

```python
asset_path = os.path.join(usd_root, "asset.usda")
asset      = usd_stage.new_stage(asset_path)
UsdGeom.Xform.Define(asset, "/Asset")
usd_stage.write_stage(asset)                                             # Save()
usd_stage.write_stage(asset, os.path.join(usd_root, "asset_copy.usda"))  # Export()
print(sorted(os.listdir(usd_root)))

again = usd_stage.new_stage(asset_path)                  # exists -> opened and cleared
print(again.GetPrimAtPath("/Asset").IsValid())
```

```python
opened = usd_stage.open_stage(os.path.join(usd_root, "asset_copy.usda"), no_cache=True)
print(opened.GetPrimAtPath("/Asset").IsValid())
```

Sublayers are stored relative to the stage directory. `add_sublayers` skips
duplicates; `replace=True` overwrites; `clear_missing_sublayers` drops paths
whose file is gone.

```python
shot_path = os.path.join(usd_root, "shot.usda")
shot      = usd_stage.new_stage(shot_path)
usd_stage.add_sublayers(shot, [asset_path, os.path.join(usd_root, "asset_copy.usda")])
print(usd_stage.get_sublayers(shot))                 # relative
print(usd_stage.get_sublayers(shot, absolute=True))  # resolved

usd_stage.set_sublayers(shot, asset_path)            # replace the list
print(usd_stage.get_sublayers(shot))

usd_stage.add_sublayers(shot, os.path.join(usd_root, "missing.usda"))
usd_stage.clear_missing_sublayers(shot)
print(usd_stage.get_sublayers(shot))

usd_stage.clear_sublayers(shot)
print(usd_stage.get_sublayers(shot))
```

---

## `usd.prim` — search, duplicate, compose

`create_scopes` builds every missing ancestor as a `Scope` (or untyped with
`as_scope=False`) and returns the leaf.

```python
stage = usd_stage.new_stage_in_memory(no_cache=True)
hero  = usd_prim.create_scopes(stage, "/World/Chars/Hero")
body  = UsdGeom.Mesh.Define(stage, "/World/Chars/Hero/Body").GetPrim()
rig   = UsdGeom.Xform.Define(stage, "/World/Chars/Hero/Rig").GetPrim()
UsdGeom.Mesh(body).CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
print(hero.GetTypeName(), stage.GetPrimAtPath("/World").GetTypeName())
```

`iter_prims` is a generator over a stage or a prim; `find_prims` is the list
form. Filter by type name, `Usd.Typed` schema, prim name or an `fnmatch`
path pattern, and prune branches with `skip_children_callback`.

```python
print([p.GetName() for p in usd_prim.iter_prims(stage, prim_type="Mesh")])
print([p.GetName() for p in usd_prim.iter_prims(stage, prim_type=UsdGeom.Xformable)])
print([p.GetName() for p in usd_prim.iter_prims(hero)])                 # a prim, not a stage
print([p.GetPath().pathString for p in usd_prim.iter_prims(stage, match_prim_path="/World/*/Hero")])
print(usd_prim.find_prims(stage, prim_name="Rig", first_only=True).GetPath())
print(usd_prim.find_prims(stage, prim_name="nobody", first_only=True))  # None

skip = lambda p: p.GetName() == "Hero"                                         # do not descend into Hero
print([p.GetName() for p in usd_prim.iter_prims(stage, skip_children_callback=skip)])
```

```python
print(usd_prim.is_prim_type(body, "Mesh"), usd_prim.is_prim_type(body, ["Xform", "Scope"]))
print(usd_prim.is_prim_type(body, UsdGeom.Gprim))
print(usd_prim.is_descendant(stage, body, hero), usd_prim.is_descendant(stage, "/World", body))
print(usd_prim.is_valid_prim(asset_path, "/Asset"), usd_prim.is_valid_prim(asset_path, "/Nope"))
```

`duplicate_prim` copies the prim type, authored metadata and authored
attributes — never relationships. `exclude_attrs` is a list of regexes,
`attr_rename` a callable, `extra_prims` more prims to pull attributes from.

```python
copy = usd_prim.duplicate_prim(body, stage, "/World/Chars/Hero/BodyCopy")
print(copy.GetTypeName(), UsdGeom.Mesh(copy).GetPointsAttr().Get())

as_xform = usd_prim.duplicate_prim(body, stage, "/World/Chars/Hero/BodyXform",
                                   prim_type="Xform", exclude_attrs=["^points$"])
print(as_xform.GetTypeName(), as_xform.HasAttribute("points"))

renamed = usd_prim.duplicate_prim(body, stage, "/World/Chars/Hero/Renamed",
                                  attr_rename=lambda n: "old_" + n)
print(renamed.HasAttribute("old_points"))
```

```python
target = stage.DefinePrim("/World/Chars/Hero/Target")
usd_prim.copy_property(body, target, "points", dst_name="pts")
usd_prim.copy_properties(body, target, exclude_names=["points"])
print(sorted(target.GetPropertyNames()))
```

References and inherits take one path, a `(file, prim_path)` tuple, or a
list of either. Absolute reference paths are stored relative to the stage
directory (`start_dir` overrides); `replace=True` clears first, and the
`set_*` forms are exactly that.

```python
ref_holder = shot.DefinePrim("/Shot/Hero")
usd_prim.add_references(ref_holder, [(asset_path, "/Asset")])
print(usd_prim.get_references(ref_holder, abspath=False))
usd_prim.set_references(ref_holder, os.path.join(usd_root, "asset_copy.usda"))
print(len(usd_prim.get_references(ref_holder)))
usd_prim.clear_references(ref_holder)
print(usd_prim.get_references(ref_holder))

usd_prim.add_inherits(body, "/World/Chars/Hero/Rig")
usd_prim.add_inherits(body, rig)                                # a prim works too; duplicates are ignored
print(usd_prim.get_inherits(body))
usd_prim.set_inherits(body, "/World")
print(usd_prim.get_inherits(body))
usd_prim.clear_inherits(body)
print(usd_prim.get_inherits(body))
```

`add_api_schema` / `get_api_schemas` operate on the `apiSchemas` metadata
directly, so they behave the same across USD versions.

```python
usd_prim.add_api_schema(body, "SkelBindingAPI")
print(usd_prim.get_api_schemas(body))
```
