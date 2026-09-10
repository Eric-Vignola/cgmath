# `cgmath.formats` Cheatsheet

Copy-paste recipes for the FBX, GLB and USD wrappers. Every `python` block below
runs, in order, as one script — the fixtures are built inline because no
`.glb` / `.fbx` assets ship with this repo. For the overview see
[`README.md`](README.md); for the rest of the library see
[`../CHEATSHEET.md`](../CHEATSHEET.md).

`__init__.py` is a docstring only. Import the submodules directly.

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
- [`fbx` — build a fixture](#fbx--build-a-fixture)
- [`fbx` — SceneData](#fbx--scenedata)
- [`fbx` — TakeData](#fbx--takedata)
- [`fbx` — LayerData](#fbx--layerdata)
- [`fbx` — CurveData](#fbx--curvedata)
- [`fbx` — the List helpers](#fbx--the-list-helpers)
- [`fbx` — BaseData / BaseList](#fbx--basedata--baselist)
- [`usd.stage` — stages, caches, sublayers](#usdstage--stages-caches-sublayers)
- [`usd.prim` — search, duplicate, compose](#usdprim--search-duplicate-compose)
- [`fbx` — FbxExporter](#fbx--fbxexporter)
- [Mesh loading lives in geometry](#mesh-loading-lives-in-geometry)
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
print(anim._fields)   # ('samplers', 'channels', 'duration')
print(anim.duration)  # max last keyframe time across samplers

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

The object wrapper. Loads on construction; `pygltflib` is required,
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

## `fbx` — build a fixture

Needs the Autodesk FBX Python SDK. Note `import cgmath.formats.fbx`
**raises `ImportError`** without it — the SDK import is unguarded.

Two nulls, one take, one keyed translate curve.

```python
import fbx                                   # the Autodesk SDK

DEMO_FBX = os.path.join(WORK, "demo.fbx")

manager  = fbx.FbxManager.Create()
manager.SetIOSettings(fbx.FbxIOSettings.Create(manager, fbx.IOSROOT))
scene = fbx.FbxScene.Create(manager, "demo")
scene.GetGlobalSettings().SetTimeMode(fbx.FbxTime.EMode.eFrames24)

parent = fbx.FbxNode.Create(scene, "pCube1")
parent.SetNodeAttribute(fbx.FbxNull.Create(manager, ""))
scene.GetRootNode().AddChild(parent)

child = fbx.FbxNode.Create(scene, "pCube2")
child.SetNodeAttribute(fbx.FbxNull.Create(manager, ""))
parent.AddChild(child)

stack = fbx.FbxAnimStack.Create(scene, "walk")
layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
stack.AddMember(layer)

start, stop = fbx.FbxTime(), fbx.FbxTime()
start.SetSecondDouble(0.0)
stop.SetSecondDouble(1.0)
stack.SetLocalTimeSpan(fbx.FbxTimeSpan(start, stop))

curve = parent.LclTranslation.GetCurve(layer, "X", True)
curve.KeyModifyBegin()
for seconds, value in [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)]:
    moment = fbx.FbxTime()
    moment.SetSecondDouble(seconds)
    added = curve.KeyAdd(moment)
    curve.KeySetValue(added[0] if isinstance(added, tuple) else added, value)
curve.KeyModifyEnd()

exporter = fbx.FbxExporter.Create(manager, "")
exporter.Initialize(
    DEMO_FBX,
    manager.GetIOPluginRegistry().GetNativeWriterFormat(),
    manager.GetIOSettings(),
)
exporter.Export(scene)
exporter.Destroy()
manager.Destroy()
```

---

## `fbx` — SceneData

The entry point. **The constructor loads** — there is no `from_file`
classmethod.

```python
from cgmath.formats.fbx import SceneData

data = SceneData(DEMO_FBX)          # ~ is expanded; FileNotFoundError if absent
print(data.name)                    # 'demo' — set from the file stem on load
```

Empty first, load later:

```python
later = SceneData()
print(later.name, later.fps, later.takes, later.nodes)   # 'No Scene' None TakeList([]) []
later.load(DEMO_FBX)
print(later.name)
later.destroy()
```

| Member | Type | Notes |
|---|---|---|
| `.name` | `str` | `"No Scene"` when nothing is loaded |
| `.rename(name)` | — | `RuntimeError` if unloaded, `ValueError` if empty |
| `.load(filename)` | — | `FileNotFoundError` / `RuntimeError` on a bad file |
| `.save(filename=None, embed_media=True, file_format=-1)` | `str` | returns the expanded path |
| `.scene` | `fbx.FbxScene` | the raw SDK scene, or `None` |
| `.fps` | `float` | get/set; `None` for an unknown time mode |
| `.linear_units` | `str` | get/set; `"centimeters"`, `"meters"`, … |
| `.scale_factor` | `float` | read-only, from the system unit |
| `.takes` | `TakeList` | rebuilt on every access |
| `.nodes` | `list[str]` | hierarchy + `"blendshape.channel"` names, root excluded |
| `.create_take(name)` | `TakeData` | also creates a `"BaseLayer"` |
| `.destroy()` | — | destroys the FBX manager; call it |

Global settings:

```python
print(data.fps, data.linear_units, data.scale_factor)   # 24.0 centimeters 1.0

data.fps          = 30        # accepts a number or a numeric string
data.linear_units = "meters"  # long or short form: 'm', 'meter', 'meters'
print(data.fps, data.linear_units, data.scale_factor)   # 30.0 meters 100.0
```

Supported `fps`: 24, 29.97, 30, 48, 50, 59.94, 60, 72, 96, 100, 120, 1000.
Supported units: mm, cm, dm, m, km, in, ft, mi, yd (and their long forms).
Anything else raises `ValueError` listing the accepted values.

```python
for bad, kind in [(17, "fps"), ("furlongs", "linear_units")]:
    try:
        setattr(data, kind, bad)
    except ValueError as error:
        print(kind, "->", str(error)[:45])
```

Nodes and renaming:

```python
print(data.nodes)                   # ['pCube1', 'pCube2'] — the root is dropped
data.rename("hero")
print(data.name)
```

Saving. `embed_media=False` with the default `file_format=-1` auto-selects
ASCII FBX.

```python
saved      = data.save(os.path.join(WORK, "out.fbx"))              # binary, media embedded
ascii_path = data.save(os.path.join(WORK, "out_ascii.fbx"), embed_media=False)
print(os.path.basename(saved), os.path.basename(ascii_path))
print(open(ascii_path, "rb").read(6))                         # b'; FBX '
```

The `.fbx` extension is appended if you leave it off, and `save()` with no
argument overwrites the file that was loaded.

---

## `fbx` — TakeData

A take is an FBX animation stack.

```python
take = data.takes[0]                # by index, or data.takes["walk"] by name
print(take.name, repr(take), str(take))
```

| Member | Type | Notes |
|---|---|---|
| `.name` | `str` | Maya-translated |
| `.rename(name)` | — | `ValueError` if another take already owns the name |
| `.layers` | `LayerList` | |
| `.create_layer(name)` | `LayerData` | `ValueError` on a duplicate name |
| `.curves` | `CurveList` | single-layer shortcut — `RuntimeError` if >1 layer |
| `.create_curve(node, property)` | `CurveData` | same shortcut; creates `"BaseLayer"` if none |
| `.start_time` / `.end_time` | `float` | seconds, get/set |
| `.start_frame` / `.end_frame` | `float` | frames, get/set — needs `fps` |
| `.duration` | `float` | `end_time - start_time`, read-only |
| `.frame_count` | `float` | `duration * fps`, read-only |

Renaming is validated against the other takes in the scene:

```python
take.rename("walk_v2")
print(data.takes)
take.rename("walk")
```

Timing:

```python
print(take.start_time, take.end_time, take.duration)       # 0.0 1.0 1.0
print(take.start_frame, take.end_frame, take.frame_count)  # at 30 fps: 0.0 30.0 30.0

take.end_frame = 60                 # frames in, seconds stored
print(take.end_time, take.duration)
take.start_time = 0.5               # seconds also work
print(take.start_frame)
```

Layers, and the single-layer `curves` shortcut:

```python
print(take.layers)  # LayerList(['BaseLayer'])
print(take.curves)  # CurveList(['pCube1.translateX'])

take.create_layer("Additive")
try:
    take.curves                     # ambiguous now
except RuntimeError as error:
    print(str(error)[:52])
take.layers.delete("Additive")
```

Create a take (a `"BaseLayer"` comes with it) and key something on it:

```python
run = data.create_take("run")
print(run.layers)                             # LayerList(['BaseLayer'])

bounce        = run.create_curve("pCube2", "translateY")   # Maya property names
bounce.times  = [0.0, 0.5, 1.0]
bounce.values = [0.0, 4.0, 0.0]
print(run.curves)
```

---

## `fbx` — LayerData

A layer is an FBX animation layer inside a take.

| Member | Type | Notes |
|---|---|---|
| `.name` | `str` | |
| `.rename(name)` | — | `ValueError` if a sibling layer owns the name |
| `.curves` | `CurveList` | rebuilt by walking the layer's curve nodes |
| `.create_curve(node, property)` | `CurveData` | Maya or FBX property names |

```python
base = take.layers[0]
print(base.name, base.curves)

base.rename("Base")
print(take.layers)                  # LayerList(['Base'])
base.rename("BaseLayer")
```

`create_curve` accepts either naming convention and returns a curve with no
keys yet:

```python
spin = base.create_curve("pCube2", "rotateZ")     # -> 'Lcl RotationZ'
print(spin.name, spin.key_count)                  # pCube2.rotateZ 0
```

| You pass | FBX gets |
|---|---|
| `"translateX"` / `"rotateY"` / `"scaleZ"` | `"Lcl TranslationX"` / `"Lcl RotationY"` / `"Lcl ScalingZ"` |
| `"visibility"` | `"Visibility"` |
| node=`"face_mesh"`, property=`"smile"` | `"face_mesh.smile"` + `"DeformPercent"` |
| an FBX name already | passed straight through |

An unknown node raises `ValueError`:

```python
try:
    base.create_curve("ghost", "translateX")
except ValueError as error:
    print(error)
```

---

## `fbx` — CurveData

One animation curve. Times are seconds, frames are `times * fps`.

| Member | Type | Notes |
|---|---|---|
| `.name` | `str` | Maya-translated `"node.attribute"` |
| `.key_count` | `int` | |
| `.times` | `np.ndarray` | get/set — **the setter clears and re-creates every key** |
| `.frames` | `np.ndarray` | get/set — same, in frames; `None` if no fps |
| `.values` | `np.ndarray` | get/set — the setter needs a matching key count |
| `.clear()` | — | removes keys one by one, keeping the attribute connection |

```python
curve_data = take.curves["pCube1.translateX"]
print(curve_data.name, curve_data.key_count)
print(curve_data.times)   # [0.  0.5 1. ]
print(curve_data.values)  # [ 0.  5. 10.]
print(curve_data.frames)  # times * fps
```

Retiming and re-valuing. Order matters — set times first, values second,
because assigning `times` (or `frames`) wipes the values to `0.0`.

```python
curve_data.times = [0.0, 1.0, 2.0]
print(curve_data.values)             # [0. 0. 0.] — reset by the times setter
curve_data.values = [0.0, 7.5, 15.0]
print(curve_data.times, curve_data.values)
```

Frames are the same door:

```python
curve_data.frames = [0.0, 15.0, 30.0]
print(curve_data.times)              # /fps
curve_data.values = [0.0, 5.0, 10.0]
```

Guard rails:

```python
empty = base.create_curve("pCube2", "scaleY")
try:
    empty.values = [1.0]             # nothing to write onto
except RuntimeError as error:
    print(str(error)[:48])

try:
    curve_data.values = [1.0]        # 1 value, 3 keys
except ValueError as error:
    print(str(error)[:48])
```

Clearing keeps the curve connected to its attribute, unlike `KeyClear()`:

```python
curve_data.clear()
print(curve_data.key_count, take.curves)    # 0, curve still listed
```

Blendshape curves are rescaled for you: FBX stores `DeformPercent` in
`0..100`, `.values` reads and writes `0..1`.

---

## `fbx` — the List helpers

`TakeList`, `LayerList` and `CurveList` all share one container. Index by
position or by name, iterate, measure, and substring-test.

```python
from cgmath.formats.fbx import CurveList, LayerList, TakeList

takes = data.takes
print(takes)                    # TakeList(['walk', 'run'])
print(len(takes))
print(takes[0].name)            # by index
print(takes["run"].name)        # by name
print("run" in takes)           # substring match against every name
print([t.name for t in takes])  # iterable
```

Missing names and bad key types raise:

```python
try:
    takes["nope"]
except KeyError as error:
    print(error)

try:
    takes[1.5]
except TypeError as error:
    print(error)
```

`CurveList` additionally accepts **either** naming convention on lookup:

```python
curves = run.curves
print(curves["pCube2.translateY"].name)     # Maya spelling
print(curves[0].name)
```

Each list is type-checked on construction and on `append`:

```python
print(TakeList(), LayerList(), CurveList())      # all empty
try:
    TakeList([1])
except TypeError as error:
    print(error)
```

`delete(name)` removes the object from the list **and destroys it in the
scene**. It works for takes and layers that hold no curves:

```python
scratch = data.create_take("scratch")
data.takes.delete("scratch")
print(data.takes)
```

<!-- notest: CurveList.delete calls DisconnectFromChannel(j) with one argument; the SDK wants (FbxAnimCurve, int), so this raises TypeError -->
```python
run.curves.delete("pCube2.translateY")  # TypeError against the real FBX SDK
data.takes.delete("run")                # same, via TakeList -> LayerList -> CurveList
```

Deleting a curve — and therefore any take or layer that still owns one — is
currently broken against the SDK. Use `CurveData.clear()` to empty a curve
instead.

---

## `fbx` — BaseData / BaseList

The shared bases. You rarely construct them, but every wrapper inherits
their behaviour.

`BaseData(scene, data_object)` gives you `.name` (Maya-translated),
`in` (substring against the name), `str()` and `repr()`:

```python
from cgmath.formats.fbx import BaseData

print(isinstance(take, BaseData))
print("Cube" in take.curves[0])  # substring, not equality
print(str(take), repr(take))     # walk  TakeData('walk')
```

`BaseList(data_objects=None)` gives the container behaviour above plus
`append`, and leaves `_delete_object_from_scene` to the subclass — a bare
`BaseList` raises `NotImplementedError` on `delete`.

```python
from cgmath.formats.fbx import BaseList

bare = BaseList()
print(len(bare), list(bare))         # 0 [] — no _allowed_type, so it takes anything

hand_rolled = TakeList()
hand_rolled.append(data.takes[0])    # type-checked on the way in
print(hand_rolled, len(hand_rolled))
```

Clean up when you are done — `SceneData.destroy()` tears down the FBX
manager and everything under it.

```python
data.destroy()
print(data.scene)                          # None
```

---

## `fbx` — FbxExporter

Writes a `HierarchyData` skeleton out as an FBX. One skeleton per exporter.

| Member | Type | Notes |
|---|---|---|
| `.manager` | `fbx.FbxManager` | lazily created, with IO settings |
| `.scene` | `fbx.FbxScene` | lazily created, Maya Y-up axis system |
| `.add_skeleton(component)` | — | `ValueError` if one is already set |
| `.export(path, as_ascii=False, zero_root=False)` | — | `path` may be a `str` or `Path` |
| `._ROTATE_ORDER_MAP` | `dict` | `HierarchyData` rotate-order → `EFbxRotationOrder` |

```python
from cgmath.formats.fbx import FbxExporter
from cgmath.hierarchy import HierarchyData, TransformData

skeleton = HierarchyData([
    TransformData("root", node_type="joint"),
    TransformData("spine", parent_node="root", translate=(0, 10, 0), node_type="joint"),
    TransformData("head", parent_node="spine", translate=(0, 10, 0), node_type="joint"),
])

exporter = FbxExporter()
print(type(exporter.manager).__name__, type(exporter.scene).__name__)
exporter.add_skeleton(skeleton)

skel_path = os.path.join(WORK, "skeleton.fbx")
exporter.export(skel_path)
print(os.path.exists(skel_path))
```

Read it straight back:

```python
check = SceneData(skel_path)
print(check.nodes)                   # ['root', 'spine', 'head']
check.destroy()
```

ASCII output, and the one-skeleton rule:

```python
ascii_exporter = FbxExporter()
ascii_exporter.add_skeleton(skeleton)
ascii_exporter.export(os.path.join(WORK, "skeleton_ascii.fbx"), as_ascii=True)

try:
    ascii_exporter.add_skeleton(skeleton)
except ValueError as error:
    print(str(error)[:30])
```

Each `TransformData` maps onto FBX like this:

| `node_type` | FBX attribute | Extras applied |
|---|---|---|
| `"joint"` | `FbxSkeleton` (`eRoot` for a root, else `eLimbNode`) | `joint_orient` → pre-rotation, `rotate_axis` → inverted post-rotation, `segment_scale_compensate` → inherit type |
| `"transform"`, `"space_transform"` | `FbxNull` | — |
| `"locator"` | `FbxMarker` | — |
| anything else | — | `RuntimeError` |

`user_defined_attributes` become real FBX properties, typed by their
`attributeType` (or `dataType`) key: `string`, `double`, `int`, `bool`,
`short`.

<!-- notest: zero_root reads skeleton_component.data.get_roots(), which a bare HierarchyData does not have -->
```python
exporter.export(skel_path, zero_root=True)
```

`zero_root=True` expects a pipeline skeleton **component** — an object with
a `.data` attribute holding the `HierarchyData`. A bare `HierarchyData`
raises `AttributeError`.

---

## Mesh loading lives in geometry

`formats` never hands you a `MeshData` (the one exception being
`GlbData.mesh_list`).

<!-- notest: illustrative import map; the loaders need real asset files -->
```python
from cgmath.geometry.mesh import load_fbx, load_glb          # [(MeshData, UVList), ...]
from cgmath.geometry.skin_weights import load_fbx as skin_fbx
from cgmath.geometry.skin_weights import load_glb as skin_glb

load_fbx("hero.fbx")
load_glb("hero.glb", scale_factor=100.0)
skin_glb("hero.glb", bind_matrices=False)                     # -> list[SkinData]
```

---

## Gotchas

| Gotcha | What to do |
|---|---|
| `import cgmath.formats.fbx` fails without the SDK | Wrap the import; `glb` only fails at call time |
| `SceneData.from_file` does not exist | `SceneData(path)` or `SceneData().load(path)` |
| `FbxExporter.save` / `.add_scene` do not exist | `.export(path)` / `.add_skeleton(component)` |
| `CurveData.keys` does not exist | `.times`, `.values`, `.frames`, `.key_count` |
| Setting `.times` wipes `.values` | Set `times` first, then `values` |
| `CurveList.delete` raises `TypeError` | Use `CurveData.clear()`; take/layer delete inherits the break |
| `set_binary` on a `GLTF2()` with no blob | `TypeError` — `binary_blob()` is `None`; load a real file first |
| `GlbData.mesh_list` is `None` | `trimesh` is not installed |
| `GlbData` mesh points look 100× too big | They are — trimesh metres are scaled to centimetres |
| `Primitive.triangles` is 1-D | `.reshape(-1, 3)` yourself |
| `JOINTS_0` slots look like node indices | They index `Skin.joints`, which then indexes the nodes |
| `Model.nodes` order surprises you | Use `Model.ordered_node_indexes` |
| `SceneData` leaks memory | Call `.destroy()` |

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
