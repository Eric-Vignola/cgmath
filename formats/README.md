# `cgmath.formats` — FBX, GLB and USD file-format wrappers

Thin, numpy-facing wrappers over the two binary formats this library reads
and writes: **glTF/GLB** (via `pygltflib`) and **FBX** (via the Autodesk FBX
Python SDK), plus stage and prim helpers for **USD** (via `pxr`).

The fbx half is one read and one write: `load_fbx(path)` gives everything a
file holds that cgmath has a class for, as one `FbxContents` — the
hierarchy or its takes, the meshes on the nodes with their colours, their
UV sets, skins, materials and blend shape targets, the cameras and lights —
and
`save_fbx(path, ...)` writes the same fields back, in Maya's Y-up and cm.
Every other fbx reader in the library (`HierarchyData.load_fbx`,
`MeshData.load_fbx`, `SkinList.load_fbx`, `cgmath.render.Scene.load_fbx`
...) is a slice of that read. The glb half is **accessor** focused: decode a
raw accessor, walk a node hierarchy, stash a payload blob inside a `.glb`.
For mesh classes see [Where mesh loading lives](#where-mesh-loading-lives).

---

## The `__init__.py` re-exports the fbx read and write

```python
from cgmath.formats import FbxContents, load_fbx, save_fbx
from cgmath.formats.glb import load_model, GlbData
from cgmath.formats.fbx import FbxExporter
from cgmath.formats.usd import prim as usd_prim, stage as usd_stage
```

`usd/` is a regular subpackage whose `__init__.py` is only a docstring, so
import `prim` and `stage` from it directly.

---

## Files

| File | Holds |
|---|---|
| `_load.py` / `_write.py` | `load_fbx` / `save_fbx`, the fbx read and write of every cgmath class |
| `_fbx_io.py` | the one place an `FbxManager` is made: `open_fbx`, the sdk lookup, the media folder |
| `glb.py` | glTF/GLB reading, accessor decode, payload blobs, `GlbData` |
| `fbx.py` | `FbxExporter`, the writer `save_fbx` and `HierarchyData.save_fbx` stand on |
| `usd/stage.py` | open / create / write stages, sublayer editing, a throw-away stage cache |
| `usd/prim.py` | prim search and type tests, duplication, references, inherits, API schemas, property copies |

---

## Optional dependencies — and how each fails

| Module | Needs | Missing behaviour |
|---|---|---|
| `glb` | `pygltflib` | Imports fine. Every entry point raises `RuntimeError("pygltflib is not installed")` |
| `glb` | `trimesh` (optional) | Imports fine. `GlbData.mesh_list` is `None` |
| `fbx` | Autodesk FBX Python SDK — not on PyPI, a manual install | Imports fine, with one `UserWarning` naming the SDK and where to get it. The FBX readers and `save_fbx` raise `ImportError` at call time; an `FbxExporter`'s methods (all but `close`) and any call on `cgmath.formats.fbx`'s placeholder `fbx` object raise `RuntimeError`, with that same message |
| `usd.prim`, `usd.stage` | `pxr`, from `usd-core` (the `usd` extra) or the host DCC's own USD | **`import cgmath.formats.usd.prim` raises `ImportError`** — `from pxr import Usd` is unguarded |

So `glb` and `fbx` degrade at call time, while `usd` needs `pxr` at import
time. Get it with `pip install "cgmath[usd]"`, except inside a DCC that
already ships USD (Maya, Houdini): use the host's `pxr` there, since a second
copy from `usd-core` conflicts with it. Guard the import if your caller may
run without either.

---

## Concepts you need before calling anything

### glb: an accessor is not a buffer view

A glTF `bufferView` can carry several accessors, and an accessor can start
part way into one, stop short of its end, or be interleaved with a
neighbour on a `byteStride`. `load_accessor_data` handles all four; reading
the view directly does not. It also de-quantizes `normalized` integer
accessors (a `SHORT` rotation comes back as `float32` in `[-1, 1]`).

### glb: `JOINTS_0` indexes the skin, not the file

A vertex's `JOINTS_0` entry is an index into `Skin.joints`, which is
*itself* a list of node indices. Skipping that indirection still runs and
still produces plausible weights — bound to the wrong joints.

### glb: node order is not parent-first

glTF makes no ordering promise, so a child can precede its parent.
`order_nodes_root_first` returns an index order where every parent comes
before its children, and stamps a `parent_index` attribute on each node as
a side effect (`-1` for roots).

### fbx: everything comes back in Maya's axes and units

`load_fbx` reads a file whose up axis is X or Z the way Maya's importer
turns it, leaves a Y-up file in its own axes, refuses a left-handed one, and
converts the file's unit to centimetres: every cgmath value is in Maya's
Y-up, right-handed, cm. A take is a `ClipData` of one `ClipList`, keyed on
every frame, the file's active stack the active take.

### fbx: a shape names its node, a skin or a set names its mesh

A mesh's `parent_uuid` is the uuid of its node in the hierarchy (or the
clip) the same read gives; a UV set's, a skin's, a material's or a target's
`mesh_uuid` is its mesh's. The uuids are minted by the read: two loads of
one file do not share them. A skin's influences are joint names.

### fbx: blendshape weights are rescaled

FBX stores `DeformPercent` and a target's full weight in `0..100`.
`MorphData.weight` and `at` divide by 100 on read and multiply on write, so
you work in `0..1` throughout, as Maya's blendShape does.

---

## Quick taste

### Export a joint chain to FBX

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

import os, tempfile

with FbxExporter() as exporter:            # the block frees the FBX SDK's memory
    exporter.add_skeleton(skeleton)
    exporter.export(os.path.join(tempfile.mkdtemp(), "skeleton.fbx"))
```

### Read a file whole, write it back

<!-- notest: no .fbx asset ships with this repo -->
```python
from cgmath.formats import load_fbx, save_fbx

held = load_fbx("hero.fbx")                   # FbxContents: hierarchy, clip, meshes, skins, uvs, cameras, lights, materials, morphs
for mesh in held.meshes or []:
    print(mesh.name, mesh.points.shape, held.hierarchy[mesh.parent_uuid].name if held.hierarchy else None)
for take in held.clip or []:                  # a ClipList, one ClipData per take
    print(take.take, take.frame_count, take.fps)
save_fbx("copy.fbx", *held)                   # the same fields, the same file
```

The FBX SDK frees what it read only when told to; every FBX reader in
cgmath frees its memory before it returns, and none of them writes anything
next to the file. An `FbxExporter` holds its scene until `close()`; a `with`
block calls it.

Textures packed inside a file are extracted to a temporary folder, which
Maya's fbx binding can only name to the SDK by a plain ASCII path. A temp
folder whose path is not ASCII is reached by its Windows short (8.3) name;
only without one are the textures extracted beside the file, with a
`RuntimeWarning`.

### Pull a GLB apart

<!-- notest: no .glb asset ships with this repo; CHEATSHEET.md builds one inline first -->
```python
from cgmath.formats.glb import load_model

model = load_model("hero.glb")
print(model.name, len(model.nodes), len(model.meshes))
for animation in model.animations:
    print(animation.duration, len(animation.channels))
```

---

## usd: stages, prims and the things attached to them

`stage.py` is about files and layers, `prim.py` about what lives inside a
stage. Both take and return raw `pxr` objects — there is no wrapper type.

- **A stage cache is opt-out, not opt-in.** `open_stage`, `new_stage` and
  `new_stage_in_memory` register with USD's default stage cache unless you
  pass `no_cache=True` (a throw-away cache, cleared afterwards) or your own
  `cache=`. `no_stage_cache()` is the context manager behind `no_cache`.
- **`write_stage` chooses `Save` or `Export` for you.** No `file_path`, or one
  equal to the root layer's own path, saves the layer in place; any other path
  exports a copy.
- **Sublayer and reference paths are stored relative.** `add_sublayers` and
  `add_references` convert absolute paths to paths relative to the stage's
  directory (or `start_dir`); pass `replace=True` to overwrite the list.
  `clear_missing_sublayers` drops entries whose file no longer exists.
- **Searching is a generator.** `iter_prims` walks a stage or a prim by type,
  name or `fnmatch` path pattern and can prune whole branches through
  `skip_children_callback`; `find_prims` is the list form, or the first hit
  with `first_only=True`.
- **`duplicate_prim` copies authored attributes only** — values, metadata and
  custom data — never relationships or connections. `copy_properties` moves
  properties between two existing prims by name.

```python
import os
import tempfile

from pxr import UsdGeom

from cgmath.formats.usd import prim as usd_prim
from cgmath.formats.usd import stage as usd_stage

stage = usd_stage.new_stage_in_memory()
hero  = usd_prim.create_scopes(stage, "/World/Chars/Hero")      # Scope prims all the way down
body  = UsdGeom.Mesh.Define(stage, "/World/Chars/Hero/Body").GetPrim()
rig   = UsdGeom.Xform.Define(stage, "/World/Chars/Hero/Rig").GetPrim()
UsdGeom.Mesh(body).CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

print([p.GetPath().pathString for p in usd_prim.iter_prims(stage, prim_type="Mesh")])
print(usd_prim.find_prims(stage, prim_name="Rig", first_only=True).GetPath())
print(usd_prim.is_descendant(stage, "/World/Chars/Hero/Body", "/World/Chars"))

copy = usd_prim.duplicate_prim(body, stage, "/World/Chars/Hero/BodyCopy")
print(copy.GetTypeName(), UsdGeom.Mesh(copy).GetPointsAttr().Get())

usd_prim.add_api_schema(body, "SkelBindingAPI")
print(usd_prim.get_api_schemas(body))
```

Files, references and sublayers need a directory:

```python
root       = tempfile.mkdtemp()
asset_path = os.path.join(root, "asset.usda")
asset      = usd_stage.new_stage(asset_path)
UsdGeom.Xform.Define(asset, "/Asset")
usd_stage.write_stage(asset)                                    # Save(): the layer's own path

shot = usd_stage.new_stage(os.path.join(root, "shot.usda"))
usd_stage.add_sublayers(shot, asset_path)                      # stored relative to shot.usda
print(usd_stage.get_sublayers(shot), usd_stage.get_sublayers(shot, absolute=True))

ref = shot.DefinePrim("/Shot/Hero")
usd_prim.add_references(ref, [(asset_path, "/Asset")])         # (file, prim path)
print(usd_prim.get_references(ref, abspath=False))

reopened = usd_stage.open_stage(os.path.join(root, "shot.usda"), no_cache=True)
print(reopened.GetPrimAtPath("/Shot/Hero").IsValid())
```

---

## Where mesh loading lives

`formats` does **not** give you a `MeshData`. Mesh, UV and skin readers live
in `geometry` (a glb's `COLOR_0` comes back as the mesh's `colors`, one a
vertex, as trimesh holds it: 8 bits on a plain mesh, as the file has it on
a textured one):

| Want | Call |
|---|---|
| Everything an FBX holds | `cgmath.formats.load_fbx(path)` → `FbxContents`; `cgmath.render.Scene.load_fbx(path)` holds the same, ready to render |
| Meshes + UVs from FBX | `cgmath.geometry.mesh.load_fbx(path)` → `[(MeshData, UVList), ...]`, as `load_fbx` reads them, each mesh's `parent_uuid` naming its node |
| Meshes + UVs from GLB | `cgmath.geometry.mesh.load_glb(path, scale_factor=100.0)` |
| Meshes, any format | `cgmath.geometry.mesh.load(path)` / `MeshList.load(path)` |
| One named mesh | `MeshData.load_fbx(path, name=...)` / `MeshData.load_glb(path, name=...)` |
| Skin weights | `cgmath.geometry.skin_weights.load / load_fbx / load_glb` → `SkinList` |
| Hierarchy | `cgmath.hierarchy.load / load_fbx / load_glb` → `HierarchyData` from FBX and GLB; a cgmath file gives whatever hierarchy it holds (a `ClipData` stays a clip) |

The one exception is `GlbData.mesh_list`, which is a `trimesh`-powered
convenience on the GLB wrapper (points scaled ×100 — metres to centimetres).

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste every call, USD included | [`CHEATSHEET.md`](CHEATSHEET.md) — runnable, builds its own fixtures |
| The rest of the library | [`../README.md`](../README.md) |
