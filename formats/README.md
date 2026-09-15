# `cgmath.formats` — FBX, GLB and USD file-format wrappers

Thin, numpy-facing wrappers over the two binary formats this library reads
and writes: **glTF/GLB** (via `pygltflib`) and **FBX** (via the Autodesk FBX
Python SDK), plus stage and prim helpers for **USD** (via `pxr`).

This package is **curve / take / layer / accessor** focused. It is what you
reach for to read animation off a file, walk a node hierarchy, decode a raw
accessor, or stash a payload blob inside a `.glb`.

For **mesh** loading go elsewhere — see [Where mesh loading lives](#where-mesh-loading-lives).

---

## The `__init__.py` re-exports nothing

`cgmath/formats/__init__.py` is a docstring. Import the submodules directly:

```python
from cgmath.formats.glb import load_model, GlbData
from cgmath.formats.fbx import SceneData, FbxExporter
from cgmath.formats.usd import prim as usd_prim, stage as usd_stage
```

`usd/` is a regular subpackage whose `__init__.py` is likewise only a
docstring, so import `prim` and `stage` from it directly.
`from cgmath.formats import *` gives you nothing.

---

## Files

| File | Holds |
|---|---|
| `glb.py` | glTF/GLB reading, accessor decode, payload blobs, `GlbData` |
| `fbx.py` | FBX scene/take/layer/curve editing, `SceneData`, `FbxExporter` |
| `usd/stage.py` | open / create / write stages, sublayer editing, a throw-away stage cache |
| `usd/prim.py` | prim search and type tests, duplication, references, inherits, API schemas, property copies |

---

## Optional dependencies — and how each fails

| Module | Needs | Missing behaviour |
|---|---|---|
| `glb` | `pygltflib` | Imports fine. Every entry point raises `RuntimeError("pygltflib is not installed")` |
| `glb` | `trimesh` (optional) | Imports fine. `GlbData.mesh_list` is `None` |
| `fbx` | Autodesk FBX Python SDK — not on PyPI, a manual install | Imports fine, with one `UserWarning` naming the SDK and where to get it. Every FBX call raises `RuntimeError` with that same message |
| `usd.prim`, `usd.stage` | `pxr`, from `usd-core` — installed with the wheel | Always present after `pip install`. Run straight off `sys.path` without it, **`import cgmath.formats.usd.prim` raises `ImportError`** — `from pxr import Usd` is unguarded |

So `glb` and `fbx` degrade at call time, while `usd` needs `pxr` at import
time — which the wheel always installs. Guard that import only if your caller
runs cgmath off `sys.path` without `usd-core`.

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

### fbx: names are translated to Maya conventions

`BaseData.name` rewrites FBX property names on the way out:

| FBX | Reported as |
|---|---|
| `pCube1.Lcl TranslationX` | `pCube1.translateX` |
| `pCube1.Lcl RotationY` | `pCube1.rotateY` |
| `pCube1.Lcl ScalingZ` | `pCube1.scaleZ` |
| `face.smile.DeformPercent` | `face.smile` |

`create_curve` translates the other way, so you pass Maya names in
(`"translateX"`) and get Maya names back. `CurveList` lookup accepts either
spelling.

### fbx: seconds and frames are both first class

Every time-bearing property comes in two flavours — `times`/`frames`,
`start_time`/`start_frame`, `duration`/`frame_count`. The frame variants
divide by `SceneData.fps` and return `None` when no scene (or no known fps)
is available.

### fbx: blendshape weights are rescaled

FBX stores `DeformPercent` in `0..100`. `CurveData.values` divides by 100 on
read and multiplies by 100 on write, so you work in `0..1` throughout.

---

## Quick taste

### Export a joint chain to FBX

```python
from cgmath.formats.fbx import FbxExporter
from cgmath.hierarchy import HierarchyData, TransformData

skeleton = HierarchyData([
    TransformData("root", node_type="joint"),
    TransformData("spine", parent_node="root", translate=(0, 10, 0), node_type="joint"),
    TransformData("head", parent_node="spine", translate=(0, 10, 0), node_type="joint"),
])

exporter = FbxExporter()
exporter.add_skeleton(skeleton)
import os, tempfile
exporter.export(os.path.join(tempfile.mkdtemp(), "skeleton.fbx"))
```

### Read animation curves back out

<!-- notest: no .fbx asset ships with this repo; CHEATSHEET.md builds one inline first -->
```python
from cgmath.formats.fbx import SceneData

scene = SceneData("hero.fbx")              # constructor loads; there is no from_file
for take in scene.takes:                   # TakeList
    for layer in take.layers:              # LayerList
        for curve in layer.curves:         # CurveList
            print(curve.name, curve.times, curve.values)
scene.destroy()
```

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
in `geometry`:

| Want | Call |
|---|---|
| Meshes + UVs from FBX | `cgmath.geometry.mesh.load_fbx(path)` → `[(MeshData, UVList), ...]` |
| Meshes + UVs from GLB | `cgmath.geometry.mesh.load_glb(path, scale_factor=100.0)` |
| One named mesh | `MeshData.load_fbx(path, name=...)` / `MeshData.load_glb(path)` |
| Skin weights | `cgmath.geometry.skin_weights.load_fbx / load_glb` |

The one exception is `GlbData.mesh_list`, which is a `trimesh`-powered
convenience on the GLB wrapper (points scaled ×100 — metres to centimetres).

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste every call, USD included | [`CHEATSHEET.md`](CHEATSHEET.md) — runnable, builds its own fixtures |
| The rest of the library | [`../README.md`](../README.md) |
