# `cgmath` — Maya's math, without Maya

A Python toolkit for what tech artists, riggers and pipeline engineers
actually need: SRT transforms with Maya's parenting rules, mesh topology
you can introspect, five deformers, a software raytracer, signed distance
fields, skin-weight transfer, and importers for OBJ / GLB / FBX / USD.

Everything is numpy-native. No DCC required. Numba-accelerated where it
matters.

```python
import numpy as np

from cgmath.geometry import MeshData
from cgmath.render import Object, Scene

cube = MeshData(
    points=np.array([
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
        [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
    ]),
    indices=np.array([0, 1, 3, 2,  2, 3, 5, 4,  4, 5, 7, 6,
                      6, 7, 1, 0,  1, 7, 5, 3,  6, 0, 2, 4]),
    counts=np.array([4, 4, 4, 4, 4, 4]),
)

scene = Scene("hero")
scene.append(Object(name="cube", mesh=cube))
frame = scene.render()
print(frame.array.shape)                       # (500, 500, 4) RGBA
```

That is a full render — the library brought its own camera, its own
3-point lighting rig sized to the bounding box, and its own resolution.

---

## Where to go next

Every subpackage has a **README** (concepts, conventions, when to reach
for it) and a **CHEATSHEET** (every public name, with a runnable example).

| Module | What it holds | |
|---|---|---|
| **this page** | conventions, the map, the quick taste | [CHEATSHEET](CHEATSHEET.md) — the 25 most common things, cross-cutting |
| **learning by doing** | ten walkthroughs that each build something complete | [TUTORIAL](TUTORIAL.md) |
| `transforms` (separate package) | matrices, quaternions, euler, axis-angle, vectors — as batched functions | [README](https://github.com/Eric-Vignola/transforms/blob/main/README.md) · [CHEATSHEET](https://github.com/Eric-Vignola/transforms/blob/main/CHEATSHEET.md) |
| `cgmath.hierarchy` | `TransformData` / `TransformList` / `HierarchyData` / `ClipData` — the scene graph | [README](hierarchy/README.md) · [CHEATSHEET](hierarchy/CHEATSHEET.md) |
| `cgmath.geometry` | `MeshData`, UVs, B-splines, SDFs, skin weights, morphs, maps, resampling | [README](geometry/README.md) · [CHEATSHEET](geometry/CHEATSHEET.md) |
| `cgmath.geometry.deform` | FFD, Delta Mush, patch relax, skinning, RBF wrap | [README](geometry/deform/README.md) · [CHEATSHEET](geometry/deform/CHEATSHEET.md) |
| `cgmath.geometry.utils` | the 59 numba kernels everything above stands on | [README](geometry/utils/README.md) · [CHEATSHEET](geometry/utils/CHEATSHEET.md) |
| `cgmath.render` | software raytracer, `Scene` / `Object` / `Camera` / `Light` / `Frame` | [README](render/README.md) · [CHEATSHEET](render/CHEATSHEET.md) |
| `cgmath.formats` | FBX + GLB curve / take / accessor I/O, USD stage and prim helpers | [README](formats/README.md) · [CHEATSHEET](formats/CHEATSHEET.md) |
| `cgmath.rbf` | 21 radial basis function kernels + a JIT LU solver | [README](rbf/README.md) · [CHEATSHEET](rbf/CHEATSHEET.md) |
| `cgmath.constraints` | `ProcrustesData` — rivet a transform to a deforming patch | [README](constraints/README.md) · [CHEATSHEET](constraints/CHEATSHEET.md) |

[`REFERENCE.md`](REFERENCE.md), a flat API listing, also sits in this
directory. It is generated from the live code by `gen_reference.py` (kept in
the `tools/` folder beside the packages, not in this repository): every public
name with its real signature and first docstring line.

---

## What's inside

`cgmath/__init__.py` is a docstring and re-exports nothing, so
`import cgmath` gives you nothing to call. Import from a subpackage.

```
cgmath/
├── utils.py               info() / docstring(), pretty_json, profile(), run_tests()
│
├── hierarchy/             TransformData, TransformList, HierarchyData, ClipData
│   ├── __init__.py        re-exports the public surface
│   └── hierarchy.py       the four types, the fbx / glb readers, the batched clip evaluator
│
├── geometry/              meshes, UVs, curves, fields, transfer
│   ├── _base.py           Data / DataList / ImmutableArray + serialization
│   ├── mesh.py            MeshData, MeshList, UVData, UVList, the loaders
│   ├── map.py morph_target.py skin_weights.py    MapData, MorphData, SkinData
│   ├── bspline.py bspline_patch.py _saddle_surface.py _cdt.py
│   ├── sdf.py             SDF primitives + dual marching cubes
│   ├── pack.py resample.py surface_plotting.py
│   ├── robust_skinweights_transfer_bilinear.py
│   ├── camera.py delta_mush.py ffd.py patch_relax.py raytracer.py texture.py
│   │                      one-release deprecation shims -- they warn and re-export
│   ├── deform/            FFDData, DeltaMushData, PatchRelaxData,
│   │                      SkinDeformData, WrapData
│   └── utils/             main.py + _numba/ -- the kernel floor
│
├── render/                scene.py raytracer.py camera.py frame.py texture.py
├── formats/               fbx.py, glb.py, usd/prim.py, usd/stage.py
├── rbf/                   _kernels.py + _numba/   (__init__.py is a docstring, on purpose)
└── constraints/           procrustes.py
```

The batched matrix / quaternion / euler / axis / vector functions that all
of this stands on live in the separate [`transforms`](https://github.com/Eric-Vignola/transforms)
package, which `cgmath` imports. Its own README and CHEATSHEET cover them.

The public surface of each subpackage is what its `__init__.py`
re-exports:

```python
from cgmath.hierarchy import TransformData, TransformList, HierarchyData, ClipData
from cgmath.geometry   import MeshData, MeshList, UVData, UVList
from cgmath.geometry   import BSplineData, BSplinePatchData, PatchSampleData
from cgmath.geometry   import MorphData, MorphList, MapData, GeomSubsetData
from cgmath.geometry   import SkinData, SkinList, CompactSkinData
from cgmath.geometry   import DeltaMushData, PatchRelaxData
from cgmath.geometry.deform import FFDData, SkinDeformData, WrapData, DeformMethod
from cgmath.render     import render, Scene, Object, Camera, Light, Frame, look_at
from cgmath.constraints import ProcrustesData
```

Note the asymmetry: `cgmath.geometry` re-exports only two of the five
deformers. `FFDData`, `SkinDeformData` and `WrapData` come from
`cgmath.geometry.deform`.

---

## Conventions, all in one place

- **Transforms are Maya row-major.** Translation lives at `M[3, :3]`,
  points are row vectors (`p' = p @ M`), and `matrix_multiply(A, B)`
  applies `A` first. Maya rotate orders:
  `XYZ=0 YZX=1 ZXY=2 XZY=3 YXZ=4 ZYX=5`. Axis constants `X=0 Y=1 Z=2`.
- **Quaternions are `(i, j, k, w)`** — scalar last. They compose in the
  *opposite* order from matrices: `quaternion_multiply(a, b)` matches
  `matrix_multiply(B, A)`.
- **Radians in functions, degrees on nodes.** Everything in
  `transforms.*` takes radians; `TransformData.rotate` and the
  angle thresholds on `MeshData` are degrees.
- **Everything is batched, with NumPy's rules.** A bare `(3,)` or `(4, 4)`
  is promoted to a stack of one, and every input must be as long as the
  longest one or exactly one, in which case it is reused for the whole batch.
  Mismatched lengths raise `ValueError`.
- **`-1` is the pad.** Every adjacency matrix is dense
  `(rows, max_width)` with short rows padded; `-1` also marks a missing
  face and a raycast miss.
- **A mesh is a face-vertex stream**, `indices` + `counts` + `points`.
  Mixed triangles, quads and n-gons in one mesh are normal.
- **Camera is OpenGL**: `-Z` forward, `+Y` up, `+X` right. `look_at()`
  and `Frame.camera_matrix` are **column**-major (eye at `M[:3, 3]`)
  while node matrices are row-major (eye at `M[3, :3]`).
- **Images** are RGBA `uint8` with pixel `(0, 0)` top-left. Colour tuples
  in the scene graph are `[0, 1]` floats; wireframe colours are `0..255`
  ints.
- **UV `v` increases up** — `v = 0` is the bottom of the texture
  (Maya / OpenGL).
- **Verbs mutate, `get_*` and `from_*` return.** `triangulate`,
  `subdivide`, `merge`, `smooth`, `pack` edit in place and drop the
  caches; `copy`, `from_faces`, `sample`, every `resample_*` hand back
  something new.
The first three are `transforms` conventions that cgmath inherits — full
detail in its [README](https://github.com/Eric-Vignola/transforms/blob/main/README.md).

### Optional dependencies

Imported lazily behind `try / except ImportError`, so the package loads
without them and only the paths that need them complain.

| Dependency | Used for | Missing behaviour |
|---|---|---|
| `pygltflib` | GLB / glTF read | `RuntimeError("pygltflib is not installed")` at call time |
| Autodesk FBX SDK | FBX read / write | `import cgmath.formats.fbx` itself raises `ImportError` |
| `pxr` | `cgmath.formats.usd` stage / prim helpers | `import cgmath.formats.usd.prim` itself raises `ImportError` |
| `trimesh` | `GlbData.mesh_list` | that attribute is `None` |
| `PIL` (Pillow) | `Frame.image` / `.save` / `.wireframe` / `.encode_gif`, `to_image`, texture I/O | `RuntimeError("... install Pillow.")` at call time |
| `cv2` | `imshow()` windows; UV rasterization in `geometry.mesh` | `imshow` raises `RuntimeError`; rasterization falls back to `skimage` |
| `skimage` | the `cv2` fallback for UV rasterization and image loading | `RuntimeError` only when neither it nor `cv2` is present |
| `pxr` | USD `from_prim` / `to_prim` / `load_usd` | `ImportError` at call time (`geometry.utils.pxr()` is the accessor) |
| `ffmpeg` / `gifski` on `PATH` | `encode_mp4` / `encode_gif` | `encode_gif` falls back to ffmpeg when gifski is absent; no ffmpeg is an error |

`numba` compiles on first call and caches to disk next to the sources
(`__pycache__/*.nbi`, `*.nbc`); set `NUMBA_CACHE_DIR` to move the cache.
Importing any subpackage is cheap; the first `sample()`, `subdivide()` or
`render()` in a fresh interpreter pays the JIT.

---

## Quick taste

### Move a point through a Maya world matrix

```python
from cgmath.hierarchy import TransformData
from transforms import matrix_point_multiply

node = TransformData("hero", translate=(0, 1, 0), rotate=(0, 90, 0))
print(node.world_matrix.round(6)[3, :3])                     # [0. 1. 0.]
print(matrix_point_multiply([1.0, 0.0, 0.0], node.world_matrix).round(6))
```

`world_matrix` is a **property** on every node type — never a call, never
parameterised by name. The local SRT matrix is `node.matrix`.

### Read mesh topology

```python
print(cube.point_count, cube.face_count, cube.edge_count)  # 8 6 12
print(cube.get_border_vertices(flatten=True))              # [] -- closed
print(cube.shell_faces)                                    # connected components
print(cube.e2v.shape, cube.ue2v.shape)                     # deduped vs raw edges
```

Counts are `point_count` / `face_count` / `edge_count`; the diagnostics
(`get_border_vertices`, `get_non_manifold_vertices`, `get_lamina_faces`,
`get_overlap_vertices`) are **methods**.

### Build a rig and pose it

```python
from cgmath.hierarchy import HierarchyData

rig = HierarchyData()
for name in ("root", "hip", "knee"):
    rig.append(TransformData(name, node_type="joint"))
rig["hip"].set_parent("root", world_space=False)
rig["knee"].set_parent("hip", world_space=False)
rig["hip"].translate  = (0.0, 10.0, 0.0)
rig["knee"].translate = (0.0, 10.0, 0.0)
rig["hip"].rotate_z   = 90.0

print(rig.world_matrix[:, 3, :3].round(6))
```

Parent with `set_parent(name)`. `parent_node` holds the parent's **uuid**,
so assigning a name to it directly leaves `get_children()` empty.

### Sweep an FFD lattice

```python
from cgmath.geometry.deform import FFDData

lattice = FFDData.create_lattice(
    (3, 3, 3),
    bbox_min = cube.points.min(axis=0) - 0.05,
    bbox_max = cube.points.max(axis=0) + 0.05,
)
ffd = FFDData.from_mesh(lattice, divisions=(3, 3, 3))   # the LATTICE, not the mesh
ffd.bind(cube)

posed = ffd.lattice.copy()
posed[:, -1, :, 0] += 0.5
print(np.abs(ffd.update(posed).points - cube.points).max().round(4))
```

### Transfer skin weights onto a denser mesh

```python
from cgmath.geometry import SkinData
from cgmath.geometry.resample import MeshDataResampler, ResampleMode

dense = cube.copy()
dense.subdivide(1)

y    = cube.points[:, 1] + 0.5
skin = SkinData(weights=np.column_stack([1.0 - y, y]), influences=["root", "tip"])

dst  = MeshDataResampler(cube, dense).resample_skin_weights(skin, mode=ResampleMode.SPATIAL)
print(dst.weights.shape, dst.influences, dst.valid)
```

---

## Audience

Written for tech artists, riggers and tools engineers who think in Maya
nodes but want clean Python they can ship to CI, a notebook or a headless
cluster — without launching a DCC.

If you have ever wanted to score a turntable for a Slack post, or to bind
skin weights to a USD prim from Python, you are in the right module.
Start with [`CHEATSHEET.md`](CHEATSHEET.md).


## Requirements

Numpy, Scipy, Numba, and the [transforms](https://github.com/Eric-Vignola/transforms) python modules.

Optional, per feature (see the table above for how each degrades):

    Pillow                  texture I/O and rendered frames
    pygltflib, trimesh      glTF / GLB read and write
    Autodesk FBX SDK        FBX read and write
    pxr (USD)               USD stage read and write
    scikit-image or OpenCV  UV rasterization


## Author

* **Eric Vignola** (eric.vignola@gmail.com)

If this was useful to you, [buy me a coffee](https://buymeacoffee.com/ericvignola) ☕


## License

BSD 3-Clause License: Copyright (c) 2026, Eric Vignola All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of copyright holders nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
