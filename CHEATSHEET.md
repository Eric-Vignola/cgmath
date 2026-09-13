# `cgmath` Cheatsheet — the cross-cutting quick start

The 24 things people actually reach for, one small runnable block each.
Every subpackage now has its own exhaustive cheatsheet; this page is the
map, not the territory. Each section ends with a pointer to the module
cheatsheet that covers the full surface.

For the concepts and conventions behind all of it, read
[`README.md`](README.md). For walkthroughs that build something complete,
read [`TUTORIAL.md`](TUTORIAL.md).

Every `python` block below runs, top to bottom, as one script — they
share a namespace, notebook style.

---

## Contents

| # | Section | Full surface |
|---|---|---|
| — | [Setup](#setup) | |
| 1 | [Where each name lives](#1-where-each-name-lives) | |
| 2 | [The conventions, in code](#2-the-conventions-in-code) | |
| 3 | [Build a rig, read world space](#3-build-a-rig-read-world-space) | [`hierarchy/`](hierarchy/CHEATSHEET.md) |
| 4 | [Animate it as a clip](#4-animate-it-as-a-clip) | [`hierarchy/`](hierarchy/CHEATSHEET.md) |
| 5 | [Retarget a pose](#5-retarget-a-pose) | [`hierarchy/`](hierarchy/CHEATSHEET.md) |
| 6 | [Build a mesh, count it](#6-build-a-mesh-count-it) | [`geometry/`](geometry/CHEATSHEET.md) |
| 7 | [Topology maps and diagnostics](#7-topology-maps-and-diagnostics) | [`geometry/`](geometry/CHEATSHEET.md) |
| 8 | [Normals and areas](#8-normals-and-areas) | [`geometry/`](geometry/CHEATSHEET.md) |
| 9 | [Edit a mesh, keep its UVs in step](#9-edit-a-mesh-keep-its-uvs-in-step) | [`geometry/`](geometry/CHEATSHEET.md) |
| 10 | [Closest point and raycast](#10-closest-point-and-raycast) | [`geometry/`](geometry/CHEATSHEET.md) |
| 11 | [Skin weights](#11-skin-weights) | [`geometry/`](geometry/CHEATSHEET.md) |
| 12 | [Pose a skinned mesh](#12-pose-a-skinned-mesh) | [`deform/`](geometry/deform/CHEATSHEET.md) |
| 13 | [Delta Mush a wobbly pose](#13-delta-mush-a-wobbly-pose) | [`deform/`](geometry/deform/CHEATSHEET.md) |
| 14 | [Sculpt with an FFD lattice](#14-sculpt-with-an-ffd-lattice) | [`deform/`](geometry/deform/CHEATSHEET.md) |
| 15 | [Wrap dense geometry to a cage](#15-wrap-dense-geometry-to-a-cage) | [`deform/`](geometry/deform/CHEATSHEET.md), [`rbf/`](rbf/CHEATSHEET.md) |
| 16 | [Transfer data onto a new topology](#16-transfer-data-onto-a-new-topology) | [`geometry/`](geometry/CHEATSHEET.md) |
| 17 | [Rivet a transform to a deforming patch](#17-rivet-a-transform-to-a-deforming-patch) | [`constraints/`](constraints/CHEATSHEET.md) |
| 18 | [CSG with signed distance fields](#18-csg-with-signed-distance-fields) | [`geometry/`](geometry/CHEATSHEET.md) |
| 19 | [Sample a B-spline](#19-sample-a-b-spline) | [`geometry/`](geometry/CHEATSHEET.md) |
| 20 | [Render a still](#20-render-a-still) | [`render/`](render/CHEATSHEET.md) |
| 21 | [Turntable and video](#21-turntable-and-video) | [`render/`](render/CHEATSHEET.md) |
| 22 | [Save and load anything](#22-save-and-load-anything) | [`geometry/`](geometry/CHEATSHEET.md) |
| 23 | [FBX and GLB](#23-fbx-and-glb) | [`formats/`](formats/CHEATSHEET.md) |
| 24 | [Pretty printing and docstrings](#24-pretty-printing-and-docstrings) | |

---

## Setup

No mesh assets ship with this package. Build the fixtures inline: a unit
cube of six quads, its UV set, and an open 5x5 quad grid.

```python
import os
import tempfile

import numpy as np

from cgmath.geometry import MeshData, UVData

tmp = tempfile.mkdtemp()

cube = MeshData(
    name="cube",
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

# One UV square per face; all six faces share the same four corners.
cube_uv = UVData(
    name    = "map1",
    points  = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    indices = np.tile([0, 1, 2, 3], 6),
    counts  = cube.counts,
)


def quad_grid(n=5, spacing=1.0):
    """An open n x n grid of quads in the XZ plane."""
    t   = np.arange(n) * spacing
    idx = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            idx += [a, a + 1, a + n + 1, a + n]
    return MeshData(
        name    = "grid",
        points  = np.array([[x, 0.0, z] for z in t for x in t]),
        indices = np.asarray(idx, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
    )


grid = quad_grid()
print(cube.point_count, cube.face_count, grid.face_count)
```

---

## 1. Where each name lives

`cgmath/__init__.py` is a docstring and re-exports nothing, so there is
nothing to import from `cgmath` itself. Always import from a subpackage. The
batched transform functions come from the separate `transforms` package;
building matrices, moving points and blending rotations are its territory —
see the [`transforms` CHEATSHEET](https://github.com/Eric-Vignola/transforms/blob/main/CHEATSHEET.md).

```python
from cgmath.hierarchy import HierarchyData, TransformData, TransformList, ClipData
from transforms import euler_to_matrix, matrix_point_multiply, quaternion_slerp
from cgmath.geometry import MeshData, MeshList, UVData, UVList
from cgmath.geometry import BSplineData, BSplinePatchData
from cgmath.geometry import MapData, GeomSubsetData, MorphData, MorphList
from cgmath.geometry import SkinData, SkinList, CompactSkinData
from cgmath.geometry import DeltaMushData, PatchRelaxData          # only these two
from cgmath.geometry.deform import FFDData, SkinDeformData, WrapData, DeformMethod
from cgmath.geometry.sdf import DMCField, SDFSphere, SDFBox, SDFCylinder
from cgmath.render import render, Scene, Object, Camera, Light, Frame, look_at, load_texture
from cgmath.constraints import ProcrustesData
from cgmath.formats.fbx import FbxExporter, SceneData
from cgmath.formats.glb import GlbData, load_model
from cgmath.rbf._kernels import get_kernel, list_kernels
from cgmath.utils import info, docstring, pretty_json
```

Three easy-to-miss facts:

| Fact | Why it matters |
|---|---|
| `cgmath.geometry` re-exports only `DeltaMushData` and `PatchRelaxData` from `deform` | `FFDData`, `WrapData`, `SkinDeformData` need `cgmath.geometry.deform` |
| `cgmath.formats.__init__` is a docstring only | import `cgmath.formats.fbx` / `.glb` / `.usd.prim` / `.usd.stage`; `from cgmath.formats import *` gives you nothing |
| `cgmath.rbf.__init__` is a docstring only, on purpose | import `cgmath.rbf._kernels`; the underscore names the file, not the visibility |

---

## 2. The conventions, in code

Row-major matrices everywhere (Maya's convention). Translation is a **row**, points are
**row vectors**, and `matrix_multiply(A, B)` applies `A` first.

Units split at the node boundary: the free functions are **radians**,
`TransformData.rotate` is **degrees**.

```python
node = TransformData("hero", rotate=(0.0, 45.0, 0.0))     # degrees
from transforms import matrix_to_euler
print(np.degrees(matrix_to_euler(node.matrix, 0)).round(4))   # [[0. 45. 0.]]
```

| Convention | Value |
|---|---|
| Matrix | row-major `(N, 4, 4)`, translation at `M[3, :3]`, `p' = p @ M` |
| Quaternion | `(i, j, k, w)` — scalar **last** |
| Rotate order | `XYZ=0 YZX=1 ZXY=2 XZY=3 YXZ=4 ZYX=5`; axes `X=0 Y=1 Z=2` |
| Angles | radians in `transforms.*` functions, degrees on node attributes |
| Camera | OpenGL: `-Z` forward, `+Y` up. `look_at()` returns a **column**-major c2w |
| Image | pixel `(0, 0)` top-left, RGBA `uint8` |
| UV | `v` increases up; `v = 0` is the bottom of the texture |
| Padding | `-1` marks "no entry" in every adjacency matrix and raycast miss |

The matrix, quaternion, rotate-order and angle rows are `transforms`
conventions that cgmath inherits — full detail in its
[README](https://github.com/Eric-Vignola/transforms/blob/main/README.md).

`look_at()` and node matrices disagree on layout, on purpose — one is
OpenGL, the other is Maya:

```python
from cgmath.render import look_at

c2w = look_at(eye=(0, 0, 5), target=(0, 0, 0))
print(c2w[:3, 3])                                               # [0 0 5] -- column-major, eye in column 3
print(TransformData("cam", translate=(0, 0, 5)).matrix[3, :3])  # [0 0 5] -- row 3
```

---

## 3. Build a rig, read world space

`world_matrix` is a property on nodes and containers alike.

```python
rig = HierarchyData()
for name in ("root", "hip", "knee"):
    rig.append(TransformData(name, node_type="joint"))
rig["hip"].set_parent("root", world_space=False)
rig["knee"].set_parent("hip", world_space=False)
rig["hip"].translate  = (0.0, 10.0, 0.0)
rig["knee"].translate = (0.0, 10.0, 0.0)

print(rig.world_matrix.shape)      # (3, 4, 4)
print(rig.world_matrix[:, 3, :3])  # [[0 0 0] [0 10 0] [0 20 0]]

rig["hip"].rotate_z = 90.0                 # the branch follows
print(rig["knee"].world_matrix[3, :3].round(6))     # [-10. 10. 0.]
```

Parenting is by **uuid**. Use `set_parent(name)` — passing a name straight
to `parent_node=` builds a rig whose `get_children()` comes back empty.

```python
print(rig["hip"].parent_node == rig["root"].uuid)   # True
print([n.name for n in rig["root"].get_branch()])   # ['root', 'hip', 'knee']
print([n.name for n in rig["hip"].get_children()])  # ['knee']
print(rig.match("*e*").name)                        # fnmatch selection
```

Full surface: [`hierarchy/CHEATSHEET.md`](hierarchy/CHEATSHEET.md).

---

## 4. Animate it as a clip

A `ClipData` is a `HierarchyData` holding `F` poses. `clip.frames` writes
whole channels at once; `clip.frame = n` scrubs the nodes to one pose.

```python
clip                         = ClipData(rig, frames=8, start_frame=1001, fps=24.0)
clip.frames["knee"].rotate_z = np.linspace(0.0, 90.0, 8).reshape(8, 1)

print(clip.frames.world_matrix.shape)      # (8, 3, 4, 4) -- every frame, batched
clip.frame = 4                             # scrub; the nodes now hold frame index 4
print(round(float(clip["knee"].rotate_z), 4))
print(clip.frame_count, clip.start_frame, clip.end_frame, clip.fps)
```

`clip.frame` is a **0-based index** into the block, not a timeline frame
number — `start_frame` / `end_frame` / `fps` are the timebase metadata
that rides alongside it.

---

## 5. Retarget a pose

`a - b` (`get_delta`) is a retargeting operator, not a subtraction: the
delta is fixed to the bone, so it re-poses `b` at any pose.
`(a - b) + b == a`.

```python
tall           = HierarchyData([node.copy() for node in rig])
tall.translate = tall.translate * 1.3        # longer bones, same topology

posed                 = HierarchyData([node.copy() for node in rig])
posed["hip"].rotate_x = 25.0

result = posed.add_delta(tall.get_delta(rig, translate=False), translate=False)
print(result.name)
```

---

## 6. Build a mesh, count it

A `MeshData` is a face-vertex stream — `indices` + `counts` + `points` —
not a triangle array. Mixed tris, quads and n-gons are normal.

```python
print(cube.point_count, cube.face_count, cube.edge_count)   # 8 6 12
print(cube.closed, cube.triangles, cube.quads, cube.ngons)  # True 0 6 0
print(round(cube.area, 4))                                  # 6.0
```

There is no `num_points`. `triangles` / `quads` / `ngons` are **counts**;
`get_triangles()` / `get_quads()` / `get_ngons()` are the index lists.

Full surface: [`geometry/CHEATSHEET.md`](geometry/CHEATSHEET.md).

---

## 7. Topology maps and diagnostics

Every adjacency map is lazy, cached and `-1` padded.

```python
print(cube.f2v.shape, cube.v2f.shape)                    # face->vert, vert->face
print(cube.e2v.shape, cube.ue2v.shape)                   # (12, 2) deduped vs (24, 2) raw
print(cube.e2v.tolist() == cube.ue2v[cube.ue].tolist())  # True
row = cube.v2f[0]
print(row[row != -1])                      # the standard un-pad idiom
```

The diagnostics are **methods**, not properties, and `valence` is the
mesh's *maximum*, not the per-vertex array.

```python
print(grid.get_border_vertices(flatten=True).size)        # 16 on a 5x5 open grid
print(grid.get_border_edges(flatten=True).size)           # 16
print(grid.get_non_manifold_vertices())                   # []
print(grid.get_lamina_faces(), grid.get_overlap_vertices())
print(grid.valence,            grid.get_valence().shape)  # 4 (a scalar) vs (25,)
print(cube.shell_count,        len(cube.shell_faces))     # 1 1
```

---

## 8. Normals and areas

Geometric normals are derived from `points` on demand. There is no
`get_normals()`.

```python
print(cube.get_vertex_normals().shape, cube.get_face_normals().shape)
print(cube.get_face_areas().round(4))
print(cube.get_edge_lengths().round(4)[:4])
```

Shading normals are a separate, storable field:

```python
hard = cube.copy()
hard.set_normals(hard_edge_angle=30.0)
print(hard.normals.shape, hard.normal_indices is not None)
```

---

## 9. Edit a mesh, keep its UVs in step

A mesh and its UV set share face topology but not vertex ids, so every
paired edit goes through a face-local **rules** object handed to both.

```python
rules = cube.get_triangulate_rules()          # face-local, no vertex ids
mesh, uv = cube.copy(), cube_uv.copy()
mesh.triangulate(rules)
uv.triangulate(rules)
print(mesh.face_count, uv.face_count, mesh.triangles)   # 12 12 12
```

`triangulate` takes rules, not keyword arguments. The same pattern covers
`get_subset_rules` / `from_vertices` and
`get_quadrangulate_rules` / `quadrangulate`.

Verbs mutate in place and drop the caches; `from_*` / `get_*` / `copy`
return something new.

```python
work = cube.copy()
work.subdivide(1)
print(work.point_count, work.face_count)      # 26 24
front = cube.from_faces(np.array([0, 1]))
print(front.face_count, front.point_count)
```

---

## 10. Closest point and raycast

Quads are integrated as genuine bilinear patches, not two triangles.
Misses are `NaN`, never exceptions — always filter by `hit`.

```python
sample = cube.sample(np.array([[0.0, 0.0, 1.5], [0.0, 0.2, 0.0]]))
print(sample.projections.round(3))
print(sample.distances.round(3), sample.indices, sample.occluded)
print(sample(cube.points * 10.0).round(3))    # push any per-vertex stream through

hit = cube.raycast(np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]))
print(hit.hit, hit.distances.round(3), hit.indices)
```

---

## 11. Skin weights

`SkinData.weights` is **dense** `(V, I)` paired with an `influences` list
of joint names. `valid` checks that rows sum to 1 and the widths agree.

```python
from cgmath.geometry import SkinData

y    = cube.points[:, 1] + 0.5
skin = SkinData(weights=np.column_stack([1.0 - y, y]), influences=["root", "tip"])
print(skin.weights.shape, skin.influences, skin.valid)

compact = skin.to_compact_skin_data()          # top-k form, for disk and USD
print(compact.max_influences, compact.weights.shape)
print(np.allclose(compact.to_skin_data().weights, skin.weights))
```

Sparse per-point offsets and painted maps are their own types:

```python
from cgmath.geometry import GeomSubsetData, MapData, MorphData

morph   = MorphData("smile", offsets=np.array([[0.0, 0.2, 0.0]] * 2), indices=np.array([2, 3]))
density = MapData(name="density", indices=np.array([0, 1]), values=np.array([0.25, 1.0]))
head    = GeomSubsetData(name="head", indices=np.array([0, 1]), component_type="f")
print(morph.offsets.shape, density.values, head.component_type)
```

---

## 12. Pose a skinned mesh

Skin *deformation* lives in `deform`, not with `SkinData`.

```python
from cgmath.geometry.deform import SkinDeformData

bind_rig = HierarchyData([
    TransformData("root", node_type="joint"),
    TransformData("tip", parent_node="root", translate=(0, 1, 0), node_type="joint"),
])
posed_rig = HierarchyData([
    TransformData("root", node_type="joint"),
    TransformData("tip", parent_node="root", translate=(0, 1, 0),
                  rotate=(0, 0, 45), node_type="joint"),
])

deformer = SkinDeformData(mesh=cube, skin=skin, bind_rig=bind_rig)
bent     = deformer.apply(posed_rig, cube)
print(type(bent).__name__, np.abs(bent.points - cube.points).max().round(4))
```

Mesh in, mesh out — the input is never mutated. Always deform the **bind
pose**; feeding a result back in compounds the deformation.

Full surface: [`geometry/deform/CHEATSHEET.md`](geometry/deform/CHEATSHEET.md).

---

## 13. Delta Mush a wobbly pose

Constructor takes the **rest** geometry; `apply()` takes the deformed one
and auto-binds on the first call.

```python
from cgmath.geometry.deform import DeltaMushData

mush    = DeltaMushData(cube, smooth_iterations=10, smooth_step_size=0.5, pin_borders=True)
cleaned = mush.apply(bent)
print(type(cleaned).__name__, cleaned.point_count)
```

There is no `iterations=` or `step=` argument, and `bind()` takes none.

---

## 14. Sculpt with an FFD lattice

`from_mesh` takes the **lattice**, not the mesh being deformed. Build the
lattice around the mesh's bounding box first.

```python
from cgmath.geometry.deform import FFDData

lattice_mesh = FFDData.create_lattice(
    (3, 3, 3),
    bbox_min = cube.points.min(axis=0) - 0.05,
    bbox_max = cube.points.max(axis=0) + 0.05,
)
ffd = FFDData.from_mesh(lattice_mesh, divisions=(3, 3, 3))
ffd.bind(cube)                                 # FFD is the one that will not auto-bind

posed = ffd.lattice.copy()                     # (lx, ly, lz, 3)
posed[:, -1, :, 0] += 0.5                      # push the max-Y plane in +X
sheared = ffd.update(posed)
print(np.abs(sheared.points - cube.points).max().round(4))
```

---

## 15. Wrap dense geometry to a cage

Source, target and radius are set through **methods**; the constructor
takes only `kernel` and `name`. The kernel is `"thin_plate_spline"` —
`"thin_plate"` is not a registered name.

```python
from cgmath.geometry.deform import WrapData

cage = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
wrap = WrapData(kernel="thin_plate_spline")
wrap.set_source(cage)
wrap.set_target(cage + np.array([1.0, 2.0, 3.0]))
print(wrap.deform(cage).round(3))
```

The 21 kernels behind it are a standalone shelf:

```python
from cgmath.rbf._kernels import get_kernel, list_kernels

print(len(list_kernels()), "thin_plate_spline" in list_kernels())
D = np.array([[0.0, 1.0], [1.0, 0.0]])
print(get_kernel("gaussian")(D, epsilon=2.0).round(4))
```

Watch the shape parameter: `epsilon` **multiplies** the distance in
`gaussian` / `multiquadric` / `inverse_multiquadric` / `inverse_quadratic`
(bigger = narrower) and **divides** it in the other ten (bigger = wider).

---

## 16. Transfer data onto a new topology

`MeshDataResampler` never changes the destination topology — it re-indexes
a source-shaped array onto the destination's vertices. Build one resampler,
call it many times; the correspondence is cached per `(mode, method)`.

```python
from cgmath.geometry.resample import MeshDataResampler, ResampleMode

dense = cube.copy()
dense.subdivide(1)                             # same shape, different topology

resampler = MeshDataResampler(cube, dense)
dst_skin  = resampler.resample_skin_weights(skin, mode=ResampleMode.SPATIAL)
print(dst_skin.weights.shape, dst_skin.influences, dst_skin.valid)
```

| Mode | Correspondence | Needs |
|---|---|---|
| `SPATIAL` | closest point on the source surface | nothing |
| `UV` | matching UV coordinates | `src_uv` **and** `dst_uv` |
| `ROBUST_BILINEAR` | closest surface + Laplacian inpaint, quad-native | nothing |

---

## 17. Rivet a transform to a deforming patch

`ProcrustesData(target)` takes one positional argument. Attach transforms
to clusters of point indices, then feed it the deformed cloud.

```python
from cgmath.constraints import ProcrustesData
from transforms import matrix_identity as ident

proc = ProcrustesData(cube)                    # any (N, 3) or anything with .points
proc.attach(ident(1)[0], [0, 1, 2, 3])  # front face
proc.attach(ident(1)[0], [4, 5, 6, 7])  # back face

moved = cube.points.copy()
moved[[4, 5, 6, 7]] += (0.0, 0.0, -1.0)
proc.update(moved)
print(proc.translate.round(4))                 # only the back rider moved
print(proc.matrix.shape, proc.scale.round(4)[:, 0])
```

Rotation, uniform scale and translation only — that is what "orthogonal
Procrustes" means. `compute()` raises until something is attached.

---

## 18. CSG with signed distance fields

The OOP layer: primitives subclass `TransformData`, so they carry the full
SRT surface. `resolution` is *subdivisions per world unit*.

```python
from cgmath.geometry.sdf import DMCField, SDFBox, SDFSphere

blob = (
    DMCField(resolution=8)
    .add(SDFSphere(radius=1.0))
    .subtract(SDFBox(half_extents=[0.4, 0.4, 2.0]))
)
out = blob.mesh_data                           # lazy; quad-only (dual marching cubes)
print(out.point_count, out.face_count, out.quads == out.face_count)
```

The functional pipeline is `make_grid` -> `eval_*` -> `sdf_*` ->
`dual_marching_cubes`:

```python
from cgmath.geometry.sdf import dual_marching_cubes, eval_sphere, make_grid, sdf_union

X, Y, Z, origin, spacing = make_grid(
    (24, 24, 24), bounds=(np.array([-1.5] * 3), np.array([1.5] * 3)))
field  = sdf_union(eval_sphere(X, Y, Z, 1.0), eval_sphere(X - 0.8, Y, Z, 0.6))
carved = dual_marching_cubes(field, grid_origin=origin, grid_spacing=spacing)
print(carved.point_count, carved.face_count)
```

There is no `SDFPlane` and no `eval_plane`, and nothing in `sdf` is
re-exported at `cgmath.geometry` level.

---

## 19. Sample a B-spline

`u` runs `[0, max_param]` in knot space unless you pass `uniform=True`,
which switches to arc length on `[0, 1]`.

```python
from cgmath.geometry import BSplineData

cvs   = np.array([[0.0, 0, 0], [1, 2, 0], [2, -2, 0], [3, 0, 0], [4, 1, 0]])
curve = BSplineData(points=cvs, degree=3, periodic=False, uniform=True)

points, tangents = curve.compute(np.linspace(0.0, 1.0, 5))[:2]
print(points.round(3))
print(curve.sample(np.array([[2.0, 2.0, 0.0]])).distances.round(3))   # closest point
```

`points` is held **by reference**. After editing the array in place call
`invalidate()` or `rebuild()`.

---

## 20. Render a still

Build a `Scene`, append nodes, render. `append()` returns `None`, so it
does **not** chain — only `configure()` does.

```python
from cgmath.render import Camera, Light, Object, Scene

scene = Scene("hero")
scene.append(Object(name="cube", mesh=cube, uv=cube_uv, base_color=(0.8, 0.5, 0.3)))
scene.configure(resolution=(160, 120), samples_per_pixel=1, return_depth=True)

frame = scene.render()
print(frame.array.shape, frame.array.dtype, frame.has_depth)
frame.save(os.path.join(tmp, "hero.png"))
```

With no camera and no light the renderer supplies an autofit 3/4 view and
a 3-point rig sized to the bounding box. Supply your own when you care:

```python
scene.append(Camera(name="cam", angle_of_view=35.0, translate=(2, 2, 3)))
scene["cam"].look_at((0, 0, 0))
scene.append(Light(name="key", kind="infinite", intensity=1.5, rotate=(-35, 30, 0)))
wired = scene.render().wireframe(scene, color=(255, 255, 255, 255), width=1)
print(wired.array.shape)
```

Every frame is RGBA, so a `.jpg` save needs a convert first:

```python
frame.image.convert("RGB").save(os.path.join(tmp, "hero.jpg"), quality=95)
```

`configure()` accepts exactly eight keys — `resolution`,
`samples_per_pixel`, `background`, `autofit`, `default_light`,
`return_depth`, `angle_of_view`, `fit_padding`. Anything else is a plain
attribute or a constructor argument.

Full surface: [`render/CHEATSHEET.md`](render/CHEATSHEET.md).

---

## 21. Turntable and video

One call orbits the camera and encodes. The extension picks the writer;
an `{frame}` token writes an image sequence instead.

```python
spin = Scene("spin")
spin.append(Object(name="cube", mesh=cube))
spin.configure(resolution=(96, 96), samples_per_pixel=1)

files = spin.turntable(os.path.join(tmp, "spin_{frame:03d}.png"), n_frames=4)
print(len(files))
movie = spin.turntable(os.path.join(tmp, "spin.mp4"), n_frames=6, fps=12)
print(os.path.basename(movie), os.path.getsize(movie) > 0)
```

`Frame.encode_mp4` / `encode_avi` / `encode_gif` / `encode` take a list of
frames directly when you want to build the motion yourself. `mp4` and
`gif` need `ffmpeg` on `PATH`.

---

## 22. Save and load anything

Every `Data` subclass — meshes, UVs, skins, morphs, maps, rigs, clips —
shares one persistence surface. The extension picks the writer; the file's
first bytes pick the reader.

```python
cube.save(os.path.join(tmp, "cube.npz"))
cube.save(os.path.join(tmp, "cube.json"))
cube.save(os.path.join(tmp, "cube.pkl"))

for ext in ("npz", "json", "pkl"):
    print(ext, MeshData.load(os.path.join(tmp, f"cube.{ext}")) == cube)
print(sorted(cube.to_dict()))                  # declared fields only -- no caches
```

Caches are never persisted and `name` is excluded from equality. Public
attributes are strict, so a typo fails loudly:

```python
try:
    cube.pointz = np.zeros((8, 3))
except AttributeError as err:
    print(type(err).__name__)
```

Rigs and clips add JSON and FBX of their own:

```python
clip.save(os.path.join(tmp, "clip.json"))
reloaded = ClipData.load(os.path.join(tmp, "clip.json"))
print(reloaded.name, len(reloaded.frames))
```

---

## 23. FBX and GLB

`cgmath.formats` is curve / take / layer focused. **Mesh** loading lives
in `geometry`.

```python
from cgmath.formats.fbx import FbxExporter

exporter = FbxExporter()
exporter.add_skeleton(rig)                     # a HierarchyData of joints
exporter.export(os.path.join(tmp, "rig.fbx"))  # not .save(); no file_format kwarg

back = HierarchyData.load_fbx(os.path.join(tmp, "rig.fbx"))
print(back.name, back.world_matrix[:, 3, :3].round(3).tolist())
```

`HierarchyData.save_fbx` is the shortcut for the same trip:

```python
rig.save_fbx(os.path.join(tmp, "rig2.fbx"))
print(HierarchyData.load_fbx(os.path.join(tmp, "rig2.fbx")).name)
```

| Want | Call |
|---|---|
| Meshes + UVs | `cgmath.geometry.mesh.load_obj / load_glb / load_fbx / load_usd` -> `[(MeshData, UVList), ...]` |
| One mesh | `MeshData.load_obj / load_glb / load_fbx` |
| Skin weights | `cgmath.geometry.skin_weights.load_fbx / load_glb` |
| Animation curves | `cgmath.formats.fbx.SceneData(path)` — the constructor loads; there is no `from_file` |
| Raw glTF | `cgmath.formats.glb.load_model(path)` -> `Model(name, nodes, ordered_node_indexes, meshes, animations, skins)` |

OBJ is pure python and always available:

```python
cube.save_obj(os.path.join(tmp, "cube.obj"))
print(MeshData.load_obj(os.path.join(tmp, "cube.obj")).point_count)
```

Full surface: [`formats/CHEATSHEET.md`](formats/CHEATSHEET.md).

---

## 24. Pretty printing and docstrings

```python
from cgmath.utils import docstring, info, pretty_json

print(info(MeshData).splitlines()[0])  # docstring() is the same function
print(docstring is info)
print(pretty_json({"m": np.eye(2)}))   # decimal-aligned, numpy-aware
```

`cgmath.utils` also ships `profile(cmd, n=1)` for quick timings and a
`json` shim whose `dumps` / `dump` produce the same aligned output.

---

## Where to go next

| Module | Read |
|---|---|
| `cgmath` overview and conventions | [`README.md`](README.md) |
| Ten walkthroughs that build something complete | [`TUTORIAL.md`](TUTORIAL.md) |
| `transforms` | [README](https://github.com/Eric-Vignola/transforms/blob/main/README.md) · [CHEATSHEET](https://github.com/Eric-Vignola/transforms/blob/main/CHEATSHEET.md) |
| `cgmath.hierarchy` | [README](hierarchy/README.md) · [CHEATSHEET](hierarchy/CHEATSHEET.md) |
| `cgmath.geometry` | [README](geometry/README.md) · [CHEATSHEET](geometry/CHEATSHEET.md) |
| `cgmath.geometry.deform` | [README](geometry/deform/README.md) · [CHEATSHEET](geometry/deform/CHEATSHEET.md) |
| `cgmath.geometry.utils` | [README](geometry/utils/README.md) · [CHEATSHEET](geometry/utils/CHEATSHEET.md) |
| `cgmath.render` | [README](render/README.md) · [CHEATSHEET](render/CHEATSHEET.md) |
| `cgmath.formats` | [README](formats/README.md) · [CHEATSHEET](formats/CHEATSHEET.md) |
| `cgmath.rbf` | [README](rbf/README.md) · [CHEATSHEET](rbf/CHEATSHEET.md) |
| `cgmath.constraints` | [README](constraints/README.md) · [CHEATSHEET](constraints/CHEATSHEET.md) |
