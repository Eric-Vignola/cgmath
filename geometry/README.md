# `cgmath.geometry` — meshes, curves, fields and transfer

The geometry half of [`cgmath`](../README.md). Polygonal meshes you can
introspect (topology, borders, shells, normals), UV sets that stay in lockstep
with them, B-spline curves and patches, signed distance fields, skin weights,
morph targets, painted maps — plus the machinery to move any of that from one
topology onto another.

Numpy-native, numba-accelerated, no DCC required. Everything is a plain
dataclass that pickles, JSONs, zips and compares by value.

```python
from cgmath.geometry.sdf import DMCField, SDFBox, SDFSphere

blob = (
    DMCField(resolution=10)
    .add(SDFSphere(radius=1.0))
    .subtract(SDFBox(half_extents=[0.4, 0.4, 2.0]))
)
print(blob.mesh_data.point_count, blob.mesh_data.face_count)
```

A CSG stack in, a quad `MeshData` out.

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste an example of every class, function and property | [`CHEATSHEET.md`](CHEATSHEET.md) — runnable top to bottom |
| Deform something (FFD, Delta Mush, patch relax, skinning, wrap) | [`deform/README.md`](deform/README.md) |
| Reach a numba kernel without the object around it | [`utils/README.md`](utils/README.md) |
| See the rest of the library | [`../README.md`](../README.md) |

---

## What's inside

| File | Contents |
|---|---|
| `__init__.py` | re-exports 22 names — the package's public surface |
| `_base.py` | `Data`, `DataList`, `ImmutableArray` and the serialization primitives |
| `mesh.py` | `MeshData`, `MeshList`, `UVData`, `UVList`, `TriangulateMethod` / `TriangulateRules`, `AreaMethod`, `Axis`, `SampleMethod`, `RenderMethod`, `load_obj` / `load_glb` / `load_fbx` / `load_usd` / `save_obj` |
| `map.py` | `MapData` (painted scalar map), `GeomSubsetData` (component selection) |
| `morph_target.py` | `MorphData`, `MorphList` — sparse point offsets |
| `skin_weights.py` | `SkinData`, `SkinList`, `CompactSkinData`, `Patterns`, FBX / GLB readers |
| `bspline.py` | `BSplineData` and its `SampleData` |
| `bspline_patch.py` | `BSplinePatchData`, `PatchSampleData`, `PatchRaycastData` |
| `_saddle_surface.py` | `sample`, `raycast`, `integrate` over bilinear / PN-Quad patches |
| `_cdt.py` | constrained Delaunay triangulation, the n-gon path of `triangulate` |
| `sdf.py` | `make_grid`, `eval_*`, `sdf_*`, `dual_marching_cubes`, `SDFSphere` / `SDFBox` / `SDFCylinder`, `DMCField` |
| `pack.py` | `pack_islands` — the UV shell packer behind `UVData.pack` |
| `resample.py` | `MeshDataResampler`, `ResampleMode`, `SpatialSkinTransferOptions`, `UVSkinTransferOptions` |
| `robust_skinweights_transfer_bilinear.py` | robust weight inpainting, quad-native, no dependency |
| `surface_plotting.py` | rig retargeting through UV surface frames |
| [`deform/`](deform/README.md) | `FFDData`, `DeltaMushData`, `PatchRelaxData`, `SkinDeformData`, `WrapData` |
| [`utils/`](utils/README.md) | the 59 numba kernels every one of the above stands on |

`camera.py`, `delta_mush.py`, `ffd.py`, `patch_relax.py`, `raytracer.py` and
`texture.py` are one-release deprecation shims. They warn on import and
re-export from `cgmath.geometry.deform.*` or `cgmath.render.*`.

---

## Concepts

### Everything is a `Data` dataclass

`MeshData`, `UVData`, `MapData`, `GeomSubsetData`, `MorphData`, `SkinData` and
`CompactSkinData` all derive from `Data`, a `@dataclass` with managed
serialization. The declared fields *are* the object: `to_dict`, `from_dict`,
`to_json`, `to_bytes`, `save`, `load`, `copy`, `__eq__` and `__reduce__`
(pickle) read exactly those fields and nothing else.

- **Caches are never persisted.** A `MeshData` memoises a dozen topology
  matrices on private `_`-prefixed attributes. They are excluded from every
  format, so a reloaded object starts cold and recomputes lazily.
  `reset_cached_data()` clears them by hand.
- **`name` is excluded from equality.** Every type lists it in
  `EQUALITY_TEST_IGNORE`, so two identically shaped meshes with different names
  compare equal.
- **`save` / `load` are format-agnostic.** The extension picks the writer
  (`.pkl`, `.json`, `.npz`), the file's first bytes pick the reader — `load`
  never needs a hint. `mode=` overrides both.
- **Public attributes are strict.** `Data.__setattr__` raises `AttributeError`
  for any public name that is not a declared field, property or method, so
  `mesh.pointz = ...` fails loudly instead of creating dead state.
- **`ImmutableArray`** is a hashable `ndarray` subclass that exists only so a
  numpy array can be a dataclass *default* (`MeshData.matrix`). `__post_init__`
  converts it back, so instances never hold one.

`DataList` is the list counterpart (`MeshList`, `UVList`, `MorphList`,
`SkinList`): a `MutableSequence` that also indexes by name (`meshes["boxA"]`),
by index array, by boolean mask and by slice, filters with `match()` (fnmatch),
sorts by `name`, type-checks `append` against `DATA_LIST_CLASS`, and carries the
same save/load surface.

### A mesh is a face-vertex stream

| Field | Shape | Meaning |
|---|---|---|
| `points` | `(V, 3)` | vertex positions (`UVData` uses `(V, 2)`) |
| `indices` | `(sum(counts),)` | vertex id per face-corner, faces back to back |
| `counts` | `(F,)` | corners per face |

Not a triangle array. Mixed triangles / quads / n-gons in one mesh are normal.
`matrix` is an optional `(4, 4)` row-major object transform that is *not* baked
into `points` until `to_identity()`.

Counts are `point_count` / `face_count` / `edge_count` / `shell_count` — there
is no `num_points`.

### Topology matrices are lazy, padded and cached

Every relation (`f2v`, `v2f`, `e2v`, `e2f`, `f2e`, `v2e`, `ue2v`, `ue2f`,
`f2ue`, `v2ue`, `ue`, `uep`) is an integer matrix built on first access and
cached. Ragged rows are padded with `-1`, so `row[row != -1]` is the standard
idiom. Any topology edit drops the caches.

Edges exist in two forms, and the names read backwards from what you expect:

- **`ue*` are the raw ones** — one row per entry of the index stream, so an
  interior edge appears twice, once per face. `ue2v` is `(sum(counts), 2)`.
- **`e*` are the deduplicated ones** — the real edges. `e2v` is `(E, 2)`, `e2f`
  is `(E, 2)` with `-1` where a border leaves a face missing, and
  `edge_count == E`.
- `ue` bridges them: `e2v == ue2v[ue]`. `uep` lists the pairs of raw rows that
  turned out to be the same edge.

On the unit cube that is `e2v (12, 2)` against `ue2v (24, 2)`.

### Face-local *rules* keep a mesh and its UVs in sync

A `MeshData` and its `UVData` share face topology but not vertex ids — UV seams
split vertices. So every operation that must apply to both is expressed in
**face-local slots** rather than vertex ids, and the identical rules object is
handed to each side.

| Producer | Consumer | Rules |
|---|---|---|
| `get_subset_rules(verts)` | `from_vertices(rules)` | `(F, max(counts))` of kept slots, `-1` padded |
| `get_triangulate_rules(method=…)` | `triangulate(rules)` | a `TriangulateRules` (per-face split code + `ngon_tris` local fans) |
| `get_quadrangulate_rules(…)` | `quadrangulate(rules)` | `(N, 4)` of `(fa, fb, oa_slot, ob_slot)` |

`triangulate` takes rules, not keyword arguments — `mesh.triangulate(mesh.get_triangulate_rules(method="fast"))`.

### Geometric normals vs shading normals

Two independent things.

- `get_face_normals()` / `get_vertex_normals()` / `get_edge_normals()` are
  **geometric** — derived from `points` on demand, Maya's angle- and
  area-weighted vertex normals by default. There is no `get_normals()`.
- `normals` / `normal_indices` are **shading** fields you populate with
  `set_normals(...)`. With no arguments they are per-vertex and
  `normal_indices` stays `None`; with `hard_edge_angle` or explicit
  `hard_edges` they become face-varying. `recompute_normals()` refreshes the
  vectors after a deformation while keeping the splits. Topology edits clear
  both.

### Skin weights, morph targets, maps

- `SkinData.weights` is **dense** `(V, I)` paired with an `influences` list of
  joint names. `valid` means every row sums to `1.0` *and* the widths agree.
  `CompactSkinData` is the top-k form used for disk and USD; convert with
  `to_compact_skin_data()` / `to_skin_data()`. Mirroring is name-driven through
  `patterns` / `Patterns`. Skin *deformation* is not here — see
  [`deform/`](deform/README.md).
- `MorphData` is **sparse**: `offsets` `(N, 3)` plus the `indices` they apply
  to. It supports `+ - * / **`, merging the two index sets, so `sum(morph_list)`
  works.
- `MapData` is a sparse per-component scalar map; `GeomSubsetData` is a
  component selection. Both carry `component_type` of `"v"`, `"f"` or `"e"` and
  convert to and from a single-influence `SkinData`.

### Quads are bilinear patches, not two triangles

`_saddle_surface` is the engine behind `MeshData.sample()`, `MeshData.raycast()`,
`MeshData.area` and `MeshData.get_face_areas()`. A quad whose corners are not
coplanar is integrated and intersected as a genuine saddle. Triangles are padded
to quads with a `-1` fourth index; **n-gons raise** — triangulate first.

Passing `surface_normals=` swaps flat bilinear patches for PN-Quad bicubic
Bézier patches: slower, but each patch curves to match the vertex normals.
`SampleMethod.BILINEAR` and `SampleMethod.BEZIER` select the same two on the
`MeshData` wrappers.

### Misses are NaN, not exceptions

Raycast results carry a boolean `hit` mask. On a miss `distances` is `NaN`,
`indices` is `-1`, and `points` / `projections` fall back to the ray origin so
the arrays keep their shape. Always filter by `hit` before reading.

`occluded` flags queries **inside** the mesh for `sample()` and **front-face**
hits for `raycast()`. The low-level functions need a `normals=` argument to
populate it; the `MeshData` wrappers pass their own vertex normals for you.

### Curves and patches live in knot space

`u` runs `[0, max_param]`, where `max_param` is `count - degree` for an open
curve and `count` for a periodic one. It is **not** normalised — pass
`uniform=True` for arc-length parameterization, where `u` runs `[0, 1]` and
equal steps in `u` are equal steps *along* the curve. Patches carry the flag per
direction (`uniform_u`, `uniform_v`), and every parameter, degree and periodic
flag comes in a `_u` / `_v` pair because a patch is a tensor product.

Open curves **extrapolate** outside the domain along the endpoint tangent; they
do not clamp. A periodic curve wraps *control points*, not parameters: `cv` is
`points` with the first `degree` control points appended, and
`basis(u, collapse=True)` folds them back. `registered=True` (periodic only)
moves `u = 0` onto the curve point nearest `points[0]`.

`basis` is the transfer operator. Every sample and raycast result carries a
`basis` (or `weights`) matrix plus `compute(values)` and an `__call__` alias —
project once, then push any per-control-point or per-vertex attribute stream
through the same weights.

### Control points are held by reference

`BSplineData(points=arr)` stores `arr` itself; it does not copy. Two splines
built from one array alias each other, and mutating the array behind a built
spline leaves stale knot vectors, scipy splines and arc-length tables. After an
in-place edit call `invalidate()` (rebuild lazily), `rebuild()` (rebuild now) or
`rebuild_arc_length_table()`.

### An SDF is a grid of floats

Negative inside, positive outside, zero on the surface. The functional pipeline
is `make_grid` → `eval_*` → `sdf_*` → `dual_marching_cubes`, and cost is cubic
in resolution. Primitives are always evaluated **at the origin** —
`transform_points()` moves the *sample grid* by the inverse transform instead.
Dual Marching Cubes places one vertex per active cube, so its output is
quad-only.

The OOP layer wraps the same thing: `SDFSphere`, `SDFBox` and `SDFCylinder`
(there is no plane) subclass `TransformData`, so they carry the whole Maya SRT
surface. A `DMCField` stacks them with `add` / `subtract` / `intersect` /
`remove` and exposes a lazy `mesh_data`; its `resolution` is *subdivisions per
world unit*, not a total grid size.

### Resampling is topology transfer, not remeshing

`MeshDataResampler(src_mesh, dst_mesh)` never changes the destination topology.
It answers *"what does this source-indexed array become when re-indexed onto the
destination's vertices?"* Morph targets, skin weights, painted maps, geom
subsets and whole meshes all ride the same correspondence — a `SampleData`
computed lazily and **cached per `(mode, method)` pair**, so build one resampler
and call it many times.

| `ResampleMode` | Correspondence | Requires |
|---|---|---|
| `SPATIAL` | closest point on the source surface | nothing |
| `UV` | matching UV coordinates | `src_uv` **and** `dst_uv` |
| `ROBUST_BILINEAR` | closest surface + Laplacian inpaint, quad-native | nothing |

`SPATIAL` and `UV` interpolate a weight for *every* destination vertex,
including ones that project somewhere nonsensical. `ROBUST_BILINEAR` additionally
**rejects** matches beyond `search_radius` (a fraction of the target bounding-box
diagonal) or `normal_threshold`, then solves for the rejects by diffusing from
their matched neighbours. That rejection step is the whole difference — reach
for robust when the two meshes genuinely differ.

Each `ResampleMode` *value* is literally that mode's options dataclass
(`ResampleMode.SPATIAL.value is SpatialSkinTransferOptions`).

### `surface_plotting` speaks (u, v, n), not world

Two heads that share a UV layout share surface coordinates even when their
geometry does not, which is how a control rig authored on one is retargeted onto
another. `transforms_to_surface_space` goes world → `(u, v, 0)`;
`transforms_from_surface_coordinates` builds world frames back out. Those frames
are Maya row-major with **X = U tangent, Y = V tangent, Z = surface normal**.
Every entry point takes a `Mesh` (a `MeshData`) plus a `UVList` selected by
`uv_map_index`.

---

## Conventions

- **Row-major matrices**, like the rest of the package: `p' = p @ M`,
  translation at `M[3, :3]`, Maya rotate orders.
- **`-1` is the pad.** Every adjacency matrix is dense `(rows, max_width)` with
  short rows padded; `-1` also marks a missing face in `e2f` and a miss in a
  raycast's `indices`.
- **Angles are degrees** (`hard_edge_angle`, `normal_threshold`, `rotate`).
- **UV `v` increases up**, `v = 0` is the bottom of the texture. A UV face
  `count` of `0` is a legal hole.
- **Verbs mutate, `from_*` and `get_*` return.** `triangulate`, `quadrangulate`,
  `merge`, `detach_faces`, `delete_faces`, `subdivide`, `smooth`, `to_identity`,
  `union`, `difference`, `normalize` and `pack` edit in place and drop the
  caches; `copy`, `from_faces`, `from_vertices`, `to_uvdata` and every
  `resample_*` hand back a new object.
- **Optional dependencies** (`cv2`, `skimage`, `PIL`, `trimesh`, `pygltflib`,
  `pxr`, the FBX SDK) are imported lazily behind `try/except ImportError`
  and raise a friendly `ImportError` when a path that needs them is taken.
- **Numba compiles on first call**, cached to disk. The first `sample()`,
  `subdivide()` or `raycast()` in a fresh interpreter pays for it.

### Which loaders exist

| Entry point | Returns |
|---|---|
| `mesh.load_obj` / `load_glb` / `load_fbx` / `load_usd` | `list[(MeshData, UVList)]` |
| `mesh.save_obj(path, data)` | writes that same list shape |
| `MeshData.load_obj` / `load_glb` / `load_fbx`, `MeshData.save_obj` | one `MeshData` |
| `MeshList.load_fbx`, `UVList.load_fbx` | a list |
| `UVData.load_fbx`, `UVData.load_glb` | one / a list of `UVData` |
| `skin_weights.load_fbx` / `load_glb` | `list[(mesh name, SkinData)]` |
| `SkinData.load_fbx` / `load_glb` | one `SkinData` |
| `from_prim` / `to_prim` on `MeshData`, `UVData`, `MapData`, `GeomSubsetData`, `MorphData`, `CompactSkinData` | USD |

OBJ is pure python and always available. There is **no** `MeshList.load_obj` and
**no** `MeshData.load_usd`.

---

## Quick taste

No mesh assets ship with this package, so build one inline. A unit cube of six
quads is the standard fixture.

```python
import numpy as np

from cgmath.geometry import MeshData, UVData

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

print(cube.point_count, cube.face_count, cube.edge_count)
print(cube.closed, cube.quads, round(cube.area, 4))
```

### Triangulate a mesh and its UVs with one rules object

```python
cube_uv = UVData(
    name="map1",
    points=np.array([
        [0.33, 0.00], [0.66, 0.00], [0.33, 0.25], [0.66, 0.25],
        [0.33, 0.50], [0.66, 0.50], [0.33, 0.75], [0.66, 0.75],
        [0.33, 1.00], [0.66, 1.00], [1.00, 0.00], [1.00, 0.25],
        [0.00, 0.00], [0.00, 0.25],
    ]),
    indices=np.array([0, 1, 3, 2,  2, 3, 5, 4,  4, 5, 7, 6,
                      6, 7, 9, 8,  1, 10, 11, 3,  12, 0, 2, 13]),
    counts=np.array([4, 4, 4, 4, 4, 4]),
)

rules = cube.get_triangulate_rules()      # face-local, no vertex ids
mesh, uv = cube.copy(), cube_uv.copy()
mesh.triangulate(rules)
uv.triangulate(rules)
print(mesh.face_count, uv.face_count, mesh.triangles)
```

### Project points onto the quad surface, then carry an attribute along

```python
queries = np.array([[0.0, 0.0, 1.5], [0.0, 0.2, 0.0]])
sd      = cube.sample(queries)

print(np.round(sd.projections, 3))          # closest point on the surface
print(np.round(sd.distances, 3), sd.indices, sd.occluded)
print(np.round(sd(cube.points * 10.0), 3))  # any per-vertex stream, same weights
```

### Transfer skin weights onto a different topology

```python
from cgmath.geometry import SkinData
from cgmath.geometry.resample import MeshDataResampler, ResampleMode

dense = cube.copy()
dense.subdivide(1)                          # same shape, different topology

y    = cube.points[:, 1] + 0.5
skin = SkinData(weights=np.column_stack([1.0 - y, y]), influences=["root", "tip"])

dst  = MeshDataResampler(cube, dense).resample_skin_weights(skin, mode=ResampleMode.SPATIAL)
print(dst.weights.shape, dst.influences, dst.valid)
```

---

## Rough edges

Real behaviour, verified — not bugs to work around blindly.

- `BSplinePatchData.area` is less accurate than it looks. `get_area()` only uses
  composite Simpson when `samples` is **odd**, and `area` calls it with the even
  default `samples=50`, so it falls back to a Riemann sum. Pass an odd count.
- The `sdf` module docstring advertises
  `from cgmath.geometry import dual_marching_cubes`, which raises
  `ImportError`. The working path is `from cgmath.geometry.sdf import …`.
  Neither `sdf` nor `_cdt` is re-exported at package level, and neither is
  `PatchRaycastData` or either `SampleData`.
- `SampleData.remap(src_mesh, dst_mesh, dst_uv)` indexes its arrays by *UV
  point*, so it needs one sample per UV point of `dst_uv` and returns one per
  mesh vertex.
- `max_influences` on `RobustBilinearSkinTransferOptions` is applied only by
  `MeshDataResampler.resample_skin_weights`, never by the transfer function
  itself.
- In `surface_plotting`, `u_vector_to_rotation_matrix` returns a **flat `(16,)`**
  array, not a `(4, 4)`.
- `MeshData.ngons` is a count, not an index list. `get_ngons()` is the list. The
  same holds for `triangles` / `get_triangles()` and `quads` / `get_quads()`.
- `get_border_vertices()`, `get_non_manifold_vertices()`, `get_lamina_faces()`
  and `get_overlap_vertices()` are **methods**, not properties.
  `shell_points` / `shell_faces` / `shell_edges` / `shell_count` *are*
  properties.
