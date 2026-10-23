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
| [`deform/`](deform/README.md) | `FFDData`, `DeltaMushData`, `PatchRelaxData`, `SkinDeformData`, `WrapData` |
| [`utils/`](utils/README.md) | the 59 numba kernels every one of the above stands on |

---

## Concepts

### Everything is a `Data` dataclass

`MeshData`, `UVData`, `MapData`, `GeomSubsetData`, `MorphData`, `SkinData` and
`CompactSkinData` all derive from `Data`, a `@dataclass` with managed
serialization. The declared fields *are* the object: `to_dict`, `from_dict`,
`to_json`, `to_bytes`, `save`, `load`, `copy`, `__eq__` and `__reduce__`
(pickle) read exactly those fields, less any `TRANSIENT_FIELDS`, and nothing
else.

- **Caches are never persisted.** A `MeshData` memoises a dozen topology
  matrices on private `_`-prefixed attributes, the ones its `_STATE` names.
  They are excluded from every format, so a reloaded object starts cold and
  recomputes lazily. `reset_cached_data()` clears them by hand.
- **`name` is excluded from equality.** Every type lists it in
  `EQUALITY_TEST_IGNORE`, so two identically shaped meshes with different names
  compare equal.
- **`save` / `load` pick the format.** The extension picks the writer:
  `.npz`, `.json`, `.pkl` / `.pickle`, and an asset's, `.obj` or `.fbx`, for a
  class that writes one. A `.gltf` and a usd name (`.usd` / `.usda` / `.usdc`
  / `.usdz`) go by their extension, to the glb and usd readers; under any
  other name, npz, json, glb and fbx files read by their first bytes, obj and
  pickle files by their extension (`.obj`, `.pkl` / `.pickle`). The glb
  readers go by the first bytes under any name, `.gltf` and `.glb` too: a glb
  under a `.gltf` name reads, and `mode="glb"` reads a gltf, which starts as
  a json does, under any other; a file that starts as neither, a web page or
  a Git LFS pointer saved under a `.glb` name, reads as a glb, so the reader
  says it is no glb, and a gltf trimesh cannot read, cut short or saved with
  utf-8's byte order mark, raises json's own error. Any other name needs
  `mode=`, which overrides both. A json in utf-16, as Windows PowerShell
  5.1's `Out-File` and `>` write one, or with utf-8's byte order mark is
  refused, saying so.
- **Attributes are strict.** An instance takes a write only to a declared
  field, a property with a setter, or private state its class declares in
  `_STATE` (each class naming its own caches, merged up the MRO).
  `Data.__setattr__` raises `AttributeError` for anything else, so
  `mesh.pointz = ...` and `mesh._scratch = ...` fail loudly instead of
  creating dead state, and so does a write over a method (`mesh.copy = ...`),
  a class setting (`EQUALITY_TEST_IGNORE`) or a read-only property, which the
  error names for what it is. A `DataList` holds the same rule: its `list`,
  `LIST_FIELDS`, annotated names, settable properties and `_STATE`.
- **`ImmutableArray`** is a hashable `ndarray` subclass that exists only so a
  numpy array can be a dataclass *default* (`MeshData.matrix`). `__post_init__`
  converts it back, so instances never hold one.

`DataList` is the list counterpart (`MeshList`, `UVList`, `MorphList`,
`SkinList`): a `MutableSequence` that also indexes by name (`meshes["boxA"]`),
by index array, by boolean mask and by slice, filters with `match()` (fnmatch),
sorts by `name`, type-checks `append` against the class it holds, and carries
the same save/load surface. A list class names that class in brackets,
`class MeshList(DataList[MeshData])`, which sets its `DATA_LIST_CLASS` and lets
an editor complete `meshes[0].` and `for mesh in meshes:` with `MeshData`'s
members. A list holding a narrower class than its base names it beside the
base, `class UVList(MeshList, DataList[UVData])`.

### Saving and loading

`pkl`, `npz` and `json` hold one tree, the dict `to_dict()` returns:
`"__class__"` (module and class, `"cgmath.geometry.mesh.MeshData"`), then every
field that differs from its default. A field exactly equal to its default is
left out and loads back as a fresh copy of the default. A `DataList` saves
`"__class__"`, `"__items__"` (one tree per item) and its `LIST_FIELDS`.
`copy()` is the same round trip, in memory.

- **A class's loader gives back that class.** `load`, `load_json`, `load_npz`,
  `load_pickle`, `from_dict` and `from_bytes` called on a class need the file
  or tree to hold exactly that class, under any name it is reached by: a
  script's `"__main__.PinData"` loads through the `PinData` another script
  imports from its module, and `"cgmath.geometry.MeshData"` is `MeshData`. Any
  other raises `TypeError`, a subclass too: `MeshData.load()` on a saved
  `UVData`, whose points are `(N, 2)` where a mesh's are `(N, 3)`, raises
  "*file* holds a UVData; load it with UVData.load() or Data.load()". `Data`'s
  own loaders give back whatever class the file holds, and `DataList`'s any
  list. A module's `load` gives back whatever its file holds:
  `cgmath.hierarchy.load` any `HierarchyData` or class deriving from it, a
  `ClipData` for a saved clip; `skin_weights.load` any `SkinList`. Only the
  whole file is held to its class: what it holds, a field's object or a list's
  items, comes back as the classes they name, which may derive from the one
  declared (a `MeshList`'s `UVData`, a `Scene`'s `Object`, `Camera` and
  `Light` nodes). A tree built by hand, with no `"__class__"`, is read as the
  class called.
- **A field is saved under the name a caller sets it by.** A private field
  behind a public property of its own name is saved as the property:
  `UVData._resolution` as `resolution`, a node's `_translate` as `translate`,
  in every format and in the `"__dtypes__"` / `"__shapes__"` notes. A
  `DataList`'s `LIST_FIELDS` follow the same rule. A load takes either name,
  so a file that names the private field still loads, and a tree that names
  one field both ways raises `ValueError`. A field with no such property
  keeps its own name. A tree built by hand, with no `"__class__"`, as
  `from_dict` takes one, sets a property's name through the property's
  setter, which checks the value; a property with no setter, such as
  `FFDData.cells`, and a `DataList`'s settings are read as the field.
- **A file saved from this version on needs this cgmath or newer.** An older
  cgmath expects the private names: it refuses an `FFDData` from such a file
  with an `AttributeError`, and reads a `ClipData` as its nodes with none of
  their frames (start frame 0, 24 fps), without raising.
- **Older files are refused.** A file written by an older cgmath has no
  `"__class__"` and raises `LegacyFileError`, a `ValueError`.
- **npz holds no pickles.** One `.npy` member per array or value, read with
  `allow_pickle=False`. A 1M-point mesh loads in about 0.2 s.
- **json is utf-8 and plain lists.** NaN and the infinities are written as the
  bare `NaN` / `Infinity` tokens Python's `json` reads back. The field
  declarations restore the dtypes; a small `"__dtypes__"` / `"__shapes__"` note
  covers what a list cannot say (an `Object` texture's own dtype, the shape of
  an empty array).
- **A pickle carries the same tree**, its arrays as their dtype, shape and raw
  bytes, so it holds no numpy objects: a pickle written under numpy 2 reads
  under Maya's numpy 1.24. `SkinData` pickles its weights as their nonzero
  entries, alone or inside a list or an `Object`; npz and json keep them dense.
- **Any dict key comes back.** An int or tuple key, or text a zip member name
  cannot hold, is saved as a list of keys beside a list of values. Text with a
  NUL, an int past 64 bits and bytes come back too.
- **A failed save leaves the file as it was.** The whole file is written to
  scratch first, then over the target in place, so a reader holding it open,
  a link or a read-only file behave as with any plain write.
- **An object held twice is saved once.** Two `Object`s over one `MeshData`,
  or one `Data` in two fields, a dict or a list's items: its first tree
  carries `"__id__"`, a number, and each later one is only
  `{"__class__": "~ref", "id": number}`. Every format and `copy()` give back
  one object, held in each place again, even in a field equal to its
  default: that field is then saved, not left out. A load builds the objects
  in the order the file holds them, so a class that reorders its fields
  still loads its files. Only an object reached twice is numbered, so a file
  that shares nothing is as before. A file whose reference comes before its
  object raises `ValueError`, and so does saving an object, a dict or a list
  that holds itself.
- **Copies.** `copy()` is the save and load round trip, in memory. `copy.copy`
  is shallow: the same field values, caches and links rebuilt; a list's is a
  new list over the same items, but a hierarchy, which owns its nodes, gets
  `copy()`'s own. `copy.deepcopy` copies every field and keeps what the
  copied structure shares shared. A reloaded module's class still loads and
  copies its own files.
- **An `Enum` member is saved as its value**, in every format. A field that
  declares the `Enum`, alone or in a `List` / `Tuple`, gives the member back.
- **json keeps no type a field does not declare.** A tuple or a numpy array in
  a field typed `dict` or `Any` reads back as a list; declared fields restore
  theirs (`Tuple[int, int]`, `List[np.ndarray]`, `Array(...)`).

Array fields are declared with `Array(dtype, *shape)`, where an int fixes a
dimension and a name such as `"N"` leaves it free. A value is converted to the
dtype whenever it is set or loaded, with no copy when it already matches. A
fixed dimension that does not match raises `ValueError`, and an empty value
takes the declared trailing size.

| Arrays | dtype |
|---|---|
| indices, counts, face and vertex ids | `int32` |
| points, normals, offsets, weights, values | `float64` |
| masks (`hit`, `occluded`) | `bool` |

`MeshData.points` is `(N, 3)`, `UVData.points` is `(N, 2)`.

Your own type follows the same rules. Subclass `Data` under
`@dataclass(repr=False, eq=False)`, which keeps its value equality and `repr`,
and declare arrays with `Array`, both from `cgmath.data`. A field named in
`TRANSIENT_FIELDS` is never saved and `==` skips it: use it for caches and
links, and rebuild them in `_post_load()`, which runs once a load or `copy()`
has set every field. An instance takes a write to nothing its class does not
declare: a cache kept on a private attribute rather than a field is named in
the class's `_STATE`, a tuple, `_STATE = ("_index",)`, each class listing its
own and a subclass adding to its parents'. A `DataList` subclass names the
class it holds in brackets, `class PinList(DataList[PinData])`, what it saves
beside its items in `LIST_FIELDS`, and its private state in `_STATE` too; a
name it annotates takes a write as well. A private field behind a property of
its own name is saved as the property; `SAVED_NAMES`, `{field: saved name}`,
saves one under another name, such as a field behind a property named
differently, and keeps a field mapped to itself, `{"_count": "_count"}`, under
its own name, for a property `count` that does not read it. Only cgmath modules
are imported on demand, so a class of your own must be imported before a file
that names it is loaded.

```python
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cgmath.data import Array, Data, DataList


@dataclass(repr=False, eq=False)
class PinData(Data):
    """Vertices pinned to target positions."""

    TRANSIENT_FIELDS = ("lookup",)

    name: str
    indices: Array(np.int32, "N")
    targets: Array(np.float64, "N", 3)
    weight: float = 1.0
    lookup: Optional[dict] = None  # vertex id -> row: a cache, never saved

    def __post_init__(self) -> None:
        super().__post_init__()
        self._post_load()

    def _post_load(self) -> None:
        self.lookup = {int(i): row for row, i in enumerate(self.indices)}


class PinList(DataList[PinData]):
    LIST_FIELDS = ("label",)

    def __init__(self, iterable=None, label=""):
        super().__init__(iterable)
        self.label = label


pins = PinData(name="lips", indices=[4, 9], targets=[[0, 1, 0], [0, 1, 0.5]])
print(pins.indices.dtype, pins.targets.dtype)  # int32 float64

# weight is at its default and lookup is transient: neither is saved
print(sorted(pins.to_dict()))  # ['__class__', 'indices', 'name', 'targets']

tmp = tempfile.mkdtemp()
path = PinList([pins], label="mouth").save(os.path.join(tmp, "pins.json"))
back = PinList.load(path)
print(back.label, type(back[0]).__name__, back[0].lookup)  # mouth PinData {4: 0, 9: 1}

try:
    pins.targets = np.zeros((2, 2))
except ValueError as err:
    print(err)  # PinData.targets must have shape (N, 3), got (2, 2)

pins.targets = []
print(pins.targets.shape)  # (0, 3)
```

One object held twice, here one pin set twice in a list, loads back as one:

```python
twice = PinList([pins, pins], label="twice")
back  = PinList.load(twice.save(os.path.join(tmp, "twice.npz")))
print(back[0] is back[1], back[0] is pins)  # True False
```

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
  both. Both families take triangles, quads and n-gons.

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
to quads with a `-1` fourth index. An n-gon's area is the sum of the triangles
it triangulates into; `sample()` and `raycast()` **raise on n-gons** —
triangulate first.

The closest point on a bilinear patch is exact: corners, edges and the inside
are all solved for, so a twisted or folded quad gives its true closest point.
The closest point's, the raycast's and the vertex match's tolerances are all
fractions of the face, so a mesh answers the same at any size, and a raycast
hit's `(u, v)` and `t` are precise to rounding. The raycast solves a face as
seen from the ray's origin and never asks for more than the rounding there, so
a small face far from the world's origin, or far from the ray's, is still hit.

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

`BSplineData(points=arr)` stores `arr` itself when it is already a `float64`
array; it does not copy. Any other input is converted to a new `float64`
array. Two splines built from one array alias each other, and mutating the
array behind a built spline leaves stale knot vectors, scipy splines and
arc-length tables. After an in-place edit call `invalidate()` (rebuild lazily),
`rebuild()` (rebuild now) or `rebuild_arc_length_table()`.

### An SDF is a grid of floats

Negative inside, positive outside, zero on the surface. The functional pipeline
is `make_grid` → `eval_*` → `sdf_*` → `dual_marching_cubes`, and cost is cubic
in resolution. Primitives are always evaluated **at the origin** —
`transform_points()` moves the *sample grid* by the inverse transform instead.
Dual Marching Cubes places one vertex per active cube, so its output is
quad-only.

The OOP layer wraps the same thing: `SDFSphere`, `SDFBox` and `SDFCylinder`
(there is no plane) subclass `TransformData`, so they carry the whole SRT
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

**UVs go both ways.** `resample_uv()` builds a destination UV set from
`src_uv` by projecting the destination onto the source in world space;
`resample_mesh(mode=ResampleMode.UV)` goes back, placing every destination
vertex where its UV sits on the source surface, and never reads the
destination's points. The way back rebuilds the destination only where the
source's seams follow destination edges: beside a seam that cuts through a
destination face, a vertex can come back on the seam: up to about half a face
away beside one seam, and up to about half a face diagonal where seams cross
inside a face.

<!-- notest -- teaser; `src`, `dst` and `src_uv` are whatever meshes and UVs you have, CHEATSHEET.md runs one -->

```python
resampler = MeshDataResampler(src, dst, src_uv=src_uv)
dst_uv    = resampler.resample_uv()              # closest point
dst_uv    = resampler.resample_uv(raycast=True)  # along dst normals
dst_uv    = resampler.resample_uv(raycast=True, use_mesh_normals=False, area_weighted=False)
```

A UV is one per destination vertex *and source shell*, not one per vertex:
each destination face reads all its corners from the shell under its centroid,
so a seam between two source shells that runs along destination edges splits
the result there instead of stretching faces across the atlas. Such a seam
cutting through a destination face clamps that face's far corners to the shell
border. A cut inside one shell is not split: faces across it stretch. With
`raycast=True`, rays follow `dst.normals` when stored (face-varying ones summed
per vertex), else `get_vertex_normals()`; neither mesh is modified, and a ray
that misses takes the closest point. A source face with UVs must be a triangle
or a quad.

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

Every module with a file interface has `load` (picks the reader from the file),
`load_fbx` and `load_glb`. No loader strips namespaces: fbx and glb node and skin
names come back as written (obj meshes are named after their group plus `Shape`,
and trimesh renames duplicate glb meshes `part`, `part_1`). A list type's
`load_<format>` gives every item in the file; a data type's gives one, the first
or `name=`, except `MeshData.load_obj`, which merges every group into one mesh.

A class's `load()` and an object's `save()` route a file to the class's own
`load_<format>` / `save_<format>`: the type the file reads as, or the save's
extension, or `mode=`, names it (`.gltf` is glb, `.usda` / `.usdc` / `.usdz`
usd, `.pkl` / `.pickle` pickle). Arguments after `mode` go to that route by
keyword, checked against its signature before the file is read or written, so
`MeshData.load("hero.fbx", name="body")` is `MeshData.load_fbx("hero.fbx",
name="body")` and a misspelt one raises `TypeError` naming the ones it takes. A
format the class has no route for, an unknown extension too, raises
`ValueError` naming the routes it has. `Data.load` and `DataList.load` read
cgmath's own files only, which name their class: an asset raises, naming the
classes that read it. An asset keeps only what its writer writes: npz, json and
pkl keep everything.

| Entry point | Returns |
|---|---|
| `mesh.load` / `load_obj` / `load_glb` / `load_fbx` / `load_usd` | `list[(MeshData, UVList)]`, from assets only: a cgmath file loads through its class or `Data.load` |
| `mesh.save_obj(path, data)` | writes that same list shape |
| `MeshList.load` / `load_obj` / `load_glb` / `load_fbx` / `load_usd` | every mesh, a `MeshList` |
| `MeshData.load` / `load_obj` / `load_glb` / `load_fbx` / `load_usd` | one `MeshData` |
| `MeshData.save("x.obj")` / `save_obj` | the points, the faces and the normals (`vn`) the mesh has; not its matrix, holes or UVs |
| `UVList.load` / `load_obj` / `load_glb` / `load_fbx` / `load_usd` | one mesh's UV channels |
| `UVData.load` / `load_obj` / `load_glb` / `load_fbx` / `load_usd` | one `UVData`; no obj writer (its points are 2D) |
| `skin_weights.load` / `load_fbx` / `load_glb`, `SkinList.load_*` | a `SkinList`, each skin named after its mesh |
| `SkinData.load_fbx` / `load_glb` | one `SkinData` |
| `cgmath.hierarchy.load` / `load_fbx` / `load_glb` | a `HierarchyData` from fbx and glb; a cgmath file gives whatever hierarchy it holds (a `ClipData` stays a clip) |
| `from_prim` / `to_prim` on `MeshData`, `UVData`, `MapData`, `GeomSubsetData`, `MorphData`, `CompactSkinData` | USD |

OBJ is pure python and always available. `cgmath.render.scene.Object.save_obj`
writes an Object's mesh where the Object puts it, with its UVs and normals.

### Namespaces

`strip_namespace()` and `namespace` work on every named data type and list.
A skin strips its influences and a skin deformer its joints along with the
name, so strip the rig first, then its skins and meshes: names that still
match bind. Anything that would give two items (or two influences) one name
raises `ValueError` and renames nothing; a type with no name raises
`AttributeError`.

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
- `MeshData.ngons` is a count, not an index list. `get_ngons()` is the list. The
  same holds for `triangles` / `get_triangles()` and `quads` / `get_quads()`.
- `get_border_vertices()`, `get_non_manifold_vertices()`, `get_lamina_faces()`
  and `get_overlap_vertices()` are **methods**, not properties.
  `shell_points` / `shell_faces` / `shell_edges` / `shell_count` *are*
  properties.
