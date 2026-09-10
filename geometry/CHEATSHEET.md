# `cgmath.geometry` — cheatsheet

Copy-paste recipes for every public class, function and notable property in the
package. The blocks run top to bottom as one script and share a namespace — the
fixtures built in **Setup** are reused everywhere below.

Concepts, conventions and the file map live in [`README.md`](README.md).

## Contents

| Section | Covers |
|---|---|
| [Setup](#setup) | fixtures: a unit cube, its UVs, two quad grids |
| [Import surface](#import-surface) | what the package re-exports, and what it does not |
| [`_base` — the `Data` contract](#_base--the-data-contract) | fields, round trips, `save`/`load`, equality, `DataList` |
| [`MeshData` — the polygonal mesh](#meshdata--the-polygonal-mesh) | construction, counts, extents, object matrix, booleans |
| [Topology maps](#topology-maps) | `f2v` / `v2f` / `e2v` / `ue2v` …, `-1` padding, face census |
| [Diagnostics](#diagnostics) | borders, shells, non-manifold, lamina, degenerate, unused |
| [Normals, areas, curvature](#normals-areas-curvature) | geometric vs shading normals, `AreaMethod`, curvature, tangents |
| [Subsets — rebuild a mesh from a selection](#subsets--rebuild-a-mesh-from-a-selection) | `from_faces`, `from_vertices`, subset rules, component plumbing |
| [Mutating ops — triangulate, quadrangulate, merge](#mutating-ops--triangulate-quadrangulate-merge) | rules objects, merge/detach, subdivide, smooth |
| [`UVData` and `UVList`](#uvdata-and-uvlist) | projections, holes, layout, rasterising, channels |
| [`MeshList`](#meshlist) | the mesh collection |
| [File loaders and savers](#file-loaders-and-savers) | OBJ round trip, and which GLB / FBX / USD entry points exist |
| [`map` — `MapData` and `GeomSubsetData`](#map--mapdata-and-geomsubsetdata) | painted weight maps and component selections |
| [`morph_target` — `MorphData` and `MorphList`](#morph_target--morphdata-and-morphlist) | sparse offsets, arithmetic, pruning |
| [`skin_weights` — `SkinData`, `SkinList`, `CompactSkinData`](#skin_weights--skindata-skinlist-compactskindata) | dense weights, cleanup, influence editing, smoothing, top-k |
| [`pack` — UV island packing](#pack--uv-island-packing) | `UVData.pack`, `pack_islands`, resolution / padding / rotations |
| [`surface_plotting` — retargeting through surface frames](#surface_plotting--retargeting-through-surface-frames) | world ↔ surface, B-spline control maps, refactoring, batch retarget |
| [`bspline` — curves](#bspline--curves) | knot space, periodic, arc length, basis, `sample()` |
| [`bspline_patch` — surfaces](#bspline_patch--surfaces) | tensor products, `compute`/`evaluate`, `sample`, `raycast`, area |
| [`_saddle_surface` — quad-aware surface queries](#_saddle_surface--quad-aware-surface-queries) | `integrate`, `sample`, `remap`, `raycast`, the `MeshData` wrappers |
| [`sdf` — signed distance fields, functional API](#sdf--signed-distance-fields-functional-api) | `make_grid`, `eval_*`, CSG, `dual_marching_cubes` |
| [`sdf` — signed distance fields, OOP API](#sdf--signed-distance-fields-oop-api) | `SDFSphere` / `SDFBox` / `SDFCylinder`, `DMCField` |
| [`_cdt` — constrained Delaunay triangulation](#_cdt--constrained-delaunay-triangulation) | the n-gon triangulator behind `get_triangulate_rules` |
| [`resample` — `MeshDataResampler`](#resample--meshdataresampler) | modes, cached sample data, mesh / morph / skin / map transfer |
| [Robust skin weight transfer](#robust-skin-weight-transfer) | bilinear weight inpainting, and when to reach for it |
| [Deprecated shims at `geometry/`](#deprecated-shims-at-geometry) | the six modules that moved |

---

## Setup

No mesh assets ship with this package — build geometry inline. Everything below
reuses `cube` (a closed unit cube), `cube_uv` (its UV layout), `src_grid` /
`dst_grid` (two quad grids over the same domain) and `tmp` (a scratch dir).

```python
import os
import tempfile

import numpy as np

from cgmath.geometry import MeshData, UVData

tmp = tempfile.mkdtemp()

# canonical unit cube -- 8 points, 6 quads, centred on the origin
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

# its UV layout -- same face topology, 2d points, seams split the corners
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


def quad_grid(rows, cols, spacing=1.0, z=0.0):
    """An all-quad grid in the XY plane, wound CCW seen from +Z."""
    xs, ys = np.meshgrid(
        np.arange(cols) * spacing, np.arange(rows) * spacing, indexing="xy"
    )
    points  = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])
    indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            indices += [i0, i0 + 1, i0 + cols + 1, i0 + cols]
    return MeshData(
        indices = np.array(indices, dtype=np.int32),
        counts  = np.full((rows - 1) * (cols - 1), 4, dtype=np.int32),
        points  = points,
    )


src_grid = quad_grid(5, 5, spacing=1.0)        # coarse source
dst_grid = quad_grid(7, 7, spacing=4.0 / 6.0)  # finer target, same domain

print(cube.point_count, cube.face_count, cube_uv.point_count)
print(src_grid.point_count, dst_grid.point_count)
```

---

## Import surface

Twenty-two names are re-exported from the package root.

```python
from cgmath.geometry import (
    BSplineData,
    BSplinePatchData,
    CompactSkinData,
    Data,
    DataList,
    DeltaMushData,
    GeomSubsetData,
    ImmutableArray,
    MapData,
    MeshData,
    MeshList,
    MorphData,
    MorphList,
    PatchRelaxData,
    PatchSampleData,
    RaycastData,
    SkinData,
    SkinList,
    TriangulateMethod,
    TriangulateRules,
    UVData,
    UVList,
)
```

Everything else comes from its own module. Note the asymmetries: `RaycastData`
is the *saddle-surface* one, `PatchRaycastData` and both `SampleData` classes
are **not** re-exported, and nothing from `sdf` or `_cdt` is either — the
`sdf` module docstring's `from cgmath.geometry import dual_marching_cubes`
raises `ImportError`.

```python
from cgmath.geometry import _saddle_surface, surface_plotting
from cgmath.geometry._cdt import triangulate_ngon
from cgmath.geometry.bspline import SampleData
from cgmath.geometry.bspline_patch import PatchRaycastData
from cgmath.geometry.mesh import AreaMethod, Axis, SampleMethod
from cgmath.geometry.pack import pack_islands
from cgmath.geometry.resample import MeshDataResampler, ResampleMode
from cgmath.geometry.sdf import DMCField, SDFBox, SDFCylinder, SDFSphere

print(AreaMethod, [m.name for m in ResampleMode])
print(_saddle_surface.RaycastData is RaycastData)
```

The two subpackages document themselves — [`deform/CHEATSHEET.md`](deform/CHEATSHEET.md)
for the deformers, [`utils/CHEATSHEET.md`](utils/CHEATSHEET.md) for the 59
numba kernels underneath everything here.

```python
from cgmath.geometry import utils as gu
from cgmath.geometry.deform import (
    DeltaMushData,
    FFDData,
    PatchRelaxData,
    SkinDeformData,
    WrapData,
)

print(gu.compute_neighbors(cube.v2f, cube.f2v).shape)
```

---

## `_base` — the `Data` contract

Every geometry type is a `@dataclass` deriving from `Data`. Fields are the
whole payload: equality, hashing, `repr`, pickling and every file format read
the same annotated fields, nothing else.

```python
from cgmath.geometry import Data, DataList, ImmutableArray

print(sorted(MeshData._dataclass_field_names()))
print(cube.to_dict().keys())  # only fields that differ from defaults
print(str(cube), repr(cube))  # __str__ is the name when one is set
```

### Round trips

`to_dict` / `from_dict` is the canonical form; every other format goes through
it.

```python
d = cube.to_dict()
assert MeshData.from_dict(d) == cube

b = cube.to_bytes()                   # zipped .npy payload
assert MeshData.from_bytes(b) == cube

s = cube.to_json()                    # human readable, columnar arrays
assert isinstance(s, str)
```

### Files — `save` / `load` pick the format

Extension drives the writer, the header byte drives the reader, so `load`
never needs to be told what it is opening. `tmp` is the scratch directory from
[Setup](#setup).

```python
for ext in ("pkl", "json", "npz"):
    path = cube.save(os.path.join(tmp, f"cube.{ext}"))
    assert MeshData.load(path) == cube

# explicit mode overrides the extension
path = cube.save(os.path.join(tmp, "cube.dat"), mode="npz")
assert MeshData.load(path) == cube
```

Cached topology is never written — a reloaded object recomputes lazily.

```python
mesh = cube.copy()
_    = mesh.e2v                          # populates the private cache
path = mesh.save(os.path.join(tmp, "nocache.pkl"))
assert MeshData.load(path)._e2v is None

mesh.reset_cached_data()              # or drop it by hand
assert mesh._e2v is None
```

### Equality, copy, match, info

```python
assert cube == cube.copy()            # deep copy, then field-by-field compare
other      = cube.copy()
other.name = "cube_B"
assert other == cube                  # 'name' is in MeshData.EQUALITY_TEST_IGNORE
print(MeshData.EQUALITY_TEST_IGNORE)

assert cube.match("cu*")              # fnmatch against .name
assert not cube.match("sphere")
assert cube.match("CU*", exact=False) # case-insensitive
print(MeshData.info())                # the class docstring
```

### Strict attribute setting

`Data.__setattr__` rejects undeclared public attributes, so a typo raises
instead of silently creating dead state.

```python
mesh          = cube.copy()
mesh.name     = "ok"  # declared field
mesh._scratch = 123   # private names always allowed

try:
    mesh.pointz = mesh.points         # typo
except AttributeError as err:
    print(type(err).__name__)
```

### `DataList` — the list flavour

`MeshList`, `UVList`, `MorphList` and `SkinList` all derive from `DataList`.
It is a `MutableSequence` that also indexes by name and by numpy-style
selections, and carries the same serialization surface as `Data`.

```python
from cgmath.geometry import MeshList

a, b = cube.copy(), cube.copy()
a.name, b.name = "boxA", "boxB"
meshes = MeshList([a, b])

print(len(meshes), meshes)
print(meshes[0].name)                           # by position
print(meshes["boxB"].name)                      # by name
print(meshes.index("boxB"))                     # name -> position
print("boxA" in meshes)                         # membership by name

print([m.name for m in meshes[[1, 0]]])         # fancy indexing -> MeshList
print([m.name for m in meshes[[True, False]]])  # boolean mask
print([m.name for m in meshes[::-1]])           # slice -> MeshList
print([m.name for m in meshes.match("box*")])   # fnmatch filter
print([m.name for m in meshes.match("boxA", exclude=True)])

meshes.sort()                                   # by .name unless a key is given
meshes.append(cube.copy())                      # type-checked against DATA_LIST_CLASS
print(MeshList.DATA_LIST_CLASS.__name__, MeshList.UNIQUE_LIST)
```

`DataList` saves and loads exactly like `Data`.

```python
path = meshes.save(os.path.join(tmp, "meshes.npz"))
assert MeshList.load(path) == meshes
assert MeshList.from_bytes(meshes.to_bytes()) == meshes
```

### Module helpers

The serialization primitives are importable on their own, for dicts that are
not `Data` objects.

```python
from cgmath.geometry._base import (
    bytes_to_dict,
    dict_to_bytes,
    dict_to_zip,
    flatten_nested_lists,
    get_annotations,
    get_file_type,
    is_ndarray_annotation,
    nan_to_none,
    none_to_nan,
    zip_to_dict,
)

payload = {"points": np.zeros((3, 3)), "nested": {"ids": np.arange(4)}}

path = os.path.join(tmp, "payload.npz")
dict_to_zip(payload, path)
back = zip_to_dict(path)
assert np.allclose(back["nested"]["ids"], [0, 1, 2, 3])

assert bytes_to_dict(dict_to_bytes(payload))["points"].shape == (3, 3)

print(get_file_type(path))                           # 'npz' | 'json' | 'pkl'
print(list(flatten_nested_lists([1, [2, [3, 4]]])))  # [1, 2, 3, 4]
print(get_annotations(MeshData)["points"])           # resolved type hints
print(is_ndarray_annotation("np.ndarray"), is_ndarray_annotation("int"))

# json has no NaN -- the writers swap it for null on the way out and back
print(nan_to_none({"a": np.nan}), none_to_nan({"a": None}))
```

`ImmutableArray` exists so a numpy array can be a dataclass *default*
(`MeshData.matrix`). `__post_init__` converts it back to a plain `ndarray`,
so instances never hold one.

```python
arr = ImmutableArray(np.eye(4))
print(isinstance(arr, np.ndarray), hash(arr) is not None)
print(type(cube.matrix).__name__)     # ndarray, not ImmutableArray
```

---

## `MeshData` — the polygonal mesh

Three required fields describe any polygon soup:

| Field | Shape | Meaning |
|---|---|---|
| `points` | `(V, 3)` | vertex positions |
| `indices` | `(sum(counts),)` | flat face-vertex stream |
| `counts` | `(F,)` | vertices per face — mixed tri/quad/n-gon is fine |

Optional fields: `name`, `matrix` `(4, 4)` row-major object transform,
`normals` / `normal_indices` (shading normals), and `hole_faces` /
`hole_counts` / `hole_indices` (n-gon hole bookkeeping).

```python
quad = MeshData(
    name    = "quad",
    points  = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]),
    indices = np.array([0, 1, 2, 3]),
    counts  = np.array([4]),
)
print(quad.point_count, quad.face_count, quad.edge_count)
print(quad.valid)                     # indices dense, counts consistent, no degenerates
```

### Counts, extents and the object matrix

```python
print(cube.point_count, cube.face_count, cube.edge_count, cube.shell_count)
print(cube.closed, cube.open)         # closed == no border edges

bbx_min, bbx_max = cube.get_extent()
print(bbx_min, bbx_max)
print(cube.get_extent(per_face=True)[0].shape)  # (F, 3)

print(cube.points4.shape)                       # (V, 4), homogeneous, w == 1
print(cube.geometry.shape)                      # (F, max(counts)) -- alias of f2v

moved               = cube.copy()
moved.matrix        = np.eye(4)
moved.matrix[3, :3] = [10.0, 0.0, 0.0]     # row-major: translation in the last row
print(moved.get_inverse_matrix()[3, :3])
moved.to_identity()                        # bake matrix into points
print(moved.points[0], moved.matrix[3, :3])
```

### Point arithmetic and booleans

`+= -= *=` act on `points`. `union` / `difference` are face-set operations,
not solid booleans.

```python
face = cube.from_faces([0])
face *= 2                # scale
face += [0.0, 0.0, 1.0]  # translate
print(face.points[0])

open_cube = cube.copy()
open_cube.delete_faces([0])
assert open_cube.open and open_cube.face_count == 5

lid = cube.from_faces([0])
lid.union(open_cube)  # or: lid += open_cube
lid.merge()           # weld the coincident vertices
assert lid == cube

shifted = cube.copy()
shifted.points[:, 0] += 1.0
carved = cube.copy()
carved.difference(shifted)             # or: carved -= shifted
print(carved.face_count)
```

---

## Topology maps

Every relation is a lazily built, `-1` padded integer matrix, cached on the
instance until `reset_cached_data()` or a topology edit.

Edges come in two flavours. `ue*` are the **raw** face-vertex edges: one row
per entry of the index stream, so an interior edge appears twice, once per
face. `e*` are those **deduplicated** — `E` real edges. `edge_count` is `E`.

| Property | Shape | Rows are | Columns are |
|---|---|---|---|
| `f2v` / `geometry` | `(F, max(counts))` | faces | vertex ids |
| `v2f` | `(V, n)` | vertices | face ids |
| `e2v` | `(E, 2)` | deduplicated edges | its two vertex ids |
| `e2f` | `(E, 2)` | deduplicated edges | the faces sharing it, `-1` on a border |
| `f2e` | `(F, max(counts))` | faces | deduplicated edge ids |
| `v2e` | `(V, n)` | vertices | deduplicated edge ids |
| `ue2v` | `(S, 2)` | raw edges (`S = sum(counts)`) | its two vertex ids |
| `ue2f` | `(S, n)` | raw edges | the faces sharing it |
| `f2ue` | `(F, max(counts))` | faces | raw edge ids (= index-stream slots) |
| `v2ue` | `(V, n)` | vertices | raw edge ids |
| `ue` | `(E,)` | — | raw row backing each deduplicated edge |
| `uep` | `(E_dup, 2)` | — | pairs of raw rows that are the same edge |

```python
print(cube.f2v)
print(cube.v2f)
print(cube.e2v.shape, cube.e2f.shape, cube.edge_count)   # (12, 2) (12, 2) 12
print(cube.f2e[5], cube.v2e[0])
print(cube.ue2v.shape, cube.ue2f.shape, cube.f2ue[5], cube.v2ue[0])
print(cube.ue, cube.uep[:3])
assert np.array_equal(cube.e2v, cube.ue2v[cube.ue])      # ue is the bridge

# -1 is padding, strip it before use
row = cube.v2f[0]
print(row[row != -1])
```

### Face-kind census

```python
print(cube.triangles, cube.quads, cube.ngons)                    # counts
print(cube.get_triangles(), cube.get_quads(), cube.get_ngons())  # face indices
print(cube.valence)                                              # max vertex degree
print(cube.get_valence())                                        # per-vertex degree
```

---

## Diagnostics

```python
print(cube.get_border_vertices())      # [] -- a closed cube has no border
print(cube.get_border_edges(), cube.get_border_faces())

lid_off = cube.copy()
lid_off.delete_faces([4])
print(lid_off.get_border_vertices())        # one array per border loop
print(lid_off.get_border_vertices(flatten=True))
print(lid_off.get_border_edges(flatten=True))
print(lid_off.get_border_faces(flatten=True))
```

Shells are connected components, as vertex / face / edge index arrays.

```python
two = cube.copy()
far = cube.copy()
far.points[:, 0] += 10.0
two.union(far)
print(two.shell_count, [s.size for s in two.shell_faces])
print([s.size for s in two.shell_points], [s.size for s in two.shell_edges])
```

Bad-geometry probes.

```python
# two faces meeting at a single vertex
pinch = MeshData(
    indices = np.array([0, 1, 3, 2, 4, 2, 6, 5]),
    counts  = np.array([4, 4]),
    points=np.array([[0., 0., 1.], [1., 0., 1.], [0., 0., 0.], [1., 0., 0.],
                     [-1., 0., 0.], [-1., 0., -1.], [0., 0., -1.]]),
)
print(pinch.get_non_manifold_vertices())

# the same quad written twice
lamina = MeshData(
    indices = np.array([0, 1, 3, 2, 0, 1, 3, 2]),
    counts  = np.array([4, 4]),
    points  = np.array([[-.5, 0., .5], [.5, 0., .5], [-.5, 0., -.5], [.5, 0., -.5]]),
)
print(lamina.get_lamina_faces(), lamina.get_non_manifold_vertices())

dupes = MeshData(
    points  = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [1., 1., 0.]]),
    indices = np.array([0, 1, 3, 2]),
    counts  = np.array([4]),
)
print(dupes.get_overlap_vertices(tolerance=1e-6))   # pairs of coincident ids

stray = MeshData(
    points  = np.vstack([dupes.points, [[9., 9., 9.]]]),
    indices = np.array([0, 1, 3, 2]),
    counts  = np.array([4]),
)
print(stray.get_unused_points())

bad = MeshData(
    points  = cube.points[:6],
    indices = np.array([0, 1, 0, 2,  0, 1, 2, 3]),
    counts  = np.array([4, 4]),
)
print(bad.has_degenerate_faces, bad.get_degenerate_faces())

flat = MeshData(
    points=np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]]),
    indices=np.arange(3), counts=np.array([3]),
)
print(flat.get_zero_area_faces())
```

---

## Normals and areas

Two independent families. `get_face_normals` / `get_vertex_normals` are
*geometric* and always computed on demand. `set_normals` stores *shading*
normals on the `normals` / `normal_indices` fields, with hard-edge splitting.

```python
print(cube.get_face_normals().shape)        # (F, 3)
print(cube.get_vertex_normals().shape)      # (V, 3), Maya weighting
print(cube.get_vertex_normals(angle_weighted=False, area_weighted=False)[0])
print(cube.get_edge_normals().shape)        # (E, 3)
print(cube.get_edge_lengths().shape)        # (E,)
print(cube.get_face_vertex_angles().shape)  # (sum(counts),)
```

```python
mesh = cube.copy()
mesh.set_normals()                              # all smooth
print(mesh.normals.shape, mesh.normal_indices)  # per-vertex, no index array

mesh = cube.copy()
mesh.set_normals(hard_edge_angle=45.0)             # split every 90 deg edge
print(mesh.normals.shape, mesh.normal_indices.size)

mesh = cube.copy()
mesh.set_normals(hard_edges=np.array([0, 1, 2]))   # explicit edge ids
mesh.points[:, 1] += 5.0
mesh.recompute_normals()                           # keeps the split topology
print(np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0))
```

Topology edits clear stored normals; `copy()` and the serializers keep them.

```python
mesh = cube.copy()
mesh.set_normals()
mesh.delete_faces([0])
print(mesh.normals, mesh.normal_indices)           # None, None
```

Face areas.

```python
from cgmath.geometry.mesh import AreaMethod

print(cube.area)                                   # sum of face areas
print(cube.get_face_areas())
print(cube.get_face_areas(method=AreaMethod.PLANAR))
print(cube.get_face_areas(method="gauss", samples=10, tolerance=1e-4))
print([m.value for m in AreaMethod])
```

Tangent space needs a UV set; returns `(UV, 4)` tangents (`w` is the
bitangent sign) and `(UV, 3)` bitangents.

```python
tangents, bitangents = cube.get_tangent_space(cube_uv)
print(tangents.shape, bitangents.shape)
print(np.unique(tangents[:, 3]))
```

---

## Subsets — rebuild a mesh from a selection

```python
print(cube.contains_vertices([0], contained=False))      # faces touching vert 0
print(cube.contains_vertices([0, 1, 2, 3], contained=True))

top = cube.from_faces([0])
print(top.point_count, top.face_count)
print(cube.from_faces([0], exclude=True).face_count)

whole = cube.from_vertices([0, 1, 2, 3])                 # whole faces only
print(whole.counts, whole.indices)
```

`get_subset_rules` keeps *part* of a face — a quad missing one vertex comes
back as a triangle. Rules are face-local slots, so the identical array
applies to the mesh and to its UV set.

```python
rules = cube.get_subset_rules([0, 1, 2, 3, 4, 5, 7])
print(rules)                                             # (F, max(counts)), -1 padded

part    = cube.from_vertices(rules)
part_uv = cube_uv.from_vertices(rules)
print(part.counts, part_uv.counts)
print(cube.rules_to_vertices(rules))                     # surviving source verts
```

Face and vertex plumbing.

```python
print(cube.faces_to_vertices([0, 1]))
print(cube.vertices_to_faces([0, 1]))
print(cube.vertices_to_edges([0, 1]))
print(cube.edges_to_vertices([0, 1, 2]))
print(cube.edges_to_vertex_pairs([2]))     # sorted pair
print(cube.vertex_pairs_to_edges([0, 1]))  # order independent
```

---

## Mutating ops — triangulate, quadrangulate, merge

`get_triangulate_rules` decides *how*, `triangulate` applies it. The rules
object is topology-only, so a `MeshData` and its `UVData` stay in lockstep.

```python
from cgmath.geometry import TriangulateMethod, TriangulateRules

print([m.value for m in TriangulateMethod])              # fast, trinity, even, odd

mesh, uv = cube.copy(), cube_uv.copy()
rules = mesh.get_triangulate_rules(method=TriangulateMethod.TRINITY)
print(type(rules).__name__, rules.rules, rules.ngon_tris)

mesh.triangulate(rules)
uv.triangulate(rules)
print(mesh.face_count, mesh.triangles, uv.face_count)
```

```python
mesh = cube.copy()
mesh.triangulate(mesh.get_triangulate_rules(method="fast"))       # str works too
print(mesh.counts)

mesh  = cube.copy()
rules = mesh.get_triangulate_rules(invert=True)                   # flip the split
mesh.triangulate(rules, preserve_order=False)                     # faster, reorders
print(mesh.face_count)

hexa = MeshData(
    points=np.array([[np.cos(t), np.sin(t), 0.0]
                     for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]),
    indices=np.arange(6), counts=np.array([6]),
)
rules = hexa.get_triangulate_rules(ngons_only=True)               # leave tris/quads
print(rules.ngon_tris)
hexa.triangulate(rules)
print(hexa.counts)
```

`quadrangulate` is the inverse: greedy maximum-weight matching on the
triangle dual graph.

```python
mesh, uv = cube.copy(), cube_uv.copy()
tri_rules = mesh.get_triangulate_rules()
mesh.triangulate(tri_rules)
uv.triangulate(tri_rules)

quad_rules = mesh.get_quadrangulate_rules()      # (N, 4): fa, fb, oa_slot, ob_slot
print(quad_rules.shape)
mesh.quadrangulate(quad_rules)
uv.quadrangulate(quad_rules)
print(mesh.counts, uv.counts)

# tune the scoring, or let it pick defaults
mesh2 = cube.copy()
mesh2.triangulate(mesh2.get_triangulate_rules())
mesh2.quadrangulate()
print(mesh2.face_count)
```

`merge` welds coincident points; `detach_faces` is its opposite.

```python
mesh = cube.copy()
mesh.detach_faces()                    # every face gets its own vertices
print(mesh.point_count, mesh.open)
mesh.merge()
print(mesh.point_count, mesh.closed)

mesh = cube.copy()
mesh.detach_faces([0, 1])              # only the listed faces
print(mesh.point_count)
```

Subdivision and Laplacian smoothing are in-place too.

```python
mesh = cube.copy()
mesh.subdivide(1)                                  # Catmull-Clark
print(mesh.point_count, mesh.face_count)

mesh = cube.copy()
mesh.smooth(iterations=3, receptions=0.5, contributions=1.0)
print(mesh.points[0])
```

---

## `UVData` and `UVList`

`UVData` is a `MeshData` subclass whose `points` are 2d and whose `counts`
mirror the mesh's, so every topology property above works on it. A count of
`0` is a legal *hole* — a face with no UVs.

```python
print(isinstance(cube_uv, MeshData), cube_uv.points.shape)
print(cube_uv.point_count, cube_uv.face_count, cube_uv.name)
print(cube_uv.f2v[0], cube_uv.get_border_vertices(flatten=True))
```

### Projections and holes

```python
from cgmath.geometry.mesh import Axis

projected = cube.to_uvdata(name="proj", axis=Axis.Y)
print(projected.points.shape, projected.name)
print(cube.to_uvdata(axis="x", maintain_ratio=True, swap_uvs=True).points.shape)

holed         = cube_uv.copy()
holed.counts  = np.array([4, 4, 4, 4, 4, 0])
holed.indices = holed.indices[:20]
print(holed.has_holes, holed.get_holes())
holed.fill_holes(cube.counts)
print(holed.has_holes, holed.face_count)
```

### Layout

```python
uv = cube_uv.copy()
uv.normalize()                                     # fit into 0..1
print(uv.get_extent())
uv.normalize(bbx_min=(0.0, 0.0), bbx_max=(0.5, 0.5))
print(uv.get_extent())

uv = cube_uv.copy()
uv.pack(resolution=128, padding=1, rotations=1)    # skyline island packer
print(uv.get_extent())
print(cube_uv.get_minimum_resolution(pixels=3))
print(cube_uv.get_overlap_faces(cube_uv, resolution=64))
```

### Rasterising

`UVData` owns a render buffer keyed by `resolution`. Faces can be drawn with
an id colour (`color=None`) that decodes back to face indices.

```python
uv            = cube_uv.copy()
uv.resolution = 64
uv.clear_buffer()
uv.draw_faces(color=(255, 255, 255))
print(uv.buffer.shape, uv.buffer.max())

uv.clear_buffer()
uv.draw_faces(indices=[0, 2], color=None)          # id-encoded
rgb = np.unique(uv.buffer.reshape(-1, 3), axis=0)
rgb = rgb[np.sum(rgb == 0, axis=1) != 3]
print(sorted(np.atleast_1d(uv.decode_rgb(rgb)).tolist()))
print(uv.encode_rgb(5), uv.decode_rgb(uv.encode_rgb(5)))

uv.draw_edges(color=(255, 0, 0), thickness=1)
uv.draw_points(color=(0, 255, 0), radius=1)
uv.draw_mask(contour=False)
uv.convolve(steps=1)  # grow
uv.erode(steps=1)     # shrink
print(uv.pixel_counts.shape, uv.pixel_ratios.shape, uv.pixel_overlaps.shape)
print(uv.to_image().size)
print(uv.imwrite(os.path.join(tmp, "uv.png")))

from cgmath.geometry.mesh import RenderMethod

print([e.value for e in RenderMethod], uv.renderer)   # cv2 if importable, else skimage
```

### `UVList`

One entry per UV channel of a mesh.

```python
from cgmath.geometry import UVList

uvs         = UVList([cube_uv.copy()])
uvs[0].name = "map1"
print(len(uvs), uvs["map1"].point_count, uvs.has_holes)
print(uvs.get_minimum_resolution())

uvs.triangulate(cube.get_triangulate_rules())      # same rules as the mesh
print(uvs[0].counts)

uvs = UVList([cube_uv.copy()])
uvs.pack(resolution=128, padding=1, rotations=1)
uvs.merge_seams()
uvs.defrag()
print(len(uvs))
```

---

## `MeshList`

```python
a, b = cube.copy(), cube.copy()
a.name, b.name = "boxA", "boxB"
b.points[:, 0] += 3.0
meshes = MeshList([a, b])

print([m.name for m in meshes], [m.point_count for m in meshes])
meshes.to_identity()  # bake every object matrix
meshes.merge()        # weld each mesh in place
print([m.point_count for m in meshes])
```

---

## File loaders and savers

Module-level readers return `list[(MeshData, UVList)]`; the classmethods pull
a single object out of the file. OBJ is pure python and round-trips here.

```python
from cgmath.geometry.mesh import load_obj, save_obj

path = os.path.join(tmp, "cube.obj")
save_obj(path, [(cube, UVList([cube_uv]))])

data = load_obj(path)
mesh2, uvs2 = data[0]
print(len(data), type(mesh2).__name__, type(uvs2).__name__, len(uvs2))
assert mesh2 == cube and uvs2[0] == cube_uv

# single-object form
cube.save_obj(path)
assert MeshData.load_obj(path) == cube
```

GLB / FBX / USD need their optional SDKs and a real asset on disk.

<!-- notest -->
```python
from cgmath.geometry.mesh import load_fbx, load_glb, load_usd

load_glb("hero.glb", scale_factor=100.0)    # list[(MeshData, UVList)]
load_fbx("hero.fbx")
load_usd("hero.usd")

MeshData.load_glb("hero.glb")
MeshData.load_fbx("hero.fbx", name="body")  # name=None -> first mesh
MeshList.load_fbx("hero.fbx")
UVData.load_fbx("hero.fbx", channel=0)
UVList.load_fbx("hero.fbx", name="body")
```

USD prims round-trip through `from_prim` / `to_prim` on `MeshData`, `UVData`,
`MapData`, `GeomSubsetData`, `MorphData` and `CompactSkinData`.

<!-- notest -->
```python
mesh = MeshData.from_prim(prim)
mesh.to_prim(prim)
```

---

## `map` — `MapData` and `GeomSubsetData`

`MapData` is a sparse per-component scalar map (a painted weight map).
`GeomSubsetData` is a component *selection*. Both carry a `component_type` of
`"v"`, `"f"` or `"e"`.

```python
from cgmath.geometry import GeomSubsetData, MapData

weights = MapData(
    name           = "tension",
    indices        = np.array([0, 2, 5]),
    values         = np.array([0.25, 0.5, 1.0]),
    default_value  = 0.0,
    component_type = "v",
)
print(weights.to_dense_array(cube.point_count))

skin = weights.to_skin_data(cube, influence="jaw")
print(skin.influences, skin.weights.shape)
print(MapData.from_skin_data(skin, name="tension").indices)
```

```python
subset = GeomSubsetData(name="cap", indices=np.array([0, 1, 3, 2]), component_type="v")
print(subset.component_type, subset.indices)

skin = subset.to_skin_data(cube, influence="cap_jnt")
print(skin.weights[:, 0])
print(GeomSubsetData.from_skin_data(skin, name="cap").indices)

faces = GeomSubsetData(name="top", indices=np.array([0]), component_type="f")
print(faces.to_skin_data(cube).weights[:, 0])

islands = subset.split_to_array_by_clusters(cube)
print(len(islands), [i.name for i in islands])
```

Both are `Data`, so the whole serialization surface applies.

```python
assert MapData.from_bytes(weights.to_bytes()) == weights
assert GeomSubsetData.load(subset.save(os.path.join(tmp, "subset.json"))) == subset
```

---

## `morph_target` — `MorphData` and `MorphList`

A morph target is a sparse set of point offsets. `indices` defaults to
`arange(len(offsets))`, i.e. the whole mesh.

```python
from cgmath.geometry import MorphData, MorphList

bulge = MorphData(
    name    = "bulge",
    indices = np.array([0, 1, 2]),
    offsets = np.array([[0.0, 0.5, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.0]]),
)
print(bulge.size, bulge.magnitudes, bulge.min, bulge.max, bulge.mse)
print(bulge.unit)                       # offsets as unit vectors
print(bulge.is_zero())

dense = MorphData(name="dense", offsets=np.zeros((cube.point_count, 3)))
print(dense.size, dense.is_zero())
```

### Deriving one from two meshes, and applying it

```python
target = cube.copy()
target.points[0] += [0.0, 1.0, 0.0]

delta = MorphData.from_mesh_data(cube, target, target_name="lift")
print(delta.name, delta.indices, delta.offsets)

mesh = cube.copy()
mesh.apply_morph_target(delta)
assert np.allclose(mesh.points, target.points)
```

### Pruning and sorting

```python
morph = bulge.copy()
morph.prune_offsets(tolerance=0.1)      # drop offsets at or below the threshold
print(morph.indices, morph.size)

morph = bulge.copy()
morph.prune_offsets(tolerance=0.1, neighbors=cube.get_edge_vertex_neighbors())
print(morph.size)                       # keeps a vertex whose neighbour survived

morph = MorphData(name="m", indices=np.array([5, 1, 3]), offsets=np.eye(3))
morph.sort()
print(morph.indices)
```

### Arithmetic

`+ - * / **` work element-wise, merging the two index sets. Names are
concatenated on the union of their `_` separated parts.

```python
a = MorphData(name="a", indices=np.array([0, 1]), offsets=np.array([[1., 0, 0], [1., 0, 0]]))
b = MorphData(name="b", indices=np.array([1, 2]), offsets=np.array([[0., 1, 0], [0., 1, 0]]))

print((a + b).name, (a + b).indices, (a + b).offsets)
print((a - b).offsets)
print((a * 2.0).offsets, (a / 2.0).offsets, (a ** 2).offsets)
```

### `MorphList`

```python
shapes = MorphList([a.copy(), b.copy(), MorphData(name="z", offsets=np.zeros((3, 3)))])
print(shapes.size, shapes.min, shapes.max, shapes.mse)

shapes.remove_unused()                  # drops all-zero targets
print([s.name for s in shapes])

shapes.prune_offsets(tolerance=0.5)
print(shapes.size)

combined = sum(MorphList([a.copy(), b.copy()]))     # __radd__ makes sum() work
print(combined.name, combined.size)

scaled = MorphList([a.copy(), b.copy()]) * 0.5
print(scaled[0].offsets.max())
```

---

## `skin_weights` — `SkinData`, `SkinList`, `CompactSkinData`

`SkinData.weights` is dense `(V, I)`; `influences` is the matching list of
joint names. Valid means every row sums to 1 and the widths agree.

```python
from cgmath.geometry import CompactSkinData, SkinData, SkinList

skin = SkinData(
    name="cube_skin",
    weights=np.array([
        [1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.5, 0.5, 0.0], [0.2, 0.8, 0.0],
        [0.0, 0.9, 0.1], [0.0, 0.5, 0.5], [0.0, 0.2, 0.8], [0.0, 0.0, 1.0],
    ]),
    influences=["root", "spine", "head"],
)
print(skin.size, skin.valid, skin.counts)  # counts = influences per vertex
print(skin.index("spine"), "head" in skin)
print(skin.get_influences("*a*"))          # fnmatch
print(skin.get_influences("HEAD", exact=False))
print(skin.rank(["head", "root"]))         # names -> column ids
```

### Cleanup

```python
s = skin.copy()
s.weights *= 0.5
print(s.valid)
s.normalize()
print(s.valid)

s = skin.copy()
s.prune(0.3)                                       # zero anything smaller, renormalise
print(s.counts.max(), s.valid)
s = skin.copy()
s.prune(0.3, optimize=True)                        # also drop emptied influences
print(s.influences)

s = skin.copy()
s.round(2)
print(s.weights[1], s.valid)

s = skin.copy()
s.sort()                                           # influences alphabetical
print(s.influences)

s = skin.copy()
print(s.get_max_influences(), s.get_indices_over_max(1))
s.set_max_influences(1)
print(s.get_max_influences(), s.valid)
```

### Editing the influence list

```python
s = skin.copy()
s.append("tail", weight=0.0)
s.insert(1, "neck", weight=0.0)
s.extend(["jaw", "tongue"])
print(s.influences, s.valid)

s.remove("tongue", optimize=False)
s.remove_unused()                                  # drop influences with no weight
print(s.influences, s.valid)

s = skin.copy()
s.transfer_influences(["spine"], ["root", "head"], weighted=True)
print(s.weights[2], s.valid)
```

### Combining and slicing

```python
other = SkinData(
    weights    = np.array([[1.0, 0.0], [0.0, 1.0]]),
    influences = ["tail", "root"],
)
merged = skin + other                              # concatenates vertices
print(merged.size, merged.influences, merged.valid)

head = merged[: skin.size]                         # slicing keeps all influences
print(head.size, head.influences)
print(merged[np.arange(2)].size)

a2, b2 = skin.copy(), other.copy()
SkinData.conform(a2, b2)       # align influence lists in place
print(a2.influences == b2.influences)
print(a2.is_equivalent(skin))  # order-insensitive comparison
```

### Smoothing over mesh topology

```python
neighbors = cube.get_edge_vertex_neighbors()

s = skin.copy()
s.smooth(neighbors, iterations=2, receptions=0.5, contributions=1.0)
print(s.valid)

s            = skin.copy()
s.weights[3] = 0.0                                 # a hole to fill
s.inpaint(neighbors, indices=np.array([3]), iterations=4)
print(np.round(s.weights[3], 3))

s = skin.copy()
s.subdivide(cube.copy(), steps=1)                  # Catmull-Clark, matches mesh.subdivide
print(s.size)
```

### Symmetry

Mirroring is name driven. `patterns` is a `{"positive": [...], "negative": [...]}`
pair of fnmatch globs that picks the two sides; `Patterns` ships the usual rig
conventions and a fresh `SkinData` starts on `Patterns.DEFAULT` (`l_*` / `r_*`).
`axis` and `pivot` say where the mesh's mirror plane is.

```python
from cgmath.geometry.skin_weights import Patterns

sym = SkinData(name="sym", weights=np.zeros((8, 3)),
               influences=["root", "l_arm", "r_arm"])
left                  = cube.points[:, 0] < 0
sym.weights[left, 1]  = 1.0
sym.weights[~left, 2] = 1.0

print(sym.patterns, sym.positive_patterns, sym.negative_patterns)
print(Patterns.DEFAULT, Patterns.SHORT, Patterns.STYLE2)

mapping, positive, negative, rest = sym.get_symmetry_map()
print(mapping, positive, negative, rest)                    # column -> mirrored column, then the sides
print(sym.symmetrical(cube))                                # every influence has a partner
print(sym.get_asymmetric_weights(cube, axis=0, pivot=0.0))  # [] -- nothing to fix
```

Break one side and mirror it back. `side=-1.0` copies the negative half onto
the positive half; `side=1.0` goes the other way.

```python
broken            = sym.copy()
broken.weights[1] = [1.0, 0.0, 0.0]
print(broken.get_asymmetric_weights(cube, axis=0, pivot=0.0))   # [0 1]
print(broken.fix_symmetry(cube, axis=0, pivot=0.0, side=-1.0))
print(broken.get_asymmetric_weights(cube, axis=0, pivot=0.0), broken.weights[1])
```

`patterns` is settable, so a rig with a different naming style just swaps it.

```python
s          = sym.copy()
s.patterns = Patterns.SHORT                  # '*_l_*' / '*_r_*'
print(s.positive_patterns, s.negative_patterns)
```

### `CompactSkinData`

Top-k storage: `max_influences` per vertex, flat `influence_indices` and
`weights`. Smaller on disk, and what USD/`SkinWeights` prims want.

```python
compact = skin.to_compact_skin_data()
print(compact.max_influences, compact.influence_indices.shape, compact.weights.shape)
print(compact.valid)
compact.round(3)
print(compact.to_skin_data().is_equivalent(skin))

manual = CompactSkinData(
    max_influences    = 2,
    influence_indices = np.array([0, 1, 0, 1]),
    weights           = np.array([0.3, 0.7, 0.5, 0.5]),
    influences        = ["root", "head"],
)
print(manual.valid, manual.to_skin_data().weights)
```

### `SkinList`

```python
s1, s2 = skin.copy(), skin.copy()
s1.name, s2.name = "bodySkin", "headSkin"
skins = SkinList([s1, s2])

print([s.name for s in skins], skins["headSkin"].size)
print([s.name for s in skins.match("body*")])
skins.transfer_influences(["head"], ["spine"])     # applies to every element
print(skins[0].valid)

path = skins.save(os.path.join(tmp, "skins.npz"))
assert SkinList.load(path) == skins
```

Skin *deformation* (LBS / DQS) lives in `cgmath.geometry.deform.SkinDeformData`.

FBX and GLB skin readers need their SDKs and an asset.

<!-- notest -->
```python
from cgmath.geometry.skin_weights import load_fbx, load_glb

load_fbx("hero.fbx", bind_matrices=False)     # [(mesh name, SkinData), ...]
load_glb("hero.glb")
SkinData.load_fbx("hero.fbx", name="body")
SkinData.load_glb("hero.glb")
```

---

## `pack` — UV island packing

Rasterized bitmap skyline packer. Rearranges UV shells inside `[0, 1]²` to
kill dead space. Typical utilisation goes 40-60% → 85-95%. A column-profile
skyline lets irregular island bottoms nestle into gaps, worth another 5-15%
over rectangular skyline packing.

### The ergonomic entry point: `UVData.pack`

```python
def quad_island(origin, size=0.4):
    ox, oy = origin
    return np.array([
        [ox, oy], [ox + size, oy], [ox + size, oy + size], [ox, oy + size],
    ])


islands = [quad_island((0.0, 0.0), 0.5),
           quad_island((3.0, 3.0), 0.3),
           quad_island((6.0, 0.0), 0.2)]

scattered = UVData(
    points  = np.vstack(islands),
    indices = np.arange(4 * len(islands), dtype=np.int32),
    counts  = np.full(len(islands), 4, dtype=np.int32),
)
print(len(scattered.shell_faces), scattered.points.max(axis=0))  # 3 islands, far outside [0,1]

scattered.pack()                                                 # in place
print(scattered.points.min(axis=0), scattered.points.max(axis=0))
assert scattered.points.min() >= -1e-7 and scattered.points.max() <= 1 + 1e-7
```

### The free function

`UVData.pack` forwards to it — reach for the function when you already hold
the module.

```python
from cgmath.geometry.pack import pack_islands

again = UVData(
    points  = np.vstack(islands),
    indices = np.arange(4 * len(islands), dtype=np.int32),
    counts  = np.full(len(islands), 4, dtype=np.int32),
)
pack_islands(again, resolution=1024, padding=2, rotations=4)
assert np.allclose(again.points, scattered.points)
```

### Parameters

| Arg | Default | Meaning |
|---|---|---|
| `resolution` | `1024` | Rasterisation pixels per UV unit. Higher = tighter, slower. |
| `padding` | `2` | Pixel gutter dilated around each island before placement. |
| `rotations` | `4` | Rotation candidates tried in `[0°, 180°)`. `1` disables rotation. |

```python
tight = UVData(points=np.vstack(islands),
               indices=np.arange(4 * len(islands), dtype=np.int32),
               counts=np.full(len(islands), 4, dtype=np.int32))
tight.pack(resolution=2048, padding=0, rotations=1)
assert tight.points.max() <= 1 + 1e-7
```

### Degenerate inputs are no-ops

```python
single = UVData(points=quad_island((5.0, 5.0), 2.0),
                indices=np.arange(4, dtype=np.int32),
                counts=np.array([4], dtype=np.int32))
single.pack()                            # one island -> just normalize()
assert single.points.max() <= 1 + 1e-7

empty = UVData(points=np.empty((0, 2)),
               indices=np.array([], dtype=int),
               counts=np.array([], dtype=int))
empty.pack()                             # no faces -> returns immediately
assert empty.face_count == 0
```

Packing preserves topology and each island's internal proportions — only a
rigid rotation plus one global uniform scale is applied.

---

## `surface_plotting` — retargeting through surface frames

Projects transforms onto a mesh's UV-parameterised surface, moves them around
as B-spline-weighted controls, and pushes them back out to world space. This
is how a control rig authored on one head is retargeted onto another: both
heads share a UV layout, so surface coordinates `(u, v, n)` are a common
space even when the geometry is not.

Everything here takes a `Mesh` (a `MeshData`) plus a `UVList` — a *list* of
`UVData`, indexed by `uv_map_index`.

```python
from cgmath.geometry import surface_plotting as sp

face        = quad_grid(6, 6, spacing=0.2)
bowed       = face.points.copy()
bowed[:, 2] = 0.25 * np.sin(np.pi * bowed[:, 0])     # give it some curvature
face.points = bowed
uv_list     = [face.to_uvdata(axis="z")]
```

### World ↔ surface

`get_points_from_surface` dispatches on the width of the input: 2-D in means
"these are UVs, give me world points"; 3-D in means "these are world points,
give me UVs".

```python
world   = np.array([[0.35, 0.65, 0.2], [0.5, 0.5, 0.2]])

surface = sp.get_points_from_surface(face, uv_list, world, as_3d=True)
print(surface)                                    # (u, v, 0) -- as_3d pads the n slot

back = sp.get_points_from_surface(face, uv_list, surface[:, :2])
print(back)                                       # world points, back on the surface
```

`transforms_to_surface_space` is the matrix-flavoured wrapper of the first
direction:

```python
mats = sp.control_array_type_conversion(world, to_transformation_matrix=True)
uvn  = sp.transforms_to_surface_space(face, uv_list, mats, as_matrices=True)
print(uvn.shape, uvn[0, 3, :3])                   # (n, 4, 4), translation is (u, v, 0)

# same trip straight from points, skipping the matrix conversion
print(np.round(sp.convert_points_to_surfacespace_matrices(world, face, uv_list)[0, 3, :3], 3))
```

`transforms_from_surface_coordinates` is the other direction and builds a full
coordinate frame at each point by finite-differencing the surface:

```python
frames = sp.transforms_from_surface_coordinates(face, uv_list, surface)
print(sorted(frames))                             # matrices, points, u_vecs, v_vecs, normals
print(frames["matrices"][0])
```

The frame convention is **X = U tangent, Y = V tangent, Z = surface normal**,
translation in row 3 (Maya row-major).
`surface_points_to_transformation_matrices` builds the same thing from
vectors you already have:

```python
built = sp.surface_points_to_transformation_matrices(
    frames["points"], frames["u_vecs"], frames["v_vecs"]
)
assert np.allclose(built["matrices"], frames["matrices"])
```

### Matrix / point plumbing

`control_array_type_conversion` converts between an `(n, 3)` point array and
an `(n, 4, 4)` matrix stack, dispatching on the width of the input.

```python
pts     = np.array([[0.2, 0.5, 0.0], [0.8, 0.5, 0.0]])
as_mats = sp.control_array_type_conversion(pts, to_transformation_matrix=True)
as_pts  = sp.control_array_type_conversion(as_mats)          # translations back out
assert np.allclose(as_pts, pts)
```

```python
scaled = np.eye(4) * 2.0; scaled[3, 3] = 1.0
print(np.diag(sp.extract_scale_matrix(scaled)))                         # [2, 2, 2, 1]
print(np.diag(sp.reset_scale(scaled)))                                  # [1, 1, 1, 1]

print(sp.build_normal_matrix(np.array([0.1, 0.2, 0.3]), normal_scale=2.0)[3, :3])
print(sp.u_vector_to_rotation_matrix(np.array([1.0, 0.0, 0.0])).shape)  # (16,) FLAT
```

> `u_vector_to_rotation_matrix` returns a **flat 16-element** array, not a
> `(4, 4)`.

### The B-spline control map

`get_bspline_map_init` fits a cubic B-spline through the control transforms,
records where each driven object sits along it (parameter + basis weights +
offset), and — when a `Mesh` is supplied — does all of it in surface space.
`bspline_weigh_transformations` then replays that map against a *new* set of
control transforms.

```python
ctrl_pts  = np.array([[0.2, 0.5, 0.0], [0.5, 0.5, 0.0], [0.8, 0.5, 0.0], [0.9, 0.6, 0.0]])
ctrl_mats = sp.control_array_type_conversion(ctrl_pts, to_transformation_matrix=True)
driven_mats = sp.control_array_type_conversion(
    np.array([[0.35, 0.5, 0.0], [0.65, 0.5, 0.0]]), to_transformation_matrix=True
)

bspline_map = sp.get_bspline_map_init(
    (["c0", "c1", "c2", "c3"], ctrl_mats),      # (names, matrices)
    (["d0", "d1"], driven_mats),
    Mesh   = face,
    UVList = uv_list,
)
print(sorted(bspline_map))
print(bspline_map["params"])           # driven position along the spline
print(np.shape(bspline_map["basis"]))  # (driven, controls) weights
print(np.shape(bspline_map["super_sample_plotting"]))
```

Now move the controls and read the driven objects back out:

```python
posed = ctrl_mats.copy()
posed[:, 3, 1] += 0.15                           # slide the controls in V

result = sp.bspline_weigh_transformations(
    {"matrices": posed, "points": sp.control_array_type_conversion(posed)},
    bspline_map,
    Mesh            = face,
    UVList          = uv_list,
    normal_scale    = 1.0,
    return_matrices = True,
)
print(sorted(result))                                # bspline_plot, plotting, transformation

print(np.array(result["plotting"])[0][3, :3])        # surface frame, ON the bowed mesh
print(np.array(result["transformation"])[0][3, :3])  # driven object, offset re-applied
```

Three stages come back, and the distinction matters:

| Key | What it is |
|---|---|
| `plotting` | the surface frame at each driven object's spline parameter (a matrix stack when `Mesh` is given, raw points otherwise) |
| `bspline_plot` | that frame with the basis-weighted control transform composed in |
| `transformation` | the above with each driven object's recorded offset re-applied — the final answer |

The driven objects were authored at `z = 0`, off the bowed surface, so the
recorded offset pulls them back off it: they follow the controls' `+0.15` V
slide while keeping their original standoff. Set `return_matrices=False` to
get bare positions instead of matrices.

Drop `Mesh`/`UVList` from both calls to run the same map in plain world space
with no surface involved:

```python
flat_map = sp.get_bspline_map_init((["c0", "c1", "c2", "c3"], ctrl_mats),
                                   (["d0", "d1"], driven_mats))
flat_result = sp.bspline_weigh_transformations(
    {"matrices": posed, "points": sp.control_array_type_conversion(posed)}, flat_map
)
print(np.array(flat_result["transformation"]))   # positions, not matrices
```

### Refactoring controls onto a new span

When the target rig's controls span a different length or direction, build a
refactor matrix from a start/end pair and re-fit the control stack to it.

```python
span = sp.control_array_type_conversion(
    np.array([[0.1, 0.5, 0.0], [0.9, 0.5, 0.0]]), to_transformation_matrix=True
)
refactor = sp.refactor_control_object_matrices_init(
    ctrl_mats, uvn_scale=[1.0, 1.0, 1.0], refactor_inputs=span, refactor_alignment=False
)
print(sorted(refactor))
print(refactor["scale"], refactor["origin"])
print(refactor["refactored_matrices"][:, 3, :3])

applied = sp.refactor_control_object_matrices(ctrl_mats, refactor)
print(sorted(applied))                            # world, offset
```

`refactor_alignment=True` additionally rotates the stack to match the target
span's direction. An identity `refactor_matrix` short-circuits and returns the
input untouched. `get_bspline_map_init` will run this for you if you pass
`refactor_inputs=`.

### Control-name helpers

Rig-side string wrangling: group controls by root and order them R → Ct → L.

```python
names = ["brow01_R_ctrl", "brow01_Ct_ctrl", "brow01_L_ctrl",
         "lip01_R_ctrl", "lip01_mover"]
print(sp.get_roots(names))           # ['brow', 'lip'] -- digits stripped
print(sp.control_sort(names))        # '_mover'/'_crv' culled
print(sp.get_control_arrays(names))  # {root: sorted controls}
```

### Nested-dict helpers

Used to walk the rig "function data" blobs.

```python
data = {"jaw": {"open": {"world": np.eye(4)}}, "brow": {"world": np.eye(4)}}
print([path for path, _ in sp.find_key(data, "world")])

sp.set_value_by_path(data, ["brow", "world"], np.zeros((4, 4)))
print(data["brow"]["world"].sum())

print(sp.convert_dict_vals_arrays({"m": np.eye(2)}))                    # arrays -> lists
print(sp.convert_dict_vals_arrays({"m": [[1.0, 0.0]]}, to_numpy_array=True))
```

### Batch retargeting

`batch_retarget_init`, `retarget_function_data` and `get_function_pose` drive
the whole thing for a dict of rigs at once. They expect the DCC-side
`function_data` / `rig_args` / `mesh_data` blobs, so there is nothing to build
inline here.

<!-- notest -->

```python
refactored = sp.batch_retarget_init(
    function_data,        # {"function_data": ..., "control_regions": ..., "region_init": ...}
    rig_args,             # {rig: {"spline": {...}, "driven": {...}}}
    mesh_data,            # {rig: {"mesh_obj": MeshData, "uv_list": [UVData], "uv_map_index": 0}}
    uvn_scale          = [1.0, 1.0, 1.0],
    refactor_alignment = False,
)
pose = sp.get_function_pose(refactored, mesh_data, "smile", relative=True)
```

---

## `bspline` — curves

### Build a curve

```python
from cgmath.geometry import BSplineData

cvs = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 0.0],
    [3.0, 2.0, 0.0],
    [4.0, 0.0, 0.0],
    [6.0, 1.0, 0.0],
])

curve = BSplineData(points=cvs, degree=3, periodic=False)

print(curve.count)      # 5 control points
print(curve.degree)     # 3
print(curve.max_param)  # count - degree == 2 for an open curve
print(curve.cv.shape)   # (5, 3) — periodic curves wrap `degree` extra CVs
print(curve.kv)         # unpadded knot vector (Maya convention, m + n - 1)
print(curve.geometry)   # index of each CV into `points`
```

### Evaluate: points and tangents

```python
u = np.linspace(0, curve.max_param, 7)
points, tangents = curve.compute(u)
print(points.shape, tangents.shape)      # (7, 3) (7, 3)

# scalar u collapses the leading axis
p, t = curve.compute(1.0)
print(p.shape, t.shape)                  # (3,) (3,)

# same result, numba-only fast path
points_fast, _ = curve.compute_fast(u)
assert np.allclose(points, points_fast)
```

### Open curves extrapolate outside the domain

```python
# u < 0 or u > max_param extends linearly along the endpoint tangent
p_before, _ = curve.compute(-0.25)
p_after, _ = curve.compute(curve.max_param + 0.25)
print(np.round(p_before, 3), np.round(p_after, 3))
```

### Periodic (closed) curves

```python
ring = BSplineData(points=cvs, degree=3, periodic=True)
print(ring.max_param)  # == count for a periodic curve
print(ring.cv.shape)   # (5 + degree, 3) — first `degree` CVs wrapped

# u wraps automatically
a, _ = ring.compute(0.0)
b, _ = ring.compute(float(ring.max_param))
assert np.allclose(a, b)
```

### `open()` / `close()` flip the topology in place

```python
ring.open()
print(ring.periodic, ring.max_param)  # False 2   (count - degree)
ring.close()
print(ring.periodic, ring.max_param)  # True 5    (count)
```

### Arc length

```python
print(curve.length)                  # fast sampled approximation
print(curve.total_length)            # cached when uniform=True
print(curve.get_length(fast=False))  # precise scipy quad integration
print(curve.get_length(fast=True, samples=500))
```

### `uniform=True` — arc-length parameterization

With `uniform=False` (default) `u` runs `[0, max_param]` in native knot
space. With `uniform=True` `u` runs `[0, 1]` and equal steps in `u` are
equal steps *along the curve*.

```python
arc  = BSplineData(points=cvs.copy(), degree=3, periodic=False, uniform=True)

even = arc.compute(np.linspace(0, 1, 9))[0]
seg  = np.linalg.norm(np.diff(even, axis=0), axis=1)
print(np.round(seg, 4))                   # near-identical segment lengths
```

### `registered=True` — align `u=0` with `points[0]`

Only meaningful on periodic curves: it shifts the parameter origin to the
curve point nearest `points[0]`.

```python
plain = BSplineData(points=cvs, degree=3, periodic=True)
reg   = BSplineData(points=cvs, degree=3, periodic=True, registered=True)

print(np.round(plain.compute(0.0)[0], 3))  # not at points[0]
print(np.round(reg.compute(0.0)[0], 3))    # close to points[0]
```

### Control point parameters

```python
print(np.round(curve.control_point_params, 4))  # one u per CV, monotonic
print(np.round(ring.control_point_params, 4))   # count + 1, last closes the loop
```

### Basis functions

`basis(u)` returns the per-control-point weights — multiply any per-CV
attribute stream by it to interpolate that attribute along the curve.

```python
b = curve.basis(np.linspace(0, curve.max_param, 5))
print(b.shape)                     # (5, n_cv)
print(np.round(b.sum(axis=1), 6))  # partition of unity → all 1.0

# reconstruct positions straight from the basis
assert np.allclose(b @ curve.cv, curve.compute(np.linspace(0, curve.max_param, 5))[0])

# collapse=True folds the wrapped CVs back onto the originals (periodic only)
print(ring.basis(np.array([0.5]), collapse=True).shape)   # (1, 5)
```

### Closest point projection — `sample()`

```python
queries = np.array([[0.0, 3.0, 0.0], [3.0, -1.0, 0.0], [6.0, 3.0, 0.0]])
sd      = curve.sample(queries)

print(sd.points.shape)         # (3, 3) closest point on the curve
print(sd.tangents.shape)       # (3, 3) tangent there
print(np.round(sd.distances, 4))
print(np.round(sd.params, 4))  # u of each projection, in user space
print(sd.basis.shape)          # (3, n_cv)
```

`sample()` also accepts anything with a `.points` attribute:

```python
sd_mesh = curve.sample(cube)
print(sd_mesh.points.shape)      # (8, 3) — one projection per cube vertex
```

### `SampleData.compute()` — transfer per-CV values to the samples

```python
colors = np.array([
    [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0], [0.0, 0.0, 1.0],
])
print(np.round(sd.compute(colors), 3))
print(np.round(sd(colors), 3))       # __call__ is an alias for compute()

sd2           = sd.copy()                      # deep copy of every array
sd2.points[0] = 0.0
assert not np.allclose(sd.points[0], sd2.points[0])
```

### Smoothing control points

```python
noisy = BSplineData(
    points=cvs + np.array([[0, 0.4, 0], [0, -0.4, 0], [0, 0.4, 0],
                           [0, -0.4, 0], [0, 0.4, 0]]),
    degree   = 3,
    periodic = False,
)
noisy.smooth(n=1, polyorder=1, lock_endpoints=True)
print(np.round(noisy.points, 3))
```

### Invalidating caches after editing control points

`points` is stored by reference, so an in-place edit leaves the cached knot
vector, scipy spline and arc-length table stale.

```python
curve.points[0] = [-1.0, 0.0, 0.0]
curve.invalidate()               # lazy: rebuilds on next access
print(np.round(curve.compute(0.0)[0], 3))

curve.points[0] = [0.0, 0.0, 0.0]
curve.rebuild()                  # eager: rebuilds everything now
print(np.round(curve.compute(0.0)[0], 3))

arc.points[-1] = [7.0, 1.0, 0.0]
arc.rebuild_arc_length_table()   # spline + arc-length table only
print(round(arc.total_length, 4))
```

### Pure-scipy evaluation

```python
scipy_curve = BSplineData(points=cvs, degree=3, periodic=False, use_numba=False)
assert np.allclose(scipy_curve.compute(u)[0], curve.compute(u)[0])
```

---

## `bspline_patch` — surfaces

### Build a patch from a control grid

Control points are a `(nu, nv, dims)` grid — a *tensor product* surface,
so every parameter, degree and periodic flag comes in a `_u` / `_v` pair.

```python
from cgmath.geometry import BSplinePatchData

gx, gy = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5), indexing="ij")
grid = np.stack([gx, gy, 0.35 * (gx**2 - gy**2)], axis=-1)   # a saddle

patch = BSplinePatchData(
    points=grid,
    degree_u=3, degree_v=3,
    periodic_u=False, periodic_v=False,
)

print(patch.count_u, patch.count_v)          # 5 5
print(patch.max_param_u, patch.max_param_v)  # 2 2
print(patch.cv.shape)                        # (5, 5, 3)
print(patch.kv_u.shape, patch.kv_v.shape)
```

### Evaluate: positions and partial derivatives

```python
uu  = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
vv  = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

pts = patch.evaluate(uu, vv)                 # positions only, fastest
print(pts.shape)                             # (5, 3)

pts, du, dv = patch.compute(uu, vv)          # + dS/du and dS/dv
print(du.shape, dv.shape)                    # (5, 3) (5, 3)

normals = np.cross(du, dv)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
print(np.round(normals[2], 3))

# compute_fast() is the numba-only variant of compute()
p2, _, _ = patch.compute_fast(uu, vv)
assert np.allclose(pts, p2)

# scalars work too
print(np.round(patch.evaluate(1.0, 1.0), 3))
```

### Sampling a full (u, v) grid

```python
n = 12
ug, vg = np.meshgrid(
    np.linspace(0, patch.max_param_u, n),
    np.linspace(0, patch.max_param_v, n),
    indexing="ij",
)
surface = patch.evaluate(ug.ravel(), vg.ravel()).reshape(n, n, 3)
print(surface.shape)
```

### Basis functions

```python
b = patch.basis(uu, vv)
print(b.shape)                     # (5, count_u * count_v) — flattened
print(np.round(b.sum(axis=1), 6))  # partition of unity

flat_cv = patch.cv.reshape(-1, 3)
assert np.allclose(b @ flat_cv, patch.evaluate(uu, vv))
```

### Closest point projection — `sample()`

```python
queries = np.array([[0.0, 0.0, 2.0], [1.0, -1.0, -1.5], [-1.5, 1.5, 0.5]])
ps      = patch.sample(queries)

print(ps.points.shape)          # (3, 3)
print(ps.params.shape)          # (3, 2) → [:, 0] = u, [:, 1] = v
print(ps.tangents.shape)        # (3, 2, 3) → [:, 0] = dS/du, [:, 1] = dS/dv
print(np.round(ps.distances, 4))
print(np.round(ps.normals, 3))  # cross(dS/du, dS/dv), normalized
print(ps.basis.shape)           # (3, count_u * count_v)
```

`PatchSampleData.compute()` takes a `(nu, nv, d)` or `(nu*nv, d)` value grid:

```python
heat = np.linspace(0.0, 1.0, patch.count_u * patch.count_v).reshape(
    patch.count_u, patch.count_v, 1
)
print(np.round(ps.compute(heat).ravel(), 3))
print(np.round(ps(heat).ravel(), 3))       # __call__ alias
ps_copy = ps.copy()
```

### Raycast — `raycast()`

```python
origins    = np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 3.0], [9.0, 9.0, 3.0]])
directions = np.array([[0.0, 0.0, -1.0]] * 3)

rc = patch.raycast(origins, directions)
print(rc.hit)                     # [ True  True False]
print(np.round(rc.distances, 4))  # ray t at hits, NaN at misses
print(np.round(rc.params[rc.hit], 4))
print(np.round(rc.normals[rc.hit], 3))
print(rc.occluded)                # True where the ray faces into the surface
print(rc.points.shape)            # misses fall back to the ray origin
```

A single direction vector is tiled across all origins, and `twosided=False`
rejects back-face hits:

```python
rc_one = patch.raycast(origins, np.array([0.0, 0.0, -1.0]))
print(rc_one.hit)

rc_front = patch.raycast(origins, np.array([0.0, 0.0, -1.0]), twosided=False)
print(rc_front.hit)
```

`raycast` also accepts a `MeshData` as origins; with `directions=None` it uses
that mesh's vertex normals:

```python
rc_mesh = patch.raycast(cube)
print(rc_mesh.hit.shape)            # (8,) — one ray per cube vertex
print(rc_mesh.compute(heat).shape)  # (8, 1) — attribute lookup at hits
```

### Lengths and area

```python
print(round(patch.length_u, 4), round(patch.length_v, 4))
print(round(patch.get_length_u(fast=False), 4))    # scipy quad, precise
print(round(patch.total_length_u, 4), round(patch.total_length_v, 4))
```

`get_area()` uses composite Simpson's rule when `samples` is **odd** and
falls back to a coarser Riemann sum when it is even. `area` is
`get_area()` with the default `samples=50` — even, so pass an odd count
when you want accuracy.

```python
print(round(patch.area, 4))                  # samples=50 → Riemann
print(round(patch.get_area(samples=51), 4))  # odd → Simpson
print(round(patch.get_area(samples=201), 4))
```

### Control point parameters, per direction

```python
cp_u, cp_v = patch.control_point_params
print(np.round(cp_u, 4))
print(np.round(cp_v, 4))
```

### Periodic directions — a tube

```python
theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
rings = []
for z in np.linspace(-1, 1, 4):
    rings.append(np.stack([np.cos(theta), np.sin(theta), np.full_like(theta, z)], -1))
tube_cv = np.stack(rings, axis=0)          # (4, 8, 3) → u = along, v = around

tube = BSplinePatchData(
    points=tube_cv,
    degree_u=2, degree_v=3,
    periodic_u=False, periodic_v=True,
)
print(tube.max_param_u, tube.max_param_v)  # 2 8
print(tube.cv.shape)                       # (4, 8 + degree_v, 3)

# v wraps
assert np.allclose(tube.evaluate(1.0, 0.0), tube.evaluate(1.0, float(tube.max_param_v)))
```

### Flipping periodicity per direction

```python
tube.open_v()
print(tube.periodic_v, tube.max_param_v)
tube.close_v()
print(tube.periodic_v, tube.max_param_v)
tube.close_u(); tube.open_u()
```

### Arc-length (`uniform_*`) and registration

```python
uni = BSplinePatchData(
    points=grid,
    degree_u=3, degree_v=3,
    periodic_u=False, periodic_v=False,
    uniform_u=True, uniform_v=True,
)
print(np.round(uni.evaluate(np.array([0.0, 0.5, 1.0]), np.array([0.5, 0.5, 0.5])), 3))

reg = BSplinePatchData(
    points=tube_cv,
    degree_u=2, degree_v=3,
    periodic_u=False, periodic_v=True,
    registered_v=True,
)
print(np.round(tube.evaluate(1.0, 0.0), 3))  # v=0 lands wherever the knots fall
print(np.round(reg.evaluate(1.0, 0.0), 3))   # v=0 now sits under tube_cv[:, 0]
print(np.round(tube_cv[1, 0], 3))
```

### Smoothing and cache control

```python
bumpy = BSplinePatchData(
    points=grid + np.random.default_rng(0).normal(0, 0.05, grid.shape),
    degree_u=3, degree_v=3,
    periodic_u=False, periodic_v=False,
)
bumpy.smooth(n_u=1, n_v=1, polyorder=2)

bumpy.points[0, 0] = [-2.0, -2.0, 1.0]
bumpy.invalidate()                # lazy rebuild
print(np.round(bumpy.evaluate(0.0, 0.0), 3))
bumpy.rebuild()                   # eager rebuild
bumpy.rebuild_arc_length_table()  # spline + arc-length tables only
```

---

## `_saddle_surface` — quad-aware surface queries

Every quad in this library is treated as a **bilinear (saddle) patch**, not
as two triangles. `MeshData.sample()`, `MeshData.raycast()` and
`MeshData.area` are the friendly wrappers; the functions below are the
engine underneath, and take raw `points` / `geometry` arrays.

### `integrate()` — true bilinear face areas

```python
from cgmath.geometry._saddle_surface import integrate

quad = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
geom = np.array([[0, 1, 2, 3]])
print(integrate(quad, geom, samples=100))          # [1.] — one area per face

# a warped (saddle) quad has more area than the flat one
warped       = quad.copy()
warped[2, 2] = 0.5
print(integrate(warped, geom, samples=100))                            # [1.079...]

print(integrate(quad, geom, samples=100, method="gauss"))
print(np.round(integrate(cube.points, cube.geometry, samples=20), 4))  # per-face
```

Triangles are accepted (padded to quads internally); n-gons raise:

```python
tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
print(integrate(tri, np.array([[0, 1, 2]]), samples=100))   # [0.5]

try:
    integrate(np.zeros((5, 3)), np.array([[0, 1, 2, 3, 4]]))
except ValueError as e:
    print(e)                                                 # "ngons detected"
```

### `sample()` — closest point on the quad surface

```python
from cgmath.geometry._saddle_surface import sample

queries = np.array([
    [0.0, 0.0, 1.5],       # outside, above the +Z face
    [0.6, 0.0, 0.0],       # just outside the +X face
    [0.0, 0.2, 0.0],       # inside the cube
])
sd = sample(queries, cube.points, cube.geometry)

print(np.round(sd.projections, 4))  # closest point on the surface
print(np.round(sd.distances, 4))    # distance to it
print(sd.indices)                   # winning face index per query
print(np.round(sd.uvs, 4))          # bilinear (u, v) inside that face
print(np.round(sd.weights, 4))      # the 4 corner weights, sum to 1
print(sd.geometry)                  # the 4 corner vertex indices
```

Pass vertex normals to get interpolated normals and an inside/outside test:

```python
normals = cube.get_vertex_normals()
sd      = sample(queries, cube.points, cube.geometry, normals=normals)
print(np.round(sd.normals, 3))
print(sd.occluded)                     # [False False  True] — True == inside
```

`surface_normals=` switches from flat bilinear patches to PN-Quad bicubic
Bézier patches — slower, but curves the patch to match the vertex normals:

```python
sd_bezier = sample(
    queries, cube.points, cube.geometry,
    normals=normals, surface_normals=normals,
)
print(np.round(sd_bezier.distances, 4))
```

### `SampleData.compute()` — transfer any per-vertex attribute

```python
attr = cube.points * 10.0                # any (V, k) stream
print(np.round(sd.compute(attr), 3))
print(np.round(sd(attr), 3))             # __call__ alias
```

### `SampleData.remap()` — collapse per-UV-point samples to per-vertex

`remap()` expects **one sample per UV point** of `dst_uv` and returns **one
sample per vertex** of `dst_mesh` — the UV split that landed closest wins.
`.geometry` comes back re-keyed to `src_mesh.geometry`.

```python
uv_to_vert                  = np.zeros(cube_uv.points.shape[0], dtype=int)
uv_to_vert[cube_uv.indices] = cube.indices
uv_queries                  = cube.points[uv_to_vert] * 1.2         # one query per UV point

sd_uv = sample(uv_queries, cube.points, cube.geometry, normals=normals)
print(sd_uv.distances.shape)          # (14,) — one per UV point

remapped = sd_uv.remap(cube, cube, cube_uv)
print(remapped.distances.shape)  # (8,) — one per mesh vertex
print(remapped.geometry.shape)   # (8, 4) — re-keyed to src_mesh.geometry
```

### `raycast()` — rays against bilinear / Bézier patches

```python
from cgmath.geometry._saddle_surface import raycast

origins    = np.array([[0.0, 0.0, 2.0], [0.25, 0.25, 2.0], [5.0, 5.0, 2.0]])
directions = np.array([[0.0, 0.0, -1.0]] * 3)

rc = raycast(cube.points, cube.points, cube.geometry, origins, directions,
             normals=normals)
print(rc.hit)                     # [ True  True False]
print(np.round(rc.distances, 4))  # ray t; NaN at misses
print(rc.indices)                 # hit face index, -1 for misses
print(np.round(rc.uvs[rc.hit], 4))
print(np.round(rc.normals[rc.hit], 3))
print(rc.occluded)                # ray entering the surface front-face
print(np.round(rc.compute(attr)[rc.hit], 3))
```

```python
# forward_only=False also casts backwards and keeps the closer hit
rc_both = raycast(cube.points, cube.points, cube.geometry, origins, directions,
                  normals=normals, forward_only=False)
print(rc_both.hit)

# twosided=False rejects back-face hits
rc_front = raycast(cube.points, cube.points, cube.geometry, origins, directions,
                   normals=normals, twosided=False)
print(rc_front.hit)

# surface_normals=... uses PN-Quad bicubic Bézier patches
rc_bezier = raycast(cube.points, cube.points, cube.geometry, origins, directions,
                    normals=normals, surface_normals=normals)
print(rc_bezier.hit)
```

### The `MeshData` wrappers

```python
print(np.round(cube.sample(queries).distances, 4))
print(np.round(cube.sample(queries, method="bezier").distances, 4))
print(cube.raycast(origins, directions).hit)
print(np.round(cube.area, 4))                  # sum of integrate() over all faces
print(np.round(cube.get_face_areas(), 4))
```

---

## `sdf` — signed distance fields, functional API

An SDF is just a `float` array sampled on a regular grid: negative inside
the shape, positive outside, zero on the surface. Build the grid, evaluate
primitives on it, combine with `min`/`max` CSG, then polygonise.

### Grid + primitive + mesh, end to end

```python
from cgmath.geometry.sdf import (
    dual_marching_cubes, eval_sphere, make_grid,
)

shape  = (32, 32, 32)
bounds = (np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0]))
X, Y, Z, origin, spacing = make_grid(shape, bounds)
print(X.shape, np.round(origin, 3), np.round(spacing, 4))

field = eval_sphere(X, Y, Z, radius=1.0)
print(field.shape, round(float(field.min()), 3), round(float(field.max()), 3))

mesh = dual_marching_cubes(field, iso_value=0.0,
                           grid_origin=origin, grid_spacing=spacing,
                           name="sphere")
print(mesh.point_count, mesh.face_count, mesh.name)
print(np.all(mesh.counts == 4))     # dual marching cubes is quad-only
```

### The primitive evaluators

```python
from cgmath.geometry.sdf import eval_box, eval_cylinder

box = eval_box(X, Y, Z, half_extents=np.array([0.8, 0.4, 0.6]))
cyl = eval_cylinder(X, Y, Z, radius=0.5, height=1.5, axis=1)   # 0=X, 1=Y, 2=Z
print(round(float(box.min()), 3), round(float(cyl.min()), 3))
```

### `transform_points()` — place a primitive in the world

Primitives are always evaluated at the origin. To move one, transform the
*grid* by the inverse of what you want the shape to do.

```python
from cgmath.geometry.sdf import transform_points

Xt, Yt, Zt = transform_points(
    X, Y, Z,
    translation = np.array([0.5, 0.0, 0.0]),
    rotation    = np.array([30.0, 0.0, 0.0]),  # euler degrees, XYZ order
    scale       = 1.0,                         # scalar (exact) or (3,) (approximate)
)
fin = eval_box(Xt, Yt, Zt, half_extents=np.array([0.15, 1.0, 0.15]))
print(round(float(fin.min()), 3))
```

### CSG

```python
from cgmath.geometry.sdf import (
    sdf_difference, sdf_intersection, sdf_smooth_difference,
    sdf_smooth_intersection, sdf_smooth_union, sdf_union,
)

a = eval_sphere(X, Y, Z, radius=1.0)
b = fin

print(round(float(sdf_union(a, b).min()), 4))         # min(a, b)
print(round(float(sdf_intersection(a, b).min()), 4))  # max(a, b)
print(round(float(sdf_difference(a, b).min()), 4))    # max(a, -b)

# smooth variants fillet the seam; k is the blend radius
print(round(float(sdf_smooth_union(a, b, k=0.25).min()), 4))
print(round(float(sdf_smooth_intersection(a, b, k=0.25).min()), 4))
print(round(float(sdf_smooth_difference(a, b, k=0.25).min()), 4))
```

```python
combined = sdf_smooth_union(a, b, k=0.2)
blob     = dual_marching_cubes(combined, grid_origin=origin, grid_spacing=spacing)
print(blob.point_count, blob.face_count, blob.valid)
```

### Ready-made test fields

`sphere_sdf` / `box_sdf` / `cylinder_sdf` build the grid *and* the field in
one call, returning `(field, grid_origin, grid_spacing)` ready for
`dual_marching_cubes`.

```python
from cgmath.geometry.sdf import box_sdf, cylinder_sdf, sphere_sdf

f, o, s = sphere_sdf((32, 32, 32), radius=1.0)
print(dual_marching_cubes(f, grid_origin=o, grid_spacing=s).point_count)

f, o, s = box_sdf((32, 32, 32), half_extents=np.array([0.5, 0.3, 0.4]))
print(dual_marching_cubes(f, grid_origin=o, grid_spacing=s).point_count)

f, o, s = cylinder_sdf((32, 32, 32), radius=0.4, height=1.2, axis=1,
                       bounds=(np.array([-1.0] * 3), np.array([1.0] * 3)))
print(dual_marching_cubes(f, grid_origin=o, grid_spacing=s).point_count)
```

### `iso_value` — offset the surface

```python
shell = dual_marching_cubes(a, iso_value=0.25,
                            grid_origin=origin, grid_spacing=spacing)
print(shell.point_count)              # a sphere inflated by 0.25
```

---

## `sdf` — signed distance fields, OOP API

`SDFSphere`, `SDFBox` and `SDFCylinder` subclass `TransformData`, so they
carry the whole Maya-style `translate` / `rotate` / `scale` surface. A
`DMCField` collects them into a CSG stack and lazily polygonises it.

### Primitives are transforms

```python
from cgmath.geometry.sdf import SDFBox, SDFCylinder, SDFSphere

sphere = SDFSphere(radius=1.0, name="dome")
box    = SDFBox(half_extents=[0.15, 1.0, 0.15], name="fin")
cyl    = SDFCylinder(radius=0.3, height=3.0, axis=1, name="hole")

box.translate = [0.5, 0.0, 0.0]
box.rotate    = [30.0, 0.0, 0.0]            # euler degrees
sphere.scale  = [1.0, 0.6, 1.0]

print(sphere.name, sphere.radius, np.round(sphere.scale, 3))
print(np.round(box.translate, 3), np.round(box.rotate, 3))
print(box.half_extents, cyl.radius, cyl.height, cyl.axis)
print(np.round(box.world_matrix, 3))
```

### `bounding_box()`, `evaluate()` and `sample()`

`evaluate()` reads *local* coordinates (the primitive at the origin);
`sample()` reads *world* coordinates and applies the transform for you.

```python
lo, hi = box.bounding_box()
print(np.round(lo, 3), np.round(hi, 3))

print(round(float(sphere.evaluate(X, Y, Z).min()), 4))  # local, ignores scale
print(round(float(sphere.sample(X, Y, Z).min()), 4))    # world, honours SRT
```

### `DMCField` — CSG stack with a lazy mesh

`resolution` is *subdivisions per world unit*; bounds are auto-derived from
the primitives' bounding boxes plus 10% padding.

```python
from cgmath.geometry.sdf import DMCField

field = DMCField(resolution=12, name="widget")
field.add(SDFSphere(radius=1.0, name="body"))
field.add(box, smoothing=0.15)  # smoothing > 0 → smooth union
field.subtract(cyl)             # boolean difference

print([p.name for p in field.primitives])
print(field.resolution, field.iso_value, field.name)
print(np.round(field.position, 3))

mesh = field.mesh_data                   # computed here, then cached
print(mesh.point_count, mesh.face_count, mesh.valid)
```

`add` / `subtract` / `intersect` / `remove` all return the field, so they chain:

```python
chained = (
    DMCField(resolution=10, name="chained")
    .add(SDFSphere(radius=1.0))
    .intersect(SDFBox(half_extents=[0.7, 0.7, 0.7]))
    .subtract(SDFCylinder(radius=0.3, height=3.0))
)
print(chained.mesh_data.point_count)
```

The stack itself is a list of private `CSGNode(primitive, operation, smoothing)`
records; `field.primitives` is the public view of it and `CSGOp` names the four
operations `add` / `intersect` / `subtract` can record.

```python
from cgmath.geometry.sdf import CSGOp

print([e.value for e in CSGOp])
```

### The mesh is lazy — edit a primitive and just read it again

```python
before             = field.mesh_data
sphere_body        = field.primitives[0]
sphere_body.radius = 1.4              # marks the field dirty
after              = field.mesh_data  # recomputed automatically
print(before.point_count, after.point_count)

assert field.mesh_data is after          # nothing changed → cache hit
```

Every mutating knob invalidates the cache: primitive `radius` /
`half_extents` / `height` / `axis` / `translate` / `rotate` / `scale`, and
the field's `resolution` / `position` / `iso_value` / `name`.

```python
field.resolution = 8
field.iso_value  = 0.05
field.position   = [0.0, 0.25, 0.0]
print(field.mesh_data.point_count)
```

### `remove()` and export

```python
field.remove(cyl)
print([p.name for p in field.primitives])

path = os.path.join(tmp, "widget.obj")
field.to_obj(path)
print(os.path.exists(path))
```

An empty field yields an empty mesh rather than raising:

```python
print(DMCField().mesh_data.point_count)   # 0
```

---

## `_cdt` — constrained Delaunay triangulation

N-gons (faces with 5+ vertices) cannot be fanned safely: a naive fan spills
outside concave faces and ignores holes. `_cdt` runs a **constrained
Delaunay triangulation** instead — scipy Delaunay, then Sloan constraint
insertion so every boundary edge survives, then edge flips to restore the
Delaunay property, a Maya-compatible cocircular tiebreak, and a flood fill
that deletes triangles outside the outline or inside a hole.

You rarely call it directly: `MeshData.get_triangulate_rules()` routes every
n-gon face through it and stashes the result in `TriangulateRules.ngon_tris`.

### Through `MeshData`

```python
pent = MeshData(
    points=np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0],
        [1.0, 0.6, 0.0],                       # concave notch
        [0.0, 2.0, 0.0],
    ]),
    indices = np.array([0, 1, 2, 3, 4]),
    counts  = np.array([5]),
    name    = "pentagon",
)
print(pent.ngons)                              # 1 — one n-gon face

rules = pent.get_triangulate_rules()
print(rules.ngon_tris)    # {0: (T, 3) local indices}

pent.triangulate(rules)
print(pent.face_count, np.all(pent.counts == 3))
print(pent.points.shape)  # no new points added
```

### Calling `triangulate_ngon()` directly

Returns **local** indices into `face_vertices + holes`, so the same result
can be reused for the matching UV face.

```python
from cgmath.geometry._cdt import triangulate_ngon

pts = np.array([
    [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0],
    [1.0, 0.6, 0.0], [0.0, 2.0, 0.0],
])
tris = triangulate_ngon(pts, [0, 1, 2, 3, 4])
print(tris)                    # (3, 3) local indices
print(tris.shape[0] == 5 - 2)  # n - 2 triangles, no holes
```

With a hole boundary, the count becomes `n_outer + n_hole` triangles:

```python
outer = [0, 1, 2, 3]
hole  = [4, 5, 6, 7]
pts = np.array([
    [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0], [0.0, 3.0, 0.0],
    [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 1.0, 0.0],
])
tris = triangulate_ngon(pts, outer, [hole])
print(tris.shape)                              # (8, 3)
```

### The helpers

```python
from cgmath.geometry._cdt import flatten_3d_to_2d, get_boundary_edges

print(sorted(get_boundary_edges([0, 1, 2, 3])))

tilted = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
pts_2d, axis_u, axis_v, origin_3d = flatten_3d_to_2d(tilted)
print(pts_2d.shape, np.round(axis_u, 3), np.round(axis_v, 3), np.round(origin_3d, 3))
```

---

## `resample` — `MeshDataResampler`

One object that transfers every mesh-dependent data type from a source
topology onto a destination topology.

```python
from cgmath.geometry.resample import MeshDataResampler, ResampleMode

resampler = MeshDataResampler(src_grid, dst_grid)
resampler.src_mesh, resampler.dst_mesh          # what it was built with
resampler.src_uv, resampler.dst_uv              # None here
```

### Modes

| `ResampleMode` | Correspondence | Needs |
|---|---|---|
| `SPATIAL` | closest point on the source surface | nothing |
| `UV` | matching UV coordinates | `src_uv` + `dst_uv` |
| `ROBUST_BILINEAR` | closest-surface + Laplacian inpaint, quad-native | nothing |

Each enum *value* is the options dataclass for that mode:

```python
print(ResampleMode.SPATIAL.value.__name__)          # SpatialSkinTransferOptions
print(ResampleMode.UV.value.__name__)               # UVSkinTransferOptions
print(ResampleMode.ROBUST_BILINEAR.value.__name__)  # RobustBilinearSkinTransferOptions
print([m.name for m in ResampleMode])
```

### Sample data is computed once and cached

```python
sample = resampler.spatial_sample_data           # == get_sample_data(ResampleMode.SPATIAL)
assert resampler.spatial_sample_data is sample   # cached by (mode, method)

print(sample.projections.shape)  # (dst points, 3) closest point on the source
print(sample.distances.shape)    # (dst points,)  distance to that point
print(sample.weights.shape)      # (dst points, 4) patch weights
print(sample.geometry.shape)     # (dst points, 4) source vertex ids, -1 padded
print(sample.normals.shape)      # (dst points, 3) source normal at the projection
```

Calling it interpolates any per-source-vertex array onto the target:

```python
transferred = sample(src_grid.points)            # -> (dst points, 3)
assert transferred.shape == (dst_grid.point_count, 3)
```

### UV mode

Needs a `UVData` on both sides. `MeshData.to_uvdata` makes a throwaway planar
projection.

```python
src_uv = src_grid.to_uvdata(axis="z")
dst_uv = dst_grid.to_uvdata(axis="z")

uv_resampler = MeshDataResampler(src_grid, dst_grid, src_uv=src_uv, dst_uv=dst_uv)
uv_resampler.uv_sample_data.projections.shape    # UV projections are 2-D!

try:
    MeshDataResampler(src_grid, dst_grid).uv_sample_data
except RuntimeError as err:
    print(err)                                   # "UV data not supplied."
```

### `resample_mesh`

Push the source mesh's *shape* onto the destination *topology*.

```python
warped        = src_grid.copy()
warped.points = warped.points + np.array([0.0, 0.0, 0.3])

out = resampler.resample_mesh(warped, mode=ResampleMode.SPATIAL)
assert out.point_count == dst_grid.point_count   # destination topology
assert np.allclose(out.points[:, 2], 0.3)        # source shape
```

`maintain_offset` keeps each destination vertex at its original signed
standoff from the source surface; `orient_offset` recomputes the offset
direction from `mesh_data`'s normals instead of reusing the cached ones.

```python
floating         = quad_grid(7, 7, spacing=4.0 / 6.0, z=0.5)
offset_resampler = MeshDataResampler(src_grid, floating)

glued = offset_resampler.resample_mesh(maintain_offset=False)
kept  = offset_resampler.resample_mesh(maintain_offset=True, orient_offset=True)
print(glued.points[:, 2].mean(), kept.points[:, 2].mean())   # 0.0 vs 0.5
```

`maintain_offset` is rejected in UV mode — UV sample data has 2-D projections
and no normals to offset along.

```python
try:
    uv_resampler.resample_mesh(mode=ResampleMode.UV, maintain_offset=True)
except ValueError as err:
    print(err)
```

### `SampleMethod` — BILINEAR vs BEZIER

`BILINEAR` treats each quad as a flat patch; `BEZIER` fits a PN-Quad bicubic
patch using the vertex normals, which is more accurate on curved surfaces.

```python
from cgmath.geometry.mesh import SampleMethod

bowed        = quad_grid(5, 5, spacing=1.0)
pts          = bowed.points.copy()
pts[:, 2]    = 0.4 * np.sin(np.pi * pts[:, 0] / 4.0)
bowed.points = pts

curved = MeshDataResampler(bowed, quad_grid(9, 9, spacing=0.5))
flat   = curved.resample_mesh(method=SampleMethod.BILINEAR)
smooth = curved.resample_mesh(method=SampleMethod.BEZIER)
print(np.abs(flat.points - smooth.points).max())     # > 0 on a curved source
```

### `resample_morph_target`

```python
from cgmath.geometry import MorphData

offsets     = np.zeros_like(src_grid.points)
offsets[12] = [0.0, 0.0, 1.0]                       # poke one vertex
blink       = MorphData(name="poke", offsets=offsets, indices=np.arange(src_grid.point_count))

moved = resampler.resample_morph_target(blink, mode=ResampleMode.SPATIAL)
print(type(moved).__name__, moved.offsets.shape, moved.indices.size)
```

Pass a `MeshData` instead of a `MorphData` (same topology as the source) and
you get a posed `MeshData` back:

```python
posed_src        = src_grid.copy()
posed_src.points = posed_src.points + offsets
posed_dst        = resampler.resample_morph_target(posed_src, mode=ResampleMode.SPATIAL)
print(type(posed_dst).__name__, posed_dst.point_count)
```

`tolerance` / `use_neighbors` control the offset pruning done on the result.
`ROBUST_BILINEAR` is not supported for morph targets:

```python
try:
    resampler.resample_morph_target(blink, mode=ResampleMode.ROBUST_BILINEAR)
except NotImplementedError as err:
    print(err)
```

### `resample_skin_weights`

```python
from cgmath.geometry import SkinData

x         = src_grid.points[:, 0]
t         = (x - x.min()) / (x.max() - x.min())
ramp_skin = SkinData(weights=np.column_stack([1.0 - t, t]), influences=["a", "b"])

dst_skin  = resampler.resample_skin_weights(ramp_skin, mode=ResampleMode.SPATIAL)
print(dst_skin.weights.shape, dst_skin.influences)
assert np.allclose(dst_skin.weights.sum(axis=1), 1.0)
```

Pass `options` to cap influences (and to feed the robust algorithm):

```python
from cgmath.geometry.resample import SpatialSkinTransferOptions

capped = resampler.resample_skin_weights(
    ramp_skin, mode=ResampleMode.SPATIAL, options=SpatialSkinTransferOptions(max_influences=1)
)
print(capped.get_max_influences())               # 1
```

Each mode takes its own options class — `ResampleMode.UV` pairs with
`UVSkinTransferOptions`:

```python
from cgmath.geometry.resample import UVSkinTransferOptions

uv_capped = uv_resampler.resample_skin_weights(
    ramp_skin, mode=ResampleMode.UV, options=UVSkinTransferOptions(max_influences=1)
)
print(uv_capped.get_max_influences())            # 1
```

`CompactSkinData` in, `CompactSkinData` out:

```python
compact = resampler.resample_skin_weights(ramp_skin.to_compact_skin_data())
print(type(compact).__name__, compact.max_influences)
```

### `resample_map` — `MapData` and `GeomSubsetData`

Painted maps and geometry subsets are round-tripped through `SkinData`, so
they reuse the same machinery. `**kwargs` go straight to
`resample_skin_weights`.

```python
from cgmath.geometry.map import GeomSubsetData, MapData

mask     = SkinData(weights=t.reshape(-1, 1), influences=["mask"])
map_data = MapData.from_skin_data(mask, mesh_data=src_grid, name="wrinkle", component_type="v")

out_map  = resampler.resample_map(map_data)
print(type(out_map).__name__, out_map.name, out_map.values.shape)

subset     = GeomSubsetData.from_skin_data(mask, mesh_data=src_grid, name="cheek", component_type="f")
out_subset = resampler.resample_map(subset)
print(type(out_subset).__name__, out_subset.indices.size, dst_grid.face_count)
```

---

## Robust skin weight transfer

*Weight inpainting*: match each
target vertex to the closest point on the source surface, reject matches that
are too far or whose normals disagree, then solve for the unmatched vertices
and smooth the seam.

### `robust_skinweights_transfer_bilinear` — quad-native

No triangulation, no third-party dependency. Returns a raw weights array.

```python
from cgmath.geometry.robust_skinweights_transfer_bilinear import (
    find_matches_closest_surface,
    robust_skinweights_transfer_bilinear,
    RobustBilinearSkinTransferOptions,
)

weights = robust_skinweights_transfer_bilinear(src_grid, ramp_skin, dst_grid)
print(weights.shape)
assert np.allclose(weights.sum(axis=1), 1.0)
```

Options:

| Field | Default | Meaning |
|---|---|---|
| `max_influences` | `4` | Only applied by the resampler, not by the function. |
| `search_radius` | `0.05` | Fraction of the **target** bounding-box diagonal. |
| `normal_threshold` | `30` | Degrees; matches beyond this are rejected. |
| `inpaint_iterations` | `10` | Diffusion passes over the unmatched vertices. |
| `smooth_iterations` | `10` | Blur passes over the inpainted region. `0` disables. |
| `smooth_strength` | `0.1` | Per-iteration blend toward the neighbour average. |

```python
options = RobustBilinearSkinTransferOptions(search_radius=0.1, smooth_iterations=0)
print(robust_skinweights_transfer_bilinear(src_grid, ramp_skin, dst_grid, options).shape)
```

Reuse a projection you already own instead of paying for it twice:

```python
cached = src_grid.sample(dst_grid)
same = robust_skinweights_transfer_bilinear(
    src_grid, ramp_skin, dst_grid, sample_data=cached
)
assert np.array_equal(same, weights)
```

`find_matches_closest_surface` is the matching stage on its own — useful to
see *which* vertices the search radius and normal threshold rejected.

```python
matched, raw = find_matches_closest_surface(cached, dst_grid, ramp_skin.weights, 1.0, 30.0)
print(matched.all(), raw.shape)                  # True -- coincident surfaces

flipped         = cached.copy()
flipped.normals = -dst_grid.get_vertex_normals(angle_weighted=True, area_weighted=False)
opposed, _ = find_matches_closest_surface(flipped, dst_grid, ramp_skin.weights, 1.0, 30.0)
print(opposed.any())                             # False -- normals disagree, all rejected
```

Errors it raises:

```python
far_away = quad_grid(5, 5, spacing=1.0, z=100.0)
try:
    robust_skinweights_transfer_bilinear(src_grid, ramp_skin, far_away)
except ValueError as err:
    print("out of range ->", err)

pentagon = MeshData(
    indices = np.arange(5, dtype=np.int32),
    counts  = np.array([5], dtype=np.int32),
    points  = np.array([[0.0, 0, 0], [1, 0, 0], [1.5, 1, 0], [0.5, 1.5, 0], [-0.5, 1, 0]]),
)
try:
    robust_skinweights_transfer_bilinear(
        pentagon, SkinData(weights=np.ones((5, 1)), influences=["a"]), dst_grid
    )
except NotImplementedError as err:
    print("n-gons ->", err)
```

It also raises `ValueError` if either mesh has unreferenced vertices — the
output row order would silently stop matching the input.

### Which one do I want?

| Situation | Use |
|---|---|
| Same topology family, dense correspondence, just need it fast | `MeshDataResampler` + `ResampleMode.SPATIAL` |
| Meshes share a UV layout | `MeshDataResampler` + `ResampleMode.UV` |
| Different topology, gaps/holes/clothing that misses the body | `ResampleMode.ROBUST_BILINEAR` |

The plain `SPATIAL`/`UV` modes interpolate weights for *every* target vertex,
including ones that project nowhere sensible. `ROBUST_BILINEAR` rejects those
and solves for them instead — that is the whole difference. Going through the
resampler also gets you the cached projection, `max_influences` capping and
`CompactSkinData` handling for free:

```python
robust = MeshDataResampler(src_grid, dst_grid).resample_skin_weights(
    ramp_skin,
    mode    = ResampleMode.ROBUST_BILINEAR,
    options = RobustBilinearSkinTransferOptions(max_influences=2),
)
print(robust.weights.shape, robust.get_max_influences())
```

---

## Deprecated shims at `geometry/`

Six modules at the top of `geometry/` are one-release re-export shims. Each
emits a `DeprecationWarning` on import and will be deleted.

| Old import | New import |
|---|---|
| `cgmath.geometry.delta_mush` | `cgmath.geometry.deform.delta_mush` |
| `cgmath.geometry.ffd` | `cgmath.geometry.deform.ffd` |
| `cgmath.geometry.patch_relax` | `cgmath.geometry.deform.patch_relax` |
| `cgmath.geometry.camera` | `cgmath.render.camera` |
| `cgmath.geometry.raytracer` | `cgmath.render.raytracer` |
| `cgmath.geometry.texture` | `cgmath.render.texture` |

The three deformers are re-exported from the `cgmath.geometry` namespace
directly, so most callers just drop the submodule:

```python
from cgmath.geometry import DeltaMushData, PatchRelaxData
from cgmath.geometry.deform import FFDData
from cgmath.render import look_at, render
```

The warning is real — catch it to prove it:

```python
import importlib
import warnings

for name in ("delta_mush", "ffd", "patch_relax", "camera", "raytracer", "texture"):
    module = importlib.import_module("cgmath.geometry." + name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(module)
    print(name, "->", caught[0].category.__name__)
```

Everything else at `geometry/` top level (`mesh`, `map`, `bspline`,
`bspline_patch`, `morph_target`, `skin_weights`, `sdf`, `pack`, `resample`,
`robust_skinweights_transfer_bilinear`, `surface_plotting`) is real code, not a shim.

