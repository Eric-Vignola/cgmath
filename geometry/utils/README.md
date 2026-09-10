# `cgmath.geometry.utils` — the numba kernel floor

59 low-level geometry kernels, re-exported from
[`main.py`](main.py). This is the layer `MeshData`, `FFDData`,
`DeltaMushData`, `PatchRelaxData` and the samplers are built on: no
classes, no state, no caching — arrays in, arrays out.

<!-- notest -- teaser; `mesh` is whatever MeshData you already have -->

```python
from cgmath.geometry import utils as gu

neighbors = gu.compute_neighbors(mesh.v2f, mesh.f2v)       # 1-ring adjacency
smooth    = gu.blur(mesh.points, neighbors, iterations=9)  # Maya-tuned Laplacian
```

Most callers never import this package directly — they reach for
`MeshData.smooth()`, `DeltaMushData.apply()` or `FFDData.update()`, which
wrap these kernels with binding, caching and validation. Come here when
you need a kernel without the object around it, or when you are building
a new deformer.

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy a working call for any kernel | [`CHEATSHEET.md`](CHEATSHEET.md) — every export, runnable top to bottom |
| Understand the wider library | [`../../README.md`](../../README.md) |
| Use the wrapped, stateful API | [`../../CHEATSHEET.md`](../../CHEATSHEET.md) — `MeshData`, deformers, samplers |

---

## Conventions

These hold across the whole package, and knowing them removes most of the
guesswork.

- **`-1` is the pad.** Adjacency comes back as a dense `(rows, max_width)`
  matrix, short rows padded with `-1`. Every kernel that consumes one
  stops at the first `-1`. Distance matrices pad with `-1.0`.
- **Two shapes for the same topology.** The *stream* form is
  `(indices, counts)` — a flat face-vertex list plus a per-face corner
  count. The *matrix* form is the padded matrix above.
  `matrix_to_stream` / `stream_to_matrix` convert between them.
- **Arguments are coerced, not validated.** Every wrapper runs
  `np.asarray(x, dtype=...)` before calling the kernel, so lists and
  wrong-dtype arrays work. Wrong *shapes* surface as a numba
  `TypingError`, not a friendly message.
- **Adjacency names are `a2b`.** `f2v` face→vertex, `v2f` vertex→face,
  `e2v` edge→vertex, `f2e` face→edge, and so on. `compute_neighbors`
  takes any matching `(i2x, x2i)` pair, so the same kernel builds
  vertex-vertex, face-face or edge-edge adjacency.
- **Most kernels return; a few mutate.** `balance_center_weights` and
  `composite_sprite_aa` edit their input in place and return `None`.
  `compute_decal_maps` and `patch_relax` accept optional output buffers
  so a loop can reuse one allocation.
- **Angles come back in degrees** (`compare_normals`,
  `vector_angle_difference`).
- **Numba imports are deferred.** Each wrapper imports its kernel from
  `_numba/` inside the function body, so importing this package is cheap
  and the JIT cost lands on first call, not on import.

---

## Files

```
utils/
├── __init__.py       re-exports 59 names from main.py — the public surface
├── main.py           thin, documented wrappers: coerce dtypes, dispatch
└── _numba/           the @njit kernels, one file per domain
    ├── _bilinear.py      bilinear + PN-Quad Bézier sampling, raycast, areas
    ├── _blur.py          Laplacian blur, inpaint, weight balancing
    ├── _bspline.py       B-spline basis evaluation
    ├── _bvh.py           BVHData + builder for the raycasters
    ├── _cdt.py           constrained Delaunay triangulation
    ├── _connectivity.py  Dijkstra neighbourhoods, edge distances
    ├── _delta_mush.py    delta encode/decode, polar decomposition, DDM
    ├── _ffd.py           inverse trilinear, Bernstein FFD
    ├── _main.py          row overlaps, index replace, stream/matrix, remap
    ├── _normals.py       hard-edge splitting, triangulation expansion
    ├── _pack.py          UV shell packing
    ├── _patch_relax.py   vertex rings, decal maps, span weights, relax step
    ├── _rasterize.py     sprite compositing
    ├── _sdf.py           signed distance fields
    ├── _skin_deform.py   linear blend / dual quaternion skinning
    ├── _skin_weights.py  weight transfer and normalisation
    ├── _subdivide.py     Catmull-Clark point + topology kernels
    ├── _subdivision.py   the standalone index rebuild
    ├── _tangent_space.py tangent/bitangent construction
    ├── _topology.py      neighbours, v2x, e2v/e2f, quad matching
    └── _wrap.py          RBF wrap helpers
```

`_numba/` is private. Only the names listed in `__init__.py` are the
public contract; anything reached through `utils._numba` can move.

Three functions live in `main.py` but are **not** re-exported at package
level — import them from `cgmath.geometry.utils.main`:
`bezier_evaluate`, `bezier_vectors`, `grow_neighbors`.

---

## What each group is for

| Group | Kernels | Wrapped by |
|---|---|---|
| Connectivity | `compute_v2x` `compute_e2v_e2f` `compute_neighbors` `build_vertex_rings` `shared_edges_test` | `MeshData` topology properties |
| Row dedup / remap | `matrix_row_overlaps` `point_row_overlaps` `matrix_row_combine` `indices_replace` `matrix_index_lookup` | `MeshData` edge derivation, merge/weld |
| Stream ↔ matrix | `matrix_to_stream` `stream_to_matrix` `remap_indices` `average_points` | `MeshData`, the samplers |
| Geodesics | `compute_neighbor_distances` `compute_topological_neighborhood` `minimize_neighbor_distances` | `MeshData.get_*_geodesic_neighborhood`, `WrapData` |
| Normals / tangents | `split_at_hard_edges` `build_uv_to_normal_map` `build_tri_expand_map` `compare_normals` `vector_angle_difference` | `MeshData.set_normals`, `.triangulate` |
| Quadrangulate | `quad_match_greedy` | `MeshData` quadrangulation |
| Surface sampling | `compute_centroids` `bilinear_vectors` `bilinear_integrate` `bilinear_sample` `compute_samples` `bezier_evaluate` `bezier_vectors` | `MeshData.sample`, `.get_face_areas`, `resample` |
| Raycast | `bilinear_raycast` `bezier_raycast` | `MeshData.raycast`, the raytracer |
| Splines / FFD | `compute_basis` `build_lattice_topology` `get_cell_corners` `trilinear` `trilinear_jacobian` `inverse_trilinear` `assign_cells` `bernstein_basis_1d` `bernstein_eval` | `BSplineData`, `BSplinePatchData`, `FFDData` |
| Blur / weights | `blur` `inpaint` `balance_center_weights` | `MeshData.smooth`, `SkinData` |
| Delta mush | `encode_local_deltas` `decode_local_deltas` `encode_local_deltas_tbn` `decode_local_deltas_tbn` `decode_world_deltas_procrustes` `ddm_precompute` `decode_world_deltas_ddm` `blend_deltas` `batch_procrustes_rotations` | `DeltaMushData` |
| Patch relax | `compute_decal_maps` `compute_span_weights` `patch_relax` | `PatchRelaxData` |
| Subdivision | `subdivide_catmull_clark` `rebuild_indices` | `MeshData.subdivide` |
| Raster | `composite_sprite` `composite_sprite_aa` | UV shell packing |
| USD | `pxr` | every USD path in `cgmath.geometry` |

---

## Quick taste

No mesh assets ship with this repo, so build one inline — a unit cube of
six quads is the standard fixture.

```python
import numpy as np

from cgmath.geometry import MeshData
from cgmath.geometry import utils as gu

points = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)
faces = [
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
]
mesh = MeshData(
    points  = points,
    indices = np.asarray(faces, dtype=np.int32).ravel(),
    counts  = np.full(6, 4, dtype=np.int32),
)
```

### Derive edge topology from a raw index stream

This is the pipeline `MeshData` runs to populate `e2v`, `e2f` and `f2e`.

```python
ue2v, ue2f = gu.compute_e2v_e2f(mesh.indices, mesh.counts)     # per-corner edges
unique, dup, orig = gu.matrix_row_overlaps(ue2v, exact=False)  # weld halves
merged = gu.matrix_row_combine(ue2f, dup, orig)
merged = gu.matrix_row_combine(merged, orig, dup)

e2v, e2f = ue2v[unique], merged[unique]
print(np.array_equal(e2v, mesh.e2v), np.array_equal(e2f, mesh.e2f))   # True True
```

### Encode and replay a delta mush

```python
neighbors = mesh.get_edge_vertex_neighbors()
smooth    = gu.blur(mesh.points, neighbors, iterations=9)

deltas          = gu.encode_local_deltas(mesh.points, smooth, neighbors)  # bind
smooth_deformed = smooth + [0.0, 0.0, 2.0]                                # "deform"
restored        = gu.decode_local_deltas(smooth_deformed, neighbors, deltas)
mushed          = gu.blend_deltas(smooth_deformed, restored, weight=1.0)
print(np.allclose(mushed, mesh.points + [0.0, 0.0, 2.0]))         # True
```

### Ray-cast a quad mesh

```python
origins    = np.array([[0.5, 0.5, 5.0]])
directions = np.array([[0.0, 0.0, -1.0]])

t, uv, faces = gu.bilinear_raycast(
    mesh.points, mesh.f2v, origins, directions, bvh=mesh.bvh_bilinear)
print(t, faces)   # [4.] [1] — the +Z cap, four units away
```

`mesh.bvh_bilinear` is `None` below 64 faces; the kernel then falls back
to a linear scan, so the call is safe either way.

---

## Performance notes

- **First call pays the JIT.** Deferred imports mean the numba compile
  happens on first invocation of each kernel. Cached to disk
  (`cache=True`), so it is a one-time cost per interpreter build.
- **Most kernels are `parallel=True`.** They scale with `NUMBA_NUM_THREADS`
  and are written to avoid write races (per-vertex delta buffers rather
  than scattered accumulation).
- **Build the BVH once.** `bilinear_raycast` and `bezier_raycast` are
  10-30x faster with a BVH on meshes of a few thousand faces. Reuse
  `mesh.bvh_bilinear` / `mesh.bvh_bezier` rather than rebuilding per call
  — and note `bezier_raycast` needs the *control-point* BVH
  (`bvh_bezier`), not the vertex one.
- **`grow_neighbors` switches strategy at `n > 5`**, falling back to the
  Dijkstra kernel because it beats repeated dilation past that point.
- **`bernstein_eval` degrades gracefully.** If its numba kernel fails to
  compile it silently falls back to a pure-NumPy implementation, so
  correctness never depends on the JIT.
