# `cgmath.geometry.utils` Cheatsheet

Every public kernel, with a runnable example. For the overview and
conventions see [`README.md`](README.md); for the rest of the library see
[`../../CHEATSHEET.md`](../../CHEATSHEET.md).

Every `python` block below runs in document order in one namespace — copy
the file top to bottom and it executes.

---

## Contents

| Section | Kernels |
|---|---|
| [Setup](#setup) | fixtures used by every later block |
| [Connectivity](#connectivity) | `compute_v2x` `compute_e2v_e2f` `compute_neighbors` `shared_edges_test` |
| [Rebuilding edge topology](#rebuilding-edge-topology) | `matrix_row_overlaps` `point_row_overlaps` `matrix_row_combine` `indices_replace` |
| [Neighbours, rings, geodesics](#neighbours-rings-geodesics) | `grow_neighbors` `compute_neighbor_distances` `compute_topological_neighborhood` `minimize_neighbor_distances` `build_vertex_rings` |
| [Streams, matrices, remaps](#streams-matrices-remaps) | `matrix_to_stream` `stream_to_matrix` `matrix_index_lookup` `remap_indices` `average_points` |
| [Normals and tangent space](#normals-and-tangent-space) | `split_at_hard_edges` `build_uv_to_normal_map` `build_tri_expand_map` `compare_normals` `vector_angle_difference` |
| [Quad matching](#quad-matching) | `quad_match_greedy` |
| [Bilinear and Bézier surfaces](#bilinear-and-bézier-surfaces) | `compute_centroids` `bilinear_vectors` `bilinear_integrate` `bilinear_sample` `compute_samples` `bezier_evaluate` `bezier_vectors` |
| [Raycast](#raycast) | `bilinear_raycast` `bezier_raycast` |
| [B-spline basis](#b-spline-basis) | `compute_basis` |
| [FFD, trilinear, Bernstein](#ffd-trilinear-bernstein) | `build_lattice_topology` `get_cell_corners` `trilinear` `trilinear_jacobian` `inverse_trilinear` `assign_cells` `bernstein_basis_1d` `bernstein_eval` |
| [Blur, inpaint, weights](#blur-inpaint-weights) | `blur` `inpaint` `balance_center_weights` |
| [Delta mush](#delta-mush) | `encode_local_deltas` `decode_local_deltas` `*_tbn` `decode_world_deltas_procrustes` `ddm_precompute` `decode_world_deltas_ddm` `blend_deltas` `batch_procrustes_rotations` |
| [Patch relaxation](#patch-relaxation) | `compute_decal_maps` `compute_span_weights` `patch_relax` |
| [Subdivision](#subdivision) | `subdivide_catmull_clark` `rebuild_indices` |
| [Raster ops](#raster-ops) | `composite_sprite` `composite_sprite_aa` |
| [USD accessor](#usd-accessor) | `pxr` |
| [Full index](#full-index) | all 59 exports + 3 `main`-only |

---

## Setup

There are no mesh assets in this repo — build geometry inline. Two
fixtures carry the whole document: a closed unit cube (every ring closes,
every edge is manifold) and an open quad grid (has a border).

```python
import numpy as np

from cgmath.geometry import MeshData
from cgmath.geometry import utils as gu


def make_cube():
    """Closed unit cube, 6 quads, 8 verts."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    return MeshData(
        points  = points,
        indices = np.asarray(faces, dtype=np.int32).ravel(),
        counts  = np.full(6, 4, dtype=np.int32),
    )


def make_grid(n=5):
    """Open (n-1)^2 quad grid in the XY plane."""
    t       = np.linspace(0.0, 1.0, n)
    points  = np.array([[x, y, 0.0] for y in t for x in t])
    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices.extend([a, a + 1, a + n + 1, a + n])
    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
    )


cube = make_cube()
grid = make_grid(5)

print(cube.point_count, cube.face_count, cube.edge_count)  # 8 6 12
print(grid.point_count, grid.face_count, grid.edge_count)  # 25 16 40
```

---

## Connectivity

`compute_v2x` inverts any *face → x* matrix into *x → face*. The second
argument is the output shape: `(num_x, max_width)`.

```python
v2f = gu.compute_v2x(cube.f2v, (cube.point_count, 6))
print(v2f)     # (8, 6) int32, -1 padded — vertex → incident faces
print(v2f[0])  # [ 0  2  5 -1 -1 -1]
```

`compute_e2v_e2f` builds the *raw* per-corner edge stream. Its first
argument is the **flat face-vertex index stream**, not the `f2v` matrix,
despite the parameter being named `f2v`.

```python
ue2v, ue2f = gu.compute_e2v_e2f(cube.indices, cube.counts)
print(ue2v.shape, ue2f.shape)   # (24, 2) (24, 1) — one edge per face corner
print(ue2v[:4])
```

`compute_neighbors(i2x, x2i, n=0)` is the general 1-ring builder: hand it any
matching *i → x* / *x → i* pair and it returns the i-to-i adjacency.

```python
vert_nbrs = gu.compute_neighbors(cube.v2f, cube.f2v)  # vertex → vertex
face_nbrs = gu.compute_neighbors(cube.f2e, cube.e2f)  # face   → face
print(vert_nbrs.shape, face_nbrs.shape)  # (8, 6) (6, 4)
print(face_nbrs[0])                      # [5 4 3 2]
```

`n=` grows the result in place of a second call.

```python
print(gu.compute_neighbors(cube.v2f, cube.f2v, n=2).shape)   # (8, 7)
```

`shared_edges_test` flags rows whose (sorted) values contain **no**
repeat. `mesh.py` feeds it each vertex's incident face-edge sets to find
vertices that join two fans without sharing an edge — the non-manifold
"shared vertex" test.

```python
verts = np.where(np.sum(cube.v2f >= 0, axis=1) > 1)[0]
edges = cube.f2e[cube.v2f[verts]].reshape(len(verts), -1)
print(gu.shared_edges_test(edges))   # tuple from np.where — empty on a clean cube
```

---

## Rebuilding edge topology

The four kernels below are exactly the pipeline `MeshData` runs to derive
`e2v` / `e2f` / `f2e` from a raw index stream.

`matrix_row_overlaps` returns `(unique, duplicates, originals)`.
`exact=False` sorts each row first, so `[2, 1]` matches `[1, 2]` — that is
what makes two half-edges collapse into one edge.

```python
unique, dup, orig = gu.matrix_row_overlaps(ue2v, exact=False)
print(unique.shape, dup.shape, orig.shape)   # (12,) (12,) (12,)
```

`matrix_row_combine` appends the entries of `from_rows` onto `to_rows`,
widening the matrix as needed.

```python
merged = gu.matrix_row_combine(ue2f, dup, orig)
merged = gu.matrix_row_combine(merged, orig, dup)
print(np.array_equal(ue2v[unique], cube.e2v))    # True
print(np.array_equal(merged[unique], cube.e2f))  # True
```

`indices_replace` swaps values through a `from → to` map. `collapse=True`
additionally compacts the surviving values to a dense 0..n-1 range.

```python
m = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
print(gu.indices_replace(m, np.array([0, 2]), np.array([10, 20])))
# [[10  1 20]
#  [ 3  4  5]]
print(gu.indices_replace(m, np.array([0, 2]), np.array([10, 20]), collapse=True))
# [[4 0 5]
#  [1 2 3]]
```

`point_row_overlaps` is the float sibling — merges rows within a distance
tolerance instead of exact equality.

```python
pts = np.array([[0.0, 0, 0], [1, 1, 1], [0, 0, 1e-9]])
uniq, dupe, origin = gu.point_row_overlaps(pts, tolerance=1e-6)
print(uniq, dupe, origin)   # [0 1] [2] [0]
```

---

## Neighbours, rings, geodesics

`grow_neighbors` widens an existing neighbour matrix by `n` hops. It is
the only one of these three not re-exported at package level.

```python
from cgmath.geometry.utils.main import grow_neighbors

print(grow_neighbors(vert_nbrs, n=1).shape)                       # (8, 7)
print(grow_neighbors(vert_nbrs, indices=np.array([0, 1]), n=1,
                     full_view=False).shape)                      # (2, 7)
```

`compute_neighbor_distances` turns a neighbour matrix into an edge-length
matrix, `-1.0` in padded slots.

```python
dists = gu.compute_neighbor_distances(vert_nbrs, cube.points)
print(dists.shape, np.round(dists[0], 4))
# (8, 6) [1.     1.4142 1.     1.4142 1.     1.4142]
```

`compute_topological_neighborhood` is Dijkstra over that matrix. Without
`distances` it returns just the indices.

```python
hood = gu.compute_topological_neighborhood(vert_nbrs, num_hops=1)
print(hood.shape)   # (8, 6)
```

With `distances` it returns `(neighborhood, distances)` in one of three
formats.

| `distance_format` | second return value |
|---|---|
| `"sparse"` (default) | `(len(indices), W)` padded with `-1.0` |
| `"dense"` | `(len(indices), num_vertices)`, `0` on the diagonal, `inf` unreachable |
| `"shortest"` | reduces to `(reached_indices, shortest_distance)` 1-D pair |

```python
d32 = dists.astype(np.float32)

hood, sparse = gu.compute_topological_neighborhood(vert_nbrs, 2, distances=d32)
print(sparse.shape)                                        # (8, 7)

hood, dense = gu.compute_topological_neighborhood(
    vert_nbrs, 2, distances=d32, distance_format="dense")
print(dense.shape, dense[0, 0])                            # (8, 8) 0.0

reached, shortest = gu.compute_topological_neighborhood(
    vert_nbrs, 2, indices=np.array([0]), distances=d32,
    distance_format="shortest")
print(reached, np.round(shortest, 3))
```

`max_distance` prunes anything beyond a cumulative path length.

```python
hood, near = gu.compute_topological_neighborhood(
    vert_nbrs, 3, distances=d32, max_distance=1.05)
print(near.shape)
```

`minimize_neighbor_distances` collapses a `(rows, W)` neighbourhood +
distance pair into the shortest distance per vertex index. Unreached
vertices come back as `inf`.

```python
hood2 = gu.compute_topological_neighborhood(vert_nbrs, 2)
d2    = gu.compute_neighbor_distances(hood2, cube.points).astype(np.float32)
print(gu.minimize_neighbor_distances(hood2, d2, cube.point_count))
# [1. 1. 1. 1. 1. 1. 1. 1.]
```

`build_vertex_rings` is the *winding-ordered* 1-ring — neighbours come
back in fan order, which the decal-map parameterization needs.
`is_boundary` is True where the fan does not close.

```python
ring, valence, is_boundary = gu.build_vertex_rings(
    grid.indices, grid.counts, grid.point_count)
print(ring.shape, valence[:5], is_boundary.sum())   # (25, 5) [2 3 3 3 2] 16
```

---

## Streams, matrices, remaps

`matrix_to_stream` / `stream_to_matrix` round-trip a `-1`-padded matrix
against the flat `(indices, counts)` form.

```python
mat = np.array([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]], dtype=np.int32)
idx, cnt = gu.matrix_to_stream(mat)
print(idx, cnt)                                            # [0..8] [3 2 4]
print(np.array_equal(gu.stream_to_matrix(idx, cnt), mat))  # True
```

`matrix_index_lookup` finds the `(row, col)` of the **first** occurrence
of each query value.

```python
rows, cols = gu.matrix_index_lookup(
    np.array([[5, 7], [9, 5]], dtype=np.int32), np.array([5, 9, 7]))
print(rows, cols)   # [0 1 0] [0 0 1]
```

`remap_indices` picks, per unique vertex index, the UV index whose entry
in `distances` is smallest — the "closest unique correspondence".

```python
print(gu.remap_indices(
    np.array([0, 1, 2, 3], dtype=np.int32),   # vertex stream
    np.array([0, 1, 1, 2], dtype=np.int32),   # parallel uv stream
    np.array([0.5, 0.1, 0.2, 0.3])))          # per-uv distance
# [0 1 1 2]
```

`average_points` averages the rows of a `-1`-padded gather matrix — the
face-centroid primitive.

```python
print(gu.average_points(cube.points, cube.f2v))   # (6, 3) face centroids
```

---

## Normals and tangent space

`split_at_hard_edges` runs union-find over smooth edges and hands back
`(normal_indices, num_normals)` — a face-varying slot per face-vertex.

```python
soft, n_soft = gu.split_at_hard_edges(
    cube.indices, cube.counts, cube.e2v, cube.e2f, None)
hard, n_hard = gu.split_at_hard_edges(
    cube.indices, cube.counts, cube.e2v, cube.e2f,
    np.arange(cube.e2v.shape[0]))
print(n_soft, n_hard)   # 8 24 — one slot per vertex vs one per face-corner
```

`build_uv_to_normal_map` aligns a UV stream to a normal-slot stream,
first occurrence wins; unreferenced UV verts get `0`.

```python
print(gu.build_uv_to_normal_map(
    np.array([0, 1, 0, 2]), np.array([10, 20, 30, 40]), 3))   # [10 20 40]
```

`build_tri_expand_map` maps triangulated face-vertex positions back to
the original stream. `counts` must be all 3s and 4s; `rules` picks the
quad diagonal (`0` = 0-2, otherwise 1-3).

```python
print(gu.build_tri_expand_map(
    np.array([3, 4], dtype=np.int32), np.array([0, 0], dtype=np.int32)))
# [0 1 2 4 5 3 3 5 6]
```

`compare_normals` and `vector_angle_difference` both return the per-row
angle in **degrees**.

```python
a = np.array([[1.0, 0, 0], [0, 1.0, 0]])
b = np.array([[1.0, 0, 0], [0, 0, 1.0]])
print(gu.compare_normals(a, b))          # [ 0. 90.]
print(gu.vector_angle_difference(a, b))  # [ 0. 90.]
```

---

## Quad matching

`quad_match_greedy` is maximum-weight matching on the triangle dual
graph: pick candidate edges by descending score, skipping any whose faces
were already claimed. Use `-inf` to disqualify a candidate.

```python
score = np.array([0.9, 0.8, 0.7, 0.5])
fa    = np.array([0, 1, 2, 0])
fb    = np.array([1, 2, 3, 3])
print(gu.quad_match_greedy(score, fa, fb, n_faces=4))
# [ True False  True False]
```

---

## Bilinear and Bézier surfaces

`compute_centroids` doubles as the broad-phase bound builder for the
sampler: `return_radiuses=True` also gives each face's bounding radius.

```python
centroids, radiuses = gu.compute_centroids(
    cube.points, cube.f2v, return_radiuses=True)
print(centroids.shape, radiuses)   # (6, 3) [0.5 ... 0.5]
```

`bilinear_vectors` returns `(dS/du, dS/dv)` at per-query `(face, uv)`
pairs. `geometry` is **per query**, one row of 4 vertex indices each.

```python
U, V = gu.bilinear_vectors(
    cube.points, cube.f2v[:2], np.array([[0.5, 0.5], [0.25, 0.75]]))
print(U[0], V[0])   # [0. 1. 0.] [1. 0. 0.]
```

`bilinear_integrate` gives per-face surface area. Three methods:
`"gauss"` (closed form, default), `"quadrature"` (fixed `samples`),
`"adaptive"` (refines to `tolerance`).

```python
print(gu.bilinear_integrate(cube.points, cube.f2v))                # [1. ... 1.]
print(gu.bilinear_integrate(cube.points, cube.f2v,
                            samples=8, method="quadrature"))
print(gu.bilinear_integrate(cube.points, cube.f2v,
                            samples=4, method="adaptive", tolerance=1e-6))
```

`bilinear_sample` is the closest-point-on-surface query. It returns five
arrays: `projected, distance, weights, uv, face`. Passing `normals=`
switches the whole thing to PN-Quad bicubic Bézier patches.

```python
queries = np.array([[0.5, 0.5, 2.0], [0.25, 0.25, -1.0]])
proj, dist, weights, uv, face = gu.bilinear_sample(
    queries, cube.points, cube.f2v, centroids, radiuses,
    centroid_distance_tolerance=100.0)
print(proj)  # [[0.5 0.5 1.] [0.25 0.25 0.]] — snapped onto the cube
print(dist)  # [1. 1.]
print(face)  # [1 0] — the winning face per query
print(uv)    # [[0.5 0.5] [0.25 0.25]]
```

`compute_samples` applies those barycentric-style weights to any
per-vertex attribute — positions, UVs, colours, skin weights.

```python
print(gu.compute_samples(cube.points, weights, cube.f2v[face]))
# same as proj, because the sampled attribute is the position
```

`bezier_evaluate` / `bezier_vectors` evaluate the PN-Quad patch directly.
Neither is re-exported at package level — import from `utils.main`.

```python
from cgmath.geometry.utils.main import bezier_evaluate, bezier_vectors

normals = np.tile([0.0, 0.0, 1.0], (cube.point_count, 1))
uv      = np.array([[0.5, 0.5], [0.5, 0.5]])

print(bezier_evaluate(cube.points, normals, cube.f2v[:2], uv))
# [[0.5 0.5 0.] [0.5 0.5 1.]] — the centre of faces 0 and 1

Ub, Vb = bezier_vectors(cube.points, normals, cube.f2v[:2], uv)
print(np.round(Ub, 6), np.round(Vb, 6))
```

---

## Raycast

Both raycasters are forward-only and return `(t, uv, face_index)` for the
closest hit. `twosided=False` rejects back faces. Faces must be tris or
quads; `< 4` wide geometry is `-1`-padded automatically.

```python
origins    = np.array([[0.5, 0.5, 5.0], [0.25, 0.25, 5.0]])
directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])

t, uv, faces = gu.bilinear_raycast(cube.points, cube.f2v, origins, directions)
print(t, faces)   # [4. 4.] [1 1]
```

Pass a prebuilt BVH for the 10-30x path. `MeshData` only builds one above
64 faces, so the cube's is `None` (and the call transparently falls back
to the linear scan).

```python
print(cube.bvh_bilinear)   # None — only 6 faces

big      = make_grid(10)        # 81 faces
gnormals = np.tile([0.0, 0.0, 1.0], (big.point_count, 1))
t, uv, faces = gu.bilinear_raycast(
    big.points, big.f2v, origins, directions, bvh=big.bvh_bilinear)
print(t, faces)
```

`bezier_raycast` takes vertex normals and intersects the bicubic patch.
Its BVH must be built over the **control-point** AABBs — that is what
`MeshData.bvh_bezier` is.

```python
t, uv, faces = gu.bezier_raycast(
    big.points, big.f2v, gnormals, origins, directions, bvh=big.bvh_bezier)
print(t, faces)

t, uv, faces = gu.bezier_raycast(
    cube.points, cube.f2v, normals, origins[:1], directions[:1],
    twosided=False)
print(t, faces)
```

---

## B-spline basis

`compute_basis(u, kv, c, d)` returns the
`(len(u), num_control_points)` basis matrix. `BSplineData` and
`BSplinePatchData` call it.

```python
basis = gu.compute_basis(
    np.array([0.0, 0.5, 1.0]),          # parameters
    np.array([0, 0, 0, 1, 1, 1]),       # knot vector
    3,                                  # control point count
    2,                                  # degree
)
print(basis)
# [[1.   0.   0.  ]
#  [0.25 0.5  0.25]
#  [0.   0.   1.  ]]
```

---

## FFD, trilinear, Bernstein

`build_lattice_topology` turns lattice dimensions into `(indices, counts)`
you can hand straight to `MeshData` — every hex-cell face, internal
included.

```python
lat_indices, lat_counts = gu.build_lattice_topology(3, 3, 3)
print(lat_indices.shape, lat_counts.shape)   # (144,) (36,)
```

`get_cell_corners` gathers the 8 corners of each `(i, j, k)` cell. Corner
order is `(0,0,0) (1,0,0) (0,1,0) (1,1,0) (0,0,1) (1,0,1) (0,1,1) (1,1,1)`.

```python
axis = np.linspace(0, 1, 3)
lattice = np.stack(
    np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).astype(float)
print(lattice.shape)   # (3, 3, 3, 3)

cells   = np.array([[0, 0, 0], [1, 1, 1]])
corners = gu.get_cell_corners(lattice, cells)
print(corners.shape)   # (2, 8, 3)
```

`trilinear` interpolates, `trilinear_jacobian` gives `dP/d(uvw)`, and
`inverse_trilinear` Newton-solves the other way.

```python
uvw = np.full((2, 3), 0.5)
print(gu.trilinear(uvw, corners))              # [[0.25 ...] [0.75 ...]]
print(gu.trilinear_jacobian(uvw, corners)[0])  # diag(0.5)

targets = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
print(gu.inverse_trilinear(targets, corners, max_iter=20, tol=1e-12))
# [[0.5 0.5 0.5] [0.5 0.5 0.5]]
```

`assign_cells` does the whole binding step: bbox guess → Newton refine →
cell-hop for anything that escaped `[0, 1]`.

```python
bound = cube.points * 0.9 + 0.05
cells, uvw = gu.assign_cells(bound, lattice, max_hops=4)
print(cells[0], np.round(uvw[0], 3))   # [0 0 0] [0.1 0.1 0.1]
```

`bernstein_basis_1d` is the raw polynomial basis; `bernstein_eval` is the
Sederberg & Parry FFD evaluation over a displacement field.
`local_influence` is the window width in cells per axis, `divisions` the
cell count per axis (lattice points − 1).

```python
print(gu.bernstein_basis_1d(np.array([0.0, 0.5, 1.0]), 3))
# [[1.    0.    0.    0.   ]
#  [0.125 0.375 0.375 0.125]
#  [0.    0.    0.    1.   ]]

delta         = np.zeros_like(lattice)
delta[..., 2] = 0.5                                  # push the whole field +Z
offsets = gu.bernstein_eval(cells, uvw, delta,
                            local_influence=(2, 2, 2),
                            divisions=np.array([2, 2, 2]))
print(np.round(offsets[0], 3))   # [0. 0. 0.5]
```

---

## Blur, inpaint, weights

`blur` is the iterative Laplacian, defaults tuned to match Maya's
`deltaMush`. Works on `(N,)` or `(N, M)` values.

```python
gnbrs = grid.get_edge_vertex_neighbors()

heat     = np.zeros((grid.point_count, 2))
heat[0]  = [1.0, 0.0]
heat[-1] = [0.0, 1.0]
print(np.round(gu.blur(heat, gnbrs, iterations=3)[:3], 4))

scalar = np.arange(grid.point_count, dtype=float)
print(np.round(gu.blur(scalar, gnbrs, iterations=2)[:4], 3))
```

`iterations`, `receptions` and `contributions` all accept a per-vertex
array as well as a scalar — that is how paint maps are expressed.

```python
per_vertex_iters      = np.zeros(grid.point_count, dtype=np.int32)
per_vertex_iters[:12] = 5                     # only blur the first half
print(np.round(gu.blur(scalar, gnbrs, iterations=per_vertex_iters)[:4], 3))
print(np.round(gu.blur(scalar, gnbrs, iterations=2,
                       receptions=np.full(grid.point_count, 0.25))[:4], 3))
```

`inpaint` is the hole-filling sibling. `indices` names the vertices to
**fill**; everything else is treated as known data.

```python
skin       = np.zeros((grid.point_count, 2))
skin[:, 0] = 1.0
hole       = np.array([12])
skin[hole] = 0.0
filled     = gu.inpaint(hole, skin, gnbrs, iterations=4)
print(filled[hole])   # [[1. 0.]] — pulled back in from the neighbours
```

`balance_center_weights` prunes a skin-weight matrix down to `max_inf`
influences, dropping centre columns first and positive/negative pairs
together so symmetry survives. It edits `weights` **in place** and
returns `None`.

```python
weights = np.array([[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]])
gu.balance_center_weights(
    weights,
    positive = np.array([0]),  # column 0 is the "left" influence
    negative = np.array([1]),  # column 1 is its "right" mirror
    center   = np.array([2]),  # column 2 sits on the midline
    max_inf  = 2,
)
print(weights)
# [[0.4 0.4 0. ]    row 0: the centre was weakest -> dropped alone
#  [0.  0.  0.5]]   row 1: a paired influence was weakest -> both dropped
```

---

## Delta mush

Encode at bind time, decode against the deformed smooth mesh. Four frame
choices, all sharing the same `(N, K)` `-1`-padded neighbour matrix.

```python
rest   = grid.points.copy()
smooth = gu.blur(rest, gnbrs, iterations=5)
```

**Averaged-Laplacian frame** — `encode_local_deltas` / `decode_local_deltas`.

```python
local = gu.encode_local_deltas(rest, smooth, gnbrs)
back  = gu.decode_local_deltas(smooth, gnbrs, local)
print(local.shape, np.allclose(back, rest))   # (25, 3) True
```

**First-face TBN frame** — matches Maya's `deltaMush` far more closely.
`first_nbrs` is `(N, 2)` of `[next, prev]` around the vertex's first face;
`-1` stores the world delta verbatim.

```python
first = np.ascontiguousarray(gnbrs[:, :2]).astype(np.int32)
tbn   = gu.encode_local_deltas_tbn(rest, smooth, first)
print(np.allclose(gu.decode_local_deltas_tbn(smooth, first, tbn), rest))  # True
```

**Procrustes** — cache the world delta and rotate it by the per-vertex
SVD fit between smoothed rest and smoothed deformed offsets.

```python
world_deltas = rest - smooth
out          = gu.decode_world_deltas_procrustes(smooth, smooth, gnbrs, world_deltas)
print(np.allclose(out, rest))   # True (rest == deformed here)
```

**Direct Delta Mush** (Le & Lewis 2019) — same idea, stabilised with
precomputed offsets and length weights.

```python
rest_offsets, rest_weights = gu.ddm_precompute(smooth, gnbrs)
print(rest_offsets.shape, rest_weights.shape)   # (25, 4, 3) (25, 4)

out = gu.decode_world_deltas_ddm(
    smooth, gnbrs, rest_offsets, rest_weights, world_deltas)
print(np.allclose(out, rest))   # True
```

`blend_deltas` is the mush-weight dial: `0` returns the smoothed mesh,
`1` the fully reconstructed one.

```python
print(np.allclose(gu.blend_deltas(smooth, rest, 0.0), smooth))  # True
print(np.allclose(gu.blend_deltas(smooth, rest, 1.0), rest))    # True
```

`batch_procrustes_rotations` is the standalone polar decomposition: give
it a batch of covariance matrices `H = v1.T @ v0`, get rotations back.

```python
H = np.tile(np.eye(3), (2, 1, 1))
print(gu.batch_procrustes_rotations(H, max_iter=20)[0])   # identity
```

---

## Patch relaxation

de Goes et al. 2018, in three steps: flatten each stencil to polar decal
coordinates (§2), weight the edges by span (§3), then Jacobi-relax (§4-5).

```python
ring, valence, is_boundary = gu.build_vertex_rings(
    grid.indices, grid.counts, grid.point_count)

decals = gu.compute_decal_maps(grid.points, ring, valence, is_boundary)
print(decals.shape)   # (25, 5, 2)

span_weights = gu.compute_span_weights(decals, valence)
print(np.round(span_weights[0], 3))   # [0.5 0.5 0. 0. 0.]
```

`compute_decal_maps` accepts an `out=` buffer so a relaxation loop can
reuse one allocation.

```python
buffer = np.zeros_like(decals)
gu.compute_decal_maps(grid.points, ring, valence, is_boundary, out=buffer)
print(np.allclose(buffer, decals))   # True
```

`patch_relax` needs `rest_vectors` — the weighted baseline offsets
`sum_k w_k (rest_k - rest_i)`.

```python
safe_ring    = np.where(ring < 0, 0, ring)
offsets      = grid.points[safe_ring] - grid.points[:, None, :]
rest_vectors = np.einsum("nk,nkd->nd", span_weights, offsets)

deformed = grid.points.copy()
deformed[12] += [0.1, 0.1, 0.0]         # poke one interior vertex

relaxed = gu.patch_relax(
    deformed, grid.points, ring, valence, is_boundary,
    span_weights, rest_vectors, decals,
    iterations=10, alpha=1.0, step_size=0.5,
)
print(relaxed.shape)   # (25, 3)
```

`step_scale` is the pin mask — `0` freezes a vertex, which is how borders
and paint weights are expressed. `surface_blend > 0` lifts the update
back onto the surface, preserving volume.

```python
pins              = np.ones(grid.point_count)
pins[is_boundary] = 0.0                 # freeze the border

relaxed = gu.patch_relax(
    deformed, grid.points, ring, valence, is_boundary,
    span_weights, rest_vectors, decals,
    iterations=5, step_scale=pins, surface_blend=0.5, polar_iters=12,
)
print(np.allclose(relaxed[is_boundary], deformed[is_boundary]))   # True
```

---

## Subdivision

`subdivide_catmull_clark` runs one step and returns
`(points, indices, counts)` for the new mesh. It wants the full
connectivity set.

```python
new_points, new_indices, new_counts = gu.subdivide_catmull_clark(
    cube.points, cube.indices, cube.counts,
    cube.f2v, cube.f2e, cube.e2v, cube.e2f, cube.v2f, cube.v2e,
)
print(new_points.shape, new_counts.shape)   # (26, 3) (24,)

subdivided = MeshData(
    points=new_points, indices=new_indices, counts=new_counts)
print(subdivided.point_count, subdivided.face_count)   # 26 24
```

`keep_borders` pins open-border vertices; `keep_edges` keeps every
original vertex and uses plain edge midpoints (linear subdivision).

```python
linear = gu.subdivide_catmull_clark(
    grid.points, grid.indices, grid.counts,
    grid.f2v, grid.f2e, grid.e2v, grid.e2f, grid.v2f, grid.v2e,
    keep_edges=True,
)[0]
print(np.allclose(linear[:grid.point_count], grid.points))   # True
```

`rebuild_indices` is the standalone index-rebuild half — one quad per
original face corner, wired `(face_point, edge0, vertex, edge1)`.

```python
n_corners = int(cube.counts.sum())
template  = np.zeros(n_corners * 4, dtype=np.int32)
cumsum    = (np.arange(n_corners) * 4).astype(np.int32)

rebuilt = gu.rebuild_indices(
    template, cube.counts, cumsum, cube.f2v, cube.f2e, cube.e2v,
    face_offset = cube.point_count + cube.edge_count,
    edge_offset = cube.point_count,
)
print(np.array_equal(rebuilt, new_indices))   # True
```

---

## Raster ops

`composite_sprite` blends a grayscale `(H, W)` sprite into an RGB
`(H, W, 3)` buffer at `offset=(x, y)`, tinted by `color`. It returns
`(count, overlap, ratio)` — how many pixels were written, how many landed
on the mask, and the ratio. Used by the UV-packer to score placements.

```python
sprite = np.full((4, 4), 255, dtype=np.uint8)
buffer = np.zeros((8, 8, 3), dtype=np.uint8)
print(gu.composite_sprite(sprite, buffer, offset=(2, 2), color=(255, 0, 0)))
# (16, 0, 0.0)
print(buffer[3, 3])   # [255 0 0]

mask       = np.zeros((8, 8, 3), dtype=np.uint8)
mask[3, 3] = 255
count, overlap, ratio = gu.composite_sprite(
    sprite, np.zeros((8, 8, 3), dtype=np.uint8), mask=mask, offset=(2, 2))
print(count, overlap, round(ratio, 4))   # 16 1 0.0625
```

`composite_sprite_aa` blends one grayscale sprite into another toward
white. It edits the buffer **in place** and returns `None`.

```python
aa_sprite = np.array([[0, 128], [255, 64]], dtype=np.uint8)
aa_buffer = np.zeros((2, 2), dtype=np.uint8)
gu.composite_sprite_aa(aa_sprite, aa_buffer)
print(aa_buffer)   # [[  0 128] [255  64]]
```

---

## USD accessor

`pxr()` returns the USD module, or raises a friendly `ImportError` when
USD is not installed. Every USD path in `cgmath.geometry` funnels
through it.

```python
try:
    usd = gu.pxr()
    print("USD available:", usd.__name__)
except ImportError as exc:
    print("no USD ->", exc)   # "USD not found."
```

---

## Full index

Every name `cgmath.geometry.utils` re-exports, with the module that
normally calls it.

### Connectivity and topology

| Kernel | Does | Called by |
|---|---|---|
| `compute_v2x(f2v, shape)` | invert face→x into x→face, `-1` padded | `geometry/mesh.py` |
| `compute_e2v_e2f(f2v, counts)` | raw per-corner `(e2v, e2f)` from the flat stream | `geometry/mesh.py` |
| `compute_neighbors(i2x, x2i, n=0)` | i→i adjacency from any adjacency pair | `geometry/mesh.py` |
| `grow_neighbors(neighbors, indices=None, n=1, full_view=True)` | widen a neighbour matrix by n hops (`main` only) | `compute_neighbors` |
| `build_vertex_rings(indices, counts, point_count)` | winding-ordered ring + valence + boundary flags | `geometry/deform/patch_relax.py` |
| `shared_edges_test(edges)` | rows with no repeated value → non-manifold vertex test | `geometry/mesh.py` |
| `matrix_row_overlaps(matrix, exact=True, sort_indices=False)` | `(unique, duplicates, originals)` for identical int rows | `geometry/mesh.py` |
| `point_row_overlaps(matrix, tolerance=1e-6)` | same, within a float tolerance | `geometry/mesh.py` |
| `matrix_row_combine(matrix, from_rows, to_rows)` | append `from_rows` entries onto `to_rows` | `geometry/mesh.py` |
| `indices_replace(matrix, from_indices, to_indices, collapse=False)` | value remap, optionally compacted | `geometry/mesh.py` |

### Neighbourhoods and distances

| Kernel | Does | Called by |
|---|---|---|
| `compute_neighbor_distances(neighbors, points)` | per-edge Euclidean lengths, `-1.0` padded | `geometry/mesh.py`, `deform/wrap.py` |
| `compute_topological_neighborhood(connectivity, num_hops=None, indices=None, distances=None, max_distance=None, distance_format="sparse")` | Dijkstra; `sparse` / `dense` / `shortest` output | `geometry/mesh.py`, `deform/wrap.py` |
| `minimize_neighbor_distances(neighborhood, neighborhood_dists, num_vertices)` | shortest distance per vertex index, `inf` if unreached | `compute_topological_neighborhood` |

### Streams, matrices, remaps

| Kernel | Does | Called by |
|---|---|---|
| `matrix_to_stream(matrix)` | `-1`-padded matrix → `(indices, counts)` | `geometry/mesh.py` |
| `stream_to_matrix(values, counts)` | `(indices, counts)` → `-1`-padded matrix | `geometry/mesh.py` |
| `matrix_index_lookup(matrix, indices)` | `(rows, cols)` of the first occurrence of each value | `geometry/_saddle_surface.py` |
| `remap_indices(vert_indices, uv_indices, distances)` | closest unique vertex↔uv correspondence | `geometry/_saddle_surface.py` |
| `average_points(points, geometry)` | row-wise mean of a `-1`-padded gather matrix | `geometry/mesh.py` |

### Normals, tangent space, triangulation

| Kernel | Does | Called by |
|---|---|---|
| `split_at_hard_edges(indices, counts, e2v, e2f, hard_edges)` | face-varying normal slots via union-find | `geometry/mesh.py` |
| `build_uv_to_normal_map(uv_indices, normal_indices, num_uv_verts)` | UV vertex → normal slot, first write wins | `geometry/mesh.py` |
| `build_tri_expand_map(counts, rules)` | triangulated face-vertex → original position | `geometry/mesh.py` |
| `compare_normals(normal_a, normal_b)` | per-row angle in degrees | `geometry/mesh.py` |
| `vector_angle_difference(vector_a, vector_b)` | per-row angle in degrees | `robust_skinweights_transfer_bilinear.py` |
| `quad_match_greedy(score, fa, fb, n_faces)` | greedy max-weight matching on the triangle dual | `geometry/mesh.py` |

### Bilinear / Bézier surface

| Kernel | Does | Called by |
|---|---|---|
| `compute_centroids(points, geometry, return_radiuses=False)` | per-face centroid (+ bounding radius) | `geometry/mesh.py`, `_saddle_surface.py` |
| `bilinear_vectors(points, geometry, uv)` | `(dS/du, dS/dv)` on a bilinear patch | `geometry/mesh.py`, `resample.py`, `_saddle_surface.py` |
| `bilinear_integrate(points, geometry, samples=10, method="gauss", tolerance=1e-4)` | per-face area; `gauss` / `quadrature` / `adaptive` | `geometry/_saddle_surface.py` |
| `bilinear_sample(p, points, geometry, centroids, radiuses, centroid_distance_tolerance, iteration_count=100, iteration_tolerance=1e-8, uv_border_tolerance=1e-5, normals=None)` | closest point → `(proj, dist, weights, uv, face)` | `geometry/_saddle_surface.py` |
| `compute_samples(values, weights, geometry)` | apply sampler weights to any per-vertex attribute | `geometry/_saddle_surface.py` |
| `bezier_evaluate(points, normals, geometry, uv)` | PN-Quad bicubic position (`main` only) | `geometry/resample.py` |
| `bezier_vectors(points, normals, geometry, uv)` | PN-Quad bicubic tangents (`main` only) | `geometry/resample.py` |

### Raycast

| Kernel | Does | Called by |
|---|---|---|
| `bilinear_raycast(points, geometry, origins, directions, twosided=True, bvh=None)` | forward hit `(t, uv, face)` on bilinear patches | `geometry/_saddle_surface.py` |
| `bezier_raycast(points, geometry, normals, origins, directions, twosided=True, bvh=None)` | same on PN-Quad Bézier patches | `geometry/_saddle_surface.py` |

### Splines, FFD, lattices

| Kernel | Does | Called by |
|---|---|---|
| `compute_basis(u, kv, c, d)` | B-spline basis matrix | `geometry/bspline.py`, `bspline_patch.py` |
| `build_lattice_topology(lx, ly, lz)` | `(indices, counts)` for every hex-cell face | `geometry/deform/ffd.py` |
| `get_cell_corners(lattice, cells)` | gather the 8 corners of each `(i, j, k)` cell | `geometry/deform/ffd.py` |
| `trilinear(uvw, corners)` | forward trilinear interpolation | `assign_cells` |
| `trilinear_jacobian(uvw, corners)` | `(N, 3, 3)` `dP/d(uvw)` | FFD gradients |
| `inverse_trilinear(points, corners, uvw_init=None, max_iter=20, tol=1e-10)` | Newton solve for `uvw` | `assign_cells` |
| `assign_cells(points, lattice, max_hops=4)` | bind points to cells → `(cells, uvw)` | `geometry/deform/ffd.py` |
| `bernstein_basis_1d(t, degree)` | `(N, degree+1)` Bernstein weights | `bernstein_eval` |
| `bernstein_eval(cells, uvw, delta, local_influence, divisions)` | Sederberg & Parry FFD displacement | `geometry/deform/ffd.py` |

### Blur, inpaint, skin weights

| Kernel | Does | Called by |
|---|---|---|
| `blur(weights, neighbors, iterations=9, receptions=0.5, contributions=1.0)` | iterative Laplacian smoothing, Maya-matched defaults | `geometry/mesh.py`, `skin_weights.py`, `deform/delta_mush.py` |
| `inpaint(indices, weights, neighbors, iterations=9, receptions=0.5, contributions=1.0)` | fill the vertices named by `indices` from their neighbours | `geometry/skin_weights.py` |
| `balance_center_weights(weights, positive, negative, center, max_inf)` | in-place influence pruning that keeps L/R symmetry | `geometry/skin_weights.py` |

### Delta mush

| Kernel | Does | Called by |
|---|---|---|
| `encode_local_deltas(rest_points, smooth_points, neighbors)` | project `rest - smooth` into the averaged-Laplacian frame | `deform/delta_mush.py` |
| `decode_local_deltas(smooth_points, neighbors, deltas)` | inverse of the above | `deform/delta_mush.py` |
| `encode_local_deltas_tbn(rest_points, smooth_points, first_nbrs)` | same, in the first-face TBN frame (Maya-matched) | `deform/delta_mush.py` |
| `decode_local_deltas_tbn(smooth_points, first_nbrs, deltas)` | inverse of the above | `deform/delta_mush.py` |
| `decode_world_deltas_procrustes(smooth_rest, smooth_def, neighbors, world_deltas)` | rotate cached world deltas by a per-vertex SVD fit | `deform/delta_mush.py` |
| `ddm_precompute(smooth_points, neighbors)` | `(rest_offsets, rest_weights)` for the DDM apply step | `deform/delta_mush.py` |
| `decode_world_deltas_ddm(smooth_def, neighbors, rest_offsets, rest_weights, world_deltas)` | Direct Delta Mush rotation (Le & Lewis 2019) | `deform/delta_mush.py` |
| `blend_deltas(base_points, deltamush_points, weight)` | `base + weight * (deltamush - base)` | `deform/delta_mush.py` |
| `batch_procrustes_rotations(H, max_iter=20)` | batched polar decomposition of `(B, 3, 3)` covariances | Procrustes / DDM |

### Patch relaxation

| Kernel | Does | Called by |
|---|---|---|
| `compute_decal_maps(points, ring, valence, is_boundary, out=None)` | flatten each stencil into geodesic polar coords | `deform/patch_relax.py` |
| `compute_span_weights(decals, valence)` | span-aware edge weights from a decal map | `deform/patch_relax.py` |
| `patch_relax(points, rest_points, ring, valence, is_boundary, weights, rest_vectors, rest_decals, iterations=30, alpha=1.0, step_size=0.5, surface_blend=0.0, step_scale=None, polar_iters=12)` | Jacobi relaxation onto the baseline patch layout | `deform/patch_relax.py` |

### Subdivision, raster, misc

| Kernel | Does | Called by |
|---|---|---|
| `subdivide_catmull_clark(points, indices, counts, f2v, f2e, e2v, e2f, v2f, v2e, keep_borders=False, keep_edges=False, border_verts=None)` | one Catmull-Clark step → `(points, indices, counts)` | `geometry/mesh.py` |
| `rebuild_indices(new_indices, counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset)` | the index-rebuild half of a subdivision step | `geometry/mesh.py` |
| `composite_sprite(sprite, buffer, mask=None, offset=(0, 0), color=(255, 255, 255))` | blit a gray sprite into an RGB buffer → `(count, overlap, ratio)` | `geometry/mesh.py` |
| `composite_sprite_aa(sprite, buffer)` | in-place antialiased gray-on-gray blend | `geometry/mesh.py` |
| `pxr()` | the USD module, or a friendly `ImportError` | `mesh.py`, `map.py`, `morph_target.py`, `skin_weights.py` |
