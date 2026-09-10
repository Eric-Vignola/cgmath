# `cgmath.constraints` — orthogonal Procrustes alignment

One class, `ProcrustesData`. Hand it a rest point cloud, staple 4x4
transforms onto clusters of point indices, then feed it a deformed copy
of that cloud. It hands back the 4x4 each transform should now have.

This is the "rivet a locator to a patch of a deforming mesh" problem.
No DCC, no constraint node, no per-frame graph evaluation — just numpy.

```python
import numpy as np
from cgmath.constraints import ProcrustesData
from transforms import euler_to_matrix, matrix_identity, matrix_point_multiply

rest = np.array([[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                 [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]])

# a known rigid move: 30 degrees about Y, then translate
rigid        = euler_to_matrix([(0.0, np.radians(30.0), 0.0)])[0]
rigid[3, :3] = (1.0, 2.0, 3.0)

p = ProcrustesData(rest)
p.attach(matrix_identity(1)[0])          # one transform, all points
p.update(matrix_point_multiply(rest, rigid))

assert np.allclose(p.matrix[0], rigid)   # solved it back out
```

---

## The model

| Piece | Lives in | Shape | What it is |
|---|---|---|---|
| Rest points | `points` | `(N, 3)` | the reference pose everything is measured against |
| Clusters | `clusters` | `(M, K)` | per-transform rows of point indices, `-1` padded |
| Attached transforms | `transforms` | `(M, 4, 4)` | the rest-pose matrix of each constrained transform |
| Deformed points | passed to `update()` | `(N, 3)` | the current pose of the same cloud |

`M` is the number of constraints, `K` the widest cluster. Short cluster
rows are padded with `-1` and those slots are ignored by the solve.

A **cluster** is just the set of points that a transform listens to.
Pick the ring of verts around a cheek and the transform rides the cheek;
pick every vert and the transform rides the whole object.

`attach(transform, indices)` appends one row to `clusters` and one matrix
to `transforms` — they stay index-parallel. There is no `detach()`;
build a fresh `ProcrustesData` instead.

---

## The solve

Per cluster, on every `compute()`:

1. Centroids `c0` (rest) and `c1` (deformed) of the cluster's points.
2. Mean-centred vectors `V0`, `V1`; covariance `H = V1ᵀ V0`.
3. `R` = the rotation factor of `H` — a polar decomposition via SVD,
   with a reflection guard so a mirrored cluster comes back as a
   rotation rather than a determinant `-1` flip.
4. Uniform scale `s = ‖V1‖ / ‖V0‖`.
5. The attached transform is carried rigidly: its rest offset from `c0`
   is rotated by `R` (and scaled by `s` when `scale_offset` is on) and
   re-planted at `c1`.

Only **rotation, uniform scale and translation** are recovered — that is
what "orthogonal Procrustes" means. Shear and non-uniform scale in the
cluster's motion are absorbed as best-fit rotation, not reproduced.

The rotation factor is looked up from an optional numba polar-decomposition
kernel first and falls back to the vectorized SVD path above.

---

## Conventions

- **Row-major, Maya style.** Translation lives at `M[3, :3]`. Points
  transform as `p @ M`. Matches everything in
  [`transforms`](https://github.com/Eric-Vignola/transforms).
- **`rotate` is radians**, XYZ order — it is `matrix_to_euler(matrix)`.
- **`scale` is uniform**, reported as `(M, 3)` with all three components
  equal.
- **Results are cached.** `matrix` / `rotate` / `translate` / `scale`
  are `None` (or raise) until a successful `compute()`.
- **Clusters need spread.** A single-point cluster has zero extent, so
  the scale ratio is `0/0` and the result is `NaN`. Three
  non-collinear points is the practical floor.

---

## When to reach for it

| Situation | Fit |
|---|---|
| Rivet a joint / locator to a deforming mesh patch | yes — this is the job |
| Best-fit rigid alignment of two matched point sets | yes — one cluster, all points |
| Recover per-region rotation from a blendshape or cache | yes — one cluster per region |
| Recover shear / non-uniform scale | no — orthogonal solve only |
| Solve unmatched point sets (no correspondence) | no — indices must correspond |

For the smoothed-rotation flavour of the same math applied per vertex,
see `DeltaMushData(method="PROCRUSTES")` in
[`cgmath.geometry.deform`](../geometry/deform).

---

## Files

| File | Contents |
|---|---|
| `__init__.py` | re-exports `ProcrustesData` |
| `procrustes.py` | the whole implementation |

---

## Quick taste

### Two clusters, two riders

```python
from cgmath.constraints import ProcrustesData
from transforms import matrix_identity
import numpy as np

rest = np.array([[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                 [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]])

p = ProcrustesData(rest)
p.attach(matrix_identity(1)[0], [0, 1, 2, 3])  # front face
p.attach(matrix_identity(1)[0], [4, 5, 6, 7])  # back face

moved = rest.copy()
moved[[4, 5, 6, 7]] += (0.0, 0.0, -1.0)          # push the back face back
p.update(moved)

print(p.translate)        # [[0, 0, 0], [0, 0, -1]] — only the back rider moved
```

### Drive it from a `MeshData`

```python
from cgmath.geometry import MeshData

cube = MeshData(
    indices=np.array([0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6,
                      6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]),
    counts = np.array([4, 4, 4, 4, 4, 4]),
    points = rest.copy(),
)

c = ProcrustesData(cube)                  # takes a copy of cube.points as the rest pose
c.attach(matrix_identity(1)[0])
cube.points = cube.points + (0.0, 2.0, 0.0)
c.update(cube)

print(c.translate[0])                     # [0, 2, 0]
```

---

## Where to go next

| You want to... | Read |
|---|---|
| Every method, property and gotcha, with runnable blocks | [`CHEATSHEET.md`](CHEATSHEET.md) |
| The rest of the library | [`../README.md`](../README.md) |
| Row-major matrix helpers used above | [`../transforms`](https://github.com/Eric-Vignola/transforms) |
