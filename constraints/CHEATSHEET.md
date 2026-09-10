# `cgmath.constraints` Cheatsheet

Copy-paste recipes for `ProcrustesData`. Every block below runs in order,
sharing one namespace — start at [Setup](#setup). For the concepts see
[`README.md`](README.md); for the rest of the library see
[`../CHEATSHEET.md`](../CHEATSHEET.md).

## Contents

- [Setup](#setup)
- [The class at a glance](#the-class-at-a-glance)
- [Construction](#construction)
- [`attach()` — clusters and riders](#attach--clusters-and-riders)
- [`valid` and `compute()`](#valid-and-compute)
- [`update()` — drive it with a deformed cloud](#update--drive-it-with-a-deformed-cloud)
- [Reading the result](#reading-the-result)
- [Round trip: recover a known transform](#round-trip-recover-a-known-transform)
- [`scale_offset`](#scale_offset)
- [Off-centroid pivots](#off-centroid-pivots)
- [Non-rigid input → best fit](#non-rigid-input--best-fit)
- [Mirrored input → the reflection guard](#mirrored-input--the-reflection-guard)
- [Degenerate clusters](#degenerate-clusters)
- [Growing a constraint after a solve](#growing-a-constraint-after-a-solve)
- [Serialization and copying](#serialization-and-copying)
- [Housekeeping](#housekeeping)
- [Gotchas](#gotchas)

---

## Setup

Everything below reuses these. No mesh assets needed — a unit cube built
inline is the only fixture.

```python
import numpy as np

from cgmath.constraints import ProcrustesData
from cgmath.geometry import MeshData
from transforms import euler_to_matrix, matrix_identity, matrix_multiply, matrix_normalize, matrix_point_multiply, matrix_to_euler

CUBE_POINTS = np.array(
    [
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
    ]
)
CUBE_INDICES = np.array(
    [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
)
CUBE_COUNTS = np.array([4, 4, 4, 4, 4, 4])

cube  = MeshData(indices=CUBE_INDICES, counts=CUBE_COUNTS, points=CUBE_POINTS.copy())

FRONT = [0, 1, 2, 3]  # +Z face
BACK  = [4, 5, 6, 7]  # -Z face
```

---

## The class at a glance

`ProcrustesData(target)` — one positional argument, the rest point cloud.

| Member | Kind | Shape | Notes |
|---|---|---|---|
| `points` | field | `(N, 3)` | rest pose, set by the constructor |
| `transforms` | field | `(M, 4, 4)` | one rest matrix per constraint, grown by `attach()` |
| `clusters` | field | `(M, K)` | point indices per constraint, `-1` padded |
| `attach(transform, indices=None)` | method | — | append one constraint |
| `update(target)` | method | — | set the deformed cloud, then `compute()` |
| `compute()` | method | — | re-solve; raises if `valid` is `False` |
| `valid` | property | `bool` | `len(clusters) == len(transforms)` |
| `matrix` | property | `(M, 4, 4)` | the solved output matrices |
| `translate` | property | `(M, 3)` | `matrix[:, 3, :3]` |
| `rotate` | property | `(M, 3)` | euler **radians**, XYZ |
| `scale` | property | `(M, 3)` | uniform, all three components equal |
| `scale_offset` | property | `bool` | default `True`; setting it re-solves |

```python
p = ProcrustesData(CUBE_POINTS)
p.attach(matrix_identity(1)[0])
p.update(CUBE_POINTS + (0.0, 0.0, 1.0))

print(p.points.shape, p.transforms.shape, p.clusters.shape)   # (8, 3) (1, 4, 4) (1, 8)
print(p.matrix.shape, p.translate.shape, p.rotate.shape, p.scale.shape)
```

---

## Construction

From a point array. Note `np.asarray` — an ndarray is **not** copied,
`points` aliases what you passed in.

```python
pts = CUBE_POINTS.copy()
a   = ProcrustesData(pts)
print(a.points is pts)          # True — same buffer
```

From anything with a `.points` attribute (`MeshData`, `SkinData`, ...).
That path *does* copy.

```python
b = ProcrustesData(cube)
print(b.points is cube.points)  # False — copied
print(np.allclose(b.points, CUBE_POINTS))
```

From a nested list — `np.asarray` builds a fresh array.

```python
c = ProcrustesData([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
print(c.points.shape)           # (3, 3)
```

Fields are plain dataclass fields — you can fill `transforms` and
`clusters` in bulk instead of calling `attach()` in a loop.

```python
bulk            = ProcrustesData(CUBE_POINTS)
bulk.transforms = matrix_identity(2)
bulk.clusters   = np.array([FRONT, BACK])
print(bulk.valid)               # True
```

---

## `attach()` — clusters and riders

No `indices` means "every point".

```python
whole = ProcrustesData(CUBE_POINTS)
whole.attach(matrix_identity(1)[0])
print(whole.clusters)           # [[0 1 2 3 4 5 6 7]]
```

An index list picks the cluster. One `attach()` per constrained
transform; `transforms` and `clusters` stay index-parallel.

```python
faces = ProcrustesData(CUBE_POINTS)
faces.attach(matrix_identity(1)[0], FRONT)
faces.attach(matrix_identity(1)[0], BACK)
print(faces.transforms.shape, faces.clusters.shape)   # (2, 4, 4) (2, 4)
```

Indices are sorted, de-duplicated and stripped of negatives.

```python
messy = ProcrustesData(CUBE_POINTS)
messy.attach(matrix_identity(1)[0], [3, 1, -1, 0, 2, 3])
print(messy.clusters)           # [[0 1 2 3]]
```

Ragged clusters are padded to the widest row with `-1`. Padding slots are
skipped by the solve.

```python
ragged = ProcrustesData(CUBE_POINTS)
ragged.attach(matrix_identity(1)[0], [0, 1])
ragged.attach(matrix_identity(1)[0], [2, 3, 4, 5, 6])
print(ragged.clusters)
# [[ 0  1 -1 -1 -1]
#  [ 2  3  4  5  6]]
```

The rest matrix is where the transform sits **in the rest pose** — it
does not have to be at the cluster centroid.

```python
pivot        = matrix_identity(1)[0]
pivot[3, :3] = (0.0, 0.0, 1.0)          # 1 unit in front of the cube
offset       = ProcrustesData(CUBE_POINTS)
offset.attach(pivot, FRONT)
print(offset.transforms[0][3, :3])      # [0. 0. 1.]
```

---

## `valid` and `compute()`

`valid` is the "one cluster per transform" invariant. `compute()` raises
on an invalid object rather than solving garbage.

```python
empty = ProcrustesData(CUBE_POINTS)
print(empty.valid)              # False — nothing attached
try:
    empty.compute()
except RuntimeError as err:
    print(err)                  # Cannot compute invalid ProcrustesData object
```

`compute()` with no deformed cloud yet solves rest-against-rest, so every
output matrix comes back equal to its attached rest matrix.

```python
idle = ProcrustesData(CUBE_POINTS)
idle.attach(matrix_identity(1)[0], FRONT)
idle.compute()
print(np.allclose(idle.matrix[0], np.eye(4)))    # True
```

---

## `update()` — drive it with a deformed cloud

`update()` swaps in the deformed points and calls `compute()` for you.
Point *i* of the new cloud must be point *i* of the rest cloud.

```python
rider = ProcrustesData(CUBE_POINTS)
rider.attach(matrix_identity(1)[0], BACK)
rider.update(CUBE_POINTS + (0.0, 0.0, -2.0))
print(rider.translate[0])       # [0. 0. -2.]
```

It takes a `MeshData` (or anything with `.points`) too.

```python
cube.points = CUBE_POINTS + (0.0, 3.0, 0.0)
rider.update(cube)
print(rider.translate[0])       # [0. 3. 0.]
cube.points = CUBE_POINTS.copy()
```

Call it once per frame — the rest pose and the clusters are kept, only
the deformed cloud changes.

```python
track = ProcrustesData(CUBE_POINTS)
track.attach(matrix_identity(1)[0])
for frame in range(3):
    track.update(CUBE_POINTS + (0.0, float(frame), 0.0))
    print(frame, track.translate[0])
```

---

## Reading the result

All four properties are cached by the last `compute()`.

```python
read = ProcrustesData(CUBE_POINTS)
read.attach(matrix_identity(1)[0])
spin        = euler_to_matrix([(0.0, 0.0, np.radians(90.0))])[0]
spin[3, :3] = (0.0, 0.0, 4.0)
read.update(matrix_point_multiply(CUBE_POINTS, spin))

print(read.matrix[0])              # (4, 4) row-major
print(read.translate[0])           # [0, 0, 4] to float precision
print(np.degrees(read.rotate[0]))  # [ 0. -0. 90.]
print(read.scale[0])               # [1. 1. 1.]
```

`translate` is the matrix row and `rotate` is its euler decomposition —
they are consistent by construction.

```python
print(np.allclose(read.translate, read.matrix[:, 3, :3]))
print(np.allclose(read.rotate, matrix_to_euler(read.matrix)))
```

Before the first `compute()` the matrix-ish properties are `None` and
`scale` raises — check `valid` and solve first.

```python
cold = ProcrustesData(CUBE_POINTS)
cold.attach(matrix_identity(1)[0])
print(cold.matrix, cold.translate, cold.rotate)   # None None None
try:
    cold.scale
except AttributeError as err:
    print("scale needs a compute() first:", err)
```

---

## Round trip: recover a known transform

The canonical use: build a rigid matrix, apply it to the cloud, and get
it back out.

```python
known        = euler_to_matrix([(np.radians(10.0), np.radians(20.0), np.radians(30.0))])[0]
known[3, :3] = (0.5, -0.25, 2.0)

solve = ProcrustesData(CUBE_POINTS)
solve.attach(matrix_identity(1)[0])
solve.update(matrix_point_multiply(CUBE_POINTS, known))

assert np.allclose(solve.matrix[0], known)
assert np.allclose(np.degrees(solve.rotate[0]), (10.0, 20.0, 30.0))
assert np.allclose(solve.translate[0], (0.5, -0.25, 2.0))
assert np.allclose(solve.scale[0], 1.0)
print("round trip ok")
```

With an off-centroid rest pivot, the answer is the pivot carried through
the same rigid move.

```python
carried = ProcrustesData(CUBE_POINTS)
carried.attach(pivot)
carried.update(matrix_point_multiply(CUBE_POINTS, known))
assert np.allclose(carried.matrix[0], matrix_multiply(pivot, known))
print("carried pivot ok")
```

---

## `scale_offset`

`scale` always reports the cluster's uniform scale. `scale_offset`
decides whether that scale is *applied* — it scales the rest offset from
the cluster centroid to the pivot, and folds `1 / scale` into the output
3x3.

```python
grown = CUBE_POINTS * 3.0            # cube triples in size about the origin

on = ProcrustesData(CUBE_POINTS)
on.attach(pivot)                  # pivot sits 1 unit off the centroid
on.update(grown)
print(on.scale[0])                # [3. 3. 3.]
print(on.translate[0])            # [0. 0. 3.] — offset grew with the cluster
print(np.diag(on.matrix[0])[:3])  # [0.333 0.333 0.333] — 1 / scale in the 3x3
```

Turn it off for a rigid rider: the pivot keeps its rest distance from the
centroid and the 3x3 stays a pure rotation.

```python
off = ProcrustesData(CUBE_POINTS)
off.attach(pivot)
off.scale_offset = False
off.update(grown)
print(off.scale[0])                # [3. 3. 3.] — still reported
print(off.translate[0])            # [0. 0. 1.] — offset unchanged
print(np.diag(off.matrix[0])[:3])  # [1. 1. 1.]
```

Assigning `scale_offset` re-solves immediately, so flip it on a live
object without touching the points.

```python
off.scale_offset = True
print(off.translate[0])              # [0. 0. 3.] again
```

Because it re-solves, setting it on an object with nothing attached
raises — attach first.

```python
early = ProcrustesData(CUBE_POINTS)
try:
    early.scale_offset = False
except RuntimeError as err:
    print(err)                       # Cannot compute invalid ProcrustesData object
```

Want the rotation without the baked scale? Normalize the output.

```python
print(matrix_normalize(on.matrix)[0])
```

---

## Off-centroid pivots

A pivot far from its cluster amplifies the cluster's rotation into
translation — that is how a rivet arcs when the surface tilts.

```python
lever        = matrix_identity(1)[0]
lever[3, :3] = (0.0, 0.0, 5.0)

arc = ProcrustesData(CUBE_POINTS)
arc.attach(lever, FRONT)

tilt = euler_to_matrix([(0.0, np.radians(90.0), 0.0)])[0]
arc.update(matrix_point_multiply(CUBE_POINTS, tilt))
print(arc.translate[0])              # [5, 0, 0] — swung a quarter turn
```

---

## Non-rigid input → best fit

Noise, shear and non-uniform scale are not reproduced; the solve returns
the closest rotation plus a uniform scale.

```python
rng   = np.random.default_rng(3)
noisy = CUBE_POINTS + rng.normal(scale=0.05, size=CUBE_POINTS.shape)

fit   = ProcrustesData(CUBE_POINTS)
fit.attach(matrix_identity(1)[0])
fit.update(noisy)
print(np.degrees(fit.rotate[0]))  # a few degrees of best-fit rotation
print(fit.scale[0])               # near 1
```

The 3x3 of a best fit is orthonormal *up to* the scale factor, so
normalize before comparing against a pure rotation.

```python
r = matrix_normalize(fit.matrix)[0][:3, :3]
print(np.allclose(r @ r.T, np.eye(3), atol=1e-9))    # True
```

Sheared input still yields a rotation, never a shear.

```python
shear = CUBE_POINTS.copy()
shear[:, 0] += 0.4 * shear[:, 1]

sheared = ProcrustesData(CUBE_POINTS)
sheared.attach(matrix_identity(1)[0])
sheared.update(shear)
print(np.degrees(sheared.rotate[0]))
```

---

## Mirrored input → the reflection guard

A mirrored cluster would decompose to a determinant `-1` matrix. The
solve flips the smallest singular direction instead, so you always get a
proper rotation.

```python
mirrored = CUBE_POINTS * (-1.0, 1.0, 1.0)

flip = ProcrustesData(CUBE_POINTS)
flip.attach(matrix_identity(1)[0])
flip.update(mirrored)
print(np.linalg.det(flip.matrix[0][:3, :3]))  # +1, not -1
print(np.degrees(flip.rotate[0]))             # [180, 0, 180] — 180 deg about Y
```

---

## Degenerate clusters

Two points still solve (the free spin about the axis is arbitrary but
stable). A single point has no extent — the scale ratio is `0/0` and the
result is `NaN`. Three non-collinear points is the practical floor.

```python
pair = ProcrustesData(CUBE_POINTS)
pair.attach(matrix_identity(1)[0], [0, 1])
pair.update(CUBE_POINTS + (0.0, 0.0, 1.0))
print(pair.translate[0])                         # [0. 0. 1.]

lonely = ProcrustesData(CUBE_POINTS)
lonely.attach(matrix_identity(1)[0], [0])
with np.errstate(invalid="ignore", divide="ignore"):
    lonely.update(CUBE_POINTS + (0.0, 0.0, 1.0))
print(np.isnan(lonely.matrix[0]).any())          # True
```

---

## Growing a constraint after a solve

`attach()` is legal at any time; the next `compute()` includes the new
rows. There is no `detach()`.

```python
grow = ProcrustesData(CUBE_POINTS)
grow.attach(matrix_identity(1)[0], FRONT)
grow.update(CUBE_POINTS)
print(grow.matrix.shape)  # (1, 4, 4)

grow.attach(matrix_identity(1)[0], BACK)
grow.compute()
print(grow.matrix.shape)  # (2, 4, 4)
```

Batching many clusters into one object is the fast path — the whole
solve is vectorized over `M`.

```python
cloud = rng.random((200, 3))
batch = ProcrustesData(cloud)
for start in range(0, 200, 10):
    batch.attach(matrix_identity(1)[0], np.arange(start, start + 10))
batch.update(cloud + (1.0, 0.0, 0.0))
print(batch.matrix.shape)            # (20, 4, 4)
print(np.allclose(batch.translate, (1.0, 0.0, 0.0)))
```

---

## Serialization and copying

`ProcrustesData` inherits the `Data` base, so the three fields
round-trip through dicts, bytes and files. Cached results and the
deformed cloud are **not** serialized — re-`update()` after loading.

```python
import os
import tempfile

state = solve.to_dict()
print(sorted(state))                 # ['clusters', 'points', 'transforms']

restored = ProcrustesData.from_dict(state)
print(restored.valid)                # True
restored.update(matrix_point_multiply(CUBE_POINTS, known))
print(np.allclose(restored.matrix, solve.matrix))
```

`copy()` is a deep copy of the fields.

```python
clone = solve.copy()
print(clone is solve, clone == solve)     # False True
```

Files: `.npz`, `.json` and `.pkl`, picked from the extension by
`save()` / `load()`.

```python
tmp = tempfile.mkdtemp()

for ext in ("npz", "json", "pkl"):
    path = solve.save(os.path.join(tmp, f"rivet.{ext}"))
    back = ProcrustesData.load(path)
    print(ext, back.valid, back.points.shape, back.transforms.shape)
```

The explicit per-format methods work too.

```python
solve.save_npz(os.path.join(tmp, "explicit.npz"))
solve.save_json(os.path.join(tmp, "explicit.json"))
solve.save_pickle(os.path.join(tmp, "explicit.pkl"))

print(ProcrustesData.load_npz(os.path.join(tmp, "explicit.npz")).valid)
print(ProcrustesData.load_json(os.path.join(tmp, "explicit.json")).valid)
print(ProcrustesData.load_pickle(os.path.join(tmp, "explicit.pkl")).valid)
```

In-memory bytes, for caches and network hops.

```python
blob = solve.to_bytes()
print(type(blob), len(blob) > 0)
print(ProcrustesData.from_bytes(blob).valid)
```

Human-readable JSON as a string.

```python
print(solve.to_json()[:80])
```

---

## Housekeeping

`reset_cached_data()` drops the solved results and the deformed cloud but
keeps the fields — the object stays `valid`.

```python
solve.reset_cached_data()
print(solve.matrix, solve.valid)     # None True
solve.update(matrix_point_multiply(CUBE_POINTS, known))
print(np.allclose(solve.matrix[0], known))
```

`info()` returns the class docstring (the dataclass signature here), and
equality compares the serialized fields.

```python
print(ProcrustesData.info())                 # ProcrustesData(target: 'np.ndarray')
print(solve == solve.copy())                 # True
print(solve == ProcrustesData(CUBE_POINTS))  # False
```

`match()` is a name-glob helper on the `Data` base. `ProcrustesData` has
no `name`, so it always answers `False`.

```python
print(solve.match("*"))              # False
```

The base class rejects writes to undeclared attributes, so typos raise
instead of quietly creating state.

```python
try:
    solve.cluster = np.array([0, 1])       # missing the 's'
except AttributeError as err:
    print(str(err)[:60])
```

---

## Gotchas

| Gotcha | What happens |
|---|---|
| `ProcrustesData(points=..., transforms=..., clusters=...)` | `TypeError` — the constructor takes one positional `target` |
| Passing an ndarray to the constructor | `points` aliases your array; pass `.copy()` if you will mutate it |
| Passing a list to `attach()` | `TypeError` — the transform must be a `(4, 4)` ndarray |
| First `attach()` | stores a view of your matrix; later ones copy. Pass a `.copy()` to be safe |
| Reading `scale` before `compute()` | `AttributeError`; `matrix` / `rotate` / `translate` return `None` |
| Setting `scale_offset` before `attach()` | `RuntimeError` — the setter re-solves |
| Single-point cluster | `NaN` matrix (zero extent → `0/0` scale) |
| Expecting shear or non-uniform scale back | not recoverable — orthogonal solve, uniform scale only |
| `rotate` in degrees | it is **radians**; wrap in `np.degrees()` |
| Loading a saved object and reading `matrix` | `None` until you `update()` or `compute()` again |
| Needing to remove a cluster | no `detach()`; rebuild, or assign `transforms` / `clusters` in bulk |
