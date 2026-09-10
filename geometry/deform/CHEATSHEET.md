# `cgmath.geometry.deform` — cheatsheet

Copy-paste recipes for every deformer in the package. The blocks run top to
bottom as one script and share a namespace — the fixtures built in **Setup**
are reused everywhere below.

## Contents

| Section | Covers |
|---|---|
| [Setup](#setup) | fixtures: a cube, a bumped grid, a noisy pose |
| [Import surface](#import-surface) | what the package exports |
| [`FFDData`](#ffddata--free-form-deformation) | lattice, bind, update, influence, outside modes |
| [`DeltaMushData`](#deltamushdata--wobble-removal) | TBN / PROCRUSTES / DDM, smoothing, weight |
| [`PatchRelaxData`](#patchrelaxdata--patch-aware-relaxation) | alpha, surface blend, mask, pinning |
| [`SkinDeformData`](#skindeformdata--skeletal-skinning) | LBS / DQS, rigs, raw matrices, batching |
| [`WrapData`](#wrapdata--rbf-wrap) | kernels, radius, geodesic radius, deform |
| [Persistence](#persistence) | what survives `copy()` and what does not |
| [Gotchas](#gotchas) | the mistakes that cost an afternoon |

---

## Setup

No mesh assets ship with this package — build geometry inline.

```python
import numpy as np
from cgmath.geometry import MeshData


def cube(size=1.0):
    """Closed unit cube: 8 points, 6 quads, no border edges."""
    points = np.array(
        [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ]
    ) * size
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    return MeshData(
        name    = "cube",
        points  = points,
        indices = np.asarray(faces, dtype=np.int32).ravel(),
        counts  = np.full(6, 4, dtype=np.int32),
    )


def grid(n=9, height=0.0):
    """Open n x n quad grid on XY in [0, 1], optionally bumped in Z."""
    t      = np.linspace(0.0, 1.0, n)
    points = np.array([[x, y, 0.0] for y in t for x in t])
    if height:
        points[:, 2] = height * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices += [a, a + 1, a + n + 1, a + n]

    return MeshData(
        name    = f"grid{n}",
        points  = points,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
    )


box   = cube()
plane = grid(9, height=0.25)          # the "rest" / baseline shape

rng          = np.random.default_rng(0)
noisy        = plane.copy()                  # a scruffy "deformed" pose of it
noisy.points = plane.points + rng.normal(scale=0.01, size=plane.points.shape)

print(box.point_count, plane.point_count)
```

---

## Import surface

Five operators plus one enum, all re-exported from the package root.

```python
from cgmath.geometry.deform import (
    DeformMethod,      # LBS | DQS   (skin deform blend math)
    DeltaMushData,     # wobble removal
    FFDData,           # free-form deformation lattice
    PatchRelaxData,    # patch-aware relaxation
    SkinDeformData,    # skeletal skinning
    WrapData,          # RBF wrap / cage deform
)
```

Every one of them subclasses `cgmath.geometry._base.Data`, so they all carry
`.copy()`, `.to_dict()`, `.info()`, `.save()` / `.load()` — see
[Persistence](#persistence) for which of those are actually useful per class.

| Class | Rest geometry given to | Cache step | Evaluate with | Returns |
|---|---|---|---|---|
| `FFDData` | `__init__` (the *lattice*) | `bind(target)` | `update(lattice)` | mesh copy or `(N, 3)` |
| `DeltaMushData` | `__init__` (the rest mesh) | `bind()` (auto) | `apply(deformed)` | mesh copy or `(N, 3)` |
| `PatchRelaxData` | `__init__` (the baseline mesh) | `bind()` (auto) | `apply(deformed)` / `relax(pts)` | mesh copy or `(N, 3)` |
| `SkinDeformData` | `__init__` (the bind mesh) | `bind()` (auto) | `apply(pose, target)` / `deform(pts, mats)` | mesh copy or `(N, 3)` / `(F, N, 3)` |
| `WrapData` | `set_source()` | lazy, on first `deform` | `deform(points)` | `(N, 3)` |

Mesh in, mesh out; array in, array out. Everything is row-vector
(`p' = p @ M`, translation in `M[3, :3]`).

---

## `FFDData` — free-form deformation

A 3-D lattice of control points drives the mesh. Displacement-based: the mesh
sees `interp(deformed_lattice − rest_lattice)`, so a lattice at rest is exactly
the identity.

Sizes follow the Maya convention — `divisions` counts **control points**, not
cells. `(2, 2, 2)` is the minimum.

### Build a lattice

```python
lattice_mesh = FFDData.create_lattice((3, 3, 3))     # unit cube [-0.5, 0.5]^3
print(lattice_mesh.points.shape, lattice_mesh.counts[:4])   # 27 CPs, all quads
```

Place it around geometry with an explicit bounding box:

```python
pad = 0.05
lattice_mesh = FFDData.create_lattice(
    (3, 3, 3),
    bbox_min = plane.points.min(axis=0) - pad,
    bbox_max = plane.points.max(axis=0) + pad,
    name     = "ffd1",
)
```

### Bind and deform — end to end

```python
ffd = FFDData.from_mesh(lattice_mesh, divisions=(3, 3, 3))
ffd.bind(plane)                       # expensive Newton solve, once per mesh

deformed_lattice = ffd.lattice.copy() # (3, 3, 3, 3) -> [ix, iy, iz, xyz]
deformed_lattice[:, -1, :, 0] += 0.5  # shear: push the max-Y CP plane in +X

sheared = ffd.update(deformed_lattice)          # MeshData in -> MeshData out
print(type(sheared).__name__, np.abs(sheared.points - plane.points).max())
assert np.abs(sheared.points - plane.points).max() > 0.4
```

A rest-pose lattice is the identity:

```python
assert np.allclose(ffd.update(ffd.lattice).points, plane.points, atol=1e-9)
```

### The three constructor forms

```python
flat   = lattice_mesh.points       # (27, 3)
grid4d = flat.reshape(3, 3, 3, 3)  # (lx, ly, lz, 3)

a = FFDData(grid4d)                                       # divisions inferred
b = FFDData(flat, divisions=(3, 3, 3))                    # flat array + divisions
c = FFDData.from_mesh(lattice_mesh, divisions=(3, 3, 3))  # MeshData + divisions
print(a.divisions, b.lattice.shape, c.lattice.shape)
```

`from_mesh` wants the **lattice**, not the mesh you are deforming — it needs
exactly `nx * ny * nz` points in structured-grid order.

```python
try:
    FFDData.from_mesh(plane, divisions=(3, 3, 3))     # 81 points != 27
except ValueError as e:
    print("as expected:", e)
```

### Bind to raw points instead of a mesh

```python
cloud   = rng.uniform(0.0, 1.0, (50, 3))
ffd_pts = FFDData.from_mesh(lattice_mesh, divisions=(3, 3, 3))
ffd_pts.bind(cloud)                              # (N, 3) in
moved = ffd_pts.update(deformed_lattice)         # (N, 3) out
print(type(moved).__name__, moved.shape)
```

### `local_influence` — interpolation width

Per-axis Bernstein window in control points (Maya's `localInfluenceS/T/U`).
`(2, 2, 2)` is plain trilinear and the default; wider windows smear each
control point over more of the lattice.

```python
ffd.local_influence = (2, 2, 2)
hard                = ffd.update(deformed_lattice).points.copy()

ffd.local_influence = 3               # scalar broadcasts to (3, 3, 3)
soft                = ffd.update(deformed_lattice).points.copy()

print(ffd.local_influence, np.abs(hard - soft).max() > 0)
```

Changing it after `bind()` is free — no rebind needed.

### `outside` modes and `falloff_radius`

| Mode | Points outside the lattice |
|---|---|
| `"extrapolate"` (default) | keep deforming past the boundary |
| `"freeze"` | never move |
| `"falloff"` | smoothstep decay over `falloff_radius` cell-widths |

```python
unit = FFDData.create_lattice((3, 3, 3), bbox_min=[0, 0, 0], bbox_max=[1, 1, 1])
probe = np.array([[0.5, 0.5, 0.5],    # inside
                  [-0.5, 0.5, 0.5]])  # outside

for mode in ("extrapolate", "freeze", "falloff"):
    f                = FFDData.from_mesh(unit, divisions=(3, 3, 3))
    f.outside        = mode
    f.falloff_radius = 2.0            # cell-widths, only used by "falloff"
    f.bind(probe)

    lifted = f.lattice.copy()
    lifted[..., 2] += 1.0             # lift the whole lattice by 1 in Z
    out = f.update(lifted)
    print(f"{mode:12s} dz={np.round(out[:, 2] - probe[:, 2], 3)}  w={np.round(f.weights, 3)}")
```

Both properties recompute the per-point weights in place — no rebind.

```python
try:
    f.outside = "wobble"
except ValueError as e:
    print("as expected:", e)
```

### Inspect the bind

```python
print("valid   ", ffd.valid)          # True once bind() ran
print("cells   ", ffd.cells.shape)    # (N, 3) containing cell per point
print("uvw     ", ffd.uvw.shape)      # (N, 3) parametric coords in that cell
print("weights ", ffd.weights.shape)  # (N,) per-point deformation weight
print("points  ", ffd.points.shape)   # (N, 3) last evaluated positions
```

### `compute()` — evaluate without the return plumbing

`update()` is `compute()` plus "hand me back a mesh or an array".

```python
ffd.compute(deformed_lattice)           # returns None, fills ffd.points
print(np.allclose(ffd.points, ffd.update(deformed_lattice).points))
```

Calling either before `bind()` raises:

```python
try:
    FFDData(grid4d).compute(grid4d)
except RuntimeError as e:
    print("as expected:", e)
```

### `to_mesh()` — visualise or export the lattice

```python
rest_cage  = ffd.to_mesh()                  # rest lattice as quads
posed_cage = ffd.to_mesh(deformed_lattice)  # any (lx,ly,lz,3) or (N,3)
print(rest_cage.point_count, posed_cage.points[:, 0].max())
```

### Low-level FFD helpers

Re-exported on `deform.ffd` from `cgmath.geometry.utils.main`.

```python
from cgmath.geometry.deform.ffd import (
    assign_cells,           # points + lattice -> (cells, uvw)
    bernstein_eval,         # cells + uvw + lattice delta -> displacement
    build_lattice_topology, # (lx, ly, lz) -> quad (indices, counts)
)

lat = FFDData.create_lattice((3, 3, 3)).points.reshape(3, 3, 3, 3)
cells, uvw = assign_cells(np.array([[0.1, 0.0, 0.0]]), lat)

delta         = np.zeros_like(lat)
delta[..., 2] = 1.0                     # translate every CP up by 1
disp          = bernstein_eval(cells, uvw, delta, (1, 1, 1), np.array([2, 2, 2]))
print(cells, np.round(uvw, 3), disp)    # -> uniform 1.0 in Z

indices, counts = build_lattice_topology(2, 3, 4)
print(indices.shape, counts.shape)
```

`bernstein_eval` takes **cell** counts, not control-point counts — that is
`local_influence - 1` and `divisions - 1`.

---

## `DeltaMushData` — wobble removal

Smooths away low-frequency deformation noise (twist, jitter, candy-wrap) while
putting the high-frequency surface detail back. Bind on a rest pose, apply to
any deformed pose of the same point cloud.

### End to end

```python
mush     = DeltaMushData(plane, smooth_iterations=10, pin_borders=True)
cleaned  = mush.apply(noisy)             # auto-binds on the first call

edge     = plane.get_border_vertices(flatten=True)
interior = np.setdiff1d(np.arange(plane.point_count), edge)

print(type(cleaned).__name__, mush.valid)
before = np.abs(noisy.points[interior] - plane.points[interior]).max()
after  = np.abs(cleaned.points[interior] - plane.points[interior]).max()
print(f"max interior deviation {before:.4f} -> {after:.4f}")
assert after < before
```

Interior vertices only: `pin_borders=True` (the default) excludes the rim from
the smoothing, so border vertices keep whatever the deformed pose gave them.

The output is `smooth(deformed) + rotated rest detail` — it keeps the deformed
pose's low frequencies and swaps its high frequencies for the rest mesh's.

Applying to the rest pose is the identity:

```python
assert np.allclose(mush.apply(plane).points, plane.points, atol=1e-7)
```

### The three frame-construction methods

| `method` | How the frame is recovered | Notes |
|---|---|---|
| `"TBN"` (default) | first-face cross-product tangent frame | closest match to Maya's `deltaMush`; needs a `MeshData` |
| `"PROCRUSTES"` | per-vertex polar decomposition of the neighbour offsets | best least-squares rotation |
| `"DDM"` | Direct Delta Mush — precomputed weighted offsets + polar fit | robust on irregular topology, cheapest per apply |

```python
for method in ("TBN", "PROCRUSTES", "DDM"):
    m   = DeltaMushData(plane, smooth_iterations=10, method=method)
    out = m.apply(noisy)
    print(f"{method:11s} {np.abs(out.points[interior] - plane.points[interior]).max():.5f}")
```

Case-insensitive, and switching invalidates the bind:

```python
m = DeltaMushData(plane, smooth_iterations=4, method="procrustes")
print(m.method)                         # -> "PROCRUSTES"
m.bind()
m.method = "ddm"
print(m.method, m.valid)                # -> "DDM" False, rebinds on next apply
```

### Smoothing knobs

```python
m = DeltaMushData(
    plane,
    smooth_iterations = 20,    # Laplacian passes; higher flattens lower frequencies
    smooth_step_size  = 0.25,  # per-iteration reception; smaller is gentler
    pin_borders       = True,  # border vertices keep their exact position
)
m.bind()
print(m.smooth_iterations, m.smooth_step_size, m.pin_borders)
```

All three invalidate the bind when changed; `weight` does not.

```python
m.smooth_iterations = 6
print("needs rebind:", not m.valid)
```

### `weight` — blend against the raw input

`1.0` is the full reconstruction, `0.0` hands back the *smoothed* mesh.

```python
soft = DeltaMushData(plane, smooth_iterations=10, weight=0.0)
assert np.allclose(soft.apply(plane).points, soft.smoothed_points, atol=1e-9)

soft.weight = 0.5  # cheap, no rebind
soft.weight = 5.0  # clamped into [0, 1]
print(soft.weight)
```

### Borders

`pin_borders=True` forces the smoothing iteration count to zero on open-edge
vertices, so an open mesh does not shrink at its rim. A closed mesh has no
border vertices and is unaffected.

```python
pinned = DeltaMushData(plane, smooth_iterations=20, pin_borders=True)
pinned.bind()
edge = plane.get_border_vertices(flatten=True)
assert np.allclose(pinned.smoothed_points[edge], plane.points[edge], atol=1e-12)

loose = DeltaMushData(plane, smooth_iterations=20, pin_borders=False)
loose.bind()
print("border moved:", np.abs(loose.smoothed_points[edge] - plane.points[edge]).max() > 1e-6)
```

### Point clouds

Without faces there is no winding, so supply a neighbour matrix — and `"TBN"`
silently falls back to `"DDM"` at bind time.

```python
points    = plane.points.copy()
neighbors = plane.get_edge_vertex_neighbors()     # (N, K) int32, -1 padded

pc = DeltaMushData(points, neighbors=neighbors, smooth_iterations=5, method="TBN")
pc.bind()
print(pc.method)                        # -> "DDM"

out = pc.apply(noisy.points)            # array in -> array out
print(type(out).__name__, out.shape)
```

Forgetting the neighbours is an error, not a guess:

```python
try:
    DeltaMushData(np.zeros((5, 3)))
except ValueError as e:
    print("as expected:", e)
```

### Inspect the bind

```python
mush.bind()
print("valid          ", mush.valid)
print("smoothed_points", mush.smoothed_points.shape)  # smoothed rest pose
print("local_deltas   ", mush.local_deltas.shape)     # (N, 3) encoded detail
print("points         ", mush.points.shape)           # last reconstruction
print("neighbors      ", mush.neighbors.shape)
print("rest_points    ", mush.rest_points.shape)
print("border_vertices", None if mush.border_vertices is None else mush.border_vertices.shape)
```

`local_deltas` are tangent-space for `"TBN"` and world-space
`rest - smooth_rest` for `"PROCRUSTES"` / `"DDM"`.

---

## `PatchRelaxData` — patch-aware relaxation

Smooths a deformed mesh while reproducing the **edge flow** of a baseline mesh.
Unlike Laplacian smoothing it does not equalise edge spans, and unlike Delta
Mush it does not collapse them. (de Goes et al., Pixar, SIGGRAPH '18.)

The baseline you bind to picks the behaviour:

| Baseline | Use |
|---|---|
| the mesh being relaxed | **modeling** — keep relative span spacing |
| the undeformed mesh | **rigging** — restore rest features under articulation |
| any shape in the rig stack | **cleanup** — resolve clumping and foldovers |

### End to end

```python
relaxer = PatchRelaxData(plane, iterations=30)
relaxed = relaxer.apply(noisy)          # auto-binds

print(type(relaxed).__name__, relaxer.valid)
print(np.linalg.norm(noisy.points - plane.points, axis=1).mean(),
      np.linalg.norm(relaxed.points - plane.points, axis=1).mean())
```

The baseline is a fixed point:

```python
assert np.allclose(PatchRelaxData(plane, iterations=25).apply(plane).points,
                   plane.points, atol=1e-9)
```

### `relax()` — arrays without the mesh plumbing

```python
out = relaxer.relax(noisy.points)       # (N, 3) -> (N, 3), input untouched
print(type(out).__name__, out.shape)
```

`apply()` is `relax()` plus mesh copy-out; a `MeshData` target gives a
`MeshData` back, a raw array gives an array.

```python
print(type(relaxer.apply(noisy.points)).__name__)
```

### `alpha` — restore the baseline, or just smooth

`1.0` fully rebuilds the baseline patch layout, `0.0` reduces to span-aware
smoothing with no baseline restoration.

```python
flat = grid(9, height=0.0).points        # the detail has been flattened away

restored = PatchRelaxData(plane, iterations=200, alpha=1.0).relax(flat)
smoothed = PatchRelaxData(plane, iterations=200, alpha=0.0).relax(flat)

print(f"peak Z  baseline={plane.points[:, 2].max():.3f} "
      f"alpha=1 {restored[:, 2].max():.3f}  alpha=0 {smoothed[:, 2].max():.3f}")
assert restored[:, 2].max() > smoothed[:, 2].max()
```

### `surface_blend` — volume control

Fraction of the surface-constrained solve mixed over the unconstrained 3-D one.
`0.0` is pure 3-D; the paper's cleanup recipe is `0.8`.

```python
free = PatchRelaxData(plane, iterations=40, surface_blend=0.0).relax(noisy.points)
kept = PatchRelaxData(plane, iterations=40, surface_blend=0.8).relax(noisy.points)
print(np.isfinite(kept).all(), np.abs(free - kept).max() > 0)
```

Costs an extra decal-map rebuild per iteration when non-zero.

### `mask` and `pin_borders`

`mask` is a per-vertex `[0, 1]` step multiplier — painted falloff. `0` pins a
vertex, but a pinned vertex still influences its neighbours.

```python
frozen = PatchRelaxData(plane, iterations=20, mask=np.zeros(plane.point_count))
assert np.allclose(frozen.relax(noisy.points), noisy.points)

half = PatchRelaxData(plane, iterations=20, mask=np.full(plane.point_count, 0.5))
print(np.abs(half.relax(noisy.points) - noisy.points).max() > 0)
```

Reassigning the mask takes effect immediately — the step scale is never cached.

```python
relaxer.mask = np.zeros(plane.point_count)
assert np.allclose(relaxer.relax(noisy.points), noisy.points)
relaxer.mask = None
```

Borders hold still by default:

```python
edge                = np.asarray(plane.get_border_vertices(flatten=True), dtype=np.int64)
relaxer.pin_borders = True
assert np.array_equal(relaxer.relax(noisy.points)[edge], noisy.points[edge])

relaxer.pin_borders = False
print("border moved:", not np.array_equal(relaxer.relax(noisy.points)[edge], noisy.points[edge]))
relaxer.pin_borders = True
```

### Runtime knobs

All of these are cheap — none of them invalidate the bind.

```python
relaxer.iterations    = 40    # Jacobi sweeps per apply
relaxer.alpha         = 0.75  # baseline restoration blend
relaxer.step_size     = 0.5   # explicit update fraction; larger can oscillate
relaxer.surface_blend = 2.0   # clamped to [0, 1]
print(relaxer.iterations, relaxer.alpha, relaxer.step_size, relaxer.surface_blend)

try:
    relaxer.iterations = -1
except ValueError as e:
    print("as expected:", e)
relaxer.iterations = 30
```

### Inspect the bind

```python
relaxer.bind()
print("valid      ", relaxer.valid)
print("ring       ", relaxer.ring.shape)        # (N, K) winding-ordered 1-ring
print("valence    ", relaxer.valence[:6])       # neighbours per vertex
print("is_boundary", relaxer.is_boundary[:6])   # open ring?
print("decal_maps ", relaxer.decal_maps.shape)  # (N, K, 2) geodesic-polar stencils
print("weights    ", relaxer.weights.shape)     # (N, K) span-aware edge weights
print("points     ", relaxer.points.shape)      # last relaxed positions
```

### It needs face topology

```python
try:
    PatchRelaxData(plane.points)            # bare points have no 1-ring order
except ValueError as e:
    print("as expected:", e)

try:
    PatchRelaxData(plane, mask=np.ones(3))  # mask must match the point count
except ValueError as e:
    print("as expected:", e)
```

---

## `SkinDeformData` — skeletal skinning

Bind-pose mesh + skin weights + joint order in, posed mesh out. The bind step
resolves influence names to joint columns once and compacts the weights to
`(N, K)`, so an animation loop never re-resolves names.

### Build a skinned bar

```python
from cgmath.geometry.skin_weights import SkinData
from cgmath.hierarchy import HierarchyData, TransformData

bar     = grid(5)                         # 25 points spanning x in [0, 1]
t       = bar.points[:, 0]
weights = np.stack([1.0 - t, t], axis=1)  # linear root -> tip falloff
skin    = SkinData(weights=weights, influences=["root", "tip"])

bind_rig = HierarchyData([
    TransformData(name="root"),
    TransformData(name="tip", parent_node="root", translate=(1.0, 0.0, 0.0)),
])
```

### End to end with a rig

```python
deformer = SkinDeformData(mesh=bar, skin=skin, bind_rig=bind_rig, name="bar")

posed_rig = HierarchyData([
    TransformData(name="root"),
    TransformData(name="tip", parent_node="root",
                  translate=(1.0, 0.0, 0.0), rotate=(0.0, 0.0, 45.0)),
])

posed = deformer.apply(posed_rig, bar)      # MeshData target -> MeshData copy
print(type(posed).__name__, np.abs(posed.points - bar.points).max() > 0)
```

The bind pose is a fixed point:

```python
assert np.allclose(deformer.apply(bind_rig, bar).points, bar.points, atol=1e-8)
```

With no target the bind points are deformed and an array comes back:

```python
print(type(deformer.apply(posed_rig)).__name__, deformer.apply(posed_rig).shape)
```

### Authored inverse bind matrices

Prefer the file's matrices when you have them — glTF stores them explicitly
because they need not equal `inv(bind_rig.world_matrix)`.

```python
identity_ibm = np.stack([np.eye(4), np.eye(4)])
authored     = SkinDeformData(mesh=bar, skin=skin, inverse_bind_matrices=identity_ibm)
print(authored.inverse_bind_matrices.shape)
```

One of `inverse_bind_matrices` or `bind_rig` is required — there is no default
bind pose:

```python
try:
    SkinDeformData(mesh=bar, skin=skin)
except ValueError as e:
    print("as expected:", e)
```

### `joints` — column order wins over influence order

```python
swapped = SkinDeformData(
    mesh=bar, skin=skin,
    joints=["tip", "root"],                 # matrices are ordered to THIS list
    inverse_bind_matrices=identity_ibm,
)
print(swapped.joints)
```

### `DeformMethod` — LBS vs DQS

| Value | Blend | Behaviour |
|---|---|---|
| `DeformMethod.LBS` (`"lbs"`, default) | linear matrix blend | fast; collapses volume on a bend |
| `DeformMethod.DQS` (`"dqs"`) | dual quaternion + separate stretch | holds volume through a bend, keeps scale |

```python
lbs = SkinDeformData(mesh=bar, skin=skin, bind_rig=bind_rig, method=DeformMethod.LBS)
dqs = SkinDeformData(mesh=bar, skin=skin, bind_rig=bind_rig, method="dqs")

print(lbs.method, dqs.method)            # stored as the enum's VALUE, a str
print(lbs.method_enum, dqs.method_enum)  # -> DeformMethod.LBS / DQS

a = lbs.apply(posed_rig)
b = dqs.apply(posed_rig)
print("they differ on a bend:", np.abs(a - b).max() > 1e-6)
```

Switching the method keeps the bind — every method shares one bind state.

```python
lbs.bind()
indices         = lbs.influence_indices
lbs.method_enum = DeformMethod.DQS
print(lbs.influence_indices is indices)

try:
    SkinDeformData(mesh=bar, skin=skin, bind_rig=bind_rig, method="wobble")
except ValueError as e:
    print("as expected:", e)
```

### `pose_indices()` and raw matrices

`apply()` also takes raw world matrices, already ordered to `joints` — resolve
the columns once and reuse them across frames.

```python
cols = deformer.pose_indices(posed_rig)     # (J,) columns of the rig
print(cols)

world = np.asarray(posed_rig.world_matrix)[cols]
same  = deformer.apply(world, bar)
assert np.allclose(same.points, posed.points, atol=1e-12)
```

### `deform()` — the raw kernel

Points plus **skin matrices**, which are `inverse_bind @ world_posed` in that
order (row-vector; the reverse produces plausible garbage rather than an error).

```python
skin_matrices = deformer.inverse_bind_matrices @ world
out           = deformer.deform(bar.points, skin_matrices)
assert np.allclose(out, posed.points, atol=1e-12)
```

### Animation loop / batched poses

A `(F, J, 4, 4)` stack returns `(F, N, 3)`. Feed the **bind** mesh every frame —
feeding last frame's result back in compounds the deformation.

```python
frames = np.stack([
    np.asarray(HierarchyData([
        TransformData(name="root"),
        TransformData(name="tip", parent_node="root",
                      translate=(1.0, 0.0, 0.0), rotate=(0.0, 0.0, float(a))),
    ]).world_matrix)[cols]
    for a in (0.0, 15.0, 30.0, 45.0)
])

sequence = deformer.apply(frames)
print(sequence.shape)                       # (4, 25, 3)
print(np.allclose(deformer.points, sequence[-1]))
```

A batched pose cannot fill a single mesh:

```python
try:
    deformer.apply(frames, bar)
except ValueError as e:
    print("as expected:", e)
```

### Mesh-less deformers

`mesh=None` is supported — bind and deform points you pass in.

```python
headless = SkinDeformData(mesh=None, skin=skin, inverse_bind_matrices=identity_ibm)
headless.bind()
print(headless.valid, headless.rest_points)
print(headless.apply(np.stack([np.eye(4), np.eye(4)]), bar.points).shape)
```

### Inspect the bind

```python
deformer.bind()
print("valid                ", deformer.valid)
print("joints               ", deformer.joints)
print("influence_indices    ", deformer.influence_indices.shape)  # (N, K) int32
print("influence_weights    ", deformer.influence_weights.shape)  # (N, K)
print("inverse_bind_matrices", deformer.inverse_bind_matrices.shape)
print("rest_points          ", deformer.rest_points.shape)
print("points               ", deformer.points.shape)
print("name                 ", deformer.name)
```

### Guards worth knowing

```python
for bad, label in (
    (np.eye(4), "a bare (4, 4) pose"),
    (np.stack([np.eye(4)]), "a 1-joint pose against 2 joints"),
):
    try:
        deformer.apply(bad)
    except ValueError as e:
        print(f"{label}: {e}")

try:
    deformer.deform(np.zeros((99, 3)), skin_matrices)   # wrong point count
except ValueError as e:
    print("as expected:", e)

negative = SkinData(weights=np.array([[0.5, -0.2]]), influences=["root", "tip"])
try:
    SkinDeformData(mesh=np.zeros((1, 3)), skin=negative,
                   inverse_bind_matrices=identity_ibm).bind()
except ValueError as e:
    print("as expected:", e)
```

---

## `WrapData` — RBF wrap

Scattered-data interpolation: move a small set of source control points to a
set of destinations, and carry any other geometry along. The classic cage
deform / shrink-wrap workflow.

Unlike the other four this one is not bind/apply — it is
`set_source` / `set_target` / `deform`, and the solve is lazy.

### End to end

```python
cage  = grid(4, height=0.4)   # 16 control points
dense = grid(9, height=0.15)  # the geometry that rides along

wrap  = WrapData(kernel="thin_plate_spline", name="wrap1")
wrap.set_source(cage)                       # MeshData or (N, 3)

moved = cage.points.copy()
moved[:, 2] += 0.3 * moved[:, 0]            # shear the cage in Z
wrap.set_target(moved)

result = wrap.deform(dense)                 # MeshData or (N, 3) -> (N, 3)
print(result.shape, np.abs(result - dense.points).max() > 0.1)
```

An unmoved cage is the identity, and a thin-plate spline carries the affine
term exactly:

```python
rest = WrapData()
rest.set_source(cage)
rest.set_target(cage.points.copy())
assert np.allclose(rest.deform(dense.points), dense.points, atol=1e-9)

expected = dense.points.copy()
expected[:, 2] += 0.3 * expected[:, 0]
assert np.allclose(result, expected, atol=1e-9)
```

Both ends are required:

```python
try:
    WrapData().deform(dense.points)
except ValueError as e:
    print("as expected:", e)
```

### Kernels

Twenty-one, from `cgmath.rbf`. Names are exact strings.

```python
from cgmath.rbf._kernels import get_kernel, list_kernels
print(list_kernels())
```

| Family | Names | Radius arg |
|---|---|---|
| Polyharmonic | `linear`, `cubic`, `quintic`, `thin_plate_spline`, `polyharmonic` | — |
| Infinitely smooth | `gaussian`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic` | — |
| Matern / misc | `matern_12`, `matern_32`, `matern_52`, `cauchy`, `log` | — |
| Compactly supported | `wendland_c0/c2/c4/c6`, `wu_c2`, `wu_c4`, `bump` | `set_radius()` |

The kernel only shows up on non-affine cage motion — every kernel here carries
the affine part exactly, through the polynomial block of the system.

```python
bumped = cage.points.copy()
bumped[5, 2] += 0.5                         # poke one control point

for name in ("thin_plate_spline", "gaussian", "multiquadric", "linear"):
    w = WrapData(kernel=name)
    w.set_source(cage)
    w.set_target(bumped)
    print(f"{name:20s} {np.abs(w.deform(dense.points) - dense.points).max():.4f}")
```

Swap the kernel on a live object:

```python
wrap.set_kernel("cubic")
print(wrap.kernel_name)
```

### Compact kernels and `set_radius()`

Only the compactly supported kernels read the radius — points further apart
than it get zero weight. Too small and the solve has nothing to work with.

```python
compact = WrapData(kernel="wendland_c2")
compact.set_source(cage)
compact.set_target(moved)
compact.set_radius(2.0)                     # world units
print(compact.radius, compact.deform(dense.points).shape)
```

### `set_geodesic_radius()` — keep influence on the surface

Restricts the source-to-source distance matrix to pairs reachable within a
geodesic surface distance, so a wrap does not leak across a gap that is close
in space but far along the mesh. Needs a `MeshData` source (that is where the
connectivity comes from).

```python
geo = WrapData(kernel="wendland_c2")
geo.set_source(cage)          # MeshData -> connectivity captured
geo.set_target(moved)
geo.set_radius(1.0)
geo.set_geodesic_radius(1.5)  # must be >= radius
print(geo.geodesic_radius, geo.deform(dense.points).shape)
```

Two warnings guard the common misconfigurations: a non-compact kernel with a
geodesic radius (ill-conditioned), and `radius > geodesic_radius` (the topology
lookup would drop neighbours that still have non-zero weight).

### Cheap re-solves

The pseudo-inverse depends on the source, the kernel and the radii.
`set_target()` moves only the right-hand side, so re-solving is a matmul rather
than another pseudo-inverse — the difference between milliseconds and seconds
on a few thousand control points.

```python
wrap.deform(dense.points)                   # settle the current configuration
pinv = wrap._system_pinv

for scale in (0.1, 0.2, 0.3):
    shifted = cage.points.copy()
    shifted[:, 2] += scale
    wrap.set_target(shifted)
    wrap.deform(dense.points)
print("pseudo-inverse reused:", wrap._system_pinv is pinv)
```

Everything else — `set_source`, `set_kernel`, `set_radius`,
`set_geodesic_radius` — rebuilds it.

### Inspect

```python
print("kernel_name    ", wrap.kernel_name)
print("name           ", wrap.name)
print("src_points     ", wrap.src_points.shape)
print("dst_points     ", wrap.dst_points.shape)
print("radius         ", wrap.radius, " geodesic_radius", wrap.geodesic_radius)
print("conn_matrix    ", wrap.conn_matrix.shape)      # None for array sources
print("conn_distances ", wrap.conn_distances.shape)
```

### Low-level wrap helpers

```python
from cgmath.geometry.deform.wrap import (
    cdist_euclidean,                  # numba pairwise distances, fast under ~1e5 pts
    compute_neighbor_distances,       # edge lengths for a neighbour matrix
    compute_topological_neighborhood, # Dijkstra over mesh connectivity
)

d = cdist_euclidean(dense.points[:4], cage.points)
print(d.shape)

conn = cage.get_face_vertex_neighbors()
print(compute_neighbor_distances(conn, cage.points).shape)

kernel = get_kernel("wendland_c2")
print(np.round(kernel(np.array([[0.0, 0.5, 2.0]]), r=1.0), 3))
```

---

## Persistence

Everything derives from `Data`, so `.info()`, `.to_dict()`, `.copy()`,
`.to_bytes()` / `.from_bytes()` and `.save()` / `.load()` exist on all five.
Whether a restored object is *usable* depends on whether its state lives in
dataclass fields.

| Class | Round-trips ready to use | Why |
|---|---|---|
| `SkinDeformData` | **yes** — deforms after `copy()` without its `SkinData` | weights, joints and matrices are all fields |
| `WrapData` | **yes** — the solved system is a field | every knob is a field |
| `FFDData` | geometry only | `outside`, `falloff_radius`, `local_influence` are not fields |
| `DeltaMushData` | geometry only | `smooth_iterations`, `method`, `weight` are not fields |
| `PatchRelaxData` | geometry only | `iterations`, `alpha`, `step_size`, `mask` scale are not fields |

```python
import pickle, tempfile, os

deformer.bind()
clone = deformer.copy()
print(clone.valid, clone._skin is None, clone.method)
assert np.allclose(clone.apply(world, bar).points, posed.points, atol=1e-12)

assert SkinDeformData.from_bytes(deformer.to_bytes()).method == deformer.method
assert pickle.loads(pickle.dumps(deformer)).valid

path = os.path.join(tempfile.mkdtemp(), "bar.json")
deformer.save(path)                          # .json | .npz | .pkl by extension
print(SkinDeformData.load(path).joints)
```

The caveat, made concrete — rebuild these operators rather than reloading them:

```python
ffd_clone = ffd.copy()
print("FFD copy loses tuning:", ffd_clone.outside, ffd_clone.local_influence)

mush_clone = mush.copy()
print("mush copy loses tuning:", mush_clone.method, mush_clone.smooth_iterations)
```

`WrapData` does survive intact:

```python
wrap_clone = wrap.copy()
assert np.allclose(wrap_clone.deform(dense.points), wrap.deform(dense.points))
print(wrap_clone.kernel_name, wrap_clone.src_points.shape)
```

Class docstrings are available at runtime:

```python
print(FFDData.info().splitlines()[0])
print(DeltaMushData.info().splitlines()[0])
```

---

## Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `FFDData.from_mesh(mesh, ...)` raises `ValueError` | it wants the **lattice**, not the deformed mesh | `create_lattice(...)` first, then `bind(mesh)` |
| `RuntimeError: call bind() before update()` | `FFDData` never bound | `ffd.bind(target)` |
| Delta Mush `method` silently becomes `"DDM"` | `"TBN"` needs face winding; you passed a point cloud | pass a `MeshData`, or accept DDM |
| `ValueError: neighbors must be supplied` | `DeltaMushData` on a raw array | `mesh.get_edge_vertex_neighbors()` |
| `PatchRelaxData` rejects your input | it needs `points` + `indices` + `counts` | pass the `MeshData`, not `mesh.points` |
| Skin result looks plausible but wrong | matrices multiplied as `world @ inverse_bind` | row-vector order is `inverse_bind @ world` |
| Skinned mesh drifts over an animation | last frame's result fed back in | always deform the **bind** mesh |
| `ValueError: negative skin weights` | compaction would silently drop them | `skin.prune()` / `skin.normalize()` first |
| Wrap output is all `nan` / wild | compact kernel with a too-small radius | raise `set_radius()`, or use `thin_plate_spline` |
| `UserWarning` about ill-conditioning | `geodesic_radius` with a non-compact kernel | use `wendland_c2` and friends |
| Reloaded FFD / mush / relax behaves oddly | tuning is not serialized | rebuild the operator from its rest geometry |

---

See also: [`README.md`](README.md) for the concepts, and the package-level
[`../../CHEATSHEET.md`](../../CHEATSHEET.md).
