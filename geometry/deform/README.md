# `cgmath.geometry.deform` — the deformers

Five numpy-native deformation operators, no DCC required: a free-form
deformation lattice, Delta Mush, patch-based relaxation, skeletal skinning, and
an RBF wrap. Numba-accelerated kernels underneath, plain `MeshData` in and out.

```python
from cgmath.geometry.deform import (
    DeformMethod, DeltaMushData, FFDData, PatchRelaxData, SkinDeformData, WrapData,
)
```

---

## Which one do I want?

| Deformer | Driven by | Reach for it when |
|---|---|---|
| `FFDData` | a 3-D lattice of control points | sculpt broad shape changes — squash, taper, bend, sweep |
| `DeltaMushData` | the rest mesh itself | a pose has skinning wobble, twist collapse or candy-wrap |
| `PatchRelaxData` | a baseline mesh | edge flow has drifted, clumped or folded and must come back |
| `SkinDeformData` | a skeleton + skin weights | pose a bind mesh from joints (LBS or dual quaternion) |
| `WrapData` | a sparse cage / correspondence | carry dense geometry along with a small set of moved points |

---

## The shape they share

Four of the five are **bind / apply** operators:

1. **Construct** with the rest geometry — `FFDData(lattice)`,
   `DeltaMushData(rest_mesh)`, `PatchRelaxData(base_mesh)`,
   `SkinDeformData(mesh=..., skin=...)`.
2. **Bind** — the expensive precompute (a Newton solve, a Laplacian smooth,
   decal maps, a weight compaction). Everything except `FFDData` auto-binds on
   the first evaluation, so calling `bind()` is only about *when* you pay.
3. **Evaluate**, as often as you like — `update()`, `apply()`, `deform()`.

`WrapData` is the odd one out: `set_source()` / `set_target()` / `deform()`,
with the solve done lazily and cached.

| Class | Cache step | Evaluate | Auto-binds? |
|---|---|---|---|
| `FFDData` | `bind(target)` | `update(lattice)`, `compute(lattice)` | no — raises |
| `DeltaMushData` | `bind()` | `apply(deformed)` | yes |
| `PatchRelaxData` | `bind()` | `apply(deformed)`, `relax(points)` | yes |
| `SkinDeformData` | `bind()` | `apply(pose, target)`, `deform(points, matrices)` | yes |
| `WrapData` | first `deform()` | `deform(points)` | yes |

---

## Conventions

- **Mesh in, mesh out.** Hand an evaluator a `MeshData` and you get a *copy*
  with new points; hand it an `(N, 3)` array and you get an array. The input is
  never mutated.
- **Row-vector matrices**, like the rest of the package: `p' = p @ M`,
  translation at `M[3, :3]`. `SkinDeformData` builds its skin matrices as
  `inverse_bind @ world_posed` — the reverse order produces plausible-looking
  garbage rather than an error.
- **Maya conventions** where there is a Maya equivalent: `FFDData.divisions`
  and `local_influence` count *control points*, not cells; `DeltaMushData`
  reproduces Maya's `smoothingIterations` off-by-one.
- **Rest geometry is a constructor argument, not a field.** `Data` subclasses
  cannot hold another `Data` as a field, so meshes, skins and rigs are passed
  in and only arrays and names persist to disk.
- **Deform the bind pose every frame.** Feeding the previous frame's result
  back into an operator compounds the deformation.
- Cheap knobs (`weight`, `alpha`, `iterations`, `local_influence`, `outside`)
  can be changed between evaluations; structural ones (`method`,
  `smooth_iterations`, `smooth_step_size`, `pin_borders` on Delta Mush)
  invalidate the bind and trigger a rebind on the next call.

---

## Files

| File | Contents |
|---|---|
| `__init__.py` | re-exports all five classes plus `DeformMethod` |
| `ffd.py` | `FFDData` — lattice construction, cell assignment, Bernstein evaluation |
| `delta_mush.py` | `DeltaMushData` — TBN / Procrustes / DDM frame reconstruction |
| `patch_relax.py` | `PatchRelaxData` — decal maps, span weights, Jacobi relaxation |
| `skin_deform.py` | `SkinDeformData`, `DeformMethod` — LBS and dual quaternion skinning |
| `wrap.py` | `WrapData` — RBF interpolation over a control-point cage |

The numba kernels live one level up in `cgmath.geometry.utils._numba`
(`_delta_mush`, `_patch_relax`, `_skin_deform`, `_wrap`), with public Python
wrappers in `cgmath.geometry.utils.main`. The RBF kernels come from
`cgmath.rbf`.

`cgmath.geometry.ffd`, `cgmath.geometry.delta_mush` and
`cgmath.geometry.patch_relax` are deprecation shims for the old import paths —
they warn and re-export. Import from `cgmath.geometry.deform`.

---

## Quick taste

No mesh assets ship with the package, so the examples build their own — a
5 × 5 quad grid spanning `x, y` in `[0, 1]`.

```python
import numpy as np
from cgmath.geometry import MeshData

n       = 5
t       = np.linspace(0.0, 1.0, n)
indices = []
for j in range(n - 1):
    for i in range(n - 1):
        a = j * n + i
        indices += [a, a + 1, a + n + 1, a + n]

mesh = MeshData(
    name    = "grid",
    points  = np.array([[x, y, 0.0] for y in t for x in t]),
    indices = np.asarray(indices, dtype=np.int32),
    counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
)
```

### Shear a mesh with an FFD lattice

`from_mesh` takes the **lattice**, not the mesh being deformed — build the
lattice around the mesh's bounding box first.

```python
from cgmath.geometry.deform import FFDData

lattice_mesh = FFDData.create_lattice(
    (3, 3, 3),
    bbox_min = mesh.points.min(axis=0) - 0.05,
    bbox_max = mesh.points.max(axis=0) + 0.05,
)

ffd = FFDData.from_mesh(lattice_mesh, divisions=(3, 3, 3))
ffd.bind(mesh)

posed = ffd.lattice.copy()          # (lx, ly, lz, 3)
posed[:, -1, :, 0] += 0.5           # push the max-Y control plane in +X
sheared = ffd.update(posed)         # a new MeshData
print(np.abs(sheared.points - mesh.points).max())
```

### Clean a wobbly pose

```python
from cgmath.geometry.deform import DeltaMushData

mush    = DeltaMushData(mesh, smooth_iterations=20, pin_borders=True)
cleaned = mush.apply(sheared)       # binds on the first call
print(type(cleaned).__name__)
```

### Pose a skinned mesh

```python
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.skin_weights import SkinData
from cgmath.hierarchy import HierarchyData, TransformData

x    = mesh.points[:, 0]
skin = SkinData(weights=np.stack([1.0 - x, x], axis=1), influences=["root", "tip"])

bind_rig = HierarchyData([
    TransformData(name="root"),
    TransformData(name="tip", parent_node="root", translate=(1.0, 0.0, 0.0)),
])
posed_rig = HierarchyData([
    TransformData(name="root"),
    TransformData(name="tip", parent_node="root",
                  translate=(1.0, 0.0, 0.0), rotate=(0.0, 0.0, 45.0)),
])

deformer = SkinDeformData(mesh=mesh, skin=skin, bind_rig=bind_rig)
bent     = deformer.apply(posed_rig, mesh)
print(np.abs(bent.points - mesh.points).max())
```

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste an example of every class, method and property | [`CHEATSHEET.md`](CHEATSHEET.md) |
| See the rest of the library | [`../../README.md`](../../README.md) |
| Look up mesh topology helpers these operators need | `cgmath.geometry.mesh` — `get_edge_vertex_neighbors`, `get_border_vertices`, `get_face_vertex_neighbors` |
