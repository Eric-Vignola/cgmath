# `cgmath.hierarchy` — the scene graph

Maya-shaped transform nodes, the containers that hold them, and the
animation clip that plays them back. Four public types, no DCC, numpy
all the way down.

```python
from cgmath.hierarchy import (
    ClipData,
    HierarchyData,
    TransformData,
    TransformList,
)
```

| Type | Role |
|---|---|
| `TransformData` | one node: name, uuid, parent, SRT, joint channels, matrices |
| `TransformList` | a **non-owning** selection of nodes — vectorized reads and writes |
| `HierarchyData` | the **owning** container — a rig; a `TransformList` that claims its nodes |
| `ClipData` | a `HierarchyData` holding `F` poses instead of one |

Each is a subclass of the one above it, so a clip answers every rig
question, and a rig answers every selection question.

---

## Concepts you need before using it

**Identity is the uuid, not the name.** `parent_node` holds the
**parent's uuid**. `HierarchyData.append` mints one for any node that
arrives without it. Names are for humans, `match()` and lookup — they
may repeat, and `unique_name` tells you when they do.

**Ownership is exclusive.** A node's `_hierarchy` back-pointer names the
one `HierarchyData` that holds it. `append` / `insert` claim it (pulling
it out of any previous owner), `pop` releases it. A `TransformList` —
a slice, a mask, a `match()` — claims nothing, which is exactly why a
node reached through a selection still answers `get_parent()`,
`world_matrix` and `index` about the whole rig.

**Row vectors, Maya composition.** Translation lives at `M[3, :3]`, and

```
matrix = S · RO · R · JO · IS · T          # local
world  = matrix · parent_world             # accumulated down the chain
```

`RO` is `rotate_axis`, `JO` is `joint_orient`, `IS` is the parent's
inverse scale (identity unless `segment_scale_compensate` is on).
`rotate_order` picks which of the six euler orders builds `R`.

**`world_matrix` is a property, on every type** — `node.world_matrix`,
`rig.world_matrix`, never a call, never parameterised by name. It is
cached per node, and any channel write clears the cache of that node and
its whole branch.

**A clip's nodes are its playback head.** Each framed channel is bound
to one row of a dense `(F, N, 3)` block. Scrubbing rebinds; reading a
channel reads the loaded frame; writing one writes it. Only `scale`,
`rotate`, `translate`, `rotate_axis` and `joint_orient` are framed —
names, parenting, `rotate_order` and the display flags are structural
and shared by every frame.

**`a - b` is a retargeting operator.** The delta holds a per-joint frame
offset fixed to the bone, not a local rotation difference, so it re-poses
`b` at *any* pose — that is what lets a delta carry animation between
rigs whose joint axes or proportions differ. `(a - b) + b == a`.

---

## When to reach for it

- You have joints, parents and world matrices, and you do not want Maya
  running to evaluate them.
- You need to read a rig out of an fbx or glb, retime or retarget it,
  and write it back.
- You need a whole animation as one `(F, N, 3)` array to feed numpy,
  a solver, or a training set — without giving up node semantics.
- You need to hand a DCC a dict of Maya-named attributes
  (`to_attributes()`).

Reach elsewhere when: you want the free functions on raw arrays rather
than nodes (`transforms` — `euler_to_matrix`,
`quaternion_slerp`, `matrix_local`, …), or you want geometry rather than
a skeleton (`cgmath.geometry`).

---

## Files

```
hierarchy/
├── __init__.py     re-exports the public surface
└── hierarchy.py    all of them, plus the fbx / glb readers and the
                    batched clip evaluator
```

`hierarchy.py` also exposes `generate_uuid()` / `validate_uuid()`, the
`MAYA_ATTRIBUTE_MAP` used by `to_attributes()`, and re-exports the
transform math it is built on (`euler_to_matrix`, `matrix_multiply`,
`matrix_to_quaternion`, `q_slerp`, …) from `transforms`.

Optional dependencies are lazy: `pygltflib` for glb, the Autodesk `fbx`
sdk for fbx. Without them only those loaders raise `ImportError`; the
rest of the module works.

---

## Quick taste

### Build a rig and read world space

```python
import numpy as np
from cgmath.hierarchy import HierarchyData, TransformData

rig = HierarchyData()
for name in ("root", "hip", "knee"):
    rig.append(TransformData(name, node_type="joint"))
rig["hip"].set_parent("root", world_space=False)
rig["knee"].set_parent("hip", world_space=False)
rig["hip"].translate  = (0.0, 10.0, 0.0)
rig["knee"].translate = (0.0, 10.0, 0.0)

rig.world_matrix[:, 3, :3]        # (3, 3) positions
# [[0, 0, 0], [0, 10, 0], [0, 20, 0]]

rig["hip"].rotate_z = 90.0        # the branch follows
np.round(rig["knee"].world_matrix[3, :3], 6)
# [-10., 10., 0.]
```

### Animate it, then read every frame at once

```python
from cgmath.hierarchy import ClipData

clip                         = ClipData(rig, frames=24, start_frame=1001, fps=24.0)
clip.frames["knee"].rotate_z = np.linspace(0.0, 90.0, 24).reshape(24, 1)

clip.frames.world_matrix.shape    # (24, 3, 4, 4) -- batched, all frames
clip.frame = 12                   # scrub; the nodes hold frame 12
clip["knee"].rotate_z             # 46.9565...
```

### Retarget a pose onto a different skeleton

```python
tall           = HierarchyData([node.copy() for node in rig])
tall.translate = tall.translate * 1.3       # longer bones, same topology

posed                 = HierarchyData([node.copy() for node in rig])
posed["hip"].rotate_x = 25.0

result = posed.add_delta(tall.get_delta(rig, translate=False), translate=False)
result.name                                 # ['root', 'hip', 'knee']
```

---

## Where to go next

| You want to... | Read |
|---|---|
| Every class, property and method, with runnable examples | [`CHEATSHEET.md`](CHEATSHEET.md) |
| The rest of the library | [`../../README.md`](../README.md) |
| The free transform functions these types are built on | [`../../CHEATSHEET.md`](../CHEATSHEET.md) |
