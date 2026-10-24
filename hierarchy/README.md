# `cgmath.hierarchy` — the scene graph

Transform nodes that follow Maya's SRT conventions, the containers that
hold them, and the animation clip that plays them back. Four public
types, no DCC, numpy all the way down.

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

**Identity is the uuid, not the name.** `parent_uuid` holds the
**parent's uuid**, or `None` for a root, and nothing else: parent by name
with `set_parent("hip")`, which keeps the uuid, and read the name back as
`parent_name`. `HierarchyData.append` mints a uuid for any node that
arrives without one. Names are for humans, `match()` and lookup — they
may repeat, and `unique_name` tells you when they do. A node joining a
hierarchy without its parent becomes a root there. `parent_node`, the
name before 2.0.0, went in 3.0.0; a file that names it still loads.

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

**`world_matrix` is a property on every type** — `node.world_matrix`,
`rig.world_matrix`. It is cached per node, and any channel write clears
the cache of that node and its whole branch. Maya's parent and inverse
matrices are properties too, read only and worked out at each read:
`parent_matrix`, `inverse_matrix`, `world_inverse_matrix` and
`parent_inverse_matrix`.

**Maya's short names work too.** `node.tx` is `node.translate_x`,
`rig.wm` is `rig.world_matrix`, `clip.frames.r` is `clip.frames.rotate`:
each short name reads and writes its long name itself, its conversions,
errors and clip frames included, and a file keeps the long names. They
are the names Maya 2025 gives a joint's attributes:

| Short | Long |
|---|---|
| `t` `tx` `ty` `tz` | `translate` `translate_x` `translate_y` `translate_z` |
| `r` `rx` `ry` `rz` | `rotate` `rotate_x` `rotate_y` `rotate_z` |
| `s` `sx` `sy` `sz` | `scale` `scale_x` `scale_y` `scale_z` |
| `jo` `jox` `joy` `joz` | `joint_orient` `joint_orient_x` `joint_orient_y` `joint_orient_z` |
| `ra` `rax` `ray` `raz` | `rotate_axis` `rotate_axis_x` `rotate_axis_y` `rotate_axis_z` |
| `ro` `ssc` `v` `radi` `ds` | `rotate_order` `segment_scale_compensate` `visibility` `radius` `draw_style` |
| `m` `wm` | `matrix` `world_matrix` |
| `pm` `im` `wim` `pim` | `parent_matrix` `inverse_matrix` `world_inverse_matrix` `parent_inverse_matrix`, read only |

Maya's `is`, `inverseScale`, is a Python keyword and has none.
`parent_scale_inverse` is the parent's scale, whatever the parent: what
Maya's `inverseScale` holds where Maya wires it from the parent joint.
`cmds.joint` wires it under a joint, and so does an absolute
`cmds.parent` that leaves the joint right under one; under a joint, one
that does not puts a transform between them, unwired. Under a joint, a
joint made with `cmds.createNode` and `parent=` is unwired, holding
(1, 1, 1), and one moved there with `cmds.parent` and `relative=True`
holds what it held, unwired or still wired from the joint it left. Under
a transform Maya never wires it from that transform: a joint drawn there
holds (1, 1, 1), and one moved there with `cmds.parent` holds the scale
of the parent joint it left, unwired, under that transform or under one
Maya inserts below it, or with `relative=True` still wired from that
joint. So a joint with `segment_scale_compensate` on that Maya did not
wire from its parent joint can divide by a scale Maya's does not.
`add_user_attribute` refuses a short name; a file's user attribute that
carries one loads, hidden, with one warning a name.

**A clip's nodes are its playback head.** Each framed channel is bound
to one row of a dense `(F, N, 3)` block. Scrubbing rebinds; reading a
channel reads the loaded frame; writing one writes it. Only `scale`,
`rotate`, `translate`, `rotate_axis` and `joint_orient` are framed —
names, parenting, `rotate_order` and the display flags are structural
and shared by every frame.

**Rig edits cover every frame.** `swapaxes`, the orient helpers,
`set_parent` in world space, a joint becoming another type and
`set_rotate_order` rebuild channels out of matrices. On a clip they solve
every frame at once, and every euler curve they rebuild goes through
`euler_filter`, frame 0 kept, so the curves come out smooth rather than
stepping by 180 and 360 degrees.

**`a - b` is a retargeting operator.** The delta holds a per-joint frame
offset fixed to the bone, not a local rotation difference, so it re-poses
`b` at *any* pose — that is what lets a delta carry animation between
rigs whose joint axes or proportions differ. `(a - b) + b == a`.

---

## When to reach for it

- You have joints, parents and world matrices, and you do not want a DCC
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

`hierarchy.py` also exposes `generate_uuid()` / `validate_uuid()` and the
`MAYA_ATTRIBUTE_MAP` used by `to_attributes()`. The matrix and quaternion
math it is built on comes from `transforms` and is not re-exported: import
`euler_to_matrix`, `quaternion_slerp` and friends from there.

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

### Changing rotate order

Assigning `rotate_order` changes the order alone, as Maya's `setAttr`
does, so the joint turns. `set_rotate_order` keeps the pose, as Maya's
`xform -rotateOrder <order> -preserve` does, and picks the euler solution
Maya picks; on a clip every frame, the curves filtered.

```python
before = rig.world_matrix
rig.set_rotate_order("zyx")            # or one order per node
np.allclose(rig.world_matrix, before)  # True

clip.set_rotate_order([0, 1, 2])       # every frame of the clip
```

---

## Where to go next

| You want to... | Read |
|---|---|
| Every class, property and method, with runnable examples | [`CHEATSHEET.md`](CHEATSHEET.md) |
| The rest of the library | [`../README.md`](../README.md) |
| The free transform functions these types are built on | [`cgmath.transforms`](../transforms/CHEATSHEET.md) |
