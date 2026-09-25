# `cgmath.hierarchy` Cheatsheet

Copy-paste recipes for the scene-graph types: `TransformData`,
`TransformList`, `HierarchyData`, `ClipData`.

Every `python` block below runs in order, sharing one namespace. For the
overview see [`README.md`](README.md); for the rest of the library see
[`../../CHEATSHEET.md`](../CHEATSHEET.md).

---

## Contents

| Topic | |
|---|---|
| [Setup](#setup) | imports, the fixtures every block reuses |
| [`TransformData` — construction](#transformdata--construction) | the 15 constructor arguments |
| [SRT channels](#srt-channels) | `scale` / `rotate` / `translate` and per-axis accessors |
| [Joint channels](#joint-channels) | `rotate_order`, `rotate_axis`, `joint_orient`, `segment_scale_compensate` |
| [Display + user channels](#display--user-channels) | `visibility`, `radius`, `draw_style`, `user_defined_attributes` |
| [Matrices](#matrices) | `matrix`, `world_matrix`, the six factor matrices, `quaternion` |
| [Cache invalidation](#cache-invalidation) | when `world_matrix` is recomputed |
| [Node queries](#node-queries) | parent / children / branch / root / index |
| [Node edits](#node-edits) | `set_parent`, `match_*`, `swapaxes`, orient helpers, prefix / suffix, namespaces, user attributes |
| [`TransformList` — views](#transformlist--views) | slicing, fancy indexing, `match` |
| [Vectorized channels](#vectorized-channels) | read / write a whole selection |
| [Vectorized queries and edits](#vectorized-queries-and-edits) | the list forms of every node method |
| [`HierarchyData` — ownership](#hierarchydata--ownership) | `append` / `insert` / `pop` / `extend` |
| [Pose algebra](#pose-algebra) | `get_delta` / `add_delta`, `-` / `+` |
| [`ClipData` — construction](#clipdata--construction) | frames, timebase |
| [Scrubbing](#scrubbing) | `clip.frame`, the nodes as a playback head |
| [The frame axis](#the-frame-axis) | `clip.frames` blocks, components, scoping, cuts, pastes |
| [Batched evaluation](#batched-evaluation) | `frames.matrix`, `frames.world_matrix` |
| [Clip deltas](#clip-deltas) | retargeting a whole clip |
| [Clip guard rails](#clip-guard-rails) | the rig edits an animated clip refuses |
| [Importers and exporters](#importers-and-exporters) | fbx, glb, auto-detect |
| [Serialization](#serialization) | json / npz / pkl, dict, bytes |
| [Gotchas](#gotchas) | the traps, in one table |

---

## Setup

No rig assets ship with this package, so build one inline. `set_parent`
is used rather than a raw `parent_node` string, because parenting is
resolved through the parent's **uuid** (see [Gotchas](#gotchas)).

```python
import os
import tempfile

import numpy as np
from cgmath.hierarchy import (
    ClipData,
    HierarchyData,
    TransformData,
    TransformList,
)

WORKDIR = tempfile.mkdtemp(prefix="hierarchy_cheatsheet_")


def make_chain(names=("root", "hip", "knee"), step=(0.0, 10.0, 0.0)):
    """a parent -> child joint chain, each joint `step` from its parent"""
    rig = HierarchyData()
    for i, name in enumerate(names):
        rig.append(TransformData(name, node_type="joint"))
        if i:
            rig[name].set_parent(names[i - 1], world_space=False)
            rig[name].translate = step
    return rig


rig = make_chain()
assert rig.name == ["root", "hip", "knee"]
assert np.allclose(rig.world_matrix[:, 3, :3], [[0, 0, 0], [0, 10, 0], [0, 20, 0]])
```

---

## `TransformData` — construction

One Maya-style transform node. `name` is the only required argument.

```python
node = TransformData(
    "hero",
    node_type    = "joint",
    translate    = (0.0, 1.0, 0.0),
    rotate       = (0.0, 45.0, 0.0),
    scale        = (1.0, 1.0, 1.0),
    rotate_order = 0,
)
node.name           # 'hero'
node.node_type      # 'joint'
node.uuid           # None until a HierarchyData claims it
str(node)           # 'hero'
```

| Argument | Default | Meaning |
|---|---|---|
| `name` | *required* | node name; drives `match()` and name lookup |
| `node_type` | `"transform"` | `"transform"`, `"joint"`, `"locator"`; the orient helpers only act on `"joint"` |
| `uuid` | `None` | identity; `HierarchyData.append` assigns one when missing |
| `parent_node` | `None` | the **parent's uuid** (see [Gotchas](#gotchas)) |
| `scale` | `(1, 1, 1)` | |
| `rotate` | `(0, 0, 0)` | degrees |
| `translate` | `(0, 0, 0)` | |
| `rotate_order` | `0` | `0=xyz 1=yzx 2=zxy 3=xzy 4=yxz 5=zyx` |
| `rotate_axis` | `(0, 0, 0)` | degrees; Maya `rotateAxis` |
| `joint_orient` | `(0, 0, 0)` | degrees; Maya `jointOrient` |
| `visibility` | `True` | |
| `segment_scale_compensate` | `True` | divide out the parent's scale |
| `radius` | `1.0` | joint display radius |
| `draw_style` | `0` | Maya `drawStyle` |
| `user_defined_attributes` | `{}` | extra attributes carried through `save_fbx` |

---

## SRT channels

Every vector channel is a read/write property returning a fresh `(3,)`
array, plus three scalar per-axis accessors.

```python
node.translate                    # array([0., 1., 0.])
node.translate   = (1.0, 2.0, 3.0)
node.translate_y = 5.0
node.translate                    # array([1., 5., 3.])

node.rotate  = (0.0, 90.0, 0.0)    # degrees
node.scale_z = 2.0
node.scale                        # array([1., 1., 2.])
```

| Channel | Vector | Per-axis |
|---|---|---|
| scale | `scale` | `scale_x` `scale_y` `scale_z` |
| rotate | `rotate` | `rotate_x` `rotate_y` `rotate_z` |
| translate | `translate` | `translate_x` `translate_y` `translate_z` |
| rotate axis | `rotate_axis` | `rotate_axis_x` `rotate_axis_y` `rotate_axis_z` |
| joint orient | `joint_orient` | `joint_orient_x` `joint_orient_y` `joint_orient_z` |

The getter hands back a copy, so mutating it changes nothing — assign
the channel back.

```python
r    = node.rotate
r[0] = 15.0
assert np.allclose(node.rotate, [0.0, 90.0, 0.0])   # unchanged
node.rotate = r                                     # now it lands
assert np.allclose(node.rotate, [15.0, 90.0, 0.0])
```

---

## Joint channels

```python
joint = TransformData("elbow", node_type="joint")

joint.rotate_order = 5             # zyx
joint.rotate_axes                  # 'zyx' -- the order as a string

joint.rotate_axis              = (0.0, 0.0, 10.0)
joint.joint_orient             = (0.0, 30.0, 0.0)
joint.segment_scale_compensate = False
```

| Property | Type | Notes |
|---|---|---|
| `rotate_order` | `int` 0..5 | read / write |
| `rotate_axes` | `str` | read only; `"xyz"` … `"zyx"` |
| `rotate_axis` | `(3,)` degrees | read / write; applied **before** `rotate` |
| `joint_orient` | `(3,)` degrees | read / write; applied **after** `rotate` |
| `segment_scale_compensate` | `bool` | read / write; when `True`, the parent's scale is divided out |
| `parent_scale_inverse` | `(3,)` | read only; the parent's scale, `(1, 1, 1)` at a root |

---

## Display + user channels

```python
joint.visibility              = False
joint.radius                  = 2.5
joint.draw_style              = 2
joint.user_defined_attributes = {"stretch": {"value": 1.0, "keyable": True}}

# Maya-named attribute dump, ready to push at a DCC
joint.to_attributes()
# {'scale': [...], 'rotate': [...], 'translate': [...], 'rotateOrder': 5,
#  'rotateAxis': [...], 'jointOrient': [...], 'segmentScaleCompensate': False,
#  'radius': 2.5, 'visibility': False, 'drawStyle': 2}
assert joint.to_attributes()["jointOrient"] == [0.0, 30.0, 0.0]
```

Pass your own mapping to rename the keys.

```python
joint.to_attributes({"_translate": "t", "_rotate": "r"})
# {'r': [0.0, 0.0, 0.0], 't': [0.0, 0.0, 0.0]}
```

---

## Matrices

Row-vector convention: translation lives at `M[3, :3]`, and a local
matrix composes as

```
matrix = S · RO · R · JO · IS · T
world  = matrix · parent_world
```

```python
n = make_chain()["hip"]
n.matrix                     # (4, 4) local
n.world_matrix               # (4, 4) local · parent's world
n.get_parent_matrix()        # (4, 4) the parent's world, eye(4) at a root
```

| Property | Read | Write | Contents |
|---|---|---|---|
| `scale_matrix` | yes | — | `S` |
| `rotate_axis_matrix` | yes | — | `RO` |
| `rotate_matrix` | yes | — | `R`, built with `rotate_order` |
| `joint_orient_matrix` | yes | — | `JO` |
| `parent_scale_inverse_matrix` | yes | — | `IS`, identity when compensation is off |
| `translate_matrix` | yes | — | `T` |
| `matrix` | yes | yes | the local product above |
| `world_matrix` | yes | yes | `matrix · parent_world` |
| `quaternion` | yes | yes | `RO · R · JO` as `(i, j, k, w)` |

`world_matrix` and `matrix` are **properties**, not calls — on both
`TransformData` and `TransformList`.

```python
rig           = make_chain()
target        = np.eye(4)
target[3, :3] = (5.0, 5.0, 5.0)

rig["knee"].world_matrix = target                  # solves the local for you
assert np.allclose(rig["knee"].world_matrix[3, :3], [5.0, 5.0, 5.0])
```

Writing `matrix` / `world_matrix` decomposes into scale, rotate and
translate, and resets `rotate_axis` to zero. Writing `quaternion` solves
only `rotate`, keeping `rotate_axis` and `joint_orient`.

```python
rig["hip"].quaternion = [0.0, 0.0, 0.0, 1.0]       # identity orientation
assert np.allclose(rig["hip"].rotate, 0.0, atol=1e-9)
```

---

## Cache invalidation

`world_matrix` is cached per node. Any channel write clears the cache of
that node **and its whole branch**, so reads downstream stay correct.

```python
rig = make_chain()
assert np.allclose(rig["knee"].world_matrix[3, :3], [0.0, 20.0, 0.0])

rig["hip"].translate = (0.0, 4.0, 0.0)             # a parent moves
assert np.allclose(rig["knee"].world_matrix[3, :3], [0.0, 14.0, 0.0])
```

The branch walk is `get_children`, which matches on **uuid**. A node
whose `parent_node` holds a *name* is not seen as a child, so its cached
world matrix is not cleared — parent with `set_parent` or a uuid.

---

## Node queries

```python
rig = make_chain(("root", "hip", "knee", "ankle"))

rig["knee"].get_parent()            # TransformData(hip)
rig["knee"].get_parent().name       # 'hip'
rig["hip"].get_children().name      # ['knee']
rig["hip"].get_branch().name        # ['hip', 'knee', 'ankle']  (self + descendants)
rig["knee"].get_root().name         # ['root', 'hip', 'knee']   (path down to self)
rig["knee"].index                   # 2   -- position in the owning hierarchy
rig["knee"].unique_name             # True
rig["root"].get_parent()            # None
```

| Call | Returns |
|---|---|
| `get_parent()` | the parent `TransformData`, `None` at a root, or the raw uuid `str` when the parent is not in the hierarchy |
| `get_parent_matrix()` | `(4, 4)`; `eye(4)` when the parent is absent or unresolved |
| `get_children()` | `TransformList` of direct children |
| `get_branch()` | `TransformList` of self + every descendant |
| `get_root()` | `TransformList` of the path root → self |
| `index` | `int` position in the owning hierarchy; raises when unowned |
| `unique_name` | `bool`; raises when unowned |

`match` is fnmatch on the name.

```python
rig["knee"].match("kn*")                       # True
rig["knee"].match("hip", "knee")               # True  -- any of them
rig["knee"].match("KNEE", exact=False)         # True  -- case insensitive
```

---

## Node edits

### Reparent

```python
rig = make_chain()
rig["knee"].set_parent("root")                 # keeps the world position
assert np.allclose(rig["knee"].world_matrix[3, :3], [0.0, 20.0, 0.0])

rig = make_chain()
rig["knee"].set_parent("root", world_space=False)   # keeps the local values
assert np.allclose(rig["knee"].world_matrix[3, :3], [0.0, 10.0, 0.0])

rig["knee"].set_parent(None)                   # to the world
assert rig["knee"].get_parent() is None
```

`set_parent` also moves the node and its branch to the end of the list,
so a parent always precedes its children.

### Match another node in world space

```python
rig   = make_chain()
other = make_chain(("a", "b"), step=(3.0, 0.0, 0.0))

other["b"].match_translate(rig["knee"])                    # position only
other["b"].match_translate(rig["knee"], x=False, z=False)  # one axis at a time
other["b"].match_translate([1.0, 2.0, 3.0])                # or a raw xyz
other["b"].match_rotate(rig["knee"])                       # orientation only
other["b"].match_rotate([0.0, 45.0, 0.0])                  # euler, degrees
other["b"].match_rotate([0.0, 0.0, 0.0, 1.0])              # or a quaternion
other["b"].match_matrix(rig["knee"])                       # the whole frame
assert np.allclose(other["b"].world_matrix, rig["knee"].world_matrix)
```

### Consolidate rotation (joints only)

```python
j        = make_chain(("j0", "j1"))
j.rotate = np.array([[10.0, 20.0, 30.0]] * 2)

j.set_rotate_to_joint_orient()      # rotate -> joint_orient, rotate zeroed
assert np.allclose(j.rotate, 0.0, atol=1e-9)

j.set_joint_orient_to_rotate()      # and back
assert np.allclose(j.joint_orient, 0.0, atol=1e-9)
```

Both zero `rotate_axis` too, because the orientation chain is
`RO · R · JO` and leaving `RO` in would double-count it.

### Swap local axes

`swapaxes(a, b)` puts axis `a` where axis `b` was and vice versa, without
moving the node or its children. One axis must flip sign to keep the
frame right handed: with `negate=True` that is `axis0` (a 90° roll,
third axis untouched), otherwise it is the third axis (which makes the
swap its own inverse).

```python
j = make_chain(("j0", "j1"))
j[0].swapaxes(0, 1)                             # +X now aims where +Y did
j[0].swapaxes(0, 1)                             # undoes itself

j[0].swapaxes(0, 1, negate=True)
np.round(j[0].world_matrix[:3, :3], 6).tolist()
# [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
j[0].swapaxes(1, 0, negate=True)                # the negated inverse
```

The swap lands in `joint_orient`, so an unposed joint stays unposed.

### Rename

```python
j.add_prefix("L_")
j.add_suffix("_JNT")
assert j.name == ["L_j0_JNT", "L_j1_JNT"]
```

Both leave a namespace in front: `add_prefix("L_")` turns `VLR_RIG:SK:root`
into `VLR_RIG:SK:L_root`.

### Namespaces

A skeleton serialized out of Maya keeps its namespaces in every name.
`strip_namespace()` removes them all. Given a namespace path it removes just
that one, and what was inside it moves up one level, as when Maya deletes a
namespace.

```python
ns = make_chain(("VLR_RIG:SK:root", "VLR_RIG:SK:spine"))
ns.append(TransformData("VLR_RIG:ctrl"))
assert ns.namespace == ["VLR_RIG:SK", "VLR_RIG:SK", "VLR_RIG"]

ns.strip_namespace("VLR_RIG:SK")
assert ns.name == ["VLR_RIG:root", "VLR_RIG:spine", "VLR_RIG:ctrl"]
ns.strip_namespace()
assert ns.name == ["root", "spine", "ctrl"]
```

A rename that would give two nodes the same name raises `ValueError` and
renames nothing.

### User attributes

User defined attributes read and set like the built-in ones. A value is
converted to the attribute's type, and one that does not fit raises.

```python
hero = make_chain(("root", "hip"))
hero["root"].add_user_attribute("heroHeight", 1.8)
hero["root"].heroHeight = 3.1416
assert hero["root"].user_defined_attributes["heroHeight"]["value"] == 3.1416

assert hero.heroHeight == [3.1416, None]
hero.heroHeight = 2
assert hero.heroHeight == [2.0, None]
```

A list gives one value per node, `None` where a node has none, and a write
goes to every node that has the attribute. A compound attribute reads as an
array when its children are all numbers, and an enum takes a field name or
its number. Assigning to a name no node has raises; `add_user_attribute`
creates one.

---

## `TransformList` — views

A `TransformList` is a **non-owning selection**. Slicing, fancy
indexing, `match()` and the query methods all return one, and the nodes
inside still point at their real hierarchy — so `get_parent`,
`world_matrix` and `index` keep answering about the whole rig.

```python
rig = make_chain(("root", "hip", "knee", "ankle"))

rig[0]                       # TransformData -- a plain int gives the node
rig["hip"]                   # TransformData -- by name
rig[1:]                      # TransformList view
rig[[0, 2]]                  # TransformList view (fancy index)
rig[np.array([True, False, True, False])]   # TransformList view (boolean mask)
rig.match("*n*")                # TransformList view (fnmatch)
rig.match("hip", exclude=True)  # everything but hip
rig.match("HIP", exact=False)   # case insensitive

view = rig[1:]
assert isinstance(view, TransformList) and not isinstance(view, HierarchyData)
assert view.name == ["hip", "knee", "ankle"]
assert view[0].get_parent().name == "root"   # still resolves against the rig
```

Build one by hand from existing nodes — it claims nothing.

```python
picked = TransformList([rig["hip"], rig["ankle"]])
assert picked.name == ["hip", "ankle"]
assert picked[0].index == 1                  # index is still the rig's
```

`copy()` on a view returns an owning `HierarchyData`, with parent
references that pointed outside the selection cleared.

```python
detached = rig[1:].copy()
assert isinstance(detached, HierarchyData)
assert detached[0].get_parent() is None      # 'root' was left behind
```

---

## Vectorized channels

Every channel settable on a node is settable on a list, and every getter
comes back stacked.

```python
rig = make_chain(("root", "hip", "knee", "ankle"))

rig.translate                # (4, 3)
rig.rotate                   # (4, 3)
rig.scale                    # (4, 3)
rig.matrix                   # (4, 4, 4)
rig.world_matrix             # (4, 4, 4)
rig.quaternion               # (4, 4)
rig.name                     # list[str]
rig.uuid                     # list[str]
rig.node_type                # list[str]
rig.indices                  # (4,) position of each node in its owner
rig.rotate_axes              # list[str]
rig.unique_name              # (4,) bool
```

Writes either broadcast one value or take one row per node.

```python
rig.rotate_y = 15.0                              # broadcast
assert np.allclose(rig.rotate_y, 15.0)

rig[1:].rotate_y = [10.0, 20.0, 30.0]            # one per node in the view
assert np.allclose(rig.rotate_y, [15.0, 10.0, 20.0, 30.0])

rig.scale                    = 2.0
rig.translate                = np.zeros((4, 3))
rig.rotate_order             = 2
rig.visibility               = False
rig.radius                   = 0.5
rig.draw_style               = 1
rig.segment_scale_compensate = False
rig.joint_orient             = np.zeros((4, 3))
rig.rotate_axis              = np.zeros((4, 3))
```

Matrix writes are applied parent-first, so the result does not depend on
the order the view happens to be in.

```python
rig    = make_chain()
target = rig.world_matrix.copy()
target[:, 3, 1] += 4.0

shuffled              = rig[[2, 0, 1]]
shuffled.world_matrix = target[[2, 0, 1]]
assert np.allclose(rig.world_matrix, target)
```

Read-only list properties: `name`, `uuid`, `node_type`, `indices`,
`rotate_axes`, `unique_name`, `parent_scale_inverse`, and the six factor
matrices (`scale_matrix`, `rotate_axis_matrix`, `rotate_matrix`,
`joint_orient_matrix`, `parent_scale_inverse_matrix`,
`translate_matrix`).

```python
rig.scale_matrix.shape                 # (3, 4, 4)
rig.parent_scale_inverse.shape         # (3, 3)
```

---

## Vectorized queries and edits

```python
rig = make_chain(("root", "hip", "knee", "ankle"))

rig.get_roots().name                   # ['root']
rig.get_parents()                      # array([-1,  0,  1,  2]) parent indices
rig.get_parent_matrix().shape          # (4, 4, 4)
rig[:2].get_children().name            # ['hip', 'knee']  (deduplicated)
rig[:2].get_branch().name              # ['root', 'hip', 'knee', 'ankle']
len(rig.to_attributes())               # 4 dicts
```

The node edits all have a list form.

```python
rig.match("knee", "ankle").set_parent("root")     # reparent a selection
rig.add_prefix("c_")
rig.add_suffix("_jnt")
rig.set_rotate_to_joint_orient()
rig.set_joint_orient_to_rotate()
rig.swapaxes(0, 2)

other = make_chain(("a", "b", "c", "d"), step=(0.0, 7.0, 0.0))
rig.match_translate(other)         # pairs positionally, 1:1
rig.match_rotate([0.0, 0.0, 0.0])  # or broadcasts one value
rig.match_matrix(other[0])         # or one node onto all of them
```

`match_*` on a list solves parent-first too, and raises when a
per-node operand is the wrong length.

```python
try:
    rig[1:].match_translate(other)     # 3 nodes, 4 operands
except ValueError as error:
    print(error)                       # expected 3 nodes to match against, got 4
```

---

## `HierarchyData` — ownership

`HierarchyData` is the owning container. `append` / `insert` claim a
node (detaching it from any previous owner), `pop` releases it.

```python
rig  = HierarchyData()
node = TransformData("solo", node_type="joint")
assert node.uuid is None

rig.append(node)
assert node.uuid is not None           # append assigns one
assert node in rig and "solo" in rig
assert rig.index("solo") == 0

rig.insert(0, TransformData("first"))
assert rig.name == ["first", "solo"]
```

A node lives in exactly one hierarchy at a time.

```python
other = HierarchyData()
other.append(node)                     # moves it out of `rig`
assert node not in rig and node in other
assert len(rig) == 1
```

`pop` clears the back-pointer and re-roots the orphans it leaves behind.

```python
rig = make_chain()
hip = rig.pop(1)
assert rig.name == ["root", "knee"]
assert rig["knee"].get_parent() is None        # its parent left
rig.insert(1, hip)                             # take it back
```

Whole-list ops come from the sequence protocol: `extend`, `remove`,
`clear`, `count`, `reverse`, `sort` (by name by default), `len`,
iteration, `in`.

```python
built = HierarchyData()
built.extend([TransformData("b"), TransformData("a")])
built.sort()
assert built.name == ["a", "b"]
built.reverse()
built.remove(built["a"])
built.clear()
assert len(built) == 0
```

Construct from an iterable, or from a dict keyed by uuid. Uuids must be
real ones — `generate_uuid()` mints them, `validate_uuid()` checks them,
and anything that fails that check is treated as a name instead.

```python
from cgmath.hierarchy import generate_uuid, validate_uuid

root_id, hip_id = generate_uuid(), generate_uuid()
assert validate_uuid(root_id) and not validate_uuid("R")

rig = HierarchyData([
    TransformData("root", uuid=root_id, node_type="joint"),
    TransformData("hip", uuid=hip_id, parent_node=root_id, node_type="joint",
                  translate=(0.0, 10.0, 0.0)),
])
assert rig["hip"].get_parent().name == "root"
assert rig["root"].get_children().name == ["hip"]

rig = HierarchyData.from_dict({
    root_id: {"name": "root", "uuid": root_id},
    hip_id: {"name": "hip", "uuid": hip_id, "parent_node": root_id,
             "translate": [0.0, 10.0, 0.0]},
})
assert np.allclose(rig["hip"].world_matrix[3, :3], [0.0, 10.0, 0.0])
```

---

## Pose algebra

`a - b` is the per-joint **frame offset** that turns `b` into `a`, and
`(a - b) + b == a`. Because the delta is fixed to the bone rather than
to a pose, it re-poses `b` at *any* pose — which is what makes it a
retargeting operator, not just a subtraction.

```python
a         = make_chain()
b         = make_chain()
b.rotate  = np.array([[5.0, 0.0, 0.0]] * 3)

delta     = a - b                                   # HierarchyData
recovered = delta + b
assert np.allclose(recovered.world_matrix, a.world_matrix, atol=1e-6)
```

The methods are the full form, and the receiver flips: `delta + rig` is
spelled `rig.add_delta(delta)`.

```python
delta     = a.get_delta(b)      # a - b
recovered = b.add_delta(delta)  # delta + b
assert np.allclose(recovered.world_matrix, a.world_matrix, atol=1e-6)
```

| Argument | Values | Meaning |
|---|---|---|
| `by` | `"index"` (default) | pair on position; names ignored; same count, order and parenting required |
| | `"name"` | pair on the intersection of the two name sets; allows a partial delta |
| `translate` | `True` (default) | the delta carries a world-space position difference |
| | `False` | orientation only; the result keeps its own bone offsets, so proportions survive |

`get_delta` always returns an owning container, even off a view.

```python
delta = a[1:].get_delta(b[1:])
assert isinstance(delta, HierarchyData) and len(delta) == 2
```

Orientation-only retargeting between rigs of different proportions:

```python
source       = make_chain(("a", "b", "c"), step=(0.0, 10.0, 0.0))
other        = make_chain(("a", "b", "c"), step=(0.0, 11.0, 0.0))
posed        = make_chain(("a", "b", "c"), step=(0.0, 11.0, 0.0))
posed.rotate = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, -35.0], [0.0, 0.0, 0.0]])

delta  = source.get_delta(other, translate=False)
result = posed.add_delta(delta, translate=False)

positions = np.array([n.world_matrix[3, :3] for n in result])
assert np.allclose(np.linalg.norm(np.diff(positions, axis=0), axis=1), 10.0)
```

Let the root keep the animation's world position:

```python
result[0].match_translate(posed[0])
```

Pairing by name lets the delta cover only part of a skeleton.

```python
whole       = make_chain(("root", "hip", "knee"))
part        = HierarchyData([n.copy() for n in whole[1:]])
part.rotate = np.array([[3.0, 0.0, 0.0]] * 2)

delta = whole.get_delta(part, by="name")
assert delta.name == ["root", "hip", "knee"]    # unpaired nodes pass through
```

Pairing errors are loud rather than silent.

```python
for args, kwargs in (((make_chain(("x",)),), {}),                  # length
                     ((make_chain(("p", "q", "r")),), {"by": "name"}),  # names
                     ((make_chain(),), {"by": "uuid"})):           # bad mode
    try:
        whole.get_delta(*args, **kwargs)
    except ValueError as error:
        print(type(error).__name__, str(error)[:40])
```

---

## `ClipData` — construction

A `ClipData` is a `HierarchyData` holding `F` poses. It **copies** the
rig it is built from, so the caller's rig is left alone.

```python
rig  = make_chain()
clip = ClipData(rig, frames=5, start_frame=1001, fps=30.0)

clip.frame_count            # 5
clip.start_frame            # 1001
clip.end_frame              # 1005  (start_frame + frame_count - 1)
clip.fps                    # 30.0
clip.frame                  # 0     -- the loaded frame
len(clip)                   # 3     -- nodes
assert len(rig) == 3        # the source rig is untouched
```

Every frame starts on the rest pose. Without `frames=` the clip holds no
animation at all (`frame_count == 0`) and behaves like a hierarchy.

```python
empty = ClipData(rig)
assert empty.frame_count == 0
```

From a sequence of poses:

```python
poses = []
for i in range(3):
    pose        = rig.copy()
    pose.rotate = np.full((3, 3), float(i) * 10.0)
    poses.append(pose)

built = ClipData.from_poses(poses, start_frame=0, fps=24.0)
assert built.frame_count == 3
assert np.allclose(built.frames.rotate[:, 0, 0], [0.0, 10.0, 20.0])
```

`start_frame` and `fps` are writable; `end_frame` and `frame_count`
follow the blocks.

```python
built.start_frame = 100
built.fps         = 25.0
assert built.end_frame == 102
```

---

## Scrubbing

The nodes *are* the playback head. Each framed channel is bound to one
row of a dense `(F, N, 3)` block, so reading a channel reads the current
frame and writing one writes it — no copy, nothing to flush.

```python
clip = ClipData(make_chain(), frames=4)

clip.frame     = 2
clip.translate = np.full((3, 3), 7.0)          # writes frame 2 only
assert np.allclose(clip.frames.translate[2], 7.0)
assert not np.allclose(clip.frames.translate[0], 7.0)

clip.frame = -1                                # negative indexes from the end
assert clip.frame == 3

clip["knee"].rotate_z = 45.0                   # per-node write, current frame
assert np.allclose(clip.frames.rotate[3, 2, 2], 45.0)
```

Framed channels: `scale`, `rotate`, `translate`, `rotate_axis`,
`joint_orient` (`ClipData.FRAMED_CHANNELS`). Everything else — names,
uuids, parenting, `rotate_order`, `segment_scale_compensate`, `radius`,
`draw_style` — is structural and shared by every frame.

```python
clip.frame                = 0
clip["knee"].rotate_order = 3
clip.frame                = 2
assert clip["knee"].rotate_order == 3          # structural, not keyed
```

Out-of-range frames raise.

```python
try:
    clip.frame = 99
except IndexError as error:
    print(error)                               # frame 99 out of range for 4 frames
```

---

## The frame axis

`clip.frames` is the frame-addressed face of the clip.

| Expression | Result |
|---|---|
| `clip.frames.translate` | `(F, N, 3)` block, **read only** |
| `clip.frames.translate = values` | write the whole block |
| `clip.frames.translate_y` | `(F, N)` one component |
| `clip.frames[12]` | scrubs to 12 and returns the clip itself |
| `clip.frames[12] = pose` | writes one pose |
| `clip.frames[10:20]` | a new 10-frame clip cut out of the range (always a copy) |
| `clip.frames[10:20] = other` | pastes a clip or a pose into the range |
| `clip.frames["hips"]` | an axis scoped to one node, all frames |
| `clip.frames[["a", "b"]]` | scoped to several nodes (names or indices) |
| `clip.frames.match("L_*")` | scoped by fnmatch, all frames |
| `clip.frames.times` | `(F,)` seconds, `(start_frame + i) / fps` |
| `len(clip.frames)` | `F` |
| `iter(clip.frames)` | scrubs frame by frame, yielding the clip |

### Whole blocks

```python
clip = ClipData(make_chain(), frames=6, start_frame=0, fps=24.0)

clip.frames.translate.shape                     # (6, 3, 3)
clip.frames.scale     = 2.0                         # broadcast
clip.frames.translate = np.arange(6 * 3 * 3, dtype=float).reshape(6, 3, 3)
assert np.allclose(clip.frames.translate[5, 2], [51.0, 52.0, 53.0])
assert np.allclose(clip.frames.times, np.arange(6) / 24.0)
```

Block reads are read-only views, so `+=` on one raises. Take a copy, or
go through a component accessor (which returns a copy).

```python
block = np.array(clip.frames.rotate)            # copy
block[:, 0, 1] += 5.0
clip.frames.rotate = block                      # write it back

clip.frames.rotate_y += 5.0                     # components work in place
```

### Components

Each framed channel has `_x`, `_y`, `_z`, shaped `(F, N)`.

```python
clip.frames.translate_y.shape                   # (6, 3)
clip.frames.translate_y = np.zeros((6, 3))
assert np.allclose(clip.frames.translate[..., 1], 0.0)
```

### Scoping to nodes

A scoped axis addresses channels across **every** frame at once.

```python
clip.frames["knee"].translate.shape             # (6, 1, 3)
clip.frames["knee"].translate = np.zeros((6, 1, 3))
clip.frames[["root", "knee"]].rotate.shape      # (6, 2, 3)
clip.frames[[0, 2]].rotate.shape                # (6, 2, 3) -- indices too

before = np.array(clip.frames.rotate_y)
clip.frames.match("k*").rotate_y += 5.0         # every frame of every match
assert np.allclose(clip.frames.rotate_y[:, 2], before[:, 2] + 5.0)
assert np.allclose(clip.frames.rotate_y[:, 0], before[:, 0])   # others untouched
```

A scoped axis cannot then be sliced or written a pose — it addresses
channels, not frames.

```python
for call in (lambda: clip.frames["knee"][0:2],
             lambda: clip.frames.__getitem__("knee").__setitem__(0, make_chain())):
    try:
        call()
    except TypeError as error:
        print(error)
```

### Cutting and pasting frames

```python
cut = clip.frames[2:5]
assert isinstance(cut, ClipData)
assert cut.frame_count == 3 and cut.start_frame == 2      # timebase moves with it
cut.frames.translate = np.zeros((3, 3, 3))
assert not np.allclose(clip.frames.translate[2], 0.0)     # a cut always copies

source                  = ClipData(make_chain(), frames=2)
source.frames.translate = np.full((2, 3, 3), -1.0)
clip.frames[0:2]        = source                                 # paste a clip
assert np.allclose(clip.frames.translate[1], -1.0)

pose             = make_chain()
pose.translate   = np.full((3, 3), 9.0)
clip.frames[4:6] = pose                                   # or fill with one pose
assert np.allclose(clip.frames.translate[5], 9.0)
```

Pasting a clip of the wrong length raises.

```python
try:
    clip.frames[0:4] = source                             # 4 selected, 2 supplied
except ValueError as error:
    print(error)                                          # 4 frames selected, 2 supplied
```

---

## Batched evaluation

`frames.matrix` and `frames.world_matrix` evaluate every frame at once
and agree exactly with scrubbing.

```python
clip               = ClipData(make_chain(("root", "hip", "knee", "ankle")), frames=5)
clip.frames.rotate = np.random.default_rng(0).normal(size=(5, 4, 3)) * 20.0

clip.frames.matrix.shape                        # (5, 4, 4, 4) local
clip.frames.world_matrix.shape                  # (5, 4, 4, 4) world

batched = clip.frames.world_matrix
for frame in range(clip.frame_count):
    clip.frame = frame
    assert np.array_equal(batched[frame], clip.world_matrix)
```

Scoped axes evaluate the same way, narrowed to their columns.

```python
clip.frames.match("hip").world_matrix.shape     # (5, 1, 4, 4)
```

---

## Clip deltas

`get_delta` / `add_delta` cover the whole clip, not just the loaded
frame, and return a `ClipData` carrying the timebase forward.

```python
rest               = make_chain()
clip               = ClipData(rest, frames=4, start_frame=1001, fps=30.0)
clip.frames.rotate = np.random.default_rng(1).normal(size=(4, 3, 3)) * 15.0

delta = clip.get_delta(rest)                    # one pose, against every frame
assert isinstance(delta, ClipData)
assert delta.frame_count == 4 and delta.start_frame == 1001

recovered = ClipData(rest, frames=4).add_delta(delta)
for frame in range(4):
    clip.frame = recovered.frame = frame
    assert np.allclose(recovered.world_matrix, clip.world_matrix, atol=1e-6)
```

The retarget case — one fixed offset applied to every frame:

```python
target     = make_chain(("root", "hip", "knee"), step=(0.0, 12.0, 0.0))
retargeted = clip.add_delta(target.get_delta(rest))
assert retargeted.frame_count == clip.frame_count
```

Two clips of equal length pair frame for frame; unequal lengths raise.

```python
pair = clip.get_delta(clip.copy())
assert pair.frame_count == 4

try:
    clip.get_delta(ClipData(rest, frames=3))
except ValueError as error:
    print(error)                                # clips hold 4 and 3 frames
```

---

## Clip guard rails

Rig-level edits rebuild `rotate` out of a matrix. Doing that per frame
lets each frame pick its own euler branch, which leaves the poses exact
and the curves full of 180° and 360° steps. So on a clip holding more
than one frame they refuse: `swapaxes`, `set_rotate_to_joint_orient`,
`set_joint_orient_to_rotate`, and `set_parent(world_space=True)`.

```python
animated               = ClipData(make_chain(), frames=3)
animated.frames.rotate = np.array([[[0.0, 0.0, f * 11.0]] * 3 for f in range(3)])

for call in (lambda: animated.swapaxes(0, 1),
             lambda: animated.set_rotate_to_joint_orient(),
             lambda: animated.set_joint_orient_to_rotate(),
             lambda: animated.set_parent(None)):
    try:
        call()
    except RuntimeError as error:
        print(str(error)[:48])
```

Do them on the rest pose and build the clip afterwards. What stays
allowed: `set_parent(world_space=False)`, and everything on a clip of
zero or one frame.

```python
animated.set_parent(None, world_space=False)    # writes no pose -- fine

single = ClipData(make_chain(), frames=1)
single.swapaxes(0, 1)                           # one frame has no curve to break
```

Node reordering keeps each block column with its node, so a
`set_parent` that sorts a parent ahead of its children does not shuffle
the animation.

---

## Importers and exporters

`save_fbx` writes the loaded pose; `load_fbx` / `load_glb` read a rig
back. `load` sniffs the file header.

```python
rig  = make_chain()
path = os.path.join(WORKDIR, "rig.fbx")

rig.save_fbx(path)                              # needs the fbx sdk

back = HierarchyData.load_fbx(path)
assert back.name == ["root", "hip", "knee"]
assert np.allclose(back.world_matrix[:, 3, :3], rig.world_matrix[:, 3, :3])

auto = HierarchyData.load(path)                 # header sniffing: fbx / glb / else
assert auto.name == back.name
```

Reading the same file as an animation gives a clip.

```python
clip = ClipData.load_fbx(path)
assert isinstance(clip, ClipData) and clip.frame_count == 1
```

| Loader | Signature |
|---|---|
| `TransformList.load` | `(filename, mode=None)` — `mode` in `"gltf" / "glb" / "fbx" / "json" / "npz" / "pkl"` |
| `TransformList.load_fbx` | `(filename, scale_factor=1.0)` |
| `TransformList.load_glb` | `(filename, scale_factor=100.0, pose_frame=None)` |
| `ClipData.load_fbx` | `(filename, scale_factor=1.0, take=None, fps=None, start_frame=None, end_frame=None)` |
| `ClipData.load_glb` | `(filename, scale_factor=100.0, animation=0, fps=24.0, start_frame=0)` |
| `TransformList.save_fbx` | `(filename, zero_root=False)` |

`zero_root=True` is meant to write the root with an identity transform.
Reaching it through `TransformList.save_fbx` currently raises
`AttributeError` — the exporter looks for a `.data` attribute on the
skeleton it was handed, which a `HierarchyData` does not have. Zero the
root yourself before exporting instead:

```python
flat                   = make_chain()
flat["root"].translate = (0.0, 0.0, 0.0)
flat["root"].rotate    = (0.0, 0.0, 0.0)
flat["root"].scale     = (1.0, 1.0, 1.0)
flat.save_fbx(os.path.join(WORKDIR, "flat.fbx"))
```

`ClipData.save_fbx` writes one pose and no animation stack, so a
multi-frame clip refuses rather than silently shipping whichever frame
is loaded. The message names the cut that would work.

```python
animated       = ClipData(make_chain(), frames=10)
animated.frame = 3
try:
    animated.save_fbx(os.path.join(WORKDIR, "nope.fbx"))
except ValueError as error:
    print(str(error)[:60])                      # ... clip holds 10 frames ...

animated.frames[3:4].save_fbx(os.path.join(WORKDIR, "frame3.fbx"))
```

For `load_glb`, `scale_factor` converts units (glTF is metres, so `100.0`
gives centimetres); `pose_frame` reads one frame of the file's animation
onto the rest pose.

glTF keys every property on its own timeline, so `load_glb` resamples
the samplers onto a uniform grid at `fps` and runs the result through
`euler_filter`. `load_fbx` does neither — fbx already stores euler
curves, and rewriting their branches would change the animation. A file
with no animation gives a one-frame clip rather than an error.

`ClipData.load_fbx` defaults to the first take that was actually keyed,
not to index 0, because a file can hold takes with no curves at all.

---

## Serialization

`save` / `load` pick the writer from the extension (or `mode=`), and
round-trip a rig or a whole clip.

```python
rig = make_chain()
for ext in ("json", "npz", "pkl"):
    path = rig.save(os.path.join(WORKDIR, f"rig.{ext}"))
    assert HierarchyData.load(path) == rig

clip                  = ClipData(rig, frames=4, start_frame=1001, fps=30.0)
clip.frames.translate = np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3)
path                  = clip.save(os.path.join(WORKDIR, "clip.json"))
loaded                = ClipData.load(path)
assert loaded == clip and loaded.frame_count == 4 and loaded.fps == 30.0
```

The explicit pairs work too, on both the node and the list.

```python
rig.save_json(os.path.join(WORKDIR, "a.json"))
rig.save_npz(os.path.join(WORKDIR, "a.npz"))
rig.save_pickle(os.path.join(WORKDIR, "a.pkl"))
HierarchyData.load_json(os.path.join(WORKDIR, "a.json"))
HierarchyData.load_npz(os.path.join(WORKDIR, "a.npz"))
HierarchyData.load_pickle(os.path.join(WORKDIR, "a.pkl"))

node_path = rig["hip"].save(os.path.join(WORKDIR, "hip.json"))
assert TransformData.load(node_path).name == "hip"
```

In memory: `to_dict` / `from_dict`, `to_json`, `to_bytes` / `from_bytes`.

```python
assert HierarchyData.from_dict(rig.to_dict()) == rig
assert HierarchyData.from_bytes(rig.to_bytes()) == rig
rig.to_json()[:1]                               # '{'

assert TransformData.from_dict(rig["hip"].to_dict()).name == "hip"
```

A list's `to_dict` is keyed by uuid (so duplicate names are safe); a
`ClipData` adds one extra key, `ClipData.CLIP_KEY` (`"__clip__"`),
holding the timebase and the blocks.

```python
sorted(clip.to_dict())[-1]                      # '__clip__'
sorted(clip.to_dict()["__clip__"])
# ['_joint_orient', '_rotate', '_rotate_axis', '_scale', '_translate',
#  'fps', 'frame', 'start_frame']
```

`copy()` is deep and owning; `==` compares values, and on a clip it also
compares the timebase and every block.

```python
twin = clip.copy()
assert twin == clip
twin.frames.rotate = np.full((4, 3, 3), 45.0)
assert twin != clip
```

Two more helpers from the base class:

```python
TransformData.info()[:14]                       # 'TransformData(' -- the docstring
rig["hip"].match("hip")                         # True
```

---

## Gotchas

| Trap | What happens | Do this |
|---|---|---|
| `parent_node` holding a **name** | `get_parent()` resolves it, but `get_children()` / `get_branch()` match on **uuid** and come back empty — so the world-matrix cache of the branch is never invalidated | parent with `set_parent(name)`, or set `parent_node` to the parent's `uuid` |
| `world_matrix` as a call | it is a **property** on every type here | `node.world_matrix`, not `node.world_matrix()` |
| mutating a channel getter | `node.rotate[0] = 5` writes to a throwaway copy | read, edit, assign back |
| `clip.frames.rotate += 5` | the block getter is read-only, so `+=` raises | `np.array(...)` then assign, or use `rotate_y += 5` |
| `rig + delta` | `+` only reads the delta from its **left** operand | `delta + rig`, or `rig.add_delta(delta)` |
| mixing `translate=` between calls | `get_delta` and `add_delta` store different things in the one channel | pass the same value to both |
| `by="name"` with a partial overlap | accepted silently; only a completely empty intersection raises | check `len(delta)` when it matters |
| `by="index"` on mirrored limbs | both arms share a parent index, so left can pair to right | use `by="name"` when sibling order is not guaranteed |
| `reset_cached_data()` | clears every non-annotated attribute — including the `_hierarchy` back-pointer, which detaches the nodes | re-append the nodes, or just write a channel to drop the world-matrix cache |
| `ClipData(rig)` with no `frames=` | `frame_count == 0`; it is a plain hierarchy until frames are allocated | pass `frames=N`, or build with `from_poses` |
| rig edits on an animated clip | `swapaxes`, the orient helpers and `set_parent(world_space=True)` raise `RuntimeError` | apply them to the rest pose, then build the clip |
| `clip.frames[12]` | scrubs and hands back the **same** clip — two frames cannot be held at once | `clip.frames[12:13]` for an independent cut, or `copy()` |
