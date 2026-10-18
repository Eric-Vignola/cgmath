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
| [`TransformData` — construction](#transformdata--construction) | the constructor arguments |
| [SRT channels](#srt-channels) | `scale` / `rotate` / `translate` and per-axis accessors |
| [Joint channels](#joint-channels) | `rotate_order`, `rotate_axis`, `joint_orient`, `segment_scale_compensate` |
| [Changing rotate order](#changing-rotate-order) | `set_rotate_order` keeps the pose; the raw `rotate_order` setter does not |
| [Display + user channels](#display--user-channels) | `visibility`, `radius`, `draw_style`, `user_defined_attributes` |
| [Matrices](#matrices) | `matrix`, `world_matrix`, the six factor matrices, `quaternion`, the parent and inverse matrices |
| [Maya's short names](#mayas-short-names) | `t` `tx` … `wm` `pm` `im` `wim` `pim`, read and written as their long names |
| [Cache invalidation](#cache-invalidation) | when `world_matrix` is recomputed |
| [Parenting: `parent_uuid`](#parenting-parent_uuid) | the parent's uuid, `parent_name`, `set_parent` by name, joining a rig, `parent_node`, older files |
| [Node queries](#node-queries) | parent / children / branch / root / index |
| [Node edits](#node-edits) | `set_parent`, `match_*`, `swapaxes`, orient helpers, node type, prefix / suffix, namespaces, user attributes |
| [`TransformList` — views](#transformlist--views) | slicing, fancy indexing, `match` |
| [Vectorized channels](#vectorized-channels) | read / write a whole selection |
| [Vectorized queries and edits](#vectorized-queries-and-edits) | the list forms of every node method |
| [`HierarchyData` — ownership](#hierarchydata--ownership) | `append` / `insert` / `pop` / `extend` |
| [Pose algebra](#pose-algebra) | `get_delta` / `add_delta`, `-` / `+` |
| [`ClipData` — construction](#clipdata--construction) | frames, timebase |
| [Scrubbing](#scrubbing) | `clip.frame`, the nodes as a playback head |
| [The frame axis](#the-frame-axis) | `clip.frames` blocks, components, scoping, cuts, pastes |
| [Batched evaluation](#batched-evaluation) | `frames.matrix`, `frames.world_matrix`, the factor matrices, `parent_matrix`, `to_attributes()` |
| [Solving every frame](#solving-every-frame) | `frames.world_matrix =`, `frames.matrix =`, `frames.quaternion`, `frames.match_*` |
| [Clip deltas](#clip-deltas) | retargeting a whole clip |
| [Rig edits on a clip](#rig-edits-on-a-clip) | every frame at once, the curves filtered |
| [Importers and exporters](#importers-and-exporters) | fbx, glb, auto-detect |
| [Serialization](#serialization) | json / npz / pkl, dict, bytes |
| [Gotchas](#gotchas) | the traps, in one table |

---

## Setup

No rig assets ship with this package, so build one inline. `set_parent`
takes the parent by name and stores its **uuid** in `parent_uuid`, the one
link a node keeps, which no rename moves (see
[Parenting](#parenting-parent_uuid)).

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
| `parent_uuid` | `None` | the **parent's uuid**, `None` for a root; a name raises `ValueError` (see [Parenting](#parenting-parent_uuid)) |
| `scale` | `(1, 1, 1)` | |
| `rotate` | `(0, 0, 0)` | degrees |
| `translate` | `(0, 0, 0)` | |
| `rotate_order` | `0` | `0=xyz 1=yzx 2=zxy 3=xzy 4=yxz 5=zyx` |
| `rotate_axis` | `(0, 0, 0)` | degrees; Maya `rotateAxis` |
| `joint_orient` | `(0, 0, 0)` | degrees; Maya `jointOrient` |
| `visibility` | `True` | |
| `segment_scale_compensate` | `False` | divide out the parent's scale |
| `radius` | `1.0` | joint display radius |
| `draw_style` | `0` | Maya `drawStyle` |
| `user_defined_attributes` | `{}` | extra attributes carried through `save_fbx` and back by `load_fbx`, but for arrays, matrices, multis and other compounds; each node gets its own dict |
| `parent_node` | `None` | `parent_uuid`'s deprecated name, with a `DeprecationWarning`; goes in 3.0.0 |

---

## SRT channels

Every vector channel is a read/write property returning a fresh float64
`(3,)` array, plus three scalar per-axis accessors. One number sets all
three, and a `(1, 3)` row, what a transforms function returns for one
matrix, writes its row; any other value that is not three long raises
`ValueError` naming the channel.

```python
node.translate                    # array([0., 1., 0.])
node.translate   = (1.0, 2.0, 3.0)
node.translate_y = 5.0
node.translate                    # array([1., 5., 3.])

node.rotate  = (0.0, 90.0, 0.0)    # degrees
node.scale_z = 2.0
node.scale                        # array([1., 1., 2.])
node.scale   = [[1.0, 1.0, 2.0]]  # one (1, 3) row writes its row
node.scale                        # array([1., 1., 2.])

try:
    node.translate = (1.0, 2.0)
except ValueError as error:
    print(error)                  # TransformData.translate must have shape (3,), got (2,)
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

### Changing rotate order

Setting `rotate_order` changes the order alone, as Maya's `setAttr` does:
`rotate` keeps its angles, which now read in the new order, so the joint
turns and its children move with it. `set_rotate_order` keeps the pose, as
`xform -rotateOrder <order> -preserve` does: only `rotate` changes,
re-expressed in the new order, and no world matrix moves. `joint_orient`
and `rotate_axis` are built in xyz whatever the order, so they stay. The
order is `0..5` or its axes, `"zyx"`; the order a node already has changes
nothing.

```python
wrist = TransformData("wrist", node_type="joint", rotate=(400.0, 30.0, -200.0))
HierarchyData([wrist])
before = wrist.world_matrix

wrist.set_rotate_order("yzx")
np.round(wrist.rotate, 4).tolist()          # [137.6136, -170.2335, 34.3339], as Maya
assert wrist.rotate_order == 1
assert np.allclose(wrist.world_matrix, before)

wrist.rotate_order = 0                      # raw: the angles stay, the joint turns
assert not np.allclose(wrist.world_matrix, before)
```

A rotation has two euler solutions in any order, plus whole turns on any
axis. The one picked is Maya's: the new order's solution with every angle
in [-180, 180), or the other one -- the first and last axes the new order
applies turned by 180, the middle one 180 minus itself, not wrapped --
when its angles sum to fewer degrees; the first on a tie. That matches
Maya 2025 on all 2550 joints of `_tests/test_assets/maya_euler.npz`, but
for an angle of exactly 180, which Maya writes as +180 or -180 as its own
arithmetic falls: the same angle.

A list, a rig or a clip takes one order for every node or one per node,
and solves them all at once. On a clip the order is structural, shared by
every frame, so every frame is re-expressed: frame 0 picks as above, and
the curves then go through `euler_filter` in the new order, frame 0 kept,
so they come out smooth.

```python
pair        = make_chain(("a", "b"))
pair.rotate = [(10.0, 95.0, 0.0), (-170.0, 20.0, 45.0)]
before      = pair.world_matrix
pair.set_rotate_order(["zyx", 4])           # one per node: zyx, yxz
pair.rotate_axes                            # ['zyx', 'yxz']
assert np.allclose(pair.world_matrix, before)

turn                 = ClipData(make_chain(("a",)), frames=36)
turn.frames.rotate_x = np.linspace(0.0, 350.0, 36).reshape(36, 1)
turn.set_rotate_order("yzx")
turn.frames.rotate[[17, 18, 35], 0, 0].tolist()
# [170.0, 180.0, 350.0] -- frame by frame, 180 and 350 would read -180 and -10
```

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

Pass your own mapping to rename the keys. It names a channel as a saved
file does, by its property, `"translate"`; the private field's own name,
`"_translate"`, works too, and a mapping holding both raises `ValueError`.
On an `SDFSphere` or `SDFCylinder` the two spellings of a radius differ:
`"radius"` is the primitive's radius, as its file saves it, and only
`"_radius"` names the joint radius. On a rig that mixes them with other
nodes, where `"radius"` is the joint radius too, name the primitive's radius
`"_sdf_radius"`.

```python
joint.to_attributes({"translate": "t", "rotate": "r"})
# {'r': [0.0, 0.0, 0.0], 't': [0.0, 0.0, 0.0]}
assert joint.to_attributes({"_translate": "t"}) == joint.to_attributes({"translate": "t"})
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
| `parent_matrix` | yes | — | `parent_world`, identity at a root: `get_parent_matrix()`, Maya's `parentMatrix` |
| `inverse_matrix` | yes | — | the inverse of `matrix`, Maya's `inverseMatrix` |
| `world_inverse_matrix` | yes | — | the inverse of `world_matrix`, Maya's `worldInverseMatrix` |
| `parent_inverse_matrix` | yes | — | the inverse of `parent_matrix`, Maya's `parentInverseMatrix` |

The four last are true inverses, `np.linalg.inv`, as Maya's are: a
non-uniform scale under a rotation shears a matrix, which
`transforms.matrix_inverse`, an inverse for a scale times a rotation, gets
wrong. A list or a rig reads them for every node at once, each owning rig
evaluated once and every node inverted in one call, and `clip.frames` for
every frame in one call. A zero scale leaves no inverse:
`numpy.linalg.LinAlgError` names the node, and on `clip.frames` the frame.
Maya holds a zero scale at 1e-12 and inverts that, into numbers of 1e12.

```python
leg  = make_chain(("hip", "knee"))
shin = leg["knee"]
assert np.array_equal(shin.parent_matrix, leg["hip"].world_matrix)
assert np.array_equal(leg["hip"].parent_matrix, np.eye(4))      # a root
assert np.allclose(shin.world_inverse_matrix @ shin.world_matrix, np.eye(4))
leg.inverse_matrix.shape                    # (2, 4, 4)

leg["hip"].scale_y = 0.0
try:
    leg.world_inverse_matrix
except np.linalg.LinAlgError as error:
    print(error)                            # the world_matrix of 'hip' has no inverse: a zero scale flattens it
```

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
translate, solving `rotate` around the `rotate_axis`, `joint_orient` and
`segment_scale_compensate` the node already has, so the matrix reads back
as written. A mirror goes on `scale_z`, the way Maya splits a matrix, and
what a zero scale leaves of the rotation is kept from the current one.
Writing `quaternion` solves only `rotate`, keeping `rotate_axis` and
`joint_orient`. `set_parent()` and `match_matrix()` write through them
too, so a reparented transform or locator keeps its `rotate_axis`, which
`save_fbx` writes for joints only (see
[Importers and exporters](#importers-and-exporters)).

```python
knee              = rig["knee"]
knee.rotate_axis  = (0.0, 0.0, 10.0)
knee.joint_orient = (0.0, 30.0, 0.0)
target            = TransformData("pose", rotate=(20.0, 0.0, 45.0), translate=(1.0, 2.0, 3.0)).matrix

knee.world_matrix = target
assert np.allclose(knee.world_matrix, target)
assert np.allclose(knee.rotate_axis, [0.0, 0.0, 10.0])   # kept; rotate is solved
assert np.allclose(knee.joint_orient, [0.0, 30.0, 0.0])

mirror            = TransformData("mirror", rotate=(-45.0, -90.0, 0.0), scale=(-1.0, 1.0, 1.0)).matrix
knee.world_matrix = mirror
assert np.allclose(knee.world_matrix, mirror)
assert np.allclose(knee.scale, [1.0, 1.0, -1.0])         # the mirror lands on z

rig["hip"].quaternion = [0.0, 0.0, 0.0, 1.0]       # identity orientation
assert np.allclose(rig["hip"].rotate, 0.0, atol=1e-9)
```

---

## Maya's short names

Each channel and matrix also answers to the short name Maya gives it:
`node.tx` is `node.translate_x`, `rig.wm` is `rig.world_matrix`. A short
name is a property that reads and writes its long name itself, so it
converts and raises as the long name does, takes one value or one per node
on a list, writes a clip's bound frame on a clip and every frame on
`clip.frames`. They are Maya 2025's, as
`cmds.attributeQuery(<long>, node=<joint>, shortName=True)` gives them;
`inverseScale`'s, `is`, is a Python keyword, so it has none. A file keeps
the long names: a short name is never saved.

```python
leg     = make_chain(("hip", "knee"))
shin    = leg["knee"]
shin.ry = 30.0              # shin.rotate_y
shin.jo = (0.0, 0.0, 90.0)  # shin.joint_orient
assert shin.r.tolist() == [0.0, 30.0, 0.0]
assert np.array_equal(shin.wm, shin.world_matrix)

leg.tx = [1.0, 2.0]                         # one per node, as leg.translate_x
assert leg.translate_x.tolist() == [1.0, 2.0]
assert "tx" in dir(shin)                    # and tab completion

try:
    shin.t = (1.0, 2.0)
except ValueError as error:
    print(error)                            # TransformData.translate must have shape (3,), got (2,)

steps           = ClipData(leg, frames=3)
steps.frames.rz = np.linspace(0.0, 90.0, 3).reshape(3, 1)   # every frame, as frames.rotate_z
steps.frames.wim.shape                      # (3, 2, 4, 4)
```

| Short | Long | Maya's long name | Writes |
|---|---|---|---|
| `t` `tx` `ty` `tz` | `translate` `translate_x` `translate_y` `translate_z` | `translate` `translateX` … | yes |
| `r` `rx` `ry` `rz` | `rotate` `rotate_x` `rotate_y` `rotate_z` | `rotate` `rotateX` … | yes |
| `s` `sx` `sy` `sz` | `scale` `scale_x` `scale_y` `scale_z` | `scale` `scaleX` … | yes |
| `jo` `jox` `joy` `joz` | `joint_orient` `joint_orient_x` … | `jointOrient` `jointOrientX` … | yes |
| `ra` `rax` `ray` `raz` | `rotate_axis` `rotate_axis_x` … | `rotateAxis` `rotateAxisX` … | yes |
| `ro` | `rotate_order` | `rotateOrder` | yes |
| `ssc` | `segment_scale_compensate` | `segmentScaleCompensate` | yes |
| `v` | `visibility` | `visibility` | yes |
| `radi` | `radius` | `radius` | yes |
| `ds` | `draw_style` | `drawStyle` | yes |
| `m` | `matrix` | `matrix` | yes |
| `wm` | `world_matrix` | `worldMatrix` | yes |
| `pm` | `parent_matrix` | `parentMatrix` | read only |
| `im` | `inverse_matrix` | `inverseMatrix` | read only |
| `wim` | `world_inverse_matrix` | `worldInverseMatrix` | read only |
| `pim` | `parent_inverse_matrix` | `parentInverseMatrix` | read only |

Every one is on a node, a list, a rig and a clip. `clip.frames` has those
of what it has, the framed channels and the matrices, but not `ro`, `ssc`,
`v`, `radi` or `ds`: every frame shares those, and reading or writing one
there raises as its long name does. On an `SDFSphere` or `SDFCylinder`,
`radi` is the primitive's `radius`, as it is the class's.

`add_user_attribute` refuses every short name, and the four new matrices'
long names, as it refuses a long name. A file whose user attribute carries
one anyway -- an fbx from another tool, a tree written by hand -- loads it
into `user_defined_attributes` with one warning a name, however many nodes
carry it, and `node.tx` stays `translate_x`; a save keeps the attribute.

```python
import warnings

try:
    shin.add_user_attribute("tx", 1.0)
except ValueError as error:
    print(error)                            # 'tx' is a built-in attribute of TransformData, Maya's short name for translate_x

spec = {"attributeType": "double", "value": 7.0, "keyable": True}
tree = HierarchyData([TransformData("a", user_defined_attributes={"tx": spec})]).to_dict()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    found = HierarchyData.from_dict(tree)
print(caught[0].message)
# user attribute 'tx' of 'a' loaded into user_defined_attributes['tx']: node.tx is
# Maya's short name for translate_x, which hides it
assert found[0].tx == found[0].translate_x == 0.0
assert found[0].user_defined_attributes["tx"]["value"] == 7.0
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

The branch walk is `get_children`: the nodes whose `parent_uuid` is the
node's uuid. The hierarchy keeps an index of every node's children for the
walk, so a write costs the size of its branch, not of the rig. While the
index is current, a node appended or a new `parent_uuid` patches it in
place; a new `uuid`, or an edit that takes nodes out of the list or
reorders it through the hierarchy's own methods, retires it. A node
inserted ahead of the end moves every later entry, a pass over the whole
index: it is made when something read the index since the last such pass,
and otherwise the insert retires it, so inserts with nothing read between
cost what they did before there was an index to patch; the build an
append makes is no such read. A rig built one node at a time, a world
matrix read or written on the way, builds the index at the first read,
and so does an append to a rig of 64 nodes or more that keeps none: from
there each node costs the same, whatever the size of the rig. An
assignment, sort or reverse made on the raw `rig.list` is not seen, so
go through `rig[i] = node` or `rig.sort()`.

A name links no node to another, so a rename moves nothing: the index
and every cached world matrix stand, and the name table a lookup by name
builds (`set_parent("hip")`) follows the rename. While the index is
current, a node added that no node names leaves every cached world matrix
standing too, and a new `parent_uuid` on a node with nothing below it
drops that node's cache alone. Any other edit of the rig can give any
node another parent, so a world matrix cached before it is not trusted
after it: the next read computes it again, down from the nearest parent
still cached, though the edit cleared no branch. An edit in another rig,
or on a node no rig holds, leaves every cache of this one standing.

The exception is a node put in with `rig[i] = node` while another rig
owns it, or none does: listed here but not held, it is moved by what
clears no branch and moves no count of this rig -- a channel written on
it or above it, an edit of the rig holding it. So a world matrix cached
below it holds that node's cached one too, and stands only while the node
keeps it.

```python
rig    = HierarchyData([TransformData("slot"), TransformData("hand")])
other  = make_chain(("base", "post"))  # post 10 above base
rig[0] = other["post"]                 # listed here, held there
rig["hand"].set_parent(other["post"], world_space=False)
assert np.allclose(rig["hand"].world_matrix[3, :3], [0.0, 10.0, 0.0])
other["base"].translate = (5.0, 0.0, 0.0)  # an edit of the other rig
assert np.allclose(rig["hand"].world_matrix[3, :3], [5.0, 10.0, 0.0])
```

```python
rig = make_chain()
rig["knee"].world_matrix                          # cached under hip
rig["knee"].parent_uuid = rig["root"].uuid        # a raw write, no branch cleared
assert np.allclose(rig["knee"].world_matrix[3, :3], [0.0, 10.0, 0.0])
```

```python
rig = make_chain()
rig["knee"].world_matrix
rig["hip"].name = "thigh"                         # moves nothing
assert rig["knee"].parent_name == "thigh"
rig["thigh"].translate = (5.0, 10.0, 0.0)         # its branch cleared, as ever
assert np.allclose(rig["knee"].world_matrix[3, :3], [5.0, 20.0, 0.0])
```

---

## Parenting: `parent_uuid`

A node's parent is the node of its hierarchy carrying its `parent_uuid`,
and `None` makes it a root. `parent_uuid` takes a uuid or `None` and
nothing else: a name raises `ValueError`, which names `set_parent`, the
way to parent by name. A `uuid.UUID` is kept as its text, as the `uuid`
keyword keeps one. `parent_name` reads the parent's name, `None` at a
root; it is read only, and no file saves it.

```python
rig = make_chain()
assert rig["knee"].parent_uuid == rig["hip"].uuid
rig["knee"].parent_name                          # 'hip'
rig["root"].parent_name                          # None

try:
    rig["knee"].parent_uuid = "hip"
except ValueError as error:
    print(error)    # TransformData.parent_uuid takes a uuid or None, not 'hip'; set_parent('hip') parents by name
```

`set_parent` takes the parent as a node, its uuid, or a name, the first
node of the hierarchy carrying it. Whichever way it is given, the node
keeps the uuid: a rename after moves nothing.

```python
rig["knee"].set_parent("root", world_space=False)            # by name
rig["knee"].set_parent(rig["hip"].uuid, world_space=False)   # by uuid
rig["knee"].set_parent(rig["hip"], world_space=False)        # by node
rig["hip"].name = "thigh"
assert rig["knee"].parent_name == "thigh"
```

A node joining a hierarchy keeps its parent only when the hierarchy holds
it. Appended, inserted, or put in place with `rig[i] = node` while no rig
owns it, a node whose `parent_uuid` no node of the hierarchy carries
becomes a root. The constructor, `extend` and a load check once every
node is in, so a child may come before its parent: a rig's `copy()` and
`copy.deepcopy` keep every parent whatever the order. A parent added later
does not take a root back. A node's `copy()` keeps its `parent_uuid` until
it joins a rig without that parent; a view's `copy()` and a clip of part
of a rig make each node whose parent was left out a root.

```python
import copy
from cgmath.hierarchy import generate_uuid

ids  = {name: generate_uuid() for name in ("root", "hip")}
kids = HierarchyData([
    TransformData("hip", uuid=ids["hip"], parent_uuid=ids["root"]),   # ahead of its parent
    TransformData("root", uuid=ids["root"]),
])
assert kids["hip"].parent_name == "root"
assert copy.deepcopy(kids)["hip"].parent_name == "root"

lone = rig["knee"].copy()                        # holds the uuid of thigh
assert lone.parent_uuid == rig["thigh"].uuid
HierarchyData([lone])                            # a rig with no thigh
assert lone.parent_uuid is None
```

`parent_node`, `parent_uuid`'s name before cgmath 2.0.0, stays until
3.0.0 as a deprecated alias: the property and the constructor keyword
read and write `parent_uuid`, and a `to_attributes` mapping naming it maps
`parent_uuid`, each with a `DeprecationWarning` filed under the caller,
through an `Object`'s, a `Camera`'s or a `Light`'s constructor too. A name
is refused the same way. A file saved before 2.0.0 names its parents under
`"parent_node"` and still loads, a uuid as it is and a name as the first
node of that name in the loaded rig (a name no node carries leaves a
root); it saves back as `"parent_uuid"`.

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert rig["knee"].parent_node == rig["thigh"].uuid
    assert rig["knee"].to_attributes({"parent_node": "parent"}) == {"parent": rig["thigh"].uuid}
assert [w.category for w in caught] == [DeprecationWarning] * 2

old = HierarchyData.from_dict({
    "a": {"name": "root", "uuid": ids["root"]},
    "b": {"name": "hip", "parent_node": "root"},  # by name, as an older file may
})
assert old["hip"].parent_uuid == ids["root"]
assert "parent_uuid" in old.to_dict()["__items__"][1]
```

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
| `get_parent()` | the parent `TransformData`, `None` at a root, or the `parent_uuid` itself when no node of the hierarchy carries it (the parent taken out with `del` or `remove()`) |
| `parent_name` | the parent's name, `None` at a root; read only |
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

The parent is a node, its uuid or a name (the first node carrying it);
the node keeps its uuid. `set_parent` also moves the node and its branch
to the end of the list, so a parent always precedes its children. A
parent that hangs below the node would close a loop: it raises before
anything moves.

```python
rig = make_chain()
try:
    rig["hip"].set_parent("knee")
except ValueError as error:
    message = str(error)    # 'hip' can not go under 'knee', which hangs below it
assert "hangs below it" in message
```

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

`match_rotate` writes `rotate` alone, solved around the `rotate_axis`,
`joint_orient` and compensated parent scale the node already has, so the
node lands on the orientation whatever those hold.

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

### Change node type

```python
typed              = make_chain()
typed.joint_orient = [[0.0, 30.0, 0.0]] * 3
world              = typed.world_matrix.copy()

typed.node_type = "transform"  # every node; a list sets one per node
assert np.allclose(typed.world_matrix, world, atol=1e-9)
assert np.allclose(typed.joint_orient, 0.0)

typed[1:].node_type = "joint"           # a view sets a subset
typed["root"].set_node_type("locator")  # one node
print(typed.node_type)                  # ['locator', 'joint', 'joint']
```

A joint that becomes another type folds `joint_orient` into `rotate` (the
pose holds and `rotate_axis` stays), turns `segment_scale_compensate` off
and resets `radius` / `draw_style` to 1.0 / 0. Under a scaled parent it
moves, since it no longer cancels that scale. Assigning `node.node_type`
directly only sets the type.

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
renames nothing. Every named data type and list has the same method: a skin
strips its influences with its name, so strip the rig first, then the skins
and meshes read from the same file.

`cgmath.hierarchy.load` / `load_fbx` / `load_glb` read a file's hierarchy
(names as in the file), like the other modules' loaders. An fbx or glb
gives a `HierarchyData`; a cgmath file gives whatever hierarchy it holds,
so a saved clip comes back a `ClipData`, where `HierarchyData.load`
refuses one.

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

A name the class has -- a method such as `copy`, a read-only property such
as `parent_name` -- is never a user attribute: assigning to it raises, on a
node and on a list, as assigning a private name the class does not declare
does.

A list gives one value per node, `None` where a node has none, and a write
goes to every node that has the attribute. A compound attribute reads as an
array when its children are all numbers, and an enum takes a field name or
its number. Assigning to a name no node has raises; `add_user_attribute`
creates one. It refuses a name a built-in has, Maya's short names
included, `tx` or `wm`: see [Maya's short names](#mayas-short-names) for a
file that carries one.

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
references that pointed outside the selection cleared. Saving, pickling
or deep-copying a view gives the same `HierarchyData`. An object two of
its nodes hold, a mesh two Objects instance, stays one object.

```python
detached = rig[1:].copy()
assert isinstance(detached, HierarchyData)
assert detached[0].get_parent() is None      # 'root' was left behind

saved = HierarchyData.load(rig[1:].save(os.path.join(WORKDIR, "view.json")))
assert type(saved) is HierarchyData and saved[0].get_parent() is None
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

Read-only list properties: `name`, `uuid`, `indices`,
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
rig.get_parents()                      # array([-1,  0,  1,  2]) parent indices, -1 at a root
                                       # or for a parent the hierarchy does not hold
rig.get_parent_matrix().shape          # (4, 4, 4)
rig[:2].get_children().name            # ['hip', 'knee']  (deduplicated)
rig[:2].get_branch().name              # ['root', 'hip', 'knee', 'ankle']
len(rig.to_attributes())               # 4 dicts
```

The node edits all have a list form.

A selection is reparented the way one node at a time would do it --
every node to the world, then each under the parent, a name resolved once
the first pass has moved them -- in one pass over the rig, however many
nodes it holds.

```python
rig.match("knee", "ankle").set_parent("root")     # reparent a selection
rig.add_prefix("c_")
rig.add_suffix("_jnt")
rig.set_rotate_to_joint_orient()
rig.set_joint_orient_to_rotate()
rig.swapaxes(0, 2)           # every node solved at once
rig.set_rotate_order("zyx")  # one order for all, or one per node

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

Construct from an iterable, or with `from_dict` from a dict of node dicts
keyed by uuid (what the fbx and glb readers build). A `parent_uuid` must be
a real uuid — `generate_uuid()` mints them, `validate_uuid()` checks them,
and anything that fails that check raises.

```python
from cgmath.hierarchy import generate_uuid, validate_uuid

root_id, hip_id = generate_uuid(), generate_uuid()
assert validate_uuid(root_id) and not validate_uuid("R")

rig = HierarchyData([
    TransformData("root", uuid=root_id, node_type="joint"),
    TransformData("hip", uuid=hip_id, parent_uuid=root_id, node_type="joint",
                  translate=(0.0, 10.0, 0.0)),
])
assert rig["hip"].get_parent().name == "root"
assert rig["root"].get_children().name == ["hip"]

rig = HierarchyData.from_dict({
    root_id: {"name": "root", "uuid": root_id},
    hip_id: {"name": "hip", "uuid": hip_id, "parent_uuid": root_id,
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
| `clip.frames.rotate_order = 0` | raises `AttributeError`: the axis writes only what it defines, and a value every frame shares is set on the clip |

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

A scoped axis addresses channels across **every** frame at once. One
node's component reads back `(F, 1)`, and takes a flat list of `F` values
as well.

```python
clip.frames["knee"].translate.shape             # (6, 1, 3)
clip.frames["knee"].translate = np.zeros((6, 1, 3))
clip.frames["knee"].translate_y.shape           # (6, 1)
clip.frames["knee"].translate_y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
assert np.allclose(clip.frames.translate[:, 2, 1], np.arange(6.0))
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

A paste copies the source's frames across whole, nodes paired by
position, and leaves the source on the frame it was on; a clip pasted
into itself reads every frame first, so `clip.frames[::-1] = clip` plays
it backwards. A pose is read once and repeated down the range. Either
way the nodes are bound once.

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

What a node builds from its channels reads the same way, every frame at
once, each equal bit for bit to the node's own read on every frame:

| Read | Every frame |
|---|---|
| `scale_matrix`, `rotate_axis_matrix`, `rotate_matrix`, `joint_orient_matrix`, `translate_matrix` | `(F, N, 4, 4)`, `rotate_matrix` in each node's rotate order |
| `parent_scale_inverse` | `(F, N, 3)`: the parent's scale, ones for a root, whatever `segment_scale_compensate` says |
| `parent_scale_inverse_matrix` | `(F, N, 4, 4)`: identity for a node that does not compensate |
| `parent_matrix`, `get_parent_matrix()` | `(F, N, 4, 4)`: identity for a root, or a parent the clip does not hold |
| `inverse_matrix`, `world_inverse_matrix`, `parent_inverse_matrix` | `(F, N, 4, 4)`: every frame inverted in one call |
| `to_attributes()` | one dict per node: a framed channel as one `[x, y, z]` per frame, anything else as its one value |

```python
parents = clip.frames.parent_matrix
rotates = clip.frames.rotate_matrix
inverse = clip.frames.world_inverse_matrix
for frame in range(clip.frame_count):
    clip.frame = frame
    assert np.array_equal(parents[frame], clip.get_parent_matrix())
    assert np.array_equal(rotates[frame], clip.rotate_matrix)
    assert np.array_equal(inverse[frame], [node.world_inverse_matrix for node in clip])

knee = clip.frames["knee"].to_attributes()[0]
len(knee["rotate"])                             # 5 -- one [x, y, z] per frame
knee["rotateOrder"]                             # 0 -- shared by every frame
```

They are reads: assigning one raises, and so does any name the axis does
not define, since each `clip.frames` is a new object a write would be lost
on.

```python
try:
    clip.frames.rotate_order = 0
except AttributeError as error:
    print(error)                                # clip.frames has no 'rotate_order' to write; ...
```

---

## Solving every frame

The pose setters have every-frame forms on the frame axis, solved for
every frame at once. Each equals the node setters run frame by frame on
a copy of each pose (world matrices within 1e-9), and a rotate rebuilt
from a matrix then goes through `euler_filter` in each node's rotate
order, frame 0 kept, so the curves carry no 180 or 360 degree steps.
Within about 1e-4 degrees of gimbal lock, where `matrix_to_euler` snaps
the angles a rotation reads back as, neither route holds a world matrix
to better than a few 1e-9.

| Write | Takes |
|---|---|
| `frames.world_matrix = W` | `(4, 4)` for every node and frame, `(n, 4, 4)` per node, `(F, n, 4, 4)` frame for frame |
| `frames.matrix = L` | the same, local matrices |
| `frames.quaternion = Q` | `(4,)`, `(n, 4)` or `(F, n, 4)`; reading it gives `(F, n, 4)` |
| `frames.match_matrix(other)` | a clip or a frame axis, frame for frame; a node, a list of nodes or a pose, on every frame |
| `frames.match_translate(other, x=, y=, z=)` | the same, or positions `(3,)`, `(n, 3)`, `(F, n, 3)` |
| `frames.match_rotate(other)` | the same, or eulers in degrees `(..., 3)` in each node's rotate order, or quaternions `(..., 4)` |

```python
take               = ClipData(make_chain(("root", "hip", "knee", "ankle")), frames=5)
take.frames.rotate = np.random.default_rng(1).normal(size=(5, 4, 3)) * 30.0

clip.frames.world_matrix = take.frames.world_matrix    # every frame, one solve
assert np.allclose(clip.frames.world_matrix, take.frames.world_matrix, atol=1e-9)

clip.frames["knee"].match_rotate([0.0, 0.0, 45.0])     # held on every frame
clip.frames.match_translate(take, y=False)             # paired frame for frame
clip.frames.quaternion.shape                           # (5, 4, 4)
```

The clip's own setters, `clip.world_matrix = W`, `clip.match_rotate(...)`
and `clip["hip"].quaternion = q` among them, write the frame the clip is
bound to, as every channel setter does. A loop over the frames picks each
frame's euler branch on its own; the frame axis filters the curve.

```python
spin                        = ClipData(make_chain(), frames=7)
spin.frames["hip"].rotate_x = np.linspace(0.0, 540.0, 7)   # 90 degrees a frame
W                           = np.array(spin.frames.world_matrix)

looped = ClipData(make_chain(), frames=7)
for frame in range(7):
    looped.frame        = frame
    looped.world_matrix = W[frame]                   # this frame alone
looped.frames["hip"].rotate_x[:, 0]                  # [0, 90, 180, -90, 0, 90, 180]

solved                     = ClipData(make_chain(), frames=7)
solved.frames.world_matrix = W
assert np.allclose(solved.frames["hip"].rotate_x[:, 0], np.linspace(0.0, 540.0, 7))
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

Every clip delta, taken or applied, against one pose or another clip, is
solved for all its frames at once, with no copy of the rig per frame:
1000 frames of a 271-joint rig take 0.5 to 1.4 s, the every-frame
in-place forms included.

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

As everywhere else on a clip, the clip itself stands for the frame it is
bound to and `clip.frames` for every frame. The operators follow that:
`clip -= pose` and `clip += pose` solve the bound frame alone, as they
always have, so a loop that scrubs and writes still writes each frame
once; `clip.frames -= pose` and `clip.frames += pose` solve every frame
at once and keep the clip on its frame. A single pose against a clip
reads the clip's bound frame -- `pose.get_delta(clip)`, `pose - clip`,
`rig.add_delta(deltas)`, `deltas + rig` -- and against its frame axis it
covers every frame: `pose.get_delta(clip.frames)`, `pose - clip.frames`,
`rig.add_delta(deltas.frames)` and `deltas.frames + rig` return a
`ClipData` on the clip's timebase, bound to its first frame. A clip's
own `get_delta` / `add_delta` (and `clip - pose`) cover every frame, as
above. A clip of one frame is one pose wherever it stands. Each rotate
the algebra rebuilds from matrices goes through `euler_filter` per node
in its rotate order, frame 0 kept as solved, so a curve turning through
180 degrees comes back continuous rather than stepping by 360; a node
the pairing leaves out keeps its curves as they were.

```python
per_frame = rest.get_delta(clip.frames)         # one pose against each frame
assert isinstance(per_frame, ClipData) and per_frame.frame_count == 4

delta = clip.copy()
delta.frames -= rest                            # in place, every frame
assert np.allclose(delta.frames.rotate, clip.get_delta(rest).frames.rotate)

delta.frames += rest                            # rest + (clip - rest): the clip
assert np.allclose(delta.frames.world_matrix, clip.frames.world_matrix, atol=1e-6)

posed = rest.add_delta(clip.get_delta(rest).frames)  # a clip of deltas poses one rig
assert isinstance(posed, ClipData) and posed.frame_count == 4

clip.frame = 2
bound      = clip.copy()
bound -= rest                                   # the bound frame, frame 2, alone
assert np.allclose(bound.frames.rotate[0], clip.frames.rotate[0])
```

---

## Rig edits on a clip

The rig-level edits rebuild channels out of matrices: `swapaxes`,
`set_rotate_to_joint_orient`, `set_joint_orient_to_rotate`,
`set_parent(world_space=True)`, a joint becoming another type
(`node_type`, `set_node_type`) and `set_rotate_order`. On a clip every
frame is solved at once, with the result the single-pose method gives
frame by frame, on the whole clip, a selection or one node.

Each changes what every frame shares -- the parenting, the rotate order,
the node type, a joint's axes -- so it covers every frame whether it is
called on the clip, a selection of it or one node, and the clip stays on
the frame it was on. A channel write or a solve is the other kind: on
the clip or its nodes (`clip["hip"].world_matrix = M`, `match_*`) it
writes the frame the clip is on, and on `clip.frames` every frame.

Solved a frame at a time, each frame would pick its own euler branch:
the poses stay exact while the curves step by 180° and 360°. So every
rebuilt euler curve -- `rotate` in the node's rotate order,
`rotate_axis` and `joint_orient` in xyz -- goes through `euler_filter`,
frame 0 kept as computed, the way Maya's Euler Filter keeps the first
key.

```python
spin                        = ClipData(make_chain(), frames=60)
spin.frames["hip"].rotate_x = np.linspace(3.0, 357.0, 60)  # one turn, crossing 180
spin["knee"].set_parent("root")                 # every frame keeps its world
spin.frames["knee"].rotate_x.ravel()[[0, 30, 59]].round(1).tolist()
# [3.0, 183.0, 357.0] -- one smooth curve, no 360 degree step

spin.set_rotate_order("zyx")
assert np.abs(np.diff(spin.frames.rotate, axis=0)).max() < 10.0
```

A swap keeps every child where it was on every frame:

```python
spin                        = ClipData(make_chain(), frames=60)
spin.frames["hip"].rotate_x = np.linspace(3.0, 357.0, 60)
before                      = spin.frames.world_matrix.copy()
spin["hip"].swapaxes(0, 1, negate=True)
assert np.allclose(spin.frames.world_matrix[:, 2], before[:, 2])   # knee
```

`set_parent(world_space=False)` writes no pose, so it moves nothing but
the parenting.

The frame axis has no rig edits of its own: `clip.frames.swapaxes`,
`clip.frames["hip"].set_parent` and the rest raise, pointing at the clip
form, which already covers every frame. A loop of one-node edits costs
the nodes it edits, not the clip each time.

```python
spin.set_parent(None, world_space=False)
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

Each node is written after its parent, whatever order the list holds them
in: a rig sorted, or built child first, writes the same tree. A node whose
parent the list leaves out is a root of the file, and siblings, the roots
among them, keep the list's order.

```python
by_name = make_chain()
by_name.sort()                                  # hip, knee, root: children first
by_name.save_fbx(path)
assert HierarchyData.load_fbx(path).name == ["root", "hip", "knee"]

by_name.match("knee", "root").save_fbx(path)    # knee's parent left out: a root
assert HierarchyData.load_fbx(path).name == ["knee", "root"]
```

User attributes make the trip too: numbers, bools, enums, strings,
stringArrays and `double3` or `float3` vectors come back as the specs they
were, binary or `as_ascii=True`. Arrays such as `doubleArray`, matrices,
multis and other compounds are skipped with a logged warning. Only a
string's value must be plain ASCII, the only text the fbx sdk takes: one
that is not raises `ValueError` before anything is written.

```python
rig["hip"].add_user_attribute("twist", 0.5, attribute_type="float")
rig["hip"].add_user_attribute("tags", ["deform", "L"])
rig["hip"].add_user_attribute("side", "gauche")
rig.save_fbx(path)

hip = HierarchyData.load_fbx(path)["hip"]
assert (hip.twist, hip.tags, hip.side) == (0.5, ["deform", "L"], "gauche")
assert hip.user_defined_attributes == rig["hip"].user_defined_attributes

rig["hip"].side = "gauche \u00e9"
try:
    rig.save_fbx(path)                          # path keeps the file it had
except ValueError as error:
    print(str(error)[:43])                      # user defined attr hip.side can't be written
rig["hip"].side = "gauche"

arrays = make_chain()
arrays["hip"].add_user_attribute("weights", [0.5, 0.25])   # a doubleArray
arrays.save_fbx(os.path.join(WORKDIR, "arrays.fbx"))       # logs: skipped ... 1 doubleArray
back = HierarchyData.load_fbx(os.path.join(WORKDIR, "arrays.fbx"))
assert "weights" not in back["hip"].user_defined_attributes
```

Reading the same file as an animation gives a clip, and with it the
file's static pose, `clip.rest_pose`.

```python
clip = ClipData.load_fbx(path)
assert isinstance(clip, ClipData) and clip.frame_count == 1
assert clip.rest_pose.name == clip.name == HierarchyData.load_fbx(path).name
```

`ClipData.load_fbx` reads each translate, rotate and scale curve's keys
once and evaluates every frame from them at once, as the SDK evaluates a
curve. On a key, so on every frame of a take baked on its frames, the
values are the SDK's bit for bit. Between keys the SDK works in single
precision, so there the two agree to its rounding, a few times 1e-6 of
the curve's range at most, but on a weighted span whose two handles
crowd one end: one weight at most about 0.01 and the other about 0.88 or
more, the smaller the one, the lower the other can be (the SDK holds a
weight to 0.0001 at least and 0.99 at most). There the SDK's own solve
for the bezier's time stops short of the exact one the loader takes, by
up to 2e-5 of the curve's range at weights 0.01 and 0.99, 3e-5 at 0.0001
and 0.88, 4e-4 at 0.0001 and 0.89, 3e-3 at 0.0001 and 0.9 and 0.55 at
0.0001 and 0.99. Maya's own import of such a span gives the loader's
values, to a few times 1e-15 of the range: on a crowded span the loader
follows Maya, not the SDK. A curve that repeats or mirrors its keys
before or after them is read where that sends each frame, for as many
periods as its count says, then holds its end key, as the SDK does; one
that repeats with an offset (Maya's cycle with offset, which the
bindings leave unnamed) climbs by its last key less its first each
period, and holds its end key shifted as many times. Keep slope holds
its end key too, unless its count is 2**32 - 1, the count Maya writes
for every infinity it exports (linear infinity is keep slope): then it
carries the end key's slope on, as this SDK evaluates it. One evaluation
of the curve past its keys checks that, and an SDK that evaluates it
otherwise raises ValueError. An extrapolation number the SDK does not
know (a file carries one) reads 0 for as many periods as the count says,
and an interpolation number it does not know 0 up to the next key, as
the SDK reads them.

Animation layers are blended as the SDK blends them, every frame at
once: from the first layer that is not muted and holds anything (a
channel it holds no curve node for starts from 0, scaling too, under a
later layer that holds it; a channel no live layer holds keeps the rest
pose), each later layer that is not muted, by its weight (its own weight
curve, if it holds one) and blend mode. Translation, rotation blended by
channel (angle by angle) and added scaling: additive adds weight x
value, override takes weight x value, passthrough mixes the two.
Rotation blended by layer (the default) goes through quaternions, and
there override takes the layer's rotation whatever the weight.
Multiplied scaling (the default): additive |below| x |value| ** weight,
override |value| ** weight, passthrough |below| ** (1 - weight) x
|value| ** weight, every term unsigned. With every curve keyed on the
frames, weights included, translation and scaling are the SDK's bit for
bit and rotation within 1e-9 degrees of it, or of it through
`euler_filter` where a rotation blended by layer is filtered (below);
within about 3e-3 degrees of a middle angle of +-90 that becomes a few
times 1e-16 / cos(middle angle) radians, the rounding both lose reading
the angles back so near gimbal. Within about 1e-15 of the cosine the SDK
splits the turn at (gimbal: the last angle 0, the first the whole turn),
each side's rounding picks the branch, and the two can be whole
rotations apart. No frame is asked of the SDK.

Maya's own evaluation of a take differs from the SDK's, and so from the
loader's, where:

- a rotation is blended by layer on a node whose rotate order is not
  xyz: the SDK hands the x, y and z angles to its euler code as they
  are, so an even order (yzx, zxy) blends as an xyz node does and an odd
  one (xzy, yxz, zyx) as the SDK blends a zyx node, whose additive layer
  turns on the other side of what is below: M(below) @ M(layer), both
  matrices of zyx angles, where Maya gives M(layer) @ M(below) in the
  node's own order. Maya blends every node in its own order the way it
  blends xyz. Additive layers agree only on xyz (but for the sign cases
  below), passthrough layers on xyz and zyx; the others are up to 2.0
  apart as matrices;
- an additive layer blends a rotation by layer at a weight other than 0
  or 100, and the quaternion Maya builds from the layer's angles (qz qy
  qx of the half angles on an xyz node) has the other sign from the one
  the SDK solves from its matrix: the scaled turn goes round the other
  way (up to 2.0 apart as matrices, in every rotate order, xyz
  included). The SDK's w is positive on every turn under 120 degrees, so
  such a turn parts whenever Maya's w is negative: an angle keyed past
  180 (a 20 degree turn keyed as (380, 0, 0) is 1.9 apart at weight 50)
  or all three angles near +-180 (a 40 degree turn Maya keyed as (150.4,
  154.1, -176.7) is 1.5 apart). Over 120 degrees either sign can come
  out;
- a rotation is overridden by layer at a weight under 100: the SDK takes
  the layer's rotation whatever the weight, and Maya weighs it (0.86
  apart as matrices at weight 50);
- a negative scale is multiplied by a layer: the SDK takes every term
  unsigned, and Maya keeps the sign (1.6 apart in scale on Maya's own
  export);
- the first live layer holds no curve node for a channel a later layer
  holds: the SDK starts it from 0, and Maya does not (0.82 units apart
  in translation, 1.94 in scale);
- a layer is soloed: solo changes nothing in the SDK, and Maya honours
  it;
- the base layer is muted: the SDK skips it, but Maya's exporter does
  not write that mute, so a Maya take whose base layer is muted loads
  with it playing;
- a curve has weighted tangents: Maya's exporter does not write them as
  the scene holds them, so between keys a Maya export loads up to about
  2.5e-2 of the curve's range off what Maya showed (over 5e-3 on 7 of 72
  test exports, whatever the infinity); Maya's own import of the file
  gives the loader's values;
- the curve has linear infinity: past the keys the SDK carries on along
  the end key's derivative as the file holds it, where Maya follows the
  key's outer tangent. Unweighted spline, clamped, flat, stepped,
  plateau and auto tangents agree; linear, fixed and every weighted
  tangent do not (up to 158 units 48 frames out on Maya's own exports).
  A weighted one on no layer is the export again: Maya's own import of
  it carries on past the keys as the loader does, but for linear and
  stepped tangents.

| Loader | Signature |
|---|---|
| `TransformList.load` | `(filename, mode=None)` — `mode` in `"gltf" / "glb" / "fbx" / "json" / "npz" / "pkl"` |
| `TransformList.load_fbx` | `(filename, scale_factor=None, axis_system="maya_y_up", pose="node", node_types=None)` |
| `TransformList.load_glb` | `(filename, scale_factor=100.0, pose_frame=None)` |
| `ClipData.load_fbx` | `(filename, scale_factor=None, take=None, fps=None, start_frame=None, end_frame=None, axis_system="maya_y_up", node_types=None)` |
| `ClipData.load_glb` | `(filename, scale_factor=100.0, animation=0, fps=24.0, start_frame=0)` |
| `TransformList.save_fbx` | `(filename, zero_root=False, as_ascii=False)` — binary unless `as_ascii` |

`zero_root=True` writes the root joint with an identity transform, so the
file carries the skeleton at the origin. The root joint is the first joint
the file writes as a root: one at the world, or the top joint of a view,
whose parent the view leaves out, as its `copy()` roots it. Meshes and
locators at the world are left as they are; a skeleton with no root joint
raises `ValueError`.

```python
make_chain().save_fbx(os.path.join(WORKDIR, "flat.fbx"), zero_root=True)

make_chain().match("hip", "knee").save_fbx(os.path.join(WORKDIR, "legs.fbx"), zero_root=True)
legs = HierarchyData.load_fbx(os.path.join(WORKDIR, "legs.fbx"))
assert legs.name == ["hip", "knee"] and np.allclose(legs["hip"].translate, 0.0)
```

`save_fbx` writes `rotate_axis` (and `joint_orient`) for joints only. A
transform or locator holding a `rotate_axis` loses it in the file, so it
reloads off by that rotation. A matrix write, `set_parent()`,
`match_translate()` and `match_matrix()` all keep a node's `rotate_axis`.
Fold it into `rotate` before saving: read `matrix`, zero `rotate_axis`,
write `matrix` back.

```python
held = HierarchyData([
    TransformData("arm", node_type="joint", rotate=(30.0, 0.0, 0.0)),
    TransformData("prop", rotate_axis=(0.0, 0.0, 20.0), rotate=(10.0, 20.0, 30.0)),
])
held["prop"].set_parent("arm")                  # keeps rotate_axis
world = held["prop"].world_matrix

local                    = held["prop"].matrix
held["prop"].rotate_axis = (0.0, 0.0, 0.0)      # fold it into rotate
held["prop"].matrix      = local
held.save_fbx(os.path.join(WORKDIR, "prop.fbx"))

back = HierarchyData.load_fbx(os.path.join(WORKDIR, "prop.fbx"))
assert np.allclose(back["prop"].world_matrix, world)
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

For `load_fbx`, `scale_factor=None` converts the file's unit to
centimetres: translations, static and keyed, are multiplied by the
centimetres one of the file's units holds (100.0 for a file in metres,
1.0 for one in cm); rotations and scales keep the file's values. A
number is the caller's own factor, and the file's unit is not read.
A file whose unit the SDK's `FbxSystemUnit.ConvertScene` or Maya's
unit-converting export wrote, as a 1/s scale on the scene root's
children over values left in the old unit, is not converted: a metres
one's nodes load where the file puts them but at world scale 0.01,
whether the read keeps the node holding that scale or folds it into the
nodes under it (`node_types` below), and its meshes 100 times too large.
The FBX mesh readers (`geometry.mesh.load_fbx`, `MeshData.load_fbx`,
`MeshList.load_fbx`) and `render.Object.load_fbx`, points and skin bind
matrices alike, take the same `scale_factor`, and the `axis_system`
below, with the same defaults, so a mesh posed by a rig or a clip read
from the same file lines up.

`axis_system="maya_y_up"`, the default, loads an FBX in Maya's Y-up
axes the way `FbxAxisSystem.MayaYUp.ConvertScene` converts it: every
node's world matrix, on every frame of a clip, is the file's times the
turn from the file's axis system (its up vector, front vector and
handedness) to Maya's. Maya's FBX importer puts a file whose up axis is
X or Z, an Unreal file among them, in the same place. It leaves a file
whose up axis is Y, pointing up or down, in its own axes, so a Y-up file
whose front is not +Z, or a -Y-up one, loads turned 90 or 180 degrees
from where Maya shows it. An Unreal file, Z-up, turns (x, y, z) to
(x, z, -y), root travel with it. On a root an additive layer moves, a
clip still keeps the file's world turned, where ConvertScene leaves that
layer's translate in the file's axes and Maya's importer parts from
both. The turn goes on the roots: a joint
root takes it in its joint orient and translate and keeps its rotate
curves; any other root's rotate is solved again, through `euler_filter`
on a clip. The mesh readers turn the points and normals, and
`Object.load_fbx` the skin's bind matrices with them. A file already in
Maya's Y-up, as `save_fbx` writes, loads as it is; a left-handed one
raises `ValueError`, since a rotation can't turn it into Maya's axes.
`axis_system=None` keeps the file's own axes. `load()` reads an FBX with
the `load_fbx` defaults and takes no `axis_system`, so a left-handed file
reads through `load_fbx(..., axis_system=None)` alone. Which way the
character faces stays as the file has it.

`HierarchyData.load_fbx(..., pose="bind")` reads a skinned file's bind
pose rather than its node values, which hold whatever pose the scene was
in at export. Every node the read keeps that a bind pose (an `FbxPose`
whose `IsBindPose()` is true) lists takes that pose's matrix as its world
matrix, in cm and Maya's axes as above; a node no bind pose lists keeps
its own values under its parent, so it follows the parent's bind. The
worlds are written parents first, each solved into scale, rotate (in the
node's rotate order) and translate, the joint orient and rotate axis
kept, and a node its bind leaves where it is keeps its own numbers: a
file at its bind pose reads the same either way. When two bind poses
list one node the first wins, with a warning naming the node when a
later one is more than 1e-6 off it. A file with no bind pose listing a
node the read keeps raises `ValueError`. `pose="node"`, the default,
reads the node values.

`node_types` keeps the kinds of node it names, `"joint"`, `"locator"`,
`"transform"` (a group) and `"mesh"` (read as a transform), and `None`,
the default, all four: `node_types=("joint",)` reads a skeleton without
its meshes and groups. A node of another kind (a camera, a light, an LOD
group), one with no node attribute and the scene root are always
dropped, and a kept node goes under its nearest kept ancestor. What the
dropped nodes between them hold is folded into its channels, on every
frame of a clip, so its world matrix stays the file's: a joint takes
their turn in its joint orient and their move in its translate, keeping
its rotate curves; any other node has its rotate solved again. A uniform
scale among them goes on the node's scale, and on each joint under it
that compensates its parent's scale; an uneven one under a node turned
in it would need shear, which no node holds: that node keeps the file's
position and axis lengths, and the nodes under it take the shear it
drops, so they stay where the file puts them. A zero scale is not
divided out, as Maya's importer leaves it, so what sits under a node
scaled to 0 collapses onto it. A node's own pivots and offsets go into
its translate, and the pre and post rotation of a node that is no joint
into its rotate, so a group whose pivot Maya centred on its children
loads where Maya shows it. A light's post rotation, FBX's view axis, is
left out, as Maya places its children; a camera's turns its children by
what it adds to the view axis Maya's exporter writes, (0, -90, 0), which
Maya's importer keeps as the camera's rotate axis. Under `pose="bind"`
a dropped node a bind pose lists is folded in at its bind, so each node
kept sits where the full bind read puts it. A file whose dropped nodes,
pivots and offsets hold the identity reads as it holds its nodes, to the
bit.

```python
grouped = HierarchyData([
    TransformData("grp", translate=(10.0, 20.0, 30.0), rotate=(0.0, 0.0, 45.0)),
    TransformData("hip", node_type="joint", translate=(0.0, 5.0, 0.0)),
    TransformData("knee", node_type="joint", translate=(0.0, -4.0, 0.0)),
])
grouped["hip"].set_parent("grp", world_space=False)
grouped["knee"].set_parent("hip", world_space=False)
grouped.save_fbx(os.path.join(WORKDIR, "grouped.fbx"))

joints = HierarchyData.load_fbx(os.path.join(WORKDIR, "grouped.fbx"), node_types=("joint",))
assert joints.name == ["hip", "knee"] and joints["hip"].parent_uuid is None
assert np.allclose(joints["hip"].joint_orient, (0.0, 0.0, 45.0))   # the group's turn
assert np.allclose(joints["knee"].world_matrix, grouped["knee"].world_matrix)
```

For `load_glb`, `scale_factor` converts units (glTF is metres, so `100.0`
gives centimetres); `pose_frame` reads one frame of the file's animation
onto the rest pose. A node holding a mesh is not read as a joint, whatever
its name, so a namespaced `ns:body` mesh neither loads as a joint nor
hides a joint named `ns`. Nor does a mesh named like a joint lend that
joint its animation: `pose_frame` and `ClipData.load_glb` read the joint's
own channels, and a skin bound to the joint loads.

glTF keys every property on its own timeline, so `load_glb` resamples
the samplers onto a uniform grid at `fps` and runs the result through
`euler_filter`. `load_fbx` does neither for the curves a file holds —
fbx already stores euler curves, and rewriting their branches would
change the animation. A rotation an animation layer blends by layer is
rebuilt from quaternions frame by frame, by the SDK as here, so that
curve, and only that one, goes through `euler_filter` in the node's
rotate order, frame 0 as the SDK gives it. A file with no animation
gives a one-frame clip rather than an error.

`ClipData.load_fbx` defaults to the first take that was actually keyed,
not to index 0, because a file can hold takes with no curves at all.

Its frames default to the take's stored span and every key of the take
together, out to whole frames: an Unreal take, whose span stops a frame
short of its last key, keeps that key, and a span wider than its keys
keeps its frames. A time within 1e-6 of a frame of a whole frame is that
frame, so a span of frames 60-121 reads 62 frames. `start_frame` and
`end_frame` override either end, and each end left out defaults on its
own: a `start_frame` passed alone no longer moves the end, which stays
the take's last frame (cgmath 2 kept the take's length instead), and one
past that frame raises `ValueError` naming it.

`clip.rest_pose` is the pose the file holds outside its take, the nodes'
own values: a `HierarchyData` with the clip's names and node order, taken
from the same read and loaded as `HierarchyData.load_fbx` loads the file,
in the same unit and axes. For an Unreal animation it is the skeleton's
bind pose, which frame 0 can be far from. It is saved with the clip, a
`copy()` or a cut of frames (`clip.frames[10:20]`) holds a copy of it, and
`strip_namespace()` on the clip strips it too. A clip the constructor,
`from_poses`, `load_glb` or the pose algebra (`get_delta`, `add_delta`)
builds holds `None`.

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

A file names its class, and a class's `load` gives back that class: a
file holding any other raises `TypeError`, a subclass too, so
`HierarchyData.load` refuses a clip. `cgmath.hierarchy.load` gives back
whatever hierarchy the file holds, and `Data.load` whatever class. A file
saved by an older cgmath names no class: it raises `LegacyFileError` (a
`ValueError`) and no longer loads.

```python
from cgmath import hierarchy
from cgmath.geometry import Data

assert type(hierarchy.load(path)) is ClipData       # the file says clip
assert type(Data.load(path)) is ClipData

try:
    HierarchyData.load(path)
except TypeError as error:
    print(error)                                # .../clip.json holds a ClipData; load it with ClipData.load() or Data.load()

try:
    ClipData.load(os.path.join(WORKDIR, "rig.json"))
except TypeError as error:
    print(error)                                # .../rig.json holds a HierarchyData; load it with HierarchyData.load() or Data.load()
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

pkl, npz and json all save the one tree `to_dict` returns. A list's tree
is `{"__class__": ..., "__items__": [...]}`, its nodes' trees in order (so
duplicate names are safe). A node's tree holds its class and only the fields that
differ from their default; a field left out loads back as the default. A
channel is saved under its property's name, `translate`, though the node
keeps it in the private field `_translate`. A load takes either name, so a
file that names `_translate` still loads; one naming both raises
`ValueError`.

```python
tree = rig.to_dict()
tree["__class__"]                               # 'cgmath.hierarchy.hierarchy.HierarchyData'
sorted(tree["__items__"][0])                    # root: everything else is default
# ['__class__', 'name', 'node_type', 'uuid']
sorted(tree["__items__"][1])
# ['__class__', 'name', 'node_type', 'parent_uuid', 'translate', 'uuid']
```

A `ClipData` saves its timebase beside the items, named as its properties:
`start_frame`, `fps`, `frame` (the loaded frame, which a load scrubs back
to), and `blocks`, one `(F, N, 3)` array per framed channel, each keyed by
the channel's property. A clip holding a `rest_pose` saves it as well,
under `rest_pose`, as a hierarchy's tree; one holding none saves no such
entry.

```python
sorted(clip.to_dict())
# ['__class__', '__items__', 'blocks', 'fps', 'frame', 'start_frame']
clip.to_dict()["blocks"]["translate"].shape     # (4, 3, 3)
```

`copy()` is deep and owning, and its nodes keep their uuids. `==` on a
hierarchy compares its nodes by uuid, not their channels. On a clip it
also compares the timebase, the loaded frame and every block by value,
and the rest pose as a hierarchy compares: whether there is one, and its
nodes' uuids.

```python
moved               = rig.copy()
moved["hip"].rotate = (0.0, 90.0, 0.0)
assert moved == rig                             # the same nodes, by uuid

twin = clip.copy()
assert twin == clip
twin.frame = 2
assert twin != clip                             # another frame is loaded
twin.frame = clip.frame
twin.frames.rotate = np.full((4, 3, 3), 45.0)
assert twin != clip

twin           = clip.copy()
twin.rest_pose = moved
assert twin != clip                             # a rest pose, against none
other                         = twin.copy()
other.rest_pose["hip"].rotate = (0.0, 0.0, 45.0)
assert other == twin                            # its nodes, by uuid
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
| `parent_uuid = "hip"` | a name raises `ValueError`: `parent_uuid` holds a uuid or `None` | `set_parent("hip")`, which keeps the uuid |
| a node appended with a `parent_uuid` the rig does not hold | it joins as a root (`parent_uuid = None`), and the parent appended after does not take it back | add the nodes together (`extend`, the constructor), or `set_parent` once both are in |
| `parent_node` | `parent_uuid`'s deprecated name: it warns, and goes in 3.0.0 | `parent_uuid` |
| `world_matrix` as a call | it is a **property** on every type here | `node.world_matrix`, not `node.world_matrix()` |
| mutating a channel getter | `node.rotate[0] = 5` writes to a throwaway copy | read, edit, assign back |
| `clip.frames.rotate += 5` | the block getter is read-only, so `+=` raises | `np.array(...)` then assign, or use `rotate_y += 5` |
| `rig + delta` | `+` only reads the delta from its **left** operand | `delta + rig`, or `rig.add_delta(delta)` |
| mixing `translate=` between calls | `get_delta` and `add_delta` store different things in the one channel | pass the same value to both |
| `by="name"` with a partial overlap | accepted silently; only a completely empty intersection raises | check `len(delta)` when it matters |
| `by="index"` on mirrored limbs | both arms share a parent index, so left can pair to right | use `by="name"` when sibling order is not guaranteed |
| `reset_cached_data()` | clears every non-annotated attribute — including the `_hierarchy` back-pointer, which detaches the nodes | re-append the nodes, or just write a channel to drop the world-matrix cache |
| `ClipData(rig)` with no `frames=` | `frame_count == 0`; it is a plain hierarchy until frames are allocated | pass `frames=N`, or build with `from_poses` |
| `rotate_order = n` on a posed joint | the angles stay and read in the new order, so the joint turns, as Maya's `setAttr` does | `set_rotate_order(n)` keeps the pose |
| a file's user attribute named `tx` (or any short name) | it loads, with one warning a name, but `node.tx` is `translate_x`, which hides it | read it as `node.user_defined_attributes["tx"]["value"]` |
| `node.is` | Maya's short name for `inverseScale` is a Python keyword: a `SyntaxError` | `node.parent_scale_inverse`, the parent's scale: what Maya's `inverseScale` holds where Maya wires it from the parent joint |
| a compensating joint Maya did not wire from its parent joint: under a transform, or under a joint after `cmds.createNode` with `parent=` or `cmds.parent` with `relative=True` | `parent_scale_inverse` is the parent's scale, and `matrix` divides by it; Maya wires `inverseScale` from a joint's scale alone, as `cmds.joint` does under a joint and an absolute `cmds.parent` that leaves the joint right under one: a joint made with `cmds.createNode`, or drawn under a transform, holds `(1, 1, 1)`, one moved with `cmds.parent` the scale of the parent joint it left, unwired, under that transform or under one Maya inserts below it, or with `relative=True` still wired from that joint, and Maya's `m` and `wm` divide by what it holds | `segment_scale_compensate = False` on that joint gives Maya's matrices where Maya's `getAttr <joint>.is` reads `(1, 1, 1)` |
| `clip.frames[12]` | scrubs and hands back the **same** clip — two frames cannot be held at once | `clip.frames[12:13]` for an independent cut, or `copy()` |
| `clip.frames[a:b] -= pose` (or `+=`, or `[:]`) | the slice is an independent cut bound to its own first frame, and `-=` on a clip solves its bound frame: the paste writes back frame `a` alone, with no error | `clip.frames -= pose` for every frame; for a range, `cut = clip.frames[a:b]`, `cut.frames -= pose`, `clip.frames[a:b] = cut` |
