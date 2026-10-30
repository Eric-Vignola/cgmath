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
| [Parenting: `parent_uuid`](#parenting-parent_uuid) | the parent's uuid, `parent_name`, `set_parent` by name, joining a rig, older files |
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

`parent_node`, `parent_uuid`'s name before cgmath 2.0.0, went in 3.0.0.
A node has no such attribute: a read or a write raises `AttributeError`,
as any name the class does not declare does, the constructor keyword
raises `TypeError`, through an `Object`'s, a `Camera`'s or a `Light`'s
constructor too, and a `to_attributes` mapping naming it maps nothing, as
a public name that is no field does. A file saved before 2.0.0 names its
parents under `"parent_node"` and still loads, a uuid as it is and a name
as the first node of that name in the loaded rig (a name no node carries
leaves a root); it saves back as `"parent_uuid"`.

```python
try:
    rig["knee"].parent_node
except AttributeError as error:
    print(error)    # 'TransformData' object has no attribute 'parent_node'
assert rig["knee"].to_attributes({"parent_node": "parent"}) == {}

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
| `clip.frames.translate_y` | `(F, N)` one component, a copy |
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

Each framed channel has `_x`, `_y`, `_z`, shaped `(F, N)`. A read is a
copy, a new array every time: `+=` works on it whatever the clip and the
scope, one frame of one node included, and writing into it changes
nothing until it is assigned back.

```python
clip.frames.translate_y.shape                   # (6, 3)
clip.frames.translate_y = np.zeros((6, 3))
assert np.allclose(clip.frames.translate[..., 1], 0.0)

one = ClipData(make_chain(), frames=1)
one.frames["knee"].rotate_y += 5.0              # one frame of one node
assert one["knee"].rotate_y == 5.0
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
  xyz, as the file stores it, rotation active or not, though a node
  whose rotation is inactive loads with `rotate_order` 0 (a root of the
  file whose rotation is inactive, kept or dropped, blends as an xyz
  node in a file the loader turns into Maya's axes, as ConvertScene
  turns its rotation on in xyz; a node kept under a dropped root is no
  root of the file): the SDK hands the x, y and z angles to its euler
  code as they are, so an even order (yzx, zxy) blends as an xyz node
  does and an odd one (xzy, yxz, zyx) as the SDK blends a zyx node,
  whose additive layer turns on the other side of what is below:
  M(below) @ M(layer), both matrices of zyx angles, where Maya gives
  M(layer) @ M(below) in the node's own order. Maya blends every node in
  its own order the way it blends xyz. Additive layers agree only on xyz
  (but for the sign cases below), passthrough layers on xyz and zyx; the
  others are up to 2.0 apart as matrices;
- Maya's FBXImport merges the layers, as it does by default, on a node
  that turns in xzy, yzx, yxz or zxy, a group or a joint, root or not
  (but a root of the next item), in a Y-up and a Z-up file alike: on
  some frames, by channel or by layer, the merged turn parts from the
  SDK's, up to 2.0 apart as matrices. By channel such a frame holds the
  SDK's angles with x and z turned by 180 degrees and y by 180 - y, the
  same turn in xyz and zyx alone (a frame the SDK turns by
  (-100, 68, -25), after one at (120, -54, 205), merges to
  (80, 112, 155), 1.77 to 1.99 apart), where the layers kept agree; by
  layer, keys of 120 to 128 degrees about y part on every frame, 1.23 to
  1.99 apart. Merged, a node in xyz or zyx agrees, and so did every
  order on the smaller turns measured, by layer too, which kept parts as
  the item above says, but for the single precision values the curves it
  merges, bakes or solves hold, the rotate's as well as the scale's and
  the translate's, which part from the SDK's the more the larger they
  are. The angles that count are, by channel, the sums of the keys,
  which Maya rounds to single precision, or the angles its curve holds
  where those are larger: it starts the curve on the take's first frame
  at the angles nearest 0 that turn the node as the sums do, whole turns
  off and on either branch (sums of (330, 89.7, 330) held as
  (-30, 89.7, -30), of (3500, 89.7, 3500) as (80, 90.3, 80)), and its
  continuity can carry the curve on past the sums where the keys jump
  (random keys summing to under 245 degrees: curves holding up to 1210);
  by layer, and where Maya solves the rotate (a root of the next item, a
  node whose pre rotation it bakes), the angles it solves, and more
  where that continuity carries them on. On the files measured, as
  matrices on a node of world scale 1, within 3.7e-7 on angles up to 250
  degrees, 5.7e-7 up to 500, 1.3e-6 up to 1000 and 4.3e-6 up to 3600, by
  channel within the single precision spacing of the largest such angle
  in radians (2.642e-7 on sums of 205 degrees and 4.228e-6 on sums of
  3505, 0.992 of it, where single precision rounds the x and z sums by
  almost half of it each near y = 90), and within 2.2e-15 on keys in
  whole degrees, whose sums single precision holds. A scale on the node
  or above it multiplies that, on the files measured, by up to the
  largest axis of the node's world scale (scales of 2 to 10 and
  (1, 2, 3): a node scaled 3.5, or under one scaled 3.5, 7.0e-7 by
  channel and 8.2e-7 by layer on angles up to 250; one whose scale is
  keyed to 10, 9.0e-7 on angles to 193); a node under it, and with a
  rotate pivot the node itself, is off in translate by about that gap at
  scale 1 times its world distance from the node's rotate pivot, up to
  1.7 times that on the files measured and 2.9 under a group scaled
  (1, 2, 3) (a locator 19.7 from a node at scale 1, 3.7e-6 off on angles
  up to 250, the one under the node keyed to 10, 1.7e-6); a translate
  both layers key is up to 6.0e-8 (2^-24) of its size off (3.8e-6 at
  71). But a frame whose blended y angle is 90 or -90, or a whole number
  of turns from either (Maya holds 270 as -90 and -270 as 90), merges by
  channel to angles that hold its turn in xyz alone, x - z kept at 90
  and x + z at -90: those nearest the key before's (the frame before's
  at the file's rate), and on a take's first frame, which has none
  before it, those nearest 0 ((25, 90, 30) merges to (-2.5, 90, 2.5)).
  On the files measured, y from -810 to 810, 8.5e-3 to 0.89 apart in
  xzy, yzx, yxz and zxy and 1.4e-2 to 1.4 in zyx, the more the further x
  and z move, and up to 2.0 where the merged angles also sit a half turn
  off the keys as above, x and z turned by 180 degrees and y by 180 - y,
  as the frames after such a first frame can go on doing (a take whose
  first frame, (-150, -90, 170), merges to (10, -90, 10): 1.58 to 1.97
  apart in those four orders on every frame). A node that is no joint
  with a pre rotation parts as the item on it below says;
- a layer blends the rotation of a root that is no joint in a file whose
  up axis is X or Z, which Maya's importer turns into its Y-up: it can
  part from ConvertScene there in any rotate order, by channel or by
  layer, the layers kept or merged, up to 2.0 apart as matrices (Maya's
  own Z-up export of a group root in xyz, its default order, under an
  additive animLayer, which it writes by channel: 1.41 apart, the
  locator under it 2.25 off). On files the SDK wrote, by channel 0.40 to
  1.96 apart, a root whose rotation is inactive as a root in xyz, which
  ConvertScene makes it, in orders that hang on the file's axes and on
  the turns: on the smaller turns measured and on a turn of 120 to 128
  degrees about y, xyz, xzy, zxy and yxz in Maya's Z-up axes and zyx,
  xzy, yzx and zxy in an X-up file whose front is Z, the other two
  agreeing; in a Z-up file whose front is X, in an X-up one whose front
  is Y, and on keys that turn far from one frame to the next (5 frames,
  up to 230 degrees apart on an axis), every order. By layer, in every
  order but xyz, 0.40 to 1.93 apart, merged on the smaller turns too,
  but yxz in that X-up file, and kept as the first item says. On every
  file and turn measured, a root in xyz, or one whose rotation is
  inactive, agrees by layer but for the single precision of the curves
  Maya solves, as the second item says, and where it holds one level
  (the next item); and a joint root parts only where any node does (the
  two items above). The loader keeps ConvertScene's values there, within
  7.1e-15;
- Maya's FBXImport keys the curves it solves or merges at the rate its
  FBXImportResamplingRateSource option names, by default "Scene", the
  rate the scene runs at when the import starts; the import then sets
  the scene to the file's rate. It solves the rotate of a root that is
  no joint in a file whose up axis is X or Z; the rotate of a node that
  is no joint with a pre rotation it reads (rotation active, or on a
  camera or a light), which it bakes into that rotate, in a Y-up file
  too; and the rotate of a node that is no camera or light whose
  rotation is inactive, in another order than xyz, but a root of an X-
  or Z-up file, which it turns by neither order (below). It merges a
  take's layers, as it does by default, into new curves on the nodes
  they hold, a joint's rotate and translate too. The other curves keep
  the file's keys, of a single layer or of kept layers: a joint's, its
  pre rotation kept as its joint orient, a camera's or a light's with no
  pre rotation, a node's whose post rotation it keeps as its rotate
  axis, translates, pivots and offsets. From a scene at another rate,
  the keys it makes sit on the frames of that rate, not the file's, and
  on the file's frames between them Maya parts from the loader by
  amounts that hang on both rates. On the files measured, a 30 fps file
  imported from Maya's default 24 fps scene, its keys 1.25 of the file's
  frames apart: a root that is no joint of an X- or Z-up file up to 0.17
  apart as matrices, the node under it up to 0.29 off; a node with a pre
  rotation up to 6.9e-2, the locator under it up to 0.11; the nodes of a
  merged take up to 1.6e-2, the locators under them up to 3.9e-2. From a
  25 fps scene: a group root of a Z-up file 0.16 apart, the locator
  under it 0.19; the nodes of a merged take up to 6.4e-2, the locators
  under them 9.8e-2. A node it turns by neither order parts from the
  loader at another rate by its gap at the file's rate and by a share of
  its own, which grows with how far the angles Maya solves bend or jump
  between that rate's frames, not with that gap, as the `axis_system`
  paragraph below says (a smooth turn about one axis, which agrees at
  the file's rate, 1.0e-2 to 6.1e-2 apart on the files measured). With
  that option set to "File", or from a scene at the file's rate or at a
  whole multiple of it, it keys every frame of the file (on the files
  measured, from 60, 120, 150 and 240 fps scenes for a 30 fps file and
  48, 120 and 240 fps ones for a 24 fps file) and agrees on them but for
  the single precision of the curves it solves or merges, as the second
  item says (as matrices within 3.7e-7 on a node of world scale 1 on
  angles up to 250 degrees, more on larger angles, under a scale and on
  the nodes under it), and where a curve it bakes or solves moves by
  less than about 1e-2 degrees a frame of the rate it keys at, which it
  can hold level over runs of frames, a faster scene holding more: on
  the files measured, moving by 1e-4 to 6e-3 degrees a frame, up to
  3.7e-4 apart as matrices on takes of 11 and 101 frames, and from 120
  to 240 fps scenes up to 3.0e-4 more than at the file's rate on curves
  moving by about 1e-2 degrees a frame of the file and 1.1e-4 on a root
  of an X-up file; and from a 60 fps scene one curve of the 3,372 it
  made lost its key on the take's last frame, 1.3e-4 apart there. From a
  90 fps scene, though, every curve it made held keys on the take's
  first and last frames alone: on the 30 and 24 fps files measured up to
  2.0 apart as matrices on nodes of world scale 1, the nodes under them
  up to 3.6 off in translate. On a node it turns by neither order it is
  as far apart at the file's rate as the `axis_system` paragraph below
  says, and it parts where another item of this list says;
- a layer above the base turns a node that is no joint whose pre
  rotation Maya's FBXImport bakes into its rotate (the item above),
  leaving the layer's own angles as they are. An additive layer adds
  them to the baked ones by channel, the layers kept or merged, where
  the SDK adds them to the node's own under the pre rotation: that parts
  in every rotate order, but for a pre rotation about one axis alone in
  the two orders that turn about that axis last (xyz and yxz for z, yzx
  and zyx for x, xzy and zxy for y), which agree on the smaller turns
  (on the files measured within 1.2e-15 in nine such cases of ten, the
  others, on the larger turns, 9.5e-2 to 2.0 apart). By layer it parts
  in every order but xyz, merged too, but for yxz under a pre rotation
  about z alone merged on the smaller turns measured (within 1.6e-7),
  where the same node with no pre rotation agrees merged on the turns
  measured. On the files measured, pre rotations about mixed axes that
  turn 7 to 161 degrees, by channel 4.3e-3 to 1.99 apart as matrices, by
  layer merged 6.5e-3 to 2.0, more as the pre rotation and the turns
  grow; by layer, a node in xyz agrees, kept or merged, but for the
  single precision of the curve Maya bakes and where it holds that curve
  level (the item above): within 4.5e-7 on keys of up to 330 degrees and
  4.5e-6 on keys to 3500 on a node of world scale 1, a scale multiplying
  these as the second item says, and 1.2e-4 on a baked curve held level
  over three frames; a post rotation, which Maya keeps as the rotate
  axis, parts only where a node with none does. An override layer's
  angles take the place of the baked ones in every order, by layer
  merged at every weight measured or kept at 100, and by channel at 100,
  and so do a passthrough layer's at weight 100: the pre rotation is
  lost where the layer holds the rotate, Maya turning the node by the
  layer's angles alone where the SDK turns it by them under the pre
  rotation, apart by about what the pre rotation turns it. By channel
  under 100 the gap stays near that one; kept by layer under 100 Maya
  weighs the override, as the item on that says, so the gap holds both.
  On the files measured, weights 10 to 100: by layer merged, at 100 and
  by channel, 8.7e-2 to 0.12 as matrices for a pre rotation of 7
  degrees, 0.42 to 0.68 for 36 to 43, 0.75 to 0.98 for 60 to 66 and 1.35
  to 1.72 for 120 to 161, by channel under 100 within 0.21 of the gap
  the layer's angles alone give; kept by layer under 100, 0.11 to 1.99,
  and 0.11 to 1.92 for 7 degrees. Under 100 a passthrough layer's angles
  mix with the baked ones, which hold the pre rotation, so part of it
  stays, the more the lower the weight: on the smaller pre rotations and
  turns the gap is about the weight times the gap at 100 (by channel, a
  pre rotation of 7 degrees on the smaller keys, 2.1e-2 to 2.6e-2 at 25,
  4.5e-2 to 5.1e-2 at 50 and 6.9e-2 to 7.5e-2 at 75), and over both
  modes and every pre rotation measured 2.1e-2 to 2.0 at 25, 4.5e-2 to
  2.0 at 50 and 6.9e-2 to 2.0 at 75, where the same node with no pre
  rotation agrees but where another item of this list says. In every
  mode measured such a node turns about its rotate pivot, so a node
  under it, and with a pivot the node itself, is off in translate by up
  to twice its world distance from that pivot times the largest over the
  smallest singular value of the parent's world 3x3, as the
  `axis_system` paragraph below says, and a node on the pivot agrees: on
  the files measured, under even scales, up to 2.0 times that distance
  (3.72 at 1.87 and 38.9 at 19.7; with pivots 0.5 to 7.1 away, the node
  itself 4.00 off with one 2 away), under a group scaled (1, 0.5, 2) up
  to 3.48 times it, and a node on the pivot within 5.7e-14;
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
| `TransformList.load` | `(filename, mode=None, **kwargs)` — `mode` in `"gltf" / "glb" / "fbx" / "json" / "npz" / "pkl" / "pickle"`; `kwargs` go to `load_fbx` / `load_glb` by keyword |
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
write `matrix` back. Every node is written with rotation active, as
Maya's exporter writes it: FBX turns a node without it in xyz, whatever
its rotate order. A file an earlier `save_fbx` wrote holds a group or a
locator with its rotation inactive, so one in another order than xyz
now loads where the SDK puts it, in xyz, `rotate_order` 0, not where it
was saved (a zxy group 0.49 off, a joint under it 1.26): load it with
the version that wrote it, `save()` it, then load that and `save_fbx`
it with this one.

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

Every node is written with the inherit type its `segment_scale_compensate`
says, `eInheritRrs` when on and `eInheritRSrs` when off, groups and
locators too, as Maya's exporter writes a group or a locator. An earlier
`save_fbx` wrote a group or a locator `eInheritRrSs`, which scales it
along its own axes by its parent's world scale: under an unevenly scaled
parent the SDK and Maya's importer put it, and the nodes under it,
elsewhere than the rig had them (a group at (1, 1, 0) turned
(40, 10, -20) under a group turned 30 about z and scaled (1, 3, 0.5)
1.58 off in its 3x3, a locator at (0, 2, 1) under it 3.17 in position),
while `load_fbx`, reading it as `eInheritRSrs`, puts it where it was
saved: load such a file and `save_fbx` it again. That does not reach a
group or a locator that had `segment_scale_compensate` on, or the nodes
under it: the file does not hold the flag.

`load_fbx` reads a node that is no joint as `eInheritRSrs` whatever the
file stores, and `save_fbx` writes the inherit type the rig gives it, so
a load then a save of a file storing one `eInheritRrSs`, the SDK's
default, or `eInheritRrs` imports where the read has it, not where the
file does, in Maya's importer and the SDK's conversion alike: a node
turned (20, 40, 60) and scaled (1, 1.5, 0.75) under a group turned
(0, 30, 0) and scaled (1, 2, 3) is 1.11 off the file in its 3x3, and a
locator at (0.5, 1, -2) under it 2.34 or 1.75 in position. An earlier
`save_fbx` wrote it `eInheritRrSs`, which put one stored so back where
the file has it, and one stored `eInheritRSrs` off it by the same 1.11
and 2.34.

A group or a locator with `segment_scale_compensate` on, which a rig
built or edited by hand can hold and no reader sets, is written
`eInheritRrs`. In a save in cm of a rig labelled Maya's Y-up the SDK
places it and the nodes under it where the rig has them, its scale or
its parent's mirrored or not (a scale mirrors with an odd number of
negative axes), and Maya's importer does too. Three cases part, Maya's
importer in each and the SDK in the last. Their numbers are one rig's,
the largest over five turns, and grow with its scales and offsets: a
group at (2, 0.5, -0.5) scaled (1, 1.5, 0.75), turned (20, 40, 60),
(35, -10, 80), (-25, 50, 10), (60, 20, -40) and (5, -70, 30), under a
group at (0, 1, 0) turned (0, 30, 0), with a locator at (0.5, 1, -2)
under it and, for the second case, a joint at (0, 0, 1) under it over a
joint at (0, 1, 0) over a second locator at (0.25, 1, 0), both joints
compensating their scale.

- Where exactly one of its own scale and its parent's own scale mirrors,
  Maya's importer can part from the SDK and the rig on it and every node
  under it: 2.70 in the group's axes and 4.27 in the locator's position
  under a parent scaled (-1, 2, 3) or (-2, -2, -2), 8.03 and 8.38 under
  one scaled (-1, 1, 1) under a group turned (10, 0, 20) and scaled
  (1, 2, 3). The group scaled (-1, 1.5, 0.75) is 2.70 and 4.27 off under
  a parent scaled (2, 2, 2) or (1, 2, 3), and also where an earlier
  `save_fbx`, which wrote it `eInheritRrSs`, had it and the locator
  where the rig has them: 2.70 and 4.27 under a parent scaled 1, 2.51
  and 4.29 with no parent. With both mirrored Maya's importer puts it
  where the rig has it.
- A joint compensating its scale directly under it, and every node under
  that joint, Maya's importer can put elsewhere than the SDK and the
  rig, unless the group's scale is positive and its parent's scale 1, or
  1 in size with two negative axes, or it has no parent: 1.33 off in
  their axes under a parent scaled (1, 2, 3), the lower joint 1.07 in
  position and the second locator 2.18; 0.98, 0.901 and 1.88 under
  (2, 2, 2). The group scaled (-1, -1.5, 0.75), which does not mirror,
  stays where the rig has it, but its joints are 1.94, 1.67 and 3.40 off
  with no parent and 1.96, 1.80 and 3.75 under a parent scaled 1, where
  an earlier `save_fbx`, which wrote it `eInheritRrSs`, had them where
  the rig has them.
- Under a parent whose scale is near 0, a scale written as 1e-12 among
  them, or far apart on its axes, or whose world is. Where the product of
  that parent's own scale is under 2.22e-16 in size the SDK and Maya's
  importer part from the rig: 1.35 in the group's axes and 2.13 in the
  locator's position under a parent scaled (s, s, s), s 6e-06 or less
  (within 9e-16 at 6.1e-06). Above it Maya's importer alone can: in the
  group's axes 3.9e-10 under a parent scaled (1.32, 1.8, z) at z 1e-03,
  2.3e-08 at 1e-04, 0.171 at 3e-06 and 2.16 at 1e-06; in the group's axes
  and the locator's position 3.85 and 3.23 under (1e4, 1, 1e-4), 3.32 and
  7.34 under (1, 1e8, 1), 5.42 and 21.2 under a parent scaled 1 under a
  group at (0, 0.5, 0) turned (10, 0, 20) and scaled (1e6, 1, 1e-6), and
  3.96e+04 and 7.27e+04 under (1e-12, 1, 3e-4). Near 0 it can bring a
  node in not finite, where the SDK and `load_fbx` stay finite.

A rig labelled anything else, X-up, Z-up or left handed, has no such
promise: such a node, and every node under it, can import elsewhere than
the SDK and the rig. At a top node in a rotate order other than xyz that
is new: an earlier `save_fbx`, which wrote it `eInheritRrSs`, had it
where the rig has it. On the rig above with no parent, in cm, labelled
`"z up, -y front, right handed"`, that is up to 2.53 in the group's axes
and 3.44 in the locator's position over rotate orders 1 to 5.

`load_fbx` reads such a group or locator back with the flag off, as
`eInheritRSrs`, with the channels saved: at its saved position unless a
group or a locator with the flag on is above it, its axes and the nodes
under it off where its parent's own scale is not 1, on the rig above
2.11 in its axes and 2.88 in the locator's position under a parent
scaled (1, 2, 3). Under a unit root it is not the channels saved: the
read divides that root's scale out of the node, as `load_fbx`'s
docstring says, so a scale of 1 under a top node scaled 0.01 in metres
reads 100. `save()` keeps the flag.

```python
flagged = HierarchyData([
    TransformData("top", rotate=(0.0, 30.0, 0.0), scale=(1.0, 2.0, 3.0)),
    TransformData("grp", translate=(2.0, 0.5, -0.5), rotate=(20.0, 40.0, 60.0), segment_scale_compensate=True),
])
flagged["grp"].set_parent("top", world_space=False)
flagged.save_fbx(os.path.join(WORKDIR, "flagged.fbx"))

back = HierarchyData.load_fbx(os.path.join(WORKDIR, "flagged.fbx"))["grp"]
assert not back.segment_scale_compensate  # read as eInheritRSrs
assert np.allclose(back.world_matrix[3], flagged["grp"].world_matrix[3])
assert not np.allclose(back.world_matrix, flagged["grp"].world_matrix)
```

A scale under 1e-12 in size, 0 and -0 among them, is written as 1e-12,
positive, as Maya's own FBX export writes a static one and as Maya's
evaluation holds it: it loads back as 1e-12, and, in a save in cm of a
rig labelled Maya's Y-up, the nodes under it where Maya's FBXImport of
the save puts them, but for a group or a locator with
`segment_scale_compensate` on and the nodes under that. A scale of 1e-12
or more in size is written as it is.

```python
squashed = HierarchyData([TransformData("flat", scale=(0.0, -1e-13, -1e-12))])
squashed.save_fbx(os.path.join(WORKDIR, "squashed.fbx"))
back = HierarchyData.load_fbx(os.path.join(WORKDIR, "squashed.fbx"))
assert back["flat"].scale.tolist() == [1e-12, 1e-12, -1e-12]
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
The SDK's `FbxSystemUnit.ConvertScene` and Maya's unit-converting export
(`FBXExportConvertUnitString`) write a scene in another unit, s cm a
unit, as a scale of 1/s on each child of the scene root, and on each
node that does not inherit its parent's scale (a joint's segment scale
compensate), over values left in cm. A child of the scene root scaled
1/s on all three axes is read as that unit, kept or dropped: its scale
is reset to 1 and divided out of the compensating joints under it, and
the lengths under it are read as they are, so such a file loads as the
cm file it came from, rig, every frame of a clip, bind pose, meshes and
skins alike. A file only labelled in metres, its numbers in metres, is
multiplied as above. The one risk: a root authored at exactly 1/s in
such a file reads as the unit. A root the export found scaled other
than 1, which it scales by 1/s all the same, holds no unit: the nodes
at and under it load as the SDK places them, their lengths multiplied
as above, which is where the cm file has them, scaled 1/s, and the
points of a mesh there s times larger, so the mesh is right once posed.
But the export leaves the curves of a take's later layers in cm, and
there they are read as the SDK places them. Such a root's translate
keyed on one is multiplied by s with the rest: on a clip the root and
everything under it are off by s - 1 times that layer's own translate
times its weight, on each channel its curve node holds: what an
additive layer adds (Maya's metres export of a group scaled 2 keyed on
an additive animLayer: up to 2178 cm off), an override or passthrough
layer's whole value, a channel it holds unkeyed too (the same group's
translate x and z on an override animLayer: 2475 cm off on frame 5,
and 198 along y). A scale an override or passthrough layer keys there,
on the root or on a node under it that does not inherit its parent's
scale, is read as the cm value it is, s times what the rest of the read
gives it, and every node under it is off (that group's scale x on an
override animLayer: its joints 792 and 1780 cm off); an additive
layer's scale, which multiplies, loads right. And a node there that
compensates its parent's scale under a parent that is neither the root
nor compensating, as the second joint of a Maya skeleton does when its
first joint sits under such a root, is scaled 1/s once more by the
export, and every node under it with it. Read by their own values
(`pose="node"`, the default, and a clip on every frame and its rest
pose), that node loads at its cm position scaled 1/s^2, the nodes under
it off their cm positions, and a mesh on or under it, or skinned to it,
is off once posed. Under `pose="bind"` a node a bind pose lists takes
its bind matrix, which the export converts as a world matrix: it loads
where the cm file has it, scaled 1/s like the rest, and so do the nodes
under it and the meshes on them or skinned to them, posed at the bind.
A node no bind pose lists keeps what its own values give it under its
parent's bind: the node the export scales once more, and a compensating
node under it, still loads scaled 1/s^2, and the nodes under them that
no bind pose lists are off, but for those under a node one lists, which
load right, as that node does; any other loads scaled 1/s like the rest,
as the joints of a skeleton whose root joint is scaled other than 1 do,
the end joints Maya's export leaves out of the bind pose among them.
Maya's export lists every skin influence, the joints between them and
the nodes above them. Maya's importer, which scales every root and every
compensating node by s, puts all of these right.
The FBX mesh readers (`geometry.mesh.load_fbx`, `MeshData.load_fbx`,
`MeshList.load_fbx`) and `render.Object.load_fbx`, points and skin bind
matrices alike, take the same `scale_factor`, with the same default, so
a mesh under its node's world, or posed by a rig or a clip read from the
same file, sits where the file puts it.

`axis_system="maya_y_up"`, the default, loads an FBX in Maya's Y-up
axes the way `FbxAxisSystem.MayaYUp.ConvertScene` converts it: every
node's world matrix, on every frame of a clip, is the file's times the
turn from the file's axis system (its up vector, front vector and
handedness) to Maya's. Maya's FBX importer puts a file whose up axis is
X or Z, an Unreal file among them, in the same place, but for a
skeleton node at the root of the file, `eRoot` or `eEffector`, that
holds no `eLimbNode` child, which it imports as a transform, 1.41 off
ConvertScene's in its 3x3 in a Z-up file and 1.28 in an X-up one, the
nodes under it up to 2.98 and 5.0 off in translate, where the loaders
follow ConvertScene: one that holds an `eLimbNode`, as an Unreal file's
skeleton root does, imports as a joint where ConvertScene puts it, and
`save_fbx` writes every joint as an `eLimbNode`. Maya's importer also
parts from the loaders on a node below the roots that is no camera or
light whose rotation is inactive, in another order than xyz, which it
turns by neither order (below). At the file's rate, with
FBXImportResamplingRateSource "File", or from a scene whose rate is a
whole multiple of the file's but 90 fps, Maya's import and the loaders
are up to 2.0 apart in the 3x3 on a node of world scale 1: on keys
within whole turns of the angles Maya's xyz solve returns (y between -90
and 90), less the less the node turns, up to about the square of that
turn in radians on turns under 20 degrees, and on keys off that branch,
as (180, 170, 180) is of (0, 10, 0), up to about twice the turn in
radians in xzy, yzx, yxz and zxy, zyx keeping the square; and on a take
up to 3.9e-4 more where Maya holds the rotate it solves level (the list
above). Against the same node under the same parents with every scale 1,
a scale multiplies that by the node's world scale, by the largest axis
of an uneven one at most, where the scales above the node are even; by
up to the product of the largest axes of its own scale and those above
it where no parent scaled unevenly is turned or sits under a turned one;
and otherwise by up to 1.19 times that product on the files measured (a
node scaled (3, 1, 1) under a group scaled (1, 1, 3) and turned), 1.46
times it in a numpy scan of the rule Maya follows and never more than
2.13 times it. Such a node turns about its rotate pivot: a node under it
is off in translate by up to twice its world distance from that pivot,
and so is the node itself where it has one, times the largest over the
smallest singular value of the node's parent's world 3x3 (1 where the
scales above it are even, and that 3x3's longest row over its shortest
only where no turned parent sits under one scaled unevenly), and a node
on the pivot agrees. On the files measured, up to 3.0e-4 on turns of 1.1
degrees, 3.3e-2 on 11, 0.32 on 35, 1.1 on 72 and 2.0 on turns past 110,
and off the branch 3.6e-2 on turns of 1.4 degrees, 7.1e-2 on 3 and 0.35
on 10; on turns of up to 111 degrees, 5.1 on a node scaled 3, 4.9 on one
scaled (1, 2, 3) and 3.4 under one scaled 2; on turns of 11 degrees, a
node scaled (3, 1, 1) under one scaled (1, 1, 3) 8.2 times its gap at
scale 1, where the largest axis of its world 3x3 is 3.1 and the product
of the two largest axes 9; a locator 1.9 away up to 3.6 off, one 5.6
away 10.6, and one 0.1 away under a group scaled (1, 0.1, 1) up to 10
times that distance, as much with a group turned 45 degrees between
them, where that 3x3's rows give 2.8; with rotate pivots 0.5 to 7.1
away, the node itself up to 1.99 times that distance off (3.94 with a
pivot 2 away) and 3.96 times it under a group scaled (1, 0.5, 2), and a
node on the pivot within 5.9e-14. From another rate Maya keys such a
node at that rate (the list above), and on the file's frames between
those keys it parts from the loaders by the gap above and by a share of
its own, the two adding or partly cancelling: that share grows with how
far the angles Maya solves bend between that rate's frames, not with the
gap, and is largest where they jump, as they do by 180 degrees where the
node's turn passes y = 90 or -90 in xyz. On the files measured, on nodes
of world scale 1, from 24, 25 and 48 fps scenes for a 30 fps file and 30
and 25 fps ones for a 24 fps file: a smooth turn about one axis, which
agrees at the file's rate, 1.0e-2 to 6.1e-2 apart; keys that turn back
each frame, 4.1e-6 apart at the file's rate on a 0.14 degree turn and
3.1e-4 to 6.4e-4 from those scenes, 4.2e-4 and 3.2e-3 to 6.6e-3 on a 1.4
degree one, 1.1e-2 and 1.6e-2 to 3.9e-2 on a 7 degree one; keys linear
in time, at their largest within 2 percent of the gap at the file's
rate, but Maya's turn parting from its own at the file's rate by up to
1.3e-3 on turns up to 45 degrees, 4.3e-2 on turns to 111 and 1.9 on
frames where the angles it solves jump. A scale multiplies the share as
it does the gap: from a 25 fps scene for a 24 fps file, the keys whose
share is 1.85 at scale 1 have one of 7.42 on a node scaled (1, 2, 3)
under one scaled (2, 2, 2). On an X- or Z-up file Maya's importer also
parts from ConvertScene, and so from a clip, where a layer blends the
rotation of a root that is no joint, and in a file of any axes where it
keys what it solves or merges at another frame rate than the file's (the
list above). Maya's importer leaves a file whose up axis is Y, pointing
up or down, in its own axes, so a Y-up file whose front is not +Z, or a
-Y-up one, loads turned 90 or 180 degrees from where Maya shows it. An
Unreal file, Z-up, turns (x, y, z) to (x, z, -y), root travel with it.
On a root a layer above the base translates, additive, override or
passthrough, a clip still keeps the root's translate in the file's world
turned, the motion as the file authors it, where ConvertScene leaves
that layer's translate in the file's axes, and so does Maya's importer
wherever the base layer holds that translate (within 7.7e-7 of
ConvertScene; where the base layer holds no curve node for it, Maya
parts from ConvertScene as the first live layer item above says). A clip
sits 2.8 to 49 off Maya's import of such a root on the files measured, a
gap that grows with how far the layer moves it; Maya's import of a Z-up
scene it exported, a group root an additive layer translates, sits 45
off that scene turned, the clip 1.5 off it, as Maya's import of the same
scene exported Y-up does, and puts that group's 3x3, in xyz, 1.41 off
ConvertScene's and the clip's, as the list above says. A root whose
rotation is inactive still blends its rotation, blended by layer, as
ConvertScene's xyz node, as Maya's importer does a group's, a locator's
or a joint's (a zyx group root the layer also translates and turns: its
3x3 1.70 off the file's world turned and within 1.7e-7 of Maya's
FBXImport into a scene at the file's frame rate, its translate the
file's world turned, 49 off Maya's import in a Z-up file and 30 in an
X-up one). The turn goes on the roots: a joint root takes it in its
joint orient and translate and keeps its rotate curves; any other root's
rotate is solved again, through `euler_filter` on a clip. A mesh takes
the turn through its node: the mesh readers keep each mesh in its own
space and take no `axis_system`, and `render.Object.load_fbx`, a mesh
with no node above it, turns its points, normals and skin bind matrices.
A file already in Maya's Y-up, as `save_fbx` writes a default read,
loads as it is; a left-handed one raises `ValueError`, since a rotation
can't turn it into Maya's axes. `axis_system=None` keeps the file's own
axes. `load()` passes `axis_system` on to `load_fbx` as it does any
argument, so a left-handed file reads through
`HierarchyData.load(path, axis_system=None)` too. Which way the
character faces stays as the file has it.

A rig and a clip record the axis system and the unit their values are
in, `axis_system` and `unit`, saved with them. The FBX loaders record
what they produced: `"maya_y_up"` and `1.0` (cm) with their defaults; the
file's own axes in words with `axis_system=None`, `"z up, -y front, right
handed"` for an Unreal file, its up and front axes, either one negated,
and its hand; and with a number for `scale_factor` the file's unit in cm
over that number (a cm file read with `0.01` records `100.0`, metres).
Hand-built data, and a file saved before there was a record, hold
`"maya_y_up"` and `1.0`. Setting either relabels the values and moves
none of them, a clip's rest pose with the clip. A rig, clip, copy, cut
or delta built from another takes its record, and a view reads its rig's
(a view taken from `TransformList.load_fbx`'s, that load's, until a rig
owns its nodes). Values of two records have no one label, so where they
would meet `ValueError` is raised: a view over rigs that record two, a
load's view holding nodes no rig owns beside a rig's of another record,
nodes of a rig joining one that records another (an empty rig takes
theirs), the pose algebra between two, matching or pasting the values of
nodes of another (`match_translate`, `match_rotate`, `match_matrix`, and
a pose or frames pasted into `clip.frames`; values pass as they are; a
node a rig lists that another rig owns, as an item assignment lists one,
counts in that rig's record), `ClipData.from_poses` of poses of two, and
a clip's rest pose of another. `save_fbx` labels the file with the
record, the unit set explicitly, and writes the values as they are (but
for a scale under 1e-12 in size, written as 1e-12), so a save loads with
the defaults, and imports into Maya, where the file the values were read
from does, in whatever axes and unit they were read.
Where the read and Maya part, as on a Y-up file whose front is not +Z
read with the defaults, or on a group the file stores `eInheritRrSs`
under an unevenly scaled parent, the save parts too (`save_fbx`'s
docstring lists the cases). A save in another unit than cm holds its
lengths in that unit, as a file authored in it does, which Maya's
importer brings in with its top nodes scaled by the unit (100 for
metres) at the positions the values say, but for a node written
`eInheritRrs` whose parent is neither a top node nor written
`eInheritRrs`: the conversion multiplies its scale by the unit, so it
takes the unit twice, with the nodes under it, and the second of the
three cases of a group with `segment_scale_compensate` on counts the
scale the conversion gives that group's parent: a joint compensating
its scale directly under such a group directly under a top node is off
in Maya's importer though that top node's scale is 1 (`save_fbx`'s
docstring gives the numbers). A mesh holds no record: its
points are in its node's object space, the same in any axes, in the unit
its reader's `scale_factor` gives, the unit a rig read with the same
`scale_factor` records. `save_fbx` writes no geometry; a mesh's node,
read as a transform, is written as a group.

```python
rig = make_chain()
assert (rig.axis_system, rig.unit) == ("maya_y_up", 1.0)      # hand-built: Maya's Y-up, cm

rig.axis_system = "z up, -y front, right handed"  # Unreal's axes: relabels, moves nothing
rig.unit        = 100.0                           # metres
rig.save_fbx(os.path.join(WORKDIR, "zup_m.fbx"))

kept = HierarchyData.load_fbx(os.path.join(WORKDIR, "zup_m.fbx"), axis_system=None, scale_factor=1.0)
assert (kept.axis_system, kept.unit) == ("z up, -y front, right handed", 100.0)
assert np.allclose(kept.world_matrix, rig.world_matrix)        # the values as they were saved

maya = HierarchyData.load_fbx(os.path.join(WORKDIR, "zup_m.fbx"))   # the defaults
assert (maya.axis_system, maya.unit) == ("maya_y_up", 1.0)
turn = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # Z-up to Maya's Y-up
assert np.allclose(maya.world_matrix[:, 3, :3], rig.world_matrix[:, 3, :3] @ turn * 100.0)
```

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
node the read keeps, or one above a kept node, raises `ValueError`.
`pose="node"`, the default, reads the node values.

`node_types` keeps the kinds of node it names, `"joint"`, `"locator"`,
`"transform"` (a group) and `"mesh"` (read as a transform), and `None`,
the default, all four: `node_types=("joint",)` reads a skeleton without
its meshes and groups. A node of another kind (a camera, a light, an LOD
group), one with no node attribute and the scene root are always
dropped, and a kept node goes under its nearest kept ancestor. What the
dropped nodes between them hold is folded into its channels, on every
frame of a clip, so its world matrix stays the file's: a joint takes
their turn in its joint orient and their move in its translate, keeping
its rotate curves, but on a frame where an uneven scale among them
under a turn solves its rotate again; any other node has its rotate
solved again. A uniform scale among them goes on the node's scale, and
on each joint under it that compensates its parent's scale; an uneven
one under a node turned in it would need shear, which no node holds:
that node keeps the file's position, and its axis lengths where its
kept parent's world scales evenly, and the nodes under it take the
shear it drops, so they stay where the file puts them, under a scale
under 1e-12 too: a joint that compensates it in its channels, any
other node in its translate alone, keeping its own rotate and scale.
Their axes can point and stretch away from the file's, and on a frame
the node is scaled exactly 0 along an axis the shear has no inverse
and stays out of them. A scale under 1e-12 on a dropped node is read
as 1e-12, as Maya holds a scale: a joint that compensates a dropped
node scaled to 0 keeps its axes and sits on that node's position, and
a node that does not collapses onto it, as in Maya. A kept node keeps
its own scale, 0 included, and `world_matrix` divides by it as it is:
a joint compensating a kept node scaled to 0, and what sits under it,
come back with a NaN `world_matrix`, their channels finite, where Maya
keeps that joint's axes. A joint compensating a kept node, kept or
dropped, divides out that node's own scale, and a kept one keeps the
scale of a node dropped above the kept node, as Maya does: a joint
under a locator scaled 5e-13 under a dropped node scaled 2 takes that
2 in its scale. A read that drops the joints under a kept node scaled
exactly 0 collapses the nodes it keeps below them onto that node along
each axis scaled 0, where Maya, holding 1e-12, keeps them apart;
under a scale that is tiny but not 0, down to about 1e-154, they
sit where Maya puts them. Under a kept node the read scales under
1e-12, its own scale times what its split takes from the uneven
dropped nodes above it (a locator scaled 1.5e-11 under a dropped node
scaled 0.05 reads 7.5e-13), the translates that place the nodes below
it can reach the inverse of that scale, which `world_matrix`
multiplies back but Maya, holding 1e-12, does not: such a frame
written with `save_fbx` imports into Maya away from where Maya imports
the file (1.34 off with a translate of 1.6e12; 0.059 under that
locator). A node's own pivots and offsets go into its translate,
and the pre and post rotation of a node that is no joint into its
rotate, so a group whose pivot Maya centred on its children loads
where Maya shows it. A node whose rotation is inactive turns in xyz, as
the SDK turns it, its `rotate_order` 0 whatever order the file stores;
Maya's importer turns such a node by neither order, keeping the stored
order on the xyz angles of that order's turn, but on a root of an X- or
Z-up file. A light's post rotation, FBX's view axis, is
left out, as Maya places its children; a camera's turns its children by
what it adds to the view axis Maya's exporter writes, (0, -90, 0), which
Maya's importer keeps as the camera's rotate axis. Maya reads a camera's
or a light's pre rotation and rotate order, and a camera's post
rotation, whether its rotation is active or not, and so does the read,
but on a root of an X- or Z-up file with its rotation inactive, where
the axis conversion drops them and turns the root in xyz. The inherit
type is taken on a joint alone, beyond a unit root's scale:
`eInheritRrs`, which leaves the parent's own scale out, as segment scale
compensate, where the SDK and Maya put it. Every other node reads as
`eInheritRSrs`, whatever the file stores: one that is no joint stored
`eInheritRrSs`, which scales it along its own axes by its parent's world
scale, is off where that scale is uneven, one stored `eInheritRrs`
wherever its parent's own scale is other than 1, and the nodes under
them with them; a unit root's scale counts as 1 there, the read dividing
its 1/s out of every node under it stored `eInheritRrs`, joint or not. A
node turned (20, 40, 60) and scaled (1, 1.5, 0.75) under a group turned
(0, 30, 0) and scaled (1, 2, 3) is 1.11 off the SDK in the 3x3 either
way, a locator at (0.5, 1, -2) under it 2.34 and 1.75 in translate, and
keyed (20, 40, 60), (35, -10, 80), (-25, 50, 10), (60, 20, -40) and
(5, -70, 30) over 5 frames 1.63 and 2.11 on a clip, its locator 2.98 and
2.88; under one scaled (2, 2, 2) the `eInheritRrSs` node is within
8.9e-16 and the `eInheritRrs` one 0.99 off (1.35 on the clip), its
locator 2.03 (2.13). Maya's importer puts both where the SDK does,
baking the inherit type into the node's own scale and shear. A joint
stored `eInheritRrSs` reads as Maya's importer reads it, `eInheritRSrs`,
as far off the SDK. Under `pose="bind"` a dropped node a bind pose lists
is folded in at its bind, so each node kept sits where the full bind
read puts it. A file whose dropped nodes, pivots and offsets hold the
identity reads as it holds its nodes, to the bit.

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
curve goes through `euler_filter` in the node's rotate order, frame 0 as
the SDK gives it, and so does every rotation the read solves again: the
rotate of a root that is no joint, turned to Maya's axes; the rotate of
any other node that is no joint, for a node the read drops above it or
for its own pre and post rotation; a joint orient, for a node dropped
above it; and the whole rotate curve of a joint an uneven dropped scale
re-solves on some frames, as `ClipData.load_fbx` says. A file with no
animation gives a one-frame clip rather than an error.

`ClipData.load_glb`'s grid starts on the earliest key. Keys on whole
frames give exactly that many frames, both ends included — keys at frames
0 and 24 give 25 — although glTF stores key times as float32 seconds, a
hair off the frames. When the last key of a clip of two frames or more
sits a hair after the last frame, the frame is sampled on the key itself,
so a held (STEP) last key still shows, unless a held channel's own key
sits exactly on that frame, which then keeps it. A last key further
between frames, more than one float32 spacing off, adds the frame after
it. Either way such a clip's tail is inside the grid, save a held key a
hair after a key of its channel exactly on the last frame: that key shows
on no frame. Keys all within one float32 spacing of each other give one
frame, on the earliest key.

A held (STEP) key shows from the frame it was keyed for: a key time up to
one float32 spacing after a frame's time, the spacing taken at the key's
own time, counts as on that frame, and of two keys on one frame that way
the later shows. A key exactly on a frame keeps it, the last frame too,
though a second key sits a spacing after it, so a one-frame clip of two
such keys shows the first. Exactly on a frame means on the grid's time
rounded to float32, whichever side of that time float32 put the key:
float32 stores a third of the frame times at 24 fps a hair early, frame 5
as 0.2083333284 s for 0.2083333333 s, and such a key keeps its frame too.
The grid starts on the earliest key's float32 time; when that time rounded
up and a key's own time rounded down, the key can sit half a spacing or
more before the grid's frame, off its float32 value, and a second key a
spacing after it, on that value, then shows there: 14 of 1,621 generated
test files hold such a pair, the key 0.5 to 0.93 of its spacings before
the frame. The grid's first key is float32 too, so a key truly between
frames is taken in up to about two spacings after a frame: in clips up to
40 minutes in, 0.01 frame after one at 60 fps from 17 minutes in and at
any rate from 34, never 0.03 frame or more. An exporter that computes key
times in float32 arithmetic can store a key more than a spacing after its
frame, so such a held key can still show a frame late, as such a span can
still count a frame long. A key further off, such as half a frame, shows
from the next frame on. LINEAR and CUBICSPLINE channels are read at the
frames' times as they are, with no snap, their last frame on the last key
as above.

`ClipData.load_fbx` defaults to the first take that holds a key, not to
index 0, because a file can hold takes with no curves at all, or with
curve nodes and curves but not one key, such as an empty "Take 001"
ahead of the animation (cgmath 3.0.0 took the first take holding a curve
node, keyed or not). In a file with no key at all it takes the first take
that holds a curve node, with or without a curve, then the first take.
`take=` picks any take by index, an empty one included. An empty take
reads back as its own static values on every frame, the rest pose where
its curve nodes hold the node's own. A curve with no key reads at its
curve node's value, where the SDK's evaluator takes the curve's own
default, and so does Maya where the curve is in the file's active take
and the last take does not key its channel (a curve with no key in any
other take, the last included, Maya does not read): under a curve node
holding Y at 55, a Y curve with no key whose default is 99 reads 55 in
the clip, 99 in the SDK's evaluator and, there, in Maya.

Maya 2025's `FBXImport`, with its default settings, picks otherwise. It
loads its curves from the file's last take, keyed or not, so an empty
last take imports no curve. That is the last take the file's Takes
section lists, not the last take object, which `take=-1` loads; the two
are the same take wherever the file lists its takes in the order it
holds them, as the SDK and Maya write them, and only a file edited by
hand, its takes reordered or one left out of its Takes section, parts
them (of the test files with two or more takes imported in Maya 2025,
the 148 written with the SDK and the 6 exported by Maya list them in
that order, and the 23 that do not were edited by hand); in a file with
no Takes section Maya loads the last take object's curves, the take
`take=-1` loads. Its playback range is the span of the file's active
take: the take the file's `ActiveAnimStackName` names, or the first take
object, the one `take=0` loads, when that name is empty, missing or
names no take. A channel the last take does not key holds the value the
active take gives it at frame 0. So the clip's curves are Maya's where
the first take holding a key is the last take, or where no take holds
one, though its frames, and a channel no take keys, can still differ
there (Maya holds the active take's value at frame 0); `take=-1` loads
the last take's curves, over that take's own frames, with a channel it
does not key at that take's own value for it (its curve node's, the rest
pose where it holds none).

Its frames default to the take's stored span and every key of the take
together, out to whole frames: an Unreal take, whose span stops a frame
short of its last key, keeps that key, and a span wider than its keys
keeps its frames. A time within 1e-6 of a frame of a whole frame is that
frame, so a span of frames 60-121 reads 62 frames. `start_frame` and
`end_frame` override either end, and each end left out defaults on its
own: a `start_frame` passed alone no longer moves the end, which stays
the take's last frame (cgmath 2 kept the take's length instead), and one
past that frame raises `ValueError` naming it.

The clip's frames are the picked take's, where Maya's playback range is
the active take's. On 5 of 32 test files imported in Maya 2025, 3.0.1
moved the frames off Maya's playback range: cgmath 3.0.0 loaded the
empty first take, the rest pose over Maya's playback range, and 3.0.1
loads the keyed take's curves, as Maya does, over that take's frames:

| The file | Maya's playback range | The clip's frames |
|---|---|---|
| An empty first take over frames 0-48, holding a curve node and recorded as the active take, then a take keyed over frames 0-10 | 0-48 | 0-10 |
| The same file, with no active take recorded | 0-48 | 0-10 |
| The same file, its empty take over frames 0-40 | 0-40 | 0-10 |
| The same file, its empty take holding a curve with no key on it | 0-48 | 0-10 |
| Maya's own export, split into an empty take over frames 100-110 and a take keyed over frames 0-10 with its original take deleted, which records "Take 001", a name neither take has, as active | 100-110 | 0-10 |

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

An fbx or a glb goes the same way to the class's own `load_fbx` /
`load_glb` / `save_fbx`, the arguments after `mode` with it, by keyword: one
it does not take raises `TypeError` before the file is read or written. An
fbx keeps what `save_fbx` writes, one pose; a clip of more than one frame
saves as npz, json or pkl. `Data.load` refuses an fbx, which names no class.

<!-- notest -->
```python
rig.save("rig.fbx", as_ascii=True)             # rig.save_fbx("rig.fbx", as_ascii=True)
HierarchyData.load("rig.fbx", pose="bind")     # HierarchyData.load_fbx(..., pose="bind")
cgmath.hierarchy.load("rig.fbx", pose="bind")  # the same rig
ClipData.load("walk.fbx", take=0, fps=30.0)    # ClipData.load_fbx(..., take=0, fps=30.0)
```

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

A `HierarchyData` saves beside the items the record of the axes and the
unit its values are in, `axis_system` and `unit`. A `ClipData` saves its
record and its timebase, named as its properties: `start_frame`, `fps`,
`frame` (the loaded frame, which a load scrubs back to), and `blocks`, one
`(F, N, 3)` array per framed channel, each keyed by the channel's
property. A clip holding a `rest_pose` saves it as well, under
`rest_pose`, as a hierarchy's tree; one holding none saves no such entry.

```python
sorted(clip.to_dict())
# ['__class__', '__items__', 'axis_system', 'blocks', 'fps', 'frame', 'start_frame', 'unit']
clip.to_dict()["blocks"]["translate"].shape     # (4, 3, 3)
```

`copy()` is deep and owning, and its nodes keep their uuids. `==` on a
hierarchy compares its nodes by uuid, not their channels, and its
`axis_system` and `unit`. On a clip it
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
| `node.parent_node`, `TransformData(parent_node=...)` | `parent_uuid`'s name before 2.0.0 went in 3.0.0: `AttributeError`, `TypeError` | `parent_uuid`, or `set_parent` by name |
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
