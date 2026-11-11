# `cgmath` Tutorial

Ten hands-on walkthroughs that build something complete. Each one stands
on its own. For copy-paste recipes see [`CHEATSHEET.md`](CHEATSHEET.md);
for the API surface see [`REFERENCE.md`](REFERENCE.md).

**No asset ships with this package.** Every walkthrough builds its own
input — a cube, a grid, a generated curve — so you can paste the whole
file into a REPL and watch it run. Where loading from disk is the real
workflow, a short `load from file` variant sits next to the runnable one.

**Projects in this tutorial:**

1. [Hello render](#1-hello-render-90-seconds) — your first render, no rig
2. [Hero shot, frame to disk](#2-hero-shot-frame-to-disk) — a polished still
3. [Slack-ready turntable](#3-slack-ready-turntable) — orbit + encode in one call
4. [Build a joint hierarchy from scratch](#4-build-a-joint-hierarchy-from-scratch)
5. [Inspect mesh topology](#5-inspect-mesh-topology)
6. [Sculpt a shape with FFD](#6-sculpt-a-shape-with-ffd)
7. [Procedural geometry with SDFs](#7-procedural-geometry-with-sdfs)
8. [Transfer skin weights between meshes](#8-transfer-skin-weights-between-meshes)
9. [Sample a B-spline curve](#9-sample-a-b-spline-curve)
10. [Round trip an FBX file](#10-round-trip-an-fbx-file)

---

## Before you start

Two imports and a scratch directory. Everything below reuses them.

```python
import os
import tempfile

import numpy as np

WORK = tempfile.mkdtemp(prefix="cgmath_tutorial_")
print(WORK)
```

---

## 1. Hello render (90 seconds)

The point of the renderer's defaults: you should not need to rig anything
to see your geometry.

A `MeshData` is three arrays — a flat face-vertex `indices` stream, a
per-face corner `counts`, and `(V, 3)` `points`. Six quads make a cube.

```python
from cgmath.geometry import MeshData

cube = MeshData(
    name="cube",
    points=np.array([
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
        [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
    ]),
    indices=np.array([0, 1, 3, 2,  2, 3, 5, 4,  4, 5, 7, 6,
                      6, 7, 1, 0,  1, 7, 5, 3,  6, 0, 2, 4]),
    counts=np.array([4, 4, 4, 4, 4, 4]),
)
print(cube.point_count, cube.face_count, cube.edge_count)
```

Put it in a `Scene` and render.

```python
from cgmath.render import Scene

hero  = Scene("hero", base_color=(0.75, 0.55, 0.35)).add(cube)
frame = hero.render(resolution=(320, 240))
print(frame.shape, frame.dtype)          # (240, 320, 4) uint8 — always RGBA
frame.save(os.path.join(WORK, "hello.png"))
```

What happened behind the scenes:

- The cube sat at identity: a mesh on no node, in the scene's `base_color`.
- A 3-point lighting rig was sized to the mesh's AABB.
- A 3/4 elevated camera was framed with `autofit`.

To pop a window instead of writing a file, use `imshow()`. It opens an
OpenCV window and blocks until you press a key, so it is a REPL tool, not
a script tool.

<!-- notest: imshow opens a blocking OpenCV window -->
```python
hero.imshow()
```

If you already have geometry on disk, the loaders fill a `Scene`
directly. OBJ is pure python: a `v` line keeps its x y z and a `vt` line
its u v, dropping any trailing `w` or vertex colour. GLB needs
`pygltflib` and `trimesh`, FBX needs the Autodesk SDK.

<!-- notest: no .obj / .glb / .fbx asset ships with this package -->
```python
Scene.load_obj("hero.obj").imshow()
Scene.load_glb("hero.glb").imshow()      # scale_factor=100.0 — metres to cm
Scene.load_fbx("hero.fbx").imshow()      # the nodes, meshes, materials, cameras and lights the file holds
```

A mesh sits where its node puts it: `mesh.parent_uuid` names a node of
`scene.hierarchy`, and a mesh naming none sits at identity. Add the node
before the shapes that name it — a node gets its uuid when it joins a
hierarchy.

---

## 2. Hero shot, frame to disk

`Scene.add()` and `Scene.configure()` both return the scene, so they
chain. A mesh's colour is a `MaterialData` naming the mesh by uuid.

```python
from cgmath.geometry import MaterialData
from cgmath.render import Scene

scene = Scene("hero").add(cube, MaterialData(mesh_uuid=cube.uuid, color=(0.75, 0.55, 0.35), diffuse=1.0))
scene.configure(
    resolution        = (640, 480),
    samples_per_pixel = 4,        # 2x2 MSAA -> clean silhouettes + true coverage alpha
    background        = (0.05, 0.05, 0.07, 1.0),
)

frame = scene.render()
frame.save(os.path.join(WORK, "hero.png"))
print(frame.width, frame.height, frame.channels)
```

`configure()` takes the render settings — `resolution`,
`samples_per_pixel`, `background`, `autofit`, `default_light`,
`return_depth`, `angle_of_view`, `fit_padding` — the shading knobs,
`default_camera` and `aspect_ratio`. Anything else raises.
`samples_per_pixel` must be `1` or a perfect square.

```python
try:
    scene.configure(aspect=2.0)
except AttributeError as error:
    print(str(error)[:60])
```

The shading knobs are scene-wide, one value for every mesh.
`sample_method="bezier"` swaps flat quads for PN-Quad bicubic patches:
~3-4x slower, much smoother silhouettes.

```python
scene.configure(sample_method="bezier", ambient=0.6)
scene.render().save(os.path.join(WORK, "hero_bezier.png"))
```

### Want a transparent PNG cutout?

Every render is RGBA and alpha is true MSAA coverage, so edge pixels get
fractional alpha. Compose into Nuke / Premiere / DaVinci without halos.

```python
scene.background = (0, 0, 0, 0)          # alpha 0 -> miss pixels transparent
cutout           = scene.render()
print(cutout[0, 0])                      # corner pixel: fully transparent
cutout.save(os.path.join(WORK, "cutout.png"))
```

JPEG cannot hold an alpha channel, so flatten first — `frame.save()`
straight to `.jpg` raises `OSError`.

```python
cutout.image.convert("RGB").save(os.path.join(WORK, "hero.jpg"), quality=95)
```

### Want a wireframe overlay?

Two strategies, and they look different. `scene.wireframe = True` bakes
every mesh's edges into its colour map, in UV space, so they ride through
a turntable for free; it needs a UV set to bake on, and this cube has none
yet. The screen-space line is drawn onto the returned `Frame` with PIL:

```python
wired = scene.render().wireframe(scene, color=(255, 255, 255, 220), width=2)
wired.save(os.path.join(WORK, "wires.png"))
print(wired.shape)
```

`Frame.wireframe(source)` takes a `Scene` (its shown meshes, placed as it
renders them) or a single `MeshData`.

---

## 3. Slack-ready turntable

One line. The extension picks the encoder.

```python
scene.background = (0.1, 0.1, 0.1, 1.0)
out = scene.turntable(
    os.path.join(WORK, "spin.mp4"),
    n_frames   = 12,
    fps        = 12,
    resolution = (240, 240),
)
print(os.path.basename(out), os.path.exists(out))
```

Customise the orbit:

```python
scene.turntable(
    os.path.join(WORK, "half.mp4"),
    n_frames      = 8,
    fps           = 8,
    rotation_axis = "y",
    start_angle   = 0.0,
    end_angle     = 180.0,
    background    = (0.1, 0.1, 0.1),   # opaque RGB — H.264 has no alpha
    resolution    = (160, 160),
)
print(sorted(f for f in os.listdir(WORK) if f.endswith(".mp4")))
```

What is nicer than writing the loop yourself:

- **One framing pass, not `n_frames`.** The renderer finds a single camera
  distance that fits the worst-case silhouette across sampled angles, so
  the model does not breathe.
- The camera orbits the scene centroid. Nothing in the scene is mutated.
- Extension picks the encoder: `.mp4` / `.mov` / `.m4v` → H.264 via
  ffmpeg. `.avi` → alpha-preserving PNG-in-AVI. `.gif` → gifski if it is
  on PATH, else an ffmpeg two-pass palette. Anything else → a PNG image
  sequence, with `{frame:04d}` injected if you did not supply it.

A PNG sequence needs no external binary at all:

```python
seq = scene.turntable(
    os.path.join(WORK, "seq", "spin.{frame:04d}.png"),
    n_frames=4, resolution=(120, 120),
)
print(len(seq), os.path.basename(seq[0]))
```

### Requirements

`ffmpeg` must be on the *calling process's* PATH for video output. From a
DCC like Maya that usually means setting `PATH` in a startup script — see
[`render/README.md`](render/README.md).

---

## 4. Build a joint hierarchy from scratch

A `TransformData` is one transform node with Maya's channel set: name,
uuid, parent, SRT, `shear`, `rotate_order`, `joint_orient`,
`segment_scale_compensate`. A `HierarchyData` owns a set of them.

Parenting is by **uuid**: `parent_uuid` holds the parent's uuid, and a
name handed to it raises. `set_parent()` takes the name, finds the node and
keeps its uuid; `parent_name` reads the name back.

```python
from cgmath.hierarchy import HierarchyData, TransformData

rig = HierarchyData()
for name in ("root", "hip", "knee", "foot"):
    rig.append(TransformData(name, node_type="joint"))

rig["hip"].set_parent("root", world_space=False)
rig["knee"].set_parent("hip", world_space=False)
rig["foot"].set_parent("knee", world_space=False)

rig["hip"].translate  = (0.0, 1.0, 0.0)
rig["knee"].translate = (0.0, 1.0, 0.0)
rig["foot"].translate = (0.0, 1.0, 0.0)

print(rig["foot"].world_matrix[3, :3])       # [0. 3. 0.] — the chain composed
```

`world_matrix` is a property on nodes and containers alike; on a
container it is a stacked `(N, 4, 4)`.

```python
print(rig.world_matrix.shape)      # (4, 4, 4)
print(rig.world_matrix[:, 3, :3])  # every joint's world position
```

Touching any channel invalidates the cached world matrix of that node and
its whole branch.

```python
rig["hip"].rotate_z = 90.0
print(np.round(rig["knee"].world_matrix[3, :3], 6))    # [-1.  1.  0.]
```

Traversal is done from the node, not by passing a name to the container.

```python
print([n.name for n in rig["hip"].get_branch()])    # ['hip', 'knee', 'foot']
print([n.name for n in rig["hip"].get_children()])  # ['knee']
print(rig["knee"].get_parent().name)                # 'hip'
print([n.name for n in rig.get_roots()])            # ['root']
print([n.name for n in rig.match("*e*")])           # fnmatch filter
```

### Give it animation

A `ClipData` is a `HierarchyData` holding `F` poses instead of one. The
nodes are its playback head: scrub with `clip.frame`, read whole channels
through `clip.frames`.

```python
from cgmath.hierarchy import ClipData

clip                         = ClipData(rig, frames=24, start_frame=1001, fps=24.0)
clip.frames["knee"].rotate_z = np.linspace(0.0, 90.0, 24).reshape(24, 1)

print(clip.frames.world_matrix.shape)        # (24, 4, 4, 4) — every frame at once
clip.frame = 12                              # scrub — a 0-based index, not 1012
print(round(float(clip["knee"].rotate_z), 3))
print(clip.start_frame, clip.end_frame, clip.fps)
```

`clip.frame` indexes the block `0 .. frame_count - 1`. `start_frame` is
metadata that rides along for round-tripping — it does **not** shift the
index.

### Read an existing rig instead

`load()` sniffs the file header, so it picks the right reader for you.

<!-- notest: no .glb / .fbx rig ships with this package; §10 writes one first -->
```python
rig  = HierarchyData.load_glb("hero.glb")  # metres -> centimetres by default
rig  = HierarchyData.load_fbx("hero.fbx")  # in cm and Maya's Y-up, whatever the file was in
rig  = HierarchyData.load("hero.fbx")      # header sniffing
clip = ClipData.load_fbx("walk.fbx")
```

Full treatment in [`hierarchy/README.md`](hierarchy/README.md).

---

## 5. Inspect mesh topology

Counts are `point_count` / `face_count` / `edge_count` — there is no
`num_points`. Diagnostics are `get_*()` **methods**, not properties.

```python
print(f"V={cube.point_count} F={cube.face_count} E={cube.edge_count}")
print(f"tris={cube.triangles} quads={cube.quads} ngons={cube.ngons}")   # counts
print(f"closed={cube.closed} shells={cube.shell_count}")
print(f"border verts: {len(cube.get_border_vertices(flatten=True))}")
print(f"non-manifold: {cube.get_non_manifold_vertices().size}")
print(f"area: {round(cube.area, 6)}")
```

`triangles` / `quads` / `ngons` are **counts**. The index lists are
`get_triangles()` / `get_quads()` / `get_ngons()`.

An open surface is more interesting. Here is a helper that builds an
`n x n` quad grid — reuse it below.

```python
def quad_grid(n, spacing=1.0, z=0.0):
    t       = np.arange(n) * spacing
    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices += [a, a + 1, a + n + 1, a + n]
    return MeshData(
        name    = f"grid{n}",
        points  = np.array([[x, y, z] for y in t for x in t], dtype=float),
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
    )

grid = quad_grid(5)
print(grid.point_count, grid.face_count, grid.closed)
print(grid.get_border_vertices(flatten=True))        # the 16 rim vertices
print(grid.get_border_edges(flatten=True).size, grid.get_border_faces(flatten=True).size)
```

`flatten=False` — the default — returns a **list, one array per border
loop**, which is what you want when a mesh has several holes. Note that a
mesh with no border returns an empty `list` either way, so use `len()`
rather than `.size` if you have not checked `closed` first.

```python
print([loop.tolist() for loop in grid.get_border_vertices()][0][:6])
print(cube.get_border_vertices(flatten=True))        # [] — a closed cube
```

### Normals

Geometric normals are computed from `points` on demand. There is no
`get_normals()`.

```python
N = cube.get_vertex_normals()  # (V, 3), Maya's rule: Newell vector x corner angle
F = cube.get_face_normals()    # (F, 3)
print(N.shape, F.shape, np.round(F[0], 6))
```

Shading normals are the `normals` / `normal_indices` fields, which
`set_normals()` fills. A mesh need not store any: it then shades by the
derived ones, broken at the edges its `hard_edges` field names (a pair of
vertices each; None, every edge soft), and `get_face_vertex_normals()`
gives the normal of every face corner, stored or derived.

```python
C = cube.get_face_vertex_normals()  # (sum(counts), 3), a normal a face corner
print(C.shape, cube.normals)  # (24, 3) None
```

### Edges come in two flavours

`ue*` are the **raw** per-face-corner edges — an interior edge shows up
once per face. `e*` are the **deduplicated** ones, and `edge_count` counts
those. `ue` bridges them.

```python
print(cube.e2v.shape, cube.ue2v.shape)               # (12, 2) (24, 2)
print(np.array_equal(cube.e2v, cube.ue2v[cube.ue]))  # True
print(cube.e2f[0])                                   # the two faces on edge 0
```

Every adjacency matrix is dense and `-1` padded, so `row[row != -1]` is
the standard idiom.

```python
v2f = cube.v2f[0]
print(v2f, v2f[v2f != -1])               # faces touching vertex 0
```

### Keeping UVs in sync

A mesh and its UV set share face topology but not vertex ids — seams split
vertices. So the operations that must apply to both are expressed as
face-local **rules**, and you hand the identical rules object to each side.
`triangulate` takes rules, not keyword arguments.

```python
from cgmath.geometry import UVData

cube_uv = UVData(
    name="map1",
    points=np.array([
        [0.33, 0.00], [0.66, 0.00], [0.33, 0.25], [0.66, 0.25],
        [0.33, 0.50], [0.66, 0.50], [0.33, 0.75], [0.66, 0.75],
        [0.33, 1.00], [0.66, 1.00], [1.00, 0.00], [1.00, 0.25],
        [0.00, 0.00], [0.00, 0.25],
    ]),
    indices=np.array([0, 1, 3, 2,  2, 3, 5, 4,  4, 5, 7, 6,
                      6, 7, 9, 8,  1, 10, 11, 3,  12, 0, 2, 13]),
    counts=np.array([4, 4, 4, 4, 4, 4]),
)

rules = cube.get_triangulate_rules(method="fast")
mesh_t, uv_t = cube.copy(), cube_uv.copy()
mesh_t.triangulate(rules)
uv_t.triangulate(rules)
print(mesh_t.face_count, uv_t.face_count, mesh_t.triangles)
```

Verbs mutate in place and drop the topology caches. `merge()` welds
coincident vertices; `quadrangulate()` pairs triangles back into quads.

```python
mesh_t.quadrangulate(mesh_t.get_quadrangulate_rules())
print(mesh_t.face_count, mesh_t.quads)

dirty        = cube.copy()
dirty.points = np.vstack([dirty.points, dirty.points[:1]])   # a duplicate vertex
print(dirty.point_count, dirty.get_unused_points())
dirty.merge()
print(dirty.point_count)
```

More in [`geometry/README.md`](geometry/README.md).

---

## 6. Sculpt a shape with FFD

Free-Form Deformation gives you a coarse control lattice that drags a
dense mesh along with it.

The single most common mistake: `FFDData.from_mesh` takes the **lattice**,
not the mesh being deformed. Build the lattice first with
`create_lattice`, then `bind()` the target to it.

```python
from cgmath.geometry.deform import FFDData

target = cube.copy()
target.subdivide(2)                                  # something worth deforming

lattice_mesh = FFDData.create_lattice(
    (3, 3, 3),
    bbox_min = target.points.min(axis=0) - 0.05,
    bbox_max = target.points.max(axis=0) + 0.05,
)
ffd = FFDData.from_mesh(lattice_mesh, divisions=(3, 3, 3))
ffd.bind(target)
print(ffd.valid, ffd.lattice.shape)                  # True (3, 3, 3, 3)
```

`lattice` is indexed `[x, y, z, component]`. Push the top plane sideways
to shear the mesh:

```python
posed = ffd.lattice.copy()
posed[:, -1, :, 0] += 0.5                            # +X on the max-Y control plane
sheared = ffd.update(posed)                          # a new MeshData
print(type(sheared).__name__, round(float(np.abs(sheared.points - target.points).max()), 4))
```

Tune the basis and the boundary behaviour between evaluations — these are
cheap and do not force a rebind.

```python
ffd.local_influence = (3, 3, 3)  # (2,2,2)=trilinear (default), (4,4,4)=cubic
ffd.outside         = "falloff"  # "extrapolate" (default) | "freeze" | "falloff"
ffd.falloff_radius  = 2.0        # in cells
smoothed            = ffd.update(posed)
print(round(float(np.abs(smoothed.points - sheared.points).max()), 4))
```

Feed the result to the renderer to eyeball it:

```python
Scene("ffd").add(smoothed).render(resolution=(200, 160)).save(
    os.path.join(WORK, "ffd.png"))
print(os.path.exists(os.path.join(WORK, "ffd.png")))
```

FFD is one of five deformers. Delta Mush, patch relax, skinning and RBF
wrap live next to it — see [`geometry/deform/README.md`](geometry/deform/README.md).

```python
from cgmath.geometry.deform import DeltaMushData

mush    = DeltaMushData(target, smooth_iterations=10, pin_borders=True)
cleaned = mush.apply(smoothed)       # binds on the first call
print(type(cleaned).__name__, cleaned.point_count)
```

---

## 7. Procedural geometry with SDFs

An SDF is just a grid of floats: negative inside, positive outside, zero
on the surface. Cost is cubic in resolution, so start small.

### Functional

Note that primitives are always evaluated **at the origin** —
`transform_points()` moves the sample grid by the inverse transform
instead of moving the shape.

```python
from cgmath.geometry.sdf import (
    dual_marching_cubes, eval_box, eval_sphere, make_grid,
    sdf_smooth_union, sdf_union, transform_points,
)

X, Y, Z, origin, spacing = make_grid((32, 32, 32),
                                     (np.array([-2.0] * 3), np.array([2.0] * 3)))

sphere = eval_sphere(X, Y, Z, radius=0.8)
Xb, Yb, Zb = transform_points(X, Y, Z, rotation=[30, 0, 0])     # degrees
box = eval_box(Xb, Yb, Zb, half_extents=np.array([0.15, 1.2, 0.15]))

blob = dual_marching_cubes(sdf_union(sphere, box),
                           grid_origin=origin, grid_spacing=spacing)
print(blob.point_count, blob.face_count, blob.quads)   # quad-only output
blob.save_obj(os.path.join(WORK, "blob.obj"))
```

The `sdf_*` operators are plain array math, so the smooth (filleted)
variants are drop-in.

```python
filleted = dual_marching_cubes(sdf_smooth_union(sphere, box, k=0.3),
                               grid_origin=origin, grid_spacing=spacing)
print(filleted.point_count, round(filleted.area, 3), round(blob.area, 3))
```

Available: `sdf_union`, `sdf_difference`, `sdf_intersection` and their
`sdf_smooth_*` counterparts. Primitives are `eval_sphere`, `eval_box`,
`eval_cylinder` — there is no plane.

### OOP — scene graph, cached, lazy

Every primitive subclasses `TransformData`, so SRT works exactly as it
does on a rig node. `DMCField.resolution` is *subdivisions per world
unit*, not a total grid size, and the bounds are computed from the
primitives.

```python
from cgmath.geometry.sdf import DMCField, SDFBox, SDFCylinder, SDFSphere

base          = SDFSphere(radius=1.0, name="base")
fin           = SDFBox(half_extents=[0.1, 1.0, 0.1], name="fin")
fin.rotate    = [30, 0, 0]
fin.translate = [0.5, 0, 0]
hole          = SDFCylinder(radius=0.3, height=3.0, name="hole")
hole.rotate   = [0, 0, 90]

field = DMCField(resolution=12)
field.add(base)
field.add(fin, smoothing=0.1)        # smooth union
field.subtract(hole)
print(len(field.primitives), field.mesh_data.point_count)
```

`add` / `subtract` / `intersect` / `remove` all return the field, so they
chain. `mesh_data` is lazy — edit a primitive and it recomputes on the
next read.

```python
before      = field.mesh_data.point_count
base.radius = 1.3
fin.rotate  = [45, 0, 0]
print(before, field.mesh_data.point_count)
field.to_obj(os.path.join(WORK, "field.obj"))
```

A primitive's settings are saved fields like its SRT, so `copy()`,
pickling and `save()` keep them. A copy starts outside any field: editing
it leaves `field` alone. `radius` and `height` are stored as floats, and
`SDFCylinder.axis` takes 0, 1 or 2 (X, Y, Z).

```python
spare        = hole.copy()  # radius, height, axis and SRT come along
spare.radius = 0.6          # the copy is in no field
print(hole.radius, spare.radius, spare.height, spare.axis)  # 0.3 0.6 3.0 1
```

---

## 8. Transfer skin weights between meshes

`SkinData.weights` is **dense** `(V, I)`, paired with an `influences` list
of joint names. Row sums of `1.0` are what `valid` checks.

```python
from cgmath.geometry import SkinData

y    = cube.points[:, 1] + 0.5                       # 0 at the bottom, 1 at the top
skin = SkinData(weights=np.column_stack([1.0 - y, y]), influences=["root", "tip"])
print(skin.weights.shape, skin.influences, skin.valid, skin.get_max_influences())
```

`MeshDataResampler` never changes the destination topology. It answers
*"what does this source-indexed array become on the destination's
vertices?"* — and it caches the correspondence, so build one and call it
many times.

```python
from cgmath.geometry.resample import MeshDataResampler, ResampleMode

dense = cube.copy()
dense.subdivide(1)                                # same shape, new topology

resampler = MeshDataResampler(cube, dense)
dst_skin  = resampler.resample_skin_weights(skin, mode=ResampleMode.SPATIAL)
print(dst_skin.weights.shape, dst_skin.influences, dst_skin.valid)
```

The same resampler moves morph targets and painted maps through the same
correspondence.

```python
from cgmath.geometry import MorphData

bulge     = MorphData(name="bulge", offsets=np.tile([0.0, 0.2, 0.0], (8, 1)))
dst_bulge = resampler.resample_morph_target(bulge, mode=ResampleMode.SPATIAL)
print(dst_bulge.name, dst_bulge.offsets.shape, round(float(dst_bulge.max), 4))
```

### Picking a mode

| `ResampleMode` | Correspondence | Needs |
|---|---|---|
| `SPATIAL` | closest point on the source surface | nothing |
| `UV` | matching UV coordinates | `src_uv` **and** `dst_uv` |
| `ROBUST_BILINEAR` | closest surface + Laplacian inpaint, quad-native | nothing |

`SPATIAL` and `UV` interpolate a weight for *every* destination vertex,
even ones that project somewhere nonsensical. `ROBUST_BILINEAR` **rejects**
matches beyond `search_radius` (a fraction of the target bounding-box
diagonal) or `normal_threshold` degrees, then diffuses from the matched
neighbours to fill the rejects in. That rejection step is the whole
difference — reach for robust when the two meshes genuinely differ.

`ROBUST_BILINEAR` has no extra dependency:

```python
from cgmath.geometry.resample import RobustBilinearSkinTransferOptions

robust = resampler.resample_skin_weights(
    skin,
    mode=ResampleMode.ROBUST_BILINEAR,
    options=RobustBilinearSkinTransferOptions(
        max_influences=4, search_radius=0.5,
        normal_threshold=60, smooth_iterations=5, smooth_strength=0.1,
    ),
)
print(robust.weights.shape, robust.valid)
```

### Then actually deform something

Weights plus a bind rig plus a posed rig is a skinned mesh.

```python
from cgmath.geometry.deform import SkinDeformData

bind_rig = HierarchyData([
    TransformData("root"),
    TransformData("tip", translate=(0.0, 1.0, 0.0)),
])
bind_rig["tip"].set_parent("root", world_space=False)

posed_rig                  = bind_rig.copy()
posed_rig["root"].rotate_z = 30.0

deformer = SkinDeformData(mesh=cube, skin=skin, bind_rig=bind_rig)
bent     = deformer.apply(posed_rig, cube)
print(round(float(np.abs(bent.points - cube.points).max()), 4))
```

---

## 9. Sample a B-spline curve

`u` runs over knot space, `[min_param, max_param]` (`[0, spans]`, or the
knots' own range when the curve has Maya's `knots`), **not** `[0, 1]` —
unless you pass `uniform=True`, which switches to arc-length
parameterisation where equal steps in `u` are equal steps *along* the curve.

```python
from cgmath.geometry import NurbsCurveData

cv = np.array([
    [0, 0, 0], [1, 2, 0], [2, 1, 0],
    [3, 3, 0], [4, 0, 0], [5, 2, 0],
], dtype=float)

curve = NurbsCurveData(points=cv, degree=3, periodic=False, uniform=True)
print(curve.count, curve.degree, curve.max_param, round(curve.total_length, 4))
```

`compute(u)` returns a `(points, tangents)` **tuple**.

```python
u = np.linspace(0.0, 1.0, 100)
points, tangents = curve.compute(u)
print(points.shape, tangents.shape, np.round(points[0], 4), np.round(points[-1], 4))
```

`sample(query)` is the closest-point projection — the exact closest spot on
the whole curve — and it returns a `CurveSampleData` rather than a tuple.

```python
nearest = curve.sample(np.array([[2.5, 1.5, 0.0]]))
print(np.round(nearest.points, 4), np.round(nearest.params, 4),
      np.round(nearest.distances, 4))
```

The `basis` matrix a sample carries is the transfer operator: project
once, then push any per-control-point attribute stream through the same
weights with `compute(values)`.

```python
temperature = np.linspace(0.0, 100.0, len(cv)).reshape(-1, 1)
print(np.round(nearest.compute(temperature), 3))     # interpolated at the hit
```

Control points are held **by reference**, not copied. After editing the
array in place, tell the curve.

```python
cv[0] = [0.0, -1.0, 0.0]
curve.invalidate()                   # or rebuild() to do it eagerly
print(np.round(curve.compute(0.0)[0], 4))
```

### Surfaces

`NurbsSurfaceData` is the tensor-product version: every parameter, degree
and periodic flag comes in a `_u` / `_v` pair.

```python
from cgmath.geometry import NurbsSurfaceData

nu = nv = 4
gu, gv = np.meshgrid(np.linspace(0, 3, nu), np.linspace(0, 3, nv), indexing="ij")
saddle = np.stack([gu, gv, (gu - 1.5) ** 2 * 0.2 - (gv - 1.5) ** 2 * 0.2], axis=-1)

patch = NurbsSurfaceData(
    points=saddle, degree_u=3, degree_v=3,
    periodic_u=False, periodic_v=False,
)
print(patch.count_u, patch.count_v, patch.max_param_u, patch.max_param_v)
print(np.round(patch.evaluate(0.5, 0.5), 4))
```

`patch.area` calls `get_area(samples=51)`. On an open direction `get_area`
uses composite Simpson when `samples` is **odd** and degrades to a Riemann
sum when it is even, so keep the count odd.

```python
print(round(patch.area, 4), round(patch.get_area(50), 4), round(patch.get_area(201), 4))
```

---

## 10. Round trip an FBX file

This walkthrough writes its own FBX, so it runs with nothing but the
Autodesk FBX Python SDK on the path.

### Skeleton out, skeleton back

```python
skel_path = os.path.join(WORK, "rig.fbx")
rig.save_fbx(skel_path)
print(os.path.exists(skel_path))

back = HierarchyData.load_fbx(skel_path)
print(back.name)
print(np.allclose(back.world_matrix[:, 3, :3], rig.world_matrix[:, 3, :3]))

auto = HierarchyData.load(skel_path)         # header sniffing picks the reader
print(auto.name == back.name)
```

`FbxExporter.export(path, zero_root=True)` writes the root joint at
identity, so the file carries the skeleton at the origin.

```python
from cgmath.formats.fbx import FbxExporter

with FbxExporter() as exporter:
    exporter.add_skeleton(rig)
    exporter.export(os.path.join(WORK, "flat.fbx"), zero_root=True)
print(os.path.exists(os.path.join(WORK, "flat.fbx")))
```

### Author a take

A take is a `ClipData` over the rig: one row of channel values per frame.
Bend the knee from 0 to 90 degrees over 24 frames and write it out; the
clip's nodes go with it.

```python
from cgmath.hierarchy import ClipData

bend                       = ClipData(rig, frames=25, fps=24.0, take="bend")
bend.frames["knee"].rotate = np.column_stack([np.zeros(25), np.zeros(25), np.linspace(0.0, 90.0, 25)])

anim_path = os.path.join(WORK, "bend.fbx")
bend.save_fbx(anim_path)
print(os.path.basename(anim_path))
```

`ClipData.save_fbx` writes one animation stack named after the take, every
frame keyed as float32. `cgmath.formats.load_fbx` reads a whole file at
once: the takes come back as the `clip` field, a `ClipList` of one
`ClipData` per take over the nodes it animates, the file's active take
active (and `hierarchy` is `None`: the nodes are the clip's).

```python
from cgmath.formats import load_fbx

held = load_fbx(anim_path)
print([take.take for take in held.clip], held.clip.active.frame_count, held.hierarchy)   # ['bend'] 25 None
```

### Read it back as a clip

```python
from cgmath.hierarchy import ClipData

loaded = ClipData.load_fbx(anim_path)
print(loaded.frame_count, loaded.fps, loaded.name)

loaded.frame = 12                                     # 0-based index into the clip
print(round(float(loaded["knee"].rotate_z), 3))       # 45.0
```

`ClipData.save_fbx` writes a clip of two frames or more as **one take**,
every frame keyed as float32, which `ClipList.load_fbx` reads back; a clip
of one frame, a cut of one frame among them, is written as one pose.

```python
from cgmath.hierarchy import ClipList

loaded.save_fbx(os.path.join(WORK, "take.fbx"))
back = ClipList.load_fbx(os.path.join(WORK, "take.fbx"))
print(back.take, back.frame_count)                    # ['bend'] [25]

loaded.frames[12:13].save_fbx(os.path.join(WORK, "frame12.fbx"))
print(os.path.exists(os.path.join(WORK, "frame12.fbx")))
```

### Meshes, skins, cameras and lights come with the same read

`cgmath.formats.load_fbx` reads everything a file holds that cgmath has a
class for; the mesh, UV and skin classes' `load_fbx` routes are slices of
that read, and `cgmath.render.Scene.load_fbx` holds all of it, ready to
render:

<!-- notest: no mesh-bearing .fbx asset ships with this package -->
```python
from cgmath.formats import load_fbx
from cgmath.render import Scene

held = load_fbx("hero.fbx")                    # FbxContents: hierarchy, clip, meshes, skins, uvs, ...
for mesh in held.meshes:
    print(mesh.name, mesh.points.shape, mesh.parent_uuid)   # its node's uuid in held.hierarchy

hero_mesh = MeshData.load_fbx("hero.fbx", name="hero_geo")   # by the mesh's name, or its node's
hero_uv   = UVData.load_fbx("hero.fbx", name="hero_geo", channel=0)
Scene.load_fbx("hero.fbx").render().save("hero.png")
```

A mesh read from an FBX file stores no normal its points already give:
where the file's normals are the ones Maya derives, `normals` is None and
`hard_edges` holds the edges they break at; authored normals
are stored as the file holds them. `mesh.get_face_vertex_normals()` gives
the normal of every face corner either way.

---

## Where to go next

| You want to... | Read |
|---|---|
| A recipe per subpackage | [`CHEATSHEET.md`](CHEATSHEET.md) |
| The exact API surface | [`REFERENCE.md`](REFERENCE.md) |
| Transform math on raw arrays | [`transforms/README.md`](transforms/README.md) |
| Rigs, clips and retargeting | [`hierarchy/README.md`](hierarchy/README.md) |
| Meshes, curves, SDFs, transfer | [`geometry/README.md`](geometry/README.md) |
| The five deformers | [`geometry/deform/README.md`](geometry/deform/README.md) |
| Raw numba geometry kernels | [`geometry/utils/README.md`](geometry/utils/README.md) |
| Lights, cameras, encoders | [`render/README.md`](render/README.md) |
| FBX takes / GLB accessors | [`formats/README.md`](formats/README.md) |
| RBF kernels | [`rbf/README.md`](rbf/README.md) |
| Procrustes constraints | [`constraints/README.md`](constraints/README.md) |

Everything above is numpy-native and DCC-free. It runs the same in a
notebook, on a devserver, or in CI.
