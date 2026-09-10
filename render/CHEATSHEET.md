# `cgmath.render` Cheatsheet

Every public name in the module, one small runnable example each. For the
narrative version see [`README.md`](README.md); for the rest of the
library see [`../README.md`](../README.md) and
[`../CHEATSHEET.md`](../CHEATSHEET.md).

Everything exported:

```text
Scene  Object  Camera  Light  Frame  render  look_at  load_texture
```

Blocks run top to bottom as one script — later blocks reuse earlier
variables. Resolutions are deliberately tiny so the page renders fast;
see [Production settings](#production-settings) for real numbers.

---

## Contents

- [Setup](#setup) — the cube and checker every block reuses
- [One-line renders](#one-line-renders)
- [`Object`](#object) — [construction](#object-construction) · [loaders](#loading-geometry) · [texture](#texture--uv) · [shading](#shading-knobs) · [wireframe](#baked-wireframe) · [transforms](#transforms--world_points) · [topology](#topology-ops) · [preview](#preview-render--to_image--imshow) · [serialization](#serialization) · [skinning](#skinning--animation)
- [`Camera`](#camera) — [lens](#camera-construction) · [aiming](#look_at--autofit--default_view) · [orthographic](#orthographic) · [`look_at()`](#the-free-function-look_at)
- [`Light`](#light) — [point vs infinite](#point-vs-infinite) · [rig](#the-default-3-point-rig)
- [`Scene`](#scene) — [building](#building-a-scene) · [queries](#inspecting-a-scene) · [`configure`](#configure) · [background](#background--coverage-alpha) · [render](#rendering-a-scene) · [frame cache](#the-frame-cache) · [file loaders](#scene-file-loaders)
- [`Frame`](#frame) — [fields](#frame-fields) · [interop](#numpy--pil-interop) · [wireframe overlay](#screen-space-wireframe-overlay) · [encoders](#encoders)
- [Turntables](#turntables)
- [Functional `render()`](#functional-render)
- [`load_texture`](#load_texture)
- [Auto-framing and defaults](#auto-framing-and-defaults)
- [Production settings](#production-settings)
- [Gotchas](#gotchas)

---

## Setup

No mesh assets ship with this package — build geometry inline. A
triangulated unit cube plus a per-face UV island is the standard fixture.

```python
import os
import tempfile

import numpy as np
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.render import (
    Camera,
    Frame,
    Light,
    load_texture,
    look_at,
    Object,
    render,
    Scene,
)

tmp = tempfile.mkdtemp(prefix="render_cheatsheet_")


def make_cube():
    """Triangulated unit cube + one [0,1]^2 UV island per face."""
    points = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    quads = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3),
    ]
    indices, counts = [], []
    for q in quads:
        indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]])
        counts.extend([3, 3])
    counts = np.asarray(counts, dtype=np.int64)
    mesh = MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int64),
        counts  = counts,
    )
    uv_points, uv_indices = [], []
    for face in range(6):
        base = face * 4
        uv_points.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        uv_indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    uv = UVData(
        points  = np.asarray(uv_points, dtype=float),
        indices = np.asarray(uv_indices, dtype=np.int64),
        counts  = counts,
    )
    return mesh, uv


def make_checker(n=64, squares=4):
    """(n, n, 3) uint8 checkerboard."""
    tex = np.zeros((n, n, 3), dtype=np.uint8)
    s   = n // squares
    for j in range(squares):
        for i in range(squares):
            light = (i + j) % 2 == 0
            tex[j * s:(j + 1) * s, i * s:(i + 1) * s] = (
                (230, 230, 230) if light else (40, 40, 40)
            )
    return tex


mesh, uv = make_cube()
checker = make_checker()
print(mesh.points.shape, uv.points.shape, checker.shape)
```

---

## One-line renders

No Camera, no Light, no config. The renderer supplies a framed 3/4 view
and a 3-point rig.

```python
scene = Scene("hero")
scene.append(Object(name="cube", mesh=mesh, uv=uv, texture=checker))
frame = scene.render(resolution=(64, 64), samples_per_pixel=1)
print(frame)                       # Frame(shape=(64, 64, 4), dtype=uint8, cam=True)
```

An `Object` renders itself without ever building a `Scene`:

```python
cube  = Object(name="cube", mesh=mesh, uv=uv, texture=checker)
frame = cube.render(resolution=(64, 64), samples_per_pixel=1)
frame.save(os.path.join(tmp, "cube.png"))
```

`Scene.append` returns `None` — chain off the constructor, not off
`append`:

```python
scene = Scene("hero")
scene.append(cube)                 # returns None; keep the reference around
print(scene.scene_name, len(scene))
```

---

## `Object`

A renderable mesh + UV + texture + shader options, on a Maya-style
transform node.

### Object construction

Every constructor argument:

| Argument | Default | Meaning |
|---|---|---|
| `name` | `"object"` | Node name |
| `mesh` | `None` | `MeshData` to raytrace |
| `uv` | `None` | `UVData`; required whenever `texture` is set |
| `texture` | `None` | Image path or `(H, W, 3)` array |
| `skin` | `None` | `SkinDeformData` driving `mesh` (see [Skinning](#skinning--animation)) |
| `base_color` | `(0.7, 0.7, 0.7)` | Flat color used when `texture is None` |
| `sample_method` | `"bilinear"` | `"bilinear"` (flat patches) or `"bezier"` (PN-quad bicubic) |
| `wrap` | `"repeat"` | `"repeat"` or `"clamp"` for UVs outside `[0, 1]` |
| `ambient` | `0.8` | Ambient term in `[0, 1]`, added once after all lights |
| `twosided` | `True` | Shade back faces too |
| `cast_shadows` | `True` | Reserved — the renderer has no shadow pass yet |
| `resolution` | `(500, 500)` | Standalone-preview `(width, height)` |
| `samples_per_pixel` | `4` | Standalone-preview MSAA; `1` or a perfect square |
| `wireframe` | `False` | Bake mesh edges into the diffuse, in UV space |
| `wireframe_color` | `(0, 0, 0)` | RGB `0-255` for the bake |
| `wireframe_thickness` | `1` | Line width in texels for the bake |
| `background` | `None` | Standalone-preview RGB/RGBA; ignored inside a user `Scene` |
| `**transform_kwargs` | — | `translate` / `rotate` / `scale` / `visibility` / `parent_node` / ... |

```python
cube = Object(
    name="cube",
    mesh=mesh,
    uv=uv,
    texture=checker,
    base_color=(0.7, 0.7, 0.7),
    sample_method="bilinear",
    wrap="repeat",
    ambient=0.8,
    twosided=True,
    samples_per_pixel=1,
    resolution=(64, 64),
    translate=(0, 0, 0),
    rotate=(0, 25, 0),
    scale=(1, 1, 1),
)
print(cube.name, cube.ambient, cube.resolution, cube.samples_per_pixel)
```

Every one of those is also a read/write property, and assigning to it
drops the cached frame:

```python
cube.ambient           = 0.5
cube.base_color        = (0.8, 0.5, 0.2)
cube.wrap              = "clamp"
cube.twosided          = False
cube.sample_method     = "bilinear"
cube.cast_shadows      = True
cube.resolution        = (64, 64)
cube.samples_per_pixel = 1
print(cube.frame)                  # None -- invalidated by the writes above
```

### Loading geometry

`load_obj` / `load_glb` / `load_fbx` each pick the `index`-th mesh in the
file and wrap it (plus its first UV set) in an `Object`. Extra kwargs go
straight to the `Object` constructor.

Write an OBJ from the fixture so the loader has something real to eat:

```python
obj_path = os.path.join(tmp, "cube.obj")
lines    = ["g cube"]
lines += [f"v {x} {y} {z}" for x, y, z in mesh.points]
lines += [f"vt {u} {v}" for u, v in uv.points]
for i in range(0, len(mesh.indices), 3):
    vs = mesh.indices[i:i + 3] + 1
    ts = uv.indices[i:i + 3] + 1
    lines.append("f " + " ".join(f"{v}/{t}" for v, t in zip(vs, ts)))
with open(obj_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")

loaded = Object.load_obj(obj_path, index=0, base_color=(0.2, 0.6, 0.9))
print(loaded.mesh.points.shape, loaded.uv is not None, loaded.base_color)
```

GLB and FBX need assets and (for FBX) the Autodesk SDK, so only their
shapes are shown:

<!-- notest -->

```python
Object.load_glb("hero.glb", index=0, scale_factor=100.0,
                extract_texture=True, load_skin=True)
Object.load_fbx("hero.fbx", index=0, extract_texture=True, load_skin=True)
```

`load_glb` pulls the PBR base-color map into `texture` automatically
(`extract_texture=True`) and attaches skin weights when present
(`load_skin=True`). An explicit `texture=` kwarg always wins.

Pull a texture onto an existing Object after the fact. Both return
`True`/`False` rather than raising when the slot is empty:

<!-- notest -->

```python
ok = loaded.extract_texture_from_glb("hero.glb", material_index=0)
ok = loaded.extract_texture_from_fbx("hero.fbx", mesh_index=0, material_index=0)
```

### Texture & UV

`texture` accepts a path or an array; a path is loaded lazily on first
render and cached.

```python
tex_path = os.path.join(tmp, "checker.png")
Frame(array=np.dstack([checker, np.full(checker.shape[:2], 255, np.uint8)])).save(tex_path)

from_array = Object(name="a", mesh=mesh, uv=uv, texture=checker)
from_path  = Object(name="b", mesh=mesh, uv=uv, texture=tex_path)
print(from_array.get_loaded_texture().shape)   # (64, 64, 3) float32 in [0, 1]
print(from_path.get_loaded_texture().dtype)
```

`wrap` controls UVs outside `[0, 1]`; `sample_method` controls surface
interpolation. `"bezier"` fits PN-quad bicubic patches — smoother
silhouettes, several times slower.

```python
tiled   = Object(name="t", mesh=mesh, uv=uv, texture=checker, wrap="repeat")
clamped = Object(name="c", mesh=mesh, uv=uv, texture=checker, wrap="clamp")
smooth  = Object(name="s", mesh=mesh, uv=uv, sample_method="bezier")
print(tiled.wrap, clamped.wrap, smooth.sample_method)
```

### Shading knobs

Lambertian diffuse plus a flat ambient term. No specular, no PBR.

```python
flat = Object(name="flat", mesh=mesh, base_color=(0.9, 0.3, 0.2), ambient=1.0)
lit  = Object(name="lit", mesh=mesh, base_color=(0.9, 0.3, 0.2), ambient=0.1)
print(flat.render(resolution=(48, 48), samples_per_pixel=1).array.mean())
print(lit.render(resolution=(48, 48), samples_per_pixel=1).array.mean())
```

### Baked wireframe

`Object.wireframe = True` rasterizes the mesh edges into a copy of the
diffuse **in UV space**. Cheap and cached, and it rides along through
turntables for free. Needs a `uv`.

```python
wired = Object(
    name="wired",
    mesh=mesh.copy(),
    uv=uv.copy(),
    texture=checker,
    wireframe=True,
    wireframe_color=(255, 80, 0),
    wireframe_thickness=3,
)
print(wired.get_loaded_texture().shape)      # the baked diffuse
wired.render(resolution=(64, 64), samples_per_pixel=1)
```

With no `texture`, the bake fills with `base_color` first, so you get
edges over flat shading. For a sharper screen-space line instead, see
[the Frame overlay](#screen-space-wireframe-overlay).

### Transforms & `world_points`

`Object` is a `TransformData`, so it has the whole Maya SRT surface.

```python
moved = Object(name="moved", mesh=mesh, translate=(2, 0, 0), rotate=(0, 45, 0))
print(moved.world_matrix[3, :3])  # row-major: translation at [3, :3]
print(moved.world_points.shape)   # mesh points in world space
print(moved.world_points[:, 0].min() > 1.0)
```

Parenting composes as expected:

```python
group       = Object(name="group", translate=(0, 5, 0))
child       = Object(name="child", mesh=mesh, parent_node="group")
group_scene = Scene("parented")
group_scene.append(group)
group_scene.append(child)
print(child.world_points[:, 1].min())        # lifted by the parent
```

### Topology ops

All three mutate in place, keep `uv` in sync, and clear the caches.

```python
topo = Object(name="topo", mesh=mesh.copy(), uv=uv.copy())
topo.quadrangulate(source="uv")     # or source="mesh"
print(np.unique(topo.mesh.counts))  # [4]
topo.triangulate(ngons_only=False)
print(np.unique(topo.mesh.counts))  # [3]
topo.merge()                        # de-duplicate coincident points
print(topo.mesh.points.shape)
```

### Preview: `render` / `to_image` / `imshow`

`Object.render` wraps the Object in a one-Object Scene, lights it with
the default 3-point rig, and caches the result on `.frame`.

```python
preview = Object(name="preview", mesh=mesh, uv=uv, texture=checker,
                 resolution=(64, 64), samples_per_pixel=1)
f = preview.render()                                         # uses .resolution / .samples_per_pixel
f = preview.render(resolution=(48, 48), samples_per_pixel=1)
f = preview.render(output=os.path.join(tmp, "preview.png"))  # render + save
print(preview.frame is f, preview.buffer.shape)          # cached pixel buffer
```

`to_image()` returns a PIL image, lazy-rendering with the Object's own
defaults when nothing is cached:

```python
image = preview.to_image()
print(image.size, image.mode)
```

`imshow()` opens an OpenCV window and blocks on a keypress:

<!-- notest -->

```python
preview.imshow()
```

### Serialization

`Object` round-trips through the inherited `Data` API.

```python
payload = cube.to_dict()
clone   = Object.from_dict(payload)
print(sorted(payload)[:4], clone.name, clone.mesh.points.shape)
```

### Skinning & animation

`skin` / `pose` / `restore_bind_pose` / `animate` drive a
`SkinDeformData` (from `cgmath.geometry.deform`) and an animation clip.
Building a rig is the deformer package's job, so only the shapes are
shown here:

<!-- notest -->

```python
hero = Object.load_glb("hero.glb", load_skin=True)   # attaches .skin
hero.pose(posed_rig)      # HierarchyData or (J, 4, 4)
hero.render("posed.png")
hero.restore_bind_pose()  # back to the bind mesh
hero.animate(clip, output="anim.{frame:04d}.png", fps=24)
```

`pose()` always deforms the *bind* mesh, never the current one, so a
sequence never compounds.

---

## `Camera`

OpenGL/Maya convention: the camera looks down its own `-Z`, `+Y` is up,
`+X` is right.

### Camera construction

| Argument | Default | Meaning |
|---|---|---|
| `name` | `"camera"` | Node name |
| `angle_of_view` | `35.0` | Vertical full FOV, degrees (perspective) |
| `is_orthographic` | `False` | Emit parallel rays instead |
| `ortho_height` | `None` | World-space frustum height; derived from the AABB when `None` |
| `near_plane` | `0.0` | Reserved — no clipping pass yet |
| `far_plane` | `inf` | Reserved — no clipping pass yet |
| `aspect_ratio` | `None` | Reserved — the renderer uses `resolution` |
| `**transform_kwargs` | — | `translate` / `rotate` / `parent_node` / ... |

```python
cam = Camera(name="cam", angle_of_view=28.0, translate=(0, 1, 5))
print(cam.angle_of_view, cam.is_orthographic, cam.near_plane, cam.far_plane)

cam_scene = Scene("with_camera")
cam_scene.append(Object(name="cube", mesh=mesh, uv=uv, texture=checker))
cam_scene.append(cam)
cam_scene.render(resolution=(64, 64), samples_per_pixel=1)
```

### `look_at` / `autofit` / `default_view`

```python
cam = Camera(name="cam", translate=(3, 3, 3))
cam.look_at((0, 0, 0))                       # keep position, aim at a world point
forward = -cam.world_matrix[2, :3]           # row-major: forward is -row 2
print(np.round(forward, 3))

cam.autofit(cube.world_points, 64, 64)       # move in/out so the points fill the frustum
print(np.round(cam.world_matrix[3, :3], 3))

auto = Camera.default_view(cube.world_points, name="cam", angle_of_view=35.0)
print(np.round(auto.world_matrix[3, :3], 3))  # 3/4 elevated front-right
```

`autofit` only works on a camera at the world root — a parented camera
raises `ValueError`:

```python
child_cam = Camera(name="child", parent_node="group")
try:
    child_cam.autofit(cube.world_points, 64, 64)
except ValueError as exc:
    print("expected:", exc)
```

### Orthographic

```python
ortho = Camera(name="ortho", is_orthographic=True, ortho_height=2.5,
               translate=(0, 0, 4))
ortho_scene = Scene("ortho")
ortho_scene.append(Object(name="cube", mesh=mesh))
ortho_scene.append(ortho)
print(ortho_scene.render(resolution=(48, 48), samples_per_pixel=1))
```

Leave `ortho_height=None` and the renderer derives it from the scene's
vertical extent in eye space (plus 5%).

### The free function `look_at`

Same math, no node — returns a **column-major** camera-to-world matrix
(eye in the last *column*), which is what `render(camera_matrix=...)`
wants.

```python
m = look_at(eye=(4, 3, 4), target=(0, 0, 0), up=(0, 1, 0))
print(m.shape, np.round(m[:3, 3], 3))        # column 3 = eye
frame = render(mesh=mesh, camera_matrix=m, resolution=(48, 48))
```

| Matrix | Layout | Eye at |
|---|---|---|
| `look_at()`, `render(camera_matrix=)`, `Frame.camera_matrix` | column-major (OpenGL) | `m[:3, 3]` |
| `Camera.world_matrix`, `Object.world_matrix` | row-major (Maya) | `m[3, :3]` |

`Camera.look_at` / `autofit` / `default_view` transpose for you.

---

## `Light`

### Point vs infinite

| `kind` | Comes from | Falloff |
|---|---|---|
| `"point"` | The light's world position | `1/r²` by default; `falloff=False` for constant |
| `"infinite"` | The light's `-Z` axis (a sun) | None — parallel rays |

```python
key = Light(name="key", kind="point", intensity=1e2, color=(1.0, 0.95, 0.9),
            falloff=True, translate=(2, 3, 2))
fill = Light(name="fill", kind="point", intensity=40.0, falloff=False,
             translate=(-2, 1, 2))
sun = Light(name="sun", kind="infinite", intensity=2.0, rotate=(-45, 0, 0))

print(key.position)                # world-space position (read-only)
print(np.round(sun.direction, 3))  # world-space -Z axis, normalised (read-only)
print(key.as_dict())               # the dict shape render(point_light=...) takes
```

Anything but `"point"` / `"infinite"` is rejected at construction:

```python
try:
    Light(name="bad", kind="spot")
except ValueError as exc:
    print("expected:", exc)
```

Point-light intensity is physical: with `1/r²` falloff you want roughly
`distance²`. Drop them in a scene:

```python
lit_scene = Scene("lit")
lit_scene.append(Object(name="cube", mesh=mesh, uv=uv, texture=checker))
lit_scene.append(key)
lit_scene.append(fill)
print(lit_scene.render(resolution=(64, 64), samples_per_pixel=1))
```

Parent a light to a camera for a headlight that tracks the view:

```python
head_cam = Camera(name="head_cam", translate=(0, 1, 4))
headlight = Light(name="headlight", kind="point", intensity=30.0,
                  parent_node="head_cam")
head_scene = Scene("headlight")
head_scene.append(Object(name="cube", mesh=mesh))
head_scene.append(head_cam)
head_scene.append(headlight)
print(headlight.position)          # inherits the camera's position
head_scene.render(resolution=(48, 48), samples_per_pixel=1)
```

### The default 3-point rig

With **no** `Light` in the scene and `default_light` not disabled, a
key/fill/rim rig sized to the scene AABB is injected for the duration of
the call and popped off afterwards — your scene is never mutated.

| Light | Direction from centroid | Intensity |
|---|---|---|
| key | `(+1, +1, +1) · d` | `d²` |
| fill | `(-1, +0.5, +1) · d` | `0.4 · d²` |
| rim | `(0, +0.5, -1) · d` | `0.3 · d²` |

where `d = 1.5 · ‖aabb_max − aabb_min‖`. Because it scales with the
bounding box, a 1-unit cube and a 100-unit ship both light correctly.

```python
auto_lit = Scene("auto")
auto_lit.append(Object(name="cube", mesh=mesh, uv=uv, texture=checker))
print(len(auto_lit.lights))  # 0 -- rig is transient
auto_lit.render(resolution=(48, 48), samples_per_pixel=1)
print(len(auto_lit.lights))  # still 0

auto_lit.render(resolution=(48, 48), samples_per_pixel=1, default_light=False)
```

With `default_light=False` and no lights, only the `ambient` term shades
the surface.

---

## `Scene`

### Building a scene

Constructor arguments are the persistent render config (all optional):

```python
scene = Scene(
    "hero",
    resolution          = (64, 64),
    samples_per_pixel   = 1,
    background          = (0.05, 0.05, 0.05, 1.0),
    angle_of_view       = 35.0,
    autofit             = None,
    default_light       = None,
    return_depth        = None,
    fit_padding         = None,
    default_camera_name = None,
    aspect_ratio        = None,
)
scene.append(Object(name="cube", mesh=mesh, uv=uv, texture=checker))
scene.append(Camera(name="cam", angle_of_view=35.0))
scene.append(Light(name="key", kind="point", intensity=1e2, translate=(2, 3, 2)))
print(scene.scene_name, len(scene))
```

`Scene` is a `HierarchyData`, so it is an ordinary mutable sequence:

```python
scene.insert(0, Object(name="second", mesh=mesh, translate=(1.5, 0, 0)))
print([node.name for node in scene])
popped = scene.pop()
print(popped.name, len(scene))
```

Only `TransformData` subclasses may be appended:

```python
try:
    scene.append({"not": "a node"})
except TypeError as exc:
    print("expected:", exc)
```

### Inspecting a scene

```python
print([o.name for o in scene.objects])
print([c.name for c in scene.cameras])
print([lt.name for lt in scene.lights])

print(scene.get_camera())            # default, then first
print(scene.get_camera("cam").name)  # by name
scene.default_camera_name = "cam"
print(scene.get_camera().name)

lo, hi = scene.union_aabb()                  # world AABB of visible Objects
print(np.round(lo, 2), np.round(hi, 2))
print(scene.union_points().shape)            # concatenated world points
```

Both accept an explicit subset:

```python
print(scene.union_aabb(objects=scene.objects[:1]))
```

### `configure`

Chainable bulk setter for the persistent render config. `None` values
are skipped. The allowed keys are exactly:

```text
resolution  samples_per_pixel  background  autofit
default_light  return_depth  angle_of_view  fit_padding
```

```python
scene.configure(
    resolution        = (64, 64),
    samples_per_pixel = 1,
    background        = (0.05, 0.05, 0.05, 1.0),
    angle_of_view     = 32.0,
    fit_padding       = 1.10,
    return_depth      = True,
).render()
print(scene.frame.has_depth)
```

Anything else raises — `aspect_ratio` and `default_camera_name` are
plain attributes, not `configure()` keys:

```python
try:
    scene.configure(aspect_ratio=1.0)
except AttributeError as exc:
    print("expected:", exc)
scene.aspect_ratio = 1.0                     # set it directly instead
scene.configure(return_depth=False)
```

Per-call kwargs always beat the persistent config:

```python
print(scene.render(resolution=(32, 32)).shape)   # (32, 32, 4), not (64, 64, 4)
```

### Background & coverage alpha

Every render is RGBA and the alpha channel is real MSAA coverage
(`hits / samples_per_pixel`). RGB triples are promoted to opaque.

```python
scene.background = (0.1, 0.2, 0.3)
print(scene.background)                      # (0.1, 0.2, 0.3, 1.0)

scene.background = (0.0, 0.0, 0.0, 0.0)      # transparent miss pixels
cutout           = scene.render(resolution=(64, 64), samples_per_pixel=4)
print(cutout.array[..., 3].min(), cutout.array[..., 3].max())   # 0 .. 255
cutout.save(os.path.join(tmp, "cutout.png"))
```

### Rendering a scene

```python
frame = scene.render()                                   # honours configure()
frame = scene.render(resolution=(48, 48), samples_per_pixel=1)
frame = scene.render(resolution=(48, 48), samples_per_pixel=1, return_depth=True)
print(frame.has_depth, np.nanmin(frame.depth))           # NaN at miss pixels

image = scene.to_image(resolution=(48, 48))              # PIL; lazy-renders
print(image.size, image.mode)
```

`imshow()` opens an OpenCV window:

<!-- notest -->

```python
scene.imshow(resolution=(500, 500))
```

### The frame cache

`scene.frame` / `scene.buffer` hold the last render, so repeated
`to_image()` / `imshow()` calls do not re-render. The cache
auto-invalidates when any render input changes — transforms anywhere in
the graph, camera lens settings, light color/intensity, object visuals,
scene topology, or anything from `configure()`.

```python
scene.render(resolution=(48, 48), samples_per_pixel=1)
print(scene.frame is not None, scene.buffer.shape)

scene.objects[0].translate = (0.5, 0, 0)     # transform change
print(scene.frame)                           # None -- invalidated

scene.render(resolution=(48, 48), samples_per_pixel=1)
scene.cameras[0].angle_of_view = 50.0        # lens change
print(scene.frame)                           # None
```

The one case it cannot see is **in-place** mutation of array contents:

```python
scene.render(resolution=(48, 48), samples_per_pixel=1)
scene.objects[0].mesh.points[0] += 0.25      # in-place: cache does NOT notice
print(scene.frame is not None)                          # True -- stale
scene.render(resolution=(48, 48), samples_per_pixel=1)  # refresh explicitly
```

### Scene file loaders

One `Object` per mesh in the file; no cameras or lights are added.

```python
from_obj = Scene.load_obj(obj_path, name="from_obj")
print(from_obj.scene_name, len(from_obj.objects))
from_obj.render(resolution=(48, 48), samples_per_pixel=1)
```

<!-- notest -->

```python
Scene.load_glb("hero.glb", name="hero", scale_factor=100.0,
               load_skin=True, extract_texture=True)
```

---

## `Frame`

What every render returns: pixels, optional depth, and the camera it was
shot with.

### Frame fields

```python
frame = scene.render(resolution=(48, 48), samples_per_pixel=1, return_depth=True)

print(frame.array.shape, frame.array.dtype)  # (H, W, 4) uint8 -- always RGBA
print(frame.depth.shape, frame.depth.dtype)  # (H, W) float64, NaN at miss
print(frame.camera_matrix.shape)             # (4, 4) column-major, as rendered
print(frame.angle_of_view)                   # vertical FOV used

print(frame.shape, frame.dtype)
print(frame.height, frame.width, frame.channels, frame.has_depth)
```

Construct one by hand when you need to (depth / camera default to
`None`):

```python
manual = Frame(array=np.zeros((8, 8, 4), dtype=np.uint8))
print(manual, manual.has_depth, manual.camera_matrix)
```

### numpy & PIL interop

```python
print(np.asarray(frame).shape)    # __array__ protocol
print(frame[0, 0])                # index straight through to .array
print(frame[10:20, 10:20].shape)  # slices too
print(len(frame))                 # number of rows

arr, depth = frame                           # legacy (image, depth) unpack
print(arr.shape, depth.shape)
(arr_only,) = manual                         # single-element when depth is None
print(arr_only.shape)

print(frame.image.size, frame.image.mode)                # lazy PIL view
frame.save(os.path.join(tmp, "frame.png"))
frame.save(os.path.join(tmp, "frame.webp"), quality=95)  # kwargs go to PIL
```

Frames are always RGBA, and JPEG carries no alpha — flatten before
saving one:

```python
frame.image.convert("RGB").save(os.path.join(tmp, "frame.jpg"), quality=95)
```

For arithmetic go through `.array` — `Frame` deliberately does not
pretend to be an ndarray.

```python
brighter = np.clip(frame.array.astype(np.int16) + 20, 0, 255).astype(np.uint8)
print(brighter.shape)
```

### Screen-space wireframe overlay

Projects mesh edges through the frame's own `camera_matrix` and draws
them with PIL. Returns a **new** Frame — nothing mutates in place.

```python
wire_frame = frame.wireframe(
    scene,                              # a Scene, an Object, or a MeshData
    color          = (255, 255, 255, 220),  # RGBA 0-255; default (0, 0, 0, 200)
    width          = 2,                     # default 1
    unique_edges   = True,                  # draw each shared edge once
    cull_backfaces = True,                  # False = X-ray, every edge drawn
)
wire_frame.save(os.path.join(tmp, "wires.png"))

xray        = frame.wireframe(scene, cull_backfaces=False)
from_object = frame.wireframe(scene.objects[0])
from_mesh   = frame.wireframe(mesh)       # points treated as world-space
print(wire_frame, xray, from_object, from_mesh)
```

It needs a camera, so a hand-built Frame is rejected:

```python
try:
    manual.wireframe(scene)
except ValueError as exc:
    print("expected:", str(exc)[:60])
```

`Frame.wireframe()` (sharp screen-space line, per-frame cost) is a
different feature from `Object.wireframe = True` (baked into the
diffuse, cached, survives turntables).

### Encoders

All four are classmethods taking any iterable of `Frame`s or
`(H, W, 3|4) uint8` arrays, and all return the expanded output path.
They shell out to `ffmpeg`.

```python
frames = [
    scene.render(resolution=(48, 48), samples_per_pixel=1,
                 camera_matrix=look_at((4 + i * 0.1, 3, 4), (0, 0, 0)))
    for i in range(4)
]

Frame.encode_mp4(frames, os.path.join(tmp, "spin.mp4"), fps=12,
                 background=(0.0, 0.0, 0.0), crf=18, preset="slow",
                 overwrite=True)
Frame.encode_avi(frames, os.path.join(tmp, "spin.avi"), fps=12, overwrite=True)
Frame.encode_gif(frames, os.path.join(tmp, "spin.gif"), fps=12,
                 use_gifski=None, overwrite=True)
print(sorted(os.listdir(tmp))[:3])
```

`Frame.encode` dispatches on the extension:

```python
Frame.encode(frames, os.path.join(tmp, "auto.mp4"), fps=12)
Frame.encode(frames, os.path.join(tmp, "auto.avi"), fps=12)
Frame.encode(frames, os.path.join(tmp, "auto.gif"), fps=12)
try:
    Frame.encode(frames, os.path.join(tmp, "auto.webm"))
except ValueError as exc:
    print("expected:", str(exc)[:48])
```

| Format | Alpha | Notes |
|---|---|---|
| `.mp4` / `.mov` / `.m4v` | flattened over `background` | H.264, `crf` 18 = visually lossless, `preset` `"slow"` |
| `.avi` | preserved per-pixel | PNG-in-AVI; bigger files, clean NLE compositing |
| `.gif` | 1-bit, thresholded | [gifski](https://gif.ski/) when installed, else ffmpeg two-pass palette |

`use_gifski=None` auto-detects, `True` forces it (and raises when
absent), `False` forces the ffmpeg path. Both binaries are found through the
calling process's `PATH` only — there is no environment-variable override;
a missing binary raises `RuntimeError` with an install hint.

---

## Turntables

One framing pass for the whole orbit, so nothing wobbles or clips. The
camera orbits the scene centroid; the scene itself is never mutated.

```python
paths = scene.turntable(
    os.path.join(tmp, "tt", "spin.{frame:04d}.png"),
    n_frames          = 4,
    resolution        = (48, 48),
    samples_per_pixel = 1,
)
print(len(paths), os.path.basename(paths[0]))
```

Every argument:

| Argument | Default | Meaning |
|---|---|---|
| `output_pattern` | `"turntable.{frame:04d}.png"` | Extension picks the encoder; `{frame}` auto-injected for sequences |
| `n_frames` | `120` | Frames in the loop |
| `rotation_axis` | `"y"` | `"x"`, `"y"` or `"z"` |
| `start_angle` | `0.0` | Degrees |
| `end_angle` | `360.0` | Last frame stops one step short, so it loops seamlessly |
| `fit` | `"auto"` | `"auto"` = AABB framing pass; `"static"` = use the scene camera as-is |
| `camera_name` | `None` | Which `Camera` supplies the base orientation |
| `fps` | `30` | Video / gif only |
| `background` | `None` | RGB to flatten alpha against for mp4; defaults to `scene.background[:3]` |
| `n_fit_samples` | `None` | Angles sampled for the framing pass; `None` = `n_frames` |
| `verbose` | `False` | Print a line per frame |
| `**render_kwargs` | — | Forwarded to `render()` |

```python
paths = scene.turntable(
    os.path.join(tmp, "tt2", "spin.{frame:04d}.png"),
    n_frames=3,
    rotation_axis="x",
    start_angle=0.0,
    end_angle=180.0,
    fit="auto",
    camera_name="cam",
    fps=12,
    background=(0.1, 0.1, 0.1),
    n_fit_samples=6,
    verbose=False,
    resolution=(32, 32),
    samples_per_pixel=1,
)
print(len(paths))
```

Video and gif return a single path instead of a list:

```python
video = scene.turntable(os.path.join(tmp, "spin_tt.mp4"), n_frames=4,
                        resolution=(48, 48), samples_per_pixel=1, fps=12)
gif = scene.turntable(os.path.join(tmp, "spin_tt.gif"), n_frames=4,
                      resolution=(48, 48), samples_per_pixel=1, fps=12)
print(os.path.basename(video), os.path.basename(gif))
```

`fit="static"` keeps your camera exactly where you put it (it still
orbits the centroid):

```python
static = scene.turntable(os.path.join(tmp, "tt3", "s.{frame:04d}.png"),
                         n_frames=2, fit="static", camera_name="cam",
                         resolution=(32, 32), samples_per_pixel=1)
print(len(static))
```

`Object.turntable` is the same thing on a one-Object preview scene, with
a flatter signature (`output`, `n_frames`, `resolution`,
`samples_per_pixel`, `fps`, `**render_kwargs`):

```python
solo = Object(name="solo", mesh=mesh, uv=uv, texture=checker,
              resolution=(32, 32), samples_per_pixel=1)
print(len(solo.turntable(os.path.join(tmp, "tt4", "o.{frame:04d}.png"), n_frames=2)))
```

Give a path with no `{frame}` placeholder and one is injected as
`<stem>.{frame:04d}<ext>`.

---

## Functional `render()`

The low-level entry point. Three call styles, all returning a `Frame`.

```python
# 1. single mesh
frame = render(mesh, uv, checker, resolution=(48, 48))

# 2. list of dicts (legacy multi-mesh)
frame = render(
    scene=[
        {"mesh": mesh, "uv": uv, "texture": checker},
        {"mesh": mesh, "base_color": (0.2, 0.6, 0.9)},
    ],
    resolution=(48, 48),
)

# 3. a Scene (preferred)
frame = render(scene=scene, resolution=(48, 48), samples_per_pixel=1)
print(frame)
```

The full signature:

| Argument | Default | Meaning |
|---|---|---|
| `mesh` / `uv` / `texture` | `None` | Single-mesh shorthand |
| `camera_matrix` | `None` | `(4, 4)` **column-major** camera-to-world; `None` builds a 3/4 view and forces `autofit=True` |
| `angle_of_view` | `35.0` | Vertical full FOV, degrees |
| `resolution` | `(512, 384)` | `(width, height)` |
| `point_light` | `None` | One dict or a list of dicts (`Light.as_dict()` shape) |
| `background` | `(0, 0, 0, 0)` | **RGBA** required here — no RGB promotion |
| `sample_method` | `"bilinear"` | Or `"bezier"`, or a `SampleMethod` enum |
| `ambient` | `0.1` | Ambient term in `[0, 1]` |
| `wrap` | `"repeat"` | Or `"clamp"` |
| `twosided` | `True` | Shade back faces |
| `autofit` | `False` | Move the camera so the scene fills the frustum |
| `fit_padding` | `None` | Autofit margin `>= 1.0`; `None` = `1.15` for bezier, else `1.0` |
| `return_depth` | `False` | Populate `Frame.depth` |
| `samples_per_pixel` | `1` | MSAA; `1` or a perfect square |
| `base_color` | `(0.7, 0.7, 0.7)` | Used when there is no texture |
| `scene` | `None` | `list[dict]` or a `Scene` |
| `default_light` | `True` | Add a camera-parented headlight when the scene has no light |

Lights as plain dicts:

```python
frame = render(
    mesh       = mesh,
    uv         = uv,
    texture    = checker,
    resolution = (48, 48),
    point_light=[
        {"kind": "point", "position": (2, 2, 2), "intensity": 20.0,
         "color": (1.0, 1.0, 1.0), "falloff": True},
        {"kind": "infinite", "direction": (0, -1, -1), "intensity": 0.5},
    ],
    ambient=0.2,
)
print(frame)
```

A `Light` node hands you that dict directly:

```python
frame = render(mesh=mesh, resolution=(32, 32), point_light=key.as_dict())
print(frame)
```

Explicit camera, autofit and depth:

```python
frame = render(
    mesh=mesh,
    camera_matrix=look_at((4, 3, 4), (0, 0, 0)),
    angle_of_view=35.0,
    resolution=(48, 48),
    autofit=True,
    fit_padding=1.15,
    return_depth=True,
    samples_per_pixel=4,
    background=(0.0, 0.0, 0.0, 1.0),
    base_color=(0.9, 0.6, 0.2),
    ambient=0.15,
)
print(frame.has_depth, frame.array[..., 3].min())    # opaque background
```

Errors are eager and specific:

```python
for bad in (lambda: render(resolution=(48, 48)),
            lambda: render(mesh=mesh, samples_per_pixel=3),
            lambda: render(mesh=mesh, wrap="mirror"),
            lambda: render(mesh=mesh, resolution=(0, 48))):
    try:
        bad()
    except ValueError as exc:
        print("expected:", str(exc)[:52])
```

---

## `load_texture`

Normalises anything image-like to a contiguous `(H, W, 3)` `float32`
array in `[0, 1]`. Greyscale is broadcast to 3 channels, RGBA is
truncated to RGB, `uint8` divides by 255 and `uint16` by 65535.

```python
print(load_texture(checker).shape, load_texture(checker).dtype)
print(load_texture(tex_path).max() <= 1.0)                # from a path, via PIL
print(load_texture(np.zeros((4, 4), np.uint8)).shape)     # grey -> (4, 4, 3)
print(load_texture(np.zeros((4, 4, 4), np.uint8)).shape)  # RGBA -> (4, 4, 3)

try:
    load_texture(np.zeros((4, 4, 2), np.uint8))
except ValueError as exc:
    print("expected:", exc)
```

Paths expand `~` and environment variables.

---

## Auto-framing and defaults

What you get when you supply nothing:

| Thing | Default |
|---|---|
| Camera | 3/4 elevated front-right (`+x, +0.7y, +z` from the AABB centre, at `1.5 ×` the diagonal), autofit on |
| Lighting | 3-point key/fill/rim rig sized to the AABB |
| Resolution | `(500, 500)` from `Scene` / `Object`; `(512, 384)` from bare `render()` |
| Samples per pixel | `4` from `Object`; `1` from bare `render()` and from an unconfigured `Scene` |
| Sample method | `"bilinear"` |
| Background | `(0, 0, 0, 0)` — transparent black, i.e. a clean cutout |
| Ambient | `0.8` on `Object`; `0.1` on bare `render()` |

Autofit runs whenever no camera matrix is supplied, or explicitly:

```python
bare = Scene("bare")
bare.append(Object(name="cube", mesh=mesh))
fitted = bare.render(resolution=(48, 48), samples_per_pixel=1)     # framed for you
print(np.round(fitted.camera_matrix[:3, 3], 2))                    # eye position

loose = bare.render(resolution=(48, 48), samples_per_pixel=1,
                    camera_matrix=look_at((10, 10, 10), (0, 0, 0)),
                    autofit=False)
print(np.round(loose.camera_matrix[:3, 3], 2))                     # left where you put it
```

`fit_padding` inflates the fitted point cloud about its AABB centre.
Bezier patches bulge outside their control hull, so they default to
`1.15`; everything else defaults to `1.0`.

```python
tight  = bare.render(resolution=(48, 48), samples_per_pixel=1, fit_padding=1.0)
padded = bare.render(resolution=(48, 48), samples_per_pixel=1, fit_padding=1.4)
print(np.round(tight.camera_matrix[:3, 3], 2))
print(np.round(padded.camera_matrix[:3, 3], 2))    # further back
```

---

## Production settings

Everything above uses toy resolutions. Real numbers:

<!-- notest -->

```python
scene = Scene.load_glb("hero.glb")
scene.configure(
    resolution        = (1920, 1080),
    samples_per_pixel = 16,          # 4x4 MSAA
    background        = (0.05, 0.05, 0.05, 1.0),
    fit_padding       = 1.10,
)
for obj in scene.objects:
    obj.sample_method = "bezier"   # PN-quad bicubic, ~3-4x slower
scene.render().save("hero.png")
scene.turntable("hero.mp4", n_frames=240, fps=60)
```

---

## Gotchas

```python
# A texture without a UV is an error, not a silent fallback.
broken = Scene("broken")
broken.append(Object(name="no_uv", mesh=mesh, texture=checker))
try:
    broken.render(resolution=(32, 32))
except ValueError as exc:
    print("expected:", exc)
```

```python
# samples_per_pixel must be 1 or a perfect square.
try:
    Object(name="bad", mesh=mesh, samples_per_pixel=3)
except ValueError as exc:
    print("expected:", str(exc)[:52])
```

```python
# to_image(resolution=...) is only honoured when NOTHING is cached.
cached = Scene("cached")
cached.append(Object(name="cube", mesh=mesh))
cached.render(resolution=(32, 32), samples_per_pixel=1)
print(cached.to_image(resolution=(200, 200)).size)   # (32, 32) -- the cached frame
```

- Per-Object `resolution` / `samples_per_pixel` / `background` apply only
  to `Object.render` / `to_image` / `imshow` / `turntable`. Inside a
  user-built `Scene`, the Scene's values win.
- `Camera.near_plane`, `far_plane`, `aspect_ratio` and `Scene.aspect_ratio`
  are stored but never read by the renderer. Same for
  `Object.cast_shadows` — there is no shadow pass yet.
- Shading is Lambertian diffuse + ambient. No specular, reflections,
  refractions, shadows, PBR or texture mip-mapping.
- `imshow()` needs OpenCV; `to_image()` / `save()` / `wireframe()` need
  PIL; the encoders need `ffmpeg` on `PATH`.
