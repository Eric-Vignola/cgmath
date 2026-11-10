# `cgmath.render` Cheatsheet

Every public name in the module, one small runnable example each. For the
narrative version see [`README.md`](README.md); for the rest of the
library see [`../README.md`](../README.md) and
[`../CHEATSHEET.md`](../CHEATSHEET.md).

Everything exported:

```text
Scene  aim  Frame  render  look_at  load_texture
```

The shapes a scene holds come from `cgmath.geometry` (`MeshData`, `UVData`,
`SkinData`, `MaterialData`, `CameraData`, `LightData`, `MorphData`), the
nodes from `cgmath.hierarchy` (`TransformData`, `HierarchyData`, `ClipData`).

Blocks run top to bottom as one script — later blocks reuse earlier
variables. Resolutions are deliberately tiny so the page renders fast;
see [Production settings](#production-settings) for real numbers.

---

## Contents

- [Setup](#setup) — the cube and checker every block reuses
- [One-line renders](#one-line-renders)
- [`Scene`](#scene) — [the fields](#the-fields) · [`add`](#add) · [nodes and `aim`](#nodes-and-aim) · [queries](#inspecting-a-scene) · [`configure`](#configure) · [background](#background--coverage-alpha) · [render](#rendering-a-scene) · [frame cache](#the-frame-cache) · [file loaders](#scene-file-loaders) · [saving](#saving-a-scene)
- [Materials](#materials) — [colour and faces](#colour-and-faces) · [texture and UV set](#texture-and-uv-set) · [shading knobs](#shading-knobs) · [baked wireframe](#baked-wireframe)
- [Cameras](#cameras) — [`CameraData`](#cameradata) · [orthographic](#orthographic) · [`look_at`](#the-free-function-look_at)
- [Lights](#lights) — [kinds](#kinds-and-falloff) · [the default rig](#the-default-3-point-rig)
- [Skins and blend shapes](#skins-and-blend-shapes) — [posing](#posing) · [`animate`](#animate)
- [`Frame`](#frame) — [fields](#frame-fields) · [interop](#numpy--pil-interop) · [wireframe overlay](#screen-space-wireframe-overlay) · [encoders](#encoders)
- [Turntables](#turntables) — [re-encoding is free](#re-encoding-is-free)
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
from cgmath.geometry import CameraData, LightData, MaterialData, MorphData, SkinData
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath.render import aim, Frame, load_texture, look_at, render, Scene

tmp = tempfile.mkdtemp(prefix="render_cheatsheet_")


def make_cube(name="cube"):
    """Triangulated unit cube + one [0,1]^2 UV island per face, the set
    naming the mesh by uuid."""
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
        name    = name,
    )
    uv_points, uv_indices = [], []
    for face in range(6):
        base = face * 4
        uv_points.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        uv_indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    uv = UVData(
        points    = np.asarray(uv_points, dtype=float),
        indices   = np.asarray(uv_indices, dtype=np.int64),
        counts    = counts,
        name      = "map1",
        mesh_uuid = mesh.uuid,
    )
    uv.resolution = (64, 64)      # the bake resolution a wireframe uses
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


def textured(mesh, texture):
    """a material putting ``texture`` on every face of ``mesh``"""
    return MaterialData(mesh_uuid=mesh.uuid, texture=texture, uv_set="map1", color=(1.0, 1.0, 1.0), diffuse=1.0)


mesh, uv = make_cube()
checker = make_checker()
print(mesh.points.shape, uv.points.shape, checker.shape)
```

---

## One-line renders

No camera, no light, no config. The renderer supplies a framed 3/4 view
and a 3-point rig.

```python
scene = Scene("hero").add(mesh, uv, textured(mesh, checker))
frame = scene.render(resolution=(64, 64), samples_per_pixel=1)
print(frame)                       # Frame(shape=(64, 64, 4), dtype=uint8, cam=True)
frame.save(os.path.join(tmp, "cube.png"))
```

`Scene.add` returns the scene, so building and rendering chain:

```python
print(Scene("bare").add(mesh).render(resolution=(32, 32), samples_per_pixel=1).shape)
```

---

## `Scene`

What an fbx holds, and the render settings, as one `Data`.

### The fields

| Field | Type | Holds |
|---|---|---|
| `name` | `str` | `"scene"` by default |
| `hierarchy` | `HierarchyData` | the nodes at rest; with skins, the bind pose |
| `clip` | `ClipList` | the takes over the nodes; the active one plays (a `ClipData` is taken as a list of one) |
| `meshes` | `MeshList` | each on the node its `parent_uuid` names, at identity for none |
| `uvs` | `UVList` | each on the mesh its `mesh_uuid` names |
| `skins` | `SkinList` | `mesh_uuid`; influences are joint names |
| `materials` | `MaterialList` | `mesh_uuid`; `faces` for a part of the mesh |
| `cameras` | `CameraList` | each on its node |
| `lights` | `LightList` | each on its node |
| `morphs` | `MorphList` | blend shape targets, by `mesh_uuid` |
| `default_camera` | `str` | the camera a render looks through; `None` for the first |
| `aspect_ratio` | `float` | reserved |
| `resolution` | int32 `(2,)` | `(500, 500)` |
| `samples_per_pixel` | `int` | `None` → the renderer's `1`; else `1` or a perfect square |
| `background` | float64 `(4,)` | RGBA; RGB is taken opaque; `None` → transparent black |
| `autofit`, `default_light`, `return_depth` | `bool` | `None` → the renderer's defaults |
| `angle_of_view` | `float` | the default camera's vertical FOV, when the scene has no camera |
| `fit_padding` | `float` | autofit margin; `None` → 1.15 for bezier, else 1 |
| `base_color` | float64 `(3,)` | faces no material covers; `(0.7, 0.7, 0.7)` |
| `sample_method` | `str` | `"bilinear"` or `"bezier"`, every mesh |
| `ambient` | `float` | `0.8`; added once after the lights |
| `twosided` | `bool` | `True`: back faces hit |
| `cast_shadows` | `bool` | reserved |
| `wireframe`, `wireframe_color`, `wireframe_thickness` | `bool`, int32 `(3,)`, `int` | edges baked over every colour map |
| `vertex_colors` | `str` | `"off"`, `"multiply"` or `"replace"`: a mesh's own colours, times the surface colour or in its place |

Every field is a constructor keyword and a plain attribute. A list field
takes one shape, a list of them or its list class; a value of another kind
raises `TypeError`, an array setting of another size `ValueError`.

```python
scene = Scene(
    "hero",
    resolution        = (64, 64),
    samples_per_pixel = 1,
    background        = (0.05, 0.05, 0.05, 1.0),
    base_color        = (0.8, 0.5, 0.2),
    ambient           = 0.6,
)
scene.meshes = mesh                          # one shape: a list of one
print(type(scene.meshes).__name__, len(scene.meshes), scene.background)

for bad in (lambda: setattr(scene, "meshes", [uv]),          # a UV set goes in uvs
            lambda: setattr(scene, "base_color", (1.0, 0.5)),
            lambda: Scene("x", samples_per_pixel=3)):
    try:
        bad()
    except (TypeError, ValueError) as exc:
        print("expected:", str(exc)[:60])
```

### `add`

`add(*items)` puts each item in the list it belongs in by its class — a
`TransformData` or `HierarchyData` into the hierarchy (made when the scene
has none), a `ClipData` or `ClipList` as the clip, every shape into its
list, a list of any of them item by item — and returns the scene.

```python
scene = Scene("hero", resolution=(64, 64), samples_per_pixel=1)
scene.add(mesh, uv, textured(mesh, checker))
print(len(scene.meshes), len(scene.uvs), len(scene.materials), scene.hierarchy)

try:
    scene.add({"not": "a shape"})
except TypeError as exc:
    print("expected:", str(exc)[:40])
```

### Nodes and `aim`

A node gets its uuid when it joins a hierarchy: add the node first, then
the shapes that name it. `aim(node, target)` turns a node to look down its
`-Z` at a world point from where it is, as a camera or a directed light
looks.

```python
hero_node = TransformData("hero", translate=(0.0, 0.5, 0.0), rotate=(0.0, 25.0, 0.0))
cam_node  = TransformData("cam", translate=(3.0, 2.0, 4.0))
scene.add(hero_node, cam_node)               # both have a uuid now
mesh.parent_uuid = hero_node.uuid            # the cube rides its node
aim(cam_node, (0.0, 0.5, 0.0))
scene.add(CameraData(name="cam", focal_length=35.0, parent_uuid=cam_node.uuid))
print(scene.hierarchy.name, scene.camera()[0].name)
print(np.round(-cam_node.world_matrix[2, :3], 3))   # the camera's forward: -row 2
```

A shape naming no node, or a node the scene lacks, sits at identity. A
node under a hidden node is hidden with it, and so are the shapes on it.

### Inspecting a scene

```python
camera, node = scene.camera()                # (CameraData, its node's uuid): named, default, or first
print(camera.name, scene.hierarchy[node].name)
print(scene.camera("cam")[0].name)
scene.default_camera = "cam"

print(scene.pose() is scene.hierarchy)       # the nodes a render places by: no clip, the hierarchy
placed, world = scene.placed_meshes()[0]     # [(mesh, world)] of every shown mesh
print(placed is mesh, np.round(world[3, :3], 2))    # itself, on its node: [0.  0.5 0. ]

lo, hi = scene.union_aabb()                  # world AABB of the shown meshes
print(np.round(lo, 2), np.round(hi, 2))
print(scene.union_points().shape)            # concatenated world points
print(scene.name)
```

Each mesh with what places, deforms and shades it, joined by uuid: a
`Renderable` a mesh, in the meshes' order. A view over the scene's own
records, built on every call; the skeleton is shared, so every mesh skinned
to it lists the same joints.

```python
from cgmath.render import Renderable

item = scene.renderables()[0]                # one Renderable per mesh
print(isinstance(item, Renderable), item.name)
print(item.mesh is mesh, item.node is hero_node, len(item.uvs), item.skin, len(item.joints), len(item.materials), len(item.morphs))
print(scene.renderable("hero") is not item, scene.renderable("hero").mesh is mesh)   # by the mesh's or the node's name; a new view each call
```

Hide the node and the mesh is left out:

```python
hero_node.visibility = False
print(scene.placed_meshes(), scene.union_points().shape)
hero_node.visibility = True
```

### `configure`

Chainable bulk setter. `None` values are skipped. The keys are the render
settings, the shading knobs, `default_camera` and `aspect_ratio`:

```text
resolution  samples_per_pixel  background  autofit  default_light  return_depth  angle_of_view  fit_padding
base_color  sample_method  ambient  twosided  cast_shadows  wireframe  wireframe_color  wireframe_thickness  vertex_colors
default_camera  aspect_ratio
```

```python
scene.configure(
    resolution        = (64, 64),
    samples_per_pixel = 1,
    background        = (0.05, 0.05, 0.05, 1.0),
    fit_padding       = 1.10,
    return_depth      = True,
    ambient           = 0.7,
).render()
print(scene.frame.has_depth)

try:
    scene.configure(bogus=1.0)
except AttributeError as exc:
    print("expected:", str(exc)[:40])
scene.configure(return_depth=False)
```

Per-call kwargs always beat the persistent config:

```python
print(scene.render(resolution=(32, 32)).shape)   # (32, 32, 4), not (64, 64, 4)
```

### Background & coverage alpha

Every render is RGBA and the alpha channel is real MSAA coverage
(`hits / samples_per_pixel`). RGB triples are promoted to opaque, and
the value is stored as a float64 `(4,)` array (`resolution` as int32
`(2,)`).

```python
scene.background = (0.1, 0.2, 0.3)
print(scene.background)                      # [0.1 0.2 0.3 1. ]

scene.background = (0.0, 0.0, 0.0, 0.0)      # transparent miss pixels
cutout           = scene.render(resolution=(64, 64), samples_per_pixel=4)
alpha            = cutout.array[..., 3]
print(alpha.min(), alpha.max())                                         # 0 255: misses clear, the cube solid
print(sorted(set(np.unique(alpha).tolist()) & {0, 63, 127, 191, 255}))  # 0 to 4 of 4 samples hit
cutout.save(os.path.join(tmp, "cutout.png"))
```

### Rendering a scene

```python
frame = scene.render()                                   # honours configure()
frame = scene.render(resolution=(48, 48), samples_per_pixel=1)
frame = scene.render(resolution=(48, 48), samples_per_pixel=1, return_depth=True)
print(frame.has_depth, np.nanmin(frame.depth) > 0)       # NaN at miss pixels

image = scene.to_image(resolution=(48, 48))              # PIL; lazy-renders
print(image.size, image.mode)

path = os.path.join(tmp, "out.png")
scene.render(output=path, resolution=(32, 32))           # render + save
print(os.path.exists(path))
```

`imshow()` opens an OpenCV window:

<!-- notest -->
```python
scene.imshow(resolution=(500, 500))
```

### The frame cache

`scene.frame` / `scene.buffer` hold the last render, so repeated
`to_image()` / `imshow()` calls do not re-render. The cache
auto-invalidates when any render input changes — a setting, a shape's
fields, a node's transform or visibility anywhere above a shape, a shape or
node added.

```python
scene.render(resolution=(48, 48), samples_per_pixel=1)
print(scene.frame is not None, scene.buffer.shape)

hero_node.translate = (0.5, 0.0, 0.0)        # a node's transform
print(scene.frame)                           # None -- invalidated

scene.render(resolution=(48, 48), samples_per_pixel=1)
scene.cameras[0].focal_length = 50.0         # a lens change
print(scene.frame)                           # None

scene.render(resolution=(48, 48), samples_per_pixel=1)
scene.materials[0].color = (0.5, 0.5, 0.5)   # a material
print(scene.frame)                           # None
```

The one case it cannot see is **in-place** mutation of array contents:

```python
scene.render(resolution=(48, 48), samples_per_pixel=1)
mesh.points[0] += 0.25                       # in-place: cache does NOT notice
print(scene.frame is not None)                          # True -- stale
scene.render(resolution=(48, 48), samples_per_pixel=1)  # refresh explicitly
mesh.points[0] -= 0.25
```

### Scene file loaders

`load_obj` reads every mesh with its UV sets, at identity; `load_glb` the
meshes, their UV sets, skins with the bind pose and a material per base
colour map; `load_fbx` everything `cgmath.formats.load_fbx` reads.
`save_fbx` writes the fields (not the render settings).

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

from_obj = Scene.load_obj(obj_path, name="from_obj")
print(from_obj.name, from_obj.meshes.name, len(from_obj.uvs), from_obj.uvs[0].mesh_uuid == from_obj.meshes[0].uuid)
from_obj.render(resolution=(48, 48), samples_per_pixel=1)
```

<!-- notest -->
```python
Scene.load_glb("hero.glb", name="hero", scale_factor=100.0, load_skin=True, extract_texture=True)
Scene.load_fbx("hero.fbx")                   # needs the FBX SDK
Scene.load_fbx("hero.fbx").save_fbx("copy.fbx", as_ascii=False)
```

### Saving a scene

`save` / `load` in pkl, npz or json keep every list as its class, every
node with its uuid and parenting, every setting. A loaded Scene has no
cached render.

```python
path  = scene.save(os.path.join(tmp, "scene.npz"))  # or .pkl / .json
again = Scene.load(path)
print([type(x).__name__ for x in (again.hierarchy, again.meshes, again.uvs, again.materials, again.cameras)])
print(again.name, again.default_camera, again.resolution, again.frame)
print(again.meshes[0].parent_uuid == again.hierarchy["hero"].uuid)   # the shapes still name their nodes
print(again == scene)
```

---

## Materials

A `MaterialData` is Maya's lambert or phong as an fbx holds it: a colour
and its diffuse factor, an optional colour map and the UV set it reads,
the faces it covers. The raytracer shades each face by the material
covering it; a face no material covers takes `scene.base_color`.

### Colour and faces

```python
red  = MaterialData(name="red", mesh_uuid=mesh.uuid, faces=[0, 1, 2, 3, 4, 5], color=(1.0, 0.0, 0.0), diffuse=1.0)
blue = MaterialData(name="blue", mesh_uuid=mesh.uuid, faces=[6, 7, 8, 9, 10, 11], kind="phong", color=(0.0, 0.0, 1.0), diffuse=0.5)
two  = Scene("two", resolution=(64, 64), samples_per_pixel=1, ambient=1.0).add(mesh, red, blue)
seen = {tuple(x) for x in two.render(default_light=False).array[..., :3].reshape(-1, 3).tolist()}
print((255, 0, 0) in seen, (0, 0, 127) in seen or (0, 0, 128) in seen)   # colour times diffuse
```

`faces` must lie in the mesh; `covers(mesh)` gives the face mask the
renderer uses:

```python
print(red.covers(mesh).sum(), MaterialData(mesh_uuid=mesh.uuid).covers(mesh).all())
try:
    MaterialData(mesh_uuid=mesh.uuid, faces=[40]).covers(mesh)
except ValueError as exc:
    print("expected:", str(exc)[:40])
```

### Texture and UV set

`texture` is an `(H, W, C)` array, kept in its dtype when it is uint8,
uint16 or float32 (other floats become float32); `uv_set` names the mesh's
UV set it reads (`None` for the mesh's first); `wrap_u` / `wrap_v` are
`"repeat"` or `"clamp"` for UVs outside `[0, 1]`.

```python
mapped = textured(mesh, checker)
print(mapped.texture.dtype, mapped.textured, mapped.uv_set, mapped.wrap_u)
clamped = MaterialData(mesh_uuid=mesh.uuid, texture=checker.astype(np.float32) / 255.0, wrap_u="clamp")
print(clamped.texture.dtype, clamped.wrap_u)

try:
    Scene("no_uv").add(mesh, mapped).render(resolution=(16, 16))   # a texture needs a UV set
except ValueError as exc:
    print("expected:", str(exc)[:48])
```

A material of another mesh is not this mesh's: the renderer matches on
`mesh_uuid`.

### Shading knobs

The knobs are scene-wide, one value for every mesh: `base_color`,
`sample_method` (`"bilinear"` flat patches or `"bezier"` PN-quad bicubic,
a `SampleMethod` enum stored as its string), `ambient`, `twosided`,
`cast_shadows` (reserved) and `vertex_colors`.

```python
flat = Scene("flat", base_color=(0.9, 0.3, 0.2), ambient=1.0).add(mesh)
lit  = Scene("lit", base_color=(0.9, 0.3, 0.2), ambient=0.1).add(mesh)
print(flat.render(resolution=(48, 48), samples_per_pixel=1).array.mean() >
      lit.render(resolution=(48, 48), samples_per_pixel=1).array.mean())

from cgmath.geometry.mesh import SampleMethod

smooth = Scene("smooth", sample_method=SampleMethod.BEZIER).add(mesh)
print(smooth.sample_method)                  # bezier
```

### Vertex colours

A mesh's `colors` (RGBA, one a vertex or one a face corner through
`color_indices`) take part through `vertex_colors`: `"multiply"` scales the
material colour and map by them, `"replace"` shows them alone (Maya's Color
Set display), `"off"` ignores them, the default, as Maya's viewport does
until asked. Blended over a face as the UVs are; the alpha channel is
carried, not rendered.

```python
tinted        = mesh.copy()
tinted.colors = np.tile([1.0, 0.5, 0.0, 1.0], (tinted.point_count, 1))   # orange, a colour a vertex
shown         = Scene("shown", vertex_colors="replace").add(tinted).render(resolution=(32, 32), default_light=False)
plain         = Scene("plain", base_color=(1.0, 0.5, 0.0)).add(mesh).render(resolution=(32, 32), default_light=False)
print(np.array_equal(shown.array, plain.array))      # True: the vertex colour is the surface colour
```

### Baked wireframe

`scene.wireframe = True` rasterizes every mesh's edges into a copy of its
colour map **in UV space** — or over the material's colour when it has no
map, or over `base_color` for a mesh with no material. Cheap, cached, and
it rides along through turntables for free. Needs a UV set.

```python
wired = Scene("wired", wireframe=True, wireframe_color=(255, 80, 0), wireframe_thickness=3)
wired.add(mesh, uv, textured(mesh, checker))
baked = wired._texture(wired.materials[0], uv)      # the baked map, (H, W, 3) float32 in [0, 1]
print(baked.shape, baked.dtype)
wired.render(resolution=(64, 64), samples_per_pixel=1)
```

For a sharper screen-space line instead, see
[the Frame overlay](#screen-space-wireframe-overlay).

---

## Cameras

### `CameraData`

Maya's camera shape, on a node. The vertical angle of view the raytracer
looks through comes from the focal length and the vertical film aperture.

| Argument | Default | Meaning |
|---|---|---|
| `name` | `None` | the camera's name; `default_camera` picks one by it |
| `focal_length` | `35.0` | mm |
| `horizontal_film_aperture`, `vertical_film_aperture` | `1.41732`, `0.94488` | inches, Maya's defaults |
| `film_fit`, `film_offset`, `lens_squeeze_ratio`, `camera_scale` | Maya's | carried |
| `near_clip_plane`, `far_clip_plane` | `0.1`, `10000.0` | carried; no clipping pass |
| `orthographic` | `False` | parallel rays |
| `orthographic_width` | `10.0` | what the frame spans across, in scene units |
| `parent_uuid` | `None` | its node |

```python
cam = CameraData(name="cam", focal_length=28.0)
print(round(cam.vertical_angle_of_view, 2), round(cam.horizontal_angle_of_view, 2))

cam_scene = Scene("with_camera", resolution=(64, 64), samples_per_pixel=1).add(mesh, uv, textured(mesh, checker))
eye       = TransformData("eye", translate=(3.0, 3.0, 3.0))
cam_scene.add(eye)
aim(eye, (0.0, 0.0, 0.0))                   # keep position, aim at a world point
cam.parent_uuid = eye.uuid
cam_scene.add(cam)
print(cam_scene.render().shape)
```

A scene renders through `default_camera` by name, else its first camera;
a scene with none gets the default 3/4 view, fit to the scene.

### Orthographic

```python
ortho_scene = Scene("ortho", resolution=(48, 48), samples_per_pixel=1).add(mesh)
eye         = TransformData("eye", translate=(0.0, 0.0, 4.0))
ortho_scene.add(eye)
aim(eye, (0.0, 0.0, 0.0))
ortho_scene.add(CameraData(name="ortho", orthographic=True, orthographic_width=2.5, parent_uuid=eye.uuid))
print(ortho_scene.render(return_depth=True).depth[24, 24])   # the front face, 3.5 away
```

The width is what the frame spans horizontally; the height follows the
frame's aspect.

### The free function `look_at`

Same math as `aim`, no node — returns a **column-major** camera-to-world
matrix (eye in the last *column*), which is what `render(camera_matrix=...)`
wants.

```python
m = look_at(eye=(4, 3, 4), target=(0, 0, 0), up=(0, 1, 0))
print(m.shape, np.round(m[:3, 3], 3))        # column 3 = eye
frame = render(mesh=mesh, camera_matrix=m, resolution=(48, 48))
```

| Matrix | Layout | Eye at |
|---|---|---|
| `look_at()`, `render(camera_matrix=)`, `Frame.camera_matrix` | column-major (OpenGL) | `m[:3, 3]` |
| `node.world_matrix` | row-major (Maya) | `m[3, :3]` |

`aim` transposes for you: `aim(node, target)` is `node.world_matrix =
look_at(eye, target).T`.

---

## Lights

### Kinds and falloff

| `kind` | Comes from | Falloff |
|---|---|---|
| `"point"` | the node's world position | `1/r²` when `decay_rate` is not `0` (Maya's none), constant at `0` |
| `"directional"` | down the node's `-Z` axis (a sun) | none — parallel rays |
| `"spot"` | as a point light, with one warning: the raytracer draws no cone | as point |

```python
key_node  = TransformData("key", translate=(2.0, 3.0, 2.0))
sun_node  = TransformData("sun", rotate=(-45.0, 0.0, 0.0))
lit_scene = Scene("lit", resolution=(64, 64), samples_per_pixel=1).add(mesh, uv, textured(mesh, checker))
lit_scene.add(key_node, sun_node)
key = LightData(name="key", kind="point", intensity=1e2, color=(1.0, 0.95, 0.9), decay_rate=2, parent_uuid=key_node.uuid)
sun = LightData(name="sun", kind="directional", intensity=0.6, parent_uuid=sun_node.uuid)
lit_scene.add(key, sun)

print(key.color, key.decay)                  # [1.   0.95 0.9 ] quadratic
print(lit_scene.render())
```

What the raytracer takes is a dict per light, as the low-level `render()`'s
`point_light=` wants them:

```python
dicts = lit_scene._render_inputs().lights
print(dicts[0]["kind"], dicts[0]["position"], dicts[0]["falloff"])
print(dicts[1]["kind"], np.round(dicts[1]["direction"], 3))   # infinite, down the node's -Z
```

A kind outside Maya's is rejected at construction:

```python
try:
    LightData(name="bad", kind="area")
except ValueError as exc:
    print("expected:", str(exc)[:40])
```

Point-light intensity with falloff is physical: you want roughly
`distance²`. Parent a light's node to a camera's for a headlight that
tracks the view:

```python
head_cam   = TransformData("head_cam", translate=(0.0, 1.0, 4.0))
lamp       = TransformData("headlight")
head_scene = Scene("headlight", resolution=(48, 48), samples_per_pixel=1).add(mesh, head_cam, lamp)
lamp.set_parent("head_cam", world_space=False)
head_scene.add(CameraData(name="head_cam", parent_uuid=head_cam.uuid))
head_scene.add(LightData(name="headlight", kind="point", intensity=30.0, decay_rate=2, parent_uuid=lamp.uuid))
print(np.round(lamp.world_matrix[3, :3], 2))   # inherits the camera's position
head_scene.render()
```

### The default 3-point rig

With **no** light in the scene and `default_light` not disabled, a
key/fill/rim rig sized to the scene AABB is handed to the raytracer for the
call — `scene.lights` is never touched.

| Light | Direction from centroid | Intensity |
|---|---|---|
| key | `(+1, +1, +1) · d` | `d²` |
| fill | `(-1, +0.5, +1) · d` | `0.4 · d²` |
| rim | `(0, +0.5, -1) · d` | `0.3 · d²` |

where `d = 1.5 · ‖aabb_max − aabb_min‖`. Because it scales with the
bounding box, a 1-unit cube and a 100-unit ship both light correctly.

```python
auto_lit = Scene("auto", resolution=(48, 48), samples_per_pixel=1).add(mesh, uv, textured(mesh, checker))
print(auto_lit.lights)                # None -- the rig is transient
auto_lit.render()
print(auto_lit.lights)                # still None

auto_lit.render(default_light=False)  # no light at all: the flat colour, unshaded
```

---

## Skins and blend shapes

### Posing

A `SkinData` names its mesh by `mesh_uuid`, its influences by the joints'
names. The scene's `hierarchy` is the bind pose; a `clip` over the nodes
poses it, and `placed_meshes` hands out a deformed copy in world space —
the scene's mesh is never changed. A `MorphData` moves the points by its
`weight` first, Maya's way (an in-between is a knot the weight blends
through).

```python
root   = TransformData("root", node_type="joint")
tip    = TransformData("tip", node_type="joint", translate=(0.0, 1.0, 0.0))
rigged = Scene("rigged").add(root, tip)
tip.set_parent(root, world_space=False)

body, _ = make_cube("body")
weights = np.column_stack([1.0 - (body.points[:, 1] + 0.5), body.points[:, 1] + 0.5])   # bottom to root, top to tip
rigged.add(body, SkinData(weights=weights, influences=["root", "tip"], mesh_uuid=body.uuid))
rigged.add(MorphData("puff", [[0.0, 0.0, 0.5]] * 4, [4, 5, 6, 7], mesh_uuid=body.uuid, weight=0.5))

at_bind = rigged.placed_meshes()[0][0]
print(np.round(at_bind.points[4:, 2], 2))    # the morph at half weight: 0.75

clip                      = ClipData(rigged.hierarchy, frames=3, fps=24.0)
clip.frames["tip"].rotate = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 45.0], [0.0, 0.0, 90.0]])
rigged.clip               = clip
posed                     = rigged.placed_meshes(clip.start_frame + 2)[0][0]  # frame 2: the top bent 90 degrees
print(np.round(posed.points[2], 2), np.round(body.points[2], 2))   # moved; the mesh itself stays
```

A skinned mesh in a scene with no hierarchy raises: the skin takes the
bind pose.

```python
try:
    Scene("unbound").add(body, rigged.skins[0]).placed_meshes()
except ValueError as exc:
    print("expected:", str(exc)[:48])
```

### `animate`

`animate(output_pattern, clip=None, ...)` renders the active take of
`clip`, or of the scene's own, frame by frame: skins deform, every other
shape rides its node. The orbit arguments (`rotation_axis`, `start_angle`,
`end_angle`), `fit`, `camera_name`, `fps` (the clip's by default) and the
output are as `turntable`'s. The scene is handed back as it was.

```python
written = rigged.animate(os.path.join(tmp, "anim", "bend.{frame:04d}.png"), resolution=(32, 32), samples_per_pixel=1)
print(len(written), os.path.basename(written[0]))
print(rigged.animate(os.path.join(tmp, "anim2", "bend.png"), clip=clip, resolution=(32, 32), samples_per_pixel=1)[1][-8:])

try:
    Scene("still").add(body).animate(os.path.join(tmp, "x.png"))
except ValueError as exc:
    print("expected:", str(exc)[:40])
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
print(round(frame.angle_of_view, 2))         # vertical FOV used

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
    scene,                              # a Scene (its shown meshes, placed) or a MeshData
    color          = (255, 255, 255, 220),  # RGBA 0-255; default (0, 0, 0, 200)
    width          = 2,                     # default 1
    unique_edges   = True,                  # draw each shared edge once
    cull_backfaces = True,                  # False = X-ray, every edge drawn
)
wire_frame.save(os.path.join(tmp, "wires.png"))

xray      = frame.wireframe(scene, cull_backfaces=False)
from_mesh = frame.wireframe(mesh)         # points treated as world-space
print(wire_frame, xray, from_mesh)
```

It needs a camera, so a hand-built Frame is rejected:

```python
try:
    manual.wireframe(scene)
except ValueError as exc:
    print("expected:", str(exc)[:60])
```

`Frame.wireframe()` (sharp screen-space line, per-frame cost) is a
different feature from `scene.wireframe = True` (baked into the colour
maps, cached, survives turntables).

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
| `camera_name` | `None` | Which camera supplies the base orientation |
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

Give a path with no `{frame}` placeholder and one is injected as
`<stem>.{frame:04d}<ext>`.

### Re-encoding is free

`turntable()` and `animate()` keep their raw frames in a scratch folder.
The same frames asked for again are read back, not rendered: only `fps`,
`background`, `threads` and the output path may differ (`threads` changes
how a frame renders, not the frame). Anything else, or anything that
clears the frame cache, renders again. The folder goes with its scene, and
all of it at exit.

```python
spin  = os.path.join(tmp, "tt5", "a.{frame:04d}.png")
again = os.path.join(tmp, "tt5", "b.{frame:04d}.png")
first = scene.turntable(spin, n_frames=3, resolution=(32, 32), samples_per_pixel=1)
reuse = scene.turntable(again, n_frames=3, resolution=(32, 32), samples_per_pixel=1)  # read back
print(open(first[0], "rb").read() == open(reuse[0], "rb").read())                    # True
```

Like the frame cache it cannot see in-place edits; drop the kept frames
by hand after one:

```python
scene.clear_render_cache()  # the next turntable renders
```

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
| `point_light` | `None` | One dict or a list of dicts |
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
| `threads` | `None` | CPU threads for the frame: `None` = numba's thread count (every core), `1` = the calling thread only, capped at `NUMBA_NUM_THREADS` and the cores; the same bytes whatever the value |
| `frame` | `None` | With a Scene holding a clip: the frame of its active take the shapes are posed at |

The legacy entries take the shading knobs per mesh (`sample_method`,
`wrap`, `ambient`, `twosided`, `base_color`); a Scene's are its own.

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

Every core, or one thread. A frame renders in bands of pixel rows side by
side; `threads=1` keeps it in the calling thread. The pixels and the depth
are the same bytes either way. `Scene.render`, turntables and `animate`
pass `threads=` on to `render()`.

```python
every_core = render(mesh, uv, checker, resolution=(96, 64), samples_per_pixel=4,
                    return_depth=True)
one_thread = render(mesh, uv, checker, resolution=(96, 64), samples_per_pixel=4,
                    return_depth=True, threads=1)
print(every_core.array.tobytes() == one_thread.array.tobytes(),
      every_core.depth.tobytes() == one_thread.depth.tobytes())    # True True
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
truncated to RGB, `uint8` divides by 255 and `uint16` by 65535. The scene
reads every material's colour map through it.

```python
tex_path = os.path.join(tmp, "checker.png")
Frame(array=np.dstack([checker, np.full(checker.shape[:2], 255, np.uint8)])).save(tex_path)

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
| Resolution | `(500, 500)` from `Scene`; `(512, 384)` from bare `render()` |
| Samples per pixel | `1` from bare `render()` and from an unconfigured `Scene` |
| Sample method | `"bilinear"` |
| Background | `(0, 0, 0, 0)` — transparent black, i.e. a clean cutout |
| Ambient | `0.8` on a `Scene`; `0.1` on bare `render()` |
| Colour | `base_color` 0.7 grey on a face no material covers |

Autofit runs whenever no camera matrix is supplied, or explicitly:

```python
bare   = Scene("bare").add(mesh)
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
scene = Scene.load_fbx("hero.fbx")
scene.configure(
    resolution        = (1920, 1080),
    samples_per_pixel = 16,        # 4x4 MSAA
    background        = (0.05, 0.05, 0.05, 1.0),
    fit_padding       = 1.10,
    sample_method     = "bezier",  # PN-quad bicubic, ~3-4x slower
)
scene.render().save("hero.png")
scene.turntable("hero.mp4", n_frames=240, fps=60)
scene.animate("walk.mp4")            # the file's active take
```

---

## Gotchas

```python
# A texture without a UV set is an error, not a silent fallback.
broken = Scene("broken").add(mesh, textured(mesh, checker))
try:
    broken.render(resolution=(32, 32))
except ValueError as exc:
    print("expected:", str(exc)[:48])
```

```python
# samples_per_pixel must be 1 or a perfect square.
try:
    Scene("bad", samples_per_pixel=3)
except ValueError as exc:
    print("expected:", str(exc)[:52])
```

```python
# to_image(resolution=...) is only honoured when NOTHING is cached.
cached = Scene("cached").add(mesh)
cached.render(resolution=(32, 32), samples_per_pixel=1)
print(cached.to_image(resolution=(200, 200)).size)   # (32, 32) -- the cached frame
```

```python
# A node gets its uuid when it joins a hierarchy: a shape built before
# that names None, and sits at identity.
late = TransformData("late", translate=(5.0, 0.0, 0.0))
early_mesh, _ = make_cube("early")
early_mesh.parent_uuid = late.uuid            # None: not in a scene yet
print(early_mesh.parent_uuid)
```

- The shading knobs are scene-wide. A per-mesh colour or map is a
  `MaterialData`; everything else — `sample_method`, `ambient`,
  `twosided`, the wireframe — is one value for the scene.
- `CameraData`'s clip planes, `film_fit`, `aspect_ratio` and
  `Scene.aspect_ratio` are carried but never read by the renderer. Same for
  `cast_shadows` — there is no shadow pass yet.
- Shading is Lambertian diffuse + ambient. No specular, reflections,
  refractions, shadows, PBR or texture mip-mapping; a spot light renders as
  a point light.
- `imshow()` needs OpenCV; `to_image()` / `save()` / `wireframe()` need
  PIL; the encoders need `ffmpeg` on `PATH`; `load_glb` needs trimesh,
  `load_fbx` the Autodesk FBX SDK.
