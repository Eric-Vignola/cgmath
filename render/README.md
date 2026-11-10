# `cgmath.render` — the software raytracer

A numpy software raytracer over the record an fbx holds.

A **Scene** is what `cgmath.formats.load_fbx` reads and `save_fbx` writes,
in one object: the **nodes** (a `HierarchyData` at rest, or their takes as
a `ClipList`), the **meshes** on the nodes, the **UV sets**, **skins**,
**materials** and **blend shape targets** on the meshes, the **cameras**
and **lights** on their nodes — plus the render settings. Every shape sits
where the node its `parent_uuid` names puts it. Hit "render" and you get a
`Frame` (image + optional depth) back. No DCC required.

- [`CHEATSHEET.md`](CHEATSHEET.md) — every public class, function, property
  and method in this module, each with a small copy-paste example.
- [`../README.md`](../README.md) — the parent `cgmath` package map.

Every Python block below runs, top to bottom, as one script — they share
a namespace, notebook style. Start here:

```python
import numpy as np
from cgmath.geometry import CameraData, LightData, MaterialData
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.hierarchy import TransformData
from cgmath.render import aim, Scene

# A unit cube: 8 points, 6 quads.  There are no mesh assets in this
# repo, so every example builds its geometry inline.
points = np.array(
    [
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
    ]
)
indices = np.array([0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4])
counts  = np.array([4, 4, 4, 4, 4, 4])
cube    = MeshData(points=points, indices=indices, counts=counts, name="cube")

# One UV square per face (all six faces share the same four corners). A UV
# set names its mesh by uuid: a shape has one from birth.
cube_uv = UVData(
    points    = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    indices   = np.tile([0, 1, 2, 3], 6),
    counts    = counts,
    name      = "map1",
    mesh_uuid = cube.uuid,
)

scene = Scene("hero").add(cube, cube_uv)
scene.render().save("cube.png")
```

That is the whole minimum-viable render. No camera, no light, no rig —
the renderer brings its own. With a real asset on disk it is one line:

<!-- notest -->
```python
Scene.load_fbx("hero.fbx").imshow()
```

`Scene.add()` returns the scene, so it chains; it takes any of the shapes
and nodes, in any order, and puts each in the list it belongs in.

---

## What you actually get for free

When you don't supply a camera or a light, the renderer fills them in:

- **Default camera**: 3/4 elevated front-right view (eye direction
  `(+x, +y·0.7, +z)`), framed so the scene fills the frustum (autofit).
- **Default lighting**: a real **3-point rig** built around the scene's
  bounding box.
  - **Key** — front-upper-right, full intensity, 1/r² falloff
  - **Fill** — front-upper-left, 40% of key
  - **Rim** — back-upper, 30% of key
  All three sit at `1.5 × the AABB diagonal` from the centroid with
  intensity `distance²`, so it looks right at any scale. The rig is handed
  to the raytracer for the call; `scene.lights` stays as it was.
- **Default resolution**: `(500, 500)`. (The bare `render()` function
  defaults to `(512, 384)`.) Bump it via `scene.configure(resolution=(W, H))`.
- **Default samples-per-pixel**: a `Scene` leaves it unset, which lands on
  the renderer's `1` — set `scene.configure(samples_per_pixel=4)` for
  clean silhouettes and real coverage alpha.
- **Default shading**: a face no material covers takes the scene's
  `base_color`, 0.7 grey; a mesh with a `MaterialData` takes the
  material's colour times its diffuse, and its colour map through the UV
  set it names.
- **Default sample method**: `"bilinear"` (flat patches), scene-wide.
  Switch to `"bezier"` for smoothed PN-Quad bicubic surfaces.
- **Default background**: `(0, 0, 0, 0)` — **transparent** black, so a
  straight `.png` save is already a cutout. Pass any RGB (auto-promoted
  to opaque RGBA) or a full RGBA tuple to change it.
- **Every core**: a frame renders in bands of pixel rows side by side.
  `threads=1` (on `render()`, `scene.render()`, turntables and
  animations) keeps it in the calling thread; the pixels and the depth
  are the same bytes either way.

You can override any of these — but the defaults exist so you can iterate
quickly without rigging anything.

---

## The mental model — the fbx bundle

A `Scene` is a `Data` with nine fields, the ones `cgmath.formats.load_fbx`
gives back, and the render settings beside them.

| Field | Holds | Placed by |
|---|---|---|
| `hierarchy` | `HierarchyData`: the nodes at rest; with skins, the bind pose | — |
| `clip` | `ClipList`: the takes over the nodes; the active one plays | — |
| `meshes` | `MeshList` | each mesh's `parent_uuid` names its node |
| `uvs` | `UVList` | each set's `mesh_uuid` names its mesh |
| `skins` | `SkinList` | `mesh_uuid`; influences are joint names |
| `materials` | `MaterialList` | `mesh_uuid`, and `faces` for a part of the mesh |
| `cameras` | `CameraList` | `parent_uuid` names the camera's node |
| `lights` | `LightList` | `parent_uuid` names the light's node |
| `morphs` | `MorphList` (blend shape targets) | `mesh_uuid` |

A shape naming no node, or a node the scene lacks, sits at identity. The
nodes are the same `TransformData` as the rest of the library — SRT,
parenting and rotate orders work as they do everywhere else; a node under a
hidden node is hidden with it, and so are the shapes on it.

```python
from cgmath.render import aim, Frame, load_texture, look_at, render, Scene
```

Those six names are everything that's exported from `cgmath.render`. The
shapes come from `cgmath.geometry`, the nodes from `cgmath.hierarchy`.

One rule to remember: **a node gets its uuid when it joins a hierarchy.**
Add the node to the scene first, then build the shapes that name it.

```python
hero             = TransformData("hero", translate=(0.0, 0.5, 0.0), rotate=(0.0, 30.0, 0.0))
scene            = Scene("placed").add(hero)  # hero.uuid is set now
cube.parent_uuid = hero.uuid                  # the cube rides the node
scene.add(cube, cube_uv)
print(scene.placed_meshes()[0][1][3, :3])  # [0.  0.5 0. ]: the mesh's world, row-major
```

The joins are done for you, one mesh at a time: `scene.renderables()` gives a
`Renderable` per mesh, in the meshes' order, with its node, UV sets, skin, the
skin's joints, materials (textures inside) and targets. It is a view over the
scene's records, built on every call, so it never goes stale; the skeleton is
shared, every mesh skinned to it lists the same joints. `scene.renderable(name)`
picks one by its mesh's or its node's name.

```python
for item in scene.renderables():
    print(item.name, item.node.name, item.mesh is cube, len(item.uvs), item.skin, len(item.materials))
print(scene.renderable("hero").node is hero)   # the mesh is unnamed: the node's name finds it
```

A scene built a second time in Maya beside the first needs uuids of its own
(rig gives the nodes it makes the data's uuids, and Maya wants them unique):
`scene.renew_uuids()` mints fresh ones on every node and shape and rewires
every link, in place, the records staying the same objects.

```python
old     = hero.uuid
mapping = scene.renew_uuids()                  # {old: new}
print(hero.uuid == mapping[old], cube.parent_uuid == hero.uuid, scene.renderable("hero").node is hero)
```

---

## Materials

A `MaterialData` (from `cgmath.geometry`) is Maya's lambert or phong as an
fbx holds it: a colour, a diffuse factor, an optional colour map and the UV
set it reads, and the faces it covers.

```python
checker = np.indices((32, 32)).sum(0) % 2                 # 0/1 squares
checker = np.repeat(checker[..., None], 3, axis=2).astype(np.float32)

skin = MaterialData(
    name      = "skin",
    mesh_uuid = cube.uuid,        # the mesh it covers
    faces     = [0, 1, 2],        # these faces; None for the whole mesh
    color     = (1.0, 1.0, 1.0),  # times `diffuse` is the base colour
    diffuse   = 1.0,
    texture   = checker,          # (H, W, C) uint8 / uint16 / float32, or None
    uv_set    = "map1",           # the mesh's UV set the map reads; None for its first
    wrap_u    = "repeat",         # or "clamp", for UVs outside [0, 1]
)
lid = MaterialData(name="lid", mesh_uuid=cube.uuid, faces=[3, 4, 5], kind="phong", color=(0.9, 0.2, 0.1))
scene.add(skin, lid)
```

The raytracer shades each face by the material covering it: the colour map
where there is one, else `color × diffuse`; a face no material covers takes
`scene.base_color`. A texture without a UV set to read it is an error, not a
silent fallback. A texture array keeps its dtype when it is uint8, uint16 or
float32 (other floats become float32).

What is **scene-wide** rather than per material: the shading knobs.

```python
scene.configure(
    base_color    = (0.7, 0.7, 0.7),  # faces no material covers; float64 (3,)
    sample_method = "bilinear",       # or "bezier", or a SampleMethod enum, every mesh
    ambient       = 0.8,              # 0..1; 0 = pure shadow, 1 = unshaded
    twosided      = True,             # render back faces too
    cast_shadows  = True,             # reserved; no shadow pass yet
    # a wireframe baked over every colour map (or over the base colour) in UV space
    wireframe           = False,
    wireframe_color     = (0, 0, 0),   # 0-255 ints; int32 (3,)
    wireframe_thickness = 1,
    # a mesh's own colours: "multiply" them into the surface colour, "replace" it, or "off"
    vertex_colors       = "off",
)
```

A mesh's `colors` (RGBA, one a vertex or one a face corner through
`color_indices`) take part through `vertex_colors`: `"multiply"` scales the
material colour and map by the vertex colour, `"replace"` shows the vertex
colour alone (Maya's Color Set display); blended over a face as the UVs
are. Off by default, as Maya's viewport shows no colour set until asked.

Colors and `resolution` take any sequence and are kept as numpy arrays;
a value of the wrong size raises `ValueError`.

---

## Cameras

A `CameraData` is Maya's camera shape: a focal length and a film back,
which give the angle of view the raytracer looks through. It sits on a
node; `aim()` turns the node to look at a point.

```python
cam = TransformData("cam", translate=(3.0, 2.0, 4.0))
scene.add(cam)
aim(cam, (0.0, 0.5, 0.0))                       # look from where it is at a world point
camera = CameraData(
    name                     = "cam",
    focal_length             = 35.0,     # mm; with the film back, the angle of view
    horizontal_film_aperture = 1.41732,  # inches, Maya's defaults
    vertical_film_aperture   = 0.94488,
    orthographic             = False,    # True + orthographic_width for ortho
    orthographic_width       = 10.0,     # cm across the frame when orthographic
    parent_uuid              = cam.uuid,
)
scene.add(camera)
print(round(camera.vertical_angle_of_view, 2))  # 37.85: what the raytracer uses as its FOV
```

Convention: a camera looks down its node's `-Z`, `+Y` is up, `+X` is right
(OpenGL and Maya alike). A scene renders through `default_camera` by name,
else its first camera; `scene.camera()` says which.

**Two matrix layouts, don't mix them up.** `node.world_matrix` is
row-major like every node — the eye sits at `world_matrix[3, :3]`. The free
function `look_at()` and `Frame.camera_matrix` are column-major OpenGL — the
eye sits at `[:3, 3]`. Transpose when you cross between them; `aim()` does.

```python
m = look_at((0.0, 2.0, 5.0), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))  # (4, 4) camera-to-world
print(m[:3, 3])                     # OpenGL column-major: eye is here

cam.world_matrix = m.T              # row-major for the node
print(cam.world_matrix[3, :3])      # -> [0. 2. 5.]
```

---

## Lights

A `LightData` is Maya's light shape, on a node too.

```python
key_node = TransformData("key", translate=(2.0, 3.0, 2.0))
sun_node = TransformData("sun", rotate=(-45.0, 30.0, 0.0))
scene.add(key_node, sun_node)
key = LightData(name="key", kind="point", intensity=17.0, decay_rate=2, parent_uuid=key_node.uuid)
sun = LightData(name="sun", kind="directional", intensity=0.6, parent_uuid=sun_node.uuid)
scene.add(key, sun)
```

| `kind` | Where the light comes from | Falloff |
|---|---|---|
| `"point"` | The node's world position (parent it to anything) | 1/r² when `decay_rate` is not `0` (Maya's none); constant at `0` |
| `"directional"` | Down the node's `-Z` axis (a sun) | None (parallel rays) |
| `"spot"` | As a point light, with one warning: the raytracer draws no cone | as point |

Point-light intensity with falloff is in physical 1/r² units, so the
number you want scales with **distance²**: ~`17` for a light 4 units off a
unit cube, ~`1e5` for a hero prop authored in centimetres. (The auto 3-point
rig does this arithmetic for you — see above.)

---

## Building and configuring a Scene

```python
hero_shot = Scene("hero", resolution=(640, 360), samples_per_pixel=4)
hero_shot.add(cube, cube_uv, skin, lid, hero, cam, camera, key_node, key)

# Persistent render config (per-call kwargs always win)
hero_shot.configure(
    background    = (0.05, 0.05, 0.05, 1.0),  # RGBA; alpha 0 = transparent
    angle_of_view = 28.0,                     # the default camera's, when the scene has none
    fit_padding   = 1.10,                     # autofit safety margin
    return_depth  = True,                     # pre-populate Frame.depth
)
```

`configure()` is **chainable** (returns the scene) and takes the render
settings — `resolution, samples_per_pixel, background, autofit,
default_light, return_depth, angle_of_view, fit_padding` — the shading
knobs above, `default_camera` and `aspect_ratio`. Anything else raises
`AttributeError`. Every setting is a plain attribute too.

```python
hero_shot.default_camera = "cam"
```

`Scene.strip_namespace()` strips `default_camera` along with the nodes
and every shape, so it still names its camera.

### Loading

An OBJ you wrote yourself round-trips: every mesh with its UV sets, at
identity.

```python
import os
import tempfile

tmp      = tempfile.mkdtemp()
obj_path = os.path.join(tmp, "cube.obj")
with open(obj_path, "w") as fp:
    fp.write("\n".join(f"v {x} {y} {z}" for x, y, z in points) + "\n")
    fp.write("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n")
    for face in indices.reshape(-1, 4):
        fp.write("f " + " ".join(f"{v + 1}/{i + 1}" for i, v in enumerate(face)) + "\n")

loaded = Scene.load_obj(obj_path)
print(loaded.meshes.name, len(loaded.uvs), loaded.uvs[0].mesh_uuid == loaded.meshes[0].uuid)
```

<!-- notest -->
```python
Scene.load_fbx("hero.fbx")  # everything load_fbx reads; needs the FBX SDK
Scene.load_glb("hero.glb")  # meshes, UV sets, skins + bind pose, a material per colour map
Scene.load_fbx("hero.fbx").save_fbx("copy.fbx")  # the fields, not the render settings
```

### Inspecting a scene

```python
hero_shot.camera()         # (CameraData, its node's uuid): named, default, or first
hero_shot.pose()           # the nodes a render places by: the hierarchy, or the active take's frame
hero_shot.renderables()    # [Renderable]: each mesh with its node, UV sets, skin, joints, materials, targets
hero_shot.placed_meshes()  # [(mesh, world)] of every shown mesh; a skinned one deformed, in world space
hero_shot.union_aabb()     # world-space AABB across the shown meshes
hero_shot.union_points()   # concatenated world points
hero_shot.name                   # "hero"
```

### Saving and loading

```python
path  = hero_shot.save(os.path.join(tmp, "hero.npz"))  # or .pkl / .json
again = Scene.load(path)
print([type(x).__name__ for x in (again.hierarchy, again.meshes, again.materials, again.cameras)])
```

Every list loads back as its class, every node with its uuid and
parenting, and the scene keeps its name and every setting. The cached
render is not saved.

---

## Rendering

Three ways to call the raytracer (all return a `Frame`):

```python
# 1. Single mesh shorthand (legacy)
frame = render(cube, cube_uv, None, resolution=(320, 240))

# 2. List of dicts (legacy multi-mesh)
frame = render(scene=[{"mesh": cube, "uv": cube_uv, "texture": None}], resolution=(320, 240))

# 3. Scene instance (preferred)
frame = render(scene=hero_shot, resolution=(320, 240))
```

Or — much more common — call right off the Scene:

```python
frame = hero_shot.render()    # full-quality, honours configure()
image = hero_shot.to_image()  # PIL Image
```

<!-- notest -->
```python
hero_shot.imshow()            # opens an OpenCV window and blocks until a keypress
```

`render(frame=...)` picks the frame of the active take the shapes are
posed at, when the scene holds a clip.

---

## The frame cache (it's smart, mostly)

`scene.frame` caches the last render. Repeated `imshow()` / `to_image()`
calls don't re-render. The cache is cleared automatically when:

- you change any setting on the scene
- you change a shape's fields: a material's colour, map or faces, a
  camera's lens, a light's kind, colour, intensity or decay, a morph's
  weight, a mesh's points
- you change any SRT or the visibility on a node a shape sits on, **or any
  ancestor**; with a skin, on any joint
- you add a shape or a node, or replace a list, or the frame rendered

```python
hero_shot.render()
print(hero_shot.frame is not None)          # True — cached
hero.translate = (0.2, 0.0, 0.0)
print(hero_shot.frame)                      # None — invalidated
```

Scene invalidation compares *values*, so re-setting a setting to the same
value keeps the cache. The one thing it can't catch: in-place mutation of
`mesh.points`, UV arrays, or texture pixel data. After those, call
`.render()` explicitly to refresh.

---

## The `Frame` you get back

```python
frame = hero_shot.render()

frame.array          # (H, W, 4) uint8 — RGBA, ALWAYS 4-channel now
frame.depth          # (H, W) float64; NaN at miss; populated when return_depth=True
frame.has_depth      # bool
frame.camera_matrix  # the (4, 4) used to render this frame (column-major OpenGL)
frame.angle_of_view  # the vertical FOV used

frame.shape, frame.dtype, frame.height, frame.width, frame.channels
frame.image          # lazy PIL.Image view

frame.save("hero.png")              # PIL save — extension picks format
frame[100:200, ...]                 # numpy-style indexing proxies through to .array
np.asarray(frame)                   # numpy interop
arr, depth = frame                  # legacy tuple unpack still works
```

Frames are **frozen** — methods like `wireframe()` and the encoders
return new Frames or write to disk; nothing mutates in place.

### Coverage alpha (this is the headline render feature)

Every render is RGBA. The alpha channel is **true coverage** computed
from MSAA — `hits / samples_per_pixel`. Edge pixels get fractional
alpha, miss pixels get 0, fully-covered pixels get 255.

```python
cutout = hero_shot.render(background=(0, 0, 0, 0), samples_per_pixel=4)
alpha  = cutout.array[..., 3]
print(alpha.min(), alpha.max())  # 0 255: misses clear, the cube solid
print(np.unique(alpha))          # [  0  63 127 191 255]: 0 to 4 of 4 samples hit
cutout.save("cutout.png")
```

This means:
- Save as `.png` → you get a clean cutout against transparency
- Composite into Premiere / DaVinci / Nuke without halos
- The renderer never fights you about whether the background "is
  there" — set `scene.background = (r, g, b, 0)` for a transparent
  render, or `(r, g, b, 1)` for an opaque one. RGB-only tuples are
  auto-promoted to opaque.

---

## Outputs — where the pixels go

### Stills

```python
frame.save("frame.png")                 # via PIL — png/tiff/webp/… all take RGBA
frame.image.convert("RGB").save("frame.jpg", quality=95)
```

JPEG has no alpha channel, and every Frame is RGBA — `frame.save("x.jpg")`
raises `OSError: cannot write mode RGBA as JPEG`. Drop the alpha through
`frame.image.convert("RGB")` first, as above. Extra kwargs to
`frame.save()` are forwarded to `PIL.Image.save`.

### Wireframe overlays (screen-space, on any Frame)

```python
wired = frame.wireframe(hero_shot,
                        color        = (255, 255, 255, 220),  # default: dark @ ~80% opacity
                        width        = 2,
                        unique_edges = True,
                        cull_backfaces=True)         # False = X-ray
wired.save("wires.png")
```

`Frame.wireframe()` is separate from `scene.wireframe = True`:
- **`scene.wireframe = True`** bakes every mesh's edges into its colour
  map (or over its colour) in UV space. Cheap, cached, rides along through
  turntables.
- **`Frame.wireframe(source)`** projects edges through the saved camera
  matrix and rasterizes them on top via PIL. `source` is a `Scene` (its
  shown meshes, placed as it renders them) or a single `MeshData`. Use
  this for hero presentation passes where you want the sharper line.

### Video — `.mp4` / `.mov` / `.m4v`

```python
spin = Scene("spin", resolution=(160, 120), samples_per_pixel=1)
turn = TransformData("turn")
spin.add(turn)
spun             = cube.copy()
spun.parent_uuid = turn.uuid
spin.add(spun)
frames = []
for angle in range(0, 360, 30):
    turn.rotate = (0, angle, 0)
    frames.append(spin.render())

Frame.encode_mp4(
    frames,                           # iterable of Frames or arrays
    "spin.mp4",
    fps        = 30,
    background = (0.0, 0.0, 0.0),  # RGB to flatten alpha against
    crf        = 18,               # x264 quality; 18 = visually lossless
    preset     = "slow",           # x264 speed/quality; "veryslow" = ~5% better
)
```

H.264 doesn't carry alpha, so the encoder Porter-Duff composites your
RGBA over the supplied `background`. `Scene.turntable` forwards
`scene.background[:3]` for you automatically.

### Alpha-preserving video — `.avi`

```python
Frame.encode_avi(frames, "spin.avi", fps=30)
```

PNG codec inside an AVI container — the cleanest way to keep per-pixel
alpha that ffmpeg ships with. Larger file than mp4, but full alpha
survives into your NLE timeline.

### High-quality GIF — `.gif`

```python
Frame.encode_gif(frames, "loop.gif", fps=30, use_gifski=None)
```

Prefers [gifski](https://gif.ski/) when available (NeuQuant + temporal
coherence — the best palette quality). Falls back to ffmpeg's two-pass
`palettegen` + `paletteuse` (sierra2_4a dither) which is in the 85-90%
quality range for smooth content.

### Auto-dispatch by extension

```python
Frame.encode(frames, "spin2.mp4")  # routes to encode_mp4
Frame.encode(frames, "spin2.avi")  # routes to encode_avi
Frame.encode(frames, "loop2.gif")  # routes to encode_gif
```

All four encoders are classmethods, take any iterable of Frames (or raw
arrays), and return the output path.

### Binary discovery (macOS DCC gotcha)

`ffmpeg` and `gifski` are resolved the way any subprocess resolves a bare
command name: through the calling process's `PATH`, and nothing else. There
is no escape-hatch environment variable; if the binary is missing you get a
`RuntimeError` with an install hint instead of a raw `FileNotFoundError`.

GUI apps launched from the macOS Finder inherit only the bare login `PATH`
(`/usr/bin:/bin:/usr/sbin:/sbin`). If `ffmpeg` works in Terminal but not
inside the DCC, fix the DCC's launch environment — `Maya.env` with
`PATH = /opt/homebrew/bin:$PATH`, for example — rather than the code.

---

## Turntables — the polished one

```python
scene.turntable("spin.mp4")    # 120 frames, y axis, 0->360°, fps=30
```

Defaults: 120 frames, full revolution around `+Y`, mp4. Override any of:

```python
scene.turntable(
    "hero.mp4",
    n_frames      = 24,
    fps           = 24,
    rotation_axis = "y",              # "x", "y" (default), or "z"
    start_angle   = 0.0,
    end_angle     = 360.0,
    fit           = "auto",           # or "static" to use the scene's camera as-is
    camera_name   = "cam",            # which camera to base orientation on
    background    = (0.1, 0.1, 0.1),  # mp4 alpha-flatten background
    n_fit_samples = 12,               # how many angles to consider for the framing pass
    verbose       = False,
)
```

What makes this one nice (vs writing a Python loop yourself):

- **One framing pass**, not 120. The renderer computes the union AABB
  over all sample angles and finds a single camera distance that fits
  the worst-case silhouette. No wobble, no clipping, no per-frame
  autofit thrash.
- **Camera orbits the scene centroid.** Nothing in your scene is
  mutated — turntable computes a per-frame camera matrix and passes it
  in via `render_kwargs`.
- **Extension drives the encoder.** `.mp4`/`.mov`/`.m4v` → H.264.
  `.avi` → alpha-preserving. `.gif` → gifski/ffmpeg. Anything else →
  PNG image sequence (with `frame_{0000:04d}` formatting, auto-injected
  if you didn't put a `{frame}` placeholder in the path).

```python
scene.turntable("seq/frame_{frame:04d}.png", n_frames=4)  # → list[str] of saved paths
scene.turntable("seq/anim.png", n_frames=4)               # auto-injects → seq/anim.{frame:04d}.png
```

---

## Animation — the complement of the turntable

`turntable()` moves the camera and holds the scene still.
`Scene.animate()` moves the scene: the active take of a clip — the one
passed, or the scene's own `clip` — poses the nodes at each frame. A
skinned mesh deforms to the pose, from the bind pose `scene.hierarchy`
holds; every other mesh, camera and light rides its node. Blend shape
targets move the points by their weights before the skin, Maya's way (an
in-between is a knot the weight blends through). The scene is handed back
as it was when the call finishes, its own clip included.

<!-- notest -->
```python
from cgmath.hierarchy import ClipData

walk = Scene.load_fbx("hero.fbx")             # the bind pose, the skins and the file's takes
walk.animate("walk.mp4")                      # the active take; fit="auto" keeps the character in frame

clip                        = ClipData(walk.hierarchy, frames=48, fps=24.0)
clip.frames["spine"].rotate = np.column_stack([np.zeros(48), np.zeros(48), np.linspace(0, 30, 48)])
walk.animate("bend.{frame:04d}.png", clip=clip)   # a clip of your own over the scene's nodes
```

A clip of another rig raises: the skins name joints the clip does not
hold. `render(frame=...)` renders one frame of the scene's own clip.

---

## Sequences render once

`turntable()` and `animate()` keep their frames, raw, in a scratch
folder: one per Scene, under one per Python process. The same frames asked
for again are read back, not rendered, so the `.gif` after the `.mp4`
costs only the encode:

```python
scene.turntable("spin.mp4", n_frames=24, fps=24)  # renders 24 frames
scene.turntable("spin.gif", n_frames=24, fps=12)  # renders none
scene.turntable("spin.mp4", n_frames=12)          # renders 12: another turntable
```

| Reads back | Renders again |
|---|---|
| a new output path or extension, `fps`, `background`, `threads` | any other argument |
| | any render setting, and anything that clears the frame cache |
| the same clip values | a different clip |

- **One sequence per folder.** A different one replaces it.
- **Nothing to clean up.** The folder goes with its Scene, the scratch
  folder at exit.
- **Blind to in-place edits**, like the frame cache: mesh points, UVs,
  texture pixels. `clear_render_cache()` drops the kept frames.

```python
scene.clear_render_cache()  # the next turntable renders
```

---

## Recipes — what to type for what shot

### Quick lookdev preview

<!-- notest -->
```python
Scene.load_fbx("hero.fbx").imshow()
```

### Hero still

```python
shot = Scene("hero", sample_method="bezier").add(cube, cube_uv)
shot.configure(resolution=(1280, 720), samples_per_pixel=4)
shot.render().save("hero.png")
```

`sample_method` is a scene setting: every mesh of the scene samples the
same way.

### Transparent PNG cutout

```python
shot.background = (0, 0, 0, 0)             # alpha=0 → miss pixels transparent
shot.render().save("cutout.png")
```

### Depth pass

```python
shot.configure(return_depth=True)
frame = shot.render()
np.save("depth.npy", frame.depth)          # NaN at miss; otherwise Euclidean t
```

### Turntable for slack

```python
shot.background = (0.08, 0.08, 0.08)
shot.configure(resolution=(320, 240))
shot.turntable("spin.mp4", n_frames=24, fps=24)
```

### Hero turntable with baked wireframe

`scene.wireframe = True` bakes every mesh's edges into its colour map (UV
space) so they ride along through `turntable()` for free:

```python
shot.configure(wireframe=True, wireframe_color=(255, 255, 255), wireframe_thickness=2)
shot.turntable("wired.mp4", n_frames=24, fps=24)
```

For a sharper screen-space line on a single hero still, post-process the
returned Frame with `Frame.wireframe()` instead — it projects edges
through the saved camera matrix.

### Custom 3-light rig overriding the auto-rig

```python
custom = Scene("custom", resolution=(320, 240), samples_per_pixel=4).add(cube, cube_uv)
for name, where, intensity, color in (
    ("key",  (2.0, 3.0, 2.0),  17.0, (1.0, 1.0, 1.0)),
    ("fill", (-2.0, 1.5, 2.0),  4.0, (0.85, 0.9, 1.0)),
    ("rim",  (0.0, 1.5, -3.0),  3.4, (1.0, 1.0, 1.0)),
):
    node = TransformData(name, translate=where)
    custom.add(node)
    custom.add(LightData(name=name, kind="point", intensity=intensity, color=color, decay_rate=2, parent_uuid=node.uuid))
custom.render().save("custom_rig.png")
```

### Headlight follows the camera (parenting trick)

```python
hl       = Scene("hl", resolution=(320, 240)).add(cube, cube_uv)
cam_node = TransformData("cam", translate=(0.0, 1.0, 5.0))
lamp     = TransformData("headlight")
hl.add(cam_node, lamp)
lamp.set_parent(cam_node, world_space=False)   # by node, name or uuid
hl.add(CameraData(name="cam", parent_uuid=cam_node.uuid))
hl.add(LightData(name="headlight", kind="point", intensity=26.0, decay_rate=2, parent_uuid=lamp.uuid))
hl.render().save("headlight.png")
```

### Orthographic shot

```python
ortho = Scene("ortho", resolution=(320, 240)).add(cube, cube_uv)
eye   = TransformData("eye", translate=(2.0, 2.0, 4.0))
ortho.add(eye)
aim(eye, (0.0, 0.0, 0.0))
ortho.add(CameraData(name="ortho", orthographic=True, orthographic_width=3.5, parent_uuid=eye.uuid))
ortho.render().save("ortho.png")
```

The orthographic width is what the frame spans horizontally, in the
scene's units; the height follows the frame's aspect.

---

## Conventions, all in one place

- **Camera** is OpenGL: `-Z` forward, `+Y` up, `+X` right.
- **Image** pixel `(0, 0)` is top-left, `+x` right, `+y` down.
- **UV** has `u` increasing right, `v` increasing up — `v=0` is the
  **bottom** of the texture (Maya/OpenGL convention).
- **Node transforms** are row-major, following Maya: translation lives
  at `M[3, :3]`, Maya rotate orders and parenting.
- **Camera-to-world matrices** handed to / returned by the renderer
  (`look_at()`, `Frame.camera_matrix`, `render(camera_matrix=...)`) are
  column-major OpenGL: eye at `M[:3, 3]`.
- **Units and axes** are Maya's: Y-up, right-handed, centimetres — what
  every cgmath value is in. An fbx converts at the boundary.
- **Colors** are `[0, 1]` floats for material colours, light colours and
  backgrounds; `0-255` ints for `wireframe_color` and
  `Frame.wireframe(color=...)`. The ones a scene or a shape keeps take any
  sequence and are stored as numpy arrays: float64, int32 for
  `wireframe_color`. Textures are uint8, uint16, or float32 in `[0, 1]`.
- **Wrap** modes: `"repeat"` (tile) or `"clamp"`, a material's `wrap_u`.
- **Sample method**: `"bilinear"` for flat patches, `"bezier"` for PN
  Quad bicubic patches (smoother surfaces, ~3-4x slower). The
  `SampleMethod` enum from `cgmath.geometry.mesh` works too.

---

## What it deliberately doesn't do (yet)

- **PBR shading.** Currently Lambertian only: a material's colour times
  its diffuse, its colour map. A phong's specular colour, cosine power and
  reflectivity are carried, not shaded. The GLB importer extracts
  metallic / roughness / normal / emissive maps but the shader doesn't
  use them.
- **Shadows.** `cast_shadows` is reserved for a future shadow pass; today
  it does nothing.
- **Vertex alpha.** A mesh's colours are RGBA; the alpha channel is carried,
  not rendered (no transparency pass).
- **Specular / reflections / refractions.** Diffuse + ambient only.
- **Texture mip-mapping.** Bilinear filtering at native resolution.
- **Near / far clipping.** A `CameraData`'s clip planes are carried, not
  applied.
- **Spot light cones.** A spot light renders as a point light.
- **In-place mutation tracking.** Edit `mesh.points` directly and the
  cached frame won't notice — call `.render()` to refresh.

For everything else (look-dev stills, turntables, alpha cutouts, depth
passes, wireframe overlays, skinned and blend-shaped animation), this is
a one-import, no-DCC workflow. Every API in the module, with a runnable
example apiece, is in [`CHEATSHEET.md`](CHEATSHEET.md).
