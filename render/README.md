# `cgmath.render` — the software raytracer

A numpy software raytracer with a small scene graph.

You build a **Scene** out of **Objects**, **Cameras**, and **Lights**, hit
"render," and get a `Frame` (image + optional depth) back. The nodes are
the same `TransformData` as the rest of the library, so SRT and parenting
work as they do everywhere else. No DCC required.

- [`CHEATSHEET.md`](CHEATSHEET.md) — every public class, function, property
  and method in this module, each with a small copy-paste example.
- [`../README.md`](../README.md) — the parent `cgmath` package map.

Every Python block below runs, top to bottom, as one script — they share
a namespace, notebook style. Start here:

```python
import numpy as np
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.render import Object, Scene

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
cube    = MeshData(points=points, indices=indices, counts=counts)

# One UV square per face (all six faces share the same four corners).
cube_uv = UVData(
    points  = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    indices = np.tile([0, 1, 2, 3], 6),
    counts  = counts,
)

scene = Scene("hero")
scene.append(Object(name="cube", mesh=cube, uv=cube_uv))
scene.render().save("cube.png")
```

That is the whole minimum-viable render. No Camera, no Light, no rig —
the renderer brings its own. With a real asset on disk it is one line:

<!-- notest -->
```python
Object.load_glb("hero.glb").imshow()
```

`Scene.append()` returns `None`, so it does **not** chain — append, then
render.

---

## What you actually get for free

When you don't supply a Camera or Light, the renderer fills them in for you:

- **Default camera**: 3/4 elevated front-right view (eye direction
  `(+x, +y·0.7, +z)`), framed so the scene fills the frustum (autofit).
- **Default lighting**: a real **3-point rig** built around the scene's
  bounding box.
  - **Key** — front-upper-right, full intensity, 1/r² falloff
  - **Fill** — front-upper-left, 40% of key
  - **Rim** — back-upper, 30% of key
  All three sit at `1.5 × the AABB diagonal` from the centroid with
  intensity `distance²`, so it looks right at any scale.
- **Default resolution**: `(500, 500)` for a `Scene` or an `Object`
  preview. (The bare `render()` function defaults to `(512, 384)`.)
  Bump it via `scene.configure(resolution=(W, H))`.
- **Default samples-per-pixel**: `4` (2×2 MSAA) for an `Object` preview.
  A `Scene` leaves it unset, which lands on the renderer's `1` — set
  `scene.configure(samples_per_pixel=4)` for clean silhouettes.
- **Default sample method**: `"bilinear"` (flat patches). Switch to
  `"bezier"` for smoothed PN-Quad bicubic surfaces.
- **Default background**: `(0, 0, 0, 0)` — **transparent** black, so a
  straight `.png` save is already a cutout. Pass any RGB (auto-promoted
  to opaque RGBA) or a full RGBA tuple to change it.

You can override any of these — but the defaults exist so you can iterate
quickly without rigging anything.

---

## The mental model — a scene graph

Three node types, all inheriting `TransformData`
(`translate / rotate / scale / parent_node`, row-major matrices, Maya
rotate orders). Parent a Light to a Camera and you get a real headlight.
Parent an Object to another Object for free instancing-with-offsets.

| Node | What it is | Lives in |
|---|---|---|
| `Object` | A renderable mesh + UV + texture (+ optional skin) | The scene graph |
| `Camera` | OpenGL-convention perspective or orthographic | The scene graph |
| `Light` | `point` (with optional 1/r² falloff) or `infinite` (parallel rays, no falloff) | The scene graph |
| `Scene` | The root container; list-like; tracks render config | The scene graph |
| `Frame` | The render result (pixels + depth + camera matrix used) | What `render()` hands back |

```python
from cgmath.render import Camera, Frame, Light, load_texture, look_at, Object, render, Scene
```

Those eight names are everything that's exported from `cgmath.render`.

---

## Building an Object

```python
checker = np.indices((32, 32)).sum(0) % 2                 # 0/1 squares
checker = np.repeat(checker[..., None], 3, axis=2).astype(np.float32)

obj = Object(
    name      = "hero",
    mesh      = cube,         # cgmath.geometry.mesh.MeshData
    uv        = cube_uv,      # cgmath.geometry.mesh.UVData (optional)
    texture   = checker,      # path, (H, W, 3) float array, or None for flat shading

    base_color           = (0.7, 0.7, 0.7),  # used when texture is None
    sample_method        = "bilinear",       # or "bezier" for smooth subdivision
    wrap                 = "repeat",         # or "clamp" for UVs outside [0, 1]
    ambient              = 0.8,              # 0..1; 0 = pure shadow, 1 = unshaded
    twosided             = True,             # render back faces too
    cast_shadows         = True,             # reserved; no shadow pass yet
    skin                 = None,             # SkinDeformData, for .pose()
    samples_per_pixel    = 4,                # MSAA, 1 or any perfect square
    resolution           = (500, 500),       # for the per-Object preview

    # baked-in (UV-space) wireframe — see also Frame.wireframe() for screen-space
    wireframe           = False,
    wireframe_color     = (0, 0, 0),        # 0-255 ints
    wireframe_thickness = 1,

    # SRT (inherited from TransformData)
    translate  = (0, 0, 0),
    rotate     = (0, 0, 0),
    scale      = (1, 1, 1),
    visibility = True,
)
```

### Loading geometry

Three loaders for the common cases, plus two texture side-loaders:

<!-- notest -->
```python
Object.load_obj("model.obj", index=0)             # mesh + UV from OBJ
Object.load_glb("model.glb", scale_factor=100.0)  # mesh + UV + texture + skin
Object.load_fbx("model.fbx", index=0)             # needs the FBX SDK
obj.extract_texture_from_glb("model.glb", material_index=0)
obj.extract_texture_from_fbx("model.fbx", material_index=0)
```

`load_glb` pulls the PBR **base color (albedo) map** into `Object.texture`
and, with `load_skin=True` (the default), attaches the mesh's
`SkinDeformData` to `Object.skin`. The GLB parser also reads
metallic-roughness, normal, occlusion and emissive maps; the shader
discards them today, but they're there for a future multi-texture pass.

An OBJ you wrote yourself round-trips the same way:

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

loaded = Object.load_obj(obj_path)
print(loaded.mesh.points.shape, loaded.uv.points.shape)
```

### Topology ops that keep your UVs in sync

```python
scratch = Object(name="scratch", mesh=cube.copy(), uv=cube_uv.copy())
scratch.triangulate(ngons_only=False)
scratch.quadrangulate(source="uv")  # promote shared-UV triangle pairs back to quads
scratch.merge()                     # de-duplicate coincident vertices
```

These mutate the Object's `mesh` / `uv` in place and clear its cached
frame for you. `MeshData` / `UVData` are handed around by reference, so
`.copy()` first if other Objects share them.

---

## Cameras

```python
from cgmath.render import Camera

cam = Camera(
    name            = "camera",
    angle_of_view   = 35.0,   # vertical FOV in degrees
    is_orthographic = False,  # set True + ortho_height for ortho
    ortho_height    = None,   # world units of vertical extent
    near_plane      = 0.0,    # reserved; no clipping in v1
    far_plane       = float("inf"),
    aspect_ratio    = None,   # taken from resolution if None
    translate       = (0, 0, 5),
    rotate          = (0, 0, 0),
)

cam.look_at((0, 1, 0))                       # aim from current position at a world point
cam.autofit(scene.union_points(), 500, 500)  # reposition to fit a point cloud in frustum
framed = Camera.default_view(scene.union_points())  # fresh 3/4 elevated framed camera
```

Convention: looks down its own `-Z`, `+Y` is up, `+X` is right (OpenGL).

**Two matrix layouts, don't mix them up.** `cam.world_matrix` is
row-major like every node — the eye sits at `cam.world_matrix[3, :3]`. The free function
`look_at()` and `Frame.camera_matrix` are column-major OpenGL — the eye
sits at `[:3, 3]`. Transpose when you cross between them.

```python
from cgmath.render import look_at

m = look_at((0.0, 2.0, 5.0), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))  # (4, 4) camera-to-world
print(m[:3, 3])                     # OpenGL column-major: eye is here

cam.world_matrix = m.T              # row-major for the node
print(cam.world_matrix[3, :3])      # -> [0. 2. 5.]
```

---

## Lights

```python
from cgmath.render import Light

key  = Light(name="key",  kind="point",    intensity=17.0, color=(1.0, 1.0, 1.0))
sun  = Light(name="sun",  kind="infinite", intensity=2.0,  rotate=(-45, 30, 0))
fill = Light(name="fill", kind="point",    intensity=0.4,  falloff=False)  # constant
```

| Kind | Where the light comes from | Falloff |
|---|---|---|
| `"point"` | The light's world position (parent it to anything) | 1/r² by default; set `falloff=False` for constant |
| `"infinite"` | The light's `-Z` axis (a sun) | None (parallel rays) |

Point-light intensity is in physical 1/r² units, so the number you want
scales with **distance²**: ~`17` for a light 4 units off a unit cube,
~`1e5` for a hero prop authored in centimetres. (The auto 3-point rig
does this arithmetic for you — see above.)

`light.position` and `light.direction` are read-only views of the
world-space position / forward axis. `light.as_dict()` gives the dict
shape the low-level `render()` accepts as `point_light=`.

```python
print(key.position, sun.direction)
print(fill.as_dict())
```

---

## Building and configuring a Scene

```python
scene = Scene("hero")
scene.append(Object(name="cube", mesh=cube, uv=cube_uv))
scene.append(Camera(name="cam", angle_of_view=28.0))
scene.append(Light(name="key", kind="point", intensity=17.0, translate=(2, 3, 2)))

# Persistent render config (per-call kwargs always win)
scene.configure(
    resolution        = (640, 360),
    samples_per_pixel = 4,
    background        = (0.05, 0.05, 0.05, 1.0),  # RGBA; alpha 0 = transparent
    angle_of_view     = 28.0,
    fit_padding       = 1.10,                     # autofit safety margin
    return_depth      = True,                     # pre-populate Frame.depth
)
```

`configure()` is **chainable** (returns the scene), and the only kwargs
it accepts are the eight render-config keys: `resolution,
samples_per_pixel, background, autofit, default_light, return_depth,
angle_of_view, fit_padding`. Anything else raises `AttributeError` —
pass it per-call to `render()` instead (`scene.render(sample_method=...)`),
or, for `aspect_ratio` / `default_camera_name`, set the attribute or the
`Scene(...)` constructor argument directly.

```python
scene.aspect_ratio        = None
scene.default_camera_name = "cam"
```

### Bulk loading

```python
Scene.load_obj(obj_path)               # one Object per `g` group
```

<!-- notest -->
```python
Scene.load_glb("complex.glb")          # one Object per geometry, textures + skins extracted
```

### Inspecting a scene

```python
scene.objects                 # list of just the Objects
scene.cameras                 # list of just the Cameras
scene.lights                  # list of just the Lights
scene.get_camera("cam")  # by name; falls back to default_camera_name then first
scene.union_aabb()       # world-space AABB across all visible Objects
scene.union_points()     # concatenated visible world points
scene.scene_name              # "hero" — `scene.name` is the list of CHILD names
```

---

## Rendering

Three ways to call the raytracer (all return a `Frame`):

```python
from cgmath.render import render

# 1. Single mesh shorthand (legacy)
frame = render(cube, cube_uv, None, resolution=(320, 240))

# 2. List of dicts (legacy multi-mesh)
frame = render(scene=[{"mesh": cube, "uv": cube_uv, "texture": None}], resolution=(320, 240))

# 3. Scene instance (preferred)
frame = render(scene=scene, resolution=(320, 240))
```

Or — much more common — call right off the Scene:

```python
frame = scene.render()    # full-quality, honours scene.configure()
image = scene.to_image()  # PIL Image
```

<!-- notest -->
```python
scene.imshow()                   # opens an OpenCV window and blocks until a keypress
```

Object has the same trio for one-shot previews without ever building a
Scene:

```python
preview = Object(name="solo", mesh=cube, uv=cube_uv)
preview.render()    # returns a Frame (3-point lighting, 500×500)
preview.to_image()  # PIL Image
```

<!-- notest -->
```python
preview.imshow()                 # OpenCV window
```

These work because Object wraps itself in a one-Object preview Scene
under the hood and runs the same pipeline. **Gotcha**: that preview Scene
`append`s the Object, which *moves* it — an Object that was already in a
Scene is removed from it. Call `render()` / `imshow()` / `to_image()` on
standalone Objects, and `scene.render()` on ones you've parented.

---

## The frame cache (it's smart, mostly)

Both `obj.frame` and `scene.frame` cache the last render. Repeated
`imshow()` / `to_image()` calls don't re-render. The cache is cleared
automatically when:

- you reassign anything visual on an Object (`mesh, uv, texture,
  base_color, wireframe, wireframe_color, wireframe_thickness,
  sample_method, wrap, ambient, twosided, cast_shadows, resolution,
  samples_per_pixel, background`)
- you change any SRT on the node **or any ancestor**
- on a Scene: you change Camera FOV / ortho settings, Light kind /
  intensity / color / falloff, Object visibility, scene topology
  (add/remove nodes), or any of the render-config values

```python
scene.render()
print(scene.frame is not None)          # True — cached
scene.objects[0].translate = (0.2, 0, 0)
print(scene.frame)                      # None — invalidated
```

Scene invalidation compares *values*, so re-setting a config key to the
same value keeps the cache. The one thing it can't catch: in-place
mutation of `mesh.points`, UV arrays, or texture pixel data. After those,
call `.render()` explicitly to refresh.

---

## The `Frame` you get back

```python
frame = scene.render()

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
cutout = scene.render(background=(0, 0, 0, 0), samples_per_pixel=4)
alpha  = cutout.array[..., 3]
print(alpha.min(), alpha.max())     # 0 (miss) .. 255 (solid)
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
wired = frame.wireframe(scene,
                        color        = (255, 255, 255, 220),  # default: dark @ ~80% opacity
                        width        = 2,
                        unique_edges = True,
                        cull_backfaces=True)         # False = X-ray
wired.save("wires.png")
```

`Frame.wireframe()` is separate from `Object.wireframe=True`:
- **`Object.wireframe=True`** bakes mesh edges into the diffuse texture
  in UV space. Cheap, cached.
- **`Frame.wireframe(source)`** projects edges through the saved camera
  matrix and rasterizes them on top via PIL. `source` is a `Scene` or a
  single `Object`. Use this for hero presentation passes where you want
  the sharper line.

### Video — `.mp4` / `.mov` / `.m4v`

```python
from cgmath.render import Frame

spin = Scene("spin", resolution=(160, 120), samples_per_pixel=1)
spin.append(Object(name="cube", mesh=cube, uv=cube_uv))
frames = []
for angle in range(0, 360, 30):
    spin.objects[0].rotate = (0, angle, 0)
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
    fit           = "auto",           # or "static" to use the scene's existing camera as-is
    camera_name   = "cam",            # which Camera to base orientation on
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

`Object.turntable()` is the same one-call convenience as
`Object.render()` — same preview Scene, same caveat about the Object
being moved into it.

---

## Animation — the complement of the turntable

`turntable()` moves the camera and holds the scene still.
`Scene.animate()` moves the scene: every visible Object carrying a
`skin` is posed to a `ClipData` frame by frame, and Objects without one
render as static props. The scene is put back in its bind pose when the
call finishes.

<!-- notest -->
```python
from cgmath.hierarchy import ClipData, HierarchyData

rig  = HierarchyData.load_glb("hero.glb")
clip = ClipData(rig, frames=48, fps=24.0)

walk = Scene("walk")
walk.append(Object.load_glb("hero.glb"))  # load_skin=True by default
walk.animate(clip, "walk.mp4")            # fit="auto" keeps the character in frame

# Or pose a single Object yourself and render the result:
Object.load_glb("hero.glb").pose(rig).render("pose.png")
```

`Object.pose()` always deforms the *bind* mesh, never the current one,
so calling it in a loop can't compound. `Object.restore_bind_pose()`
puts it back.

---

## Recipes — what to type for what shot

### Quick lookdev preview

<!-- notest -->
```python
Object.load_glb("hero.glb").imshow()
```

### Hero still

```python
shot = Scene("hero")
shot.append(Object(name="cube", mesh=cube, uv=cube_uv, sample_method="bezier"))
shot.configure(resolution=(1280, 720), samples_per_pixel=4)
shot.render().save("hero.png")
```

`sample_method` is a per-Object attribute (or a per-call `render()`
kwarg) — it is **not** one of the `configure()` keys.

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

`Object.wireframe=True` bakes mesh edges into the diffuse texture (UV
space) so they ride along through `turntable()` for free:

```python
for obj in shot.objects:
    obj.wireframe           = True
    obj.wireframe_color     = (255, 255, 255)
    obj.wireframe_thickness = 2
shot.turntable("wired.mp4", n_frames=24, fps=24)
```

For a sharper screen-space line on a single hero still, post-process the
returned Frame with `Frame.wireframe()` instead — it projects edges
through the saved camera matrix.

### Custom 3-light rig overriding the auto-rig

```python
custom = Scene("custom", resolution=(320, 240), samples_per_pixel=4)
custom.append(Object(name="cube", mesh=cube, uv=cube_uv))
custom.append(Light(name="key",  kind="point", intensity=17.0,
                    translate=(2, 3, 2)))
custom.append(Light(name="fill", kind="point", intensity=4.0, falloff=True,
                    translate=(-2, 1.5, 2), color=(0.85, 0.9, 1.0)))
custom.append(Light(name="rim",  kind="point", intensity=3.4,
                    translate=(0, 1.5, -3)))
custom.render().save("custom_rig.png")
```

### Headlight follows the camera (parenting trick)

```python
cam       = Camera(name="cam", translate=(0, 1, 5))
headlight = Light(name="headlight", kind="point", intensity=26.0, parent_node=cam)

hl = Scene("hl", resolution=(320, 240))
hl.append(Object(name="cube", mesh=cube, uv=cube_uv))
hl.append(cam)
hl.append(headlight)
hl.render().save("headlight.png")
```

### Orthographic shot

```python
ortho = Scene("ortho", resolution=(320, 240))
ortho.append(Object(name="cube", mesh=cube, uv=cube_uv))
ortho.append(Camera(name="ortho", is_orthographic=True, ortho_height=2.5,
                    translate=(2, 2, 4)))
ortho.render().save("ortho.png")
```

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
- **Color tuples** are `[0, 1]` floats for textures, base colors and
  backgrounds; `0-255` ints for `wireframe_color` and
  `Frame.wireframe(color=...)`.
- **Wrap** modes: `"repeat"` (tile) or `"clamp"`.
- **Sample method**: `"bilinear"` for flat patches, `"bezier"` for PN
  Quad bicubic patches (smoother surfaces, ~3-4x slower).

---

## What it deliberately doesn't do (yet)

- **PBR shading.** Currently Lambertian only. The GLB importer extracts
  metallic / roughness / normal / emissive maps but the shader doesn't
  use them. They're sitting there for a future shading pass.
- **Shadows.** `Object.cast_shadows` is reserved for a future shadow
  pass; today it does nothing.
- **Specular / reflections / refractions.** Diffuse + ambient only.
- **Texture mip-mapping.** Bilinear filtering at native resolution.
- **Near / far clipping.** `Camera.near_plane` / `far_plane` are
  reserved; v1 ignores them.
- **In-place mutation tracking.** Edit `mesh.points` directly and the
  cached frame won't notice — call `.render()` to refresh.

For everything else (look-dev stills, turntables, alpha cutouts, depth
passes, wireframe overlays, skinned animation), this is a one-import,
no-DCC workflow. Every API in the module, with a runnable example
apiece, is in [`CHEATSHEET.md`](CHEATSHEET.md).
