"""
Simple textured raytracer for ``MeshData`` / ``UVData`` pairs.

Three ways to call :func:`render`:

  1. Single mesh (legacy)::

       img = render(mesh, uv, texture)

  2. List-of-dicts scene (legacy multi-mesh)::

       img = render(scene=[
           {"mesh": m1, "uv": uv1, "texture": "wood.png"},
           {"mesh": m2, "uv": uv2, "texture": "metal.png"},
       ])

  3. ``Scene`` instance (preferred)::

       from cgmath.render.scene import Camera, Light, Object, Scene
       scene = Scene("my_scene")
       scene.append(Object(name="hero", mesh=m, uv=uv, texture="hero.png"))
       scene.append(Camera(name="cam"))
       scene.append(Light(name="key", kind="point", intensity=1e5))
       img = render(scene=scene, resolution=(640, 480))

Internally:
  1. Generate per-pixel rays from a 4x4 ``camera_matrix`` (camera-to-world,
     OpenGL/Maya: -Z is forward, +Y is up, +X is right).
  2. For each scene entry: cast rays (object-local space if the entry has
     a non-identity world_matrix), interpolate UVs, sample texture,
     shade with all scene lights, accumulate per-sample.
  3. Z-merge across entries per sub-pixel sample, then MSAA resolve to
     final image.

Texture wrap mode defaults to ``"repeat"``.  Background is RGBA in [0, 1]
and defaults to ``(0, 0, 0, 0)`` (transparent black) -- the rendered image
is composited over the background using premultiplied Porter-Duff "over",
and the output is ALWAYS ``(H, W, 4)`` ``uint8`` (use ``frame.array[..., :3]``
for RGB-only callers).
``samples_per_pixel`` enables MSAA (must be a perfect square).
``return_depth=True`` returns ``(image, depth)`` with NaN at misses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from cgmath.geometry.mesh import MeshData, SampleMethod, UVData
from cgmath.render.camera import (
    _autofit_camera_for_points,
    _default_camera_for_points,
    _derive_ortho_height,
    _inflate_points_about_center,
    look_at,
)
from cgmath.render.frame import Frame
from cgmath.render.scene import Scene
from cgmath.render.texture import _load_texture, _sample_texture_bilinear

# Re-exports kept for backward compatibility.  ``look_at`` and the autofit /
# default-camera helpers historically lived in this module; older callers may
# still do ``from cgmath.render.raytracer import look_at, _load_texture``.
__all__ = [
    "render",
    "Frame",
    "look_at",
    "_load_texture",
    "_sample_texture_bilinear",
    "_default_camera_for_points",
    "_autofit_camera_for_points",
    "_derive_ortho_height",
]


# -- Public API ----------------------------------------------------------------


def render(
    mesh:              Optional[MeshData]                   = None,
    uv:                Optional[UVData]                     = None,
    texture:           Optional[Union[str, np.ndarray]]     = None,
    camera_matrix:     Optional[np.ndarray]                 = None,
    angle_of_view:     float                                = 35.0,
    resolution:        Tuple[int, int]                      = (512, 384),
    point_light:       Optional[Union[dict, List[dict]]]    = None,
    background:        Tuple[float, float, float, float]    = (0.0, 0.0, 0.0, 0.0),
    sample_method:     Union[SampleMethod, str]             = SampleMethod.BILINEAR,
    ambient:           float                                = 0.1,
    wrap:              str                                  = "repeat",
    twosided:          bool                                 = True,
    autofit:           bool                                 = False,
    fit_padding:       Optional[float]                      = None,
    return_depth:      bool                                 = False,
    samples_per_pixel: int                                  = 1,
    base_color:        Tuple[float, float, float]           = (0.7, 0.7, 0.7),
    scene:             Optional[Union[List[dict], "Scene"]] = None,
    default_light:     bool                                 = True,
) -> Frame:
    """Render a textured mesh / scene to a :class:`Frame`.

    See module docstring for the three call styles.

    Args:
        mesh / uv / texture: legacy single-mesh shorthand.
        camera_matrix: ``(4, 4)`` camera-to-world OpenGL-style matrix.
            When ``None``, a default 3/4 elevated view is built and
            ``autofit`` is forced ``True``.  When a Scene is given,
            this argument (if set) overrides the scene's active Camera.
        angle_of_view: Vertical full FOV in degrees.
        resolution: ``(width, height)``.
        point_light: A single dict or list of dicts.  Each dict has
            ``"kind"`` (``"point"`` default or ``"infinite"``),
            ``"position"`` / ``"direction"``, ``"color"``, ``"intensity"``,
            ``"falloff"``.
        background: RGBA quadruple in [0, 1].  The rendered image is
            composited over this color using premultiplied Porter-Duff
            "over".  Default is ``(0, 0, 0, 0)`` -- transparent black,
            which yields a 4-channel image with coverage alpha ready
            for downstream compositing.  Pass an opaque value (e.g.
            ``(0, 0, 0, 1)``) to flatten misses to a solid color.  For
            convenience, :class:`Scene` and :class:`Object` accept RGB
            triples and auto-promote them to opaque RGBA; this function
            requires RGBA directly.
        sample_method: ``"bilinear"`` or ``"bezier"``.  Per-mesh override
            via ``scene[i]["sample_method"]``.
        ambient: Ambient term in [0, 1] added once after all lights.
        wrap: ``"repeat"`` or ``"clamp"`` for UVs outside [0, 1].
        twosided: Forwarded to :meth:`MeshData.raycast`.
        autofit: Override camera position to fit the scene.  Implicit
            when ``camera_matrix=None``.
        fit_padding: Scale factor (>= 1.0) applied to the autofit point
            cloud about its AABB center, adding a safety margin around
            the silhouette.  When ``None`` (default), auto-defaults to
            ``1.15`` if any entry uses bezier sampling (whose patches
            bulge outside the control-point hull and clip otherwise),
            else ``1.0``.  Only applies on the perspective autofit
            path; orthographic and explicit-camera renders are unaffected.
        return_depth: Also compute and attach a per-pixel depth buffer
            (NaN at miss); accessible via :attr:`Frame.depth`.
        samples_per_pixel: MSAA factor (1 / 4 / 9 / 16 / ...).
        base_color: Surface color when ``texture`` is None.
        scene: ``list[dict]`` (legacy) or :class:`Scene` instance.
        default_light: When True (default) and the scene has no Light, a
            "headlight" point light parented to the camera is added.

    Returns:
        A :class:`Frame` whose :attr:`~Frame.array` is ALWAYS
        ``(H, W, 4)`` ``uint8`` (un-premultiplied).  Use
        ``frame.array[..., :3]`` for RGB-only callers.  When
        ``return_depth=True``, the frame's :attr:`~Frame.depth` is also
        populated; otherwise it is ``None``.  ``Frame`` supports
        ``np.asarray(frame)``, ``frame[y, x]``, ``arr, depth = frame``
        (legacy unpack), and ``frame.save(path)``.
    """

    # ---- normalize scene input -> entries + camera + lights ----
    scene_entries, scene_camera, scene_lights = _build_scene_entries_dispatch(
        scene=scene,
        single_mesh=mesh,
        single_uv=uv,
        single_texture=texture,
        single_base_color=base_color,
        single_sample_method=sample_method,
        single_wrap=wrap,
        single_ambient=ambient,
        single_twosided=twosided,
    )
    if not scene_entries:
        raise ValueError("nothing to render: provide either `mesh` or `scene`")

    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    spp    = int(samples_per_pixel)
    n_axis = int(round(np.sqrt(spp)))
    if spp < 1 or n_axis * n_axis != spp:
        raise ValueError(
            f"samples_per_pixel must be 1 or a perfect square (4, 9, 16, ...), "
            f"got {samples_per_pixel}"
        )

    if wrap not in ("repeat", "clamp"):
        raise ValueError(f"wrap must be 'repeat' or 'clamp', got {wrap!r}")

    # ---- camera setup (uses union AABB of WORLD-space mesh points) ----
    union_points = _union_world_points(scene_entries)

    # Explicit camera_matrix wins over the Scene's active Camera.  If
    # neither is given, build a default 3/4 view on the union AABB.
    is_orthographic = (
        bool(scene_camera.get("is_orthographic")) if scene_camera else False
    )
    ortho_height = scene_camera.get("ortho_height") if scene_camera else None

    if camera_matrix is None and scene_camera is not None:
        camera_matrix = scene_camera["world_matrix"]
        if scene_camera.get("angle_of_view") is not None:
            angle_of_view = float(scene_camera["angle_of_view"])

    if camera_matrix is None:
        camera_matrix = _default_camera_for_points(union_points)
        autofit       = True

    cam = np.asarray(camera_matrix, dtype=np.float64)
    if cam.shape != (4, 4):
        raise ValueError(f"camera_matrix must be (4, 4), got {cam.shape}")

    if autofit and not is_orthographic:
        # Autofit safety margin: bezier patches bulge outside the
        # control-point hull, so a tight autofit clips the silhouette.
        # See _resolve_fit_padding for the explicit-vs-default policy.
        pad = _resolve_fit_padding(fit_padding, scene_entries)
        cam = _autofit_camera_for_points(
            _inflate_points_about_center(union_points, pad),
            cam,
            float(angle_of_view),
            width,
            height,
        )

    # ---- generate camera rays once, shared across all meshes ----
    if is_orthographic:
        if ortho_height is None:
            ortho_height = _derive_ortho_height(union_points, cam) * 1.05
        origins, directions = _generate_camera_rays_orthographic(
            cam, float(ortho_height), width, height, n_axis
        )
    else:
        origins, directions = _generate_camera_rays(
            cam, float(angle_of_view), width, height, n_axis
        )
    n_total_samples = origins.shape[0]
    n_pixels        = width * height

    # ---- assemble light list ----
    lights = _build_light_list(
        scene_lights        = scene_lights,
        point_light_arg     = point_light,
        default_light       = default_light,
        camera_world_matrix = cam,
    )

    # ---- per-mesh shading + z-merge ----
    z_color, z_hit, z_depth = _render_scene(
        scene_entries = scene_entries,
        origins       = origins,
        directions    = directions,
        n_total       = n_total_samples,
        lights        = lights,
    )

    # ---- assemble the final per-sample buffer + MSAA resolve ----
    return _compose_scene_output(
        z_color       = z_color,
        z_hit         = z_hit,
        z_depth       = z_depth,
        n_pixels      = n_pixels,
        spp           = spp,
        height        = height,
        width         = width,
        background    = background,
        return_depth  = return_depth,
        camera_matrix = cam,
        angle_of_view = float(angle_of_view),
    )


# -- Scene normalization ------------------------------------------------------


def _build_scene_entries_dispatch(
    scene,
    single_mesh,
    single_uv,
    single_texture,
    single_base_color,
    single_sample_method,
    single_wrap,
    single_ambient,
    single_twosided,
):
    """Routes scene input to the right normalizer.

    Returns ``(entries, camera_dict_or_None, lights_list)`` where:
      - entries is the per-mesh list[dict] consumed by _render_scene
      - camera_dict has keys ``world_matrix`` / ``angle_of_view`` /
        ``is_orthographic`` / ``ortho_height``, or None if no Scene Camera
      - lights_list is the list of light dicts from Scene Lights, or []
    """
    if isinstance(scene, Scene):
        return _build_scene_entries_from_scene(
            scene_obj            = scene,
            single_sample_method = single_sample_method,
            single_wrap          = single_wrap,
            single_ambient       = single_ambient,
            single_twosided      = single_twosided,
        )

    # Legacy paths: list-of-dicts OR positional single-mesh args
    entries = _build_scene_entries(
        scene=scene,
        single_mesh=single_mesh,
        single_uv=single_uv,
        single_texture=single_texture,
        single_base_color=single_base_color,
        single_sample_method=single_sample_method,
        single_wrap=single_wrap,
        single_ambient=single_ambient,
        single_twosided=single_twosided,
    )
    return entries, None, []


def _build_scene_entries(
    scene,
    single_mesh,
    single_uv,
    single_texture,
    single_base_color,
    single_sample_method,
    single_wrap,
    single_ambient,
    single_twosided,
) -> List[dict]:
    """Normalize legacy inputs (list[dict] or single mesh+uv+texture) to
    the per-mesh entry dicts ``_render_scene`` consumes.

    Each entry has an identity ``world_matrix`` because legacy callers
    work in world space already.
    """
    if scene is None:
        if single_mesh is None:
            return []
        raw = [
            {
                "mesh":          single_mesh,
                "uv":            single_uv,
                "texture":       single_texture,
                "base_color":    single_base_color,
                "sample_method": single_sample_method,
                "wrap":          single_wrap,
                "ambient":       single_ambient,
                "twosided":      single_twosided,
            }
        ]
    else:
        raw = list(scene)

    entries: List[dict] = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict) or "mesh" not in e:
            raise ValueError(f"scene[{i}] must be a dict with at least a 'mesh' key")
        mesh_i    = e["mesh"]
        uv_i      = e.get("uv")
        tex_i_raw = e.get("texture")
        if tex_i_raw is not None and uv_i is None:
            raise ValueError(f"scene[{i}]: uv is required when texture is provided")
        if uv_i is not None:
            if mesh_i.face_count != uv_i.face_count:
                raise ValueError(
                    f"scene[{i}]: mesh.face_count ({mesh_i.face_count}) != "
                    f"uv.face_count ({uv_i.face_count})"
                )
            if not np.array_equal(mesh_i.counts, uv_i.counts):
                raise ValueError(f"scene[{i}]: mesh.counts != uv.counts")

        entries.append(
            {
                "mesh":          mesh_i,
                "uv":            uv_i,
                "tex":           _load_texture(tex_i_raw) if tex_i_raw is not None else None,
                "base_color":    e.get("base_color", single_base_color),
                "sample_method": e.get("sample_method", single_sample_method),
                "wrap":          e.get("wrap", single_wrap),
                "ambient":       e.get("ambient", single_ambient),
                "twosided":      e.get("twosided", single_twosided),
                "world_matrix":  np.eye(4, dtype=np.float64),
            }
        )
    return entries


def _build_scene_entries_from_scene(
    scene_obj,
    single_sample_method,
    single_wrap,
    single_ambient,
    single_twosided,
):
    """Convert a :class:`Scene` to ``(entries, camera_dict, lights_list)``.

    Each visible :class:`Object` becomes an entry dict.  The active
    :class:`Camera` (or first if none designated) becomes the camera
    dict.  All :class:`Light` nodes become light dicts.

    Object/Camera world matrices come from TransformData in **row-major**
    Maya convention; we transpose them to **column-major** OpenGL form
    here so the renderer's downstream code (which is column-major
    throughout) sees the convention it expects.
    """
    entries: List[dict] = []
    for obj in scene_obj.objects:
        if not obj.visibility or obj.mesh is None:
            continue
        if obj.texture is not None and obj.uv is None:
            raise ValueError(
                f"Object {obj.name!r}: uv is required when texture is provided"
            )
        if obj.uv is not None:
            if obj.mesh.face_count != obj.uv.face_count:
                raise ValueError(
                    f"Object {obj.name!r}: mesh.face_count != uv.face_count"
                )
            if not np.array_equal(obj.mesh.counts, obj.uv.counts):
                raise ValueError(f"Object {obj.name!r}: mesh.counts != uv.counts")

        # Row-major (TransformData) -> column-major (renderer) for the
        # Object's world transform.  The downstream raycast uses M @ vec
        # math in column-major form so this transpose is required.
        wm_row = np.asarray(obj.world_matrix, dtype=np.float64)
        entries.append(
            {
                "mesh":          obj.mesh,
                "uv":            obj.uv,
                "tex":           obj.get_loaded_texture(),
                "base_color":    obj.base_color,
                "sample_method": obj.sample_method,
                "wrap":          obj.wrap,
                "ambient":       obj.ambient,
                "twosided":      obj.twosided,
                "world_matrix":  wm_row.T,
            }
        )

    cam_dict = None
    cam      = scene_obj.get_camera()
    if cam is not None:
        # Same row->col convention bridge for the camera.
        cam_world_row = np.asarray(cam.world_matrix, dtype=np.float64)
        cam_dict = {
            "world_matrix":    cam_world_row.T,
            "angle_of_view":   float(cam.angle_of_view),
            "is_orthographic": bool(cam.is_orthographic),
            "ortho_height":    cam.ortho_height,
        }

    lights = [light.as_dict() for light in scene_obj.lights]
    return entries, cam_dict, lights


def _union_world_points(scene_entries: List[dict]) -> np.ndarray:
    """Concatenate all entries' WORLD-space points (per-entry world_matrix
    applied) for camera autofit / default view.

    Entries store ``world_matrix`` in column-major OpenGL convention
    (translation in ``M[:3, 3]``).  For a batch ``pts`` (N, 3), the
    column-major forward transform is::

        world = pts @ M[:3, :3].T + M[:3, 3]
    """
    chunks = []
    # Wrap matmul in errstate to silence spurious BLAS warnings from Apple's
    # Accelerate framework on arm64 (divide/over/invalid raised on plain matmul).
    with np.errstate(all="ignore"):
        for e in scene_entries:
            pts = np.asarray(e["mesh"].points, dtype=np.float64)
            if pts.shape[1] != 3:
                chunks.append(pts)
                continue
            M = e["world_matrix"]
            chunks.append(pts @ M[:3, :3].T + M[:3, 3])
    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks, axis=0)


def _scene_uses_bezier(scene_entries: List[dict]) -> bool:
    """True if any entry uses bezier sampling.

    Bezier patches bulge outside the control-point hull, so a tight
    autofit clips the silhouette.  The autofit path uses this to decide
    whether to default ``fit_padding`` to 1.15 (margin) or 1.0 (no
    change).  Accepts both string (``"bezier"``) and enum
    (``SampleMethod.BEZIER``) forms of the per-entry sample_method.
    """
    for e in scene_entries:
        sm = e.get("sample_method")
        if sm is None:
            continue
        if sm is SampleMethod.BEZIER:
            return True
        if isinstance(sm, str) and sm.lower() == "bezier":
            return True
    return False


def _resolve_fit_padding(
    fit_padding: Optional[float], scene_entries: List[dict]
) -> float:
    """Pick the autofit safety-margin factor.

    Explicit kwarg wins; otherwise bezier sampling auto-defaults to 1.15
    (~15% silhouette margin) and everything else stays at 1.0 so legacy
    bilinear renders frame identically.
    """
    if fit_padding is not None:
        return float(fit_padding)
    return 1.15 if _scene_uses_bezier(scene_entries) else 1.0


def _build_light_list(
    scene_lights:        List[dict],
    point_light_arg:     Optional[Union[dict, List[dict]]],
    default_light:       bool,
    camera_world_matrix: np.ndarray,
) -> List[dict]:
    """Combine Scene lights + point_light= argument; fall back to a
    headlight if neither is set and ``default_light`` is True."""
    out = list(scene_lights)
    if point_light_arg is not None:
        if isinstance(point_light_arg, dict):
            out.append(point_light_arg)
        else:
            out.extend(list(point_light_arg))

    if not out and default_light:
        # Headlight: at the camera's world position with modest intensity.
        # ``camera_world_matrix`` is column-major OpenGL, so the position
        # lives in the last COLUMN, not the last row.
        cam_pos = camera_world_matrix[:3, 3]
        ref     = max(float(np.linalg.norm(cam_pos)), 1.0)
        out.append(
            {
                "kind":      "point",
                "position":  tuple(cam_pos),
                "color":     (1.0, 1.0, 1.0),
                "intensity": ref**2,  # roughly unit-bright at distance ref
                "falloff":   True,
            }
        )
    return out


# -- Per-scene-entry shading --------------------------------------------------


def _render_scene(
    scene_entries: List[dict],
    origins:       np.ndarray,
    directions:    np.ndarray,
    n_total:       int,
    lights:        List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast + shade each scene entry, then z-merge per sample.

    Returns:
        z_color  : ``(n_total, 3)`` float32, color of the closest hit.
        z_hit    : ``(n_total,)`` bool, True where any mesh hit.
        z_depth  : ``(n_total,)`` float64, distance to the closest hit
                   (``+inf`` where no hit).
    """
    z_color = np.zeros((n_total, 3), dtype=np.float32)
    z_depth = np.full(n_total, np.inf, dtype=np.float64)
    z_hit   = np.zeros(n_total, dtype=bool)

    for e in scene_entries:
        sample_color, sample_hit, sample_distance = _shade_mesh_per_sample(
            entry      = e,
            origins    = origins,
            directions = directions,
            n_total    = n_total,
            lights     = lights,
        )
        winner = sample_hit & (sample_distance < z_depth)
        if winner.any():
            z_color[winner] = sample_color[winner]
            z_depth[winner] = sample_distance[winner]
            z_hit[winner]   = True

    return z_color, z_hit, z_depth


def _shade_mesh_per_sample(
    entry:      dict,
    origins:    np.ndarray,
    directions: np.ndarray,
    n_total:    int,
    lights:     List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast + shade ONE entry, returning per-sample color / hit / distance.

    Per-sample misses get color = 0 and distance = +inf so the z-merge
    in :func:`_render_scene` ignores them.

    When the entry has a non-identity ``world_matrix``, rays are cast in
    object-local space (cheap matrix-vec per ray; the Object's cached
    BVH stays valid) and the resulting hit projections + normals are
    transformed back to world space before shading.  World-space hit
    distance is recomputed for correct z-merging across objects.
    """
    mesh     = entry["mesh"]
    M        = entry["world_matrix"]  # column-major OpenGL convention
    is_world = np.allclose(M, np.eye(4))

    if is_world:
        cast_origins    = origins
        cast_directions = directions
    else:
        Minv = np.linalg.inv(M)
        R3   = Minv[:3, :3]
        # Column-major forward: world = obj @ M[:3, :3].T + M[:3, 3]
        # So inverse:        obj = world @ Minv[:3, :3].T + Minv[:3, 3]
        # errstate: silence spurious BLAS warnings on Apple Accelerate (arm64).
        with np.errstate(all="ignore"):
            cast_origins    = origins @ R3.T + Minv[:3, 3]
            cast_directions = directions @ R3.T  # vectors: no translation

    raycast = mesh.raycast(
        cast_origins,
        cast_directions,
        method       = entry["sample_method"],
        forward_only = True,
        twosided     = entry["twosided"],
    )

    sample_color    = np.zeros((n_total, 3), dtype=np.float32)
    sample_distance = np.full(n_total, np.inf, dtype=np.float64)
    sample_hit      = raycast.hit

    if not np.any(sample_hit):
        return sample_color, sample_hit, sample_distance

    n_hit = int(np.sum(sample_hit))
    tex   = entry["tex"]
    if tex is None:
        sampled = np.tile(np.asarray(entry["base_color"], dtype=np.float32), (n_hit, 1))
    else:
        interp_uvs = _interpolate_uvs(
            entry["uv"], raycast.indices[sample_hit], raycast.weights[sample_hit]
        )
        sampled = _sample_texture_bilinear(tex, interp_uvs, entry["wrap"] == "repeat")

    # ---- back-transform projections + normals to world space ----
    hit_proj = raycast.projections[sample_hit]
    hit_norm = raycast.normals[sample_hit]
    if not is_world:
        # Column-major forward transform of points (translate in [:3, 3]).
        # errstate: silence spurious BLAS warnings on Apple Accelerate (arm64).
        with np.errstate(all="ignore"):
            hit_proj = hit_proj @ M[:3, :3].T + M[:3, 3]
            # Normals: inverse-transpose of the upper-left 3x3.
            N3       = np.linalg.inv(M[:3, :3]).T
            hit_norm = hit_norm @ N3.T
        nlen     = np.linalg.norm(hit_norm, axis=1, keepdims=True)
        nlen     = np.maximum(nlen, 1e-12)
        hit_norm = hit_norm / nlen

    if lights:
        shaded = _shade_lights(sampled, hit_proj, hit_norm, lights, entry["ambient"])
    else:
        # Pure ambient (no lights): scale base color by max(ambient, 1)
        # to match the prior single-mesh "no-light" behavior.
        shaded = sampled * max(entry["ambient"], 1.0)

    sample_color[sample_hit] = np.clip(shaded, 0.0, 1.0).astype(np.float32)

    # World-space hit distance (object-local t is meaningless across objects).
    if is_world:
        sample_distance[sample_hit] = raycast.distances[sample_hit]
    else:
        diff                        = hit_proj - origins[sample_hit]
        sample_distance[sample_hit] = np.linalg.norm(diff, axis=1)

    return sample_color, sample_hit, sample_distance


def _compose_scene_output(
    z_color:       np.ndarray,
    z_hit:         np.ndarray,
    z_depth:       np.ndarray,
    n_pixels:      int,
    spp:           int,
    height:        int,
    width:         int,
    background:    Tuple[float, float, float, float],
    return_depth:  bool,
    camera_matrix: Optional[np.ndarray]              = None,
    angle_of_view: Optional[float]                   = None,
) -> Frame:
    """MSAA resolve + premultiplied "over" composite.  Always 4-channel.

    Each subsample is either a hit (shaded RGB, alpha=1) or a miss
    (alpha=0).  After MSAA averaging the alpha channel encodes coverage
    in [0, 1].  We then composite the rendered RGBA over the user-
    supplied RGBA background using premultiplied Porter-Duff "over" and
    return un-premultiplied RGBA so saved PNGs look right.

    ``camera_matrix`` (column-major) and ``angle_of_view`` (degrees) are
    attached to the Frame so post-processing helpers like
    :meth:`Frame.wireframe` can project geometry to screen without the
    caller re-passing camera info.
    """
    n_total = n_pixels * spp

    per_sample            = np.zeros((n_total, 4), dtype=np.float32)
    per_sample[z_hit, :3] = z_color[z_hit]
    per_sample[z_hit, 3]  = 1.0

    rendered   = _resolve_msaa(per_sample, n_pixels, spp)
    composited = _composite_over(rendered, background)
    image      = _to_uint8_image(composited, height, width, n_channels=4)

    depth_buffer: Optional[np.ndarray] = None
    if return_depth:
        distances = np.where(z_hit, z_depth, np.nan)
        depth_buffer = _resolve_msaa_depth(distances, z_hit, n_pixels, spp).reshape(
            height, width
        )

    return Frame(
        array = image,
        depth = depth_buffer,
        camera_matrix=(
            np.asarray(camera_matrix, dtype=np.float64)
            if camera_matrix is not None
            else None
        ),
        angle_of_view=(float(angle_of_view) if angle_of_view is not None else None),
    )


def _resolve_msaa(per_sample: np.ndarray, n_pixels: int, spp: int) -> np.ndarray:
    """Average ``spp`` RGBA samples per pixel.  Alpha = coverage in [0, 1].

    RGB is averaged only over hit samples (a straight mean over all
    samples would darken edges as miss-zeros pull the average down);
    alpha is set to ``hits / spp``.
    """
    if spp == 1:
        return per_sample

    grouped      = per_sample.reshape(n_pixels, spp, 4)
    sample_alpha = grouped[..., 3]
    n_hits       = sample_alpha.sum(axis=1)
    coverage     = n_hits / spp
    sum_color    = grouped[..., :3].sum(axis=1)
    safe_n       = np.maximum(n_hits, 1.0)[:, None]
    avg_color    = sum_color / safe_n
    out          = np.zeros((n_pixels, 4), dtype=np.float32)
    out[:, :3]   = avg_color.astype(np.float32)
    out[:, 3]    = coverage.astype(np.float32)
    return out


def _composite_over(
    src: np.ndarray, bg_rgba: Tuple[float, float, float, float]
) -> np.ndarray:
    """Premultiplied Porter-Duff "over": rendered ``src`` over ``bg_rgba``.

    ``src`` arrives un-premultiplied with alpha = coverage in [0, 1].
    Returns un-premultiplied RGBA so the saved PNG behaves intuitively.

    Boundary cases (assuming a hit pixel and a miss pixel)::

        bg=(0,0,0,0)        miss -> (0,0,0,0)        hit -> (R,G,B,1)
        bg=(0,0,0,1)        miss -> (0,0,0,1)        hit -> (R,G,B,1)
        bg=(1,1,1,1)        miss -> (1,1,1,1)        hit -> (R,G,B,1)
        bg=(1,0,0,0.5)      miss -> (1,0,0,0.5)      hit -> (R,G,B,1)

    Args:
        src: ``(N, 4)`` float32 in [0, 1] -- rendered RGBA.
        bg_rgba: 4-tuple of floats in [0, 1] -- background.

    Returns:
        ``(N, 4)`` float32 in [0, 1] -- composited, un-premultiplied.

    Raises:
        ValueError: When ``bg_rgba`` is not a 4-tuple of floats in
            ``[0, 1]``.
    """
    bg = np.asarray(bg_rgba, dtype=np.float32)
    if bg.shape != (4,):
        raise ValueError(f"background must be RGBA 4-tuple of floats; got {bg_rgba!r}")
    if np.any(bg < 0.0) or np.any(bg > 1.0):
        raise ValueError(f"background components must be in [0, 1]; got {bg_rgba!r}")

    src_a  = src[:, 3:4]
    dst_a  = float(bg[3])
    src_pm = src[:, :3] * src_a
    dst_pm = bg[:3] * dst_a

    one_minus_src_a = 1.0 - src_a
    out_pm_rgb      = src_pm + dst_pm * one_minus_src_a
    out_a           = src_a + dst_a * one_minus_src_a

    safe_a  = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = np.where(out_a > 1e-6, out_pm_rgb / safe_a, 0.0)

    return np.concatenate([out_rgb, out_a], axis=1).astype(np.float32)


def _resolve_msaa_depth(
    distances: np.ndarray, hit: np.ndarray, n_pixels: int, spp: int
) -> np.ndarray:
    """Per-pixel depth = closest hit-sample distance, NaN where all miss."""
    if spp == 1:
        depth      = np.full(n_pixels, np.nan, dtype=np.float64)
        depth[hit] = distances[hit]
        return depth

    masked  = np.where(hit, distances, np.inf).reshape(n_pixels, spp)
    closest = masked.min(axis=1)
    return np.where(np.isfinite(closest), closest, np.nan)


def _interpolate_uvs(
    uv: UVData, hit_face_idx: np.ndarray, hit_weights: np.ndarray
) -> np.ndarray:
    """Interpolate UVs at hit points using bilinear weights."""
    uv_geom = uv.geometry
    if uv_geom.shape[1] < 4:
        padded                        = np.full((uv_geom.shape[0], 4), -1, dtype=uv_geom.dtype)
        padded[:, : uv_geom.shape[1]] = uv_geom
        uv_geom                       = padded
    hit_uv_idx     = uv_geom[hit_face_idx]
    safe_uv_idx    = np.where(hit_uv_idx >= 0, hit_uv_idx, 0)
    hit_uv_corners = uv.points[safe_uv_idx]
    return np.einsum("ij,ijk->ik", hit_weights, hit_uv_corners)


# -- Camera ray generation ---------------------------------------------------


def _generate_camera_rays(
    camera_matrix: np.ndarray,
    fov_y_deg:     float,
    width:         int,
    height:        int,
    n_axis:        int        = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel ray origins (all = camera position) and world-space dirs.

    Pixel (0, 0) is top-left.  Camera looks along -Z (Maya/OpenGL).
    ``n_axis > 1`` enables MSAA on a regular sub-pixel grid.
    """
    fov_y  = np.deg2rad(fov_y_deg)
    aspect = width / height
    half_h = np.tan(fov_y * 0.5)
    half_w = half_h * aspect

    sub = (np.arange(n_axis, dtype=np.float64) + 0.5) / n_axis
    sub_x, sub_y = np.meshgrid(sub, sub, indexing="xy")
    sub_x = sub_x.ravel()
    sub_y = sub_y.ravel()

    cols  = np.arange(width, dtype=np.float64)
    rows  = np.arange(height, dtype=np.float64)

    px    = (cols[None, :, None] + sub_x[None, None, :]) / width
    py    = (rows[:, None, None] + sub_y[None, None, :]) / height
    ndc_x = px * 2.0 - 1.0
    ndc_y = 1.0 - py * 2.0
    spp   = n_axis * n_axis
    ndc_x = np.broadcast_to(ndc_x, (height, width, spp))
    ndc_y = np.broadcast_to(ndc_y, (height, width, spp))

    cam_dirs = np.stack(
        [ndc_x * half_w, ndc_y * half_h, -np.ones_like(ndc_x)], axis=-1
    ).reshape(-1, 3)
    cam_dirs = cam_dirs / np.linalg.norm(cam_dirs, axis=1, keepdims=True)

    R = camera_matrix[:3, :3]
    # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64).
    with np.errstate(all="ignore"):
        world_dirs = cam_dirs @ R.T

    cam_pos = camera_matrix[:3, 3]
    origins = np.tile(cam_pos[None, :], (cam_dirs.shape[0], 1))

    return origins.astype(np.float64), world_dirs.astype(np.float64)


def _generate_camera_rays_orthographic(
    camera_matrix: np.ndarray,
    ortho_height:  float,
    width:         int,
    height:        int,
    n_axis:        int        = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthographic ray bundle: parallel -Z rays at origins spanning a
    rectangle of size ``(ortho_height * aspect, ortho_height)`` centered
    on the camera position.
    """
    aspect = width / height
    half_h = 0.5 * ortho_height
    half_w = half_h * aspect

    sub = (np.arange(n_axis, dtype=np.float64) + 0.5) / n_axis
    sub_x, sub_y = np.meshgrid(sub, sub, indexing="xy")
    sub_x = sub_x.ravel()
    sub_y = sub_y.ravel()

    cols  = np.arange(width, dtype=np.float64)
    rows  = np.arange(height, dtype=np.float64)

    px    = (cols[None, :, None] + sub_x[None, None, :]) / width
    py    = (rows[:, None, None] + sub_y[None, None, :]) / height
    ndc_x = px * 2.0 - 1.0
    ndc_y = 1.0 - py * 2.0
    spp   = n_axis * n_axis
    ndc_x = np.broadcast_to(ndc_x, (height, width, spp))
    ndc_y = np.broadcast_to(ndc_y, (height, width, spp))

    cam_origins = np.stack(
        [ndc_x * half_w, ndc_y * half_h, np.zeros_like(ndc_x)], axis=-1
    ).reshape(-1, 3)
    cam_dirs = np.tile(
        np.array([0.0, 0.0, -1.0], dtype=np.float64), (cam_origins.shape[0], 1)
    )

    R = camera_matrix[:3, :3]
    # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64).
    with np.errstate(all="ignore"):
        world_dirs    = cam_dirs @ R.T
        world_origins = cam_origins @ R.T + camera_matrix[3, :3]

    return world_origins.astype(np.float64), world_dirs.astype(np.float64)


# -- Shading -----------------------------------------------------------------


def _shade_lights(
    base_color: np.ndarray,
    hit_points: np.ndarray,
    normals:    np.ndarray,
    lights:     List[dict],
    ambient:    float,
) -> np.ndarray:
    """Multi-light Lambertian: ``base * (ambient + sum_i contribution_i)``.

    Each light dict supports ``kind`` (``"point"`` default or ``"infinite"``),
    ``position`` / ``direction``, ``color``, ``intensity``, ``falloff``.
    """
    n     = hit_points.shape[0]
    accum = np.zeros((n, 3), dtype=np.float64)

    # Pre-normalize normals once.
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norm = np.maximum(n_norm, 1e-12)
    N      = normals / n_norm

    for light in lights:
        accum += _shade_one_light(hit_points, N, light)

    out = base_color * (ambient + accum.astype(np.float32))
    return out.astype(np.float32)


def _shade_one_light(
    hit_points: np.ndarray,
    N:          np.ndarray,
    light:      dict,
) -> np.ndarray:
    """Returns one light's per-hit (R, G, B) contribution (unweighted by base
    color).  Assumes ``N`` is already unit length.
    """
    kind      = light.get("kind", "point")
    color     = np.asarray(light.get("color", (1.0, 1.0, 1.0)), dtype=np.float64)
    intensity = float(light.get("intensity", 1.0))

    if kind == "infinite":
        d = np.asarray(light.get("direction"), dtype=np.float64)
        if d is None or d.shape != (3,):
            raise ValueError("infinite light requires 'direction' (3-vector)")
        L     = -d / max(float(np.linalg.norm(d)), 1e-12)
        NdotL = np.maximum(N @ L, 0.0)
        return (NdotL * intensity)[:, None] * color[None, :]

    # default: point light
    pos = np.asarray(light.get("position"), dtype=np.float64)
    if pos is None or pos.shape != (3,):
        raise ValueError("point light requires 'position' (3-vector)")
    falloff  = bool(light.get("falloff", True))

    to_light = pos[None, :] - hit_points
    dist2    = np.einsum("ij,ij->i", to_light, to_light)
    dist     = np.sqrt(np.maximum(dist2, 1e-20))
    L        = to_light / dist[:, None]

    NdotL = np.maximum(np.einsum("ij,ij->i", N, L), 0.0)
    if falloff:
        attenuation = intensity / dist2
    else:
        attenuation = np.full_like(NdotL, intensity)

    return (NdotL * attenuation)[:, None] * color[None, :]


def _shade_lambert(
    base_color: np.ndarray,
    hit_points: np.ndarray,
    normals:    np.ndarray,
    light:      dict,
    ambient:    float,
) -> np.ndarray:
    """Backward-compat single-light shader.  Kept so anything that imports
    this directly keeps working; new code should use :func:`_shade_lights`.
    """
    return _shade_lights(base_color, hit_points, normals, [light], ambient)


# -- Output formatting -------------------------------------------------------


def _to_uint8_image(
    flat: np.ndarray, height: int, width: int, n_channels: int = 3
) -> np.ndarray:
    return (np.clip(flat, 0.0, 1.0).reshape(height, width, n_channels) * 255.0).astype(
        np.uint8
    )