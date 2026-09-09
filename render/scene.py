"""
Scene-graph primitives for the raytracer.

A :class:`Scene` is a :class:`HierarchyData` of :class:`Object`,
:class:`Camera`, and :class:`Light` nodes (plus optional plain
:class:`TransformData` group nodes for hierarchy).  Each renderable node
inherits Maya-style transforms (translate / rotate / scale / parenting)
from :class:`TransformData`, so:

  - parent a Camera under an Object -> orbit camera
  - parent a Light under a Camera -> headlight
  - same MeshData under multiple Objects -> free instancing (BVH shared)

Minimum viable scene::

    from cgmath.geometry.mesh import MeshData, UVData
    from cgmath.render.scene import Camera, Light, Object, Scene
    from cgmath.render.raytracer import render

    scene = Scene("scene")
    scene.append(Object(name="hero", mesh=mesh, uv=uv, texture="hero.png"))
    scene.append(Camera(name="cam"))
    scene.append(Light(name="key", kind="point", intensity=1e5))
    img = render(scene=scene, resolution=(640, 480))

If no Camera is in the scene, a default 3/4 elevated camera is built and
autofit to the scene.  If no Light is in the scene and ``default_light``
is not disabled, a "headlight" is added parented to the camera.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.mesh import (
    load_fbx as _load_fbx,
    load_glb as _load_glb,
    load_obj as _load_obj,
    MeshData,
    UVData,
)
from cgmath.geometry.skin_weights import (
    load_fbx as _load_skin_fbx,
    load_glb as _load_skin_glb,
)
from cgmath.render.camera import (
    _autofit_camera_for_points,
    _default_camera_for_points,
    _inflate_points_about_center,
    look_at as _look_at,
)
from cgmath.render.texture import _load_texture
from cgmath.hierarchy import HierarchyData, TransformData

# opencv (optional) -- required by :meth:`Object.imshow` / :meth:`Scene.imshow`.
# Lazy-import pattern matching ``cgmath.geometry.mesh`` so the module loads
# fine in headless / no-cv2 environments; the imshow methods raise a clear
# RuntimeError when actually invoked without cv2.
try:
    import cv2
except ImportError:
    cv2 = None

# PIL (optional) -- required by :meth:`Object.to_image` / :meth:`Scene.to_image`.
# Mirrors the lazy-import pattern in ``cgmath.geometry.mesh`` so the module
# loads fine in environments without PIL; the to_image methods raise a clear
# RuntimeError when actually invoked without PIL.
try:
    from PIL import Image
except ImportError:
    Image = None

# Autodesk FBX SDK (optional) -- only required by :meth:`Object.load_fbx` and
# :meth:`Object.extract_texture_from_fbx`. Lazy-import keeps the module
# loadable in environments without the FBX SDK installed.
try:
    import fbx as _fbx
except ImportError:
    _fbx = None


# -- Object --------------------------------------------------------------------


@dataclass(repr=False, eq=False)
class Object(TransformData):
    """A renderable mesh with attached UVs, texture and shader options.

    Domain fields are declared as ``@dataclass`` defaults so that the
    inherited :meth:`Data.to_dict` / :meth:`Data.from_dict` round-trip
    them automatically.

    The ``texture`` attribute may be a path to an image or a pre-loaded
    ``(H, W, 3)`` float32 array.  When it's a path, the renderer loads
    it lazily on first render and caches the result on the Object via
    the (non-serialized) ``_loaded_texture`` attribute.

    Render cache (``frame`` / ``buffer``):
      :meth:`render`, :meth:`imshow`, and :meth:`to_image` populate a
      :attr:`frame` cache so repeated display calls don't re-run the
      raytracer.  The cache auto-invalidates lazily on access when:

        - any tracked visual attribute is reassigned (``mesh`` /
          ``uv`` / ``texture`` / ``base_color`` / ``wireframe*`` /
          ``sample_method`` / ``wrap`` / ``ambient`` / ``twosided`` /
          ``cast_shadows`` / ``resolution`` / ``samples_per_pixel``)
          -- via the existing setter -> :meth:`_mark_render_dirty`
          plumbing, OR
        - the Object's local transform (``translate`` / ``rotate`` /
          ``scale``) OR any ancestor's transform has changed since
          the cached frame was rendered -- detected via the
          transform-state snapshot taken in the :attr:`frame` setter
          (see :meth:`_capture_transform_state`).

      In-place mutation of ``mesh.points`` or texture pixel arrays
      still requires an explicit :meth:`render` call to refresh.
    """

    # Renderable payload
    _mesh:    Optional[MeshData] = None
    _uv:      Optional[UVData] = None
    _texture: Optional[Union[str, np.ndarray]] = None

    # Skeletal deformer driving ``_mesh`` -- see :meth:`pose`.  Not a render
    # input: attaching one changes nothing on screen until ``pose()`` runs,
    # and that assigns ``mesh``, which invalidates through the usual setter.
    _skin: Optional[SkinDeformData] = None

    # Per-Object shader knobs (override Scene/render() defaults)
    _base_color:    Tuple[float, float, float] = (0.7, 0.7, 0.7)
    _sample_method: str = "bilinear"  # or "bezier"
    _wrap:          str = "repeat"    # or "clamp"
    _ambient:       float = 0.8
    _twosided:      bool = True
    _cast_shadows:  bool = True       # reserved for shadow-ray pass

    # Per-Object preview RGBA background.  Used only when this Object
    # is rendered via :meth:`render` / :meth:`to_image` / :meth:`imshow`
    # / :meth:`turntable` standalone.  IGNORED when parented to a
    # :class:`Scene` -- ``Scene.background`` wins there, matching the
    # ``resolution`` / ``samples_per_pixel`` precedence pattern.  RGB
    # 3-tuples are auto-promoted to opaque ``(R, G, B, 1)`` by the
    # :func:`_normalize_rgba` helper; ``None`` falls through to the
    # renderer's own default ``(0, 0, 0, 0)`` (transparent black).
    _background: Optional[Tuple[float, float, float, float]] = None

    # Per-Object frame-level render config -- only consulted by the
    # standalone preview entry points (:meth:`render` / :meth:`turntable`
    # / :meth:`imshow` / :meth:`to_image`).  Renders that go through a
    # user-built parent :class:`Scene` IGNORE these -- frame settings on
    # the Scene win there (see :attr:`Scene._RENDER_CONFIG_KEYS`).
    # Settable via property setters that invalidate the cached frame.
    _resolution:        Tuple[int, int] = (500, 500)
    _samples_per_pixel: Optional[int] = None

    # Per-Object wireframe overlay (baked into a copy of the diffuse via
    # ``UVData.draw_edges()``).  Settable via property setters that
    # invalidate the bake cache when any of these or ``texture`` / ``uv``
    # change.  Default off; opt in per Object.
    _wireframe:           bool = False
    _wireframe_color:     Tuple[int, int, int] = (0, 0, 0)
    _wireframe_thickness: int = 1

    # Lazy-loaded texture cache (intentionally NOT a dataclass field).
    _loaded_texture: Optional[np.ndarray] = None
    # Lazy-baked wireframe-on-diffuse cache (also non-serialized).
    _wired_texture_cache: Optional[np.ndarray] = None
    # Lazy-built bind-pose mesh handed to the deformer each frame (also
    # non-serialized -- rebuilt from ``_skin.rest_points``, which persists).
    _bind_mesh: Optional[MeshData] = None
    # Snapshot of the local transform state + composed world matrix
    # at the time ``frame`` was populated.  ``None`` whenever the cache
    # is empty -- used by the ``frame`` getter to lazy-invalidate when
    # transforms (``translate`` / ``rotate`` / ``scale`` on self OR any
    # ancestor) have changed since the last render.  We snapshot the
    # raw local SRT values (NOT just ``world_matrix``) because
    # ``TransformData``'s ``_world_matrix`` cache can lag local-transform
    # mutations on standalone nodes; raw SRT is always current.  We also
    # include the composed ``world_matrix`` bytes so ancestor mutations
    # (which DO propagate through ``_reset_branch_world_matrices``) are
    # caught even when our own SRT hasn't changed.
    _render_transform_state: Optional[tuple] = None

    def __init__(
        self,
        name:                str                                             = "object",
        mesh:                Optional[MeshData]                              = None,
        uv:                  Optional[UVData]                                = None,
        texture:             Optional[Union[str, np.ndarray]]                = None,
        skin:                Optional[SkinDeformData]                        = None,
        base_color:          Tuple[float, float, float]                      = (0.7, 0.7, 0.7),
        sample_method:       str                                             = "bilinear",
        wrap:                str                                             = "repeat",
        ambient:             float                                           = 0.8,
        twosided:            bool                                            = True,
        cast_shadows:        bool                                            = True,
        resolution:          Tuple[int, int]                                 = (500, 500),
        samples_per_pixel:   Optional[int]                                   = 4,
        wireframe:           bool                                            = False,
        wireframe_color:     Tuple[int, int, int]                            = (0, 0, 0),
        wireframe_thickness: int                                             = 1,
        background:          Optional[Union[Tuple[float, ...], List[float]]] = None,
        **transform_kwargs,
    ) -> None:
        super().__init__(name=name, node_type="object", **transform_kwargs)
        # Use private storage so the property setters below can validate
        # and invalidate the wired-texture cache.
        self._mesh          = mesh
        self._uv            = uv
        self._texture       = texture
        self._skin          = skin
        self._base_color    = tuple(base_color)
        self._sample_method = str(sample_method)
        self._wrap          = str(wrap)
        self._ambient       = float(ambient)
        self._twosided      = bool(twosided)
        self._cast_shadows  = bool(cast_shadows)
        self._resolution    = tuple(int(v) for v in resolution)
        self._samples_per_pixel = _validate_samples_per_pixel(
            samples_per_pixel, field_name="Object.samples_per_pixel"
        )
        self._wireframe           = bool(wireframe)
        self._wireframe_color     = tuple(wireframe_color)
        self._wireframe_thickness = int(wireframe_thickness)
        self._background          = _normalize_rgba(background, "Object.background")
        self._loaded_texture      = None
        self._wired_texture_cache = None
        self._bind_mesh           = None
        # Cached most-recent render output, accessed via the ``frame``
        # property below.  Use private storage so the property setter
        # can snapshot a transform-state tuple for later invalidation.
        # ``None`` until :meth:`render` (or :meth:`imshow`) has been
        # called.
        self._frame: Optional["Frame"] = None
        self._render_transform_state = None

    # -- file IO: load an Object from disk -----------------------------

    @classmethod
    def load_obj(cls, file_path: str, index: int = 0, **object_kwargs) -> "Object":
        """Load a mesh from an OBJ file as an :class:`Object`.

        Convenience wrapper around :func:`cgmath.geometry.mesh.load_obj`.
        Picks the ``index``-th mesh in the file (default: first) and wraps
        it in an Object with its associated UV (if any).  ``object_kwargs``
        are forwarded to :class:`Object` so callers can override
        ``texture``, ``base_color``, transforms, etc.

        Args:
            file_path: Path to the ``.obj`` file (``~`` and env vars OK).
            index: Which mesh to load when the file contains multiple
                ``g`` groups.  Negative indices count from the end.
            **object_kwargs: Forwarded to :class:`Object` (``name``,
                ``texture``, ``base_color``, ``translate``, ...).

        Returns:
            A new Object with ``mesh`` and ``uv`` populated.
        """
        data = _load_obj(file_path)
        if not data:
            raise ValueError(f"{file_path}: contains no meshes")
        if not -len(data) <= index < len(data):
            raise IndexError(
                f"{file_path}: index {index} out of range (file has {len(data)} meshes)"
            )
        mesh, uv_list = data[index]
        uv = uv_list[0] if len(uv_list) > 0 else None
        object_kwargs.setdefault("name", mesh.name or "object")
        return cls(mesh=mesh, uv=uv, **object_kwargs)

    @classmethod
    def load_glb(
        cls,
        file_path:       str,
        index:           int   = 0,
        scale_factor:    float = 100.0,
        extract_texture: bool  = True,
        load_skin:       bool  = True,
        **object_kwargs,
    ) -> "Object":
        """Load a mesh from a GLB file as an :class:`Object`.

        Convenience wrapper around :func:`cgmath.geometry.mesh.load_glb`.
        See :meth:`load_obj` for parameter semantics.

        Args:
            file_path: Path to the ``.glb`` file (``~`` and env vars OK).
            index: Which mesh to load when the file contains multiple
                geometries.  Negative indices count from the end.
            scale_factor: Multiplier applied to vertex positions (GLB
                files are typically authored in metres; the default of
                100 matches :meth:`MeshData.load_glb`).
            extract_texture: When True (default), also extract the
                material's base color map from the GLB and assign it to
                the returned Object's :attr:`texture` via
                :meth:`extract_texture_from_glb`.  An explicit
                ``texture=`` in ``object_kwargs`` always wins -- auto-
                extraction is suppressed so caller-supplied textures are
                honoured.
            load_skin: When True (default), also read this mesh's skin
                weights and bind pose and attach a :class:`SkinDeformData`
                to :attr:`skin`, ready for :meth:`pose`.  A file with no
                skinning simply yields ``None``.  When the file's meshes
                and skins cannot be paired up confidently this warns and
                leaves :attr:`skin` unset rather than mis-binding.
            **object_kwargs: Forwarded to :class:`Object`.

        Returns:
            A new Object with ``mesh``, ``uv`` and (when available and
            not overridden) ``texture`` populated.

        Raises:
            ImportError: If ``trimesh`` is not installed.
        """
        data = _load_glb(file_path, scale_factor=scale_factor)
        if not data:
            raise ValueError(f"{file_path}: contains no meshes")
        if not -len(data) <= index < len(data):
            raise IndexError(
                f"{file_path}: index {index} out of range (file has {len(data)} meshes)"
            )
        mesh, uv_list = data[index]
        uv = uv_list[0] if len(uv_list) > 0 else None
        object_kwargs.setdefault("name", mesh.name or "object")

        if load_skin and object_kwargs.get("skin") is None:
            meshes                = [m for m, _ in data]
            object_kwargs["skin"] = _glb_skins(file_path, meshes, scale_factor)[index]

        obj = cls(mesh=mesh, uv=uv, **object_kwargs)
        # Only auto-extract when the caller didn't supply an explicit
        # texture; explicit always wins.  No-op when the GLB has no
        # base color map (returns False) or PIL is unavailable.
        if extract_texture and obj.texture is None:
            obj.extract_texture_from_glb(file_path)
        return obj

    @classmethod
    def load_fbx(
        cls,
        file_path:       str,
        index:           int  = 0,
        extract_texture: bool = True,
        load_skin:       bool = True,
        **object_kwargs,
    ) -> "Object":
        """Load a mesh from an FBX file as an :class:`Object`.

        Convenience wrapper around :func:`cgmath.geometry.mesh.load_fbx`.
        See :meth:`load_obj` for parameter semantics.

        Args:
            file_path: Path to the ``.fbx`` file (``~`` and env vars OK).
            index: Which mesh to load when the file contains multiple
                geometries.  Negative indices count from the end.
            extract_texture: When True (default), also extract the
                material's diffuse (base color) map from the FBX and
                assign it to the returned Object's :attr:`texture` via
                :meth:`extract_texture_from_fbx`.  Embedded media is
                automatically unpacked.  An explicit ``texture=`` in
                ``object_kwargs`` always wins -- auto-extraction is
                suppressed so caller-supplied textures are honoured.
            load_skin: When True (default), also read this mesh's skin
                weights and bind pose and attach a :class:`SkinDeformData`
                to :attr:`skin`, ready for :meth:`pose`.  A mesh with no
                skin simply yields ``None``.
            **object_kwargs: Forwarded to :class:`Object`.

        Returns:
            A new Object with ``mesh``, ``uv`` (first UV channel) and
            (when available and not overridden) ``texture`` populated.

        Raises:
            ImportError: If the Autodesk FBX Python SDK is not installed.
        """
        data = _load_fbx(file_path)
        if not data:
            raise ValueError(f"{file_path}: contains no meshes")
        if not -len(data) <= index < len(data):
            raise IndexError(
                f"{file_path}: index {index} out of range (file has {len(data)} meshes)"
            )
        mesh, uv_list = data[index]
        uv = uv_list[0] if len(uv_list) > 0 else None
        object_kwargs.setdefault("name", mesh.name or "object")

        if load_skin and object_kwargs.get("skin") is None:
            object_kwargs["skin"] = _fbx_skin(file_path, mesh)

        obj = cls(mesh=mesh, uv=uv, **object_kwargs)
        if extract_texture and obj.texture is None:
            obj.extract_texture_from_fbx(file_path, mesh_index=index)
        return obj

    def extract_texture_from_glb(
        self,
        file_path:      str,
        material_index: int = 0,
    ) -> bool:
        """Extract the PBR base color (diffuse / albedo) texture from a
        GLB file and wire it into this Object's :attr:`texture`.

        Uses :func:`_extract_glb_textures` internally, which also
        extracts the other standard PBR slots (``metallic_roughness`` /
        ``normal`` / ``occlusion`` / ``emissive``).  Those are discarded
        here but the parser is ready for a future multi-texture shader
        upgrade.

        Args:
            file_path: Path to the ``.glb`` file (``~`` and env vars OK).
            material_index: Which material to pull the base color from.
                Defaults to the first material.  Negative indices count
                from the end.

        Returns:
            True when a base color map was found and assigned (and the
            texture / wireframe caches were invalidated via the existing
            :attr:`texture` setter).  False when the GLB had no base
            color slot, no embedded image bytes, or PIL isn't available
            to decode them -- in that case this Object's existing
            :attr:`texture` is preserved unchanged.
        """
        textures = _extract_glb_textures(file_path, material_index=material_index)
        color    = textures.get("base_color")
        if color is None:
            return False
        # The texture setter invalidates ``_loaded_texture`` and
        # ``_wired_texture_cache`` -- assigning by attribute is enough.
        self.texture = color
        return True

    def extract_texture_from_fbx(
        self,
        file_path:      str,
        mesh_index:     int = 0,
        material_index: int = 0,
    ) -> bool:
        """Extract the diffuse (base color) texture from an FBX file and
        wire it into this Object's :attr:`texture`.

        Uses :func:`_extract_fbx_textures` internally, which enables FBX
        embedded-media extraction (``IMP_FBX_EXTRACT_EMBEDDED_DATA``) so
        textures packed inside the FBX container resolve to readable
        files on disk.  Also extracts the other standard PBR-ish slots
        (``normal`` / ``specular`` / ``emissive`` / ``occlusion``);
        those are discarded here but the parser is ready for a future
        multi-texture shader upgrade.

        Args:
            file_path: Path to the ``.fbx`` file (``~`` and env vars OK).
            mesh_index: Which mesh node to pull the material from
                (default first).  Negative indices count from the end.
            material_index: Which material on that mesh node to pull
                textures from.  Defaults to the first material.

        Returns:
            True when a diffuse map was found and assigned (and the
            texture / wireframe caches were invalidated via the existing
            :attr:`texture` setter).  False when the FBX had no diffuse
            slot, the connected texture file couldn't be resolved on
            disk, the FBX SDK isn't installed, or PIL isn't available to
            decode the image -- in that case this Object's existing
            :attr:`texture` is preserved unchanged.
        """
        textures = _extract_fbx_textures(
            file_path, mesh_index=mesh_index, material_index=material_index
        )
        color = textures.get("base_color")
        if color is None:
            return False
        self.texture = color
        return True

    # -- mesh / uv / texture: invalidate caches on assignment ----------

    def _mark_render_dirty(self) -> None:
        """Drop the cached render frame so :meth:`imshow` / :meth:`to_image`
        rebuild on next call.  Invoked from setters of attributes that
        affect the rendered output (``mesh`` / ``uv`` / ``texture`` /
        ``base_color`` / ``wireframe*`` / ``sample_method`` / ``wrap`` /
        ``ambient`` / ``twosided`` / ``cast_shadows`` / ``resolution`` /
        ``samples_per_pixel``).  Cheap by design -- never re-renders
        eagerly; the next display call lazy-renders fresh.

        Transform-driven invalidation (``translate`` / ``rotate`` /
        ``scale`` on self or any ancestor) is NOT routed through here
        -- it's handled lazily by the :attr:`frame` getter via the
        transform-state snapshot.
        """
        self._frame                  = None
        self._render_transform_state = None

    @property
    def mesh(self) -> Optional[MeshData]:
        return self._mesh

    @mesh.setter
    def mesh(self, value: Optional[MeshData]) -> None:
        self._mesh = value
        # Wireframe bake depends on mesh edges -> invalidate.
        self._wired_texture_cache = None
        # The bind pose borrows this mesh's topology, so a swap invalidates
        # it.  :meth:`pose` reinstates it straight after -- it is the one
        # caller that knows the topology did not actually change.
        self._bind_mesh = None
        self._mark_render_dirty()

    @property
    def uv(self) -> Optional[UVData]:
        return self._uv

    @uv.setter
    def uv(self, value: Optional[UVData]) -> None:
        self._uv = value
        # UV layout drives where wires get baked -> invalidate both caches.
        self._loaded_texture      = None
        self._wired_texture_cache = None
        self._mark_render_dirty()

    @property
    def texture(self) -> Optional[Union[str, np.ndarray]]:
        return self._texture

    @texture.setter
    def texture(self, value: Optional[Union[str, np.ndarray]]) -> None:
        self._texture = value
        # Diffuse changed -> invalidate both the loaded copy and the wired
        # bake (which composites onto the diffuse).
        self._loaded_texture      = None
        self._wired_texture_cache = None
        self._mark_render_dirty()

    @property
    def skin(self) -> Optional[SkinDeformData]:
        return self._skin

    @skin.setter
    def skin(self, value: Optional[SkinDeformData]) -> None:
        self._skin = value
        # Deliberately NOT _mark_render_dirty(): a deformer changes nothing
        # on screen until :meth:`pose` runs, and that assigns ``mesh``.
        # Drop the bind pose -- it is derived from the deformer.
        self._bind_mesh = None

    def _bind_target(self) -> MeshData:
        """The bind-pose mesh to hand the deformer, built once and cached.

        Topology comes from ``_mesh`` and points from ``_skin.rest_points``,
        rather than from the deformer's own ``_rest_mesh``, because that one
        is a cache and does not survive a save/load -- ``rest_points`` is a
        real field and does.  Rebuilding it also means this stays correct
        when ``_mesh`` currently holds a POSED mesh, which it does from the
        second frame onward.
        """
        if self._bind_mesh is None:
            mesh        = self._mesh.copy()
            mesh.points = np.asarray(self._skin.rest_points, dtype=float).copy()
            if getattr(mesh, "normals", None) is not None:
                mesh.normals = None
            self._bind_mesh = mesh
        return self._bind_mesh

    def pose(self, pose) -> "Object":
        """Deform :attr:`mesh` for a posed skeleton, in place on this Object.

        Every call deforms the BIND pose, never the current :attr:`mesh` --
        feeding the last frame's result back in would compound the
        deformation and melt the geometry over a sequence.

        Args:
            pose: A posed skeleton (:class:`HierarchyData`), or raw world
                matrices shaped ``(J, 4, 4)`` ordered to the deformer's
                joints.  Batched ``(F, J, 4, 4)`` poses are rejected -- an
                Object holds one mesh, not F of them.

        Returns:
            ``self``, so a render can be chained off it.

        Raises:
            ValueError: If no skin is attached, or there is no mesh to deform.
        """
        if self._skin is None:
            raise ValueError(
                f"Object {self.name!r} has no skin to pose -- load it with "
                f"load_skin=True, or assign one to .skin"
            )
        if self._mesh is None:
            raise ValueError(f"Object {self.name!r} has no mesh to deform")

        target    = self._bind_target()
        self.mesh = self._skin.apply(pose, target)
        # The mesh setter just dropped the bind pose; hand it back rather
        # than rebuild it next frame -- we only replaced the points.
        self._bind_mesh = target
        return self

    def restore_bind_pose(self) -> "Object":
        """Put :attr:`mesh` back to the undeformed bind pose.

        The inverse of :meth:`pose`, and what :meth:`Scene.animate` calls when
        it finishes so a render does not leave the scene deformed.  A copy is
        assigned rather than the cached bind mesh itself, so a later
        :meth:`pose` cannot write through :attr:`mesh` into the bind pose.

        Returns:
            ``self``.  A no-op with no skin or no mesh.
        """
        if self._skin is None or self._mesh is None:
            return self

        target          = self._bind_target()
        self.mesh       = target.copy()
        self._bind_mesh = target
        return self

    @property
    def base_color(self) -> Tuple[float, float, float]:
        return self._base_color

    @base_color.setter
    def base_color(self, value: Tuple[float, float, float]) -> None:
        new = tuple(value)
        if new != self._base_color:
            self._base_color = new
            # base_color participates in the wireframe bake when there is
            # no diffuse texture (it fills the buffer before edges are
            # drawn), so invalidate the cache.
            self._wired_texture_cache = None
            self._mark_render_dirty()

    @property
    def background(self) -> Optional[Tuple[float, float, float, float]]:
        """Per-Object preview RGBA background in [0, 1].

        Used only by :meth:`render` / :meth:`to_image` / :meth:`imshow`
        / :meth:`turntable` standalone preview.  IGNORED when this
        Object is rendered via a parent :class:`Scene`
        (``Scene.background`` wins there).

        Accepts:
          - ``None`` -> renderer falls through to ``(0, 0, 0, 0)``
            (transparent black).
          - 3-tuple ``(R, G, B)`` -> auto-promoted to ``(R, G, B, 1)``
            (opaque) for convenience.
          - 4-tuple ``(R, G, B, A)`` -> stored as-is.

        See :class:`Scene.background` for the alpha semantics in the
        renderer (bg.alpha=0 transparent miss, bg.alpha=1 opaque, in
        between yields a Porter-Duff "over" composite).
        """
        return self._background

    @background.setter
    def background(self, value) -> None:
        new = _normalize_rgba(value, "Object.background")
        if new != self._background:
            self._background = new
            self._mark_render_dirty()

    # -- wireframe attributes: setters invalidate the bake cache -------

    @property
    def wireframe(self) -> bool:
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value: bool) -> None:
        new = bool(value)
        if new != self._wireframe:
            self._wireframe           = new
            self._wired_texture_cache = None
            self._mark_render_dirty()

    @property
    def wireframe_color(self) -> Tuple[int, int, int]:
        return self._wireframe_color

    @wireframe_color.setter
    def wireframe_color(self, value: Tuple[int, int, int]) -> None:
        new = tuple(value)
        if new != self._wireframe_color:
            self._wireframe_color     = new
            self._wired_texture_cache = None
            self._mark_render_dirty()

    @property
    def wireframe_thickness(self) -> int:
        return self._wireframe_thickness

    @wireframe_thickness.setter
    def wireframe_thickness(self, value: int) -> None:
        new = int(value)
        if new != self._wireframe_thickness:
            self._wireframe_thickness = new
            self._wired_texture_cache = None
            self._mark_render_dirty()

    # -- shader knobs: setters invalidate the cached frame -------------

    @property
    def sample_method(self) -> str:
        return self._sample_method

    @sample_method.setter
    def sample_method(self, value: str) -> None:
        # Coerce numeric scalars to str (preserves the legacy
        # ``sample_method = 42`` contract that
        # ``test_sample_method_setter_coerces_to_str`` pins down) but
        # preserve everything else (str, enum, mock).  This lets
        # ``_scene_objects_use_bezier`` read ``.name`` as a fallback for
        # enum-shaped values without forcing this leaf module to import
        # the enum class.
        if isinstance(value, str):
            new = value
        elif isinstance(value, (int, float, bool)):
            new = str(value)
        else:
            new = value
        if new != self._sample_method:
            self._sample_method = new
            self._mark_render_dirty()

    @property
    def wrap(self) -> str:
        return self._wrap

    @wrap.setter
    def wrap(self, value: str) -> None:
        new = str(value)
        if new != self._wrap:
            self._wrap = new
            self._mark_render_dirty()

    @property
    def ambient(self) -> float:
        return self._ambient

    @ambient.setter
    def ambient(self, value: float) -> None:
        new = float(value)
        if new != self._ambient:
            self._ambient = new
            self._mark_render_dirty()

    @property
    def twosided(self) -> bool:
        return self._twosided

    @twosided.setter
    def twosided(self, value: bool) -> None:
        new = bool(value)
        if new != self._twosided:
            self._twosided = new
            self._mark_render_dirty()

    @property
    def cast_shadows(self) -> bool:
        return self._cast_shadows

    @cast_shadows.setter
    def cast_shadows(self, value: bool) -> None:
        new = bool(value)
        if new != self._cast_shadows:
            self._cast_shadows = new
            self._mark_render_dirty()

    # -- frame-level render config: setters invalidate the cached frame --

    @property
    def resolution(self) -> Tuple[int, int]:
        """Per-Object preview render resolution ``(width, height)``.

        Used as the default by :meth:`render` / :meth:`turntable` /
        :meth:`imshow` / :meth:`to_image` when no explicit ``resolution``
        kwarg is supplied.  IGNORED when this Object is rendered via a
        parent :class:`Scene` (Scene-level resolution wins there).
        """
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        new = tuple(int(v) for v in value)
        if new != self._resolution:
            self._resolution = new
            self._mark_render_dirty()

    @property
    def samples_per_pixel(self) -> Optional[int]:
        """Per-Object preview MSAA sample count.

        Used as the default by :meth:`render` / :meth:`turntable` /
        :meth:`imshow` / :meth:`to_image` when no explicit
        ``samples_per_pixel`` kwarg is supplied.  Default is ``4``
        (2x2 MSAA -- the smallest perfect square greater than 1).
        Set to ``None`` to defer to the renderer's own internal
        default.  IGNORED when this Object is rendered via a parent
        :class:`Scene`.

        The renderer requires the value to be ``1`` or a perfect
        square (``1, 4, 9, 16, ...``) because it lays samples out on
        a regular ``n_axis x n_axis`` sub-pixel grid.  Invalid values
        are rejected here at assignment time -- not deferred to the
        next render call -- so typos surface immediately.
        """
        return self._samples_per_pixel

    @samples_per_pixel.setter
    def samples_per_pixel(self, value: Optional[int]) -> None:
        new = _validate_samples_per_pixel(value, field_name="Object.samples_per_pixel")
        if new != self._samples_per_pixel:
            self._samples_per_pixel = new
            self._mark_render_dirty()

    # -- frame cache: lazy invalidation via transform-state snapshot ---

    def _capture_transform_state(self) -> tuple:
        """Snapshot of the local SRT values + composed world_matrix.

        The local SRT tuples (``_translate`` / ``_rotate`` / ``_scale``)
        are read from the dataclass storage directly, so they're always
        current -- they don't depend on ``TransformData``'s
        ``_world_matrix`` cache (which has a known lag bug for direct
        local mutations on standalone, parent-less nodes: the rotate /
        translate / scale setter calls ``_reset_branch_world_matrices``
        BEFORE assigning the new private value, so anything that re-
        reads ``world_matrix`` between the reset and the assignment
        will repopulate the cache with stale data).

        Including the composed ``world_matrix`` bytes catches ancestor
        mutations that propagate through ``_reset_branch_world_matrices``
        and DO update the child's ``world_matrix`` correctly (because
        the child's own SRT hasn't changed but its world position has).
        """
        return (
            tuple(np.asarray(self._translate, dtype=float).ravel()),
            tuple(np.asarray(self._rotate, dtype=float).ravel()),
            tuple(np.asarray(self._scale, dtype=float).ravel()),
            np.asarray(self.world_matrix).tobytes(),
        )

    @property
    def frame(self) -> Optional["Frame"]:
        """Cached most-recent render output, or ``None`` when the cache
        is invalid.  Auto-invalidates when:

        - any tracked visual attribute is reassigned (``mesh`` / ``uv``
          / ``texture`` / ``base_color`` / ``wireframe*`` /
          ``sample_method`` / ``wrap`` / ``ambient`` / ``twosided`` /
          ``cast_shadows`` / ``resolution`` / ``samples_per_pixel``)
          -- via the existing setter -> :meth:`_mark_render_dirty`
          plumbing, OR
        - the Object's local transform values OR composed
          ``world_matrix`` have changed since the cached frame was
          rendered, i.e., ``translate`` / ``rotate`` / ``scale`` on self
          OR any ancestor was mutated.  Detected lazily here by
          comparing the live transform-state to the snapshot taken in
          the setter below.

        In-place mutation of ``mesh.points`` / texture pixel arrays
        still requires an explicit :meth:`render` call to refresh.
        """
        if self._frame is None:
            return None
        if (
            self._render_transform_state is not None
            and self._capture_transform_state() != self._render_transform_state
        ):
            self._frame                  = None
            self._render_transform_state = None
            return None
        return self._frame

    @frame.setter
    def frame(self, value: Optional["Frame"]) -> None:
        self._frame = value
        if value is None:
            self._render_transform_state = None
        else:
            # Snapshot the transform state at cache-population time so
            # the getter can detect later transform mutations.
            self._render_transform_state = self._capture_transform_state()

    @property
    def buffer(self) -> Optional[np.ndarray]:
        """Convenience accessor for ``self.frame.array`` -- the pixel
        buffer of the most recent render (``None`` until rendered).
        Matches the ``buffer`` vocabulary used by :class:`UVData`.
        """
        return self.frame.array if self.frame is not None else None

    def get_loaded_texture(self) -> Optional[np.ndarray]:
        """Returns the texture as a contiguous ``(H, W, 3) float32`` array,
        loading from disk on first call.  Returns ``None`` if no texture
        is set AND the wireframe bake is not enabled.

        When :attr:`wireframe` is True AND :attr:`uv` is non-None, returns
        a copy of the diffuse with the mesh wireframe baked on top via
        :meth:`UVData.draw_edges` (cached after first build; invalidated
        when ``texture`` / ``uv`` / ``base_color`` / ``wireframe*`` change).
        When :attr:`texture` is None in this mode the bake fills the
        buffer with :attr:`base_color` first, so wireframe-only Objects
        render as edges over their flat-shading color.
        """
        # Wireframe takes precedence: an Object can be wireframe-only
        # (no diffuse texture) -- the bake fills the buffer with
        # base_color before drawing edges.
        if self._wireframe and self._uv is not None:
            if self._wired_texture_cache is None:
                self._wired_texture_cache = self._bake_wired_texture()
            return self._wired_texture_cache
        if self._texture is None:
            return None
        if self._loaded_texture is not None:
            return self._loaded_texture
        self._loaded_texture = _load_texture(self._texture)
        return self._loaded_texture

    def _bake_wired_texture(self) -> np.ndarray:
        """Bake the mesh wireframe (in UV space) onto a copy of the
        diffuse and return the wired texture as float32 ``(H, W, 3)``
        in [0, 1] (matches the format ``_load_texture`` returns).

        Mirrors the 4-step recipe documented in render_test.py:
          1. Load the diffuse into the UV render buffer.  When
             :attr:`texture` is None, fill the buffer with
             :attr:`base_color` (uint8) instead -- buffer dimensions
             come from the existing ``UVData._resolution`` (default
             2048x2048 when no buffer has been set yet).
          2. Turn on anti-aliasing for nicer line edges.
          3. Rasterise the unique mesh edges via ``UVData.draw_edges``.
          4. Snapshot + flip the buffer (UVData is bottom-up; renderer
             wants top-down) and convert to float32 [0, 1].
        """
        if self._uv is None:
            raise RuntimeError(
                "Object._bake_wired_texture: cannot bake wireframe without a UV"
            )
        self._uv.antialias = True
        # Resolve texture to a uint8 (H, W, 3) numpy array.  We load file
        # paths via PIL (matching ``_load_texture``) instead of routing
        # through ``UVData.buffer_from_file``, which requires cv2 or a full
        # skimage install (transitively pulling in tifffile).
        if self._texture is None:
            # No diffuse: fill the UV buffer with the per-Object
            # base_color (matches the renderer's flat-shading fallback
            # for textureless Objects).  Reading ``self._uv.buffer``
            # lazily allocates the buffer at the UV's current
            # ``_resolution`` if none was set, giving us H/W to match.
            # No flip needed -- a uniform colour is direction-agnostic.
            existing = self._uv.buffer
            h, w = existing.shape[:2]
            bg_uint8 = (
                (np.asarray(self._base_color, dtype=np.float64) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            self._uv.buffer = np.broadcast_to(bg_uint8, (h, w, 3)).copy()
        else:
            if isinstance(self._texture, str):
                if Image is None:
                    raise RuntimeError(
                        "PIL required to bake wireframe from a texture path; "
                        "pass a numpy array instead, or install Pillow."
                    )
                path = os.path.expandvars(os.path.expanduser(self._texture))
                with Image.open(path) as img:
                    tex_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
            else:
                tex_uint8 = np.asarray(self._texture, dtype=np.uint8)
            # UVData buffer is bottom-up; flip top-down -> bottom-up.
            self._uv.buffer = np.flipud(tex_uint8).copy()
        self._uv.draw_edges(
            color     = self._wireframe_color,
            thickness = self._wireframe_thickness,
        )
        wired_uint8 = np.flipud(self._uv.buffer).copy()
        # Match _load_texture's output format: float32 [0, 1], (H, W, 3).
        return wired_uint8.astype(np.float32) / 255.0

    @property
    def world_points(self) -> np.ndarray:
        """Mesh points transformed to world space via :attr:`world_matrix`.

        TransformData stores transforms in row-major / Maya convention.
        For a row-vector ``p`` and row-major ``M``, world = ``p_row @ M``
        which is equivalent to extracting ``M[:3, :3]`` (rows are basis
        vectors) and ``M[3, :3]`` (translation).
        """
        if self.mesh is None:
            return np.empty((0, 3), dtype=np.float64)
        pts = np.asarray(self.mesh.points, dtype=np.float64)
        if pts.shape[1] != 3:
            return pts
        M = self.world_matrix
        # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64).
        with np.errstate(all="ignore"):
            return pts @ M[:3, :3] + M[3, :3]

    # -- topology ops --------------------------------------------------

    def quadrangulate(self, source: str = "uv") -> None:
        """Merges pairs of triangles into quads, in-place, on this
        Object's mesh and (when present) its UV.

        Convenience wrapper around :meth:`MeshData.quadrangulate` /
        :meth:`UVData.quadrangulate` that keeps the mesh and UV in sync:
        a single :class:`~numpy.ndarray` of merge rules is generated once
        and applied to both.  This is safe because the rule format is
        purely face-local (``(fa, fb, oa_slot, ob_slot)`` tuples), so
        the same rules apply to a :class:`MeshData` and to its
        corresponding :class:`UVData` even though they may have
        different vertex indices due to UV seams.

        Args:
            source: When this Object has a UV (``self.uv is not None``),
                controls which topology drives rule generation:

                * ``"uv"`` (default) -- rules from
                  :meth:`UVData.get_quadrangulate_rules`.  Useful when
                  the UV layout's topology should dictate the merge
                  pattern (e.g. preserves UV-island shapes).
                * ``"mesh"`` -- rules from
                  :meth:`MeshData.get_quadrangulate_rules`.  Useful when
                  the 3D mesh quality (planarity, squareness) should
                  dictate the merge pattern.

                In both cases the resulting rules are applied to BOTH
                the mesh and the UV so they stay in sync.

                When ``self.uv is None`` this argument is ignored and
                rules are always taken from the mesh.

        Raises:
            ValueError: If ``self.mesh`` is None, or if ``source`` is
                not ``"uv"`` or ``"mesh"``.
        """
        if self.mesh is None:
            raise ValueError("Object.quadrangulate: self.mesh is None")
        if source not in ("uv", "mesh"):
            raise ValueError(
                f"Object.quadrangulate: source must be 'uv' or 'mesh', got {source!r}"
            )

        if self.uv is None:
            rules = self.mesh.get_quadrangulate_rules()
            self.mesh.quadrangulate(rules)
        else:
            if source == "uv":
                rules = self.uv.get_quadrangulate_rules()
            else:
                rules = self.mesh.get_quadrangulate_rules()
            self.uv.quadrangulate(rules)
            self.mesh.quadrangulate(rules)

        # Mesh / UV topology changed in place (the setters did NOT fire),
        # so manually invalidate the wireframe bake which depends on
        # mesh edges and UV layout.
        self._wired_texture_cache = None
        self._mark_render_dirty()

    def triangulate(self, ngons_only=False) -> None:
        """Triangulates polygons (quads / n-gons) into triangles, in-place,
        on this Object's mesh and (when present) its UV.

        Convenience wrapper around :meth:`MeshData.triangulate` /
        :meth:`UVData.triangulate` that keeps the mesh and UV in sync:
        a single :class:`~numpy.ndarray` of triangulation rules is
        generated once via :meth:`MeshData.get_triangulate_rules` on
        ``self.mesh`` and applied to both.  This is safe because the
        rule format is purely face-local, so the same rules apply to
        a :class:`MeshData` and to its corresponding :class:`UVData`
        even though they may have different vertex indices due to
        UV seams.

        Always uses the mesh's topology to drive rule generation
        (unlike :meth:`quadrangulate`, which can be driven by either
        mesh or UV topology) -- triangulation is local enough that
        the choice doesn't change the resulting partition.

        Raises:
            ValueError: If ``self.mesh`` is None.
        """
        if self.mesh is None:
            raise ValueError("Object.triangulate: self.mesh is None")

        rules = self.mesh.get_triangulate_rules(ngons_only=ngons_only)
        self.mesh.triangulate(rules)
        if self.uv is not None:
            self.uv.triangulate(rules)

        # Mesh / UV topology changed in place (the setters did NOT fire),
        # so manually invalidate the wireframe bake (depends on mesh edges
        # and UV layout) and the render cache.
        self._wired_texture_cache = None
        self._mark_render_dirty()

    def merge(self) -> None:
        """Merges overlapping points (and resulting overlapping faces),
        in-place, on this Object's mesh and (when present) its UV.

        Thin convenience wrapper that calls :meth:`MeshData.merge` and
        :meth:`UVData.merge` so callers don't have to remember to merge
        the UV separately.  Each underlying ``merge()`` deduplicates its
        own ``points`` array independently -- the mesh merges in 3D and
        the UV merges in 2D -- so the two are NOT guaranteed to share
        face counts after the call (e.g. the mesh may merge a seam
        across UV islands while the UV keeps the seam's two distinct
        UV-vertices).

        Raises:
            ValueError: If ``self.mesh`` is None.
        """
        if self.mesh is None:
            raise ValueError("Object.merge: self.mesh is None")
        self.mesh.merge()
        if self.uv is not None:
            self.uv.merge()
        # Mesh / UV topology changed in place (the setters did NOT fire),
        # so manually invalidate the wireframe bake which depends on
        # mesh edges and UV layout.
        self._wired_texture_cache = None
        self._mark_render_dirty()

    # -- one-call preview / turntable shims ----------------------------

    def render(
        self,
        output:            Optional[str]             = None,
        resolution:        Optional[Tuple[int, int]] = None,
        samples_per_pixel: Optional[int]             = None,
        **render_kwargs,
    ) -> "Frame":
        """Quick standalone preview render of this Object.

        Builds a one-Object scene with a 3-point default lighting rig
        sized to the Object's AABB, runs the renderer, and (optionally)
        saves the image to disk.  Default ``resolution`` falls back to
        :attr:`resolution` (500x500 unless overridden) which is
        convenient for asset spreadsheets / asset thumbnails.

        For more control (multi-Object scenes, custom lighting, specific
        camera placement) build a :class:`Scene` directly.

        Args:
            output: If given, save the rendered :class:`Frame` to this
                path.  Path supports ``~`` and env vars.  When ``None``,
                only the Frame is returned.
            resolution: ``(width, height)`` in pixels.  When ``None``
                (default), falls back to :attr:`resolution`.
            samples_per_pixel: MSAA sample count for this preview render.
                When ``None`` (default), falls back to
                :attr:`samples_per_pixel` (which itself defaults to ``2``).
            **render_kwargs: Forwarded to :func:`render`.

        Returns:
            The rendered :class:`Frame` (saved to ``output`` if given).

        Note:
            If this Object is parented to a HierarchyData, only its OWN
            world_matrix snapshot is used in the preview scene -- the
            parent transform is NOT applied.
        """
        if resolution is None:
            resolution = self.resolution
        if samples_per_pixel is None:
            samples_per_pixel = self.samples_per_pixel
        scene = Scene(f"preview:{self.name}")
        scene.append(self)
        # Forward this Object's preview background (if any) to the
        # one-Object Scene so obj.to_image() respects it.
        if self._background is not None:
            scene.background = self._background
        # Lighting is auto-injected by Scene.render()'s default-rig path
        # (sized to the scene's union AABB), so a one-Object preview
        # here gets the same 3-point rig as scene.to_image().
        if samples_per_pixel is not None:
            render_kwargs.setdefault("samples_per_pixel", samples_per_pixel)
        frame = scene.render(resolution=resolution, **render_kwargs)
        if output is not None:
            frame.save(output)
        # Cache the rendered Frame so callers can :meth:`imshow` it later
        # without re-rendering.
        self.frame = frame
        return frame

    def to_image(self) -> "Image":
        """Returns the rendered frame as a PIL Image.

        Mirrors :meth:`cgmath.geometry.mesh.UVData.to_image`.  Lazy-renders
        with this Object's :meth:`render` defaults (500x500 + a 3-point
        lighting rig) when no frame is cached yet, so ``obj.to_image()`` is
        a single-call "give me a PIL image" shortcut suitable for
        notebooks, tests, and image pipelines.

        Raises:
            RuntimeError: If PIL is not available.

        Note:
            Unlike :meth:`UVData.to_image` (which flips its bottom-up
            buffer), no flip is needed here because ``Frame.array`` is
            already top-down RGB uint8.

            Mutating tracked visual attributes (``mesh`` / ``uv`` /
            ``texture`` / ``base_color`` / ``wireframe*`` /
            ``sample_method`` / ``wrap`` / ``ambient`` / ``twosided`` /
            ``cast_shadows`` / ``resolution`` / ``samples_per_pixel``)
            auto-invalidates the cached frame, AND mutating the Object's
            transform (``translate`` / ``rotate`` / ``scale`` on self OR
            any ancestor) ALSO auto-invalidates via the lazy transform-
            state check on :attr:`frame`.  In-place mutation of
            ``mesh.points`` / texture pixels still requires an explicit
            :meth:`render` call to refresh.
        """
        if Image is None:
            raise RuntimeError("PIL required to generate PIL.Image data")
        if self.frame is None:
            self.render()  # uses defaults; sets ``self.frame``
        return Image.fromarray(self.frame.array)

    def imshow(self) -> None:
        """Show the rendered frame in a window via OpenCV.

        Mirrors :meth:`cgmath.geometry.mesh.UVData.imshow`.  Lazy-renders
        with this Object's :meth:`render` defaults (500x500 + a 3-point
        lighting rig) when no frame is cached yet, so ``obj.imshow()`` is a
        single-call "show me the asset" shortcut.

        Raises:
            RuntimeError: If OpenCV is not available (it's required to
                show the image in a window).

        Note:
            Mutating tracked visual attributes (``mesh`` / ``uv`` /
            ``texture`` / ``base_color`` / ``wireframe*`` /
            ``sample_method`` / ``wrap`` / ``ambient`` / ``twosided`` /
            ``cast_shadows`` / ``resolution`` / ``samples_per_pixel``)
            auto-invalidates the cached frame, AND mutating the Object's
            transform (``translate`` / ``rotate`` / ``scale`` on self OR
            any ancestor) ALSO auto-invalidates via the lazy transform-
            state check on :attr:`frame`.  In-place mutation of
            ``mesh.points`` / texture pixels still requires an explicit
            :meth:`render` call to refresh.
        """
        if cv2 is None:
            raise RuntimeError("opencv required to show images")
        if self.frame is None:
            self.render()  # uses defaults; sets ``self.frame``
        # Frame.array is top-down RGB; cv2 wants BGR.  No vertical flip
        # needed (UVData flips because its buffer is bottom-up).
        buffer = cv2.cvtColor(self.frame.array, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.name, buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def turntable(
        self,
        output:            str                       = "turntable.{frame:04d}.png",
        n_frames:          int                       = 120,
        resolution:        Optional[Tuple[int, int]] = None,
        samples_per_pixel: Optional[int]             = None,
        fps:               int                       = 30,
        **render_kwargs,
    ) -> Union[str, List[str]]:
        """Quick standalone turntable of this Object.

        Same convenience as :meth:`render`: builds a one-Object scene
        with a default 3-point lighting rig and orbits the camera.
        Output extension picks the encoder (``.mp4`` / ``.gif`` / image
        sequence) -- see :meth:`Scene.turntable`.

        Default ``resolution`` falls back to :attr:`resolution`
        (500x500 unless overridden) -- useful for asset spreadsheet
        thumbnails or per-asset turntable tiles.

        Args:
            output: Output file path (or ``{frame}`` pattern for image
                sequences).  Defaults to ``"turntable.{frame:04d}.png"``
                in the current directory.
            n_frames: Number of frames in the loop.  Default 120.
            resolution: ``(width, height)`` in pixels.  When ``None``
                (default), falls back to :attr:`resolution`.
            samples_per_pixel: MSAA sample count for this turntable.
                When ``None`` (default), falls back to
                :attr:`samples_per_pixel`.
            fps: Frames per second for video / gif outputs.  Default 30.
            **render_kwargs: Forwarded to :func:`render`.

        Returns:
            Single output path (``str``) for video / gif, or a list of
            frame paths (``list[str]``) for image sequences.

        Note:
            If this Object is parented to a HierarchyData, only its OWN
            world_matrix snapshot is used in the preview scene -- the
            parent transform is NOT applied.
        """
        if resolution is None:
            resolution = self.resolution
        if samples_per_pixel is None:
            samples_per_pixel = self.samples_per_pixel
        scene = Scene(f"preview:{self.name}")
        scene.append(self)
        # Forward this Object's preview background (if any) so per-frame
        # turntable renders + the encoder dispatch (mp4 bg flatten) see
        # the user-chosen colour.
        if self._background is not None:
            scene.background = self._background
        # Lighting is auto-injected by Scene.render() per frame (sized
        # to the scene's union AABB) -- same rig as Object.render().
        if samples_per_pixel is not None:
            render_kwargs.setdefault("samples_per_pixel", samples_per_pixel)
        return scene.turntable(
            output_pattern = output,
            n_frames       = n_frames,
            resolution     = resolution,
            fps            = fps,
            **render_kwargs,
        )

    def animate(
        self,
        clip,
        output:            str                       = "animation.{frame:04d}.png",
        resolution:        Optional[Tuple[int, int]] = None,
        samples_per_pixel: Optional[int]             = None,
        fps:               Optional[int]             = None,
        **animate_kwargs,
    ) -> Union[str, List[str]]:
        """Quick standalone render of this Object animated by *clip*.

        Same convenience as :meth:`turntable`: builds a one-Object scene with
        a default lighting rig and hands off to :meth:`Scene.animate`, which
        is where the arguments and the output-format rules are documented.

        Args:
            clip: A :class:`ClipData` holding the animation.
            output: Output file path, or a ``{frame}`` pattern for sequences.
            resolution: ``(width, height)``.  Defaults to :attr:`resolution`.
            samples_per_pixel: MSAA count.  Defaults to
                :attr:`samples_per_pixel`.
            fps: Frames per second.  Defaults to the clip's own ``fps``.
            **animate_kwargs: Forwarded to :meth:`Scene.animate` (``fit``,
                ``end_angle``, ``verbose``, ...).

        Returns:
            A single output path for video / gif, or the list of frame paths.

        Raises:
            ValueError: If this Object has no skin to animate.

        Note:
            As with :meth:`turntable`, only this Object's OWN world_matrix is
            used in the preview scene -- a parent transform is NOT applied.
        """
        if self._skin is None:
            raise ValueError(
                f"Object {self.name!r} has no skin to animate -- load it with "
                f"load_skin=True, or assign one to .skin"
            )
        if resolution is None:
            resolution = self.resolution
        if samples_per_pixel is None:
            samples_per_pixel = self.samples_per_pixel

        scene = Scene(f"preview:{self.name}")
        scene.append(self)
        if self._background is not None:
            scene.background = self._background
        if samples_per_pixel is not None:
            animate_kwargs.setdefault("samples_per_pixel", samples_per_pixel)

        return scene.animate(
            clip,
            output_pattern = output,
            resolution     = resolution,
            fps            = fps,
            **animate_kwargs,
        )

    # -- serialization: round-trip mesh / uv / texture -------------

    def to_dict(self) -> dict:
        """Serializable dict for :meth:`save` (npz / json / pkl).

        Overrides :meth:`Data.to_dict` to handle the three Object-
        specific fields the base mechanism can't flatten on its own:

          - ``_mesh`` (:class:`MeshData`) -> nested dict via the
            child's own ``to_dict()`` (recursive).
          - ``_uv`` (:class:`UVData`) -> same.
          - ``_skin`` (:class:`SkinDeformData`) -> same.  Nesting it by
            hand is not optional: the base mechanism only writes a field
            whose value differs from the dataclass default, and ``Data``
            compares equal to a foreign type, so the check would drop it.
          - ``_texture`` (``str`` path | ``np.ndarray`` | None) ->
            materialized to a ``(H, W, C) uint8`` numpy array so the
            saved file is self-contained (path-loaded textures are
            read off disk now, not at load time).

        Transient render caches (``_loaded_texture``,
        ``_wired_texture_cache``, ``_render_transform_state``) are
        scrubbed so a render-then-save doesn't bloat the output.
        """
        data = super().to_dict()

        if self._mesh is not None:
            data["_mesh"] = self._mesh.to_dict()
        if self._uv is not None:
            data["_uv"] = self._uv.to_dict()
        if self._skin is not None:
            data["_skin"] = self._skin.to_dict()

        if self._texture is not None:
            if isinstance(self._texture, str):
                data["_texture"] = _load_texture_path_to_uint8(self._texture)
            else:
                arr = np.asarray(self._texture)
                if arr.dtype != np.uint8:
                    if arr.dtype.kind == "f":
                        arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                data["_texture"] = arr

        # Drop transient render caches in case base to_dict picked them up
        # (they're declared as Optional dataclass fields but are caches).
        for k in (
            "_loaded_texture",
            "_wired_texture_cache",
            "_render_transform_state",
            "_bind_mesh",
        ):
            data.pop(k, None)

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Object":
        """Reconstruct an Object from :meth:`to_dict`'s output.

        Nested ``_mesh`` / ``_uv`` dicts are rebuilt via
        :meth:`MeshData.from_dict` / :meth:`UVData.from_dict`.  The
        texture (always restored as a numpy uint8 array) is assigned
        to ``_texture`` directly -- the renderer's
        :meth:`get_loaded_texture` accepts ndarrays so no temp file
        is needed.  If you need a file path, write the array to disk
        yourself after loading.
        """
        # Copy so we don't mutate the caller's dict.
        data      = dict(data)
        mesh_dict = data.pop("_mesh",    None)
        uv_dict   = data.pop("_uv",      None)
        tex_arr   = data.pop("_texture", None)
        skin_dict = data.pop("_skin",    None)

        obj = super().from_dict(data)

        if mesh_dict is not None:
            obj._mesh = MeshData.from_dict(mesh_dict)
        if uv_dict is not None:
            obj._uv = UVData.from_dict(uv_dict)
        if tex_arr is not None:
            obj._texture = np.asarray(tex_arr)
        if skin_dict is not None:
            obj._skin = SkinDeformData.from_dict(skin_dict)

        # Initialize non-serialized caches (also defends against pickle
        # paths that bypass __init__).
        obj._loaded_texture         = None
        obj._wired_texture_cache    = None
        obj._frame                  = None
        obj._render_transform_state = None
        obj._bind_mesh              = None

        return obj

    def __reduce__(self):
        """Pickle round-trip through :meth:`to_dict` / :meth:`from_dict`
        so nested :class:`MeshData` / :class:`UVData` are properly
        reconstructed.  The inherited :meth:`Data.__reduce__` uses
        ``_reconstruct`` which calls ``__dict__.update(state)`` --
        that would leave ``_mesh`` / ``_uv`` as raw dicts and skip
        our cache reset.
        """
        return (self.__class__.from_dict, (self.to_dict(),))


# -- Camera -------------------------------------------------------------------------


@dataclass(repr=False, eq=False)
class Camera(TransformData):
    """Maya/OpenGL-style camera: looks down its own -Z axis.

    Uses :attr:`world_matrix` (inherited from TransformData) for
    position + orientation.  Domain knobs:

    Attributes:
        angle_of_view: Vertical full FOV in degrees (perspective only).
        is_orthographic: When True, emits parallel rays.
        ortho_height: World-space frustum height when orthographic.
            When ``None``, derived at render time from the scene AABB.
        near_plane / far_plane: Reserved for future clipping; unused in v1.
        aspect_ratio: When ``None``, derived from render() resolution.
    """

    angle_of_view:   float = 35.0
    is_orthographic: bool = False
    ortho_height:    Optional[float] = None
    near_plane:      float = 0.0
    far_plane:       float = float("inf")
    aspect_ratio:    Optional[float] = None

    def __init__(
        self,
        name:            str             = "camera",
        angle_of_view:   float           = 35.0,
        is_orthographic: bool            = False,
        ortho_height:    Optional[float] = None,
        near_plane:      float           = 0.0,
        far_plane:       float           = float("inf"),
        aspect_ratio:    Optional[float] = None,
        **transform_kwargs,
    ) -> None:
        super().__init__(name=name, node_type="camera", **transform_kwargs)
        self.angle_of_view   = float(angle_of_view)
        self.is_orthographic = bool(is_orthographic)
        self.ortho_height    = ortho_height
        self.near_plane      = float(near_plane)
        self.far_plane       = float(far_plane)
        self.aspect_ratio    = aspect_ratio

    def look_at(
        self,
        target: Union[Tuple[float, float, float], np.ndarray],
        up:     Tuple[float, float, float]                    = (0.0, 1.0, 0.0),
    ) -> None:
        """Aim this camera at ``target`` from its current world position.

        Preserves the camera's world-space translation and overwrites the
        rotation so its -Z axis points at ``target``.

        Internally bridges the renderer's column-major / OpenGL matrix
        convention with TransformData's row-major / Maya convention by
        transposing.
        """
        # Read the eye position from the row-major TransformData world matrix.
        eye = self.world_matrix[3, :3]
        opengl_world = _look_at(
            tuple(eye), tuple(np.asarray(target).ravel()), tuple(up)
        )
        # Transpose into row-major form before storing.
        self.world_matrix = opengl_world.T

    def autofit(
        self,
        points: np.ndarray,
        width:  int,
        height: int,
    ) -> None:
        """Override this camera's position so that ``points`` fit the frustum.

        Preserves orientation; uses the shift/slide/center algorithm.
        Only valid for cameras at world root - parented cameras raise.
        """
        if self.get_parent() is not None:
            raise ValueError(
                "Camera.autofit only works for cameras parented to world root"
            )

        # Convert row-major world_matrix to column-major OpenGL form for
        # the autofit helper, then transpose the result back to row-major.
        opengl_world_in = self.world_matrix.T
        opengl_world_out = _autofit_camera_for_points(
            np.asarray(points, dtype=np.float64),
            opengl_world_in,
            float(self.angle_of_view),
            int(width),
            int(height),
        )
        self.world_matrix = opengl_world_out.T

    @classmethod
    def default_view(
        cls,
        points:        np.ndarray,
        name:          str        = "camera",
        angle_of_view: float      = 35.0,
    ) -> "Camera":
        """Build a 3/4 elevated view Camera looking at the points' bbox center."""
        cam          = cls(name=name, angle_of_view=angle_of_view)
        opengl_world = _default_camera_for_points(np.asarray(points, dtype=np.float64))
        # Transpose into row-major form for TransformData.
        cam.world_matrix = opengl_world.T
        return cam


# -- Light ---------------------------------------------------------------------


@dataclass(repr=False, eq=False)
class Light(TransformData):
    """A scene light.

    ``kind="point"`` uses :attr:`world_matrix` translation as the light
    position with 1/r^2 falloff.  ``kind="infinite"`` uses the camera-style
    forward axis (``-world_matrix[:3, 2]``) as the parallel-ray direction
    with no falloff (useful as a "sun" light).
    """

    kind:      str = "point"  # "point" or "infinite"
    color:     Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    falloff:   bool = True    # only meaningful for kind=="point"

    def __init__(
        self,
        name:      str                        = "light",
        kind:      str                        = "point",
        color:     Tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float                      = 1.0,
        falloff:   bool                       = True,
        **transform_kwargs,
    ) -> None:
        super().__init__(name=name, node_type="light", **transform_kwargs)
        if kind not in ("point", "infinite"):
            raise ValueError(f"Light.kind must be 'point' or 'infinite', got {kind!r}")
        self.kind      = kind
        self.color     = tuple(color)
        self.intensity = float(intensity)
        self.falloff   = bool(falloff)

    @property
    def position(self) -> np.ndarray:
        """World-space position derived from :attr:`world_matrix`.

        TransformData stores transforms in row-major / Maya convention,
        so the translation lives in ``world_matrix[3, :3]``.
        """
        return np.asarray(self.world_matrix[3, :3], dtype=np.float64)

    @property
    def direction(self) -> np.ndarray:
        """World-space forward direction (-Z axis) for infinite lights.

        In row-major TransformData convention, the matrix's rows are
        the basis vectors, so the local -Z axis maps to row 2 (negated).
        """
        d = -np.asarray(self.world_matrix[2, :3], dtype=np.float64)
        n = float(np.linalg.norm(d))
        return d / max(n, 1e-12)

    def as_dict(self) -> dict:
        """Returns a dict matching the renderer's existing point_light shape."""
        return {
            "kind":      self.kind,
            "position":  tuple(self.position),
            "direction": tuple(self.direction),
            "color":     tuple(self.color),
            "intensity": float(self.intensity),
            "falloff":   bool(self.falloff),
        }


# -- Scene ---------------------------------------------------------------------


class Scene(HierarchyData):
    """A hierarchical container of Objects, Cameras and Lights.

    Inherits all of :class:`HierarchyData`'s list-like + uuid lookup +
    parenting machinery.  Validates that appended nodes are renderer
    primitives (or plain ``TransformData`` group nodes).

    Render cache (``frame`` / ``buffer``):
      :meth:`render`, :meth:`imshow`, and :meth:`to_image` populate a
      :attr:`frame` cache so repeated display calls don't re-run the
      raytracer.  The cache auto-invalidates lazily on access when ANY
      render input on this Scene OR any of its child Object / Camera /
      Light nodes has changed since the last :meth:`render` call --
      including:

        - transforms (``translate`` / ``rotate`` / ``scale``) on any
          node, including ancestors of nodes inside the Scene,
        - Camera domain attrs (``angle_of_view``, ``is_orthographic``,
          ``ortho_height``, ``near_plane``, ``far_plane``,
          ``aspect_ratio``),
        - Light domain attrs (``kind``, ``color``, ``intensity``,
          ``falloff``),
        - Object visual attrs (``mesh`` / ``uv`` / ``texture`` /
          ``base_color`` / ``wireframe*`` / ``visibility`` /
          shader knobs),
        - scene topology (nodes added or removed),
        - Scene-level render config (every key in
          :attr:`_RENDER_CONFIG_KEYS`: ``resolution``,
          ``samples_per_pixel``, ``background``, ``autofit``,
          ``default_light``, ``return_depth``, ``angle_of_view``,
          ``fit_padding``) plus ``aspect_ratio`` and
          ``default_camera_name``.

      In-place mutation of ``mesh.points`` / ``uv.points`` / texture
      pixel arrays is the only common case that does NOT trigger
      auto-invalidation -- call :meth:`render` again to refresh.
      See :meth:`_compute_render_signature` for the full per-node
      signature spec.
    """

    def __init__(
        self,
        name:                str                                             = "scene",
        aspect_ratio:        Optional[float]                                 = None,
        default_camera_name: Optional[str]                                   = None,
        resolution:          Optional[Tuple[int, int]]                       = (500, 500),
        samples_per_pixel:   Optional[int]                                   = None,
        background:          Optional[Union[Tuple[float, ...], List[float]]] = None,
        autofit:             Optional[bool]                                  = None,
        default_light:       Optional[bool]                                  = None,
        return_depth:        Optional[bool]                                  = None,
        angle_of_view:       Optional[float]                                 = None,
        fit_padding:         Optional[float]                                 = None,
    ) -> None:
        super().__init__()
        # NB: HierarchyData inherits a ``name`` property from TransformList
        # that returns the list of child names, with no setter.  We store
        # the scene's own label under a different attribute name.
        self._scene_name         = str(name) if name is not None else None
        self.aspect_ratio        = aspect_ratio
        self.default_camera_name = default_camera_name
        # Cached most-recent render output, accessed via the ``frame``
        # property below.  Use private storage so the property setter
        # can snapshot a full render-input signature for lazy
        # invalidation when any child node attribute changes.
        self._frame:            Optional["Frame"] = None
        self._render_signature: Optional[tuple] = None
        # Persistent render-config attrs.  Each ``None`` means "fall through
        # to the per-call kwarg or render()'s internal default".  Per-call
        # kwargs always win; these are the Scene-level defaults that flow
        # through render() and turntable() without being repeated.
        self.resolution:        Optional[Tuple[int, int]] = resolution
        self.samples_per_pixel: Optional[int] = samples_per_pixel
        # ``background`` is normalized to RGBA via the property setter
        # below; users may pass RGB or RGBA (or None) and it always
        # round-trips to a 4-tuple in storage.
        self._background: Optional[Tuple[float, float, float, float]] = _normalize_rgba(
            background, "Scene.background"
        )
        self.autofit:       Optional[bool] = autofit
        self.default_light: Optional[bool] = default_light
        self.return_depth:  Optional[bool] = return_depth
        self.angle_of_view: Optional[float] = angle_of_view
        # Safety margin for autofit (>= 1.0).  ``None`` lets the renderer
        # auto-pick (1.15 for bezier-sampled meshes, 1.0 otherwise).
        self.fit_padding: Optional[float] = fit_padding

    # -- file IO: build a Scene from disk -----------------------------

    @classmethod
    def load_obj(cls, file_path: str, name: Optional[str] = None) -> "Scene":
        """Load every mesh in an OBJ file into a new :class:`Scene` as
        :class:`Object` nodes.

        Convenience wrapper around :func:`cgmath.geometry.mesh.load_obj`.
        Each mesh in the file becomes one Object in the returned Scene.
        Lights and cameras are NOT added -- :meth:`render` / :meth:`imshow`
        will lazy-build a default lighting rig and camera if none are
        present.

        Args:
            file_path: Path to the ``.obj`` file (``~`` and env vars OK).
            name: Optional name for the Scene (defaults to ``"scene"``).

        Returns:
            A Scene populated with one Object per ``g`` group in the file.
        """
        scene = cls(name=name or "scene")
        for mesh, uv_list in _load_obj(file_path):
            uv = uv_list[0] if len(uv_list) > 0 else None
            scene.append(Object(name=mesh.name or "object", mesh=mesh, uv=uv))
        return scene

    @classmethod
    def load_glb(
        cls,
        file_path:       str,
        name:            Optional[str] = None,
        scale_factor:    float         = 100.0,
        load_skin:       bool          = True,
        extract_texture: bool          = True,
    ) -> "Scene":
        """Load every mesh in a GLB file into a new :class:`Scene` as
        :class:`Object` nodes.  See :meth:`load_obj`.

        Args:
            file_path: Path to the ``.glb`` file (``~`` and env vars OK).
            name: Optional name for the Scene (defaults to ``"scene"``).
            scale_factor: Multiplier applied to vertex positions (matches
                :meth:`MeshData.load_glb`'s default of 100).
            load_skin: When True (default), attach a
                :class:`SkinDeformData` to each Object that has skin
                weights, ready for :meth:`Object.pose`.  The file is read
                once for the whole Scene, not once per Object.
            extract_texture: When True (default), also extract the
                material's base color map and assign it to each Object's
                :attr:`texture` via :meth:`Object.extract_texture_from_glb`,
                matching :meth:`Object.load_glb`.  A file with no base
                color map leaves :attr:`texture` unset rather than raising,
                so an untextured GLB loads exactly as it did before.

        Returns:
            A Scene populated with one Object per geometry in the file.

        Raises:
            ImportError: If ``trimesh`` is not installed.
        """
        scene  = cls(name=name or "scene")
        data   = _load_glb(file_path, scale_factor=scale_factor)
        meshes = [m for m, _ in data]
        skins = (
            _glb_skins(file_path, meshes, scale_factor)
            if load_skin
            else [None] * len(meshes)
        )
        for (mesh, uv_list), skin in zip(data, skins):
            uv  = uv_list[0] if len(uv_list) > 0 else None
            obj = Object(name=mesh.name or "object", mesh=mesh, uv=uv, skin=skin)
            if extract_texture:
                obj.extract_texture_from_glb(file_path)
            scene.append(obj)
        return scene

    # -- render-config helpers --------------------------------------------

    # The Scene-level keys we merge into render() / turntable() kwargs.
    _RENDER_CONFIG_KEYS = (
        "resolution",
        "samples_per_pixel",
        "background",
        "autofit",
        "default_light",
        "return_depth",
        "angle_of_view",
        "fit_padding",
    )

    def configure(self, **kwargs) -> "Scene":
        """Bulk-set persistent render defaults; ``None`` values are ignored.
        Returns ``self`` so it's chainable::

            scene.configure(samples_per_pixel=4, resolution=(1280, 720)).imshow()

        Raises:
            AttributeError: If any kwarg isn't a known render-config attribute.
        """
        for name, val in kwargs.items():
            if val is None:
                continue
            if name not in self._RENDER_CONFIG_KEYS:
                raise AttributeError(
                    f"Scene.configure: unknown render-config attribute "
                    f"{name!r}; allowed: {self._RENDER_CONFIG_KEYS}"
                )
            setattr(self, name, val)
        return self

    def _visible_world_points(self, context: str) -> np.ndarray:
        """Every visible Object's world points as one cloud, for camera fitting.

        Args:
            context: Name of the calling entry point, used in the error so a
                caller knows which of their calls found nothing to frame.
        """
        chunks = [
            o.world_points for o in self.objects if o.visibility and o.mesh is not None
        ]
        if not chunks:
            raise ValueError(f"{context}: scene has no visible Objects with meshes")
        return np.concatenate(chunks, axis=0)

    def _base_camera(self, all_pts: np.ndarray, camera_name, render_kwargs: dict):
        """Orientation and lens a sequence render should fit to.

        Uses an existing Camera when the scene has one, else a default 3/4
        elevated view.  Nothing is appended to the scene -- the matrix is
        handed to :func:`render` per frame instead, so the scene is left
        exactly as the caller set it up.

        Returns:
            ``(base_cam_col, R_cam, aov, width, height)``, the camera matrix
            column-major (OpenGL), its camera-to-world rotation part, the
            angle of view in degrees and the pixel resolution to fit.
        """
        existing = self.get_camera(camera_name)
        if existing is not None:
            # row-major -> column-major
            base_cam_col = np.asarray(existing.world_matrix, dtype=np.float64).T
            base_aov     = float(existing.angle_of_view)
        else:
            base_cam_col = _default_camera_for_points(all_pts)
            base_aov     = 35.0

        res = render_kwargs.get("resolution", (512, 384))
        aov = float(render_kwargs.get("angle_of_view", base_aov))
        return base_cam_col, base_cam_col[:3, :3], aov, int(res[0]), int(res[1])

    def _resolve_mp4_background(self, background=None) -> Tuple[float, float, float]:
        """Opaque RGB for formats that carry no alpha, notably H.264.

        Explicit argument first, then this Scene's own background, then black,
        so a chosen miss colour survives into the video instead of turning up
        as an unexpected black surround.
        """
        if background is not None:
            return tuple(float(v) for v in background)
        if self._background is not None:
            return tuple(float(v) for v in self._background[:3])
        return (0.0, 0.0, 0.0)

    def _merge_render_config(self, kwargs: dict) -> dict:
        """Resolve render-config defaults: per-call kwarg > Scene attr.

        Per-Object render-config does NOT participate -- frame-level
        settings (resolution, samples_per_pixel, background, etc.) are
        inherently uniform across the whole render and live on the Scene.
        Per-Object visualisation tweaks (e.g., ``Object.wireframe``) are
        handled by the Object's own attributes / its texture loader.
        """
        defaults = {k: getattr(self, k) for k in self._RENDER_CONFIG_KEYS}
        merged   = {k: v for k, v in defaults.items() if v is not None}
        merged.update(kwargs)
        return merged

    # -- frame cache: lazy invalidation via render-signature snapshot ---

    def _compute_render_signature(self) -> tuple:
        """Build a hashable snapshot of every render-affecting attribute
        on this Scene and its child Object/Camera/Light nodes.  Used by
        the :attr:`frame` getter to lazy-invalidate the cache when any
        render input has changed since the last :meth:`render` call.

        Captures:
          - Scene-level render config (every key in
            :attr:`_RENDER_CONFIG_KEYS`) plus :attr:`aspect_ratio` and
            :attr:`default_camera_name` (these affect render output but
            are not forwarded as render() kwargs).
          - Per-node: type, uuid, visibility, raw local SRT
            (``_translate`` / ``_rotate`` / ``_scale``) AND composed
            ``world_matrix`` bytes -- the local SRT catches direct
            mutations even when ``TransformData``'s ``_world_matrix``
            cache lags (see :meth:`Object._capture_transform_state`),
            and ``world_matrix`` bytes catch ancestor mutations.
          - Per-Object: shader knobs + wireframe attrs + per-Object
            ``resolution`` / ``samples_per_pixel`` + ``id()`` of
            ``mesh`` / ``uv`` / ``texture`` (identity, not contents --
            matches the existing setter-based invalidation semantics).
          - Per-Camera: ``angle_of_view``, ``is_orthographic``,
            ``ortho_height``, ``near_plane``, ``far_plane``,
            ``aspect_ratio``.
          - Per-Light: ``kind``, ``color``, ``intensity``, ``falloff``.

        NOT captured (won't trigger invalidation):
          - In-place mutation of ``mesh.points`` / ``uv.points`` /
            ``texture`` array contents.  We use ``id()`` -- if the user
            replaces the array via the setter, ``id`` changes and we
            invalidate; if they mutate in place, neither this signature
            nor the existing setter-based invalidation fires (matches
            the documented texture-setter behavior).
          - In-place topology mutation via ``mesh.indices`` /
            ``mesh.counts``.

        Cost: O(num_children) per check; ~one tuple build + a few attr
        reads per child.  Negligible for typical scenes.
        """
        # _RENDER_CONFIG_KEYS covers the attrs forwarded into render()
        # kwargs.  ``aspect_ratio`` and ``default_camera_name`` are NOT
        # forwarded (the renderer derives aspect from the resolution and
        # we resolve the camera by name in :meth:`get_camera` internally)
        # but they DO affect the final render output, so we include them
        # in the signature -- mutating either invalidates ``self.frame``
        # / ``self.buffer`` on the next access.
        scene_part = (
            tuple(getattr(self, k) for k in self._RENDER_CONFIG_KEYS),
            self.aspect_ratio,
            self.default_camera_name,
        )
        children = []
        for node in self:
            transform_state = (
                tuple(np.asarray(node._translate, dtype=float).ravel()),
                tuple(np.asarray(node._rotate, dtype=float).ravel()),
                tuple(np.asarray(node._scale, dtype=float).ravel()),
                np.asarray(node.world_matrix).tobytes(),
            )
            if isinstance(node, Object):
                children.append(
                    (
                        "object",
                        node.uuid,
                        node.visibility,
                        transform_state,
                        id(node._mesh),
                        id(node._uv),
                        id(node._texture),
                        node._base_color,
                        node._wireframe,
                        node._wireframe_color,
                        node._wireframe_thickness,
                        node._resolution,
                        node._samples_per_pixel,
                        node.sample_method,
                        node.wrap,
                        node.ambient,
                        node.twosided,
                        node.cast_shadows,
                    )
                )
            elif isinstance(node, Camera):
                children.append(
                    (
                        "camera",
                        node.uuid,
                        node.visibility,
                        transform_state,
                        node.angle_of_view,
                        node.is_orthographic,
                        node.ortho_height,
                        node.near_plane,
                        node.far_plane,
                        node.aspect_ratio,
                    )
                )
            elif isinstance(node, Light):
                children.append(
                    (
                        "light",
                        node.uuid,
                        node.visibility,
                        transform_state,
                        node.kind,
                        node.color,
                        node.intensity,
                        node.falloff,
                    )
                )
            else:
                # Plain TransformData group node -- only transform matters.
                children.append(("group", node.uuid, node.visibility, transform_state))
        return (scene_part, tuple(children))

    @property
    def frame(self) -> Optional["Frame"]:
        """Cached most-recent render output, or ``None`` when the cache
        is invalid.  Auto-invalidates when ANY render-affecting attribute
        of this Scene or its child Objects/Cameras/Lights has changed
        since the last :meth:`render` call -- including transforms
        (``translate`` / ``rotate`` / ``scale`` on any node), Camera
        domain attrs (e.g., ``angle_of_view``), Light domain attrs
        (e.g., ``intensity``), Object visual attrs, scene topology
        (nodes added / removed), and Scene-level render config (e.g.,
        ``resolution``).  See :meth:`_compute_render_signature` for the
        full list of tracked attributes.
        """
        if self._frame is None:
            return None
        if (
            self._render_signature is not None
            and self._compute_render_signature() != self._render_signature
        ):
            self._frame            = None
            self._render_signature = None
            return None
        return self._frame

    @frame.setter
    def frame(self, value: Optional["Frame"]) -> None:
        self._frame = value
        if value is None:
            self._render_signature = None
        else:
            self._render_signature = self._compute_render_signature()

    @property
    def buffer(self) -> Optional[np.ndarray]:
        """Convenience accessor for ``self.frame.array`` -- the pixel
        buffer of the most recent render (``None`` until rendered).
        Matches the ``buffer`` vocabulary used by :class:`UVData`.
        """
        return self.frame.array if self.frame is not None else None

    @property
    def scene_name(self) -> Optional[str]:
        """Returns the scene's own label (distinct from the inherited
        :attr:`name` property which lists child node names)."""
        return self._scene_name

    @scene_name.setter
    def scene_name(self, value: Optional[str]) -> None:
        self._scene_name = str(value) if value is not None else None

    @property
    def background(self) -> Optional[Tuple[float, float, float, float]]:
        """Optional RGBA background color in [0, 1] used by :meth:`render`.

        ``None`` (default) -> renderer falls through to its own default
        ``(0, 0, 0, 0)`` (transparent black).  RGB 3-tuples passed to
        the constructor or this setter are auto-promoted to opaque
        ``(R, G, B, 1.0)`` via :func:`_normalize_rgba`.

        Mapping to the renderer:
          - ``alpha == 0`` -> transparent miss pixels (compositing-friendly).
          - ``alpha == 1`` -> opaque miss fill in the chosen RGB.
          - ``0 < alpha < 1`` -> Porter-Duff "over" composite at miss.
        """
        return self._background

    @background.setter
    def background(self, value) -> None:
        self._background = _normalize_rgba(value, "Scene.background")

    # ---- type-checked add ----
    def append(self, node: TransformData) -> None:
        self._validate(node)
        # Work around a pre-existing bug in TransformList.append: when
        # node.uuid is None, the actual list-append is unreachable
        # (sits inside a `while True` after `break`).  Pre-assigning a
        # uuid here sidesteps the bug entirely.
        if node.uuid is None:
            from cgmath.hierarchy import generate_uuid

            node.uuid = generate_uuid()
        super().append(node)

    def insert(self, index: int, node: TransformData) -> None:
        self._validate(node)
        if node.uuid is None:
            from cgmath.hierarchy import generate_uuid

            node.uuid = generate_uuid()
        super().insert(index, node)

    @staticmethod
    def _validate(node) -> None:
        if not isinstance(node, TransformData):
            raise TypeError(
                f"Scene only accepts TransformData (got {type(node).__name__}); "
                "wrap your data in Object/Camera/Light"
            )

    # ---- typed views ----
    @property
    def objects(self) -> List[Object]:
        return [n for n in self if isinstance(n, Object)]

    @property
    def cameras(self) -> List[Camera]:
        return [n for n in self if isinstance(n, Camera)]

    @property
    def lights(self) -> List[Light]:
        return [n for n in self if isinstance(n, Light)]

    # ---- camera selection ----
    def get_camera(self, name: Optional[str] = None) -> Optional[Camera]:
        """Returns the named Camera, the default Camera, or the first one.

        Returns ``None`` if there are no Cameras in the scene.
        """
        cams = self.cameras
        if not cams:
            return None
        chosen = name or self.default_camera_name
        if chosen is not None:
            for c in cams:
                if c.name == chosen:
                    return c
        return cams[0]

    # ---- bbox helpers ----
    def union_aabb(
        self, objects: Optional[List[Object]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(min_corner, max_corner)`` of the world-space AABB
        across the requested Objects (or all visible Objects)."""
        if objects is None:
            objects = [o for o in self.objects if o.visibility]
        if not objects:
            return np.zeros(3), np.zeros(3)
        pts_chunks = [o.world_points for o in objects if o.mesh is not None]
        if not pts_chunks:
            return np.zeros(3), np.zeros(3)
        all_pts = np.concatenate(pts_chunks, axis=0)
        return all_pts.min(axis=0), all_pts.max(axis=0)

    def union_points(self, objects: Optional[List[Object]] = None) -> np.ndarray:
        """Returns the concatenation of all visible Objects' world-space
        points, suitable for autofit / default-camera calls."""
        if objects is None:
            objects = [o for o in self.objects if o.visibility]
        chunks = [o.world_points for o in objects if o.mesh is not None]
        if not chunks:
            return np.zeros((0, 3))
        return np.concatenate(chunks, axis=0)

    # ---- convenience: scene.render(...) ----
    def render(self, **kwargs):
        """Convenience wrapper around the module-level ``render(scene=self, ...)``.

        Persistent render-config attributes set on this Scene
        (:attr:`resolution`, :attr:`samples_per_pixel`, etc.) are merged
        into ``kwargs`` here.  Per-call kwargs always win.

        Default lighting:
          When the Scene has no :class:`Light` children and
          ``default_light`` is not disabled, a 3-point key/fill/rim rig
          is auto-injected for the duration of this render call (sized
          to the union AABB of all visible Objects via
          :func:`_default_lighting_rig_for`, so it scales with the
          scene's bounding box and is independent of mesh topology /
          vertex density).  The raytracer's built-in single-headlight
          fallback is suppressed when the auto-rig fires, so we don't
          double-light.  The auto-rig is removed before this method
          returns -- the user's Scene is left untouched.

          This is the SAME rig that :meth:`Object.render` ends up using
          (since :meth:`Object.render` builds a one-Object Scene and
          calls :meth:`Scene.render`), so ``obj.to_image()`` and a
          one-Object ``scene.to_image()`` produce matching lighting
          (modulo resolution / SPP).

        The import of :func:`cgmath.render.raytracer.render` stays lazy
        because raytracer imports :class:`Scene` at top level -- this is the
        one legitimate forward reference (model -> renderer convenience
        shim) and the only remaining lazy import in this module.
        """
        from cgmath.render.raytracer import render

        merged = self._merge_render_config(kwargs)

        # Promote any per-call ``background=`` from RGB / list to RGBA
        # so the renderer (which requires RGBA) sees a 4-tuple.  This
        # mirrors the :class:`Scene.background` setter's normalization
        # for the kwargs path.
        if "background" in merged:
            merged["background"] = _normalize_rgba(
                merged["background"], "render(background=...)"
            )

        # Auto-inject the 3-point default rig when (a) the user hasn't
        # disabled it via default_light=False and (b) no Lights are in
        # the scene.  Sized to the full scene's union AABB so multi-
        # object scenes get a rig fitted to the whole composition.
        auto_lights: List[Light] = []
        if merged.get("default_light", True) is not False and not self.lights:
            pts = self.union_points()
            if pts.size > 0:
                auto_lights = _default_lighting_rig_for(pts)

        try:
            for light in auto_lights:
                self.append(light)
            if auto_lights:
                # Suppress the raytracer's single-headlight fallback so
                # the auto-rig isn't doubled up with a camera headlight.
                merged["default_light"] = False
            frame = render(scene=self, **merged)
        finally:
            # Pop the auto-rig back off so the user's Scene isn't
            # mutated.  We always appended at the end of the list, so
            # a plain pop() N times is the inverse.
            for _ in range(len(auto_lights)):
                self.pop()

        # Cache the rendered Frame so callers can :meth:`imshow` it later.
        self.frame = frame
        return frame

    def to_image(self, resolution: Tuple[int, int] = (500, 500)) -> "Image":
        """Returns the rendered frame as a PIL Image.

        Mirrors :meth:`cgmath.geometry.mesh.UVData.to_image`.  Lazy-renders
        at ``resolution`` (default 500x500 -- a fast preview) when no
        frame is cached.  For a full-quality preview, call :meth:`render`
        explicitly first then :meth:`to_image`.

        Args:
            resolution: Resolution to use when lazy-rendering (only when
                ``self.frame`` is ``None``).  Defaults to a quick
                500x500 preview.

        Raises:
            RuntimeError: If PIL is not available.

        Note:
            Unlike :meth:`UVData.to_image` (which flips its bottom-up
            buffer), no flip is needed here because ``Frame.array`` is
            already top-down RGB uint8.

            Mutating any render input on this Scene or its child
            Objects/Cameras/Lights -- transforms, Camera FOV, Light
            intensity, Object texture/wireframe, scene topology, or
            Scene-level config -- auto-invalidates the cache via the
            lazy signature check on :attr:`frame`.  In-place mutation
            of mesh / UV / texture array contents is the only common
            case that still requires an explicit :meth:`render` call.
        """
        if Image is None:
            raise RuntimeError("PIL required to generate PIL.Image data")
        if self.frame is None:
            self.render(resolution=resolution)  # sets ``self.frame``
        return Image.fromarray(self.frame.array)

    def imshow(self, resolution: Tuple[int, int] = (500, 500)) -> None:
        """Show the rendered frame in a window via OpenCV.

        Mirrors :meth:`cgmath.geometry.mesh.UVData.imshow`.  Lazy-renders
        at ``resolution`` (default 500x500 -- a fast preview) when no
        frame is cached.  For a full-quality preview, call :meth:`render`
        explicitly first then :meth:`imshow`.

        Args:
            resolution: Resolution to use when lazy-rendering (only when
                ``self.frame`` is ``None``).  Defaults to a quick
                500x500 preview.

        Raises:
            RuntimeError: If OpenCV is not available (it's required to
                show the image in a window).

        Note:
            Mutating any render input on this Scene or its child
            Objects/Cameras/Lights -- transforms, Camera FOV, Light
            intensity, Object texture/wireframe, scene topology, or
            Scene-level config -- auto-invalidates the cache via the
            lazy signature check on :attr:`frame`.  In-place mutation
            of mesh / UV / texture array contents is the only common
            case that still requires an explicit :meth:`render` call.
        """
        if cv2 is None:
            raise RuntimeError("opencv required to show images")
        if self.frame is None:
            self.render(resolution=resolution)  # sets ``self.frame``
        # Frame.array is top-down RGB; cv2 wants BGR.  No vertical flip
        # needed (UVData flips because its buffer is bottom-up).
        buffer = cv2.cvtColor(self.frame.array, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.scene_name or "Scene", buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---- turntable ----
    def _skinned_objects(self, clip) -> Tuple[list, list]:
        """Visible skinned Objects and the joint columns each needs from *clip*.

        Resolving the columns once here is the difference between an animation
        loop that re-matches joint NAMES every frame and one that does not.
        """
        skinned = [o for o in self.objects if o.visibility and o.skin is not None]
        if not skinned:
            raise ValueError(
                "animate: scene has no visible skinned Objects -- load them "
                "with load_skin=True, or use turntable() for a static scene"
            )

        columns = []
        for obj in skinned:
            try:
                columns.append(obj.skin.pose_indices(clip))
            except Exception as exc:
                raise ValueError(
                    f"animate: {obj.name!r} is skinned to joints the clip does "
                    f"not hold ({exc})"
                ) from exc
        return skinned, columns

    def animate(
        self,
        clip,
        output_pattern: str                                  = "animation.{frame:04d}.png",
        fit:            str                                  = "auto",
        rotation_axis:  str                                  = "y",
        start_angle:    float                                = 0.0,
        end_angle:      float                                = 0.0,
        camera_name:    Optional[str]                        = None,
        fps:            Optional[int]                        = None,
        background:     Optional[Tuple[float, float, float]] = None,
        n_pose_samples: Optional[int]                        = None,
        n_fit_samples:  Optional[int]                        = None,
        verbose:        bool                                 = False,
        **render_kwargs,
    ) -> Union[str, List[str]]:
        """Render a :class:`ClipData` frame by frame, deforming as it goes.

        The complement of :meth:`turntable`, which moves the camera and holds
        the scene still.  This moves the scene and, by default, holds the
        camera still -- but the same orbit arguments are here, so passing
        ``end_angle=360`` gives a turntable OF an animation.

        Every visible Object carrying a :attr:`~Object.skin` is posed to the
        clip each frame; Objects without one render as static props.  The work
        that does not change per frame -- binding, resolving joint columns and
        evaluating the clip's world matrices -- is hoisted out of the loop.

        The scene is returned to its bind pose when this finishes, so an
        animate call does not leave your Objects deformed and calling it twice
        gives the same answer.

        Output format is detected from ``output_pattern``'s extension exactly
        as in :meth:`turntable`: ``.mp4`` / ``.mov`` / ``.m4v`` / ``.avi`` /
        ``.gif`` encode to one file, anything else writes an image sequence.

        Args:
            clip: A :class:`ClipData` holding the animation.  Its joints are
                matched to each Object's skin by name, once, up front.
            output_pattern: Output filename.  Image sequences support a
                ``{frame[:NNd]}`` placeholder and get one if absent.
            fit: ``"auto"`` (default) samples poses across the clip and
                positions the camera to fit all of them, so a character that
                travels does not walk out of frame.  ``"static"`` uses the
                scene's Camera as-is.
            rotation_axis / start_angle / end_angle: Optional camera orbit
                over the length of the clip, same meaning as in
                :meth:`turntable`.  Defaults describe no orbit at all.
            camera_name: Which Scene Camera to use as the base orientation.
            fps: Frames per second for video / gif.  Defaults to the clip's
                own ``fps``.
            background: Opaque RGB to composite over for formats with no
                alpha.  Falls back to the Scene's background, then black.
            n_pose_samples: How many poses to sample when fitting.  See
                :meth:`_pose_sample_frames` for the default.
            n_fit_samples: How many orbit angles to sample when fitting.
                Defaults to the frame count.  Irrelevant without an orbit.
            verbose: Print a line per frame.
            **render_kwargs: Forwarded to :func:`render`.

        Returns:
            A single output path for video / gif, or the list of frame paths.

        Raises:
            ValueError: If the scene holds no visible skinned Objects, if the
                clip has no frames, if an Object is skinned to joints the clip
                does not hold, or if *fit* is not one of the two options.
        """
        render_kwargs = self._merge_render_config(render_kwargs)

        frame_count   = int(clip.frame_count)
        if frame_count <= 0:
            raise ValueError("animate: the clip has no frames")

        skinned, columns = self._skinned_objects(clip)

        # One evaluation for the whole clip rather than one per frame.
        world = np.asarray(clip.frames.world_matrix, dtype=np.float64)

        if fps is None:
            fps = int(round(float(getattr(clip, "fps", 30.0)) or 30.0))

        try:
            chosen_cam_col, centroid = self._fit_animated_camera(
                skinned,
                columns,
                world,
                frame_count,
                fit,
                rotation_axis,
                start_angle,
                end_angle,
                camera_name,
                n_pose_samples,
                n_fit_samples,
                render_kwargs,
                verbose,
            )
        finally:
            # The fit pass deforms to sample poses; do not leave it that way
            # even if it failed part way through.
            for obj in skinned:
                obj.restore_bind_pose()

        render_kwargs.setdefault("autofit", False)
        span = float(end_angle - start_angle)

        def _frame_stream():
            try:
                for index in range(frame_count):
                    for obj, column in zip(skinned, columns):
                        obj.pose(world[index, column])
                    angle = start_angle + span * index / max(frame_count, 1)
                    render_kwargs["camera_matrix"] = _orbit_camera_col(
                        chosen_cam_col, centroid, rotation_axis, angle
                    )
                    yield self.render(**render_kwargs)
            finally:
                # Runs on early close too, so an encoder that dies part way
                # through still hands the scene back undeformed.
                for obj in skinned:
                    obj.restore_bind_pose()

        return _encode_frame_stream(
            _frame_stream(),
            output_pattern,
            fps            = int(fps),
            mp4_background = self._resolve_mp4_background(background),
            verbose        = verbose,
        )

    def _fit_animated_camera(
        self,
        skinned,
        columns,
        world,
        frame_count,
        fit,
        rotation_axis,
        start_angle,
        end_angle,
        camera_name,
        n_pose_samples,
        n_fit_samples,
        render_kwargs,
        verbose,
    ):
        """Camera matrix and orbit centroid for :meth:`animate`.

        Poses are sampled and their clouds CONCATENATED, then the orbit sweep
        runs once over the union.  Sweeping per pose instead would multiply
        the angle work by the pose count for the same answer.
        """
        if fit not in ("auto", "static"):
            raise ValueError(f"animate: fit must be 'auto' or 'static', got {fit!r}")

        clouds = []
        for index in _pose_sample_frames_for(frame_count, n_pose_samples):
            for obj, column in zip(skinned, columns):
                obj.pose(world[index, column])
            clouds.append(self._visible_world_points("animate"))
        all_pts  = np.concatenate(clouds, axis=0)

        centroid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
        base_cam_col, R_cam, aov, width, height = self._base_camera(
            all_pts, camera_name, render_kwargs
        )
        render_kwargs.setdefault("angle_of_view", aov)

        if fit == "static":
            return base_cam_col, centroid

        pts_centered = _inflate_points_about_center(
            all_pts - centroid,
            _resolve_turntable_fit_padding(render_kwargs, self),
        )
        n_samples = (
            int(n_fit_samples) if n_fit_samples is not None else int(frame_count)
        )
        chosen = _fit_camera_over_orbit(
            pts_centered,
            R_cam,
            base_cam_col,
            centroid,
            aov,
            width,
            height,
            rotation_axis,
            start_angle,
            end_angle,
            max(n_samples, 1),
            verbose=verbose,
        )
        return chosen, centroid

    def turntable(
        self,
        output_pattern: str                                  = "turntable.{frame:04d}.png",
        n_frames:       int                                  = 120,
        rotation_axis:  str                                  = "y",
        start_angle:    float                                = 0.0,
        end_angle:      float                                = 360.0,
        fit:            str                                  = "auto",
        camera_name:    Optional[str]                        = None,
        fps:            int                                  = 30,
        background:     Optional[Tuple[float, float, float]] = None,
        n_fit_samples:  Optional[int]                        = None,
        verbose:        bool                                 = False,
        **render_kwargs,
    ) -> Union[str, List[str]]:
        """Render a turntable animation: orbit a camera around the scene.

        The whole scene is treated as a single rigid point cloud and the
        camera orbits the world ``rotation_axis`` (``"y"`` by default)
        around the scene's centroid (AABB midpoint).  Per frame, the
        chosen base camera is rotated by the frame's angle around that
        pivot -- nothing in the scene is mutated, so all Object
        transforms stay exactly as you set them.

        Camera framing is computed once before the render loop:

        1. Concatenate every visible Object's world points into one cloud.
        2. Choose a base camera *orientation* (existing Camera in the
           scene, or a default 3/4 elevated view).
        3. Express the points in a *centroid-centered, camera-oriented*
           frame (X right, Y up, Z behind the camera).
        4. For each of N sampled angles, rotate the centred point cloud
           around the world rotation_axis (= around the centroid) and
           accumulate the union AABB in that camera-oriented frame.
        5. Solve analytically for the camera position that exactly fits
           the union AABB into the FOV: lateral X/Y to centre the box on
           the optical axis, and Z to push back just enough that the box
           edges touch the frustum walls.

        This single pass replaces the older per-sample autofit + max
        pull-back heuristic and gives pixel-perfect framing at every
        angle (no off-centre clipping).

        The output format is detected from ``output_pattern``'s extension:

        * ``.mp4`` / ``.mov`` / ``.m4v`` -> H.264 video via ffmpeg.  Frames
          are streamed to ffmpeg's stdin so no temporary PNGs hit disk.
          Returns a single file path (string).
        * ``.gif`` -> high-quality GIF via gifski (when installed) or
          ffmpeg's two-pass palettegen + paletteuse.  Frames are written
          to a temp directory for palette analysis, then encoded.  Returns
          a single file path (string).
        * any other extension (default ``.png``) -> per-frame image
          sequence as before.  Returns a list of file paths.

        Args:
            output_pattern: Output filename.  For image sequences, supports
                a ``{frame[:NNd]}`` placeholder (defaults to
                ``<stem>.{frame:04d}<ext>`` if no placeholder is present).
                For video / gif, this is just the output file path.
            n_frames: Number of frames in the loop.
            rotation_axis: ``"x"``, ``"y"`` (default), or ``"z"``.
            start_angle / end_angle: Rotation range in degrees.  Last
                frame is at ``end_angle - (end_angle - start_angle)/N``
                so the loop seamlessly repeats.
            fit: ``"auto"`` (default) runs the BBX-based framing pass and
                positions the camera to exactly fit the worst-case
                silhouette across all sampled angles.  ``"static"`` uses
                the scene's existing Camera as-is (no autofit) -- the
                camera still orbits the centroid, just at the user's
                supplied position.
            camera_name: Which Scene Camera to use as the base orientation
                (default = first / named).  When the scene has no Camera,
                a default 3/4 elevated view is built.
            fps: Frames per second for video / gif outputs (ignored for
                image sequences).  Defaults to 30.
            n_fit_samples: How many angles to sample when computing the
                union AABB.  ``None`` (default) uses ``n_frames`` -- exact
                coverage, still cheap.  Set lower for very high frame counts.
            verbose: When True, print a progress line per frame.
            **render_kwargs: Forwarded to :func:`render`.

        Returns:
            A single output path (``str``) for video / gif outputs, or a
            list of frame paths (``list[str]``) for image sequences.
        """

        import os

        # ---- merge persistent Scene render-config into kwargs ----
        # Per-call kwargs always win; Scene attrs fill the gaps.  This
        # mirrors what :meth:`render` does so persistent settings (e.g.
        # ``scene.samples_per_pixel = 1``) propagate to every turntable
        # frame without being repeated.
        render_kwargs = self._merge_render_config(render_kwargs)

        # ---- gather all visible world-space points (one cloud) ----
        all_pts = self._visible_world_points("turntable")

        # Centroid = AABB midpoint.  More robust than a vertex mean for
        # meshes with uneven vertex density.
        centroid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))

        # ---- pick / build a base camera (orientation only) ----
        base_cam_col, R_cam, aov, width, height = self._base_camera(
            all_pts, camera_name, render_kwargs
        )

        # ---- compute chosen camera position ----
        if fit == "auto":
            # Express points in a centroid-centered, camera-oriented frame.
            # In OpenGL convention the camera looks down -Z; with the camera
            # at the centroid, points in front of the camera have negative
            # local Z.  For row vectors: pts_local = (pts - centroid) @ R_cam
            # (R_cam is column-major camera-to-world; transposing implicitly
            # via row-vector convention takes world -> camera-local).
            # Inflation about the AABB center: ``pts_centered`` already has
            # its AABB center at the origin (centroid is the AABB midpoint),
            # so this is just a uniform scale.  See _resolve_turntable_fit_padding
            # for the explicit-vs-default policy (matches single-frame render()).
            pts_centered = _inflate_points_about_center(
                all_pts - centroid,
                _resolve_turntable_fit_padding(render_kwargs, self),
            )

            n_samples = (
                int(n_fit_samples) if n_fit_samples is not None else int(n_frames)
            )
            chosen_cam_col = _fit_camera_over_orbit(
                pts_centered,
                R_cam,
                base_cam_col,
                centroid,
                aov,
                width,
                height,
                rotation_axis,
                start_angle,
                end_angle,
                max(n_samples, 1),
                verbose=verbose,
            )
        elif fit == "static":
            chosen_cam_col = base_cam_col
        else:
            raise ValueError(f"fit must be 'auto' or 'static', got {fit!r}")

        # Suppress per-frame autofit on top of our chosen camera, and
        # keep the FOV consistent with what we just fit to.
        render_kwargs.setdefault("autofit", False)
        render_kwargs.setdefault("angle_of_view", aov)

        span = float(end_angle - start_angle)

        def _frame_stream():
            for fi in range(int(n_frames)):
                angle = start_angle + span * fi / max(int(n_frames), 1)
                render_kwargs["camera_matrix"] = _orbit_camera_col(
                    chosen_cam_col, centroid, rotation_axis, angle
                )
                yield self.render(**render_kwargs)

        return _encode_frame_stream(
            _frame_stream(),
            output_pattern,
            fps            = int(fps),
            mp4_background = self._resolve_mp4_background(background),
            verbose        = verbose,
        )


# -- helpers ---------------------------------------------------------------------------


def _validate_samples_per_pixel(
    value:      Optional[int],
    *,
    field_name: str           = "samples_per_pixel",
) -> Optional[int]:
    """Validate ``samples_per_pixel`` against the renderer's contract.

    The raytracer lays samples out on a regular ``n_axis x n_axis``
    sub-pixel grid, so the value must be ``1`` or a perfect square
    (``1, 4, 9, 16, 25, ...``).  Anything else raises
    :class:`ValueError` with a message naming ``field_name`` so the
    user knows which attribute or kwarg was bad.

    ``None`` is accepted (returned as-is) -- it means "defer to the
    renderer's internal default".

    Returns the validated ``int`` (or ``None``) so callers can use
    this both as a guard and as a normalizing converter::

        self._samples_per_pixel = _validate_samples_per_pixel(
            value, field_name="Object.samples_per_pixel"
        )
    """
    if value is None:
        return None
    try:
        spp = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must be an int (or None); got {value!r}") from e
    if spp < 1:
        raise ValueError(f"{field_name} must be >= 1 (or None); got {spp}")
    n_axis = int(round(spp**0.5))
    if n_axis * n_axis != spp:
        raise ValueError(
            f"{field_name} must be 1 or a perfect square "
            f"(1, 4, 9, 16, 25, ...); got {spp}"
        )
    return spp


def _normalize_rgba(
    value:      Optional[Union[Tuple[float, ...], List[float]]],
    field_name: str,
) -> Optional[Tuple[float, float, float, float]]:
    """Normalize an optional RGB / RGBA color into a 4-tuple.

    Accepts:
      - ``None`` -> returns ``None`` (caller falls through to the
        renderer's own default ``(0, 0, 0, 0)`` transparent black).
      - 3-element sequence ``(R, G, B)`` -> returns ``(R, G, B, 1.0)``
        (opaque).  Convenience for callers passing RGB.
      - 4-element sequence ``(R, G, B, A)`` -> kept as-is.

    Each component must be a float in ``[0, 1]``.  Anything else raises
    :class:`ValueError` with a message naming ``field_name`` (e.g.
    ``"Object.background"``) so the user knows which attribute was bad.

    Returned tuples are plain ``float``-element ``tuple`` (hashable, so
    they participate cleanly in :meth:`Scene._compute_render_signature`).
    """
    if value is None:
        return None
    try:
        seq = tuple(float(v) for v in value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} must be a 3- or 4-tuple of floats; got {value!r}"
        ) from e
    if len(seq) == 3:
        seq = seq + (1.0,)
    elif len(seq) != 4:
        raise ValueError(
            f"{field_name} must be a 3- or 4-tuple of floats; "
            f"got length {len(seq)}: {value!r}"
        )
    if any(c < 0.0 or c > 1.0 for c in seq):
        raise ValueError(f"{field_name} components must be in [0, 1]; got {seq!r}")
    return seq


def _scene_objects_use_bezier(scene: "Scene") -> bool:
    """True if any visible Object in ``scene`` uses bezier sampling.

    Mirrors :func:`cgmath.render.raytracer._scene_uses_bezier` but
    operates on Scene Objects rather than per-render entry dicts.  Used
    by :meth:`Scene.turntable` to pick the same default ``fit_padding``
    (1.15 for bezier, 1.0 otherwise) as a single-frame
    :func:`render` call.
    """
    for obj in scene.objects:
        if not obj.visibility or obj.mesh is None:
            continue
        sm = obj.sample_method
        if isinstance(sm, str) and sm.lower() == "bezier":
            return True
        # Tolerate enum-valued sample_method without importing the enum
        # here (avoids a hard dependency on cgmath.geometry.mesh in this
        # leaf helper).  Fall back to name-equality.
        if getattr(sm, "name", "").lower() == "bezier":
            return True
    return False


def _resolve_turntable_fit_padding(render_kwargs: dict, scene: "Scene") -> float:
    """Pick the autofit safety-margin factor for :meth:`Scene.turntable`.

    Pops ``fit_padding`` from ``render_kwargs`` (so it doesn't leak into
    per-frame :func:`render` calls and double-inflate an already inflated
    point cloud).  Explicit kwarg wins; bezier auto-defaults to 1.15.
    """
    pad = render_kwargs.pop("fit_padding", None)
    if pad is not None:
        return float(pad)
    return 1.15 if _scene_objects_use_bezier(scene) else 1.0


def _glb_skins(
    file_path:    str,
    meshes:       List[MeshData],
    scale_factor: float,
) -> List[Optional[SkinDeformData]]:
    """One :class:`SkinDeformData` per entry of *meshes*, ``None`` where the
    mesh carries no skin.  Used by both :meth:`Object.load_glb` and
    :meth:`Scene.load_glb`, so the file is read once here rather than once
    per mesh.

    The two lists are paired POSITIONALLY, not by name.  Both readers walk
    the same file, but they get their names from different libraries:
    ``MeshData.load_glb`` uses trimesh's ``scene.geometry`` keys while the
    skin reader uses the glb's own mesh names.  Trimesh has to keep its keys
    unique, so a file holding two meshes both called ``part`` reads back as
    ``part`` and ``part_1`` on one side and ``part`` twice on the other --
    a name lookup would miss the second mesh and double-bind the first.
    The same split happens to a mesh with two primitives.

    The lists only diverge in length when trimesh drops a primitive it
    cannot read (anything that is not triangles), and that is exactly the
    case this cannot pair.  Rather than guess, or raise on a file that
    loaded fine before skinning existed, it warns and skins nothing.
    """
    try:
        skins = _load_skin_glb(file_path, bind_matrices=True)
    except Exception as exc:
        warnings.warn(f"{file_path}: could not read skin weights ({exc})", stacklevel=2)
        return [None] * len(meshes)

    # Checked before the length test on purpose: an unskinned file has
    # nothing that could be mis-bound, so it is not worth a warning, and
    # most glbs are unskinned.
    if not any(x is not None for x in skins):
        return [None] * len(meshes)

    if len(skins) != len(meshes):
        warnings.warn(
            f"{file_path}: {len(meshes)} meshes but {len(skins)} skin entries, "
            f"so they cannot be paired up -- loading without skinning. This "
            f"usually means the file holds a primitive that is not triangles.",
            stacklevel=2,
        )
        return [None] * len(meshes)

    return [
        None
        if entry is None
        else _bound_deformer(
            file_path, mesh, entry[0], _scaled_binds(entry[1], scale_factor)
        )
        for mesh, entry in zip(meshes, skins)
    ]


def _scaled_binds(matrices: np.ndarray, scale_factor: float) -> np.ndarray:
    """A glb's inverse bind matrices in the units its points were read in.

    The file authors them in its own units while :meth:`MeshData.load_glb`
    scales the points it reads, so the two only line up once the bind is
    scaled to match.  Only the translation row moves: scaling a rig scales
    its local translations and leaves rotation alone, and for a bind world
    ``[[R, 0], [t, 1]]`` scaled to ``[[R, 0], [t s, 1]]`` the inverse goes
    from ``[[R', 0], [-t R', 1]]`` to ``[[R', 0], [-t s R', 1]]``.
    """
    if scale_factor == 1.0:
        return matrices
    scaled = np.array(matrices, dtype=np.float64, copy=True)
    scaled[:, 3, :3] *= scale_factor
    return scaled


def _bound_deformer(
    file_path:             str,
    mesh:                  MeshData,
    skin,
    inverse_bind_matrices: np.ndarray,
) -> Optional[SkinDeformData]:
    """Build a deformer for *mesh* and bind it, or warn and give back ``None``.

    Bound here rather than left for the first :meth:`Object.pose` because
    binding is what turns the influence NAMES into joint columns, and it is
    those columns that persist.  The raw ``SkinData`` behind them is only a
    cache, so an unbound deformer that gets saved comes back unable to bind
    at all.  Binding now also means a bad skin is reported while the file is
    being read, rather than part way through an animation.

    The matrices come from the file rather than from ``inv`` of a rig read
    back out of it: those node transforms are wherever the take left them,
    which for an animated file is not the pose the mesh was bound in.

    Nothing in here is allowed to be fatal: reading a file without skinning
    worked before this argument existed and has to keep working.
    """
    try:
        deformer = SkinDeformData(
            mesh, skin, inverse_bind_matrices=inverse_bind_matrices, name=mesh.name
        )
        deformer.bind()
    except Exception as exc:
        warnings.warn(
            f"{file_path}: could not bind the skin for {mesh.name!r} ({exc}) "
            f"-- loading it without skinning.",
            stacklevel=2,
        )
        return None
    return deformer


def _fbx_skin(file_path: str, mesh: MeshData) -> Optional[SkinDeformData]:
    """The :class:`SkinDeformData` for *mesh*, or ``None`` if it has no skin.

    Matched BY NAME, which is the opposite of what :func:`_glb_skins` does,
    because the fbx readers fail the other way round.  Both of them name a
    mesh from the same ``FbxNode``, so the names are trustworthy here -- but
    the skin reader only yields an entry for a mesh that actually has a
    deformer, while the mesh reader yields every mesh.  One static prop in
    the file is enough to shift the lists out of step, so a positional pair
    would quietly bind a mesh to some other mesh's weights.
    """
    try:
        entries = _load_skin_fbx(file_path, bind_matrices=True)
    except Exception as exc:
        warnings.warn(f"{file_path}: could not read skin weights ({exc})", stacklevel=2)
        return None

    matches = [entry[1:] for entry in entries if entry[0] == mesh.name]
    if not matches:
        return None
    if len(matches) > 1:
        warnings.warn(
            f"{file_path}: {len(matches)} skinned meshes are named "
            f"{mesh.name!r}, so the right one cannot be told apart -- "
            f"loading without skinning.",
            stacklevel=2,
        )
        return None

    return _bound_deformer(file_path, mesh, *matches[0])


def _encode_frame_stream(
    frames,
    output_pattern: str,
    fps:            int                        = 30,
    mp4_background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    verbose:        bool                       = False,
) -> Union[str, List[str]]:
    """Write a stream of rendered frames out, encoder chosen by extension.

    * ``.mp4`` / ``.mov`` / ``.m4v`` -> H.264 via ffmpeg.  Frames are fed to
      ffmpeg's stdin, so no temporary PNGs hit disk.  Returns the path.
    * ``.avi`` -> avi.  Returns the path.
    * ``.gif`` -> gif via gifski when installed, else ffmpeg's two pass
      palettegen / paletteuse.  Returns the path.
    * anything else -> a numbered image sequence, honouring a
      ``{frame[:NNd]}`` placeholder and inventing one when absent.  Returns
      the list of frame paths.

    *frames* is consumed lazily, so a caller rendering frame by frame never
    holds more than one frame in memory at a time.
    """
    from cgmath.render.frame import Frame

    ext = os.path.splitext(output_pattern)[1].lower()

    if ext in (".mp4", ".mov", ".m4v", ".avi", ".gif"):
        out_path = os.path.expanduser(output_pattern)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        def announced():
            for index, frame in enumerate(frames):
                if verbose:
                    print(f"  frame {index:04d}")
                yield frame

        if ext == ".gif":
            return Frame.encode_gif(announced(), out_path, fps=int(fps))
        if ext == ".avi":
            return Frame.encode_avi(announced(), out_path, fps=int(fps))
        # H.264 carries no alpha, so it composites over an opaque RGB.
        return Frame.encode_mp4(
            announced(), out_path, fps=int(fps), background=mp4_background
        )

    out = output_pattern
    if "{frame" not in out:
        stem, ext_or_default = os.path.splitext(out)
        out = stem + ".{frame:04d}" + (ext_or_default or ".png")
    os.makedirs(os.path.dirname(os.path.expanduser(out)) or ".", exist_ok=True)

    saved: List[str] = []
    for index, frame in enumerate(frames):
        path = os.path.expanduser(out.format(frame=index))
        frame.save(path)
        saved.append(path)
        if verbose:
            print(f"  frame {index:04d} -> {path}")
    return saved


def _pose_sample_frames_for(
    frame_count: int, n_pose_samples: Optional[int]
) -> np.ndarray:
    """Which frames :meth:`Scene.animate` should pose to work out its framing.

    Sampling rather than posing every frame keeps the cost of aiming the
    camera bounded on a long clip: 10 frames by default, rising to a tenth of
    the clip once that is more than 10, and never more than the clip has.
    Deforming is the expensive part, which is why this is the number that
    matters -- sweeping the orbit angles afterwards is a matmul per angle and
    no deformation at all.

    The trade is that a pose falling between samples can poke slightly outside
    the frame.  Pass ``n_pose_samples=frame_count`` for an exact fit.
    """
    if n_pose_samples is None:
        n_pose_samples = max(10, -(-frame_count // 10))
    # Clamping to frame_count cannot change what comes back -- np.unique
    # already collapses a longer linspace onto the same frames -- but it does
    # stop an over-large request allocating a huge intermediate first.
    count = max(1, min(int(n_pose_samples), frame_count))
    return np.unique(np.linspace(0, frame_count - 1, count).astype(int))


def _orbit_camera_col(
    cam_col:       np.ndarray,
    centroid:      np.ndarray,
    rotation_axis: str,
    angle:         float,
) -> np.ndarray:
    """*cam_col* orbited by *angle* about *rotation_axis* through *centroid*.

    Rotation around an arbitrary world point is the standard translate ->
    rotate -> translate-back composition.  Column-major (OpenGL) in, column
    major out, ready to hand to :func:`render` as ``camera_matrix``.
    """
    R                = _rotation_matrix_col(rotation_axis, angle)
    to_origin        = np.eye(4, dtype=np.float64)
    to_origin[:3, 3] = -centroid
    back             = np.eye(4, dtype=np.float64)
    back[:3, 3]      = centroid
    return back @ R @ to_origin @ cam_col


def _fit_camera_over_orbit(
    pts_centered:  np.ndarray,
    R_cam:         np.ndarray,
    base_cam_col:  np.ndarray,
    centroid:      np.ndarray,
    aov:           float,
    width:         int,
    height:        int,
    rotation_axis: str,
    start_angle:   float,
    end_angle:     float,
    n_samples:     int,
    verbose:       bool       = False,
) -> np.ndarray:
    """Camera matrix that exactly fits *pts_centered* at every sampled angle.

    *pts_centered* is the cloud already centred on the AABB midpoint and
    already inflated by the fit padding.  For an animation it is the union of
    several posed clouds, so a pose that only appears mid-clip still gets
    framed; for a turntable it is the single static cloud.  Either way the
    solve below is the same, which is the reason this is shared.

    Sweeping the angles costs one matmul per sample over the whole cloud and
    no deformation at all, so a caller that has already paid to pose the mesh
    should concatenate its poses and sweep once rather than sweep per pose.
    """
    bbx_min = np.full(3, np.inf, dtype=np.float64)
    bbx_max = np.full(3, -np.inf, dtype=np.float64)
    span    = float(end_angle - start_angle)

    # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64):
    # two matmuls per sample, neither does anything dodgy.
    with np.errstate(all="ignore"):
        for i in range(n_samples):
            angle = start_angle + span * i / max(n_samples, 1)
            # World-axis rotation around the centroid.  pts_centered are
            # already centered, so rotating about the local origin IS
            # rotating about the centroid in world.
            R_world_3 = _rotation_matrix_col(rotation_axis, angle)[:3, :3]
            # Row-vector convention for column-major R: pts @ R.T
            rotated_world = pts_centered @ R_world_3.T
            # Transform to camera-oriented (camera at centroid) frame.
            rotated_local = rotated_world @ R_cam
            bbx_min       = np.minimum(bbx_min, rotated_local.min(axis=0))
            bbx_max       = np.maximum(bbx_max, rotated_local.max(axis=0))

    # Solve for the camera position in the centroid-centered camera-oriented
    # frame.  The BBX extends in [bbx_min, bbx_max].
    #   - centre laterally so the BBX midpoint sits on the optical axis
    #   - push back along +Z so the closest BBX corner (at z=bbx_max[2]) is
    #     far enough away that the half-extents fit the FOV.
    half_aov_v = 0.5 * np.deg2rad(aov)
    half_aov_h = np.arctan(np.tan(half_aov_v) * (width / height))
    half_w     = 0.5 * (bbx_max[0] - bbx_min[0])
    half_h     = 0.5 * (bbx_max[1] - bbx_min[1])
    cx         = 0.5 * (bbx_min[0] + bbx_max[0])
    cy         = 0.5 * (bbx_min[1] + bbx_max[1])
    required_dist = max(
        half_w / max(np.tan(half_aov_h), 1e-12),
        half_h / max(np.tan(half_aov_v), 1e-12),
    )
    cz               = float(bbx_max[2]) + required_dist
    cam_local_offset = np.array([cx, cy, cz], dtype=np.float64)

    # World-space camera position: centroid + R_cam @ offset_col
    cam_world_pos = centroid + R_cam @ cam_local_offset

    if verbose:
        print(
            f"  autofit: union BBX (w x h x d) = "
            f"{bbx_max[0] - bbx_min[0]:.3f} x {bbx_max[1] - bbx_min[1]:.3f} x "
            f"{bbx_max[2] - bbx_min[2]:.3f}, "
            f"camera pull-back = {required_dist:.3f}"
        )

    chosen        = np.array(base_cam_col, dtype=np.float64, copy=True)
    chosen[:3, 3] = cam_world_pos
    return chosen


def _rotation_matrix_col(axis: str, angle_deg: float) -> np.ndarray:
    """4x4 column-major (OpenGL) rotation matrix about a world axis.

    Right-hand rule: positive angle about ``+Y`` sends ``+X`` toward ``+Z``
    (and the cyclic equivalents for ``X`` and ``Z``).

    Use ``R.T`` when feeding row-vector multiplication routines
    (e.g. ``transforms.matrix.point``).
    """
    rad = float(np.deg2rad(angle_deg))
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4, dtype=np.float64)
    if axis == "x":
        R[1, 1] = c
        R[1, 2] = -s
        R[2, 1] = s
        R[2, 2] = c
    elif axis == "y":
        R[0, 0] = c
        R[0, 2] = -s
        R[2, 0] = s
        R[2, 2] = c
    elif axis == "z":
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
    return R


def _default_lighting_rig_for(points: np.ndarray) -> List[Light]:
    """Returns a 3-point (key / fill / rim) :class:`Light` rig sized to
    fit the AABB of ``points``.

    Distances scale with the points' diagonal extent so a 1-unit cube
    and a 100-unit spaceship both light up reasonably without manual
    tuning.  Intensities are tuned for the renderer's 1/r^2 point-light
    falloff so the surface receives roughly unit-bright illumination
    at the rig's distance.

    Layout (centroid-relative, classic CG portrait):

    * **key** -- front-upper-right, full intensity
    * **fill** -- front-upper-left, ~40% intensity (softens shadows)
    * **rim** -- behind-upper, ~30% intensity (silhouette pop)

    Used internally by :meth:`Object.render` and :meth:`Object.turntable`
    to give one-call previews sensible default lighting.  Power users can
    call this directly to seed a Scene::

        scene.extend(_default_lighting_rig_for(scene.union_points()))
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        # Degenerate: place a single key light at a unit distance so the
        # scene isn't pitch black for empty / point-only geometry.
        return [
            Light(
                name      = "key",
                kind      = "point",
                translate = [1.0, 1.0, 1.0],
                intensity = 1.0,
            )
        ]

    mn       = pts.min(axis=0)
    mx       = pts.max(axis=0)
    centroid = 0.5 * (mn + mx)
    extent   = max(float(np.linalg.norm(mx - mn)), 1e-6)

    distance = extent * 1.5
    # 1/r^2 falloff: distance^2 gives ~unit-bright at this distance.
    intensity = distance * distance

    key_pos   = centroid + np.array([1.0, 1.0, 1.0]) * distance
    fill_pos  = centroid + np.array([-1.0, 0.5, 1.0]) * distance
    rim_pos   = centroid + np.array([0.0, 0.5, -1.0]) * distance

    return [
        Light(
            name      = "key",
            kind      = "point",
            translate = key_pos.tolist(),
            intensity = intensity,
            falloff   = True,
        ),
        Light(
            name      = "fill",
            kind      = "point",
            translate = fill_pos.tolist(),
            intensity = intensity * 0.4,
            falloff   = True,
        ),
        Light(
            name      = "rim",
            kind      = "point",
            translate = rim_pos.tolist(),
            intensity = intensity * 0.3,
            falloff   = True,
        ),
    ]


# -- GLB texture extraction -----------------------------------------------------
#
# Full glTF (PBR metallic-roughness) texture extraction from a GLB
# container.  Only the ``"base_color"`` slot is wired into the renderer
# today (see :meth:`Object.extract_texture_from_glb`) but the helper
# extracts all 5 standard PBR slots so a future multi-texture shader
# upgrade can pick up metallic-roughness / normal / occlusion / emissive
# without rewriting the GLB walk.


# GLB container constants (binary-glTF 2.0 spec).
_GLB_MAGIC      = 0x46546C67  # ASCII 'glTF' little-endian
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK  = 0x004E4942

# Material shader-slot paths (dotted) to our internal short names.
# Ordered so ``base_color`` (the only one wired today) is first.
_GLB_TEXTURE_SLOTS = {
    "pbrMetallicRoughness.baseColorTexture": "base_color",
    "pbrMetallicRoughness.metallicRoughnessTexture": "metallic_roughness",
    "normalTexture": "normal",
    "occlusionTexture": "occlusion",
    "emissiveTexture": "emissive",
}


def _parse_glb(glb_path: str) -> Tuple[Optional[dict], Optional[bytes]]:
    """Parse a GLB binary-glTF container into ``(json_chunk, bin_chunk)``.

    The GLB layout:

      - 12-byte header: magic (``'glTF'``) | version | total length
      - Chunk 0 (JSON):  chunk_length | ``0x4E4F534A`` | JSON payload
      - Chunk 1 (BIN):   chunk_length | ``0x004E4942`` | binary buffer

    Returns ``(None, None)`` if the file isn't a valid GLB.  Callers can
    handle missing BIN chunks gracefully (e.g., GLBs that reference only
    external image URIs).
    """
    with open(os.path.expanduser(glb_path), "rb") as f:
        data = f.read()

    if len(data) < 12:
        return None, None

    magic, _version, _total_length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        return None, None

    offset = 12
    json_chunk: Optional[dict] = None
    bin_chunk:  Optional[bytes] = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk_data = data[offset + 8 : offset + 8 + chunk_length]
        if chunk_type == _GLB_JSON_CHUNK:
            json_chunk = json.loads(chunk_data)
        elif chunk_type == _GLB_BIN_CHUNK:
            bin_chunk = chunk_data
        # Chunks are 4-byte aligned.
        offset += 8 + chunk_length
        offset += (4 - (offset % 4)) % 4

    return json_chunk, bin_chunk


def _extract_glb_image_bytes(
    json_chunk:  dict,
    bin_chunk:   Optional[bytes],
    image_index: int,
) -> Optional[bytes]:
    """Resolve the encoded image bytes for ``json_chunk["images"][image_index]``.

    Handles all three glTF image sources:

      * embedded via ``bufferView`` (bytes from ``bin_chunk``),
      * base64-encoded ``data:`` URI,
      * external ``uri`` (not extractable from the GLB alone - returns None).

    Returns None on any lookup failure so callers can skip the slot.
    """
    images       = json_chunk.get("images", [])
    buffer_views = json_chunk.get("bufferViews", [])
    if not 0 <= image_index < len(images):
        return None
    img = images[image_index]

    # Path A: embedded in the BIN chunk via bufferView.
    if "bufferView" in img:
        bv_idx = img["bufferView"]
        if not 0 <= bv_idx < len(buffer_views) or bin_chunk is None:
            return None
        bv        = buffer_views[bv_idx]
        bv_offset = bv.get("byteOffset", 0)
        bv_length = bv["byteLength"]
        return bin_chunk[bv_offset : bv_offset + bv_length]

    # Path B: base64 data URI.
    uri = img.get("uri")
    if uri is not None and uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        return base64.b64decode(encoded)

    # Path C: external file reference - not resolvable from the GLB alone.
    return None


def _decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode encoded image bytes (PNG/JPEG/WebP/...) into a ``(H, W, C)``
    ``uint8`` numpy array via PIL.  Returns None when PIL is not available
    or when the bytes can't be decoded.

    Normalises exotic PIL modes (palette, L, LA, ...) into RGB or RGBA.
    """
    if Image is None:
        return None
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGBA" if "A" in pil_img.mode else "RGB")
    return np.asarray(pil_img, dtype=np.uint8)


def _load_texture_path_to_uint8(path: str) -> np.ndarray:
    """Load an image file into a ``(H, W, C) uint8`` numpy array via PIL.

    Used by :meth:`Object.to_dict` to materialize path-loaded textures
    into the saved file so the result is self-contained.  Mirrors the
    PIL-based path in :meth:`Object._bake_wired_texture` -- normalises
    palette / L / LA modes into RGB or RGBA.

    Raises:
        RuntimeError: If PIL is not available.
        FileNotFoundError: If the path does not resolve to an existing
            image file (raised by ``Image.open``).
    """
    if Image is None:
        raise RuntimeError(
            "PIL required to serialize texture paths via Object.to_dict; "
            "install Pillow or assign a numpy array to obj.texture before saving."
        )
    resolved = os.path.expandvars(os.path.expanduser(path))
    with Image.open(resolved) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")
        return np.asarray(img, dtype=np.uint8)


def _extract_glb_textures(
    file_path:      str,
    material_index: int = 0,
) -> dict:
    """Extract every PBR texture referenced by ``materials[material_index]``.

    Walks the canonical glTF metallic-roughness slots and returns a dict
    keyed by short slot name, mapping to a ``(H, W, C)`` ``uint8``
    numpy array (``C`` is 3 for RGB images, 4 for RGBA):

      * ``"base_color"`` -- ``pbrMetallicRoughness.baseColorTexture``
      * ``"metallic_roughness"`` -- ``pbrMetallicRoughness.metallicRoughnessTexture``
        (R = occlusion when combined, G = roughness, B = metallic)
      * ``"normal"`` -- ``normalTexture`` (tangent-space)
      * ``"occlusion"`` -- ``occlusionTexture``
      * ``"emissive"`` -- ``emissiveTexture``

    Slots absent from the material are omitted from the returned dict.
    Returns an empty dict on any parse / decode failure (e.g., invalid
    GLB, missing material, externally-referenced images, PIL not
    installed).

    Args:
        file_path: Path to the ``.glb`` file (``~`` and env vars OK).
        material_index: Which material to pull textures from (default
            first).  Negative indices count from the end.

    Note:
        Only the ``"base_color"`` slot is wired into the renderer today
        (see :meth:`Object.extract_texture_from_glb`).  The other slots
        are extracted here so a future multi-texture shader upgrade can
        consume them without rewriting this parser.
    """
    json_chunk, bin_chunk = _parse_glb(file_path)
    if json_chunk is None:
        return {}

    materials = json_chunk.get("materials", [])
    textures  = json_chunk.get("textures", [])
    if not materials or not -len(materials) <= material_index < len(materials):
        return {}
    mat = materials[material_index]

    result: dict = {}
    for slot_path, slot_name in _GLB_TEXTURE_SLOTS.items():
        parts = slot_path.split(".")
        node  = mat
        for p in parts[:-1]:
            node = node.get(p, {})
        tex_info = node.get(parts[-1])
        if tex_info is None:
            continue
        tex_idx = tex_info.get("index")
        if tex_idx is None or not 0 <= tex_idx < len(textures):
            continue
        img_idx = textures[tex_idx].get("source")
        if img_idx is None:
            continue
        img_bytes = _extract_glb_image_bytes(json_chunk, bin_chunk, img_idx)
        if img_bytes is None:
            continue
        decoded = _decode_image_bytes(img_bytes)
        if decoded is not None:
            result[slot_name] = decoded
    return result


# FBX material property name -> GLB-style slot name. Shares the vocabulary
# of ``_GLB_TEXTURE_SLOTS`` so a future multi-texture shader can consume
# both extractors uniformly. ``Bump`` and ``NormalMap`` both map to
# ``"normal"``; later wins are skipped via ``dict.setdefault`` so the
# more modern ``NormalMap`` takes precedence when both are authored.
_FBX_TEXTURE_SLOTS = {
    "DiffuseColor":  "base_color",
    "NormalMap":     "normal",
    "Bump":          "normal",
    "SpecularColor": "specular",
    "EmissiveColor": "emissive",
    "AmbientColor":  "occlusion",
}


def _walk_fbx_mesh_nodes(node, out: list) -> None:
    """Recursively collect FbxNode objects whose attribute is a mesh.

    Local copy of the helper in ``cgmath.geometry.mesh`` so this module
    keeps its FBX dependency self-contained on the texture-extraction
    side (the mesh helper is module-private and not part of any public
    API contract).
    """
    attr = node.GetNodeAttribute()
    if (
        attr is not None
        and attr.GetAttributeType() == _fbx.FbxNodeAttribute.EType.eMesh
    ):
        out.append(node)
    for i in range(node.GetChildCount()):
        _walk_fbx_mesh_nodes(node.GetChild(i), out)


def _find_fbx_file_texture(prop):
    """Return the first ``FbxFileTexture`` connected to an FBX material
    property, or None if no file texture is connected.

    Handles direct ``FbxFileTexture`` connections plus one level of
    ``FbxLayeredTexture`` indirection (common when materials author a
    stack of textures with blend modes).
    """
    for i in range(prop.GetSrcObjectCount()):
        obj = prop.GetSrcObject(i)
        cid = obj.GetClassId()
        if cid == _fbx.FbxFileTexture.ClassId:
            return obj
        if cid == _fbx.FbxLayeredTexture.ClassId:
            for j in range(obj.GetSrcObjectCount()):
                sub = obj.GetSrcObject(j)
                if sub.GetClassId() == _fbx.FbxFileTexture.ClassId:
                    return sub
    return None


def _load_fbx_texture_to_array(file_texture, fbx_path: str) -> Optional[np.ndarray]:
    """Resolve an FbxFileTexture to a ``(H, W, C) uint8`` numpy array.

    Tries the absolute path first (this is what the SDK populates after
    ``IMP_FBX_EXTRACT_EMBEDDED_DATA`` extracts embedded media to disk),
    then falls back to the relative path resolved against the FBX file's
    directory.  Returns None when neither resolves to a readable file.
    """
    path = file_texture.GetFileName()
    if not path or not os.path.exists(path):
        rel = file_texture.GetRelativeFileName()
        if rel:
            path = os.path.join(os.path.dirname(fbx_path), rel)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        img_bytes = f.read()
    return _decode_image_bytes(img_bytes)


def _extract_fbx_textures(
    file_path:      str,
    mesh_index:     int = 0,
    material_index: int = 0,
) -> dict:
    """Extract every texture referenced by ``mesh_index``'s
    ``material_index``-th material.

    Walks the FBX material-property slots known to map onto the GLB
    metallic-roughness vocabulary and returns a dict keyed by the same
    short slot name as :func:`_extract_glb_textures`:

      * ``"base_color"`` -- ``DiffuseColor``
      * ``"normal"`` -- ``NormalMap`` (preferred) or ``Bump``
      * ``"specular"`` -- ``SpecularColor``
      * ``"emissive"`` -- ``EmissiveColor``
      * ``"occlusion"`` -- ``AmbientColor``

    Slots absent from the material are omitted from the returned dict.
    Returns an empty dict on any failure (FBX SDK missing, PIL missing,
    bad indices, no materials, unresolvable texture path).

    Args:
        file_path: Path to the ``.fbx`` file (``~`` and env vars OK).
        mesh_index: Which mesh node to pull the material from. Negative
            indices count from the end.
        material_index: Which material on that mesh node. Negative
            indices count from the end.

    Note:
        Only the ``"base_color"`` slot is wired into the renderer today
        (see :meth:`Object.extract_texture_from_fbx`).  The other slots
        are extracted here so a future multi-texture shader upgrade can
        consume them without rewriting this parser.

        Enables ``IMP_FBX_EXTRACT_EMBEDDED_DATA`` on import so textures
        packed inside the FBX container resolve to readable temp files.
        Those files are written next to the FBX in a ``<name>.fbm/``
        directory (Maya / Motionbuilder convention) and left in place
        for subsequent calls.
    """
    if _fbx is None or Image is None:
        return {}

    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        return {}

    manager = _fbx.FbxManager.Create()
    try:
        ios = _fbx.FbxIOSettings.Create(manager, _fbx.IOSROOT)
        # ``IMP_FBX_EXTRACT_EMBEDDED_DATA`` is a C macro in the FBX SDK
        # that expands to the literal string ``"Import|ExtractEmbeddedData"``.
        # Some Python bindings (notably Maya's bundled fbx module) expose
        # only the ``EXP_*`` constants and omit the ``IMP_*`` ones, so we
        # fall back to the raw IOSettings path string when the attribute
        # is missing.
        ios.SetBoolProp(
            getattr(
                _fbx,
                "IMP_FBX_EXTRACT_EMBEDDED_DATA",
                "Import|ExtractEmbeddedData",
            ),
            True,
        )
        manager.SetIOSettings(ios)

        importer = _fbx.FbxImporter.Create(manager, "TexImporter")
        if not importer.Initialize(file_path, -1, manager.GetIOSettings()):
            importer.Destroy()
            return {}
        scene = _fbx.FbxScene.Create(manager, "TexExtractScene")
        if not importer.Import(scene):
            importer.Destroy()
            return {}
        importer.Destroy()

        mesh_nodes: list = []
        _walk_fbx_mesh_nodes(scene.GetRootNode(), mesh_nodes)
        if not mesh_nodes or not -len(mesh_nodes) <= mesh_index < len(mesh_nodes):
            return {}
        node = mesh_nodes[mesh_index]

        n_materials = node.GetMaterialCount()
        if n_materials == 0 or not -n_materials <= material_index < n_materials:
            return {}
        material = node.GetMaterial(material_index)

        result: dict = {}
        for fbx_prop, slot_name in _FBX_TEXTURE_SLOTS.items():
            prop = material.FindProperty(fbx_prop)
            if not prop.IsValid():
                continue
            tex = _find_fbx_file_texture(prop)
            if tex is None:
                continue
            decoded = _load_fbx_texture_to_array(tex, file_path)
            if decoded is not None:
                # setdefault so NormalMap wins over a later Bump entry
                result.setdefault(slot_name, decoded)
        return result
    finally:
        manager.Destroy()