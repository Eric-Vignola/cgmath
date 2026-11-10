"""
A scene for the raytracer: what an fbx holds, and the render settings.

A :class:`Scene` is the record ``cgmath.formats.load_fbx`` reads and
``save_fbx`` writes, in one object: the nodes, as a ``hierarchy`` at rest or
as their takes, a ``clip``; the ``meshes`` on the nodes; the ``uvs``,
``skins``, ``materials`` and blend shape targets (``morphs``) on the meshes;
the ``cameras`` and ``lights`` on their nodes; and the settings a render
takes. Every shape sits where the node its ``parent_uuid`` names puts it, at
identity when it names none; a skinned mesh is deformed by the pose, from
the bind pose ``hierarchy`` holds; a target moves the mesh's points by its
weight.

Minimum viable scene::

    from cgmath.render import Scene

    scene = Scene.load_fbx("hero.fbx")
    frame = scene.render(resolution=(640, 480))
    frame.save("hero.png")

Built by hand, shape by shape::

    from cgmath.geometry import CameraData, LightData, MaterialData
    from cgmath.hierarchy import TransformData
    from cgmath.render import Scene, aim

    scene = Scene("scene", resolution=(640, 480))
    scene.add(mesh, MaterialData(mesh_uuid=mesh.uuid, color=(0.8, 0.2, 0.1)))
    cam = TransformData("cam", translate=(0.0, 150.0, 400.0))
    aim(cam, (0.0, 80.0, 0.0))
    scene.add(cam, CameraData(parent_uuid=cam.uuid, focal_length=50.0))
    scene.render().save("scene.png")

If the scene has no camera, a default 3/4 elevated view is built and fit to
the scene. If it has no light and ``default_light`` is not False, a 3-point
rig sized to the scene lights it for the render.
"""

import atexit
import base64
import hashlib
import io
import json
import os
import pickle
import shutil
import struct
import tempfile
import warnings
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from cgmath.data import _strip_namespace, Array, Data, ImmutableArray as numpy_array
from cgmath.formats import FbxContents, load_fbx as _load_fbx, save_fbx as _save_fbx
from cgmath.geometry.camera import CameraData, CameraList
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.light import LightData, LightList
from cgmath.geometry.material import MaterialData, MaterialList
from cgmath.geometry.mesh import _corner_colors, load_glb as _load_glb, load_obj as _load_obj, MeshData, MeshList, UVData, UVList
from cgmath.geometry.morph_target import MorphData, MorphList
from cgmath.geometry.skin_weights import _read_skins_glb, SkinData, SkinList
from cgmath.geometry.utils import stream_to_matrix
from cgmath.hierarchy import ClipData, ClipList, generate_uuid, HierarchyData, TransformData
from cgmath.render.camera import (
    _default_camera_for_points,
    _inflate_points_about_center,
    look_at as _look_at,
)
from cgmath.render.texture import _load_texture

# opencv (optional) -- required by :meth:`Scene.imshow`
try:
    import cv2
except ImportError:
    cv2 = None

# PIL (optional) -- required by :meth:`Scene.to_image` and the glb textures
try:
    from PIL import Image
except ImportError:
    Image = None


class RenderInputs(NamedTuple):
    """what the raytracer renders: the per-mesh entries, the camera dict or
    None, the light dicts (:meth:`Scene._render_inputs`)"""

    entries: list
    camera:  Optional[dict]
    lights:  list


class Renderable(NamedTuple):
    """
    One mesh of a :class:`Scene` with what places, deforms and shades it,
    joined by uuid (:meth:`Scene.renderables`): a view, not a container.
    Its fields are the scene's own records, so an edit in place
    (``item.mesh.points[:] = ...``) edits the scene; its lists are new lists
    of those records, so appending to one changes nothing: a shape joins the
    scene through :meth:`Scene.add`. Built on every call, never cached.

    Attributes
    ----------
    mesh : MeshData
        The mesh itself.
    node : TransformData or None
        The node of :meth:`Scene.pose` that carries ``mesh.parent_uuid`` and
        places the mesh: the hierarchy's node itself or, in a scene holding
        takes alone, a copy at the active take's first frame. None for a
        mesh on no node, which sits at identity.
    uvs : UVList
        The UV sets naming the mesh, in the scene's order.
    skin : SkinData or None
        The first skin naming the mesh.
    joints : list of TransformData
        The skin's influences: the nodes of :meth:`Scene.pose` carrying their
        names, in the skin's order, a name no node carries left out. The
        skeleton is shared, so every mesh skinned to it lists the same nodes.
    materials : MaterialList
        The materials naming the mesh, their textures inside them.
    morphs : MorphList
        The blend shape targets naming the mesh.
    """

    mesh:      MeshData
    node:      Optional[TransformData]
    uvs:       UVList
    skin:      Optional[SkinData]
    joints:    list
    materials: MaterialList
    morphs:    MorphList

    @property
    def name(self) -> Optional[str]:
        """the mesh's name, else its node's (Maya's export leaves the FbxMesh
        unnamed and names the node), else None"""
        if self.mesh.name:
            return self.mesh.name
        return None if self.node is None else self.node.name


# the settings merged into render() / turntable() / animate() kwargs
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

# the shading settings, one value for every mesh of the scene
_SHADING_KEYS = (
    "base_color",
    "sample_method",
    "ambient",
    "twosided",
    "cast_shadows",
    "wireframe",
    "wireframe_color",
    "wireframe_thickness",
    "vertex_colors",
)

# what a mesh's colours do to its surface colour
_VERTEX_COLOR_MODES = ("off", "multiply", "replace")

# a shape's list field, by the class of what it holds
_LISTS = (
    ("meshes", MeshData, MeshList),
    ("uvs", UVData, UVList),
    ("skins", SkinData, SkinList),
    ("materials", MaterialData, MaterialList),
    ("cameras", CameraData, CameraList),
    ("lights", LightData, LightList),
    ("morphs", MorphData, MorphList),
)


@dataclass(repr=False, eq=False)
class Scene(Data):
    """
    What an fbx holds, and the render settings: nodes, meshes, UV sets,
    skins, materials, cameras, lights and blend shape targets, each shape
    placed by the node its ``parent_uuid`` names.

    Parameters
    ----------
    name : str, optional
        The scene's name. Default "scene".
    hierarchy : HierarchyData, optional
        The nodes at rest; with a clip, the bind pose the skins deform from.
    clip : ClipList, optional
        The takes over the nodes; the active one plays. A ClipData is taken
        as a list of one.
    meshes, uvs, skins, materials, cameras, lights, morphs : lists, optional
        The shapes, as ``cgmath.formats.load_fbx`` gives them: a MeshList,
        UVList, SkinList, MaterialList, CameraList, LightList, MorphList. One
        shape, or a list of them, is taken as the list.
    default_camera : str, optional
        The name of the camera a render looks through; None for the first.
    aspect_ratio : float, optional
        Reserved; the resolution gives the aspect.
    resolution : (2,) int32, optional
        The frame's width and height. Default (500, 500); None falls through
        to the per-call keyword and the renderer's default.
    samples_per_pixel : int, optional
        1 or a perfect square (4, 9, 16 ...); None for the renderer's default.
    background : (4,) float64, optional
        RGBA in 0 to 1 behind the scene; RGB is taken opaque; None for
        transparent black.
    autofit, default_light, return_depth : bool, optional
        Fit the camera to the scene, light an unlit scene with the 3-point
        rig, keep a depth buffer; None for the renderer's defaults.
    angle_of_view : float, optional
        The vertical angle of view, in degrees, of the default camera a
        scene without a camera is seen through.
    fit_padding : float, optional
        The margin autofit leaves around the scene, 1 none; None picks 1.15
        for bezier sampling, else 1.
    base_color : (3,) float64
        The colour of a face no material covers. Default 0.7 grey.
    sample_method : str
        "bilinear" or "bezier" (PN quads), every mesh. Default bilinear.
    ambient : float
        The ambient term, added once after the lights. Default 0.8.
    twosided : bool
        Back faces hit when True. Default True.
    cast_shadows : bool
        Reserved. Default True.
    wireframe : bool
        Edges baked over every textured mesh's colour map, or over its base
        colour on its first UV set. Default False.
    wireframe_color : (3,) int32
        The edges' colour, 0 to 255. Default black.
    wireframe_thickness : int
        The edges' width, in texels. Default 1.
    vertex_colors : str
        What a mesh's ``colors`` do to its surface colour: "off", "multiply"
        (the material colour and map times the vertex colour, the game
        engines' way) or "replace" (the vertex colour alone, Maya's Color
        Set display); blended over a face as the UVs are. The alpha channel
        is carried, not rendered. Default off.

    A render through :meth:`render`, :meth:`to_image` or :meth:`imshow`
    keeps its :attr:`frame`, dropped when a setting, a shape, a node or the
    frame rendered changes (:meth:`_signature`); an edit in place of
    a mesh's points or a texture's pixels is not seen: render again. A
    turntable or an animation keeps its frames on disk for a second encode
    (:meth:`clear_render_cache`).
    """

    name:                Optional[str] = "scene"
    hierarchy:           Optional[HierarchyData] = None
    clip:                Optional[ClipList] = None
    meshes:              Optional[MeshList] = None
    uvs:                 Optional[UVList] = None
    skins:               Optional[SkinList] = None
    materials:           Optional[MaterialList] = None
    cameras:             Optional[CameraList] = None
    lights:              Optional[LightList] = None
    morphs:              Optional[MorphList] = None
    default_camera:      Optional[str] = None
    aspect_ratio:        Optional[float] = None
    resolution:          Optional[Array(np.int32, 2)] = numpy_array(np.array([500, 500], dtype=np.int32))
    samples_per_pixel:   Optional[int] = None
    background:          Optional[Array(np.float64, 4)] = None
    autofit:             Optional[bool] = None
    default_light:       Optional[bool] = None
    return_depth:        Optional[bool] = None
    angle_of_view:       Optional[float] = None
    fit_padding:         Optional[float] = None
    base_color:          Array(np.float64, 3) = numpy_array([0.7, 0.7, 0.7])
    sample_method:       str = "bilinear"
    ambient:             float = 0.8
    twosided:            bool = True
    cast_shadows:        bool = True
    wireframe:           bool = False
    wireframe_color:     Array(np.int32, 3) = numpy_array(np.zeros(3, dtype=np.int32))
    wireframe_thickness: int = 1
    vertex_colors:       str = "off"

    # the render caches: class defaults, never saved, a loaded Scene starts without
    _frame              = None
    _render_signature   = None
    _rendered_frame     = None
    _render_cache_owner = None
    _deformers          = None
    _textures           = None

    # the private state a Scene holds beside its fields: the caches above
    _STATE = ("_frame", "_render_signature", "_rendered_frame", "_render_cache_owner", "_deformers", "_textures")

    def __setattr__(self, name: str, value) -> None:
        if name == "hierarchy" and value is not None:
            if isinstance(value, (ClipData, ClipList)):
                raise TypeError(f"Scene.hierarchy is a HierarchyData at rest; pass a {type(value).__name__} as clip=")
            if isinstance(value, TransformData):
                value = HierarchyData([value])
            elif not isinstance(value, HierarchyData):
                value = HierarchyData(list(value))
        elif name == "clip" and value is not None:
            if isinstance(value, ClipData):
                value = ClipList([value])
            elif not isinstance(value, ClipList):
                raise TypeError(f"Scene.clip is a ClipList or a ClipData, not a {type(value).__name__}")
        elif name == "background":
            value = _normalize_rgba(value, "Scene.background")
        elif name == "samples_per_pixel":
            value = _validate_samples_per_pixel(value, field_name="Scene.samples_per_pixel")
        elif name == "sample_method":
            value = str(_enum_value(value))
        elif name == "vertex_colors":
            value = str(value).lower()
            if value not in _VERTEX_COLOR_MODES:
                raise ValueError(f"Scene.vertex_colors is one of {', '.join(_VERTEX_COLOR_MODES)}, not {value!r}")
        elif name == "default_camera" and value is not None:
            value = str(value)
        else:
            for field, item, cls in _LISTS:
                if name == field and value is not None:
                    value = _as_list(value, item, cls, f"Scene.{field}")
                    break
        super().__setattr__(name, value)

    # -- building --------------------------------------------------------------------

    def add(self, *items) -> "Scene":
        """
        Adds each item to the list it belongs in, by its class: a
        TransformData to the hierarchy (made when the scene has none), a
        ClipData or ClipList as the clip, a MeshData, UVData, SkinData,
        MaterialData, CameraData, LightData or MorphData to its list, and a
        list of any of them item by item. Returns the scene, so calls chain.
        TypeError for anything else.
        """
        for item in items:
            if isinstance(item, (ClipData, ClipList)):
                self.clip = item
            elif isinstance(item, (HierarchyData, TransformData)):
                nodes = [item] if isinstance(item, TransformData) else list(item)
                if self.hierarchy is None:
                    self.hierarchy = HierarchyData()
                for node in nodes:
                    self.hierarchy.append(node)
            elif isinstance(item, UVData):
                self.uvs = (list(self.uvs) if self.uvs is not None else []) + [item]
            else:
                for field, kind, _ in _LISTS:
                    if isinstance(item, kind) and kind is not UVData:
                        held = getattr(self, field)
                        setattr(self, field, (list(held) if held is not None else []) + [item])
                        break
                else:
                    if isinstance(item, (list, tuple)) or type(item).__name__.endswith("List"):
                        for held in item:
                            self.add(held)
                        continue
                    raise TypeError(
                        f"Scene.add takes nodes, a clip, meshes, UV sets, skins, materials, cameras, lights and blend "
                        f"shape targets, not a {type(item).__name__}"
                    )
        return self

    def renew_uuids(self) -> dict:
        """
        Fresh uuids on every node and shape, every link rewired, in place:
        the records stay the same objects. The hierarchy and the takes renew
        alike, sharing the nodes' uuids as before; each shape's ``uuid``,
        ``parent_uuid`` and ``mesh_uuid`` follow; the render caches are
        dropped. Returns ``{old: new}``.

        For a scene built a second time in Maya beside the first: rig gives
        the nodes it makes the data's uuids, and Maya wants them unique.
        """
        mapping = {}
        if self.hierarchy is not None:
            self.hierarchy.renew_uuids(mapping)
        if self.clip is not None:
            self.clip.renew_uuids(mapping)
        for field, _, _ in _LISTS:
            for record in getattr(self, field) or []:
                for link in ("uuid", "parent_uuid", "mesh_uuid"):
                    value = getattr(record, link, None)
                    if isinstance(value, str):
                        setattr(record, link, mapping.setdefault(value, generate_uuid()))
        self._deformers = None
        self._textures  = None
        _clear_render_cache(self)
        return mapping

    # -- placement -------------------------------------------------------------------

    def pose(self, frame: Optional[int] = None) -> Optional[HierarchyData]:
        """
        The nodes a render places the shapes by: the active take at
        ``frame`` (its first frame for None) when the scene has a clip, else
        the hierarchy; None without either.
        """
        if self.clip is not None and len(self.clip):
            take = self.clip.active
            return take.at(take.start_frame if frame is None else frame)
        return self.hierarchy

    def camera(self, name: Optional[str] = None) -> Tuple[Optional[CameraData], Optional[str]]:
        """
        ``(camera, node uuid)``: the camera named ``name``, else the one
        :attr:`default_camera` names, else the first; its node's uuid, None
        for a camera on no node. ``(None, None)`` for a scene with no camera.
        """
        if not self.cameras:
            return None, None
        chosen = name or self.default_camera
        if chosen is not None:
            for camera in self.cameras:
                if camera.name == chosen:
                    return camera, camera.parent_uuid
        return self.cameras[0], self.cameras[0].parent_uuid

    # -- the meshes with what deforms and shades them --------------------------------

    def renderables(self) -> List[Renderable]:
        """
        One :class:`Renderable` per mesh, in the meshes' order: the mesh, the
        node of :meth:`pose` placing it, its UV sets, skin, the skin's joints,
        materials and blend shape targets, joined by uuid. Views over the
        scene's records, built on every call: a shape added since shows in
        the next call, never in an item already held. ``[]`` for a scene with
        no mesh.
        """
        return self._joined(self.pose())

    def renderable(self, name: str) -> Renderable:
        """
        The one renderable whose mesh, or whose node, is named ``name``
        (Maya's export names the node and leaves the mesh unnamed);
        ValueError for none or several.
        """
        found = [
            item for item in self.renderables()
            if item.mesh.name == name or (item.node is not None and item.node.name == name)
        ]
        if len(found) != 1:
            raise ValueError(
                f"Scene.renderable: {len(found)} meshes are named {name!r} or sit on a node named so; one expected"
            )
        return found[0]

    def _joined(self, pose) -> List[Renderable]:
        """the renderables, each mesh's node and joints taken from ``pose``
        (None: no node, no joints), one pass over each list"""
        nodes, names = {}, {}
        for node in pose or []:
            nodes[node.uuid] = node
            names.setdefault(node.name, node)
        held = {}
        for field in ("uvs", "skins", "materials", "morphs"):
            groups = held[field] = {}
            for record in getattr(self, field) or []:
                groups.setdefault(record.mesh_uuid, []).append(record)
        out = []
        for mesh in self.meshes or []:
            skins  = held["skins"].get(mesh.uuid)
            skin   = skins[0] if skins else None
            joints = [] if skin is None else [names[n] for n in skin.influences if n in names]
            out.append(
                Renderable(
                    mesh,
                    nodes.get(mesh.parent_uuid),
                    UVList(held["uvs"].get(mesh.uuid, [])),
                    skin,
                    joints,
                    MaterialList(held["materials"].get(mesh.uuid, [])),
                    MorphList(held["morphs"].get(mesh.uuid, [])),
                )
            )
        return out

    def placed_meshes(self, frame: Optional[int] = None) -> List[Tuple[MeshData, np.ndarray]]:
        """
        ``[(mesh, world)]`` of every shown mesh at ``frame``: a mesh on a
        hidden node is left out; a skinned or morphed mesh is a copy moved
        to its pose, in world space, its world identity; any other is itself
        with its node's world (4, 4), row vectors, identity for a mesh on no
        node.
        """
        pose = self.pose(frame)
        return [(mesh, world) for mesh, world, _ in self._placed(pose)]

    def union_points(self, frame: Optional[int] = None) -> np.ndarray:
        """every shown mesh's points at ``frame``, in world space, as one
        (N, 3) array, (0, 3) for none"""
        chunks = [_world_points(mesh, world) for mesh, world in self.placed_meshes(frame)]
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))

    def union_aabb(self, frame: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """``(min corner, max corner)`` of :meth:`union_points`, zeros for none"""
        points = self.union_points(frame)
        if not len(points):
            return np.zeros(3), np.zeros(3)
        return points.min(axis=0), points.max(axis=0)

    def _placed(self, pose) -> list:
        """``[(mesh, world, renderable)]`` of every shown mesh placed by
        ``pose`` (:meth:`placed_meshes`): the mesh as shown, a copy deformed
        by its skin or its targets or the mesh itself, its world, and its
        :class:`Renderable` (:meth:`_joined`), whose ``mesh`` is the original"""
        worlds, shown = _worlds(pose)
        found = []
        for item in self._joined(None):
            mesh = item.mesh
            if mesh.parent_uuid is not None and mesh.parent_uuid in shown and not shown[mesh.parent_uuid]:
                continue
            world  = worlds.get(mesh.parent_uuid, np.eye(4))
            points = _morphed_points(mesh, item.morphs)
            if item.skin is not None:
                deformer = self._deformer(mesh, item.skin)
                if pose is None:
                    raise ValueError(f"Scene: {_mesh_label(mesh)} is skinned, which takes a pose: the scene holds no nodes")
                points = deformer.apply(pose, mesh.points if points is None else points)
                world  = np.eye(4)
            if points is None:
                found.append((mesh, world, item))
                continue
            moved        = mesh.copy()
            moved.points = np.asarray(points, dtype=np.float64)
            if moved.normals is not None:
                moved.normals = None
            found.append((moved, world, item))
        return found

    def _deformer(self, mesh, skin) -> SkinDeformData:
        """the deformer of a skinned mesh, bound from the bind pose
        (:attr:`hierarchy`), kept while mesh, skin and hierarchy stay"""
        if self.hierarchy is None:
            raise ValueError(
                f"Scene: {_mesh_label(mesh)} is skinned, which takes the bind pose; set scene.hierarchy to it (a file's "
                f"load_fbx gives it with the skins)"
            )
        key = (mesh.uuid, id(skin), id(skin.weights), id(self.hierarchy), id(mesh.points))
        if self._deformers is None:
            self._deformers = {}
        held = self._deformers.get(mesh.uuid)
        if held is not None and held[0] == key:
            return held[1]
        deformer = SkinDeformData(mesh, skin, bind_rig=self.hierarchy, name=skin.name)
        deformer.bind()
        self._deformers[mesh.uuid] = (key, deformer)
        return deformer

    # -- what the raytracer takes ---------------------------------------------------

    def _render_inputs(self, frame: Optional[int] = None) -> RenderInputs:
        """the entries, camera dict and light dicts :func:`cgmath.render.raytracer.render`
        consumes for the scene at ``frame``"""
        pose = self.pose(frame)
        worlds, shown = _worlds(pose)
        entries = []
        for mesh, world, item in self._placed(pose):
            entries.append(self._entry(mesh, item, world))
        return RenderInputs(entries, self._camera_dict(worlds), self._light_dicts(worlds, shown))

    def _entry(self, mesh, item, world) -> dict:
        """one mesh's entry, ``mesh`` as shown and ``item`` its renderable:
        its materials, each the faces it covers (None for all), its colour
        times its diffuse, its colour map through the UV set it names (the
        mesh's first for None) and its wrap; a mesh with no material takes
        the scene's base colour on every face"""
        original  = item.mesh
        sets      = list(item.uvs)
        materials = []
        for material in item.materials:
            uv = next((x for x in sets if x.name == material.uv_set), sets[0] if sets else None)
            materials.append({
                "faces":      None if material.faces is None else material.covers(original),
                "base_color": np.asarray(material.color, dtype=np.float64) * float(material.diffuse),
                "tex":        self._texture(material, uv),
                "wrap":       material.wrap_u,
                "uv":         uv,
            })
        if not materials:
            uv = sets[0] if sets else None
            materials.append({
                "faces":      None,
                "base_color": np.asarray(self.base_color, dtype=np.float64),
                "tex":        self._texture(None, uv),
                "wrap":       "repeat",
                "uv":         uv,
            })
        for material in materials:
            if material["tex"] is not None and material["uv"] is None:
                raise ValueError(f"Scene: {_mesh_label(original)} has a texture but no UV set to map it")
        return {
            "mesh":          mesh,
            "uv":            materials[0]["uv"],
            "tex":           materials[0]["tex"],
            "base_color":    np.asarray(self.base_color, dtype=np.float64),
            "sample_method": self.sample_method,
            "wrap":          materials[0]["wrap"],
            "ambient":       float(self.ambient),
            "twosided":      bool(self.twosided),
            "world_matrix":  np.asarray(world, dtype=np.float64).T,
            "materials":     materials,
            "colors":        self._colors(original),
            "vertex_colors": self.vertex_colors,
        }

    def _colors(self, mesh) -> Optional[tuple]:
        """``(colors, table)`` for the raytracer: the mesh's RGBA colours,
        float32, and the colour each face corner names as an (F, slots)
        table, -1 past a face's corners, as a UV set's geometry names its
        points; None without colours, or with ``vertex_colors`` off"""
        if self.vertex_colors == "off" or mesh.colors is None:
            return None
        colors, ids = _corner_colors(mesh)
        table = stream_to_matrix(np.asarray(ids, dtype=np.int32), np.asarray(mesh.counts, dtype=np.int32))
        return np.asarray(colors, dtype=np.float32), np.asarray(table)

    def _texture(self, material, uv) -> Optional[np.ndarray]:
        """a material's colour map as the renderer samples it, (H, W, 3)
        float32 in 0 to 1, with the wireframe baked over it (or over the
        base colour) when the scene draws one and ``uv`` maps the mesh;
        None for no map and no wireframe. Kept by the pixels' identity"""
        pixels = None if material is None else material.texture
        wired  = bool(self.wireframe) and uv is not None
        if pixels is None and not wired:
            return None
        color = np.asarray(self.base_color if material is None else material.color, dtype=np.float64)
        key = (id(pixels), id(uv) if wired else None, tuple(color.tolist()) if wired else None,
               tuple(np.asarray(self.wireframe_color).tolist()) if wired else None, int(self.wireframe_thickness) if wired else None)
        if self._textures is None:
            self._textures = {}
        held = self._textures.get(key)
        if held is not None and held[0] is pixels:
            return held[1]
        if wired:
            loaded = _bake_wireframe(uv, pixels, color, self.wireframe_color, self.wireframe_thickness)
        else:
            loaded = _load_texture(_three_channels(pixels))
        self._textures[key] = (pixels, loaded)
        return loaded

    def _camera_dict(self, worlds: dict) -> Optional[dict]:
        camera, node = self.camera()
        if camera is None:
            return None
        world = worlds.get(node, np.eye(4))
        return {
            "world_matrix":    np.asarray(world, dtype=np.float64).T,
            "angle_of_view":   float(camera.vertical_angle_of_view),
            "is_orthographic": bool(camera.orthographic),
            "ortho_height":    None,
            "ortho_width":     float(camera.orthographic_width),
        }

    def _light_dicts(self, worlds: dict, shown: dict) -> list:
        """the lights as the renderer takes them: a point light with falloff
        when its decay is not none, a directional light infinite, a spot a
        point light with one warning (the cone is not rendered); a light on
        a hidden node left out"""
        found = []
        spots = []
        for light in self.lights or []:
            node = light.parent_uuid
            if node is not None and node in shown and not shown[node]:
                continue
            world = np.asarray(worlds.get(node, np.eye(4)), dtype=np.float64)
            if light.kind == "spot":
                spots.append(light.name or "a spot light")
            direction = -world[2, :3]
            found.append({
                "kind":      "infinite" if light.kind == "directional" else "point",
                "position":  tuple(float(x) for x in world[3, :3]),
                "direction": tuple(float(x) for x in direction / max(float(np.linalg.norm(direction)), 1e-12)),
                "color":     tuple(float(x) for x in light.color),
                "intensity": float(light.intensity),
                "falloff":   int(light.decay_rate) != 0,
            })
        if spots:
            warnings.warn(f"Scene: {', '.join(spots)} rendered as point lights; the raytracer draws no cone", stacklevel=3)
        return found

    # -- rendering --------------------------------------------------------------------

    def _merge_render_config(self, kwargs: dict) -> dict:
        """per-call keyword, else the scene's setting, else nothing"""
        defaults = {k: getattr(self, k) for k in _RENDER_CONFIG_KEYS}
        merged   = {k: v for k, v in defaults.items() if v is not None}
        merged.update(kwargs)
        return merged

    def render(self, frame: Optional[int] = None, output: Optional[str] = None, **kwargs):
        """
        The scene rendered as a :class:`Frame`: the shapes placed by
        :meth:`pose` at ``frame`` (the active take's first frame, or the
        hierarchy), seen through :meth:`camera` or a default view fit to the
        scene, lit by its lights or, with none and ``default_light`` not
        False, a 3-point rig sized to the scene. The scene's settings fill
        the keywords of :func:`cgmath.render.raytracer.render` not given;
        ``threads=`` reaches it. Saved to ``output`` when given. The frame
        is kept as :attr:`frame`.
        """
        from cgmath.render.raytracer import render

        merged = self._merge_render_config(kwargs)
        if "background" in merged:
            merged["background"] = _normalize_rgba(merged["background"], "render(background=...)")
        if merged.get("default_light", True) is not False and not self.lights:
            points = self.union_points(frame)
            if points.size:
                merged.setdefault("point_light", _default_lighting_rig_for(points))
                merged["default_light"] = False
        result = render(scene=self, frame=frame, **merged)
        if output is not None:
            result.save(output)
        self._rendered_frame = frame
        self.frame           = result
        return result

    def to_image(self, resolution: Tuple[int, int] = (500, 500)) -> "Image":
        """the frame as a PIL image, rendered at ``resolution`` when none is kept"""
        if Image is None:
            raise RuntimeError("PIL required to generate PIL.Image data")
        if self.frame is None:
            self.render(resolution=resolution)
        return Image.fromarray(self.frame.array)

    def imshow(self, resolution: Tuple[int, int] = (500, 500)) -> None:
        """the frame in an OpenCV window, rendered at ``resolution`` when none is kept"""
        if cv2 is None:
            raise RuntimeError("opencv required to show images")
        if self.frame is None:
            self.render(resolution=resolution)
        buffer = cv2.cvtColor(self.frame.array, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.name or "Scene", buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -- the frame cache --------------------------------------------------------------

    def _signature(self, frame: Optional[int] = None) -> tuple:
        """
        What the frame rendered depends on, as a hashable tuple: every
        setting's value, each shape's fields and the identity of its arrays,
        each node's world at ``frame`` and whether it is shown. An edit in
        place of an array is not seen.
        """
        pose = self.pose(frame)
        worlds, shown = _worlds(pose)

        def placed(uuid):
            return (None if uuid is None else _frozen(worlds.get(uuid)), shown.get(uuid, True))

        settings = tuple(_frozen(getattr(self, k)) for k in _RENDER_CONFIG_KEYS + _SHADING_KEYS)
        parts    = [settings, self.default_camera, self.aspect_ratio, frame, len(pose) if pose is not None else None]
        parts.append(tuple(
            (m.uuid, id(m.points), id(m.indices), id(m.counts), id(m.colors), id(m.color_indices), placed(m.parent_uuid))
            for m in self.meshes or []
        ))
        parts.append(tuple((uv.name, uv.mesh_uuid, id(uv.points), id(uv.indices)) for uv in self.uvs or []))
        parts.append(tuple((s.mesh_uuid, id(s.weights), tuple(s.influences)) for s in self.skins or []))
        if self.skins:
            # a skinned mesh follows every joint: the whole pose
            parts.append(tuple((uuid, _frozen(worlds[uuid])) for uuid in sorted(worlds)))
        parts.append(tuple(
            (m.name, m.mesh_uuid, None if m.faces is None else id(m.faces), m.kind, _frozen(m.color), m.diffuse,
             id(m.texture), m.uv_set, m.wrap_u, m.wrap_v)
            for m in self.materials or []
        ))
        parts.append(tuple(
            (c.name, c.focal_length, c.horizontal_film_aperture, c.vertical_film_aperture, c.orthographic,
             c.orthographic_width, placed(c.parent_uuid))
            for c in self.cameras or []
        ))
        parts.append(tuple(
            (x.name, x.kind, _frozen(x.color), x.intensity, x.decay_rate, placed(x.parent_uuid)) for x in self.lights or []
        ))
        parts.append(tuple(
            (m.name, m.deformer, m.mesh_uuid, m.weight, m.at, id(m.offsets), id(m.indices)) for m in self.morphs or []
        ))
        return tuple(parts)

    @property
    def frame(self):
        """the last render, None when none is kept or a render input changed since"""
        if self._frame is None:
            return None
        if self._render_signature is not None and self._render_signature != self._render_signature_now():
            self._frame            = None
            self._render_signature = None
            return None
        return self._frame

    @frame.setter
    def frame(self, value) -> None:
        self._frame            = value
        self._render_signature = None if value is None else self._render_signature_now()

    def _render_signature_now(self) -> tuple:
        return self._render_signature_of(self._rendered_frame)

    def _render_signature_of(self, frame) -> tuple:
        try:
            return self._signature(frame)
        except ValueError:
            # a scene that can not be posed renders nothing: no frame to keep
            return ("unposed",)

    @property
    def buffer(self) -> Optional[np.ndarray]:
        """the kept frame's pixels, None for none"""
        return self.frame.array if self.frame is not None else None

    def clear_render_cache(self) -> None:
        """drops the frames kept from the last turntable or animation, so the
        next one renders again: needed after an edit in place of points or
        pixels, which the cache does not see"""
        _clear_render_cache(self)

    def configure(self, **kwargs) -> "Scene":
        """sets the render and shading settings given, None values skipped;
        returns the scene. AttributeError for a name that is no setting"""
        allowed = _RENDER_CONFIG_KEYS + _SHADING_KEYS + ("default_camera", "aspect_ratio")
        for name, value in kwargs.items():
            if value is None:
                continue
            if name not in allowed:
                raise AttributeError(f"Scene.configure: unknown setting {name!r}; the settings are {allowed}")
            setattr(self, name, value)
        return self

    # -- sequences ---------------------------------------------------------------------

    def _base_camera(self, all_pts: np.ndarray, camera_name, render_kwargs: dict, worlds: dict):
        """the orientation and lens a sequence fits to: the scene's camera
        at ``worlds``, else a default 3/4 view; ``(camera matrix column
        major, its rotation, the angle of view, width, height)``"""
        camera, node = self.camera(camera_name)
        if camera is not None:
            base_cam_col = np.asarray(worlds.get(node, np.eye(4)), dtype=np.float64).T
            base_aov     = float(camera.vertical_angle_of_view)
        else:
            base_cam_col = _default_camera_for_points(all_pts)
            base_aov     = 35.0
        res = render_kwargs.get("resolution", (512, 384))
        aov = float(render_kwargs.get("angle_of_view", base_aov))
        return base_cam_col, base_cam_col[:3, :3], aov, int(res[0]), int(res[1])

    def _resolve_mp4_background(self, background=None) -> Tuple[float, float, float]:
        if background is not None:
            return tuple(float(v) for v in background)
        if self.background is not None:
            return tuple(float(v) for v in np.asarray(self.background)[:3])
        return (0.0, 0.0, 0.0)

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
        """
        The camera orbiting the scene, ``n_frames`` frames from
        ``start_angle`` to ``end_angle`` about the world ``rotation_axis``
        through the scene's centroid, the scene itself untouched. With
        ``fit`` "auto" the camera is placed once so the scene fits the frame
        at every sampled angle (``n_fit_samples``, the frame count by
        default); "static" keeps the camera where it is. The base camera is
        the scene's (``camera_name``, the default, the first), else a 3/4
        view. ``output_pattern`` picks the encoder by its extension: .mp4,
        .mov, .m4v, .avi, .gif one file, anything else an image sequence
        with a ``{frame:04d}`` placeholder. The frames are kept for a second
        encode. Returns the file's path, or the frames' paths.
        """
        render_kwargs = self._merge_render_config(render_kwargs)
        owner         = self._render_cache_owner or self
        key = _sequence_key(
            self,
            (
                "turntable", int(n_frames), rotation_axis, start_angle, end_angle,
                fit, camera_name, n_fit_samples, _key_render_kwargs(render_kwargs),
            ),
        )
        frames = _cached_sequence(owner, key)
        if frames is None:
            frames = _record_sequence(
                owner,
                key,
                self._turntable_frames(
                    render_kwargs, n_frames, rotation_axis, start_angle, end_angle, fit, camera_name, n_fit_samples, verbose,
                ),
            )
        return _encode_frame_stream(
            frames, output_pattern, fps=int(fps), mp4_background=self._resolve_mp4_background(background), verbose=verbose,
        )

    def _turntable_frames(
        self, render_kwargs, n_frames, rotation_axis, start_angle, end_angle, fit, camera_name, n_fit_samples, verbose,
    ):
        render_kwargs = dict(render_kwargs)
        all_pts       = self.union_points()
        if not len(all_pts):
            raise ValueError("turntable: scene has no shown mesh")
        centroid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
        worlds, _ = _worlds(self.pose())
        base_cam_col, R_cam, aov, width, height = self._base_camera(all_pts, camera_name, render_kwargs, worlds)

        if fit == "auto":
            pts_centered = _inflate_points_about_center(all_pts - centroid, _resolve_turntable_fit_padding(render_kwargs, self))
            n_samples    = int(n_fit_samples) if n_fit_samples is not None else int(n_frames)
            chosen_cam_col = _fit_camera_over_orbit(
                pts_centered, R_cam, base_cam_col, centroid, aov, width, height, rotation_axis, start_angle, end_angle,
                max(n_samples, 1), verbose=verbose,
            )
        elif fit == "static":
            chosen_cam_col = base_cam_col
        else:
            raise ValueError(f"fit must be 'auto' or 'static', got {fit!r}")

        render_kwargs.setdefault("autofit", False)
        render_kwargs.setdefault("angle_of_view", aov)
        span = float(end_angle - start_angle)

        def _frame_stream():
            for fi in range(int(n_frames)):
                angle                          = start_angle + span * fi / max(int(n_frames), 1)
                render_kwargs["camera_matrix"] = _orbit_camera_col(chosen_cam_col, centroid, rotation_axis, angle)
                yield self.render(**render_kwargs)

        return _frame_stream()

    def animate(
        self,
        output_pattern: str                                  = "animation.{frame:04d}.png",
        clip                                                 = None,
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
        """
        The scene's clip rendered frame by frame: the active take of
        ``clip`` (a ClipData or ClipList), or of the scene's own, poses the
        nodes at each frame; skinned meshes deform to it, any other mesh,
        camera and light rides its node. With ``fit`` "auto" the camera is
        placed once to hold every sampled pose (``n_pose_samples``, 10 or a
        tenth of the clip by default); "static" keeps the scene's camera.
        The orbit arguments turn the camera over the clip as
        :meth:`turntable` does. ``fps`` defaults to the clip's. The output
        and the kept frames are as :meth:`turntable`'s. ValueError for no
        clip, a clip of no frame or a skin naming joints the clip lacks.
        """
        if clip is None:
            if self.clip is None or not len(self.clip):
                raise ValueError("animate: the scene holds no clip; pass clip= or set scene.clip")
            clip = self.clip
        take          = _one_take(clip)
        render_kwargs = self._merge_render_config(render_kwargs)
        frame_count   = int(take.frame_count)
        if frame_count <= 0:
            raise ValueError("animate: the clip has no frames")
        if not self.meshes:
            raise ValueError("animate: the scene holds no mesh")
        for item in self._joined(None):
            if item.skin is not None:
                try:
                    self._deformer(item.mesh, item.skin).pose_indices(take)
                except ValueError as exc:
                    raise ValueError(
                        f"animate: {_mesh_label(item.mesh)} is skinned to joints the clip does not hold ({exc})"
                    ) from exc
        if fps is None:
            fps = int(round(float(getattr(take, "fps", 30.0)) or 30.0))

        world = np.asarray(take.frames.world_matrix, dtype=np.float64)
        owner = self._render_cache_owner or self
        key = _sequence_key(
            self,
            (
                "animate", world, take.take, int(take.start_frame), fit, rotation_axis, start_angle, end_angle, camera_name,
                n_pose_samples, n_fit_samples, _key_render_kwargs(render_kwargs),
            ),
        )
        frames = _cached_sequence(owner, key)
        if frames is None:
            frames = _record_sequence(
                owner,
                key,
                self._animate_frames(
                    take, frame_count, fit, rotation_axis, start_angle, end_angle, camera_name, n_pose_samples,
                    n_fit_samples, render_kwargs, verbose,
                ),
            )
        return _encode_frame_stream(
            frames, output_pattern, fps=int(fps), mp4_background=self._resolve_mp4_background(background), verbose=verbose,
        )

    def _animate_frames(
        self, take, frame_count, fit, rotation_axis, start_angle, end_angle, camera_name, n_pose_samples, n_fit_samples,
        render_kwargs, verbose,
    ):
        render_kwargs = dict(render_kwargs)
        frames        = int(take.start_frame) + np.arange(frame_count)
        chosen_cam_col, centroid = self._fit_animated_camera(
            take, frames, fit, rotation_axis, start_angle, end_angle, camera_name, n_pose_samples, n_fit_samples,
            render_kwargs, verbose,
        )
        render_kwargs.setdefault("autofit", False)
        span = float(end_angle - start_angle)

        def _frame_stream():
            for index, frame in enumerate(frames.tolist()):
                angle                          = start_angle + span * index / max(frame_count, 1)
                render_kwargs["camera_matrix"] = _orbit_camera_col(chosen_cam_col, centroid, rotation_axis, angle)
                yield self._render_take_frame(take, frame, render_kwargs)

        return _frame_stream()

    def _render_take_frame(self, take, frame, render_kwargs):
        """one frame of an animation: the scene at ``frame`` of ``take``,
        whatever clip the scene holds, through :meth:`render`"""
        held      = self.clip
        self.clip = ClipList([take])
        try:
            return self.render(frame=frame, **render_kwargs)
        finally:
            self.clip = held

    def _fit_animated_camera(
        self, take, frames, fit, rotation_axis, start_angle, end_angle, camera_name, n_pose_samples, n_fit_samples,
        render_kwargs, verbose,
    ):
        """the camera matrix and orbit centroid of :meth:`animate`: the
        shown meshes' points at the sampled frames, as one cloud, fit once"""
        if fit not in ("auto", "static"):
            raise ValueError(f"animate: fit must be 'auto' or 'static', got {fit!r}")
        held      = self.clip
        self.clip = ClipList([take])
        try:
            samples = frames[_pose_sample_frames_for(len(frames), n_pose_samples)]
            clouds  = [self.union_points(int(frame)) for frame in samples.tolist()]
            all_pts = np.concatenate([c for c in clouds if len(c)], axis=0) if any(len(c) for c in clouds) else np.zeros((0, 3))
            if not len(all_pts):
                raise ValueError("animate: scene has no shown mesh")
            centroid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
            worlds, _ = _worlds(self.pose(int(frames[0])))
        finally:
            self.clip = held
        base_cam_col, R_cam, aov, width, height = self._base_camera(all_pts, camera_name, render_kwargs, worlds)
        render_kwargs.setdefault("angle_of_view", aov)
        if fit == "static":
            return base_cam_col, centroid
        pts_centered = _inflate_points_about_center(all_pts - centroid, _resolve_turntable_fit_padding(render_kwargs, self))
        n_samples    = int(n_fit_samples) if n_fit_samples is not None else len(frames)
        if end_angle == start_angle:
            n_samples = 1
        chosen = _fit_camera_over_orbit(
            pts_centered, R_cam, base_cam_col, centroid, aov, width, height, rotation_axis, start_angle, end_angle,
            max(n_samples, 1), verbose=verbose,
        )
        return chosen, centroid

    # -- files --------------------------------------------------------------------------

    @classmethod
    def load_fbx(cls, filename: str, name: Optional[str] = None) -> "Scene":
        """the file's nodes or takes, meshes, UV sets, skins, materials,
        cameras, lights and blend shape targets, as :func:`cgmath.formats.load_fbx`
        reads them, named after the file unless ``name`` says"""
        contents = _load_fbx(filename)
        stem     = os.path.splitext(os.path.basename(os.path.expanduser(filename)))[0]
        return cls(name=name or stem, **contents._asdict())

    def save_fbx(self, filename: str, as_ascii: bool = False) -> str:
        """the scene written as :func:`cgmath.formats.save_fbx` writes its
        fields; the render settings are not in a file. Returns the path"""
        _save_fbx(filename, as_ascii=as_ascii, **{field: getattr(self, field) for field in FbxContents._fields})
        return filename

    @classmethod
    def load_obj(cls, filename: str, name: Optional[str] = None) -> "Scene":
        """every mesh of an OBJ file, with its UV sets, at identity (an OBJ
        holds no nodes)"""
        scene = cls(name=name or os.path.splitext(os.path.basename(os.path.expanduser(filename)))[0])
        for mesh, uv_list in _load_obj(filename):
            scene.add(mesh)
            for uv in uv_list:
                uv.mesh_uuid = mesh.uuid
                scene.add(uv)
        return scene

    @classmethod
    def load_glb(
        cls,
        filename:        str,
        name:            Optional[str] = None,
        scale_factor:    float         = 100.0,
        load_skin:       bool          = True,
        extract_texture: bool          = True,
    ) -> "Scene":
        """
        Every mesh of a GLB file, with its UV sets, at identity, its points
        times ``scale_factor`` (a glb is in metres; 100 gives cm). With
        ``load_skin`` the skins, paired to the meshes by position as the glb
        holds them (a warning and no skin when they can not be paired), and
        the file's nodes as the bind pose, each joint moved to the world its
        inverse bind matrix holds; with ``extract_texture`` a material per
        mesh holding its base colour map (none for a mesh with no map). A
        file's animations are the clip. Requires trimesh.
        """
        scene = cls(name=name or os.path.splitext(os.path.basename(os.path.expanduser(filename)))[0])
        data  = _load_glb(filename, scale_factor=scale_factor)
        for mesh, uv_list in data:
            scene.add(mesh)
            for uv in uv_list:
                uv.mesh_uuid = mesh.uuid
                scene.add(uv)
        meshes = [mesh for mesh, _ in data]
        if load_skin:
            _glb_skins(scene, filename, meshes, scale_factor)
        if extract_texture:
            for mesh, index in zip(meshes, _glb_material_indices(filename, len(meshes))):
                if index is None:
                    continue
                color = _extract_glb_textures(filename, material_index=index).get("base_color")
                if color is not None:
                    sets = [uv for uv in scene.uvs or [] if uv.mesh_uuid == mesh.uuid]
                    scene.add(MaterialData(mesh_uuid=mesh.uuid, texture=color, uv_set=sets[0].name if sets else None))
        return scene

    def strip_namespace(self, namespace: Optional[str] = None) -> None:
        """``strip_namespace()`` on the nodes, the clip and every shape list
        that has one, and on :attr:`default_camera`"""
        for field in ("hierarchy", "clip", "meshes", "uvs", "skins", "materials", "cameras", "lights", "morphs"):
            held = getattr(self, field)
            if held is not None and hasattr(held, "strip_namespace"):
                held.strip_namespace(namespace)
        self.default_camera = _strip_namespace(self.default_camera, namespace)


# -- helpers ---------------------------------------------------------------------------


def aim(node: TransformData, target, up=(0.0, 1.0, 0.0)) -> TransformData:
    """``node`` turned to look down its -Z at ``target`` from where it is,
    its +Y toward ``up``, as a camera or a directed light looks; the node"""
    eye               = np.asarray(node.world_matrix, dtype=np.float64)[3, :3]
    node.world_matrix = _look_at(tuple(eye), tuple(np.asarray(target, dtype=np.float64).ravel()), tuple(up)).T
    return node


def _as_list(value, item, cls, where: str):
    """``value`` as a ``cls``: one ``item``, a ``cls``, or an iterable of items"""
    if isinstance(value, cls):
        return value
    if isinstance(value, item) and not (item is MeshData and isinstance(value, UVData)):
        return cls([value])
    if isinstance(value, Data):
        raise TypeError(f"{where} holds {item.__name__}, not a {type(value).__name__}")
    try:
        held = list(value)
    except TypeError:
        raise TypeError(f"{where} is a {cls.__name__}, not a {type(value).__name__}") from None
    for x in held:
        if not isinstance(x, item) or (item is MeshData and isinstance(x, UVData)):
            raise TypeError(f"{where} holds {item.__name__}, not a {type(x).__name__}")
    return cls(held)


def _mesh_label(mesh) -> str:
    return f"mesh {mesh.name!r}" if mesh.name else f"the mesh {mesh.uuid}"


def _worlds(pose) -> Tuple[dict, dict]:
    """``({uuid: world (4, 4)}, {uuid: shown})`` of a pose's nodes; a node
    under a hidden node is hidden"""
    if pose is None:
        return {}, {}
    worlds = {node.uuid: np.asarray(node.world_matrix, dtype=np.float64) for node in pose}
    shown  = {}
    for node in pose:
        visible = bool(node.visibility)
        parent  = node
        while visible:
            parent = parent.get_parent()
            if parent is None:
                break
            visible = bool(parent.visibility)
        shown[node.uuid] = visible
    return worlds, shown


def _world_points(mesh, world) -> np.ndarray:
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.shape[1] != 3:
        return points
    world = np.asarray(world, dtype=np.float64)
    with np.errstate(all="ignore"):
        return points @ world[:3, :3] + world[3, :3]


def _morphed_points(mesh, morphs) -> Optional[np.ndarray]:
    """the mesh's points moved by ``morphs``, its blend shape targets, at
    their weights, Maya's way (a weight between two targets of a channel
    blends them, beyond the last continues its slope); None for a mesh with
    no target weighing anything"""
    channels = {}
    for morph in morphs:
        channels.setdefault((morph.deformer, morph.name), []).append(morph)
    points = None
    for targets in channels.values():
        weight = float(targets[0].weight)
        if weight == 0.0:
            continue
        if points is None:
            points = np.array(mesh.points, dtype=np.float64)
        points += _channel_offsets(targets, weight, len(points))
    return points


def _channel_offsets(targets, weight: float, count: int) -> np.ndarray:
    """a channel's offsets at ``weight`` over ``count`` points: its targets
    by ``at`` are knots after the rest at 0; a weight between two blends
    them linearly, past the last continues the last slope, as Maya's
    blendShape blends in-betweens"""
    knots = sorted(targets, key=lambda m: float(m.at))
    ats   = [0.0] + [float(m.at) for m in knots]
    dense = [np.zeros((count, 3))]
    for morph in knots:
        held = np.zeros((count, 3))
        held[np.asarray(morph.indices, dtype=np.int64)] += np.asarray(morph.offsets, dtype=np.float64)
        dense.append(held)
    if weight >= ats[-1]:
        i = len(ats) - 2
    else:
        i = max(0, next(k for k in range(len(ats) - 1) if weight < ats[k + 1]))
    t = (weight - ats[i]) / (ats[i + 1] - ats[i])
    return dense[i] + t * (dense[i + 1] - dense[i])


def _three_channels(pixels) -> np.ndarray:
    """a (H, W, C) map as the loader takes it: a grey map as (H, W), a grey
    and alpha one as its grey"""
    array = np.asarray(pixels)
    if array.ndim == 3 and array.shape[2] in (1, 2):
        return array[:, :, 0]
    return array


def _bake_wireframe(uv, pixels, color, wire_color, thickness: int) -> np.ndarray:
    """the mesh's edges drawn in UV space over the colour map (or over
    ``color`` with no map), through a copy of ``uv``'s buffer; (H, W, 3)
    float32 in 0 to 1"""
    held           = uv.copy()
    held.antialias = True
    if pixels is None:
        existing = held.buffer
        h, w = existing.shape[:2]
        fill        = (np.asarray(color, dtype=np.float64) * 255.0).clip(0, 255).astype(np.uint8)
        held.buffer = np.broadcast_to(fill, (h, w, 3)).copy()
    else:
        held.buffer = np.flipud(_texture_as_uint8(_load_texture(_three_channels(pixels)) * 255.0)).copy()
    held.draw_edges(color=tuple(int(c) for c in wire_color), thickness=int(thickness))
    return np.flipud(held.buffer).copy().astype(np.float32) / 255.0


def _one_take(clip):
    """the one take ``animate`` plays: a ClipData as it is, a ClipList's
    active take (``clips.active``, the first take until set)

    Raises
    ------
    ValueError
        A ClipList of no take.
    TypeError
        Anything else, a take's name included.
    """
    if isinstance(clip, ClipList):
        if not len(clip):
            raise ValueError("animate: the ClipList holds no take")
        return clip.active
    if isinstance(clip, ClipData):
        return clip
    raise TypeError(f"animate plays a ClipData or a ClipList's active take, not {type(clip).__name__}: {clip!r}")


def _validate_samples_per_pixel(
    value:      Optional[int],
    *,
    field_name: str           = "samples_per_pixel",
) -> Optional[int]:
    """``value`` as the renderer takes it: None, or 1 or a perfect square
    (1, 4, 9, 16 ...), the sub-pixel grid's side squared; ValueError naming
    ``field_name`` otherwise"""
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
    value:      Optional[Union[Tuple[float, ...], List[float], np.ndarray]],
    field_name: str,
) -> Optional[np.ndarray]:
    """an optional RGB or RGBA colour as float64 (4,): RGB taken opaque,
    None kept; ValueError naming ``field_name`` for anything else. A new
    array, never ``value`` itself"""
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
    return np.array(seq, dtype=np.float64)


def _enum_value(value):
    """an Enum setting as its value (``SampleMethod.BEZIER`` as ``"bezier"``),
    anything else as given"""
    return value.value if isinstance(value, Enum) else value


def _frozen(value):
    """a setting as a hashable value for the render signature: an array
    becomes the tuple of its values"""
    if isinstance(value, np.ndarray):
        return tuple(value.ravel().tolist())
    return value


def _texture_as_uint8(texture: np.ndarray) -> np.ndarray:
    """``texture``'s pixels as uint8, scaled by dtype as :func:`_load_texture`
    reads them: uint8 as is, uint16 by 1/257, floats from [0, 1]"""
    texture = np.asarray(texture)
    if texture.dtype == np.uint8:
        return texture
    if texture.dtype == np.uint16:
        return (texture // 257).astype(np.uint8)
    if texture.dtype.kind == "f":
        return np.rint(texture.clip(0.0, 255.0 if texture.max() > 1.0 else 1.0) * (1.0 if texture.max() > 1.0 else 255.0)).astype(np.uint8)
    return texture.astype(np.uint8)


def _resolve_turntable_fit_padding(render_kwargs: dict, scene: "Scene") -> float:
    """the autofit margin of a sequence: ``fit_padding`` popped from the
    keywords (so a frame does not inflate again), else 1.15 for bezier
    sampling, else 1"""
    pad = render_kwargs.pop("fit_padding", None)
    if pad is not None:
        return float(pad)
    return 1.15 if str(scene.sample_method).lower() == "bezier" else 1.0


# -- glb -----------------------------------------------------------------------------


def _glb_skins(scene: "Scene", filename: str, meshes: list, scale_factor: float) -> None:
    """the file's skins on ``scene``'s meshes, paired by position (trimesh
    and the skin reader name a mesh differently, and both walk the file in
    order), and the bind pose: the file's nodes with each joint at the world
    its inverse bind matrix holds, scaled with the points. A file whose
    skins and meshes can not be paired, or whose skin can not be read, warns
    and skins nothing"""
    try:
        entries = _read_skins_glb(filename, bind_matrices=True)
    except Exception as exc:
        warnings.warn(f"{filename}: could not read skin weights ({exc})", stacklevel=3)
        return
    if not any(x is not None for x in entries):
        return
    if len(entries) != len(meshes):
        warnings.warn(
            f"{filename}: {len(meshes)} meshes but {len(entries)} skin entries, so they cannot be paired up -- "
            f"loading without skinning. This usually means the file holds a primitive that is not triangles.",
            stacklevel=3,
        )
        return
    rig = HierarchyData.load_glb(filename, scale_factor=scale_factor)
    try:
        binds = {}
        for mesh, entry in zip(meshes, entries):
            if entry is None:
                continue
            skin, matrices = entry
            skin.mesh_uuid = mesh.uuid
            for joint, inverse in zip(skin.influences, _scaled_binds(matrices, scale_factor)):
                binds.setdefault(str(joint), np.linalg.inv(inverse))
            scene.add(skin)
        _settle_glb_bind(rig, binds)
    except Exception as exc:
        warnings.warn(f"{filename}: could not bind the skins ({exc}) -- loading without skinning.", stacklevel=3)
        scene.skins = None
        return
    scene.hierarchy = rig


def _settle_glb_bind(rig: HierarchyData, binds: dict) -> None:
    """each joint of ``rig`` named in ``binds`` moved to the world matrix
    its inverse bind matrix inverts to, parents first; ValueError for a
    joint ``rig`` lacks or a world its channels can not hold"""
    names = list(rig.name)
    for joint in binds:
        if joint not in names:
            raise ValueError(f"the skins name joint {joint!r}, which the file's nodes do not hold")

    def depth(node) -> int:
        count, parent = 0, node.get_parent()
        while parent is not None:
            count, parent = count + 1, parent.get_parent()
        return count

    for node in sorted(rig, key=depth):
        world = binds.get(str(node.name))
        if world is None:
            continue
        node.world_matrix = world
        if np.abs(np.asarray(node.world_matrix) - world).max() > 1e-6 * max(1.0, float(np.abs(world).max())):
            raise ValueError(f"joint {node.name!r}: its channels can not hold the bind world {np.round(world, 6).tolist()}")


def _scaled_binds(matrices: np.ndarray, scale_factor: float) -> np.ndarray:
    """a glb's inverse bind matrices in the unit its points were read in:
    the translation row moves, the rotation stays"""
    if scale_factor == 1.0:
        return np.asarray(matrices, dtype=np.float64)
    scaled = np.array(matrices, dtype=np.float64, copy=True)
    scaled[:, 3, :3] *= scale_factor
    return scaled


def _glb_material_indices(file_path: str, count: int) -> List[Optional[int]]:
    """the material index of each of the ``count`` meshes ``mesh.load_glb``
    reads from the file, None where a mesh has no material; paired by
    position as the skins are, the primitives trimesh skips skipped here"""
    json_chunk, _ = _parse_glb(file_path)
    if json_chunk is None:
        return [None] * count

    primitives = [
        (primitive.get("mode", 4), primitive.get("material"))
        for mesh in json_chunk.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    for kept in (None, (4, 5), (4,)):
        indices = [m for mode, m in primitives if kept is None or mode in kept]
        if len(indices) == count:
            return indices

    indices = [m for _, m in primitives]
    if len(json_chunk.get("materials", [])) == 1 and set(indices) <= {0}:
        return [0] * count
    if not any(x is not None for x in indices):
        return [None] * count
    warnings.warn(
        f"{file_path}: {count} meshes but {len(indices)} primitives, so their materials cannot be paired up -- "
        f"loading without textures.",
        stacklevel=3,
    )
    return [None] * count


# GLB container constants (binary-glTF 2.0 spec).
_GLB_MAGIC      = 0x46546C67  # ASCII 'glTF' little-endian
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK  = 0x004E4942

# Material shader-slot paths (dotted) to our internal short names.
_GLB_TEXTURE_SLOTS = {
    "pbrMetallicRoughness.baseColorTexture": "base_color",
    "pbrMetallicRoughness.metallicRoughnessTexture": "metallic_roughness",
    "normalTexture": "normal",
    "occlusionTexture": "occlusion",
    "emissiveTexture": "emissive",
}


def _parse_glb(glb_path: str) -> Tuple[Optional[dict], Optional[bytes]]:
    """``(json chunk, bin chunk)`` of a GLB container, ``(None, None)`` for a
    file that is none"""
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
        offset += 8 + chunk_length
        offset += (4 - (offset % 4)) % 4

    return json_chunk, bin_chunk


def _extract_glb_image_bytes(
    json_chunk:  dict,
    bin_chunk:   Optional[bytes],
    image_index: int,
) -> Optional[bytes]:
    """the encoded bytes of ``json_chunk["images"][image_index]``: a buffer
    view of the bin chunk or a base64 data URI; None for an external file or
    a bad index"""
    images       = json_chunk.get("images", [])
    buffer_views = json_chunk.get("bufferViews", [])
    if not 0 <= image_index < len(images):
        return None
    img = images[image_index]

    if "bufferView" in img:
        bv_idx = img["bufferView"]
        if not 0 <= bv_idx < len(buffer_views) or bin_chunk is None:
            return None
        bv        = buffer_views[bv_idx]
        bv_offset = bv.get("byteOffset", 0)
        bv_length = bv["byteLength"]
        return bin_chunk[bv_offset : bv_offset + bv_length]

    uri = img.get("uri")
    if uri is not None and uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        return base64.b64decode(encoded)
    return None


def _decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """encoded image bytes as (H, W, C) uint8 through PIL, RGB or RGBA; None
    without PIL"""
    if Image is None:
        return None
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGBA" if "A" in pil_img.mode else "RGB")
    return np.asarray(pil_img, dtype=np.uint8)


def _extract_glb_textures(
    file_path:      str,
    material_index: int = 0,
) -> dict:
    """every PBR texture of ``materials[material_index]`` of a GLB, by slot
    name (base_color, metallic_roughness, normal, occlusion, emissive), each
    (H, W, C) uint8; a slot absent, external or undecodable left out"""
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


# -- the render cache ------------------------------------------------------------------
#
# Each Scene keeps the frames of its last turntable / animate in its own
# folder, under one scratch folder per process, so a second encode of the
# same frames (a .gif, then an .mp4) reads them back instead of rendering. The
# key is everything the frames depend on: the scene's render signature and the
# sequence's own settings. A different key wipes the folder first.

_CACHE_ROOT: Optional[str] = None
_CACHE_DIRS: dict = {}


def _render_cache_root() -> str:
    """this process's scratch folder for rendered sequences, removed at exit"""
    global _CACHE_ROOT
    if _CACHE_ROOT is None or not os.path.isdir(_CACHE_ROOT):
        _CACHE_ROOT = tempfile.mkdtemp(prefix="cgmath_render_")
        atexit.register(shutil.rmtree, _CACHE_ROOT, True)
    return _CACHE_ROOT


def _render_cache_dir(owner) -> str:
    """*owner*'s own folder: made on first use, removed when *owner* goes"""
    key    = id(owner)
    folder = _CACHE_DIRS.get(key)
    if folder is None or not os.path.isdir(folder):
        folder           = tempfile.mkdtemp(dir=_render_cache_root())
        _CACHE_DIRS[key] = folder
        weakref.finalize(owner, _drop_render_cache_dir, key, folder)
    return folder


def _drop_render_cache_dir(key: int, folder: str) -> None:
    if _CACHE_DIRS.get(key) == folder:
        del _CACHE_DIRS[key]
    shutil.rmtree(folder, ignore_errors=True)


def _clear_render_cache(owner) -> None:
    """empties *owner*'s folder, if it has one"""
    folder = _CACHE_DIRS.get(id(owner))
    if folder is not None and os.path.isdir(folder):
        _wipe_folder(folder)


def _wipe_folder(folder: str) -> None:
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)


def _canonical(value):
    """*value* as plain, comparable data for a cache key"""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _canonical(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, np.ndarray):
        if value.size <= 64 and value.dtype.kind in "biuf":
            return _canonical(value.tolist())
        array = np.ascontiguousarray(value)
        return ("ndarray", array.dtype.str, array.shape, hashlib.sha1(array.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    return (type(value).__name__, id(value))


def _key_render_kwargs(render_kwargs: dict) -> dict:
    """*render_kwargs* as a sequence key holds them: ``threads`` sets how many
    CPU threads render a frame, not the frame. A bad ``threads`` raises here"""
    from cgmath.render._tiles import check_threads

    check_threads(render_kwargs.get("threads"))
    return {k: v for k, v in render_kwargs.items() if k != "threads"}


def _sequence_key(scene: "Scene", settings) -> str:
    """the cache key of a turntable / animate on *scene* with *settings*"""
    signature = scene._render_signature_of(None)
    payload   = pickle.dumps(_canonical((signature, settings)), protocol=4)
    return hashlib.sha1(payload).hexdigest()


def _cached_sequence(owner, key: str):
    """*owner*'s kept frames for *key*, read back one by one, or None"""
    folder   = _render_cache_dir(owner)
    manifest = os.path.join(folder, "sequence.json")
    try:
        with open(manifest) as f:
            kept = json.load(f)
    except (OSError, ValueError):
        return None
    if kept.get("key") != key:
        return None
    return _replay_sequence(folder, kept["frames"])


def _replay_sequence(folder: str, frames: list):
    from cgmath.render.frame import Frame

    for index, meta in enumerate(frames):
        stem = os.path.join(folder, f"{index:05d}")
        yield Frame(
            array         = np.load(stem + ".npy"),
            depth         = np.load(stem + ".depth.npy") if meta["depth"] else None,
            camera_matrix = np.load(stem + ".camera.npy") if meta["camera"] else None,
            angle_of_view = meta["angle_of_view"],
        )


def _record_sequence(owner, key: str, frames):
    """*frames*, passed on as they render, each saved to *owner*'s wiped
    folder; kept only once the last frame is through"""
    folder = _render_cache_dir(owner)
    _wipe_folder(folder)
    kept = []
    try:
        for index, frame in enumerate(frames):
            stem = os.path.join(folder, f"{index:05d}")
            np.save(stem + ".npy", np.asarray(frame.array))
            if frame.depth is not None:
                np.save(stem + ".depth.npy", np.asarray(frame.depth))
            if frame.camera_matrix is not None:
                np.save(stem + ".camera.npy", np.asarray(frame.camera_matrix))
            kept.append(
                {
                    "depth":         frame.depth is not None,
                    "camera":        frame.camera_matrix is not None,
                    "angle_of_view": None if frame.angle_of_view is None else float(frame.angle_of_view),
                }
            )
            yield frame
    finally:
        frames.close()

    with open(os.path.join(folder, "sequence.json"), "w") as f:
        json.dump({"key": key, "frames": kept}, f)


def _encode_frame_stream(
    frames,
    output_pattern: str,
    fps:            int                        = 30,
    mp4_background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    verbose:        bool                       = False,
) -> Union[str, List[str]]:
    """a stream of frames written out, the encoder chosen by the extension:
    .mp4 / .mov / .m4v H.264 through ffmpeg, .avi, .gif (gifski, else
    ffmpeg's two passes), anything else a numbered image sequence honouring
    a ``{frame[:NNd]}`` placeholder and inventing one when absent. Returns
    the file's path, or the frames' paths. The frames are consumed lazily"""
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
        return Frame.encode_mp4(announced(), out_path, fps=int(fps), background=mp4_background)

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


def _pose_sample_frames_for(frame_count: int, n_pose_samples: Optional[int]) -> np.ndarray:
    """the frame indices :meth:`Scene.animate` poses to fit its camera: 10,
    or a tenth of the clip once that is more, never more than the clip has"""
    if n_pose_samples is None:
        n_pose_samples = max(10, -(-frame_count // 10))
    count = max(1, min(int(n_pose_samples), frame_count))
    return np.unique(np.linspace(0, frame_count - 1, count).astype(int))


def _orbit_camera_col(
    cam_col:       np.ndarray,
    centroid:      np.ndarray,
    rotation_axis: str,
    angle:         float,
) -> np.ndarray:
    """*cam_col* orbited by *angle* about *rotation_axis* through *centroid*,
    column major in and out"""
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
    """the camera matrix that fits *pts_centered* (centred on the AABB
    midpoint, inflated by the fit padding) at every sampled angle of the
    orbit: the union box in the camera's frame, centred laterally and pushed
    back along +Z until it fits the field of view"""
    bbx_min = np.full(3, np.inf, dtype=np.float64)
    bbx_max = np.full(3, -np.inf, dtype=np.float64)
    span    = float(end_angle - start_angle)

    with np.errstate(all="ignore"):
        for i in range(n_samples):
            angle         = start_angle + span * i / max(n_samples, 1)
            R_world_3     = _rotation_matrix_col(rotation_axis, angle)[:3, :3]
            rotated_world = pts_centered @ R_world_3.T
            rotated_local = rotated_world @ R_cam
            bbx_min       = np.minimum(bbx_min, rotated_local.min(axis=0))
            bbx_max       = np.maximum(bbx_max, rotated_local.max(axis=0))

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
    cam_world_pos    = centroid + R_cam @ cam_local_offset

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
    """4x4 column-major rotation about a world axis, right-hand rule"""
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


def _default_lighting_rig_for(points: np.ndarray) -> List[dict]:
    """a 3-point rig (key, fill, rim) as the renderer's light dicts, sized
    to the AABB of ``points``: the distances scale with the diagonal and the
    intensities with its square, for the 1/r^2 falloff, so a unit cube and a
    100-unit ship both light up"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return [{"kind": "point", "position": (1.0, 1.0, 1.0), "color": (1.0, 1.0, 1.0), "intensity": 1.0, "falloff": True}]

    mn        = pts.min(axis=0)
    mx        = pts.max(axis=0)
    centroid  = 0.5 * (mn + mx)
    extent    = max(float(np.linalg.norm(mx - mn)), 1e-6)

    distance  = extent * 1.5
    intensity = distance * distance
    rig       = []
    for offset, scale in (((1.0, 1.0, 1.0), 1.0), ((-1.0, 0.5, 1.0), 0.4), ((0.0, 0.5, -1.0), 0.3)):
        position = centroid + np.array(offset) * distance
        rig.append({
            "kind":      "point",
            "position":  tuple(float(x) for x in position),
            "color":     (1.0, 1.0, 1.0),
            "intensity": intensity * scale,
            "falloff":   True,
        })
    return rig
