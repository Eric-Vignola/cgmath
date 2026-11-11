"""``cgmath.formats.save_fbx``: a hierarchy or its takes, the meshes on its nodes

The r17-skeleton fork writes a hierarchy of joints and transforms, or a
ClipList's nodes and its takes (a ClipData as a ClipList of one), each mesh
as the shape of its node, a bind pose with skins only. Nothing here imports
the fbx sdk, nor the classes, until a file is written: the classes' modules
import this one.
"""

import contextlib
import logging
import os
import re
import shutil
import tempfile

import numpy as np

# the fbx writer's logger: its warnings and save_fbx's are one stream
LOGGER = logging.getLogger("cgmath.formats.fbx")

FORK   = "the r17-skeleton fork"

# a scale whose components differ by more than this, relative to the
# largest, is not uniform
UNIFORM_SCALE = 1e-6

# a skinned mesh's node this far from its bind world on a frame, relative to
# the largest element, has moved; Maya's FBXExport writes the bind pose in
# float32
STILL = 1e-5

# the node kinds a hierarchy holds: a locator is a transform drawn as a cross
NODE_TYPES = ("joint", "transform", "locator")

# the hidden property keeping each node's place in the hierarchy's list
ORDER_PROPERTY = "cgmath:index"

# a NURBS geometry of the sdk's ASCII draft, by the name it was made under,
# and an array of numbers inside one
_NURBS_BLOCK = re.compile(r'Geometry: (\d+), "Geometry::(g\d+)", "(\w+)" \{')
_NURBS_ARRAY = re.compile(r"(Points|KnotVectorU|KnotVectorV|KnotVector): \*(\d+) \{\s*a: [^}]*\}")


def save_fbx(
    filename,
    hierarchy      = None,
    clip           = None,
    meshes         = None,
    skins          = None,
    uvs            = None,
    cameras        = None,
    lights         = None,
    materials      = None,
    morphs         = None,
    curves         = None,
    surfaces       = None,
    as_ascii: bool = False,
) -> None:
    """writes one of ``hierarchy``, a HierarchyData of joints and transforms,
    or ``clip``, a ClipList of ClipData over joints and transforms (one
    ClipData is written as a ClipList of one), and ``meshes``, ``cameras``,
    ``lights``, ``curves`` and ``surfaces`` on their nodes, ``materials``
    and ``morphs`` on their meshes; binary unless ``as_ascii``

    A hierarchy alone: its nodes and their own values, no take. A clip
    alone: its nodes, their static values the active take's pose at its
    first frame, and one animation stack a take, in the list's order, named
    ``take`` or "Take 001", "Take 002" ... by position, each spanning its
    take's frames at the takes' one fps, the active take's stack the file's
    current one (``ActiveAnimStackName``); every channel of every node has
    its curves in every stack, one a component: a constant channel one key
    on the take's first frame, an animated one a key a frame; a node's
    visibility, where a take changes it, a stepped key a frame, as Maya keys
    one. A hierarchy given with a clip is a bind pose, which comes with
    skins (NotImplementedError without them).

    ``meshes``, a MeshList or one MeshData: each the shape of the transform
    its ``parent_uuid`` names, named after the mesh (unnamed for a mesh of
    no name, as Maya's FBXExport leaves one); a mesh naming no node of the
    hierarchy or the clip goes on a new root node at identity, named after
    it (``"mesh"`` for one of no name). ``uvs``, a UVList
    or one UVData: each set on the mesh its ``mesh_uuid`` names, in their
    order, named after the set; an empty UVList writes none. ``skins``, a
    SkinList or one SkinData: each on the mesh its ``mesh_uuid`` names, a
    cluster an influence, a node of the hierarchy found by name, the mesh's
    node at its world as every cluster's transform (Maya's geomMatrix), and
    one bind pose listing every node at its world.

    ``cameras``, a CameraList or one CameraData, and ``lights``, a LightList
    or one LightData: each the shape of the transform its ``parent_uuid``
    names, or of a new root node at identity named after it, written as
    Maya's exporter writes them (measured): the node turned to fbx's camera
    and light axes with a post rotation, (0, -90, 0) on a camera, (-90, 0,
    0) on a directional or spot light; the camera's focal length and film
    back as they are (``ApertureMode`` eFocalLength), its film fit as the
    gate fit, its orthographic width over 30 as the ortho zoom, its
    position, up vector and interest 5 units down its view; a light's
    intensity times 100, its decay rate as the decay type, a spot's cone as
    the inner angle and its cone plus twice its penumbra as the outer. Fbx
    holds no camera scale and no dropoff: one warning names a value that is
    not 1, or not 0, and the file holds neither, as Maya's own export drops
    them.

    ``materials``, a MaterialList or one MaterialData: each on the mesh its
    ``mesh_uuid`` names, written as Maya's exporter writes a lambert or a
    phong (measured): an FbxSurfaceLambert or FbxSurfacePhong named after
    it, its colour as the diffuse colour and its diffuse as the diffuse
    factor, its ambient colour, incandescence and transparency as the
    ambient, emissive and transparent colours with factors of 1, a phong's
    specular colour as the specular with a factor of 1, its cosine power as
    the shininess and its reflectivity as the reflection factor; the mesh's
    material element all-same for one material covering the whole mesh,
    by-polygon for materials covering faces (``faces``); a texture as a PNG
    embedded in the file, binary or ASCII, feeding the diffuse colour
    through an FbxFileTexture named after its file, reading the UV set and
    wrapping as the material says (Maya's import takes the UV set and
    leaves both wraps on; measured).

    ``morphs``, a MorphList or one MorphData: each on the mesh its
    ``mesh_uuid`` names, written as Maya's exporter writes a blendShape
    (measured): one FbxBlendShape a mesh, named after its targets'
    ``deformer`` (None as "blendShape"), one channel the targets of a name,
    named "deformer.name" as Maya names one (its import strips the prefix),
    its DeformPercent the targets' weight times 100, one shape a target
    holding every point of the mesh, moved by the offsets, in-betweens
    first by ``at``, each at its ``at`` times 100 as the shape's full
    weight; a target's normals are not written. Two deformer names on one
    mesh are refused: Maya's import deforms a mesh by its last blend shape
    deformer alone (measured).

    ``curves``, a NurbsCurveList or one NurbsCurveData, and ``surfaces``, a
    NurbsSurfaceList or one NurbsSurfaceData: each the shape of the
    transform its ``parent_uuid`` names, or of a new root node at identity
    named "curve" or "surface", written as Maya's exporter writes them
    (measured): a curve as an FbxNurbsCurve, a surface as an FbxNurbsSurface
    with u running fastest, unnamed, holding the order, the form, open or
    periodic, the points each once with its weight (1 where the shape
    holds no ``weights``; a point is written as it stands, not multiplied
    by its weight, as Maya writes it, measured), and the knots, the
    ``knots`` held or with none the ones the shape makes for itself, with
    one more at each end, a step past it as its neighbour's; a curve of
    degree 1 that is not periodic and holds no knots as a polyline of an
    FbxLine, the ones of one node as the lines of one FbxLine. A file holds
    no ``uniform`` and no ``registered``: neither is written. The sdk's
    Python binding sets no knot, so the NURBS shapes pass through an ASCII
    draft in a temporary folder, its numbers replaced and read back bit for
    bit.

    Refused, ValueError naming node, channel and frame, before anything is
    written: a node that is no joint, transform or locator, a transform or
    locator holding segment scale compensate or a joint orient, a shear, a
    scale that is not
    uniform on a node with children, in the hierarchy or on any frame of any
    take, a joint orient or rotate axis a take changes over its frames or
    holds other than the active take's (a file holds one a node), takes at
    two frame rates (a file holds one time mode), two takes written under
    one name, a ClipList of no take, a take of no frame; a
    mesh holding holes or faces naming points it does not hold, two meshes
    on one node, a mesh on a joint (Maya's FBX drops one); a UV set or a
    skin naming no mesh; a UV set of no name, of
    a name another set of its mesh holds or that is not ASCII, whose faces
    are not its mesh's, or naming UVs it does not hold; a skin naming no
    node of the hierarchy, whose weights are not its mesh's points by its
    influences, a second skin on a mesh, a clip moving a skinned mesh's node
    off its bind world (Maya's skinCluster applies the node's own transform
    on top of the skin; measured); a camera or light on a joint, on a node
    holding another shape, on a node holding a rotate axis or children (the
    turn is the node's own); a curve or surface whose points are not 3D, on
    a joint, or on a node holding another shape, the curves of degree 1
    holding no knots of one node apart (Maya's exporter writes two NURBS
    curves on one transform as an empty node; measured); a material naming
    no mesh, covering faces its mesh does not hold or a face twice, two
    materials covering one face,
    materials leaving faces of their mesh uncovered (a mesh is covered whole
    or not at all), a material covering the whole mesh beside another on
    it, a texture that is not (H, W, C) uint8 with 1 to 4 channels (a .fbx
    embeds an 8 bit PNG); a morph naming no mesh, moving points its mesh
    does not hold or as many offsets as it has not indices, two deformer
    names on one mesh, a channel (the targets of one name on one mesh)
    without one target at 1, with two targets at one ``at``, an in-between
    above 1, or targets of two weights. UV sets, materials or morphs given
    without meshes, skins
    without the hierarchy and the meshes, nothing given, a hierarchy that is
    not a HierarchyData (a ClipData or a ClipList is passed as ``clip``), a
    clip that is not a ClipData or a ClipList, meshes, skins, uvs, cameras,
    lights, materials, morphs, curves or surfaces of another class:
    TypeError. A texture with no PIL (pillow) to encode it: ImportError.
    """
    # imported here: the classes' modules import this one
    from cgmath.geometry.bspline import NurbsCurveData, NurbsCurveList
    from cgmath.geometry.bspline_patch import NurbsSurfaceData, NurbsSurfaceList
    from cgmath.geometry.camera import CameraData, CameraList
    from cgmath.geometry.light import LightData, LightList
    from cgmath.geometry.material import MaterialData, MaterialList
    from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
    from cgmath.geometry.morph_target import MorphData, MorphList
    from cgmath.geometry.skin_weights import SkinData, SkinList
    from cgmath.hierarchy.hierarchy import FBX, ClipData, ClipList, FbxExporter, HierarchyData

    if hierarchy is not None and type(hierarchy) is not HierarchyData:
        hint = ", pass it as clip=" if isinstance(hierarchy, (ClipData, ClipList)) else ""
        raise TypeError(f"save_fbx: the hierarchy is a HierarchyData, not a {type(hierarchy).__name__}{hint}")
    if isinstance(clip, ClipData):
        clip = ClipList([clip])  # one take, the active one
    if clip is not None and not isinstance(clip, ClipList):
        raise TypeError(f"save_fbx: the clip is a ClipData or a ClipList, not a {type(clip).__name__}")
    if clip is not None and not len(clip):
        raise ValueError("save_fbx: the ClipList holds no take")
    if isinstance(meshes, MeshData) and not isinstance(meshes, UVData):
        meshes = MeshList([meshes])
    if meshes is not None and (not isinstance(meshes, MeshList) or isinstance(meshes, UVList)):
        raise TypeError(f"save_fbx: the meshes are a MeshList or a MeshData, not a {type(meshes).__name__}")
    if isinstance(skins, SkinData):
        skins = SkinList([skins])
    if skins is not None and not isinstance(skins, SkinList):
        raise TypeError(f"save_fbx: the skins are a SkinList or a SkinData, not a {type(skins).__name__}")
    if isinstance(uvs, UVData):
        uvs = UVList([uvs])
    if uvs is not None and not isinstance(uvs, UVList):
        raise TypeError(f"save_fbx: the uvs are a UVList or a UVData, not a {type(uvs).__name__}")
    if isinstance(cameras, CameraData):
        cameras = CameraList([cameras])
    if cameras is not None and not isinstance(cameras, CameraList):
        raise TypeError(f"save_fbx: the cameras are a CameraList or a CameraData, not a {type(cameras).__name__}")
    if isinstance(lights, LightData):
        lights = LightList([lights])
    if lights is not None and not isinstance(lights, LightList):
        raise TypeError(f"save_fbx: the lights are a LightList or a LightData, not a {type(lights).__name__}")
    if isinstance(materials, MaterialData):
        materials = MaterialList([materials])
    if materials is not None and not isinstance(materials, MaterialList):
        raise TypeError(f"save_fbx: the materials are a MaterialList or a MaterialData, not a {type(materials).__name__}")
    if isinstance(morphs, MorphData):
        morphs = MorphList([morphs])
    if morphs is not None and not isinstance(morphs, MorphList):
        raise TypeError(f"save_fbx: the morphs are a MorphList or a MorphData, not a {type(morphs).__name__}")
    if isinstance(curves, NurbsCurveData):
        curves = NurbsCurveList([curves])
    if curves is not None and not isinstance(curves, NurbsCurveList):
        raise TypeError(f"save_fbx: the curves are a NurbsCurveList or a NurbsCurveData, not a {type(curves).__name__}")
    if isinstance(surfaces, NurbsSurfaceData):
        surfaces = NurbsSurfaceList([surfaces])
    if surfaces is not None and not isinstance(surfaces, NurbsSurfaceList):
        raise TypeError(
            f"save_fbx: the surfaces are a NurbsSurfaceList or a NurbsSurfaceData, not a {type(surfaces).__name__}"
        )
    meshes    = meshes if meshes is not None and len(meshes) else None
    skins     = skins if skins is not None and len(skins) else None
    uvs       = uvs if uvs is not None and len(uvs) else None
    cameras   = cameras if cameras is not None and len(cameras) else None
    lights    = lights if lights is not None and len(lights) else None
    materials = materials if materials is not None and len(materials) else None
    morphs    = morphs if morphs is not None and len(morphs) else None
    curves    = curves if curves is not None and len(curves) else None
    surfaces  = surfaces if surfaces is not None and len(surfaces) else None
    if uvs is not None and meshes is None:
        raise TypeError("save_fbx: UV sets are written on their mesh; missing: meshes=")
    if materials is not None and meshes is None:
        raise TypeError("save_fbx: materials are written on their mesh; missing: meshes=")
    if morphs is not None and meshes is None:
        raise TypeError("save_fbx: morphs are written on their mesh; missing: meshes=")
    if skins is not None:
        missing = [
            what for what, given in (("hierarchy (a HierarchyData, the bind pose)", hierarchy), ("meshes=", meshes))
            if given is None
        ]
        if missing:
            raise TypeError(f"save_fbx: a skin is written with its hierarchy and its mesh; missing: {' and '.join(missing)}")
    elif hierarchy is not None and clip is not None:
        raise NotImplementedError(
            f"save_fbx on {FORK}: a hierarchy given with a clip is a bind pose, which comes with skins; "
            f"pass meshes= and skins= with them, or clip= alone, or a hierarchy alone"
        )
    if all(given is None for given in (hierarchy, clip, meshes, cameras, lights, curves, surfaces)):
        raise TypeError(
            "save_fbx takes a hierarchy (a HierarchyData) or clip= (a ClipData or a ClipList), and meshes= (a MeshList "
            "or a MeshData), cameras= (a CameraList or a CameraData), lights= (a LightList or a LightData), curves= (a "
            "NurbsCurveList or a NurbsCurveData), surfaces= (a NurbsSurfaceList or a NurbsSurfaceData)"
        )
    if FBX is None or FbxExporter is None:
        raise ImportError("fbx module not found")

    filename = os.path.expanduser(filename)
    written  = hierarchy
    if hierarchy is not None:
        _check_hierarchy(hierarchy)
    if clip is not None:
        _check_takes(clip)
        if hierarchy is None:
            written = clip.active.at(clip.active.start_frame)
        else:
            for take in clip:
                _check_same(hierarchy, take)
    if meshes is not None:
        _check_meshes(meshes, written)
    if uvs is not None:
        _check_uvs(uvs, meshes)
    if skins is not None:
        _check_skins(skins, meshes, hierarchy)
        if clip is not None:
            for take in clip:
                _check_still(skins, meshes, hierarchy, take)
    if cameras is not None or lights is not None:
        _check_views(cameras, lights, meshes, written)
    if materials is not None:
        _check_materials(materials, meshes)
    if morphs is not None:
        _check_morphs(morphs, meshes)
    if curves is not None or surfaces is not None:
        _check_nurbs(curves, surfaces, meshes, cameras, lights, written)

    fbx = FBX
    with FbxExporter() as exporter, _texture_folder(materials) as folder:
        nodes = []
        if written is not None:
            exporter.add_skeleton(written)
            nodes = [exporter._fbx_nodes[id(node)] for node in written]
        for index, fbx_node in enumerate(nodes):
            _index(fbx_node, index, fbx)
        if clip is not None:
            _write_takes(exporter, nodes, clip)
        by_uuid = {node.uuid: fbx_node for node, fbx_node in zip(written, nodes)} if written is not None else {}
        made    = {}
        if meshes is not None:
            shapes = {}
            for mesh in meshes:
                fbx_node = by_uuid.get(mesh.parent_uuid)
                fbx_node, shape = _write_mesh(exporter, mesh, fbx_node)
                if mesh.parent_uuid not in by_uuid:
                    _index(fbx_node, len(nodes) + len(made), fbx)
                    made[mesh.uuid] = fbx_node
                shapes[mesh.uuid] = (fbx_node, shape)
            if uvs is not None:
                for mesh in meshes:
                    sets = [uv for uv in uvs if uv.mesh_uuid == mesh.uuid]
                    if sets:
                        exporter.add_uvs(shapes[mesh.uuid][1], sets)
            if materials is not None:
                shaders, taken = {}, set()  # the surfaces written so far by uuid, the PNG names taken so far
                for mesh in meshes:
                    held = [material for material in materials if material.mesh_uuid == mesh.uuid]
                    if held:
                        _write_materials(exporter, mesh, held, *shapes[mesh.uuid], folder, shaders, taken)
                if any(material.texture is not None or material.normal_map is not None for material in materials):
                    # the PNGs go into the file, as Maya's Embed Media writes them
                    exporter.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
            if morphs is not None:
                for mesh in meshes:
                    held = [morph for morph in morphs if morph.mesh_uuid == mesh.uuid]
                    if held:
                        _write_morphs(exporter, mesh, held, shapes[mesh.uuid][1])
            if skins is not None:
                column = {node.uuid: i for i, node in enumerate(hierarchy)}
                placed = {mesh.uuid: column.get(mesh.parent_uuid) for mesh in meshes}
                for skin in skins:
                    _write_skin(exporter, *shapes[skin.mesh_uuid], skin, hierarchy, nodes, placed[skin.mesh_uuid])
                roots = [made[skin.mesh_uuid] for skin in skins if skin.mesh_uuid in made]
                _write_bind_pose(exporter, nodes, hierarchy, roots)
        if cameras is not None or lights is not None:
            worlds = {node.uuid: np.asarray(node.world_matrix, dtype=np.float64) for node in written} if written is not None else {}
            for shape, write in [(camera, _write_camera) for camera in cameras or []] + [(light, _write_light) for light in lights or []]:
                fbx_node = write(exporter, shape, by_uuid.get(shape.parent_uuid), worlds.get(shape.parent_uuid, np.eye(4)))
                if shape.parent_uuid not in by_uuid:
                    _index(fbx_node, len(nodes) + len(made), fbx)
                    made[shape.uuid] = fbx_node
        if curves is not None or surfaces is not None:
            _write_shapes(exporter, curves, surfaces, by_uuid, made, len(nodes), fbx)
        exporter.export(path=filename, as_ascii=as_ascii)


def nonuniform(scale: np.ndarray) -> np.ndarray:
    """(...) bool, the (..., 3) scales whose components differ by more than
    UNIFORM_SCALE of the largest"""
    scale  = np.asarray(scale, dtype=np.float64)
    spread = scale.max(axis=-1) - scale.min(axis=-1)
    return spread > UNIFORM_SCALE * np.abs(scale).max(axis=-1)


def parents_of(nodes) -> np.ndarray:
    """(N,) the position of each node's parent among ``nodes``, -1 for a root"""
    place = {node.uuid: i for i, node in enumerate(nodes)}
    return np.array([place.get(node.parent_uuid, -1) for node in nodes], dtype=np.intp)


def refuse_values(what: str, nodes, shear, scale, frames=None) -> None:
    """ValueError naming the first node, channel and frame holding a shear
    other than 0, or a scale that is not uniform on a node with children;
    ``shear`` and ``scale`` (F, N, 3), ``frames`` the frame numbers, None
    for one pose"""
    nodes   = list(nodes)
    parents = parents_of(nodes)
    held    = np.zeros(len(nodes), dtype=bool)
    held[parents[parents >= 0]] = True
    for channel, bad, why in (
        ("shear", np.any(np.asarray(shear) != 0.0, axis=-1), "a shear, which fbx and Maya's joints hold none of"),
        ("scale", nonuniform(scale) & held[None], "a scale that is not uniform on a node with children"),
    ):
        found = np.argwhere(bad)
        if len(found):
            frame, column = (int(x) for x in found[0])
            value = np.asarray(shear if channel == "shear" else scale)[frame, column]
            when  = "" if frames is None else f" frame {int(frames[frame])}"
            raise ValueError(
                f"{what}: node {nodes[column].name!r} channel {channel!r}{when} holds {why}: "
                f"{np.round(value, 12).tolist()} ({len(np.unique(found[:, 1]))} nodes, {len(found)} values)"
            )


def _check_kinds(nodes, orients) -> None:
    """a node that is no joint, transform or locator, a transform or locator
    holding segment scale compensate, or a joint orient (``orients`` (F, N,
    3)): refused"""
    for column, node in enumerate(nodes):
        if node.node_type not in NODE_TYPES:
            raise ValueError(
                f"save_fbx: node {node.name!r} is a {node.node_type!r}; {FORK} writes joints, transforms and locators"
            )
        if node.node_type != "joint" and node.segment_scale_compensate:
            raise ValueError(
                f"save_fbx: node {node.name!r} channel 'segment_scale_compensate' is on a {node.node_type}, which "
                f"Maya holds on joints only"
            )
        if node.node_type != "joint" and np.any(orients[:, column] != 0.0):
            raise ValueError(
                f"save_fbx: node {node.name!r} channel 'joint_orient' is on a {node.node_type}, which Maya holds on "
                f"joints only"
            )


def _check_hierarchy(hierarchy) -> None:
    nodes = list(hierarchy)
    _check_kinds(nodes, np.array([node.joint_orient for node in nodes], dtype=np.float64).reshape(1, -1, 3))
    shear = np.array([node.shear for node in nodes], dtype=np.float64).reshape(1, -1, 3)
    scale = np.array([node.scale for node in nodes], dtype=np.float64).reshape(1, -1, 3)
    refuse_values("save_fbx, the hierarchy", nodes, shear, scale)


def _check_clip(clip) -> None:
    from cgmath.hierarchy.hierarchy import _fbx_rate_mode

    if not clip.frame_count:
        raise ValueError("save_fbx: the clip holds no frame")
    if _fbx_rate_mode(float(clip.fps)) is None:
        raise ValueError(f"save_fbx: no fbx time mode carries {clip.fps} fps")

    nodes  = list(clip)
    frames = clip.start_frame + np.arange(clip.frame_count)
    _check_kinds(nodes, np.asarray(clip.frames.joint_orient, dtype=np.float64))

    # a file holds one joint orient and one rotate axis a node, unkeyed
    for channel in ("joint_orient", "rotate_axis"):
        held = np.asarray(getattr(clip.frames, channel), dtype=np.float64)
        bad  = np.argwhere(np.any(held != held[:1], axis=-1))
        if len(bad):
            frame, column = (int(x) for x in bad[0])
            raise ValueError(
                f"save_fbx, the clip: node {nodes[column].name!r} channel {channel!r} frame {int(frames[frame])} "
                f"is {held[frame, column].tolist()}, its first frame {held[0, column].tolist()}: a file holds one "
                f"{channel} a node, which no curve animates"
            )
    refuse_values("save_fbx, the clip", nodes, clip.frames.shear, clip.frames.scale, frames)


def _check_takes(clips) -> None:
    """every take as :func:`_check_clip` checks one, then the list as one
    file holds it: one frame rate (a file holds one time mode), every take's
    joint orient and rotate axis the active take's (a file holds one a node,
    unkeyed; each take holds one over its frames, checked first), and no two
    takes written under one name"""
    from cgmath.hierarchy.hierarchy import _blank_take, _take_label

    for clip in clips:
        _check_clip(clip)
    rates = sorted({float(clip.fps) for clip in clips})
    if len(rates) > 1:
        raise ValueError(f"save_fbx: the takes play at {rates} fps; a file holds one frame rate")

    active = clips.active
    at     = clips._active
    for channel in ("joint_orient", "rotate_axis"):
        held = np.asarray(getattr(active.frames, channel), dtype=np.float64)[0]
        for index, clip in enumerate(clips):
            if clip is active:
                continue
            values = np.asarray(getattr(clip.frames, channel), dtype=np.float64)[0]
            bad    = np.flatnonzero(np.any(values != held, axis=-1))
            if len(bad):
                column = int(bad[0])
                raise ValueError(
                    f"save_fbx: node {clip[column].name!r} channel {channel!r} is {values[column].tolist()} in "
                    f"{_take_label(clip, index)} and {held[column].tolist()} in the active {_take_label(active, at)}: "
                    f"a file holds one {channel} a node, the active take's"
                )

    names = [clip.take.strip() or _blank_take(index) for index, clip in enumerate(clips)]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"save_fbx: two takes would be written as {twice}; take names must differ")


def _user_kinds(node) -> dict:
    return {name: spec.get("attributeType", spec.get("dataType")) for name, spec in node.user_defined_attributes.items()}


def _check_same(hierarchy, clip) -> None:
    """a clip whose nodes are not the hierarchy's: refused, naming the first
    node and what differs"""
    nodes, held = list(hierarchy), list(clip)
    if [node.name for node in nodes] != [node.name for node in held]:
        raise ValueError(
            f"save_fbx: the clip's nodes {[node.name for node in held]} are not the hierarchy's "
            f"{[node.name for node in nodes]}, in its order"
        )
    parents, kept = parents_of(nodes), parents_of(held)
    first = {
        "joint_orient": np.asarray(clip.frames.joint_orient, dtype=np.float64)[0],
        "rotate_axis":  np.asarray(clip.frames.rotate_axis, dtype=np.float64)[0],
    }
    for column, (node, other) in enumerate(zip(nodes, held)):
        for channel, mine, theirs in (
            ("parent", parents[column], kept[column]),
            ("node_type", node.node_type, other.node_type),
            ("rotate_order", node.rotate_order, other.rotate_order),
            ("segment_scale_compensate", bool(node.segment_scale_compensate), bool(other.segment_scale_compensate)),
            ("joint_orient", np.asarray(node.joint_orient, dtype=np.float64).tolist(), first["joint_orient"][column].tolist()),
            ("rotate_axis", np.asarray(node.rotate_axis, dtype=np.float64).tolist(), first["rotate_axis"][column].tolist()),
            ("user attributes", _user_kinds(node), _user_kinds(other)),
        ):
            if mine != theirs:
                raise ValueError(
                    f"save_fbx: node {node.name!r} channel {channel!r} is {theirs!r} in the clip, {mine!r} in the "
                    f"hierarchy: a clip with a skin animates the hierarchy's own nodes"
                )


def _mesh_label(mesh, nodes=None) -> str:
    """"mesh 'body'", or for a mesh of no name the node of ``nodes`` (uuid to
    node) it sits on"""
    if mesh.name:
        return f"mesh {mesh.name!r}"
    node = None if nodes is None else nodes.get(mesh.parent_uuid)
    return "the unnamed mesh" if node is None else f"the unnamed mesh on node {node.name!r}"


def _check_meshes(meshes, nodes) -> None:
    """each mesh checked; two meshes on one node of ``nodes``, the ones
    written, or a mesh on a joint (Maya's FBXExport and FBXImport both drop
    one; measured): refused"""
    held = {} if nodes is None else {node.uuid: node for node in nodes}
    on   = {}
    for mesh in meshes:
        _check_mesh(mesh)
        node = held.get(mesh.parent_uuid)
        if node is None:
            continue
        if node.node_type == "joint":
            raise ValueError(
                f"save_fbx: {_mesh_label(mesh, held)} sits on joint {node.name!r}; Maya's FBX holds a mesh on a "
                f"transform only"
            )
        if node.uuid in on:
            raise ValueError(
                f"save_fbx: {_mesh_label(on[node.uuid], held)} and {_mesh_label(mesh, held)} sit on node "
                f"{node.name!r}; a node holds one mesh"
            )
        on[node.uuid] = mesh


def _check_mesh(mesh) -> None:
    """a mesh holding holes, normals that fit no face corner, or faces
    naming points it does not hold: refused"""
    from cgmath.geometry.mesh import _corner_normals

    label = _mesh_label(mesh)
    if mesh.hole_faces is not None and len(mesh.hole_faces):
        raise ValueError(f"save_fbx: {label} holds {len(mesh.hole_faces)} polygon holes, which {FORK} does not write")
    counts  = np.asarray(mesh.counts, dtype=np.int64)
    indices = np.asarray(mesh.indices, dtype=np.int64)
    if counts.sum() != indices.size or (indices.size and (indices.min() < 0 or indices.max() >= mesh.point_count)):
        raise ValueError(f"save_fbx: {label} has faces that name points it does not hold")
    _corner_normals(mesh)


def _view_label(shape, nodes=None) -> str:
    """"camera 'shot'", "spot light 'key'", or for one of no name the node it
    sits on"""
    kind = "camera" if type(shape).__name__ == "CameraData" else f"{shape.kind} light"
    if shape.name:
        return f"{kind} {shape.name!r}"
    node = None if nodes is None else nodes.get(shape.parent_uuid)
    return f"the unnamed {kind}" if node is None else f"the unnamed {kind} on node {node.name!r}"


def _check_views(cameras, lights, meshes, nodes) -> None:
    """a camera or light on a joint, on a node holding a mesh or another
    camera or light, on a node holding a rotate axis or children (the turn
    Maya's exporter writes is the node's own, and a child would move with
    it): refused. A camera scale that is not 1, a dropoff that is not 0:
    one warning each, fbx holds neither"""
    import warnings

    held   = {} if nodes is None else {node.uuid: node for node in nodes}
    on     = {}
    scaled = []
    drops  = []
    for mesh in meshes or []:
        if mesh.parent_uuid in held:
            on[mesh.parent_uuid] = _mesh_label(mesh, held)
    for shape in list(cameras or []) + list(lights or []):
        label = _view_label(shape, held)
        if type(shape).__name__ == "CameraData" and shape.camera_scale != 1.0:
            scaled.append(label)
        if type(shape).__name__ == "LightData" and shape.dropoff != 0.0:
            drops.append(label)
        node = held.get(shape.parent_uuid)
        if node is None:
            continue
        if node.node_type == "joint":
            raise ValueError(f"save_fbx: {label} sits on joint {node.name!r}; Maya holds a camera or light on a transform")
        if node.uuid in on:
            raise ValueError(f"save_fbx: {on[node.uuid]} and {label} sit on node {node.name!r}; a node holds one shape")
        if np.any(np.asarray(node.rotate_axis, dtype=np.float64) != 0.0):
            raise ValueError(
                f"save_fbx: {label} sits on node {node.name!r}, which holds a rotate axis; the turn to fbx's camera and "
                f"light axes is the node's post rotation, so {FORK} writes a camera or light on a node with none"
            )
        if any(other.parent_uuid == node.uuid for other in nodes):
            raise ValueError(
                f"save_fbx: {label} sits on node {node.name!r}, which has children; they would turn with the node, so "
                f"{FORK} writes a camera or light on a leaf node"
            )
        on[node.uuid] = label
    if scaled:
        warnings.warn(f"save_fbx: {', '.join(scaled)}: fbx holds no camera scale; the file holds none, as Maya's own export", stacklevel=3)
    if drops:
        warnings.warn(f"save_fbx: {', '.join(drops)}: fbx holds no dropoff; the file holds none, as Maya's own export", stacklevel=3)


def _is_line(curve) -> bool:
    """whether Maya's exporter writes the curve as a line: of degree 1, not
    periodic, holding no knots and no weights (a line holds neither)"""
    return curve.degree == 1 and not curve.periodic and curve.knots is None and curve.weights is None


def _check_nurbs(curves, surfaces, meshes, cameras, lights, nodes) -> None:
    """a curve or surface whose points are not 3D, whose weights are not
    one a control point, finite and above zero, on a joint, or on a node
    of ``nodes``, the ones written, holding a mesh, a camera, a light or
    another curve or surface: refused. The curves of degree 1 holding no
    knots of one node are one shape, the lines of one FbxLine, as Maya's
    exporter writes them; it writes a transform of two NURBS curves as an
    empty node (measured)"""
    from cgmath.geometry.bspline import _checked_weights

    held  = {} if nodes is None else {node.uuid: node for node in nodes}
    on    = {}
    lines = set()
    for mesh in meshes or []:
        if mesh.parent_uuid in held:
            on[mesh.parent_uuid] = _mesh_label(mesh, held)
    for shape in list(cameras or []) + list(lights or []):
        if shape.parent_uuid in held:
            on[shape.parent_uuid] = _view_label(shape, held)
    for kind, shapes in (("curve", curves or []), ("surface", surfaces or [])):
        for at, shape in enumerate(shapes):
            label = f"{kind} {at} of {kind}s="
            width = np.asarray(shape.points).shape[-1]
            if width != 3:
                raise ValueError(f"save_fbx: {label} holds points of {width} components; a .fbx holds 3D points")
            if shape.weights is not None:
                try:
                    _checked_weights(shape.weights, np.asarray(shape.points).shape[:-1])
                except ValueError as error:
                    raise ValueError(f"save_fbx: {label}: {error}") from None
            node = held.get(shape.parent_uuid)
            if node is None:
                continue
            if node.node_type == "joint":
                raise ValueError(
                    f"save_fbx: {label} sits on joint {node.name!r}; Maya's FBX holds a {kind} on a transform only"
                )
            line = kind == "curve" and _is_line(shape)
            if node.uuid in on and not (line and node.uuid in lines):
                raise ValueError(
                    f"save_fbx: {on[node.uuid]} and {label} sit on node {node.name!r}; a node holds one shape, or "
                    f"the curves of degree 1 holding no knots as the lines of one"
                )
            on[node.uuid] = label
            if line:
                lines.add(node.uuid)


def _pil():
    """PIL's Image module, or None where pillow is not installed"""
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _material_label(material, meshes=None) -> str:
    """"material 'skin'", or for one of no name the mesh of ``meshes`` (uuid
    to mesh) it covers"""
    if material.name:
        return f"material {material.name!r}"
    mesh = None if meshes is None else meshes.get(material.mesh_uuid)
    return "the unnamed material" if mesh is None else f"the unnamed material on {_mesh_label(mesh)}"


# the fields the records of one material (one uuid) agree on: everything but
# its name, its mesh, its faces and the uuid itself
_SHADING_FIELDS = (
    "kind", "color", "transparency", "ambient_color", "incandescence", "diffuse", "specular_color", "cosine_power",
    "reflectivity", "texture", "texture_file", "uv_set", "wrap_u", "wrap_v", "normal_map", "normal_map_file",
    "normal_map_kind", "bump_depth",
)


def _differing(a, b) -> list:
    """the shading fields two records of one material disagree on"""
    out = []
    for name in _SHADING_FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            same = x is None and y is None
        elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            same = np.array_equal(np.asarray(x), np.asarray(y))
        else:
            same = x == y
        if not same:
            out.append(name)
    return out


def _check_materials(materials, meshes) -> None:
    """a material naming no mesh, covering faces its mesh does not hold or a
    face twice, two materials covering one face, materials leaving faces of
    their mesh uncovered, a material covering the whole mesh beside another
    on it, a texture or normal map that is not (H, W, C) uint8 with 1 to 4
    channels, records sharing a uuid that differ past their faces (one
    material is written per uuid): refused. A map with no PIL to encode it:
    ImportError"""
    by_uuid = {mesh.uuid: mesh for mesh in meshes}
    covered = {}
    shaders = {}
    for material in materials:
        label = _material_label(material, by_uuid)
        mesh  = by_uuid.get(material.mesh_uuid)
        if mesh is None:
            raise ValueError(
                f"save_fbx: {label} names no mesh of meshes=: mesh_uuid {material.mesh_uuid!r}; set it to its mesh's uuid"
            )
        if material.uuid is not None:
            first = shaders.setdefault(material.uuid, (label, mesh, material))
            if first[2] is not material:
                differ = _differing(first[2], material)
                if differ:
                    raise ValueError(
                        f"save_fbx: {label} on {_mesh_label(mesh)} and {first[0]} on {_mesh_label(first[1])} share uuid "
                        f"{material.uuid!r} but differ in {', '.join(differ)}; one material is written per uuid, so "
                        f"records sharing one agree but for their faces: give one of them a uuid of its own"
                    )
        for channel, what in (("texture", "a texture"), ("normal_map", "a normal map")):
            pixels = getattr(material, channel)
            if pixels is None:
                continue
            pixels = np.asarray(pixels)
            if pixels.dtype != np.uint8:
                raise ValueError(
                    f"save_fbx: {label} channel {channel!r} is {pixels.dtype.name}; a .fbx embeds an 8 bit PNG, so {FORK} "
                    f"writes a uint8 map: scale it to 0..255"
                )
            if pixels.ndim != 3 or pixels.shape[2] not in (1, 2, 3, 4) or not pixels.shape[0] or not pixels.shape[1]:
                raise ValueError(
                    f"save_fbx: {label} channel {channel!r} is {pixels.shape}, not (H, W, C) with 1 to 4 channels"
                )
            if _pil() is None:
                raise ImportError(f"save_fbx: {label} holds {what}, written as a PNG through PIL (pillow), which is not installed")
        count = len(mesh.counts)
        held  = covered.setdefault(mesh.uuid, [])
        if material.faces is None:
            held.append((label, None))
            continue
        faces = np.asarray(material.faces, dtype=np.int64)
        if faces.size and (faces.min() < 0 or faces.max() >= count):
            raise ValueError(f"save_fbx: {label} channel 'faces' names faces {_mesh_label(mesh)} of {count} faces does not hold")
        if np.unique(faces).size != faces.size:
            raise ValueError(f"save_fbx: {label} channel 'faces' names a face twice")
        held.append((label, faces))
    for uuid, held in covered.items():
        mesh  = by_uuid[uuid]
        count = len(mesh.counts)
        whole = [label for label, faces in held if faces is None]
        if whole and len(held) > 1:
            others = [label for label, faces in held if label != whole[0]] if len(whole) == 1 else whole[1:]
            raise ValueError(
                f"save_fbx: {whole[0]} covers the whole {_mesh_label(mesh)} while {', '.join(others)} "
                f"{'covers' if len(others) == 1 else 'cover'} it too; a face holds one material"
            )
        if whole:
            continue
        owner = np.full(count, -1, dtype=np.int64)
        for place, (label, faces) in enumerate(held):
            twice = faces[owner[faces] >= 0]
            if twice.size:
                raise ValueError(
                    f"save_fbx: {held[int(owner[twice[0]])][0]} and {label} both cover face {int(twice[0])} of "
                    f"{_mesh_label(mesh)}; a face holds one material"
                )
            owner[faces] = place
        left = int(np.count_nonzero(owner < 0))
        if left:
            raise ValueError(
                f"save_fbx: {', '.join(label for label, _ in held)} {'leaves' if len(held) == 1 else 'leave'} {left} "
                f"faces of {_mesh_label(mesh)} without a material; a mesh is covered whole or not at all"
            )


def _morph_label(morph, meshes=None) -> str:
    """"blend shape target 'smile'", on its mesh of ``meshes`` (uuid to mesh)
    when it names one"""
    mesh = None if meshes is None else meshes.get(morph.mesh_uuid)
    return f"blend shape target {morph.name!r}" + ("" if mesh is None else f" of {_mesh_label(mesh)}")


def _morph_channels(morphs) -> dict:
    """{(mesh uuid, deformer name, target name): [morphs]}, in first
    appearance order: the channels the targets form"""
    channels = {}
    for morph in morphs:
        channels.setdefault((morph.mesh_uuid, morph.deformer or "blendShape", morph.name), []).append(morph)
    return channels


def _check_morphs(morphs, meshes) -> None:
    """a morph naming no mesh, moving points its mesh does not hold or as
    many offsets as it has not indices, two deformer names on one mesh
    (Maya's import deforms a mesh by its last blend shape deformer alone,
    measured), a channel without one target at 1, with two targets at one
    ``at``, an in-between above 1, or targets of two weights: refused"""
    by_uuid = {mesh.uuid: mesh for mesh in meshes}
    for morph in morphs:
        label = _morph_label(morph, by_uuid)
        mesh  = by_uuid.get(morph.mesh_uuid)
        if mesh is None:
            raise ValueError(f"save_fbx: {label} names no mesh of meshes=: mesh_uuid {morph.mesh_uuid!r}; set it to its mesh's uuid")
        indices = np.asarray(morph.indices, dtype=np.int64)
        offsets = np.asarray(morph.offsets, dtype=np.float64)
        if indices.size != len(offsets):
            raise ValueError(f"save_fbx: {label} holds {len(offsets)} offsets for {indices.size} indices")
        if indices.size and (indices.min() < 0 or indices.max() >= mesh.point_count):
            raise ValueError(f"save_fbx: {label} channel 'indices' names points {_mesh_label(mesh)} of {mesh.point_count} points does not hold")
    deformers = {}
    for uuid, deformer, _ in _morph_channels(morphs):
        deformers.setdefault(uuid, [])
        if deformer not in deformers[uuid]:
            deformers[uuid].append(deformer)
    for uuid, names in deformers.items():
        if len(names) > 1:
            raise ValueError(
                f"save_fbx: {_mesh_label(by_uuid[uuid])} holds blend shape deformers {names}; Maya's import deforms a "
                f"mesh by its last blend shape deformer alone (measured), so {FORK} writes one a mesh"
            )
    for (uuid, deformer, name), held in _morph_channels(morphs).items():
        label = f"blend shape target {name!r} of {_mesh_label(by_uuid[uuid])} under deformer {deformer!r}"
        ats   = [float(morph.at) for morph in held]
        if ats.count(1.0) != 1:
            raise ValueError(
                f"save_fbx: {label} has {ats.count(1.0)} targets at 1 of {len(held)}; a channel holds one full target, at 1, and "
                f"its in-betweens below 1"
            )
        if len(set(ats)) != len(ats):
            raise ValueError(f"save_fbx: {label} has two targets at {sorted(x for x in ats if ats.count(x) > 1)[0]}; a weight is reached by one")
        if max(ats) > 1.0:
            raise ValueError(f"save_fbx: {label} has a target at {max(ats)}, above its full target at 1")
        weights = {float(morph.weight) for morph in held}
        if len(weights) > 1:
            raise ValueError(f"save_fbx: {label} holds weights {sorted(weights)}; a channel holds one weight")


def _write_morphs(exporter, mesh, morphs, shape) -> None:
    """``morphs``, the targets of ``mesh``, on ``shape`` (its FbxMesh) as
    Maya's exporter writes a blendShape (measured): one FbxBlendShape named
    after the targets' deformer, one channel the targets of a name, named
    "deformer.name", its DeformPercent the weight times ``WEIGHT_UNIT``, one
    FbxShape a target over every point of the mesh, moved by the offsets, by
    ``at``, each added at its ``at`` times ``WEIGHT_UNIT``, an in-between
    named "name_50" for one at 0.5 (Maya's import mangles a dot); no
    normals"""
    from cgmath.formats._load import WEIGHT_UNIT
    from cgmath.hierarchy.hierarchy import FBX as fbx

    scene    = exporter.scene
    points   = np.asarray(mesh.points, dtype=np.float64)
    deformer = None
    for (_, dname, name), held in _morph_channels(morphs).items():
        if deformer is None:
            deformer = fbx.FbxBlendShape.Create(scene, dname)
        channel = fbx.FbxBlendShapeChannel.Create(scene, f"{dname}.{name}")
        channel.DeformPercent.Set(float(held[0].weight) * WEIGHT_UNIT)
        for morph in sorted(held, key=lambda x: float(x.at)):
            at     = float(morph.at)
            target = fbx.FbxShape.Create(scene, name if at == 1.0 else f"{name}_{round(at * WEIGHT_UNIT):g}")
            moved  = points.copy()
            moved[np.asarray(morph.indices, dtype=np.int64)] += np.asarray(morph.offsets, dtype=np.float64)
            target.InitControlPoints(len(moved))
            for index, point in enumerate(moved.tolist()):
                target.SetControlPointAt(fbx.FbxVector4(*point, 1.0), index)
            channel.AddTargetShape(target, at * WEIGHT_UNIT)
        deformer.AddBlendShapeChannel(channel)
    shape.AddDeformer(deformer)


@contextlib.contextmanager
def _texture_folder(materials):
    """a folder for the PNGs the textures and normal maps of ``materials``
    are written to on their way into the file, gone when the block ends;
    None for no map"""
    if materials is None or all(material.texture is None and material.normal_map is None for material in materials):
        yield None
        return
    from cgmath.formats._fbx_io import media_folder

    # an ASCII path: the sdk takes no other
    folder = media_folder()
    if folder is None:
        folder = tempfile.mkdtemp(prefix="cgmath_fbx_textures_")
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _check_uvs(uvs, meshes) -> None:
    """a UV set naming no mesh, of no name, of a name another set of its
    mesh holds or that is not ASCII, whose faces are not its mesh's, or
    naming UVs it does not hold: refused"""
    by_uuid = {mesh.uuid: mesh for mesh in meshes}
    names   = {}
    for place, uv in enumerate(uvs):
        name  = uv.name
        label = f"UV set {place} {name!r}"
        mesh  = by_uuid.get(uv.mesh_uuid)
        if mesh is None:
            raise ValueError(
                f"save_fbx: {label} names no mesh of meshes=: mesh_uuid {uv.mesh_uuid!r}; set it to its mesh's uuid"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(f"save_fbx: {label} has no name; name each set")
        if not name.isascii():
            raise ValueError(f"save_fbx: {label} has a name that is not ASCII, which the fbx sdk does not write")
        held = names.setdefault(mesh.uuid, [])
        if name in held:
            raise ValueError(f"save_fbx: {label} has the name of set {held.index(name)} of {_mesh_label(mesh)}; name each set apart")
        held.append(name)
        counts = np.asarray(mesh.counts, dtype=np.int64)
        got    = np.asarray(uv.counts, dtype=np.int64)
        if not np.array_equal(got, counts):
            unmapped = int(np.count_nonzero(got == 0)) if got.shape == counts.shape else None
            why = (
                f"leaves {unmapped} faces without UVs" if unmapped
                else f"holds {got.size} faces of {got.sum()} corners, {_mesh_label(mesh)} {counts.size} of {counts.sum()}"
            )
            raise ValueError(f"save_fbx: {label} channel 'counts' {why}; a set maps every face of its mesh")
        indices = np.asarray(uv.indices, dtype=np.int64)
        points  = np.asarray(uv.points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"save_fbx: {label} channel 'points' is {points.shape}, not (N, 2)")
        if indices.size != counts.sum() or (indices.size and (indices.min() < 0 or indices.max() >= len(points))):
            raise ValueError(f"save_fbx: {label} channel 'indices' names UVs the set does not hold")


def _check_skins(skins, meshes, hierarchy) -> None:
    """a skin naming no mesh, a second skin on a mesh, a skin whose
    influences are not nodes of the hierarchy, or whose weights are not its
    mesh's points by its influences: refused"""
    by_uuid = {mesh.uuid: mesh for mesh in meshes}
    on      = {}
    for skin in skins:
        label = f"skin {skin.name!r}"
        mesh  = by_uuid.get(skin.mesh_uuid)
        if mesh is None:
            raise ValueError(
                f"save_fbx: {label} names no mesh of meshes=: mesh_uuid {skin.mesh_uuid!r}; set it to its mesh's uuid"
            )
        if mesh.uuid in on:
            raise ValueError(
                f"save_fbx: skins {on[mesh.uuid]!r} and {skin.name!r} both name {_mesh_label(mesh)}; a mesh holds one skin"
            )
        on[mesh.uuid] = skin.name
        _check_skin(skin, mesh, hierarchy)


def _check_still(skins, meshes, hierarchy, clip) -> None:
    """a clip moving a skinned mesh's node off its bind world: refused.
    Maya's skinCluster deforms in world space and the node's own transform
    applies on top (measured: the mesh lands at the skin's result times the
    node's bind inverse times its world), so a node the clip moves moves the
    mesh twice; cgmath's deformer leaves the node's motion out"""
    by_uuid = {mesh.uuid: mesh for mesh in meshes}
    column  = {node.uuid: i for i, node in enumerate(hierarchy)}
    bind    = np.asarray(hierarchy.world_matrix, dtype=np.float64)
    worlds  = np.asarray(clip.frames.world_matrix, dtype=np.float64)
    frames  = clip.start_frame + np.arange(clip.frame_count)
    for skin in skins:
        mesh = by_uuid[skin.mesh_uuid]
        at   = column.get(mesh.parent_uuid)
        if at is None:
            continue
        off = np.flatnonzero(np.abs(worlds[:, at] - bind[at]).max(axis=(1, 2)) > STILL * max(1.0, float(np.abs(bind[at]).max())))
        if len(off):
            raise ValueError(
                f"save_fbx: the clip moves node {hierarchy[at].name!r}, which holds skinned {_mesh_label(mesh)}, off "
                f"its bind world on frame {int(frames[off[0]])}: Maya's skinCluster applies the node's own transform on "
                f"top of the skin, so the mesh would move twice"
            )


def _check_skin(skin, mesh, hierarchy) -> None:
    """a skin whose influences are not nodes of the hierarchy, or whose
    weights are not the mesh's points by its influences: refused"""
    label      = f"skin {skin.name!r}"
    influences = [str(x) for x in skin.influences]
    names      = [node.name for node in hierarchy]
    for problem, found in (
        ("lists twice", sorted({x for x in influences if influences.count(x) > 1})),
        ("names no node of the hierarchy", [x for x in influences if x not in names]),
        ("names several nodes of the hierarchy", [x for x in influences if names.count(x) > 1]),
    ):
        if found:
            raise ValueError(f"save_fbx: {label} {problem}: influences {found}")
    weights = np.asarray(skin.weights)
    rows    = weights.shape[0] if weights.ndim == 2 else None
    if rows != mesh.point_count:
        raise ValueError(
            f"save_fbx: {label} channel 'weights' holds {rows} rows, {_mesh_label(mesh)} holds {mesh.point_count} points"
        )
    if weights.shape[1] != len(influences):
        raise ValueError(
            f"save_fbx: {label} channel 'weights' holds {weights.shape[1]} columns for its {len(influences)} influences"
        )


def _index(fbx_node, index: int, fbx) -> None:
    """the node's place in the hierarchy's list, kept on it"""
    prop = fbx.FbxProperty.Create(fbx_node, fbx.FbxIntDT, ORDER_PROPERTY, "")
    prop.Set(index)


def _fbx_matrix(matrix, fbx):
    """an FbxMatrix holding the (4, 4) row-vector ``matrix`` element for
    element"""
    held = fbx.FbxMatrix()
    for row, values in enumerate(np.asarray(matrix, dtype=np.float64).tolist()):
        for column, value in enumerate(values):
            held.Set(row, column, value)
    return held


def _fbx_affine(matrix, fbx):
    """an FbxAMatrix of the (4, 4) row-vector ``matrix``, scaled, turned and
    moved; the sdk's affine matrix takes no element"""
    from cgmath.transforms import matrix_to_euler

    matrix = np.asarray(matrix, dtype=np.float64)
    scale  = np.linalg.norm(matrix[:3, :3], axis=1)
    if np.linalg.det(matrix[:3, :3]) < 0.0:
        scale[2] = -scale[2]
    turn         = np.eye(4)
    turn[:3, :3] = matrix[:3, :3] / scale[:, None]
    rotate       = np.degrees(matrix_to_euler(turn, 0))[0]
    held         = fbx.FbxAMatrix()
    held.SetTRS(fbx.FbxVector4(*matrix[3, :3], 1.0), fbx.FbxVector4(*rotate, 0.0), fbx.FbxVector4(*scale, 0.0))
    return held


def _write_mesh(exporter, mesh, node) -> tuple:
    """(the node, the FbxMesh): the mesh as the shape of ``node``, the
    FbxNode of the transform its ``parent_uuid`` names, or of a new root
    node at identity named after the mesh for None; its points, faces and
    normals, one a face corner through their indices (the stored ones, or
    with none stored and ``hard_edges`` held the ones derived between them;
    none for a mesh holding neither); its colours as one
    set, :data:`COLOR_SET`, on the mesh's first layer where Maya's exporter
    puts a set, by control point for a colour a vertex and by polygon vertex
    otherwise, index to direct as Maya writes both; the FbxMesh named after
    the mesh, unnamed for a mesh of no name"""
    from cgmath.geometry.mesh import _corner_colors, _corner_normals
    from cgmath.hierarchy.hierarchy import FBX as fbx

    scene = exporter.scene
    if node is None:
        node = fbx.FbxNode.Create(scene, mesh.name or "mesh")
        node.SetTransformationInheritType(fbx.FbxTransform.EInheritType.eInheritRSrs)
        scene.GetRootNode().AddChild(node)

    # Maya's FBXExport leaves the FbxMesh unnamed: a mesh of no name reads back
    # with none
    shape  = fbx.FbxMesh.Create(scene, mesh.name or "")
    points = np.asarray(mesh.points, dtype=np.float64).tolist()
    shape.InitControlPoints(len(points))
    for index, point in enumerate(points):
        shape.SetControlPointAt(fbx.FbxVector4(*point, 1.0), index)
    indices = np.asarray(mesh.indices, dtype=np.int64).tolist()
    start   = 0
    for size in np.asarray(mesh.counts, dtype=np.int64).tolist():
        shape.BeginPolygon(-1, -1, -1, False)
        for vertex in indices[start : start + size]:
            shape.AddPolygon(vertex)
        shape.EndPolygon()
        start += size

    normals, ids = _corner_normals(mesh)
    if normals is not None:
        element = shape.CreateElementNormal()
        element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
        element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
        direct = element.GetDirectArray()
        for normal in np.asarray(normals, dtype=np.float64).tolist():
            direct.Add(fbx.FbxVector4(*normal, 0.0))
        held = element.GetIndexArray()
        for index in np.asarray(ids, dtype=np.int64).tolist():
            held.Add(index)

    colors, ids = _corner_colors(mesh)
    if colors is not None:
        # on the first layer by hand: the sdk's CreateElementVertexColor would
        # take the last layer and pad the ones under it, as it does a material
        per_vertex = mesh.color_indices is None
        if shape.GetLayerCount() == 0:
            shape.CreateLayer()
        element = fbx.FbxLayerElementVertexColor.Create(shape, COLOR_SET)
        shape.GetLayer(0).SetVertexColors(element)
        element.SetMappingMode(
            fbx.FbxLayerElement.EMappingMode.eByControlPoint if per_vertex else fbx.FbxLayerElement.EMappingMode.eByPolygonVertex
        )
        element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
        direct = element.GetDirectArray()
        for color in np.asarray(colors, dtype=np.float64).tolist():
            direct.Add(fbx.FbxColor(*color))
        held = element.GetIndexArray()
        for index in (range(len(colors)) if per_vertex else np.asarray(ids, dtype=np.int64).tolist()):
            held.Add(int(index))
    node.SetNodeAttribute(shape)
    return node, shape


# the name a mesh's colours are written under: Maya's name for a first colour
# set (its importer names an unnamed element "colorSet0")
COLOR_SET = "colorSet1"


def _enum(owner, scope: str, name: str) -> int:
    """an fbx enum's int, which the binding's property setters take"""
    from cgmath.hierarchy.hierarchy import _fbx_enum

    value = _fbx_enum(owner, scope, name)
    return int(getattr(value, "value", value))


def _png_name(file, fallback, taken: set) -> str:
    """the file name a map is embedded under: the stem of ``file``, the name
    it came from, else ``fallback`` (the material's name), else "texture",
    as a .png, told apart from the names ``taken`` by a number"""
    stem = os.path.splitext(os.path.basename(file or ""))[0] or fallback or "texture"
    stem = "".join(x if x.isascii() and (x.isalnum() or x in "-_.") else "_" for x in stem)
    name = stem + ".png"
    k    = 1
    while name in taken:
        name = f"{stem}_{k}.png"
        k += 1
    taken.add(name)
    return name


def _write_png(pixels, file, fallback, folder: str, taken: set) -> str:
    """``pixels``, an (H, W, C) uint8 array, written as a PNG into ``folder``
    as L, LA, RGB or RGBA by its channels, named by :func:`_png_name`; the
    path"""
    Image  = _pil()
    pixels = np.ascontiguousarray(np.asarray(pixels, dtype=np.uint8))
    modes  = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}
    mode   = modes[pixels.shape[2]]
    image  = Image.fromarray(pixels[:, :, 0] if mode == "L" else pixels, mode)
    path   = os.path.join(folder, _png_name(file, fallback, taken))
    image.save(path, format="PNG")
    return path


def _surface(scene, material, folder: str, taken: set, wraps: dict, fbx):
    """the FbxSurfaceMaterial of ``material``, as Maya's exporter writes a
    lambert or a phong (measured): the colour as the diffuse colour, the
    diffuse as the diffuse factor, the ambient colour, incandescence and
    transparency as the ambient, emissive and transparent colours with
    factors of 1, a phong's specular colour with a factor of 1, its cosine
    power as the shininess, its reflectivity as the reflection factor with
    a reflection colour of 0; a texture as a PNG in ``folder`` (embedded by
    the export, its name told from ``taken``) feeding the diffuse colour
    through an FbxFileTexture reading the UV set and wrapping as the
    material says; a normal map the same way on the slot Maya's exporter
    writes a bump2d's map to (measured): NormalMap for tangent space
    normals, Bump for a bump map, the depth as BumpFactor"""
    phong = material.kind == "phong"
    made  = (fbx.FbxSurfacePhong if phong else fbx.FbxSurfaceLambert).Create(scene, material.name or "")
    made.Diffuse.Set(fbx.FbxDouble3(*[float(x) for x in material.color]))
    made.DiffuseFactor.Set(float(material.diffuse))
    made.Ambient.Set(fbx.FbxDouble3(*[float(x) for x in material.ambient_color]))
    made.AmbientFactor.Set(1.0)
    made.Emissive.Set(fbx.FbxDouble3(*[float(x) for x in material.incandescence]))
    made.EmissiveFactor.Set(1.0)
    made.TransparentColor.Set(fbx.FbxDouble3(*[float(x) for x in material.transparency]))
    made.TransparencyFactor.Set(1.0)
    if phong:
        made.Specular.Set(fbx.FbxDouble3(*[float(x) for x in material.specular_color]))
        made.SpecularFactor.Set(1.0)
        made.Shininess.Set(float(material.cosine_power))
        made.Reflection.Set(fbx.FbxDouble3(0.0, 0.0, 0.0))
        made.ReflectionFactor.Set(float(material.reflectivity))

    def file_texture(pixels, file, fallback):
        path    = _write_png(pixels, file, fallback, folder, taken)
        texture = fbx.FbxFileTexture.Create(scene, os.path.splitext(os.path.basename(path))[0])
        texture.SetFileName(path)
        texture.SetRelativeFileName(os.path.basename(path))
        texture.SetTextureUse(fbx.FbxTexture.ETextureUse.eStandard)
        texture.SetMappingType(fbx.FbxTexture.EMappingType.eUV)
        texture.SetMaterialUse(fbx.FbxFileTexture.EMaterialUse.eModelMaterial)
        if material.uv_set:
            texture.UVSet.Set(fbx.FbxString(material.uv_set))
        texture.WrapModeU.Set(wraps[material.wrap_u])
        texture.WrapModeV.Set(wraps[material.wrap_v])
        return texture

    if material.texture is not None:
        made.Diffuse.ConnectSrcObject(file_texture(material.texture, material.texture_file, material.name))
    if material.normal_map is not None:
        slot = made.NormalMap if material.normal_map_kind == "tangent" else made.Bump
        slot.ConnectSrcObject(file_texture(material.normal_map, material.normal_map_file, f"{material.name or 'texture'}_normal"))
        made.BumpFactor.Set(float(material.bump_depth))
    return made


def _write_materials(exporter, mesh, materials, node, shape, folder, shaders: dict, taken: set) -> None:
    """``materials``, the ones covering ``mesh``, on ``node`` (its FbxNode)
    in their order, and the material element of ``shape`` (its FbxMesh):
    all-same for one material covering the whole mesh, by-polygon for
    materials covering faces; each an FbxSurfaceMaterial as :func:`_surface`
    makes it, one per uuid: ``shaders`` holds the ones made so far by uuid,
    across the meshes, and a record of a material already there puts that
    one on its node (a material on several meshes, one record a mesh);
    ``taken`` the PNG names written so far. The element goes on the mesh's
    first layer, where Maya's exporter puts it: the sdk's
    ``CreateElementMaterial`` takes the last layer (a UV set makes one each)
    and pads the layers under it with empty eNone elements, which a reader
    meets first"""
    from cgmath.hierarchy.hierarchy import FBX as fbx

    scene = exporter.scene
    wraps = {"repeat": _enum(fbx.FbxTexture, "EWrapMode", "eRepeat"), "clamp": _enum(fbx.FbxTexture, "EWrapMode", "eClamp")}
    for material in materials:
        made = None if material.uuid is None else shaders.get(material.uuid)
        if made is None:
            made = _surface(scene, material, folder, taken, wraps, fbx)
            if material.uuid is not None:
                shaders[material.uuid] = made
        node.AddMaterial(made)

    if shape.GetLayerCount() == 0:
        shape.CreateLayer()
    element = fbx.FbxLayerElementMaterial.Create(shape, "")
    shape.GetLayer(0).SetMaterials(element)
    element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
    indices = element.GetIndexArray()
    if len(materials) == 1 and materials[0].faces is None:
        element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eAllSame)
        indices.Add(0)
        return
    element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygon)
    owner = np.zeros(len(mesh.counts), dtype=np.int64)
    for place, material in enumerate(materials):
        owner[np.asarray(material.faces, dtype=np.int64)] = place
    for index in owner.tolist():
        indices.Add(index)


def _view_node(exporter, shape, node, axis, fbx):
    """the FbxNode a camera or light goes on: ``node``, or a new root node at
    identity named after the shape for None; turned to fbx's camera or light
    axes with the post rotation Maya's exporter writes, the inverse of the
    rotate axis ``axis`` (:data:`_VIEW_AXES` of the reader), None for none"""
    from cgmath.formats.fbx import _post_rotation

    scene = exporter.scene
    if node is None:
        node = fbx.FbxNode.Create(scene, shape.name or ("camera" if type(shape).__name__ == "CameraData" else "light"))
        node.SetTransformationInheritType(fbx.FbxTransform.EInheritType.eInheritRSrs)
        node.SetRotationActive(True)
        scene.GetRootNode().AddChild(node)
    if axis is not None:
        node.SetPostRotation(fbx.FbxNode.EPivotSet.eSourcePivot, _post_rotation(axis))
    return node


def _write_camera(exporter, camera, node, world):
    """``camera`` as the shape of ``node`` (see :func:`_view_node`), its
    attributes as Maya's exporter writes a camera (measured): the focal
    length and film back as they are, the gate fit for the film fit, the
    ortho zoom as the orthographic width over 30, the position, up vector
    and interest 5 units down the view from ``world``, the node's; the
    FbxNode"""
    from cgmath.hierarchy.hierarchy import FBX as fbx

    node  = _view_node(exporter, camera, node, (0.0, 90.0, 0.0), fbx)
    shape = fbx.FbxCamera.Create(exporter.scene, camera.name or "")
    fits  = {"fill": "eFitFill", "horizontal": "eFitHorizontal", "vertical": "eFitVertical", "overscan": "eFitOverscan"}
    shape.FocalLength.Set(float(camera.focal_length))
    shape.FilmWidth.Set(float(camera.horizontal_film_aperture))
    shape.FilmHeight.Set(float(camera.vertical_film_aperture))
    shape.FilmAspectRatio.Set(float(camera.horizontal_film_aperture / camera.vertical_film_aperture))
    shape.FilmOffsetX.Set(float(camera.film_offset[0]))
    shape.FilmOffsetY.Set(float(camera.film_offset[1]))
    shape.FilmSqueezeRatio.Set(float(camera.lens_squeeze_ratio))
    shape.ApertureMode.Set(_enum(fbx.FbxCamera, "EApertureMode", "eFocalLength"))
    shape.GateFit.Set(_enum(fbx.FbxCamera, "EGateFit", fits[camera.film_fit]))
    shape.NearPlane.Set(float(camera.near_clip_plane))
    shape.FarPlane.Set(float(camera.far_clip_plane))
    shape.ProjectionType.Set(_enum(fbx.FbxCamera, "EProjectionType", "eOrthogonal" if camera.orthographic else "ePerspective"))
    if camera.orthographic:
        # Maya writes the ortho zoom of an orthographic camera alone (measured)
        shape.OrthoZoom.Set(float(camera.orthographic_width) / 30.0)
    eye      = np.asarray(world, dtype=np.float64)[3, :3]
    up       = np.asarray(world, dtype=np.float64)[1, :3]
    forward  = -np.asarray(world, dtype=np.float64)[2, :3]
    interest = eye + 5.0 * forward / max(float(np.linalg.norm(forward)), 1e-12)
    shape.Position.Set(fbx.FbxDouble3(*eye.tolist()))
    shape.UpVector.Set(fbx.FbxDouble3(*(up / max(float(np.linalg.norm(up)), 1e-12)).tolist()))
    shape.InterestPosition.Set(fbx.FbxDouble3(*interest.tolist()))
    node.SetNodeAttribute(shape)
    return node


def _write_light(exporter, light, node, world):
    """``light`` as the shape of ``node`` (see :func:`_view_node`), its
    attributes as Maya's exporter writes a light (measured): the type, the
    colour, the intensity times 100, the decay rate as the decay type, a
    spot's cone as the inner angle and its cone plus twice its penumbra as
    the outer; the FbxNode"""
    from cgmath.hierarchy.hierarchy import FBX as fbx

    kinds = {"point": "ePoint", "directional": "eDirectional", "spot": "eSpot"}
    decay = {0: "eNone", 1: "eLinear", 2: "eQuadratic", 3: "eCubic"}
    node  = _view_node(exporter, light, node, None if light.kind == "point" else (90.0, 0.0, 0.0), fbx)
    shape = fbx.FbxLight.Create(exporter.scene, light.name or "")
    shape.LightType.Set(_enum(fbx.FbxLight, "EType", kinds[light.kind]))
    shape.Color.Set(fbx.FbxDouble3(*[float(x) for x in light.color]))
    shape.Intensity.Set(float(light.intensity) * 100.0)
    shape.DecayType.Set(_enum(fbx.FbxLight, "EDecayType", decay[int(light.decay_rate)]))
    shape.CastLight.Set(True)
    if light.kind == "spot":
        shape.InnerAngle.Set(float(light.cone_angle))
        shape.OuterAngle.Set(float(light.cone_angle) + 2.0 * float(light.penumbra_angle))
    node.SetNodeAttribute(shape)
    return node


def _write_shapes(exporter, curves, surfaces, by_uuid: dict, made: dict, count: int, fbx) -> None:
    """the curves and surfaces, each the shape of the FbxNode of ``by_uuid``
    its ``parent_uuid`` names, or of a new root node at identity named
    "curve" or "surface", kept in ``made`` by the shape's uuid and placed
    after the ``count`` nodes of the hierarchy and the ones made so far: the
    curves of degree 1 holding no knots of one node as the lines of one
    FbxLine (:func:`_write_lines`), every other curve and every surface as a
    NURBS shape (:func:`_write_nurbs`)"""
    scene = exporter.scene
    lines = {}  # the node's uuid, or the curve's own on a new node, to (the FbxNode, its lines)
    nurbs = []
    for kind, shapes in (("curve", curves or []), ("surface", surfaces or [])):
        for shape in shapes:
            fbx_node = by_uuid.get(shape.parent_uuid)
            if fbx_node is None:
                fbx_node = fbx.FbxNode.Create(scene, kind)
                fbx_node.SetTransformationInheritType(fbx.FbxTransform.EInheritType.eInheritRSrs)
                scene.GetRootNode().AddChild(fbx_node)
                _index(fbx_node, count + len(made), fbx)
                made[shape.uuid] = fbx_node
            if kind == "curve" and _is_line(shape):
                key = shape.parent_uuid if shape.parent_uuid in by_uuid else shape.uuid
                lines.setdefault(key, (fbx_node, []))[1].append(shape)
            else:
                nurbs.append((kind, shape, fbx_node))
    for fbx_node, held in lines.values():
        _write_lines(exporter, held, fbx_node, fbx)
    if nurbs:
        _write_nurbs(exporter, nurbs, fbx)


def _write_lines(exporter, lines: list, node, fbx) -> None:
    """the curves of ``lines``, each of degree 1 holding no knots, as the
    lines of one unnamed FbxLine on ``node``, as Maya's exporter writes the
    curves of degree 1 of a transform (measured): their points one after
    another, each line ending at its last point"""
    shape  = fbx.FbxLine.Create(exporter.scene, "")
    points = np.concatenate([np.asarray(line.points, dtype=np.float64) for line in lines], axis=0).tolist()
    shape.InitControlPoints(len(points))
    shape.SetIndexArraySize(len(points))
    for index, point in enumerate(points):
        shape.SetControlPointAt(fbx.FbxVector4(*point, 1.0), index)
        shape.SetPointIndexAt(index, index)
    end = -1
    for line in lines:
        end += len(line.points)
        shape.AddEndPoint(end)
    node.SetNodeAttribute(shape)


def _fbx_knots(held, computed) -> list:
    """the knots fbx holds for a shape: ``held``, its ``knots``, or for None
    ``computed``, the ones it makes for itself, and one more at each end, a
    step past it as its neighbour's, as Maya's exporter writes them
    (measured)"""
    knots = np.asarray(computed if held is None else held, dtype=np.float64)
    return [2.0 * knots[0] - knots[1], *knots.tolist(), 2.0 * knots[-1] - knots[-2]]


def _fbx_points(points, weights=None) -> list:
    """x, y, z and its weight a point, flat, as fbx lists a NURBS shape's
    control points: the point as it stands, not multiplied by its weight,
    and 1 for ``weights`` None"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if weights is None:
        weights = np.ones(len(points))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([points, weights], axis=1).reshape(-1).tolist()


def _write_nurbs(exporter, held: list, fbx) -> None:
    """the NURBS curves and surfaces of ``held``, each ("curve" or
    "surface", the shape, its FbxNode), as an unnamed FbxNurbsCurve or
    FbxNurbsSurface on its node. The sdk's Python binding sets no knot, so
    the shapes go through a text: made in a scene of their own with the
    order, form and point count that give each array its length, written as
    ASCII to a temporary folder, their points and knots replaced in the text
    by the repr of each float, which reads back bit for bit, the text read
    into another scene, and its geometries moved to the exporter's scene,
    onto their nodes. A surface's points run u fastest"""
    from cgmath.formats._fbx_io import import_scene, media_folder

    manager = exporter.manager
    # an ASCII path: the sdk takes no other
    folder = media_folder()
    if folder is None:
        folder = tempfile.mkdtemp(prefix="cgmath_fbx_nurbs_")
    draft  = fbx.FbxScene.Create(manager, "")
    loaded = None
    try:
        arrays = {}
        for at, (kind, shape, _) in enumerate(held):
            tag = f"g{at}"
            if kind == "curve":
                forms    = fbx.FbxNurbsCurve.EType
                geometry = fbx.FbxNurbsCurve.Create(draft, tag)
                geometry.SetOrder(shape.degree + 1)
                geometry.InitControlPoints(len(shape.points), forms.ePeriodic if shape.periodic else forms.eOpen)
                arrays[tag] = {
                    "Points":     _fbx_points(shape.points, shape.weights),
                    "KnotVector": _fbx_knots(shape.knots, shape.kv),
                }
            else:
                forms    = fbx.FbxNurbsSurface.EType
                points   = np.asarray(shape.points, dtype=np.float64)
                geometry = fbx.FbxNurbsSurface.Create(draft, tag)
                geometry.SetOrder(shape.degree_u + 1, shape.degree_v + 1)
                geometry.InitControlPoints(
                    points.shape[0],
                    forms.ePeriodic if shape.periodic_u else forms.eOpen,
                    points.shape[1],
                    forms.ePeriodic if shape.periodic_v else forms.eOpen,
                )
                geometry.SetStep(1, 1)
                arrays[tag] = {
                    "Points": _fbx_points(
                        points.transpose(1, 0, 2),
                        None if shape.weights is None else np.asarray(shape.weights, dtype=np.float64).T,
                    ),
                    "KnotVectorU": _fbx_knots(shape.knots_u, shape.kv_u),
                    "KnotVectorV": _fbx_knots(shape.knots_v, shape.kv_v),
                }
            holder = fbx.FbxNode.Create(draft, f"holder_{tag}")
            holder.SetNodeAttribute(geometry)
            draft.GetRootNode().AddChild(holder)

        path   = os.path.join(folder, "draft.fbx")
        writer = fbx.FbxExporter.Create(manager, "")
        try:
            if not writer.Initialize(path, exporter._get_acsii_type(), manager.GetIOSettings()) or not writer.Export(draft):
                raise RuntimeError(f"save_fbx: no draft of the NURBS shapes: {writer.GetStatus().GetErrorString()}")
        finally:
            writer.Destroy()
        with open(path, "r", newline="", encoding="latin-1") as handle:
            text = handle.read()
        starts = [(match.start(), match.group(2)) for match in _NURBS_BLOCK.finditer(text)]
        if sorted(tag for _, tag in starts) != sorted(arrays):
            raise RuntimeError(f"save_fbx: the draft of the NURBS shapes holds {len(starts)} of {len(arrays)}")
        pieces, last = [], 0
        for index, (start, tag) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else text.find("\n\tModel:", start)
            end = len(text) if end < 0 else end

            def swap(match, values=arrays[tag]):
                row = values[match.group(1)]
                if int(match.group(2)) != len(row):
                    raise RuntimeError(
                        f"save_fbx: the sdk holds {match.group(2)} numbers as a NURBS shape's {match.group(1)}, where "
                        f"the shape holds {len(row)}"
                    )
                return f"{match.group(1)}: *{len(row)} {{\n\t\t\ta: {','.join(repr(float(x)) for x in row)}\n\t\t}}"

            pieces += [text[last:start], _NURBS_ARRAY.sub(swap, text[start:end])]
            last = end
        pieces.append(text[last:])
        path = os.path.join(folder, "shapes.fbx")
        with open(path, "w", newline="", encoding="latin-1") as handle:
            handle.write("".join(pieces))

        loaded = import_scene(manager, path, fbx)
        found  = {str(loaded.GetGeometry(i).GetName()): loaded.GetGeometry(i) for i in range(loaded.GetGeometryCount())}
        for at, (_, _, fbx_node) in enumerate(held):
            geometry = found[f"g{at}"]
            geometry.SetName("")
            geometry.GetNode(0).RemoveNodeAttribute(geometry)
            loaded.DisconnectSrcObject(geometry)
            exporter.scene.ConnectSrcObject(geometry)
            fbx_node.SetNodeAttribute(geometry)
    finally:
        draft.Destroy()
        if loaded is not None:
            loaded.Destroy()
        shutil.rmtree(folder, ignore_errors=True)


def _write_skin(exporter, mesh_node, shape, skin, hierarchy, nodes, at) -> None:
    """one FbxSkin on ``shape``, a cluster an influence, linked to its node
    at its world in the hierarchy, every cluster's transform the mesh's node
    at its world (column ``at`` of the hierarchy; identity for None, a root
    made for the mesh); the mesh node's channels locked, as Maya's FBXExport
    locks a skinned mesh's"""
    from cgmath.hierarchy.hierarchy import FBX as fbx

    scene    = exporter.scene
    worlds   = np.asarray(hierarchy.world_matrix, dtype=np.float64)
    bound    = _fbx_affine(np.eye(4) if at is None else worlds[at], fbx)
    place    = {node.name: column for column, node in enumerate(hierarchy)}
    weights  = np.asarray(skin.weights, dtype=np.float64)
    deformer = fbx.FbxSkin.Create(scene, skin.name or "")
    deformer.SetSkinningType(fbx.FbxSkin.EType.eLinear)
    for column, influence in enumerate(skin.influences):
        linked  = place[str(influence)]
        cluster = fbx.FbxCluster.Create(scene, str(influence))
        cluster.SetLink(nodes[linked])
        cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)
        rows = np.flatnonzero(weights[:, column])
        for vertex, weight in zip(rows.tolist(), weights[rows, column].tolist()):
            cluster.AddControlPointIndex(vertex, weight)
        cluster.SetTransformMatrix(bound)
        cluster.SetTransformLinkMatrix(_fbx_affine(worlds[linked], fbx))
        deformer.AddCluster(cluster)
    shape.AddDeformer(deformer)

    flags = fbx.FbxPropertyFlags.EFlags
    for prop in (mesh_node.LclTranslation, mesh_node.LclRotation, mesh_node.LclScaling):
        for member in (flags.eLockedMember0, flags.eLockedMember1, flags.eLockedMember2):
            prop.ModifyFlag(member, True)


def _write_bind_pose(exporter, nodes, hierarchy, roots) -> None:
    """one bind pose listing every node of the hierarchy at its world, and
    ``roots``, the nodes made for skinned meshes on no node, at identity"""
    from cgmath.hierarchy.hierarchy import FBX as fbx

    scene  = exporter.scene
    worlds = np.asarray(hierarchy.world_matrix, dtype=np.float64)
    pose   = fbx.FbxPose.Create(scene, "BindPose")
    pose.SetIsBindPose(True)
    for at, fbx_node in enumerate(nodes):
        pose.Add(fbx_node, _fbx_matrix(worlds[at], fbx))
    for fbx_node in roots:
        pose.Add(fbx_node, _fbx_matrix(np.eye(4), fbx))
    scene.AddPose(pose)


def _keys(curve, times, values, mode, fbx) -> None:
    """every key of ``curve`` replaced by one at each of ``times``"""
    key = fbx.FbxAnimCurveKey()
    key.SetInterpolation(mode)
    curve.KeyModifyBegin()
    try:
        curve.KeyClear()
        for when, value in zip(times, np.asarray(values, dtype=np.float64).tolist()):
            key.SetValue(value)
            curve.KeyAdd(when, key)
    finally:
        curve.KeyModifyEnd()


def _visibility_keys(fbx_node, layer, times, shown, fbx) -> None:
    """``fbx_node``'s Visibility keyed in ``layer`` at each of ``times``, 1
    where ``shown`` else 0, constant interpolation in its standard mode: each
    key holds to the next, as Maya keys a visibility"""
    from cgmath.formats._fbx_io import typed

    fbx_node.Visibility.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
    curve    = typed(fbx_node.Visibility.GetCurve(layer, True), fbx.FbxAnimCurve, fbx)
    constant = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant
    standard = fbx.FbxAnimCurveDef.EConstantMode.eConstantStandard
    curve.KeyModifyBegin()
    try:
        curve.KeyClear()
        for when, value in zip(times, np.asarray(shown, dtype=bool).tolist()):
            key = curve.KeyAdd(when)
            key = key[0] if isinstance(key, tuple) else key
            curve.KeySet(key, when, 1.0 if value else 0.0, constant)
            curve.KeySetConstantMode(key, standard)
    finally:
        curve.KeyModifyEnd()


def _write_takes(exporter, nodes, clips) -> None:
    """every take of ``clips``, one animation stack each (:func:`_write_take`)
    in the list's order, named ``take`` or by position, the file's time mode
    the takes' one frame rate, the active take's stack the file's current one"""
    from cgmath.hierarchy.hierarchy import FBX as fbx
    from cgmath.hierarchy.hierarchy import _blank_take, _fbx_rate_mode

    scene = exporter.scene
    mode  = getattr(fbx.FbxTime.EMode, _fbx_rate_mode(float(clips[0].fps)))
    scene.GetGlobalSettings().SetTimeMode(mode)
    active = clips.active
    held   = None
    for index, clip in enumerate(clips):
        stack = _write_take(exporter, nodes, clip, clip.take.strip() or _blank_take(index), mode)
        if clip is active:
            held = stack
    scene.SetCurrentAnimationStack(held)
    scene.ActiveAnimStackName.Set(fbx.FbxString(held.GetName()))


def _write_take(exporter, nodes, clip, name: str, mode):
    """one take: a stack named ``name`` spanning the clip's frames in time
    mode ``mode``, a BaseLayer, and the curves of every channel of every
    node; the stack"""
    from cgmath.formats._fbx_io import typed
    from cgmath.formats.fbx import _user_property_value
    from cgmath.hierarchy.hierarchy import FBX as fbx
    from cgmath.hierarchy.hierarchy import _user_channels

    scene  = exporter.scene
    frames = clip.start_frame + np.arange(clip.frame_count)
    times  = []
    for frame in frames.tolist():
        when = fbx.FbxTime()
        when.SetFrame(int(frame), mode)
        times.append(when)
    first, last = fbx.FbxTime(times[0].Get()), fbx.FbxTime(times[-1].Get())
    # the sdk writes no span (0, 0) and reads another in its place: one tick
    # past 0, which a read puts on frame 0
    if first.Get() == 0 and last.Get() == 0:
        last.Set(1)

    stack = fbx.FbxAnimStack.Create(scene, name)
    stack.SetLocalTimeSpan(fbx.FbxTimeSpan(first, last))
    stack.SetReferenceTimeSpan(fbx.FbxTimeSpan(first, last))
    layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    stack.AddMember(layer)

    linear = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    clip._settle()
    blocks = {
        "_translate": "LclTranslation",
        "_rotate":    "LclRotation",
        "_scale":     "LclScaling",
    }
    for channel, prop_name in blocks.items():
        block = np.asarray(clip._blocks[channel], dtype=np.float64)
        keyed = clip._keyed(channel)
        for column, fbx_node in enumerate(nodes):
            prop = getattr(fbx_node, prop_name)
            prop.GetCurveNode(layer, True)
            rows = block[:, column] if keyed[column] else block[:1, column]
            for axis, component in enumerate("XYZ"):
                curve = typed(prop.GetCurve(layer, component, True), fbx.FbxAnimCurve, fbx)
                _keys(curve, times[: len(rows)], rows[:, axis], linear, fbx)

    # a node's visibility, where the take changes it: one stepped key a
    # frame, 1 or 0, as Maya's own export keys it; a constant one is the
    # node's own value, written with the nodes
    shown = clip._shown()
    for column, fbx_node in enumerate(nodes):
        if not np.all(shown[:, column] == shown[0, column]):
            _visibility_keys(fbx_node, layer, times, shown[:, column], fbx)

    # each animatable user property: one curve, or three for a vector
    flag = fbx.FbxPropertyFlags.EFlags.eAnimatable
    for node, fbx_node in zip(clip, nodes):
        attrs = node.user_defined_attributes
        keys  = clip._user_keys.get(node.uuid, {})
        for attr, children in _user_channels(attrs).items():
            prop = fbx_node.FindProperty(attr)
            if not prop.IsValid():
                continue
            prop.ModifyFlag(flag, True)
            rows = keys.get(attr)
            if rows is None:
                rows = np.array([[float(attrs[x]["value"]) for x in ((attr,) if children is None else children)]])
            rows = np.asarray(rows, dtype=np.float64)
            prop.GetCurveNode(layer, True)
            if children is None:
                # an enum keys its field's place in the list, as its property holds it
                if attrs[attr].get("attributeType") == "enum":
                    rows = np.array([[float(_user_property_value(attrs[attr], int(x)))] for x in rows[:, 0]])
                curve = typed(prop.GetCurve(layer, True), fbx.FbxAnimCurve, fbx)
                _keys(curve, times[: len(rows)], rows[:, 0], linear, fbx)
                continue
            for axis, component in enumerate("XYZ"):
                curve = typed(prop.GetCurve(layer, component, True), fbx.FbxAnimCurve, fbx)
                _keys(curve, times[: len(rows)], rows[:, axis], linear, fbx)
    return stack
