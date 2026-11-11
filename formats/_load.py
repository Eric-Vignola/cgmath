"""``cgmath.formats.load_fbx``: a hierarchy or its takes, the meshes on its nodes

The r17-skeleton fork reads every node of a file as a joint (a skeleton
node), a locator (a null drawn as a cross, as Maya's own export writes one)
or a transform (any other: a group, or a node holding a shape, a mesh, a
camera or a light), at their own values, a bind pose left out, or the
file's takes animating them, every animation stack a ClipData
of one ClipList, the file's active stack the active take; each mesh as a
MeshData on its node, each camera and light as a CameraData or LightData on
its node, in Maya's attributes, the node turned back from fbx's camera and
light axes to Maya's,
the mesh's UV sets and its skin naming it by uuid. Everything comes back in
Maya's Y-up and cm, the axes and unit every cgmath value is in: a file
whose up axis is X or Z is turned as Maya's importer turns it, one whose up
axis is Y is left in its own axes whatever its front, as Maya leaves it, a
left handed one is refused, and the file's unit is converted.
"""

import contextlib
import os
import warnings
from collections import namedtuple

import numpy as np

from cgmath.formats._write import FORK, ORDER_PROPERTY, nonuniform, refuse_values

# a parent scale this close to 1 leaves an inherit type nothing to change
UNIT_SCALE = 1e-6

# a node whose own values place it this close to its bind matrix keeps them
BIND_AGREE = 1e-8

# a mesh's geometric transform this close to identity is none
MESH_IDENTITY = 1e-12

# a file's normals this close, a component, to the ones Maya derives from
# the points are those ones: the mesh stores none (Maya hands its normals
# out as floats)
NORMALS_DERIVED = 1e-5

# a bind matrix the solved channels place a node this far from: refused
BIND_SOLVED = 1e-6

# a skin's matrices and its bind pose this far apart, relative to the
# largest element, disagree; Maya's FBXExport writes the cluster's in
# float32
SKIN_AGREE = 1e-5

FbxContents = namedtuple(
    "FbxContents", ("hierarchy", "clip", "meshes", "skins", "uvs", "cameras", "lights", "materials", "morphs")
)


def load_fbx(
    filename,
    *,
    takes:     bool = True,
    meshes:    bool = True,
    uvs:       bool = True,
    skins:     bool = True,
    materials: bool = True,
    textures:  bool = True,
    morphs:    bool = True,
    cameras:   bool = True,
    lights:    bool = True,
    verbose:   bool = True,
) -> FbxContents:
    """``(hierarchy, clip, meshes, skins, uvs, cameras, lights, materials,
    morphs)``, None where the file holds none, or where the read was told to
    leave a kind out (the flags below)

    ``hierarchy`` every node of the file at its own values, a mesh's node
    among them; None for a file holding a take, whose nodes ``clip`` holds at
    the takes' frames: a ClipList, one ClipData per animation stack in file
    order, each over every node, named as the stack is (a blank name "Take
    001", "Take 002" ... by position, a name two stacks share told apart as
    ``name_1``, with one warning), the stack the file marks active
    (``ActiveAnimStackName``) its active take, else the first. With a skin,
    ``hierarchy`` is the bind pose and ``clip`` the takes, if any. ``meshes``
    a MeshList of the file's meshes in
    their nodes' order, each ``parent_uuid`` the uuid of its node in
    ``hierarchy`` or ``clip``; ``skins`` a SkinList, one a skinned mesh, in
    that order; ``uvs`` a UVList of every mesh's UV sets, mesh by mesh in the
    file's order; each skin and set names its mesh by ``mesh_uuid``. A mesh
    is named as its FbxMesh is, None for one unnamed (Maya's FBXExport
    leaves every FbxMesh unnamed). A set of no UVs, as Maya's FBXExport
    writes an empty one, is no set; ValueError for a set leaving faces
    without UVs. A node holding a shape that is no mesh is a transform, the
    shape left out; a joint is a joint whatever else it holds (Maya's FBX
    holds no mesh on one).

    A mesh stores the least normals that shade it as the file does. The
    file's normals are held, a face corner at a time, against the ones Maya
    derives from the points (:meth:`MeshData.get_face_vertex_normals`):
    with every edge soft, then broken at the edges the file's own normals
    part, an edge hard where its two faces hold different normals at one of
    its ends. Within 1e-5 a component the mesh stores none: ``normals`` and
    ``normal_indices`` None, and ``hard_edges`` None when every edge is
    soft, else those edges, a pair of vertices each. Any other normals,
    authored ones (Maya's locked normals), are stored as the file holds
    them, each corner naming its own, with ``hard_edges`` None. Maya's
    importer, for its part, locks every normal it reads and tells the hard
    edges from them (measured).

    ``cameras`` a CameraList and ``lights`` a LightList, in their nodes'
    order, each ``parent_uuid`` its node's, each in Maya's attributes as
    Maya's own import would hold them (measured): a camera's focal length
    as the file holds it, or the one its field of view gives on its film
    back where the file holds the view (``ApertureMode``), the film back and
    offsets in inches, the gate fit as the film fit (fill for fbx's none and
    stretch), the clip planes and, on an orthographic camera, the
    orthographic width (the ortho zoom times 30) in cm, a perspective
    camera's being Maya's default 10; a light's kind, colour, intensity (the file's over
    100), decay rate, and a spot's cone (the inner angle) and penumbra (half
    the outer angle's excess). Fbx holds no camera scale and no dropoff: 1
    and 0. A node holding a camera or a directed light is turned from fbx's
    camera and light axes to Maya's (fbx's camera looks down its node's +X,
    its light shines down -Y, Maya's both down -Z): a node Maya wrote, whose
    post rotation is the turn, keeps its rotate to the bit; any other's
    rotate is solved for the turn, frame by frame through euler_filter on a
    clip. A node under one raises ValueError. An ambient, area or volume
    light is no LightData: one warning names them, the node a transform. A
    camera or light property a take keys is read at the active take's
    first frame, with one warning naming shape and properties.

    ``materials`` a MaterialList of the materials the meshes' material
    elements assign faces to, mesh by mesh in the meshes' order, each
    naming its mesh by ``mesh_uuid``, its ``faces`` the mesh's faces it
    covers, None for the whole mesh (an all-same element, or a by-polygon
    one giving it every face), as Maya's own import would hold them
    (measured on Maya's export): an FbxSurfacePhong a phong, any other
    FbxSurfaceLambert a lambert, its colour the diffuse colour and its
    diffuse the diffuse factor, its ambient colour, incandescence,
    transparency and specular colour the ambient, emissive, transparent and
    specular colours times their factors, a phong's cosine power the
    shininess and its reflectivity the reflection factor; a material of
    another shading model (Maya writes a standardSurface as one, bare) a
    default lambert of its name, with one warning naming them, but Maya's
    own default material (``MAYA_DEFAULT_MATERIALS``), written bare, is no
    material where it covers the whole mesh with no map: the mesh comes
    back with none, in silence, as a mesh made in Maya by hand starts with
    none of its own (Maya's importer, for its part, makes a fresh
    standardSurface2 of such a file's default material, and leaves a mesh
    holding no material on no shading group; measured). The file
    texture feeding the diffuse colour, through one layered texture if so,
    gives the material its texture: the pixels decoded through PIL, 8 bit
    unless the file holds 16, rows top down, as (H, W, C); ``texture_file``
    the file's name; the UV set it reads (None for the default) and its wrap
    modes (repeat from a file Maya wrote, whatever its place2dTexture held:
    Maya's export writes no other, measured). An embedded texture is
    extracted to a temporary folder while the
    file is read; a linked one is read at its path, or by its relative name
    beside the file. A texture whose file is not found, or with no PIL to
    decode it, is None, its file name kept, with one warning; a texture on
    any other slot (specular, bump, emissive ...) or that is no file texture
    is left out with one warning naming slot and material. ValueError for a
    material element mapping other than all-same or by-polygon, or leaving
    faces without a material.

    ``morphs`` a MorphList of the meshes' blend shape targets, mesh by mesh
    in the meshes' order, deformer by deformer, channel by channel, shape by
    shape as the file orders them, each a MorphData naming its mesh by
    ``mesh_uuid``, as Maya's own import would hold them (measured on Maya's
    export): ``deformer`` the FbxBlendShape's name (Maya's blendShape node),
    ``name`` the channel's name less the deformer's name and a dot, which
    Maya's export prefixes and its import strips, ``weight`` the channel's
    DeformPercent over 100, ``at`` the shape's full weight over 100 (1 for
    a full target, below 1 for an in-between: a channel's shapes are the
    targets of one name), ``offsets`` the shape's points less the mesh's at
    the points that differ (a file holds every point of a target, absolute;
    its normals are not read). A channel the active take keys is read at
    the take's first frame, with one warning naming its targets. ValueError
    for a target of another point count than its mesh, or one reached at a
    weight that is not above 0.

    The values are in Maya's Y-up and cm, as every cgmath value is. A file
    whose up axis is X or Z is turned the way Maya's importer turns it
    (measured): its roots take the turn, a joint root in its joint orient and
    translate, any other root in its rotate, solved again frame by frame
    through euler_filter, and translate, and a bind pose with them; a node
    under a root is relative to it and keeps its values, and so does a mesh,
    in its node's space. A file whose up axis is Y, up or down, whatever its
    front, is left in its own axes, as Maya's importer leaves it. A left
    handed file raises ValueError: no rotation turns it into Maya's axes.
    Lengths are multiplied by the centimeters one of the file's units holds,
    translations, points and bind matrices alike (a file written in metres
    loads a hundred times its numbers); a scale a unit converting export
    left on a root stays as the file holds it. Maya's importer holds that
    unit as a scale on the file's roots and its compensating joints instead,
    the numbers under them as the file has them (measured): the same scene,
    every position the same, every world's 3x3 the unit times load_fbx's.

    A transform's rotate pivot, scale pivot, rotate offset and scale offset
    (Maya's rotatePivot, scalePivot, rotatePivotTranslate and
    scalePivotTranslate) are folded into its translate, as the matrix Maya
    composes holds them: its 3x3 is untouched, its position moves, on every
    frame of a take where its translate, rotate or scale is keyed (measured
    against Maya's own evaluation). ValueError for pivots on a joint, which
    sit before its segment scale compensate, or keyed pivots. A take's
    layers holding no curve are passed over (an Unreal export carries empty
    ones); curves on two layers, or on a muted one, raise
    NotImplementedError. A take covers its stored span widened to its
    keys: an Unreal export stores the span a frame short of its last key.

    The flags leave a kind unread, its field None, so a caller after one
    thing pays for that alone: ``takes`` the animation stacks (the nodes
    then come back as ``hierarchy`` at their own values), ``meshes`` the
    meshes and with them everything on a mesh, ``uvs``, ``skins`` (without
    skins ``hierarchy`` is the nodes' own values, not the bind pose),
    ``materials``, ``morphs``, ``cameras`` and ``lights``; ``textures``
    False keeps each material's ``texture_file``, UV set and wraps but
    decodes no pixels, ``texture`` None. The hierarchy is always read. A
    texture file is decoded once a read, however many materials name it,
    which all share the one array. ``verbose`` False silences the read's
    warnings.
    """
    parts = {
        "takes":     takes,
        "meshes":    meshes,
        "uvs":       uvs,
        "skins":     skins,
        "materials": materials,
        "textures":  textures,
        "morphs":    morphs,
        "cameras":   cameras,
        "lights":    lights,
    }
    with _quiet(verbose):
        return _load(filename, parts)


@contextlib.contextmanager
def _quiet(verbose: bool):
    """the read's warnings, silenced unless ``verbose``"""
    with warnings.catch_warnings():
        if not verbose:
            warnings.simplefilter("ignore")
        yield


def _load(filename, parts: dict) -> FbxContents:
    """:func:`load_fbx`'s read of the kinds ``parts`` turns on"""
    read = _open(filename, parts)
    hierarchy, order = _build(read)
    # Maya's FBXExport writes a take whether anything moves or not: a take
    # over no node is no clip
    clip = None if read["takes"] is None or not len(hierarchy) else _build_clips(read, hierarchy, order)
    # the shapes in their nodes' order, the hierarchy's
    rank = {at: column for column, at in enumerate(order)}
    for kind in ("meshes", "cameras", "lights"):
        read[kind].sort(key=lambda held: rank[held["node"]])
    meshes    = _build_meshes(read) if parts["meshes"] else None
    uvs       = _build_uvs(read, meshes) if parts["uvs"] else None
    cameras   = _build_cameras(read) if parts["cameras"] else None
    lights    = _build_lights(read) if parts["lights"] else None
    materials = _build_materials(read, meshes) if parts["materials"] else None
    morphs    = _build_morphs(read, meshes) if parts["morphs"] else None
    if parts["skins"] and any(held["skin"] is not None for held in read["meshes"]):
        bind, order = _build(read)
        _settle_bind(read, bind, order)
        return FbxContents(bind, clip, meshes, _build_skins(read, meshes), uvs, cameras, lights, materials, morphs)
    if clip is not None:
        return FbxContents(None, clip, meshes, None, uvs, cameras, lights, materials, morphs)
    return FbxContents(hierarchy if len(hierarchy) else None, None, meshes, None, uvs, cameras, lights, materials, morphs)


# the reads of one kind: the nodes alone, and the nodes with their takes
_PARTS_NODES = {kind: False for kind in ("takes", "meshes", "uvs", "skins", "materials", "textures", "morphs", "cameras", "lights")}
_PARTS_TAKES = dict(_PARTS_NODES, takes=True)


def load_nodes(filename, verbose: bool = True):
    """the HierarchyData of an fbx file's nodes at their own values, whether
    it holds a take or not (the take left unread), a mesh's node among them,
    its shape left out; ``verbose`` False silences the read's warnings"""
    with _quiet(verbose):
        return _build(_open(filename, _PARTS_NODES))[0]


def load_takes(filename, verbose: bool = True):
    """the ClipList of an fbx file's takes over every node, a mesh's among
    them, its shape and skin left out, the file's active take active; None
    for a file holding no take, or no node; ``verbose`` False silences the
    read's warnings"""
    with _quiet(verbose):
        read = _open(filename, _PARTS_TAKES)
        if read["takes"] is None or not read["nodes"]:
            return None
        hierarchy, order = _build(read)
        return _build_clips(read, hierarchy, order)


def load_only(filename, *, verbose: bool = True, **kinds) -> FbxContents:
    """:func:`load_fbx` with every kind off but the ``kinds`` named True,
    the class routes' read: ``load_only(path, meshes=True)`` reads the
    nodes and the meshes alone"""
    return load_fbx(filename, verbose=verbose, **dict(_PARTS_NODES, **kinds))


def _open(filename, parts: dict) -> dict:
    # imported here: the classes' modules import this one
    from cgmath.formats._fbx_io import open_fbx
    from cgmath.hierarchy.hierarchy import FBX

    if FBX is None:
        raise ImportError("fbx module not found")
    filename = os.path.expanduser(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    # with the meshes come their materials, whose embedded textures are
    # extracted to a temporary folder for the read, when their pixels are
    # read at all
    media = bool(parts["meshes"] and parts["materials"] and parts["textures"])
    with open_fbx(filename, FBX, media=media) as scene:
        return _read(scene, FBX, parts, filename)


def _walk(scene) -> list:
    """[(node, parent position or -1)] of every node under the scene root,
    depth first, each parent ahead of its children, in the file's order"""
    found   = []
    root    = scene.GetRootNode()
    pending = [(root.GetChild(i), -1) for i in reversed(range(root.GetChildCount()))]
    while pending:
        node, parent = pending.pop()
        found.append((node, parent))
        here = len(found) - 1
        for i in reversed(range(node.GetChildCount())):
            pending.append((node.GetChild(i), here))
    return found


def _amatrix(matrix) -> np.ndarray:
    """an FbxAMatrix or FbxMatrix as its (4, 4) row-vector layout"""
    return np.array([[matrix.Get(row, column) for column in range(4)] for row in range(4)], dtype=np.float64)


def _read_mesh(node, at: int, filename: str, fbx, parts: dict, cache: dict) -> dict:
    """the mesh on the node at place ``at`` of the file's nodes: its name,
    points, faces, normals, one a face corner, and its UV sets, materials
    and targets where ``parts`` turns them on (``cache`` the read's decoded
    textures by file); ValueError for a node holding a geometric transform,
    which no field holds, or a mesh holding holes"""
    from cgmath.formats._fbx_io import typed
    from cgmath.geometry.mesh import (
        _fbx_control_points,
        _fbx_direct_array_to_ndarray,
        _fbx_face_varying_indices,
        _fbx_polygon_streams,
    )

    shape  = typed(node.GetNodeAttribute(), fbx.FbxMesh, fbx)
    name   = str(shape.GetName()) or None
    label  = f"mesh {name!r}" if name else f"the mesh on node {str(node.GetName())!r}"
    source = fbx.FbxNode.EPivotSet.eSourcePivot
    offset = fbx.FbxAMatrix(node.GetGeometricTranslation(source), node.GetGeometricRotation(source), node.GetGeometricScaling(source))
    offset = _amatrix(offset)
    if np.abs(offset - np.eye(4)).max() > MESH_IDENTITY:
        raise ValueError(
            f"load_fbx: {label} channel 'geometric transform' is {np.round(offset, 12).tolist()} on node "
            f"{str(node.GetName())!r}, which no field holds; {FORK} reads a mesh at its node"
        )
    holes = [i for i in range(shape.GetPolygonCount()) if shape.GetPolyHoleInfo(i)]
    if holes:
        raise ValueError(f"load_fbx: {label} holds {len(holes)} polygon holes, which {FORK} does not read")
    indices, counts = _fbx_polygon_streams(shape)
    normals, ids = None, None
    if shape.GetElementNormalCount():
        element = shape.GetElementNormal(0)
        normals = _fbx_direct_array_to_ndarray(element.GetDirectArray(), dim=3)
        ids     = _fbx_face_varying_indices(shape, element, counts)
    points = _fbx_control_points(shape)
    colors, color_ids = _read_colors(shape, counts, label, fbx)
    return {
        "uvs":            _read_uvs(label, shape, counts, fbx) if parts["uvs"] else [],
        "materials":      _read_materials(node, shape, counts, label, filename, fbx, parts["textures"], cache) if parts["materials"] else [],
        "morphs":         _read_morphs(shape, points, label, fbx) if parts["morphs"] else [],
        "name":           name,
        "label":          label,
        "node":           at,
        "points":         points,
        "counts":         counts,
        "indices":        indices,
        "normals":        normals,
        "normal_indices": ids,
        "colors":         colors,
        "color_indices":  color_ids,
        "shape":          shape,
    }


def _read_colors(shape, counts, label: str, fbx) -> tuple:
    """``(colors, color_indices)`` of the mesh's first vertex colour element,
    eNone elements padding a layer passed over: its RGBA doubles, one a
    control point (``color_indices`` None; Maya reads such an element) or
    one a face corner through the indices
    :func:`_fbx_face_varying_indices` gives (Maya writes a colour set so,
    a shared one too); a mesh holding several sets gives its first with a
    warning naming the others, ``MeshData`` holding one; ``(None, None)``
    for none. ValueError for another mapping"""
    from cgmath.geometry.mesh import _fbx_face_varying_indices

    mapping   = fbx.FbxLayerElement.EMappingMode
    reference = fbx.FbxLayerElement.EReferenceMode
    held      = [shape.GetElementVertexColor(k) for k in range(shape.GetElementVertexColorCount())]
    mapped    = [e for e in held if e.GetMappingMode() != mapping.eNone]
    if not mapped:
        return None, None
    element = mapped[0]
    if len(mapped) > 1:
        warnings.warn(
            f"load_fbx: {label} holds the colour sets {[str(e.GetName()) for e in mapped[1:]]} beside "
            f"{str(element.GetName())!r}; {FORK} reads the first",
            stacklevel=2,
        )
    mode = element.GetMappingMode()
    if mode not in (mapping.eByControlPoint, mapping.eByPolygonVertex):
        raise ValueError(
            f"load_fbx: {label} channel 'colors' is mapped {str(mode).rsplit('.', 1)[-1]}, which {FORK} does not read; "
            f"a colour a vertex or a face corner"
        )
    direct = element.GetDirectArray()
    colors = np.array(
        [[c.mRed, c.mGreen, c.mBlue, c.mAlpha] for c in (direct.GetAt(i) for i in range(direct.GetCount()))], dtype=np.float64
    ).reshape(-1, 4)
    if mode == mapping.eByPolygonVertex:
        return colors, _fbx_face_varying_indices(shape, element, counts)
    if element.GetReferenceMode() == reference.eDirect:
        return colors, None
    index = element.GetIndexArray()
    return colors[np.array([index.GetAt(i) for i in range(index.GetCount())], dtype=np.int64)], None


# fbx's DeformPercent and full weights are Maya's weights times this
WEIGHT_UNIT = 100.0


def _read_morphs(shape, points, label: str, fbx) -> list:
    """the blend shape targets on the mesh (``label``), deformer by deformer,
    channel by channel, shape by shape in the file's order: each its name
    (the channel's, less the deformer's name and a dot where Maya's export
    prefixes it), its deformer's name, its weight (the channel's
    DeformPercent over ``WEIGHT_UNIT``), the weight it is reached at (its
    full weight over ``WEIGHT_UNIT``), and its offsets from ``points``, the
    mesh's, at the points it moves; the channel kept for :func:`_morph_keys`.
    ValueError for a target of another point count, or one reached at a
    weight that is not above 0"""
    from cgmath.formats._fbx_io import typed
    from cgmath.geometry.mesh import _fbx_control_points

    kind  = fbx.FbxDeformer.EDeformerType.eBlendShape
    found = []
    for d in range(shape.GetDeformerCount(kind)):
        deformer = typed(shape.GetDeformer(d, kind), fbx.FbxBlendShape, fbx)
        dname    = str(deformer.GetName()) or None
        for c in range(deformer.GetBlendShapeChannelCount()):
            channel = deformer.GetBlendShapeChannel(c)
            cname   = str(channel.GetName())
            if dname and cname.startswith(dname + "."):
                cname = cname[len(dname) + 1:]
            weights = [float(w) for w in channel.GetTargetShapeFullWeights()]
            weight  = float(channel.DeformPercent.Get()) / WEIGHT_UNIT
            for s in range(channel.GetTargetShapeCount()):
                target = channel.GetTargetShape(s)
                name   = cname or str(target.GetName()) or f"target{c + 1}"
                count  = target.GetControlPointsCount()
                if count != len(points):
                    raise ValueError(
                        f"load_fbx: {label} blend shape target {name!r} holds {count} points, the mesh {len(points)}; "
                        f"{FORK} reads a target over every point of its mesh"
                    )
                at = weights[s] / WEIGHT_UNIT if s < len(weights) else 1.0
                if not at > 0.0:
                    raise ValueError(
                        f"load_fbx: {label} blend shape target {name!r} is reached at weight {at}, which no target is; "
                        f"{FORK} reads a target reached above 0"
                    )
                diff  = np.asarray(_fbx_control_points(target), dtype=np.float64) - points
                moved = np.flatnonzero(np.any(diff != 0.0, axis=1))
                found.append({
                    "name":     name,
                    "label":    f"blend shape target {name!r} of {label}",
                    "deformer": dname,
                    "channel":  channel,
                    "weight":   weight,
                    "at":       at,
                    "offsets":  diff[moved],
                    "indices":  moved.astype(np.int32),
                })
    return found


def _morph_keys(read: dict, fbx) -> None:
    """a blend shape channel's DeformPercent the active take keys: every
    target of the channel takes the take's first frame's value, and one
    warning names them; the fork reads a weight static"""
    import warnings

    from cgmath.formats._fbx_io import typed

    takes = read["takes"]
    if not takes:
        return
    take  = takes[read["active"]]
    layer = take["layer"]
    if layer is None:
        return
    when  = fbx.FbxTime(int(take["ticks"][0]))
    keyed = []
    for held in read["meshes"]:
        for morph in held["morphs"]:
            prop = morph["channel"].DeformPercent
            if prop.GetCurveNode(layer, False) is None:
                continue
            curve = prop.GetCurve(layer, False)
            if curve is None:
                continue
            value           = typed(curve, fbx.FbxAnimCurve, fbx).Evaluate(when)
            morph["weight"] = float(value[0] if isinstance(value, tuple) else value) / WEIGHT_UNIT
            if morph["at"] == 1.0:
                keyed.append(morph["label"])
    if keyed:
        warnings.warn(
            f"load_fbx: {', '.join(keyed)}: the weight is keyed in take {take['name']!r}; {FORK} reads a blend shape "
            f"weight static, at the take's first frame",
            stacklevel=2,
        )


# the material property a texture feeds that the fork reads: the colour map
COLOUR_SLOT = "DiffuseColor"

# the slots a bump2d's map lands on, by how Maya reads it (measured): tangent
# space normals as NormalMap, a bump map (and object space normals) as Bump
NORMAL_SLOTS = {"NormalMap": "tangent", "Bump": "bump"}

# Maya's default material, the surface shader of initialShadingGroup, which
# its export writes as a bare FbxSurfaceMaterial of that name (measured on
# Maya 2025): on a whole mesh with no map it is no material of ours. Maya's
# own import makes a fresh standardSurface2 in a new shading group of it,
# and leaves a mesh holding no material on no shading group at all; a mesh
# made in Maya by hand lands on initialShadingGroup (measured)
MAYA_DEFAULT_MATERIALS = ("standardSurface1",)

# fbx's texture UVSet meaning the mesh's first set
DEFAULT_UV_SET = "default"


def _material_faces(shape, count: int, faces: int, label: str, fbx) -> list:
    """[(material index, faces or None)] of the mesh's material element, in
    material order: an all-same element, or none, gives its one material
    the whole mesh; a by-polygon one gives each material the polygons that
    name it, every face as None, no face left out; an empty eNone element
    padding a layer under the mapped one (the sdk's ``CreateElementMaterial``
    on a mesh whose UV sets came first) is passed over. ValueError for
    another mapping, or polygons naming no material of the node's ``count``"""
    mapping = fbx.FbxLayerElement.EMappingMode
    found   = [shape.GetElementMaterial(k) for k in range(shape.GetElementMaterialCount())]
    if not found:
        return [(0, None)]
    element = next((e for e in found if e.GetMappingMode() != mapping.eNone), found[0])
    held    = element.GetIndexArray()
    indices = np.array([int(held.GetAt(k)) for k in range(held.GetCount())], dtype=np.int64)
    mode    = element.GetMappingMode()
    if mode == mapping.eAllSame:
        index = int(indices[0]) if indices.size else 0
        if not 0 <= index < count:
            raise ValueError(f"load_fbx: {label} names material {index} of its node's {count}")
        return [(index, None)]
    if mode != mapping.eByPolygon:
        raise ValueError(
            f"load_fbx: {label} channel 'materials' is mapped {str(mode).rsplit('.', 1)[-1]}, which {FORK} does not read; "
            f"a material covers the whole mesh or faces of it"
        )
    if indices.size != faces:
        raise ValueError(f"load_fbx: {label} channel 'materials' names a material for {indices.size} faces of {faces}")
    off = np.flatnonzero((indices < 0) | (indices >= count))
    if off.size:
        left = int(np.count_nonzero(indices < 0))
        why  = f"leaves {left} faces without a material" if left else f"names material {int(indices[off[0]])} of its node's {count}"
        raise ValueError(f"load_fbx: {label} channel 'materials' {why}; {FORK} reads a mesh covered whole")
    found = []
    for index in range(count):
        covered = np.flatnonzero(indices == index)
        if covered.size == faces:
            found.append((index, None))
        elif covered.size:
            found.append((index, covered.astype(np.int32)))
    return found


def _pixels(image) -> np.ndarray:
    """a PIL image as (H, W, C): L, LA, RGB, RGBA, a 16 bit grey (I;16) and a
    float (F) image as they are, any other mode as RGB or RGBA"""
    mode = image.mode
    if mode not in ("L", "LA", "RGB", "RGBA", "I;16", "F"):
        image = image.convert("RGBA" if "A" in mode or "transparency" in image.info else "RGB")
    pixels = np.asarray(image)
    if pixels.dtype == np.int32:
        pixels = pixels.astype(np.uint16)  # PIL hands a 16 bit grey as int32 on some builds
    return pixels[:, :, None] if pixels.ndim == 2 else pixels


def _read_texture(texture, filename: str, fbx, pixels: bool = True, cache=None) -> dict:
    """a file texture, as the fields of a material: its pixels through PIL,
    its file name, UV set and wrap modes; ``pixels`` None with ``missing``
    the path when the file is not found, at its own path, by its relative
    name beside ``filename`` (the .fbx) or by its own name there, and
    ``no_pil`` True when PIL is not installed. ``pixels`` False names the
    file and decodes nothing; ``cache`` holds the read's decoded files by
    path, so a file many materials name is decoded once and shared"""
    from cgmath.formats._write import _pil

    path     = str(texture.GetFileName())
    relative = str(texture.GetRelativeFileName())
    wraps    = {_enum_int(_fbx_enum_of(fbx.FbxTexture, "EWrapMode", x)): mode for x, mode in (("eRepeat", "repeat"), ("eClamp", "clamp"))}
    uv_set   = str(texture.UVSet.Get())
    held = {
        "file":    os.path.basename(path or relative) or None,
        "uv_set":  None if uv_set in ("", DEFAULT_UV_SET) else uv_set,
        "wrap_u":  wraps.get(_enum_int(texture.WrapModeU.Get()), "repeat"),
        "wrap_v":  wraps.get(_enum_int(texture.WrapModeV.Get()), "repeat"),
        "pixels":  None,
        "missing": None,
        "no_pil":  False,
    }
    if not pixels:
        return held
    beside = os.path.dirname(os.path.abspath(filename))
    tried  = [x for x in (path, os.path.join(beside, relative) if relative else "", os.path.join(beside, os.path.basename(path)) if path else "") if x]
    found  = next((x for x in tried if os.path.isfile(x)), None)
    if found is None:
        held["missing"] = path or relative
        return held
    Image = _pil()
    if Image is None:
        held["no_pil"] = True
        return held
    key    = os.path.normcase(os.path.abspath(found))
    shared = None if cache is None else cache.get(key)
    if shared is None:
        with Image.open(found) as image:
            shared = np.array(_pixels(image))
        if cache is not None:
            cache[key] = shared
    held["pixels"] = shared
    return held


def _fbx_enum_of(owner, scope: str, name: str):
    from cgmath.hierarchy.hierarchy import _fbx_enum

    return _fbx_enum(owner, scope, name)


def _file_texture(prop, fbx):
    """the first FbxFileTexture feeding ``prop``, directly or inside a
    layered texture, or None; ``others`` True in the second slot when a
    texture of another kind feeds it alone"""
    from cgmath.formats._fbx_io import typed

    other = False
    for i in range(prop.GetSrcObjectCount()):
        obj = prop.GetSrcObject(i)
        cid = obj.GetClassId()
        if cid.Is(fbx.FbxFileTexture.ClassId):
            return typed(obj, fbx.FbxFileTexture, fbx), False
        if cid.Is(fbx.FbxLayeredTexture.ClassId):
            for j in range(obj.GetSrcObjectCount()):
                sub = obj.GetSrcObject(j)
                if sub.GetClassId().Is(fbx.FbxFileTexture.ClassId):
                    return typed(sub, fbx.FbxFileTexture, fbx), False
            other = True
        elif cid.Is(fbx.FbxTexture.ClassId):
            other = True
    return None, other


def _read_materials(node, shape, counts, label: str, filename: str, fbx, pixels: bool = True, cache=None) -> list:
    """the materials the mesh's material element assigns faces to, in the
    node's order: each its name, kind, Maya's attributes off the file's
    properties (a colour times its factor, the diffuse colour and factor
    apart, which are Maya's color and diffuse), its faces, its colour map
    and the bump2d's map on its NormalMap or Bump slot through
    :func:`_read_texture` (``pixels`` and ``cache`` as it takes them), the
    bump map of a material holding both, as Maya's importer takes it
    (measured), with its kind and the BumpFactor as ``bump_depth``; a
    material of another shading model a default lambert, flagged ``bare``,
    but Maya's default material, bare on the whole mesh with no map, left
    out; textures on other slots, or that are no file texture, listed in
    ``left`` by slot"""
    from cgmath.formats._fbx_io import typed

    count = node.GetMaterialCount()
    if not count:
        return []
    found = []
    for index, faces in _material_faces(shape, count, len(counts), label, fbx):
        material = typed(node.GetMaterial(index), fbx.FbxSurfaceMaterial, fbx)
        name     = str(material.GetName()) or None
        cid      = material.GetClassId()
        phong    = cid.Is(fbx.FbxSurfacePhong.ClassId)
        lambert  = phong or cid.Is(fbx.FbxSurfaceLambert.ClassId)
        fields   = {"kind": "phong" if phong else "lambert"}
        if lambert:
            held = typed(material, fbx.FbxSurfacePhong if phong else fbx.FbxSurfaceLambert, fbx)

            def colour(prop, factor=None):
                values = np.array(tuple(prop.Get())[:3], dtype=np.float64)
                return values if factor is None else values * float(factor.Get())

            fields["color"]         = colour(held.Diffuse)
            fields["diffuse"]       = float(held.DiffuseFactor.Get())
            fields["ambient_color"] = colour(held.Ambient,          held.AmbientFactor)
            fields["incandescence"] = colour(held.Emissive,         held.EmissiveFactor)
            fields["transparency"]  = colour(held.TransparentColor, held.TransparencyFactor)
            if phong:
                fields["specular_color"] = colour(held.Specular, held.SpecularFactor)
                fields["cosine_power"]   = float(held.Shininess.Get())
                fields["reflectivity"]   = float(held.ReflectionFactor.Get())
        # the textures: the colour map and the bump2d's map read, any other slot's noted
        texture = None
        normals = {}
        left    = []
        prop    = material.GetFirstProperty()
        while prop.IsValid():
            if prop.GetSrcObjectCount():
                file, other = _file_texture(prop, fbx)
                slot = str(prop.GetName())
                if slot == COLOUR_SLOT and file is not None:
                    texture = _read_texture(file, filename, fbx, pixels, cache)
                elif slot in NORMAL_SLOTS and file is not None:
                    normals[slot] = file
                elif file is not None or other:
                    left.append(slot)
            prop = material.GetNextProperty(prop)
        # Maya's importer keeps the bump map of a material holding both (measured)
        normal = kind = None
        if normals:
            slot   = "Bump" if "Bump" in normals else "NormalMap"
            normal = _read_texture(normals[slot], filename, fbx, pixels, cache)
            kind   = NORMAL_SLOTS[slot]
            left.extend(other_slot for other_slot in normals if other_slot != slot)
        # Maya's default material on the whole mesh, with no map, is the mesh
        # with no material of its own (MAYA_DEFAULT_MATERIALS)
        if not lambert and name in MAYA_DEFAULT_MATERIALS and faces is None and texture is None and normal is None:
            continue
        found.append({
            "name":        name,
            "label":       f"material {name!r}" if name else f"the unnamed material on {label}",
            "model":       str(material.ShadingModel.Get()),
            "bare":        not lambert,
            "fields":      fields,
            "faces":       faces,
            "texture":     texture,
            "normal":      normal,
            "normal_kind": kind,
            "bump_depth":  float(held.BumpFactor.Get()) if lambert else 1.0,
            "left":        left,
            "id":          int(material.GetUniqueID()),  # the one FbxSurfaceMaterial, on every node it is on
        })
    return found


def _read_uvs(label, shape, counts, fbx) -> list:
    """the mesh's UV sets, (name, points, indices) each, in the file's
    order, one index a face corner; a set of no UVs left out, ValueError for
    one leaving faces without UVs"""
    from cgmath.geometry.mesh import _fbx_direct_array_to_ndarray, _fbx_face_varying_indices

    mapping = fbx.FbxLayerElement.EMappingMode
    refs    = fbx.FbxLayerElement.EReferenceMode
    total   = int(np.sum(counts))
    found   = []
    for place in range(shape.GetElementUVCount()):
        element = shape.GetElementUV(place)
        if element is None:
            continue
        set_name = str(element.GetName()) or f"map{place + 1}"
        points   = _fbx_direct_array_to_ndarray(element.GetDirectArray(), dim=2)
        if not len(points):
            continue
        held = element.GetIndexArray().GetCount()
        if (
            element.GetMappingMode() == mapping.eByPolygonVertex
            and element.GetReferenceMode() == refs.eIndexToDirect
            and held != total
        ):
            raise ValueError(
                f"load_fbx: {label} UV set {set_name!r} maps {held} face corners of {total}; {FORK} reads a set "
                f"mapping every face"
            )
        indices = _fbx_face_varying_indices(shape, element, counts)
        if indices.size and (indices.min() < 0 or indices.max() >= len(points)):
            unmapped = int(np.count_nonzero(indices < 0))
            why      = f"leaves {unmapped} face corners without UVs" if unmapped else "names UVs it does not hold"
            raise ValueError(f"load_fbx: {label} UV set {set_name!r} {why}; {FORK} reads a set mapping every face")
        found.append((set_name, points, indices))
    return found


def _read_skin(scene, node, at: int, label: str, shape, found, fbx) -> dict:
    """the one skin on the mesh (``label``) of the node at place ``at``: its
    clusters' nodes (places in ``found``), weights and matrices, and the
    bind pose listing the mesh's node; None for no skin, ValueError for
    several skins or several bind poses"""
    from cgmath.formats._fbx_io import typed

    kind  = fbx.FbxDeformer.EDeformerType.eSkin
    count = shape.GetDeformerCount(kind)
    if count == 0:
        return None
    if count > 1:
        raise ValueError(f"load_fbx: {label} holds {count} skin deformers; {FORK} reads one")
    place    = {entry.GetUniqueID(): i for i, (entry, _) in enumerate(found)}
    deformer = typed(shape.GetDeformer(0, kind), fbx.FbxSkin, fbx)
    clusters = []
    for k in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(k)
        link    = cluster.GetLink()
        if link is None:
            continue
        linked_at = place.get(link.GetUniqueID())
        if linked_at is None:
            raise ValueError(
                f"load_fbx: skin {str(deformer.GetName())!r} cluster {k} links {str(link.GetName())!r}, no node of the "
                f"hierarchy"
            )
        linked, placed = fbx.FbxAMatrix(), fbx.FbxAMatrix()
        cluster.GetTransformLinkMatrix(linked)
        cluster.GetTransformMatrix(placed)
        clusters.append(
            {
                "node":    linked_at,
                "indices": np.asarray(cluster.GetControlPointIndices(), dtype=np.int64),
                "weights": np.asarray(cluster.GetControlPointWeights(), dtype=np.float64),
                "link":    _amatrix(linked),
                "mesh":    _amatrix(placed),
            }
        )

    poses = [scene.GetPose(i) for i in range(scene.GetPoseCount())]
    poses = [pose for pose in poses if pose.IsBindPose() and pose.Find(node) >= 0]
    if len(poses) > 1:
        raise ValueError(f"load_fbx: {label} is listed by {len(poses)} bind poses; {FORK} reads one")
    binds = {}
    for pose in poses:
        for k in range(pose.GetCount()):
            target = pose.GetNode(k)
            if target is not None and target.GetUniqueID() in place:
                binds[place[target.GetUniqueID()]] = _amatrix(pose.GetMatrix(k))
    return {
        "name":     str(deformer.GetName()),
        "mesh":     label,
        "node":     at,
        "count":    shape.GetControlPointsCount(),
        "clusters": clusters,
        "binds":    binds,
    }


def _read(scene, fbx, parts: dict, filename: str = "") -> dict:
    """everything the build needs, out of the scene as python and numpy, in
    Maya's Y-up and cm (:func:`_convert`): every node, one uuid each, and
    the kinds ``parts`` turns on (:func:`load_fbx`'s flags): every take's
    curves and the active take's position, each mesh with what sits on it,
    its skin, the cameras and the lights"""
    from cgmath.formats._fbx_io import typed
    from cgmath.hierarchy.hierarchy import (
        _FBX_ROTATE_ORDERS,
        _FBX_VISIBLE,
        _fbx_enum,
        _fbx_joint_look,
        _fbx_negative_zeros,
        _fbx_user_attributes,
        _fbx_written_axis,
        generate_uuid,
    )

    source  = fbx.FbxNode.EPivotSet.eSourcePivot
    inherit = fbx.FbxTransform.EInheritType
    cross   = _fbx_enum(fbx.FbxNull, "ELook", "eCross")
    found   = _walk(scene)
    kinds   = fbx.FbxNodeAttribute.EType
    shapes  = []
    cameras = []
    lights  = []
    cache   = {}
    if parts["meshes"] or parts["cameras"] or parts["lights"]:
        for at, (node, _) in enumerate(found):
            attribute = node.GetNodeAttribute()
            kind      = None if attribute is None else attribute.GetAttributeType()
            if kind == kinds.eMesh and parts["meshes"]:
                mesh         = _read_mesh(node, at, filename, fbx, parts, cache)
                shape        = mesh.pop("shape")
                mesh["skin"] = _read_skin(scene, node, at, mesh["label"], shape, found, fbx) if parts["skins"] else None
                shapes.append(mesh)
            elif kind == kinds.eCamera and parts["cameras"]:
                cameras.append(_read_camera(node, at, fbx))
            elif kind == kinds.eLight and parts["lights"]:
                lights.append(_read_light(node, at, fbx))
    nodes = []
    for node, parent in found:
        name      = str(node.GetName())
        attribute = node.GetNodeAttribute()
        kind      = None if attribute is None else attribute.GetAttributeType()
        joint     = kind == kinds.eSkeleton
        # a null drawn as a cross is a locator, as Maya's own export writes
        # one; a bare null is a group
        locator = kind == kinds.eNull and typed(attribute, fbx.FbxNull, fbx).Look.Get() == cross
        # a camera or a light on the node: its transform is turned into
        # Maya's (_turn_views), whether the shape is read or not
        view = None
        if kind == kinds.eCamera:
            view = "camera"
        elif kind == kinds.eLight:
            view = _light_kind(typed(attribute, fbx.FbxLight, fbx), fbx)
        active = bool(node.GetRotationActive())
        stored = node.GetRotationOrder(source)
        stored = int(getattr(stored, "value", stored))
        signs  = _fbx_negative_zeros(node)

        translate = np.array(tuple(node.LclTranslation.Get())[:3], dtype=np.float64)
        rotate    = np.array(tuple(node.LclRotation.Get())[:3],    dtype=np.float64)
        scale     = np.array(tuple(node.LclScaling.Get())[:3],     dtype=np.float64)
        orient    = np.zeros(3)
        axis      = np.zeros(3)
        if active:
            orient = np.array(tuple(node.GetPreRotation(source))[:3], dtype=np.float64)
            post   = fbx.FbxAMatrix()
            post.SetR(node.GetPostRotation(source))
            axis = np.array(_fbx_written_axis(node, tuple(x + 0.0 for x in tuple(post.Inverse().GetR())[:3])))
        for channel, value in (("translate", translate), ("rotate", rotate), ("joint_orient", orient)):
            if channel in signs and not value.any():
                value[...] = np.where(signs[channel], -0.0, value)

        if not joint and np.any(orient != 0.0):
            raise ValueError(
                f"load_fbx: node {name!r} channel 'joint_orient' is a pre rotation on a transform, which Maya "
                f"folds into its rotate and {FORK} does not read"
            )
        # the rotate pivot, scale pivot, rotate offset and scale offset, which
        # _fold_pivots folds into a transform's translate once the takes are
        # read; on a joint they would sit before the segment scale compensate
        # (measured: Maya's joint honours them), which the fold does not take
        pivots = np.array(
            [
                tuple(getter(source))[:3]
                for getter in (node.GetRotationPivot, node.GetScalingPivot, node.GetRotationOffset, node.GetScalingOffset)
            ],
            dtype=np.float64,
        )
        if not np.any(pivots != 0.0):
            pivots = None
        elif joint:
            raise ValueError(
                f"load_fbx: joint {name!r} channel 'pivots' holds pivots or offsets, which {FORK} folds on a transform, not on "
                f"a joint"
            )

        kind = node.GetTransformationInheritType()
        if kind == inherit.eInheritRrs:
            held = "Rrs"
        elif kind == inherit.eInheritRrSs:
            held = "RrSs"
        else:
            held = "RSrs"

        entry = {
            "name":         name,
            "parent":       parent,
            "uuid":         generate_uuid(),
            "node_type":    "joint" if joint else "locator" if locator else "transform",
            "view":         view,
            "translate":    translate,
            "rotate":       rotate,
            "scale":        scale,
            "rotate_order": _FBX_ROTATE_ORDERS[stored if active else 0],
            "joint_orient": orient,
            "rotate_axis":  axis,
            "inherit":      held,
            "visibility":   float(node.Visibility.Get()) > _FBX_VISIBLE,
            "attrs":        _fbx_user_attributes(node),
            "pivots":       pivots,
            "index":        None,
            "id":           node.GetUniqueID(),
        }
        if joint:
            entry["radius"], entry["draw_style"] = _fbx_joint_look(node)
        kept = node.FindProperty(ORDER_PROPERTY)
        if kept.IsValid():
            entry["index"] = int(fbx.FbxPropertyInteger1(kept).Get())
        nodes.append(entry)

    # a bind pose (FbxPose) is left out: no node takes its values from one
    read = {"nodes": nodes, "takes": None, "active": 0, "meshes": shapes, "cameras": cameras, "lights": lights}
    if parts["takes"]:
        read["takes"], read["active"] = _takes(scene, fbx, found, nodes)
    _fold_pivots(read)
    _turn_views(read)
    _view_keys(read, fbx)
    _morph_keys(read, fbx)
    _convert(read, *_axes(scene, fbx, filename))
    return read


def _shifts(vectors) -> np.ndarray:
    """(..., 4, 4) translation matrices moving by ``vectors`` (..., 3)"""
    vectors = np.asarray(vectors, dtype=np.float64)
    out     = np.broadcast_to(np.eye(4), vectors.shape[:-1] + (4, 4)).copy()

    out[..., 3, :3] = vectors
    return out


def _folded(translate, rotate, scale, orient, axis, order: int, pivots) -> np.ndarray:
    """(F, 3) the translate holding a node's pivots and offsets, frame by
    frame: the position of the matrix Maya composes, Sp^-1 S Sp St Rp^-1 RA
    R JO Rp Rt T (measured against Maya's own evaluation and the sdk's),
    whose 3x3, S RA R JO, the pivots leave alone"""
    from cgmath.transforms import euler_to_matrix

    translate = np.asarray(translate, dtype=np.float64).reshape(-1, 3)
    rotate    = np.asarray(rotate, dtype=np.float64).reshape(-1, 3)
    scale     = np.asarray(scale, dtype=np.float64).reshape(-1, 3)
    count     = len(translate)
    rp, sp, rt, st = (np.asarray(pivots, dtype=np.float64)[k] for k in range(4))

    scaling = np.zeros((count, 4, 4))
    scaling[:, [0, 1, 2], [0, 1, 2]] = scale
    scaling[:, 3, 3]                 = 1.0
    turned = euler_to_matrix(np.radians(np.broadcast_to(np.asarray(axis, dtype=np.float64), (count, 3))), 0)
    turned = turned @ euler_to_matrix(np.radians(rotate), int(order))
    turned = turned @ euler_to_matrix(np.radians(np.broadcast_to(np.asarray(orient, dtype=np.float64), (count, 3))), 0)
    matrices = (
        _shifts(-sp) @ scaling @ _shifts(sp) @ _shifts(st)
        @ _shifts(-rp) @ turned @ _shifts(rp) @ _shifts(rt) @ _shifts(translate)
    )
    return matrices[:, 3, :3]


def _fold_pivots(read: dict) -> None:
    """every node's pivots and offsets folded into its translate, in place:
    its own values, and on every frame of every take, the frames filled
    from its own values where a channel holds no curve, the folded
    translate keyed wherever translate, rotate or scale is; the node's 3x3
    is untouched (:func:`_folded`). The file's own axes and unit: the fold
    comes before the roots are turned and the lengths converted"""
    entries = read["nodes"]
    takes   = read["takes"] or []
    for i, entry in enumerate(entries):
        pivots = entry.pop("pivots")
        if pivots is None:
            continue
        own = {channel: np.asarray(entry[channel], dtype=np.float64) for channel in ("translate", "rotate", "scale")}
        for take in takes:
            values = take["values"][i][0]
            if all(value is None for channel in own for value in values[channel]):
                continue
            count  = len(take["frames"])
            blocks = {}
            for channel, static in own.items():
                block = np.empty((count, 3))
                for k in range(3):
                    value       = values[channel][k]
                    block[:, k] = static[k] if value is None else value
                blocks[channel] = block
            moved = _folded(
                blocks["translate"], blocks["rotate"], blocks["scale"], entry["joint_orient"], entry["rotate_axis"],
                entry["rotate_order"], pivots,
            )
            keyed               = any(isinstance(value, np.ndarray) for channel in own for value in values[channel])
            values["translate"] = [moved[:, k] if keyed else float(moved[0, k]) for k in range(3)]
        entry["translate"] = _folded(
            own["translate"], own["rotate"], own["scale"], entry["joint_orient"], entry["rotate_axis"], entry["rotate_order"],
            pivots,
        )[0]


# Maya's exporter turns a camera's node and a directed light's node to fbx's
# axes with a post rotation (measured): fbx's camera looks down its node's
# +X and its light shines down -Y, Maya's both down -Z. The walk reads a post
# rotation as a rotate axis; these are the rotate axes it reads on a node Maya
# wrote, by what the node holds
_VIEW_AXES = {
    "camera":      (0.0, 90.0, 0.0),
    "directional": (90.0, 0.0, 0.0),
    "spot":        (90.0, 0.0, 0.0),
    "area":        (90.0, 0.0, 0.0),
}

# Maya's orthographic width is fbx's OrthoZoom times this (measured: a width
# of 50 is written as 1.6667)
ORTHO_UNIT = 30.0

# fbx's light Intensity is Maya's times this (measured: 2.5 is written as 250)
INTENSITY_UNIT = 100.0


def _enum_int(value) -> int:
    """an fbx enum read off a property, as its int, whichever spelling the
    binding gives"""
    return int(getattr(value, "value", value))


def _read_camera(node, at: int, fbx) -> dict:
    """the camera on the node at place ``at`` of the file's nodes, as Maya's
    camera attributes: the focal length as the file holds it, or the one its
    field of view gives for a file holding the view rather than the lens
    (``ApertureMode``), the film back and offsets in inches, the gate fit as
    the film fit (fill for fbx's none and stretch), the clip planes, and the
    orthographic width as ``ORTHO_UNIT`` times the ortho zoom; fbx holds no
    camera scale, so 1"""
    import math

    from cgmath.formats._fbx_io import typed
    from cgmath.hierarchy.hierarchy import _fbx_enum

    camera = typed(node.GetNodeAttribute(), fbx.FbxCamera, fbx)
    name   = str(camera.GetName()) or None
    label  = f"camera {name!r}" if name else f"the camera on node {str(node.GetName())!r}"
    modes  = {_enum_int(_fbx_enum(fbx.FbxCamera, "EApertureMode", x)): x for x in ("eHorizAndVert", "eHorizontal", "eVertical", "eFocalLength")}
    fits = {
        _enum_int(_fbx_enum(fbx.FbxCamera, "EGateFit", x)): fit
        for x, fit in (
            ("eFitNone", "fill"), ("eFitVertical", "vertical"), ("eFitHorizontal", "horizontal"), ("eFitFill", "fill"),
            ("eFitOverscan", "overscan"), ("eFitStretch", "fill"),
        )
    }
    width  = float(camera.FilmWidth.Get())
    height = float(camera.FilmHeight.Get())
    focal  = float(camera.FocalLength.Get())
    mode   = modes.get(_enum_int(camera.ApertureMode.Get()), "eFocalLength")
    if mode != "eFocalLength":
        # the view is the file's word: the lens that gives it on the film back
        if mode == "eVertical":
            view, aperture = float(camera.FieldOfView.Get()), height
        elif mode == "eHorizontal":
            view, aperture = float(camera.FieldOfView.Get()), width
        else:
            view, aperture = float(camera.FieldOfViewX.Get()), width
        focal = aperture * 25.4 / (2.0 * math.tan(math.radians(view) / 2.0))
    ortho = _enum_int(camera.ProjectionType.Get()) == _enum_int(_fbx_enum(fbx.FbxCamera, "EProjectionType", "eOrthogonal"))
    return {
        "node":      at,
        "name":      name,
        "label":     label,
        "attribute": camera,
        "fields": {
            "focal_length": focal,
            "horizontal_film_aperture": width,
            "vertical_film_aperture": height,
            "film_fit": fits.get(_enum_int(camera.GateFit.Get()), "fill"),
            "film_offset": [float(camera.FilmOffsetX.Get()), float(camera.FilmOffsetY.Get())],
            "lens_squeeze_ratio": float(camera.FilmSqueezeRatio.Get()),
            "camera_scale": 1.0,
            "near_clip_plane": float(camera.NearPlane.Get()),
            "far_clip_plane": float(camera.FarPlane.Get()),
            "orthographic": ortho,
            # the ortho zoom means nothing on a perspective camera, where
            # Maya's export leaves it 1: Maya's default width then
            "orthographic_width": float(camera.OrthoZoom.Get()) * ORTHO_UNIT if ortho else 10.0,
        },
    }


def _light_kind(light, fbx) -> str:
    """what an FbxLight is: point, directional, spot, area or volume by its
    type, ambient for a point light that casts no light, which is how Maya's
    exporter writes an ambient light (measured)"""
    from cgmath.hierarchy.hierarchy import _fbx_enum

    names = {
        _enum_int(_fbx_enum(fbx.FbxLight, "EType", x)): kind
        for x, kind in (("ePoint", "point"), ("eDirectional", "directional"), ("eSpot", "spot"), ("eArea", "area"), ("eVolume", "volume"))
    }
    kind = names.get(_enum_int(light.LightType.Get()), "point")
    if kind == "point" and not bool(light.CastLight.Get()):
        return "ambient"
    return kind


def _read_light(node, at: int, fbx) -> dict:
    """the light on the node at place ``at`` of the file's nodes, as Maya's
    light attributes: its kind (:func:`_light_kind`), colour, the intensity
    as the file's over ``INTENSITY_UNIT``, the decay type as the decay rate,
    and a spot's inner angle as its cone, half the outer angle's excess as
    its penumbra; fbx holds no dropoff, so 0"""
    from cgmath.formats._fbx_io import typed

    light = typed(node.GetNodeAttribute(), fbx.FbxLight, fbx)
    name  = str(light.GetName()) or None
    kind  = _light_kind(light, fbx)
    # the cone is a spot's; any other kind keeps Maya's defaults
    inner = float(light.InnerAngle.Get()) if kind == "spot" else 40.0
    outer = float(light.OuterAngle.Get()) if kind == "spot" else 40.0
    return {
        "node":      at,
        "name":      name,
        "label":     f"{kind} light {name!r}" if name else f"the {kind} light on node {str(node.GetName())!r}",
        "attribute": light,
        "kind":      kind,
        "fields": {
            "kind":           kind,
            "color":          [float(x) for x in tuple(light.Color.Get())[:3]],
            "intensity":      float(light.Intensity.Get()) / INTENSITY_UNIT,
            "decay_rate":     min(max(_enum_int(light.DecayType.Get()), 0), 3),
            "cone_angle":     inner,
            "penumbra_angle": (outer - inner) / 2.0,
            "dropoff":        0.0,
        },
    }


def _turn_views(read: dict) -> None:
    """every node holding a camera or a directed light turned into Maya's
    camera or light transform, in place. Fbx's camera looks down its node's
    +X and its light shines down -Y, Maya's both down -Z, and Maya's
    exporter bridges them with a post rotation on the node, (0, -90, 0) on a
    camera and (-90, 0, 0) on a directional, spot or area light (measured),
    which the walk reads as a rotate axis (``_VIEW_AXES``). A node holding
    that rotate axis, Maya's own, keeps its rotate to the bit and loses the
    axis; any other node's rotation, rotate axis and rotate together, is
    turned by the inverse of Maya's axis and solved into its rotate, every
    frame of every take at once through euler_filter where a take keys it,
    its rotate axis zeroed, so the camera looks, or the light shines, where
    the file points it. ValueError for a node under one: a child's place
    would move with the turn"""
    from cgmath.hierarchy.hierarchy import _filtered
    from cgmath.transforms import euler_to_matrix, matrix_to_euler

    entries = read["nodes"]
    views   = [i for i, entry in enumerate(entries) if entry["view"] in _VIEW_AXES]
    if not views:
        return
    held = set(views)
    for entry in entries:
        if entry["parent"] in held:
            above = entries[entry["parent"]]
            raise ValueError(
                f"load_fbx: node {entry['name']!r} sits under node {above['name']!r}, which holds a "
                f"{above['view'] if above['view'] == 'camera' else above['view'] + ' light'}; {FORK} reads a camera or "
                f"light on a leaf node"
            )

    def rotation(angles, order):
        """(F, 3, 3) of (F, 3) degrees in ``order``"""
        return euler_to_matrix(np.radians(np.asarray(angles, dtype=np.float64).reshape(-1, 3)), order)[:, :3, :3]

    def solved(matrices, order):
        """(F, 3) degrees of (F, 3, 3) in ``order``"""
        whole            = np.zeros((len(matrices), 4, 4))
        whole[:, :3, :3] = matrices
        whole[:, 3, 3]   = 1.0
        return np.degrees(matrix_to_euler(whole, order)).reshape(-1, 3)

    for i in views:
        entry = entries[i]
        axis0 = np.array(_VIEW_AXES[entry["view"]], dtype=np.float64)
        order = int(entry["rotate_order"])
        if np.allclose(entry["rotate_axis"], axis0, rtol=0, atol=1e-9):
            entry["rotate_axis"] = np.zeros(3)
            continue
        turn                 = rotation(-axis0, 0)[0] @ rotation(entry["rotate_axis"], 0)[0]  # inv(RA0) @ RA, xyz both
        static               = entry["rotate"].copy()
        entry["rotate"]      = solved((turn @ rotation(static, order))[None], order)[0] + 0.0
        entry["rotate_axis"] = np.zeros(3)
        for take in read["takes"] or []:
            values = take["values"][i][0]["rotate"]
            if all(value is None for value in values):
                continue
            frames = len(take["frames"])
            rows   = np.empty((frames, 3))
            for k in range(3):
                rows[:, k] = static[k] if values[k] is None else values[k]
            turned = solved(turn @ rotation(rows, order), order)
            if any(isinstance(value, np.ndarray) for value in values):
                turned                         = _filtered(np.concatenate([entry["rotate"][None], turned])[:, None], [order])[1:, 0] + 0.0
                take["values"][i][0]["rotate"] = [turned[:, k] for k in range(3)]
            else:
                take["values"][i][0]["rotate"] = [float(turned[0, k]) for k in range(3)]


# the camera and light properties a take may key, each as the field it
# fills: read at the active take's first frame, static, with one warning
_CAMERA_KEYS = {
    "FocalLength":      "focal_length",
    "FilmWidth":        "horizontal_film_aperture",
    "FilmHeight":       "vertical_film_aperture",
    "FilmOffsetX":      "film_offset",
    "FilmOffsetY":      "film_offset",
    "FilmSqueezeRatio": "lens_squeeze_ratio",
    "NearPlane":        "near_clip_plane",
    "FarPlane":         "far_clip_plane",
    "OrthoZoom":        "orthographic_width",
}
_LIGHT_KEYS = {"Color": "color", "Intensity": "intensity", "InnerAngle": "cone_angle", "OuterAngle": "penumbra_angle"}


def _view_keys(read: dict, fbx) -> None:
    """a camera or light property the active take keys: the shape takes the
    take's first frame's value, and one warning names the shape and its
    keyed properties; the fork reads cameras and lights static"""
    import warnings

    from cgmath.formats._fbx_io import typed

    takes = read["takes"]
    if not takes:
        return
    take  = takes[read["active"]]
    layer = take["layer"]
    if layer is None:
        return
    when = fbx.FbxTime(int(take["ticks"][0]))

    def evaluated(curve) -> float:
        """the curve's value at ``when``; the binding may hand back (value, index)"""
        value = typed(curve, fbx.FbxAnimCurve, fbx).Evaluate(when)
        return float(value[0] if isinstance(value, tuple) else value)

    for held, keys in ((read["cameras"], _CAMERA_KEYS), (read["lights"], _LIGHT_KEYS)):
        for shape in held:
            attribute = shape["attribute"]
            keyed     = []
            for prop_name, field in keys.items():
                prop = getattr(attribute, prop_name)
                if prop.GetCurveNode(layer, False) is None:
                    continue
                keyed.append(prop_name)
                if prop_name == "Color":
                    shape["fields"]["color"] = [
                        evaluated(prop.GetCurve(layer, component, False)) if prop.GetCurve(layer, component, False) is not None else old
                        for component, old in zip("XYZ", shape["fields"]["color"])
                    ]
                    continue
                curve = prop.GetCurve(layer, False)
                if curve is None:
                    continue
                value = evaluated(curve)
                if prop_name == "FilmOffsetX":
                    shape["fields"]["film_offset"][0] = value
                elif prop_name == "FilmOffsetY":
                    shape["fields"]["film_offset"][1] = value
                elif prop_name == "OrthoZoom":
                    shape["fields"]["orthographic_width"] = value * ORTHO_UNIT
                elif prop_name == "Intensity":
                    shape["fields"]["intensity"] = value / INTENSITY_UNIT
                elif prop_name == "InnerAngle":
                    shape["fields"]["penumbra_angle"] += (shape["fields"]["cone_angle"] - value) / 2.0
                    shape["fields"]["cone_angle"] = value
                elif prop_name == "OuterAngle":
                    shape["fields"]["penumbra_angle"] = (value - shape["fields"]["cone_angle"]) / 2.0
                else:
                    shape["fields"][field] = value
            if keyed:
                warnings.warn(
                    f"load_fbx: {shape['label']} keys {keyed} in take {take['name']!r}; {FORK} reads a camera or "
                    f"light static, at the take's first frame",
                    stacklevel=2,
                )


def _takes(scene, fbx, found, nodes) -> tuple:
    """``(takes, active)``: every animation stack of the file, in its order,
    each sampled on every frame of its span widened to its keys, its name
    (:func:`_take_names`: a blank one named by its position, a repeat told
    apart), fps, frame numbers and values (:func:`_sampled`) off the one
    layer holding curves; and the position of the active one, the stack
    ``ActiveAnimStackName`` names, else the first. ``(None, 0)`` for a file
    holding no stack"""
    from cgmath.formats._fbx_io import typed
    from cgmath.hierarchy.hierarchy import _fbx_frame_rate, _take_names

    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
    stacks   = [typed(scene.GetSrcObject(criteria, i), fbx.FbxAnimStack, fbx) for i in range(scene.GetSrcObjectCount(criteria))]
    if not stacks:
        return None, 0
    raw      = [str(stack.GetName()) for stack in stacks]
    names    = _take_names(raw)
    marked   = str(scene.ActiveAnimStackName.Get())
    active   = next((i for i, name in enumerate(raw) if name == marked), 0)

    settings = scene.GetGlobalSettings()
    mode     = settings.GetTimeMode()
    fps      = _fbx_frame_rate(mode)
    if fps <= 0.0:
        fps = float(settings.GetCustomFrameRate())
    second = fbx.FbxTime()
    second.SetSecondDouble(1.0)
    per_second = int(second.Get())
    layers     = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)

    takes = []
    for stack, name in zip(stacks, names):
        span  = stack.GetLocalTimeSpan()
        first = int(round(span.GetStart().Get() * fps / per_second))
        last  = int(round(span.GetStop().Get() * fps / per_second))
        if last < first:
            first, last = last, first

        # the layer holding curves: one holding none (an Unreal export
        # carries empty ones) is passed over; curves on two layers, or on a
        # muted one, blend, which the fork does not read
        held  = [typed(stack.GetMember(layers, k), fbx.FbxAnimLayer, fbx) for k in range(stack.GetMemberCount(layers))]
        keyed = [(layer, _layer_keys(layer, fbx)) for layer in held]
        keyed = [(layer, keys) for layer, keys in keyed if keys is not None]
        if len(keyed) > 1:
            raise NotImplementedError(
                f"load_fbx: the take {str(stack.GetName())!r} holds {len(keyed)} animation layers with curves; {FORK} reads one"
            )
        layer, keys = keyed[0] if keyed else (None, None)
        if layer is not None and bool(layer.Mute.Get()):
            raise NotImplementedError(
                f"load_fbx: the take {str(stack.GetName())!r} holds its curves on a muted layer, which {FORK} does not read"
            )

        # the take covers its stored span and every key: an Unreal export
        # stores the span a frame short of its last key (measured), a Maya
        # export can store one wider than its keys
        if keys is not None:
            first = min(first, int(np.floor(keys[0] * fps / per_second + 1e-6)))
            last  = max(last, int(np.ceil(keys[1] * fps / per_second - 1e-6)))
        frames = np.arange(first, last + 1)
        ticks  = np.round(frames * (per_second / fps)).astype(np.int64)

        curves = []
        for (node, _), entry in zip(found, nodes):
            if entry["pivots"] is not None and layer is not None and any(
                getattr(node, prop).GetCurveNode(layer, False) is not None
                for prop in ("RotationPivot", "ScalingPivot", "RotationOffset", "ScalingOffset")
            ):
                raise ValueError(
                    f"load_fbx: node {entry['name']!r} keys its pivots in take {str(stack.GetName())!r}, which {FORK} "
                    f"does not read"
                )
            held = {}
            for channel, prop in (("translate", node.LclTranslation), ("rotate", node.LclRotation), ("scale", node.LclScaling)):
                held[channel] = [
                    typed(prop.GetCurve(layer, component, False), fbx.FbxAnimCurve, fbx) if layer is not None else None
                    for component in "XYZ"
                ]
            # the node's visibility, one curve, stepped as Maya keys it
            held["visibility"] = [
                typed(node.Visibility.GetCurve(layer, False), fbx.FbxAnimCurve, fbx) if layer is not None else None
            ]
            users = {}
            for attr, spec in entry["attrs"].items():
                prop = node.FindProperty(attr)
                if layer is None or not prop.IsValid():
                    continue
                if spec.get("attributeType") in ("double3", "float3"):
                    users[attr] = [typed(prop.GetCurve(layer, component, False), fbx.FbxAnimCurve, fbx) for component in "XYZ"]
                elif "numberOfChildren" not in spec:
                    users[attr] = [typed(prop.GetCurve(layer, False), fbx.FbxAnimCurve, fbx)]
            curves.append((held, users))

        takes.append({
            "name":   name,
            "fps":    fps,
            "frames": frames,
            "ticks":  ticks,
            "layer":  layer,
            "values": _sampled(curves, ticks, per_second),
        })
    return takes, active


def _layer_keys(layer, fbx):
    """``(first, last)`` key time in ticks over every curve of the layer
    holding a key, None for a layer holding none. A curve keeps its keys in
    time order, so its first and last keys are its ends"""
    from cgmath.formats._fbx_io import typed

    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
    first    = last = None
    for j in range(layer.GetSrcObjectCount(criteria)):
        curve_node = typed(layer.GetSrcObject(criteria, j), fbx.FbxAnimCurveNode, fbx)
        for channel in range(curve_node.GetChannelsCount()):
            for k in range(curve_node.GetCurveCount(channel)):
                curve = typed(curve_node.GetCurve(channel, k), fbx.FbxAnimCurve, fbx)
                count = curve.KeyGetCount() if curve is not None else 0
                if not count:
                    continue
                lo    = int(curve.KeyGetTime(0).Get())
                hi    = int(curve.KeyGetTime(count - 1).Get())
                first = lo if first is None else min(first, lo)
                last  = hi if last is None else max(last, hi)
    return None if first is None else (first, last)


def _sampled(curves, ticks, per_second) -> list:
    """each node's ({channel: [component value or (F,) values]}, {user
    attribute: [...]}), None for a component with no curve; a curve of one
    key its value, of more its values on every frame"""
    from cgmath.hierarchy.hierarchy import _fbx_sample_curves

    many, where = [], []
    out = []
    for n, (held, users) in enumerate(curves):
        done = ({}, {})
        for side, group in enumerate((held, users)):
            for channel, components in group.items():
                values = []
                for k, curve in enumerate(components):
                    keys = curve.KeyGetCount() if curve is not None else 0
                    if keys == 0:
                        values.append(None)
                    elif keys == 1:
                        values.append(float(curve.KeyGetValue(0)))
                    else:
                        values.append(None)
                        many.append(curve)
                        where.append((n, side, channel, k))
                done[side][channel] = values
        out.append(done)
    if many:
        sampled = _fbx_sample_curves(many, ticks, per_second)
        for row, (n, side, channel, k) in zip(sampled, where):
            out[n][side][channel][k] = np.array(row, dtype=np.float64)
    return out


def _axes(scene, fbx, filename: str) -> tuple:
    """``(turn, cm)``: the (3, 3) rotation from the file's axes to Maya's
    Y-up, None for a file whose up axis is y, which Maya's importer leaves
    in its own axes whatever its front (measured), and the centimeters one
    of the file's units holds; ValueError for a left handed file, which no
    rotation turns"""
    from cgmath.hierarchy.hierarchy import _fbx_axis_turn, _fbx_scene_axes

    up, _, _, _, right = _fbx_scene_axes(scene, fbx)
    if not right:
        raise ValueError(
            f"load_fbx: {filename!r} is in a left handed axis system, which no rotation turns into Maya's Y-up; "
            f"{FORK} reads right handed files"
        )
    turn = None if up == 1 else _fbx_axis_turn(scene, "maya_y_up", filename, fbx)
    return turn, float(scene.GetGlobalSettings().GetSystemUnit().GetScaleFactor())


def _convert(read: dict, turn, cm: float) -> None:
    """``read`` put in Maya's Y-up and cm, in place: every length times
    ``cm``, the centimeters one of the file's units holds (translations,
    static and keyed, points and bind matrices), and with ``turn`` (3, 3)
    the file's roots turned (:func:`_fbx_turn_roots`, measured against
    Maya's importer): a joint root takes the turn in its joint orient and
    translate and keeps its rotate curves, any other root's rotate is solved
    again, frame by frame through euler_filter when the take keys it, and
    every bind matrix, a world, turns with them. A node under a root is
    relative to it and keeps its values; so does a mesh's points, in its
    node's space"""
    from cgmath.hierarchy.hierarchy import _fbx_turn_roots

    entries = read["nodes"]
    takes   = read["takes"] or []
    if cm != 1.0:
        for entry in entries:
            entry["translate"] = entry["translate"] * cm
        for take in takes:
            for held, _ in take["values"]:
                held["translate"] = [None if value is None else value * cm for value in held["translate"]]
        # a camera's lengths: its clip planes and its orthographic width
        for held in read["cameras"]:
            for name in ("near_clip_plane", "far_clip_plane", "orthographic_width"):
                held["fields"][name] = held["fields"][name] * cm
        for held in read["meshes"]:
            held["points"] = held["points"] * cm
            for morph in held["morphs"]:
                morph["offsets"] = morph["offsets"] * cm
            skin = held["skin"]
            if skin is None:
                continue
            for cluster in skin["clusters"]:
                cluster["link"][3, :3] *= cm
                cluster["mesh"][3, :3] *= cm
            for matrix in skin["binds"].values():
                matrix[3, :3] *= cm
    if turn is None:
        return

    turn          = np.asarray(turn, dtype=np.float64)
    turn4         = np.eye(4)
    turn4[:3, :3] = turn
    for held in read["meshes"]:
        skin = held["skin"]
        if skin is None:
            continue
        for cluster in skin["clusters"]:
            cluster["link"] = cluster["link"] @ turn4
            cluster["mesh"] = cluster["mesh"] @ turn4
        for at, matrix in skin["binds"].items():
            skin["binds"][at] = matrix @ turn4

    # the roots' own values, then the take's curves on them: each component
    # the file keys, the statics filling the others, turned every frame at
    # once. A translate component lands on the axis the turn sends its own
    # to, keyed as it was; a turned rotate mixes its components, so one key
    # keys all three, and a turned static stays static
    roots = [i for i, entry in enumerate(entries) if entry["parent"] < 0]
    if not roots:
        return
    channels = ("translate", "rotate", "joint_orient")
    joints   = [entries[i]["node_type"] == "joint" for i in roots]
    orders   = [entries[i]["rotate_order"] for i in roots]
    statics  = [np.array([[entries[i][channel] for i in roots]], dtype=np.float64) for channel in channels]
    turned   = _fbx_turn_roots(turn, joints, orders, *statics)
    for column, i in enumerate(roots):
        entries[i]["translate"] = turned[0][0, column]
        if joints[column]:
            entries[i]["joint_orient"] = turned[2][0, column]
        else:
            entries[i]["rotate"] = turned[1][0, column]

    sends = [int(np.flatnonzero(turn[k])[0]) for k in range(3)]
    for take in takes:
        frames = len(take["frames"])
        blocks = []
        for static, channel in zip(statics, channels):
            block = np.empty((frames, len(roots), 3))
            for column, i in enumerate(roots):
                values = take["values"][i][0].get(channel)
                for k in range(3):
                    value               = None if values is None else values[k]
                    block[:, column, k] = static[0, column, k] if value is None else value
            blocks.append(block)
        translate, rotate, _ = _fbx_turn_roots(turn, joints, orders, *blocks)
        for column, i in enumerate(roots):
            values = take["values"][i][0]
            was    = values["translate"]
            moved  = [None, None, None]
            for k in range(3):
                if was[k] is not None:
                    row             = translate[:, column, sends[k]]
                    moved[sends[k]] = row if isinstance(was[k], np.ndarray) else float(row[0])
            values["translate"] = moved
            if joints[column]:
                continue
            was = values["rotate"]
            if any(isinstance(value, np.ndarray) for value in was):
                values["rotate"] = [rotate[:, column, k] for k in range(3)]
            elif any(value is not None for value in was):
                values["rotate"] = [float(rotate[0, column, k]) for k in range(3)]


def _build(read: dict) -> tuple:
    """(the HierarchyData of the file's nodes at their own values, each the
    uuid the read gave it, the file's place of each of its nodes)"""
    from cgmath.hierarchy.hierarchy import HierarchyData, TransformData

    entries = read["nodes"]
    made    = []
    for entry in entries:
        joint  = entry["node_type"] == "joint"
        parent = entry["parent"]
        node = TransformData(
            entry["name"],
            node_type=entry["node_type"],
            uuid=entry["uuid"],
            parent_uuid=entries[parent]["uuid"] if parent >= 0 else None,
            scale=entry["scale"],
            rotate=entry["rotate"],
            translate=entry["translate"],
            rotate_order=entry["rotate_order"],
            rotate_axis=entry["rotate_axis"],
            joint_orient=entry["joint_orient"],
            visibility=entry["visibility"],
            segment_scale_compensate = joint and entry["inherit"] == "Rrs",
            radius=entry.get("radius"),
            draw_style=entry.get("draw_style"),
            user_defined_attributes=entry["attrs"] or None,
        )
        made.append(node)

    # the hierarchy's order: the one save_fbx kept, else the file's
    order = list(range(len(made)))
    kept  = [entry["index"] for entry in entries]
    if made and None not in kept and sorted(kept) == order:
        order = sorted(order, key=lambda i: kept[i])
    hierarchy = HierarchyData([made[i] for i in order])

    parents   = np.array([entry["parent"] for entry in entries], dtype=np.intp)
    in_order  = [made[i] for i in order]
    refuse_values(
        "load_fbx, the nodes",
        in_order,
        np.array([node.shear for node in in_order]).reshape(1, -1, 3),
        np.array([node.scale for node in in_order]).reshape(1, -1, 3),
    )
    back = np.argsort(order)
    _refuse_inherit(
        entries,
        parents,
        np.asarray(hierarchy.world_matrix)[back][None],
        np.array([node.scale for node in made], dtype=np.float64)[None],
        None,
    )
    return hierarchy, order


def _build_clips(read: dict, hierarchy, order):
    """the ClipList of the file's takes over ``hierarchy``, its nodes: one
    ClipData per take (:func:`_build_clip`), each over copies of the same
    nodes, so the takes share their uuids and the list admits them as one
    skeleton; the file's active take active"""
    from cgmath.hierarchy.hierarchy import ClipList

    clips        = ClipList([_build_clip(read, hierarchy, order, take) for take in read["takes"]])
    clips.active = read["active"]
    return clips


def _build_clip(read: dict, hierarchy, order, take: dict):
    """the ClipData of one of the file's takes, ``take`` as :func:`_takes`
    read it, over ``hierarchy``, its nodes"""
    from cgmath.hierarchy.hierarchy import ClipData, _enum_fields, _user_channels

    entries = read["nodes"]
    parents = np.array([entry["parent"] for entry in entries], dtype=np.intp)
    frames  = take["frames"]
    count   = len(frames)
    clip    = ClipData(hierarchy, frames=count, start_frame=int(frames[0]), fps=take["fps"], take=take["name"])
    clip._realign()
    for channel in ("translate", "rotate", "scale"):
        field = "_" + channel
        block = clip._blocks[field]
        keyed = np.zeros(len(clip), dtype=bool)
        for column, i in enumerate(order):
            values = take["values"][i][0][channel]
            static = entries[i][channel]
            rows   = np.empty((count, 3))
            for k in range(3):
                value      = values[k]
                rows[:, k] = static[k] if value is None else value
            keyed[column]    = any(isinstance(value, np.ndarray) for value in values) and count > 1
            block[:, column] = rows
        clip._animated[field] = keyed

    # each node's visibility: the file's value, or its Visibility curve's,
    # held frame by frame where the take changes it
    from cgmath.hierarchy.hierarchy import _FBX_VISIBLE

    shown = np.repeat(np.array([[entries[i]["visibility"] for i in order]], dtype=bool), count, axis=0)
    for column, i in enumerate(order):
        value = take["values"][i][0]["visibility"][0]
        if value is not None:
            shown[:, column] = np.asarray(value, dtype=np.float64) > _FBX_VISIBLE
    clip._show(shown)

    # the user channels: a key's value as the attribute holds it, an enum's
    # field place read back as its index
    for column, i in enumerate(order):
        node    = clip[column]
        attrs   = node.user_defined_attributes
        sampled = take["values"][i][1]
        for name, children in _user_channels(attrs).items():
            values = sampled.get(name)
            if values is None:
                continue
            width = 1 if children is None else 3
            names = (name,) if children is None else children
            rows  = np.empty((count, width))
            for k in range(width):
                value = values[k] if k < len(values) else None
                if value is None:
                    rows[:, k] = float(attrs[names[k]]["value"])
                    continue
                value = np.asarray(value, dtype=np.float64)
                if attrs[names[k]].get("attributeType") == "enum":
                    fields = _enum_fields(attrs[names[k]].get("enumName") or "")
                    place  = np.round(value).astype(int)
                    value  = np.array([fields[p][1] if 0 <= p < len(fields) else p for p in np.atleast_1d(place)], dtype=np.float64)
                    value  = value.reshape(np.shape(np.asarray(values[k])))
                rows[:, k] = value
            for k, child in enumerate(names):
                kind = attrs[child].get("attributeType")
                if kind == "bool":
                    rows[:, k] = (rows[:, k] != 0.0).astype(np.float64)
                elif kind in ("long", "short", "byte", "char", "enum"):
                    rows[:, k] = np.round(rows[:, k])
            if any(isinstance(value, np.ndarray) for value in values) and count > 1:
                clip._user_keys.setdefault(node.uuid, {})[name] = rows
            else:
                from cgmath.hierarchy.hierarchy import _user_put

                _user_put(attrs, name, children, rows[0])
    clip._bind(0)

    shown = clip.start_frame + np.arange(count)
    refuse_values("load_fbx, the take", clip, clip.frames.shear, clip.frames.scale, shown)
    back = np.argsort(order)
    _refuse_inherit(
        entries, parents, np.asarray(clip.frames.world_matrix)[:, back], np.asarray(clip.frames.scale)[:, back], shown
    )
    return clip


def _scales(worlds: np.ndarray) -> np.ndarray:
    """(F, N, 3) the scale of each world matrix: its rows' lengths"""
    return np.linalg.norm(np.asarray(worlds)[..., :3, :3], axis=-1)


def _refuse_inherit(entries, parents, worlds, scales, frames) -> None:
    """ValueError for an inherit type a joint or transform can not hold:
    eInheritRrSs under a parent whose world scale is not uniform (under a
    uniform one, whatever its size, the parent's scale commutes with the
    child's rotation and the type is the standard product), eInheritRrs on
    a transform under a parent whose own scale is not 1 (Maya holds it on
    joints alone, as segment scale compensate); ``worlds`` (F, N, 4, 4) and
    ``scales`` (F, N, 3) in the file's order"""
    for i, entry in enumerate(entries):
        parent = int(parents[i])
        if parent < 0 or entry["inherit"] == "RSrs" or (entry["inherit"] == "Rrs" and entry["node_type"] == "joint"):
            continue
        if entry["inherit"] == "RrSs":
            scale = _scales(worlds[:, parent])
            off   = np.flatnonzero(nonuniform(scale))
            what  = "eInheritRrSs under a parent whose world scale is not uniform"
        else:
            scale = np.asarray(scales[:, parent], dtype=np.float64)
            off   = np.flatnonzero(np.any(np.abs(scale - 1.0) > UNIT_SCALE, axis=-1))
            what  = "eInheritRrs on a transform under a parent whose scale is not 1"
        if len(off):
            when = "" if frames is None else f" frame {int(frames[off[0]])}"
            raise ValueError(
                f"load_fbx: node {entry['name']!r} channel 'inherit type'{when} is {what}, "
                f"{np.round(scale[off[0]], 9).tolist()}, which a joint or transform can not hold"
            )


def _build_cameras(read: dict):
    """the CameraList of the file's cameras, as ``read`` orders them, each
    on its node by uuid; None for none"""
    from cgmath.geometry.camera import CameraData, CameraList

    entries = read["nodes"]
    if not read["cameras"]:
        return None
    return CameraList(
        [CameraData(name=held["name"], parent_uuid=entries[held["node"]]["uuid"], **held["fields"]) for held in read["cameras"]]
    )


def _build_lights(read: dict):
    """the LightList of the file's point, directional and spot lights, as
    ``read`` orders them, each on its node by uuid; None for none. An
    ambient, area or volume light is no LightData: one warning names them,
    and each one's node is a transform"""
    import warnings

    from cgmath.geometry.light import LIGHT_KINDS, LightData, LightList

    entries = read["nodes"]
    skipped = [held["label"] for held in read["lights"] if held["kind"] not in LIGHT_KINDS]
    if skipped:
        warnings.warn(
            f"load_fbx: {', '.join(skipped)}: {FORK} reads point, directional and spot lights; the node is a transform",
            stacklevel=2,
        )
    lights = [
        LightData(name=held["name"], parent_uuid=entries[held["node"]]["uuid"], **held["fields"])
        for held in read["lights"]
        if held["kind"] in LIGHT_KINDS
    ]
    return LightList(lights) if lights else None


def _build_materials(read: dict, meshes):
    """the MaterialList of the meshes' materials, mesh by mesh in the
    meshes' order, each naming its mesh by uuid, a material on several
    meshes one record a mesh sharing one uuid; None for none. One warning
    each, naming each material and slot once: materials of another shading
    model, read as default lamberts; textures on slots the fork does not
    read, or that are no file texture; texture files not found; textures
    with no PIL to decode them"""
    import warnings

    from cgmath.geometry.material import MaterialData, MaterialList
    from cgmath.hierarchy.hierarchy import generate_uuid

    if meshes is None:
        return None
    materials = []
    uuids     = {}  # one uuid an FbxSurfaceMaterial, whatever nodes hold it
    bare      = []
    left      = []
    missing   = []
    no_pil    = []
    for held, mesh in zip(read["meshes"], meshes):
        for entry in held["materials"]:
            fields         = dict(entry["fields"])
            fields["uuid"] = uuids.setdefault(entry["id"], generate_uuid())
            texture        = entry["texture"]
            normal         = entry["normal"]
            if entry["bare"]:
                bare.append(f"{entry['label']} (shading model {entry['model']!r})")
            left.extend(f"{entry['label']} {slot}" for slot in entry["left"])
            for held_map, is_colour in ((texture, True), (normal, False)):
                if held_map is None:
                    continue
                if is_colour:
                    fields.update(texture=held_map["pixels"], texture_file=held_map["file"])
                else:
                    fields.update(
                        normal_map      = held_map["pixels"],
                        normal_map_file = held_map["file"],
                        normal_map_kind = entry["normal_kind"],
                        bump_depth      = entry["bump_depth"],
                    )
                if is_colour or texture is None:  # the UV set and wraps: the colour map's, else the normal map's
                    fields.update(uv_set=held_map["uv_set"], wrap_u=held_map["wrap_u"], wrap_v=held_map["wrap_v"])
                if held_map["missing"] is not None:
                    missing.append(f"{entry['label']} {held_map['missing']}")
                if held_map["no_pil"]:
                    no_pil.append(entry["label"])
            materials.append(MaterialData(name=entry["name"], mesh_uuid=mesh.uuid, faces=entry["faces"], **fields))
    # a material on many meshes is read once a mesh: each case named once
    bare, left, missing, no_pil = (list(dict.fromkeys(held)) for held in (bare, left, missing, no_pil))
    if bare:
        warnings.warn(
            f"load_fbx: {', '.join(bare)}: {FORK} reads lambert and phong materials; read as a default lambert of its name",
            stacklevel=2,
        )
    if left:
        warnings.warn(f"load_fbx: textures on slots {FORK} does not read, left out: {', '.join(left)}", stacklevel=2)
    if missing:
        warnings.warn(f"load_fbx: texture files not found, the texture None: {', '.join(missing)}", stacklevel=2)
    if no_pil:
        warnings.warn(
            f"load_fbx: {', '.join(no_pil)}: a texture is decoded through PIL (pillow), which is not installed; the texture is None",
            stacklevel=2,
        )
    return MaterialList(materials) if materials else None


def _build_morphs(read: dict, meshes):
    """the MorphList of the meshes' blend shape targets, mesh by mesh in the
    meshes' order, each naming its mesh by uuid; None for none"""
    from cgmath.geometry.morph_target import MorphData, MorphList

    if meshes is None:
        return None
    morphs = [
        MorphData(
            name      = held["name"],
            offsets   = held["offsets"],
            indices   = held["indices"],
            mesh_uuid = mesh.uuid,
            weight    = held["weight"],
            at        = held["at"],
            deformer  = held["deformer"],
        )
        for entry, mesh in zip(read["meshes"], meshes)
        for held in entry["morphs"]
    ]
    return MorphList(morphs) if morphs else None


def _parted_edges(mesh, corners: np.ndarray) -> np.ndarray:
    """the edges the corner normals ``corners`` part, as the mesh's own edge
    ids: an edge of two faces whose normals, at one of its ends, are more
    than NORMALS_DERIVED apart in a component. A border edge parts nothing;
    of an edge of more than two faces the first two are read, the ones
    split_at_hard_edges joins"""
    e2f = np.asarray(mesh.e2f)
    if e2f.shape[1] < 2:
        return np.empty(0, dtype=np.int32)

    # every edge's first two faces, its -1 padding last
    first  = np.argsort(e2f < 0, axis=1, kind="stable")[:, :2]
    faces  = np.take_along_axis(e2f, first, axis=1).astype(np.int64)
    shared = np.flatnonzero((faces >= 0).all(axis=1))
    faces  = faces[shared]
    ends   = np.asarray(mesh.e2v, dtype=np.int64)[shared]

    # the corner a face holds at a vertex, found by a sorted key
    indices = np.asarray(mesh.indices, dtype=np.int64)
    counts  = np.asarray(mesh.counts)
    width   = int(indices.max()) + 1
    keys    = np.repeat(np.arange(counts.size, dtype=np.int64), counts) * width + indices
    order   = np.argsort(keys, kind="stable")
    keys    = keys[order]

    parted  = np.zeros(shared.size, dtype=bool)
    for end in range(2):
        wanted = [faces[:, side] * width + ends[:, end] for side in range(2)]
        places = [np.minimum(np.searchsorted(keys, key), keys.size - 1) for key in wanted]
        found  = (keys[places[0]] == wanted[0]) & (keys[places[1]] == wanted[1])
        apart  = np.abs(corners[order[places[0]]] - corners[order[places[1]]])
        parted |= found & ~np.all(apart <= NORMALS_DERIVED, axis=1)
    return shared[parted].astype(np.int32)


def _least_normals(held: dict) -> tuple:
    """``(normals, normal_indices, hard_edges)`` of the read mesh ``held``:
    the least that shades it as the file does. The file's normals are held,
    a face corner at a time, against the ones Maya derives from the points
    (MeshData.get_face_vertex_normals): with every edge soft, then broken
    at the edges the file's own normals part (:func:`_parted_edges`).
    Within NORMALS_DERIVED a component the mesh stores none: ``(None, None,
    None)`` with every edge soft, else ``(None, None, those edges)``, a
    pair of vertices each. Any other normals, authored ones, come back as
    read, with no hard edge"""
    from cgmath.geometry.mesh import MeshData, _canonical_pairs, _compute_shading_normals
    from cgmath.geometry.utils.main import split_at_hard_edges

    normals, ids = held["normals"], held["normal_indices"]
    if normals is None or not ids.size or ids.min() < 0 or ids.max() >= normals.shape[0]:
        return normals, ids, None

    # a mesh of its own: what it keeps to derive normals stays off the one built
    mesh    = MeshData(points=held["points"], counts=held["counts"], indices=held["indices"])
    indices = np.asarray(mesh.indices)
    corners = normals[ids]

    def derived(groups, count: int) -> bool:
        ours = _compute_shading_normals(mesh, groups, count, True, True, force=False)[groups]
        return bool(np.all(np.abs(ours - corners) <= NORMALS_DERIVED))

    if derived(indices, mesh.point_count):
        return None, None, None
    parted = _parted_edges(mesh, corners)
    if parted.size:
        # the groups straight from the edge ids: no pair looked up again
        if derived(*split_at_hard_edges(indices, mesh.counts, mesh.e2v, mesh.e2f, parted)):
            return None, None, _canonical_pairs(np.asarray(mesh.e2v)[parted])
    return normals, ids, None


def _build_meshes(read: dict):
    """the MeshList of the file's meshes, as ``read`` orders them, each on
    its node by uuid and storing the least normals that shade it as the
    file does (:func:`_least_normals`); None for none"""
    from cgmath.geometry.mesh import MeshData, MeshList

    entries = read["nodes"]
    if not read["meshes"]:
        return None
    meshes = []
    for held in read["meshes"]:
        normals, ids, hard = _least_normals(held)
        meshes.append(
            MeshData(
                name           = held["name"],
                points         = held["points"],
                counts         = held["counts"],
                indices        = held["indices"],
                normals        = normals,
                normal_indices = ids,
                hard_edges     = hard,
                colors         = held["colors"],
                color_indices  = held["color_indices"],
                parent_uuid    = entries[held["node"]]["uuid"],
            )
        )
    return MeshList(meshes)


def _build_uvs(read: dict, meshes):
    """the UVList of every mesh's UV sets, in the file's order, each naming
    its mesh by uuid; None for no set"""
    from cgmath.geometry.mesh import UVData, UVList

    if meshes is None:
        return None
    sets = []
    for held, mesh in zip(read["meshes"], meshes):
        for set_name, points, indices in held["uvs"]:
            sets.append(
                UVData(
                    name      = set_name,
                    points    = points,
                    counts    = np.asarray(held["counts"], dtype=np.int32).copy(),
                    indices   = indices,
                    mesh_uuid = mesh.uuid,
                )
            )
    return UVList(sets) if sets else None


def _build_skins(read: dict, meshes):
    """the SkinList of the skinned meshes' skins in the file's order, each
    naming its mesh by uuid; None for none"""
    from cgmath.geometry.skin_weights import SkinList

    skins = [
        _build_skin(held["skin"], read["nodes"], mesh)
        for held, mesh in zip(read["meshes"], meshes)
        if held["skin"] is not None
    ]
    return SkinList(skins) if skins else None


def _build_skin(skin: dict, entries: list, mesh):
    """the SkinData of a mesh's skin, naming the mesh by its uuid: its
    points by its influences, in cluster order, a node reached by two
    clusters one column"""
    from cgmath.geometry.skin_weights import SkinData

    places = []
    for cluster in skin["clusters"]:
        if cluster["node"] not in places:
            places.append(cluster["node"])
    influences = [entries[at]["name"] for at in places]
    shared     = sorted({x for x in influences if influences.count(x) > 1})
    if shared:
        raise ValueError(f"load_fbx: skin {skin['name']!r} links several nodes named {shared}")
    weights = np.zeros((skin["count"], len(places)), dtype=np.float64)
    for cluster in skin["clusters"]:
        np.add.at(weights, (cluster["indices"], places.index(cluster["node"])), cluster["weights"])
    return SkinData(weights=weights, influences=influences, name=skin["name"] or None, mesh_uuid=mesh.uuid)


def _near(a, b, within: float) -> bool:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= within


def _agree(held: dict, at: int, matrix, where: str, entries: list) -> None:
    """``held[at]`` set to (``matrix``, ``where``); ValueError for one held
    already, from another skin or pose, that disagrees"""
    other = held.get(at)
    if other is None:
        held[at] = (matrix, where)
        return
    if not _near(other[0], matrix, SKIN_AGREE * max(1.0, float(np.abs(other[0]).max()))):
        raise ValueError(
            f"load_fbx: node {entries[at]['name']!r} channel 'bind pose' is {np.round(other[0], 9).tolist()} in "
            f"{other[1]}, {np.round(matrix, 9).tolist()} in {where}"
        )


def _settle_bind(read: dict, bind, order) -> None:
    """``bind``, the file's nodes at their own values, moved to the bind
    pose: a node the bind pose (or a cluster) places where its own values
    do keeps them, any other its channels solved from the bind matrix,
    parents first. A skinned mesh's node is placed by its skin's cluster
    transform, the mesh's world at the bind (Maya's geomMatrix), one on
    every cluster, which the bind pose agrees with. ValueError for skins
    whose clusters and bind poses disagree, or a bind matrix no channels
    hold"""
    entries = read["nodes"]
    links   = {}
    binds   = {}
    for held in read["meshes"]:
        skin = held["skin"]
        if skin is None:
            continue
        label = f"skin {skin['name']!r}"
        # the pose first: its matrices are written element by element, a
        # cluster's through the sdk's affine matrix, within 1e-15 of them
        for at, matrix in skin["binds"].items():
            _agree(binds, at, matrix, f"the bind pose of {skin['mesh']}", entries)
        for cluster in skin["clusters"]:
            _agree(links, cluster["node"], cluster["link"], f"{label}'s cluster", entries)
            _agree(binds, skin["node"], cluster["mesh"], f"{label}'s cluster transform", entries)
    for at, (matrix, where) in links.items():
        held = binds.get(at)
        if held is not None and not _near(held[0], matrix, SKIN_AGREE * max(1.0, float(np.abs(held[0]).max()))):
            raise ValueError(
                f"load_fbx: node {entries[at]['name']!r} channel 'bind pose' is {np.round(held[0], 9).tolist()} in "
                f"{held[1]}, {np.round(matrix, 9).tolist()} in {where}"
            )
    matrices = {at: matrix for at, (matrix, _) in links.items()}
    matrices.update({at: matrix for at, (matrix, _) in binds.items()})
    column = {at: i for i, at in enumerate(order)}
    # the file's order is depth first, each parent ahead of its children
    for at in range(len(entries)):
        matrix = matrices.get(at)
        if matrix is None:
            continue
        node = bind[column[at]]
        if _near(node.world_matrix, matrix, BIND_AGREE):
            continue
        node.world_matrix = matrix
        # a transform takes the shear a matrix holds: a float32 one's
        # rounding is none
        if np.any(node.shear != 0.0) and np.abs(node.shear).max() <= BIND_SOLVED:
            node.shear = np.zeros(3)
        if not _near(node.world_matrix, matrix, BIND_SOLVED * max(1.0, float(np.abs(matrix).max()))):
            raise ValueError(
                f"load_fbx: node {node.name!r} channel 'bind pose' is {np.round(matrix, 9).tolist()}, which its "
                f"channels can not hold (a shear)"
            )
    refuse_values(
        "load_fbx, the bind pose",
        bind,
        np.array([node.shear for node in bind]).reshape(1, -1, 3),
        np.array([node.scale for node in bind]).reshape(1, -1, 3),
    )
