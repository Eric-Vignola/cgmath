"""``cgmath.formats.save_fbx``: one fbx file from cgmath's objects

Each class's ``save_fbx`` is a one-line call to :func:`save_fbx`. Nothing
here imports the fbx sdk, nor the classes, until a file is written: the
classes' modules import this one.
"""

import logging
import os
import sys

import numpy as np

# the fbx writer's logger: its warnings and save_fbx's are one stream
LOGGER = logging.getLogger("cgmath.formats.fbx")

# fbx has no channel for shear: a node holding more than this is refused
SHEAR_LIMIT = 1e-9


def save_fbx(filename, *objects, zero_root: bool = False, as_ascii: bool = False) -> None:
    """writes ``objects`` to one fbx file, binary unless ``as_ascii``

    ``objects``, in any order, are paired by name:

    - one rig (a ``TransformList``, ``HierarchyData`` among them, or a
      ``ClipData`` of one frame, its pose), written as
      :meth:`TransformList.save_fbx` says, its label the file's (Maya's
      Y-up and cm when nothing labels it). Without one, the first take's
      rig at its frame is written, as :meth:`ClipList.save_fbx` writes it.
      The rig as passed in is the bind pose of the skins, the one
      ``HierarchyData.load_fbx(path, pose="bind")`` reads;
    - takes: ``ClipData`` of two frames or more and ``ClipList``, every
      one checked before anything is written (:meth:`ClipList.save_fbx`)
      and keyed on every frame, each node on the rig's node of its name.
      With a rig passed beside them, every node of a take must be the
      rig's and the takes the rig's axes and unit; a joint orient or rotate
      axis other than the rig's is keyed in the rotate, as
      :meth:`ClipList.save_fbx` keys one that moves; a rig's node no take
      holds is not keyed;
    - meshes (``MeshData``, ``MeshList``), each as
      :meth:`FbxExporter.add_mesh` writes it, on the rig's group of its
      name, else on a node of its own at the scene root named after it;
    - UV sets (``UVData``, ``UVList``), as :meth:`FbxExporter.add_uvs`
      writes them: a ``UVList`` holds one mesh's sets and pairs as one, a
      ``UVData`` alone, with the mesh its name or a name of the list's sets
      matches, else with the one mesh whose faces are its own (the same
      vertex count, face by face);
    - skins (``SkinData``, ``SkinList``), as :meth:`FbxExporter.add_skin`
      writes them, each influence the rig's node of its name, each skin
      paired with the mesh of its name, else with the one mesh of as many
      vertices as it has rows;
    - blend shapes (``MorphData``, ``MorphList``), cameras and lights
      (``Camera``, ``Light``) are left out, with one warning naming them.

    ``zero_root`` writes the first root joint at identity, unkeyed. ``~``
    in ``filename`` is expanded. Raises ImportError without the fbx sdk,
    before anything is checked; then, before anything is written,
    TypeError for an object of another kind (a render ``Scene`` among
    them), for UV sets or skins with no mesh, a skin with no rig and a call
    with nothing to write; ValueError for two rigs, node names that repeat
    where the file pairs by name (meshes of one name; a rig's, beside
    meshes, skins or takes), a mesh named after a rig node that is no group
    or with a matrix on a group of its name, a UV set or a skin that no
    mesh or two meshes fit (naming the candidates), a mesh given two sets
    of one name or two skins, a skin whose rows are not its mesh's
    vertices, an influence or a take's node the rig lacks, and each refusal
    of :meth:`ClipList.save_fbx`.
    """
    # imported here: the classes' modules import this one
    from cgmath.hierarchy import hierarchy

    if hierarchy.FBX is None or hierarchy.FbxExporter is None:
        raise ImportError("fbx module not found")

    # expand ~, consistent with the other IO paths
    filename = os.path.expanduser(filename)
    meshes, groups, skins, rigs, takes = _sort(_opened(objects))
    if (groups or skins) and not meshes:
        raise TypeError("save_fbx writes a UV set or a skin with its mesh: pass the mesh too")

    # every take checked before the sdk is reached, then against the rig;
    # takes is None when no clip was passed, [] for an empty ClipList
    names = hierarchy._takes_to_save(takes) if takes is not None else None
    if not (meshes or rigs or takes):
        held = ", ".join(type(x).__name__ for x in objects) or "nothing"
        raise TypeError(f"save_fbx: nothing to write, got {held}")
    if len(rigs) > 1:
        raise ValueError(f"save_fbx: {len(rigs)} rigs, a file holds one")
    if skins and not (rigs or takes):
        raise TypeError("save_fbx writes a skin with the rig its influences name: pass the rig too")
    _refuse_shear(rigs[0] if rigs else None, takes)

    columns = None
    if rigs:
        rig = rigs[0]
        if meshes or takes:
            _refuse_repeats([node.name for node in rig])
        if takes:
            columns, held = _takes_on_rig(takes, rig)
            takes = hierarchy._takes_written(takes, held)
    elif takes:
        takes = hierarchy._takes_written(takes)
        rig   = takes[0]
    else:
        rig = None

    _refuse_repeats([mesh.name or "mesh" for mesh in meshes])
    sets  = _pair_uvs(meshes, groups)
    owner = _pair_skins(meshes, skins)
    homes = _homes(meshes, owner, rig)

    with hierarchy.FbxExporter() as exporter:
        if rig is not None:
            exporter.add_skeleton(rig)
        nodes = {node.name: exporter._fbx_nodes.get(id(node)) for node in (rig if rig is not None else ())}
        for mesh, uvs, skin, home in zip(meshes, sets, owner, homes):
            shape = exporter.add_mesh(mesh, nodes[home] if home is not None else None)
            exporter.add_uvs(shape, uvs)
            if skin is not None:
                exporter.add_skin(shape, skin, [nodes[name] for name in skin.influences])
        if takes:
            mode = getattr(hierarchy.FBX.FbxTime.EMode, hierarchy._fbx_rate_mode(float(takes[0].fps)))
            exporter.add_takes(takes, names, mode, zero_root=zero_root, columns=columns)
        exporter.export(path=filename, as_ascii=as_ascii, zero_root=zero_root)


def _opened(objects) -> list:
    """``objects``, each FbxContents among them in its parts"""
    from cgmath.formats._load import FbxContents

    opened = []
    for item in objects:
        opened.extend(item.objects() if isinstance(item, FbxContents) else (item,))
    return opened


def _sort(objects) -> tuple:
    """``(meshes, groups, skins, rigs, takes)`` of ``objects``: MeshLists,
    SkinLists and ClipLists opened, each UVList a group of its sets and each
    UVData one of its own, a ClipData of one frame a rig and one of more a
    take, ``takes`` None when no clip is among them; blend shapes, cameras
    and lights left out with one warning; TypeError for any other object"""
    from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
    from cgmath.geometry.morph_target import MorphData, MorphList
    from cgmath.geometry.skin_weights import SkinData, SkinList
    from cgmath.hierarchy.hierarchy import ClipData, ClipList, TransformList

    # an object of render's is one only once render is imported: a save
    # does not import it
    scene  = sys.modules.get("cgmath.render.scene")
    unread = (scene.Camera, scene.Light) if scene is not None else ()
    whole  = (scene.Scene,) if scene is not None else ()
    meshes, groups, skins, rigs, takes, left = [], [], [], [], None, {}
    for item in objects:
        if isinstance(item, UVData):
            groups.append([item])
        elif isinstance(item, UVList):
            groups.append(list(item))
        elif isinstance(item, MeshData):
            meshes.append(item)
        elif isinstance(item, MeshList):
            meshes.extend(item)
        elif isinstance(item, SkinData):
            skins.append(item)
        elif isinstance(item, SkinList):
            skins.extend(item)
        elif isinstance(item, ClipList):
            takes = (takes or []) + item.list
        elif isinstance(item, ClipData) and item.frame_count > 1:
            takes = (takes or []) + [item]
        elif isinstance(item, (MorphData, MorphList) + unread):
            kind       = type(item).__name__
            left[kind] = left.get(kind, 0) + 1
        elif isinstance(item, TransformList) and not isinstance(item, whole):
            rigs.append(item)
        else:
            raise TypeError(
                f"save_fbx writes rigs, takes, meshes, UV sets and skins, not a {type(item).__name__}"
            )
    if left:
        counts = ", ".join(f"{count} {kind}" for kind, count in sorted(left.items()))
        LOGGER.warning(f"save_fbx left out what it does not write (blend shapes, cameras, lights): {counts}")
    return meshes, groups, skins, rigs, takes


def _takes_on_rig(takes, rig) -> tuple:
    """``(columns, held)``: the take column each node of ``rig`` is keyed
    from, by name, None for a node no take holds, and the joint orient and
    rotate axis the file holds for each take column, the rig's on a joint
    and none on any other node; ValueError for a take's node the rig lacks
    and takes labelled other than the rig"""
    from cgmath.hierarchy.hierarchy import TransformList

    first  = takes[0]
    whole  = rig if isinstance(rig, TransformList) else TransformList(list(rig))
    record = whole._records()
    if tuple(first._records()) != tuple(record):
        raise ValueError(
            f"save_fbx: the takes are in {first._records()}, the rig in {record}: one file holds one up axis "
            f"and one unit"
        )
    on_rig  = {node.name: node for node in rig}
    missing = [node.name for node in first if node.name not in on_rig]
    if missing:
        raise ValueError(f"save_fbx: the takes hold nodes the rig does not: {missing}")

    orient = np.zeros((len(first), 3))
    axis   = np.zeros((len(first), 3))
    for index, node in enumerate(first):
        held = on_rig[node.name]
        if held.node_type == "joint":
            orient[index] = np.asarray(held.joint_orient, dtype=np.float64)
            axis[index]   = np.asarray(held.rotate_axis, dtype=np.float64)
    column = {node.name: index for index, node in enumerate(first)}
    return [column.get(node.name) for node in rig], (orient, axis)


def _pair_uvs(meshes, groups) -> list:
    """the UV sets each of ``meshes`` holds, ``groups`` paired by name, else
    by face layout; ValueError, naming the candidates, for a group no mesh or
    two meshes fit, and for a mesh given two sets of one name"""
    names = [mesh.name or "mesh" for mesh in meshes]
    sets  = [[] for _ in meshes]
    where = {name: index for index, name in enumerate(names)}
    for whole in groups:
        # a set that names its mesh pairs with it, each set on its own
        group = []
        for uv in whole:
            index = where.get(uv.mesh_name) if uv.mesh_name is not None else None
            if index is None:
                group.append(uv)
                continue
            if not np.array_equal(uv.counts, meshes[index].counts):
                raise ValueError(f"save_fbx: the UV set {uv.name!r} holds other faces than mesh {names[index]!r}")
            sets[index].append(uv)
        if not group:
            continue
        labels = [uv.name for uv in group]
        named  = sorted({i for i, name in enumerate(names) if name in labels})
        if not named:
            layout = np.asarray(group[0].counts)
            if any(not np.array_equal(uv.counts, layout) for uv in group):
                raise ValueError(f"save_fbx: the UV sets {labels} hold other faces than each other: one mesh's sets")
            named = [i for i, mesh in enumerate(meshes) if np.array_equal(mesh.counts, layout)]
        if len(named) != 1:
            found = [names[i] for i in named]
            raise ValueError(
                f"save_fbx: the UV sets {labels} fit {'no mesh' if not found else f'meshes {found}'} by name or "
                f"by their faces: name a set after its mesh, or save them apart"
            )
        (index,) = named
        if not np.array_equal(group[0].counts, meshes[index].counts):
            raise ValueError(f"save_fbx: the UV sets {labels} hold other faces than mesh {names[index]!r}")
        sets[index].extend(group)
    for name, held in zip(names, sets):
        seen  = [uv.name for uv in held]
        twice = sorted({x for x in seen if seen.count(x) > 1})
        if twice:
            raise ValueError(f"save_fbx: mesh {name!r} would hold two UV sets named {twice}")
    return sets


def _pair_skins(meshes, skins) -> list:
    """the skin each of ``meshes`` holds, or None: each skin paired with the
    mesh of its name, else with the one mesh of as many vertices as it has
    rows; ValueError naming the candidates when two fit, or none, for two
    skins on one mesh and for a skin whose rows are not its mesh's vertices"""
    names = [mesh.name or "mesh" for mesh in meshes]
    owner = [None] * len(meshes)
    for skin in skins:
        rows  = np.asarray(skin.weights).shape[0]
        named = [i for i, name in enumerate(names) if name == skin.name]
        if not named:
            named = [i for i, mesh in enumerate(meshes) if mesh.point_count == rows]
        if len(named) != 1:
            found = [names[i] for i in named]
            raise ValueError(
                f"save_fbx: skin {skin.name!r} fits {'no mesh' if not found else f'meshes {found}'} by name or by "
                f"its {rows} rows: name the skin after its mesh, or save them apart"
            )
        (index,) = named
        if meshes[index].point_count != rows:
            raise ValueError(
                f"save_fbx: skin {skin.name!r} weighs {rows} vertices, mesh {names[index]!r} holds "
                f"{meshes[index].point_count}"
            )
        if owner[index] is not None:
            raise ValueError(f"save_fbx: two skins on mesh {names[index]!r}")
        owner[index] = skin
    return owner


def _homes(meshes, owner, rig) -> list:
    """the rig node each mesh goes on, by name, or None for a node of its
    own; ValueError for a mesh named after a rig node that is no group, or
    with a matrix on a group of its name, and a skin's influence the rig
    lacks"""
    if rig is None or not meshes:
        return [None] * len(meshes)
    from cgmath.hierarchy import SUPPORTED_NODE_TYPES

    kinds = {node.name: node.node_type for node in rig if node.node_type in SUPPORTED_NODE_TYPES}
    homes = []
    for mesh, skin in zip(meshes, owner):
        name = mesh.name or "mesh"
        kind = kinds.get(name)
        if kind is not None and kind != "transform":
            raise ValueError(f"save_fbx: mesh {name!r} is named after the rig's {kind}, which holds no mesh")
        if kind is not None and not np.array_equal(mesh.matrix, np.eye(4)):
            raise ValueError(
                f"save_fbx: mesh {name!r} has a matrix and a group of its name in the rig, whose transform it "
                f"takes: bake the matrix in (to_identity) or rename the mesh"
            )
        homes.append(name if kind is not None else None)
        missing = [x for x in (skin.influences if skin is not None else ()) if x not in kinds]
        if missing:
            raise ValueError(f"save_fbx: skin {skin.name!r} names influences the rig does not write: {missing}")
    return homes


def _refuse_repeats(names) -> None:
    """ValueError naming the names that repeat: fbx pairs objects by name"""
    seen, repeated = set(), []
    for name in names:
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    if repeated:
        raise ValueError(f"save_fbx: node names repeat, which a file pairs by: {repeated}")


def _refuse_shear(rig, takes, what: str = "save_fbx") -> None:
    """ValueError naming the nodes of ``rig`` and of each of ``takes`` that
    hold a shear over :data:`SHEAR_LIMIT` on any frame: fbx has no channel
    for it, and a save without it moves them and every node under them"""
    found, most = [], 0.0
    if rig is not None and len(rig):
        shear = np.abs(np.array([node.shear for node in rig], dtype=np.float64))
        bad   = np.flatnonzero(shear.max(axis=1) > SHEAR_LIMIT)
        if bad.size:
            found.append(f"the rig's {[rig[int(i)].name for i in bad]}")
            most = max(most, float(shear.max()))
    for index, clip in enumerate(takes or ()):
        if not clip.frame_count or not len(clip):
            continue
        shear = np.abs(np.asarray(clip.frames.shear, dtype=np.float64))
        held  = shear.max(axis=(0, 2)) > SHEAR_LIMIT
        if not held.any():
            continue
        frames = np.flatnonzero(shear.max(axis=(1, 2)) > SHEAR_LIMIT) + clip.start_frame
        most   = max(most, float(shear.max()))
        label  = clip.take.strip() or f"take {index}"
        found.append(
            f"take {label!r}'s {[clip[int(i)].name for i in np.flatnonzero(held)]} "
            f"(frames {int(frames[0])} to {int(frames[-1])})"
        )
    if found:
        raise ValueError(
            f"{what}: fbx has no shear, and these nodes hold one over {SHEAR_LIMIT:g}, up to {most:.3g}: "
            + "; ".join(found)
        )
