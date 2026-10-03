"""Filled-in examples of the geometry data classes, for test_roundtrip.

Each builder returns a new object, built the way a user builds one, with
every saved field holding something other than its default wherever that is
valid for the class. The meshes are one small bent panel: an 8-gon framing a
square hole, a quad folded up along one edge and a triangle folded up along
another.

Saved fields left at their default, and why:

- UVData: ``matrix``, ``normals`` and ``normal_indices``. UV space has no
  object transform and no shading normals.
- SDFSphere, SDFBox, SDFCylinder: ``node_type`` stays "transform", and the
  joint-only attributes (``_joint_orient``, ``_segment_scale_compensate``,
  ``_radius``, ``_draw_style``) stay unset: a primitive is not a joint. On the
  sphere and the cylinder the ``radius`` property is the SDF radius, which
  hides the joint radius anyway.

Lists also carry the empty cases: an empty mesh, an unused UV set, a morph
target equal to its base, a skin with no vertices.
"""

from __future__ import annotations

import numpy as np
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import MeshData, MeshList, UVData, UVList
from cgmath.geometry.morph_target import MorphData, MorphList
from cgmath.geometry.sdf import SDFBox, SDFCylinder, SDFSphere
from cgmath.geometry.skin_weights import (
    CompactSkinData,
    Patterns,
    SkinData,
    SkinList,
)

# ---------------------------------- PANEL ----------------------------------- #

# face 0 lists its outer loop, then its hole wound the other way
_PANEL_POINTS = (
    (0.0, 0.0, 0.0),  # outer loop
    (2.0, 0.0, 0.0),
    (2.0, 2.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.5, 0.5, 0.0),  # hole loop
    (0.5, 1.5, 0.0),
    (1.5, 1.5, 0.0),
    (1.5, 0.5, 0.0),
    (3.0, 0.0, 0.5),  # quad, folded up 26.6 degrees along edge 1-2
    (3.0, 2.0, 0.5),
    (1.0, 3.0, 0.25),  # triangle, folded up 14 degrees along edge 2-3
)
_PANEL_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 1, 8, 9, 2, 3, 2, 10)
_PANEL_COUNTS  = (8, 4, 3)

# the hole of face 0
_HOLE_FACES   = (0,)
_HOLE_COUNTS  = (4,)
_HOLE_INDICES = (4, 5, 6, 7)


def _panel_matrix() -> np.ndarray:
    """turned 30 degrees about Y and moved, rows as Maya writes them"""
    angle = np.radians(30.0)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [5.0, 0.0, -2.0, 1.0],
        ]
    )


def _plain_panel() -> MeshData:
    """the panel's topology and points, nothing else"""
    return MeshData(
        indices = list(_PANEL_INDICES),
        counts  = list(_PANEL_COUNTS),
        points  = np.array(_PANEL_POINTS),
    )


# --------------------------------- MESHES ----------------------------------- #


def mesh_data() -> MeshData:
    """the panel, placed, with its hole and hard-edged shading normals"""
    mesh = MeshData(
        indices      = list(_PANEL_INDICES),
        counts       = list(_PANEL_COUNTS),
        points       = np.array(_PANEL_POINTS),
        name         = "panelShape",
        matrix       = _panel_matrix(),
        hole_faces   = list(_HOLE_FACES),
        hole_counts  = list(_HOLE_COUNTS),
        hole_indices = list(_HOLE_INDICES),
    )

    # the quad's fold (26.6 degrees) is hard, the triangle's (14) smooth:
    # face-varying normals, split along edge 1-2. Corner angle and face area
    # weighting refuse n-gons, so the face normals are averaged as they are.
    mesh.set_normals(hard_edge_angle=20.0, angle_weighted=False, area_weighted=False)
    return mesh


def _empty_mesh() -> MeshData:
    """a mesh with no faces left: every array empty, points still (0, 3)"""
    return MeshData(
        indices = np.empty(0, dtype=np.int32),
        counts  = np.empty(0, dtype=np.int32),
        points  = np.empty((0, 3)),
        name    = "emptyShape",
    )


# a lightmap laid out for faces 0 and 1; face 2 has no UVs yet
_LIGHTMAP_UVS = (
    (0.0, 0.0),  # face 0, outer loop
    (0.5, 0.0),
    (0.5, 0.5),
    (0.0, 0.5),
    (0.125, 0.125),  # face 0, hole
    (0.125, 0.375),
    (0.375, 0.375),
    (0.375, 0.125),
    (0.6, 0.0),  # face 1
    (0.9, 0.0),
    (0.9, 0.5),
    (0.6, 0.5),
)


def uv_data() -> UVData:
    """
    A lightmap fragment: faces 0 and 1 laid out, face 2 a hole (a count of
    zero), the 8-gon's hole kept, rendered at 512 x 256 with antialiasing.
    """
    uv = UVData(
        indices      = np.arange(len(_LIGHTMAP_UVS)),
        counts       = [8, 4, 0],
        points       = np.array(_LIGHTMAP_UVS),
        name         = "lightmap",
        hole_faces   = list(_HOLE_FACES),
        hole_counts  = list(_HOLE_COUNTS),
        hole_indices = [4, 5, 6, 7],
    )
    uv.resolution = (512, 256)
    uv.antialias  = True
    return uv


def _lightmap_rest() -> UVData:
    """the other lightmap fragment: face 2 only, as defrag() would combine"""
    uv = UVData(
        indices = [0, 1, 2],
        counts  = [0, 0, 3],
        points  = np.array([[0.6, 0.6], [0.9, 0.6], [0.75, 0.9]]),
        name    = "lightmap",
    )
    uv.resolution = (512, 256)
    uv.antialias  = True
    return uv


def _empty_uv_set() -> UVData:
    """a UV set no face uses: counts all zero, no UVs, points still (0, 2)"""
    return UVData(
        indices = np.empty(0, dtype=np.int32),
        counts  = [0, 0, 0],
        points  = np.empty((0, 2)),
        name    = "uvSet_unused",
    )


def mesh_list() -> MeshList:
    """the panel, a UVData (a MeshData subclass) and an empty mesh"""
    return MeshList([mesh_data(), uv_data(), _empty_mesh()])


def uv_list() -> UVList:
    """
    Every UV set of the panel: map1 projected from the mesh (its name left at
    the default), the lightmap in two fragments, and an unused set.
    """
    map1 = mesh_data().to_uvdata(axis="z", maintain_ratio=True)
    return UVList([map1, uv_data(), _lightmap_rest(), _empty_uv_set()])


# ---------------------------------- MAPS ------------------------------------ #


def map_data() -> MapData:
    """a face map painted on two faces; unpainted faces read NaN"""
    return MapData(
        name           = "panel_wetness",
        indices        = [0, 2],
        values         = [0.35, 1.0],
        default_value  = float("nan"),
        component_type = "f",
        categories     = ["wetness", "fx"],
    )


def geom_subset_data() -> GeomSubsetData:
    """the panel's frame and flap faces, bound to one material"""
    return GeomSubsetData(
        name           = "frame_faces",
        indices        = [0, 1],
        component_type = "f",
        category       = "materialBind",
    )


# --------------------------------- MORPHS ----------------------------------- #


def morph_data() -> MorphData:
    """the hole's rim pushed out and the triangle's tip lifted: sparse indices"""
    base   = _plain_panel()
    target = _plain_panel()
    target.points[[4, 5, 6, 7]] += (0.0, 0.0, 0.2)
    target.points[10]           += (0.0, 0.1, 0.3)
    return MorphData.from_mesh_data(base, target, target_name="panel_bulge")


def morph_list() -> MorphList:
    """the bulge, a fold of the quad, and a target equal to its base"""
    base   = _plain_panel()

    folded = _plain_panel()
    folded.points[[8, 9]] += (-0.25, 0.0, 0.75)

    return MorphList(
        [
            morph_data(),
            MorphData.from_mesh_data(base, folded, target_name="panel_fold"),
            # an unchanged target: no offsets, no indices
            MorphData.from_mesh_data(base, _plain_panel(), target_name="panel_rest"),
        ]
    )


# ---------------------------------- SKINS ----------------------------------- #


def _panel_weights() -> np.ndarray:
    """
    Left hinge falling off across x, right hinge rising, the spine taking the
    rest: rows sum to 1, with zeros where an influence does not reach.
    """
    x     = np.array(_PANEL_POINTS)[:, 0]
    left  = np.clip(1.0 - x / 2.0, 0.0, 1.0)
    right = np.clip((x - 1.0) / 2.0, 0.0, 1.0)
    return np.column_stack([left, right, 1.0 - left - right])


def skin_data() -> SkinData:
    """the panel's skin, with Trinity 3 symmetry patterns"""
    skin = SkinData(
        weights    = _panel_weights(),
        influences = ["b_l_hinge", "b_r_hinge", "b_spine"],
        name       = "panelShape",
    )
    skin.patterns = Patterns.TRINITY3
    return skin


def compact_skin_data() -> CompactSkinData:
    """the panel's skin, namespaced, cut to two influences per vertex"""
    skin = SkinData(
        weights    = _panel_weights(),
        influences = ["rig:b_l_hinge", "rig:b_r_hinge", "rig:b_spine"],
    )
    skin.set_max_influences(2)
    return skin.to_compact_skin_data()


def skin_list() -> SkinList:
    """the panel's skin, a bolt's skin from a namespaced rig, an empty skin"""
    bolt = SkinData(
        weights    = np.array([[1.0, 0.0], [0.75, 0.25], [0.0, 1.0], [0.5, 0.5]]),
        influences = ["rig:b_spine", "rig:b_root"],
        name       = "rig:boltShape",
    )
    empty = SkinData(
        weights    = np.empty((0, 2)),
        influences = ["rig:b_spine", "rig:b_root"],
        name       = "rig:emptyShape",
    )
    return SkinList([skin_data(), bolt, empty])


# ----------------------------------- SDF ------------------------------------ #


def _place_in_scene(node, uuid: str) -> None:
    """what a primitive picks up in a scene: an id, a parent, user attributes"""
    node.uuid         = uuid
    node.parent_node  = "sdf_grp"
    node.rotate_order = 2  # zxy
    node.rotate_axis  = [0.0, 0.0, 10.0]
    node.visibility   = False
    node.add_user_attribute("blend", 0.25)
    node.add_user_attribute("material", "rubber")


def sdf_sphere() -> SDFSphere:
    sphere = SDFSphere(
        radius    = 0.75,
        translate = [0.5, 1.0, -0.25],
        rotate    = [0.0, 45.0, 0.0],
        scale     = [1.0, 1.5, 1.0],
        name      = "sdf:head",
    )
    _place_in_scene(sphere, "6f1c2b0e-3a4d-4e5f-8a9b-0c1d2e3f4a5b")
    return sphere


def sdf_box() -> SDFBox:
    box = SDFBox(
        half_extents = [0.1, 1.0, 0.1],
        translate    = [0.5, 0.0, 0.0],
        rotate       = [30.0, 0.0, 0.0],
        scale        = [2.0, 1.0, 1.0],
        name         = "sdf:fin",
    )
    _place_in_scene(box, "7a2d3c1f-4b5e-4f60-9bac-1d2e3f4a5b6c")
    return box


def sdf_cylinder() -> SDFCylinder:
    cylinder = SDFCylinder(
        radius    = 0.3,
        height    = 3.0,
        axis      = 2,
        translate = [0.0, 0.25, 0.0],
        rotate    = [0.0, 0.0, 90.0],
        scale     = [1.0, 1.0, 0.5],
        name      = "sdf:hole",
    )
    _place_in_scene(cylinder, "8b3e4d20-5c6f-4071-acbd-2e3f4a5b6c7d")
    return cylinder


EXAMPLES = {
    MeshData:        mesh_data,
    UVData:          uv_data,
    MeshList:        mesh_list,
    UVList:          uv_list,
    MapData:         map_data,
    GeomSubsetData:  geom_subset_data,
    MorphData:       morph_data,
    MorphList:       morph_list,
    SkinData:        skin_data,
    CompactSkinData: compact_skin_data,
    SkinList:        skin_list,
    SDFSphere:       sdf_sphere,
    SDFBox:          sdf_box,
    SDFCylinder:     sdf_cylinder,
}
