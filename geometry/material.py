"""A material: Maya's lambert and phong attributes, on the faces of one mesh.

A :class:`MaterialData` holds what Maya's lambert and phong shaders hold: the
colour, transparency, ambient colour, incandescence and diffuse of a lambert,
and a phong's specular colour, cosine power and reflectivity; the colour map's
pixels, where a file texture feeds the colour, with the UV set it reads and
its wrap modes; the map a bump2d feeds the normal from, tangent space normals
or a bump map, with its depth; and the faces of one mesh it covers, the mesh named by
``mesh_uuid`` as a skin names its. The defaults are a fresh Maya lambert's
and phong's (measured in Maya 2025).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from cgmath.data import Array, Data, DataList, ImmutableArray as numpy_array
from cgmath.geometry._one_mesh import OneMeshList
from cgmath.hierarchy.hierarchy import _is_uuid, generate_uuid

# the kinds a MaterialData holds: Maya's lambert and phong. Maya's own FBX
# export writes a blinn as a phong and a standardSurface as a bare material
MATERIAL_KINDS = ("lambert", "phong")

# how a texture repeats past its edges, Maya's place2dTexture wrapU / wrapV
WRAP_MODES = ("repeat", "clamp")

# how Maya's bump2d reads a material's normal map (its bumpInterp): tangent
# space normals, or a bump map
NORMAL_MAP_KINDS = ("tangent", "bump")


@dataclass(repr=False, eq=False)
class MaterialData(Data):
    """
    A material, as Maya's lambert and phong shaders hold one, assigned to
    the faces of one mesh.

    Parameters
    ----------
    name : str, optional
        The shader's name, None for none.
    mesh_uuid : str, optional
        The uuid of the mesh it covers, None for none yet.
    uuid : str, optional
        The material's own identity, made when none is given. The records
        of one material on several meshes share it: a shader on two Maya
        meshes reads as two records, one a mesh, with one uuid, and
        ``save_fbx`` writes one shader per uuid, so records sharing one
        agree but for their faces. A copy keeps it.
    faces : (N,) int32, optional
        The faces of that mesh it covers, None for the whole mesh.
    kind : str
        "lambert" or "phong". Default "lambert".
    color : (3,) float64
        Maya's color, RGB in 0 to 1. Default 0.5 grey. Where a texture feeds
        the colour the texture is the colour; Maya's own FBX export writes
        such a colour white.
    transparency : (3,) float64
        Maya's transparency, 0 opaque. Default 0.
    ambient_color : (3,) float64
        Maya's ambientColor. Default 0.
    incandescence : (3,) float64
        Maya's incandescence. Default 0.
    diffuse : float
        Maya's diffuse, which scales the colour. Default 0.8.
    specular_color : (3,) float64
        A phong's specularColor. Default 0.5 grey.
    cosine_power : float
        A phong's cosinePower, the highlight's tightness. Default 20.
    reflectivity : float
        A phong's reflectivity. Default 0.5.
    texture : (H, W, C) uint8, uint16 or float32, optional
        The colour map's pixels, rows top down, C channels (1 grey, 3 RGB, 4
        RGBA), the dtype kept as given. None for a plain colour.
    texture_file : str, optional
        The file name the pixels came from, or go to: a .fbx embeds the map
        under it. None for none.
    uv_set : str, optional
        The name of the mesh's UV set the texture reads, None for the
        mesh's first.
    wrap_u, wrap_v : str
        "repeat" or "clamp": how the texture continues past its edges
        (Maya's wrapU and wrapV). Default "repeat". FBX holds them as the
        texture's wrap modes; Maya's own export writes both as repeat and
        its import leaves both on, whatever a file says (measured).
    normal_map : (H, W, C) uint8, uint16 or float32, optional
        The pixels of the map a bump2d feeds the shader's normal from, rows
        top down, as ``texture``; None for none. It reads the colour map's
        UV set and wraps. Carried through a scene and a file; the renderer
        does not shade by it.
    normal_map_file : str, optional
        The file name the pixels came from, or go to. None for none.
    normal_map_kind : str
        How Maya's bump2d reads the map (its bumpInterp): "tangent" for
        tangent space normals, FBX's NormalMap slot, or "bump" for a bump
        map, FBX's Bump slot, which is also what Maya exports object space
        normals as (measured). Default "tangent". A file holding both slots
        reads as the bump map, as Maya's importer takes it (measured).
    bump_depth : float
        Maya's bumpDepth, FBX's BumpFactor. Default 1.
    """

    EQUALITY_TEST_IGNORE = ["name", "mesh_uuid", "uuid"]
    NAMESPACED_FIELDS    = ("name",)

    name:            Optional[str] = None
    mesh_uuid:       Optional[str] = None
    uuid:            Optional[str] = field(default=None, kw_only=True)
    faces:           Optional[Array(np.int32, "N")] = None
    kind:            str = "lambert"
    color:           Array(np.float64, 3) = numpy_array([0.5, 0.5, 0.5])
    transparency:    Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    ambient_color:   Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    incandescence:   Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    diffuse:         float = 0.8
    specular_color:  Array(np.float64, 3) = numpy_array([0.5, 0.5, 0.5])
    cosine_power:    float = 20.0
    reflectivity:    float = 0.5
    texture:         Optional[Array((np.uint8, np.uint16, np.float32), "H", "W", "C")] = None
    texture_file:    Optional[str] = None
    uv_set:          Optional[str] = None
    wrap_u:          str = "repeat"
    wrap_v:          str = "repeat"
    normal_map:      Optional[Array((np.uint8, np.uint16, np.float32), "H", "W", "C")] = None
    normal_map_file: Optional[str] = None
    normal_map_kind: str = "tangent"
    bump_depth:      float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.uuid is None:
            self.uuid = generate_uuid()

    def __setattr__(self, name: str, value) -> None:
        if name == "kind" and value not in MATERIAL_KINDS:
            raise ValueError(f"MaterialData.kind is one of {list(MATERIAL_KINDS)}, not {value!r}")
        if name in ("wrap_u", "wrap_v") and value not in WRAP_MODES:
            raise ValueError(f"MaterialData.{name} is one of {list(WRAP_MODES)}, not {value!r}")
        if name == "normal_map_kind" and value not in NORMAL_MAP_KINDS:
            raise ValueError(f"MaterialData.normal_map_kind is one of {list(NORMAL_MAP_KINDS)}, not {value!r}")
        if name == "uuid" and value is not None:
            value = str(value)  # a uuid.UUID as its text
            if not _is_uuid(value):
                raise ValueError(f"MaterialData.uuid takes a uuid or None, not {value!r}")
        super().__setattr__(name, value)

    def _post_load(self) -> None:
        super()._post_load()
        if self.uuid is None:
            self.uuid = generate_uuid()  # a file saved before materials had one

    @property
    def textured(self) -> bool:
        """whether a texture feeds the colour"""
        return self.texture is not None

    def covers(self, mesh) -> np.ndarray:
        """
        (F,) bool, one a face of ``mesh``: the faces this material covers,
        every face when ``faces`` is None. ValueError for a face the mesh
        does not hold.
        """
        count = len(mesh.counts)
        if self.faces is None:
            return np.ones(count, dtype=bool)
        faces = np.asarray(self.faces, dtype=np.int64)
        if faces.size and (faces.min() < 0 or faces.max() >= count):
            raise ValueError(f"MaterialData.faces name faces the mesh of {count} faces does not hold")
        mask        = np.zeros(count, dtype=bool)
        mask[faces] = True
        return mask

    # --- the faces follow the mesh's edits ---------------------------------
    #
    # A mesh edit that renumbers faces is called on each record of the mesh,
    # as it is on its UV sets and its skin: the same call, with the same
    # arguments. A material covering its whole mesh (``faces`` None) still
    # does, whatever the edit.

    def keep_borders(self, rules) -> np.ndarray:
        """
        ``rules``, the pairs of triangles :meth:`MeshData.get_quadrangulate_rules`
        or :meth:`UVData.get_quadrangulate_rules` would merge, without the
        pairs that would merge a triangle this material covers with one it
        does not: a quad carries one material. Pass the mesh's rules through
        each of its materials, then quadrangulate the mesh, its UV sets and
        its materials with what is left.
        """
        rules = np.asarray(rules, dtype=np.int64).reshape(-1, 4)
        if self.faces is None or not rules.size:
            return rules
        held = np.isin(rules[:, :2], np.asarray(self.faces, dtype=np.int64))
        return rules[held[:, 0] == held[:, 1]]

    def quadrangulate(self, rules) -> None:
        """
        The faces after ``mesh.quadrangulate(rules)``, in place: the two
        triangles of each pair become one quad, and every later face moves
        down. ValueError for rules that merge a triangle of this material
        with one of another (:meth:`keep_borders` leaves those pairs out).
        """
        rules = np.asarray(rules, dtype=np.int64).reshape(-1, 4)
        if self.faces is None or not rules.size:
            return
        faces = np.asarray(self.faces, dtype=np.int64)
        held  = np.isin(rules[:, :2], faces)
        split = int(np.count_nonzero(held[:, 0] != held[:, 1]))
        if split:
            raise ValueError(
                f"MaterialData.quadrangulate: {split} of the rules' quads would merge a triangle of material "
                f"{self.name!r} with one of another material; pass the rules through keep_borders() of each of the "
                f"mesh's materials first"
            )
        # the later triangle of a pair joins the earlier one, where the quad is made
        first = rules[:, :2].min(axis=1)
        later = rules[:, :2].max(axis=1)
        order = np.argsort(later)
        first, later = first[order], later[order]
        place      = np.searchsorted(later, faces)
        merged     = (place < later.size) & (later[np.minimum(place, later.size - 1)] == faces)
        faces      = np.where(merged, first[np.minimum(place, later.size - 1)], faces)
        self.faces = np.unique(faces - np.searchsorted(later, faces)).astype(np.int32)

    def triangulate(self, mesh) -> None:
        """
        The faces after ``mesh.triangulate(...)`` in its default order
        (``preserve_order``), in place: each face of n corners becomes n - 2
        triangles, where it was. ``mesh`` is the mesh before the edit, so
        call this first, as ``skin.subdivide(mesh)`` is called before
        ``mesh.subdivide()``.
        """
        self._spread(np.asarray(mesh.counts, dtype=np.int64) - 2)

    def subdivide(self, mesh, steps: int = 1) -> None:
        """
        The faces after ``mesh.subdivide(steps)``, in place: each face of n
        corners becomes n quads, where it was, and each of those four more
        a further step. ``mesh`` is the mesh before the edit, so call this
        first, as ``skin.subdivide(mesh)`` is called before
        ``mesh.subdivide()``.
        """
        if int(steps) < 1:
            return
        counts = np.asarray(mesh.counts, dtype=np.int64)
        self._spread(counts * 4 ** (int(steps) - 1))

    def _spread(self, pieces: np.ndarray) -> None:
        """each face replaced by ``pieces[face]`` faces, where it was"""
        if self.faces is None:
            return
        faces = np.asarray(self.faces, dtype=np.int64)
        if faces.size and (faces.min() < 0 or faces.max() >= pieces.size):
            raise ValueError(f"MaterialData.faces name faces the mesh of {pieces.size} faces does not hold")
        start      = np.concatenate([[0], np.cumsum(pieces)])
        spans      = [np.arange(start[face], start[face + 1]) for face in np.sort(faces).tolist()]
        self.faces = (np.concatenate(spans) if spans else np.zeros(0)).astype(np.int32)

    def delete_faces(self, faces) -> None:
        """
        The faces after ``mesh.delete_faces(faces)``, in place: the deleted
        ones leave, every later face moves down. A material left covering
        no face holds an empty ``faces``: take it off its list.
        """
        if self.faces is None:
            return
        gone       = np.unique(np.asarray(faces, dtype=np.int64))
        held       = np.asarray(self.faces, dtype=np.int64)
        held       = held[~np.isin(held, gone)]
        self.faces = np.unique(held - np.searchsorted(gone, held)).astype(np.int32)

    def from_faces(self, faces, mesh_uuid: Optional[str] = None) -> Optional["MaterialData"]:
        """
        This material on the mesh ``mesh.from_faces(faces)`` returns: a new
        record of the same material (the same ``uuid``) covering, of that
        part, the faces this one covers of the whole; None where it covers
        none of them. ``mesh_uuid`` names the part: pass its ``uuid``.
        """
        kept = np.unique(np.asarray(faces, dtype=np.int64))
        part = self.copy()
        if self.faces is not None:
            held = np.asarray(self.faces, dtype=np.int64)
            held = held[np.isin(held, kept)]
            if not held.size:
                return None
            part.faces = np.searchsorted(kept, np.unique(held)).astype(np.int32)
        part.mesh_uuid = mesh_uuid
        return part

    def from_vertices(
        self, mesh, vertices, contained: bool = True, exclude: bool = False, mesh_uuid: Optional[str] = None
    ) -> Optional["MaterialData"]:
        """
        This material on the mesh ``mesh.from_vertices(vertices, contained,
        exclude)`` returns, as :meth:`from_faces` gives it for the faces
        that call keeps: the faces holding (or, with ``contained`` False,
        touching) the vertices, or with ``exclude`` the others; for subset
        rules (a 2d array of :meth:`MeshData.get_subset_rules`), every face
        the rules keep any corner of.
        """
        vertices = np.asarray(vertices)
        if vertices.ndim > 1:
            if not contained or exclude:
                raise ValueError(
                    "contained and exclude do not apply to subset rules, which already describe exactly what is kept"
                )
            faces = np.flatnonzero((vertices >= 0).any(axis=1))
        else:
            faces = np.asarray(mesh.contains_vertices(np.unique(vertices), contained=contained), dtype=np.int64)
            if exclude:
                faces = np.setdiff1d(np.arange(len(mesh.counts)), faces)
        return self.from_faces(faces, mesh_uuid=mesh_uuid)

    def union(self, mesh, *others) -> None:
        """
        This record after ``mesh.union(*others)``, in place: a record of
        ``mesh`` keeps its faces, listed where it covered the whole mesh,
        which it no longer does; a record of one of ``others`` moves onto
        ``mesh``, its faces after those of the meshes merged ahead of it.
        The meshes are the ones before the edit, so call this first. A
        record that names none of them is left as it is.
        """
        if not others:
            return
        start = 0
        for held in (mesh,) + tuple(others):
            count = len(held.counts)
            if self.mesh_uuid == held.uuid:
                faces          = np.arange(count) if self.faces is None else np.asarray(self.faces, dtype=np.int64)
                self.faces     = (faces + start).astype(np.int32)
                self.mesh_uuid = mesh.uuid
                return
            start += count


class MaterialList(OneMeshList, DataList[MaterialData]):
    """A list of :class:`MaterialData`, as a file holds its materials, or
    one mesh's alone (``mesh_uuid``, :class:`~cgmath.geometry._one_mesh.OneMeshList`).

    A mesh edit that renumbers faces is called on the list with the mesh
    first, as the mesh was before the edit: it reaches the records that name
    that mesh and leaves the others alone, so the list a file holds for all
    its meshes takes the call as it is.
    """

    def of(self, mesh) -> list:
        """the records that name ``mesh``, in the list's order"""
        return [material for material in self.list if material.mesh_uuid == mesh.uuid]

    def keep_borders(self, mesh, rules) -> np.ndarray:
        """``rules`` without the pairs that would merge two triangles of
        different materials of ``mesh`` (:meth:`MaterialData.keep_borders`)"""
        for material in self.of(mesh):
            rules = material.keep_borders(rules)
        return np.asarray(rules, dtype=np.int64).reshape(-1, 4)

    def quadrangulate(self, mesh, rules) -> None:
        """the faces of ``mesh``'s materials after ``mesh.quadrangulate(rules)``
        (:meth:`MaterialData.quadrangulate`)"""
        for material in self.of(mesh):
            material.quadrangulate(rules)

    def triangulate(self, mesh) -> None:
        """the faces of ``mesh``'s materials after ``mesh.triangulate(...)``;
        call it before the mesh's (:meth:`MaterialData.triangulate`)"""
        for material in self.of(mesh):
            material.triangulate(mesh)

    def subdivide(self, mesh, steps: int = 1) -> None:
        """the faces of ``mesh``'s materials after ``mesh.subdivide(steps)``;
        call it before the mesh's (:meth:`MaterialData.subdivide`)"""
        for material in self.of(mesh):
            material.subdivide(mesh, steps)

    def delete_faces(self, mesh, faces) -> None:
        """the faces of ``mesh``'s materials after ``mesh.delete_faces(faces)``;
        a material left covering no face is taken off the list"""
        for material in self.of(mesh):
            material.delete_faces(faces)
        self.list[:] = [
            material for material in self.list
            if material.mesh_uuid != mesh.uuid or material.faces is None or len(material.faces)
        ]

    def from_faces(self, mesh, faces, part=None) -> "MaterialList":
        """the materials of ``mesh.from_faces(faces)``: a new list of new
        records, one for each material of ``mesh`` covering any of those
        faces, each naming ``part``, the mesh that call returned"""
        uuid  = None if part is None else part.uuid
        found = [material.from_faces(faces, mesh_uuid=uuid) for material in self.of(mesh)]
        return MaterialList([material for material in found if material is not None])

    def from_vertices(self, mesh, vertices, part=None, contained: bool = True, exclude: bool = False) -> "MaterialList":
        """the materials of ``mesh.from_vertices(vertices, contained,
        exclude)``: a new list of new records, each naming ``part``, the
        mesh that call returned (:meth:`MaterialData.from_vertices`)"""
        uuid  = None if part is None else part.uuid
        found = [material.from_vertices(mesh, vertices, contained, exclude, mesh_uuid=uuid) for material in self.of(mesh)]
        return MaterialList([material for material in found if material is not None])

    def union(self, mesh, *others) -> None:
        """the materials after ``mesh.union(*others)``; call it before the
        mesh's. The records of ``others`` move onto ``mesh``, their faces
        after its own (:meth:`MaterialData.union`); the records of one
        material (one uuid) become one record covering the faces of all,
        the first of them in the list"""
        merged = {held.uuid for held in (mesh,) + tuple(others)}
        first  = {}
        kept   = []
        for material in self.list:
            if material.mesh_uuid not in merged:
                kept.append(material)
                continue
            material.union(mesh, *others)
            held = first.get(material.uuid)
            if held is None or material.faces is None or held.faces is None:
                first.setdefault(material.uuid, material)
                kept.append(material)
                continue
            held.faces = np.unique(np.concatenate([held.faces, material.faces])).astype(np.int32)
        self.list[:] = kept
