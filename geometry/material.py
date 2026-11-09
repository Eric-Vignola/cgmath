"""A material: Maya's lambert and phong attributes, on the faces of one mesh.

A :class:`MaterialData` holds what Maya's lambert and phong shaders hold: the
colour, transparency, ambient colour, incandescence and diffuse of a lambert,
and a phong's specular colour, cosine power and reflectivity; the colour map's
pixels, where a file texture feeds the colour, with the UV set it reads and
its wrap modes; and the faces of one mesh it covers, the mesh named by
``mesh_uuid`` as a skin names its. The defaults are a fresh Maya lambert's
and phong's (measured in Maya 2025).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.data import Array, Data, DataList, ImmutableArray as numpy_array

# the kinds a MaterialData holds: Maya's lambert and phong. Maya's own FBX
# export writes a blinn as a phong and a standardSurface as a bare material
MATERIAL_KINDS = ("lambert", "phong")

# how a texture repeats past its edges, Maya's place2dTexture wrapU / wrapV
WRAP_MODES = ("repeat", "clamp")


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
    """

    EQUALITY_TEST_IGNORE = ["name", "mesh_uuid"]
    NAMESPACED_FIELDS    = ("name",)

    name:           Optional[str] = None
    mesh_uuid:      Optional[str] = None
    faces:          Optional[Array(np.int32, "N")] = None
    kind:           str = "lambert"
    color:          Array(np.float64, 3) = numpy_array([0.5, 0.5, 0.5])
    transparency:   Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    ambient_color:  Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    incandescence:  Array(np.float64, 3) = numpy_array([0.0, 0.0, 0.0])
    diffuse:        float = 0.8
    specular_color: Array(np.float64, 3) = numpy_array([0.5, 0.5, 0.5])
    cosine_power:   float = 20.0
    reflectivity:   float = 0.5
    texture:        Optional[Array((np.uint8, np.uint16, np.float32), "H", "W", "C")] = None
    texture_file:   Optional[str] = None
    uv_set:         Optional[str] = None
    wrap_u:         str = "repeat"
    wrap_v:         str = "repeat"

    def __setattr__(self, name: str, value) -> None:
        if name == "kind" and value not in MATERIAL_KINDS:
            raise ValueError(f"MaterialData.kind is one of {list(MATERIAL_KINDS)}, not {value!r}")
        if name in ("wrap_u", "wrap_v") and value not in WRAP_MODES:
            raise ValueError(f"MaterialData.{name} is one of {list(WRAP_MODES)}, not {value!r}")
        super().__setattr__(name, value)

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


class MaterialList(DataList[MaterialData]):
    """A list of :class:`MaterialData`, as a file holds its materials."""
