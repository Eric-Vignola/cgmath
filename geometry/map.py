from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, List, Optional

import numpy as np
from cgmath.geometry._base import Data
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.skin_weights import SkinData
from cgmath.geometry.utils import pxr

GEOM_SUBSET_PRIM_TYPE = "GeomSubset"
VERT_MAP_PRIM_TYPE    = "VertFloatMap"
FACE_MAP_PRIM_TYPE    = "FaceFloatMap"


@dataclass(repr=False, eq=False)
class GeomSubsetData(Data):
    """
    Geometry subset data class.
    """

    name:           str
    indices:        np.ndarray
    component_type: str = "v"  # v, f, e, etc
    category:       Optional[str] = None

    EQUALITY_TEST_IGNORE = ["name"]

    def to_skin_data(
        self, mesh_data: MeshData | None = None, influence: str = "inf"
    ) -> SkinData:
        """Converts this map object to a skin data object.

        Args:
            mesh_data: The mesh data to use for convering faces and edges to vertices.
                Only used if component_type is not "v".
            influence: The influence name to use in the skin data.
        """
        if self.component_type != "v" and not mesh_data:
            raise ValueError("MeshData required for non-vertex GeomSubsetData")

        if self.component_type == "f":
            verts = mesh_data.faces_to_vertices(self.indices)
        elif self.component_type == "e":
            verts = mesh_data.edges_to_vertices(self.indices)
        else:
            verts = self.indices

        weights           = np.zeros((mesh_data.point_count, 1))
        weights[verts, 0] = 1.0

        return SkinData(weights=weights, influences=[influence])

    @classmethod
    def from_skin_data(
        cls, skin_data: SkinData, mesh_data: MeshData | None = None, **kwargs
    ) -> GeomSubsetData:
        """Converts a skin data to a GeomSubsetData object, assuming there is
        only 1 influence.

        Args:
            skin_data: The source skin data.
            mesh_data: The mesh data to use for the conversion.
            kwargs: geomsubset creation kwargs
        """
        if len(skin_data.influences) != 1:
            raise ValueError("SkinData must have only 1 influence.")

        component_type           = kwargs.pop("component_type", "v")
        kwargs["component_type"] = component_type

        weights = skin_data.weights.flatten()
        indices = np.nonzero(weights)[0]
        if component_type != "v" and not mesh_data:
            raise ValueError("MeshData missing")
        elif component_type == "f":
            indices = mesh_data.vertices_to_faces(indices)
        elif component_type == "e":
            indices = mesh_data.vertices_to_edges(indices)
        elif component_type != "v":
            raise ValueError(f"Invalid component type: {component_type}")
        kwargs["indices"] = indices

        return cls(**kwargs)

    # --- USD interface

    @classmethod
    def from_prim(cls, prim: Any) -> "GeomSubsetData":
        """Constructs a data object from a prim."""
        set_api = pxr().UsdGeom.Subset(prim)
        return cls(
            component_type = set_api.GetElementTypeAttr().Get()[0].lower(),
            indices        = np.array(set_api.GetIndicesAttr().Get()),
            name           = prim.GetName(),
            category       = set_api.GetElementTypeAttr().Get() or None,
        )

    def to_prim(self, prim: Any) -> None:
        """Streams data into a prim."""
        if not prim.GetTypeName():
            prim.SetTypeName(GEOM_SUBSET_PRIM_TYPE)
        if self.component_type == "v":
            element_type = "vertex"
        elif self.component_type == "f":
            element_type = "face"
        else:
            element_type = "edge"

        set_api = pxr().UsdGeom.Subset(prim)
        set_api.CreateElementTypeAttr().Set(element_type)
        if self.category:
            set_api.CreateFamilyNameAttr().Set(self.category)
        set_api.CreateIndicesAttr().Set(self.indices)

    def split_to_array_by_clusters(
        self, mesh_data: MeshData | None = None
    ) -> list[GeomSubsetData]:
        """Splits this GeomSubsetData object into a list of GeomSubsetData objects based on local vert clusters."""
        clusters = mesh_data.get_edge_vertex_clusters(self.indices)

        island_maps = []
        for i, island in enumerate(clusters):
            island_maps.append(
                GeomSubsetData(
                    name           = "{name}_island_{i}".format(name=self.name, i=i),
                    indices        = island,
                    component_type = self.component_type,
                )
            )

        return island_maps


@dataclass(repr=False, eq=False)
class MapData(Data):
    """
    Maps/Painted Maps general data class.
    """

    name:           str
    indices:        np.ndarray
    values:         np.ndarray
    default_value:  Number = 0
    component_type: str = "v"  # v, f, e, etc
    categories:     Optional[List[str]] = None

    EQUALITY_TEST_IGNORE = ["name"]

    def to_dense_array(self, component_count: int) -> np.ndarray:
        """Convert this object to a dense value array."""
        array               = np.full((component_count,), self.default_value)
        array[self.indices] = self.values
        return array

    def to_skin_data(self, mesh_data: MeshData, influence: str = "inf") -> SkinData:
        """Converts this map object to a skin data object.

        Args:
            mesh_data: The mesh data to use in the conversion.
            influence: The influence name to use in the skin data.
        """
        if self.component_type != "v":
            raise NotImplementedError(
                "Converting non-vertex MapData to SkinData is not supported yet."
            )
        weights                  = np.full((mesh_data.point_count, 1), self.default_value)
        weights[self.indices, 0] = self.values
        return SkinData(weights=weights, influences=[influence])

    @classmethod
    def from_skin_data(
        cls, skin_data: SkinData, mesh_data: MeshData | None = None, **kwargs
    ) -> MapData:
        """Converts this skin data to a MapData object, assuming there is
        only 1 influence.

        Args:
            skin_data: The source skin data.
            kwargs: map data creation kwargs
        """
        if len(skin_data.influences) != 1:
            raise ValueError("SkinData must have only 1 influence.")

        component_type = kwargs.pop("component_type", "v")
        if component_type != "v":
            raise ValueError("Can't convert SkinData non-vertex MapData.")
        kwargs["component_type"] = component_type
        weights                  = skin_data.weights.flatten()
        kwargs["indices"]        = np.nonzero(weights)[0]
        kwargs["values"]         = weights[kwargs["indices"]]
        kwargs["default_value"]  = 0.0
        return MapData(**kwargs)

    # --- USD interface

    @classmethod
    def from_prim(cls, prim: Any) -> "GeomSubsetData":
        """Constructs a data object from a prim."""
        prim_type     = prim.GetTypeName()
        namespace     = prim_type[0].lower() + prim_type[1:]
        default_value = prim.GetAttribute(f"{namespace}:default_value").Get()
        return cls(
            name          = prim.GetName(),
            indices       = np.array(prim.GetAttribute(f"{namespace}:indices").Get()),
            values        = np.array(prim.GetAttribute(f"{namespace}:values").Get()),
            default_value = 0.0 if default_value is None else default_value,
        )

    def to_prim(self, prim: Any) -> None:
        """Streams data into a prim."""
        prim_type = (
            VERT_MAP_PRIM_TYPE if self.component_type == "v" else FACE_MAP_PRIM_TYPE
        )
        if not prim.GetTypeName():
            prim.SetTypeName(prim_type)
        namespace = prim_type[0].lower() + prim_type[1:]

        Sdf  = pxr().Sdf
        args = (False, Sdf.VariabilityVarying)
        attr = prim.CreateAttribute(
            f"{namespace}:indices", Sdf.ValueTypeNames.IntArray, *args
        )
        attr.Set(self.indices)
        attr = prim.CreateAttribute(
            f"{namespace}:values", Sdf.ValueTypeNames.FloatArray, *args
        )
        attr.Set(self.values)
        attr = prim.CreateAttribute(
            f"{namespace}:defaultValue", Sdf.ValueTypeNames.Float, *args
        )
        attr.Set(self.default_value)