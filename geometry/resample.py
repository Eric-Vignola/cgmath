"""
A resampler class interfacing mesh-dependent data types (morph targets and skin weights).

Usage example:
```python
from cgmath.geometry.resample import MeshDataResampler, ResampleMode
from cgmath.geometry import MeshData, MorphData

# stream data from USD prims (this can be from elsewhere)
src_mesh_data = MeshData.from_prim(src_mesh_prim)
src_uv_data = MeshData.from_prim(src_mesh_prim)[0]
dst_mesh_data = MeshData.from_prim(dst_mesh_prim)
dst_uv_data = MeshData.from_prim(dst_mesh_prim)[0]

# construct a resampler
resampler = MeshDataResampler(
    src_mesh_data, dst_mesh_data, src_uv=src_uv_data, dst_uv=dst_uv_data
)

# resample morph targets in UV space
src_target_data = MorphData.from_prim(src_target_prim)
dst_target_data = resampler.resample_morph_target(src_target_data, mode=ResampleMode.UV)
dst_target_data.to_prim(dst_target_prim)

# resample skin weights in UV space
src_skin_data = MorphData.from_prim(src_skin_prim)
dst_skin_data = resampler.resample_skin_weights(src_skin_data, mode=ResampleMode.UV)
dst_skin_data.to_prim(dst_skin_prim)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from cgmath.geometry._saddle_surface import SampleData
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import MeshData, SampleMethod, UVData
from cgmath.geometry.morph_target import MorphData
from cgmath.geometry.robust_skinweights_transfer_bilinear import (
    robust_skinweights_transfer_bilinear,
    RobustBilinearSkinTransferOptions,
)
from cgmath.geometry.skin_weights import CompactSkinData, SkinData
from cgmath.geometry.utils.main import (
    bezier_evaluate,
    bezier_vectors,
    bilinear_vectors,
)


@dataclass
class SpatialSkinTransferOptions:
    max_influences: int = 4


@dataclass
class UVSkinTransferOptions:
    max_influences: int = 4


class ResampleMode(Enum):
    """
    Data resample modes.
    """

    # spatial resample (closest point)
    SPATIAL = SpatialSkinTransferOptions
    # UV space resample
    UV = UVSkinTransferOptions
    # resample using the robust transfer algorithm on bilinear patches
    ROBUST_BILINEAR = RobustBilinearSkinTransferOptions


class MeshDataResampler:
    """
    Resampler/transfer class for mesh-dependent data types.

    Supported types:
        * Morph Target
        * Skin Weights
        * Mesh
        * Geom Subset
        * Mesh Map
    """

    def __init__(
        self,
        src_mesh: MeshData,
        dst_mesh: MeshData,
        src_uv:   UVData   | None = None,
        dst_uv:   UVData   | None = None,
    ) -> None:
        """Initialize internal vars.

        Args:
            src_mesh: Source mesh data object.
            dst_mesh: Destination mesh data object.
            src_uv: Source UV data object, only required in uv resample mode.
            dst_uv: Destination UV data object, only required in uv resample mode.
        """
        # shared vars
        self._sample_cache: dict[tuple, SampleData] = {}
        self._src_mesh = src_mesh
        self._dst_mesh = dst_mesh
        self._src_uv   = src_uv
        self._dst_uv   = dst_uv

        # morph target resample vars
        self._buffer = np.zeros(self.src_mesh.points.shape)
        self._arange = np.arange(self.dst_mesh.point_count)

    # --- properties

    @property
    def spatial_sample_data(self) -> SampleData:
        """Cache and return a sample data object in world space (bilinear)."""
        return self.get_sample_data(ResampleMode.SPATIAL)

    @property
    def uv_sample_data(self) -> SampleData:
        """Cache and return a sample data object in UV space."""
        return self.get_sample_data(ResampleMode.UV)

    @property
    def src_mesh(self) -> MeshData:
        """Returns the source mesh data."""
        return self._src_mesh

    @property
    def dst_mesh(self) -> MeshData:
        """Returns the destination mesh data."""
        return self._dst_mesh

    @property
    def src_uv(self) -> UVData:
        """Returns the source uv data."""
        return self._src_uv

    @property
    def dst_uv(self) -> UVData:
        """Returns the destination uv data."""
        return self._dst_uv

    # --- helpers

    def get_sample_data(
        self,
        mode:   ResampleMode,
        method: SampleMethod | str = SampleMethod.BILINEAR,
    ) -> SampleData:
        """Returns a cached sample data object based on mode and method."""
        method    = SampleMethod(method)
        cache_key = (mode, method)
        if cache_key not in self._sample_cache:
            if mode == ResampleMode.SPATIAL:
                self._sample_cache[cache_key] = self.src_mesh.sample(
                    self.dst_mesh, method=method
                )
            elif mode == ResampleMode.UV:
                if not self.src_uv or not self.dst_uv:
                    raise RuntimeError("UV data not supplied.")
                resample_uv = self.src_uv.sample(self.dst_uv)
                self._sample_cache[cache_key] = resample_uv.remap(
                    self.src_mesh, self.dst_mesh, self.dst_uv
                )
            else:
                raise ValueError(f"Unsupported resample mode: {mode}")
        return self._sample_cache[cache_key]

    # --- resample methods

    def resample_mesh(
        self,
        mesh_data:       MeshData     | None = None,
        mode:            ResampleMode        = ResampleMode.SPATIAL,
        method:          SampleMethod | str  = SampleMethod.BILINEAR,
        maintain_offset: bool                = False,
        orient_offset:   bool                = True,
    ):
        """Resamples mesh_data

        Args:
            mesh_data: Optional mesh data whose points are resampled onto the
                destination topology.  When *None*, ``self.src_mesh`` is used.
            mode: Data resample mode.
            method: Surface fitting method -- BILINEAR (flat patches) or BEZIER
                (PN Quad bicubic Bezier patches for higher accuracy on curved
                surfaces).  Affects both the closest-point projection and the
                offset-normal computation when *orient_offset* is True.
            maintain_offset: If True, preserve the signed distance between each
                destination vertex and its projection on the source surface
                along the surface normal.  This keeps the destination mesh at
                its original standoff distance after resampling.
            orient_offset: If True (and ``maintain_offset`` is also True),
                recompute the surface normal from *mesh_data* at each sample
                location and use it as the offset direction.  When False, the
                original source-mesh normal stored in *sample_data* is used.

        Returns:
            The resampled data object.
        """
        method = SampleMethod(method)

        if maintain_offset and mode == ResampleMode.UV:
            # UV sample data carries 2-D projections and all-zero normals, so
            # the offset arithmetic below broadcasts (n,3) against (n,2) and
            # dies inside numpy. Even with matching shapes the zero normals
            # would make every offset 0, so there is nothing to fall back to.
            raise ValueError(
                "maintain_offset is not supported in UV mode: UV sample data "
                "has 2-D projections and no surface normals to offset along. "
                "Use ResampleMode.SPATIAL, or set maintain_offset=False."
            )

        resampled_mesh = self.dst_mesh.copy()
        sample_data    = self.get_sample_data(mode, method)

        source         = mesh_data if mesh_data is not None else self.src_mesh
        source_normals = None

        if method == SampleMethod.BEZIER:
            source_normals = source.get_vertex_normals(
                angle_weighted=True, area_weighted=False
            )
            resampled_mesh.points = bezier_evaluate(
                source.points,
                source_normals,
                sample_data.geometry,
                sample_data.uvs,
            )
        else:
            resampled_mesh.points = sample_data(source.points)

        if maintain_offset:
            offsets = np.einsum(
                "ij,ij->i",
                self.dst_mesh.points - sample_data.projections,
                sample_data.normals,
            )

            if orient_offset:
                if method == SampleMethod.BEZIER:
                    U, V = bezier_vectors(
                        source.points,
                        source_normals,
                        sample_data.geometry,
                        sample_data.uvs,
                    )
                else:
                    U, V = bilinear_vectors(
                        source.points, sample_data.geometry, sample_data.uvs
                    )

                normals = np.cross(U, V)
                norms   = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normals = normals / norms
            else:
                normals = sample_data.normals

            resampled_mesh.points += offsets[:, np.newaxis] * normals

        return resampled_mesh

    def resample_morph_target(
        self,
        target_data:   MorphData    | MeshData,
        mode:          ResampleMode            = ResampleMode.SPATIAL,
        method:        SampleMethod | str      = SampleMethod.BILINEAR,
        tolerance:     float        | None     = None,
        use_neighbors: bool                    = True,
    ) -> MorphData | MeshData:
        """Resamples a morph target data.

        Args:
            target_data: A morph target or mesh data object to resample.
                If mesh data is used, we assume it has the same topology
                as the source mesh, and will also return a mesh data object.
            mode: Data resample mode.
            method: Surface fitting method -- BILINEAR or BEZIER.
            tolerance: Tolerance for pruning offsets.
            use_neighbors: If True, use neighboring vertices to prune offsets.

        Returns:
            The resampled data object.
        """
        method = SampleMethod(method)
        if mode == ResampleMode.ROBUST_BILINEAR:
            raise NotImplementedError(
                "Robust resampling for morph targets is not implemented."
            )

        is_mesh = isinstance(target_data, MeshData)
        if is_mesh:
            target_data = MorphData.from_mesh_data(self.src_mesh, target_data)

        # set morph target data in a zero buffer
        self._buffer[:] = 0
        self._buffer[target_data.indices] = target_data.offsets

        # resample the offsets
        out_data    = target_data.copy()
        sample_data = self.get_sample_data(mode, method)
        out_data.offsets = sample_data(self._buffer)
        out_data.indices = self._arange

        neighbors = self.dst_mesh.get_edge_vertex_neighbors() if use_neighbors else None
        out_data.prune_offsets(tolerance=tolerance, neighbors=neighbors)

        if is_mesh:
            out_mesh = self.dst_mesh.copy()
            out_mesh.apply_morph_target(out_data)
            return out_mesh
        return out_data

    def resample_skin_weights(
        self,
        skin_data: SkinData     | CompactSkinData,
        mode:      ResampleMode                   = ResampleMode.SPATIAL,
        method:    SampleMethod | str             = SampleMethod.BILINEAR,
        options:   Any                            = None,
    ) -> SkinData | CompactSkinData:
        """Resamples a skin data.

        Args:
            skin_data: A skin data object to resample.
            mode: Data resample mode.
            method: Surface fitting method -- BILINEAR or BEZIER.
            options: Transfer options.

        Returns:
            The resampled data object.
        """
        method     = SampleMethod(method)
        is_compact = isinstance(skin_data, CompactSkinData)
        if is_compact:
            skin_data = skin_data.to_skin_data()

        # ensure source skin are normalized
        skin_data.normalize()

        dst_skin = skin_data.copy()
        if mode == ResampleMode.ROBUST_BILINEAR:
            dst_skin.weights = robust_skinweights_transfer_bilinear(
                self._src_mesh,
                skin_data,
                self._dst_mesh,
                options,
                sample_data=self.get_sample_data(ResampleMode.SPATIAL, method),
            )
        else:
            sample_data = self.get_sample_data(mode, method)
            dst_skin.weights = sample_data(skin_data.weights)

        # max infs
        if options:
            dst_skin.set_max_influences(options.max_influences)

        if is_compact:
            return dst_skin.to_compact_skin_data()
        return dst_skin

    def resample_map(
        self,
        map_data: MapData      | GeomSubsetData,
        method:   SampleMethod | str            = SampleMethod.BILINEAR,
        **kwargs,
    ) -> MapData | GeomSubsetData:
        """Resamples a map data.

        Args:
            map_data: A map data object to resample.
            method: Surface fitting method -- BILINEAR or BEZIER.
            kwargs: kwargs supported by resample_skin_weights().

        Returns:
            The resampled data object.
        """
        method = SampleMethod(method)
        name, typ = map_data.name, map_data.component_type
        skin_data     = map_data.to_skin_data(self._src_mesh)
        new_skin_data = self.resample_skin_weights(skin_data, method=method, **kwargs)
        if isinstance(map_data, MapData):
            return MapData.from_skin_data(
                new_skin_data, mesh_data=self._dst_mesh, name=name, component_type=typ
            )
        return GeomSubsetData.from_skin_data(
            new_skin_data, mesh_data=self._dst_mesh, name=name, component_type=typ
        )