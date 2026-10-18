"""
A resampler class for mesh-dependent data: morphs, skins, meshes, maps and UVs.

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

# build a UV set for the destination from the source UVs, projected in world
# space (closest point, or a ray along each destination vertex normal)
dst_uv_data = resampler.resample_uv(raycast=True)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from cgmath.geometry._saddle_surface import raycast as raycast_patches
from cgmath.geometry._saddle_surface import sample as sample_patches
from cgmath.geometry._saddle_surface import SampleData
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import _require_no_ngons, MeshData, SampleMethod, UVData
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
    compute_samples,
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
        * UV Set (spatial projection only, see resample_uv)
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
            src_uv: Source UV data object, required in uv resample mode and
                by resample_uv() when it is not given a UV set.
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

    def _get_ray_directions(
        self,
        use_mesh_normals: bool,
        angle_weighted:   bool,
        area_weighted:    bool,
    ) -> np.ndarray:
        """Returns one unit direction per destination vertex.

        Reads the stored dst_mesh.normals when asked to and they exist, else
        computes them with get_vertex_normals(). Never writes normals onto the
        mesh.

        Raises:
            ValueError: The stored normals fit neither the vertex count nor
                the face-vertex count, and carry no normal_indices.
        """
        dst     = self.dst_mesh
        normals = dst.normals if use_mesh_normals else None

        if normals is None:
            directions = dst.get_vertex_normals(
                angle_weighted = angle_weighted,
                area_weighted  = area_weighted,
            )
        else:
            normals = np.asarray(normals, dtype=np.float64)
            if dst.normal_indices is not None:
                corner_normals = normals[dst.normal_indices]
            elif normals.shape[0] == dst.point_count:
                corner_normals = None
            elif normals.shape[0] == dst.indices.size:
                corner_normals = normals  # face-varying, not indexed
            else:
                raise ValueError(
                    f"dst_mesh.normals holds {normals.shape[0]} normals, which "
                    f"matches neither its {dst.point_count} vertices nor its "
                    f"{dst.indices.size} face-vertices, and it has no "
                    "normal_indices. Pass use_mesh_normals=False to compute them."
                )

            if corner_normals is None:
                directions = normals.copy()
            else:
                directions = np.zeros((dst.point_count, 3))
                np.add.at(directions, dst.indices, corner_normals)

        lengths               = np.linalg.norm(directions, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return directions / lengths

    def _project_onto_faces(
        self,
        points:          np.ndarray,
        faces:           np.ndarray,
        surface_normals: np.ndarray | None = None,
        directions:      np.ndarray | None = None,
        forward_only:    bool              = False,
        twosided:        bool              = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projects points onto the given source faces.

        Takes the closest point when directions is None. Otherwise casts a ray
        per point, and a ray that misses takes the closest point instead.

        Args:
            surface_normals: None projects onto the flat faces (BILINEAR). The
                source's vertex normals, as MeshData.sample() computes them,
                project onto its PN-quad patches (BEZIER). The caller computes
                them once and passes them to every call.

        Returns:
            (projections, weights, face indices): world space points on the
            source surface, (n, 4) corner weights, and source face indices,
            numbered over the whole source mesh.
        """
        src   = self.src_mesh
        faces = np.asarray(faces)
        whole = faces.size == src.face_count  # unique ids, so every face
        if whole:
            faces = np.arange(src.face_count)

        # src.geometry is as wide as its widest face; a subset holds no n-gon
        geometry = src.geometry if whole else src.geometry[faces][:, :4]

        # the closest point on a subset reads only the vertices its faces
        # hold: the sampler scans every vertex it is given, so the whole
        # source in every call would make a map of many shells cost shells x
        # source size in resample_uv
        face_points, face_geometry, face_normals = src.points, geometry, surface_normals
        if not whole:
            held = geometry >= 0
            used, inverse = np.unique(geometry[held], return_inverse=True)
            face_geometry       = np.full(geometry.shape, -1, dtype=geometry.dtype)
            face_geometry[held] = inverse.reshape(-1)
            face_points         = src.points[used]
            if surface_normals is not None:
                face_normals = surface_normals[used]

        def closest(query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # the vertex match looks only at the vertices these faces hold,
            # so a subset answers from its own faces
            result = sample_patches(
                query, face_points, face_geometry, surface_normals=face_normals
            )
            return result.projections, result.weights, faces[result.indices]

        if directions is None:
            return closest(points)

        if whole:
            # the mesh method reuses the source's cached BVH
            method = SampleMethod.BILINEAR
            if surface_normals is not None:
                method = SampleMethod.BEZIER
            hits = src.raycast(
                points,
                directions,
                method       = method,
                forward_only = forward_only,
                twosided     = twosided,
            )
        else:
            hits = raycast_patches(
                src.points,
                src.points,
                geometry,
                points,
                directions,
                surface_normals = surface_normals,
                forward_only    = forward_only,
                twosided        = twosided,
            )

        projections = np.array(hits.projections, dtype=np.float64)
        weights     = np.array(hits.weights, dtype=np.float64)
        indices     = np.where(hits.hit, faces[np.maximum(hits.indices, 0)], -1)

        missed = ~hits.hit
        if np.any(missed):
            projections[missed], weights[missed], indices[missed] = closest(points[missed])
        return projections, weights, indices

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

                normals           = np.cross(U, V)
                norms             = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normals           = normals / norms
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
        self._buffer[:]                   = 0
        self._buffer[target_data.indices] = target_data.offsets

        # resample the offsets
        out_data         = target_data.copy()
        sample_data      = self.get_sample_data(mode, method)
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
            sample_data      = self.get_sample_data(mode, method)
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

    def resample_uv(
        self,
        uv_data:          UVData       | None = None,
        method:           SampleMethod | str  = SampleMethod.BILINEAR,
        raycast:          bool                = False,
        forward_only:     bool                = False,
        twosided:         bool                = True,
        use_mesh_normals: bool                = True,
        angle_weighted:   bool                = True,
        area_weighted:    bool                = True,
    ) -> UVData:
        """Builds a dst UV set from a src UV set, projected in world space.

        Every destination face takes the source UV shell under its centroid,
        on the flat source faces whatever the method, and all its corners are
        read from that shell. A seam between two source UV shells that runs
        along destination edges therefore splits the result there, instead of
        stretching faces across the atlas. Such a seam cutting through a
        destination face clamps that face's far corners to the shell border.
        A cut inside one shell is not split: faces across it stretch.

        Args:
            uv_data: Source UV set, on the source topology. Defaults to src_uv.
            method: Surface fitting method -- BILINEAR or BEZIER.
            raycast: If True, cast a ray from each destination vertex along its
                normal instead of taking the closest point. A ray that misses
                the source takes the closest point.
            forward_only: Raycast only. If False, also cast backward and keep
                the closer hit.
            twosided: Raycast only. If False, back-face hits are rejected.
            use_mesh_normals: Raycast only. If True and dst_mesh stores
                normals, rays follow them, face-varying normals summed per
                vertex. Otherwise normals come from get_vertex_normals().
            angle_weighted: get_vertex_normals() option, for computed normals.
            area_weighted: get_vertex_normals() option, for computed normals.

        Returns:
            A new UV set, named after the source one, with one UV per
            destination vertex and source shell. Neither mesh is modified.

        Raises:
            RuntimeError: No UV set was given and src_uv is None.
            ValueError: The UV set's faces do not match src_mesh, or it has
                no face with UVs, or, with raycast=True, dst_mesh.normals fits
                neither its vertices nor its face-vertices and has no
                normal_indices (pass use_mesh_normals=False).
            NotImplementedError: A source face with UVs is an n-gon. N-gons
                without UVs are never sampled.
        """
        method = SampleMethod(method)
        src_uv = self.src_uv if uv_data is None else uv_data
        if src_uv is None:
            raise RuntimeError("UV data not supplied.")

        src = self.src_mesh
        dst = self.dst_mesh
        if src_uv.counts.size != src.face_count or np.any(
            (src_uv.counts > 0) & (src_uv.counts != src.counts)
        ):
            raise ValueError("uv_data does not belong to src_mesh: their faces differ.")

        uv_faces = np.flatnonzero(src_uv.counts > 0)
        if uv_faces.size == 0:
            raise ValueError("uv_data has no faces with UVs.")
        _require_no_ngons("resample_uv", src.counts[uv_faces])

        # one UV shell id per source face, -1 on a face without UVs
        shell_of_face = np.full(src.face_count, -1, dtype=np.int64)
        for shell, faces in enumerate(src_uv.shell_faces):
            shell_of_face[faces] = shell
        shell_of_face[src_uv.counts == 0] = -1

        geometry  = dst.geometry
        valid     = geometry >= 0
        corners   = np.where(valid, geometry, 0)
        centroids = (dst.points[corners] * valid[..., None]).sum(axis=1) / dst.counts[:, None]

        vertex_directions = None
        face_directions   = None
        if raycast:
            vertex_directions = self._get_ray_directions(
                use_mesh_normals, angle_weighted, area_weighted
            )
            face_directions       = (vertex_directions[corners] * valid[..., None]).sum(axis=1)
            lengths               = np.linalg.norm(face_directions, axis=1, keepdims=True)
            lengths[lengths == 0] = 1.0
            face_directions /= lengths

        # BEZIER's patches follow the source's vertex normals: computed once
        # here, not once per projection below
        surface_normals = None
        if method == SampleMethod.BEZIER:
            surface_normals = src.get_vertex_normals(
                angle_weighted=True, area_weighted=False
            )

        projections, weights, hit_faces = self._project_onto_faces(
            dst.points,
            uv_faces,
            surface_normals,
            directions   = vertex_directions,
            forward_only = forward_only,
            twosided     = twosided,
        )
        # a face's shell is read off the flat faces, whatever the method: a
        # face's PN-quad patch can bulge away from its centroid and leave a
        # neighbour's patch, in another shell, closer
        _, _, centroid_faces = self._project_onto_faces(
            centroids,
            uv_faces,
            None,
            directions   = face_directions,
            forward_only = forward_only,
            twosided     = twosided,
        )

        corner_shell = np.repeat(shell_of_face[centroid_faces], dst.counts)
        corner_faces = hit_faces[dst.indices]
        uvs = compute_samples(
            src_uv.points, weights[dst.indices], src_uv.geometry[corner_faces]
        )

        # a corner whose vertex landed in another shell sits on or past a
        # seam: read it from its face's shell, at the nearest point there.
        # The stray corners and the source faces are sorted by shell once,
        # so each shell's pass reads only its own runs, and one
        # compute_samples() reads them all: no pass scans the whole mesh
        stray = np.flatnonzero(shell_of_face[corner_faces] != corner_shell)
        if stray.size:
            stray = stray[np.argsort(corner_shell[stray], kind="stable")]
            shells, starts = np.unique(corner_shell[stray], return_index=True)
            ends          = np.append(starts[1:], stray.size)
            face_order    = np.argsort(shell_of_face, kind="stable")
            sorted_shells = shell_of_face[face_order]
            face_starts   = np.searchsorted(sorted_shells, shells, side="left")
            face_ends     = np.searchsorted(sorted_shells, shells, side="right")
            stray_weights = np.empty((stray.size, weights.shape[1]))
            stray_faces   = np.empty(stray.size, dtype=np.int64)
            for start, end, face_start, face_end in zip(starts, ends, face_starts, face_ends):
                _, shell_weights, shell_faces = self._project_onto_faces(
                    projections[dst.indices[stray[start:end]]],
                    face_order[face_start:face_end],
                    surface_normals,
                )
                stray_weights[start:end] = shell_weights
                stray_faces[start:end]   = shell_faces
            uvs[stray] = compute_samples(
                src_uv.points, stray_weights, src_uv.geometry[stray_faces]
            )

        # weld: one UV vertex per (destination vertex, shell)
        keys = np.stack([dst.indices, corner_shell], axis=1)
        _, first, uv_indices = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )

        return UVData(
            name    = src_uv.name,
            indices = uv_indices.reshape(-1).astype(np.int32),
            counts  = np.array(dst.counts, dtype=np.int32),
            points  = uvs[first],
        )