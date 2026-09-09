"""
Robust skin weights transfer via weight inpainting, on bilinear patches.

Every stage is quad-native:

    * the closest point search projects onto bilinear patches, so a target
      vertex samples up to four source vertices instead of three and neither
      mesh has to be triangulated
    * inpainting and smoothing run on the edge-vertex neighbourhood through
      :class:`cgmath.geometry.skin_weights.SkinData`

N-gons are still unsupported -- ``MeshData.sample`` rejects them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cgmath.geometry._saddle_surface import SampleData
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.skin_weights import SkinData
from cgmath.geometry.utils import vector_angle_difference


@dataclass
class RobustBilinearSkinTransferOptions:
    """User set parameters for controlling the behavior of the robust bilinear skin transfer algorithm."""

    max_influences:     int = 4
    search_radius:      float = 0.05
    normal_threshold:   float = 30
    inpaint_iterations: int = 10
    smooth_iterations:  int = 10
    smooth_strength:    float = 0.1


def find_matches_closest_surface(
    sample_data:          SampleData,
    dst_mesh:             MeshData,
    src_skin_weights:     np.ndarray,
    search_radius:        float,
    normal_threshold_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each vertex on the target mesh find a match on the source mesh.

    Arguments:
        sample_data: sample data projecting the target vertices onto the source mesh
        dst_mesh: target mesh data
        src_skin_weights: #src_verts by num_bones source mesh skin weights
        search_radius: scalar distance threshold
        normal_threshold_deg: scalar normal threshold

    Returns:
        matched: #target_verts array of bools, where matched[i] is True if we found a good match for vertex i on the
                 source mesh
        target_weights: #target_verts by num_bones, where target_weights[i,:] are skinning weights copied directly from
                        source using closest point method
    """
    target_weights = sample_data(src_skin_weights)

    # sample_data.normals are the source vertex normals interpolated at the
    # projections, matching the weighting MeshData.sample uses for the source
    target_normals = dst_mesh.get_vertex_normals(
        angle_weighted=True, area_weighted=False
    )
    angles = vector_angle_difference(sample_data.normals, target_normals)

    matched = (sample_data.distances <= search_radius) & (
        angles <= normal_threshold_deg
    )

    return matched, target_weights


def robust_skinweights_transfer_bilinear(
    src_mesh:         MeshData,
    src_skin_data:    SkinData,
    dst_mesh:         MeshData,
    transfer_options: RobustBilinearSkinTransferOptions | None = None,
    sample_data:      SampleData                        | None = None,
) -> np.ndarray:
    """
    Transfers skin weights from a source mesh onto a target mesh.

    Arguments:
        src_mesh: source mesh data
        src_skin_data: source skin data
        dst_mesh: target mesh data
        transfer_options: algorithm parameters
        sample_data: optional precomputed projection of the target vertices onto
            the source mesh, so callers holding a cached one do not pay for it twice

    Returns:
        target_weights: #target_verts by num_bones skinning weights
    """
    transfer_options = transfer_options or RobustBilinearSkinTransferOptions()

    if not src_mesh.valid:
        raise ValueError("Source mesh has unreferenced vertices.")

    if not dst_mesh.valid:
        raise ValueError(
            "Target mesh has unreferenced vertices. The output skinning weights "
            "may not match the original vertex order."
        )

    if sample_data is None:
        sample_data = src_mesh.sample(dst_mesh)

    bounds        = dst_mesh.points.max(axis=0) - dst_mesh.points.min(axis=0)
    search_radius = transfer_options.search_radius * float(np.linalg.norm(bounds))

    # for every vertex on the target mesh find the closest point on the source
    # mesh and copy weights over
    matched, target_weights = find_matches_closest_surface(
        sample_data,
        dst_mesh,
        src_skin_data.weights,
        search_radius,
        transfer_options.normal_threshold,
    )

    unmatched = np.where(~matched)[0]
    if unmatched.size == dst_mesh.point_count:
        raise ValueError("No target vertex matched the source surface.")

    dst_skin_data = SkinData(
        weights=target_weights, influences=list(src_skin_data.influences)
    )
    if unmatched.size == 0:
        return dst_skin_data.weights

    neighbors = dst_mesh.get_edge_vertex_neighbors()

    # skinning weights inpainting
    dst_skin_data.inpaint(
        neighbors, unmatched, iterations=transfer_options.inpaint_iterations
    )

    # smooth the inpainted regions and their close neighbours
    if transfer_options.smooth_iterations > 0:
        neighborhood = dst_mesh.get_edge_vertex_geodesic_neighborhood(
            max_distance=search_radius, indices=unmatched
        )[0]
        dst_skin_data.smooth(
            neighbors,
            indices    = np.union1d(unmatched, neighborhood[neighborhood >= 0]),
            iterations = transfer_options.smooth_iterations,
            receptions = transfer_options.smooth_strength,
        )

    return dst_skin_data.weights