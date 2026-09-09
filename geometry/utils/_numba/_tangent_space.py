"""
MikkTSpace-compatible tangent space computation.

Numba-accelerated per-triangle tangent / bitangent from UV gradients.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _compute_mikktspace_triangles(
    mesh_points,
    mesh_geometry,
    uv_points,
    uv_geometry,
):
    """Per-triangle tangent / bitangent from texture UV gradients.

    Args:
        mesh_points:   (num_verts, 3)  vertex positions.
        mesh_geometry: (num_faces, 3)  triangulated face-vertex indices.
        uv_points:     (num_uv_verts, 2) texture coordinates.
        uv_geometry:   (num_faces, 3)  triangulated UV face-vertex indices.

    Returns:
        T: (num_faces, 3) unnormalized tangent per face.
        B: (num_faces, 3) unnormalized bitangent per face.
    """
    num_faces = mesh_geometry.shape[0]
    T         = np.zeros((num_faces, 3), dtype=mesh_points.dtype)
    B         = np.zeros((num_faces, 3), dtype=mesh_points.dtype)

    for fi in prange(num_faces):
        mi0 = mesh_geometry[fi, 0]
        mi1 = mesh_geometry[fi, 1]
        mi2 = mesh_geometry[fi, 2]

        ui0 = uv_geometry[fi, 0]
        ui1 = uv_geometry[fi, 1]
        ui2 = uv_geometry[fi, 2]

        # position edges
        e1x = mesh_points[mi1, 0] - mesh_points[mi0, 0]
        e1y = mesh_points[mi1, 1] - mesh_points[mi0, 1]
        e1z = mesh_points[mi1, 2] - mesh_points[mi0, 2]

        e2x = mesh_points[mi2, 0] - mesh_points[mi0, 0]
        e2y = mesh_points[mi2, 1] - mesh_points[mi0, 1]
        e2z = mesh_points[mi2, 2] - mesh_points[mi0, 2]

        # UV edges
        ds1 = uv_points[ui1, 0] - uv_points[ui0, 0]
        dt1 = uv_points[ui1, 1] - uv_points[ui0, 1]
        ds2 = uv_points[ui2, 0] - uv_points[ui0, 0]
        dt2 = uv_points[ui2, 1] - uv_points[ui0, 1]

        det = ds1 * dt2 - ds2 * dt1

        if abs(det) < 1e-18:
            # degenerate UV triangle -- contribute zero T/B so that
            # downstream angle-weighted accumulation resolves the
            # tangent from neighboring non-degenerate triangles
            # (matches gltf-rs/mikktspace reference behaviour).
            continue

        r = 1.0 / det

        T[fi, 0] = (e1x * dt2 - e2x * dt1) * r
        T[fi, 1] = (e1y * dt2 - e2y * dt1) * r
        T[fi, 2] = (e1z * dt2 - e2z * dt1) * r

        B[fi, 0] = (e2x * ds1 - e1x * ds2) * r
        B[fi, 1] = (e2y * ds1 - e1y * ds2) * r
        B[fi, 2] = (e2z * ds1 - e1z * ds2) * r

    return T, B