"""
Camera math helpers (look-at, default 3/4 view, perspective autofit,
orthographic height derivation).

These helpers are renderer-agnostic so both
:mod:`cgmath.render.scene` (for Camera.look_at / Camera.autofit /
Camera.default_view) and :mod:`cgmath.render.raytracer` (for the
default-camera / autofit paths inside ``render()``) can import them
without forming a circular dependency on the renderer.

All camera-to-world matrices here follow OpenGL/Maya convention:
column 0 = right, column 1 = up, column 2 = -forward, column 3 = eye.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from cgmath.geometry.mesh import MeshData


# -- look-at -----------------------------------------------------------------


def look_at(
    eye:    Tuple[float, float, float],
    target: Tuple[float, float, float],
    up:     Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """OpenGL/Maya-style camera-to-world matrix pointing ``eye`` at ``target``.

    Column 0 = right, column 1 = up, column 2 = -forward, column 3 = eye.
    """
    eye_v    = np.asarray(eye,    dtype=np.float64)
    tgt_v    = np.asarray(target, dtype=np.float64)
    up_v     = np.asarray(up,     dtype=np.float64)

    forward  = tgt_v - eye_v
    forward  = forward / max(np.linalg.norm(forward), 1e-12)
    right    = np.cross(forward, up_v)
    right    = right / max(np.linalg.norm(right), 1e-12)
    new_up   = np.cross(right, forward)

    m        = np.eye(4, dtype=np.float64)
    m[:3, 0] = right
    m[:3, 1] = new_up
    m[:3, 2] = -forward
    m[:3, 3] = eye_v
    return m


# -- Default 3/4 view + orthographic height ----------------------------------


def _default_camera(mesh: MeshData) -> np.ndarray:
    """Build a 3/4 elevated view camera looking at the mesh's bbox center."""
    points = np.asarray(mesh.points, dtype=np.float64)
    return _default_camera_for_points(points)


def _default_camera_for_points(points: np.ndarray) -> np.ndarray:
    """Build a 3/4 elevated view from the points' bbox.  Used by both the
    single-mesh and multi-mesh paths (the latter passes the union AABB).
    """
    if points.shape[1] != 3:
        raise ValueError("default camera requires 3D points")

    mn     = points.min(axis=0)
    mx     = points.max(axis=0)
    center = (mn + mx) * 0.5
    extent = float(np.linalg.norm(mx - mn))
    if extent <= 0.0:
        extent = 1.0

    direction = np.array([1.0, 0.7, 1.0], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    eye       = center + direction * extent * 1.5

    return look_at(tuple(eye), tuple(center))


def _derive_ortho_height(points: np.ndarray, camera_matrix: np.ndarray) -> float:
    """Vertical extent of ``points`` as seen in the camera's eye space."""
    if points.shape[0] == 0:
        return 1.0
    R   = camera_matrix[:3, :3]
    eye = (points - camera_matrix[3, :3]) @ R
    return float(eye[:, 1].max() - eye[:, 1].min())


# -- AABB inflation (safety margin for autofit) ------------------------------


def _inflate_points_about_center(points: np.ndarray, factor: float) -> np.ndarray:
    """Returns a copy of ``points`` scaled outward from their AABB center by
    ``factor`` (>= 1.0).  ``factor == 1.0`` returns the input unchanged.

    Used by the autofit path to add a safety margin around bezier-sampled
    meshes whose evaluated surface bulges beyond the control-point hull,
    causing tight autofit to clip the silhouette.

    Empty inputs and degenerate (single-point) inputs are returned as-is.
    """
    if factor == 1.0 or points.shape[0] == 0:
        return points
    pts    = np.asarray(points, dtype=np.float64)
    mn     = pts.min(axis=0)
    mx     = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    return center + (pts - center) * float(factor)


# -- Perspective autofit (shift / slide / center) ----------------------------


def _autofit_camera(
    mesh:          MeshData,
    camera_matrix: np.ndarray,
    fov_v_deg:     float,
    width:         int,
    height:        int,
) -> np.ndarray:
    """Autofit a camera so that ``mesh`` exactly fills the frustum."""
    points = np.asarray(mesh.points, dtype=np.float64)
    return _autofit_camera_for_points(points, camera_matrix, fov_v_deg, width, height)


def _autofit_camera_for_points(
    points:        np.ndarray,
    camera_matrix: np.ndarray,
    fov_v_deg:     float,
    width:         int,
    height:        int,
) -> np.ndarray:
    """Override camera position so the supplied points exactly fit the
    frustum.  Preserves the rotation (top-left 3x3) of ``camera_matrix``.
    """
    if points.shape[1] != 3:
        raise ValueError("autofit requires 3D points")

    R = np.asarray(camera_matrix[:3, :3], dtype=np.float64)
    # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64).
    with np.errstate(all="ignore"):
        p = points @ R

    aov_v  = 0.5 * np.deg2rad(fov_v_deg)
    aspect = width / height
    aov_h  = np.arctan(np.tan(aov_v) * aspect)

    move   = _af_shift(p, aov_h, vertical=False)
    p      = p + move
    s      = _af_slide(p, aov_h, vertical=False)
    p      = p + s
    move   = move + s
    bbx0_z = float(p[:, 2].max())

    s      = _af_shift(p, aov_v, vertical=True)
    p      = p + s
    move   = move + s
    s      = _af_slide(p, aov_v, vertical=True)
    p      = p + s
    move   = move + s
    bbx1_z = float(p[:, 2].max())

    if bbx1_z > bbx0_z:
        adj  = np.array([0.0, 0.0, bbx0_z - bbx1_z], dtype=np.float64)
        p    = p + adj
        move = move + adj
        move = move + _af_center(p, aov_v, vertical=True)
    else:
        move = move + _af_center(p, aov_h, vertical=False)

    cam_pos = -R @ move

    new_cam        = np.array(camera_matrix, dtype=np.float64, copy=True)
    new_cam[:3, 3] = cam_pos
    return new_cam


def _af_shift(
    points: np.ndarray, half_aov: float, vertical: bool = False
) -> np.ndarray:
    angle  = 0.5 * np.pi - half_aov
    normal = np.array([np.sin(angle), 0.0, -np.cos(angle)], dtype=np.float64)
    if vertical:
        normal = normal[[1, 0, 2]]

    # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64).
    with np.errstate(all="ignore"):
        proj = points - (points @ normal)[:, None] * normal
        P    = points - proj
        dot  = P @ normal
        dist = np.sqrt(np.einsum("ij,ij->i", P, P))
    i = np.argsort(dist)

    inside  = dot[i] <= 0
    outside = dot[i] > 0

    if np.any(inside) and dist[i][inside][-1] > 0:
        return (proj[i][inside] - points[i][inside])[-1]
    return (proj[i][outside] - points[i][outside])[0]


def _af_slide(
    points: np.ndarray, half_aov: float, vertical: bool = False
) -> np.ndarray:
    angle   = -(0.5 * np.pi - half_aov)
    normal  = np.array([np.sin(angle), 0.0, -np.cos(angle)], dtype=np.float64)
    tangent = np.array([np.sin(half_aov), 0.0, np.cos(half_aov)], dtype=np.float64)
    if vertical:
        normal  = normal[[1, 0, 2]]
        tangent = tangent[[1, 0, 2]]

    # Silence spurious BLAS warnings on Apple Accelerate (matmul on arm64),
    # plus the legitimate divide-by-zero risk on `dist / denom` below.
    with np.errstate(all="ignore"):
        denom = float(tangent @ normal)
        dist  = points @ normal
        proj  = points - (dist / denom)[:, None] * tangent

        P     = points - proj
        dot   = P @ normal
        dist  = np.sqrt(np.einsum("ij,ij->i", P, P))
    i = np.argsort(dist)

    inside  = dot[i] <= 0
    outside = dot[i] > 0

    if np.any(inside) and dist[i][inside][-1] > 0:
        return (proj[i][inside] - points[i][inside])[-1]
    return (proj[i][outside] - points[i][outside])[0]


def _af_center(
    points: np.ndarray, half_aov: float, vertical: bool = False
) -> np.ndarray:
    angle   = -(0.5 * np.pi - half_aov)
    normal0 = np.array([np.sin(angle), 0.0, -np.cos(angle)], dtype=np.float64)
    normal1 = normal0 * np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if vertical:
        normal0 = normal0[[1, 0, 2]]
        normal1 = normal1[[1, 0, 2]]
        tangent = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    with np.errstate(all="ignore"):
        denom0 = float(tangent @ normal0)
        proj0  = points - ((points @ normal0) / denom0)[:, None] * tangent
        denom1 = float(tangent @ normal1)
        proj1  = points - ((points @ normal1) / denom1)[:, None] * tangent

        v0     = proj1 - proj0
        v1     = points - proj0
        m0     = np.sqrt(np.einsum("ij,ij->i", v0, v0))
        m0     = np.maximum(m0, 1e-20)
        ratio  = np.einsum("ij,ij->i", v1, v0 / m0[:, None]) / m0

    i     = np.argsort(ratio)
    near0 = float(ratio[i[0]])
    near1 = float(1.0 - ratio[i[-1]])
    avg   = 0.5 * (near0 + near1)

    if near1 > near0:
        return ((1.0 - avg) - ratio[i[-1]]) * (proj1[i[-1]] - proj0[i[-1]])
    return (avg - near0) * (proj1[i[0]] - proj0[i[0]])