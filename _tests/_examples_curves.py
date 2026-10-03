"""
Round-trip builders for the curve and surface data: BSplineData and its
sample, BSplinePatchData and its sample and raycast, the saddle surface's
sample and raycast, and ProcrustesData.

Each builder makes a new object the way users do, through the public
constructors, fit, sample and raycast, with every saved field away from its
default where that is valid. ``test_roundtrip`` saves and loads each one.
None of these classes has a name or nested data. Their arrays share one
length, so an empty one leaves them all empty: ``test_curves_empty`` round
trips those.
"""

import numpy as np
from cgmath.constraints.procrustes import ProcrustesData
from cgmath.geometry._saddle_surface import RaycastData as SaddleRaycastData
from cgmath.geometry._saddle_surface import SampleData as SaddleSampleData
from cgmath.geometry.bspline import BSplineData
from cgmath.geometry.bspline import SampleData as CurveSampleData
from cgmath.geometry.bspline_patch import (
    BSplinePatchData,
    PatchRaycastData,
    PatchSampleData,
)
from cgmath.geometry.mesh import MeshData

# ---------------------------------- SHAPES ---------------------------------- #


def arm(n=6):
    """the joint positions of a bending arm chain"""
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([t * 10.0, np.sin(t * 3.0) * 2.0, np.cos(t * 2.0)])


def ring(n=8):
    """a closed loop of n points, an ellipse that rises and falls twice"""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(a) * 4.0, np.sin(a) * 2.5, np.sin(a * 2.0) * 0.5])


def tube(nu=6, nv=5):
    """the control grid of a tube: u closes around it, v runs along it"""
    a      = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)[:, None]
    height = np.linspace(0.0, 6.0, nv)[None, :]
    radius = 2.0 + 0.3 * np.sin(height)
    return np.stack(
        np.broadcast_arrays(np.cos(a) * radius, np.sin(a) * radius, height), axis=-1
    )


def sheet(nu=5, nv=6):
    """the control grid of a wavy sheet in the xy plane, facing +z"""
    u, v = np.meshgrid(np.linspace(-4, 4, nu), np.linspace(-5, 5, nv), indexing="ij")
    return np.stack([u, v, np.sin(u * 0.5) * np.cos(v * 0.5)], axis=-1)


def strip():
    """a mesh of one quad and two triangles, facing +z"""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.2],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.2],
        ]
    )
    return MeshData(
        points  = points,
        indices = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4]),
        counts  = np.array([4, 3, 3]),
    )


# --------------------------------- BUILDERS --------------------------------- #


def curve() -> BSplineData:
    """a closed quadratic through a ring of edit points, every setting changed"""
    return BSplineData.from_edit_points(
        ring(),
        degree             = 2,
        periodic           = True,
        uniform            = True,
        registered         = True,
        use_numba          = False,
        arc_length_samples = 257,
    )


def curve_sample() -> CurveSampleData:
    """points beside a spine with stacked ends and Maya's knots, projected onto it"""
    spine = BSplineData.from_edit_points(arm(), collapse=(2, 2))
    offsets = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.2, -0.4, 0.3],
            [0.0, 0.0, -0.6],
            [-0.3, 0.2, 0.1],
            [0.4, 0.4, 0.4],  # past the stacked end, where there is no tangent
        ]
    )
    return spine.sample(arm(5) + offsets)


def patch() -> BSplinePatchData:
    """a tube closed in u, open in v, every setting changed"""
    return BSplinePatchData(
        points             = tube(),
        degree_u           = 2,
        degree_v           = 3,
        periodic_u         = True,
        periodic_v         = False,
        uniform_u          = True,
        uniform_v          = True,
        use_numba          = False,
        registered_u       = True,
        registered_v       = True,
        arc_length_samples = 300,
    )


def patch_sample() -> PatchSampleData:
    """points inside and outside a tube, projected onto it"""
    surface = BSplinePatchData(
        points=tube(), degree_u=2, degree_v=3, periodic_u=True, periodic_v=False
    )
    queries = np.array(
        [[3.0, 0.0, 1.0], [0.0, 1.0, 2.5], [-2.5, -2.5, 4.0], [0.5, -3.0, 5.5]]
    )
    return surface.sample(queries)


def patch_raycast() -> PatchRaycastData:
    """rays down onto a sheet, one up into it from below, and one that misses (NaN)"""
    surface = BSplinePatchData(
        points=sheet(), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False
    )
    origins = np.array(
        [[0.0, 0.0, 5.0], [1.0, -2.0, 5.0], [-1.0, 1.0, -5.0], [40.0, 40.0, 5.0]]
    )
    directions = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    return surface.raycast(origins, directions, forward_only=False)


def saddle_sample() -> SaddleSampleData:
    """points above, below and on a quad-and-triangles mesh, sampled on it"""
    queries = np.array(
        [
            [0.5, 0.5, 0.5],  # above the quad
            [0.3, 0.6, -0.4],  # below it: occluded
            [1.6, 0.4, 0.3],  # above a triangle: its geometry is -1 padded
            [2.0, 1.0, 0.2],  # on a vertex: matched exactly
        ]
    )
    return strip().sample(queries)


def saddle_raycast() -> SaddleRaycastData:
    """a ray down onto a quad, one up into a triangle, and one that misses (NaN, -1)"""
    origins    = np.array([[0.5, 0.5, 2.0], [1.7, 0.5, -2.0], [9.0, 9.0, 2.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    return strip().raycast(origins, directions, forward_only=False)


def procrustes() -> ProcrustesData:
    """a cube held by two clusters, one of half its points, no scale offset"""
    cube = np.array(
        [[x, y, z] for z in (0.5, -0.5) for y in (-0.5, 0.5) for x in (-0.5, 0.5)]
    )
    lid        = np.eye(4)
    lid[3, :3] = [0.0, 0.5, 0.0]

    constraint = ProcrustesData(cube)
    constraint.attach(lid, [2, 3, 4, 5])  # the shorter cluster is -1 padded
    constraint.attach(np.eye(4))
    constraint.scale_offset = False

    moved = cube * 1.2
    moved[:2, 2] += 0.3
    constraint.update(moved)  # the per-frame target and results are not saved
    return constraint


EXAMPLES = {
    BSplineData:       curve,
    CurveSampleData:   curve_sample,
    BSplinePatchData:  patch,
    PatchSampleData:   patch_sample,
    PatchRaycastData:  patch_raycast,
    SaddleSampleData:  saddle_sample,
    SaddleRaycastData: saddle_raycast,
    ProcrustesData:    procrustes,
}
