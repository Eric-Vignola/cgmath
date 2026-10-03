"""Filled-in deformers, for test_roundtrip.

Each builder makes a deformer the way a user does -- from a mesh, a skin, a
rig or a lattice -- changes every setting from its default and binds it, so
every saved field holds a value of its own. The meshes are small: a few dozen
points.
"""

import numpy as np
from cgmath.geometry.deform import (
    DeltaMushData,
    FFDData,
    PatchRelaxData,
    SkinDeformData,
    WrapData,
)
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.skin_weights import SkinData
from cgmath.hierarchy import HierarchyData, TransformData


def _grid(n: int = 6, height: float = 0.3, name: str = "grid") -> MeshData:
    """an open quad grid on XY with a sine bump in Z: it has a border"""
    t            = np.linspace(0.0, 1.0, n)
    points       = np.array([[x, y, 0.0] for y in t for x in t])
    points[:, 2] = height * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices.extend([a, a + 1, a + n + 1, a + n])

    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full((n - 1) ** 2, 4, dtype=np.int32),
        name    = name,
    )


def _torus(around: int = 8, across: int = 6, name: str = "torus") -> MeshData:
    """a closed quad torus: it has no border"""
    u = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, across, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ring = 1.0 + 0.35 * np.cos(vv)
    points = np.stack(
        [ring * np.cos(uu), ring * np.sin(uu), 0.35 * np.sin(vv)], axis=-1
    ).reshape(-1, 3)

    indices = []
    for i in range(around):
        for j in range(across):
            k, l = (i + 1) % around, (j + 1) % across
            indices.extend(
                [i * across + j, k * across + j, k * across + l, i * across + l]
            )

    return MeshData(
        points  = points,
        indices = np.asarray(indices, dtype=np.int32),
        counts  = np.full(around * across, 4, dtype=np.int32),
        name    = name,
    )


def delta_mush() -> DeltaMushData:
    """bound with DDM on an open mesh; pinning turned off after the border was read"""
    mush = DeltaMushData(
        _grid(),
        smooth_iterations = 6,
        smooth_step_size  = 0.4,
        weight            = 0.7,
        method            = "ddm",
    )
    # the border read at construction stays saved, pinned or not
    mush.pin_borders = False
    mush.bind()
    return mush


def patch_relax() -> PatchRelaxData:
    """bound on a closed mesh, so its border is an empty array, with a painted mask"""
    mesh = _torus()
    relaxer = PatchRelaxData(
        mesh,
        iterations    = 12,
        alpha         = 0.8,
        surface_blend = 0.3,
        step_size     = 0.4,
        pin_borders   = False,
        mask          = np.linspace(0.0, 1.0, len(mesh.points)),
    )
    relaxer.bind()
    return relaxer


def ffd() -> FFDData:
    """a lattice over part of a mesh, bound, with falloff past its edge"""
    lattice = FFDData.create_lattice(
        (3, 4, 3), bbox_min=[-0.1, -0.1, -0.5], bbox_max=[0.7, 1.1, 0.5], name="ffd_box"
    )
    ffd                 = FFDData.from_mesh(lattice, divisions=(3, 4, 3))
    ffd.local_influence = (3, 4, 2)
    ffd.outside         = "falloff"
    ffd.falloff_radius  = 1.5

    # the grid runs to x = 1, past the lattice: those points fall off
    ffd.bind(_grid())

    deformed = ffd.lattice.copy()
    deformed[:, :, -1, 2] += 0.4
    ffd.update(deformed)
    return ffd


def skin_deform() -> SkinDeformData:
    """dual quaternion, bound, its matrices from a bind rig, joints reordered"""
    rig = HierarchyData(
        [
            TransformData(name="A:root", translate=(0.0, 0.0, 0.0)),
            TransformData(name="A:spine", parent_node="A:root", translate=(0.5, 0.0, 0.0)),
            TransformData(
                name        = "A:tip",
                parent_node = "A:spine",
                translate   = (0.5, 0.0, 0.0),
                rotate      = (0.0, 0.0, 15.0),
            ),
        ]
    )

    mesh = _grid(name="A:body")
    x    = mesh.points[:, 0]
    # at most two influences per vertex: root fades into spine, spine into tip
    weights = np.zeros((len(x), 3))
    low, high = x <= 0.5, x > 0.5
    weights[low, 0]  = 1.0 - 2.0 * x[low]
    weights[low, 1]  = 2.0 * x[low]
    weights[high, 1] = 2.0 - 2.0 * x[high]
    weights[high, 2] = 2.0 * x[high] - 1.0
    skin             = SkinData(weights=weights, influences=["A:root", "A:spine", "A:tip"], name="A:skin")

    deformer = SkinDeformData(
        mesh,
        skin,
        bind_rig = rig,
        joints   = ["A:tip", "A:root", "A:spine"],
        method   = "dqs",
        name     = "A:body_skin",
    )
    deformer.bind()
    return deformer


def wrap() -> WrapData:
    """a geodesic cage wrap with a compact kernel, solved, so its pseudo-inverse is saved"""
    cage  = _grid(4, height=0.4, name="cage")
    moved = cage.points + np.random.default_rng(5).normal(scale=0.05, size=cage.points.shape)

    wrap  = WrapData(kernel="wendland_c2", name="cage_wrap")
    wrap.set_source(cage)
    wrap.set_target(moved)
    wrap.set_radius(0.9)
    wrap.set_geodesic_radius(1.0)
    wrap.deform(_grid().points)
    return wrap


EXAMPLES = {
    DeltaMushData:  delta_mush,
    FFDData:        ffd,
    PatchRelaxData: patch_relax,
    SkinDeformData: skin_deform,
    WrapData:       wrap,
}
