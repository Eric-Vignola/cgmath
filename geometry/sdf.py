"""Dual Marching Cubes (DMC) implementation for isosurface extraction.

This module provides a high-performance implementation of the Dual Marching Cubes
algorithm using numba for parallel execution. DMC produces quad-dominant meshes
which are often preferable to the triangle-only output of standard Marching Cubes.

The module also provides composable SDF transform utilities (rotation, translation,
non-uniform scale), pure SDF evaluators, and CSG boolean operations for building
complex shapes from simple primitives.

Functional API Example:
    >>> import numpy as np
    >>> from cgmath.geometry.sdf import (
    ...     dual_marching_cubes, make_grid, transform_points,
    ...     eval_sphere, eval_box, sdf_union,
    ... )
    >>>
    >>> # Set up a shared grid
    >>> shape = (128, 128, 128)
    >>> bounds = (np.array([-2, -2, -2]), np.array([2, 2, 2]))
    >>> X, Y, Z, origin, spacing = make_grid(shape, bounds)
    >>>
    >>> # Sphere at the origin
    >>> sdf_a = eval_sphere(X, Y, Z, radius=0.8)
    >>>
    >>> # Thin box rotated 30 degrees around X
    >>> Xt, Yt, Zt = transform_points(X, Y, Z, rotation=[30, 0, 0])
    >>> sdf_b = eval_box(Xt, Yt, Zt, half_extents=np.array([0.1, 1.0, 0.1]))
    >>>
    >>> # Union and extract
    >>> mesh = dual_marching_cubes(
    ...     sdf_union(sdf_a, sdf_b),
    ...     grid_origin=origin, grid_spacing=spacing,
    ... )
    >>> mesh.save_obj("/tmp/combined.obj")

OOP Scene API Example:
    >>> from cgmath.geometry.sdf import SDFSphere, SDFBox, SDFCylinder, DMCField
    >>>
    >>> # Create primitives (using TransformData properties: translate, rotate, scale)
    >>> sphere = SDFSphere(radius=1.0, name="base_sphere")
    >>> box = SDFBox(half_extents=[0.1, 1.0, 0.1], name="fin")
    >>> box.rotate = [30, 0, 0]  # 30deg around X
    >>> box.translate = [0.5, 0, 0]
    >>>
    >>> cutter = SDFCylinder(radius=0.3, height=3.0, name="hole")
    >>> cutter.rotate = [0, 0, 90]
    >>>
    >>> # Build field (bounds auto-computed from primitives)
    >>> field = DMCField(resolution=(128, 128, 128))
    >>> field.add(sphere)
    >>> field.add(box, smoothing=0.1)  # smooth union
    >>> field.subtract(cutter)
    >>>
    >>> # Get result
    >>> mesh = field.mesh_data  # MeshData object
    >>> mesh.save_obj("/tmp/result.obj")
    >>>
    >>> # Modify and re-query -- only recomputes what changed
    >>> sphere.radius = 1.2
    >>> box.rotate = [45, 0, 0]
    >>> mesh = field.mesh_data  # recomputes automatically
"""

from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.utils._numba._sdf import (
    _build_cube_to_vertex_map,
    _classify_cubes,
    _compute_dual_vertices,
    _count_active_cubes,
    _extract_active_cubes,
    _generate_faces,
)
from transforms import euler_to_matrix
from cgmath.hierarchy import TransformData
from transforms import matrix_inverse, matrix_point_multiply as matrix_point


# =============================================================================
# LOOKUP TABLES
# =============================================================================

# Edge table: maps cube configuration (8-bit) to list of edges crossed (12-bit)
# Each bit in the result indicates whether that edge is crossed by the isosurface.
EDGE_TABLE = np.array([
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
], dtype=np.uint16)

# Edge vertices: which two corners (0-7) form each edge (0-11)
# Edges 0-3: bottom face, 4-7: top face, 8-11: vertical edges
EDGE_VERTICES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom edges
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top edges
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ],
    dtype=np.int32,
)

# Corner offsets in (x, y, z) for a unit cube
# Corner ordering matches the standard marching cubes convention
CORNER_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],  # bottom
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],  # top
    ],
    dtype=np.int32,
)

# Edge axis: which axis each edge is aligned with (0=X, 1=Y, 2=Z)
EDGE_AXIS = np.array([0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2], dtype=np.int32)

# Face cube offsets: for each edge axis, the 4 cube offsets sharing that edge
# These define which 4 cubes share an edge, used to create quad faces
FACE_CUBE_OFFSETS_X = np.array(
    [[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]], dtype=np.int32
)
FACE_CUBE_OFFSETS_Y = np.array(
    [[0, 0, 0], [-1, 0, 0], [-1, 0, -1], [0, 0, -1]], dtype=np.int32
)
FACE_CUBE_OFFSETS_Z = np.array(
    [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]], dtype=np.int32
)


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def dual_marching_cubes(
    scalar_field: np.ndarray,
    iso_value:    float                = 0.0,
    grid_origin:  Optional[np.ndarray] = None,
    grid_spacing: Optional[np.ndarray] = None,
    name:         Optional[str]        = None,
) -> MeshData:
    """
    Generate a mesh from a scalar field using Dual Marching Cubes.

    Dual Marching Cubes places vertices at the center of each active cube
    (where the isosurface passes through) and connects them with quad faces.
    This produces cleaner, more uniform meshes compared to standard Marching Cubes.

    Args:
        scalar_field: 3D numpy array of scalar values with shape (nx, ny, nz).
            The isosurface is extracted where values equal iso_value.
        iso_value: The threshold value for the isosurface. Points with
            scalar_field >= iso_value are considered "inside". Default: 0.0
        grid_origin: Origin point [x, y, z] of the grid in world space.
            Default: [0.0, 0.0, 0.0]
        grid_spacing: Spacing [dx, dy, dz] between grid points.
            Default: [1.0, 1.0, 1.0]
        name: Optional name for the resulting mesh. Default: "dmc_mesh"

    Returns:
        MeshData object containing the extracted isosurface mesh.
        The mesh is quad-dominant (all faces are quads).

    Raises:
        ValueError: If scalar_field is not 3D or has fewer than 2 points
            in any dimension.

    Example:
        >>> import numpy as np
        >>> from cgmath.geometry import dual_marching_cubes
        >>>
        >>> # Create a sphere signed distance field
        >>> x = np.linspace(-2, 2, 64)
        >>> X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        >>> sdf = np.sqrt(X**2 + Y**2 + Z**2) - 1.0
        >>>
        >>> # Extract the isosurface at distance 0 (the sphere surface)
        >>> mesh = dual_marching_cubes(sdf, iso_value=0.0)
        >>>
        >>> print(f"Vertices: {mesh.point_count}, Faces: {mesh.face_count}")
        >>> mesh.save_obj("/tmp/sphere.obj")
    """
    # Validate input
    if scalar_field.ndim != 3:
        raise ValueError(f"scalar_field must be 3D, got {scalar_field.ndim}D array")

    nx, ny, nz = scalar_field.shape
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(
            f"scalar_field must have at least 2 points in each dimension, "
            f"got shape {scalar_field.shape}"
        )

    # Set defaults
    if grid_origin is None:
        grid_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        grid_origin = np.asarray(grid_origin, dtype=np.float64)

    if grid_spacing is None:
        grid_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        grid_spacing = np.asarray(grid_spacing, dtype=np.float64)

    if name is None:
        name = "dmc_mesh"

    # Ensure contiguous arrays for numba
    scalar_field = np.ascontiguousarray(scalar_field, dtype=np.float64)

    # Step 1: Classify cubes
    cube_config = _classify_cubes(scalar_field, iso_value)

    # Step 2: Find active cubes
    counts      = _count_active_cubes(cube_config)
    total_count = int(counts.sum())

    if total_count == 0:
        # No isosurface - return empty mesh
        return MeshData(
            indices = np.array([], dtype=np.int32),
            counts  = np.array([], dtype=np.int32),
            points  = np.zeros((0, 3), dtype=np.float64),
            name    = name,
        )

    active_cubes = _extract_active_cubes(cube_config, counts, total_count)

    # Step 3: Build cube-to-vertex mapping
    cube_to_vertex = _build_cube_to_vertex_map(active_cubes, nx - 1, ny - 1, nz - 1)

    # Step 4: Compute dual vertex positions
    points = _compute_dual_vertices(
        scalar_field,
        cube_config,
        active_cubes,
        iso_value,
        grid_origin,
        grid_spacing,
        EDGE_TABLE,
        EDGE_VERTICES,
        CORNER_OFFSETS,
    )

    # Step 5: Generate faces
    faces, total_faces = _generate_faces(
        active_cubes,
        cube_to_vertex,
        cube_config,
        scalar_field,
        EDGE_TABLE,
        EDGE_VERTICES,
        EDGE_AXIS,
        FACE_CUBE_OFFSETS_X,
        FACE_CUBE_OFFSETS_Y,
        FACE_CUBE_OFFSETS_Z,
        CORNER_OFFSETS,
        nx - 1,
        ny - 1,
        nz - 1,
    )

    if total_faces == 0:
        # No faces generated - return mesh with only points
        return MeshData(
            indices = np.array([], dtype=np.int32),
            counts  = np.array([], dtype=np.int32),
            points  = points,
            name    = name,
        )

    # Convert faces to MeshData format
    indices     = faces[:total_faces].ravel()
    mesh_counts = np.full(total_faces, 4, dtype=np.int32)

    return MeshData(
        indices = indices,
        counts  = mesh_counts,
        points  = points,
        name    = name,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def sphere_sdf(
    shape:  Tuple[int, int, int],
    center: Optional[np.ndarray]                    = None,
    radius: float                                   = 1.0,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a sphere signed distance field for testing.

    Args:
        shape: Grid resolution (nx, ny, nz).
        center: Center of the sphere [x, y, z]. Default: [0, 0, 0]
        radius: Radius of the sphere. Default: 1.0
        bounds: Tuple of (min_bound, max_bound) arrays defining the grid extent.
            Default: sphere centered with padding of 0.5 * radius

    Returns:
        Tuple of (scalar_field, grid_origin, grid_spacing)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)

    if bounds is None:
        padding   = 0.5 * radius
        min_bound = center - radius - padding
        max_bound = center + radius + padding
    else:
        min_bound, max_bound = bounds
        min_bound = np.asarray(min_bound, dtype=np.float64)
        max_bound = np.asarray(max_bound, dtype=np.float64)

    # Create grid
    x = np.linspace(min_bound[0], max_bound[0], shape[0])
    y = np.linspace(min_bound[1], max_bound[1], shape[1])
    z = np.linspace(min_bound[2], max_bound[2], shape[2])

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Compute SDF
    sdf = (
        np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
        - radius
    )

    # Compute grid parameters
    grid_origin  = min_bound.copy()
    grid_spacing = (max_bound - min_bound) / (np.array(shape) - 1)

    return sdf, grid_origin, grid_spacing


# =============================================================================
# OOP SCENE API
# =============================================================================


class CSGOp(str, Enum):
    """CSG operation types for combining SDF primitives."""

    UNION        = "union"
    INTERSECTION = "intersection"
    DIFFERENCE   = "difference"
    SMOOTH_UNION = "smooth_union"


class SDFSphere(TransformData):
    """Sphere SDF primitive inheriting transform properties from TransformData."""

    def __init__(
        self,
        radius:    float                                    = 1.0,
        translate: Optional[Union[List[float], np.ndarray]] = None,
        rotate:    Optional[Union[List[float], np.ndarray]] = None,
        scale:     Optional[Union[List[float], np.ndarray]] = None,
        name:      str                                      = "sphere",
    ) -> None:
        super().__init__(
            name      = name,
            translate = translate,
            rotate    = rotate,
            scale     = scale,
        )
        self._sdf_radius: float = radius
        self._field:      Optional["DMCField"] = None

    def _invalidate(self) -> None:
        """Notify parent field that this primitive has changed."""
        if self._field is not None:
            self._field._invalidate()

    @property
    def radius(self) -> float:
        """Sphere radius."""
        return self._sdf_radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._sdf_radius = value
        self._invalidate()

    def evaluate(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate the SDF on pre-transformed (local) coordinates."""
        return eval_sphere(X, Y, Z, self._sdf_radius)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sphere AABB: abs(R) @ (radius * scale) offset by translate."""
        local_half = np.full(3, self._sdf_radius, dtype=np.float64) * self.scale
        R          = self.rotate_matrix[:3, :3].T
        world_half = np.abs(R) @ local_half
        return (self.translate - world_half, self.translate + world_half)

    def sample(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate the SDF on world-space coordinates.

        Uses separate handling of scale to avoid shearing artifacts:
        1. Transform points by rigid (rotation + translation) inverse
        2. Apply inverse scale separately
        3. Correct distance by min(abs(scale))
        """
        scale     = self.scale
        min_scale = np.min(np.abs(scale))

        R         = self.rotate_matrix
        T         = np.eye(4)
        T[3, :3] = self.translate
        rigid_matrix   = R @ T
        rigid_inv      = matrix_inverse(rigid_matrix)[0]

        original_shape = X.shape
        points         = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        local_points   = matrix_point(points, rigid_inv)
        local_points /= scale

        Xt = local_points[:, 0].reshape(original_shape)
        Yt = local_points[:, 1].reshape(original_shape)
        Zt = local_points[:, 2].reshape(original_shape)

        return self.evaluate(Xt, Yt, Zt) * min_scale


class SDFBox(TransformData):
    """Axis-aligned box SDF primitive inheriting transform properties from TransformData."""

    def __init__(
        self,
        half_extents: Optional[Union[List[float], np.ndarray]] = None,
        translate:    Optional[Union[List[float], np.ndarray]] = None,
        rotate:       Optional[Union[List[float], np.ndarray]] = None,
        scale:        Optional[Union[List[float], np.ndarray]] = None,
        name:         str                                      = "box",
    ) -> None:
        super().__init__(
            name      = name,
            translate = translate,
            rotate    = rotate,
            scale     = scale,
        )
        self._half_extents: np.ndarray = (
            np.array([0.5, 0.5, 0.5], dtype=np.float64)
            if half_extents is None
            else np.asarray(half_extents, dtype=np.float64)
        )
        self._field: Optional["DMCField"] = None

    def _invalidate(self) -> None:
        """Notify parent field that this primitive has changed."""
        if self._field is not None:
            self._field._invalidate()

    @property
    def half_extents(self) -> np.ndarray:
        """Half-extents [hx, hy, hz] along each axis."""
        return self._half_extents

    @half_extents.setter
    def half_extents(self, value: Union[List[float], np.ndarray]) -> None:
        self._half_extents = np.asarray(value, dtype=np.float64)
        self._invalidate()

    def evaluate(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate the SDF on pre-transformed (local) coordinates."""
        return eval_box(X, Y, Z, self._half_extents)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box AABB: abs(R) @ (half_extents * scale) offset by translate."""
        local_half = self._half_extents * self.scale
        R          = self.rotate_matrix[:3, :3].T
        world_half = np.abs(R) @ local_half
        return (self.translate - world_half, self.translate + world_half)

    def sample(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate the SDF on world-space coordinates.

        Uses separate handling of scale to avoid shearing artifacts:
        1. Transform points by rigid (rotation + translation) inverse
        2. Apply inverse scale separately
        3. Correct distance by min(abs(scale))
        """
        scale     = self.scale
        min_scale = np.min(np.abs(scale))

        R         = self.rotate_matrix
        T         = np.eye(4)
        T[3, :3] = self.translate
        rigid_matrix   = R @ T
        rigid_inv      = matrix_inverse(rigid_matrix)[0]

        original_shape = X.shape
        points         = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        local_points   = matrix_point(points, rigid_inv)
        local_points /= scale

        Xt = local_points[:, 0].reshape(original_shape)
        Yt = local_points[:, 1].reshape(original_shape)
        Zt = local_points[:, 2].reshape(original_shape)

        return self.evaluate(Xt, Yt, Zt) * min_scale


class SDFCylinder(TransformData):
    """Capped cylinder SDF primitive inheriting transform properties from TransformData."""

    def __init__(
        self,
        radius:    float                                    = 0.5,
        height:    float                                    = 1.0,
        axis:      int                                      = 1,
        translate: Optional[Union[List[float], np.ndarray]] = None,
        rotate:    Optional[Union[List[float], np.ndarray]] = None,
        scale:     Optional[Union[List[float], np.ndarray]] = None,
        name:      str                                      = "cylinder",
    ) -> None:
        super().__init__(
            name      = name,
            translate = translate,
            rotate    = rotate,
            scale     = scale,
        )
        self._sdf_radius: float = radius
        self._height:     float = height
        self._axis:       int = axis
        self._field:      Optional["DMCField"] = None

    def _invalidate(self) -> None:
        """Notify parent field that this primitive has changed."""
        if self._field is not None:
            self._field._invalidate()

    @property
    def radius(self) -> float:
        """Cylinder radius."""
        return self._sdf_radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._sdf_radius = value
        self._invalidate()

    @property
    def height(self) -> float:
        """Total height of the cylinder."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = value
        self._invalidate()

    @property
    def axis(self) -> int:
        """Axis the cylinder is aligned with (0=X, 1=Y, 2=Z)."""
        return self._axis

    @axis.setter
    def axis(self, value: int) -> None:
        self._axis = value
        self._invalidate()

    def evaluate(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate the SDF on pre-transformed (local) coordinates."""
        return eval_cylinder(X, Y, Z, self._sdf_radius, self._height, self._axis)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cylinder AABB: compute local extent, rotate, take AABB of rotated box."""
        local_half = np.full(3, self._sdf_radius, dtype=np.float64)
        local_half[self._axis] = self._height / 2.0
        local_half = local_half * self.scale
        R          = self.rotate_matrix[:3, :3].T
        world_half = np.abs(R) @ local_half
        return (self.translate - world_half, self.translate + world_half)

    def sample(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate the SDF on world-space coordinates.

        Uses separate handling of scale to avoid shearing artifacts:
        1. Transform points by rigid (rotation + translation) inverse
        2. Apply inverse scale separately
        3. Correct distance by min(abs(scale))
        """
        scale     = self.scale
        min_scale = np.min(np.abs(scale))

        R         = self.rotate_matrix
        T         = np.eye(4)
        T[3, :3] = self.translate
        rigid_matrix   = R @ T
        rigid_inv      = matrix_inverse(rigid_matrix)[0]

        original_shape = X.shape
        points         = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        local_points   = matrix_point(points, rigid_inv)
        local_points /= scale

        Xt = local_points[:, 0].reshape(original_shape)
        Yt = local_points[:, 1].reshape(original_shape)
        Zt = local_points[:, 2].reshape(original_shape)

        return self.evaluate(Xt, Yt, Zt) * min_scale


# Type alias for SDF primitives
SDFPrimitive = Union[SDFSphere, SDFBox, SDFCylinder]


class CSGNode:
    """Internal node holding a primitive and its CSG operation."""

    __slots__ = ("primitive", "operation", "smoothing")

    def __init__(
        self,
        primitive: SDFPrimitive,
        operation: CSGOp,
        smoothing: float        = 0.0,
    ) -> None:
        self.primitive = primitive
        self.operation = operation
        self.smoothing = smoothing


class DMCField:
    """
    A field that combines SDF primitives via CSG operations.

    The field manages a list of primitives and their CSG operations,
    automatically recomputing the mesh when primitives change. Bounds are
    auto-computed from the registered primitives' bounding boxes with 10%
    padding.

    Resolution is specified as "subdivisions per world-space unit" in each
    axis. The actual grid shape is computed dynamically based on the
    auto-computed bounds.
    """

    def __init__(
        self,
        resolution: Union[int, Tuple[int, int, int]]         = (64, 64, 64),
        position:   Optional[Union[List[float], np.ndarray]] = None,
        iso_value:  float                                    = 0.0,
        name:       str                                      = "dmc_mesh",
    ) -> None:
        """
        Initialize a DMCField.

        Args:
            resolution: Subdivisions per world-space unit in each axis.
                Can be specified as a single integer (applied to all axes)
                or a tuple ``(rx, ry, rz)``. The actual grid shape is computed
                as ``ceil(extent[i] * resolution[i])`` for each axis, with a
                minimum of 2 points per axis. Default: ``64``.
            position: World-space offset applied to the auto-computed bounds.
                Default: ``[0, 0, 0]``.
            iso_value: Extraction threshold for the isosurface. Default: ``0.0``.
            name: Name for the resulting mesh. Default: ``"dmc_mesh"``.
        """
        self._resolution = (
            (resolution, resolution, resolution)
            if isinstance(resolution, int)
            else resolution
        )
        self._position: np.ndarray = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64)
            if position is None
            else np.asarray(position, dtype=np.float64)
        )
        self._iso_value = iso_value
        self._name      = name

        self._nodes:       List[CSGNode] = []
        self._dirty:       bool = True
        self._cached_mesh: Optional[MeshData] = None

    def _invalidate(self) -> None:
        """Mark the cached mesh as dirty."""
        self._dirty = True

    @property
    def resolution(self) -> Tuple[int, int, int]:
        """Subdivisions per world-space unit ``(rx, ry, rz)``."""
        return self._resolution

    @resolution.setter
    def resolution(self, value: Union[int, Tuple[int, int, int]]) -> None:
        self._resolution = (value, value, value) if isinstance(value, int) else value
        self._invalidate()

    @property
    def position(self) -> np.ndarray:
        """World-space offset of the field center."""
        return self._position

    @position.setter
    def position(self, value: Union[List[float], np.ndarray]) -> None:
        self._position = np.asarray(value, dtype=np.float64)
        self._invalidate()

    @property
    def iso_value(self) -> float:
        """Extraction threshold for the isosurface."""
        return self._iso_value

    @iso_value.setter
    def iso_value(self, value: float) -> None:
        self._iso_value = value
        self._invalidate()

    @property
    def name(self) -> str:
        """Mesh name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value
        self._invalidate()

    @property
    def primitives(self) -> List[SDFPrimitive]:
        """All registered primitives."""
        return [node.primitive for node in self._nodes]

    def _add_node(
        self, primitive: SDFPrimitive, operation: CSGOp, smoothing: float = 0.0
    ) -> "DMCField":
        """Internal method to add a CSG node."""
        primitive._field = self
        self._nodes.append(CSGNode(primitive, operation, smoothing))
        self._invalidate()
        return self

    def add(
        self, primitive: SDFPrimitive, smoothing: Optional[float] = None
    ) -> "DMCField":
        """
        Add a primitive via union (or smooth union if smoothing > 0).

        Args:
            primitive: The SDF primitive to add.
            smoothing: Blend radius for smooth union. If None or 0, uses hard union.

        Returns:
            Self for chaining.
        """
        if smoothing is not None and smoothing > 0:
            return self._add_node(primitive, CSGOp.SMOOTH_UNION, smoothing)
        return self._add_node(primitive, CSGOp.UNION)

    def subtract(self, primitive: SDFPrimitive) -> "DMCField":
        """
        Subtract a primitive via difference.

        Args:
            primitive: The SDF primitive to subtract.

        Returns:
            Self for chaining.
        """
        return self._add_node(primitive, CSGOp.DIFFERENCE)

    def intersect(self, primitive: SDFPrimitive) -> "DMCField":
        """
        Intersect with a primitive.

        Args:
            primitive: The SDF primitive to intersect with.

        Returns:
            Self for chaining.
        """
        return self._add_node(primitive, CSGOp.INTERSECTION)

    def remove(self, primitive: SDFPrimitive) -> "DMCField":
        """
        Remove a primitive from the field.

        Args:
            primitive: The SDF primitive to remove.

        Returns:
            Self for chaining.
        """
        self._nodes      = [node for node in self._nodes if node.primitive is not primitive]
        primitive._field = None
        self._invalidate()
        return self

    def _compute_effective_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Auto-compute bounds from primitives' bounding boxes with 10% padding."""
        if not self._nodes:
            return (
                np.array([-1.0, -1.0, -1.0], dtype=np.float64) + self._position,
                np.array([1.0, 1.0, 1.0], dtype=np.float64) + self._position,
            )

        all_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        all_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        for node in self._nodes:
            prim_min, prim_max = node.primitive.bounding_box()
            all_min = np.minimum(all_min, prim_min)
            all_max = np.maximum(all_max, prim_max)

        # Add 10% padding
        extent  = all_max - all_min
        padding = extent * 0.1
        all_min -= padding
        all_max += padding

        return (all_min + self._position, all_max + self._position)

    @property
    def mesh_data(self) -> MeshData:
        """
        The extracted mesh.

        Recomputes if dirty or not cached.
        """
        if not self._dirty and self._cached_mesh is not None:
            return self._cached_mesh

        if not self._nodes:
            self._cached_mesh = MeshData(
                indices = np.array([], dtype=np.int32),
                counts  = np.array([], dtype=np.int32),
                points  = np.zeros((0, 3), dtype=np.float64),
                name    = self._name,
            )
            self._dirty = False
            return self._cached_mesh

        # Build grid
        effective_bounds = self._compute_effective_bounds()
        min_bound, max_bound = effective_bounds
        extent = max_bound - min_bound
        shape = tuple(
            max(2, int(np.ceil(extent[i] * self._resolution[i]))) for i in range(3)
        )
        X, Y, Z, grid_origin, grid_spacing = make_grid(shape, effective_bounds)

        # Evaluate first primitive
        combined_sdf = self._nodes[0].primitive.sample(X, Y, Z)

        # Apply CSG operations in order
        for node in self._nodes[1:]:
            sdf = node.primitive.sample(X, Y, Z)

            if node.operation == CSGOp.UNION:
                combined_sdf = sdf_union(combined_sdf, sdf)
            elif node.operation == CSGOp.INTERSECTION:
                combined_sdf = sdf_intersection(combined_sdf, sdf)
            elif node.operation == CSGOp.DIFFERENCE:
                combined_sdf = sdf_difference(combined_sdf, sdf)
            elif node.operation == CSGOp.SMOOTH_UNION:
                combined_sdf = sdf_smooth_union(combined_sdf, sdf, node.smoothing)

        # Extract mesh
        self._cached_mesh = dual_marching_cubes(
            combined_sdf,
            iso_value    = self._iso_value,
            grid_origin  = grid_origin,
            grid_spacing = grid_spacing,
            name         = self._name,
        )
        self._dirty = False
        return self._cached_mesh

    def to_obj(self, filename: str) -> None:
        """
        Export the mesh to an OBJ file.

        Args:
            filename: Path to the output file.
        """
        self.mesh_data.save_obj(filename)


# =============================================================================
# GRID AND TRANSFORM UTILITIES
# =============================================================================


def make_grid(
    shape:  Tuple[int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular 3D sample grid.

    Args:
        shape: Grid resolution ``(nx, ny, nz)``.
        bounds: ``(min_bound, max_bound)`` arrays of length 3 defining the
            axis-aligned extent of the grid in world space.

    Returns:
        ``(X, Y, Z, grid_origin, grid_spacing)`` where ``X``, ``Y``, ``Z``
        are 3D coordinate arrays (shape *shape*) and ``grid_origin`` /
        ``grid_spacing`` are length-3 vectors suitable for
        :func:`dual_marching_cubes`.
    """
    min_bound = np.asarray(bounds[0], dtype=np.float64)
    max_bound = np.asarray(bounds[1], dtype=np.float64)

    x         = np.linspace(min_bound[0], max_bound[0], shape[0])
    y         = np.linspace(min_bound[1], max_bound[1], shape[1])
    z         = np.linspace(min_bound[2], max_bound[2], shape[2])

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    grid_origin  = min_bound.copy()
    grid_spacing = (max_bound - min_bound) / (np.array(shape, dtype=np.float64) - 1)

    return X, Y, Z, grid_origin, grid_spacing


def transform_points(
    X:           np.ndarray,
    Y:           np.ndarray,
    Z:           np.ndarray,
    *,
    translation: Optional[np.ndarray]               = None,
    rotation:    Optional[np.ndarray]               = None,
    scale:       Optional[Union[float, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map world-space grid points into local object space for SDF evaluation.

    Applies the **inverse** of the specified rigid transform so that
    axis-aligned SDF primitives (evaluated at the origin) appear translated,
    rotated, and scaled in world space.

    Args:
        X, Y, Z: 3D coordinate arrays (typically from :func:`make_grid`).
        translation: World-space position of the object centre ``[tx, ty, tz]``.
        rotation: Euler angles in degrees ``[rx, ry, rz]`` (XYZ order).
        scale: Scalar for uniform scale **or** length-3 array for non-uniform
            scale.  Non-uniform scale produces an *approximate* SDF (distances
            are distorted by the anisotropy); uniform scale is exact.

    Returns:
        ``(X', Y', Z')`` -- transformed coordinate arrays, same shape as inputs.
    """
    # Build the 4x4 transform matrix
    M = np.eye(4, dtype=np.float64)

    # Apply rotation (euler angles in degrees -> radians)
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64)
        if not np.allclose(rot, 0.0):
            M = euler_to_matrix(np.radians(rot))[0]

    # Apply scale to the rotation portion
    if scale is not None:
        s = np.asarray(scale, dtype=np.float64)
        if s.ndim == 0:
            M[:3, :3] *= s
        else:
            M[:3, :3] *= s

    # Apply translation
    if translation is not None:
        t = np.asarray(translation, dtype=np.float64)
        if not np.allclose(t, 0.0):
            M[3, :3] = t

    # Compute inverse transform
    M_inv = matrix_inverse(M)[0]

    # Flatten grid points, transform, and reshape
    original_shape = X.shape
    points         = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    local_points   = matrix_point(points, M_inv)

    Xo             = local_points[:, 0].reshape(original_shape)
    Yo             = local_points[:, 1].reshape(original_shape)
    Zo             = local_points[:, 2].reshape(original_shape)

    return Xo, Yo, Zo


# =============================================================================
# SDF EVALUATORS
# =============================================================================


def eval_sphere(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, radius: float = 1.0
) -> np.ndarray:
    """
    Evaluate a sphere SDF centered at the origin.

    Args:
        X, Y, Z: Coordinate arrays (any shape, must broadcast).
        radius: Sphere radius.

    Returns:
        SDF array -- negative inside, positive outside.
    """
    return np.sqrt(X**2 + Y**2 + Z**2) - radius


def eval_box(
    X:            np.ndarray,
    Y:            np.ndarray,
    Z:            np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """
    Evaluate an axis-aligned box SDF centered at the origin.

    Args:
        X, Y, Z: Coordinate arrays.
        half_extents: ``[hx, hy, hz]`` half-sizes along each axis.

    Returns:
        SDF array -- exact signed distance to the box surface.
    """
    half_extents = np.asarray(half_extents, dtype=np.float64)
    dx           = np.abs(X) - half_extents[0]
    dy           = np.abs(Y) - half_extents[1]
    dz           = np.abs(Z) - half_extents[2]

    outside = np.sqrt(
        np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2 + np.maximum(dz, 0) ** 2
    )
    inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0)
    return outside + inside


def eval_cylinder(
    X:      np.ndarray,
    Y:      np.ndarray,
    Z:      np.ndarray,
    radius: float      = 0.5,
    height: float      = 1.0,
    axis:   int        = 1,
) -> np.ndarray:
    """
    Evaluate a capped cylinder SDF centered at the origin.

    Args:
        X, Y, Z: Coordinate arrays.
        radius: Cylinder radius.
        height: Total height of the cylinder.
        axis: Axis the cylinder is aligned with (0=X, 1=Y, 2=Z).

    Returns:
        SDF array -- exact signed distance to the cylinder surface.
    """
    coords      = [X, Y, Z]
    half_height = height / 2.0

    radial_axes = [i for i in range(3) if i != axis]
    d_radial = (
        np.sqrt(coords[radial_axes[0]] ** 2 + coords[radial_axes[1]] ** 2) - radius
    )
    d_height = np.abs(coords[axis]) - half_height

    return np.sqrt(
        np.maximum(d_radial, 0) ** 2 + np.maximum(d_height, 0) ** 2
    ) + np.minimum(np.maximum(d_radial, d_height), 0)


# =============================================================================
# CSG OPERATIONS
# =============================================================================


def sdf_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean union (A union B).  Keeps both shapes."""
    return np.minimum(a, b)


def sdf_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean intersection (A intersect B).  Keeps only the overlap."""
    return np.maximum(a, b)


def sdf_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean difference (A \\ B).  Subtracts B from A."""
    return np.maximum(a, -b)


def sdf_smooth_union(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    """
    Smooth (filleted) union.

    Args:
        a, b: SDF arrays.
        k: Blending radius -- larger values give a wider fillet.
    """
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def sdf_smooth_intersection(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    """
    Smooth (filleted) intersection.

    Args:
        a, b: SDF arrays.
        k: Blending radius.
    """
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h + k * h * (1 - h)


def sdf_smooth_difference(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    """
    Smooth (filleted) difference -- subtracts B from A with blended edge.

    Args:
        a, b: SDF arrays.
        k: Blending radius.
    """
    return sdf_smooth_intersection(a, -b, k)


def box_sdf(
    shape:        Tuple[int, int, int],
    center:       Optional[np.ndarray]                    = None,
    half_extents: Optional[np.ndarray]                    = None,
    bounds:       Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a box signed distance field for testing.

    Args:
        shape: Grid resolution (nx, ny, nz).
        center: Center of the box [x, y, z]. Default: [0, 0, 0]
        half_extents: Half-size of the box [hx, hy, hz]. Default: [0.5, 0.5, 0.5]
        bounds: Tuple of (min_bound, max_bound) arrays defining the grid extent.
            Default: box centered with padding of 0.5 * max(half_extents)

    Returns:
        Tuple of (scalar_field, grid_origin, grid_spacing)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)

    if half_extents is None:
        half_extents = np.array([0.5, 0.5, 0.5])
    else:
        half_extents = np.asarray(half_extents, dtype=np.float64)

    if bounds is None:
        padding   = 0.5 * np.max(half_extents)
        min_bound = center - half_extents - padding
        max_bound = center + half_extents + padding
    else:
        min_bound, max_bound = bounds
        min_bound = np.asarray(min_bound, dtype=np.float64)
        max_bound = np.asarray(max_bound, dtype=np.float64)

    # Create grid
    x = np.linspace(min_bound[0], max_bound[0], shape[0])
    y = np.linspace(min_bound[1], max_bound[1], shape[1])
    z = np.linspace(min_bound[2], max_bound[2], shape[2])

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Compute SDF (box SDF formula)
    qx = np.abs(X - center[0]) - half_extents[0]
    qy = np.abs(Y - center[1]) - half_extents[1]
    qz = np.abs(Z - center[2]) - half_extents[2]

    # Distance outside the box
    outside = np.sqrt(
        np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2
    )

    # Distance inside the box (negative)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)

    sdf    = outside + inside

    # Compute grid parameters
    grid_origin  = min_bound.copy()
    grid_spacing = (max_bound - min_bound) / (np.array(shape) - 1)

    return sdf, grid_origin, grid_spacing


def cylinder_sdf(
    shape:  Tuple[int, int, int],
    center: Optional[np.ndarray]                    = None,
    radius: float                                   = 0.5,
    height: float                                   = 1.0,
    axis:   int                                     = 1,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a capped cylinder signed distance field for testing.

    Args:
        shape: Grid resolution (nx, ny, nz).
        center: Center of the cylinder [x, y, z]. Default: [0, 0, 0]
        radius: Radius of the cylinder. Default: 0.5
        height: Total height of the cylinder. Default: 1.0
        axis: Axis the cylinder is aligned along (0=X, 1=Y, 2=Z). Default: 1
        bounds: Tuple of (min_bound, max_bound) arrays defining the grid extent.
            Default: cylinder centered with padding of 0.5 * radius

    Returns:
        Tuple of (scalar_field, grid_origin, grid_spacing)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)

    half_height = height / 2.0

    if bounds is None:
        extent = np.full(3, radius)
        extent[axis] = half_height
        padding   = 0.5 * radius
        min_bound = center - extent - padding
        max_bound = center + extent + padding
    else:
        min_bound, max_bound = bounds
        min_bound = np.asarray(min_bound, dtype=np.float64)
        max_bound = np.asarray(max_bound, dtype=np.float64)

    # Create grid
    x = np.linspace(min_bound[0], max_bound[0], shape[0])
    y = np.linspace(min_bound[1], max_bound[1], shape[1])
    z = np.linspace(min_bound[2], max_bound[2], shape[2])

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coords = [X, Y, Z]

    # Radial distance on the two non-axis dimensions
    radial_axes = [i for i in range(3) if i != axis]
    d_radial = (
        np.sqrt(
            (coords[radial_axes[0]] - center[radial_axes[0]]) ** 2
            + (coords[radial_axes[1]] - center[radial_axes[1]]) ** 2
        )
        - radius
    )

    # Height distance along the cylinder axis
    d_height = np.abs(coords[axis] - center[axis]) - half_height

    # Capped cylinder SDF
    sdf = np.sqrt(
        np.maximum(d_radial, 0) ** 2 + np.maximum(d_height, 0) ** 2
    ) + np.minimum(np.maximum(d_radial, d_height), 0)

    # Compute grid parameters
    grid_origin  = min_bound.copy()
    grid_spacing = (max_bound - min_bound) / (np.array(shape) - 1)

    return sdf, grid_origin, grid_spacing