from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cgmath.geometry._base import Data
from cgmath.geometry.mesh import MeshData
from cgmath.geometry.utils.main import (
    compute_neighbor_distances,
    compute_topological_neighborhood,
)
from cgmath.rbf._kernels import get_kernel

# Kernels whose support radius is controlled by the `r` parameter
_COMPACT_KERNELS = frozenset(
    {
        "wendland_c0",
        "wendland_c2",
        "wendland_c4",
        "wendland_c6",
        "wu_c2",
        "wu_c4",
        "bump",
    }
)


def cdist_euclidean(X, Y):
    """
    Compute the Euclidean distance between each pair of rows in X and Y.
    Numba implementation is magnitudes faster than scipy.spatial.distance.cdist
    on poin clouds with under 10**5 points.

    Parameters
    ----------
    X : ndarray of shape (m, k)
    Y : ndarray of shape (n, k)

    Returns
    -------
    distances : ndarray of shape (m, n)
        distances[i, j] is the Euclidean distance between X[i] and Y[j]
    """
    from cgmath.geometry.utils._numba._wrap import _cdist_euclidean

    return _cdist_euclidean(X, Y)


@dataclass(repr=False, eq=False)
class WrapData(Data):
    geodesic_radius: float | None  # max geodesic distance for vertex connectivity
    radius:          float | None  # kernel support radius for compact kernels

    src_points:      np.ndarray    # source control points (Nx3)
    dst_points:      np.ndarray    # destination control points (Nx3)

    conn_matrix:     np.ndarray    # face-vertex neighbor connectivity (from MeshData)
    conn_distances:  np.ndarray    # edge lengths for conn_matrix neighbors

    # --- cached attributes --- #
    _system_matrix:   np.ndarray   # augmented [K, 1, P] interpolation matrix
    _distance_matrix: np.ndarray   # pairwise Euclidean distances between source points
    _weights:         np.ndarray   # solved RBF weight matrix (system_pinv @ target_matrix)

    _system_pinv:     np.ndarray   # pseudo-inverse of system_matrix
    _target_matrix:   np.ndarray   # augmented target-point RHS column
    _dirty:           bool = True  # True when the system matrix needs recomputation

    # Deliberately unannotated, so it stays out of ``fields()`` and therefore
    # out of ``to_dict()`` and everything built on it. ``_dirty`` is a declared
    # field, so its name and meaning are part of the serialized schema and
    # cannot be repurposed.
    _target_dirty = True  # True when only the right-hand side needs re-solving

    name:        str = "wrap_data1"         # identifier for this wrap instance
    kernel_name: str = "thin_plate_spline"  # active RBF kernel name

    def __init__(self, kernel: str = "thin_plate_spline", name: str = "wrap_data1"):
        """Initialize the wrap data using a target mesh"""

        self.name            = name
        self.kernel_name     = kernel
        self._system_matrix  = None
        self._target_matrix  = None

        self._system_pinv    = None
        self.geodesic_radius = None
        self.radius          = None

        self.src_points      = None
        self.dst_points      = None
        self._weights        = None

        self._dirty          = True
        self._target_dirty   = True
        self.conn_matrix     = None
        self.conn_distances  = None

    def _effective_radius(self) -> float | None:
        """Resolve the kernel support radius.

        Returns ``radius`` if explicitly set, otherwise falls back to
        ``geodesic_radius`` for backward compatibility, or ``None`` if
        neither is set.
        """
        if self.radius is not None:
            return self.radius
        return self.geodesic_radius

    def _get_points(self, target):
        if isinstance(target, MeshData):
            return target.points.copy()
        return np.asarray(target)

    def _eval_kernel(self, cdist):
        """Evaluate kernel on distance matrix with appropriate parameters."""
        kernel_fn = get_kernel(self.kernel_name)
        if self.kernel_name in _COMPACT_KERNELS:
            r = self._effective_radius()
            if r is not None:
                return kernel_fn(cdist, r=r)
        return kernel_fn(cdist)

    def _build_kernel_matrix(self, cdist):
        """Apply kernel to distance matrix and zero out non-finite entries."""
        K = self._eval_kernel(cdist)
        K[~np.isfinite(K)] = 0.0
        return K

    def _build_system(self):
        """Build the augmented interpolation matrix and cache LU factorization."""
        identity = np.ones((self.src_points.shape[0], 1))
        self._system_matrix = np.block(
            [
                [
                    self._build_kernel_matrix(self._distance_matrix),
                    identity,
                    self.src_points,
                ],
                [identity.T, np.zeros((1, 1)), np.zeros((1, 3))],
                [self.src_points.T, np.zeros((3, 1)), np.zeros((3, 3))],
            ]
        )

        # silence finnicky matmult warnings
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self._system_pinv = np.linalg.pinv(self._system_matrix)

    def set_kernel(self, kernel: str):
        """set the kernel function"""
        self.kernel_name = kernel
        self._dirty      = True

    def set_source(self, target: np.ndarray | MeshData) -> None:
        """set source points

        Parameters
        ----------
        target : np.ndarray or MeshData
            Source control points. If MeshData and geodesic_radius property
            is set, geodesic surface distance determines vertex pair
            connectivity.
        """
        self.src_points = self._get_points(target)

        if isinstance(target, MeshData):
            self.conn_matrix = target.get_face_vertex_neighbors()
            self.conn_distances = compute_neighbor_distances(
                self.conn_matrix, target.points
            )
        else:
            self.conn_matrix    = None
            self.conn_distances = None

        self._dirty = True

    def set_target(self, target: np.ndarray) -> None:
        """set target points

        Only the right-hand side of the interpolation system moves, so this
        does not invalidate the kernel matrix or its pseudo-inverse.
        """
        self.dst_points    = self._get_points(target)
        self._target_dirty = True

    def set_geodesic_radius(self, value: float | None) -> None:
        """Set the geodesic radius for vertex pair connectivity.

        Parameters
        ----------
        value : float or None
            Maximum geodesic surface distance for vertex pair
            connectivity. Set to None to disable geodesic filtering.
        """
        self.geodesic_radius = value
        self._dirty          = True

    def set_radius(self, value: float | None) -> None:
        """Set the support radius for compactly supported kernels.

        Parameters
        ----------
        value : float or None
            Kernel support radius. Points beyond this distance receive
            zero kernel weight. Set to None to use the default behavior.
        """
        self.radius = value
        self._dirty = True

    def _rebuild(self) -> None:
        """Consolidate all deferred computation."""
        if self.src_points is None:
            raise ValueError("Source points not set. Call set_source() first.")
        if self.dst_points is None:
            raise ValueError("Target points not set. Call set_target() first.")

        # The distance matrix and the pseudo-inverse depend only on the source
        # points, the kernel and the radii. Moving the target changes the
        # right-hand side alone, and re-solving that is an O(N^2) matmul rather
        # than another O(N^3) pinv -- the difference between a 7 ms and a 6.9 s
        # update on a 2500-point cage.
        if self._dirty:
            if self.conn_matrix is not None and self.geodesic_radius is not None:
                neighborhood, _ = compute_topological_neighborhood(
                    self.conn_matrix,
                    distances       = self.conn_distances,
                    max_distance    = self.geodesic_radius,
                    distance_format = "sparse",
                )
                self._distance_matrix = cdist_euclidean(
                    self.src_points, self.src_points
                )
                n            = len(self.src_points)
                valid        = neighborhood >= 0
                row_idx      = np.where(valid)[0]
                neighbor_idx = neighborhood[valid]
                reachable    = np.zeros((n, n), dtype=bool)
                reachable[row_idx, neighbor_idx] = True
                np.fill_diagonal(reachable, True)
                self._distance_matrix[~reachable] = np.inf
            else:
                self._distance_matrix = cdist_euclidean(
                    self.src_points, self.src_points
                )

            self._build_system()

        self._target_matrix = np.block(
            [[self.dst_points], [np.zeros((1, 3))], [np.zeros((3, 3))]]
        )

        # silence finnicky matmult warnings
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self._weights = self._system_pinv @ self._target_matrix

        self._dirty        = False
        self._target_dirty = False

    def _warn_configuration(self) -> None:
        """Emit warnings for potentially problematic configurations."""
        import warnings

        if (
            self.geodesic_radius is not None
            and self.kernel_name not in _COMPACT_KERNELS
        ):
            warnings.warn(
                f"Using geodesic_radius with '{self.kernel_name}' kernel can "
                "cause ill-conditioning. Consider using a compactly supported "
                "kernel (e.g., 'wendland_c2') for geodesic RBF.",
                UserWarning,
                stacklevel=2,
            )

        r = self._effective_radius()
        if (
            r is not None
            and self.geodesic_radius is not None
            and r > self.geodesic_radius
        ):
            warnings.warn(
                f"radius ({r}) > geodesic_radius ({self.geodesic_radius}). "
                "The topology lookup will exclude neighbors that have "
                "non-zero kernel contributions, producing incorrect results. "
                "Set geodesic_radius >= radius.",
                UserWarning,
                stacklevel=2,
            )

    def deform(self, target: np.ndarray) -> np.ndarray:
        """deform the source points to the target points"""
        if self._dirty or self._target_dirty:
            self._rebuild()

        self._warn_configuration()
        points   = self._get_points(target)

        identity = np.ones((points.shape[0], 1))
        result   = cdist_euclidean(points, self.src_points)
        result   = self._build_kernel_matrix(result)
        h        = np.block([[result, identity, points]])

        # return with divide-by-zero/overflow warnings silenced
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return h @ self._weights