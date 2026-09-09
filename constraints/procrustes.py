from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from cgmath.geometry._base import Data
from transforms import matrix_inverse, matrix_multiply, matrix_to_euler


@dataclass(repr=False, eq=False)
class ProcrustesData(Data):
    points:     np.ndarray = None
    transforms: np.ndarray = None
    clusters:   np.ndarray = None

    # --- cached attributes --- #
    _scale        = None  # scale factors
    _rotate       = None  # euler angles in radians
    _translate    = None  # translation vectors
    _matrix       = None  # output matrices
    _points       = None  # the updated points
    _scale_offset = None  # whether to apply scale to offset matrix

    @property
    def scale_offset(self):
        if self._scale_offset is None:
            self._scale_offset = True
        return self._scale_offset

    @scale_offset.setter
    def scale_offset(self, state: bool):
        if self._scale_offset != state:
            self._scale_offset = state
            self.compute()

    @property
    def scale(self):
        return np.ones((self._scale.size, 3)) * self._scale[:, None]

    @property
    def rotate(self):
        return self._rotate

    @property
    def translate(self):
        return self._translate

    @property
    def matrix(self):
        return self._matrix

    def __init__(self, target: np.ndarray):
        """Initialize the procrustes data using a target mesh"""

        # be nice and support native dataclasses (eg: MeshData)
        if not isinstance(target, np.ndarray) and hasattr(target, "points"):
            self.points = target.points.copy()
        else:
            self.points = np.asarray(target)

    def attach(self, transform: np.ndarray, indices: np.ndarray | None = None):
        """Attach a 4x4 transform matrix to a specified cluster of point indices"""

        # if indices is None, then use all points
        if indices is None:
            indices = np.arange(self.points.shape[0])

        # make sure indices are unique
        else:
            indices = np.unique(indices).astype(np.intp)
            indices = indices[indices >= 0]  # get rid of -1's

        # store inside self.matrix
        if self.transforms is None:
            self.transforms = transform[None]
        else:
            self.transforms = np.concatenate((self.transforms, [transform]))

        # rebuild neighborhood matrix
        if self.clusters is None:
            self.clusters = indices[None]
        else:
            max_length = self.clusters.shape[1]
            if indices.size > self.clusters.shape[1]:
                max_length = indices.size

            clusters = np.ones((self.clusters.shape[0] + 1, max_length), dtype=int) * -1
            clusters[: self.clusters.shape[0], : self.clusters.shape[1]] = self.clusters
            clusters[-1, : indices.size] = indices
            self.clusters = clusters

    def update(self, target: np.ndarray) -> None:
        """updates procrustes points and triggers compute"""

        # be nice and support native dataclasses (eg: MeshData)
        if not isinstance(target, np.ndarray) and hasattr(target, "points"):
            self._points = target.points
        else:
            self._points = np.asarray(target)

        self.compute()

    @property
    def valid(self):
        """a valid constraint has the same count of clusters and transforms"""
        if self.clusters is not None and self.transforms is not None:
            return self.clusters.shape[0] == self.transforms.shape[0]
        return False

    def compute(self) -> None:
        """compute procrustes"""

        # compute only if
        if self.valid:
            clusters = self.clusters
            points0  = self.points
            points1  = self._points
            if points1 is None:
                points1 = points0

            # pad 0's at end of matrix stack to handle -1 neighbor elements
            points0_ = np.zeros((points0.shape[0] + 1, points0.shape[1]))
            points0_[:-1] = points0
            points0_ = points0_[clusters]

            points1_ = np.zeros((points1.shape[0] + 1, points1.shape[1]))
            points1_[:-1] = points1
            points1_ = points1_[clusters]

            # get the actual count of each clusters
            counts = (clusters > -1).sum(axis=1)

            # compute the centroids using the clusters counts
            centroid0 = np.sum(points0_, axis=1) / counts[:, None]
            centroid1 = np.sum(points1_, axis=1) / counts[:, None]

            vectors0  = points0_ - centroid0[:, None]
            vectors0[clusters == -1] = 0  # so outer product ignores the -1's
            vectors1 = points1_ - centroid1[:, None]
            vectors1[clusters == -1] = 0  # so outer product ignores the -1's

            # sum the vectorized outer products
            H = np.einsum("bji,bjk->bik", vectors1, vectors0)

            # rotation factor of H -- try the parallel numba polar
            # decomposition first (much faster for many small clusters
            # because it skips per-call np.linalg.svd dispatch),
            # fall back to vectorized SVD if the kernel is unavailable.
            R = self._batch_rotation(H)

            # compute scale
            sx = np.sqrt(np.einsum("...ij,...ij", vectors0, vectors0)) / counts
            sy = np.sqrt(np.einsum("...ij,...ij", vectors1, vectors1)) / counts
            S  = sy / sx

            # apply scale to matrix if desired
            if self.scale_offset:
                R *= S[:, None, None]

            # compute new position using the scaled matrices
            T = np.einsum("ijk,ki->ji", -R, centroid0.T).T + centroid1
            p = np.einsum("...ij,...j->...i", R, self.transforms[:, 3, :3])
            p = p + T

            # Embed the 3x3 rotation into a 4x4 homogeneous matrix
            R4 = np.zeros((R.shape[0], 4, 4))
            R4[:, :3, :3] = R
            R4[:, 3, 3] = 1.0

            # compute new transform orientations
            R4 = matrix_inverse(R4)
            M  = matrix_multiply(self.transforms, R4)
            M[:, 3, :3] = p

            # set internals
            self._scale     = S
            self._rotate    = matrix_to_euler(M)
            self._translate = p
            self._matrix    = M

        else:
            raise RuntimeError("Cannot compute invalid ProcrustesData object")

    @staticmethod
    def _determinants(H: np.ndarray) -> np.ndarray:
        """Determinants of a batch of 3x3 matrices, by cofactor expansion.

        ``np.linalg.det`` goes through LAPACK and costs ~11 ms on 100k 3x3
        matrices against ~3 ms here, which matters because this runs on every
        batch purely to decide which rows may take the fast path.
        """
        return (
            H[:, 0, 0] * (H[:, 1, 1] * H[:, 2, 2] - H[:, 1, 2] * H[:, 2, 1])
            - H[:, 0, 1] * (H[:, 1, 0] * H[:, 2, 2] - H[:, 1, 2] * H[:, 2, 0])
            + H[:, 0, 2] * (H[:, 1, 0] * H[:, 2, 1] - H[:, 1, 1] * H[:, 2, 0])
        )

    @staticmethod
    def _svd_rotation(H: np.ndarray) -> np.ndarray:
        """Kabsch rotation: maximises ``trace(R.T @ H)``, reflections included.

        For a reflection (``det(H) < 0``) the unconstrained optimum is itself
        a reflection, so the smallest singular direction is flipped; the
        resulting objective is ``s0 + s1 - s2``.
        """
        U, S, V = np.linalg.svd(H)
        R    = np.einsum("bji,bkj->bki", V, U)

        refl = np.where(np.linalg.det(R) < 0)[0]
        if refl.size:
            V = V.copy()  # np.linalg.svd's output is not ours to mutate
            V[refl, 2, :] *= -1
            R[refl] = np.einsum("...ji,...kj->...ki", V[refl], U[refl])
        return R

    @staticmethod
    def _batch_rotation(H: np.ndarray) -> np.ndarray:
        """Per-cluster rotation factor of a batch of 3x3 covariance matrices.

        Clusters with ``det(H) >= 0`` go through the parallel numba polar
        decomposition in ``cgmath.geometry.utils.main``, which avoids the
        per-call ``np.linalg.svd`` dispatch overhead. Measured on 100k
        clusters: 5.0 ms against 164 ms for pure Kabsch when nothing is
        reflected (33x), falling to 1.8x on a batch that is half reflected
        because those rows still take the SVD branch.

        The polar factor is the Procrustes optimum only when the covariance
        is not a reflection, so ``det(H) < 0`` clusters go through Kabsch
        instead. On those the polar iteration converges to a rotation that
        is orthonormal with ``det == +1`` -- so nothing downstream can tell
        it apart -- but does not maximise ``trace(R.T @ H)``; on random
        reflected covariances it averages 1.41 against Kabsch's 3.46.

        Falls back to Kabsch for the whole batch when numba is unavailable.
        """
        H = np.ascontiguousarray(np.asarray(H, dtype=np.float64))

        try:
            from cgmath.geometry.utils.main import batch_procrustes_rotations
        except ImportError:
            return ProcrustesData._svd_rotation(H)

        reflected = ProcrustesData._determinants(H) < 0.0
        if not reflected.any():
            return batch_procrustes_rotations(H, max_iter=16)
        if reflected.all():
            return ProcrustesData._svd_rotation(H)

        R = np.empty_like(H)
        R[~reflected] = batch_procrustes_rotations(H[~reflected], max_iter=16)
        R[reflected] = ProcrustesData._svd_rotation(H[reflected])
        return R