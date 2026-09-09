"""
Bounding Volume Hierarchy (BVH) for fast ray-mesh intersection.

Build is pure numpy (one-shot, amortizes over many ray queries) using a
top-down median split on the longest AABB axis with a small leaf size
(default 8 faces).  Traversal helpers live in ``_bilinear.py`` and reuse
the existing ``_ray_aabb_intersect`` slab test.

Layout (six small parallel arrays):

    node_aabb_min:  (N, 3)  float64   per-node min corner
    node_aabb_max:  (N, 3)  float64   per-node max corner
    node_left:      (N,)    int32     left-child index, or -1 for leaves
    node_right:     (N,)    int32     right-child index, or -1 for leaves
    node_first:     (N,)    int32     first face slot (in face_perm) for leaves
    node_count:     (N,)    int32     number of faces in this leaf (0 = internal)
    face_perm:      (F,)    int32     permutation of original face indices,
                                      grouped contiguously by leaf

The root node is always index 0.  ``node_count[i] > 0`` indicates a leaf;
``node_left[i] >= 0`` indicates an internal node (children are mutually
exclusive with face range).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BVHData(NamedTuple):
    """Flat parallel-array BVH ready to hand to a numba traversal kernel."""

    node_aabb_min: np.ndarray  # (N, 3) float64
    node_aabb_max: np.ndarray  # (N, 3) float64
    node_left:     np.ndarray  # (N,)   int32   -1 for leaves
    node_right:    np.ndarray  # (N,)   int32   -1 for leaves
    node_first:    np.ndarray  # (N,)   int32   first face slot for leaves
    node_count:    np.ndarray  # (N,)   int32   0 for internal nodes
    face_perm:     np.ndarray  # (F,)   int32   permutation of face ids


def build_bvh(
    aabb_min:  np.ndarray,
    aabb_max:  np.ndarray,
    leaf_size: int        = 8,
) -> BVHData:
    """Build a top-down median-split BVH over per-face AABBs.

    Args:
        aabb_min: ``(F, 3)`` per-face AABB minimum corner.
        aabb_max: ``(F, 3)`` per-face AABB maximum corner.
        leaf_size: maximum number of faces in a leaf node.  Smaller leaves
            mean a deeper tree (more traversal overhead) and larger leaves
            mean more linear face tests at each leaf.  8 is a balanced
            default for typical mesh sizes.

    Returns:
        :class:`BVHData` -- flat parallel arrays.  Root is node 0.

    Raises:
        ValueError: if input AABB arrays are malformed or empty.
    """

    aabb_min = np.ascontiguousarray(aabb_min, dtype=np.float64)
    aabb_max = np.ascontiguousarray(aabb_max, dtype=np.float64)

    if aabb_min.shape != aabb_max.shape:
        raise ValueError(
            f"aabb_min shape {aabb_min.shape} != aabb_max shape {aabb_max.shape}"
        )
    if aabb_min.ndim != 2 or aabb_min.shape[1] != 3:
        raise ValueError(f"aabb_min must be (F, 3), got {aabb_min.shape}")
    if aabb_min.shape[0] == 0:
        raise ValueError("cannot build a BVH over zero faces")
    leaf_size = max(1, int(leaf_size))

    n_faces   = aabb_min.shape[0]
    centroids = 0.5 * (aabb_min + aabb_max)  # (F, 3) for split heuristic

    # Working storage for the to-be-built node arrays.  We reserve worst-case
    # size (2 * F) and trim at the end.  A balanced binary tree over F leaves
    # of size 1 has at most 2*F - 1 nodes; with leaf_size > 1 it is fewer.
    max_nodes     = 2 * n_faces
    node_aabb_min = np.empty((max_nodes, 3), dtype=np.float64)
    node_aabb_max = np.empty((max_nodes, 3), dtype=np.float64)
    node_left     = np.full(max_nodes, -1, dtype=np.int32)
    node_right    = np.full(max_nodes, -1, dtype=np.int32)
    node_first    = np.zeros(max_nodes, dtype=np.int32)
    node_count    = np.zeros(max_nodes, dtype=np.int32)
    face_perm     = np.arange(n_faces, dtype=np.int32)

    # Simple iterative stack-driven build.  Each work item describes the
    # contiguous face range [first, first + count) in ``face_perm`` that
    # must be turned into the node at ``node_id``.
    next_node_id = 1  # 0 is root, allocate as we go
    work: list = [(0, 0, n_faces)]  # (node_id, first, count)

    while work:
        node_id, first, count = work.pop()
        face_ids = face_perm[first : first + count]

        # Bounds for this node: union of the contained per-face AABBs.
        sub_min                = aabb_min[face_ids]
        sub_max                = aabb_max[face_ids]
        node_aabb_min[node_id] = sub_min.min(axis=0)
        node_aabb_max[node_id] = sub_max.max(axis=0)

        if count <= leaf_size:
            # Leaf: store face range and stop.
            node_first[node_id] = first
            node_count[node_id] = count
            continue

        # Internal: split along the longest axis at the median centroid.
        cents = centroids[face_ids]
        spans = cents.max(axis=0) - cents.min(axis=0)
        axis  = int(np.argmax(spans))

        # If all centroids collapse onto a single point along every axis we
        # cannot make a useful split - fall back to a leaf even past the
        # leaf_size cap.  Avoids infinite recursion on degenerate input.
        if spans[axis] <= 0.0:
            node_first[node_id] = first
            node_count[node_id] = count
            continue

        order      = np.argsort(cents[:, axis], kind="stable")
        sorted_ids = face_ids[order]
        face_perm[first : first + count] = sorted_ids
        mid = count // 2

        left_id  = next_node_id
        right_id = next_node_id + 1
        next_node_id += 2
        node_left[node_id]  = left_id
        node_right[node_id] = right_id

        # Push right first so left is processed next - keeps cache locality
        # on the centroid array a touch better, no functional impact.
        work.append((right_id, first + mid, count - mid))
        work.append((left_id, first, mid))

    # Trim to actually used range.
    return BVHData(
        node_aabb_min = np.ascontiguousarray(node_aabb_min[:next_node_id]),
        node_aabb_max = np.ascontiguousarray(node_aabb_max[:next_node_id]),
        node_left     = np.ascontiguousarray(node_left[:next_node_id]),
        node_right    = np.ascontiguousarray(node_right[:next_node_id]),
        node_first    = np.ascontiguousarray(node_first[:next_node_id]),
        node_count    = np.ascontiguousarray(node_count[:next_node_id]),
        face_perm     = np.ascontiguousarray(face_perm),
    )