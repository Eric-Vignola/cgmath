"""
Numba kernels for face-varying normal helpers.

These power ``MeshData.set_normals`` (hard-edge splitting) and
``MeshData.get_tangent_space`` (triangulated face-vertex expansion).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _split_at_hard_edges(
    indices:   np.ndarray,
    counts:    np.ndarray,
    e2v:       np.ndarray,
    e2f:       np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    """Compute face-varying normal slot indices via union-find.

    Two face-vertex positions that reference the same mesh vertex are
    unioned when they share a *smooth* edge.  The resulting connected
    components give the face-varying normal indices.

    The original Python implementation built a ``(face, vertex) -> fv_pos``
    dict in a double Python loop, then walked the edge list with dict
    lookups. This kernel replaces the dict with a sorted ``key`` array
    (where ``key = face * n_verts + vid``) and uses ``np.searchsorted``
    inside an inlined union-find with path halving.

    NOT marked ``parallel=True`` -- the union-find ``parent`` array has
    write-write races across edges that share a face-vertex slot, so
    this is intentionally sequential.

    Args:
        indices: ``(n_fv,)`` flat face-vertex stream.
        counts: ``(n_faces,)`` vertex count per face.
        e2v: ``(n_edges, 2)`` edge -> vertex pair.
        e2f: ``(n_edges, k)`` edge -> faces (k >= 1; -1 padded).
        hard_mask: ``(n_edges,) bool`` -- True at hard-edge slots.

    Returns:
        ``(n_fv,) int64`` root id per face-vertex position. The caller
        runs ``np.unique(roots, return_inverse=True)`` to compress to
        contiguous slot ids.
    """
    n_fv    = indices.size
    n_faces = counts.size
    n_verts = 0
    for i in range(n_fv):
        if indices[i] > n_verts:
            n_verts = indices[i]
    n_verts += 1  # vertex ids are 0..max, so n_verts = max + 1

    # Build (face * n_verts + vid) keys for each face-vertex position.
    keys = np.empty(n_fv, dtype=np.int64)
    pos  = 0
    for fi in range(n_faces):
        for _li in range(counts[fi]):
            keys[pos] = np.int64(fi) * np.int64(n_verts) + np.int64(indices[pos])
            pos += 1

    order        = np.argsort(keys)
    sorted_keys  = keys[order]

    parent       = np.arange(n_fv, dtype=np.int64)

    n_edges      = e2f.shape[0]
    n_face_slots = e2f.shape[1]
    for ei in range(n_edges):
        if hard_mask[ei]:
            continue

        # Need at least two non-negative incident faces to define a shared
        # smooth edge. Edges with k > 2 incident faces (non-manifold) are
        # only handled for the first two slots -- matches the prior dict
        # behavior which only looked at e2f[:, 0] and e2f[:, 1].
        f0 = -1
        f1 = -1
        for s in range(n_face_slots):
            v = e2f[ei, s]
            if v < 0:
                continue
            if f0 < 0:
                f0 = v
            else:
                f1 = v
                break
        if f0 < 0 or f1 < 0:
            continue

        v0 = e2v[ei, 0]
        v1 = e2v[ei, 1]
        for vid in (v0, v1):
            if vid < 0:
                continue
            k0 = np.int64(f0) * np.int64(n_verts) + np.int64(vid)
            k1 = np.int64(f1) * np.int64(n_verts) + np.int64(vid)
            i0 = np.searchsorted(sorted_keys, k0)
            i1 = np.searchsorted(sorted_keys, k1)
            if i0 >= n_fv or sorted_keys[i0] != k0:
                continue
            if i1 >= n_fv or sorted_keys[i1] != k1:
                continue
            a = order[i0]
            b = order[i1]

            # union-find with path halving
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                if a < b:
                    parent[b] = a
                else:
                    parent[a] = b

    # Final root resolution (full path compression).
    roots = np.empty(n_fv, dtype=np.int64)
    for i in range(n_fv):
        j = i
        while parent[j] != j:
            j = parent[j]
        roots[i] = j

    return roots


@njit(cache=True)
def _build_tri_expand_map(counts: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """Map original face-vertex stream positions to triangulated positions.

    For a tri (count == 3) emits ``[pos, pos+1, pos+2]``.
    For a quad (count == 4):
      * rule 0 (split 0-2): emits ``[pos+1, pos+2, pos, pos, pos+2, pos+3]``
      * rule 1 (split 1-3): emits ``[pos, pos+1, pos+3, pos+3, pos+1, pos+2]``

    n-gons are not supported (the caller asserts this via
    ``_require_no_ngons``).

    Args:
        counts: ``(n_faces,)`` vertex count per face. Each entry must be 3 or 4.
        rules: ``(n_faces,)`` quad split rule (only consulted when count == 4).

    Returns:
        ``(n_tri_fv,) int64`` mapping each triangulated face-vertex back to
        its original face-vertex position.
    """
    n_faces = counts.size

    # First pass: total triangulated face-vertex count.
    n_out = 0
    for fi in range(n_faces):
        if counts[fi] == 3:
            n_out += 3
        else:  # 4
            n_out += 6

    out   = np.empty(n_out, dtype=np.int64)
    pos   = 0
    write = 0
    for fi in range(n_faces):
        c = counts[fi]
        if c == 3:
            out[write] = pos
            out[write + 1] = pos + 1
            out[write + 2] = pos + 2
            write += 3
        else:  # c == 4
            r = rules[fi]
            if r == 0:
                # split 0-2: (1,2,0) + (0,2,3)
                out[write] = pos + 1
                out[write + 1] = pos + 2
                out[write + 2] = pos
                out[write + 3] = pos
                out[write + 4] = pos + 2
                out[write + 5] = pos + 3
            else:
                # split 1-3: (0,1,3) + (3,1,2)
                out[write] = pos
                out[write + 1] = pos + 1
                out[write + 2] = pos + 3
                out[write + 3] = pos + 3
                out[write + 4] = pos + 1
                out[write + 5] = pos + 2
            write += 6
        pos += c

    return out