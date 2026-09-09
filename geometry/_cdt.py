"""
Constrained Delaunay Triangulation (CDT) for arbitrary n-gon faces.

Implements CDT with Maya-compatible cocircular tiebreaking and flood-fill
triangle removal for polygons with optional holes.

Algorithm:
    1. scipy Delaunay -> enforce boundary constraints (Sloan's algorithm)
    2. Restore Delaunay property via edge flipping
    3. Cocircular tiebreak: prefer longer diagonal (Maya compatibility)
    4. Remove exterior and hole triangles via flood fill

All Numba kernels are imported lazily from ``utils._numba._cdt``.
"""

from __future__ import annotations

import warnings
from collections import defaultdict, deque

import numpy as np


# ---------------------------------------------------------------------------
# Edge key utility
# ---------------------------------------------------------------------------


def _ek(a, b):
    """Canonical (sorted) edge key for use as dict keys."""
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------------------
# Boundary edge builder
# ---------------------------------------------------------------------------


def get_boundary_edges(boundary):
    """Return the set of canonical edge keys for a closed vertex loop.

    Args:
        boundary: Sequence of vertex indices forming a closed polygon loop.

    Returns:
        Set of ``(min, max)`` edge tuples.
    """
    edges = set()
    n     = len(boundary)
    for i in range(n):
        a, b = boundary[i], boundary[(i + 1) % n]
        edges.add((min(a, b), max(a, b)))
    return edges


# ---------------------------------------------------------------------------
# 3D -> 2D flattening (SVD best-fit plane + Newell sign correction)
# ---------------------------------------------------------------------------


def flatten_3d_to_2d(points_3d):
    """Project 3D n-gon vertices onto their best-fit 2D plane.

    Uses SVD to find the optimal projection plane, then Newell's method to
    orient the normal so the projected 2D winding matches the original 3D
    winding (avoids mirrored output).

    Args:
        points_3d: ``(N, 3)`` float64 array of vertex positions.

    Returns:
        Tuple of ``(points_2d, axis_u, axis_v, origin)``.
    """
    pts      = np.asarray(points_3d, dtype=np.float64)
    n        = len(pts)

    origin   = pts.mean(axis=0)
    centered = pts - origin

    _, S, Vh = np.linalg.svd(centered, full_matrices=False)

    axis_u     = Vh[0]
    axis_v     = Vh[1]
    normal_svd = Vh[2]

    # Newell's method for reference normal
    newell = np.zeros(3, dtype=np.float64)
    for i in range(n):
        curr = pts[i]
        nxt  = pts[(i + 1) % n]
        newell[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
        newell[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
        newell[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])

    newell_len = np.linalg.norm(newell)
    if newell_len > 1e-15:
        if np.dot(normal_svd, newell) < 0:
            normal_svd = -normal_svd
            axis_v     = -axis_v

    points_2d = np.empty((n, 2), dtype=np.float64)
    points_2d[:, 0] = centered @ axis_u
    points_2d[:, 1] = centered @ axis_v

    return points_2d, axis_u, axis_v, origin


# ---------------------------------------------------------------------------
# Triangle mesh with adjacency
# ---------------------------------------------------------------------------


class TriMesh:
    """Half-edge-style triangle mesh for CDT operations.

    Uses Python ``dict``/``set`` for adjacency (not Numba-compatible).
    """

    def __init__(self, pts):
        self.pts         = np.asarray(pts, dtype=np.float64)
        self.tris        = {}
        self._next_tid   = 0
        self.edge2tri    = defaultdict(set)
        self.constrained = set()

    def add_tri(self, a, b, c):
        from cgmath.geometry.utils._numba._cdt import orient2d

        if (
            orient2d(
                self.pts[a, 0],
                self.pts[a, 1],
                self.pts[b, 0],
                self.pts[b, 1],
                self.pts[c, 0],
                self.pts[c, 1],
            )
            < 0
        ):
            b, c = c, b
        tid = self._next_tid
        self._next_tid += 1
        self.tris[tid] = (a, b, c)
        for e in [_ek(a, b), _ek(b, c), _ek(c, a)]:
            self.edge2tri[e].add(tid)
        return tid

    def remove_tri(self, tid):
        if tid not in self.tris:
            return
        a, b, c = self.tris.pop(tid)
        for e in [_ek(a, b), _ek(b, c), _ek(c, a)]:
            self.edge2tri[e].discard(tid)
            if not self.edge2tri[e]:
                del self.edge2tri[e]

    def opposite_vertex(self, tid, edge):
        a, b, c = self.tris[tid]
        ek = _ek(*edge)
        for v in (a, b, c):
            if v != ek[0] and v != ek[1]:
                return v
        return None

    def has_edge(self, a, b):
        return _ek(a, b) in self.edge2tri

    def flip_edge(self, a, b):
        from cgmath.geometry.utils._numba._cdt import check_convex

        ek   = _ek(a, b)
        tids = list(self.edge2tri.get(ek, []))
        if len(tids) != 2:
            return None
        c = self.opposite_vertex(tids[0], (a, b))
        d = self.opposite_vertex(tids[1], (a, b))
        if c is None or d is None:
            return None
        if not check_convex(self.pts, c, d, a, b):
            return None
        self.remove_tri(tids[0])
        self.remove_tri(tids[1])
        self.add_tri(c, d, a)
        self.add_tri(c, d, b)
        return _ek(c, d)

    def edges_as_array(self):
        """Return all current edges as an ``(M, 2)`` int64 numpy array."""
        edges = list(self.edge2tri.keys())
        if not edges:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# Constraint enforcement (Sloan's algorithm)
# ---------------------------------------------------------------------------


def enforce_constraints(mesh, constrained_edges):
    """Insert constrained edges into the mesh via Sloan's algorithm.

    Uses Numba-accelerated parallel crossing scan for bulk edge detection.
    """
    from cgmath.geometry.utils._numba._cdt import (
        find_crossing_edges,
        segments_intersect_proper,
    )

    mesh.constrained = set(constrained_edges)
    pts = mesh.pts

    for target in constrained_edges:
        a, b = target
        if mesh.has_edge(a, b):
            continue

        edges_arr = mesh.edges_as_array()
        mask      = find_crossing_edges(pts, edges_arr, a, b)
        crossing  = deque()
        for i in range(len(mask)):
            if mask[i]:
                crossing.append(_ek(edges_arr[i, 0], edges_arr[i, 1]))

        from cgmath.geometry.utils._numba._cdt import check_convex

        for _ in range(max(len(crossing) ** 2 + 20, 500)):
            if not crossing:
                break
            edge = crossing.popleft()
            if edge not in mesh.edge2tri or edge in mesh.constrained:
                continue
            if edge == _ek(a, b):
                break
            c, d = edge
            tids = list(mesh.edge2tri[edge])
            if len(tids) != 2:
                crossing.append(edge)
                continue
            e1 = mesh.opposite_vertex(tids[0], edge)
            e2 = mesh.opposite_vertex(tids[1], edge)
            if e1 is None or e2 is None:
                crossing.append(edge)
                continue
            if not check_convex(pts, e1, e2, c, d):
                crossing.append(edge)
                continue
            new_ek = mesh.flip_edge(c, d)
            if new_ek is None:
                crossing.append(edge)
                continue
            if new_ek == _ek(a, b):
                break
            nc, nd = new_ek
            if nc != a and nc != b and nd != a and nd != b:
                if segments_intersect_proper(
                    pts[a, 0],
                    pts[a, 1],
                    pts[b, 0],
                    pts[b, 1],
                    pts[nc, 0],
                    pts[nc, 1],
                    pts[nd, 0],
                    pts[nd, 1],
                ):
                    crossing.append(new_ek)


# ---------------------------------------------------------------------------
# Delaunay restoration
# ---------------------------------------------------------------------------


def restore_delaunay(mesh):
    """Restore the Delaunay property by flipping non-constrained edges."""
    from cgmath.geometry.utils._numba._cdt import (
        check_convex,
        check_delaunay_violation,
    )

    pts = mesh.pts
    for _pass_num in range(200):
        flipped = False
        for edge in list(mesh.edge2tri.keys()):
            if edge in mesh.constrained or edge not in mesh.edge2tri:
                continue
            tids = list(mesh.edge2tri[edge])
            if len(tids) != 2:
                continue
            a, b = edge
            c = mesh.opposite_vertex(tids[0], edge)
            d = mesh.opposite_vertex(tids[1], edge)
            if c is None or d is None:
                continue
            if check_delaunay_violation(pts, a, b, c, d):
                if not check_convex(pts, c, d, a, b):
                    continue
                if _ek(c, d) in mesh.constrained:
                    continue
                mesh.remove_tri(tids[0])
                mesh.remove_tri(tids[1])
                mesh.add_tri(c, d, a)
                mesh.add_tri(c, d, b)
                flipped = True
        if not flipped:
            break


# ---------------------------------------------------------------------------
# Maya-compatible cocircular tiebreak
# ---------------------------------------------------------------------------


def maya_tiebreak(mesh):
    """Swap near-cocircular edges to prefer the longer diagonal.

    Only swaps when ALL conditions are met:

    1. The quad is convex (required for a valid flip).
    2. The alternative diagonal is strictly longer.
    3. Min angles of both triangulations are nearly equal (< 0.005 rad).
    4. Circumcircle violation is small relative to quad geometry.
    """
    from cgmath.geometry.utils._numba._cdt import (
        check_convex,
        in_circumcircle,
        min_tri_angle,
        orient2d,
    )

    pts     = mesh.pts
    total   = 0
    changed = True
    while changed:
        changed = False
        for edge in list(mesh.edge2tri.keys()):
            if edge in mesh.constrained or edge not in mesh.edge2tri:
                continue
            tids = list(mesh.edge2tri[edge])
            if len(tids) != 2:
                continue
            a, b = edge
            c = mesh.opposite_vertex(tids[0], edge)
            d = mesh.opposite_vertex(tids[1], edge)
            if c is None or d is None:
                continue

            pax, pay = pts[a, 0], pts[a, 1]
            pbx, pby = pts[b, 0], pts[b, 1]
            pcx, pcy = pts[c, 0], pts[c, 1]
            pdx, pdy = pts[d, 0], pts[d, 1]

            if not check_convex(pts, c, d, a, b):
                continue

            len_ab_sq = (pax - pbx) ** 2 + (pay - pby) ** 2
            len_cd_sq = (pcx - pdx) ** 2 + (pcy - pdy) ** 2
            if len_cd_sq <= len_ab_sq:
                continue

            curr_min = min(
                min_tri_angle(pax, pay, pbx, pby, pcx, pcy),
                min_tri_angle(pax, pay, pbx, pby, pdx, pdy),
            )
            alt_min = min(
                min_tri_angle(pcx, pcy, pdx, pdy, pax, pay),
                min_tri_angle(pcx, pcy, pdx, pdy, pbx, pby),
            )

            if abs(curr_min - alt_min) > 0.005:
                continue

            if orient2d(pcx, pcy, pdx, pdy, pax, pay) > 0:
                cc_v = in_circumcircle(pcx, pcy, pdx, pdy, pax, pay, pbx, pby)
            else:
                cc_v = in_circumcircle(pdx, pdy, pcx, pcy, pax, pay, pbx, pby)

            max_esq = max(
                (pax - pbx) ** 2 + (pay - pby) ** 2,
                (pax - pcx) ** 2 + (pay - pcy) ** 2,
                (pax - pdx) ** 2 + (pay - pdy) ** 2,
                (pbx - pcx) ** 2 + (pby - pcy) ** 2,
                (pbx - pdx) ** 2 + (pby - pdy) ** 2,
                (pcx - pdx) ** 2 + (pcy - pdy) ** 2,
            )
            if cc_v > max_esq * 0.2:
                continue

            new_ek = _ek(c, d)
            if new_ek not in mesh.constrained:
                mesh.remove_tri(tids[0])
                mesh.remove_tri(tids[1])
                mesh.add_tri(c, d, a)
                mesh.add_tri(c, d, b)
                changed = True
                total += 1
    return total


# ---------------------------------------------------------------------------
# Triangle removal via flood fill (supports multiple holes)
# ---------------------------------------------------------------------------


def remove_exterior_and_holes(mesh, outer_boundary, hole_boundaries, points):
    """Remove triangles outside outer boundary and inside any holes.

    Args:
        mesh: :class:`TriMesh` instance.
        outer_boundary: Vertex indices for the outer boundary loop.
        hole_boundaries: List of vertex-index lists for hole loops,
            or ``None`` if no holes.
        points: ``(N, 2)`` point coordinates.
    """
    from cgmath.geometry.utils._numba._cdt import point_in_polygon_ray

    # Build adjacency (constrained edges are barriers)
    adj = defaultdict(set)
    for edge, tids in mesh.edge2tri.items():
        if edge in mesh.constrained:
            continue
        tids_list = list(tids)
        if len(tids_list) == 2:
            adj[tids_list[0]].add(tids_list[1])
            adj[tids_list[1]].add(tids_list[0])

    # Flood-fill to find connected regions
    regions     = {}
    region_tids = {}
    rid         = 0
    for start_tid in mesh.tris:
        if start_tid in regions:
            continue
        queue   = deque([start_tid])
        current = set()
        while queue:
            tid = queue.popleft()
            if tid in regions or tid not in mesh.tris:
                continue
            regions[tid] = rid
            current.add(tid)
            for nb in adj.get(tid, []):
                if nb not in regions and nb in mesh.tris:
                    queue.append(nb)
        region_tids[rid] = current
        rid += 1

    # Classify regions using point-in-polygon
    outer_x    = np.array([points[v, 0] for v in outer_boundary], dtype=np.float64)
    outer_y    = np.array([points[v, 1] for v in outer_boundary], dtype=np.float64)

    hole_polys = []
    if hole_boundaries:
        for hb in hole_boundaries:
            hx = np.array([points[v, 0] for v in hb], dtype=np.float64)
            hy = np.array([points[v, 1] for v in hb], dtype=np.float64)
            hole_polys.append((hx, hy))

    keep = set()
    for r, tids in region_tids.items():
        sample = next(iter(tids))
        a, b, c = mesh.tris[sample]
        cx       = (points[a, 0] + points[b, 0] + points[c, 0]) / 3.0
        cy       = (points[a, 1] + points[b, 1] + points[c, 1]) / 3.0
        in_outer = point_in_polygon_ray(cx, cy, outer_x, outer_y)
        in_hole  = False
        for hx, hy in hole_polys:
            if point_in_polygon_ray(cx, cy, hx, hy):
                in_hole = True
                break
        if in_outer and not in_hole:
            keep.add(r)

    for tid in [t for t, r in regions.items() if r not in keep]:
        if tid in mesh.tris:
            mesh.remove_tri(tid)


# ---------------------------------------------------------------------------
# Public API: triangulate_ngon
# ---------------------------------------------------------------------------


def triangulate_ngon(points, face_vertices, hole_boundaries=None):
    """Triangulate an n-gon face using Constrained Delaunay Triangulation.

    Args:
        points: Global ``(V, D)`` point array (2D or 3D).
        face_vertices: Vertex indices of the outer boundary loop.
        hole_boundaries: Optional list of vertex-index lists for holes.

    Returns:
        ``(T, 3)`` int array of **local** vertex indices (0 to ``n-1``
        within *face_indices*), where *face_indices* is the concatenation
        of ``face_vertices + all hole vertices``. Local indices make the
        result topology-only and portable to UV data.
    """
    from scipy.spatial import Delaunay

    # Build face_indices: outer + all holes
    face_indices = list(face_vertices)
    if hole_boundaries:
        for hb in hole_boundaries:
            face_indices.extend(hb)

    g2l = {g: idx for idx, g in enumerate(face_indices)}
    l2g = {idx: g for g, idx in g2l.items()}

    # Extract face points and flatten 3D -> 2D if needed
    face_pts_raw = points[face_indices].copy()
    if face_pts_raw.shape[1] == 3:
        local_pts, _, _, _ = flatten_3d_to_2d(face_pts_raw)
    else:
        local_pts = face_pts_raw[:, :2].copy()

    # Build constrained edges from all boundary loops
    constrained = get_boundary_edges([g2l[g] for g in face_vertices])
    if hole_boundaries:
        for hb in hole_boundaries:
            constrained |= get_boundary_edges([g2l[g] for g in hb])

    local_outer = [g2l[g] for g in face_vertices]
    local_holes = None
    if hole_boundaries:
        local_holes = [[g2l[g] for g in hb] for hb in hole_boundaries]

    has_holes    = hole_boundaries is not None and len(hole_boundaries) > 0
    n_outer      = len(face_vertices)
    n_hole_verts = sum(len(hb) for hb in hole_boundaries) if has_holes else 0
    expected     = n_outer + n_hole_verts if has_holes else n_outer - 2

    # Step 1: scipy Delaunay + constraints + Delaunay restoration
    dt   = Delaunay(local_pts)
    mesh = TriMesh(local_pts)
    for simplex in dt.simplices:
        mesh.add_tri(simplex[0], simplex[1], simplex[2])

    enforce_constraints(mesh, constrained)
    restore_delaunay(mesh)

    # Step 2: Maya-compatible tiebreak
    maya_tiebreak(mesh)

    # Step 3: Remove exterior and hole triangles
    remove_exterior_and_holes(mesh, local_outer, local_holes, local_pts)

    # Collect results as local vertex indices
    result = []
    from cgmath.geometry.utils._numba._cdt import orient2d

    if face_pts_raw.shape[1] == 3:
        # 3D: align winding with face normal (Newell's method on outer boundary)
        outer_pts_3d = points[list(face_vertices)]
        n_out        = len(face_vertices)
        face_normal  = np.zeros(3, dtype=np.float64)
        for i in range(n_out):
            curr = outer_pts_3d[i]
            nxt  = outer_pts_3d[(i + 1) % n_out]
            face_normal[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
            face_normal[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
            face_normal[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])

        for tri in mesh.tris.values():
            la, lb, lc = tri
            ga, gb, gc = l2g[la], l2g[lb], l2g[lc]
            tri_normal = np.cross(points[gb] - points[ga], points[gc] - points[ga])
            if np.dot(tri_normal, face_normal) < 0:
                lb, lc = lc, lb
            result.append((la, lb, lc))
    else:
        # 2D: ensure CCW winding
        for tri in mesh.tris.values():
            la, lb, lc = tri
            if (
                orient2d(
                    local_pts[la, 0],
                    local_pts[la, 1],
                    local_pts[lb, 0],
                    local_pts[lb, 1],
                    local_pts[lc, 0],
                    local_pts[lc, 1],
                )
                < 0
            ):
                lb, lc = lc, lb
            result.append((la, lb, lc))

    if len(result) != expected:
        warnings.warn(
            f"CDT produced {len(result)} triangles, expected {expected} "
            f"for face with {n_outer} outer + {n_hole_verts} hole vertices.",
            stacklevel=2,
        )

    if not result:
        return np.empty((0, 3), dtype=np.intp)

    return np.array(result, dtype=np.intp).reshape(-1, 3)