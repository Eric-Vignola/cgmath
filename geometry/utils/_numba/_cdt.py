"""
Numba-accelerated geometric predicates for Constrained Delaunay Triangulation.

All functions use scalar arguments for maximum JIT performance.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def orient2d(ax, ay, bx, by, cx, cy):
    """Orientation test: positive if (A, B, C) is CCW, negative if CW, zero if collinear."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def in_circumcircle(ax, ay, bx, by, cx, cy, dx, dy):
    """Positive if D lies inside the circumcircle of CCW triangle (A, B, C)."""
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    return (
        (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
        - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    )


@njit(cache=True)
def segments_intersect_proper(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """True if segments (p1,p2) and (p3,p4) properly intersect (no shared endpoints)."""
    d1 = orient2d(p3x, p3y, p4x, p4y, p1x, p1y)
    d2 = orient2d(p3x, p3y, p4x, p4y, p2x, p2y)
    d3 = orient2d(p1x, p1y, p2x, p2y, p3x, p3y)
    d4 = orient2d(p1x, p1y, p2x, p2y, p4x, p4y)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    )


@njit(cache=True)
def point_in_polygon_ray(px, py, poly_x, poly_y):
    """Ray-casting point-in-polygon test using Numba arrays for coordinates."""
    n      = len(poly_x)
    inside = False
    j      = n - 1
    for i in range(n):
        yi, yj = poly_y[i], poly_y[j]
        xi, xj = poly_x[i], poly_x[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@njit(cache=True)
def check_delaunay_violation(pts, a, b, c, d):
    """True if edge (a,b) violates Delaunay w.r.t. opposite vertices c, d."""
    pax, pay = pts[a, 0], pts[a, 1]
    pbx, pby = pts[b, 0], pts[b, 1]
    pcx, pcy = pts[c, 0], pts[c, 1]
    pdx, pdy = pts[d, 0], pts[d, 1]
    if orient2d(pax, pay, pbx, pby, pcx, pcy) < 0:
        pax, pay, pbx, pby = pbx, pby, pax, pay
    return in_circumcircle(pax, pay, pbx, pby, pcx, pcy, pdx, pdy) > 0


@njit(cache=True)
def check_convex(pts, c, d, a, b):
    """True if quad (a,b) with opposite verts (c,d) is convex for flip to (c,d)."""
    return (
        orient2d(pts[c, 0], pts[c, 1], pts[d, 0], pts[d, 1], pts[a, 0], pts[a, 1])
        * orient2d(pts[c, 0], pts[c, 1], pts[d, 0], pts[d, 1], pts[b, 0], pts[b, 1])
    ) < 0


@njit(cache=True)
def min_tri_angle(p1x, p1y, p2x, p2y, p3x, p3y):
    """Minimum angle (radians) of the triangle with given vertex coordinates."""
    la = np.sqrt((p2x - p3x) ** 2 + (p2y - p3y) ** 2)
    lb = np.sqrt((p1x - p3x) ** 2 + (p1y - p3y) ** 2)
    lc = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
    if la < 1e-15 or lb < 1e-15 or lc < 1e-15:
        return 0.0
    cA = max(-1.0, min(1.0, (lb * lb + lc * lc - la * la) / (2 * lb * lc)))
    cB = max(-1.0, min(1.0, (la * la + lc * lc - lb * lb) / (2 * la * lc)))
    cC = max(-1.0, min(1.0, (la * la + lb * lb - lc * lc) / (2 * la * lb)))
    return min(np.arccos(cA), np.arccos(cB), np.arccos(cC))


@njit(parallel=True, cache=True)
def find_crossing_edges(pts, edges_array, a, b):
    """Find edges in edges_array that properly cross segment (a, b).

    Args:
        pts: (N, 2) float64 point coordinates
        edges_array: (M, 2) int64 edge vertex pairs
        a, b: vertex indices of the constraint segment

    Returns:
        boolean mask of length M
    """
    m      = len(edges_array)
    result = np.zeros(m, dtype=np.bool_)
    ax, ay = pts[a, 0], pts[a, 1]
    bx, by = pts[b, 0], pts[b, 1]
    for i in prange(m):
        c = edges_array[i, 0]
        d = edges_array[i, 1]
        if c == a or c == b or d == a or d == b:
            continue
        cx, cy = pts[c, 0], pts[c, 1]
        dx, dy = pts[d, 0], pts[d, 1]
        if segments_intersect_proper(ax, ay, bx, by, cx, cy, dx, dy):
            result[i] = True
    return result