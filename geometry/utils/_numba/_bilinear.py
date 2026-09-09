import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def _remap_indices(vert_indices, uv_indices, distances):
    # returns closest unique corresponding streams of indices

    uv_distances = distances[uv_indices]
    dists        = np.full(np.unique(vert_indices).size, uv_distances.max())
    match = np.arange(np.unique(vert_indices).size, dtype=vert_indices.dtype)

    for i in range(vert_indices.size):
        if uv_distances[i] <= dists[vert_indices[i]]:
            dists[vert_indices[i]] = uv_distances[i]
            match[vert_indices[i]] = uv_indices[i]

    return match


@njit(fastmath=True, parallel=True, cache=True)
def _compute_samples(values, weights, geometry):
    new_values = np.zeros((weights.shape[0], values.shape[1]), dtype=values.dtype)

    # for each geometry row
    for i in prange(geometry.shape[0]):
        # for each geometry index
        for j in range(geometry.shape[1]):
            # skip triangles
            if geometry[i, j] != -1:
                # sum each weighted value element
                for k in range(values.shape[1]):
                    new_values[i, k] += values[geometry[i, j], k] * weights[i, j]

    return new_values


# --- Bilinear area computation helpers --- #


@njit(fastmath=True, cache=True)
def _cross3d(a, b):
    """Cross product of two 3D vectors."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@njit(fastmath=True, cache=True)
def _norm3d(a):
    """Euclidean norm of a 3D vector."""
    return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


@njit(fastmath=True, cache=True)
def _get_bilinear_derivatives(p0, p1, p2, p3, u, v):
    """Partial derivatives of bilinear surface at (u, v)."""
    dpdu = np.empty(3)
    dpdv = np.empty(3)
    for i in range(3):
        C = p0[i] - p1[i] + p2[i] - p3[i]
        D = p1[i] - p0[i]
        E = p3[i] - p0[i]
        dpdu[i] = C * v + D
        dpdv[i] = C * u + E
    return dpdu, dpdv


@njit(fastmath=True, cache=True)
def _bilinear_area_element(p0, p1, p2, p3, u, v):
    """Differential area element ||dS/du x dS/dv|| at (u, v)."""
    dpdu, dpdv = _get_bilinear_derivatives(p0, p1, p2, p3, u, v)
    return _norm3d(_cross3d(dpdu, dpdv))


@njit(fastmath=True, cache=True)
def _bilinear_face_area_quadrature(p0, p1, p2, p3, n):
    """Area of a single bilinear face using n*n midpoint quadrature."""
    delta = 1.0 / n
    total = 0.0
    for i in range(n):
        u = (i + 0.5) * delta
        for j in range(n):
            v = (j + 0.5) * delta
            total += _bilinear_area_element(p0, p1, p2, p3, u, v)
    return total * delta * delta


@njit(fastmath=True, cache=True)
def _bilinear_face_area_gauss(p0, p1, p2, p3):
    """Area of a single bilinear face using 2x2 Gaussian quadrature."""
    sqrt3_inv = 0.5773502691896257  # 1/sqrt(3)
    g1        = 0.5 * (1.0 - sqrt3_inv)
    g2        = 0.5 * (1.0 + sqrt3_inv)
    w         = 0.25
    total     = 0.0
    for u in (g1, g2):
        for v in (g1, g2):
            total += _bilinear_area_element(p0, p1, p2, p3, u, v)
    return total * w


@njit(parallel=True, fastmath=True, cache=True)
def _bilinear_integrate_gauss(points, geometry):
    """2x2 Gaussian quadrature over all faces (parallel)."""
    num_faces = geometry.shape[0]
    areas     = np.empty(num_faces, dtype=points.dtype)
    for fi in prange(num_faces):
        p0 = points[geometry[fi, 0]]
        p1 = points[geometry[fi, 1]]
        p2 = points[geometry[fi, 2]]
        i3 = geometry[fi, 3]
        if i3 < 0:
            p3 = p2
        else:
            p3 = points[i3]
        areas[fi] = _bilinear_face_area_gauss(p0, p1, p2, p3)
    return areas


@njit(parallel=True, fastmath=True, cache=True)
def _bilinear_integrate_quadrature(points, geometry, n):
    """n*n midpoint rule over all faces (parallel)."""
    num_faces = geometry.shape[0]
    areas     = np.empty(num_faces, dtype=points.dtype)
    for fi in prange(num_faces):
        p0 = points[geometry[fi, 0]]
        p1 = points[geometry[fi, 1]]
        p2 = points[geometry[fi, 2]]
        i3 = geometry[fi, 3]
        if i3 < 0:
            p3 = p2
        else:
            p3 = points[i3]
        areas[fi] = _bilinear_face_area_quadrature(p0, p1, p2, p3, n)
    return areas


@njit(parallel=True, fastmath=True, cache=True)
def _bilinear_integrate_adaptive(points, geometry, base_n, max_n, tol):
    """Adaptive refinement per face (parallel)."""
    num_faces = geometry.shape[0]
    areas     = np.empty(num_faces, dtype=points.dtype)
    for fi in prange(num_faces):
        p0 = points[geometry[fi, 0]]
        p1 = points[geometry[fi, 1]]
        p2 = points[geometry[fi, 2]]
        i3 = geometry[fi, 3]
        if i3 < 0:
            p3 = p2
        else:
            p3 = points[i3]
        n         = base_n
        prev_area = _bilinear_face_area_quadrature(p0, p1, p2, p3, n)
        while n < max_n:
            n *= 2
            curr_area = _bilinear_face_area_quadrature(p0, p1, p2, p3, n)
            if abs(curr_area - prev_area) < tol * abs(curr_area):
                break
            prev_area = curr_area
        areas[fi] = curr_area
    return areas


@njit(parallel=True, fastmath=True, cache=True)
def _bilinear_integrate(points, geometry, n):
    # Deprecated: delegates to proper 2D quadrature.
    # Use _bilinear_integrate_gauss, _bilinear_integrate_quadrature,
    # or _bilinear_integrate_adaptive directly for new code.
    return _bilinear_integrate_quadrature(points, geometry, n)


# --- Gauss-Newton closest point helpers --- #


@njit(fastmath=True, cache=True)
def _clamp(x, lo, hi):
    """Clamp scalar x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(fastmath=True, cache=True)
def _closest_point_on_face_newton(p0, p1, p2, p3, query, u0, v0, eps, max_its):
    """
    Gauss-Newton solver for closest point on bilinear patch.
    Generalized to N dimensions.

    Minimizes f(u,v) = ||S(u,v) - query||^2
    where S(u,v) = C*u*v + D*u + E*v + F
    """
    ndim = p0.shape[0]
    u    = u0
    v    = v0

    C    = np.empty(ndim, dtype=p0.dtype)
    D    = np.empty(ndim, dtype=p0.dtype)
    E    = np.empty(ndim, dtype=p0.dtype)
    F    = np.empty(ndim, dtype=p0.dtype)

    for i in range(ndim):
        C[i] = p0[i] - p1[i] + p2[i] - p3[i]
        D[i] = p1[i] - p0[i]
        E[i] = p3[i] - p0[i]
        F[i] = p0[i]

    eps_sq = eps * eps

    for _ in range(max_its):
        grad_u = 0.0
        grad_v = 0.0
        H00    = 0.0
        H01    = 0.0
        H11    = 0.0

        for i in range(ndim):
            si     = C[i] * u * v + D[i] * u + E[i] * v + F[i]
            du_i   = C[i] * v + D[i]
            dv_i   = C[i] * u + E[i]
            diff_i = si - query[i]

            grad_u += 2.0 * diff_i * du_i
            grad_v += 2.0 * diff_i * dv_i
            H00 += 2.0 * du_i * du_i
            H01 += 2.0 * du_i * dv_i
            H11 += 2.0 * dv_i * dv_i

        if grad_u * grad_u + grad_v * grad_v < eps_sq:
            break

        det = H00 * H11 - H01 * H01
        if abs(det) < 1e-12:
            break

        inv_det = 1.0 / det
        delta_u = (H11 * grad_u - H01 * grad_v) * inv_det
        delta_v = (H00 * grad_v - H01 * grad_u) * inv_det

        u       = _clamp(u - delta_u, 0.0, 1.0)
        v       = _clamp(v - delta_v, 0.0, 1.0)

    dist_sq = 0.0
    for i in range(ndim):
        si   = C[i] * u * v + D[i] * u + E[i] * v + F[i]
        diff = si - query[i]
        dist_sq += diff * diff

    return u, v, dist_sq


@njit(fastmath=True, cache=True)
def _closest_point_on_edge_sq(a, b, query):
    """
    Closest point on line segment a->b to query point.
    N-dimensional. Returns (t, dist_sq).
    """
    ndim    = a.shape[0]
    elen_sq = 0.0
    dot     = 0.0

    for i in range(ndim):
        ei = b[i] - a[i]
        elen_sq += ei * ei
        dot += (query[i] - a[i]) * ei

    if elen_sq < 1e-30:
        dsq = 0.0
        for i in range(ndim):
            d = query[i] - a[i]
            dsq += d * d
        return 0.0, dsq

    t = dot / elen_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    dist_sq = 0.0
    for i in range(ndim):
        pi = a[i] + t * (b[i] - a[i])
        d  = query[i] - pi
        dist_sq += d * d

    return t, dist_sq


@njit(fastmath=True, cache=True)
def _closest_point_on_face(p0, p1, p2, p3, query, eps=1e-6, max_its=20):
    """
    Combined Newton + edge fallback for closest point on bilinear face.
    Runs Newton from (0.5, 0.5). If the result lands on the domain
    boundary, also checks all 4 edge projections for a better solution.
    Returns (u, v, dist_sq).
    """
    u, v, dist_sq = _closest_point_on_face_newton(
        p0, p1, p2, p3, query, 0.5, 0.5, eps, max_its
    )

    if 0.0 < u < 1.0 and 0.0 < v < 1.0:
        return u, v, dist_sq

    best_u   = u
    best_v   = v
    best_dsq = dist_sq

    # Edge v=0: p0 -> p1
    t, dsq = _closest_point_on_edge_sq(p0, p1, query)
    if dsq < best_dsq:
        best_u   = t
        best_v   = 0.0
        best_dsq = dsq

    # Edge v=1: p3 -> p2
    t, dsq = _closest_point_on_edge_sq(p3, p2, query)
    if dsq < best_dsq:
        best_u   = t
        best_v   = 1.0
        best_dsq = dsq

    # Edge u=0: p0 -> p3
    t, dsq = _closest_point_on_edge_sq(p0, p3, query)
    if dsq < best_dsq:
        best_u   = 0.0
        best_v   = t
        best_dsq = dsq

    # Edge u=1: p1 -> p2
    t, dsq = _closest_point_on_edge_sq(p1, p2, query)
    if dsq < best_dsq:
        best_u   = 1.0
        best_v   = t
        best_dsq = dsq

    return best_u, best_v, best_dsq


@njit(parallel=True, fastmath=True, cache=True)
def _bilinear_sample(
    p,
    points,
    geometry,
    centroids,
    radiuses,
    centroid_distance_tolerance,
    iteration_count     = 100,
    iteration_tolerance = 1e-8,
    uv_border_tolerance = 1e-5,
):
    """
    Finds the closest points to a mesh using Gauss-Newton optimization
    on bilinear patches with adaptive bounding-sphere culling.
    """
    ndim          = p.shape[1]
    num_queries   = p.shape[0]
    num_faces     = geometry.shape[0]

    closest_proj  = np.empty(p.shape,          dtype=points.dtype)
    closest_dist  = np.empty(num_queries,      dtype=points.dtype)
    closest_w     = np.empty((num_queries, 4), dtype=points.dtype)
    closest_uv    = np.empty((num_queries, 2), dtype=points.dtype)
    closest_index = np.empty(num_queries,      dtype=geometry.dtype)

    eps           = iteration_tolerance
    max_its       = iteration_count

    for k in prange(num_queries):
        best_dist_sq = np.inf
        best_dist    = np.inf
        best_index   = -1
        best_u       = 0.5
        best_v       = 0.5

        p0           = np.empty(ndim, dtype=points.dtype)
        p1           = np.empty(ndim, dtype=points.dtype)
        p2           = np.empty(ndim, dtype=points.dtype)
        p3           = np.empty(ndim, dtype=points.dtype)

        for j in range(num_faces):
            # Adaptive bounding sphere culling
            dist_to_centroid_sq = 0.0
            for i in range(ndim):
                diff = p[k, i] - centroids[j, i]
                dist_to_centroid_sq += diff * diff

            radius = radiuses[j] ** 0.5
            upper  = best_dist + radius
            if dist_to_centroid_sq > upper * upper:
                continue

            # Build face vertices
            for i in range(ndim):
                p0[i] = points[geometry[j, 0], i]
                p1[i] = points[geometry[j, 1], i]
                p2[i] = points[geometry[j, 2], i]

            i3 = geometry[j, 3]
            if i3 < 0:
                for i in range(ndim):
                    p3[i] = p2[i]
            else:
                for i in range(ndim):
                    p3[i] = points[i3, i]

            u, v, dist_sq = _closest_point_on_face(p0, p1, p2, p3, p[k], eps, max_its)

            # UV border tolerance snapping
            if u < uv_border_tolerance:
                u = 0.0
            elif (1.0 - u) < uv_border_tolerance:
                u = 1.0
            if v < uv_border_tolerance:
                v = 0.0
            elif (1.0 - v) < uv_border_tolerance:
                v = 1.0

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_dist    = dist_sq**0.5
                best_index   = j
                best_u       = u
                best_v       = v

                if dist_sq == 0.0:
                    break

        # Write results
        closest_dist[k] = best_dist_sq
        closest_index[k] = best_index
        closest_uv[k, 0] = best_u
        closest_uv[k, 1] = best_v

        ou = 1.0 - best_u
        ov = 1.0 - best_v
        closest_w[k, 0] = ou * ov
        closest_w[k, 1] = best_u * ov
        closest_w[k, 2] = best_u * best_v

        if best_index >= 0 and geometry[best_index, 3] == -1:
            closest_w[k, 2] += ou * best_v
            closest_w[k, 3] = 0.0
        else:
            closest_w[k, 3] = ou * best_v

        # Compute projection S(u,v) = (1-u)(1-v)*p0 + u(1-v)*p1 + u*v*p2 + (1-u)*v*p3
        if best_index >= 0:
            for i in range(ndim):
                p0i = points[geometry[best_index, 0], i]
                p1i = points[geometry[best_index, 1], i]
                p2i = points[geometry[best_index, 2], i]
                i3b = geometry[best_index, 3]
                if i3b < 0:
                    p3i = p2i
                else:
                    p3i = points[i3b, i]

                closest_proj[k, i] = (
                    ou * ov * p0i
                    + best_u * ov * p1i
                    + best_u * best_v * p2i
                    + ou * best_v * p3i
                )
        else:
            for i in range(ndim):
                closest_proj[k, i] = p[k, i]

    return closest_proj, closest_dist, closest_w, closest_uv, closest_index


@njit(fastmath=True, parallel=True, cache=True)
def _bilinear_vectors(points, geometry, uv):
    """
    computes U,V vectors at given uv coordinates per face
    """

    # init buffers
    U = np.zeros((geometry.shape[0], points.shape[1]), dtype=points.dtype)
    V = np.zeros((geometry.shape[0], points.shape[1]), dtype=points.dtype)

    for i in prange(geometry.shape[0]):
        # if this is a valid face
        if geometry[i, 2] > -1:
            p0 = points[geometry[i, 0]]
            p1 = points[geometry[i, 1]]
            p2 = points[geometry[i, 2]]

            u  = uv[i, 0]
            v  = uv[i, 1]

            # if this is a triangle
            if geometry.shape[1] == 3 or geometry[i, 3] == -1:
                for j in range(points.shape[1]):
                    U[i, j] = p1[j] - p0[j]
                    V[i, j] = p2[j] - p0[j]

            # else its a quad
            else:
                p3 = points[geometry[i, 3]]

                for j in range(points.shape[1]):
                    U[i, j] = (p1[j] - p0[j]) * (1 - v) + (p2[j] - p3[j]) * v
                    V[i, j] = (p3[j] - p0[j]) * (1 - u) + (p2[j] - p1[j]) * u

    return U, V


# ======================================================================
# PN Quad Bicubic Bezier: Control Point Derivation
# ======================================================================


@njit(fastmath=True, cache=True)
def _compute_edge_cp(p_near, p_far, normal):
    """
    Compute a PN Quad edge control point.
    The 1/3 point from p_near toward p_far projected onto the tangent
    plane at p_near defined by the given normal.
    """
    qx = (2.0 * p_near[0] + p_far[0]) / 3.0
    qy = (2.0 * p_near[1] + p_far[1]) / 3.0
    qz = (2.0 * p_near[2] + p_far[2]) / 3.0
    d = (
        (qx - p_near[0]) * normal[0]
        + (qy - p_near[1]) * normal[1]
        + (qz - p_near[2]) * normal[2]
    )
    return qx - d * normal[0], qy - d * normal[1], qz - d * normal[2]


@njit(fastmath=True, cache=True)
def _compute_pn_quad_cps_single(p0, p1, p2, p3, n0, n1, n2, n3, cp):
    """
    Compute 16 bicubic Bezier control points for a single PN Quad face.

    Control point grid cp[i][j] (4x4x3):
        i indexes the u-direction (0->3 maps to p0->p1 at v=0)
        j indexes the v-direction (0->3 maps to p0->p3 at u=0)

    Corner mapping:
        cp[0,0] = p0  (u=0, v=0)
        cp[3,0] = p1  (u=1, v=0)
        cp[3,3] = p2  (u=1, v=1)
        cp[0,3] = p3  (u=0, v=1)
    """
    for k in range(3):
        cp[0, 0, k] = p0[k]
        cp[3, 0, k] = p1[k]
        cp[3, 3, k] = p2[k]
        cp[0, 3, k] = p3[k]

    # 8 Edge control points via tangent plane projection
    cp[1, 0, 0], cp[1, 0, 1], cp[1, 0, 2] = _compute_edge_cp(p0, p1, n0)
    cp[2, 0, 0], cp[2, 0, 1], cp[2, 0, 2] = _compute_edge_cp(p1, p0, n1)
    cp[1, 3, 0], cp[1, 3, 1], cp[1, 3, 2] = _compute_edge_cp(p3, p2, n3)
    cp[2, 3, 0], cp[2, 3, 1], cp[2, 3, 2] = _compute_edge_cp(p2, p3, n2)
    cp[0, 1, 0], cp[0, 1, 1], cp[0, 1, 2] = _compute_edge_cp(p0, p3, n0)
    cp[0, 2, 0], cp[0, 2, 1], cp[0, 2, 2] = _compute_edge_cp(p3, p0, n3)
    cp[3, 1, 0], cp[3, 1, 1], cp[3, 1, 2] = _compute_edge_cp(p1, p2, n1)
    cp[3, 2, 0], cp[3, 2, 1], cp[3, 2, 2] = _compute_edge_cp(p2, p1, n2)

    # 4 Interior control points via Coons patch construction
    for i in range(1, 3):
        s  = i / 3.0
        s1 = 1.0 - s
        for j in range(1, 3):
            t  = j / 3.0
            t1 = 1.0 - t
            for k in range(3):
                rs_bt = t1 * cp[i, 0, k] + t * cp[i, 3, k]
                rs_lr = s1 * cp[0, j, k] + s * cp[3, j, k]
                bilinear = (
                    s1 * t1 * cp[0, 0, k]
                    + s * t1 * cp[3, 0, k]
                    + s1 * t * cp[0, 3, k]
                    + s * t * cp[3, 3, k]
                )
                cp[i, j, k] = rs_bt + rs_lr - bilinear


@njit(fastmath=True, cache=True)
def _compute_all_pn_quad_cps(faces, points, normals):
    """
    Compute bicubic Bezier control points for all faces.

    Args:
        faces: (num_faces, 4) vertex indices (-1 for triangles)
        points: (num_points, 3) vertex positions
        normals: (num_points, 3) vertex normals (unit length)

    Returns:
        control_points: (num_faces, 4, 4, 3)
    """
    num_faces      = faces.shape[0]
    control_points = np.empty((num_faces, 4, 4, 3), dtype=np.float64)

    for fi in range(num_faces):
        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        i3 = faces[fi, 3]

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        n0 = normals[i0]
        n1 = normals[i1]
        n2 = normals[i2]

        if i3 < 0:
            p3 = p2
            n3 = n2
        else:
            p3 = points[i3]
            n3 = normals[i3]

        _compute_pn_quad_cps_single(p0, p1, p2, p3, n0, n1, n2, n3, control_points[fi])

    return control_points


@njit(fastmath=True, cache=True)
def _compute_bezier_face_bounds(control_points):
    """
    Compute bounding sphere per face from all 16 control points.

    Returns:
        centroids: (num_faces, 3) bounding sphere centers
        radii: (num_faces,) bounding sphere radii (squared, consistent
                with _bilinear_sample convention)
    """
    num_faces = control_points.shape[0]
    centroids = np.empty((num_faces, 3), dtype=np.float64)
    radii     = np.empty(num_faces, dtype=np.float64)

    for fi in range(num_faces):
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for i in range(4):
            for j in range(4):
                cx += control_points[fi, i, j, 0]
                cy += control_points[fi, i, j, 1]
                cz += control_points[fi, i, j, 2]
        cx /= 16.0
        cy /= 16.0
        cz /= 16.0
        centroids[fi, 0] = cx
        centroids[fi, 1] = cy
        centroids[fi, 2] = cz

        max_r_sq = 0.0
        for i in range(4):
            for j in range(4):
                dx   = control_points[fi, i, j, 0] - cx
                dy   = control_points[fi, i, j, 1] - cy
                dz   = control_points[fi, i, j, 2] - cz
                r_sq = dx * dx + dy * dy + dz * dz
                if r_sq > max_r_sq:
                    max_r_sq = r_sq
        radii[fi] = max_r_sq

    return centroids, radii


# ======================================================================
# Bicubic Bezier Evaluation + Newton Solver
# ======================================================================


@njit(fastmath=True, cache=True)
def _eval_bezier_surface(cp, u, v):
    """
    Evaluate a bicubic Bezier surface and its partial derivatives.

    Returns:
        (sx, sy, sz, dux, duy, duz, dvx, dvy, dvz)
    """
    u1   = 1.0 - u
    v1   = 1.0 - v

    Bu0  = u1 * u1 * u1
    Bu1  = 3.0 * u1 * u1 * u
    Bu2  = 3.0 * u1 * u * u
    Bu3  = u * u * u

    dBu0 = -3.0 * u1 * u1
    dBu1 = 3.0 * u1 * (1.0 - 3.0 * u)
    dBu2 = 3.0 * u * (2.0 - 3.0 * u)
    dBu3 = 3.0 * u * u

    Bv0  = v1 * v1 * v1
    Bv1  = 3.0 * v1 * v1 * v
    Bv2  = 3.0 * v1 * v * v
    Bv3  = v * v * v

    dBv0 = -3.0 * v1 * v1
    dBv1 = 3.0 * v1 * (1.0 - 3.0 * v)
    dBv2 = 3.0 * v * (2.0 - 3.0 * v)
    dBv3 = 3.0 * v * v

    sx   = 0.0
    sy   = 0.0
    sz   = 0.0
    dux  = 0.0
    duy  = 0.0
    duz  = 0.0
    dvx  = 0.0
    dvy  = 0.0
    dvz  = 0.0

    for i in range(4):
        if i == 0:
            bui  = Bu0
            dbui = dBu0
        elif i == 1:
            bui  = Bu1
            dbui = dBu1
        elif i == 2:
            bui  = Bu2
            dbui = dBu2
        else:
            bui  = Bu3
            dbui = dBu3

        for j in range(4):
            if j == 0:
                bvj  = Bv0
                dbvj = dBv0
            elif j == 1:
                bvj  = Bv1
                dbvj = dBv1
            elif j == 2:
                bvj  = Bv2
                dbvj = dBv2
            else:
                bvj  = Bv3
                dbvj = dBv3

            w   = bui * bvj
            wu  = dbui * bvj
            wv  = bui * dbvj

            cpx = cp[i, j, 0]
            cpy = cp[i, j, 1]
            cpz = cp[i, j, 2]

            sx += w * cpx
            sy += w * cpy
            sz += w * cpz
            dux += wu * cpx
            duy += wu * cpy
            duz += wu * cpz
            dvx += wv * cpx
            dvy += wv * cpy
            dvz += wv * cpz

    return sx, sy, sz, dux, duy, duz, dvx, dvy, dvz


@njit(fastmath=True, cache=True)
def _closest_point_on_bezier_face_newton(cp, query, u0, v0, eps=1e-6, max_its=20):
    """
    Gauss-Newton iteration for closest point on bicubic Bezier patch.
    Returns (u, v, dist_sq).
    """
    u      = u0
    v      = v0
    eps_sq = eps * eps

    for _ in range(max_its):
        sx, sy, sz, dux, duy, duz, dvx, dvy, dvz = _eval_bezier_surface(cp, u, v)

        diffx  = sx - query[0]
        diffy  = sy - query[1]
        diffz  = sz - query[2]

        grad_u = 2.0 * (diffx * dux + diffy * duy + diffz * duz)
        grad_v = 2.0 * (diffx * dvx + diffy * dvy + diffz * dvz)

        if grad_u * grad_u + grad_v * grad_v < eps_sq:
            break

        H00 = 2.0 * (dux * dux + duy * duy + duz * duz)
        H01 = 2.0 * (dux * dvx + duy * dvy + duz * dvz)
        H11 = 2.0 * (dvx * dvx + dvy * dvy + dvz * dvz)

        det = H00 * H11 - H01 * H01
        if abs(det) < 1e-12:
            break

        inv_det = 1.0 / det
        delta_u = (H11 * grad_u - H01 * grad_v) * inv_det
        delta_v = (H00 * grad_v - H01 * grad_u) * inv_det

        u       = _clamp(u - delta_u, 0.0, 1.0)
        v       = _clamp(v - delta_v, 0.0, 1.0)

    sx, sy, sz, _, _, _, _, _, _ = _eval_bezier_surface(cp, u, v)
    dist_sq = (sx - query[0]) ** 2 + (sy - query[1]) ** 2 + (sz - query[2]) ** 2
    return u, v, dist_sq


@njit(fastmath=True, cache=True)
def _closest_point_on_cubic_curve(b0, b1, b2, b3, query):
    """
    Find closest point on a cubic Bezier curve to a query point.
    Samples 5 initial points, refines with Newton iteration.
    Returns (t, dist_sq).
    """
    best_t   = 0.0
    best_dsq = 1e300

    for si in range(5):
        t  = si * 0.25
        t1 = 1.0 - t
        cx = (
            t1 * t1 * t1 * b0[0]
            + 3.0 * t1 * t1 * t * b1[0]
            + 3.0 * t1 * t * t * b2[0]
            + t * t * t * b3[0]
        )
        cy = (
            t1 * t1 * t1 * b0[1]
            + 3.0 * t1 * t1 * t * b1[1]
            + 3.0 * t1 * t * t * b2[1]
            + t * t * t * b3[1]
        )
        cz = (
            t1 * t1 * t1 * b0[2]
            + 3.0 * t1 * t1 * t * b1[2]
            + 3.0 * t1 * t * t * b2[2]
            + t * t * t * b3[2]
        )
        dsq = (cx - query[0]) ** 2 + (cy - query[1]) ** 2 + (cz - query[2]) ** 2
        if dsq < best_dsq:
            best_dsq = dsq
            best_t   = t

    t = best_t
    for _ in range(20):
        t1 = 1.0 - t

        cx = (
            t1 * t1 * t1 * b0[0]
            + 3.0 * t1 * t1 * t * b1[0]
            + 3.0 * t1 * t * t * b2[0]
            + t * t * t * b3[0]
        )
        cy = (
            t1 * t1 * t1 * b0[1]
            + 3.0 * t1 * t1 * t * b1[1]
            + 3.0 * t1 * t * t * b2[1]
            + t * t * t * b3[1]
        )
        cz = (
            t1 * t1 * t1 * b0[2]
            + 3.0 * t1 * t1 * t * b1[2]
            + 3.0 * t1 * t * t * b2[2]
            + t * t * t * b3[2]
        )

        dc0 = -3.0 * t1 * t1
        dc1 = 3.0 * t1 * (1.0 - 3.0 * t)
        dc2 = 3.0 * t * (2.0 - 3.0 * t)
        dc3 = 3.0 * t * t
        dpx = dc0 * b0[0] + dc1 * b1[0] + dc2 * b2[0] + dc3 * b3[0]
        dpy = dc0 * b0[1] + dc1 * b1[1] + dc2 * b2[1] + dc3 * b3[1]
        dpz = dc0 * b0[2] + dc1 * b1[2] + dc2 * b2[2] + dc3 * b3[2]

        ddx = 6.0 * t1 * (b0[0] - 2.0 * b1[0] + b2[0]) + 6.0 * t * (
            b1[0] - 2.0 * b2[0] + b3[0]
        )
        ddy = 6.0 * t1 * (b0[1] - 2.0 * b1[1] + b2[1]) + 6.0 * t * (
            b1[1] - 2.0 * b2[1] + b3[1]
        )
        ddz = 6.0 * t1 * (b0[2] - 2.0 * b1[2] + b2[2]) + 6.0 * t * (
            b1[2] - 2.0 * b2[2] + b3[2]
        )

        ex = cx - query[0]
        ey = cy - query[1]
        ez = cz - query[2]
        f  = ex * dpx + ey * dpy + ez * dpz
        fp = dpx * dpx + dpy * dpy + dpz * dpz + ex * ddx + ey * ddy + ez * ddz

        if abs(fp) < 1e-12:
            break

        t_new = t - f / fp
        t     = _clamp(t_new, 0.0, 1.0)

        if abs(f) < 1e-10:
            break

    t1 = 1.0 - t
    cx = (
        t1 * t1 * t1 * b0[0]
        + 3.0 * t1 * t1 * t * b1[0]
        + 3.0 * t1 * t * t * b2[0]
        + t * t * t * b3[0]
    )
    cy = (
        t1 * t1 * t1 * b0[1]
        + 3.0 * t1 * t1 * t * b1[1]
        + 3.0 * t1 * t * t * b2[1]
        + t * t * t * b3[1]
    )
    cz = (
        t1 * t1 * t1 * b0[2]
        + 3.0 * t1 * t1 * t * b1[2]
        + 3.0 * t1 * t * t * b2[2]
        + t * t * t * b3[2]
    )
    dist_sq = (cx - query[0]) ** 2 + (cy - query[1]) ** 2 + (cz - query[2]) ** 2
    return t, dist_sq


@njit(fastmath=True, cache=True)
def _closest_point_on_bezier_face(cp, query):
    """
    Find closest point on a bicubic Bezier face.
    Newton from center, with boundary curve fallback.
    Returns (u, v, dist_sq).
    """
    u, v, dist_sq = _closest_point_on_bezier_face_newton(cp, query, 0.5, 0.5)

    if 0.0 < u < 1.0 and 0.0 < v < 1.0:
        return u, v, dist_sq

    best_u   = u
    best_v   = v
    best_dsq = dist_sq

    # Bottom boundary (v=0)
    t, dsq = _closest_point_on_cubic_curve(
        cp[0, 0], cp[1, 0], cp[2, 0], cp[3, 0], query
    )
    if dsq < best_dsq:
        best_u, best_v, best_dsq = t, 0.0, dsq

    # Top boundary (v=1)
    t, dsq = _closest_point_on_cubic_curve(
        cp[0, 3], cp[1, 3], cp[2, 3], cp[3, 3], query
    )
    if dsq < best_dsq:
        best_u, best_v, best_dsq = t, 1.0, dsq

    # Left boundary (u=0)
    t, dsq = _closest_point_on_cubic_curve(
        cp[0, 0], cp[0, 1], cp[0, 2], cp[0, 3], query
    )
    if dsq < best_dsq:
        best_u, best_v, best_dsq = 0.0, t, dsq

    # Right boundary (u=1)
    t, dsq = _closest_point_on_cubic_curve(
        cp[3, 0], cp[3, 1], cp[3, 2], cp[3, 3], query
    )
    if dsq < best_dsq:
        best_u, best_v, best_dsq = 1.0, t, dsq

    return best_u, best_v, best_dsq


# ======================================================================
# Bezier tangent-vector parallel kernel
# ======================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _bezier_vectors(points, normals, geometry, uv):
    """Compute PN Quad Bezier surface tangent vectors at given (u, v) params.

    For each query, gathers the face corner positions and normals,
    builds the 4x4 bicubic control-point grid, then evaluates the
    partial derivatives of the Bezier surface.

    Args:
        points: (num_verts, 3) vertex positions.
        normals: (num_verts, 3) vertex normals.
        geometry: (num_queries, 4) face vertex indices (-1 for tris).
        uv: (num_queries, 2) parametric coordinates.

    Returns:
        U: (num_queries, 3) dS/du tangent vectors.
        V: (num_queries, 3) dS/dv tangent vectors.
    """
    n = geometry.shape[0]
    U = np.empty((n, 3), dtype=points.dtype)
    V = np.empty((n, 3), dtype=points.dtype)

    for i in prange(n):
        cp = np.empty((4, 4, 3), dtype=points.dtype)

        i0 = geometry[i, 0]
        i1 = geometry[i, 1]
        i2 = geometry[i, 2]
        i3 = geometry[i, 3]

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        n0 = normals[i0]
        n1 = normals[i1]
        n2 = normals[i2]

        if i3 == -1:
            p3 = p2
            n3 = n2
        else:
            p3 = points[i3]
            n3 = normals[i3]

        _compute_pn_quad_cps_single(p0, p1, p2, p3, n0, n1, n2, n3, cp)
        _, _, _, dux, duy, duz, dvx, dvy, dvz = _eval_bezier_surface(
            cp, uv[i, 0], uv[i, 1]
        )

        U[i, 0] = dux
        U[i, 1] = duy
        U[i, 2] = duz
        V[i, 0] = dvx
        V[i, 1] = dvy
        V[i, 2] = dvz

    return U, V


@njit(parallel=True, fastmath=True, cache=True)
def _bezier_evaluate(points, normals, geometry, uv):
    """Evaluate PN Quad Bezier surface positions at given (u, v) params.

    For each query, gathers the face corner positions and normals,
    builds the 4x4 bicubic control-point grid, then evaluates the
    surface position.

    Args:
        points: (num_verts, 3) vertex positions.
        normals: (num_verts, 3) vertex normals.
        geometry: (num_queries, 4) face vertex indices (-1 for tris).
        uv: (num_queries, 2) parametric coordinates.

    Returns:
        positions: (num_queries, 3) surface positions.
    """
    n         = geometry.shape[0]
    positions = np.empty((n, 3), dtype=points.dtype)

    for i in prange(n):
        cp = np.empty((4, 4, 3), dtype=points.dtype)

        i0 = geometry[i, 0]
        i1 = geometry[i, 1]
        i2 = geometry[i, 2]
        i3 = geometry[i, 3]

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        n0 = normals[i0]
        n1 = normals[i1]
        n2 = normals[i2]

        if i3 == -1:
            p3 = p2
            n3 = n2
        else:
            p3 = points[i3]
            n3 = normals[i3]

        _compute_pn_quad_cps_single(p0, p1, p2, p3, n0, n1, n2, n3, cp)
        sx, sy, sz, _, _, _, _, _, _ = _eval_bezier_surface(cp, uv[i, 0], uv[i, 1])

        positions[i, 0] = sx
        positions[i, 1] = sy
        positions[i, 2] = sz

    return positions


# ======================================================================
# Bezier closest-point parallel kernel
# ======================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _bezier_sample(
    p,
    points,
    geometry,
    control_points,
    centroids,
    radiuses,
    centroid_distance_tolerance,
    iteration_count     = 100,
    iteration_tolerance = 1e-8,
    uv_border_tolerance = 1e-5,
):
    """
    Finds the closest points to a mesh using Gauss-Newton optimization
    on bicubic Bezier patches (PN Quads) with bounding-sphere culling.

    Same return signature as _bilinear_sample for drop-in compatibility.
    Bilinear weights are still computed from UV for attribute interpolation.
    """
    num_queries   = p.shape[0]
    num_faces     = geometry.shape[0]

    closest_proj  = np.empty((num_queries, 3), dtype=points.dtype)
    closest_dist  = np.empty(num_queries,      dtype=points.dtype)
    closest_w     = np.empty((num_queries, 4), dtype=points.dtype)
    closest_uv    = np.empty((num_queries, 2), dtype=points.dtype)
    closest_index = np.empty(num_queries,      dtype=geometry.dtype)

    for k in prange(num_queries):
        best_dist_sq = np.inf
        best_dist    = np.inf
        best_index   = -1
        best_u       = 0.5
        best_v       = 0.5

        for j in range(num_faces):
            # Bounding sphere culling (radiuses are squared)
            dist_to_centroid_sq = 0.0
            for i in range(3):
                diff = p[k, i] - centroids[j, i]
                dist_to_centroid_sq += diff * diff

            radius = radiuses[j] ** 0.5
            upper  = best_dist + radius
            if dist_to_centroid_sq > upper * upper:
                continue

            u, v, dist_sq = _closest_point_on_bezier_face(control_points[j], p[k])

            # UV border tolerance snapping
            if u < uv_border_tolerance:
                u = 0.0
            elif (1.0 - u) < uv_border_tolerance:
                u = 1.0
            if v < uv_border_tolerance:
                v = 0.0
            elif (1.0 - v) < uv_border_tolerance:
                v = 1.0

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_dist    = dist_sq**0.5
                best_index   = j
                best_u       = u
                best_v       = v

                if dist_sq == 0.0:
                    break

        # Write results
        closest_dist[k] = best_dist_sq
        closest_index[k] = best_index
        closest_uv[k, 0] = best_u
        closest_uv[k, 1] = best_v

        # Bilinear weights from UV (for vertex attribute interpolation)
        ou = 1.0 - best_u
        ov = 1.0 - best_v
        closest_w[k, 0] = ou * ov
        closest_w[k, 1] = best_u * ov
        closest_w[k, 2] = best_u * best_v

        if best_index >= 0 and geometry[best_index, 3] == -1:
            closest_w[k, 2] += ou * best_v
            closest_w[k, 3] = 0.0
        else:
            closest_w[k, 3] = ou * best_v

        # Compute projection from Bezier surface
        if best_index >= 0:
            sx, sy, sz, _, _, _, _, _, _ = _eval_bezier_surface(
                control_points[best_index], best_u, best_v
            )
            closest_proj[k, 0] = sx
            closest_proj[k, 1] = sy
            closest_proj[k, 2] = sz
        else:
            closest_proj[k, 0] = p[k, 0]
            closest_proj[k, 1] = p[k, 1]
            closest_proj[k, 2] = p[k, 2]

    return closest_proj, closest_dist, closest_w, closest_uv, closest_index


# --- Raycast helpers --- #


@njit(fastmath=True, cache=True)
def _ray_aabb_intersect(o, inv_d, aabb_min, aabb_max):
    """Test if ray intersects axis-aligned bounding box in forward direction."""
    t1x    = (aabb_min[0] - o[0]) * inv_d[0]
    t1y    = (aabb_min[1] - o[1]) * inv_d[1]
    t1z    = (aabb_min[2] - o[2]) * inv_d[2]

    t2x    = (aabb_max[0] - o[0]) * inv_d[0]
    t2y    = (aabb_max[1] - o[1]) * inv_d[1]
    t2z    = (aabb_max[2] - o[2]) * inv_d[2]

    tmin_x = min(t1x, t2x)
    tmin_y = min(t1y, t2y)
    tmin_z = min(t1z, t2z)

    tmax_x = max(t1x, t2x)
    tmax_y = max(t1y, t2y)
    tmax_z = max(t1z, t2z)

    tmin   = max(tmin_x, tmin_y, tmin_z)
    tmax   = min(tmax_x, tmax_y, tmax_z)

    return tmax >= tmin and tmax >= 0.0


@njit(fastmath=True, cache=True)
def _compute_face_aabbs(faces, points):
    """Precompute axis-aligned bounding box for each face.

    A tiny epsilon expansion ensures rays exactly on AABB boundaries
    are not rejected when the ray direction has zero components (the
    inv_d ~= 1e10 approximation produces tmax=0 at exact boundaries).
    """
    num_faces = faces.shape[0]
    aabb_min  = np.empty((num_faces, 3), dtype=np.float64)
    aabb_max  = np.empty((num_faces, 3), dtype=np.float64)
    aabb_eps  = 1e-10

    for fi in range(num_faces):
        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        i3 = faces[fi, 3]

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]

        if i3 < 0:
            p3 = p2
        else:
            p3 = points[i3]

        for k in range(3):
            aabb_min[fi, k] = min(p0[k], p1[k], p2[k], p3[k]) - aabb_eps
            aabb_max[fi, k] = max(p0[k], p1[k], p2[k], p3[k]) + aabb_eps

    return aabb_min, aabb_max


@njit(fastmath=True, cache=True)
def _intersect_bilinear_newton(p0, p1, p2, p3, o, d, u0, v0, eps=1e-6, max_its=20):
    """Newton iteration to find ray-bilinear patch intersection."""
    u      = u0
    v      = v0

    Cx     = p0[0] - p1[0] + p2[0] - p3[0]
    Cy     = p0[1] - p1[1] + p2[1] - p3[1]
    Cz     = p0[2] - p1[2] + p2[2] - p3[2]
    Dx     = p1[0] - p0[0]
    Dy     = p1[1] - p0[1]
    Dz     = p1[2] - p0[2]
    Ex     = p3[0] - p0[0]
    Ey     = p3[1] - p0[1]
    Ez     = p3[2] - p0[2]
    Fx     = p0[0]
    Fy     = p0[1]
    Fz     = p0[2]

    dx     = d[0]
    dy     = d[1]
    dz     = d[2]
    ox     = o[0]
    oy     = o[1]
    oz     = o[2]

    eps_sq = eps * eps

    for _ in range(max_its):
        px        = Cx * u * v + Dx * u + Ex * v + Fx
        py        = Cy * u * v + Dy * u + Ey * v + Fy
        pz        = Cz * u * v + Dz * u + Ez * v + Fz

        dux       = Cx * v + Dx
        duy       = Cy * v + Dy
        duz       = Cz * v + Dz
        dvx       = Cx * u + Ex
        dvy       = Cy * u + Ey
        dvz       = Cz * u + Ez

        diffx     = px - ox
        diffy     = py - oy
        diffz     = pz - oz

        fx        = diffy * dz - diffz * dy
        fy        = diffz * dx - diffx * dz
        fz        = diffx * dy - diffy * dx

        f_norm_sq = fx * fx + fy * fy + fz * fz
        if f_norm_sq < eps_sq:
            t = diffx * dx + diffy * dy + diffz * dz
            return t, u, v, True

        J00   = duy * dz - duz * dy
        J01   = dvy * dz - dvz * dy
        J10   = duz * dx - dux * dz
        J11   = dvz * dx - dvx * dz
        J20   = dux * dy - duy * dx
        J21   = dvx * dy - dvy * dx

        JTJ00 = J00 * J00 + J10 * J10 + J20 * J20
        JTJ01 = J00 * J01 + J10 * J11 + J20 * J21
        JTJ11 = J01 * J01 + J11 * J11 + J21 * J21

        JTf0  = J00 * fx + J10 * fy + J20 * fz
        JTf1  = J01 * fx + J11 * fy + J21 * fz

        det   = JTJ00 * JTJ11 - JTJ01 * JTJ01
        if abs(det) < 1e-12:
            return 1e308, -1.0, -1.0, False

        inv_det = 1.0 / det
        delta_u = (JTJ11 * JTf0 - JTJ01 * JTf1) * inv_det
        delta_v = (JTJ00 * JTf1 - JTJ01 * JTf0) * inv_det

        u       = u - delta_u
        v       = v - delta_v

    return 1e308, -1.0, -1.0, False


@njit(fastmath=True, cache=True)
def _intersect_face(p0, p1, p2, p3, o, d, twosided=True, eps=1e-6, max_its=20):
    """Find forward intersection of ray with a bilinear face (t >= 0 only).

    When twosided=False, back-face hits (where the geometric surface normal
    faces the same direction as the ray) are rejected.

    Uses small tolerances on (u, v) and t boundaries to avoid rejecting
    valid intersections that land exactly on a face edge or at the ray origin
    due to floating-point precision.
    """
    best_t = 1e308
    best_u = -1.0
    best_v = -1.0

    uv_tol = 1e-6
    t_tol  = 1e-10

    # precompute bilinear coefficients for back-face test and planarity check
    Cx = p0[0] - p1[0] + p2[0] - p3[0]
    Cy = p0[1] - p1[1] + p2[1] - p3[1]
    Cz = p0[2] - p1[2] + p2[2] - p3[2]
    Dx = p1[0] - p0[0]
    Dy = p1[1] - p0[1]
    Dz = p1[2] - p0[2]
    Ex = p3[0] - p0[0]
    Ey = p3[1] - p0[1]
    Ez = p3[2] - p0[2]

    t, u, v, converged = _intersect_bilinear_newton(
        p0, p1, p2, p3, o, d, 0.5, 0.5, eps, max_its
    )
    if (
        converged
        and -uv_tol <= u <= 1 + uv_tol
        and -uv_tol <= v <= 1 + uv_tol
        and t >= -t_tol
    ):
        u      = max(0.0, min(1.0, u))
        v      = max(0.0, min(1.0, v))
        t      = max(0.0, t)

        accept = True
        if not twosided:
            dux = Cx * v + Dx
            duy = Cy * v + Dy
            duz = Cz * v + Dz
            dvx = Cx * u + Ex
            dvy = Cy * u + Ey
            dvz = Cz * u + Ez
            nx  = duy * dvz - duz * dvy
            ny  = duz * dvx - dux * dvz
            nz  = dux * dvy - duy * dvx
            if nx * d[0] + ny * d[1] + nz * d[2] > 0:
                accept = False

        if accept:
            best_t = t
            best_u = u
            best_v = v

            if Cx * Cx + Cy * Cy + Cz * Cz < 1e-10:
                return best_t, best_u, best_v

    for su, sv in ((0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)):
        t, u, v, converged = _intersect_bilinear_newton(
            p0, p1, p2, p3, o, d, su, sv, eps, max_its
        )
        if (
            converged
            and -uv_tol <= u <= 1 + uv_tol
            and -uv_tol <= v <= 1 + uv_tol
            and t >= -t_tol
        ):
            u      = max(0.0, min(1.0, u))
            v      = max(0.0, min(1.0, v))
            t      = max(0.0, t)

            accept = True
            if not twosided:
                dux = Cx * v + Dx
                duy = Cy * v + Dy
                duz = Cz * v + Dz
                dvx = Cx * u + Ex
                dvy = Cy * u + Ey
                dvz = Cz * u + Ez
                nx  = duy * dvz - duz * dvy
                ny  = duz * dvx - dux * dvz
                nz  = dux * dvy - duy * dvx
                if nx * d[0] + ny * d[1] + nz * d[2] > 0:
                    accept = False

            if accept and t < best_t:
                best_t = t
                best_u = u
                best_v = v

    return best_t, best_u, best_v


@njit(fastmath=True, cache=True)
def _intersect_mesh_single(faces, points, aabb_min, aabb_max, o, d, twosided=True):
    """Find closest forward intersection (t >= 0) with AABB culling."""
    best_t    = 1e308
    best_u    = np.nan
    best_v    = np.nan
    best_face = -1

    inv_d     = np.empty(3)
    for k in range(3):
        if abs(d[k]) < 1e-30:
            inv_d[k] = np.inf if d[k] >= 0 else -np.inf
        else:
            inv_d[k] = 1.0 / d[k]

    num_faces = faces.shape[0]

    for fi in range(num_faces):
        if not _ray_aabb_intersect(o, inv_d, aabb_min[fi], aabb_max[fi]):
            continue

        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        i3 = faces[fi, 3]

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]

        if i3 < 0:
            p3 = p2
        else:
            p3 = points[i3]

        t, u, v = _intersect_face(p0, p1, p2, p3, o, d, twosided)

        if t < best_t:
            best_t    = t
            best_u    = u
            best_v    = v
            best_face = fi

    if best_t == 1e308:
        best_t = np.nan

    return best_t, best_u, best_v, best_face


@njit(parallel=True, fastmath=True, cache=True)
def _intersect_mesh_parallel(
    faces, points, aabb_min, aabb_max, origins, directions, twosided=True
):
    """Parallel forward-only ray-mesh intersection with AABB culling."""
    n            = origins.shape[0]
    t            = np.empty(n)
    uv           = np.empty((n, 2))
    face_indices = np.empty(n, dtype=np.int32)

    for i in prange(n):
        o = origins[i]
        d = directions[i]
        ti, ui, vi, fi = _intersect_mesh_single(
            faces, points, aabb_min, aabb_max, o, d, twosided
        )
        t[i] = ti
        uv[i, 0] = ui
        uv[i, 1] = vi
        face_indices[i] = fi

    return t, uv, face_indices


# --- BVH-accelerated raycast kernels --- #
#
# These mirror the linear-scan kernels above (`_intersect_mesh_parallel`,
# `_intersect_bezier_mesh_parallel`) but replace the per-face linear scan
# with a depth-first BVH traversal.  Per-face geometric tests are exactly
# the same primitives (`_intersect_face`, `_intersect_bezier_face`), so
# correctness is unchanged - only the set of faces tested per ray shrinks.
#
# BVH layout: see `cgmath/geometry/utils/_numba/_bvh.py`.

# Maximum BVH depth supported by the per-ray traversal stack.  64 covers a
# completely balanced binary tree of ~1.8 * 10^19 leaves.  The build code
# never produces a tree this deep on realistic input.
_BVH_STACK_SIZE = 64


@njit(fastmath=True, cache=True)
def _intersect_mesh_single_bvh(
    faces,
    points,
    bvh_aabb_min,
    bvh_aabb_max,
    bvh_left,
    bvh_right,
    bvh_first,
    bvh_count,
    face_perm,
    o,
    d,
    twosided=True,
):
    """Single-ray bilinear-mesh intersection accelerated by a BVH."""
    best_t    = 1e308
    best_u    = np.nan
    best_v    = np.nan
    best_face = -1

    inv_d     = np.empty(3)
    for k in range(3):
        if abs(d[k]) < 1e-30:
            inv_d[k] = np.inf if d[k] >= 0 else -np.inf
        else:
            inv_d[k] = 1.0 / d[k]

    # Iterative DFS using a fixed-size stack of node ids.
    stack = np.empty(_BVH_STACK_SIZE, dtype=np.int32)
    sp    = 0
    stack[sp] = 0  # root
    sp += 1

    while sp > 0:
        sp -= 1
        node = stack[sp]

        if not _ray_aabb_intersect(o, inv_d, bvh_aabb_min[node], bvh_aabb_max[node]):
            continue

        cnt = bvh_count[node]
        if cnt > 0:
            # Leaf - test each face in this leaf's range.
            first = bvh_first[node]
            for slot in range(first, first + cnt):
                fi = face_perm[slot]
                i0 = faces[fi, 0]
                i1 = faces[fi, 1]
                i2 = faces[fi, 2]
                i3 = faces[fi, 3]
                p0 = points[i0]
                p1 = points[i1]
                p2 = points[i2]
                if i3 < 0:
                    p3 = p2
                else:
                    p3 = points[i3]
                t, u, v = _intersect_face(p0, p1, p2, p3, o, d, twosided)
                if t < best_t:
                    best_t    = t
                    best_u    = u
                    best_v    = v
                    best_face = fi
            continue
        # Internal node - push both children, near one last so it pops first.
        left  = bvh_left[node]
        right = bvh_right[node]

        # Front-to-back ordering by ray sign on the major split axis.
        # Cheap heuristic: take the child whose AABB centroid projects
        # closer along the ray direction.
        left_dot = (
            ((bvh_aabb_min[left, 0] + bvh_aabb_max[left, 0]) * 0.5 - o[0]) * d[0]
            + ((bvh_aabb_min[left, 1] + bvh_aabb_max[left, 1]) * 0.5 - o[1]) * d[1]
            + ((bvh_aabb_min[left, 2] + bvh_aabb_max[left, 2]) * 0.5 - o[2]) * d[2]
        )
        right_dot = (
            ((bvh_aabb_min[right, 0] + bvh_aabb_max[right, 0]) * 0.5 - o[0]) * d[0]
            + ((bvh_aabb_min[right, 1] + bvh_aabb_max[right, 1]) * 0.5 - o[1]) * d[1]
            + ((bvh_aabb_min[right, 2] + bvh_aabb_max[right, 2]) * 0.5 - o[2]) * d[2]
        )

        if left_dot < right_dot:
            stack[sp] = right
            sp += 1
            stack[sp] = left
            sp += 1
        else:
            stack[sp] = left
            sp += 1
            stack[sp] = right
            sp += 1

    if best_t == 1e308:
        best_t = np.nan
    return best_t, best_u, best_v, best_face


@njit(parallel=True, fastmath=True, cache=True)
def _intersect_mesh_parallel_bvh(
    faces,
    points,
    bvh_aabb_min,
    bvh_aabb_max,
    bvh_left,
    bvh_right,
    bvh_first,
    bvh_count,
    face_perm,
    origins,
    directions,
    twosided=True,
):
    """Parallel BVH-accelerated bilinear ray-mesh intersection."""
    n            = origins.shape[0]
    t            = np.empty(n)
    uv           = np.empty((n, 2))
    face_indices = np.empty(n, dtype=np.int32)

    for i in prange(n):
        ti, ui, vi, fi = _intersect_mesh_single_bvh(
            faces,
            points,
            bvh_aabb_min,
            bvh_aabb_max,
            bvh_left,
            bvh_right,
            bvh_first,
            bvh_count,
            face_perm,
            origins[i],
            directions[i],
            twosided,
        )
        t[i] = ti
        uv[i, 0] = ui
        uv[i, 1] = vi
        face_indices[i] = fi

    return t, uv, face_indices


@njit(fastmath=True, cache=True)
def _intersect_bezier_mesh_single_bvh(
    control_points,
    bvh_aabb_min,
    bvh_aabb_max,
    bvh_left,
    bvh_right,
    bvh_first,
    bvh_count,
    face_perm,
    o,
    d,
    twosided=True,
):
    """Single-ray bicubic-Bezier mesh intersection accelerated by a BVH."""
    best_t    = 1e308
    best_u    = np.nan
    best_v    = np.nan
    best_face = -1

    inv_d     = np.empty(3)
    for k in range(3):
        if abs(d[k]) < 1e-30:
            inv_d[k] = np.inf if d[k] >= 0 else -np.inf
        else:
            inv_d[k] = 1.0 / d[k]

    stack = np.empty(_BVH_STACK_SIZE, dtype=np.int32)
    sp    = 0
    stack[sp] = 0
    sp += 1

    while sp > 0:
        sp -= 1
        node = stack[sp]

        if not _ray_aabb_intersect(o, inv_d, bvh_aabb_min[node], bvh_aabb_max[node]):
            continue

        cnt = bvh_count[node]
        if cnt > 0:
            first = bvh_first[node]
            for slot in range(first, first + cnt):
                fi = face_perm[slot]
                t, u, v = _intersect_bezier_face(control_points[fi], o, d, twosided)
                if t < best_t:
                    best_t    = t
                    best_u    = u
                    best_v    = v
                    best_face = fi
            continue
        left  = bvh_left[node]
        right = bvh_right[node]
        left_dot = (
            ((bvh_aabb_min[left, 0] + bvh_aabb_max[left, 0]) * 0.5 - o[0]) * d[0]
            + ((bvh_aabb_min[left, 1] + bvh_aabb_max[left, 1]) * 0.5 - o[1]) * d[1]
            + ((bvh_aabb_min[left, 2] + bvh_aabb_max[left, 2]) * 0.5 - o[2]) * d[2]
        )
        right_dot = (
            ((bvh_aabb_min[right, 0] + bvh_aabb_max[right, 0]) * 0.5 - o[0]) * d[0]
            + ((bvh_aabb_min[right, 1] + bvh_aabb_max[right, 1]) * 0.5 - o[1]) * d[1]
            + ((bvh_aabb_min[right, 2] + bvh_aabb_max[right, 2]) * 0.5 - o[2]) * d[2]
        )

        if left_dot < right_dot:
            stack[sp] = right
            sp += 1
            stack[sp] = left
            sp += 1
        else:
            stack[sp] = left
            sp += 1
            stack[sp] = right
            sp += 1

    if best_t == 1e308:
        best_t = np.nan
    return best_t, best_u, best_v, best_face


@njit(parallel=True, fastmath=True, cache=True)
def _intersect_bezier_mesh_parallel_bvh(
    control_points,
    bvh_aabb_min,
    bvh_aabb_max,
    bvh_left,
    bvh_right,
    bvh_first,
    bvh_count,
    face_perm,
    origins,
    directions,
    twosided=True,
):
    """Parallel BVH-accelerated bicubic Bezier ray-mesh intersection."""
    n            = origins.shape[0]
    t            = np.empty(n)
    uv           = np.empty((n, 2))
    face_indices = np.empty(n, dtype=np.int32)

    for i in prange(n):
        ti, ui, vi, fi = _intersect_bezier_mesh_single_bvh(
            control_points,
            bvh_aabb_min,
            bvh_aabb_max,
            bvh_left,
            bvh_right,
            bvh_first,
            bvh_count,
            face_perm,
            origins[i],
            directions[i],
            twosided,
        )
        t[i] = ti
        uv[i, 0] = ui
        uv[i, 1] = vi
        face_indices[i] = fi

    return t, uv, face_indices


# --- Bezier raycast helpers --- #


@njit(fastmath=True, cache=True)
def _compute_bezier_face_aabbs(control_points):
    """Precompute AABBs from bicubic Bezier control points.

    The convex hull property guarantees the surface lies within the
    bounding box of its control points, so the AABB is tight.
    """
    num_faces = control_points.shape[0]
    aabb_min  = np.empty((num_faces, 3), dtype=np.float64)
    aabb_max  = np.empty((num_faces, 3), dtype=np.float64)
    aabb_eps  = 1e-10

    for fi in range(num_faces):
        for k in range(3):
            mn = control_points[fi, 0, 0, k]
            mx = control_points[fi, 0, 0, k]
            for i in range(4):
                for j in range(4):
                    val = control_points[fi, i, j, k]
                    if val < mn:
                        mn = val
                    if val > mx:
                        mx = val
            aabb_min[fi, k] = mn - aabb_eps
            aabb_max[fi, k] = mx + aabb_eps

    return aabb_min, aabb_max


@njit(fastmath=True, cache=True)
def _intersect_bezier_newton(cp, o, d, u0, v0, eps=1e-6, max_its=20):
    """Newton iteration to find ray-bicubic Bezier patch intersection.

    Same formulation as _intersect_bilinear_newton but evaluates the
    surface and derivatives via _eval_bezier_surface.
    """
    u      = u0
    v      = v0

    dx     = d[0]
    dy     = d[1]
    dz     = d[2]
    ox     = o[0]
    oy     = o[1]
    oz     = o[2]

    eps_sq = eps * eps

    for _ in range(max_its):
        sx, sy, sz, dux, duy, duz, dvx, dvy, dvz = _eval_bezier_surface(cp, u, v)

        diffx     = sx - ox
        diffy     = sy - oy
        diffz     = sz - oz

        fx        = diffy * dz - diffz * dy
        fy        = diffz * dx - diffx * dz
        fz        = diffx * dy - diffy * dx

        f_norm_sq = fx * fx + fy * fy + fz * fz
        if f_norm_sq < eps_sq:
            t = diffx * dx + diffy * dy + diffz * dz
            return t, u, v, True

        J00   = duy * dz - duz * dy
        J01   = dvy * dz - dvz * dy
        J10   = duz * dx - dux * dz
        J11   = dvz * dx - dvx * dz
        J20   = dux * dy - duy * dx
        J21   = dvx * dy - dvy * dx

        JTJ00 = J00 * J00 + J10 * J10 + J20 * J20
        JTJ01 = J00 * J01 + J10 * J11 + J20 * J21
        JTJ11 = J01 * J01 + J11 * J11 + J21 * J21

        JTf0  = J00 * fx + J10 * fy + J20 * fz
        JTf1  = J01 * fx + J11 * fy + J21 * fz

        det   = JTJ00 * JTJ11 - JTJ01 * JTJ01
        if abs(det) < 1e-12:
            return 1e308, -1.0, -1.0, False

        inv_det = 1.0 / det
        delta_u = (JTJ11 * JTf0 - JTJ01 * JTf1) * inv_det
        delta_v = (JTJ00 * JTf1 - JTJ01 * JTf0) * inv_det

        u       = u - delta_u
        v       = v - delta_v

    return 1e308, -1.0, -1.0, False


@njit(fastmath=True, cache=True)
def _intersect_bezier_face(cp, o, d, twosided=True, eps=1e-6, max_its=20):
    """Find forward intersection of ray with a bicubic Bezier face.

    Uses a 3x3 grid of initial guesses to handle multiple roots.
    When twosided=False, back-face hits are rejected.

    Uses small tolerances on (u, v) and t boundaries to avoid rejecting
    valid intersections that land exactly on a face edge or at the ray origin
    due to floating-point precision.
    """
    best_t = 1e308
    best_u = -1.0
    best_v = -1.0

    uv_tol = 1e-6
    t_tol  = 1e-10

    for su in (0.5, 0.25, 0.75):
        for sv in (0.5, 0.25, 0.75):
            t, u, v, converged = _intersect_bezier_newton(
                cp, o, d, su, sv, eps, max_its
            )
            if (
                converged
                and -uv_tol <= u <= 1 + uv_tol
                and -uv_tol <= v <= 1 + uv_tol
                and t >= -t_tol
            ):
                u      = max(0.0, min(1.0, u))
                v      = max(0.0, min(1.0, v))
                t      = max(0.0, t)

                accept = True
                if not twosided:
                    _, _, _, dux, duy, duz, dvx, dvy, dvz = _eval_bezier_surface(
                        cp, u, v
                    )
                    nx = duy * dvz - duz * dvy
                    ny = duz * dvx - dux * dvz
                    nz = dux * dvy - duy * dvx
                    if nx * d[0] + ny * d[1] + nz * d[2] > 0:
                        accept = False

                if accept and t < best_t:
                    best_t = t
                    best_u = u
                    best_v = v

    return best_t, best_u, best_v


@njit(fastmath=True, cache=True)
def _intersect_bezier_mesh_single(
    control_points, aabb_min, aabb_max, o, d, twosided=True
):
    """Find closest forward intersection with a bicubic Bezier mesh."""
    best_t    = 1e308
    best_u    = np.nan
    best_v    = np.nan
    best_face = -1

    inv_d     = np.empty(3)
    for k in range(3):
        if abs(d[k]) < 1e-30:
            inv_d[k] = np.inf if d[k] >= 0 else -np.inf
        else:
            inv_d[k] = 1.0 / d[k]

    num_faces = control_points.shape[0]

    for fi in range(num_faces):
        if not _ray_aabb_intersect(o, inv_d, aabb_min[fi], aabb_max[fi]):
            continue

        t, u, v = _intersect_bezier_face(control_points[fi], o, d, twosided)

        if t < best_t:
            best_t    = t
            best_u    = u
            best_v    = v
            best_face = fi

    if best_t == 1e308:
        best_t = np.nan

    return best_t, best_u, best_v, best_face


@njit(parallel=True, fastmath=True, cache=True)
def _intersect_bezier_mesh_parallel(
    control_points, aabb_min, aabb_max, origins, directions, twosided=True
):
    """Parallel forward-only ray-Bezier mesh intersection with AABB culling."""
    n            = origins.shape[0]
    t            = np.empty(n)
    uv           = np.empty((n, 2))
    face_indices = np.empty(n, dtype=np.int32)

    for i in prange(n):
        o = origins[i]
        d = directions[i]
        ti, ui, vi, fi = _intersect_bezier_mesh_single(
            control_points, aabb_min, aabb_max, o, d, twosided
        )
        t[i] = ti
        uv[i, 0] = ui
        uv[i, 1] = vi
        face_indices[i] = fi

    return t, uv, face_indices