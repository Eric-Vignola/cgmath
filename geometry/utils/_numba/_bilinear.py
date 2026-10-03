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
        C       = p0[i] - p1[i] + p2[i] - p3[i]
        D       = p1[i] - p0[i]
        E       = p3[i] - p0[i]
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


# --- Exact closest point helpers --- #

# The samplers, _bilinear_sample and _bezier_sample, everything they call
# and the control points they read are compiled without fast math. numba
# runs a fresh kernel with the callees already loaded in the process and a
# cached one with the copies of them it was compiled with, and under fast
# math each copy rounds its own way: with the seed's face solve cached but
# not the loop's, as two processes compiling at once can leave it, a seed
# moved answers by 3e-16 (Bezier: 2e-9, or onto the next face), and a fresh
# kernel and a cached one disagreed by up to 4e-9. Without it every copy
# rounds alike.


@njit(cache=True)
def _clamp(x, lo, hi):
    """Clamp scalar x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _closest_point_on_face_newton(p0, p1, p2, p3, query, u0, v0, eps, max_its):
    """
    Gauss-Newton solver for closest point on bilinear patch.
    Generalized to N dimensions.

    Minimizes f(u,v) = ||S(u,v) - query||^2
    where S(u,v) = C*u*v + D*u + E*v + F

    Returns (u, v, dist_sq, flat): flat when parallel tangents (a face with
    no area there) stopped it, so (u, v) is no closest point.
    """
    ndim = p0.shape[0]
    u    = u0
    v    = v0

    C = np.empty(ndim, dtype=p0.dtype)
    D = np.empty(ndim, dtype=p0.dtype)
    E = np.empty(ndim, dtype=p0.dtype)
    F = np.empty(ndim, dtype=p0.dtype)

    for i in range(ndim):
        C[i] = p0[i] - p1[i] + p2[i] - p3[i]
        D[i] = p1[i] - p0[i]
        E[i] = p3[i] - p0[i]
        F[i] = p0[i]

    # both tests are in face terms, so they hold the same on a face of any size:
    # the step is in (u, v), and det is against the tangents' own lengths
    flat = False
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
            H00    += 2.0 * du_i * du_i
            H01    += 2.0 * du_i * dv_i
            H11    += 2.0 * dv_i * dv_i

        det = H00 * H11 - H01 * H01
        if abs(det) <= 1e-12 * H00 * H11:
            flat = True  # the tangents are parallel: no step to take
            break

        inv_det = 1.0 / det
        delta_u = (H11 * grad_u - H01 * grad_v) * inv_det
        delta_v = (H00 * grad_v - H01 * grad_u) * inv_det

        new_u   = _clamp(u - delta_u, 0.0, 1.0)
        new_v   = _clamp(v - delta_v, 0.0, 1.0)
        step    = max(abs(new_u - u), abs(new_v - v))
        u       = new_u
        v       = new_v
        if step < eps:
            break

    dist_sq = 0.0
    for i in range(ndim):
        si   = C[i] * u * v + D[i] * u + E[i] * v + F[i]
        diff = si - query[i]
        dist_sq += diff * diff

    return u, v, dist_sq, flat


@njit(cache=True)
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
        dot     += (query[i] - a[i]) * ei

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


@njit(cache=True)
def _face_size(p0, p1, p2, p3):
    """the largest coordinate difference between a face's first corner and the others"""
    size = 0.0
    for i in range(p0.shape[0]):
        size = max(size, abs(p1[i] - p0[i]), abs(p2[i] - p0[i]), abs(p3[i] - p0[i]))
    return size


@njit(cache=True)
def _face_is_convex(p0, p1, p2, p3, query):
    """
    True when f(u, v) = ||S(u, v) - query||^2 is convex over the whole face,
    so the face holds one minimum and Newton's answer is the closest point.

    The Hessian of f is 2 * (J^T J + [[0, r.C], [r.C, 0]]), J the two
    tangents and r = S - query. J is the tangents at the face centre, give
    or take |C| / sqrt(2), and r.C is largest at a corner, as S stays
    inside its corners' hull. So the Hessian is positive all over the
    face when (smallest singular value of J at the centre - |C| / sqrt(2))^2
    is above max |(corner - query).C|. N-dimensional.
    """
    ndim = p0.shape[0]
    bb   = 0.0
    be   = 0.0
    ee   = 0.0
    cc   = 0.0
    r0   = 0.0
    r1   = 0.0
    r2   = 0.0
    r3   = 0.0

    for i in range(ndim):
        ci = p0[i] - p1[i] + p2[i] - p3[i]
        bi = 0.5 * (p1[i] + p2[i] - p0[i] - p3[i])  # dS/du at the centre
        ei = 0.5 * (p2[i] + p3[i] - p0[i] - p1[i])  # dS/dv at the centre
        bb += bi * bi
        be += bi * ei
        ee += ei * ei
        cc += ci * ci
        r0 += (p0[i] - query[i]) * ci
        r1 += (p1[i] - query[i]) * ci
        r2 += (p2[i] - query[i]) * ci
        r3 += (p3[i] - query[i]) * ci

    # smallest eigenvalue of J^T J at the centre, written to keep its digits
    det = bb * ee - be * be
    if det <= 0.0:
        return False
    lam   = 2.0 * det / (bb + ee + np.sqrt((bb - ee) * (bb - ee) + 4.0 * be * be))
    slack = np.sqrt(lam) - np.sqrt(0.5 * cc)
    return slack > 0.0 and slack * slack > max(abs(r0), abs(r1), abs(r2), abs(r3))


@njit(cache=True)
def _border_point_is_closest(p0, p1, p2, p3, query, u, v):
    """
    On a face where f is convex, a point on its border that is the closest
    point of its edge is the closest point of the face when f does not drop
    going into the face: its slope across each border it sits on points out.
    """
    ndim   = p0.shape[0]
    grad_u = 0.0
    grad_v = 0.0

    for i in range(ndim):
        ci     = p0[i] - p1[i] + p2[i] - p3[i]
        di     = p1[i] - p0[i]
        ei     = p3[i] - p0[i]
        diff_i = ci * u * v + di * u + ei * v + p0[i] - query[i]
        grad_u += diff_i * (ci * v + di)
        grad_v += diff_i * (ci * u + ei)

    if (u == 0.0 and grad_u < 0.0) or (u == 1.0 and grad_u > 0.0):
        return False
    if (v == 0.0 and grad_v < 0.0) or (v == 1.0 and grad_v > 0.0):
        return False
    return True


@njit(cache=True)
def _corners_beyond(p0, p1, p2, p3, query, u, v):
    """
    How far the face's corners stay from the query along n, the direction
    from the query to S(u, v): the smallest (corner - query).n. The face
    lies inside its corners' hull, so no point of it is closer than that.
    """
    ndim    = p0.shape[0]
    dist_sq = 0.0
    l0      = 0.0
    l1      = 0.0
    l2      = 0.0
    l3      = 0.0

    for i in range(ndim):
        ci     = p0[i] - p1[i] + p2[i] - p3[i]
        di     = p1[i] - p0[i]
        ei     = p3[i] - p0[i]
        diff_i = ci * u * v + di * u + ei * v + p0[i] - query[i]
        dist_sq += diff_i * diff_i
        l0      += (p0[i] - query[i]) * diff_i
        l1      += (p1[i] - query[i]) * diff_i
        l2      += (p2[i] - query[i]) * diff_i
        l3      += (p3[i] - query[i]) * diff_i

    if dist_sq == 0.0:
        return 0.0
    return min(l0, l1, l2, l3) / np.sqrt(dist_sq)


@njit(cache=True)
def _poly_value(c, deg, x):
    """value and slope at x of c[0] + c[1] * x + ... + c[deg] * x^deg"""
    p  = c[deg]
    dp = 0.0
    for i in range(deg - 1, -1, -1):
        dp = dp * x + p
        p  = p * x + c[i]
    return p, dp


@njit(cache=True)
def _poly_bracket_root(c, deg, lo, hi, p_lo, eps, max_its):
    """
    The root of polynomial c in [lo, hi], where its sign changes once (p_lo
    is its value at lo). Newton from the middle, kept inside the bracket
    by bisection; stops once a step moves less than eps.
    """
    x = 0.5 * (lo + hi)
    for _ in range(max_its):
        p, dp = _poly_value(c, deg, x)
        if p == 0.0:
            break

        # keep the half that still holds the sign change
        if (p < 0.0) == (p_lo < 0.0):
            lo   = x
            p_lo = p
        else:
            hi = x

        new_x = lo - 1.0  # outside: bisect when there is no slope
        if dp != 0.0:
            new_x = x - p / dp
        if not lo < new_x < hi:
            new_x = 0.5 * (lo + hi)

        step = abs(new_x - x)
        x    = new_x
        if step < eps:
            break

    return x


@njit(cache=True)
def _poly_unit_roots(chain, deg, roots, counts, eps, max_its):
    """
    Real roots inside (0, 1) of the polynomial in chain[deg], in order.

    chain[k] receives its derivatives, chain[k] of degree k, and roots[k]
    with counts[k] the roots of each. A polynomial is monotonic between
    the roots of its derivative, so each of those pieces holds at most one
    root, solved by _poly_bracket_root. Only the roots of chain[deg] honour
    eps and max_its; the derivatives' roots, which split the range, are
    always solved to full precision.
    """
    for k in range(deg, 0, -1):
        for i in range(k):
            chain[k - 1, i] = (i + 1) * chain[k, i + 1]

    counts[0] = 0
    for k in range(1, deg + 1):
        counts[k] = 0
        tol       = eps if k == deg else 1e-14
        its       = max_its if k == deg else 60

        lo = 0.0
        p_lo, _ = _poly_value(chain[k], k, lo)
        for j in range(counts[k - 1] + 1):
            hi = roots[k - 1, j] if j < counts[k - 1] else 1.0
            p_hi, _ = _poly_value(chain[k], k, hi)
            if (p_lo < 0.0 < p_hi) or (p_hi < 0.0 < p_lo):
                roots[k, counts[k]] = _poly_bracket_root(
                    chain[k], k, lo, hi, p_lo, tol, its
                )
                counts[k] += 1
            lo   = hi
            p_lo = p_hi


@njit(cache=True)
def _closest_point_inside_face(p0, p1, p2, p3, query, eps, max_its):
    """
    The closest of the points inside a bilinear face where the gradient of
    f(u, v) = ||S(u, v) - query||^2 vanishes. N-dimensional.

    With a(v) = p0 - query + E v and b(v) = D + C v, S - query is
    a + u b, so df/du = 0 gives u = -N / M, N = a.b and M = b.b. Putting
    it in df/dv = 0, (a + u b).(E + C u) = 0, and clearing M leaves

        P(v) = (a.E) M^2 - (a.C + b.E) N M + (b.C) N^2

    of degree 5, whose roots in (0, 1) are the candidates. The roots of P'
    join them, so a double root of P is not missed. Where M is 0 the
    whole isoline is one point, which an edge already holds.

    Returns (u, v, dist_sq, found).
    """
    ndim = p0.shape[0]

    # in face terms: the polynomial is the same for the face at any size
    size = _face_size(p0, p1, p2, p3)
    if size == 0.0:
        return 0.0, 0.0, 0.0, False
    scale = 1.0 / size

    wD = 0.0
    wE = 0.0
    wC = 0.0
    DD = 0.0
    DE = 0.0
    DC = 0.0
    EE = 0.0
    EC = 0.0
    CC = 0.0
    for i in range(ndim):
        ci = (p0[i] - p1[i] + p2[i] - p3[i]) * scale
        di = (p1[i] - p0[i]) * scale
        ei = (p3[i] - p0[i]) * scale
        wi = (p0[i] - query[i]) * scale
        wD += wi * di
        wE += wi * ei
        wC += wi * ci
        DD += di * di
        DE += di * ei
        DC += di * ci
        EE += ei * ei
        EC += ei * ci
        CC += ci * ci

    # N = n0 + n1 v + n2 v^2 and M = m0 + m1 v + m2 v^2
    n0 = wD
    n1 = wC + DE
    n2 = EC
    m0 = DD
    m1 = 2.0 * DC
    m2 = CC

    # M^2, N M and N^2
    mm0 = m0 * m0
    mm1 = 2.0 * m0 * m1
    mm2 = m1 * m1 + 2.0 * m0 * m2
    mm3 = 2.0 * m1 * m2
    mm4 = m2 * m2
    nm0 = n0 * m0
    nm1 = n0 * m1 + n1 * m0
    nm2 = n0 * m2 + n1 * m1 + n2 * m0
    nm3 = n1 * m2 + n2 * m1
    nm4 = n2 * m2
    nn0 = n0 * n0
    nn1 = 2.0 * n0 * n1
    nn2 = n1 * n1 + 2.0 * n0 * n2
    nn3 = 2.0 * n1 * n2
    nn4 = n2 * n2

    # a.E = wE + EE v, a.C + b.E = n1 + 2 EC v, b.C = DC + CC v
    chain   = np.empty((6, 6))
    c       = chain[5]
    c[0]    = wE * mm0 - n1 * nm0 + DC * nn0
    c[1]    = wE * mm1 + EE * mm0 - n1 * nm1 - 2.0 * EC * nm0 + DC * nn1 + CC * nn0
    c[2]    = wE * mm2 + EE * mm1 - n1 * nm2 - 2.0 * EC * nm1 + DC * nn2 + CC * nn1
    c[3]    = wE * mm3 + EE * mm2 - n1 * nm3 - 2.0 * EC * nm2 + DC * nn3 + CC * nn2
    c[4]    = wE * mm4 + EE * mm3 - n1 * nm4 - 2.0 * EC * nm3 + DC * nn4 + CC * nn3
    c[5]    = EE * mm4 - 2.0 * EC * nm4 + CC * nn4

    largest = 0.0
    for i in range(6):
        largest = max(largest, abs(c[i]))
    if largest == 0.0:
        return 0.0, 0.0, 0.0, False  # P is 0: every candidate is on an edge
    for i in range(6):
        c[i] /= largest

    roots  = np.empty((6, 6))
    counts = np.empty(6, dtype=np.int64)
    _poly_unit_roots(chain, 5, roots, counts, eps, max_its)

    found    = False
    best_u   = 0.0
    best_v   = 0.0
    best_dsq = 0.0
    m_tiny   = 1e-12 * (m0 + abs(m1) + m2)

    for k in range(4, 6):
        for j in range(counts[k]):
            v = roots[k, j]
            m = m0 + m1 * v + m2 * v * v
            if m <= m_tiny:
                continue
            u = _clamp(-(n0 + n1 * v + n2 * v * v) / m, 0.0, 1.0)

            dist_sq = 0.0
            for i in range(ndim):
                si = (
                    (p0[i] - p1[i] + p2[i] - p3[i]) * u * v
                    + (p1[i] - p0[i]) * u
                    + (p3[i] - p0[i]) * v
                    + p0[i]
                )
                diff = si - query[i]
                dist_sq += diff * diff

            if not found or dist_sq < best_dsq:
                found    = True
                best_u   = u
                best_v   = v
                best_dsq = dist_sq

    return best_u, best_v, best_dsq, found


@njit(cache=True)
def _closest_point_on_face(
    p0, p1, p2, p3, query, eps=1e-6, max_its=20, bound_dsq=np.inf
):
    """
    Exact closest point on a bilinear face. N-dimensional.

    The closest point is a corner, the closest point of one of the 4
    straight edges, or a point inside where the gradient of
    ||S(u, v) - query||^2 vanishes. Newton from (0.5, 0.5) finds the inside
    one, and the edges are solved. The best of those is the closest point
    when it is on the query to within rounding, when _face_is_convex
    proves the face holds a single minimum and nothing inside is closer,
    or when every corner lies beyond the plane through it square to the
    query. Otherwise every inside candidate is solved for
    (_closest_point_inside_face) and the closest of them all wins.

    eps and max_its stop Newton and the candidates' root solve.
    bound_dsq is the caller's best so far: a face whose corners all lie
    beyond it, along the direction to the best point, cannot come closer
    and is not solved inside; its dist_sq is then no smaller than
    bound_dsq.
    Returns (u, v, dist_sq).
    """
    u, v, dist_sq, flat = _closest_point_on_face_newton(
        p0, p1, p2, p3, query, 0.5, 0.5, eps, max_its
    )

    # nothing is closer than a point on the query, to within rounding
    size     = _face_size(p0, p1, p2, p3)
    on_query = 1e-24 * size * size

    # Newton's answer inside the face stands when the face holds a single
    # minimum or the answer is on the query; a face with no area (its
    # corners on a line) has its closest point on an edge
    convex = not flat and _face_is_convex(p0, p1, p2, p3, query)
    inside = not flat and 0.0 < u < 1.0 and 0.0 < v < 1.0
    if inside and (convex or dist_sq <= on_query):
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

    if best_dsq <= on_query or (
        convex and _border_point_is_closest(p0, p1, p2, p3, query, best_u, best_v)
    ):
        return best_u, best_v, best_dsq

    beyond = _corners_beyond(p0, p1, p2, p3, query, best_u, best_v)
    if beyond >= np.sqrt(best_dsq) - 1e-12 * size or (
        beyond > 0.0 and beyond * beyond >= bound_dsq
    ):
        return best_u, best_v, best_dsq

    u, v, dsq, found = _closest_point_inside_face(p0, p1, p2, p3, query, eps, max_its)
    if found and dsq < best_dsq:
        best_u   = u
        best_v   = v
        best_dsq = dsq

    return best_u, best_v, best_dsq


@njit(cache=True)
def _gather_face(points, geometry, j, p0, p1, p2, p3):
    """fills p0..p3 with face j's corners, a triangle's last one twice"""
    for i in range(p0.shape[0]):
        p0[i] = points[geometry[j, 0], i]
        p1[i] = points[geometry[j, 1], i]
        p2[i] = points[geometry[j, 2], i]

    i3 = geometry[j, 3]
    if i3 < 0:
        for i in range(p0.shape[0]):
            p3[i] = p2[i]
    else:
        for i in range(p0.shape[0]):
            p3[i] = points[i3, i]


@njit(parallel=True, cache=True)
def _bilinear_sample(
    p,
    points,
    geometry,
    centroids,
    radiuses,
    centroid_distance_tolerance,
    seed_faces,
    iteration_count     = 100,
    iteration_tolerance = 1e-8,
    uv_border_tolerance = 1e-5,
):
    """
    Finds the closest points to a mesh, the exact closest point of each
    bilinear patch (_closest_point_on_face), with adaptive bounding-sphere
    culling.

    seed_faces holds, per query, a face to solve first, or -1. Its closest
    point bounds the query's distance, so the faces that lie beyond it are
    skipped: any face gives the same answer, one near the query (sample()
    passes a face of its nearest vertex) gives it soonest.
    """
    ndim          = p.shape[1]
    num_queries   = p.shape[0]
    num_faces     = geometry.shape[0]

    closest_proj  = np.empty(p.shape,          dtype=points.dtype)
    closest_dist  = np.empty(num_queries,      dtype=points.dtype)
    closest_w     = np.empty((num_queries, 4), dtype=points.dtype)
    closest_uv    = np.empty((num_queries, 2), dtype=points.dtype)
    closest_index = np.empty(num_queries,      dtype=geometry.dtype)

    eps     = iteration_tolerance
    max_its = iteration_count
    radii   = np.sqrt(radiuses)  # the radiuses are squared

    for k in prange(num_queries):
        best_dist_sq = np.inf
        best_dist    = np.inf
        best_index   = -1
        best_u       = 0.5
        best_v       = 0.5

        p0 = np.empty(ndim, dtype=points.dtype)
        p1 = np.empty(ndim, dtype=points.dtype)
        p2 = np.empty(ndim, dtype=points.dtype)
        p3 = np.empty(ndim, dtype=points.dtype)

        # the seed face's closest point: no winner is farther away
        seed      = seed_faces[k]
        seed_u    = 0.5
        seed_v    = 0.5
        seed_dsq  = np.inf
        seed_dist = np.inf
        if seed >= 0:
            _gather_face(points, geometry, seed, p0, p1, p2, p3)
            seed_u, seed_v, seed_dsq = _closest_point_on_face(
                p0, p1, p2, p3, p[k], eps, max_its
            )
            seed_dist = seed_dsq**0.5

        for j in range(num_faces):
            # Adaptive bounding sphere culling
            dist_to_centroid_sq = 0.0
            for i in range(ndim):
                diff = p[k, i] - centroids[j, i]
                dist_to_centroid_sq += diff * diff

            # past the seed's distance, with a millionth of the face to spare
            # for rounding, a face can neither win nor tie
            radius = radii[j]
            reach  = seed_dist + 1e-6 * radius
            upper  = min(best_dist, reach) + radius
            if dist_to_centroid_sq > upper * upper:
                continue

            if j == seed:
                u       = seed_u
                v       = seed_v
                dist_sq = seed_dsq
            else:
                _gather_face(points, geometry, j, p0, p1, p2, p3)
                u, v, dist_sq = _closest_point_on_face(
                    p0, p1, p2, p3, p[k], eps, max_its, min(best_dist_sq, reach * reach)
                )

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
        closest_dist[k]  = best_dist_sq
        closest_index[k] = best_index
        closest_uv[k, 0] = best_u
        closest_uv[k, 1] = best_v

        ou               = 1.0 - best_u
        ov               = 1.0 - best_v
        closest_w[k, 0]  = ou * ov
        closest_w[k, 1]  = best_u * ov
        closest_w[k, 2]  = best_u * best_v

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

# without fast math: _bezier_sample reads them (see the closest point helpers)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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

# without fast math: _bezier_sample calls them (see the closest point helpers)


@njit(cache=True)
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

            sx  += w * cpx
            sy  += w * cpy
            sz  += w * cpz
            dux += wu * cpx
            duy += wu * cpy
            duz += wu * cpz
            dvx += wv * cpx
            dvy += wv * cpy
            dvz += wv * cpz

    return sx, sy, sz, dux, duy, duz, dvx, dvy, dvz


@njit(cache=True)
def _eval_bezier_second(cp, u, v):
    """
    Second partial derivatives of a bicubic Bezier surface.

    Returns:
        (uux, uuy, uuz, uvx, uvy, uvz, vvx, vvy, vvz)
    """
    u1 = 1.0 - u
    v1 = 1.0 - v
    bu = (u1 * u1 * u1, 3.0 * u1 * u1 * u, 3.0 * u1 * u * u, u * u * u)
    bv = (v1 * v1 * v1, 3.0 * v1 * v1 * v, 3.0 * v1 * v * v, v * v * v)
    dbu = (
        -3.0 * u1 * u1,
        3.0 * u1 * (1.0 - 3.0 * u),
        3.0 * u * (2.0 - 3.0 * u),
        3.0 * u * u,
    )
    dbv = (
        -3.0 * v1 * v1,
        3.0 * v1 * (1.0 - 3.0 * v),
        3.0 * v * (2.0 - 3.0 * v),
        3.0 * v * v,
    )
    ddu = (6.0 * u1, 18.0 * u - 12.0, 6.0 - 18.0 * u, 6.0 * u)
    ddv = (6.0 * v1, 18.0 * v - 12.0, 6.0 - 18.0 * v, 6.0 * v)

    uux = 0.0
    uuy = 0.0
    uuz = 0.0
    uvx = 0.0
    uvy = 0.0
    uvz = 0.0
    vvx = 0.0
    vvy = 0.0
    vvz = 0.0
    for i in range(4):
        for j in range(4):
            wuu = ddu[i] * bv[j]
            wuv = dbu[i] * dbv[j]
            wvv = bu[i] * ddv[j]
            uux += wuu * cp[i, j, 0]
            uuy += wuu * cp[i, j, 1]
            uuz += wuu * cp[i, j, 2]
            uvx += wuv * cp[i, j, 0]
            uvy += wuv * cp[i, j, 1]
            uvz += wuv * cp[i, j, 2]
            vvx += wvv * cp[i, j, 0]
            vvy += wvv * cp[i, j, 1]
            vvz += wvv * cp[i, j, 2]

    return uux, uuy, uuz, uvx, uvy, uvz, vvx, vvy, vvz


@njit(cache=True)
def _closest_point_on_bezier_face_newton(cp, query, u0, v0, eps=1e-6, max_its=20):
    """
    Newton iteration for closest point on bicubic Bezier patch.

    A coordinate on the face's border whose slope points out of the face
    is held there, and the step solves for the other one only. The step
    is Gauss-Newton's; after 3 steps, which Gauss-Newton takes slowly
    away from the face, the full Newton step, with the surface's
    curvature, is tried first where its Hessian is positive. It only moves
    to a closer point: a step that is not closer is halved until it is,
    and where no Newton step gets closer it steps down the gradient
    instead. It stops once a step would move (u, v) less than eps, or
    where no step gets closer.
    Returns (u, v, dist_sq, flat), flat when parallel tangents were met
    where it stopped.
    """
    u = u0
    v = v0

    sx, sy, sz, dux, duy, duz, dvx, dvy, dvz = _eval_bezier_surface(cp, u, v)
    dist_sq = (sx - query[0]) ** 2 + (sy - query[1]) ** 2 + (sz - query[2]) ** 2

    # both tests are in face terms, as in _closest_point_on_face_newton
    flat = False
    for it in range(max_its):
        diffx  = sx - query[0]
        diffy  = sy - query[1]
        diffz  = sz - query[2]

        grad_u = 2.0 * (diffx * dux + diffy * duy + diffz * duz)
        grad_v = 2.0 * (diffx * dvx + diffy * dvy + diffz * dvz)

        H00    = 2.0 * (dux * dux + duy * duy + duz * duz)
        H01    = 2.0 * (dux * dvx + duy * dvy + duz * dvz)
        H11    = 2.0 * (dvx * dvx + dvy * dvy + dvz * dvz)
        flat   = abs(H00 * H11 - H01 * H01) <= 1e-12 * H00 * H11  # parallel tangents

        # the border holds a coordinate whose slope points out of the face
        hold_u = (u == 0.0 and grad_u > 0.0) or (u == 1.0 and grad_u < 0.0)
        hold_v = (v == 0.0 and grad_v > 0.0) or (v == 1.0 and grad_v < 0.0)
        if hold_u and hold_v:
            break  # a corner nothing inside the face improves on

        closer = False
        for attempt in range(3):
            if attempt < 2:
                if attempt == 0:
                    if it < 3:
                        continue
                    uux, uuy, uuz, uvx, uvy, uvz, vvx, vvy, vvz = _eval_bezier_second(
                        cp, u, v
                    )
                    A00 = H00 + 2.0 * (diffx * uux + diffy * uuy + diffz * uuz)
                    A01 = H01 + 2.0 * (diffx * uvx + diffy * uvy + diffz * uvz)
                    A11 = H11 + 2.0 * (diffx * vvx + diffy * vvy + diffz * vvz)
                else:
                    A00 = H00
                    A01 = H01
                    A11 = H11

                # no step unless the Hessian says a minimum is ahead
                if hold_u:
                    if A11 <= 0.0:
                        continue
                    delta_u = 0.0
                    delta_v = grad_v / A11
                elif hold_v:
                    if A00 <= 0.0:
                        continue
                    delta_u = grad_u / A00
                    delta_v = 0.0
                else:
                    det = A00 * A11 - A01 * A01
                    if A00 <= 0.0 or det <= 1e-12 * A00 * A11:
                        continue
                    delta_u = (A11 * grad_u - A01 * grad_v) / det
                    delta_v = (A00 * grad_v - A01 * grad_u) / det
            else:
                # down the gradient, where the border lets it go, as far
                # as the Gauss-Newton curvature says
                free_u = 0.0 if hold_u else grad_u
                free_v = 0.0 if hold_v else grad_v
                slope  = free_u * free_u + free_v * free_v
                curve = (
                    free_u * free_u * H00
                    + 2.0 * free_u * free_v * H01
                    + free_v * free_v * H11
                )
                if slope == 0.0 or curve <= 0.0:
                    break
                delta_u = free_u * slope / curve
                delta_v = free_v * slope / curve

            # halve the step until it lands closer; one under eps is done
            new_u = _clamp(u - delta_u, 0.0, 1.0)
            new_v = _clamp(v - delta_v, 0.0, 1.0)
            step  = max(abs(new_u - u), abs(new_v - v))
            for _ in range(8):
                if step < eps:
                    break
                nsx, nsy, nsz, ndux, nduy, nduz, ndvx, ndvy, ndvz = (
                    _eval_bezier_surface(cp, new_u, new_v)
                )
                new_dsq = (
                    (nsx - query[0]) ** 2
                    + (nsy - query[1]) ** 2
                    + (nsz - query[2]) ** 2
                )
                if new_dsq < dist_sq:
                    closer = True
                    break
                new_u = 0.5 * (u + new_u)
                new_v = 0.5 * (v + new_v)
                step  = 0.5 * step
            if closer or step < eps:
                break
        if not closer:
            break

        u       = new_u
        v       = new_v
        dist_sq = new_dsq
        sx      = nsx
        sy      = nsy
        sz      = nsz
        dux     = ndux
        duy     = nduy
        duz     = nduz
        dvx     = ndvx
        dvy     = ndvy
        dvz     = ndvz
        if step < eps:
            break

    return u, v, dist_sq, flat


@njit(cache=True)
def _cubic_curve_dist_sq(b0, b1, b2, b3, query, t):
    """squared distance from query to a cubic Bezier curve's point at t"""
    t1      = 1.0 - t
    w0      = t1 * t1 * t1
    w1      = 3.0 * t1 * t1 * t
    w2      = 3.0 * t1 * t * t
    w3      = t * t * t
    dist_sq = 0.0
    for k in range(3):
        diff = w0 * b0[k] + w1 * b1[k] + w2 * b2[k] + w3 * b3[k] - query[k]
        dist_sq += diff * diff
    return dist_sq


@njit(cache=True)
def _closest_point_on_cubic_curve(b0, b1, b2, b3, query, eps=1e-6, max_its=20):
    """
    Exact closest point on a cubic Bezier curve to a query point.

    With the curve B(t) = a0 + a1 t + a2 t^2 + a3 t^3, the slope of
    ||B(t) - query||^2 / 2 is (B(t) - query).B'(t), of degree 5: the
    closest point is an end or one of its roots in (0, 1). The roots of
    its derivative join them, so a double root is not missed.
    eps and max_its stop the root solve.
    Returns (t, dist_sq).
    """
    best_t   = 0.0
    best_dsq = _cubic_curve_dist_sq(b0, b1, b2, b3, query, 0.0)
    dsq      = _cubic_curve_dist_sq(b0, b1, b2, b3, query, 1.0)
    if dsq < best_dsq:
        best_t   = 1.0
        best_dsq = dsq

    # in curve terms: the polynomial is the same for the curve at any size
    size = 0.0
    for k in range(3):
        size = max(size, abs(b1[k] - b0[k]), abs(b2[k] - b0[k]), abs(b3[k] - b0[k]))
    if size == 0.0:
        return best_t, best_dsq
    scale = 1.0 / size

    e0a1  = 0.0
    e0a2  = 0.0
    e0a3  = 0.0
    a1a1  = 0.0
    a1a2  = 0.0
    a1a3  = 0.0
    a2a2  = 0.0
    a2a3  = 0.0
    a3a3  = 0.0
    for k in range(3):
        e0 = (b0[k] - query[k]) * scale
        a1 = 3.0 * (b1[k] - b0[k]) * scale
        a2 = 3.0 * (b2[k] - 2.0 * b1[k] + b0[k]) * scale
        a3 = (b3[k] - 3.0 * b2[k] + 3.0 * b1[k] - b0[k]) * scale
        e0a1 += e0 * a1
        e0a2 += e0 * a2
        e0a3 += e0 * a3
        a1a1 += a1 * a1
        a1a2 += a1 * a2
        a1a3 += a1 * a3
        a2a2 += a2 * a2
        a2a3 += a2 * a3
        a3a3 += a3 * a3

    chain   = np.empty((6, 6))
    c       = chain[5]
    c[0]    = e0a1
    c[1]    = 2.0 * e0a2 + a1a1
    c[2]    = 3.0 * (e0a3 + a1a2)
    c[3]    = 4.0 * a1a3 + 2.0 * a2a2
    c[4]    = 5.0 * a2a3
    c[5]    = 3.0 * a3a3

    largest = 0.0
    for i in range(6):
        largest = max(largest, abs(c[i]))
    if largest == 0.0:
        return best_t, best_dsq
    for i in range(6):
        c[i] /= largest

    roots  = np.empty((6, 6))
    counts = np.empty(6, dtype=np.int64)
    _poly_unit_roots(chain, 5, roots, counts, eps, max_its)

    for k in range(4, 6):
        for j in range(counts[k]):
            t   = roots[k, j]
            dsq = _cubic_curve_dist_sq(b0, b1, b2, b3, query, t)
            if dsq < best_dsq:
                best_t   = t
                best_dsq = dsq

    return best_t, best_dsq


@njit(cache=True)
def _box_dist_sq(b0, b1, b2, b3, query):
    """squared distance from query to the box around 4 points, 0 inside it"""
    dist_sq = 0.0
    for k in range(3):
        lo = min(b0[k], b1[k], b2[k], b3[k])
        hi = max(b0[k], b1[k], b2[k], b3[k])
        if query[k] < lo:
            dist_sq += (lo - query[k]) ** 2
        elif query[k] > hi:
            dist_sq += (query[k] - hi) ** 2
    return dist_sq


@njit(cache=True)
def _control_points_beyond(cp, query, nx, ny, nz):
    """
    How far the control points stay from the query along the direction
    (nx, ny, nz): the smallest (control point - query).n, n of unit length.
    The face lies inside its control points' hull, so no point of it is
    closer than that.
    """
    beyond = (
        (cp[0, 0, 0] - query[0]) * nx
        + (cp[0, 0, 1] - query[1]) * ny
        + (cp[0, 0, 2] - query[2]) * nz
    )
    for i in range(4):
        for j in range(4):
            beyond = min(
                beyond,
                (cp[i, j, 0] - query[0]) * nx
                + (cp[i, j, 1] - query[1]) * ny
                + (cp[i, j, 2] - query[2]) * nz,
            )
    return beyond / np.sqrt(nx * nx + ny * ny + nz * nz)


@njit(cache=True)
def _closest_point_on_bezier_face(cp, query, eps=1e-6, max_its=20, bound_dsq=np.inf):
    """
    Find closest point on a bicubic Bezier face.

    Newton from the centre. Its answer is the closest point when it is on
    the query to within rounding, or when every control point lies beyond
    the plane through it square to the query. Otherwise Newton runs again
    from the (u, v) of the control point closest to the query, then each
    of the 4 boundary curves, corners included, is solved whose control
    points' box is closer than the best point so far: a curve stays inside
    that box, so the others cannot be closer.

    bound_dsq is the caller's best so far: a face whose control points all
    lie beyond it, along the direction to the closest control point or
    to Newton's answer, cannot come closer and is cut short; its dist_sq
    is then no smaller than bound_dsq.
    Returns (u, v, dist_sq).
    """
    near_i   = 0
    near_j   = 0
    near_dsq = 0.0
    lo_x     = cp[0, 0, 0]
    hi_x     = cp[0, 0, 0]
    lo_y     = cp[0, 0, 1]
    hi_y     = cp[0, 0, 1]
    lo_z     = cp[0, 0, 2]
    hi_z     = cp[0, 0, 2]
    for i in range(4):
        for j in range(4):
            dsq = (
                (cp[i, j, 0] - query[0]) ** 2
                + (cp[i, j, 1] - query[1]) ** 2
                + (cp[i, j, 2] - query[2]) ** 2
            )
            if (i == 0 and j == 0) or dsq < near_dsq:
                near_i   = i
                near_j   = j
                near_dsq = dsq
            lo_x = min(lo_x, cp[i, j, 0])
            hi_x = max(hi_x, cp[i, j, 0])
            lo_y = min(lo_y, cp[i, j, 1])
            hi_y = max(hi_y, cp[i, j, 1])
            lo_z = min(lo_z, cp[i, j, 2])
            hi_z = max(hi_z, cp[i, j, 2])
    size = max(hi_x - lo_x, hi_y - lo_y, hi_z - lo_z)

    if near_dsq > 0.0:
        beyond = _control_points_beyond(
            cp,
            query,
            cp[near_i, near_j, 0] - query[0],
            cp[near_i, near_j, 1] - query[1],
            cp[near_i, near_j, 2] - query[2],
        )
        if beyond > 0.0 and beyond * beyond >= bound_dsq:
            return 0.5, 0.5, beyond * beyond

    best_u, best_v, best_dsq, _ = _closest_point_on_bezier_face_newton(
        cp, query, 0.5, 0.5, eps, max_its
    )

    # nothing is closer than a point on the query, to within rounding
    if best_dsq <= 1e-24 * size * size:
        return best_u, best_v, best_dsq

    sx, sy, sz, _, _, _, _, _, _ = _eval_bezier_surface(cp, best_u, best_v)
    beyond = _control_points_beyond(
        cp, query, sx - query[0], sy - query[1], sz - query[2]
    )
    if beyond >= np.sqrt(best_dsq) - 1e-12 * size or (
        beyond > 0.0 and beyond * beyond >= bound_dsq
    ):
        return best_u, best_v, best_dsq

    # a second start, at the control point closest to the query
    u, v, dsq, _ = _closest_point_on_bezier_face_newton(
        cp, query, near_i / 3.0, near_j / 3.0, eps, max_its
    )
    if dsq < best_dsq:
        best_u, best_v, best_dsq = u, v, dsq

    # Bottom boundary (v=0)
    if _box_dist_sq(cp[0, 0], cp[1, 0], cp[2, 0], cp[3, 0], query) < best_dsq:
        t, dsq = _closest_point_on_cubic_curve(
            cp[0, 0], cp[1, 0], cp[2, 0], cp[3, 0], query, eps, max_its
        )
        if dsq < best_dsq:
            best_u, best_v, best_dsq = t, 0.0, dsq

    # Top boundary (v=1)
    if _box_dist_sq(cp[0, 3], cp[1, 3], cp[2, 3], cp[3, 3], query) < best_dsq:
        t, dsq = _closest_point_on_cubic_curve(
            cp[0, 3], cp[1, 3], cp[2, 3], cp[3, 3], query, eps, max_its
        )
        if dsq < best_dsq:
            best_u, best_v, best_dsq = t, 1.0, dsq

    # Left boundary (u=0)
    if _box_dist_sq(cp[0, 0], cp[0, 1], cp[0, 2], cp[0, 3], query) < best_dsq:
        t, dsq = _closest_point_on_cubic_curve(
            cp[0, 0], cp[0, 1], cp[0, 2], cp[0, 3], query, eps, max_its
        )
        if dsq < best_dsq:
            best_u, best_v, best_dsq = 0.0, t, dsq

    # Right boundary (u=1)
    if _box_dist_sq(cp[3, 0], cp[3, 1], cp[3, 2], cp[3, 3], query) < best_dsq:
        t, dsq = _closest_point_on_cubic_curve(
            cp[3, 0], cp[3, 1], cp[3, 2], cp[3, 3], query, eps, max_its
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


@njit(parallel=True, cache=True)
def _bezier_sample(
    p,
    points,
    geometry,
    control_points,
    centroids,
    radiuses,
    centroid_distance_tolerance,
    seed_faces,
    iteration_count     = 100,
    iteration_tolerance = 1e-8,
    uv_border_tolerance = 1e-5,
):
    """
    Finds the closest points to a mesh on bicubic Bezier patches (PN
    Quads), each by _closest_point_on_bezier_face, with bounding-sphere
    culling.

    Same arguments and return signature as _bilinear_sample for drop-in
    compatibility, seed_faces included. Bilinear weights are still
    computed from UV for attribute interpolation.
    """
    num_queries   = p.shape[0]
    num_faces     = geometry.shape[0]

    closest_proj  = np.empty((num_queries, 3), dtype=points.dtype)
    closest_dist  = np.empty(num_queries,      dtype=points.dtype)
    closest_w     = np.empty((num_queries, 4), dtype=points.dtype)
    closest_uv    = np.empty((num_queries, 2), dtype=points.dtype)
    closest_index = np.empty(num_queries,      dtype=geometry.dtype)

    eps     = iteration_tolerance
    max_its = iteration_count
    radii   = np.sqrt(radiuses)  # the radiuses are squared

    for k in prange(num_queries):
        best_dist_sq = np.inf
        best_dist    = np.inf
        best_index   = -1
        best_u       = 0.5
        best_v       = 0.5

        # the seed face's closest point: no winner is farther away
        seed      = seed_faces[k]
        seed_u    = 0.5
        seed_v    = 0.5
        seed_dsq  = np.inf
        seed_dist = np.inf
        if seed >= 0:
            seed_u, seed_v, seed_dsq = _closest_point_on_bezier_face(
                control_points[seed], p[k], eps, max_its
            )
            seed_dist = seed_dsq**0.5

        for j in range(num_faces):
            # Bounding sphere culling (radiuses are squared)
            dist_to_centroid_sq = 0.0
            for i in range(3):
                diff = p[k, i] - centroids[j, i]
                dist_to_centroid_sq += diff * diff

            # past the seed's distance, with a millionth of the face to spare
            # for rounding, a face can neither win nor tie
            radius = radii[j]
            reach  = seed_dist + 1e-6 * radius
            upper  = min(best_dist, reach) + radius
            if dist_to_centroid_sq > upper * upper:
                continue

            if j == seed:
                u       = seed_u
                v       = seed_v
                dist_sq = seed_dsq
            else:
                u, v, dist_sq = _closest_point_on_bezier_face(
                    control_points[j], p[k], eps, max_its, min(best_dist_sq, reach * reach)
                )

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
        closest_dist[k]  = best_dist_sq
        closest_index[k] = best_index
        closest_uv[k, 0] = best_u
        closest_uv[k, 1] = best_v

        # Bilinear weights from UV (for vertex attribute interpolation)
        ou              = 1.0 - best_u
        ov              = 1.0 - best_v
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
    t1x = (aabb_min[0] - o[0]) * inv_d[0]
    t1y = (aabb_min[1] - o[1]) * inv_d[1]
    t1z = (aabb_min[2] - o[2]) * inv_d[2]

    t2x = (aabb_max[0] - o[0]) * inv_d[0]
    t2y = (aabb_max[1] - o[1]) * inv_d[1]
    t2z = (aabb_max[2] - o[2]) * inv_d[2]

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
def _ray_offset(p, o, d):
    """(p - o) x d: each component is p's side of a plane through the ray"""
    rx = p[0] - o[0]
    ry = p[1] - o[1]
    rz = p[2] - o[2]
    return ry * d[2] - rz * d[1], rz * d[0] - rx * d[2], rx * d[1] - ry * d[0]


@njit(fastmath=True, cache=True)
def _ray_misses_corners(p0, p1, p2, p3, o, d, margin):
    """
    True when the ray passes farther than margin from the hull of a
    bilinear face's corners, which holds the face: the corners all lie on
    one side of a plane through the ray, one of the 3 square to d x X,
    d x Y or d x Z. d is of unit length.
    """
    ax, ay, az = _ray_offset(p0, o, d)
    bx, by, bz = _ray_offset(p1, o, d)
    cx, cy, cz = _ray_offset(p2, o, d)
    ex, ey, ez = _ray_offset(p3, o, d)
    return (
        min(ax, bx, cx, ex) > margin
        or max(ax, bx, cx, ex) < -margin
        or min(ay, by, cy, ey) > margin
        or max(ay, by, cy, ey) < -margin
        or min(az, bz, cz, ez) > margin
        or max(az, bz, cz, ez) < -margin
    )


@njit(fastmath=True, cache=True)
def _ray_misses_control_points(cp, o, d, margin):
    """the same test as _ray_misses_corners, on a Bezier face's 16 control points"""
    lo_x, lo_y, lo_z = _ray_offset(cp[0, 0], o, d)
    hi_x = lo_x
    hi_y = lo_y
    hi_z = lo_z
    for i in range(4):
        for j in range(4):
            wx, wy, wz = _ray_offset(cp[i, j], o, d)
            lo_x = min(lo_x, wx)
            hi_x = max(hi_x, wx)
            lo_y = min(lo_y, wy)
            hi_y = max(hi_y, wy)
            lo_z = min(lo_z, wz)
            hi_z = max(hi_z, wz)
    return (
        lo_x > margin
        or hi_x < -margin
        or lo_y > margin
        or hi_y < -margin
        or lo_z > margin
        or hi_z < -margin
    )


@njit(fastmath=True, cache=True)
def _intersect_bilinear_newton(
    p0, p1, p2, p3, o, d, tol, u0, v0, eps=1e-6, max_its=20
):
    """Newton iteration to find ray-bilinear patch intersection.

    Both tests are in face terms, so they hold the same on a face of any
    size: it has converged once the point is closer to the ray than tol,
    eps times the face's size unless the rounding is coarser, and det is
    against the tangents' own lengths. The face is seen from the ray's
    origin, so its distance from the world's origin costs no precision.
    Once converged, one last step makes (u, v) and t as precise on any size.
    """
    u  = u0
    v  = v0

    Cx = p0[0] - p1[0] + p2[0] - p3[0]
    Cy = p0[1] - p1[1] + p2[1] - p3[1]
    Cz = p0[2] - p1[2] + p2[2] - p3[2]
    Dx = p1[0] - p0[0]
    Dy = p1[1] - p0[1]
    Dz = p1[2] - p0[2]
    Ex = p3[0] - p0[0]
    Ey = p3[1] - p0[1]
    Ez = p3[2] - p0[2]
    Fx = p0[0] - o[0]
    Fy = p0[1] - o[1]
    Fz = p0[2] - o[2]

    dx = d[0]
    dy = d[1]
    dz = d[2]

    eps_sq = tol * tol

    # once converged, one last step doubles the digits. It is taken when it
    # is short enough that the face's bend keeps it within eps of the ray,
    # so a point off the face by more than twice that length stays off it
    last_sq = eps / 32.0
    reach   = 2.0 * np.sqrt(last_sq)

    for _ in range(max_its):
        # the point, from the ray's origin
        diffx = Cx * u * v + Dx * u + Ex * v + Fx
        diffy = Cy * u * v + Dy * u + Ey * v + Fy
        diffz = Cz * u * v + Dz * u + Ez * v + Fz

        dux   = Cx * v + Dx
        duy   = Cy * v + Dy
        duz   = Cz * v + Dz
        dvx   = Cx * u + Ex
        dvy   = Cy * u + Ey
        dvz   = Cz * u + Ez

        fx = diffy * dz - diffz * dy
        fy = diffz * dx - diffx * dz
        fz = diffx * dy - diffy * dx

        f_norm_sq = fx * fx + fy * fy + fz * fz
        converged = f_norm_sq < eps_sq
        if converged and (u < -reach or u > 1 + reach or v < -reach or v > 1 + reach):
            return diffx * dx + diffy * dy + diffz * dz, u, v, True

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
        if abs(det) <= 1e-12 * JTJ00 * JTJ11:
            if converged:
                return diffx * dx + diffy * dy + diffz * dz, u, v, True
            return 1e308, -1.0, -1.0, False

        inv_det = 1.0 / det
        delta_u = (JTJ11 * JTf0 - JTJ01 * JTf1) * inv_det
        delta_v = (JTJ00 * JTf1 - JTJ01 * JTf0) * inv_det

        # the last step; t moves with it, exactly on a bilinear face
        if converged:
            t = diffx * dx + diffy * dy + diffz * dz
            if delta_u * delta_u + delta_v * delta_v <= last_sq:
                t -= (dux * delta_u + dvx * delta_v - Cx * delta_u * delta_v) * dx
                t -= (duy * delta_u + dvy * delta_v - Cy * delta_u * delta_v) * dy
                t -= (duz * delta_u + dvz * delta_v - Cz * delta_u * delta_v) * dz
                u -= delta_u
                v -= delta_v
            return t, u, v, True

        u = u - delta_u
        v = v - delta_v

    return 1e308, -1.0, -1.0, False


@njit(fastmath=True, cache=True)
def _intersect_face(p0, p1, p2, p3, o, d, twosided=True, eps=1e-6, max_its=20):
    """Find forward intersection of ray with a bilinear face (t >= 0 only).

    When twosided=False, back-face hits (where the geometric surface normal
    faces the same direction as the ray) are rejected.

    Uses small tolerances on (u, v) and t boundaries to avoid rejecting
    valid intersections that land exactly on a face edge or at the ray origin
    due to floating-point precision. Like eps, the t and planarity
    tolerances are in face terms. A ray that passes clear of the hull of
    the face's corners is not solved for.
    """
    best_t = 1e308
    best_u = -1.0
    best_v = -1.0

    # a face with no size is no surface to hit
    size = _face_size(p0, p1, p2, p3)
    if size == 0.0:
        return best_t, best_u, best_v

    # Newton sees the face from the ray's origin: the farther the face is
    # from it, for its size, the coarser the rounding of a point's offset
    # from the ray. No test asks for less than that rounding
    from_origin = max(abs(p0[0] - o[0]), abs(p0[1] - o[1]), abs(p0[2] - o[2]))
    rounding    = 1e-14 * (size + from_origin)
    tol         = max(eps * size, rounding)
    uv_tol      = max(1e-6, rounding / size)
    t_tol       = 1e-10 * size

    # a hit lies within 2 * tol of the ray and within uv_tol of the face,
    # which is at most 7 * uv_tol * size out of its corners' hull
    if _ray_misses_corners(p0, p1, p2, p3, o, d, 2.0 * tol + 20.0 * uv_tol * size):
        return best_t, best_u, best_v

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
        p0, p1, p2, p3, o, d, tol, 0.5, 0.5, eps, max_its
    )
    if (
        converged
        and -uv_tol <= u <= 1 + uv_tol
        and -uv_tol <= v <= 1 + uv_tol
        and t >= -t_tol
    ):
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        t = max(0.0, t)

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

            if Cx * Cx + Cy * Cy + Cz * Cz < 1e-10 * size * size:
                return best_t, best_u, best_v

    for su, sv in ((0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)):
        t, u, v, converged = _intersect_bilinear_newton(
            p0, p1, p2, p3, o, d, tol, su, sv, eps, max_its
        )
        if (
            converged
            and -uv_tol <= u <= 1 + uv_tol
            and -uv_tol <= v <= 1 + uv_tol
            and t >= -t_tol
        ):
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            t = max(0.0, t)

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

    inv_d = np.empty(3)
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
        t[i]            = ti
        uv[i, 0]        = ui
        uv[i, 1]        = vi
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

    inv_d = np.empty(3)
    for k in range(3):
        if abs(d[k]) < 1e-30:
            inv_d[k] = np.inf if d[k] >= 0 else -np.inf
        else:
            inv_d[k] = 1.0 / d[k]

    # Iterative DFS using a fixed-size stack of node ids.
    stack     = np.empty(_BVH_STACK_SIZE, dtype=np.int32)
    sp        = 0
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
        t[i]            = ti
        uv[i, 0]        = ui
        uv[i, 1]        = vi
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

    inv_d = np.empty(3)
    for k in range(3):
        if abs(d[k]) < 1e-30:
            inv_d[k] = np.inf if d[k] >= 0 else -np.inf
        else:
            inv_d[k] = 1.0 / d[k]

    stack     = np.empty(_BVH_STACK_SIZE, dtype=np.int32)
    sp        = 0
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
        t[i]            = ti
        uv[i, 0]        = ui
        uv[i, 1]        = vi
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
def _intersect_bezier_newton(cp, d, tol, u0, v0, eps=1e-6, max_its=20):
    """Newton iteration to find ray-bicubic Bezier patch intersection.

    Same formulation as _intersect_bilinear_newton, its tests in face
    terms and its last step past the tolerance too, but evaluates the
    surface and derivatives via _eval_bezier_surface. cp is seen from the
    ray's origin, so the face's distance from the world's origin costs no
    precision.
    """
    u  = u0
    v  = v0

    dx = d[0]
    dy = d[1]
    dz = d[2]

    eps_sq = tol * tol

    # once converged, one last step doubles the digits. It is taken when it
    # is short enough that the face's bend keeps it within eps of the ray,
    # so a point off the face by more than twice that length stays off it
    last_sq = eps / 32.0
    reach   = 2.0 * np.sqrt(last_sq)

    for _ in range(max_its):
        # the point, from the ray's origin
        diffx, diffy, diffz, dux, duy, duz, dvx, dvy, dvz = _eval_bezier_surface(
            cp, u, v
        )

        fx = diffy * dz - diffz * dy
        fy = diffz * dx - diffx * dz
        fz = diffx * dy - diffy * dx

        f_norm_sq = fx * fx + fy * fy + fz * fz
        converged = f_norm_sq < eps_sq
        if converged and (u < -reach or u > 1 + reach or v < -reach or v > 1 + reach):
            return diffx * dx + diffy * dy + diffz * dz, u, v, True

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
        if abs(det) <= 1e-12 * JTJ00 * JTJ11:
            if converged:
                return diffx * dx + diffy * dy + diffz * dz, u, v, True
            return 1e308, -1.0, -1.0, False

        inv_det = 1.0 / det
        delta_u = (JTJ11 * JTf0 - JTJ01 * JTf1) * inv_det
        delta_v = (JTJ00 * JTf1 - JTJ01 * JTf0) * inv_det

        # the last step; t moves with it, to first order
        if converged:
            t = diffx * dx + diffy * dy + diffz * dz
            if delta_u * delta_u + delta_v * delta_v <= last_sq:
                t -= (dux * delta_u + dvx * delta_v) * dx
                t -= (duy * delta_u + dvy * delta_v) * dy
                t -= (duz * delta_u + dvz * delta_v) * dz
                u -= delta_u
                v -= delta_v
            return t, u, v, True

        u = u - delta_u
        v = v - delta_v

    return 1e308, -1.0, -1.0, False


@njit(fastmath=True, cache=True)
def _intersect_bezier_face(cp, o, d, twosided=True, eps=1e-6, max_its=20):
    """Find forward intersection of ray with a bicubic Bezier face.

    Uses a 3x3 grid of initial guesses to handle multiple roots.
    When twosided=False, back-face hits are rejected.

    Uses small tolerances on (u, v) and t boundaries to avoid rejecting
    valid intersections that land exactly on a face edge or at the ray origin
    due to floating-point precision. Like eps, the t tolerance is in face
    terms, and the tests never ask for less than the rounding, as for a
    bilinear face. A ray that passes clear of the hull of the face's control
    points is not solved for.
    """
    best_t = 1e308
    best_u = -1.0
    best_v = -1.0

    # the face's size: the largest coordinate span of its control points
    size = 0.0
    for k in range(3):
        lo = cp[0, 0, k]
        hi = cp[0, 0, k]
        for i in range(4):
            for j in range(4):
                lo = min(lo, cp[i, j, k])
                hi = max(hi, cp[i, j, k])
        size = max(size, hi - lo)

    # a face with no size is no surface to hit
    if size == 0.0:
        return best_t, best_u, best_v

    # the rounding, as for a bilinear face
    from_origin = max(
        abs(cp[0, 0, 0] - o[0]), abs(cp[0, 0, 1] - o[1]), abs(cp[0, 0, 2] - o[2])
    )
    rounding = 1e-14 * (size + from_origin)
    tol      = max(eps * size, rounding)
    uv_tol   = max(1e-6, rounding / size)
    t_tol    = 1e-10 * size

    # a hit lies within 2 * tol of the ray and within uv_tol of the face,
    # which is at most 11 * uv_tol * size out of the hull
    if _ray_misses_control_points(cp, o, d, 2.0 * tol + 20.0 * uv_tol * size):
        return best_t, best_u, best_v

    # the control points, from the ray's origin
    local = np.empty((4, 4, 3))
    for i in range(4):
        for j in range(4):
            for k in range(3):
                local[i, j, k] = cp[i, j, k] - o[k]

    for su in (0.5, 0.25, 0.75):
        for sv in (0.5, 0.25, 0.75):
            t, u, v, converged = _intersect_bezier_newton(
                local, d, tol, su, sv, eps, max_its
            )
            if (
                converged
                and -uv_tol <= u <= 1 + uv_tol
                and -uv_tol <= v <= 1 + uv_tol
                and t >= -t_tol
            ):
                u = max(0.0, min(1.0, u))
                v = max(0.0, min(1.0, v))
                t = max(0.0, t)

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

    inv_d = np.empty(3)
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
        t[i]            = ti
        uv[i, 0]        = ui
        uv[i, 1]        = vi
        face_indices[i] = fi

    return t, uv, face_indices