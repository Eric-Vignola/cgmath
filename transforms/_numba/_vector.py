from math import acos, atan2, cos, sin, sqrt

import numpy as np
from numba import njit, prange

ROUNDING = 2.0**-50
"""A perpendicular this short is rounding: the two directions it was taken
from are opposite to within a few ulps, and give it no direction."""

SQUARES_LOW = 2.0**-968
"""A sum of squares from this up kept every bit: below it, the squares of
the smaller components can have lost bits as subnormals."""

SQUARES_HIGH = np.finfo(np.float64).max
"""A sum of squares above this overflowed, or had an inf or a nan in it."""


@njit(cache=True, error_model="numpy")
def _unit3(x, y, z):
    """The length of ``(x, y, z)`` and the three scaled to unit length.

    A vector under about 1e-146 or over about 1e154 long is first divided
    by its largest component, so that the squares neither lose bits nor
    overflow. A zero vector has a length of 0 and comes back as it is; a
    nan or an inf in it makes all four nan.
    """
    squares = x * x + y * y + z * z
    if not (SQUARES_LOW <= squares <= SQUARES_HIGH):
        m = max(max(abs(x), abs(y)), abs(z))
        if m == 0.0 and squares == 0.0:
            return 0.0, x, y, z

        x       = x / m
        y       = y / m
        z       = z / m
        squares = x * x + y * y + z * z
        n       = sqrt(squares)
        r       = 1.0 / n
        return m * n, x * r, y * r, z * r

    n = sqrt(squares)
    r = 1.0 / n
    return n, x * r, y * r, z * r


@njit(parallel=True, cache=True)
def _vector_arc(vector0, vector1):
    angle = np.empty(vector0.shape[0], dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        # a zero vector stays zero, a right angle from anything. scratch
        # arrays from np.empty handed it whatever vector was last normalised
        # in them, a different one on one thread and on many
        X0, Y0, Z0 = 0.0, 0.0, 0.0
        X1, Y1, Z1 = 0.0, 0.0, 0.0

        m = (vector0[i, 0] ** 2 + vector0[i, 1] ** 2 + vector0[i, 2] ** 2) ** 0.5
        if m > 0.0:
            X0 = vector0[i, 0] / m
            Y0 = vector0[i, 1] / m
            Z0 = vector0[i, 2] / m

        m = (vector1[i, 0] ** 2 + vector1[i, 1] ** 2 + vector1[i, 2] ** 2) ** 0.5
        if m > 0.0:
            X1 = vector1[i, 0] / m
            Y1 = vector1[i, 1] / m
            Z1 = vector1[i, 2] / m

        dot = (X0 * X1) + (Y0 * Y1) + (Z0 * Z1)
        if dot > 1.0:
            dot = 1.0
        elif dot < -1.0:
            dot = -1.0

        angle[i] = acos(dot)

    return angle


@njit(parallel=True, cache=True)
def _vector_slerp(vector0, vector1, weight):
    """Spherical interpolation between two lists of vectors.

    ``(vector0 * sin((1 - weight) * angle) + vector1 * sin(weight * angle))
    / sin(angle)``, ``angle`` the one between the two, on the vectors as
    given: whatever their lengths, the direction goes from ``vector0``'s to
    ``vector1``'s along the great arc between them, and between two of one
    length it turns at a steady rate and keeps the length.

    The arc is found as in ``_quaternion_slerp``, from the difference and
    the sum of the two directions, which keep every digit of nearly
    parallel and nearly opposite vectors, and ``sin(angle)`` with them. The
    sum is taken from the shorter vector: its length along ``vector0``'s
    direction turned by ``weight * angle``, plus what is left of the
    longer one's term, ``(|vector1| - |vector0|) * sin(weight * angle) /
    sin(angle)`` along ``vector1``'s direction, or the same of ``vector0``
    with ``1 - weight``. That is the formula's vector, with nothing in it
    to cancel: not its two terms as the vectors near opposite, nor the
    longer length against itself.

    A weight of 0 or 1 returns ``vector0`` or ``vector1`` exactly. Parallel
    vectors lerp, the formula's limit. Vectors of different lengths
    nearly opposite grow as ``1 / (pi - angle)``, as the formula does.
    Vectors opposite to within rounding have no plane to turn in, and the
    formula no answer: the direction turns toward the world axis along
    ``vector0``'s smallest component, the first of a tie, and the length
    goes linearly from one's to the other's -- between two of one length,
    the formula's limit. A zero vector has no direction: the result is
    ``vector0 + weight * (vector1 - vector0)``. Between the ends, a nan or
    an inf component gives nans.
    """
    v = np.empty((vector0.shape[0], 3), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        t = weight[i]

        m0, X0, Y0, Z0 = _unit3(vector0[i, 0], vector0[i, 1], vector0[i, 2])
        m1, X1, Y1, Z1 = _unit3(vector1[i, 0], vector1[i, 1], vector1[i, 2])

        if t == 0.0:
            v[i, 0] = vector0[i, 0]
            v[i, 1] = vector0[i, 1]
            v[i, 2] = vector0[i, 2]

        elif t == 1.0:
            v[i, 0] = vector1[i, 0]
            v[i, 1] = vector1[i, 1]
            v[i, 2] = vector1[i, 2]

        elif m0 == 0.0 or m1 == 0.0:
            v[i, 0] = vector0[i, 0] + t * (vector1[i, 0] - vector0[i, 0])
            v[i, 1] = vector0[i, 1] + t * (vector1[i, 1] - vector0[i, 1])
            v[i, 2] = vector0[i, 2] + t * (vector1[i, 2] - vector0[i, 2])

        else:
            # the angle between the two from their difference and their sum
            dx    = X1 - X0
            dy    = Y1 - Y0
            dz    = Z1 - Z0
            sx    = X1 + X0
            sy    = Y1 + Y0
            sz    = Z1 + Z0
            dn    = sqrt(dx * dx + dy * dy + dz * dz)
            sn    = sqrt(sx * sx + sy * sy + sz * sz)
            angle = 2.0 * atan2(dn, sn)

            # the second direction's part perpendicular to the first, from
            # the shorter of the two
            near = dn <= sn
            if near:
                dot = X0 * dx + Y0 * dy + Z0 * dz
                px  = dx - dot * X0
                py  = dy - dot * Y0
                pz  = dz - dot * Z0
            else:
                dot = X0 * sx + Y0 * sy + Z0 * sz
                px  = sx - dot * X0
                py  = sy - dot * Y0
                pz  = sz - dot * Z0
            pn = sqrt(px * px + py * py + pz * pz)

            opposite = not near and pn <= ROUNDING
            if opposite:
                # the world axis along the smallest component, less its
                # part along the first direction
                ex, ey, ez = 0.0, 0.0, 0.0
                if abs(X0) <= abs(Y0) and abs(X0) <= abs(Z0):
                    ex = 1.0
                elif abs(Y0) <= abs(Z0):
                    ey = 1.0
                else:
                    ez = 1.0
                dot = X0 * ex + Y0 * ey + Z0 * ez
                px  = ex - dot * X0
                py  = ey - dot * Y0
                pz  = ez - dot * Z0
                pn  = sqrt(px * px + py * py + pz * pz)

            if pn > 0.0:
                a = cos(t * angle)
                b = sin(t * angle) / pn
            else:
                a = 1.0
                b = 0.0

            # the first direction turned by t * angle: unit to an ulp
            # already, the normalisation is a guard
            x = a * X0 + b * px
            y = a * Y0 + b * py
            z = a * Z0 + b * pz
            r = 1.0 / sqrt(x * x + y * y + z * z)

            if opposite:
                # no plane, and no answer from the formula: the length
                # lerps
                r       = r * (m0 + t * (m1 - m0))
                v[i, 0] = x * r
                v[i, 1] = y * r
                v[i, 2] = z * r

            else:
                # sin(angle), from the difference and the sum: 0 between
                # parallel vectors, where the formula's limit is the lerp
                sine = 2.0 * dn * sn / (dn * dn + sn * sn)

                # the formula from the shorter of the two: its length
                # along the turned direction, plus the rest of the
                # other's term, with nothing to cancel
                if m0 <= m1:
                    share = t
                    if sine > 0.0:
                        share = sin(t * angle) / sine
                    share   = share * (m1 - m0)
                    r       = r * m0
                    v[i, 0] = x * r + share * X1
                    v[i, 1] = y * r + share * Y1
                    v[i, 2] = z * r + share * Z1
                else:
                    share = 1.0 - t
                    if sine > 0.0:
                        share = sin((1.0 - t) * angle) / sine
                    share   = share * (m0 - m1)
                    r       = r * m1
                    v[i, 0] = x * r + share * X0
                    v[i, 1] = y * r + share * Y0
                    v[i, 2] = z * r + share * Z0

    return v


@njit(parallel=True, cache=True)
def _vector_lerp(vector0, vector1, weight):
    v = np.empty((vector0.shape[0], vector0.shape[1]), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        for j in range(vector0.shape[1]):
            v[i, j] = vector0[i, j] + weight[i] * (vector1[i, j] - vector0[i, j])

    return v


# NOTE: no explicit signature. Levelled inputs may arrive as read-only
# broadcast views, which an eager ``float64[:,:]`` signature rejects; lazy
# compilation specialises per layout instead, as every other kernel here does.
@njit(parallel=True, cache=True)
def _vector_cross(vector0, vector1):
    vector = np.zeros((vector0.shape[0], 3), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        vector[i, 0] = vector0[i, 1] * vector1[i, 2] - vector0[i, 2] * vector1[i, 1]
        vector[i, 1] = vector0[i, 2] * vector1[i, 0] - vector0[i, 0] * vector1[i, 2]
        vector[i, 2] = vector0[i, 0] * vector1[i, 1] - vector0[i, 1] * vector1[i, 0]

    return vector


@njit(parallel=True, cache=True)
def _vector_dot(vector0, vector1):
    dot = np.empty(vector0.shape[0], dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        dot[i] = 0
        for j in range(vector0.shape[1]):
            dot[i] += vector0[i, j] * vector1[i, j]

    return dot


@njit(parallel=True, cache=True)
def _vector_magnitude(vector):
    mag = np.empty(vector.shape[0], dtype=vector.dtype)
    for i in prange(vector.shape[0]):
        mag[i] = 0.0
        for j in range(vector.shape[1]):
            mag[i] = mag[i] + vector[i, j] ** 2

        mag[i] = mag[i] ** 0.5

    return mag


@njit(parallel=True, cache=True)
def _vector_normalize(vector):
    vector_ = np.empty(vector.shape, dtype=vector.dtype)

    for i in prange(vector.shape[0]):
        mag = 0
        for j in range(vector.shape[1]):
            mag += vector[i, j] ** 2

        mag = mag**0.5

        for j in range(vector.shape[1]):
            vector_[i, j] = vector[i, j] / mag

    return vector_


@njit(parallel=True, cache=True)
def _vector_to_matrix(vector0, vector1, aim_axis, up_axis):
    matrix = np.empty((vector0.shape[0], 4, 4), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        ii   = aim_axis[i]
        jj   = up_axis[i]
        kk   = (min(ii, jj) - max(ii, jj) + min(ii, jj)) % 3

        flip = 0
        if ii == 0 and jj == 2:
            flip = 1
        elif ii == 1 and jj == 0:
            flip = 1
        elif ii == 2 and jj == 1:
            flip = 1

        # the vectors in scalars: a scratch array made here is allocated
        # once an element on one thread, where the threaded loop hoists it
        X0, Y0, Z0 = vector0[i, 0], vector0[i, 1], vector0[i, 2]
        X1, Y1, Z1 = vector1[i, 0], vector1[i, 1], vector1[i, 2]

        # init matrix output
        matrix[i, 0, 0] = 1.0
        matrix[i, 0, 1] = 0.0
        matrix[i, 0, 2] = 0.0
        matrix[i, 0, 3] = 0.0

        matrix[i, 1, 0] = 0.0
        matrix[i, 1, 1] = 1.0
        matrix[i, 1, 2] = 0.0
        matrix[i, 1, 3] = 0.0

        matrix[i, 2, 0] = 0.0
        matrix[i, 2, 1] = 0.0
        matrix[i, 2, 2] = 1.0
        matrix[i, 2, 3] = 0.0

        matrix[i, 3, 0] = 0.0
        matrix[i, 3, 1] = 0.0
        matrix[i, 3, 2] = 0.0
        matrix[i, 3, 3] = 1.0

        x  = Y0 * Z1 - Z0 * Y1
        y  = Z0 * X1 - X0 * Z1
        z  = X0 * Y1 - Y0 * X1

        na = (X0**2 + Y0**2 + Z0**2) ** 0.5
        nc = (x**2 + y**2 + z**2) ** 0.5

        if na > 0.0 and nc > 0.0:
            matrix[i, kk, 0] = x / nc
            matrix[i, ii, 0] = X0 / na
            matrix[i, kk, 1] = y / nc
            matrix[i, ii, 1] = Y0 / na
            matrix[i, kk, 2] = z / nc
            matrix[i, ii, 2] = Z0 / na

            matrix[i, jj, 0] = (
                matrix[i, kk, 1] * matrix[i, ii, 2]
                - matrix[i, kk, 2] * matrix[i, ii, 1]
            )
            matrix[i, jj, 1] = (
                matrix[i, kk, 2] * matrix[i, ii, 0]
                - matrix[i, kk, 0] * matrix[i, ii, 2]
            )
            matrix[i, jj, 2] = (
                matrix[i, kk, 0] * matrix[i, ii, 1]
                - matrix[i, kk, 1] * matrix[i, ii, 0]
            )

            if flip:
                matrix[i, kk, 0] = 0 - matrix[i, kk, 0]
                matrix[i, kk, 1] = 0 - matrix[i, kk, 1]
                matrix[i, kk, 2] = 0 - matrix[i, kk, 2]

    return matrix


@njit(parallel=True, cache=True)
def _vector_arc_to_quaternion(vector0, vector1):
    """Computes a list of quaternions qi,qj,qk,qw representing the arc between lists of vectors"""

    quat = np.empty((vector0.shape[0], 4), dtype=vector0.dtype)

    for i in prange(vector0.shape[0]):
        # the unit vectors and their half vector in scalars, as in
        # _vector_arc: a scratch array made here is allocated once an
        # element on one thread, where the threaded loop hoists it
        mag = (vector0[i, 0] ** 2 + vector0[i, 1] ** 2 + vector0[i, 2] ** 2) ** 0.5
        X0  = vector0[i, 0] / mag
        Y0  = vector0[i, 1] / mag
        Z0  = vector0[i, 2] / mag

        mag = (vector1[i, 0] ** 2 + vector1[i, 1] ** 2 + vector1[i, 2] ** 2) ** 0.5
        X1  = vector1[i, 0] / mag
        Y1  = vector1[i, 1] / mag
        Z1  = vector1[i, 2] / mag

        HX  = X0 + X1
        HY  = Y0 + Y1
        HZ  = Z0 + Z1

        mag = (HX**2 + HY**2 + HZ**2) ** 0.5

        HX  = HX / mag
        HY  = HY / mag
        HZ  = HZ / mag

        quat[i, 0] = Y0 * HZ - Z0 * HY
        quat[i, 1] = Z0 * HX - X0 * HZ
        quat[i, 2] = X0 * HY - Y0 * HX
        quat[i, 3] = X0 * HX + Y0 * HY + Z0 * HZ

    return quat