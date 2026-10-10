from math import cos, sin

import numpy as np
from numba import njit, prange

EPSILON     = np.finfo(np.float32).eps
EULER_SAFE  = np.array([0, 1, 2, 0],          dtype=np.intp)
EULER_NEXT  = np.array([1, 2, 0, 1],          dtype=np.intp)
EULER_ORDER = np.array([0, 8, 16, 4, 12, 20], dtype=np.intp)
MAYA_EA = np.array(
    [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.intp
)  # maya to euler angle
EA_MAYA = np.array(
    [[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.intp
)  # euler angle to maya


@njit(cache=True)
def _get_euler_order(axis):
    o_ = EULER_ORDER[axis]
    f  = o_ & 1
    o_ >>= 1
    s = o_ & 1
    o_ >>= 1
    n = o_ & 1
    o_ >>= 1
    i = EULER_SAFE[o_ & 3]
    j = EULER_NEXT[i + n]
    k = EULER_NEXT[i + 1 - n]
    h = i
    if s:
        h = k

    return i, j, k, h, n, s, f


@njit(parallel=True, cache=True)
def _euler_to_matrix(euler, axes):
    """Converts Maya euler angles to 4x4 matrices"""
    matrix = np.empty((euler.shape[0], 4, 4), dtype=euler.dtype)

    for ii in prange(euler.shape[0]):
        ea_ = np.empty(3, dtype=euler.dtype)

        ea_[0], ea_[1], ea_[2] = euler[ii, 0], euler[ii, 1], euler[ii, 2]
        ea_[0], ea_[1], ea_[2] = (
            ea_[MAYA_EA[axes[ii], 0]],
            ea_[MAYA_EA[axes[ii], 1]],
            ea_[MAYA_EA[axes[ii], 2]],
        )

        i, j, k, h, n, s, f = _get_euler_order(axes[ii])

        if f == 1:
            ea_[0], ea_[2] = ea_[2], ea_[0]
        if n == 1:
            ea_[0], ea_[1], ea_[2] = -ea_[0], -ea_[1], -ea_[2]

        ci = cos(ea_[0])
        cj = cos(ea_[1])
        ch = cos(ea_[2])
        si = sin(ea_[0])
        sj = sin(ea_[1])
        sh = sin(ea_[2])
        cc = ci * ch
        cs = ci * sh
        sc = si * ch
        ss = si * sh

        if s:
            matrix[ii, i, i] = cj
            matrix[ii, j, i] = sj * si
            matrix[ii, k, i] = sj * ci
            matrix[ii, i, j] = sj * sh
            matrix[ii, j, j] = -cj * ss + cc
            matrix[ii, k, j] = -cj * cs - sc
            matrix[ii, i, k] = -sj * ch
            matrix[ii, j, k] = cj * sc + cs
            matrix[ii, k, k] = cj * cc - ss

        else:
            matrix[ii, i, i] = cj * ch
            matrix[ii, j, i] = sj * sc - cs
            matrix[ii, k, i] = sj * cc + ss
            matrix[ii, i, j] = cj * sh
            matrix[ii, j, j] = sj * ss + cc
            matrix[ii, k, j] = sj * cs - sc
            matrix[ii, i, k] = -sj
            matrix[ii, j, k] = cj * si
            matrix[ii, k, k] = cj * ci

        matrix[ii, 0, 3] = 0.0
        matrix[ii, 1, 3] = 0.0
        matrix[ii, 2, 3] = 0.0
        matrix[ii, 3, 0] = 0.0
        matrix[ii, 3, 1] = 0.0
        matrix[ii, 3, 2] = 0.0
        matrix[ii, 3, 3] = 1.0

    return matrix


@njit(parallel=True, cache=True)
def _euler_to_quaternion(euler, axes):
    """Converts list of Maya euler angles to quaternions qi,qj,qk,qw"""
    quat = np.empty((euler.shape[0], 4), dtype=euler.dtype)

    for ii in prange(euler.shape[0]):
        ea_ = np.empty(3, dtype=euler.dtype)

        ea_[0], ea_[1], ea_[2] = euler[ii, 0], euler[ii, 1], euler[ii, 2]
        ea_[0], ea_[1], ea_[2] = (
            ea_[MAYA_EA[axes[ii], 0]],
            ea_[MAYA_EA[axes[ii], 1]],
            ea_[MAYA_EA[axes[ii], 2]],
        )

        i, j, k, h, n, s, f = _get_euler_order(axes[ii])

        if f == 1:
            ea_[0], ea_[2] = ea_[2], ea_[0]
        if n == 1:
            ea_[1] = -ea_[1]

        ti = ea_[0] * 0.5
        tj = ea_[1] * 0.5
        th = ea_[2] * 0.5
        ci = cos(ti)
        cj = cos(tj)
        ch = cos(th)
        si = sin(ti)
        sj = sin(tj)
        sh = sin(th)
        cc = ci * ch
        cs = ci * sh
        sc = si * ch
        ss = si * sh

        # the imaginary components belong in the order-dependent slots i, j, k
        # -- they coincide with 0, 1, 2 only for xyz
        if s == 1:
            quat[ii, i] = cj * (cs + sc)
            quat[ii, j] = sj * (cc + ss)
            quat[ii, k] = sj * (cs - sc)
            quat[ii, 3] = cj * (cc - ss)

        else:
            quat[ii, i] = cj * sc - sj * cs
            quat[ii, j] = cj * ss + sj * cc
            quat[ii, k] = cj * cs - sj * sc
            quat[ii, 3] = cj * cc + sj * ss

        if n == 1:
            quat[ii, j] = -quat[ii, j]

    return quat


@njit(parallel=True, cache=True)
def _euler_filter(euler, middle):
    """Filters (frames, curves, 3) euler angles along the frames, in place.

    The curves are shared out between threads; within a curve the frames run
    in order, since each one depends on the frame before it, filtered. A
    frame is replaced by whichever of its two representations, slid onto
    the turn nearest the frame before, moves the least. The other
    representation turns every axis by half a turn and reflects the
    ``middle`` one of the curve's rotate order. The cost sums the axes, left
    to right, as maya's filterCurve does; on a tie the frame keeps its own
    representation.

    A frame holding a nan or an inf on any axis is a gap: it is left as it
    is, and the next finite frame is filtered against the last finite one,
    so one bad key stays on its own frame instead of turning every later
    key of its curve into nan. The first finite frame is kept as given.

    No fastmath: on finite frames every step is the arithmetic numpy did
    when this was a loop over frames, so the result is bit for bit the same.
    """
    turn = 2.0 * np.pi
    for c in prange(euler.shape[1]):
        reflect = middle[c]
        kept    = np.empty(3, dtype=euler.dtype)
        flipped = np.empty(3, dtype=euler.dtype)
        last    = -1  # the last finite frame, filtered
        for f in range(euler.shape[0]):
            finite = True
            for axis in range(3):
                if not np.isfinite(euler[f, c, axis]):
                    finite = False
            if not finite:
                continue
            if last < 0:
                last = f
                continue

            kept_cost    = 0.0
            flipped_cost = 0.0
            for axis in range(3):
                previous = euler[last, c, axis]
                current  = euler[f, c, axis]
                if axis == reflect:
                    other = np.pi - current
                else:
                    other = current + np.pi

                kept[axis]    = current + turn * np.rint((previous - current) / turn)
                flipped[axis] = other + turn * np.rint((previous - other) / turn)
                kept_cost    += abs(kept[axis] - previous)
                flipped_cost += abs(flipped[axis] - previous)

            # numpy's argmin: the first minimum wins, and a nan counts as one
            if flipped_cost < kept_cost or (
                np.isnan(flipped_cost) and not np.isnan(kept_cost)
            ):
                for axis in range(3):
                    euler[f, c, axis] = flipped[axis]
            else:
                for axis in range(3):
                    euler[f, c, axis] = kept[axis]
            last = f
