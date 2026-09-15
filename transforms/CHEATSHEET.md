# `cgmath.transforms` Cheatsheet

Copy-paste recipes for the functional transform math. Every block below is
self-contained: it carries its own imports and its own data, so you can jump
straight to the section you need and run it. For the concepts see
[`README.md`](README.md).

- [Conventions, in one worked example](#conventions-in-one-worked-example)
- [Constants](#constants)
- [Matrix](#matrix)
- [Quaternion](#quaternion)
- [Quaternion calculus and splines](#quaternion-calculus-and-splines)
- [Euler](#euler)
- [Axis / angle](#axis--angle)
- [Vector](#vector)
- [Broadcasting](#broadcasting)
- [Gotchas](#gotchas)

---

## Conventions, in one worked example

Row-major, row-vector, radians, `(i, j, k, w)`. Everything else follows.

```python
import numpy as np
from cgmath.transforms import axis_angle_to_quaternion, euler_to_matrix, matrix_point_multiply

np.set_printoptions(precision=4, suppress=True)      # so cos(90 deg) reads as 0, not 6e-17

M = euler_to_matrix([0.0, 0.0, np.radians(90)], 0)   # +90 deg about Z, XYZ order
print(M[0])

# rows 0..2 are the node's local X / Y / Z axes expressed in world space
assert np.allclose(M[0, 0, :3], [0, 1, 0])           # local +X now points along world +Y
assert np.allclose(M[0, 1, :3], [-1, 0, 0])          # local +Y now points along world -X

# row 3 is the translation -- NOT column 3
M[0, 3, :3] = [10.0, 0.0, 0.0]

# points are ROW vectors: p' = p * M
assert np.allclose(matrix_point_multiply([1.0, 0.0, 0.0], M), [[10.0, 1.0, 0.0]])

# quaternions are (i, j, k, w) -- the scalar is LAST
q = axis_angle_to_quaternion([0.0, 0.0, 1.0], np.radians(90))
print(q)                                             # [[0, 0, 0.7071, 0.7071]]
assert np.isclose(q[0, 3], np.cos(np.radians(45)))   # index 3 is w
```

Angles are radians everywhere in this module.

---

## Constants

Maya's rotate-order indices and XYZ axis indices, importable from the package
root.

| Rotate order | Value | | Axis | Value |
|---|---|---|---|---|
| `XYZ` | 0 | | `X` | 0 |
| `YZX` | 1 | | `Y` | 1 |
| `ZXY` | 2 | | `Z` | 2 |
| `XZY` | 3 | | | |
| `YXZ` | 4 | | | |
| `ZYX` | 5 | | | |

```python
from cgmath.transforms import X, XYZ, XZY, Y, YXZ, YZX, Z, ZXY, ZYX

assert (XYZ, YZX, ZXY, XZY, YXZ, ZYX) == (0, 1, 2, 3, 4, 5)
assert (X, Y, Z) == (0, 1, 2)
```

Rotate order is a plain `int`, so a bare `0` works anywhere `axes=` is taken —
and a **list** of orders works too, one per row.

```python
import numpy as np
from cgmath.transforms import XYZ, XZY, YXZ, YZX, ZXY, ZYX, euler_to_matrix, matrix_to_euler

one = euler_to_matrix([np.radians(10), np.radians(20), np.radians(30)], XYZ)
print(np.degrees(matrix_to_euler(one, [XYZ, YZX, ZXY, XZY, YXZ, ZYX])))   # (6, 3)
```

---

## Matrix

### Build

```python
from cgmath.transforms import matrix_identity, matrix_random

I  = matrix_identity(3)                              # (3, 4, 4)
R  = matrix_random(5, seed=7)                        # rotation only
RT = matrix_random(5, seed=7, random_position=True)  # + translation in [-1, 1]
print(I.shape, R.shape, RT.shape)
```

### Convert

```python
import numpy as np
from cgmath.transforms import XYZ, euler_to_matrix, matrix_random, matrix_to_euler, matrix_to_quaternion

mats = matrix_random(4, seed=12345)  # (4, 4, 4) rotation only

ea   = matrix_to_euler(mats, XYZ)    # (4, 3) radians
q    = matrix_to_quaternion(mats)    # (4, 4) (i, j, k, w)
assert np.allclose(euler_to_matrix(ea, XYZ), mats)
```

### Compose, invert, transpose

`matrix_multiply(A, B)` is `A @ B`. Under the row-vector convention that means
**A is applied first**, then B.

```python
import numpy as np
from cgmath.transforms import (
    XYZ,
    euler_to_matrix,
    matrix_identity,
    matrix_inverse,
    matrix_multiply,
    matrix_point_multiply,
    matrix_random,
    matrix_transpose,
)

mats = matrix_random(4, seed=12345)

T           = matrix_identity(1)
T[0, 3, :3] = [10.0, 20.0, 30.0]
Rz          = euler_to_matrix([0.0, 0.0, np.radians(90)], XYZ)

rotate_then_move = matrix_multiply(Rz, T)
move_then_rotate = matrix_multiply(T, Rz)
print(matrix_point_multiply([1.0, 0.0, 0.0], rotate_then_move))  # [[10, 21, 30]]
print(matrix_point_multiply([1.0, 0.0, 0.0], move_then_rotate))  # [[-20, 11, 30]]

assert np.allclose(matrix_multiply(mats, matrix_inverse(mats)), matrix_identity(4))
assert np.allclose(matrix_transpose(mats), matrix_inverse(mats))  # true for pure rotations
```

### Normalize away scale

```python
import numpy as np
from cgmath.transforms import matrix_normalize, matrix_random

scaled = matrix_random(4, seed=12345)
scaled[:, :3, :3] *= 5.0

clean = matrix_normalize(scaled)
assert np.allclose(np.linalg.norm(clean[:, :3, :3], axis=2), 1.0)
```

### Transform points

`matrix_point_multiply(point, matrix)` — points first, matrices second.

```python
import numpy as np
from cgmath.transforms import matrix_identity, matrix_point_multiply

# eight corners of a unit cube, as a plain (8, 3) point cloud
cube = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5],
])

world           = matrix_identity(1)
world[0, 3, :3] = [0.0, 2.0, 0.0]
print(matrix_point_multiply(cube, world))     # one matrix, eight points
```

### Change of space

`matrix_local(child_world, parent_world)` and `matrix_delta(parent, child)` do
the same thing with the **arguments in the opposite order**.

```python
import numpy as np
from cgmath.transforms import matrix_delta, matrix_identity, matrix_local

parent           = matrix_identity(1)
parent[0, 3, :3] = [0.0, 5.0, 0.0]
child            = matrix_identity(1)
child[0, 3, :3]  = [0.0, 8.0, 0.0]

print(matrix_local(child, parent)[0, 3, :3])              # [0, 3, 0]
assert np.allclose(matrix_delta(parent, child), matrix_local(child, parent))
```

### Interpolate

`matrix_slerp` blends rotation only and **zeroes translation**.
`matrix_interpolate` blends scale (lerp), rotation (slerp) and translation (lerp).

```python
from cgmath.transforms import matrix_identity, matrix_interpolate, matrix_slerp

A           = matrix_identity(1)
A[0, 3, :3] = [10.0, 0.0, 0.0]
B           = matrix_identity(1)
B[0, 3, :3] = [20.0, 0.0, 0.0]

print(matrix_slerp(A, B, 0.5)[0, 3, :3])        # [0, 0, 0] -- dropped
print(matrix_interpolate(A, B, 0.5)[0, 3, :3])  # [15, 0, 0]
```

Per-row weights, and `shortest=False` to take the long way round:

```python
import numpy as np
from cgmath.transforms import matrix_random, matrix_slerp

mats   = matrix_random(4, seed=12345)                      # rotation only
mats_t = matrix_random(4, seed=999, random_position=True)  # with translation

w = np.linspace(0.0, 1.0, 4)
print(matrix_slerp(mats, mats_t, w).shape)
print(matrix_slerp(mats, mats_t, 0.5, shortest=False).shape)
```

### Maya-flat 16-element matrices

Any function taking a matrix also accepts a flat 16-float list.

```python
from cgmath.transforms import matrix_flatten, matrix_to_quaternion

flat = matrix_flatten([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
print(flat)                        # 16 floats, Maya order
print(matrix_to_quaternion(flat))  # [[0, 0, 0, 1]]
assert matrix_flatten(flat) is flat           # already flat -> returned as-is
```

### Decompose

SVD split into three 4x4s: `(translation, rotation, scale)`.

```python
import numpy as np
from cgmath.transforms import XYZ, euler_to_matrix, matrix_decompose

M = euler_to_matrix([0.0, 0.0, np.radians(45)], XYZ)[0].copy()
M[:3, :3] *= 2.0                              # uniform scale
M[3, :3] = [1.0, 2.0, 3.0]

t, r, s = matrix_decompose(M)
print(t[3, :3], np.diag(s)[:3])               # [1 2 3]  [2 2 2]
assert np.allclose(s @ r @ t, M)              # exact for uniform scale
```

### Weighted averages

```python
import numpy as np
from cgmath.transforms import (
    XYZ,
    euler_to_matrix,
    matrix_identity,
    matrix_weighted_rotational,
    matrix_weighted_transformation,
)

R0 = np.eye(3)
R1 = euler_to_matrix([0.0, 0.0, np.radians(90)], XYZ)[0, :3, :3]
print(matrix_weighted_rotational([R0, R1], [0.5, 0.5]))       # 3x3, halfway

M0        = matrix_identity(1)[0]
M1        = euler_to_matrix([0.0, 0.0, np.radians(90)], XYZ)[0].copy()
M1[3, :3] = [10.0, 0.0, 0.0]
print(matrix_weighted_transformation([M0, M1], [1.0, 1.0]))                     # weights normalised for you
print(len(matrix_weighted_transformation([M0, M1], [1.0, 1.0], flatten=True)))  # 16
```

---

## Quaternion

### Build and convert

```python
import numpy as np
from cgmath.transforms import (
    XYZ,
    euler_to_matrix,
    quaternion_random,
    quaternion_to_euler,
    quaternion_to_matrix,
)

Q = quaternion_random(4, seed=12345)
print(quaternion_to_matrix(Q).shape)            # (4, 4, 4)
print(np.degrees(quaternion_to_euler(Q, XYZ)))  # (4, 3) degrees for reading
assert np.allclose(quaternion_to_matrix(Q), euler_to_matrix(quaternion_to_euler(Q, XYZ), XYZ))
```

### Algebra

```python
import numpy as np
from cgmath.transforms import (
    quaternion_add,
    quaternion_conjugate,
    quaternion_dot,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_negate,
    quaternion_normalize,
    quaternion_random,
    quaternion_sub,
)

Q          = quaternion_random(4, seed=12345)
identity_q = np.array([[0.0, 0.0, 0.0, 1.0]])

assert np.allclose(quaternion_multiply(identity_q, Q), Q)
assert np.allclose(quaternion_multiply(Q, quaternion_inverse(Q)), identity_q)
assert np.allclose(quaternion_conjugate(Q), Q * [-1, -1, -1, 1])
assert np.allclose(quaternion_negate(Q), -Q)
assert np.allclose(quaternion_add(Q, Q), Q * 2)
assert np.allclose(quaternion_sub(Q, Q), 0.0)
assert np.allclose(quaternion_dot(Q, Q), 1.0)                     # unit quaternions
assert np.allclose(quaternion_normalize(Q * 0.1), Q)
```

`quaternion_multiply` is the Hamilton product, so composition runs in the
**opposite order** from `matrix_multiply`:

```python
import numpy as np
from cgmath.transforms import (
    axis_angle_to_quaternion,
    matrix_multiply,
    quaternion_multiply,
    quaternion_to_matrix,
)

qx = axis_angle_to_quaternion([1.0, 0.0, 0.0], np.radians(90))
qy = axis_angle_to_quaternion([0.0, 1.0, 0.0], np.radians(90))

assert np.allclose(
    quaternion_to_matrix(quaternion_multiply(qx, qy)),
    matrix_multiply(quaternion_to_matrix(qy), quaternion_to_matrix(qx)),   # note: y then x
)
```

### Slerp

```python
from cgmath.transforms import quaternion_random, quaternion_slerp

q0 = quaternion_random(1, seed=1)
q1 = quaternion_random(1, seed=2)

print(quaternion_slerp(q0, q1, 0.25))
print(quaternion_slerp(q0, q1, [0.0, 0.5, 1.0]).shape)  # (3, 4)
print(quaternion_slerp(q0, q1, 0.5, shortest=False))    # long way round
```

---

## Quaternion calculus and splines

Log / exp, nlerp and the squad spline. Less common than `quaternion_slerp`,
so they get their own section.

```python
from cgmath.transforms import (
    quaternion_exp,
    quaternion_intermediate,
    quaternion_log,
    quaternion_nlerp,
    quaternion_squad,
)
```

### log / exp — the rotation-vector tangent space

`quaternion_log` maps a unit quaternion to `axis * angle`; `quaternion_exp` maps
it back.

```python
import numpy as np
from cgmath.transforms import quaternion_random, quaternion_to_matrix
from cgmath.transforms import quaternion_exp, quaternion_log

Q = quaternion_random(4, seed=12345)

rotvec = quaternion_log(Q)                       # (4, 3)
assert np.allclose(quaternion_to_matrix(quaternion_exp(rotvec)), quaternion_to_matrix(Q))
print(np.degrees(np.linalg.norm(rotvec, axis=1)))    # rotation magnitudes in degrees
```

### nlerp — cheap slerp

Normalised linear interpolation. Fast, exact at the endpoints, but not
constant-angular-velocity in between.

```python
import numpy as np
from cgmath.transforms import quaternion_random, quaternion_slerp
from cgmath.transforms import quaternion_nlerp

q0 = quaternion_random(1, seed=1)
q1 = quaternion_random(1, seed=2)

print(quaternion_nlerp(q0, q1, 0.25))
print(quaternion_slerp(q0, q1, 0.25))            # differs in the middle, matches at 0 and 1

assert np.allclose(quaternion_nlerp(q0, q1, 0.0), quaternion_slerp(q0, q1, 0.0))
assert np.allclose(quaternion_nlerp(q0, q1, 1.0), quaternion_slerp(q0, q1, 1.0))
```

### squad — C1-continuous quaternion spline

Build the inner-quadrangle controls with `quaternion_intermediate`, then feed a
segment to `quaternion_squad`.

```python
import numpy as np
from cgmath.transforms import quaternion_random
from cgmath.transforms import quaternion_intermediate, quaternion_squad

keys = quaternion_random(4, seed=11)

c0   = quaternion_intermediate(keys[0:1], keys[1:2], keys[2:3])
c1   = quaternion_intermediate(keys[1:2], keys[2:3], keys[3:4])

mid  = quaternion_squad(keys[1:2], c0, c1, keys[2:3], 0.5)
print(mid, np.linalg.norm(mid))                  # unit length

assert np.allclose(quaternion_squad(keys[1:2], c0, c1, keys[2:3], 0.0), keys[1:2])
```

---

## Euler

All euler angles are **radians**, ordered `(x, y, z)` regardless of rotate order.

```python
import numpy as np
from cgmath.transforms import XYZ, euler_to_matrix, euler_to_quaternion

ea = np.radians([[10.0, 20.0, 30.0]])
print(euler_to_matrix(ea, XYZ).shape)  # (1, 4, 4)
print(euler_to_quaternion(ea, XYZ))    # (1, 4)
```

### Reorder

Same pose, different rotate order.

```python
import numpy as np
from cgmath.transforms import XYZ, ZYX, euler_reorder, euler_to_matrix

ea  = np.radians([[10.0, 20.0, 30.0]])

zyx = euler_reorder(ea, XYZ, ZYX)
print(np.degrees(zyx))
assert np.allclose(euler_to_matrix(ea, XYZ), euler_to_matrix(zyx, ZYX))
```

### Slerp

Both endpoints and the result can each carry their own rotate order.

```python
from cgmath.transforms import XYZ, YZX, ZXY, euler_random, euler_slerp

e0 = euler_random(4, seed=1)
e1 = euler_random(4, seed=2)

print(euler_slerp(e0, e1, 0.5).shape)                            # (4, 3), all XYZ
print(euler_slerp(e0, e1, 0.5, axes0=XYZ, axes1=ZXY, axes=YZX).shape)
```

### Filter

Removes 180 / 360 degree branch jumps from euler curves sampled over time.
Frames run along the first axis. `axes` is **required** — the flip identity
depends on the rotate order.

```python
import numpy as np
from cgmath.transforms import (
    XYZ,
    euler_filter,
    euler_to_matrix,
    euler_to_quaternion,
    quaternion_to_euler,
)

# a smooth sweep about Y, decomposed one frame at a time -> curve steps by half a turn
sweep = np.array([
    quaternion_to_euler(
        euler_to_quaternion(np.radians([[0.0, a, 0.0]]), XYZ), XYZ
    )[0]
    for a in np.linspace(-170.0, 170.0, 48)
])

raw_step   = np.degrees(np.abs(np.diff(sweep, axis=0)).max())
clean      = euler_filter(sweep, XYZ)
clean_step = np.degrees(np.abs(np.diff(clean, axis=0)).max())

print(raw_step, "->", clean_step)                # ~180+ -> < 10
assert np.allclose(euler_to_matrix(sweep, XYZ), euler_to_matrix(clean, XYZ))   # pose preserved
```

It filters stacks too — one rotate order per curve:

```python
import numpy as np
from cgmath.transforms import XYZ, XZY, euler_filter, euler_to_quaternion, quaternion_to_euler

sweep = np.array([
    quaternion_to_euler(
        euler_to_quaternion(np.radians([[0.0, a, 0.0]]), XYZ), XYZ
    )[0]
    for a in np.linspace(-170.0, 170.0, 48)
])

block = np.stack([sweep, sweep], axis=1)         # (48, 2, 3)
print(euler_filter(block, [XYZ, XZY]).shape)     # (48, 2, 3)
```

### Random

```python
from cgmath.transforms import euler_random

print(euler_random(3, seed=5))                   # (3, 3) radians in [-2pi, 2pi]
```

---

## Axis / angle

An axis vector plus an angle in radians.

```python
import numpy as np
from cgmath.transforms import (
    XYZ,
    axis_angle_to_euler,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
)

np.set_printoptions(precision=4, suppress=True)      # so sin(180 deg) reads as 0, not 1e-16

print(axis_angle_to_quaternion([1.0, 0.0, 0.0], np.radians(90)))
print(axis_angle_to_matrix([0.0, 1.0, 0.0], np.radians(180))[0])
print(np.degrees(axis_angle_to_euler([0.0, 0.0, 1.0], np.radians(90), axes=XYZ)))
```

Batched — one axis and one angle per row:

```python
import numpy as np
from cgmath.transforms import axis_angle_to_matrix

axes3  = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
angles = np.radians([45.0, 90.0, 180.0])
print(axis_angle_to_matrix(axes3, angles).shape)     # (3, 4, 4)
```

`angle` defaults to `0.0`, which gives identity:

```python
import numpy as np
from cgmath.transforms import axis_angle_to_quaternion

assert np.allclose(axis_angle_to_quaternion([1.0, 0.0, 0.0]), [[0.0, 0.0, 0.0, 1.0]])
```

---

## Vector

### Basics

```python
import numpy as np
from cgmath.transforms import (
    vector_angle,
    vector_cross,
    vector_dot,
    vector_magnitude,
    vector_normalize,
    vector_random,
)

V = vector_random(5, seed=3)                  # (5, 3) in [-1, 1]
U = vector_random(5, seed=3, normalize=True)  # unit length

assert np.allclose(vector_magnitude(U), 1.0)
assert np.allclose(vector_normalize(V), V / vector_magnitude(V)[:, None])
assert np.allclose(vector_cross(V, U), np.cross(V, U))
assert np.allclose(vector_dot(V, U), np.einsum("...i,...i", V, U))

print(np.degrees(vector_angle([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])))    # 90
```

### Interpolate

```python
from cgmath.transforms import vector_lerp, vector_slerp

print(vector_lerp([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0.25))          # [[2.5, 0, 0]]
print(vector_slerp([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 1.0]))
```

`vector_lerp` maps NaNs to 0:

```python
import numpy as np
from cgmath.transforms import vector_lerp

print(vector_lerp([0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], 0.5))         # [[0, 0.5, 0.5]]
```

### Shortest arc between two directions

```python
import numpy as np
from cgmath.transforms import (
    matrix_point_multiply,
    vector_arc_to_euler,
    vector_arc_to_matrix,
    vector_arc_to_quaternion,
)

print(vector_arc_to_quaternion([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
print(np.degrees(vector_arc_to_euler([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])))   # [[0, 0, 90]]

arc = vector_arc_to_matrix([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
assert np.allclose(matrix_point_multiply([1.0, 0.0, 0.0], arc), [[0.0, 1.0, 0.0]])
```

### Aim + up frames

Build an orientation from an aim direction and an up direction. `aim_axis` and
`up_axis` say which **local** axis each vector drives. The up vector is
orthogonalised against the aim vector for you.

```python
import numpy as np
from cgmath.transforms import (
    X,
    Y,
    matrix_identity,
    vector_to_euler,
    vector_to_matrix,
    vector_to_quaternion,
)

M = vector_to_matrix([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], aim_axis=X, up_axis=Y)
print(M[0])                                      # row 0 == aim, row 1 == up

assert np.allclose(vector_to_matrix([1.0, 0, 0], [0.0, 1, 0]), matrix_identity(1))
assert np.allclose(vector_to_quaternion([1.0, 0, 0], [0.0, 1, 0]), [[0.0, 0, 0, 1]])
assert np.allclose(vector_to_euler([1.0, 0, 0], [0.0, 1, 0]), [[0.0, 0, 0]])

# a non-orthogonal up still resolves cleanly
assert np.allclose(vector_to_matrix([1.0, 0, 0], [1.0, 1.0, 0.0]), matrix_identity(1))
```

Aim a whole grid of points at the origin:

```python
import numpy as np
from cgmath.transforms import Y, Z, vector_normalize, vector_to_matrix

# a 3 x 3 grid of points on the XZ plane
grid   = np.array([[x, 0.0, z] for x in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)])

aim    = vector_normalize(-grid + [0.0, 1e-9, 0.0])
frames = vector_to_matrix(aim, [0.0, 1.0, 0.0], aim_axis=Z, up_axis=Y)
print(frames.shape)                              # (9, 4, 4)
```

---

## Broadcasting

Inputs are levelled by `cgmath.transforms.utils._match_depth`, which follows NumPy's
rules: every input must be as long as the longest one, or exactly **1**. A
length-1 input is expanded to match — as a zero-copy view, not a copy — so a
single matrix pairs with N points, a single weight with N quaternions, and so
on.

```python
from cgmath.transforms import (
    matrix_point_multiply,
    matrix_random,
    quaternion_random,
    quaternion_slerp,
    vector_random,
)

points     = vector_random(8, seed=2)                         # (8, 3)
one_matrix = matrix_random(1, seed=4)
print(matrix_point_multiply(points, one_matrix).shape)    # (8, 3) -- matrix reused

quats = quaternion_random(4, seed=12345)
print(quaternion_slerp(quats, quats[::-1], 0.5).shape)    # (4, 4)
print(quaternion_slerp(quats, quats[::-1], [0.0, 0.25, 0.5, 1.0]).shape)
```

A bare `(3,)` vector or `(4, 4)` matrix is promoted to a stack of one, so the
return value is always batched:

```python
import numpy as np
from cgmath.transforms import matrix_to_quaternion, vector_normalize

print(vector_normalize([1.0, 2.0, 3.0]).shape)  # (1, 3), not (3,)
print(matrix_to_quaternion(np.eye(4)).shape)    # (1, 4)
```

Any other length is a mistake and raises. Nothing is padded, recycled or
truncated to make the call go through:

```python
from cgmath.transforms import vector_dot, vector_random

try:
    vector_dot(vector_random(5, seed=5), vector_random(2, seed=6))
except ValueError as err:
    print(err)      # length mismatch: got [5, 2]; every input must be length 5 or 1
```

This matters most where a wrong answer looks plausible — an off-by-one in a
joint walk, or a rotate-order list built from a partial selection:

```python
from cgmath.transforms import XYZ, ZYX, matrix_random, matrix_to_euler

try:
    matrix_to_euler(matrix_random(10, seed=3), [XYZ, ZYX, XYZ])
except ValueError as err:
    print(err)      # length mismatch: got [10, 3]; every input must be length 10 or 1
```

A length-0 input raises for the same reason, rather than being skipped.

---

## Gotchas

- **`matrix_slerp` discards translation.** The result is a pure rotation with a
  zero translation row. Use `matrix_interpolate` for full SRT.

- **`matrix_decompose` is SVD-based**, so it is exact only for
  rotation + uniform scale + translation. With non-uniform scale the singular
  values come back sorted descending and no longer line up with X / Y / Z.

  ```python
  import numpy as np
  from cgmath.transforms import XYZ, euler_to_matrix, matrix_decompose

  M         = euler_to_matrix([0.0, 0.0, np.radians(45)], XYZ)[0].copy()
  M[:3, :3] = np.diag([2.0, 3.0, 4.0]) @ M[:3, :3]
  _, _, s = matrix_decompose(M)
  print(np.diag(s)[:3])                           # [4, 3, 2] -- sorted, not per-axis
  ```

- **Quaternion products compose in the opposite order from matrix products.**
  `quaternion_multiply(a, b)` equals `matrix_multiply(B, A)`.

- **`matrix_local` and `matrix_delta` take their arguments in opposite orders.**
  `matrix_local(child, parent)` versus `matrix_delta(parent, child)`.

- **Rotate order matters for `euler_filter`** and it has no default. A wrong
  order stops preserving the pose instead of erroring.

- **Kernels are Numba JIT.** The first call to each compiles; import stays cheap
  because the kernel imports are deferred to call time.
