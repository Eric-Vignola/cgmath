# `cgmath.rbf` Cheatsheet

Copy-paste recipes for the RBF kernel shelf and the JIT LU solver behind it.
For the overview see [`README.md`](README.md); for the rest of the library see
[`../README.md`](../README.md) and [`../CHEATSHEET.md`](../CHEATSHEET.md).

Every `python` block below runs in order, in one namespace. Start at Setup.

---

## Contents

- [Setup](#setup)
- [Import paths — the package `__init__` is empty](#import-paths--the-package-__init__-is-empty)
- [The call convention](#the-call-convention)
- [Kernel discovery — `list_kernels`, `KERNEL_REGISTRY`, `get_kernel`](#kernel-discovery--list_kernels-kernel_registry-get_kernel)
- [Infinitely smooth kernels](#infinitely-smooth-kernels)
- [Polyharmonic kernels](#polyharmonic-kernels)
- [Compactly supported kernels — the `r` argument](#compactly-supported-kernels--the-r-argument)
- [Specialized kernels](#specialized-kernels)
- [The shape parameter gotcha](#the-shape-parameter-gotcha)
- [Distances — `cdist_euclidean`](#distances--cdist_euclidean)
- [Scattered-data interpolation, end to end](#scattered-data-interpolation-end-to-end)
- [Polynomial augmentation](#polynomial-augmentation)
- [Vector-valued interpolation — warp a point cloud](#vector-valued-interpolation--warp-a-point-cloud)
- [`rbf._numba._lu` — the JIT LU solver](#rbf_numba_lu--the-jit-lu-solver)
- [Reusing a factorization — `lu_solve_factored`](#reusing-a-factorization--lu_solve_factored)
- [Calling `lu_solve` from your own `@njit` code](#calling-lu_solve-from-your-own-njit-code)
- [Singular systems](#singular-systems)
- [`geometry.deform.WrapData` — the high-level consumer](#geometrydeformwrapdata--the-high-level-consumer)
- [Numba notes](#numba-notes)

---

## Setup

No mesh assets ship with this repo. Everything is built inline.

```python
import numpy as np

rng = np.random.default_rng(0)

# A small distance matrix to poke kernels with.
D = np.array([[0.0, 0.5, 1.0],
              [0.5, 0.0, 1.5],
              [1.0, 1.5, 0.0]])

# Scattered 2D sites + a smooth field sampled on them.
sites = rng.random((80, 2)) * 2.0 - 1.0
field = np.sin(2.0 * sites[:, 0]) * np.cos(2.0 * sites[:, 1])

# A regular query grid over the same domain.
gx, gy = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))
grid = np.column_stack([gx.ravel(), gy.ravel()])

# A unit cube, the standard MeshData fixture (8 points, 6 quads).
cube_points = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
])
cube_indices = np.array([
    0, 3, 2, 1,   4, 5, 6, 7,   0, 1, 5, 4,
    1, 2, 6, 5,   2, 3, 7, 6,   3, 0, 4, 7,
], dtype=np.int32)
cube_counts = np.array([4, 4, 4, 4, 4, 4], dtype=np.int32)

print(D.shape, sites.shape, grid.shape, cube_points.shape)
```

---

## Import paths — the package `__init__` is empty

`cgmath.rbf/__init__.py` exports **nothing**. That is deliberate:
`geometry.deform.wrap` imports `rbf._kernels` at module scope, so re-exporting
`WrapData` from here would import back into a partially-initialized module.

```python
import cgmath.rbf as rbf

assert [n for n in dir(rbf) if not n.startswith("_")] == []
```

Import the real surface from the submodule. The leading underscore names the
file, not the visibility — `_kernels` is the public shelf.

```python
from cgmath.rbf._kernels import (
    # discovery
    get_kernel, list_kernels, KERNEL_REGISTRY,
    # infinitely smooth
    gaussian, multiquadric, inverse_multiquadric, inverse_quadratic,
    # polyharmonic
    linear, cubic, quintic, thin_plate_spline, polyharmonic,
    # compactly supported
    wendland_c0, wendland_c2, wendland_c4, wendland_c6, wu_c2, wu_c4, bump,
    # specialized
    matern_12, matern_32, matern_52, cauchy, log_kernel,
)
```

`lu_solve` / `lu_solve_factored` come off the same module via a module-level
`__getattr__`, so they cannot be listed in a `from ... import (...)` tuple that
static tools check — but a plain attribute access works.

```python
from cgmath.rbf import _kernels

lu_solve          = _kernels.lu_solve
lu_solve_factored = _kernels.lu_solve_factored

try:
    _kernels.no_such_thing
except AttributeError as err:
    print(err)
```

---

## The call convention

Every kernel takes a **distance matrix**, not a scalar, and returns an array of
the same shape in `float64`. Inputs are coerced with `np.asarray(..., float64)`,
so lists and integer arrays are fine.

| | |
|---|---|
| Signature | `phi(X, epsilon=1.0)` or `phi(X, r=1.0)` |
| `X` | `(n, m)` array of distances — must be 2-D |
| Returns | `(n, m)` `float64` |
| Second arg | `epsilon` = shape parameter (14 kernels) · `r` = support radius (7 kernels) |

```python
print(gaussian([[0, 1], [2, 3]]))          # list in, float64 out
print(gaussian(D).shape, gaussian(D).dtype)

# 2-D only: the njit kernels unpack X.shape into (n, m).
try:
    gaussian(np.array([0.0, 1.0, 2.0]))
except Exception as err:
    print(type(err).__name__)
```

`polyharmonic` is the one kernel with three parameters —
`polyharmonic(X, k=2, epsilon=1.0)`.

---

## Kernel discovery — `list_kernels`, `KERNEL_REGISTRY`, `get_kernel`

```python
print(len(list_kernels()))
print(list_kernels())
```

`KERNEL_REGISTRY` is the underlying `name -> callable` dict.
`list_kernels()` returns its sorted keys, de-duplicated by function identity
(there are currently no aliases, so all 21 come back).

```python
assert set(KERNEL_REGISTRY) == set(list_kernels())
assert KERNEL_REGISTRY["gaussian"] is gaussian
```

`get_kernel(name)` is case-insensitive and raises `ValueError` on a miss.

```python
assert get_kernel("Gaussian") is get_kernel("GAUSSIAN") is gaussian

try:
    get_kernel("rbf")
except ValueError as err:
    print(str(err)[:60], "...")
```

One name/function mismatch to know: `log_kernel` is registered under `"log"`.

```python
assert get_kernel("log") is log_kernel

try:
    get_kernel("log_kernel")
except ValueError:
    print("registered as 'log', not 'log_kernel'")
```

The seven compactly-supported kernels take `r=`, not `epsilon=`. Branch on the
name when you dispatch generically — `geometry.deform.wrap` exports the set.

```python
from cgmath.geometry.deform.wrap import _COMPACT_KERNELS

print(sorted(_COMPACT_KERNELS))

for name in list_kernels():
    fn  = get_kernel(name)
    out = fn(D, r=2.0) if name in _COMPACT_KERNELS else fn(D)
    assert out.shape == D.shape and np.all(np.isfinite(out)), name
print("all 21 kernels finite on D")
```

---

## Infinitely smooth kernels

`C-infinity`, global support. `epsilon` **multiplies** the distance here, so it
is an *inverse* length scale: bigger `epsilon` = narrower kernel.

| Function | phi(r) | Notes |
|---|---|---|
| `gaussian(X, epsilon=1.0)` | `exp(-(eps*r)^2)` | strictly positive definite, the default choice |
| `multiquadric(X, epsilon=1.0)` | `sqrt(1 + (eps*r)^2)` | conditionally PD order 1 — augment with a constant |
| `inverse_multiquadric(X, epsilon=1.0)` | `1 / sqrt(1 + (eps*r)^2)` | strictly PD, no augmentation needed |
| `inverse_quadratic(X, epsilon=1.0)` | `1 / (1 + (eps*r)^2)` | strictly PD, faster decay than IMQ |

```python
r = np.array([[0.0, 0.5, 1.0, 2.0]])

print("gaussian  ", np.round(gaussian(r), 4))
print("multiquad ", np.round(multiquadric(r), 4))
print("inv multiq", np.round(inverse_multiquadric(r), 4))
print("inv quad  ", np.round(inverse_quadratic(r), 4))
```

Tightening `epsilon` on the gaussian:

```python
for eps in (0.5, 1.0, 4.0):
    print(f"eps={eps:4}", np.round(gaussian(r, epsilon=eps), 4))
```

All four are 1 (or `sqrt(1)`) at zero distance:

```python
z = np.zeros((1, 1))
assert gaussian(z)[0, 0] == 1.0
assert multiquadric(z)[0, 0] == 1.0
assert inverse_multiquadric(z)[0, 0] == 1.0
assert inverse_quadratic(z)[0, 0] == 1.0
```

---

## Polyharmonic kernels

Piecewise-smooth splines with unbounded growth. Here `epsilon` **divides** the
distance — it is a true length scale.

| Function | phi(r) | Continuity |
|---|---|---|
| `linear(X, epsilon=1.0)` | `r` | C0 |
| `cubic(X, epsilon=1.0)` | `r^3` | C1 |
| `quintic(X, epsilon=1.0)` | `r^5` | C2 |
| `thin_plate_spline(X, epsilon=1.0)` | `r^2 log(r)`, `0` at `r=0` | C1, minimises bending energy |
| `polyharmonic(X, k=2, epsilon=1.0)` | `r^k` for odd `k`, `r^k log(r)` for even `k` | family generaliser |

```python
print("linear   ", np.round(linear(r), 4))
print("cubic    ", np.round(cubic(r), 4))
print("quintic  ", np.round(quintic(r), 4))
print("tps      ", np.round(thin_plate_spline(r), 4))
```

TPS goes **negative** below `r = 1` — `r^2 log(r) < 0` there. That is correct,
not a bug; the polynomial tail of the augmented system absorbs it.

```python
assert thin_plate_spline(np.array([[0.5]]))[0, 0] < 0.0
assert thin_plate_spline(np.array([[0.0]]))[0, 0] == 0.0
```

`polyharmonic` reproduces the named members of the family:

```python
assert np.allclose(polyharmonic(r, k=1), linear(r))
assert np.allclose(polyharmonic(r, k=2), thin_plate_spline(r))
assert np.allclose(polyharmonic(r, k=3), cubic(r))
assert np.allclose(polyharmonic(r, k=5), quintic(r))
print("k=4:", np.round(polyharmonic(r, k=4), 4))
```

Scaling with `epsilon`: bigger `epsilon` shrinks the response.

```python
for eps in (0.5, 1.0, 2.0):
    print(f"cubic eps={eps:4}", np.round(cubic(r, epsilon=eps), 4))
```

---

## Compactly supported kernels — the `r` argument

These take `r=` (support radius), not `epsilon=`. Internally they evaluate at
`arg = X / r` and return **exactly zero** for `arg >= 1`, i.e. for any distance
at or beyond `r`. That zero block is what makes the interpolation matrix sparse
and well conditioned for local deformation.

| Function | phi(q), `q = X/r` | Continuity | phi(0) |
|---|---|---|---|
| `wendland_c0(X, r=1.0)` | `(1-q)+^2` | C0 | 1 |
| `wendland_c2(X, r=1.0)` | `(1-q)+^4 (4q + 1)` | C2 | 1 |
| `wendland_c4(X, r=1.0)` | `(1-q)+^6 (35q^2 + 18q + 3)` | C4 | **3** |
| `wendland_c6(X, r=1.0)` | `(1-q)+^8 (32q^3 + 25q^2 + 8q + 1)` | C6 | 1 |
| `wu_c2(X, r=1.0)` | `(1-q)+^5 (8q^2 + 5q + 1)` | C2 | 1 |
| `wu_c4(X, r=1.0)` | `(1-q)+^7 (16q^3 + 12q^2 + 5q + 1)` | C4 | 1 |
| `bump(X, r=1.0)` | `exp(-1/(1-q^2))` for `q < 1` else `0` | C-infinity | **exp(-1)** |

```python
q = np.array([[0.0, 0.25, 0.5, 0.9, 1.0, 2.0]])

for fn in (wendland_c0, wendland_c2, wendland_c4, wendland_c6, wu_c2, wu_c4, bump):
    print(f"{fn.__name__:12}", np.round(fn(q, r=1.0), 4))
```

Everything at or past the radius is zero:

```python
for fn in (wendland_c0, wendland_c2, wendland_c4, wendland_c6, wu_c2, wu_c4, bump):
    assert fn(np.array([[1.0, 1.5, 9.0]]), r=1.0).max() == 0.0
```

Widening the radius keeps more neighbours alive:

```python
d = np.array([[0.5, 1.5, 2.5]])
for radius in (1.0, 2.0, 4.0):
    print(f"r={radius}", np.round(wendland_c2(d, r=radius), 4),
          "nonzero:", int(np.count_nonzero(wendland_c2(d, r=radius))))
```

Two of them are **not normalised to 1** at zero distance — the polynomials are
written in their textbook form, un-scaled. `wendland_c4` peaks at **3**, `bump`
peaks at **exp(-1)**. Divide the peak out if you need a partition-of-unity
weight; for interpolation it does not matter, a constant factor on `K` cancels
in the solve.

```python
z1 = np.zeros((1, 1))
peaks = {fn.__name__: float(fn(z1, r=1.0)[0, 0])
         for fn in (wendland_c0, wendland_c2, wendland_c4, wendland_c6,
                    wu_c2, wu_c4, bump)}
print(peaks)

assert peaks["wendland_c4"] == 3.0
assert np.isclose(peaks["bump"], np.exp(-1.0))
assert all(peaks[n] == 1.0 for n in
           ("wendland_c0", "wendland_c2", "wendland_c6", "wu_c2", "wu_c4"))
```

Wendland C0 is the only one that is not differentiable at the boundary; C2 is
the usual default for deformation work.

---

## Specialized kernels

`epsilon` **divides** for all five.

| Function | phi(r) | Notes |
|---|---|---|
| `matern_12(X, epsilon=1.0)` | `exp(-r/eps)` | exponential kernel, non-differentiable paths |
| `matern_32(X, epsilon=1.0)` | `(1 + z) exp(-z)`, `z = sqrt(3) r/eps` | once differentiable |
| `matern_52(X, epsilon=1.0)` | `(1 + z + 5r^2/(3 eps^2)) exp(-z)`, `z = sqrt(5) r/eps` | twice differentiable |
| `cauchy(X, epsilon=1.0)` | `1 / (1 + (r/eps)^2)` | heavy tailed |
| `log_kernel(X, epsilon=1.0)` | `log(1 + r/eps)` | conditionally PD, slow growth |

```python
print("matern_12 ", np.round(matern_12(r), 4))
print("matern_32 ", np.round(matern_32(r), 4))
print("matern_52 ", np.round(matern_52(r), 4))
print("cauchy    ", np.round(cauchy(r), 4))
print("log_kernel", np.round(log_kernel(r), 4))
```

The Matern family orders itself by smoothness at a fixed distance:

```python
one = np.array([[1.0]])
assert matern_12(one) < matern_32(one) < matern_52(one)
```

`cauchy` with `epsilon=1` equals `inverse_quadratic` with `epsilon=1` — the two
differ only in whether the shape parameter multiplies or divides.

```python
assert np.allclose(cauchy(r, epsilon=1.0), inverse_quadratic(r, epsilon=1.0))
assert np.allclose(cauchy(r, epsilon=2.0), inverse_quadratic(r, epsilon=0.5))
```

---

## The shape parameter gotcha

`epsilon` is **not** consistent across the shelf. Four kernels multiply by it,
ten divide by it. Get this backwards and your kernel width goes the wrong way.

| Group | Kernels | Effect of `epsilon` |
|---|---|---|
| Multiply — `eps*r` | `gaussian`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic` | inverse length scale: **bigger = narrower** |
| Divide — `r/eps` | `linear`, `cubic`, `quintic`, `thin_plate_spline`, `polyharmonic`, `matern_12/32/52`, `cauchy`, `log_kernel` | length scale: **bigger = wider** |
| Divide — `r/radius` | `wendland_c0/c2/c4/c6`, `wu_c2/c4`, `bump` (`r=` arg) | support cutoff: **bigger = wider** |

```python
probe = np.array([[1.0]])

# gaussian: eps up -> value down (narrower)
assert gaussian(probe, epsilon=4.0) < gaussian(probe, epsilon=1.0)

# cauchy: eps up -> value up (wider)
assert cauchy(probe, epsilon=4.0) > cauchy(probe, epsilon=1.0)
```

---

## Distances — `cdist_euclidean`

There is no distance helper inside `rbf`. The numba pairwise distance the
kernels are normally fed lives with the wrap deformer:

```python
from cgmath.geometry.deform.wrap import cdist_euclidean

A = np.array([[0.0, 0.0, 0.0]])
B = np.array([[3.0, 4.0, 0.0]])
assert np.isclose(cdist_euclidean(A, B)[0, 0], 5.0)

DD = cdist_euclidean(sites, sites)
print(DD.shape)                       # (m, n) for X (m, k), Y (n, k)
assert np.allclose(np.diag(DD), 0.0)
assert np.allclose(DD, DD.T)
```

Or stay dependency-free with three lines of numpy — fine below a few thousand
points, and it works on any dimensionality:

```python
def cdist(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))

assert np.allclose(cdist(sites, sites), cdist_euclidean(sites, sites))
```

---

## Scattered-data interpolation, end to end

The whole recipe, with a strictly positive-definite kernel so no polynomial
augmentation is needed:

1. `K = phi(cdist(centres, centres))`
2. solve `K @ w = values`
3. evaluate `phi(cdist(query, centres)) @ w`

```python
EPS = 1.0

K   = gaussian(cdist_euclidean(sites, sites), epsilon=EPS)
w   = lu_solve(K, field)

H         = gaussian(cdist_euclidean(grid, sites), epsilon=EPS)
predicted = H @ w

print(predicted.shape)
```

An interpolant reproduces its own data exactly at the centres:

```python
at_sites = gaussian(cdist_euclidean(sites, sites), epsilon=EPS) @ w
print("max residual at centres:", np.abs(at_sites - field).max())
assert np.allclose(at_sites, field, atol=1e-6)
```

And it tracks the true field between them:

```python
truth = np.sin(2.0 * grid[:, 0]) * np.cos(2.0 * grid[:, 1])
print("max grid error:", round(float(np.abs(predicted - truth).max()), 5))
```

Expect noisy `RuntimeWarning: divide by zero / overflow / invalid value
encountered in matmul` from these matmuls on some BLAS builds even though the
result is finite and correct. `geometry.deform.wrap` wraps its own solves in
`np.errstate` for exactly this reason:

```python
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    quiet = H @ w
assert np.allclose(quiet, predicted)
```

Swap the kernel by name to compare — note the shared `epsilon` means something
different to each one, so this is a ranking of *these settings*, not of the
kernels:

```python
for name in ("gaussian", "inverse_multiquadric", "matern_52", "cauchy"):
    fn  = get_kernel(name)
    ww  = lu_solve(fn(cdist_euclidean(sites, sites), epsilon=EPS), field)
    err = np.abs(fn(cdist_euclidean(grid, sites), epsilon=EPS) @ ww - truth).max()
    print(f"{name:22} max grid error {err:.4f}")
```

A compactly supported kernel trades accuracy for a **sparse** matrix and a
purely local interpolant. Pick a radius that comfortably covers the site
spacing — too small and the system degenerates:

```python
for RADIUS in (0.6, 1.2, 1.6):
    Kc    = wendland_c2(cdist_euclidean(sites, sites), r=RADIUS)
    wc    = lu_solve(Kc, field)
    local = wendland_c2(cdist_euclidean(grid, sites), r=RADIUS) @ wc
    print(f"r={RADIUS}  err {np.abs(local - truth).max():.4f}"
          f"  sparsity {1.0 - np.count_nonzero(Kc) / Kc.size:.3f}")
```

---

## Polynomial augmentation

`thin_plate_spline`, `linear`, `cubic`, `quintic`, `multiquadric` and
`log_kernel` are only *conditionally* positive definite. Bordering the system
with a low-order polynomial makes it solvable and lets the interpolant
reproduce that polynomial exactly.

```
[ K    P ] [w]   [f]
[ P^T  0 ] [c] = [0]
```

```python
def tps_fit(centres, values):
    n      = len(centres)
    K      = thin_plate_spline(cdist_euclidean(centres, centres))
    P      = np.column_stack([np.ones(n), centres])          # [1, x, y]
    Z      = np.zeros((P.shape[1], P.shape[1]))
    system = np.block([[K, P], [P.T, Z]])
    rhs    = np.concatenate([values, np.zeros(P.shape[1])])
    return lu_solve(system, rhs)


def tps_eval(solution, query, centres):
    H  = thin_plate_spline(cdist_euclidean(query, centres))
    Pq = np.column_stack([np.ones(len(query)), query])
    return np.block([H, Pq]) @ solution


sol = tps_fit(sites, field)
print("tps grid error:", round(float(np.abs(tps_eval(sol, grid, sites) - truth).max()), 4))
```

The linear tail means a linear function comes back to machine precision:

```python
plane       = 2.0 * sites[:, 0] - 3.0 * sites[:, 1] + 1.0
plane_truth = 2.0 * grid[:, 0] - 3.0 * grid[:, 1] + 1.0

got = tps_eval(tps_fit(sites, plane), grid, sites)
print("plane reproduction error:", float(np.abs(got - plane_truth).max()))
assert np.allclose(got, plane_truth, atol=1e-8)
```

---

## Vector-valued interpolation — warp a point cloud

The right-hand side can be `(n, m)`; you get `(n, m)` coefficients back and one
solve covers all columns.

```python
src = cube_points
dst = cube_points.copy()
dst[4:, 2] += 0.5                                # lift the top face

Kv = thin_plate_spline(cdist_euclidean(src, src))
Pv = np.column_stack([np.ones(len(src)), src])
Sv = np.block([[Kv, Pv], [Pv.T, np.zeros((4, 4))]])
Rv = np.vstack([dst, np.zeros((4, 3))])          # (n+4, 3) RHS

coeff = lu_solve(Sv, Rv)
print(coeff.shape)
```

Evaluate anywhere — here at the cube's own vertices and at its centre:

```python
def warp(points):
    H = thin_plate_spline(cdist_euclidean(points, src))
    return np.block([H, np.column_stack([np.ones(len(points)), points])]) @ coeff


assert np.allclose(warp(src), dst, atol=1e-8)
print(np.round(warp(np.array([[0.5, 0.5, 0.5]])), 4))
```

---

## `rbf._numba._lu` — the JIT LU solver

`cgmath.rbf._numba._lu` is a small dense linear-algebra kernel written in
numba: Doolittle LU with partial pivoting, plus forward/backward substitution.
It exists so RBF systems can be solved **from inside other jit-compiled code**,
where `np.linalg.solve` is available but LAPACK factorization reuse is not.
From pure Python, `np.linalg.solve` is usually the better choice; reach for
`lu_solve` when you need the factorization back, or when the call site is
`@njit`.

| Symbol | Signature | What |
|---|---|---|
| `lu_solve` | `(A, B) -> X` | solve `A X = B`, `A` is `(n, n)`, `B` is `(n,)` or `(n, m)` |
| `lu_solve_factored` | `(L, U, perm, B) -> X` | reuse a factorization, `O(n^2)` |
| `_lu_decompose` | `(A) -> (L, U, perm)` | `P A = L U`, `perm` is the row permutation vector |
| `_forward_substitute` | `(L, b, perm) -> y` | solve `L y = P b` |
| `_backward_substitute` | `(U, y) -> x` | solve `U x = y` |

The two public names are re-exposed on `rbf._kernels` through a module-level
`__getattr__`, which hands back the raw `CPUDispatcher` rather than a Python
wrapper — a wrapper could not be called from jit code.

```python
from cgmath.rbf._numba._lu import (
    _backward_substitute, _forward_substitute, _lu_decompose,
)

M  = rng.random((6, 6)) + np.eye(6) * 6.0
b1 = rng.random(6)
B3 = rng.random((6, 3))

x1 = lu_solve(M, b1)
X3 = lu_solve(M, B3)

print(x1.shape, X3.shape)
assert np.allclose(M @ x1, b1)
assert np.allclose(M @ X3, B3)
assert np.allclose(x1, np.linalg.solve(M, b1))
```

The factorization itself:

```python
L, U, perm = _lu_decompose(M)

assert np.allclose(np.tril(L), L) and np.allclose(np.diag(L), 1.0)
assert np.allclose(np.triu(U), U)
assert np.allclose(L @ U, M[perm])                 # P A = L U
print("perm:", perm)
```

Driving the substitutions by hand is exactly what `lu_solve` does:

```python
y = _forward_substitute(L, b1, perm)
assert np.allclose(_backward_substitute(U, y), x1)
```

`_lu_decompose` copies `A`, so your input is never clobbered:

```python
before = M.copy()
_lu_decompose(M)
assert np.array_equal(M, before)
```

---

## Reusing a factorization — `lu_solve_factored`

Factor once, then every extra right-hand side is `O(n^2)` instead of `O(n^3)`.
This is the pattern for animating targets over a fixed set of RBF centres.

```python
Kf = gaussian(cdist_euclidean(sites, sites), epsilon=EPS)
Lf, Uf, pf = _lu_decompose(Kf)

for phase in (0.0, 0.5, 1.0):
    rhs = np.sin(2.0 * sites[:, 0] + phase) * np.cos(2.0 * sites[:, 1])
    wf  = lu_solve_factored(Lf, Uf, pf, rhs)
    assert np.allclose(Kf @ wf, rhs, atol=1e-8)
print("3 re-solves off one factorization")
```

It agrees with the full solve:

```python
assert np.allclose(lu_solve_factored(Lf, Uf, pf, field), lu_solve(Kf, field))
```

Matrix right-hand sides work here too:

```python
multi = np.column_stack([field, field * 2.0, np.ones(len(field))])
assert np.allclose(Kf @ lu_solve_factored(Lf, Uf, pf, multi), multi, atol=1e-8)
```

---

## Calling `lu_solve` from your own `@njit` code

This is the reason the solver exists. `lu_solve` is a `CPUDispatcher`, so numba
can inline it into your own kernel — no object-mode fallback.

```python
from numba import njit


@njit
def solve_many(A, rhs_stack):
    out = np.empty_like(rhs_stack)
    for i in range(rhs_stack.shape[0]):
        out[i] = lu_solve(A, rhs_stack[i])
    return out


stack = rng.random((4, 6))
res   = solve_many(M, stack)
for i in range(4):
    assert np.allclose(M @ res[i], stack[i])
print("njit driver ok:", res.shape)
```

---

## Singular systems

`lu_solve` does **not** raise on a singular matrix. Pivots with magnitude below
`1e-15` are skipped and the corresponding unknown is set to `0.0`, so you get a
finite but meaningless answer. Check the residual yourself if the input might
be rank deficient.

```python
S   = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
rhs = np.array([1.0, 3.0])                # inconsistent

out = lu_solve(S, rhs)
print("solution:", out, "residual:", np.abs(S @ out - rhs).max())
assert np.all(np.isfinite(out))

try:
    np.linalg.solve(S, rhs)                      # numpy is stricter
except np.linalg.LinAlgError as err:
    print("numpy raises:", err)
```

Guard with a residual test:

```python
def solve_checked(A, b, tol=1e-8):
    x = lu_solve(A, b)
    if np.abs(A @ x - b).max() > tol:
        raise ValueError("lu_solve did not converge; matrix is rank deficient")
    return x


try:
    solve_checked(S, rhs)
except ValueError as err:
    print(err)
```

Compactly supported kernels with too small a radius are the usual way to
produce a singular RBF matrix — every off-diagonal entry falls outside support
and `K` degenerates to the identity, so nothing interpolates.

```python
tiny = wendland_c2(cdist_euclidean(sites, sites), r=0.01)
print("off-diagonal nonzeros:", int(np.count_nonzero(tiny) - len(sites)))
```

---

## `geometry.deform.WrapData` — the high-level consumer

`WrapData` is the only in-library consumer of `rbf._kernels`. It wraps the whole
recipe above — distance matrix, polynomial augmentation, solve, evaluate — with
dirty-flag caching and optional geodesic masking. Prefer it over hand-rolling
when you are deforming geometry.

Note it solves with `np.linalg.pinv`, not with `lu_solve`.

```python
from cgmath.geometry.deform import WrapData

cage   = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
offset = np.array([1.0, 2.0, 3.0])

wrap   = WrapData(kernel="thin_plate_spline")
wrap.set_source(cage)
wrap.set_target(cage + offset)

print(np.round(wrap.deform(cage), 4))
assert np.allclose(wrap.deform(cage), cage + offset, atol=1e-5)
```

Constructor is `WrapData(kernel="thin_plate_spline", name="wrap_data1")` —
source, target and radii are set with methods, not constructor arguments.

```python
w2 = WrapData(kernel="wendland_c2", name="local_wrap")
w2.set_radius(10.0)                    # kernel support radius
w2.set_source(cage)
w2.set_target(cage + offset)

print(w2.kernel_name, w2.name, w2.radius, w2.geodesic_radius)
assert np.allclose(w2.deform(cage), cage + offset, atol=1e-3)
```

Feed it a `MeshData` and `set_geodesic_radius` masks pairs that are further
apart *across the surface* than the radius — that is how you stop a wrap from
bleeding across a gap that is close in space but far along the mesh.

```python
from cgmath.geometry.mesh import MeshData

cube = MeshData(points=cube_points, indices=cube_indices, counts=cube_counts)

w3   = WrapData(kernel="wendland_c2")
w3.set_geodesic_radius(3.0)
w3.set_source(cube)                    # MeshData -> topology is captured
w3.set_target(cube.points)

out = w3.deform(cube.points)
assert np.allclose(out, cube.points, atol=1e-4)
print("geodesic wrap identity ok")
```

Pairing a *non*-compact kernel with a geodesic radius warns, because masking
distances to infinity ill-conditions a global kernel:

```python
import warnings

w4 = WrapData(kernel="thin_plate_spline")
w4.set_geodesic_radius(2.0)
w4.set_source(cage)
w4.set_target(cage)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    w4.deform(cage)
    print([str(c.message)[:48] for c in caught])
```

`WrapData` round-trips through `.npz`:

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "wrap.npz")
w3.save(path)

loaded = WrapData.load(path)
print(loaded.kernel_name, loaded.geodesic_radius)
assert np.allclose(loaded.deform(cube.points), out)
```

---

## Numba notes

- Every kernel is `@njit(parallel=True, fastmath=True, cache=True)` and rows are
  distributed with `prange`. Big matrices win; tiny ones pay thread-dispatch
  overhead and are slower than plain numpy.
- `fastmath=True` means results can differ in the last few ULP between runs and
  platforms. Compare with `np.allclose`, never `==`.
- The LU kernels are `@njit(fastmath=True, cache=True)` — serial, not parallel.
- First call to any kernel pays a one-off compile (order of a second). Compiled
  artefacts are cached on disk next to the sources (`__pycache__/*.nbi`, `*.nbc`);
  nothing in `cgmath` touches numba's global configuration. Set `NUMBA_CACHE_DIR`
  to move the cache.
- Warm the cache before timing anything:

```python
import time

_   = gaussian(np.zeros((2, 2)))                  # already compiled by now
t0  = time.perf_counter()
big = gaussian(rng.random((600, 600)))
print("600x600 gaussian:", round(time.perf_counter() - t0, 4), "s", big.shape)
```
