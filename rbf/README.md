# `cgmath.rbf` — the kernel shelf

Twenty-one radial basis function kernels, numba-compiled and parallelised, plus
a small JIT LU solver to put behind them.

This is a **primitives** package. It has no `RBFInterpolator` class, no fitting
object, no state. You bring a distance matrix, a kernel turns it into a system
matrix, you solve it. If you want the batteries-included version for geometry,
use [`geometry.deform.WrapData`](../geometry/deform/wrap.py) — it is built on
this and is the only in-library consumer.

```python
from cgmath.rbf._kernels import gaussian, get_kernel

D = [[0.0, 1.0], [1.0, 0.0]]
gaussian(D, epsilon=2.0)          # (2, 2) float64
get_kernel("wendland_c2")(D, r=1.5)
```

---

## The `__init__.py` exports nothing, on purpose

`cgmath.rbf` exports nothing. `WrapData` used to live here; it now lives in
`cgmath.geometry.deform`, and that module imports `rbf._kernels` **at module
scope**. Re-exporting `WrapData` from this `__init__` would import back into a
partially-initialized module — a circular import. So the package docstring says
so and the file stops there.

Consequence: always import from the submodule.

```python
from cgmath.rbf._kernels import thin_plate_spline    # yes

try:
    from cgmath.rbf import thin_plate_spline         # no
except ImportError as err:
    print(err)
```

The leading underscore in `_kernels` names the file, not the visibility. It is
the public surface of this package.

---

## Files

```
rbf/
├── __init__.py      a docstring, by design (see above)
├── _kernels.py      21 public kernels, KERNEL_REGISTRY, get_kernel,
│                    list_kernels, and a module __getattr__ that re-exposes
│                    lu_solve / lu_solve_factored
└── _numba/
    ├── __init__.py  documents the on-disk cache policy (NUMBA_CACHE_DIR)
    ├── _kernels.py  the @njit(parallel=True, fastmath=True, cache=True) bodies
    └── _lu.py       lu_solve, lu_solve_factored + private LU/substitution steps
```

`_kernels.py` is a thin coercion layer: it forces the input to `float64`, the
parameter to `float`, then defers the import of the matching numba kernel and
calls it. The deferred import is what keeps `import cgmath` cheap — numba only
compiles a kernel the first time you actually use it.

---

## Conventions a caller must know

**Kernels take a distance *matrix*, not a radius scalar.** Input is `(n, m)`,
output is `(n, m)` `float64`. A 1-D array raises a numba `TypingError`.

**The second argument is one of two things**, and which one depends on the
kernel:

| Second arg | Kernels | Meaning |
|---|---|---|
| `epsilon=1.0` | the 14 globally-supported kernels | shape parameter |
| `r=1.0` | `wendland_c0/c2/c4/c6`, `wu_c2/c4`, `bump` | support radius — everything at or beyond `r` is exactly `0` |

**`epsilon` is not consistent.** Four kernels multiply by it, ten divide by it:

| Behaviour | Kernels | Bigger `epsilon` means |
|---|---|---|
| `eps * r` | `gaussian`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic` | **narrower** |
| `r / eps` | `linear`, `cubic`, `quintic`, `thin_plate_spline`, `polyharmonic`, `matern_12/32/52`, `cauchy`, `log_kernel` | **wider** |

This is the single most common way to get an RBF fit backwards. Check the table
before you tune.

**Not every kernel is 1 at zero distance.** `wendland_c4` peaks at `3`, `bump`
peaks at `exp(-1)`, `thin_plate_spline` is `0` at zero and *negative* below
`r = 1`. All are correct; the constants cancel in a solve.

**Some kernels need polynomial augmentation.** `thin_plate_spline`, `linear`,
`cubic`, `quintic`, `multiquadric` and `log_kernel` are only conditionally
positive definite — border the system with `[1, x, y, z]` or the solve is not
unique. `gaussian`, `inverse_multiquadric`, `inverse_quadratic` and the compact
kernels are strictly positive definite and need nothing.

**One name/function mismatch.** `log_kernel` is registered as `"log"`.

**`fastmath=True` everywhere**, so results move in the last few ULP across runs
and platforms. Compare with `np.allclose`, never `==`.

---

## The kernels

| Group | Names |
|---|---|
| Infinitely smooth | `gaussian`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic` |
| Polyharmonic | `linear`, `cubic`, `quintic`, `thin_plate_spline`, `polyharmonic(X, k, epsilon)` |
| Compactly supported | `wendland_c0`, `wendland_c2`, `wendland_c4`, `wendland_c6`, `wu_c2`, `wu_c4`, `bump` |
| Specialized | `matern_12`, `matern_32`, `matern_52`, `cauchy`, `log_kernel` |

Discovery is `list_kernels()` (21 names), `KERNEL_REGISTRY` (the dict), and
`get_kernel(name)` (case-insensitive, raises `ValueError` on a miss).

---

## `rbf._numba._lu` — why there is a hand-written LU here

`cgmath.rbf._numba._lu` is Doolittle LU with partial pivoting, written in
numba. `lu_solve(A, B)` and `lu_solve_factored(L, U, perm, B)` are the public
entry points; `_lu_decompose`, `_forward_substitute` and `_backward_substitute`
are the steps.

It exists for two reasons `np.linalg.solve` cannot cover:

1. **It is callable from inside `@njit` code.** Both entry points are raw
   `CPUDispatcher` objects, which is why `_kernels.__getattr__` hands them back
   unwrapped rather than putting a Python shim in front of them.
2. **It gives you the factorization back**, so re-solving with a new
   right-hand side over a fixed set of RBF centres is `O(n^2)` rather than
   `O(n^3)`.

From ordinary Python with a one-shot system, `np.linalg.solve` is still the
better call. Two caveats if you do use `lu_solve`: it **does not raise on a
singular matrix** (pivots under `1e-15` are skipped and the unknown is set to
`0.0`, so you get a finite but meaningless answer — check the residual), and
nothing else in `cgmath` currently calls it. `WrapData` solves its own
system with `np.linalg.pinv`.

---

## Where the distance function is

There isn't one in this package. The numba pairwise distance the kernels are
normally fed lives with the wrap deformer:

```python
import numpy as np
from cgmath.geometry.deform.wrap import cdist_euclidean

X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # (m, k)
Y = np.array([[3.0, 4.0, 0.0]])                   # (n, k)
print(cdist_euclidean(X, Y))                         # -> (m, n)
```

It delegates to `cgmath.geometry.utils._numba._wrap._cdist_euclidean` and beats
`scipy.spatial.distance.cdist` under about `10^5` points.

---

## Quick taste

### Scattered-data interpolation in five lines

```python
import numpy as np
from cgmath.geometry.deform.wrap import cdist_euclidean
from cgmath.rbf import _kernels
from cgmath.rbf._kernels import gaussian

rng     = np.random.default_rng(0)
sites   = rng.random((60, 2)) * 2.0 - 1.0
values  = np.sin(2.0 * sites[:, 0]) * np.cos(2.0 * sites[:, 1])

weights = _kernels.lu_solve(gaussian(cdist_euclidean(sites, sites)), values)
query   = np.array([[0.1, -0.2], [0.5, 0.5]])
print(gaussian(cdist_euclidean(query, sites)) @ weights)
```

### A local, sparse system with a compact kernel

```python
from cgmath.rbf._kernels import wendland_c2

K = wendland_c2(cdist_euclidean(sites, sites), r=0.8)
print("sparsity:", round(1.0 - np.count_nonzero(K) / K.size, 3))
```

### Deform geometry — use `WrapData`, not the raw kernels

```python
from cgmath.geometry.deform import WrapData

cage = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

wrap = WrapData(kernel="thin_plate_spline")
wrap.set_source(cage)
wrap.set_target(cage + np.array([1.0, 2.0, 3.0]))
print(wrap.deform(cage))
```

Note the constructor takes only `kernel` and `name` — source, target, radius
and geodesic radius are all set through methods.

---

## Where to go next

| You want to... | Read |
|---|---|
| Every kernel, every argument, copy-paste | [`CHEATSHEET.md`](CHEATSHEET.md) |
| The rest of the library | [`../README.md`](../README.md) |
| The wrap deformer that consumes this | [`../geometry/deform/wrap.py`](../geometry/deform/wrap.py) |
