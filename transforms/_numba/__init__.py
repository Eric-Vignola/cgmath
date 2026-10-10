"""Numba kernels backing the public functions in :mod:`cgmath.transforms.main`.

Every kernel is compiled with ``cache=True``, so Numba writes its compiled
artefacts -- ``.nbi`` index and ``.nbc`` object files -- next to the sources
in ``__pycache__/``, alongside the ``.pyc`` files, and the repository ignores
all three.

This package deliberately does **not** touch Numba's global configuration.
``numba.config`` is a single object shared by every consumer in the process,
so assigning ``config.CACHE_DIR`` here would silently relocate the cache of
every other Numba package too, import-order dependent. Set the
``NUMBA_CACHE_DIR`` environment variable if you want the artefacts elsewhere;
Numba also falls back to a user-wide cache on its own when the source tree is
not writable.

Small inputs run on one thread
------------------------------
Every kernel is ``parallel=True``: its loop over the batch is shared out
between Numba's threads, and waking them costs every call whatever its
size: 15 to 65 us on a quiet 24 core machine, against well under a
microsecond for one element on one thread, and 10 to 145 ms once other
processes keep every core busy, as the call waits for its last thread to
be given the CPU. A loop of calls on one matrix or one euler at a time
spent its time waking threads. :func:`sized` hands the public functions
each kernel's single-threaded twin below :data:`SINGLE_THREAD_BELOW`
elements, and the kernel itself from there on.

The twin is the kernel's own body compiled with ``parallel=False``, under
the error model Numba gives a parallel loop's body (``"numpy"``: a
division by zero is an inf or a nan, not a ``ZeroDivisionError``). Its
answers are the kernel's bit for bit -- every value, signed zeros
included, and a nan wherever the kernel has one -- which is why no kernel
here uses fast math: with it, LLVM reassociates sums and fuses
multiply-adds one way in a threaded loop and another in a plain one, and
14 of the 52 public functions answered differently on one thread, on 3 to
36% of their values; 15 once the kernels came from Numba's cache.
"""

import types

SINGLE_THREAD_BELOW = 1000
"""Element count from which a kernel runs on every thread.

Every kernel against its twin from 1 to 65536 elements, on 24 cores. On
a quiet machine the threaded kernel wins from 256 to 1000 elements for the
euler and axis angle conversions (sines, cosines, arc tangents) and the
vector slerp, 1024 to 4096 for the quaternion slerps, logs and exps, the
arc quaternion and the aim matrix, 4096 to 65536 for the 4x4 products,
inverses and builds, and not below 65536 for an add, a negate or a
quaternion product; with every core busy the twin wins at every size,
for every kernel. Summed over every kernel and size, the time lost
against the faster of the two is least for a threshold from 1000 to
2048. Below 1000 a twin is at most 75 us behind its kernel on a quiet
machine (matrix_to_euler at 999), and on one thread it keeps up with
its kernel; from 1000 on, a call runs the threaded kernel, as every call
did before.
"""

_TWINS = {}


def sized(kernel, count):
    """The kernel to run on ``count`` elements.

    Args:
        kernel: a ``parallel=True`` kernel from this package.
        count: how many elements the call runs on.

    Returns:
        ``kernel`` itself for :data:`SINGLE_THREAD_BELOW` elements or more,
        else its single-threaded twin, compiled on first use.
    """
    if count >= SINGLE_THREAD_BELOW:
        return kernel

    twin = _TWINS.get(kernel)
    if twin is None:
        twin = _TWINS[kernel] = _single_threaded(kernel)
    return twin


def _single_threaded(kernel):
    """``kernel``'s body, compiled with ``parallel=False`` and cached.

    Numba keys a cache file on the function's module, qualified name and
    first line, never on its compile options, so compiling the kernel's own
    function a second time would share, and overwrite, the kernel's cache.
    The twin is a copy of the function under a name of its own. Its other
    options are the kernel's, read off the dispatcher, so the decorator in
    the kernel's file stays the one place they are set, and editing them
    there invalidates both caches.

    Under ``NUMBA_DISABLE_JIT=1`` (stepping through a kernel in a debugger,
    measuring coverage) ``njit`` hands back the python function itself, its
    ``prange`` a ``range``: it already runs on one thread, its own twin.
    """
    from numba import njit

    if isinstance(kernel, types.FunctionType):
        return kernel

    func = kernel.py_func
    name = func.__name__ + "_single_threaded"
    copy = types.FunctionType(
        func.__code__, func.__globals__, name, func.__defaults__, func.__closure__
    )
    copy.__qualname__ = func.__qualname__ + "_single_threaded"
    copy.__module__   = func.__module__
    copy.__doc__      = func.__doc__

    options = {
        key: value
        for key, value in kernel.targetoptions.items()
        if key not in ("parallel", "nopython")
    }
    return njit(error_model="numpy", cache=True, **options)(copy)
