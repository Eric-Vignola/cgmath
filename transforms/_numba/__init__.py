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
"""
