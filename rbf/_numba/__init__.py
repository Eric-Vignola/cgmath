"""Numba kernels for the RBF solvers.

Every kernel is compiled with ``cache=True``, so Numba writes its ``.nbi``
index and ``.nbc`` object files into ``__pycache__/`` next to the sources.

This package deliberately does **not** touch Numba's global configuration.
``numba.config`` is a single object shared by every consumer in the process,
so assigning ``config.CACHE_DIR`` here would silently relocate the cache of
``transforms`` and every other Numba package too, import-order dependent. Set
the ``NUMBA_CACHE_DIR`` environment variable to move the artefacts; Numba also
falls back to a user-wide cache on its own when the source tree is read-only.
"""
