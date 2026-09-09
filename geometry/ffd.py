"""DEPRECATED: this module moved to :mod:`cgmath.geometry.deform.ffd`.

Update your imports::

    # before
    from cgmath.geometry.ffd import FFDData

    # after
    from cgmath.geometry.deform import FFDData
    # or, if you want the submodule explicitly:
    from cgmath.geometry.deform.ffd import FFDData

This shim re-exports the moved names for one release.  It will be removed
in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cgmath.geometry.ffd has moved to cgmath.geometry.deform.ffd; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from cgmath.geometry.deform.ffd import *
from cgmath.geometry.deform.ffd import FFDData