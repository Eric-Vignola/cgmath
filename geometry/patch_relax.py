"""DEPRECATED: this module moved to :mod:`cgmath.geometry.deform.patch_relax`.

Update your imports::

    # before
    from cgmath.geometry.patch_relax import PatchRelaxData

    # after
    from cgmath.geometry import PatchRelaxData
    # or, if you want the submodule explicitly:
    from cgmath.geometry.deform.patch_relax import PatchRelaxData

This shim re-exports the moved names for one release.  It will be removed
in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cgmath.geometry.patch_relax has moved to cgmath.geometry.deform.patch_relax; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from cgmath.geometry.deform.patch_relax import *
from cgmath.geometry.deform.patch_relax import PatchRelaxData