"""DEPRECATED: this module moved to :mod:`cgmath.geometry.deform.delta_mush`.

Update your imports::

    # before
    from cgmath.geometry.delta_mush import DeltaMushData

    # after
    from cgmath.geometry import DeltaMushData
    # or, if you want the submodule explicitly:
    from cgmath.geometry.deform.delta_mush import DeltaMushData

This shim re-exports the moved names for one release.  It will be removed
in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cgmath.geometry.delta_mush has moved to cgmath.geometry.deform.delta_mush; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from cgmath.geometry.deform.delta_mush import *
from cgmath.geometry.deform.delta_mush import DeltaMushData