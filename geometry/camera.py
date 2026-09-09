"""DEPRECATED: this module moved to :mod:`cgmath.render.camera`.

Update your imports::

    # before
    from cgmath.geometry.camera import look_at

    # after
    from cgmath.render import look_at
    # or, if you want the submodule explicitly:
    from cgmath.render.camera import look_at

This shim re-exports the moved names for one release.  It will be removed
in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cgmath.geometry.camera has moved to cgmath.render.camera; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from cgmath.render.camera import *
from cgmath.render.camera import (
    _af_center,
    _af_shift,
    _af_slide,
    _autofit_camera,
    _autofit_camera_for_points,
    _default_camera,
    _default_camera_for_points,
    _derive_ortho_height,
    look_at,
)