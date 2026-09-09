"""DEPRECATED: this module moved to :mod:`cgmath.render.raytracer`.

Update your imports::

    # before
    from cgmath.geometry.raytracer import render, look_at

    # after
    from cgmath.render import render, look_at
    # or, if you want the submodule explicitly:
    from cgmath.render.raytracer import render

This shim re-exports the moved names for one release.  It will be removed
in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cgmath.geometry.raytracer has moved to cgmath.render.raytracer; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from cgmath.render.raytracer import *
from cgmath.render.raytracer import (
    _autofit_camera_for_points,
    _default_camera_for_points,
    _derive_ortho_height,
    _load_texture,
    _sample_texture_bilinear,
    Frame,
    look_at,
    render,
)