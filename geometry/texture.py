"""DEPRECATED: this module moved to :mod:`cgmath.render.texture`.

Update your imports::

    # before
    from cgmath.geometry.texture import _load_texture

    # after
    from cgmath.render import load_texture
    # or, if you want the submodule explicitly:
    from cgmath.render.texture import _load_texture

This shim re-exports the moved names for one release.  It will be removed
in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cgmath.geometry.texture has moved to cgmath.render.texture; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from cgmath.render.texture import *
from cgmath.render.texture import _load_texture, _sample_texture_bilinear