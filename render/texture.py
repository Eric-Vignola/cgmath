"""
Texture I/O + bilinear sampling helpers.

These are renderer-agnostic so both :mod:`cgmath.render.scene` (for
``Object.get_loaded_texture`` lazy loading) and
:mod:`cgmath.render.raytracer` (for per-hit shading) can import them
without forming a circular dependency on the renderer.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np

# PIL is optional -- only needed when ``_load_texture`` is called with a
# file path.  Match the lazy-import pattern used in
# ``cgmath.geometry.mesh``: import once at module load time and let
# call sites guard with ``if Image is None``.
try:
    from PIL import Image
except ImportError:
    Image = None


# -- Texture I/O -------------------------------------------------------------


def _load_texture(texture: Union[str, np.ndarray]) -> np.ndarray:
    """Returns a contiguous ``(H, W, 3)`` float32 array in [0, 1]."""
    if isinstance(texture, str):
        if Image is None:
            raise RuntimeError(
                "PIL required to load a texture from a path; "
                "pass a numpy array instead, or install Pillow."
            )
        path = os.path.expandvars(os.path.expanduser(texture))
        img  = Image.open(path).convert("RGB")
        arr  = np.asarray(img)
    else:
        arr = np.asarray(texture)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"texture must have 1, 3, or 4 channels; got {arr.shape}")

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)

    return np.ascontiguousarray(arr)


# -- Texture sampling --------------------------------------------------------


def _sample_texture_bilinear(
    texture: np.ndarray,
    uvs:     np.ndarray,
    repeat:  bool,
) -> np.ndarray:
    """Bilinear texture lookup at ``(u, v)`` coords.

    UV convention: ``u`` increases right, ``v`` increases up.
    """
    from cgmath.render._numba._texture import _sample_texture_bilinear_fast

    return _sample_texture_bilinear_fast(texture, uvs, repeat)