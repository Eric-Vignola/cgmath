import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _sample_texture_bilinear_fast(
    texture: np.ndarray,
    uvs:     np.ndarray,
    repeat:  bool,
) -> np.ndarray:
    """Bilinear texture lookup at ``(u, v)`` coords."""
    h, w = texture.shape[0], texture.shape[1]
    n   = uvs.shape[0]
    out = np.empty((n, 3), dtype=np.float32)

    for i in prange(n):
        u = uvs[i, 0]
        v = uvs[i, 1]

        if repeat:
            u = u - np.floor(u)
            v = v - np.floor(v)
        else:
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

        x  = u * (w - 1)
        y  = (1.0 - v) * (h - 1)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= w:
            x1 = w - 1
        if y1 >= h:
            y1 = h - 1

        fx = x - np.floor(x)
        fy = y - np.floor(y)

        for c in range(3):
            t00 = texture[y0, x0, c]
            t10 = texture[y0, x1, c]
            t01 = texture[y1, x0, c]
            t11 = texture[y1, x1, c]
            top = t00 * (1.0 - fx) + t10 * fx
            bot = t01 * (1.0 - fx) + t11 * fx
            out[i, c] = top * (1.0 - fy) + bot * fy

    return out