"""
Optimized Numba implementations for sprite rasterization operations.

These functions provide parallel, JIT-compiled sprite compositing for
rendering 2D elements onto image buffers with overlap detection.
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
#                     Original (kept for compatibility)                       #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, parallel=False, cache=True)
def _composite_sprite(sprite, buffer, mask, offset, color):
    """
    Composites a sprite to a buffer and computes the overlap to a mask.

    Original sequential implementation kept for compatibility and small sprites.

    Parameters
    ----------
    sprite : np.ndarray
        Grayscale sprite image (H, W), uint8, values 0-255.
    buffer : np.ndarray
        RGB destination buffer (H, W, 3), uint8.
    mask : np.ndarray
        Existing mask to check for overlaps (H, W, 3), uint8.
    offset : np.ndarray
        (x, y) position to place sprite, int32.
    color : np.ndarray
        RGB color tint (3,), uint8.

    Returns
    -------
    tuple
        (count, overlap, ratio) - pixel count, overlap count, overlap ratio.
    """
    count   = 0
    overlap = 0
    ratio   = 0.0

    for x in range(sprite.shape[1]):
        xx = offset[0] + x

        if xx >= 0 and xx < buffer.shape[1]:
            for y in range(sprite.shape[0]):
                yy = offset[1] + y

                if yy >= 0 and yy < buffer.shape[0]:
                    if sprite[y, x] > 0:
                        count += 1

                        # blend in the color
                        w = float(sprite[y, x]) / 255.0
                        c = 0.0

                        for i in range(3):
                            c = float(color[i]) - float(buffer[yy, xx, i])
                            c = np.round(c * w)
                            c = c + float(buffer[yy, xx, i])
                            buffer[yy, xx, i] = c

                        if np.sum(mask[yy, xx]) > 0:
                            overlap += 1

    if count > 0:
        ratio = overlap / count

    return count, overlap, ratio


# --------------------------------------------------------------------------- #
#                     Optimized parallel implementation                       #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, parallel=True, cache=True)
def _composite_sprite_parallel(sprite, buffer, mask, offset, color):
    """
    Parallelized sprite compositing with overlap detection.

    Optimizations:
    - Parallel execution across sprite rows (prange)
    - Removed np.sum() in inner loop (direct channel comparison)
    - Cached float conversions
    - Early bounds checking per row
    - Per-row accumulation with final reduction

    Parameters
    ----------
    sprite : np.ndarray
        Grayscale sprite image (H, W), uint8, values 0-255.
    buffer : np.ndarray
        RGB destination buffer (H, W, 3), uint8.
    mask : np.ndarray
        Existing mask to check for overlaps (H, W, 3), uint8.
    offset : np.ndarray
        (x, y) position to place sprite, int32.
    color : np.ndarray
        RGB color tint (3,), uint8.

    Returns
    -------
    tuple
        (count, overlap, ratio) - pixel count, overlap count, overlap ratio.
    """
    sprite_h = sprite.shape[0]
    sprite_w = sprite.shape[1]
    buf_h    = buffer.shape[0]
    buf_w    = buffer.shape[1]
    off_x    = offset[0]
    off_y    = offset[1]

    # Pre-compute color as float for blending
    color_r = float(color[0])
    color_g = float(color[1])
    color_b = float(color[2])

    # Per-row accumulators for parallel reduction
    row_counts   = np.zeros(sprite_h, dtype=np.int32)
    row_overlaps = np.zeros(sprite_h, dtype=np.int32)

    # Parallel over sprite rows
    for y in prange(sprite_h):
        yy = off_y + y

        # Skip entire row if out of bounds
        if yy < 0 or yy >= buf_h:
            continue

        local_count   = 0
        local_overlap = 0

        for x in range(sprite_w):
            xx = off_x + x

            # Bounds check
            if xx < 0 or xx >= buf_w:
                continue

            sprite_val = sprite[y, x]
            if sprite_val > 0:
                local_count += 1

                # Blend weight from sprite intensity
                w = float(sprite_val) * (1.0 / 255.0)

                # Blend RGB channels (inlined for performance)
                buf_r = float(buffer[yy, xx, 0])
                buf_g = float(buffer[yy, xx, 1])
                buf_b = float(buffer[yy, xx, 2])

                # Alpha blend: result = buffer + (color - buffer) * weight
                new_r = buf_r + (color_r - buf_r) * w
                new_g = buf_g + (color_g - buf_g) * w
                new_b = buf_b + (color_b - buf_b) * w

                # Round and store (use +0.5 trick for rounding)
                buffer[yy, xx, 0] = np.uint8(new_r + 0.5)
                buffer[yy, xx, 1] = np.uint8(new_g + 0.5)
                buffer[yy, xx, 2] = np.uint8(new_b + 0.5)

                # Check mask overlap - avoid np.sum(), check channels directly
                if mask[yy, xx, 0] > 0 or mask[yy, xx, 1] > 0 or mask[yy, xx, 2] > 0:
                    local_overlap += 1

        row_counts[y] = local_count
        row_overlaps[y] = local_overlap

    # Reduce per-row counts to totals
    count   = 0
    overlap = 0
    for y in range(sprite_h):
        count += row_counts[y]
        overlap += row_overlaps[y]

    ratio = 0.0
    if count > 0:
        ratio = float(overlap) / float(count)

    return count, overlap, ratio


@njit(fastmath=True, parallel=True, cache=True)
def _composite_sprite_aa(sprite, sprite_buffer):
    """
    Composites an antialiased sprite to a grayscale sprite buffer.

    Each pixel blends toward white (255) based on sprite intensity.

    Parameters
    ----------
    sprite : np.ndarray
        Grayscale sprite (H, W), uint8, values 0-255.
    sprite_buffer : np.ndarray
        Grayscale destination buffer (H, W), uint8. Modified in-place.
    """
    for y in prange(sprite.shape[0]):
        for x in range(sprite.shape[1]):
            if sprite[y, x] > 0:
                w = float(sprite[y, x]) / 255.0
                c = 255.0 - float(sprite_buffer[y, x])
                c = np.round(c * w)
                c = c + float(sprite_buffer[y, x])
                sprite_buffer[y, x] = c


@njit(fastmath=True, parallel=True, cache=True)
def _composite_sprite_aa_optimized(sprite, sprite_buffer):
    """
    Optimized antialiased sprite compositing.

    Same as _composite_sprite_aa but with micro-optimizations:
    - Multiplication instead of division
    - Inline rounding with +0.5 trick

    Parameters
    ----------
    sprite : np.ndarray
        Grayscale sprite (H, W), uint8, values 0-255.
    sprite_buffer : np.ndarray
        Grayscale destination buffer (H, W), uint8. Modified in-place.
    """
    inv_255 = 1.0 / 255.0

    for y in prange(sprite.shape[0]):
        for x in range(sprite.shape[1]):
            sprite_val = sprite[y, x]
            if sprite_val > 0:
                buf_val = float(sprite_buffer[y, x])
                w       = float(sprite_val) * inv_255
                # Blend toward white: result = buffer + (255 - buffer) * weight
                new_val = buf_val + (255.0 - buf_val) * w
                sprite_buffer[y, x] = np.uint8(new_val + 0.5)


# --------------------------------------------------------------------------- #
#                          Batch operations                                   #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, parallel=True, cache=True)
def _composite_sprites_batch(
    sprites,
    sprite_offsets,
    sprite_sizes,
    buffer,
    mask,
    colors,
):
    """
    Batch composite multiple sprites in parallel.

    Parameters
    ----------
    sprites : np.ndarray
        Flattened array of all sprite pixels.
    sprite_offsets : np.ndarray
        Start offset in sprites array for each sprite, shape (N,), int32.
    sprite_sizes : np.ndarray
        (width, height) for each sprite, shape (N, 2), int32.
    buffer : np.ndarray
        RGB destination buffer (H, W, 3), uint8.
    mask : np.ndarray
        Existing mask (H, W, 3), uint8.
    colors : np.ndarray
        RGB color for each sprite, shape (N, 3), uint8.

    Returns
    -------
    tuple
        (counts, overlaps, ratios) - arrays of shape (N,) for each sprite.
    """
    n_sprites = sprite_offsets.shape[0]
    buf_h     = buffer.shape[0]
    buf_w     = buffer.shape[1]

    counts    = np.zeros(n_sprites, dtype=np.int32)
    overlaps  = np.zeros(n_sprites, dtype=np.int32)
    ratios    = np.zeros(n_sprites, dtype=np.float32)

    # Process sprites in parallel
    for s in prange(n_sprites):
        sprite_w      = sprite_sizes[s, 0]
        sprite_h      = sprite_sizes[s, 1]
        sprite_start  = sprite_offsets[s]
        off_x         = sprite_sizes[s, 0]  # Assuming offset stored elsewhere
        off_y         = sprite_sizes[s, 1]

        color_r       = float(colors[s, 0])
        color_g       = float(colors[s, 1])
        color_b       = float(colors[s, 2])

        local_count   = 0
        local_overlap = 0

        for y in range(sprite_h):
            yy = off_y + y
            if yy < 0 or yy >= buf_h:
                continue

            for x in range(sprite_w):
                xx = off_x + x
                if xx < 0 or xx >= buf_w:
                    continue

                # Get sprite value from flattened array
                sprite_idx = sprite_start + y * sprite_w + x
                sprite_val = sprites[sprite_idx]

                if sprite_val > 0:
                    local_count += 1
                    w     = float(sprite_val) / 255.0

                    buf_r = float(buffer[yy, xx, 0])
                    buf_g = float(buffer[yy, xx, 1])
                    buf_b = float(buffer[yy, xx, 2])

                    buffer[yy, xx, 0] = np.uint8(buf_r + (color_r - buf_r) * w + 0.5)
                    buffer[yy, xx, 1] = np.uint8(buf_g + (color_g - buf_g) * w + 0.5)
                    buffer[yy, xx, 2] = np.uint8(buf_b + (color_b - buf_b) * w + 0.5)

                    if (
                        mask[yy, xx, 0] > 0
                        or mask[yy, xx, 1] > 0
                        or mask[yy, xx, 2] > 0
                    ):
                        local_overlap += 1

        counts[s] = local_count
        overlaps[s] = local_overlap
        if local_count > 0:
            ratios[s] = float(local_overlap) / float(local_count)

    return counts, overlaps, ratios


# --------------------------------------------------------------------------- #
#                          Fill operations                                    #
# --------------------------------------------------------------------------- #


@njit(fastmath=True, parallel=True, cache=True)
def _fill_rect(buffer, x0, y0, x1, y1, color):
    """
    Fill a rectangular region with a solid color.

    Parameters
    ----------
    buffer : np.ndarray
        RGB buffer (H, W, 3), uint8.
    x0, y0 : int
        Top-left corner.
    x1, y1 : int
        Bottom-right corner (exclusive).
    color : np.ndarray
        RGB color (3,), uint8.
    """
    buf_h = buffer.shape[0]
    buf_w = buffer.shape[1]

    # Clamp to buffer bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(buf_w, x1)
    y1 = min(buf_h, y1)

    for y in prange(y0, y1):
        for x in range(x0, x1):
            buffer[y, x, 0] = color[0]
            buffer[y, x, 1] = color[1]
            buffer[y, x, 2] = color[2]


@njit(fastmath=True, parallel=True, cache=True)
def _fill_rect_alpha(buffer, x0, y0, x1, y1, color, alpha):
    """
    Fill a rectangular region with alpha blending.

    Parameters
    ----------
    buffer : np.ndarray
        RGB buffer (H, W, 3), uint8.
    x0, y0 : int
        Top-left corner.
    x1, y1 : int
        Bottom-right corner (exclusive).
    color : np.ndarray
        RGB color (3,), uint8.
    alpha : float
        Blend factor 0.0-1.0.
    """
    buf_h     = buffer.shape[0]
    buf_w     = buffer.shape[1]

    x0        = max(0, x0)
    y0        = max(0, y0)
    x1        = min(buf_w, x1)
    y1        = min(buf_h, y1)

    color_r   = float(color[0])
    color_g   = float(color[1])
    color_b   = float(color[2])
    inv_alpha = 1.0 - alpha

    for y in prange(y0, y1):
        for x in range(x0, x1):
            buf_r = float(buffer[y, x, 0])
            buf_g = float(buffer[y, x, 1])
            buf_b = float(buffer[y, x, 2])

            buffer[y, x, 0] = np.uint8(buf_r * inv_alpha + color_r * alpha + 0.5)
            buffer[y, x, 1] = np.uint8(buf_g * inv_alpha + color_g * alpha + 0.5)
            buffer[y, x, 2] = np.uint8(buf_b * inv_alpha + color_b * alpha + 0.5)


@njit(fastmath=True, parallel=True, cache=True)
def _clear_buffer(buffer, color):
    """
    Clear entire buffer to a solid color.

    Parameters
    ----------
    buffer : np.ndarray
        RGB buffer (H, W, 3), uint8.
    color : np.ndarray
        RGB color (3,), uint8.
    """
    for y in prange(buffer.shape[0]):
        for x in range(buffer.shape[1]):
            buffer[y, x, 0] = color[0]
            buffer[y, x, 1] = color[1]
            buffer[y, x, 2] = color[2]