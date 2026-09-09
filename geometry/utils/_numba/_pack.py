import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _rasterize_faces(bitmap, points, f2v, counts):
    """
    Scanline polygon rasterizer using even-odd fill rule.

    Fills polygon faces into a boolean bitmap. Handles triangles and quads
    (the only polygon types in typical UV meshes).

    Parameters
    ----------
    bitmap : ndarray (H, W), bool
        Output bitmap to fill into (modified in-place).
    points : ndarray (N, 2), float64
        Vertex positions in pixel coordinates.
    f2v : ndarray (F, max_verts), int
        Face-to-vertex index matrix, -1 padded.
    counts : ndarray (F,), int
        Per-face vertex counts.
    """
    h, w = bitmap.shape
    x_buf = np.empty(16, dtype=np.float64)

    for fi in range(f2v.shape[0]):
        nv = counts[fi]
        if nv < 3:
            continue

        # y bounding box
        y_min_f = np.float64(1e30)
        y_max_f = np.float64(-1e30)
        for vi in range(nv):
            py = points[f2v[fi, vi], 1]
            if py < y_min_f:
                y_min_f = py
            if py > y_max_f:
                y_max_f = py

        row_lo = max(int(np.floor(y_min_f)), 0)
        row_hi = min(int(np.floor(y_max_f)), h - 1)

        for row in range(row_lo, row_hi + 1):
            y   = row + 0.5
            n_x = 0

            # find edge-scanline intersections
            for ei in range(nv):
                v0 = f2v[fi, ei]
                v1 = f2v[fi, (ei + 1) % nv]
                y0 = points[v0, 1]
                y1 = points[v1, 1]

                if (y0 <= y and y1 > y) or (y1 <= y and y0 > y):
                    t  = (y - y0) / (y1 - y0)
                    ix = points[v0, 0] + t * (points[v1, 0] - points[v0, 0])
                    if n_x < 16:
                        x_buf[n_x] = ix
                        n_x += 1

            # bubble sort intersections
            for i in range(n_x - 1):
                for j in range(i + 1, n_x):
                    if x_buf[j] < x_buf[i]:
                        tmp = x_buf[i]
                        x_buf[i] = x_buf[j]
                        x_buf[j] = tmp

            # fill between pairs (even-odd rule)
            for i in range(0, n_x - 1, 2):
                col_lo = max(int(np.ceil(x_buf[i])), 0)
                col_hi = min(int(np.floor(x_buf[i + 1])), w - 1)
                for c in range(col_lo, col_hi + 1):
                    bitmap[row, c] = True


@njit(fastmath=True, cache=True)
def _skyline_find_best_position(
    skyline, col_top, col_occupied, bitmap_h, bitmap_w, canvas_width
):
    """
    Find the best (x, y) position for an island on the skyline canvas.

    Uses column profiles so irregular island bottoms nestle into gaps.

    Parameters
    ----------
    skyline : ndarray (canvas_width,), int
        Current skyline height per column.
    col_top : ndarray (bitmap_w,), int
    col_occupied : ndarray (bitmap_w,), bool
    bitmap_h : int
    bitmap_w : int
    canvas_width : int

    Returns
    -------
    best_x : int
    best_y : int
    """
    best_x       = 0
    best_place_y = 0
    best_total_h = 2147483647  # int32 max

    max_x        = canvas_width - bitmap_w
    for x in range(max_x + 1):
        # compute placement y: for each occupied column, the island's top-row
        # in that column must sit above the skyline
        placement_y = 0
        for c in range(bitmap_w):
            if col_occupied[c]:
                # island needs col_top[c] rows above its placement origin
                # to reach the first occupied pixel in this column
                needed_y = skyline[x + c] - col_top[c]
                if needed_y > placement_y:
                    placement_y = needed_y

        # total height if placed here
        total_h = placement_y + bitmap_h
        if total_h < best_total_h:
            best_total_h = total_h
            best_place_y = placement_y
            best_x       = x

    return best_x, best_place_y


@njit(fastmath=True, cache=True)
def _skyline_update(skyline, col_bottom, col_occupied, place_x, place_y, bitmap_w):
    """
    Advance skyline after placing an island.

    Parameters
    ----------
    skyline : ndarray (canvas_width,), int
    col_bottom : ndarray (bitmap_w,), int
    col_occupied : ndarray (bitmap_w,), bool
    place_x : int
    place_y : int
    bitmap_w : int
    """
    for c in range(bitmap_w):
        if col_occupied[c]:
            new_h = place_y + col_bottom[c] + 1
            if new_h > skyline[place_x + c]:
                skyline[place_x + c] = new_h