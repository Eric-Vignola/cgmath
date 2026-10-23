"""
A frame rendered as bands of whole pixel rows, the bands side by side on a
thread pool: :func:`cgmath.render.raytracer.render`'s path (``threads=``).

Each band runs, on its own rows, the operations the whole-frame path
(``raytracer._render_scene`` and ``_compose_scene_output``) runs on every
row at once, so a banded frame is byte for byte the whole frame: colour,
alpha and depth. What that takes:

* MeshData.raycast's per-mesh work (vertex normals, the BVH, the quad
  geometry, PN-quad control points) is done once a frame, in the calling
  thread, and a band casts its rays straight into the kernel that
  bilinear_raycast / bezier_raycast would call. Only the rays that hit are
  post-processed, with _saddle_surface.raycast's own operations
  (:func:`_cast_hits`).
* numpy's OpenBLAS can crash or deadlock when matmuls run in several
  threads at once (see ``_BLAS``): every matmul, dot or matvec a band makes
  holds one lock (:func:`_matmul`, :func:`_shade_one_light`).
* numpy sends a one-row matmul to gemv / dot and two rows or more to gemm,
  and the two round differently. A band always holds two samples or more;
  a band holding a single hit of a mesh that the frame hits more often
  shades it as two identical rows, and a mesh the whole frame hits once is
  shaded again, in its band, as one row (:func:`render_bands`).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional, Tuple

import numba
import numpy as np
from cgmath.geometry._saddle_surface import _quad_geometry
from cgmath.geometry.mesh import SampleMethod, _require_no_ngons
from cgmath.render import raytracer as _rt
from cgmath.render.frame import Frame

# A frame of fewer samples than this, or one sample wide, is one band.
_ONE_BAND_BELOW = 16384

# Otherwise it splits into about two bands a thread, each of _MIN_BAND to
# _MAX_BAND samples in whole rows (one row, when a row holds more).
_MIN_BAND = 8192
_MAX_BAND = 65536

# Every BLAS call a band makes holds this lock. numpy 2.4.6's OpenBLAS
# (scipy-openblas 0.3.31, Windows), 24 threads doing matmuls at once, with
# no cgmath: an access violation, then a deadlock; clean with
# OPENBLAS_NUM_THREADS=1 (2026-10-23).
_BLAS = threading.Lock()

_POOL      = None
_POOL_LOCK = threading.Lock()

# Tests: called with each banded frame's bookkeeping (see render_bands).
_stats_hook = None


# -- threads --------------------------------------------------------------------


def check_threads(threads) -> None:
    """Raises ValueError unless *threads* is None or a positive int."""
    if threads is None:
        return
    if (
        isinstance(threads, (bool, np.bool_))
        or not isinstance(threads, (int, np.integer))
        or threads < 1
    ):
        raise ValueError(f"threads must be None or a positive int, got {threads!r}")


def thread_budget(threads: Optional[int]) -> int:
    """The CPU threads one render may use: bands side by side times the numba
    threads inside each. None: numba's thread count in the calling thread."""
    if threads is None:
        return numba.get_num_threads()
    return max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS, os.cpu_count() or 1))


def _get_pool() -> ThreadPoolExecutor:
    """The process's render pool, made on the first threaded render."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(
                max_workers=os.cpu_count() or 1, thread_name_prefix="cgmath-render"
            )
        return _POOL


def _after_fork_in_child() -> None:
    """A forked child has none of the pool's threads, and a lock another
    thread held stays held: it starts both afresh."""
    global _BLAS, _POOL, _POOL_LOCK
    _BLAS      = threading.Lock()
    _POOL      = None
    _POOL_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):  # not on Windows
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _threading_layer() -> Optional[str]:
    """numba's threading layer, or None before it is launched."""
    try:
        return numba.threading_layer()
    except ValueError:
        return None


# -- BLAS under the lock ----------------------------------------------------------


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` under the BLAS lock.

    errstate: silence spurious BLAS warnings on Apple Accelerate (arm64), as
    the whole-frame path does.
    """
    with _BLAS, np.errstate(all="ignore"):
        return a @ b


def _shade_one_light(hit_points: np.ndarray, N: np.ndarray, light: dict) -> np.ndarray:
    """raytracer._shade_one_light; an infinite light's under the BLAS lock
    (``N @ L``, and the norm of its direction, a dot)."""
    if light.get("kind", "point") == "infinite":
        with _BLAS:
            return _rt._shade_one_light(hit_points, N, light)
    return _rt._shade_one_light(hit_points, N, light)


def _shade_lights(
    base_color: np.ndarray,
    hit_points: np.ndarray,
    normals:    np.ndarray,
    lights:     List[dict],
    ambient:    float,
) -> np.ndarray:
    """raytracer._shade_lights, through :func:`_shade_one_light`."""
    n      = hit_points.shape[0]
    accum  = np.zeros((n, 3), dtype=np.float64)

    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norm = np.maximum(n_norm, 1e-12)
    N      = normals / n_norm

    for light in lights:
        accum += _shade_one_light(hit_points, N, light)

    out = base_color * (ambient + accum.astype(np.float32))
    return out.astype(np.float32)


# -- per-mesh work, once a frame ----------------------------------------------------


class _Prepared:
    """One scene entry, with what MeshData.raycast and _shade_mesh_per_sample
    work out per mesh worked out once."""

    __slots__ = (
        "entry", "is_world", "R3", "T3", "N3", "normals", "bvh", "geometry",
        "points", "cps", "aabb",
    )


def _prepare(entry: dict) -> _Prepared:
    """MeshData.raycast's per-mesh steps, minus the rays: the same calls on
    the same mesh, so the same arrays."""
    from cgmath.geometry.utils._numba import _bilinear as _kernels

    mesh = entry["mesh"]
    M    = entry["world_matrix"]
    prep = _Prepared()

    prep.entry    = entry
    prep.is_world = np.allclose(M, np.eye(4))
    prep.R3       = prep.T3 = prep.N3 = None
    if not prep.is_world:
        with _BLAS:
            Minv    = np.linalg.inv(M)
            prep.N3 = np.linalg.inv(M[:3, :3]).T
        prep.R3 = Minv[:3, :3]
        prep.T3 = Minv[:3, 3]

    _require_no_ngons("raycast", mesh.counts)
    method = SampleMethod(entry["sample_method"])
    if mesh.points.shape[1] > 2:
        normals = mesh.get_vertex_normals(angle_weighted=True, area_weighted=False)
    else:
        normals = None
    bezier = method == SampleMethod.BEZIER and normals is not None

    prep.normals  = normals
    prep.bvh      = mesh.bvh_bezier if bezier else mesh.bvh_bilinear
    prep.geometry = _quad_geometry(mesh.geometry)
    prep.points   = np.ascontiguousarray(mesh.points, dtype=np.float64)
    prep.cps      = prep.aabb = None
    if bezier:
        prep.cps = _kernels._compute_all_pn_quad_cps(
            prep.geometry, prep.points, np.ascontiguousarray(normals, dtype=np.float64)
        )
        if prep.bvh is None:
            prep.aabb = _kernels._compute_bezier_face_aabbs(prep.cps)
    elif prep.bvh is None:
        prep.aabb = _kernels._compute_face_aabbs(prep.geometry, prep.points)

    # built lazily: built here, before the bands read it
    if entry["uv"] is not None:
        entry["uv"].geometry
    return prep


def _cast_hits(prep: _Prepared, origins: np.ndarray, directions: np.ndarray):
    """_saddle_surface.raycast (forward only) for the rays that hit.

    The kernel bilinear_raycast / bezier_raycast would call, on the arrays
    :func:`_prepare` made, then raycast's post-processing on the hit rows
    alone: the same operations on the same rows, so the same bits.

    Returns:
        ``(hit, None)`` when nothing is hit, else ``(hit, (indices,
        weights, projections, normals, distances))``, one row per hit.
    """
    from cgmath.geometry.utils._numba import _bilinear as _kernels

    o   = np.ascontiguousarray(origins, dtype=np.float64)
    d   = np.ascontiguousarray(directions, dtype=np.float64)
    dn  = d / np.linalg.norm(d, axis=1, keepdims=True)
    bvh = prep.bvh
    tw  = prep.entry["twosided"]
    if prep.cps is not None:
        if bvh is not None:
            t, uv, fi = _kernels._intersect_bezier_mesh_parallel_bvh(
                prep.cps,
                bvh.node_aabb_min,
                bvh.node_aabb_max,
                bvh.node_left,
                bvh.node_right,
                bvh.node_first,
                bvh.node_count,
                bvh.face_perm,
                o,
                dn,
                tw,
            )
        else:
            t, uv, fi = _kernels._intersect_bezier_mesh_parallel(
                prep.cps, prep.aabb[0], prep.aabb[1], o, dn, tw
            )
    elif bvh is not None:
        t, uv, fi = _kernels._intersect_mesh_parallel_bvh(
            prep.geometry,
            prep.points,
            bvh.node_aabb_min,
            bvh.node_aabb_max,
            bvh.node_left,
            bvh.node_right,
            bvh.node_first,
            bvh.node_count,
            bvh.face_perm,
            o,
            dn,
            tw,
        )
    else:
        t, uv, fi = _kernels._intersect_mesh_parallel(
            prep.geometry, prep.points, prep.aabb[0], prep.aabb[1], o, dn, tw
        )

    hit = fi >= 0
    if not np.any(hit):
        return hit, None

    # bilinear weights from the hits' uv, triangles folding w3 into w2
    indices       = fi[hit]
    uv_hit        = uv[hit]
    u             = uv_hit[:, 0]
    v             = uv_hit[:, 1]
    ou            = 1.0 - u
    ov            = 1.0 - v
    weights       = np.empty((indices.shape[0], 4), dtype=np.float64)
    weights[:, 0] = ou * ov
    weights[:, 1] = u * ov
    weights[:, 2] = u * v
    weights[:, 3] = ou * v

    hit_geometry  = prep.geometry[indices]
    folded        = np.flatnonzero(hit_geometry[:, 3] == -1)
    weights[folded, 2] += weights[folded, 3]
    weights[folded, 3] = 0.0

    # the point hit, along the unit ray
    distances = t[hit]
    hit_dirs  = directions[hit]
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = hit_dirs / np.linalg.norm(hit_dirs, axis=1, keepdims=True)
    projections = origins[hit] + distances[:, None] * unit

    if prep.normals is not None:
        result = prep.normals[hit_geometry] * weights[:, :, None]
        result[hit_geometry < 0] = 0.0
        normals = np.sum(result, axis=1)
        mag     = np.einsum("...i,...i", normals, normals) ** 0.5
        mag[mag == 0] = 1.0
        normals /= mag[:, None]
    else:
        normals = np.zeros_like(projections)

    return hit, (indices, weights, projections, normals, distances)


# -- one band -------------------------------------------------------------------------


class _Job:
    """One banded frame: what every band reads, and the buffers it writes."""

    __slots__ = (
        "preps", "camera_matrix", "is_orthographic", "ortho_height", "angle_of_view",
        "width", "height", "n_axis", "lights", "background", "bands", "image", "depth",
        "idents", "numpy_err",
    )


def _band_rays(job: _Job, r0: int, r1: int) -> Tuple[np.ndarray, np.ndarray]:
    """raytracer._generate_camera_rays (or _orthographic) for rows r0..r1-1."""
    width, height, n_axis = job.width, job.height, job.n_axis
    camera_matrix = job.camera_matrix

    aspect = width / height
    if job.is_orthographic:
        half_h = 0.5 * job.ortho_height
    else:
        fov_y  = np.deg2rad(job.angle_of_view)
        half_h = np.tan(fov_y * 0.5)
    half_w = half_h * aspect

    sub = (np.arange(n_axis, dtype=np.float64) + 0.5) / n_axis
    sub_x, sub_y = np.meshgrid(sub, sub, indexing="xy")
    sub_x = sub_x.ravel()
    sub_y = sub_y.ravel()

    cols  = np.arange(width, dtype=np.float64)
    rows  = np.arange(r0, r1, dtype=np.float64)

    px    = (cols[None, :, None] + sub_x[None, None, :]) / width
    py    = (rows[:, None, None] + sub_y[None, None, :]) / height
    ndc_x = px * 2.0 - 1.0
    ndc_y = 1.0 - py * 2.0
    spp   = n_axis * n_axis
    ndc_x = np.broadcast_to(ndc_x, (r1 - r0, width, spp))
    ndc_y = np.broadcast_to(ndc_y, (r1 - r0, width, spp))

    R = camera_matrix[:3, :3]
    if job.is_orthographic:
        cam_origins = np.stack(
            [ndc_x * half_w, ndc_y * half_h, np.zeros_like(ndc_x)], axis=-1
        ).reshape(-1, 3)
        cam_dirs = np.tile(
            np.array([0.0, 0.0, -1.0], dtype=np.float64), (cam_origins.shape[0], 1)
        )
        world_dirs = _matmul(cam_dirs, R.T)
        # camera_matrix[3, :3] (its bottom row, not the eye at [:3, 3]):
        # the whole-frame path's term, kept so the two match
        with np.errstate(all="ignore"):
            world_origins = _matmul(cam_origins, R.T) + camera_matrix[3, :3]
        return world_origins.astype(np.float64), world_dirs.astype(np.float64)

    cam_dirs = np.stack(
        [ndc_x * half_w, ndc_y * half_h, -np.ones_like(ndc_x)], axis=-1
    ).reshape(-1, 3)
    cam_dirs   = cam_dirs / np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    world_dirs = _matmul(cam_dirs, R.T)

    cam_pos = camera_matrix[:3, 3]
    origins = np.tile(cam_pos[None, :], (cam_dirs.shape[0], 1))
    return origins.astype(np.float64), world_dirs.astype(np.float64)


def _shade_band(
    prep:       _Prepared,
    origins:    np.ndarray,
    directions: np.ndarray,
    lights:     List[dict],
    pad:        bool,
):
    """raytracer._shade_mesh_per_sample on one band's samples.

    With *pad*, a single hit is shaded as two identical rows (see the module
    docstring). Returns ``(color, hit, distance, hit count)``.
    """
    entry   = prep.entry
    M       = entry["world_matrix"]
    n_total = origins.shape[0]

    if prep.is_world:
        cast_origins    = origins
        cast_directions = directions
    else:
        with np.errstate(all="ignore"):
            cast_origins    = _matmul(origins, prep.R3.T) + prep.T3
            cast_directions = _matmul(directions, prep.R3.T)

    sample_color    = np.zeros((n_total, 3), dtype=np.float32)
    sample_distance = np.full(n_total, np.inf, dtype=np.float64)
    sample_hit, hits = _cast_hits(prep, cast_origins, cast_directions)
    if hits is None:
        return sample_color, sample_hit, sample_distance, 0

    hit_idx, hit_w, hit_proj, hit_norm, hit_dist = hits
    n_hit = hit_idx.shape[0]
    if pad and n_hit == 1:
        hit_idx, hit_w, hit_proj, hit_norm, hit_dist = (
            np.concatenate([a, a]) for a in (hit_idx, hit_w, hit_proj, hit_norm, hit_dist)
        )

    tex = entry["tex"]
    if tex is None:
        sampled = np.tile(
            np.asarray(entry["base_color"], dtype=np.float32), (hit_idx.shape[0], 1)
        )
    else:
        interp_uvs = _rt._interpolate_uvs(entry["uv"], hit_idx, hit_w)
        sampled    = _rt._sample_texture_bilinear(tex, interp_uvs, entry["wrap"] == "repeat")

    # ---- back-transform projections + normals to world space ----
    if not prep.is_world:
        with np.errstate(all="ignore"):
            hit_proj = _matmul(hit_proj, M[:3, :3].T) + M[:3, 3]
            hit_norm = _matmul(hit_norm, prep.N3.T)
        nlen     = np.linalg.norm(hit_norm, axis=1, keepdims=True)
        nlen     = np.maximum(nlen, 1e-12)
        hit_norm = hit_norm / nlen

    if lights:
        shaded = _shade_lights(sampled, hit_proj, hit_norm, lights, entry["ambient"])
    else:
        shaded = sampled * max(entry["ambient"], 1.0)

    sample_color[sample_hit] = np.clip(shaded, 0.0, 1.0).astype(np.float32)[:n_hit]

    # World-space hit distance (object-local t is meaningless across objects).
    if prep.is_world:
        sample_distance[sample_hit] = hit_dist[:n_hit]
    else:
        diff                        = hit_proj[:n_hit] - origins[sample_hit]
        sample_distance[sample_hit] = np.linalg.norm(diff, axis=1)

    return sample_color, sample_hit, sample_distance, n_hit


def _band(job: _Job, index: int, pads: List[bool]) -> List[int]:
    """Renders band *index* into the frame's buffers: _render_scene's z-merge,
    then _compose_scene_output's resolve and composite, on its rows.
    Returns each entry's hit count in the band."""
    r0, r1 = job.bands[index]
    width = job.width
    spp   = job.n_axis * job.n_axis

    origins, directions = _band_rays(job, r0, r1)
    n_total = origins.shape[0]

    z_color = np.zeros((n_total, 3), dtype=np.float32)
    z_depth = np.full(n_total, np.inf, dtype=np.float64)
    z_hit   = np.zeros(n_total, dtype=bool)
    counts  = []
    for prep, pad in zip(job.preps, pads):
        sample_color, sample_hit, sample_distance, n_hit = _shade_band(
            prep, origins, directions, job.lights, pad
        )
        counts.append(n_hit)
        winner = sample_hit & (sample_distance < z_depth)
        if winner.any():
            z_color[winner] = sample_color[winner]
            z_depth[winner] = sample_distance[winner]
            z_hit[winner]   = True

    n_pixels              = (r1 - r0) * width
    per_sample            = np.zeros((n_total, 4), dtype=np.float32)
    per_sample[z_hit, :3] = z_color[z_hit]
    per_sample[z_hit, 3]  = 1.0

    rendered         = _rt._resolve_msaa(per_sample, n_pixels, spp)
    composited       = _rt._composite_over(rendered, job.background)
    job.image[r0:r1] = _rt._to_uint8_image(composited, r1 - r0, width, n_channels=4)

    if job.depth is not None:
        distances = np.where(z_hit, z_depth, np.nan)
        job.depth[r0:r1] = _rt._resolve_msaa_depth(
            distances, z_hit, n_pixels, spp
        ).reshape(r1 - r0, width)

    job.idents[index] = threading.get_ident()
    return counts


# -- scheduling --------------------------------------------------------------------


def _bands(width: int, height: int, spp: int, budget: int) -> List[Tuple[int, int]]:
    """``[(first row, end row), ...]``: whole pixel rows, so every sample of a
    pixel is in one band, and two samples or more in every band."""
    n = width * height * spp
    if width * spp < 2 or n < _ONE_BAND_BELOW:
        return [(0, height)]
    target = min(_MAX_BAND, max(_MIN_BAND, -(-n // (2 * budget))))
    rows   = max(1, target // (width * spp))
    return [(r0, min(r0 + rows, height)) for r0 in range(0, height, rows)]


class _Queue:
    """The bands not taken yet, the first errors, and how many bands the
    pool's threads are rendering."""

    def __init__(self, indices):
        self._lock  = threading.Condition()
        self._next  = iter(indices)
        self._stop  = False
        self._busy  = 0
        self.errors = {}

    def take(self, pooled: bool) -> Optional[int]:
        """The next band, or None. A pool thread (*pooled*) calls
        :meth:`done` when the band ends."""
        with self._lock:
            if self._stop:
                return None
            index = next(self._next, None)
            if pooled and index is not None:
                self._busy += 1
            return index

    def done(self) -> None:
        with self._lock:
            self._busy -= 1
            self._lock.notify_all()

    def fail(self, index: int, exc: BaseException) -> None:
        with self._lock:
            self.errors[index] = exc
            self._stop         = True

    def halt(self) -> None:
        with self._lock:
            self._stop = True

    def settle(self) -> None:
        """Halts, and returns once the pool's threads end their bands. The
        calling thread, which settles, keeps no count: a Ctrl-C landing
        between its take() and its band would leave it off."""
        with self._lock:
            self._stop = True
            while self._busy:
                self._lock.wait()


def _interrupted(exc: BaseException) -> bool:
    """True when a KeyboardInterrupt caused *exc*: a Ctrl-C landing in
    Python code that a numba kernel calls back into comes out of the
    kernel as a SystemError raised from it, at any depth of
    ``__cause__``. Not ``__context__``: a render called while a
    KeyboardInterrupt is handled (``except KeyboardInterrupt:``,
    ``finally:``) finds it there on its own errors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, KeyboardInterrupt):
            return True
        seen.add(id(exc))
        exc = exc.__cause__
    return False


def _runner(
    job:    _Job,
    queue:  _Queue,
    counts: list,
    pads:   List[bool],
    nb:     int,
    pooled: bool,
) -> None:
    """Renders bands until none is left, or one failed, under the caller's
    numpy floating-point policy: a pool thread starts at numpy's default
    (np.errstate is per thread, or a context variable)."""
    err, call = job.numpy_err
    with np.errstate(call=call, **err):
        while True:
            index = queue.take(pooled)
            if index is None:
                return
            try:
                numba.set_num_threads(nb)
                counts[index] = _band(job, index, pads)
            except Exception as exc:
                if _interrupted(exc):
                    # Ctrl-C (only the calling thread gets one), not a band's error
                    queue.halt()
                    raise KeyboardInterrupt from exc
                queue.fail(index, exc)
                return
            finally:
                if pooled:
                    queue.done()


def _run(job: _Job, indices, counts: list, pads: List[bool], runners: int, budget: int) -> int:
    """Renders the bands *indices* on *runners* threads, the calling thread
    one of them, each casting with ``budget // runners`` numba threads.
    Returns how many threads ran: the calling thread and the runners the
    pool started. Raises the error of the lowest failing band; on a
    Ctrl-C, once the bands in flight end."""
    queue   = _Queue(indices)
    nb      = max(1, budget // runners)
    futures = []
    try:
        # inside the try: a Ctrl-C landing in submit (the pool's threads
        # start there) halts the runners already queued
        if runners > 1:
            pool = _get_pool()
            try:
                for _ in range(runners - 1):
                    futures.append(pool.submit(_runner, job, queue, counts, pads, nb, True))
            except RuntimeError:
                # the interpreter is shutting down (a render from an atexit
                # handler): the pool takes no new work, so the calling thread
                # renders the rest, the same bytes
                if not futures:
                    nb = budget
        _runner(job, queue, counts, pads, nb, False)
        # no band is left: a runner still queued in the pool, behind other
        # renders' runners, would find none, so it is cancelled, not waited
        # for. A timed wait for the others, so that Ctrl-C lands
        pending = {future for future in futures if not future.cancel()}
        while pending:
            _, pending = wait(pending, timeout=0.1)
    except BaseException:
        queue.settle()
        wait([future for future in futures if not future.cancel()])
        raise
    started = [future for future in futures if not future.cancelled()]
    for future in started:
        future.result()
    if queue.errors:
        raise queue.errors[min(queue.errors)]
    return 1 + len(started)


def render_bands(
    scene_entries:   List[dict],
    camera_matrix:   np.ndarray,
    is_orthographic: bool,
    ortho_height:    Optional[float],
    angle_of_view:   float,
    width:           int,
    height:          int,
    n_axis:          int,
    lights:          List[dict],
    background:      Tuple[float, float, float, float],
    return_depth:    bool,
    threads:         Optional[int],
) -> Frame:
    """render()'s frame, band by band: the arguments are render()'s, resolved
    (camera, autofit, ortho height, lights)."""
    # the background is checked before any work, as _composite_over checks it
    _rt._composite_over(np.zeros((0, 4), dtype=np.float32), background)

    try:
        preps = [_prepare(e) for e in scene_entries]
    except Exception as exc:
        if _interrupted(exc):
            # Ctrl-C, as in _runner
            raise KeyboardInterrupt from exc
        raise
    budget = thread_budget(threads)
    spp    = n_axis * n_axis
    bands  = _bands(width, height, spp, budget)

    job                 = _Job()
    job.preps           = preps
    job.camera_matrix   = camera_matrix
    job.is_orthographic = is_orthographic
    job.ortho_height    = ortho_height
    job.angle_of_view   = angle_of_view
    job.width           = width
    job.height          = height
    job.n_axis          = n_axis
    job.lights          = lights
    job.background      = background
    job.bands           = bands
    job.image           = np.empty((height, width, 4), dtype=np.uint8)
    job.depth           = np.empty((height, width), dtype=np.float64) if return_depth else None
    job.idents          = [None] * len(bands)
    # the caller's numpy floating-point policy (np.errstate, np.seterr),
    # for the bands the pool's threads render
    job.numpy_err = (np.geterr(), np.geterrcall())

    pads    = [len(bands) > 1] * len(preps)
    counts  = [None] * len(bands)
    runners = min(budget, len(bands))
    redone  = []
    caller  = numba.get_num_threads()
    try:
        rest = list(range(len(bands)))
        if runners > 1 and _threading_layer() is None:
            # numba.get_num_threads() launches numba's threading layer, so
            # it is known by now; were it not, band 0 runs alone first, and
            # its kernels launch it
            _run(job, rest[:1], counts, pads, 1, budget)
            rest = rest[1:]
        runners = min(runners, len(rest))
        if runners > 1 and _threading_layer() in ("omp", "tbb"):
            runners = _run(job, rest, counts, pads, runners, budget)
        else:
            # one band, one thread, or numba's workqueue layer, which aborts
            # when two threads launch kernels at once: the calling thread
            runners = 1
            _run(job, rest, counts, pads, 1, budget)

        # a mesh the whole frame hits once is shaded as one row, as the
        # whole frame shades it: its band again, that mesh unpadded
        if len(bands) > 1:
            once     = [sum(c[e] for c in counts) == 1 for e in range(len(preps))]
            unpadded = [p and not o for p, o in zip(pads, once)]
            redone = [
                index
                for index, band_counts in enumerate(counts)
                if any(n == 1 and o for n, o in zip(band_counts, once))
            ]
            for index in redone:
                _run(job, [index], counts, unpadded, 1, budget)
    finally:
        numba.set_num_threads(caller)

    if _stats_hook is not None:
        _stats_hook(
            {
                "bands":   bands,
                "counts":  counts,
                "redone":  redone,
                "idents":  job.idents,
                "runners": runners,
                "budget":  budget,
            }
        )

    return Frame(
        array         = job.image,
        depth         = job.depth,
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64),
        angle_of_view = float(angle_of_view),
    )
