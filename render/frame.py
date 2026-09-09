"""
Render result wrapper.

A :class:`Frame` is the value :func:`render` returns: it holds the raw
``uint8`` pixel buffer, an optional depth buffer, and exposes a few
ergonomic conveniences (lazy PIL ``Image`` view, ``.save()`` shortcut,
``np.asarray()`` interop, tuple unpacking for ``(image, depth)``).

Why a wrapper instead of returning a bare ``np.ndarray``?

* Saving to disk is one call: ``frame.save("out.png")`` instead of
  ``Image.fromarray(arr).save("out.png")``.
* Numpy interop is preserved: ``np.asarray(frame)`` and ``np.array_equal(a, b)``
  work transparently via :meth:`__array__`.
* The depth buffer travels alongside the colour buffer in a single object
  rather than as a sibling tuple element.
* Future fields (timing, samples-per-pixel actually used, AOVs) can be
  added without further breaking changes.

Shape / dtype contract:

* ``frame.array`` is ``(H, W, 3)`` or ``(H, W, 4)`` ``uint8`` (RGB or RGBA).
* ``frame.depth`` is ``(H, W)`` ``float64`` with NaN at miss pixels, or
  ``None`` when the renderer was not asked for depth.

For arithmetic / boolean / comparison operations, go through ``frame.array``
explicitly.  ``Frame`` only proxies indexing (``frame[y, x]``) and shape
metadata (``.shape``, ``.dtype``, ``.height``, ``.width``) -- it deliberately
does not pretend to be an ndarray.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

import numpy as np

# PIL is optional -- only needed by ``Frame.image`` / ``Frame.save`` /
# ``Frame.wireframe`` / ``Frame.encode_gif``.  Match the lazy-import
# pattern used in ``cgmath.geometry.mesh``: import once at module load
# time and let call sites guard with ``if Image is None``.
try:
    from PIL import Image, ImageDraw
except ImportError:
    Image     = None
    ImageDraw = None


# -- ffmpeg / gifski subprocess helpers -----------------------------
#
# The video / gif encoders rely on external binaries (``ffmpeg``,
# optionally ``gifski``).  We deliberately do NOT search for these
# beyond the standard PATH lookup that the OS performs automatically
# when ``subprocess`` spawns a bare command name: if ``which ffmpeg``
# resolves in your shell, the encoder works; if not, install ffmpeg
# and make sure it's on PATH for the calling process.
#
# The only thing these helpers add over a plain ``subprocess.Popen`` /
# ``subprocess.run`` is converting the low-level ``FileNotFoundError``
# (raised by ``execvp`` / ``CreateProcess`` when the binary isn't on
# PATH) into a friendly ``RuntimeError`` with an actionable hint.
#
# Note for DCC users (Maya / Houdini / etc. on macOS): GUI apps
# launched from Finder inherit only the bare Apple login PATH
# (``/usr/bin:/bin:/usr/sbin:/sbin``).  If ffmpeg works in Terminal
# but not inside the DCC, fix the DCC's launch environment
# (e.g. ``Maya.env`` -> ``PATH = /opt/homebrew/bin:$PATH``) rather
# than working around it here.


def _spawn_or_raise(cmd: list[str], **popen_kwargs: Any) -> subprocess.Popen:
    """:func:`subprocess.Popen` with a friendly error when the binary
    at ``cmd[0]`` is not on the calling process's PATH.
    """
    try:
        return subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{cmd[0]!r} not found on PATH.  Install it and make sure "
            f"it is on the PATH of the process calling this function "
            f"(verify with `which {cmd[0]}` on macOS/Linux or "
            f"`where {cmd[0]}` on Windows)."
        ) from e


def _run_or_raise(
    cmd: list[str], **run_kwargs: Any
) -> "subprocess.CompletedProcess[bytes]":
    """:func:`subprocess.run` with a friendly error when the binary
    at ``cmd[0]`` is not on the calling process's PATH.
    """
    try:
        return subprocess.run(cmd, **run_kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{cmd[0]!r} not found on PATH.  Install it and make sure "
            f"it is on the PATH of the process calling this function "
            f"(verify with `which {cmd[0]}` on macOS/Linux or "
            f"`where {cmd[0]}` on Windows)."
        ) from e


@dataclass(frozen=True, eq=False)
class Frame:
    """One rendered image (plus optional depth buffer)."""

    array: np.ndarray
    """``(H, W, 3)`` or ``(H, W, 4)`` ``uint8`` pixel buffer."""

    depth: Optional[np.ndarray] = None
    """``(H, W)`` ``float64`` depth buffer (NaN at miss), or None."""

    camera_matrix: Optional[np.ndarray] = None
    """Column-major ``(4, 4)`` camera-to-world matrix used to render this
    frame.  Populated automatically by :func:`render`; ``None`` for Frames
    constructed manually.  Used by :meth:`wireframe` to project edges to
    screen space without re-passing camera info."""

    angle_of_view: Optional[float] = None
    """Vertical FOV (degrees) used to render this frame, populated by
    :func:`render`.  Used by :meth:`wireframe`."""

    # -- shape / dtype proxies ----------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of :attr:`array`."""
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        """``dtype`` of :attr:`array`."""
        return self.array.dtype

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.array.shape[1])

    @property
    def channels(self) -> int:
        """Number of colour channels (3 for RGB, 4 for RGBA)."""
        return int(self.array.shape[2]) if self.array.ndim >= 3 else 1

    @property
    def has_depth(self) -> bool:
        """Whether a depth buffer is attached."""
        return self.depth is not None

    # -- PIL bridge ---------------------------------------------------------

    @property
    def image(self):
        """Lazily-built PIL ``Image`` view of :attr:`array`.

        PIL is imported once at module load time; if it isn't available
        this raises :class:`RuntimeError`.
        """
        if Image is None:
            raise RuntimeError(
                "PIL required to build a PIL.Image view of a Frame; install Pillow."
            )
        return Image.fromarray(self.array)

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the frame to disk via PIL.

        ``path`` may use ``~`` and environment variables (expanded
        automatically).  Any extra keyword args are forwarded to
        :meth:`PIL.Image.Image.save` (e.g. ``quality=95`` for JPEG).
        """
        expanded = os.path.expandvars(os.path.expanduser(path))
        self.image.save(expanded, **kwargs)

    # -- numpy + iteration interop ------------------------------------------

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Numpy array protocol: ``np.asarray(frame)`` returns ``frame.array``.

        Lets ``Frame`` slot transparently into anything that calls
        ``np.asarray`` internally (``np.array_equal``, ``np.testing.*``,
        most array-consuming functions).
        """
        if dtype is None:
            return self.array
        return self.array.astype(dtype)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Tuple-style unpacking: ``arr, depth = frame``.

        When :attr:`depth` is ``None``, yields just the array (so plain
        ``arr, = frame`` also works).  This preserves the legacy
        ``(image, depth) = render(..., return_depth=True)`` call shape.
        """
        if self.depth is None:
            return iter((self.array,))
        return iter((self.array, self.depth))

    def __getitem__(self, key: Any) -> Any:
        """Index / slice through to :attr:`array`.

        Lets ``frame[y, x]`` and ``frame[mask]`` work without an explicit
        ``.array`` hop.
        """
        return self.array[key]

    def __len__(self) -> int:
        """Number of rows in :attr:`array` (matches numpy semantics)."""
        return int(self.array.shape[0])

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        # Don't dump the entire pixel buffer.
        parts = [f"shape={self.shape}", f"dtype={self.dtype}"]
        if self.depth is not None:
            parts.append("depth=True")
        if self.camera_matrix is not None:
            parts.append("cam=True")
        return f"Frame({', '.join(parts)})"

    # -- post-process: wireframe overlay -----------------------------------

    def wireframe(
        self,
        source:         Any,
        color:          Tuple[int, int, int, int] = (0, 0, 0, 200),
        width:          int                       = 1,
        unique_edges:   bool                      = True,
        cull_backfaces: bool                      = True,
    ) -> "Frame":
        """Draw mesh edges over this frame as a wireframe overlay.

        Edges are projected to screen via the frame's :attr:`camera_matrix`
        and :attr:`angle_of_view` (set automatically when the frame comes
        out of :func:`render`), drawn with PIL's anti-aliased line
        rasterizer, and composited onto a copy of the image.

        Back-face culling is on by default: an edge is drawn only when at
        least one of its adjacent faces faces the camera.  Manifold border
        edges (one adjacent face) and silhouettes get drawn naturally.
        Pass ``cull_backfaces=False`` for an X-ray overlay (every edge
        drawn, regardless of occlusion or facing).

        Args:
            source: Where to get edges from.  May be a
                :class:`cgmath.render.scene.Scene` (every visible
                Object's mesh contributes), a single
                :class:`cgmath.render.scene.Object` (its mesh's edges
                in world space), or a
                :class:`cgmath.geometry.mesh.MeshData` (uses the mesh's
                own ``points`` directly, treated as world-space).
            color: RGBA tuple in 0-255 range.  Default is dark with
                ~80% opacity for a tasteful overlay.  Pass full opacity
                ``(r, g, b, 255)`` for stronger lines.
            width: Line width in pixels (PIL ImageDraw width).  ``1``
                is the standard wireframe; bump to 2-3 for hero shots.
            unique_edges: When True (default), each shared edge is drawn
                once.  Set False to draw per-face edges (effectively
                doubling shared edges -- usually not what you want).
            cull_backfaces: When True (default), drop edges whose every
                adjacent face is back-facing (pointing away from the
                camera).  Set False for an X-ray wireframe with all
                edges visible regardless of orientation.

        Returns:
            A new :class:`Frame` with the same depth / camera-matrix /
            angle-of-view, but an updated pixel array.

        Raises:
            ValueError: If this frame has no ``camera_matrix`` or
                ``angle_of_view`` (e.g., it was constructed manually
                rather than coming from :func:`render`).
            TypeError: If ``source`` is not a Scene / Object / MeshData.

        Example::

            frame = scene.render(resolution=(1024, 1024))
            wired = frame.wireframe(scene, color=(255, 255, 255, 220), width=2)
            wired.save("with_wires.png")

            # X-ray overlay (back wires visible too)
            xray = frame.wireframe(scene, cull_backfaces=False)
        """
        if self.camera_matrix is None or self.angle_of_view is None:
            raise ValueError(
                "Frame.wireframe: this Frame has no camera_matrix / "
                "angle_of_view (was it constructed manually?).  Render "
                "via Scene.render() / Object.render() to populate them."
            )
        if Image is None or ImageDraw is None:
            raise RuntimeError(
                "PIL required to draw a wireframe overlay; install Pillow."
            )

        primitives = list(
            _iter_wireframe_primitives(
                source, unique=unique_edges, with_cull_data=cull_backfaces
            )
        )
        if not primitives:
            # Nothing to draw -- return a copy preserving identity.
            return Frame(
                array         = self.array.copy(),
                depth         = self.depth,
                camera_matrix = self.camera_matrix,
                angle_of_view = self.angle_of_view,
            )

        # Lazy PIL import (matches the pattern elsewhere in this module).
        # Guarded above by the ``if Image is None`` check.
        _PIL_Image = Image

        H, W = self.height, self.width
        cam_col   = np.asarray(self.camera_matrix, dtype=np.float64)
        R_cam     = cam_col[:3, :3]
        eye       = cam_col[:3, 3]

        fov_v_rad = float(np.deg2rad(self.angle_of_view))
        half_h    = float(np.tan(fov_v_rad * 0.5))
        half_w    = half_h * (W / H)

        base      = self.image.convert("RGBA")
        overlay   = _PIL_Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw      = ImageDraw.Draw(overlay)

        # Silence Apple Accelerate matmul-warning noise during projection.
        with np.errstate(all="ignore"):
            for prim in primitives:
                world_pts  = np.asarray(prim["world_pts"], dtype=np.float64)
                edge_pairs = np.asarray(prim["edge_pairs"], dtype=np.int64)
                if world_pts.size == 0 or edge_pairs.size == 0:
                    continue

                # World -> camera-local: row-vector convention with a
                # column-major camera-to-world R_cam means
                # (pts - eye) @ R_cam takes world -> camera-local.
                pts_local = (world_pts - eye) @ R_cam

                # In OpenGL the camera looks down -Z, so points in front
                # have negative local-Z.  Cull anything at or behind the
                # camera plane.
                in_front = pts_local[:, 2] < -1e-6
                # Avoid divide-by-zero on culled points -- they get masked
                # off by `edge_visible` below before any pixel is drawn.
                z_safe = np.where(in_front, pts_local[:, 2], -1.0)
                pix_x  = (pts_local[:, 0] / -z_safe / half_w + 1.0) * 0.5 * W
                pix_y  = (1.0 - pts_local[:, 1] / -z_safe / half_h) * 0.5 * H

                a            = edge_pairs[:, 0]
                b            = edge_pairs[:, 1]
                edge_visible = in_front[a] & in_front[b]

                # Back-face culling: drop edges where every adjacent face
                # is back-facing (pointing away from the camera).
                if cull_backfaces and prim.get("face_normals") is not None:
                    face_normals = prim["face_normals"]  # (F, 3) world-space
                    face_centres = prim["face_centres"]  # (F, 3) world-space
                    e2f          = prim["e2f"]           # (E, max_adj), -1 padded

                    # View direction: from face centre toward camera eye.
                    view_dirs = eye[None, :] - face_centres
                    # Front-facing if the face's outward normal has a positive
                    # component along the view direction.
                    front_facing = (
                        np.einsum("ij,ij->i", face_normals, view_dirs) > 0.0
                    )  # shape (F,)

                    # For each edge, check if any of its adjacent faces
                    # (e2f rows) is front-facing.  -1 in e2f means "no face
                    # in this slot"; mask those out.
                    e2f_valid      = e2f >= 0
                    e2f_safe       = np.where(e2f_valid, e2f, 0)
                    adj_front      = front_facing[e2f_safe] & e2f_valid
                    edge_keep_cull = adj_front.any(axis=1)
                    edge_visible &= edge_keep_cull

                xs0     = pix_x[a]
                ys0     = pix_y[a]
                xs1     = pix_x[b]
                ys1     = pix_y[b]
                vis_idx = np.where(edge_visible)[0]
                for i in vis_idx:
                    draw.line(
                        [
                            float(xs0[i]),
                            float(ys0[i]),
                            float(xs1[i]),
                            float(ys1[i]),
                        ],
                        fill  = color,
                        width = width,
                    )

        composited = _PIL_Image.alpha_composite(base, overlay)
        # Preserve the input frame's channel count: input was 3-channel
        # (legacy callers) -> output 3-channel; input 4-channel (the new
        # post-RGBA-refactor renderer default) -> output 4-channel.
        out_mode = "RGBA" if self.array.shape[-1] == 4 else "RGB"
        return Frame(
            array         = np.asarray(composited.convert(out_mode), dtype=np.uint8),
            depth         = self.depth,
            camera_matrix = self.camera_matrix,
            angle_of_view = self.angle_of_view,
        )

    # -- video / gif encoders -----------------------------------------------

    # Type alias for "anything that quacks like an RGB(A) uint8 frame":
    # a Frame, a (H, W, 3|4) uint8 ndarray, or anything with __array__.
    FrameLike = Union["Frame", np.ndarray]

    @classmethod
    def encode(
        cls,
        frames: Iterable["Frame.FrameLike"],
        output: str,
        fps:    int                         = 30,
        **kwargs: Any,
    ) -> str:
        """Dispatch to :meth:`encode_mp4` / :meth:`encode_avi` /
        :meth:`encode_gif` from the output extension.

        Recognised extensions:

          * ``.mp4`` / ``.mov`` / ``.m4v`` -> :meth:`encode_mp4`
            (H.264; alpha auto-flattened over a background colour).
          * ``.avi`` -> :meth:`encode_avi` (PNG codec inside AVI;
            alpha is preserved).
          * ``.gif`` -> :meth:`encode_gif` (1-bit alpha only;
            semi-transparent pixels are thresholded).

        Anything else raises ``ValueError``.
        """
        ext = os.path.splitext(output)[1].lower()
        if ext in (".mp4", ".mov", ".m4v"):
            return cls.encode_mp4(frames, output, fps=fps, **kwargs)
        if ext == ".avi":
            return cls.encode_avi(frames, output, fps=fps, **kwargs)
        if ext == ".gif":
            return cls.encode_gif(frames, output, fps=fps, **kwargs)
        raise ValueError(
            f"Frame.encode: unsupported extension {ext!r}; "
            "use Frame.encode_mp4 / encode_avi / encode_gif explicitly, "
            "or pass a path ending in .mp4 / .mov / .m4v / .avi / .gif"
        )

    @classmethod
    def encode_mp4(
        cls,
        frames:     Iterable["Frame.FrameLike"],
        output:     str,
        fps:        int                         = 30,
        background: Tuple[float, float, float]  = (0.0, 0.0, 0.0),
        crf:        int                         = 18,
        preset:     str                         = "slow",
        overwrite:  bool                        = True,
    ) -> str:
        """Encode a sequence of frames to an H.264 mp4 (YouTube-compatible).

        Streams raw RGB bytes into ``ffmpeg`` over a pipe -- no temporary
        PNG files on disk.  Each frame is held in memory only briefly.

        H.264 carries no alpha, so RGBA inputs are auto-flattened by
        compositing over the supplied opaque ``background`` colour using
        premultiplied Porter-Duff "over" before ffmpeg sees them.  The
        renderer always produces 4-channel output now, so this code path
        fires every time you pipe a render through to mp4 -- the math
        is a no-op for already-opaque pixels and gives the user-chosen
        miss colour back for transparent pixels.

        :meth:`Scene.turntable` and :meth:`Object.turntable` forward
        their parent's :attr:`Scene.background` (RGB part) here
        automatically when the output extension is mp4 / mov / m4v, so
        videos respect the background you set on the Scene.

        Args:
            frames: Iterable of :class:`Frame` or ``(H, W, 3|4) uint8``
                arrays.  All frames must share the dimensions of the
                first.  May be a generator -- consumed lazily.
            output: Output file path.  ``~`` and env vars are expanded.
            fps: Frames per second (default 30 -- the most universal).
            background: Opaque RGB triple in [0, 1] used to flatten
                any alpha channel.  Default opaque black ``(0, 0, 0)``.
                Pass the same RGB you used for the renderer's bg if you
                want the video's miss pixels to match the rendered
                frames.  ``Scene.turntable`` does this automatically.
            crf: x264 constant-quality (lower = better; 18 = visually
                lossless, 23 = libx264 default, anything <= 18 is
                overkill for YouTube because YouTube re-encodes anyway).
            preset: x264 speed/quality tradeoff (default ``"slow"``).
                ``"veryslow"`` ~5% better quality, ~10x slower.
                ``"medium"`` is the libx264 default but lower quality
                at the same CRF.
            overwrite: If True (default), pass ``-y`` to ffmpeg so
                existing files are overwritten.

        Returns:
            The (expanded) absolute output path.

        Raises:
            RuntimeError: If ``ffmpeg`` is not on the calling process's
                PATH, or if ffmpeg exits non-zero (stderr is included
                in the message).
            ValueError: If ``frames`` is empty, contains non-uint8
                frames, or contains frames whose channel count is
                neither 3 nor 4.

        ffmpeg invocation (matches our YouTube-compatibility recommendation):
            ``-c:v libx264 -preset {preset} -crf {crf} -pix_fmt yuv420p
            -profile:v high -level 4.0 -movflags +faststart -tune animation
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"``
        """
        # Validate background first so callers get a clean ValueError
        # regardless of whether ffmpeg is installed -- CI runners
        # (Sandcastle Linux) often don't have ffmpeg on PATH and we
        # don't want shape / range bugs to surface as a misleading
        # "ffmpeg not found" RuntimeError.
        bg_rgb = np.asarray(background, dtype=np.float32)
        if bg_rgb.shape != (3,):
            raise ValueError(
                f"encode_mp4: background must be RGB 3-tuple of floats; "
                f"got {background!r}"
            )
        if np.any(bg_rgb < 0.0) or np.any(bg_rgb > 1.0):
            raise ValueError(
                f"encode_mp4: background components must be in [0, 1]; "
                f"got {background!r}"
            )
        bg_uint8 = (bg_rgb * 255.0).astype(np.float32)  # kept as float for math

        out = os.path.expandvars(os.path.expanduser(output))

        # Peek at the first frame to learn dimensions and validate the format.
        frames_iter = iter(frames)
        try:
            first = next(frames_iter)
        except StopIteration:
            raise ValueError("encode_mp4: no frames provided")

        first_arr = np.ascontiguousarray(np.asarray(first), dtype=np.uint8)
        if first_arr.ndim != 3:
            raise ValueError(
                f"encode_mp4: frames must be (H, W, C); got shape {first_arr.shape}"
            )
        if first_arr.shape[-1] not in (3, 4):
            raise ValueError(
                f"encode_mp4: frames must be (H, W, 3) or (H, W, 4); "
                f"got {first_arr.shape[-1]} channels"
            )

        height, width = int(first_arr.shape[0]), int(first_arr.shape[1])

        def _flatten_to_rgb(arr: np.ndarray) -> np.ndarray:
            """RGBA -> RGB by premultiplied composite over ``bg_uint8``."""
            if arr.shape[-1] == 3:
                return arr
            a = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = arr[..., :3].astype(np.float32) * a + bg_uint8[None, None, :] * (
                1.0 - a
            )
            return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

        cmd = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-loglevel",
            "error",
            # raw RGB input over stdin
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(fps)),
            "-i",
            "pipe:",
            # output codec (YouTube-compatible H.264)
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.0",
            "-movflags",
            "+faststart",
            "-tune",
            "animation",
            # Pad to even dimensions (yuv420p chroma subsampling requires it).
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out,
        ]

        # ``shell=False`` (default) -- on POSIX this lets execvp do the
        # PATH lookup directly; on Windows ``CreateProcess`` does the
        # same and honours ``PATHEXT`` so ``"ffmpeg"`` resolves to
        # ``ffmpeg.exe`` automatically.  Wrapping in ``_spawn_or_raise``
        # converts the low-level ``FileNotFoundError`` to a friendly
        # ``RuntimeError`` when ffmpeg isn't on PATH.
        proc = _spawn_or_raise(
            cmd,
            stdin  = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        assert proc.stdin is not None  # for type checkers

        try:
            # Stream the first frame we already pulled, then the rest.
            proc.stdin.write(_flatten_to_rgb(first_arr).tobytes())
            for i, f in enumerate(frames_iter, start=1):
                arr = np.ascontiguousarray(np.asarray(f), dtype=np.uint8)
                if arr.shape[:2] != (height, width):
                    raise ValueError(
                        f"encode_mp4: frame {i} has shape {arr.shape}, "
                        f"expected ({height}, {width}, 3|4) matching first frame"
                    )
                if arr.shape[-1] not in (3, 4):
                    raise ValueError(
                        f"encode_mp4: frame {i} must be 3- or 4-channel; "
                        f"got shape {arr.shape}"
                    )
                proc.stdin.write(_flatten_to_rgb(arr).tobytes())
        except BrokenPipeError:
            # ffmpeg died mid-stream; we'll surface its stderr below.
            pass
        finally:
            # Close stdin manually so ffmpeg sees EOF and exits.  Guard
            # against double-close + the BrokenPipeError that close() can
            # raise if ffmpeg already disappeared.
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        # Note: we do NOT call ``proc.communicate()`` here.  Python 3.11's
        # subprocess.communicate() unconditionally calls ``self.stdin.flush()``
        # before reading -- and our stdin is already closed, which raises
        # ``ValueError: flush of closed file``.  Drain stderr and wait
        # directly instead.
        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = proc.stderr.read()
            finally:
                try:
                    proc.stderr.close()
                except OSError:
                    pass
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}):\n"
                f"{stderr_bytes.decode(errors='replace')}"
            )
        return out

    @classmethod
    def encode_avi(
        cls,
        frames:    Iterable["Frame.FrameLike"],
        output:    str,
        fps:       int                         = 30,
        overwrite: bool                        = True,
    ) -> str:
        """Encode RGBA frames to an alpha-preserving AVI via the PNG codec.

        H.264 / H.265 carry no alpha; PNG-in-AVI is the cleanest option
        for transparent video that ffmpeg can produce out of the box.
        Tradeoff: each frame is encoded as an independent PNG, so the
        file is larger than an equivalent H.264 mp4 -- but you keep
        full per-pixel alpha for compositing into NLE timelines.

        Accepts both ``(H, W, 3)`` and ``(H, W, 4)`` ``uint8`` frames;
        RGB frames are auto-promoted to opaque RGBA so ffmpeg always
        sees a uniform 4-channel stream.

        Args:
            frames: Iterable of :class:`Frame` or ``(H, W, 3|4) uint8``
                arrays.  All frames must share the dimensions of the
                first.  Consumed lazily.
            output: Output file path; ``.avi`` extension recommended.
            fps: Frames per second (default 30).
            overwrite: If True (default), pass ``-y`` to ffmpeg.

        Returns:
            The (expanded) absolute output path.

        Raises:
            RuntimeError: If ffmpeg is not on the calling process's
                PATH, or if ffmpeg exits non-zero.
            ValueError: If ``frames`` is empty or contains frames whose
                channel count is neither 3 nor 4.
        """
        out = os.path.expandvars(os.path.expanduser(output))

        frames_iter = iter(frames)
        try:
            first = next(frames_iter)
        except StopIteration:
            raise ValueError("encode_avi: no frames provided")

        first_arr = np.ascontiguousarray(np.asarray(first), dtype=np.uint8)
        if first_arr.ndim != 3 or first_arr.shape[-1] not in (3, 4):
            raise ValueError(
                f"encode_avi: frames must be (H, W, 3|4); got shape {first_arr.shape}"
            )
        height, width = int(first_arr.shape[0]), int(first_arr.shape[1])

        def _to_rgba(arr: np.ndarray) -> np.ndarray:
            """Promote (H, W, 3) -> (H, W, 4) opaque; pass (H, W, 4) through."""
            if arr.shape[-1] == 4:
                return arr
            alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
            return np.concatenate([arr, alpha], axis=-1)

        cmd = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(fps)),
            "-i",
            "pipe:",
            # PNG codec inside AVI container preserves per-pixel alpha.
            "-c:v",
            "png",
            "-f",
            "avi",
            out,
        ]

        # See note in :meth:`encode_mp4` re: ``shell=False`` and
        # :func:`_spawn_or_raise`.
        proc = _spawn_or_raise(
            cmd,
            stdin  = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        assert proc.stdin is not None  # for type checkers

        try:
            proc.stdin.write(_to_rgba(first_arr).tobytes())
            for i, f in enumerate(frames_iter, start=1):
                arr = np.ascontiguousarray(np.asarray(f), dtype=np.uint8)
                if arr.shape[:2] != (height, width):
                    raise ValueError(
                        f"encode_avi: frame {i} has shape {arr.shape}, "
                        f"expected ({height}, {width}, 3|4) matching first frame"
                    )
                if arr.shape[-1] not in (3, 4):
                    raise ValueError(
                        f"encode_avi: frame {i} must be 3- or 4-channel; "
                        f"got shape {arr.shape}"
                    )
                proc.stdin.write(_to_rgba(arr).tobytes())
        except BrokenPipeError:
            pass
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = proc.stderr.read()
            finally:
                try:
                    proc.stderr.close()
                except OSError:
                    pass
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}):\n"
                f"{stderr_bytes.decode(errors='replace')}"
            )
        return out

    @classmethod
    def encode_gif(
        cls,
        frames:     Iterable["Frame.FrameLike"],
        output:     str,
        fps:        int                         = 30,
        use_gifski: Optional[bool]              = None,
        overwrite:  bool                        = True,
    ) -> str:
        """Encode a sequence of frames to a high-quality GIF.

        Prefers `gifski <https://gif.ski/>`_ when available (best palette
        quality, NeuQuant + temporal coherence).  Falls back to ffmpeg's
        two-pass ``palettegen=stats_mode=full`` + ``paletteuse=dither=sierra2_4a``
        which is in the 85-90% range of gifski quality on most content.

        Args:
            frames: Iterable of :class:`Frame` or ``(H, W, 3|4) uint8`` arrays.
            output: Output file path.  ``~`` and env vars are expanded.
            fps: Frames per second (default 30).
            use_gifski: If ``None`` (default), use gifski when found on PATH,
                otherwise ffmpeg.  If ``True``, require gifski (raise if
                missing).  If ``False``, force ffmpeg even when gifski is
                available.
            overwrite: If True (default), overwrite existing output.

        Returns:
            The (expanded) absolute output path.

        Raises:
            RuntimeError: If neither encoder is available, or if the chosen
                encoder fails.
            ValueError: If ``frames`` is empty or any frame has an
                unsupported shape / dtype.

        Note:
            GIF requires the palette to be analysed across all frames, so
            this method always materialises the sequence to a temp directory
            of PNGs (cleaned up automatically).  Unlike :meth:`encode_mp4`,
            it cannot be fully streamed.
        """
        out = os.path.expandvars(os.path.expanduser(output))
        if not overwrite and os.path.exists(out):
            raise FileExistsError(
                f"{out} already exists (pass overwrite=True to replace)"
            )

        # Decide which encoder to invoke.  This is the one place we use
        # :func:`shutil.which` -- not to locate the binary for spawning
        # (subprocess does that itself via PATH), but to answer the
        # "do I have gifski available?" question for the auto-detect
        # default without paying the cost of materialising frames just
        # to discover the missing binary.
        if use_gifski is True:
            use_gifski_resolved = True
        elif use_gifski is False:
            use_gifski_resolved = False
        else:
            use_gifski_resolved = shutil.which("gifski") is not None

        # Lazy PIL import (matches the rest of the module's lazy-PIL pattern).
        if Image is None:
            raise RuntimeError("PIL required to write GIF frames; install Pillow.")

        # Materialise frames to PNG -- palette analysis needs them all anyway.
        with tempfile.TemporaryDirectory(prefix="frame_encode_gif_") as tmp:
            n = 0
            for i, f in enumerate(frames):
                arr = np.ascontiguousarray(np.asarray(f), dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
                    raise ValueError(
                        f"encode_gif: frame {i} must be (H, W, 3|4) uint8; "
                        f"got shape {arr.shape}"
                    )
                Image.fromarray(arr).save(os.path.join(tmp, f"frame_{i:06d}.png"))
                n += 1

            if n == 0:
                raise ValueError("encode_gif: no frames provided")

            input_pattern = os.path.join(tmp, "frame_%06d.png")

            if use_gifski_resolved:
                # -- gifski path (best quality) -------------------------
                # gifski takes individual file args in lexicographic order.
                frame_paths = sorted(
                    os.path.join(tmp, f)
                    for f in os.listdir(tmp)
                    if f.startswith("frame_") and f.endswith(".png")
                )
                cmd = [
                    "gifski",
                    "--fps",
                    str(int(fps)),
                    "--quality",
                    "100",
                    "-o",
                    out,
                    *frame_paths,
                ]
                proc = _run_or_raise(cmd, capture_output=True)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"gifski failed (exit {proc.returncode}):\n"
                        f"{proc.stderr.decode(errors='replace')}"
                    )
            else:
                # -- ffmpeg two-pass fallback ---------------------------
                palette = os.path.join(tmp, "_palette.png")

                # Pass 1: build optimised palette across all frames.
                # stats_mode=full (not "diff") because turntable content
                # moves every pixel between frames.
                pass1 = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(int(fps)),
                    "-i",
                    input_pattern,
                    "-vf",
                    "palettegen=stats_mode=full",
                    palette,
                ]
                p1 = _run_or_raise(pass1, capture_output=True)
                if p1.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg palettegen failed (exit {p1.returncode}):\n"
                        f"{p1.stderr.decode(errors='replace')}"
                    )

                # Pass 2: apply palette with high-quality dithering.
                # sierra2_4a is the best general-purpose dither in ffmpeg
                # for smooth-gradient content like rendered turntables.
                pass2 = [
                    "ffmpeg",
                    "-y" if overwrite else "-n",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(int(fps)),
                    "-i",
                    input_pattern,
                    "-i",
                    palette,
                    "-lavfi",
                    "paletteuse=dither=sierra2_4a",
                    out,
                ]
                p2 = _run_or_raise(pass2, capture_output=True)
                if p2.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg paletteuse failed (exit {p2.returncode}):\n"
                        f"{p2.stderr.decode(errors='replace')}"
                    )

        return out


# -- helpers ---------------------------------------------------------------


def _iter_wireframe_primitives(
    source:         Any,
    unique:         bool = True,
    with_cull_data: bool = False,
) -> Iterable[dict]:
    """Yield wireframe primitive dicts from various source types.

    ``source`` may be a :class:`Scene`, :class:`Object`, or :class:`MeshData`.
    Imports of these classes are lazy to avoid a circular dependency
    (``scene.py`` imports :class:`Frame`).

    Each yielded dict carries:

      * ``world_pts``: ``(N, 3)`` float array of vertex world-space positions.
      * ``edge_pairs``: ``(E, 2)`` int array of vertex indices (into
        ``world_pts``) defining one edge each.

    When ``with_cull_data=True``, the dict also carries the data needed
    for back-face culling:

      * ``face_normals``: ``(F, 3)`` world-space face normals.
      * ``face_centres``: ``(F, 3)`` world-space face centroids.
      * ``e2f``: ``(E, max_adj)`` int array; row ``i`` lists face indices
        adjacent to edge ``i``, with ``-1`` sentinels for empty slots.

    When ``unique=True`` (default), ``mesh.e2v`` and ``mesh.e2f`` are used
    (deduplicated unique edges).  Otherwise ``mesh.ue2v``/``mesh.ue2f``
    are used (per-face edges).
    """
    # Lazy imports break the frame -> scene cycle (scene.py already
    # imports Frame for its own encoding methods).
    from cgmath.geometry.mesh import MeshData
    from cgmath.render.scene import Object, Scene

    def _edges(mesh) -> np.ndarray:
        # Counter-intuitive naming in MeshData:
        #   * ``e2v``  -> UNIQUE edges to vertices (what we want for ``unique=True``)
        #   * ``ue2v`` -> raw per-face edges (one entry per face-edge, shared
        #                 edges duplicated -- what we want for ``unique=False``)
        attr = "e2v" if unique else "ue2v"
        try:
            pairs = getattr(mesh, attr)
        except Exception:
            pairs = mesh.e2v  # fallback: every reasonable mesh exposes e2v
        return np.asarray(pairs, dtype=np.int64)

    def _e2f(mesh) -> np.ndarray:
        # Same naming flip as above for edge -> face adjacency.
        attr = "e2f" if unique else "ue2f"
        try:
            adj = getattr(mesh, attr)
        except Exception:
            adj = mesh.e2f
        return np.asarray(adj, dtype=np.int64)

    def _face_centres_world(world_pts: np.ndarray, mesh) -> np.ndarray:
        """Per-face centroids in world space, computed from world_pts.
        Handles mixed-gon meshes via ``mesh.f2v`` with -1 padding."""
        f2v      = np.asarray(mesh.f2v, dtype=np.int64)  # (F, max_verts), -1 padded
        valid    = f2v >= 0
        f2v_safe = np.where(valid, f2v, 0)
        verts    = world_pts[f2v_safe]                   # (F, max_verts, 3)
        verts    = verts * valid[..., None]              # zero-out padded slots
        counts   = valid.sum(axis=1, keepdims=True).astype(np.float64)
        return verts.sum(axis=1) / np.maximum(counts, 1.0)

    def _world_face_normals(mesh, world_rotation_3x3: np.ndarray) -> np.ndarray:
        """Local-space face normals transformed to world space.

        For row vectors and the row-major TransformData convention used by
        ``Object.world_matrix``, the right multiply is
        ``local @ world_matrix[:3, :3]``.  Re-normalise after the transform
        so any non-uniform scale embedded in the matrix doesn't bias the
        sign-of-dot test for back-face culling.
        """
        local = np.asarray(mesh.get_face_normals(), dtype=np.float64)
        with np.errstate(all="ignore"):
            world = local @ world_rotation_3x3
        n = np.linalg.norm(world, axis=1, keepdims=True)
        return world / np.maximum(n, 1e-12)

    def _emit(world_pts: np.ndarray, mesh, world_rotation_3x3: np.ndarray) -> dict:
        prim: dict = {
            "world_pts":  world_pts,
            "edge_pairs": _edges(mesh),
        }
        if with_cull_data:
            try:
                prim["face_normals"] = _world_face_normals(mesh, world_rotation_3x3)
                prim["face_centres"] = _face_centres_world(world_pts, mesh)
                prim["e2f"]          = _e2f(mesh)
            except Exception:
                # If any of the topology accessors blow up, fall back to
                # X-ray for this primitive (skip cull rather than crashing).
                prim["face_normals"] = None
                prim["face_centres"] = None
                prim["e2f"]          = None
        return prim

    if isinstance(source, Scene):
        for obj in source.objects:
            if not obj.visibility or obj.mesh is None:
                continue
            wm = np.asarray(obj.world_matrix, dtype=np.float64)
            yield _emit(obj.world_points, obj.mesh, wm[:3, :3])
        return

    if isinstance(source, Object):
        if source.mesh is None:
            return
        wm = np.asarray(source.world_matrix, dtype=np.float64)
        yield _emit(source.world_points, source.mesh, wm[:3, :3])
        return

    if isinstance(source, MeshData):
        # Treat the mesh's own points as world space (no transform).
        # World rotation is identity in this case.
        yield _emit(
            np.asarray(source.points, dtype=np.float64),
            source,
            np.eye(3, dtype=np.float64),
        )
        return

    raise TypeError(
        f"Frame.wireframe: source must be Scene, Object, or MeshData "
        f"(got {type(source).__name__})"
    )