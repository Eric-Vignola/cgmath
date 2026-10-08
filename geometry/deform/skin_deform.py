"""
Skeletal skin deformation operator.

Public class:
    :class:`SkinDeformData` -- bind/apply operator that takes a bind-pose mesh,
        its skin weights and a posed skeleton, and returns the deformed mesh.

The blend math is selected by :class:`DeformMethod` rather than by subclassing,
following :class:`~cgmath.geometry.deform.delta_mush.DeltaMushData` (one class,
three frame-construction methods) and
:class:`~cgmath.geometry.deform.wrap.WrapData` (one class, twenty-one kernels).
LBS, DQS and twist-swing take identical inputs and return identical outputs, so
they are variants of one deformer, not three deformers.

Conventions worth knowing before reading the code:

* **Row-vector.** A point transforms as ``p' = p @ M`` with the translation in
  ``M[3, :3]``, matching the rest of the package.  The skin matrix is therefore
  ``inverse_bind @ world_posed``, in that order -- the reverse produces
  plausible-looking garbage rather than an error.
* **Inverse bind matrices are data, not a derivation.**  glTF stores them
  explicitly because they need not equal ``inv(bind_rig.world_matrix)``: an
  authored matrix can bake a geometry-node transform that no bind pose can
  recover.  Pass the file's matrices when you have them; deriving from a bind
  rig is the fallback, not the default.
* **Fields are plain arrays and strings.**  Meshes, skins and rigs are
  constructor arguments, read once: only the bind-pose points, the matrices,
  the joint names and the compacted weights are kept, and only they survive
  to disk.  Bind before saving -- a deformer saved unbound has no weights,
  and its skin is not saved with it.

The numba-accelerated kernels backing this class live in
:mod:`cgmath.geometry.utils._numba._skin_deform`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from cgmath.geometry._base import Array, Data


if TYPE_CHECKING:
    # NEVER import cgmath.hierarchy at module scope: this module is reachable
    # during cgmath.geometry.__init__, and hierarchy imports back into a
    # partially initialized cgmath.geometry via formats.glb.
    from cgmath.hierarchy import HierarchyData


# ============================= posed rigs ================================= #


class _BoundFrame:
    """The frame a ClipData is bound to, shaped as a one-frame clip.

    It holds exactly what the clip's batched evaluator reads -- the channel
    blocks, the node list and the parent lookup -- with every block cut to
    the bound row, so reading one frame costs one frame rather than the
    whole clip.  The alternative, ``clip.world_matrix``, solves node by node
    and is about ten times slower on a few hundred joints: 9.5 ms against
    1.0 ms for 271.  The clip's own one-span read, ``_span_world``, builds
    the local matrices from their factors and lands within 1e-14 of the
    nodes' world matrices; this goes through the evaluator that mirrors
    ``TransformData.matrix`` factor for factor, so a frame skinned here is
    bit for bit the frame skinned through its nodes.

    A node put back in the clip from elsewhere -- ``clip[i] = rig[i]``, a
    copy -- keeps its own channels rather than a row of the blocks, and the
    clip does not own it: the bound frame is then read from the nodes
    themselves, each read once, as the nodes would answer.
    """

    def __init__(self, clip):
        from cgmath.hierarchy.hierarchy import _PoseFrame

        if all(node._hierarchy is clip for node in clip.list):
            row          = slice(clip.frame, clip.frame + 1)
            self._blocks = {name: block[row] for name, block in clip._blocks.items()}
        else:
            self._blocks = _PoseFrame(clip.list)._blocks
        self.list            = clip.list
        self._parent_indices = clip._parent_indices

    def _realign(self) -> None:
        """the frame is cut from a clip already lined up with its nodes"""


@contextmanager
def _one_thread():
    """Runs numba's parallel kernels on the calling thread alone.

    One frame of a few hundred joints is microseconds of work, less than
    waking every core for each kernel: 1.0 ms against 1.2 to 1.3 ms on a
    quiet machine for 271 joints.  On a machine other processes keep busy,
    each kernel can also wait a whole scheduler slice for its slowest
    thread, hundreds of milliseconds.
    """
    import numba

    count = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        yield
    finally:
        numba.set_num_threads(count)


def _world_matrices(pose) -> tuple:
    """``(names, world)`` for a posed skeleton, one name per world matrix.

    ``world`` is ``(N, 4, 4)`` for one pose -- a rig, or a ClipData at its
    bound frame -- and ``(F, N, 4, 4)`` for a clip's frame axis
    (``clip.frames``, or one scoped to some of its nodes).  A clip is read
    from its blocks in one batched evaluation, never node by node.
    """
    # imported here, not at module scope: see the TYPE_CHECKING note above
    from cgmath.hierarchy.hierarchy import ClipData, _FrameAxis

    if isinstance(pose, _FrameAxis):
        # a scoped axis reads back its own columns, in the order it holds them
        world = pose.world_matrix
        names = [node.name for node in pose._nodes()]
        return names, np.asarray(world, dtype=np.float64)

    if isinstance(pose, ClipData) and pose.frame_count:
        clip = pose.frames._clip  # .frames realigns the blocks with the nodes
        with _one_thread():
            world = _FrameAxis(_BoundFrame(clip)).world_matrix[0]
        return list(clip.name), np.asarray(world, dtype=np.float64)

    return list(pose.name), np.asarray(pose.world_matrix, dtype=np.float64)


# ============================ DeformMethod ================================ #


class DeformMethod(Enum):
    """Blend math used to combine a vertex's joint influences."""

    LBS = "lbs"
    DQS = "dqs"


# =========================== SkinDeformData =============================== #


@dataclass(repr=False, eq=False)
class SkinDeformData(Data):
    """Skeletal skin deformation operator.

    Binds a bind-pose mesh to a set of skin weights and a joint order, then
    deforms that mesh for any posed skeleton sharing those joints.

    The bind step resolves the influence-name to joint-column mapping once and
    compacts the weights to ``(N, K)``, so an animation loop never densifies to
    ``(N, J)`` and never re-resolves names.

    Example::

        from cgmath.geometry import MeshData, SkinData
        from cgmath.geometry.deform import SkinDeformData
        from cgmath.hierarchy import HierarchyData

        mesh = MeshData.load_glb("hero.glb")
        rig = HierarchyData.load_glb("hero.glb")

        deformer = SkinDeformData(mesh, skin, bind_rig=rig)
        posed = deformer.apply(rig, mesh)      # -> a new MeshData

    Every frame of a clip at once, in one kernel call rather than one call
    per frame::

        sequence = deformer.apply(clip.frames)  # (F, N, 3)

        # the same from world matrices already in hand, e.g. a chunk of them
        cols     = deformer.pose_indices(clip)
        world    = clip.frames.world_matrix     # (F, N, 4, 4), one evaluation
        sequence = deformer.apply(world[:, cols])

    Every frame deforms the BIND points: feeding a posed mesh back in would
    deform the previous frame's result and compound.
    """

    rest_points:           Optional[Array(np.float64, "N", 3)] = None
    joints:                List[str] = None
    inverse_bind_matrices: Optional[Array(np.float64, "J", 4, 4)] = None
    influence_indices:     Optional[Array(np.int32, "N", "K")] = None
    influence_weights:     Optional[Array(np.float64, "N", "K")] = None
    method:                str = DeformMethod.LBS.value
    name:                  Optional[str] = None

    # joints are node names: strip_namespace() strips them with the name
    NAMESPACED_FIELDS = ("name", "joints")

    # --- cached --- #
    _skin   = None  # the SkinData bind() compacts; not saved
    _points = None

    def _namespace_parts(self) -> list:
        """the SkinData the deformer was built from: its influences name the same joints"""
        return [self._skin]

    # --------------------------- construction -------------------------------- #

    def __init__(
        self,
        mesh                                       = None,
        skin                                       = None,
        bind_rig                                   = None,
        inverse_bind_matrices: np.ndarray   | None = None,
        joints:                List[str]    | None = None,
        method:                DeformMethod | str  = DeformMethod.LBS,
        name:                  str          | None = None,
    ):
        """Create a skin deformation operator.

        Args:
            mesh: Bind-pose geometry as a :class:`MeshData` (only its
                points are kept), or a raw ``(N, 3)`` array.  :meth:`apply`
                hands back geometry when its *target* is a mesh.
            skin: :class:`SkinData` or :class:`CompactSkinData` holding the
                per-vertex weights and their influence names.
            bind_rig: Bind-pose skeleton, used to derive the inverse bind
                matrices when *inverse_bind_matrices* is not supplied.  A
                :class:`ClipData` binds at the frame it is on.
            inverse_bind_matrices: ``(J, 4, 4)`` authored matrices, ordered to
                *joints*.  Preferred over *bind_rig*: a file's matrices can
                encode a transform the bind pose alone cannot reproduce.
            joints: Joint order for every column below.  Defaults to
                ``skin.influences``.
            method: Blend math, a :class:`DeformMethod` or its value.
            name: Optional name.

        Raises:
            ValueError: If *skin* is missing while *joints* is not given, if
                neither *inverse_bind_matrices* nor *bind_rig* is supplied, or
                if the matrices do not match the joint count.
        """
        self._skin   = skin
        self._points = None

        self.name    = name
        self.method  = DeformMethod(method).value

        self.rest_points = None
        if mesh is not None:
            if hasattr(mesh, "points") and not isinstance(mesh, np.ndarray):
                points = mesh.points
            else:
                points = mesh
            # a copy: the deformer must not move with later edits to the mesh
            self.rest_points = np.array(points, dtype=np.float64, order="C")

        if joints is not None:
            self.joints = [str(x) for x in joints]
        elif skin is not None:
            self.joints = [str(x) for x in skin.influences]
        else:
            raise ValueError("either skin or joints must be given")

        self.inverse_bind_matrices = None
        if inverse_bind_matrices is not None:
            ibm                        = np.asarray(inverse_bind_matrices, dtype=np.float64)
            self.inverse_bind_matrices = np.ascontiguousarray(ibm.reshape(-1, 4, 4))
        elif bind_rig is not None:
            world = self._posed_world(bind_rig)
            if world.ndim != 3:
                raise ValueError(
                    "bind_rig must hold one pose -- a rig, or a ClipData at its "
                    "bind frame, not every frame of a clip"
                )
            self.inverse_bind_matrices = np.ascontiguousarray(np.linalg.inv(world))
        else:
            raise ValueError(
                "one of inverse_bind_matrices or bind_rig is required -- there "
                "is no meaningful default bind pose"
            )

        count = self.inverse_bind_matrices.shape[0]
        if count != len(self.joints):
            raise ValueError(
                f"inverse_bind_matrices has {count} entries but there are "
                f"{len(self.joints)} joints"
            )

        self.influence_indices = None
        self.influence_weights = None

    # ------------------------------ properties ------------------------------- #

    @property
    def method_enum(self) -> DeformMethod:
        """Blend math as a :class:`DeformMethod`.

        ``method`` itself stores the enum's *value*: an Enum dataclass field
        does not survive json, npz or bytes serialization.
        """
        return DeformMethod(self.method)

    @method_enum.setter
    def method_enum(self, value: DeformMethod | str) -> None:
        # no bind invalidation -- every method shares the same bind state
        self.method = DeformMethod(value).value

    @property
    def points(self) -> np.ndarray | None:
        """Most recent deformed positions ``(N, 3)``."""
        return self._points

    @property
    def valid(self) -> bool:
        """True once :meth:`bind` has run.

        Checks only persisted fields, so a copy or a reload is usable without
        re-binding.
        """
        # bind products only. rest_points and the matrices are constructor
        # inputs, and a mesh-less deformer is a supported configuration --
        # including them would leave it permanently unbound, re-gathering on
        # every apply and failing outright once _skin is gone after a copy.
        return self.influence_indices is not None and self.influence_weights is not None

    # ------------------------ private helpers -------------------------------- #

    def _lookup(self, names: List[str], what: str) -> dict:
        """Name to column map, rejecting the ambiguity a bare index would hide."""
        lookup     = {}
        duplicates = set()
        for i, n in enumerate(names):
            n = str(n)
            if n in lookup:
                duplicates.add(n)
            else:
                lookup[n] = i

        clashing = duplicates.intersection(self.joints)
        if clashing:
            raise ValueError(
                f"{what} has duplicate names for {sorted(clashing)[:8]} -- the "
                "joint mapping would be ambiguous"
            )
        return lookup

    def _compact(self):
        """Compact the bound skin to ``(N, K)`` columns into :attr:`joints`."""
        from cgmath.geometry.skin_weights import CompactSkinData
        from cgmath.geometry.utils._numba._skin_weights import (
            count_nonzero_fast,
            gather_top_k_fast,
        )

        skin = self._skin
        if skin is None:
            raise ValueError(
                "no skin to bind -- pass one to the constructor, or use an "
                "object that was already bound before it was copied"
            )

        source = [str(x) for x in skin.influences]

        if isinstance(skin, CompactSkinData):
            k       = int(skin.max_influences)
            indices = np.asarray(skin.influence_indices, dtype=np.int64).reshape(-1, k)
            weights = np.asarray(skin.weights, dtype=np.float64).reshape(-1, k)
        else:
            dense = np.ascontiguousarray(np.asarray(skin.weights, dtype=np.float64))

            # count_nonzero_fast counts |w| > eps but gather_top_k_fast picks
            # the k largest SIGNED values, so a zero column outranks a negative
            # weight and silently evicts it -- wrong geometry, no error.
            # Refuse rather than guess which the author meant.
            if (dense < 0.0).any():
                raise ValueError(
                    "negative skin weights are not supported -- they would be "
                    "silently dropped during compaction; prune or normalize first"
                )

            k = int(max(1, count_nonzero_fast(dense).max()))
            indices, weights = gather_top_k_fast(dense, k)
            indices = np.asarray(indices, dtype=np.int64)
            weights = np.asarray(weights, dtype=np.float64)

        # -1 is this package's padding sentinel (see _scatter_weights_to_dense).
        # numpy would wrap it onto the LAST joint instead of ignoring it.
        padded = indices < 0
        if padded.any():
            indices = np.where(padded, 0, indices)
            weights = np.where(padded, 0.0, weights)

        # Remap the skin's own columns onto self.joints. SkinData.rank() is
        # deliberately not used: it is a bare np.searchsorted with no
        # membership test, so an influence that is absent resolves to some
        # other joint's column instead of raising.
        self._lookup(source, "skin influences")
        joint_lookup = self._lookup(self.joints, "joints")

        missing = [n for n in source if n not in joint_lookup]
        if missing:
            raise ValueError(
                f"{len(missing)} skin influence(s) are absent from joints, "
                f"first few: {missing[:8]}"
            )

        remap = np.array([joint_lookup[n] for n in source], dtype=np.int32)
        return (
            np.ascontiguousarray(remap[indices].astype(np.int32)),
            np.ascontiguousarray(weights),
        )

    def _columns(self, names: List[str]) -> np.ndarray:
        """Positions in *names* of each of :attr:`joints`, in that order."""
        lookup  = self._lookup(names, "rig")

        missing = [j for j in self.joints if j not in lookup]
        if missing:
            raise ValueError(
                f"{len(missing)} joint(s) are absent from the rig, first few: "
                f"{missing[:8]}"
            )

        return np.array([lookup[j] for j in self.joints], dtype=np.int64)

    def _posed_world(self, pose) -> np.ndarray:
        """World matrices of a posed skeleton, ordered to :attr:`joints`.

        ``(J, 4, 4)`` for one pose and ``(F, J, 4, 4)`` for a clip's frame
        axis -- see :func:`_world_matrices`.
        """
        names, world = _world_matrices(pose)
        return world[..., self._columns(names), :, :]

    def _skin_matrices(self, pose) -> np.ndarray:
        """``inverse_bind @ world_posed`` -- row-vector order, do not swap."""
        # never hasattr(pose, ...): that evaluates the property, which on a
        # rig solves every node's world matrix just to answer the question
        if not isinstance(pose, np.ndarray) and (
            hasattr(type(pose), "world_matrix")
            or "world_matrix" in getattr(pose, "__dict__", ())
        ):
            world = self._posed_world(pose)
        else:
            world = np.asarray(pose, dtype=np.float64)

        # rank first: shape[-3] on a bare (4, 4) would be an IndexError, and a
        # single (4, 4) is the likeliest caller slip
        if (
            world.ndim < 3
            or world.shape[-2:] != (4, 4)
            or world.shape[-3] != len(self.joints)
        ):
            raise ValueError(
                f"pose must be (..., {len(self.joints)}, 4, 4) to match the "
                f"joint list, got {world.shape}"
            )

        return self.inverse_bind_matrices @ world

    # ---------------------------- bind / apply ------------------------------- #

    def bind(self) -> None:
        """Resolve the influence to joint mapping and compact the weights.

        Calling this is **optional** -- :meth:`deform` and :meth:`apply`
        auto-bind on the first call.  Use it to pay the cost up front before a
        tight animation loop.
        """
        self.influence_indices, self.influence_weights = self._compact()

        if self.rest_points is not None:
            rows = self.influence_indices.shape[0]
            if rows != self.rest_points.shape[0]:
                raise ValueError(
                    f"skin has {rows} vertices but the bind mesh has "
                    f"{self.rest_points.shape[0]}"
                )
            self._points = self.rest_points.copy()

    def pose_indices(self, rig: "HierarchyData") -> np.ndarray:
        """Columns of *rig* corresponding to :attr:`joints`, in that order.

        Resolve once and reuse across frames -- this is a name lookup, and
        doing it per frame is pure overhead.

        Args:
            rig: Skeleton to index, any object exposing a ``name`` list.

        Returns:
            ``(J,)`` integer columns.

        Raises:
            ValueError: If a joint is missing from *rig*, or *rig* carries
                duplicate names for one of the joints.
        """
        return self._columns(list(rig.name))

    def deform(self, points: np.ndarray, matrices: np.ndarray) -> np.ndarray:
        """Deform a raw point array with a set of skin matrices.

        Args:
            points: ``(N, 3)`` bind-pose positions.
            matrices: ``(J, 4, 4)`` skin matrices, already
                ``inverse_bind @ world_posed`` and ordered to :attr:`joints`.

        Returns:
            ``(N, 3)`` deformed positions.  The input is not modified.

        Raises:
            ValueError: If the point count does not match the bound weights.
            NotImplementedError: If :attr:`method` is not yet implemented.
        """
        return self._deform(points, matrices, frames=False)

    def _deform(self, points: np.ndarray, matrices: np.ndarray, frames: bool):
        """:meth:`deform` for one pose, or with *frames* for ``(F, J, 4, 4)``
        skin matrices, every frame in one kernel call: ``(F, N, 3)``."""
        from cgmath.geometry.utils._numba._skin_deform import (
            dqs_compact_fast,
            lbs_compact_fast,
        )

        if self.influence_indices is None or self.influence_weights is None:
            self.bind()

        points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        rows   = self.influence_indices.shape[0]
        if points.shape[0] != rows:
            raise ValueError(
                f"points has {points.shape[0]} vertices but the bound weights "
                f"have {rows}"
            )

        # The kernel indexes matrices[influence_indices[i, c]] and numba does
        # not bounds-check, so a short stack is an out-of-bounds read: silent
        # garbage geometry, or a SIGBUS on a large overrun. apply() validates
        # this in _skin_matrices; deform() is public and must too. The index
        # check also covers a restored object, which never re-runs bind().
        matrices = np.asarray(matrices, dtype=np.float64)
        rank     = 4 if frames else 3
        if matrices.ndim != rank or matrices.shape[-2:] != (4, 4):
            shape = ("F, " if frames else "") + f"{len(self.joints)}, 4, 4"
            raise ValueError(f"matrices must be ({shape}), got {matrices.shape}")
        if matrices.shape[-3] <= int(self.influence_indices.max()):
            raise ValueError(
                f"matrices has {matrices.shape[-3]} entries but the bound weights "
                f"reference joint {int(self.influence_indices.max())}"
            )

        if self.method == DeformMethod.DQS.value:
            return dqs_compact_fast(
                points, self.influence_indices, self.influence_weights, matrices
            )

        if self.method != DeformMethod.LBS.value:
            raise NotImplementedError(f"{self.method!r} is not implemented yet")

        return lbs_compact_fast(
            points, self.influence_indices, self.influence_weights, matrices
        )

    def apply(self, pose, target=None):
        """Deform the bind pose for a posed skeleton.

        Auto-binds on first call.

        Args:
            pose: A posed skeleton (:class:`HierarchyData`, or a
                :class:`ClipData` at the frame it is on), every frame of a
                clip (``clip.frames``), or raw world matrices shaped
                ``(J, 4, 4)`` for one frame or ``(F, J, 4, 4)`` for a
                sequence.  Raw matrices must already be ordered to
                :attr:`joints` -- see :meth:`pose_indices`.  A sequence is
                deformed in one call, every frame at once.
            target: What to deform and what to hand back.  A :class:`MeshData`
                returns a copy of it with deformed points; a raw ``(N, 3)``
                array, or None (meaning the bind points), returns an array.

        Returns:
            A copy of *target* with deformed points when *target* is mesh-like,
            otherwise ``(N, 3)`` -- or ``(F, N, 3)`` for a batched pose.

        Raises:
            ValueError: If a batched pose is combined with a mesh target, which
                would have to return F meshes.
        """
        if not self.valid:
            self.bind()

        is_mesh = target is not None and (
            hasattr(target, "points") and not isinstance(target, np.ndarray)
        )
        if target is None:
            points = self.rest_points
        else:
            points = target.points if is_mesh else target

        matrices = self._skin_matrices(pose)

        if matrices.ndim == 4:
            if is_mesh:
                raise ValueError(
                    "a batched pose returns F point arrays, which cannot be "
                    "packed into a single MeshData -- pass target=None and "
                    "build the meshes yourself"
                )
            stacked      = self._deform(points, matrices, frames=True)
            self._points = stacked[-1] if len(stacked) else None
            return stacked

        self._points = self.deform(points, matrices)

        if is_mesh:
            result        = target.copy()
            result.points = self._points.copy()
            # copy() drops the underscore caches (the BVH included) by
            # rebuilding from to_dict(), but `normals` is a real field and
            # would survive as the bind-pose normals.
            if getattr(result, "normals", None) is not None:
                result.normals = None
            return result

        return self._points