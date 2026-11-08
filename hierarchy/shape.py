"""
A shape: a record of its own, placed by the transform it names.

A :class:`ShapeData` is what sits under a transform in a scene: a mesh, a
NURBS curve or surface, in time a camera, a light, an SDF primitive. It holds
a ``uuid`` of its own and the ``parent_uuid`` of the :class:`TransformData`
it sits under, as a node holds its parent's, and nothing of the transform
itself: the hierarchy holding the transform places the shape, at the
transform's world matrix, frame by frame when the hierarchy is a clip. A
shape naming no transform, or one the hierarchy does not hold, sits at
identity.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from cgmath.data import Data
from cgmath.hierarchy.hierarchy import _is_uuid, generate_uuid, TransformData


@dataclass(repr=False, eq=False)
class ShapeData(Data):
    """
    The base of every shape: a ``uuid`` of its own, made when none is given,
    and the ``parent_uuid`` of the transform it sits under, None for none.
    Both are keyword-only, so a subclass's own fields come first, required
    ones among them. Neither counts in ``==``: two equal meshes under two
    transforms are equal.
    """

    # compare=False and repr=False for a subclass that lets @dataclass write
    # its own __eq__ and __repr__: the ids count in neither there either
    uuid:        Optional[str] = field(default=None, kw_only=True, compare=False, repr=False)
    parent_uuid: Optional[str] = field(default=None, kw_only=True, compare=False, repr=False)

    EQUALITY_TEST_IGNORE = ["uuid", "parent_uuid"]

    def __post_init__(self):
        super().__post_init__()
        if self.uuid is None:
            self.uuid = generate_uuid()

    def __setattr__(self, name: str, value) -> None:
        if name in ("uuid", "parent_uuid") and value is not None:
            value = str(value)  # a uuid.UUID as its text
            if not _is_uuid(value):
                raise ValueError(f"{type(self).__name__}.{name} takes a uuid or None, not {value!r}")
        super().__setattr__(name, value)

    def _post_load(self) -> None:
        super()._post_load()
        if self.uuid is None:
            self.uuid = generate_uuid()  # a file saved before shapes had one

    def transform(self, hierarchy) -> Optional[TransformData]:
        """
        The transform of ``hierarchy`` this shape sits under: the node whose
        uuid is ``parent_uuid``. None when the shape names none, or the
        hierarchy (a HierarchyData, a ClipData, or None) holds none.
        """
        if self.parent_uuid is None or hierarchy is None or self.parent_uuid not in hierarchy:
            return None
        return hierarchy[self.parent_uuid]

    def world_matrix(self, hierarchy) -> np.ndarray:
        """
        Where the shape sits: the world matrix of its transform in
        ``hierarchy``, at the frame a clip is on, and identity when it has
        none.
        """
        node = self.transform(hierarchy)
        return np.eye(4) if node is None else np.array(node.world_matrix, dtype=np.float64)
