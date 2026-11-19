"""What a mesh edit does to its points, for the records holding a row a point.

A skin and a blend shape target hold their values a point of their mesh, and
no faces. To follow an edit of the mesh they ask the edit itself where each
point went: it is run on a copy of the mesh whose points hold their own
place, and read back off the points of the result.
"""

import numpy as np


def rows_after(mesh, edit) -> np.ndarray:
    """
    The point of ``mesh`` each point of the edited mesh was, (M,) int64.

    ``edit`` takes a mesh and edits it in place, or returns the edited
    mesh (``lambda held: held.from_faces(faces)``); it is run on a copy of
    ``mesh`` whose points hold their own index, so the rows follow exactly
    what the edit does to the points. For edits that keep, drop, copy or
    reorder points: one that blends them (``subdivide``) has no such rows.
    """
    count        = len(mesh.points)
    proxy        = mesh.copy()
    proxy.points = np.repeat(np.arange(count, dtype=np.float64)[:, None], 3, axis=1)
    out          = edit(proxy)
    held         = proxy if out is None else out
    return np.rint(np.asarray(held.points, dtype=np.float64)[:, 0]).astype(np.int64)
